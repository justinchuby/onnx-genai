//! The sequential CPU executor (Track D, `docs/ORT2.md` §20, §11.3).
//!
//! Turns a loaded [`Graph`] plus its live [`WeightStore`] into a runnable plan:
//! resolve every value's concrete shape from the actual bound inputs at
//! `run`, size a device buffer per value from those *resolved* shapes, resolve
//! a kernel per node through the execution provider (keyed by the resolved
//! input shapes), then walk the topological order binding
//! [`TensorView`]/[`TensorMut`] windows over those buffers and invoking each
//! kernel. It is generic over any [`Graph`] and any [`ExecutionProvider`]; the
//! Phase-1 build wires in the CPU EP only, but nothing here is op- or
//! model-specific.
//!
//! ## Symbolic → concrete shape resolution (§3.2, §11)
//!
//! Real models carry *symbolic* input dims (e.g. `batch`, `max_seq_len`): the
//! loader produces a [`Shape`] whose dims are a mix of [`Dim::Static`] and
//! [`Dim::Symbolic`]. This executor is model-agnostic about them — a symbol is
//! whatever [`SymbolId`] the graph interned. At `run` it reads the actual shape
//! of each bound input, **binds** the graph's symbols to concrete sizes from
//! those inputs (conflicts across inputs are an error), and **substitutes**
//! those bindings into every value's loader shape to obtain a fully-concrete
//! shape. Buffers are sized from the resolved shapes and become run-scoped when
//! shapes are dynamic (reused when the resolved shape is unchanged, re-allocated
//! when it changes). A fully-static graph is simply the special case where
//! there are no symbols: resolution is a no-op and every buffer/kernel is
//! materialized once at build.
//!
//! The session does **not** infer op output shapes — that is the loader's job
//! (the loader runs `onnx-runtime-shape-inference` at load time). If a value's
//! loader shape still contains an unbound symbol after substitution, the
//! session resolves genuinely data-dependent extents just-in-time during
//! execution (see [`dynamic_output_shapes`]); anything it still cannot size is
//! reported as [`SessionError::UnresolvedShape`] naming the value and its
//! producing op, rather than guessing.
//!
//! ## Holden's precondition (ep-api safety review #1) — enforced here
//!
//! A [`TensorView`] carries no backing length, so it cannot self-check storage
//! bounds. This executor owns every buffer, so it is the layer that *can*: for
//! **every** input and output view of **every** node it calls
//! [`strided::view_in_bounds`] (or, for sub-byte dtypes, the `storage_bytes`
//! equivalent in [`view_bounds`]) against the **run-scoped resolved** buffer and
//! refuses to dispatch on failure. That check is the sole thing that makes
//! ep-cpu's unchecked pointer derefs sound.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use onnx_runtime_ep_api::{
    CaptureRegionShapeStatus, DeviceBuffer, DevicePtr, DevicePtrMut, EpError, ExecutionProvider,
    ExternalMmapRegion, Kernel, KernelInput, KernelMatch, LazyWeight, LazyWeightBoundary,
    ResidentWeight, StructuralCaptureDecline, TensorBacking, TensorMut, TensorView, WeightHandle,
};

type OptionalTensorSpecs = Vec<Option<(DataType, Vec<usize>)>>;
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ep_cpu::strided::view_in_bounds;
use onnx_runtime_ir::Attribute;
use onnx_runtime_ir::{
    DataType, DeviceType, Dim, Graph, Node, NodeId, Shape, SymbolId, TensorLayout, ValueId,
    WeightRef, as_static_shape, broadcast_shapes, compute_contiguous_strides,
};
use onnx_runtime_loader::WeightStore;
use onnx_runtime_optimizer::InitializerResolver;
use onnx_runtime_shape_inference::{
    DimExpr, InferenceRegistry, MAX_SHAPE_DATA_ELEMS, MergePolicy, NodeIo, ShapeData,
    SymbolInterner, TypeInfo,
};
use onnx_runtime_tracer::{Args, SpanGuard, TraceContext, annotate_current_span_with};

use crate::SessionOutput;
use crate::error::{Result, SessionError};
use crate::sequence::{
    ConcatPlan, SeqTensor, SequenceError, SequenceValue, SplitSpec, split_tensor, stack_new_axis,
};
use crate::tensor::{DeviceIoBinding, SharedTensorBuffer, Tensor};

fn profile_ops_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("ONNX_GENAI_PROFILE_OPS")
            .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
    })
}

/// Low-overhead, env-gated (`NXRT_EXEC_PHASE_PROFILE=1`) phase profiler for the
/// executor's control-flow / subgraph-dispatch machinery. When disabled every
/// entry point is a single relaxed-atomic load and an early return, so the
/// production decode hot path pays no measurable cost. When enabled it
/// accumulates wall-clock nanoseconds and a call count per named phase, which
/// [`executor_phase_stats`] exposes. This exists to attribute the
/// per-decode-step control-flow overhead (`exec_if` / `run_subgraph` / child
/// setup) that the op-level profiler folds into the single `If` bucket.
mod phase_profile {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
    use std::sync::{Mutex, OnceLock};
    use std::time::Instant;

    static STATE: AtomicU8 = AtomicU8::new(0); // 0 = unknown, 1 = off, 2 = on

    pub fn enabled() -> bool {
        match STATE.load(Ordering::Relaxed) {
            1 => false,
            2 => true,
            _ => {
                let on = std::env::var("NXRT_EXEC_PHASE_PROFILE")
                    .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
                STATE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
                on
            }
        }
    }

    /// Test-only override of the env-derived enable state.
    #[cfg(test)]
    pub(super) fn force_enabled(on: bool) {
        STATE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
    }

    /// Test-only snapshot of a phase's accumulated `(total_ns, count)`.
    #[cfg(test)]
    pub(super) fn snapshot(phase: &'static str) -> Option<(u128, u64)> {
        registry()
            .lock()
            .ok()
            .and_then(|reg| reg.get(phase).map(|s| (s.total_ns, s.count)))
    }

    #[derive(Default, Clone, Copy)]
    struct PhaseStat {
        total_ns: u128,
        count: u64,
    }

    fn registry() -> &'static Mutex<BTreeMap<&'static str, PhaseStat>> {
        static REGISTRY: OnceLock<Mutex<BTreeMap<&'static str, PhaseStat>>> = OnceLock::new();
        REGISTRY.get_or_init(|| Mutex::new(BTreeMap::new()))
    }

    /// Accumulate `nanos` against `phase`. No-op unless the profiler is enabled.
    pub fn record(phase: &'static str, nanos: u128) {
        if !enabled() {
            return;
        }
        if let Ok(mut reg) = registry().lock() {
            let entry = reg.entry(phase).or_default();
            entry.total_ns += nanos;
            entry.count += 1;
        }
    }

    /// Every recorded phase, most expensive first.
    ///
    /// Exposed as data rather than rendered here: this crate has no dependency
    /// on the process-wide stage profiler that everything else reports through,
    /// and reversing that would point the native runtime at the ONNX Runtime
    /// crate. The caller that already depends on both does the merging.
    pub fn all_stats() -> Vec<(&'static str, u128, u64)> {
        let Ok(reg) = registry().lock() else {
            return Vec::new();
        };
        let mut rows = reg
            .iter()
            .map(|(phase, stat)| (*phase, stat.total_ns, stat.count))
            .collect::<Vec<_>>();
        rows.sort_by_key(|row| std::cmp::Reverse(row.1));
        rows
    }

    /// Scoped timer that records its lifetime to `phase` on drop.
    pub struct PhaseSpan {
        phase: &'static str,
        start: Option<Instant>,
    }

    impl PhaseSpan {
        pub fn new(phase: &'static str) -> Self {
            let active = enabled();
            Self {
                phase,
                // Avoid the clock read entirely on the disabled hot path.
                start: if active { Some(Instant::now()) } else { None },
            }
        }
    }

    impl Drop for PhaseSpan {
        fn drop(&mut self) {
            if let Some(start) = self.start {
                record(self.phase, start.elapsed().as_nanos());
            }
        }
    }

    /// Render and reset the accumulated per-phase table to stderr. Called once at
    /// process exit (or on demand) so a single line-oriented dump is available.
    pub fn report_to_stderr() {
        if !enabled() {
            return;
        }
        let rows: Vec<(&'static str, PhaseStat)> = match registry().lock() {
            Ok(reg) => reg.iter().map(|(n, s)| (*n, *s)).collect(),
            Err(_) => return,
        };
        static PRINTED: AtomicBool = AtomicBool::new(false);
        if PRINTED.swap(true, Ordering::Relaxed) {
            return;
        }
        let mut rows = rows;
        rows.sort_by_key(|r| std::cmp::Reverse(r.1.total_ns));
        eprintln!("[nxrt-phase] phase,total_ms,calls,us/call");
        for (name, stat) in &rows {
            if name.ends_with("_bytes") {
                continue;
            }
            let total_ms = stat.total_ns as f64 / 1_000_000.0;
            let us_per_call = if stat.count > 0 {
                (stat.total_ns as f64 / 1_000.0) / stat.count as f64
            } else {
                0.0
            };
            eprintln!(
                "[nxrt-phase] {name},{total_ms:.3},{},{us_per_call:.2}",
                stat.count
            );
        }
        // Byte-valued counters (host traffic) are reported separately in MB.
        for (name, stat) in &rows {
            if !name.ends_with("_bytes") {
                continue;
            }
            let total_mb = stat.total_ns as f64 / (1024.0 * 1024.0);
            let mb_per_call = if stat.count > 0 {
                total_mb / stat.count as f64
            } else {
                0.0
            };
            eprintln!(
                "[nxrt-phase] {name},total_mb={total_mb:.1},calls={},mb/call={mb_per_call:.3}",
                stat.count
            );
        }
    }
}

/// Open an env-gated executor phase-profiling span (see [`phase_profile`]).
macro_rules! phase_span {
    ($phase:expr) => {
        phase_profile::PhaseSpan::new($phase)
    };
}

fn trace_span(name: &'static str, cat: &'static str) -> Option<SpanGuard> {
    onnx_runtime_tracer::global_context()
        .filter(|trace| trace.is_enabled())
        .map(|trace| trace.span(name, cat))
}

/// Public re-export so the bench/profile harness can dump the phase table.
/// Accumulated executor phase costs as `(phase, total_ns, calls)`, most
/// expensive first.
///
/// Empty unless `NXRT_EXEC_PHASE_PROFILE` is set. Kept separate from the
/// process-wide stage profiler because this crate cannot depend on it without
/// pointing the native runtime at the ONNX Runtime crate; the caller that
/// depends on both merges the two.
pub fn exec_phase_stats() -> Vec<(&'static str, u128, u64)> {
    phase_profile::all_stats()
}

pub fn print_exec_phase_profile() {
    phase_profile::report_to_stderr();
}

fn host_dtype_alignment(dtype: DataType) -> usize {
    match dtype {
        DataType::Float16 | DataType::BFloat16 | DataType::Int16 | DataType::Uint16 => 2,
        DataType::Float32 | DataType::Int32 | DataType::Uint32 | DataType::Complex64 => 4,
        DataType::Float64 | DataType::Int64 | DataType::Uint64 | DataType::Complex128 => 8,
        _ => 1,
    }
}

fn print_op_profile(total: Duration, timings: HashMap<String, (Duration, usize)>) {
    let mut timings = timings.into_iter().collect::<Vec<_>>();
    timings.sort_unstable_by_key(|entry| std::cmp::Reverse(entry.1.0));
    let total_ms = total.as_secs_f64() * 1_000.0;
    eprintln!("[onnx-genai-profile] node execution: {total_ms:.3} ms");
    eprintln!("[onnx-genai-profile] op_type,total_ms,percent,calls");
    for (op_type, (elapsed, calls)) in timings {
        let elapsed_ms = elapsed.as_secs_f64() * 1_000.0;
        let percent = if total_ms == 0.0 {
            0.0
        } else {
            elapsed_ms / total_ms * 100.0
        };
        eprintln!("[onnx-genai-profile] {op_type},{elapsed_ms:.3},{percent:.2},{calls}");
    }
}

/// Print, to stderr, how the capture pass split a claimed subgraph into captured
/// device-graph segments and eager seam nodes, and why each seam exists. Gated
/// by `ONNX_GENAI_LOG_CAPTURE_SEGMENTS` for transparency into segmentation.
fn log_capture_segmentation(schedule: &CaptureSchedule) {
    let captured = schedule.captured_segments();
    let seams = schedule.segments.len() - captured;
    eprintln!(
        "[onnx-genai-capture] segmented CUDA graph: {captured} captured segment(s), \
         {seams} eager seam(s)"
    );
    for boundary in &schedule.boundaries {
        match boundary.node_id {
            Some(id) => {
                let seam_label = boundary
                    .seam_reason
                    .map(SeamReason::label)
                    .unwrap_or("unclassified-seam");
                eprintln!(
                    "[onnx-genai-capture]   seam node {id} ({}::{}) [{seam_label}] ran eagerly: {}",
                    boundary.domain, boundary.op_type, boundary.reason
                );
            }
            None => eprintln!(
                "[onnx-genai-capture]   seam ({}): {}",
                boundary.op_type, boundary.reason
            ),
        }
    }
}

/// A per-node compiled entry: the structural facts the run loop needs without
/// re-deriving them from the graph. Shapes are **not** baked here — they are
/// resolved per run from the bound inputs (see module docs).
#[derive(Debug)]
pub(crate) struct NodePlan {
    pub node_id: NodeId,
    /// Positional input value ids in ONNX signature order. An omitted optional
    /// input (ONNX empty-string input name → `None` slot) is preserved as
    /// `None` so a later present input is never misread as the omitted one
    /// (e.g. `Slice(data, starts, ends, "", steps)`). Trailing `None`s are
    /// trimmed — a truly absent trailing optional simply lowers the arity.
    pub inputs: Vec<Option<ValueId>>,
    /// Output value ids, in positional order.
    pub outputs: Vec<ValueId>,
    /// Element types of the inputs, positional (matches `inputs`). An omitted
    /// optional (`None`) slot carries [`DataType::Undefined`] so EP claim-time
    /// validation can distinguish it from a supplied tensor.
    pub input_dtypes: Vec<DataType>,
    /// Element types of the outputs.
    pub output_dtypes: Vec<DataType>,
}

/// Cache key for a compiled kernel (§11.1). Keyed by the concrete node and its
/// **resolved** (concrete) input shapes: attributes are fixed per node, so this
/// is correct, and the shape component makes it *shape-keyed* — a re-run with
/// the same resolved shapes hits, a different shape (e.g. a new batch/seq)
/// misses and re-compiles. This preserves Chew's guarantee: a kernel is never
/// reused for a shape it was not compiled for.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct KernelKey {
    node: u32,
    shapes: Vec<Vec<usize>>,
}

/// Observable kernel-cache statistics (§11.1) — enough to prove reuse in tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Distinct compiled entries currently held.
    pub entries: usize,
    /// Lookups served from an existing entry.
    pub hits: u64,
    /// Lookups that compiled a new kernel.
    pub misses: u64,
}

/// Observable control-flow executor statistics. These counters make subgraph
/// reuse deterministic to test without relying on timing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ControlFlowStats {
    /// Child executors built, including shape-signature rebuilds.
    pub subgraph_builds: u64,
    /// Child subgraph invocations served by those executors.
    pub subgraph_runs: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceAllocationCounts {
    pub allocations: u64,
    pub frees: u64,
}

/// Structural execution path used by a node during a captured run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CapturePathKind {
    /// Recorded into a device graph and replayed.
    CaptureRegion,
    /// Dispatched eagerly while remaining on the device.
    EagerDeviceSeam,
    /// Host-driven work or a host round-trip between captured regions.
    HostSeam,
}

impl CapturePathKind {
    /// Stable short label used by capture diagnostics.
    pub const fn label(self) -> &'static str {
        match self {
            Self::CaptureRegion => "capture-region",
            Self::EagerDeviceSeam => "eager-device-seam",
            Self::HostSeam => "host-seam",
        }
    }
}

/// Structural reason a node forms an eager seam during device-graph capture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeamReason {
    /// Host-driven control-flow or sequence semantics.
    HostControlFlowOrSequence,
    /// A data-dependent output shape was unresolved before capture.
    UnresolvedOutputShape,
    /// A data-dependent input shape was unresolved before capture.
    UnresolvedInputShape,
    /// The requested concrete kernel shape has not completed warmup.
    KernelNotWarmed,
    /// The selected device kernel explicitly opts out of capture.
    KernelCaptureUnsupported,
    /// The kernel aborted device-graph *recording* (e.g. it advertised capture
    /// support but issued a stream synchronize, which CUDA rejects mid-capture)
    /// and was quarantined to a forced eager seam so the rest of the graph can
    /// still be captured.
    CaptureRecordingFailed,
}

impl SeamReason {
    /// Execution path implied by this structural seam cause.
    pub const fn path_kind(self) -> CapturePathKind {
        match self {
            Self::HostControlFlowOrSequence => CapturePathKind::HostSeam,
            Self::UnresolvedOutputShape
            | Self::UnresolvedInputShape
            | Self::KernelNotWarmed
            | Self::CaptureRecordingFailed
            | Self::KernelCaptureUnsupported => CapturePathKind::EagerDeviceSeam,
        }
    }

    /// Stable short path-kind label used by capture diagnostics.
    pub const fn label(self) -> &'static str {
        self.path_kind().label()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// One actionable reason a device-graph capture attempt was rejected.
pub struct CaptureDecline {
    /// Graph node id, or `None` for graph/capture-lifecycle requirements.
    pub node_id: Option<u32>,
    /// ONNX operator type, or `"<graph>"` for graph-level requirements.
    pub op_type: String,
    /// Canonical ONNX domain (`"ai.onnx"` by default), or `"nxrt"` graph-level.
    pub domain: String,
    /// Failed precondition and, where applicable, how to reach the capture path.
    pub reason: String,
    /// Structural seam classification, or `None` for graph-level hard preconditions.
    pub seam_reason: Option<SeamReason>,
}

impl CaptureDecline {
    fn node(
        node_id: NodeId,
        node: &Node,
        seam_reason: SeamReason,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            node_id: Some(node_id.0),
            op_type: node.op_type.clone(),
            domain: canonical_domain(node),
            reason: reason.into(),
            seam_reason: Some(seam_reason),
        }
    }

    fn graph(reason: impl Into<String>) -> Self {
        Self {
            node_id: None,
            op_type: "<graph>".to_string(),
            domain: "nxrt".to_string(),
            reason: reason.into(),
            seam_reason: None,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
/// Structured reasons a device graph could not be captured.
pub struct CaptureDeclineReport {
    /// All graph- and node-level declines found by the pre-capture audit.
    pub entries: Vec<CaptureDecline>,
}

impl CaptureDeclineReport {
    fn one(decline: CaptureDecline) -> Self {
        Self {
            entries: vec![decline],
        }
    }

    /// Whether the capture audit found no declines.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// One node-level reason the requested execution provider declined placement.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionProviderDecline {
    /// Stable graph/subgraph node identity used in diagnostics.
    pub node: String,
    /// Canonical ONNX domain (`"ai.onnx"` for the default domain).
    pub domain: String,
    /// ONNX operator type.
    pub op_type: String,
    /// Actionable reason returned by [`ExecutionProvider::supports_op`].
    pub reason: String,
}

/// Structured report for an accelerator request that executes on CPU.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecutionProviderFallbackReport {
    /// Requested provider name, such as `"cuda_ep"`.
    pub requested_provider: String,
    /// Provider that will execute the graph.
    pub fallback_provider: String,
    /// Number of executable graph/subgraph nodes assigned to the fallback EP.
    pub assigned_node_count: usize,
    /// Sorted distinct `domain::op` classes assigned to the fallback EP.
    pub assigned_ops: Vec<String>,
    /// Nodes the requested provider did not claim, with colocated reasons.
    pub declines: Vec<ExecutionProviderDecline>,
}

impl std::fmt::Display for ExecutionProviderFallbackReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} nodes assigned to CPU (ops: {}) — GPU EP {} did not claim {} node(s): {}. \
             Heterogeneous CUDA+CPU placement is unavailable, so the whole session uses {}",
            self.assigned_node_count,
            self.assigned_ops.join(", "),
            self.requested_provider,
            self.declines.len(),
            format_cuda_coverage_issues(&self.declines),
            self.fallback_provider,
        )
    }
}

impl std::fmt::Display for CaptureDeclineReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CUDA graph capture rejected")?;
        for (index, decline) in self.entries.iter().enumerate() {
            if index == 0 {
                write!(f, ": ")?;
            } else {
                write!(f, "; ")?;
            }
            match decline.node_id {
                Some(node_id) => write!(
                    f,
                    "node {node_id} ({}::{}) — {}",
                    decline.domain, decline.op_type, decline.reason
                )?,
                None => write!(f, "{} — {}", decline.op_type, decline.reason)?,
            }
        }
        Ok(())
    }
}

pub enum DeviceGraphCaptureResult {
    Captured(Vec<Option<Tensor>>),
    NotCapturable(CaptureDeclineReport),
}

enum ScopedRunResult {
    Executed(Vec<Option<SessionOutput>>),
    NotCapturable(CaptureDeclineReport),
}

fn kernel_capture_decline(
    node_id: NodeId,
    node: &Node,
    kernel: &dyn Kernel,
) -> Option<CaptureDecline> {
    kernel.capture_support().reason().map(|reason| {
        CaptureDecline::node(node_id, node, SeamReason::KernelCaptureUnsupported, reason)
    })
}

fn structural_capture_decline(
    node_id: NodeId,
    node: &Node,
    decline: StructuralCaptureDecline,
) -> CaptureDecline {
    let seam_reason = match decline {
        StructuralCaptureDecline::HostControlFlowOrSequence => {
            SeamReason::HostControlFlowOrSequence
        }
        StructuralCaptureDecline::UnresolvedOutputShape => SeamReason::UnresolvedOutputShape,
        StructuralCaptureDecline::UnresolvedInputShape => SeamReason::UnresolvedInputShape,
    };
    CaptureDecline::node(node_id, node, seam_reason, decline.reason())
}

/// Whether verbose segmented-capture diagnostics are printed to stderr.
///
/// Gated identically to op profiling so a run can surface exactly where the
/// CUDA EP split a claimed subgraph into captured segments and eager seam nodes.
fn capture_segmentation_logging_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("ONNX_GENAI_LOG_CAPTURE_SEGMENTS")
            .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
    })
}

/// How a scoped run drives the device-graph lifecycle.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RunMode {
    /// No capture: execute every node eagerly on the stream.
    Eager,
    /// First capture pass: partition the plan into segments, record each
    /// capturable segment into its own device graph, and run the non-capturable
    /// seam nodes eagerly in between.
    Capture,
    /// Subsequent steps: replay each captured segment graph in order, re-running
    /// only the eager seam nodes.
    Replay,
}

/// The device-graph capture disposition of a single op, used to annotate its
/// trace span with **why** it was or was not captured. Carries a borrowed
/// reason string rather than an owned one so an untraced run never allocates.
#[derive(Clone, Copy)]
enum OpCaptureTrace<'a> {
    /// Plain eager run — no capture attempt is in progress for this op.
    Eager,
    /// The op was recorded into a captured device-graph segment.
    Captured,
    /// The op runs eagerly as a capture seam; `reason` explains why it could
    /// not be recorded into a device graph (which kernel/predicate declined).
    Rejected(&'a str),
}

/// Trace-arg key: whether an op was captured into a device graph.
const ARG_CAPTURE_STATUS: &str = "capture_status";
/// Trace-arg key: why an op was not captured into a device graph.
const ARG_CAPTURE_REASON: &str = "capture_reason";

impl OpCaptureTrace<'_> {
    /// Annotate the active op-span with this capture disposition. A no-op for
    /// [`OpCaptureTrace::Eager`] (nothing was being captured) and when no span
    /// is active.
    fn annotate(self) {
        match self {
            OpCaptureTrace::Eager => {}
            OpCaptureTrace::Captured => {
                annotate_current_span_with(|| {
                    onnx_runtime_tracer::Args::new().with(ARG_CAPTURE_STATUS, "captured")
                });
            }
            OpCaptureTrace::Rejected(reason) => {
                annotate_current_span_with(|| {
                    onnx_runtime_tracer::Args::new()
                        .with(ARG_CAPTURE_STATUS, "rejected")
                        .with(ARG_CAPTURE_REASON, reason)
                });
            }
        }
    }
}

/// Scope guard that guarantees an in-progress segment capture is always ended
/// before its enclosing function returns.
///
/// During [`RunMode::Capture`], nodes are recorded between
/// `begin_device_graph_capture` and `end_device_graph_capture`. If a node fails
/// mid-record, the `?` early return would otherwise skip the end call and leave
/// the CUDA stream wedged in capture mode — the caller's
/// `reset_device_graph()` is then a no-op (reset is rejected while capturing),
/// so every later eager/replay launch fails with `STREAM_CAPTURE_INVALIDATED`.
///
/// While armed, [`Drop`] aborts the capture (ending stream capture and
/// discarding the half-recorded graph). The success path calls [`disarm`] and
/// then ends the capture normally via `end_device_graph_capture`.
///
/// [`disarm`]: SegmentCaptureGuard::disarm
struct SegmentCaptureGuard<'a> {
    ep: &'a dyn ExecutionProvider,
    armed: bool,
}

impl<'a> SegmentCaptureGuard<'a> {
    fn arm(ep: &'a dyn ExecutionProvider) -> Self {
        Self { ep, armed: true }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for SegmentCaptureGuard<'_> {
    fn drop(&mut self) {
        if self.armed {
            // Best-effort: the abort itself may fail, but the caller is already
            // unwinding a capture failure and will reset the lifecycle next.
            let _ = self.ep.abort_device_graph_capture();
        }
    }
}

/// One contiguous run of plan nodes that either share a captured device graph or
/// all execute eagerly (a non-capturable seam).
#[derive(Clone, Debug, PartialEq, Eq)]
struct ScheduledSegment {
    /// First plan index (inclusive).
    start: usize,
    /// One past the last plan index (exclusive).
    end: usize,
    /// `true` when `[start, end)` is captured into a device graph; `false` for an
    /// eager seam of non-capturable (but still device-placed or CPU) nodes.
    captured: bool,
    /// Capture-order index of this segment's graph in the EP, set only when
    /// `captured`.
    graph_index: usize,
}

/// The plan's partition into captured segments and eager seams, plus the
/// structured reason each segment boundary exists (which node forced the split).
///
/// Recorded once during the capture pass and reused for every subsequent replay
/// so the interleaving of graph replays and eager seam execution is stable.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct CaptureSchedule {
    segments: Vec<ScheduledSegment>,
    /// One entry per non-capturable seam node, explaining why it forced a
    /// boundary (its `CaptureSupport::Unsupported` reason or structural cause).
    boundaries: Vec<CaptureDecline>,
}

impl CaptureSchedule {
    /// Number of captured device-graph segments (1 for a whole-subgraph capture).
    fn captured_segments(&self) -> usize {
        self.segments.iter().filter(|seg| seg.captured).count()
    }

    /// Whether the whole plan captured as a single graph (no eager seams).
    fn is_single_graph(&self) -> bool {
        self.segments.len() == 1 && self.segments[0].captured
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DeviceBindingSignature {
    input_name: String,
    binds_input: bool,
    output_name: Option<String>,
    dtype: DataType,
    physical_shape: Vec<usize>,
    device_ptr: usize,
}

/// Shape-keyed kernel cache (§11.1). Owns the compiled kernels for the session.
#[derive(Default)]
pub(crate) struct KernelCache {
    entries: HashMap<KernelKey, Box<dyn onnx_runtime_ep_api::Kernel>>,
    hits: u64,
    misses: u64,
}

impl KernelCache {
    fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.entries.len(),
            hits: self.hits,
            misses: self.misses,
        }
    }

    /// Return the cached kernel for `(node, resolved_input_shapes)`, verifying
    /// EP support and compiling+inserting it on a miss. The EP support check
    /// lives on the miss path so a re-planned shape is re-validated exactly
    /// once per distinct shape.
    fn get_or_create(
        &mut self,
        node_id: NodeId,
        node: &Node,
        input_shapes: &[Vec<usize>],
        input_dtypes: &[DataType],
        constant_inputs: &[bool],
        opset: u64,
        ep: &dyn ExecutionProvider,
    ) -> Result<&dyn onnx_runtime_ep_api::Kernel> {
        let key = KernelKey {
            node: node_id.0,
            shapes: input_shapes.to_vec(),
        };
        if self.entries.contains_key(&key) {
            self.hits += 1;
        } else {
            // Verify the EP claims this op at these concrete shapes/layouts
            // before compiling — same gate the static path used at build.
            let shape_dims: Vec<Shape> = input_shapes
                .iter()
                .map(|s| s.iter().map(|&d| Dim::Static(d)).collect())
                .collect();
            let layouts = vec![TensorLayout::contiguous(); input_shapes.len()];
            if let KernelMatch::Unsupported { reason } =
                ep.supports_op(node, opset, &shape_dims, input_dtypes, &layouts)
            {
                return Err(SessionError::unsupported_op(
                    node,
                    node_id,
                    opset,
                    ep.name(),
                    reason,
                ));
            }
            let mut kernel = match ep.get_kernel(node, input_shapes, opset) {
                Ok(kernel) => kernel,
                Err(EpError::NoEpForOp {
                    domain,
                    op_type,
                    opset,
                }) => {
                    // Opset-aware claims should make this unreachable. Preserve
                    // the actionable diagnostic if an EP's claim drifts.
                    return Err(SessionError::unsupported_op(
                        node,
                        node_id,
                        opset,
                        ep.name(),
                        format!(
                            "no handler for {domain}::{op_type} at opset {opset} — add a claim+handler"
                        ),
                    ));
                }
                Err(error) => return Err(error.into()),
            };
            kernel.set_constant_inputs(constant_inputs);
            self.entries.insert(key.clone(), kernel);
            self.misses += 1;
        }
        Ok(self.entries.get(&key).expect("just inserted").as_ref())
    }
}

/// The compiled, runnable graph: buffers + plan + kernel cache. Owned by the
/// public [`InferenceSession`](crate::InferenceSession).
pub(crate) struct Executor {
    graph: Graph,
    /// Kept alive so external-weight memory maps outlive buffer population —
    /// **and**, since the weight-streaming change, so borrowed initializer
    /// buffers that alias this store's mmap bytes stay valid for the executor's
    /// whole lifetime. `weights` MUST outlive every live use of `buffers`: a
    /// borrowed `DeviceBuffer` in `buffers` points into `weights`' mmap/inline
    /// storage. Teardown is safe because `Executor::drop` **drains and
    /// deallocates `buffers` first** (a borrowed deallocate is a no-op free), so
    /// no buffer still aliases `weights` when the `Arc<WeightStore>` field is
    /// dropped afterwards — no use-after-free regardless of field drop order.
    weights: Arc<WeightStore>,
    ep: Arc<dyn ExecutionProvider>,
    /// Lazy external initializers available only at the nxrt fused-MoE boundary.
    /// Stock EPs ignore this map and keep receiving the resident buffers below.
    weight_handles: HashMap<ValueId, WeightHandle>,
    /// One device buffer per backed value. Static values are allocated once at
    /// build; dynamic (symbol-shaped) values are allocated per run and cached
    /// here so a run whose resolved shape is unchanged reuses the allocation.
    buffers: HashMap<ValueId, DeviceBuffer>,
    /// The concrete shape each live buffer in [`Self::buffers`] is currently
    /// sized for — the reuse key for run-scoped buffers.
    buffer_shapes: HashMap<ValueId, Vec<usize>>,
    /// Loader-produced (possibly symbolic) shape of every value.
    value_shapes: HashMap<ValueId, Shape>,
    /// Element type of every value.
    value_dtypes: HashMap<ValueId, DataType>,
    /// Topologically ordered execution plan (structure only; shapes per run).
    plan: Vec<NodePlan>,
    /// name → value id for the graph inputs the caller must supply.
    input_index: HashMap<String, ValueId>,
    /// Value ids the caller must supply at `run` (graph inputs minus initializers).
    required_inputs: Vec<ValueId>,
    /// Whether any value in the graph carries a symbolic dim. A fully-static
    /// graph is materialized eagerly at build; a symbolic graph defers buffer
    /// allocation and kernel compilation to the first `run` that fixes shapes.
    has_symbols: bool,
    cache: KernelCache,
    /// name → value id for every named value in this graph (inputs, outputs,
    /// initializers and interior SSA values). Used to resolve outer-scope
    /// captures referenced by name from a nested control-flow subgraph body.
    name_index: HashMap<String, ValueId>,
    /// Reusable child executors for this graph's control-flow subgraph bodies,
    /// keyed by `(control-flow node, subgraph attr key)`. Built lazily on first
    /// execution (once concrete input shapes are known) and **reused across
    /// Loop/Scan iterations** — the whole point of the efficiency directive: a
    /// body's topo-sort, buffer sizing and kernel compilation happen once, then
    /// every iteration is just a re-bind + dispatch. Rebuilt only if a later
    /// invocation's external input shapes differ from the ones it was compiled
    /// for (a shape-varying loop body — rare).
    subgraph_execs: HashMap<(NodeId, String), ChildExecutor>,
    control_flow_stats: ControlFlowStats,
    /// Per-`If` memo of the last observed branch predicate. During steady decode
    /// a loop-invariant `If` (e.g. the LongRoPE cos/sin cache selector) keeps the
    /// same predicate every step, so its branch outputs are already resident in
    /// their persistent buffers. The predicate is still read each step (the
    /// correctness guard); only the redundant branch re-execution — here two
    /// `Constant` materializations plus their host→device cache copies — is
    /// skipped. A predicate flip re-runs the branch (and, on an output-shape
    /// change, retires the captured graph via the existing seam invalidation).
    if_last_predicate: HashMap<NodeId, bool>,
    device_graph_signature: Option<Vec<DeviceBindingSignature>>,
    /// The captured-segment schedule from the most recent successful capture,
    /// reused to interleave segment replays with eager seam nodes on each
    /// subsequent step. `None` when no device graph is installed.
    capture_schedule: Option<CaptureSchedule>,
    /// Structured segment-boundary reasons from the most recent capture, retained
    /// for diagnostics after `capture_schedule` is taken for replay.
    capture_segmentation: Vec<CaptureDecline>,
    /// Output value ids of every control-flow (`If`/`Loop`/`Scan`) node. ONNX
    /// shape inference cannot statically resolve a control-flow output whose
    /// branches declare different shapes (e.g. LongRoPE's `If` selecting a short
    /// vs. long RoPE cos/sin cache), so such outputs stay symbolic and any
    /// downstream capturable kernel that reads one would form a per-consumer
    /// eager seam. Within a decode generation the selected branch is stable, so
    /// [`Self::seed_control_flow_capture_shapes`] seeds each output's concrete
    /// shape from the prior run for capture planning, folding those consumers
    /// back into captured segments. Computed once at build.
    control_flow_output_values: HashSet<ValueId>,
    /// Concrete control-flow output shapes the most recent capture assumed (a
    /// snapshot of the seeded shapes from [`Self::control_flow_output_values`]).
    /// On replay the control-flow seam re-executes eagerly; if it now produces a
    /// different shape (a branch flip, e.g. LongRoPE short↔long at the context
    /// threshold) the installed graph's baked device pointers are stale, so the
    /// step falls back to eager and the graph is retired for re-capture.
    capture_cf_shapes: HashMap<ValueId, Vec<usize>>,
    /// Persistent-binding signature the most recent eager warmup ran under (see
    /// [`ExternalBindings::capture_signature`]). Capture-mode shape seeding only
    /// trusts the warm just-in-time shapes recorded in [`Self::buffer_shapes`]
    /// when a later step presents this exact signature, so any changed pointer
    /// or capacity withholds the seed instead of baking a stale shape.
    capture_warm_signature: Option<Vec<ExternalCaptureSig>>,
    /// Every value's concrete just-in-time shape as resolved by the most recent
    /// eager warmup. The data-dependent decode shapes we seed for capture are
    /// JIT-sized on the compute path (which populates `buffers` but not
    /// [`Self::buffer_shapes`]), so the authoritative warm geometry is snapshotted
    /// from the eager run's fully-resolved shape map, not the buffer bookkeeping.
    capture_warm_shapes: HashMap<ValueId, Vec<usize>>,
    /// The warm decode shapes actually seeded into the most recent capture. After
    /// the capture pass re-resolves each node's true shape, a divergence here
    /// means the warm seed was stale for this step, so the graph is retired and
    /// the caller re-warms/re-captures rather than replaying a stale shape.
    capture_warm_seeded: HashMap<ValueId, Vec<usize>>,
    /// `(domain, op_type)` pairs whose kernel aborted device-graph *recording*
    /// during a capture pass (e.g. it declared `CaptureSupport::Supported` but
    /// issued a stream synchronize, which CUDA rejects mid-capture). Warm-decode
    /// shape seeding can admit such a node once its output shape is known; if the
    /// resulting capture fails, the offending op-type is quarantined here and
    /// [`Self::node_capture_reason`] then forces every node of that op-type to a
    /// forced eager seam, so the capture is re-planned and the remaining
    /// genuinely-capturable ops still fold. Grows monotonically within an
    /// executor: a kernel that breaks recording once breaks it every time.
    capture_quarantine_ops: HashSet<(String, String)>,
    /// Node whose kernel returned an error while recording a captured segment,
    /// set transiently by [`Self::run_plan_segmented`] so the capture retry loop
    /// can quarantine its op-type. `None` outside a failed capture pass.
    last_capture_failed_node: Option<NodeId>,
    /// Run-scoped zero-copy **view** metadata (§5.4). A value id present here is
    /// a strided view aliasing another value's buffer (a layout/movement-op
    /// output such as `Slice`) rather than an owner in [`Self::buffers`]. Built
    /// during the run loop and cleared at the start of every run.
    views: HashMap<ValueId, ValueView>,
    /// Run-scoped set of buffer-owning value ids that have ≥1 live view alias.
    /// A pinned buffer must not be reused or deallocated for the remainder of
    /// the run (conservative liveness: a source buffer outlives every view that
    /// aliases it, guaranteeing no use-after-free). Cleared each run.
    pinned: HashSet<ValueId>,
    /// Value ids whose runtime value is a **sequence of tensors** rather than a
    /// single tensor (produced by `SequenceEmpty`/`SequenceConstruct`/
    /// `SequenceInsert`/`SequenceErase`/`SplitToSequence`). Computed once at
    /// build; these values own no [`DeviceBuffer`] and are skipped by buffer
    /// sizing — their storage lives in [`Self::sequences`] at run time.
    sequence_values: HashSet<ValueId>,
    /// Allocation owners promoted into ref-counted storage when a tensor enters
    /// an ONNX Sequence. `buffers` retains a non-owning dispatch alias, while
    /// sequence elements clone the owner Arc. At the next run boundary, after
    /// all sequence handles are cleared, the unique owner is restored to
    /// `buffers` before any input/output can be mutated.
    shared_buffers: HashMap<ValueId, Arc<SharedTensorBuffer>>,
    /// Run-scoped storage for sequence values: `value id → SequenceValue`. A
    /// [`SequenceValue`] holds its elements as `Arc`-shared immutable tensors,
    /// so a sequence op that inserts/erases/etc. shares element `Arc`s with the
    /// source rather than deep-copying bytes (see [`crate::sequence`] for the
    /// no-copy + no-race invariants). Cleared each run.
    sequences: HashMap<ValueId, SequenceValue>,
    /// Run-scoped **zero-copy** backing for a *tensor* value whose bytes are a
    /// shared sequence element (the output of `SequenceAt`): the tensor aliases
    /// the element's `Arc` instead of owning a `DeviceBuffer`, so no bytes are
    /// copied out of the sequence. A downstream kernel reads it through a
    /// [`TensorView`] over the `Arc`'s bytes; it is materialized to owned bytes
    /// only at the graph-output/control-flow boundary. Cleared each run.
    seq_elem_values: HashMap<ValueId, SeqTensor>,
    execution_provider_fallback_report: Option<ExecutionProviderFallbackReport>,
    /// Shared runtime trace context. Defaults to a disabled [`TraceContext::noop`]
    /// so an untraced run pays only a single relaxed atomic load per op when
    /// deciding whether to open a span. When enabled, the executor opens one
    /// span per executed op so kernels can attach kernel-variant and
    /// capture-rejection reasons via [`annotate_current_span_with`].
    trace: TraceContext,
    /// Reusable scratch for the resolved input shapes of the node currently
    /// being dispatched by [`Self::exec_kernel_node`]. Refilled (truncate +
    /// refill, retaining inner `Vec` capacity) once per node via
    /// [`Self::refill_input_shapes`], so a steady-state decode step performs no
    /// per-node `Vec<Vec<usize>>` allocation for shape lookup. Reuse invariant:
    /// it is fully rewritten at the top of each `exec_kernel_node` call and only
    /// read within that same call — never aliased or carried across nodes.
    scratch_input_shapes: Vec<Vec<usize>>,
    /// F5 Stage 1 — master switch for the steady-state decode-plan memo. Default
    /// ON; disabled by `ONNX_GENAI_DECODE_MEMO=0`. Consulted on the top-level CPU
    /// eager decode path — including the normal persistent-KV-binding case.
    decode_memo_enabled: bool,
    /// When set (`ONNX_GENAI_DECODE_MEMO_VERIFY=1`, or always under
    /// `debug_assertions`), every memo replay is asserted equal to a fresh
    /// `resolve_soft` — the R1 verifiable safety net. Off in release by default.
    decode_memo_verify: bool,
    /// The active decode-plan memo, primed after two consecutive plan-matching
    /// eager steps and rebuilt on any signature change.
    decode_memo: Option<DecodePlanMemo>,
    /// Bindings of the previous memo-eligible eager step, diffed against the
    /// current step to derive the varying-symbol set (R1 two-real-step rule).
    decode_memo_prev_bindings: Option<HashMap<SymbolId, usize>>,
    /// Diagnostic: what the memo did on the most recent memo-eligible eager
    /// step. Exposed to the guard tests.
    decode_memo_last_action: DecodeMemoAction,
    /// F5 Stage 1 — persistent working shape map reused across decode steps.
    /// On a replay step it is taken in place (no allocation): its previous
    /// just-in-time entries are stripped, the length-invariant partition is left
    /// untouched (byte-identical by construction), and only the variant tail is
    /// re-substituted into its existing `Vec`s. The run loop then refills the
    /// small data-dependent tail. This is what makes replay genuinely
    /// allocation-amortized (Stage 1's whole purpose) rather than a per-token
    /// `HashMap`/`Vec` rebuild.
    decode_memo_resolved: HashMap<ValueId, Vec<usize>>,
    /// Diagnostic counters (proof the memo actually fires on the real path, so a
    /// gate that silently excludes it is never shipped again). Incremented per
    /// memo-eligible eager step; a summary is emitted on drop when
    /// `ONNX_GENAI_DECODE_MEMO_STATS=1`.
    decode_memo_primed_count: u64,
    decode_memo_rebuilt_count: u64,
    decode_memo_replayed_count: u64,
    /// Steps that routed through the memo path but were structurally ineligible
    /// (memo OFF, CUDA, nested, or non-eager) — counted only when the master
    /// switch is on, to make an over-restrictive gate observable.
    decode_memo_ineligible_count: u64,
    /// F5 Stage 2 — cached invariant buffer/view plan. Present only after a
    /// successful memo rebuild that found ≥1 fully-invariant pure-view node; it
    /// records the zero-copy view aliases to reinstate and the pure-view plan
    /// nodes to elide on a matching replay, guarded by a per-source buffer
    /// identity signature. Invalidated on every non-replay step (mirrors the
    /// Stage-1 Chew defense-in-depth) so a stale plan from a retired/errored
    /// step can never serve a future replay. Default ON (shares the Stage-1
    /// `ONNX_GENAI_DECODE_MEMO` gate; set =0 to disable).
    decode_view_plan: Option<DecodeViewPlan>,
    /// F5 Stage 2 counters. `views_reused` = zero-copy view aliases reinstated
    /// without rebuild; `dispatch_elided` = pure-view plan nodes whose re-dispatch
    /// was skipped. Both prove non-vacuous firing on the real decode path.
    decode_views_reused_count: u64,
    decode_dispatch_elided_count: u64,
    /// F5 Stage 2 defense-in-depth: consecutive replay steps whose buffer-identity
    /// signature failed to match (a source buffer moved/resized under a plan that
    /// classified it invariant). After [`STAGE2_SIG_MISMATCH_LIMIT`] such steps the
    /// view plan is disabled for the rest of the session — an invariant-buffer
    /// assumption that keeps breaking must never keep serving cached views.
    decode_view_plan_sig_mismatch_streak: u32,
    /// Latched off after repeated signature mismatches (see above).
    decode_view_plan_disabled: bool,
}

/// After this many consecutive buffer-identity signature mismatches, F5 Stage 2
/// view reuse is latched off for the session (Chew defense-in-depth).
const STAGE2_SIG_MISMATCH_LIMIT: u32 = 2;

/// Run-scoped metadata for a zero-copy view value: it owns no buffer but
/// borrows `source`'s buffer with the given (real, possibly non-contiguous or
/// negative-strided) geometry. `strides`/`byte_offset` are expressed relative
/// to `source`'s allocation base, so a view-of-a-view is flattened to a single
/// hop whose `source` is always a real buffer owner (never itself a view).
#[derive(Clone, Debug)]
struct ValueView {
    source: ValueId,
    shape: Vec<usize>,
    strides: Vec<i64>,
    byte_offset: usize,
}

/// F5 Stage 1 — steady-state decode-plan memo.
///
/// [`Executor::resolve_soft`] is a **pure function of the current symbol
/// `bindings`** (see [`substitute`]): a value's resolved shape depends only on
/// its interned [`Shape`] and the bindings, and [`Executor::bind_symbols`]
/// derives bindings purely from the input *shapes*. During steady-state
/// single-token (M=1) decode only a small set of length symbols changes each
/// step, so every value whose shape references no such symbol resolves to a
/// byte-identical shape every step. This memo caches that length-invariant
/// partition and, on a plan-matching step, re-substitutes only the small
/// length-varying tail — avoiding a full ~600-entry map rebuild per token.
///
/// **Soundness (why a wrong shape can never be replayed).** A step may replay
/// the invariant partition iff every symbol the memo did *not* classify as
/// varying has the same binding it was built under and the bound-symbol set is
/// identical ([`DecodePlanMemo::matches`]). Because each invariant shape
/// references only static dims and non-varying symbols, an unchanged
/// non-varying binding set guarantees it re-substitutes to the identical value —
/// so the replayed map is byte-identical to a fresh `resolve_soft`. Crucially,
/// if a symbol that *actually* varies were mis-classified invariant, its next
/// change is a change to a **non-varying** binding ⇒ `matches` fails ⇒ the memo
/// rebuilds; a stale shape is therefore never emitted, regardless of how
/// `decode_varying` was derived. The variant tail is always re-substituted from
/// the fresh bindings, never replayed. A debug/opt-in full re-resolve
/// ([`Executor::decode_memo_verify`]) asserts equality every replay (R1 net).
struct DecodePlanMemo {
    /// Bindings the invariant partition was built under — the replay guard.
    reference_bindings: HashMap<SymbolId, usize>,
    /// Symbols observed to change value between two consecutive real eager
    /// steps (R1: derived by diffing, never guessed).
    decode_varying: HashSet<SymbolId>,
    /// Resolved shape of every value whose [`Shape`] references no varying
    /// symbol — replayed verbatim.
    invariant_shapes: HashMap<ValueId, Vec<usize>>,
    /// Values whose [`Shape`] references ≥1 varying symbol — re-substituted from
    /// the fresh bindings on every replay step.
    variant_values: Vec<ValueId>,
    /// All value ids the memo owns (`invariant_shapes` keys ∪ `variant_values`) —
    /// i.e. exactly the keys `resolve_soft` produces for this regime. Used to
    /// strip the previous step's just-in-time (data-dependent) entries from the
    /// persistent working map before replay, so the run loop recomputes them.
    canonical: HashSet<ValueId>,
    /// L-abstracted structural fingerprint of the persistent device-I/O binding
    /// set the memo was built under. Pure-L KV growth leaves this unchanged (so
    /// the step replays); a binding appearing/disappearing, a role flip, or a
    /// dtype change alters it and forces a rebuild. See [`DecodeBindingSig`].
    reference_external_sig: Vec<DecodeBindingSig>,
}

/// L-abstracted structural fingerprint of one persistent (device-bound) I/O
/// binding, for the decode-plan memo replay guard.
///
/// Unlike [`ExternalCaptureSig`] — which is pointer/capacity- and concrete-shape-
/// exact for CUDA capture seeding — this abstracts the growing length symbol `L`
/// to its symbolic identity by fingerprinting the binding's *declared* (symbolic)
/// shape (`value_shapes[vid]`, which is graph-static). Two decode steps that
/// differ only by KV length therefore compare **equal** and replay, while a
/// structural change (a binding added/removed, an input/output role flip, or a
/// dtype change) compares unequal and forces a rebuild. Pointer and byte capacity
/// are deliberately **excluded**: Stage 1 memoizes shape resolution only (buffers
/// are re-sized every step outside the memo), so a KV-cache realloc must not
/// invalidate the plan — including `ptr`/`len` here would force a rebuild on every
/// growth-driven reallocation and leave the memo perpetually dead on the real
/// decode path.
#[derive(Clone, PartialEq, Eq)]
struct DecodeBindingSig {
    vid: ValueId,
    is_input: bool,
    dtype: DataType,
    decl_shape: Shape,
}

impl DecodePlanMemo {
    /// A step is a plan-match (may replay) iff it binds exactly the same symbol
    /// set as the reference and agrees with it on every non-varying symbol (only
    /// the varying length / past-length symbols may differ) **and** presents the
    /// same L-abstracted persistent-binding signature.
    fn matches(
        &self,
        bindings: &HashMap<SymbolId, usize>,
        external_sig: &[DecodeBindingSig],
    ) -> bool {
        if external_sig != self.reference_external_sig {
            return false;
        }
        if bindings.len() != self.reference_bindings.len() {
            return false;
        }
        bindings.iter().all(|(sym, &val)| {
            match self.reference_bindings.get(sym) {
                Some(&ref_val) => val == ref_val || self.decode_varying.contains(sym),
                // A symbol the reference did not bind: the plan shape differs.
                None => false,
            }
        })
    }
}

/// F5 Stage 2 — cached invariant buffer/view plan.
///
/// Stage 1 proved that during steady single-token decode a large partition of
/// values resolve to a byte-identical shape every step (the memo's
/// `invariant_shapes`). Empirically, on real decoders the ~113 pure layout ops
/// (`Reshape`/`Squeeze`/`Unsqueeze`/no-op views) produce a **byte-identical
/// zero-copy [`ValueView`] every step** — yet Stage 1 still re-cleared and
/// re-dispatched every one per token. This plan caches those view aliases and the
/// nodes that produce them so a matching replay step can:
///   1. reinstate the invariant view aliases instead of clearing+rebuilding them,
///   2. exclude the invariant partition from per-step buffer sizing, and
///   3. elide re-dispatch of the pure-view nodes entirely.
///
/// **Membership (why an elided view is never geometrically stale).** A node is a
/// *candidate* iff every output's shape is in the memo's proven-invariant partition
/// (`invariant_shapes`) — so Stage 1 already guarantees the output shape is
/// byte-identical every replay step and the replayed `resolved` map always carries
/// it. A candidate is *promoted* to the active elision set only after its produced
/// view is observed **byte-identical across a second real decode step**
/// ([`Executor::validate_decode_view_plan`]) — the same two-real-step confirmation
/// Stage 1 uses to derive its varying set. Contiguous-view strides are a pure
/// function of the (invariant) output shape, and any per-step `byte_offset` drift
/// (e.g. a position-indexed slice into a fixed-capacity KV buffer) would differ
/// across the two observed steps and so is rejected before it can ever be elided.
///
/// **Soundness — the buffer-identity obligation.** Stage 1 could exclude
/// pointer/capacity from its replay key because it cached *shapes only* and every
/// kernel re-read fresh bytes each step. Stage 2 caches actual **buffers/views**, so
/// a realloc or pointer move of a cached view's source would leave the reinstated
/// alias pointing at a stale/dangling region. Therefore this plan records
/// `source_buffer_sig` = `(source, base_ptr, capacity)` for every buffer a retained
/// view aliases, and a replay step reinstates the plan **iff** every signature still
/// matches ([`Executor::stage2_buffer_sig_matches`]); any mismatch forces a full
/// rebuild. (A retained [`ValueView`] references its source by [`ValueId`], so a
/// consumer already re-reads the *current* base pointer — but the byte offset and
/// capacity assumptions are exactly what the pointer+capacity guard protects.)
///
/// The plan is only ever *built* at the successful end of a memo Rebuilt step,
/// *validated* on the following replay, and *used* on a memo Replayed step whose
/// bindings, external signature (Stage 1) and buffer identity (Stage 2) all match;
/// it is invalidated on every non-replay step so an errored/retired step can never
/// serve a stale alias. Under `decode_memo_verify` every reinstated view is also
/// asserted equal to a freshly built one in-flight (the R1 safety net).
struct DecodeViewPlan {
    /// Plan-node indices (into [`Executor::plan`]) whose every output shape is in
    /// the memo's invariant partition — candidates until validated, then the active
    /// elision set (re-dispatch skipped on a matching replay).
    elided_nodes: HashSet<usize>,
    /// The zero-copy view aliases to reinstate each replay step (`vid` → its
    /// invariant [`ValueView`]), verbatim from the reference step.
    retained_views: Vec<(ValueId, ValueView)>,
    /// Distinct buffer-owning source value ids to re-pin (conservative liveness:
    /// a source with a live view is never reused/freed within the run).
    pinned_sources: Vec<ValueId>,
    /// Buffer-identity signature `(source, base_ptr as usize, capacity)` for every
    /// retained view's source buffer. The Stage-2 replay guard.
    source_buffer_sig: Vec<(ValueId, usize, usize)>,
    /// `false` for a freshly built candidate; set `true` once every retained view
    /// has been confirmed byte-identical on a second real decode step. Only a
    /// validated plan is ever used to elide dispatch.
    validated: bool,
}

/// Outcome of the most recent memo-eligible eager resolve, exposed for the F5
/// guard tests to distinguish a rebuild from a replay.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecodeMemoAction {
    /// The memo was disabled, or the step was not memo-eligible.
    Disabled,
    /// First observation of a regime: bindings recorded, no memo built yet
    /// (the two-real-step derivation needs a second matching step).
    Primed,
    /// A full resolve whose result (re)built the memo by diffing this step with
    /// the previous eligible step.
    Rebuilt,
    /// The invariant partition was replayed and only the variant tail
    /// re-substituted.
    Replayed,
}

/// True iff two binding maps bind exactly the same symbol set.
fn same_symbol_keys(a: &HashMap<SymbolId, usize>, b: &HashMap<SymbolId, usize>) -> bool {
    a.len() == b.len() && a.keys().all(|k| b.contains_key(k))
}

/// M==1 single-token-decode gate (residual #3): admit a memo (re)build only for
/// a steady autoregressive-decode transition, where sequence/KV length symbols
/// only ever *grow*. `prev`→`cur` qualifies iff both bind the same symbol set,
/// at least one symbol increased, and **no** symbol decreased. This excludes the
/// prefill→decode transition (the query-length symbol drops from the prompt
/// length P to 1) and any non-decode reshape, so the memo activates only on
/// single-token decode — not prefill — tightening the blast radius. Soundness
/// does not rely on this gate (the `matches` guard is the correctness invariant);
/// it only decides *when* the memo is worth building.
fn is_decode_growth_transition(
    prev: &HashMap<SymbolId, usize>,
    cur: &HashMap<SymbolId, usize>,
) -> bool {
    if !same_symbol_keys(prev, cur) {
        return false;
    }
    let mut any_grew = false;
    for (sym, &c) in cur {
        let p = prev[sym];
        if c > p {
            any_grew = true;
        } else if c < p {
            return false; // a shrinking extent is not steady decode (e.g. prefill→decode)
        }
    }
    any_grew
}

/// True iff `shape` references any symbol in `symbols`.
fn shape_references_any(shape: &Shape, symbols: &HashSet<SymbolId>) -> bool {
    shape
        .iter()
        .any(|d| matches!(d, Dim::Symbolic(s) if symbols.contains(s)))
}

/// Whether the decode-plan memo master switch (`ONNX_GENAI_DECODE_MEMO`) is on.
/// Default ON; set `ONNX_GENAI_DECODE_MEMO=0` to disable.
///
/// The explicit OFF values are `0`, `false`, and `off` (case-insensitive,
/// surrounding whitespace trimmed). Every other state — unset, empty, or an
/// unrecognized value — enables the memo, so parsing fails safe toward the
/// validated fast path (worst case: rebuild every step, no speedup, never
/// wrong). Ripley's authoritative A/B recorded 0 token flips and a non-negative
/// speedup at every tested core count, so default-ON is token-exact by
/// construction.
fn decode_memo_env_enabled() -> bool {
    match std::env::var("ONNX_GENAI_DECODE_MEMO") {
        Ok(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off"
        ),
        Err(_) => true,
    }
}

/// Whether the opt-in per-step replay verification (`ONNX_GENAI_DECODE_MEMO_VERIFY`)
/// is set. Always on under `debug_assertions`.
fn decode_memo_verify_env_enabled() -> bool {
    matches!(
        std::env::var("ONNX_GENAI_DECODE_MEMO_VERIFY")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("on")
    )
}

/// Per-input geometry the run loop resolves once per node: the raw base pointer
/// of the backing (root) buffer plus the real view (shape, element strides —
/// possibly non-contiguous or negative — and byte offset) to read it through.
/// A plain owned value yields contiguous strides at offset 0; a view value
/// yields its recorded strides/offset over its source buffer. `present` is false
/// for an omitted optional input (an absent placeholder).
struct InInfo {
    present: bool,
    dtype: DataType,
    shape: Vec<usize>,
    strides: Vec<i64>,
    byte_offset: usize,
    base_ptr: *const std::ffi::c_void,
    device: onnx_runtime_ir::DeviceId,
    backing: TensorBacking,
    /// Length in bytes of the backing (root) allocation, for the bounds gate.
    root_len: usize,
}

#[derive(Clone)]
struct ExternalValue {
    dtype: DataType,
    shape: Vec<usize>,
    accepts_subshape: bool,
    ptr: *mut std::ffi::c_void,
    len: usize,
    alignment: usize,
    device: onnx_runtime_ir::DeviceId,
}

impl ExternalValue {
    fn accepts_output(&self, dtype: DataType, shape: &[usize], bytes: usize) -> bool {
        self.dtype == dtype
            && self.len >= bytes
            && if self.accepts_subshape {
                shape.len() == self.shape.len()
                    && shape
                        .iter()
                        .zip(&self.shape)
                        .all(|(&required, &capacity)| required <= capacity)
            } else {
                self.shape == shape
            }
    }

    fn writable_buffer(&self) -> Result<DeviceBuffer> {
        // SAFETY: `prepare_external_bindings` obtains this pointer from a live
        // `DeviceIoBinding` exclusively borrowed for the run. The binding owns
        // the allocation, outlives this alias, and is not otherwise accessed
        // until execution returns.
        unsafe {
            DeviceBuffer::from_borrowed_mut_parts(self.ptr, self.device, self.len, self.alignment)
        }
        .ok_or_else(|| SessionError::Internal("external output binding has a null pointer".into()))
    }
}

#[derive(Default)]
struct ExternalBindings {
    inputs: HashMap<ValueId, ExternalValue>,
    outputs: HashMap<ValueId, ExternalValue>,
}

/// One persistent (device-bound) I/O binding's identity for capture: its value,
/// role, dtype, kernel-visible shape, backing device pointer and byte capacity.
/// The full set forms the *decode binding signature* under which a warm eager
/// resolution's just-in-time shapes are trusted for capture-mode seeding: a
/// change to any pointer or shape means the warm geometry may be stale, so the
/// seed is withheld (nodes stay eager) rather than baked into a captured graph.
#[derive(Clone, PartialEq, Eq)]
struct ExternalCaptureSig {
    vid: ValueId,
    is_input: bool,
    dtype: DataType,
    shape: Vec<usize>,
    ptr: usize,
    len: usize,
}

impl ExternalBindings {
    fn seed_capture_shapes(&self, resolved: &mut HashMap<ValueId, Vec<usize>>) {
        for (&vid, value) in self.inputs.iter().chain(&self.outputs) {
            resolved.entry(vid).or_insert_with(|| value.shape.clone());
        }
    }

    /// Order-independent signature of every persistent binding (pointer, byte
    /// capacity and kernel-visible shape). Two runs whose signatures compare
    /// equal present pointer- and capacity-stable buffers, which is the exact
    /// precondition for trusting a prior eager run's just-in-time shapes.
    fn capture_signature(&self) -> Vec<ExternalCaptureSig> {
        let mut sig: Vec<ExternalCaptureSig> = self
            .inputs
            .iter()
            .map(|(&vid, v)| (vid, true, v))
            .chain(self.outputs.iter().map(|(&vid, v)| (vid, false, v)))
            .map(|(vid, is_input, v)| ExternalCaptureSig {
                vid,
                is_input,
                dtype: v.dtype,
                shape: v.shape.clone(),
                ptr: v.ptr as usize,
                len: v.len,
            })
            .collect();
        sig.sort_by_key(|a| (a.vid.0, a.is_input));
        sig
    }
}

/// Concrete child plan cached for one external-input dtype/shape signature.
struct CompiledChildPlan {
    exec: Executor,
    signature: Vec<ChildInputSignature>,
}

/// Control-flow bodies commonly alternate among a handful of stable shapes.
/// Four entries cover those cases without retaining an unbounded set of plans.
const CHILD_EXECUTOR_CACHE_CAPACITY: usize = 4;

#[derive(Clone, Debug, PartialEq, Eq)]
struct ChildInputSignature {
    dtype: DataType,
    shape: Vec<usize>,
}

/// A reusable executor for one nested graph body.
///
/// The body signature and lexical-capture set are resolved once at construction.
/// Concrete [`Executor`]s are compiled lazily and retained in a small,
/// deterministic LRU keyed by external-input dtype/shapes, so alternating
/// Loop/Scan/If signatures reuse prior plans instead of recompiling each switch.
pub(crate) struct ChildExecutor {
    name: String,
    template: Graph,
    inherited_opsets: HashMap<String, u64>,
    weights: Arc<WeightStore>,
    ep: Arc<dyn ExecutionProvider>,
    formal_names: Vec<String>,
    capture_names: Vec<String>,
    input_names: Vec<String>,
    compiled: Vec<CompiledChildPlan>,
    builds: u64,
    runs: u64,
    /// Shared trace context, propagated to every compiled child plan's executor.
    trace: TraceContext,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ChildExecutorStats {
    pub builds: u64,
    pub runs: u64,
}

/// Invocation-invariant binding metadata for one selected subgraph. Loop/Scan
/// prepare this once outside the iteration loop, including one-time capture
/// materialization, then only rebind the changing formal tensors each step.
struct PreparedSubgraph {
    key: (NodeId, String),
    /// Direct captures plus transitive captures needed only by nested bodies.
    captures: HashMap<String, Tensor>,
}

/// The `[shape, strides, byte_offset]` storage-bounds gate (Holden's
/// precondition). Uses [`view_in_bounds`] for fixed-width dtypes and a
/// `storage_bytes` check for sub-byte packed dtypes (which have no integral
/// per-element byte size).
fn view_bounds(
    shape: &[usize],
    strides: &[i64],
    byte_offset: usize,
    dtype: DataType,
    buffer_len: usize,
) -> Result<()> {
    let esize = dtype.byte_size();
    if esize == 0 {
        // Sub-byte (int4/uint4) or variable-width: size via `storage_bytes`.
        let numel: usize = shape.iter().product();
        let need = byte_offset + dtype.storage_bytes(numel);
        if need > buffer_len {
            return Err(SessionError::from(
                onnx_runtime_ep_api::EpError::InvalidTensorView {
                    reason: format!(
                        "sub-byte view needs {need} bytes but backing allocation is {buffer_len}"
                    ),
                },
            ));
        }
        return Ok(());
    }
    view_in_bounds(shape, strides, byte_offset, esize, buffer_len)?;
    Ok(())
}

/// Gather a strided view over `src` into a fresh contiguous row-major byte
/// buffer. `strides` are in **elements** (may be negative); `byte_offset` is the
/// byte position of the element origin within `src`. `esize` is the element
/// size in bytes (fixed-width types only — callers exclude sub-byte dtypes).
/// This is the materialization copy that turns a zero-copy view back into a
/// contiguous tensor for a strided-unaware consumer or the output boundary.
fn gather_view(
    src: &[u8],
    shape: &[usize],
    strides: &[i64],
    byte_offset: usize,
    esize: usize,
) -> Vec<u8> {
    let n: usize = shape.iter().product();
    let mut out = vec![0u8; n * esize];
    if n == 0 {
        return out;
    }
    let rank = shape.len();
    let mut idx = vec![0usize; rank];
    let mut w = 0usize;
    loop {
        let mut off = byte_offset as i64;
        for d in 0..rank {
            off += strides[d] * idx[d] as i64 * esize as i64;
        }
        let s = off as usize;
        out[w..w + esize].copy_from_slice(&src[s..s + esize]);
        w += esize;
        // Advance the row-major index; stop when it wraps to all-zero.
        let mut carried = true;
        for axis in (0..rank).rev() {
            idx[axis] += 1;
            if idx[axis] < shape[axis] {
                carried = false;
                break;
            }
            idx[axis] = 0;
        }
        if carried {
            break;
        }
    }
    out
}

/// Element count of a shape with overflow checking. A malicious or corrupt
/// shape whose dims multiply past `usize::MAX` would silently wrap under a plain
/// `iter().product()`, under-sizing the backing buffer. Returns
/// [`SessionError::ShapeOverflow`] instead so the caller allocates nothing.
fn checked_numel(dims: &[usize], value: impl FnOnce() -> String) -> Result<usize> {
    let mut acc = 1usize;
    for &d in dims {
        acc = match acc.checked_mul(d) {
            Some(n) => n,
            None => {
                return Err(SessionError::ShapeOverflow {
                    value: value(),
                    dims: dims.to_vec(),
                });
            }
        };
    }
    Ok(acc)
}

/// Byte size of `numel` elements of `dtype` with overflow checking. Even when
/// the element *count* fits in `usize` (guarded by [`checked_numel`]), the
/// element-count → bytes multiply can still wrap for a fixed-width dtype and
/// under-size the backing buffer. Returns [`SessionError::ShapeOverflow`] so the
/// caller allocates nothing rather than a wrapped, undersized buffer.
fn checked_storage_bytes(
    dtype: DataType,
    numel: usize,
    value: impl FnOnce() -> String,
    dims: &[usize],
) -> Result<usize> {
    dtype
        .checked_storage_bytes(numel)
        .ok_or_else(|| SessionError::ShapeOverflow {
            value: value(),
            dims: dims.to_vec(),
        })
}

/// The effective operator-set version governing `node` — the graph's imported
/// opset for the node's domain. Loaded IR is canonical (the default domain is
/// `""`, never `"ai.onnx"`; see [`onnx_runtime_ir::normalize_domain`]), so the
/// node's domain keys directly into the opset-import map.
fn effective_opset(graph: &Graph, node: &Node) -> u64 {
    graph
        .opset_imports
        .get(node.domain.as_str())
        .copied()
        .unwrap_or_else(|| {
            unreachable!(
                "internal invariant violated: node #{} ({}::{}) has no opset import",
                node.id.0,
                if node.domain.is_empty() {
                    "ai.onnx"
                } else {
                    &node.domain
                },
                node.op_type
            )
        })
}

/// Substitute concrete symbol bindings into a (possibly symbolic) shape.
/// Returns `None` if any dim is a symbol with no binding.
fn substitute(shape: &Shape, bindings: &HashMap<SymbolId, usize>) -> Option<Vec<usize>> {
    shape
        .iter()
        .map(|d| match d {
            Dim::Static(n) => Some(*n),
            Dim::Symbolic(s) => bindings.get(s).copied(),
        })
        .collect()
}

/// Like [`substitute`] but writes into `out` in place, reusing its existing
/// capacity (no heap allocation). Returns `false` (leaving `out` empty) if any
/// dim is an unbound symbol. Used by the decode-plan memo replay to refresh the
/// variant tail without allocating a fresh `Vec` per value per token.
fn substitute_into(
    shape: &Shape,
    bindings: &HashMap<SymbolId, usize>,
    out: &mut Vec<usize>,
) -> bool {
    out.clear();
    for d in shape {
        match d {
            Dim::Static(n) => out.push(*n),
            Dim::Symbolic(s) => match bindings.get(s) {
                Some(&v) => out.push(v),
                None => {
                    out.clear();
                    return false;
                }
            },
        }
    }
    true
}

/// Decode raw little-endian integer bytes as `i64` for `dtype`, or `None` if the
/// dtype is not an integer the shape math understands. Shared by the owned-buffer
/// and materialized-view integer-input readers.
fn bytes_as_i64(bytes: &[u8], dtype: DataType) -> Option<Vec<i64>> {
    match dtype {
        DataType::Int64 => Some(
            bytes
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        DataType::Int32 => Some(
            bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64)
                .collect(),
        ),
        _ => None,
    }
}

fn bytes_as_f64(bytes: &[u8], dtype: DataType) -> Option<Vec<f64>> {
    match dtype {
        DataType::Float32 => Some(
            bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64)
                .collect(),
        ),
        DataType::Float64 => Some(
            bytes
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        _ => None,
    }
}

/// Whether a runtime input is small enough to materialize as shape-propagation
/// data. Keep this gate ahead of `contiguous_bytes`: unsupported tensors must
/// degrade to absent shape-data without allocating or copying their contents.
fn bounded_shape_input(dtype: DataType, shape: &[usize]) -> bool {
    if !matches!(dtype, DataType::Int32 | DataType::Int64) {
        return false;
    }
    if shape.len() > 1 {
        return false;
    }
    shape
        .iter()
        .try_fold(1usize, |count, &dim| count.checked_mul(dim))
        .is_some_and(|count| count <= MAX_SHAPE_DATA_ELEMS)
}

/// Whether a node needs a float runtime input to resolve a data-dependent
/// output extent. The list is deliberately explicit, so shape propagation never
/// copies unrelated tensor data to host.
fn reads_float_shape_input(node: &Node, input_index: usize, opset: u64) -> bool {
    node.is_default_domain()
        && ((node.op_type == "Resize" && input_index == if opset == 10 { 1 } else { 2 })
            || (node.op_type == "NonMaxSuppression" && matches!(input_index, 0 | 1 | 3 | 4)))
}

fn kernel_input_uses_physical_capacity(node: &Node, input_index: usize) -> bool {
    // GQA treats the cache tensor extent as capacity and obtains the valid past
    // length from seqlens_k. Standard Attention instead derives past length from
    // the cache tensor extent itself.
    if node.domain == "com.microsoft"
        && node.op_type == "GroupQueryAttention"
        && matches!(input_index, 3 | 4)
    {
        return true;
    }
    // Default-domain `Attention` with an in-op KV cache (past_key=input 4,
    // past_value=input 5) can likewise treat the cache extent as physical
    // capacity, deriving the valid attended length on-device from the additive
    // attention mask (input 3) instead of the growing cache extent. This mirrors
    // the GQA treatment and is what lets the decode step bind the KV cache at a
    // fixed capacity so whole-step CUDA-graph capture stays shape-static. Gated
    // to the mask-driven, non-causal form (a present mask input and no
    // `is_causal` attribute): that path derives length from the mask frontier,
    // so the cache extent is pure capacity. Causal-attribute or mask-less
    // Attention still reads the cache extent as the valid length.
    node.is_default_domain()
        && node.op_type == "Attention"
        && matches!(input_index, 4 | 5)
        && node.inputs.get(3).is_some_and(Option::is_some)
        && node
            .attr("is_causal")
            .and_then(|attr| attr.as_int())
            .unwrap_or(0)
            == 0
        || (
            // `pkg.nxrt::IndexShare` mirrors the mask-driven Attention treatment.
            // Its capacity form emits the 3-output present that ALIASES the
            // fixed-capacity past bindings in place (past_key=input 3, past_value=
            // input 4) instead of a growing `past ⧺ current`, and carries the valid
            // length via the additive attention_bias (input 6) frontier — which the
            // kernel scans on-device to place the current token and bound the gather.
            // Binding those caches at physical capacity is what keeps whole-step
            // capture shape-static. Gated on the 3-output present + present bias so
            // the growing-concat form (1 output, or bias-less) still reads the cache
            // extent as the valid length.
            node.domain == "pkg.nxrt"
                && node.op_type == "IndexShare"
                && matches!(input_index, 3 | 4)
                && node.outputs.len() == 3
                && node.inputs.get(6).is_some_and(Option::is_some)
        )
}

fn kernel_input_uses_padded_capacity(node: &Node, input_index: usize) -> bool {
    // Persistent decode masks have a zero-filled suffix. Capacity-oriented
    // graphs intentionally read Shape at the allocation extent and ReduceSum is
    // unchanged by that suffix; prefix-sensitive transforms such as CumSum must
    // instead see the logical valid length.
    node.is_default_domain()
        && input_index == 0
        && matches!(node.op_type.as_str(), "Shape" | "ReduceSum")
}

/// Recompute the output shape of standard elementwise broadcasting ops from
/// their concrete runtime inputs. Loader inference is only a prior: a
/// data-dependent upstream value may acquire a different live shape.
fn runtime_elementwise_output_shape(
    node: &Node,
    input_shapes: &[Vec<usize>],
) -> Option<std::result::Result<Vec<usize>, onnx_runtime_ir::IrError>> {
    if !node.is_default_domain() {
        return None;
    }

    let input_count = match node.op_type.as_str() {
        "Add" | "Sub" | "Mul" | "Div" | "Pow" | "Mod" | "BitShift" | "Less" | "Greater"
        | "Equal" | "And" | "Or" | "Xor" | "LessOrEqual" | "GreaterOrEqual" => 2,
        "Where" => 3,
        "Min" | "Max" | "Sum" | "Mean" => input_shapes.len(),
        _ => return None,
    };
    if input_count == 0 || input_shapes.len() < input_count {
        return None;
    }

    let mut shape = input_shapes[0].clone();
    for input in &input_shapes[1..input_count] {
        shape = match broadcast_shapes(&shape, input) {
            Ok(shape) => shape,
            Err(error) => return Some(Err(error)),
        };
    }
    Some(Ok(shape))
}

/// Compute concrete output shapes from already-resolved input shapes and the
/// runtime *values* of integer inputs. This is the executor's fallback for the
/// rare value whose shape the loader's static (symbolic) inference could not pin
/// down — e.g. a `Slice` whose `ends` is produced by a runtime
/// `Shape → Min → Cast` chain, followed by movement/broadcast nodes.
///
/// Model-agnostic: it dispatches on the op type alone. Returns `None` for ops
/// this executor cannot resolve dynamically, which surfaces as
/// [`SessionError::UnresolvedShape`] exactly as before.
fn dynamic_output_shapes(
    node: &Node,
    input_shapes: &[Vec<usize>],
    input_dtypes: &[DataType],
    input_values: &[Option<Vec<i64>>],
    input_float_values: &[Option<Vec<f64>>],
    opset: u64,
) -> Option<Vec<Vec<usize>>> {
    match node.op_type.as_str() {
        "Resize" if node.is_default_domain() => {
            let input = input_shapes.first()?;
            let rank = input.len();
            let axes = if let Some(raw) = node.attr("axes").and_then(Attribute::as_ints) {
                let mut axes = Vec::with_capacity(raw.len());
                for &axis in raw {
                    let axis = if axis < 0 { axis + rank as i64 } else { axis };
                    let axis = usize::try_from(axis).ok()?;
                    if axis >= rank || axes.contains(&axis) {
                        return None;
                    }
                    axes.push(axis);
                }
                if axes.is_empty() {
                    (0..rank).collect()
                } else {
                    axes
                }
            } else {
                (0..rank).collect()
            };
            let scales_index = if opset == 10 { 1 } else { 2 };
            let scales = input_float_values
                .get(scales_index)
                .and_then(|values| values.as_deref())
                .filter(|values| !values.is_empty());
            let sizes = (opset >= 11)
                .then(|| input_values.get(3).and_then(|values| values.as_deref()))
                .flatten()
                .filter(|values| !values.is_empty());
            if scales.is_some() == sizes.is_some() {
                return None;
            }
            let mut output = input.clone();
            if let Some(scales) = scales {
                if scales.len() != axes.len()
                    || node
                        .attr("keep_aspect_ratio_policy")
                        .and_then(Attribute::as_str)
                        .is_some_and(|policy| policy != "stretch")
                {
                    return None;
                }
                for (&axis, &scale) in axes.iter().zip(scales) {
                    if !scale.is_finite() || scale <= 0.0 {
                        return None;
                    }
                    let extent = input[axis] as f64 * scale;
                    if extent > usize::MAX as f64 {
                        return None;
                    }
                    output[axis] = extent.floor() as usize;
                }
            } else {
                let sizes = sizes?;
                if sizes.len() != axes.len() {
                    return None;
                }
                let requested = sizes
                    .iter()
                    .map(|&size| usize::try_from(size).ok().filter(|&size| size > 0))
                    .collect::<Option<Vec<_>>>()?;
                match node
                    .attr("keep_aspect_ratio_policy")
                    .and_then(Attribute::as_str)
                    .unwrap_or("stretch")
                {
                    "stretch" => {
                        for (&axis, &size) in axes.iter().zip(&requested) {
                            output[axis] = size;
                        }
                    }
                    policy @ ("not_larger" | "not_smaller") => {
                        if axes.iter().any(|&axis| input[axis] == 0) {
                            return None;
                        }
                        let (numerator, denominator) = axes
                            .iter()
                            .zip(&requested)
                            .map(|(&axis, &size)| (size, input[axis]))
                            .reduce(|left, right| {
                                let order = (left.0 as u128 * right.1 as u128)
                                    .cmp(&(right.0 as u128 * left.1 as u128));
                                if (policy == "not_larger" && order.is_le())
                                    || (policy == "not_smaller" && order.is_ge())
                                {
                                    left
                                } else {
                                    right
                                }
                            })?;
                        if denominator == 0 {
                            return None;
                        }
                        for &axis in &axes {
                            let product = (input[axis] as u128).checked_mul(numerator as u128)?;
                            output[axis] = usize::try_from(
                                (product + denominator as u128 / 2) / denominator as u128,
                            )
                            .ok()?;
                        }
                    }
                    _ => return None,
                }
            }
            Some(vec![output])
        }
        // Opset-10+ `Slice`: data, starts, ends, [axes], [steps] as inputs. The
        // per-axis element count mirrors the `Slice` kernel's clamp semantics
        // exactly (ONNX reference), so the buffer we size here matches what the
        // kernel writes.
        "Slice" if node.is_default_domain() => {
            let data_shape = input_shapes.first()?;
            let starts = input_values.get(1)?.as_ref()?;
            let ends = input_values.get(2)?.as_ref()?;
            let (axes, steps) = onnx_runtime_ep_cpu::slice_axes_steps(
                starts.len(),
                input_values.get(3).and_then(|v| v.as_deref()),
                input_values.get(4).and_then(|v| v.as_deref()),
            );
            // Reuse the exact kernel geometry helper so the buffer we size here
            // always matches what the Slice kernel writes. Any error (length
            // mismatch, out-of-range axis, zero step) means "cannot resolve".
            let plan =
                onnx_runtime_ep_cpu::slice_plan(data_shape, starts, ends, &axes, &steps).ok()?;
            let count: Vec<usize> = plan.iter().map(|p| p.count).collect();
            Some(vec![count])
        }
        "NonMaxSuppression" if node.is_default_domain() => {
            let boxes_shape = input_shapes.first()?;
            let scores_shape = input_shapes.get(1)?;
            let boxes = input_float_values.first()?.as_ref()?;
            let scores = input_float_values.get(1)?.as_ref()?;
            let max_output_boxes_per_class = input_values
                .get(2)
                .and_then(|value| value.as_ref())
                .filter(|value| value.len() == 1)
                .map(|value| value[0])
                .unwrap_or(0);
            let iou_threshold = input_float_values
                .get(3)
                .and_then(|value| value.as_ref())
                .filter(|value| value.len() == 1)
                .map(|value| value[0] as f32)
                .unwrap_or(0.0);
            let score_threshold = input_float_values
                .get(4)
                .and_then(|value| value.as_ref())
                .filter(|value| value.len() == 1)
                .map(|value| value[0] as f32)
                .unwrap_or(f32::NEG_INFINITY);
            let center_point_box = node
                .attr("center_point_box")
                .and_then(Attribute::as_int)
                .unwrap_or(0);
            let boxes = boxes.iter().map(|&value| value as f32).collect::<Vec<_>>();
            let scores = scores.iter().map(|&value| value as f32).collect::<Vec<_>>();
            let selected = onnx_runtime_ep_cpu::non_max_suppression(
                &boxes,
                boxes_shape,
                &scores,
                scores_shape,
                max_output_boxes_per_class,
                iou_threshold,
                score_threshold,
                center_point_box,
            )
            .ok()?;
            Some(vec![vec![selected.len(), 3]])
        }
        "GroupQueryAttention" if node.domain == "com.microsoft" => {
            let query = input_shapes.first()?;
            let past_key = input_shapes.get(3)?;
            if query.len() != 3 || past_key.len() != 4 {
                return None;
            }
            let num_heads = usize::try_from(node.attr("num_heads")?.as_int()?).ok()?;
            let kv_heads = usize::try_from(node.attr("kv_num_heads")?.as_int()?).ok()?;
            if num_heads == 0 || kv_heads == 0 {
                return None;
            }
            let (output, head_dim) = if node.inputs.get(1).and_then(|input| *input).is_some() {
                let key = input_shapes.get(1)?;
                if key.len() != 3 || !key[2].is_multiple_of(kv_heads) {
                    return None;
                }
                (query.clone(), key[2] / kv_heads)
            } else {
                let packed_heads = num_heads.checked_add(kv_heads.checked_mul(2)?)?;
                if !query[2].is_multiple_of(packed_heads) {
                    return None;
                }
                let head_dim = query[2] / packed_heads;
                (
                    vec![query[0], query[1], head_dim.checked_mul(num_heads)?],
                    head_dim,
                )
            };
            let total_sequence_values = input_values.get(6)?.as_ref()?;
            if total_sequence_values.len() != 1 {
                return None;
            }
            let total_sequence = usize::try_from(total_sequence_values[0]).ok()?;
            let present_sequence = past_key[2].max(total_sequence);
            let present = vec![query[0], kv_heads, present_sequence, head_dim];
            let mut shapes = vec![output];
            if node.outputs.len() >= 2 {
                shapes.push(present.clone());
            }
            if node.outputs.len() >= 3 {
                shapes.push(present);
            }
            Some(shapes)
        }
        _ => {
            // Re-run the standard, opset-aware shape rule with the concrete
            // runtime input shapes and any small integer input values now
            // available. This covers shape-preserving movement and broadcasting
            // ops after a data-dependent node without duplicating their ONNX
            // semantics here (notably Unsqueeze axis normalization).
            let inputs = node
                .inputs
                .iter()
                .enumerate()
                .map(|(i, input)| {
                    if input.is_none() {
                        return Some(NodeIo::default());
                    }
                    let shape = input_shapes
                        .get(i)?
                        .iter()
                        .map(|&dim| i64::try_from(dim).ok().map(DimExpr::constant))
                        .collect::<Option<Vec<_>>>()?;
                    let dtype = *input_dtypes.get(i)?;
                    let shape_data = input_values.get(i)?.as_ref().and_then(|values| {
                        let elems = values
                            .iter()
                            .copied()
                            .map(DimExpr::constant)
                            .collect::<Vec<_>>();
                        match input_shapes[i].as_slice() {
                            [] if elems.len() == 1 => {
                                Some(ShapeData::scalar(dtype, elems[0].clone()))
                            }
                            [len] if *len == elems.len() => Some(ShapeData::vector(dtype, elems)),
                            _ => None,
                        }
                    });
                    Some(NodeIo {
                        type_info: Some(TypeInfo::new(dtype, shape)),
                        shape_data,
                    })
                })
                .collect::<Option<Vec<_>>>()?;
            let mut imports = HashMap::new();
            imports.insert(node.domain.clone(), opset);
            let mut interner = SymbolInterner::new(0x8000_0000);
            static REGISTRY: std::sync::OnceLock<InferenceRegistry> = std::sync::OnceLock::new();
            REGISTRY
                .get_or_init(InferenceRegistry::default_registry)
                .infer_node(node, &imports, inputs, MergePolicy::Strict, &mut interner)
                .ok()?
                .into_iter()
                .map(|output| {
                    output
                        .type_info?
                        .shape
                        .into_iter()
                        .map(|dim| usize::try_from(dim.as_const()?).ok())
                        .collect()
                })
                .collect()
        }
    }
}

/// Lower an exact `x * Sigmoid(x)` pair to the CPU EP's fused SiLU kernel.
///
/// The Sigmoid result must have exactly one consumer and must not be a graph
/// output, so removing its materialized value cannot change observable behavior.
fn fuse_silu_patterns(graph: &mut Graph) -> usize {
    let sigmoid_ids: Vec<NodeId> = graph
        .nodes
        .iter()
        .filter_map(|(id, node)| {
            (node.op_type == "Sigmoid"
                && node.is_default_domain()
                && node.inputs.len() == 1
                && node.outputs.len() == 1)
                .then_some(id)
        })
        .collect();
    let mut fused = 0;

    for sigmoid_id in sigmoid_ids {
        let Some(sigmoid) = graph.try_node(sigmoid_id) else {
            continue;
        };
        let Some(x) = sigmoid.inputs[0] else {
            continue;
        };
        let sigmoid_output = sigmoid.outputs[0];
        if graph.outputs.contains(&sigmoid_output) {
            continue;
        }
        let consumers = graph.consumers(sigmoid_output);
        if consumers.len() != 1 {
            continue;
        }
        let mul_id = consumers[0];
        let mul = graph.node(mul_id);
        if mul.op_type != "Mul"
            || !mul.is_default_domain()
            || mul.inputs.len() != 2
            || mul.outputs.len() != 1
            || !((mul.inputs[0] == Some(x) && mul.inputs[1] == Some(sigmoid_output))
                || (mul.inputs[1] == Some(x) && mul.inputs[0] == Some(sigmoid_output)))
        {
            continue;
        }

        let mut silu = mul.clone();
        silu.op_type = "Silu".to_string();
        silu.domain = "com.microsoft".to_string();
        silu.inputs = vec![Some(x)];
        silu.attributes.clear();
        graph.replace_node(mul_id, silu);
        graph.remove_node(sigmoid_id);
        fused += 1;
    }

    if fused != 0 {
        graph
            .opset_imports
            .entry("com.microsoft".to_string())
            .or_insert(1);
    }
    fused
}

struct WeightStoreInitializerResolver(Arc<WeightStore>);

impl InitializerResolver for WeightStoreInitializerResolver {
    fn bytes<'a>(&'a self, weight: &'a onnx_runtime_ir::WeightRef) -> Option<&'a [u8]> {
        self.0.bytes(weight)
    }
}

fn run_ep_scoped_passes(
    graph: &mut Graph,
    weights: &Arc<WeightStore>,
    ep: &dyn ExecutionProvider,
) -> Result<()> {
    let passes = ep.custom_passes();
    if passes.is_empty() {
        return Ok(());
    }

    let resolver = Arc::new(WeightStoreInitializerResolver(Arc::clone(weights)));
    let context = onnx_runtime_optimizer::PassContext::new().with_initializer_resolver(resolver);
    onnx_runtime_optimizer::run_passes(graph, &passes, &context)?;

    // Best-effort shape refresh: the passes may have rewritten nodes whose
    // output shapes downstream reads. A *data-dependent* invalidity (e.g. a
    // `Slice` with step 0) is the runtime kernel's contract to reject, not a
    // load-time error — before EP passes existed this re-inference did not run,
    // so the graph built and the actionable diagnostic surfaced at `run`.
    // Re-infer on a clone and adopt the refreshed shapes only on success so such
    // a failure neither aborts the build nor leaves the graph partially updated;
    // the executor's own resolution still validates shapes at run time.
    let registry = InferenceRegistry::default_registry();
    let opset_imports = graph.opset_imports.clone();
    let mut refreshed = graph.clone();
    if registry
        .infer_graph(&mut refreshed, &opset_imports, MergePolicy::Permissive)
        .is_ok()
    {
        *graph = refreshed;
    }
    Ok(())
}

fn validate_if_branch_outputs(graph: &Graph, node: &Node) -> Result<()> {
    let Some(then_branch) = graph.subgraphs.get(&(node.id, "then_branch".to_string())) else {
        return Ok(());
    };
    let Some(else_branch) = graph.subgraphs.get(&(node.id, "else_branch".to_string())) else {
        return Ok(());
    };

    if then_branch.outputs.len() != else_branch.outputs.len() {
        return Err(SessionError::ControlFlow {
            op: "If".to_string(),
            reason: format!(
                "branches declare different output counts: then_branch has {}, \
                 else_branch has {}",
                then_branch.outputs.len(),
                else_branch.outputs.len()
            ),
        });
    }
    if then_branch.outputs.len() != node.outputs.len() {
        return Err(SessionError::ControlFlow {
            op: "If".to_string(),
            reason: format!(
                "node declares {} output(s), but each branch declares {}",
                node.outputs.len(),
                then_branch.outputs.len()
            ),
        });
    }
    for (index, (&then_output, &else_output)) in then_branch
        .outputs
        .iter()
        .zip(&else_branch.outputs)
        .enumerate()
    {
        if then_branch.value_type_is_known(then_output)
            && else_branch.value_type_is_known(else_output)
        {
            let then_dtype = then_branch.value(then_output).dtype;
            let else_dtype = else_branch.value(else_output).dtype;
            if then_dtype != else_dtype {
                return Err(SessionError::ControlFlow {
                    op: "If".to_string(),
                    reason: format!(
                        "branches declare different dtypes for output {index}: \
                         then_branch is {then_dtype:?}, else_branch is {else_dtype:?}"
                    ),
                });
            }
        }
    }
    Ok(())
}

fn validate_control_flow_signatures(graph: &Graph) -> Result<()> {
    for (_, node) in graph.nodes.iter() {
        if node.op_type == "If" && matches!(node.domain.as_str(), "" | "ai.onnx") {
            validate_if_branch_outputs(graph, node)?;
        }
    }
    for subgraph in graph.subgraphs.values() {
        validate_control_flow_signatures(subgraph)?;
    }
    Ok(())
}

/// Reject operators no execution provider can run, before EP optimizer passes
/// run. An optimizer pass's postcondition validation walks the whole graph and
/// would otherwise surface a less actionable structural error (e.g. an
/// opset-import invariant) instead of the actionable unsupported-operator
/// diagnostic callers rely on.
///
/// A CUDA graph may legitimately delegate unsupported nodes to a CPU fallback
/// (see [`cuda_fallback_report`]), so an unsupported op is not fatal there; the
/// check is limited to the terminal (non-CUDA) EP. Only nodes with fully static
/// declared input shapes are pre-validated: a symbolic/data-dependent shape is
/// resolved and validated at run time, so pre-checking a contrib op whose
/// support is shape-conditional would change behavior for valid graphs.
fn reject_unsupported_operators(graph: &Graph, ep: &dyn ExecutionProvider) -> Result<()> {
    if ep.device_type() == DeviceType::Cuda {
        return Ok(());
    }
    for (node_id, node) in graph.nodes.iter() {
        if onnx_runtime_loader::is_ep_context_op(&node.op_type, &node.domain)
            || is_control_flow_op(&node.op_type, &node.domain)
            || is_sequence_op(&node.op_type, &node.domain)
        {
            continue;
        }

        let shapes = node
            .inputs
            .iter()
            .map(|input| {
                input
                    .map(|value| graph.value(value).shape.clone())
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>();
        // Defer nodes with any non-static declared input shape to the run-time
        // kernel gate, which sees concrete shapes.
        if !shapes.iter().all(|shape| as_static_shape(shape).is_some()) {
            continue;
        }
        let input_dtypes = node
            .inputs
            .iter()
            .map(|input| {
                input
                    .map(|value| graph.value(value).dtype)
                    .unwrap_or(DataType::Undefined)
            })
            .collect::<Vec<_>>();
        let layouts = vec![TensorLayout::contiguous(); shapes.len()];
        let opset = effective_opset(graph, node);
        if let KernelMatch::Unsupported { reason } =
            ep.supports_op(node, opset, &shapes, &input_dtypes, &layouts)
        {
            return Err(SessionError::unsupported_op(
                node,
                node_id,
                opset,
                ep.name(),
                reason,
            ));
        }
    }
    Ok(())
}

fn cuda_fallback_report(
    graph: &Graph,
    ep: &dyn ExecutionProvider,
) -> Option<ExecutionProviderFallbackReport> {
    if ep.device_type() != DeviceType::Cuda {
        return None;
    }

    let mut issues = Vec::new();
    collect_cuda_coverage_issues(graph, graph, ep, "graph", &mut issues);
    if issues.is_empty() {
        return None;
    }

    let mut assigned_ops = BTreeSet::new();
    let assigned_node_count = collect_executable_ops(graph, &mut assigned_ops);
    Some(ExecutionProviderFallbackReport {
        requested_provider: ep.name().to_string(),
        fallback_provider: "cpu_ep".to_string(),
        assigned_node_count,
        assigned_ops: assigned_ops.into_iter().collect(),
        declines: issues,
    })
}

fn collect_executable_ops(graph: &Graph, ops: &mut BTreeSet<String>) -> usize {
    let mut count = 0;
    for (_, node) in graph.nodes.iter() {
        if !onnx_runtime_loader::is_ep_context_op(&node.op_type, &node.domain) {
            count += 1;
            ops.insert(format!("{}::{}", canonical_domain(node), node.op_type));
        }
    }
    for subgraph in graph.subgraphs.values() {
        count += collect_executable_ops(subgraph, ops);
    }
    count
}

fn format_cuda_coverage_issues(issues: &[ExecutionProviderDecline]) -> String {
    const MAX_EXAMPLES_PER_CLASS: usize = 3;

    let mut groups: BTreeMap<(String, String, String), Vec<String>> = BTreeMap::new();
    for issue in issues {
        groups
            .entry((
                issue.domain.clone(),
                issue.op_type.clone(),
                issue.reason.clone(),
            ))
            .or_default()
            .push(issue.node.clone());
    }

    groups
        .into_iter()
        .map(|((domain, op_type, reason), mut nodes)| {
            nodes.sort();
            let count = nodes.len();
            nodes.truncate(MAX_EXAMPLES_PER_CLASS);
            format!(
                "{domain}::{op_type}: {reason} [count={count}; examples: {}]",
                nodes.join(", ")
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

fn collect_cuda_coverage_issues(
    graph: &Graph,
    opset_graph: &Graph,
    ep: &dyn ExecutionProvider,
    scope: &str,
    issues: &mut Vec<ExecutionProviderDecline>,
) {
    for (node_id, node) in graph.nodes.iter() {
        if onnx_runtime_loader::is_ep_context_op(&node.op_type, &node.domain)
            || is_control_flow_op(&node.op_type, &node.domain)
            || is_sequence_op(&node.op_type, &node.domain)
        {
            continue;
        }

        let shapes = node
            .inputs
            .iter()
            .map(|input| {
                input
                    .map(|value| graph.value(value).shape.clone())
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>();
        let layouts = node
            .inputs
            .iter()
            .map(|input| {
                input
                    .map(|value| graph.value(value).layout.clone())
                    .unwrap_or_else(TensorLayout::contiguous)
            })
            .collect::<Vec<_>>();
        let input_dtypes = node
            .inputs
            .iter()
            .map(|input| {
                input
                    .map(|value| graph.value(value).dtype)
                    .unwrap_or(DataType::Undefined)
            })
            .collect::<Vec<_>>();

        let opset = effective_opset(opset_graph, node);
        if let KernelMatch::Unsupported { reason } =
            ep.supports_op(node, opset, &shapes, &input_dtypes, &layouts)
        {
            issues.push(ExecutionProviderDecline {
                node: format_node_identity(scope, node_id, node),
                domain: canonical_domain(node),
                op_type: node.op_type.clone(),
                reason: reason.into_owned(),
            });
            continue;
        }

        let Some(concrete_shapes) = shapes
            .iter()
            .map(|shape| as_static_shape(shape))
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        if let Err(error) = ep.get_kernel(node, &concrete_shapes, opset) {
            issues.push(ExecutionProviderDecline {
                node: format_node_identity(scope, node_id, node),
                domain: canonical_domain(node),
                op_type: node.op_type.clone(),
                reason: format!("kernel creation failed: {error}"),
            });
        }
    }

    for ((node_id, attribute), subgraph) in &graph.subgraphs {
        let sub_scope = format!("{scope}/node#{}/{}", node_id.0, attribute);
        collect_cuda_coverage_issues(subgraph, opset_graph, ep, &sub_scope, issues);
    }
}

fn canonical_domain(node: &Node) -> String {
    if node.domain.is_empty() {
        "ai.onnx".to_string()
    } else {
        node.domain.clone()
    }
}

fn format_node_identity(scope: &str, node_id: NodeId, node: &Node) -> String {
    if node.name.is_empty() {
        format!("{scope}/node#{}", node_id.0)
    } else {
        format!("{scope}/node#{} {:?}", node_id.0, node.name)
    }
}

fn build_lazy_weight_handles(
    graph: &Graph,
    weights: &Arc<WeightStore>,
    ep: &dyn ExecutionProvider,
) -> Result<HashMap<ValueId, WeightHandle>> {
    let capabilities = ep.capabilities();
    if !capabilities.advertises(onnx_runtime_ep_api::NXRT_WEIGHT_PAGING_CAPABILITY) {
        return Ok(HashMap::new());
    }

    let boundary = LazyWeightBoundary::BlockQuantizedMoe;
    let mut handles = HashMap::new();
    for (&value, weight) in &graph.initializers {
        let graph_value = graph.value(value);
        let consumers = graph.consumers(value);
        let lazy_only = graph_value.producer.is_none()
            && !graph.outputs.contains(&value)
            && !consumers.is_empty()
            && consumers.into_iter().all(|consumer| {
                let node = graph.node(consumer);
                boundary.matches(&node.domain, &node.op_type)
            });
        if !lazy_only {
            continue;
        }
        let Some((mapping_id, offset, len)) = weights.external_mmap_provenance(weight) else {
            continue;
        };
        let region = ExternalMmapRegion {
            mapping_id,
            offset,
            len,
        };
        let dtype = weight.dtype();
        let shape = weight.dims().to_vec();
        let weight = weight.clone();
        let store = Arc::clone(weights);
        let lazy = LazyWeight::block_quantized_moe(vec![region], move || {
            let bytes = store.bytes(&weight).ok_or_else(|| {
                onnx_runtime_ep_api::WeightHandleError::InvalidResident(
                    "external weight bytes are no longer available".into(),
                )
            })?;
            ResidentWeight::new(dtype, shape.clone(), Arc::<[u8]>::from(bytes))
        })
        .map_err(|error| {
            SessionError::Internal(format!(
                "cannot create lazy weight handle for value#{}: {error}",
                value.0
            ))
        })?;
        handles.insert(value, WeightHandle::Lazy(lazy));
    }
    Ok(handles)
}



/// Map a [`crate::sequence::SequenceError`] into an actionable `SessionError`.
fn seq_err(e: crate::sequence::SequenceError) -> SessionError {
    e.into()
}

/// Normalize a possibly-negative ONNX `axis` against `rank`, returning the
/// non-negative axis or `None` when out of `[-rank, rank-1]`.
fn normalize_axis(axis: i64, rank: usize) -> Option<usize> {
    let r = rank as i64;
    let a = if axis < 0 { axis + r } else { axis };
    if a < 0 || a >= r {
        None
    } else {
        Some(a as usize)
    }
}

fn scan_list_attr(node: &Node, name: &str, count: usize, default: i64) -> Result<Vec<i64>> {
    match node.attr(name) {
        None => Ok(vec![default; count]),
        Some(attr) => {
            let values = attr.as_ints().ok_or_else(|| SessionError::ControlFlow {
                op: "Scan".to_string(),
                reason: format!("attribute '{name}' must be an INTS list"),
            })?;
            if values.len() != count {
                return Err(SessionError::ControlFlow {
                    op: "Scan".to_string(),
                    reason: format!(
                        "attribute '{name}' has {} value(s), expected {count}",
                        values.len()
                    ),
                });
            }
            Ok(values.to_vec())
        }
    }
}

/// Whether `(op_type, domain)` is one of the standard subgraph-bearing
/// control-flow ops the executor handles recursively (default `ai.onnx`
/// domain). Kept in lock-step with the loader's `validate_no_control_flow`
/// allow-list.
fn is_control_flow_op(op_type: &str, domain: &str) -> bool {
    domain.is_empty() && matches!(op_type, "If" | "Loop" | "Scan")
}

/// Whether `(op_type, domain)` is an ONNX **Sequence** op the executor handles
/// directly (default `ai.onnx` domain). Like control-flow ops these are handled
/// at the executor level rather than as leaf [`Kernel`](onnx_runtime_ep_api::Kernel)s
/// because a `Kernel` sees only tensor views, never a *sequence-of-tensors*
/// runtime value. Kept as a small self-contained routing predicate (mirroring
/// [`is_control_flow_op`]) so it never collides with the EP kernel registry.
fn is_sequence_op(op_type: &str, domain: &str) -> bool {
    domain.is_empty()
        && matches!(
            op_type,
            "SequenceEmpty"
                | "SequenceConstruct"
                | "SequenceInsert"
                | "SequenceErase"
                | "SequenceAt"
                | "SequenceLength"
                | "SplitToSequence"
                | "ConcatFromSequence"
        )
}

/// Whether a Sequence op yields a *sequence* value (vs. a tensor). Used at build
/// to mark sequence-typed values so they are excluded from tensor buffer sizing.
fn produces_sequence_output(op_type: &str, domain: &str) -> bool {
    domain.is_empty()
        && matches!(
            op_type,
            "SequenceEmpty"
                | "SequenceConstruct"
                | "SequenceInsert"
                | "SequenceErase"
                | "SplitToSequence"
        )
}

/// Read a single scalar `i64` element from a length-1 tensor (Loop's `M`).
fn tensor_scalar_i64(t: &Tensor) -> Option<i64> {
    if t.dtype != DataType::Int64 || t.numel() != 1 {
        return None;
    }
    t.as_bytes()
        .get(..8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
}

/// Read a single scalar bool from a length-1 `BOOL` tensor (a `BOOL` is one
/// byte; any nonzero is true, per ONNX).
fn tensor_scalar_bool(t: &Tensor) -> Option<bool> {
    if t.dtype != DataType::Bool || t.numel() != 1 {
        return None;
    }
    t.as_bytes().first().map(|&b| b != 0)
}

/// Build a length-1 `i64` scalar tensor (Loop's `iter_num` body input).
fn scalar_i64_tensor(v: i64) -> Result<Tensor> {
    Tensor::from_raw(DataType::Int64, vec![], &v.to_le_bytes())
}

/// Build a scalar `BOOL` tensor (Loop's `cond` body input).
fn scalar_bool_tensor(v: bool) -> Result<Tensor> {
    Tensor::from_raw(DataType::Bool, vec![], &[u8::from(v)])
}

fn missing_capture_error(attr_key: &str, name: &str) -> SessionError {
    SessionError::Internal(format!(
        "control-flow body '{attr_key}' captures free variable '{name}', but it is not \
         available in the enclosing scope. RULES #1: a subgraph may only reference outer \
         values that are graph inputs, initializers, or produced by an upstream node in an \
         enclosing graph; '{name}' matches none of these"
    ))
}

/// Names a graph or any nested body needs from its enclosing lexical scope.
/// A nested requirement stops propagating when this graph defines that name,
/// because the nested body will bind the local value at execution time.
fn required_outer_names(graph: &Graph) -> HashSet<String> {
    let formal_set: HashSet<ValueId> = graph.inputs.iter().copied().collect();
    let local_names: HashSet<&str> = graph
        .values
        .iter()
        .filter_map(|(_, value)| value.name.as_deref())
        .collect();
    let mut required = HashSet::new();
    for (vid, value) in graph.values.iter() {
        if value.producer.is_none()
            && !formal_set.contains(&vid)
            && !graph.initializers.contains_key(&vid)
            && let Some(name) = &value.name
        {
            required.insert(name.clone());
        }
    }
    for nested in graph.subgraphs.values() {
        for name in required_outer_names(nested) {
            if !local_names.contains(name.as_str()) {
                required.insert(name);
            }
        }
    }
    required
}


impl Drop for Executor {
    fn drop(&mut self) {
        // Observability (F5): a one-line memo activity summary when
        // `ONNX_GENAI_DECODE_MEMO_STATS=1`, so an on-model A/B can confirm the
        // memo actually fired (`replayed>0`) rather than being silently gated out.
        if self.decode_memo_enabled
            && std::env::var("ONNX_GENAI_DECODE_MEMO_STATS")
                .map(|v| matches!(v.as_str(), "1" | "true" | "on"))
                .unwrap_or(false)
        {
            let (primed, rebuilt, replayed, ineligible) = self.decode_memo_counts();
            let (views_reused, dispatch_elided) = self.decode_view_plan_counts();
            eprintln!(
                "[decode-memo] primed={primed} rebuilt={rebuilt} replayed={replayed} \
                 ineligible={ineligible} views_reused={views_reused} \
                 dispatch_elided={dispatch_elided}"
            );
        }
        let _ = self.ep.reset_device_graph();
        self.device_graph_signature = None;
        // Free every buffer via the owning EP (DeviceBuffer has no Drop).
        for (_, buf) in self.buffers.drain() {
            let _ = self.ep.deallocate(buf);
        }
        self.shared_buffers.clear();
    }
}


mod build;
mod bindings;
mod run;
mod capture;
mod dispatch;
mod control_flow;
mod platform;
mod sequence_ops;
pub(crate) use platform::auto_detect_cpu_ep;

#[cfg(test)]
mod tests;

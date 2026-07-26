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

impl Executor {
    /// Compile a graph + weights into a runnable executor on the CPU EP.
    pub(crate) fn build(
        graph: Graph,
        weights: Arc<WeightStore>,
        ep: Arc<dyn ExecutionProvider>,
    ) -> Result<Self> {
        Self::build_with_cuda_requirement(
            graph,
            weights,
            ep,
            onnx_genai_runtime_config::runtime_config().require_cuda,
        )
    }

    fn build_with_cuda_requirement(
        mut graph: Graph,
        weights: Arc<WeightStore>,
        mut ep: Arc<dyn ExecutionProvider>,
        require_cuda: bool,
    ) -> Result<Self> {
        let mut placement_span = trace_span("session.node_placement", "session");
        let requested_provider = placement_span.as_ref().map(|_| ep.name().to_string());
        let requested_device = placement_span
            .as_ref()
            .map(|_| ep.device_type().trace_name().into_owned());
        let nodes_before_placement = graph.num_nodes();
        // Reject incompatible control-flow signatures before EP optimizers run:
        // optimizer postconditions recursively validate subgraphs and can
        // otherwise obscure the actionable If diagnostic with a structural
        // error from a malformed branch.
        validate_control_flow_signatures(&graph)?;
        // Reject structurally invalid graphs (a non-DAG) and operators no EP can
        // run *before* EP optimizers run. An optimizer pass's postcondition
        // validation would otherwise obscure the actionable load-time diagnostic
        // (a wrapped `CycleDetected`, or an opset-import invariant instead of the
        // unsupported-operator error) with a structural error. Mirrors the
        // control-flow signature check above.
        graph.topological_order()?;
        reject_unsupported_operators(&graph, ep.as_ref())?;
        let silu_fused = fuse_silu_patterns(&mut graph);
        let graph_before_ep_passes = graph.clone();
        let ep_pass_nodes_before = graph.num_nodes();
        run_ep_scoped_passes(&mut graph, &weights, ep.as_ref())?;
        let ep_pass_nodes_after = graph.num_nodes();
        let mut execution_provider_fallback_report = cuda_fallback_report(&graph, ep.as_ref());
        let fallback_declines = execution_provider_fallback_report
            .as_ref()
            .map_or(0, |report| report.declines.len());
        if let Some(report) = &mut execution_provider_fallback_report {
            if require_cuda {
                return Err(SessionError::HeterogeneousPlacementRequired {
                    unsupported_nodes: report.to_string(),
                });
            }
            graph = graph_before_ep_passes;
            ep = auto_detect_cpu_ep()?;
            run_ep_scoped_passes(&mut graph, &weights, ep.as_ref())?;
            let mut assigned_ops = BTreeSet::new();
            report.assigned_node_count = collect_executable_ops(&graph, &mut assigned_ops);
            report.assigned_ops = assigned_ops.into_iter().collect();
            eprintln!(
                "[onnx-genai-warning] {report}. Set ONNX_GENAI_REQUIRE_CUDA=1 to reject this fallback"
            );
        }
        if let Some(span) = placement_span.as_mut() {
            let mut assigned_ops = BTreeSet::new();
            let assigned_nodes = collect_executable_ops(&graph, &mut assigned_ops);
            span.set_args(
                Args::new()
                    .with("requested_provider", requested_provider.unwrap_or_default())
                    .with("requested_device", requested_device.unwrap_or_default())
                    .with("selected_provider", ep.name().to_string())
                    .with(
                        "selected_device",
                        ep.device_type().trace_name().into_owned(),
                    )
                    .with("nodes_before", nodes_before_placement as u64)
                    .with("nodes_after", graph.num_nodes() as u64)
                    .with("ep_pass_nodes_before", ep_pass_nodes_before as u64)
                    .with("ep_pass_nodes_after", ep_pass_nodes_after as u64)
                    .with("silu_fused", silu_fused as u64)
                    .with("assigned_nodes", assigned_nodes as u64)
                    .with("assigned_op_classes", assigned_ops.len() as u64)
                    .with("fallback_declines", fallback_declines as u64),
            );
        }
        drop(placement_span);
        // Topological order up front: also validates the selected graph is a DAG.
        let order = graph.topological_order()?;
        let weight_handles = {
            let mut span = trace_span("session.lazy_weight_handles", "session");
            let handles = build_lazy_weight_handles(&graph, &weights, ep.as_ref())?;
            if let Some(span) = span.as_mut() {
                span.set_args(
                    Args::new()
                        .with("handles", handles.len() as u64)
                        .with("initializers", graph.initializers.len() as u64),
                );
            }
            handles
        };

        let mut value_shapes: HashMap<ValueId, Shape> = HashMap::new();
        let mut value_dtypes: HashMap<ValueId, DataType> = HashMap::new();
        let mut buffers: HashMap<ValueId, DeviceBuffer> = HashMap::new();
        let mut buffer_shapes: HashMap<ValueId, Vec<usize>> = HashMap::new();

        // 1) Initializers: record metadata and back resident consumers with a
        //    device buffer. A non-host nxrt initializer used exclusively at the
        //    lazy fused-MoE boundary deliberately has no eager buffer; the EP
        //    materializes it through its WeightHandle on demand. If any resident
        //    consumer (or graph output) coexists, no handle is built and the one
        //    eager buffer is shared by every consumer. Host mmap bytes retain the
        //    existing zero-copy borrow path.
        let init_align = TensorLayout::contiguous().alignment;
        let mut initializer_span = trace_span("session.initializer_buffers", "session");
        let mut initializer_count = 0_u64;
        let mut initializer_bytes = 0_u64;
        let mut borrowed_initializers = 0_u64;
        let mut copied_initializers = 0_u64;
        let mut lazy_initializers = 0_u64;
        for (&vid, weight) in &graph.initializers {
            let dtype = weight.dtype();
            let dims = weight.dims().to_vec();
            value_dtypes.insert(vid, dtype);
            value_shapes.insert(vid, dims.iter().map(|&d| Dim::Static(d)).collect());
            if !ep.device_id().is_host_accessible() && weight_handles.contains_key(&vid) {
                if initializer_span.is_some() {
                    lazy_initializers += 1;
                }
                continue;
            }
            let bytes = weights.bytes(weight).ok_or_else(|| {
                SessionError::Internal(format!("weight bytes unavailable for value#{}", vid.0))
            })?;
            if initializer_span.is_some() {
                initializer_count += 1;
                initializer_bytes += bytes.len() as u64;
            }
            // Only borrow when the value has NO producer. The borrowed
            // `DeviceBuffer` aliases read-only mmap/inline storage, so it must
            // never be written. A legitimate initializer always has
            // `producer == None`; a malformed graph can reuse an initializer's
            // `ValueId` as a node output (see loader `validate_no_initializer_producer`),
            // giving it a producer — a kernel would then write through
            // `as_mut_ptr()` into read-only mmap (SIGSEGV / aliasing UB). In
            // that case fall back to the owned writable copy below.
            let producer_less = graph.value(vid).producer.is_none();
            let borrow_align = if matches!(weight, WeightRef::External { .. }) {
                host_dtype_alignment(dtype)
            } else {
                init_align
            };
            let buf = if ep.device_id().is_host_accessible()
                && producer_less
                && !bytes.is_empty()
                && (bytes.as_ptr() as usize).is_multiple_of(borrow_align)
            {
                if initializer_span.is_some() {
                    borrowed_initializers += 1;
                }
                // Zero-copy: alias the suitably aligned initializer bytes. For
                // external data this is only the dtype alignment; inline data
                // retains the EP allocation alignment requirement.
                // SAFETY: `bytes` borrows live mmap storage in `weights` or
                // inline storage in `graph`; both executor fields outlive every
                // buffer use. The range is `bytes.len()` long,
                // `borrow_align`-aligned, and treated as read-only.
                unsafe {
                    DeviceBuffer::from_borrowed_parts(
                        bytes.as_ptr() as *mut std::ffi::c_void,
                        ep.device_id(),
                        bytes.len(),
                        borrow_align,
                    )
                }
            } else {
                if initializer_span.is_some() {
                    copied_initializers += 1;
                }
                let mut owned = ep.allocate(bytes.len().max(1), init_align)?;
                ep.copy_from_host(bytes, &mut owned)?;
                owned
            };
            buffer_shapes.insert(vid, dims);
            buffers.insert(vid, buf);
        }
        if let Some(span) = initializer_span.as_mut() {
            span.set_args(
                Args::new()
                    .with("initializers", initializer_count)
                    .with("bytes", initializer_bytes)
                    .with("borrowed_initializers", borrowed_initializers)
                    .with("copied_initializers", copied_initializers)
                    .with("lazy_initializers", lazy_initializers)
                    .with("buffers", buffers.len() as u64),
            );
        }

        // 2) Record the loader shape + dtype of every remaining value (graph
        //    inputs and node outputs). No allocation yet — shapes may be
        //    symbolic and are only sized once resolved.
        for &vid in &graph.inputs {
            value_shapes
                .entry(vid)
                .or_insert_with(|| graph.value(vid).shape.clone());
            value_dtypes.entry(vid).or_insert(graph.value(vid).dtype);
        }
        for &nid in &order {
            for &out in &graph.node(nid).outputs {
                value_shapes
                    .entry(out)
                    .or_insert_with(|| graph.value(out).shape.clone());
                value_dtypes.entry(out).or_insert(graph.value(out).dtype);
            }
        }

        let has_symbols = value_shapes.values().any(|s| as_static_shape(s).is_none());

        // Sequence-typed values own no tensor buffer: a Sequence op stores its
        // list in `sequences` at run time. Mark every value produced by a
        // sequence-producing op so buffer sizing skips it (and so a Sequence
        // graph output is diagnosed cleanly rather than read as tensor bytes).
        let mut sequence_values: HashSet<ValueId> = HashSet::new();
        for &nid in &order {
            let node = graph.node(nid);
            if produces_sequence_output(&node.op_type, &node.domain) {
                for &out in &node.outputs {
                    sequence_values.insert(out);
                }
            }
        }

        // Output value ids of every control-flow node, used to seed their
        // concrete (branch-selected) shapes into the capture plan so downstream
        // capturable consumers do not each form an eager seam.
        let mut control_flow_output_values: HashSet<ValueId> = HashSet::new();
        for &nid in &order {
            let node = graph.node(nid);
            if is_control_flow_op(&node.op_type, &node.domain) {
                for &out in &node.outputs {
                    control_flow_output_values.insert(out);
                }
            }
        }

        // 3) Build the structural per-node plan.
        let mut plan_span = trace_span("session.execution_plan", "session");
        let mut plan = Vec::with_capacity(order.len());
        let mut skipped_epcontext = 0_u64;
        for &nid in &order {
            let node = graph.node(nid);
            // EPContext nodes are pre-compiled: they bypass placement and were
            // already restored through their owning EP by the session's
            // consume path (§55.3). They must never be resolved as ordinary
            // kernels — the CPU EP has no `EPContext` kernel — so skip them
            // here.
            if onnx_runtime_loader::is_ep_context_op(&node.op_type, &node.domain) {
                if plan_span.is_some() {
                    skipped_epcontext += 1;
                }
                continue;
            }
            // Preserve positional input arity: keep interior `None` (omitted
            // optional) slots so a later present input is not misread as the
            // omitted one, but trim trailing `None`s (a trailing omitted
            // optional just lowers the arity, matching ONNX semantics).
            let mut slots: Vec<Option<ValueId>> = node.inputs.clone();
            while matches!(slots.last(), Some(None)) {
                slots.pop();
            }
            let inputs = slots;
            let outputs: Vec<ValueId> = node.outputs.clone();
            let input_dtypes: Vec<DataType> = inputs
                .iter()
                .map(|v| {
                    v.map(|vid| value_dtypes[&vid])
                        .unwrap_or(DataType::Undefined)
                })
                .collect();
            let output_dtypes: Vec<DataType> = outputs.iter().map(|v| value_dtypes[v]).collect();
            plan.push(NodePlan {
                node_id: nid,
                inputs,
                outputs,
                input_dtypes,
                output_dtypes,
            });
        }
        if let Some(span) = plan_span.as_mut() {
            span.set_args(
                Args::new()
                    .with("topological_nodes", order.len() as u64)
                    .with("plan_len", plan.len() as u64)
                    .with("skipped_epcontext_nodes", skipped_epcontext)
                    .with("values", graph.values.len() as u64)
                    .with("inputs", graph.inputs.len() as u64)
                    .with("outputs", graph.outputs.len() as u64)
                    .with("has_symbols", has_symbols),
            );
        }

        // 4) name → value id and the set of caller-required inputs.
        let mut input_index = HashMap::new();
        let mut required_inputs = Vec::new();
        for &vid in &graph.inputs {
            if graph.initializers.contains_key(&vid) {
                continue; // pre-filled; not a caller input
            }
            required_inputs.push(vid);
            if let Some(name) = &graph.value(vid).name {
                input_index.insert(name.clone(), vid);
            }
        }

        // Full name → value id map (every named value in the graph), used to
        // resolve a nested subgraph's outer-scope captures by name.
        let mut name_index = HashMap::new();
        for (vid, value) in graph.values.iter() {
            if let Some(name) = &value.name {
                name_index.insert(name.clone(), vid);
            }
        }

        let mut exec = Self {
            graph,
            weights,
            ep,
            weight_handles,
            buffers,
            buffer_shapes,
            value_shapes,
            value_dtypes,
            plan,
            input_index,
            required_inputs,
            has_symbols,
            cache: KernelCache::default(),
            name_index,
            subgraph_execs: HashMap::new(),
            control_flow_stats: ControlFlowStats::default(),
            if_last_predicate: HashMap::new(),
            device_graph_signature: None,
            capture_schedule: None,
            capture_segmentation: Vec::new(),
            control_flow_output_values,
            capture_cf_shapes: HashMap::new(),
            capture_warm_signature: None,
            capture_warm_shapes: HashMap::new(),
            capture_warm_seeded: HashMap::new(),
            capture_quarantine_ops: HashSet::new(),
            last_capture_failed_node: None,
            views: HashMap::new(),
            pinned: HashSet::new(),
            sequence_values,
            shared_buffers: HashMap::new(),
            sequences: HashMap::new(),
            seq_elem_values: HashMap::new(),
            execution_provider_fallback_report,
            trace: TraceContext::noop(),
            scratch_input_shapes: Vec::new(),
            decode_memo_enabled: decode_memo_env_enabled(),
            decode_memo_verify: cfg!(debug_assertions) || decode_memo_verify_env_enabled(),
            decode_memo: None,
            decode_memo_prev_bindings: None,
            decode_memo_last_action: DecodeMemoAction::Disabled,
            decode_memo_resolved: HashMap::new(),
            decode_memo_primed_count: 0,
            decode_memo_rebuilt_count: 0,
            decode_memo_replayed_count: 0,
            decode_memo_ineligible_count: 0,
            decode_view_plan: None,
            decode_views_reused_count: 0,
            decode_dispatch_elided_count: 0,
            decode_view_plan_sig_mismatch_streak: 0,
            decode_view_plan_disabled: false,
        };

        // 5) Fully-static graphs are materialized eagerly (buffers + the whole
        //    "compiled plan" of kernels), so the first `run` sees only cache
        //    hits. Symbolic graphs cannot be sized until a `run` fixes their
        //    shapes, so their buffers/kernels are created on first use.
        if !exec.has_symbols {
            let mut span = trace_span("session.static_materialize", "session");
            let empty = HashMap::new();
            let resolved = exec.resolve_all(&empty)?;
            exec.size_buffers(&resolved)?;
            exec.compile_all(&resolved)?;
            if let Some(span) = span.as_mut() {
                span.set_args(
                    Args::new()
                        .with("resolved_values", resolved.len() as u64)
                        .with("buffers", exec.buffers.len() as u64)
                        .with("plan_len", exec.plan.len() as u64)
                        .with("cache_entries", exec.cache.stats().entries as u64),
                );
            }
        }
        Ok(exec)
    }

    /// Allocate `vid`'s buffer for `dims`, or reuse the existing allocation when
    /// it is already sized for `dims` (the run-scoped reuse path).
    fn ensure_buffer(&mut self, vid: ValueId, dtype: DataType, dims: &[usize]) -> Result<()> {
        if self.buffer_shapes.get(&vid).map(|s| s.as_slice()) == Some(dims) {
            return Ok(()); // identical shape → reuse allocation
        }
        if let Some(old) = self.buffers.remove(&vid) {
            self.ep.deallocate(old)?;
        }
        self.shared_buffers.remove(&vid);
        let numel = checked_numel(dims, || format!("value#{}", vid.0))?;
        let size = checked_storage_bytes(dtype, numel, || format!("value#{}", vid.0), dims)?;
        let buf = self
            .ep
            .allocate(size.max(1), TensorLayout::contiguous().alignment)?;
        self.buffers.insert(vid, buf);
        self.buffer_shapes.insert(vid, dims.to_vec());
        Ok(())
    }

    /// Resolve every value's concrete shape by substituting `bindings` into its
    /// loader shape. A value whose shape stays symbolic (unbound) cannot be
    /// sized: report it as an uninferred shape, naming its producing op.
    fn resolve_all(
        &self,
        bindings: &HashMap<SymbolId, usize>,
    ) -> Result<HashMap<ValueId, Vec<usize>>> {
        let mut resolved = HashMap::with_capacity(self.value_shapes.len());
        for (&vid, shape) in &self.value_shapes {
            // Sequence-typed values have no meaningful tensor shape and are
            // never buffer-sized; skip them so a static graph does not trip the
            // unresolved-shape check on a sequence value.
            if self.sequence_values.contains(&vid) {
                continue;
            }
            match substitute(shape, bindings) {
                Some(dims) => {
                    resolved.insert(vid, dims);
                }
                None => {
                    let value = self.graph.value(vid);
                    let name = value
                        .name
                        .clone()
                        .unwrap_or_else(|| format!("value#{}", vid.0));
                    let op = value
                        .producer
                        .map(|nid| self.graph.node(nid).op_type.clone())
                        .unwrap_or_else(|| "<graph input>".to_string());
                    return Err(SessionError::UnresolvedShape { value: name, op });
                }
            }
        }
        Ok(resolved)
    }

    /// Like [`Self::resolve_all`] but never errors: values whose shape stays
    /// symbolic (a data-dependent extent the loader could not pin down) are
    /// simply omitted, to be resolved just-in-time during execution once their
    /// producing node's inputs are concrete.
    fn resolve_soft(&self, bindings: &HashMap<SymbolId, usize>) -> HashMap<ValueId, Vec<usize>> {
        let mut resolved = HashMap::with_capacity(self.value_shapes.len());
        for (&vid, shape) in &self.value_shapes {
            if let Some(dims) = substitute(shape, bindings) {
                resolved.insert(vid, dims);
            }
        }
        resolved
    }

    /// F5 Stage 1: resolve every value's concrete shape for a memo-eligible
    /// eager step, replaying the length-invariant partition through the
    /// [`DecodePlanMemo`] when the step is plan-identical to the memoized one,
    /// and re-substituting only the length-varying tail. On any signature change
    /// (prefill→decode, batch change, non-length dim change, …) it falls back to
    /// a full [`Self::resolve_soft`] and (re)builds the memo by diffing this
    /// step's bindings with the previous eligible step's (R1 two-real-step
    /// derivation). The output is provably byte-identical to `resolve_soft`
    /// (asserted every replay when [`Self::decode_memo_verify`] is set).
    fn resolve_soft_decode_memo(
        &mut self,
        bindings: &HashMap<SymbolId, usize>,
        external: &ExternalBindings,
    ) -> HashMap<ValueId, Vec<usize>> {
        // L-abstracted fingerprint of the persistent binding set (KV cache). Pure
        // length growth leaves it unchanged; a structural change forces a rebuild.
        let external_sig = self.decode_external_signature(external);
        // --- Fast path: an active memo whose non-varying bindings and binding
        //     signature are unchanged. Replays the invariant partition with ZERO
        //     allocation: the persistent working map is taken in place, the
        //     previous step's just-in-time entries are stripped, invariant entries
        //     are left untouched (byte-identical by construction), and only the
        //     variant tail is re-substituted into its existing `Vec`s.
        if self
            .decode_memo
            .as_ref()
            .is_some_and(|memo| memo.matches(bindings, &external_sig))
        {
            // Own the memo for the duration so `self.value_shapes` /
            // `decode_memo_resolved` can be borrowed disjointly; restored below.
            let memo = self.decode_memo.take().unwrap();
            let mut resolved = std::mem::take(&mut self.decode_memo_resolved);
            // Drop the previous step's data-dependent (JIT) entries so the run
            // loop recomputes them; the canonical partition is retained in place.
            resolved.retain(|vid, _| memo.canonical.contains(vid));
            // Restore any length-invariant entry missing from the persistent map.
            // By construction (the run loop only adds/overwrites, never drops
            // canonical keys, and the rebuild step persisted the full map) this
            // never fires in steady state, so replay stays allocation-free; it is
            // a defensive re-seed from the memo's authoritative invariant plan.
            for (&vid, dims) in &memo.invariant_shapes {
                resolved.entry(vid).or_insert_with(|| dims.clone());
            }
            // Re-substitute only the variant tail, reusing each `Vec`'s capacity.
            for &vid in &memo.variant_values {
                let shape = &self.value_shapes[&vid];
                match resolved.get_mut(&vid) {
                    Some(slot) => {
                        if !substitute_into(shape, bindings, slot) {
                            resolved.remove(&vid);
                        }
                    }
                    None => {
                        if let Some(dims) = substitute(shape, bindings) {
                            resolved.insert(vid, dims);
                        }
                    }
                }
            }
            if self.decode_memo_verify {
                // R1 verifiable safety net: the replay must equal a fresh resolve.
                let fresh = self.resolve_soft(bindings);
                assert_eq!(
                    resolved, fresh,
                    "decode-plan memo replay diverged from resolve_soft (unsound invariant \
                     classification)"
                );
            }
            self.decode_memo = Some(memo);
            self.decode_memo_last_action = DecodeMemoAction::Replayed;
            self.decode_memo_replayed_count += 1;
            self.decode_memo_prev_bindings = Some(bindings.clone());
            return resolved;
        }

        // --- Slow path: full resolve, then try to (re)build the memo by diffing
        //     this step with the previous eligible step (two real steps, R1) —
        //     but only for a steady single-token-decode growth transition (M==1
        //     gate), so the memo never activates on prefill.
        //
        // Defense-in-depth (Chew): drop the persistent working map on every
        // non-replay step so a stale invariant `Vec` from a retired plan can
        // never leak into a future replay (e.g. if a run errored before the
        // end-of-step persist-back). It is repopulated by this step's persist-back
        // (or, if that step errors, left empty and defensively re-seeded next
        // replay), so the clear costs nothing on the steady path.
        self.decode_memo_resolved.clear();
        // F5 Stage 2 defense-in-depth: retire the cached view plan on every
        // non-replay (rebuild/prime) step. A Rebuilt step rebuilds it fresh only
        // at its successful end (below, in `run_scoped_mode`); a step that errors
        // before that leaves it `None`, so a stale invariant view alias from a
        // retired plan can never be reinstated into a later replay.
        self.decode_view_plan = None;
        let resolved = self.resolve_soft(bindings);
        match self.decode_memo_prev_bindings.take() {
            Some(prev) if is_decode_growth_transition(&prev, bindings) => {
                let decode_varying: HashSet<SymbolId> = bindings
                    .iter()
                    .filter(|(sym, val)| prev.get(*sym) != Some(*val))
                    .map(|(&sym, _)| sym)
                    .collect();
                let mut invariant_shapes = HashMap::with_capacity(resolved.len());
                let mut variant_values = Vec::new();
                let mut canonical = HashSet::with_capacity(resolved.len());
                for (&vid, dims) in &resolved {
                    canonical.insert(vid);
                    if shape_references_any(&self.value_shapes[&vid], &decode_varying) {
                        variant_values.push(vid);
                    } else {
                        invariant_shapes.insert(vid, dims.clone());
                    }
                }
                self.decode_memo = Some(DecodePlanMemo {
                    reference_bindings: bindings.clone(),
                    decode_varying,
                    invariant_shapes,
                    variant_values,
                    canonical,
                    reference_external_sig: external_sig,
                });
                self.decode_memo_last_action = DecodeMemoAction::Rebuilt;
                self.decode_memo_rebuilt_count += 1;
            }
            _ => {
                // First observation of a regime, a bound-symbol-set change, or a
                // non-decode transition (e.g. prefill): drop any stale memo and
                // wait for the next steady-decode step to diff against.
                self.decode_memo = None;
                self.decode_memo_last_action = DecodeMemoAction::Primed;
                self.decode_memo_primed_count += 1;
            }
        }
        self.decode_memo_prev_bindings = Some(bindings.clone());
        resolved
    }

    /// L-abstracted structural fingerprint of the persistent device-I/O binding
    /// set (see [`DecodeBindingSig`]). Order-independent; the declared symbolic
    /// shape (graph-static) stands in for the concrete one, so pure-L KV growth
    /// yields an unchanged signature while a binding added/removed, a role flip,
    /// or a dtype change yields a different one.
    fn decode_external_signature(&self, external: &ExternalBindings) -> Vec<DecodeBindingSig> {
        let mut sig: Vec<DecodeBindingSig> = external
            .inputs
            .keys()
            .map(|&vid| (vid, true))
            .chain(external.outputs.keys().map(|&vid| (vid, false)))
            .map(|(vid, is_input)| DecodeBindingSig {
                vid,
                is_input,
                dtype: self.value_dtypes[&vid],
                decl_shape: self.value_shapes[&vid].clone(),
            })
            .collect();
        sig.sort_by_key(|s| (s.vid.0, s.is_input));
        sig
    }

    #[cfg(test)]
    fn set_decode_memo_enabled(&mut self, enabled: bool) {
        self.decode_memo_enabled = enabled;
        self.decode_memo_verify = true;
        self.decode_memo = None;
        self.decode_memo_prev_bindings = None;
        self.decode_memo_resolved.clear();
        self.decode_memo_last_action = DecodeMemoAction::Disabled;
        self.decode_memo_primed_count = 0;
        self.decode_memo_rebuilt_count = 0;
        self.decode_memo_replayed_count = 0;
        self.decode_memo_ineligible_count = 0;
        self.decode_view_plan = None;
        self.decode_views_reused_count = 0;
        self.decode_dispatch_elided_count = 0;
        self.decode_view_plan_sig_mismatch_streak = 0;
        self.decode_view_plan_disabled = false;
    }

    #[cfg(test)]
    fn decode_memo_action(&self) -> DecodeMemoAction {
        self.decode_memo_last_action
    }

    /// F5 Stage 1 memo activity counters `(primed, rebuilt, replayed, ineligible)`
    /// accumulated over this executor's lifetime. `replayed > 0` on a real decode
    /// run is the proof the memo actually fires (not silently gated out); the
    /// coordinator's on-model A/B reads these to reject a vacuous pass.
    pub(crate) fn decode_memo_counts(&self) -> (u64, u64, u64, u64) {
        (
            self.decode_memo_primed_count,
            self.decode_memo_rebuilt_count,
            self.decode_memo_replayed_count,
            self.decode_memo_ineligible_count,
        )
    }

    /// F5 Stage 2 activity counters `(views_reused, dispatch_elided)` accumulated
    /// over this executor's lifetime. Both `> 0` on a real decode run prove the
    /// invariant view-reuse / dispatch-elision path actually fired (not a vacuous
    /// pass); an on-model A/B reads these alongside the Stage-1 counters.
    pub(crate) fn decode_view_plan_counts(&self) -> (u64, u64) {
        (
            self.decode_views_reused_count,
            self.decode_dispatch_elided_count,
        )
    }

    /// F5 Stage 2 replay guard: every retained view's source buffer must still be
    /// the identical allocation (same base pointer *and* capacity) it was under
    /// when the plan was built. A realloc or move — even one that preserves the
    /// logical shape — invalidates the cached byte offsets/strides, so this must
    /// return `false` and force a full rebuild. This is the pointer/capacity
    /// obligation Stage 1 deferred (it cached shapes only); Stage 2 pays it here.
    fn stage2_buffer_sig_matches(&self, plan: &DecodeViewPlan) -> bool {
        plan.source_buffer_sig.iter().all(|(vid, ptr, cap)| {
            self.buffers
                .get(vid)
                .is_some_and(|buf| buf.as_ptr() as usize == *ptr && buf.len() == *cap)
        })
    }

    /// F5 Stage 2: build the *candidate* view plan from the state left by a
    /// successful memo Rebuilt step. A node is a candidate iff every one of its
    /// outputs is a zero-copy view (`self.views`) whose **shape is in the memo's
    /// proven-invariant partition** — so Stage 1 guarantees the replayed `resolved`
    /// map carries that exact shape every step. The candidate's source buffers can
    /// still be classified variant (e.g. a fixed-capacity KV buffer whose logical
    /// length grows): its concrete stability is confirmed separately by
    /// [`Self::validate_decode_view_plan`] (byte-identical view across a second real
    /// step) and guarded each replay by the buffer-identity signature. Returns
    /// `None` if nothing is a candidate.
    fn build_decode_view_plan(&self) -> Option<DecodeViewPlan> {
        let memo = self.decode_memo.as_ref()?;
        let invariant = |vid: &ValueId| memo.invariant_shapes.contains_key(vid);
        let mut elided_nodes = HashSet::new();
        let mut retained_views = Vec::new();
        let mut sources: HashSet<ValueId> = HashSet::new();
        for pi in 0..self.plan.len() {
            let outputs = &self.plan[pi].outputs;
            if outputs.is_empty() {
                continue;
            }
            // Every output must be a zero-copy view whose shape Stage 1 already
            // proves invariant (so `resolved[output]` is stable and correct when
            // the node is elided).
            let all_view_invariant = outputs
                .iter()
                .all(|ovid| invariant(ovid) && self.views.contains_key(ovid));
            if !all_view_invariant {
                continue;
            }
            elided_nodes.insert(pi);
            for ovid in outputs {
                let view = self.views[ovid].clone();
                sources.insert(view.source);
                retained_views.push((*ovid, view));
            }
        }
        if elided_nodes.is_empty() {
            return None;
        }
        // Record the buffer identity of every aliased source (the Stage-2 guard).
        let mut source_buffer_sig = Vec::with_capacity(sources.len());
        for &src in &sources {
            let buf = self.buffers.get(&src)?;
            source_buffer_sig.push((src, buf.as_ptr() as usize, buf.len()));
        }
        Some(DecodeViewPlan {
            elided_nodes,
            retained_views,
            pinned_sources: sources.into_iter().collect(),
            source_buffer_sig,
            validated: false,
        })
    }

    /// F5 Stage 2: confirm a candidate plan on a second real decode step. The step
    /// ran every node normally (no elision), so `self.views` now holds freshly
    /// built aliases; keep only the candidate nodes whose every output view is
    /// **byte-identical** (source, shape, strides, byte offset) to the one captured
    /// when the plan was built. This two-real-step confirmation (mirroring Stage 1's
    /// varying-set derivation) rejects any view whose geometry actually drifts — e.g.
    /// a position-indexed slice into a fixed-capacity buffer — before it is ever
    /// elided. Sources and the buffer-identity signature are recomputed from the
    /// surviving views. The plan is marked validated iff anything survives.
    fn validate_decode_view_plan(&self, mut plan: DecodeViewPlan) -> Option<DecodeViewPlan> {
        let view_matches = |a: &ValueView, b: &ValueView| {
            a.source == b.source
                && a.shape == b.shape
                && a.strides == b.strides
                && a.byte_offset == b.byte_offset
        };
        // A node survives iff every one of its retained outputs still matches the
        // freshly rebuilt view this step.
        let mut surviving_nodes: HashSet<usize> = HashSet::new();
        let node_outputs = |pi: usize| self.plan[pi].outputs.clone();
        for &pi in &plan.elided_nodes {
            let ok = node_outputs(pi).iter().all(|ovid| {
                match (
                    plan.retained_views.iter().find(|(v, _)| v == ovid),
                    self.views.get(ovid),
                ) {
                    (Some((_, cached)), Some(fresh)) => view_matches(cached, fresh),
                    _ => false,
                }
            });
            if ok {
                surviving_nodes.insert(pi);
            }
        }
        if surviving_nodes.is_empty() {
            return None;
        }
        // Rebuild retained views / sources / signature from the survivors only,
        // using the freshly built (identical) views.
        let surviving_outputs: HashSet<ValueId> = surviving_nodes
            .iter()
            .flat_map(|&pi| self.plan[pi].outputs.clone())
            .collect();
        let mut retained_views = Vec::new();
        let mut sources: HashSet<ValueId> = HashSet::new();
        for ovid in surviving_outputs {
            let view = self.views.get(&ovid)?.clone();
            sources.insert(view.source);
            retained_views.push((ovid, view));
        }
        let mut source_buffer_sig = Vec::with_capacity(sources.len());
        for &src in &sources {
            let buf = self.buffers.get(&src)?;
            source_buffer_sig.push((src, buf.as_ptr() as usize, buf.len()));
        }
        plan.elided_nodes = surviving_nodes;
        plan.retained_views = retained_views;
        plan.pinned_sources = sources.into_iter().collect();
        plan.source_buffer_sig = source_buffer_sig;
        plan.validated = true;
        Some(plan)
    }

    /// Size (allocate or reuse) a backing buffer for every value from its
    /// resolved concrete shape. Initializers already hold their weights and are
    /// left untouched. Values whose shape is not (yet) in `resolved` — the
    /// data-dependent ones filled in during execution — are skipped here and
    /// sized just-in-time in the run loop.
    fn size_buffers(&mut self, resolved: &HashMap<ValueId, Vec<usize>>) -> Result<()> {
        self.size_buffers_excluding(resolved, &HashSet::new())
    }

    fn size_buffers_excluding(
        &mut self,
        resolved: &HashMap<ValueId, Vec<usize>>,
        excluded: &HashSet<ValueId>,
    ) -> Result<()> {
        let vids: Vec<ValueId> = self.value_shapes.keys().copied().collect();
        for vid in vids {
            if self.graph.initializers.contains_key(&vid) || excluded.contains(&vid) {
                continue;
            }
            // Sequence-typed values own no tensor buffer (their list lives in
            // `sequences` at run time), so never size one for them.
            if self.sequence_values.contains(&vid) {
                continue;
            }
            let dtype = self.value_dtypes[&vid];
            let Some(dims) = resolved.get(&vid).cloned() else {
                continue;
            };
            self.ensure_buffer(vid, dtype, &dims)?;
        }
        Ok(())
    }

    /// Resolved input shapes of a plan node, in positional order. An omitted
    /// optional input (`None` slot) has no shape; it takes an empty shape,
    /// which the run loop only ever pairs with an absent placeholder view.
    fn node_input_shapes(
        plan: &NodePlan,
        resolved: &HashMap<ValueId, Vec<usize>>,
    ) -> Vec<Vec<usize>> {
        plan.inputs
            .iter()
            .map(|v| v.map(|vid| resolved[&vid].clone()).unwrap_or_default())
            .collect()
    }

    /// Populate the kernel cache for the compiled plan against `resolved` shapes.
    fn compile_all(&mut self, resolved: &HashMap<ValueId, Vec<usize>>) -> Result<()> {
        let mut span = trace_span("session.kernel_compile_plan", "session");
        let cache_entries_before = self.cache.stats().entries;
        let mut compiled_nodes = 0_u64;
        let mut skipped_control_flow = 0_u64;
        let mut skipped_sequence = 0_u64;
        for i in 0..self.plan.len() {
            let node_id = self.plan[i].node_id;
            let node = self.graph.node(node_id);
            // Control-flow ops (If/Loop/Scan) are not leaf kernels — they execute
            // nested subgraphs through the executor's own path, so they have no
            // entry in the EP kernel registry and must not be compiled here.
            if is_control_flow_op(&node.op_type, &node.domain) {
                if span.is_some() {
                    skipped_control_flow += 1;
                }
                continue;
            }
            // Sequence ops are executor-handled (they operate on sequence-of-
            // tensor values, not tensor views) — they have no EP kernel and must
            // not be compiled here, exactly like control-flow ops.
            if is_sequence_op(&node.op_type, &node.domain) {
                if span.is_some() {
                    skipped_sequence += 1;
                }
                continue;
            }
            if span.is_some() {
                compiled_nodes += 1;
            }
            let input_shapes = Self::node_input_shapes(&self.plan[i], resolved);
            let input_dtypes = self.plan[i].input_dtypes.clone();
            let constant_inputs: Vec<bool> = self.plan[i]
                .inputs
                .iter()
                .map(|input| input.is_some_and(|vid| self.graph.initializers.contains_key(&vid)))
                .collect();
            let node = self.graph.node(node_id);
            let opset = effective_opset(&self.graph, node);
            self.cache.get_or_create(
                node_id,
                node,
                &input_shapes,
                &input_dtypes,
                &constant_inputs,
                opset,
                self.ep.as_ref(),
            )?;
        }
        if let Some(span) = span.as_mut() {
            span.set_args(
                Args::new()
                    .with("plan_len", self.plan.len() as u64)
                    .with("compiled_nodes", compiled_nodes)
                    .with("skipped_control_flow", skipped_control_flow)
                    .with("skipped_sequence", skipped_sequence)
                    .with("cache_entries_before", cache_entries_before as u64)
                    .with("cache_entries_after", self.cache.stats().entries as u64),
            );
        }
        Ok(())
    }

    pub(crate) fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    pub(crate) fn control_flow_stats(&self) -> ControlFlowStats {
        self.control_flow_stats
    }

    pub(crate) fn device_id(&self) -> onnx_runtime_ir::DeviceId {
        self.ep.device_id()
    }

    pub(crate) fn allocate_device_binding(
        &self,
        input_name: String,
        output_name: Option<String>,
        dtype: DataType,
        physical_shape: Vec<usize>,
        logical_shape: Vec<usize>,
    ) -> Result<DeviceIoBinding> {
        let expose_logical_input_shape = self.input_index.get(&input_name).is_none_or(|&vid| {
            if output_name.is_some() {
                !self.binding_consumers_use_physical_capacity(vid)
            } else {
                !self.binding_consumers_use_padded_capacity(vid)
            }
        });
        DeviceIoBinding::allocate(
            self.ep.clone(),
            input_name,
            true,
            output_name,
            dtype,
            physical_shape,
            logical_shape,
            expose_logical_input_shape,
        )
    }

    pub(crate) fn allocate_device_output_binding(
        &self,
        output_name: String,
        dtype: DataType,
        physical_shape: Vec<usize>,
        logical_shape: Vec<usize>,
    ) -> Result<DeviceIoBinding> {
        DeviceIoBinding::allocate(
            self.ep.clone(),
            String::new(),
            false,
            Some(output_name),
            dtype,
            physical_shape,
            logical_shape,
            false,
        )
    }

    fn binding_consumers_use_physical_capacity(&self, input: ValueId) -> bool {
        let mut found = false;
        for plan in &self.plan {
            for (slot, value) in plan.inputs.iter().enumerate() {
                if *value != Some(input) {
                    continue;
                }
                found = true;
                if !kernel_input_uses_physical_capacity(self.graph.node(plan.node_id), slot) {
                    return false;
                }
            }
        }
        found
    }

    fn binding_consumers_use_padded_capacity(&self, input: ValueId) -> bool {
        let mut found = false;
        for plan in &self.plan {
            for (slot, value) in plan.inputs.iter().enumerate() {
                if *value != Some(input) {
                    continue;
                }
                found = true;
                if !kernel_input_uses_padded_capacity(self.graph.node(plan.node_id), slot) {
                    return false;
                }
            }
        }
        found
    }

    /// The compiled graph, retained for the §55.4 EPContext dump path: the
    /// exporter needs the (post-optimize) graph to serialise a `*_ctx.onnx`
    /// context-cache model with compiled partitions spliced out.
    pub(crate) fn graph(&self) -> &Graph {
        &self.graph
    }

    pub(crate) fn execution_provider_fallback_report(
        &self,
    ) -> Option<&ExecutionProviderFallbackReport> {
        self.execution_provider_fallback_report.as_ref()
    }

    /// Attach the shared runtime trace context. When enabled, the executor opens
    /// one span per executed op so kernels can attach kernel-variant and
    /// capture-rejection reasons. Propagated to any already-built child
    /// (control-flow subgraph) executors so nested ops are traced too.
    pub(crate) fn set_trace_context(&mut self, trace: TraceContext) {
        for child in self.subgraph_execs.values_mut() {
            child.set_trace_context(trace.clone());
        }
        self.trace = trace;
    }

    /// Live weight bytes backing the graph, needed alongside [`Self::graph`] so
    /// the EPContext dump can encode initializers into the context model.
    pub(crate) fn weights(&self) -> &Arc<WeightStore> {
        &self.weights
    }

    /// Warmup: re-touch the shape-keyed cache for the compiled plan so the first
    /// real `run` sees only cache hits (§11.3). Only meaningful for fully-static
    /// graphs, whose plan shapes are known at build; symbolic graphs cannot be
    /// pre-compiled without a concrete shape and warm up on their first `run`.
    pub(crate) fn warmup(&mut self) -> Result<()> {
        if self.has_symbols {
            return Ok(());
        }
        let empty = HashMap::new();
        let resolved = self.resolve_all(&empty)?;
        self.compile_all(&resolved)
    }

    /// Bind the graph's symbols to concrete sizes from the actual bound-input
    /// shapes, validating rank and static dims and detecting symbol conflicts.
    fn bind_symbols(
        &self,
        inputs: &[(&str, &Tensor)],
        external: &ExternalBindings,
    ) -> Result<HashMap<SymbolId, usize>> {
        let mut bindings: HashMap<SymbolId, usize> = HashMap::new();
        for (name, tensor) in inputs {
            let vid = *self
                .input_index
                .get(*name)
                .ok_or_else(|| SessionError::InputNotFound {
                    name: (*name).to_string(),
                })?;
            self.bind_input_shape(name, vid, tensor.dtype, &tensor.shape, &mut bindings)?;
        }
        for (&vid, value) in &external.inputs {
            let name = self.graph.value(vid).name.as_deref().unwrap_or("<unnamed>");
            self.bind_input_shape(name, vid, value.dtype, &value.shape, &mut bindings)?;
        }
        Ok(bindings)
    }

    fn bind_input_shape(
        &self,
        name: &str,
        vid: ValueId,
        dtype: DataType,
        shape: &[usize],
        bindings: &mut HashMap<SymbolId, usize>,
    ) -> Result<()> {
        let want_dtype = self.value_dtypes[&vid];
        if dtype != want_dtype {
            return Err(SessionError::DtypeMismatch {
                name: name.to_string(),
                expected: format!("{want_dtype:?}"),
                got: format!("{dtype:?}"),
            });
        }
        let decl = &self.value_shapes[&vid];
        if decl.len() != shape.len() {
            return Err(SessionError::RankMismatch {
                name: name.to_string(),
                expected: decl.len(),
                got: shape.len(),
            });
        }
        for (dim, &actual) in decl.iter().zip(shape) {
            match dim {
                Dim::Static(n) if *n != actual => {
                    return Err(SessionError::ShapeMismatch {
                        name: name.to_string(),
                        expected: as_static_shape(decl).unwrap_or_default(),
                        got: shape.to_vec(),
                    });
                }
                Dim::Static(_) => {}
                Dim::Symbolic(s) => {
                    if let Some(&prev) = bindings.get(s) {
                        if prev != actual {
                            let sym = self
                                .symbol_name(*s)
                                .unwrap_or_else(|| format!("symbol#{}", s.0));
                            return Err(SessionError::SymbolConflict {
                                symbol: sym,
                                first: prev,
                                second: actual,
                            });
                        }
                    } else {
                        bindings.insert(*s, actual);
                    }
                }
            }
        }
        Ok(())
    }

    /// Human-readable name of a symbol, if the graph recorded one.
    fn symbol_name(&self, s: SymbolId) -> Option<String> {
        self.graph
            .symbol_constraints
            .get(&s)
            .and_then(|c| c.name.clone())
    }

    /// Sequential topological executor.
    pub(crate) fn run(&mut self, inputs: &[(&str, &Tensor)]) -> Result<Vec<Tensor>> {
        self.run_outputs(inputs)?
            .into_iter()
            .map(|output| {
                match output {
                    SessionOutput::Tensor(tensor) => Ok(tensor),
                    SessionOutput::Sequence(_) => Err(SessionError::SequenceOp {
                        op: "<graph output>".to_string(),
                        reason: "the tensor-only run API received a Sequence graph output; use InferenceSession::run_outputs to preserve sequence values".to_string(),
                    }),
                }
            })
            .collect()
    }

    pub(crate) fn run_outputs(&mut self, inputs: &[(&str, &Tensor)]) -> Result<Vec<SessionOutput>> {
        self.run_scoped(inputs, &HashMap::new(), &ExternalBindings::default())?
            .into_iter()
            .map(|output| {
                output.ok_or_else(|| {
                    SessionError::Internal(
                        "ordinary run unexpectedly suppressed a bound graph output".into(),
                    )
                })
            })
            .collect()
    }

    pub(crate) fn run_with_device_bindings(
        &mut self,
        inputs: &[(&str, &Tensor)],
        bindings: &mut [DeviceIoBinding],
    ) -> Result<Vec<Option<Tensor>>> {
        let external = self.prepare_external_bindings(bindings)?;
        self.run_scoped(inputs, &HashMap::new(), &external)?
            .into_iter()
            .map(|output| match output {
                None => Ok(None),
                Some(SessionOutput::Tensor(tensor)) => Ok(Some(tensor)),
                Some(SessionOutput::Sequence(_)) => Err(SessionError::SequenceOp {
                    op: "<graph output>".to_string(),
                    reason: "run_with_device_bindings cannot return an unbound Sequence graph output; use run_outputs without tensor device bindings".to_string(),
                }),
            })
            .collect()
    }

    pub(crate) fn try_capture_with_device_bindings(
        &mut self,
        inputs: &[(&str, &Tensor)],
        bindings: &mut [DeviceIoBinding],
    ) -> Result<DeviceGraphCaptureResult> {
        let external = self.prepare_external_bindings(bindings)?;
        match self.run_scoped_mode(inputs, &HashMap::new(), &external, RunMode::Capture)? {
            ScopedRunResult::Executed(outputs) => {
                let mut tensors = Vec::with_capacity(outputs.len());
                for output in outputs {
                    match output {
                        None => tensors.push(None),
                        Some(SessionOutput::Tensor(tensor)) => tensors.push(Some(tensor)),
                        Some(SessionOutput::Sequence(_)) => {
                            self.reset_device_graph()?;
                            return Ok(DeviceGraphCaptureResult::NotCapturable(
                                CaptureDeclineReport::one(CaptureDecline::graph(
                                    "device graph capture cannot return a Sequence graph output",
                                )),
                            ));
                        }
                    }
                }
                self.device_graph_signature = Some(Self::binding_signature(bindings));
                Ok(DeviceGraphCaptureResult::Captured(tensors))
            }
            ScopedRunResult::NotCapturable(reason) => {
                Ok(DeviceGraphCaptureResult::NotCapturable(reason))
            }
        }
    }

    /// Replay the installed device graph for one decode step. Returns `true` when
    /// the graph remains installed and valid for the next step, or `false` when a
    /// control-flow branch flip retired it mid-step (the token was still produced
    /// correctly via an eager fallback) and the caller must re-warm/re-capture.
    pub(crate) fn replay_device_graph(&mut self, bindings: &mut [DeviceIoBinding]) -> Result<bool> {
        let external = self.prepare_external_bindings(bindings)?;
        let signature = Self::binding_signature(bindings);
        if self.device_graph_signature.as_ref() != Some(&signature) {
            self.reset_device_graph()?;
            return Err(SessionError::Internal(
                "device graph replay bindings changed shape, address, or I/O identity; graph was invalidated"
                    .into(),
            ));
        }
        // Whole-subgraph capture (a single graph, no eager seams) keeps the
        // zero-host-work fast path: just relaunch the one installed graph.
        // Segmented capture must re-establish the run context and interleave
        // segment replays with eager seam-node execution, so it routes through
        // the scoped runner in replay mode.
        let single_graph = self
            .capture_schedule
            .as_ref()
            .is_none_or(CaptureSchedule::is_single_graph);
        if single_graph {
            self.ep.replay_device_graph()?;
            return Ok(true);
        }
        match self.run_scoped_mode(&[], &HashMap::new(), &external, RunMode::Replay)? {
            // `run_scoped_mode` clears `capture_schedule` when a branch flip
            // retired the graph this step; report that so the caller re-arms.
            ScopedRunResult::Executed(_) => Ok(self.capture_schedule.is_some()),
            ScopedRunResult::NotCapturable(reason) => {
                self.reset_device_graph()?;
                Err(SessionError::Internal(format!(
                    "segmented device graph replay lost its schedule: {reason}"
                )))
            }
        }
    }

    pub(crate) fn reset_device_graph(&mut self) -> Result<bool> {
        self.device_graph_signature = None;
        self.capture_schedule = None;
        self.capture_cf_shapes.clear();
        self.capture_warm_seeded.clear();
        Ok(self.ep.reset_device_graph()?)
    }

    /// Structured segment-boundary reasons from the most recent capture: one
    /// entry per non-capturable seam node the CUDA EP ran eagerly between
    /// captured segments. Empty for a whole-subgraph (single-graph) capture.
    pub(crate) fn capture_segmentation(&self) -> &[CaptureDecline] {
        &self.capture_segmentation
    }

    /// Number of captured device-graph segments installed by the most recent
    /// capture (1 for a whole-subgraph capture, >=2 when seams split it).
    pub(crate) fn captured_segment_count(&self) -> usize {
        self.capture_schedule
            .as_ref()
            .map(CaptureSchedule::captured_segments)
            .unwrap_or(0)
    }

    pub(crate) fn check_device_capture_error(&self) -> Result<u32> {
        Ok(self.ep.check_device_capture_error()?)
    }

    pub(crate) fn device_allocation_counts(&self) -> Option<DeviceAllocationCounts> {
        self.ep
            .device_allocation_counts()
            .map(|(allocations, frees)| DeviceAllocationCounts { allocations, frees })
    }

    fn binding_signature(bindings: &[DeviceIoBinding]) -> Vec<DeviceBindingSignature> {
        bindings
            .iter()
            .map(|binding| DeviceBindingSignature {
                input_name: binding.input_name().to_string(),
                binds_input: binding.binds_input(),
                output_name: binding.output_name().map(str::to_string),
                dtype: binding.dtype,
                physical_shape: binding.physical_shape().to_vec(),
                device_ptr: binding.device_ptr() as usize,
            })
            .collect()
    }

    fn prepare_external_bindings(
        &self,
        bindings: &mut [DeviceIoBinding],
    ) -> Result<ExternalBindings> {
        let mut external = ExternalBindings::default();
        for binding in bindings {
            let input_name = binding.input_name().to_string();
            let bind_input = binding.binds_input();
            let output_name = binding.output_name().map(str::to_string);
            let dtype = binding.dtype;
            let len = binding.buffer().len();
            let alignment = binding.buffer().alignment();
            let device = binding.buffer().device();
            if device != self.ep.device_id() {
                return Err(SessionError::Internal(format!(
                    "device binding '{input_name}' is on {device:?}, session is on {:?}",
                    self.ep.device_id()
                )));
            }
            let physical_shape = binding.physical_shape();
            let required = dtype.storage_bytes(physical_shape.iter().product());
            if required > len {
                return Err(SessionError::Internal(format!(
                    "device binding '{input_name}' needs {required} bytes for {physical_shape:?}, allocation has {len}"
                )));
            }
            let ptr = binding.buffer_mut().as_mut_ptr();
            if bind_input {
                let input_vid = *self.input_index.get(&input_name).ok_or_else(|| {
                    SessionError::InputNotFound {
                        name: input_name.clone(),
                    }
                })?;
                let value = ExternalValue {
                    dtype,
                    shape: binding.kernel_input_shape().to_vec(),
                    accepts_subshape: false,
                    ptr,
                    len,
                    alignment,
                    device,
                };
                if external.inputs.insert(input_vid, value).is_some() {
                    return Err(SessionError::Internal(format!(
                        "duplicate device input binding '{input_name}'"
                    )));
                }
            }
            if let Some(output_name) = output_name {
                let output_vid = self
                    .graph
                    .outputs
                    .iter()
                    .copied()
                    .find(|&vid| {
                        self.graph.value(vid).name.as_deref() == Some(output_name.as_str())
                    })
                    .ok_or_else(|| {
                        SessionError::Internal(format!(
                            "device binding output not found: {output_name}"
                        ))
                    })?;
                if self.sequence_values.contains(&output_vid) {
                    return Err(SessionError::SequenceOp {
                        op: "<graph output binding>".to_string(),
                        reason: format!(
                            "graph output '{output_name}' is a Sequence value and cannot be bound to tensor device storage"
                        ),
                    });
                }
                if self.value_dtypes[&output_vid] != dtype {
                    return Err(SessionError::DtypeMismatch {
                        name: output_name.clone(),
                        expected: format!("{:?}", self.value_dtypes[&output_vid]),
                        got: format!("{dtype:?}"),
                    });
                }
                let value = ExternalValue {
                    dtype,
                    shape: binding.physical_shape().to_vec(),
                    accepts_subshape: bind_input
                        && binding.logical_shape() != binding.physical_shape(),
                    ptr,
                    len,
                    alignment,
                    device,
                };
                if external.outputs.insert(output_vid, value).is_some() {
                    return Err(SessionError::Internal(format!(
                        "duplicate device output binding '{output_name}'"
                    )));
                }
            }
        }
        Ok(external)
    }

    /// Execute the graph with `inputs` bound by name, plus an `outer_scope` of
    /// enclosing named values a nested control-flow subgraph body may capture.
    /// The top-level session `run` passes an empty scope; a control-flow body's
    /// child executor is invoked with its enclosing graph's live values so a
    /// deeply-nested body can still reach an outer capture (ONNX lexical scope).
    fn run_scoped(
        &mut self,
        inputs: &[(&str, &Tensor)],
        outer_scope: &HashMap<String, Tensor>,
        external: &ExternalBindings,
    ) -> Result<Vec<Option<SessionOutput>>> {
        match self.run_scoped_mode(inputs, outer_scope, external, RunMode::Eager)? {
            ScopedRunResult::Executed(outputs) => Ok(outputs),
            ScopedRunResult::NotCapturable(_) => unreachable!("eager runs are always executed"),
        }
    }

    fn run_scoped_mode(
        &mut self,
        inputs: &[(&str, &Tensor)],
        outer_scope: &HashMap<String, Tensor>,
        external: &ExternalBindings,
        mode: RunMode,
    ) -> Result<ScopedRunResult> {
        // Distinguish the outermost (top-level graph) run from nested
        // control-flow subgraph runs so the phase profiler can attribute
        // overhead to the right layer.
        thread_local! {
            static RUN_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
        }
        let depth = RUN_DEPTH.with(|d| {
            let cur = d.get();
            d.set(cur + 1);
            cur
        });
        struct DepthGuard;
        impl Drop for DepthGuard {
            fn drop(&mut self) {
                RUN_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
            }
        }
        let _depth_guard = DepthGuard;
        let nested = depth > 0;
        // Zero-copy view metadata is run-scoped: a value that aliased another's
        // buffer last run must not leak into this one (buffers may be resized).
        self.views.clear();
        self.pinned.clear();
        // Sequence values and their zero-copy element-backed tensors are equally
        // run-scoped (element Arcs from a prior run must not leak in).
        self.sequences.clear();
        self.seq_elem_values.clear();
        self.restore_shared_buffers()?;

        // --- Resolve shapes from the actual bound inputs --------------------
        let _phase_setup = phase_span!(if nested {
            "run_scoped.setup_total.child"
        } else {
            "run_scoped.setup_total.top"
        });
        let bindings = self.bind_symbols(inputs, external)?;

        for (name, _) in inputs {
            let vid = self.input_index[*name];
            if external.inputs.contains_key(&vid) {
                return Err(SessionError::Internal(format!(
                    "input '{name}' is bound both as a host tensor and a persistent device buffer"
                )));
            }
        }

        // Every required input must be supplied.
        let mut provided: HashSet<ValueId> = inputs
            .iter()
            .filter_map(|(name, _)| self.input_index.get(*name).copied())
            .collect();
        provided.extend(external.inputs.keys().copied());
        for &vid in &self.required_inputs {
            if !provided.contains(&vid) {
                let name = self
                    .graph
                    .value(vid)
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("value#{}", vid.0));
                return Err(SessionError::InputNotFound { name });
            }
        }

        // Substitute the bindings into every value → concrete shapes, then size
        // the run-scoped buffers from them (reused when unchanged). Values with a
        // data-dependent shape stay unresolved here and are filled in during the
        // execution loop, once their producing node's inputs are concrete.
        //
        // F5 Stage 1: on the top-level CPU eager decode path the steady-state
        // decode-plan memo replays the length-invariant partition of this map
        // instead of rebuilding it every token. It is a pure optimization of
        // `resolve_soft` (a function of `bindings` only, since on the eager path
        // no external/control-flow/warm seeding runs — that is Capture/Replay
        // only), gated OFF by default and asserted byte-identical under
        // `decode_memo_verify`.
        //
        // Persistent device-I/O bindings (the KV cache) are the NORMAL decode
        // case, not an exclusion: the real native decode path always carries them
        // (ext_in/ext_out non-empty), and `bind_symbols` already folds every
        // external *input* binding's shape into `bindings`, so the growing KV
        // length symbol L is captured by the replay guard exactly like any other
        // varying symbol. The memo additionally fingerprints the external binding
        // set (`decode_external_signature`) with L abstracted to its symbolic
        // identity, so pure-L growth replays while any structural change (binding
        // added/removed, role flip, dtype change) forces a rebuild.
        let decode_memo_eligible = self.decode_memo_enabled
            && mode == RunMode::Eager
            && !nested
            && self.ep.device_type() != DeviceType::Cuda;
        let mut resolved = {
            let _s = phase_span!("run_scoped.resolve_soft");
            if decode_memo_eligible {
                self.resolve_soft_decode_memo(&bindings, external)
            } else {
                // Observability: if the master switch is on but this step is
                // structurally ineligible (CUDA, nested, non-eager), count it so
                // an over-restrictive gate silently excluding the real decode path
                // is never shipped again (the F5 regression Ripley caught).
                if self.decode_memo_enabled && !nested {
                    self.decode_memo_ineligible_count += 1;
                }
                let mut resolved = self.resolve_soft(&bindings);
                if mode != RunMode::Eager {
                    // Persistent bindings seed the kernel-visible geometry selected by
                    // their input/output contracts. Seed only unresolved values:
                    // statically/symbolically resolved shapes remain authoritative.
                    external.seed_capture_shapes(&mut resolved);
                    // Control-flow outputs (e.g. LongRoPE cos/sin caches) are symbolic to
                    // shape inference but stable within a generation: seed their concrete
                    // prior-run shape so downstream capturable consumers fold into
                    // captured segments instead of forming per-consumer eager seams.
                    self.seed_control_flow_capture_shapes(&mut resolved);
                    // Steady-state decode ops (Cast/Mul/QMoE/ScatterElements …) whose
                    // output shape is data-dependent stay unresolved in `resolve_soft`
                    // and would each form an eager seam even though their kernels are
                    // already capture-safe. Seed their exact just-in-time shapes from
                    // the eager warmup — but only for the identical persistent-binding
                    // signature the warmup ran under, so a changed pointer/capacity
                    // withholds the seed instead of baking a stale shape.
                    self.seed_warm_decode_capture_shapes(&mut resolved, external);
                }
                resolved
            }
        };
        // --- F5 Stage 2: reinstate the cached invariant view/buffer plan --------
        // On a memo Replayed step whose per-source buffer identity still matches,
        // reinstate the zero-copy view aliases (lever 1) instead of clearing and
        // rebuilding them, mark the pure-view nodes for dispatch elision (lever 3),
        // and exclude the invariant partition from buffer sizing (lever 2). Taken
        // out of `self` for the duration so an errored step drops it (a stale alias
        // can never be reinstated into a later replay); restored on success.
        let mut stage2_plan: Option<DecodeViewPlan> = None;
        let mut stage2_candidate: Option<DecodeViewPlan> = None;
        let mut stage2_excluded: Option<HashSet<ValueId>> = None;
        if decode_memo_eligible
            && !self.decode_view_plan_disabled
            && self.decode_memo_last_action == DecodeMemoAction::Replayed
            && let Some(plan) = self.decode_view_plan.take()
        {
            if !plan.validated {
                // Candidate plan built on the preceding Rebuilt step: run this step
                // in full (no reinstate/elide) so every invariant view is freshly
                // built, then confirm two-real-step byte-identity below before it is
                // ever used to elide. This is the second-real-step confirmation.
                stage2_candidate = Some(plan);
            } else if self.stage2_buffer_sig_matches(&plan) {
                self.decode_view_plan_sig_mismatch_streak = 0;
                // Lever 1: reinstate the invariant zero-copy view aliases and
                // re-pin their source buffers (conservative liveness). Also
                // restore each elided output's resolved shape to the view's own
                // shape — the value the elided view node would have written into
                // `resolved` (which can differ from the pre-loop `resolve_soft`
                // shape Stage 1 restored, e.g. a Reshape with an inferred dim), so
                // downstream consumers read the identical geometry as a full step.
                for (vid, view) in &plan.retained_views {
                    self.views.insert(*vid, view.clone());
                    resolved.insert(*vid, view.shape.clone());
                }
                for &src in &plan.pinned_sources {
                    self.pinned.insert(src);
                }
                self.decode_views_reused_count += plan.retained_views.len() as u64;
                self.decode_dispatch_elided_count += plan.elided_nodes.len() as u64;
                // Lever 2: exclude the memo's proven-invariant partition from
                // per-step buffer sizing — those buffers were sized under the
                // rebuild and are byte-identical (guarded by the buffer-identity
                // signature above); the compute path still self-heals any output
                // whose length unexpectedly differs.
                if let Some(memo) = self.decode_memo.as_ref() {
                    stage2_excluded = Some(memo.invariant_shapes.keys().copied().collect());
                }
                stage2_plan = Some(plan);
            } else {
                // A source buffer moved/resized under a plan that classified it
                // invariant: retire the plan (dropped here) and run the full step.
                // After repeated mismatches the assumption is untrustworthy on this
                // model, so latch Stage 2 off for the session (defense-in-depth).
                self.decode_view_plan_sig_mismatch_streak += 1;
                if self.decode_view_plan_sig_mismatch_streak >= STAGE2_SIG_MISMATCH_LIMIT {
                    self.decode_view_plan_disabled = true;
                }
            }
        }
        let external_values = external
            .inputs
            .keys()
            .chain(external.outputs.keys())
            .copied()
            .collect::<HashSet<_>>();
        for &vid in &external_values {
            if let Some(old) = self.buffers.remove(&vid) {
                self.ep.deallocate(old)?;
            }
            self.shared_buffers.remove(&vid);
            self.buffer_shapes.remove(&vid);
        }
        {
            let _s = phase_span!("run_scoped.size_buffers");
            match &stage2_excluded {
                // Stage 2 (lever 2): size only the values outside the memo's
                // invariant partition (variant/JIT/external) — the invariant
                // buffers are reused untouched from the rebuild step.
                Some(invariant) => {
                    let mut excluded = external_values.clone();
                    excluded.extend(invariant.iter().copied());
                    self.size_buffers_excluding(&resolved, &excluded)?;
                }
                None => {
                    self.size_buffers_excluding(&resolved, &external_values)?;
                }
            }
        }

        // --- Bind input bytes into their (now correctly sized) buffers ------
        for (name, tensor) in inputs {
            let vid = self.input_index[*name];
            let buf = self
                .buffers
                .get_mut(&vid)
                .expect("input value has a buffer");
            self.ep.copy_from_host(tensor.as_bytes(), buf)?;
        }
        drop(_phase_setup);

        // --- Execute nodes ---------------------------------------------------
        // Iterate by index so a control-flow node can take `&mut self` (it must
        // build/reuse child executors) while an ordinary kernel node uses the
        // disjoint-field borrow split inside `exec_kernel_node`.
        match mode {
            RunMode::Eager => {
                let _s = phase_span!(if nested {
                    "run_scoped.plan_eager.child"
                } else {
                    "run_scoped.plan_eager.top"
                });
                // F5 Stage 2: elide the plan's pure-view nodes only in production.
                // Under `decode_memo_verify` (the R1 safety net) run every node so
                // the invariant views are freshly rebuilt, then assert each equals
                // the reinstated alias (bytes/shape/ptr) — proving reuse is exact.
                let verify_stage2 = self.decode_memo_verify && stage2_plan.is_some();
                let verify_snapshot: Option<Vec<(ValueId, ValueView)>> = if verify_stage2 {
                    stage2_plan.as_ref().map(|p| p.retained_views.clone())
                } else {
                    None
                };
                let elided = if verify_stage2 {
                    None
                } else {
                    stage2_plan.as_ref().map(|p| &p.elided_nodes)
                };
                self.run_plan_eager(&mut resolved, outer_scope, external, elided)?;
                if let (Some(snapshot), Some(plan)) = (&verify_snapshot, &stage2_plan) {
                    for (vid, cached) in snapshot {
                        let fresh = self.views.get(vid).unwrap_or_else(|| {
                            panic!(
                                "F5 Stage 2 verify: elided view value#{} was not rebuilt by a \
                                 full dispatch",
                                vid.0
                            )
                        });
                        assert!(
                            fresh.source == cached.source
                                && fresh.shape == cached.shape
                                && fresh.strides == cached.strides
                                && fresh.byte_offset == cached.byte_offset,
                            "F5 Stage 2 verify: cached view for value#{} ({cached:?}) diverged \
                             from a freshly built one ({fresh:?}) — invariant view reuse is unsound",
                            vid.0
                        );
                    }
                    assert!(
                        self.stage2_buffer_sig_matches(plan),
                        "F5 Stage 2 verify: a cached view source buffer moved during the step"
                    );
                }
                // F5 Stage 2 plan lifecycle: rebuild the cached view plan at the
                // successful end of a memo Rebuilt step (the plan was invalidated
                // at the top of the rebuild path, so a mid-step error leaves it
                // `None`); restore the in-flight plan after a successful replay.
                if decode_memo_eligible {
                    match self.decode_memo_last_action {
                        DecodeMemoAction::Rebuilt => {
                            self.decode_view_plan = self.build_decode_view_plan();
                        }
                        DecodeMemoAction::Replayed => {
                            if let Some(cand) = stage2_candidate.take() {
                                // This replay ran full dispatch as the candidate's
                                // second-real-step confirmation: keep only the nodes
                                // whose view is byte-identical to the built one, and
                                // promote to validated (or drop if none survive).
                                self.decode_view_plan = self.validate_decode_view_plan(cand);
                            } else if let Some(plan) = stage2_plan.take() {
                                self.decode_view_plan = Some(plan);
                            }
                        }
                        _ => {}
                    }
                }
                // Snapshot the exact just-in-time shapes this warm run resolved,
                // together with the persistent-binding signature they were
                // derived under. Capture-mode seeding replays these shapes only
                // when a later step presents this exact signature (pointer- and
                // capacity-stable), so a changed binding forces recapture, never
                // a stale-shape replay. Skipped on the memo-eligible CPU decode
                // path: that path never captures (CPU EP), so cloning the whole
                // ~600-entry resolved map every token would be pure waste and
                // would defeat the memo's allocation amortization.
                if !decode_memo_eligible {
                    self.capture_warm_shapes = resolved.clone();
                    self.capture_warm_signature = Some(external.capture_signature());
                }
            }
            RunMode::Capture => {
                // A fresh capture may have resized/reallocated the `If` output
                // buffers, so force every `If` to actually execute its branch
                // this run (repopulating those buffers) rather than trusting the
                // steady-decode memo. Cleared before segmentation so the branch
                // runs as a normal eager seam during the capture pass.
                self.if_last_predicate.clear();
                // Partition the claimed subgraph into maximal capturable segments
                // separated by non-capturable seam nodes. Only a graph-level hard
                // decline (e.g. no persistent output binding, or nothing
                // capturable at all) falls back to a fully eager run.
                //
                // Warm-decode shape seeding can admit a node whose kernel wrongly
                // advertises capture support but aborts device-graph recording
                // (e.g. a stream synchronize, which CUDA rejects mid-capture).
                // A single such kernel aborts the whole segmented capture. Rather
                // than regress to a fully eager step, quarantine the offending
                // op-type to a forced eager seam and re-plan/re-record: the
                // genuinely-capturable ops still fold while the mislabeled kernel
                // stays eager. Re-recording a fixed-capacity decode step is
                // idempotent (same position/token → same values into the same
                // slots), so retrying is safe. Bounded by the node count.
                let max_capture_attempts = self.plan.len() + 1;
                let schedule = 'capture: loop {
                    let schedule = match self.plan_capture_segments(&resolved, external) {
                        Ok(schedule) => schedule,
                        Err(report) => return Ok(ScopedRunResult::NotCapturable(report)),
                    };
                    self.last_capture_failed_node = None;
                    match self.run_plan_segmented(
                        &schedule,
                        RunMode::Capture,
                        &mut resolved,
                        outer_scope,
                        external,
                    ) {
                        Ok(_) => break 'capture schedule,
                        Err(error) => {
                            let _ = self.ep.reset_device_graph();
                            // Quarantine the op-type that aborted recording and
                            // retry, unless we already quarantined it (no
                            // progress), hit the attempt bound, or cannot
                            // attribute the failure to a node.
                            let quarantined =
                                self.last_capture_failed_node.take().and_then(|node_id| {
                                    let node = self.graph.node(node_id);
                                    let key = (canonical_domain(node), node.op_type.clone());
                                    self.capture_quarantine_ops.insert(key).then_some(())
                                });
                            if quarantined.is_some()
                                && self.capture_quarantine_ops.len() < max_capture_attempts
                            {
                                // Re-plan with the offending op-type forced eager.
                                self.if_last_predicate.clear();
                                continue 'capture;
                            }
                            self.capture_schedule = None;
                            self.capture_segmentation.clear();
                            self.capture_cf_shapes.clear();
                            self.capture_warm_seeded.clear();
                            return Ok(ScopedRunResult::NotCapturable(CaptureDeclineReport::one(
                                CaptureDecline::graph(format!(
                                    "segmented CUDA graph capture failed: {error}"
                                )),
                            )));
                        }
                    }
                };
                // A warm-seeded shape that the capture pass re-resolved to a
                // different value means the seed was stale for this step (a
                // genuinely per-step-varying interior extent). The recorded
                // graph would replay that shape unconditionally, so retire it
                // and decline: the caller re-warms and either re-captures (if
                // the shape restabilizes) or keeps this op eager. This upholds
                // "recapture when any shape changes; never replay a stale graph."
                if let Some((vid, seeded)) = self
                    .capture_warm_seeded
                    .iter()
                    .find(|(vid, seeded)| resolved.get(vid) != Some(*seeded))
                    .map(|(vid, seeded)| (*vid, seeded.clone()))
                {
                    let current = resolved.get(&vid).cloned();
                    let _ = self.ep.reset_device_graph();
                    self.capture_schedule = None;
                    self.capture_segmentation.clear();
                    self.capture_cf_shapes.clear();
                    self.capture_warm_seeded.clear();
                    return Ok(ScopedRunResult::NotCapturable(CaptureDeclineReport::one(
                        CaptureDecline::graph(format!(
                            "warm decode shape seed for value#{} ({seeded:?}) diverged from the \
                             captured shape ({current:?}); recapturing",
                            vid.0
                        )),
                    )));
                }
                // Snapshot the concrete control-flow output shapes this capture
                // assumed so a later replay can detect a branch flip that changes
                // them and retire the now-stale installed graph.
                self.capture_cf_shapes = self
                    .control_flow_output_values
                    .iter()
                    .filter_map(|vid| resolved.get(vid).map(|shape| (*vid, shape.clone())))
                    .collect();
                self.capture_segmentation = schedule.boundaries.clone();
                if capture_segmentation_logging_enabled() {
                    log_capture_segmentation(&schedule);
                }
                self.capture_schedule = Some(schedule);
            }
            RunMode::Replay => {
                // Move the schedule out so the segmented runner can take `&mut
                // self`; restore it afterwards for the next step's replay.
                let Some(schedule) = self.capture_schedule.take() else {
                    return Ok(ScopedRunResult::NotCapturable(CaptureDeclineReport::one(
                        CaptureDecline::graph(
                            "segmented device graph replay requested without a capture schedule",
                        ),
                    )));
                };
                let still_valid = self.run_plan_segmented(
                    &schedule,
                    RunMode::Replay,
                    &mut resolved,
                    outer_scope,
                    external,
                )?;
                if still_valid {
                    self.capture_schedule = Some(schedule);
                } else {
                    // A control-flow branch flip changed a seeded output shape:
                    // the remaining plan already ran eagerly this step (correct
                    // token), but the installed segments are stale. Retire the
                    // device graph so the caller re-warms and re-captures for the
                    // new branch. `capture_schedule` stays `None`.
                    self.capture_segmentation.clear();
                    self.capture_cf_shapes.clear();
                    self.device_graph_signature = None;
                    self.ep.reset_device_graph()?;
                }
            }
        }

        // --- Collect graph outputs into owned tensors -----------------------
        // A view output (a layout op whose result aliases an input buffer) is
        // materialized to contiguous owned bytes here — external consumers and
        // the Python/DLPack boundary expect contiguous tensors.
        let _phase_collect = phase_span!(if nested {
            "run_scoped.collect_outputs.child"
        } else {
            "run_scoped.collect_outputs.top"
        });
        let mut results = Vec::with_capacity(self.graph.outputs.len());
        let mut host_output_bytes = 0usize;
        let output_vids: Vec<ValueId> = self.graph.outputs.clone();
        for vid in output_vids {
            if external.outputs.contains_key(&vid) {
                results.push(None);
                continue;
            }
            if self.sequence_values.contains(&vid) {
                let sequence = self.sequences.get(&vid).cloned().ok_or_else(|| {
                    SessionError::Internal(format!(
                        "sequence graph output value#{} has no live runtime value",
                        vid.0
                    ))
                })?;
                results.push(Some(SessionOutput::Sequence(sequence)));
                continue;
            }

            let dtype = self.value_dtypes[&vid];
            let shape = resolved[&vid].clone();
            // Top-level outputs: hand the produced host buffer to the caller
            // zero-copy when safe (the KV-cache round-trip the decode hot path
            // otherwise pays every step). Child (subgraph) outputs are copied
            // back into the parent scope, so keep them on the copy path.
            if !nested && let Some(tensor) = self.try_move_host_output(vid, &shape, dtype)? {
                results.push(Some(SessionOutput::Tensor(tensor)));
                continue;
            }
            let bytes = self.contiguous_bytes(vid, &shape, dtype)?;
            host_output_bytes += bytes.len();
            results.push(Some(SessionOutput::Tensor(Tensor::from_raw(
                dtype, shape, &bytes,
            )?)));
        }
        // Attribution aid: at the top level, the number of graph-output bytes
        // materialized to host each run is the per-step cost of *not* keeping
        // outputs (e.g. a growing KV cache) in persistent device/host bindings.
        // Recorded as a counter (bytes as the "nanos" field) so the phase table
        // exposes total and per-call host-output traffic without extra logging.
        if !nested {
            phase_profile::record("collect_outputs.top_host_bytes", host_output_bytes as u128);
        }
        // F5 Stage 1: hand the just-used shape map (now including this step's
        // data-dependent JIT tail) back to the persistent working buffer so the
        // next replay step can take it in place — retaining every invariant
        // `Vec`'s allocation — rather than allocating a fresh map/`Vec`s per
        // token. Only on the memo-eligible CPU decode path; otherwise the buffer
        // stays untouched (and empty).
        if decode_memo_eligible {
            self.decode_memo_resolved = std::mem::take(&mut resolved);
        }
        Ok(ScopedRunResult::Executed(results))
    }

    /// Classify why one plan node cannot be recorded into a device graph, or
    /// `None` when it is capturable. Mirrors the per-node predicates the
    /// all-or-nothing audit used, but returns the reason instead of aborting so
    /// the caller can form segments around each non-capturable seam node.
    /// Seed the concrete shapes of control-flow (`If`/`Loop`/`Scan`) outputs from
    /// the previous run's buffer allocation so downstream capturable kernels that
    /// read them (e.g. GroupQueryAttention reading LongRoPE's `If`-selected
    /// cos/sin caches) resolve their input shapes and fold into captured segments
    /// instead of each forming an eager seam.
    ///
    /// ONNX shape inference cannot statically resolve a control-flow output whose
    /// branches declare different shapes, so it stays symbolic. Within a decode
    /// generation the selected branch — and thus the concrete output shape — is
    /// stable across steps, so the prior run's shape is authoritative for capture
    /// planning. A branch flip changes the shape and is detected on replay
    /// ([`Self::control_flow_seam_invalidated`]), which retires the captured graph
    /// for re-capture, so seeding never risks replaying against a stale shape.
    ///
    /// Only genuinely-unresolved outputs are seeded: a statically/symbolically
    /// resolved shape stays authoritative, matching [`ExternalBindings::seed_capture_shapes`].
    fn seed_control_flow_capture_shapes(&self, resolved: &mut HashMap<ValueId, Vec<usize>>) {
        for &vid in &self.control_flow_output_values {
            if resolved.contains_key(&vid) {
                continue;
            }
            if let Some(shape) = self.buffer_shapes.get(&vid) {
                resolved.insert(vid, shape.clone());
            }
        }
    }

    /// Seed every still-unresolved value's shape from the most recent eager
    /// warmup's fully-resolved shape map ([`Self::capture_warm_shapes`]) so the
    /// decode ops whose output shape is data-dependent (omitted by
    /// [`Self::resolve_soft`]) — Cast/Mul/QMoE/ScatterElements downstream of a
    /// data-dependent extent — resolve their input/output shapes and fold into
    /// captured segments instead of each forming an eager seam. This generalizes
    /// the control-flow seeding above from `If`/`Loop`/`Scan` outputs to any
    /// warmed data-dependent value.
    ///
    /// Correctness rests entirely on the *decode binding signature*: the warm
    /// shapes are trusted only when the current persistent-binding signature is
    /// byte-for-byte identical to the one the warmup ran under
    /// ([`ExternalBindings::capture_signature`]). A changed pointer or capacity
    /// withholds every seed (those values stay unresolved → eager seams), and the
    /// top-level replay guard ([`Self::replay_device_graph`]) independently
    /// retires the installed graph on any binding change. Values resolvable from
    /// the current symbol bindings are never overridden — only genuinely
    /// unresolved (value-dependent) extents are seeded — and the capture pass
    /// re-resolves each seeded shape, retiring the graph if any diverged, so a
    /// per-step-varying extent can never be replayed against a stale shape.
    /// Persistent bindings and initializers are excluded (seeded/owned elsewhere).
    fn seed_warm_decode_capture_shapes(
        &mut self,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        external: &ExternalBindings,
    ) {
        self.capture_warm_seeded.clear();
        // Trust the warm just-in-time shapes only for the exact signature they
        // were derived under; otherwise leave values unresolved (eager seams).
        if self.capture_warm_signature.as_ref() != Some(&external.capture_signature()) {
            return;
        }
        let external_values: HashSet<ValueId> = external
            .inputs
            .keys()
            .chain(external.outputs.keys())
            .copied()
            .collect();
        let warm: Vec<(ValueId, Vec<usize>)> = self
            .capture_warm_shapes
            .iter()
            .map(|(&vid, shape)| (vid, shape.clone()))
            .collect();
        for (vid, shape) in warm {
            if resolved.contains_key(&vid)
                || external_values.contains(&vid)
                || self.graph.initializers.contains_key(&vid)
                || self.sequence_values.contains(&vid)
            {
                continue;
            }
            self.capture_warm_seeded.insert(vid, shape.clone());
            resolved.insert(vid, shape);
        }
    }

    /// Whether the control-flow seam node at plan index `pi` produced a different
    /// output shape than the most recent capture assumed. A change means a branch
    /// flip (e.g. LongRoPE short↔long at the context threshold) reallocated an
    /// output buffer a later captured segment reads, so that segment's baked
    /// device pointer is now stale and the installed graph must be retired.
    fn control_flow_seam_invalidated(
        &self,
        pi: usize,
        resolved: &HashMap<ValueId, Vec<usize>>,
    ) -> bool {
        let node = self.graph.node(self.plan[pi].node_id);
        if !is_control_flow_op(&node.op_type, &node.domain) {
            return false;
        }
        self.plan[pi].outputs.iter().any(|out| {
            match (self.capture_cf_shapes.get(out), resolved.get(out)) {
                (Some(captured), Some(current)) => captured != current,
                (Some(_), None) => true,
                _ => false,
            }
        })
    }

    fn node_capture_reason(
        &self,
        plan: &NodePlan,
        resolved: &HashMap<ValueId, Vec<usize>>,
    ) -> Option<CaptureDecline> {
        let node = self.graph.node(plan.node_id);
        // A kernel that aborted device-graph recording on a prior capture pass is
        // quarantined by op-type: force it (and every sibling of the same op-type)
        // to an eager seam so warm-decode shape seeding can still fold the rest of
        // the graph instead of one mislabeled kernel aborting the whole capture.
        if self
            .capture_quarantine_ops
            .contains(&(canonical_domain(node), node.op_type.clone()))
        {
            return Some(CaptureDecline::node(
                plan.node_id,
                node,
                SeamReason::CaptureRecordingFailed,
                "kernel aborted device-graph recording on a prior capture pass; \
                 quarantined to an eager seam",
            ));
        }
        let outputs_resolved = plan
            .outputs
            .iter()
            .all(|output| resolved.contains_key(output));
        let inputs_resolved = plan.inputs.iter().all(|input| match input {
            Some(value) => resolved.contains_key(value),
            None => true,
        });
        if let Some(decline) = self.ep.plan_capture_region(
            node,
            CaptureRegionShapeStatus {
                inputs_resolved,
                outputs_resolved,
            },
        ) {
            return Some(structural_capture_decline(plan.node_id, node, decline));
        }
        assert!(
            inputs_resolved && outputs_resolved,
            "EP capture-region policy admitted a node with unresolved shapes"
        );
        let input_shapes = plan
            .inputs
            .iter()
            .map(|input| {
                input.map_or_else(Vec::new, |value| {
                    resolved
                        .get(&value)
                        .cloned()
                        .expect("resolved input shape checked above")
                })
            })
            .collect();
        let key = KernelKey {
            node: plan.node_id.0,
            shapes: input_shapes,
        };
        let Some(kernel) = self.cache.entries.get(&key) else {
            return Some(CaptureDecline::node(
                plan.node_id,
                node,
                SeamReason::KernelNotWarmed,
                "kernel has not been warmed for the requested capture shape",
            ));
        };
        kernel_capture_decline(plan.node_id, node, kernel.as_ref())
    }

    /// Partition the plan into maximal contiguous captured segments separated by
    /// eager (non-capturable) seam nodes.
    ///
    /// The CUDA EP keeps ownership of the whole claimed subgraph: this never
    /// declines a run because *some* node is non-capturable. It only returns a
    /// hard [`CaptureDeclineReport`] for a graph-level precondition (outputs must
    /// land in persistent device bindings) or when *nothing* is capturable — in
    /// which case a device graph adds no value and the caller runs fully eager
    /// (still on the CUDA EP, so placement is unchanged).
    fn plan_capture_segments(
        &self,
        resolved: &HashMap<ValueId, Vec<usize>>,
        external: &ExternalBindings,
    ) -> std::result::Result<CaptureSchedule, CaptureDeclineReport> {
        if self
            .graph
            .outputs
            .iter()
            .any(|output| !external.outputs.contains_key(output))
        {
            return Err(CaptureDeclineReport::one(CaptureDecline::graph(
                "every graph output must use a persistent device binding during capture",
            )));
        }

        let declines: Vec<Option<CaptureDecline>> = self
            .plan
            .iter()
            .map(|plan| self.node_capture_reason(plan, resolved))
            .collect();

        let mut segments: Vec<ScheduledSegment> = Vec::new();
        let mut boundaries: Vec<CaptureDecline> = Vec::new();
        let mut next_graph_index = 0usize;
        let mut pi = 0usize;
        while pi < declines.len() {
            let captured = declines[pi].is_none();
            let start = pi;
            while pi < declines.len() && declines[pi].is_none() == captured {
                if let Some(decline) = &declines[pi] {
                    boundaries.push(decline.clone());
                }
                pi += 1;
            }
            let graph_index = if captured {
                let index = next_graph_index;
                next_graph_index += 1;
                index
            } else {
                0
            };
            segments.push(ScheduledSegment {
                start,
                end: pi,
                captured,
                graph_index,
            });
        }

        if next_graph_index == 0 {
            return Err(CaptureDeclineReport {
                entries: boundaries,
            });
        }

        Ok(CaptureSchedule {
            segments,
            boundaries,
        })
    }

    /// Gather the warmed, capturable kernels backing one captured segment, in
    /// plan order, ready to hand to the EP's `begin_device_graph_capture` audit.
    fn collect_segment_kernels(
        &self,
        seg: &ScheduledSegment,
        resolved: &HashMap<ValueId, Vec<usize>>,
    ) -> Result<Vec<&dyn onnx_runtime_ep_api::Kernel>> {
        let mut kernels = Vec::with_capacity(seg.end - seg.start);
        for pi in seg.start..seg.end {
            let plan = &self.plan[pi];
            let input_shapes = plan
                .inputs
                .iter()
                .map(|input| {
                    input
                        .map(|value| resolved.get(&value).cloned())
                        .unwrap_or(Some(Vec::new()))
                })
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| {
                    SessionError::Internal(format!(
                        "segment kernel node {} lost its resolved input shape before capture",
                        plan.node_id.0
                    ))
                })?;
            let key = KernelKey {
                node: plan.node_id.0,
                shapes: input_shapes,
            };
            let kernel = self.cache.entries.get(&key).ok_or_else(|| {
                SessionError::Internal(format!(
                    "segment kernel node {} was not warmed before capture",
                    plan.node_id.0
                ))
            })?;
            kernels.push(kernel.as_ref());
        }
        Ok(kernels)
    }

    /// Dispatch one plan node to its execution path (control-flow, sequence, or
    /// leaf kernel). Shared by the eager loop and the segmented runner.
    ///
    /// When tracing is enabled, opens one span per op so the dispatched kernel
    /// can attach kernel-variant and capture-rejection reasons via
    /// [`annotate_current_span_with`]; `capture` records the node's device-graph
    /// disposition onto that span. When tracing is disabled this costs a single
    /// relaxed atomic load and never allocates.
    fn exec_plan_node(
        &mut self,
        pi: usize,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        outer_scope: &HashMap<String, Tensor>,
        external: &ExternalBindings,
        capture: OpCaptureTrace<'_>,
    ) -> Result<()> {
        // Dispatch by op-type/domain borrowed straight from the node, so a
        // steady-state decode step compares `&str`s and never clones the
        // op-type/domain `String`s per node. The immutable borrow of
        // `self.graph` is confined to this block and dropped before the
        // `&mut self` dispatch below; the span guard it yields owns its name
        // (and a cheap `Arc`-clone of the trace context), so it borrows nothing
        // from `self` and can stay live across the dispatch.
        let (is_control_flow, is_sequence, _span) = {
            let node = self.graph.node(self.plan[pi].node_id);
            let is_control_flow = is_control_flow_op(&node.op_type, &node.domain);
            let is_sequence = is_sequence_op(&node.op_type, &node.domain);
            // Open the span only when tracing is live so an untraced decode step
            // never allocates a span name or touches the thread-local span stack.
            // Everything that clones a node field lives inside this closure for
            // the same reason: an untraced step must not pay for identity it is
            // never going to record.
            let span = self.trace.is_enabled().then(|| {
                // Every op span comes from this one line, so its source location
                // would be the same string on all of them; the node args below
                // identify each span far better. Keeping it cost 22% of a trace.
                let span = self.trace.span(node.op_type.clone(), "op").without_source();
                // Identify *which* node this is. The span name stays the bare op
                // type so Perfetto still aggregates all `MatMul`s together; the
                // identity rides along as args. A model has hundreds of
                // same-typed nodes, and without this a slow one cannot be told
                // from a fast one.
                //
                // Device is stamped here rather than by each kernel. Kernels have
                // to opt in to annotating themselves and in practice most never
                // do -- the CPU provider annotates 11 of its 122 kernels, the
                // CUDA provider annotated none -- so a per-kernel convention
                // leaves most of a trace unlabelled. The node's placement is
                // known here for every node on every provider, which makes the
                // coverage structural instead of something each kernel must
                // remember.
                annotate_current_span_with(|| {
                    let mut args = Args::new().with("node_id", node.id.0 as u64);
                    if !node.name.is_empty() {
                        args = args.with("node", node.name.clone());
                    }
                    // Only non-default domains are worth the bytes: `Attention`
                    // and `MatMulNBits` exist in both the default and
                    // `com.microsoft` domains, so the op type alone is ambiguous
                    // for custom ops.
                    if !node.domain.is_empty() {
                        args = args.with("domain", node.domain.clone());
                    }
                    if let Some(device) = node.device {
                        args = args.with(
                            onnx_runtime_ep_api::ARG_DEVICE,
                            device.device_type.trace_name().into_owned(),
                        );
                    }
                    args
                });
                // Span is now active on this thread; stamp the capture disposition
                // (and let the kernel below stamp its selected variant).
                capture.annotate();
                span
            });
            (is_control_flow, is_sequence, span)
        };
        if is_control_flow {
            self.exec_control_flow(pi, resolved, outer_scope)
        } else if is_sequence {
            self.exec_sequence_node(pi, resolved, external)
        } else {
            self.exec_kernel_node(pi, resolved, external)
        }
    }

    /// Execute every plan node eagerly on the stream (no capture).
    ///
    /// F5 Stage 2: when `elided` is `Some`, the plan-node indices it contains are
    /// pure invariant view nodes whose zero-copy output aliases have already been
    /// reinstated into `self.views` for this step, so their re-dispatch is skipped.
    /// The set is empty (or `None`) on every non-Stage-2 run, so ordinary steps
    /// pay only one `HashSet::is_empty`/`contains` check per node.
    fn run_plan_eager(
        &mut self,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        outer_scope: &HashMap<String, Tensor>,
        external: &ExternalBindings,
        elided: Option<&HashSet<usize>>,
    ) -> Result<()> {
        let elided = elided.filter(|set| !set.is_empty());
        if profile_ops_enabled() {
            let run_start = Instant::now();
            let mut timings: HashMap<String, (Duration, usize)> = HashMap::new();
            for pi in 0..self.plan.len() {
                if elided.is_some_and(|set| set.contains(&pi)) {
                    continue;
                }
                let op_type = self.graph.node(self.plan[pi].node_id).op_type.clone();
                let start = Instant::now();
                let result =
                    self.exec_plan_node(pi, resolved, outer_scope, external, OpCaptureTrace::Eager);
                let elapsed = start.elapsed();
                let entry = timings.entry(op_type).or_insert((Duration::ZERO, 0));
                entry.0 += elapsed;
                entry.1 += 1;
                result?;
            }
            print_op_profile(run_start.elapsed(), timings);
        } else {
            for pi in 0..self.plan.len() {
                if elided.is_some_and(|set| set.contains(&pi)) {
                    continue;
                }
                self.exec_plan_node(pi, resolved, outer_scope, external, OpCaptureTrace::Eager)?;
            }
        }
        Ok(())
    }

    /// Run the plan against a [`CaptureSchedule`], interleaving captured device
    /// graphs with eager seam nodes.
    ///
    /// * [`RunMode::Capture`] records each captured segment into its own device
    ///   graph, then immediately replays it so the following eager seam node
    ///   reads real bytes from the stable seam buffers. Eager seam nodes execute
    ///   normally on the stream (not recorded).
    /// * [`RunMode::Replay`] launches each captured segment's installed graph in
    ///   order and re-runs only the eager seam nodes.
    ///
    /// Seam correctness relies on the executor's per-value buffer reuse: for a
    /// fixed decode shape, intermediate buffers keep the same device address
    /// every step, so a captured segment and the eager node on either side of a
    /// seam always read and write the same stable buffers.
    fn run_plan_segmented(
        &mut self,
        schedule: &CaptureSchedule,
        mode: RunMode,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        outer_scope: &HashMap<String, Tensor>,
        external: &ExternalBindings,
    ) -> Result<bool> {
        let ep = Arc::clone(&self.ep);
        // Set once a control-flow branch flip retires the installed graph mid
        // replay: every remaining node then runs eagerly (its captured segment's
        // baked device pointers are stale) so the step still produces a correct
        // token. Only ever set in `RunMode::Replay`.
        let mut invalidated = false;
        for seg in &schedule.segments {
            if invalidated {
                // Graph retired earlier this step: run this segment's nodes
                // eagerly instead of replaying a stale installed graph.
                for pi in seg.start..seg.end {
                    self.exec_plan_node(
                        pi,
                        resolved,
                        outer_scope,
                        external,
                        OpCaptureTrace::Eager,
                    )?;
                }
                continue;
            }
            if seg.captured {
                match mode {
                    RunMode::Capture => {
                        {
                            let kernels = self.collect_segment_kernels(seg, resolved)?;
                            ep.begin_device_graph_capture(&kernels)?;
                        }
                        // Any early return (`?`) while recording this segment
                        // must end the stream capture before it propagates —
                        // otherwise the stream stays wedged in capture mode and
                        // the caller's `reset_device_graph()` is a no-op (reset
                        // is rejected while capturing). The guard aborts the
                        // capture on drop; `disarm()` hands off to the normal
                        // `end_device_graph_capture()` on the success path.
                        let mut capture_guard = SegmentCaptureGuard::arm(ep.as_ref());
                        for pi in seg.start..seg.end {
                            let node_id = self.plan[pi].node_id;
                            if let Err(error) = self.exec_plan_node(
                                pi,
                                resolved,
                                outer_scope,
                                external,
                                OpCaptureTrace::Captured,
                            ) {
                                // Record which node aborted recording so the
                                // capture retry loop can quarantine its op-type.
                                // `capture_guard` drops here, ending the wedged
                                // stream capture before the error propagates.
                                self.last_capture_failed_node = Some(node_id);
                                return Err(error);
                            }
                        }
                        capture_guard.disarm();
                        ep.end_device_graph_capture()?;
                        ep.replay_device_graph_segment(seg.graph_index)?;
                    }
                    RunMode::Replay => {
                        ep.replay_device_graph_segment(seg.graph_index)?;
                    }
                    RunMode::Eager => {
                        unreachable!("eager runs never build a segment schedule")
                    }
                }
            } else {
                for pi in seg.start..seg.end {
                    // Seam node: eager because some kernel/predicate declined
                    // capture. Surface that reason on the node's span.
                    let node_id = self.plan[pi].node_id.0;
                    let reason = schedule
                        .boundaries
                        .iter()
                        .find(|decline| decline.node_id == Some(node_id))
                        .map(|decline| decline.reason.as_str())
                        .unwrap_or("non-capturable seam node (no recorded reason)");
                    self.exec_plan_node(
                        pi,
                        resolved,
                        outer_scope,
                        external,
                        OpCaptureTrace::Rejected(reason),
                    )?;
                    // A control-flow seam (e.g. LongRoPE's `If`) that now selects
                    // a different-shaped branch than capture assumed reallocated
                    // an output a later captured segment reads: retire the graph
                    // and finish this step eagerly.
                    if mode == RunMode::Replay && self.control_flow_seam_invalidated(pi, resolved) {
                        invalidated = true;
                    }
                }
            }
        }
        Ok(!invalidated)
    }

    /// Refill [`Self::scratch_input_shapes`] with the resolved shapes of plan
    /// node `pi`'s inputs, so the dispatch path reads shapes from a reused buffer
    /// instead of allocating a fresh `Vec<Vec<usize>>` per node per token.
    ///
    /// The scratch is truncated to the node's arity and each inner `Vec` is
    /// cleared and refilled in place (retaining its heap capacity), so a
    /// steady-state decode step — a fixed sequence of fixed-arity nodes — does
    /// zero shape-vector allocation after warmup. An omitted optional input
    /// (`None` slot) yields an empty inner shape, exactly as the previous
    /// `.unwrap_or_default()` collect did. `self.plan` and
    /// `self.scratch_input_shapes` are disjoint fields, so the shared read of the
    /// former coexists with the `&mut` refill of the latter.
    fn refill_input_shapes(&mut self, pi: usize, resolved: &HashMap<ValueId, Vec<usize>>) {
        let inputs = &self.plan[pi].inputs;
        let scratch = &mut self.scratch_input_shapes;
        scratch.truncate(inputs.len());
        for (i, slot) in inputs.iter().enumerate() {
            if i < scratch.len() {
                scratch[i].clear();
            } else {
                scratch.push(Vec::new());
            }
            if let Some(vid) = slot {
                scratch[i].extend_from_slice(&resolved[vid]);
            }
        }
    }

    /// Execute one ordinary (leaf-kernel) plan node: resolve any data-dependent
    /// output shapes, size buffers, build the input/output views (with Holden's
    /// bounds gate), resolve the shape-keyed kernel, and dispatch it.
    fn exec_kernel_node(
        &mut self,
        pi: usize,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        external: &ExternalBindings,
    ) -> Result<()> {
        // Whole-node dispatch span: its lifetime minus `exec_kernel.compute` is
        // the serial per-node dispatch glue (shape resolve, input/output view
        // build, kernel-cache lookup) the F5 Stage 3 record would elide.
        let _node_span = phase_span!("exec_kernel.node");
        // Borrow the plan facts in place rather than cloning them per node per
        // token: `self.plan` is a distinct field from the buffer/view/cache
        // fields mutated below, so these shared borrows coexist with the
        // disjoint `&mut self.<field>` borrows the compute path takes (the
        // dispatch never goes through a `&mut self` method while they are held).
        let node_id = self.plan[pi].node_id;
        // Refill the reusable per-executor input-shape scratch first (before the
        // shared borrows below), so a steady-state decode step allocates no
        // fresh `Vec<Vec<usize>>` for shape lookup — see `refill_input_shapes`.
        self.refill_input_shapes(pi, resolved);
        let inputs = &self.plan[pi].inputs;
        let outputs = &self.plan[pi].outputs;
        let input_dtypes = &self.plan[pi].input_dtypes;
        let output_dtypes = &self.plan[pi].output_dtypes;
        let input_shapes = &self.scratch_input_shapes;

        let node = self.graph.node(node_id);
        if let Some(output_shape) = runtime_elementwise_output_shape(node, input_shapes) {
            let output_shape = output_shape.map_err(|_| {
                let node_name = if node.name.is_empty() {
                    format!("<unnamed node #{}>", node_id.0)
                } else {
                    format!("{:?}", node.name)
                };
                SessionError::RuntimeBroadcastIncompatible {
                    node: node_name,
                    domain: canonical_domain(node),
                    op_type: node.op_type.clone(),
                    input_shapes: input_shapes.to_vec(),
                }
            })?;
            if outputs.len() != 1 {
                return Err(SessionError::OutputShapeCountMismatch {
                    op: node.op_type.clone(),
                    expected: outputs.len(),
                    got: 1,
                });
            }
            resolved.insert(outputs[0], output_shape);
        }

        // Data-dependent shapes: if any output's shape is still unresolved,
        // compute it now from the concrete input shapes + the runtime *values*
        // of this node's integer inputs. Buffers are NOT sized here — a view
        // output needs none, and the compute path sizes them just below.
        if outputs.iter().any(|v| !resolved.contains_key(v)) {
            let opset = effective_opset(&self.graph, node);
            let input_values: Vec<Option<Vec<i64>>> = inputs
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    v.and_then(|vid| self.shape_input_i64(vid, &input_shapes[i], input_dtypes[i]))
                })
                .collect();
            // Only materialize a *float* input value for the specific inputs an
            // op actually reads as float shape data (today: `Resize` scales).
            // Downloading any other float input here would both waste a host copy
            // and break the "reject an invalid shape input before any host
            // materialization" contract — e.g. a data tensor feeding an
            // `Unsqueeze` whose integer axes is invalid must never be copied to
            // host just to reach the unresolved-shape rejection.
            let input_float_values: Vec<Option<Vec<f64>>> = inputs
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    if !reads_float_shape_input(node, i, opset) {
                        return None;
                    }
                    v.and_then(|vid| {
                        if node.is_default_domain() && node.op_type == "NonMaxSuppression" {
                            self.nms_input_f64(vid, &input_shapes[i], input_dtypes[i])
                        } else {
                            self.shape_input_f64(vid, &input_shapes[i], input_dtypes[i])
                        }
                    })
                })
                .collect();
            let out_shapes = dynamic_output_shapes(
                node,
                input_shapes,
                input_dtypes,
                &input_values,
                &input_float_values,
                opset,
            )
            .ok_or_else(|| {
                let vid = outputs
                    .iter()
                    .find(|v| !resolved.contains_key(v))
                    .copied()
                    .unwrap_or(outputs[0]);
                let value = self.graph.value(vid);
                SessionError::UnresolvedShape {
                    value: value
                        .name
                        .clone()
                        .unwrap_or_else(|| format!("value#{}", vid.0)),
                    op: node.op_type.clone(),
                }
            })?;
            if out_shapes.len() != outputs.len() {
                return Err(SessionError::OutputShapeCountMismatch {
                    op: self.graph.node(node_id).op_type.clone(),
                    expected: outputs.len(),
                    got: out_shapes.len(),
                });
            }
            for (oi, &ovid) in outputs.iter().enumerate() {
                resolved.insert(ovid, out_shapes[oi].clone());
            }
        }
        let mut output_shapes: Vec<Vec<usize>> =
            outputs.iter().map(|v| resolved[v].clone()).collect();
        // Fixed-capacity KV for the default-domain Attention op. Its present
        // K/V outputs (slots 1..) are consumer-less graph outputs bound to a
        // growing device cache. Expose them to the kernel at the binding's
        // physical capacity so the kernel can append the new token into a fixed
        // per-head slot (constant stride, no per-step restride) instead of
        // repacking the whole cache densely. The valid attended length is still
        // derived from the logical past+current extent, so this only widens the
        // *storage* stride and never changes what the kernel attends over. Only
        // present slots that are bound sub-shape (logical != physical) capacity
        // buffers are widened; a dense/unbound present keeps its inferred shape.
        {
            let node = self.graph.node(node_id);
            if node.is_default_domain() && node.op_type == "Attention" {
                // When the past K/V inputs are themselves bound at physical
                // capacity (fixed-capacity decode, capture path), the standard
                // `present = past + current` shape rule sees the *physical* past
                // extent and over-counts the present seq axis beyond the bound
                // buffer. In that case the present buffer's true shape is simply
                // its physical capacity (mirroring GroupQueryAttention, whose
                // present rule takes `past_capacity.max(total)`); the valid
                // length lives on-device and context-overflow is caught earlier
                // in the decoder (`total_len > max_len`). Otherwise keep the
                // conservative `physical >= logical` guard.
                let kv_capacity_bound = kernel_input_uses_physical_capacity(node, 4)
                    && kernel_input_uses_physical_capacity(node, 5);
                for (oi, &ovid) in outputs.iter().enumerate() {
                    if oi == 0 {
                        continue;
                    }
                    if let Some(value) = external.outputs.get(&ovid)
                        && value.accepts_subshape
                        && value.shape.len() == output_shapes[oi].len()
                        && value
                            .shape
                            .iter()
                            .zip(&output_shapes[oi])
                            .enumerate()
                            .all(|(axis, (&physical, &logical))| axis == 2 || physical == logical)
                        && (kv_capacity_bound
                            || value
                                .shape
                                .get(2)
                                .zip(output_shapes[oi].get(2))
                                .is_some_and(|(&physical, &logical)| physical >= logical))
                    {
                        output_shapes[oi] = value.shape.clone();
                    }
                }
            }
        }
        let capabilities = self.ep.capabilities();
        let accepts_lazy_weights =
            LazyWeightBoundary::BlockQuantizedMoe.matches(&node.domain, &node.op_type);
        let has_lazy_inputs = accepts_lazy_weights
            && inputs.iter().any(|input| {
                input
                    .and_then(|value| self.weight_handles.get(&value))
                    .is_some_and(|handle| handle.is_lazy_for(&capabilities))
            });

        // Resolve each input's real geometry (root buffer + strides/offset) and
        // bounds-check it. View inputs read through their recorded strides.
        let mut in_infos: Vec<InInfo> = Vec::with_capacity(inputs.len());
        let _build_inputs_span = phase_span!("exec_kernel.build_inputs");
        for (i, slot) in inputs.iter().enumerate() {
            let Some(vid) = *slot else {
                in_infos.push(InInfo {
                    present: false,
                    dtype: input_dtypes[i],
                    shape: Vec::new(),
                    strides: Vec::new(),
                    byte_offset: 0,
                    base_ptr: std::ptr::null(),
                    device: self.ep.device_id(),
                    backing: TensorBacking::Opaque,
                    root_len: 0,
                });
                continue;
            };
            if let Some(value) = external
                .inputs
                .get(&vid)
                .or_else(|| external.outputs.get(&vid))
            {
                let strides = compute_contiguous_strides(&value.shape);
                view_bounds(&value.shape, &strides, 0, value.dtype, value.len)?;
                in_infos.push(InInfo {
                    present: true,
                    dtype: value.dtype,
                    shape: value.shape.clone(),
                    strides,
                    byte_offset: 0,
                    base_ptr: value.ptr.cast_const(),
                    device: value.device,
                    backing: TensorBacking::Opaque,
                    root_len: value.len,
                });
                continue;
            }
            // A tensor input backed by a shared sequence element (SequenceAt
            // output) owns no DeviceBuffer: read its possibly-strided view
            // directly over the immutable shared allocation.
            if let Some(elem) = self.seq_elem_values.get(&vid) {
                let shape = input_shapes[i].clone();
                let strides = elem.layout.resolved_strides(&shape);
                let root_len = elem.root_len();
                let base_ptr = elem.as_ptr();
                view_bounds(
                    &shape,
                    &strides,
                    elem.byte_offset(),
                    input_dtypes[i],
                    root_len,
                )?;
                in_infos.push(InInfo {
                    present: true,
                    dtype: input_dtypes[i],
                    shape,
                    strides,
                    byte_offset: elem.byte_offset(),
                    base_ptr,
                    device: elem.device(),
                    backing: TensorBacking::Opaque,
                    root_len,
                });
                continue;
            }
            if accepts_lazy_weights
                && self
                    .weight_handles
                    .get(&vid)
                    .is_some_and(|handle| handle.is_lazy_for(&capabilities))
            {
                in_infos.push(InInfo {
                    present: false,
                    dtype: input_dtypes[i],
                    shape: input_shapes[i].clone(),
                    strides: compute_contiguous_strides(&input_shapes[i]),
                    byte_offset: 0,
                    base_ptr: std::ptr::null(),
                    device: self.ep.device_id(),
                    backing: TensorBacking::Opaque,
                    root_len: 0,
                });
                continue;
            }
            let root = self.root_of(vid);
            let buf = self.buffers.get(&root).ok_or_else(|| {
                SessionError::Internal(format!("missing buffer for input value#{}", vid.0))
            })?;
            let root_len = buf.len();
            let base_ptr = buf.as_ptr();
            let (shape, strides, byte_offset) = match self.views.get(&vid) {
                Some(view) => (view.shape.clone(), view.strides.clone(), view.byte_offset),
                None => {
                    let shape = input_shapes[i].clone();
                    let strides = compute_contiguous_strides(&shape);
                    (shape, strides, 0)
                }
            };
            view_bounds(&shape, &strides, byte_offset, input_dtypes[i], root_len)?;
            let backing = self
                .graph
                .initializers
                .get(&root)
                .filter(|_| buf.is_borrowed())
                .and_then(|weight| self.weights.external_mmap_provenance(weight))
                .map(|(mapping_id, offset, len)| {
                    TensorBacking::ExternalMmap(ExternalMmapRegion {
                        mapping_id,
                        offset,
                        len,
                    })
                })
                .unwrap_or(TensorBacking::Opaque);
            in_infos.push(InInfo {
                present: true,
                dtype: input_dtypes[i],
                shape,
                strides,
                byte_offset,
                base_ptr,
                device: buf.device(),
                backing,
                root_len,
            });
        }
        drop(_build_inputs_span);

        let ep = self.ep.clone();

        // Bind the mutated fields as disjoint locals so `self` is never borrowed
        // whole while the kernel (from `cache`) and the buffers/views are held.
        let graph = &self.graph;
        let cache = &mut self.cache;
        let weight_handles = &self.weight_handles;
        let buffers = &mut self.buffers;
        let buffer_shapes = &mut self.buffer_shapes;
        let shared_buffers = &mut self.shared_buffers;
        let views_meta = &mut self.views;
        let pinned = &mut self.pinned;

        // Build the (possibly strided) input views once; they feed both the
        // view-output probe and, on the compute path, the kernel itself.
        let mut views: Vec<TensorView> = Vec::with_capacity(in_infos.len());
        for info in &in_infos {
            if !info.present {
                views.push(TensorView::absent(info.dtype));
                continue;
            }
            views.push(
                TensorView::new(
                    DevicePtr(info.base_ptr),
                    info.dtype,
                    &info.shape,
                    &info.strides,
                    info.device,
                )
                .with_byte_offset(info.byte_offset)
                .with_backing(info.backing),
            );
        }

        let opset = effective_opset(graph, node);
        let constant_inputs: Vec<bool> = inputs
            .iter()
            .map(|input| {
                input.is_some_and(|vid| {
                    graph.initializers.contains_key(&vid)
                        || views_meta
                            .get(&vid)
                            .is_some_and(|view| graph.initializers.contains_key(&view.source))
                })
            })
            .collect();
        let kernel = {
            let _s = phase_span!("exec_kernel.get_kernel");
            cache.get_or_create(
                node_id,
                node,
                input_shapes,
                input_dtypes,
                &constant_inputs,
                opset,
                ep.as_ref(),
            )?
        };
        // --- Zero-copy view fast path ---------------------------------------
        // Ask the kernel whether its outputs are strided views over its inputs
        // (a layout/movement op such as Slice). If so, record view metadata
        // aliasing the source buffer and skip compute + allocation entirely.
        if !has_lazy_inputs && let Some(specs) = kernel.view_outputs(&views, outputs.len()) {
            if outputs
                .iter()
                .any(|output| external.outputs.contains_key(output))
            {
                return Err(SessionError::Internal(format!(
                    "op '{}' cannot bind a zero-copy view output to external storage",
                    node.op_type
                )));
            }
            drop(views);
            if specs.len() != outputs.len() {
                return Err(SessionError::Internal(format!(
                    "op '{}' returned {} view outputs for {} outputs",
                    node.op_type,
                    specs.len(),
                    outputs.len()
                )));
            }
            for (oi, spec) in specs.into_iter().enumerate() {
                let ovid = outputs[oi];
                let Some(in_vid) = inputs.get(spec.input_index).copied().flatten() else {
                    return Err(SessionError::Internal(format!(
                        "op '{}' view output {} references invalid input index {}",
                        node.op_type, oi, spec.input_index
                    )));
                };
                let root = match views_meta.get(&in_vid) {
                    Some(v) => v.source,
                    None => in_vid,
                };
                let root_len = buffers.get(&root).map(|b| b.len()).ok_or_else(|| {
                    SessionError::Internal(format!("view source value#{} has no buffer", root.0))
                })?;
                // Bounds-gate the composed view against the source allocation.
                view_bounds(
                    &spec.shape,
                    &spec.strides,
                    spec.byte_offset,
                    output_dtypes[oi],
                    root_len,
                )?;
                // The output becomes a view: drop any buffer it used to own so a
                // later run re-sizes cleanly, then record the alias and pin the
                // source (conservative liveness — a source with any live view is
                // never reused/freed for the rest of the run; no use-after-free).
                // A freshly-produced output can never already be pinned (its
                // viewers run strictly after it under SSA topo order).
                debug_assert!(
                    !pinned.contains(&ovid),
                    "value#{} is pinned as a live view source yet is being reproduced",
                    ovid.0
                );
                if let Some(old) = buffers.remove(&ovid) {
                    ep.deallocate(old)?;
                }
                shared_buffers.remove(&ovid);
                buffer_shapes.remove(&ovid);
                views_meta.insert(
                    ovid,
                    ValueView {
                        source: root,
                        shape: spec.shape.clone(),
                        strides: spec.strides,
                        byte_offset: spec.byte_offset,
                    },
                );
                pinned.insert(root);
                resolved.insert(ovid, spec.shape);
            }
            return Ok(());
        }

        // --- Compute path ----------------------------------------------------
        // Size (allocate or reuse) each output's contiguous buffer, JIT-sizing
        // data-dependent ones. A value that was a view on a prior run has no
        // buffer here and is freshly allocated.
        for (oi, &ovid) in outputs.iter().enumerate() {
            let dims = &output_shapes[oi];
            let numel = checked_numel(dims, || format!("value#{}", ovid.0))?;
            let need = checked_storage_bytes(
                output_dtypes[oi],
                numel,
                || format!("value#{}", ovid.0),
                dims,
            )?
            .max(1);
            if let Some(value) = external.outputs.get(&ovid) {
                if !value.accepts_output(output_dtypes[oi], dims, need) {
                    let name = graph.value(ovid).name.as_deref().unwrap_or("<unnamed>");
                    return Err(SessionError::Internal(format!(
                        "external output '{name}' has {:?} {:?} ({} bytes), kernel requires {:?} {:?} ({need} bytes)",
                        value.dtype, value.shape, value.len, output_dtypes[oi], dims
                    )));
                }
                continue;
            }
            let fits = buffers.get(&ovid).map(|b| b.len() == need).unwrap_or(false);
            if !fits {
                // Never free a buffer that has a live view alias (would dangle
                // the viewer). Unreachable under SSA topo order, but enforced.
                debug_assert!(
                    !pinned.contains(&ovid),
                    "value#{} is pinned as a live view source yet is being resized",
                    ovid.0
                );
                if let Some(old) = buffers.remove(&ovid) {
                    ep.deallocate(old)?;
                }
                shared_buffers.remove(&ovid);
                let buf = ep.allocate(need, TensorLayout::contiguous().alignment)?;
                buffers.insert(ovid, buf);
            }
        }

        // Auto-materialization gate: a strided (view) input feeding a kernel
        // that does not accept strided input on that slot is gathered into a
        // private contiguous temp so contiguous-assuming kernels stay correct.
        // Temps must outlive the views that borrow them.
        let mut mat: Vec<Option<(Vec<u8>, Vec<i64>)>> = Vec::with_capacity(in_infos.len());
        for (i, info) in in_infos.iter().enumerate() {
            if !info.present {
                mat.push(None);
                continue;
            }
            let contiguous = onnx_runtime_ir::is_contiguous(&info.shape, &info.strides);
            if contiguous || kernel.supports_strided_input(i) {
                mat.push(None);
                continue;
            }
            if !info.device.is_host_accessible() {
                return Err(SessionError::Internal(format!(
                    "op '{}' requires host-only strided materialization for CUDA input {i}",
                    node.op_type
                )));
            }
            let esize = info.dtype.byte_size();
            if esize == 0 {
                return Err(SessionError::from(
                    onnx_runtime_ep_api::EpError::InvalidTensorView {
                        reason: format!(
                            "cannot materialize sub-byte strided input {i} of op '{}'",
                            node.op_type
                        ),
                    },
                ));
            }
            let src =
                unsafe { std::slice::from_raw_parts(info.base_ptr as *const u8, info.root_len) };
            let gathered = gather_view(src, &info.shape, &info.strides, info.byte_offset, esize);
            let strides = compute_contiguous_strides(&info.shape);
            mat.push(Some((gathered, strides)));
        }

        // Rebuild input views, swapping any materialized slot to its contiguous
        // temp (offset 0, contiguous strides over the fresh buffer).
        drop(views);
        let mut views: Vec<TensorView> = Vec::with_capacity(in_infos.len());
        for (i, info) in in_infos.iter().enumerate() {
            if !info.present {
                views.push(TensorView::absent(info.dtype));
                continue;
            }
            match &mat[i] {
                Some((buf, strides)) => views.push(TensorView::new(
                    DevicePtr(buf.as_ptr() as *const std::ffi::c_void),
                    info.dtype,
                    &info.shape,
                    strides,
                    onnx_runtime_ir::DeviceId::cpu(),
                )),
                None => views.push(
                    TensorView::new(
                        DevicePtr(info.base_ptr),
                        info.dtype,
                        &info.shape,
                        &info.strides,
                        info.device,
                    )
                    .with_byte_offset(info.byte_offset)
                    .with_backing(info.backing),
                ),
            }
        }

        // Take output buffers out so they can be borrowed `&mut` disjointly from
        // the input reads (SSA guarantees outputs are disjoint from inputs).
        let out_strides: Vec<Vec<i64>> = output_shapes
            .iter()
            .map(|s| compute_contiguous_strides(s))
            .collect();
        struct OutBacking {
            vid: ValueId,
            internal: Option<DeviceBuffer>,
            ptr: *mut std::ffi::c_void,
            len: usize,
            device: onnx_runtime_ir::DeviceId,
        }
        let mut out_bufs: Vec<OutBacking> = Vec::with_capacity(outputs.len());
        for &vid in outputs {
            if let Some(value) = external.outputs.get(&vid) {
                out_bufs.push(OutBacking {
                    vid,
                    internal: None,
                    ptr: value.ptr,
                    len: value.len,
                    device: value.device,
                });
            } else {
                let mut buf = buffers.remove(&vid).ok_or_else(|| {
                    SessionError::Internal(format!("missing buffer for output value#{}", vid.0))
                })?;
                let ptr = buf.as_mut_ptr();
                out_bufs.push(OutBacking {
                    vid,
                    ptr,
                    len: buf.len(),
                    device: buf.device(),
                    internal: Some(buf),
                });
            }
        }
        let mut outs: Vec<TensorMut> = Vec::with_capacity(out_bufs.len());
        for (i, backing) in out_bufs.iter_mut().enumerate() {
            view_bounds(
                &output_shapes[i],
                &out_strides[i],
                0,
                output_dtypes[i],
                backing.len,
            )?;
            outs.push(TensorMut::new(
                DevicePtrMut(backing.ptr),
                output_dtypes[i],
                &output_shapes[i],
                &out_strides[i],
                backing.device,
            ));
        }

        let kernel_inputs = has_lazy_inputs.then(|| {
            inputs
                .iter()
                .zip(views.iter().copied())
                .map(|(value, view)| {
                    value
                        .and_then(|value| weight_handles.get(&value))
                        .filter(|handle| handle.is_lazy_for(&capabilities))
                        .map(KernelInput::Weight)
                        .unwrap_or(KernelInput::Tensor(view))
                })
                .collect::<Vec<_>>()
        });
        let execution = {
            let _s = phase_span!("exec_kernel.compute");
            match &kernel_inputs {
                Some(inputs) => kernel.execute_with_inputs(inputs, &mut outs),
                None => kernel.execute(&views, &mut outs),
            }
        };
        execution.map_err(|error| {
                let input_types = views.iter().map(|view| view.dtype).collect::<Vec<_>>();
                let output_types = outs.iter().map(|output| output.dtype).collect::<Vec<_>>();
                let input_shapes = views
                    .iter()
                    .map(|view| view.shape.to_vec())
                    .collect::<Vec<_>>();
                let output_shapes = outs
                    .iter()
                    .map(|output| output.shape.to_vec())
                    .collect::<Vec<_>>();
                let input_names = inputs
                    .iter()
                    .map(|input| {
                        input
                            .map(|value| {
                                self.graph.value(value).name.as_deref().unwrap_or("<unnamed>")
                            })
                            .unwrap_or("<absent>")
                    })
                    .collect::<Vec<_>>();
                let output_names = outputs
                    .iter()
                    .map(|&value| {
                        self.graph.value(value).name.as_deref().unwrap_or("<unnamed>")
                    })
                    .collect::<Vec<_>>();
                SessionError::Internal(format!(
                    "node {} ({:?}, op '{}::{}', inputs {input_names:?} {input_types:?} {input_shapes:?}, outputs {output_names:?} {output_types:?} {output_shapes:?}) failed: {error}",
                    node.id.0, node.name, node.domain, node.op_type,
                ))
            })?;

        drop(kernel_inputs);
        drop(views);
        drop(outs);
        for backing in out_bufs {
            if let Some(buf) = backing.internal {
                buffers.insert(backing.vid, buf);
            }
        }
        Ok(())
    }

    /// Read the integer *values* of input `vid` as `i64`, materializing a view
    /// first if needed. Used to resolve data-dependent output shapes (e.g. a
    /// `Slice` whose `ends` is produced at runtime). Returns `None` if the value
    /// has no readable buffer/view or its dtype is not an integer.
    fn input_i64(&self, vid: ValueId, shape: &[usize], dtype: DataType) -> Option<Vec<i64>> {
        let bytes = self.contiguous_bytes(vid, shape, dtype).ok()?;
        bytes_as_i64(&bytes, dtype)
    }

    /// Bounded integer reader for dynamic shape propagation. Views and sequence
    /// elements can have a tiny logical shape backed by a much larger root
    /// allocation, so cap that allocation before `contiguous_bytes` can copy it.
    fn shape_input_i64(&self, vid: ValueId, shape: &[usize], dtype: DataType) -> Option<Vec<i64>> {
        if !bounded_shape_input(dtype, shape) {
            return None;
        }
        let max_bytes = MAX_SHAPE_DATA_ELEMS.checked_mul(dtype.byte_size())?;
        if let Some(view) = self.views.get(&vid) {
            let source = self.buffers.get(&view.source)?;
            if source.len() > max_bytes {
                return None;
            }
        }
        if self
            .seq_elem_values
            .get(&vid)
            .is_some_and(|elem| elem.root_len() > max_bytes)
        {
            return None;
        }
        self.input_i64(vid, shape, dtype)
    }

    fn shape_input_f64(&self, vid: ValueId, shape: &[usize], dtype: DataType) -> Option<Vec<f64>> {
        if !matches!(dtype, DataType::Float32 | DataType::Float64)
            || shape.len() > 1
            || shape
                .iter()
                .try_fold(1usize, |count, &dim| count.checked_mul(dim))
                .is_none_or(|count| count > MAX_SHAPE_DATA_ELEMS)
        {
            return None;
        }
        let max_bytes = MAX_SHAPE_DATA_ELEMS.checked_mul(dtype.byte_size())?;
        if let Some(view) = self.views.get(&vid) {
            let source = self.buffers.get(&view.source)?;
            if source.len() > max_bytes {
                return None;
            }
        }
        if self
            .seq_elem_values
            .get(&vid)
            .is_some_and(|elem| elem.root_len() > max_bytes)
        {
            return None;
        }
        let bytes = self.contiguous_bytes(vid, shape, dtype).ok()?;
        bytes_as_f64(&bytes, dtype)
    }

    /// `NonMaxSuppression` needs its boxes and scores to determine the exact
    /// data-dependent output extent. Unlike ordinary shape tensors these inputs
    /// are rank 3 and may exceed `MAX_SHAPE_DATA_ELEMS`; materialize them only
    /// for this operator, immediately before its output allocation.
    fn nms_input_f64(&self, vid: ValueId, shape: &[usize], dtype: DataType) -> Option<Vec<f64>> {
        if dtype != DataType::Float32 {
            return None;
        }
        let bytes = self.contiguous_bytes(vid, shape, dtype).ok()?;
        bytes_as_f64(&bytes, dtype)
    }
}

// === Sequence-of-tensors ops: SequenceEmpty / SequenceConstruct /
// SequenceInsert / SequenceErase / SequenceAt / SequenceLength /
// SplitToSequence / ConcatFromSequence ===
//
// These are handled at the executor level (like control-flow ops) rather than as
// leaf kernels, because they operate on a *sequence-of-tensors* runtime value
// that a `Kernel` — which sees only individual tensor views — cannot represent.
//
// ## No-copy design
//
// A sequence stores its elements as `Arc`-shared **immutable** [`SeqTensor`]s
// (see [`crate::sequence`]). Insert/Erase/Construct build a NEW list that SHARES
// the surviving element `Arc`s — only handles (a refcount bump), never element
// bytes, are cloned. `SequenceAt` yields the shared element `Arc` and backs its
// output tensor value with that same allocation (`seq_elem_values`), so a
// downstream kernel reads it through a zero-copy [`TensorView`] and no bytes are
// copied out of the sequence until the graph-output boundary. Tensor→sequence
// entry promotes the existing `DeviceBuffer` into an Arc owner and leaves a
// non-owning dispatch alias in the executor. `SplitToSequence` creates
// shape/stride/offset views over that same owner. `ConcatFromSequence` is the
// only sequence data op that materializes a new contiguous tensor.
//
// ## No-race design
//
// Elements are immutable after construction and only ever shared read-only
// through `Arc`; there is no interior mutability, so concurrent readers cannot
// race (the only cross-thread state is `Arc`'s atomic refcount).
impl Executor {
    /// Execute one Sequence-op plan node.
    fn exec_sequence_node(
        &mut self,
        pi: usize,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        external: &ExternalBindings,
    ) -> Result<()> {
        let node_id = self.plan[pi].node_id;
        let inputs = self.plan[pi].inputs.clone();
        let outputs = self.plan[pi].outputs.clone();
        let op = self.graph.node(node_id).op_type.clone();

        match op.as_str() {
            "SequenceEmpty" => {
                let dtype_attr = self
                    .graph
                    .node(node_id)
                    .attr("dtype")
                    .and_then(|a| a.as_int());
                let dtype = match dtype_attr {
                    None => DataType::Float32, // ONNX default element type.
                    Some(raw) => i32::try_from(raw)
                        .ok()
                        .and_then(DataType::from_onnx)
                        .ok_or_else(|| SessionError::SequenceOp {
                            op: op.clone(),
                            reason: format!(
                                "attribute 'dtype' = {raw} is not a known ONNX \
                                 TensorProto.DataType. To fix: use a valid element \
                                 dtype id (e.g. 1=float32, 7=int64)"
                            ),
                        })?,
                };
                self.sequences
                    .insert(outputs[0], SequenceValue::empty(dtype));
                Ok(())
            }
            "SequenceConstruct" => {
                let mut items = Vec::with_capacity(inputs.len());
                for slot in &inputs {
                    let vid = slot.ok_or_else(|| self.seq_missing_input(&op))?;
                    items.push(self.read_seq_element(vid, resolved)?);
                }
                let seq = SequenceValue::construct(items).map_err(seq_err)?;
                self.sequences.insert(outputs[0], seq);
                Ok(())
            }
            "SequenceInsert" => {
                let seq = self.get_sequence(inputs.first().copied().flatten(), &op)?;
                let tvid = inputs
                    .get(1)
                    .copied()
                    .flatten()
                    .ok_or_else(|| self.seq_missing_input(&op))?;
                let tensor = self.read_seq_element(tvid, resolved)?;
                let position = match inputs.get(2).copied().flatten() {
                    Some(pvid) => Some(self.read_scalar_i64(pvid, resolved, &op)?),
                    None => None,
                };
                let out = seq.insert(tensor, position).map_err(seq_err)?;
                self.sequences.insert(outputs[0], out);
                Ok(())
            }
            "SequenceErase" => {
                let seq = self.get_sequence(inputs.first().copied().flatten(), &op)?;
                let position = match inputs.get(1).copied().flatten() {
                    Some(pvid) => Some(self.read_scalar_i64(pvid, resolved, &op)?),
                    None => None,
                };
                let out = seq.erase(position).map_err(seq_err)?;
                self.sequences.insert(outputs[0], out);
                Ok(())
            }
            "SequenceAt" => {
                let seq = self.get_sequence(inputs.first().copied().flatten(), &op)?;
                let pvid =
                    inputs
                        .get(1)
                        .copied()
                        .flatten()
                        .ok_or_else(|| SessionError::SequenceOp {
                            op: op.clone(),
                            reason: "requires a 'position' input. To fix: supply the \
                                 index tensor of the element to read"
                                .to_string(),
                        })?;
                let pos = self.read_scalar_i64(pvid, resolved, &op)?;
                let elem = seq.at(pos).map_err(seq_err)?;
                self.store_seq_element_output(outputs[0], elem, resolved, external)
            }
            "SequenceLength" => {
                let seq = self.get_sequence(inputs.first().copied().flatten(), &op)?;
                let len = i64::try_from(seq.length()).map_err(|_| {
                    seq_err(SequenceError::LengthOverflow {
                        op: "SequenceLength",
                        len: seq.length(),
                    })
                })?;
                self.store_raw_tensor_output(
                    outputs[0],
                    DataType::Int64,
                    Vec::new(),
                    &len.to_le_bytes(),
                    resolved,
                    external,
                )
            }
            "SplitToSequence" => {
                self.exec_split_to_sequence(node_id, &op, &inputs, &outputs, resolved)
            }
            "ConcatFromSequence" => {
                self.exec_concat_from_sequence(node_id, &op, &inputs, &outputs, resolved, external)
            }
            other => Err(SessionError::SequenceOp {
                op: other.to_string(),
                reason: "unrecognized Sequence op (executor routing bug)".to_string(),
            }),
        }
    }

    /// `SplitToSequence`: split a tensor into a sequence along `axis`.
    fn exec_split_to_sequence(
        &mut self,
        node_id: NodeId,
        op: &str,
        inputs: &[Option<ValueId>],
        outputs: &[ValueId],
        resolved: &mut HashMap<ValueId, Vec<usize>>,
    ) -> Result<()> {
        let (axis_attr, keepdims) = {
            let node = self.graph.node(node_id);
            (
                node.attr("axis").and_then(|a| a.as_int()).unwrap_or(0),
                node.attr("keepdims").and_then(|a| a.as_int()).unwrap_or(1) != 0,
            )
        };

        let ivid = inputs
            .first()
            .copied()
            .flatten()
            .ok_or_else(|| self.seq_missing_input(op))?;
        let input = self.read_seq_element(ivid, resolved)?;

        let split_input = match inputs.get(1).copied().flatten() {
            None => None,
            Some(svid) => {
                let split_shape = resolved
                    .get(&svid)
                    .cloned()
                    .ok_or_else(|| self.seq_unresolved(op, svid))?;
                let values = self.read_i64_vec(svid, &split_shape, op)?;
                Some((split_shape, values))
            }
        };
        let split_spec = match split_input.as_ref() {
            None => SplitSpec::Each,
            Some((split_shape, values)) if split_shape.is_empty() => {
                let [chunk] = values.as_slice() else {
                    return Err(SessionError::SequenceOp {
                        op: op.to_string(),
                        reason: format!(
                            "scalar 'split' input contains {} values, expected exactly one",
                            values.len()
                        ),
                    });
                };
                SplitSpec::Chunk(*chunk)
            }
            Some((split_shape, values)) if split_shape.len() == 1 => SplitSpec::Sizes(values),
            Some((split_shape, _)) => {
                return Err(SessionError::SequenceOp {
                    op: op.to_string(),
                    reason: format!(
                        "'split' input must be rank 0 (chunk size) or rank 1 (explicit sizes), \
                         got rank {} with shape {split_shape:?}",
                        split_shape.len()
                    ),
                });
            }
        };
        let sequence = split_tensor(&input, axis_attr, split_spec, keepdims).map_err(seq_err)?;
        self.sequences.insert(outputs[0], sequence);
        Ok(())
    }

    /// `ConcatFromSequence`: concatenate (or stack, when `new_axis=1`) a
    /// sequence's tensors into one freshly-allocated output.
    fn exec_concat_from_sequence(
        &mut self,
        node_id: NodeId,
        op: &str,
        inputs: &[Option<ValueId>],
        outputs: &[ValueId],
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        external: &ExternalBindings,
    ) -> Result<()> {
        let node = self.graph.node(node_id);
        let axis_attr =
            node.attr("axis")
                .and_then(|a| a.as_int())
                .ok_or_else(|| SessionError::SequenceOp {
                    op: op.to_string(),
                    reason: "requires the mandatory 'axis' attribute. To fix: set 'axis'"
                        .to_string(),
                })?;
        let new_axis = node.attr("new_axis").and_then(|a| a.as_int()).unwrap_or(0) != 0;

        let seq = self.get_sequence(inputs.first().copied().flatten(), op)?;
        let plan = ConcatPlan::new(&seq, axis_attr, new_axis).map_err(seq_err)?;
        self.prepare_tensor_output(
            outputs[0],
            plan.dtype,
            plan.shape.clone(),
            plan.bytes,
            resolved,
            external,
        )?;
        let ep = Arc::clone(&self.ep);
        if let Some(value) = external.outputs.get(&outputs[0]) {
            let mut buffer = value.writable_buffer()?;
            plan.write(&seq, |offset, bytes| {
                ep.copy_from_host_at(bytes, &mut buffer, offset)?;
                Ok(())
            })?;
        } else {
            let buffer = self.buffers.get_mut(&outputs[0]).ok_or_else(|| {
                SessionError::Internal(format!(
                    "missing ConcatFromSequence output buffer for value#{}",
                    outputs[0].0
                ))
            })?;
            plan.write(&seq, |offset, bytes| {
                ep.copy_from_host_at(bytes, buffer, offset)?;
                Ok(())
            })?;
        }
        Ok(())
    }

    /// Build (or share) a `SeqTensor` for a tensor value entering a sequence.
    /// Existing sequence elements clone their Arc. Ordinary tensors promote the
    /// existing allocation into a shared owner and keep a non-owning executor
    /// alias, so no element bytes move.
    fn read_seq_element(
        &mut self,
        vid: ValueId,
        resolved: &HashMap<ValueId, Vec<usize>>,
    ) -> Result<SeqTensor> {
        if self.sequence_values.contains(&vid) {
            return Err(SessionError::SequenceOp {
                op: "Sequence".to_string(),
                reason: format!(
                    "input value#{} is a Sequence value, expected a tensor element",
                    vid.0
                ),
            });
        }
        if let Some(elem) = self.seq_elem_values.get(&vid) {
            return Ok(elem.clone()); // zero-copy Arc share
        }
        let dtype = self.value_dtypes[&vid];
        let shape = resolved
            .get(&vid)
            .cloned()
            .ok_or_else(|| self.seq_unresolved("Sequence", vid))?;
        let (root, layout, byte_offset) = match self.views.get(&vid) {
            Some(view) => (
                view.source,
                TensorLayout::strided(view.strides.clone()),
                view.byte_offset,
            ),
            None => (vid, TensorLayout::contiguous(), 0),
        };
        if !self.shared_buffers.contains_key(&root) {
            let buffer = self
                .buffers
                .remove(&root)
                .ok_or_else(|| SessionError::SequenceOp {
                    op: "Sequence".to_string(),
                    reason: format!("tensor value#{} has no live backing buffer", vid.0),
                })?;
            let storage = SharedTensorBuffer::new(Arc::clone(&self.ep), buffer);
            self.buffers.insert(root, storage.alias());
            self.shared_buffers.insert(root, storage);
        }
        self.pinned.insert(root);
        SeqTensor::from_shared(
            Arc::clone(&self.shared_buffers[&root]),
            dtype,
            shape,
            layout,
            byte_offset,
        )
        .map_err(SessionError::from)
    }

    fn restore_shared_buffers(&mut self) -> Result<()> {
        let mut retained = Vec::new();
        for (vid, storage) in self.shared_buffers.drain() {
            if let Some(alias) = self.buffers.remove(&vid) {
                self.ep.deallocate(alias)?;
            }
            match Arc::try_unwrap(storage) {
                Ok(storage) => {
                    self.buffers.insert(vid, storage.into_buffer());
                }
                Err(storage) if self.graph.initializers.contains_key(&vid) => {
                    self.buffers.insert(vid, storage.alias());
                    retained.push((vid, storage));
                }
                Err(storage) => {
                    let replacement = self
                        .ep
                        .allocate(storage.buffer().len(), storage.buffer().alignment())?;
                    self.buffers.insert(vid, replacement);
                }
            }
        }
        for (vid, storage) in retained {
            self.shared_buffers.insert(vid, storage);
        }
        Ok(())
    }

    /// Fetch (clone) the sequence value bound to `vid` (cheap — `Arc` handle
    /// clones), or an actionable error if the input is missing / not a sequence.
    fn get_sequence(&self, vid: Option<ValueId>, op: &str) -> Result<SequenceValue> {
        let vid = vid.ok_or_else(|| self.seq_missing_input(op))?;
        self.sequences
            .get(&vid)
            .cloned()
            .ok_or_else(|| SessionError::SequenceOp {
                op: op.to_string(),
                reason: format!(
                    "input value#{} is not a live sequence. To fix: ensure it is produced \
                 by a Sequence-producing op (SequenceEmpty/Construct/Insert/Erase/\
                 SplitToSequence)",
                    vid.0
                ),
            })
    }

    /// Read a scalar `i64`/`i32` position input.
    fn read_scalar_i64(
        &self,
        vid: ValueId,
        resolved: &HashMap<ValueId, Vec<usize>>,
        op: &str,
    ) -> Result<i64> {
        let shape = resolved.get(&vid).cloned().unwrap_or_default();
        if !shape.is_empty() {
            return Err(SessionError::SequenceOp {
                op: op.to_string(),
                reason: format!(
                    "position input must be a rank-0 scalar, got rank {} with shape {shape:?}",
                    shape.len()
                ),
            });
        }
        let dtype = self.value_dtypes[&vid];
        let vals = self
            .input_i64(vid, &shape, dtype)
            .ok_or_else(|| SessionError::SequenceOp {
                op: op.to_string(),
                reason: format!(
                    "position input has dtype {dtype:?}, expected an integer (int32/int64). \
                 To fix: provide an int64 scalar index"
                ),
            })?;
        let [value] = vals.as_slice() else {
            return Err(SessionError::SequenceOp {
                op: op.to_string(),
                reason: format!(
                    "position input contains {} values; expected exactly one scalar index",
                    vals.len()
                ),
            });
        };
        Ok(*value)
    }

    /// Read an `i64` vector from an integer tensor input (SplitToSequence's
    /// `split`).
    fn read_i64_vec(&self, vid: ValueId, shape: &[usize], op: &str) -> Result<Vec<i64>> {
        let dtype = self.value_dtypes[&vid];
        self.input_i64(vid, shape, dtype)
            .ok_or_else(|| SessionError::SequenceOp {
                op: op.to_string(),
                reason: format!(
                    "'split' input has dtype {dtype:?}, expected int32/int64. To fix: \
                 provide integer split sizes"
                ),
            })
    }

    /// Back a tensor *output* value with a shared sequence element (SequenceAt).
    /// The element retains its original device allocation and view metadata.
    fn store_seq_element_output(
        &mut self,
        vid: ValueId,
        elem: SeqTensor,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        external: &ExternalBindings,
    ) -> Result<()> {
        if elem.device() != self.ep.device_id() {
            return Err(SessionError::SequenceOp {
                op: "SequenceAt".to_string(),
                reason: format!(
                    "sequence element is on {:?}, but the active execution provider is on {:?}",
                    elem.device(),
                    self.ep.device_id()
                ),
            });
        }
        if external.outputs.contains_key(&vid) {
            let dtype = elem.dtype;
            let shape = elem.shape.clone();
            let bytes = elem.contiguous_bytes().map_err(seq_err)?;
            return self.store_raw_tensor_output(vid, dtype, shape, &bytes, resolved, external);
        }
        if let Some(old) = self.buffers.remove(&vid) {
            self.ep.deallocate(old)?;
        }
        self.shared_buffers.remove(&vid);
        self.buffer_shapes.remove(&vid);
        self.views.remove(&vid);
        resolved.insert(vid, elem.shape.clone());
        self.value_dtypes.insert(vid, elem.dtype);
        self.seq_elem_values.insert(vid, elem);
        Ok(())
    }

    /// Store freshly-computed contiguous bytes into a tensor output value
    /// (SequenceLength / ConcatFromSequence): (re)allocate its buffer, copy the
    /// bytes once, and record its dtype/shape.
    fn store_raw_tensor_output(
        &mut self,
        vid: ValueId,
        dtype: DataType,
        dims: Vec<usize>,
        bytes: &[u8],
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        external: &ExternalBindings,
    ) -> Result<()> {
        self.prepare_tensor_output(vid, dtype, dims, bytes.len(), resolved, external)?;
        if let Some(value) = external.outputs.get(&vid) {
            let mut buffer = value.writable_buffer()?;
            self.ep.copy_from_host(bytes, &mut buffer)?;
        } else {
            let buffer = self.buffers.get_mut(&vid).ok_or_else(|| {
                SessionError::Internal(format!("missing tensor output buffer for value#{}", vid.0))
            })?;
            self.ep.copy_from_host(bytes, buffer)?;
        }
        Ok(())
    }

    fn prepare_tensor_output(
        &mut self,
        vid: ValueId,
        dtype: DataType,
        dims: Vec<usize>,
        bytes: usize,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        external: &ExternalBindings,
    ) -> Result<()> {
        self.seq_elem_values.remove(&vid);
        self.views.remove(&vid);
        let need = bytes.max(1);
        if let Some(value) = external.outputs.get(&vid) {
            if !value.accepts_output(dtype, &dims, need) {
                let name = self.graph.value(vid).name.as_deref().unwrap_or("<unnamed>");
                return Err(SessionError::Internal(format!(
                    "external output '{name}' has {:?} {:?} ({} bytes), sequence op requires {:?} {:?} ({need} bytes)",
                    value.dtype, value.shape, value.len, dtype, dims
                )));
            }
        } else {
            let fits = self
                .buffers
                .get(&vid)
                .map(|buffer| buffer.len() == need)
                .unwrap_or(false);
            if !fits {
                if let Some(old) = self.buffers.remove(&vid) {
                    self.ep.deallocate(old)?;
                }
                self.shared_buffers.remove(&vid);
                let buffer = self
                    .ep
                    .allocate(need, TensorLayout::contiguous().alignment)?;
                self.buffers.insert(vid, buffer);
            }
            self.buffer_shapes.insert(vid, dims.clone());
        }
        self.value_dtypes.insert(vid, dtype);
        resolved.insert(vid, dims);
        Ok(())
    }

    fn seq_missing_input(&self, op: &str) -> SessionError {
        SessionError::SequenceOp {
            op: op.to_string(),
            reason: "a required input is missing (omitted None slot). To fix: connect \
                     all required inputs of this Sequence op"
                .to_string(),
        }
    }

    fn seq_unresolved(&self, op: &str, vid: ValueId) -> SessionError {
        let name = self
            .graph
            .try_value(vid)
            .and_then(|v| v.name.clone())
            .unwrap_or_else(|| format!("value#{}", vid.0));
        SessionError::SequenceOp {
            op: op.to_string(),
            reason: format!(
                "input {name} has no resolved shape yet. To fix: ensure its producer \
                 runs before this Sequence op"
            ),
        }
    }
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

impl ChildExecutor {
    /// Create the reusable wrapper for a loaded subgraph body.
    ///
    /// `body.inputs` and `body.outputs` are the loader-preserved ordered formal
    /// signature. Producer-less named values that are neither formals nor local
    /// initializers are lexical captures and are bound from `outer_scope`.
    pub(crate) fn new(
        name: impl Into<String>,
        body: Graph,
        inherited_opsets: HashMap<String, u64>,
        weights: Arc<WeightStore>,
        ep: Arc<dyn ExecutionProvider>,
    ) -> Result<Self> {
        let name = name.into();
        let formal_names = body
            .inputs
            .iter()
            .map(|&vid| {
                body.value(vid).name.clone().ok_or_else(|| {
                    SessionError::Internal(format!(
                        "subgraph '{name}' has an unnamed formal input value#{}",
                        vid.0
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let formal_set: HashSet<ValueId> = body.inputs.iter().copied().collect();
        let mut capture_names = body
            .values
            .iter()
            .filter_map(|(vid, value)| {
                (value.producer.is_none()
                    && !formal_set.contains(&vid)
                    && !body.initializers.contains_key(&vid))
                .then(|| value.name.clone())
                .flatten()
            })
            .collect::<Vec<_>>();
        capture_names.sort();
        let input_names = formal_names
            .iter()
            .chain(capture_names.iter())
            .cloned()
            .collect();

        Ok(Self {
            name,
            template: body,
            inherited_opsets,
            weights,
            ep,
            formal_names,
            capture_names,
            input_names,
            compiled: Vec::new(),
            builds: 0,
            runs: 0,
            trace: TraceContext::noop(),
        })
    }

    pub(crate) fn stats(&self) -> ChildExecutorStats {
        ChildExecutorStats {
            builds: self.builds,
            runs: self.runs,
        }
    }

    /// Attach the shared trace context, propagating it to every already-compiled
    /// child plan and to plans compiled later.
    pub(crate) fn set_trace_context(&mut self, trace: TraceContext) {
        for plan in &mut self.compiled {
            plan.exec.set_trace_context(trace.clone());
        }
        self.trace = trace;
    }

    fn compile(&self, externals: &[&Tensor]) -> Result<CompiledChildPlan> {
        let mut graph = self.template.clone();
        // GraphProto has no opset table: nested graphs inherit the model-level
        // imports from their enclosing graph.
        graph.opset_imports = self.inherited_opsets.clone();

        let body_names = graph
            .values
            .iter()
            .filter_map(|(vid, value)| value.name.clone().map(|name| (name, vid)))
            .collect::<HashMap<_, _>>();

        // Direct captures become required graph inputs. Local inline
        // initializers stay in `graph.initializers`, preserving their scope.
        for name in &self.capture_names {
            let vid = *body_names.get(name).ok_or_else(|| {
                SessionError::Internal(format!(
                    "subgraph '{}' lost captured value '{name}'",
                    self.name
                ))
            })?;
            if !graph.inputs.contains(&vid) {
                graph.add_input(vid);
            }
        }

        for (name, tensor) in self.input_names.iter().zip(externals) {
            let vid = *body_names.get(name).ok_or_else(|| {
                SessionError::Internal(format!(
                    "subgraph '{}' is missing bound input '{name}'",
                    self.name
                ))
            })?;
            let value = graph.value_mut(vid);
            value.dtype = tensor.dtype;
            value.shape = tensor.shape.iter().map(|&dim| Dim::Static(dim)).collect();
        }

        // Seeded formal/capture shapes let inference resolve the body once.
        // Truly data-dependent outputs remain on Executor's JIT shape path.
        let registry = InferenceRegistry::default_registry();
        registry.infer_graph(&mut graph, &self.inherited_opsets, MergePolicy::Permissive)?;

        Ok(CompiledChildPlan {
            exec: {
                let mut exec = Executor::build(graph, self.weights.clone(), self.ep.clone())?;
                exec.set_trace_context(self.trace.clone());
                exec
            },
            signature: externals
                .iter()
                .map(|tensor| ChildInputSignature {
                    dtype: tensor.dtype,
                    shape: tensor.shape.clone(),
                })
                .collect(),
        })
    }

    /// Execute the body with formal inputs in declared order and lexical values
    /// supplied by name. A cached plan is reused for matching dtype/shapes.
    pub(crate) fn run(
        &mut self,
        formal_inputs: &[&Tensor],
        outer_scope: &HashMap<String, Tensor>,
    ) -> Result<Vec<Tensor>> {
        if self.formal_names.len() != formal_inputs.len() {
            return Err(SessionError::Internal(format!(
                "subgraph '{}' expects {} formal input(s) but {} were supplied",
                self.name,
                self.formal_names.len(),
                formal_inputs.len()
            )));
        }

        let mut externals = Vec::with_capacity(formal_inputs.len() + self.capture_names.len());
        externals.extend_from_slice(formal_inputs);
        for name in &self.capture_names {
            externals.push(
                outer_scope
                    .get(name)
                    .ok_or_else(|| missing_capture_error(&self.name, name))?,
            );
        }

        let signature = externals
            .iter()
            .map(|tensor| ChildInputSignature {
                dtype: tensor.dtype,
                shape: tensor.shape.clone(),
            })
            .collect::<Vec<_>>();
        let cache_index = if let Some(index) = self
            .compiled
            .iter()
            .position(|compiled| compiled.signature == signature)
        {
            let compiled = self.compiled.remove(index);
            self.compiled.push(compiled);
            self.compiled.len() - 1
        } else {
            let compiled = self.compile(&externals)?;
            if self.compiled.len() == CHILD_EXECUTOR_CACHE_CAPACITY {
                self.compiled.remove(0);
            }
            self.compiled.push(compiled);
            self.builds += 1;
            self.compiled.len() - 1
        };

        self.runs += 1;
        let inputs = self
            .input_names
            .iter()
            .map(String::as_str)
            .zip(externals)
            .collect::<Vec<_>>();
        self.compiled[cache_index]
            .exec
            .run_scoped(&inputs, outer_scope, &ExternalBindings::default())?
            .into_iter()
            .map(|output| {
                let output = output.ok_or_else(|| {
                    SessionError::Internal(format!(
                        "subgraph '{}' unexpectedly suppressed an output",
                        self.name
                    ))
                })?;
                match output {
                    SessionOutput::Tensor(tensor) => Ok(tensor),
                    SessionOutput::Sequence(_) => Err(SessionError::SequenceOp {
                        op: "<control-flow output>".to_string(),
                        reason: format!(
                            "subgraph '{}' produced a Sequence output where this control-flow path requires a tensor",
                            self.name
                        ),
                    }),
                }
            })
            .collect()
    }
}

// === Control-flow (subgraph-executing) ops: If / Loop / Scan ===
//
// These are handled at the executor level rather than as leaf kernels because
// they must recursively execute a nested ONNX [`Graph`] with the enclosing
// scope bound — something a `Kernel` (which sees only tensor views, never the
// session/graph context) cannot do. Each body is compiled to a child
// [`Executor`] once and **reused across iterations** (see [`ChildExecutor`]).
impl Executor {
    /// Materialize value `vid`'s current bytes into an owned host [`Tensor`],
    /// using its resolved concrete shape and recorded dtype.
    fn value_tensor(
        &self,
        vid: ValueId,
        resolved: &HashMap<ValueId, Vec<usize>>,
    ) -> Result<Tensor> {
        let dtype = self.value_dtypes[&vid];
        let shape = resolved.get(&vid).cloned().ok_or_else(|| {
            let name = self
                .graph
                .try_value(vid)
                .and_then(|v| v.name.clone())
                .unwrap_or_else(|| format!("value#{}", vid.0));
            SessionError::UnresolvedShape {
                value: name,
                op: "<control-flow input>".to_string(),
            }
        })?;
        // A view value owns no buffer; materialize its strided bytes contiguous.
        let bytes = self.contiguous_bytes(vid, &shape, dtype)?;
        Tensor::from_raw(dtype, shape, &bytes)
    }

    /// The buffer-owning (root) value backing `vid`: `vid` itself if it owns a
    /// buffer, or the `source` recorded in its view metadata (always a root,
    /// since views are flattened at creation).
    fn root_of(&self, vid: ValueId) -> ValueId {
        match self.views.get(&vid) {
            Some(v) => v.source,
            None => vid,
        }
    }

    /// Zero-copy hand-off of a top-level graph output: move the produced host
    /// buffer straight into the returned tensor instead of copying it to host
    /// and re-allocating it in [`Tensor::from_raw`]. This eliminates two full
    /// per-output memcpys on the decode hot path (the growing KV-cache present
    /// outputs re-materialized every step) while being numerically identical —
    /// the tensor keeps the exact bytes the kernel wrote.
    ///
    /// Returns `None` (caller falls back to the copy path) unless every safety
    /// precondition holds: the value is an owned, host-resident, exactly-sized
    /// producer output that is not a view/sequence element, not pinned as a live
    /// view source, not shared, and not listed as a graph output more than once.
    /// Moving the buffer out forfeits this value's cross-run allocation reuse, so
    /// its `buffer_shapes` entry is cleared to force a fresh allocation next run.
    fn try_move_host_output(
        &mut self,
        vid: ValueId,
        shape: &[usize],
        dtype: DataType,
    ) -> Result<Option<Tensor>> {
        // Values the copy path materializes specially (strided gather, shared
        // sequence element, in-place share, or a pinned live view source) must
        // not have their backing buffer stolen.
        if self.views.contains_key(&vid)
            || self.seq_elem_values.contains_key(&vid)
            || self.shared_buffers.contains_key(&vid)
            || self.pinned.contains(&vid)
        {
            return Ok(None);
        }
        // Only a produced value owns a writable buffer. A producer-less output
        // (initializer or graph-input passthrough) may alias read-only mmap or
        // foreign/borrowed memory that a tensor must never free.
        if self
            .graph
            .try_value(vid)
            .is_none_or(|value| value.producer.is_none())
        {
            return Ok(None);
        }
        // A value produced by a memoized loop-invariant `If` is served on later
        // steps directly from its resident buffer (the branch is skipped, see
        // `exec_if`). Moving that buffer out would leave the next memoized skip
        // handing back freed/reallocated memory, so fall back to the copy path
        // and keep the produced buffer resident for reuse.
        if let Some(producer) = self.graph.try_value(vid).and_then(|value| value.producer)
            && self.if_last_predicate.contains_key(&producer)
        {
            return Ok(None);
        }
        // A value listed as a graph output more than once would be taken twice.
        if self.graph.outputs.iter().filter(|&&o| o == vid).count() != 1 {
            return Ok(None);
        }
        let value_name = || format!("value#{}", vid.0);
        let numel = checked_numel(shape, value_name)?;
        let n = checked_storage_bytes(dtype, numel, value_name, shape)?;
        let movable = self.buffers.get(&vid).is_some_and(|buf| {
            buf.device().is_host_accessible() && !buf.is_borrowed() && buf.len() == n
        });
        if !movable {
            return Ok(None);
        }
        let buffer = self
            .buffers
            .remove(&vid)
            .expect("buffer presence checked above");
        // The buffer now belongs to the tensor; force a fresh allocation on the
        // next run instead of the reuse fast path (which assumes it is present).
        self.buffer_shapes.remove(&vid);
        Ok(Some(Tensor::from_owned_buffer(
            self.ep.clone(),
            dtype,
            shape.to_vec(),
            buffer,
        )))
    }

    /// Contiguous row-major bytes of `vid` for `shape`/`dtype`, materializing a
    /// view (strided gather over its source buffer) or truncating an owned
    /// buffer to its logical size. This is the single materialization seam used
    /// by the graph-output boundary and control-flow scope capture.
    fn contiguous_bytes(&self, vid: ValueId, shape: &[usize], dtype: DataType) -> Result<Vec<u8>> {
        let value_name = || {
            self.graph
                .try_value(vid)
                .and_then(|value| value.name.clone())
                .unwrap_or_else(|| format!("value#{}", vid.0))
        };
        let numel = checked_numel(shape, value_name)?;
        let n = checked_storage_bytes(dtype, numel, value_name, shape)?;
        // A tensor value backed by a shared sequence element (SequenceAt output)
        // owns no buffer; its bytes are the element's contiguous bytes. This is
        // the one materialization point where they are copied out (the boundary
        // back into owned tensors); the compute path reads them zero-copy.
        if let Some(elem) = self.seq_elem_values.get(&vid) {
            let bytes = elem.contiguous_bytes().map_err(SessionError::from)?;
            return Ok(bytes[..n.min(bytes.len())].to_vec());
        }
        if let Some(view) = self.views.get(&vid) {
            let buf = self.buffers.get(&view.source).ok_or_else(|| {
                SessionError::Internal(format!(
                    "view value#{} aliases missing source buffer value#{}",
                    vid.0, view.source.0
                ))
            })?;
            let esize = dtype.byte_size();
            if esize == 0 {
                // Sub-byte views are never created (Slice falls back to copy),
                // so reaching here is an internal invariant violation.
                return Err(SessionError::Internal(format!(
                    "cannot materialize sub-byte view value#{}",
                    vid.0
                )));
            }
            let mut host = vec![0u8; buf.len()];
            self.ep.copy_to_host(buf, &mut host)?;
            Ok(gather_view(
                &host,
                &view.shape,
                &view.strides,
                view.byte_offset,
                esize,
            ))
        } else {
            let buf = self
                .buffers
                .get(&vid)
                .ok_or_else(|| SessionError::Internal(format!("value#{} not produced", vid.0)))?;
            let mut host = vec![0u8; n];
            self.ep.copy_to_host(buf, &mut host)?;
            Ok(host)
        }
    }

    /// Store a control-flow op's produced output `tensor` into this graph's
    /// output value `vid`: (re)size the backing buffer, copy the bytes, and
    /// record the runtime dtype/shape so the caller (and the final output
    /// collection) reads them back correctly. Control-flow output shapes are
    /// data-dependent (the loader never inferred inside the body), so they are
    /// resolved here, exactly as the JIT data-dependent path does for kernels.
    fn store_output_tensor(
        &mut self,
        vid: ValueId,
        tensor: &Tensor,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
    ) -> Result<()> {
        self.store_output_bytes(
            vid,
            tensor.dtype,
            tensor.shape.clone(),
            tensor.as_bytes(),
            resolved,
        )
    }

    fn store_output_bytes(
        &mut self,
        vid: ValueId,
        dtype: DataType,
        dims: Vec<usize>,
        bytes: &[u8],
        resolved: &mut HashMap<ValueId, Vec<usize>>,
    ) -> Result<()> {
        let numel = checked_numel(&dims, || format!("value#{}", vid.0))?;
        let need =
            checked_storage_bytes(dtype, numel, || format!("value#{}", vid.0), &dims)?.max(1);
        let fits = self
            .buffers
            .get(&vid)
            .map(|b| b.len() == need)
            .unwrap_or(false);
        if !fits {
            if let Some(old) = self.buffers.remove(&vid) {
                self.ep.deallocate(old)?;
            }
            self.shared_buffers.remove(&vid);
            let buf = self
                .ep
                .allocate(need, TensorLayout::contiguous().alignment)?;
            self.buffers.insert(vid, buf);
        }
        let buf = self.buffers.get_mut(&vid).expect("just ensured");
        self.ep.copy_from_host(bytes, buf)?;
        self.value_dtypes.insert(vid, dtype);
        self.buffer_shapes.insert(vid, dims.clone());
        resolved.insert(vid, dims);
        Ok(())
    }

    /// Prepare one selected control-flow subgraph and materialize only the free
    /// variables that body actually captures. This avoids copying every named
    /// value in the enclosing graph and, for Loop/Scan, keeps captures stable
    /// across all iterations.
    fn prepare_subgraph(
        &self,
        node_id: NodeId,
        attr_key: &str,
        resolved: &HashMap<ValueId, Vec<usize>>,
        outer_scope: &HashMap<String, Tensor>,
    ) -> Result<PreparedSubgraph> {
        let key = (node_id, attr_key.to_string());
        let body = self.graph.subgraphs.get(&key).ok_or_else(|| {
            SessionError::Internal(format!(
                "control-flow node #{} references missing subgraph '{attr_key}'",
                node_id.0
            ))
        })?;

        let mut scope_names = required_outer_names(body).into_iter().collect::<Vec<_>>();
        scope_names.sort();
        let mut captures = HashMap::with_capacity(scope_names.len());
        for name in scope_names {
            let tensor = if let Some(&vid) = self.name_index.get(&name) {
                let materialized = self.buffers.contains_key(&vid)
                    || self.views.contains_key(&vid)
                    || self.seq_elem_values.contains_key(&vid);
                if resolved.contains_key(&vid) && materialized {
                    self.value_tensor(vid, resolved)?
                } else {
                    outer_scope
                        .get(&name)
                        .cloned()
                        .ok_or_else(|| missing_capture_error(attr_key, &name))?
                }
            } else {
                outer_scope
                    .get(&name)
                    .cloned()
                    .ok_or_else(|| missing_capture_error(attr_key, &name))?
            };
            captures.insert(name, tensor);
        }

        Ok(PreparedSubgraph { key, captures })
    }

    /// Run a prepared control-flow body with changing formal inputs. Captures and
    /// signature metadata are reused; only a concrete shape change rebuilds the
    /// child executor.
    fn run_subgraph(
        &mut self,
        prepared: &PreparedSubgraph,
        formal_inputs: &[&Tensor],
    ) -> Result<Vec<Tensor>> {
        if !self.subgraph_execs.contains_key(&prepared.key) {
            let body = self
                .graph
                .subgraphs
                .get(&prepared.key)
                .cloned()
                .ok_or_else(|| {
                    SessionError::Internal(format!(
                        "control-flow node #{} has no registered subgraph '{}'",
                        prepared.key.0.0, prepared.key.1
                    ))
                })?;
            let mut child = ChildExecutor::new(
                format!("node#{}/{}", prepared.key.0.0, prepared.key.1),
                body,
                self.graph.opset_imports.clone(),
                self.weights.clone(),
                self.ep.clone(),
            )?;
            child.set_trace_context(self.trace.clone());
            self.subgraph_execs.insert(prepared.key.clone(), child);
        }

        let child = self
            .subgraph_execs
            .get_mut(&prepared.key)
            .expect("child present");
        let before = child.stats();
        let result = child.run(formal_inputs, &prepared.captures);
        let after = child.stats();
        self.control_flow_stats.subgraph_builds += after.builds - before.builds;
        self.control_flow_stats.subgraph_runs += after.runs - before.runs;
        result
    }

    /// Dispatch a control-flow plan node to its op-specific handler.
    fn exec_control_flow(
        &mut self,
        pi: usize,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        outer_scope: &HashMap<String, Tensor>,
    ) -> Result<()> {
        let node = self.graph.node(self.plan[pi].node_id).clone();
        match node.op_type.as_str() {
            "If" => self.exec_if(&node, resolved, outer_scope),
            "Loop" => self.exec_loop(&node, resolved, outer_scope),
            "Scan" => self.exec_scan(&node, resolved, outer_scope),
            other => Err(SessionError::Internal(format!(
                "exec_control_flow reached non-control-flow op {other:?}"
            ))),
        }
    }

    /// ONNX `If`: read the scalar `cond`, execute exactly one branch subgraph
    /// (0 formal inputs), and route the branch's outputs to `If`'s outputs.
    fn exec_if(
        &mut self,
        node: &Node,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        outer_scope: &HashMap<String, Tensor>,
    ) -> Result<()> {
        {
            let then_branch = self
                .graph
                .subgraphs
                .get(&(node.id, "then_branch".to_string()))
                .ok_or_else(|| SessionError::ControlFlow {
                    op: "If".to_string(),
                    reason: "missing required 'then_branch' subgraph".to_string(),
                })?;
            let else_branch = self
                .graph
                .subgraphs
                .get(&(node.id, "else_branch".to_string()))
                .ok_or_else(|| SessionError::ControlFlow {
                    op: "If".to_string(),
                    reason: "missing required 'else_branch' subgraph".to_string(),
                })?;

            if !then_branch.inputs.is_empty() || !else_branch.inputs.is_empty() {
                return Err(SessionError::ControlFlow {
                    op: "If".to_string(),
                    reason: format!(
                        "branch subgraphs must declare zero formal inputs, but then_branch has {} \
                         and else_branch has {}",
                        then_branch.inputs.len(),
                        else_branch.inputs.len()
                    ),
                });
            }
            validate_if_branch_outputs(&self.graph, node)?;
        }

        let cond_vid =
            node.inputs
                .first()
                .and_then(|s| *s)
                .ok_or_else(|| SessionError::ControlFlow {
                    op: "If".to_string(),
                    reason: "missing required 'cond' input".to_string(),
                })?;
        let cond_t = self.value_tensor(cond_vid, resolved)?;
        if cond_t.dtype != DataType::Bool {
            return Err(SessionError::DtypeMismatch {
                name: "If cond".to_string(),
                expected: format!("{:?}", DataType::Bool),
                got: format!("{:?}", cond_t.dtype),
            });
        }
        let cond = tensor_scalar_bool(&cond_t).ok_or_else(|| SessionError::ControlFlow {
            op: "If".to_string(),
            reason: format!(
                "'cond' must be a BOOL scalar or single-element tensor, got shape {:?}",
                cond_t.shape
            ),
        })?;

        // Capture-safe loop-invariant control-flow specialization. The predicate
        // is read every step (above) so a genuine branch flip is never missed.
        // When it matches the last observed value AND that value was recorded
        // only for a branch with *no outer captures* (so its outputs depend on
        // nothing but its own constants/initializers and are therefore invariant
        // across decode steps) AND those outputs are still resident in their
        // persistent buffers, re-running the branch is pure waste — skip it. The
        // downstream captured segment reads the unchanged buffers correctly. A
        // branch that reads loop-varying outer values is never memoized, so a
        // stale output is impossible.
        if self.if_last_predicate.get(&node.id) == Some(&cond)
            && node.outputs.iter().all(|v| resolved.contains_key(v))
        {
            return Ok(());
        }

        let attr_key = if cond { "then_branch" } else { "else_branch" };
        // A branch with outer captures may depend on values that change between
        // steps, so its output is not loop-invariant and must never be memoized.
        let taken_branch_is_invariant = self
            .graph
            .subgraphs
            .get(&(node.id, attr_key.to_string()))
            .map(|body| required_outer_names(body).is_empty())
            .unwrap_or(false);
        let prepared = {
            let _s = phase_span!("execif.prepare_subgraph");
            self.prepare_subgraph(node.id, attr_key, resolved, outer_scope)?
        };
        let outs = {
            let _s = phase_span!("execif.run_subgraph");
            self.run_subgraph(&prepared, &[])?
        };

        if outs.len() != node.outputs.len() {
            return Err(SessionError::OutputShapeCountMismatch {
                op: format!("If/{attr_key}"),
                expected: node.outputs.len(),
                got: outs.len(),
            });
        }
        {
            let _s = phase_span!("execif.store_output");
            for (vid, t) in node.outputs.iter().zip(outs.iter()) {
                self.store_output_tensor(*vid, t, resolved)?;
            }
        }
        // Only enable future skips when the taken branch is loop-invariant.
        // Otherwise drop any stale memo so this `If` always re-runs.
        if taken_branch_is_invariant {
            self.if_last_predicate.insert(node.id, cond);
        } else {
            self.if_last_predicate.remove(&node.id);
        }
        Ok(())
    }

    /// Validate a Loop body's positional contract before the first iteration and
    /// retain each scan output's element type/shape for the zero-iteration case.
    fn loop_body_scan_specs(
        &self,
        node: &Node,
        carried: &[Tensor],
        num_scan: usize,
        resolved: &HashMap<ValueId, Vec<usize>>,
    ) -> Result<OptionalTensorSpecs> {
        let body = self
            .graph
            .subgraphs
            .get(&(node.id, "body".to_string()))
            .ok_or_else(|| SessionError::ControlFlow {
                op: "Loop".to_string(),
                reason: "missing required 'body' subgraph".to_string(),
            })?;
        let expected_inputs = 2 + carried.len();
        if body.inputs.len() != expected_inputs {
            return Err(SessionError::ControlFlow {
                op: "Loop".to_string(),
                reason: format!(
                    "body declares {} formal input(s), expected {expected_inputs}",
                    body.inputs.len()
                ),
            });
        }
        let expected_outputs = 1 + carried.len() + num_scan;
        if body.outputs.len() != expected_outputs {
            return Err(SessionError::ControlFlow {
                op: "Loop".to_string(),
                reason: format!(
                    "body declares {} output(s), expected {expected_outputs}",
                    body.outputs.len()
                ),
            });
        }

        for (index, expected) in [(0, DataType::Int64), (1, DataType::Bool)] {
            let input = body.inputs[index];
            if body.value_type_is_known(input) && body.value(input).dtype != expected {
                return Err(SessionError::ControlFlow {
                    op: "Loop".to_string(),
                    reason: format!(
                        "body formal input {index} must be {expected:?}, got {:?}",
                        body.value(input).dtype
                    ),
                });
            }
        }
        let cond_out = body.outputs[0];
        if body.value_type_is_known(cond_out) && body.value(cond_out).dtype != DataType::Bool {
            return Err(SessionError::ControlFlow {
                op: "Loop".to_string(),
                reason: format!(
                    "body output 0 ('cond_out') must be Bool, got {:?}",
                    body.value(cond_out).dtype
                ),
            });
        }

        for (index, initial) in carried.iter().enumerate() {
            for (kind, value) in [
                ("formal input", body.inputs[2 + index]),
                ("output", body.outputs[1 + index]),
            ] {
                if body.value_type_is_known(value) && body.value(value).dtype != initial.dtype {
                    return Err(SessionError::ControlFlow {
                        op: "Loop".to_string(),
                        reason: format!(
                            "loop-carried {kind} {index} has dtype {:?}, but its initial value has \
                             dtype {:?}",
                            body.value(value).dtype,
                            initial.dtype
                        ),
                    });
                }
            }
        }

        body.outputs
            .iter()
            .skip(1 + carried.len())
            .zip(node.outputs.iter().skip(carried.len()))
            .enumerate()
            .map(|(index, (&body_output, &node_output))| {
                let body_value = body.value(body_output);
                let node_dtype = self.value_dtypes[&node_output];
                let dtype = if body.value_type_is_known(body_output) {
                    if self.graph.value_type_is_known(node_output)
                        && body_value.dtype != node_dtype
                    {
                        return Err(SessionError::ControlFlow {
                            op: "Loop".to_string(),
                            reason: format!(
                                "scan output {index} has body dtype {:?}, but the Loop node declares \
                                 {node_dtype:?}",
                                body_value.dtype
                            ),
                        });
                    }
                    body_value.dtype
                } else {
                    node_dtype
                };
                let elem_shape = body
                    .value_shape_is_known(body_output)
                    .then(|| as_static_shape(&body_value.shape))
                    .flatten()
                    .or_else(|| {
                        resolved
                            .get(&node_output)
                            .and_then(|shape| shape.get(1..).map(<[_]>::to_vec))
                    });
                Ok(elem_shape.map(|shape| (dtype, shape)))
            })
            .collect()
    }

    /// ONNX `Loop`: inputs `[M?, cond?, v_initial...]`, body signature
    /// `(iter_num, cond_in, carried...) -> (cond_out, carried..., scan_out...)`.
    /// Iterates while `cond` is true and `iter < M`, threading loop-carried
    /// values across iterations and stacking each scan output along a new
    /// leading iteration axis.
    fn exec_loop(
        &mut self,
        node: &Node,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        outer_scope: &HashMap<String, Tensor>,
    ) -> Result<()> {
        // Inputs: [M, cond, v_initial...]. M and cond may be omitted (None slot)
        // or an empty-name optional; absence means "unbounded" / "true".
        let m: Option<i64> = match node.inputs.first().and_then(|s| *s) {
            Some(vid) => {
                let t = self.value_tensor(vid, resolved)?;
                if t.dtype != DataType::Int64 {
                    return Err(SessionError::DtypeMismatch {
                        name: "Loop M".to_string(),
                        expected: format!("{:?}", DataType::Int64),
                        got: format!("{:?}", t.dtype),
                    });
                }
                let m = tensor_scalar_i64(&t).ok_or_else(|| SessionError::ControlFlow {
                    op: "Loop".to_string(),
                    reason: format!(
                        "'M' must be an INT64 scalar or single-element tensor, got shape {:?}",
                        t.shape
                    ),
                })?;
                Some(m)
            }
            None => None,
        };
        let mut cond: Option<bool> =
            match node.inputs.get(1).and_then(|s| *s) {
                Some(vid) => {
                    let t = self.value_tensor(vid, resolved)?;
                    if t.dtype != DataType::Bool {
                        return Err(SessionError::DtypeMismatch {
                            name: "Loop cond".to_string(),
                            expected: format!("{:?}", DataType::Bool),
                            got: format!("{:?}", t.dtype),
                        });
                    }
                    Some(tensor_scalar_bool(&t).ok_or_else(|| SessionError::ControlFlow {
                    op: "Loop".to_string(),
                    reason: format!(
                        "'cond' must be a BOOL scalar or single-element tensor, got shape {:?}",
                        t.shape
                    ),
                })?)
                }
                None => None,
            };

        // Initial loop-carried dependencies (inputs after M and cond).
        let mut carried: Vec<Tensor> = Vec::new();
        for slot in node.inputs.iter().skip(2) {
            let vid = slot.ok_or_else(|| {
                SessionError::Internal(
                    "Loop: an interior loop-carried input is omitted (empty), which ONNX does not \
                 allow — every v_initial must be provided"
                        .to_string(),
                )
            })?;
            carried.push(self.value_tensor(vid, resolved)?);
        }
        let num_carried = carried.len();
        let carried_invariants: Vec<(DataType, Vec<usize>)> = carried
            .iter()
            .map(|tensor| (tensor.dtype, tensor.shape.clone()))
            .collect();
        // Loop outputs = carried finals ++ scan outputs. Scan-output count is
        // whatever remains after the carried finals.
        let num_outputs = node.outputs.len();
        if num_outputs < num_carried {
            return Err(SessionError::Internal(format!(
                "Loop: node declares {num_outputs} output(s) but has {num_carried} loop-carried \
                 dependency(ies); outputs must be carried-finals followed by scan-outputs"
            )));
        }
        let num_scan = num_outputs - num_carried;
        let empty_scan_specs = self.loop_body_scan_specs(node, &carried, num_scan, resolved)?;
        let mut scan_acc: Vec<TensorStackAccumulator> = (0..num_scan)
            .map(|_| TensorStackAccumulator::new())
            .collect();
        let prepared = self.prepare_subgraph(node.id, "body", resolved, outer_scope)?;
        let mut iter_tensor = scalar_i64_tensor(0)?;
        let mut cond_tensor = scalar_bool_tensor(cond.unwrap_or(true))?;

        let mut iter: i64 = 0;
        loop {
            if let Some(m) = m
                && iter >= m
            {
                break;
            }
            if cond == Some(false) {
                break;
            }

            iter_tensor.overwrite_bytes(&iter.to_le_bytes())?;
            cond_tensor.overwrite_bytes(&[u8::from(cond.unwrap_or(true))])?;
            let mut formal: Vec<&Tensor> = Vec::with_capacity(2 + num_carried);
            formal.push(&iter_tensor);
            formal.push(&cond_tensor);
            formal.extend(carried.iter());

            let outs = self.run_subgraph(&prepared, &formal)?;
            drop(formal);
            // Body outputs: cond_out, carried..., scan_out...
            let expected = 1 + num_carried + num_scan;
            if outs.len() != expected {
                return Err(SessionError::OutputShapeCountMismatch {
                    op: "Loop/body".to_string(),
                    expected,
                    got: outs.len(),
                });
            }
            let mut it = outs.into_iter();
            let cond_out = it.next().expect("cond_out present");
            cond = Some(tensor_scalar_bool(&cond_out).ok_or_else(|| {
                SessionError::Internal(format!(
                    "Loop: body's first output 'cond_out' must be a BOOL scalar, got dtype {:?}",
                    cond_out.dtype
                ))
            })?);
            let next_carried: Vec<Tensor> = (&mut it).take(num_carried).collect();
            for (index, (tensor, (expected_dtype, expected_shape))) in
                next_carried.iter().zip(&carried_invariants).enumerate()
            {
                if tensor.dtype != *expected_dtype {
                    return Err(SessionError::ControlFlow {
                        op: "Loop".to_string(),
                        reason: format!(
                            "loop-carried output {index} dtype mismatch: expected \
                             {expected_dtype:?}, got {:?}",
                            tensor.dtype
                        ),
                    });
                }
                if tensor.shape != *expected_shape {
                    return Err(SessionError::ControlFlow {
                        op: "Loop".to_string(),
                        reason: format!(
                            "loop-carried output {index} shape mismatch: expected \
                             {expected_shape:?}, got {:?}",
                            tensor.shape
                        ),
                    });
                }
            }
            carried = next_carried;
            for acc in scan_acc.iter_mut() {
                acc.push(it.next().expect("scan output present"))?;
            }

            iter = iter
                .checked_add(1)
                .ok_or_else(|| SessionError::ControlFlow {
                    op: "Loop".to_string(),
                    reason: "iteration counter overflowed INT64".to_string(),
                })?;
        }

        // Emit outputs: carried finals, then stacked scan outputs.
        for (i, t) in carried.iter().enumerate() {
            self.store_output_tensor(node.outputs[i], t, resolved)?;
        }
        for (s, (acc, empty_spec)) in scan_acc.into_iter().zip(empty_scan_specs).enumerate() {
            let (dtype, shape, bytes) = acc.finish_with_empty(empty_spec, s)?;
            self.store_output_bytes(
                node.outputs[num_carried + s],
                dtype,
                shape,
                &bytes,
                resolved,
            )?;
        }
        Ok(())
    }

    fn scan_body_specs(
        &self,
        node: &Node,
        state: &[Tensor],
        scan_inputs: &[Tensor],
        input_axes: &[usize],
        num_scan_outputs: usize,
        output_axes: &[i64],
        resolved: &HashMap<ValueId, Vec<usize>>,
    ) -> Result<OptionalTensorSpecs> {
        let body = self
            .graph
            .subgraphs
            .get(&(node.id, "body".to_string()))
            .ok_or_else(|| SessionError::ControlFlow {
                op: "Scan".to_string(),
                reason: "missing required 'body' subgraph".to_string(),
            })?;
        let expected_inputs = state.len() + scan_inputs.len();
        if body.inputs.len() != expected_inputs {
            return Err(SessionError::ControlFlow {
                op: "Scan".to_string(),
                reason: format!(
                    "body declares {} formal input(s), expected {expected_inputs}",
                    body.inputs.len()
                ),
            });
        }
        let expected_outputs = state.len() + num_scan_outputs;
        if body.outputs.len() != expected_outputs {
            return Err(SessionError::ControlFlow {
                op: "Scan".to_string(),
                reason: format!(
                    "body declares {} output(s), expected {expected_outputs}",
                    body.outputs.len()
                ),
            });
        }

        for (index, initial) in state.iter().enumerate() {
            for (kind, value) in [
                ("formal input", body.inputs[index]),
                ("output", body.outputs[index]),
            ] {
                if body.value_type_is_known(value) && body.value(value).dtype != initial.dtype {
                    return Err(SessionError::ControlFlow {
                        op: "Scan".to_string(),
                        reason: format!(
                            "state {kind} {index} has dtype {:?}, but its initial value has dtype {:?}",
                            body.value(value).dtype,
                            initial.dtype
                        ),
                    });
                }
            }
        }
        for (index, ((input, &axis), &formal)) in scan_inputs
            .iter()
            .zip(input_axes)
            .zip(body.inputs.iter().skip(state.len()))
            .enumerate()
        {
            if body.value_type_is_known(formal) && body.value(formal).dtype != input.dtype {
                return Err(SessionError::ControlFlow {
                    op: "Scan".to_string(),
                    reason: format!(
                        "scan formal input {index} has dtype {:?}, but scan input {index} has dtype {:?}",
                        body.value(formal).dtype,
                        input.dtype
                    ),
                });
            }
            let mut slice_shape = input.shape.clone();
            slice_shape.remove(axis);
            if body.value_shape_is_known(formal)
                && let Some(shape) = as_static_shape(&body.value(formal).shape)
                && shape != slice_shape
            {
                return Err(SessionError::ControlFlow {
                    op: "Scan".to_string(),
                    reason: format!(
                        "scan formal input {index} has shape {shape:?}, but slicing input shape {:?} \
                         along axis {axis} produces {slice_shape:?}",
                        input.shape
                    ),
                });
            }
        }

        body.outputs
            .iter()
            .skip(state.len())
            .zip(node.outputs.iter().skip(state.len()))
            .zip(output_axes)
            .enumerate()
            .map(|(index, ((&body_output, &node_output), &axis))| {
                let body_value = body.value(body_output);
                let node_dtype = self.value_dtypes[&node_output];
                let dtype = if body.value_type_is_known(body_output) {
                    if self.graph.value_type_is_known(node_output)
                        && body_value.dtype != node_dtype
                    {
                        return Err(SessionError::ControlFlow {
                            op: "Scan".to_string(),
                            reason: format!(
                                "scan output {index} has body dtype {:?}, but the Scan node declares \
                                 {node_dtype:?}",
                                body_value.dtype
                            ),
                        });
                    }
                    body_value.dtype
                } else {
                    node_dtype
                };
                let elem_shape = body
                    .value_shape_is_known(body_output)
                    .then(|| as_static_shape(&body_value.shape))
                    .flatten()
                    .or_else(|| {
                        resolved.get(&node_output).and_then(|shape| {
                            normalize_axis(axis, shape.len()).map(|axis| {
                                let mut elem_shape = shape.clone();
                                elem_shape.remove(axis);
                                elem_shape
                            })
                        })
                    });
                if let Some(shape) = &elem_shape
                    && normalize_axis(axis, shape.len() + 1).is_none()
                {
                    return Err(SessionError::ControlFlow {
                        op: "Scan".to_string(),
                        reason: format!(
                            "scan_output_axes[{index}]={axis} is out of range for output rank {}",
                            shape.len() + 1
                        ),
                    });
                }
                Ok(elem_shape.map(|shape| (dtype, shape)))
            })
            .collect()
    }

    /// ONNX `Scan`: slice configured input axes/directions, thread invariant
    /// state through the body, and stack scan outputs on configured axes.
    fn exec_scan(
        &mut self,
        node: &Node,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        outer_scope: &HashMap<String, Tensor>,
    ) -> Result<()> {
        let raw_num_scan_inputs = node
            .attr("num_scan_inputs")
            .and_then(|a| a.as_int())
            .ok_or_else(|| SessionError::ControlFlow {
                op: "Scan".to_string(),
                reason: "required attribute 'num_scan_inputs' is missing or not an INT".to_string(),
            })?;
        let num_scan_inputs = usize::try_from(raw_num_scan_inputs)
            .ok()
            .filter(|&count| count != 0)
            .ok_or_else(|| SessionError::ControlFlow {
                op: "Scan".to_string(),
                reason: format!(
                    "'num_scan_inputs' must be a positive INT, got {raw_num_scan_inputs}"
                ),
            })?;

        let total_inputs = node.inputs.len();
        if total_inputs < num_scan_inputs {
            return Err(SessionError::ControlFlow {
                op: "Scan".to_string(),
                reason: format!(
                    "node has {total_inputs} input(s) but num_scan_inputs={num_scan_inputs}"
                ),
            });
        }
        let num_state = total_inputs - num_scan_inputs;
        if node.outputs.len() < num_state {
            return Err(SessionError::ControlFlow {
                op: "Scan".to_string(),
                reason: format!(
                    "declares {} output(s) but has {num_state} state variable(s)",
                    node.outputs.len()
                ),
            });
        }
        let num_scan_outputs = node.outputs.len() - num_state;
        let input_axes_raw = scan_list_attr(node, "scan_input_axes", num_scan_inputs, 0)?;
        let input_directions = scan_list_attr(node, "scan_input_directions", num_scan_inputs, 0)?;
        let output_axes = scan_list_attr(node, "scan_output_axes", num_scan_outputs, 0)?;
        let output_directions =
            scan_list_attr(node, "scan_output_directions", num_scan_outputs, 0)?;
        for (name, values) in [
            ("scan_input_directions", &input_directions),
            ("scan_output_directions", &output_directions),
        ] {
            for (index, &value) in values.iter().enumerate() {
                if !matches!(value, 0 | 1) {
                    return Err(SessionError::ControlFlow {
                        op: "Scan".to_string(),
                        reason: format!(
                            "{name}[{index}] must be 0 (forward) or 1 (reverse), got {value}"
                        ),
                    });
                }
            }
        }

        let mut state: Vec<Tensor> = Vec::with_capacity(num_state);
        for slot in node.inputs.iter().take(num_state) {
            let vid = slot.ok_or_else(|| SessionError::ControlFlow {
                op: "Scan".to_string(),
                reason: "an initial-state input is omitted (empty), which ONNX does not allow"
                    .to_string(),
            })?;
            state.push(self.value_tensor(vid, resolved)?);
        }
        let mut scan_inputs: Vec<Tensor> = Vec::with_capacity(num_scan_inputs);
        for slot in node.inputs.iter().skip(num_state) {
            let vid = slot.ok_or_else(|| SessionError::ControlFlow {
                op: "Scan".to_string(),
                reason: "a scan input is omitted (empty), which ONNX does not allow".to_string(),
            })?;
            scan_inputs.push(self.value_tensor(vid, resolved)?);
        }

        let mut input_axes = Vec::with_capacity(num_scan_inputs);
        for (index, (input, &raw_axis)) in scan_inputs.iter().zip(&input_axes_raw).enumerate() {
            let axis = normalize_axis(raw_axis, input.shape.len()).ok_or_else(|| {
                SessionError::ControlFlow {
                    op: "Scan".to_string(),
                    reason: format!(
                        "scan_input_axes[{index}]={raw_axis} is out of range for input rank {}",
                        input.shape.len()
                    ),
                }
            })?;
            input_axes.push(axis);
        }
        let trip_count = scan_inputs[0].shape[input_axes[0]];
        for (index, (input, &axis)) in scan_inputs.iter().zip(&input_axes).enumerate() {
            let length = input.shape[axis];
            if length != trip_count {
                return Err(SessionError::ControlFlow {
                    op: "Scan".to_string(),
                    reason: format!(
                        "scan input {index} has scan-axis length {length}, but the first scan input \
                         has {trip_count}; all scan inputs must agree"
                    ),
                });
            }
        }

        let state_specs: Vec<(DataType, Vec<usize>)> = state
            .iter()
            .map(|tensor| (tensor.dtype, tensor.shape.clone()))
            .collect();
        let empty_specs = self.scan_body_specs(
            node,
            &state,
            &scan_inputs,
            &input_axes,
            num_scan_outputs,
            &output_axes,
            resolved,
        )?;
        let mut scan_acc: Vec<TensorStackAccumulator> = (0..num_scan_outputs)
            .map(|_| TensorStackAccumulator::new())
            .collect();
        let prepared = self.prepare_subgraph(node.id, "body", resolved, outer_scope)?;
        let mut scan_slices = Vec::with_capacity(num_scan_inputs);
        if trip_count != 0 {
            for (index, ((input, &axis), &direction)) in scan_inputs
                .iter()
                .zip(&input_axes)
                .zip(&input_directions)
                .enumerate()
            {
                let source_index = if direction == 0 { 0 } else { trip_count - 1 };
                let (shape, bytes) = scan_slice(input, axis, source_index, index)?;
                scan_slices.push(Tensor::from_raw(input.dtype, shape, &bytes)?);
            }
        }
        for step in 0..trip_count {
            if step != 0 {
                for (index, (((input, &axis), &direction), slice)) in scan_inputs
                    .iter()
                    .zip(&input_axes)
                    .zip(&input_directions)
                    .zip(scan_slices.iter_mut())
                    .enumerate()
                {
                    let source_index = if direction == 0 {
                        step
                    } else {
                        trip_count - 1 - step
                    };
                    let (_, bytes) = scan_slice(input, axis, source_index, index)?;
                    slice.overwrite_bytes(&bytes)?;
                }
            }
            let mut formal: Vec<&Tensor> = Vec::with_capacity(num_state + num_scan_inputs);
            formal.extend(state.iter());
            formal.extend(scan_slices.iter());

            let outs = self.run_subgraph(&prepared, &formal)?;
            drop(formal);
            let expected = num_state + num_scan_outputs;
            if outs.len() != expected {
                return Err(SessionError::OutputShapeCountMismatch {
                    op: "Scan/body".to_string(),
                    expected,
                    got: outs.len(),
                });
            }
            let mut it = outs.into_iter();
            let next_state: Vec<Tensor> = (&mut it).take(num_state).collect();
            for (index, (tensor, (expected_dtype, expected_shape))) in
                next_state.iter().zip(&state_specs).enumerate()
            {
                if tensor.dtype != *expected_dtype {
                    return Err(SessionError::ControlFlow {
                        op: "Scan".to_string(),
                        reason: format!(
                            "state output {index} dtype mismatch: expected {expected_dtype:?}, got {:?}",
                            tensor.dtype
                        ),
                    });
                }
                if tensor.shape != *expected_shape {
                    return Err(SessionError::ControlFlow {
                        op: "Scan".to_string(),
                        reason: format!(
                            "state output {index} shape mismatch: expected {expected_shape:?}, got {:?}",
                            tensor.shape
                        ),
                    });
                }
            }
            state = next_state;
            for acc in scan_acc.iter_mut() {
                acc.push(it.next().expect("scan output present"))?;
            }
        }

        for (i, t) in state.iter().enumerate() {
            self.store_output_tensor(node.outputs[i], t, resolved)?;
        }
        for (s, ((acc, empty_spec), (&axis, &direction))) in scan_acc
            .into_iter()
            .zip(empty_specs)
            .zip(output_axes.iter().zip(&output_directions))
            .enumerate()
        {
            let (dtype, shape, bytes) = acc.finish_scan(axis, direction, empty_spec, s)?;
            self.store_output_bytes(node.outputs[num_state + s], dtype, shape, &bytes, resolved)?;
        }
        Ok(())
    }
}

fn scan_slice(
    t: &Tensor,
    axis: usize,
    index: usize,
    input_index: usize,
) -> Result<(Vec<usize>, Vec<u8>)> {
    let axis_len = t.shape[axis];
    if index >= axis_len {
        return Err(SessionError::ControlFlow {
            op: "Scan".to_string(),
            reason: format!(
                "slice index {index} is out of range for scan input {input_index} axis {axis}"
            ),
        });
    }
    let esize = t.dtype.byte_size();
    if esize == 0 {
        return Err(SessionError::ControlFlow {
            op: "Scan".to_string(),
            reason: format!(
                "sub-byte dtype {:?} for scan input {input_index} is not supported",
                t.dtype
            ),
        });
    }
    let mut shape = t.shape.clone();
    shape.remove(axis);
    let outer = checked_numel(&t.shape[..axis], || format!("Scan input {input_index}"))?;
    let inner = checked_numel(&t.shape[axis + 1..], || format!("Scan input {input_index}"))?;
    let inner_bytes = checked_storage_bytes(
        t.dtype,
        inner,
        || format!("Scan input {input_index}"),
        &t.shape,
    )?;
    let total_bytes =
        outer
            .checked_mul(inner_bytes)
            .ok_or_else(|| SessionError::ShapeOverflow {
                value: format!("Scan input {input_index} slice"),
                dims: shape.clone(),
            })?;
    let source = t.as_bytes();
    let mut bytes = vec![0u8; total_bytes];
    for outer_index in 0..outer {
        let src = (outer_index * axis_len + index) * inner_bytes;
        let dst = outer_index * inner_bytes;
        bytes[dst..dst + inner_bytes].copy_from_slice(&source[src..src + inner_bytes]);
    }
    Ok((shape, bytes))
}

/// Incremental accumulator for Loop/Scan outputs. Iteration tensors are copied
/// into one byte buffer and dropped; non-leading Scan axes are rearranged once
/// when the final tensor is materialized.
struct TensorStackAccumulator {
    dtype: Option<DataType>,
    elem_shape: Vec<usize>,
    len: usize,
    bytes: Vec<u8>,
}

impl TensorStackAccumulator {
    fn new() -> Self {
        Self {
            dtype: None,
            elem_shape: Vec::new(),
            len: 0,
            bytes: Vec::new(),
        }
    }

    fn push(&mut self, tensor: Tensor) -> Result<()> {
        if let Some(dtype) = self.dtype {
            if tensor.shape != self.elem_shape || tensor.dtype != dtype {
                return Err(SessionError::Internal(format!(
                    "Loop/Scan: scan output slice {} has shape {:?} dtype {:?} but the first slice \
                     is shape {:?} dtype {:?}; every iteration's scan output must match",
                    self.len, tensor.shape, tensor.dtype, self.elem_shape, dtype
                )));
            }
        } else {
            if tensor.dtype.byte_size() == 0 {
                return Err(SessionError::Internal(format!(
                    "Loop/Scan: sub-byte dtype {:?} scan outputs are not supported",
                    tensor.dtype
                )));
            }
            self.dtype = Some(tensor.dtype);
            self.elem_shape = tensor.shape.clone();
        }
        self.bytes.extend_from_slice(tensor.as_bytes());
        self.len += 1;
        Ok(())
    }

    fn finish(self) -> (DataType, Vec<usize>, Vec<u8>) {
        if self.len == 0 {
            return (DataType::Float32, vec![0], Vec::new());
        }
        let dtype = self.dtype.expect("non-empty accumulator has dtype");
        let mut shape = Vec::with_capacity(1 + self.elem_shape.len());
        shape.push(self.len);
        shape.extend(self.elem_shape);
        (dtype, shape, self.bytes)
    }

    fn finish_with_empty(
        self,
        empty_spec: Option<(DataType, Vec<usize>)>,
        output_index: usize,
    ) -> Result<(DataType, Vec<usize>, Vec<u8>)> {
        if self.len != 0 {
            return Ok(self.finish());
        }
        let (dtype, elem_shape) = empty_spec.ok_or_else(|| SessionError::ControlFlow {
            op: "Loop".to_string(),
            reason: format!(
                "cannot determine the element shape of scan output {output_index} for a \
                 zero-iteration result"
            ),
        })?;
        let mut shape = Vec::with_capacity(1 + elem_shape.len());
        shape.push(0);
        shape.extend(elem_shape);
        Ok((dtype, shape, Vec::new()))
    }

    fn finish_scan(
        self,
        axis: i64,
        direction: i64,
        empty_spec: Option<(DataType, Vec<usize>)>,
        output_index: usize,
    ) -> Result<(DataType, Vec<usize>, Vec<u8>)> {
        let (dtype, elem_shape) = match self.dtype {
            Some(dtype) => (dtype, self.elem_shape.clone()),
            None => empty_spec.ok_or_else(|| SessionError::ControlFlow {
                op: "Scan".to_string(),
                reason: format!(
                    "cannot determine the element shape of scan output {output_index} for a \
                     zero-iteration result"
                ),
            })?,
        };
        let output_rank = elem_shape.len() + 1;
        let axis = normalize_axis(axis, output_rank).ok_or_else(|| SessionError::ControlFlow {
            op: "Scan".to_string(),
            reason: format!(
                "scan_output_axes[{output_index}]={axis} is out of range for output rank \
                 {output_rank}"
            ),
        })?;
        if self.len == 0 {
            let mut shape = elem_shape;
            shape.insert(axis, 0);
            return Ok((dtype, shape, Vec::new()));
        }
        if axis == 0 && direction == 0 {
            let mut shape = Vec::with_capacity(output_rank);
            shape.push(self.len);
            shape.extend(elem_shape);
            return Ok((dtype, shape, self.bytes));
        }

        let elem_numel = checked_numel(&elem_shape, || {
            format!("Scan output {output_index} element")
        })?;
        let elem_bytes = checked_storage_bytes(
            dtype,
            elem_numel,
            || format!("Scan output {output_index} element"),
            &elem_shape,
        )?;
        let mut elements: Vec<&[u8]> = if elem_bytes == 0 {
            (0..self.len).map(|_| &self.bytes[..]).collect()
        } else {
            self.bytes.chunks_exact(elem_bytes).collect()
        };
        if direction == 1 {
            elements.reverse();
        }
        let (shape, bytes) = stack_new_axis(&elements, &elem_shape, axis, dtype.byte_size())?;
        Ok((dtype, shape, bytes))
    }
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

/// Instantiate and initialize the Phase-1 CPU execution provider (§20.7,
/// CPU-only auto-detection). A GPU/accelerator EP would be prepended here in a
/// later phase; for Phase 1 the CPU EP is the sole, always-available backend.
pub(crate) fn auto_detect_cpu_ep() -> Result<Arc<dyn ExecutionProvider>> {
    let mut ep = CpuExecutionProvider::new();
    ep.initialize(&Default::default())?;
    Ok(Arc::new(ep))
}

#[cfg(test)]
mod tests;

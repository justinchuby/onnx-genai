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


mod state;
use state::*;
pub(crate) use state::{ChildExecutor, ChildExecutorStats, Executor};
mod dynamic_shapes;
mod geometry;
use dynamic_shapes::*;
use geometry::*;
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

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
    DeviceBuffer, DevicePtr, DevicePtrMut, EpError, ExecutionProvider, ExternalMmapRegion, Kernel,
    KernelInput, KernelMatch, LazyWeight, LazyWeightBoundary, ResidentWeight, TensorBacking,
    TensorMut, TensorView, WeightHandle,
};

type OptionalTensorSpecs = Vec<Option<(DataType, Vec<usize>)>>;
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ep_cpu::strided::view_in_bounds;
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
use onnx_runtime_tracer::{TraceContext, annotate_current_span_with};

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
            Some(id) => eprintln!(
                "[onnx-genai-capture]   seam node {id} ({}::{}) ran eagerly: {}",
                boundary.domain, boundary.op_type, boundary.reason
            ),
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
}

impl CaptureDecline {
    fn node(node_id: NodeId, node: &Node, reason: impl Into<String>) -> Self {
        Self {
            node_id: Some(node_id.0),
            op_type: node.op_type.clone(),
            domain: canonical_domain(node),
            reason: reason.into(),
        }
    }

    fn graph(reason: impl Into<String>) -> Self {
        Self {
            node_id: None,
            op_type: "<graph>".to_string(),
            domain: "nxrt".to_string(),
            reason: reason.into(),
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
    kernel
        .capture_support()
        .reason()
        .map(|reason| CaptureDecline::node(node_id, node, reason))
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
    device_graph_signature: Option<Vec<DeviceBindingSignature>>,
    /// The captured-segment schedule from the most recent successful capture,
    /// reused to interleave segment replays with eager seam nodes on each
    /// subsequent step. `None` when no device graph is installed.
    capture_schedule: Option<CaptureSchedule>,
    /// Structured segment-boundary reasons from the most recent capture, retained
    /// for diagnostics after `capture_schedule` is taken for replay.
    capture_segmentation: Vec<CaptureDecline>,
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
}

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

impl ExternalBindings {
    fn seed_capture_shapes(&self, resolved: &mut HashMap<ValueId, Vec<usize>>) {
        for (&vid, value) in self.inputs.iter().chain(&self.outputs) {
            resolved.entry(vid).or_insert_with(|| value.shape.clone());
        }
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
/// opset for the node's domain. The default ONNX domain is spelled both `""`
/// and `"ai.onnx"`; both map to the same import. Loader and programmatic session
/// entry points validate this invariant before executor construction.
fn effective_opset(graph: &Graph, node: &Node) -> u64 {
    let domain = node.domain.as_str();
    graph
        .opset_imports
        .get(domain)
        .or_else(|| {
            if domain.is_empty() {
                graph.opset_imports.get("ai.onnx")
            } else if domain == "ai.onnx" {
                graph.opset_imports.get("")
            } else {
                None
            }
        })
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

fn kernel_input_uses_physical_capacity(node: &Node, input_index: usize) -> bool {
    // GQA treats the cache tensor extent as capacity and obtains the valid past
    // length from seqlens_k. Standard Attention instead derives past length from
    // the cache tensor extent itself.
    node.domain == "com.microsoft"
        && node.op_type == "GroupQueryAttention"
        && matches!(input_index, 3 | 4)
}

/// Recompute the output shape of standard elementwise broadcasting ops from
/// their concrete runtime inputs. Loader inference is only a prior: a
/// data-dependent upstream value may acquire a different live shape.
fn runtime_elementwise_output_shape(
    node: &Node,
    input_shapes: &[Vec<usize>],
) -> Option<std::result::Result<Vec<usize>, onnx_runtime_ir::IrError>> {
    if !(node.domain.is_empty() || node.domain == "ai.onnx") {
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
    opset: u64,
) -> Option<Vec<Vec<usize>>> {
    match node.op_type.as_str() {
        // Opset-10+ `Slice`: data, starts, ends, [axes], [steps] as inputs. The
        // per-axis element count mirrors the `Slice` kernel's clamp semantics
        // exactly (ONNX reference), so the buffer we size here matches what the
        // kernel writes.
        "Slice" if node.domain.is_empty() || node.domain == "ai.onnx" => {
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
            let domain = if node.domain == "ai.onnx" {
                String::new()
            } else {
                node.domain.clone()
            };
            imports.insert(domain, opset);
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
                && (node.domain.is_empty() || node.domain == "ai.onnx")
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
            || !(mul.domain.is_empty() || mul.domain == "ai.onnx")
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

    let registry = InferenceRegistry::default_registry();
    let opset_imports = graph.opset_imports.clone();
    registry.infer_graph(graph, &opset_imports, MergePolicy::Permissive)?;
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
        fuse_silu_patterns(&mut graph);
        let graph_before_ep_passes = graph.clone();
        run_ep_scoped_passes(&mut graph, &weights, ep.as_ref())?;
        let mut execution_provider_fallback_report = cuda_fallback_report(&graph, ep.as_ref());
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
        // Topological order up front: also validates the selected graph is a DAG.
        let order = graph.topological_order()?;
        let weight_handles = build_lazy_weight_handles(&graph, &weights, ep.as_ref())?;

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
        for (&vid, weight) in &graph.initializers {
            let dtype = weight.dtype();
            let dims = weight.dims().to_vec();
            value_dtypes.insert(vid, dtype);
            value_shapes.insert(vid, dims.iter().map(|&d| Dim::Static(d)).collect());
            if !ep.device_id().is_host_accessible() && weight_handles.contains_key(&vid) {
                continue;
            }
            let bytes = weights.bytes(weight).ok_or_else(|| {
                SessionError::Internal(format!("weight bytes unavailable for value#{}", vid.0))
            })?;
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
                let mut owned = ep.allocate(bytes.len().max(1), init_align)?;
                ep.copy_from_host(bytes, &mut owned)?;
                owned
            };
            buffer_shapes.insert(vid, dims);
            buffers.insert(vid, buf);
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

        // 3) Build the structural per-node plan.
        let mut plan = Vec::with_capacity(order.len());
        for &nid in &order {
            let node = graph.node(nid);
            // EPContext nodes are pre-compiled: they bypass placement and were
            // already restored through their owning EP by the session's
            // consume path (§55.3). They must never be resolved as ordinary
            // kernels — the CPU EP has no `EPContext` kernel — so skip them
            // here.
            if onnx_runtime_loader::is_ep_context_op(&node.op_type, &node.domain) {
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
            device_graph_signature: None,
            capture_schedule: None,
            capture_segmentation: Vec::new(),
            views: HashMap::new(),
            pinned: HashSet::new(),
            sequence_values,
            shared_buffers: HashMap::new(),
            sequences: HashMap::new(),
            seq_elem_values: HashMap::new(),
            execution_provider_fallback_report,
            trace: TraceContext::noop(),
        };

        // 5) Fully-static graphs are materialized eagerly (buffers + the whole
        //    "compiled plan" of kernels), so the first `run` sees only cache
        //    hits. Symbolic graphs cannot be sized until a `run` fixes their
        //    shapes, so their buffers/kernels are created on first use.
        if !exec.has_symbols {
            let empty = HashMap::new();
            let resolved = exec.resolve_all(&empty)?;
            exec.size_buffers(&resolved)?;
            exec.compile_all(&resolved)?;
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
        for i in 0..self.plan.len() {
            let node_id = self.plan[i].node_id;
            let node = self.graph.node(node_id);
            // Control-flow ops (If/Loop/Scan) are not leaf kernels — they execute
            // nested subgraphs through the executor's own path, so they have no
            // entry in the EP kernel registry and must not be compiled here.
            if is_control_flow_op(&node.op_type, &node.domain) {
                continue;
            }
            // Sequence ops are executor-handled (they operate on sequence-of-
            // tensor values, not tensor views) — they have no EP kernel and must
            // not be compiled here, exactly like control-flow ops.
            if is_sequence_op(&node.op_type, &node.domain) {
                continue;
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
        let expose_logical_input_shape = output_name.is_some()
            && self
                .input_index
                .get(&input_name)
                .is_none_or(|&vid| !self.binding_consumers_use_physical_capacity(vid));
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

    pub(crate) fn replay_device_graph(&mut self, bindings: &mut [DeviceIoBinding]) -> Result<()> {
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
            return Ok(());
        }
        match self.run_scoped_mode(&[], &HashMap::new(), &external, RunMode::Replay)? {
            ScopedRunResult::Executed(_) => Ok(()),
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
        let mut resolved = self.resolve_soft(&bindings);
        if mode != RunMode::Eager {
            // Persistent bindings seed the kernel-visible geometry selected by
            // their input/output contracts. Seed only unresolved values:
            // statically/symbolically resolved shapes remain authoritative.
            external.seed_capture_shapes(&mut resolved);
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
        self.size_buffers_excluding(&resolved, &external_values)?;

        // --- Bind input bytes into their (now correctly sized) buffers ------
        for (name, tensor) in inputs {
            let vid = self.input_index[*name];
            let buf = self
                .buffers
                .get_mut(&vid)
                .expect("input value has a buffer");
            self.ep.copy_from_host(tensor.as_bytes(), buf)?;
        }

        // --- Execute nodes ---------------------------------------------------
        // Iterate by index so a control-flow node can take `&mut self` (it must
        // build/reuse child executors) while an ordinary kernel node uses the
        // disjoint-field borrow split inside `exec_kernel_node`.
        match mode {
            RunMode::Eager => {
                self.run_plan_eager(&mut resolved, outer_scope, external)?;
            }
            RunMode::Capture => {
                // Partition the claimed subgraph into maximal capturable segments
                // separated by non-capturable seam nodes. Only a graph-level hard
                // decline (e.g. no persistent output binding, or nothing
                // capturable at all) falls back to a fully eager run.
                let schedule = match self.plan_capture_segments(&resolved, external) {
                    Ok(schedule) => schedule,
                    Err(report) => return Ok(ScopedRunResult::NotCapturable(report)),
                };
                if let Err(error) = self.run_plan_segmented(
                    &schedule,
                    RunMode::Capture,
                    &mut resolved,
                    outer_scope,
                    external,
                ) {
                    let _ = self.ep.reset_device_graph();
                    self.capture_schedule = None;
                    self.capture_segmentation.clear();
                    return Ok(ScopedRunResult::NotCapturable(CaptureDeclineReport::one(
                        CaptureDecline::graph(format!(
                            "segmented CUDA graph capture failed: {error}"
                        )),
                    )));
                }
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
                let result = self.run_plan_segmented(
                    &schedule,
                    RunMode::Replay,
                    &mut resolved,
                    outer_scope,
                    external,
                );
                self.capture_schedule = Some(schedule);
                result?;
            }
        }

        // --- Collect graph outputs into owned tensors -----------------------
        // A view output (a layout op whose result aliases an input buffer) is
        // materialized to contiguous owned bytes here — external consumers and
        // the Python/DLPack boundary expect contiguous tensors.
        let mut results = Vec::with_capacity(self.graph.outputs.len());
        for &vid in &self.graph.outputs {
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
            let bytes = self.contiguous_bytes(vid, &shape, dtype)?;
            results.push(Some(SessionOutput::Tensor(Tensor::from_raw(
                dtype, shape, &bytes,
            )?)));
        }
        Ok(ScopedRunResult::Executed(results))
    }

    /// Classify why one plan node cannot be recorded into a device graph, or
    /// `None` when it is capturable. Mirrors the per-node predicates the
    /// all-or-nothing audit used, but returns the reason instead of aborting so
    /// the caller can form segments around each non-capturable seam node.
    fn node_capture_reason(
        &self,
        plan: &NodePlan,
        resolved: &HashMap<ValueId, Vec<usize>>,
    ) -> Option<CaptureDecline> {
        let node = self.graph.node(plan.node_id);
        if is_control_flow_op(&node.op_type, &node.domain)
            || is_sequence_op(&node.op_type, &node.domain)
        {
            return Some(CaptureDecline::node(
                plan.node_id,
                node,
                "control-flow and sequence nodes are not device-graph capturable",
            ));
        }
        if plan
            .outputs
            .iter()
            .any(|output| !resolved.contains_key(output))
        {
            return Some(CaptureDecline::node(
                plan.node_id,
                node,
                "data-dependent output shape was unresolved before capture",
            ));
        }
        let Some(input_shapes) = plan
            .inputs
            .iter()
            .map(|input| {
                input
                    .map(|value| resolved.get(&value).cloned())
                    .unwrap_or(Some(Vec::new()))
            })
            .collect::<Option<Vec<_>>>()
        else {
            return Some(CaptureDecline::node(
                plan.node_id,
                node,
                "data-dependent input shape was unresolved before capture",
            ));
        };
        let key = KernelKey {
            node: plan.node_id.0,
            shapes: input_shapes,
        };
        let Some(kernel) = self.cache.entries.get(&key) else {
            return Some(CaptureDecline::node(
                plan.node_id,
                node,
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
        let node = self.graph.node(self.plan[pi].node_id);
        let op_type = node.op_type.clone();
        let domain = node.domain.clone();
        // Open the span only when tracing is live so an untraced decode step
        // never allocates a span name or touches the thread-local span stack.
        let _span = self.trace.is_enabled().then(|| {
            let span = self.trace.span(op_type.clone(), "op");
            // Span is now active on this thread; stamp the capture disposition
            // (and let the kernel below stamp its selected variant).
            capture.annotate();
            span
        });
        if is_control_flow_op(&op_type, &domain) {
            self.exec_control_flow(pi, resolved, outer_scope)
        } else if is_sequence_op(&op_type, &domain) {
            self.exec_sequence_node(pi, resolved, external)
        } else {
            self.exec_kernel_node(pi, resolved, external)
        }
    }

    /// Execute every plan node eagerly on the stream (no capture).
    fn run_plan_eager(
        &mut self,
        resolved: &mut HashMap<ValueId, Vec<usize>>,
        outer_scope: &HashMap<String, Tensor>,
        external: &ExternalBindings,
    ) -> Result<()> {
        if profile_ops_enabled() {
            let run_start = Instant::now();
            let mut timings: HashMap<String, (Duration, usize)> = HashMap::new();
            for pi in 0..self.plan.len() {
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
    ) -> Result<()> {
        let ep = Arc::clone(&self.ep);
        for seg in &schedule.segments {
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
                            self.exec_plan_node(
                                pi,
                                resolved,
                                outer_scope,
                                external,
                                OpCaptureTrace::Captured,
                            )?;
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
                }
            }
        }
        Ok(())
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
        // Small owned copies of the plan facts so the buffer/view/cache fields
        // can be mutated below without fighting a borrow of `self.plan`.
        let node_id = self.plan[pi].node_id;
        let inputs = self.plan[pi].inputs.clone();
        let outputs = self.plan[pi].outputs.clone();
        let input_dtypes = self.plan[pi].input_dtypes.clone();
        let output_dtypes = self.plan[pi].output_dtypes.clone();

        let input_shapes: Vec<Vec<usize>> = inputs
            .iter()
            .map(|v| v.map(|vid| resolved[&vid].clone()).unwrap_or_default())
            .collect();

        let node = self.graph.node(node_id);
        if let Some(output_shape) = runtime_elementwise_output_shape(node, &input_shapes) {
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
                    input_shapes: input_shapes.clone(),
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
            let input_values: Vec<Option<Vec<i64>>> = inputs
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    v.and_then(|vid| self.shape_input_i64(vid, &input_shapes[i], input_dtypes[i]))
                })
                .collect();
            let out_shapes = dynamic_output_shapes(
                node,
                &input_shapes,
                &input_dtypes,
                &input_values,
                effective_opset(&self.graph, node),
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
        let output_shapes: Vec<Vec<usize>> = outputs.iter().map(|v| resolved[v].clone()).collect();
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
            .map(|input| input.is_some_and(|vid| graph.initializers.contains_key(&vid)))
            .collect();
        let kernel = cache.get_or_create(
            node_id,
            node,
            &input_shapes,
            &input_dtypes,
            &constant_inputs,
            opset,
            ep.as_ref(),
        )?;
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
        for &vid in &outputs {
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
        let execution = match &kernel_inputs {
            Some(inputs) => kernel.execute_with_inputs(inputs, &mut outs),
            None => kernel.execute(&views, &mut outs),
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
    (domain.is_empty() || domain == "ai.onnx") && matches!(op_type, "If" | "Loop" | "Scan")
}

/// Whether `(op_type, domain)` is an ONNX **Sequence** op the executor handles
/// directly (default `ai.onnx` domain). Like control-flow ops these are handled
/// at the executor level rather than as leaf [`Kernel`](onnx_runtime_ep_api::Kernel)s
/// because a `Kernel` sees only tensor views, never a *sequence-of-tensors*
/// runtime value. Kept as a small self-contained routing predicate (mirroring
/// [`is_control_flow_op`]) so it never collides with the EP kernel registry.
fn is_sequence_op(op_type: &str, domain: &str) -> bool {
    (domain.is_empty() || domain == "ai.onnx")
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
    (domain.is_empty() || domain == "ai.onnx")
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

        let attr_key = if cond { "then_branch" } else { "else_branch" };
        let prepared = self.prepare_subgraph(node.id, attr_key, resolved, outer_scope)?;
        let outs = self.run_subgraph(&prepared, &[])?;

        if outs.len() != node.outputs.len() {
            return Err(SessionError::OutputShapeCountMismatch {
                op: format!("If/{attr_key}"),
                expected: node.outputs.len(),
                got: outs.len(),
            });
        }
        for (vid, t) in node.outputs.iter().zip(outs.iter()) {
            self.store_output_tensor(*vid, t, resolved)?;
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
mod tests {
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    use onnx_runtime_ep_api::{
        CaptureSupport, Cost, EpConfig, EpError, ExecutionProviderCapabilities, Fence, Kernel,
        NegotiatedWeight,
    };

    use super::*;

    struct CaptureDecliningKernel;

    impl Kernel for CaptureDecliningKernel {
        fn execute(
            &self,
            _inputs: &[TensorView],
            _outputs: &mut [TensorMut],
        ) -> onnx_runtime_ep_api::Result<()> {
            Ok(())
        }

        fn capture_support(&self) -> CaptureSupport {
            CaptureSupport::unsupported(
                "requires M==1 decode GEMV without group_indices; got a prefill signature",
            )
        }
    }

    #[test]
    fn kernel_capture_reason_propagates_into_structured_report() {
        let mut node = Node::new(NodeId(9), "MatMulNBits", vec![], vec![]);
        node.domain = "com.microsoft".to_string();
        let decline =
            kernel_capture_decline(node.id, &node, &CaptureDecliningKernel).expect("decline");
        let report = CaptureDeclineReport::one(decline);

        assert_eq!(
            report.entries,
            vec![CaptureDecline {
                node_id: Some(9),
                op_type: "MatMulNBits".to_string(),
                domain: "com.microsoft".to_string(),
                reason: "requires M==1 decode GEMV without group_indices; got a prefill signature"
                    .to_string(),
            }]
        );
        assert!(report.to_string().contains("node 9"));
        assert!(
            report
                .to_string()
                .contains("requires M==1 decode GEMV without group_indices")
        );
    }

    #[test]
    fn capture_shapes_seed_unresolved_external_values_without_overwriting_resolved_shapes() {
        let external_value = |shape| ExternalValue {
            dtype: DataType::Float32,
            shape,
            accepts_subshape: false,
            ptr: std::ptr::null_mut(),
            len: 0,
            alignment: 1,
            device: onnx_runtime_ir::DeviceId::cpu(),
        };
        let mut external = ExternalBindings::default();
        external
            .inputs
            .insert(ValueId(0), external_value(vec![1, 2]));
        external
            .outputs
            .insert(ValueId(1), external_value(vec![1, 4, 128, 64]));
        external
            .outputs
            .insert(ValueId(2), external_value(vec![1, 4, 128, 64]));

        let mut resolved = HashMap::from([(ValueId(0), vec![1, 1])]);
        external.seed_capture_shapes(&mut resolved);

        assert_eq!(resolved[&ValueId(0)], vec![1, 1]);
        assert_eq!(resolved[&ValueId(1)], vec![1, 4, 128, 64]);
        assert_eq!(resolved[&ValueId(2)], vec![1, 4, 128, 64]);
    }

    #[test]
    fn only_gqa_cache_inputs_use_physical_capacity_as_kernel_geometry() {
        let mut gqa = Node::new(NodeId(0), "GroupQueryAttention", vec![], vec![]);
        gqa.domain = "com.microsoft".to_string();
        let attention = Node::new(NodeId(1), "Attention", vec![], vec![]);

        assert!(kernel_input_uses_physical_capacity(&gqa, 3));
        assert!(kernel_input_uses_physical_capacity(&gqa, 4));
        assert!(!kernel_input_uses_physical_capacity(&gqa, 0));
        assert!(!kernel_input_uses_physical_capacity(&attention, 4));
    }

    struct WeightDeliveryKernel {
        deliveries: Arc<std::sync::Mutex<Vec<&'static str>>>,
    }

    impl WeightDeliveryKernel {
        fn copy_bytes(bytes: &[u8], output: &mut TensorMut<'_>) -> onnx_runtime_ep_api::Result<()> {
            if bytes.len() != output.byte_size() {
                return Err(EpError::KernelFailed(
                    "test output byte count mismatch".into(),
                ));
            }
            // SAFETY: the executor bounds-checked and exclusively borrowed the
            // output allocation, which is exactly `output.byte_size()` bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    output.data.0.cast::<u8>(),
                    bytes.len(),
                );
            }
            Ok(())
        }
    }

    impl Kernel for WeightDeliveryKernel {
        fn execute(
            &self,
            inputs: &[TensorView],
            outputs: &mut [TensorMut],
        ) -> onnx_runtime_ep_api::Result<()> {
            self.deliveries.lock().unwrap().push("resident");
            let bytes = unsafe {
                std::slice::from_raw_parts(inputs[0].data_ptr::<u8>(), inputs[0].byte_size())
            };
            Self::copy_bytes(bytes, &mut outputs[0])
        }

        fn execute_with_inputs(
            &self,
            inputs: &[KernelInput<'_>],
            outputs: &mut [TensorMut],
        ) -> onnx_runtime_ep_api::Result<()> {
            match &inputs[0] {
                KernelInput::Tensor(view) => self.execute(std::slice::from_ref(view), outputs),
                KernelInput::Weight(handle) => {
                    self.deliveries.lock().unwrap().push("lazy");
                    let NegotiatedWeight::Lazy(lazy) =
                        handle.negotiate(&ExecutionProviderCapabilities::nxrt_weight_paging())?
                    else {
                        return Err(EpError::KernelFailed(
                            "nxrt test EP expected a lazy WeightHandle".into(),
                        ));
                    };
                    let resident = lazy.materialize()?;
                    Self::copy_bytes(resident.bytes(), &mut outputs[0])
                }
            }
        }
    }

    struct WeightDeliveryEp {
        cpu: CpuExecutionProvider,
        lazy: bool,
        optional_input_contract: bool,
        deliveries: Arc<std::sync::Mutex<Vec<&'static str>>>,
        device: onnx_runtime_ir::DeviceId,
        allocations: Arc<AtomicUsize>,
        host_uploads: Arc<AtomicUsize>,
    }

    impl WeightDeliveryEp {
        fn new(lazy: bool, deliveries: Arc<std::sync::Mutex<Vec<&'static str>>>) -> Self {
            Self::with_device(
                lazy,
                deliveries,
                onnx_runtime_ir::DeviceId::cpu(),
                Arc::new(AtomicUsize::new(0)),
                Arc::new(AtomicUsize::new(0)),
            )
        }

        fn non_host(
            lazy: bool,
            deliveries: Arc<std::sync::Mutex<Vec<&'static str>>>,
            allocations: Arc<AtomicUsize>,
            host_uploads: Arc<AtomicUsize>,
        ) -> Self {
            Self::with_device(
                lazy,
                deliveries,
                onnx_runtime_ir::DeviceId::new(onnx_runtime_ir::DeviceType::Custom(7), 0),
                allocations,
                host_uploads,
            )
        }

        fn with_device(
            lazy: bool,
            deliveries: Arc<std::sync::Mutex<Vec<&'static str>>>,
            device: onnx_runtime_ir::DeviceId,
            allocations: Arc<AtomicUsize>,
            host_uploads: Arc<AtomicUsize>,
        ) -> Self {
            let mut cpu = CpuExecutionProvider::new();
            cpu.initialize(&EpConfig::default()).unwrap();
            Self {
                cpu,
                lazy,
                optional_input_contract: false,
                deliveries,
                device,
                allocations,
                host_uploads,
            }
        }

        fn copy_bytes(
            &self,
            src: *const u8,
            dst: *mut u8,
            size: usize,
        ) -> onnx_runtime_ep_api::Result<()> {
            if size != 0 {
                // The test EP tags host allocations as a non-host custom device
                // so executor placement is realistic while bytes stay inspectable.
                unsafe { std::ptr::copy_nonoverlapping(src, dst, size) };
            }
            Ok(())
        }
    }

    impl ExecutionProvider for WeightDeliveryEp {
        fn name(&self) -> &str {
            if self.lazy {
                "nxrt_test_ep"
            } else {
                "stock_test_ep"
            }
        }

        fn device_type(&self) -> onnx_runtime_ir::DeviceType {
            self.device.device_type
        }

        fn device_id(&self) -> onnx_runtime_ir::DeviceId {
            self.device
        }

        fn capabilities(&self) -> ExecutionProviderCapabilities {
            if self.lazy {
                ExecutionProviderCapabilities::nxrt_weight_paging()
            } else {
                ExecutionProviderCapabilities::stock()
            }
        }

        fn initialize(&mut self, _config: &EpConfig) -> onnx_runtime_ep_api::Result<()> {
            Ok(())
        }

        fn shutdown(&mut self) -> onnx_runtime_ep_api::Result<()> {
            Ok(())
        }

        fn supports_op(
            &self,
            op: &Node,
            opset: u64,
            _shapes: &[Shape],
            input_dtypes: &[DataType],
            _layouts: &[TensorLayout],
        ) -> KernelMatch {
            if self.optional_input_contract && op.op_type == "OptionalContract" {
                if input_dtypes == [DataType::Float32, DataType::Undefined, DataType::Bool] {
                    return KernelMatch::Supported {
                        cost: Cost::ZERO,
                        required_input_layouts: None,
                        output_layouts: vec![TensorLayout::contiguous()],
                    };
                }
                return KernelMatch::unsupported(format!(
                    "OptionalContract requires [Float32, Undefined, Bool] input dtypes, got {input_dtypes:?}"
                ));
            }
            if LazyWeightBoundary::BlockQuantizedMoe.matches(&op.domain, &op.op_type)
                || (op.domain.is_empty() && op.op_type == "Identity")
            {
                KernelMatch::Supported {
                    cost: Cost::ZERO,
                    required_input_layouts: None,
                    output_layouts: vec![TensorLayout::contiguous()],
                }
            } else {
                KernelMatch::unsupported(format!(
                    "no handler for {}::{} at opset {opset} — test EP intentionally declines this op",
                    canonical_domain(op),
                    op.op_type
                ))
            }
        }

        fn get_kernel(
            &self,
            _op: &Node,
            _shapes: &[Vec<usize>],
            _opset: u64,
        ) -> onnx_runtime_ep_api::Result<Box<dyn Kernel>> {
            Ok(Box::new(WeightDeliveryKernel {
                deliveries: Arc::clone(&self.deliveries),
            }))
        }

        fn allocate(
            &self,
            size: usize,
            alignment: usize,
        ) -> onnx_runtime_ep_api::Result<DeviceBuffer> {
            self.allocations.fetch_add(1, Ordering::Relaxed);
            if self.device.is_host_accessible() {
                return self.cpu.allocate(size, alignment);
            }
            let layout = std::alloc::Layout::from_size_align(size.max(1), alignment)
                .map_err(|_| EpError::AlignmentError)?;
            let ptr = unsafe { std::alloc::alloc(layout) };
            if ptr.is_null() {
                return Err(EpError::OutOfMemory {
                    requested: size,
                    available: 0,
                });
            }
            Ok(unsafe { DeviceBuffer::from_raw_parts(ptr.cast(), self.device, size, alignment) })
        }

        fn deallocate(&self, buffer: DeviceBuffer) -> onnx_runtime_ep_api::Result<()> {
            if self.device.is_host_accessible() {
                return self.cpu.deallocate(buffer);
            }
            let size = buffer.len();
            let alignment = buffer.alignment();
            let ptr = buffer.into_raw().cast::<u8>();
            let layout = std::alloc::Layout::from_size_align(size.max(1), alignment)
                .expect("test EP allocated this layout");
            unsafe { std::alloc::dealloc(ptr, layout) };
            Ok(())
        }

        fn copy(
            &self,
            src: &DeviceBuffer,
            dst: &mut DeviceBuffer,
            size: usize,
        ) -> onnx_runtime_ep_api::Result<()> {
            if size > src.len() || size > dst.len() {
                return Err(EpError::KernelFailed("test EP copy out of bounds".into()));
            }
            self.copy_bytes(src.as_ptr().cast(), dst.as_mut_ptr().cast(), size)
        }

        fn copy_async(
            &self,
            src: &DeviceBuffer,
            dst: &mut DeviceBuffer,
            size: usize,
        ) -> onnx_runtime_ep_api::Result<Fence> {
            self.copy(src, dst, size)?;
            Ok(Fence::default())
        }

        fn sync(&self) -> onnx_runtime_ep_api::Result<()> {
            Ok(())
        }

        fn copy_from_host(
            &self,
            src: &[u8],
            dst: &mut DeviceBuffer,
        ) -> onnx_runtime_ep_api::Result<()> {
            if src.len() > dst.len() {
                return Err(EpError::KernelFailed(
                    "test EP host upload out of bounds".into(),
                ));
            }
            self.host_uploads.fetch_add(1, Ordering::Relaxed);
            self.copy_bytes(src.as_ptr(), dst.as_mut_ptr().cast(), src.len())
        }

        fn copy_to_host(
            &self,
            src: &DeviceBuffer,
            dst: &mut [u8],
        ) -> onnx_runtime_ep_api::Result<()> {
            if dst.len() > src.len() {
                return Err(EpError::KernelFailed(
                    "test EP host download out of bounds".into(),
                ));
            }
            self.copy_bytes(src.as_ptr().cast(), dst.as_mut_ptr(), dst.len())
        }
    }

    fn weight_delivery_fixture() -> (Graph, Arc<WeightStore>, std::path::PathBuf) {
        static NEXT_FILE: AtomicU64 = AtomicU64::new(0);
        let root = std::env::var_os("CARGO_TARGET_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::env::current_dir().unwrap().join("target"))
            .join("weight-handle-tests");
        std::fs::create_dir_all(&root).unwrap();
        let id = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
        let path = root.join(format!(
            "block-quantized-moe-{}-{id}.bin",
            std::process::id()
        ));
        std::fs::write(&path, [1u8, 2, 3, 4]).unwrap();

        let mut graph = Graph::new();
        graph.opset_imports.insert("pkg.nxrt".into(), 1);
        let weight = graph.create_named_value("weight", DataType::Uint8, static_shape([4]));
        graph.set_initializer(
            weight,
            WeightRef::External {
                path: path.clone(),
                offset: 0,
                length: 4,
                dtype: DataType::Uint8,
                dims: vec![4],
            },
        );
        let output = graph.create_named_value("output", DataType::Uint8, static_shape([4]));
        let mut node = Node::new(
            NodeId(0),
            "BlockQuantizedMoE",
            vec![Some(weight)],
            vec![output],
        );
        node.domain = "pkg.nxrt".into();
        graph.insert_node(node);
        graph.add_output(output);

        let mut store = WeightStore::new();
        store.map_external(&path).unwrap();
        (graph, Arc::new(store), path)
    }

    #[test]
    fn claim_time_optional_input_dtype_is_undefined_not_silently_float32() {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 1);
        let data = graph.create_named_value("data", DataType::Float32, static_shape([1]));
        let training_mode =
            graph.create_named_value("training_mode", DataType::Bool, static_shape([]));
        let output = graph.create_named_value("output", DataType::Float32, static_shape([1]));
        graph.add_input(data);
        graph.add_input(training_mode);
        graph.add_output(output);
        graph.insert_node(Node::new(
            NodeId(0),
            "OptionalContract",
            vec![Some(data), None, Some(training_mode)],
            vec![output],
        ));

        let deliveries = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut ep = WeightDeliveryEp::new(false, deliveries);
        ep.optional_input_contract = true;
        let executor = Executor::build(graph, Arc::new(WeightStore::new()), Arc::new(ep));
        assert!(
            executor.is_ok(),
            "an omitted optional input must reach supports_op as DataType::Undefined"
        );
    }

    #[test]
    fn executor_opens_per_op_span_only_when_tracing_enabled() {
        use onnx_runtime_tracer::TraceContext;

        // Disabled (default noop): no spans recorded, hot path stays quiet.
        {
            let (graph, weights, path) = weight_delivery_fixture();
            let deliveries = Arc::new(std::sync::Mutex::new(Vec::new()));
            let ep: Arc<dyn ExecutionProvider> =
                Arc::new(WeightDeliveryEp::new(false, Arc::clone(&deliveries)));
            let mut executor = Executor::build(graph, weights, ep).unwrap();
            let (trace, events) = TraceContext::in_memory();
            trace.set_enabled(false);
            executor.set_trace_context(trace);
            let _ = executor.run(&[]).unwrap();
            drop(executor);
            std::fs::remove_file(path).unwrap();
            assert!(
                events.events().is_empty(),
                "a disabled trace context must not open op spans"
            );
        }

        // Enabled: exactly one op span per executed node, named by op type.
        {
            let (graph, weights, path) = weight_delivery_fixture();
            let deliveries = Arc::new(std::sync::Mutex::new(Vec::new()));
            let ep: Arc<dyn ExecutionProvider> =
                Arc::new(WeightDeliveryEp::new(false, Arc::clone(&deliveries)));
            let mut executor = Executor::build(graph, weights, ep).unwrap();
            let (trace, events) = TraceContext::in_memory();
            executor.set_trace_context(trace);
            let _ = executor.run(&[]).unwrap();
            drop(executor);
            std::fs::remove_file(path).unwrap();
            let spans = events.events();
            assert_eq!(spans.len(), 1, "one op span per executed node");
            assert_eq!(spans[0].name, "BlockQuantizedMoE");
            assert_eq!(spans[0].cat, "op");
        }
    }

    #[test]
    fn op_capture_trace_annotates_span_with_status_and_reason() {
        use onnx_runtime_tracer::TraceContext;

        // Rejected: both a status and the actionable why-not reason land on the
        // active op-span. This is the branch a production model that declines
        // capture would hit; the qwen fixture captures cleanly so it is proven
        // here directly rather than in the live trace.
        {
            let (trace, events) = TraceContext::in_memory();
            {
                let _span = trace.span("MatMulNBits", "op");
                OpCaptureTrace::Rejected(
                    "kernel declares CaptureSupport::Unsupported: per-call workspace alloc",
                )
                .annotate();
            }
            let recorded = events.events();
            assert_eq!(recorded.len(), 1);
            let args = recorded[0].args.as_ref().unwrap();
            assert_eq!(args[ARG_CAPTURE_STATUS], "rejected");
            assert!(
                args[ARG_CAPTURE_REASON]
                    .as_str()
                    .unwrap()
                    .contains("CaptureSupport::Unsupported")
            );
        }

        // Captured: status only, no reason (nothing was declined).
        {
            let (trace, events) = TraceContext::in_memory();
            {
                let _span = trace.span("MatMulNBits", "op");
                OpCaptureTrace::Captured.annotate();
            }
            let recorded = events.events();
            let args = recorded[0].args.as_ref().unwrap();
            assert_eq!(args[ARG_CAPTURE_STATUS], "captured");
        }

        // Eager: no capture attempt, so no capture annotation at all.
        {
            let (trace, events) = TraceContext::in_memory();
            {
                let _span = trace.span("MatMulNBits", "op");
                OpCaptureTrace::Eager.annotate();
            }
            let recorded = events.events();
            assert!(
                recorded[0]
                    .args
                    .as_ref()
                    .map(|a| a.get(ARG_CAPTURE_STATUS).is_none())
                    .unwrap_or(true),
                "eager ops carry no capture status"
            );
        }
    }

    #[test]
    fn executor_selects_lazy_or_resident_weight_delivery_from_ep_capability() {
        for (lazy, expected) in [(true, "lazy"), (false, "resident")] {
            let (graph, weights, path) = weight_delivery_fixture();
            let deliveries = Arc::new(std::sync::Mutex::new(Vec::new()));
            let ep: Arc<dyn ExecutionProvider> =
                Arc::new(WeightDeliveryEp::new(lazy, Arc::clone(&deliveries)));
            let mut executor = Executor::build(graph, weights, ep).unwrap();
            let outputs = executor.run(&[]).unwrap();

            assert_eq!(outputs[0].as_bytes(), &[1, 2, 3, 4]);
            assert_eq!(&*deliveries.lock().unwrap(), &[expected]);
            drop(executor);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn non_host_lazy_only_initializer_skips_eager_device_residency() {
        for (lazy, expected_allocations, expected_uploads, expected_delivery) in
            [(true, 1, 0, "lazy"), (false, 2, 1, "resident")]
        {
            let (graph, weights, path) = weight_delivery_fixture();
            let deliveries = Arc::new(std::sync::Mutex::new(Vec::new()));
            let allocations = Arc::new(AtomicUsize::new(0));
            let host_uploads = Arc::new(AtomicUsize::new(0));
            let ep: Arc<dyn ExecutionProvider> = Arc::new(WeightDeliveryEp::non_host(
                lazy,
                Arc::clone(&deliveries),
                Arc::clone(&allocations),
                Arc::clone(&host_uploads),
            ));
            let mut executor = Executor::build(graph, weights, ep).unwrap();

            assert_eq!(
                allocations.load(Ordering::Relaxed),
                expected_allocations,
                "lazy nxrt builds only the output; stock EPs also allocate the initializer"
            );
            assert_eq!(
                host_uploads.load(Ordering::Relaxed),
                expected_uploads,
                "lazy nxrt must not upload the initializer during build"
            );

            let outputs = executor.run(&[]).unwrap();
            assert_eq!(outputs[0].as_bytes(), &[1, 2, 3, 4]);
            assert_eq!(&*deliveries.lock().unwrap(), &[expected_delivery]);
            assert_eq!(
                host_uploads.load(Ordering::Relaxed),
                expected_uploads,
                "dispatch must not introduce a second EP upload"
            );
            drop(executor);
            std::fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn initializer_shared_with_resident_consumer_uses_one_device_copy() {
        let (mut graph, weights, path) = weight_delivery_fixture();
        graph.opset_imports.insert(String::new(), 17);
        let weight = graph
            .values
            .iter()
            .find_map(|(vid, value)| (value.name.as_deref() == Some("weight")).then_some(vid))
            .unwrap();
        let resident_output =
            graph.create_named_value("resident_output", DataType::Uint8, static_shape([4]));
        graph.insert_node(Node::new(
            NodeId(1),
            "Identity",
            vec![Some(weight)],
            vec![resident_output],
        ));
        graph.add_output(resident_output);

        let deliveries = Arc::new(std::sync::Mutex::new(Vec::new()));
        let allocations = Arc::new(AtomicUsize::new(0));
        let host_uploads = Arc::new(AtomicUsize::new(0));
        let ep: Arc<dyn ExecutionProvider> = Arc::new(WeightDeliveryEp::non_host(
            true,
            Arc::clone(&deliveries),
            Arc::clone(&allocations),
            Arc::clone(&host_uploads),
        ));
        let mut executor = Executor::build(graph, weights, ep).unwrap();

        assert!(
            !executor.weight_handles.contains_key(&weight),
            "a resident consumer makes the single eager device copy authoritative"
        );
        assert_eq!(allocations.load(Ordering::Relaxed), 3);
        assert_eq!(host_uploads.load(Ordering::Relaxed), 1);

        let outputs = executor.run(&[]).unwrap();
        assert_eq!(outputs[0].as_bytes(), &[1, 2, 3, 4]);
        assert_eq!(outputs[1].as_bytes(), &[1, 2, 3, 4]);
        assert_eq!(&*deliveries.lock().unwrap(), &["resident", "resident"]);
        assert_eq!(
            host_uploads.load(Ordering::Relaxed),
            1,
            "both consumers must share the one resident initializer"
        );
        drop(executor);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn coverage_collector_surfaces_ep_decline_reason() {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        let input = graph.create_named_value("x", DataType::Float32, vec![Dim::Static(1)]);
        let output = graph.create_named_value("y", DataType::Float32, vec![Dim::Static(1)]);
        graph.insert_node(Node::new(
            NodeId(0),
            "NotRegistered",
            vec![Some(input)],
            vec![output],
        ));

        let ep = CpuExecutionProvider::new();
        let mut issues = Vec::new();
        collect_cuda_coverage_issues(&graph, &graph, &ep, "graph", &mut issues);

        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].op_type, "NotRegistered");
        assert_eq!(issues[0].domain, "ai.onnx");
        assert!(
            issues[0]
                .reason
                .contains("no handler for ai.onnx::NotRegistered at opset 17"),
            "{}",
            issues[0].reason
        );
        assert!(
            !issues[0].reason.contains("unsupported by"),
            "{}",
            issues[0].reason
        );
    }

    #[test]
    fn cuda_coverage_report_groups_all_distinct_failure_classes_deterministically() {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        let input = graph.create_named_value("x", DataType::Float32, vec![Dim::Static(1)]);

        let op_types = [
            "RepeatedMissing",
            "Missing08",
            "RepeatedMissing",
            "Missing07",
            "Missing06",
            "RepeatedMissing",
            "Missing05",
            "Missing04",
            "Missing03",
            "Missing02",
            "Missing01",
            "Missing00",
            "RepeatedMissing",
        ];
        for (index, op_type) in op_types.into_iter().enumerate() {
            let output = graph.create_named_value(
                format!("output_{index}"),
                DataType::Float32,
                vec![Dim::Static(1)],
            );
            graph.insert_node(Node::new(
                NodeId(index as u32),
                op_type,
                vec![Some(input)],
                vec![output],
            ));
        }

        let ep = WeightDeliveryEp::with_device(
            false,
            Arc::new(std::sync::Mutex::new(Vec::new())),
            onnx_runtime_ir::DeviceId::cuda(0),
            Arc::new(AtomicUsize::new(0)),
            Arc::new(AtomicUsize::new(0)),
        );
        let report = || {
            cuda_fallback_report(&graph, &ep)
                .expect("CUDA declines must produce a fallback report")
                .to_string()
        };
        let first = report();
        let second = report();

        assert_eq!(first, second);
        assert!(first.contains("13 nodes assigned to CPU"));
        assert!(first.contains("GPU EP stock_test_ep did not claim 13 node(s)"));
        assert!(first.contains("the whole session uses cpu_ep"));
        assert_eq!(first.matches("ai.onnx::RepeatedMissing:").count(), 1);
        assert!(first.contains("ai.onnx::RepeatedMissing: no handler"));
        assert!(first.contains("[count=4; examples: graph/node#0, graph/node#12, graph/node#2]"));
        assert!(!first.contains("graph/node#5"));

        for op_type in [
            "Missing00",
            "Missing01",
            "Missing02",
            "Missing03",
            "Missing04",
            "Missing05",
            "Missing06",
            "Missing07",
            "Missing08",
        ] {
            assert_eq!(
                first.matches(&format!("ai.onnx::{op_type}:")).count(),
                1,
                "{first}"
            );
            assert!(
                first.contains(&format!("ai.onnx::{op_type}: no handler")),
                "{first}"
            );
        }
        assert!(!first.contains("more unsupported node"));
    }

    #[test]
    fn cuda_decline_warns_and_falls_back_to_cpu_unless_strict() {
        let graph = || {
            let mut graph = Graph::new();
            graph.opset_imports.insert(String::new(), 17);
            let input = graph.create_named_value("input", DataType::Float32, vec![Dim::Static(1)]);
            let output =
                graph.create_named_value("output", DataType::Float32, vec![Dim::Static(1)]);
            graph.add_input(input);
            graph.add_output(output);
            graph.insert_node(Node::new(
                NodeId(0),
                "Relu",
                vec![Some(input)],
                vec![output],
            ));
            graph
        };
        let cuda_ep = || {
            Arc::new(WeightDeliveryEp::with_device(
                false,
                Arc::new(std::sync::Mutex::new(Vec::new())),
                onnx_runtime_ir::DeviceId::cuda(0),
                Arc::new(AtomicUsize::new(0)),
                Arc::new(AtomicUsize::new(0)),
            )) as Arc<dyn ExecutionProvider>
        };

        let exec = Executor::build_with_cuda_requirement(
            graph(),
            Arc::new(WeightStore::new()),
            cuda_ep(),
            false,
        )
        .expect("default CUDA decline must use the CPU fallback");
        assert_eq!(exec.device_id().device_type, DeviceType::Cpu);
        let report = exec
            .execution_provider_fallback_report()
            .expect("fallback must remain observable");
        assert_eq!(report.assigned_node_count, 1);
        assert_eq!(report.assigned_ops, ["ai.onnx::Relu"]);
        assert_eq!(report.declines.len(), 1);
        assert_eq!(report.declines[0].op_type, "Relu");
        assert!(report.declines[0].reason.contains("intentionally declines"));

        let strict = Executor::build_with_cuda_requirement(
            graph(),
            Arc::new(WeightStore::new()),
            cuda_ep(),
            true,
        )
        .err()
        .expect("strict CUDA must reject CPU fallback");
        assert!(strict.to_string().contains("ONNX_GENAI_REQUIRE_CUDA=1"));
    }

    #[test]
    fn sequence_executor_preserves_element_arc_identity() {
        use onnx_runtime_ir::{TensorData, WeightRef, static_shape};

        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);

        let input = graph.create_named_value("input", DataType::Float32, static_shape([2]));
        graph.set_initializer(
            input,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float32,
                vec![2],
                [7.0f32, 8.0]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect(),
            )),
        );
        let zero = graph.create_named_value("zero", DataType::Int64, static_shape([]));
        graph.set_initializer(
            zero,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Int64,
                vec![],
                0i64.to_le_bytes().to_vec(),
            )),
        );
        let one = graph.create_named_value("one", DataType::Int64, static_shape([]));
        graph.set_initializer(
            one,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Int64,
                vec![],
                1i64.to_le_bytes().to_vec(),
            )),
        );

        let first_sequence = graph.create_value(DataType::Float32, static_shape([]));
        graph.insert_node(Node::new(
            NodeId(0),
            "SequenceConstruct",
            vec![Some(input)],
            vec![first_sequence],
        ));
        let first_at = graph.create_value(DataType::Float32, static_shape([2]));
        graph.insert_node(Node::new(
            NodeId(0),
            "SequenceAt",
            vec![Some(first_sequence), Some(zero)],
            vec![first_at],
        ));
        let inserted_sequence = graph.create_value(DataType::Float32, static_shape([]));
        graph.insert_node(Node::new(
            NodeId(0),
            "SequenceInsert",
            vec![Some(first_sequence), Some(first_at)],
            vec![inserted_sequence],
        ));
        let second_at = graph.create_value(DataType::Float32, static_shape([2]));
        graph.insert_node(Node::new(
            NodeId(0),
            "SequenceAt",
            vec![Some(inserted_sequence), Some(one)],
            vec![second_at],
        ));
        graph.add_output(second_at);

        let mut executor = Executor::build(
            graph,
            Arc::new(WeightStore::new()),
            auto_detect_cpu_ep().unwrap(),
        )
        .unwrap();
        let output = executor.run(&[]).unwrap();
        assert_eq!(output[0].to_vec_f32(), vec![7.0, 8.0]);

        let original = &executor.sequences[&first_sequence].elements()[0];
        let first_at_arc = &executor.seq_elem_values[&first_at];
        let inserted = &executor.sequences[&inserted_sequence].elements()[1];
        let second_at_arc = &executor.seq_elem_values[&second_at];
        assert!(original.shares_storage_with(first_at_arc));
        assert!(original.shares_storage_with(inserted));
        assert!(original.shares_storage_with(second_at_arc));
        assert_eq!(original.as_ptr(), executor.buffers[&input].as_ptr());
    }

    #[test]
    fn fuses_only_single_consumer_silu_pattern() {
        let mut graph = Graph::new();
        let shape = vec![Dim::Static(2)];
        let x = graph.create_named_value("x", DataType::Float32, shape.clone());
        let sigmoid_out = graph.create_named_value("sigmoid", DataType::Float32, shape.clone());
        let silu_out = graph.create_named_value("silu", DataType::Float32, shape);
        graph.add_input(x);
        graph.add_output(silu_out);
        graph.insert_node(Node::new(
            NodeId(0),
            "Sigmoid",
            vec![Some(x)],
            vec![sigmoid_out],
        ));
        graph.insert_node(Node::new(
            NodeId(0),
            "Mul",
            vec![Some(sigmoid_out), Some(x)],
            vec![silu_out],
        ));

        assert_eq!(fuse_silu_patterns(&mut graph), 1);
        assert_eq!(graph.num_nodes(), 1);
        let fused = graph.nodes.values().next().unwrap();
        assert_eq!(fused.op_type, "Silu");
        assert_eq!(fused.domain, "com.microsoft");
        assert_eq!(fused.inputs, vec![Some(x)]);
        assert_eq!(fused.outputs, vec![silu_out]);
        assert_eq!(graph.opset_imports["com.microsoft"], 1);
    }

    #[test]
    fn does_not_fuse_silu_when_sigmoid_has_second_consumer() {
        let mut graph = Graph::new();
        let shape = vec![Dim::Static(2)];
        let x = graph.create_named_value("x", DataType::Float32, shape.clone());
        let sigmoid_out = graph.create_named_value("sigmoid", DataType::Float32, shape.clone());
        let mul_out = graph.create_named_value("mul", DataType::Float32, shape.clone());
        let identity_out = graph.create_named_value("identity", DataType::Float32, shape);
        graph.add_input(x);
        graph.add_output(mul_out);
        graph.add_output(identity_out);
        graph.insert_node(Node::new(
            NodeId(0),
            "Sigmoid",
            vec![Some(x)],
            vec![sigmoid_out],
        ));
        graph.insert_node(Node::new(
            NodeId(0),
            "Mul",
            vec![Some(x), Some(sigmoid_out)],
            vec![mul_out],
        ));
        graph.insert_node(Node::new(
            NodeId(0),
            "Identity",
            vec![Some(sigmoid_out)],
            vec![identity_out],
        ));

        assert_eq!(fuse_silu_patterns(&mut graph), 0);
        assert_eq!(graph.num_nodes(), 3);
        assert_eq!(
            graph
                .nodes
                .values()
                .filter(|node| node.op_type == "Sigmoid")
                .count(),
            1
        );
        assert_eq!(
            graph
                .nodes
                .values()
                .filter(|node| node.op_type == "Mul")
                .count(),
            1
        );
        assert!(graph.validate().is_ok());
    }

    /// Holden's precondition: the dispatch-boundary gate must reject a view that
    /// addresses bytes past its backing allocation, rather than letting a kernel
    /// dereference out of bounds (UB).
    #[test]
    fn view_bounds_rejects_out_of_bounds_view() {
        // A [2, 3] f32 view needs 24 bytes; give it a 16-byte backing length.
        let shape = [2usize, 3];
        let strides = compute_contiguous_strides(&shape);
        let err = view_bounds(&shape, &strides, 0, DataType::Float32, 16);
        assert!(err.is_err(), "gate must reject an oversized view");

        // Exactly-fitting length is accepted.
        assert!(view_bounds(&shape, &strides, 0, DataType::Float32, 24).is_ok());
    }

    /// A negative byte offset region (via a byte_offset that pushes the origin
    /// past the buffer) is also rejected.
    #[test]
    fn view_bounds_rejects_offset_overrun() {
        let shape = [4usize];
        let strides = compute_contiguous_strides(&shape);
        // 4 f32 = 16 bytes; origin at byte 8 leaves only 8 bytes → overrun.
        assert!(view_bounds(&shape, &strides, 8, DataType::Float32, 16).is_err());
        assert!(view_bounds(&shape, &strides, 0, DataType::Float32, 16).is_ok());
    }

    /// Symbol substitution: static dims pass through, bound symbols resolve, an
    /// unbound symbol yields `None` (the uninferred-shape signal).
    #[test]
    fn substitute_resolves_bound_symbols_only() {
        let mut bindings = HashMap::new();
        bindings.insert(SymbolId(0), 7usize);
        let shape = vec![Dim::Symbolic(SymbolId(0)), Dim::Static(4)];
        assert_eq!(substitute(&shape, &bindings), Some(vec![7, 4]));

        let unbound = vec![Dim::Symbolic(SymbolId(1)), Dim::Static(4)];
        assert_eq!(substitute(&unbound, &bindings), None);
    }

    /// H-D1: element-count multiplication must be overflow-checked so a huge or
    /// malicious shape reports `ShapeOverflow` instead of wrapping `usize` and
    /// under-sizing the buffer.
    #[test]
    fn checked_numel_detects_overflow() {
        // Well-formed shapes multiply normally.
        assert_eq!(checked_numel(&[2, 3, 4], || "v".into()).unwrap(), 24);
        assert_eq!(checked_numel(&[], || "v".into()).unwrap(), 1);

        // A product past usize::MAX overflows.
        let huge = [usize::MAX, 2];
        let err = checked_numel(&huge, || "value#9".into());
        assert!(matches!(err, Err(SessionError::ShapeOverflow { .. })));
    }

    /// H-D1 (byte layer): even when the element *count* fits in `usize`, the
    /// count → bytes multiply can wrap for a fixed-width dtype. The allocation
    /// path must report `ShapeOverflow` rather than under-allocating.
    #[test]
    fn checked_storage_bytes_detects_byte_overflow() {
        // `usize::MAX / 4` elements fit in usize (pass checked_numel) but
        // `* 8` bytes for Float64 wraps — this is the exploited under-alloc.
        let numel = usize::MAX / 4;
        let err = checked_storage_bytes(DataType::Float64, numel, || "value#9".into(), &[numel]);
        assert!(matches!(err, Err(SessionError::ShapeOverflow { .. })));

        // A well-formed size passes through unchanged.
        assert_eq!(
            checked_storage_bytes(DataType::Float32, 4, || "v".into(), &[4]).unwrap(),
            16
        );
    }

    /// The data-dependent shape sizer must return exactly one shape per output
    /// so the run loop's `out_shapes[oi]` indexing can never misindex. Slice is
    /// single-output, so it returns a 1-element Vec; the run loop additionally
    /// guards the count (see `OutputShapeCountMismatch`).
    #[test]
    fn dynamic_output_shapes_slice_is_single_output() {
        let node = Node::new(NodeId(0), "Slice", vec![], vec![]);
        let input_shapes = vec![vec![4usize, 2]];
        let input_values = vec![
            None,          // data (unused by sizer)
            Some(vec![1]), // starts
            Some(vec![3]), // ends
            Some(vec![0]), // axes
            Some(vec![1]), // steps
        ];
        let input_dtypes = vec![
            DataType::Float32,
            DataType::Int64,
            DataType::Int64,
            DataType::Int64,
            DataType::Int64,
        ];
        let out =
            dynamic_output_shapes(&node, &input_shapes, &input_dtypes, &input_values, 17).unwrap();
        assert_eq!(out.len(), 1, "Slice must resolve exactly one output shape");
        assert_eq!(out[0], vec![2, 2]);

        let mut custom_slice = Node::new(NodeId(1), "Slice", vec![], vec![ValueId(0)]);
        custom_slice.domain = "example.custom".into();
        assert!(
            dynamic_output_shapes(
                &custom_slice,
                &input_shapes,
                &input_dtypes,
                &input_values,
                17
            )
            .is_none(),
            "ONNX Slice semantics must not be applied to an unrelated custom-domain op"
        );

        // An op the sizer cannot resolve returns None (surfaces as UnresolvedShape).
        let other = Node::new(
            NodeId(2),
            "NxrtNeverRegisteredSentinelOp",
            vec![],
            vec![ValueId(0)],
        );
        assert!(
            dynamic_output_shapes(&other, &input_shapes, &input_dtypes, &input_values, 17)
                .is_none()
        );
    }

    #[test]
    fn dynamic_output_shapes_gqa_supports_packed_qkv() {
        use onnx_runtime_ir::{Attribute, ValueId};

        let mut node = Node::new(
            NodeId(0),
            "GroupQueryAttention",
            vec![
                Some(ValueId(0)),
                None,
                None,
                Some(ValueId(3)),
                Some(ValueId(4)),
                Some(ValueId(5)),
                Some(ValueId(6)),
            ],
            vec![ValueId(7), ValueId(8), ValueId(9)],
        );
        node.domain = "com.microsoft".into();
        node.attributes
            .insert("num_heads".into(), Attribute::Int(14));
        node.attributes
            .insert("kv_num_heads".into(), Attribute::Int(2));
        let input_shapes = vec![
            vec![1, 1, 1152],
            vec![],
            vec![],
            vec![1, 2, 16, 64],
            vec![1, 2, 16, 64],
            vec![1],
            vec![],
        ];
        let input_values = vec![None, None, None, None, None, None, Some(vec![17])];

        assert_eq!(
            dynamic_output_shapes(
                &node,
                &input_shapes,
                &[
                    DataType::Float32,
                    DataType::Undefined,
                    DataType::Undefined,
                    DataType::Float32,
                    DataType::Float32,
                    DataType::Int32,
                    DataType::Int32,
                ],
                &input_values,
                1,
            ),
            Some(vec![
                vec![1, 1, 896],
                vec![1, 2, 17, 64],
                vec![1, 2, 17, 64],
            ])
        );
    }

    /// The effective opset is read from the graph's import for the op's domain,
    /// with the default and `ai.onnx` spellings treated as one.
    #[test]
    fn effective_opset_reads_graph_import() {
        let mut graph = Graph::default();
        graph.opset_imports.insert(String::new(), 12);
        let node = Node::new(NodeId(0), "Softmax", vec![], vec![]);
        assert_eq!(effective_opset(&graph, &node), 12);

        graph.opset_imports.insert(String::new(), 0);
        assert_eq!(effective_opset(&graph, &node), 0);
    }

    #[test]
    #[should_panic(expected = "internal invariant violated")]
    fn effective_opset_requires_validated_import() {
        effective_opset(
            &Graph::default(),
            &Node::new(NodeId(0), "Softmax", vec![], vec![]),
        );
    }

    #[test]
    fn child_executor_binds_formals_captures_and_inline_initializers_in_output_order() {
        use onnx_runtime_ir::{TensorData, WeightRef, static_shape};

        let mut body = Graph::new();
        let formal = body.create_named_value("formal", DataType::Float32, static_shape([2]));
        body.add_input(formal);
        let captured = body.create_named_value("captured", DataType::Float32, static_shape([2]));
        let one = body.create_named_value("one", DataType::Float32, static_shape([2]));
        body.set_initializer(
            one,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float32,
                vec![2],
                [1.0f32, 1.0]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect(),
            )),
        );
        let sum = body.create_named_value("sum", DataType::Float32, static_shape([2]));
        body.insert_node(Node::new(
            NodeId(0),
            "Add",
            vec![Some(formal), Some(captured)],
            vec![sum],
        ));
        let adjusted = body.create_named_value("adjusted", DataType::Float32, static_shape([2]));
        body.insert_node(Node::new(
            NodeId(0),
            "Add",
            vec![Some(sum), Some(one)],
            vec![adjusted],
        ));
        // Deliberately reverse production order to prove formal output ordering.
        body.add_output(adjusted);
        body.add_output(sum);

        let mut opsets = HashMap::new();
        opsets.insert(String::new(), 17);
        let mut child = ChildExecutor::new(
            "direct-test",
            body,
            opsets,
            Arc::new(WeightStore::new()),
            auto_detect_cpu_ep().unwrap(),
        )
        .unwrap();
        let mut outer_scope = HashMap::new();
        outer_scope.insert(
            "captured".to_string(),
            Tensor::from_f32(&[2], &[10.0, 20.0]).unwrap(),
        );

        let first = Tensor::from_f32(&[2], &[2.0, 3.0]).unwrap();
        let outputs = child.run(&[&first], &outer_scope).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].to_vec_f32(), vec![13.0, 24.0]);
        assert_eq!(outputs[1].to_vec_f32(), vec![12.0, 23.0]);
        assert_eq!(child.stats(), ChildExecutorStats { builds: 1, runs: 1 });

        let second = Tensor::from_f32(&[2], &[-1.0, 4.0]).unwrap();
        let outputs = child.run(&[&second], &outer_scope).unwrap();
        assert_eq!(outputs[0].to_vec_f32(), vec![10.0, 25.0]);
        assert_eq!(outputs[1].to_vec_f32(), vec![9.0, 24.0]);
        assert_eq!(
            child.stats(),
            ChildExecutorStats { builds: 1, runs: 2 },
            "matching input signatures must reuse the compiled child plan"
        );
    }

    fn unary_child(name: &str) -> ChildExecutor {
        let mut body = Graph::new();
        let input = body.create_named_value("input", DataType::Float32, Vec::new());
        body.add_input(input);
        let output = body.create_named_value("output", DataType::Float32, Vec::new());
        body.insert_node(Node::new(
            NodeId(0),
            "Relu",
            vec![Some(input)],
            vec![output],
        ));
        body.add_output(output);

        let mut opsets = HashMap::new();
        opsets.insert(String::new(), 17);
        ChildExecutor::new(
            name,
            body,
            opsets,
            Arc::new(WeightStore::new()),
            auto_detect_cpu_ep().unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn child_executor_reuses_a_signature_after_an_intervening_signature() {
        let mut child = unary_child("a-b-a");
        let outer_scope = HashMap::new();
        let a = Tensor::from_f32(&[1], &[-1.0]).unwrap();
        let b = Tensor::from_f32(&[2], &[-2.0, 3.0]).unwrap();

        assert_eq!(
            child.run(&[&a], &outer_scope).unwrap()[0].to_vec_f32(),
            vec![0.0]
        );
        assert_eq!(
            child.run(&[&b], &outer_scope).unwrap()[0].to_vec_f32(),
            vec![0.0, 3.0]
        );
        assert_eq!(
            child.run(&[&a], &outer_scope).unwrap()[0].to_vec_f32(),
            vec![0.0]
        );
        assert_eq!(child.stats(), ChildExecutorStats { builds: 2, runs: 3 });
    }

    #[test]
    fn child_executor_lru_evicts_oldest_signature_only() {
        let mut child = unary_child("lru-eviction");
        let outer_scope = HashMap::new();
        let inputs = (1..=CHILD_EXECUTOR_CACHE_CAPACITY + 1)
            .map(|len| Tensor::from_f32(&[len], &vec![len as f32; len]).unwrap())
            .collect::<Vec<_>>();

        for input in &inputs {
            child.run(&[input], &outer_scope).unwrap();
        }
        assert_eq!(
            child.stats(),
            ChildExecutorStats {
                builds: (CHILD_EXECUTOR_CACHE_CAPACITY + 1) as u64,
                runs: (CHILD_EXECUTOR_CACHE_CAPACITY + 1) as u64,
            }
        );

        child.run(&[&inputs[0]], &outer_scope).unwrap();
        child.run(&[inputs.last().unwrap()], &outer_scope).unwrap();
        assert_eq!(
            child.stats(),
            ChildExecutorStats {
                builds: (CHILD_EXECUTOR_CACHE_CAPACITY + 2) as u64,
                runs: (CHILD_EXECUTOR_CACHE_CAPACITY + 3) as u64,
            },
            "the evicted oldest signature must rebuild while a recent entry remains cached"
        );
    }

    fn captured_add_child(name: &str) -> ChildExecutor {
        let mut body = Graph::new();
        let input = body.create_named_value("input", DataType::Float32, Vec::new());
        body.add_input(input);
        let captured = body.create_named_value("captured", DataType::Float32, Vec::new());
        let output = body.create_named_value("output", DataType::Float32, Vec::new());
        body.insert_node(Node::new(
            NodeId(0),
            "Add",
            vec![Some(input), Some(captured)],
            vec![output],
        ));
        body.add_output(output);

        let mut opsets = HashMap::new();
        opsets.insert(String::new(), 17);
        ChildExecutor::new(
            name,
            body,
            opsets,
            Arc::new(WeightStore::new()),
            auto_detect_cpu_ep().unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn child_executor_cached_plan_rebinds_captures_without_stale_state() {
        let mut child = captured_add_child("capture-shadowing");
        let a_input = Tensor::from_f32(&[1], &[1.0]).unwrap();
        let b_input = Tensor::from_f32(&[2], &[2.0, 3.0]).unwrap();

        let mut scope = HashMap::new();
        scope.insert(
            "captured".to_string(),
            Tensor::from_f32(&[1], &[10.0]).unwrap(),
        );
        assert_eq!(
            child.run(&[&a_input], &scope).unwrap()[0].to_vec_f32(),
            vec![11.0]
        );

        scope.insert(
            "captured".to_string(),
            Tensor::from_f32(&[2], &[20.0, 30.0]).unwrap(),
        );
        assert_eq!(
            child.run(&[&b_input], &scope).unwrap()[0].to_vec_f32(),
            vec![22.0, 33.0]
        );

        scope.insert(
            "captured".to_string(),
            Tensor::from_f32(&[1], &[40.0]).unwrap(),
        );
        let cached = child.run(&[&a_input], &scope).unwrap()[0].to_vec_f32();
        let mut fresh = captured_add_child("capture-shadowing-fresh");
        let freshly_compiled = fresh.run(&[&a_input], &scope).unwrap()[0].to_vec_f32();

        assert_eq!(cached, vec![41.0]);
        assert_eq!(cached, freshly_compiled);
        assert_eq!(child.stats(), ChildExecutorStats { builds: 2, runs: 3 });
    }

    // --- weight-streaming: zero-copy borrowed initializer buffers -----------

    use onnx_runtime_ir::{WeightRef, static_shape};
    use std::path::PathBuf;

    /// A writable scratch dir under the workspace `target/` (never `/tmp`).
    fn weightstream_tmp_dir() -> PathBuf {
        let dir = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../target/weightstream_test"
        ));
        std::fs::create_dir_all(&dir).expect("create weight-streaming test dir");
        dir
    }

    fn f32_le(data: &[f32]) -> Vec<u8> {
        data.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// (b) An aligned external-data initializer is backed **zero-copy** by a
    /// borrowed buffer whose data pointer EQUALS the WeightStore's mmap slice —
    /// no allocation, no copy. A model larger than RAM relies on this.
    #[test]
    fn aligned_external_initializer_is_borrowed_zero_copy() {
        let align = TensorLayout::contiguous().alignment;
        let path = weightstream_tmp_dir().join("aligned_init.bin");
        let w_data = [1.0f32, 2.0, 3.0, 4.0];
        std::fs::write(&path, f32_le(&w_data)).unwrap();

        let mut store = WeightStore::new();
        store.map_external(&path).unwrap();

        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let w = g.create_named_value("W", DataType::Float32, static_shape([4]));
        g.set_initializer(
            w,
            WeightRef::External {
                path: path.clone(),
                offset: 0, // mmap base is page-aligned -> 0 is `align`-aligned
                length: 16,
                dtype: DataType::Float32,
                dims: vec![4],
            },
        );
        let y = g.create_value(DataType::Float32, static_shape([4]));
        g.insert_node(Node::new(NodeId(0), "Relu", vec![Some(w)], vec![y]));
        g.add_output(y);

        let ep = auto_detect_cpu_ep().unwrap();
        let exec = Executor::build(g, Arc::new(store), ep).unwrap();

        let weight = &exec.graph.initializers[&w];
        let src = exec.weights().bytes(weight).unwrap();
        assert!(
            (src.as_ptr() as usize).is_multiple_of(align),
            "mmap window must be aligned for this test to exercise the zero-copy path"
        );
        let buf = &exec.buffers[&w];
        assert!(
            buf.is_borrowed(),
            "aligned initializer must be borrowed, not copied"
        );
        assert_eq!(
            buf.as_ptr() as *const u8,
            src.as_ptr(),
            "zero-copy: the buffer must alias the mmap bytes (no copy)"
        );

        let _ = std::fs::remove_file(&path);
    }

    /// (c) An external-data initializer that is dtype-aligned but not 64-byte
    /// aligned remains a zero-copy mmap borrow and is numerically correct.
    #[test]
    fn device_unaligned_external_initializer_is_borrowed_at_dtype_alignment() {
        let align = TensorLayout::contiguous().alignment;
        let path = weightstream_tmp_dir().join("unaligned_init.bin");
        // Prefix the weight window with 8 bytes so it starts at offset 8, which
        // is f32-aligned but not a multiple of the EP allocation alignment (64).
        let offset = 8usize;
        let w_data = [5.0f32, 6.0, 7.0, 8.0];
        let mut file = vec![0u8; offset];
        file.extend_from_slice(&f32_le(&w_data));
        std::fs::write(&path, &file).unwrap();

        let mut store = WeightStore::new();
        store.map_external(&path).unwrap();

        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let w = g.create_named_value("W", DataType::Float32, static_shape([4]));
        g.set_initializer(
            w,
            WeightRef::External {
                path: path.clone(),
                offset,
                length: 16,
                dtype: DataType::Float32,
                dims: vec![4],
            },
        );
        let x = g.create_named_value("X", DataType::Float32, static_shape([4]));
        g.add_input(x);
        let y = g.create_value(DataType::Float32, static_shape([4]));
        g.insert_node(Node::new(NodeId(0), "Add", vec![Some(x), Some(w)], vec![y]));
        g.add_output(y);

        let ep = auto_detect_cpu_ep().unwrap();
        let mut exec = Executor::build(g, Arc::new(store), ep).unwrap();

        let weight = &exec.graph.initializers[&w];
        let src = exec.weights().bytes(weight).unwrap();
        assert!(
            !(src.as_ptr() as usize).is_multiple_of(align),
            "window must be unaligned for this test to exercise the fallback"
        );
        let buf = &exec.buffers[&w];
        assert!(
            buf.is_borrowed(),
            "dtype-aligned mmap initializer must remain borrowed"
        );
        assert_eq!(
            buf.as_ptr() as *const u8,
            src.as_ptr(),
            "zero-copy buffer must alias the mmap window"
        );
        assert_eq!(buf.alignment(), std::mem::align_of::<f32>());

        // The copy is numerically correct: Y = X + W.
        let x_tensor = Tensor::from_f32(&[4], &[10.0, 20.0, 30.0, 40.0]).unwrap();
        let out = exec.run(&[("X", &x_tensor)]).unwrap();
        assert_eq!(out.len(), 1);
        let got = out[0].to_vec_f32();
        let want = [15.0f32, 26.0, 37.0, 48.0];
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-5, "got {g}, want {w}");
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn unaligned_external_qmoe_keeps_route_first_enabled_and_matches_legacy() {
        use std::ffi::OsString;
        use std::sync::{Mutex, OnceLock};

        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        let _env_guard = ENV_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .expect("weight-offload env lock");

        struct RestoreEnv(Option<OsString>);
        impl Drop for RestoreEnv {
            fn drop(&mut self) {
                if let Some(value) = self.0.take() {
                    // SAFETY: this test serializes all mutations it performs.
                    unsafe { std::env::set_var(onnx_runtime_ep_cpu::WEIGHT_OFFLOAD_ENV, value) };
                } else {
                    // SAFETY: this test serializes all mutations it performs.
                    unsafe { std::env::remove_var(onnx_runtime_ep_cpu::WEIGHT_OFFLOAD_ENV) };
                }
            }
        }

        let _restore = RestoreEnv(std::env::var_os(onnx_runtime_ep_cpu::WEIGHT_OFFLOAD_ENV));
        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../onnx-runtime-ep-cpu/tests/fixtures/qmoe_weight_offload/model.onnx");
        let input_values: Vec<f32> = (0..64).map(|index| index as f32 * 0.03125 - 1.0).collect();
        let router_values = vec![
            9.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 9.0,
        ];
        let input = Tensor::from_f32(&[4, 16], &input_values).unwrap();
        let router = Tensor::from_f32(&[4, 4], &router_values).unwrap();

        // SAFETY: guarded above; both executors compile synchronously here.
        unsafe { std::env::set_var(onnx_runtime_ep_cpu::WEIGHT_OFFLOAD_ENV, "0") };
        let (legacy_graph, legacy_weights) =
            onnx_runtime_loader::load_model_with_weights(&fixture).unwrap();
        let mut legacy =
            Executor::build(legacy_graph, legacy_weights, auto_detect_cpu_ep().unwrap()).unwrap();
        let legacy_output = legacy.run(&[("X", &input), ("router", &router)]).unwrap();

        // SAFETY: guarded above; the offload kernel captures the flag at build.
        unsafe { std::env::set_var(onnx_runtime_ep_cpu::WEIGHT_OFFLOAD_ENV, "1") };
        let before = onnx_runtime_ep_cpu::weight_offload_stats();
        let (offload_graph, offload_weights) =
            onnx_runtime_loader::load_model_with_weights(&fixture).unwrap();
        let mut offload = Executor::build(
            offload_graph,
            offload_weights,
            auto_detect_cpu_ep().unwrap(),
        )
        .unwrap();
        for (&value, weight) in &offload.graph.initializers {
            let WeightRef::External { .. } = weight else {
                continue;
            };
            let source = offload.weights.bytes(weight).unwrap();
            assert!(
                !(source.as_ptr() as usize).is_multiple_of(TensorLayout::contiguous().alignment)
            );
            let buffer = &offload.buffers[&value];
            assert!(buffer.is_borrowed());
            assert_eq!(buffer.as_ptr() as *const u8, source.as_ptr());
        }
        let offload_output = offload.run(&[("X", &input), ("router", &router)]).unwrap();
        let after = onnx_runtime_ep_cpu::weight_offload_stats();

        assert_eq!(
            offload_output[0].to_vec_f32(),
            legacy_output[0].to_vec_f32()
        );
        assert!(
            after.layer_executions
                >= before
                    .layer_executions
                    .checked_add(1)
                    .expect("layer execution counter overflow")
        );
        assert!(after.bytes_read_from_mmap > before.bytes_read_from_mmap);
    }

    /// (d) Soundness guard: even when an initializer's mmap bytes are aligned
    /// (so the zero-copy path would otherwise fire), the executor must NOT
    /// borrow them if the value also has a producer — i.e. a malformed graph
    /// reused the initializer's `ValueId` as a node output. Borrowing yields a
    /// read-only buffer; a kernel writing that output would write through the
    /// mmap (SIGSEGV / aliasing UB). The build must fall back to an owned,
    /// writable copy instead.
    #[test]
    fn producer_backed_initializer_is_not_borrowed() {
        let align = TensorLayout::contiguous().alignment;
        let path = weightstream_tmp_dir().join("producer_backed_init.bin");
        let w_data = [1.0f32, 2.0, 3.0, 4.0];
        std::fs::write(&path, f32_le(&w_data)).unwrap();

        let mut store = WeightStore::new();
        store.map_external(&path).unwrap();

        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let x = g.create_named_value("X", DataType::Float32, static_shape([4]));
        g.add_input(x);
        let w = g.create_named_value("W", DataType::Float32, static_shape([4]));
        g.set_initializer(
            w,
            WeightRef::External {
                path: path.clone(),
                offset: 0, // aligned: without the producer guard this would borrow
                length: 16,
                dtype: DataType::Float32,
                dims: vec![4],
            },
        );
        // Reuse the initializer's ValueId as a node output -> gives `w` a
        // producer, exactly the malformed shape the loader also rejects.
        g.insert_node(Node::new(NodeId(0), "Identity", vec![Some(x)], vec![w]));
        let y = g.create_value(DataType::Float32, static_shape([4]));
        g.insert_node(Node::new(NodeId(1), "Add", vec![Some(x), Some(w)], vec![y]));
        g.add_output(y);

        assert!(
            g.value(w).producer.is_some(),
            "test setup: initializer value must have a producer",
        );

        let ep = auto_detect_cpu_ep().unwrap();
        let exec = Executor::build(g, Arc::new(store), ep).unwrap();

        let weight = &exec.graph.initializers[&w];
        let src = exec.weights().bytes(weight).unwrap();
        assert!(
            (src.as_ptr() as usize).is_multiple_of(align),
            "mmap window must be aligned so only the producer guard prevents borrowing",
        );
        let buf = &exec.buffers[&w];
        assert!(
            !buf.is_borrowed(),
            "producer-backed initializer must fall back to an owned writable copy",
        );
        assert_ne!(
            buf.as_ptr() as *const u8,
            src.as_ptr(),
            "producer-backed initializer must not alias read-only mmap bytes",
        );

        let _ = std::fs::remove_file(&path);
    }
}

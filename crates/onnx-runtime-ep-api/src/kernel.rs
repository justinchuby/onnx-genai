//! The [`Kernel`] trait and kernel-match / cost types (§4.2).

use std::borrow::Cow;

use onnx_runtime_ir::{DataType, TensorLayout};
use onnx_runtime_memory_governor::MemoryRole;

use crate::error::Result;
use crate::tensor::{DevicePtrMut, TensorMut, TensorView};
use crate::weight::WeightHandle;

/// A cost estimate for running a kernel, consumed by the placement cost model
/// (`docs/architecture/ORT2.md` §6). All time fields are in **microseconds**; a fuller model
/// (roofline, calibration) lands in `onnx-runtime-cost-model` (Phase 2).
///
/// The struct is `#[non_exhaustive]`: the Phase-2 cost model may add fields
/// (e.g. energy, occupancy) without breaking EP crates. Construct it via
/// [`Cost::ZERO`], [`Cost::new`], or the `with_*` builders rather than a struct
/// literal so those additions stay source-compatible.
///
/// The three time components (`compute_us`, `memory_us`, `transfer_us`) map onto
/// the design's *compute*, *memory-traffic*, and *layout/transfer* estimates;
/// `launch_us` captures fixed dispatch latency (§6.2 `launch_overhead`) and
/// `bytes_moved` carries the raw memory-traffic figure a roofline model needs
/// (§6.3, mirroring the design `Cost::memory_bytes`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct Cost {
    /// Estimated compute time (µs).
    pub compute_us: f64,
    /// Estimated memory-traffic time (µs).
    pub memory_us: f64,
    /// Estimated layout-conversion / cross-device copy time at boundaries (µs).
    pub transfer_us: f64,
    /// Fixed kernel-launch / dispatch latency (µs), independent of size.
    pub launch_us: f64,
    /// Estimated bytes of memory traffic (for roofline / bandwidth models).
    pub bytes_moved: u64,
}

impl Cost {
    /// The zero cost (free op).
    pub const ZERO: Cost = Cost {
        compute_us: 0.0,
        memory_us: 0.0,
        transfer_us: 0.0,
        launch_us: 0.0,
        bytes_moved: 0,
    };

    /// A cost from its three time components; `launch_us` and `bytes_moved`
    /// default to zero (set them via the builders).
    pub fn new(compute_us: f64, memory_us: f64, transfer_us: f64) -> Self {
        Self {
            compute_us,
            memory_us,
            transfer_us,
            ..Self::ZERO
        }
    }

    /// Set the fixed launch/dispatch latency.
    pub fn with_launch_us(mut self, launch_us: f64) -> Self {
        self.launch_us = launch_us;
        self
    }

    /// Set the estimated memory-traffic volume.
    pub fn with_bytes_moved(mut self, bytes_moved: u64) -> Self {
        self.bytes_moved = bytes_moved;
        self
    }

    /// Total estimated wall time (µs): the sum of all time components.
    pub fn total_us(&self) -> f64 {
        self.compute_us + self.memory_us + self.transfer_us + self.launch_us
    }
}

/// Structural memory-traffic estimate (bytes) for a node's inputs, honest about
/// the real element type and sub-byte packing.
///
/// This is the *machine-independent* half of a kernel's cost (issue #995): a
/// property of the shapes and dtypes, not of the host. It replaces the old
/// `elems * 4` fabrication, which assumed every tensor was f32 and was wrong by
/// 8× for the int4 weights that dominate decode. Each input's element count is
/// multiplied by its own dtype's *storage* size via
/// [`DataType::checked_storage_bytes`], so a packed int4 weight counts as
/// `ceil(numel/2)` bytes, an f16 tensor as `2*numel`, and so on.
///
/// Symbolic (dynamic) dimensions are treated as extent 1 so the estimate stays
/// finite and monotonic in the statically-known problem size; it is a lower
/// bound when a dimension is unknown, never an invented larger number. Absent
/// inputs (dtype/shape length mismatch) contribute nothing.
pub fn structural_input_bytes(
    shapes: &[onnx_runtime_ir::Shape],
    input_dtypes: &[onnx_runtime_ir::DataType],
) -> u64 {
    shapes
        .iter()
        .zip(input_dtypes.iter())
        .map(|(shape, &dtype)| {
            let numel: usize = shape
                .iter()
                .map(|d| d.as_static().unwrap_or(1))
                .product::<usize>();
            dtype.checked_storage_bytes(numel).unwrap_or(usize::MAX) as u64
        })
        .fold(0_u64, u64::saturating_add)
}

/// Result of [`crate::ExecutionProvider::supports_op`].
///
/// A decline reason travels with the decision that produced it. EPs must not
/// maintain a separate reason table: colocating the reason with `Unsupported`
/// keeps diagnostics from drifting away from the actual claim predicate.
#[derive(Debug)]
pub enum KernelMatch {
    Supported {
        cost: Cost,
        /// Layouts the kernel requires for each input, if constrained.
        required_input_layouts: Option<Vec<TensorLayout>>,
        /// Layouts the kernel produces for each output.
        output_layouts: Vec<TensorLayout>,
    },
    Unsupported {
        /// Actionable explanation of what the EP accepts or how to fix fallback.
        reason: Cow<'static, str>,
    },
}

impl KernelMatch {
    /// Construct an unsupported match with its actionable decline reason.
    pub fn unsupported(reason: impl Into<Cow<'static, str>>) -> Self {
        Self::Unsupported {
            reason: reason.into(),
        }
    }

    /// Whether the op is supported.
    pub fn is_supported(&self) -> bool {
        matches!(self, KernelMatch::Supported { .. })
    }

    /// The EP's decline reason, or `None` when the op is supported.
    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Supported { .. } => None,
            Self::Unsupported { reason } => Some(reason),
        }
    }
}

/// Whether a compiled kernel can participate in device-graph capture.
///
/// As with [`KernelMatch`], a decline reason travels with the decision that
/// produced it. EPs must not maintain a separate reason table: capture
/// eligibility is often shape-, dtype-, warmup-, or implementation-dependent,
/// so separating the reason from the predicate would let diagnostics drift.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CaptureSupport {
    /// The kernel's warmed execution path satisfies the device-graph contract.
    Supported,
    /// The kernel cannot currently be captured.
    Unsupported {
        /// Actionable explanation of the failed capture precondition.
        reason: Cow<'static, str>,
    },
}

impl CaptureSupport {
    /// Construct an unsupported result with its actionable decline reason.
    pub fn unsupported(reason: impl Into<Cow<'static, str>>) -> Self {
        Self::Unsupported {
            reason: reason.into(),
        }
    }

    /// Whether the kernel can currently participate in device-graph capture.
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Supported)
    }

    /// The kernel's capture-decline reason, or `None` when capture is supported.
    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Supported => None,
            Self::Unsupported { reason } => Some(reason),
        }
    }
}

/// The concrete implementation selected by a kernel's internal dispatcher.
///
/// Unlike [`KernelMatch`] (the EP's claim over a node) and [`CaptureSupport`]
/// (graph-capture eligibility), this records **which** implementation ran for an
/// already-claimed node and **why** the dispatch predicate chose it. A single
/// claimed op (e.g. `MatMulNBits`) can pick materially different kernels for the
/// same shape family, so node-level claims alone do not explain what executed.
/// The reason travels with the variant so a trace explains both.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelVariantSelection {
    /// Stable implementation name suitable for filtering and aggregation.
    pub variant: &'static str,
    /// Human-readable explanation of the dispatch predicate that selected it.
    pub reason: Cow<'static, str>,
}

impl KernelVariantSelection {
    /// Construct a selected variant with its dispatch reason.
    pub fn new(variant: &'static str, reason: impl Into<Cow<'static, str>>) -> Self {
        Self {
            variant,
            reason: reason.into(),
        }
    }
}

/// Trace-argument key carrying the selected kernel implementation.
pub const ARG_KERNEL_VARIANT: &str = "kernel_variant";
/// Trace-argument key carrying why the kernel variant was selected.
pub const ARG_KERNEL_VARIANT_REASON: &str = "kernel_variant_reason";
/// Trace-argument key naming the device a kernel ran on (`cpu`, `cuda`, ...).
pub const ARG_DEVICE: &str = "device";
/// Trace-argument key carrying bytes moved by a kernel (inputs plus outputs).
pub const ARG_BYTES: &str = "bytes";
/// Trace-argument key carrying a kernel's floating-point operation count.
pub const ARG_FLOPS: &str = "flops";
/// Trace category for a span covering one worker's slice of a fanned-out node.
pub const CAT_KERNEL_WORKER: &str = "op.worker";

/// Record the standard per-kernel metrics on the active executor op-span.
///
/// This is the one implementation every execution provider shares, so a trace
/// means the same thing whichever provider produced it: `device` names where
/// the kernel ran, `bytes` is inputs plus outputs (absent inputs excluded), and
/// `flops` is the provider's own estimate. Providers previously each owned a
/// private copy of this, which is how the CUDA provider ended up recording
/// nothing at all.
///
/// `flops` is a closure so the estimate is never computed on the untraced path.
#[inline]
pub fn record_kernel_metrics(
    device: &'static str,
    inputs: &[TensorView<'_>],
    outputs: &[TensorMut<'_>],
    flops: impl FnOnce() -> u64,
) {
    if !kernel_variant_tracing_enabled() {
        return;
    }
    onnx_runtime_tracer::annotate_current_span_with(|| {
        let input_bytes = inputs
            .iter()
            .filter(|input| !input.is_absent())
            .fold(0_u64, |total, input| {
                total.saturating_add(input.byte_size() as u64)
            });
        let bytes = outputs.iter().fold(input_bytes, |total, output| {
            total.saturating_add(output.byte_size() as u64)
        });
        onnx_runtime_tracer::Args::new()
            .with(ARG_DEVICE, device)
            .with(ARG_BYTES, bytes)
            .with(ARG_FLOPS, flops())
    });
}

/// Open a span covering one worker's slice of a node that was fanned out across
/// a thread pool or stream, returning `None` when it should not be recorded.
///
/// The executor's op-span lives on the thread that dispatched the node, and
/// thread-local span state does not follow work onto a pool's threads. A worker
/// must therefore open its own span, which is what this does: it lands on that
/// worker's own trace lane and shows the parallel decomposition that the single
/// op-span flattens away.
///
/// Two things keep this affordable. It is gated at [`TraceVerbosity::Full`], so
/// the default op-level trace is unaffected; and a span costs a few hundred
/// nanoseconds, which is noise against a slice of a node but *not* against an
/// inner loop iteration. **Wrap a worker's whole chunk of work, never an inner
/// loop.**
///
/// `label` should identify the partitioned work (typically the op type), not
/// the worker: the worker is already identified by the lane the span lands on.
#[must_use]
#[inline]
pub fn kernel_worker_span(label: &'static str) -> Option<onnx_runtime_tracer::SpanGuard> {
    let context = onnx_runtime_tracer::global_context()?;
    if !context
        .verbosity()
        .includes(onnx_runtime_tracer::TraceVerbosity::Full)
    {
        return None;
    }
    Some(context.span(label, CAT_KERNEL_WORKER))
}

/// Whether kernel-variant trace annotations would currently be recorded.
///
/// This is true only when an enabled executor op-span is active on the current
/// thread, so callers can guard shape-dependent reason formatting behind it and
/// keep the disabled dispatch path allocation-free. It is the cheap thread-local
/// check that closes the "dead write" gap: without a live span, an annotation
/// has nothing to attach to, so there is no point formatting a reason.
#[must_use]
#[inline]
pub fn kernel_variant_tracing_enabled() -> bool {
    onnx_runtime_tracer::tracing_active()
}

/// Record a selected kernel variant on the active runtime trace span.
///
/// The annotation enriches the per-op span the executor opens for a traced run,
/// so it carries the node identity already. Callers should normally use
/// [`record_kernel_variant!`] so formatted reasons are built only when a span is
/// active.
#[inline]
pub fn record_kernel_variant_selection(selection: &KernelVariantSelection) {
    if !kernel_variant_tracing_enabled() {
        return;
    }
    onnx_runtime_tracer::annotate_current_span_with(|| {
        onnx_runtime_tracer::Args::new()
            .with(ARG_KERNEL_VARIANT, selection.variant)
            .with(ARG_KERNEL_VARIANT_REASON, selection.reason.as_ref())
    });
}

/// Record a selected kernel variant for a named *sub-decision* of the current
/// op.
///
/// A single claimed node can make several independent kernel-path choices in
/// sequence (e.g. `GroupQueryAttention` picks a prep-fusion path *and* an
/// attention-backend path). Recording each under the shared
/// [`ARG_KERNEL_VARIANT`] key would let a later choice overwrite an earlier one
/// on the same span, so each sub-decision is namespaced by `stage` into
/// `kernel_variant.<stage>` / `kernel_variant_reason.<stage>`. Use the plain
/// [`record_kernel_variant_selection`] for the node's terminal/primary variant.
#[inline]
pub fn record_kernel_variant_stage_selection(stage: &str, selection: &KernelVariantSelection) {
    if !kernel_variant_tracing_enabled() {
        return;
    }
    let variant_key = format!("{ARG_KERNEL_VARIANT}.{stage}");
    let reason_key = format!("{ARG_KERNEL_VARIANT_REASON}.{stage}");
    onnx_runtime_tracer::annotate_current_span_with(|| {
        onnx_runtime_tracer::Args::new()
            .with(variant_key, selection.variant)
            .with(reason_key, selection.reason.as_ref())
    });
}

/// Record a selected kernel implementation and lazily formatted reason on the
/// active executor op-span.
///
/// Formatting is skipped entirely unless a span is active
/// ([`kernel_variant_tracing_enabled`]), so instrumented dispatch sites stay
/// allocation-free on the hot decode path when tracing is off.
#[macro_export]
macro_rules! record_kernel_variant {
    ($variant:expr, $($arg:tt)+) => {{
        if $crate::kernel_variant_tracing_enabled() {
            let selection = $crate::KernelVariantSelection::new(
                $variant,
                ::std::format!($($arg)+),
            );
            $crate::record_kernel_variant_selection(&selection);
        }
    }};
}

/// Record a named *sub-decision* kernel variant on the active executor op-span.
///
/// Like [`record_kernel_variant!`] but namespaces the annotation under `$stage`
/// so multiple kernel-path choices made while executing one node do not clobber
/// each other. Formatting is skipped entirely unless a span is active.
#[macro_export]
macro_rules! record_kernel_variant_stage {
    ($stage:expr, $variant:expr, $($arg:tt)+) => {{
        if $crate::kernel_variant_tracing_enabled() {
            let selection = $crate::KernelVariantSelection::new(
                $variant,
                ::std::format!($($arg)+),
            );
            $crate::record_kernel_variant_stage_selection($stage, &selection);
        }
    }};
}

/// Decline the current capture-support query with an actionable reason.
///
/// Formatting is evaluated only on the decline path.
#[macro_export]
macro_rules! decline_capture {
    ($($arg:tt)+) => {
        return $crate::CaptureSupport::unsupported(format!($($arg)+))
    };
}

/// Require a capture precondition, declining with an actionable reason when false.
///
/// Formatting is evaluated only when the condition fails.
#[macro_export]
macro_rules! require_capture {
    ($condition:expr, $($arg:tt)+) => {
        if !$condition {
            $crate::decline_capture!($($arg)+);
        }
    };
}

/// Decline the current `supports_op` call with an actionable reason.
///
/// Formatting is evaluated only on the decline path.
#[macro_export]
macro_rules! deny {
    ($($arg:tt)+) => {
        return $crate::KernelMatch::unsupported(format!($($arg)+))
    };
}

/// Require a claim condition, declining with an actionable reason when false.
///
/// Formatting is evaluated only when the condition fails, keeping the hot
/// supported path allocation-free.
#[macro_export]
macro_rules! require {
    ($condition:expr, $($arg:tt)+) => {
        if !$condition {
            $crate::deny!($($arg)+);
        }
    };
}

/// A zero-copy **view output**: a kernel's declaration that one of its outputs
/// is a strided view aliasing one of its inputs' buffers, rather than freshly
/// computed bytes (`docs/architecture/ORT2.md` §5.4, lazy PyTorch-style views).
///
/// The `shape` / `strides` / `byte_offset` describe the output tensor relative
/// to the **same base pointer** as the referenced input view (i.e. relative to
/// the input's backing allocation, honoring any offset that input itself
/// already carried). Strides are in **elements** and may be negative (DLPack).
/// The executor records this metadata against the output value and does **not**
/// allocate a buffer or invoke the compute path for that slot; the source
/// buffer is kept alive until the view's consumers have run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ViewOutput {
    /// Positional index (into the kernel's `inputs`) of the aliased input.
    pub input_index: usize,
    /// Output shape.
    pub shape: Vec<usize>,
    /// Output element strides relative to the aliased input's base pointer.
    pub strides: Vec<i64>,
    /// Byte offset of the output element origin from the aliased input's base.
    pub byte_offset: usize,
}

/// One data-dependent output produced in owned host memory before the host
/// runtime allocates its final tensor.
///
/// This is an opt-in escape hatch for operators such as `Unique`, where
/// discovering an output extent is the algorithm itself. The bytes own their
/// storage and therefore cannot borrow an input or kernel temporary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelSizedOutput {
    pub shape: Vec<usize>,
    pub dtype: DataType,
    pub bytes: Vec<u8>,
}

/// An executor-delivered kernel input. Existing EPs receive `Tensor` variants;
/// an EP advertising the `nxrt` capability may receive a lazy `Weight` at the
/// `pkg.nxrt::BlockQuantizedMoE` boundary.
pub enum KernelInput<'a> {
    Tensor(TensorView<'a>),
    Weight(&'a WeightHandle),
}

impl<'a> KernelInput<'a> {
    pub fn tensor(&self) -> Option<&TensorView<'a>> {
        match self {
            Self::Tensor(view) => Some(view),
            Self::Weight(_) => None,
        }
    }

    pub fn weight(&self) -> Option<&WeightHandle> {
        match self {
            Self::Tensor(_) => None,
            Self::Weight(weight) => Some(weight),
        }
    }
}

/// Concrete tensor facts available during prepare-only graph traversal.
#[derive(Clone, Copy, Debug)]
pub struct TensorMetadata<'a> {
    pub dtype: onnx_runtime_ir::DataType,
    pub shape: &'a [usize],
    pub present: bool,
}

impl<'a> TensorMetadata<'a> {
    pub const fn new(dtype: onnx_runtime_ir::DataType, shape: &'a [usize], present: bool) -> Self {
        Self {
            dtype,
            shape,
            present,
        }
    }
}

/// How long prepared kernel workspace remains live.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkspaceLifetime {
    StepScoped,
    SessionPersistent,
}

/// A kernel's exact owned-scratch request for one concrete input geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspaceRequirement {
    pub bytes: u64,
    pub alignment: usize,
    pub lifetime: WorkspaceLifetime,
    pub role: MemoryRole,
}

impl WorkspaceRequirement {
    pub const NONE: Self = Self {
        bytes: 0,
        alignment: 1,
        lifetime: WorkspaceLifetime::StepScoped,
        role: MemoryRole::Workspace { step_scoped: true },
    };
}

/// Executor-owned prepared workspace handed to a kernel for one dispatch.
#[derive(Clone, Copy, Debug)]
pub struct WorkspaceView {
    ptr: DevicePtrMut,
    bytes: usize,
}

impl WorkspaceView {
    pub const fn new(ptr: DevicePtrMut, bytes: usize) -> Self {
        Self { ptr, bytes }
    }

    pub const fn ptr(self) -> DevicePtrMut {
        self.ptr
    }

    pub const fn bytes(self) -> usize {
        self.bytes
    }
}

/// A kernel ready to execute a specific op with specific shapes (§4.2).
pub trait Kernel: Send {
    /// Tell the kernel which positional inputs are immutable graph constants.
    ///
    /// The session calls this exactly once, immediately after construction.
    /// Kernels may use it to prepack or memoize those inputs. Runtime inputs must
    /// never be marked constant: caching them would return stale results.
    fn set_constant_inputs(&mut self, constant_inputs: &[bool]) {
        let _ = constant_inputs;
    }

    /// Tell the kernel whether all of this node's outputs have fully-static
    /// (sequence-independent) symbolic shapes, as derived from the graph's IR
    /// metadata — **not** from runtime shape values.
    ///
    /// The session calls this exactly once, immediately after construction. A
    /// kernel that gates CUDA-graph capture on sequence-independence (the
    /// pointwise/elementwise/bitwise/prelu family) uses this to admit a
    /// fully-static head-major decode shape (e.g. `[1,1,heads,dim]`) that the
    /// runtime-extent heuristic cannot recognize, while a shape carrying a
    /// symbolic (growing sequence) dimension stays eager. Deriving eligibility
    /// from metadata is essential: `[heads=32, feat=128]` (capturable) is
    /// indistinguishable from `[tokens=32, feat=128]` (unsafe) by extent alone.
    /// Default: no-op.
    fn set_capture_seq_independent(&mut self, seq_independent: bool) {
        let _ = seq_independent;
    }

    /// Execute over device-resident inputs/outputs.
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()>;

    /// Whether this kernel opts into Compute-time, kernel-sized outputs.
    ///
    /// The ORT plugin adapter uses this only for nodes whose shape strategy is
    /// explicitly marked kernel-sized. Existing kernels keep the ordinary
    /// infer-shape, allocate, execute path unchanged.
    fn has_kernel_sized_outputs(&self) -> bool {
        false
    }

    /// Execute once into owned host buffers and report concrete output facts.
    ///
    /// `requested_outputs` preserves node output-slot positions. A `false`
    /// slot is an omitted optional output and must be returned as `None`; a
    /// `true` slot must be returned as `Some`. The adapter validates every
    /// dtype, checked shape byte length, and buffer length before allocating
    /// final outputs, then performs one materialization copy per present slot.
    ///
    /// The first implementation is intentionally host-only. Implementations
    /// must reject non-host-accessible inputs before reading them.
    fn execute_kernel_sized(
        &self,
        inputs: &[TensorView],
        requested_outputs: &[bool],
    ) -> Result<Vec<Option<KernelSizedOutput>>> {
        let _ = (inputs, requested_outputs);
        Err(crate::EpError::KernelFailed(
            "kernel does not implement kernel-sized outputs".into(),
        ))
    }

    /// Report exact owned scratch needed for these concrete inputs.
    fn workspace_requirement(&self, inputs: &[TensorMetadata<'_>]) -> Result<WorkspaceRequirement> {
        let _ = inputs;
        Ok(WorkspaceRequirement::NONE)
    }

    /// Refine the workspace requirement with runtime-visible tensor inputs.
    ///
    /// Prepare-only reservation walks have only [`TensorMetadata`], so
    /// [`Kernel::workspace_requirement`] remains the planning hook. Execution
    /// dispatch, though, may already hold device views whose *values* determine
    /// the exact scratch size (for example a reduction whose axes come from an
    /// input tensor). The default preserves the metadata-only contract.
    fn workspace_requirement_for_execution(
        &self,
        inputs: &[TensorView],
        metadata: &[TensorMetadata<'_>],
    ) -> Result<WorkspaceRequirement> {
        let _ = inputs;
        self.workspace_requirement(metadata)
    }

    /// Execute using workspace prepared before request admission.
    fn execute_with_workspace(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        let _ = workspace;
        self.execute(inputs, outputs)
    }

    /// Execute through the general weight-delivery seam.
    ///
    /// The default adapter accepts only resident tensor inputs and forwards to
    /// [`Kernel::execute`], so existing EPs compile and behave identically.
    /// Paging-aware kernels override this method to consume lazy handles.
    fn execute_with_inputs(
        &self,
        inputs: &[KernelInput<'_>],
        outputs: &mut [TensorMut],
    ) -> Result<()> {
        let views = inputs
            .iter()
            .map(|input| {
                input.tensor().copied().ok_or_else(|| {
                    crate::EpError::KernelFailed(
                        "kernel received a lazy WeightHandle without implementing \
                         execute_with_inputs"
                            .into(),
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.execute(&views, outputs)
    }

    /// Estimated FLOPs, if known (for the cost model).
    fn estimated_flops(&self) -> Option<u64> {
        None
    }

    /// Attempt to express this node's outputs as zero-copy [`ViewOutput`]s over
    /// its inputs instead of computing bytes (the layout/movement-op fast path).
    ///
    /// `inputs` carries the real (possibly already-strided) input views,
    /// `output_shapes` carries the executor-resolved concrete shape of every
    /// output (already computed by shape inference, so a kernel need not — and
    /// on a device EP during CUDA-graph capture must not — re-derive a
    /// data-dependent output shape from a device-resident shape operand), and
    /// `num_outputs` is the node's output arity (`== output_shapes.len()`).
    /// Returning:
    /// * `None` — the default — means "compute normally": the executor allocates
    ///   output buffers and calls [`Kernel::execute`].
    /// * `Some(specs)` means every output is a view; `specs.len()` MUST equal
    ///   `num_outputs`. A kernel that can view some but not all outputs must
    ///   return `None` (all-or-nothing) so correctness never regresses.
    ///
    /// When `Some` is returned, [`Kernel::execute`] is **not** invoked.
    fn view_outputs(
        &self,
        inputs: &[TensorView],
        output_shapes: &[Vec<usize>],
        num_outputs: usize,
    ) -> Option<Vec<ViewOutput>> {
        let _ = (inputs, output_shapes, num_outputs);
        None
    }

    /// Whether [`Self::view_outputs`] can ever return a zero-copy alias.
    ///
    /// Heterogeneous execution uses independently owned boundary buffers. Its
    /// first slice rejects view-producing kernels before execution rather than
    /// discovering an alias after an upstream partition has already run.
    fn may_produce_views(&self) -> bool {
        false
    }

    /// Whether the output may overwrite the input at `input_index`.
    ///
    /// This is an opt-in semantic promise: the executor performs the separate
    /// liveness, ownership, shape, dtype, and layout checks before it reuses an
    /// input allocation. Kernels must return `false` unless their read/write
    /// ordering is correct when the two tensor ranges are identical.
    fn can_run_in_place(&self, input_index: usize) -> bool {
        let _ = input_index;
        false
    }

    /// Whether the kernel accepts a non-contiguous (strided) input at `idx`.
    fn supports_strided_input(&self, input_idx: usize) -> bool {
        let _ = input_idx;
        false
    }

    /// The layout the kernel writes most efficiently, if it has a preference.
    fn preferred_output_layout(&self) -> Option<TensorLayout> {
        None
    }

    /// Whether this kernel can participate in device-graph capture.
    ///
    /// The provided default is supported because capture is a runtime/EP
    /// capability: kernels that need restrictions override this method and
    /// return the exact failed precondition.
    fn capture_support(&self) -> CaptureSupport {
        CaptureSupport::Supported
    }

    /// Compatibility shim for existing CUDA-graph callers.
    ///
    /// New capture gates must use [`Kernel::capture_support`] so a decline
    /// reason is never discarded.
    fn cuda_graph_compatible(&self) -> bool {
        self.capture_support().is_supported()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::*;
    use crate::{DeviceId, DevicePtr};

    #[test]
    fn cost_zero_and_total() {
        assert_eq!(Cost::ZERO.total_us(), 0.0);
        let c = Cost::new(10.0, 5.0, 2.0);
        assert_eq!(c.total_us(), 17.0);
        assert_eq!(c.bytes_moved, 0);
        assert_eq!(c.launch_us, 0.0);
    }

    #[test]
    fn cost_builders_are_additive() {
        let c = Cost::new(10.0, 5.0, 2.0)
            .with_launch_us(3.0)
            .with_bytes_moved(4096);
        // launch time folds into the total; bytes_moved is metadata for roofline.
        assert_eq!(c.total_us(), 20.0);
        assert_eq!(c.bytes_moved, 4096);
    }

    #[test]
    fn kernel_match_reports_support() {
        let supported = KernelMatch::Supported {
            cost: Cost::new(1.0, 0.0, 0.0),
            required_input_layouts: None,
            output_layouts: vec![],
        };
        assert!(supported.is_supported());
        assert_eq!(supported.reason(), None);

        let unsupported = KernelMatch::unsupported("test EP supports no ops");
        assert!(!unsupported.is_supported());
        assert_eq!(unsupported.reason(), Some("test EP supports no ops"));
    }

    fn require_positive(value: i32) -> KernelMatch {
        require!(value > 0, "value must be positive, got {value}");
        KernelMatch::Supported {
            cost: Cost::ZERO,
            required_input_layouts: None,
            output_layouts: vec![],
        }
    }

    #[test]
    fn require_macro_carries_formatted_decline_reason() {
        let rejected = require_positive(-2);
        assert_eq!(rejected.reason(), Some("value must be positive, got -2"));
        assert!(require_positive(2).is_supported());
    }

    #[test]
    fn kernel_variant_selection_carries_winner_and_reason() {
        let selection =
            KernelVariantSelection::new("gemv", "M==1 decode uses the packed int4 GEMV path");
        assert_eq!(selection.variant, "gemv");
        assert_eq!(
            selection.reason,
            "M==1 decode uses the packed int4 GEMV path"
        );
    }

    #[test]
    fn record_kernel_variant_without_active_span_skips_formatting() {
        // No executor span is open here, so `tracing_active()` is false: the
        // macro must not evaluate its format arguments (the dead-write guard).
        let formatted = AtomicBool::new(false);
        record_kernel_variant!("gemv", "{}", {
            formatted.store(true, Ordering::Relaxed);
            "M==1 decode"
        });
        assert!(!formatted.load(Ordering::Relaxed));
    }

    #[test]
    fn record_kernel_variant_annotates_active_span() {
        use onnx_runtime_tracer::TraceContext;
        let (trace, events) = TraceContext::in_memory();
        {
            let _span = trace.span("MatMulNBits", "op");
            record_kernel_variant!(
                "gemv_f16",
                "K={}, N={} chose the vectorized GEMV",
                4864,
                896
            );
        }
        let events = events.events();
        assert_eq!(events.len(), 1);
        let args = events[0].args.as_ref().unwrap();
        assert_eq!(args[ARG_KERNEL_VARIANT], "gemv_f16");
        assert!(
            args[ARG_KERNEL_VARIANT_REASON]
                .as_str()
                .unwrap()
                .contains("K=4864, N=896")
        );
    }

    #[test]
    fn record_kernel_variant_stage_namespaces_multiple_subdecisions() {
        use onnx_runtime_tracer::TraceContext;
        let (trace, events) = TraceContext::in_memory();
        {
            let _span = trace.span("GroupQueryAttention", "op");
            // Two independent sub-decisions on the same node: the staged prep
            // choice must survive the terminal attention-backend choice rather
            // than being overwritten by the shared key.
            record_kernel_variant_stage!("prep", "gqa_prep_fused", "Sq==1, even head_dim");
            record_kernel_variant!("attention_gqa_decode_fp16_splitk", "split-K decode");
        }
        let events = events.events();
        assert_eq!(events.len(), 1);
        let args = events[0].args.as_ref().unwrap();
        assert_eq!(args[ARG_KERNEL_VARIANT], "attention_gqa_decode_fp16_splitk");
        assert_eq!(args["kernel_variant.prep"], "gqa_prep_fused");
        assert!(
            args["kernel_variant_reason.prep"]
                .as_str()
                .unwrap()
                .contains("even head_dim")
        );
    }

    struct DecliningCaptureKernel;

    impl Kernel for DecliningCaptureKernel {
        fn execute(&self, _inputs: &[TensorView], _outputs: &mut [TensorMut]) -> Result<()> {
            Ok(())
        }

        fn capture_support(&self) -> CaptureSupport {
            CaptureSupport::unsupported("per-call workspace allocation is not capturable")
        }
    }

    #[test]
    fn cuda_graph_compatible_shim_matches_capture_support() {
        let supported = LegacyKernel {
            called: Arc::new(AtomicBool::new(false)),
        };
        assert_eq!(
            supported.cuda_graph_compatible(),
            supported.capture_support().is_supported()
        );

        let unsupported = DecliningCaptureKernel;
        assert_eq!(
            unsupported.cuda_graph_compatible(),
            unsupported.capture_support().is_supported()
        );
        assert_eq!(
            unsupported.capture_support().reason(),
            Some("per-call workspace allocation is not capturable")
        );
    }

    struct LegacyKernel {
        called: Arc<AtomicBool>,
    }

    impl Kernel for LegacyKernel {
        fn execute(&self, inputs: &[TensorView], _outputs: &mut [TensorMut]) -> Result<()> {
            assert_eq!(inputs.len(), 1);
            assert_eq!(inputs[0].shape, &[4]);
            self.called.store(true, Ordering::Relaxed);
            Ok(())
        }
    }

    #[test]
    fn legacy_kernel_adapter_receives_the_resident_tensor_path() {
        let called = Arc::new(AtomicBool::new(false));
        let kernel = LegacyKernel {
            called: Arc::clone(&called),
        };
        let bytes = [1u8, 2, 3, 4];
        let shape = [4usize];
        let strides = [1i64];
        let inputs = [KernelInput::Tensor(TensorView::new(
            DevicePtr(bytes.as_ptr().cast()),
            onnx_runtime_ir::DataType::Uint8,
            &shape,
            &strides,
            DeviceId::cpu(),
        ))];

        kernel.execute_with_inputs(&inputs, &mut []).unwrap();
        assert!(called.load(Ordering::Relaxed));
    }
}

#[cfg(test)]
mod trace_standard_tests {
    use super::*;
    use onnx_runtime_tracer::{TraceContext, TraceVerbosity, set_global_context};

    /// Worker spans are the expensive tier and must stay off unless asked for.
    #[test]
    fn worker_span_is_gated_on_full_verbosity() {
        set_global_context(None);
        assert!(
            kernel_worker_span("probe").is_none(),
            "a worker span opened with no ambient context installed"
        );

        let (context, _collector) = TraceContext::in_memory();
        set_global_context(Some(context.with_verbosity(TraceVerbosity::Ops)));
        assert!(
            kernel_worker_span("probe").is_none(),
            "a worker span opened at Ops verbosity, which would put the \
             per-worker cost on every ordinary traced run"
        );

        let (context, collector) = TraceContext::in_memory();
        set_global_context(Some(context.with_verbosity(TraceVerbosity::Full)));
        {
            let span = kernel_worker_span("probe");
            assert!(span.is_some(), "no worker span at Full verbosity");
        }
        let events = collector.events();
        assert_eq!(events.len(), 1, "expected exactly one recorded worker span");
        assert_eq!(events[0].cat, CAT_KERNEL_WORKER);
        assert_eq!(events[0].name, "probe");

        set_global_context(None);
    }
}

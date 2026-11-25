//! The [`Kernel`] trait and kernel-match / cost types (§4.2).

use std::any::Any;
use std::borrow::Cow;
use std::sync::Arc;

use onnx_runtime_ir::{DataType, TensorLayout};
use onnx_runtime_memory_api::MemoryRole;

use crate::ExecutionProvider;
use crate::error::Result;
use crate::tensor::{DevicePtrMut, TensorMut, TensorView};
use crate::weight::WeightHandle;

/// Graph-derived evolution of a node's input/output shapes. Concrete extents
/// alone never establish growth. Unknown is the default for callers without IR facts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ShapeEvolution {
    /// No growing or unproven dependency under the current binding configuration.
    /// A different request/profile may still use different concrete extents.
    Stable,
    /// At least one edge depends on a positively identified growing shape symbol.
    Growing,
    /// Growth or stability could not be established from graph metadata.
    #[default]
    Unknown,
}

/// Cold kernel configuration. Capture policy and positive growth evidence are
/// independent: a conservative capture classifier may reject an unknown shape.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct KernelShapeFacts {
    pub evolution: ShapeEvolution,
    pub capture_eligible: bool,
}

/// One immutable graph initializer supplied during kernel preparation.
///
/// `bytes` borrows the session's inline storage or read-only external mmap; no
/// copy is made to construct this descriptor.
#[derive(Clone, Copy, Debug)]
pub struct KernelConstantInput<'a> {
    pub dtype: DataType,
    pub shape: &'a [usize],
    pub bytes: &'a [u8],
}

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
        /// Shape-independent scratch promise of this selected implementation.
        workspace_planning: WorkspacePlanning,
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

/// Shape-independent promise carried by the selected kernel registration.
/// Exact capacity remains solely a function of `Kernel::workspace_requirement`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkspacePlanning {
    /// Every legal specialization uses no executor-owned scratch.
    None,
    /// A specialization may need scratch. Metadata resolves a requirement or
    /// explicitly defers it until real bindings/values exist before admission.
    Metadata,
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

/// Validate every returned payload before allocating or publishing any outputs.
/// Both the native executor and ORT plugin bridge use this contract.
pub fn validate_kernel_sized_outputs(
    outputs: &[Option<KernelSizedOutput>],
    requested_outputs: &[bool],
    declared_dtypes: &[DataType],
    context: &str,
) -> std::result::Result<(), String> {
    if declared_dtypes.len() != requested_outputs.len() {
        return Err(format!(
            "{context}: output dtype metadata has {} slots, expected {}",
            declared_dtypes.len(),
            requested_outputs.len()
        ));
    }
    if outputs.len() != requested_outputs.len() {
        return Err(format!(
            "{context}: kernel-sized output count mismatch: kernel returned {}, node has {} slots",
            outputs.len(),
            requested_outputs.len()
        ));
    }

    for (slot, ((output, &requested), &declared_dtype)) in outputs
        .iter()
        .zip(requested_outputs)
        .zip(declared_dtypes)
        .enumerate()
    {
        match (requested, output) {
            (false, None) => continue,
            (false, Some(_)) => {
                return Err(format!(
                    "{context}: absent output slot {slot} unexpectedly returned bytes"
                ));
            }
            (true, None) => {
                return Err(format!(
                    "{context}: present output slot {slot} returned no bytes"
                ));
            }
            (true, Some(output)) => {
                if declared_dtype == DataType::Undefined {
                    return Err(format!(
                        "{context}: present output slot {slot} has no declared dtype"
                    ));
                }
                if output.dtype != declared_dtype {
                    return Err(format!(
                        "{context}: kernel-sized output slot {slot} returned dtype {:?}, but the \
                         graph declares {:?}",
                        output.dtype, declared_dtype
                    ));
                }
                let element_size = output.dtype.byte_size();
                if element_size == 0 {
                    return Err(format!(
                        "{context}: kernel-sized output slot {slot} has unsupported zero-sized \
                         dtype {:?}",
                        output.dtype
                    ));
                }
                let numel = output.shape.iter().try_fold(1usize, |product, &extent| {
                    if extent > i64::MAX as usize {
                        return Err(format!(
                            "{context}: kernel-sized output slot {slot} extent {extent} exceeds \
                             ORT's i64 shape range"
                        ));
                    }
                    product.checked_mul(extent).ok_or_else(|| {
                        format!(
                            "{context}: kernel-sized output slot {slot} shape {:?} overflows usize",
                            output.shape
                        )
                    })
                })?;
                let expected_bytes = numel.checked_mul(element_size).ok_or_else(|| {
                    format!(
                        "{context}: kernel-sized output slot {slot} byte length overflows usize \
                         for shape {:?} and dtype {:?}",
                        output.shape, output.dtype
                    )
                })?;
                if output.bytes.len() != expected_bytes {
                    return Err(format!(
                        "{context}: kernel-sized output slot {slot} returned {} bytes, expected \
                         {expected_bytes} for shape {:?} and dtype {:?}",
                        output.bytes.len(),
                        output.shape,
                        output.dtype
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Where an opted-in kernel produces data-dependent output payloads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelSizedOutputPolicy {
    /// The kernel returns owned host bytes from [`Kernel::execute_kernel_sized`].
    HostOwned,
    /// The kernel prepares metadata and device-resident state in governed
    /// workspace, then writes directly into final device outputs.
    DeviceWorkspace,
}

/// Concrete facts for one device-resident kernel-sized output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelSizedOutputMetadata {
    pub shape: Vec<usize>,
    pub dtype: DataType,
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

/// Kernel-visible output facts available for workspace planning. An unresolved
/// shape is distinct from both an omitted output and a known scalar shape.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OutputMetadata<'a> {
    Absent,
    Unresolved {
        dtype: DataType,
    },
    Tensor {
        dtype: DataType,
        shape: std::borrow::Cow<'a, [usize]>,
    },
}

impl<'a> OutputMetadata<'a> {
    pub fn tensor(dtype: DataType, shape: &'a [usize]) -> Self {
        Self::Tensor {
            dtype,
            shape: std::borrow::Cow::Borrowed(shape),
        }
    }

    pub fn from_view(view: &TensorMut<'a>) -> Self {
        if view.is_absent() {
            Self::Absent
        } else {
            Self::tensor(view.dtype, view.shape)
        }
    }

    pub fn into_owned(self) -> OutputMetadata<'static> {
        match self {
            Self::Absent => OutputMetadata::Absent,
            Self::Unresolved { dtype } => OutputMetadata::Unresolved { dtype },
            Self::Tensor { dtype, shape } => OutputMetadata::Tensor {
                dtype,
                shape: std::borrow::Cow::Owned(shape.into_owned()),
            },
        }
    }
}

/// Retention policy for prepared kernel scratch capacity.
///
/// Contents are unspecified between dispatches: serial kernels share the same
/// lifetime peak. Retain reusable artifacts in their own owned storage. Either
/// lifetime may be reclaimed after proven completion when unpinned; installed
/// capture addresses and in-flight work keep their physical backing alive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkspaceLifetime {
    /// Retire at a request boundary unless completed capacity can remain in the
    /// executor's pressure-reclaimable idle lane.
    StepScoped,
    /// Retain fitting capacity across requests, including deferred submissions.
    /// Completed, unpinned capacity may still be reclaimed under pressure.
    SessionPersistent,
}

/// A kernel's owned-scratch bound for one concrete input geometry.
///
/// Kernels choose the lifetime; the corresponding ledger role is derived.
///
/// ```
/// use onnx_runtime_ep_api::{WorkspaceLifetime, WorkspaceRequirement};
/// use onnx_runtime_memory_api::MemoryRole;
/// let step = WorkspaceRequirement { bytes: 4096, alignment: 64, ..WorkspaceRequirement::NONE };
/// assert_eq!(step.role(), MemoryRole::Workspace { step_scoped: true });
/// let persistent = WorkspaceRequirement { lifetime: WorkspaceLifetime::SessionPersistent, ..step };
/// assert_eq!(persistent.role(), MemoryRole::Workspace { step_scoped: false });
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspaceRequirement {
    pub bytes: u64,
    pub alignment: usize,
    pub lifetime: WorkspaceLifetime,
}

impl WorkspaceRequirement {
    pub const NONE: Self = Self {
        bytes: 0,
        alignment: 1,
        lifetime: WorkspaceLifetime::StepScoped,
    };

    /// Ledger classification follows the workspace lifetime. Kernels cannot
    /// declare an unrelated role or disagree with the executor's slot choice.
    pub const fn role(self) -> MemoryRole {
        MemoryRole::Workspace {
            step_scoped: matches!(self.lifetime, WorkspaceLifetime::StepScoped),
        }
    }
}

/// Metadata preparation can resolve a real requirement or defer until storage
/// origins or runtime values exist. Deferred is never an admitted zero request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkspacePreparation {
    Ready(WorkspaceRequirement),
    Deferred,
}

impl WorkspacePreparation {
    /// Require a completed preparation at a metadata-only admission boundary.
    pub fn require_ready(self) -> Result<WorkspaceRequirement> {
        match self {
            Self::Ready(requirement) => Ok(requirement),
            Self::Deferred => Err(crate::EpError::KernelFailed(
                "workspace preparation requires live tensor bindings or runtime values; provide actual input/output views to execution-time planning before admitting or launching the kernel".into(),
            )),
        }
    }
}

/// Execution preparation either has a complete requirement or needs one real
/// prefix binding before it can choose its final layout/algorithm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkspaceExecutionPreparation {
    Ready(WorkspaceRequirement),
    Bind(WorkspaceRequirement),
}

impl WorkspaceExecutionPreparation {
    /// For callers that cannot bind address-dependent plans. Never treat an
    /// unbound prefix as a complete workspace requirement.
    pub fn require_ready(self) -> Result<WorkspaceRequirement> {
        match self {
            Self::Ready(requirement) => Ok(requirement),
            Self::Bind(_) => Err(crate::EpError::KernelFailed(
                "workspace planning requires a real prefix binding; use WorkspaceExecutionPreparation::resolve with the caller's workspace owner before executing".into(),
            )),
        }
    }

    /// Resolve through the caller's existing owner. Ready plans have one
    /// admission and no bound hook; Bind plans have exactly one bound hook and
    /// at most two admissions. Zero demand never calls `prepare`. For nonzero
    /// demand, `prepare` supplies a validated live enclosing view from its owner.
    ///
    /// Preparation borrows geometry only: it may select host algorithms but may
    /// not submit device work, upload metadata, or retain a prefix address as
    /// live storage. Final admission may relocate the owner. A Bind kernel must
    /// prove that final binding preserves its algorithm contract, then rebuild
    /// parameters/metadata from that binding before executing. Kernels needing
    /// stable addresses across admission cannot use this protocol. The caller
    /// holds exclusive kernel ownership through preparation, both admissions,
    /// and execution. A zero-byte prefix still declares its final lifetime and
    /// alignment: Bind(NONE) cannot later change StepScoped to Persistent.
    pub fn resolve<E: From<crate::EpError>>(
        self,
        kernel: &dyn Kernel,
        mut prepare: impl FnMut(WorkspaceRequirement) -> std::result::Result<Option<WorkspaceView>, E>,
    ) -> std::result::Result<(WorkspaceRequirement, Option<WorkspaceView>), E> {
        let mut admit = |requirement: WorkspaceRequirement| {
            if requirement.bytes == 0 {
                Ok(None)
            } else {
                prepare(requirement)
            }
        };
        match self {
            Self::Ready(requirement) => Ok((requirement, admit(requirement)?)),
            Self::Bind(prefix) => {
                if prefix.alignment == 0 || !prefix.alignment.is_power_of_two() {
                    return Err(crate::EpError::KernelFailed(format!(
                        "bound workspace prefix requested invalid alignment {}; declare a nonzero power-of-two alignment before binding",
                        prefix.alignment,
                    )).into());
                }
                let prefix_layout = if prefix.bytes != 0 {
                    let bytes = usize::try_from(prefix.bytes).map_err(|_| crate::EpError::KernelFailed(
                        format!("bound workspace prefix {} bytes exceeds addressable capacity; reduce the workspace geometry", prefix.bytes),
                    ))?;
                    let layout = std::alloc::Layout::from_size_align(bytes, prefix.alignment)
                        .map_err(|_| crate::EpError::KernelFailed(format!("bound workspace prefix {} bytes aligned to {} cannot form an addressable layout; reduce the workspace geometry or alignment", prefix.bytes, prefix.alignment)))?;
                    Some(layout)
                } else {
                    None
                };
                let binding = admit(prefix)?;
                let binding = match prefix_layout {
                    Some(layout) => {
                        let [used] = crate::WorkspaceLayout::new([layout])?.bind(binding)?;
                        Some(used)
                    }
                    None => None,
                };
                let final_requirement =
                    kernel.workspace_requirement_for_bound_execution(binding)?;
                if final_requirement.lifetime != prefix.lifetime
                    || final_requirement.bytes < prefix.bytes
                    || final_requirement.alignment < prefix.alignment
                    || !final_requirement.alignment.is_power_of_two()
                {
                    return Err(crate::EpError::KernelFailed(format!(
                        "bound workspace refinement changed prefix {prefix:?} to incompatible final requirement {final_requirement:?}; preserve lifetime, prefix offsets and alignment, and include all prefix bytes in the final layout",
                    )).into());
                }
                let final_binding = if final_requirement == prefix {
                    binding
                } else {
                    admit(final_requirement)?
                };
                Ok((final_requirement, final_binding))
            }
        }
    }
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

/// Object-safe concrete-type access for provider-specific diagnostics.
pub trait KernelType {
    fn as_any(&self) -> &dyn Any;
}

impl<T: Any> KernelType for T {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// A workspace extent derived from a validated, pinned binding allocation.
/// Bare graph tokens cannot construct one or grant address-coverage permission.
#[derive(Clone, Copy, Debug)]
pub struct WorkspaceCaptureSpan {
    binding: onnx_runtime_memory_api::BindingIdentity,
    start: usize,
    bytes: usize,
}

impl WorkspaceCaptureSpan {
    pub fn binding(self) -> onnx_runtime_memory_api::BindingIdentity {
        self.binding
    }

    pub fn require_context(
        self,
        expected: onnx_runtime_memory_api::ProviderContextIdentity,
    ) -> Result<()> {
        let actual = self.binding.provider_context();
        if actual != expected {
            return Err(crate::EpError::KernelFailed(format!(
                "workspace capture pin belongs to provider context {actual:?}, expected {expected:?}; allocate capture workspace through this provider"
            )));
        }
        Ok(())
    }

    /// Whether one pinned owner covers the entire requested byte range.
    pub fn covers(self, address: usize, bytes: usize) -> bool {
        self.start
            .checked_add(self.bytes)
            .zip(address.checked_add(bytes))
            .is_some_and(|(limit, end)| address >= self.start && end <= limit)
    }
}

/// Immutable owner retained by an installed device graph.
///
/// Kernels and executors return these owners before capture begins so address-
/// bearing resources are pinned before any device node can embed their pointers.
/// Workspace coverage is granted only by a validated binding-issued view.
#[derive(Clone)]
pub struct DeviceGraphResource {
    identity: usize,
    owner: Arc<dyn Any + Send + Sync>,
    workspace_span: Option<WorkspaceCaptureSpan>,
}

impl DeviceGraphResource {
    pub fn new<T>(identity: usize, owner: Arc<T>) -> Self
    where
        T: Any + Send + Sync,
    {
        Self {
            identity,
            owner,
            workspace_span: None,
        }
    }

    /// Pin a complete workspace through its binding-issued owner. Creation is
    /// capture preparation work; it does not reserve or refund memory quota.
    pub fn for_workspace(owner: &onnx_runtime_memory_api::OwningAllocation) -> Result<Self> {
        let pin = owner.view(0, owner.len()).map_err(|error| crate::EpError::KernelFailed(format!(
            "cannot pin workspace for device graph capture: {error}; retain its live binding-issued owner until graph retirement"
        )))?;
        let span = pin.binding().with_view(pin.view(), |validated| WorkspaceCaptureSpan {
            binding: pin.allocation_identity().binding(),
            start: validated.as_ptr().as_ptr() as usize,
            bytes: validated.len(),
        }).map_err(|error| crate::EpError::KernelFailed(format!(
            "cannot validate workspace for device graph capture: {error}; allocate it through the active provider binding"
        )))?;
        if span.start.checked_add(span.bytes).is_none() {
            return Err(crate::EpError::KernelFailed(
                "workspace capture extent wraps the address space; use a valid binding allocation"
                    .into(),
            ));
        }
        let pin = Arc::new(pin);
        Ok(Self {
            identity: Arc::as_ptr(&pin) as usize,
            owner: pin,
            workspace_span: Some(span),
        })
    }

    pub fn workspace_span(&self) -> Option<WorkspaceCaptureSpan> {
        self.workspace_span
    }

    pub fn identity(&self) -> usize {
        self.identity
    }

    pub fn owner(&self) -> &Arc<dyn Any + Send + Sync> {
        &self.owner
    }
}

/// One coarse phase's production BlockQuantizedMoE traffic projection.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BlockQuantizedMoeTraffic {
    /// Bytes uploaded once while admitting immutable whole projection banks.
    pub uploaded_whole_bank_bytes: u64,
    /// Device bytes committed for those whole projection banks.
    pub committed_whole_bank_bytes: u64,
    /// Logical route demand, including repeated expert selections.
    pub logical_route_demand_bytes: u64,
    /// Extent of distinct experts selected in the phase.
    pub unique_selected_expert_bytes: u64,
    /// Measured physical DRAM bytes, when a hardware counter supplies them.
    pub physical_dram_bytes: Option<u64>,
    /// Expert page-ins performed by a residency implementation.
    pub page_ins: u64,
    /// Byte-weighted hit rate, when expert-indexed residency is active.
    pub byte_hit_rate: Option<f64>,
}

/// A borrowed, dense host string tensor. It has no byte or device representation.
#[derive(Clone, Copy)]
pub struct HostStringInput<'a> {
    pub shape: &'a [usize],
    pub values: &'a [String],
}

/// Borrowed dense operands for native host kernels on managed-string nodes.
pub enum HostKernelInput<'a> {
    Strings(HostStringInput<'a>),
    /// One byte per BOOL element, interpreted as false for zero and true otherwise.
    Bool {
        shape: &'a [usize],
        bytes: &'a [u8],
    },
}

/// An owned output whose shape may be determined by the string kernel.
pub struct HostKernelOutput {
    pub shape: Vec<usize>,
    pub data: HostKernelData,
}

/// Managed strings or ordinary little-endian tensor bytes.
pub enum HostKernelData {
    Strings(Vec<String>),
    Raw { dtype: DataType, bytes: Vec<u8> },
}

/// A kernel ready to execute a specific op with specific shapes (§4.2).
pub trait Kernel: Send + KernelType {
    /// Execute a managed-string node using borrowed host operands.
    fn execute_managed_host(
        &self,
        _inputs: &[HostKernelInput<'_>],
    ) -> Result<Vec<HostKernelOutput>> {
        Err(crate::EpError::KernelFailed(
            "kernel does not support native host string tensors; use a string-aware host kernel"
                .into(),
        ))
    }

    /// Tell the kernel which positional inputs are immutable graph constants.
    ///
    /// The session calls this exactly once, immediately after construction.
    /// Kernels may use it to prepack or memoize those inputs. Runtime inputs must
    /// never be marked constant: caching them would return stale results.
    fn set_constant_inputs(&mut self, constant_inputs: &[bool]) {
        let _ = constant_inputs;
    }

    /// Validate and bind immutable graph constants before the first launch.
    ///
    /// The session calls this for a new kernel without compatible adopted
    /// preparation, after [`Kernel::set_constant_inputs`]. Stock kernels need none.
    fn prepare_constant_inputs(
        &mut self,
        constants: &[Option<KernelConstantInput<'_>>],
        provider: &dyn ExecutionProvider,
    ) -> Result<()> {
        let _ = (constants, provider);
        Ok(())
    }

    /// Share immutable preparation across shape-specialized variants of the
    /// same session node. The executor's kernel cache is node-scoped, so this
    /// state never crosses sessions or graph nodes.
    fn shareable_constant_state(&self) -> Option<Arc<dyn std::any::Any + Send + Sync>> {
        None
    }

    /// Adopt state returned by [`Kernel::shareable_constant_state`] or
    /// [`ExecutionProvider::prepare_static_constants`]. Unrecognized state must
    /// return false so ordinary preparation can run.
    ///
    /// Returns `true` when the state was recognized and installed.
    fn adopt_shareable_constant_state(
        &mut self,
        state: Arc<dyn std::any::Any + Send + Sync>,
    ) -> Result<bool> {
        let _ = state;
        Ok(false)
    }

    /// Replace a positional constant input with a kernel-owned immutable view.
    ///
    /// The session applies overrides after kernel lookup and before execution;
    /// direct callers that bypass this binding seam are rejected by kernels
    /// whose launch contract requires a sealed view.
    fn constant_input_override(&self, input_idx: usize) -> Option<TensorView<'_>> {
        let _ = input_idx;
        None
    }

    /// Immutable owners whose addresses the next device-graph capture may
    /// embed. Called before capture starts; warmed execution never registers
    /// resources with the graph lifecycle.
    fn device_graph_resources(&self) -> Vec<DeviceGraphResource> {
        Vec::new()
    }

    /// Arm production BlockQuantizedMoE route traffic for this kernel.
    ///
    /// Returns `true` when the kernel participates. Session APIs call this only
    /// under exclusive request ownership and before capture.
    fn arm_block_quantized_moe_traffic(&mut self, request_id: u32) -> Result<bool> {
        let _ = request_id;
        Ok(false)
    }

    /// Reset the current BlockQuantizedMoE accumulation phase in place.
    fn reset_block_quantized_moe_traffic(&mut self) -> Result<()> {
        Ok(())
    }

    /// Snapshot the current BlockQuantizedMoE phase.
    fn snapshot_block_quantized_moe_traffic(&self) -> Result<Option<BlockQuantizedMoeTraffic>> {
        Ok(None)
    }

    /// Disarm BlockQuantizedMoE traffic after dependent graphs are retired.
    fn disarm_block_quantized_moe_traffic(&mut self) -> Result<()> {
        Ok(())
    }

    /// Configure graph-derived shape evolution and capture policy before use.
    /// Session specializes/cache-keys kernels by these facts; changed capacity
    /// pinning must not reuse an instance configured with different facts.
    /// Eager/plugin callers without graph evidence retain the Unknown default.
    /// Concrete runtime extents must not be used to infer growth.
    fn configure_shape_facts(&mut self, facts: KernelShapeFacts) {
        let _ = facts;
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

    /// Select the payload policy for this opted-in kernel.
    ///
    /// Host-owned is the compatibility default used by CPU `Unique`. Device
    /// kernels override this explicitly; the adapter never infers a transfer
    /// policy from pointer placement.
    fn kernel_sized_output_policy(&self) -> KernelSizedOutputPolicy {
        KernelSizedOutputPolicy::HostOwned
    }

    /// Execute once into owned host buffers and report concrete output facts.
    ///
    /// `requested_outputs` preserves node output-slot positions. A `false`
    /// slot is an omitted optional output and must be returned as `None`; a
    /// `true` slot must be returned as `Some`. The adapter validates every
    /// dtype, checked shape byte length, and buffer length before allocating
    /// final outputs, then performs one materialization copy per present slot.
    ///
    /// `workspace` is borrowed executor/adapter scratch prepared from the same
    /// metadata contract as ordinary execution. Retain no borrow after returning:
    /// every returned output must independently own its bytes. An adapter may
    /// decline an unsupported lifetime by passing `None`; a kernel requiring
    /// scratch must reject that case with an actionable error.
    ///
    /// This output policy is host-only. Implementations must reject
    /// non-host-accessible inputs before reading them.
    fn execute_kernel_sized(
        &self,
        inputs: &[TensorView],
        requested_outputs: &[bool],

        workspace: Option<crate::WorkspaceView>,
    ) -> Result<Vec<Option<KernelSizedOutput>>> {
        let _ = (inputs, requested_outputs, workspace);
        Err(crate::EpError::KernelFailed(
            "kernel does not implement kernel-sized outputs".into(),
        ))
    }

    /// Run the metadata phase of a device-workspace kernel-sized operation.
    ///
    /// The kernel may launch device work and copy only compact metadata to the
    /// host. Any state needed by [`Kernel::materialize_kernel_sized_device`]
    /// must live in the governed `workspace`, whose lifetime covers both calls.
    fn prepare_kernel_sized_device(
        &self,
        inputs: &[TensorView],
        requested_outputs: &[bool],
        workspace: Option<WorkspaceView>,
    ) -> Result<Vec<Option<KernelSizedOutputMetadata>>> {
        let _ = (inputs, requested_outputs, workspace);
        Err(crate::EpError::KernelFailed(
            "kernel does not implement device-workspace kernel-sized outputs".into(),
        ))
    }

    /// Materialize a prepared device-workspace operation into final outputs.
    ///
    /// `outputs` preserves positional optional slots; omitted slots are marked
    /// absent and must not be written. This phase must reuse the state produced
    /// by [`Kernel::prepare_kernel_sized_device`] rather than rerunning the full
    /// algorithm.
    fn materialize_kernel_sized_device(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        let _ = (inputs, outputs, workspace);
        Err(crate::EpError::KernelFailed(
            "kernel does not implement device-workspace materialization".into(),
        ))
    }

    /// Report a safe owned-scratch bound for the input and output geometry.
    /// Output slots describe the views this kernel receives, after any graph
    /// lowering, and preserve positions marked absent by that caller. Unknown geometry must
    /// remain Unresolved. Return Deferred when values or storage origins are
    /// needed; Ready(NONE) means a proven zero requirement, never unknown.
    /// Runtime layout facts may refine this bound before dispatch. For a fixed
    /// compiled kernel, provider workspace context and input/output metadata this
    /// is a stable upper bound: executors may
    /// memoize it. It must cover both cold and warm cache residency; values or
    /// layouts that refine the bound belong in the execution-time hook below.
    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
        _outputs: &[crate::OutputMetadata<'_>],
    ) -> Result<WorkspacePreparation> {
        let _ = inputs;
        Ok(WorkspacePreparation::Ready(WorkspaceRequirement::NONE))
    }

    /// Refine the workspace requirement with real input and output views.
    /// Output storage is uninitialized: inspect geometry and pointer alignment,
    /// never read output values. The caller keeps exclusive kernel execution
    /// ownership through this refinement, admission and the following launch.
    /// `cached` contains only the same kernel's metadata-only preparation in the
    /// same provider workspace context; address/value-dependent refinements must not enter a geometry cache.
    /// None asks the default to query metadata once; Some avoids that query.
    ///
    /// Prepare-only reservation walks have only [`TensorMetadata`], so
    /// [`Kernel::workspace_requirement`] remains the planning hook. Execution
    /// dispatch, though, may already hold device views whose *values* determine
    /// the exact scratch size (for example a reduction whose axes come from an
    /// input tensor). Metadata presence is logical: a lazy initializer can be
    /// present in metadata while its TensorView is absent. Check the view before
    /// reading bytes; metadata alone never grants readable storage. The default
    /// preserves a ready metadata-only contract and refuses unresolved Deferred.
    fn workspace_requirement_for_execution(
        &self,
        inputs: &[TensorView],
        outputs: &[TensorMut],
        metadata: &[TensorMetadata<'_>],
        output_metadata: &[crate::OutputMetadata<'_>],
        cached: Option<WorkspacePreparation>,
    ) -> Result<WorkspaceExecutionPreparation> {
        let _ = (inputs, outputs);
        match cached {
            Some(preparation) => preparation,
            None => self.workspace_requirement(metadata, output_metadata)?,
        }
        .require_ready()
        .map(WorkspaceExecutionPreparation::Ready)
    }

    /// Refine one staged Bind plan using an admitted prefix. The final owner
    /// may move; preserve the prefix's layout/alignment contract and return a
    /// complete requirement. This hook must not submit device work or upload
    /// address-bearing metadata. Warm plans should return Ready directly from
    /// execution preparation. Exclusive kernel ownership spans all phases.
    fn workspace_requirement_for_bound_execution(
        &self,
        _workspace: Option<WorkspaceView>,
    ) -> Result<WorkspaceRequirement> {
        Err(crate::EpError::KernelFailed(
            "kernel requested bound workspace preparation without implementing its binding hook; implement workspace_requirement_for_bound_execution or return a complete Ready requirement".into(),
        ))
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
    /// Paging-aware kernels receive the same executor-prepared workspace as
    /// resident kernels. Scratch ownership, admission and completion remain with
    /// the executor; this hook only borrows the regions during execution.
    /// The default accepts resident inputs and forwards both views and workspace
    /// to [`Kernel::execute_with_workspace`]. A kernel consuming lazy handles
    /// overrides this hook without needing a separate scratch protocol.
    fn execute_with_inputs(
        &self,
        inputs: &[KernelInput<'_>],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        let views = inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                input.tensor().copied().ok_or_else(|| {
                    crate::EpError::KernelFailed(format!(
                        "kernel input {index} is an unresolved weight handle; implement execute_with_inputs to consume it with the prepared workspace"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        self.execute_with_workspace(&views, outputs, workspace)
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
    fn may_produce_views(&self, _input_dtypes: &[DataType]) -> bool {
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
            workspace_planning: crate::WorkspacePlanning::Metadata,
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
            workspace_planning: crate::WorkspacePlanning::Metadata,
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

    struct WorkspaceAdapterKernel;

    impl Kernel for WorkspaceAdapterKernel {
        fn execute(&self, _inputs: &[TensorView], _outputs: &mut [TensorMut]) -> Result<()> {
            panic!("the general input adapter must retain the workspace")
        }

        fn execute_with_workspace(
            &self,
            inputs: &[TensorView],
            _outputs: &mut [TensorMut],
            workspace: Option<WorkspaceView>,
        ) -> Result<()> {
            assert_eq!(inputs.len(), 1);
            assert_eq!(inputs[0].shape, &[4]);
            let workspace = workspace.ok_or_else(|| {
                crate::EpError::KernelFailed(
                    "workspace adapter fixture requires a borrowed region".into(),
                )
            })?;
            assert_eq!(workspace.bytes(), 4);
            // SAFETY: this test lends an exclusive four-byte Host array.
            unsafe { workspace.ptr().0.cast::<u8>().write(37) };
            Ok(())
        }
    }

    #[test]
    fn general_input_adapter_forwards_workspace_and_rejects_unresolved_weights() {
        let bytes = [1u8, 2, 3, 4];
        let shape = [4usize];
        let strides = [1i64];
        let inputs = [KernelInput::Tensor(TensorView::new(
            DevicePtr(bytes.as_ptr().cast()),
            DataType::Uint8,
            &shape,
            &strides,
            DeviceId::cpu(),
        ))];
        let mut scratch = [0u8; 4];
        let workspace = WorkspaceView::new(DevicePtrMut(scratch.as_mut_ptr().cast()), 4);
        WorkspaceAdapterKernel
            .execute_with_inputs(&inputs, &mut [], Some(workspace))
            .unwrap();
        assert_eq!(scratch[0], 37);
        let error = WorkspaceAdapterKernel
            .execute_with_inputs(&inputs, &mut [], None)
            .unwrap_err();
        assert!(error.to_string().contains("requires a borrowed region"));
        scratch[0] = 0;
        let handle = WeightHandle::Resident(
            crate::ResidentWeight::new(DataType::Uint8, vec![4], bytes.to_vec()).unwrap(),
        );
        let error = WorkspaceAdapterKernel
            .execute_with_inputs(&[KernelInput::Weight(&handle)], &mut [], Some(workspace))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("input 0") && error.contains("implement execute_with_inputs"),
            "{error}"
        );
        assert_eq!(scratch[0], 0, "rejection must happen before compute");
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

        kernel.execute_with_inputs(&inputs, &mut [], None).unwrap();
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

#[cfg(test)]
mod output_workspace_metadata_tests {
    use super::*;
    use crate::EpError;

    #[test]
    fn workspace_preparation_keeps_zero_distinct_from_deferred_and_uses_cached_facts() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct Counted(AtomicUsize);
        impl Kernel for Counted {
            fn execute(&self, _: &[TensorView], _: &mut [TensorMut]) -> Result<()> {
                Ok(())
            }
            fn workspace_requirement(
                &self,
                _: &[TensorMetadata<'_>],
                _: &[OutputMetadata<'_>],
            ) -> Result<WorkspacePreparation> {
                self.0.fetch_add(1, Ordering::Relaxed);
                Ok(WorkspacePreparation::Ready(WorkspaceRequirement::NONE))
            }
        }
        let kernel = Counted(AtomicUsize::new(0));
        let zero = WorkspacePreparation::Ready(WorkspaceRequirement::NONE);
        assert_ne!(zero, WorkspacePreparation::Deferred);
        assert_eq!(
            kernel
                .workspace_requirement_for_execution(&[], &[], &[], &[], Some(zero))
                .unwrap()
                .require_ready()
                .unwrap(),
            WorkspaceRequirement::NONE
        );
        assert_eq!(kernel.0.load(Ordering::Relaxed), 0);
        assert_eq!(
            kernel
                .workspace_requirement_for_execution(&[], &[], &[], &[], None)
                .unwrap()
                .require_ready()
                .unwrap(),
            WorkspaceRequirement::NONE
        );
        assert_eq!(kernel.0.load(Ordering::Relaxed), 1);
        let error = kernel
            .workspace_requirement_for_execution(
                &[],
                &[],
                &[],
                &[],
                Some(WorkspacePreparation::Deferred),
            )
            .unwrap_err()
            .to_string();
        assert!(error.contains("live tensor bindings"), "{error}");
        assert_eq!(
            kernel.0.load(Ordering::Relaxed),
            1,
            "cached deferred must not requery or pretend to be zero"
        );
    }

    #[test]
    fn workspace_execution_refinement_receives_real_output_origin_and_metadata() {
        struct Bound;
        impl Kernel for Bound {
            fn execute(&self, _: &[TensorView], _: &mut [TensorMut]) -> Result<()> {
                Ok(())
            }
            fn workspace_requirement(
                &self,
                _: &[TensorMetadata<'_>],
                _: &[OutputMetadata<'_>],
            ) -> Result<WorkspacePreparation> {
                Ok(WorkspacePreparation::Deferred)
            }
            fn workspace_requirement_for_execution(
                &self,
                _: &[TensorView],
                outputs: &[TensorMut],
                _: &[TensorMetadata<'_>],
                metadata: &[OutputMetadata<'_>],
                cached: Option<WorkspacePreparation>,
            ) -> Result<WorkspaceExecutionPreparation> {
                assert_eq!(cached, Some(WorkspacePreparation::Deferred));
                let [output] = outputs else {
                    return Err(crate::EpError::KernelFailed("bound planner requires one actual output allocation; do not substitute unresolved geometry".into()));
                };
                assert_eq!(metadata, [OutputMetadata::from_view(output)]);
                let origin = output.data.0 as usize + output.byte_offset;
                Ok(WorkspaceExecutionPreparation::Ready(WorkspaceRequirement {
                    bytes: origin.trailing_zeros() as u64,
                    ..WorkspaceRequirement::NONE
                }))
            }
        }
        let mut backing = [0u64; 4];
        let mut output = TensorMut::new(
            crate::DevicePtrMut(backing.as_mut_ptr().cast()),
            DataType::Float32,
            &[1],
            &[1],
            crate::DeviceId::cpu(),
        );
        output.byte_offset = 4;
        let metadata = [OutputMetadata::from_view(&output)];
        let expected = (output.data.0 as usize + 4).trailing_zeros() as u64;
        let kernel = Bound;
        let preparation = kernel.workspace_requirement(&[], &metadata).unwrap();
        assert_eq!(preparation, WorkspacePreparation::Deferred);
        assert!(preparation.require_ready().is_err());
        let requirement = kernel
            .workspace_requirement_for_execution(&[], &[output], &[], &metadata, Some(preparation))
            .unwrap()
            .require_ready()
            .unwrap();
        assert_eq!(requirement.bytes, expected);
        assert!(
            kernel
                .workspace_requirement_for_execution(&[], &[], &[], &metadata, Some(preparation))
                .unwrap_err()
                .to_string()
                .contains("actual output allocation")
        );
    }

    #[test]
    fn output_workspace_metadata_distinguishes_scalar_unknown_and_absent() {
        let scalar = OutputMetadata::tensor(DataType::Float32, &[]);
        assert_ne!(scalar, OutputMetadata::Absent);
        assert_ne!(
            scalar,
            OutputMetadata::Unresolved {
                dtype: DataType::Float32
            }
        );
        let owned = {
            let shape = vec![2, 7];
            OutputMetadata::tensor(DataType::Float16, &shape).into_owned()
        };
        assert_eq!(owned, OutputMetadata::tensor(DataType::Float16, &[2, 7]));
    }

    #[test]
    fn execution_refinement_forwards_output_facts_to_canonical_planner() {
        struct OutputBound;
        impl Kernel for OutputBound {
            fn execute(&self, _: &[TensorView], _: &mut [TensorMut]) -> Result<()> {
                Ok(())
            }
            fn workspace_requirement(
                &self,
                _: &[TensorMetadata<'_>],
                outputs: &[OutputMetadata<'_>],
            ) -> Result<WorkspacePreparation> {
                let [
                    OutputMetadata::Absent,
                    OutputMetadata::Tensor {
                        dtype: DataType::Float32,
                        shape,
                    },
                ] = outputs
                else {
                    return Err(EpError::KernelFailed(
                        "output #1 must have resolved Float32 geometry before preparation".into(),
                    ));
                };
                Ok(WorkspacePreparation::Ready(WorkspaceRequirement {
                    bytes: shape.iter().product::<usize>() as u64 * 4,
                    ..WorkspaceRequirement::NONE
                }))
            }
        }
        let outputs = [
            OutputMetadata::Absent,
            OutputMetadata::tensor(DataType::Float32, &[3, 5]),
        ];
        assert_eq!(
            OutputBound
                .workspace_requirement_for_execution(&[], &[], &[], &outputs, None)
                .unwrap()
                .require_ready()
                .unwrap()
                .bytes,
            60
        );
        let unknown = [
            OutputMetadata::Absent,
            OutputMetadata::Unresolved {
                dtype: DataType::Float32,
            },
        ];
        assert!(
            OutputBound
                .workspace_requirement_for_execution(&[], &[], &[], &unknown, None)
                .is_err()
        );
    }
}

#[cfg(test)]
mod bound_workspace_tests {
    use super::*;
    use std::sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    };

    struct BoundProbe {
        final_requirement: WorkspaceRequirement,
        bindings: Mutex<Vec<Option<(usize, usize)>>>,
        fail: bool,
    }
    impl Kernel for BoundProbe {
        fn execute(&self, _: &[TensorView], _: &mut [TensorMut]) -> Result<()> {
            panic!("preparation must not execute a kernel")
        }
        fn workspace_requirement_for_bound_execution(
            &self,
            binding: Option<WorkspaceView>,
        ) -> Result<WorkspaceRequirement> {
            self.bindings
                .lock()
                .unwrap()
                .push(binding.map(|view| (view.ptr().0 as usize, view.bytes())));
            if self.fail {
                return Err(crate::EpError::KernelFailed(
                    "injected bound planner failure; retry preparation".into(),
                ));
            }
            Ok(self.final_requirement)
        }
    }
    fn requirement(bytes: u64) -> WorkspaceRequirement {
        WorkspaceRequirement {
            bytes,
            alignment: 16,
            lifetime: WorkspaceLifetime::SessionPersistent,
        }
    }
    fn probe(final_requirement: WorkspaceRequirement) -> BoundProbe {
        BoundProbe {
            final_requirement,
            bindings: Default::default(),
            fail: false,
        }
    }

    #[test]
    fn bound_workspace_protocol_has_one_hook_and_at_most_two_admissions() {
        for (prefix_bytes, final_bytes, expected_admissions) in
            [(0, 0, 0), (0, 64, 1), (16, 16, 1), (16, 64, 2)]
        {
            let kernel = probe(requirement(final_bytes));
            let mut prefix = [0u128; 4];
            let mut final_backing = [0u128; 4];
            let mut calls = 0;
            let (resolved, view) = WorkspaceExecutionPreparation::Bind(requirement(prefix_bytes))
                .resolve::<crate::EpError>(&kernel, |_req| {
                    calls += 1;
                    let pointer = if calls == 1 && prefix_bytes != 0 {
                        prefix.as_mut_ptr()
                    } else {
                        final_backing.as_mut_ptr()
                    };
                    Ok(Some(WorkspaceView::new(
                        crate::DevicePtrMut(pointer.cast()),
                        64,
                    )))
                })
                .unwrap();
            assert_eq!(resolved, requirement(final_bytes));
            assert_eq!(calls, expected_admissions);
            let bindings = kernel.bindings.lock().unwrap();
            assert_eq!(bindings.len(), 1);
            if prefix_bytes == 0 {
                assert_eq!(bindings[0], None);
            } else {
                assert_eq!(
                    bindings[0],
                    Some((prefix.as_ptr() as usize, prefix_bytes as usize))
                );
            }
            if final_bytes == 0 {
                assert!(view.is_none());
            } else {
                assert_eq!(view.unwrap().bytes(), final_bytes as usize);
                if calls == 2 {
                    assert_eq!(
                        view.unwrap().ptr().0 as usize,
                        final_backing.as_ptr() as usize
                    );
                }
            }
        }
    }

    #[test]
    fn ready_workspace_never_calls_bound_hook_and_zero_never_admits() {
        for bytes in [0, 64] {
            let kernel = probe(requirement(999));
            let calls = AtomicUsize::new(0);
            let mut backing = [0u128; 4];
            WorkspaceExecutionPreparation::Ready(requirement(bytes))
                .resolve::<crate::EpError>(&kernel, |_| {
                    calls.fetch_add(1, Ordering::Relaxed);
                    Ok(Some(WorkspaceView::new(
                        crate::DevicePtrMut(backing.as_mut_ptr().cast()),
                        64,
                    )))
                })
                .unwrap();
            assert_eq!(calls.load(Ordering::Relaxed), usize::from(bytes != 0));
            assert!(kernel.bindings.lock().unwrap().is_empty());
        }
        let error = WorkspaceExecutionPreparation::Bind(requirement(16))
            .require_ready()
            .unwrap_err()
            .to_string();
        assert!(error.contains("real prefix binding"), "{error}");
    }

    #[test]
    fn bound_workspace_rejects_incompatible_final_contract_before_second_admission() {
        for final_requirement in [
            requirement(8),
            WorkspaceRequirement {
                alignment: 8,
                ..requirement(64)
            },
            WorkspaceRequirement {
                alignment: 24,
                ..requirement(64)
            },
            WorkspaceRequirement {
                lifetime: WorkspaceLifetime::StepScoped,
                ..requirement(64)
            },
        ] {
            let kernel = probe(final_requirement);
            let mut backing = [0u128; 4];
            let mut calls = 0;
            let error = WorkspaceExecutionPreparation::Bind(requirement(16))
                .resolve::<crate::EpError>(&kernel, |_| {
                    calls += 1;
                    Ok(Some(WorkspaceView::new(
                        crate::DevicePtrMut(backing.as_mut_ptr().cast()),
                        64,
                    )))
                })
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("incompatible final requirement")
                    && error.contains("preserve lifetime"),
                "{error}"
            );
            assert_eq!(calls, 1);
            assert_eq!(kernel.bindings.lock().unwrap().len(), 1);
        }
    }

    #[test]
    fn bound_workspace_rejects_invalid_prefix_layout_before_admission() {
        for prefix in [
            WorkspaceRequirement {
                alignment: 0,
                ..requirement(16)
            },
            WorkspaceRequirement {
                alignment: 24,
                ..requirement(16)
            },
            requirement(u64::MAX),
        ] {
            let kernel = probe(requirement(64));
            let error = WorkspaceExecutionPreparation::Bind(prefix)
                .resolve::<crate::EpError>(&kernel, |_| {
                    panic!("invalid prefix must not call admission")
                })
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("bound workspace prefix")
                    && (error.contains("alignment") || error.contains("addressable")),
                "{error}"
            );
            assert!(kernel.bindings.lock().unwrap().is_empty());
        }
    }

    #[test]
    fn bound_workspace_refuses_bad_prefix_before_planning_and_propagates_failures() {
        let kernel = probe(requirement(64));
        let error = WorkspaceExecutionPreparation::Bind(requirement(16))
            .resolve::<crate::EpError>(&kernel, |_| Ok(None))
            .unwrap_err()
            .to_string();
        assert!(error.contains("workspace needs"), "{error}");
        assert!(kernel.bindings.lock().unwrap().is_empty());
        let error = WorkspaceExecutionPreparation::Bind(requirement(16))
            .resolve::<crate::EpError>(&kernel, |_| {
                Err(crate::EpError::KernelFailed(
                    "injected admission failure".into(),
                ))
            })
            .unwrap_err()
            .to_string();
        assert!(error.contains("injected admission failure"), "{error}");
        assert!(kernel.bindings.lock().unwrap().is_empty());
        let kernel = BoundProbe {
            fail: true,
            ..probe(requirement(64))
        };
        let mut backing = [0u128; 4];
        let mut calls = 0;
        let error = WorkspaceExecutionPreparation::Bind(requirement(16))
            .resolve::<crate::EpError>(&kernel, |_| {
                calls += 1;
                Ok(Some(WorkspaceView::new(
                    crate::DevicePtrMut(backing.as_mut_ptr().cast()),
                    64,
                )))
            })
            .unwrap_err()
            .to_string();
        assert!(error.contains("injected bound planner failure"), "{error}");
        assert_eq!(calls, 1);
    }
}

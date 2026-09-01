//! The owned, device-aware [`Tensor`] handed to and returned from
//! [`InferenceSession::run`](crate::InferenceSession::run), plus the isolated
//! host-buffer accessors it and the executor share.
//!
//! ## Placement decision (design open question, §20 / plan §2.D)
//!
//! The plan flagged *where* the real tensor type should live as an open
//! question: keep it in `onnx-runtime-session`, or hoist it into a shared
//! `onnx-runtime-tensor` crate. For Phase 1 (CPU only) it lives **here**: the
//! type is a thin owner over an [`onnx_runtime_ep_api::DeviceBuffer`] plus the
//! IR vocabulary ([`DataType`], [`TensorLayout`], shape) — nothing CPU-specific
//! leaks into its shape. When `ep-cuda` lands and non-host tensors need DLPack
//! import/export and cross-device copies, this is a mechanical move into a
//! shared crate that both the session and the C-API can depend on; nothing in
//! its public surface here presumes a host device beyond the accessors, which
//! already gate on [`DeviceId::is_host_accessible`].
//!
//! ## The single `unsafe` seam
//!
//! A [`DeviceBuffer`] hands out only raw base pointers; reading or writing the
//! bytes is `unsafe` and sound only on host-accessible devices. Every direct
//! host read in this crate funnels through [`host_bytes`], while writes use the
//! owning execution provider's host-copy API, so the rest of the crate — the
//! executor and the public API — is safe Rust over the EP contract.

use std::sync::{Arc, OnceLock};

use onnx_runtime_ep_api::{
    DeviceBuffer, DeviceGraphToken, DeviceValidationRegistration, DeviceValidationToken,
    ExecutionProvider,
};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ir::{DataType, DeviceId, TensorLayout, checked_expected_bytes, read_vec_le};

use crate::error::{Result, SessionError};
use crate::sequence::{SequenceError, SequenceResult, clone_shape};

/// A process-wide, already-initialized CPU execution provider used to back
/// user-constructed [`Tensor`]s (host `malloc`/`free` is global, so any
/// `CpuExecutionProvider` can free any other's CPU allocation).
pub(crate) fn shared_cpu_ep() -> Arc<CpuExecutionProvider> {
    static EP: OnceLock<Arc<CpuExecutionProvider>> = OnceLock::new();
    EP.get_or_init(|| {
        let mut ep = CpuExecutionProvider::new();
        // Pure-Rust CPU EP: `initialize` only flips a flag and never fails.
        let _ = ep.initialize(&Default::default());
        Arc::new(ep)
    })
    .clone()
}

/// The shared CPU execution provider as an [`ExecutionProvider`] trait object.
///
/// Exposed so callers building a [`Tensor`] from *borrowed* host memory (e.g.
/// the Python binding's zero-copy DLPack import) can supply the allocator
/// [`Tensor::from_borrowed_parts_with_guard`] requires. Because a borrowed
/// buffer is never actually freed by the EP, any CPU provider suffices.
pub fn cpu_allocator() -> Arc<dyn ExecutionProvider> {
    shared_cpu_ep()
}

/// Ref-counted owner for one device allocation shared by immutable runtime
/// tensor values such as ONNX Sequence elements.
///
/// The executor may keep a non-owning [`DeviceBuffer`] alias for ordinary
/// kernel dispatch while sequence handles retain this owner. The allocation is
/// released exactly once when the last owner drops.
pub(crate) struct SharedTensorBuffer {
    buffer: Option<DeviceBuffer>,
    allocator: Arc<dyn ExecutionProvider>,
    import_guard: Option<Box<dyn core::any::Any + Send + Sync>>,
}

impl SharedTensorBuffer {
    pub(crate) fn new(allocator: Arc<dyn ExecutionProvider>, buffer: DeviceBuffer) -> Arc<Self> {
        Arc::new(Self {
            buffer: Some(buffer),
            allocator,
            import_guard: None,
        })
    }

    fn with_guard(
        allocator: Arc<dyn ExecutionProvider>,
        buffer: DeviceBuffer,
        import_guard: Option<Box<dyn core::any::Any + Send + Sync>>,
    ) -> Arc<Self> {
        Arc::new(Self {
            buffer: Some(buffer),
            allocator,
            import_guard,
        })
    }

    pub(crate) fn allocate_cpu(bytes: usize) -> Result<Arc<Self>> {
        let allocator: Arc<dyn ExecutionProvider> = shared_cpu_ep();
        let buffer = allocator.allocate(bytes.max(1), TensorLayout::contiguous().alignment)?;
        Ok(Self::new(allocator, buffer))
    }

    pub(crate) fn buffer(&self) -> &DeviceBuffer {
        self.buffer
            .as_ref()
            .expect("SharedTensorBuffer buffer taken only in Drop")
    }

    pub(crate) fn buffer_mut(&mut self) -> &mut DeviceBuffer {
        self.buffer
            .as_mut()
            .expect("SharedTensorBuffer buffer taken only in Drop")
    }

    pub(crate) fn allocator(&self) -> &Arc<dyn ExecutionProvider> {
        &self.allocator
    }

    /// Create a non-owning alias suitable for the executor's existing
    /// `DeviceBuffer` dispatch path. The returned handle must not outlive `self`.
    pub(crate) fn alias(&self) -> DeviceBuffer {
        let buffer = self.buffer();
        // SAFETY: `self` owns the allocation and the executor keeps an Arc<Self>
        // alive for at least as long as this alias. The alias is never freed by
        // the EP because it is marked borrowed.
        unsafe {
            DeviceBuffer::from_borrowed_parts(
                buffer.as_ptr() as *mut std::ffi::c_void,
                buffer.device(),
                buffer.len(),
                buffer.alignment(),
            )
        }
    }

    pub(crate) fn into_buffer(mut self) -> DeviceBuffer {
        debug_assert!(
            self.import_guard.is_none(),
            "executor-promoted buffers never carry a foreign import guard"
        );
        self.buffer
            .take()
            .expect("SharedTensorBuffer buffer taken only by into_buffer or Drop")
    }
}

impl std::fmt::Debug for SharedTensorBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedTensorBuffer")
            .field("device", &self.buffer().device())
            .field("len", &self.buffer().len())
            .field("ptr", &self.buffer().as_ptr())
            .finish()
    }
}

impl Drop for SharedTensorBuffer {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            let _ = self.allocator.deallocate(buffer);
        }
        let _ = self.import_guard.take();
    }
}

/// Borrow the raw bytes of a host-accessible device buffer.
///
/// # Safety
///
/// `buffer` must live on a host-accessible device (asserted) and own a valid
/// allocation of `buffer.len()` bytes (the EP contract for
/// [`DeviceBuffer`]). The returned slice borrows `buffer`, so it cannot outlive
/// it. No concurrent writer may exist for the borrow's duration — enforced by
/// the `&DeviceBuffer` shared borrow in safe code.
pub(crate) fn host_bytes(buffer: &DeviceBuffer) -> &[u8] {
    assert!(
        buffer.device().is_host_accessible(),
        "host_bytes on non-host device {:?}",
        buffer.device()
    );
    if buffer.is_empty() {
        return &[];
    }
    // SAFETY: host-accessible device (asserted) means `as_ptr` is a real,
    // readable host address; the EP guarantees `len()` valid bytes behind it.
    // The lifetime is tied to `&buffer`, so the slice cannot dangle, and the
    // shared borrow forbids an aliasing writer while it is live.
    unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const u8, buffer.len()) }
}

/// An owned, host-resident, device-aware tensor (§5, §20.2).
///
/// Owns the [`DeviceBuffer`] that holds its elements and the EP that must free
/// it. On Phase-1 CPU the buffer is a host allocation, so [`Tensor::as_bytes`]
/// and the typed accessors read it directly; the design leaves room for
/// non-host devices (the accessors gate on host accessibility).
pub struct Tensor {
    /// Element type.
    pub dtype: DataType,
    /// Logical shape (static dims).
    pub shape: Vec<usize>,
    /// Physical layout of [`Tensor::buffer`]. Row-major contiguous for tensors
    /// this crate produces.
    pub layout: TensorLayout,
    device: DeviceId,
    /// `Some` while the tensor is live; taken by [`Drop`] to free exactly once.
    buffer: Option<DeviceBuffer>,
    /// The EP that allocated [`Tensor::buffer`] and must deallocate it.
    allocator: Arc<dyn ExecutionProvider>,
    /// Optional opaque guard that owns *foreign* memory this tensor merely
    /// borrows (e.g. a DLPack `DLManagedTensor` imported zero-copy). It is
    /// `None` for every tensor that owns its own allocation. When present,
    /// [`Tensor::buffer`] is a **borrowed** [`DeviceBuffer`] aliasing memory the
    /// guard is responsible for releasing; the guard's own `Drop` runs the
    /// foreign deleter exactly once. [`Drop`] takes it **after** the buffer is
    /// deallocated (a no-op for borrowed buffers) so the memory is never freed
    /// while the buffer still aliases it. The concrete type lives in the caller
    /// crate (the Python binding) — this crate only stores and drops it, so it
    /// stays free of DLPack ABI knowledge.
    import_guard: Option<Box<dyn core::any::Any + Send + Sync>>,
}

/// Debug counters for host traffic explicitly requested through a persistent
/// device binding.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DeviceBindingTransferStats {
    pub host_upload_calls: u64,
    pub host_upload_bytes: u64,
    pub host_download_calls: u64,
    pub host_download_bytes: u64,
}

/// The parameters for [`DeviceIoBinding::allocate`], grouped so the constructor
/// takes a single spec rather than a long positional argument list.
pub(crate) struct DeviceBindingSpec {
    pub(crate) input_name: String,
    pub(crate) bind_input: bool,
    pub(crate) output_name: Option<String>,
    pub(crate) dtype: DataType,
    pub(crate) physical_shape: Vec<usize>,
    pub(crate) logical_shape: Vec<usize>,
    /// Whether graph inputs see the valid prefix rather than allocation capacity.
    pub(crate) expose_logical_input_shape: bool,
    /// Whether an attention-mask binding that exposes its logical length for
    /// multi-token prefill may nonetheless be **frozen to physical capacity for a
    /// single-token decode step** (its causal-window arithmetic saturates at
    /// `q_seq == 1`, so a frozen mask is byte-identical and stays CUDA-graph
    /// capture-eligible). Only meaningful together with a mask input binding.
    pub(crate) decode_freeze_safe_mask: bool,
    /// Bytes reserved in the allocation. Defaults to the physical shape's byte
    /// size; larger values let a binding grow its exposed shape without moving.
    pub(crate) allocation_bytes: Option<usize>,
    /// Byte ranges to commit immediately. Eager providers ignore this because
    /// allocation already commits the whole buffer.
    pub(crate) committed_ranges: Option<Vec<std::ops::Range<usize>>>,
}

/// The parameters for [`DeviceIoBinding::from_external_memory`], grouped the
/// same way [`DeviceBindingSpec`] groups the allocating constructor's.
///
/// `ptr`/`len_bytes` describe memory the **caller** owns; the rest describes how
/// the graph should see it.
pub struct ExternalMemorySpec {
    /// Graph input to bind. Empty when the binding is output-only.
    pub input_name: String,
    /// Whether the buffer is bound as a graph input at all. `false` gives an
    /// output-only binding: the graph writes into the caller's memory without
    /// reading it first.
    pub bind_input: bool,
    pub output_name: Option<String>,
    pub dtype: DataType,
    pub physical_shape: Vec<usize>,
    pub logical_shape: Vec<usize>,
    /// Base address of the caller's allocation, on the session's device.
    pub ptr: *mut core::ffi::c_void,
    /// Length of the caller's allocation in bytes.
    pub len_bytes: usize,
}

impl ExternalMemorySpec {
    /// A buffer bound as a graph input, optionally aliased by an output.
    pub fn input(
        input_name: impl Into<String>,
        output_name: Option<impl Into<String>>,
        dtype: DataType,
        physical_shape: Vec<usize>,
        logical_shape: Vec<usize>,
        ptr: *mut core::ffi::c_void,
        len_bytes: usize,
    ) -> Self {
        Self {
            input_name: input_name.into(),
            bind_input: true,
            output_name: output_name.map(Into::into),
            dtype,
            physical_shape,
            logical_shape,
            ptr,
            len_bytes,
        }
    }

    /// A buffer the graph only writes to.
    pub fn output(
        output_name: impl Into<String>,
        dtype: DataType,
        physical_shape: Vec<usize>,
        logical_shape: Vec<usize>,
        ptr: *mut core::ffi::c_void,
        len_bytes: usize,
    ) -> Self {
        Self {
            input_name: String::new(),
            bind_input: false,
            output_name: Some(output_name.into()),
            dtype,
            physical_shape,
            logical_shape,
            ptr,
            len_bytes,
        }
    }
}

pub struct DeviceIoBinding {
    input_name: String,
    bind_input: bool,
    output_name: Option<String>,
    pub dtype: DataType,
    physical_shape: Vec<usize>,
    logical_shape: Vec<usize>,
    /// Whether graph inputs see the valid prefix rather than allocation capacity.
    expose_logical_input_shape: bool,
    /// Whether a logical-exposing mask binding may be frozen to physical capacity
    /// for a single-token decode step (see [`DeviceBindingSpec::decode_freeze_safe_mask`]).
    decode_freeze_safe_mask: bool,
    buffer: Option<DeviceBuffer>,
    allocator: Arc<dyn ExecutionProvider>,
    transfer_stats: DeviceBindingTransferStats,
    /// Most recent graph generation that captured this address. A stale token
    /// cannot reset another executor's graph.
    device_graph_token: Option<DeviceGraphToken>,
    /// Setup-time registered owner of this binding's sticky validation slot.
    validation_registration: Option<DeviceValidationRegistration>,
    /// Exact deferred-validation generation whose output was submitted into
    /// this binding. Foreign bindings never receive this token.
    device_validation: Option<DeviceValidationToken>,
}

impl DeviceIoBinding {
    pub(crate) fn allocate(
        allocator: Arc<dyn ExecutionProvider>,
        spec: DeviceBindingSpec,
    ) -> Result<Self> {
        let DeviceBindingSpec {
            input_name,
            bind_input,
            output_name,
            dtype,
            physical_shape,
            logical_shape,
            expose_logical_input_shape,
            decode_freeze_safe_mask,
            allocation_bytes,
            committed_ranges,
        } = spec;
        validate_logical_shape(&physical_shape, &logical_shape)?;
        let bytes = checked_expected_bytes(dtype, &physical_shape)
            .ok_or_else(|| SessionError::ShapeOverflow {
                value: format!("device binding '{input_name}'"),
                dims: physical_shape.clone(),
            })?
            .max(1);
        let allocation_bytes = allocation_bytes.unwrap_or(bytes).max(1);
        if allocation_bytes < bytes {
            return Err(SessionError::ExternalBuffer {
                binding: input_name,
                reason: format!(
                    "allocation is {allocation_bytes} bytes but physical shape {physical_shape:?} needs {bytes}"
                ),
            });
        }
        let allocator_for_buffer = allocator.clone();
        let default_range;
        let ranges = match committed_ranges.as_deref() {
            Some(ranges) => ranges,
            None => {
                default_range = 0..allocation_bytes;
                std::slice::from_ref(&default_range)
            }
        };
        let buffer = allocator_for_buffer.allocate_committed(
            allocation_bytes,
            TensorLayout::contiguous().alignment,
            ranges,
        )?;
        let validation_registration = match allocator.register_device_validation_owner() {
            Ok(registration) => registration,
            Err(error) => {
                let _ = allocator.deallocate(buffer);
                return Err(error.into());
            }
        };
        Ok(Self {
            input_name,
            bind_input,
            output_name,
            dtype,
            physical_shape,
            logical_shape,
            expose_logical_input_shape,
            decode_freeze_safe_mask,
            buffer: Some(buffer),
            allocator,
            transfer_stats: DeviceBindingTransferStats::default(),
            device_graph_token: None,
            validation_registration: Some(validation_registration),
            device_validation: None,
        })
    }

    /// Wrap memory the **caller** allocated, rather than allocating here.
    ///
    /// This is the native counterpart of handing ONNX Runtime an externally
    /// allocated tensor: it lets a memory manager outside this crate own the
    /// device allocation and lend it to a session for the binding's lifetime.
    /// Without it the only way to get a persistent device binding is to let the
    /// EP allocate, which puts the bytes outside any external budget.
    ///
    /// The binding **borrows**: `Drop` will not free `ptr`.
    ///
    /// # Safety
    ///
    /// The caller must guarantee all of:
    /// * `ptr` is non-null, points to at least `expected_bytes(dtype,
    ///   physical_shape)` bytes on the EP's device, and is aligned to at least
    ///   the contiguous-layout alignment.
    /// * The allocation outlives this binding **and** every run that reads or
    ///   writes it, including any captured device graph that recorded its
    ///   address.
    /// * No other live handle writes to the same memory while this binding
    ///   exists; the binding assumes exclusive access.
    ///
    /// The byte-length requirement is checked and reported; the lifetime and
    /// aliasing requirements cannot be, which is why this is `unsafe`.
    pub(crate) unsafe fn from_external_memory(
        allocator: Arc<dyn ExecutionProvider>,
        spec: DeviceBindingSpec,
        ptr: *mut core::ffi::c_void,
        len_bytes: usize,
    ) -> Result<Self> {
        let DeviceBindingSpec {
            input_name,
            bind_input,
            output_name,
            dtype,
            physical_shape,
            logical_shape,
            expose_logical_input_shape,
            decode_freeze_safe_mask,
            allocation_bytes: _,
            committed_ranges: _,
        } = spec;
        validate_logical_shape(&physical_shape, &logical_shape)?;
        let required = checked_expected_bytes(dtype, &physical_shape)
            .ok_or_else(|| SessionError::ShapeOverflow {
                value: format!("device binding '{input_name}'"),
                dims: physical_shape.clone(),
            })?
            .max(1);
        if len_bytes < required {
            return Err(SessionError::ExternalBuffer {
                binding: input_name,
                reason: format!(
                    "it is {len_bytes} bytes but {physical_shape:?} of {dtype:?} needs \
                     {required}; pass a buffer at least that large or reduce the physical shape"
                ),
            });
        }
        // Borrowed memory only has to satisfy the dtype's alignment. The EP's
        // 64-byte figure is an *allocation* requirement it imposes on memory it
        // hands out, not a precondition for reading memory someone else owns —
        // the same distinction the zero-copy initializer path already makes.
        let alignment = crate::executor::host_dtype_alignment(dtype);
        if !ptr.is_null() && !ptr.addr().is_multiple_of(alignment) {
            return Err(SessionError::ExternalBuffer {
                binding: input_name,
                reason: format!(
                    "it is at address {:#x}, which is not a multiple of the {alignment}-byte \
                     alignment {dtype:?} requires; allocate it with at least that alignment",
                    ptr.addr()
                ),
            });
        }
        // SAFETY: delegated to this function's own contract; the size check
        // above is the part that can be verified here.
        let buffer = unsafe {
            DeviceBuffer::from_borrowed_mut_parts(ptr, allocator.device_id(), required, alignment)
        }
        .ok_or_else(|| SessionError::ExternalBuffer {
            binding: input_name.clone(),
            reason: "it is null; pass the address of a real allocation".to_string(),
        })?;
        let validation_registration = allocator.register_device_validation_owner()?;
        Ok(Self {
            input_name,
            bind_input,
            output_name,
            dtype,
            physical_shape,
            logical_shape,
            expose_logical_input_shape,
            decode_freeze_safe_mask,
            buffer: Some(buffer),
            allocator,
            transfer_stats: DeviceBindingTransferStats::default(),
            device_graph_token: None,
            validation_registration: Some(validation_registration),
            device_validation: None,
        })
    }

    pub fn input_name(&self) -> &str {
        &self.input_name
    }

    pub(crate) fn binds_input(&self) -> bool {
        self.bind_input
    }

    pub fn output_name(&self) -> Option<&str> {
        self.output_name.as_deref()
    }

    pub fn physical_shape(&self) -> &[usize] {
        &self.physical_shape
    }

    pub fn logical_shape(&self) -> &[usize] {
        &self.logical_shape
    }

    pub(crate) fn kernel_input_shape(&self) -> &[usize] {
        if self.expose_logical_input_shape {
            &self.logical_shape
        } else {
            &self.physical_shape
        }
    }

    /// Whether kernels see a logical prefix whose shape can differ from capacity.
    pub fn has_dynamic_logical_input_shape(&self) -> bool {
        self.bind_input
            && self.expose_logical_input_shape
            && self.logical_shape != self.physical_shape
    }

    /// Whether this input binding is configured to expose its *logical* prefix
    /// (valid length) to kernels rather than the padded physical capacity. Unlike
    /// [`Self::has_dynamic_logical_input_shape`], this is a *static* property of
    /// the binding (fixed at allocation from the consumer-scoped capacity policy)
    /// and does not depend on the current logical shape. Callers that freeze a
    /// growing input to physical capacity for CUDA-graph eligibility must consult
    /// this: a binding that exposes its logical prefix cannot be frozen, because
    /// at least one consumer requires the valid length rather than the padded
    /// capacity (e.g. GLM-5.2's indexer arithmetic branch).
    pub fn exposes_logical_input_shape(&self) -> bool {
        self.bind_input && self.expose_logical_input_shape
    }

    /// Whether this mask input binding — even if it [`exposes_logical_input_shape`]
    /// for multi-token prefill — may be frozen to physical capacity on a
    /// **single-token decode step** without changing the additive bias. Holds
    /// when the mask feeds only the additive causal-mask builder cone, whose
    /// query-window arithmetic saturates at `q_seq == 1`
    /// (see `mask_binding_feeds_additive_causal_builder`). A frozen decode mask
    /// keeps `logical == physical`, so the step stays CUDA-graph capture-eligible.
    ///
    /// [`exposes_logical_input_shape`]: Self::exposes_logical_input_shape
    pub fn mask_decode_freeze_safe(&self) -> bool {
        self.bind_input && self.decode_freeze_safe_mask
    }

    pub(crate) fn set_device_graph_token(&mut self, token: DeviceGraphToken) {
        self.device_graph_token = Some(token);
    }

    pub(crate) fn validation_registration(&self) -> &DeviceValidationRegistration {
        self.validation_registration
            .as_ref()
            .expect("device binding validation registration exists until Drop")
    }

    pub(crate) fn set_device_validation(&mut self, token: DeviceValidationToken) {
        self.device_validation = Some(token);
    }

    #[cfg(test)]
    pub(crate) fn device_validation_token_for_test(&self) -> Option<DeviceValidationToken> {
        self.device_validation
    }

    pub fn set_logical_shape(&mut self, shape: Vec<usize>) -> Result<()> {
        validate_logical_shape(&self.physical_shape, &shape)?;
        self.logical_shape = shape;
        Ok(())
    }

    pub fn set_physical_and_logical_shapes(
        &mut self,
        physical_shape: Vec<usize>,
        logical_shape: Vec<usize>,
    ) -> Result<()> {
        validate_logical_shape(&physical_shape, &logical_shape)?;
        let required = checked_expected_bytes(self.dtype, &physical_shape).ok_or_else(|| {
            SessionError::ShapeOverflow {
                value: format!("device binding '{}'", self.input_name),
                dims: physical_shape.clone(),
            }
        })?;
        if required > self.buffer().len() {
            return Err(SessionError::ExternalBuffer {
                binding: self.input_name.clone(),
                reason: format!(
                    "shape {physical_shape:?} needs {required} bytes but allocation has {}",
                    self.buffer().len()
                ),
            });
        }
        self.physical_shape = physical_shape;
        self.logical_shape = logical_shape;
        Ok(())
    }

    pub fn commit_range(&mut self, byte_offset: usize, bytes: usize) -> Result<()> {
        let buffer = self
            .buffer
            .as_ref()
            .expect("DeviceIoBinding buffer taken only in Drop");
        self.allocator
            .commit_allocation_range(buffer, byte_offset, bytes)?;
        Ok(())
    }

    pub fn commit_binding_ranges(&self, ranges: &[(&DeviceIoBinding, usize, usize)]) -> Result<()> {
        let buffers = ranges
            .iter()
            .map(|&(binding, offset, bytes)| {
                (
                    binding
                        .buffer
                        .as_ref()
                        .expect("DeviceIoBinding buffer taken only in Drop"),
                    offset,
                    bytes,
                )
            })
            .collect::<Vec<_>>();
        self.allocator.commit_allocation_ranges(&buffers)?;
        Ok(())
    }

    pub fn commit_binding_ranges_with_mapped_growth(
        &self,
        ranges: &[(&DeviceIoBinding, usize, usize)],
        grant: &mut onnx_runtime_memory_governor::MappedGrowthGrant,
    ) -> Result<u64> {
        let buffers = ranges
            .iter()
            .map(|&(binding, offset, bytes)| {
                (
                    binding
                        .buffer
                        .as_ref()
                        .expect("DeviceIoBinding buffer taken only in Drop"),
                    offset,
                    bytes,
                )
            })
            .collect::<Vec<_>>();
        Ok(self
            .allocator
            .commit_allocation_ranges_with_mapped_growth(&buffers, grant)?)
    }

    pub fn mapped_bytes_for_binding_ranges(
        &self,
        ranges: &[(&DeviceIoBinding, usize, usize)],
    ) -> Result<u64> {
        let buffers = ranges
            .iter()
            .map(|&(binding, offset, bytes)| {
                (
                    binding
                        .buffer
                        .as_ref()
                        .expect("DeviceIoBinding buffer taken only in Drop"),
                    offset,
                    bytes,
                )
            })
            .collect::<Vec<_>>();
        Ok(self
            .allocator
            .mapped_bytes_for_allocation_ranges(&buffers)?)
    }

    pub fn decommit_range(&mut self, byte_offset: usize, bytes: usize) -> Result<()> {
        let buffer = self
            .buffer
            .as_ref()
            .expect("DeviceIoBinding buffer taken only in Drop");
        self.allocator
            .decommit_allocation_range(buffer, byte_offset, bytes)?;
        Ok(())
    }

    pub fn committed_bytes(&self) -> usize {
        let buffer = self
            .buffer
            .as_ref()
            .expect("DeviceIoBinding buffer taken only in Drop");
        self.allocator.allocation_committed_bytes(buffer)
    }

    pub fn device_ptr(&self) -> *const std::ffi::c_void {
        self.buffer().as_ptr()
    }

    /// The execution provider that owns this binding's device allocation.
    /// Callers use it to allocate device scratch compatible with
    /// [`Self::snapshot_device_into`] / [`Self::restore_device_from`].
    pub fn allocator(&self) -> &Arc<dyn ExecutionProvider> {
        &self.allocator
    }

    /// Copy the first `bytes` of this binding's device allocation into
    /// `scratch` (device→device, no host round-trip). Used to snapshot the
    /// destructive recurrent/conv state before a speculative verify overwrites
    /// it. `scratch` must be owned by the same EP and hold at least `bytes`.
    pub fn snapshot_device_into(&self, scratch: &mut DeviceBuffer, bytes: usize) -> Result<()> {
        let buffer = self.buffer();
        if bytes > buffer.len() {
            return Err(SessionError::ExternalBuffer {
                binding: self.input_name.clone(),
                reason: format!(
                    "device snapshot of {bytes} bytes exceeds allocation of {}",
                    buffer.len()
                ),
            });
        }
        self.allocator
            .copy_device_to_device(buffer, 0, scratch, 0, bytes)?;
        Ok(())
    }

    /// Copy the first `bytes` of `scratch` back into this binding's device
    /// allocation (device→device). Inverse of [`Self::snapshot_device_into`];
    /// leaves the binding shape unchanged so it never invalidates a captured
    /// decode graph.
    pub fn restore_device_from(&mut self, scratch: &DeviceBuffer, bytes: usize) -> Result<()> {
        let buffer = self
            .buffer
            .as_mut()
            .expect("DeviceIoBinding buffer taken only in Drop");
        if bytes > buffer.len() {
            return Err(SessionError::ExternalBuffer {
                binding: self.input_name.clone(),
                reason: format!(
                    "device restore of {bytes} bytes exceeds allocation of {}",
                    buffer.len()
                ),
            });
        }
        self.allocator
            .copy_device_to_device(scratch, 0, buffer, 0, bytes)?;
        Ok(())
    }

    pub fn transfer_stats(&self) -> DeviceBindingTransferStats {
        self.transfer_stats
    }

    pub fn write_bytes(&mut self, byte_offset: usize, bytes: &[u8]) -> Result<()> {
        let buffer = self
            .buffer
            .as_mut()
            .expect("DeviceIoBinding buffer taken only in Drop");
        self.allocator
            .copy_from_host_at(bytes, buffer, byte_offset)?;
        self.transfer_stats.host_upload_calls += 1;
        self.transfer_stats.host_upload_bytes += bytes.len() as u64;
        Ok(())
    }

    pub fn read_bytes(&mut self) -> Result<Vec<u8>> {
        let mut bytes = vec![0; self.buffer().len()];
        self.read_bytes_into(&mut bytes)?;
        Ok(bytes)
    }

    pub fn read_bytes_into(&mut self, bytes: &mut [u8]) -> Result<()> {
        if bytes.is_empty() {
            self.allocator.sync()?;
        } else {
            self.allocator.copy_to_host(self.buffer(), bytes)?;
        }
        self.check_and_reset_device_validation()?;
        self.transfer_stats.host_download_calls += 1;
        self.transfer_stats.host_download_bytes += bytes.len() as u64;
        Ok(())
    }

    fn check_and_reset_device_validation(&self) -> Result<()> {
        let Some(token) = self.device_validation else {
            return Ok(());
        };
        let flags = self
            .allocator
            .consume_device_validation_error(self.validation_registration(), token)?;
        if flags != 0 {
            return Err(onnx_runtime_ep_api::EpError::KernelFailed(format!(
                "{}: device validation failed (flags=0x{flags:x})",
                self.allocator.name()
            ))
            .into());
        }
        Ok(())
    }

    pub fn read_bytes_range(&mut self, byte_offset: usize, byte_len: usize) -> Result<Vec<u8>> {
        let end =
            byte_offset
                .checked_add(byte_len)
                .ok_or_else(|| SessionError::ExternalBuffer {
                    binding: self.input_name.clone(),
                    reason: format!(
                        "read range offset {byte_offset} plus {byte_len} bytes overflows"
                    ),
                })?;
        let buffer = self.buffer();
        if end > buffer.len() {
            return Err(SessionError::ExternalBuffer {
                binding: self.input_name.clone(),
                reason: format!(
                    "read range {byte_offset}..{end} exceeds allocation of {} bytes",
                    buffer.len()
                ),
            });
        }
        let mut bytes = vec![0; byte_len];
        if byte_len == 0 {
            self.allocator.sync()?;
            self.check_and_reset_device_validation()?;
            return Ok(bytes);
        }
        // SAFETY: the offset range is checked inside the live allocation above;
        // the borrowed handle owns nothing and `DeviceBuffer` has no destructor.
        let alias = unsafe {
            DeviceBuffer::from_borrowed_parts(
                (buffer.as_ptr() as *const u8).add(byte_offset) as *mut std::ffi::c_void,
                buffer.device(),
                byte_len,
                buffer.alignment(),
            )
        };
        self.allocator.copy_to_host(&alias, &mut bytes)?;
        self.check_and_reset_device_validation()?;
        self.transfer_stats.host_download_calls += 1;
        self.transfer_stats.host_download_bytes += bytes.len() as u64;
        Ok(bytes)
    }

    pub fn device_argmax_supported(&self) -> bool {
        // Gate on both EP capability and logits dtype: the device-argmax kernel
        // handles f32/f16/bf16 only. Returning false for any other dtype routes
        // the greedy step to the host path instead of dispatching a kernel that
        // would reject the dtype (the greedy fast path reads this predicate).
        self.allocator.device_argmax_supported()
            && matches!(
                self.dtype,
                DataType::Float32 | DataType::Float16 | DataType::BFloat16
            )
    }

    pub fn device_argmax(
        &self,
        elements: usize,
        batch: usize,
        result: &mut DeviceIoBinding,
    ) -> Result<()> {
        self.device_argmax_with_tie_break(
            elements,
            batch,
            result,
            onnx_runtime_ep_api::ArgmaxTieBreak::LowestIndex,
        )
    }

    pub fn device_argmax_with_tie_break(
        &self,
        elements: usize,
        batch: usize,
        result: &mut DeviceIoBinding,
        tie_break: onnx_runtime_ep_api::ArgmaxTieBreak,
    ) -> Result<()> {
        if !matches!(
            self.dtype,
            DataType::Float32 | DataType::Float16 | DataType::BFloat16
        ) || result.dtype != DataType::Uint32
        {
            return Err(SessionError::Internal(format!(
                "device argmax requires f32/f16/bf16 logits and u32 result, got {:?} and {:?}",
                self.dtype, result.dtype
            )));
        }
        if !Arc::ptr_eq(&self.allocator, &result.allocator) {
            return Err(SessionError::Internal(
                "device argmax bindings must belong to the same execution provider".into(),
            ));
        }
        Ok(self.allocator.device_argmax(
            self.buffer(),
            elements,
            batch,
            self.dtype,
            result.buffer_mut(),
            tie_break,
        )?)
    }

    /// Fold the just-selected greedy token into the persistent decode bindings
    /// device-to-device for the native CUDA device-token-loop. `self` is the
    /// device-argmax result binding (`result[0]` = token id, `result[1]` =
    /// capture-error word). The token is written as an `i64` into `input_ids`,
    /// `next_position` into `position_ids`, a `1` into `attention_mask` at
    /// `next_position` (guarded by the binding's mask width), the token is
    /// appended to `scratch[step]`, and the capture-error word is OR-ed into
    /// `scratch[capacity]`. No host sync is issued; the caller drains `scratch`
    /// once per chain. All bindings must belong to the same execution provider.
    #[allow(clippy::too_many_arguments)]
    pub fn device_token_writer(
        &self,
        input_ids: &DeviceIoBinding,
        position_ids: Option<&DeviceIoBinding>,
        attention_mask: &DeviceIoBinding,
        scratch: &DeviceIoBinding,
        capacity: usize,
        next_position: i64,
        mask_len: usize,
        step: u32,
    ) -> Result<()> {
        if self.dtype != DataType::Uint32 || scratch.dtype != DataType::Uint32 {
            return Err(SessionError::Internal(format!(
                "device token writer requires u32 result/scratch, got {:?} and {:?}",
                self.dtype, scratch.dtype
            )));
        }
        let write_position = position_ids.is_some();
        // When the model has no persistent position_ids binding (position is
        // derived from the mask), reuse input_ids as a harmless stand-in buffer
        // and gate the position write off in the kernel.
        let position_binding = position_ids.unwrap_or(input_ids);
        for binding in [input_ids, position_binding, attention_mask, scratch] {
            if !Arc::ptr_eq(&self.allocator, &binding.allocator) {
                return Err(SessionError::Internal(
                    "device token writer bindings must belong to the same execution provider"
                        .into(),
                ));
            }
        }
        Ok(self.allocator.device_token_writer(
            self.buffer(),
            input_ids.buffer(),
            position_binding.buffer(),
            attention_mask.buffer(),
            scratch.buffer(),
            capacity,
            next_position,
            mask_len,
            write_position,
            step,
        )?)
    }

    pub(crate) fn buffer(&self) -> &DeviceBuffer {
        self.buffer
            .as_ref()
            .expect("DeviceIoBinding buffer taken only in Drop")
    }

    pub(crate) fn buffer_mut(&mut self) -> &mut DeviceBuffer {
        self.buffer
            .as_mut()
            .expect("DeviceIoBinding buffer taken only in Drop")
    }
}

fn validate_logical_shape(physical: &[usize], logical: &[usize]) -> Result<()> {
    if physical.len() != logical.len()
        || physical
            .iter()
            .zip(logical)
            .any(|(&capacity, &valid)| valid > capacity)
    {
        return Err(SessionError::Internal(format!(
            "device binding logical shape {logical:?} exceeds physical capacity {physical:?}"
        )));
    }
    Ok(())
}

impl std::fmt::Debug for DeviceIoBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceIoBinding")
            .field("input_name", &self.input_name)
            .field("bind_input", &self.bind_input)
            .field("output_name", &self.output_name)
            .field("dtype", &self.dtype)
            .field("physical_shape", &self.physical_shape)
            .field("logical_shape", &self.logical_shape)
            .field(
                "expose_logical_input_shape",
                &self.expose_logical_input_shape,
            )
            .field("decode_freeze_safe_mask", &self.decode_freeze_safe_mask)
            .field("device", &self.buffer().device())
            .field("device_ptr", &self.device_ptr())
            .field("transfer_stats", &self.transfer_stats)
            .finish()
    }
}

impl Drop for DeviceIoBinding {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            let mut safe_to_release = true;
            if let Err(error) = self.allocator.sync() {
                safe_to_release = false;
                eprintln!(
                    "[onnx-runtime-session] device binding drop could not synchronize deferred \
                     work before release: {error}"
                );
            }
            if safe_to_release && let Some(token) = self.device_validation {
                match self
                    .allocator
                    .consume_device_validation_error(self.validation_registration(), token)
                {
                    Ok(0) => {}
                    Ok(flags) => eprintln!(
                        "[onnx-runtime-session] device binding drop consumed its deferred \
                         validation failure (flags=0x{flags:x})"
                    ),
                    Err(error) => {
                        safe_to_release = false;
                        eprintln!(
                            "[onnx-runtime-session] device binding drop could not consume its \
                             deferred validation: {error}"
                        );
                    }
                }
            }
            if let Some(token) = self.device_graph_token.take()
                && let Err(error) = self.allocator.reset_owned_device_graph(token)
            {
                safe_to_release = false;
                eprintln!(
                    "[onnx-runtime-session] device binding drop could not retire its captured \
                     graph generation: {error}"
                );
            }
            if safe_to_release {
                let _ = self.allocator.deallocate(buffer);
            } else {
                // `DeviceBuffer` has no freeing Drop. Retaining the allocation
                // is safer than releasing storage that in-flight work or an
                // unretired graph may still reference.
                eprintln!(
                    "[onnx-runtime-session] quarantining device binding allocation after failed \
                     drop cleanup"
                );
                drop(buffer);
            }
        }
        if let Some(registration) = self.validation_registration.as_mut() {
            let owner = registration.owner();
            if let Err(error) = self
                .allocator
                .unregister_device_validation_owner(registration)
            {
                eprintln!(
                    "[onnx-runtime-session] device binding drop could not unregister validation \
                     owner {}: {error}",
                    owner.get()
                );
            } else {
                self.validation_registration = None;
            }
        }
    }
}

impl Tensor {
    pub(crate) fn copy_from_device_buffer(
        source_allocator: &Arc<dyn ExecutionProvider>,
        source: &DeviceBuffer,
        dtype: DataType,
        shape: Vec<usize>,
    ) -> Result<Self> {
        let mut tensor = Self::allocate_cpu(dtype, shape)?;
        let destination = tensor.buffer.as_mut().ok_or_else(|| {
            SessionError::Internal("new host output tensor has no backing buffer".into())
        })?;
        if source.len() != destination.len() {
            return Err(SessionError::Internal(format!(
                "device output has {} bytes but its host tensor allocation has {}",
                source.len(),
                destination.len()
            )));
        }
        // SAFETY: `allocate_cpu` returned a live host-accessible allocation,
        // exclusively borrowed here, with exactly `destination.len()` bytes.
        let host = unsafe {
            std::slice::from_raw_parts_mut(destination.as_mut_ptr().cast::<u8>(), destination.len())
        };
        source_allocator.copy_to_host(source, host)?;
        Ok(tensor)
    }

    pub(crate) fn allocate_cpu(dtype: DataType, shape: Vec<usize>) -> Result<Self> {
        let numel = shape.iter().try_fold(1usize, |product, &dim| {
            product.checked_mul(dim).ok_or_else(|| {
                SessionError::Internal(format!(
                    "Tensor::allocate_cpu: element count overflows for shape {shape:?}"
                ))
            })
        })?;
        let bytes = dtype.checked_storage_bytes(numel).ok_or_else(|| {
            SessionError::Internal(format!(
                "Tensor::allocate_cpu: byte count overflows for shape {shape:?} dtype {dtype:?}"
            ))
        })?;
        let allocator: Arc<dyn ExecutionProvider> = shared_cpu_ep();
        let layout = TensorLayout::contiguous();
        let buffer = allocator.allocate(bytes.max(1), layout.alignment)?;
        Ok(Self {
            dtype,
            shape,
            layout,
            device: buffer.device(),
            buffer: Some(buffer),
            allocator,
            import_guard: None,
        })
    }

    pub(crate) fn copy_from_host_at(&mut self, offset: usize, bytes: &[u8]) -> Result<()> {
        let buffer = self.buffer.as_mut().ok_or_else(|| {
            SessionError::Internal("Tensor buffer is unavailable for writing".to_string())
        })?;
        self.allocator.copy_from_host_at(bytes, buffer, offset)?;
        Ok(())
    }

    /// Allocate a tensor from raw little-endian element bytes using `allocator`.
    ///
    /// `bytes` must hold exactly `storage_bytes(numel)` bytes for `dtype` and
    /// `shape`.
    pub(crate) fn from_raw_in(
        allocator: Arc<dyn ExecutionProvider>,
        dtype: DataType,
        shape: Vec<usize>,
        bytes: &[u8],
    ) -> Result<Self> {
        let expected =
            checked_expected_bytes(dtype, &shape).ok_or_else(|| SessionError::ShapeOverflow {
                value: "Tensor::from_raw_in".to_string(),
                dims: shape.clone(),
            })?;
        if bytes.len() != expected {
            return Err(SessionError::Internal(format!(
                "Tensor::from_raw_in: {} bytes for shape {shape:?} dtype {dtype:?}, expected {expected}",
                bytes.len()
            )));
        }
        let layout = TensorLayout::contiguous();
        let align = layout.alignment;
        let mut buffer = allocator.allocate(expected.max(1), align)?;
        allocator.copy_from_host(bytes, &mut buffer)?;
        Ok(Self {
            dtype,
            shape,
            layout,
            device: buffer.device(),
            buffer: Some(buffer),
            allocator,
            import_guard: None,
        })
    }

    /// Build a tensor from raw little-endian bytes on the shared CPU device.
    pub fn from_raw(dtype: DataType, shape: Vec<usize>, bytes: &[u8]) -> Result<Self> {
        Self::from_raw_in(shared_cpu_ep(), dtype, shape, bytes)
    }

    /// Allocate a zero-initialised tensor on the shared CPU device.
    ///
    /// Zeroes the buffer in place. `from_raw` with a zeroed `Vec` allocates
    /// these bytes twice and memcpys between them, which on a hybrid decoder's
    /// per-layer `conv_state`/`recurrent_state` is the whole cost of the call.
    pub fn zeros(dtype: DataType, shape: Vec<usize>) -> Result<Self> {
        let allocator = shared_cpu_ep();
        let numel: usize = shape.iter().product();
        let expected = dtype.storage_bytes(numel);
        let layout = TensorLayout::contiguous();
        let mut buffer = allocator.allocate(expected.max(1), layout.alignment)?;
        assert!(
            buffer.device().is_host_accessible(),
            "zeros on non-host device {:?}",
            buffer.device()
        );
        if expected > 0 {
            let dst = buffer.as_mut_ptr() as *mut u8;
            // SAFETY: host-accessible device (asserted); `dst` is a unique
            // writable host pointer obtained via `&mut buffer` with no alias,
            // and the allocation is at least `expected` bytes because that is
            // what was requested.
            unsafe { std::ptr::write_bytes(dst, 0, expected) };
        }
        Ok(Self::from_owned_buffer(allocator, dtype, shape, buffer))
    }

    /// Allocate a host tensor and let `fill` write its bytes in place.
    ///
    /// The alternative is to build the bytes in a `Vec` and hand them to
    /// [`from_raw`], which allocates the same bytes twice and memcpys between
    /// them -- the objection [`zeros`] already records, and the reason `zeros`
    /// exists. It matters most on the view-graph-output path, where the producer
    /// is a strided gather whose result is the tensor and nothing else: there
    /// the second allocation and copy are the entire remaining overhead of
    /// materializing the output.
    ///
    /// `fill` receives exactly `storage_bytes(numel)` bytes of **uninitialized**
    /// memory as a `&mut [u8]` and **must write every one of them**. Reading an
    /// uninitialized `u8` is undefined behaviour -- Miri rejects it -- so this
    /// is a hard obligation on the caller, not a "you get arbitrary bytes"
    /// convenience. The only caller is the view-materialization path, whose
    /// gather writes `numel * esize` bytes in disjoint blocks covering the whole
    /// destination; `gather_view_into`'s falsifiers (`no_output_byte_is_left_
    /// unwritten`, and the parallel-vs-serial bit-identity test) exist to keep
    /// that true.
    ///
    /// If `fill` panics the buffer is dropped without reaching
    /// [`ExecutionProvider::deallocate`], which per the ep-api contract leaks it
    /// rather than double-freeing. That is the same behaviour as any other
    /// panic between `allocate` and tensor construction, and is why `fill`
    /// should stay a straight-line writer.
    pub(crate) fn from_host_fill(
        dtype: DataType,
        shape: Vec<usize>,
        fill: impl FnOnce(&mut [u8]),
    ) -> Result<Self> {
        let expected =
            checked_expected_bytes(dtype, &shape).ok_or_else(|| SessionError::ShapeOverflow {
                value: "Tensor::from_host_fill".to_string(),
                dims: shape.clone(),
            })?;
        let allocator = shared_cpu_ep();
        let layout = TensorLayout::contiguous();
        let mut buffer = allocator.allocate(expected.max(1), layout.alignment)?;
        assert!(
            buffer.device().is_host_accessible(),
            "from_host_fill on non-host device {:?}",
            buffer.device()
        );
        if expected > 0 {
            // SAFETY: host-accessible device (asserted); `as_mut_ptr` comes from
            // a unique `&mut buffer` with no alias, and the allocation is at
            // least `expected` bytes because that is what was requested. `u8` has
            // no invalid bit patterns, so a `&mut [u8]` over it is valid before
            // it is written.
            let dst =
                unsafe { std::slice::from_raw_parts_mut(buffer.as_mut_ptr() as *mut u8, expected) };
            fill(dst);
        }
        Ok(Self::from_owned_buffer(allocator, dtype, shape, buffer))
    }

    /// Take ownership of an already-allocated, contiguous `buffer` (allocated by
    /// `allocator`) and wrap it as a row-major tensor **without copying**. The
    /// executor uses this to hand a produced output's host buffer straight to
    /// the caller instead of round-tripping the bytes through
    /// [`copy_to_host`](ExecutionProvider::copy_to_host) plus a second
    /// allocate+copy in [`from_raw`]. The bytes are the exact same memory the
    /// kernel wrote, so this is numerically identical to the copy path.
    ///
    /// `buffer` must hold exactly `storage_bytes(numel)` bytes for `dtype` and
    /// `shape`, must be host-resident, and must be owned (not borrowed) so the
    /// tensor can free it exactly once on drop.
    pub(crate) fn from_owned_buffer(
        allocator: Arc<dyn ExecutionProvider>,
        dtype: DataType,
        shape: Vec<usize>,
        buffer: DeviceBuffer,
    ) -> Self {
        debug_assert!(
            !buffer.is_borrowed(),
            "from_owned_buffer requires an owned buffer"
        );
        debug_assert_eq!(
            buffer.len(),
            dtype.storage_bytes(shape.iter().product::<usize>()).max(1),
            "from_owned_buffer size mismatch for shape {shape:?} dtype {dtype:?}",
        );
        let device = buffer.device();
        Self {
            dtype,
            shape,
            layout: TensorLayout::contiguous(),
            device,
            buffer: Some(buffer),
            allocator,
            import_guard: None,
        }
    }

    /// Build an `f32` tensor from a dense row-major slice.
    pub fn from_f32(shape: &[usize], data: &[f32]) -> Result<Self> {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Self::from_raw(DataType::Float32, shape.to_vec(), &bytes)
    }

    /// Build an `i64` tensor from a dense row-major slice.
    pub fn from_i64(shape: &[usize], data: &[i64]) -> Result<Self> {
        let mut bytes = Vec::with_capacity(data.len() * 8);
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Self::from_raw(DataType::Int64, shape.to_vec(), &bytes)
    }

    pub(crate) fn into_shared_parts(
        mut self,
    ) -> (Arc<SharedTensorBuffer>, DataType, Vec<usize>, TensorLayout) {
        let buffer = self
            .buffer
            .take()
            .expect("Tensor buffer taken only by into_shared_parts or Drop");
        let storage = SharedTensorBuffer::with_guard(
            Arc::clone(&self.allocator),
            buffer,
            self.import_guard.take(),
        );
        let dtype = self.dtype;
        let shape = std::mem::take(&mut self.shape);
        let layout = std::mem::take(&mut self.layout);
        (storage, dtype, shape, layout)
    }

    /// The device this tensor lives on.
    pub fn device(&self) -> DeviceId {
        self.device
    }

    /// Wrap **foreign, borrowed** memory in a `Tensor`, with an opaque `guard`
    /// that releases the foreign allocation when the tensor is dropped.
    ///
    /// This is the zero-copy *import* constructor: `buffer` must be a
    /// **borrowed** [`DeviceBuffer`] (built via
    /// [`DeviceBuffer::from_borrowed_parts`]) aliasing memory owned by whatever
    /// `guard` boxes up — for a DLPack import, `guard` owns the foreign
    /// `DLManagedTensor` and its `Drop` calls that tensor's `deleter` exactly
    /// once. Because the buffer is borrowed, the owning EP's `deallocate` is a
    /// no-op for it, so the *only* thing that frees the aliased memory is the
    /// guard.
    ///
    /// # Ordering invariant
    ///
    /// [`Drop`] deallocates `buffer` (a no-op for a borrowed buffer) and only
    /// **then** drops the guard, so the guard's deleter never runs while the
    /// buffer still aliases the foreign memory. Do not rely on the guard freeing
    /// anything the buffer still points at before `drop` completes.
    ///
    /// # Panics
    ///
    /// Panics (debug builds) if `buffer` is not borrowed — an owned buffer here
    /// would be double-freed (once by the EP, once by the guard).
    pub fn from_borrowed_parts_with_guard(
        allocator: Arc<dyn ExecutionProvider>,
        dtype: DataType,
        shape: Vec<usize>,
        layout: TensorLayout,
        buffer: DeviceBuffer,
        guard: Box<dyn core::any::Any + Send + Sync>,
    ) -> Self {
        debug_assert!(
            buffer.is_borrowed(),
            "from_borrowed_parts_with_guard requires a borrowed DeviceBuffer; \
             an owned buffer would be freed twice (EP deallocate + guard)"
        );
        Self {
            dtype,
            shape,
            layout,
            device: buffer.device(),
            buffer: Some(buffer),
            allocator,
            import_guard: Some(guard),
        }
    }

    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Base pointer of this tensor's backing allocation.
    ///
    /// For host-accessible devices (CPU, MLX) this is a dereferenceable host
    /// pointer; for device memory (CUDA/ROCm) it is an **opaque device address**
    /// only meaningful inside the owning EP's context — never dereference it on
    /// the host. This is the device-agnostic base the zero-copy DLPack **export**
    /// path hands to a consumer, so a CUDA-resident output can be borrowed as a
    /// `kDLCUDA` tensor without a host round-trip. Returns null for an empty
    /// (zero-element) tensor.
    pub fn device_ptr(&self) -> *const std::ffi::c_void {
        if self.numel() == 0 {
            std::ptr::null()
        } else {
            self.buffer().as_ptr()
        }
    }

    /// Block until all pending work on the owning EP's stream completes.
    ///
    /// Device-agnostic: the CPU EP's `sync` is a no-op, while the CUDA EP fully
    /// synchronizes its compute stream. The DLPack **export** path calls this
    /// before handing a `kDLCUDA` buffer to a foreign consumer, so the producer's
    /// device work is guaranteed complete (and thus the data valid) regardless of
    /// which stream the consumer reads on — the conservative, always-correct end
    /// of the DLPack stream handshake.
    pub fn sync(&self) -> Result<()> {
        self.allocator.sync()?;
        Ok(())
    }

    /// Deep-copy this tensor while reporting allocation and shape failures.
    ///
    /// Prefer this over [`Clone`] in fallible runtime control-flow paths where an
    /// allocator failure must be propagated instead of panicking.
    pub fn try_clone(&self) -> SequenceResult<Tensor> {
        const OP: &str = "Tensor::try_clone";
        let shape = clone_shape(OP, &self.shape)?;
        if checked_expected_bytes(self.dtype, &shape).is_none() {
            return Err(SequenceError::ShapeOverflow {
                op: OP,
                context: "tensor byte count",
                shape,
            });
        }
        Self::from_raw_in(
            Arc::clone(&self.allocator),
            self.dtype,
            shape,
            self.as_bytes(),
        )
        .map_err(|source| SequenceError::TensorCreation { op: OP, source })
    }

    fn buffer(&self) -> &DeviceBuffer {
        self.buffer
            .as_ref()
            .expect("Tensor buffer taken only in Drop")
    }

    /// Borrow the raw little-endian element bytes (host tensors only).
    pub fn as_bytes(&self) -> &[u8] {
        let n = self.dtype.storage_bytes(self.numel());
        &host_bytes(self.buffer())[..n]
    }

    /// Replace this tensor's logical bytes without reallocating its backing
    /// buffer. Used by control-flow iteration inputs whose dtype/shape stay
    /// constant while their values change.
    pub(crate) fn overwrite_bytes(&mut self, bytes: &[u8]) -> Result<()> {
        let expected = self.dtype.storage_bytes(self.numel());
        if bytes.len() != expected {
            return Err(SessionError::Internal(format!(
                "Tensor::overwrite_bytes: got {} bytes for shape {:?} dtype {:?}, expected {expected}",
                bytes.len(),
                self.shape,
                self.dtype
            )));
        }
        let buffer = self
            .buffer
            .as_mut()
            .expect("Tensor buffer taken only in Drop");
        self.allocator.copy_from_host(bytes, buffer)?;
        Ok(())
    }

    /// Borrow the elements as `f32` without copying (host tensors only).
    /// Returns `None` on big-endian or unexpectedly misaligned hosts; panics
    /// only when called on a non-`Float32` tensor.
    pub fn try_as_slice_f32(&self) -> Option<&[f32]> {
        assert_eq!(
            self.dtype,
            DataType::Float32,
            "try_as_slice_f32 on non-f32 tensor"
        );
        if cfg!(target_endian = "big") {
            return None;
        }
        let bytes = self.as_bytes();
        if bytes.as_ptr().align_offset(std::mem::align_of::<f32>()) != 0 {
            return None;
        }
        // SAFETY: Float32 tensor storage contains `numel` contiguous, aligned
        // f32 elements and the returned slice cannot outlive the tensor borrow.
        Some(unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), self.numel()) })
    }

    /// Borrow Float16/BFloat16 storage as raw 16-bit elements without copying.
    /// Returns `None` on big-endian or unexpectedly misaligned hosts.
    pub fn try_as_slice_u16(&self) -> Option<&[u16]> {
        assert!(
            matches!(self.dtype, DataType::Float16 | DataType::BFloat16),
            "try_as_slice_u16 on non-16-bit float tensor"
        );
        if cfg!(target_endian = "big") {
            return None;
        }
        let bytes = self.as_bytes();
        if bytes.as_ptr().align_offset(std::mem::align_of::<u16>()) != 0 {
            return None;
        }
        // SAFETY: 16-bit float storage contains `numel` contiguous, aligned u16
        // bit patterns and the returned slice cannot outlive the tensor borrow.
        Some(unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<u16>(), self.numel()) })
    }

    /// Copy out the elements as `f32`. Panics if the dtype is not `Float32`.
    pub fn to_vec_f32(&self) -> Vec<f32> {
        assert_eq!(
            self.dtype,
            DataType::Float32,
            "to_vec_f32 on non-f32 tensor"
        );
        self.try_as_slice_f32().map_or_else(
            || {
                read_vec_le::<f32>(self.as_bytes())
                    .expect("Float32 tensor storage length must be a multiple of 4 bytes")
            },
            <[f32]>::to_vec,
        )
    }

    /// Copy out the elements as `i64`. Panics if the dtype is not `Int64`.
    pub fn to_vec_i64(&self) -> Vec<i64> {
        assert_eq!(self.dtype, DataType::Int64, "to_vec_i64 on non-i64 tensor");
        read_vec_le(self.as_bytes())
            .expect("Int64 tensor storage length must be a multiple of 8 bytes")
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        self.try_clone()
            .expect("Tensor::clone: re-allocation of identical bytes")
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .field("device", &self.device)
            .finish()
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            // `DeviceBuffer` has no `Drop`; the owning EP must free it exactly
            // once (ep-api §4.4 invariant #2). Errors here cannot be surfaced
            // from `drop`, so we swallow them — a failed free leaks, never
            // double-frees.
            let _ = self.allocator.deallocate(buffer);
        }
        // Release any foreign (DLPack-imported) allocation *after* the buffer
        // aliasing it has been handed back to the EP. For a borrowed buffer the
        // `deallocate` above is a no-op, so this guard's `Drop` (which runs the
        // foreign deleter) is the sole owner that frees the memory — and it must
        // run last, once the buffer no longer aliases it. `None` for tensors
        // that own their allocation.
        let _ = self.import_guard.take();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::raw::c_void;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A guard whose `Drop` bumps a shared counter — stands in for the DLPack
    /// deleter the Python binding boxes into an imported tensor.
    struct CountingGuard(Arc<AtomicUsize>);
    impl Drop for CountingGuard {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn from_raw_rejects_geometry_overflow() {
        let error = Tensor::from_raw(DataType::Float32, vec![usize::MAX, 2], &[])
            .expect_err("overflowing tensor geometry must be rejected");
        assert!(matches!(error, SessionError::ShapeOverflow { .. }));
    }

    #[test]
    fn try_clone_deep_copies_shape_and_data() {
        let tensor = Tensor::from_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let cloned = tensor.try_clone().unwrap();

        assert_eq!(cloned.dtype, tensor.dtype);
        assert_eq!(cloned.shape, tensor.shape);
        assert_eq!(cloned.layout, tensor.layout);
        assert_eq!(cloned.as_bytes(), tensor.as_bytes());
        assert_ne!(cloned.device_ptr(), tensor.device_ptr());
    }

    #[test]
    fn device_binding_allocation_rejects_byte_overflow() {
        let element_count = usize::MAX / 4;
        let error = DeviceIoBinding::allocate(
            shared_cpu_ep(),
            DeviceBindingSpec {
                input_name: "huge".into(),
                bind_input: true,
                output_name: None,
                dtype: DataType::Float64,
                physical_shape: vec![element_count],
                logical_shape: vec![element_count],
                expose_logical_input_shape: false,
                decode_freeze_safe_mask: false,
                allocation_bytes: None,
                committed_ranges: None,
            },
        )
        .expect_err("overflowing device binding byte count must be rejected");
        assert!(matches!(error, SessionError::ShapeOverflow { .. }));
    }

    #[test]
    fn borrowed_guard_ctor_runs_guard_exactly_once_on_drop() {
        let drops = Arc::new(AtomicUsize::new(0));
        // Some real host memory the borrowed buffer can alias.
        let mut backing = [1.0f32, 2.0, 3.0, 4.0];
        let ptr = backing.as_mut_ptr() as *mut c_void;
        // SAFETY: `backing` outlives the tensor built below; 16 bytes, 4-aligned.
        let buffer = unsafe {
            DeviceBuffer::from_borrowed_parts(ptr, DeviceId::cpu(), backing.len() * 4, 4)
        };
        assert!(buffer.is_borrowed());

        let guard = Box::new(CountingGuard(drops.clone()));
        let tensor = Tensor::from_borrowed_parts_with_guard(
            shared_cpu_ep(),
            DataType::Float32,
            vec![4],
            TensorLayout::contiguous(),
            buffer,
            guard,
        );

        // The tensor aliases the backing store without copying it.
        assert_eq!(tensor.as_bytes().len(), 16);
        assert_eq!(tensor.try_as_slice_f32().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(tensor.to_vec_f32(), vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            drops.load(Ordering::SeqCst),
            0,
            "guard alive while tensor is"
        );

        drop(tensor);
        assert_eq!(
            drops.load(Ordering::SeqCst),
            1,
            "guard runs exactly once on drop"
        );
    }

    #[test]
    fn borrows_aligned_half_storage_as_raw_bits() {
        let bits = [0x3c00u16, 0x4000];
        let bytes = bits
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let tensor = Tensor::from_raw(DataType::Float16, vec![2], &bytes).unwrap();
        assert_eq!(tensor.try_as_slice_u16().unwrap(), bits);
    }

    /// `exposes_logical_input_shape` is a *static* property of the binding (fixed
    /// at allocation from the consumer-scoped capacity policy), whereas
    /// `has_dynamic_logical_input_shape` additionally requires the current logical
    /// shape to differ from physical. The single-token decode mask freeze relies
    /// on this distinction: a mask that exposes its logical prefix must NOT be
    /// frozen to physical capacity even while its logical shape still equals
    /// physical at construction — otherwise the padded width leaks into a
    /// non-capacity-aware consumer (GLM-5.2's indexer arithmetic).
    #[test]
    fn exposes_logical_is_static_while_dynamic_tracks_current_shape() {
        // A mask-like binding that exposes its logical prefix, allocated with
        // logical == physical (the state at decode construction time).
        let mut logical_mask = DeviceIoBinding::allocate(
            shared_cpu_ep(),
            DeviceBindingSpec {
                input_name: "attention_mask".into(),
                bind_input: true,
                output_name: None,
                dtype: DataType::Int64,
                physical_shape: vec![1, 4096],
                logical_shape: vec![1, 4096],
                expose_logical_input_shape: true,
                decode_freeze_safe_mask: false,
                allocation_bytes: None,
                committed_ranges: None,
            },
        )
        .unwrap();
        assert!(logical_mask.exposes_logical_input_shape());
        // Not yet dynamic: logical still equals physical.
        assert!(!logical_mask.has_dynamic_logical_input_shape());
        // Kernels observe the logical prefix, so once decode drives the mask to
        // the growing valid length it becomes dynamic (and forfeits capture).
        logical_mask.set_logical_shape(vec![1, 5]).unwrap();
        assert!(logical_mask.exposes_logical_input_shape());
        assert!(logical_mask.has_dynamic_logical_input_shape());
        assert_eq!(logical_mask.kernel_input_shape(), &[1, 5]);

        // A capacity-exposing binding (all consumers padded-safe) never exposes
        // its logical prefix: it stays frozen at physical capacity regardless of
        // the current logical shape, so kernels always see the padded width.
        let mut physical_mask = DeviceIoBinding::allocate(
            shared_cpu_ep(),
            DeviceBindingSpec {
                input_name: "attention_mask".into(),
                bind_input: true,
                output_name: None,
                dtype: DataType::Int64,
                physical_shape: vec![1, 4096],
                logical_shape: vec![1, 5],
                expose_logical_input_shape: false,
                decode_freeze_safe_mask: false,
                allocation_bytes: None,
                committed_ranges: None,
            },
        )
        .unwrap();
        assert!(!physical_mask.exposes_logical_input_shape());
        assert!(!physical_mask.has_dynamic_logical_input_shape());
        assert_eq!(physical_mask.kernel_input_shape(), &[1, 4096]);
        physical_mask.set_logical_shape(vec![1, 4096]).unwrap();
        assert!(!physical_mask.exposes_logical_input_shape());
    }

    /// A zeroed tensor is zero even when the allocation is not.
    ///
    /// `zeros` writes the bytes in place instead of copying a zeroed `Vec` in,
    /// which is cheaper and just as correct -- but it moves the guarantee from
    /// obvious to asserted. Freshly mapped pages are usually already zero, so a
    /// test that only allocated would pass whether or not the zeroing happened.
    /// This dirties the buffer first, through the same public surface.
    #[test]
    fn a_zeroed_tensor_is_zero_even_over_dirty_memory() {
        // Allocate, poison, drop: the allocator is very likely to hand the same
        // block back to the next request of the same size.
        let poison = Tensor::from_raw(DataType::Float32, vec![2, 8], &[0xABu8; 64])
            .expect("a poisoned tensor");
        drop(poison);

        let zeroed = Tensor::zeros(DataType::Float32, vec![2, 8]).expect("a zeroed tensor");
        let values = zeroed.to_vec_f32();
        assert_eq!(values.len(), 16);
        assert!(
            values.iter().all(|value| *value == 0.0),
            "zeros() returned {values:?}"
        );
    }

    /// A growable KV input starts with an empty sequence axis, so this shape is
    /// reached on a hybrid decoder's first step.
    #[test]
    fn a_zeroed_tensor_of_no_elements_is_valid() {
        let empty = Tensor::zeros(DataType::Float32, vec![1, 8, 0, 4]).expect("allocatable");
        assert_eq!(empty.shape, vec![1, 8, 0, 4]);
        assert!(empty.to_vec_f32().is_empty());
    }
}

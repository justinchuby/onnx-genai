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

use onnx_runtime_ep_api::{DeviceBuffer, ExecutionProvider};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ir::{DataType, DeviceId, TensorLayout};

use crate::error::{Result, SessionError};

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

/// An externally owned persistent device allocation bound to a graph input and
/// optionally aliased by a graph output.
pub struct DeviceIoBinding {
    input_name: String,
    bind_input: bool,
    output_name: Option<String>,
    pub dtype: DataType,
    physical_shape: Vec<usize>,
    logical_shape: Vec<usize>,
    buffer: Option<DeviceBuffer>,
    allocator: Arc<dyn ExecutionProvider>,
    transfer_stats: DeviceBindingTransferStats,
}

impl DeviceIoBinding {
    pub(crate) fn allocate(
        allocator: Arc<dyn ExecutionProvider>,
        input_name: String,
        bind_input: bool,
        output_name: Option<String>,
        dtype: DataType,
        physical_shape: Vec<usize>,
        logical_shape: Vec<usize>,
    ) -> Result<Self> {
        validate_logical_shape(&physical_shape, &logical_shape)?;
        let numel = physical_shape.iter().try_fold(1usize, |product, &dim| {
            product.checked_mul(dim).ok_or_else(|| {
                SessionError::Internal(format!(
                    "device binding '{input_name}' physical shape overflows: {physical_shape:?}"
                ))
            })
        })?;
        let bytes = dtype.storage_bytes(numel).max(1);
        let allocator_for_buffer = allocator.clone();
        let buffer = allocator_for_buffer.allocate(bytes, TensorLayout::contiguous().alignment)?;
        Ok(Self {
            input_name,
            bind_input,
            output_name,
            dtype,
            physical_shape,
            logical_shape,
            buffer: Some(buffer),
            allocator,
            transfer_stats: DeviceBindingTransferStats::default(),
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

    pub fn set_logical_shape(&mut self, shape: Vec<usize>) -> Result<()> {
        validate_logical_shape(&self.physical_shape, &shape)?;
        self.logical_shape = shape;
        Ok(())
    }

    pub fn device_ptr(&self) -> *const std::ffi::c_void {
        self.buffer().as_ptr()
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
        self.allocator.copy_to_host(self.buffer(), bytes)?;
        self.transfer_stats.host_download_calls += 1;
        self.transfer_stats.host_download_bytes += bytes.len() as u64;
        Ok(())
    }

    pub fn device_argmax_supported(&self) -> bool {
        self.allocator.device_argmax_supported()
    }

    pub fn device_argmax(&self, elements: usize, result: &mut DeviceIoBinding) -> Result<()> {
        if self.dtype != DataType::Float32 || result.dtype != DataType::Uint32 {
            return Err(SessionError::Internal(format!(
                "device argmax requires f32 logits and u32 result, got {:?} and {:?}",
                self.dtype, result.dtype
            )));
        }
        if !Arc::ptr_eq(&self.allocator, &result.allocator) {
            return Err(SessionError::Internal(
                "device argmax bindings must belong to the same execution provider".into(),
            ));
        }
        Ok(self
            .allocator
            .device_argmax(self.buffer(), elements, result.buffer_mut())?)
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
            .field("device", &self.buffer().device())
            .field("device_ptr", &self.device_ptr())
            .field("transfer_stats", &self.transfer_stats)
            .finish()
    }
}

impl Drop for DeviceIoBinding {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            let _ = self.allocator.reset_device_graph();
            let _ = self.allocator.deallocate(buffer);
        }
    }
}

impl Tensor {
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
        let numel: usize = shape.iter().product();
        let expected = dtype.storage_bytes(numel);
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

    /// Copy out the elements as `f32`. Panics if the dtype is not `Float32`.
    pub fn to_vec_f32(&self) -> Vec<f32> {
        assert_eq!(
            self.dtype,
            DataType::Float32,
            "to_vec_f32 on non-f32 tensor"
        );
        self.as_bytes()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Copy out the elements as `i64`. Panics if the dtype is not `Int64`.
    pub fn to_vec_i64(&self) -> Vec<i64> {
        assert_eq!(self.dtype, DataType::Int64, "to_vec_i64 on non-i64 tensor");
        self.as_bytes()
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        // Deep copy: a fresh allocation with identical bytes. Cannot fail for
        // host allocations of the same size; propagate as a panic-free fallback
        // by re-using `from_raw_in` and unwrapping the size-checked path.
        Self::from_raw_in(
            self.allocator.clone(),
            self.dtype,
            self.shape.clone(),
            self.as_bytes(),
        )
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
}

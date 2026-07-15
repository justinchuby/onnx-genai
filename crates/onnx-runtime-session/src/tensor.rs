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
//! bytes is `unsafe` and sound only on host-accessible devices. Every such
//! access in this crate funnels through [`host_bytes`] / [`write_host`] /
//! [`copy_host`], which assert host accessibility and length, so the rest of
//! the crate — the executor and the public API — is safe Rust over the EP
//! contract.

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

/// Copy `src` into the front of a host-accessible device buffer.
pub(crate) fn write_host(buffer: &mut DeviceBuffer, src: &[u8]) -> Result<()> {
    assert!(
        buffer.device().is_host_accessible(),
        "write_host on non-host device {:?}",
        buffer.device()
    );
    if src.len() > buffer.len() {
        return Err(SessionError::Internal(format!(
            "write_host: source {} bytes exceeds buffer {} bytes",
            src.len(),
            buffer.len()
        )));
    }
    if src.is_empty() {
        return Ok(());
    }
    let dst = buffer.as_mut_ptr() as *mut u8;
    // SAFETY: host-accessible device (asserted); `dst` is a unique writable host
    // pointer obtained via `&mut buffer` (no alias), with at least `src.len()`
    // bytes of capacity (checked above). `src` is a distinct owned slice, so the
    // ranges do not overlap.
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
    }
    Ok(())
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

impl Tensor {
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
        write_host(&mut buffer, bytes)?;
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
                bytes.len(), self.shape, self.dtype
            )));
        }
        let buffer = self.buffer.as_mut().expect("Tensor buffer taken only in Drop");
        write_host(buffer, bytes)
    }

    /// Copy out the elements as `f32`. Panics if the dtype is not `Float32`.
    pub fn to_vec_f32(&self) -> Vec<f32> {
        assert_eq!(self.dtype, DataType::Float32, "to_vec_f32 on non-f32 tensor");
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
        assert_eq!(drops.load(Ordering::SeqCst), 0, "guard alive while tensor is");

        drop(tensor);
        assert_eq!(drops.load(Ordering::SeqCst), 1, "guard runs exactly once on drop");
    }
}

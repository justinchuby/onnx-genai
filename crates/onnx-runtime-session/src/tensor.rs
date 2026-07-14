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
    }
}

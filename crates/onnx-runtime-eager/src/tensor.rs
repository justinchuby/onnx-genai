//! The owned, device-aware [`Tensor`] the eager engine dispatches over
//! (`docs/EAGER.md` §3, §10.1).
//!
//! This mirrors the owned tensor in `onnx-runtime-session` (`src/tensor.rs`):
//! a thin owner over an [`onnx_runtime_ep_api::DeviceBuffer`] plus the IR
//! vocabulary ([`DataType`], [`TensorLayout`], shape). It is duplicated here
//! rather than shared because the eager crate must not depend on the session
//! crate (see `docs/EAGER.md` §12 crate layout). When the tensor type is hoisted
//! into a shared `onnx-runtime-tensor` crate (the design's open question), both
//! this crate and the session collapse onto it.
//!
//! ## The single `unsafe` seam
//!
//! A [`DeviceBuffer`] hands out only raw base pointers; reading/writing the
//! bytes is `unsafe` and sound only on host-accessible devices. Every such
//! access funnels through [`host_bytes`] / [`write_host`], which assert host
//! accessibility, so the rest of the crate is safe Rust over the EP contract.
//!
//! DLPack / numpy interop (`docs/EAGER.md` §3) is DEFERRED — those are binding
//! concerns handled by the (also DEFERRED) PyO3 layer (§11).

use std::ffi::c_void;
use std::sync::Arc;
use std::sync::OnceLock;

use onnx_runtime_ep_api::{DeviceBuffer, ExecutionProvider};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ir::{DataType, DeviceId, TensorLayout};

use crate::error::{EagerError, Result};

/// A process-wide, already-initialized CPU execution provider used to back
/// user-constructed [`Tensor`]s. Host `malloc`/`free` is global, so any
/// `CpuExecutionProvider` can free any other's CPU allocation (mirrors
/// `onnx-runtime-session`'s shared CPU EP).
fn shared_cpu_ep() -> Arc<dyn ExecutionProvider> {
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
/// # Panics
///
/// Panics if `buffer` is not on a host-accessible device.
fn host_bytes(buffer: &DeviceBuffer) -> &[u8] {
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
    // The lifetime is tied to `&buffer`, so the slice cannot dangle.
    unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const u8, buffer.len()) }
}

/// Copy `src` into the front of a host-accessible device buffer.
fn write_host(buffer: &mut DeviceBuffer, src: &[u8]) -> Result<()> {
    assert!(
        buffer.device().is_host_accessible(),
        "write_host on non-host device {:?}",
        buffer.device()
    );
    if src.len() > buffer.len() {
        return Err(EagerError::Kernel(onnx_runtime_ep_api::EpError::KernelFailed(
            format!(
                "write_host: source {} bytes exceeds buffer {} bytes",
                src.len(),
                buffer.len()
            ),
        )));
    }
    if src.is_empty() {
        return Ok(());
    }
    let dst = buffer.as_mut_ptr() as *mut u8;
    // SAFETY: host-accessible device (asserted); `dst` is a unique writable host
    // pointer obtained via `&mut buffer` (no alias), with at least `src.len()`
    // bytes of capacity (checked above). `src` is a distinct owned slice.
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
    }
    Ok(())
}

/// An owned, device-aware tensor (`docs/EAGER.md` §3).
///
/// Owns the [`DeviceBuffer`] that holds its elements and the EP that must free
/// it. On Phase-1 CPU the buffer is a host allocation, so [`Tensor::as_bytes`]
/// and the typed accessors read it directly.
pub struct Tensor {
    dtype: DataType,
    shape: Vec<usize>,
    layout: TensorLayout,
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
    pub fn from_raw_in(
        allocator: Arc<dyn ExecutionProvider>,
        dtype: DataType,
        shape: Vec<usize>,
        bytes: &[u8],
    ) -> Result<Self> {
        let numel: usize = shape.iter().product();
        let expected = dtype.storage_bytes(numel);
        if bytes.len() != expected {
            return Err(EagerError::Kernel(onnx_runtime_ep_api::EpError::KernelFailed(
                format!(
                    "Tensor::from_raw_in: {} bytes for shape {shape:?} dtype {dtype:?}, expected {expected}",
                    bytes.len()
                ),
            )));
        }
        let layout = TensorLayout::contiguous();
        let mut buffer = allocator.allocate(expected.max(1), layout.alignment)?;
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

    /// Allocate a zero-initialized tensor of `shape`/`dtype` using `allocator`.
    ///
    /// Used by dispatch to materialise output tensors before kernel execution
    /// (`docs/EAGER.md` §10.1 step 6).
    pub fn zeros_in(
        allocator: Arc<dyn ExecutionProvider>,
        dtype: DataType,
        shape: Vec<usize>,
    ) -> Result<Self> {
        let numel: usize = shape.iter().product();
        let bytes = vec![0u8; dtype.storage_bytes(numel)];
        Self::from_raw_in(allocator, dtype, shape, &bytes)
    }

    /// Build a tensor from raw little-endian bytes on the shared CPU device.
    pub fn from_raw(dtype: DataType, shape: Vec<usize>, bytes: &[u8]) -> Result<Self> {
        Self::from_raw_in(shared_cpu_ep(), dtype, shape, bytes)
    }

    /// Build a zero-initialized tensor on the shared CPU device.
    pub fn zeros(dtype: DataType, shape: Vec<usize>) -> Result<Self> {
        Self::zeros_in(shared_cpu_ep(), dtype, shape)
    }

    /// Build an `f32` tensor from a dense row-major slice on the CPU device.
    pub fn from_f32(shape: &[usize], data: &[f32]) -> Result<Self> {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Self::from_raw(DataType::Float32, shape.to_vec(), &bytes)
    }

    /// Build an `i64` tensor from a dense row-major slice on the CPU device.
    pub fn from_i64(shape: &[usize], data: &[i64]) -> Result<Self> {
        let mut bytes = Vec::with_capacity(data.len() * 8);
        for v in data {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        Self::from_raw(DataType::Int64, shape.to_vec(), &bytes)
    }

    /// The element type.
    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    /// The logical shape (static dims).
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The physical layout (row-major contiguous for tensors this crate produces).
    pub fn layout(&self) -> &TensorLayout {
        &self.layout
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

    /// Shared raw base pointer to the element storage. Safe to obtain;
    /// dereferencing is `unsafe` and sound only within the owning EP's context.
    pub(crate) fn device_ptr(&self) -> *const c_void {
        self.buffer().as_ptr()
    }

    /// Unique raw base pointer to the element storage.
    pub(crate) fn device_ptr_mut(&mut self) -> *mut c_void {
        self.buffer
            .as_mut()
            .expect("Tensor buffer taken only in Drop")
            .as_mut_ptr()
    }

    /// Borrow the raw little-endian element bytes (host tensors only).
    pub fn as_bytes(&self) -> &[u8] {
        let n = self.dtype.storage_bytes(self.numel());
        &host_bytes(self.buffer())[..n]
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
            // once (ep-api §4.4 invariant #2). A failed free leaks, never
            // double-frees, so we swallow the error.
            let _ = self.allocator.deallocate(buffer);
        }
    }
}

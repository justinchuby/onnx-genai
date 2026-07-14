//! Shared CUDA runtime state: the driver context, its default stream, and the
//! cuBLASLt handle. One [`CudaRuntime`] is created per [`CudaExecutionProvider`]
//! and shared (via `Arc`) into every kernel the provider hands out, so the whole
//! EP drives a single device + stream + library handle.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaContext, CudaStream};

use onnx_runtime_ep_api::Result;

use crate::blas::CublasLt;
use crate::error::driver_err;

/// Device context + stream + cuBLASLt handle shared across the EP's kernels.
pub struct CudaRuntime {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: CublasLt,
    ordinal: u32,
}

impl std::fmt::Debug for CudaRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaRuntime")
            .field("ordinal", &self.ordinal)
            .finish()
    }
}

impl CudaRuntime {
    /// Initialise the primary context on CUDA device `ordinal`, its default
    /// stream, and a cuBLASLt handle. Returns an error (never panics) when no
    /// such device exists or the CUDA driver / cuBLASLt cannot be loaded.
    pub fn new(ordinal: u32) -> Result<Self> {
        let context =
            CudaContext::new(ordinal as usize).map_err(|e| driver_err("CudaContext::new", e))?;
        let stream = context.default_stream();
        let blas = CublasLt::new()?;
        Ok(Self {
            context,
            stream,
            blas,
            ordinal,
        })
    }

    /// The CUDA device ordinal this runtime drives.
    pub fn ordinal(&self) -> u32 {
        self.ordinal
    }

    /// The cuBLASLt handle.
    pub fn blas(&self) -> &CublasLt {
        &self.blas
    }

    /// The raw CUDA stream the EP submits work on.
    pub fn stream_ptr(&self) -> cudarc::driver::sys::CUstream {
        self.stream.cu_stream()
    }

    /// Bind this runtime's context to the calling thread. Required before any
    /// driver call (`malloc`, `memcpy`, cuBLASLt) that targets the current
    /// context.
    pub fn bind(&self) -> Result<()> {
        self.context
            .bind_to_thread()
            .map_err(|e| driver_err("bind_to_thread", e))
    }

    /// Block until all submitted work on the default stream completes.
    pub fn synchronize(&self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| driver_err("stream synchronize", e))
    }

    /// Allocate `bytes` (>= 1) of device memory, returning the raw device
    /// pointer. Binds the context first.
    pub fn alloc_raw(&self, bytes: usize) -> Result<CUdeviceptr> {
        self.bind()?;
        // SAFETY: `malloc_sync` returns a fresh device allocation on the current
        // (bound) context; we own it and free it exactly once via `free_raw`.
        unsafe { cudarc::driver::result::malloc_sync(bytes.max(1)) }
            .map_err(|e| driver_err("cuMemAlloc", e))
    }

    /// Free a device pointer previously returned by [`CudaRuntime::alloc_raw`].
    ///
    /// # Safety
    /// `ptr` must have come from this runtime's `alloc_raw` and not been freed.
    pub unsafe fn free_raw(&self, ptr: CUdeviceptr) -> Result<()> {
        self.bind()?;
        // SAFETY: caller upholds the single-free contract.
        unsafe { cudarc::driver::result::free_sync(ptr) }.map_err(|e| driver_err("cuMemFree", e))
    }

    /// Copy `bytes` host → device (H2D). `dst` must be large enough.
    ///
    /// # Safety
    /// `dst` is a live device allocation of at least `src.len()` bytes.
    pub unsafe fn htod(&self, src: &[u8], dst: CUdeviceptr) -> Result<()> {
        self.bind()?;
        // SAFETY: bound context; `dst` covers `src.len()` bytes per the contract.
        unsafe { cudarc::driver::result::memcpy_htod_sync(dst, src) }
            .map_err(|e| driver_err("cuMemcpyHtoD", e))
    }

    /// Copy `dst.len()` bytes device → host (D2H). `src` must be large enough.
    ///
    /// # Safety
    /// `src` is a live device allocation of at least `dst.len()` bytes.
    pub unsafe fn dtoh(&self, dst: &mut [u8], src: CUdeviceptr) -> Result<()> {
        self.bind()?;
        // SAFETY: bound context; `src` covers `dst.len()` bytes per the contract.
        unsafe { cudarc::driver::result::memcpy_dtoh_sync(dst, src) }
            .map_err(|e| driver_err("cuMemcpyDtoH", e))?;
        self.synchronize()
    }

    /// Copy `bytes` device → device (D2D).
    ///
    /// # Safety
    /// Both pointers are live allocations of at least `bytes` bytes.
    pub unsafe fn dtod(&self, src: CUdeviceptr, dst: CUdeviceptr, bytes: usize) -> Result<()> {
        self.bind()?;
        // SAFETY: bound context; both endpoints cover `bytes` per the contract.
        unsafe { cudarc::driver::result::memcpy_dtod_sync(dst, src, bytes) }
            .map_err(|e| driver_err("cuMemcpyDtoD", e))
    }
}

/// Reinterpret an EP [`onnx_runtime_ep_api::DeviceBuffer`] raw pointer (or a
/// [`onnx_runtime_ep_api::TensorView`] data pointer) as a CUDA device pointer.
/// CUDA device pointers are integer addresses; the EP stores them in the opaque
/// pointer slot, so this is a value reinterpretation, never a host deref.
#[inline]
pub fn cuptr(raw: *const c_void) -> CUdeviceptr {
    raw as usize as CUdeviceptr
}

/// Inverse of [`cuptr`]: pack a CUDA device pointer into the opaque pointer slot
/// used by [`onnx_runtime_ep_api::DeviceBuffer`].
#[inline]
pub fn raw_ptr(dptr: CUdeviceptr) -> *mut c_void {
    dptr as usize as *mut c_void
}

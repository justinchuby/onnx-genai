//! CUDA device memory, through the allocator seam both backends share.
//!
//! # Why this exists
//!
//! The CPU execution provider and ONNX Runtime's governed allocator already go
//! through [`DeviceAllocator`]. This is the CUDA end of the same contract, so
//! an allocator a caller writes serves every backend rather than one per
//! backend.
//!
//! # Why device memory is the case that will need an arena
//!
//! Host memory does not: measured, an arena over the system allocator was
//! *slower* than going straight to it, because `malloc` already pools with
//! per-thread caches and anything layered on top adds a lock.
//!
//! `cudaMalloc` is the opposite. It is a synchronising driver call in the
//! microseconds with no thread cache — three orders of magnitude worse than
//! host `malloc`. That is why ONNX Runtime ships a BFC arena for CUDA and not
//! for CPU, and it is why the seam matters here: an arena belongs *behind* this
//! trait, written once, rather than inside each backend.
//!
//! This implementation is deliberately the thin one. It makes CUDA memory
//! reachable through the contract; the arena is a separate implementation of
//! the same trait and a separate change.

use std::ptr::NonNull;
use std::sync::Arc;

use onnx_runtime_memory_governor::{DeviceAllocator, DeviceKey, MemoryError, Tier};

use cudarc::driver::CudaContext;

/// Device memory from `cuMemAlloc`, on one CUDA device.
#[derive(Debug, Clone)]
pub struct CudaDeviceAllocator {
    context: Arc<CudaContext>,
    device: DeviceKey,
}

impl CudaDeviceAllocator {
    /// Allocate in `context`, on device `ordinal`.
    ///
    /// Takes a context rather than the execution provider's full runtime for
    /// the same reason `CudaVirtualBacking` does: `cuMemAlloc` and `cuMemFree`
    /// are **driver** API, needing no cudart, no cuBLAS and no kernels.
    /// Requiring the runtime would couple this to libraries it does not use,
    /// and on a machine with only the driver that coupling is the difference
    /// between the tests running and silently skipping.
    pub fn new(context: Arc<CudaContext>, ordinal: u32) -> Self {
        Self {
            context,
            device: DeviceKey::device(ordinal),
        }
    }

    fn bind(&self, what: &'static str, bytes: usize) -> Result<(), MemoryError> {
        self.context
            .bind_to_thread()
            .map_err(|_| MemoryError::InvalidRequest {
                tier: Tier::Device.name(),
                requested: bytes as u64,
                reason: what,
            })
    }
}

// Every pointer comes from this runtime's lloc_raw and is freed exactly once
// through ree_raw; device names the CUDA device the runtime is bound to, so
// nothing dereferences these addresses on the host.
impl DeviceAllocator for CudaDeviceAllocator {
    fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
        if align == 0 || !align.is_power_of_two() {
            return Err(MemoryError::InvalidRequest {
                tier: Tier::Device.name(),
                requested: bytes as u64,
                reason: "the alignment must be a power of two",
            });
        }
        // `cuMemAlloc` returns at least 256-byte-aligned device pointers, which
        // covers any realistic tensor alignment. A larger request would need
        // over-allocating and adjusting, which is not needed and would make the
        // size passed to `deallocate` disagree with the size allocated.
        if align > 256 {
            return Err(MemoryError::InvalidRequest {
                tier: Tier::Device.name(),
                requested: bytes as u64,
                reason: "CUDA guarantees 256-byte alignment and this allocator does not \
                         over-allocate to exceed it; request 256 bytes or less of alignment",
            });
        }
        self.bind("could not bind the CUDA context before allocating", bytes)?;
        // SAFETY: `malloc_sync` returns a fresh device allocation on the bound
        // context; this allocator owns it and frees it exactly once.
        let dptr = unsafe { cudarc::driver::result::malloc_sync(bytes.max(1)) }.map_err(|_| {
            MemoryError::InvalidRequest {
                tier: Tier::Device.name(),
                requested: bytes as u64,
                reason: "the CUDA driver refused the allocation; the device is out of memory",
            }
        })?;
        NonNull::new(dptr as *mut u8).ok_or(MemoryError::InvalidRequest {
            tier: Tier::Device.name(),
            requested: bytes as u64,
            reason: "the CUDA driver returned a null device pointer",
        })
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, _align: usize) {
        if self.bind("binding to free", bytes).is_err() {
            // Leaking beats freeing on the wrong context, which would either
            // fail or free someone else's allocation.
            return;
        }
        // SAFETY: delegated to this method's contract -- the pointer came from
        // `allocate` on this allocator and is freed once.
        unsafe {
            let _ =
                cudarc::driver::result::free_sync(ptr.as_ptr() as cudarc::driver::sys::CUdeviceptr);
        }
    }

    fn device(&self) -> DeviceKey {
        self.device
    }
}

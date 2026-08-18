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
//!
//! # Why an arena here is a correctness fix and not only a speed one
//!
//! A driver allocation made by a thread that is *currently capturing a CUDA
//! graph* invalidates that capture. An arena that can satisfy a request without
//! entering the driver makes allocation during capture legal, which the plain
//! `cuMemAlloc` below cannot.

use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use onnx_runtime_memory_governor::{
    AllocationReleaseOutcome, AllocationReleaseState, DeviceAllocator, DeviceKey, MemoryError,
    QuarantineReason, ReleaseAccounting, ResidualOwnership, Tier,
};

use cudarc::driver::CudaContext;

/// One `cuMemAlloc` allocation this allocator could not give back.
///
/// The exact address, size, and context are kept so the loss is reconcilable
/// against provider-side records rather than showing up later as VRAM that
/// simply went missing. Nothing is ever taken out of the quarantine: an
/// allocation whose free failed has genuinely uncertain ownership and must
/// never be handed out again.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuarantinedCudaAllocation {
    /// Device address. Never dereferenced on the host.
    pub address: usize,
    pub bytes: usize,
    pub align: usize,
    /// CUDA device ordinal of the context the allocation belongs to.
    pub device_ordinal: u32,
    pub reason: String,
}

/// Device memory from `cuMemAlloc`, on one CUDA device.
#[derive(Debug)]
pub struct CudaDeviceAllocator {
    context: Arc<CudaContext>,
    device: DeviceKey,
    /// Frees this allocator could not perform, so the bytes are still held.
    ///
    /// `DeviceAllocator::deallocate` returns `()`, so a failed free has nowhere
    /// to be reported. Unreported it shows up only as VRAM that went missing
    /// and an out-of-memory error much later, with nothing connecting the two.
    leaked_frees: AtomicU64,
    leaked_bytes: AtomicU64,
    /// Successful `cuMemAlloc` calls this allocator has made.
    ///
    /// This is the number the capture-safety story turns on: a `cuMemAlloc`
    /// per decode dispatch is exactly the residual issue #956 tracks, and it is
    /// illegal during CUDA-graph capture. Counting it here — rather than
    /// inferring it from the *absence* of an out-of-memory or a capture error —
    /// lets a test show directly that this call site scales with decode steps
    /// on the default path and is bypassed entirely once the VMM arena serves
    /// the same requests (`measurement-discipline`: measure the thing, do not
    /// infer it from a missing symptom).
    cumemalloc_calls: AtomicU64,
    /// Allocations whose release failed, retained exactly.
    quarantine: Mutex<Vec<QuarantinedCudaAllocation>>,
}

impl CudaDeviceAllocator {
    /// Allocate in `context`.
    ///
    /// Takes a context rather than the execution provider's full runtime for
    /// the same reason `CudaVirtualBacking` does: `cuMemAlloc` and `cuMemFree`
    /// are **driver** API, needing no cudart, no cuBLAS and no kernels.
    /// Requiring the runtime would couple this to libraries it does not use,
    /// and on a machine with only the driver that coupling is the difference
    /// between the tests running and silently skipping.
    ///
    /// The device it reports is read from the context rather than passed in.
    /// Callers use [`DeviceAllocator::device`] to decide whether a pointer may
    /// be dereferenced on the host and which device it belongs to, so an
    /// ordinal that disagreed with the context the pointers actually come from
    /// would be a lie no caller could detect.
    pub fn new(context: Arc<CudaContext>) -> Self {
        let ordinal = context.ordinal() as u32;
        Self {
            context,
            device: DeviceKey::device(ordinal),
            leaked_frees: AtomicU64::new(0),
            leaked_bytes: AtomicU64::new(0),
            cumemalloc_calls: AtomicU64::new(0),
            quarantine: Mutex::new(Vec::new()),
        }
    }

    /// Allocations this allocator retained because their release failed.
    ///
    /// Non-empty means VRAM is held with uncertain ownership. It is observable
    /// here, where it happened, with the exact pointer/bytes/context metadata
    /// needed to reconcile it — rather than being inferred from an unrelated
    /// out-of-memory error much later.
    pub fn quarantined(&self) -> Vec<QuarantinedCudaAllocation> {
        self.quarantine
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// Record retained ownership and report it as a structured quarantine.
    fn quarantine(
        &self,
        ptr: NonNull<u8>,
        bytes: usize,
        align: usize,
        reason: String,
    ) -> AllocationReleaseOutcome {
        self.leaked_frees.fetch_add(1, Ordering::Relaxed);
        self.leaked_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
        let address = ptr.as_ptr() as usize;
        self.quarantine
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(QuarantinedCudaAllocation {
                address,
                bytes,
                align,
                device_ordinal: self.device.index,
                reason: reason.clone(),
            });
        eprintln!(
            "cuda_ep: WARNING: could not free a {bytes} byte CUDA allocation at {address:#x} on \
             device {}: {reason}; its ownership is retained and it will never be reused",
            self.device.index
        );
        AllocationReleaseOutcome::quarantined(
            ReleaseAccounting::eager(bytes as u64),
            ResidualOwnership {
                state: AllocationReleaseState::Quarantined,
                reason: QuarantineReason::PartialRelease,
                retained_bytes: bytes as u64,
                address,
                align,
            },
        )
    }

    /// Successful `cuMemAlloc` calls this allocator has made since construction.
    ///
    /// Exposed so a test can prove — directly, not by inference — that this
    /// call site scales one-for-one with allocation requests, which is the
    /// per-dispatch driver allocation issue #956 removes by routing device
    /// memory through the VMM arena instead.
    pub fn cumemalloc_calls(&self) -> u64 {
        self.cumemalloc_calls.load(Ordering::Relaxed)
    }

    /// How many frees this allocator could not perform, and how many bytes they
    /// held.
    ///
    /// Non-zero means VRAM is being lost. Exposed so the loss is observable
    /// where it happens rather than inferred from an unrelated failure later.
    pub fn leaked(&self) -> (u64, u64) {
        (
            self.leaked_frees.load(Ordering::Relaxed),
            self.leaked_bytes.load(Ordering::Relaxed),
        )
    }

    fn bind(&self, bytes: usize) -> Result<(), MemoryError> {
        self.context
            .bind_to_thread()
            .map_err(|error| MemoryError::AllocationFailed {
                tier: Tier::Device.name(),
                requested: bytes as u64,
                reason: format!(
                    "could not bind the CUDA context for device {}: {error}",
                    self.device.index
                ),
            })
    }
}

// Every pointer comes from `cuMemAlloc` on this allocator's context and is
// freed exactly once through `cuMemFree`; `device` names the CUDA device that
// context belongs to, so nothing dereferences these addresses on the host.
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
        self.bind(bytes)?;
        // A zero-byte request is normalised here rather than by the caller, so
        // that the caller passes the same `bytes` to `allocate` and to
        // `deallocate` -- which is exactly what the contract lets an
        // implementation rely on. `cuMemAlloc(0)` fails, and a null pointer is
        // not a valid allocation.
        let request = bytes.max(1);
        // SAFETY: `malloc_sync` returns a fresh device allocation on the bound
        // context; this allocator owns it and frees it exactly once.
        let dptr = unsafe { cudarc::driver::result::malloc_sync(request) }.map_err(|error| {
            // Not necessarily out of memory: an uninitialised driver, a torn
            // down context and an invalid argument all land here too. Report
            // what the driver said rather than guessing, because "out of
            // memory" sends the next reader looking at the wrong thing.
            MemoryError::AllocationFailed {
                tier: Tier::Device.name(),
                requested: bytes as u64,
                reason: format!("cuMemAlloc refused {request} bytes: {error}"),
            }
        })?;
        NonNull::new(dptr as *mut u8)
            .ok_or(MemoryError::AllocationFailed {
                tier: Tier::Device.name(),
                requested: bytes as u64,
                reason: String::from("cuMemAlloc returned a null device pointer"),
            })
            .inspect(|_| {
                self.cumemalloc_calls.fetch_add(1, Ordering::Relaxed);
            })
    }

    /// Documented migration adapter.
    ///
    /// It cannot report partial failure, so it delegates to
    /// [`release`](Self::release) and discards the structured outcome; the
    /// retained ownership is still recorded in the allocator's own quarantine
    /// and is readable through [`CudaDeviceAllocator::quarantined`].
    unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) {
        // SAFETY: forwarded under this method's identical contract.
        let _ = unsafe { self.release(ptr, bytes, align) };
    }

    /// Whole-allocation release with an honest terminal state.
    ///
    /// A failure to bind the context or a refused `cuMemFree` is **not**
    /// reported as "nothing happened": the bytes are still owned, so the
    /// outcome is a quarantine that keeps the exact address, size, alignment,
    /// and context. Leaking beats freeing on the wrong context, which would
    /// either fail or free someone else's allocation.
    unsafe fn release(
        &self,
        ptr: NonNull<u8>,
        bytes: usize,
        align: usize,
    ) -> AllocationReleaseOutcome {
        if let Err(error) = self.bind(bytes) {
            return self.quarantine(
                ptr,
                bytes,
                align,
                format!("the CUDA context could not be bound for the free: {error}"),
            );
        }
        // SAFETY: delegated to this method's contract -- the pointer came from
        // `allocate` on this allocator and is freed once.
        let freed = unsafe {
            cudarc::driver::result::free_sync(ptr.as_ptr() as cudarc::driver::sys::CUdeviceptr)
        };
        match freed {
            Ok(()) => AllocationReleaseOutcome::complete(ReleaseAccounting::eager(bytes as u64)),
            Err(error) => self.quarantine(ptr, bytes, align, format!("cuMemFree refused: {error}")),
        }
    }

    fn device(&self) -> DeviceKey {
        self.device
    }
}

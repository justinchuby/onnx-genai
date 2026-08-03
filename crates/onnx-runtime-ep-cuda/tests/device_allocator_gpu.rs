//! The CUDA execution provider allocates through the shared `DeviceAllocator`
//! contract.
//!
//! The CPU EP and ONNX Runtime's governed allocator already do. This is the
//! third backend on the same seam, and the claim is that an allocator a caller
//! writes serves all of them rather than one each.
//!
//! Device memory is also the case that will actually need an arena —
//! `cudaMalloc` is a synchronising driver call in the microseconds, where host
//! `malloc` already pools with per-thread caches — so this is the seam an arena
//! will sit behind. That makes "can a caller really substitute here" worth
//! pinning now rather than after there is something to substitute.
//!
//! Needs a real GPU. Skips loudly when there is none: a skip that reads like a
//! pass is worse than a failure, because nobody investigates it.

use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

use cudarc::driver::CudaContext;
use onnx_runtime_ep_cuda::device_allocator::CudaDeviceAllocator;
use onnx_runtime_memory_governor::{DeviceAllocator, DeviceKey, MemoryError, Tier};

/// A CUDA device allocator, or `None` on a machine with no driver.
///
/// Uses the **driver** API only, which ships with the display driver. The
/// execution provider additionally gates on cudart and cuBLAS, so constructing
/// one here would make these tests skip on a machine that can run them
/// perfectly well. Nothing on this path calls either: allocation is
/// `cuMemAlloc`/`cuMemFree`, both driver entry points.
fn allocator() -> Option<CudaDeviceAllocator> {
    match CudaContext::new(0) {
        Ok(context) => Some(CudaDeviceAllocator::new(context, 0)),
        Err(error) => {
            eprintln!(
                "SKIPPED (no CUDA driver): {error}. This test verifies device allocation \
                 through the shared contract and did NOT run."
            );
            None
        }
    }
}
/// Counts what passes through it, so a test can tell "used" from "ignored".
#[derive(Debug)]
struct CountingAllocator {
    inner: CudaDeviceAllocator,
    allocations: AtomicU64,
    deallocations: AtomicU64,
    live_bytes: AtomicU64,
}

impl DeviceAllocator for CountingAllocator {
    fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
        let ptr = self.inner.allocate(bytes, align)?;
        self.allocations.fetch_add(1, Ordering::Relaxed);
        self.live_bytes.fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(ptr)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) {
        // SAFETY: forwarded unchanged from this method's own contract.
        unsafe { self.inner.deallocate(ptr, bytes, align) };
        self.deallocations.fetch_add(1, Ordering::Relaxed);
        self.live_bytes.fetch_sub(bytes as u64, Ordering::Relaxed);
    }

    fn device(&self) -> DeviceKey {
        self.inner.device()
    }
}

/// Device memory really comes from this allocator, and really works.
///
/// Counting alone proves nothing about the memory, so the buffer is written
/// and read back through the driver: a counted no-op would pass the counter
/// assertions and fail this one.
#[test]
fn allocated_device_memory_round_trips_and_is_returned() {
    let Some(inner) = allocator() else { return };
    let counters = CountingAllocator {
        inner,
        allocations: AtomicU64::new(0),
        deallocations: AtomicU64::new(0),
        live_bytes: AtomicU64::new(0),
    };

    let bytes = 1 << 20;
    let ptr = counters.allocate(bytes, 256).expect("granted");
    assert_eq!(counters.allocations.load(Ordering::Relaxed), 1);
    assert_eq!(counters.live_bytes.load(Ordering::Relaxed), bytes as u64);

    let pattern: Vec<u8> = (0..bytes).map(|index| (index % 251) as u8).collect();
    let mut read_back = vec![0u8; bytes];
    unsafe {
        use cudarc::driver::sys as cu;
        let address = ptr.as_ptr() as cu::CUdeviceptr;
        assert_eq!(
            cu::cuMemcpyHtoD_v2(address, pattern.as_ptr().cast(), bytes),
            cu::CUresult::CUDA_SUCCESS,
            "host-to-device copy into the allocated buffer"
        );
        assert_eq!(
            cu::cuMemcpyDtoH_v2(read_back.as_mut_ptr().cast(), address, bytes),
            cu::CUresult::CUDA_SUCCESS,
            "device-to-host copy back"
        );
    }
    assert_eq!(read_back, pattern, "the device memory did not round-trip");

    // SAFETY: exactly what `allocate` returned.
    unsafe { counters.deallocate(ptr, bytes, 256) };
    assert_eq!(counters.deallocations.load(Ordering::Relaxed), 1);
    assert_eq!(
        counters.live_bytes.load(Ordering::Relaxed),
        0,
        "the allocator must see its own bytes balance"
    );
}

/// The allocator reports the device it serves, and it is a device tier.
///
/// Callers decide from this whether a pointer may be dereferenced on the host,
/// so an allocator claiming the host tier for CUDA memory would turn a read
/// into a wild access rather than an error.
#[test]
fn the_cuda_allocator_reports_a_device_tier() {
    let Some(allocator) = allocator() else { return };
    let key = allocator.device();
    assert_eq!(key.tier, Tier::Device, "CUDA memory is not host memory");
    assert_eq!(key.index, 0);
}

/// An alignment CUDA does not guarantee is refused rather than silently
/// under-delivered.
///
/// `cuMemAlloc` guarantees 256 bytes. Returning a 256-aligned pointer for a
/// 4096-aligned request would fault only in the vector kernels that need it,
/// which is the worst possible way to find out.
#[test]
fn an_alignment_cuda_does_not_guarantee_is_refused() {
    let Some(allocator) = allocator() else { return };
    let error = allocator
        .allocate(4096, 4096)
        .expect_err("4096-byte alignment is beyond what cuMemAlloc guarantees");
    assert!(
        error.to_string().contains("256"),
        "the error must say what is guaranteed, got: {error}"
    );
}

/// Concurrent `allocate` calls return regions that do not overlap.
///
/// One allocator serves every session on a device, so several sessions
/// allocating at once is the normal case, not an edge one. The trait takes
/// `&self` and is bound `Send + Sync` to say exactly this, and a device
/// allocator that handed two threads overlapping memory would corrupt one
/// session's tensors with another's writes -- silently, and only under load.
///
/// Disjointness is the property worth asserting: it is what a caller relies on
/// and what a wrong lock would break.
#[test]
fn concurrent_allocations_do_not_overlap() {
    let Some(allocator) = allocator() else { return };
    let allocator = std::sync::Arc::new(allocator);

    const THREADS: usize = 8;
    const PER_THREAD: usize = 16;
    const BYTES: usize = 4096;

    let handles: Vec<_> = (0..THREADS)
        .map(|_| {
            let allocator = std::sync::Arc::clone(&allocator);
            std::thread::spawn(move || {
                (0..PER_THREAD)
                    .map(|_| {
                        allocator
                            .allocate(BYTES, 256)
                            .expect("a concurrent device allocation")
                            .as_ptr() as usize
                    })
                    .collect::<Vec<_>>()
            })
        })
        .collect();

    let mut regions: Vec<usize> = handles
        .into_iter()
        .flat_map(|handle| handle.join().expect("an allocating thread panicked"))
        .collect();

    assert_eq!(regions.len(), THREADS * PER_THREAD);
    regions.sort_unstable();
    for pair in regions.windows(2) {
        assert!(
            pair[0] + BYTES <= pair[1],
            "two concurrent allocations overlap: {:#x} + {BYTES} runs into {:#x}",
            pair[0],
            pair[1]
        );
    }

    for address in regions {
        let ptr = NonNull::new(address as *mut u8).expect("a recorded allocation");
        // SAFETY: each address came from `allocate` above with this size and
        // alignment, and each appears once, so this is its single free.
        unsafe { allocator.deallocate(ptr, BYTES, 256) };
    }
}

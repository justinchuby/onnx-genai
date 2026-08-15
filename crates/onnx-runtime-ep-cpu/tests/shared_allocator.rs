//! One allocator, both backends.
//!
//! The claim the `DeviceAllocator` seam exists to make is that a caller writes
//! *one* allocator and it serves the native execution provider and the ONNX
//! Runtime allocator alike. This test covers the native half from outside the
//! crate — the same view a third party has.
//!
//! The ORT half is `a_caller_supplied_allocator_backs_ort_allocations` in
//! `onnx-genai-ort`. Neither crate can host both, because neither depends on
//! the other; that is exactly why the contract lives in a third crate.

use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use onnx_runtime_ep_api::ExecutionProvider;
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_memory_governor::{DeviceAllocator, DeviceKey, HostAllocator, MemoryError};

/// Counts what passes through it, so a test can tell "used" from "ignored".
#[derive(Debug, Default)]
struct CountingAllocator {
    inner: HostAllocator,
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
        DeviceKey::HOST
    }
}

/// The EP's buffers come from the supplied allocator, and go back to it.
///
/// Constructing the EP proves nothing: an implementation that ignored
/// `with_memory` and kept calling `std::alloc` would pass any test that only
/// checks the memory works. So the counters assert it was *called*, and the
/// buffer is written to end-to-end to prove the memory is real.
#[test]
fn the_execution_provider_allocates_through_a_caller_supplied_allocator() {
    let counters = Arc::new(CountingAllocator::default());
    let ep =
        CpuExecutionProvider::new().with_memory(Arc::clone(&counters) as Arc<dyn DeviceAllocator>);

    let mut buffer = ep.allocate(4096, 64).expect("granted");
    assert_eq!(
        counters.allocations.load(Ordering::Relaxed),
        1,
        "the EP must allocate through the supplied allocator, not the system one"
    );
    assert_eq!(counters.live_bytes.load(Ordering::Relaxed), 4096);
    assert_eq!(
        buffer.as_mut_ptr() as usize % 64,
        0,
        "the alignment the EP asked for must survive the indirection"
    );

    // SAFETY: a live host allocation of 4096 bytes.
    unsafe { std::ptr::write_bytes(buffer.as_mut_ptr().cast::<u8>(), 0x7E, 4096) };

    ep.deallocate(buffer).expect("returned");
    assert_eq!(
        counters.deallocations.load(Ordering::Relaxed),
        1,
        "every buffer must be returned to the allocator that produced it"
    );
    assert_eq!(
        counters.live_bytes.load(Ordering::Relaxed),
        0,
        "the supplied allocator must see its own bytes balance"
    );
}

/// A borrowed buffer must not reach the allocator's `deallocate`.
///
/// Borrowed buffers alias memory someone else owns — an mmap'd weight file, or
/// a caller's own array. Passing one to the allocator would free memory it
/// never produced.
#[test]
fn a_borrowed_buffer_is_not_returned_to_the_allocator() {
    use onnx_runtime_ep_api::DeviceBuffer;
    use onnx_runtime_ir::DeviceId;

    let counters = Arc::new(CountingAllocator::default());
    let ep =
        CpuExecutionProvider::new().with_memory(Arc::clone(&counters) as Arc<dyn DeviceAllocator>);

    let mut backing = vec![0u8; 256];
    // SAFETY: `backing` outlives the buffer and nothing else writes it.
    let borrowed = unsafe {
        DeviceBuffer::from_borrowed_mut_parts(backing.as_mut_ptr().cast(), DeviceId::cpu(), 256, 64)
    };
    let Some(borrowed) = borrowed else {
        return; // the test vector was not suitably aligned on this platform
    };

    ep.deallocate(borrowed)
        .expect("a borrowed buffer is a no-op");
    assert_eq!(
        counters.deallocations.load(Ordering::Relaxed),
        0,
        "a borrowed buffer must never reach the allocator that did not produce it"
    );
    // `backing` is still valid.
    backing[0] = 1;
    assert_eq!(backing[0], 1);
}

/// The default is unchanged for callers who supply nothing.
#[test]
fn the_default_execution_provider_still_allocates() {
    let ep = CpuExecutionProvider::new();
    let buffer = ep.allocate(128, 64).expect("granted");
    ep.deallocate(buffer).expect("returned");
}

/// A refusal keeps the allocator's own account of why.
///
/// Every failure used to be reported as `OutOfMemory { available: 0 }`. A
/// substituted allocator that refuses for a reason of its own -- a budget, an
/// alignment it will not serve, a device it does not own -- was described to the
/// caller as exhausted RAM, which sends whoever reads the log looking in the
/// wrong place. That matters most for exactly the allocators this seam exists
/// to admit, because their reasons are ones this crate has never heard of.
#[test]
fn a_refusal_from_the_supplied_allocator_keeps_its_reason() {
    #[derive(Debug)]
    struct AlwaysRefuses;

    impl DeviceAllocator for AlwaysRefuses {
        fn allocate(&self, bytes: usize, _align: usize) -> Result<NonNull<u8>, MemoryError> {
            Err(MemoryError::AllocationFailed {
                tier: "host",
                requested: bytes as u64,
                reason: String::from("the tenant quota for this model is already spent"),
            })
        }

        unsafe fn deallocate(&self, _ptr: NonNull<u8>, _bytes: usize, _align: usize) {
            unreachable!("nothing was ever allocated");
        }

        fn device(&self) -> DeviceKey {
            DeviceKey::HOST
        }
    }

    let ep = CpuExecutionProvider::new().with_memory(Arc::new(AlwaysRefuses));
    let error = ep
        .allocate(4096, 64)
        .expect_err("the allocator refused, so the EP must not hand back a buffer");
    let message = error.to_string();
    assert!(
        message.contains("the tenant quota for this model is already spent"),
        "the allocator's own reason must survive: {message}"
    );
    assert!(
        message.contains("4096"),
        "the request must be named too: {message}"
    );
}

/// A provider with a standing pool can join the governor's accounting.
///
/// The seam is on the provider contract rather than on one backend because it
/// is not a CUDA question. A third-party provider that keeps its own pool
/// should be able to put it on the same ledger instead of running a second one
/// -- which is exactly what the CUDA weight-residency cache did, with a 4 GiB
/// default reconciled against nothing.
#[test]
fn a_third_party_provider_can_put_its_standing_pool_on_the_governor() {
    use onnx_runtime_memory_governor::{
        HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor, MemoryRole, Tier,
    };

    #[derive(Debug)]
    struct PoolingProvider {
        pool_bytes: u64,
        lease: std::sync::Mutex<Option<onnx_runtime_memory_governor::MemoryLease>>,
    }

    impl DeviceAllocator for PoolingProvider {
        fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
            HostAllocator.allocate(bytes, align)
        }
        unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) {
            // SAFETY: forwarded unchanged from this method's own contract.
            unsafe { HostAllocator.deallocate(ptr, bytes, align) };
        }
        fn device(&self) -> DeviceKey {
            DeviceKey::HOST
        }
    }

    let provider = PoolingProvider {
        pool_bytes: 600,
        lease: std::sync::Mutex::new(None),
    };
    let governor = LedgerGovernor::new(LeaseLedger::new(1000, 0, 0));

    // What `adopt_memory_governor` does for a provider that has a pool.
    let lease = governor
        .reserve(
            Tier::Device,
            provider.pool_bytes,
            MemoryRole::Weights,
            HolderId::new(4),
        )
        .expect("600 of 1000 is affordable");
    *provider.lease.lock().unwrap() = Some(lease);

    assert_eq!(governor.available(Tier::Device), 400);
    assert!(
        governor
            .reserve(Tier::Device, 600, MemoryRole::KvCache, HolderId::new(1))
            .is_err(),
        "the provider's pool must be visible to the next holder"
    );

    // Releasing it returns the bytes, so unloading a model frees its pool.
    *provider.lease.lock().unwrap() = None;
    assert_eq!(governor.available(Tier::Device), 1000);
}

/// A provider with no standing pool reports zero rather than failing.
#[test]
fn a_provider_without_a_standing_pool_adopts_nothing() {
    use onnx_runtime_memory_governor::{
        HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor, Tier,
    };

    let ep = CpuExecutionProvider::new();
    let governor = LedgerGovernor::new(LeaseLedger::new(0, 1000, 0));
    let governed = ep
        .adopt_memory_governor(&governor, Tier::Host, HolderId::new(4))
        .expect("holding no pool is not a failure");
    assert_eq!(governed, 0);
    assert_eq!(governor.available(Tier::Host), 1000);
}

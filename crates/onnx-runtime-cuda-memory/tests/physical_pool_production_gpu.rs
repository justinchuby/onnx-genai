//! Production allocator proof for cross-reservation physical-handle reuse.

use cudarc::driver::CudaContext;
use onnx_runtime_memory_governor::VirtualBacking as _;
use onnx_runtime_cuda_memory::vmm_allocator::{
    CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV, CudaVmmAllocator,
};
use onnx_runtime_memory_governor::{
    DeviceAllocator, DeviceKey, HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor, MemoryRole,
    Tier,
};
use std::sync::{Arc, Barrier};

const HOLDER: HolderId = HolderId::new(736);

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn production_allocators_share_handles_under_one_authority() {
    // SAFETY: this integration test has its own process and sets the production
    // option before constructing allocators or spawning threads.
    unsafe {
        std::env::set_var(
            CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV,
            (64usize << 20).to_string(),
        );
    }

    let governor = LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0));
    let first = CudaVmmAllocator::new(
        CudaContext::new(0).expect("first CUDA context wrapper"),
        DeviceKey::device(0),
        0,
        64 << 20,
        &governor,
        HOLDER,
        MemoryRole::Weights,
    )
    .expect("first production allocator");
    let second = CudaVmmAllocator::new(
        CudaContext::new(0).expect("second CUDA context wrapper"),
        DeviceKey::device(0),
        0,
        64 << 20,
        &governor,
        HOLDER,
        MemoryRole::Weights,
    )
    .expect("second production allocator");
    let stats = first
        .physical_pool_stats()
        .expect("production allocator must install the pool");
    let granule = 2usize << 20;

    let a = first.allocate(granule, 256).expect("map A");
    let after_a = stats.snapshot();
    assert_eq!(after_a.creates, 1);
    assert_eq!(after_a.releases, 0);
    let owned = after_a.total_owned_bytes;
    assert_eq!(after_a.mapped_bytes, owned);
    assert_eq!(
        (8u64 << 30) - governor.available(Tier::Device),
        owned,
        "the pool is the only physical-memory charge"
    );

    // SAFETY: A belongs to `first`; no CUDA work uses it.
    unsafe { first.deallocate(a, granule, 256) };
    let pooled = stats.snapshot();
    assert_eq!(pooled.releases, 0);
    assert_eq!(pooled.mapped_bytes, 0);
    assert_eq!(pooled.pooled_unmapped_bytes, owned);
    assert_eq!(pooled.total_owned_bytes, owned);

    let b = second.allocate(granule, 256).expect("map B");
    let after_b = stats.snapshot();
    assert_eq!(after_b.creates, after_a.creates);
    assert_eq!(after_b.releases, 0);
    assert_eq!(after_b.pool_hits, 1);
    assert_eq!(after_b.mapped_bytes, owned);
    assert_eq!(after_b.pooled_unmapped_bytes, 0);
    assert_eq!(after_b.total_owned_bytes, owned);

    let pattern = vec![0x5au8; granule];
    unsafe {
        use cudarc::driver::sys as cu;
        assert_eq!(
            cu::cuMemcpyHtoD_v2(
                b.as_ptr() as cu::CUdeviceptr,
                pattern.as_ptr().cast(),
                granule,
            ),
            cu::CUresult::CUDA_SUCCESS
        );
        let mut read_back = vec![0u8; granule];
        assert_eq!(
            cu::cuMemcpyDtoH_v2(
                read_back.as_mut_ptr().cast(),
                b.as_ptr() as cu::CUdeviceptr,
                granule,
            ),
            cu::CUresult::CUDA_SUCCESS
        );
        assert_eq!(read_back, pattern);
    }

    // SAFETY: B belongs to `second`; the synchronous copies completed.
    unsafe { second.deallocate(b, granule, 256) };
    drop(first);
    drop(second);
    let torn_down = stats.snapshot();
    assert_eq!(torn_down.creates, 1);
    assert_eq!(torn_down.releases, 1);
    assert_eq!(torn_down.total_owned_bytes, 0);
    assert_eq!(governor.available(Tier::Device), 8 << 30);
    eprintln!(
        "production pool transfer: creates={} releases={} hits={} owned={}",
        torn_down.creates, torn_down.releases, torn_down.pool_hits, torn_down.total_owned_bytes
    );

    let tx_governor = LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0));
    let tx_allocator = CudaVmmAllocator::new(
        CudaContext::new(0).expect("transaction CUDA context"),
        DeviceKey::device(0),
        0,
        64 << 20,
        &tx_governor,
        HolderId::new(738),
        MemoryRole::Weights,
    )
    .expect("transaction allocator");
    let transactional = tx_allocator
        .allocate_committed(granule * 3, 256, &[])
        .expect("reserve an uncommitted candidate");
    assert_eq!(
        tx_allocator
            .incremental_owned_bytes_for_span(transactional, granule * 3, 0, granule)
            .expect("estimate one granule"),
        granule as u64
    );
    let one = tx_allocator
        .try_commit_span(
            transactional,
            granule * 3,
            0,
            granule,
            granule as u64,
            granule as u64,
        )
        .expect("commit one granule");
    assert_eq!(one.additional_owned_bytes, granule as u64);
    assert_eq!(
        tx_allocator
            .incremental_owned_bytes_for_span(transactional, granule * 3, 0, granule)
            .expect("already-covered estimate"),
        0
    );
    let remaining = tx_allocator
        .try_commit_span(
            transactional,
            granule * 3,
            0,
            granule * 3,
            (granule * 2) as u64,
            (granule * 2) as u64,
        )
        .expect("commit multiple granules");
    assert_eq!(remaining.additional_owned_bytes, (granule * 2) as u64);
    unsafe { tx_allocator.deallocate(transactional, granule * 3, 256) };
    let reused = tx_allocator
        .allocate_committed(granule, 256, &[])
        .expect("reserve pooled candidate");
    let pooled_commit = tx_allocator
        .try_commit_span(reused, granule, 0, granule, granule as u64, 0)
        .expect("an already-owned pooled handle needs no physical headroom");
    assert_eq!(pooled_commit.additional_owned_bytes, 0);
    unsafe { tx_allocator.deallocate(reused, granule, 256) };
    let shared_anchor = tx_allocator.allocate(4096, 256).expect("shared anchor");
    let shared_candidate = tx_allocator
        .allocate_committed(4096, 256, &[])
        .expect("shared-granule candidate");
    assert_eq!(
        tx_allocator
            .incremental_owned_bytes_for_span(shared_candidate, 4096, 0, 4096)
            .expect("shared granule estimate"),
        0,
        "an already-mapped shared granule needs no new physical ownership"
    );
    tx_allocator
        .try_commit_span(shared_candidate, 4096, 0, 4096, 0, 0)
        .expect("shared granule commits with zero headroom");
    unsafe {
        tx_allocator.deallocate(shared_candidate, 4096, 256);
        tx_allocator.deallocate(shared_anchor, 4096, 256);
    }

    let arithmetic_governor =
        LedgerGovernor::new(LeaseLedger::new((granule + 742 * 1024) as u64, 0, 0));
    let arithmetic_allocator = CudaVmmAllocator::new(
        CudaContext::new(0).expect("arithmetic CUDA context"),
        DeviceKey::device(0),
        0,
        64 << 20,
        &arithmetic_governor,
        HolderId::new(739),
        MemoryRole::Weights,
    )
    .expect("arithmetic allocator");
    let held = arithmetic_allocator
        .allocate(granule, 256)
        .expect("first granule fits");
    let refused = arithmetic_allocator
        .allocate_committed(granule, 256, &[])
        .expect("reserve refused candidate");
    let error = arithmetic_allocator
        .try_commit_span(
            refused,
            granule,
            0,
            granule,
            granule as u64,
            arithmetic_governor.available(Tier::Device),
        )
        .expect_err("742 KiB cannot admit one 2 MiB granule");
    let message = error.to_string();
    assert!(message.contains(&format!("{granule} incremental committed bytes")));
    assert!(message.contains(&(742 * 1024).to_string()));
    unsafe {
        arithmetic_allocator.deallocate(refused, granule, 256);
        arithmetic_allocator.deallocate(held, granule, 256);
    }

    let rollback_governor = LedgerGovernor::new(LeaseLedger::new(granule as u64, 0, 0));
    let rollback_allocator = CudaVmmAllocator::new(
        CudaContext::new(0).expect("rollback CUDA context"),
        DeviceKey::device(0),
        0,
        64 << 20,
        &rollback_governor,
        HolderId::new(740),
        MemoryRole::Weights,
    )
    .expect("rollback allocator");
    let rollback_stats = rollback_allocator
        .physical_pool_stats()
        .expect("rollback pool stats");
    for _ in 0..2 {
        let candidate = rollback_allocator
            .allocate_committed(granule * 2, 256, &[])
            .expect("rollback candidate");
        rollback_allocator
            .try_commit_span(
                candidate,
                granule * 2,
                0,
                granule * 2,
                (granule * 2) as u64,
                (granule * 2) as u64,
            )
            .expect_err("second handle exceeds the real authority limit");
        let snapshot = rollback_stats.snapshot();
        assert_eq!(snapshot.total_owned_bytes, 0);
        assert_eq!(snapshot.pooled_unmapped_bytes, 0);
        assert_eq!(snapshot.mapped_bytes, 0);
        assert_eq!(snapshot.creates, snapshot.releases);
        assert_eq!(rollback_governor.used(Tier::Device), 0);
        unsafe { rollback_allocator.deallocate(candidate, granule * 2, 256) };
    }

    let race_governor = Arc::new(LedgerGovernor::new(LeaseLedger::new(granule as u64, 0, 0)));
    let race_allocator = Arc::new(
        CudaVmmAllocator::new(
            CudaContext::new(0).expect("race CUDA context"),
            DeviceKey::device(0),
            0,
            64 << 20,
            race_governor.as_ref(),
            HolderId::new(737),
            MemoryRole::Weights,
        )
        .expect("race allocator"),
    );
    let barrier = Arc::new(Barrier::new(2));
    let mut threads = Vec::new();
    for _ in 0..2 {
        let allocator = Arc::clone(&race_allocator);
        let governor = Arc::clone(&race_governor);
        let barrier = Arc::clone(&barrier);
        threads.push(std::thread::spawn(move || {
            let ptr = allocator
                .allocate_committed(granule, 256, &[])
                .expect("race candidate");
            barrier.wait();
            let result = allocator.try_commit_span(
                ptr,
                granule,
                0,
                granule,
                granule as u64,
                governor.available(Tier::Device),
            );
            if result.is_err() {
                unsafe { allocator.deallocate(ptr, granule, 256) };
                None
            } else {
                Some(ptr.as_ptr() as usize)
            }
        }));
    }
    let winners = threads
        .into_iter()
        .filter_map(|thread| thread.join().expect("race thread"))
        .collect::<Vec<_>>();
    assert_eq!(
        winners.len(),
        1,
        "exactly one admission may own the granule"
    );
    assert_eq!(race_governor.used(Tier::Device), granule as u64);
    let winner = std::ptr::NonNull::new(winners[0] as *mut u8).expect("winner pointer");
    unsafe { race_allocator.deallocate(winner, granule, 256) };
}

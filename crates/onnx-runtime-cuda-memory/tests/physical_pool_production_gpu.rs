//! Production allocator proof for cross-reservation physical-handle reuse.

use cudarc::driver::CudaContext;
use onnx_runtime_cuda_memory::vmm_allocator::{
    CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV, CudaVmmAllocator,
};
use onnx_runtime_memory_governor::{
    DeviceAllocator, DeviceKey, HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor, MemoryRole,
    Tier,
};

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
}

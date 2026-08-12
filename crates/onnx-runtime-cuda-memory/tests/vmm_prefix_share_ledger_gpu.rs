//! Q2 for #777 — does the **real** production allocator and ledger charge a
//! shared prefix granule **once**, keep it alive until the **last** sharer
//! leaves, and cost the Nth sharer only its **private** bytes? Tested against
//! `CudaVmmAllocator` and `LedgerGovernor` — no mock — because this is the most
//! extreme sharing case in the system and the 0->1 / 1->0 granule-ref
//! attribution (#740/#745) has never been exercised this way.
//!
//! # What "the real allocator" can and cannot prove today
//!
//! The driver-level probe (`vmm_prefix_share_gpu.rs`, Q1) proves the physical
//! primitive: one handle mapped into N *separate* reservations. The production
//! allocator does **not** yet expose a cross-reservation multi-map API — that is
//! precisely the #777 integration that is *not yet built*. What it does expose,
//! and what the future integration will rest on, is the **granule reference
//! count**: when several allocations' committed ranges cover the same physical
//! granule, that granule is created once (ref 0->1), its refcount rises with
//! each additional sharer, and it is released only when the last sharer drops it
//! (ref 1->0). This test drives that attribution to the limit and reads the
//! charge off the **real** `LedgerGovernor`, on the physical (owned) axis.
//!
//! The production analogue of "N sequences sharing one prefix granule" that the
//! current allocator can express is N sub-granule allocations packed into one
//! granule: each is an independent tenant of the same physical page, exactly the
//! refcount case above. The honest caveat (also in the PR writeup): true
//! cross-reservation prefix multi-map needs a new allocator entry point; this
//! test validates the *attribution* that entry point must use, not the entry
//! point itself.

use cudarc::driver::CudaContext;
use onnx_runtime_cuda_memory::vmm_allocator::{
    CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV, CudaVmmAllocator,
};
use onnx_runtime_memory_governor::{
    DeviceAllocator, DeviceKey, HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor, MemoryRole,
    Tier,
};

const HOLDER: HolderId = HolderId::new(777);

fn granularity(device_ordinal: i32) -> usize {
    use cudarc::driver::sys as cu;
    let mut prop: cu::CUmemAllocationProp = unsafe { std::mem::zeroed() };
    prop.type_ = cu::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_ordinal;
    let mut granularity = 0usize;
    let result = unsafe {
        cu::cuMemGetAllocationGranularity(
            &mut granularity,
            &prop,
            cu::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        )
    };
    assert_eq!(
        result,
        cu::CUresult::CUDA_SUCCESS,
        "granularity: {result:?}"
    );
    assert_ne!(granularity, 0, "CUDA reported zero VMM granularity");
    granularity
}

/// Q2 — one shared prefix granule, N tenants: charged once, alive until the
/// last leaves, and each additional tenant needs **zero** incremental physical
/// ownership. The final assertion states the admission consequence directly:
/// admitting the Nth sharer costs only its private bytes.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn shared_prefix_granule_charged_once_alive_until_last_and_nth_sharer_is_free() {
    // SAFETY: this integration test owns its process and sets the production
    // pool option before constructing any allocator. A nonzero bound installs
    // the production pool, whose retention is accounted for below.
    unsafe {
        std::env::set_var(
            CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV,
            (64usize << 20).to_string(),
        );
    }

    let context = CudaContext::new(0).expect("CUDA context");
    let device = 0;
    let granule = granularity(device);

    let governor = LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0));
    let allocator = CudaVmmAllocator::new(
        context,
        DeviceKey::device(0),
        0,
        64 << 20,
        &governor,
        HOLDER,
        MemoryRole::KvCache,
    )
    .expect("production allocator");
    let pool = allocator
        .physical_pool_stats()
        .expect("production allocator installs the pool");

    // N tenants of one prefix granule. Sized so the whole set fits inside one
    // 2 MiB granule: N sub-granule allocations packed together are N independent
    // references to the same physical page — the charge-once case.
    const N: usize = 8;
    let tenant_bytes = granule / (N * 2); // 8 tenants use half the granule
    assert!(
        N * tenant_bytes < granule,
        "the sharing set must fit inside one granule to be the charge-once case"
    );

    // The first tenant is the shared prefix owner: it creates the granule
    // (ref 0->1) and is charged exactly one granule.
    let first = allocator
        .allocate(tenant_bytes, 256)
        .expect("first prefix tenant");
    let charged_after_first = governor.used(Tier::Device);
    assert_eq!(
        charged_after_first, granule as u64,
        "the shared prefix granule must be charged exactly once (one granule) by the first tenant"
    );
    assert_eq!(
        pool.snapshot().mapped_bytes,
        granule as u64,
        "one physical handle, mapped once, backs the shared prefix"
    );

    // Each additional sharer joins the already-committed granule. Its
    // incremental physical ownership is ZERO — this is the admission arithmetic
    // the serving path needs: the Nth request pays only its private bytes.
    let mut tenants = vec![first];
    for n in 1..N {
        let incremental = allocator
            .incremental_owned_bytes_for_span(tenants[0], tenant_bytes, 0, tenant_bytes)
            .expect("estimate the shared granule for an existing tenant");
        assert_eq!(
            incremental, 0,
            "sharer {n} joins an already-committed prefix granule and must need zero incremental \
             physical ownership"
        );
        let next = allocator
            .allocate(tenant_bytes, 256)
            .expect("additional prefix tenant");
        tenants.push(next);
        assert_eq!(
            governor.used(Tier::Device),
            granule as u64,
            "adding sharer {n} must NOT re-charge the shared granule — it stays one granule"
        );
    }

    eprintln!(
        "Q2 charge-once: {N} tenants share one {} MiB prefix granule. Real ledger charges {} MiB \
         total (one granule), not {} MiB ({N} per-tenant copies). Each additional sharer needed 0 \
         incremental owned bytes -> admitting the Nth request costs only its PRIVATE bytes.",
        granule / (1024 * 1024),
        governor.used(Tier::Device) / (1024 * 1024),
        (N * granule) / (1024 * 1024),
    );

    // Alive until the last sharer leaves: drop N-1 tenants and confirm the
    // granule stays charged and mapped throughout (refcount N -> 1).
    for n in 0..(N - 1) {
        let ptr = tenants.pop().expect("tenant to drop");
        // SAFETY: each pointer came from this allocator and is still live.
        unsafe { allocator.deallocate(ptr, tenant_bytes, 256) };
        assert_eq!(
            governor.used(Tier::Device),
            granule as u64,
            "dropping sharer {n} of {N} must NOT release the shared granule — others still hold it"
        );
        assert_eq!(
            pool.snapshot().mapped_bytes,
            granule as u64,
            "the shared physical handle must remain mapped while any sharer references it"
        );
    }

    // The last sharer leaves: refcount 1->0, the granule is unmapped, and the
    // shared granule becomes freeable. The physical handle is *retained* by the
    // #740 pool for reuse (not leaked), so the governor lease stays until
    // teardown — mapping refcount reaching zero is the "alive until last" proof.
    let last = tenants.pop().expect("last tenant");
    // SAFETY: last live pointer from this allocator.
    unsafe { allocator.deallocate(last, tenant_bytes, 256) };
    let freed = pool.snapshot();
    assert_eq!(
        freed.mapped_bytes, 0,
        "once the LAST sharer leaves, the shared granule's mapping refcount reaches zero"
    );
    assert_eq!(
        freed.pooled_unmapped_bytes, granule as u64,
        "the shared physical handle is retained by the pool for reuse, not released, after the \
         last sharer"
    );

    // Teardown releases the retained handle entirely and the ledger returns to
    // zero.
    drop(allocator);
    let torn_down = pool.snapshot();
    assert_eq!(
        torn_down.total_owned_bytes, 0,
        "every physical handle must be released on teardown"
    );
    assert_eq!(
        torn_down.creates, torn_down.releases,
        "every created handle released exactly once"
    );
    assert_eq!(
        governor.used(Tier::Device),
        0,
        "the ledger returns fully to zero after teardown"
    );
}

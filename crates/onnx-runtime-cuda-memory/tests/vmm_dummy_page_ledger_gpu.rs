//! Q4 for #759 — is a dummy handle mapped across a whole tail *charged once*,
//! and does its charge return to zero when freed? Two complementary proofs:
//!
//! * **Part A (driver-level physical truth)** — one physical handle mapped at
//!   many virtual addresses consumes *one page* of device memory, not one per
//!   VA, and every alias sees the same bytes. This is the dummy-page shape #759
//!   proposes, measured directly with `cuMemGetInfo`.
//! * **Part B (real production allocator + ledger)** — the attribution the
//!   whole system rests on (global granule refcount `0->1` / `1->0`, #740/#745)
//!   charges a *shared* granule to the [`LedgerGovernor`] exactly once no matter
//!   how many allocations reference it, and releases it back to zero when the
//!   last reference leaves. A dummy handle backing a whole tail is the most
//!   extreme sharing case in the system, so this exercises the real allocator
//!   and real ledger — no mock — at that limit.
//!
//! Together: the dummy page's *physical* cost is one page (Part A), and the
//! ledger's charge-once/free-to-zero invariant holds under maximal sharing
//! (Part B). The honest caveat, spelled out in the PR writeup: the production
//! commit path maps a *distinct* handle per granule, so `mapped_bytes` (virtual
//! coverage) would count each dummy VA while `total_owned_bytes` (physical)
//! counts the one handle once — a genuine dummy-page integration must charge on
//! the owned axis, which is what the governor already bills.

use cudarc::driver::CudaContext;
use cudarc::driver::sys as cu;
use onnx_runtime_cuda_memory::vmm_allocator::{
    CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV, CudaVmmAllocator,
};
use onnx_runtime_memory_governor::{
    DeviceAllocator, DeviceKey, HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor, MemoryRole,
    Tier,
};

const HOLDER: HolderId = HolderId::new(759);
/// Arbitrary non-zero byte used to prove VA aliasing; not the production fill
/// (the masking rule decides that — zeros, never NaN).
const READBACK_MARKER: u8 = 0x5a;

fn check(call: &'static str, result: cu::CUresult) {
    assert_eq!(result, cu::CUresult::CUDA_SUCCESS, "{call}: {result:?}");
}

fn allocation_prop(device_ordinal: i32) -> cu::CUmemAllocationProp {
    let mut prop: cu::CUmemAllocationProp = unsafe { std::mem::zeroed() };
    prop.type_ = cu::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_ordinal;
    prop
}

fn granularity(device_ordinal: i32) -> usize {
    let prop = allocation_prop(device_ordinal);
    let mut granularity = 0usize;
    check("cuMemGetAllocationGranularity", unsafe {
        cu::cuMemGetAllocationGranularity(
            &mut granularity,
            &prop,
            cu::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        )
    });
    assert_ne!(granularity, 0, "CUDA reported zero VMM granularity");
    granularity
}

fn free_bytes() -> usize {
    let mut free = 0usize;
    let mut total = 0usize;
    check("cuMemGetInfo_v2", unsafe {
        cu::cuMemGetInfo_v2(&mut free, &mut total)
    });
    free
}

/// Q4 Part A — one physical handle mapped at N virtual addresses is one page of
/// physical memory, and all N addresses alias it. This is the dummy-page shape:
/// if it were charged per VA, backing a 96-object tail would cost 96 pages; it
/// costs one.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn one_dummy_handle_at_many_vas_is_one_physical_page() {
    let context = CudaContext::new(0).expect("CUDA context");
    context.bind_to_thread().expect("bind CUDA context");
    let device = 0;
    let granule = granularity(device);

    const VAS: usize = 96;

    // Baseline free memory, then create exactly one handle and map it at VAS
    // separate granule-aligned virtual addresses.
    let baseline_free = free_bytes();
    let prop = allocation_prop(device);
    let mut dummy = 0;
    check("cuMemCreate", unsafe {
        cu::cuMemCreate(&mut dummy, granule, &prop, 0)
    });

    let mut base = 0;
    check("cuMemAddressReserve", unsafe {
        cu::cuMemAddressReserve(&mut base, granule * VAS, 0, 0, 0)
    });
    for i in 0..VAS {
        let addr = base + (i * granule) as u64;
        check("cuMemMap", unsafe {
            cu::cuMemMap(addr, granule, 0, dummy, 0)
        });
        let mut access: cu::CUmemAccessDesc = unsafe { std::mem::zeroed() };
        access.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = device;
        access.flags = cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        check("cuMemSetAccess", unsafe {
            cu::cuMemSetAccess(addr, granule, &access, 1)
        });
    }

    let mapped_free = free_bytes();
    let used = baseline_free.saturating_sub(mapped_free);

    // Aliasing: a write through the first VA is visible through the last.
    let payload = vec![READBACK_MARKER; 4096];
    check("cuMemcpyHtoD_v2", unsafe {
        cu::cuMemcpyHtoD_v2(base, payload.as_ptr().cast(), payload.len())
    });
    let mut back = vec![0u8; 4096];
    check("cuMemcpyDtoH_v2", unsafe {
        cu::cuMemcpyDtoH_v2(
            back.as_mut_ptr().cast(),
            base + ((VAS - 1) * granule) as u64,
            back.len(),
        )
    });
    let aliased = back.iter().all(|&b| b == READBACK_MARKER);

    for i in 0..VAS {
        let _ = unsafe { cu::cuMemUnmap(base + (i * granule) as u64, granule) };
    }
    let _ = unsafe { cu::cuMemAddressFree(base, granule * VAS) };
    let _ = unsafe { cu::cuMemRelease(dummy) };
    let freed_free = free_bytes();

    eprintln!(
        "Q4A one handle at {VAS} VAs: used {} KiB physical (granule {} KiB); charged-per-VA would \
         be {} KiB; aliasing across VAs = {aliased}",
        used / 1024,
        granule / 1024,
        (VAS * granule) / 1024,
    );

    assert!(aliased, "all VAs must alias the one physical page");
    assert!(
        used < 16 * granule,
        "one handle mapped at {VAS} VAs must cost ~1 page, not {VAS}: measured {used} bytes used, \
         which is not far below the {} bytes a per-VA charge would cost",
        VAS * granule
    );
    // The page is freeable: after unmap+release, the memory comes back.
    assert!(
        freed_free + 4 * granule >= baseline_free,
        "releasing the dummy handle must return its physical page (baseline {baseline_free}, \
         after free {freed_free})"
    );
}

/// Q4 Part B — the real ledger charges a shared granule once, drops its mapping
/// refcount to zero when the last tenant leaves, and fully releases the
/// physical handle on teardown. Many small allocations carved into the same
/// granule are the production analogue of maximal sharing; the granule refcount
/// that #740/#745 attribute on must map that to a single physical charge.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn shared_granule_is_charged_once_and_freed_to_zero_in_the_real_ledger() {
    // SAFETY: this integration test owns its process and sets the option before
    // constructing the allocator. A nonzero retention bound installs the
    // production pool; the pool retains freed handles until teardown, which the
    // assertions below account for (refcount->0 on free, full release on drop).
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
        MemoryRole::Workspace { step_scoped: false },
    )
    .expect("production allocator");
    let pool = allocator
        .physical_pool_stats()
        .expect("production allocator installs the pool");

    // Many sub-granule allocations, sized so the whole set fits well inside a
    // single 2 MiB granule -- the extreme sharing case.
    const ALLOCS: usize = 128;
    const ALLOC_BYTES: usize = 8 << 10; // 8 KiB; 128 * 8 KiB = 1 MiB < granule
    assert!(
        ALLOCS * ALLOC_BYTES < granule,
        "the sharing set must fit inside one granule to be the charge-once case"
    );

    let mut live = Vec::with_capacity(ALLOCS);
    for _ in 0..ALLOCS {
        live.push(
            allocator
                .allocate(ALLOC_BYTES, 256)
                .expect("sub-granule allocation"),
        );
    }

    let charged = governor.used(Tier::Device);
    let mapped = pool.snapshot().mapped_bytes;
    eprintln!(
        "Q4B {ALLOCS} allocations x {ALLOC_BYTES} B sharing one granule: governor charged {} KiB, \
         pool mapped {} KiB; a per-allocation charge would be {} KiB",
        charged / 1024,
        mapped / 1024,
        (ALLOCS * ALLOC_BYTES) / 1024,
    );
    assert!(
        (granule as u64..=2 * granule as u64).contains(&charged),
        "the shared granule(s) must be charged once (1-2 granules for {ALLOCS} allocations), not \
         per allocation; governor charged {charged} bytes"
    );
    assert_eq!(
        mapped, charged,
        "the pool's mapped physical bytes must equal the governor charge -- one physical handle \
         per shared granule, not per allocation"
    );

    for ptr in live {
        // SAFETY: each pointer came from this allocator and is still live.
        unsafe { allocator.deallocate(ptr, ALLOC_BYTES, 256) };
    }

    // The mapping refcount returned to zero: no granule is mapped any more, so
    // the shared granule is freeable. The physical handle is *retained* by the
    // pool for reuse (not a leak) until teardown.
    let freed = pool.snapshot();
    eprintln!(
        "Q4B after freeing all: pool mapped {} B, retained {} B (handle pooled for reuse)",
        freed.mapped_bytes, freed.pooled_unmapped_bytes
    );
    assert_eq!(
        freed.mapped_bytes, 0,
        "the shared granule's mapping refcount must reach zero -- nothing stays mapped once every \
         tenant leaves"
    );

    // Teardown releases the retained handle: the charge returns fully to zero.
    drop(allocator);
    let torn_down = pool.snapshot();
    eprintln!(
        "Q4B after teardown: pool owns {} B, creates {}, releases {}, governor used {} B",
        torn_down.total_owned_bytes,
        torn_down.creates,
        torn_down.releases,
        governor.used(Tier::Device),
    );
    assert_eq!(
        torn_down.total_owned_bytes, 0,
        "every physical handle must be released on teardown -- refcount fully to zero"
    );
    assert_eq!(
        torn_down.creates, torn_down.releases,
        "every created handle must be released exactly once"
    );
    assert_eq!(
        governor.used(Tier::Device),
        0,
        "the ledger charge must return to zero once the shared granule is freed and released"
    );
}

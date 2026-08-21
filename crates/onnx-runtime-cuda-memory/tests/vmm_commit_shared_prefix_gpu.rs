//! The `commit_shared_prefix` allocator entry point (#777) — the smallest next
//! increment named by the #793 probe. These tests exercise the **production**
//! `CudaVmmAllocator` + `LedgerGovernor` (no mock), because the whole point of
//! the primitive is that it reuses the #740 authority-scoped pool and the
//! 0->1 / 1->0 granule-ref attribution the probe already proved (Q2) rather
//! than inventing a second allocator.
//!
//! What the probe left to build was the **cross-reservation multi-map API
//! itself**: Q1 proved one physical handle reads identically at N virtual
//! addresses at the driver level; Q2 proved the attribution against the real
//! ledger. This binary proves the allocator entry point that ties them
//! together: create a pinned prefix once, map it read-only into many sequences'
//! reservations at zero incremental owned bytes, keep it alive until the last
//! sharer, fault loudly on a stray write, and error rather than mis-map on an
//! unsupported request.
//!
//! # Harness discipline (per #797)
//!
//! A readback on the legacy default stream can race a memset issued on a
//! *created* non-blocking stream — those two are mutually exempt from implicit
//! synchronization, and the result is a silent partial fill with no CUDA error
//! that `cuCtxSynchronize` does not fix. Every device operation here is a
//! synchronous `cuMemcpy*_v2` on the one implicit stream, except the Q3-style
//! asynchronous write, which is issued **and** synced on the *same* created
//! stream so one `cuStreamSynchronize` is a total order over it. Run this
//! binary single-threaded (`--test-threads=1`).
//!
//! # Shared-prefix fallback coverage matrix
//!
//! | Edge | Test |
//! |---|---|
//! | First-map rejection: capture, overlap, offset/length geometry | `unsupported_shared_prefix_requests_error_rather_than_mismap` |
//! | Wrong logical device / physical-pool authority | `foreign_device_and_authority_prefixes_are_not_free_and_are_rejected` |
//! | Repeated private fallback; bounded pool/ref/mapping accounting | `repeated_precommitted_fallback_reuses_without_leaking` |
//! | Repeated deferred release drains before physical-handle reuse | `provider::tests::standalone_vmm_scratch_reuse_pools_committed_memory_and_does_not_scale_cumemcreate` |
//! | One- and two-granule prefix/tail boundaries | `one_and_two_granule_prefix_boundaries_map_and_cleanup` |
//! | Owner/sharer lifetime, including one sharer exiting first | `n_sequences_share_one_pinned_prefix_charged_once_alive_until_last` |
//! | Peer + private-fallback fp16 GQA isolation | `gqa_shared_prefix_parity_gpu::shared_peers_survive_request_local_private_fallback` |
//!
//! The current instance-scoped `DriverFaultPlan` injects release `Unmap`,
//! rollback `Remap`, and `Dispose` failures; it has no initial shared-map or
//! `cuMemSetAccess` injection point. Partial multi-granule shared-map rollback
//! therefore remains unforced here rather than being simulated by a new,
//! test-only production seam.

use std::panic::AssertUnwindSafe;
use std::ptr::NonNull;
use std::sync::Arc;

use cudarc::driver::CudaContext;
use cudarc::driver::sys as cu;
use onnx_runtime_cuda_memory::vmm_allocator::{
    CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV, CudaVmmAllocator,
};
use onnx_runtime_memory_governor::{
    DeviceAllocator, DeviceKey, HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor, MemoryRole,
    Tier, VirtualBacking,
};

const HOLDER: HolderId = HolderId::new(777);
/// Distinct byte the prefix is filled with, to prove every sharer reads the
/// same physical page. Not a production fill.
const PREFIX_MARKER: u8 = 0x5a;
const PROBE_LEN: usize = 4096;

fn require_cuda() -> Arc<CudaContext> {
    match CudaContext::new(0) {
        Ok(context) => context,
        Err(error) => panic!(
            "commit_shared_prefix test requires a CUDA driver; CPU-only runs must leave it ignored: {error}"
        ),
    }
}

fn set_pool_env() {
    // SAFETY: each integration test owns its process; the pool option is set
    // before any allocator is constructed. A non-zero bound installs the
    // production pool whose retention the assertions account for.
    unsafe {
        std::env::set_var(
            CUDA_PHYSICAL_HANDLE_POOL_BYTES_ENV,
            (64usize << 20).to_string(),
        );
    }
}

fn granularity(device_ordinal: i32) -> usize {
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

fn allocator(governor: &dyn MemoryGovernor, capacity: usize) -> CudaVmmAllocator {
    CudaVmmAllocator::new(
        CudaContext::new(0).expect("CUDA context"),
        DeviceKey::device(0),
        0,
        capacity,
        governor,
        HOLDER,
        MemoryRole::KvCache,
    )
    .expect("production allocator")
}

fn write_host(address: u64, value: u8, len: usize) {
    let bytes = vec![value; len];
    let result = unsafe { cu::cuMemcpyHtoD_v2(address, bytes.as_ptr().cast(), bytes.len()) };
    assert_eq!(
        result,
        cu::CUresult::CUDA_SUCCESS,
        "cuMemcpyHtoD_v2: {result:?}"
    );
}

fn read_host(address: u64, len: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; len];
    let result = unsafe { cu::cuMemcpyDtoH_v2(bytes.as_mut_ptr().cast(), address, bytes.len()) };
    assert_eq!(
        result,
        cu::CUresult::CUDA_SUCCESS,
        "cuMemcpyDtoH_v2: {result:?}"
    );
    bytes
}

/// Fresh reservation + commit + write + read on a brand-new VA — nothing to do
/// with the shared prefix. If an earlier write poisoned the context these fail
/// with a sticky error. Mirrors the Q3 probe's health check.
fn context_is_healthy(device: i32, granule: usize) -> Result<(), cu::CUresult> {
    let sync = unsafe { cu::cuCtxSynchronize() };
    if sync != cu::CUresult::CUDA_SUCCESS {
        return Err(sync);
    }
    let mut base = 0;
    let reserve = unsafe { cu::cuMemAddressReserve(&mut base, granule, 0, 0, 0) };
    if reserve != cu::CUresult::CUDA_SUCCESS {
        return Err(reserve);
    }
    let mut prop: cu::CUmemAllocationProp = unsafe { std::mem::zeroed() };
    prop.type_ = cu::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    let mut handle = 0;
    let created = unsafe { cu::cuMemCreate(&mut handle, granule, &prop, 0) };
    let outcome = (|| {
        if created != cu::CUresult::CUDA_SUCCESS {
            return Err(created);
        }
        let map = unsafe { cu::cuMemMap(base, granule, 0, handle, 0) };
        if map != cu::CUresult::CUDA_SUCCESS {
            return Err(map);
        }
        let mut access: cu::CUmemAccessDesc = unsafe { std::mem::zeroed() };
        access.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = device;
        access.flags = cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        let grant = unsafe { cu::cuMemSetAccess(base, granule, &access, 1) };
        if grant != cu::CUresult::CUDA_SUCCESS {
            return Err(grant);
        }
        let probe = vec![0x11u8; PROBE_LEN];
        let w = unsafe { cu::cuMemcpyHtoD_v2(base, probe.as_ptr().cast(), probe.len()) };
        if w != cu::CUresult::CUDA_SUCCESS {
            return Err(w);
        }
        let mut back = vec![0u8; PROBE_LEN];
        let r = unsafe { cu::cuMemcpyDtoH_v2(back.as_mut_ptr().cast(), base, back.len()) };
        if r != cu::CUresult::CUDA_SUCCESS {
            return Err(r);
        }
        if back.iter().any(|&b| b != 0x11) {
            return Err(cu::CUresult::CUDA_ERROR_UNKNOWN);
        }
        Ok(())
    })();
    unsafe {
        let _ = cu::cuMemUnmap(base, granule);
        let _ = cu::cuMemRelease(handle);
        let _ = cu::cuMemAddressFree(base, granule);
    }
    outcome
}

/// Allocate one sequence's KV reservation: a `SEQ_GRANULES`-granule span,
/// granule-aligned so its prefix maps whole physical granules, with only the
/// private tail committed and the prefix region left uncommitted for the shared
/// map to fill.
fn allocate_sequence(
    allocator: &CudaVmmAllocator,
    granule: usize,
    seq_granules: usize,
) -> NonNull<u8> {
    let bytes = granule * seq_granules;
    allocator
        .allocate_committed(bytes, granule, std::slice::from_ref(&(granule..bytes)))
        .expect("sequence reservation with a private tail")
}

/// Q-cross-reservation — N sequences share one pinned prefix: it is charged
/// **once**, each shared map costs **zero** incremental owned bytes, every
/// sharer reads the identical physical page, and the prefix stays alive for the
/// **union** of its sharers — dropping the owner does not release it while any
/// sharer still maps it.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn n_sequences_share_one_pinned_prefix_charged_once_alive_until_last() {
    set_pool_env();
    let context = require_cuda();
    context.bind_to_thread().expect("bind CUDA context");
    let device = 0;
    let granule = granularity(device);

    let governor = LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0));
    let allocator = allocator(&governor, 256 << 20);
    let stats = allocator
        .physical_pool_stats()
        .expect("production allocator installs the pool");

    const N: usize = 8;
    const SEQ_GRANULES: usize = 2;

    // Create the pinned prefix — one granule, charged exactly once on the owned
    // axis — and fill its content through the owner's writable window.
    let prefix = allocator
        .create_shared_prefix(granule)
        .expect("pinned shared prefix");
    assert_eq!(
        prefix.committed_physical_bytes(),
        granule as u64,
        "the pinned prefix owns exactly one granule of physical memory"
    );
    assert_eq!(
        governor.used(Tier::Device),
        granule as u64,
        "the shared prefix granule must be charged exactly once (one granule)"
    );
    write_host(prefix.device_ptr(), PREFIX_MARKER, granule);

    // Map the prefix read-only into N sequences. Each sequence's PRIVATE tail
    // was already committed (one granule each); the shared map itself must add
    // ZERO incremental owned bytes.
    let mut sequences = Vec::with_capacity(N);
    for n in 0..N {
        let ptr = allocate_sequence(&allocator, granule, SEQ_GRANULES);
        let owned_before = governor.used(Tier::Device);
        assert_eq!(
            allocator
                .incremental_owned_bytes_for_shared_prefix(&prefix)
                .expect("same-device same-authority prefix"),
            0,
            "admitting sharer {n} must need zero incremental owned bytes for the prefix"
        );
        let commit = allocator
            .commit_shared_prefix(&prefix, ptr, granule * SEQ_GRANULES, 0)
            .expect("map the shared prefix into a sequence");
        assert_eq!(commit.additional_owned_bytes, 0);
        assert_eq!(commit.granules, 1);
        assert_eq!(
            governor.used(Tier::Device),
            owned_before,
            "mapping the shared prefix into sharer {n} must NOT charge the owned axis"
        );
        sequences.push(ptr);
    }

    // Every sharer reads the identical prefix bytes through its own VA.
    for (n, &ptr) in sequences.iter().enumerate() {
        let via_sharer = read_host(ptr.as_ptr() as u64, PROBE_LEN);
        assert!(
            via_sharer.iter().all(|&b| b == PREFIX_MARKER),
            "sharer {n} must read the shared prefix content"
        );
    }

    // The owned axis now holds one prefix granule + N private tails: the prefix
    // was charged once, each sequence paid only its private bytes.
    assert_eq!(
        governor.used(Tier::Device),
        ((1 + N) * granule) as u64,
        "owned bytes are one shared prefix granule plus one private tail per sequence"
    );
    eprintln!(
        "commit_shared_prefix: {N} sequences share one {} MiB prefix. Owned = {} MiB (1 prefix + \
         {N} private tails), not {} MiB ({N} full copies + tails). Each shared map cost 0 owned \
         bytes.",
        granule / (1024 * 1024),
        governor.used(Tier::Device) / (1024 * 1024),
        (2 * N * granule) / (1024 * 1024),
    );

    // Union lifetime: drop the OWNER while sharers still map the prefix. The
    // physical page must stay alive — every sharer still reads it correctly.
    drop(prefix);
    for (n, &ptr) in sequences.iter().enumerate() {
        let after_owner_gone = read_host(ptr.as_ptr() as u64, PROBE_LEN);
        assert!(
            after_owner_gone.iter().all(|&b| b == PREFIX_MARKER),
            "sharer {n} must still read the shared prefix after the owner is dropped (union \
             lifetime)"
        );
    }

    // Alive until the LAST sharer: drop all but one; the survivor still reads
    // the shared page.
    while sequences.len() > 1 {
        let ptr = sequences.pop().unwrap();
        // SAFETY: from this allocator, still live, no CUDA work in flight.
        unsafe { allocator.deallocate(ptr, granule * SEQ_GRANULES, granule) };
    }
    let survivor = sequences[0];
    let last_read = read_host(survivor.as_ptr() as u64, PROBE_LEN);
    assert!(
        last_read.iter().all(|&b| b == PREFIX_MARKER),
        "the final sharer must still read the shared prefix"
    );
    // SAFETY: last live sharer from this allocator.
    unsafe { allocator.deallocate(survivor, granule * SEQ_GRANULES, granule) };

    drop(allocator);
    let torn_down = stats.snapshot();
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

/// Admitting an additional sharer costs only its private bytes: the shared map
/// is free on the owned axis, and the sequence's own private granule is what
/// moves the real ledger.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn admitting_a_sharer_costs_only_private_bytes() {
    set_pool_env();
    let context = require_cuda();
    context.bind_to_thread().expect("bind CUDA context");
    let device = 0;
    let granule = granularity(device);

    let governor = LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0));
    let allocator = allocator(&governor, 64 << 20);

    let prefix = allocator
        .create_shared_prefix(granule)
        .expect("pinned shared prefix");
    let baseline = governor.used(Tier::Device);
    assert_eq!(baseline, granule as u64, "prefix charged once");

    // Reserve a 2-granule sequence with NOTHING committed yet.
    let bytes = granule * 2;
    let ptr = allocator
        .allocate_committed(bytes, granule, &[])
        .expect("uncommitted sequence reservation");
    assert_eq!(
        governor.used(Tier::Device),
        baseline,
        "reserving address space charges nothing"
    );

    // Mapping the shared prefix at offset 0 is free on the owned axis.
    assert_eq!(
        allocator
            .incremental_owned_bytes_for_shared_prefix(&prefix)
            .expect("same-device same-authority prefix"),
        0
    );
    allocator
        .commit_shared_prefix(&prefix, ptr, bytes, 0)
        .expect("map shared prefix");
    assert_eq!(
        governor.used(Tier::Device),
        baseline,
        "the shared map is free — admitting the sharer has cost nothing yet"
    );

    // Committing the sequence's own PRIVATE tail is the only thing that charges
    // the ledger: exactly one granule.
    allocator
        .commit_allocation_range(ptr, bytes, granule, granule, granule)
        .expect("commit the private tail");
    assert_eq!(
        governor.used(Tier::Device),
        baseline + granule as u64,
        "admitting the sharer costs exactly its private granule, nothing for the shared prefix"
    );

    drop(prefix);
    // SAFETY: live pointer from this allocator, no CUDA work in flight.
    unsafe { allocator.deallocate(ptr, bytes, granule) };
}

/// A stray write into a shared prefix faults loudly and non-stickily (Q3
/// property, preserved by `commit_shared_prefix` mapping `PROT_READ`): the
/// context survives, and the owner's copy is uncorrupted.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn a_write_into_a_shared_prefix_faults_and_the_context_survives() {
    set_pool_env();
    let context = require_cuda();
    context.bind_to_thread().expect("bind CUDA context");
    let device = 0;
    let granule = granularity(device);

    let governor = LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0));
    let allocator = allocator(&governor, 64 << 20);

    let prefix = allocator
        .create_shared_prefix(granule)
        .expect("pinned shared prefix");
    write_host(prefix.device_ptr(), PREFIX_MARKER, granule);

    let bytes = granule * 2;
    let ptr = allocate_sequence(&allocator, granule, 2);
    allocator
        .commit_shared_prefix(&prefix, ptr, bytes, 0)
        .expect("map shared prefix read-only");
    let shared_va = ptr.as_ptr() as u64;

    // A read through the sharer works.
    let via_sharer = read_host(shared_va, PROBE_LEN);
    assert!(via_sharer.iter().all(|&b| b == PREFIX_MARKER));

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        // Path 1: synchronous copy-engine write into the read-only shared prefix.
        let payload = vec![0x00u8; PROBE_LEN];
        let sync_write =
            unsafe { cu::cuMemcpyHtoD_v2(shared_va, payload.as_ptr().cast(), payload.len()) };
        eprintln!("synchronous write into PROT_READ shared prefix: {sync_write:?}");
        assert_ne!(
            sync_write,
            cu::CUresult::CUDA_SUCCESS,
            "a write into the read-only shared prefix must fault, or the page is unprotected"
        );
        match context_is_healthy(device, granule) {
            Ok(()) => eprintln!("context healthy after synchronous write fault (non-sticky)"),
            Err(err) => panic!(
                "KILL FINDING: a synchronous write into the shared prefix poisoned the context \
                 (recovery probe {err:?}); the fault is sticky"
            ),
        }

        // Path 2: asynchronous memset issued AND synced on the same created
        // stream (per #797, one stream = one total order).
        let mut stream = std::ptr::null_mut();
        let created = unsafe {
            cu::cuStreamCreate(
                &mut stream,
                cu::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
            )
        };
        assert_eq!(created, cu::CUresult::CUDA_SUCCESS, "cuStreamCreate");
        let async_write = unsafe { cu::cuMemsetD8Async(shared_va, 0x00, granule, stream) };
        let async_sync = unsafe { cu::cuStreamSynchronize(stream) };
        let _ = unsafe { cu::cuStreamDestroy_v2(stream) };
        eprintln!(
            "async write into PROT_READ shared prefix: enqueue {async_write:?}, sync {async_sync:?}"
        );
        assert!(
            async_write != cu::CUresult::CUDA_SUCCESS || async_sync != cu::CUresult::CUDA_SUCCESS,
            "an async write into the read-only shared prefix must surface a fault"
        );
        match context_is_healthy(device, granule) {
            Ok(()) => eprintln!("context healthy after asynchronous write fault (non-sticky)"),
            Err(err) => panic!(
                "KILL FINDING: an async write into the shared prefix poisoned the context \
                 (recovery probe {err:?}); the fault is sticky"
            ),
        }

        // The owner's copy is uncorrupted by the rejected writes.
        let via_owner = read_host(prefix.device_ptr(), PROBE_LEN);
        assert!(
            via_owner.iter().all(|&b| b == PREFIX_MARKER),
            "the rejected writes must NOT have corrupted the shared page"
        );
    }));

    // SAFETY: live pointer from this allocator, no CUDA work in flight.
    unsafe { allocator.deallocate(ptr, bytes, granule) };
    drop(prefix);
    result.unwrap();
}

/// Unsupported requests error rather than mis-mapping.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn unsupported_shared_prefix_requests_error_rather_than_mismap() {
    set_pool_env();
    let context = require_cuda();
    context.bind_to_thread().expect("bind CUDA context");
    let device = 0;
    let granule = granularity(device);

    let governor = LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0));
    let allocator = allocator(&governor, 64 << 20);
    let stats = allocator
        .physical_pool_stats()
        .expect("production allocator installs the pool");
    let prefix = allocator
        .create_shared_prefix(granule)
        .expect("pinned shared prefix");
    let bytes = granule * 2;
    write_host(prefix.device_ptr(), PREFIX_MARKER, PROBE_LEN);

    let healthy = allocate_sequence(&allocator, granule, 2);
    allocator
        .commit_shared_prefix(&prefix, healthy, bytes, 0)
        .expect("healthy peer map");

    // Misaligned offset: a shared prefix maps whole granules.
    let a = allocate_sequence(&allocator, granule, 2);
    let before = stats.snapshot();
    let used_before = governor.used(Tier::Device);
    let error = allocator
        .commit_shared_prefix(&prefix, a, bytes, granule / 2)
        .expect_err("a non-granule-aligned offset must error");
    assert!(
        error.to_string().contains("not granule-aligned"),
        "misalignment must be classified explicitly: {error}"
    );
    assert_eq!(stats.snapshot().mapped_bytes, before.mapped_bytes);
    assert_eq!(governor.used(Tier::Device), used_before);

    // Would not fit inside the allocation.
    let before = stats.snapshot();
    let used_before = governor.used(Tier::Device);
    let error = allocator
        .commit_shared_prefix(&prefix, a, bytes, bytes)
        .expect_err("a prefix that does not fit must error");
    assert!(
        error.to_string().contains("exceeds the allocation"),
        "overlong geometry must be classified explicitly: {error}"
    );
    assert_eq!(stats.snapshot().mapped_bytes, before.mapped_bytes);
    assert_eq!(governor.used(Tier::Device), used_before);

    // Overlay onto an already-committed granule: never overwrite live KV.
    let b = allocator
        .allocate_committed(bytes, granule, std::slice::from_ref(&(0..bytes)))
        .expect("fully committed sequence");
    let before = stats.snapshot();
    let used_before = governor.used(Tier::Device);
    let error = allocator
        .commit_shared_prefix(&prefix, b, bytes, 0)
        .expect_err("mapping over a committed granule must error");
    assert!(
        error.to_string().contains("already committed"),
        "overlap must be classified explicitly: {error}"
    );
    assert_eq!(stats.snapshot().mapped_bytes, before.mapped_bytes);
    assert_eq!(governor.used(Tier::Device), used_before);

    // A graph capture is declared open: mapping is not proven replayable.
    let c = allocate_sequence(&allocator, granule, 2);
    let before = stats.snapshot();
    let used_before = governor.used(Tier::Device);
    {
        let _capture = allocator.enter_graph_capture();
        let error = allocator
            .commit_shared_prefix(&prefix, c, bytes, 0)
            .expect_err("mapping a shared prefix while capture is open must error");
        assert!(
            error.to_string().contains("graph capture is open"),
            "capture-open refusal must be classified explicitly: {error}"
        );
    }
    assert_eq!(stats.snapshot().mapped_bytes, before.mapped_bytes);
    assert_eq!(governor.used(Tier::Device), used_before);
    // Once the capture guard lifts, the same map succeeds.
    allocator
        .commit_shared_prefix(&prefix, c, bytes, 0)
        .expect("mapping succeeds once the capture guard is gone");

    assert!(
        read_host(healthy.as_ptr() as u64, PROBE_LEN)
            .iter()
            .all(|&byte| byte == PREFIX_MARKER),
        "all request-local refusals must leave the healthy peer unchanged"
    );

    // An allocator without the production pool cannot express a shared prefix.
    let detached = CudaVmmAllocator::detached(
        CudaContext::new(0).expect("CUDA context"),
        DeviceKey::device(0),
        0,
        granule * 4,
        HOLDER,
        MemoryRole::KvCache,
    )
    .expect("detached allocator");
    assert!(
        detached.create_shared_prefix(granule).is_err(),
        "a shared prefix requires the production physical-handle pool"
    );
    assert!(
        DeviceAllocator::as_shared_mapping(&detached).is_none(),
        "a detached/pool-less allocator must not advertise SharedMapping"
    );
    assert!(
        DeviceAllocator::as_shared_mapping(&allocator).is_some(),
        "a pooled allocator must advertise SharedMapping"
    );

    // SAFETY: live pointers from this allocator, no CUDA work in flight.
    unsafe {
        allocator.deallocate(a, bytes, granule);
        allocator.deallocate(b, bytes, granule);
        allocator.deallocate(c, bytes, granule);
        allocator.deallocate(healthy, bytes, granule);
    }
    drop(prefix);
    drop(allocator);
    let torn_down = stats.snapshot();
    assert_eq!(torn_down.mapped_bytes, 0);
    assert_eq!(torn_down.pooled_unmapped_bytes, 0);
    assert_eq!(torn_down.total_owned_bytes, 0);
    assert_eq!(torn_down.creates, torn_down.releases);
    assert_eq!(governor.used(Tier::Device), 0);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn repeated_precommitted_fallback_reuses_without_leaking() {
    set_pool_env();
    let context = require_cuda();
    context.bind_to_thread().expect("bind CUDA context");
    let granule = granularity(0);

    let governor = LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0));
    let allocator = allocator(&governor, 64 << 20);
    let stats = allocator
        .physical_pool_stats()
        .expect("production allocator installs the pool");
    let prefix = allocator
        .create_shared_prefix(granule)
        .expect("pinned shared prefix");
    write_host(prefix.device_ptr(), PREFIX_MARKER, PROBE_LEN);

    let bytes = 2 * granule;
    let full = 0..bytes;
    let mut steady_state = None;
    for cycle in 0..16u8 {
        let private = allocator
            .allocate_committed(bytes, granule, std::slice::from_ref(&full))
            .expect("fully private fallback target");
        let before = stats.snapshot();
        let error = allocator
            .commit_shared_prefix(&prefix, private, bytes, 0)
            .expect_err("precommitted target must fall back");
        assert!(
            error.to_string().contains("already committed"),
            "cycle {cycle} must report the overlap classification: {error}"
        );
        assert_eq!(
            stats.snapshot(),
            before,
            "cycle {cycle} refusal must not add a mapping, handle, or shared reference"
        );

        let private_marker = cycle.wrapping_add(1);
        write_host(private.as_ptr() as u64, private_marker, PROBE_LEN);
        assert!(
            read_host(private.as_ptr() as u64, PROBE_LEN)
                .iter()
                .all(|&byte| byte == private_marker),
            "cycle {cycle} must continue over private KV"
        );
        assert!(
            read_host(prefix.device_ptr(), PROBE_LEN)
                .iter()
                .all(|&byte| byte == PREFIX_MARKER),
            "cycle {cycle} private fallback must not contaminate the shared prefix"
        );

        unsafe { allocator.deallocate(private, bytes, granule) };
        let after = stats.snapshot();
        assert_eq!(
            after.mapped_bytes, granule as u64,
            "only the shared-prefix owner window remains mapped after cycle {cycle}"
        );
        assert_eq!(after.quarantined_bytes, 0);
        assert_eq!(after.quarantined_handles, 0);
        if let Some(expected) = steady_state {
            assert_eq!(
                after.total_owned_bytes, expected,
                "repeated fallback must reuse retained handles rather than grow ownership"
            );
        } else {
            steady_state = Some(after.total_owned_bytes);
        }
        assert_eq!(
            governor.used(Tier::Device),
            after.total_owned_bytes,
            "ledger and physical-pool ownership must agree after cycle {cycle}"
        );
    }
    assert!(
        stats.snapshot().pool_hits >= 30,
        "cycles after the first must reuse both private-cache granules"
    );

    drop(prefix);
    drop(allocator);
    let torn_down = stats.snapshot();
    assert_eq!(torn_down.mapped_bytes, 0);
    assert_eq!(torn_down.pooled_unmapped_bytes, 0);
    assert_eq!(torn_down.total_owned_bytes, 0);
    assert_eq!(torn_down.quarantined_bytes, 0);
    assert_eq!(torn_down.creates, torn_down.releases);
    assert_eq!(governor.used(Tier::Device), 0);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn one_and_two_granule_prefix_boundaries_map_and_cleanup() {
    set_pool_env();
    let context = require_cuda();
    context.bind_to_thread().expect("bind CUDA context");
    let granule = granularity(0);

    let governor = LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0));
    let allocator = allocator(&governor, 64 << 20);
    let stats = allocator
        .physical_pool_stats()
        .expect("production allocator installs the pool");

    for prefix_granules in [1usize, 2] {
        let prefix_bytes = prefix_granules * granule;
        let tail_bytes = prefix_granules * granule;
        let bytes = prefix_bytes + tail_bytes;
        let marker = PREFIX_MARKER.wrapping_add(prefix_granules as u8);
        let prefix = allocator
            .create_shared_prefix(prefix_bytes)
            .expect("boundary prefix");
        write_host(prefix.device_ptr(), marker, prefix_bytes);
        let private_tail = prefix_bytes..bytes;
        let sequence = allocator
            .allocate_committed(bytes, granule, std::slice::from_ref(&private_tail))
            .expect("boundary sequence");
        let commit = allocator
            .commit_shared_prefix(&prefix, sequence, bytes, 0)
            .expect("boundary prefix map");
        assert_eq!(commit.granules, prefix_granules);
        assert_eq!(
            commit.newly_mapped_bytes, prefix_bytes as u64,
            "the exact prefix boundary must be mapped"
        );
        assert!(
            read_host(sequence.as_ptr() as u64, PROBE_LEN)
                .iter()
                .all(|&byte| byte == marker)
        );
        assert!(
            read_host(
                sequence.as_ptr() as u64 + prefix_bytes as u64 - PROBE_LEN as u64,
                PROBE_LEN,
            )
            .iter()
            .all(|&byte| byte == marker),
            "the final bytes of the {prefix_granules}-granule prefix must be shared"
        );

        unsafe { allocator.deallocate(sequence, bytes, granule) };
        drop(prefix);
        assert_eq!(stats.snapshot().quarantined_bytes, 0);
    }

    drop(allocator);
    let torn_down = stats.snapshot();
    assert_eq!(torn_down.mapped_bytes, 0);
    assert_eq!(torn_down.pooled_unmapped_bytes, 0);
    assert_eq!(torn_down.total_owned_bytes, 0);
    assert_eq!(torn_down.creates, torn_down.releases);
    assert_eq!(governor.used(Tier::Device), 0);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn foreign_device_and_authority_prefixes_are_not_free_and_are_rejected() {
    set_pool_env();
    let context = require_cuda();
    context.bind_to_thread().expect("bind CUDA context");
    let granule = granularity(0);

    let governor_a = LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0));
    let allocator_a = allocator(&governor_a, 64 << 20);
    let stats_a = allocator_a
        .physical_pool_stats()
        .expect("allocator a uses the production pool");
    let prefix = allocator_a
        .create_shared_prefix(granule)
        .expect("prefix on allocator a");
    write_host(prefix.device_ptr(), PREFIX_MARKER, PROBE_LEN);
    let healthy = allocate_sequence(&allocator_a, granule, 2);
    allocator_a
        .commit_shared_prefix(&prefix, healthy, 2 * granule, 0)
        .expect("healthy peer");

    let governor_b = LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0));
    let wrong_device = CudaVmmAllocator::new(
        context,
        DeviceKey::device(1),
        0,
        64 << 20,
        &governor_b,
        HOLDER,
        MemoryRole::KvCache,
    )
    .expect("logical device-one allocator");
    let stats_b = wrong_device
        .physical_pool_stats()
        .expect("wrong-device allocator uses a production pool");
    let before_b = stats_b.snapshot();
    let error = wrong_device
        .incremental_owned_bytes_for_shared_prefix(&prefix)
        .expect_err("a wrong-device prefix must be rejected before cost");
    assert!(
        error.to_string().contains("belongs to device"),
        "wrong-device cost refusal must be classified explicitly: {error}"
    );
    let error = wrong_device
        .commit_shared_prefix(&prefix, NonNull::dangling(), granule, 0)
        .expect_err("wrong-device mapping must be rejected");
    assert!(
        error.to_string().contains("belongs to device"),
        "wrong-device commit refusal must be classified explicitly: {error}"
    );
    assert_eq!(stats_b.snapshot(), before_b);
    assert_eq!(governor_b.used(Tier::Device), 0);

    let governor_c = LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0));
    let wrong_authority = allocator(&governor_c, 64 << 20);
    let stats_c = wrong_authority
        .physical_pool_stats()
        .expect("wrong-authority allocator uses a production pool");
    assert_ne!(
        allocator_a.physical_pool_authority(),
        wrong_authority.physical_pool_authority()
    );
    let before_c = stats_c.snapshot();
    let error = wrong_authority
        .incremental_owned_bytes_for_shared_prefix(&prefix)
        .expect_err("a wrong-authority prefix must be rejected before cost");
    assert!(
        error.to_string().contains("different pool authority"),
        "wrong-authority cost refusal must be classified explicitly: {error}"
    );
    let error = wrong_authority
        .commit_shared_prefix(&prefix, NonNull::dangling(), granule, 0)
        .expect_err("wrong-authority mapping must be rejected");
    assert!(
        error.to_string().contains("different pool authority"),
        "wrong-authority commit refusal must be classified explicitly: {error}"
    );
    assert_eq!(stats_c.snapshot(), before_c);
    assert_eq!(governor_c.used(Tier::Device), 0);

    assert!(
        read_host(healthy.as_ptr() as u64, PROBE_LEN)
            .iter()
            .all(|&byte| byte == PREFIX_MARKER),
        "foreign-device and foreign-authority refusals must not disturb a healthy peer"
    );

    unsafe { allocator_a.deallocate(healthy, 2 * granule, granule) };
    drop(prefix);
    drop(wrong_authority);
    drop(wrong_device);
    drop(allocator_a);
    assert_eq!(stats_a.snapshot().total_owned_bytes, 0);
    assert_eq!(stats_b.snapshot().total_owned_bytes, 0);
    assert_eq!(stats_c.snapshot().total_owned_bytes, 0);
    assert_eq!(governor_a.used(Tier::Device), 0);
    assert_eq!(governor_b.used(Tier::Device), 0);
    assert_eq!(governor_c.used(Tier::Device), 0);
}

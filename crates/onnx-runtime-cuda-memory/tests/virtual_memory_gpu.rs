//! Does CUDA virtual memory actually give one contiguous device address over
//! separate physical allocations?
//!
//! That is the property the whole approach rests on — it is what lets
//! `GroupQueryAttention` read a paged KV cache without anyone copying it into a
//! flat buffer first. It is also a claim about the CUDA driver, not about our
//! code, so it is checked against a real device rather than assumed.
//!
//! CPU-only CI reports this as ignored unless `gpu-tests` is enabled; an
//! enabled run fails if no GPU is present.

use std::sync::Arc;

use cudarc::driver::CudaContext;
use onnx_runtime_cuda_memory::virtual_memory::{
    CudaVirtualBacking, PhysicalHandlePool, physical_pool_authority_gate,
    trim_physical_handle_pools,
};
use onnx_runtime_memory_governor::{
    HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor, MemoryRole, Tier,
};
use onnx_runtime_virtual_memory::{VirtualBacking, VirtualBuffer};

const HOLDER: HolderId = HolderId::new(11);

/// A CUDA context, or `None` on a machine with no driver.
///
/// Uses the **driver** API only — `cuInit` and a primary-context retain. The
/// execution provider additionally needs cudart and cuBLAS, and requiring those
/// here would make these tests skip on a machine that can run them perfectly
/// well, which is how a suite ends up green while proving nothing.
fn require_cuda_backing() -> CudaVirtualBacking {
    match CudaContext::new(0) {
        Ok(context) => CudaVirtualBacking::new(context, 0),
        Err(error) => panic!(
            "CUDA virtual-memory test requires a CUDA driver; CPU-only runs must leave this test ignored: {error}"
        ),
    }
}

/// The driver reports a usable granularity, and it is a power of two.
///
/// Everything else rounds to this, so a wrong answer here misaligns every
/// subsequent request and the driver rejects them.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn the_device_reports_a_sane_allocation_granularity() {
    let backing = require_cuda_backing();
    let granularity = backing.granularity();
    assert!(granularity > 0, "granularity must be positive");
    assert!(
        granularity.is_power_of_two(),
        "granularity {granularity} is not a power of two, which every offset \
         calculation assumes"
    );
    eprintln!("CUDA allocation granularity: {granularity} bytes");
}

/// Reserving device address space must not consume device memory.
///
/// The design reserves generously — for the largest a KV buffer could ever be —
/// and that is only safe if reserving is free. A reservation far larger than
/// the card's 8 GiB proves it is address space, not memory.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn reserving_far_more_than_vram_succeeds_because_nothing_is_committed() {
    let backing = require_cuda_backing();
    let huge = 64usize << 30; // 64 GiB, well past any consumer card
    let reservation = backing
        .reserve(huge)
        .expect("reserving device address space must not need device memory");
    assert_ne!(CudaVirtualBacking::base(&reservation), 0);
    drop(reservation);
}

/// **The claim.** Two separate physical allocations, one contiguous address,
/// and a write that crosses the seam between them reads back correctly.
///
/// If this fails, the whole "no copy for GroupQueryAttention" plan fails with
/// it — so it writes *across* the join rather than into each block, which is
/// the only pattern that distinguishes real contiguity from two buffers that
/// happen to be adjacent in a table.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn one_address_spans_two_separate_physical_allocations() {
    let backing = require_cuda_backing();
    let granule = backing.granularity();

    let mut reservation = backing.reserve(granule * 4).expect("address space");
    backing
        .commit(&mut reservation, 0, granule)
        .expect("first granule");
    backing
        .commit(&mut reservation, granule, granule)
        .expect("second granule, from a different physical allocation");

    let base = CudaVirtualBacking::base(&reservation);
    let span = granule * 2;
    let pattern: Vec<u8> = (0..span).map(|index| (index % 251) as u8).collect();

    // One copy across both granules. A device address that only looked
    // contiguous would fault or truncate here.
    unsafe {
        use cudarc::driver::sys as cu;
        let write = cu::cuMemcpyHtoD_v2(base as cu::CUdeviceptr, pattern.as_ptr().cast(), span);
        assert_eq!(
            write,
            cu::CUresult::CUDA_SUCCESS,
            "host-to-device copy across the seam"
        );

        let mut read_back = vec![0u8; span];
        let read =
            cu::cuMemcpyDtoH_v2(read_back.as_mut_ptr().cast(), base as cu::CUdeviceptr, span);
        assert_eq!(
            read,
            cu::CUresult::CUDA_SUCCESS,
            "device-to-host copy across the seam"
        );

        assert_eq!(
            read_back, pattern,
            "the bytes written across the boundary between two physical \
             allocations did not read back; the address range is not really \
             contiguous"
        );
    }
}

/// A `VirtualBuffer` over the CUDA backing grows without moving.
///
/// The stable address is what keeps a captured CUDA graph valid across growth,
/// which is the reason to do any of this rather than reallocating.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn a_device_buffer_grows_without_moving() {
    let backing = require_cuda_backing();
    let granule = backing.granularity();
    let governor = LedgerGovernor::new(LeaseLedger::new(64 << 20, 0, 0));

    let mut buffer = VirtualBuffer::with_backing(
        backing,
        granule * 8,
        Arc::new(governor.clone()),
        Tier::Device,
        MemoryRole::KvCache,
        HOLDER,
    )
    .expect("device address space");

    let base = buffer.as_ptr();
    assert_eq!(buffer.committed(), 0, "reserving must commit nothing");
    assert_eq!(
        governor.available(Tier::Device),
        64 << 20,
        "reserving device address space must not lease device memory"
    );

    for step in 1..=4usize {
        buffer.grow_to(granule * step).expect("within capacity");
        assert_eq!(
            buffer.as_ptr(),
            base,
            "growing to {step} granules moved the device buffer"
        );
    }
    assert_eq!(
        (64u64 << 20) - governor.available(Tier::Device),
        buffer.committed() as u64,
        "the governor must be charged exactly the committed device bytes"
    );

    buffer.shrink_to(0).expect("shrunk");
    assert_eq!(
        governor.available(Tier::Device),
        64 << 20,
        "shrinking must return the device memory to the governor"
    );
}

/// A multi-granule growth is recorded as independently releasable blocks.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn a_multi_granule_growth_can_shrink_one_granule() {
    let backing = require_cuda_backing();
    let granule = backing.granularity();
    let governor = LedgerGovernor::new(LeaseLedger::new(64 << 20, 0, 0));
    let mut buffer = VirtualBuffer::with_backing(
        backing,
        granule * 2,
        Arc::new(governor.clone()),
        Tier::Device,
        MemoryRole::KvCache,
        HOLDER,
    )
    .expect("device address space");

    buffer.grow_to(granule * 2).expect("two-granule commit");
    buffer.shrink_to(granule).expect("release upper granule");
    assert_eq!(buffer.committed(), granule);
    assert_eq!(
        (64u64 << 20) - governor.available(Tier::Device),
        granule as u64
    );
    buffer
        .grow_to(granule * 2)
        .expect("released granule can be mapped again");
}

/// Granule handles move between independently reserved ranges without a
/// create/release cycle, while the governor remains charged for every owned
/// physical byte.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn pooled_handles_move_between_separate_reservations() {
    let context = CudaContext::new(0).expect("CUDA driver");
    let governor = LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0));
    let probe = CudaVirtualBacking::new(Arc::clone(&context), 0);
    let granule = probe.granularity();
    let count = 3usize;
    let bytes = granule * count;
    let pool = PhysicalHandlePool::get_or_create(
        context,
        0,
        bytes,
        &governor,
        HOLDER,
        MemoryRole::KvCache,
    )
    .expect("pool lease");
    let stats = pool.stats();
    let backing = CudaVirtualBacking::with_physical_pool(Arc::clone(&pool));
    let mut a = backing.reserve(bytes).expect("range A");
    let mut b = backing.reserve(bytes).expect("range B");

    backing.commit(&mut a, 0, bytes).expect("map A");
    let after_a = stats.snapshot();
    assert_eq!(after_a.creates, count as u64);
    assert_eq!(after_a.releases, 0);
    assert_eq!(after_a.mapped_bytes, bytes as u64);
    assert_eq!(after_a.pooled_unmapped_bytes, 0);
    assert_eq!(after_a.total_owned_bytes, bytes as u64);
    assert_eq!(
        (8u64 << 30) - governor.available(Tier::Device),
        bytes as u64,
        "the governor lease covers all pool-owned VRAM"
    );

    backing.release(&mut a, 0, bytes).expect("unmap A");
    let pooled = stats.snapshot();
    assert_eq!(pooled.creates, after_a.creates);
    assert_eq!(pooled.releases, 0);
    assert_eq!(pooled.mapped_bytes, 0);
    assert_eq!(pooled.pooled_unmapped_bytes, bytes as u64);
    assert_eq!(pooled.total_owned_bytes, bytes as u64);
    assert_eq!(
        (8u64 << 30) - governor.available(Tier::Device),
        bytes as u64,
        "unmapping must not report pool-owned VRAM as free"
    );

    backing.commit(&mut b, 0, bytes).expect("map B");
    let after_b = stats.snapshot();
    assert_eq!(
        after_b.creates, after_a.creates,
        "mapping B must not create replacement handles"
    );
    assert_eq!(
        after_b.releases, 0,
        "transferring handles A to B must not release physical memory"
    );
    assert_eq!(after_b.pool_hits, count as u64);
    assert_eq!(after_b.mapped_bytes, bytes as u64);
    assert_eq!(after_b.pooled_unmapped_bytes, 0);
    assert_eq!(after_b.total_owned_bytes, bytes as u64);

    let base = CudaVirtualBacking::base(&b);
    let pattern: Vec<u8> = (0..bytes).map(|index| (index % 251) as u8).collect();
    unsafe {
        use cudarc::driver::sys as cu;
        assert_eq!(
            cu::cuMemcpyHtoD_v2(base as cu::CUdeviceptr, pattern.as_ptr().cast(), bytes),
            cu::CUresult::CUDA_SUCCESS
        );
        let mut read_back = vec![0u8; bytes];
        assert_eq!(
            cu::cuMemcpyDtoH_v2(
                read_back.as_mut_ptr().cast(),
                base as cu::CUdeviceptr,
                bytes
            ),
            cu::CUresult::CUDA_SUCCESS
        );
        assert_eq!(read_back, pattern);
    }

    backing.release(&mut b, 0, bytes).expect("unmap B");
    drop(a);
    drop(b);
    drop(backing);
    drop(pool);

    let torn_down = stats.snapshot();
    assert_eq!(torn_down.creates, count as u64);
    assert_eq!(
        torn_down.releases, count as u64,
        "pool teardown releases each created handle exactly once"
    );
    assert_eq!(torn_down.mapped_bytes, 0);
    assert_eq!(torn_down.pooled_unmapped_bytes, 0);
    assert_eq!(torn_down.total_owned_bytes, 0);
    assert_eq!(governor.available(Tier::Device), 8 << 30);
    eprintln!(
        "physical pool transfer: creates={} releases={} hits={} mapped={} pooled={} owned={}",
        torn_down.creates,
        torn_down.releases,
        torn_down.pool_hits,
        torn_down.mapped_bytes,
        torn_down.pooled_unmapped_bytes,
        torn_down.total_owned_bytes
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn lowering_an_authority_can_trim_unmapped_pool_handles() {
    let context = CudaContext::new(0).expect("CUDA driver");
    let governor = LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0));
    let granule = CudaVirtualBacking::new(Arc::clone(&context), 0).granularity();
    let pool = PhysicalHandlePool::get_or_create(
        context,
        0,
        granule * 2,
        &governor,
        HOLDER,
        MemoryRole::KvCache,
    )
    .expect("pool");
    let stats = pool.stats();
    let backing = CudaVirtualBacking::with_physical_pool(pool);
    let mut reservation = backing.reserve(granule * 2).expect("reservation");
    backing
        .commit(&mut reservation, 0, granule * 2)
        .expect("commit");
    backing
        .release(&mut reservation, 0, granule * 2)
        .expect("return");

    assert_eq!(
        trim_physical_handle_pools(governor.authority_id(), granule as u64)
            .expect("trim one granule"),
        granule as u64
    );
    let trimmed = stats.snapshot();
    assert_eq!(trimmed.total_owned_bytes, granule as u64);
    assert_eq!(trimmed.pooled_unmapped_bytes, granule as u64);
    assert_eq!(trimmed.releases, 1);
    assert_eq!(
        (8u64 << 30) - governor.available(Tier::Device),
        granule as u64
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn failed_teardown_synchronization_never_reuses_or_releases_a_live_handle() {
    use std::sync::atomic::{AtomicU64, Ordering};

    let context = CudaContext::new(0).expect("CUDA driver");
    let governor = LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0));
    let granule = CudaVirtualBacking::new(Arc::clone(&context), 0).granularity();
    let pool = PhysicalHandlePool::get_or_create(
        context,
        0,
        granule,
        &governor,
        HOLDER,
        MemoryRole::KvCache,
    )
    .expect("pool");
    let stats = pool.stats();
    let sync_attempts = Arc::new(AtomicU64::new(0));
    let attempts = Arc::clone(&sync_attempts);
    let backing = CudaVirtualBacking::with_physical_pool(Arc::clone(&pool))
        .with_teardown_synchronizer(Arc::new(move || {
            attempts.fetch_add(1, Ordering::Relaxed);
            Err(String::from("injected synchronization failure"))
        }));
    let mut reservation = backing.reserve(granule).expect("reservation");
    backing
        .commit(&mut reservation, 0, granule)
        .expect("commit");
    drop(reservation);

    let snapshot = stats.snapshot();
    assert_eq!(sync_attempts.load(Ordering::Relaxed), 1);
    assert_eq!(snapshot.creates, 1);
    assert_eq!(snapshot.releases, 0);
    assert_eq!(snapshot.pool_hits, 0);
    assert_eq!(snapshot.mapped_bytes, granule as u64);
    assert_eq!(snapshot.pooled_unmapped_bytes, 0);
    assert_eq!(snapshot.total_owned_bytes, granule as u64);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn successful_teardown_synchronization_allows_later_reuse() {
    use std::sync::atomic::{AtomicU64, Ordering};

    let context = CudaContext::new(0).expect("CUDA driver");
    let governor = LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0));
    let granule = CudaVirtualBacking::new(Arc::clone(&context), 0).granularity();
    let pool = PhysicalHandlePool::get_or_create(
        context,
        0,
        granule,
        &governor,
        HOLDER,
        MemoryRole::KvCache,
    )
    .expect("pool");
    let stats = pool.stats();
    let sync_attempts = Arc::new(AtomicU64::new(0));
    let attempts = Arc::clone(&sync_attempts);
    let backing = CudaVirtualBacking::with_physical_pool(pool).with_teardown_synchronizer(
        Arc::new(move || {
            attempts.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }),
    );
    let mut first = backing.reserve(granule).expect("first reservation");
    backing
        .commit(&mut first, 0, granule)
        .expect("first commit");
    drop(first);
    let mut second = backing.reserve(granule).expect("second reservation");
    backing
        .commit(&mut second, 0, granule)
        .expect("second commit");

    assert_eq!(sync_attempts.load(Ordering::Relaxed), 1);
    assert_eq!(stats.snapshot().creates, 1);
    assert_eq!(stats.snapshot().pool_hits, 1);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn authority_write_gate_freezes_pool_checkout_and_return() {
    use std::{
        sync::{Barrier, mpsc},
        time::Duration,
    };

    let context = CudaContext::new(0).expect("CUDA driver");
    let governor = LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0));
    let granule = CudaVirtualBacking::new(Arc::clone(&context), 0).granularity();
    let pool = PhysicalHandlePool::get_or_create(
        context,
        0,
        granule,
        &governor,
        HOLDER,
        MemoryRole::KvCache,
    )
    .expect("pool");
    let stats = pool.stats();
    let backing = CudaVirtualBacking::with_physical_pool(pool);
    let mut first = backing.reserve(granule).expect("first reservation");
    backing
        .commit(&mut first, 0, granule)
        .expect("first commit");
    backing.release(&mut first, 0, granule).expect("seed pool");

    let gate = physical_pool_authority_gate(governor.authority_id());
    let writer = gate
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let barrier = Arc::new(Barrier::new(2));
    let worker_barrier = Arc::clone(&barrier);
    let worker_backing = backing.clone();
    let (send, receive) = mpsc::channel();
    let checkout = std::thread::spawn(move || {
        let mut reservation = worker_backing.reserve(granule).expect("reservation");
        worker_barrier.wait();
        worker_backing
            .commit(&mut reservation, 0, granule)
            .expect("checkout after transaction");
        send.send(reservation).expect("return reservation");
    });
    barrier.wait();
    assert!(
        receive.recv_timeout(Duration::from_millis(50)).is_err(),
        "checkout must wait while reconfiguration owns the write gate"
    );
    let before = stats.snapshot();
    drop(writer);
    let mut checked_out = receive
        .recv_timeout(Duration::from_secs(5))
        .expect("checkout completes after transaction");
    checkout.join().expect("checkout thread");
    assert_eq!(stats.snapshot().pool_hits, before.pool_hits + 1);

    let writer = gate
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let barrier = Arc::new(Barrier::new(2));
    let worker_barrier = Arc::clone(&barrier);
    let worker_backing = backing.clone();
    let (send, receive) = mpsc::channel();
    let returned = std::thread::spawn(move || {
        worker_barrier.wait();
        worker_backing
            .release(&mut checked_out, 0, granule)
            .expect("return after transaction");
        send.send(()).expect("return completion");
    });
    barrier.wait();
    assert!(
        receive.recv_timeout(Duration::from_millis(50)).is_err(),
        "return must wait while reconfiguration owns the write gate"
    );
    assert_eq!(stats.snapshot().pooled_unmapped_bytes, 0);
    drop(writer);
    receive
        .recv_timeout(Duration::from_secs(5))
        .expect("return completes after transaction");
    returned.join().expect("return thread");
    assert_eq!(stats.snapshot().pooled_unmapped_bytes, granule as u64);

    let writer = gate
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let before_trim = stats.snapshot();
    assert_eq!(
        trim_physical_handle_pools(governor.authority_id(), granule as u64)
            .expect("transaction trim"),
        granule as u64
    );
    let barrier = Arc::new(Barrier::new(2));
    let worker_barrier = Arc::clone(&barrier);
    let worker_backing = backing.clone();
    let (send, receive) = mpsc::channel();
    let post_trim_checkout = std::thread::spawn(move || {
        let mut reservation = worker_backing.reserve(granule).expect("reservation");
        worker_barrier.wait();
        worker_backing
            .commit(&mut reservation, 0, granule)
            .expect("post-trim checkout");
        send.send(()).expect("checkout completion");
    });
    barrier.wait();
    assert!(receive.recv_timeout(Duration::from_millis(50)).is_err());
    drop(writer);
    receive
        .recv_timeout(Duration::from_secs(5))
        .expect("post-trim checkout completes");
    post_trim_checkout.join().expect("checkout thread");
    let after_trim_checkout = stats.snapshot();
    assert_eq!(after_trim_checkout.pool_hits, before_trim.pool_hits);
    assert_eq!(after_trim_checkout.creates, before_trim.creates + 1);
}

/// The pool retains at most its configured whole-granule bound.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn physical_pool_releases_handles_above_its_bound() {
    let context = CudaContext::new(0).expect("CUDA driver");
    let governor = LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0));
    let probe = CudaVirtualBacking::new(Arc::clone(&context), 0);
    let granule = probe.granularity();
    let pool = PhysicalHandlePool::get_or_create(
        context,
        0,
        granule,
        &governor,
        HOLDER,
        MemoryRole::KvCache,
    )
    .expect("pool lease");
    let stats = pool.stats();
    let backing = CudaVirtualBacking::with_physical_pool(Arc::clone(&pool));
    let mut reservation = backing.reserve(granule * 2).expect("range");

    backing
        .commit(&mut reservation, 0, granule * 2)
        .expect("two handles");
    backing
        .release(&mut reservation, 0, granule * 2)
        .expect("return handles");

    let bounded = stats.snapshot();
    assert_eq!(pool.max_retained_bytes(), granule);
    assert_eq!(bounded.creates, 2);
    assert_eq!(bounded.releases, 1);
    assert_eq!(bounded.pooled_unmapped_bytes, granule as u64);
    assert_eq!(bounded.total_owned_bytes, granule as u64);
    assert_eq!(
        (8u64 << 30) - governor.available(Tier::Device),
        granule as u64
    );
}

/// Different accounting authorities never exchange physical handles.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn different_authorities_use_different_pools() {
    let context = CudaContext::new(0).expect("CUDA driver");
    let first_governor = LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0));
    let second_governor = LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0));
    let granule = CudaVirtualBacking::new(Arc::clone(&context), 0).granularity();
    let first = PhysicalHandlePool::get_or_create(
        Arc::clone(&context),
        0,
        granule,
        &first_governor,
        HOLDER,
        MemoryRole::KvCache,
    )
    .expect("first authority pool");
    let second = PhysicalHandlePool::get_or_create(
        context,
        0,
        granule,
        &second_governor,
        HOLDER,
        MemoryRole::KvCache,
    )
    .expect("second authority pool");
    assert_ne!(first.authority(), second.authority());

    let first_backing = CudaVirtualBacking::with_physical_pool(Arc::clone(&first));
    let second_backing = CudaVirtualBacking::with_physical_pool(Arc::clone(&second));
    let mut a = first_backing.reserve(granule).expect("range A");
    let mut b = second_backing.reserve(granule).expect("range B");
    first_backing.commit(&mut a, 0, granule).expect("map A");
    first_backing.release(&mut a, 0, granule).expect("return A");
    second_backing.commit(&mut b, 0, granule).expect("map B");

    assert_eq!(first.stats().snapshot().creates, 1);
    assert_eq!(second.stats().snapshot().creates, 1);
    assert_eq!(
        second.stats().snapshot().pool_hits,
        0,
        "authority B must not reuse authority A's retained handle"
    );
}

/// `VirtualBuffer` delegates accounting to a pooled backing instead of taking
/// a second lease for the same physical handle.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn pooled_virtual_buffer_is_not_double_charged() {
    let context = CudaContext::new(0).expect("CUDA driver");
    let governor = LedgerGovernor::new(LeaseLedger::new(8 << 30, 0, 0));
    let granule = CudaVirtualBacking::new(Arc::clone(&context), 0).granularity();
    let pool = PhysicalHandlePool::get_or_create(
        context,
        0,
        granule,
        &governor,
        HOLDER,
        MemoryRole::KvCache,
    )
    .expect("pool lease");
    let backing = CudaVirtualBacking::with_physical_pool(Arc::clone(&pool));
    let mut buffer = VirtualBuffer::with_backing(
        backing,
        granule,
        Arc::new(governor.clone()),
        Tier::Device,
        MemoryRole::KvCache,
        HOLDER,
    )
    .expect("buffer reservation");

    buffer.grow_to(1).expect("map one granule");
    assert_eq!(
        (8u64 << 30) - governor.available(Tier::Device),
        granule as u64,
        "the pool lease, not a pool lease plus a buffer lease, covers the handle"
    );
    buffer.shrink_to(0).expect("return handle to pool");
    assert_eq!(
        (8u64 << 30) - governor.available(Tier::Device),
        granule as u64,
        "the retained unmapped handle remains charged"
    );
}

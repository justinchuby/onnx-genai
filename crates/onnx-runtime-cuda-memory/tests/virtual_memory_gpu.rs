//! Does CUDA virtual memory actually give one contiguous device address over
//! separate physical allocations?
//!
//! That is the property the whole approach rests on — it is what lets
//! `GroupQueryAttention` read a paged KV cache without anyone copying it into a
//! flat buffer first. It is also a claim about the CUDA driver, not about our
//! code, so it is checked against a real device rather than assumed.
//!
//! Skips when no GPU is present, in the same way the other `*_gpu` tests do.

use std::sync::Arc;

use cudarc::driver::CudaContext;
use onnx_runtime_cuda_memory::virtual_memory::CudaVirtualBacking;
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
fn backing() -> Option<CudaVirtualBacking> {
    match CudaContext::new(0) {
        Ok(context) => Some(CudaVirtualBacking::new(context, 0)),
        Err(error) => {
            // Loud on purpose: a skip that reads like a pass is worse than a
            // failure, because nobody investigates it.
            eprintln!(
                "SKIPPED (no CUDA driver): {error}. These tests verify device \
                 virtual memory and did NOT run."
            );
            None
        }
    }
}

/// The driver reports a usable granularity, and it is a power of two.
///
/// Everything else rounds to this, so a wrong answer here misaligns every
/// subsequent request and the driver rejects them.
#[test]
fn the_device_reports_a_sane_allocation_granularity() {
    let Some(backing) = backing() else { return };
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
#[test]
fn reserving_far_more_than_vram_succeeds_because_nothing_is_committed() {
    let Some(backing) = backing() else { return };
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
#[test]
fn one_address_spans_two_separate_physical_allocations() {
    let Some(backing) = backing() else { return };
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
#[test]
fn a_device_buffer_grows_without_moving() {
    let Some(backing) = backing() else { return };
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

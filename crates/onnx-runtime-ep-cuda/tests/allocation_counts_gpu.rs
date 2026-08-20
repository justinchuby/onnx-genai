#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::unusual_byte_groupings,
    clippy::doc_lazy_continuation,
    clippy::uninlined_format_args,
    clippy::cloned_ref_to_slice_refs,
    clippy::type_complexity,
    clippy::drop_non_drop,
    clippy::manual_repeat_n,
    clippy::manual_is_multiple_of,
    clippy::err_expect,
    clippy::clone_on_copy
)]
//! Allocations this execution provider makes stay visible in the count it
//! reports.
//!
//! Roughly twenty-five tests assert that a warmed, capture-safe path performs no
//! further allocations, and they assert it by reading
//! `device_allocation_counts`. That makes the counter a load-bearing part of the
//! capture-safety contract rather than telemetry.
//!
//! It is also the kind of assertion that fails silently in the wrong direction:
//! if the counter stops observing a path, every one of those tests becomes
//! `0 == 0` and stays green. That is exactly what happened when EP buffers moved
//! from `CudaRuntime::alloc_raw` to the replaceable `DeviceAllocator` seam, and
//! nothing went red.
//!
//! So this asserts the counter *moves*, which is the direction no other test
//! covers.
//!
//! Needs a real GPU. Skips loudly when there is none.

use onnx_runtime_ep_api::ExecutionProvider;
use onnx_runtime_ep_cuda::CudaExecutionProvider;

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn an_execution_provider_allocation_is_visible_in_the_reported_counts() {
    let Ok(ep) = CudaExecutionProvider::new(0) else {
        eprintln!(
            "SKIPPED (no CUDA runtime): the allocation-counter contract did NOT run. A skip that \
             reads like a pass is how the counter stopped observing anything in the first place."
        );
        panic!(
            "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
        );
    };

    let before = ep
        .device_allocation_counts()
        .expect("the CUDA EP reports allocation counts");

    let buffer = ep.allocate(1 << 20, 256).expect("a device allocation");
    let after_allocate = ep
        .device_allocation_counts()
        .expect("counts remain available");
    assert_eq!(
        after_allocate.0,
        before.0 + 1,
        "allocating through the EP must move the allocation count it reports"
    );
    assert_eq!(
        after_allocate.1, before.1,
        "allocating must not move the free count"
    );

    ep.deallocate(buffer).expect("the buffer is returned");
    // The free is ordered behind both stream tails now, so the count moves when
    // the release actually completes rather than when `deallocate` returns.
    // Waiting on the queue is what makes the assertion observe the real event
    // instead of a hopeful one.
    assert!(
        ep.release_queue()
            .wait_until_idle(std::time::Duration::from_secs(30)),
        "the deferred release must complete: {:?}",
        ep.deferred_release_stats()
    );
    let after_free = ep
        .device_allocation_counts()
        .expect("counts remain available");
    assert_eq!(
        after_free.1,
        before.1 + 1,
        "a completed release through the EP must move the free count it reports"
    );
    assert_eq!(
        after_free.0, after_allocate.0,
        "freeing must not move the allocation count"
    );
}

/// A borrowed buffer is not counted, because it was never allocated here.
///
/// `deallocate` returns early for one. Counting it would make the frees exceed
/// the allocations and turn a leak check into noise.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn a_buffer_that_was_never_allocated_here_does_not_move_the_free_count() {
    let Ok(ep) = CudaExecutionProvider::new(0) else {
        eprintln!("SKIPPED (no CUDA runtime): the borrowed-buffer count check did NOT run.");
        panic!(
            "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
        );
    };

    let buffer = ep.allocate(4096, 256).expect("a device allocation");
    let before = ep.device_allocation_counts().expect("counts");
    // A second handle onto the same device memory, marked borrowed. It owns
    // nothing, so `deallocate` returns early without touching the allocator.
    // SAFETY: `buffer` names a live device allocation of 4096 bytes on this
    // device and outlives `borrowed`; nothing is written through either handle,
    // and `borrowed` is released before `buffer` is.
    let borrowed = unsafe {
        onnx_runtime_ep_api::DeviceBuffer::from_borrowed_parts(
            buffer.as_ptr().cast_mut(),
            buffer.device(),
            buffer.len(),
            buffer.alignment(),
        )
    };
    ep.deallocate(borrowed).expect("a borrowed view is a no-op");
    let after = ep.device_allocation_counts().expect("counts");
    assert_eq!(
        after, before,
        "a borrowed view owns nothing, so releasing it must not count as a free"
    );

    ep.deallocate(buffer).expect("the real buffer is returned");
}

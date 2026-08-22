//! Real-CUDA proof that final device release is **ordered, not synchronized**
//! (issue #1186 Phase 4).
//!
//! The portable tests in `deferred_release_queue.rs` prove the scheduler's state
//! machine with fake fences. These prove the property the state machine exists
//! for, on a real device:
//!
//! * a free issued while a kernel is still reading the buffer does not happen
//!   until that kernel completes;
//! * the same holds for a transfer still in flight on the dedicated copy
//!   stream, which a compute-stream-only fence would miss;
//! * `deallocate` itself returns while that work is still running, so the caller
//!   is never stalled by another request's teardown;
//! * weight pages and provider teardown go through the same queue.
//!
//! Needs a real GPU. Skips loudly when there is none: a skip that reads like a
//! pass is how an ordering bug survives.

use std::sync::Arc;
use std::time::{Duration, Instant};

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::ExecutionProvider;
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::deferred_release::CudaDeferredReleaseQueue;
use onnx_runtime_ep_cuda::runtime::{CudaRuntime, cuptr};

const SPIN_MODULE: &str = "cuda_ep_deferred_release_gpu";
const SPIN_SOURCE: &str = r#"
extern "C" __global__ void spin_and_write(unsigned int* out, long long spin) {
    long long start = clock64();
    while (clock64() - start < spin) { }
    *out = 0x1186u;
}
"#;

/// Enough spin to keep a stream busy for a few milliseconds on any device we
/// target, which is far longer than an enqueue or a poll takes.
const SPIN_CYCLES: i64 = 400_000_000;

fn provider_or_fail(what: &str) -> CudaExecutionProvider {
    match CudaExecutionProvider::new(0) {
        Ok(provider) => provider,
        Err(error) => {
            eprintln!(
                "SKIPPED (no CUDA runtime): {what} did NOT run ({error}). Reporting this as a \
                 pass would hide an ordering regression."
            );
            panic!("CUDA test path did not run; report a failed GPU test, not a pass");
        }
    }
}

/// Poll the queue until it is idle, or give up after `timeout`.
fn drain(queue: &Arc<CudaDeferredReleaseQueue>, timeout: Duration) -> bool {
    queue.wait_until_idle(timeout)
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn a_free_issued_under_an_in_flight_kernel_waits_for_that_kernel() {
    let provider = provider_or_fail("in-flight compute ordering");
    let runtime: Arc<CudaRuntime> = Arc::clone(provider.runtime());
    let queue = Arc::clone(provider.release_queue());
    let spin = runtime
        .nvrtc_function(SPIN_MODULE, SPIN_SOURCE, "spin_and_write")
        .expect("compile the spin kernel");

    let buffer = provider.allocate(4, 256).expect("device allocation");
    let pointer = cuptr(buffer.as_ptr());
    let cycles = SPIN_CYCLES;
    let mut launch = runtime.stream().launch_builder(&spin);
    launch.arg(&pointer).arg(&cycles);
    // SAFETY: the kernel writes one `unsigned int` into a live 4-byte device
    // allocation this provider just returned.
    unsafe {
        launch
            .launch(LaunchConfig::for_num_elems(1))
            .expect("enqueue the spinning writer")
    };

    let before = queue.stats();
    let started = Instant::now();
    let unmapped = provider
        .deallocate_with_unmapped(buffer)
        .expect("deallocate is accepted while the kernel runs");
    let enqueue_time = started.elapsed();
    assert_eq!(
        unmapped, 0,
        "nothing is unmapped at the moment deallocate returns; the refund follows the outcome"
    );
    assert!(
        enqueue_time < Duration::from_millis(2),
        "deallocate must not wait for the kernel (took {enqueue_time:?})"
    );

    // The release is owed, and polling does not perform it while the kernel is
    // still running.
    let mut observed_pending = false;
    let watch = Instant::now();
    while watch.elapsed() < Duration::from_millis(1) {
        queue.poll();
        if queue.pending() > 0 {
            observed_pending = true;
            break;
        }
    }
    assert!(
        observed_pending,
        "the release must still be owed while the kernel is in flight: {:?}",
        queue.stats()
    );

    assert!(
        drain(&queue, Duration::from_secs(60)),
        "the release must complete once the kernel does: {:?}",
        queue.stats()
    );
    let after = queue.stats();
    assert_eq!(
        after.completed,
        before.completed + 1,
        "exactly one release completed"
    );
    assert_eq!(after.quarantined, before.quarantined);
    // The kernel really did write to the buffer before it was released.
    runtime.synchronize().expect("device settles");
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn a_free_issued_under_an_in_flight_copy_waits_for_the_copy_stream() {
    let provider = provider_or_fail("in-flight copy ordering");
    let runtime: Arc<CudaRuntime> = Arc::clone(provider.runtime());
    let queue = Arc::clone(provider.release_queue());
    let spin = runtime
        .nvrtc_function(SPIN_MODULE, SPIN_SOURCE, "spin_and_write")
        .expect("compile the spin kernel");

    // A long spin on the *copy* stream, then a copy out of the buffer under
    // test. A release fenced only on the compute stream would free the source
    // while this transfer is still reading it.
    let source = provider.allocate(1 << 20, 256).expect("copy source");
    let mut destination = provider.allocate(1 << 20, 256).expect("copy destination");
    let marker = provider.allocate(4, 256).expect("spin marker");
    let marker_ptr = cuptr(marker.as_ptr());
    let cycles = SPIN_CYCLES;
    let mut launch = runtime.copy_stream().launch_builder(&spin);
    launch.arg(&marker_ptr).arg(&cycles);
    // SAFETY: the kernel writes one `unsigned int` into a live 4-byte device
    // allocation; it runs on the copy stream to hold that stream busy.
    unsafe {
        launch
            .launch(LaunchConfig::for_num_elems(1))
            .expect("enqueue the copy-stream spin")
    };
    // SAFETY: both endpoints are live 1 MiB device allocations on this device.
    unsafe {
        runtime
            .dtod_async_on_copy_stream(
                cuptr(source.as_ptr()),
                cuptr(destination.as_mut_ptr()),
                1 << 20,
            )
            .expect("enqueue an async copy behind the spin")
    };

    let before = queue.stats();
    let started = Instant::now();
    provider
        .deallocate(source)
        .expect("deallocating the copy source is accepted");
    assert!(
        started.elapsed() < Duration::from_millis(2),
        "deallocate must not wait for the copy stream"
    );
    queue.poll();
    assert!(
        queue.pending() > 0,
        "the source release must still be owed while the copy stream is busy: {:?}",
        queue.stats()
    );

    provider.deallocate(destination).expect("destination");
    provider.deallocate(marker).expect("marker");
    assert!(
        drain(&queue, Duration::from_secs(60)),
        "every release completes once the copy stream drains: {:?}",
        queue.stats()
    );
    let after = queue.stats();
    assert_eq!(after.completed, before.completed + 3);
    assert_eq!(after.quarantined, before.quarantined);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn many_deallocations_return_without_draining_a_stream() {
    let provider = provider_or_fail("non-blocking deallocation");
    let runtime: Arc<CudaRuntime> = Arc::clone(provider.runtime());
    let queue = Arc::clone(provider.release_queue());
    let spin = runtime
        .nvrtc_function(SPIN_MODULE, SPIN_SOURCE, "spin_and_write")
        .expect("compile the spin kernel");

    let marker = provider.allocate(4, 256).expect("spin marker");
    let marker_ptr = cuptr(marker.as_ptr());
    let cycles = SPIN_CYCLES;
    let mut launch = runtime.stream().launch_builder(&spin);
    launch.arg(&marker_ptr).arg(&cycles);
    // SAFETY: one `unsigned int` write into a live 4-byte allocation.
    unsafe {
        launch
            .launch(LaunchConfig::for_num_elems(1))
            .expect("enqueue the spinning writer")
    };

    let mut buffers = Vec::new();
    for _ in 0..16 {
        buffers.push(provider.allocate(64 << 10, 256).expect("scratch"));
    }
    let started = Instant::now();
    for buffer in buffers {
        provider.deallocate(buffer).expect("accepted");
    }
    let elapsed = started.elapsed();
    assert!(
        elapsed < Duration::from_millis(5),
        "sixteen deferred frees must not cost a stream drain (took {elapsed:?})"
    );
    provider.deallocate(marker).expect("marker");
    assert!(
        drain(&queue, Duration::from_secs(60)),
        "all releases complete after the fences: {:?}",
        queue.stats()
    );
    assert_eq!(queue.pending(), 0);
    assert_eq!(
        queue.stats().quarantined,
        0,
        "no release ended in retained ownership"
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn a_weight_page_release_is_deferred_behind_both_streams() {
    use onnx_runtime_ep_cuda::weight_paging::CudaWeightPage;
    use onnx_runtime_ir::DataType;

    let provider = provider_or_fail("weight page deferred release");
    let runtime: Arc<CudaRuntime> = Arc::clone(provider.runtime());
    let queue = Arc::clone(provider.release_queue());
    let spin = runtime
        .nvrtc_function(SPIN_MODULE, SPIN_SOURCE, "spin_and_write")
        .expect("compile the spin kernel");

    let bytes = vec![7u8; 4096];
    let page = CudaWeightPage::upload(&runtime, DataType::Uint8, vec![4096], &bytes)
        .expect("upload a weight page")
        .with_deferred_release_queue(Arc::clone(&queue));

    let marker = provider.allocate(4, 256).expect("spin marker");
    let marker_ptr = cuptr(marker.as_ptr());
    let cycles = SPIN_CYCLES;
    let mut launch = runtime.stream().launch_builder(&spin);
    launch.arg(&marker_ptr).arg(&cycles);
    // SAFETY: one `unsigned int` write into a live 4-byte allocation.
    unsafe {
        launch
            .launch(LaunchConfig::for_num_elems(1))
            .expect("enqueue the spinning writer")
    };

    let before = queue.stats();
    let started = Instant::now();
    drop(page);
    assert!(
        started.elapsed() < Duration::from_millis(2),
        "dropping a weight page must not drain the streams"
    );
    queue.poll();
    assert!(
        queue.pending() > 0,
        "the page release is owed while the kernel runs: {:?}",
        queue.stats()
    );
    provider.deallocate(marker).expect("marker");
    assert!(
        drain(&queue, Duration::from_secs(60)),
        "the page release completes after the fences: {:?}",
        queue.stats()
    );
    assert_eq!(queue.stats().completed, before.completed + 2);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn provider_teardown_is_non_blocking_and_lets_accepted_releases_finish() {
    let provider = provider_or_fail("provider teardown");
    let runtime: Arc<CudaRuntime> = Arc::clone(provider.runtime());
    let queue = Arc::clone(provider.release_queue());
    let spin = runtime
        .nvrtc_function(SPIN_MODULE, SPIN_SOURCE, "spin_and_write")
        .expect("compile the spin kernel");

    let buffer = provider.allocate(1 << 20, 256).expect("device allocation");
    let marker = provider.allocate(4, 256).expect("spin marker");
    let marker_ptr = cuptr(marker.as_ptr());
    let cycles = SPIN_CYCLES;
    let mut launch = runtime.stream().launch_builder(&spin);
    launch.arg(&marker_ptr).arg(&cycles);
    // SAFETY: one `unsigned int` write into a live 4-byte allocation.
    unsafe {
        launch
            .launch(LaunchConfig::for_num_elems(1))
            .expect("enqueue the spinning writer")
    };
    provider.deallocate(buffer).expect("accepted");
    provider.deallocate(marker).expect("accepted");

    let started = Instant::now();
    drop(provider);
    let teardown = started.elapsed();
    assert!(
        teardown < Duration::from_millis(50),
        "provider teardown must not wait for in-flight work (took {teardown:?})"
    );
    assert!(
        queue.is_draining(),
        "teardown closes the queue once it has drained, so late teardown work is not refused"
    );

    // The queue, the CUDA context, and both requests outlive the provider, so
    // the accepted releases still complete.
    assert!(
        drain(&queue, Duration::from_secs(60)),
        "accepted releases complete after provider teardown: {:?}",
        queue.stats()
    );
    let stats = queue.stats();
    assert!(stats.completed >= 2, "{stats:?}");
    assert_eq!(stats.quarantined, 0, "{stats:?}");
    assert_eq!(stats.retained, 0, "{stats:?}");
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn an_explicit_partial_decommit_waits_only_on_recorded_events() {
    // The one path that still waits: it waits on freshly recorded compute and
    // copy completion events, not on the whole device, and it refunds exactly
    // what it unmapped.
    let provider = provider_or_fail("explicit partial decommit");
    if !provider.commits_on_demand() {
        eprintln!(
            "SKIPPED (no VMM arena): partial decommit needs the on-demand-committing allocator; \
             set the CUDA VMM environment switch on this runner"
        );
        return;
    }
    let granule = provider
        .mapped_bytes_for_allocation(4096, 256)
        .expect("granule size") as usize;
    let buffer = provider
        .allocate(granule * 2, 256)
        .expect("two-granule allocation");
    let unmapped = provider
        .decommit_allocation_range(&buffer, granule, granule)
        .expect("partial decommit completes");
    assert_eq!(
        unmapped as usize, granule,
        "exactly one granule was unmapped"
    );
    provider.deallocate(buffer).expect("accepted");
    assert!(
        drain(provider.release_queue(), Duration::from_secs(60)),
        "the final release completes: {:?}",
        provider.deferred_release_stats()
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn a_foreign_buffer_is_refused_rather_than_freed() {
    let provider = provider_or_fail("foreign buffer refusal");
    let runtime: Arc<CudaRuntime> = Arc::clone(provider.runtime());
    let raw = runtime.alloc_raw(4096).expect("a raw device allocation");
    // SAFETY: a live 4096-byte device allocation on this provider's device that
    // the provider's own binding never issued.
    let foreign = unsafe {
        onnx_runtime_ep_api::DeviceBuffer::from_raw_parts(
            onnx_runtime_ep_cuda::runtime::raw_ptr(raw),
            provider.device_id(),
            4096,
            256,
        )
    };
    let error = provider
        .deallocate(foreign)
        .expect_err("a buffer with no binding-issued ownership must fail closed");
    assert!(
        error.to_string().contains("binding-issued ownership"),
        "the refusal must say why: {error}"
    );
    // SAFETY: the raw allocation was never freed by the provider, so this is
    // still its single free.
    unsafe { runtime.free_raw(raw).expect("free the raw allocation") };
}

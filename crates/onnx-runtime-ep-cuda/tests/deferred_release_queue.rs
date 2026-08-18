//! Portable tests for the CUDA deferred release queue's scheduler and state
//! machine (issue #1186 Phase 4).
//!
//! These run on **any** machine: the queue's ordering, bounding, exactly-once
//! execution, close, and device-loss behaviour are expressed over the
//! [`ReleaseFence`]/[`DeferredReleaseAction`] contracts, so a fake fence and a
//! fake action exercise the whole state machine with no CUDA runtime present.
//! Owner/view/ABA semantics are covered by the memory-api tests and are not
//! duplicated here.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::time::Duration;

use onnx_runtime_ep_cuda::deferred_release::{
    CudaDeferredReleaseQueue, DeferredActionOutcome, DeferredReleaseAction, ReleaseFence,
    ReleaseFenceSource, RetainedOwnership,
};
use onnx_runtime_memory_governor::{AllocationReleaseState, DeferredEnqueueRejection};

/// A fence whose completion the test controls.
#[derive(Debug)]
struct FakeFence {
    complete: Arc<AtomicBool>,
    /// Incremented by `Drop`, so a test can prove a device-lost fence was
    /// retained rather than destroyed.
    destroyed: Arc<AtomicUsize>,
}

impl ReleaseFence for FakeFence {
    fn is_complete(&self) -> bool {
        self.complete.load(Ordering::Acquire)
    }
}

impl Drop for FakeFence {
    fn drop(&mut self) {
        self.destroyed.fetch_add(1, Ordering::Relaxed);
    }
}

/// Stands in for the provider's two streams: one fence per stream.
#[derive(Debug)]
struct FakeStreams {
    compute: Arc<AtomicBool>,
    copy: Arc<AtomicBool>,
    recorded: Arc<AtomicUsize>,
    destroyed: Arc<AtomicUsize>,
    fail: Arc<AtomicBool>,
}

impl ReleaseFenceSource for FakeStreams {
    fn record(&self) -> Result<Vec<Box<dyn ReleaseFence>>, String> {
        if self.fail.load(Ordering::Acquire) {
            return Err("the fake driver refused to record an event".into());
        }
        self.recorded.fetch_add(1, Ordering::Relaxed);
        Ok(vec![
            Box::new(FakeFence {
                complete: Arc::clone(&self.compute),
                destroyed: Arc::clone(&self.destroyed),
            }),
            Box::new(FakeFence {
                complete: Arc::clone(&self.copy),
                destroyed: Arc::clone(&self.destroyed),
            }),
        ])
    }
}

#[derive(Debug)]
struct BlockingFence {
    entered: Arc<Barrier>,
    resume: Arc<Barrier>,
}

impl ReleaseFence for BlockingFence {
    fn is_complete(&self) -> bool {
        self.entered.wait();
        self.resume.wait();
        false
    }
}

#[derive(Debug)]
struct BlockingStreams {
    entered: Arc<Barrier>,
    resume: Arc<Barrier>,
}

impl ReleaseFenceSource for BlockingStreams {
    fn record(&self) -> Result<Vec<Box<dyn ReleaseFence>>, String> {
        Ok(vec![Box::new(BlockingFence {
            entered: Arc::clone(&self.entered),
            resume: Arc::clone(&self.resume),
        })])
    }
}

/// The shared handles a test uses to drive the fake device.
struct Harness {
    queue: Arc<CudaDeferredReleaseQueue>,
    compute: Arc<AtomicBool>,
    copy: Arc<AtomicBool>,
    recorded: Arc<AtomicUsize>,
    destroyed: Arc<AtomicUsize>,
    fail_record: Arc<AtomicBool>,
}

fn harness(capacity: usize) -> Harness {
    let compute = Arc::new(AtomicBool::new(false));
    let copy = Arc::new(AtomicBool::new(false));
    let recorded = Arc::new(AtomicUsize::new(0));
    let destroyed = Arc::new(AtomicUsize::new(0));
    let fail_record = Arc::new(AtomicBool::new(false));
    let queue = CudaDeferredReleaseQueue::manual(
        Box::new(FakeStreams {
            compute: Arc::clone(&compute),
            copy: Arc::clone(&copy),
            recorded: Arc::clone(&recorded),
            destroyed: Arc::clone(&destroyed),
            fail: Arc::clone(&fail_record),
        }),
        capacity,
    );
    Harness {
        queue,
        compute,
        copy,
        recorded,
        destroyed,
        fail_record,
    }
}

/// A release that records that it ran, and how many times.
#[derive(Debug)]
struct CountingRelease {
    executed: Arc<AtomicUsize>,
    dropped_unexecuted: Arc<AtomicUsize>,
    bytes: u64,
    outcome: ReleaseResult,
}

#[derive(Clone, Copy, Debug)]
enum ReleaseResult {
    Complete { unmapped: u64 },
    Quarantine,
}

impl DeferredReleaseAction for CountingRelease {
    fn execute(mut self: Box<Self>) -> DeferredActionOutcome {
        self.executed.fetch_add(1, Ordering::AcqRel);
        let outcome = self.outcome;
        let bytes = self.bytes;
        // Mark as executed so `Drop` does not count it as abandoned.
        self.outcome = ReleaseResult::Complete { unmapped: 0 };
        match outcome {
            ReleaseResult::Complete { unmapped } => DeferredActionOutcome::released(unmapped),
            ReleaseResult::Quarantine => DeferredActionOutcome::quarantined(
                AllocationReleaseState::Quarantined,
                0,
                "the fake allocator kept the bytes",
                Some(RetainedOwnership {
                    bytes,
                    detail: "fake retained ownership".into(),
                    keep_alive: Box::new(bytes),
                }),
            ),
        }
    }

    fn label(&self) -> &'static str {
        "test release"
    }

    fn bytes(&self) -> u64 {
        self.bytes
    }
}

impl Drop for CountingRelease {
    fn drop(&mut self) {
        if self.executed.load(Ordering::Acquire) == 0 {
            self.dropped_unexecuted.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn release(executed: &Arc<AtomicUsize>, abandoned: &Arc<AtomicUsize>) -> CountingRelease {
    CountingRelease {
        executed: Arc::clone(executed),
        dropped_unexecuted: Arc::clone(abandoned),
        bytes: 4096,
        outcome: ReleaseResult::Complete { unmapped: 4096 },
    }
}

#[test]
fn enqueue_records_a_fence_on_both_streams_and_returns_immediately() {
    let h = harness(8);
    let executed = Arc::new(AtomicUsize::new(0));
    let abandoned = Arc::new(AtomicUsize::new(0));
    let started = std::time::Instant::now();
    h.queue
        .enqueue(release(&executed, &abandoned))
        .expect("accepted");
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "enqueue must not wait on anything"
    );
    assert_eq!(
        h.recorded.load(Ordering::Acquire),
        1,
        "one recording covering both streams"
    );
    assert_eq!(h.queue.pending(), 1);
    assert_eq!(executed.load(Ordering::Acquire), 0);
}

#[test]
fn an_incomplete_compute_fence_blocks_the_release() {
    let h = harness(8);
    let executed = Arc::new(AtomicUsize::new(0));
    let abandoned = Arc::new(AtomicUsize::new(0));
    h.queue
        .enqueue(release(&executed, &abandoned))
        .expect("accepted");
    // Only the copy stream has drained.
    h.copy.store(true, Ordering::Release);
    assert_eq!(h.queue.poll(), 0);
    assert_eq!(executed.load(Ordering::Acquire), 0);
    assert_eq!(h.queue.pending(), 1);
}

#[test]
fn an_incomplete_copy_fence_blocks_the_release() {
    let h = harness(8);
    let executed = Arc::new(AtomicUsize::new(0));
    let abandoned = Arc::new(AtomicUsize::new(0));
    h.queue
        .enqueue(release(&executed, &abandoned))
        .expect("accepted");
    // Only the compute stream has drained: a release that ignored the copy
    // stream would free memory a transfer is still reading.
    h.compute.store(true, Ordering::Release);
    assert_eq!(h.queue.poll(), 0);
    assert_eq!(executed.load(Ordering::Acquire), 0);
    assert_eq!(h.queue.pending(), 1);
}

#[test]
fn a_release_runs_once_both_fences_have_completed() {
    let h = harness(8);
    let executed = Arc::new(AtomicUsize::new(0));
    let abandoned = Arc::new(AtomicUsize::new(0));
    h.queue
        .enqueue(release(&executed, &abandoned))
        .expect("accepted");
    h.compute.store(true, Ordering::Release);
    h.copy.store(true, Ordering::Release);
    assert_eq!(h.queue.poll(), 1);
    assert_eq!(executed.load(Ordering::Acquire), 1);
    assert_eq!(h.queue.pending(), 0);
    let stats = h.queue.stats();
    assert_eq!(stats.completed, 1);
    assert_eq!(stats.quarantined, 0);
    assert_eq!(stats.mapped_refunded_bytes, 4096);
    assert_eq!(
        h.destroyed.load(Ordering::Acquire),
        2,
        "both events are released once their ordering has been observed"
    );
}

#[test]
fn a_bounded_queue_rejects_and_hands_the_exact_request_back() {
    let h = harness(2);
    let executed = Arc::new(AtomicUsize::new(0));
    let abandoned = Arc::new(AtomicUsize::new(0));
    h.queue
        .enqueue(release(&executed, &abandoned))
        .expect("first");
    h.queue
        .enqueue(release(&executed, &abandoned))
        .expect("second");
    let refused = h
        .queue
        .enqueue(release(&executed, &abandoned))
        .expect_err("the bound is enforced");
    assert_eq!(refused.rejection, DeferredEnqueueRejection::Full);
    assert_eq!(refused.action.bytes(), 4096, "the exact action comes back");
    assert_eq!(h.queue.stats().enqueue_failures, 1);
    assert_eq!(
        h.recorded.load(Ordering::Acquire),
        2,
        "a refused enqueue records no fence"
    );
    // Dropping the refusal never frees: the action was never executed.
    drop(refused);
    assert_eq!(executed.load(Ordering::Acquire), 0);
    assert_eq!(abandoned.load(Ordering::Acquire), 1);
}

#[test]
fn a_fence_recording_failure_refuses_without_taking_ownership() {
    let h = harness(4);
    let executed = Arc::new(AtomicUsize::new(0));
    let abandoned = Arc::new(AtomicUsize::new(0));
    h.fail_record.store(true, Ordering::Release);
    let refused = h
        .queue
        .enqueue(release(&executed, &abandoned))
        .expect_err("no ordering, no ownership");
    assert_eq!(refused.rejection, DeferredEnqueueRejection::Refused);
    assert_eq!(h.queue.pending(), 0, "the reserved slot is given back");
    assert_eq!(h.queue.stats().enqueue_failures, 1);
    drop(refused);
    assert_eq!(executed.load(Ordering::Acquire), 0);
}

#[test]
fn concurrent_pollers_execute_each_release_exactly_once() {
    let h = harness(256);
    let executed = Arc::new(AtomicUsize::new(0));
    let abandoned = Arc::new(AtomicUsize::new(0));
    const RELEASES: usize = 200;
    for _ in 0..RELEASES {
        h.queue
            .enqueue(release(&executed, &abandoned))
            .expect("accepted");
    }
    h.compute.store(true, Ordering::Release);
    h.copy.store(true, Ordering::Release);

    const POLLERS: usize = 8;
    let barrier = Arc::new(Barrier::new(POLLERS));
    let mut handles = Vec::new();
    for _ in 0..POLLERS {
        let queue = Arc::clone(&h.queue);
        let barrier = Arc::clone(&barrier);
        handles.push(std::thread::spawn(move || {
            barrier.wait();
            let mut ran = 0;
            for _ in 0..64 {
                ran += queue.poll();
            }
            ran
        }));
    }
    let total: usize = handles.into_iter().map(|h| h.join().expect("poller")).sum();
    assert_eq!(
        total, RELEASES,
        "each entry is executed by exactly one poller"
    );
    assert_eq!(executed.load(Ordering::Acquire), RELEASES);
    assert_eq!(h.queue.pending(), 0);
    assert_eq!(h.queue.stats().completed, RELEASES as u64);
}

#[test]
fn a_closed_queue_refuses_new_work_but_finishes_what_it_accepted() {
    let h = harness(8);
    let executed = Arc::new(AtomicUsize::new(0));
    let abandoned = Arc::new(AtomicUsize::new(0));
    h.queue
        .enqueue(release(&executed, &abandoned))
        .expect("accepted before close");
    h.queue.close();
    let refused = h
        .queue
        .enqueue(release(&executed, &abandoned))
        .expect_err("closed");
    assert_eq!(refused.rejection, DeferredEnqueueRejection::Closed);
    assert!(h.queue.is_closed());
    // Closing never drains and never waits: the accepted release is still
    // pending until its fences complete.
    assert_eq!(h.queue.pending(), 1);
    assert_eq!(executed.load(Ordering::Acquire), 0);
    h.compute.store(true, Ordering::Release);
    h.copy.store(true, Ordering::Release);
    assert_eq!(h.queue.poll(), 1);
    assert_eq!(h.queue.pending(), 0);
}

#[test]
fn pending_work_that_never_completes_is_never_freed_early() {
    let h = harness(8);
    let executed = Arc::new(AtomicUsize::new(0));
    let abandoned = Arc::new(AtomicUsize::new(0));
    h.queue
        .enqueue(release(&executed, &abandoned))
        .expect("accepted");
    h.queue.close();
    for _ in 0..100 {
        assert_eq!(h.queue.poll(), 0);
    }
    assert_eq!(h.queue.pending(), 1, "still owed");
    assert_eq!(
        executed.load(Ordering::Acquire),
        0,
        "a release whose ordering never completes is never performed"
    );
    assert_eq!(h.queue.stats().completed, 0);
}

#[test]
fn device_loss_retains_ownership_without_executing_or_destroying_events() {
    let h = harness(8);
    let executed = Arc::new(AtomicUsize::new(0));
    let abandoned = Arc::new(AtomicUsize::new(0));
    h.queue
        .enqueue(release(&executed, &abandoned))
        .expect("accepted");
    h.queue.mark_device_lost("Xid 13");
    assert!(h.queue.is_device_lost());
    assert_eq!(
        executed.load(Ordering::Acquire),
        0,
        "the allocator is never called after device loss"
    );
    assert_eq!(
        h.destroyed.load(Ordering::Acquire),
        0,
        "events are retained, not destroyed through a lost context"
    );
    let stats = h.queue.stats();
    assert_eq!(stats.pending, 0, "nothing is owed to the device any more");
    assert_eq!(stats.quarantined, 1);
    assert_eq!(stats.retained, 1);
    assert_eq!(stats.mapped_refunded_bytes, 0, "no refund on assumption");
    let retained = h.queue.retained();
    assert_eq!(retained.len(), 1);
    assert_eq!(retained[0].state, AllocationReleaseState::DeviceLost);
    assert_eq!(retained[0].bytes, 4096);
    // Polling a lost device does nothing at all.
    assert_eq!(h.queue.poll(), 0);
    // And a later enqueue is refused as device-lost, with the request returned.
    let refused = h
        .queue
        .enqueue(release(&executed, &abandoned))
        .expect_err("device lost");
    assert_eq!(refused.rejection, DeferredEnqueueRejection::DeviceLost);
}

#[test]
fn device_loss_captures_an_incomplete_entry_held_by_a_concurrent_poller() {
    let entered = Arc::new(Barrier::new(2));
    let resume = Arc::new(Barrier::new(2));
    let queue = CudaDeferredReleaseQueue::manual(
        Box::new(BlockingStreams {
            entered: Arc::clone(&entered),
            resume: Arc::clone(&resume),
        }),
        8,
    );
    let executed = Arc::new(AtomicUsize::new(0));
    let abandoned = Arc::new(AtomicUsize::new(0));
    queue
        .enqueue(release(&executed, &abandoned))
        .expect("accepted");

    let poll_queue = Arc::clone(&queue);
    let poller = std::thread::spawn(move || poll_queue.poll());
    entered.wait();

    let loss_queue = Arc::clone(&queue);
    let loss = std::thread::spawn(move || loss_queue.mark_device_lost("concurrent Xid"));
    resume.wait();

    assert_eq!(poller.join().expect("poller"), 0);
    loss.join().expect("loss marker");
    assert_eq!(executed.load(Ordering::Acquire), 0);
    assert_eq!(queue.pending(), 0, "the poller's carry must be retained");
    assert_eq!(queue.stats().retained, 1);
    assert!(queue.is_closed(), "a lost queue needs no live worker");
}

#[test]
fn a_quarantined_release_is_not_reported_as_a_completed_free() {
    let h = harness(8);
    let executed = Arc::new(AtomicUsize::new(0));
    let abandoned = Arc::new(AtomicUsize::new(0));
    h.queue
        .enqueue(CountingRelease {
            executed: Arc::clone(&executed),
            dropped_unexecuted: Arc::clone(&abandoned),
            bytes: 8192,
            outcome: ReleaseResult::Quarantine,
        })
        .expect("accepted");
    h.compute.store(true, Ordering::Release);
    h.copy.store(true, Ordering::Release);
    assert_eq!(h.queue.poll(), 1);
    let stats = h.queue.stats();
    assert_eq!(stats.completed, 0, "retained ownership is not a free");
    assert_eq!(stats.quarantined, 1);
    assert_eq!(stats.mapped_refunded_bytes, 0);
    let retained = h.queue.retained();
    assert_eq!(retained.len(), 1);
    assert_eq!(retained[0].bytes, 8192);
    assert_eq!(retained[0].state, AllocationReleaseState::Quarantined);
}

#[test]
fn the_ready_queue_does_not_grow_under_deterministic_stress() {
    let h = harness(64);
    let executed = Arc::new(AtomicUsize::new(0));
    let abandoned = Arc::new(AtomicUsize::new(0));
    let mut high_water = 0usize;
    // Steady state: enqueue, complete, drain, repeat. Pending must return to
    // zero every round rather than accumulating a backlog of ready work.
    for round in 0..500 {
        h.compute.store(false, Ordering::Release);
        h.copy.store(false, Ordering::Release);
        for _ in 0..(round % 5 + 1) {
            h.queue
                .enqueue(release(&executed, &abandoned))
                .expect("accepted");
        }
        high_water = high_water.max(h.queue.pending());
        h.compute.store(true, Ordering::Release);
        h.copy.store(true, Ordering::Release);
        h.queue.poll();
        assert_eq!(h.queue.pending(), 0, "round {round} left work behind");
    }
    assert!(
        high_water <= 5,
        "high water mark stayed bounded: {high_water}"
    );
    assert_eq!(
        executed.load(Ordering::Acquire),
        h.queue.stats().completed as usize
    );
    assert_eq!(h.queue.stats().enqueue_failures, 0);
    assert_eq!(abandoned.load(Ordering::Acquire), 0);
}

#[test]
fn an_autonomous_worker_drains_without_any_caller_polling() {
    let compute = Arc::new(AtomicBool::new(true));
    let copy = Arc::new(AtomicBool::new(true));
    let queue = CudaDeferredReleaseQueue::new(
        Box::new(FakeStreams {
            compute: Arc::clone(&compute),
            copy: Arc::clone(&copy),
            recorded: Arc::new(AtomicUsize::new(0)),
            destroyed: Arc::new(AtomicUsize::new(0)),
            fail: Arc::new(AtomicBool::new(false)),
        }),
        16,
    );
    let executed = Arc::new(AtomicUsize::new(0));
    let abandoned = Arc::new(AtomicUsize::new(0));
    for _ in 0..4 {
        queue
            .enqueue(release(&executed, &abandoned))
            .expect("accepted");
    }
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while queue.pending() > 0 && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(2));
    }
    assert_eq!(queue.pending(), 0, "the worker drained the queue itself");
    assert_eq!(executed.load(Ordering::Acquire), 4);
    // Closing is non-blocking; the worker exits on its own once empty.
    queue.close();
}

/// A release that enqueues *more* work while it executes, the way a real one
/// does when it drops the last reference to an allocator whose reservation then
/// has to be torn down.
#[derive(Debug)]
struct CascadingRelease {
    queue: Arc<CudaDeferredReleaseQueue>,
    executed: Arc<AtomicUsize>,
    follow_up: Arc<AtomicUsize>,
    abandoned: Arc<AtomicUsize>,
}

impl DeferredReleaseAction for CascadingRelease {
    fn execute(self: Box<Self>) -> DeferredActionOutcome {
        self.executed.fetch_add(1, Ordering::AcqRel);
        // This is the teardown a hard close would refuse.
        if self
            .queue
            .enqueue(release(&self.follow_up, &self.abandoned))
            .is_err()
        {
            return DeferredActionOutcome::quarantined(
                AllocationReleaseState::Quarantined,
                0,
                "the follow-up teardown was refused",
                None,
            );
        }
        DeferredActionOutcome::released(0)
    }

    fn label(&self) -> &'static str {
        "cascading test release"
    }
}

#[test]
fn closing_after_drain_still_accepts_the_teardown_a_release_produces() {
    let h = harness(8);
    let executed = Arc::new(AtomicUsize::new(0));
    let follow_up = Arc::new(AtomicUsize::new(0));
    let abandoned = Arc::new(AtomicUsize::new(0));
    h.queue
        .enqueue(CascadingRelease {
            queue: Arc::clone(&h.queue),
            executed: Arc::clone(&executed),
            follow_up: Arc::clone(&follow_up),
            abandoned: Arc::clone(&abandoned),
        })
        .expect("accepted");
    // Provider teardown: close once drained, not now.
    h.queue.close_after_drain();
    assert!(h.queue.is_draining());
    assert!(
        !h.queue.is_closed(),
        "a draining queue is still reachable, so it must still accept"
    );
    h.compute.store(true, Ordering::Release);
    h.copy.store(true, Ordering::Release);
    assert_eq!(h.queue.poll(), 1);
    assert_eq!(executed.load(Ordering::Acquire), 1);
    assert_eq!(
        h.queue.pending(),
        1,
        "the teardown the release produced was accepted, not refused"
    );
    assert_eq!(h.queue.poll(), 1);
    assert_eq!(follow_up.load(Ordering::Acquire), 1);
    assert_eq!(h.queue.stats().enqueue_failures, 0);
    assert_eq!(h.queue.stats().quarantined, 0);
}

#[test]
fn a_hard_close_refuses_immediately_even_while_work_is_owed() {
    let h = harness(8);
    let executed = Arc::new(AtomicUsize::new(0));
    let abandoned = Arc::new(AtomicUsize::new(0));
    h.queue
        .enqueue(release(&executed, &abandoned))
        .expect("accepted");
    h.queue.close();
    assert!(h.queue.is_closed());
    let refused = h
        .queue
        .enqueue(release(&executed, &abandoned))
        .expect_err("a hard close refuses at once");
    assert_eq!(refused.rejection, DeferredEnqueueRejection::Closed);
    assert_eq!(h.queue.pending(), 1, "what was accepted is still owed");
}

#[test]
fn dropping_a_queue_with_pending_work_neither_frees_nor_blocks() {
    let h = harness(8);
    let executed = Arc::new(AtomicUsize::new(0));
    let abandoned = Arc::new(AtomicUsize::new(0));
    h.queue
        .enqueue(release(&executed, &abandoned))
        .expect("accepted");
    let started = std::time::Instant::now();
    drop(h.queue);
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "dropping the queue must not wait"
    );
    assert_eq!(
        executed.load(Ordering::Acquire),
        0,
        "an un-ordered release is never performed at teardown"
    );
    assert_eq!(
        abandoned.load(Ordering::Acquire),
        1,
        "the action is dropped without being executed, which is a retain"
    );
}

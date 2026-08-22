//! Process-wide exclusion between CUDA graph capture and device-synchronizing
//! memory operations.
//!
//! # The hazard
//!
//! Stream capture records work instead of executing it. CUDA invalidates an
//! in-progress capture whenever the process performs an operation that forces a
//! device-wide synchronization, and the VMM calls behind this crate's arena --
//! `cuMemCreate`, `cuMemMap`, `cuMemSetAccess`, `cuMemUnmap`, `cuMemRelease` --
//! are exactly such operations.
//!
//! The critical and easily missed half is that **the hazard is process-wide,
//! not stream-wide and not runtime-wide**. `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`
//! is often read as "other threads may do as they please"; what it actually
//! relaxes is only the *legality check* CUDA applies to unsafe API calls. It
//! does not stop a genuine device-wide synchronization on another thread from
//! invalidating the capture. A thread committing a granule for an unrelated
//! allocation, on an unrelated stream, owned by an unrelated `CudaRuntime`, will
//! still kill a capture running on this one. The failure surfaces far from its
//! cause, as `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED` ("operation failed due to a
//! previous error during capture") raised inside the *victim's* capture region,
//! naming a kernel that did nothing wrong.
//!
//! So the exclusion cannot live on the stream, the graph lifecycle, or the
//! runtime: each of those admits a second instance in the same process. It has
//! to be a single global.
//!
//! # The contract
//!
//! * A capturing thread holds the [`CaptureExclusion`] guard for the entire
//!   capture region, from `begin` through `end`/`abort`.
//! * Every code path that can synchronize the device takes
//!   [`synchronizing_section`] first.
//! * Callback paths that cannot block, or that may run on the capture thread,
//!   use [`run_or_defer_synchronizing`] so capture-unsafe work is owned by the
//!   gate and runs immediately after capture instead.
//!
//! Captures exclude synchronizers; synchronizers exclude captures; synchronizers
//! run concurrently with one another. That is a writer-preferring reader/writer
//! lock in shape, but it is spelled out by hand here rather than built on
//! `RwLock` for one reason: the capture guard has to be storable in the graph
//! lifecycle struct across the `begin`/`end` call boundary, and `std`'s
//! `RwLockWriteGuard` is `!Send`, which would infect every type that owns it.
//! These guards carry no borrow, so they are plain `Send` values.
//!
//! # Re-entrancy
//!
//! The capturing thread allocates *during* capture -- that is the entire point
//! of capturing a decode step. Those allocations must not block on the capture
//! the same thread is holding. [`synchronizing_section`] therefore consults a
//! thread-local flag and returns `None` on the capturing thread, letting its own
//! commits proceed. This is not a soundness hole: CUDA's constraint is on
//! *other* threads perturbing the capture, and a capturing thread's own
//! allocation is already ordered with respect to its own recording.
//!
//! Nested captures on one thread (two lifecycles, two streams) are permitted and
//! refcounted; only the outermost guard opens and closes the gate.
//!
//! # Cost
//!
//! One uncontended mutex acquire/release per commit or release call. Decode
//! steady state performs a few dozen allocator operations per token, so the
//! added cost is on the order of a microsecond per token against a ~24 ms step:
//! unmeasurable. Contention only exists during capture, which happens once per
//! graph rather than per token.

use std::cell::Cell;
use std::collections::VecDeque;
use std::sync::{Condvar, Mutex, OnceLock};

/// Reader/writer counts for the gate.
#[derive(Default)]
struct GateState {
    /// Threads currently inside a [`synchronizing_section`].
    synchronizers: u32,
    /// Threads blocked in [`CaptureExclusion::acquire`]. Non-zero turns new
    /// synchronizing sections away so a steady stream of allocations cannot
    /// starve a capture that is waiting to start.
    captures_waiting: u32,
    /// Whether a thread currently holds the capture exclusion.
    capturing: bool,
    /// Capture-unsafe work submitted while a capture is live. The outermost
    /// capture guard drains it after ending capture while still excluding the
    /// next capture.
    deferred: VecDeque<Box<dyn FnOnce() + Send + 'static>>,
}

fn gate() -> &'static (Mutex<GateState>, Condvar) {
    static GATE: OnceLock<(Mutex<GateState>, Condvar)> = OnceLock::new();
    GATE.get_or_init(|| (Mutex::new(GateState::default()), Condvar::new()))
}

thread_local! {
    /// Depth of capture guards held by *this* thread, so its own allocations
    /// skip the gate and nested captures refcount correctly.
    static CAPTURE_DEPTH: Cell<u32> = const { Cell::new(0) };

    /// Depth of synchronizing sections held by *this* thread.
    ///
    /// Required for correctness, not just economy. Because a waiting capture
    /// blocks *new* sections, a thread that already holds one and then re-enters
    /// -- `commit` recurses once per granule for a multi-granule request --
    /// would queue behind a capture that is itself waiting for that thread's
    /// outer section to drain. Only the outermost section counts.
    static SHARED_DEPTH: Cell<u32> = const { Cell::new(0) };
}

fn lock_state() -> std::sync::MutexGuard<'static, GateState> {
    // A panic inside the gate leaves only these two counters inconsistent, and
    // refusing to allocate forever afterwards is strictly worse than proceeding.
    gate()
        .0
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Exclusive access held for the lifetime of a CUDA graph capture region.
///
/// Blocks on construction until every in-flight [`synchronizing_section`] has
/// finished, then keeps new ones out until it is dropped. Acquire it *before*
/// `cuStreamBeginCapture` and drop it *after* `cuStreamEndCapture` (or after
/// aborting), so no window exists in which the capture is live but unprotected.
#[must_use = "the exclusion ends when the guard is dropped; binding it to `_` drops it immediately"]
pub struct CaptureExclusion {
    /// `false` for a nested guard, which must not close the gate on drop.
    outermost: bool,
    /// Whether this guard set aside a synchronizing section its own thread was
    /// holding, to be restored on drop. See [`CaptureExclusion::acquire`].
    yielded_own_section: bool,
}

impl CaptureExclusion {
    /// Acquire the capture exclusion, waiting for in-flight synchronizing
    /// sections to drain.
    pub fn acquire() -> Self {
        let nested = CAPTURE_DEPTH.with(|depth| {
            let current = depth.get();
            depth.set(current + 1);
            current > 0
        });
        if nested {
            return Self {
                outermost: false,
                yielded_own_section: false,
            };
        }

        // A thread may already hold a synchronizing section when it starts
        // capturing -- a long-lived one covering a whole unit of work, for
        // instance. Waiting for it to drain would be waiting on itself, and two
        // such threads racing to capture would each wait for the other's
        // section forever. Set our own aside for the duration and restore it on
        // drop; it exists to keep *other* threads' captures safe from this
        // thread's driver calls, and this thread's own capture is already
        // ordered with those calls.
        let yielded_own_section = SHARED_DEPTH.with(Cell::get) > 0;

        let (_, condvar) = gate();
        let mut state = lock_state();
        if yielded_own_section {
            state.synchronizers = state.synchronizers.saturating_sub(1);
        }
        state.captures_waiting += 1;
        while state.capturing || state.synchronizers > 0 {
            state = condvar
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        state.captures_waiting -= 1;
        state.capturing = true;
        drop(state);
        condvar.notify_all();
        Self {
            outermost: true,
            yielded_own_section,
        }
    }
}

impl Drop for CaptureExclusion {
    fn drop(&mut self) {
        CAPTURE_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
        if !self.outermost {
            return;
        }
        let (_, condvar) = gate();
        let mut state = lock_state();
        state.capturing = false;
        let deferred = std::mem::take(&mut state.deferred);
        if !deferred.is_empty() {
            state.synchronizers += 1;
        }
        if self.yielded_own_section {
            state.synchronizers += 1;
        }
        drop(state);
        condvar.notify_all();
        if !deferred.is_empty() {
            SHARED_DEPTH.with(|depth| depth.set(depth.get() + 1));
            for action in deferred {
                if std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err() {
                    eprintln!("cuda capture gate: deferred synchronizing action panicked");
                }
            }
            SHARED_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
            let mut state = lock_state();
            state.synchronizers = state.synchronizers.saturating_sub(1);
            let drained = state.synchronizers == 0;
            drop(state);
            if drained {
                condvar.notify_all();
            }
        }
    }
}

/// Shared access held across an operation that may synchronize the device.
///
/// Constructed via [`synchronizing_section`].
#[must_use = "the section ends when the guard is dropped; binding it to `_` drops it immediately"]
pub struct SynchronizingSection {
    /// `false` for a nested section, which must not decrement the shared count.
    outermost: bool,
}

impl Drop for SynchronizingSection {
    fn drop(&mut self) {
        SHARED_DEPTH.with(|depth| depth.set(depth.get().saturating_sub(1)));
        if !self.outermost {
            return;
        }
        let (_, condvar) = gate();
        let mut state = lock_state();
        state.synchronizers = state.synchronizers.saturating_sub(1);
        let drained = state.synchronizers == 0;
        drop(state);
        if drained {
            condvar.notify_all();
        }
    }
}

/// Enter a section that may synchronize the device, blocking while another
/// thread is capturing a CUDA graph.
///
/// Returns `None` -- taking no lock at all -- when the calling thread is itself
/// the one capturing, so a capture's own allocations never deadlock against it.
///
/// Wrap the narrowest region that contains the driver call; holding it longer
/// than necessary delays capture start but is never incorrect.
pub fn synchronizing_section() -> Option<SynchronizingSection> {
    if CAPTURE_DEPTH.with(Cell::get) > 0 {
        return None;
    }

    let nested = SHARED_DEPTH.with(|depth| {
        let current = depth.get();
        depth.set(current + 1);
        current > 0
    });
    if nested {
        return Some(SynchronizingSection { outermost: false });
    }

    let (_, condvar) = gate();
    let mut state = lock_state();
    while state.capturing || state.captures_waiting > 0 {
        state = condvar
            .wait(state)
            .unwrap_or_else(|poisoned| poisoned.into_inner());
    }
    state.synchronizers += 1;
    Some(SynchronizingSection { outermost: true })
}

/// Execute capture-unsafe work now, or defer it until the active capture ends.
///
/// Unlike [`synchronizing_section`], this never blocks behind an already-live
/// capture and never lets the capturing thread use its re-entrancy carve-out to
/// invoke a capture-unsafe API. The action runs under synchronizer ownership,
/// either immediately or while the outermost capture guard drains its queue.
///
/// Returns `true` when the action was deferred.
pub fn run_or_defer_synchronizing(action: impl FnOnce() + Send + 'static) -> bool {
    if CAPTURE_DEPTH.with(Cell::get) > 0 {
        lock_state().deferred.push_back(Box::new(action));
        return true;
    }

    // Re-entry already owns a synchronizer. In particular, deferred actions
    // run with SHARED_DEPTH set while the capture-drop drain owns one shared
    // gate entry. Waiting behind a newly queued capture here would make that
    // capture wait for the very synchronizer this call is refusing to reuse.
    if SHARED_DEPTH.with(Cell::get) > 0 {
        action();
        return false;
    }

    let (_, condvar) = gate();
    let mut state = lock_state();
    while !state.capturing && state.captures_waiting > 0 {
        state = condvar
            .wait(state)
            .unwrap_or_else(|poisoned| poisoned.into_inner());
    }
    if state.capturing {
        state.deferred.push_back(Box::new(action));
        return true;
    }
    state.synchronizers += 1;
    drop(state);

    SHARED_DEPTH.with(|depth| depth.set(depth.get() + 1));
    let section = SynchronizingSection { outermost: true };
    action();
    drop(section);
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Command, Stdio};
    use std::sync::Arc;
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    /// The gate is process-global and these tests deliberately contend on it, so
    /// they must not run concurrently with each other.
    fn test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn serialized() -> std::sync::MutexGuard<'static, ()> {
        test_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn wait_for_capture_waiter() {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if lock_state().captures_waiting > 0 {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "capture did not enter the waiting state"
            );
            std::thread::yield_now();
        }
    }

    fn assert_child_does_not_deadlock(test_filter: &str, marker: &str) {
        let mut child = Command::new(std::env::current_exe().expect("current test executable"))
            .arg(test_filter)
            .arg("--nocapture")
            .env(marker, "1")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn deadlock regression child");
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(status) = child.try_wait().expect("poll regression child") {
                assert!(
                    status.success(),
                    "deadlock regression child failed: {status}"
                );
                return;
            }
            if Instant::now() >= deadline {
                child.kill().expect("kill deadlocked regression child");
                let _ = child.wait();
                panic!("deadlock regression child exceeded {deadline:?}");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn a_synchronizing_section_on_another_thread_waits_for_capture_to_end() {
        let _serial = serialized();
        let capture = CaptureExclusion::acquire();
        let entered = Arc::new(AtomicBool::new(false));

        let flag = Arc::clone(&entered);
        let waiter = std::thread::spawn(move || {
            let _section = synchronizing_section();
            flag.store(true, Ordering::SeqCst);
        });

        // Not a race: the assertion is that the thread is *still* blocked. A
        // false pass would require the gate to admit it, which is the bug.
        std::thread::sleep(Duration::from_millis(50));
        assert!(
            !entered.load(Ordering::SeqCst),
            "a synchronizing section entered while a capture was active"
        );

        drop(capture);
        waiter.join().expect("waiter thread panicked");
        assert!(entered.load(Ordering::SeqCst));
    }

    #[test]
    fn capture_waits_for_an_in_flight_synchronizing_section_to_drain() {
        let _serial = serialized();
        let section = synchronizing_section().expect("not capturing on this thread");
        let captured = Arc::new(AtomicBool::new(false));

        let flag = Arc::clone(&captured);
        let capturer = std::thread::spawn(move || {
            let _capture = CaptureExclusion::acquire();
            flag.store(true, Ordering::SeqCst);
        });

        std::thread::sleep(Duration::from_millis(50));
        assert!(
            !captured.load(Ordering::SeqCst),
            "capture began while a synchronizing section was in flight"
        );

        drop(section);
        capturer.join().expect("capture thread panicked");
        assert!(captured.load(Ordering::SeqCst));
    }

    #[test]
    fn the_capturing_thread_may_synchronize_without_deadlocking() {
        let _serial = serialized();
        let _capture = CaptureExclusion::acquire();
        // The negative control for the re-entrancy carve-out: this call would
        // block forever if the capturing thread took the gate like any other.
        assert!(
            synchronizing_section().is_none(),
            "the capturing thread must bypass the gate rather than queue on it"
        );
    }

    #[test]
    fn capture_unsafe_action_from_capture_thread_is_deferred_until_capture_ends() {
        let _serial = serialized();
        let ran = Arc::new(AtomicBool::new(false));
        let capture = CaptureExclusion::acquire();
        let flag = Arc::clone(&ran);
        assert!(run_or_defer_synchronizing(move || {
            flag.store(true, Ordering::SeqCst);
        }));
        assert!(!ran.load(Ordering::SeqCst));
        drop(capture);
        assert!(ran.load(Ordering::SeqCst));
    }

    #[test]
    fn capture_unsafe_action_from_another_thread_does_not_block_on_capture() {
        let _serial = serialized();
        let capture = CaptureExclusion::acquire();
        let submitted = Arc::new(AtomicBool::new(false));
        let ran = Arc::new(AtomicBool::new(false));
        let submitted_flag = Arc::clone(&submitted);
        let ran_flag = Arc::clone(&ran);
        let worker = std::thread::spawn(move || {
            let deferred = run_or_defer_synchronizing(move || {
                ran_flag.store(true, Ordering::SeqCst);
            });
            assert!(deferred);
            submitted_flag.store(true, Ordering::SeqCst);
        });
        worker.join().expect("deferred submitter panicked");
        assert!(submitted.load(Ordering::SeqCst));
        assert!(!ran.load(Ordering::SeqCst));
        drop(capture);
        assert!(ran.load(Ordering::SeqCst));
    }

    #[test]
    fn nested_captures_on_one_thread_open_and_close_the_gate_once() {
        let _serial = serialized();
        let outer = CaptureExclusion::acquire();
        let inner = CaptureExclusion::acquire();
        drop(inner);

        // The gate must still be shut: dropping the inner guard released a
        // nesting level, not the exclusion.
        let entered = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&entered);
        let waiter = std::thread::spawn(move || {
            let _section = synchronizing_section();
            flag.store(true, Ordering::SeqCst);
        });
        std::thread::sleep(Duration::from_millis(50));
        assert!(
            !entered.load(Ordering::SeqCst),
            "dropping a nested capture guard reopened the gate early"
        );

        drop(outer);
        waiter.join().expect("waiter thread panicked");
        assert!(entered.load(Ordering::SeqCst));
    }

    #[test]
    fn synchronizing_sections_do_not_exclude_one_another() {
        let _serial = serialized();
        let threads = 4;
        // Every thread must be inside its section simultaneously for the
        // barrier to release, so a mutually exclusive gate would hang here.
        let barrier = Arc::new(Barrier::new(threads));
        let peak = Arc::new(AtomicU32::new(0));

        let workers: Vec<_> = (0..threads)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                let peak = Arc::clone(&peak);
                std::thread::spawn(move || {
                    let _section = synchronizing_section().expect("not capturing");
                    peak.fetch_add(1, Ordering::SeqCst);
                    barrier.wait();
                })
            })
            .collect();

        for worker in workers {
            worker.join().expect("worker thread panicked");
        }
        assert_eq!(peak.load(Ordering::SeqCst), threads as u32);
    }

    #[test]
    fn a_nested_synchronizing_section_does_not_queue_behind_a_waiting_capture() {
        let _serial = serialized();
        let outer = synchronizing_section().expect("not capturing on this thread");

        // Park a capture in the waiting state so the gate turns new sections
        // away. It cannot proceed: `outer` is still in flight.
        let capturer = std::thread::spawn(|| {
            let _capture = CaptureExclusion::acquire();
        });
        std::thread::sleep(Duration::from_millis(50));

        // `commit` re-enters itself once per granule of a multi-granule
        // request. Without the shared-depth carve-out this call queues behind
        // the waiting capture, which is itself waiting for `outer` to drain --
        // a deadlock with no escape. Reaching the next line is the assertion.
        let inner = synchronizing_section().expect("not capturing on this thread");
        drop(inner);
        drop(outer);

        capturer.join().expect("capture thread panicked");
    }

    #[test]
    fn nested_run_or_defer_reuses_a_section_while_capture_waits() {
        const MARKER: &str = "ONNX_GENAI_CAPTURE_GATE_NESTED_RUN_CHILD";
        if std::env::var_os(MARKER).is_none() {
            assert_child_does_not_deadlock(
                "nested_run_or_defer_reuses_a_section_while_capture_waits",
                MARKER,
            );
            return;
        }

        let _serial = serialized();
        let outer = synchronizing_section().expect("not capturing on this thread");
        let capturer = std::thread::spawn(|| {
            let _capture = CaptureExclusion::acquire();
        });
        wait_for_capture_waiter();

        let ran = Arc::new(AtomicBool::new(false));
        let ran_in_action = Arc::clone(&ran);
        assert!(!run_or_defer_synchronizing(move || {
            ran_in_action.store(true, Ordering::SeqCst);
        }));
        assert!(ran.load(Ordering::SeqCst));

        drop(outer);
        capturer.join().expect("capture thread panicked");
    }

    #[test]
    fn recursive_deferred_action_reuses_the_drain_synchronizer() {
        const MARKER: &str = "ONNX_GENAI_CAPTURE_GATE_RECURSIVE_DRAIN_CHILD";
        if std::env::var_os(MARKER).is_none() {
            assert_child_does_not_deadlock(
                "recursive_deferred_action_reuses_the_drain_synchronizer",
                MARKER,
            );
            return;
        }

        let _serial = serialized();
        let (queued_tx, queued_rx) = mpsc::channel();
        let (drop_tx, drop_rx) = mpsc::channel();
        let (draining_tx, draining_rx) = mpsc::channel();
        let (continue_tx, continue_rx) = mpsc::channel();
        let (recursive_tx, recursive_rx) = mpsc::channel();

        let drainer = std::thread::spawn(move || {
            let capture = CaptureExclusion::acquire();
            assert!(run_or_defer_synchronizing(move || {
                draining_tx.send(()).expect("announce drain");
                continue_rx.recv().expect("continue recursive action");
                assert!(!run_or_defer_synchronizing(move || {
                    recursive_tx.send(()).expect("recursive action completed");
                }));
            }));
            queued_tx.send(()).expect("announce queued action");
            drop_rx.recv().expect("start capture drop");
            drop(capture);
        });

        queued_rx.recv().expect("deferred action queued");
        drop_tx.send(()).expect("start draining");
        draining_rx.recv().expect("deferred drain started");
        let capturer = std::thread::spawn(|| {
            let _capture = CaptureExclusion::acquire();
        });
        wait_for_capture_waiter();
        continue_tx.send(()).expect("continue recursive action");
        recursive_rx
            .recv()
            .expect("recursive deferred action did not complete");
        drainer.join().expect("drainer thread panicked");
        capturer.join().expect("capture thread panicked");
    }

    #[test]
    fn two_threads_holding_sections_can_both_capture_without_deadlocking() {
        let _serial = serialized();
        let ready = Arc::new(Barrier::new(2));
        let done = Arc::new(AtomicU32::new(0));

        // Both threads hold a long-lived section -- the shape a test harness
        // produces when a section is scoped to a device runtime's lifetime --
        // and then both try to capture. Without setting its own section aside,
        // each waits for the other's to drain and neither ever proceeds.
        let workers: Vec<_> = (0..2)
            .map(|_| {
                let ready = Arc::clone(&ready);
                let done = Arc::clone(&done);
                std::thread::spawn(move || {
                    let _section = synchronizing_section().expect("not capturing");
                    ready.wait();
                    let _capture = CaptureExclusion::acquire();
                    done.fetch_add(1, Ordering::SeqCst);
                })
            })
            .collect();

        for worker in workers {
            worker.join().expect("worker thread panicked or deadlocked");
        }
        assert_eq!(done.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn a_yielded_section_is_restored_when_the_capture_ends() {
        let _serial = serialized();
        let section = synchronizing_section().expect("not capturing on this thread");
        drop(CaptureExclusion::acquire());

        // The capture set this thread's section aside; dropping the capture must
        // put it back, or the count underflows when `section` drops and a later
        // capture no longer waits for anyone.
        assert_eq!(
            lock_state().synchronizers,
            1,
            "the yielded section was not restored when the capture ended"
        );
        drop(section);
        assert_eq!(lock_state().synchronizers, 0);
    }

    #[test]
    fn the_gate_returns_to_idle_after_guards_drop() {
        let _serial = serialized();
        drop(CaptureExclusion::acquire());
        drop(synchronizing_section());
        let state = lock_state();
        assert!(!state.capturing, "capture flag leaked past the guard");
        assert_eq!(state.synchronizers, 0, "synchronizer count leaked");
    }
}

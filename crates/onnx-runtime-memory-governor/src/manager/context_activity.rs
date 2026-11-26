//! One linearization point for admission and lifecycle transitions. Normal
//! transaction entry/exit never takes the cold quiescence-wait mutex.
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};

const LIFECYCLE_SHIFT: u32 = usize::BITS - 2;
const COUNT_MASK: usize = (1usize << LIFECYCLE_SHIFT) - 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub(super) enum ContextLifecycle {
    Active = 0,
    Retiring = 1,
    Lost = 2,
    Terminated = 3,
}
impl ContextLifecycle {
    fn of(state: usize) -> Self {
        match state >> LIFECYCLE_SHIFT {
            0 => Self::Active,
            1 => Self::Retiring,
            2 => Self::Lost,
            _ => Self::Terminated,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AdmissionError {
    Closed,
    CountExhausted,
}

#[derive(Debug, Default)]
pub(super) struct ContextActivity {
    // Upper two bits are lifecycle, remaining bits count live guards. Admission
    // and closure CAS the same word: no new transaction can pass a closed gate.
    state: AtomicUsize,
    wait: Mutex<()>,
    wake: Condvar,
}
impl ContextActivity {
    pub(super) fn snapshot(&self) -> (ContextLifecycle, usize) {
        let state = self.state.load(Ordering::Acquire);
        (ContextLifecycle::of(state), state & COUNT_MASK)
    }

    pub(super) fn enter(&self) -> Result<(), AdmissionError> {
        let mut state = self.state.load(Ordering::Relaxed);
        loop {
            if ContextLifecycle::of(state) != ContextLifecycle::Active {
                return Err(AdmissionError::Closed);
            }
            if state & COUNT_MASK == COUNT_MASK {
                return Err(AdmissionError::CountExhausted);
            }
            match self.state.compare_exchange_weak(
                state,
                state + 1,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(()),
                Err(current) => state = current,
            }
        }
    }

    pub(super) fn leave(&self) {
        // Every count/lifecycle change is an RMW. The final Release therefore
        // extends the release sequences of prior exits, so a zero-count Acquire
        // observes all admitted transactions, not just the final guard's writes.
        let previous = self.state.fetch_sub(1, Ordering::Release);
        debug_assert_ne!(previous & COUNT_MASK, 0, "context transaction underflow");
        if previous & COUNT_MASK == 1 && ContextLifecycle::of(previous) != ContextLifecycle::Active
        {
            // Publish zero before acquiring the waiter's mutex. If the waiter
            // already checked nonzero, it holds this mutex through Condvar::wait;
            // otherwise it observes zero and does not sleep. No lost wakeup.
            let _wait = self
                .wait
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            self.wake.notify_all();
        }
    }

    pub(super) fn mark_retiring(&self) {
        self.transition(|lifecycle| {
            if lifecycle == ContextLifecycle::Active {
                ContextLifecycle::Retiring
            } else {
                lifecycle
            }
        });
    }
    pub(super) fn mark_lost(&self) {
        self.transition(|lifecycle| {
            if lifecycle == ContextLifecycle::Terminated {
                lifecycle
            } else {
                ContextLifecycle::Lost
            }
        });
    }
    pub(super) fn mark_terminated(&self) {
        self.transition(|_| ContextLifecycle::Terminated);
    }
    fn transition(&self, next: impl Fn(ContextLifecycle) -> ContextLifecycle) {
        let _ = self
            .state
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
                let lifecycle = ContextLifecycle::of(state);
                let next = next(lifecycle);
                (next != lifecycle)
                    .then_some((next as usize) << LIFECYCLE_SHIFT | (state & COUNT_MASK))
            });
    }

    /// Only called after closing admission, so reaching zero is terminal for
    /// the count. Acquire observes everything published by departing guards.
    pub(super) fn wait_quiescent(&self) {
        if self.snapshot().1 == 0 {
            return;
        }
        let mut wait = self
            .wait
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while self.snapshot().1 != 0 {
            wait = self
                .wake
                .wait(wait)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier, mpsc};
    use std::time::Duration;

    #[test]
    fn warm_transactions_do_not_touch_the_wait_mutex() {
        let activity = ContextActivity::default();
        // Taking the cold mutex is a positive deadlock control: any accidental
        // hot-path lock prevents this worker from completing before the timeout.
        let wait = activity.wait.lock().unwrap();
        std::thread::scope(|scope| {
            let (tx, rx) = mpsc::channel();
            let activity = &activity;
            scope.spawn(move || {
                for _ in 0..4096 {
                    activity.enter().unwrap();
                    activity.leave();
                }
                tx.send(()).unwrap();
            });
            let result = rx.recv_timeout(Duration::from_secs(5));
            drop(wait); // release even on regression, so scope join cannot hang
            result.expect("Active transaction entry/exit must not take the wait mutex");
        });
        assert_eq!(activity.snapshot(), (ContextLifecycle::Active, 0));
    }

    #[test]
    fn exhaustion_never_overwrites_lifecycle_and_closed_states_are_monotone() {
        let activity = ContextActivity::default();
        activity.state.store(COUNT_MASK, Ordering::Relaxed);
        assert_eq!(activity.enter(), Err(AdmissionError::CountExhausted));
        assert_eq!(activity.snapshot(), (ContextLifecycle::Active, COUNT_MASK));
        activity.mark_retiring();
        assert_eq!(activity.enter(), Err(AdmissionError::Closed));
        activity.leave();
        assert_eq!(
            activity.snapshot(),
            (ContextLifecycle::Retiring, COUNT_MASK - 1)
        );
        activity.mark_lost();
        activity.mark_retiring();
        assert_eq!(
            activity.snapshot(),
            (ContextLifecycle::Lost, COUNT_MASK - 1)
        );
        activity.mark_terminated();
        activity.mark_lost();
        activity.mark_retiring();
        assert_eq!(
            activity.snapshot(),
            (ContextLifecycle::Terminated, COUNT_MASK - 1)
        );
        assert_eq!(activity.enter(), Err(AdmissionError::Closed));
    }

    #[test]
    fn closure_races_admission_without_losing_any_live_count() {
        for _ in 0..64 {
            let activity = Arc::new(ContextActivity::default());
            let start = Barrier::new(5);
            std::thread::scope(|scope| {
                let workers: Vec<_> = (0..4)
                    .map(|_| {
                        scope.spawn(|| {
                            start.wait();
                            activity.enter().is_ok()
                        })
                    })
                    .collect();
                start.wait();
                activity.mark_retiring();
                let admitted = workers
                    .into_iter()
                    .map(|w| w.join().unwrap())
                    .filter(|admitted| *admitted)
                    .count();
                assert_eq!(activity.snapshot(), (ContextLifecycle::Retiring, admitted));
                assert_eq!(activity.enter(), Err(AdmissionError::Closed));
                for _ in 0..admitted {
                    activity.leave();
                }
                activity.wait_quiescent();
            });
        }
    }

    #[test]
    fn quiescence_acquires_writes_from_every_departing_transaction() {
        let activity = ContextActivity::default();
        let payload = std::array::from_fn::<_, 4, _>(|_| AtomicUsize::new(0));
        let start = Barrier::new(5);
        for _ in 0..4 {
            activity.enter().unwrap();
        }
        activity.mark_retiring();
        std::thread::scope(|scope| {
            for (index, value) in payload.iter().enumerate() {
                let activity = &activity;
                let start = &start;
                scope.spawn(move || {
                    start.wait();
                    value.store(index + 1, Ordering::Relaxed);
                    activity.leave();
                });
            }
            start.wait();
            activity.wait_quiescent();
            // Read before scoped join: publication must come from quiescence,
            // rather than thread join or each payload's own atomic ordering.
            for (index, value) in payload.iter().enumerate() {
                assert_eq!(value.load(Ordering::Relaxed), index + 1);
            }
        });
    }

    #[test]
    fn last_exit_wakes_all_waiters_including_racing_wait_entry() {
        for _ in 0..64 {
            let activity = ContextActivity::default();
            activity.enter().unwrap();
            activity.mark_lost();
            let start = Barrier::new(3);
            std::thread::scope(|scope| {
                let (tx, rx) = mpsc::channel();
                for _ in 0..2 {
                    let tx = tx.clone();
                    let start = &start;
                    let activity = &activity;
                    scope.spawn(move || {
                        start.wait();
                        activity.wait_quiescent();
                        tx.send(()).unwrap();
                    });
                }
                start.wait();
                activity.leave();
                for _ in 0..2 {
                    let result = rx.recv_timeout(Duration::from_secs(5));
                    if result.is_err() {
                        // A lost-wakeup regression must fail, not hang forever
                        // while scoped joining waits for the sleeping worker.
                        let _wait = activity.wait.lock().unwrap();
                        activity.wake.notify_all();
                    }
                    result.expect("last exit must wake every waiter");
                }
            });
            assert_eq!(activity.snapshot(), (ContextLifecycle::Lost, 0));
        }
    }
}

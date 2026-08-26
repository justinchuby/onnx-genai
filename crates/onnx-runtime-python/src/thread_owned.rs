//! Own a thread-affine value on one thread and reach it by message.
//!
//! ## Why this exists
//!
//! [`onnx_genai_engine::Engine`] is `!Send` on purpose. Its worker state group
//! (`WorkerRuntimeState`) carries `Rc`, `RefCell` and raw ORT handles behind a
//! `ThreadBound = PhantomData<*const ()>` marker whose stated job is to make the
//! compiler "refuse to let a future pool hand it to a second thread".
//!
//! A `#[pyclass]` must be `Send + Sync`: CPython can call any method from any
//! thread. Until #2132 those two facts coexisted only because the engine crate
//! also carried `unsafe impl Send for Engine`, which cancelled the marker from
//! one layer up. That impl named its own expiry condition — "this would stop
//! being sound if an execution provider introduced a non-migratable handle" —
//! and #2132, which shards ORT sessions across owner workers, is that
//! condition. Removing it was correct and left this crate uncompilable.
//!
//! Re-asserting `unsafe impl Send` here would be strictly worse than where it
//! was: this crate cannot see ORT worker affinity at all.
//!
//! ## The property that replaces the unsafe impl
//!
//! `ThreadOwned<T>` is `Send + Sync` **for every `T`, including `!Send` and
//! `!Sync` ones, with no `unsafe` anywhere in this file**. That is not an
//! assertion, it is the ordinary auto-trait derivation: the struct holds no
//! `T` and no `PhantomData<T>`, only a channel of
//! `Box<dyn FnOnce(&mut T) + Send>`, which is `Send` because it says so and
//! regardless of `T`. `T` is built, used and dropped on the worker and never
//! crosses a thread boundary; only the job closure and its return value do, and
//! both carry their own `Send` bounds.
//!
//! The engine is therefore also *dropped* on the thread that owns its ORT
//! handles, which under the old `unsafe impl` it was not.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex, TryLockError};
use std::thread::JoinHandle;

/// What a job did, reported to the worker loop so it can decide whether to
/// keep going. The panic *message* also travels back to the caller on its own
/// reply channel — see [`ThreadOwned::with`] — so that the call that caused the
/// panic never has to race the worker to read it.
enum JobOutcome {
    Completed,
    Panicked(Option<String>),
}

type Job<T> = Box<dyn FnOnce(&mut T) -> JobOutcome + Send + 'static>;

/// Why a call could not reach the owned value.
///
/// Deliberately two cases and not three: from a caller's point of view a worker
/// that panicked, a worker that exited and a lock poisoned by a panic in a
/// previous call are the same situation — this handle is spent and a new one is
/// needed. Distinguishing them would offer a choice that does not exist. The
/// panic *message* is a different matter and is carried along, because losing it
/// would be a debuggability regression against the `Mutex` this replaced: there,
/// a panicking call unwound into the caller and Python saw the payload.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum CallError {
    /// Another thread is inside a call. Reported instead of queueing, because
    /// the Python-facing contract is "not re-entrant, fails fast", not "serial".
    InUse,
    /// The owning thread is gone, or a previous call panicked.
    WorkerLost {
        /// The panic payload, when the worker died of one and it was a string.
        panic: Option<String>,
    },
}

/// Why a `ThreadOwned` could not be created.
#[derive(Debug)]
pub(crate) enum SpawnError<E> {
    /// The OS refused a thread.
    Thread(std::io::Error),
    /// The value's own constructor returned an error.
    Build(E),
    /// The value's own constructor panicked. Distinct from [`Self::Thread`]
    /// because the thread did start — reporting a construction panic as a
    /// spawn failure would name the wrong subsystem and discard the payload.
    BuildPanicked(Option<String>),
}

/// A value that lives on, and only on, a thread of its own.
pub(crate) struct ThreadOwned<T> {
    /// `None` once [`Drop`] has closed the channel.
    ///
    /// The mutex is doing two jobs: it is the non-re-entrancy token whose
    /// `try_lock` produces [`CallError::InUse`], and it makes `Sync` hold
    /// without depending on `Sender`'s own `Sync` impl.
    jobs: Mutex<Option<Sender<Job<T>>>>,
    /// Set by the worker just before it dies of a panic, read by whichever
    /// caller notices. Shared rather than returned because the caller that
    /// *caused* the panic and the callers that merely arrive afterwards both
    /// deserve the message.
    last_panic: Arc<Mutex<Option<String>>>,
    worker: Option<JoinHandle<()>>,
}

/// The panic payload as a string, for the two shapes `panic!` produces.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> Option<String> {
    if let Some(text) = payload.downcast_ref::<&'static str>() {
        Some((*text).to_owned())
    } else {
        payload.downcast_ref::<String>().cloned()
    }
}

impl<T: 'static> ThreadOwned<T> {
    /// Run `build` on a fresh thread and keep whatever it returns there.
    ///
    /// `build` runs on the worker, not on the caller, so a value that is only
    /// valid on its constructing thread is valid where it is used.
    pub(crate) fn new<F, E>(name: &str, build: F) -> Result<Self, SpawnError<E>>
    where
        F: FnOnce() -> Result<T, E> + Send + 'static,
        E: Send + 'static,
    {
        let (jobs_tx, jobs_rx) = mpsc::channel::<Job<T>>();
        let (init_tx, init_rx) = mpsc::channel::<Result<(), E>>();
        let last_panic = Arc::new(Mutex::new(None::<String>));
        let worker_panic = Arc::clone(&last_panic);

        let worker = std::thread::Builder::new()
            .name(name.to_owned())
            .spawn(move || {
                let mut value = match build() {
                    Ok(value) => {
                        if init_tx.send(Ok(())).is_err() {
                            // Nobody is left to hand it to; drop it here, which
                            // is the only thread allowed to.
                            return;
                        }
                        value
                    }
                    Err(err) => {
                        let _ = init_tx.send(Err(err));
                        return;
                    }
                };
                while let Ok(job) = jobs_rx.recv() {
                    // The job catches its own panic and reports it here, so the
                    // caller reads the message from its own reply channel
                    // rather than racing this write.
                    if let JobOutcome::Panicked(message) = job(&mut value) {
                        if let Ok(mut slot) = worker_panic.lock() {
                            *slot = message;
                        }
                        break;
                    }
                }
                // Reached on both exits: the channel closing, and the `break`
                // above. A panicking job's possibly-inconsistent value is
                // released here, on the only thread allowed to release it.
                drop(value);
            })
            .map_err(SpawnError::Thread)?;

        match init_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                jobs: Mutex::new(Some(jobs_tx)),
                last_panic,
                worker: Some(worker),
            }),
            Ok(Err(err)) => Err(SpawnError::Build(err)),
            // The worker died inside `build`. Almost always a panic; join to
            // recover the payload rather than reporting a spawn failure the
            // thread plainly did not have.
            Err(_) => match worker.join() {
                Err(payload) => Err(SpawnError::BuildPanicked(panic_message(payload.as_ref()))),
                Ok(()) => Err(SpawnError::Thread(std::io::Error::other(
                    "the owning thread exited before constructing the value",
                ))),
            },
        }
    }

    /// Run `job` against the owned value on its own thread and return its result.
    ///
    /// Blocks the calling thread until the job finishes. Callers that hold the
    /// GIL must release it first, exactly as they had to when the work ran
    /// inline.
    pub(crate) fn with<R, F>(&self, job: F) -> Result<R, CallError>
    where
        F: FnOnce(&mut T) -> R + Send + 'static,
        R: Send + 'static,
    {
        let guard = self.jobs.try_lock().map_err(|err| match err {
            TryLockError::WouldBlock => CallError::InUse,
            TryLockError::Poisoned(_) => self.worker_lost(),
        })?;
        let sender = guard.as_ref().ok_or_else(|| self.worker_lost())?;

        let (result_tx, result_rx) = mpsc::channel::<Result<R, Option<String>>>();
        let boxed: Job<T> = Box::new(move |value| {
            // `AssertUnwindSafe` is honest rather than convenient: a panicking
            // job ends the worker loop, so the possibly-inconsistent value is
            // dropped and never observed again. That is the bargain a poisoned
            // `Mutex` strikes, and it is why the caller's answer must be sent
            // from inside this frame — the reply and the panic message then
            // travel together instead of through two unordered paths.
            match catch_unwind(AssertUnwindSafe(|| job(value))) {
                Ok(result) => {
                    let _ = result_tx.send(Ok(result));
                    JobOutcome::Completed
                }
                Err(payload) => {
                    let message = panic_message(payload.as_ref());
                    let _ = result_tx.send(Err(message.clone()));
                    JobOutcome::Panicked(message)
                }
            }
        });
        sender.send(boxed).map_err(|_| self.worker_lost())?;
        match result_rx.recv() {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(panic)) => Err(CallError::WorkerLost { panic }),
            // The worker died without answering — it was already gone, or it
            // unwound somewhere this frame does not cover.
            Err(_) => Err(self.worker_lost()),
        }
    }

    /// The lost-worker error, carrying the panic message if there was one.
    ///
    /// Only for callers that did *not* cause the panic: the worker's write to
    /// `last_panic` happens before it drops the job channel, which is what makes
    /// their `send` fail, so they observe it.
    fn worker_lost(&self) -> CallError {
        CallError::WorkerLost {
            panic: self.last_panic.lock().ok().and_then(|slot| slot.clone()),
        }
    }
}

impl<T> Drop for ThreadOwned<T> {
    fn drop(&mut self) {
        // Close the channel first so the worker's `recv` ends and it drops the
        // value, then wait for that drop to finish. Teardown order is the whole
        // reason to join: ORT handles must be released before the process (or a
        // later engine) moves on.
        match self.jobs.get_mut() {
            Ok(slot) => *slot = None,
            Err(poisoned) => *poisoned.into_inner() = None,
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::convert::Infallible;
    use std::marker::PhantomData;
    use std::rc::Rc;
    use std::sync::Arc;
    use std::sync::mpsc;
    use std::thread::ThreadId;

    use super::{CallError, SpawnError, ThreadOwned};

    /// The point of the type: a `!Send`, `!Sync` payload in a `Send + Sync`
    /// handle. `Rc<Cell<_>>` stands in for the engine's `Rc`/`RefCell` worker
    /// state — if this stops compiling, the property this file exists for is
    /// gone, whatever the engine crate does.
    #[test]
    fn handle_is_send_and_sync_around_a_payload_that_is_neither() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ThreadOwned<Rc<Cell<i32>>>>();

        // The negative half needs a probe rather than a bare turbofish:
        // `fn assert_not_send<T>()` compiles for every `T`, so it would keep
        // passing if the payload silently became `Send` -- it asserts nothing.
        // `Probe<T>`'s inherent `is_send` exists only for `T: Send` and wins
        // method resolution when it applies; otherwise the call falls through
        // to the trait impl, which answers `false`.
        struct Probe<T>(PhantomData<T>);
        trait MaybeSend {
            fn is_send(&self) -> bool {
                false
            }
        }
        impl<T> MaybeSend for Probe<T> {}
        impl<T: Send> Probe<T> {
            fn is_send(&self) -> bool {
                true
            }
        }

        assert!(
            Probe::<u32>(PhantomData).is_send(),
            "positive control: if a plainly-Send type reads as !Send the probe is broken, \
             and every negative below is meaningless"
        );
        assert!(
            !Probe::<Rc<Cell<i32>>>(PhantomData).is_send(),
            "the payload must really be !Send or this test proves nothing about ThreadOwned"
        );
    }

    #[test]
    fn owned_value_is_built_used_and_dropped_on_the_worker_thread() {
        struct Recorder {
            built_on: ThreadId,
            dropped_on: mpsc::Sender<ThreadId>,
            _not_send: Rc<()>,
        }
        impl Drop for Recorder {
            fn drop(&mut self) {
                let _ = self.dropped_on.send(std::thread::current().id());
            }
        }

        let (drop_tx, drop_rx) = mpsc::channel();
        let caller = std::thread::current().id();

        let owned = ThreadOwned::new("probe", move || {
            Ok::<_, Infallible>(Recorder {
                built_on: std::thread::current().id(),
                dropped_on: drop_tx,
                _not_send: Rc::new(()),
            })
        })
        .expect("spawn");

        let (built_on, used_on) = owned
            .with(|value| (value.built_on, std::thread::current().id()))
            .expect("call");
        assert_ne!(built_on, caller, "value was built on the calling thread");
        assert_eq!(used_on, built_on, "value was used off its owning thread");

        drop(owned);
        let dropped_on = drop_rx.recv().expect("value was never dropped");
        assert_eq!(
            dropped_on, built_on,
            "value was dropped off its owning thread"
        );
    }

    #[test]
    fn drop_waits_for_the_value_to_be_released() {
        let released = Arc::new(std::sync::atomic::AtomicBool::new(false));

        struct Slow(Arc<std::sync::atomic::AtomicBool>);
        impl Drop for Slow {
            fn drop(&mut self) {
                std::thread::sleep(std::time::Duration::from_millis(50));
                self.0.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }

        let flag = Arc::clone(&released);
        let owned =
            ThreadOwned::new("probe", move || Ok::<_, Infallible>(Slow(flag))).expect("spawn");
        drop(owned);

        assert!(
            released.load(std::sync::atomic::Ordering::SeqCst),
            "drop returned before the owned value was released"
        );
    }

    #[test]
    fn a_second_caller_is_told_it_is_in_use_rather_than_queued() {
        let (enter_tx, enter_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();

        let owned =
            Arc::new(ThreadOwned::new("probe", || Ok::<_, Infallible>(0u32)).expect("spawn"));
        let busy = Arc::clone(&owned);
        let holder = std::thread::spawn(move || {
            busy.with(move |_| {
                enter_tx.send(()).expect("nobody waiting");
                release_rx.recv().expect("never released");
            })
        });

        enter_rx.recv().expect("first call never started");
        assert_eq!(owned.with(|value| *value).unwrap_err(), CallError::InUse);

        release_tx.send(()).expect("holder gone");
        holder.join().expect("holder panicked").expect("first call");
        // And the handle is usable again once the first call returns.
        assert_eq!(owned.with(|value| *value), Ok(0));
    }

    /// A panicking call must become an error, and must keep its message: the
    /// `Mutex` this replaced unwound into the caller, so Python saw the payload.
    #[test]
    fn a_panicking_job_reports_a_lost_worker_with_its_message() {
        let owned = ThreadOwned::new("probe", || Ok::<_, Infallible>(0u32)).expect("spawn");

        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let first = owned.with(|_| panic!("job blew up"));
        std::panic::set_hook(previous);

        let expected = CallError::WorkerLost {
            panic: Some("job blew up".to_owned()),
        };
        assert_eq!(first.map(|_: ()| ()).unwrap_err(), expected);
        // Later callers get the same explanation, not a bare "gone".
        assert_eq!(owned.with(|value| *value).unwrap_err(), expected);
    }

    #[test]
    fn a_failing_constructor_is_reported_as_a_build_error() {
        let result = ThreadOwned::<Rc<()>>::new("probe", || Err("no model there"));
        match result {
            Err(SpawnError::Build(err)) => assert_eq!(err, "no model there"),
            Err(other) => panic!("wrong variant: {other:?}"),
            Ok(_) => panic!("a failing constructor produced a handle"),
        }
    }

    /// A constructor that panics is not a failure to start a thread — the
    /// thread started fine. Reporting it as one would name the wrong subsystem
    /// and throw away the only useful detail.
    #[test]
    fn a_panicking_constructor_is_reported_separately_and_keeps_its_message() {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result = ThreadOwned::<Rc<()>>::new("probe", || -> Result<Rc<()>, Infallible> {
            panic!("metadata said 0 layers")
        });
        std::panic::set_hook(previous);

        match result {
            Err(SpawnError::BuildPanicked(panic)) => {
                assert_eq!(panic.as_deref(), Some("metadata said 0 layers"));
            }
            Err(other) => panic!("wrong variant: {other:?}"),
            Ok(_) => panic!("a panicking constructor produced a handle"),
        }
    }
}

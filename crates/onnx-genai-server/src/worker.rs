//! Engine worker identity, handles, and the pool that owns them.
//!
//! An [`Engine`](onnx_genai::Engine) is thread-affine: its ORT sessions, KV
//! pages, and continuous-batch state are created, used, and destroyed on the
//! one thread that owns them. The server therefore never moves an engine
//! between threads — it moves *commands* to the thread that holds it.
//!
//! This module names that thread. A [`WorkerHandle`] is one engine-owning
//! thread plus the command channel that reaches it and the `JoinHandle` needed
//! to observe its exit; a [`WorkerPool`] is the set of those handles, addressed
//! by [`WorkerId`].
//!
//! The pool holds exactly one worker today, and every session is placed on
//! [`WorkerId::PRIMARY`]. Nothing here shards work or runs two engines: the
//! shape exists so a session can name the worker that owns its KV state, which
//! is the fact a second worker would need and the current code cannot express.
use std::{
    fmt,
    sync::{Arc, Mutex, MutexGuard, PoisonError},
    thread,
};

use onnx_genai::SessionId;
use tokio::sync::mpsc;

use crate::driver::DriverCommand;

/// Identifies one engine worker inside a [`WorkerPool`].
///
/// A worker id is not an engine [`SessionId`] and not an index into any other
/// collection, so it is a newtype rather than a bare integer (see `RULES.md`
/// rule 5): the two are transposable at every call site that carries both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct WorkerId(usize);

impl WorkerId {
    /// The worker every session is placed on while the pool holds one worker.
    pub(crate) const PRIMARY: Self = Self(0);

    /// Only tests name a worker other than the primary today; production code
    /// reads an id back out of a session placement rather than minting one.
    #[cfg(test)]
    pub(crate) const fn new(index: usize) -> Self {
        Self(index)
    }

    pub(crate) const fn index(self) -> usize {
        self.0
    }
}

impl fmt::Display for WorkerId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// Where a client's conversation lives: the worker that owns it, and the engine
/// session id inside that worker.
///
/// An engine session id is only meaningful to the engine that issued it, so it
/// never travels alone. Carrying the pair means a later turn is routed back to
/// the thread that holds the conversation's KV state instead of being sent to
/// whichever worker answers first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SessionPlacement {
    /// The worker that owns this session.
    pub(crate) worker: WorkerId,
    /// The engine session id, valid only on `worker`.
    pub(crate) engine_session_id: SessionId,
}

impl SessionPlacement {
    pub(crate) const fn new(worker: WorkerId, engine_session_id: SessionId) -> Self {
        Self {
            worker,
            engine_session_id,
        }
    }
}

/// Why a command could not be handed to a worker.
///
/// Kept distinct from "the engine refused the work": nothing was attempted, and
/// the caller learns which worker was unreachable and why (see `RULES.md` rule 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WorkerUnavailable {
    /// No worker with this id is in the pool.
    Unknown { worker: WorkerId, pool_size: usize },
    /// The worker's command channel has been shut down; its thread is exiting
    /// or has exited.
    Stopped(WorkerId),
}

impl fmt::Display for WorkerUnavailable {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown { worker, pool_size } => write!(
                formatter,
                "engine worker {worker} is not in this pool ({pool_size} worker(s) loaded)"
            ),
            // Deliberately the wording every caller already surfaces for a
            // closed command channel: a stopped worker is a stopped driver as
            // far as a request is concerned.
            Self::Stopped(_) => formatter.write_str("engine driver stopped"),
        }
    }
}

impl std::error::Error for WorkerUnavailable {}

/// One engine-owning thread, its command channel, and its `JoinHandle`.
///
/// Held behind an `Arc` so every clone of the driver addresses the same thread.
/// Both the sender and the join handle live behind mutexes because
/// [`shutdown`](Self::shutdown) has to be callable from any clone, any number of
/// times, and still mean "the engine has been destroyed" when it returns.
pub(crate) struct WorkerHandle {
    id: WorkerId,
    /// The one long-lived command sender. Dropping it (via `shutdown` or this
    /// handle's own `Drop`) closes the channel, which is how the worker loop is
    /// told to finish: it drains what is queued, returns, and destroys the
    /// engine on its own thread.
    commands: Mutex<Option<mpsc::Sender<DriverCommand>>>,
    /// `None` once the thread has been joined (or once this handle was dropped
    /// without an explicit shutdown, which detaches it).
    join: Mutex<Option<thread::JoinHandle<()>>>,
}

impl WorkerHandle {
    /// Spawn the worker thread that will own the engine.
    ///
    /// `run` receives nothing and returns nothing on purpose: everything it
    /// needs — the engine, the command receiver — is moved into it by the
    /// caller, because that move is the point at which the engine becomes
    /// thread-affine.
    pub(crate) fn spawn<F>(
        id: WorkerId,
        thread_name: String,
        commands: mpsc::Sender<DriverCommand>,
        run: F,
    ) -> Arc<Self>
    where
        F: FnOnce() + Send + 'static,
    {
        let join = thread::Builder::new()
            .name(thread_name)
            .spawn(run)
            .expect("failed to spawn onnx-genai engine driver");
        Arc::new(Self {
            id,
            commands: Mutex::new(Some(commands)),
            join: Mutex::new(Some(join)),
        })
    }

    /// A handle to a worker that was never spawned, for tests that drive the
    /// command channel themselves and never start an engine.
    #[cfg(test)]
    pub(crate) fn detached(id: WorkerId, commands: mpsc::Sender<DriverCommand>) -> Arc<Self> {
        Arc::new(Self {
            id,
            commands: Mutex::new(Some(commands)),
            join: Mutex::new(None),
        })
    }

    pub(crate) fn id(&self) -> WorkerId {
        self.id
    }

    /// A sender for this worker, or the reason it cannot be reached.
    ///
    /// Returns a clone rather than a borrow so a caller can `await` a send
    /// without holding the lock across the await point.
    pub(crate) fn sender(&self) -> Result<mpsc::Sender<DriverCommand>, WorkerUnavailable> {
        lock(&self.commands)
            .clone()
            .ok_or(WorkerUnavailable::Stopped(self.id))
    }

    /// Whether this worker still accepts commands.
    pub(crate) fn is_running(&self) -> bool {
        lock(&self.commands).is_some()
    }

    /// Close the command channel and wait for the worker thread to exit.
    ///
    /// Idempotent, and safe to call from several threads at once: the second
    /// caller blocks on the same lock the joiner holds, so when *any* call
    /// returns, the thread is gone and the engine — with its ORT sessions, KV
    /// pages, and device allocations — has been destroyed on the thread that
    /// created it.
    ///
    /// **Blocking.** Never call this from an async task without
    /// `spawn_blocking`: it waits for the worker to finish the command it is
    /// running, which may be a full generation.
    pub(crate) fn shutdown(&self) {
        let closed = lock(&self.commands).take().is_some();
        let mut join = lock(&self.join);
        if let Some(handle) = join.take() {
            if handle.join().is_err() {
                tracing::error!(worker = %self.id, "engine worker thread panicked");
            } else if closed {
                tracing::debug!(worker = %self.id, "engine worker stopped");
            }
        }
    }

    /// Whether the worker thread has been joined by an explicit shutdown.
    #[cfg(test)]
    pub(crate) fn is_joined(&self) -> bool {
        lock(&self.join).is_none()
    }
}

impl Drop for WorkerHandle {
    /// Release the worker without waiting for it.
    ///
    /// Dropping the sender closes the channel, so the thread still finishes its
    /// queued commands and destroys the engine on its own thread — the affinity
    /// contract holds either way. What an implicit drop does *not* give is
    /// ordering: the last owner may be an async task, and blocking a runtime
    /// thread on a generation-length join is not something a `Drop` should do
    /// behind the caller's back. Callers that need the engine gone before they
    /// continue — model unload, registry teardown — call
    /// [`shutdown`](Self::shutdown) explicitly.
    fn drop(&mut self) {
        if lock(&self.commands).take().is_some() {
            tracing::debug!(
                worker = %self.id,
                "engine worker released without an explicit shutdown; not joined",
            );
        }
    }
}

impl fmt::Debug for WorkerHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WorkerHandle")
            .field("id", &self.id)
            .field("running", &self.is_running())
            .finish()
    }
}

/// The set of engine workers a driver commands.
///
/// One worker today. The pool is addressed by [`WorkerId`] rather than by "the
/// sender", so every session-bound command already names the thread it must
/// reach; growing the pool is then a question of how a *new* session picks a
/// worker, not of rewriting every call site.
#[derive(Clone, Debug)]
pub(crate) struct WorkerPool {
    workers: Arc<Vec<Arc<WorkerHandle>>>,
}

impl WorkerPool {
    /// A pool of exactly one worker, which owns the whole engine.
    pub(crate) fn single(worker: Arc<WorkerHandle>) -> Self {
        Self {
            workers: Arc::new(vec![worker]),
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.workers.len()
    }

    /// The worker a new session is placed on.
    ///
    /// With one worker this is that worker. It is a named method rather than an
    /// index so the placement policy has exactly one home when there is a
    /// choice to make.
    pub(crate) fn placement_worker(&self) -> WorkerId {
        self.primary().id()
    }

    /// The pool's first worker, which every non-session command goes to.
    pub(crate) fn primary(&self) -> &Arc<WorkerHandle> {
        self.workers
            .first()
            .expect("worker pool is constructed with at least one worker")
    }

    pub(crate) fn worker(&self, id: WorkerId) -> Result<&Arc<WorkerHandle>, WorkerUnavailable> {
        self.workers
            .get(id.index())
            .filter(|worker| worker.id() == id)
            .ok_or(WorkerUnavailable::Unknown {
                worker: id,
                pool_size: self.len(),
            })
    }

    /// A sender for the worker that owns a session's state.
    pub(crate) fn sender_for(
        &self,
        id: WorkerId,
    ) -> Result<mpsc::Sender<DriverCommand>, WorkerUnavailable> {
        self.worker(id)?.sender()
    }

    /// A sender for the worker that serves commands bound to no session.
    pub(crate) fn primary_sender(&self) -> Result<mpsc::Sender<DriverCommand>, WorkerUnavailable> {
        self.primary().sender()
    }

    /// Shut down and join every worker in the pool.
    ///
    /// **Blocking**, and idempotent, with the same contract as
    /// [`WorkerHandle::shutdown`].
    pub(crate) fn shutdown(&self) {
        for worker in self.workers.iter() {
            worker.shutdown();
        }
    }
}

/// A worker lock is only ever held across a `take`, a `clone`, or the join
/// itself, so a poisoned lock carries state that is still sound to act on:
/// refusing to shut a worker down because an unrelated thread panicked would
/// leak the engine instead of destroying it.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    /// Spawn a worker whose "engine" is a counter, so a test can prove the
    /// thread ran and exited without loading a model.
    fn counting_worker(
        counter: Arc<AtomicUsize>,
    ) -> (Arc<WorkerHandle>, mpsc::Sender<DriverCommand>) {
        let (commands, mut rx) = mpsc::channel(4);
        let worker = WorkerHandle::spawn(
            WorkerId::PRIMARY,
            "worker-test".to_string(),
            commands.clone(),
            move || {
                while rx.blocking_recv().is_some() {
                    counter.fetch_add(1, Ordering::SeqCst);
                }
                counter.fetch_add(100, Ordering::SeqCst);
            },
        );
        (worker, commands)
    }

    #[test]
    fn shutdown_joins_the_worker_thread_and_is_idempotent() {
        let counter = Arc::new(AtomicUsize::new(0));
        let (worker, extra_sender) = counting_worker(Arc::clone(&counter));
        drop(extra_sender);

        assert!(worker.is_running());
        assert!(!worker.is_joined());

        worker.shutdown();
        assert!(worker.is_joined(), "shutdown must join the worker thread");
        assert!(!worker.is_running());
        assert_eq!(
            counter.load(Ordering::SeqCst),
            100,
            "the worker loop must have returned before shutdown did",
        );

        // Second and third calls are no-ops rather than a double join panic.
        worker.shutdown();
        worker.shutdown();
        assert!(worker.is_joined());
    }

    #[test]
    fn a_stopped_worker_reports_a_stopped_driver() {
        let counter = Arc::new(AtomicUsize::new(0));
        let (worker, extra_sender) = counting_worker(counter);
        drop(extra_sender);
        worker.shutdown();

        let error = worker.sender().expect_err("a stopped worker has no sender");
        assert_eq!(error, WorkerUnavailable::Stopped(WorkerId::PRIMARY));
        assert_eq!(error.to_string(), "engine driver stopped");
    }

    /// An implicit drop still releases the engine on its own thread — it just
    /// does not wait for it. The worker keeps running while any sender a caller
    /// already took out lives, and exits once the last one goes.
    #[test]
    fn dropping_a_handle_closes_the_channel_without_joining() {
        let counter = Arc::new(AtomicUsize::new(0));
        let (worker, extra_sender) = counting_worker(Arc::clone(&counter));
        drop(extra_sender);
        let sender = worker.sender().expect("worker is running");

        drop(worker);
        assert_eq!(counter.load(Ordering::SeqCst), 0);

        drop(sender);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while counter.load(Ordering::SeqCst) != 100 {
            assert!(
                std::time::Instant::now() < deadline,
                "worker must exit once its last sender is dropped",
            );
            thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    #[test]
    fn a_pool_of_one_routes_by_worker_id() {
        let (commands, _rx) = mpsc::channel(1);
        let pool = WorkerPool::single(WorkerHandle::detached(WorkerId::PRIMARY, commands));

        assert_eq!(pool.len(), 1);
        assert_eq!(pool.placement_worker(), WorkerId::PRIMARY);
        assert!(pool.sender_for(WorkerId::PRIMARY).is_ok());

        let error = pool
            .sender_for(WorkerId::new(1))
            .expect_err("a pool of one has no second worker");
        assert_eq!(
            error,
            WorkerUnavailable::Unknown {
                worker: WorkerId::new(1),
                pool_size: 1,
            }
        );
        assert_eq!(
            error.to_string(),
            "engine worker 1 is not in this pool (1 worker(s) loaded)"
        );
    }

    #[test]
    fn shutting_down_a_pool_stops_every_worker() {
        let counter = Arc::new(AtomicUsize::new(0));
        let (worker, extra_sender) = counting_worker(Arc::clone(&counter));
        drop(extra_sender);
        let pool = WorkerPool::single(worker);

        pool.shutdown();

        assert_eq!(counter.load(Ordering::SeqCst), 100);
        assert!(matches!(
            pool.primary_sender(),
            Err(WorkerUnavailable::Stopped(WorkerId::PRIMARY))
        ));
        pool.shutdown();
    }
}

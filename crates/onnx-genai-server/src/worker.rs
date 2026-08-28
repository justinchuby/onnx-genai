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
//! The pool defaults to one worker and may opt into multiple supported ORT
//! workers. Session placements always name the worker that owns their KV state.
use std::{
    any::Any,
    fmt,
    panic::{self, AssertUnwindSafe},
    sync::{Arc, Mutex, MutexGuard, PoisonError, mpsc as std_mpsc},
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
    /// The sole worker in the default configuration.
    pub(crate) const PRIMARY: Self = Self(0);

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
    /// The worker thread panicked or otherwise failed after startup. Sessions
    /// owned by this worker are lost and are never migrated implicitly.
    Failed(WorkerId),
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
            Self::Failed(worker) => write!(
                formatter,
                "engine worker {worker} failed; sessions placed on this worker are unavailable \
                 and must be recreated"
            ),
        }
    }
}

impl std::error::Error for WorkerUnavailable {}

/// Why an engine worker could not become ready.
#[derive(Debug)]
pub(crate) enum WorkerStartError<E> {
    /// The worker thread could not be created.
    ThreadSpawn(std::io::Error),
    /// The worker thread ran the supplied initialization plan, which failed.
    Initialization(E),
    /// Initialization panicked after the worker thread started.
    InitializationPanicked(String),
    /// The worker exited without reporting either readiness or an error.
    InitializationChannelClosed,
}

impl<E: fmt::Display> fmt::Display for WorkerStartError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ThreadSpawn(error) => write!(formatter, "failed to spawn engine worker: {error}"),
            Self::Initialization(error) => {
                write!(formatter, "engine worker initialization failed: {error}")
            }
            Self::InitializationPanicked(message) => {
                write!(
                    formatter,
                    "engine worker initialization panicked: {message}"
                )
            }
            Self::InitializationChannelClosed => formatter
                .write_str("engine worker exited before reporting its initialization result"),
        }
    }
}

impl<E> std::error::Error for WorkerStartError<E>
where
    E: std::error::Error + 'static,
{
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ThreadSpawn(error) => Some(error),
            Self::Initialization(error) => Some(error),
            Self::InitializationPanicked(_) | Self::InitializationChannelClosed => None,
        }
    }
}

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
    state: Arc<WorkerState>,
}

struct WorkerState {
    lifecycle: Mutex<WorkerLifecycle>,
}

struct WorkerLifecycle {
    health: WorkerHealth,
    live_sessions: usize,
    pending_sessions: usize,
    active_turns: usize,
}

impl WorkerState {
    fn new() -> Self {
        Self {
            lifecycle: Mutex::new(WorkerLifecycle {
                health: WorkerHealth::Starting,
                live_sessions: 0,
                pending_sessions: 0,
                active_turns: 0,
            }),
        }
    }

    fn health(&self) -> WorkerHealth {
        lock(&self.lifecycle).health
    }

    fn mark_healthy(&self) {
        lock(&self.lifecycle).health = WorkerHealth::Healthy;
    }

    fn mark_stopped(&self) {
        let mut lifecycle = lock(&self.lifecycle);
        if lifecycle.health != WorkerHealth::Failed {
            lifecycle.health = WorkerHealth::Stopped;
        }
        lifecycle.live_sessions = 0;
        lifecycle.pending_sessions = 0;
        lifecycle.active_turns = 0;
    }

    fn mark_failed(&self) {
        let mut lifecycle = lock(&self.lifecycle);
        lifecycle.health = WorkerHealth::Failed;
        lifecycle.live_sessions = 0;
        lifecycle.active_turns = 0;
    }

    fn placement_load(&self) -> Option<usize> {
        let lifecycle = lock(&self.lifecycle);
        (lifecycle.health == WorkerHealth::Healthy).then(|| {
            lifecycle
                .live_sessions
                .saturating_add(lifecycle.pending_sessions)
        })
    }

    fn reserve_session(
        self: &Arc<Self>,
        worker: WorkerId,
    ) -> Result<SessionPlacementReservation, WorkerUnavailable> {
        let mut lifecycle = lock(&self.lifecycle);
        match lifecycle.health {
            WorkerHealth::Healthy => {
                lifecycle.pending_sessions += 1;
                Ok(SessionPlacementReservation {
                    worker,
                    state: Arc::clone(self),
                    committed: false,
                })
            }
            WorkerHealth::Failed => Err(WorkerUnavailable::Failed(worker)),
            WorkerHealth::Starting | WorkerHealth::Stopped => {
                Err(WorkerUnavailable::Stopped(worker))
            }
        }
    }

    fn commit_session(
        self: &Arc<Self>,
        worker: WorkerId,
    ) -> Result<CommittedSession, WorkerUnavailable> {
        let mut lifecycle = lock(&self.lifecycle);
        checked_decrement(&mut lifecycle.pending_sessions, worker, "pending-session");
        match lifecycle.health {
            WorkerHealth::Healthy => {
                lifecycle.live_sessions += 1;
                Ok(CommittedSession {
                    worker,
                    state: Arc::clone(self),
                    armed: true,
                })
            }
            WorkerHealth::Failed => Err(WorkerUnavailable::Failed(worker)),
            WorkerHealth::Starting | WorkerHealth::Stopped => {
                Err(WorkerUnavailable::Stopped(worker))
            }
        }
    }

    fn release_pending_session(&self, worker: WorkerId) {
        checked_decrement(
            &mut lock(&self.lifecycle).pending_sessions,
            worker,
            "pending-session",
        );
    }

    fn reserve_turn(
        self: &Arc<Self>,
        worker: WorkerId,
    ) -> Result<WorkerTurnGuard, WorkerUnavailable> {
        let mut lifecycle = lock(&self.lifecycle);
        match lifecycle.health {
            WorkerHealth::Healthy => {
                lifecycle.active_turns += 1;
                Ok(WorkerTurnGuard {
                    worker,
                    state: Arc::clone(self),
                })
            }
            WorkerHealth::Failed => Err(WorkerUnavailable::Failed(worker)),
            WorkerHealth::Starting | WorkerHealth::Stopped => {
                Err(WorkerUnavailable::Stopped(worker))
            }
        }
    }

    fn release_turn(&self, worker: WorkerId) {
        checked_decrement(
            &mut lock(&self.lifecycle).active_turns,
            worker,
            "active-turn",
        );
    }

    fn release_live_session(&self, worker: WorkerId) {
        checked_decrement(
            &mut lock(&self.lifecycle).live_sessions,
            worker,
            "live-session",
        );
    }

    fn status(&self, id: WorkerId) -> WorkerStatusSnapshot {
        let lifecycle = lock(&self.lifecycle);
        WorkerStatusSnapshot {
            id,
            active_turns: lifecycle.active_turns,
            live_sessions: lifecycle.live_sessions,
            health: lifecycle.health,
        }
    }
}

fn checked_decrement(value: &mut usize, worker: WorkerId, counter: &'static str) {
    if let Some(next) = value.checked_sub(1) {
        *value = next;
    } else {
        tracing::warn!(
            worker = %worker,
            counter,
            "ignored a stale worker counter release after lifecycle invalidation"
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WorkerHealth {
    Starting,
    Healthy,
    Stopped,
    Failed,
}

impl WorkerHealth {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Starting => "starting",
            Self::Healthy => "healthy",
            Self::Stopped => "stopped",
            Self::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WorkerStatusSnapshot {
    pub(crate) id: WorkerId,
    pub(crate) active_turns: usize,
    pub(crate) live_sessions: usize,
    pub(crate) health: WorkerHealth,
}

/// One queued or active turn charged to a worker.
///
/// The counter is incremented before enqueue and released by `Drop`, so send
/// failure, cancellation, backend error, and normal completion share one exact
/// accounting path.
pub(crate) struct WorkerTurnGuard {
    worker: WorkerId,
    state: Arc<WorkerState>,
}

impl Drop for WorkerTurnGuard {
    fn drop(&mut self) {
        self.state.release_turn(self.worker);
    }
}

impl fmt::Debug for WorkerTurnGuard {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("WorkerTurnGuard")
    }
}

/// In-flight session placement. Dropping before commit rolls the pending load
/// back; committing moves it into the live-session count atomically with
/// respect to the placement policy.
pub(crate) struct SessionPlacementReservation {
    worker: WorkerId,
    state: Arc<WorkerState>,
    committed: bool,
}

pub(crate) struct CommittedSession {
    worker: WorkerId,
    state: Arc<WorkerState>,
    armed: bool,
}

impl CommittedSession {
    /// Transfer ownership of the live count to the persistent engine session.
    pub(crate) fn persist(mut self) {
        self.armed = false;
    }
}

impl Drop for CommittedSession {
    fn drop(&mut self) {
        if self.armed {
            self.state.release_live_session(self.worker);
        }
    }
}

pub(crate) struct SessionCloseAccounting {
    worker: WorkerId,
    state: Arc<WorkerState>,
}

impl SessionCloseAccounting {
    /// Release the persistent live-session count after the engine confirms close.
    pub(crate) fn session_closed(self) {
        self.state.release_live_session(self.worker);
    }
}

impl SessionPlacementReservation {
    pub(crate) fn worker(&self) -> WorkerId {
        self.worker
    }

    pub(crate) fn commit(mut self) -> Result<CommittedSession, WorkerUnavailable> {
        self.committed = true;
        self.state.commit_session(self.worker)
    }
}

impl Drop for SessionPlacementReservation {
    fn drop(&mut self) {
        if !self.committed {
            self.state.release_pending_session(self.worker);
        }
    }
}

impl WorkerHandle {
    /// Spawn a worker, initialize its thread-affine state there, and wait until
    /// it reports ready.
    ///
    /// `State` has no `Send` bound: it is created and consumed inside the new
    /// thread and therefore cannot cross the spawn boundary. Only the
    /// construction plan and the small `Ready` payload cross threads.
    pub(crate) fn spawn<State, Ready, Error, Initialize, Run>(
        id: WorkerId,
        thread_name: String,
        commands: mpsc::Sender<DriverCommand>,
        initialize: Initialize,
        run: Run,
    ) -> Result<(Arc<Self>, Ready), WorkerStartError<Error>>
    where
        Ready: Send + 'static,
        Error: Send + 'static,
        Initialize: FnOnce() -> Result<(State, Ready), Error> + Send + 'static,
        Run: FnOnce(State) + Send + 'static,
    {
        if thread_name.as_bytes().contains(&0) {
            return Err(WorkerStartError::ThreadSpawn(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "worker thread name contains an interior NUL byte",
            )));
        }
        let (ready_tx, ready_rx) = std_mpsc::sync_channel(1);
        let state = Arc::new(WorkerState::new());
        let thread_state = Arc::clone(&state);
        let thread_id = id;
        let join = thread::Builder::new()
            .name(thread_name)
            .spawn(
                move || match panic::catch_unwind(AssertUnwindSafe(initialize)) {
                    Ok(Ok((state, ready))) => {
                        thread_state.mark_healthy();
                        if ready_tx.send(Ok(ready)).is_ok() {
                            match panic::catch_unwind(AssertUnwindSafe(|| run(state))) {
                                Ok(()) => {
                                    thread_state.mark_stopped();
                                }
                                Err(payload) => {
                                    thread_state.mark_failed();
                                    tracing::error!(
                                        worker = %thread_id,
                                        panic = %panic_message(payload),
                                        "engine worker failed; its sessions are unavailable",
                                    );
                                }
                            }
                        } else {
                            thread_state.mark_stopped();
                        }
                    }
                    Ok(Err(error)) => {
                        thread_state.mark_failed();
                        let _ = ready_tx.send(Err(WorkerStartError::Initialization(error)));
                    }
                    Err(payload) => {
                        thread_state.mark_failed();
                        let _ = ready_tx.send(Err(WorkerStartError::InitializationPanicked(
                            panic_message(payload),
                        )));
                    }
                },
            )
            .map_err(WorkerStartError::ThreadSpawn)?;
        let (join, ready) = await_initialization(ready_rx, join)?;
        Ok((
            Arc::new(Self {
                id,
                commands: Mutex::new(Some(commands)),
                join: Mutex::new(Some(join)),
                state,
            }),
            ready,
        ))
    }

    /// A handle to a worker that was never spawned, for tests that drive the
    /// command channel themselves and never start an engine.
    #[cfg(test)]
    pub(crate) fn detached(id: WorkerId, commands: mpsc::Sender<DriverCommand>) -> Arc<Self> {
        Arc::new(Self {
            id,
            commands: Mutex::new(Some(commands)),
            join: Mutex::new(None),
            state: Arc::new(WorkerState {
                lifecycle: Mutex::new(WorkerLifecycle {
                    health: WorkerHealth::Healthy,
                    live_sessions: 0,
                    pending_sessions: 0,
                    active_turns: 0,
                }),
            }),
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
        match self.state.health() {
            WorkerHealth::Failed => return Err(WorkerUnavailable::Failed(self.id)),
            WorkerHealth::Stopped => return Err(WorkerUnavailable::Stopped(self.id)),
            WorkerHealth::Starting | WorkerHealth::Healthy => {}
        }
        lock(&self.commands)
            .clone()
            .ok_or(WorkerUnavailable::Stopped(self.id))
    }

    /// Whether this worker still accepts commands.
    pub(crate) fn is_running(&self) -> bool {
        self.state.health() == WorkerHealth::Healthy && lock(&self.commands).is_some()
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
        self.state.mark_stopped();
    }

    /// Whether the worker thread has been joined by an explicit shutdown.
    #[cfg(test)]
    pub(crate) fn is_joined(&self) -> bool {
        lock(&self.join).is_none()
    }

    fn placement_load(&self) -> usize {
        self.state.placement_load().unwrap_or(usize::MAX)
    }

    fn active_turns(&self) -> usize {
        self.status().active_turns
    }

    fn reserve_session(&self) -> Result<SessionPlacementReservation, WorkerUnavailable> {
        self.state.reserve_session(self.id)
    }

    fn reserve_turn(&self) -> Result<WorkerTurnGuard, WorkerUnavailable> {
        self.state.reserve_turn(self.id)
    }

    pub(crate) fn session_close_accounting(&self) -> SessionCloseAccounting {
        SessionCloseAccounting {
            worker: self.id,
            state: Arc::clone(&self.state),
        }
    }

    pub(crate) fn status(&self) -> WorkerStatusSnapshot {
        self.state.status(self.id)
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
/// The pool is addressed by [`WorkerId`] rather than by "the sender", so every
/// session-bound command names the thread it must reach.
#[derive(Clone, Debug)]
pub(crate) struct WorkerPool {
    workers: Arc<Vec<Arc<WorkerHandle>>>,
    selection: Arc<Mutex<()>>,
}

impl WorkerPool {
    /// A pool of exactly one worker, which owns the whole engine.
    #[cfg(test)]
    pub(crate) fn single(worker: Arc<WorkerHandle>) -> Self {
        Self::new(vec![worker])
    }

    pub(crate) fn new(workers: Vec<Arc<WorkerHandle>>) -> Self {
        assert!(
            !workers.is_empty(),
            "worker pool is constructed with at least one worker"
        );
        for (index, worker) in workers.iter().enumerate() {
            assert_eq!(
                worker.id(),
                WorkerId::new(index),
                "worker ids are dense and ordered"
            );
        }
        Self {
            workers: Arc::new(workers),
            selection: Arc::new(Mutex::new(())),
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
    pub(crate) fn reserve_session_placement(
        &self,
    ) -> Result<SessionPlacementReservation, WorkerUnavailable> {
        let _selection = lock(&self.selection);
        loop {
            let worker = self
                .workers
                .iter()
                .filter(|worker| worker.is_running())
                .min_by_key(|worker| (worker.placement_load(), worker.id()))
                .ok_or_else(|| self.no_running_worker_error())?;
            if let Ok(reservation) = worker.reserve_session() {
                return Ok(reservation);
            }
        }
    }

    /// Reserve a child session on the worker that owns its fork source.
    pub(crate) fn reserve_session_on(
        &self,
        worker: WorkerId,
    ) -> Result<SessionPlacementReservation, WorkerUnavailable> {
        self.worker(worker)?.reserve_session()
    }

    /// Reserve one stateless turn on the least-loaded healthy worker.
    pub(crate) fn reserve_stateless_turn(
        &self,
    ) -> Result<(WorkerId, WorkerTurnGuard), WorkerUnavailable> {
        let _selection = lock(&self.selection);
        loop {
            let worker = self
                .workers
                .iter()
                .filter(|worker| worker.is_running())
                .min_by_key(|worker| (worker.active_turns(), worker.id()))
                .ok_or_else(|| self.no_running_worker_error())?;
            if let Ok(turn) = worker.reserve_turn() {
                return Ok((worker.id(), turn));
            }
        }
    }

    pub(crate) fn reserve_turn(&self, id: WorkerId) -> Result<WorkerTurnGuard, WorkerUnavailable> {
        let worker = self.worker(id)?;
        worker.sender()?;
        worker.reserve_turn()
    }

    fn no_running_worker_error(&self) -> WorkerUnavailable {
        self.workers
            .iter()
            .find(|worker| worker.state.health() == WorkerHealth::Failed)
            .map_or(WorkerUnavailable::Stopped(WorkerId::PRIMARY), |worker| {
                WorkerUnavailable::Failed(worker.id())
            })
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

    pub(crate) fn statuses(&self) -> Vec<WorkerStatusSnapshot> {
        self.workers.iter().map(|worker| worker.status()).collect()
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

fn await_initialization<Ready, Error>(
    ready_rx: std_mpsc::Receiver<Result<Ready, WorkerStartError<Error>>>,
    join: thread::JoinHandle<()>,
) -> Result<(thread::JoinHandle<()>, Ready), WorkerStartError<Error>> {
    match ready_rx.recv() {
        Ok(Ok(ready)) => Ok((join, ready)),
        Ok(Err(error)) => {
            let _ = join.join();
            Err(error)
        }
        Err(_) => match join.join() {
            Ok(()) => Err(WorkerStartError::InitializationChannelClosed),
            Err(payload) => Err(WorkerStartError::InitializationPanicked(panic_message(
                payload,
            ))),
        },
    }
}

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        convert::Infallible,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use super::*;

    struct DropThread(std_mpsc::SyncSender<thread::ThreadId>);

    impl Drop for DropThread {
        fn drop(&mut self) {
            let _ = self.0.send(thread::current().id());
        }
    }

    /// Spawn a worker whose "engine" is a counter, so a test can prove the
    /// thread ran and exited without loading a model.
    fn counting_worker(
        counter: Arc<AtomicUsize>,
    ) -> (Arc<WorkerHandle>, mpsc::Sender<DriverCommand>) {
        let (commands, mut rx) = mpsc::channel(4);
        let (worker, ()) = WorkerHandle::spawn(
            WorkerId::PRIMARY,
            "worker-test".to_string(),
            commands.clone(),
            || Ok::<_, Infallible>(((), ())),
            move |()| {
                while rx.blocking_recv().is_some() {
                    counter.fetch_add(1, Ordering::SeqCst);
                }
                counter.fetch_add(100, Ordering::SeqCst);
            },
        )
        .expect("spawn counting worker");
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
        let reservation = pool
            .reserve_session_placement()
            .expect("the primary worker accepts sessions");
        assert_eq!(reservation.worker(), WorkerId::PRIMARY);
        reservation.commit().unwrap().persist();
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

    fn detached_pool(size: usize) -> (WorkerPool, Vec<mpsc::Receiver<DriverCommand>>) {
        let mut receivers = Vec::with_capacity(size);
        let workers = (0..size)
            .map(|index| {
                let (commands, rx) = mpsc::channel(4);
                receivers.push(rx);
                WorkerHandle::detached(WorkerId::new(index), commands)
            })
            .collect();
        (WorkerPool::new(workers), receivers)
    }

    #[test]
    fn session_placement_is_least_loaded_with_lowest_id_ties() {
        let (pool, _receivers) = detached_pool(2);

        let first = pool.reserve_session_placement().unwrap();
        assert_eq!(first.worker(), WorkerId::new(0));
        first.commit().unwrap().persist();
        let second = pool.reserve_session_placement().unwrap();
        assert_eq!(second.worker(), WorkerId::new(1));
        second.commit().unwrap().persist();
        let tied = pool.reserve_session_placement().unwrap();
        assert_eq!(tied.worker(), WorkerId::new(0));
        drop(tied);

        let statuses = pool.statuses();
        assert_eq!(statuses[0].live_sessions, 1);
        assert_eq!(statuses[1].live_sessions, 1);
    }

    #[test]
    fn pending_session_reservations_participate_and_release_exactly() {
        let (pool, _receivers) = detached_pool(2);

        let pending = pool.reserve_session_placement().unwrap();
        assert_eq!(pending.worker(), WorkerId::new(0));
        let next = pool.reserve_session_placement().unwrap();
        assert_eq!(next.worker(), WorkerId::new(1));
        drop(pending);
        drop(next);

        assert_eq!(pool.statuses()[0].live_sessions, 0);
        assert_eq!(pool.statuses()[1].live_sessions, 0);
        assert_eq!(
            pool.reserve_session_placement().unwrap().worker(),
            WorkerId::new(0)
        );
    }

    #[test]
    fn stateless_turns_balance_and_cancellation_releases_counters() {
        let (pool, _receivers) = detached_pool(2);

        let (first_worker, first) = pool.reserve_stateless_turn().unwrap();
        let (second_worker, second) = pool.reserve_stateless_turn().unwrap();
        assert_eq!(first_worker, WorkerId::new(0));
        assert_eq!(second_worker, WorkerId::new(1));
        assert!(
            !pool
                .sender_for(first_worker)
                .unwrap()
                .same_channel(&pool.sender_for(second_worker).unwrap()),
            "different workers have distinct command loops and cannot share a continuous batch"
        );
        assert_eq!(pool.statuses()[0].active_turns, 1);
        assert_eq!(pool.statuses()[1].active_turns, 1);

        drop(first);
        assert_eq!(
            pool.reserve_stateless_turn().unwrap().0,
            WorkerId::new(0),
            "a cancelled turn must return its worker to the least-loaded set"
        );
        drop(second);
        assert_eq!(pool.statuses()[1].active_turns, 0);
    }

    #[test]
    fn failure_is_atomic_with_session_commit_and_stale_release() {
        let (pool, _receivers) = detached_pool(1);
        let worker = Arc::clone(pool.primary());
        let pending = pool.reserve_session_placement().unwrap();
        let turn = pool.reserve_turn(WorkerId::PRIMARY).unwrap();

        worker.state.mark_failed();

        assert!(matches!(
            pending.commit(),
            Err(WorkerUnavailable::Failed(WorkerId::PRIMARY))
        ));
        drop(turn);
        worker.session_close_accounting().session_closed();
        let status = worker.status();
        assert_eq!(status.health, WorkerHealth::Failed);
        assert_eq!(status.live_sessions, 0);
        assert_eq!(status.active_turns, 0);
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

    #[test]
    fn graceful_shutdown_clears_worker_counts() {
        let counter = Arc::new(AtomicUsize::new(0));
        let (worker, extra_sender) = counting_worker(counter);
        drop(extra_sender);
        let pool = WorkerPool::single(Arc::clone(&worker));
        pool.reserve_session_placement()
            .unwrap()
            .commit()
            .unwrap()
            .persist();
        let turn = pool.reserve_turn(WorkerId::PRIMARY).unwrap();

        pool.shutdown();

        let status = worker.status();
        assert_eq!(status.health, WorkerHealth::Stopped);
        assert_eq!(status.live_sessions, 0);
        assert_eq!(status.active_turns, 0);
        drop(turn);
    }

    #[test]
    fn pool_shutdown_drops_each_worker_state_on_its_owner_thread() {
        let caller = thread::current().id();
        let (dropped_tx, dropped_rx) = std_mpsc::sync_channel(2);
        let workers = (0..2)
            .map(|index| {
                let id = WorkerId::new(index);
                let (commands, mut rx) = mpsc::channel(1);
                let drop_tx = dropped_tx.clone();
                WorkerHandle::spawn(
                    id,
                    format!("drop-probe-{id}"),
                    commands,
                    move || Ok::<_, Infallible>((DropThread(drop_tx), ())),
                    move |_state| while rx.blocking_recv().is_some() {},
                )
                .expect("start drop-probe worker")
                .0
            })
            .collect();
        drop(dropped_tx);
        let pool = WorkerPool::new(workers);

        pool.shutdown();

        let first = dropped_rx.recv().expect("first worker dropped");
        let second = dropped_rx.recv().expect("second worker dropped");
        assert_ne!(first, caller);
        assert_ne!(second, caller);
        assert_ne!(first, second, "each worker owns a distinct OS thread");
    }

    #[test]
    fn initialization_error_is_returned_after_the_worker_exits() {
        let (commands, _rx) = mpsc::channel(1);
        let error = WorkerHandle::spawn(
            WorkerId::PRIMARY,
            "worker-init-error".to_string(),
            commands,
            || Err::<((), ()), _>("fixture initialization failed"),
            |()| panic!("a failed worker must not enter its run loop"),
        )
        .expect_err("initialization failure must be returned to the caller");

        assert!(matches!(
            &error,
            WorkerStartError::Initialization(message)
                if *message == "fixture initialization failed"
        ));
        assert_eq!(
            error.to_string(),
            "engine worker initialization failed: fixture initialization failed"
        );
    }

    #[test]
    fn initialization_panic_unwinds_on_the_worker_and_is_returned() {
        let caller_thread = thread::current().id();
        let (dropped_tx, dropped_rx) = std_mpsc::sync_channel(1);
        let (commands, _rx) = mpsc::channel(1);
        let error = WorkerHandle::spawn(
            WorkerId::PRIMARY,
            "worker-init-panic".to_string(),
            commands,
            move || -> Result<((), ()), Infallible> {
                let _partial_state = DropThread(dropped_tx);
                panic!("fixture initializer panicked");
            },
            |()| panic!("a panicked worker must not enter its run loop"),
        )
        .expect_err("initialization panic must be returned to the caller");

        assert!(matches!(
            error,
            WorkerStartError::InitializationPanicked(ref message)
                if message == "fixture initializer panicked"
        ));
        assert_ne!(
            dropped_rx
                .recv()
                .expect("partial initialization state must be dropped"),
            caller_thread,
            "partial state must unwind on the worker thread",
        );
    }

    #[test]
    fn invalid_thread_configuration_is_returned_as_startup_failure() {
        let (commands, _rx) = mpsc::channel(1);
        let error = WorkerHandle::spawn(
            WorkerId::PRIMARY,
            "invalid\0worker-name".to_string(),
            commands,
            || Ok::<_, Infallible>(((), ())),
            |()| {},
        )
        .expect_err("a thread name containing NUL cannot be started");

        assert!(
            matches!(&error, WorkerStartError::ThreadSpawn(_)),
            "unexpected startup error: {error}"
        );
    }

    #[test]
    fn closed_initialization_channel_is_not_reported_as_ready() {
        let (ready_tx, ready_rx) =
            std_mpsc::sync_channel::<Result<(), WorkerStartError<Infallible>>>(1);
        drop(ready_tx);
        let join = thread::spawn(|| {});

        let error = await_initialization(ready_rx, join)
            .expect_err("a worker that never handshakes is not ready");

        assert!(matches!(
            error,
            WorkerStartError::InitializationChannelClosed
        ));
    }
}

//! The provider/context-owned deferred release queue (issue #1186 Phase 4).
//!
//! # Why final release cannot happen where the owner is dropped
//!
//! A CUDA device allocation may still be read by kernels already enqueued on
//! the compute stream and by transfers already enqueued on the dedicated copy
//! stream. Freeing it at the moment the last owner goes away is therefore only
//! safe if the caller first *waits* for both streams — which is what the
//! production EP used to do, at the cost of a full stream drain on every free,
//! every weight eviction, and every reservation teardown.
//!
//! This module replaces the wait with an ordering fact. Enqueue records a
//! completion event at the current tail of **both** streams and returns
//! immediately; a worker polls those events without blocking and performs the
//! physical release only once every recorded event has completed. Nothing here
//! ever calls `cuCtxSynchronize`, `cuStreamSynchronize`, or `cuEventSynchronize`.
//!
//! # What the queue owns
//!
//! The queue owns the *exact* ownership it was handed — a
//! [`PreparedAllocationRelease`], a weight page's allocator/allowance, a
//! reservation teardown ticket — plus the CUDA context and streams reached
//! through the fence source. Ownership is released only by a terminal action,
//! and ownership that cannot be released is **retained**, never dropped
//! silently and never refunded.
//!
//! # Lock order
//!
//! The queue holds exactly one lock ([`CudaDeferredReleaseQueue::state`]) and it
//! is a **leaf**:
//!
//! | while holding the queue lock | allowed |
//! |------------------------------|---------|
//! | reading/mutating queue state | yes |
//! | recording a fence | no |
//! | querying a fence (`cuEventQuery`) | no |
//! | executing a release action | no |
//! | calling an allocator | no |
//! | taking a governor / mapped-allowance lock | no |
//! | taking a binding registry or mechanism lock | no |
//! | waiting on anything | no |
//!
//! Poll therefore *drains* the pending queue under the lock, drops the lock,
//! and only then queries fences and executes. Draining is also what makes
//! execution exactly-once under concurrent pollers: an entry is owned by
//! whichever poller removed it.

use std::any::Any;
use std::collections::VecDeque;
use std::fmt::Debug;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::time::{Duration, Instant};

use onnx_runtime_memory_governor::{
    AllocationReleaseOutcome, AllocationReleaseState, DeferredEnqueueError,
    DeferredEnqueueRejection, DeferredReleaseQueue, PreparedAllocationRelease, QuarantineReason,
};

/// Production queue capacity.
///
/// Every entry represents device ownership that already exists. Rejecting one
/// merely because many earlier fences are pending converts temporary stream
/// pressure into permanent quarantine (and can poison a stable weight slot).
/// The provider therefore accepts all representable ownership; tests and custom
/// callers can still construct a deliberately bounded queue to exercise `Full`.
pub const DEFAULT_DEFERRED_RELEASE_CAPACITY: usize = usize::MAX;

/// How long the worker sleeps between polls when nothing is ready.
pub const DEFAULT_POLL_INTERVAL: Duration = Duration::from_micros(250);

/// One recorded completion point that final release must not precede.
///
/// A CUDA implementation wraps one `CUevent` recorded on one stream; the
/// portable tests implement it with a flag.
pub trait ReleaseFence: Send + Sync + Debug {
    /// Non-blocking completion query. Never waits, never synchronizes.
    fn is_complete(&self) -> bool;

    /// Give the fence up **without destroying it**.
    ///
    /// Called when the device or context is known lost: destroying an event
    /// through a lost context is itself a driver call on state that provably no
    /// longer behaves, so the default retains (leaks) the event instead.
    fn retain_after_device_loss(self: Box<Self>) {
        std::mem::forget(self);
    }
}

/// Where a release request's ordering fences come from.
///
/// The production implementation records one event on the compute stream and
/// one on the dedicated copy stream, which is the whole ordering claim this
/// queue makes: a release runs after every kernel and every transfer that could
/// still touch the memory.
pub trait ReleaseFenceSource: Send + Sync + Debug {
    /// Record completion fences at the current tail of every relevant stream.
    ///
    /// Must not block on the device. Returning `Err` refuses the enqueue and
    /// hands the caller's ownership straight back.
    fn record(&self) -> Result<Vec<Box<dyn ReleaseFence>>, String>;
}

/// Ownership an action could not give back, kept alive by the queue.
///
/// The `keep_alive` box is the point: whatever the action still owns (a pool
/// handle, a reservation, an allowance, a prepared request) stays alive and
/// unusable rather than being dropped into a state where something else could
/// hand the same bytes out again.
pub struct RetainedOwnership {
    pub bytes: u64,
    pub detail: String,
    pub keep_alive: Box<dyn Any + Send>,
}

impl Debug for RetainedOwnership {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RetainedOwnership")
            .field("bytes", &self.bytes)
            .field("detail", &self.detail)
            .finish_non_exhaustive()
    }
}

/// The structured result of one deferred release action.
#[derive(Debug)]
pub struct DeferredActionOutcome {
    /// Terminal lifecycle state. Only [`AllocationReleaseState::Released`] is
    /// success-shaped.
    pub state: AllocationReleaseState,
    /// Bytes whose mapped attribution the action actually refunded. Zero is a
    /// valid complete result and never signals failure.
    pub unmapped_bytes: u64,
    pub detail: Option<String>,
    /// Ownership retained instead of released.
    pub retained: Option<RetainedOwnership>,
}

impl DeferredActionOutcome {
    pub fn released(unmapped_bytes: u64) -> Self {
        Self {
            state: AllocationReleaseState::Released,
            unmapped_bytes,
            detail: None,
            retained: None,
        }
    }

    pub fn quarantined(
        state: AllocationReleaseState,
        unmapped_bytes: u64,
        detail: impl Into<String>,
        retained: Option<RetainedOwnership>,
    ) -> Self {
        Self {
            state,
            unmapped_bytes,
            detail: Some(detail.into()),
            retained,
        }
    }

    pub fn is_complete(&self) -> bool {
        self.state == AllocationReleaseState::Released
    }
}

/// One unit of deferred final release.
///
/// `execute` consumes the action, so the ownership it carries is released (or
/// retained) exactly once. It runs with **no** queue lock held.
///
/// Actions are `'static` because the queue may outlive every caller that
/// handed it one, and because device-loss settlement stores what an action
/// could not give back as `Box<dyn Any + Send>`.
pub trait DeferredReleaseAction: Send + Debug + 'static {
    fn execute(self: Box<Self>) -> DeferredActionOutcome;

    /// Settle this action after the device or provider context was lost.
    ///
    /// This is a *terminal* path, not a deferral: it consumes the action
    /// exactly like `execute` does, but it must never call an allocator, never
    /// touch the device, and never refund. It runs with no queue lock held.
    ///
    /// The return value is the ownership the **queue** must keep alive, and it
    /// is deliberately narrower than the action itself. An action that carries
    /// a settlement of its own — a prepared request that owes a mechanism a
    /// terminal state — is expected to perform that settlement here and hand
    /// back only the residual physical ownership. Handing back the whole
    /// unexecuted action instead would leave that settlement permanently owed,
    /// and would keep alive whatever provider context the action pins, which
    /// for a context-owned queue is the queue itself.
    ///
    /// The default retains the whole action, which is correct for an action
    /// whose ownership is purely physical (a page's allocator/allowance, a
    /// reservation ticket) and owes no mechanism anything.
    fn settle_device_lost(self: Box<Self>, detail: &str) -> Option<RetainedOwnership> {
        let bytes = self.bytes();
        Some(RetainedOwnership {
            bytes,
            detail: detail.to_owned(),
            keep_alive: Box::new(self),
        })
    }

    /// Stable short label for stats and shutdown diagnostics.
    fn label(&self) -> &'static str;

    /// Bytes this action is responsible for, for observability only.
    fn bytes(&self) -> u64 {
        0
    }
}

/// Why the queue refused, together with the **exact** ownership handed back.
///
/// Nothing is cloned or reconstructed: the action returned is the action the
/// caller offered, so a refusal cannot lose or free anything.
#[derive(Debug)]
pub struct RefusedRelease<A> {
    pub rejection: DeferredEnqueueRejection,
    pub action: A,
}

impl<A> RefusedRelease<A> {
    fn new(rejection: DeferredEnqueueRejection, action: A) -> Self {
        Self { rejection, action }
    }
}

/// One retained (non-released) record the queue keeps for diagnostics.
#[derive(Debug)]
pub struct RetainedRelease {
    pub label: &'static str,
    pub state: AllocationReleaseState,
    pub bytes: u64,
    pub detail: String,
    /// Whatever the action still owned. Never dropped while the queue lives.
    #[allow(dead_code)]
    ownership: Option<RetainedOwnership>,
}

/// A snapshot of one retained record, without its ownership.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RetainedReleaseInfo {
    pub label: &'static str,
    pub state: AllocationReleaseState,
    pub bytes: u64,
    pub detail: String,
}

/// Structured queue observability, for tests, shutdown reporting, and metrics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeferredReleaseStats {
    /// Accepted requests that have not reached a terminal state yet.
    pub pending: usize,
    /// Requests accepted since construction.
    pub accepted: u64,
    /// Requests whose action reported a complete physical release.
    pub completed: u64,
    /// Requests that ended in retained ownership (quarantine or device loss).
    pub quarantined: u64,
    /// Requests the queue refused.
    pub enqueue_failures: u64,
    /// Mapped bytes refunded from actual structured outcomes.
    pub mapped_refunded_bytes: u64,
    pub closed: bool,
    /// Closing when drained: still accepting, so teardown work the queue's own
    /// releases produce is not refused.
    pub draining: bool,
    pub device_lost: bool,
    /// Retained records currently held.
    pub retained: usize,
}

#[derive(Debug)]
struct PendingRelease {
    fences: Vec<Box<dyn ReleaseFence>>,
    action: Box<dyn DeferredReleaseAction>,
}

#[derive(Debug, Default)]
struct QueueState {
    pending: VecDeque<PendingRelease>,
    retained: Vec<RetainedRelease>,
    closed: bool,
    /// Closing *when drained*: still accepting, because teardown of ownership
    /// the queue itself holds can still produce work.
    draining: bool,
    device_lost: bool,
    worker_started: bool,
}

#[derive(Debug, Default)]
struct ExecutionGateState {
    owner: Option<std::thread::ThreadId>,
    depth: usize,
}

#[derive(Debug, Default)]
struct ExecutionGate {
    state: Mutex<ExecutionGateState>,
    wake: Condvar,
}

impl ExecutionGate {
    fn lock(&self) -> ExecutionGateGuard<'_> {
        let current = std::thread::current().id();
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        while state.owner.as_ref().is_some_and(|owner| *owner != current) {
            state = self
                .wake
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        state.owner = Some(current);
        state.depth = state
            .depth
            .checked_add(1)
            .expect("execution-gate recursion depth overflow");
        ExecutionGateGuard { gate: self }
    }
}

struct ExecutionGateGuard<'a> {
    gate: &'a ExecutionGate,
}

impl Drop for ExecutionGateGuard<'_> {
    fn drop(&mut self) {
        let current = std::thread::current().id();
        let mut state = self
            .gate
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        debug_assert_eq!(state.owner.as_ref(), Some(&current));
        state.depth = state.depth.saturating_sub(1);
        if state.depth == 0 {
            state.owner = None;
            self.gate.wake.notify_all();
        }
    }
}

#[derive(Debug, Default)]
struct Counters {
    accepted: AtomicU64,
    completed: AtomicU64,
    quarantined: AtomicU64,
    enqueue_failures: AtomicU64,
    mapped_refunded_bytes: AtomicU64,
}

/// A bounded, non-blocking, provider/context-owned deferred release queue.
///
/// One queue belongs to one CUDA provider context. It is held by `Arc` because
/// the worker, every enqueued action's pins, and the provider all keep it — and
/// therefore the context and streams — alive until the last request reaches a
/// terminal state.
pub struct CudaDeferredReleaseQueue {
    me: Weak<Self>,
    fences: Box<dyn ReleaseFenceSource>,
    capacity: usize,
    poll_interval: Duration,
    /// Whether a background worker may be spawned. Tests that need determinism
    /// drive [`CudaDeferredReleaseQueue::poll`] themselves instead.
    autonomous: bool,
    /// Serializes the boundary between "device is usable" and starting a
    /// release/query. Device-loss marking takes this gate before publishing the
    /// lost state, so no new driver call can begin after loss is observed.
    execution_gate: ExecutionGate,
    state: Mutex<QueueState>,
    /// Signalled on enqueue, close, and device loss so the worker wakes without
    /// busy-waiting. The worker always waits with a timeout, so a missed
    /// notification can only cost latency, never liveness.
    wake: Condvar,
    /// Accepted-but-not-terminal requests, including ones a poller has drained
    /// and not finished. Capacity is enforced against this, so a request in
    /// flight still occupies its slot.
    outstanding: AtomicUsize,
    device_lost: AtomicBool,
    closed: AtomicBool,
    draining: AtomicBool,
    /// Provider teardown callback. It uses weak manager ownership and is invoked
    /// with every queue/execution lock dropped once normal draining is idle.
    drain_callback: Mutex<Option<Box<dyn FnMut() -> bool + Send>>>,
    counters: Counters,
}

impl Debug for CudaDeferredReleaseQueue {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.stats();
        formatter
            .debug_struct("CudaDeferredReleaseQueue")
            .field("capacity", &self.capacity)
            .field("pending", &stats.pending)
            .field("completed", &stats.completed)
            .field("quarantined", &stats.quarantined)
            .field("closed", &stats.closed)
            .field("device_lost", &stats.device_lost)
            .finish_non_exhaustive()
    }
}

impl CudaDeferredReleaseQueue {
    /// Build a queue that spawns its worker on the first accepted request.
    pub fn new(fences: Box<dyn ReleaseFenceSource>, capacity: usize) -> Arc<Self> {
        Self::build(fences, capacity, DEFAULT_POLL_INTERVAL, true)
    }

    /// Build a queue with no background worker; the caller drives
    /// [`poll`](Self::poll). Used by deterministic tests and by callers that
    /// already have a service thread.
    pub fn manual(fences: Box<dyn ReleaseFenceSource>, capacity: usize) -> Arc<Self> {
        Self::build(fences, capacity, DEFAULT_POLL_INTERVAL, false)
    }

    /// Build a queue with an explicit worker poll interval.
    pub fn with_poll_interval(
        fences: Box<dyn ReleaseFenceSource>,
        capacity: usize,
        poll_interval: Duration,
    ) -> Arc<Self> {
        Self::build(fences, capacity, poll_interval, true)
    }

    fn build(
        fences: Box<dyn ReleaseFenceSource>,
        capacity: usize,
        poll_interval: Duration,
        autonomous: bool,
    ) -> Arc<Self> {
        Arc::new_cyclic(|me| Self {
            me: me.clone(),
            fences,
            capacity: capacity.max(1),
            poll_interval,
            autonomous,
            execution_gate: ExecutionGate::default(),
            state: Mutex::new(QueueState::default()),
            wake: Condvar::new(),
            outstanding: AtomicUsize::new(0),
            device_lost: AtomicBool::new(false),
            closed: AtomicBool::new(false),
            draining: AtomicBool::new(false),
            drain_callback: Mutex::new(None),
            counters: Counters::default(),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, QueueState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Bound on simultaneously outstanding requests.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Accepted requests that have not reached a terminal state.
    pub fn pending(&self) -> usize {
        self.outstanding.load(Ordering::Acquire)
    }

    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire)
    }

    pub fn is_device_lost(&self) -> bool {
        self.device_lost.load(Ordering::Acquire)
    }

    /// Whether the queue is finishing what it holds and will close once nothing
    /// can reach it any more.
    pub fn is_draining(&self) -> bool {
        self.draining.load(Ordering::Acquire)
    }

    pub fn stats(&self) -> DeferredReleaseStats {
        let retained = self.lock().retained.len();
        DeferredReleaseStats {
            pending: self.pending(),
            accepted: self.counters.accepted.load(Ordering::Relaxed),
            completed: self.counters.completed.load(Ordering::Relaxed),
            quarantined: self.counters.quarantined.load(Ordering::Relaxed),
            enqueue_failures: self.counters.enqueue_failures.load(Ordering::Relaxed),
            mapped_refunded_bytes: self.counters.mapped_refunded_bytes.load(Ordering::Relaxed),
            closed: self.is_closed(),
            draining: self.is_draining(),
            device_lost: self.is_device_lost(),
            retained,
        }
    }

    /// Snapshot of retained (never-released) ownership, for shutdown reporting.
    pub fn retained(&self) -> Vec<RetainedReleaseInfo> {
        self.lock()
            .retained
            .iter()
            .map(|record| RetainedReleaseInfo {
                label: record.label,
                state: record.state,
                bytes: record.bytes,
                detail: record.detail.clone(),
            })
            .collect()
    }

    /// Stop accepting new work. Never blocks and never drains.
    ///
    /// Already-accepted requests keep their fences, their ownership, and this
    /// queue (hence the CUDA context) alive until they complete. The worker
    /// exits only once the queue is closed *and* empty.
    pub fn close(&self) {
        self.closed.store(true, Ordering::Release);
        self.lock().closed = true;
        self.wake.notify_all();
    }

    /// Close once everything the queue holds has been settled.
    ///
    /// This is the teardown path, and it is deliberately *not* `close`: an
    /// executing release drops the last reference to an allocator or a VMM
    /// arena, whose own teardown then produces more work for this queue. A hard
    /// close at provider teardown would refuse exactly that work and leak an
    /// address range. So the queue keeps accepting while it is reachable, and
    /// the worker closes it only once it is both empty and unreachable — at
    /// which point nothing can enqueue anything ever again.
    ///
    /// Never blocks and never drains here.
    pub fn close_after_drain(&self) {
        self.draining.store(true, Ordering::Release);
        self.lock().draining = true;
        self.run_drain_callback_if_ready();
        self.wake.notify_all();
    }

    /// Install one callback that removes provider registrations after all
    /// normally ordered releases settle.
    ///
    /// Returning `false` asks the worker to retry on a later idle poll (for
    /// example while a live allocation still pins the mechanism). Device-loss
    /// teardown is excluded; it requires explicit context-termination
    /// confirmation instead of normal removal.
    pub fn set_drain_callback(&self, callback: impl FnMut() -> bool + Send + 'static) {
        *self
            .drain_callback
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(Box::new(callback));
        self.run_drain_callback_if_ready();
        self.wake.notify_all();
    }

    fn run_drain_callback_if_ready(&self) {
        if !self.is_draining() || self.is_device_lost() || self.pending() != 0 {
            return;
        }
        let callback = self
            .drain_callback
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take();
        let Some(mut callback) = callback else {
            return;
        };
        if !callback() {
            let mut slot = self
                .drain_callback
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if slot.is_none() {
                *slot = Some(callback);
            }
        }
    }

    /// Record that the device or provider context was lost.
    ///
    /// From this point the queue never queries a fence, never calls an
    /// allocator, and never refunds. Every pending request is *settled* rather
    /// than abandoned: its action records its own terminal device-loss state —
    /// for a binding-prepared release that is device-lost quarantine at the
    /// exact allocation identity — and hands back only the ownership that must
    /// stay alive, which the queue then holds. Its events are kept rather than
    /// destroyed through a context that is known bad.
    ///
    /// Settling rather than freezing the whole action is what lets the
    /// mechanism reach quiescence, so the provider context can confirm
    /// termination and discharge the quarantine it was handed.
    pub fn mark_device_lost(&self, reason: impl Into<String>) {
        let reason = reason.into();
        let gate = self.execution_gate.lock();
        if self.device_lost.swap(true, Ordering::AcqRel) {
            return;
        }
        drop(gate);
        {
            let mut state = self.lock();
            state.device_lost = true;
        }
        self.wake.notify_all();
        self.retain_all_pending(&format!("device lost: {reason}"));
        // Device loss is terminal for this queue. Retained records may keep the
        // queue allocation alive for diagnostics, but no worker thread is needed
        // once every pending action has been moved to owned quarantine.
        self.closed.store(true, Ordering::Release);
        self.lock().closed = true;
        self.wake.notify_all();
    }

    /// Execute every request whose fences have all completed.
    ///
    /// Returns the number of actions executed. Non-blocking: fences are only
    /// *queried*. Safe to call from several threads at once — each entry is
    /// drained under the lock, so exactly one poller ever executes it.
    pub fn poll(&self) -> usize {
        if self.is_device_lost() {
            self.retain_all_pending("device lost before the release could be ordered");
            return 0;
        }
        let drained: Vec<PendingRelease> = {
            let mut state = self.lock();
            state.pending.drain(..).collect()
        };
        if drained.is_empty() {
            self.run_drain_callback_if_ready();
            return 0;
        }
        let mut carry = VecDeque::with_capacity(drained.len());
        let mut executed = 0usize;
        let mut retained = Vec::new();
        for entry in drained {
            let execution = self.execution_gate.lock();
            if self.is_device_lost() {
                drop(execution);
                retained.push(self.settle_lost_entry(
                    entry,
                    "device lost while a poller owned the deferred release",
                ));
                continue;
            }
            if !entry.fences.iter().all(|fence| fence.is_complete()) {
                drop(execution);
                carry.push_back(entry);
                continue;
            }
            let PendingRelease { fences, action } = entry;
            let label = action.label();
            // Fences are dropped (destroying their events) before the action
            // runs: the ordering claim has already been observed, and holding
            // them across the release would only delay reclaiming them.
            drop(fences);
            let outcome = action.execute();
            drop(execution);
            executed += 1;
            self.record_outcome(label, outcome, &mut retained);
            self.outstanding.fetch_sub(1, Ordering::AcqRel);
        }
        let carry_gate = self.execution_gate.lock();
        if self.is_device_lost() {
            while let Some(entry) = carry.pop_front() {
                retained.push(self.settle_lost_entry(
                    entry,
                    "device lost after a poller observed an incomplete release fence",
                ));
            }
        }
        if !carry.is_empty() || !retained.is_empty() {
            let mut state = self.lock();
            state.retained.append(&mut retained);
            // Preserve FIFO: unfinished entries go back in front of anything a
            // concurrent enqueue added while the lock was released.
            while let Some(entry) = carry.pop_back() {
                state.pending.push_front(entry);
            }
        }
        drop(carry_gate);
        if executed > 0 {
            self.wake.notify_all();
        }
        self.run_drain_callback_if_ready();
        executed
    }

    fn record_outcome(
        &self,
        label: &'static str,
        outcome: DeferredActionOutcome,
        retained: &mut Vec<RetainedRelease>,
    ) {
        if outcome.unmapped_bytes > 0 {
            self.counters
                .mapped_refunded_bytes
                .fetch_add(outcome.unmapped_bytes, Ordering::Relaxed);
        }
        if outcome.is_complete() {
            self.counters.completed.fetch_add(1, Ordering::Relaxed);
            return;
        }
        self.counters.quarantined.fetch_add(1, Ordering::Relaxed);
        let detail = outcome
            .detail
            .unwrap_or_else(|| String::from("the release did not complete"));
        let bytes = outcome
            .retained
            .as_ref()
            .map_or(0, |ownership| ownership.bytes);
        eprintln!(
            "cuda_ep: WARNING: deferred {label} release did not complete ({}): {detail}; \
             {bytes} byte(s) of ownership are retained and will not be reused",
            outcome.state
        );
        retained.push(RetainedRelease {
            label,
            state: outcome.state,
            bytes,
            detail,
            ownership: outcome.retained,
        });
    }

    /// Settle one pending entry that can never reach the device again.
    ///
    /// The fences are retained rather than destroyed, and the action settles
    /// its own device-loss state and hands back only the ownership the queue
    /// must keep. The queue deliberately does **not** keep the unexecuted
    /// action: an action that owes a mechanism a terminal state would owe it
    /// forever, its binding/provider-context pins would never be released, and
    /// for a context-owned queue those pins reach back to this queue, so the
    /// retained record would keep itself alive.
    ///
    /// Callers must hold neither the queue lock nor the execution gate.
    fn settle_lost_entry(&self, entry: PendingRelease, detail: &str) -> RetainedRelease {
        let PendingRelease { fences, action } = entry;
        for fence in fences {
            // The context is gone; destroying the event would be a driver call
            // on state that no longer exists.
            fence.retain_after_device_loss();
        }
        let label = action.label();
        let bytes = action.bytes();
        let ownership = action.settle_device_lost(detail);
        self.counters.quarantined.fetch_add(1, Ordering::Relaxed);
        self.outstanding.fetch_sub(1, Ordering::AcqRel);
        RetainedRelease {
            label,
            state: AllocationReleaseState::DeviceLost,
            bytes,
            detail: detail.to_owned(),
            ownership,
        }
    }

    /// Settle every pending request as device-lost without touching the device.
    fn retain_all_pending(&self, detail: &str) {
        let drained: Vec<PendingRelease> = {
            let mut state = self.lock();
            state.pending.drain(..).collect()
        };
        if drained.is_empty() {
            return;
        }
        let mut records = Vec::with_capacity(drained.len());
        for entry in drained {
            records.push(self.settle_lost_entry(entry, detail));
        }
        let mut state = self.lock();
        state.retained.append(&mut records);
    }

    /// Take final ownership of `action`, ordered after the current tail of every
    /// stream the fence source records.
    ///
    /// Returns immediately: no stream or device synchronization happens here,
    /// and no unbounded wait is possible. On refusal the caller gets the exact
    /// action back inside [`RefusedRelease`].
    pub fn enqueue<A: DeferredReleaseAction + 'static>(
        &self,
        action: A,
    ) -> Result<(), RefusedRelease<A>> {
        if let Some(rejection) = self.refusal_reason() {
            self.counters
                .enqueue_failures
                .fetch_add(1, Ordering::Relaxed);
            return Err(RefusedRelease::new(rejection, action));
        }
        // Reserve the slot before recording fences so two racing enqueues cannot
        // both pass a capacity check.
        if let Err(rejection) = self.reserve_slot() {
            self.counters
                .enqueue_failures
                .fetch_add(1, Ordering::Relaxed);
            return Err(RefusedRelease::new(rejection, action));
        }
        let execution = self.execution_gate.lock();
        if let Some(rejection) = self.refusal_reason() {
            drop(execution);
            self.outstanding.fetch_sub(1, Ordering::AcqRel);
            self.counters
                .enqueue_failures
                .fetch_add(1, Ordering::Relaxed);
            return Err(RefusedRelease::new(rejection, action));
        }
        let fences = match self.fences.record() {
            Ok(fences) => fences,
            Err(error) => {
                drop(execution);
                self.outstanding.fetch_sub(1, Ordering::AcqRel);
                self.counters
                    .enqueue_failures
                    .fetch_add(1, Ordering::Relaxed);
                eprintln!(
                    "cuda_ep: WARNING: could not record deferred-release ordering fences for a \
                     {} release: {error}",
                    action.label()
                );
                return Err(RefusedRelease::new(
                    DeferredEnqueueRejection::Refused,
                    action,
                ));
            }
        };
        {
            let mut state = self.lock();
            // Re-check under the lock: close and device loss may have landed
            // while the fences were being recorded.
            if state.closed || state.device_lost {
                let rejection = if state.device_lost {
                    DeferredEnqueueRejection::DeviceLost
                } else {
                    DeferredEnqueueRejection::Closed
                };
                drop(state);
                self.outstanding.fetch_sub(1, Ordering::AcqRel);
                self.counters
                    .enqueue_failures
                    .fetch_add(1, Ordering::Relaxed);
                for fence in fences {
                    if rejection == DeferredEnqueueRejection::DeviceLost {
                        fence.retain_after_device_loss();
                    }
                }
                // The action is handed back untouched; it was never boxed.
                return Err(RefusedRelease::new(rejection, action));
            }
            state.pending.push_back(PendingRelease {
                fences,
                action: Box::new(action),
            });
        }
        drop(execution);
        self.counters.accepted.fetch_add(1, Ordering::Relaxed);
        self.wake.notify_all();
        self.ensure_worker();
        Ok(())
    }

    /// Take final ownership of a prepared release, refunding through `observer`
    /// when the release actually completes.
    pub fn enqueue_prepared(
        &self,
        request: PreparedAllocationRelease,
        observer: Option<Arc<dyn ReleaseObserver>>,
    ) -> Result<(), DeferredEnqueueError> {
        let action = PreparedReleaseAction {
            request: Some(request),
            observer,
        };
        match self.enqueue(action) {
            Ok(()) => Ok(()),
            Err(refused) => {
                let request = refused
                    .action
                    .into_request()
                    .expect("a refused prepared release still holds its request");
                Err(DeferredEnqueueError::new(refused.rejection, request))
            }
        }
    }

    fn refusal_reason(&self) -> Option<DeferredEnqueueRejection> {
        if self.is_device_lost() {
            return Some(DeferredEnqueueRejection::DeviceLost);
        }
        if self.is_closed() {
            return Some(DeferredEnqueueRejection::Closed);
        }
        None
    }

    fn reserve_slot(&self) -> Result<(), DeferredEnqueueRejection> {
        let mut current = self.outstanding.load(Ordering::Acquire);
        loop {
            if current >= self.capacity {
                return Err(DeferredEnqueueRejection::Full);
            }
            match self.outstanding.compare_exchange_weak(
                current,
                current + 1,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return Ok(()),
                Err(observed) => current = observed,
            }
        }
    }

    fn ensure_worker(&self) {
        if !self.autonomous {
            return;
        }
        {
            let mut state = self.lock();
            if state.worker_started {
                return;
            }
            state.worker_started = true;
        }
        let Some(queue) = self.me.upgrade() else {
            return;
        };
        // The worker holds an `Arc` to this queue, so the queue, its fence
        // source, and the CUDA context/streams behind it stay alive until every
        // accepted request is terminal — even if the provider is dropped first.
        let spawned = std::thread::Builder::new()
            .name("cuda-deferred-release".into())
            .spawn(move || queue.run_worker());
        if let Err(error) = spawned {
            self.lock().worker_started = false;
            eprintln!(
                "cuda_ep: WARNING: could not start the deferred-release worker ({error}); \
                 releases will be drained by later provider calls instead"
            );
        }
    }

    fn run_worker(self: Arc<Self>) {
        loop {
            if !self.is_device_lost() {
                self.poll();
            }
            let mut state = self.lock();
            let mut idle =
                state.pending.is_empty() && self.outstanding.load(Ordering::Acquire) == 0;
            if idle && state.closed {
                // Closed and empty: nothing can arrive and nothing is owed.
                return;
            }
            if idle && state.draining && !state.device_lost {
                drop(state);
                self.run_drain_callback_if_ready();
                state = self.lock();
                idle = state.pending.is_empty() && self.outstanding.load(Ordering::Acquire) == 0;
            }
            // Draining: the queue is empty *and* this worker holds the only
            // reference to it, so no provider, allocator, arena, or reservation
            // can still reach it. Only then is closing safe — the late work a
            // hard close would have refused is precisely the teardown that
            // reaching the queue produces.
            if idle && state.draining && Arc::strong_count(&self) == 1 {
                drop(state);
                self.closed.store(true, Ordering::Release);
                self.lock().closed = true;
                return;
            }
            // A lost device is never polled, so the worker only wakes to notice
            // a close. It stays alive while anything is outstanding, which is
            // what keeps the context pin and retained ownership from being
            // released early.
            let interval = if state.device_lost {
                self.poll_interval.max(Duration::from_millis(50))
            } else {
                self.poll_interval
            };
            let (guard, _) = self
                .wake
                .wait_timeout(state, interval)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            drop(guard);
        }
    }

    /// Explicitly wait — with a deadline — until nothing is outstanding.
    ///
    /// This is a diagnostic/test entry point, never used on a release path and
    /// never called from `Drop`. It polls; it does not synchronize a stream.
    pub fn wait_until_idle(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            self.poll();
            if self.pending() == 0 {
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(self.poll_interval.min(Duration::from_millis(1)));
        }
    }
}

/// The queue is the memory-api deferred sink for this provider context.
impl DeferredReleaseQueue for CudaDeferredReleaseQueue {
    fn enqueue(&self, request: PreparedAllocationRelease) -> Result<(), DeferredEnqueueError> {
        self.enqueue_prepared(request, None)
    }

    fn pending(&self) -> usize {
        self.outstanding.load(Ordering::Acquire)
    }
}

impl onnx_runtime_memory_governor::DeviceLossListener for CudaDeferredReleaseQueue {
    fn mark_device_lost(&self, reason: &str) {
        CudaDeferredReleaseQueue::mark_device_lost(self, reason);
    }
}

/// Notified when a deferred allocation release actually completes.
///
/// This is where provider-side accounting happens — mapped refunds and free
/// counters — so it is driven by the *actual* structured outcome instead of an
/// assumption made at enqueue time.
pub trait ReleaseObserver: Send + Sync + Debug {
    fn released(&self, outcome: &AllocationReleaseOutcome);
}

/// A binding-prepared allocation release, deferred behind stream fences.
#[derive(Debug)]
pub struct PreparedReleaseAction {
    /// `None` only after `execute` consumed it.
    request: Option<PreparedAllocationRelease>,
    observer: Option<Arc<dyn ReleaseObserver>>,
}

impl PreparedReleaseAction {
    pub fn new(
        request: PreparedAllocationRelease,
        observer: Option<Arc<dyn ReleaseObserver>>,
    ) -> Self {
        Self {
            request: Some(request),
            observer,
        }
    }

    fn into_request(mut self) -> Option<PreparedAllocationRelease> {
        self.request.take()
    }
}

impl DeferredReleaseAction for PreparedReleaseAction {
    fn execute(mut self: Box<Self>) -> DeferredActionOutcome {
        let Some(request) = self.request.take() else {
            return DeferredActionOutcome::quarantined(
                AllocationReleaseState::Quarantined,
                0,
                "a prepared release action was executed without its request",
                None,
            );
        };
        let allocation_bytes = request.len() as u64;
        let allocator = Arc::clone(request.allocator());
        let outcome = request.execute();
        if let Some(observer) = self.observer.as_ref() {
            observer.released(&outcome);
        }
        match outcome {
            AllocationReleaseOutcome::Complete { accounting } => {
                DeferredActionOutcome::released(accounting.unmapped_bytes)
            }
            AllocationReleaseOutcome::Quarantined {
                accounting,
                residual,
            } => DeferredActionOutcome::quarantined(
                residual.state,
                accounting.unmapped_bytes,
                format!(
                    "{} ({} byte(s) retained at {:#x})",
                    residual.reason, residual.retained_bytes, residual.address
                ),
                Some(RetainedOwnership {
                    bytes: residual.retained_bytes,
                    detail: String::from("binding-prepared provider allocation"),
                    keep_alive: Box::new(allocator),
                }),
            ),
            // `PreparedAllocationRelease::execute` never returns `Failed`; treat
            // it as retained ownership rather than as success.
            AllocationReleaseOutcome::Failed { failure } => DeferredActionOutcome::quarantined(
                AllocationReleaseState::Quarantined,
                0,
                failure.to_string(),
                Some(RetainedOwnership {
                    bytes: allocation_bytes,
                    detail: String::from("binding-prepared provider allocation"),
                    keep_alive: Box::new(allocator),
                }),
            ),
        }
    }

    /// Settle the prepared request as device-lost quarantine.
    ///
    /// No allocator is called and nothing is refunded: the binding records the
    /// exact allocation identity as device-lost, the mechanism's queued-release
    /// count settles, and the active-operation pin is released, so the
    /// mechanism can reach quiescence and the provider context can confirm
    /// termination and discharge the quarantine.
    ///
    /// Only the pinned allocator is handed back — the same residual the normal
    /// quarantine path retains. It is what keeps the physical ownership from
    /// being handed out again, and unlike the request it does not pin the
    /// provider context, so retaining it cannot keep this queue alive.
    fn settle_device_lost(mut self: Box<Self>, detail: &str) -> Option<RetainedOwnership> {
        let request = self.request.take()?;
        let bytes = request.len() as u64;
        let allocator = Arc::clone(request.allocator());
        let outcome = request.quarantine_device_lost();
        // The observer sees the real terminal outcome. It refunds only what the
        // allocator reported unmapped, which is zero here, and counts no free.
        if let Some(observer) = self.observer.as_ref() {
            observer.released(&outcome);
        }
        Some(RetainedOwnership {
            bytes,
            detail: detail.to_owned(),
            keep_alive: Box::new(allocator),
        })
    }

    fn label(&self) -> &'static str {
        "provider allocation"
    }

    fn bytes(&self) -> u64 {
        self.request
            .as_ref()
            .map_or(0, |request| request.len() as u64)
    }
}

impl Drop for PreparedReleaseAction {
    /// An action that is dropped without executing quarantines its request at
    /// the mechanism. It never frees and never waits.
    fn drop(&mut self) {
        if let Some(request) = self.request.take() {
            let outcome = request.quarantine(QuarantineReason::AbandonedRequest);
            if let Some(observer) = self.observer.as_ref() {
                observer.released(&outcome);
            }
        }
    }
}

/// Ordering fences recorded on this provider's CUDA streams.
///
/// One event is recorded at the tail of the compute stream and one at the tail
/// of the dedicated copy stream, so a release is ordered after *both* the
/// kernels and the transfers that could still be reading the memory. Recording
/// an event is asynchronous: this never waits for either stream.
#[derive(Debug)]
pub struct CudaStreamFences {
    runtime: Arc<crate::runtime::CudaRuntime>,
}

impl CudaStreamFences {
    pub fn new(runtime: Arc<crate::runtime::CudaRuntime>) -> Self {
        Self { runtime }
    }
}

impl ReleaseFenceSource for CudaStreamFences {
    fn record(&self) -> Result<Vec<Box<dyn ReleaseFence>>, String> {
        self.runtime
            .bind()
            .map_err(|error| format!("could not bind the CUDA context: {error}"))?;
        let context = self.runtime.cuda_context();
        let mut fences: Vec<Box<dyn ReleaseFence>> = Vec::with_capacity(2);
        for (stream, name) in [
            (self.runtime.stream(), "compute"),
            (self.runtime.copy_stream(), "copy"),
        ] {
            let event = context
                .new_event(None)
                .map_err(|error| format!("cuEventCreate for the {name} stream failed: {error}"))?;
            event
                .record(stream)
                .map_err(|error| format!("cuEventRecord on the {name} stream failed: {error}"))?;
            fences.push(Box::new(CudaEventFence { event }));
        }
        Ok(fences)
    }
}

/// One recorded CUDA completion event.
///
/// `is_complete` is `cuEventQuery`, which never blocks. The event is destroyed
/// when the fence is dropped, except after device loss, where the fence is
/// retained instead so no driver call is made through a dead context.
#[derive(Debug)]
struct CudaEventFence {
    event: cudarc::driver::CudaEvent,
}

impl ReleaseFence for CudaEventFence {
    fn is_complete(&self) -> bool {
        self.event.is_complete()
    }
}

/// A CUDA address-space reservation teardown, deferred behind stream fences.
///
/// This is what makes `CudaReservation::Drop` non-blocking: the reservation
/// hands over its exact VA, mapped blocks, quarantined blocks, pool, and
/// context, and the unmap/release/`cuMemAddressFree` sequence runs here, after
/// both stream tails.
#[derive(Debug)]
pub struct ReservationTeardownAction {
    /// `None` only after `execute` consumed it.
    ticket: Option<crate::virtual_memory::ReservationTeardownTicket>,
}

impl DeferredReleaseAction for ReservationTeardownAction {
    fn execute(mut self: Box<Self>) -> DeferredActionOutcome {
        let Some(ticket) = self.ticket.take() else {
            return DeferredActionOutcome::quarantined(
                AllocationReleaseState::Quarantined,
                0,
                "a reservation teardown action was executed without its ticket",
                None,
            );
        };
        let bytes = ticket.len() as u64;
        let outcome = ticket.execute_outcome();
        let report = outcome.report;
        if report.is_complete() {
            return DeferredActionOutcome::released(report.unmapped_bytes);
        }
        DeferredActionOutcome::quarantined(
            AllocationReleaseState::PartiallyUnmapped,
            report.unmapped_bytes,
            format!(
                "{} reservation block(s) could not be released, so the {bytes} byte address \
                 range was not returned to the driver",
                report.retained_blocks
            ),
            outcome.retained.map(|ticket| RetainedOwnership {
                bytes,
                detail: String::from("CUDA reservation teardown"),
                keep_alive: Box::new(ticket),
            }),
        )
    }

    fn label(&self) -> &'static str {
        "reservation teardown"
    }

    fn bytes(&self) -> u64 {
        self.ticket.as_ref().map_or(0, |ticket| ticket.len() as u64)
    }
}

/// The queue is also the sink for VMM reservation teardown, so a shared-prefix
/// owner's or arena's `Drop` is event-fenced instead of stream-synchronized.
impl crate::virtual_memory::DeferredReservationQueue for CudaDeferredReleaseQueue {
    fn enqueue_reservation(
        &self,
        ticket: crate::virtual_memory::ReservationTeardownTicket,
    ) -> Result<(), crate::virtual_memory::ReservationEnqueueError> {
        match self.enqueue(ReservationTeardownAction {
            ticket: Some(ticket),
        }) {
            Ok(()) => Ok(()),
            Err(refused) => {
                let rejection = match refused.rejection {
                    DeferredEnqueueRejection::Closed => {
                        crate::virtual_memory::ReservationEnqueueRejection::Closed
                    }
                    DeferredEnqueueRejection::Full => {
                        crate::virtual_memory::ReservationEnqueueRejection::Full
                    }
                    DeferredEnqueueRejection::DeviceLost => {
                        crate::virtual_memory::ReservationEnqueueRejection::DeviceLost
                    }
                    DeferredEnqueueRejection::Refused => {
                        crate::virtual_memory::ReservationEnqueueRejection::Refused
                    }
                };
                Err(crate::virtual_memory::ReservationEnqueueError {
                    rejection,
                    ticket: refused
                        .action
                        .ticket
                        .expect("a refused reservation teardown still holds its ticket"),
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[derive(Debug, Default)]
    struct FakeFence {
        complete: Arc<AtomicBool>,
        destroyed: Option<Arc<AtomicUsize>>,
    }

    impl ReleaseFence for FakeFence {
        fn is_complete(&self) -> bool {
            self.complete.load(Ordering::Acquire)
        }
    }

    impl Drop for FakeFence {
        fn drop(&mut self) {
            if let Some(counter) = &self.destroyed {
                counter.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    #[derive(Debug)]
    struct FakeFenceSource {
        compute: Arc<AtomicBool>,
        copy: Arc<AtomicBool>,
        destroyed: Arc<AtomicUsize>,
    }

    impl ReleaseFenceSource for FakeFenceSource {
        fn record(&self) -> Result<Vec<Box<dyn ReleaseFence>>, String> {
            Ok(vec![
                Box::new(FakeFence {
                    complete: Arc::clone(&self.compute),
                    destroyed: Some(Arc::clone(&self.destroyed)),
                }),
                Box::new(FakeFence {
                    complete: Arc::clone(&self.copy),
                    destroyed: Some(Arc::clone(&self.destroyed)),
                }),
            ])
        }
    }

    #[derive(Debug)]
    struct CountingAction {
        executed: Arc<AtomicUsize>,
    }

    impl DeferredReleaseAction for CountingAction {
        fn execute(self: Box<Self>) -> DeferredActionOutcome {
            self.executed.fetch_add(1, Ordering::AcqRel);
            DeferredActionOutcome::released(0)
        }

        fn label(&self) -> &'static str {
            "test"
        }
    }

    fn manual_queue(
        capacity: usize,
    ) -> (
        Arc<CudaDeferredReleaseQueue>,
        Arc<AtomicBool>,
        Arc<AtomicBool>,
    ) {
        let compute = Arc::new(AtomicBool::new(false));
        let copy = Arc::new(AtomicBool::new(false));
        let queue = CudaDeferredReleaseQueue::manual(
            Box::new(FakeFenceSource {
                compute: Arc::clone(&compute),
                copy: Arc::clone(&copy),
                destroyed: Arc::new(AtomicUsize::new(0)),
            }),
            capacity,
        );
        (queue, compute, copy)
    }

    #[test]
    fn drain_callback_runs_outside_queue_locks_and_retries_until_cleanup_succeeds() {
        let (queue, _, _) = manual_queue(1);
        let calls = Arc::new(AtomicUsize::new(0));
        let callback_calls = Arc::clone(&calls);
        let reentrant = Arc::clone(&queue);
        queue.set_drain_callback(move || {
            let call = callback_calls.fetch_add(1, Ordering::AcqRel);
            // Reentrant observability would deadlock if a queue lock were held.
            let _ = reentrant.stats();
            call != 0
        });
        queue.close_after_drain();
        assert_eq!(calls.load(Ordering::Acquire), 1);
        queue.poll();
        assert_eq!(calls.load(Ordering::Acquire), 2);
        queue.poll();
        assert_eq!(calls.load(Ordering::Acquire), 2, "callback settles once");
    }

    #[test]
    fn both_stream_fences_must_complete_before_release() {
        let (queue, compute, copy) = manual_queue(8);
        let executed = Arc::new(AtomicUsize::new(0));
        queue
            .enqueue(CountingAction {
                executed: Arc::clone(&executed),
            })
            .expect("accepted");
        assert_eq!(queue.poll(), 0, "neither stream has completed");
        compute.store(true, Ordering::Release);
        assert_eq!(queue.poll(), 0, "the copy stream is still in flight");
        copy.store(true, Ordering::Release);
        assert_eq!(queue.poll(), 1);
        assert_eq!(executed.load(Ordering::Acquire), 1);
        assert_eq!(queue.pending(), 0);
    }

    #[test]
    fn a_bounded_queue_refuses_and_returns_the_exact_action() {
        let (queue, _compute, _copy) = manual_queue(1);
        let executed = Arc::new(AtomicUsize::new(0));
        queue
            .enqueue(CountingAction {
                executed: Arc::clone(&executed),
            })
            .expect("first accepted");
        let refused = queue
            .enqueue(CountingAction {
                executed: Arc::clone(&executed),
            })
            .expect_err("the bound is enforced");
        assert_eq!(refused.rejection, DeferredEnqueueRejection::Full);
        assert_eq!(queue.stats().enqueue_failures, 1);
        // The refused action is handed back intact and never executed.
        assert_eq!(executed.load(Ordering::Acquire), 0);
        drop(refused);
    }
}

//! The routing-layer exclusive turn lease.
//!
//! A session is one conversation, and a turn reads that conversation before it
//! writes the turn back into it. Two turns overlapping on one session is
//! therefore not a throughput question but a correctness one: whichever turn
//! writes second replaces a conversation the other had already read, and
//! nothing reports that the first turn's prompt and generation went nowhere.
//!
//! The refusal that prevents it has to be taken *here*, in the routing layer,
//! rather than inside the worker loop. A lease taken on the worker thread is
//! taken after the command is already queued, so a second turn on a busy
//! session would be admitted, parked behind the first, and eventually succeed —
//! a slow success rather than the refusal
//! [`PackageCapabilityError::ExclusiveLeaseConflict`] names. Acquiring before
//! the command exists is what makes that refusal reachable with one worker.
//!
//! What this module is *not*: it does not run two turns, does not shard an
//! engine, and does not make anything concurrent. It decides which turns are
//! allowed to start, and refuses the rest by name. See
//! `docs/architecture/SESSION_CONCURRENCY.md` §4.2 and §4.2.1.

use std::{
    collections::HashSet,
    fmt,
    hash::{DefaultHasher, Hash, Hasher},
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

use onnx_genai_engine::PackageCapabilityError;

use crate::worker::SessionPlacement;

/// One lock per shard is plenty for a lease map whose critical section is a
/// single set insert or removal; the count is fixed because the map spans every
/// loaded model rather than one model's worker pool.
const DEFAULT_SHARDS: usize = 8;

/// Which model owns an engine, as a lease-key component and as the id the
/// registry resolves that engine back from.
///
/// A newtype rather than a bare `String` so a model id cannot be passed where a
/// client session id is expected, and `Arc<str>` so it is cheap to carry inside
/// every lease key and every registry entry.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct ModelKey(Arc<str>);

impl ModelKey {
    pub(crate) fn new(id: &str) -> Self {
        Self(Arc::from(id))
    }

    /// The registry id this key names, for resolving the engine that owns a
    /// leased conversation.
    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for ModelKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.0, formatter)
    }
}

impl fmt::Display for ModelKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// The globally unique identity of one conversation: which model's engine owns
/// it, and where inside that engine it lives.
///
/// [`SessionPlacement`] alone is *not* unique. It names a worker and an engine
/// session id, and both are per-engine: every model's pool starts at worker 0
/// and every model's engine hands out session ids from its own counter, so two
/// loaded models routinely produce the identical placement for two entirely
/// different conversations. A lease map spanning models — and the session
/// registry is one map spanning models — must therefore be keyed by the model
/// as well, or a turn on one model's session would refuse, evict, or close
/// another model's.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct ModelSessionPlacement {
    model: ModelKey,
    placement: SessionPlacement,
}

impl ModelSessionPlacement {
    pub(crate) fn new(model: ModelKey, placement: SessionPlacement) -> Self {
        Self { model, placement }
    }

    /// The model whose engine owns this conversation.
    pub(crate) fn model(&self) -> &ModelKey {
        &self.model
    }

    /// Where the conversation lives inside that model's engine.
    pub(crate) fn placement(&self) -> SessionPlacement {
        self.placement
    }
}

impl fmt::Debug for ModelSessionPlacement {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ModelSessionPlacement")
            .field("model", &self.model)
            .field("placement", &self.placement)
            .finish()
    }
}

/// Which sessions currently have a turn in flight, across every loaded model.
///
/// One map, owned by the session registry, because the session registry is
/// itself one map across models: a decision the registry makes about a binding
/// — evict it, close it, refuse a turn on it — has to consult the same lease
/// the turn on that binding took. Per-engine maps would have made "the lease
/// for this session" depend on which engine the caller happened to be holding,
/// which is exactly the question a routing conflict cannot afford to get wrong.
///
/// Sharded so the critical section stays short and unrelated sessions' request
/// tasks rarely contend for one lock.
pub(crate) struct SessionLeases {
    shards: Vec<Mutex<HashSet<ModelSessionPlacement>>>,
}

impl SessionLeases {
    /// The lease map for one server: every model loaded into one registry.
    pub(crate) fn new() -> Arc<Self> {
        Self::with_shards(DEFAULT_SHARDS)
    }

    /// A lease map with an explicit shard count, for tests that want every key
    /// to land in one shard.
    pub(crate) fn with_shards(shards: usize) -> Arc<Self> {
        Arc::new(Self {
            shards: (0..shards.max(1))
                .map(|_| Mutex::new(HashSet::new()))
                .collect(),
        })
    }

    /// The shard a binding's lease lives in.
    ///
    /// Hashed over the whole key rather than indexed by worker: the map spans
    /// models, so the worker index is neither unique nor well distributed
    /// across it.
    fn shard(&self, binding: &ModelSessionPlacement) -> &Mutex<HashSet<ModelSessionPlacement>> {
        let mut hasher = DefaultHasher::new();
        binding.hash(&mut hasher);
        let index = (hasher.finish() % self.shards.len() as u64) as usize;
        &self.shards[index]
    }

    /// Take the exclusive turn lease for a session, or refuse by name.
    ///
    /// `session` is the identity the caller used — the client's session id —
    /// and appears in the refusal so the answer names something the caller can
    /// act on. It is not the key: the key is the model-qualified placement.
    pub(crate) fn acquire(
        self: &Arc<Self>,
        binding: ModelSessionPlacement,
        session: &str,
    ) -> Result<SessionLeaseGuard, PackageCapabilityError> {
        if lock(self.shard(&binding)).insert(binding.clone()) {
            Ok(SessionLeaseGuard {
                leases: Arc::clone(self),
                binding,
            })
        } else {
            Err(PackageCapabilityError::ExclusiveLeaseConflict {
                session: session.to_string(),
            })
        }
    }

    /// Whether a turn is in flight for this session.
    ///
    /// Observation only, and deliberately not reachable from production code: a
    /// decision made from this answer would be made against a fact that can
    /// change before it is acted on. Everything that must *not* disturb a live
    /// conversation — LRU eviction above all — takes the lease instead, because
    /// taking it is the only check that cannot go stale. Tests assert with it
    /// that a lease is held while a turn runs and released when it ends.
    #[cfg(test)]
    pub(crate) fn is_held(&self, binding: &ModelSessionPlacement) -> bool {
        lock(self.shard(binding)).contains(binding)
    }

    fn release(&self, binding: &ModelSessionPlacement) {
        lock(self.shard(binding)).remove(binding);
    }

    /// How many sessions hold a lease right now. Tests assert nothing leaked.
    #[cfg(test)]
    pub(crate) fn held(&self) -> usize {
        self.shards.iter().map(|shard| lock(shard).len()).sum()
    }
}

impl fmt::Debug for SessionLeases {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SessionLeases")
            .field("shards", &self.shards.len())
            .finish()
    }
}

/// A held exclusive turn lease, released when this guard is dropped.
///
/// Release is a `Drop` obligation and never a cleanup call, because a turn has
/// more ways to end than anyone remembers to enumerate: it completes, it fails
/// in the engine, its client disconnects, its command is never accepted because
/// the driver stopped, or it unwinds. Every one of those drops the guard, and
/// none of them has to know it exists.
///
/// The guard therefore travels *with* the work: into the
/// [`DriverCommand`](crate::driver::DriverCommand) that carries the turn, so a
/// failed send returns it to the sending task and drops it there, and into the
/// worker's per-row state once the command is accepted.
///
/// It also carries the identity of what it leased, model included, so a caller
/// holding one never has to ask a second source which engine to act on — the
/// answer that produced the lease is the answer it acts with.
#[must_use = "dropping the guard releases the lease and admits a second turn on this session"]
pub(crate) struct SessionLeaseGuard {
    leases: Arc<SessionLeases>,
    binding: ModelSessionPlacement,
}

impl SessionLeaseGuard {
    /// The leased conversation, model and placement together.
    pub(crate) fn binding(&self) -> &ModelSessionPlacement {
        &self.binding
    }

    /// The model whose engine owns the leased conversation.
    pub(crate) fn model(&self) -> &ModelKey {
        self.binding.model()
    }

    /// Where the leased conversation lives inside that model's engine.
    pub(crate) fn placement(&self) -> SessionPlacement {
        self.binding.placement()
    }
}

impl Drop for SessionLeaseGuard {
    fn drop(&mut self) {
        self.leases.release(&self.binding);
    }
}

impl fmt::Debug for SessionLeaseGuard {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SessionLeaseGuard")
            .field("binding", &self.binding)
            .finish()
    }
}

/// A lease shard is only ever held across a set insert, lookup or removal, so a
/// poisoned lock carries state that is still sound to act on. Refusing to
/// release a lease because an unrelated thread panicked would strand a session
/// nobody can ever take a turn on again — the opposite of what the guard is for.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Barrier,
        atomic::{AtomicUsize, Ordering},
    };
    use std::thread;

    use super::*;
    use crate::worker::WorkerId;

    fn binding(engine_session_id: onnx_genai::SessionId) -> ModelSessionPlacement {
        model_binding("model-a", engine_session_id)
    }

    fn model_binding(
        model: &str,
        engine_session_id: onnx_genai::SessionId,
    ) -> ModelSessionPlacement {
        ModelSessionPlacement::new(
            ModelKey::new(model),
            SessionPlacement::new(WorkerId::PRIMARY, engine_session_id),
        )
    }

    #[test]
    fn a_lease_is_exclusive_and_names_the_session_it_refuses() {
        let leases = SessionLeases::with_shards(1);
        let held = leases.acquire(binding(7), "sess-a").expect("first turn");

        let conflict = leases
            .acquire(binding(7), "sess-a")
            .expect_err("a second turn on a live session is refused");
        assert_eq!(
            conflict,
            PackageCapabilityError::ExclusiveLeaseConflict {
                session: "sess-a".to_string(),
            }
        );
        assert!(conflict.is_retryable(), "a busy session is retryable");

        // A different conversation is unaffected, even on the same worker.
        let other = leases.acquire(binding(8), "sess-b").expect("other session");
        assert_eq!(other.binding(), &binding(8));

        drop(held);
        assert!(!leases.is_held(&binding(7)));
        assert!(leases.is_held(&binding(8)));
        drop(leases.acquire(binding(7), "sess-a").expect("released"));
    }

    /// The same engine session id on two workers is two conversations.
    #[test]
    fn a_lease_is_keyed_by_placement_not_by_engine_session_id() {
        let leases = SessionLeases::with_shards(2);
        let _first = leases
            .acquire(
                ModelSessionPlacement::new(
                    ModelKey::new("model-a"),
                    SessionPlacement::new(WorkerId::PRIMARY, 3),
                ),
                "sess-a",
            )
            .expect("worker 0");
        let _second = leases
            .acquire(
                ModelSessionPlacement::new(
                    ModelKey::new("model-a"),
                    SessionPlacement::new(WorkerId::new(1), 3),
                ),
                "sess-b",
            )
            .expect("worker 1 holds a different conversation with the same id");
        assert_eq!(leases.held(), 2);
    }

    /// The same placement on two models is two conversations.
    ///
    /// Every engine numbers its own sessions from its own counter and every
    /// pool starts at worker 0, so this collision is the common case rather
    /// than a contrived one: load two models, open one session in each, and
    /// both are worker 0 / session 0. Keyed by placement alone, the second
    /// model's first turn would have been refused as a conflict with the
    /// first model's.
    #[test]
    fn a_lease_is_keyed_by_model_as_well_as_placement() {
        let leases = SessionLeases::with_shards(1);
        let first = leases
            .acquire(model_binding("model-a", 0), "sess-a")
            .expect("model-a's first session");
        let second = leases
            .acquire(model_binding("model-b", 0), "sess-b")
            .expect("model-b's identical placement is a different conversation");
        assert_eq!(leases.held(), 2);

        // And each still refuses its own second turn.
        assert!(
            leases
                .acquire(model_binding("model-b", 0), "sess-b")
                .is_err()
        );
        drop(first);
        assert!(!leases.is_held(&model_binding("model-a", 0)));
        assert!(
            leases.is_held(&model_binding("model-b", 0)),
            "releasing one model's lease does not release the other's",
        );
        drop(second);
        assert_eq!(leases.held(), 0);
    }

    /// Real threads, one barrier, one session: exactly one turn may start.
    ///
    /// Interleaving two calls on one thread would pass against a lease that
    /// never locks anything, which is why this spawns threads and releases them
    /// together rather than calling `acquire` twice in a row.
    #[test]
    fn only_one_of_many_racing_threads_takes_the_lease() {
        const THREADS: usize = 8;
        let leases = SessionLeases::with_shards(1);
        let barrier = Arc::new(Barrier::new(THREADS));
        let winners = Arc::new(AtomicUsize::new(0));
        let conflicts = Arc::new(AtomicUsize::new(0));

        thread::scope(|scope| {
            for _ in 0..THREADS {
                let leases = Arc::clone(&leases);
                let barrier = Arc::clone(&barrier);
                let winners = Arc::clone(&winners);
                let conflicts = Arc::clone(&conflicts);
                scope.spawn(move || {
                    barrier.wait();
                    match leases.acquire(binding(1), "sess-race") {
                        Ok(guard) => {
                            winners.fetch_add(1, Ordering::SeqCst);
                            // Hold it past every other thread's attempt, so a
                            // late loser cannot win by arriving after a release.
                            thread::sleep(std::time::Duration::from_millis(50));
                            drop(guard);
                        }
                        Err(PackageCapabilityError::ExclusiveLeaseConflict { session }) => {
                            assert_eq!(session, "sess-race");
                            conflicts.fetch_add(1, Ordering::SeqCst);
                        }
                        Err(other) => panic!("unexpected refusal: {other}"),
                    }
                });
            }
        });

        assert_eq!(winners.load(Ordering::SeqCst), 1, "one turn may start");
        assert_eq!(conflicts.load(Ordering::SeqCst), THREADS - 1);
        assert_eq!(leases.held(), 0, "every guard released its lease");
    }

    /// Distinct sessions never contend, however many threads ask at once.
    #[test]
    fn racing_threads_on_distinct_sessions_all_take_their_lease() {
        const THREADS: usize = 8;
        let leases = SessionLeases::with_shards(1);
        let barrier = Arc::new(Barrier::new(THREADS));
        let winners = Arc::new(AtomicUsize::new(0));

        thread::scope(|scope| {
            for index in 0..THREADS {
                let leases = Arc::clone(&leases);
                let barrier = Arc::clone(&barrier);
                let winners = Arc::clone(&winners);
                scope.spawn(move || {
                    barrier.wait();
                    let guard = leases
                        .acquire(binding(index as u64), "sess-distinct")
                        .expect("a session of its own");
                    winners.fetch_add(1, Ordering::SeqCst);
                    drop(guard);
                });
            }
        });

        assert_eq!(winners.load(Ordering::SeqCst), THREADS);
        assert_eq!(leases.held(), 0);
    }

    /// Two models' identically placed sessions never contend, even when every
    /// thread asks at the same instant.
    #[test]
    fn racing_threads_on_two_models_identical_placement_both_take_their_lease() {
        const THREADS: usize = 8;
        let leases = SessionLeases::with_shards(1);
        let barrier = Arc::new(Barrier::new(THREADS));
        let winners = Arc::new(AtomicUsize::new(0));

        thread::scope(|scope| {
            for index in 0..THREADS {
                let leases = Arc::clone(&leases);
                let barrier = Arc::clone(&barrier);
                let winners = Arc::clone(&winners);
                scope.spawn(move || {
                    let model = format!("model-{index}");
                    barrier.wait();
                    let guard = leases
                        .acquire(model_binding(&model, 0), "sess-shared-placement")
                        .expect("each model's worker 0 / session 0 is its own conversation");
                    winners.fetch_add(1, Ordering::SeqCst);
                    drop(guard);
                });
            }
        });

        assert_eq!(winners.load(Ordering::SeqCst), THREADS);
        assert_eq!(leases.held(), 0);
    }

    /// A panic while the lease is held still releases it: `Drop` runs during an
    /// unwind, which is the whole reason release is not a cleanup call.
    #[test]
    fn a_panic_while_holding_the_lease_releases_it() {
        let leases = SessionLeases::with_shards(1);
        let panicking = {
            let leases = Arc::clone(&leases);
            thread::spawn(move || {
                let _guard = leases.acquire(binding(5), "sess-panic").expect("held");
                panic!("the turn panicked");
            })
        };
        assert!(panicking.join().is_err(), "the thread panicked");
        assert!(!leases.is_held(&binding(5)), "an unwind releases the lease");
        drop(
            leases
                .acquire(binding(5), "sess-panic")
                .expect("the session is leasable again"),
        );
    }
}

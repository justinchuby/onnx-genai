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
    sync::{Arc, Mutex, MutexGuard, PoisonError},
};

use onnx_genai_engine::PackageCapabilityError;

use crate::worker::{SessionPlacement, WorkerPool};

/// Which sessions currently have a turn in flight.
///
/// Keyed by [`SessionPlacement`] rather than by a bare engine session id: an
/// engine session id is only meaningful on the worker that issued it, so two
/// workers could hand out the same number for two different conversations. The
/// pair is the only thing that names a conversation without ambiguity, which is
/// exactly what a lease key has to do.
///
/// Sharded by worker so the critical section stays short and two workers'
/// request tasks never contend for one lock. With a pool of one there is one
/// shard, and the sharding is the shape rather than the optimization.
pub(crate) struct SessionLeases {
    shards: Vec<Mutex<HashSet<SessionPlacement>>>,
}

impl SessionLeases {
    /// One shard per worker in the pool that will serve these sessions.
    pub(crate) fn for_pool(workers: &WorkerPool) -> Arc<Self> {
        Self::with_shards(workers.len())
    }

    /// A lease map with an explicit shard count, for tests that hold no pool.
    pub(crate) fn with_shards(shards: usize) -> Arc<Self> {
        Arc::new(Self {
            shards: (0..shards.max(1))
                .map(|_| Mutex::new(HashSet::new()))
                .collect(),
        })
    }

    /// The shard a placement's lease lives in.
    ///
    /// Modulo rather than a direct index so the mapping is total: a placement
    /// naming a worker this pool does not have still resolves to a shard, and
    /// is refused later — by [`WorkerPool::worker`](crate::worker::WorkerPool)
    /// — for the reason it is actually wrong, rather than panicking here.
    fn shard(&self, placement: SessionPlacement) -> &Mutex<HashSet<SessionPlacement>> {
        &self.shards[placement.worker.index() % self.shards.len()]
    }

    /// Take the exclusive turn lease for a session, or refuse by name.
    ///
    /// `session` is the identity the caller used — the client's session id —
    /// and appears in the refusal so the answer names something the caller can
    /// act on. It is not the key: the key is the placement.
    pub(crate) fn acquire(
        self: &Arc<Self>,
        placement: SessionPlacement,
        session: &str,
    ) -> Result<SessionLeaseGuard, PackageCapabilityError> {
        if lock(self.shard(placement)).insert(placement) {
            Ok(SessionLeaseGuard {
                leases: Arc::clone(self),
                placement,
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
    pub(crate) fn is_held(&self, placement: SessionPlacement) -> bool {
        lock(self.shard(placement)).contains(&placement)
    }

    fn release(&self, placement: SessionPlacement) {
        lock(self.shard(placement)).remove(&placement);
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
#[must_use = "dropping the guard releases the lease and admits a second turn on this session"]
pub(crate) struct SessionLeaseGuard {
    leases: Arc<SessionLeases>,
    placement: SessionPlacement,
}

impl SessionLeaseGuard {
    /// Where the leased conversation lives.
    pub(crate) fn placement(&self) -> SessionPlacement {
        self.placement
    }
}

impl Drop for SessionLeaseGuard {
    fn drop(&mut self) {
        self.leases.release(self.placement);
    }
}

impl fmt::Debug for SessionLeaseGuard {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SessionLeaseGuard")
            .field("placement", &self.placement)
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

    fn placement(engine_session_id: onnx_genai::SessionId) -> SessionPlacement {
        SessionPlacement::new(WorkerId::PRIMARY, engine_session_id)
    }

    #[test]
    fn a_lease_is_exclusive_and_names_the_session_it_refuses() {
        let leases = SessionLeases::with_shards(1);
        let held = leases.acquire(placement(7), "sess-a").expect("first turn");

        let conflict = leases
            .acquire(placement(7), "sess-a")
            .expect_err("a second turn on a live session is refused");
        assert_eq!(
            conflict,
            PackageCapabilityError::ExclusiveLeaseConflict {
                session: "sess-a".to_string(),
            }
        );
        assert!(conflict.is_retryable(), "a busy session is retryable");

        // A different conversation is unaffected, even on the same worker.
        let other = leases
            .acquire(placement(8), "sess-b")
            .expect("other session");
        assert_eq!(other.placement(), placement(8));

        drop(held);
        assert!(!leases.is_held(placement(7)));
        assert!(leases.is_held(placement(8)));
        drop(leases.acquire(placement(7), "sess-a").expect("released"));
    }

    /// The same engine session id on two workers is two conversations.
    #[test]
    fn a_lease_is_keyed_by_placement_not_by_engine_session_id() {
        let leases = SessionLeases::with_shards(2);
        let _first = leases
            .acquire(SessionPlacement::new(WorkerId::PRIMARY, 3), "sess-a")
            .expect("worker 0");
        let _second = leases
            .acquire(SessionPlacement::new(WorkerId::new(1), 3), "sess-b")
            .expect("worker 1 holds a different conversation with the same id");
        assert_eq!(leases.held(), 2);
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
                    match leases.acquire(placement(1), "sess-race") {
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
                        .acquire(placement(index as u64), "sess-distinct")
                        .expect("a session of its own");
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
                let _guard = leases.acquire(placement(5), "sess-panic").expect("held");
                panic!("the turn panicked");
            })
        };
        assert!(panicking.join().is_err(), "the thread panicked");
        assert!(
            !leases.is_held(placement(5)),
            "an unwind releases the lease"
        );
        drop(
            leases
                .acquire(placement(5), "sess-panic")
                .expect("the session is leasable again"),
        );
    }
}

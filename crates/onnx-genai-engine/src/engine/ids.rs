//! Typed allocators for the id namespaces an engine mints into.
//!
//! `docs/architecture/SESSION_CONCURRENCY.md` §1.2 records that the engine
//! already has more than one session id space, that they "agree by convention,
//! not by type", and §13 Phase 3 requires shard-encoded ids once a worker pool
//! grows past one. Before any of that, a smaller thing has to be true: it has to
//! be possible to say *which* namespace a counter belongs to, and whether a
//! second worker minting from a second copy of it would collide.
//!
//! Two kinds of namespace exist, and the difference decides the mechanism:
//!
//! * **Shared** — the id leaves the engine. A caller holds it, the server's
//!   `SessionRegistry` routes on it, and a future `WorkerPool` may hand two
//!   requests for it to two different threads. Two workers minting from
//!   independent counters would hand the same id to two conversations, so the
//!   counter is a [`SharedSessionIds`] and is atomic.
//! * **Worker-local** — the id never leaves the thread that minted it, and is
//!   only ever compared against state that thread owns. Those live in
//!   `crate::pipeline::runtime_state` (`PassIdAllocator`,
//!   `GraphCaptureIdAllocator`) and are deliberately *not* atomic, because
//!   making them atomic would advertise a sharing §3.2 forbids.
//!
//! **What is not covered.** Decode-core session ids come from
//! `PagedKvCache::create_sequence`, not from here, so they carry their own
//! namespace inside the KV crate. Unifying the two is §13 Phase 0's `SessionId`
//! newtype work and is deliberately not attempted here.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::config::SessionId;

/// Mints session ids that are handed out beyond the engine.
///
/// One allocator owns one namespace. Ids are minted from 1 upward, which is
/// exactly the sequence the `counter += 1; use counter` code it replaces
/// produced — no externally observed id changes.
///
/// `Ordering::Relaxed` is enough: the only guarantee required is that no two
/// mints return the same number. Nothing orders other memory against the mint,
/// and the id's *use* is ordered by whatever handed the id to the other thread.
#[derive(Debug, Default)]
pub(crate) struct SharedSessionIds {
    next: AtomicU64,
}

impl SharedSessionIds {
    pub(crate) fn new() -> Self {
        Self {
            next: AtomicU64::new(0),
        }
    }

    /// The next id in this namespace. Never returns the same id twice.
    pub(crate) fn mint(&self) -> SessionId {
        // `fetch_add` returns the previous value, so the first id is 1.
        SessionId::from(self.next.fetch_add(1, Ordering::Relaxed) + 1)
    }

    /// How many ids this namespace has handed out.
    #[cfg(test)]
    pub(crate) fn minted(&self) -> u64 {
        self.next.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::Arc;

    #[test]
    fn ids_start_at_one_and_never_repeat() {
        let ids = SharedSessionIds::new();
        assert_eq!(ids.minted(), 0);
        let minted = (0..1_000).map(|_| ids.mint()).collect::<Vec<_>>();
        assert_eq!(
            minted.first().copied(),
            Some(SessionId::from(1_u64)),
            "the first session id must stay 1: it is observed by callers"
        );
        assert_eq!(minted.len(), minted.iter().collect::<HashSet<_>>().len());
        assert_eq!(ids.minted(), 1_000);
    }

    #[test]
    fn one_namespace_is_collision_free_across_threads() {
        // The engine is single-threaded today. This is the property that has to
        // hold before it is not, and testing it now is what makes the atomic a
        // decision rather than a habit: an allocator that a future worker pool
        // shares must never hand two workers the same session id.
        let ids = Arc::new(SharedSessionIds::new());
        let threads = (0..8)
            .map(|_| {
                let ids = Arc::clone(&ids);
                std::thread::spawn(move || (0..1_000).map(|_| ids.mint()).collect::<Vec<_>>())
            })
            .collect::<Vec<_>>();
        let minted = threads
            .into_iter()
            .flat_map(|thread| thread.join().expect("mint thread panicked"))
            .collect::<Vec<_>>();
        let unique = minted.iter().copied().collect::<HashSet<_>>();
        assert_eq!(unique.len(), minted.len(), "session ids collided");
        assert_eq!(ids.minted(), 8_000);
    }

    #[test]
    fn separate_namespaces_are_independent() {
        // The interpreted and native session spaces are separate allocators, as
        // they were separate counters. They are never mixed in one engine — a
        // package either holds a decode core or it does not — so equal ids
        // across them are not a collision.
        let interpreted = SharedSessionIds::new();
        let native = SharedSessionIds::new();
        assert_eq!(interpreted.mint(), native.mint());
    }
}

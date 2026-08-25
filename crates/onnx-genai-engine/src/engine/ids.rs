//! Typed allocators for the id namespaces an engine mints into.
//!
//! `docs/architecture/SESSION_CONCURRENCY.md` §1.2 records that the engine
//! already has more than one session id space, that they "agree by convention,
//! not by type". The server now pairs these local ids with `WorkerId`; separate
//! workers may therefore mint the same number without aliasing conversations.
//! Within one engine, it must still be possible to say which namespace a
//! counter belongs to and to mint safely from every code path that shares it.
//!
//! Two kinds of namespace exist, and the difference decides the mechanism:
//!
//! * **Engine-shared** — the local id leaves the engine and is shared by the
//!   engine's session APIs. The server routes the worker-qualified placement,
//!   not this number alone. The counter is a [`SharedSessionIds`] and remains
//!   atomic so all in-engine minting paths share one collision-free namespace.
//! * **Worker-local** — the id never leaves the thread that minted it, and is
//!   only ever compared against state that thread owns. Those live in
//!   `crate::pipeline::runtime_state` (`PassIdAllocator`,
//!   `GraphCaptureIdAllocator`) and are deliberately *not* atomic, because
//!   making them atomic would advertise a sharing §3.2 forbids.
//!
//! **What is not covered.** Decode-core session ids come from
//! `PagedKvCache::create_sequence`, not from here, so they carry their own
//! namespace inside the KV crate. The server's typed placement is the
//! unification boundary; the two local allocators deliberately remain separate.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::config::SessionId;

/// Mints session ids that are handed out beyond one engine.
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
        // One allocator remains collision-free even if multiple in-engine
        // callers mint concurrently; separate workers own separate allocators.
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

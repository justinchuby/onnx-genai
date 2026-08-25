use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use anyhow::Context;

use crate::lease::{SessionLeaseGuard, SessionLeases};
use crate::worker::SessionPlacement;

#[derive(Clone)]
pub(crate) struct SessionRegistry {
    inner: Arc<Mutex<SessionRegistryInner>>,
    max_sessions: usize,
}

#[derive(Debug)]
struct SessionRegistryInner {
    sessions: HashMap<String, SessionEntry>,
    access_clock: u64,
}

/// The outcome of binding a client id to an engine session.
#[derive(Debug)]
pub(crate) enum SessionClaim {
    /// Another request bound this client id first; use its session and release
    /// the one this caller opened.
    Existing(SessionPlacement),
    /// This caller's session is now the client id's session.
    ///
    /// `evicted` is the binding dropped to make room, handed back *holding its
    /// exclusive lease* so the caller can close it on the worker that owns it
    /// without racing a turn: nothing else can start one while the guard lives,
    /// and the guard is released when the close returns.
    Claimed { evicted: Option<SessionLeaseGuard> },
}

/// One client id's binding: where its conversation lives, and when it was last
/// touched (the LRU key).
#[derive(Debug)]
struct SessionEntry {
    /// The worker that owns this conversation and the engine session id inside
    /// it. Stored as a pair because a later turn has to be routed back to that
    /// worker, and an engine session id alone cannot say which one it is.
    placement: SessionPlacement,
    last_access: u64,
}

impl SessionRegistry {
    pub(crate) fn new(max_sessions: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(SessionRegistryInner {
                sessions: HashMap::new(),
                access_clock: 0,
            })),
            max_sessions,
        }
    }

    /// Bind a client id to an engine session, evicting under pressure.
    ///
    /// Takes the lease map because eviction is a *mutation of somebody else's
    /// conversation*: the victim is chosen from the bindings no turn is in
    /// flight on, and is returned holding its lease so the close that follows
    /// cannot race a turn that starts in between.
    pub(crate) fn insert(
        &self,
        client_id: String,
        placement: SessionPlacement,
        leases: &Arc<SessionLeases>,
    ) -> anyhow::Result<Option<SessionLeaseGuard>> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| anyhow::anyhow!("session registry mutex poisoned"))?;
        let previous_len = inner.sessions.len();
        let evicted = if previous_len >= self.max_sessions {
            evict_lru(&mut inner, leases)
        } else {
            None
        };
        inner.access_clock = inner.access_clock.saturating_add(1);
        let last_access = inner.access_clock;
        inner.sessions.insert(
            client_id,
            SessionEntry {
                placement,
                last_access,
            },
        );
        if inner.sessions.len() > previous_len {
            crate::metrics::active_sessions_added(1);
        }
        Ok(evicted)
    }

    /// Bind a client id to an engine session, atomically.
    ///
    /// Two requests carrying the same session id race to open it. Reading and
    /// then inserting lets both miss, both open an engine session, and the
    /// second insert orphan the first — so the two turns run in different
    /// conversations and one of them is silently lost. The decision is made
    /// under one lock instead, and the caller that loses closes the session it
    /// had already opened.
    pub(crate) fn claim(
        &self,
        client_id: String,
        placement: SessionPlacement,
        leases: &Arc<SessionLeases>,
    ) -> anyhow::Result<SessionClaim> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| anyhow::anyhow!("session registry mutex poisoned"))?;
        if inner.sessions.contains_key(&client_id) {
            inner.access_clock = inner.access_clock.saturating_add(1);
            let last_access = inner.access_clock;
            let entry = inner
                .sessions
                .get_mut(&client_id)
                .expect("entry checked above");
            entry.last_access = last_access;
            return Ok(SessionClaim::Existing(entry.placement));
        }
        let previous_len = inner.sessions.len();
        let evicted = if previous_len >= self.max_sessions {
            evict_lru(&mut inner, leases)
        } else {
            None
        };
        inner.access_clock = inner.access_clock.saturating_add(1);
        let last_access = inner.access_clock;
        inner.sessions.insert(
            client_id,
            SessionEntry {
                placement,
                last_access,
            },
        );
        if inner.sessions.len() > previous_len {
            crate::metrics::active_sessions_added(1);
        }
        Ok(SessionClaim::Claimed { evicted })
    }

    pub(crate) fn get(&self, client_id: &str) -> anyhow::Result<Option<SessionPlacement>> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| anyhow::anyhow!("session registry mutex poisoned"))?;
        if !inner.sessions.contains_key(client_id) {
            return Ok(None);
        }
        inner.access_clock = inner.access_clock.saturating_add(1);
        let last_access = inner.access_clock;
        let entry = inner
            .sessions
            .get_mut(client_id)
            .expect("entry checked above");
        entry.last_access = last_access;
        Ok(Some(entry.placement))
    }

    pub(crate) fn remove(&self, client_id: &str) -> anyhow::Result<Option<SessionPlacement>> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|_| anyhow::anyhow!("session registry mutex poisoned"))?;
        let removed = inner
            .sessions
            .remove(client_id)
            .map(|entry| entry.placement);
        if removed.is_some() {
            crate::metrics::active_sessions_removed(1);
        }
        Ok(removed)
    }

    pub(crate) fn next_client_id(&self) -> anyhow::Result<String> {
        let mut bytes = [0_u8; 16];
        getrandom::fill(&mut bytes).context("OS CSPRNG failed")?;
        Ok(format!("sess-{}", hex_token(&bytes)))
    }

    pub(crate) fn client_ids_redacted(&self) -> anyhow::Result<Vec<String>> {
        let inner = self
            .inner
            .lock()
            .map_err(|_| anyhow::anyhow!("session registry mutex poisoned"))?;
        let mut ids: Vec<String> = inner
            .sessions
            .keys()
            .map(|id| redact_session_id(id))
            .collect();
        ids.sort_unstable();
        Ok(ids)
    }

    pub(crate) fn max_sessions(&self) -> usize {
        self.max_sessions
    }
}

impl Drop for SessionRegistry {
    fn drop(&mut self) {
        if Arc::strong_count(&self.inner) == 1
            && let Ok(inner) = self.inner.lock()
        {
            crate::metrics::active_sessions_removed(inner.sessions.len());
        }
    }
}

/// Drop the least recently accessed binding that has no turn in flight, and
/// return it holding its exclusive lease so the caller can close it on the
/// worker that owns it.
///
/// One implementation for both `insert` and `claim`: the two make the same
/// eviction decision, and a second copy of it is a second answer waiting to
/// drift.
///
/// **The victim is chosen by taking its lease, not by asking whether it is
/// free.** Asking first and closing after would leave the window this whole
/// design exists to shut: a turn could take the lease in between, and the close
/// would then destroy the conversation that turn is writing. The candidates are
/// therefore walked oldest-first and the first one whose lease can be taken is
/// the victim.
///
/// If every binding is mid-turn there is no victim, and the registry is left one
/// over its bound rather than a live conversation being closed under its caller.
/// The overshoot is bounded by the number of turns in flight, which the
/// generation-capacity semaphore already bounds, and it resolves at the next
/// insert once any turn ends.
fn evict_lru(
    inner: &mut SessionRegistryInner,
    leases: &Arc<SessionLeases>,
) -> Option<SessionLeaseGuard> {
    let mut candidates: Vec<(u64, &str, SessionPlacement)> = inner
        .sessions
        .iter()
        .map(|(id, entry)| (entry.last_access, id.as_str(), entry.placement))
        .collect();
    candidates.sort_unstable_by_key(|(last_access, _, _)| *last_access);
    let victim = candidates.into_iter().find_map(|(_, id, placement)| {
        leases
            .acquire(placement, id)
            .ok()
            .map(|lease| (id.to_string(), lease))
    });
    let (client_id, lease) = match victim {
        Some(victim) => victim,
        None => {
            tracing::warn!(
                bound = inner.sessions.len(),
                "every registered session has a turn in flight; admitting one over the session \
                 bound rather than closing a live conversation",
            );
            return None;
        }
    };
    inner.sessions.remove(&client_id);
    Some(lease)
}

fn hex_token(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

/// Redact a session capability ID to prevent replay attacks.
///
/// Full IDs like `sess-{32hex}` are bearer tokens; we show only the first 8 hex
/// chars (32 bits) followed by `…`, enough for log correlation without enabling
/// session hijacking or deletion.
fn redact_session_id(id: &str) -> String {
    // Expected format: "sess-<32 hex chars>"
    // Keep the prefix up to and including the first 8 hex chars, then append "…".
    const PREFIX: &str = "sess-";
    const VISIBLE_HEX: usize = 8;
    if let Some(hex_part) = id.strip_prefix(PREFIX) {
        let keep = hex_part.len().min(VISIBLE_HEX);
        format!("{PREFIX}{}…", &hex_part[..keep])
    } else {
        // Unknown format — redact entirely.
        "[redacted]".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::WorkerId;

    fn placement(engine_session_id: onnx_genai::SessionId) -> SessionPlacement {
        SessionPlacement::new(WorkerId::PRIMARY, engine_session_id)
    }

    fn leases() -> Arc<SessionLeases> {
        SessionLeases::with_shards(1)
    }

    /// What was evicted, by placement, so a test can assert on it after the
    /// guard that carried it has been released.
    fn evicted_placement(evicted: &Option<SessionLeaseGuard>) -> Option<SessionPlacement> {
        evicted.as_ref().map(SessionLeaseGuard::placement)
    }

    #[test]
    fn a_bound_session_remembers_the_worker_that_owns_it() {
        let registry = SessionRegistry::new(4);
        let leases = leases();
        let bound = SessionPlacement::new(WorkerId::new(0), 7);

        assert!(
            registry
                .insert("sess-a".to_string(), bound, &leases)
                .unwrap()
                .is_none()
        );

        let found = registry.get("sess-a").unwrap().expect("session is bound");
        assert_eq!(found, bound);
        assert_eq!(found.worker, WorkerId::PRIMARY);
        assert_eq!(found.engine_session_id, 7);
    }

    #[test]
    fn an_unbound_client_id_has_no_session() {
        let registry = SessionRegistry::new(4);
        assert_eq!(registry.get("sess-missing").unwrap(), None);
        assert_eq!(registry.remove("sess-missing").unwrap(), None);
    }

    #[test]
    fn removing_a_session_returns_its_placement_once() {
        let registry = SessionRegistry::new(4);
        registry
            .insert("sess-a".to_string(), placement(1), &leases())
            .unwrap();

        assert_eq!(registry.remove("sess-a").unwrap(), Some(placement(1)));
        assert_eq!(registry.remove("sess-a").unwrap(), None);
        assert_eq!(registry.get("sess-a").unwrap(), None);
    }

    /// The registry is capped, and the binding evicted to make room is the one
    /// accessed longest ago — not the one inserted first.
    #[test]
    fn insert_evicts_the_least_recently_accessed_session() {
        let registry = SessionRegistry::new(2);
        let leases = leases();
        registry
            .insert("sess-a".to_string(), placement(1), &leases)
            .unwrap();
        registry
            .insert("sess-b".to_string(), placement(2), &leases)
            .unwrap();

        // Touching "a" makes "b" the least recently accessed binding.
        registry.get("sess-a").unwrap().expect("a is bound");

        let evicted = registry
            .insert("sess-c".to_string(), placement(3), &leases)
            .unwrap();
        assert_eq!(
            evicted_placement(&evicted),
            Some(placement(2)),
            "LRU 'b' must be evicted"
        );
        assert!(
            leases.is_held(placement(2)),
            "an evicted session is handed back leased, so its close cannot race a turn",
        );
        drop(evicted);
        assert!(!leases.is_held(placement(2)));
        assert_eq!(registry.get("sess-b").unwrap(), None);
        assert_eq!(registry.get("sess-a").unwrap(), Some(placement(1)));
        assert_eq!(registry.get("sess-c").unwrap(), Some(placement(3)));
    }

    /// A second request for a client id that is already bound loses: it is told
    /// which session to use, and closes the one it opened.
    #[test]
    fn claim_returns_the_existing_binding_and_refreshes_it() {
        let registry = SessionRegistry::new(2);
        let leases = leases();
        registry
            .insert("sess-a".to_string(), placement(1), &leases)
            .unwrap();
        registry
            .insert("sess-b".to_string(), placement(2), &leases)
            .unwrap();

        let claim = registry
            .claim("sess-a".to_string(), placement(9), &leases)
            .unwrap();
        assert!(matches!(claim, SessionClaim::Existing(existing) if existing == placement(1)));

        // The losing claim also counts as an access, so "b" is now the LRU.
        let evicted = registry
            .insert("sess-c".to_string(), placement(3), &leases)
            .unwrap();
        assert_eq!(evicted_placement(&evicted), Some(placement(2)));
    }

    #[test]
    fn claim_binds_a_new_client_id_and_evicts_under_pressure() {
        let registry = SessionRegistry::new(1);
        let leases = leases();
        registry
            .insert("sess-a".to_string(), placement(1), &leases)
            .unwrap();

        let claim = registry
            .claim("sess-b".to_string(), placement(2), &leases)
            .unwrap();
        let SessionClaim::Claimed { evicted } = claim else {
            panic!("an unbound client id is claimed, not matched to an existing session");
        };
        assert_eq!(evicted_placement(&evicted), Some(placement(1)));
        assert_eq!(registry.get("sess-b").unwrap(), Some(placement(2)));
        assert_eq!(registry.get("sess-a").unwrap(), None);
    }

    /// A binding with a turn in flight is not the one eviction takes.
    ///
    /// The LRU binding is the obvious victim right up until it is the one being
    /// generated into: closing it would destroy the conversation its own caller
    /// is mid-way through writing, and the caller would see the turn fail for a
    /// reason it never asked about. Eviction skips it and takes the next oldest
    /// binding that is idle.
    #[test]
    fn eviction_skips_a_session_with_a_turn_in_flight() {
        let registry = SessionRegistry::new(2);
        let leases = leases();
        registry
            .insert("sess-busy".to_string(), placement(1), &leases)
            .unwrap();
        registry
            .insert("sess-idle".to_string(), placement(2), &leases)
            .unwrap();

        // "busy" is the least recently accessed binding, and is mid-turn.
        let turn = leases
            .acquire(placement(1), "sess-busy")
            .expect("a turn on the oldest session");

        let evicted = registry
            .insert("sess-new".to_string(), placement(3), &leases)
            .unwrap();
        assert_eq!(
            evicted_placement(&evicted),
            Some(placement(2)),
            "the idle binding is evicted, not the one being generated into",
        );
        assert_eq!(registry.get("sess-busy").unwrap(), Some(placement(1)));
        drop(turn);
    }

    /// When every binding is mid-turn there is no victim, and the registry
    /// admits one over its bound rather than closing a live conversation.
    #[test]
    fn eviction_refuses_to_close_the_only_sessions_that_are_all_busy() {
        let registry = SessionRegistry::new(1);
        let leases = leases();
        registry
            .insert("sess-busy".to_string(), placement(1), &leases)
            .unwrap();
        let turn = leases.acquire(placement(1), "sess-busy").expect("a turn");

        let evicted = registry
            .insert("sess-new".to_string(), placement(2), &leases)
            .unwrap();
        assert!(evicted.is_none(), "a live conversation is not evictable");
        assert_eq!(registry.get("sess-busy").unwrap(), Some(placement(1)));
        assert_eq!(registry.get("sess-new").unwrap(), Some(placement(2)));

        // And the overshoot resolves as soon as the turn ends.
        drop(turn);
        let evicted = registry
            .insert("sess-later".to_string(), placement(3), &leases)
            .unwrap();
        assert_eq!(evicted_placement(&evicted), Some(placement(1)));
    }

    #[test]
    fn claimed_sessions_are_listed_redacted() {
        let registry = SessionRegistry::new(4);
        let client_id = registry.next_client_id().unwrap();
        registry
            .insert(client_id.clone(), placement(1), &leases())
            .unwrap();

        let listed = registry.client_ids_redacted().unwrap();
        assert_eq!(listed.len(), 1);
        assert!(listed[0].starts_with("sess-"));
        assert!(listed[0].ends_with('…'));
        assert!(!listed[0].contains(&client_id[13..]));
        assert_eq!(registry.max_sessions(), 4);
    }
}

use std::{
    collections::HashMap,
    fmt,
    sync::{Arc, Mutex},
};

use anyhow::Context;
use onnx_genai_engine::PackageCapabilityError;

use crate::lease::{ModelSessionPlacement, SessionLeaseGuard, SessionLeases};

/// Where a client's conversations live, and which of them have a turn in flight.
///
/// One registry spans every loaded model, so every binding it holds is stored
/// model-qualified and every lease it takes is taken on the *same*
/// model-qualified key. The lease map is owned here rather than by an engine
/// for exactly that reason: an engine-owned map would answer "is this session
/// busy?" only for the engine the caller happened to be holding, and the
/// caller holding the wrong engine is the failure this registry has to make
/// impossible.
///
/// **Lock order: registry mutex, then lease shard.** Eviction and close both
/// take a lease while holding the registry lock, so that the binding they
/// remove is the binding they leased. Nothing takes the registry lock while
/// holding a lease shard, and `acquire` never blocks on anything, so the order
/// is total and deadlock-free.
#[derive(Clone)]
pub(crate) struct SessionRegistry {
    inner: Arc<Mutex<SessionRegistryInner>>,
    /// Which bindings have a turn in flight. Shared with every route, and the
    /// only map any of them consults.
    leases: Arc<SessionLeases>,
    max_sessions: usize,
    /// Where the size of the map is reported. Bound once at construction so
    /// that no mutation site can pick a different destination.
    gauge: SessionGauge,
}

/// Where the registry reports that the number of live conversations changed.
///
/// Production reports to the process-global `active_sessions` gauge. Tests
/// point a registry at a counter of their own, because the global one cannot be
/// asserted on exactly: every other test in this binary that opens or closes a
/// session moves the same counter, so a baseline-and-delta assertion against it
/// races the rest of the suite rather than measuring this registry. The
/// arithmetic under test is identical either way — only the destination differs.
#[derive(Debug, Clone)]
pub(crate) enum SessionGauge {
    /// The process-global gauge served by `/metrics` and the admin snapshot.
    Global,
    #[cfg(test)]
    Local(Arc<std::sync::atomic::AtomicI64>),
}

impl SessionGauge {
    fn added(&self, count: usize) {
        match self {
            Self::Global => crate::metrics::active_sessions_added(count),
            #[cfg(test)]
            Self::Local(counter) => {
                counter.fetch_add(count as i64, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    fn removed(&self, count: usize) {
        match self {
            Self::Global => crate::metrics::active_sessions_removed(count),
            #[cfg(test)]
            Self::Local(counter) => {
                counter.fetch_sub(count as i64, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }
}

#[derive(Debug)]
struct SessionRegistryInner {
    sessions: HashMap<String, SessionEntry>,
    access_clock: u64,
}

/// Why the registry would not bind a client id.
#[derive(Debug)]
pub(crate) enum SessionRegistryError {
    /// The registry mutex was poisoned by a panic in another request.
    Poisoned,
    /// The registry is full and every binding in it has a turn in flight, so
    /// there is nothing that can be evicted to make room.
    ///
    /// A refusal rather than an overshoot: `max_sessions` is what an operator
    /// sized the server's session memory against, and a bound that yields under
    /// load is not a bound. The condition is transient — it clears as soon as
    /// any turn in flight ends — which is why it is reported as a resource
    /// limit the caller may retry, not as a permanent rejection.
    AtCapacity { bound: usize },
    /// The client id is already bound. Callers that mint their own id mint a
    /// fresh one; callers that take an id from a client use
    /// [`SessionRegistry::claim`], which resolves the collision instead.
    AlreadyBound,
}

impl fmt::Display for SessionRegistryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Poisoned => formatter.write_str("session registry mutex poisoned"),
            Self::AtCapacity { bound } => write!(
                formatter,
                "all {bound} sessions are at capacity with a turn in flight; retry once one ends"
            ),
            Self::AlreadyBound => formatter.write_str("session id is already bound"),
        }
    }
}

impl std::error::Error for SessionRegistryError {}

/// Why the registry would not hand a binding over to be closed.
#[derive(Debug)]
pub(crate) enum SessionCloseError {
    Registry(SessionRegistryError),
    /// No such client id — either never bound, or a concurrent close won.
    NotFound,
    /// A turn is in flight on this session. Closing it would destroy the state
    /// that turn is writing, so the close is refused with the same conflict an
    /// overlapping turn gets.
    Busy(PackageCapabilityError),
}

impl fmt::Display for SessionCloseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Registry(error) => error.fmt(formatter),
            Self::NotFound => formatter.write_str("session not found"),
            Self::Busy(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for SessionCloseError {}

/// The outcome of binding a client id to an engine session.
#[derive(Debug)]
pub(crate) enum SessionClaim {
    /// Another request bound this client id first; use its session and release
    /// the one this caller opened.
    Existing(ModelSessionPlacement),
    /// This caller's session is now the client id's session.
    ///
    /// `evicted` is the binding dropped to make room, handed back *holding its
    /// exclusive lease* so the caller can close it on the model that owns it
    /// without racing a turn: nothing else can start one while the guard lives,
    /// and the guard is released when the close returns.
    Claimed { evicted: Option<SessionLeaseGuard> },
}

/// One client id's binding: which conversation it names, and when it was last
/// touched (the LRU key).
#[derive(Debug)]
struct SessionEntry {
    /// The model whose engine owns this conversation, the worker inside that
    /// engine, and the engine session id inside that worker. All three, because
    /// a later turn has to be routed back to exactly that engine: an engine
    /// session id alone cannot say which worker issued it, and a placement
    /// alone cannot say which model's engine did.
    binding: ModelSessionPlacement,
    last_access: u64,
}

impl SessionRegistry {
    pub(crate) fn new(max_sessions: usize) -> Self {
        Self::with_gauge(max_sessions, SessionGauge::Global)
    }

    fn with_gauge(max_sessions: usize, gauge: SessionGauge) -> Self {
        Self {
            inner: Arc::new(Mutex::new(SessionRegistryInner {
                sessions: HashMap::new(),
                access_clock: 0,
            })),
            leases: SessionLeases::new(),
            max_sessions,
            gauge,
        }
    }

    /// A registry that reports its size changes to a counter the caller owns.
    #[cfg(test)]
    pub(crate) fn with_local_gauge(
        max_sessions: usize,
        counter: &Arc<std::sync::atomic::AtomicI64>,
    ) -> Self {
        Self::with_gauge(max_sessions, SessionGauge::Local(Arc::clone(counter)))
    }

    /// The one lease map every route and every registry decision consults.
    ///
    /// Production code never needs it: every acquisition goes through
    /// [`SessionRegistry::acquire`], and every decision that must not disturb a
    /// live conversation takes the lease rather than asking about it. Tests use
    /// it to assert that a lease is held while a turn runs, and that nothing
    /// leaked once every turn has ended.
    #[cfg(test)]
    pub(crate) fn leases(&self) -> &Arc<SessionLeases> {
        &self.leases
    }

    /// Take the exclusive turn lease for a binding, or refuse by name.
    ///
    /// Routed through the registry so no caller has to find a lease map: there
    /// is one, it belongs to the registry that holds the bindings, and asking
    /// any other source would be asking the wrong question.
    pub(crate) fn acquire(
        &self,
        binding: ModelSessionPlacement,
        client_id: &str,
    ) -> Result<SessionLeaseGuard, PackageCapabilityError> {
        self.leases.acquire(binding, client_id)
    }

    /// Bind a *fresh* client id to a conversation, evicting under pressure.
    ///
    /// Eviction is a mutation of somebody else's conversation, so the victim is
    /// chosen from the bindings no turn is in flight on, and is returned holding
    /// its lease: the close that follows cannot race a turn that starts in
    /// between, and it closes on the model the guard names rather than on
    /// whichever model the caller happens to be holding.
    ///
    /// Refuses with [`SessionRegistryError::AtCapacity`] rather than admitting
    /// one over the bound when every binding is busy.
    pub(crate) fn insert(
        &self,
        client_id: String,
        binding: ModelSessionPlacement,
    ) -> Result<Option<SessionLeaseGuard>, SessionRegistryError> {
        let mut inner = self.lock()?;
        if inner.sessions.contains_key(&client_id) {
            return Err(SessionRegistryError::AlreadyBound);
        }
        let evicted = self.make_room(&mut inner)?;
        inner.bind(client_id, binding, &self.gauge);
        Ok(evicted)
    }

    /// Bind a client id to a conversation, atomically.
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
        binding: ModelSessionPlacement,
    ) -> Result<SessionClaim, SessionRegistryError> {
        let mut inner = self.lock()?;
        if let Some(existing) = inner.touch(&client_id) {
            return Ok(SessionClaim::Existing(existing));
        }
        let evicted = self.make_room(&mut inner)?;
        inner.bind(client_id, binding, &self.gauge);
        Ok(SessionClaim::Claimed { evicted })
    }

    /// The conversation a client id names, model included.
    pub(crate) fn get(
        &self,
        client_id: &str,
    ) -> Result<Option<ModelSessionPlacement>, SessionRegistryError> {
        Ok(self.lock()?.touch(client_id))
    }

    /// Unbind a client id and hand its conversation over to be closed, under
    /// the lease that proves no turn is in flight on it.
    ///
    /// One call rather than get → acquire → remove, because those three are
    /// three separate decisions about a binding that can change between them:
    /// the entry read first can be rebound to a different conversation before
    /// the remove, and the close would then destroy a session the caller never
    /// leased while orphaning the one it did. Holding the registry lock across
    /// the whole window makes the binding that is leased, the binding that is
    /// removed, and the binding that is closed the same one by construction.
    ///
    /// The returned guard names the model, so the caller closes on the engine
    /// that owns the conversation instead of on a default that may not.
    pub(crate) fn take_for_close(
        &self,
        client_id: &str,
    ) -> Result<SessionLeaseGuard, SessionCloseError> {
        let mut inner = self.lock().map_err(SessionCloseError::Registry)?;
        let binding = inner
            .sessions
            .get(client_id)
            .map(|entry| entry.binding.clone())
            .ok_or(SessionCloseError::NotFound)?;
        let guard = self
            .leases
            .acquire(binding.clone(), client_id)
            .map_err(SessionCloseError::Busy)?;
        let removed = inner
            .unbind(client_id, &self.gauge)
            .expect("entry read under the same lock");
        debug_assert_eq!(
            removed.binding, binding,
            "the binding removed must be the binding leased",
        );
        Ok(guard)
    }

    pub(crate) fn next_client_id(&self) -> anyhow::Result<String> {
        let mut bytes = [0_u8; 16];
        getrandom::fill(&mut bytes).context("OS CSPRNG failed")?;
        Ok(format!("sess-{}", hex_token(&bytes)))
    }

    pub(crate) fn client_ids_redacted(&self) -> Result<Vec<String>, SessionRegistryError> {
        let inner = self.lock()?;
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

    /// How many client ids are bound right now. Tests assert the bound holds.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.lock().expect("registry lock").sessions.len()
    }

    fn lock(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, SessionRegistryInner>, SessionRegistryError> {
        self.inner
            .lock()
            .map_err(|_| SessionRegistryError::Poisoned)
    }

    /// Make room for one more binding, or refuse.
    ///
    /// Returns the evicted binding holding its lease when one was taken, and
    /// `None` when the registry was below its bound and nothing had to go.
    fn make_room(
        &self,
        inner: &mut SessionRegistryInner,
    ) -> Result<Option<SessionLeaseGuard>, SessionRegistryError> {
        if inner.sessions.len() < self.max_sessions {
            return Ok(None);
        }
        evict_lru(inner, &self.leases, &self.gauge).map(Some)
    }
}

impl SessionRegistryInner {
    /// Record an access and report the binding, or `None` if unbound.
    fn touch(&mut self, client_id: &str) -> Option<ModelSessionPlacement> {
        if !self.sessions.contains_key(client_id) {
            return None;
        }
        self.access_clock = self.access_clock.saturating_add(1);
        let last_access = self.access_clock;
        let entry = self
            .sessions
            .get_mut(client_id)
            .expect("entry checked above");
        entry.last_access = last_access;
        Some(entry.binding.clone())
    }

    /// Bind a client id that is known not to be bound yet.
    ///
    /// **The gauge moves here, at the mutation, and only here.** Reporting from
    /// the callers instead is how the count and the map drift apart: `insert`
    /// and `claim` cannot see whether `make_room` evicted somebody, so an
    /// unconditional increment in either of them counts a replacement as a new
    /// conversation and the gauge climbs forever under LRU churn. Reporting
    /// from the `HashMap` call itself makes the gauge a function of what the map
    /// actually did — a rebind that displaces an entry is a replacement, not an
    /// addition, and is reported as neither.
    fn bind(&mut self, client_id: String, binding: ModelSessionPlacement, gauge: &SessionGauge) {
        self.access_clock = self.access_clock.saturating_add(1);
        let last_access = self.access_clock;
        let displaced = self.sessions.insert(
            client_id,
            SessionEntry {
                binding,
                last_access,
            },
        );
        if displaced.is_none() {
            gauge.added(1);
        }
    }

    /// Drop a binding and report it, or report nothing if it was not bound.
    ///
    /// The counterpart of [`SessionRegistryInner::bind`], and the only place an
    /// entry leaves the map while the registry is alive: eviction and close
    /// both go through it, so neither can decrement twice or forget to.
    fn unbind(&mut self, client_id: &str, gauge: &SessionGauge) -> Option<SessionEntry> {
        let removed = self.sessions.remove(client_id);
        if removed.is_some() {
            gauge.removed(1);
        }
        removed
    }
}

impl Drop for SessionRegistry {
    fn drop(&mut self) {
        if Arc::strong_count(&self.inner) == 1
            && let Ok(inner) = self.inner.lock()
        {
            self.gauge.removed(inner.sessions.len());
        }
    }
}

/// Drop the least recently accessed binding that has no turn in flight, and
/// return it holding its exclusive lease so the caller can close it on the
/// model that owns it.
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
/// the victim, and it is removed while that lease is still held.
///
/// If every binding is mid-turn there is no victim, and the caller is refused.
/// Admitting one over the bound instead would have made `max_sessions` advisory
/// — permanently, since nothing ever walks the registry back down — so a server
/// sized for *n* conversations could be pushed to hold *n + k* of them and stay
/// there. The refusal is transient by construction: it clears the moment any
/// turn in flight ends.
///
/// The victim leaves through [`SessionRegistryInner::unbind`], so the gauge is
/// decremented here and the replacement increments it in `bind`: an eviction
/// followed by an insertion nets to zero, which is what the map does.
fn evict_lru(
    inner: &mut SessionRegistryInner,
    leases: &Arc<SessionLeases>,
    gauge: &SessionGauge,
) -> Result<SessionLeaseGuard, SessionRegistryError> {
    let mut candidates: Vec<(u64, &str, ModelSessionPlacement)> = inner
        .sessions
        .iter()
        .map(|(id, entry)| (entry.last_access, id.as_str(), entry.binding.clone()))
        .collect();
    candidates.sort_unstable_by_key(|(last_access, _, _)| *last_access);
    let victim = candidates.into_iter().find_map(|(_, id, binding)| {
        leases
            .acquire(binding, id)
            .ok()
            .map(|lease| (id.to_string(), lease))
    });
    let Some((client_id, lease)) = victim else {
        let bound = inner.sessions.len();
        tracing::warn!(
            bound,
            "every registered session has a turn in flight; refusing a new one rather than \
             closing a live conversation or exceeding the session bound",
        );
        return Err(SessionRegistryError::AtCapacity { bound });
    };
    let removed = inner
        .unbind(&client_id, gauge)
        .expect("victim read under the same lock");
    debug_assert_eq!(
        &removed.binding,
        lease.binding(),
        "the binding evicted must be the binding leased",
    );
    Ok(lease)
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
    use crate::lease::ModelKey;
    use crate::worker::{SessionPlacement, WorkerId};

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

    /// What was evicted, by binding, so a test can assert on it after the
    /// guard that carried it has been released.
    fn evicted_binding(evicted: &Option<SessionLeaseGuard>) -> Option<ModelSessionPlacement> {
        evicted.as_ref().map(|lease| lease.binding().clone())
    }

    #[test]
    fn a_bound_session_remembers_the_model_and_worker_that_own_it() {
        let registry = SessionRegistry::new(4);
        let bound = model_binding("model-a", 7);

        assert!(
            registry
                .insert("sess-a".to_string(), bound.clone())
                .unwrap()
                .is_none()
        );

        let found = registry.get("sess-a").unwrap().expect("session is bound");
        assert_eq!(found, bound);
        assert_eq!(found.model().as_str(), "model-a");
        assert_eq!(found.placement().worker, WorkerId::PRIMARY);
        assert_eq!(found.placement().engine_session_id, 7);
    }

    #[test]
    fn an_unbound_client_id_has_no_session() {
        let registry = SessionRegistry::new(4);
        assert_eq!(registry.get("sess-missing").unwrap(), None);
        assert!(matches!(
            registry.take_for_close("sess-missing"),
            Err(SessionCloseError::NotFound)
        ));
    }

    #[test]
    fn taking_a_session_for_close_unbinds_it_exactly_once() {
        let registry = SessionRegistry::new(4);
        registry.insert("sess-a".to_string(), binding(1)).unwrap();

        let taken = registry.take_for_close("sess-a").expect("bound");
        assert_eq!(taken.binding(), &binding(1));
        assert!(
            registry.leases().is_held(&binding(1)),
            "the binding is handed over leased, so nothing can take a turn on it",
        );
        assert!(matches!(
            registry.take_for_close("sess-a"),
            Err(SessionCloseError::NotFound)
        ));
        assert_eq!(registry.get("sess-a").unwrap(), None);
        drop(taken);
        assert_eq!(registry.leases().held(), 0);
    }

    /// A close that races a live turn is refused, and the binding survives.
    #[test]
    fn taking_a_busy_session_for_close_is_refused_and_leaves_it_bound() {
        let registry = SessionRegistry::new(4);
        registry.insert("sess-a".to_string(), binding(1)).unwrap();
        let turn = registry.acquire(binding(1), "sess-a").expect("a turn");

        let refused = registry.take_for_close("sess-a").expect_err("busy");
        assert!(matches!(
            refused,
            SessionCloseError::Busy(PackageCapabilityError::ExclusiveLeaseConflict { ref session })
                if session == "sess-a"
        ));
        assert_eq!(
            registry.get("sess-a").unwrap(),
            Some(binding(1)),
            "a refused close leaves the conversation bound",
        );

        drop(turn);
        assert!(registry.take_for_close("sess-a").is_ok());
    }

    /// The registry is capped, and the binding evicted to make room is the one
    /// accessed longest ago — not the one inserted first.
    #[test]
    fn insert_evicts_the_least_recently_accessed_session() {
        let registry = SessionRegistry::new(2);
        registry.insert("sess-a".to_string(), binding(1)).unwrap();
        registry.insert("sess-b".to_string(), binding(2)).unwrap();

        // Touching "a" makes "b" the least recently accessed binding.
        registry.get("sess-a").unwrap().expect("a is bound");

        let evicted = registry.insert("sess-c".to_string(), binding(3)).unwrap();
        assert_eq!(
            evicted_binding(&evicted),
            Some(binding(2)),
            "LRU 'b' must be evicted"
        );
        assert!(
            registry.leases().is_held(&binding(2)),
            "an evicted session is handed back leased, so its close cannot race a turn",
        );
        drop(evicted);
        assert!(!registry.leases().is_held(&binding(2)));
        assert_eq!(registry.get("sess-b").unwrap(), None);
        assert_eq!(registry.get("sess-a").unwrap(), Some(binding(1)));
        assert_eq!(registry.get("sess-c").unwrap(), Some(binding(3)));
        assert_eq!(registry.len(), 2, "the bound holds");
    }

    /// A second request for a client id that is already bound loses: it is told
    /// which session to use, and closes the one it opened.
    #[test]
    fn claim_returns_the_existing_binding_and_refreshes_it() {
        let registry = SessionRegistry::new(2);
        registry.insert("sess-a".to_string(), binding(1)).unwrap();
        registry.insert("sess-b".to_string(), binding(2)).unwrap();

        let claim = registry.claim("sess-a".to_string(), binding(9)).unwrap();
        assert!(matches!(claim, SessionClaim::Existing(ref existing) if existing == &binding(1)));

        // The losing claim also counts as an access, so "b" is now the LRU.
        let evicted = registry.insert("sess-c".to_string(), binding(3)).unwrap();
        assert_eq!(evicted_binding(&evicted), Some(binding(2)));
    }

    #[test]
    fn claim_binds_a_new_client_id_and_evicts_under_pressure() {
        let registry = SessionRegistry::new(1);
        registry.insert("sess-a".to_string(), binding(1)).unwrap();

        let claim = registry.claim("sess-b".to_string(), binding(2)).unwrap();
        let SessionClaim::Claimed { evicted } = claim else {
            panic!("an unbound client id is claimed, not matched to an existing session");
        };
        assert_eq!(evicted_binding(&evicted), Some(binding(1)));
        assert_eq!(registry.get("sess-b").unwrap(), Some(binding(2)));
        assert_eq!(registry.get("sess-a").unwrap(), None);
        assert_eq!(registry.len(), 1);
    }

    /// An id that is already bound is refused rather than silently orphaning
    /// the conversation it already names.
    #[test]
    fn insert_refuses_an_id_that_is_already_bound() {
        let registry = SessionRegistry::new(4);
        registry.insert("sess-a".to_string(), binding(1)).unwrap();
        assert!(matches!(
            registry.insert("sess-a".to_string(), binding(2)),
            Err(SessionRegistryError::AlreadyBound)
        ));
        assert_eq!(registry.get("sess-a").unwrap(), Some(binding(1)));
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
        registry
            .insert("sess-busy".to_string(), binding(1))
            .unwrap();
        registry
            .insert("sess-idle".to_string(), binding(2))
            .unwrap();

        // "busy" is the least recently accessed binding, and is mid-turn.
        let turn = registry
            .acquire(binding(1), "sess-busy")
            .expect("a turn on the oldest session");

        let evicted = registry.insert("sess-new".to_string(), binding(3)).unwrap();
        assert_eq!(
            evicted_binding(&evicted),
            Some(binding(2)),
            "the idle binding is evicted, not the one being generated into",
        );
        assert_eq!(registry.get("sess-busy").unwrap(), Some(binding(1)));
        assert_eq!(registry.len(), 2, "the bound holds");
        drop(turn);
    }

    /// When every binding is mid-turn there is no victim, and the new session
    /// is refused rather than admitted over the bound.
    ///
    /// The refusal is what keeps `max_sessions` a bound. Admitting one instead
    /// would have left the registry permanently at `max + k`, because nothing
    /// ever walks it back down — the next insert would evict one and add one.
    #[test]
    fn a_full_registry_of_busy_sessions_refuses_a_new_one() {
        let registry = SessionRegistry::new(1);
        registry
            .insert("sess-busy".to_string(), binding(1))
            .unwrap();
        let turn = registry.acquire(binding(1), "sess-busy").expect("a turn");

        let refused = registry
            .insert("sess-new".to_string(), binding(2))
            .expect_err("a live conversation is not evictable, and the bound is a bound");
        assert!(matches!(
            refused,
            SessionRegistryError::AtCapacity { bound: 1 }
        ));
        assert_eq!(registry.get("sess-busy").unwrap(), Some(binding(1)));
        assert_eq!(registry.get("sess-new").unwrap(), None);
        assert_eq!(registry.len(), 1, "the registry never exceeded its bound");

        // The refusal is transient: it clears the moment the turn ends.
        drop(turn);
        let evicted = registry
            .insert("sess-later".to_string(), binding(3))
            .unwrap();
        assert_eq!(evicted_binding(&evicted), Some(binding(1)));
        assert_eq!(registry.len(), 1);
    }

    /// `claim` refuses on the same terms `insert` does.
    #[test]
    fn a_full_registry_of_busy_sessions_refuses_a_claim() {
        let registry = SessionRegistry::new(1);
        registry
            .insert("sess-busy".to_string(), binding(1))
            .unwrap();
        let turn = registry.acquire(binding(1), "sess-busy").expect("a turn");

        assert!(matches!(
            registry.claim("sess-new".to_string(), binding(2)),
            Err(SessionRegistryError::AtCapacity { bound: 1 })
        ));
        assert_eq!(registry.len(), 1);
        drop(turn);
        assert!(registry.claim("sess-new".to_string(), binding(2)).is_ok());
        assert_eq!(registry.len(), 1);
    }

    /// Two models' sessions can have the identical placement, and the registry
    /// tells them apart.
    ///
    /// Both engines number their sessions from their own counter and both pools
    /// start at worker 0, so this is what a two-model server looks like after
    /// one session is opened in each. Keyed by placement alone, model A's turn
    /// would have refused as a conflict with model B's, and A's eviction would
    /// have closed B's conversation on A's engine.
    #[test]
    fn two_models_identical_placements_are_two_conversations() {
        let registry = SessionRegistry::new(4);
        let a = model_binding("model-a", 0);
        let b = model_binding("model-b", 0);
        registry.insert("sess-a".to_string(), a.clone()).unwrap();
        registry.insert("sess-b".to_string(), b.clone()).unwrap();

        let turn_on_b = registry.acquire(b.clone(), "sess-b").expect("a turn on B");
        // A's turn is not blocked by B's, and A's close is not blocked either.
        let turn_on_a = registry.acquire(a.clone(), "sess-a").expect("a turn on A");
        drop(turn_on_a);

        let refused = registry.take_for_close("sess-b").expect_err("B is busy");
        assert!(matches!(refused, SessionCloseError::Busy(_)));
        let taken = registry.take_for_close("sess-a").expect("A is idle");
        assert_eq!(
            taken.model().as_str(),
            "model-a",
            "the guard names the model to close on",
        );
        drop(taken);
        drop(turn_on_b);
    }

    /// An active session on one model cannot be evicted to make room for
    /// another model's, and the eviction that does happen names the right
    /// model.
    #[test]
    fn eviction_across_models_skips_the_busy_one_and_names_its_owner() {
        let registry = SessionRegistry::new(2);
        let a = model_binding("model-a", 0);
        let b = model_binding("model-b", 0);
        registry.insert("sess-a".to_string(), a.clone()).unwrap();
        registry.insert("sess-b".to_string(), b.clone()).unwrap();

        // "a" is the oldest binding, and is mid-turn: "b" goes instead.
        let turn_on_a = registry.acquire(a.clone(), "sess-a").expect("a turn on A");
        let evicted = registry
            .insert("sess-c".to_string(), model_binding("model-a", 1))
            .unwrap()
            .expect("something had to go");
        assert_eq!(evicted.binding(), &b);
        assert_eq!(
            evicted.model().as_str(),
            "model-b",
            "the evicted conversation is closed on model-b's engine, not the default's",
        );
        drop(evicted);

        // With both remaining bindings busy the registry refuses rather than
        // closing the live one.
        let turn_on_c = registry
            .acquire(model_binding("model-a", 1), "sess-c")
            .expect("a turn on C");
        assert!(matches!(
            registry.insert("sess-d".to_string(), model_binding("model-b", 1)),
            Err(SessionRegistryError::AtCapacity { bound: 2 })
        ));
        assert_eq!(registry.len(), 2);
        drop(turn_on_a);
        drop(turn_on_c);
    }

    #[test]
    fn claimed_sessions_are_listed_redacted() {
        let registry = SessionRegistry::new(4);
        let client_id = registry.next_client_id().unwrap();
        registry.insert(client_id.clone(), binding(1)).unwrap();

        let listed = registry.client_ids_redacted().unwrap();
        assert_eq!(listed.len(), 1);
        assert!(listed[0].starts_with("sess-"));
        assert!(listed[0].ends_with('…'));
        assert!(!listed[0].contains(&client_id[13..]));
        assert_eq!(registry.max_sessions(), 4);
    }

    /// Threads racing to delete one binding: exactly one removes it, and the
    /// conversation it closes is the one it removed.
    ///
    /// The property this pins is that a close never orphans: with get → acquire
    /// → remove, two racers could both read the same entry, one could remove and
    /// rebind it, and the other would then remove a binding it never leased.
    #[test]
    fn racing_deletes_remove_one_binding_exactly_once() {
        use std::sync::Barrier;
        use std::sync::atomic::{AtomicUsize, Ordering};

        const THREADS: usize = 8;
        let registry = SessionRegistry::new(4);
        registry.insert("sess-a".to_string(), binding(1)).unwrap();
        let barrier = Arc::new(Barrier::new(THREADS));
        let taken = Arc::new(AtomicUsize::new(0));
        let missing = Arc::new(AtomicUsize::new(0));

        std::thread::scope(|scope| {
            for _ in 0..THREADS {
                let registry = registry.clone();
                let barrier = Arc::clone(&barrier);
                let taken = Arc::clone(&taken);
                let missing = Arc::clone(&missing);
                scope.spawn(move || {
                    barrier.wait();
                    match registry.take_for_close("sess-a") {
                        Ok(guard) => {
                            assert_eq!(guard.binding(), &binding(1));
                            taken.fetch_add(1, Ordering::SeqCst);
                        }
                        Err(SessionCloseError::NotFound) => {
                            missing.fetch_add(1, Ordering::SeqCst);
                        }
                        Err(other) => panic!("unexpected close refusal: {other}"),
                    }
                });
            }
        });

        assert_eq!(taken.load(Ordering::SeqCst), 1, "one delete wins");
        assert_eq!(missing.load(Ordering::SeqCst), THREADS - 1);
        assert_eq!(registry.len(), 0);
        assert_eq!(registry.leases().held(), 0, "no lease leaked");
    }

    /// A delete racing a rebind of the same client id never orphans a
    /// conversation and never closes one twice.
    ///
    /// This is the race the old `get` → `acquire` → `remove` close could lose.
    /// Both callers read the same entry; one removes it and a third caller
    /// rebinds the id to a new conversation; the first then removes *that*
    /// binding — closing a conversation it never leased while leaving the one
    /// it did lease running with nothing naming it. Deciding all three steps
    /// under one lock makes the binding that is leased and the binding that is
    /// removed the same one, which is what the accounting below checks: across
    /// many rounds, every conversation is either still bound or was handed to
    /// exactly one closer, and never both.
    #[test]
    fn a_delete_racing_a_rebind_never_orphans_a_conversation() {
        use std::sync::Barrier;
        use std::sync::Mutex as StdMutex;

        const ROUNDS: usize = 64;
        for round in 0..ROUNDS {
            let registry = SessionRegistry::new(4);
            let original = binding(1);
            let replacement = binding(2);
            registry
                .insert("sess-x".to_string(), original.clone())
                .unwrap();

            let barrier = Arc::new(Barrier::new(2));
            let closed: Arc<StdMutex<Vec<ModelSessionPlacement>>> =
                Arc::new(StdMutex::new(Vec::new()));

            std::thread::scope(|scope| {
                {
                    let registry = registry.clone();
                    let barrier = Arc::clone(&barrier);
                    let closed = Arc::clone(&closed);
                    scope.spawn(move || {
                        barrier.wait();
                        if let Ok(guard) = registry.take_for_close("sess-x") {
                            closed
                                .lock()
                                .expect("closed list")
                                .push(guard.binding().clone());
                        }
                    });
                }
                {
                    let registry = registry.clone();
                    let barrier = Arc::clone(&barrier);
                    let closed = Arc::clone(&closed);
                    let replacement = replacement.clone();
                    scope.spawn(move || {
                        barrier.wait();
                        match registry.claim("sess-x".to_string(), replacement.clone()) {
                            // The id was already bound, so this caller's own
                            // freshly opened conversation is the one it closes.
                            Ok(SessionClaim::Existing(_)) => {
                                closed.lock().expect("closed list").push(replacement)
                            }
                            Ok(SessionClaim::Claimed { evicted }) => {
                                assert!(evicted.is_none(), "the registry was not full");
                            }
                            Err(error) => panic!("unexpected claim refusal: {error}"),
                        }
                    });
                }
            });

            let closed = closed.lock().expect("closed list").clone();
            let bound = registry.get("sess-x").unwrap();
            // Every conversation that was ever opened is accounted for exactly
            // once: closed, or still bound — never neither, never both.
            let mut accounted = closed.clone();
            accounted.extend(bound.clone());
            assert_eq!(
                accounted.len(),
                2,
                "round {round}: closed {closed:?}, bound {bound:?}",
            );
            assert!(accounted.contains(&original), "round {round}");
            assert!(accounted.contains(&replacement), "round {round}");
            assert_eq!(
                registry.leases().held(),
                0,
                "round {round}: every guard released its lease",
            );
        }
    }

    /// Threads racing to bind fresh ids against a bound of one never leave the
    /// registry over its bound, and never leave a lease behind.
    #[test]
    fn racing_inserts_never_exceed_the_bound() {
        use std::sync::Barrier;
        use std::sync::atomic::{AtomicUsize, Ordering};

        const THREADS: usize = 8;
        let registry = SessionRegistry::new(1);
        let barrier = Arc::new(Barrier::new(THREADS));
        let bound = Arc::new(AtomicUsize::new(0));

        std::thread::scope(|scope| {
            for index in 0..THREADS {
                let registry = registry.clone();
                let barrier = Arc::clone(&barrier);
                let bound = Arc::clone(&bound);
                scope.spawn(move || {
                    barrier.wait();
                    if let Ok(evicted) =
                        registry.insert(format!("sess-{index}"), binding(index as u64))
                    {
                        bound.fetch_add(1, Ordering::SeqCst);
                        drop(evicted);
                    }
                });
            }
        });

        assert!(bound.load(Ordering::SeqCst) >= 1, "at least one bound");
        assert_eq!(registry.len(), 1, "the bound holds under contention");
        assert_eq!(registry.leases().held(), 0, "no eviction lease leaked");
    }

    /// The `active_sessions` gauge counts conversations, so it must track what
    /// the map does rather than how many times a route asked for a session.
    ///
    /// These read a counter of their own rather than the process-global gauge
    /// (see [`SessionGauge`]): the rest of this binary opens and closes sessions
    /// while they run, so an exact assertion against the global counter would be
    /// a race. The arithmetic they exercise is the same arithmetic production
    /// runs — the destination is the only difference — and
    /// [`the_registry_reports_its_size_to_the_process_global_gauge`] pins that
    /// the production destination is still wired up.
    mod gauge {
        use super::*;
        use std::sync::atomic::{AtomicI64, Ordering};

        fn counter() -> Arc<AtomicI64> {
            Arc::new(AtomicI64::new(0))
        }

        fn live(counter: &Arc<AtomicI64>) -> i64 {
            counter.load(Ordering::SeqCst)
        }

        /// Churn at the bound is a replacement, not growth.
        ///
        /// The regression this pins had `insert` increment unconditionally while
        /// eviction removed its victim silently, so every round of LRU churn
        /// added one to a gauge whose map never grew. Sixty-four rounds make the
        /// difference between "off by a constant" and "climbs forever"
        /// unmistakable: the buggy arithmetic reports sixty-five conversations
        /// on a registry holding one.
        #[test]
        fn evicting_to_make_room_leaves_the_count_where_it_was() {
            const ROUNDS: u64 = 64;
            let counter = counter();
            let registry = SessionRegistry::with_local_gauge(1, &counter);

            registry
                .insert("sess-a".to_string(), binding(1))
                .expect("first insert fits under the bound");
            assert_eq!(live(&counter), 1, "the first conversation is counted once");

            for round in 0..ROUNDS {
                let evicted = registry
                    .insert(format!("sess-{round}"), binding(round + 2))
                    .expect("an idle victim is always available");
                assert!(evicted.is_some(), "round {round}: somebody was evicted");
                drop(evicted);

                assert_eq!(
                    registry.len(),
                    1,
                    "round {round}: the registry still holds one conversation",
                );
                assert_eq!(
                    live(&counter),
                    1,
                    "round {round}: an eviction plus an insertion is a replacement",
                );
            }
        }

        /// The same, through `claim`, which makes the same eviction decision.
        #[test]
        fn claiming_over_a_full_registry_leaves_the_count_where_it_was() {
            const ROUNDS: u64 = 64;
            let counter = counter();
            let registry = SessionRegistry::with_local_gauge(1, &counter);

            for round in 0..ROUNDS {
                let claim = registry
                    .claim(format!("sess-{round}"), binding(round))
                    .expect("an idle victim is always available");
                match claim {
                    SessionClaim::Claimed { evicted } => drop(evicted),
                    SessionClaim::Existing(_) => panic!("round {round}: id is fresh"),
                }
                assert_eq!(registry.len(), 1, "round {round}: one conversation");
                assert_eq!(live(&counter), 1, "round {round}: one counted");
            }
        }

        /// Below the bound there is nothing to replace, so an insertion is
        /// growth and is counted as growth.
        #[test]
        fn an_insertion_that_evicts_nobody_counts_a_new_conversation() {
            let counter = counter();
            let registry = SessionRegistry::with_local_gauge(4, &counter);

            for index in 0..4_u64 {
                registry
                    .insert(format!("sess-{index}"), binding(index))
                    .expect("under the bound");
                let bound = i64::from(u32::try_from(index).unwrap()) + 1;
                assert_eq!(
                    live(&counter),
                    bound,
                    "{bound} conversations, {bound} counted"
                );
                assert_eq!(registry.len(), usize::try_from(bound).unwrap());
            }
        }

        /// A refusal mutates nothing, so it reports nothing.
        #[test]
        fn refusing_at_capacity_leaves_the_count_untouched() {
            let counter = counter();
            let registry = SessionRegistry::with_local_gauge(1, &counter);
            registry
                .insert("sess-busy".to_string(), binding(1))
                .expect("first insert fits");
            let busy = registry
                .acquire(binding(1), "sess-busy")
                .expect("the only conversation takes its lease");

            let refused = registry.insert("sess-new".to_string(), binding(2));
            assert!(
                matches!(refused, Err(SessionRegistryError::AtCapacity { bound: 1 })),
                "a full registry of busy conversations refuses, {refused:?}",
            );
            assert_eq!(registry.len(), 1, "the refusal bound nothing");
            assert_eq!(live(&counter), 1, "the refusal counted nothing");

            let refused = registry.claim("sess-new".to_string(), binding(2));
            assert!(
                matches!(refused, Err(SessionRegistryError::AtCapacity { bound: 1 })),
                "claim refuses on the same terms, {refused:?}",
            );
            assert_eq!(live(&counter), 1, "the refused claim counted nothing");

            // And the refusal is transient: releasing the turn makes the next
            // insertion a replacement rather than growth.
            drop(busy);
            let evicted = registry
                .insert("sess-new".to_string(), binding(2))
                .expect("the freed conversation is now evictable");
            assert!(evicted.is_some(), "the idle conversation was the victim");
            drop(evicted);
            assert_eq!(registry.len(), 1, "still one conversation");
            assert_eq!(live(&counter), 1, "still one counted");
        }

        /// A rebind of an id already bound is refused, so it cannot be counted
        /// twice — and if `bind` is ever reached with a bound id anyway, the
        /// `HashMap` displacement is reported as the replacement it is.
        #[test]
        fn binding_over_an_existing_id_is_refused_and_counts_nothing() {
            let counter = counter();
            let registry = SessionRegistry::with_local_gauge(4, &counter);
            registry
                .insert("sess-a".to_string(), binding(1))
                .expect("first insert");

            let again = registry.insert("sess-a".to_string(), binding(2));
            assert!(
                matches!(again, Err(SessionRegistryError::AlreadyBound)),
                "an id that is already bound is refused, {again:?}",
            );
            assert_eq!(live(&counter), 1, "the refusal counted nothing");

            // The mutation site is what reports, so a displacement reported
            // through it is a replacement even if a caller ever reaches it.
            {
                let mut inner = registry.lock().expect("registry lock");
                inner.bind("sess-a".to_string(), binding(3), &registry.gauge);
            }
            assert_eq!(live(&counter), 1, "a displacement is not an addition");
            assert_eq!(registry.len(), 1, "and the map did not grow either");
        }

        /// An explicit close decrements exactly once, and only for a binding it
        /// actually removed.
        #[test]
        fn taking_a_session_for_close_decrements_once() {
            let counter = counter();
            let registry = SessionRegistry::with_local_gauge(4, &counter);
            registry
                .insert("sess-a".to_string(), binding(1))
                .expect("first insert");
            registry
                .insert("sess-b".to_string(), binding(2))
                .expect("second insert");
            assert_eq!(live(&counter), 2, "two conversations");

            let closed = registry
                .take_for_close("sess-a")
                .expect("idle, so closable");
            assert_eq!(live(&counter), 1, "the close counted exactly one departure");
            assert_eq!(registry.len(), 1);
            drop(closed);

            // The id is gone, so a second close removes nothing and reports
            // nothing — the double-decrement this arrangement has to exclude.
            let again = registry.take_for_close("sess-a");
            assert!(
                matches!(again, Err(SessionCloseError::NotFound)),
                "closing a closed session is not found, {again:?}",
            );
            assert_eq!(live(&counter), 1, "and it did not decrement again");

            // A refused close is not a close either.
            let busy = registry
                .acquire(binding(2), "sess-b")
                .expect("sess-b takes its lease");
            let refused = registry.take_for_close("sess-b");
            assert!(
                matches!(refused, Err(SessionCloseError::Busy(_))),
                "a session mid-turn is not closable, {refused:?}",
            );
            assert_eq!(live(&counter), 1, "a refused close counted nothing");
            assert_eq!(registry.len(), 1, "and unbound nothing");
            drop(busy);
        }

        /// Dropping the registry returns the count to where it started, so a
        /// test that opens sessions does not leave the gauge raised forever.
        #[test]
        fn dropping_the_registry_returns_the_count_to_zero() {
            let counter = counter();
            {
                let registry = SessionRegistry::with_local_gauge(4, &counter);
                for index in 0..3_u64 {
                    registry
                        .insert(format!("sess-{index}"), binding(index))
                        .expect("under the bound");
                }
                assert_eq!(live(&counter), 3, "three conversations");
            }
            assert_eq!(live(&counter), 0, "the registry took its count with it");
        }

        /// Churn under contention does not accumulate either. Eight threads
        /// race to bind fresh ids against a bound of one; whatever order they
        /// land in, the count must equal the number of bindings left.
        #[test]
        fn racing_churn_leaves_the_count_equal_to_the_map() {
            use std::sync::Barrier;

            const THREADS: usize = 8;
            const ROUNDS: u64 = 16;
            let counter = counter();
            let registry = SessionRegistry::with_local_gauge(1, &counter);
            let barrier = Arc::new(Barrier::new(THREADS));

            std::thread::scope(|scope| {
                for thread in 0..THREADS {
                    let registry = registry.clone();
                    let barrier = Arc::clone(&barrier);
                    scope.spawn(move || {
                        barrier.wait();
                        for round in 0..ROUNDS {
                            let id = format!("sess-{thread}-{round}");
                            if let Ok(evicted) = registry.insert(id, binding(round)) {
                                drop(evicted);
                            }
                        }
                    });
                }
            });

            assert_eq!(registry.len(), 1, "the bound held");
            assert_eq!(
                live(&counter),
                1,
                "{} rounds of churn counted one conversation, not the churn",
                THREADS as u64 * ROUNDS,
            );
            assert_eq!(registry.leases().held(), 0, "no eviction lease leaked");
        }
    }

    /// The production registry reports to the process-global gauge.
    ///
    /// [`mod gauge`] asserts the arithmetic against a counter of its own; this
    /// asserts the wire is still attached, which is the one thing a local
    /// counter cannot see. It measures a delta rather than an absolute, because
    /// the rest of this binary is moving the same gauge, and it asserts only a
    /// lower bound on that delta for the same reason — anything exact would be
    /// asserting that no other test opened a session at the same instant.
    #[test]
    fn the_registry_reports_its_size_to_the_process_global_gauge() {
        let before = crate::metrics::snapshot().active_sessions;
        let registry = SessionRegistry::new(4);
        registry
            .insert("sess-global".to_string(), binding(1))
            .expect("under the bound");
        let after = crate::metrics::snapshot().active_sessions;
        assert!(
            after > before,
            "binding a conversation must reach the global gauge ({before} -> {after})",
        );
        drop(registry);
    }
}

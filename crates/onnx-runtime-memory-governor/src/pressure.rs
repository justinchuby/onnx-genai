//! Authority/tier-scoped synchronous pressure rounds shared by lease admission
//! and process allocation transactions. This does not replace ticket arbitration.
use crate::{MemoryAuthorityId, PressureResponder};
use onnx_runtime_memory_api::{MemoryError, Tier};
use std::cell::RefCell;
use std::sync::{Arc, Mutex, Weak};

thread_local! {
    static ACTIVE: RefCell<Vec<(MemoryAuthorityId, Tier)>> = const { RefCell::new(Vec::new()) };
}

struct ActiveRound((MemoryAuthorityId, Tier));
impl ActiveRound {
    fn enter(key: (MemoryAuthorityId, Tier)) -> Option<Self> {
        ACTIVE
            .try_with(|active| {
                let mut active = active.borrow_mut();
                if active.contains(&key) {
                    return None;
                }
                active.push(key);
                Some(Self(key))
            })
            .ok()
            .flatten()
    }
}
impl Drop for ActiveRound {
    fn drop(&mut self) {
        let _ = ACTIVE.try_with(|active| {
            let finished = active.borrow_mut().pop();
            debug_assert_eq!(finished, Some(self.0));
        });
    }
}

#[derive(Default, Debug)]
struct State {
    next_id: u64,
    first: usize,
    entries: Vec<(u64, Weak<dyn PressureResponder>)>,
}

/// Explicit pressure capability for one physical authority and one tier.
/// Cloning this handle does not create another registry or budget.
#[derive(Clone, Debug)]
pub struct PressureRegistry {
    authority: MemoryAuthorityId,
    tier: Tier,
    state: Arc<Mutex<State>>,
}

/// Keeps a weak callback enrolled. Drop removes the enrollment; it keeps neither
/// the callback nor the authority alive. An already snapshotted round may finish.
#[derive(Debug)]
pub struct PressureRegistration {
    state: Weak<Mutex<State>>,
    id: u64,
}
impl Drop for PressureRegistration {
    fn drop(&mut self) {
        if let Some(state) = self.state.upgrade() {
            state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .entries
                .retain(|(id, _)| *id != self.id);
        }
    }
}

pub(crate) struct PressureRound {
    pub(crate) responders: Vec<Arc<dyn PressureResponder>>,
    _active: ActiveRound,
}
impl PressureRegistry {
    /// Construct a registry owned by the supplied authority/tier's accounting.
    /// The governor must return this same registry for its entire lifetime.
    pub fn new(authority: MemoryAuthorityId, tier: Tier) -> Self {
        Self {
            authority,
            tier,
            state: Arc::new(Mutex::new(State::default())),
        }
    }
    pub fn authority_id(&self) -> MemoryAuthorityId {
        self.authority
    }
    pub fn tier(&self) -> Tier {
        self.tier
    }
    /// Reject a capability wired to different physical books or a different tier.
    pub fn validate(&self, authority: MemoryAuthorityId, tier: Tier) -> Result<(), MemoryError> {
        if self.authority != authority || self.tier != tier {
            return Err(MemoryError::InvalidRequest {
                tier: tier.name(),
                requested: 0,
                reason: "pressure registry belongs to different authority/tier books; return the owning tier's shared registry",
            });
        }
        Ok(())
    }
    pub fn register(
        &self,
        responder: &Arc<dyn PressureResponder>,
    ) -> Result<PressureRegistration, MemoryError> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let id = state.next_id.checked_add(1).ok_or(MemoryError::InvalidRequest {
            tier: self.tier.name(), requested: 0,
            reason: "pressure registration identity exhausted; recreate the owning memory authority",
        })?;
        state.entries.retain(|(_, entry)| entry.strong_count() != 0);
        state.next_id = id;
        state.entries.push((id, Arc::downgrade(responder)));
        Ok(PressureRegistration {
            state: Arc::downgrade(&self.state),
            id,
        })
    }
    pub(crate) fn round(&self) -> Option<PressureRound> {
        let active = ActiveRound::enter((self.authority, self.tier))?;
        let responders = {
            let mut state = self
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.entries.retain(|(_, entry)| entry.strong_count() != 0);
            let mut responders: Vec<_> = state
                .entries
                .iter()
                .filter_map(|(_, entry)| entry.upgrade())
                .collect();
            if !responders.is_empty() {
                let first = state.first % responders.len();
                responders.rotate_left(first);
                state.first = (first + 1) % responders.len();
            }
            responders
        };
        Some(PressureRound {
            responders,
            _active: active,
        })
    }
    /// Ask idle holders for capacity outside every registry/accounting lock.
    /// The report is advisory: callers must still obtain an actual grant.
    pub fn reclaim(&self, want: u64) -> Result<u64, MemoryError> {
        if want == 0 {
            return Ok(0);
        }
        let Some(round) = self.round() else {
            return Ok(0);
        };
        let mut released = 0u64;
        for responder in &round.responders {
            released = released
                .saturating_add(responder.on_pressure(self.tier, want.saturating_sub(released))?);
            if released >= want {
                break;
            }
        }
        Ok(released)
    }
}

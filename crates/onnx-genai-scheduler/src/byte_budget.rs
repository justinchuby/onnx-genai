//! Global cross-session KV byte budget (DESIGN.md §26.4 / §26.11).
//!
//! The per-scheduler `max_total_tokens` gate (see [`crate::SchedulerConfig`])
//! bounds one scheduler instance in *tokens*. It cannot express the machine-level
//! reality a user actually cares about: *"do not use more than N bytes of
//! accelerator KV memory across every session and model on this device"*.
//!
//! [`ByteBudget`] is that missing piece — a small, thread-safe, cloneable
//! accounting primitive that tracks live KV **bytes** against a dynamic ceiling
//! and is **shared across sessions/models** (clone the handle; every clone
//! observes the same running total). Bytes are authoritative and pages/tokens
//! are derived (DESIGN.md §26.11), so this type speaks only in bytes and stays
//! model-, vendor-, and EP-agnostic (RULES.md #2): the caller converts its own
//! token footprint into bytes via a per-model `bytes_per_token` cost.
//!
//! The ceiling is reconfigurable live ([`ByteBudget::reconfigure`]) so a governor
//! can turn the limit down while another workload needs the device and back up
//! afterwards, exactly as DESIGN.md §26.11.2 describes.

use std::sync::{Arc, Mutex};

/// Over-budget rejection for a byte reservation.
///
/// Carries the full what/why/how contract (RULES.md #1): the caller sees exactly
/// how many bytes it asked for, how many are already in use, the ceiling, and the
/// concrete headroom it must free or raise to succeed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error(
    "KV byte budget exceeded: requested {requested} B but only {available} B free \
     (used {used} B of {limit} B limit); free at least {shortfall} B by preempting a \
     session or raise the budget with ByteBudget::reconfigure"
)]
pub struct ByteBudgetError {
    /// Bytes the rejected reservation asked for.
    pub requested: u64,
    /// Bytes already reserved across all sessions when the request was rejected.
    pub used: u64,
    /// The active byte ceiling.
    pub limit: u64,
    /// Bytes free at rejection time (`limit - used`).
    pub available: u64,
    /// Bytes that must be freed (or added to the limit) to admit the request.
    pub shortfall: u64,
}

/// Immutable view of the budget at a point in time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BudgetSnapshot {
    /// Active byte ceiling.
    pub limit: u64,
    /// Bytes reserved across all sessions.
    pub used: u64,
    /// Bytes free (`limit.saturating_sub(used)`).
    pub available: u64,
}

/// Result of a live [`ByteBudget::reconfigure`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReconfigureOutcome {
    /// Ceiling before the change.
    pub old_limit: u64,
    /// Ceiling after the change.
    pub new_limit: u64,
    /// Bytes reserved at reconfigure time (unchanged by this call).
    pub used: u64,
    /// Bytes by which live usage exceeds the new ceiling, if the limit was
    /// lowered below current usage. Non-zero means the caller must drive its
    /// eviction tiers (DESIGN.md §26.11.2) until this many bytes are released.
    pub overage: u64,
}

#[derive(Debug)]
struct BudgetState {
    limit: u64,
    used: u64,
}

/// A shared, dynamic, cross-session KV byte budget.
///
/// Clone to share the *same* budget across multiple [`crate::Scheduler`]
/// instances (one per session/model). All clones account against a single
/// running total, so no single session can blow the global ceiling.
#[derive(Debug, Clone)]
pub struct ByteBudget {
    state: Arc<Mutex<BudgetState>>,
}

impl ByteBudget {
    /// Create a budget with an absolute byte `limit`.
    pub fn new(limit_bytes: u64) -> Self {
        Self {
            state: Arc::new(Mutex::new(BudgetState {
                limit: limit_bytes,
                used: 0,
            })),
        }
    }

    /// Try to reserve `bytes` against the shared budget.
    ///
    /// On success the running total grows and the caller owns those bytes until
    /// it calls [`ByteBudget::release`] with the same amount. On failure nothing
    /// changes and the caller learns the exact shortfall.
    pub fn try_reserve(&self, bytes: u64) -> Result<(), ByteBudgetError> {
        let mut state = self.lock();
        let available = state.limit.saturating_sub(state.used);
        if bytes > available {
            return Err(ByteBudgetError {
                requested: bytes,
                used: state.used,
                limit: state.limit,
                available,
                shortfall: bytes - available,
            });
        }
        state.used += bytes;
        Ok(())
    }

    /// Release `bytes` previously reserved via [`ByteBudget::try_reserve`].
    ///
    /// Saturates at zero so a double release can never underflow the total.
    pub fn release(&self, bytes: u64) {
        let mut state = self.lock();
        state.used = state.used.saturating_sub(bytes);
    }

    /// Replace the ceiling live (DESIGN.md §26.11.2).
    ///
    /// Never evicts on its own — it reports how far usage now exceeds the ceiling
    /// via [`ReconfigureOutcome::overage`] so the caller can drive eviction. New
    /// reservations observe the tightened ceiling immediately.
    pub fn reconfigure(&self, new_limit_bytes: u64) -> ReconfigureOutcome {
        let mut state = self.lock();
        let old_limit = state.limit;
        state.limit = new_limit_bytes;
        ReconfigureOutcome {
            old_limit,
            new_limit: new_limit_bytes,
            used: state.used,
            overage: state.used.saturating_sub(new_limit_bytes),
        }
    }

    /// Bytes currently reserved across all sessions.
    pub fn used(&self) -> u64 {
        self.lock().used
    }

    /// The active byte ceiling.
    pub fn limit(&self) -> u64 {
        self.lock().limit
    }

    /// Bytes free (`limit - used`, saturating).
    pub fn available(&self) -> u64 {
        let state = self.lock();
        state.limit.saturating_sub(state.used)
    }

    /// Point-in-time view of limit/used/available.
    pub fn snapshot(&self) -> BudgetSnapshot {
        let state = self.lock();
        BudgetSnapshot {
            limit: state.limit,
            used: state.used,
            available: state.limit.saturating_sub(state.used),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, BudgetState> {
        // The critical sections are tiny and panic-free, so the mutex can only be
        // poisoned by a panic in unrelated code; recover the guard rather than
        // propagating a poison error onto the hot admission path.
        self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve_within_limit_then_release_restores_headroom() {
        let budget = ByteBudget::new(1000);
        budget.try_reserve(600).unwrap();
        assert_eq!(budget.used(), 600);
        assert_eq!(budget.available(), 400);
        budget.release(600);
        assert_eq!(budget.used(), 0);
        assert_eq!(budget.available(), 1000);
    }

    #[test]
    fn over_budget_reservation_is_rejected_with_actionable_shortfall() {
        let budget = ByteBudget::new(1000);
        budget.try_reserve(800).unwrap();
        let err = budget.try_reserve(500).unwrap_err();
        assert_eq!(
            err,
            ByteBudgetError {
                requested: 500,
                used: 800,
                limit: 1000,
                available: 200,
                shortfall: 300,
            }
        );
        // Rejection leaves the running total untouched.
        assert_eq!(budget.used(), 800);
        let text = err.to_string();
        assert!(text.contains("free at least 300 B"), "{text}");
        assert!(text.contains("reconfigure"), "{text}");
    }

    #[test]
    fn shared_handle_accounts_across_sessions() {
        let device_budget = ByteBudget::new(1000);
        let session_a = device_budget.clone();
        let session_b = device_budget.clone();

        session_a.try_reserve(700).unwrap();
        // Session B sees A's usage: only 300 B remain, so 400 is rejected.
        assert!(session_b.try_reserve(400).is_err());
        session_b.try_reserve(300).unwrap();
        assert_eq!(device_budget.used(), 1000);
        assert_eq!(device_budget.available(), 0);
    }

    #[test]
    fn reconfigure_lower_reports_overage_without_evicting() {
        let budget = ByteBudget::new(1000);
        budget.try_reserve(800).unwrap();
        let outcome = budget.reconfigure(500);
        assert_eq!(outcome.old_limit, 1000);
        assert_eq!(outcome.new_limit, 500);
        assert_eq!(outcome.used, 800);
        assert_eq!(outcome.overage, 300);
        // Usage is not touched, but new reservations see the tighter ceiling.
        assert_eq!(budget.used(), 800);
        assert!(budget.try_reserve(1).is_err());
    }

    #[test]
    fn reconfigure_raise_adds_headroom() {
        let budget = ByteBudget::new(1000);
        budget.try_reserve(900).unwrap();
        assert!(budget.try_reserve(200).is_err());
        let outcome = budget.reconfigure(2000);
        assert_eq!(outcome.overage, 0);
        budget.try_reserve(200).unwrap();
        assert_eq!(budget.used(), 1100);
    }
}

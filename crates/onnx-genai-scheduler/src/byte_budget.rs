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

/// Successful reservation after optionally reducing the requested amount to fit
/// the currently available budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteBudgetReservation {
    /// Bytes actually reserved.
    pub reserved: u64,
    /// Bytes free before this reservation was taken.
    pub available_before: u64,
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

/// A live ceiling on admission, read from whatever actually owns the memory.
///
/// # Why the configured limit is not enough
///
/// [`ByteBudget::new`] takes the KV budget derived at load: the device limit
/// less an estimate of everything that is not KV. That is a *snapshot*, and the
/// things it subtracted are not constants. Weight residency grows when a model
/// touches more experts than its budget assumed; recurrent state is charged when
/// a hybrid model loads; a third-party execution provider may lease for reasons
/// this crate cannot enumerate. Every one of those shrinks the memory available
/// for KV, and none of them is visible to a number computed once at load.
///
/// The failure that produces is not a rejected request, which would be fine. It
/// is admission succeeding against a ceiling that no longer exists, and the
/// allocation failing later, further from the decision that caused it.
///
/// So the ceiling is a question asked at admission time rather than an answer
/// remembered from load. Implement this over the memory governor to have
/// admission and allocation consult one book.
///
/// # Contract
///
/// `ceiling_bytes` returns the total bytes admission may reserve *including*
/// what this budget has already reserved -- an absolute ceiling, not headroom.
/// The effective limit is the smaller of this and the configured limit, so an
/// implementation can only ever tighten what the operator configured.
///
/// It is called on the admission path, so it must not block. It is called
/// without any of this budget's locks held, so it may take its own.
pub trait AdmissionCeiling: Send + Sync + std::fmt::Debug {
    /// Total bytes admission may hold right now.
    fn ceiling_bytes(&self) -> u64;
}

/// A shared, dynamic, cross-session KV byte budget.
///
/// Clone to share the *same* budget across multiple [`crate::Scheduler`]
/// instances (one per session/model). All clones account against a single
/// running total, so no single session can blow the global ceiling.
#[derive(Debug, Clone)]
pub struct ByteBudget {
    state: Arc<Mutex<BudgetState>>,
    /// Tightens `state.limit` to what the memory governor says is actually
    /// there. `None` keeps the configured limit as the only ceiling, which is
    /// what a standalone scheduler with no governor wants.
    ceiling: Option<Arc<dyn AdmissionCeiling>>,
}

impl ByteBudget {
    /// Create a budget with an absolute byte `limit`.
    pub fn new(limit_bytes: u64) -> Self {
        Self {
            state: Arc::new(Mutex::new(BudgetState {
                limit: limit_bytes,
                used: 0,
            })),
            ceiling: None,
        }
    }

    /// The same budget, additionally bounded by a live [`AdmissionCeiling`].
    ///
    /// Clones share the ceiling, because they share the running total: a budget
    /// whose clones disagreed about the limit would not be one budget.
    #[must_use]
    pub fn with_ceiling(mut self, ceiling: Arc<dyn AdmissionCeiling>) -> Self {
        self.ceiling = Some(ceiling);
        self
    }

    /// The governor's ceiling, if one is attached.
    ///
    /// Read *before* taking the budget lock so the admission critical section
    /// stays free of foreign code, and so an implementation is free to take its
    /// own locks without ordering against this one.
    fn live_ceiling(&self) -> Option<u64> {
        self.ceiling.as_ref().map(|ceiling| ceiling.ceiling_bytes())
    }

    /// Try to reserve `bytes` against the shared budget.
    ///
    /// On success the running total grows and the caller owns those bytes until
    /// it calls [`ByteBudget::release`] with the same amount. On failure nothing
    /// changes and the caller learns the exact shortfall.
    pub fn try_reserve(&self, bytes: u64) -> Result<(), ByteBudgetError> {
        let live = self.live_ceiling();
        let mut state = self.lock();
        let limit = live.map_or(state.limit, |live| state.limit.min(live));
        let available = limit.saturating_sub(state.used);
        if bytes > available {
            return Err(ByteBudgetError {
                requested: bytes,
                used: state.used,
                limit,
                available,
                shortfall: bytes - available,
            });
        }
        state.used += bytes;
        Ok(())
    }

    /// Reserve `requested` bytes, or the largest aligned amount that fits.
    ///
    /// The fit calculation and reservation happen while holding the same budget
    /// lock, so a shared-budget peer cannot consume the computed headroom before
    /// this reservation is recorded. `minimum` is the smallest useful reservation
    /// and `unit` is the alignment quantum for partial reservations.
    pub fn try_reserve_at_most(
        &self,
        requested: u64,
        minimum: u64,
        unit: u64,
    ) -> Result<ByteBudgetReservation, ByteBudgetError> {
        let live = self.live_ceiling();
        let mut state = self.lock();
        let limit = live.map_or(state.limit, |live| state.limit.min(live));
        let available = limit.saturating_sub(state.used);
        let reserved = if requested <= available {
            requested
        } else if unit == 0 {
            0
        } else {
            available - (available % unit)
        };
        if reserved < minimum {
            return Err(ByteBudgetError {
                requested,
                used: state.used,
                limit,
                available,
                shortfall: minimum.saturating_sub(available),
            });
        }
        state.used += reserved;
        Ok(ByteBudgetReservation {
            reserved,
            available_before: available,
        })
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
        let live = self.live_ceiling();
        let mut state = self.lock();
        let old_limit = state.limit;
        state.limit = new_limit_bytes;
        // Overage is what the caller must evict, so it is measured against the
        // ceiling that actually binds -- a raised configured limit frees
        // nothing if the governor is the tighter of the two.
        let effective = live.map_or(new_limit_bytes, |live| new_limit_bytes.min(live));
        ReconfigureOutcome {
            old_limit,
            new_limit: new_limit_bytes,
            used: state.used,
            overage: state.used.saturating_sub(effective),
        }
    }

    /// Bytes currently reserved across all sessions.
    pub fn used(&self) -> u64 {
        self.lock().used
    }

    /// The byte ceiling in force: the configured limit, tightened by the
    /// [`AdmissionCeiling`] if one is attached.
    pub fn limit(&self) -> u64 {
        let live = self.live_ceiling();
        let state = self.lock();
        live.map_or(state.limit, |live| state.limit.min(live))
    }

    /// Bytes free (`limit - used`, saturating).
    pub fn available(&self) -> u64 {
        let live = self.live_ceiling();
        let state = self.lock();
        let limit = live.map_or(state.limit, |live| state.limit.min(live));
        limit.saturating_sub(state.used)
    }

    /// Point-in-time view of limit/used/available.
    pub fn snapshot(&self) -> BudgetSnapshot {
        let live = self.live_ceiling();
        let state = self.lock();
        let limit = live.map_or(state.limit, |live| state.limit.min(live));
        BudgetSnapshot {
            limit,
            used: state.used,
            available: limit.saturating_sub(state.used),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, BudgetState> {
        // The critical sections are tiny and panic-free, so the mutex can only be
        // poisoned by a panic in unrelated code; recover the guard rather than
        // propagating a poison error onto the hot admission path.
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A ceiling that reports whatever it is told, standing in for the ledger.
    #[derive(Debug)]
    struct FixedCeiling(std::sync::atomic::AtomicU64);

    impl AdmissionCeiling for FixedCeiling {
        fn ceiling_bytes(&self) -> u64 {
            self.0.load(std::sync::atomic::Ordering::Relaxed)
        }
    }

    fn ceiling(bytes: u64) -> Arc<FixedCeiling> {
        Arc::new(FixedCeiling(std::sync::atomic::AtomicU64::new(bytes)))
    }

    #[test]
    fn a_governor_ceiling_tightens_admission_below_the_configured_limit() {
        let budget = ByteBudget::new(1000).with_ceiling(ceiling(400));

        assert_eq!(budget.limit(), 400, "the tighter of the two must bind");
        budget.try_reserve(400).unwrap();
        let error = budget.try_reserve(1).unwrap_err();
        assert_eq!(error.limit, 400);
        assert_eq!(error.shortfall, 1);
    }

    #[test]
    fn a_configured_limit_below_the_ceiling_still_binds() {
        // The ceiling only ever tightens: an operator who asked for 300 B of KV
        // does not get 900 just because the device happens to be empty.
        let budget = ByteBudget::new(300).with_ceiling(ceiling(900));

        assert_eq!(budget.limit(), 300);
        assert!(budget.try_reserve(301).is_err());
    }

    #[test]
    fn admission_shrinks_when_another_holder_takes_device_memory() {
        // The regression this exists for: residency grows after load, the
        // ledger knows, and admission kept saying yes against the ceiling it
        // was seeded with. Now the next reservation sees the smaller number.
        let live = ceiling(1000);
        let budget = ByteBudget::new(1000).with_ceiling(live.clone());
        budget.try_reserve(600).unwrap();
        assert_eq!(budget.available(), 400);

        // Another holder leases 700 B, leaving 300 B for KV.
        live.0.store(300, std::sync::atomic::Ordering::Relaxed);

        assert_eq!(budget.limit(), 300);
        assert_eq!(
            budget.available(),
            0,
            "already over the new ceiling, so nothing more is admissible"
        );
        assert!(budget.try_reserve(1).is_err());
    }

    #[test]
    fn reconfigure_reports_overage_against_the_ceiling_that_binds() {
        // Raising the configured limit frees nothing while the governor is the
        // tighter of the two, so the caller must still be told to evict.
        let budget = ByteBudget::new(1000).with_ceiling(ceiling(400));
        budget.try_reserve(400).unwrap();

        let outcome = budget.reconfigure(2000);

        assert_eq!(outcome.new_limit, 2000, "the configured limit did change");
        assert_eq!(outcome.overage, 0, "400 used is within the 400 B ceiling");

        let outcome = budget.reconfigure(100);
        assert_eq!(outcome.overage, 300, "the configured limit now binds");
    }

    #[test]
    fn an_ungoverned_budget_is_unchanged() {
        let budget = ByteBudget::new(1000);
        assert_eq!(budget.limit(), 1000);
        budget.try_reserve(1000).unwrap();
        assert_eq!(budget.available(), 0);
    }

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
    fn reserve_at_most_caps_and_reserves_under_one_lock() {
        let budget = ByteBudget::new(1000);
        budget.try_reserve(450).unwrap();

        let reservation = budget.try_reserve_at_most(800, 200, 100).unwrap();

        assert_eq!(
            reservation,
            ByteBudgetReservation {
                reserved: 500,
                available_before: 550,
            }
        );
        assert_eq!(budget.used(), 950);

        let err = budget.try_reserve_at_most(800, 200, 100).unwrap_err();
        assert_eq!(err.available, 50);
        assert_eq!(err.shortfall, 150);
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

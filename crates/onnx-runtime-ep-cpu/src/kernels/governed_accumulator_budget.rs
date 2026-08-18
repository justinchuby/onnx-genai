//! A process-wide budget for a per-thread scratch buffer that is parked between
//! calls.
//!
//! # Why this type exists
//!
//! A kernel that reuses a `thread_local!` scratch buffer between calls -- to
//! avoid re-faulting a fresh mapping and paying a TLB shootdown on free every
//! call -- retains one buffer *per worker thread that ever ran it*, for the life
//! of the process. Bounding that buffer with a **per-buffer** constant is silent
//! about the multiplier: the real process-wide exposure is `per_thread_cap x
//! threads`, and the thread count is exactly the variable the reuse optimisation
//! scales with. A 32 MiB per-thread cap is 1 GiB on a 32-vCPU box and 4 GiB on a
//! 128-vCPU server -- none of which appears in a comment that reasons about one
//! buffer.
//!
//! This is the same shape as #1051 (an owned copy correct about one instance,
//! silent about the shape-keyed second) and #1100 (a ratio test that drove a
//! single instantiation and so structurally could not observe the x2). The fix
//! is to bound the *process*, not the thread.
//!
//! # The contract (mirrors #1056 / [`super::governed_weight_cache`])
//!
//! - **Declared before allocated / declinable.** The budget carries an
//!   admit/decline flag set by the memory plan. When declined, [`Self::try_park`]
//!   refuses every buffer, so the kernel keeps nothing and recomputes per call
//!   (byte-identical, only slower) -- exactly the `GovernedWeightCache` decline
//!   contract, applied to a scratch buffer instead of a weight-derived one.
//! - **Bytes, not entries.** [`Self::live_bytes`] reports the sum actually
//!   parked across all threads, so a wrong ceiling is detectable in one run
//!   rather than argued from a formula. A single-thread test cannot move this
//!   figure past one buffer, which is why the test that defends it drives
//!   multiple threads.
//! - **Two ceilings, both stated.** A per-thread cap still releases any single
//!   buffer larger than it (the surviving-case behaviour that was already
//!   correct), and a hard process cap bounds the *sum* regardless of how many
//!   threads run the kernel. The process-wide ceiling is therefore
//!   `min(process_cap, per_thread_cap x threads)` -- a flat number that does not
//!   grow with the vCPU count.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Process-wide accounting for a per-thread parked scratch buffer.
///
/// A single instance is meant to live in a `static` and be shared by every
/// thread that parks a buffer. `retained_bytes` is the figure the per-thread
/// constant was silent about: the sum currently held across all threads.
pub struct GovernedAccumulatorBudget {
    /// The memory plan's verdict. `false` makes [`Self::try_park`] refuse every
    /// buffer, so the kernel retains nothing and recomputes per call.
    admitted: AtomicBool,
    /// Bytes parked across *all* threads right now. This is what
    /// [`Self::live_bytes`] reports and what the process cap bounds.
    retained_bytes: AtomicU64,
    /// Largest single buffer a thread may park. A buffer over this is released
    /// rather than parked -- at that size the GEMM dwarfs the allocation, so
    /// parking it would trade real memory for nothing. This is the
    /// already-correct surviving-case behaviour, kept intact.
    per_thread_cap_bytes: AtomicU64,
    /// Hard ceiling on `retained_bytes` regardless of thread count. This is the
    /// number the old per-thread-only bound was missing.
    process_cap_bytes: AtomicU64,
}

impl GovernedAccumulatorBudget {
    /// Create a budget. `const` so a shared instance can live in a `static`.
    /// Admitted by default; the memory plan lowers it at load with
    /// [`Self::set_admitted`] when the process footprint would not fit.
    pub const fn new(per_thread_cap_bytes: u64, process_cap_bytes: u64) -> Self {
        Self {
            admitted: AtomicBool::new(true),
            retained_bytes: AtomicU64::new(0),
            per_thread_cap_bytes: AtomicU64::new(per_thread_cap_bytes),
            process_cap_bytes: AtomicU64::new(process_cap_bytes),
        }
    }

    /// Record the memory plan's admit/decline decision.
    pub fn set_admitted(&self, admitted: bool) {
        self.admitted.store(admitted, Ordering::Relaxed);
    }

    pub fn is_admitted(&self) -> bool {
        self.admitted.load(Ordering::Relaxed)
    }

    /// Bytes currently parked across all threads -- the figure a predicted
    /// ceiling is checked against.
    pub fn live_bytes(&self) -> u64 {
        self.retained_bytes.load(Ordering::Relaxed)
    }

    pub fn per_thread_cap_bytes(&self) -> u64 {
        self.per_thread_cap_bytes.load(Ordering::Relaxed)
    }

    pub fn process_cap_bytes(&self) -> u64 {
        self.process_cap_bytes.load(Ordering::Relaxed)
    }

    /// Retune the process-wide ceiling at runtime. A deployment may raise or
    /// lower the transient-scratch budget; the RSS measurement harness uses it
    /// to contrast the process-bounded ceiling against the pre-fix per-thread-
    /// only behaviour (`u64::MAX`, i.e. unbounded) in separate processes.
    pub fn set_process_cap_bytes(&self, bytes: u64) {
        self.process_cap_bytes.store(bytes, Ordering::Relaxed);
    }

    /// Try to reserve `bytes` for a thread that wants to park its buffer.
    ///
    /// Succeeds (and adds `bytes` to the process-wide total) only when the
    /// budget is admitted, the single buffer is within the per-thread cap, and
    /// the new total stays within the process cap. On failure the caller must
    /// drop its buffer and recompute next call -- a pure performance tradeoff,
    /// never a numerical one.
    pub fn try_park(&self, bytes: u64) -> bool {
        if bytes == 0 || !self.is_admitted() {
            return false;
        }
        if bytes > self.per_thread_cap_bytes() {
            return false;
        }
        let cap = self.process_cap_bytes();
        let mut current = self.retained_bytes.load(Ordering::Relaxed);
        loop {
            let Some(next) = current.checked_add(bytes) else {
                return false;
            };
            if next > cap {
                return false;
            }
            match self.retained_bytes.compare_exchange_weak(
                current,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(observed) => current = observed,
            }
        }
    }

    /// Release `bytes` a thread had parked, because it is taking that buffer
    /// back out to reuse it (or dropping it). Must be paired with a prior
    /// successful [`Self::try_park`] of the same size. Saturating, so a stray
    /// release can never underflow the counter below zero.
    pub fn release(&self, bytes: u64) {
        if bytes == 0 {
            return;
        }
        let mut current = self.retained_bytes.load(Ordering::Relaxed);
        loop {
            let next = current.saturating_sub(bytes);
            match self.retained_bytes.compare_exchange_weak(
                current,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(observed) => current = observed,
            }
        }
    }

    /// Override both caps for a test that wants to observe the process bound
    /// without allocating hundreds of megabytes.
    #[cfg(test)]
    pub fn set_caps_for_test(&self, per_thread_cap_bytes: u64, process_cap_bytes: u64) {
        self.per_thread_cap_bytes
            .store(per_thread_cap_bytes, Ordering::Relaxed);
        self.process_cap_bytes
            .store(process_cap_bytes, Ordering::Relaxed);
    }

    /// Reset the parked-bytes counter for a test that runs in isolation.
    #[cfg(test)]
    pub fn reset_for_test(&self) {
        self.retained_bytes.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A declined budget parks nothing, so its live figure stays at zero no
    /// matter how many buffers ask to be parked.
    #[test]
    fn a_declined_budget_parks_nothing() {
        let budget = GovernedAccumulatorBudget::new(1 << 20, 1 << 20);
        budget.set_admitted(false);
        for _ in 0..8 {
            assert!(!budget.try_park(4096), "declined must refuse every buffer");
        }
        assert_eq!(budget.live_bytes(), 0);
    }

    /// The per-thread cap still releases any single buffer larger than it -- the
    /// surviving-case behaviour that was already correct and must not regress.
    #[test]
    fn a_buffer_over_the_per_thread_cap_is_refused() {
        let budget = GovernedAccumulatorBudget::new(4096, 1 << 30);
        assert!(budget.try_park(4096), "a buffer at the cap is parked");
        assert!(!budget.try_park(4097), "a buffer over the cap is released");
        assert_eq!(budget.live_bytes(), 4096);
    }

    /// The process cap bounds the *sum* across threads, which is the whole
    /// point: parking more buffers than fit refuses the overflow rather than
    /// letting the total scale with the thread count.
    #[test]
    fn the_process_cap_bounds_the_sum_not_one_buffer() {
        // Four buffers fit; the fifth would exceed the process cap.
        let budget = GovernedAccumulatorBudget::new(1000, 4000);
        for expected in 1..=4 {
            assert!(budget.try_park(1000), "buffer {expected} fits under the sum");
            assert_eq!(budget.live_bytes(), expected * 1000);
        }
        assert!(
            !budget.try_park(1000),
            "the fifth buffer exceeds the process cap and is refused"
        );
        assert_eq!(budget.live_bytes(), 4000, "the sum is bounded by the cap");
    }

    /// Releasing returns capacity to the pool so a later park can use it, and it
    /// saturates rather than underflowing on an over-release.
    #[test]
    fn release_returns_capacity_and_saturates() {
        let budget = GovernedAccumulatorBudget::new(1 << 20, 4000);
        assert!(budget.try_park(4000));
        assert!(!budget.try_park(1), "full");
        budget.release(4000);
        assert_eq!(budget.live_bytes(), 0);
        assert!(budget.try_park(4000), "released capacity is reusable");

        budget.release(1 << 30);
        assert_eq!(budget.live_bytes(), 0, "over-release saturates at zero");
    }
}

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
//! - **Declared before allocated / declinable.** The owning kernel decides
//!   whether retention was admitted before calling [`Self::try_park`]. The
//!   process budget deliberately carries no mutable admission verdict: that
//!   decision belongs to the session, while this type answers only whether the
//!   process-wide byte ceiling has room.
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

use std::sync::atomic::{AtomicU64, Ordering};

/// Established retention ceilings for CPU accumulator scratch.
///
/// Scratch users share these defaults rather than choosing independent
/// per-kernel policies: one thread may retain at most 32 MiB and one budget may
/// retain at most 128 MiB process-wide. Each scratch family has its own byte
/// budget; the owning kernel or session applies its own admission verdict
/// before asking this process counter to reserve bytes.
pub(crate) const DEFAULT_PER_THREAD_ACCUMULATOR_BYTES: u64 = 32 << 20;
pub(crate) const DEFAULT_PROCESS_ACCUMULATOR_BYTES: u64 = 128 << 20;

/// Process-wide accounting for a per-thread parked scratch buffer.
///
/// A single instance is meant to live in a `static` and be shared by every
/// thread that parks a buffer. `retained_bytes` is the figure the per-thread
/// constant was silent about: the sum currently held across all threads.
pub struct GovernedAccumulatorBudget {
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
    pub const fn new(per_thread_cap_bytes: u64, process_cap_bytes: u64) -> Self {
        Self {
            retained_bytes: AtomicU64::new(0),
            per_thread_cap_bytes: AtomicU64::new(per_thread_cap_bytes),
            process_cap_bytes: AtomicU64::new(process_cap_bytes),
        }
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
    /// single buffer is within the per-thread cap and the new total stays
    /// within the process cap. On failure the caller must drop its buffer and
    /// recompute next call -- a pure performance tradeoff, never a numerical
    /// one.
    pub fn try_park(&self, bytes: u64) -> bool {
        self.try_reserve_bytes(bytes)
    }

    /// Try to reserve `bytes` for a governed per-thread accumulator.
    ///
    /// The returned token owns exactly one successful accounting increment and
    /// releases it on drop. Requiring a token for a retained buffer makes
    /// normal return, unwind, replacement, and thread exit use the same release
    /// path instead of relying on every caller to remember a matching
    /// [`Self::release`].
    pub(crate) fn try_reserve(&'static self, bytes: u64) -> Option<GovernedAccumulatorReservation> {
        self.try_reserve_bytes(bytes)
            .then(|| GovernedAccumulatorReservation {
                budget: self,
                bytes,
            })
    }

    fn try_reserve_bytes(&self, bytes: u64) -> bool {
        if bytes == 0 {
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

    /// Release exactly `bytes` previously reserved by [`Self::try_park`].
    ///
    /// Returns `false` and leaves the counter unchanged if the release was not
    /// backed by enough retained bytes. This refuses an accounting underflow
    /// rather than saturating it away and making a later leak indistinguishable
    /// from correct accounting.
    pub fn release(&self, bytes: u64) -> bool {
        if bytes == 0 {
            return true;
        }
        let mut current = self.retained_bytes.load(Ordering::Relaxed);
        loop {
            let Some(next) = current.checked_sub(bytes) else {
                return false;
            };
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

/// One successful process-budget reservation.
///
/// This type is intentionally not `Clone`: one accounting increment has one
/// owner and therefore one release, including when a thread-local accumulator
/// is destroyed at thread exit.
pub(crate) struct GovernedAccumulatorReservation {
    budget: &'static GovernedAccumulatorBudget,
    bytes: u64,
}

impl Drop for GovernedAccumulatorReservation {
    fn drop(&mut self) {
        // Never panic from Drop. The token can only be constructed after one
        // successful increment and is not Clone, so a failed release would
        // indicate internal memory-accounting corruption rather than an input
        // condition that can be recovered here.
        let _ = self.budget.release(self.bytes);
    }
}

/// A reusable per-thread accumulator whose retained allocation is governed by
/// a [`GovernedAccumulatorBudget`].
///
/// The reservation lives beside the `Vec`, so taking the buffer for one call
/// releases the parked-byte accounting, replacing or declining it drops the
/// old reservation, and TLS destruction releases it automatically. Callers
/// only park after successful work; an error or unwind drops the checked-out
/// allocation without retaining it.
pub(crate) struct GovernedAccumulator<T> {
    reservation: Option<GovernedAccumulatorReservation>,
    buffer: Vec<T>,
}

impl<T> GovernedAccumulator<T> {
    pub(crate) const fn new() -> Self {
        Self {
            reservation: None,
            buffer: Vec::new(),
        }
    }

    /// Check out the retained buffer for one call.
    ///
    /// Dropping the reservation before returning makes `live_bytes()` describe
    /// only allocations currently parked between calls, never transient
    /// workspace in active use.
    pub(crate) fn take(&mut self) -> Vec<T> {
        self.reservation.take();
        std::mem::take(&mut self.buffer)
    }

    /// Park `buffer` for reuse when both retention ceilings admit its actual
    /// allocation capacity.
    ///
    /// A refused, zero-capacity, or byte-count-overflowing buffer is dropped
    /// here. The caller cannot accidentally retain it outside the governed
    /// container after this method returns.
    pub(crate) fn try_park(
        &mut self,
        buffer: Vec<T>,
        budget: &'static GovernedAccumulatorBudget,
    ) -> bool {
        self.clear();
        let Some(bytes) = capacity_bytes(&buffer) else {
            return false;
        };
        let Some(reservation) = budget.try_reserve(bytes) else {
            return false;
        };
        self.buffer = buffer;
        self.reservation = Some(reservation);
        true
    }

    /// Bytes currently retained by this thread's parked buffer.
    pub(crate) fn capacity_bytes(&self) -> usize {
        self.buffer
            .capacity()
            .saturating_mul(std::mem::size_of::<T>())
    }

    /// Drop any parked allocation and its reservation.
    pub(crate) fn clear(&mut self) {
        self.reservation = None;
        self.buffer = Vec::new();
    }
}

impl<T> Default for GovernedAccumulator<T> {
    fn default() -> Self {
        Self::new()
    }
}

fn capacity_bytes<T>(buffer: &Vec<T>) -> Option<u64> {
    u64::try_from(buffer.capacity())
        .ok()?
        .checked_mul(u64::try_from(std::mem::size_of::<T>()).ok()?)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Zero bytes never create a reservation.
    #[test]
    fn zero_bytes_are_not_parked() {
        let budget = GovernedAccumulatorBudget::new(1 << 20, 1 << 20);
        for _ in 0..8 {
            assert!(!budget.try_park(0));
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
            assert!(
                budget.try_park(1000),
                "buffer {expected} fits under the sum"
            );
            assert_eq!(budget.live_bytes(), expected * 1000);
        }
        assert!(
            !budget.try_park(1000),
            "the fifth buffer exceeds the process cap and is refused"
        );
        assert_eq!(budget.live_bytes(), 4000, "the sum is bounded by the cap");
    }

    /// Releasing returns capacity to the pool so a later park can use it, while
    /// an unmatched release is refused without changing the live count.
    #[test]
    fn release_returns_capacity_and_refuses_underflow() {
        let budget = GovernedAccumulatorBudget::new(1 << 20, 4000);
        assert!(budget.try_park(4000));
        assert!(!budget.try_park(1), "full");
        assert!(budget.release(4000));
        assert_eq!(budget.live_bytes(), 0);
        assert!(budget.try_park(4000), "released capacity is reusable");

        assert!(!budget.release(1 << 30));
        assert_eq!(
            budget.live_bytes(),
            4000,
            "an unmatched release must not corrupt the live-byte count"
        );
    }
}

//! A weight-derived cache that cannot be filled without a memory-plan verdict.
//!
//! # Why this type exists
//!
//! Kernels keep derived copies of constant weights so they are built once per
//! session instead of once per call: a dequantised expansion, a packed panel, a
//! transpose, a widened f32 copy. Each is a residency policy, whether or not it
//! was written as one, and each scales with model size.
//!
//! Seven of them were found in this crate, and the three that had already caused
//! visible damage were governed one at a time after the damage:
//!
//! - the resident f32 dequant cache took a 14B int4 model to ~66 GB (#971/#979,
//!   governed by #987);
//! - the MLAS SQNBit packed buffer took the same model's peak working set from
//!   8.17 GB to 25.5 GB while the ledger read zero (#1027, governed by #1051);
//! - the weight transpose cache reported entry counts rather than bytes, so it
//!   could not answer "will this fit" at all (#1035, governed by #1079).
//!
//! The remaining ones were found by grepping for the *pattern* rather than by
//! chasing a symptom (#1056), which is the point: every author simply did not
//! know the rule existed. A bare `OnceLock<Vec<T>>` field is easy to add and
//! invisible to the plan, so the accounting has to be structural rather than
//! remembered.
//!
//! # The contract
//!
//! From #1056: **any allocation that outlives a single kernel call and scales
//! with weight size must be declared to the plan before it is allocated, in the
//! bytes actually allocated, and must be declinable.**
//!
//! This type enforces the three halves that can be enforced in code:
//!
//! - **Declared before allocated.** There is no way to construct one without
//!   stating the predicted bytes and the plan's answer, so a cache cannot come
//!   into existence unaccounted.
//! - **Declinable.** A declined cache never stores anything; [`Self::get_or_fill`]
//!   returns `None` and the caller recomputes per call. Declining must therefore
//!   be a pure performance tradeoff, never a numerical one -- every existing
//!   decline path in this crate has a working fallback for exactly this reason.
//! - **Bytes, not entries.** [`Self::live_bytes`] reports what is actually held,
//!   so a prediction can be checked against reality in one run rather than
//!   against a formula. #1051's first accounting attempt reported 40% of the
//!   truth because it modelled the allocation instead of measuring it, and an
//!   under-reporting ledger is worse than an absent one: zero is obviously blind
//!   and gets caught, while 40% passes admission and then overruns the budget.

use std::sync::OnceLock;

/// The plan's answer for one weight-derived cache, carried alongside the
/// prediction it was made from so the two cannot drift apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheVerdict {
    predicted_bytes: u64,
    admitted: bool,
}

impl CacheVerdict {
    /// The plan admitted `predicted_bytes` for this cache.
    pub fn admit(predicted_bytes: u64) -> Self {
        Self {
            predicted_bytes,
            admitted: true,
        }
    }

    /// The plan declined: the cache must hold nothing and the kernel must fall
    /// back to recomputing per call.
    pub fn decline(predicted_bytes: u64) -> Self {
        Self {
            predicted_bytes,
            admitted: false,
        }
    }

    /// Bytes the plan was told this cache would cost.
    pub fn predicted_bytes(self) -> u64 {
        self.predicted_bytes
    }

    pub fn is_admitted(self) -> bool {
        self.admitted
    }
}

/// A session-lifetime cache of weight-derived data, fillable only under an
/// admitting [`CacheVerdict`].
///
/// `T` is the element type of the derived buffer; the byte figure is computed
/// from the buffer's own length, so it describes what was allocated rather than
/// what was expected.
pub struct GovernedWeightCache<T> {
    verdict: CacheVerdict,
    entry: OnceLock<Vec<T>>,
}

impl<T> GovernedWeightCache<T> {
    /// Create a cache under a plan verdict. There is deliberately no `Default`
    /// and no `new()` without a verdict: an unaccounted cache should be
    /// impossible to write by accident, which is the whole purpose of the type.
    pub fn new(verdict: CacheVerdict) -> Self {
        Self {
            verdict,
            entry: OnceLock::new(),
        }
    }

    pub fn verdict(&self) -> CacheVerdict {
        self.verdict
    }

    /// Bytes currently held. Zero when declined or not yet filled.
    ///
    /// This is the figure a prediction must be checked against. Note it counts
    /// the buffer only: a caller whose derived data carries owned side data (as
    /// MLAS's packed buffer carries a copy of the scales, ~+25%) must include
    /// that in its own prediction *and* in what it stores here, or the two will
    /// disagree the way #1051's first attempt did.
    pub fn live_bytes(&self) -> u64 {
        self.entry
            .get()
            .map_or(0, |values| (values.len() * std::mem::size_of::<T>()) as u64)
    }

    /// Whether anything is currently held.
    pub fn is_filled(&self) -> bool {
        self.entry.get().is_some()
    }

    /// Return the cached buffer, filling it with `build` on first use.
    ///
    /// Returns `None` when the plan declined, and then `build` is **never
    /// called** -- declining must not pay the construction cost only to throw
    /// the result away. The caller recomputes transiently instead.
    pub fn get_or_fill(&self, build: impl FnOnce() -> Vec<T>) -> Option<&[T]> {
        if !self.verdict.admitted {
            return None;
        }
        Some(self.entry.get_or_init(build).as_slice())
    }

    /// Like [`Self::get_or_fill`] but for a builder that can fail. A failed
    /// build leaves the cache empty and is retried on the next call, which
    /// matches the existing `OnceLock` kernels' behaviour.
    pub fn get_or_try_fill<E>(
        &self,
        build: impl FnOnce() -> Result<Vec<T>, E>,
    ) -> Result<Option<&[T]>, E> {
        if !self.verdict.admitted {
            return Ok(None);
        }
        if let Some(values) = self.entry.get() {
            return Ok(Some(values.as_slice()));
        }
        let values = build()?;
        // A concurrent filler may have won; either way the stored value is a
        // correct derivation of the same constant weight.
        let _ = self.entry.set(values);
        Ok(self.entry.get().map(Vec::as_slice))
    }
}

impl<T> std::fmt::Debug for GovernedWeightCache<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GovernedWeightCache")
            .field("admitted", &self.verdict.admitted)
            .field("predicted_bytes", &self.verdict.predicted_bytes)
            .field("live_bytes", &self.live_bytes())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// The declined path must hold nothing *and* must not pay to build it.
    /// Declining is a memory decision; making it cost the construction anyway
    /// would turn it into a pure loss.
    #[test]
    fn a_declined_cache_holds_nothing_and_never_builds() {
        let builds = AtomicUsize::new(0);
        let cache = GovernedWeightCache::<f32>::new(CacheVerdict::decline(4096));

        for _ in 0..3 {
            let got = cache.get_or_fill(|| {
                builds.fetch_add(1, Ordering::Relaxed);
                vec![1.0; 1024]
            });
            assert!(got.is_none(), "a declined cache must serve nothing");
        }

        assert_eq!(builds.load(Ordering::Relaxed), 0, "declined must not build");
        assert_eq!(cache.live_bytes(), 0);
        assert!(!cache.is_filled());
    }

    /// An admitted cache builds once and then reports the bytes it actually
    /// holds -- the figure a prediction is checked against.
    #[test]
    fn an_admitted_cache_builds_once_and_reports_real_bytes() {
        let builds = AtomicUsize::new(0);
        let cache = GovernedWeightCache::<f32>::new(CacheVerdict::admit(4096));

        for _ in 0..3 {
            let got = cache
                .get_or_fill(|| {
                    builds.fetch_add(1, Ordering::Relaxed);
                    vec![1.0; 1024]
                })
                .expect("an admitted cache serves its buffer");
            assert_eq!(got.len(), 1024);
        }

        assert_eq!(
            builds.load(Ordering::Relaxed),
            1,
            "built once, reused after"
        );
        assert_eq!(cache.live_bytes(), 1024 * 4);
        assert!(cache.is_filled());
    }

    /// `live_bytes` must describe the allocation, not the prediction. When they
    /// disagree the accounting is wrong and a test comparing them can say so;
    /// a type that simply echoed the prediction back could not.
    #[test]
    fn live_bytes_describes_the_allocation_not_the_prediction() {
        // A deliberately wrong prediction: half the true size.
        let cache = GovernedWeightCache::<u16>::new(CacheVerdict::admit(1024));
        cache
            .get_or_fill(|| vec![0u16; 1024])
            .expect("admitted cache fills");

        assert_eq!(cache.verdict().predicted_bytes(), 1024);
        assert_eq!(cache.live_bytes(), 2048, "reports what was allocated");
        assert_ne!(
            cache.live_bytes(),
            cache.verdict().predicted_bytes(),
            "the type must be able to expose a prediction that is wrong, which is \
             how #1051's 2.4x under-report would have been caught"
        );
    }

    /// A failed build leaves the cache empty and is retried, rather than
    /// poisoning the slot with a partial result.
    #[test]
    fn a_failed_build_leaves_the_cache_empty_and_retries() {
        let attempts = AtomicUsize::new(0);
        let cache = GovernedWeightCache::<f32>::new(CacheVerdict::admit(64));

        let first = cache.get_or_try_fill(|| {
            attempts.fetch_add(1, Ordering::Relaxed);
            Err::<Vec<f32>, &str>("no")
        });
        assert_eq!(first, Err("no"));
        assert_eq!(cache.live_bytes(), 0);

        let second = cache.get_or_try_fill(|| {
            attempts.fetch_add(1, Ordering::Relaxed);
            Ok::<Vec<f32>, &str>(vec![2.0; 16])
        });
        assert_eq!(second.unwrap().map(<[f32]>::len), Some(16));
        assert_eq!(
            attempts.load(Ordering::Relaxed),
            2,
            "the failure was retried"
        );
        assert_eq!(cache.live_bytes(), 64);
    }
}

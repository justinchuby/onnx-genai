//! The single prefix-reuse policy, shared by every decoder backend.
//!
//! Both backends answer the same question at the top of a turn: the retained
//! context says some prompt prefix is reusable, the KV holds some number of
//! tokens — how much of it survives, and what has to happen to the cache
//! first? Only the *mechanism* differs. ORT owns its past as pipeline-held
//! tensors it slices; a native decoder owns a session-resident cache it
//! rewinds. That difference belongs in [`KvPrefixStore`], and nowhere else.
//!
//! Keeping the policy in one place is not tidiness. When it lived in two, the
//! copies drifted: the native copy had no `use_kv` gate and the ORT copy had no
//! clamp against the KV's real length, so each was missing a guard the other
//! had. Neither omission could show up as a crash — a prefix-reuse bug produces
//! fluent text conditioned on the wrong context.

/// A prefix length that came from the shared policy.
///
/// This is the DRY mechanism, not decoration. The field is private and the
/// only constructors live in this module, so a new backend **cannot produce a
/// reuse length at all** without going through [`apply_prefix_reuse`] — the
/// exact mistake that produced two divergent copies of this policy in the
/// first place is now a compile error rather than a review catch.
///
/// If you find yourself wanting a fourth constructor, that is the signal to
/// stop: you are about to add a second policy. Extend [`plan_prefix_reuse`]
/// instead, so every backend gets the change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReusedPrefix(usize);

impl ReusedPrefix {
    /// Nothing carried over; the turn prefills the whole prompt.
    pub(crate) const NONE: Self = Self(0);

    /// The paged cache admits its own shared prefix through `claim_paged_prefix`,
    /// which is already a single implementation both backends share. It is a
    /// different *source* of a prefix, not a different policy.
    pub(crate) fn from_paged_admission(len: usize) -> Self {
        Self(len)
    }

    /// Tokens the turn may skip re-prefilling.
    pub(crate) fn len(self) -> usize {
        self.0
    }

    pub(crate) fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// What the KV cache must become before this turn can reuse a prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PrefixDecision {
    /// Nothing is reusable; the cache starts empty.
    Reset,
    /// The cache already holds exactly the reusable prefix. No work.
    Keep(usize),
    /// The cache holds more than the prompt shares; drop the tail first.
    Rewind(usize),
}

impl PrefixDecision {
    /// Tokens the turn may skip re-prefilling, before any rewind is attempted.
    #[cfg(test)]
    pub(crate) fn reused_len(self) -> usize {
        match self {
            Self::Reset => 0,
            Self::Keep(len) | Self::Rewind(len) => len,
        }
    }
}

/// Decide how much of `current` KV a prompt sharing `shared` leading tokens may
/// reuse. Pure, so the policy is testable without a model or a device.
///
/// `shared` comes from the retained context and describes *tokens*; `current`
/// is what the cache actually holds. They are maintained separately, so the
/// smaller governs — trusting `shared` alone would reuse tokens the cache never
/// had.
pub(crate) fn plan_prefix_reuse(
    shared: usize,
    current: usize,
    use_kv: bool,
    positions_are_linear: bool,
) -> PrefixDecision {
    if !use_kv || shared == 0 {
        return PrefixDecision::Reset;
    }
    let reusable = shared.min(current);
    if reusable == 0 {
        return PrefixDecision::Reset;
    }
    if reusable == current {
        // Pure extension: the cache is a prefix of this prompt, so the carried
        // position state stays valid and nothing has to be rewound.
        return PrefixDecision::Keep(reusable);
    }
    // Truncating invalidates carried positions, and only a linear continuation
    // can be rebuilt from the absolute past length alone. A model that carries
    // or is handed its coordinates would resume from positions describing
    // tokens that no longer exist.
    if !positions_are_linear {
        return PrefixDecision::Reset;
    }
    PrefixDecision::Rewind(reusable)
}

/// A rewindable KV cache. The only thing the two backends do differently.
pub(crate) trait KvPrefixStore {
    /// Tokens currently held. `0` when the backend keeps no reusable cache.
    fn current_kv_len(&self) -> usize;

    /// Whether this decoder carries a KV cache at all.
    fn use_kv(&self) -> bool;

    /// Drop everything after `target`. `Ok(false)` declines — an opaque past
    /// with no identifiable sequence axis, or fixed loop-carried state.
    fn rewind_to(&mut self, target: usize) -> anyhow::Result<bool>;

    /// Return to an empty cache. Must not leave stale entries behind.
    fn reset(&mut self) -> anyhow::Result<()>;
}

/// Apply [`plan_prefix_reuse`] to `store`, returning the reusable prefix length.
///
/// A declined rewind resets. This is the whole reason application is shared
/// rather than left to each call site: "I could not rewind" and "reuse nothing"
/// are only consistent if the cache is actually emptied. Reporting `0` while
/// leaving the old tokens in place would prefill the new prompt *on top of* a
/// previous turn's KV.
pub(crate) fn apply_prefix_reuse<S: KvPrefixStore + ?Sized>(
    store: &mut S,
    shared: usize,
    positions_are_linear: bool,
) -> anyhow::Result<ReusedPrefix> {
    let decision = plan_prefix_reuse(
        shared,
        store.current_kv_len(),
        store.use_kv(),
        positions_are_linear,
    );
    match decision {
        PrefixDecision::Reset => {
            store.reset()?;
            Ok(ReusedPrefix::NONE)
        }
        PrefixDecision::Keep(len) => Ok(ReusedPrefix(len)),
        PrefixDecision::Rewind(len) => {
            if store.rewind_to(len)? {
                Ok(ReusedPrefix(len))
            } else {
                store.reset()?;
                Ok(ReusedPrefix::NONE)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_decoder_without_kv_reuses_nothing() {
        assert_eq!(plan_prefix_reuse(8, 8, false, true), PrefixDecision::Reset);
        assert_eq!(plan_prefix_reuse(8, 8, false, true).reused_len(), 0);
    }

    #[test]
    fn no_shared_prompt_prefix_reuses_nothing() {
        assert_eq!(plan_prefix_reuse(0, 8, true, true), PrefixDecision::Reset);
    }

    #[test]
    fn an_empty_cache_reuses_nothing_however_much_the_prompt_shares() {
        assert_eq!(plan_prefix_reuse(8, 0, true, true), PrefixDecision::Reset);
    }

    #[test]
    fn a_cache_that_is_exactly_the_shared_prefix_is_kept_untouched() {
        assert_eq!(plan_prefix_reuse(8, 8, true, true), PrefixDecision::Keep(8));
    }

    #[test]
    fn the_cache_length_bounds_the_reuse_when_the_context_claims_more() {
        // The retained context and the cache are maintained separately, so a
        // context claiming more than the cache holds must not be believed.
        assert_eq!(
            plan_prefix_reuse(12, 8, true, true),
            PrefixDecision::Keep(8)
        );
    }

    #[test]
    fn a_diverging_prompt_rewinds_the_cache_to_the_shared_head() {
        assert_eq!(
            plan_prefix_reuse(5, 8, true, true),
            PrefixDecision::Rewind(5)
        );
    }

    #[test]
    fn a_diverging_prompt_cannot_rewind_nonlinear_positions() {
        assert_eq!(plan_prefix_reuse(5, 8, true, false), PrefixDecision::Reset);
    }

    #[test]
    fn nonlinear_positions_still_allow_a_pure_extension() {
        // Nothing is truncated, so no carried position is invalidated.
        assert_eq!(
            plan_prefix_reuse(8, 8, true, false),
            PrefixDecision::Keep(8)
        );
    }

    /// Records what a backend was actually asked to do.
    #[derive(Default)]
    struct FakeStore {
        len: usize,
        use_kv: bool,
        rewind_declines: bool,
        rewound_to: Option<usize>,
        was_reset: bool,
    }

    impl KvPrefixStore for FakeStore {
        fn current_kv_len(&self) -> usize {
            self.len
        }
        fn use_kv(&self) -> bool {
            self.use_kv
        }
        fn rewind_to(&mut self, target: usize) -> anyhow::Result<bool> {
            self.rewound_to = Some(target);
            Ok(!self.rewind_declines)
        }
        fn reset(&mut self) -> anyhow::Result<()> {
            self.was_reset = true;
            Ok(())
        }
    }

    #[test]
    fn a_kept_prefix_touches_the_cache_at_all() -> anyhow::Result<()> {
        let mut store = FakeStore {
            len: 8,
            use_kv: true,
            ..FakeStore::default()
        };
        assert_eq!(apply_prefix_reuse(&mut store, 8, true)?.len(), 8);
        assert_eq!(store.rewound_to, None, "a kept prefix needs no rewind");
        assert!(!store.was_reset);
        Ok(())
    }

    #[test]
    fn a_declined_rewind_empties_the_cache_rather_than_leaving_its_tail() -> anyhow::Result<()> {
        // Reporting zero reuse while the previous turn's tokens stay in the
        // cache would prefill the new prompt on top of them.
        let mut store = FakeStore {
            len: 8,
            use_kv: true,
            rewind_declines: true,
            ..FakeStore::default()
        };
        assert!(apply_prefix_reuse(&mut store, 5, true)?.is_empty());
        assert_eq!(store.rewound_to, Some(5));
        assert!(store.was_reset, "a declined rewind must reset");
        Ok(())
    }

    #[test]
    fn reusing_nothing_empties_the_cache() -> anyhow::Result<()> {
        let mut store = FakeStore {
            len: 8,
            use_kv: true,
            ..FakeStore::default()
        };
        assert!(apply_prefix_reuse(&mut store, 0, true)?.is_empty());
        assert!(store.was_reset);
        Ok(())
    }

    /// Files allowed to drive the KV-rewind primitives directly.
    ///
    /// `flat_autoregressive.rs` holds the two `KvPrefixStore` adapters, which
    /// are the sanctioned mechanism half of this module. `state.rs` defines
    /// `truncate_past` and exercises it in its own unit tests.
    const KV_REWIND_CALLERS: [&str; 3] = [
        "pipeline\\flat_autoregressive.rs",
        "pipeline\\prefix_reuse.rs",
        "decode\\state.rs",
    ];

    /// The primitives that mutate or inspect a KV cache's length. Reaching for
    /// one of these outside the adapters means reimplementing prefix reuse.
    const KV_REWIND_PRIMITIVES: [&str; 3] = [".rewind_kv(", ".truncate_past(", ".current_kv_len("];

    fn rust_sources(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).expect("crate src is readable") {
            let path = entry.expect("readable entry").path();
            if path.is_dir() {
                rust_sources(&path, out);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                out.push(path);
            }
        }
    }

    /// The DRY tripwire.
    ///
    /// A second copy of this policy cannot appear without calling one of the
    /// KV primitives, so confining those calls to the adapters confines the
    /// policy to this module. This runs under `cargo test`, so it cannot be
    /// missed by not looking at CI.
    ///
    /// If this fails, the fix is to route the new backend through
    /// [`apply_prefix_reuse`] and add a [`KvPrefixStore`] adapter -- **not** to
    /// add the offending file to `KV_REWIND_CALLERS`. Widening the allowlist is
    /// how the duplication this module exists to prevent gets back in.
    #[test]
    fn the_kv_rewind_primitives_are_only_driven_by_the_shared_policy() {
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut sources = Vec::new();
        rust_sources(&src, &mut sources);
        assert!(
            sources.len() > 10,
            "the source scan found almost nothing, so it cannot be trusted to \
             have looked: {} files under {}",
            sources.len(),
            src.display()
        );

        let mut offenders = Vec::new();
        for path in sources {
            let relative = path.strip_prefix(&src).expect("scanned under src");
            let display = relative.display().to_string().replace('/', "\\");
            if KV_REWIND_CALLERS
                .iter()
                .any(|allowed| display.ends_with(allowed))
            {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("source is readable");
            for (line_number, line) in text.lines().enumerate() {
                for primitive in KV_REWIND_PRIMITIVES {
                    if line.contains(primitive) {
                        offenders.push(format!("{display}:{} {}", line_number + 1, line.trim()));
                    }
                }
            }
        }

        assert!(
            offenders.is_empty(),
            "these call a KV-rewind primitive outside the `KvPrefixStore` \
             adapters, which is how prefix reuse gets reimplemented per \
             backend. Route them through `apply_prefix_reuse` instead of \
             widening `KV_REWIND_CALLERS`:\n{}",
            offenders.join("\n")
        );
    }
}

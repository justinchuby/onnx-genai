//! The single persistent-session state policy, shared by every decode backend.
//!
//! Both backends answer the same questions about a persistent session: does it
//! exist, how many logical tokens does it hold, is a rewind target in bounds,
//! and what token boundary does a checkpoint record? Only the *mechanism*
//! differs. The ORT backend keeps its session in `Engine::sessions` alongside a
//! paged KV cache, a scheduler entry, and optional draft state; the native
//! backend keeps only a token history in `Engine::native_sessions` plus a single
//! in-process decoder it rewinds. That difference belongs in [`SessionStore`],
//! and nowhere else.
//!
//! Keeping the policy in one place is not tidiness. When it lived in two, the
//! copies drifted risk was real and concrete: the `checked_sub` bound check and
//! its error string were duplicated *verbatim* between the native and ORT arms
//! of `rewind_session_by`, the `position > current` bound check and its string
//! between the two arms of `rewind_session_to`, and the "session {id} not found"
//! lookup across every method. A silent divergence in any of these would not
//! crash — it would reject a valid rewind, accept an out-of-bounds one, or emit
//! a different error than a caller's test expects, on exactly one backend.
//!
//! This mirrors [`super::prefix_reuse`], which unified the KV *length* policy
//! the same way: one pure decision, one trait for the mechanism, one DRY guard.

use crate::config::{SessionCheckpoint, SessionId, SessionPosition};
use anyhow::Context;

/// A rewind target that has already been bound-checked against the session's
/// logical length by [`rewind_to`].
///
/// This is the DRY mechanism, not decoration. The field is private and the only
/// constructor lives in this module, so a new backend **cannot produce a rewind
/// target at all** without going through the shared bound check — the exact
/// mistake that produced two verbatim copies of that bound check in the first
/// place is now a compile error rather than a review catch.
///
/// If you find yourself wanting another constructor, that is the signal to stop:
/// you are about to add a second copy of the bound-check policy. Route the new
/// backend through [`rewind_to`] instead, so every backend gets the same check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CheckedPosition(usize);

impl CheckedPosition {
    /// The bound-checked absolute token boundary this rewind targets.
    pub(crate) fn get(self) -> usize {
        self.0
    }
}

/// Read side of a persistent session: the logical token vector's length is the
/// single source of the "session not found" answer that every method needs.
pub(crate) trait SessionLen {
    /// Logical tokens retained by `id`, or `None` when the session is unknown.
    fn logical_len(&self, id: SessionId) -> Option<usize>;
}

/// The backend-specific half of session state: everything that touches KV,
/// scheduler, or decoder mechanism. The shared policy owns the lookup, the
/// bound checks, the error text, and the checkpoint arithmetic; a `SessionStore`
/// owns only the mechanism, and its methods are handed inputs the policy has
/// already validated.
pub(crate) trait SessionStore: SessionLen {
    /// Check whether a rewind to `target` is admissible **without mutating
    /// anything**, so a rejected rewind leaves the session untouched. The ORT
    /// backend validates that its decode runner and any draft state can honour
    /// the truncation here; the native backend always admits and clamps in
    /// [`SessionStore::rewind`].
    fn validate_rewind(&self, id: SessionId, target: CheckedPosition) -> anyhow::Result<()>;

    /// Truncate the session's logical tokens and backend KV to `target`.
    /// Called only after [`SessionStore::validate_rewind`] has accepted it.
    fn rewind(&mut self, id: SessionId, target: CheckedPosition) -> anyhow::Result<()>;

    /// Clear the session's logical tokens and reset its backend KV, keeping the
    /// id usable.
    fn reset(&mut self, id: SessionId) -> anyhow::Result<()>;

    /// Remove the session and free its backend state.
    fn close(&mut self, id: SessionId) -> anyhow::Result<()>;
}

/// The single source of the "session {id} not found" lookup and message.
///
/// Every policy function starts here, so the observable not-found text lives in
/// exactly one place across both backends.
fn require_session<S: SessionLen + ?Sized>(store: &S, id: SessionId) -> anyhow::Result<usize> {
    store
        .logical_len(id)
        .with_context(|| format!("session {id} not found"))
}

/// Record the current logical token boundary as a checkpoint.
pub(crate) fn checkpoint<S: SessionLen + ?Sized>(
    store: &S,
    id: SessionId,
) -> anyhow::Result<SessionCheckpoint> {
    let len = require_session(store, id)?;
    Ok(SessionCheckpoint {
        session_id: id,
        position: SessionPosition::new(len),
    })
}

/// Number of logical tokens retained by a session.
pub(crate) fn token_count<S: SessionLen + ?Sized>(
    store: &S,
    id: SessionId,
) -> anyhow::Result<usize> {
    require_session(store, id)
}

/// Rewind a session by `tokens` logical tokens, returning the new position.
pub(crate) fn rewind_by<S: SessionStore + ?Sized>(
    store: &mut S,
    id: SessionId,
    tokens: usize,
) -> anyhow::Result<SessionPosition> {
    let current = require_session(store, id)?;
    let target = current.checked_sub(tokens).with_context(|| {
        format!("cannot rewind session {id} by {tokens} tokens from length {current}")
    })?;
    let position = SessionPosition::new(target);
    rewind_to(store, id, position)?;
    Ok(position)
}

/// Rewind a session to an absolute logical token position.
///
/// This is the *only* producer of a [`CheckedPosition`]: the bound check that
/// gates it cannot be skipped or reimplemented by a backend.
pub(crate) fn rewind_to<S: SessionStore + ?Sized>(
    store: &mut S,
    id: SessionId,
    position: SessionPosition,
) -> anyhow::Result<()> {
    let requested = position.get();
    let current = require_session(store, id)?;
    if requested > current {
        anyhow::bail!(
            "cannot rewind session {id} to token {requested}; current length is {current}"
        );
    }
    let target = CheckedPosition(requested);
    store.validate_rewind(id, target)?;
    store.rewind(id, target)
}

/// Reset a session, clearing its logical tokens and backend KV.
pub(crate) fn reset<S: SessionStore + ?Sized>(
    store: &mut S,
    id: SessionId,
) -> anyhow::Result<()> {
    require_session(store, id)?;
    store.reset(id)
}

/// Close a session and free its backend state.
pub(crate) fn close<S: SessionStore + ?Sized>(
    store: &mut S,
    id: SessionId,
) -> anyhow::Result<()> {
    require_session(store, id)?;
    store.close(id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// A backend that records what the shared policy asked it to do, so the
    /// policy can be exercised without a model, KV cache, or device.
    #[derive(Default)]
    struct FakeStore {
        sessions: HashMap<SessionId, usize>,
        rewind_rejects: bool,
        rewound_to: Option<usize>,
        was_reset: bool,
        was_closed: bool,
    }

    impl FakeStore {
        fn with_session(id: SessionId, len: usize) -> Self {
            let mut store = FakeStore::default();
            store.sessions.insert(id, len);
            store
        }
    }

    impl SessionLen for FakeStore {
        fn logical_len(&self, id: SessionId) -> Option<usize> {
            self.sessions.get(&id).copied()
        }
    }

    impl SessionStore for FakeStore {
        fn validate_rewind(&self, _id: SessionId, target: CheckedPosition) -> anyhow::Result<()> {
            if self.rewind_rejects {
                anyhow::bail!("backend rejected the rewind");
            }
            // A shared-side error would mutate nothing; record only.
            let _ = target;
            Ok(())
        }

        fn rewind(&mut self, id: SessionId, target: CheckedPosition) -> anyhow::Result<()> {
            self.rewound_to = Some(target.get());
            self.sessions.insert(id, target.get());
            Ok(())
        }

        fn reset(&mut self, id: SessionId) -> anyhow::Result<()> {
            self.was_reset = true;
            self.sessions.insert(id, 0);
            Ok(())
        }

        fn close(&mut self, id: SessionId) -> anyhow::Result<()> {
            self.was_closed = true;
            self.sessions.remove(&id);
            Ok(())
        }
    }

    fn sid(id: u64) -> SessionId {
        id
    }

    #[test]
    fn a_missing_session_is_reported_the_same_way_everywhere() {
        let store = FakeStore::default();
        for message in [
            checkpoint(&store, sid(1)).unwrap_err().to_string(),
            token_count(&store, sid(1)).unwrap_err().to_string(),
        ] {
            assert_eq!(message, "session 1 not found");
        }

        let mut store = FakeStore::default();
        assert_eq!(
            reset(&mut store, sid(1)).unwrap_err().to_string(),
            "session 1 not found"
        );
        assert_eq!(
            close(&mut store, sid(1)).unwrap_err().to_string(),
            "session 1 not found"
        );
        assert_eq!(
            rewind_to(&mut store, sid(1), SessionPosition::new(0))
                .unwrap_err()
                .to_string(),
            "session 1 not found"
        );
        assert_eq!(
            rewind_by(&mut store, sid(1), 0).unwrap_err().to_string(),
            "session 1 not found"
        );
    }

    #[test]
    fn a_checkpoint_records_the_logical_length() {
        let store = FakeStore::with_session(sid(7), 5);
        assert_eq!(
            checkpoint(&store, sid(7)).unwrap().position,
            SessionPosition::new(5)
        );
        assert_eq!(token_count(&store, sid(7)).unwrap(), 5);
    }

    #[test]
    fn rewinding_past_the_length_is_rejected_with_the_shared_message() {
        let mut store = FakeStore::with_session(sid(3), 4);
        assert_eq!(
            rewind_to(&mut store, sid(3), SessionPosition::new(5))
                .unwrap_err()
                .to_string(),
            "cannot rewind session 3 to token 5; current length is 4"
        );
        assert_eq!(store.rewound_to, None, "a rejected rewind must not mutate");
    }

    #[test]
    fn rewinding_by_more_than_the_length_is_rejected_with_the_shared_message() {
        let mut store = FakeStore::with_session(sid(3), 4);
        assert_eq!(
            rewind_by(&mut store, sid(3), 5).unwrap_err().to_string(),
            "cannot rewind session 3 by 5 tokens from length 4"
        );
        assert_eq!(store.rewound_to, None);
    }

    #[test]
    fn an_in_bounds_rewind_reaches_the_backend() {
        let mut store = FakeStore::with_session(sid(3), 8);
        rewind_to(&mut store, sid(3), SessionPosition::new(5)).unwrap();
        assert_eq!(store.rewound_to, Some(5));
    }

    #[test]
    fn rewind_by_converts_to_an_absolute_target() {
        let mut store = FakeStore::with_session(sid(3), 8);
        let position = rewind_by(&mut store, sid(3), 3).unwrap();
        assert_eq!(position, SessionPosition::new(5));
        assert_eq!(store.rewound_to, Some(5));
    }

    #[test]
    fn a_backend_rejection_leaves_the_session_untouched() {
        let mut store = FakeStore {
            rewind_rejects: true,
            ..FakeStore::with_session(sid(3), 8)
        };
        assert!(rewind_to(&mut store, sid(3), SessionPosition::new(5)).is_err());
        assert_eq!(store.rewound_to, None, "a rejected rewind must not mutate");
    }

    #[test]
    fn reset_and_close_require_the_session_then_reach_the_backend() {
        let mut store = FakeStore::with_session(sid(3), 8);
        reset(&mut store, sid(3)).unwrap();
        assert!(store.was_reset);
        assert_eq!(store.logical_len(sid(3)), Some(0));

        close(&mut store, sid(3)).unwrap();
        assert!(store.was_closed);
        assert_eq!(store.logical_len(sid(3)), None);
    }

    /// Files allowed to spell the session-state policy strings.
    ///
    /// This module holds the one copy of the bound-check policy. A new backend
    /// that open-codes a session rewind would reintroduce this fragment
    /// elsewhere, which is the drift this module exists to prevent.
    const SESSION_POLICY_CALLERS: [&str; 1] = ["engine\\session_state.rs"];

    /// The bound-check message fragment unique to the rewind policy. `rewind_by`
    /// and `rewind_to` both format it; `fork_session` uses "cannot fork
    /// session", which is deliberately distinct.
    const SESSION_POLICY_FRAGMENT: &str = "cannot rewind session";

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

    /// The DRY tripwire, mirroring `prefix_reuse`'s KV-rewind guard.
    ///
    /// The rewind bound check is the piece that was duplicated *verbatim*
    /// between the two backends. Confining its message fragment to this module,
    /// together with the [`CheckedPosition`] compile-time guard, means a third
    /// backend cannot open-code a rewind: it can neither construct a validated
    /// target nor spell the bound-check error without tripping a test.
    ///
    /// If this fails, route the new backend through [`rewind_to`] and add a
    /// [`SessionStore`] adapter -- **not** add the offending file to
    /// `SESSION_POLICY_CALLERS`. Widening the allowlist is how the duplication
    /// this module exists to prevent gets back in.
    #[test]
    fn the_rewind_bound_check_lives_only_in_the_shared_policy() {
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
            if SESSION_POLICY_CALLERS
                .iter()
                .any(|allowed| display.ends_with(allowed))
            {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("source is readable");
            for (line_number, line) in text.lines().enumerate() {
                if line.contains(SESSION_POLICY_FRAGMENT) {
                    offenders.push(format!("{display}:{} {}", line_number + 1, line.trim()));
                }
            }
        }

        assert!(
            offenders.is_empty(),
            "these open-code the session rewind bound check outside the shared \
             `session_state` policy, which is how it gets reimplemented per \
             backend. Route them through `rewind_to` instead of widening \
             `SESSION_POLICY_CALLERS`:\n{}",
            offenders.join("\n")
        );
    }
}

//! The measurement window, as a type.
//!
//! [`hostlock`] gives a reading of the lock at an instant.
//! What a result row needs is a statement about a *window*, and those are not
//! the same claim: a lock read once at the end reports a credible holder for a
//! run that changed hands halfway through, which is worse than reporting
//! nothing because it is convincing.
//!
//! So the two-ended read is not left to each caller to remember. [`Window`]
//! takes the first reading when it is opened and the second when it is closed,
//! and there is no way to obtain a [`Report`] without both. A benchmark that
//! wanted to publish a one-ended verdict would have to go around this module
//! rather than merely forget something.
//!
//! # Why the strings live here
//!
//! The row text and the warning text are produced here rather than at each
//! call site. Ten benchmark binaries formatting their own `host_lock=` line is
//! ten chances for the vocabulary to drift, and a field that means one thing in
//! one matrix and another thing in the next is worse than an absent field --
//! the reader cannot tell which they are looking at. The same argument the
//! campaign has been making about `efficiency` naming two quantities applies to
//! its own instruments.

use crate::hostlock::{self, LockField, LockState};
use std::fmt;

/// A measurement window, bracketed by two readings of the advisory host lock.
#[derive(Debug)]
pub struct Window {
    before: LockState,
}

impl Window {
    /// Reads the lock as the window opens.
    ///
    /// Call this **before** any warmup, not just before the timed region. A
    /// lock taken after a run began covers the part of the run somebody else
    /// could still have been on.
    pub fn open() -> Self {
        Self {
            before: hostlock::read(),
        }
    }

    /// Opens a window on a reading supplied by the caller. For tests, and for
    /// a caller that has already read the lock for another purpose.
    pub fn opened_at(before: LockState) -> Self {
        Self { before }
    }

    /// Reads the lock again and produces the verdict for the whole window.
    ///
    /// Consumes the window: a second `close` would read a third instant and
    /// silently widen the claim.
    pub fn close(self) -> Report {
        self.close_with(hostlock::read)
    }

    /// [`close`](Self::close) against a reading supplied by the caller.
    pub fn closed_at(self, after: LockState) -> Report {
        self.close_with(move || after)
    }

    /// The one implementation of closing, with the second read injected.
    ///
    /// Injected for the same reason [`hostlock::classify_io`] takes its probe
    /// as a parameter: otherwise the only way to exercise this is to point the
    /// process at a lock directory, and the arm that matters -- the two
    /// readings disagreeing -- cannot be produced on demand at all. A mutation
    /// run found `close` untested precisely because it was the one path no test
    /// could reach.
    pub fn close_with(self, read_after: impl FnOnce() -> LockState) -> Report {
        let after = read_after();
        // One reading, used for both the field and the reason. Reading twice
        // would let the printed reason describe a different lock than the
        // printed verdict -- `host_lock=changed lock_reason=acc0` names a
        // holder for a window the field itself says had none, which invites a
        // reader to dismiss the `changed`.
        Report {
            field: hostlock::field_from_env(&self.before, &after),
            reason: hostlock::reason(&self.before, &after),
        }
    }
}

/// What the lock said about a whole measurement window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Report {
    pub field: LockField,
    pub reason: Option<String>,
}

impl Report {
    /// Whether this window was covered end to end by a lock held by this
    /// process. Only `mine:` qualifies; see [`LockField::is_protected`].
    pub fn is_protected(&self) -> bool {
        self.field.is_protected()
    }

    /// The loud complaint for an unprotected window, or `None` when there is
    /// nothing to complain about.
    ///
    /// Deliberately not fatal. Refusing to print a matrix because nobody took a
    /// lock would mostly teach people to stop taking the lock, and an unlocked
    /// run on a genuinely idle box is fine -- what is not fine is one that
    /// cannot be told apart from it afterwards.
    pub fn warning(&self) -> Option<String> {
        if self.is_protected() {
            return None;
        }
        Some(format!(
            "UNPROTECTED host_lock={} -- this run was not covered end-to-end by a lock held by \
             this process (HOSTLOCK_OWNER={}). Take one with `scripts/hostlock.sh run` before \
             publishing these numbers.",
            self.field,
            std::env::var("HOSTLOCK_OWNER").unwrap_or_else(|_| "unset".into())
        ))
    }
}

/// Renders the `key=value` pair(s) for a result row.
impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.reason {
            Some(reason) => write!(f, "host_lock={} lock_reason={reason}", self.field),
            None => write!(f, "host_lock={}", self.field),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hostlock::parse_meta;

    fn held(owner: &str, reason: &str) -> LockState {
        let meta = format!("anchor_pid=1\nstart_time=900\nowner={owner}\nreason={reason}\n");
        LockState::Held(parse_meta(&meta).expect("fixture parses"))
    }

    #[test]
    fn a_window_that_changed_hands_names_no_holder_and_gives_no_reason() {
        let report = Window::opened_at(held("roy", "acc0")).closed_at(LockState::Free);
        // Both halves matter. `changed` with a reason attached would let a
        // reader take the reason as evidence the window had a holder after all.
        assert_eq!(report.field, LockField::Changed);
        assert_eq!(report.reason, None);
        assert_eq!(report.to_string(), "host_lock=changed");
    }

    #[test]
    fn a_steady_lock_is_reported_with_the_holders_reason() {
        let report =
            Window::opened_at(held("roy", "thp-probe")).closed_at(held("roy", "thp-probe"));
        // `held:` rather than `foreign:` because `HOSTLOCK_OWNER` is unset in a
        // test process, so there is nothing to compare the holder against. That
        // is the honest answer and it is unprotected either way; the
        // owner-dependent split between `mine:` and `foreign:` belongs to
        // `hostlock::field`, which takes the owner as an argument and is tested
        // there. This asserted `foreign:` on first writing and failed, which is
        // the reason it is spelled out: the env-dependent arm is not reachable
        // from here without setting a process-wide variable, and a test that
        // leaks `HOSTLOCK_OWNER` into every later test in the binary would be a
        // worse bug than the one it checks.
        assert_eq!(report.field, LockField::Held("roy".into()));
        assert_eq!(
            report.to_string(),
            "host_lock=held:roy lock_reason=thp-probe"
        );
        assert!(!report.is_protected());
    }

    #[test]
    fn an_unprotected_window_warns_and_a_protected_one_does_not() {
        let unprotected = Window::opened_at(LockState::Free).closed_at(LockState::Free);
        let warning = unprotected.warning().expect("a free window must complain");
        assert!(warning.starts_with("UNPROTECTED host_lock=free"));

        let mine = Report {
            field: LockField::Mine("sebastian".into()),
            reason: None,
        };
        assert_eq!(
            mine.warning(),
            None,
            "a window this process held end to end is the one case with nothing to say"
        );
    }

    #[test]
    fn a_holders_reason_cannot_splice_extra_fields_into_a_row() {
        // The row is a `key=value` list, so an unsanitised reason could append
        // pairs of its own. `hostlock.sh` documents this hazard against its own
        // provenance line; a result row has exactly the same shape.
        let evil = "acc0 host_lock=free declared=yes";
        let report = Window::opened_at(held("roy", evil)).closed_at(held("roy", evil));
        let rendered = report.to_string();
        assert_eq!(
            rendered.split_whitespace().count(),
            2,
            "a reason must not be able to add fields to the row, got: {rendered}"
        );
        assert!(!rendered.contains("declared=yes"), "{rendered}");
    }

    #[test]
    fn closing_reads_the_lock_again_rather_than_reusing_the_first_reading() {
        // The defect this pins: `close` ignoring the reading it was opened
        // with, so a window that began on a free host and ended on a claimed
        // one reports the claim as if it had covered the whole run. The
        // injected reader is the only way to make the two ends differ on
        // demand; with the real one the arm is unreachable in a test.
        let mut reads = 0;
        let report = Window::opened_at(LockState::Free).close_with(|| {
            reads += 1;
            held("gaff", "verify")
        });
        assert_eq!(reads, 1, "closing must read exactly once");
        assert_eq!(report.field, LockField::Changed);
        assert!(!report.is_protected());

        // And the converse: two agreeing ends are not reported as a change.
        let steady =
            Window::opened_at(held("gaff", "verify")).close_with(|| held("gaff", "verify"));
        assert_eq!(steady.field, LockField::Held("gaff".into()));
    }

    #[test]
    fn a_report_cannot_be_had_from_one_reading() {
        // Not an assertion about behaviour but about the shape of the API: the
        // only constructors of `Report` outside this crate go through a
        // `Window`, which cannot be built without a first reading and cannot
        // yield a report without a second. Stated as a test so that a future
        // change adding a one-ended shortcut has to delete a named claim rather
        // than merely add a function.
        let window = Window::opened_at(LockState::Unknown);
        let report = window.closed_at(LockState::Unknown);
        assert_eq!(report.field, LockField::Unknown);
        assert!(!report.is_protected());
    }
}

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
//! and outside this module there is no way to obtain a [`Report`] without both:
//! its fields are private and it has no public constructor. A benchmark that
//! wanted to publish a one-ended verdict would have to change this file rather
//! than merely forget something.
//!
//! That is a claim about the API, so it is asserted the only way an API claim
//! can be -- by code that must fail to compile, on [`Report`], pinned to the
//! specific error rather than to "it did not build".
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

    /// Reads the lock again and produces the verdict for the whole window,
    /// attributing it against `HOSTLOCK_OWNER`.
    ///
    /// Consumes the window: a second `close` would read a third instant and
    /// silently widen the claim.
    ///
    /// This is the one function a benchmark calls, and the only one that
    /// touches the environment. Everything it decides is in
    /// [`close_as`](Self::close_as), which takes both the owner and the second
    /// reading as arguments; the split is what makes the decision testable
    /// without a process-wide variable.
    pub fn close(self) -> Report {
        // No `$USER` fallback, for the reason `hostlock::field_from_env`
        // documents: every agent on this host shares a user, so a default would
        // print `mine:` over a co-tenant's lock.
        let owner = std::env::var("HOSTLOCK_OWNER").ok();
        self.close_as(owner.as_deref(), hostlock::read)
    }

    /// The one implementation of closing, with the second reading and the
    /// self-owner both supplied by the caller.
    ///
    /// Injected for the same reason [`hostlock::classify_io`] takes its probe
    /// as a parameter: otherwise the only way to exercise this is to point the
    /// process at a lock directory and set `HOSTLOCK_OWNER`, and the arm that
    /// matters -- the two readings disagreeing -- cannot be produced on demand
    /// at all. A mutation run found closing untested precisely because it was
    /// the one path no test could reach.
    ///
    /// Taking the owner as an argument is not only for tests. A test that read
    /// it from the environment would assert an ambient value: it passes on a
    /// bare shell and fails in the one this subsystem is used from, where
    /// `HOSTLOCK_OWNER` is set. That is an assertion about the shell, not about
    /// the code.
    pub fn close_as(
        self,
        self_owner: Option<&str>,
        read_after: impl FnOnce() -> LockState,
    ) -> Report {
        let after = read_after();
        // One reading, used for both the field and the reason. Reading twice
        // would let the printed reason describe a different lock than the
        // printed verdict -- `host_lock=changed lock_reason=acc0` names a
        // holder for a window the field itself says had none, which invites a
        // reader to dismiss the `changed`.
        Report {
            field: hostlock::field(&self.before, &after, self_owner),
            reason: hostlock::reason(&self.before, &after),
            self_owner: self_owner.map(str::to_string),
        }
    }
}

/// What the lock said about a whole measurement window.
///
/// # Why the fields are private
///
/// A `Report` is a verdict about a span, and the only honest way to reach one
/// is to have read the lock at both ends of that span. Public fields would make
/// the two readings optional in practice: `Report { field: Mine("me"), .. }`
/// compiles, reads as protected, prints no warning, and is derived from no
/// measurement at all -- the flattering answer, available by accident. Private
/// fields plus [`Window`] as the only constructor make the fabricated verdict a
/// compile error instead of a plausible row.
///
/// It also keeps the sanitising in one place. `reason` is a holder's untrusted
/// string in a `key=value` row; it is scrubbed on the way in by
/// [`hostlock::reason`], and a settable field would let a caller put the
/// unscrubbed original back.
///
/// Both halves of that are asserted here rather than asserted in prose, by two
/// blocks that must fail to compile.
///
/// A `compile_fail` block is a weak instrument on its own: it passes when the
/// snippet fails to build *for any reason*, including a mistyped path or a
/// renamed type, at which point it checks nothing and still reports `ok`.
/// rustdoc does not enforce the error code on stable -- annotating these
/// `compile_fail,E0451` and then mistyping `Report` as `Reportt` still passes,
/// which was measured rather than assumed. So the first block below is a
/// positive control: it uses the same names by the same paths and must
/// compile, which is what makes the two failures afterwards evidence about
/// privacy rather than about spelling. `scripts/hostlock_mutants.py` carries
/// the other direction, a mutant that makes these fields `pub` and must be
/// killed here.
///
/// ```
/// use onnx_runtime_hostmon::hostlock::{LockField, LockState};
/// use onnx_runtime_hostmon::window::{Report, Window};
///
/// // The supported route: two readings in, a verdict out, read back through
/// // the accessors. No lock directory is touched.
/// let report: Report = Window::opened_at(LockState::Free).close_as(None, || LockState::Free);
/// assert_eq!(report.field(), &LockField::Free);
/// assert_eq!(report.reason(), None);
/// assert!(!report.is_protected());
/// ```
///
/// ```compile_fail
/// use onnx_runtime_hostmon::hostlock::LockField;
/// use onnx_runtime_hostmon::window::Report;
///
/// // A verdict fabricated from no readings at all: `field` is private.
/// let forged = Report {
///     field: LockField::Mine("nobody-read-the-lock".into()),
///     reason: None,
///     self_owner: None,
/// };
/// ```
///
/// ```compile_fail
/// use onnx_runtime_hostmon::hostlock::LockState;
/// use onnx_runtime_hostmon::window::Window;
///
/// // Nor can a caller edit a verdict after the fact.
/// let mut report = Window::opened_at(LockState::Free).close_as(None, || LockState::Free);
/// report.reason = Some("acc0 declared=yes".to_string());
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Report {
    field: LockField,
    reason: Option<String>,
    /// The `HOSTLOCK_OWNER` this verdict was decided against, kept so that
    /// [`warning`](Self::warning) can name it without reading the environment a
    /// second time. A warning that re-read it could report an owner the field
    /// was never compared with.
    self_owner: Option<String>,
}

impl Report {
    /// The `host_lock=` value.
    pub fn field(&self) -> &LockField {
        &self.field
    }

    /// The holder's stated reason, when both readings agreed on a holder.
    pub fn reason(&self) -> Option<&str> {
        self.reason.as_deref()
    }

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
             this process (HOSTLOCK_OWNER={}). Take one with `scripts/hostlock.sh run --owner \
             <you> --reason \"<what this measures>\"` before publishing these numbers.",
            self.field,
            self.self_owner.as_deref().unwrap_or("unset")
        ))
    }
}

/// Renders the `key=value` pair(s) for a result row.
impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.reason() {
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

    /// Every test here closes with [`Window::close_as`] and never with
    /// [`Window::close`]. `close` reads `HOSTLOCK_OWNER` from the environment,
    /// so a test that used it would assert whatever the shell happened to
    /// export -- passing on a bare shell and failing inside
    /// `scripts/hostlock.sh run`, which is the only place this code ever runs
    /// for real. The environment-reading half is covered end to end by
    /// `agrees_with_hostlock_sh.rs`, in a child process, where an override
    /// cannot leak into any other test.
    #[test]
    fn a_window_that_changed_hands_names_no_holder_and_gives_no_reason() {
        let report = Window::opened_at(held("roy", "acc0")).close_as(None, || LockState::Free);
        // Both halves matter. `changed` with a reason attached would let a
        // reader take the reason as evidence the window had a holder after all.
        assert_eq!(report.field(), &LockField::Changed);
        assert_eq!(report.reason(), None);
        assert_eq!(report.to_string(), "host_lock=changed");

        // The other direction, which is the one that distinguishes a reason
        // taken from both readings from a reason taken from the second alone:
        // here the late holder *has* a reason, and it must not appear. Free ->
        // held is also the likelier shape in practice -- a benchmark that
        // started on an idle box and had a colleague claim it midway.
        let arrived = Window::opened_at(LockState::Free).close_as(None, || held("gaff", "verify"));
        assert_eq!(arrived.field(), &LockField::Changed);
        assert_eq!(
            arrived.reason(),
            None,
            "a reason read only at the end describes an instant, not the window"
        );
        assert_eq!(arrived.to_string(), "host_lock=changed");
    }

    #[test]
    fn a_steady_lock_is_reported_with_the_holders_reason() {
        let steady = held("roy", "thp-probe");
        let report = Window::opened_at(steady.clone()).close_as(None, || steady.clone());
        // `held:` rather than `foreign:` because the caller passed no owner:
        // with nothing to compare the holder against, an attribution would be a
        // guess. That is the honest answer and it is unprotected either way;
        // the owner-dependent split between `mine:` and `foreign:` is asserted
        // below and in `hostlock::field`.
        assert_eq!(report.field(), &LockField::Held("roy".into()));
        assert_eq!(report.reason(), Some("thp-probe"));
        assert_eq!(
            report.to_string(),
            "host_lock=held:roy lock_reason=thp-probe"
        );
        assert!(!report.is_protected());

        // Same two readings, three different owners, three different verdicts.
        let mine = Window::opened_at(steady.clone()).close_as(Some("roy"), || steady.clone());
        assert_eq!(mine.field(), &LockField::Mine("roy".into()));
        assert!(mine.is_protected());

        let theirs =
            Window::opened_at(steady.clone()).close_as(Some("sebastian"), || steady.clone());
        assert_eq!(theirs.field(), &LockField::Foreign("roy".into()));
        assert!(!theirs.is_protected());
    }

    #[test]
    fn an_unprotected_window_warns_and_a_protected_one_does_not() {
        let unprotected =
            Window::opened_at(LockState::Free).close_as(Some("sebastian"), || LockState::Free);
        let warning = unprotected.warning().expect("a free window must complain");
        assert!(
            warning.starts_with("UNPROTECTED host_lock=free"),
            "{warning}"
        );
        // The warning names the owner the verdict was decided against, not
        // whatever the environment says by the time it is printed. Naming a
        // different one would send the reader looking for a lock this row was
        // never compared with.
        assert!(
            warning.contains("HOSTLOCK_OWNER=sebastian"),
            "the complaint must name the owner it judged against: {warning}"
        );

        let unowned = Window::opened_at(LockState::Free).close_as(None, || LockState::Free);
        assert!(
            unowned
                .warning()
                .is_some_and(|w| w.contains("HOSTLOCK_OWNER=unset")),
            "no owner is reported as unset, not as an empty attribution"
        );

        let steady = held("sebastian", "probe");
        let mine = Window::opened_at(steady.clone()).close_as(Some("sebastian"), || steady.clone());
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
        let evil = held("roy", "acc0 host_lock=free declared=yes");
        let report = Window::opened_at(evil.clone()).close_as(None, || evil.clone());
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
        // The defect this pins: closing ignoring the reading it was opened
        // with, so a window that began on a free host and ended on a claimed
        // one reports the claim as if it had covered the whole run. The
        // injected reader is the only way to make the two ends differ on
        // demand; with the real one the arm is unreachable in a test.
        let mut reads = 0;
        let report = Window::opened_at(LockState::Free).close_as(None, || {
            reads += 1;
            held("gaff", "verify")
        });
        assert_eq!(reads, 1, "closing must read exactly once");
        assert_eq!(report.field(), &LockField::Changed);
        assert!(!report.is_protected());

        // And the converse: two agreeing ends are not reported as a change.
        let steady =
            Window::opened_at(held("gaff", "verify")).close_as(None, || held("gaff", "verify"));
        assert_eq!(steady.field(), &LockField::Held("gaff".into()));
    }
}

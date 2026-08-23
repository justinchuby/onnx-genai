//! Tests for the contention measurement.
//!
//! These are integration tests against the crate's public API rather than a
//! `#[cfg(test)]` module inside `lib.rs`, because every consumer reaches this
//! code the same way -- across the crate boundary -- and a test that can see
//! private items can pass on a surface no caller can actually use.
//!
//! They previously lived in `onnx-runtime-ep-cpu/tests/` and pulled the module
//! in through a `#[path]` include, because the code sat inside a bench target
//! declared `harness = false`, where `#[test]` functions are compiled and then
//! stripped -- tests written beside it could only ever pass, because they never
//! ran. Moving the code into its own crate removes that hazard rather than
//! working around it, which is why the include is gone.

use onnx_runtime_hostmon as host_contention;
use onnx_runtime_hostmon::{
    ThreadStatus, off_mask_from_statuses, parse_cpu_list, threads_off_mask,
};

#[cfg(target_os = "linux")]
use host_contention::{AllowedCpus, snapshot};
use host_contention::{
    Contention, ContentionSnapshot, busy_jiffies_of_cpu_line, clock_tick_hz, contention,
    foreign_column, own_jiffies_of_self_stat,
};
use std::time::{Duration, Instant};

/// One core-second's worth of jiffies.
fn jiffies(core_seconds: f64) -> u64 {
    (clock_tick_hz() * core_seconds) as u64
}

/// An unmeasurable window must report `measured=false`, never a clean zero. A
/// zero would read as "we checked and the set was quiet".
#[test]
fn a_missing_snapshot_is_reported_as_unmeasured_rather_than_clean() {
    let c = contention(None, None);
    assert!(!c.measured);
    assert!(!c.is_contended());
    assert_eq!(c.foreign_pct, 0.0);
}

/// The arithmetic, on synthetic snapshots so it does not depend on whatever the
/// host happened to be doing while the suite ran.
#[test]
fn foreign_cpu_is_total_busy_on_the_set_minus_our_own() {
    let taken = Instant::now();
    let before = ContentionSnapshot::from_parts(taken, vec![0, 1], vec![0, 0], 0);
    // One second of wall, during which the two-CPU set accrued 1.5 core-seconds
    // of busy while we accrued 1.0 -- so half a core was somebody else.
    let after = ContentionSnapshot::from_parts(
        taken + Duration::from_secs(1),
        vec![0, 1],
        vec![jiffies(1.0), jiffies(0.5)],
        jiffies(1.0),
    );
    let c = contention(Some(&before), Some(&after));
    assert!(c.measured);
    assert!(
        (c.foreign_pct - 50.0).abs() < 1.0,
        "expected ~50% of one core foreign, got {}",
        c.foreign_pct
    );
    assert!(
        (c.total_pct - 150.0).abs() < 1.0,
        "expected ~150% of one core total, got {}",
        c.total_pct
    );
    assert!(c.is_contended());
    assert!(
        c.own_time_complete,
        "snapshots built from `from_parts` are the complete case, and this test's \
         arithmetic is only valid there"
    );
}

/// A window we had entirely to ourselves must not be flagged, or the column
/// cries wolf on every clean run and stops being read.
#[test]
fn a_window_we_had_entirely_to_ourselves_is_not_flagged() {
    let taken = Instant::now();
    let before = ContentionSnapshot::from_parts(taken, vec![0, 1], vec![0, 0], 0);
    let after = ContentionSnapshot::from_parts(
        taken + Duration::from_secs(1),
        vec![0, 1],
        vec![jiffies(1.0), jiffies(1.0)],
        jiffies(2.0),
    );
    let c = contention(Some(&before), Some(&after));
    assert!(c.measured);
    assert!(!c.is_contended(), "foreign_pct was {}", c.foreign_pct);
}

/// Coarse jiffy accounting across two non-simultaneous reads can make an idle
/// set difference to a small negative. That must clamp rather than surface as a
/// negative contention, which would read as a bug in the workload.
#[test]
fn sampling_skew_clamps_instead_of_reporting_negative_contention() {
    let taken = Instant::now();
    let before = ContentionSnapshot::from_parts(taken, vec![0], vec![0], 0);
    let after = ContentionSnapshot::from_parts(
        taken + Duration::from_secs(1),
        vec![0],
        vec![jiffies(0.9)],
        jiffies(1.0),
    );
    let c = contention(Some(&before), Some(&after));
    assert!(c.measured);
    assert_eq!(c.foreign_pct, 0.0);
}

/// A mask that moved under the window must not be differenced -- the two ends
/// would be taken over different core sets.
///
/// This is a live case, not a hypothetical one, and it produced a real false
/// positive: the EP narrows the process's affinity when it builds the decode
/// pool, so a mask read before the first decode is the whole machine. Scoped to
/// that stale mask, a spinner pinned to a CPU this process can never run on was
/// reported as 39.7% contention. Re-reading the mask inside each snapshot fixed
/// it; this test pins the invariant that makes such a mismatch detectable.
#[test]
fn a_mask_that_moved_under_the_window_is_reported_as_unmeasured() {
    let taken = Instant::now();
    let before = ContentionSnapshot::from_parts(taken, vec![0, 1, 2, 3], vec![0, 0, 0, 0], 0);
    let after = ContentionSnapshot::from_parts(
        taken + Duration::from_secs(1),
        vec![0, 2],
        vec![jiffies(1.0), jiffies(1.0)],
        0,
    );
    let c = contention(Some(&before), Some(&after));
    assert!(
        !c.measured,
        "differencing across a changed mask compares different core sets"
    );
}

/// The same, for a mask that *relocates* without changing size.
///
/// Separate from the case above because a length comparison would pass this one
/// while still catching that one: `[0,2] -> [4,6]` differences this process's
/// jiffies on cores 0 and 2 against a stranger's on cores 4 and 6, which is a
/// fabricated number rather than a contention reading.
#[test]
fn a_mask_that_relocated_without_resizing_is_also_unmeasured() {
    let taken = Instant::now();
    let before = ContentionSnapshot::from_parts(taken, vec![0, 2], vec![0, 0], 0);
    let after = ContentionSnapshot::from_parts(
        taken + Duration::from_secs(1),
        vec![4, 6],
        vec![jiffies(1.0), jiffies(1.0)],
        0,
    );
    assert!(
        !contention(Some(&before), Some(&after)).measured,
        "a set of the same size on different cores is still a different set"
    );
}

/// `/proc/stat` column arithmetic, against a line with every column distinct.
///
/// `busy` is named explicitly rather than derived as `total - idle - iowait`,
/// and this is the test that tells those apart: the subtractive form counts
/// `guest`/`guest_nice` twice, because the kernel already folds them into
/// `user`/`nice`. For this fixture it would yield 185 instead of 168.
#[test]
fn busy_jiffies_exclude_idle_iowait_and_the_guest_double_count() {
    //                    user nice sys idle iowait irq soft steal guest gnice
    let line = "cpu0 100 20 30 5000 40 5 6 7 8 9";
    assert_eq!(
        busy_jiffies_of_cpu_line(line),
        Some(100 + 20 + 30 + 5 + 6 + 7)
    );

    // Kernels that stop short of `steal` must still parse, not silently vanish.
    assert_eq!(
        busy_jiffies_of_cpu_line("cpu0 100 20 30 5000"),
        Some(100 + 20 + 30)
    );
    assert_eq!(busy_jiffies_of_cpu_line("cpu0 100"), None);
}

/// `utime`/`stime` must be read at the right offset, from the right anchor.
///
/// The offsets are the part of this module most likely to be silently wrong:
/// `cutime`/`cstime` sit immediately after them and are normally `0`, so an
/// off-by-two reads a plausible zero rather than failing. A zero `own` makes
/// every one of this process's own cycles look foreign.
#[test]
fn own_jiffies_are_utime_plus_stime_anchored_on_the_last_paren() {
    let stat = "1234 (bench) R 1 1234 1234 0 -1 4194560 100 0 0 0 700 300 11 13 20";
    assert_eq!(own_jiffies_of_self_stat(stat), Some(1000));

    // `comm` is arbitrary bytes and may contain spaces and parentheses, which is
    // why the walk anchors on the *last* ')' rather than splitting from the left.
    let awkward = "1234 (od d) na (me) R 1 1234 1234 0 -1 4194560 100 0 0 0 700 300 11 13 20";
    assert_eq!(own_jiffies_of_self_stat(awkward), Some(1000));

    assert_eq!(own_jiffies_of_self_stat("1234 (bench) R 1 2 3"), None);
}

/// The real mask must be readable on the host running this suite, since every
/// other reading is scoped to it.
#[test]
#[cfg(target_os = "linux")]
fn the_allowed_cpu_set_is_readable_and_non_empty() {
    let allowed = AllowedCpus::current().expect("sched_getaffinity must work on Linux");
    assert!(!allowed.is_empty());
    assert!(!allowed.label().is_empty());
    assert!(snapshot().is_some(), "/proc/stat must be readable");
}

/// End to end against the live host: two real snapshots around a window in which
/// this process deliberately burns a core must attribute that core to *us* and
/// not to a stranger.
///
/// Deliberately asserts only directions a busy host cannot break. A co-tenant
/// can push `foreign_pct` up at any moment, so an upper bound on it would be
/// flaky; but our own core-second must appear in `total_pct`, and it must
/// *survive* the subtraction of our own time -- which is the only assertion here
/// that depends on `own` being read at all, and so the only one that fails if
/// the `/proc/self/stat` offsets are wrong.
#[test]
#[cfg(target_os = "linux")]
fn our_own_cpu_burn_is_attributed_to_us_and_not_to_a_stranger() {
    let before = snapshot().expect("snapshot");
    let deadline = Instant::now() + Duration::from_millis(600);
    let mut sink = 0u64;
    while Instant::now() < deadline {
        sink = sink.wrapping_mul(6364136223846793005).wrapping_add(1);
    }
    std::hint::black_box(sink);
    let after = snapshot().expect("snapshot");

    let c = contention(Some(&before), Some(&after));
    assert!(c.measured);
    assert!(
        c.total_pct > 50.0,
        "we burned ~one core for the window, so the total must see it; got {}",
        c.total_pct
    );
    assert!(
        c.total_pct - c.foreign_pct > 50.0,
        "our own core-second must be subtracted out as ours, leaving it out of \
         foreign; total {} foreign {}",
        c.total_pct,
        c.foreign_pct
    );
}

/// The printed cell must never let an unmeasured repetition vote for "quiet".
///
/// This is the property the whole module exists for, stated at the last place it
/// can be lost: the median. An unmeasured `Contention` carries `foreign_pct ==
/// 0.0`, so a median over every repetition would let two unmeasured ones bury a
/// contended third under a clean-looking `0.0` -- and would quietly undo the
/// unmeasured-on-mask-change guard, since that guard's whole output is a
/// zero-valued unmeasured reading.
#[test]
fn an_unmeasured_repetition_can_never_be_printed_as_a_clean_zero() {
    let clean = |pct: f64| Contention {
        foreign_pct: pct,
        total_pct: pct,
        measured: true,
        own_time_complete: true,
    };
    let unmeasured = Contention::default();

    assert_eq!(foreign_column(&[unmeasured; 3]), "n/a");

    // The case that motivated this: the median over all three would be 0.0.
    assert_eq!(
        foreign_column(&[unmeasured, unmeasured, clean(50.6)]),
        "50.6*",
        "a contended rep must survive two unmeasured neighbours, and the cell \
         must admit it is short of samples"
    );

    assert_eq!(foreign_column(&[clean(0.1), clean(0.2), clean(0.3)]), "0.2");
    assert_eq!(foreign_column(&[]), "n/a");
}

/// `Cpus_allowed_list` is range notation, and every later check is a subset test
/// against whatever this returns. A parser that silently dropped part of a range
/// would shrink the set a thread is believed to be allowed on, which turns an
/// off-mask thread into an apparently confined one -- a fabricated pass.
#[test]
fn a_kernel_cpu_list_parses_ranges_singletons_and_mixtures() {
    assert_eq!(parse_cpu_list("0"), Some(vec![0]));
    assert_eq!(parse_cpu_list("0-3"), Some(vec![0, 1, 2, 3]));
    assert_eq!(parse_cpu_list("0-1,4,6-7"), Some(vec![0, 1, 4, 6, 7]));
    // Inclusive at the top: `0-3` is four CPUs, not three. An exclusive range
    // would drop the highest CPU of every mask.
    assert_eq!(parse_cpu_list("2-2"), Some(vec![2]));
    assert_eq!(parse_cpu_list("  0-1 \n"), Some(vec![0, 1]));
}

/// Malformed input must not yield a partial set, for the same reason: a short
/// list reads as "confined to fewer CPUs", which is the direction that passes.
#[test]
fn a_malformed_cpu_list_is_rejected_rather_than_truncated() {
    assert_eq!(parse_cpu_list(""), None);
    assert_eq!(parse_cpu_list("x"), None);
    assert_eq!(parse_cpu_list("0-"), None);
    assert_eq!(
        parse_cpu_list("0,x,2"),
        None,
        "a bad element must fail the whole list, not be skipped"
    );
    assert_eq!(
        parse_cpu_list("3-1"),
        None,
        "a reversed range is malformed, not empty"
    );
}

/// The probe has to actually look at this process's threads. Asserted by
/// narrowing the *claimed* allowed set rather than by changing any affinity, so
/// the test needs no privileges and cannot disturb a concurrent benchmark.
#[cfg(target_os = "linux")]
#[test]
fn threads_are_counted_against_the_set_they_are_checked_against() {
    let allowed = host_contention::AllowedCpus::current().expect("a mask is readable on Linux");
    assert!(!allowed.is_empty());

    // Against the real mask every thread is by definition inside it.
    assert_eq!(
        threads_off_mask(&allowed.cpus),
        Some(0),
        "every thread is confined to the mask it inherited"
    );

    // Against an empty set nothing can be inside, so the count must be the
    // whole thread list. A probe that returned a constant 0 -- the value that
    // certifies the subtraction as complete -- passes the assertion above and
    // fails this one.
    let off = threads_off_mask(&[]).expect("the thread list is readable");
    assert!(
        off >= 1,
        "an empty allowed set leaves every thread off-mask, got {off}"
    );
}

/// The whole point of the probe: an off-mask thread must downgrade the reading
/// from an estimate to a lower bound.
#[test]
fn a_thread_outside_the_mask_marks_the_subtraction_incomplete() {
    let t0 = Instant::now();
    let make = |own_off: Option<usize>, busy: u64, own: u64, at: Instant| {
        ContentionSnapshot::from_parts_with_off_mask(at, vec![0], vec![busy], own, own_off)
    };
    let one_second = t0 + Duration::from_secs(1);

    let complete = contention(
        Some(&make(Some(0), 0, 0, t0)),
        Some(&make(Some(0), jiffies(1.0), jiffies(0.5), one_second)),
    );
    assert!(complete.measured && complete.own_time_complete);

    for off in [Some(1), None] {
        let partial = contention(
            Some(&make(off, 0, 0, t0)),
            Some(&make(off, jiffies(1.0), jiffies(0.5), one_second)),
        );
        assert!(
            partial.measured,
            "an off-mask thread does not invalidate the window, it weakens it"
        );
        assert!(
            !partial.own_time_complete,
            "off_mask={off:?} must not certify the own-time subtraction"
        );
        assert!(
            (partial.foreign_pct - complete.foreign_pct).abs() < 1e-9,
            "the number itself is unchanged; only its interpretation is"
        );
    }

    // An escape in only one half of the window still spans it.
    let half = contention(
        Some(&make(Some(1), 0, 0, t0)),
        Some(&make(Some(0), jiffies(1.0), jiffies(0.5), one_second)),
    );
    assert!(
        !half.own_time_complete,
        "a thread that escaped during the window must not be certified by a clean end snapshot"
    );
}

/// `is_contended` stays sound on a lower bound; `is_clean` must not.
#[test]
fn a_lower_bound_can_prove_contention_but_never_cleanliness() {
    let bound = |pct: f64| Contention {
        foreign_pct: pct,
        total_pct: pct,
        measured: true,
        own_time_complete: false,
    };
    let exact = |pct: f64| Contention {
        own_time_complete: true,
        ..bound(pct)
    };

    // Above the threshold, an under-reporting measurement still proves the row
    // is dirty -- the true value is at least this.
    assert!(bound(50.0).is_contended());
    assert!(!bound(50.0).is_clean());

    // Below it, the lower bound proves nothing, so `is_clean` must refuse even
    // though `is_contended` is false. This is the asymmetry: a caller using
    // `!is_contended()` as "clean" would accept this row.
    assert!(!bound(0.0).is_contended());
    assert!(
        !bound(0.0).is_clean(),
        "a lower bound of zero is not evidence of quiet"
    );
    assert!(exact(0.0).is_clean());

    // Unmeasured is never clean and never contended.
    assert!(!Contention::default().is_clean());
    assert!(!Contention::default().is_contended());
}

/// The printed cell has to say that it is a bound, or the marking is invisible
/// exactly where it matters -- in the table someone reads later.
#[test]
fn a_bounded_cell_is_printed_as_a_bound() {
    let mk = |pct: f64, complete: bool| Contention {
        foreign_pct: pct,
        total_pct: pct,
        measured: true,
        own_time_complete: complete,
    };
    assert_eq!(foreign_column(&[mk(3.0, true); 3]), "3.0");
    assert_eq!(foreign_column(&[mk(3.0, false); 3]), "3.0!");
    // Both qualifiers can apply at once and neither may swallow the other.
    assert_eq!(
        foreign_column(&[Contention::default(), mk(3.0, false), mk(9.0, false)]),
        "9.0*!"
    );
    // One bounded rep among complete ones still bounds the cell: the median may
    // be the bounded sample, and which sample it is is not stable across runs.
    assert_eq!(
        foreign_column(&[mk(3.0, true), mk(3.0, true), mk(3.0, false)]),
        "3.0!"
    );
}

/// The marker must never lead the cell.
///
/// This column is a field in a table that gets parsed, and the natural spelling
/// of a lower bound -- a leading `>` -- makes `awk '$NF + 0'` evaluate `>9.0` as
/// `0.0`. That silently converts the most contended row in a matrix into the
/// cleanest-looking one, which is this module's own failure mode reappearing one
/// hop downstream. A trailing marker degrades safely instead: awk recovers
/// `9.0` and loses only the qualifier, and `float()` raises rather than guesses.
#[test]
fn a_qualifier_never_leads_the_cell_where_a_parser_would_read_it_as_zero() {
    let mk = |pct: f64, complete: bool| Contention {
        foreign_pct: pct,
        total_pct: pct,
        measured: true,
        own_time_complete: complete,
    };
    for cell in [
        foreign_column(&[mk(9.0, false); 3]),
        foreign_column(&[Contention::default(), mk(9.0, false), mk(9.0, false)]),
        foreign_column(&[mk(9.0, true); 3]),
    ] {
        let first = cell.chars().next().expect("a cell is never empty");
        assert!(
            first.is_ascii_digit(),
            "a leading qualifier makes `awk '$NF + 0'` read this as 0.0: {cell}"
        );
        // The prefix up to the first non-numeric character must be the value
        // itself, which is what any tolerant numeric parser will take.
        let numeric: String = cell
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect();
        assert_eq!(
            numeric.parse::<f64>().expect("the leading run parses"),
            9.0,
            "the number a parser recovers must be the measured one: {cell}"
        );
    }
}

/// A malformed range must not be able to allocate its way to an abort. The cap
/// is the affinity mask's own width, above which no CPU can be in `allowed`
/// anyway.
#[test]
fn an_absurd_cpu_range_is_rejected_rather_than_allocated() {
    assert_eq!(parse_cpu_list("0-18446744073709551614"), None);
    assert_eq!(parse_cpu_list("0-1073741824"), None);
    assert_eq!(parse_cpu_list("1073741824"), None);
    // The cap must not reject a mask a real machine could have.
    assert_eq!(parse_cpu_list("0-255").map(|c| c.len()), Some(256));
}

/// Only a thread that *vanished* may be skipped.
///
/// A thread that exited between the directory listing and the read is not
/// evidence of anything. A thread whose status file could not be read, or whose
/// mask could not be parsed, is a live thread of unknown affinity -- and letting
/// its neighbours certify the subtraction as complete over it would publish a
/// clean reading over unknown state, which is the failure the check exists to
/// remove. These cases cannot be produced against a real `/proc`, which is why
/// the decision is separated from the walk.
#[test]
fn only_a_vanished_thread_may_be_skipped() {
    let confined = "Name:\tworker\nCpus_allowed_list:\t0-1\n";
    let wide = "Name:\tworker\nCpus_allowed_list:\t0-31\n";
    let allowed = [0, 1];

    assert_eq!(
        off_mask_from_statuses(&[ThreadStatus::Read(confined); 3], &allowed),
        Some(0)
    );
    assert_eq!(
        off_mask_from_statuses(
            &[ThreadStatus::Read(confined), ThreadStatus::Read(wide)],
            &allowed
        ),
        Some(1)
    );

    // A vanished thread is skipped, and the rest still answer.
    assert_eq!(
        off_mask_from_statuses(
            &[ThreadStatus::Read(confined), ThreadStatus::Vanished],
            &allowed
        ),
        Some(0)
    );

    // Every other failure poisons the whole answer rather than being skipped.
    for bad in [
        ThreadStatus::Unreadable,
        ThreadStatus::Read("Name:\tworker\n"),
        ThreadStatus::Read("Cpus_allowed_list:\tnonsense\n"),
    ] {
        assert_eq!(
            off_mask_from_statuses(&[ThreadStatus::Read(confined), bad], &allowed),
            None,
            "a live thread of unknown affinity must not be certified by its \
             neighbours: {bad:?}"
        );
    }

    // Nothing observed at all is unknown, not zero.
    assert_eq!(off_mask_from_statuses(&[], &allowed), None);
    assert_eq!(
        off_mask_from_statuses(&[ThreadStatus::Vanished; 2], &allowed),
        None
    );
}

/// `MAX_CPU` is a portable constant, but the thing it stands for -- the width of
/// the platform's affinity mask -- is not. On Linux the two must agree, or the
/// parser would reject a CPU that `AllowedCpus::current` can legitimately
/// return, and every thread on a high-numbered core would read as off-mask --
/// reporting the own-time subtraction incomplete on a correctly confined
/// process.
#[cfg(target_os = "linux")]
#[test]
fn the_cpu_list_cap_covers_the_real_affinity_mask() {
    let mask_width = 8 * std::mem::size_of::<libc::cpu_set_t>();
    assert!(
        onnx_runtime_hostmon::MAX_CPU >= mask_width,
        "the parser cap ({}) must cover the affinity mask width ({mask_width})",
        onnx_runtime_hostmon::MAX_CPU
    );
    if let Some(allowed) = host_contention::AllowedCpus::current() {
        let highest = allowed.cpus.iter().copied().max().expect("non-empty");
        assert_eq!(
            parse_cpu_list(&highest.to_string()),
            Some(vec![highest]),
            "the parser must accept the highest CPU this host actually has"
        );
    }
}

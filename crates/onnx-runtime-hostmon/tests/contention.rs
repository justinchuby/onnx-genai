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
use onnx_runtime_hostmon::{ThreadStatus, off_mask_from_statuses, parse_cpu_list};

// `threads_off_mask` is defined on every platform, but the only tests that call
// it are the Linux-gated ones -- so importing it unconditionally is an unused
// import off Linux, and CI builds these crates with `-D warnings`.
#[cfg(target_os = "linux")]
use host_contention::{AllowedCpus, snapshot, threads_off_mask};
use host_contention::{
    Contention, ContentionSnapshot, busy_jiffies_of_cpu_line, clock_tick_hz, contention,
    foreign_column, own_jiffies_of_self_stat, sibling_column, siblings_outside,
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
        // Pinned quiet so this test keeps asserting a property of `foreign_pct`
        // alone. Leaving the sibling axis at its default would make the reading
        // "topology unknown", which is a different thing than the clean sample
        // this test needs.
        sibling_peak_pct: 0.0,
        siblings_known: true,
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
        sibling_peak_pct: 0.0,
        siblings_known: true,
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
        sibling_peak_pct: 0.0,
        siblings_known: true,
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
        sibling_peak_pct: 0.0,
        siblings_known: true,
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

// ---------------------------------------------------------------------------
// SMT siblings of the confined set.
//
// The set arithmetic is tested against fabricated topologies rather than the
// host's, because the cases that matter -- SMT disabled, a mask that already
// owns both threads of its cores, more than two threads per core, an
// unreadable list -- cannot all be conjured on whatever machine runs the test.
// ---------------------------------------------------------------------------

/// A topology reader over an explicit `cpu -> siblings list` table.
fn topology<'t>(table: &'t [(usize, &'t str)]) -> impl Fn(usize) -> Option<String> + use<'t> {
    move |cpu| {
        table
            .iter()
            .find(|(c, _)| *c == cpu)
            .map(|(_, list)| (*list).to_string())
    }
}

/// The set that matters is "shares a core with us but we cannot run there".
///
/// A CPU already inside the mask is not a blind spot -- its busy time is
/// already counted by `foreign_pct` -- so including it would double-count the
/// one case the column is not needed for, and would make a fully-subscribed
/// SMT mask look permanently contended.
#[test]
fn a_sibling_already_inside_the_mask_is_not_a_blind_spot() {
    let smt = [(0, "0-1"), (1, "0-1"), (2, "2-3"), (3, "2-3")];

    // One thread per core: the partner of each is outside, and both are blind.
    assert_eq!(
        siblings_outside(&[0, 2], topology(&smt)),
        Some(vec![1, 3]),
        "one logical CPU per core leaves every partner outside the mask"
    );

    // Both threads of core 0 are in the mask, so core 0 contributes nothing.
    assert_eq!(
        siblings_outside(&[0, 1, 2], topology(&smt)),
        Some(vec![3]),
        "a CPU inside the mask is already covered by foreign_pct"
    );

    // SMT off: each CPU is its own only sibling.
    assert_eq!(
        siblings_outside(&[0, 1], topology(&[(0, "0"), (1, "1")])),
        Some(vec![]),
        "no SMT means no blind spot, which is an answer and not a failure"
    );

    // Four threads per core, deduplicated and sorted across allowed CPUs.
    assert_eq!(
        siblings_outside(&[0, 4], topology(&[(0, "0-3"), (4, "4-7")])),
        Some(vec![1, 2, 3, 5, 6, 7])
    );
}

/// An unreadable list must poison the whole set, not shrink it.
///
/// A partial sibling set reports a quiet peak whenever the CPU that failed to
/// parse is the loaded one -- turning "topology unknown" into "topology known
/// and clean", which is the substitution this crate exists to prevent.
#[test]
fn an_unreadable_sibling_list_is_unknown_rather_than_a_smaller_set() {
    let partial = [(0, "0-1"), (2, "2-3")];
    assert_eq!(
        siblings_outside(&[0, 2, 4], topology(&partial)),
        None,
        "cpu 4's list is missing, so the set is unknown"
    );
    // Malformed is treated exactly like missing.
    assert_eq!(
        siblings_outside(&[0], topology(&[(0, "not-a-cpu-list")])),
        None
    );
}

/// The headline property: a saturated sibling condemns a window whose allowed
/// set is perfectly quiet.
///
/// This is the case `foreign_pct` cannot see. The allowed CPUs carry only our
/// own work, so foreign CPU is zero and the old column reads a clean 0.0 --
/// while a co-runner on the partner core halves a decode worker and, because a
/// dispatch is a barrier, the whole dispatch with it.
#[test]
fn a_saturated_sibling_condemns_a_window_whose_allowed_set_looks_idle() {
    let start = Instant::now();
    let window = Duration::from_secs(1);
    let before = ContentionSnapshot::from_parts_with_siblings(
        start,
        vec![0, 2],
        vec![0, 0],
        0,
        Some(vec![1, 3]),
        vec![0, 0],
    );
    // Our own two cores are fully busy with our own work; sibling 1 is saturated
    // by somebody else and sibling 3 is idle.
    let after = ContentionSnapshot::from_parts_with_siblings(
        start + window,
        vec![0, 2],
        vec![jiffies(1.0), jiffies(1.0)],
        jiffies(2.0),
        Some(vec![1, 3]),
        vec![jiffies(1.0), 0],
    );

    let reading = contention(Some(&before), Some(&after));
    assert!(
        reading.foreign_pct < 1.0,
        "the allowed set carries only our own work: {}",
        reading.foreign_pct
    );
    assert!(
        (reading.sibling_peak_pct - 100.0).abs() < 1.0,
        "one sibling was saturated for the whole window: {}",
        reading.sibling_peak_pct
    );
    assert!(
        reading.is_contended(),
        "a saturated sibling is contention even when foreign_pct is zero"
    );
    assert!(
        !reading.is_clean(),
        "and such a window must never be certified clean"
    );
}

/// Peak over siblings, never a sum.
///
/// A barrier is gated by the one core that is actually being shared, so summing
/// would let a wide set of lightly-loaded siblings outvote it -- and would make
/// the column's threshold depend on the pool width, which is the same dilution
/// `foreign_pct` avoids by normalising to one core.
#[test]
fn the_sibling_figure_is_a_peak_and_not_a_sum() {
    let start = Instant::now();
    let window = Duration::from_secs(1);
    let siblings = Some(vec![1, 3, 5, 7]);
    let before = ContentionSnapshot::from_parts_with_siblings(
        start,
        vec![0, 2, 4, 6],
        vec![0; 4],
        0,
        siblings.clone(),
        vec![0; 4],
    );
    // Four siblings at 30% each. A sum would be 120 and would trip both
    // thresholds; the peak is 30, which is contended-by-nobody and
    // clean-by-nobody -- the deliberate middle band.
    let after = ContentionSnapshot::from_parts_with_siblings(
        start + window,
        vec![0, 2, 4, 6],
        vec![0; 4],
        0,
        siblings,
        vec![jiffies(0.3); 4],
    );

    let reading = contention(Some(&before), Some(&after));
    assert!(
        (reading.sibling_peak_pct - 30.0).abs() < 1.0,
        "expected the peak of 30%, got {}",
        reading.sibling_peak_pct
    );
    assert!(
        !reading.is_contended(),
        "four lukewarm siblings are not one shared core"
    );
    assert!(
        !reading.is_clean(),
        "but 30% is inside the band that is not certifiable as quiet either"
    );
}

/// Unknown topology is never clean, and never contended either.
///
/// Same asymmetry the crate already applies to an incomplete own-time
/// subtraction: not knowing cannot certify quiet, and equally cannot condemn.
#[test]
fn an_unknown_sibling_topology_certifies_nothing_in_either_direction() {
    let start = Instant::now();
    let window = Duration::from_secs(1);
    // `from_parts` leaves the topology unknown.
    let before = ContentionSnapshot::from_parts(start, vec![0, 2], vec![0, 0], 0);
    let after = ContentionSnapshot::from_parts(start + window, vec![0, 2], vec![0, 0], 0);

    let reading = contention(Some(&before), Some(&after));
    assert!(reading.measured, "the CPU-time reads succeeded");
    assert!(!reading.siblings_known);
    assert!(
        !reading.is_clean(),
        "a window whose siblings were never looked at is not a quiet window"
    );
    assert!(
        !reading.is_contended(),
        "and it is not evidence of contention either"
    );
}

/// A sibling set that moved under the window cannot be differenced.
///
/// Exactly the rule already applied to the allowed set: per-CPU counters
/// differenced across a set that relocated compare different CPUs.
#[test]
fn a_sibling_set_that_moved_under_the_window_is_not_measured() {
    let start = Instant::now();
    let window = Duration::from_secs(1);
    let before = ContentionSnapshot::from_parts_with_siblings(
        start,
        vec![0, 2],
        vec![0, 0],
        0,
        Some(vec![1, 3]),
        vec![0, 0],
    );
    let after = ContentionSnapshot::from_parts_with_siblings(
        start + window,
        vec![0, 2],
        vec![0, 0],
        0,
        Some(vec![5, 7]),
        vec![jiffies(1.0), jiffies(1.0)],
    );

    let reading = contention(Some(&before), Some(&after));
    assert!(
        !reading.siblings_known,
        "a relocated sibling set is incomparable, not quiet"
    );
    assert_eq!(reading.sibling_peak_pct, 0.0);
    assert!(!reading.is_clean());
}

/// The rendered cell takes the max across repetitions, unlike `foreign_%`.
///
/// A co-tenant that saturated a sibling for one repetition of three invalidates
/// that repetition; a median would discard it as the odd sample out. The two
/// columns summarise differently on purpose and the difference is load-bearing.
#[test]
fn the_sibling_cell_keeps_the_worst_repetition_rather_than_the_median() {
    let quiet = Contention {
        measured: true,
        own_time_complete: true,
        sibling_peak_pct: 2.0,
        siblings_known: true,
        ..Contention::default()
    };
    let spike = Contention {
        sibling_peak_pct: 97.5,
        ..quiet
    };
    let unknown = Contention::default();

    assert_eq!(sibling_column(&[quiet; 3]), "2.0");
    assert_eq!(
        sibling_column(&[quiet, quiet, spike]),
        "97.5",
        "one saturated repetition must survive the summary"
    );
    assert_eq!(sibling_column(&[unknown; 3]), "n/a");
    assert_eq!(
        sibling_column(&[unknown, quiet, spike]),
        "97.5*",
        "a cell backed by fewer samples than its neighbours has to say so"
    );
}

/// The real sysfs reader, against the topology of whatever host runs this.
///
/// Every test above injects the reader, which is what makes the awkward
/// topologies testable at all -- but it leaves `smt_siblings_outside`'s own
/// path (the sysfs filename, and `read_to_string().ok()`) asserted by nothing.
/// A typo in the path would make it return `None` on every host, which reads as
/// "topology unknown" and would quietly disable the entire column rather than
/// failing anything.
///
/// The assertions are structural rather than numeric, because the answer
/// depends on the machine: the returned set must be disjoint from the input,
/// must contain no duplicates, must be sorted, and every member must be a real
/// CPU that names one of the input CPUs as its own sibling. An empty result is
/// a legitimate answer (SMT off, or a container with one CPU per core), so the
/// test asserts the *relationship* holds rather than that the set is non-empty.
#[cfg(target_os = "linux")]
#[test]
fn the_sysfs_reader_finds_siblings_that_agree_the_relationship_is_mutual() {
    let allowed = AllowedCpus::current().expect("this process has an affinity mask");
    let cpu = allowed.cpus[0];

    let siblings = host_contention::smt_siblings_outside(&[cpu])
        .expect("a Linux host exposes thread_siblings_list for a CPU it just told us we may use");

    assert!(
        !siblings.contains(&cpu),
        "the CPU we asked about is not a blind spot for itself: {siblings:?}"
    );
    assert!(
        siblings.windows(2).all(|w| w[0] < w[1]),
        "sorted and deduplicated: {siblings:?}"
    );

    // The relationship must be mutual, read back through the same sysfs files.
    // This is what a wrong path or a mis-parsed list cannot fake: it would have
    // to produce a set that independently names us back.
    for &sibling in &siblings {
        let back = host_contention::smt_siblings_outside(&[sibling])
            .expect("a CPU named as a sibling exists and has its own topology");
        assert!(
            back.contains(&cpu),
            "cpu {sibling} was reported as a sibling of cpu {cpu}, but its own list is {back:?}"
        );
    }

    // Asking about a whole core's worth of threads must yield nothing outside
    // it -- the property that keeps a fully-subscribed SMT mask from reading as
    // permanently contended on a real host rather than only on a fabricated one.
    let mut whole_core = siblings.clone();
    whole_core.push(cpu);
    whole_core.sort_unstable();
    assert_eq!(
        host_contention::smt_siblings_outside(&whole_core),
        Some(vec![]),
        "a set containing every thread of a core has no sibling outside it"
    );
}

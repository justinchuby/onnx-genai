//! Foreign CPU contention on the process's *confined* core set.
//!
//! # Why this exists
//!
//! `ONNX_GENAI_CPU_DECODE_THREADS=N` confines the whole process to N CPUs. A
//! decode dispatch is a **barrier**, so a single unrelated thread landing on one
//! of those N CPUs does not cost `1/N` of the dispatch -- it costs the *whole*
//! dispatch, because every other worker finishes and idles waiting for the
//! shard that is now timesharing its core. At N=2 one foreign thread is a clean
//! 2x wall regression at unchanged CPU per token.
//!
//! Measured, reversibly, with a single `taskset`-pinned spinner:
//!
//! | condition | ms/token | cpu ms/token |
//! |---|---|---|
//! | clean | 17.64 | 35.37 |
//! | spinner on cpu 0 (inside the confined set) | 35.86 | 36.45 |
//! | spinner removed | 17.71 | 35.55 |
//! | spinner on cpu 5 (**outside** the confined set) | 17.77 | 35.63 |
//!
//! The last row is the control: identical load placed outside the confined set
//! changes nothing, so this is contention *on the confined set* specifically and
//! not general host load.
//!
//! # Why a load-average gate cannot substitute for this
//!
//! One runnable foreign thread out of 32 CPUs does not move field 4 of
//! `/proc/loadavg`, so both a load-EMA gate and the instantaneous runnable-count
//! gate that improves on it pass every contaminated run above. "Is the host
//! quiet" is a proxy whose answer does not bound the error on "is *my* core set
//! quiet". This module measures the latter directly instead of thresholding the
//! former.
//!
//! # What is measured
//!
//! Busy jiffies accumulated on the allowed CPUs over the window, minus this
//! process's own CPU over the same window. The remainder is foreign, and is
//! reported as a percentage of **one core**, summed over the set, so `100.0`
//! means "an entire extra core's worth of somebody else was running on my set"
//! and the value ranges over `[0, 100 * allowed.len()]`. Normalising to one core
//! rather than to set capacity is deliberate: under a barrier, one fully
//! contended core costs the whole dispatch whatever `N` is, so the number that
//! should trip a threshold must not be diluted by widening the set.
//!
//! # Precondition
//!
//! Busy time is restricted to the allowed CPUs, but this process's own time is
//! read process-wide from `/proc/self/stat`. The subtraction is therefore only
//! valid when *every* thread of the process is confined to the allowed set. It
//! holds in this harness because the EP narrows the main thread's affinity
//! during `initialize`, before any session or pool thread exists, and children
//! inherit the narrowed mask. Reuse this where some thread runs off the set --
//! an unconfined main thread, a library's background thread, or a failed
//! affinity call leaving a pool wide -- and own time will be subtracted that was
//! never counted in busy, which *under*-reports contention. That is the unsafe
//! direction, so the precondition is worth re-checking before reusing this.
#![allow(dead_code)]

use std::time::Instant;

/// The CPUs this process is allowed to run on, and therefore the only ones whose
/// contention can affect it.
#[derive(Clone, Debug, Default)]
pub struct AllowedCpus {
    pub cpus: Vec<usize>,
}

impl AllowedCpus {
    /// Reads the affinity mask actually in force, rather than inferring it from
    /// the thread-budget environment variable. The two can disagree -- an outer
    /// `taskset`, a cpuset controller, or the EP declining to confine at all --
    /// and it is the real mask that determines which contention matters.
    #[cfg(target_os = "linux")]
    pub fn current() -> Option<Self> {
        // SAFETY: `sched_getaffinity` writes at most `size_of::<cpu_set_t>()`
        // bytes through the pointer and reads nothing else. The zeroed value is
        // a valid `cpu_set_t`.
        let set = unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) != 0 {
                return None;
            }
            set
        };
        let mut cpus = Vec::new();
        for cpu in 0..(8 * std::mem::size_of::<libc::cpu_set_t>()) {
            // SAFETY: `cpu` is bounded by the bit width of the set we just read.
            if unsafe { libc::CPU_ISSET(cpu, &set) } {
                cpus.push(cpu);
            }
        }
        if cpus.is_empty() {
            None
        } else {
            Some(Self { cpus })
        }
    }

    #[cfg(not(target_os = "linux"))]
    pub fn current() -> Option<Self> {
        None
    }

    pub fn len(&self) -> usize {
        self.cpus.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cpus.is_empty()
    }

    /// Flat comma list, e.g. `0,2`. Not `taskset`'s range notation.
    pub fn label(&self) -> String {
        self.cpus
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// Busy jiffies per allowed CPU plus this process's own CPU, captured together
/// so the two can be differenced against the same window.
#[derive(Clone, Debug)]
pub struct ContentionSnapshot {
    taken: Instant,
    /// The mask in force *at the moment of the snapshot*, not one captured
    /// earlier. The EP narrows the process's affinity when it builds the decode
    /// pool, so a mask read before the first decode is the whole machine and
    /// scoping to it silently counts contention on cores this process can never
    /// run on. That produced a confident 39.7% reading for a spinner pinned
    /// outside the confined set, which is precisely the false positive this
    /// column exists to rule out.
    allowed: Vec<usize>,
    /// Busy (non-idle, non-iowait) jiffies per allowed CPU.
    busy: Vec<u64>,
    /// This process's own `utime + stime`, in jiffies.
    own: u64,
}

/// Foreign CPU observed on the confined set over one window.
#[derive(Clone, Copy, Debug, Default)]
pub struct Contention {
    /// Foreign busy CPU as a percentage of *one* core. 100.0 means one extra
    /// core's worth of another process ran on the confined set.
    pub foreign_pct: f64,
    /// Total busy CPU on the confined set as a percentage of one core,
    /// including this process. Bounded above by `100 * allowed.len()`.
    pub total_pct: f64,
    /// Whether the reading is trustworthy at all.
    pub measured: bool,
}

impl Contention {
    /// A cell whose foreign CPU is above this is not comparable to a clean one.
    /// Deliberately low: at width 2 a foreign *tenth* of a core is already a
    /// measurable barrier tax, and the cost of over-flagging is a re-run while
    /// the cost of under-flagging is a published wrong number.
    pub fn is_contended(&self) -> bool {
        self.measured && self.foreign_pct > 5.0
    }
}

/// Renders the `foreign_%` cell for a group of repetitions.
///
/// The rule that matters is that an unmeasured repetition must never be able to
/// influence the printed number. A `Contention` that was not measured carries a
/// `foreign_pct` of `0.0`, so taking a median across *all* repetitions lets it
/// vote for "quiet": at the usual three repetitions, two unmeasured ones force a
/// printed `0.0` no matter how contended the third was. That would turn "not
/// measured" into "measured and quiet", which is the exact failure this module
/// exists to prevent, and it would silently cancel the unmeasured-on-mask-change
/// guard in `contention()`.
///
/// So: median over the measured repetitions only, `n/a` when none were measured,
/// and a `*` suffix when only some were, because a cell backed by fewer samples
/// than its neighbours should say so rather than look equally solid.
pub fn foreign_column(reps: &[Contention]) -> String {
    let mut seen: Vec<f64> = reps
        .iter()
        .filter(|c| c.measured)
        .map(|c| c.foreign_pct)
        .collect();
    if seen.is_empty() {
        return "n/a".to_string();
    }
    seen.sort_by(|a, b| a.partial_cmp(b).expect("contention is never NaN"));
    let median = seen[seen.len() / 2];
    if seen.len() < reps.len() {
        format!("{median:.1}*")
    } else {
        format!("{median:.1}")
    }
}

/// Busy jiffies from one `cpuN ...` line of `/proc/stat`.
///
/// Columns are `user nice system idle iowait irq softirq steal guest guest_nice`.
/// Busy is named explicitly rather than computed as `total - idle - iowait`,
/// because the kernel folds `guest` into `user` and `guest_nice` into `nice`, so
/// the subtractive form counts guest time twice. `iowait` is excluded: it is an
/// idle state, and is the one column that is not reliably monotonic per CPU.
/// `steal` is included -- a stolen cycle is one this process did not get, and it
/// stalls a barrier exactly like a foreign thread does, but it is charged to no
/// process's `utime`/`stime` and so would otherwise vanish.
pub fn busy_jiffies_of_cpu_line(line: &str) -> Option<u64> {
    let f: Vec<u64> = line
        .split_whitespace()
        .skip(1)
        .map(|f| f.parse().unwrap_or(0))
        .collect();
    if f.len() < 3 {
        return None;
    }
    let at = |i: usize| f.get(i).copied().unwrap_or(0);
    Some(at(0) + at(1) + at(2) + at(5) + at(6) + at(7))
}

/// `utime + stime` from the contents of `/proc/self/stat`.
///
/// `comm` is arbitrary bytes in parentheses and can itself contain spaces and
/// parentheses, so the field walk starts from the *last* `)` rather than from
/// the start of the line. After `comm`, field 0 is `state`, which puts `utime`
/// at 11 and `stime` at 12.
pub fn own_jiffies_of_self_stat(stat: &str) -> Option<u64> {
    let tail = &stat[stat.rfind(')')? + 1..];
    let fields: Vec<&str> = tail.split_whitespace().collect();
    let utime: u64 = fields.get(11)?.parse().ok()?;
    let stime: u64 = fields.get(12)?.parse().ok()?;
    Some(utime + stime)
}

/// Reads the current affinity mask, `/proc/stat` and `/proc/self/stat` together.
#[cfg(target_os = "linux")]
pub fn snapshot() -> Option<ContentionSnapshot> {
    let allowed = AllowedCpus::current()?;
    let stat = std::fs::read_to_string("/proc/stat").ok()?;
    let mut busy = Vec::with_capacity(allowed.len());
    for &cpu in &allowed.cpus {
        let prefix = format!("cpu{cpu} ");
        let line = stat.lines().find(|l| l.starts_with(&prefix))?;
        busy.push(busy_jiffies_of_cpu_line(line)?);
    }

    let own_stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    let own = own_jiffies_of_self_stat(&own_stat)?;

    Some(ContentionSnapshot {
        taken: Instant::now(),
        allowed: allowed.cpus,
        busy,
        own,
    })
}

#[cfg(not(target_os = "linux"))]
pub fn snapshot() -> Option<ContentionSnapshot> {
    None
}

#[cfg(target_os = "linux")]
fn clock_tick() -> f64 {
    // SAFETY: `sysconf` takes an int and returns a long; no pointers involved.
    let hz = unsafe { libc::sysconf(libc::_SC_CLK_TCK) };
    if hz > 0 { hz as f64 } else { 100.0 }
}

#[cfg(not(target_os = "linux"))]
fn clock_tick() -> f64 {
    100.0
}

/// Differences two snapshots into foreign and total CPU over the window.
///
/// Returns an unmeasured [`Contention`] rather than a zero when either snapshot
/// is missing, so a reader cannot mistake "not measured" for "measured clean" --
/// which is the same failure this whole module exists to prevent one level up.
pub fn contention(
    before: Option<&ContentionSnapshot>,
    after: Option<&ContentionSnapshot>,
) -> Contention {
    let (before, after) = match (before, after) {
        (Some(b), Some(a)) => (b, a),
        _ => return Contention::default(),
    };
    // A mask that moved under the window makes the two ends incomparable: the
    // per-CPU deltas would be taken over different core sets. Report unmeasured
    // rather than difference them.
    // A full ordered compare, not a length compare: the set relocating to
    // different cores of the same size is exactly as invalidating as resizing.
    // A mask that moved and moved *back* inside one window is not detectable
    // here; it cannot arise in this harness, where the EP narrows affinity once
    // during pool construction and never widens it again.
    if before.allowed != after.allowed || before.busy.is_empty() {
        return Contention::default();
    }
    let window = after.taken.duration_since(before.taken).as_secs_f64();
    if window <= 0.0 {
        return Contention::default();
    }
    let tick = clock_tick();
    let total_jiffies: u64 = after
        .busy
        .iter()
        .zip(&before.busy)
        .map(|(a, b)| a.saturating_sub(*b))
        .sum();
    let own_jiffies = after.own.saturating_sub(before.own);

    let total_s = total_jiffies as f64 / tick;
    let own_s = own_jiffies as f64 / tick;
    // Jiffy accounting is coarse and the two reads are not simultaneous, so a
    // genuinely clean window can difference to a small negative. Clamp rather
    // than report a negative contention, which would read as an anomaly.
    let foreign_s = (total_s - own_s).max(0.0);

    Contention {
        foreign_pct: foreign_s / window * 100.0,
        total_pct: total_s / window * 100.0,
        measured: true,
    }
}

impl ContentionSnapshot {
    /// Builds a snapshot from known parts.
    ///
    /// This exists because the tests live in `tests/bench_host_contention.rs`
    /// rather than in a `#[cfg(test)]` module here. The benches that own this
    /// file are `harness = false`, and a `#[cfg(test)]` module inside a
    /// `harness = false` bench is compiled with its `#[test]` functions stripped
    /// -- so tests written next to the code would never run, while still being
    /// counted as passing. Putting them in a real test target makes them run,
    /// and that costs one deliberate seam.
    ///
    /// Synthetic parts are also what the arithmetic tests want: differencing two
    /// fabricated snapshots asserts the formula, whereas sampling the real host
    /// asserts whatever the host happened to be doing.
    pub fn from_parts(taken: Instant, allowed: Vec<usize>, busy: Vec<u64>, own: u64) -> Self {
        Self {
            taken,
            allowed,
            busy,
            own,
        }
    }
}

/// Ticks per second, exposed so a test can express jiffies in core-seconds.
pub fn clock_tick_hz() -> f64 {
    clock_tick()
}

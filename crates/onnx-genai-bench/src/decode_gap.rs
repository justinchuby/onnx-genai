//! Measurement primitives for a *decode-shaped* model-level benchmark.
//!
//! # Why this exists
//!
//! Every model-level number this campaign published came from one of two
//! harness shapes, and both of them are dishonest about decode in opposite
//! directions:
//!
//! * **Few runs in a tight loop** (the `bench_generic` default of 7-10) sits
//!   entirely inside the thread-pool warm-up transient. A 32-wide decode pool
//!   needs roughly 60 inferences before its marginal cost stops falling, so a
//!   7-run measurement reports pool construction and calls it inference.
//! * **Many runs in a tight loop** (400+) leaves the transient but replaces it
//!   with the opposite bias: with no gap between iterations the workers never
//!   park, so every dispatch hits an already-spinning pool. That is the one
//!   regime real decode never occupies.
//!
//! Real decode is neither. It has microsecond-to-millisecond serial stretches
//! between parallel regions -- sampling, KV bookkeeping, the host side of the
//! next token -- and the interesting scheduler behaviour is precisely what
//! happens across those gaps: whether a worker is still spinning when the next
//! fan-out arrives, or has parked and must be woken through the kernel.
//!
//! This module supplies the pieces to measure that shape at the model level:
//! a configurable gap distribution, warm-up that is defined by observed
//! steady state rather than by a guessed constant, and the process counters
//! that distinguish "spun" from "parked".
//!
//! The equivalent micro-benchmark already exists in the CPU EP's
//! `task_runtime_latency` integration test, which sweeps `GAPS_US` against a
//! bare fan-out. That test answers "what does a dispatch cost after a gap of
//! N microseconds"; this module answers "what does a *model* cost when it is
//! run the way decode runs it", and the two are cross-validated against each
//! other in [`crate::decode_gap::tests`] and in the harness's `--validate`
//! mode.

use std::time::{Duration, Instant};

/// How the harness spends the gap between two model iterations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GapKind {
    /// Spin without yielding, the way a serial stretch of host-side decode
    /// work does. The pool's workers keep their cores warm and are free to
    /// keep spinning, so this measures the spin-window hit path.
    Busy,
    /// Sleep, releasing the core. This is the pessimistic shape: the OS is
    /// free to migrate or deschedule the workers, so the next dispatch is
    /// likely to pay a real wake.
    Sleep,
    /// Alternate `Busy` and `Sleep` per iteration. Real decode is a mixture --
    /// some inter-token gaps are pure compute, others block on a tokenizer,
    /// a sampler allocation or a detokenizer write -- and a harness that only
    /// ever does one of the two will tune the spin window to the wrong shape.
    Mixed,
}

impl GapKind {
    /// Parses the `--gap-kind` flag.
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "busy" => Ok(Self::Busy),
            "sleep" => Ok(Self::Sleep),
            "mixed" => Ok(Self::Mixed),
            other => Err(format!(
                "unknown gap kind '{other}'; expected busy, sleep or mixed"
            )),
        }
    }
}

/// A deterministic inter-iteration gap generator.
///
/// Jitter matters. A fixed gap can sit permanently just inside or just outside
/// the spin window and produce a bimodal result that looks like a clean number;
/// spreading the gap across the window boundary reports the mixture that real
/// decode actually pays. The generator is a seeded xorshift rather than a real
/// RNG so a run is reproducible from its printed seed.
#[derive(Debug, Clone)]
pub struct GapDistribution {
    mean_us: u64,
    jitter: f64,
    kind: GapKind,
    state: u64,
    issued: u64,
}

impl GapDistribution {
    /// `jitter` is the fractional half-width of a uniform spread around
    /// `mean_us`: `0.0` is a fixed gap, `0.5` spreads over `[0.5x, 1.5x]`.
    /// It is clamped to `[0, 1]` so the gap can never go negative.
    pub fn new(mean_us: u64, jitter: f64, kind: GapKind, seed: u64) -> Self {
        Self {
            mean_us,
            jitter: jitter.clamp(0.0, 1.0),
            kind,
            // A zero seed is a fixed point of xorshift, which would emit an
            // endless run of zeros and silently turn jitter off.
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
            issued: 0,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// The next gap length. Deterministic given the seed.
    pub fn next_gap(&mut self) -> Duration {
        let mean = self.mean_us;
        if mean == 0 {
            self.issued = self.issued.wrapping_add(1);
            return Duration::ZERO;
        }
        let micros = if self.jitter == 0.0 {
            mean
        } else {
            // Uniform in [mean*(1-jitter), mean*(1+jitter)], computed in
            // floating point and rounded, so small means still jitter.
            let unit = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            let low = mean as f64 * (1.0 - self.jitter);
            let high = mean as f64 * (1.0 + self.jitter);
            (low + unit * (high - low)).round().max(0.0) as u64
        };
        self.issued = self.issued.wrapping_add(1);
        Duration::from_micros(micros)
    }

    /// Which of [`GapKind::Busy`] / [`GapKind::Sleep`] the next gap uses.
    /// [`GapKind::Mixed`] alternates on the issue counter, so a run of N
    /// iterations gets as close to an even split as N allows.
    pub fn next_kind(&self) -> GapKind {
        match self.kind {
            GapKind::Mixed => {
                if self.issued.is_multiple_of(2) {
                    GapKind::Busy
                } else {
                    GapKind::Sleep
                }
            }
            other => other,
        }
    }
}

/// Spends `gap` the way `kind` says to.
///
/// The busy arm deliberately does not call `yield_now`: the point is to hold
/// the core the way host-side decode work would, so the pool's spin window
/// sees a realistic neighbour.
pub fn spend_gap(gap: Duration, kind: GapKind) {
    if gap.is_zero() {
        return;
    }
    match kind {
        GapKind::Busy | GapKind::Mixed => {
            let until = Instant::now() + gap;
            while Instant::now() < until {
                std::hint::spin_loop();
            }
        }
        GapKind::Sleep => std::thread::sleep(gap),
    }
}

/// Process-wide counters sampled from `/proc/self`.
///
/// `voluntary_ctxt_switches` is the load-bearing one: a worker that parks and
/// is later woken through a futex records exactly one voluntary switch, so the
/// delta across a measured window is a direct count of park/wake round trips.
/// A pool that spins through every gap moves this number by almost nothing;
/// one that parks on every gap moves it by roughly `iterations x workers`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProcessMetrics {
    pub user_us: u64,
    pub sys_us: u64,
    pub rss_kb: u64,
    pub peak_rss_kb: u64,
    pub threads: u64,
    pub voluntary_ctxt_switches: u64,
    pub involuntary_ctxt_switches: u64,
}

impl ProcessMetrics {
    /// Field-wise `self - earlier`, saturating so a counter that wrapped or
    /// was re-read out of order reports zero rather than a huge bogus delta.
    /// RSS is a level, not a counter, so it is carried through rather than
    /// differenced.
    pub fn since(&self, earlier: &Self) -> Self {
        Self {
            user_us: self.user_us.saturating_sub(earlier.user_us),
            sys_us: self.sys_us.saturating_sub(earlier.sys_us),
            rss_kb: self.rss_kb,
            peak_rss_kb: self.peak_rss_kb,
            threads: self.threads,
            voluntary_ctxt_switches: self
                .voluntary_ctxt_switches
                .saturating_sub(earlier.voluntary_ctxt_switches),
            involuntary_ctxt_switches: self
                .involuntary_ctxt_switches
                .saturating_sub(earlier.involuntary_ctxt_switches),
        }
    }

    pub fn cpu_us(&self) -> u64 {
        self.user_us + self.sys_us
    }
}

/// Parses the subset of `/proc/self/status` this harness reports.
///
/// Split out from the read so it can be tested against a captured sample --
/// the format is stable but the field set differs across kernels, and a silent
/// parse failure here would show up as a suspiciously flat counter rather than
/// as an error.
pub fn parse_status(text: &str) -> ProcessMetrics {
    let mut metrics = ProcessMetrics::default();
    for line in text.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let number = value
            .split_whitespace()
            .next()
            .and_then(|token| token.parse::<u64>().ok())
            .unwrap_or(0);
        match key {
            "VmRSS" => metrics.rss_kb = number,
            "VmHWM" => metrics.peak_rss_kb = number,
            "Threads" => metrics.threads = number,
            "voluntary_ctxt_switches" => metrics.voluntary_ctxt_switches = number,
            "nonvoluntary_ctxt_switches" => metrics.involuntary_ctxt_switches = number,
            _ => {}
        }
    }
    metrics
}

/// Parses `utime`/`stime` (fields 14 and 15) out of a `/proc/.../stat` line.
///
/// The comm field is parenthesized and may itself contain spaces and
/// parentheses, so the parse has to start after the *last* `)` rather than
/// splitting the whole line on whitespace.
pub fn parse_stat_cpu_ticks(line: &str) -> Option<(u64, u64)> {
    let rest = &line[line.rfind(')')? + 1..];
    let fields = rest.split_whitespace().collect::<Vec<_>>();
    // `rest` starts at field 3 (state), so utime/stime are indices 11 and 12.
    let utime = fields.get(11)?.parse().ok()?;
    let stime = fields.get(12)?.parse().ok()?;
    Some((utime, stime))
}

fn clock_ticks_per_second() -> u64 {
    // `sysconf(_SC_CLK_TCK)` is 100 on every Linux target this runs on. Rather
    // than link libc for one constant, use the value and let the CPU numbers
    // be reported in the same units the kernel already uses.
    100
}

/// Samples this process's counters now. Returns defaults off Linux or if
/// `/proc` is not mounted, so the harness degrades to timing-only rather than
/// failing.
///
/// Three different files, because Linux reports these three things in three
/// different scopes and getting that wrong produces confident nonsense:
///
/// * CPU time comes from `/proc/self/stat`, whose `utime`/`stime` *are*
///   summed over the whole thread group. `status` carries no CPU accounting at
///   all, and reading only it is how a harness reports `0.00 cpu-s`.
/// * RSS and the thread count come from `/proc/self/status`, which is
///   process-wide for those fields.
/// * Context switches are summed over `/proc/self/task/*/status`, because the
///   counters in `/proc/self/status` describe **only the leader thread**. A
///   pool of sixteen workers parking and waking on every dispatch moves the
///   leader's counter by approximately nothing, so reading the process file
///   reports "nothing ever parks" no matter what the pool does. This is the
///   measurement that the park/spin question turns on. The sum only covers
///   threads that are *still alive*: a thread that exits between two samples
///   takes its counters with it, so a teardown-heavy workload undercounts.
///   Pool workers live for the whole measured window, so this does not affect
///   the numbers this harness reports.
pub fn sample_process_metrics() -> ProcessMetrics {
    let mut metrics = std::fs::read_to_string("/proc/self/status")
        .map(|text| parse_status(&text))
        .unwrap_or_default();
    if let Some((utime, stime)) = std::fs::read_to_string("/proc/self/stat")
        .ok()
        .and_then(|line| parse_stat_cpu_ticks(&line))
    {
        let ticks = clock_ticks_per_second();
        metrics.user_us = utime * 1_000_000 / ticks;
        metrics.sys_us = stime * 1_000_000 / ticks;
    }
    let (voluntary, involuntary) = sum_thread_ctxt_switches();
    metrics.voluntary_ctxt_switches = voluntary;
    metrics.involuntary_ctxt_switches = involuntary;
    metrics
}

/// Sums `(voluntary, nonvoluntary)` context switches over every thread.
fn sum_thread_ctxt_switches() -> (u64, u64) {
    let Ok(entries) = std::fs::read_dir("/proc/self/task") else {
        return (0, 0);
    };
    let mut voluntary = 0;
    let mut involuntary = 0;
    for entry in entries.flatten() {
        if let Ok(text) = std::fs::read_to_string(entry.path().join("status")) {
            let metrics = parse_status(&text);
            voluntary += metrics.voluntary_ctxt_switches;
            involuntary += metrics.involuntary_ctxt_switches;
        }
    }
    (voluntary, involuntary)
}

/// One OS thread's identity and accumulated cost.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThreadInfo {
    pub tid: u64,
    /// `comm`, which is what shows up in `top -H` and in a debugger. Rust's
    /// `std::thread::Builder::name` sets it; Rayon's default pools do not,
    /// which is exactly why unnamed threads are hard to attribute.
    pub name: String,
    pub cpu_us: u64,
    pub voluntary_ctxt_switches: u64,
    pub involuntary_ctxt_switches: u64,
}

/// Every thread in this process, with its name and CPU time.
///
/// This is the tool for attributing pool threads to their owner. A thread with
/// no name is either a Rayon worker from a pool built without a thread-name
/// callback, or a thread created by a C library (ORT's intra-op pool, the
/// allocator's background threads). Naming our own pools is what makes the
/// remainder identifiable by elimination.
pub fn thread_census() -> Vec<ThreadInfo> {
    let Ok(entries) = std::fs::read_dir("/proc/self/task") else {
        return Vec::new();
    };
    let ticks = clock_ticks_per_second();
    let mut threads = Vec::new();
    for entry in entries.flatten() {
        let Ok(tid) = entry.file_name().to_string_lossy().parse::<u64>() else {
            continue;
        };
        let path = entry.path();
        let name = std::fs::read_to_string(path.join("comm"))
            .map(|text| text.trim().to_string())
            .unwrap_or_default();
        let cpu_us = std::fs::read_to_string(path.join("stat"))
            .ok()
            .and_then(|line| parse_stat_cpu_ticks(&line))
            .map(|(utime, stime)| (utime + stime) * 1_000_000 / ticks)
            .unwrap_or(0);
        let status = std::fs::read_to_string(path.join("status"))
            .map(|text| parse_status(&text))
            .unwrap_or_default();
        threads.push(ThreadInfo {
            tid,
            name,
            cpu_us,
            voluntary_ctxt_switches: status.voluntary_ctxt_switches,
            involuntary_ctxt_switches: status.involuntary_ctxt_switches,
        });
    }
    threads.sort_by(|a, b| b.cpu_us.cmp(&a.cpu_us).then(a.tid.cmp(&b.tid)));
    threads
}

/// Groups a census by thread name, returning `(name, count, total_cpu_us)`
/// sorted by descending count. Unnamed threads are grouped under `"<unnamed>"`.
pub fn census_by_name(threads: &[ThreadInfo]) -> Vec<(String, usize, u64)> {
    let mut groups: Vec<(String, usize, u64)> = Vec::new();
    for thread in threads {
        let key = if thread.name.is_empty() {
            "<unnamed>"
        } else {
            thread.name.as_str()
        };
        match groups.iter_mut().find(|(name, _, _)| name == key) {
            Some(group) => {
                group.1 += 1;
                group.2 += thread.cpu_us;
            }
            None => groups.push((key.to_string(), 1, thread.cpu_us)),
        }
    }
    groups.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    groups
}

/// Finds where a timing series stops trending, i.e. where warm-up ends.
///
/// Returns the first index `s` such that every non-overlapping block of
/// `window` samples in `samples[s..]` has a median within `tolerance`
/// (relative) of the median of the whole tail `samples[s..]`. `None` means the
/// series never settled -- which is itself the answer: the run was too short
/// to have a steady state, and any number taken from it describes the
/// transient.
///
/// Three deliberate choices, each of which was wrong in an earlier version:
///
/// * **Medians, not means.** A shared host delivers occasional multi-
///   millisecond outliers that are another tenant's scheduling decision, not
///   ours. One of those moves a window's mean past any sane tolerance and
///   reports "never settled" for a series that plainly settled.
/// * **Non-overlapping blocks, not a sliding window.** Requiring every one of
///   ~n overlapping windows to pass makes the verdict hostage to a single
///   blip anywhere in the run, because that blip appears in `window`
///   consecutive windows. Blocks reduce the test to `n/window` weakly
///   correlated checks.
/// * **Compared against the whole tail, not against the final block.** The
///   final block cannot be its own reference -- it always matches, which is
///   how a monotonically rising series gets declared steady at the very end.
///
/// At least two full blocks must fit, so a run too short to demonstrate
/// stability reports `None` rather than a confident answer.
///
/// This replaces the guessed `--warmups` constant. A 32-wide pool and a 4-wide
/// pool have warm-ups that differ by more than an order of magnitude, so one
/// constant cannot be right for both, and picking the constant per-arm by hand
/// is how an A/B ends up comparing a warm arm against a cold one.
pub fn steady_state_start(samples: &[f64], window: usize, tolerance: f64) -> Option<usize> {
    if window == 0 || samples.len() < window * 2 {
        return None;
    }
    let median_of = |slice: &[f64]| -> f64 {
        let mut sorted = slice.to_vec();
        sorted.sort_by(f64::total_cmp);
        sorted[sorted.len() / 2]
    };
    let settled_from = |start: usize| -> bool {
        let tail = &samples[start..];
        if tail.len() < window * 2 {
            return false;
        }
        let reference = median_of(tail);
        if reference <= 0.0 {
            return false;
        }
        tail.chunks_exact(window)
            .all(|block| ((median_of(block) - reference) / reference).abs() <= tolerance)
    };
    (0..=samples.len() - window * 2).find(|&start| settled_from(start))
}

/// Threads present in `after` but not in `before`, i.e. created during the
/// phase the two censuses bracket.
///
/// This is the attribution tool. An unnamed thread is anonymous in a census
/// but not in a *delta*: bracketing "build the native session", "build the ORT
/// session" and "run the first inference" separately says which component
/// created it, which is the difference between an unexplained thread count and
/// an owned one.
pub fn census_delta(before: &[ThreadInfo], after: &[ThreadInfo]) -> Vec<ThreadInfo> {
    after
        .iter()
        .filter(|thread| !before.iter().any(|earlier| earlier.tid == thread.tid))
        .cloned()
        .collect()
}

/// Nearest-rank percentile summary of a timing series.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Summary {
    pub count: usize,
    pub min: f64,
    pub p50: f64,
    pub p90: f64,
    pub p99: f64,
    pub max: f64,
    pub mean: f64,
}

impl Summary {
    /// `samples` must be non-empty.
    pub fn from(samples: &[f64]) -> Self {
        assert!(!samples.is_empty(), "Summary::from needs a sample");
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        let rank = |fraction: f64| -> f64 {
            let index = ((sorted.len() as f64) * fraction).ceil().max(1.0) as usize;
            sorted[index.min(sorted.len()) - 1]
        };
        Self {
            count: sorted.len(),
            min: sorted[0],
            p50: sorted[sorted.len() / 2],
            p90: rank(0.9),
            p99: rank(0.99),
            max: sorted[sorted.len() - 1],
            mean: sorted.iter().sum::<f64>() / sorted.len() as f64,
        }
    }

    /// p90/p50. A quiet host sits near 1.0; this run's host does not, and the
    /// ratio is printed so a reader can tell which.
    pub fn spread(&self) -> f64 {
        if self.p50 == 0.0 {
            0.0
        } else {
            self.p90 / self.p50
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_zero_mean_gap_is_always_zero() {
        let mut gaps = GapDistribution::new(0, 0.5, GapKind::Busy, 7);
        for _ in 0..16 {
            assert_eq!(gaps.next_gap(), Duration::ZERO);
        }
    }

    #[test]
    fn zero_jitter_repeats_the_mean_exactly() {
        let mut gaps = GapDistribution::new(25, 0.0, GapKind::Busy, 7);
        for _ in 0..16 {
            assert_eq!(gaps.next_gap(), Duration::from_micros(25));
        }
    }

    #[test]
    fn jittered_gaps_stay_inside_the_requested_band_and_actually_vary() {
        let mut gaps = GapDistribution::new(100, 0.5, GapKind::Busy, 12345);
        let drawn = (0..512).map(|_| gaps.next_gap()).collect::<Vec<_>>();
        for gap in &drawn {
            let micros = gap.as_micros() as u64;
            assert!(
                (50..=150).contains(&micros),
                "gap {micros}us escaped the +/-50% band"
            );
        }
        let distinct = drawn.iter().collect::<std::collections::HashSet<_>>();
        assert!(
            distinct.len() > 16,
            "jitter produced only {} distinct gaps",
            distinct.len()
        );
    }

    #[test]
    fn jitter_is_clamped_so_a_gap_can_never_be_negative() {
        let mut gaps = GapDistribution::new(10, 9.0, GapKind::Busy, 3);
        for _ in 0..256 {
            let micros = gaps.next_gap().as_micros() as u64;
            assert!((0..=20).contains(&micros), "gap {micros}us escaped clamp");
        }
    }

    #[test]
    fn a_seed_reproduces_its_sequence() {
        let draw = || {
            let mut gaps = GapDistribution::new(80, 0.4, GapKind::Sleep, 99);
            (0..32).map(|_| gaps.next_gap()).collect::<Vec<_>>()
        };
        assert_eq!(draw(), draw());
    }

    #[test]
    fn a_zero_seed_still_jitters() {
        let mut gaps = GapDistribution::new(100, 0.5, GapKind::Busy, 0);
        let drawn = (0..64).map(|_| gaps.next_gap()).collect::<Vec<_>>();
        let distinct = drawn.iter().collect::<std::collections::HashSet<_>>();
        assert!(distinct.len() > 8, "seed 0 collapsed to a constant gap");
    }

    #[test]
    fn mixed_alternates_between_spinning_and_sleeping() {
        let mut gaps = GapDistribution::new(10, 0.0, GapKind::Mixed, 1);
        let mut kinds = Vec::new();
        for _ in 0..4 {
            kinds.push(gaps.next_kind());
            let _ = gaps.next_gap();
        }
        assert_eq!(
            kinds,
            vec![GapKind::Busy, GapKind::Sleep, GapKind::Busy, GapKind::Sleep]
        );
    }

    #[test]
    fn a_fixed_kind_never_alternates() {
        let mut gaps = GapDistribution::new(10, 0.0, GapKind::Sleep, 1);
        for _ in 0..4 {
            assert_eq!(gaps.next_kind(), GapKind::Sleep);
            let _ = gaps.next_gap();
        }
    }

    #[test]
    fn gap_kinds_parse_and_reject_junk() {
        assert_eq!(GapKind::parse("busy"), Ok(GapKind::Busy));
        assert_eq!(GapKind::parse("sleep"), Ok(GapKind::Sleep));
        assert_eq!(GapKind::parse("mixed"), Ok(GapKind::Mixed));
        assert!(GapKind::parse("spin").is_err());
    }

    const SAMPLE_STATUS: &str = "\
Name:\tbench_decode_gap
Umask:\t0002
State:\tR (running)
Tgid:\t4242
Threads:\t33
VmHWM:\t  184320 kB
VmRSS:\t  151552 kB
voluntary_ctxt_switches:\t1611
nonvoluntary_ctxt_switches:\t97
";

    #[test]
    fn status_parses_the_fields_the_harness_reports() {
        let metrics = parse_status(SAMPLE_STATUS);
        assert_eq!(metrics.threads, 33);
        assert_eq!(metrics.rss_kb, 151_552);
        assert_eq!(metrics.peak_rss_kb, 184_320);
        assert_eq!(metrics.voluntary_ctxt_switches, 1611);
        assert_eq!(metrics.involuntary_ctxt_switches, 97);
    }

    #[test]
    fn status_parsing_survives_a_kernel_without_those_fields() {
        let metrics = parse_status("Name:\tx\nState:\tS (sleeping)\n");
        assert_eq!(metrics, ProcessMetrics::default());
    }

    #[test]
    fn deltas_difference_counters_and_carry_levels() {
        let before = ProcessMetrics {
            user_us: 100,
            sys_us: 40,
            rss_kb: 900,
            peak_rss_kb: 950,
            threads: 4,
            voluntary_ctxt_switches: 10,
            involuntary_ctxt_switches: 2,
        };
        let after = ProcessMetrics {
            user_us: 350,
            sys_us: 90,
            rss_kb: 1200,
            peak_rss_kb: 1300,
            threads: 33,
            voluntary_ctxt_switches: 610,
            involuntary_ctxt_switches: 5,
        };
        let delta = after.since(&before);
        assert_eq!(delta.user_us, 250);
        assert_eq!(delta.sys_us, 50);
        assert_eq!(delta.cpu_us(), 300);
        assert_eq!(delta.voluntary_ctxt_switches, 600);
        assert_eq!(delta.involuntary_ctxt_switches, 3);
        // Levels are carried, not differenced.
        assert_eq!(delta.rss_kb, 1200);
        assert_eq!(delta.peak_rss_kb, 1300);
        assert_eq!(delta.threads, 33);
    }

    #[test]
    fn a_counter_read_out_of_order_reports_zero_rather_than_wrapping() {
        let high = ProcessMetrics {
            user_us: 500,
            voluntary_ctxt_switches: 900,
            ..ProcessMetrics::default()
        };
        let delta = ProcessMetrics::default().since(&high);
        assert_eq!(delta.user_us, 0);
        assert_eq!(delta.voluntary_ctxt_switches, 0);
    }

    #[test]
    fn stat_cpu_parsing_survives_a_comm_containing_spaces_and_parens() {
        let line = "4242 (weird ) name) R 1 4242 4242 0 -1 4194560 100 0 0 0 \
                    731 219 0 0 20 0 33 0 900";
        assert_eq!(parse_stat_cpu_ticks(line), Some((731, 219)));
    }

    #[test]
    fn stat_cpu_parsing_rejects_a_truncated_line() {
        assert_eq!(parse_stat_cpu_ticks("4242 (x) R 1 2 3"), None);
        assert_eq!(parse_stat_cpu_ticks("no parens here"), None);
    }

    fn thread(tid: u64, name: &str, cpu_us: u64) -> ThreadInfo {
        ThreadInfo {
            tid,
            name: name.to_string(),
            cpu_us,
            voluntary_ctxt_switches: 0,
            involuntary_ctxt_switches: 0,
        }
    }

    #[test]
    fn a_census_groups_by_name_and_buckets_the_unnamed() {
        let threads = vec![
            thread(1, "bench_decode_gap", 500),
            thread(2, "nxrt-decode-0", 300),
            thread(3, "nxrt-decode-1", 280),
            thread(4, "", 40),
            thread(5, "", 35),
            thread(6, "", 30),
        ];
        let groups = census_by_name(&threads);
        assert_eq!(groups[0], ("<unnamed>".to_string(), 3, 105));
        assert!(groups.contains(&("nxrt-decode-0".to_string(), 1, 300)));
        assert!(groups.contains(&("bench_decode_gap".to_string(), 1, 500)));
    }

    #[test]
    fn an_empty_census_groups_to_nothing() {
        assert!(census_by_name(&[]).is_empty());
    }

    #[test]
    fn a_census_delta_reports_only_threads_created_in_the_phase() {
        let before = vec![thread(1, "main", 10), thread(2, "", 5)];
        let after = vec![
            thread(1, "main", 20),
            thread(2, "", 9),
            thread(7, "nxrt-task-0", 3),
            thread(8, "", 1),
        ];
        let created = census_delta(&before, &after);
        assert_eq!(created.len(), 2);
        assert!(created.iter().any(|t| t.tid == 7));
        assert!(created.iter().any(|t| t.tid == 8));
    }

    #[test]
    fn a_census_delta_over_a_phase_that_created_nothing_is_empty() {
        let census = vec![thread(1, "main", 10), thread(2, "", 5)];
        assert!(census_delta(&census, &census).is_empty());
    }

    #[test]
    fn a_census_delta_ignores_threads_that_exited() {
        let before = vec![thread(1, "main", 10), thread(2, "", 5)];
        let after = vec![thread(1, "main", 20)];
        assert!(census_delta(&before, &after).is_empty());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn process_metrics_report_real_cpu_and_threads_on_linux() {
        // Burn a little CPU so utime is certain to be non-zero, then check the
        // sampler actually observed it. This is the regression guard for
        // reading CPU out of `/proc/self/status`, which does not carry it.
        let started = Instant::now();
        let mut sink = 0u64;
        while started.elapsed() < Duration::from_millis(60) {
            sink = sink.wrapping_add(started.elapsed().as_nanos() as u64);
        }
        std::hint::black_box(sink);
        let metrics = sample_process_metrics();
        assert!(metrics.threads >= 1, "census saw no threads");
        assert!(metrics.rss_kb > 0, "no RSS reported");
        assert!(metrics.cpu_us() > 0, "no CPU time reported");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn ctxt_switches_are_summed_over_every_thread_not_just_the_leader() {
        // A parked-then-woken worker records a voluntary switch on *its own*
        // task file, not the leader's. Park a spawned thread deliberately and
        // check the process-wide sampler sees it: reading only
        // `/proc/self/status` would report zero here, which is exactly the bug
        // that made a spinning pool and a parking pool look identical.
        //
        // The worker is sampled while it is still alive, because a thread that
        // has exited takes its `/proc/self/task/<tid>` entry -- and its
        // counters -- with it.
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let slept = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let before = sample_process_metrics();
        let handle = {
            let slept = Arc::clone(&slept);
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                for _ in 0..40 {
                    std::thread::sleep(Duration::from_micros(200));
                }
                slept.store(true, Ordering::SeqCst);
                while !stop.load(Ordering::SeqCst) {
                    std::hint::spin_loop();
                }
            })
        };
        while !slept.load(Ordering::SeqCst) {
            std::hint::spin_loop();
        }
        let delta = sample_process_metrics().since(&before);
        stop.store(true, Ordering::SeqCst);
        handle.join().expect("worker thread");
        assert!(
            delta.voluntary_ctxt_switches >= 20,
            "sampler saw only {} voluntary switches for a thread that slept 40 times; \
             counters are probably being read from the leader alone",
            delta.voluntary_ctxt_switches
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn a_live_census_sees_at_least_this_thread() {
        let threads = thread_census();
        assert!(!threads.is_empty(), "live census was empty");
        let groups = census_by_name(&threads);
        assert_eq!(
            groups.iter().map(|(_, count, _)| count).sum::<usize>(),
            threads.len()
        );
    }

    #[test]
    fn steady_state_is_found_after_a_decaying_transient() {
        // A warm-up that decays over the first 20 samples, then flat at 1.0.
        let mut samples = (0..20).map(|i| 5.0 - 0.2 * i as f64).collect::<Vec<_>>();
        samples.extend(std::iter::repeat_n(1.0, 40));
        let start = steady_state_start(&samples, 8, 0.05).expect("series settles");
        assert!(
            (16..=24).contains(&start),
            "steady state reported at {start}, expected near the end of the transient"
        );
    }

    #[test]
    fn an_already_flat_series_is_steady_from_the_first_sample() {
        let samples = vec![2.0; 40];
        assert_eq!(steady_state_start(&samples, 8, 0.05), Some(0));
    }

    #[test]
    fn a_series_that_never_settles_reports_no_steady_state() {
        // Monotonically rising: no early window matches the final one, and the
        // final window is not allowed to qualify by matching itself.
        let samples = (0..40).map(|i| 1.0 + i as f64).collect::<Vec<_>>();
        assert_eq!(steady_state_start(&samples, 8, 0.01), None);
    }

    #[test]
    fn a_dip_inside_the_transient_is_not_mistaken_for_steady_state() {
        // Decays from 5.0, touches the eventual steady value across indices
        // 8..16, then climbs back before finally settling at index 24. A
        // detector that matched a single window would report the dip.
        let mut samples = vec![5.0; 8];
        samples.extend(std::iter::repeat_n(1.0, 8));
        samples.extend(std::iter::repeat_n(4.0, 8));
        samples.extend(std::iter::repeat_n(1.0, 24));
        let start = steady_state_start(&samples, 8, 0.05).expect("series settles eventually");
        // Past the dip is the property under test. A robust estimator is
        // allowed a sample or two of slack at the final step, so this asserts
        // the dip was rejected rather than pinning an exact index.
        assert!(
            start >= 16,
            "steady state reported at {start}, inside the transient dip at 8..16"
        );
    }

    #[test]
    fn exactly_two_windows_of_flat_samples_is_the_minimum_settled_series() {
        assert_eq!(steady_state_start(&[1.0; 16], 8, 0.05), Some(0));
        assert_eq!(steady_state_start(&[1.0; 15], 8, 0.05), None);
    }

    #[test]
    fn a_series_shorter_than_two_windows_has_no_steady_state() {
        let samples = vec![1.0; 10];
        assert_eq!(steady_state_start(&samples, 8, 0.05), None);
        assert_eq!(steady_state_start(&samples, 0, 0.05), None);
    }

    #[test]
    fn summary_percentiles_use_nearest_rank() {
        let samples = (1..=100).map(|i| i as f64).collect::<Vec<_>>();
        let summary = Summary::from(&samples);
        assert_eq!(summary.count, 100);
        assert_eq!(summary.min, 1.0);
        assert_eq!(summary.max, 100.0);
        assert_eq!(summary.p90, 90.0);
        assert_eq!(summary.p99, 99.0);
        assert!((summary.mean - 50.5).abs() < 1e-9);
    }

    #[test]
    fn summary_of_one_sample_is_that_sample() {
        let summary = Summary::from(&[3.5]);
        assert_eq!(summary.p50, 3.5);
        assert_eq!(summary.p99, 3.5);
        assert_eq!(summary.spread(), 1.0);
    }
}

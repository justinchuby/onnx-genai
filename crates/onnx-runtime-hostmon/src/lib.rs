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
//! # The blind spot the confined set has by construction: SMT siblings
//!
//! That control row is true as written and misleading as a general rule, because
//! cpu 5 is not the SMT sibling of any CPU in that set. A decode budget of `N`
//! confines the process to `N` *physical cores* -- on an SMT host the pool picks
//! one logical CPU per core, so the partner logical CPU of every core it runs on
//! is **outside** the mask. Foreign work there is invisible to `foreign_pct` by
//! construction: it is not on an allowed CPU, so it is never counted, and yet it
//! shares the core's execution resources with a decode worker.
//!
//! That is not a hypothetical. Measured on a 16-core/32-thread host at a budget
//! of 12, with a verified 100%-busy spinner pinned to one chosen CPU and the
//! per-worker profile from `decode_spmd` naming which worker slowed down:
//!
//! | spinner on | predicted slow worker | observed slow worker | `foreign_%` |
//! |---|---|---|---|
//! | cpu 19 (sibling of core 18) | 18 | **18** | 5.7 / 8.4 |
//! | cpu 3 (sibling of core 2) | 2 | **2** (3 of 3) | 26.1 / 0.0 / 1.4 |
//! | cpu 27 (sibling of core 26, **not** in the set or its siblings) | — | miss (3 of 3) | 13.5 / 27.5 / 4.5 |
//!
//! Five of six sibling arms named the predicted worker (p ~ 2e-5 against a
//! uniform choice among 12), the off-set control named it zero times out of
//! three, and the affected worker's in-shard time rose ~1.7x for an exactly equal
//! row segment -- while `foreign_%` read as low as **0.0**. A dispatch is a
//! barrier, so that one worker set the whole dispatch: wall went 3.2 -> 5.2-7.3
//! ms/token.
//!
//! So [`Contention`] also reports [`sibling_peak_pct`](Contention::sibling_peak_pct):
//! the busiest single logical CPU that is an SMT sibling of an allowed CPU and is
//! not itself allowed. Two properties make it a cleaner measurement than
//! `foreign_pct` rather than a weaker one:
//!
//! * **No own-time subtraction is needed.** This process cannot run on those CPUs,
//!   so every busy jiffy there is foreign by definition. The
//!   [`own_time_complete`](Contention::own_time_complete) caveat does not apply.
//! * **Peak, not sum.** Under a barrier one saturated sibling gates the dispatch
//!   and a dozen lukewarm ones do not, so summing would let ambient noise spread
//!   across a wide set outvote the single core that is actually being halved.
//!   This is the opposite normalisation from `foreign_pct` for the opposite
//!   reason, and the two are not interchangeable.
//!
//! The two columns were then cross-validated against each other end to end, with
//! a bounded hog whose own occupancy is read from `/proc/stat` rather than
//! assumed. Decode budget 4 (cpus 0, 2, 4, 6), 60 tokens x 3 reps:
//!
//! | injected | `foreign_%` | `sibling_peak_%` | ms/token | hog occupancy |
//! |---|---|---|---|---|
//! | nothing | 0.0 | 12.1 | 9.14 | — |
//! | hog on cpu 1 (sibling of decode cpu 0) | **1.1** | **100.6** | 15.39 | 100.1% |
//! | hog on cpu 4 (a decode CPU) | **48.1** | 6.4 | 17.96 | 100.2% |
//!
//! Each figure moves for its own mechanism and stays put for the other's, which
//! is the property that makes the pair worth carrying: either arm on its own is
//! equally consistent with a second column that merely mirrors the first. The
//! middle row is the case this whole section exists for -- a 1.68x regression
//! that the pre-existing column certifies as clean.
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
//! # The own-time precondition, and why it is checked rather than documented
//!
//! Busy time is restricted to the allowed CPUs, but this process's own time is
//! read process-wide from `/proc/self/stat`. The subtraction is therefore only
//! valid when *every* thread of the process is confined to the allowed set. If
//! some thread runs off the set -- an unconfined main thread, a library's
//! background thread, or a failed affinity call leaving a pool wide -- own time
//! is subtracted that was never counted in busy, and contention is
//! *under*-reported. That is the unsafe direction: it moves a contended reading
//! toward "clean".
//!
//! While this lived beside its single caller the precondition was documented and
//! held by inspection, because the EP narrows the main thread's affinity during
//! `initialize`, before any session or pool thread exists, and children inherit
//! the narrowed mask. That reasoning does not travel with the code. So
//! [`threads_off_mask`] checks it at both ends of the window, and
//! [`Contention::own_time_complete`] records the answer: when it is false the
//! figure is a *lower bound* rather than an estimate, [`foreign_column`] marks
//! it with a trailing `!`, and [`Contention::is_clean`] refuses to certify the
//! window. The check is a boundary property and does not see a thread that is
//! spawned wide and joined entirely between the two snapshots; see
//! [`Contention::own_time_complete`] for why that is stated rather than
//! sampled around.
//!
//! # Measured contention and declared intent are different questions
//!
//! Everything above measures what the host *did*. [`hostlock`] reads what
//! somebody *said they were doing* -- the advisory lock in `scripts/hostlock.sh`
//! that has existed on `main` with no in-tree consumer. Neither substitutes for
//! the other: an unlocked run on a genuinely idle box is fine, and a locked run
//! next to a co-tenant's unannounced `cargo test` is not. A row wants both, and
//! [`hostlock::field`] deliberately reads at both ends of the window so it
//! cannot report a single credible holder for a window that changed hands.

pub mod hostlock;

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

/// The number of CPUs online on the host, independent of this process's mask.
///
/// [`AllowedCpus::current`] answers "which CPUs may I use"; this answers "how
/// many exist". The pair is what distinguishes a process that was confined from
/// one that simply has a small machine, and only the first is evidence of a
/// narrowed budget.
#[cfg(target_os = "linux")]
pub fn online_cpus() -> Option<usize> {
    // SAFETY: `sysconf` takes an int and returns a long; no pointers involved.
    let n = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) };
    if n > 0 { Some(n as usize) } else { None }
}

#[cfg(not(target_os = "linux"))]
pub fn online_cpus() -> Option<usize> {
    None
}

/// The SMT siblings of `allowed` that are **not** themselves in `allowed`,
/// sorted and deduplicated, given a reader for one CPU's sibling list.
///
/// Split from the sysfs read so the set arithmetic can be tested against
/// fabricated topologies -- a host with SMT off, a host whose partner CPU is
/// already inside the mask, and a host with more than two threads per core all
/// have to behave, and none of them can be conjured on the machine running the
/// test.
///
/// `read_list` returns the contents of
/// `/sys/devices/system/cpu/cpuN/topology/thread_siblings_list`, which is a
/// kernel CPU list in range notation, the same format [`parse_cpu_list`] takes.
///
/// Returns `None` if **any** allowed CPU's list is missing or malformed, rather
/// than the siblings it managed to find. A partial sibling set would report a
/// quiet peak for a host whose hot sibling happened to be the one that failed to
/// parse, turning "topology unknown" into "topology known and clean" -- the same
/// substitution this crate exists to prevent one level up.
pub fn siblings_outside<F>(allowed: &[usize], read_list: F) -> Option<Vec<usize>>
where
    F: Fn(usize) -> Option<String>,
{
    let mut siblings = Vec::new();
    for &cpu in allowed {
        let list = read_list(cpu)?;
        for sibling in parse_cpu_list(&list)? {
            if sibling != cpu && !allowed.contains(&sibling) {
                siblings.push(sibling);
            }
        }
    }
    siblings.sort_unstable();
    siblings.dedup();
    Some(siblings)
}

/// [`siblings_outside`] against the running host's sysfs topology.
///
/// An empty vector is a real answer -- SMT disabled, or a mask that already
/// contains both threads of every core it uses -- and is distinct from `None`,
/// which means the topology could not be read at all.
#[cfg(target_os = "linux")]
pub fn smt_siblings_outside(allowed: &[usize]) -> Option<Vec<usize>> {
    siblings_outside(allowed, |cpu| {
        std::fs::read_to_string(format!(
            "/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        ))
        .ok()
    })
}

#[cfg(not(target_os = "linux"))]
pub fn smt_siblings_outside(_allowed: &[usize]) -> Option<Vec<usize>> {
    None
}

/// The largest CPU number [`parse_cpu_list`] will accept.
///
/// No CPU above the affinity mask's own width can ever be in an allowed set, so
/// a larger number is malformed rather than merely unusual. The bound exists
/// because `cpus.extend(lo..=hi)` allocates eagerly: a truncated or corrupt read
/// of `0-18446744073709551614` would otherwise try to allocate exabytes and
/// abort the benchmark. This crate's whole argument is that it does not rely on
/// "the kernel would never emit that".
///
/// A plain constant rather than `8 * size_of::<libc::cpu_set_t>()`, because
/// `cpu_set_t` is Linux-only and this parser is not. `glibc`'s `CPU_SETSIZE` is
/// 1024, and the Linux-only test `the_cpu_list_cap_covers_the_real_affinity_mask`
/// asserts the constant cannot silently drift below the platform's real mask
/// width.
pub const MAX_CPU: usize = 1024;

/// Parses a kernel CPU list such as `0-3,8,12-13` into individual CPU numbers.
///
/// This is the format of `Cpus_allowed_list` in `/proc/<pid>/status`, which is
/// range notation -- unlike [`AllowedCpus::label`], which is a flat comma list.
/// A malformed field returns `None` rather than a partial set, because a
/// silently truncated mask would read as "this thread is confined to fewer CPUs
/// than it is", which is the direction that fabricates a passing check.
pub fn parse_cpu_list(list: &str) -> Option<Vec<usize>> {
    let mut cpus = Vec::new();
    for part in list.trim().split(',').filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((lo, hi)) => {
                let (lo, hi) = (lo.trim().parse().ok()?, hi.trim().parse::<usize>().ok()?);
                if hi < lo || hi >= MAX_CPU {
                    return None;
                }
                cpus.extend(lo..=hi);
            }
            None => {
                let cpu = part.trim().parse().ok()?;
                if cpu >= MAX_CPU {
                    return None;
                }
                cpus.push(cpu);
            }
        }
    }
    if cpus.is_empty() { None } else { Some(cpus) }
}

/// How many of this process's threads can run *outside* `allowed`.
///
/// # Why this is measured rather than assumed
///
/// Foreign CPU is `busy on the allowed set` minus `this process's own CPU`, but
/// own CPU is read process-wide from `/proc/self/stat` while busy is restricted
/// to the allowed set. The subtraction is only valid when every thread of the
/// process is confined to that set. When one is not, its CPU is subtracted
/// without ever having been added, and the result *under*-reports contention --
/// it moves a contended reading toward "clean", which is the direction that
/// publishes a wrong number rather than merely losing a row.
///
/// The original single-consumer version of this code carried that as a
/// documented precondition, which held because the EP narrows the main thread's
/// affinity before any pool thread exists and children inherit the mask. A
/// documented precondition is fine for one caller who has checked it; it is not
/// fine for a shared crate, where the first caller with an unconfined logging or
/// telemetry thread gets a quiet under-report and no indication. So it is
/// checked.
///
/// `None` when the thread list cannot be read at all, which is treated as
/// "unknown" and therefore not-confined by callers -- unknown must not be
/// allowed to certify a reading as complete.
/// One thread's `/proc/<tid>/status`, as seen by the scan.
#[derive(Clone, Copy, Debug)]
pub enum ThreadStatus<'a> {
    /// The file was read.
    Read(&'a str),
    /// The file was gone -- the thread exited between the directory listing and
    /// the read.
    Vanished,
    /// The file existed but could not be read.
    Unreadable,
}

/// Decides the off-mask count from a set of thread status blobs.
///
/// Split from the directory walk in [`threads_off_mask`] so the decision can be
/// asserted. The cases that matter -- a live thread whose mask could not be
/// established -- cannot be produced by pointing the real scan at a real
/// `/proc`, so testing them through the I/O would mean not testing them at all.
///
/// The rule is that only a *vanished* thread may be skipped. Every other
/// failure is a live thread of unknown affinity, and letting the remaining
/// threads certify the subtraction over it is exactly the "number that was never
/// measured" this crate exists to remove.
pub fn off_mask_from_statuses(statuses: &[ThreadStatus<'_>], allowed: &[usize]) -> Option<usize> {
    let mut off = 0usize;
    let mut seen_any = false;
    for status in statuses {
        let text = match status {
            ThreadStatus::Read(text) => *text,
            ThreadStatus::Vanished => continue,
            ThreadStatus::Unreadable => return None,
        };
        let line = text
            .lines()
            .find_map(|l| l.strip_prefix("Cpus_allowed_list:"))?;
        let cpus = parse_cpu_list(line)?;
        seen_any = true;
        if cpus.iter().any(|c| !allowed.contains(c)) {
            off += 1;
        }
    }
    if seen_any { Some(off) } else { None }
}

#[cfg(target_os = "linux")]
pub fn threads_off_mask(allowed: &[usize]) -> Option<usize> {
    let mut statuses = Vec::new();
    for entry in std::fs::read_dir("/proc/self/task").ok()? {
        let Ok(entry) = entry else {
            return None;
        };
        statuses.push(match std::fs::read_to_string(entry.path().join("status")) {
            Ok(text) => Ok(text),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Err(ThreadStatus::Vanished),
            Err(_) => Err(ThreadStatus::Unreadable),
        });
    }
    let borrowed: Vec<ThreadStatus<'_>> = statuses
        .iter()
        .map(|s| match s {
            Ok(text) => ThreadStatus::Read(text),
            Err(other) => *other,
        })
        .collect();
    off_mask_from_statuses(&borrowed, allowed)
}

#[cfg(not(target_os = "linux"))]
pub fn threads_off_mask(_allowed: &[usize]) -> Option<usize> {
    None
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
    /// Threads able to run outside `allowed` at snapshot time; `None` when the
    /// thread list could not be read. Any value other than `Some(0)` means the
    /// own-CPU subtraction is incomplete and the foreign figure is a *lower
    /// bound* rather than an estimate.
    off_mask: Option<usize>,
    /// SMT siblings of `allowed` that are not themselves allowed, and their busy
    /// jiffies, paired positionally. `None` when the topology could not be read,
    /// which is distinct from `Some(vec![])` -- "no siblings" is a measurement
    /// and "topology unknown" is not.
    siblings: Option<Vec<usize>>,
    sibling_busy: Vec<u64>,
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
    /// Whether every thread was confined to the set **at both ends of the
    /// window**, which is the precondition that makes `foreign_pct` an estimate
    /// rather than a lower bound. See [`threads_off_mask`].
    ///
    /// A boundary property, not a window property, and the difference is real:
    /// a thread spawned wide, burning CPU off-set, and joined entirely between
    /// the two snapshots is invisible to both, while its CPU still lands in the
    /// process-wide `own` total. `true` therefore means "no off-mask thread was
    /// observed at either boundary", which is exact for a stable thread set --
    /// the case both shipped consumers are in, since the EP's decode pool and
    /// ORT's intra-op pool are built once and persist -- and is not a guarantee
    /// for a caller that spawns transient wide threads inside a measured
    /// window. Sampling more often would narrow that hole without closing it,
    /// so it is stated rather than papered over.
    pub own_time_complete: bool,
    /// The busiest single SMT sibling of the confined set that is not itself in
    /// the set, as a percentage of that one CPU.
    ///
    /// Bounded below at zero and *not* clamped above it, for the same reason
    /// [`foreign_pct`](Contention::foreign_pct) claims no upper bound: jiffy
    /// accounting is coarse and the two reads are not simultaneous, so a
    /// saturated sibling routinely differences to a little over `100.0`. A
    /// clamp would turn that measurement artefact into a silent one.
    ///
    /// This is the contention `foreign_pct` cannot see: a decode budget picks one
    /// logical CPU per physical core, so the partner CPU of every core in use is
    /// outside the mask, and a co-runner there halves a core while contributing
    /// nothing to any allowed CPU's busy total. See the module docs for the
    /// pinned-spinner experiment that establishes both the effect and the
    /// blindness.
    ///
    /// **Peak, not sum**, and deliberately not the same normalisation as
    /// `foreign_pct`: a dispatch is a barrier, so one saturated sibling gates it
    /// while a dozen lightly-loaded ones do not. Summing would let ambient noise
    /// spread over a wide set outvote the single core actually being halved.
    ///
    /// Needs no own-time subtraction -- this process cannot run on these CPUs, so
    /// every busy jiffy there is foreign by construction. `own_time_complete`
    /// therefore does not qualify this figure.
    pub sibling_peak_pct: f64,
    /// Whether the sibling topology was readable and stable across the window.
    ///
    /// `false` means `sibling_peak_pct` is **not measured**, never "measured and
    /// zero". Separated from `measured` because the CPU-time reads and the sysfs
    /// topology read can fail independently, and a foreign figure is still worth
    /// having when only the topology is missing.
    pub siblings_known: bool,
}

impl Contention {
    /// Whether this window is provably contended.
    ///
    /// Sound on a lower bound, and deliberately so: when
    /// [`own_time_complete`](Self::own_time_complete) is false `foreign_pct`
    /// under-reports, so a figure above the threshold proves the true value is
    /// above it too. That is what lets an incomplete window still condemn a row,
    /// and it is the property [`is_clean`](Self::is_clean) cannot share -- do
    /// not "fix" this by requiring `own_time_complete` here.
    ///
    /// A cell whose foreign CPU is above this is not comparable to a clean one.
    /// Deliberately low: at width 2 a foreign *tenth* of a core is already a
    /// measurable barrier tax, and the cost of over-flagging is a re-run while
    /// the cost of under-flagging is a published wrong number.
    pub fn is_contended(&self) -> bool {
        (self.measured && self.foreign_pct > 5.0)
            || (self.siblings_known && self.sibling_peak_pct > SIBLING_CONTENDED_PCT)
    }

    /// Whether this window can be relied on as *clean*.
    ///
    /// Deliberately not `!is_contended()`. The two are asymmetric because an
    /// incomplete own-time subtraction makes `foreign_pct` a lower bound: a
    /// lower bound above the threshold still proves contention, so
    /// [`is_contended`](Self::is_contended) stays sound without
    /// `own_time_complete`, while a lower bound below the threshold proves
    /// nothing at all. Concluding "quiet" therefore needs the stronger
    /// condition, and a caller that gates a benchmark on `!is_contended()` gets
    /// the weaker one by accident.
    pub fn is_clean(&self) -> bool {
        self.measured
            && self.own_time_complete
            && self.foreign_pct <= 5.0
            && self.siblings_known
            && self.sibling_peak_pct <= SIBLING_CLEAN_PCT
    }
}

/// A sibling above this is provably co-running with a decode worker.
///
/// Set well above the ambient sibling occupancy measured on a shared host
/// (peaks of 12-22% with nothing pathological happening) and well below the
/// 75-98% observed when a real co-tenant lands on a sibling, so the two cannot
/// be confused in either direction. Half a core is also the point past which the
/// partner core's throughput loss stops being arguable.
pub const SIBLING_CONTENDED_PCT: f64 = 50.0;

/// A window is not certified clean unless every sibling is below this.
///
/// Lower than [`SIBLING_CONTENDED_PCT`], for the same asymmetry that separates
/// [`Contention::is_contended`] from [`Contention::is_clean`] on `foreign_pct`:
/// the band between the two is "not proven contended, not certifiable as quiet",
/// and a caller asking whether it may publish a number must land outside it.
/// Calling that band clean would certify exactly the readings most likely to be
/// mildly and invisibly degraded.
pub const SIBLING_CLEAN_PCT: f64 = 25.0;

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
    let mut cell = format!("{median:.1}");
    if seen.len() < reps.len() {
        cell.push('*');
    }
    // A trailing `!` means the figure is a lower bound, not an estimate. Marked
    // rather than suppressed: the bound is still usable to *flag* a row, and
    // blanking it would hide a large positive reading that is real.
    //
    // Deliberately a suffix. The obvious spelling of a lower bound is a leading
    // `>`, but this cell is a column in a table that gets parsed, and `awk
    // '$NF + 0'` on `>9.0` yields `0.0` -- silently turning the most contended
    // row in a matrix into the cleanest-looking one. That is precisely the
    // failure this module exists to prevent, one hop downstream. With the marker
    // trailing, the same awk yields `9.0`: the qualifier is lost but the number
    // survives, and Python's `float()` raises instead of guessing. Both are the
    // safe direction.
    if reps.iter().any(|c| c.measured && !c.own_time_complete) {
        cell.push('!');
    }
    cell
}

/// Renders the `sib_%` cell for a group of repetitions.
///
/// Peak over the repetitions, not median, and this is the one place the two
/// columns deliberately disagree on their summary statistic. `foreign_%` takes a
/// median because it is estimating a level that persisted through the window.
/// `sibling_peak_pct` is already a peak over CPUs, and it exists to answer "did a
/// co-runner appear on any core I was using": a co-tenant that saturated a
/// sibling for one repetition of three invalidates that repetition, and a median
/// would discard it as the odd one out. Taking the max keeps the evidence.
///
/// `n/a` when no repetition had readable topology, and `*` when only some did --
/// same contract as [`foreign_column`], for the same reason: a cell backed by
/// fewer samples than its neighbours has to say so.
pub fn sibling_column(reps: &[Contention]) -> String {
    let seen: Vec<f64> = reps
        .iter()
        .filter(|c| c.siblings_known)
        .map(|c| c.sibling_peak_pct)
        .collect();
    if seen.is_empty() {
        return "n/a".to_string();
    }
    let peak = seen
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |a, b| if b > a { b } else { a });
    let mut cell = format!("{peak:.1}");
    if seen.len() < reps.len() {
        cell.push('*');
    }
    cell
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
    // Read from the same `/proc/stat` text as `busy`, so the sibling and allowed
    // figures describe the same instant rather than two reads a syscall apart.
    let siblings = smt_siblings_outside(&allowed.cpus);
    let mut sibling_busy = Vec::new();
    let mut siblings = siblings;
    if let Some(list) = &siblings {
        for &cpu in list {
            let prefix = format!("cpu{cpu} ");
            match stat
                .lines()
                .find(|l| l.starts_with(&prefix))
                .and_then(busy_jiffies_of_cpu_line)
            {
                Some(jiffies) => sibling_busy.push(jiffies),
                // A sibling the topology named but `/proc/stat` does not carry
                // (offlined between the two reads) makes the set incomplete.
                // Drop to "topology unknown" rather than report a peak over the
                // CPUs that happened to still be there.
                None => {
                    siblings = None;
                    sibling_busy.clear();
                    break;
                }
            }
        }
    }
    // Anchored here rather than in the struct literal below, so the timestamp
    // sits next to the jiffy reads it has to be differenced against. The
    // thread scan that follows is bounded by the number of threads, and leaving
    // it inside the bracket would add a thread-count-dependent offset to
    // `window` that is not present in `busy` or `own`.
    let taken = Instant::now();
    let off_mask = threads_off_mask(&allowed.cpus);

    Some(ContentionSnapshot {
        taken,
        allowed: allowed.cpus,
        busy,
        own,
        off_mask,
        siblings,
        sibling_busy,
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

    // The sibling peak is computed independently of the foreign arithmetic
    // above: it needs no own-time subtraction, so it survives a window whose
    // `foreign_pct` is only a lower bound. Requires the sibling *set* to be
    // identical at both ends for the same reason the allowed set must be --
    // differencing per-CPU counters across a set that moved compares different
    // CPUs.
    let (sibling_peak_pct, siblings_known) = match (&before.siblings, &after.siblings) {
        (Some(b), Some(a)) if b == a && before.sibling_busy.len() == after.sibling_busy.len() => {
            let peak = after
                .sibling_busy
                .iter()
                .zip(&before.sibling_busy)
                .map(|(a, b)| a.saturating_sub(*b) as f64 / tick / window * 100.0)
                .fold(0.0f64, f64::max);
            (peak, true)
        }
        _ => (0.0, false),
    };

    Contention {
        foreign_pct: foreign_s / window * 100.0,
        total_pct: total_s / window * 100.0,
        measured: true,
        // Both ends must be clean. Requiring only the later snapshot would
        // certify a window that a thread was off-mask for the whole first half
        // of, since it may have been joined before the end. Checking both is
        // strictly stronger, though still a boundary property -- see
        // `own_time_complete`.
        own_time_complete: before.off_mask == Some(0) && after.off_mask == Some(0),
        sibling_peak_pct,
        siblings_known,
    }
}

impl ContentionSnapshot {
    /// Builds a snapshot from known parts.
    ///
    /// This exists because the tests are integration tests, so they see only
    /// the public surface and cannot reach these fields directly. One
    /// deliberate seam is cheaper than making the fields public, which would
    /// let a caller construct a snapshot whose `allowed` and `busy` disagree in
    /// length.
    ///
    /// Synthetic parts are also what the arithmetic tests want: differencing two
    /// fabricated snapshots asserts the formula, whereas sampling the real host
    /// asserts whatever the host happened to be doing.
    pub fn from_parts(taken: Instant, allowed: Vec<usize>, busy: Vec<u64>, own: u64) -> Self {
        Self::from_parts_with_off_mask(taken, allowed, busy, own, Some(0))
    }

    /// As [`from_parts`](Self::from_parts) but with an explicit off-mask thread
    /// count, so a test can construct the incomplete-subtraction case without
    /// having to actually escape the affinity mask.
    pub fn from_parts_with_off_mask(
        taken: Instant,
        allowed: Vec<usize>,
        busy: Vec<u64>,
        own: u64,
        off_mask: Option<usize>,
    ) -> Self {
        Self {
            taken,
            allowed,
            busy,
            own,
            off_mask,
            siblings: None,
            sibling_busy: Vec::new(),
        }
    }

    /// As [`from_parts`](Self::from_parts) but carrying a sibling set and its
    /// busy jiffies, so a test can construct an SMT topology the host running
    /// the test does not have.
    ///
    /// Defaults elsewhere leave `siblings` as `None` -- "topology unknown" --
    /// which keeps every pre-existing test asserting exactly what it asserted
    /// before rather than silently acquiring a second, unexamined dimension.
    pub fn from_parts_with_siblings(
        taken: Instant,
        allowed: Vec<usize>,
        busy: Vec<u64>,
        own: u64,
        siblings: Option<Vec<usize>>,
        sibling_busy: Vec<u64>,
    ) -> Self {
        Self {
            taken,
            allowed,
            busy,
            own,
            off_mask: Some(0),
            siblings,
            sibling_busy,
        }
    }
}

/// Ticks per second, exposed so a test can express jiffies in core-seconds.
pub fn clock_tick_hz() -> f64 {
    clock_tick()
}

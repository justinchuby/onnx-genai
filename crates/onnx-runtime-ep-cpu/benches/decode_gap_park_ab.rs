//! Gap-aware decode harness: what the persistent SPMD pool's spin/park policy
//! costs as a function of the **inter-token gap**.
//!
//! # Why this exists
//!
//! `int4_decode_loop_ab` drives tokens back to back with no gap. That is the one
//! workload on which parking can never pay off: the next op is always already
//! there, so a worker that parks has strictly wasted a wake and a worker that
//! spins is always right. Tuning `ONNX_GENAI_CPU_DECODE_BLOCKTIME_US` against a
//! zero-gap loop therefore has a foregone conclusion -- "spin longer" -- and
//! says nothing about a served process, where the engine spends time in
//! sampling, detokenization, KV bookkeeping and the network between tokens, and
//! where a pool that holds 16 cores through every gap is burning a machine to
//! save a microsecond.
//!
//! The blocktime window is the knob that trades those two against each other,
//! and it cannot be set from a measurement that cannot see one side of the
//! trade. So this harness puts a **configurable gap between tokens** and reports
//! both sides in the same row: decode latency (which parking hurts) and CPU/
//! context-switch cost (which parking helps).
//!
//! # Relationship to `tests/task_runtime_latency.rs`
//!
//! That test is the closest existing thing and it measures a **different pool**.
//! `task_runtime/pool.rs` has an *adaptive* window (20 us doubling to 500 us on
//! a spin hit, halving on a park); `decode_spmd` has a *fixed* 500 us one. They
//! are separate executors with separate wait policies, and production int4
//! decode runs on the second. Conclusions do not transfer between them and a
//! number from one must never be quoted for the other.
//!
//! What *is* shared is the expected **shape**: flat while the gap fits inside
//! the window, then rising as the gap crosses it and every token starts paying a
//! wake. Validating this harness against `task_runtime_latency` means checking
//! that shape appears here too, not that the milliseconds agree.
//!
//! # Reading a row
//!
//! | column | meaning |
//! |---|---|
//! | `kind` | `busy` (the gap burns a core) or `sleep` (the gap idles); a leading `~` marks the **null control** for that generator -- the same gaps, the same RNG stream, no decode |
//! | `gap_us` | requested mean gap |
//! | `gap_act` | *measured* mean gap. `sleep` has ~50 us granularity, so a requested 20 us is not a 20 us gap and the row says so rather than mislabelling its own axis |
//! | `ms_tok` / `p90` | decode time per token, gap **excluded** -- the clock starts after the gap |
//! | `cpu_ms` / `sys_ms` | process user/system CPU per token from `getrusage`. Includes the harness's own gap generator, which is what the `null` rows bound |
//! | `disp/tok` | barriers per token. A route invariant: if it moves across gaps, the gap knob changed the work, not the scheduling |
//! | `spin/tok`, `park/tok` | op-observations caught in the spin window vs paid for with a futex wake |
//! | `park%` | `parks / (spin_hits + parks)` -- the width-independent form, and the quantity blocktime actually moves |
//! | `obs/disp` | op-observations per dispatch. Derived, and its job is to be **boring**: integral and identical in every cell. If it moves, that cell dispatched to a different number of workers and its latency is not comparable however tight its spread. It is the count of *spawned* workers, which is the header's `realized` width **minus one** whenever the inline dispatcher owns a compute shard -- so `obs/disp = 15` beside `realized = 16` is correct and informative, not an off-by-one. That case is reachable on the explicit-budget path and not on the default one, which is exactly why the two are not interchangeable arms |
//! | `inline/tok` | dispatches that found the slot claimed and ran serially. Nonzero means concurrent sessions or nested dispatch contended for the pool |
//! | `dy/tok` | dispatcher yields per token. Strongly **width-dependent**: the dispatcher publishes, computes its own shard, then spins ~10 us before yielding once, so a yield means the last worker lagged the dispatcher's own arrival by more than that. Measured at zero gap on an idle host: 0.004/dispatch at width 4, ~0.9 at width 16, non-monotonic between. Straggler spread, not contention, and only comparable at a fixed width |
//! | `vcsw` / `ivcsw` | voluntary / involuntary context switches per token. `vcsw` is the kernel's independent view of parking and should track `park/tok` |
//! | `rss_mb` | `ru_maxrss`, a process **high-water mark**. Absolute, never a delta -- a high-water mark cannot be differenced, and doing so prints noise that looks like a leak |
//! | `foreign_%` | CPU consumed by *other processes* on this process's confined core set, as a percentage of one core, median over the reps that were measurable. Read this **before** `ms_tok`: a dispatch is a barrier, so foreign work on the confined set costs the whole dispatch and `ms_tok` is not comparable across rows with different values here. `cpu_ms` is. Prints `n/a` if no rep could be measured and suffixes `*` if only some could -- never a clean `0.0` for an unmeasured window |
//! | `spread%` | `(max - min) / median` of `ms_tok` across repetitions |
//!
//! # Controls
//!
//! Three, because every defect this line of work has hit was a number that was
//! never measured rather than a number that was wrong:
//!
//! 1. **Null rows** (`~busy`, `~sleep`) run the gap generator, the RNG and the
//!    timing scaffolding with no decode at all, one per generator so a row's
//!    control is built from the same generator it controls. A busy gap burns a
//!    core by construction, so `cpu_ms` on a `busy` row is *mostly the harness*
//!    at large gaps; `~busy` at the same gap is the amount to subtract before
//!    claiming the pool spent anything. `~sleep` bounds a different cost that
//!    lands in the same column parking is read from: `nanosleep`'s own voluntary
//!    context switches. Null rows also assert the scheduler counters do not
//!    move, which is what proves the generator never dispatches.
//! 2. **Interleaving**: the whole cell list is run once per repetition rather
//!    than each cell being repeated in place, so a cell's repetitions are
//!    separated in time by the rest of the matrix. A drifting host then shows up
//!    as `spread%` instead of silently ordering the results.
//! 3. **An output hash** over every cell's final activations. The gap is
//!    supposed to change *when* work happens and never *what* it computes; if
//!    two cells disagree the harness prints `GAP-CHANGED-OUTPUT` and the matrix
//!    is void.
//!
//!    Its limits, stated because the first draft of this comment overclaimed
//!    them: `MatMulNBits` at `m = 1` does each output column's whole K-reduction
//!    inside one worker's shard, so the result is bit-identical no matter which
//!    pool ran it or how wide. Verified -- width 4 and width 16 print the same
//!    digest. So this catches a gap that changed the *computation*; it cannot
//!    catch one that changed only the *schedule*. `obs/disp` and the header's
//!    realized width are the controls for that.
//!
//! # Env
//!
//! - `PROBE_GAP_US_LIST` -- mean inter-token gaps in microseconds (default
//!   `0,20,100,500,2000,10000`). Chosen to straddle the 500 us default window:
//!   two points inside it, one on it, two beyond.
//! - `PROBE_GAP_KIND` -- `busy` | `sleep` | `both` (default `both`). A busy gap
//!   holds the core, so the workers' spinning competes with it; a sleep gap
//!   leaves the machine idle. Real engines do some of each and the two bracket
//!   it.
//! - `PROBE_GAP_DIST` -- `fixed` | `exp` (default `exp`). A fixed gap can sit
//!   entirely on one side of the spin window and manufacture a cliff that no
//!   real workload has, and it can phase-lock with the window. An exponential
//!   with the same mean straddles the window and reports the average cost of a
//!   *distribution*, which is what a server sees.
//! - `PROBE_NULL` -- set `0` to drop the null rows (default on).
//! - `PROBE_WARMUP` -- warmup steps per session **with the gap applied**
//!   (default 32). Warming only the pool construction is not enough: the first
//!   gap of a cell finds every worker hot from the previous cell, so a short
//!   warmup measures the previous row's park state.
//! - `PROBE_MODEL`, `PROBE_BLOCK`, `PROBE_ACCURACY`, `PROBE_SESSIONS`,
//!   `PROBE_TOKENS`, `PROBE_LAYERS`, `PROBE_SPMD`, `PROBE_REPS` -- identical
//!   meaning to `int4_decode_loop_ab`, which shares this harness's workload
//!   verbatim through `common::decode_workload`.
//!
//! To vary the pool width set `ONNX_GENAI_CPU_DECODE_THREADS` (**not**
//! `RAYON_NUM_THREADS`, which does not size this pool). To vary the spin window
//! set `ONNX_GENAI_CPU_DECODE_BLOCKTIME_US` -- and note that it latches into a
//! `OnceLock` on first use, so a blocktime sweep is **across process launches**.
//! Setting it twice in one process changes nothing and both arms silently
//! measure the same window. The header prints the window the run actually
//! resolved, read back from the EP rather than re-parsed here.
//!
//! # First result: what the 500 us default window actually buys
//!
//! llama shapes, block 32, accuracy 0, 1 session, width 16 (realized, asserted
//! in the header), 64 tokens after 32 gap-warmed steps, 3 interleaved
//! repetitions, `sleep` gaps drawn `exp(mean = 2000 us)` (measured mean 2790 us),
//! four independent process launches on an idle host:
//!
//! | blocktime | ms_tok | spread | user cpu_ms/tok | **sys_ms/tok** | park% |
//! |---|---|---|---|---|---|
//! | 0 us | 6.477 | 4.5% | 81.0 | **5.55** | 30.6 |
//! | 50 us | 6.022 | 15.2% | 80.7 | **5.35** | 28.4 |
//! | 500 us (default) | 6.963 | 1.7% | 75.6 | **22.30** | 33.4 |
//! | 5000 us | 7.042 | 0.8% | 82.9 | **65.43** | 3.2 |
//!
//! Each row is one process launch, which -- see the bimodality section below --
//! makes the `ms_tok` column a draw from a two-mode distribution rather than a
//! point estimate. That weakens nothing here: the conclusion drawn from it is
//! that latency is *indistinguishable*, and an unmodelled second mode can only
//! make it more so. The `sys_ms` column is not affected, being monotone across
//! four points with a mechanism.
//!
//! Two things fall out, and the second was not what the window was believed to
//! be trading:
//!
//! 1. **Latency is indistinguishable across four orders of magnitude of
//!    window.** The 0.5 ms range spanned by `ms_tok` is inside the repetitions'
//!    own spread. A wake is a few microseconds against a 7 ms token, so at this
//!    op size the window cannot move the number it exists to protect. That is a
//!    statement about *this shape*, not a proposal to change the default: a
//!    model with many more, much smaller ops has a different ratio, and this
//!    harness is how that gets measured rather than assumed.
//!
//! 2. **`sys_ms` scales with the window and not with parking.** It rises 4x from
//!    a 0 us to a 500 us window and 12x to 5000 us, while `park/tok` moves in
//!    the *opposite* direction (24.5 -> 26.7 -> 2.6). So the system time is not
//!    futex wakes: it is `worker_wait`'s `sched_yield` loop, which runs for the
//!    remainder of the window after the first ~4096 pure spins. A longer window
//!    is more `sched_yield` syscalls. Meanwhile user `cpu_ms` is flat, because a
//!    yielding worker on an idle host is rescheduled immediately and burns the
//!    core either way.
//!
//! Point 2 supersedes the reading that the ~20x `sys` jump at width >= 4 was
//! futex/wake traffic from `dispatch_rows_across_workers` fanning out. It is the
//! yield loop, and width enters only by multiplying the number of workers
//! running it. Anything that shortens the window -- or replaces the yield ramp
//! with a park -- moves that column; adding or removing wakes does not.
//!
//! # Second result: launches were bimodal, and the cause was off-process
//!
//! Read the table above as a distribution, not as four numbers. At width 16 and
//! zero gap, repeated launches of the *same* binary with the *same* environment
//! land in one of two clearly separated regimes, and `park%` names which:
//!
//! | regime | `ms_tok` | `cpu_ms/tok` | `park%` | in-run `spread_%` |
//! |---|---|---|---|---|
//! | fast | ~2.51 | ~41.8 | ~5 | 0.1--0.5 |
//! | slow | ~11.5--11.9 | ~55--65 | ~71--73 | 1.4--78 |
//!
//! Twelve launches across three configurations: every configuration produced
//! both regimes. This matters more than any single row here, for two reasons.
//!
//! **Resolved: it is foreign CPU on the confined core set.** See the
//! `foreign_%` column and `common::host_contention`. `ONNX_GENAI_CPU_DECODE_THREADS=N`
//! confines the process to N CPUs, and a dispatch is a *barrier*, so one
//! unrelated thread on one of those N cores costs the whole dispatch rather than
//! a share of it -- every other worker finishes and idles waiting. The workers
//! really are asleep rather than thrashing, which is why the slow regime burns
//! less CPU; they are asleep waiting on a shard whose core is being timeshared
//! with somebody else. Injecting one `taskset`-pinned spinner reproduces the
//! slow regime on demand and removing it restores the fast one, while the same
//! spinner placed *outside* the confined set changes nothing.
//!
//! The two paragraphs below are kept because their methodological point stands
//! on its own and is what led here -- but the cause is no longer unknown, and a
//! run is no longer un-diagnosable: read `foreign_%` before reading `ms_tok`.
//!
//! **A small sample will hand you a confident wrong answer.** Three launches of
//! an explicit `ONNX_GENAI_CPU_DECODE_THREADS=16` arm against three of the
//! default arm gave 9.3/11.0/12.0 against 4.83/4.82/4.81 -- consistent, tight
//! in-run spreads, and a clean mechanistic story (the explicit path builds 15
//! workers plus a dispatcher shard rather than 16 workers). A fourth and fifth
//! launch put the explicit arm at 2.51 and the default arm at 10.0, and the
//! story evaporated. In-run `spread_%` did not warn: the fast regime's runs are
//! internally *stable*, at 0.1--0.5%.
//!
//! So: **no per-launch number from this harness is publishable at width >= 8
//! without repeated launches**, and a difference between two configurations
//! needs enough launches to see both regimes in both arms. Ratios taken *within*
//! one launch (the interleaved cells) are safe -- that is what the interleaving
//! is for -- but a comparison across launches is not. The blocktime sweep above
//! survives this because its conclusion rests on `sys_ms` moving 12x
//! monotonically across four points with a mechanism, and on the latency column
//! being *indistinguishable*; bimodality can only widen that latter claim.
//!
//! What made this so durable is worth recording. The slow regime is internally
//! *stable* (in-run `spread_%` of 0.1--0.5), it persists for a whole run and
//! across consecutive runs while the foreign process lives, and it presents at
//! unchanged CPU per token. Every one of those reads as a property of the code
//! under test rather than as a stranger on one of two cores. The flat
//! `cpu_ms/tok` beside a doubled `ms_tok` is the fingerprint: same work, twice
//! the wall, which no code change of ours produces.

mod common;

use std::time::{Duration, Instant};

use common::Tensor;
use common::decode_workload::{Weight, asymmetric_zero_points, build_kernel, floats, weights};
use common::host_contention::{self, Contention};
use onnx_runtime_ep_api::Kernel;
use onnx_runtime_ep_cpu::{decode_spmd, with_decode_pool_scope};

/// Deterministic xorshift64*, so every cell sees the same gap sequence and two
/// rows differ only by their gap parameters.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in `(0, 1]`. The open end at zero matters: the exponential
    /// transform takes a logarithm.
    fn unit(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64 + 1.0) / ((1u64 << 53) as f64)
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GapKind {
    Busy,
    Sleep,
}

impl GapKind {
    fn label(self) -> &'static str {
        match self {
            Self::Busy => "busy",
            Self::Sleep => "sleep",
        }
    }
}

/// One row: a gap generator, a mean, and whether any decode runs at all.
///
/// The null control carries the *same* generator as the row it controls rather
/// than one fixed generator. The first version busy-waited for every null row,
/// which made the null a valid control for the `busy` rows and a meaningless one
/// for the `sleep` rows -- it bounded a cost the sleep rows never pay and said
/// nothing about the one they do (`nanosleep`'s own context switches, which land
/// in the same `vcsw` column parking is read from).
#[derive(Clone, Copy)]
struct Cell {
    kind: GapKind,
    gap_us: f64,
    decode: bool,
}

impl Cell {
    fn label(self) -> String {
        if self.decode {
            self.kind.label().to_string()
        } else {
            format!("~{}", self.kind.label())
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum GapDist {
    Fixed,
    Exp,
}

/// Draw one gap. `mean_us == 0` short-circuits so the zero-gap row is exactly
/// the back-to-back loop `int4_decode_loop_ab` runs, with no RNG call and no
/// clock read between tokens.
fn draw_gap(rng: &mut Rng, dist: GapDist, mean_us: f64) -> Duration {
    if mean_us <= 0.0 {
        return Duration::ZERO;
    }
    let us = match dist {
        GapDist::Fixed => mean_us,
        // Inverse-transform sampling: -mean * ln(U) is Exponential(1/mean).
        GapDist::Exp => -mean_us * rng.unit().ln(),
    };
    Duration::from_nanos((us * 1e3) as u64)
}

/// Spend `gap` and return what it actually cost. A busy gap holds the core; a
/// sleep gap releases it. The elapsed time is returned rather than assumed
/// because `thread::sleep` overshoots by tens of microseconds and a row that
/// printed its *request* would be mislabelling its own axis.
fn spend_gap(kind: GapKind, gap: Duration) -> Duration {
    if gap.is_zero() {
        return Duration::ZERO;
    }
    let start = Instant::now();
    match kind {
        GapKind::Busy => {
            while start.elapsed() < gap {
                std::hint::spin_loop();
            }
        }
        GapKind::Sleep => std::thread::sleep(gap),
    }
    start.elapsed()
}

/// FNV-1a over the raw bits of every output element.
fn hash_outputs(outputs: &[Tensor], seed: u64) -> u64 {
    let mut h = seed;
    for tensor in outputs {
        for value in tensor.f32s() {
            for byte in value.to_bits().to_le_bytes() {
                h ^= byte as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    h
}

/// Process-wide resource usage. Process-wide and not per-thread on purpose: the
/// pool's workers are the threads whose CPU and parking we are asking about, and
/// `RUSAGE_THREAD` on the driving thread cannot see them.
#[derive(Clone, Copy, Default)]
struct Usage {
    user_ms: f64,
    sys_ms: f64,
    /// High-water mark, in MiB. Never differenced -- see the column table.
    maxrss_mb: f64,
    vcsw: u64,
    ivcsw: u64,
}

#[cfg(unix)]
fn usage() -> Usage {
    // SAFETY: `getrusage` writes a plain POD struct through the pointer and
    // reads nothing else. The zeroed value is a valid `rusage`.
    let ru = unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut ru);
        ru
    };
    let ms = |tv: libc::timeval| tv.tv_sec as f64 * 1e3 + tv.tv_usec as f64 / 1e3;
    Usage {
        user_ms: ms(ru.ru_utime),
        sys_ms: ms(ru.ru_stime),
        // Linux reports kilobytes; macOS reports bytes.
        maxrss_mb: if cfg!(target_os = "macos") {
            ru.ru_maxrss as f64 / (1024.0 * 1024.0)
        } else {
            ru.ru_maxrss as f64 / 1024.0
        },
        vcsw: ru.ru_nvcsw as u64,
        ivcsw: ru.ru_nivcsw as u64,
    }
}

/// No `getrusage` off Unix. Returns zeros, and `main` prints a
/// `resource_metrics=unavailable` banner so a reader cannot mistake a column of
/// `0.000` for a measured one -- an unmeasured number that looks like a measured
/// number is the failure this harness exists to avoid.
#[cfg(not(unix))]
fn usage() -> Usage {
    Usage::default()
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

/// One measured pass of one cell.
#[derive(Clone, Copy)]
struct Pass {
    ms_tok: f64,
    p90: f64,
    gap_act_us: f64,
    cpu_ms_tok: f64,
    sys_ms_tok: f64,
    disp_tok: f64,
    spin_tok: f64,
    park_tok: f64,
    park_pct: f64,
    inline_tok: f64,
    dyield_tok: f64,
    vcsw_tok: f64,
    ivcsw_tok: f64,
    rss_mb: f64,
    /// Foreign CPU on the *confined* core set over this pass's window. See
    /// `common::host_contention`: a decode dispatch is a barrier, so foreign
    /// work on the confined set costs the whole dispatch rather than a share of
    /// it, and a host-wide load gate cannot see it.
    contention: Contention,
    /// Observations per dispatch: `(spin_hits + parks) / dispatches`. Derived,
    /// not measured, and its job is to be *boring* -- it must be integral and
    /// identical in every cell. A cell where it moves dispatched to a different
    /// number of workers than its neighbours, which makes that cell's latency
    /// incomparable no matter how tight its spread.
    ///
    /// This is the count of *spawned* workers, so it is one below the header's
    /// realized width whenever the inline dispatcher owns a compute shard.
    obs_disp: f64,
    hash: u64,
    /// Whether every session in the cell computed the same outputs.
    sessions_agree: bool,
}

fn main() {
    // Match the decode thread topology a served session runs in (#1749).
    common::init_decode_topology();

    let env_usize = |key: &str, default: usize| -> usize {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };

    let block_size = env_usize("PROBE_BLOCK", 32);
    let accuracy: i64 = std::env::var("PROBE_ACCURACY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let sessions = env_usize("PROBE_SESSIONS", 1);
    let tokens = env_usize("PROBE_TOKENS", 64);
    let layers = env_usize("PROBE_LAYERS", 1);
    let warmup = env_usize("PROBE_WARMUP", 32);
    let reps = env_usize("PROBE_REPS", 3);
    let spmd: bool = std::env::var("PROBE_SPMD")
        .map(|v| v != "0")
        .unwrap_or(true);
    let with_null: bool = std::env::var("PROBE_NULL")
        .map(|v| v != "0")
        .unwrap_or(true);

    let dist = match std::env::var("PROBE_GAP_DIST")
        .unwrap_or_else(|_| "exp".into())
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "fixed" => GapDist::Fixed,
        "exp" => GapDist::Exp,
        other => panic!("PROBE_GAP_DIST must be fixed or exp, got {other:?}"),
    };
    let kinds: Vec<GapKind> = match std::env::var("PROBE_GAP_KIND")
        .unwrap_or_else(|_| "both".into())
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "busy" => vec![GapKind::Busy],
        "sleep" => vec![GapKind::Sleep],
        "both" => vec![GapKind::Busy, GapKind::Sleep],
        other => panic!("PROBE_GAP_KIND must be busy, sleep or both, got {other:?}"),
    };
    let gaps: Vec<f64> = std::env::var("PROBE_GAP_US_LIST")
        .unwrap_or_else(|_| "0,20,100,500,2000,10000".into())
        .split(',')
        .map(|v| v.trim().parse().expect("PROBE_GAP_US_LIST must be numbers"))
        .collect();

    let asymmetric = asymmetric_zero_points();
    let weights: Vec<Weight> = weights(block_size, asymmetric);

    // Build the cell list once, so every repetition walks it in the same order
    // and a cell's repetitions are separated in time by the rest of the matrix.
    let mut cells: Vec<Cell> = Vec::new();
    for (index, &kind) in kinds.iter().enumerate() {
        for &gap in &gaps {
            // A zero gap is the same workload whichever generator is nominally
            // selected, so it gets one row rather than two identical ones.
            if gap <= 0.0 && index > 0 {
                continue;
            }
            cells.push(Cell {
                kind,
                gap_us: gap,
                decode: true,
            });
        }
    }
    if with_null {
        // One null per (generator, gap) that has decode rows, so every row has a
        // control built from the same generator it is controlling. A zero gap
        // spends nothing, so it needs no null.
        for &kind in &kinds {
            for &gap in &gaps {
                if gap > 0.0 {
                    cells.push(Cell {
                        kind,
                        gap_us: gap,
                        decode: false,
                    });
                }
            }
        }
    }

    println!(
        "model={} block_size={block_size} accuracy={accuracy} sessions={sessions} tokens={tokens} layers={layers} spmd={spmd} warmup={warmup} reps={reps} dist={}",
        std::env::var("PROBE_MODEL").unwrap_or_else(|_| "llama".into()),
        match dist {
            GapDist::Fixed => "fixed",
            GapDist::Exp => "exp",
        }
    );

    // One session's measured pass. `decode` false is the null control: the same
    // gaps, the same RNG stream, the same clock reads, no work.
    let run_session = |kind: GapKind,
                       gap_us: f64,
                       seed: u64,
                       decode: bool,
                       barrier: &std::sync::Barrier|
     -> (Vec<f64>, f64, u64) {
        let mut kernels: Vec<Box<dyn Kernel>> = Vec::new();
        let mut activations: Vec<Tensor> = Vec::new();
        let mut outputs: Vec<Tensor> = Vec::new();
        if decode {
            for _ in 0..layers {
                for weight in &weights {
                    kernels.push(build_kernel(
                        weight.k, weight.n, block_size, accuracy, asymmetric,
                    ));
                    activations.push(Tensor::floats(
                        common::FloatDType::F32,
                        &[1, weight.k],
                        &floats(weight.k, 1.1),
                    ));
                    outputs.push(Tensor::zeros(common::FloatDType::F32, &[1, weight.n]));
                }
            }
        }
        // Same move-in/move-out shape as `int4_decode_loop_ab`: `dyn Kernel` is
        // not `Sync` and `with_decode_pool_scope` may run the closure on another
        // thread.
        let mut state = (kernels, outputs);
        let step = |(kernels, mut outputs): (Vec<Box<dyn Kernel>>, Vec<Tensor>),
                    activations: &[Tensor],
                    weights: &[Weight]|
         -> (Vec<Box<dyn Kernel>>, Vec<Tensor>) {
            for (index, kernel) in kernels.iter().enumerate() {
                let weight = &weights[index % weights.len()];
                let ins = weight.inputs(&activations[index]);
                kernel
                    .execute(&ins, &mut [outputs[index].view_mut()])
                    .expect("execute");
            }
            (kernels, outputs)
        };

        let mut rng = Rng::new(seed);
        // Warm up **through the gap**, not just through pool construction. The
        // park/wake state at the first measured token is otherwise inherited
        // from whichever cell ran before this one.
        for _ in 0..warmup {
            spend_gap(kind, draw_gap(&mut rng, dist, gap_us));
            if decode {
                let (acts, ws) = (&activations, &weights);
                let moved = state;
                state = with_decode_pool_scope(spmd, move || step(moved, acts, ws));
            }
        }

        let mut samples = Vec::with_capacity(tokens);
        let mut gap_total = Duration::ZERO;
        // Two-phase rendezvous on one reusable barrier. The first wait says
        // "this session has finished warming"; the second releases everyone at
        // once. The driving thread reads the counters *between* the two, which
        // is the only point at which every session is simultaneously warm and
        // not yet measuring. With a single barrier the sessions race ahead of
        // the snapshot and the first dispatch of each is charged to nobody --
        // visible in the smoke run as `disp/tok = 4.9` where the workload emits
        // exactly 5.
        barrier.wait();
        barrier.wait();
        for _ in 0..tokens {
            // The gap is spent outside the clock and outside the pool scope:
            // this measures what the gap does to the *next* token, not the gap.
            gap_total += spend_gap(kind, draw_gap(&mut rng, dist, gap_us));
            if !decode {
                samples.push(0.0);
                continue;
            }
            let (acts, ws) = (&activations, &weights);
            let moved = state;
            let start = Instant::now();
            let (returned, elapsed) = with_decode_pool_scope(spmd, move || {
                let returned = step(moved, acts, ws);
                (returned, start.elapsed().as_secs_f64() * 1e3)
            });
            state = returned;
            samples.push(elapsed);
        }
        let gap_act_us = gap_total.as_secs_f64() * 1e6 / tokens as f64;
        (
            samples,
            gap_act_us,
            hash_outputs(&state.1, 0xcbf2_9ce4_8422_2325),
        )
    };

    let measure = |cell: Cell| -> Pass {
        let Cell {
            kind,
            gap_us,
            decode,
        } = cell;
        let barrier = std::sync::Barrier::new(sessions + 1);
        let (per_session, before, after) = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..sessions)
                .map(|session| {
                    let barrier = &barrier;
                    // Distinct streams per session, deterministic per cell.
                    let seed = 0x5eb_0000 ^ ((session as u64) << 8);
                    scope.spawn(move || run_session(kind, gap_us, seed, decode, barrier))
                })
                .collect();
            barrier.wait();
            let before = (
                usage(),
                decode_spmd::counters().unwrap_or_default(),
                host_contention::snapshot(),
            );
            barrier.wait();
            let out: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
            let after = (
                usage(),
                decode_spmd::counters().unwrap_or_default(),
                host_contention::snapshot(),
            );
            (out, before, after)
        });

        let total_tokens = (sessions * tokens) as f64;
        let (ru0, c0, cn0) = before;
        let (ru1, c1, cn1) = after;
        let contention = host_contention::contention(cn0.as_ref(), cn1.as_ref());
        let per = |a: u64, b: u64| (b.saturating_sub(a)) as f64 / total_tokens;
        let spin = per(c0.spin_hits, c1.spin_hits);
        let park = per(c0.parks, c1.parks);

        let mut all: Vec<f64> = per_session.iter().flat_map(|s| s.0.clone()).collect();
        all.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let hashes: Vec<u64> = per_session.iter().map(|s| s.2).collect();
        let gap_act_us = per_session.iter().map(|s| s.1).sum::<f64>() / per_session.len() as f64;

        Pass {
            ms_tok: median(all.clone()),
            p90: all[(all.len() * 9 / 10).min(all.len() - 1)],
            gap_act_us,
            cpu_ms_tok: (ru1.user_ms - ru0.user_ms) / total_tokens,
            sys_ms_tok: (ru1.sys_ms - ru0.sys_ms) / total_tokens,
            disp_tok: per(c0.dispatches, c1.dispatches),
            spin_tok: spin,
            park_tok: park,
            park_pct: if spin + park > 0.0 {
                park / (spin + park) * 100.0
            } else {
                0.0
            },
            inline_tok: per(c0.inline_dispatches, c1.inline_dispatches),
            dyield_tok: per(c0.dispatcher_yields, c1.dispatcher_yields),
            vcsw_tok: per(ru0.vcsw, ru1.vcsw),
            ivcsw_tok: per(ru0.ivcsw, ru1.ivcsw),
            rss_mb: ru1.maxrss_mb,
            contention,
            obs_disp: {
                let disp = per(c0.dispatches, c1.dispatches);
                if disp > 0.0 {
                    (spin + park) / disp
                } else {
                    0.0
                }
            },
            // Deliberately not a fold over the session hashes. The first
            // version XORed them together and then XORed the result with
            // `hashes[0]`, which at `sessions = 1` is `h ^ h == 0`: a control
            // that reported a constant for every input and would have passed
            // against any implementation at all. Every session computes the
            // same chain from the same shared weights, so the check is
            // equality, and a disagreement across sessions is as much a defect
            // as one across cells.
            hash: hashes[0],
            sessions_agree: hashes.iter().all(|&h| h == hashes[0]),
        }
    };

    // Interleaved: the whole matrix, `reps` times. Not each cell `reps` times in
    // place -- that orders the results by whatever the host was doing.
    let mut passes: Vec<Vec<Pass>> = vec![Vec::with_capacity(reps); cells.len()];
    for _ in 0..reps {
        for (index, cell) in cells.iter().enumerate() {
            passes[index].push(measure(*cell));
        }
    }

    // The window the workers actually obeyed, read back from the EP rather than
    // re-parsed here: a second parser is a second implementation that can drift
    // from the policy it claims to describe.
    println!(
        "blocktime_us={} (ONNX_GENAI_CPU_DECODE_BLOCKTIME_US; latched once per process)",
        decode_spmd::blocktime().as_micros()
    );
    if cfg!(not(unix)) {
        println!(
            "resource_metrics=unavailable (no getrusage on this target: cpu_ms, sys_ms, vcsw, ivcsw and rss_mb are NOT measured)"
        );
    }
    common::report_decode_width();

    println!(
        "{:>6} {:>7} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9} {:>9} {:>7} {:>8} {:>8} {:>7} {:>9} {:>9} {:>7} {:>8} {:>9}",
        "kind",
        "gap_us",
        "gap_act",
        "ms_tok",
        "p90",
        "cpu_ms",
        "sys_ms",
        "disp/tok",
        "spin/tok",
        "park/tok",
        "park_%",
        "obs/disp",
        "inline",
        "dy/tok",
        "vcsw/tok",
        "ivcsw/tok",
        "rss_mb",
        "spread_%",
        "foreign_%",
    );

    let mut decode_hashes: Vec<(String, u64)> = Vec::new();
    for (index, cell) in cells.iter().enumerate() {
        let (kind, gap) = (cell.kind, cell.gap_us);
        let rows = &passes[index];
        let pick = |f: fn(&Pass) -> f64| median(rows.iter().map(f).collect());
        let ms = pick(|p| p.ms_tok);
        let mut per_rep: Vec<f64> = rows.iter().map(|p| p.ms_tok).collect();
        per_rep.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let spread = if ms > 0.0 {
            (per_rep[per_rep.len() - 1] - per_rep[0]) / ms * 100.0
        } else {
            0.0
        };
        println!(
            "{:>6} {:>7.0} {:>8.1} {:>8.3} {:>8.3} {:>8.3} {:>8.3} {:>8.1} {:>9.1} {:>9.1} {:>7.1} {:>8.2} {:>8.2} {:>7.2} {:>9.1} {:>9.1} {:>7.1} {:>8.1} {:>9}",
            cell.label(),
            gap,
            pick(|p| p.gap_act_us),
            ms,
            pick(|p| p.p90),
            pick(|p| p.cpu_ms_tok),
            pick(|p| p.sys_ms_tok),
            pick(|p| p.disp_tok),
            pick(|p| p.spin_tok),
            pick(|p| p.park_tok),
            pick(|p| p.park_pct),
            pick(|p| p.obs_disp),
            pick(|p| p.inline_tok),
            pick(|p| p.dyield_tok),
            pick(|p| p.vcsw_tok),
            pick(|p| p.ivcsw_tok),
            pick(|p| p.rss_mb),
            spread,
            host_contention::foreign_column(&rows.iter().map(|p| p.contention).collect::<Vec<_>>(),),
        );

        if rows.iter().any(|p| p.contention.is_contended()) {
            let worst = rows
                .iter()
                .map(|p| p.contention.foreign_pct)
                .fold(0.0_f64, f64::max);
            println!(
                "  CONTENDED foreign_%={worst:.1} of one CPU on this process's confined core \
                 set -- a dispatch is a barrier, so this row's wall time is not comparable to a \
                 clean one (cpu_ms is)"
            );
        }

        if !cell.decode {
            // Control 1: the generator must not itself dispatch. If it does,
            // every null row is subtracting a cost that includes decode.
            let dispatched: f64 = rows.iter().map(|p| p.disp_tok).sum();
            if dispatched > 0.0 {
                println!(
                    "  NULL-DISPATCHED disp/tok={dispatched:.3} (null rows are not a control)"
                );
            }
        } else {
            if !rows.iter().all(|p| p.sessions_agree) {
                println!("  SESSIONS-DISAGREE concurrent sessions computed different outputs");
            }
            decode_hashes.push((format!("{} gap={gap:.0}", kind.label()), rows[0].hash));
        }
    }

    // Control 3: the gap changes when work happens, never what it computes.
    let mismatched: Vec<&(String, u64)> = decode_hashes
        .iter()
        .filter(|(_, h)| *h != decode_hashes[0].1)
        .collect();
    if mismatched.is_empty() {
        println!(
            "output_hash={:#018x} identical across {} decode cells",
            decode_hashes[0].1,
            decode_hashes.len()
        );
    } else {
        println!(
            "GAP-CHANGED-OUTPUT baseline={} {:#018x}; disagreeing: {}",
            decode_hashes[0].0,
            decode_hashes[0].1,
            mismatched
                .iter()
                .map(|(name, h)| format!("{name}={h:#018x}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
}

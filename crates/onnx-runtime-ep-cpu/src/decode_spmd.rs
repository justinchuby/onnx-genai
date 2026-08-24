//! Persistent SPMD decode pool: one hot worker set joined by a lightweight
//! reusable barrier, replacing the ~141 per-token Rayon fork-join regions.
//!
//! # Why
//!
//! Native M=1 int4 decode issues ~141 `MatMulNBits` projections per token, each
//! run today as a *separate* Rayon parallel region (`par_chunks_mut(..).for_each`
//! -- see [`crate::kernels::matmul_nbits::parallel_output_rows`]). Even with the
//! pool kept hot by [`crate::kernels::matmul_nbits::with_decode_pool_scope`],
//! every projection still pays Rayon's per-region machinery: task publication on
//! the crossbeam deque, work-stealing coordination, `crossbeam-epoch` memory
//! reclamation, and a join latch. Profiling attributes ~27% of the decode step
//! to this fork-join glue, and it is exactly the term that makes >32 cross-socket
//! threads regress.
//!
//! This module keeps a fixed set of worker threads hot-then-parked and drives
//! them with a hand-rolled **sense-reversing broadcast + counting barrier**: a
//! per-node sense counter bump publishes the op, workers observe their node's
//! sense advance, run their pre-assigned output-row shard, and decrement a
//! per-node completion counter; the dispatcher spins on those counters. Workers
//! wait with a KMP_BLOCKTIME-style bounded active spin then park on a futex
//! (`atomic-wait`), so a single `wake` per node releases the whole node in one
//! syscall and idle workers yield the CPU. No per-op allocation, no deque, no
//! epoch GC -- just a handful of atomics per projection. An unwind-only
//! completion guard still decrements the counter if a worker panics, poisons the
//! pool, and makes the dispatcher report an actionable panic instead of hanging.
//!
//! # Two-level, NUMA-aware (mirrors `numa-split`)
//!
//! To use both sockets' memory bandwidth without a toxic flat cross-socket
//! barrier, workers are split into per-node groups (16+16 on a 2-node host),
//! each pinned to its node and reading a node-local first-touched weight shard,
//! exactly like [`crate::decode_numa`]. Row-sharding a GEMV is exactly
//! associative -- each output row is an independent dot product over the whole K
//! dimension -- so concatenating the per-worker row slices reproduces the flat
//! result bit-for-bit, with no cross-row/-node reduction. The only cross-socket
//! traffic per op is the dispatcher reading each node's own completion counter
//! (one line per node), not an N-way shared barrier.
//!
//! # Generality (rule 2)
//!
//! Topology is queried at runtime, never hardcoded. On a single-node host, a
//! non-NUMA machine, or a platform without CPU pinning, the pool degrades to one
//! unpinned worker group -- it still replaces the per-op Rayon barrier with the
//! lightweight one, and stays correct.
//!
//! # Deterministic by default, with opt-in adaptive calibration (rule 5)
//!
//! The pool is **on by default** -- the deterministic, predictable choice.
//! `ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL` selects the policy:
//!
//! * **unset (the default) or `=1`**: the persistent SPMD pool is always used.
//!   No host probing, no calibration, fully deterministic. The same prompt at
//!   temperature 0 always follows the same floating-point reduction order.
//! * **`=auto`**: opt-in load-adaptive calibration. A runtime heuristic times
//!   the pool and flat paths on the live workload and keeps the faster one
//!   (see [`Calibrator`]). Under co-tenant load the flat path usually wins;
//!   on a quiet host the pool wins. The tradeoff is that the selected path
//!   depends on transient system state, so results are not reproducible across
//!   runs on differently-loaded machines.
//! * **`=0`**: explicit opt-out; the decode path stays on the flat legacy pool.
//!
//! **The path is frozen once committed.** Whether the pool was selected by the
//! default, by `=1`, or by the adaptive calibrator, the path stays fixed for
//! the lifetime of the generation. The flat and pool paths use different
//! floating-point reduction orders (single-threaded vs partitioned parallel),
//! so switching mid-generation changes logits and can produce different tokens
//! under greedy decode.
//!
//! The worker count is
//! [`crate::kernels::matmul_nbits::configured_persistent_decode_threads`] (one
//! worker per *allowed physical core*, falling back to half the logical CPUs
//! only when the core topology is undiscoverable); a `THREADS=0` opt-out leaves
//! the decode path unchanged.
//!
//! # Precedence when on (default or `=1`) vs the affinity control
//!
//! When the pool is on (the default, or `=1`), the decode strategy precedence is,
//! highest first:
//!
//! 1. **`ONNX_GENAI_CPU_DECODE_AFFINITY=numa-split`** -- the explicit multi-node
//!    split wins when its two-level layout can be built (the mutually-exclusive
//!    selection vs the persistent pool is reported once).
//! 2. **Persistent SPMD pool** (default or `=1`) -- its own per-node pinning applies.
//! 3. **Flat Rayon + auto-`compact`** legacy path -- reached by `=0` (Off). Under
//!    `=auto` (adaptive), an explicit `numa-split` affinity likewise takes
//!    precedence over calibration (the user picked a specific strategy), so the
//!    adaptive calibrator measures the persistent SPMD pool against the flat path
//!    only.

#[cfg(test)]
use std::cell::Cell;
use std::cell::UnsafeCell;
use std::sync::atomic::{
    AtomicBool, AtomicI64, AtomicU32, AtomicU64, AtomicUsize, Ordering, fence,
};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::decode_affinity::{NodeShard, NumaTopology};
use crate::kernels::matmul_nbits::output_chunk_len_for;

/// Environment switch selecting the persistent SPMD decode pool policy:
/// **unset (the default) or `=1`** uses the persistent SPMD pool deterministically
/// (no host probing); `=auto` opts in to load-adaptive calibration (see
/// [`Calibrator`]); `=0` forces the flat legacy path. The pool beats the flat
/// path on a quiet or dedicated host; under heavy co-tenant load the flat path
/// degrades more gracefully, which is why the adaptive mode exists -- but the
/// default prioritises predictability over adaptation.
/// See `.squad/decisions.md` (Voight 2026-07-24; Chu 2026-07-27 opt-in adaptive).
pub const PERSISTENT_POOL_ENV: &str = "ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL";
/// Selects how the persistent decode pool assigns output-column work inside an
/// op. Unset / `fixed` keeps the existing deterministic one-shard-per-worker
/// SPMD split; `steal` decomposes the op into coarser tiles and lets resident
/// workers claim tiles dynamically, so a delayed worker does not strand a whole
/// static shard behind the per-op barrier.
pub const DECODE_SCHEDULE_ENV: &str = "ONNX_GENAI_CPU_DECODE_SCHEDULE";

/// Bounded active-spin window before a worker parks, mirroring the
/// **LLVM/Intel OpenMP runtime `KMP_BLOCKTIME`** design: after a fork/join the
/// worker busy-waits ("blocktime") for the next barrier release, and only sleeps
/// on a futex once genuinely idle for longer than the window. This is the gold
/// standard for exactly this OMP-style fine-grained fork/join HPC workload.
///
/// The window is **load-bearing, not a bad habit**: decode fires ~400 fork/join
/// barriers per token, microseconds apart, so a worker must catch the next
/// barrier while spinning. Parking inside the active path would pay a futex wake
/// (~1-5 us) on every barrier and tank throughput. The window is sized to span
/// the inter-op / inter-token gaps of tight decode (so workers effectively never
/// park mid-generation) yet expire quickly once a generation ends, so idle CPU
/// returns to ~0 between requests -- the spin is bounded to genuinely-active
/// decode, not "unconditional". Tunable via [`decode_blocktime`].
///
/// Default 500 us: comfortably longer than the ~microsecond inter-op gap and the
/// serial dispatcher glue between barriers, and shorter than any human-scale idle
/// gap between requests. Analogous to (though far shorter than) OpenMP's 200 ms
/// default, which targets coarse parallel regions rather than per-token decode.
const DEFAULT_BLOCKTIME: Duration = Duration::from_micros(500);

/// Environment variable naming the worker active-spin window, in microseconds.
const DECODE_BLOCKTIME_ENV: &str = "ONNX_GENAI_CPU_DECODE_BLOCKTIME_US";

/// Environment variable enabling the per-worker wake/work timing reported by
/// [`SpmdDecodePools::worker_profiles`]. Off by default; see
/// [`parse_worker_profile`].
pub const WORKER_PROFILE_ENV: &str = "ONNX_GENAI_CPU_DECODE_WORKER_PROFILE";

/// Pure `spin_loop` iterations at the start of the active window before the
/// worker begins yielding the core to co-tenants between clock checks. A
/// crossbeam-`Backoff`-style ramp: hammer the sense line first to catch the
/// common immediately-ready case with the lowest latency, then relax to
/// `yield_now` so a busy host can schedule other work while we finish out the
/// blocktime window. Sized (~4096 spins, a few microseconds) to cover the
/// typical inter-op dispatcher gap so back-to-back decode barriers are caught by
/// pure spinning and only genuinely idle gaps ramp into yielding then parking.
const SPIN_LOOP_BUDGET: u32 = 1 << 12;

/// How long [`SpmdDecodePools::build_with_schedule`]'s readiness barrier waits
/// for every spawned worker to announce itself before declaring the pool
/// unbuildable.
///
/// Deliberately far beyond any real startup: workers announce within
/// microseconds, so this can only be reached when the condition has become
/// unsatisfiable. It exists so that failure mode is a loud panic instead of an
/// unbounded spin that holds a machine at full occupancy indefinitely.
const POOL_READY_TIMEOUT: Duration = Duration::from_secs(120);

// Both knobs below are scoped to the thread that builds the pool, not global. A
// global knob is read by *every* concurrently-building pool in the same test
// binary, so injecting a fault reaches pools belonging to unrelated tests --
// which is exactly how the first version of this fault injection failed
// `an_idle_gap_far_longer_than_the_blocktime_parks_the_workers`. A `Mutex` held
// by the injecting tests does not fix that, because the tests being corrupted
// never take it. Thread scoping needs no lock and cannot be forgotten by a
// future test, since the builder thread is the test's own thread.
#[cfg(test)]
thread_local! {
    /// Test override for [`POOL_READY_TIMEOUT`], in milliseconds; `0` means
    /// "use the real one". A liveness backstop is only testable if the test
    /// does not have to wait out the production deadline to observe it.
    static POOL_READY_TIMEOUT_MS: Cell<u64> = const { Cell::new(0) };

    /// Test fault injection: the global worker index that must die before
    /// announcing readiness, or `usize::MAX` for none. Reproduces the one
    /// failure the barrier cannot distinguish from slowness -- a worker that
    /// will never arrive.
    static FAIL_WORKER_BEFORE_READY: Cell<usize> = const { Cell::new(usize::MAX) };

    /// Test mutation: skip the affinity syscall entirely and report the pin as
    /// applied anyway.
    ///
    /// This reproduces, in-tree and permanently, the exact defect that makes a
    /// planner-derived placement label untrustworthy -- pinning that answers
    /// `Ok(())` and changes nothing. It is what a stubbed, `seccomp`-filtered
    /// or emulated-away `sched_setaffinity` looks like from inside the pool,
    /// and any placement check that stays green under it is reporting the plan
    /// rather than the machine.
    ///
    /// Thread-scoped like its neighbours, and latched on the builder thread, so
    /// a pool built by any other test is untouched. `cfg(test)` throughout: the
    /// production build has no branch here at all.
    static STUB_PIN_SYSCALL: Cell<bool> = const { Cell::new(false) };

    /// Test injection of the *other* case: every worker announces, but late.
    /// Without this the healthy-pool test is vacuous, because real workers
    /// announce within the spin budget and the deadline is never consulted at
    /// all -- so it would pass with a 1ms timeout just as happily as a 120s
    /// one. A delay long enough to push the barrier onto its yield/clock path
    /// is what makes "slow is not the same as broken" an actual assertion.
    static DELAY_WORKER_BEFORE_READY_MS: Cell<u64> = const { Cell::new(0) };

    /// Test injection of a *contended* yield, in microseconds; `0` means "yield
    /// normally". The readiness backstop's stride defect is invisible when
    /// yields are cheap: an uncontended `yield_now` costs ~1.2us, so a stride of
    /// 64 moves the deadline by ~78us and no assertion against a millisecond
    /// deadline can see it. The production measurement that found it recorded
    /// ~7ms per yield on a contended pair of CPUs -- three orders of magnitude
    /// larger -- which is what turns the stride into a 3x deadline overrun.
    /// Manufacturing real contention in a unit test is exactly the kind of
    /// load-dependent arrangement that flakes; injecting the yield *cost* makes
    /// the same regime deterministic.
    static SLOW_YIELD_US: Cell<u64> = const { Cell::new(0) };

    /// Test override for [`WORKER_PROFILE_ENV`]: `Some(v)` forces the per-worker
    /// timing gate to `v` for pools built on this thread. Thread-scoped for the
    /// same reason as the knobs above, and additionally because the alternative
    /// -- `set_var` around a build -- is a data race against every other test
    /// thread's `getenv` and would need a lock this module does not have.
    static FORCE_WORKER_PROFILE: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Read on the builder thread only: both the barrier and the pre-spawn fault
/// latch run there, so the thread-local scoping above is what makes this
/// answer the injecting test's question rather than some other test's.
fn pool_ready_timeout() -> Duration {
    #[cfg(test)]
    {
        let ms = POOL_READY_TIMEOUT_MS.with(Cell::get);
        if ms > 0 {
            return Duration::from_millis(ms);
        }
    }
    POOL_READY_TIMEOUT
}

/// KMP_BLOCKTIME analog: the active-spin window a worker holds a core before
/// parking on the futex, read once. `ONNX_GENAI_CPU_DECODE_BLOCKTIME_US` sets it
/// in microseconds; `0` parks as soon as the sense line is not already advanced
/// (maximally polite, higher wake latency). Unset uses [`DEFAULT_BLOCKTIME`].
fn decode_blocktime() -> Duration {
    static V: OnceLock<Duration> = OnceLock::new();
    *V.get_or_init(|| parse_decode_blocktime(std::env::var(DECODE_BLOCKTIME_ENV).ok().as_deref()))
}

/// Parse a [`DECODE_BLOCKTIME_ENV`] value, falling back to
/// [`DEFAULT_BLOCKTIME`] for anything unset or unparseable.
///
/// Split out from [`decode_blocktime`] so the policy is testable without the
/// environment. That is not a stylistic preference: `decode_blocktime` latches
/// into a `OnceLock` on first `worker_wait`, so a test that set the variable and
/// called it twice would compare the first value against itself and pass while
/// measuring nothing -- the same latched-control defect as #1736. Testing the
/// pure function cannot silently degrade that way, and `cargo test` runs a
/// binary's tests on parallel threads where `set_var` racing another thread's
/// `getenv` is a data race Rust 2024 forbids outright.
fn parse_decode_blocktime(raw: Option<&str>) -> Duration {
    raw.and_then(|raw| raw.trim().parse::<u64>().ok())
        .map_or(DEFAULT_BLOCKTIME, Duration::from_micros)
}

/// Env gate for the per-worker wake/work timing in [`SpmdWorkerProfile`].
///
/// Read **once per pool build**, into [`SharedState::profile_workers`] --
/// deliberately not into a `OnceLock`. A latched gate would make the obvious
/// A/B (build a profiled pool, build an unprofiled one, compare) compare one
/// configuration against itself and pass without measuring anything: that is
/// #1736 exactly, and it is not a defect worth reintroducing in the very
/// mechanism built to catch unmeasured claims. Production builds the pool once,
/// so a per-build read costs one `getenv` for the process.
fn parse_worker_profile(raw: Option<&str>) -> bool {
    matches!(
        raw.map(str::trim),
        Some("1") | Some("on") | Some("true") | Some("TRUE") | Some("On") | Some("True")
    )
}

fn worker_profile_enabled() -> bool {
    #[cfg(test)]
    if let Some(forced) = FORCE_WORKER_PROFILE.with(Cell::get) {
        return forced;
    }
    parse_worker_profile(std::env::var(WORKER_PROFILE_ENV).ok().as_deref())
}

/// Process-wide monotonic origin for [`SpmdWorkerProfile`] timestamps. Only ever
/// read while profiling is on, and only to produce differences, so the origin
/// itself is arbitrary; it exists because `Instant` is not storable in an
/// `AtomicU64` and the publish timestamp has to cross threads.
fn profile_epoch() -> Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

/// Nanoseconds since [`profile_epoch`], saturating at `u64::MAX` (585 years).
fn profile_now_ns() -> u64 {
    u64::try_from(profile_epoch().elapsed().as_nanos()).unwrap_or(u64::MAX)
}

/// Spin iterations the (single, never-idle) dispatcher busy-waits on the
/// completion counters before yielding. The dispatcher runs the barrier inline
/// and needs the workers' results the instant they land, so it spins before it
/// yields; the yields after the budget let a descheduled straggler worker get a
/// core under oversubscription, and are themselves bounded by
/// [`dispatcher_yields_before_park`].
const DISPATCHER_SPIN_BEFORE_YIELD: u32 = 1 << 12;

/// Default number of `sched_yield` calls the dispatcher makes, after its spin
/// budget, before parking on the completion futex.
///
/// This covers the case the yield backstop actually exists for -- a straggler
/// that is merely descheduled and needs a core to finish a shard that is nearly
/// done -- while bounding the cost when the shard is genuinely long. At roughly
/// a microsecond per yield on an idle core this is a window of order 100 us,
/// two orders of magnitude below the millisecond-scale dispatches where the
/// unbounded loop was burning a whole core, and comfortably above the wake
/// latency a park pays to replace it.
const DEFAULT_DISPATCHER_YIELDS_BEFORE_PARK: u32 = 128;

/// Env override for [`DEFAULT_DISPATCHER_YIELDS_BEFORE_PARK`]. `0` parks as soon
/// as the spin budget is exhausted; a very large value restores the historical
/// never-park behaviour. Latched once per process, like the blocktime.
const DISPATCHER_YIELDS_ENV: &str = "ONNX_GENAI_CPU_DECODE_DISPATCHER_YIELDS";

fn dispatcher_yields_before_park() -> u32 {
    static V: OnceLock<u32> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var(DISPATCHER_YIELDS_ENV)
            .ok()
            .and_then(|raw| raw.trim().parse::<u32>().ok())
            .unwrap_or(DEFAULT_DISPATCHER_YIELDS_BEFORE_PARK)
    })
}

/// Default number of dynamic tiles per resident worker in work-stealing mode.
/// One tile per worker preserves the coarse MLAS QNBit shard size that made
/// fixed-SPMD fast, while Deckard's pool can still steal a not-yet-claimed tile
/// from a delayed worker. Finer 2x/3x tiling improved theoretical steal
/// opportunities but split Qwen3 projection shards too narrowly and regressed
/// measured throughput.
const DEFAULT_STEAL_TILES_PER_WORKER: usize = 1;
/// Minimum output columns in one dynamic tile. Keeps MLAS/KAI/hand GEMV calls
/// coarse enough that the extra atomic `fetch_add` scheduling is amortized.
const MIN_STEAL_OUTPUTS_PER_TASK: usize = 32;

/// Cache-line pad so per-node completion counters and per-worker park flags do
/// not false-share (which would reintroduce cross-socket coherency traffic).
#[repr(align(128))]
struct Padded<T>(T);

/// A type-erased decode job: run the shard for the given global worker index.
/// The data pointer is only dereferenced between [`SharedState::publish`] and the
/// matching dispatcher wait, so the borrowed closure always outlives its use.
#[derive(Clone, Copy)]
struct Job {
    data: *const (),
    call: unsafe fn(*const (), usize),
}

/// A node's two dispatcher-to-worker words, deliberately on one cache line.
///
/// `wake` is the word workers park on, so it has to advance for both a new op
/// and for shutdown -- shutdown must break a parked worker out, and bumping the
/// parked-on word is the only way to do that without a lost wakeup. That makes
/// `wake` ambiguous on its own: an advance means "wake up and look", not "there
/// is work", and *both* ways of resolving that ambiguity from a single word are
/// unsafe. Treat a shutdown bump as an op and the worker re-runs the previous
/// op's `Job`, whose closure lives on a stack frame that has already returned --
/// a use-after-free. Decide instead from the shutdown flag, and a dispatch that
/// lands just before a concurrent shutdown is abandoned unacknowledged, which
/// strands the dispatcher waiting on it forever.
///
/// So the two questions get two words: `wake` answers "should I wake up", `ops`
/// answers "is there an op I have not run yet". Only `ops` gates the `Job` read,
/// and only `publish` bumps it.
///
/// They share a `Padded` block rather than taking a line each because the
/// dispatcher writes both back to back and each woken worker reads both back to
/// back, so splitting them would add a second coherency miss per worker per op
/// -- at roughly 400 barriers per token that is a cost worth not paying for a
/// word that is only ever read beside its neighbour.
#[derive(Default)]
struct NodeSense {
    wake: AtomicU32,
    ops: AtomicU32,
    /// Nanoseconds since [`profile_epoch`] at which the current op was
    /// published, written by the dispatcher only while
    /// [`SharedState::profile_workers`] is set.
    ///
    /// Stored `Relaxed` *before* the `Release` bump of `ops`, so a worker whose
    /// `Acquire` load of `ops` observes the new op also observes this value: the
    /// release/acquire edge that already carries the job pointer carries this
    /// too, and it needs no ordering of its own.
    ///
    /// Lives on the sense line rather than on its own because it is written by
    /// the same thread, in the same instruction stream, as the two words beside
    /// it -- and, when profiling is off, never written at all, so it costs a
    /// disabled pool nothing.
    publish_ns: AtomicU64,
}

/// Observable scheduling behaviour of the persistent SPMD decode pool.
///
/// Every field is monotonic since process start and is read as a *delta* across
/// a measured phase, the way [`crate::task_runtime::testing::counters`] already
/// is. They exist so a harness can assert **what the scheduler did** instead of
/// inferring it from a timing difference: on a shared runner a park/wake
/// regression and a noisy neighbour produce the same slowdown, and only one of
/// them changes `parks`.
///
/// This closes a gap that made the blocktime knob untunable. Whether a worker
/// spins through an inter-token gap or parks and pays a futex wake was
/// observable only as wall time and process-wide `sys` time, both of which are
/// contaminated by anything else on the host and by the harness's own gap
/// generator. `spin_hits` and `parks` are per-worker, per-op, and immune to
/// both.
///
/// Not collected on the `--features mlas` work-stealing schedule, which is a
/// separate executor with its own wait policy and is never the production path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpmdCounters {
    /// Ops published to the worker set through the barrier.
    pub dispatches: u64,
    /// Dispatches that found the single job slot already claimed and ran every
    /// shard inline on the calling thread instead. A direct nested-dispatch and
    /// concurrent-session signal: the pool serves one dispatcher at a time, so
    /// the surplus degrades to serial rather than racing.
    pub inline_dispatches: u64,
    /// Times a worker's blocktime window caught the next op before it parked.
    ///
    /// Includes the case where the window expired but the op landed before the
    /// worker actually slept: no wake was paid, which is the distinction this
    /// pair of counters exists to draw.
    pub spin_hits: u64,
    /// Times a worker had to be woken from a futex park to observe an op.
    ///
    /// Counted when the observation completes rather than when the worker goes
    /// to sleep, so `spin_hits + parks` is **exactly** the number of
    /// op-observations: `dispatches * spawned_workers` at any point where no
    /// dispatch is in flight. That identity is what makes these assertable
    /// without timing. It holds while the pool is live; teardown bumps the same
    /// sense line to release the workers, and that bump is indistinguishable
    /// from an op here, so snapshots are taken before
    /// [`SpmdDecodePools::shutdown`].
    pub parks: u64,
    /// Times a parked worker woke without its node's sense having advanced and
    /// went back to sleep.
    ///
    /// Unlike `spin_hits` and `parks`, an increment here is not followed by a
    /// barrier acknowledgement, so it has no release edge to make it visible to
    /// a reader on another thread at a defined point. Treat it as eventually
    /// consistent: fine for a threshold or a rate, not for an exact assertion
    /// taken immediately after the wake.
    pub spurious_wakes: u64,
    /// Barriers where the dispatcher exhausted its spin budget waiting for the
    /// workers and yielded the core.
    ///
    /// **Strongly width- and regime-dependent; not an oversubscription signal.**
    /// The dispatcher publishes the op, computes its own shard when it has one,
    /// and only then spins on the completion counters, yielding once after
    /// `DISPATCHER_SPIN_BEFORE_YIELD` (~10 us). So a yield means the *last*
    /// worker lagged the dispatcher's own arrival at the barrier by more than
    /// that -- a statement about spread across workers, not about how long the
    /// op takes. Measured on an idle, un-oversubscribed host at zero gap:
    /// 0.004 yields per dispatch at width 4, ~0.9 at width 16, with the
    /// intervening widths non-monotonic. Read it as a straggler-spread
    /// indicator, and only ever against a fixed width.
    ///
    /// (Two earlier versions of this doc were wrong in opposite directions --
    /// first "only reachable when a worker is descheduled", then "any op longer
    /// than the spin budget yields, so it saturates at 1.0/dispatch". Both were
    /// reasoned from the code; the first was falsified by measuring at width 16
    /// and the second by measuring at width 4. Neither had been measured across
    /// the axis that actually moves it.)
    pub dispatcher_yields: u64,
}

/// One worker's counters, on its own cache line.
///
/// Per-worker rather than shared, and that is not micro-optimisation: these are
/// bumped once per worker per op, at roughly 400 barriers per token, so a single
/// shared line would take ~6400 contended RMWs per token at width 16. The module
/// already has a measurement of what that costs -- sharing a line with
/// [`SharedState::dispatching`] cost ~60% of dispatch latency (see its doc). An
/// exclusively-owned line stays in the owning core's L1 and the RMW is a few
/// nanoseconds.
#[derive(Default)]
struct WorkerCounters {
    spin_hits: AtomicU64,
    parks: AtomicU64,
    spurious_wakes: AtomicU64,
    /// Ops this worker retired *last* within its node -- i.e. the shard the
    /// dispatcher was still waiting on when every other shard was already done.
    ///
    /// Free: the barrier already computes it (the `fetch_sub` that returns 1 is
    /// by definition the last acknowledgement), so this adds one uncontended
    /// RMW on a line the worker owns, and only on the op it was last for. It is
    /// therefore on unconditionally, unlike the ns timings beside it.
    last_arrivals: AtomicU64,
    /// Ops for which `wake_ns`/`work_ns` below were both accumulated. Not equal
    /// to `spin_hits + parks`: those count *observations*, including the wake a
    /// worker pays for a shutdown bump, whereas an op is only timed here if it
    /// was actually run.
    timed_ops: AtomicU64,
    /// Summed nanoseconds from the dispatcher publishing an op to this worker
    /// observing it. The wake-latency half of a straggler.
    wake_ns: AtomicU64,
    /// Summed nanoseconds this worker spent inside its shard closure. The
    /// work-imbalance half of a straggler.
    work_ns: AtomicU64,
}

/// One worker's share of the barrier, for attributing a slow dispatch to a
/// *specific* worker rather than to the pool as a whole.
///
/// [`SpmdCounters`] sums across workers, which cannot distinguish "one worker is
/// consistently late" from "all workers are uniformly slower" -- and those have
/// different causes and different owners (a late single worker points at
/// placement, wakeup or a co-tenant on one CPU; uniform slowness points at the
/// kernel or at memory). No aggregate counter can separate them, which is why
/// this exists.
///
/// `last_arrivals` is always populated. `wake_ns`/`work_ns`/`timed_ops` are zero
/// unless the pool was built with [`WORKER_PROFILE_ENV`] set, because they cost
/// two clock reads per worker per op (~4% at width 12 on a 400-barrier token)
/// and must not be paid by production.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpmdWorkerProfile {
    /// Global worker index. Spawned workers only; the dispatcher's own shard
    /// never waits on the barrier and has no entry.
    pub worker: usize,
    /// The CPU this worker is pinned to, or `None` if unpinned or the pin
    /// failed. Same realized-not-requested semantics as
    /// [`SpmdDecodePools::worker_cpus`].
    pub cpu: Option<usize>,
    /// See [`SpmdCounters::spin_hits`].
    pub spin_hits: u64,
    /// See [`SpmdCounters::parks`].
    pub parks: u64,
    /// See [`SpmdCounters::spurious_wakes`].
    pub spurious_wakes: u64,
    /// Ops this worker was the last in its node to retire. Summed over a node's
    /// workers this equals that node's dispatch count exactly, which is what
    /// makes an *uneven* distribution meaningful: with `w` workers, chance alone
    /// gives each `1/w` of them.
    pub last_arrivals: u64,
    /// Ops contributing to `wake_ns` and `work_ns`. Zero when profiling is off.
    pub timed_ops: u64,
    /// Summed publish-to-observe latency, nanoseconds. Zero when profiling is
    /// off.
    pub wake_ns: u64,
    /// Summed in-shard time, nanoseconds. Zero when profiling is off.
    pub work_ns: u64,
}

/// Dispatcher-side counters. Bumped by whichever thread owns the barrier, at
/// most once per op, and never touched from a worker's wait loop.
#[derive(Default)]
struct DispatchCounters {
    dispatches: AtomicU64,
    inline_dispatches: AtomicU64,
    dispatcher_yields: AtomicU64,
}

/// State shared between the dispatcher (the engine thread running the forward)
/// and the persistent worker threads.
struct SharedState {
    /// Per-node sense counter, bumped once per dispatched op; a node's workers
    /// wait for *their* node's counter to advance. This is a **sense-reversing
    /// barrier** generalised from a 1-bit phase to a monotonic `u32`: each op has
    /// a strictly-increasing sense value, so barrier reuse across the ~400
    /// barriers/token can never race (no ABA -- wrap needs 2^32 barriers while a
    /// worker sleeps through exactly that many, which is impossible). Splitting it
    /// **per node** is the key locality fix: every worker spins on / futex-waits
    /// on its own node's line, so there is no single shared cache line ping-ponging
    /// across the UPI link between sockets.
    node_sense: Vec<Padded<NodeSense>>,
    /// The current op, published before the sense counters bump and read after the
    /// bump is observed (release/acquire pairing on `node_sense`).
    job: UnsafeCell<Option<Job>>,
    /// Outstanding worker acknowledgements for the current op, one counter per
    /// node so the dispatcher only reads each node's own (mostly node-local)
    /// line instead of an N-way shared barrier.
    node_pending: Vec<Padded<AtomicUsize>>,
    /// The node each global worker index belongs to (drives which pending
    /// counter it decrements and which sense line it waits on).
    worker_node: Vec<usize>,
    /// Count of workers that have entered their loop and are ready to receive
    /// ops. `build` blocks until this reaches `total_workers` so no dispatch can
    /// race a not-yet-started worker (which would miss the op and hang the
    /// barrier).
    ready: AtomicUsize,
    /// Nonzero after a worker panics while running an op (`worker_index + 1`).
    /// A poisoned pool rejects this and every later dispatch instead of hanging
    /// forever waiting for a worker that has unwound.
    poisoned_worker: AtomicUsize,
    /// Set while one thread owns the publish/wait protocol above.
    ///
    /// `job` is a **single slot** and the `node_pending` counters are a single
    /// barrier, so the whole `publish` -> `wait` sequence is one critical
    /// section over the entire worker set. Two threads running it at once would
    /// overwrite each other's job pointer and both sets of workers would run
    /// whichever closure landed last -- producing *silently wrong tensors*
    /// rather than a crash or a hang, since each shard writes a disjoint output
    /// region and the barrier still balances.
    ///
    /// [`SpmdDecodePools::dispatch`] claims this with a compare-exchange and
    /// runs the shards inline when the claim fails, so a second dispatcher
    /// degrades to serial instead of corrupting the first. This also makes a
    /// re-entrant dispatch (a shard closure that itself dispatches) run inline
    /// rather than deadlock against the barrier it is already inside.
    ///
    /// Padded onto its own line, which is not optional. It is written twice per
    /// op by the dispatcher while every worker polls `shutdown` in its wait
    /// loop, so sharing a line with those fields makes each claim invalidate
    /// all 16 workers' copies. Empty 16-worker dispatch, best of 7 x 20k,
    /// three alternating rounds:
    ///
    /// | variant | ns/dispatch |
    /// |---|---|
    /// | ungated (pre-fix) | 908 / 903 / 907 |
    /// | gated, padded | 898 / 881 / 898 |
    /// | gated, unpadded | 1452 / 1497 / 1432 |
    ///
    /// Padded, the gate costs nothing measurable. Unpadded it costs ~60%, which
    /// at ~400 barriers per token is ~230 us/token of pure coherency traffic.
    dispatching: Padded<AtomicBool>,
    shutdown: AtomicBool,
    /// Count of workers that have left [`worker_loop`] for good.
    ///
    /// The mirror of `ready`: `ready` makes "the workers have started" an
    /// observable fact that `build` can block on, and this makes "the workers
    /// have stopped" one that a teardown test can assert on. Without it the
    /// only evidence of a completed teardown is the absence of thread names in
    /// `/proc/self/task`, and a joined thread's entry lingers there for a short
    /// window after `join` returns -- the futex that unblocks `join` is
    /// signalled at `mm_release`, before the task is unhashed -- so an
    /// assertion on that count is a race under load rather than an invariant.
    /// This counter is ordered by the join itself and cannot lag it.
    ///
    /// One `fetch_add` per worker per process, so the cost is not on any path
    /// that runs more than once.
    workers_exited: AtomicUsize,
    /// Whether the per-worker ns timings are collected. Latched per *pool*, not
    /// per process (see [`parse_worker_profile`]), and immutable after build so
    /// every read is a plain load of a shared read-only word.
    profile_workers: bool,
    /// One padded counter block per *spawned* worker, indexed by global worker
    /// index. The dispatcher's own shard (global index `total_threads`) never
    /// enters [`SharedState::worker_wait`], so it has no entry here.
    worker_counters: Vec<Padded<WorkerCounters>>,
    dispatch_counters: Padded<DispatchCounters>,
    /// Bumped by the worker that retires the last outstanding shard of an op, so
    /// a dispatcher that parked in [`SharedState::wait`] can be woken.
    ///
    /// Separate from `node_sense`, which runs the other direction (dispatcher to
    /// workers). Only the *last* worker of an op touches this, so it is one
    /// extra RMW per dispatch, not per worker. Each node's last worker does also
    /// scan every node's pending counter on its way here, so the full per-op cost
    /// is `node_count` scans plus the one RMW -- immaterial at the one or two
    /// nodes a decode pool actually builds.
    completion_sense: Padded<AtomicU32>,
    /// Set while the dispatcher is parked, or committed to parking, on
    /// `completion_sense`. Lets the completing worker skip the `wake` syscall
    /// entirely in the common case where the dispatcher is still spinning.
    dispatcher_parked: Padded<AtomicBool>,
}

// SAFETY: `job` is a raw pointer guarded by the publish/observe protocol on
// `node_sense`; it is only read by workers while the dispatcher blocks in
// `dispatch`, so the pointee outlives every access. All other fields are atomics.
unsafe impl Sync for SharedState {}
unsafe impl Send for SharedState {}

impl SharedState {
    /// Publish `job` for `node_pending[node] = counts[node]` workers, then wake
    /// each node's sleeping workers with a single futex `wake` per node. Must be
    /// paired with [`SharedState::wait`].
    fn publish(&self, job: Job, counts: &[usize]) {
        // Publish the job pointer, then the per-node counts, before the sense
        // bumps make them visible to workers.
        unsafe {
            *self.job.get() = Some(job);
        }
        for (counter, &count) in self.node_pending.iter().zip(counts) {
            counter.0.store(count, Ordering::Release);
        }
        // Advance each node's sense (Release) so the job + count writes above are
        // visible to any worker that observes the new sense (Acquire), then wake
        // that node's parked workers. `wake` is a single futex_wake(i32::MAX) per
        // node -- one syscall releases the whole node, not an O(workers) unpark
        // fan-out. A worker that raced into parking re-checks the sense under the
        // futex guard, so this ordering (bump-then-wake) loses no wakeup: if the
        // worker armed `wait(last_seen)` before this bump it wakes here; if it
        // armed after, `wait` sees the advanced sense and returns without sleeping.
        for (node, sense) in self.node_sense.iter().enumerate() {
            if counts.get(node).copied().unwrap_or(0) == 0 {
                continue;
            }
            // Ordered before the `ops` bump for the same reason the counts above
            // are: the Release/Acquire pair on `ops` is what publishes it.
            if self.profile_workers {
                sense
                    .0
                    .publish_ns
                    .store(profile_now_ns(), Ordering::Relaxed);
            }
            // Ordered before the wake bump so a worker that observes the wake
            // (Acquire) also observes this.
            sense.0.ops.fetch_add(1, Ordering::Release);
            sense.0.wake.fetch_add(1, Ordering::Release);
            atomic_wait::wake_all(&sense.0.wake);
        }
    }

    /// Spin-wait until every node's workers have finished the published op.
    ///
    /// There is exactly one dispatcher at a time -- enforced by
    /// [`SharedState::dispatching`], which callers claim through
    /// [`DispatchClaim::try_claim`] before publishing. It wants the results the
    /// instant they land, so it spins first; but it must not spin *forever*.
    ///
    /// # Why this parks
    ///
    /// This loop used to call `thread::yield_now()` on **every** iteration once
    /// past `DISPATCHER_SPIN_BEFORE_YIELD`, with no upper bound -- so the
    /// dispatcher hammered `sched_yield` for the entire remaining duration of
    /// every dispatch it did not win the race on. That is a syscall storm
    /// proportional to the shard length, not to the wake latency it was there to
    /// hide. Measured on llama int4 `accuracy_level=0` decode, zero inter-token
    /// gap, one worker per physical core, quiet host:
    ///
    /// | width | wall ms/token | **sys** ms/token | dispatches/token |
    /// |---|---|---|---|
    /// | 2 | 35.7 | **16.0** | 5 |
    /// | 3 | 23.9 | **12.0** | 5 |
    /// | 4 | 18.0 | **10.2** | 5 |
    ///
    /// At width 2 that is 3.2 ms of kernel time per dispatch against a ~7 ms
    /// dispatch -- an entire core burned to produce nothing, with roughly 45% of
    /// it inside `sched_yield` itself and the rest spent spinning between the
    /// calls. The `sys` column is the part that is unambiguously attributable, so
    /// it is the one quoted.
    ///
    /// Note this is *not* the same loop as [`SharedState::worker_wait`], where
    /// substituting spinning for yielding was measured to be a wash (kernel time
    /// down, user time up by the same amount, total CPU flat). That result does
    /// not transfer: a worker that stops yielding still burns its core in user
    /// mode, whereas a dispatcher that *parks* burns nothing at all. Spinning
    /// harder here would only relabel the waste from `sys` to `user`.
    ///
    /// So the escalation is now bounded: spin, then a fixed number of yields to
    /// cover the short waits that a park would only slow down, then park on
    /// `completion_sense` until the last worker retires the op. The yield budget
    /// is [`dispatcher_yields_before_park`].
    fn wait(&self) {
        self.wait_with_yield_budget(dispatcher_yields_before_park());
    }

    fn wait_with_yield_budget(&self, yield_budget: u32) {
        let mut spins = 0u32;
        let mut yielded = false;
        let mut yields = 0u32;
        loop {
            if self.all_workers_done() {
                return;
            }
            std::hint::spin_loop();
            spins = spins.wrapping_add(1);
            if spins < DISPATCHER_SPIN_BEFORE_YIELD {
                continue;
            }
            if !yielded {
                yielded = true;
                self.dispatch_counters
                    .0
                    .dispatcher_yields
                    .fetch_add(1, Ordering::Relaxed);
            }
            if yields < yield_budget {
                yields = yields.saturating_add(1);
                thread::yield_now();
                continue;
            }
            self.park_until_complete();
        }
    }

    fn all_workers_done(&self) -> bool {
        self.node_pending
            .iter()
            .all(|counter| counter.0.load(Ordering::Acquire) == 0)
    }

    /// Block on `completion_sense` until a worker signals that the op is done.
    ///
    /// The handshake with [`SharedState::signal_completion`] is the standard
    /// two-sided futex protocol, and both sides need `SeqCst` on the
    /// store-then-load pair or a wakeup can be lost: the dispatcher publishes
    /// `dispatcher_parked` then reads `completion_sense`, while the worker
    /// publishes `completion_sense` then reads `dispatcher_parked`. If either
    /// side's store were allowed to sink past its load, both could conclude the
    /// other had not arrived -- the worker skipping the `wake` while the
    /// dispatcher commits to the sleep. `atomic_wait::wait` re-checks the value
    /// under the futex bucket lock, so a bump that lands inside the call itself
    /// returns immediately rather than sleeping.
    ///
    /// That pair is necessary but *not* the whole argument: it only orders the
    /// dispatcher against a worker that reaches the bump. The separate reason
    /// some worker always does reach it lives on
    /// [`SharedState::signal_completion`]. The `all_workers_done` call here is
    /// only a fast-path gate and may stay `Acquire` -- missing a late store just
    /// costs a park that the guaranteed bump then ends.
    fn park_until_complete(&self) {
        let observed = self.completion_sense.0.load(Ordering::SeqCst);
        self.dispatcher_parked.0.store(true, Ordering::SeqCst);
        if !self.all_workers_done() && self.completion_sense.0.load(Ordering::SeqCst) == observed {
            atomic_wait::wait(&self.completion_sense.0, observed);
        }
        self.dispatcher_parked.0.store(false, Ordering::SeqCst);
    }

    /// Called by the worker that brought its node's pending count to zero. Wakes
    /// a parked dispatcher, and does nothing but one relaxed check when the
    /// dispatcher is still spinning -- which is the common case.
    ///
    /// The opening fence is load-bearing on any weakly ordered target, and its
    /// absence is a deadlock rather than a slowdown. Each node's last worker
    /// arrives here having just stored zero to *its own* counter and about to
    /// load *every other* node's counter, which is the store-buffering shape: with
    /// nothing but the release store and acquire loads, two nodes' last workers
    /// may each read the other's pre-decrement value, so neither passes the gate
    /// below, nobody bumps `completion_sense`, and a dispatcher that has already
    /// parked sleeps until the process ends. The fence puts every such arrival
    /// into one total order, and the worker whose fence is last in it is
    /// sequenced after all the other zero stores, so it cannot miss them --
    /// giving at least one signaller for any node count. `x86_64` hides this
    /// because the `lock`-prefixed decrement is already a full barrier; `aarch64`
    /// does not. It costs one fence per node per op and only on the path that was
    /// about to signal anyway, so it is off the per-worker fast path entirely.
    fn signal_completion(&self) {
        fence(Ordering::SeqCst);
        if !self.all_workers_done() {
            return;
        }
        self.completion_sense.0.fetch_add(1, Ordering::SeqCst);
        if self.dispatcher_parked.0.load(Ordering::SeqCst) {
            atomic_wait::wake_all(&self.completion_sense.0);
        }
    }

    /// Wait for the next op published to `node` (its sense advancing past
    /// `last_seen`) or for shutdown, then return the observed sense value; the
    /// caller re-checks `shutdown`.
    ///
    /// KMP_BLOCKTIME-style bounded active spin then futex park. Phase 1 busy-waits
    /// (crossbeam-`Backoff`-style `spin_loop` ramp into `yield_now`) for up to
    /// `blocktime`, catching the common immediately-ready barrier release with the
    /// lowest latency. Phase 2 parks on the futex once idle past the window.
    ///
    /// No lost wakeup: `atomic_wait::wait(sense, last_seen)` sleeps only while the
    /// sense still equals `last_seen`; the kernel re-checks it under the futex
    /// bucket lock, so a [`SharedState::publish`] bump that races the arm makes
    /// `wait` return immediately instead of sleeping. The Acquire load pairs with
    /// publish's Release bump to make the job pointer and pending counts visible.
    fn worker_wait(
        &self,
        node: usize,
        global_index: usize,
        last_seen: u32,
        blocktime: Duration,
    ) -> u32 {
        let counters = &self.worker_counters[global_index].0;
        let sense = &self.node_sense[node].0.wake;
        // Phase 1: bounded active spin (blocktime), spin_loop ramping into yield.
        let mut spins = 0u32;
        let start = Instant::now();
        loop {
            let current = sense.load(Ordering::Acquire);
            if current != last_seen {
                counters.spin_hits.fetch_add(1, Ordering::Relaxed);
                return current;
            }
            if self.shutdown.load(Ordering::Acquire) {
                return current;
            }
            spins = spins.wrapping_add(1);
            if spins < SPIN_LOOP_BUDGET {
                std::hint::spin_loop();
            } else {
                thread::yield_now();
                // Check the clock on *every* yield, not on a stride -- the same
                // correction #1825 made to the readiness barrier below, which
                // missed this site. The clock is never read during the pure
                // spin phase here, so a stride amortised nothing: its only
                // effect was to multiply the granularity of the blocktime
                // deadline by 64 yields. `SPIN_LOOP_BUDGET` (4096) is itself a
                // multiple of that removed 64-iteration stride, so the yield
                // phase began exactly on a stride boundary -- the deadline was
                // evaluated once, on the first yield, and then not again for 64
                // more. A yield costs microseconds to milliseconds under
                // contention, and contention is exactly when a worker holding a
                // core past the window it was told to release at does the most
                // damage. `Instant::now()` is a vDSO read against a yield that
                // costs orders of magnitude more -- measured on this host,
                // 32ns per read against 1214ns for an *uncontended*
                // `yield_now`, i.e. 2.6%, and the fraction only shrinks as
                // contention makes the yield slower. Checking every time is
                // free exactly where it matters.
                // With the stride gone this file has no clock-stride constant
                // left: its only use was in the phase its own doc comment said
                // it did not apply to.
                if start.elapsed() >= blocktime {
                    break;
                }
            }
        }
        // Phase 2: park on the futex until the sense advances (or shutdown wakes
        // us via its own sense bump). Re-check under the guard for no lost wakeup.
        //
        // `parks` is counted when the observation *completes*, not when the
        // worker goes to sleep. That is what makes `spin_hits + parks` exactly
        // the number of op-observations: counting at sleep time would leave a
        // worker parked for a not-yet-published op counted against a dispatch
        // that has not happened, and any snapshot would be off by up to one per
        // worker. It also measures closer to the quantity that matters -- wakes
        // that served an op -- rather than sleeps entered. Not *exactly* wake
        // latency paid: a publish landing during the `wait` call itself returns
        // from the futex immediately, with no real sleep, and still counts here.
        // That race is narrow and does not affect the identity (still counted
        // once), but `parks` is an upper bound on wakes paid, not an equality.
        //
        // An observation can also land *here* without the worker ever sleeping:
        // the window expires, and the publish lands in the gap between breaking
        // out of phase 1 and the first check below. That worker paid no wake, so
        // it is a spin hit. Attributing it to neither (the obvious reading of
        // "phase 2 means parked") silently drops an observation, which is how
        // this branch was found -- the accounting identity failed under a loaded
        // runner, where the race is wide enough to hit.
        let mut parked = false;
        loop {
            let current = sense.load(Ordering::Acquire);
            if current != last_seen {
                if parked {
                    counters.parks.fetch_add(1, Ordering::Relaxed);
                } else {
                    counters.spin_hits.fetch_add(1, Ordering::Relaxed);
                }
                return current;
            }
            if self.shutdown.load(Ordering::Acquire) {
                return current;
            }
            if parked {
                // Woken with the sense unchanged: a spurious futex wake, or a
                // `wake_all` aimed at a sibling that had not yet re-armed.
                counters.spurious_wakes.fetch_add(1, Ordering::Relaxed);
            }
            parked = true;
            atomic_wait::wait(sense, last_seen);
        }
    }

    /// Block until no dispatch holds the pool's single publish slot, so a
    /// teardown cannot land in the middle of one.
    ///
    /// `publish` commits each node's pending count *before* it bumps that node's
    /// `ops`, so there is a window inside `publish` where a shard is already
    /// outstanding but no worker can yet see that it exists. A shutdown landing
    /// in that window is invisible to the `ops` gate in `worker_loop`: the
    /// worker correctly concludes it has no op to run, sees the flag, and
    /// leaves -- while the count it never decremented keeps the dispatcher in
    /// `wait` forever. Splitting the sense words cannot close this one, because
    /// the whole point is that the op has not been announced yet.
    ///
    /// Waiting for the claim closes it instead: `dispatch` holds it from before
    /// `publish` until after `wait` returns, so once it is free every committed
    /// count has already been retired. This cannot deadlock against a dispatcher
    /// blocked in `wait`, because that dispatcher is waiting on *workers*, which
    /// are still running -- the stop flag is not set until after this returns.
    ///
    /// Returns whether the pool actually went quiet within the bound.
    fn await_quiescent_dispatch(&self, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(claim) = DispatchClaim::try_claim(self) {
                // Dropped immediately: holding it across the flag store would
                // deadlock any dispatcher that is about to claim it, and the
                // flag alone is enough once the pool is quiet.
                drop(claim);
                return true;
            }
            if Instant::now() >= deadline {
                return false;
            }
            thread::yield_now();
        }
    }

    /// Flag shutdown, but not while a dispatch is mid-`publish`.
    ///
    /// The wait is what makes the flag safe to set; the two belong together, so
    /// they live in one function rather than as two statements a later edit can
    /// separate or reorder. Bounded, so a wedged dispatcher degrades teardown to
    /// the pre-existing behaviour with a warning instead of hanging it.
    ///
    /// Returns whether the pool went quiet before the flag went up.
    fn begin_shutdown(&self, timeout: Duration) -> bool {
        let quiet = self.await_quiescent_dispatch(timeout);
        self.shutdown.store(true, Ordering::SeqCst);
        quiet
    }

    fn counters(&self) -> SpmdCounters {
        let dispatch = &self.dispatch_counters.0;
        let mut counters = SpmdCounters {
            dispatches: dispatch.dispatches.load(Ordering::Relaxed),
            inline_dispatches: dispatch.inline_dispatches.load(Ordering::Relaxed),
            dispatcher_yields: dispatch.dispatcher_yields.load(Ordering::Relaxed),
            ..SpmdCounters::default()
        };
        for worker in &self.worker_counters {
            counters.spin_hits += worker.0.spin_hits.load(Ordering::Relaxed);
            counters.parks += worker.0.parks.load(Ordering::Relaxed);
            counters.spurious_wakes += worker.0.spurious_wakes.load(Ordering::Relaxed);
        }
        counters
    }

    fn worker_profiles(&self, cpus: &[Option<usize>]) -> Vec<SpmdWorkerProfile> {
        self.worker_counters
            .iter()
            .enumerate()
            .map(|(worker, counters)| SpmdWorkerProfile {
                worker,
                cpu: cpus.get(worker).copied().flatten(),
                spin_hits: counters.0.spin_hits.load(Ordering::Relaxed),
                parks: counters.0.parks.load(Ordering::Relaxed),
                spurious_wakes: counters.0.spurious_wakes.load(Ordering::Relaxed),
                last_arrivals: counters.0.last_arrivals.load(Ordering::Relaxed),
                timed_ops: counters.0.timed_ops.load(Ordering::Relaxed),
                wake_ns: counters.0.wake_ns.load(Ordering::Relaxed),
                work_ns: counters.0.work_ns.load(Ordering::Relaxed),
            })
            .collect()
    }

    fn panic_if_poisoned(&self) {
        let poisoned = self.poisoned_worker.load(Ordering::Acquire);
        if poisoned != 0 {
            let worker = poisoned - 1;
            panic!(
                "persistent SPMD decode worker {worker} panicked while executing a decode op; \
                 the pool is poisoned and cannot continue. Disable \
                 ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL or restart the process"
            );
        }
    }
}

/// Exclusive ownership of the pool's single job slot and barrier.
///
/// Claiming and releasing live together here so the two halves cannot drift
/// apart: the only way to construct one is [`DispatchClaim::try_claim`], and
/// `Drop` releases on every exit including an unwind. A leaked claim would send
/// every later dispatch inline forever, so the panic path matters.
struct DispatchClaim<'a> {
    shared: &'a SharedState,
}

/// How long [`SharedState::await_quiescent_dispatch`] waits for an in-flight
/// dispatch before giving up and tearing down anyway.
///
/// Bounded rather than unbounded on purpose: a dispatcher wedged for an
/// unrelated reason must not convert into a teardown that never returns. Giving
/// up degrades to the pre-existing behaviour (a possibly stranded dispatcher),
/// which is strictly better than hanging every shutdown behind one bad op.
const SHUTDOWN_DISPATCH_QUIESCE: Duration = Duration::from_secs(5);

impl<'a> DispatchClaim<'a> {
    /// Take exclusive ownership, or `None` if another thread already holds it.
    ///
    /// Acquire on success pairs with the Release in `Drop`, so the new owner
    /// observes the previous op's completion before overwriting the job slot.
    fn try_claim(shared: &'a SharedState) -> Option<Self> {
        shared
            .dispatching
            .0
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .ok()
            .map(|_| Self { shared })
    }
}

impl Drop for DispatchClaim<'_> {
    fn drop(&mut self) {
        self.shared.dispatching.0.store(false, Ordering::Release);
    }
}

/// What happened when a worker tried to take the CPU its plan named.
///
/// Three states because "no pin was asked for" and "a pin was asked for and did
/// not happen" are different facts with different consequences, and folding
/// either into success is the bug this whole type exists to prevent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PinAttempt {
    /// The plan left this worker unpinned, so nothing was attempted.
    NotRequested,
    /// The pin call returned success. Note carefully: this is the call's
    /// *claim*, and on its own it proves nothing -- see
    /// [`WorkerPlacement::observed`] for what the kernel actually did.
    Applied,
    /// The pin call returned an error, including targets that have no affinity
    /// mechanism and say so.
    Failed,
}

/// One spawned worker's realized placement, recorded by that worker about
/// itself immediately after its pin attempt.
///
/// Every field is either something the worker did or something the worker
/// asked the kernel; none of it is copied from the plan after the fact. That
/// separation is the whole point. A placement report assembled on the builder
/// thread from the assignment it just handed out is a restatement of the
/// request, and it stays true when the pinning underneath it has stopped
/// working.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkerPlacement {
    /// The worker's OS thread id, or `None` where the target exposes none.
    /// Present so a report can be correlated with `/proc/self/task/<tid>` by a
    /// human or an external tool.
    pub tid: Option<i64>,
    /// The CPU the plan named for this worker, `None` if it planned no pin.
    pub attempted_cpu: Option<usize>,
    /// What the pin call reported.
    pub attempt: PinAttempt,
    /// What the worker's own affinity mask actually was afterwards.
    pub observed: crate::decode_affinity::ObservedAffinity,
}

impl WorkerPlacement {
    /// The single CPU this worker is actually confined to, if it is confined to
    /// one at all. A multi-bit mask is not a pin regardless of what was asked
    /// for: the scheduler may use any CPU in it.
    pub fn realized_cpu(&self) -> Option<usize> {
        self.observed.pinned_cpu()
    }

    /// Does the observed mask match the pin this worker reports having taken?
    ///
    /// `None` when unanswerable (nothing was pinned here, or the mask could not
    /// be read). `Some(false)` is the #1792 shape in miniature: a worker that
    /// believes it is on a CPU the kernel never put it on.
    pub fn report_is_honest(&self) -> Option<bool> {
        let observed = self.observed.cpus()?;
        match (self.attempt, self.attempted_cpu) {
            (PinAttempt::Applied, Some(cpu)) => Some(observed == [cpu]),
            // A retracted or absent pin claims nothing, so there is nothing to
            // contradict -- but it must not be scored as honest *success*
            // either, hence `None` rather than `Some(true)`.
            _ => None,
        }
    }

    /// Does this worker assert a placement that something could contradict?
    ///
    /// The difference between the two kinds of `None` [`Self::report_is_honest`]
    /// returns, and the whole reason the pool-level verdict can be precise: a
    /// worker that never asked for a pin claims nothing and cannot be lying,
    /// while a worker that claims an applied pin and whose mask cannot be read
    /// is an *unverified claim*. Only the second has to withdraw a verdict.
    pub fn claims_a_pin(&self) -> bool {
        matches!(self.attempt, PinAttempt::Applied) && self.attempted_cpu.is_some()
    }
}

/// Why a realized-placement question could not be answered.
///
/// Named individually so a caller can fail closed on the ones that mean "this
/// host should have been able to answer" while still skipping the ones that
/// mean "no mechanism exists here". Collapsing them into one `None` is what
/// lets a regression hide inside a legitimate skip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlacementBlindSpot {
    /// The target has no per-thread affinity query at all (macOS and friends).
    AffinityQueryUnsupported,
    /// The query exists on this target but failed at runtime.
    AffinityQueryFailed,
    /// Core topology could not be detected, so siblings are unknown.
    TopologyUndetected,
}

/// Whether the pool's workers *are* -- as observed, not as planned -- one per
/// physical core.
///
/// Deliberately not `Option<bool>`. The old predicate returned `None` for both
/// "nothing was pinned" and "we could not tell", and every caller then wrote
/// `if let Some(ok) = ...`, which silently treats "could not tell" as "nothing
/// to assert". That is a guard switched off by the failure it guards against.
/// Here the unanswerable cases carry a reason and are trivially distinguishable
/// from [`Self::OneWorkerPerPhysicalCore`], which is the *only* success.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RealizedPlacement {
    /// Every worker is confined to a single CPU and no two of those CPUs share
    /// a physical core's front end.
    OneWorkerPerPhysicalCore,
    /// At least two workers can run on the same physical core -- either they
    /// landed on sibling CPUs, or one of them is not confined to a single CPU
    /// and so may land anywhere, including on top of a pinned peer.
    SharedCore,
    /// No worker is pinned, so the pool makes no placement claim to check.
    Unpinned,
    /// The question could not be answered, and this is never success.
    Unobservable(PlacementBlindSpot),
}

impl RealizedPlacement {
    /// True only for [`Self::OneWorkerPerPhysicalCore`].
    ///
    /// Provided so callers cannot accidentally write `!= SharedCore` and thereby
    /// count an unobservable pool as placed.
    pub fn is_one_worker_per_physical_core(&self) -> bool {
        matches!(self, Self::OneWorkerPerPhysicalCore)
    }
}

/// [`SpmdDecodePools::realized_placement`]'s decision, over supplied evidence.
///
/// Split out from the pool for the same reason
/// `core_topology::topology_or_fail_closed` takes its input as an argument: the
/// interesting cases here -- a target with no affinity query, a query that
/// failed, an undetectable topology -- cannot be produced on demand by building
/// a real pool on the hosts this suite runs on. A predicate reachable only
/// through a real pool is a predicate whose blind-spot handling nothing proves,
/// and blind-spot handling is the entire point of this type.
///
/// A global toggle would not do: the lib test binary is multi-threaded, so
/// unrelated tests building pools concurrently would observe the injected state
/// at random.
fn realized_placement_of(
    placements: &[WorkerPlacement],
    topology: Option<&crate::core_topology::CoreTopology>,
) -> RealizedPlacement {
    if placements.is_empty()
        || placements
            .iter()
            .all(|placement| placement.attempt == PinAttempt::NotRequested)
    {
        return RealizedPlacement::Unpinned;
    }
    // Answer the "we cannot see" cases before the "it is wrong" cases, so a
    // blind spot is never reported as a defect and -- far more importantly --
    // never as a success.
    for placement in placements {
        match &placement.observed {
            crate::decode_affinity::ObservedAffinity::Unsupported => {
                return RealizedPlacement::Unobservable(
                    PlacementBlindSpot::AffinityQueryUnsupported,
                );
            }
            crate::decode_affinity::ObservedAffinity::QueryFailed(_) => {
                return RealizedPlacement::Unobservable(PlacementBlindSpot::AffinityQueryFailed);
            }
            crate::decode_affinity::ObservedAffinity::Cpus(_) => {}
        }
    }
    let Some(cores) = topology else {
        return RealizedPlacement::Unobservable(PlacementBlindSpot::TopologyUndetected);
    };
    let mut seen = std::collections::BTreeSet::new();
    for placement in placements {
        // Not confined to one CPU -- whether because no pin was asked for or
        // because the pin did not take -- means this worker may land on any
        // core, including on top of a pinned peer, so no distinctness claim
        // survives it.
        let Some(cpu) = placement.realized_cpu() else {
            return RealizedPlacement::SharedCore;
        };
        let core: Vec<usize> = cores
            .siblings_of(cpu)
            .map_or_else(|| vec![cpu], <[usize]>::to_vec);
        if !seen.insert(core) {
            return RealizedPlacement::SharedCore;
        }
    }
    RealizedPlacement::OneWorkerPerPhysicalCore
}

/// Whether every worker that claims a pin is observably on the CPU it claims.
///
/// `None` means *unanswerable*, and it is answered for the pool rather than
/// per worker on purpose. Round-one review caught the obvious version of this
/// returning `Some(true)` the moment a single worker verified clean, discarding
/// the `None`s from workers whose masks could not be read -- the
/// unanswerable-read-as-answered defect this whole change exists to remove,
/// reintroduced one level up from where it was removed.
///
/// The two `None`s a worker can produce are *not* the same, and collapsing them
/// would swing the check from too weak to too strict:
///
/// * a worker that never asked for a pin claims nothing, cannot contradict
///   anything, and must not withdraw a verdict its peers can support -- a
///   partially pinned pool would otherwise become permanently unanswerable;
/// * a worker that claims an applied pin and cannot read its mask is an
///   *unverified claim*, and one of those is enough to sink the pool's verdict.
///
/// A definite `Some(false)` outranks both: "this worker is not where it says it
/// is" stays true however many of its peers went unchecked.
///
/// Taken as a free function over the evidence so the case that matters -- some
/// workers readable, some not -- is testable at all. It cannot be produced by
/// building a real pool on any host in CI, where readability is a property of
/// the target and so is uniform across a pool's workers.
fn placement_report_is_honest_of(placements: &[WorkerPlacement]) -> Option<bool> {
    if placements
        .iter()
        .any(|placement| placement.report_is_honest() == Some(false))
    {
        return Some(false);
    }
    if placements
        .iter()
        .any(|placement| placement.claims_a_pin() && placement.report_is_honest().is_none())
    {
        return None;
    }
    placements
        .iter()
        .any(|placement| placement.report_is_honest() == Some(true))
        .then_some(true)
}

/// A persistent SPMD decode pool: hot worker threads plus the shared barrier
/// state that drives them.
pub struct SpmdDecodePools {
    shared: Option<Arc<SharedState>>,
    /// Owned join handles, held behind a `Mutex` so [`SpmdDecodePools::shutdown`]
    /// can join the workers through a shared `&self` (the pool is reached as a
    /// `&'static` through [`pools`]). Only touched at teardown, never on the hot
    /// path. Drained on the first shutdown so the operation is idempotent.
    join_handles: Mutex<Vec<JoinHandle<()>>>,
    /// Deckard's Eigen-style persistent work-stealing pool. In `Steal` mode this
    /// is the only resident worker set: decode ops publish coarse MLAS/KAI tiles
    /// directly to this pool so delayed workers do not strand a static SPMD
    /// shard behind a hard barrier.
    #[cfg(feature = "mlas")]
    work_stealing_pool: Option<mlas_sys::WorkStealingThreadPool>,
    /// Compute shards assigned to each node, node-major, matching global worker
    /// index order (shards `0..counts[0]` are node 0, and so on).
    ///
    /// This is the *partition* width. It exceeds the number of spawned worker
    /// threads by one on the last node when [`Self::dispatcher_shard`] is set,
    /// because the dispatcher computes that shard itself.
    node_worker_counts: Vec<usize>,
    /// Worker *threads* per node, node-major. Equal to [`Self::node_worker_counts`]
    /// except on the last node when the dispatcher owns a shard.
    ///
    /// This is what [`SharedState::publish`] counts down: only spawned threads
    /// decrement `node_pending`, so the dispatcher's shard must not be included
    /// or [`SharedState::wait`] would never observe zero.
    node_thread_counts: Vec<usize>,
    /// The global shard index the dispatcher computes inline, if any.
    ///
    /// Always the last index (`total_workers - 1`), which keeps the spawned
    /// threads on a contiguous `0..total_threads` range so their global indices,
    /// their node assignment and their CPU pinning are all unchanged.
    dispatcher_shard: Option<usize>,
    total_workers: usize,
    schedule: DecodeSchedule,
    /// The CPU each spawned worker was *asked* to take, global-worker-index
    /// order, with the request retracted to `None` where the pin call reported
    /// failure.
    ///
    /// This is the **plan**, not the placement. It is derived from the
    /// assignment and from what `pin_current_thread_to_cpu` *returned*, and a
    /// return value is not an observation: a build where the syscall is stubbed,
    /// seccomp-filtered or emulated away answers `Ok(())` while every worker
    /// stays free to roam, and this field would record the pin as taken. What
    /// the kernel actually enforced lives in [`Self::worker_placements`], and
    /// realized-placement questions must be asked there.
    worker_cpus: Vec<Option<usize>>,
    /// What each spawned worker observed about *itself* after its pin attempt,
    /// global-worker-index order.
    ///
    /// Recorded by the worker thread, on the worker thread, by asking the kernel
    /// for its own effective mask -- so it is evidence rather than intent. This
    /// is the field #1792 needed and did not have: a pool could report
    /// `realized=16 as_requested` with a placement nothing had ever looked at.
    worker_placements: Vec<WorkerPlacement>,
    /// The allowed CPU that [`DISPATCHER_RESERVED_CPUS`] freed for the inline
    /// dispatcher, or `None` when the pool has no dispatcher shard or the
    /// reservation could not be located (unpinned workers, empty CPU list).
    ///
    /// Reserving the CPU and *using* it are two different things. The
    /// reservation is made in [`reserve_single_group_headroom`] /
    /// [`reserve_split_headroom`] and only guarantees that no worker is pinned
    /// there; the dispatcher itself is an ordinary unpinned thread that the
    /// scheduler is free to leave on a worker's core while the reserved one
    /// sits idle. Recording the reserved CPU here is what lets
    /// [`Self::dispatcher_observed_cpu`] check whether that actually happens,
    /// and [`DISPATCHER_PIN_ENV`] test whether closing the gap is worth
    /// anything. Measured answer so far: it is worth 9.5%, which is below the
    /// bar that was set for it.
    dispatcher_cpu: Option<usize>,
    /// OS thread id of the first thread to dispatch on this pool, or `0` before
    /// any dispatch has happened.
    ///
    /// Recorded because the dispatcher is *not* the thread that owns the pool,
    /// nor in general the process's main thread, so "which CPU is the
    /// dispatcher on" cannot be answered by asking whoever is reporting. The
    /// first attempt at this measurement read `sched_getcpu()` on the reporting
    /// thread and produced an exactly inverted result -- with the dispatcher
    /// unpinned the reporter sat on the reserved CPU (it is idle while the pool
    /// works, so the scheduler parks it on the one free core), and pinning the
    /// dispatcher *evicted* the reporter from that CPU. Both readings were of
    /// the wrong thread.
    ///
    /// Written once per pool via `compare_exchange` from the dispatch path's
    /// existing per-thread one-shot, so it costs one syscall per process and
    /// nothing on the steady path.
    dispatcher_tid: AtomicI64,
    /// The CPU the dispatcher was last seen on, sampled every
    /// [`DISPATCHER_CPU_SAMPLE_MASK`] + 1 dispatches, or `-1` before the first
    /// sample.
    ///
    /// Sampled inside the dispatch path because the dispatcher is a transient
    /// thread: by the time a harness reports, `/proc/self/task/<tid>` is
    /// usually already gone, so its placement has to be recorded while it is
    /// still running.
    dispatcher_observed_cpu: AtomicI64,
    /// How many consecutive dispatcher CPU samples differed from the one
    /// before.
    ///
    /// A lower bound on migrations, not a count of them: sampling every
    /// [`DISPATCHER_CPU_SAMPLE_MASK`] + 1 dispatches sees a thread that left
    /// and came back as no change at all. That is the right direction for the
    /// question it answers -- "does the unpinned dispatcher stay put?" -- since
    /// any non-zero reading is a migration that definitely happened, while zero
    /// is only evidence of stillness at this sampling rate.
    ///
    /// Sampled from one thread only (see `DISPATCHER_IS_RECORDED`), so a
    /// successfully pinned dispatcher reads exactly zero and this doubles as a
    /// check on the pin.
    dispatcher_cpu_changes: AtomicU64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DecodeSchedule {
    Fixed,
    Steal,
}

fn decode_schedule_from_raw(raw: Option<&str>) -> DecodeSchedule {
    match raw.map(str::trim) {
        Some(value) if value.eq_ignore_ascii_case("fixed") => DecodeSchedule::Fixed,
        Some(value) if value.eq_ignore_ascii_case("spmd") => DecodeSchedule::Fixed,
        #[cfg(feature = "mlas")]
        Some(value) if value.eq_ignore_ascii_case("steal") => DecodeSchedule::Steal,
        #[cfg(feature = "mlas")]
        Some(value) if value.eq_ignore_ascii_case("work-stealing") => DecodeSchedule::Steal,
        _ => DecodeSchedule::Fixed,
    }
}

fn decode_schedule() -> DecodeSchedule {
    decode_schedule_from_raw(std::env::var(DECODE_SCHEDULE_ENV).ok().as_deref())
}

fn steal_tiles_per_worker() -> usize {
    static TILES: OnceLock<usize> = OnceLock::new();
    *TILES.get_or_init(|| {
        std::env::var("ONNX_GENAI_CPU_DECODE_STEAL_TILES_PER_WORKER")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(DEFAULT_STEAL_TILES_PER_WORKER)
    })
}

impl SpmdDecodePools {
    /// Build the pool from per-node worker shards. Global worker indices are
    /// laid out node-major (node 0's workers first) so row segments and weight
    /// placement line up with the node assignment.
    ///
    /// `dispatcher_shard` adds one compute shard, owned by the dispatcher rather
    /// than by a spawned thread. See [`SpmdDecodePools::dispatcher_shard`].
    fn build(shards: &[NodeShard], dispatcher_shard: bool) -> Self {
        Self::build_with_schedule(shards, decode_schedule(), dispatcher_shard)
    }

    fn build_with_schedule(
        shards: &[NodeShard],
        schedule: DecodeSchedule,
        dispatcher_shard: bool,
    ) -> Self {
        let node_count = shards.len();
        let mut worker_node = Vec::new();
        let mut node_thread_counts = Vec::with_capacity(node_count);
        // Global (index, pinned cpu) assignment, node-major.
        let mut assignment: Vec<(usize, Option<usize>)> = Vec::new();
        for (node_position, shard) in shards.iter().enumerate() {
            node_thread_counts.push(shard.workers);
            for worker in 0..shard.workers {
                worker_node.push(node_position);
                let cpu = shard.cpus.get(worker % shard.cpus.len().max(1)).copied();
                assignment.push((node_position, cpu));
            }
        }
        let total_threads = assignment.len();
        // The dispatcher's shard is the last global index, so the spawned threads
        // keep the contiguous `0..total_threads` range they already had.
        let dispatcher_shard = (dispatcher_shard && node_count > 0).then_some(total_threads);
        let mut node_worker_counts = node_thread_counts.clone();
        if dispatcher_shard.is_some()
            && let Some(last) = node_worker_counts.last_mut()
        {
            *last += 1;
        }
        let total_workers = total_threads + usize::from(dispatcher_shard.is_some());
        // The CPU the headroom reservation freed. Workers take `shard.cpus` in
        // order (`worker % len`), so index `shard.workers` is the first CPU of
        // that node that no worker was pinned to -- exactly the CPU the reserve
        // exists to keep clear. The dispatcher's shard lives on the last node,
        // so that node's spare is the one it should sit on.
        let dispatcher_cpu = dispatcher_shard.and_then(|_| {
            let last = shards.last()?;
            last.cpus.get(last.workers).copied()
        });

        #[cfg(feature = "mlas")]
        if schedule == DecodeSchedule::Steal {
            // The work-stealing pool is its own executor with no inline
            // dispatcher, so there is no reserved CPU to reclaim: it spawns a
            // thread per shard and the dispatcher never computes.
            return Self {
                shared: None,
                join_handles: Mutex::new(Vec::new()),
                work_stealing_pool: Some(
                    mlas_sys::WorkStealingThreadPool::new(total_threads)
                        .expect("spawn persistent work-stealing decode pool"),
                ),
                node_worker_counts: node_thread_counts.clone(),
                node_thread_counts,
                dispatcher_shard: None,
                // The work-stealing pool spawns and places its own threads; we
                // assign no pin target, so there is no placement of ours to
                // report. Empty (rather than a vec of `None`) says "no claim"
                // instead of "claimed nothing", and `worker_cpus()` documents
                // the distinction.
                worker_cpus: Vec::new(),
                // Likewise: threads we neither spawned nor pinned have no
                // placement of ours to observe.
                worker_placements: Vec::new(),
                total_workers: total_threads,
                schedule,
                // No inline dispatcher on this path, so nothing reserved one.
                dispatcher_cpu: None,
                dispatcher_tid: AtomicI64::new(0),
                dispatcher_observed_cpu: AtomicI64::new(-1),
                dispatcher_cpu_changes: AtomicU64::new(0),
            };
        }

        let shared = Arc::new(SharedState {
            node_sense: (0..node_count)
                .map(|_| Padded(NodeSense::default()))
                .collect(),
            job: UnsafeCell::new(None),
            node_pending: (0..node_count)
                .map(|_| Padded(AtomicUsize::new(0)))
                .collect(),
            worker_node,
            ready: AtomicUsize::new(0),
            poisoned_worker: AtomicUsize::new(0),
            dispatching: Padded(AtomicBool::new(false)),
            shutdown: AtomicBool::new(false),
            workers_exited: AtomicUsize::new(0),
            profile_workers: worker_profile_enabled(),
            worker_counters: (0..total_threads)
                .map(|_| Padded(WorkerCounters::default()))
                .collect(),
            dispatch_counters: Padded(DispatchCounters::default()),
            completion_sense: Padded(AtomicU32::new(0)),
            dispatcher_parked: Padded(AtomicBool::new(false)),
        });

        let mut handles = Vec::with_capacity(total_threads);
        let mut worker_cpus: Vec<Option<usize>> = assignment.iter().map(|&(_, cpu)| cpu).collect();
        // A pin target is a request, and this whole change exists because a
        // request nobody verified is how #1729 and #1792 stayed invisible. Each
        // worker therefore records, about itself, what it attempted *and what
        // the kernel actually gave it* -- one `sched_getaffinity` per worker on
        // the one-time build path, nothing on the dispatch path.
        //
        // Read-back needs no further synchronization: a worker fills its slot
        // before incrementing `ready`, and the builder below waits on `ready`
        // with `Acquire`. `OnceLock` rather than a lock because exactly one
        // writer touches each slot exactly once.
        let placements: Arc<Vec<OnceLock<WorkerPlacement>>> =
            Arc::new((0..total_threads).map(|_| OnceLock::new()).collect());
        // Latched here, on the builder thread, so the fault belongs to *this*
        // pool. Reading it inside the spawned worker would read the worker's own
        // (default) thread-local and silently never fire.
        #[cfg(test)]
        let fail_worker_before_ready = FAIL_WORKER_BEFORE_READY.with(Cell::get);
        #[cfg(test)]
        let stub_pin_syscall = STUB_PIN_SYSCALL.with(Cell::get);
        #[cfg(test)]
        let delay_worker_before_ready = DELAY_WORKER_BEFORE_READY_MS.with(Cell::get);
        for (global_index, (node_position, cpu)) in assignment.into_iter().enumerate() {
            let shared = Arc::clone(&shared);
            let placements = Arc::clone(&placements);
            let handle = thread::Builder::new()
                .name(format!("onnx-genai-spmd-n{node_position}-{global_index}"))
                .spawn(move || {
                    let attempt = match cpu {
                        None => PinAttempt::NotRequested,
                        Some(cpu) => {
                            #[cfg(test)]
                            let pinned = if stub_pin_syscall {
                                // The mutation: the syscall removed, success
                                // reported. See `STUB_PIN_SYSCALL`.
                                Ok(())
                            } else {
                                crate::decode_affinity::pin_current_thread_to_cpu(cpu)
                            };
                            #[cfg(not(test))]
                            let pinned = crate::decode_affinity::pin_current_thread_to_cpu(cpu);
                            match pinned {
                                Ok(()) => PinAttempt::Applied,
                                Err(message) => {
                                    report_spmd_fallback(&format!(
                                        "worker {global_index} could not pin to cpu {cpu}: \
                                         {message}"
                                    ));
                                    PinAttempt::Failed
                                }
                            }
                        }
                    };
                    // Read back *after* the attempt, on this thread, from the
                    // kernel. This is the only line in the pool that can tell a
                    // pin that happened from a pin that was merely accepted.
                    let observed = crate::decode_affinity::observe_current_thread_cpus();
                    let _ = placements[global_index].set(WorkerPlacement {
                        tid: current_thread_os_id(),
                        attempted_cpu: cpu,
                        attempt,
                        observed,
                    });
                    #[cfg(test)]
                    assert!(
                        fail_worker_before_ready != global_index,
                        "injected fault: worker {global_index} dies before announcing"
                    );
                    #[cfg(test)]
                    if delay_worker_before_ready > 0 {
                        thread::sleep(Duration::from_millis(delay_worker_before_ready));
                    }
                    worker_loop(shared, global_index);
                })
                .expect("spawn persistent SPMD decode worker");
            handles.push(handle);
        }

        // Block until every worker has entered its loop and is waiting for ops.
        // Without this, a dispatch issued before a worker starts would set the
        // op's pending count for that worker, which would never arrive to
        // decrement it -- hanging the barrier. This counts spawned *threads*,
        // not compute shards: the dispatcher's shard has no thread to wait for.
        //
        // Spin briefly, then *yield*, then give up loudly. All three matter:
        //
        // * A pure `spin_loop` here waits on threads that need a core to make
        //   progress, so where the builder and its workers contend for the same
        //   CPU the spinner can starve the very workers it is waiting for. That
        //   is a livelock, not slow progress, and it is most reachable exactly
        //   where cores are scarcest -- an emulated target, a cpuset-confined
        //   process, or a test harness building several pools at once.
        // * A worker that dies before announcing readiness -- a panic anywhere
        //   in its pre-loop setup -- makes the condition permanently
        //   unsatisfiable. Spinning on it burns every core it holds forever,
        //   which is indistinguishable from work: an aarch64 suite hung this
        //   way for 5h40m at ~18 cores' worth of occupancy and was noticed only
        //   because someone went looking at `/proc`.
        //
        // The deadline is a liveness backstop, not a performance bound. Workers
        // announce readiness within microseconds of spawning, so a wait that
        // reaches it is reporting a broken pool, not a slow one.
        let ready_since = Instant::now();
        let mut spins = 0u32;
        while shared.ready.load(Ordering::Acquire) < total_threads {
            spins = spins.wrapping_add(1);
            if spins < SPIN_LOOP_BUDGET {
                std::hint::spin_loop();
            } else {
                thread::yield_now();
                #[cfg(test)]
                {
                    let slow = SLOW_YIELD_US.with(Cell::get);
                    if slow != 0 {
                        thread::sleep(Duration::from_micros(slow));
                    }
                }
                // Check the clock on *every* yield, not on a stride. The stride
                // is right for the spin phase, where an iteration costs
                // nanoseconds and `Instant::now()` would dominate; it is wrong
                // here, because a yield under contention costs microseconds to
                // milliseconds and a stride of N multiplies the deadline's
                // granularity by N yields of an already-starved thread.
                //
                // Measured: with a 100ms deadline and workers delayed 300ms on
                // a contended pair of CPUs, the loop reached spin 4141 in
                // 312ms. The yield phase starts at 4096, so the only multiple
                // of 64 it ever saw was 4096 itself -- the deadline was
                // evaluated once, early, and never again, and a build that had
                // blown its deadline by 3x completed as if nothing was wrong.
                // The starvation this backstop exists to escape is exactly the
                // condition that made the check unreachable.
                if ready_since.elapsed() >= pool_ready_timeout() {
                    let ready = shared.ready.load(Ordering::Acquire);
                    // The loop condition and this load are two separate reads,
                    // so a worker can announce between them. Accept that pool
                    // rather than tearing down a healthy one and reporting the
                    // self-contradictory "N of N workers announced ... never
                    // became ready". The deadline is a backstop against a
                    // condition that can no longer be satisfied; one that just
                    // was satisfied is not that case.
                    //
                    // This `Acquire` re-load gives the break path exactly the
                    // synchronisation the normal loop exit gives, so the
                    // `Relaxed` `pin_failed` reads below are equally justified
                    // on both. Note the branch is asserted by reasoning, not by
                    // a test: reaching it requires a worker to announce inside
                    // the window between two adjacent atomic loads, which no
                    // deterministic test can force.
                    if ready >= total_threads {
                        break;
                    }
                    // Release the workers that *did* start, using the same
                    // publish-then-wake sequence as `shutdown()`: the stop flag
                    // alone is not enough, because a worker already parked on
                    // the futex never re-reads it and would linger for the life
                    // of the process. Deliberately not joined -- a build that
                    // failed this way may have a wedged worker, and blocking on
                    // it here would reintroduce exactly the unbounded wait this
                    // backstop exists to remove. Woken threads exit on their
                    // own; the panic does not wait for them.
                    //
                    // Deliberately *not* `begin_shutdown`: that first waits out
                    // `SHUTDOWN_DISPATCH_QUIESCE` for an in-flight dispatch to
                    // drain. No dispatch can be in flight here -- this pool has
                    // never been returned to a caller and no job has ever been
                    // published -- so there is nothing to quiesce, and waiting
                    // would put a fresh timed wait on the failure path whose
                    // entire purpose is to stop waiting.
                    shared.shutdown.store(true, Ordering::SeqCst);
                    for sense in &shared.node_sense {
                        // Only the wake word, matching `shutdown()`: leaving
                        // `ops` untouched is what lets a woken worker tell this
                        // from a published op.
                        sense.0.wake.fetch_add(1, Ordering::Release);
                        atomic_wait::wake_all(&sense.0.wake);
                    }
                    panic!(
                        "persistent SPMD decode pool never became ready: {ready} of \
                         {total_threads} workers announced within {:?}. \
                         A worker most likely panicked before entering its loop; \
                         failing here rather than spinning on a condition that can \
                         no longer be satisfied.",
                        pool_ready_timeout()
                    );
                }
            }
        }

        // Every worker has now recorded its placement (it does so before
        // incrementing `ready`, which the `Acquire` load above synchronizes
        // with), so retract the targets whose pin call reported failure.
        let worker_placements: Vec<WorkerPlacement> = placements
            .iter()
            .enumerate()
            .map(|(index, slot)| {
                slot.get().cloned().unwrap_or_else(|| {
                    // Unreachable while `ready` counts the same threads that
                    // fill these slots, and a placement that is missing must
                    // read as "unknown", never as a successful pin.
                    WorkerPlacement {
                        tid: None,
                        attempted_cpu: worker_cpus.get(index).copied().flatten(),
                        attempt: PinAttempt::Failed,
                        observed: crate::decode_affinity::ObservedAffinity::QueryFailed(
                            "worker never recorded its placement".to_string(),
                        ),
                    }
                })
            })
            .collect();
        for (slot, placement) in worker_cpus.iter_mut().zip(worker_placements.iter()) {
            if placement.attempt != PinAttempt::Applied {
                *slot = None;
            }
        }

        // #1792 was invisible for as long as it was because nothing ever said
        // out loud that the placement had collapsed -- the width label was
        // honest and the placement label did not exist. Say it once, here, at
        // build time, off the dispatch path entirely.
        let realized = realized_placement_of(&worker_placements, crate::core_topology::host());
        if !matches!(
            realized,
            RealizedPlacement::OneWorkerPerPhysicalCore | RealizedPlacement::Unpinned
        ) {
            report_spmd_placement(&format!(
                "decode pool placement is {realized:?}: workers observed on {:?}",
                worker_placements
                    .iter()
                    .map(|placement| (placement.attempted_cpu, placement.realized_cpu()))
                    .collect::<Vec<_>>()
            ));
        }

        // Snapshot the worker `Thread`s for teardown join only; the hot dispatch
        // path wakes workers via the per-node futex, not per-thread `unpark`.
        Self {
            shared: Some(shared),
            join_handles: Mutex::new(handles),
            #[cfg(feature = "mlas")]
            work_stealing_pool: None,
            node_worker_counts,
            node_thread_counts,
            dispatcher_shard,
            total_workers,
            schedule,
            worker_cpus,
            worker_placements,
            dispatcher_cpu,
            dispatcher_tid: AtomicI64::new(0),
            dispatcher_observed_cpu: AtomicI64::new(-1),
            dispatcher_cpu_changes: AtomicU64::new(0),
        }
    }

    /// Total decode workers across all node groups.
    pub fn total_workers(&self) -> usize {
        self.total_workers
    }

    /// How many *spawned* worker threads this pool has, excluding the inline
    /// dispatcher shard.
    ///
    /// [`Self::total_workers`] counts compute participants, and the dispatching
    /// thread is one of them whenever `reserve_single_group_headroom` keeps a
    /// CPU free -- which is the common case on a fully-subscribed host, not a
    /// corner. That participant runs its shard inline and never enters
    /// [`worker_loop`], so it is not something teardown can join and not
    /// something [`Self::workers_exited`] can ever count. This is the number to
    /// compare an exit count against.
    pub fn spawned_workers(&self) -> usize {
        self.total_workers - usize::from(self.dispatcher_shard.is_some())
    }

    /// How many workers have left their loop for good.
    ///
    /// After [`Self::shutdown`] returns this equals [`Self::spawned_workers`],
    /// because `shutdown` joins and a join happens-after the counted exit.
    /// Deliberately *not* [`Self::total_workers`]: that includes the inline
    /// dispatcher shard, which never runs [`worker_loop`] and so can never be
    /// counted here. Before shutdown this is whatever has happened so far and
    /// is only meaningful as "not all of them". Exists so that "teardown
    /// finished" is assertable rather than inferred from thread names that
    /// outlive the join.
    pub fn workers_exited(&self) -> usize {
        self.shared
            .as_ref()
            .map_or(0, |shared| shared.workers_exited.load(Ordering::Acquire))
    }

    /// The CPU each spawned worker is *actually* pinned to, in global worker
    /// index order.
    ///
    /// `None` in a slot means that worker is unpinned -- either it was never
    /// assigned a target, or it was assigned one and the pin call failed, in
    /// which case the target is retracted here rather than reported as
    /// placement. That distinction is the point: reporting an intended pin as a
    /// realized one would reproduce, inside the very mechanism built to catch
    /// it, the unverified-label defect of #1729 and #1792.
    ///
    /// An **empty** slice means the pool makes no placement claim at all (the
    /// `mlas` work-stealing pool places its own threads); an **all-`None`**
    /// slice means this pool spawned workers and deliberately left them free
    /// (`ONNX_GENAI_CPU_DECODE_AFFINITY=off`, or a host without pinning).
    ///
    /// See [`Self::planned_placement_is_one_worker_per_physical_core`].
    pub fn worker_cpus(&self) -> &[Option<usize>] {
        &self.worker_cpus
    }

    /// The allowed CPU that the dispatcher reservation freed, if any.
    ///
    /// This is a *reservation*, not a placement: no worker is pinned here, but
    /// nothing pins the dispatcher here either unless [`DISPATCHER_PIN_ENV`] is
    /// on. Exposed so a harness can check the two independently -- which CPU
    /// was kept clear, and which CPU the dispatching thread actually ran on --
    /// rather than inferring one from the other.
    pub fn dispatcher_cpu(&self) -> Option<usize> {
        self.dispatcher_cpu
    }

    /// OS thread id of the thread that dispatched on this pool, once one has.
    ///
    /// `None` before the first dispatch, and on platforms with no stable
    /// per-thread id. The dispatcher is neither the pool's builder nor
    /// necessarily the process's main thread, so this is the only reliable way
    /// to ask where the dispatcher is actually running.
    pub fn dispatcher_thread_id(&self) -> Option<i64> {
        let tid = self.dispatcher_tid.load(Ordering::Relaxed);
        (tid != 0).then_some(tid)
    }

    /// The CPU the dispatcher was last sampled on, or `None` before any
    /// dispatch (and on platforms without `sched_getcpu`).
    ///
    /// A *sample*, not a residence: an unpinned dispatcher migrates, so equality
    /// with [`Self::dispatcher_cpu`] means "it was there when last looked", and
    /// only a pinned dispatcher can be said to stay.
    pub fn dispatcher_observed_cpu(&self) -> Option<usize> {
        let cpu = self.dispatcher_observed_cpu.load(Ordering::Relaxed);
        (cpu >= 0).then_some(cpu as usize)
    }

    /// Observed dispatcher CPU changes between consecutive samples.
    ///
    /// See [`Self::dispatcher_cpu_changes`]: a lower bound on migrations, and
    /// necessarily zero once the dispatcher is pinned.
    pub fn dispatcher_cpu_changes(&self) -> u64 {
        self.dispatcher_cpu_changes.load(Ordering::Relaxed)
    }

    /// Record the calling thread's current CPU as the dispatcher's placement,
    /// and count the sample as a change if it moved since the last one.
    ///
    /// Inert under Miri. Miri has no `sched_getcpu` shim, and the module's
    /// panic-safety test dispatches on a real pool under it, so an
    /// unconditional call aborts that test with an unsupported-operation
    /// error. Nothing is lost: a CPU-placement sample is meaningless under an
    /// interpreter that does not model CPUs, and the thing Miri is here to
    /// check -- that the unsafe blocks around it are sound -- is unaffected by
    /// not taking the sample.
    fn sample_dispatcher_cpu(&self) {
        #[cfg(all(target_os = "linux", not(miri)))]
        {
            // SAFETY: `sched_getcpu` takes no arguments and only reads the
            // calling thread's current CPU.
            let cpu = unsafe { libc::sched_getcpu() };
            if cpu < 0 {
                return;
            }
            let cpu = i64::from(cpu);
            let previous = self.dispatcher_observed_cpu.swap(cpu, Ordering::Relaxed);
            // `previous < 0` is the first sample, which has nothing to differ
            // from and must not be counted as a move.
            if previous >= 0 && previous != cpu {
                self.dispatcher_cpu_changes.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Whether the workers' **planned** CPUs occupy distinct physical cores.
    ///
    /// Reads [`Self::worker_cpus`], which is the assignment plus whatever
    /// `pin_current_thread_to_cpu` *returned*. That makes this a statement about
    /// the plan, not about the machine, and it must never be described as
    /// realized placement: it stays `Some(true)` on a build whose pinning does
    /// nothing but reports success. [`Self::realized_placement`] is the one that
    /// asks the kernel.
    ///
    /// Retained because the plan is still worth testing on its own -- a planner
    /// that scatters onto siblings is a defect even where pinning works -- and
    /// because separating the two is what makes each of them checkable.
    ///
    /// `None` only when the question is unanswerable: an unpinned pool (no plan
    /// to check) or an undiscoverable core topology. `Some(false)` states
    /// literally that at least two planned CPUs share a core's front end. It is
    /// deliberately not qualified by whether that was avoidable -- a pool asked
    /// for more workers than the host has cores must still admit that it doubled
    /// up. The caller knows the budget; this does not, and inferring it from the
    /// pinned set alone is circular, because a collapsed placement covers
    /// exactly as few cores as it landed on.
    pub fn planned_placement_is_one_worker_per_physical_core(&self) -> Option<bool> {
        let cores = crate::core_topology::host()?;
        let pinned: Vec<usize> = self.worker_cpus.iter().flatten().copied().collect();
        if pinned.is_empty() {
            return None;
        }
        // A pool that pinned only *some* of its workers has not established the
        // property, and must not be scored on its pinned subset alone: a free
        // worker is schedulable onto any core, including one a pinned worker
        // already owns. Judging the subset would let a pool with an entirely
        // unplaced node report `Some(true)` -- count honest, placement
        // unexamined, the same half-answer this predicate exists to end.
        // All-unpinned is a different case and returned `None` above: no claim,
        // rather than a broken one.
        if pinned.len() < self.worker_cpus.len() {
            return Some(false);
        }
        let mut seen = std::collections::BTreeSet::new();
        Some(pinned.iter().all(|&cpu| {
            let core: Vec<usize> = cores
                .siblings_of(cpu)
                .map_or_else(|| vec![cpu], <[usize]>::to_vec);
            seen.insert(core)
        }))
    }

    /// What each spawned worker observed about its own placement.
    ///
    /// Empty for a pool that spawned nothing of its own (the `--features mlas`
    /// work-stealing schedule).
    pub fn worker_placements(&self) -> &[WorkerPlacement] {
        &self.worker_placements
    }

    /// Whether the workers **are** one per physical core, as observed by the
    /// workers themselves.
    ///
    /// This is the placement counterpart to `decode_width().is_as_requested()`,
    /// and unlike [`Self::planned_placement_is_one_worker_per_physical_core`] it
    /// cannot be satisfied by a pin that did not happen. Width was always
    /// reported and asserted; placement was neither, and #1792 is what that
    /// costs -- a pool can report `realized=16 as_requested` while running on
    /// half the cores it claims.
    ///
    /// Three things have to hold for [`RealizedPlacement::OneWorkerPerPhysicalCore`]:
    /// every worker was asked to pin, every worker's *observed* mask is a single
    /// CPU, and those CPUs sit on distinct physical cores. A pool that pinned
    /// only some of its workers is `SharedCore`, not a partial pass: a free
    /// worker is schedulable onto any core, including one a pinned worker
    /// already owns, so its pinned subset establishes nothing.
    pub fn realized_placement(&self) -> RealizedPlacement {
        realized_placement_of(&self.worker_placements, crate::core_topology::host())
    }

    /// Whether every worker that reports a pin is actually on the CPU it
    /// reports.
    ///
    /// The **policy-neutral** half of the placement contract: it asserts nothing
    /// about *where* workers should run -- spread, compact, one per core, two per
    /// core are all equally acceptable to it -- only that
    /// [`Self::worker_cpus`], which benchmark rows label placement from, is not
    /// a claim the process failed to carry out.
    ///
    /// `None` when unanswerable: nothing was pinned, or a worker that claims a
    /// pin could not have that claim checked. Answered from each worker's own
    /// recorded mask rather than by scanning `/proc/self/task` for thread names,
    /// so a second pool running concurrently cannot contaminate the answer. See
    /// [`placement_report_is_honest_of`] for why the two unanswerable cases are
    /// kept apart.
    pub fn placement_report_is_honest(&self) -> Option<bool> {
        placement_report_is_honest_of(&self.worker_placements)
    }

    /// Number of node groups in the layout.
    pub fn node_count(&self) -> usize {
        self.node_worker_counts.len()
    }

    /// Snapshot this pool's scheduling counters. See [`SpmdCounters`].
    ///
    /// Monotonic since the pool was built, so a harness subtracts two snapshots
    /// around the phase it cares about. Returns [`SpmdCounters::default`] on the
    /// `--features mlas` work-stealing schedule, which has no barrier of ours to
    /// instrument.
    pub fn counters(&self) -> SpmdCounters {
        self.shared
            .as_ref()
            .map_or_else(SpmdCounters::default, |shared| shared.counters())
    }

    /// Per-worker barrier attribution, in global worker index order. See
    /// [`SpmdWorkerProfile`].
    ///
    /// Monotonic since the pool was built, like [`Self::counters`], so a harness
    /// subtracts two snapshots around the phase it cares about. Empty on the
    /// `--features mlas` work-stealing schedule, which has no barrier of ours.
    ///
    /// The dispatcher's own shard is deliberately absent: it never waits to be
    /// woken and never acknowledges the barrier, so every field here would be
    /// either meaningless or zero for it, and including a permanently-zero row
    /// would read as "this worker is never late" rather than "not measured".
    pub fn worker_profiles(&self) -> Vec<SpmdWorkerProfile> {
        self.shared
            .as_ref()
            .map_or_else(Vec::new, |shared| shared.worker_profiles(&self.worker_cpus))
    }

    fn uses_work_stealing(&self) -> bool {
        self.schedule == DecodeSchedule::Steal
    }

    /// Signal every worker to stop and **join** them. Idempotent: the join
    /// handles are drained on the first call, so a second call (e.g. `Drop`
    /// after an explicit [`shutdown_pools`]) is a no-op.
    ///
    /// Reachable through a shared `&self` (the process-wide pool is handed out as
    /// a `&'static` by [`pools`]); the join handles live behind a `Mutex` so this
    /// can consume them without owning the pool. This is a teardown-only call and
    /// must not race a live decode dispatch -- after it returns the workers are
    /// gone and the pool must not be dispatched to again.
    pub fn shutdown(&self) {
        if self.shared.is_none() {
            return;
        }
        let Some(shared) = self.shared.as_ref() else {
            return;
        };
        // Take the handles first so concurrent callers can't double-join; if
        // another thread already drained them, there is nothing to do.
        let handles: Vec<JoinHandle<()>> = {
            let mut guard = self
                .join_handles
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard.drain(..).collect()
        };
        if handles.is_empty() {
            return;
        }
        // Let any in-flight dispatch finish publishing and draining, then publish
        // the stop flag; see `await_quiescent_dispatch` for why the `ops` split
        // alone cannot cover a shutdown that lands *inside* `publish`.
        if !shared.begin_shutdown(SHUTDOWN_DISPATCH_QUIESCE) {
            // Deliberately not `report_spmd_fallback`: that latches the decode
            // path label to "flat", which would be a false statement about how
            // the pool ran, emitted at teardown purely to carry a warning.
            //
            // `warn` rather than `debug`, and ungated: reaching this means a
            // dispatcher is probably about to hang, and per #1812 the lesson of
            // this campaign is that a debug-gated diagnostic is one nobody reads.
            #[cfg(feature = "tracing")]
            tracing_crate::warn!(
                timeout = ?SHUTDOWN_DISPATCH_QUIESCE,
                "cpu decode pool shutdown proceeded with a dispatch still in flight; a \
                 dispatcher may be left waiting on a shard that will not be retired"
            );
            #[cfg(not(feature = "tracing"))]
            eprintln!(
                "onnx-genai: persistent SPMD decode pool: shutdown proceeded with a \
                 dispatch still in flight after {SHUTDOWN_DISPATCH_QUIESCE:?}; a dispatcher \
                 may be left waiting on a shard that will not be retired"
            );
        }
        // Bump every node's sense so spinning workers observe the change and
        // re-check `shutdown`, and futex-wake any parked worker so it leaves the
        // park. The bump-then-wake ordering (mirroring the dispatch path) wakes a
        // worker that raced into parking: it either sees the advanced sense under
        // the futex guard or is woken by `wake_all`.
        for sense in &shared.node_sense {
            // Only the wake word: `ops` stays put, so a woken worker can tell
            // this from a published op and will not re-run a retired `Job`.
            sense.0.wake.fetch_add(1, Ordering::Release);
            atomic_wait::wake_all(&sense.0.wake);
        }
        for handle in handles {
            let _ = handle.join();
        }
    }

    /// Run every shard of `job` on this thread, in global worker order.
    ///
    /// The shard closures partition their output into disjoint regions indexed
    /// by global worker index, so running all of them on one thread is exactly
    /// the same computation as broadcasting them -- only serial. This is the
    /// fallback when the pool's single publish/wait slot is already claimed.
    fn dispatch_inline<F>(&self, job: &F)
    where
        F: Fn(usize) + Sync,
    {
        for global_index in 0..self.total_workers {
            job(global_index);
        }
    }

    /// Broadcast `job` to the workers and block until all have finished.
    ///
    /// `job(global_worker_index)` runs the shard owned by that worker. The
    /// dispatcher (this thread) does not compute; it only publishes and waits,
    /// mirroring an external `pool.install` where the caller blocks.
    ///
    /// The pool has one job slot and one barrier, so it can serve one
    /// dispatcher at a time. When another thread already owns them this runs
    /// the shards inline instead of waiting or racing -- see
    /// [`SharedState::dispatching`].
    /// Record this thread as the dispatcher and, if [`DISPATCHER_PIN_ENV`] is
    /// on, bind it to the CPU the headroom reserve freed. At most once per
    /// thread.
    ///
    /// The tid is recorded whether or not the pin is requested: with the knob
    /// off, "which CPU did the scheduler leave the dispatcher on" is the
    /// measurement the knob exists to answer, and it needs the same identity.
    ///
    /// The pin attempt is recorded before it is made, so a host that refuses it
    /// costs one failed syscall for the life of the thread rather than one per
    /// op.
    ///
    /// Called from [`Self::dispatch`] rather than at build time because the
    /// pool is a process-wide static built on whichever thread decodes first,
    /// which need not be the thread that goes on to dispatch.
    fn bind_dispatcher_to_reserved_cpu(&self) {
        let tick = DISPATCHER_TICK.with(|t| {
            let seen = t.get();
            t.set(seen.wrapping_add(1));
            seen
        });
        if tick != 0 {
            if tick & DISPATCHER_CPU_SAMPLE_MASK == 0
                && DISPATCHER_IS_RECORDED.with(std::cell::Cell::get)
            {
                self.sample_dispatcher_cpu();
            }
            return;
        }
        // First dispatcher wins the identity slot: it is the one whose
        // placement is sampled from here on. Later dispatching threads still
        // take the reserved CPU below -- the point of the reserve is that
        // whoever is dispatching sits there -- they just do not report.
        let recorded = match current_thread_os_id() {
            Some(tid) => self
                .dispatcher_tid
                .compare_exchange(0, tid, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok(),
            None => false,
        };
        DISPATCHER_IS_RECORDED.with(|flag| flag.set(recorded));
        let Some(cpu) = self.dispatcher_cpu else {
            if recorded {
                self.sample_dispatcher_cpu();
            }
            return;
        };
        if !dispatcher_pin_requested() {
            if recorded {
                self.sample_dispatcher_cpu();
            }
            return;
        }
        // Miri has no `sched_setaffinity` shim either, and a pin is not a
        // property Miri can check. Refuse rather than abort, so that setting
        // the knob in a Miri environment degrades to "not pinned" instead of
        // failing an unrelated test.
        if cfg!(miri) {
            return;
        }
        match crate::decode_affinity::pin_current_thread_to_cpu(cpu) {
            Ok(()) => report_dispatcher_pin(&format!(
                "{DISPATCHER_PIN_ENV} on: dispatcher pinned to reserved cpu {cpu}"
            )),
            Err(message) => report_dispatcher_pin(&format!(
                "{DISPATCHER_PIN_ENV} on, but pinning the dispatcher to reserved cpu \
                 {cpu} failed: {message}; dispatcher left unpinned"
            )),
        }
        // After the pin, never before: the first sample is the baseline every
        // later one is compared against, so taking it pre-pin would score the
        // pin itself as a migration and a successfully pinned dispatcher could
        // never read zero.
        if recorded {
            self.sample_dispatcher_cpu();
        }
    }

    fn dispatch<F>(&self, job: &F)
    where
        F: Fn(usize) + Sync,
    {
        #[cfg(feature = "mlas")]
        if let Some(pool) = &self.work_stealing_pool {
            pool.parallel_for(0, self.total_workers, 1, |begin, end| {
                for global_index in begin..end {
                    job(global_index);
                }
            });
            return;
        }
        let shared = self
            .shared
            .as_ref()
            .expect("fixed SPMD dispatch requires shared worker state");
        shared.panic_if_poisoned();
        self.bind_dispatcher_to_reserved_cpu();
        unsafe fn call<F>(data: *const (), global_index: usize)
        where
            F: Fn(usize) + Sync,
        {
            // SAFETY: `data` came from a live `&F`; synchronous dispatch keeps
            // that borrow alive until every worker acknowledges this op.
            let job = unsafe { &*data.cast::<F>() };
            job(global_index);
        }
        // The pool serves one dispatcher at a time; a second one runs the same
        // shards inline rather than racing for the slot.
        let Some(claim) = DispatchClaim::try_claim(shared) else {
            shared
                .dispatch_counters
                .0
                .inline_dispatches
                .fetch_add(1, Ordering::Relaxed);
            self.dispatch_inline(job);
            return;
        };
        let job_ptr = Job {
            data: std::ptr::from_ref(job).cast(),
            call: call::<F>,
        };
        shared
            .dispatch_counters
            .0
            .dispatches
            .fetch_add(1, Ordering::Relaxed);
        shared.publish(job_ptr, &self.node_thread_counts);
        // Compute the dispatcher's own shard, when it has one, on the CPU the
        // headroom reservation kept free. `catch_unwind` is load-bearing rather
        // than defensive: the workers are already running against a `Job` that
        // borrows `job` off *this* stack frame, so unwinding straight out of
        // here would free the pointee while they still read it. Catch, complete
        // the barrier, then resume the panic.
        let unwind = self.dispatcher_shard.and_then(|global_index| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job(global_index))).err()
        });
        shared.wait();
        drop(claim);
        if let Some(payload) = unwind {
            // A worker panic during the same op stays latched in
            // `poisoned_worker` and is reported by the next dispatch's
            // `panic_if_poisoned`, which runs before it publishes anything. The
            // dispatcher's own payload wins this frame because it is the one
            // carrying the caller's stack; neither is lost, and `wait` above
            // already guaranteed no worker is still running.
            std::panic::resume_unwind(payload);
        }
        shared.panic_if_poisoned();
    }

    /// Split `n` output rows across the node groups proportionally to their
    /// worker counts (contiguous, non-overlapping, last node absorbs the
    /// remainder), matching [`crate::decode_numa`] so weight placement and
    /// compute dispatch always line up.
    fn node_row_lengths(&self, n: usize) -> Vec<usize> {
        let node_count = self.node_worker_counts.len();
        let mut lengths = Vec::with_capacity(node_count);
        let mut assigned = 0;
        for (position, &node_workers) in self.node_worker_counts.iter().enumerate() {
            let rows = if position + 1 == node_count {
                n - assigned
            } else {
                n.saturating_mul(node_workers) / self.total_workers
            };
            assigned += rows;
            lengths.push(rows);
        }
        lengths
    }

    /// Contiguous `(start, len)` output-row segment for each global worker index,
    /// node-major: a node's rows are split evenly across that node's workers.
    fn worker_row_segments(&self, n: usize) -> Vec<(usize, usize)> {
        let node_lengths = self.node_row_lengths(n);
        let mut segments = Vec::with_capacity(self.total_workers);
        let mut node_start = 0;
        for (&node_len, &node_workers) in node_lengths.iter().zip(&self.node_worker_counts) {
            let base = node_len / node_workers;
            let remainder = node_len % node_workers;
            let mut offset = node_start;
            for worker in 0..node_workers {
                let len = base + usize::from(worker < remainder);
                segments.push((offset, len));
                offset += len;
            }
            node_start += node_len;
        }
        segments
    }

    /// [`Self::worker_row_segments`] with every interior boundary snapped to a
    /// multiple of `align`. The per-worker split is computed exactly as the
    /// unaligned version (so node-major ordering and weight placement still line
    /// up), then each cumulative boundary except the final `n` is rounded to the
    /// nearest multiple of `align`, kept monotonic and in `[0, n]`. The result
    /// still covers `0..n` exactly once; a boundary collision can leave a worker
    /// with a zero-length segment (it simply runs no work), which the dispatch
    /// and shard-build paths already tolerate.
    ///
    /// `align <= 1` is the identity (returns the unaligned segments): callers
    /// whose per-column arithmetic is partition-independent pass `1`.
    fn worker_row_segments_aligned(&self, n: usize, align: usize) -> Vec<(usize, usize)> {
        let base = self.worker_row_segments(n);
        if align <= 1 {
            return base;
        }
        let mut segments = Vec::with_capacity(base.len());
        let mut prev_boundary = 0;
        let mut cumulative = 0;
        let last = base.len().saturating_sub(1);
        for (index, &(_, len)) in base.iter().enumerate() {
            cumulative += len;
            let boundary = if index == last {
                // The final boundary is always `n`, even when `n` is not a
                // multiple of `align`: the last shard's start is aligned, so its
                // trailing partial N-tile matches the full-width call's tail.
                n
            } else {
                // Round the ideal cumulative boundary to the nearest multiple of
                // `align`, staying monotonic and within bounds.
                let rounded = ((cumulative + align / 2) / align) * align;
                rounded.clamp(prev_boundary, n)
            };
            segments.push((prev_boundary, boundary - prev_boundary));
            prev_boundary = boundary;
        }
        segments
    }

    fn work_stealing_segments_aligned(&self, n: usize, align: usize) -> Vec<(usize, usize)> {
        if n == 0 {
            return vec![(0, 0)];
        }
        let align = align.max(1);
        let min_tile = MIN_STEAL_OUTPUTS_PER_TASK.div_ceil(align) * align;
        let max_by_size = n.div_ceil(min_tile).max(1);
        let target = self
            .total_workers
            .saturating_mul(steal_tiles_per_worker())
            .max(1)
            .min(max_by_size)
            .min(n);
        if target <= self.total_workers {
            return self.worker_row_segments_aligned(n, align);
        }
        let mut segments = Vec::with_capacity(target);
        let mut prev = 0usize;
        for index in 0..target {
            let boundary = if index + 1 == target {
                n
            } else {
                let ideal = n.saturating_mul(index + 1).div_ceil(target);
                let rounded = ((ideal + align / 2) / align) * align;
                rounded.clamp(prev, n)
            };
            segments.push((prev, boundary - prev));
            prev = boundary;
        }
        segments
    }

    fn output_segments_aligned(&self, n: usize, align: usize) -> Vec<(usize, usize)> {
        if self.uses_work_stealing() {
            self.work_stealing_segments_aligned(n, align)
        } else {
            self.worker_row_segments_aligned(n, align)
        }
    }

    /// Shard `result`'s output rows across the workers and run `compute` on each
    /// worker's contiguous slice under one lightweight barrier.
    ///
    /// `compute(output_start, outputs)` fills the rows
    /// `output_start .. output_start + outputs.len()` -- the same closure the
    /// flat path hands to `par_chunks_mut`, so the arithmetic is identical.
    /// Tiny ops (below the flat path's parallelization threshold) run serially
    /// on the dispatcher, so the same set of ops parallelize as before.
    ///
    /// The serial threshold is sized from [`Self::total_workers`], the executor
    /// that will actually run the fan-out, and *not* from the ambient Rayon
    /// width -- this pool is not a Rayon pool. See [`output_chunk_len_for`].
    pub fn dispatch_output_rows<F>(&self, result: &mut [f32], k: usize, compute: &F)
    where
        F: Fn(usize, &mut [f32]) + Sync,
    {
        let n = result.len();
        if self.total_workers <= 1 || output_chunk_len_for(self.total_workers, n, k) >= n {
            compute(0, result);
            return;
        }
        if self.uses_work_stealing() {
            self.dispatch_rows_work_stealing(result, 1, compute);
            return;
        }
        self.dispatch_rows_across_workers(result, &compute);
    }

    /// Public view of the contiguous `(start, len)` output-column segment each
    /// global worker owns when a length-`n` GEMV output is sharded across the
    /// pool. Callers that pre-partition a weight along N (e.g. one MLAS SQNBit
    /// packed shard per worker) use this to build shards that line up exactly
    /// with [`Self::dispatch_output_rows_indexed`].
    ///
    /// Every segment boundary is snapped to a multiple of `align` (the last,
    /// `n`-terminated segment excepted). This matters for kernels whose SIMD
    /// column-tiling is *not* bit-stable across an arbitrary N-partition: MLAS's
    /// SQNBit GEMV processes output columns in fixed-width N-tiles, so a shard
    /// boundary that falls *mid-tile* forces MLAS's remainder path to reduce a
    /// block-sum in a different order than the full-width call, shifting that
    /// column by ~1 ULP. Aligning every interior boundary to the N-tile width
    /// keeps every tile whole inside a single shard, so each shard reproduces
    /// the full-width tiling exactly and the concatenated output is
    /// bit-identical to the unsharded call (verified `max_ulp = 0`). Pass
    /// `align = 1` for kernels whose per-column result is already
    /// partition-independent (e.g. the hand int4/int8 GEMV).
    pub fn output_column_segments(&self, n: usize, align: usize) -> Vec<(usize, usize)> {
        self.output_segments_aligned(n, align)
    }

    /// Like [`Self::dispatch_output_rows`], but hands each worker its global
    /// index alongside its output slice and always dispatches across the pool
    /// (no serial-threshold short-circuit), so a caller can select the matching
    /// pre-partitioned weight shard (`compute(global_index, output_start,
    /// outputs)`). `result.len()` must equal `n` passed to
    /// [`Self::output_column_segments`], and `align` must match so the dispatch
    /// segments line up byte-for-byte with the caller's pre-built shards; each
    /// worker writes only its own segment, so the concatenated result is
    /// bit-identical to the single-worker path.
    pub fn dispatch_output_rows_indexed<F>(&self, result: &mut [f32], align: usize, compute: &F)
    where
        F: Fn(usize, usize, &mut [f32]) + Sync,
    {
        let n = result.len();
        let segments = self.output_segments_aligned(n, align);
        let table = RowTable {
            base: result.as_mut_ptr(),
            segments: &segments,
        };
        let table = &table;
        if self.uses_work_stealing() {
            let next = AtomicUsize::new(0);
            let next = &next;
            #[cfg(feature = "mlas")]
            if let Some(pool) = &self.work_stealing_pool {
                pool.parallel_for(0, segments.len(), 1, |begin, end| {
                    for task_index in begin..end {
                        let (start, len) = table.segments[task_index];
                        if len == 0 {
                            continue;
                        }
                        // SAFETY: dynamic segments are disjoint, in-bounds
                        // column ranges; each task index is claimed once by
                        // exactly one work-stealing chunk.
                        let outputs =
                            unsafe { std::slice::from_raw_parts_mut(table.base.add(start), len) };
                        compute(task_index, start, outputs);
                    }
                });
                return;
            }
            let job = move |_global_index: usize| loop {
                let task_index = next.fetch_add(1, Ordering::Relaxed);
                let Some(&(start, len)) = table.segments.get(task_index) else {
                    break;
                };
                if len == 0 {
                    continue;
                }
                // SAFETY: dynamic segments are disjoint, in-bounds column ranges;
                // each task index is claimed once by exactly one worker.
                let outputs = unsafe { std::slice::from_raw_parts_mut(table.base.add(start), len) };
                compute(task_index, start, outputs);
            };
            self.dispatch(&job);
            return;
        }
        let job = move |global_index: usize| {
            let (start, len) = table.segments[global_index];
            if len == 0 {
                return;
            }
            // SAFETY: `worker_row_segments` produces disjoint, in-bounds column
            // ranges covering `0..n` exactly once, so each worker's slice never
            // aliases another's.
            let outputs = unsafe { std::slice::from_raw_parts_mut(table.base.add(start), len) };
            compute(global_index, start, outputs);
        };
        self.dispatch(&job);
    }

    /// Shard `result`'s `num_rows` fixed-width rows (each `row_len` elements)
    /// across the resident workers and run `compute(row_index, row_slice)` on
    /// each whole row under one lightweight barrier.
    ///
    /// Unlike [`Self::dispatch_output_rows`] (which shards a GEMV's scalar output
    /// rows), this keeps every `row_len`-element row intact on a single worker,
    /// so a caller whose per-row closure needs the full contiguous row (e.g. an
    /// attention head's output vector) can run on the persistent decode pool
    /// instead of a second, contending thread pool. Rows are handed out
    /// contiguously, so concatenating the per-worker slices reproduces the
    /// single-threaded result bit-for-bit (each row is independent).
    pub fn dispatch_output_row_blocks<F>(
        &self,
        result: &mut [f32],
        row_len: usize,
        num_rows: usize,
        compute: &F,
    ) where
        F: Fn(usize, &mut [f32]) + Sync,
    {
        debug_assert_eq!(result.len(), row_len.saturating_mul(num_rows));
        if self.total_workers <= 1 || num_rows <= 1 || row_len == 0 {
            for row in 0..num_rows {
                compute(row, &mut result[row * row_len..(row + 1) * row_len]);
            }
            return;
        }
        let segments = self.worker_row_segments(num_rows);
        let table = RowBlockTable {
            base: result.as_mut_ptr(),
            row_len,
            segments: &segments,
        };
        let table = &table;
        let job = move |global_index: usize| {
            let (start, len) = table.segments[global_index];
            for row in start..start + len {
                // SAFETY: `worker_row_segments` produces disjoint, in-bounds row
                // ranges covering `0..num_rows` exactly once, so each worker's
                // `[row*row_len, (row+1)*row_len)` slice never aliases another's.
                let slice = unsafe {
                    std::slice::from_raw_parts_mut(
                        table.base.add(row * table.row_len),
                        table.row_len,
                    )
                };
                compute(row, slice);
            }
        };
        self.dispatch(&job);
    }

    /// Run `num_tasks` independent subtasks across the resident workers under one
    /// lightweight barrier: each worker runs a contiguous range of task indices
    /// and invokes `compute(task_index)` for each.
    ///
    /// Unlike [`Self::dispatch_output_row_blocks`] this partitions only the
    /// *index space*, not a shared result buffer — the caller's closure is
    /// responsible for writing disjoint scratch per task index (flash-decoding
    /// writes one `(max, sum, value-accumulator)` partial per task). The
    /// contiguous, non-overlapping partition is exactly
    /// [`Self::worker_row_segments`], so every index in `0..num_tasks` runs on
    /// exactly one worker, once.
    pub fn dispatch_index_tasks<F>(&self, num_tasks: usize, compute: &F)
    where
        F: Fn(usize) + Sync,
    {
        if self.total_workers <= 1 || num_tasks <= 1 {
            for task in 0..num_tasks {
                compute(task);
            }
            return;
        }
        let segments = self.worker_row_segments(num_tasks);
        let segments = &segments;
        let job = move |global_index: usize| {
            let (start, len) = segments[global_index];
            for task in start..start + len {
                compute(task);
            }
        };
        self.dispatch(&job);
    }

    /// Broadcast the output-row shards to every worker under one barrier,
    /// unconditionally (no serial-threshold check). The public
    /// [`Self::dispatch_output_rows`] applies the threshold before calling this;
    /// tests exercise the multi-worker path directly through it.
    fn dispatch_rows_across_workers<F>(&self, result: &mut [f32], compute: &F)
    where
        F: Fn(usize, &mut [f32]) + Sync,
    {
        let n = result.len();
        let segments = self.worker_row_segments(n);
        let table = RowTable {
            base: result.as_mut_ptr(),
            segments: &segments,
        };
        // Bind a reference so the `move` closure captures the whole `RowTable`
        // (which carries the manual `Sync` impl) rather than its raw pointer
        // field individually (disjoint capture does not reach through a
        // reference).
        let table = &table;
        let job = move |global_index: usize| {
            let (start, len) = table.segments[global_index];
            if len == 0 {
                return;
            }
            // SAFETY: `worker_row_segments` produces disjoint, in-bounds row
            // ranges covering `0..n` exactly once, and each worker touches only
            // its own segment, so these mutable slices never alias.
            let outputs = unsafe { std::slice::from_raw_parts_mut(table.base.add(start), len) };
            compute(start, outputs);
        };
        self.dispatch(&job);
    }

    fn dispatch_rows_work_stealing<F>(&self, result: &mut [f32], align: usize, compute: &F)
    where
        F: Fn(usize, &mut [f32]) + Sync,
    {
        let segments = self.work_stealing_segments_aligned(result.len(), align);
        let table = RowTable {
            base: result.as_mut_ptr(),
            segments: &segments,
        };
        let table = &table;
        #[cfg(feature = "mlas")]
        if let Some(pool) = &self.work_stealing_pool {
            pool.parallel_for(0, segments.len(), 1, |begin, end| {
                for task_index in begin..end {
                    let (start, len) = table.segments[task_index];
                    if len == 0 {
                        continue;
                    }
                    // SAFETY: each dynamic tile is a disjoint, in-bounds row
                    // range, and the work-stealing pool hands every tile chunk
                    // to exactly one worker.
                    let outputs =
                        unsafe { std::slice::from_raw_parts_mut(table.base.add(start), len) };
                    compute(start, outputs);
                }
            });
            return;
        }
        let next = AtomicUsize::new(0);
        let next = &next;
        let job = move |_global_index: usize| loop {
            let task_index = next.fetch_add(1, Ordering::Relaxed);
            let Some(&(start, len)) = table.segments.get(task_index) else {
                break;
            };
            if len == 0 {
                continue;
            }
            // SAFETY: each dynamic tile is a disjoint, in-bounds row range, and
            // the atomic cursor hands every tile to exactly one worker.
            let outputs = unsafe { std::slice::from_raw_parts_mut(table.base.add(start), len) };
            compute(start, outputs);
        };
        self.dispatch(&job);
    }

    /// Copy `src` into a fresh buffer whose per-node row shards are first-touched
    /// on their owning node, so each worker later streams node-local memory.
    ///
    /// `src` is a row-major `[n, stride]` weight component; the row split matches
    /// [`Self::worker_row_segments`] exactly so it lines up with dispatch.
    pub fn place_rows<T: Copy + Send + Sync>(&self, src: &[T], n: usize) -> Vec<T> {
        if n == 0 || src.is_empty() || self.total_workers <= 1 {
            return src.to_vec();
        }
        let stride = src.len() / n;
        debug_assert_eq!(stride * n, src.len());
        let mut dst: Vec<T> = Vec::with_capacity(src.len());
        // Leave the buffer uninitialized on purpose: zero-filling here would
        // fault every page onto the dispatcher's node, defeating the node-local
        // first-touch performed by the pinned workers below.
        // SAFETY: `T: Copy` has no `Drop`, capacity is exactly `src.len()`, and
        // every element is overwritten by the per-worker `copy_from_slice`
        // (`worker_row_segments` covers `0..n` exactly once) before the buffer is
        // read.
        #[allow(clippy::uninit_vec)]
        unsafe {
            dst.set_len(src.len());
        }
        let segments = self.worker_row_segments(n);
        let table = CopyTable {
            dst: dst.as_mut_ptr(),
            src: src.as_ptr(),
            stride,
            segments: &segments,
        };
        // Capture the whole `CopyTable` (manual `Sync`) rather than its raw
        // pointer fields individually.
        let table = &table;
        let job = move |global_index: usize| {
            let (start, len) = table.segments[global_index];
            if len == 0 {
                return;
            }
            // SAFETY: disjoint, in-bounds `[start, start+len)` row ranges (in
            // units of `stride`), covering every row exactly once; the pinned
            // worker's write faults these destination pages onto its own node.
            unsafe {
                let dst = table.dst.add(start * table.stride);
                let src = table.src.add(start * table.stride);
                std::ptr::copy_nonoverlapping(src, dst, len * table.stride);
            }
        };
        self.dispatch(&job);
        dst
    }
}

impl Drop for SpmdDecodePools {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Row-major output view handed to a dispatched compute job. Disjoint per-worker
/// segments make the raw base pointer safe to share.
struct RowTable<'a> {
    base: *mut f32,
    segments: &'a [(usize, usize)],
}
// SAFETY: each global worker index reads only its own disjoint row segment.
unsafe impl Sync for RowTable<'_> {}

/// Output view for fixed-width row-block dispatch: `base` is a `[num_rows,
/// row_len]` row-major buffer, and each worker owns the disjoint row range
/// `segments[worker]`.
struct RowBlockTable<'a> {
    base: *mut f32,
    row_len: usize,
    segments: &'a [(usize, usize)],
}
// SAFETY: each global worker index writes only its own disjoint row range.
unsafe impl Sync for RowBlockTable<'_> {}

/// Source/destination view for node-local weight placement.
struct CopyTable<'a, T> {
    dst: *mut T,
    src: *const T,
    stride: usize,
    segments: &'a [(usize, usize)],
}
// SAFETY: each worker copies only its own disjoint row range.
unsafe impl<T: Send + Sync> Sync for CopyTable<'_, T> {}

/// Ensures a worker always acknowledges the current op while making the normal
/// path no more expensive than the existing atomic decrement. `complete`
/// forgets the guard after decrementing; only unwinding executes `Drop`, which
/// poisons the pool before decrementing so the dispatcher cannot miss the panic.
struct WorkerCompletion<'a> {
    shared: &'a SharedState,
    node: usize,
    global_index: usize,
}

impl WorkerCompletion<'_> {
    fn complete(self) {
        let shared = self.shared;
        let node = self.node;
        let global_index = self.global_index;
        std::mem::forget(self);
        if shared.node_pending[node].0.fetch_sub(1, Ordering::AcqRel) == 1 {
            // This worker retired the node's final shard, so it is the one the
            // dispatcher was still waiting on. Attributed here rather than
            // derived from timing because the barrier already knows.
            if let Some(counters) = shared.worker_counters.get(global_index) {
                counters.0.last_arrivals.fetch_add(1, Ordering::Relaxed);
            }
            shared.signal_completion();
        }
    }
}

impl Drop for WorkerCompletion<'_> {
    fn drop(&mut self) {
        self.shared
            .poisoned_worker
            .compare_exchange(
                0,
                self.global_index + 1,
                Ordering::Release,
                Ordering::Relaxed,
            )
            .ok();
        if self.shared.node_pending[self.node]
            .0
            .fetch_sub(1, Ordering::AcqRel)
            == 1
        {
            // Counted on this path too, so `sum(last_arrivals) == dispatches *
            // node_count` holds unconditionally rather than "except when a
            // worker panicked". An identity with an exception is not an
            // identity, and a harness that subtracts snapshots has no way to
            // learn the exception applied.
            if let Some(counters) = self.shared.worker_counters.get(self.global_index) {
                counters.0.last_arrivals.fetch_add(1, Ordering::Relaxed);
            }
            // A panicking worker still has to release a parked dispatcher, or
            // the poisoned pool would hang instead of reporting.
            self.shared.signal_completion();
        }
    }
}

/// The persistent worker main loop: wait for a published op, run this worker's
/// shard, acknowledge, repeat until shutdown.
fn worker_loop(shared: Arc<SharedState>, global_index: usize) {
    /// Counts this worker out however it leaves -- including a panic unwind, so
    /// a worker that dies mid-shard does not leave the pool permanently looking
    /// as though it is still running one.
    struct ExitCount<'a>(&'a SharedState);
    impl Drop for ExitCount<'_> {
        fn drop(&mut self) {
            self.0.workers_exited.fetch_add(1, Ordering::Release);
        }
    }
    let _exit_count = ExitCount(&shared);
    let node = shared.worker_node[global_index];
    // Track this node's sense line: 0 until the first op is published. Announce
    // readiness only after establishing the baseline; the dispatcher blocks in
    // `build` until every worker has done this, so no op can be published before
    // this worker is waiting for it.
    let mut last_seen: u32 = 0;
    let mut last_op: u32 = 0;
    let blocktime = decode_blocktime();
    // Hoisted so the hot loop pays a local branch rather than a shared load,
    // and so a disabled pool never touches the timing path at all.
    let profile = shared.profile_workers;
    // The `Release` half is load-bearing: it orders this worker's pre-readiness
    // stores -- notably `pin_failed[i]`, which the builder reads `Relaxed`
    // after the barrier -- before the builder's `Acquire` load of `ready`.
    //
    // `Release` alone would suffice. A release sequence headed by a release
    // store extends through every subsequent read-modify-write on that
    // location whatever ordering those RMWs use, and every announcement here
    // is an RMW; so the final `fetch_add` that brings `ready` to
    // `total_threads` lies in the release sequence headed by *each* worker's
    // store, and the builder's `Acquire` load of that value synchronizes-with
    // all of them. The `Acquire` half is therefore not required -- no worker
    // reads another worker's pre-readiness state -- and is kept only as a
    // harmless superset. Do not read it as "Release would be a race": it would
    // not be. What must not change is that this stays a *release* operation
    // and stays sequenced after the `pin_failed` store.
    shared.ready.fetch_add(1, Ordering::AcqRel);
    loop {
        // Bounded active spin (blocktime) then futex park; returns the observed
        // sense, or an unchanged value if shutdown was seen (re-checked below).
        last_seen = shared.worker_wait(node, global_index, last_seen, blocktime);
        // Whether there is an op to run is asked of the node's `ops` word, not
        // of its wake word and not of the shutdown flag. Shutdown advances the
        // wake word to break parked workers out, so an advance does not imply work;
        // and a dispatch that lands just before a concurrent shutdown *is* work,
        // even though the flag is set by the time this worker looks. Answering
        // from the flag drops that shard and strands the dispatcher, which since
        // it parks rather than spins is a silent hang at zero CPU.
        let op = shared.node_sense[node].0.ops.load(Ordering::Acquire);
        if op != last_op {
            last_op = op;
            // Timed only when the op is actually run, so `timed_ops` counts ops
            // and not observations (a shutdown bump advances `wake` but not
            // `ops`, and must not be charged as a dispatch).
            let observed_ns = if profile {
                let published = shared.node_sense[node].0.publish_ns.load(Ordering::Relaxed);
                let now = profile_now_ns();
                if let Some(counters) = shared.worker_counters.get(global_index) {
                    counters
                        .0
                        .wake_ns
                        .fetch_add(now.saturating_sub(published), Ordering::Relaxed);
                }
                now
            } else {
                0
            };
            // Read and run the published op. The acquire above, paired with the
            // Release bump in `publish`, established visibility of the job
            // pointer and the pending counts.
            // SAFETY: the dispatcher keeps the pointee alive until every node
            // counter reaches zero, i.e. until after this worker acknowledges
            // below -- including when it is shutting down, because `wait` runs
            // before `shutdown` can join this thread.
            let job = unsafe { (*shared.job.get()).expect("published SPMD job") };
            let completion = WorkerCompletion {
                shared: &shared,
                node,
                global_index,
            };
            // SAFETY: `dispatch` keeps the closure alive until this worker
            // acknowledges through `completion`.
            unsafe { (job.call)(job.data, global_index) };
            // Before `complete`, so the shard time excludes the barrier
            // acknowledgement and the wake syscall it may issue -- this measures
            // the shard, not the protocol around it.
            if profile && let Some(counters) = shared.worker_counters.get(global_index) {
                counters.0.work_ns.fetch_add(
                    profile_now_ns().saturating_sub(observed_ns),
                    Ordering::Relaxed,
                );
                counters.0.timed_ops.fetch_add(1, Ordering::Relaxed);
            }
            completion.complete();
        }
        // Checked *after* retiring any outstanding shard, so a dispatch racing
        // shutdown is completed rather than abandoned.
        if shared.shutdown.load(Ordering::Acquire) {
            return;
        }
    }
}

/// The process-wide persistent SPMD layout. `None` (once initialized) means the
/// mode opted out or the safe auto-enable gate declined. Held at module scope so
/// [`shutdown_pools`] can *peek* (join the workers if the pool was ever built)
/// without forcing a build.
static POOLS: OnceLock<Option<SpmdDecodePools>> = OnceLock::new();

/// The label describing the active decode path, set once at pool build time.
/// Queryable via [`decode_path_label`] so callers can inspect which strategy is
/// active without parsing stderr or comparing throughput numbers.
static DECODE_PATH_LABEL: OnceLock<&'static str> = OnceLock::new();

/// Human-readable label for the decode path that was selected. Returns the
/// path name once the pool initialization has run, or `"unresolved"` if it
/// has not been queried yet.
///
/// Examples:
/// - `"spmd-pool"` — the persistent SPMD pool (default or `=1`)
/// - `"adaptive"` — load-adaptive calibration (`=auto`)
/// - `"flat"` — the flat legacy path (`=0` or fallback)
/// - `"unresolved"` — pool initialization has not run yet
pub fn decode_path_label() -> &'static str {
    DECODE_PATH_LABEL.get().copied().unwrap_or("unresolved")
}

/// The width a decode request asked for, what it actually got, and which path
/// served it.
///
/// Three separate mechanisms can quietly hand back fewer compute lanes than
/// were requested: the pre-clamp to `available_parallelism` in
/// `resolve_persistent_decode_threads_with_override`,
/// [`reserve_split_headroom`], and the single-CPU-cpuset fallback in
/// [`build_from_env`]. Only the last reports through [`report_spmd_fallback`],
/// which is itself `tracing::debug!` or `NXRT_CALIB_DEBUG`-gated; the other two
/// log nothing at all. In a default benchmark run none of them are visible, so a
/// `t=N` row in a results table is a *label* rather than a measurement of width
/// `N`. This is the read that turns it back into a measurement.
///
/// [`reserve_single_group_headroom`] is deliberately *not* in that list. It does
/// reduce the number of spawned threads, but it only runs in the single-group
/// case, which is exactly the case where `dispatcher_owns_a_shard` is true, so
/// the lane it takes is added straight back by the dispatcher's own shard. A
/// 2-lane budget on a 2-CPU cpuset spawns one thread and still reports
/// `realized = 2`, measured, because two lanes really do compute.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeWidth {
    /// Compute lanes asked for: the explicit `ONNX_GENAI_CPU_DECODE_THREADS`
    /// budget, or the default physical-core budget when unset. `None` means
    /// either the width was never resolved (no decode has run yet) or the
    /// caller opted out with `=0`, which is what `path` disambiguates.
    pub requested: Option<usize>,
    /// Compute lanes actually available. `None` until a decode path is chosen.
    ///
    /// On the persistent pool this is exact and includes the dispatcher's own
    /// shard. On the flat path there is no fixed layout to report, so it is the
    /// width of the Rayon pool **the calling thread is on** -- meaningful inside
    /// `with_decode_pool_scope`, and the global width outside it. Same
    /// convention as [`crate::kernels::matmul_nbits::active_decode_worker_count`].
    ///
    /// This counts **compute participants**, which is the number a throughput
    /// row should be labelled with. It is *not* the number of threads: on the
    /// persistent pool one participant is the dispatching thread running its
    /// shard inline, so the joinable-thread count is one lower whenever the
    /// group is fully subscribed. Anything reading `/proc`, attributing thread
    /// counts or joining at teardown wants
    /// [`SpmdDecodePools::spawned_workers`] instead. Conflating the two is a
    /// live hazard rather than a hypothetical one -- it produced an off-by-one
    /// that passed on a 32-CPU host and would have failed every run on a
    /// 2-core runner.
    pub realized: Option<usize>,
    /// The selected path, as [`decode_path_label`] reports it.
    pub path: &'static str,
}

impl DecodeWidth {
    /// Whether the realized width matches what was asked for.
    ///
    /// `false` whenever a request was silently reduced, and also whenever the
    /// width has not been resolved yet -- a harness that asserts on this cannot
    /// accidentally pass by querying too early.
    pub fn is_as_requested(&self) -> bool {
        match (self.requested, self.realized) {
            (Some(requested), Some(realized)) => requested == realized,
            _ => false,
        }
    }
}

/// Report the requested vs realized decode width and the path that served it.
///
/// Deliberately **does not** build the pool: it reads the already-initialized
/// statics, so calling it cannot change which path a process takes or when it is
/// chosen. A harness should run at least one decode step first and then assert;
/// before that the fields read `None` / `"unresolved"` and
/// [`DecodeWidth::is_as_requested`] is `false`.
pub fn decode_width() -> DecodeWidth {
    let requested = REQUESTED_WIDTH.get().copied().flatten();
    let realized = match POOLS.get() {
        Some(Some(pools)) => Some(pools.total_workers()),
        // The pool was resolved and declined: decode is on the flat path, whose
        // width is whatever Rayon pool the caller is standing in.
        Some(None) => Some(rayon::current_num_threads()),
        None => None,
    };
    DecodeWidth {
        requested,
        realized,
        path: decode_path_label(),
    }
}

/// Snapshot the persistent decode pool's scheduling counters, or `None` when no
/// persistent pool exists (never built, or the flat path was chosen).
///
/// Like [`decode_width`], this **peeks** rather than forcing: reading counters
/// must not be what decides a process's decode path, or the instrument changes
/// the thing it measures. A harness runs at least one decode step first.
pub fn counters() -> Option<SpmdCounters> {
    match POOLS.get() {
        Some(Some(pools)) => Some(pools.counters()),
        _ => None,
    }
}

/// Snapshot the persistent decode pool's per-worker barrier attribution, or
/// `None` when no persistent pool exists.
///
/// Peeks, never forces, for the same reason [`counters`] does. See
/// [`SpmdWorkerProfile`] for what is always collected and what needs
/// [`WORKER_PROFILE_ENV`].
pub fn worker_profiles() -> Option<Vec<SpmdWorkerProfile>> {
    match POOLS.get() {
        Some(Some(pools)) => Some(pools.worker_profiles()),
        _ => None,
    }
}

/// The active-spin window a decode worker holds a core before parking, as the
/// running process resolved it.
///
/// `decode_blocktime` latches into a `OnceLock` on the first `worker_wait`, so
/// a sweep over `ONNX_GENAI_CPU_DECODE_BLOCKTIME_US` has to be *across process
/// launches*; setting the variable a second time inside one process changes
/// nothing and the two arms silently measure the same window (#1736's shape). A
/// harness therefore has to report the window it actually ran with, and it has
/// to read it from here rather than re-parse the environment itself -- a second
/// parser is a second implementation that can drift from the one the workers
/// obey, and would keep printing the requested value after the real policy
/// changed.
///
/// Reading this *does* latch the window if nothing has yet, which is harmless
/// (the value is a pure function of the environment) but means a harness should
/// still call it at report time, after the decode it describes.
pub fn blocktime() -> Duration {
    decode_blocktime()
}

/// The width the caller asked for, recorded when the pool is resolved and before
/// any of the reductions that can shrink it.
///
/// When an explicit request exists this is the *unclamped*
/// [`crate::kernels::matmul_nbits::decode_thread_budget`], not the resolved
/// width `build_from_env` receives. The resolver already clamps to
/// `available_parallelism`, so recording its output would report a request of 8
/// inside a 2-CPU cpuset as `2 requested, 2 realized` -- satisfied. That is
/// exactly the silent mislabelling this read exists to expose, so the comparison
/// has to be against what the user actually asked for.
///
/// With no explicit request this falls back to the resolved default width, which
/// under default policy *is* the intended width -- there is no request for it to
/// disagree with, so [`DecodeWidth::is_as_requested`] is meaningful on a
/// default-configured run rather than trivially `false`.
///
/// `None` only while the pool is still unresolved, or when the caller opted out
/// with `=0` (which never reaches the latch in [`pools`]).
static REQUESTED_WIDTH: OnceLock<Option<usize>> = OnceLock::new();

/// The lazily built persistent SPMD layout, or `None` when the mode is opted out
/// or the safe auto-enable gate declines. Built once and reused for the whole
/// process.
pub fn pools() -> Option<&'static SpmdDecodePools> {
    POOLS
        .get_or_init(|| {
            let threads = default_threads();
            // Latched here rather than inside `build_from_env` so the recorded
            // request can only ever describe the pool that `POOLS` actually
            // holds. `build_from_env` is `pub`; latching there would let a direct
            // call record a width that no realized pool matches -- reintroducing
            // the exact "the t=N label is a lie" failure this read exists to
            // catch, through the front door.
            //
            // Prefer the unclamped `decode_thread_budget()`, falling back to
            // `threads` only when nobody asked for a specific width. Under
            // default policy the clamped width *is* the intended width, so there
            // is no request for it to disagree with.
            REQUESTED_WIDTH
                .get_or_init(|| crate::kernels::matmul_nbits::decode_thread_budget().or(threads));
            build_from_env(threads)
        })
        .as_ref()
}

/// Signal the persistent pool's workers to stop and **join** them, if the pool
/// was ever built. Idempotent and safe to call when the pool was never built (it
/// only inspects the already-initialized static; it never forces a build).
///
/// The persistent pool lives in a module-level `static`, and Rust does **not**
/// run `Drop` on statics at process exit. Without an explicit join the pool's
/// hot worker threads stay alive (spinning or parked on `Arc<SharedState>`)
/// while the process tears down its runtime. On weakly-ordered targets (notably
/// native Windows ARM64) a worker still touching that shared state during
/// teardown can fault the whole process with `STATUS_ACCESS_VIOLATION`
/// (0xC0000005) and an empty stderr -- not a Rust panic. Calling this before the
/// process exits makes teardown deterministic: workers observe the stop flag,
/// return, and are joined, so none is live during runtime teardown.
///
/// This is a teardown-only operation: it takes no locks on the decode hot path
/// and must not be called between decode ops (it stops the workers).
pub fn shutdown_pools() {
    if let Some(Some(pool)) = POOLS.get() {
        pool.shutdown();
    }
}

/// Resolve the persistent pool's worker count. Honors `ONNX_GENAI_CPU_DECODE_THREADS`
/// when set (`0` opts out); when unset it uses the persistent-specific default
/// (one worker per allowed physical core), *not* the flat pool's eight-worker
/// ceiling -- see
/// [`crate::kernels::matmul_nbits::configured_persistent_decode_threads`].
fn default_threads() -> Option<usize> {
    crate::kernels::matmul_nbits::configured_persistent_decode_threads()
}

/// How the persistent pool was selected, parsed from `PERSISTENT_POOL_ENV`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PersistenceMode {
    /// `=0`: explicit opt-out; the decode path stays on the flat legacy pool.
    Off,
    /// Unset (or `=1`, or an unrecognized value): the default. The persistent
    /// SPMD pool is always used, deterministically -- no host probing.
    On,
    /// `=auto`: opt-in load-adaptive calibration. The pool is built but the
    /// [`Calibrator`] times the live decode step both ways and keeps the faster
    /// path. Under co-tenant load the flat path usually wins.
    Adaptive,
}

/// Parse the persistence mode from the raw env value (`None` = unset). Unset,
/// `=1`, and any unrecognized value map to `On` (the deterministic pool
/// default); `=0` is the explicit opt-out; `=auto` enables load-adaptive
/// calibration.
pub(crate) fn persistence_mode_from_raw(raw: Option<&str>) -> PersistenceMode {
    match raw.map(str::trim) {
        Some("0") => PersistenceMode::Off,
        Some(v) if v.eq_ignore_ascii_case("auto") => PersistenceMode::Adaptive,
        _ => PersistenceMode::On,
    }
}

/// Default when [`PERSISTENT_POOL_ENV`] is unset. `true` (the historical
/// default) builds the persistent SPMD pool; a host that never enters an SPMD
/// decode scope sets this to `false`.
static PERSISTENT_POOL_DEFAULT: AtomicBool = AtomicBool::new(true);

/// Opt this process out of (or back into) the persistent SPMD decode pool when
/// [`PERSISTENT_POOL_ENV`] is unset.
///
/// The pool only pays off for a host that drives decode *inside* an SPMD scope
/// (`onnx-genai-engine`'s native decode loop): resident workers, one barrier
/// per projection, no fork/join. A host that never enters such a scope -- the
/// plugin EP, where ONNX Runtime owns the graph and the threads -- gets only
/// the costs: resident workers competing with ORT's own intra-op pool, and an
/// MLAS weight partitioned into one shard per persistent decode worker
/// (`available_parallelism() / 2` by default), which caps an unscoped decode
/// GEMV at that worker count however many threads the host has.
///
/// Measured on the plugin path (32 vCPU AMD EPYC 9V74, MLAS build, int4
/// block-32, K=N=2048, M=1, p50 of 41 runs, each backend alone in its process):
/// 0.376 ms with the pool built against 0.092 ms without it, with ONNX
/// Runtime's own CPU EP at 0.097 ms.
///
/// An explicit `PERSISTENT_POOL_ENV` setting always wins. Must be called before
/// the first [`pools()`] query; afterwards the built layout is fixed for the
/// process. `Relaxed` is sufficient because the only supported caller writes
/// this during library initialization, before the host has created a session or
/// spawned the threads that read it -- that handoff is the synchronisation.
pub fn set_persistent_decode_pool_default(enabled: bool) {
    PERSISTENT_POOL_DEFAULT.store(enabled, Ordering::Relaxed);
}

/// Resolve the persistence mode from the raw env value and the process default.
/// An explicit setting always wins; the default only decides the unset case.
/// Pure so the precedence is unit-tested without env races.
pub(crate) fn resolve_persistence_mode(
    raw: Option<&str>,
    default_enabled: bool,
) -> PersistenceMode {
    match raw {
        Some(value) => persistence_mode_from_raw(Some(value)),
        None if !default_enabled => PersistenceMode::Off,
        None => persistence_mode_from_raw(None),
    }
}

fn persistence_mode() -> PersistenceMode {
    resolve_persistence_mode(
        std::env::var(PERSISTENT_POOL_ENV).ok().as_deref(),
        PERSISTENT_POOL_DEFAULT.load(Ordering::Relaxed),
    )
}

/// Whether a persistence mode **builds** the persistent SPMD pool. Both `On`
/// (the default or `=1`) and `Adaptive` (`=auto`) build it: `On` always
/// dispatches to it, and `Adaptive` needs it available so calibration can time
/// the real workload on it. Only `Off` (`=0`) never builds it. Pure so the
/// gating is unit-tested without env races.
fn pool_mode_builds(mode: PersistenceMode) -> bool {
    matches!(mode, PersistenceMode::On | PersistenceMode::Adaptive)
}

/// Whether a persistence mode **unconditionally** dispatches to the pool (no
/// calibration): `On` (the default or `=1`). `Adaptive` builds the pool but
/// lets the [`Calibrator`] pick per step; `Off` never uses it.
fn pool_mode_forces(mode: PersistenceMode) -> bool {
    matches!(mode, PersistenceMode::On)
}

/// Whether the persistent pool is unconditionally dispatched to (the default,
/// or `PERSISTENT_POOL=1`). Used to keep the `numa-split` mutual-exclusion
/// diagnostic scoped to users who actually asked for the persistent pool, to
/// make dense-f32 decode still eligible for the pool, and to skip calibration.
pub(crate) fn is_forced() -> bool {
    pool_mode_forces(persistence_mode())
}

/// Build the persistent SPMD layout when the mode builds it (`On` -- the default
/// or `=1` -- and `Adaptive` `=auto`); `=0` (Off) or `THREADS=0` return `None`
/// so decode stays on the flat path. Under `Adaptive` the pool is built but only
/// *used* when calibration adopts it (see [`Calibrator`]); under `On` it is
/// always used.
///
/// Two or more usable NUMA nodes yield the two-level node-pinned layout; a
/// single-node host, a non-NUMA machine, or a platform without pinning yields a
/// single unpinned worker group (still the lightweight barrier, still correct).
pub fn build_from_env(threads: Option<usize>) -> Option<SpmdDecodePools> {
    // Build for `On` (the default or `=1`) and `Adaptive` (`=auto`). `On` always
    // dispatches to the pool; `Adaptive` needs the pool available so the
    // calibrator can time the live decode step on it. `Off` (`=0`) and `THREADS=0`
    // leave decode on the flat Rayon path. See `PERSISTENT_POOL_ENV`.
    let mode = persistence_mode();
    if !pool_mode_builds(mode) {
        // An explicit `=0` opt-out is a decision, not an unresolved state: decode
        // is definitively on the flat path. Without this the label reads
        // "unresolved" forever and a harness cannot tell an opt-out apart from a
        // process that has not decoded yet.
        DECODE_PATH_LABEL.get_or_init(|| "flat");
        return None;
    }
    // Adaptive defers to an explicit decode-affinity request: if the user set
    // `ONNX_GENAI_CPU_DECODE_AFFINITY` (numa-split, compact, node:N, off, ...),
    // they picked a specific strategy, so Adaptive does not build/calibrate the
    // persistent pool and lets that request drive decode (numa-split via
    // `numa_pools`, everything else via the flat path + `plan_decode_affinity`).
    // `On` still builds the pool and keeps its documented precedence.
    if matches!(mode, PersistenceMode::Adaptive) && explicit_decode_affinity_set() {
        // Also a decision: the user's affinity request drives decode via
        // `numa_pools` or the flat path, so record that rather than leaving the
        // label unresolved.
        DECODE_PATH_LABEL.get_or_init(|| "flat");
        return None;
    }
    let Some(total) = threads else {
        report_spmd_fallback(
            "ONNX_GENAI_CPU_DECODE_THREADS=0 opts out of the bounded pool; the persistent \
             SPMD pool needs a bounded worker count -- leaving the decode path unchanged",
        );
        return None;
    };
    if total == 0 {
        DECODE_PATH_LABEL.get_or_init(|| "flat");
        return None;
    }
    // Cpuset-aware dispatcher headroom: when the requested spinning-worker count
    // would occupy *every* CPU the process is allowed to run on, a single
    // allowed CPU leaves no core for the inline dispatcher (the engine thread
    // that publishes each op and spins on the per-node completion counters).
    // With N spinning workers pinned across all N allowed CPUs the dispatcher is
    // migrated onto a busy worker core and can no longer publish jobs / read
    // counters promptly, collapsing throughput ~20-60x (measured 1.47 tok/s at
    // 32 workers on `taskset -c 0-31` vs ~29 tok/s once one CPU is spare). If the
    // whole allowed cpuset is a single CPU there is no core to reserve, so the
    // pool cannot run without starving itself -- fall back to the flat path.
    if let Some(allowed) = crate::decode_affinity::allowed_cpus()
        && allowed.len() == 1
    {
        report_spmd_fallback(
            "the process is confined to a single CPU (cpuset/taskset), which leaves no core \
             for the inline dispatcher alongside a spinning worker -- leaving decode on the \
             flat path instead of starving the persistent SPMD pool",
        );
        return None;
    }
    report_pool_built(mode);
    let shards = node_shards(total);
    // The dispatcher computes a shard exactly when the headroom reservation took
    // a worker away to keep its CPU free. Then `total - 1` pinned threads plus
    // the dispatcher give `total` compute lanes on `total` CPUs -- the requested
    // width, with the anti-starvation reservation still in force.
    //
    // This is the common case, not a corner case: `bound_process_to_decode_budget`
    // confines the process to exactly `total` CPUs at EP initialization, so an
    // explicit budget *always* arrives here fully subscribed. Without the
    // dispatcher shard, `ONNX_GENAI_CPU_DECODE_THREADS=N` silently buys `N-1`
    // lanes -- and at `N=2` that is one lane, which trips the `total_workers <= 1`
    // serial short-circuit and makes the knob indistinguishable from `=1`.
    //
    // Restricted to the single-group layout. On a NUMA split the dispatcher's
    // node is not known at build time, so handing it a shard could pull that
    // shard's weights across sockets; those layouts keep the previous behavior.
    Some(SpmdDecodePools::build(
        &shards,
        dispatcher_owns_a_shard(&shards, total),
    ))
}

/// Whether the inline dispatcher should own a compute shard.
///
/// True exactly when the single-group headroom reservation took a worker away
/// to keep a CPU free for the dispatcher: giving that CPU back a compute lane
/// restores the requested width without re-pinning a worker to every core. A
/// group that already had headroom is left alone, or the pool would be one lane
/// wider than the user asked for.
fn dispatcher_owns_a_shard(shards: &[NodeShard], requested: usize) -> bool {
    shards.len() == 1 && shards[0].workers < requested
}

/// Whether `ONNX_GENAI_CPU_DECODE_AFFINITY` is set to a non-empty value.
/// Adaptive calibration only engages when it is unset, so an explicit affinity
/// request is honored on the flat/numa path exactly as before.
fn explicit_decode_affinity_set() -> bool {
    std::env::var(crate::decode_affinity::DECODE_AFFINITY_ENV)
        .ok()
        .is_some_and(|value| !value.trim().is_empty())
}

/// What an explicit [`DECODE_AFFINITY_ENV`] request means for the persistent
/// pool's CPU set.
///
/// Kept pure and separate from topology so the precedence is unit-tested
/// without env races or a particular host, in the same style as
/// [`persistence_mode_from_raw`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExplicitAffinity {
    /// Unset, empty, or `numa-split`: the existing default placement already
    /// implements it, so nothing is overridden.
    DeferToDefault,
    /// `off`: the user asked for unpinned workers, so the pool must not pin.
    Unpinned,
    /// `compact` / `node:<index>`: the CPU set comes from the shared planner,
    /// which owns topology detection and the allowed-set intersection.
    FromPlan,
    /// A value that is not a selector at all. Placement defers exactly as for
    /// `DeferToDefault`, but the caller reports it: a typo is precisely the
    /// case a user needs told about, and silence here is how this knob went
    /// unnoticed for so long.
    Malformed,
}

/// Map a raw [`DECODE_AFFINITY_ENV`] value to its meaning for this path.
///
/// `numa-split` defers because [`node_shards`]'s multi-node branch *is* the
/// numa-split layout; deferring keeps one implementation rather than two that
/// can drift.
///
/// An unparseable value defers too, rather than being rejected: turning a bad
/// env var into "decode silently unpinned" would be worse than ignoring it.
/// It is reported as [`ExplicitAffinity::Malformed`] rather than folded into
/// `DeferToDefault` so the caller can still say so out loud.
fn explicit_affinity_request(raw: Option<&str>) -> ExplicitAffinity {
    let Some(raw) = raw.map(str::trim).filter(|value| !value.is_empty()) else {
        return ExplicitAffinity::DeferToDefault;
    };
    match crate::decode_affinity::DecodeAffinity::parse(Some(raw)) {
        Ok(crate::decode_affinity::DecodeAffinity::Off) => ExplicitAffinity::Unpinned,
        Ok(crate::decode_affinity::DecodeAffinity::Compact)
        | Ok(crate::decode_affinity::DecodeAffinity::Node(_)) => ExplicitAffinity::FromPlan,
        Ok(crate::decode_affinity::DecodeAffinity::NumaSplit) => ExplicitAffinity::DeferToDefault,
        Err(_) => ExplicitAffinity::Malformed,
    }
}

/// Resolve the shards an explicit affinity request asks for, or `None` to leave
/// the default placement in charge.
///
/// This is the fix for the case where `ONNX_GENAI_CPU_DECODE_AFFINITY` was
/// parsed for nothing on the default pool: the worker CPUs came straight from
/// `sched_getaffinity`, so `off`, `compact` and `node:<index>` all produced
/// byte-identical placement. Only the *CPU set* is decided here -- how workers
/// are then laid out across it stays with the placement policy, so the two
/// concerns do not have to know about each other.
fn explicit_affinity_shards(total: usize) -> Option<Vec<NodeShard>> {
    let raw = std::env::var(crate::decode_affinity::DECODE_AFFINITY_ENV).ok();
    let request = explicit_affinity_request(raw.as_deref());
    if request == ExplicitAffinity::Malformed {
        // Re-parse purely for the message, so the accepted-modes menu stays
        // owned by `decode_affinity` rather than restated here.
        if let Err(message) =
            crate::decode_affinity::DecodeAffinity::parse(raw.as_deref().map(str::trim))
        {
            crate::kernels::matmul_nbits::report_decode_affinity_policy(&format!(
                "{message}; the persistent decode pool is using default placement instead"
            ));
        }
    }
    explicit_affinity_shards_for(
        request,
        total,
        || match crate::decode_affinity::plan_decode_affinity(total) {
            Ok(plan) => {
                if let Some(message) = plan.log {
                    crate::kernels::matmul_nbits::report_decode_affinity_policy(&message);
                }
                plan.cpus
            }
            // The flat path propagates this as a hard error. Here the request is
            // still reported -- deferring silently is how this knob went inert
            // in the first place -- but the pool falls back to default placement
            // rather than refusing to start decode over a placement preference.
            Err(message) => {
                crate::kernels::matmul_nbits::report_decode_affinity_policy(&format!(
                    "{message}; the persistent decode pool is using default placement instead"
                ));
                None
            }
        },
        crate::decode_affinity::allowed_cpus,
    )
}

/// The shard decision itself, with topology supplied by the caller.
///
/// Split from [`explicit_affinity_shards`] for the reason given on
/// [`parse_decode_blocktime`]: the env read is the only impure part, so keeping
/// it out means the precedence can be tested exhaustively without `set_var` and
/// without depending on the test host's NUMA layout.
fn explicit_affinity_shards_for(
    request: ExplicitAffinity,
    total: usize,
    planned_cpus: impl FnOnce() -> Option<Vec<usize>>,
    allowed_cpus: impl FnOnce() -> Option<Vec<usize>>,
) -> Option<Vec<NodeShard>> {
    match request {
        ExplicitAffinity::DeferToDefault | ExplicitAffinity::Malformed => None,
        // An empty CPU list is how `SpmdDecodePools::build_with_schedule` reads
        // "do not pin": it resolves `cpus.get(worker % len.max(1))` to `None`
        // and skips the pin, keeping every worker.
        //
        // The worker *count* is still reserved against the allowed set, because
        // `off` means "do not pin", not "also run wider". Dropping the
        // reservation here would add an oversubscription mode that does not
        // exist today: under an external `taskset` of K CPUs with a requested
        // width of K, the pool would run K spinning workers plus the inline
        // dispatcher on K CPUs, and the dispatcher's preemption makes it the
        // straggler the whole barrier waits on. Reserving costs nothing on an
        // unconfined host, where an explicit request already stands the
        // process self-confinement down (`bound_process_to_decode_budget`) and
        // the allowed set is the whole machine.
        ExplicitAffinity::Unpinned => {
            let allowed = allowed_cpus().unwrap_or_default();
            let core_count = crate::core_topology::host()
                .map_or(0, |cores| cores.leaders_within(&allowed).len());
            let workers = reserve_single_group_headroom(total, allowed.len(), core_count);
            Some(vec![NodeShard {
                index: 0,
                cpus: Vec::new(),
                workers,
            }])
        }
        ExplicitAffinity::FromPlan => {
            let cpus = planned_cpus()?;
            if cpus.is_empty() {
                return None;
            }
            // Order through the same spread the default path uses. An explicit
            // request selects *which* CPUs, never how workers are laid out
            // across them; without this, `compact` would pin two workers per
            // physical core and quietly reintroduce the defect #1729 fixed.
            let cores = crate::core_topology::host();
            let cpus = crate::decode_affinity::order_pin_targets(&cpus, cores);
            let core_count = cores.map_or(0, |cores| cores.leaders_within(&cpus).len());
            let workers = reserve_single_group_headroom(total, cpus.len(), core_count);
            Some(vec![NodeShard {
                index: 0,
                cpus,
                workers,
            }])
        }
    }
}

/// Resolve the node shards for `total` workers: the multi-node split when the
/// (cpuset-restricted) topology exposes >=2 nodes, otherwise a single group.
///
/// In both cases the pinned-worker count is capped so the inline dispatcher
/// always keeps at least one allowed CPU free (see [`DISPATCHER_RESERVED_CPUS`]
/// and [`reserve_single_group_headroom`] / [`reserve_split_headroom`]). Without
/// that reservation a fully-subscribed cpuset (workers == allowed CPUs) pins a
/// spinning worker on every core, starving the dispatcher and collapsing
/// throughput.
///
/// An explicit affinity request is resolved first (see
/// [`explicit_affinity_shards`]); it selects the CPU *set*, and the worker
/// count is still capped by the same reservation.
fn node_shards(total: usize) -> Vec<NodeShard> {
    node_shards_with(total, explicit_affinity_shards)
}

/// [`node_shards`] with the explicit-request lookup injected.
///
/// The seam exists so a test can prove the request is *consulted at all*. That
/// is the whole defect in #1792: the helpers below can each be correct while
/// `node_shards` never calls them, and every unit test of the helpers still
/// passes. Deleting the early return has to fail something.
fn node_shards_with(
    total: usize,
    explicit: impl FnOnce(usize) -> Option<Vec<NodeShard>>,
) -> Vec<NodeShard> {
    // An explicit request wins over the default placement. Without this the
    // persistent pool reads the env var for exactly nothing.
    if let Some(shards) = explicit(total) {
        return shards;
    }
    let allowed = crate::decode_affinity::allowed_cpus();
    let cores = crate::core_topology::host();
    if let Some(topology) = NumaTopology::detect() {
        let topology = topology.restrict_to_allowed(allowed.as_deref());
        if let Some(mut shards) = topology.split_workers(total) {
            // Reserve a dispatcher core on every node so the engine thread has a
            // free core on whichever socket the scheduler places it, and each
            // node's completion counter can be read without contending with a
            // pinned spinning worker.
            reserve_split_headroom(&mut shards);
            for shard in shards.iter_mut() {
                shard.cpus = crate::decode_affinity::order_pin_targets(&shard.cpus, cores);
            }
            return shards;
        }
    }
    // Single-node / non-NUMA / no-pinning fallback: one group. Pin to the
    // process's allowed CPUs when known (best-effort), else leave unpinned.
    let cpus = crate::decode_affinity::order_pin_targets(&allowed.unwrap_or_default(), cores);
    let core_count = cores.map_or(0, |cores| cores.leaders_within(&cpus).len());
    let workers = reserve_single_group_headroom(total, cpus.len(), core_count);
    vec![NodeShard {
        index: 0,
        cpus,
        workers,
    }]
}

/// Allowed CPUs kept free for the inline dispatcher per pinned worker group.
///
/// The dispatcher is a single thread, so one spare core is sufficient for it to
/// run; reserving *per node* (see [`reserve_split_headroom`]) rather than once
/// globally guarantees the spare core lands on whichever socket the scheduler
/// places the dispatcher on, and keeps every node's completion-counter reads
/// unblocked on a NUMA-split layout. One is enough because there is exactly one
/// inline dispatcher thread to house, so giving up the single highest-index core
/// costs one worker's share of a bandwidth-bound loop while removing the
/// starvation cliff.
///
/// This used to be justified by the workers plateauing "around half the logical
/// CPUs", which stopped being the sizing rule when the default became one worker
/// per allowed physical core: on a non-SMT host the pool is now deliberately
/// wide enough that this reserve is what keeps it from being *fully*
/// subscribed.
const DISPATCHER_RESERVED_CPUS: usize = 1;

/// Opt-in: bind the inline dispatcher to the CPU [`DISPATCHER_RESERVED_CPUS`]
/// reserved for it (`1`/`on`/`true`/`yes`). Default off.
///
/// The reservation already keeps one allowed CPU clear of workers, because a
/// dispatcher sharing a core with a worker makes that worker a straggler the
/// whole barrier waits on -- 1.57x on qwen int4 at 16 cores, which is why
/// [`reserve_single_group_headroom`] exists. But reserving a CPU does not put
/// anything on it: the dispatcher is an ordinary unpinned thread, and nothing
/// stops the scheduler leaving it on a worker's core while the reserved CPU
/// sits idle. Whether that happens is now measurable rather than assumed --
/// [`SpmdDecodePools::dispatcher_observed_cpu`] samples the dispatcher's actual
/// CPU during the run, and a harness can compare it against
/// [`SpmdDecodePools::dispatcher_cpu`]. If the two diverge, the collision the
/// reserve was built to prevent is happening anyway, non-deterministically per
/// launch, which would also make it a candidate explanation for the
/// launch-to-launch dispersion that swamps width-16 A/B work.
///
/// Off by default deliberately, and for two independent reasons.
///
/// The first is that it has not earned a default. Measured against the
/// pre-registered single-knob rule on `7e274a4e2` -- 16 launches, 15 trusted --
/// it is faster in **15 of 15** launches and the median gain is **1.0953**,
/// under the 1.10 the rule requires: **REJECT**. A companion rule about
/// launch-to-launch dispersion failed its own self-test and certified nothing.
/// An earlier 6-launch run scored 1.1910 and ACCEPT; it did not replicate. The
/// mechanism is also unproven -- the migration counter says the unpinned
/// dispatcher moves at most once per launch, which is far too little to explain
/// anything, so whatever this does is not "it stops migrating". See
/// `docs/benchmarks/2026-08-24-acc0-dispatcher-placement.md`.
///
/// The second is that binding the dispatching thread is not free of
/// consequence: it is the session thread, so it keeps that affinity after the
/// decode loop ends, and a subsequent prefill on the same thread would run
/// one-CPU-wide. Turning this into a default needs evidence that covers prefill
/// as well as decode, and that evidence does not exist yet. The knob is here so
/// the decode half can be measured at all.
pub const DISPATCHER_PIN_ENV: &str = "ONNX_GENAI_CPU_DECODE_DISPATCHER_PIN";

/// Whether [`DISPATCHER_PIN_ENV`] asks for the dispatcher to be pinned.
/// Read once per process, like every other decode knob, so a mid-run
/// environment change cannot make two ops disagree.
pub fn dispatcher_pin_requested() -> bool {
    static REQUESTED: OnceLock<bool> = OnceLock::new();
    *REQUESTED.get_or_init(|| {
        std::env::var(DISPATCHER_PIN_ENV)
            .ok()
            .map(|raw| dispatcher_pin_from_raw(Some(raw.as_str())))
            .unwrap_or(false)
    })
}

/// Parse of [`DISPATCHER_PIN_ENV`], split out so the accepted spellings are
/// directly testable without touching process environment.
fn dispatcher_pin_from_raw(raw: Option<&str>) -> bool {
    matches!(
        raw.map(|value| value.trim().to_ascii_lowercase())
            .as_deref(),
        Some("1" | "on" | "true" | "yes")
    )
}

/// Sample the dispatcher's CPU once every this many dispatches (minus one).
///
/// The dispatcher's *placement* is the quantity under study, and it is not a
/// constant: an unpinned thread migrates, so a single reading taken at the
/// first dispatch describes startup rather than steady state. Sampling costs
/// one `sched_getcpu` -- a vDSO read, no syscall -- per 1024 dispatches, which
/// at ~400 barriers per token is under three tokens' spacing and immaterial
/// beside the barrier itself.
const DISPATCHER_CPU_SAMPLE_MASK: u32 = 1023;

thread_local! {
    /// Per-dispatching-thread tick. Zero means "this thread has not dispatched
    /// before", which drives the one-shot identity record and pin attempt; the
    /// low bits then drive periodic CPU sampling.
    ///
    /// The pin is attempted exactly once per thread, success or failure. A
    /// retry loop would put a failing syscall on the hot path forever on any
    /// host that refuses the pin.
    static DISPATCHER_TICK: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    /// Whether this thread is the one whose id the pool recorded.
    ///
    /// A process can dispatch from more than one thread over its life -- a
    /// session per phase is enough -- and every one of them takes the reserved
    /// CPU, which is the intended behaviour: the point is that whoever is
    /// dispatching sits there. But *placement sampling* has to follow a single
    /// thread or it reports thread changes as movement. Measured before this
    /// distinction existed, a pinned dispatcher read 2 to 7 "moves" when it
    /// could only ever have made one -- the samples were coming from different
    /// threads.
    static DISPATCHER_IS_RECORDED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Cap a single pinned worker group so at least [`DISPATCHER_RESERVED_CPUS`]
/// allowed CPU stays free for the inline dispatcher.
///
/// `allowed_count == 0` means the allowed set is unknown, so workers run
/// unpinned and cannot starve the dispatcher by occupying every core -- the
/// count is returned unchanged. When there is already headroom
/// (`total < allowed_count`) the count is likewise unchanged, so this is a
/// strict no-op unless the group would otherwise be fully subscribed. The result
/// is floored at one worker: even a user who sets
/// `ONNX_GENAI_CPU_DECODE_THREADS=N` on an exactly-N-CPU cpuset gets N-1 pinned
/// workers (never zero); the single-CPU case is handled earlier by falling back
/// to the flat path in [`build_from_env`].
fn reserve_single_group_headroom(total: usize, allowed_count: usize, core_count: usize) -> usize {
    if allowed_count == 0 {
        return total;
    }
    // Within the physical-core budget the pool runs one worker per core, so the
    // inline dispatcher has nowhere to go but some worker's SMT sibling -- and
    // that worker becomes a straggler the whole barrier waits for on every op.
    // Measured on qwen int4 `accuracy_level=0` decode, 16 physical cores, no
    // cpuset: 16 workers 4.41 ms/token against 15 workers 2.81 ms/token (1.57x).
    // Past the core budget the workers already share cores because the user asked
    // for more threads than there are cores, so the historical logical-CPU
    // reserve applies instead -- reserving cores there is a 1.28x regression
    // (8-logical/4-core cpuset: 7 workers 10.58 ms against 3 workers 13.56 ms).
    if core_count > 0 && total <= core_count {
        return total
            .min(core_count.saturating_sub(DISPATCHER_RESERVED_CPUS))
            .max(1);
    }
    if total < allowed_count {
        return total;
    }
    allowed_count
        .saturating_sub(DISPATCHER_RESERVED_CPUS)
        .max(1)
}

/// Cap each NUMA-split shard so at least [`DISPATCHER_RESERVED_CPUS`] CPU of that
/// node stays free for the inline dispatcher. Only nodes that would be fully
/// subscribed (`workers == node CPUs`) are reduced; a node that already has a
/// spare core is left untouched, so this is a no-op wherever headroom exists.
/// Each shard keeps at least one worker.
fn reserve_split_headroom(shards: &mut [NodeShard]) {
    for shard in shards.iter_mut() {
        let cap = shard
            .cpus
            .len()
            .saturating_sub(DISPATCHER_RESERVED_CPUS)
            .max(1);
        shard.workers = shard.workers.min(cap);
    }
}

/// The calling thread's OS thread id, or `None` where the platform exposes no
/// stable per-thread id. Linux only in practice: this exists so a harness can
/// find the dispatcher's `/proc/self/task/<tid>` entry, which has no
/// counterpart elsewhere.
///
/// `None` under Miri as well, for the same reason
/// [`SpmdDecodePools::sample_dispatcher_cpu`] is inert there: there is no
/// `/proc` to look the id up in, so recording one buys nothing and calling the
/// shim risks an unsupported-operation abort in a test that is running for a
/// different reason entirely.
fn current_thread_os_id() -> Option<i64> {
    #[cfg(all(target_os = "linux", not(miri)))]
    {
        // SAFETY: `gettid` takes no arguments, cannot fail, and returns the
        // calling thread's kernel id.
        let tid = unsafe { libc::gettid() };
        (tid > 0).then_some(i64::from(tid))
    }
    #[cfg(not(all(target_os = "linux", not(miri))))]
    {
        None
    }
}

/// Report a realized-placement anomaly once.
///
/// Its own static for the same reason [`report_dispatcher_pin`] has one: a
/// placement that collapsed is not a fallback, and sharing a one-shot with
/// anything else would let whichever fired first silence this permanently --
/// which is materially how #1792 stayed unnoticed.
fn report_spmd_placement(message: &str) {
    static REPORTED: OnceLock<()> = OnceLock::new();
    if REPORTED.set(()).is_ok() {
        #[cfg(feature = "tracing")]
        tracing_crate::debug!(placement = %message, "cpu decode pool placement");
        #[cfg(not(feature = "tracing"))]
        if std::env::var("NXRT_CALIB_DEBUG").is_ok() {
            eprintln!("onnx-genai: {message}");
        }
    }
}

/// Report the dispatcher-pin outcome once. Separate static from
/// [`report_spmd_fallback`]'s: this is not a fallback, and folding the two
/// would let whichever fired first silence the other.
fn report_dispatcher_pin(message: &str) {
    static REPORTED: OnceLock<()> = OnceLock::new();
    if REPORTED.set(()).is_ok() {
        #[cfg(feature = "tracing")]
        tracing_crate::debug!(dispatcher_pin = %message, "cpu decode dispatcher pin");
        #[cfg(not(feature = "tracing"))]
        if std::env::var("NXRT_CALIB_DEBUG").is_ok() {
            eprintln!("onnx-genai: {message}");
        }
    }
}

/// Log the first persistent-pool fallback/pinning problem once so a restricted/// or unsupported host surfaces the reason without spamming every worker.
/// Emitted as `tracing::debug!` when the `tracing` feature is enabled, or
/// gated behind `NXRT_CALIB_DEBUG` otherwise.
fn report_spmd_fallback(message: &str) {
    DECODE_PATH_LABEL.get_or_init(|| "flat");
    static REPORTED: OnceLock<()> = OnceLock::new();
    if REPORTED.set(()).is_ok() {
        #[cfg(feature = "tracing")]
        tracing_crate::debug!(path = "flat", reason = %message, "cpu decode pool fallback");
        #[cfg(not(feature = "tracing"))]
        if std::env::var("NXRT_CALIB_DEBUG").is_ok() {
            eprintln!("onnx-genai: persistent SPMD decode pool: {message}");
        }
    }
}

/// Record the selected decode path in the queryable [`DECODE_PATH_LABEL`] static
/// so callers can inspect it via [`decode_path_label`]. Emitted as
/// `tracing::debug!` when the `tracing` feature is enabled (visible to any
/// subscriber at `debug` level or below), or gated behind `NXRT_CALIB_DEBUG`
/// otherwise. See `docs/architecture/ERROR_AND_LOGGING_CONVENTIONS.md` for level guidance.
fn report_pool_built(mode: PersistenceMode) {
    let label = match mode {
        PersistenceMode::On if decode_schedule() == DecodeSchedule::Steal => "work-stealing-pool",
        PersistenceMode::On => "spmd-pool",
        PersistenceMode::Adaptive => "adaptive",
        PersistenceMode::Off => "flat",
    };
    DECODE_PATH_LABEL.get_or_init(|| label);

    #[cfg(feature = "tracing")]
    {
        let workers = POOLS
            .get()
            .and_then(|p| p.as_ref())
            .map(|p| p.total_workers())
            .unwrap_or(0);
        tracing_crate::debug!(path = label, workers, "cpu decode path selected");
    }
    #[cfg(not(feature = "tracing"))]
    if std::env::var("NXRT_CALIB_DEBUG").is_ok() {
        static REPORTED: OnceLock<()> = OnceLock::new();
        if REPORTED.set(()).is_ok() {
            match mode {
                PersistenceMode::On => eprintln!(
                    "NXRT_CALIB: decode path = persistent SPMD pool (default). \
                     Set ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL=auto for load-adaptive \
                     selection, =0 for the flat legacy path"
                ),
                PersistenceMode::Adaptive => eprintln!(
                    "NXRT_CALIB: decode path = adaptive \
                     (ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL=auto); \
                     the initial decode steps are timed both ways and the faster path is \
                     kept permanently. Set =1 or unset for the deterministic pool, =0 for flat"
                ),
                PersistenceMode::Off => {}
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Adaptive calibration (opt-in via =auto): pick the pool or the flat path by
// measuring the live decode step both ways, keeping the faster one.
// ---------------------------------------------------------------------------

/// Which decode path a single adaptive-mode decode step should take. Both paths
/// are token-exact, so the choice never changes the emitted tokens -- only how
/// fast the step runs. The flat and pool paths use different floating-point
/// reduction orders, so switching between them can produce different logits under
/// greedy decode. The calibrator freezes the path once committed to avoid
/// mid-generation non-determinism.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AutoPath {
    /// Dispatch this step's projections to the persistent SPMD pool.
    Pool,
    /// Run this step on the flat Rayon decode pool (the safe default under load).
    Flat,
}

/// Decode steps spent on the pool before the first measurement, so the pool's
/// one-time constant weights are prepacked and node-locally first-touched (see
/// [`SpmdDecodePools::place_rows`]) and caches are warm before it is timed.
/// Without this the pool would be measured reading cross-node memory and unfairly
/// lose; it is a handful of steps, amortized over the whole generation.
const CALIB_WARMUP_STEPS: u64 = 2;
/// Samples collected per path during a probe. The decision uses the median, so an
/// odd count with a small majority rejects a single load-spike outlier while
/// keeping the probe short (a probe costs `2 * CALIB_PROBE_SAMPLES` real steps,
/// half of them possibly-slower pool steps).
const CALIB_PROBE_SAMPLES: usize = 5;
/// Hysteresis margin: the pool is adopted only when its median step time is at
/// least this percent faster than the flat path. Biases toward the flat path (the
/// regression-safe default) and prevents flapping when the two paths are close.
const CALIB_SWITCH_MARGIN_PCT: u64 = 8;
/// Samples discarded at the start of each probe block. The persistent pool's
/// worker threads keep spinning for a short while after their last dispatch, so
/// the *first* flat step after a pool step (and vice-versa) is polluted by the
/// other path's threads still winding down. Discarding the transition sample
/// makes each block measure its path in isolation -- critical because measuring
/// flat while pool workers are still hot makes flat look slow and would bias the
/// choice *toward* the pool (the regression). See the block-ordered probe below.
const CALIB_PROBE_DISCARD: usize = 1;
/// A probe block is treated as **contended** (a co-tenant burst landed on it, so
/// its median overstates that path's true cost) when its fast-half spread
/// `(median - min) / median` exceeds this percent. Every decode step does the
/// identical M=1 work, so an *uncontended* block -- whether uniformly fast (idle)
/// or uniformly slow (sustained load) -- has a tiny spread; only a transient burst
/// that hits some, but not all, of the block's steps inflates it. Sized above the
/// observed clean-block jitter (flat blocks measured ~1-7%; a burst-poisoned pool
/// block measured ~28%) so sustained load is *not* misread as a burst (it commits
/// flat as before) while a transient burst *is* caught.
const CALIB_CONTENDED_SPREAD_PCT: u64 = 15;
/// Maximum times a single recalibration re-collects its probe when a block looks
/// contended before giving up and committing on the (possibly noisy) median. A
/// small bound guarantees the calibrator always commits within a handful of
/// probes -- a genuinely (uniformly) loaded host re-probes at most this many times
/// then correctly commits flat, so re-probing can never loop or sustain overhead.
const CALIB_MAX_REPROBES: u32 = 3;

/// The calibration probe measures the two paths in **separate contiguous blocks**
/// (all flat, then all pool) rather than interleaving them, so a just-finished
/// pool step's still-spinning workers never pollute a flat measurement. Flat is
/// measured first, while the pool is quiesced (parked), which is exactly the
/// steady state a committed-flat adaptive run experiences.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CalibPhase {
    /// Warm the pool (prepack + node-local first-touch) before any measurement.
    Warmup,
    /// Collect the flat block, with the pool quiesced (its worker threads parked).
    ProbeFlat,
    /// Collect the pool block.
    ProbePool,
    /// Run the committed path until the recalibration period elapses.
    Committed,
}

/// Runtime calibrate-and-pick state machine for `Adaptive` mode (`=auto`).
///
/// # Why this cannot regress under load
///
/// * The committed path starts as -- and defaults back to -- [`AutoPath::Flat`],
///   today's flat decode path. A host that never lets the pool win keeps running
///   exactly the flat path.
/// * The pool is adopted only when its *measured* median step time beats the flat
///   path's by [`CALIB_SWITCH_MARGIN_PCT`]. Under co-tenant load the spinning
///   pool is slower, so it never clears the bar and the flat path stays committed.
/// * Flat and pool are measured in **separate blocks** (flat first, pool
///   quiesced), with the transition sample discarded ([`CALIB_PROBE_DISCARD`]),
///   so a pool step's still-spinning workers cannot make the flat measurement
///   look slow. (An interleaved probe *did* mis-commit to the pool under load;
///   see `.squad/decisions.md`, Hudson 2026-07-24.)
/// * The only pool work done while the flat path is committed is a bounded probe
///   (`<= CALIB_PROBE_SAMPLES` pool steps during calibration, plus a one-time
///   warmup), so the worst case is a small number of possibly-slower steps during
///   the initial calibration -- never a sustained regression.
/// * A probe block hit by a *transient* co-tenant burst (non-uniform samples --
///   see [`block_contended`]) is discarded and re-collected (bounded by
///   [`CALIB_MAX_REPROBES`]) rather than committed, so a momentary spike on the
///   pool block no longer locks the slower flat path in permanently. A
///   *uniformly* slow block (genuine sustained load) is never flagged, so this
///   only ever avoids acting on unreliable data -- it cannot regress a clean or a
///   steadily-loaded measurement.
///
/// # Path freezing
///
/// Once the calibrator commits to a path (pool or flat), it stays committed
/// permanently. The flat and pool paths use different floating-point reduction
/// orders, so switching mid-generation changes logits and produces different
/// tokens under greedy decode. A host that becomes loaded after commitment will
/// run the (now-suboptimal) pool path for the rest of the session; this is
/// the correct trade-off because deterministic output is more important than
/// adapting to load changes. Use `ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL=0` to
/// force flat if the host is known to be loaded.
///
/// The decision logic is pure (no threads, no clock; the only env read is an
/// optional `NXRT_CALIB_DEBUG` diagnostic print that never affects the choice), so
/// it is unit-tested deterministically by feeding synthetic per-path samples.
struct Calibrator {
    phase: CalibPhase,
    warmup_left: u64,
    discard_left: usize,
    pool_ns: Vec<u64>,
    flat_ns: Vec<u64>,
    committed: AutoPath,
    reprobes_left: u32,
}

impl Calibrator {
    fn new() -> Self {
        Self {
            phase: CalibPhase::Warmup,
            warmup_left: CALIB_WARMUP_STEPS,
            discard_left: 0,
            pool_ns: Vec::with_capacity(CALIB_PROBE_SAMPLES),
            flat_ns: Vec::with_capacity(CALIB_PROBE_SAMPLES),
            // Default to the flat path: the safe, no-regression baseline that a
            // host which never lets the pool win keeps forever.
            committed: AutoPath::Flat,
            reprobes_left: CALIB_MAX_REPROBES,
        }
    }

    /// The path the next decode step should take. Warmup uses the pool (to place
    /// weights node-locally); the probe runs the flat block then the pool block;
    /// a committed phase returns the committed path.
    fn choose(&self) -> AutoPath {
        match self.phase {
            CalibPhase::Warmup | CalibPhase::ProbePool => AutoPath::Pool,
            CalibPhase::ProbeFlat => AutoPath::Flat,
            CalibPhase::Committed => self.committed,
        }
    }

    /// Feed back the measured wall time (nanoseconds) of a step that took `path`.
    fn record(&mut self, path: AutoPath, ns: u64) {
        match self.phase {
            CalibPhase::Warmup => {
                self.warmup_left = self.warmup_left.saturating_sub(1);
                if self.warmup_left == 0 {
                    self.enter_flat_probe();
                }
            }
            CalibPhase::ProbeFlat => {
                if path == AutoPath::Flat {
                    self.push_sample_or_discard(ns, true);
                }
                if self.flat_ns.len() >= CALIB_PROBE_SAMPLES {
                    self.enter_pool_probe();
                }
            }
            CalibPhase::ProbePool => {
                if path == AutoPath::Pool {
                    self.push_sample_or_discard(ns, false);
                }
                if self.pool_ns.len() >= CALIB_PROBE_SAMPLES {
                    self.commit_from_samples();
                }
            }
            CalibPhase::Committed => {
                // Path is frozen once committed. The flat and pool paths use
                // different FP reduction orders, so switching mid-generation
                // produces non-deterministic tokens under greedy decode.
            }
        }
    }

    /// Record a sample into the current block, discarding the leading transition
    /// sample(s) so the other path's winding-down threads do not pollute it.
    fn push_sample_or_discard(&mut self, ns: u64, flat: bool) {
        if self.discard_left > 0 {
            self.discard_left -= 1;
            return;
        }
        let block = if flat {
            &mut self.flat_ns
        } else {
            &mut self.pool_ns
        };
        if block.len() < CALIB_PROBE_SAMPLES {
            block.push(ns);
        }
    }

    fn enter_flat_probe(&mut self) {
        self.phase = CalibPhase::ProbeFlat;
        self.flat_ns.clear();
        self.pool_ns.clear();
        self.discard_left = CALIB_PROBE_DISCARD;
    }

    fn enter_pool_probe(&mut self) {
        self.phase = CalibPhase::ProbePool;
        self.discard_left = CALIB_PROBE_DISCARD;
    }

    fn commit_from_samples(&mut self) {
        let pool = median_ns(&mut self.pool_ns);
        let flat = median_ns(&mut self.flat_ns);
        // A transient co-tenant burst that lands on part of a probe block inflates
        // that block's median above the path's true cost, which can lock in the
        // wrong path for the whole recalibration period (observed: a burst on the
        // pool block poisoned its median and stuck decode on the flat path for 600
        // steps). When a block looks contended (non-uniform -- see
        // `CALIB_CONTENDED_SPREAD_PCT`) and re-probe budget remains, discard this
        // probe and re-collect rather than commit on unreliable data. A uniformly
        // slow block (genuine sustained load) is NOT flagged, so a loaded host
        // still commits flat exactly as before -- this only ever avoids acting on
        // a burst-poisoned probe, so it cannot regress a clean measurement.
        let contended =
            block_contended(pool, &self.pool_ns) || block_contended(flat, &self.flat_ns);
        if contended && self.reprobes_left > 0 {
            self.reprobes_left -= 1;
            if std::env::var("NXRT_CALIB_DEBUG").is_ok() {
                eprintln!(
                    "NXRT_CALIB: contended probe (pool_median={}us flat_median={}us) -> re-probe ({} left)",
                    pool / 1000,
                    flat / 1000,
                    self.reprobes_left,
                );
            }
            self.enter_flat_probe();
            return;
        }
        // Adopt the pool only when it is at least CALIB_SWITCH_MARGIN_PCT faster:
        // pool <= flat * (100 - margin) / 100. Use u128 so the multiply cannot
        // overflow for pathologically large samples.
        let pool_scaled = u128::from(pool) * 100;
        let flat_scaled = u128::from(flat) * u128::from(100 - CALIB_SWITCH_MARGIN_PCT);
        self.committed = if pool_scaled <= flat_scaled {
            AutoPath::Pool
        } else {
            AutoPath::Flat
        };
        if std::env::var("NXRT_CALIB_DEBUG").is_ok() {
            eprintln!(
                "NXRT_CALIB: pool_median={}us flat_median={}us margin={}% -> {:?} (pool_samples={:?} flat_samples={:?})",
                pool / 1000,
                flat / 1000,
                CALIB_SWITCH_MARGIN_PCT,
                self.committed,
                self.pool_ns.iter().map(|n| n / 1000).collect::<Vec<_>>(),
                self.flat_ns.iter().map(|n| n / 1000).collect::<Vec<_>>(),
            );
        }
        self.phase = CalibPhase::Committed;
        // Path is frozen permanently — no re-probe. See the "Path freezing"
        // section in the struct-level docs.
        self.reprobes_left = CALIB_MAX_REPROBES;
        self.pool_ns.clear();
        self.flat_ns.clear();
    }
}

/// Whether a probe block was contended by a *transient* burst: its fast-half
/// spread `(median - min) / median` exceeds [`CALIB_CONTENDED_SPREAD_PCT`]. Every
/// M=1 decode step does identical work, so an uncontended block -- uniformly fast
/// (idle) or uniformly slow (sustained load) -- has a tiny spread; only a burst
/// that hits some steps but not others pulls the median well above the min. Empty
/// or single-sample blocks are never flagged (no spread to measure). Pure, so the
/// gate is unit-tested without a clock.
fn block_contended(median: u64, samples: &[u64]) -> bool {
    let Some(&min) = samples.iter().min() else {
        return false;
    };
    if median <= min {
        return false;
    }
    u128::from(median - min) * 100 > u128::from(median) * u128::from(CALIB_CONTENDED_SPREAD_PCT)
}

/// Median of the samples (upper-middle for an even count). `u64::MAX` for an empty
/// slice so an unmeasured path never looks like the fast choice.
fn median_ns(samples: &mut [u64]) -> u64 {
    if samples.is_empty() {
        return u64::MAX;
    }
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn calibrator() -> &'static Mutex<Calibrator> {
    static CALIBRATOR: OnceLock<Mutex<Calibrator>> = OnceLock::new();
    CALIBRATOR.get_or_init(|| Mutex::new(Calibrator::new()))
}

/// The path the next adaptive-mode decode step should take (see [`Calibrator`]).
pub(crate) fn auto_choose_path() -> AutoPath {
    calibrator()
        .lock()
        .map(|calib| calib.choose())
        // A poisoned lock (a panic in a prior decode step) should never change
        // tokens or crash decode -- fall back to the safe flat path.
        .unwrap_or(AutoPath::Flat)
}

/// Feed the measured wall time of an adaptive-mode decode step back to calibration.
pub(crate) fn auto_record_sample(path: AutoPath, elapsed: Duration) {
    let ns = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
    if let Ok(mut calib) = calibrator().lock() {
        calib.record(path, ns);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A host that never enters an SPMD decode scope (the plugin EP) turns the
    /// pool off by default, and an explicit environment setting still wins in
    /// both directions -- otherwise opting the plugin out would take the pool
    /// away from anyone who asked for it by name.
    #[test]
    fn process_default_decides_only_the_unset_case() {
        assert_eq!(
            resolve_persistence_mode(None, true),
            PersistenceMode::On,
            "unset with the historical default builds the pool"
        );
        assert_eq!(
            resolve_persistence_mode(None, false),
            PersistenceMode::Off,
            "unset with the pool opted out leaves decode on the flat path"
        );
        assert_eq!(
            resolve_persistence_mode(Some("1"), false),
            PersistenceMode::On,
            "an explicit =1 wins over a process opt-out"
        );
        assert_eq!(
            resolve_persistence_mode(Some("auto"), false),
            PersistenceMode::Adaptive,
            "an explicit =auto wins over a process opt-out"
        );
        assert_eq!(
            resolve_persistence_mode(Some("0"), true),
            PersistenceMode::Off,
            "an explicit =0 wins over the historical default"
        );
    }

    /// #1680: a shard must hand out one CPU per *physical* core before it reuses
    /// any SMT sibling. `node_shards` previously returned `allowed_cpus()` in
    /// kernel order, and workers pin to `cpus[worker % len]`, so on a host whose
    /// siblings are adjacent (`0,1` one core, `2,3` the next) a 16-worker pool
    /// packed onto 8 physical cores. Measured cost on qwen int4 `accuracy_level=0`
    /// decode: 4.65 ms/token against 2.80 ms/token one-worker-per-core (1.66x).
    ///
    /// Skipped where the SMT map is unavailable, and vacuous-but-harmless on a
    /// host without SMT (there every CPU is its own leader).
    #[test]
    fn shard_cpus_prefer_distinct_physical_cores() {
        // Host-independent half. The loop below can only run where `/sys`
        // exposes a sibling map, which a minimal container does not, so on its
        // own this test passes vacuously exactly where it would be most useful.
        // Assert the ordering contract the shard builder relies on against a
        // synthetic SMT host first, so the property is covered everywhere.
        let synthetic = crate::core_topology::CoreTopology::from_sibling_groups(
            (0..8).map(|c| vec![c * 2, c * 2 + 1]),
        );
        let all: Vec<usize> = (0..16).collect();
        let ordered = crate::decode_affinity::order_pin_targets(&all, Some(&synthetic));
        let mut leaders = synthetic.leaders_within(&all);
        leaders.sort_unstable();
        let mut first_eight: Vec<usize> = ordered.iter().take(8).copied().collect();
        first_eight.sort_unstable();
        assert_eq!(
            first_eight, leaders,
            "the first workers must take one CPU per physical core before any \
             worker doubles up on an SMT sibling: {ordered:?}"
        );

        let cores = match crate::core_topology::require_host_for_placement() {
            Ok(cores) => cores,
            Err(reason) => {
                eprintln!("skipping sibling-doubling check: {reason}");
                return;
            }
        };
        for shard in node_shards(4) {
            if shard.cpus.is_empty() {
                continue;
            }
            let mut want = cores.leaders_within(&shard.cpus);
            want.sort_unstable();
            let mut got: Vec<usize> = shard.cpus.iter().take(want.len()).copied().collect();
            got.sort_unstable();
            assert_eq!(
                got, want,
                "shard {} must place one worker per physical core before reusing a sibling",
                shard.index
            );
        }
    }

    fn two_group_pool() -> SpmdDecodePools {
        let shards = vec![
            NodeShard {
                index: 0,
                cpus: vec![],
                workers: 2,
            },
            NodeShard {
                index: 1,
                cpus: vec![],
                workers: 2,
            },
        ];
        SpmdDecodePools::build_with_schedule(&shards, DecodeSchedule::Fixed, false)
    }

    fn single_group_pool(workers: usize) -> SpmdDecodePools {
        single_group_pool_with_schedule(workers, DecodeSchedule::Fixed)
    }

    fn single_group_pool_with_schedule(
        workers: usize,
        schedule: DecodeSchedule,
    ) -> SpmdDecodePools {
        let shards = vec![NodeShard {
            index: 0,
            cpus: vec![],
            workers,
        }];
        SpmdDecodePools::build_with_schedule(&shards, schedule, false)
    }

    /// A single group of `threads` spawned workers plus a dispatcher-owned
    /// shard: the layout an explicit `ONNX_GENAI_CPU_DECODE_THREADS=threads + 1`
    /// budget produces once the headroom reservation has taken its CPU back.
    fn dispatcher_shard_pool(threads: usize) -> SpmdDecodePools {
        let shards = vec![NodeShard {
            index: 0,
            cpus: vec![],
            workers: threads,
        }];
        SpmdDecodePools::build_with_schedule(&shards, DecodeSchedule::Fixed, true)
    }

    #[test]
    fn dispatcher_pin_env_accepts_only_affirmative_spellings() {
        for on in ["1", "on", "true", "yes", " ON ", "True"] {
            assert!(
                dispatcher_pin_from_raw(Some(on)),
                "{on:?} should enable the dispatcher pin"
            );
        }
        for off in ["0", "off", "false", "no", "", "  ", "maybe"] {
            assert!(
                !dispatcher_pin_from_raw(Some(off)),
                "{off:?} should not enable the dispatcher pin"
            );
        }
        // Unset is off: this knob never opts a host in by accident.
        assert!(!dispatcher_pin_from_raw(None));
    }

    #[test]
    fn dispatcher_cpu_is_the_cpu_the_headroom_reserve_freed() {
        // A node with more CPUs than workers is exactly what the reserve
        // produces, and the first unused CPU is the one it kept clear.
        let shards = vec![NodeShard {
            index: 0,
            cpus: vec![0, 2, 4, 6],
            workers: 3,
        }];
        let pools = SpmdDecodePools::build_with_schedule(&shards, DecodeSchedule::Fixed, true);
        assert_eq!(pools.dispatcher_cpu(), Some(6));
        // The reserved CPU is precisely the one no worker claimed.
        let taken: Vec<Option<usize>> = pools.worker_cpus().to_vec();
        assert!(
            !taken.contains(&Some(6)),
            "reserved cpu must not also be a worker's pin target: {taken:?}"
        );
        pools.shutdown();
    }

    #[test]
    fn dispatcher_cpu_is_none_without_a_reservation_or_a_dispatcher_shard() {
        // Fully subscribed: every CPU has a worker, so nothing was reserved and
        // there is no free CPU to name. Reporting one here would be the
        // unverified-label failure the placement accessors exist to avoid.
        let full = vec![NodeShard {
            index: 0,
            cpus: vec![0, 2],
            workers: 2,
        }];
        let pools = SpmdDecodePools::build_with_schedule(&full, DecodeSchedule::Fixed, true);
        assert_eq!(pools.dispatcher_cpu(), None);
        pools.shutdown();

        // No dispatcher shard: the dispatcher computes nothing, so the pool
        // makes no claim on a CPU for it even when one is spare.
        let spare = vec![NodeShard {
            index: 0,
            cpus: vec![0, 2, 4],
            workers: 2,
        }];
        let pools = SpmdDecodePools::build_with_schedule(&spare, DecodeSchedule::Fixed, false);
        assert_eq!(pools.dispatcher_cpu(), None);
        pools.shutdown();
    }

    #[test]
    fn dispatcher_cpu_comes_from_the_node_that_owns_the_dispatcher_shard() {
        // `node_worker_counts` adds the dispatcher's shard to the *last* node,
        // so the reserved CPU must come from that node -- taking node 0's spare
        // would pin the dispatcher on the far side of the barrier it serves.
        let shards = vec![
            NodeShard {
                index: 0,
                cpus: vec![0, 2, 4],
                workers: 2,
            },
            NodeShard {
                index: 1,
                cpus: vec![16, 18, 20],
                workers: 2,
            },
        ];
        let pools = SpmdDecodePools::build_with_schedule(&shards, DecodeSchedule::Fixed, true);
        assert_eq!(pools.dispatcher_cpu(), Some(20));
        pools.shutdown();
    }

    #[test]
    fn decode_schedule_parses_env_values() {
        assert_eq!(decode_schedule_from_raw(None), DecodeSchedule::Fixed);
        assert_eq!(decode_schedule_from_raw(Some("")), DecodeSchedule::Fixed);
        assert_eq!(
            decode_schedule_from_raw(Some("fixed")),
            DecodeSchedule::Fixed
        );
        let expected_steal = if cfg!(feature = "mlas") {
            DecodeSchedule::Steal
        } else {
            DecodeSchedule::Fixed
        };
        assert_eq!(decode_schedule_from_raw(Some("steal")), expected_steal);
        assert_eq!(
            decode_schedule_from_raw(Some(" work-stealing ")),
            expected_steal
        );
        assert_eq!(
            decode_schedule_from_raw(Some("bogus")),
            DecodeSchedule::Fixed
        );
    }

    #[test]
    fn an_unset_or_unparseable_blocktime_uses_the_default() {
        assert_eq!(parse_decode_blocktime(None), DEFAULT_BLOCKTIME);
        // Every rejected spelling must land on the default rather than on zero:
        // silently parking immediately is a real behaviour change, not a
        // conservative fallback.
        for raw in [
            "",
            "   ",
            "abc",
            "-1",
            "1.5",
            "500us",
            "18446744073709551616",
        ] {
            assert_eq!(
                parse_decode_blocktime(Some(raw)),
                DEFAULT_BLOCKTIME,
                "{raw:?} is not a microsecond count and must fall back"
            );
        }
    }

    #[test]
    fn an_explicit_zero_blocktime_parks_immediately_rather_than_taking_the_default() {
        // Load-bearing, and the reason this policy is worth a test at all: the
        // sibling knob `steal_tiles_per_worker` filters `> 0` and treats zero as
        // "unset". Blocktime must not, because `0` is the one setting that makes
        // a worker park without spinning -- the maximally-polite mode used to
        // measure park/wake latency. Folding it into the default would silently
        // give every such run a 500us spin window and make the measurement
        // impossible while still appearing to work.
        assert_eq!(parse_decode_blocktime(Some("0")), Duration::ZERO);
        assert_ne!(parse_decode_blocktime(Some("0")), DEFAULT_BLOCKTIME);
    }

    #[test]
    fn a_blocktime_is_read_as_microseconds_and_tolerates_surrounding_space() {
        assert_eq!(
            parse_decode_blocktime(Some("250")),
            Duration::from_micros(250)
        );
        assert_eq!(
            parse_decode_blocktime(Some(" 250\n")),
            Duration::from_micros(250)
        );
        // Pins the unit. Reading the same digits as millis or nanos would be a
        // 1000x error in either direction and every one of the tests above would
        // still pass.
        assert_ne!(
            parse_decode_blocktime(Some("250")),
            Duration::from_millis(250)
        );
        assert_ne!(
            parse_decode_blocktime(Some("250")),
            Duration::from_nanos(250)
        );
    }

    #[test]
    fn reserve_single_group_headroom_frees_a_dispatcher_cpu_when_fully_subscribed() {
        // The pathological forced case: requested workers == allowed CPUs (e.g.
        // `taskset -c 0-31` with THREADS=32). One CPU must be reserved for the
        // inline dispatcher, so 31 workers are pinned and the highest-index
        // allowed CPU stays free (workers pin to cpus[0..workers] round-robin).
        // `core_count == 0` is "SMT map unavailable" and keeps the logical rule.
        assert_eq!(reserve_single_group_headroom(32, 32, 0), 31);
        // Oversubscription (workers > allowed) is likewise capped to allowed - 1.
        assert_eq!(reserve_single_group_headroom(40, 32, 0), 31);
        // Even an explicit THREADS=N on an exactly-N-CPU cpuset still reserves the
        // dispatcher core: N-1 workers, never N and never zero.
        assert_eq!(reserve_single_group_headroom(2, 2, 0), 1);
    }

    /// #1680: on an SMT host the logical-CPU reserve never fires for the default
    /// budget -- 16 workers against 32 allowed logical CPUs looks like headroom,
    /// but the 16 workers occupy all 16 *physical* cores and the inline
    /// dispatcher is left to share a core with one of them. Measured 4.41 ms/token
    /// against 2.81 ms/token for 15 workers (qwen int4 `accuracy_level=0`).
    #[test]
    fn reserve_single_group_headroom_reserves_a_physical_core_within_the_core_budget() {
        // The measured defect: 16-core SMT host, no cpuset, default budget.
        assert_eq!(reserve_single_group_headroom(16, 32, 16), 15);
        // Genuine headroom below the core budget is untouched.
        assert_eq!(reserve_single_group_headroom(8, 32, 16), 8);
        assert_eq!(reserve_single_group_headroom(1, 32, 16), 1);
        // A cpuset of 4 cores / 8 logical CPUs, asked for exactly the core budget.
        assert_eq!(reserve_single_group_headroom(4, 8, 4), 3);
        // Past the core budget the user explicitly asked for SMT oversubscription,
        // where reserving cores measured 1.28x slower (10.58 ms -> 13.56 ms), so
        // the logical-CPU reserve still governs.
        assert_eq!(reserve_single_group_headroom(16, 8, 4), 7);
        // Single-core cpuset still yields a worker rather than zero.
        assert_eq!(reserve_single_group_headroom(2, 2, 1), 1);
    }

    // --- #1792: the persistent pool must honor an explicit affinity request ---

    /// Every accepted `ONNX_GENAI_CPU_DECODE_AFFINITY` value maps to the meaning
    /// the module docs promise. `off` in particular must not be read as "no
    /// request": the defect this fixes is that the pool pinned workers to
    /// `sched_getaffinity` regardless, so `off` and an explicit node produced
    /// byte-identical placement.
    #[test]
    fn every_accepted_affinity_value_reaches_the_persistent_pool() {
        assert_eq!(
            explicit_affinity_request(Some("off")),
            ExplicitAffinity::Unpinned
        );
        assert_eq!(
            explicit_affinity_request(Some("compact")),
            ExplicitAffinity::FromPlan
        );
        assert_eq!(
            explicit_affinity_request(Some("node:0")),
            ExplicitAffinity::FromPlan
        );
        assert_eq!(
            explicit_affinity_request(Some("node:3")),
            ExplicitAffinity::FromPlan
        );
        // `numa-split` defers because the multi-node branch of `node_shards`
        // already *is* that layout -- deferring keeps one implementation.
        assert_eq!(
            explicit_affinity_request(Some("numa-split")),
            ExplicitAffinity::DeferToDefault
        );
        // Surrounding whitespace is the same request, matching `DecodeAffinity`.
        assert_eq!(
            explicit_affinity_request(Some("  off  ")),
            ExplicitAffinity::Unpinned
        );
    }

    /// Unset, empty and unparseable all leave the default placement alone.
    ///
    /// The unparseable case is deliberate rather than incidental: rejecting it
    /// here would turn a typo into silently-unpinned decode, whereas deferring
    /// leaves the flat path free to report it with the full accepted menu.
    #[test]
    fn only_a_real_affinity_request_overrides_default_placement() {
        assert_eq!(
            explicit_affinity_request(None),
            ExplicitAffinity::DeferToDefault
        );
        assert_eq!(
            explicit_affinity_request(Some("")),
            ExplicitAffinity::DeferToDefault
        );
        assert_eq!(
            explicit_affinity_request(Some("   ")),
            ExplicitAffinity::DeferToDefault
        );
        // A typo defers like the rest, but is distinguishable so the caller can
        // report it instead of leaving the user with a silently ignored knob.
        assert_eq!(
            explicit_affinity_request(Some("node:notanumber")),
            ExplicitAffinity::Malformed
        );
        assert_eq!(
            explicit_affinity_request(Some("sideways")),
            ExplicitAffinity::Malformed
        );
        assert!(
            explicit_affinity_shards_for(
                ExplicitAffinity::Malformed,
                4,
                || panic!("a malformed value must not consult topology"),
                || panic!("a malformed value must not consult topology"),
            )
            .is_none(),
            "a malformed value must leave default placement in charge"
        );
    }

    /// `off` yields a shard with no CPUs, which is what
    /// `SpmdDecodePools::build_with_schedule` reads as "do not pin", and keeps
    /// every worker: unpinned workers cannot pin the dispatcher out of a core,
    /// so there is nothing for the headroom reservation to protect.
    #[test]
    fn affinity_off_leaves_the_pool_unpinned_without_widening_it() {
        // Plenty of headroom: 4 workers over 16 allowed CPUs.
        let shards = explicit_affinity_shards_for(
            ExplicitAffinity::Unpinned,
            4,
            || panic!("`off` must not consult the affinity plan"),
            || Some((0..16).collect()),
        )
        .expect("`off` is an explicit request");
        assert_eq!(shards.len(), 1);
        assert!(
            shards[0].cpus.is_empty(),
            "`off` must produce no CPUs to pin to, got {:?}",
            shards[0].cpus
        );
        assert_eq!(
            shards[0].workers, 4,
            "`off` must not narrow a pool with headroom"
        );

        // `off` means "do not pin", not "also run wider": a fully subscribed
        // allowed set still reserves a slot for the inline dispatcher, because
        // an unpinned spinning worker preempts it just as a pinned one does.
        let saturated = explicit_affinity_shards_for(
            ExplicitAffinity::Unpinned,
            4,
            || panic!("`off` must not consult the affinity plan"),
            || Some(vec![0, 1, 2, 3]),
        )
        .expect("`off` is an explicit request");
        assert!(saturated[0].cpus.is_empty());
        assert!(
            saturated[0].workers < 4,
            "a saturated `off` pool must still leave the dispatcher somewhere to run, got {}",
            saturated[0].workers
        );

        // No knowable allowed set (pinning unsupported): nothing to reserve
        // against, so the requested width stands -- same as the default path.
        let unknown = explicit_affinity_shards_for(
            ExplicitAffinity::Unpinned,
            4,
            || panic!("`off` must not consult the affinity plan"),
            || None,
        )
        .expect("`off` is an explicit request");
        assert_eq!(unknown[0].workers, 4);
    }

    /// A `compact` / `node:<index>` request takes its CPU set from the shared
    /// planner -- the one that owns topology detection and the allowed-set
    /// intersection -- and is then laid out by the *same* spread the default
    /// path uses, so an explicit request cannot pin two workers onto one
    /// physical core (the defect #1729 fixed).
    #[test]
    fn a_planned_affinity_request_pins_to_the_planned_cpus() {
        let planned = vec![8, 9, 10, 11];
        let shards = explicit_affinity_shards_for(
            ExplicitAffinity::FromPlan,
            4,
            || Some(planned.clone()),
            || panic!("a planned request must not consult the allowed set"),
        )
        .expect("a planned request is honored");
        assert_eq!(shards.len(), 1);

        // The request decides *which* CPUs: exactly the planned set, nothing
        // invented and nothing dropped.
        let mut got = shards[0].cpus.clone();
        got.sort_unstable();
        assert_eq!(got, planned, "the planned CPU set must be honored exactly");

        // ...and the placement policy decides the order. Asserted as the
        // property rather than by comparing against `order_pin_targets`, which
        // would just be restating the implementation: the leading pin targets
        // must land on *distinct physical cores*, so the first workers never
        // share a front end. Dropping the spread from this arm fails here on
        // any host with core topology.
        let detected = match crate::core_topology::require_host_for_placement() {
            Ok(cores) => Some(cores),
            Err(reason) => {
                eprintln!("skipping planned-affinity spread check: {reason}");
                None
            }
        };
        if let Some(cores) = detected {
            let core_count = cores.leaders_within(&planned).len();
            let mut seen_cores = std::collections::BTreeSet::new();
            for &cpu in &shards[0].cpus[..core_count] {
                // Identify the core by its sibling group, not by
                // `leaders_within(&[cpu])` -- that answers "the leader among
                // *these* CPUs" and so returns `cpu` itself for a one-element
                // slice, which would make this check vacuously pass.
                let core: Vec<usize> = cores
                    .siblings_of(cpu)
                    .map_or_else(|| vec![cpu], <[usize]>::to_vec);
                assert!(
                    seen_cores.insert(core),
                    "cpu {cpu} shares a physical core with an earlier pin target in {:?}",
                    shards[0].cpus
                );
            }
        }

        // Fully subscribed (4 workers, 4 CPUs), so the inline dispatcher still
        // gets headroom on the explicit path exactly as on the default one.
        // Every topology branch of `reserve_single_group_headroom` agrees on 3
        // here, whether the four CPUs are two SMT pairs, four cores, or a host
        // with no discoverable core topology at all.
        assert_eq!(
            shards[0].workers, 3,
            "an explicit request must not starve the dispatcher"
        );

        // With genuine headroom the requested count is untouched.
        let roomy = explicit_affinity_shards_for(
            ExplicitAffinity::FromPlan,
            2,
            || Some(vec![0, 1, 2, 3, 4, 5, 6, 7]),
            || panic!("a planned request must not consult the allowed set"),
        )
        .expect("a planned request is honored");
        assert_eq!(roomy[0].workers, 2);
    }

    /// If the planner cannot produce a CPU set -- no topology, an empty
    /// intersection, a platform without pinning -- the default placement stays
    /// in charge rather than the pool silently running unpinned.
    #[test]
    fn an_unresolvable_plan_falls_back_to_default_placement() {
        let no_allowed = || panic!("the allowed set is only for `off`");
        assert!(
            explicit_affinity_shards_for(ExplicitAffinity::FromPlan, 4, || None, no_allowed)
                .is_none()
        );
        assert!(
            explicit_affinity_shards_for(
                ExplicitAffinity::FromPlan,
                4,
                || Some(Vec::new()),
                no_allowed
            )
            .is_none()
        );
        assert!(
            explicit_affinity_shards_for(
                ExplicitAffinity::DeferToDefault,
                4,
                || panic!("deferring must not consult the plan"),
                || panic!("deferring must not consult the allowed set"),
            )
            .is_none(),
            "deferring must consult nothing at all"
        );
    }

    /// The regression this fixes, stated as the property that failed: two
    /// different explicit requests must not produce identical placement.
    ///
    /// Before the fix both went through `sched_getaffinity` and were
    /// byte-identical, which is why the knob could be inert without any test
    /// noticing. Reverting `node_shards` to ignore the request makes this fail.
    #[test]
    fn different_affinity_requests_do_not_collapse_to_the_same_placement() {
        let planned = || Some(vec![16, 17, 18, 19]);
        let off =
            explicit_affinity_shards_for(ExplicitAffinity::Unpinned, 4, planned, || Some(vec![16]))
                .expect("`off` is honored");
        let node =
            explicit_affinity_shards_for(ExplicitAffinity::FromPlan, 4, planned, || Some(vec![16]))
                .expect("`node:<index>` is honored");
        assert_ne!(
            off[0].cpus, node[0].cpus,
            "`off` and `node:<index>` must not resolve to the same CPU set"
        );
        assert!(off[0].cpus.is_empty());
        let mut node_cpus = node[0].cpus.clone();
        node_cpus.sort_unstable();
        assert_eq!(node_cpus, vec![16, 17, 18, 19]);
    }

    /// Whether this environment can pin a thread to every one of `cpus`.
    ///
    /// Probes the pinning primitive *directly* rather than through the pool, so
    /// it stays independent of the bookkeeping under test: a defect that made
    /// `worker_cpus` report nothing must not also switch these tests off.
    ///
    /// It probes the exact CPUs the caller will use, not a representative one.
    /// Under Miri the affinity shim accepts cpu 0 and rejects the rest, so a
    /// single-CPU probe reports "pinning works" and the test then fails on the
    /// sandbox's virtual topology instead of on the pool.
    fn environment_can_pin(cpus: &[usize]) -> bool {
        !cpus.is_empty()
            && cpus.iter().all(|&cpu| {
                std::thread::spawn(move || {
                    crate::decode_affinity::pin_current_thread_to_cpu(cpu).is_ok()
                })
                .join()
                .unwrap_or(false)
            })
    }

    /// Selects child mode for [`ready_leak_child`].
    #[cfg(target_os = "linux")]
    const READY_LEAK_CHILD_ENV: &str = "ONNX_GENAI_TEST_READY_LEAK_CHILD";

    /// Threads in this process whose name marks them as SPMD decode workers.
    ///
    /// The 15-byte `comm` truncation is why this is a prefix test and why it is
    /// only sound in a single-pool process.
    #[cfg(target_os = "linux")]
    fn live_spmd_worker_threads() -> usize {
        let Ok(entries) = std::fs::read_dir("/proc/self/task") else {
            return 0;
        };
        entries
            .filter_map(Result::ok)
            .filter(|entry| {
                std::fs::read_to_string(entry.path().join("comm"))
                    .is_ok_and(|comm| comm.trim_end().starts_with("onnx-genai-spmd"))
            })
            .count()
    }

    #[test]
    #[ignore = "child process driven by a_failed_build_leaves_no_workers_running"]
    #[cfg(target_os = "linux")]
    fn ready_leak_child() {
        if std::env::var(READY_LEAK_CHILD_ENV).is_err() {
            return;
        }
        POOL_READY_TIMEOUT_MS.with(|slot| slot.set(250));
        FAIL_WORKER_BEFORE_READY.with(|slot| slot.set(3));
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let outcome = std::panic::catch_unwind(|| {
            SpmdDecodePools::build_with_schedule(
                &[NodeShard {
                    index: 0,
                    cpus: Vec::new(),
                    workers: 6,
                }],
                DecodeSchedule::Fixed,
                false,
            )
        });
        std::panic::set_hook(previous);
        assert!(outcome.is_err(), "the injected fault must fail the build");

        // Poll rather than sleep a fixed interval: the claim is that they leave,
        // not that they leave within one arbitrary instant. A worker that is
        // parked on the futex and never woken never leaves, so this reports the
        // steady state either way.
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut live = live_spmd_worker_threads();
        while live > 0 && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(20));
            live = live_spmd_worker_threads();
        }
        println!("spmd_threads_after={live}");
    }

    /// A failed build must not leave its surviving workers running.
    ///
    /// The backstop's whole purpose is to stop holding a machine, so panicking
    /// while leaving parked or spinning workers behind would only relocate the
    /// defect. Runs in a child process because thread identity cannot be
    /// established in-process: `/proc/<pid>/task/*/comm` truncates at 15 bytes,
    /// so every worker of every pool reports the same `onnx-genai-spmd`, and a
    /// concurrently-running test's pool is indistinguishable from this one's.
    /// A child owns all of its threads, which makes the count exact.
    #[test]
    #[cfg(target_os = "linux")]
    #[cfg_attr(miri, ignore = "Miri cannot spawn the child process this needs")]
    fn a_failed_build_leaves_no_workers_running() {
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
        cmd.arg("--exact")
            .arg("decode_spmd::tests::ready_leak_child")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .arg("--ignored")
            .env(READY_LEAK_CHILD_ENV, "1")
            // Park immediately. A worker still in its spin ramp would drift out
            // on the stop flag alone, which would let a build that never wakes
            // anyone pass this test; parked workers can only leave if they are
            // actually woken.
            .env(DECODE_BLOCKTIME_ENV, "0")
            .env(PERSISTENT_POOL_ENV, "1")
            .env_remove(DECODE_SCHEDULE_ENV)
            .env_remove(crate::decode_affinity::DECODE_AFFINITY_ENV);
        let output = cmd.output().expect("run readiness-leak child");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let live: usize = stdout
            .split_once("spmd_threads_after=")
            .map(|(_, rest)| {
                rest.chars()
                    .take_while(char::is_ascii_digit)
                    .collect::<String>()
            })
            .and_then(|digits| digits.parse().ok())
            .unwrap_or_else(|| {
                panic!(
                    "child never reported its worker-thread count; \
                     stdout: {stdout}\nstderr: {}",
                    String::from_utf8_lossy(&output.stderr)
                )
            });
        assert_eq!(
            live, 0,
            "a failed build left {live} decode worker(s) running; \
             panicking while holding threads relocates the defect rather than \
             fixing it"
        );
    }

    /// Selects child mode for [`shutdown_join_child`].
    #[cfg(target_os = "linux")]
    const SHUTDOWN_JOIN_CHILD_ENV: &str = "ONNX_GENAI_TEST_SHUTDOWN_JOIN_CHILD";

    /// Child half of [`shutdown_pools_is_a_barrier_not_a_request`].
    #[test]
    #[ignore = "child process driven by shutdown_pools_is_a_barrier_not_a_request"]
    #[cfg(target_os = "linux")]
    fn shutdown_join_child() {
        if std::env::var(SHUTDOWN_JOIN_CHILD_ENV).is_err() {
            return;
        }
        let built = pools().map_or(0, SpmdDecodePools::total_workers);
        let spawned = pools().map_or(0, SpmdDecodePools::spawned_workers);
        let threads_before = live_spmd_worker_threads();
        shutdown_pools();
        // Read immediately, with no polling and no sleep. Polling would test
        // "the workers leave eventually", which is a weaker claim that a
        // shutdown with no join also satisfies -- and the whole reason three
        // child tests call this at exit is that "eventually" is not soon enough
        // when the next thing the process does is tear down its C runtime.
        let exited = pools().map_or(0, SpmdDecodePools::workers_exited);
        let threads_after = live_spmd_worker_threads();
        println!(
            "spmd_built={built} spmd_spawned={spawned} spmd_exited={exited} \
             spmd_threads_before={threads_before} spmd_threads_after={threads_after} \
             spmd_avail={}",
            std::thread::available_parallelism().map_or(0, std::num::NonZeroUsize::get)
        );
    }

    /// `shutdown_pools` must not return until the workers are gone.
    ///
    /// Three child processes (the parity child, the realized-width child and
    /// the affinity-defer child) call this as their last act, and #1745 is a
    /// `STATUS_ACCESS_VIOLATION` in exactly those children on Windows ARM64.
    /// The remedy assumes shutdown is a *barrier*: if it merely asked the
    /// workers to stop, the child would still be racing its own runtime
    /// teardown against threads parked on an `Arc<SharedState>`, and the call
    /// would be decoration.
    ///
    /// In a child process because `/proc/<pid>/task/*/comm` truncates at 15
    /// bytes, so every SPMD worker of every pool reports the same name and a
    /// concurrent test's pool is indistinguishable from this one's. A child owns
    /// all of its threads, which makes the count exact. Same reasoning as
    /// [`a_failed_build_leaves_no_workers_running`].
    ///
    /// The reference count is [`SpmdDecodePools::spawned_workers`], not
    /// [`SpmdDecodePools::total_workers`]. The latter counts the inline
    /// dispatcher shard, which never runs [`worker_loop`] and so is never
    /// counted out -- an earlier version of this test compared against it and
    /// failed on every fully-subscribed host, which is to say on a stock 2-core
    /// CI runner, while the shutdown it was accusing was entirely correct.
    ///
    /// The assertion is on [`SpmdDecodePools::workers_exited`] rather than on a
    /// thread count, and that choice was forced by evidence. The obvious
    /// assertion -- zero `onnx-genai-spmd` threads left in `/proc/self/task` --
    /// flaked here under load, because the futex that unblocks `join` is
    /// signalled before the kernel unhashes the task, so a joined thread's
    /// directory can still exist when the next statement runs. That is a race
    /// in the *instrument*, and an intermittently-red teardown test would teach
    /// exactly the wrong lesson about intermittently-red teardown. The counter
    /// is incremented by the worker before it returns and therefore
    /// happens-before the join that observes it. The thread counts are still
    /// printed, but as a report, not an assertion.
    ///
    /// Dropping the join makes this *probabilistic* rather than
    /// always-failing -- the workers do leave, just not before the next
    /// statement -- so it is a falsifier with a measured catch rate rather than
    /// a guaranteed one: **18 of 20 runs** on this host. The thread count would
    /// have caught 20 of 20 in that same batch, and it is still the wrong
    /// instrument: it also fails on a *correct* implementation, and a test that
    /// is red for two different reasons cannot tell you which one it is. The
    /// `built` count is asserted alongside so that a pass obtained by building
    /// nothing cannot masquerade as a clean teardown.
    #[test]
    #[cfg(target_os = "linux")]
    #[cfg_attr(miri, ignore = "Miri cannot spawn the child process this needs")]
    fn shutdown_pools_is_a_barrier_not_a_request() {
        let available = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        // Two widths, because the two regimes fail differently. A width with
        // headroom to spare spawns one thread per compute participant; a width
        // that consumes the whole budget makes `reserve_single_group_headroom`
        // hand one participant's shard to the dispatching thread inline, which
        // is a participant that never enters `worker_loop` and can never be
        // joined. An earlier version of this test compared the exit count
        // against the participant count and so was wrong by exactly one in the
        // second regime -- invisible on this 32-CPU host and a hard failure on
        // a stock 2-core runner. Asking for the whole machine reproduces the
        // second regime on any host wide enough to have one.
        let mut saw_inline_shard = false;
        for width in [8, available] {
            saw_inline_shard |= shutdown_join_arm(width);
        }
        assert!(
            saw_inline_shard || available < 4,
            "no arm reached the inline-dispatcher-shard regime on a host \
             reporting {available} CPUs, so the case that broke this test the \
             first time went uncovered"
        );
    }

    /// One width of [`shutdown_pools_is_a_barrier_not_a_request`]. Returns
    /// whether this arm reached the inline-dispatcher-shard regime.
    #[cfg(target_os = "linux")]
    fn shutdown_join_arm(width: usize) -> bool {
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
        cmd.arg("--exact")
            .arg("decode_spmd::tests::shutdown_join_child")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .arg("--ignored")
            .env(SHUTDOWN_JOIN_CHILD_ENV, "1")
            // Parked, not spinning. A worker still in its spin ramp would drift
            // out on the stop flag alone, which would let a shutdown that never
            // woke anyone pass; a parked worker can only leave if it is woken
            // *and* waited for.
            .env(DECODE_BLOCKTIME_ENV, "0")
            .env(PERSISTENT_POOL_ENV, "1")
            .env(
                crate::kernels::matmul_nbits::DECODE_THREADS_ENV,
                width.to_string(),
            )
            .env_remove(DECODE_SCHEDULE_ENV)
            .env_remove(crate::decode_affinity::DECODE_AFFINITY_ENV);
        let output = cmd.output().expect("run shutdown-join child");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let field = |key: &str| -> usize {
            let needle = format!("{key}=");
            stdout
                .split_once(&needle)
                .map(|(_, rest)| {
                    rest.chars()
                        .take_while(char::is_ascii_digit)
                        .collect::<String>()
                })
                .and_then(|digits| digits.parse().ok())
                .unwrap_or_else(|| {
                    panic!(
                        "child never reported `{key}` at width {width}; stdout: \
                         {stdout}\nstderr: {}",
                        String::from_utf8_lossy(&output.stderr)
                    )
                })
        };
        let built = field("spmd_built");
        // Anti-vacuity: a child that spawned no worker thread would satisfy the
        // teardown assertion for the least interesting reason there is.
        let spawned = field("spmd_spawned");
        if spawned == 0 {
            // A host narrow enough to have no worker to join cannot exercise
            // this property. That is an environment limit, not a pass -- but it
            // is only allowed to be an environment limit on an environment
            // narrow enough to explain it. On anything wider, a pool that
            // spawned nothing is a defect and fails here rather than skipping.
            let host = field("spmd_avail");
            assert!(
                host <= 2,
                "width {width} spawned no worker threads on a host reporting \
                 {host} CPUs, which no cpuset or headroom reservation explains: \
                 stdout: {stdout}"
            );
            println!(
                "SKIPPED (width {width}): {host} available CPU(s) left no \
                 spawned worker to join (the child reported {built} compute \
                 participant(s), none of them a spawned thread), so there is no \
                 teardown to observe here"
            );
            return false;
        }
        assert!(
            field("spmd_threads_before") > 0,
            "width {width} spawned {spawned} worker thread(s) but none were \
             visible in /proc before shutdown: stdout: {stdout}"
        );
        assert_eq!(
            field("spmd_exited"),
            spawned,
            "shutdown_pools returned with decode workers still inside their \
             loop at width {width}, so the call three child processes make \
             before exiting does not actually wait for anything: stdout: \
             {stdout}"
        );
        built > spawned
    }

    /// A worker that never announces must fail the build loudly, not spin.
    ///
    /// This is the defect that cost a shared host 5h40m: the barrier waited on
    /// a condition that had become unsatisfiable, and did it by spinning, so it
    /// held ~18 cores' worth of occupancy indefinitely while looking exactly
    /// like a long-running job. A hang that burns the machine is strictly worse
    /// than one that blocks, because nothing distinguishes it from work.
    ///
    /// Both knobs are thread-local and `catch_unwind` keeps the build on this
    /// thread, so the fault reaches this pool and no other. An earlier global
    /// version of this test injected a worker panic into whichever unrelated
    /// pool happened to be building concurrently.
    #[test]
    fn a_worker_that_never_announces_fails_the_build_instead_of_spinning() {
        POOL_READY_TIMEOUT_MS.with(|slot| slot.set(250));
        FAIL_WORKER_BEFORE_READY.with(|slot| slot.set(1));
        // The injected worker panic is expected; keep it off the test log.
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));

        let started = Instant::now();
        let outcome = std::panic::catch_unwind(|| {
            SpmdDecodePools::build_with_schedule(
                &[NodeShard {
                    index: 0,
                    cpus: Vec::new(),
                    workers: 3,
                }],
                DecodeSchedule::Fixed,
                false,
            )
        });

        std::panic::set_hook(previous);
        FAIL_WORKER_BEFORE_READY.with(|slot| slot.set(usize::MAX));
        POOL_READY_TIMEOUT_MS.with(|slot| slot.set(0));

        let panic = outcome
            .err()
            .expect("a pool missing a worker cannot be built; the barrier must not report success");
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or_default();
        assert!(
            message.contains("never became ready"),
            "the build must say why it gave up, got: {message}"
        );
        assert!(
            message.contains("2 of 3"),
            "the diagnostic must report how many workers did arrive, got: {message}"
        );
        // Bounds the *test*, and with it the claim: an unbounded barrier fails
        // this by never returning at all rather than by returning late.
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "the barrier gave up, but took {:?} to do it",
            started.elapsed()
        );
    }

    /// The backstop must not fire on a pool that is merely slow to start.
    ///
    /// A liveness guard that trips under load is worse than none: it converts a
    /// scheduling hiccup into a crash, which is why the production deadline is
    /// far beyond any real startup rather than tuned close to it.
    ///
    /// The injected delay is what makes this an assertion rather than a
    /// tautology. Real workers announce within the spin budget, so the barrier
    /// never reaches its clock check and the deadline is never consulted -- a
    /// version of this test without the delay passes just as happily with a 1ms
    /// timeout as with 120s, and so proves nothing about the deadline at all.
    /// Delaying every worker past the spin budget forces the barrier onto the
    /// yield/clock path with the deadline genuinely in play, which is the only
    /// arrangement in which "slow is not the same as broken" can be tested.
    #[test]
    fn a_healthy_pool_never_trips_the_readiness_backstop() {
        assert_eq!(
            FAIL_WORKER_BEFORE_READY.with(Cell::get),
            usize::MAX,
            "no fault may be injected on the thread building a healthy pool"
        );
        const DELAY_MS: u64 = 300;
        const DEADLINE_MS: u64 = 5_000;
        POOL_READY_TIMEOUT_MS.with(|slot| slot.set(DEADLINE_MS));
        DELAY_WORKER_BEFORE_READY_MS.with(|slot| slot.set(DELAY_MS));

        let started = Instant::now();
        let pools = SpmdDecodePools::build_with_schedule(
            &[NodeShard {
                index: 0,
                cpus: Vec::new(),
                workers: 4,
            }],
            DecodeSchedule::Fixed,
            false,
        );
        let elapsed = started.elapsed();
        DELAY_WORKER_BEFORE_READY_MS.with(|slot| slot.set(0));
        POOL_READY_TIMEOUT_MS.with(|slot| slot.set(0));

        assert_eq!(pools.total_workers(), 4);
        // Proves the barrier actually waited rather than exiting on the fast
        // path: without this, a build that never consulted the deadline would
        // satisfy the test and the deadline logic would go unexercised.
        assert!(
            elapsed >= Duration::from_millis(DELAY_MS),
            "the barrier returned in {elapsed:?}, before the injected {DELAY_MS}ms \
             delay could have elapsed, so it never reached the deadline path"
        );
        pools.shutdown();
    }

    /// The backstop must fire *within* its deadline, not a stride of yields
    /// later.
    ///
    /// This is the assertion #1825 and #1868 were both missing. Both fixed a
    /// deadline that was evaluated on a `spins.is_multiple_of(64)` stride in a
    /// *yield* phase that began at `SPIN_LOOP_BUDGET` -- itself a multiple of
    /// 64 -- so the clock was read once, on the first yield, and then not for
    /// 64 more. Reverting either fix leaves the whole crate green, so nothing
    /// stops a third reintroduction.
    ///
    /// The observable here is deliberately *binary* rather than a duration.
    /// Overshoot bounds are the natural way to test a granularity defect and
    /// they are flaky by construction: the margin that makes them stable on a
    /// loaded runner is wider than the defect. Choosing a deadline the pool is
    /// guaranteed to exceed converts the same defect into panic-versus-success
    /// -- with the stride, the deadline is consulted once while it is still
    /// unexpired and the build reports success having blown its budget 3x,
    /// which is exactly the measurement recorded at the production site. Load
    /// can only make the injected delay longer, so this cannot flake toward a
    /// false failure.
    #[test]
    fn the_readiness_backstop_fires_within_its_deadline_not_a_stride_later() {
        assert_eq!(
            FAIL_WORKER_BEFORE_READY.with(Cell::get),
            usize::MAX,
            "no fault may be injected; the workers here are slow, not broken"
        );
        // Sized so the two behaviours are separated by a wide margin rather
        // than a granularity: the deadline expires on the yield path at
        // ~DEADLINE_MS, while a stride of 64 injected yields would not consult
        // it again until 64 * SLOW_US = 1280ms -- long after the workers
        // announce at DELAY_MS and the loop has already exited. Fires-late
        // becomes never-fires, which is an outcome rather than a duration.
        //
        // The 8x gap between when the barrier should give up (~100ms) and when
        // it would be let off the hook (800ms) is deliberate headroom for a
        // loaded runner: this test may only fail because the deadline was not
        // consulted, never because the builder thread was descheduled.
        const DELAY_MS: u64 = 800;
        const DEADLINE_MS: u64 = 100;
        const SLOW_US: u64 = 20_000;
        POOL_READY_TIMEOUT_MS.with(|slot| slot.set(DEADLINE_MS));
        DELAY_WORKER_BEFORE_READY_MS.with(|slot| slot.set(DELAY_MS));
        SLOW_YIELD_US.with(|slot| slot.set(SLOW_US));
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));

        let started = Instant::now();
        let outcome = std::panic::catch_unwind(|| {
            SpmdDecodePools::build_with_schedule(
                &[NodeShard {
                    index: 0,
                    cpus: Vec::new(),
                    workers: 2,
                }],
                DecodeSchedule::Fixed,
                false,
            )
        });
        let elapsed = started.elapsed();

        std::panic::set_hook(previous);
        DELAY_WORKER_BEFORE_READY_MS.with(|slot| slot.set(0));
        POOL_READY_TIMEOUT_MS.with(|slot| slot.set(0));
        SLOW_YIELD_US.with(|slot| slot.set(0));

        let panic = outcome.err().unwrap_or_else(|| {
            panic!(
                "the pool exceeded its {DEADLINE_MS}ms deadline by design and the \
                 barrier reported success after {elapsed:?}: the deadline was \
                 consulted once, before it expired, and never again"
            )
        });
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or_default();
        assert!(
            message.contains("never became ready"),
            "the build must say why it gave up, got: {message}"
        );
        // Pins the granularity itself: the barrier may overrun its deadline by
        // the cost of one yield, never by the injected delay that a strided
        // check would have slept through.
        assert!(
            elapsed < Duration::from_millis(DELAY_MS),
            "the barrier gave up only after {elapsed:?}, past the {DELAY_MS}ms \
             delay it was supposed to abandon at {DEADLINE_MS}ms"
        );
    }

    /// The **planner** must lay one worker out per physical core.
    ///
    /// Scoped deliberately: this reads `worker_cpus()`, which is the assignment
    /// plus what the pin call *returned*, so it tests the layout the pool asked
    /// for and nothing about the layout the kernel granted. Stubbing
    /// `sched_setaffinity` to a no-op that reports success leaves this green,
    /// which is why it must never be described as realized placement.
    /// `a_pinned_pool_is_observed_one_worker_per_physical_core` is the one that
    /// asks the kernel.
    #[test]
    fn the_planner_lays_out_one_worker_per_physical_core() {
        let cores = match crate::core_topology::require_host_for_placement() {
            Ok(cores) => cores,
            Err(reason) => {
                eprintln!("skipping pinned-pool placement check: {reason}");
                return;
            }
        };
        // A single-core budget cannot express "one worker per core" as a
        // distinguishable claim. That is a host fact and a legitimate skip;
        // an unanswerable topology is not, and panics above.
        if cores.core_count() < 2 {
            return;
        }
        // Two full physical cores' worth of CPUs, spread the way the placement
        // policy spreads them.
        let cpus: Vec<usize> = crate::decode_affinity::order_pin_targets(
            &(0..cores.logical_count()).collect::<Vec<_>>(),
            Some(cores),
        );
        let workers = cores.core_count().min(4);
        if !environment_can_pin(&cpus[..workers.min(cpus.len())]) {
            return;
        }
        let pools = SpmdDecodePools::build_with_schedule(
            &[NodeShard {
                index: 0,
                cpus,
                workers,
            }],
            DecodeSchedule::Fixed,
            false,
        );
        let pinned: Vec<usize> = pools.worker_cpus().iter().flatten().copied().collect();
        assert_eq!(
            pinned.len(),
            workers,
            "every worker of a pinned pool must have a pin target"
        );
        assert_eq!(
            pools.planned_placement_is_one_worker_per_physical_core(),
            Some(true),
            "pinned workers landed on {pinned:?}, which shares a physical core"
        );
        pools.shutdown();
    }

    /// The same **planner** predicate must be able to say *no*, or it is
    /// decoration.
    #[test]
    fn the_planner_reports_a_shared_core_as_a_defect() {
        // Fails closed. Under the mutation battery this test -- the negative
        // control, the one whose whole job is to prove the detector detects --
        // passed vacuously with detection forced to `None`. A control that
        // survives the removal of the thing it controls for is not a control.
        let cores = match crate::core_topology::require_host_for_placement() {
            Ok(cores) => cores,
            Err(reason) => {
                eprintln!("skipping shared-core control: {reason}");
                return;
            }
        };
        // Find one physical core with at least two siblings, and pin two
        // workers onto it -- exactly the layout #1729 removed.
        let Some(pair) = cores.cores().iter().find(|group| group.len() >= 2) else {
            return; // non-SMT host: the bad layout is unrepresentable
        };
        if !environment_can_pin(&pair[..2]) {
            return;
        }
        let pools = SpmdDecodePools::build_with_schedule(
            &[NodeShard {
                index: 0,
                cpus: vec![pair[0], pair[1]],
                workers: 2,
            }],
            DecodeSchedule::Fixed,
            false,
        );
        assert_eq!(
            pools.planned_placement_is_one_worker_per_physical_core(),
            Some(false),
            "two workers on cpus {:?} share a core and must be reported as such",
            &pair[..2]
        );
        pools.shutdown();
    }

    /// The pool is **observed** to be one worker per physical core.
    ///
    /// The difference from `the_planner_lays_out_one_worker_per_physical_core`
    /// is the whole reason this exists. That one reads `worker_cpus()`, which
    /// is the assignment plus what the pin call *returned*. This one reads what
    /// each worker asked the kernel about itself after pinning, so it cannot be
    /// satisfied by a pin that was accepted and not enforced --
    /// `a_pin_that_reports_success_without_the_syscall_fails_the_realized_check`
    /// is the proof of that, and it is the mutation this test exists to survive.
    #[test]
    fn a_pinned_pool_is_observed_one_worker_per_physical_core() {
        let cores = match crate::core_topology::require_host_for_placement() {
            Ok(cores) => cores,
            Err(reason) => {
                eprintln!("skipping observed-placement check: {reason}");
                return;
            }
        };
        if cores.core_count() < 2 {
            return; // one core cannot express "one per core" distinguishably
        }
        // Stated, not implied. Today no target can pin without also being able
        // to read a mask back, so on every real host this is decided by
        // `environment_can_pin` below -- but *that* gates on the pinning
        // capability while everything after it needs the observation
        // capability, and a guard standing in for a different capability than
        // the one it protects is the defect this whole change is about.
        if !crate::decode_affinity::affinity_observation_supported() {
            eprintln!(
                "skipping observed-placement check: this target has no per-thread affinity \
                 query, so there is no realized placement to observe"
            );
            return;
        }
        let cpus: Vec<usize> = crate::decode_affinity::order_pin_targets(
            &(0..cores.logical_count()).collect::<Vec<_>>(),
            Some(cores),
        );
        let workers = cores.core_count().min(4);
        if !environment_can_pin(&cpus[..workers.min(cpus.len())]) {
            // A sandbox that refuses `sched_setaffinity` is a genuine
            // "unsupported here", stated rather than silent. It is *not*
            // scored as a pass of the property below.
            eprintln!(
                "skipping observed-placement check: this environment refuses to pin to \
                 {:?}",
                &cpus[..workers.min(cpus.len())]
            );
            return;
        }
        let pools = SpmdDecodePools::build_with_schedule(
            &[NodeShard {
                index: 0,
                cpus,
                workers,
            }],
            DecodeSchedule::Fixed,
            false,
        );

        // Fail closed, per worker: a pin this pool says it applied must be a
        // pin the kernel can be seen to have applied. No `if let Some(..)`
        // anywhere in here -- an unreadable mask on a target that has the query
        // is a defect in the apparatus, and it panics.
        let mut applied = 0usize;
        let mut tids = std::collections::BTreeSet::new();
        for (index, placement) in pools.worker_placements().iter().enumerate() {
            assert_eq!(
                placement.attempt,
                PinAttempt::Applied,
                "worker {index} did not take its pin after `environment_can_pin` said this \
                 environment allows pinning ({placement:?})"
            );
            applied += 1;
            let observed = match &placement.observed {
                crate::decode_affinity::ObservedAffinity::Cpus(cpus) => cpus,
                other => panic!(
                    "worker {index} pinned successfully but could not read back its own \
                     affinity on a target that has the query -- the observation is switched \
                     off, which is the exact shape of a check that cannot fail ({other:?})"
                ),
            };
            assert_eq!(
                observed.as_slice(),
                [placement
                    .attempted_cpu
                    .expect("an applied pin names the cpu it took")],
                "worker {index} reports a pin the kernel did not enforce ({placement:?})"
            );
            if let Some(tid) = placement.tid {
                assert!(
                    tids.insert(tid),
                    "two workers reported the same OS thread id {tid}, so the placement \
                     report is not per-worker"
                );
            }
        }
        assert_eq!(
            applied, workers,
            "the pool must report one placement per spawned worker"
        );
        #[cfg(target_os = "linux")]
        assert_eq!(
            tids.len(),
            workers,
            "every worker must report its own TID on Linux, or the report cannot be \
             correlated with `/proc/self/task`"
        );

        assert_eq!(
            pools.realized_placement(),
            RealizedPlacement::OneWorkerPerPhysicalCore,
            "workers observed themselves on {:?}, which is not one per physical core",
            pools
                .worker_placements()
                .iter()
                .map(WorkerPlacement::realized_cpu)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            pools.placement_report_is_honest(),
            Some(true),
            "the pool's reported CPUs and its workers' observed CPUs disagree"
        );
        pools.shutdown();
    }

    /// The observed predicate must be able to say *no*, or it is decoration.
    ///
    /// Two workers pinned onto SMT siblings of one physical core -- the #1729
    /// layout -- judged from the masks the workers read for themselves.
    #[test]
    fn an_observed_shared_core_pool_is_reported_as_a_defect() {
        let cores = match crate::core_topology::require_host_for_placement() {
            Ok(cores) => cores,
            Err(reason) => {
                eprintln!("skipping observed shared-core control: {reason}");
                return;
            }
        };
        let allowed: std::collections::BTreeSet<usize> = crate::decode_affinity::allowed_cpus()
            .unwrap_or_else(|| (0..cores.logical_count()).collect())
            .into_iter()
            .collect();
        // This control judges from observed masks, so it needs the observation
        // capability specifically -- see the note on the positive case above.
        if !crate::decode_affinity::affinity_observation_supported() {
            eprintln!(
                "skipping observed shared-core control: this target has no per-thread \
                 affinity query"
            );
            return;
        }
        // Pick the sibling pair from CPUs this process may actually use, or a
        // `taskset`ed runner -- the most common way this suite runs -- skips.
        let pair = cores.cores().iter().find_map(|group| {
            let mut inside = group.iter().copied().filter(|cpu| allowed.contains(cpu));
            match (inside.next(), inside.next()) {
                (Some(first), Some(second)) => Some([first, second]),
                _ => None,
            }
        });
        let Some(pair) = pair else {
            eprintln!(
                "skipping observed shared-core control: no physical core has two SMT \
                 siblings inside the allowed set {allowed:?}"
            );
            return;
        };
        if !environment_can_pin(&pair) {
            eprintln!(
                "skipping observed shared-core control: this environment refuses to pin to \
                 {pair:?}"
            );
            return;
        }
        let pools = SpmdDecodePools::build_with_schedule(
            &[NodeShard {
                index: 0,
                cpus: pair.to_vec(),
                workers: 2,
            }],
            DecodeSchedule::Fixed,
            false,
        );
        assert_eq!(
            pools.realized_placement(),
            RealizedPlacement::SharedCore,
            "two workers observed on cpus {pair:?} share a physical core and must be \
             reported as such ({:?})",
            pools.worker_placements()
        );
        // The layout is wrong and the *report* is still honest: these are two
        // independent properties, and conflating them is how a policy change
        // would quietly delete the honesty check.
        assert_eq!(
            pools.placement_report_is_honest(),
            Some(true),
            "a badly placed pool can still report its placement truthfully"
        );
        pools.shutdown();
    }

    /// **The mutation.** Pinning that reports success and does nothing.
    ///
    /// This is the falsifier for the whole apparatus, kept in-tree rather than
    /// applied by hand, because the property it protects is not "the pool pins
    /// correctly" but "the pool's placement label is evidence". With
    /// `sched_setaffinity` removed and `Ok(())` returned in its place:
    ///
    /// * the planner's predicate still answers `Some(true)` -- it only ever saw
    ///   the assignment and a return value, and both are unchanged; while
    /// * the observed predicate answers something that is *not*
    ///   `OneWorkerPerPhysicalCore`, because the workers read their real masks.
    ///
    /// Asserting both directions in one test is deliberate. The first half is
    /// what makes the second half meaningful: it shows the two predicates
    /// genuinely disagree here, so this is a demonstration that the old check
    /// was a false oracle rather than merely a new check that happens to pass.
    #[test]
    fn a_pin_that_reports_success_without_the_syscall_fails_the_realized_check() {
        let cores = match crate::core_topology::require_host_for_placement() {
            Ok(cores) => cores,
            Err(reason) => {
                eprintln!("skipping pin-stub mutation: {reason}");
                return;
            }
        };
        if cores.core_count() < 2 {
            return;
        }
        let cpus: Vec<usize> = crate::decode_affinity::order_pin_targets(
            &(0..cores.logical_count()).collect::<Vec<_>>(),
            Some(cores),
        );
        let workers = cores.core_count().min(4).min(cpus.len());
        if workers < 2 {
            return;
        }

        STUB_PIN_SYSCALL.with(|slot| slot.set(true));
        let pools = SpmdDecodePools::build_with_schedule(
            &[NodeShard {
                index: 0,
                cpus: cpus.clone(),
                workers,
            }],
            DecodeSchedule::Fixed,
            false,
        );
        // Cleared immediately, and before any assertion, so a failure below
        // cannot leave the mutation latched for anything else this thread
        // builds.
        STUB_PIN_SYSCALL.with(|slot| slot.set(false));

        assert_eq!(
            pools.planned_placement_is_one_worker_per_physical_core(),
            Some(true),
            "the mutation is supposed to be invisible to the planner's predicate; if it is \
             not, this test is no longer demonstrating what it claims ({:?})",
            pools.worker_cpus()
        );

        let realized = pools.realized_placement();
        assert!(
            !realized.is_one_worker_per_physical_core(),
            "pinning did nothing at all and the observed-placement check still reported \
             one worker per physical core, so it is reading the plan rather than the \
             machine ({realized:?}, {:?})",
            pools.worker_placements()
        );
        assert_ne!(
            pools.placement_report_is_honest(),
            Some(true),
            "the pool vouched for CPUs its workers were never put on ({:?})",
            pools.worker_placements()
        );

        // The two assertions above are the property, and they hold on every
        // target. The two below are the *sharper* claim, and it is available
        // only where a thread can read its own mask. Split rather than relaxed:
        // each branch still names one exact expected value, so neither is a
        // skip.
        //
        // The gate is `cfg!`-derived -- a property of the target -- and not
        // "did the observation happen to work", which is the thing whose
        // failure this test exists to report and so cannot be allowed to
        // decide whether the test runs.
        if crate::decode_affinity::affinity_observation_supported() {
            assert_eq!(
                realized,
                RealizedPlacement::SharedCore,
                "unpinned workers share whatever cores the scheduler picks; an \
                 `Unobservable` on a target that has the affinity query means the masks \
                 could not be read, which is a different failure ({:?})",
                pools.worker_placements()
            );
            assert_eq!(
                pools.placement_report_is_honest(),
                Some(false),
                "the pool reports CPUs its workers are not confined to, which is the #1792 \
                 shape exactly ({:?})",
                pools.worker_placements()
            );
        } else {
            // macOS arm64 reaches here: the stub makes the pin *report* success
            // on a target whose real `pin_current_thread_to_cpu` returns `Err`,
            // and there is no affinity query to catch it out. Failing closed is
            // the only correct answer, and it is worth an explicit assertion --
            // this is the one lane that proves the blind-spot path is not
            // decorative.
            assert_eq!(
                realized,
                RealizedPlacement::Unobservable(PlacementBlindSpot::AffinityQueryUnsupported),
                "this target has no per-thread affinity query, so the only honest verdict \
                 on a stubbed pin is that the placement is unobservable ({:?})",
                pools.worker_placements()
            );
            assert_eq!(
                pools.placement_report_is_honest(),
                None,
                "every worker claims a pin whose mask cannot be read, so the pool's report \
                 is unverified and must not be scored either way ({:?})",
                pools.worker_placements()
            );
        }
        pools.shutdown();
    }

    /// A blind spot is never success -- asserted on the decision itself, over
    /// evidence that cannot be produced on the hosts this suite runs on.
    ///
    /// `QueryFailed` is unreachable through a real pool anywhere in CI, and
    /// `Unsupported` is reachable only on macOS, where nothing else in this
    /// module can construct a pinned pool to observe. Reachable only through a
    /// real pool would mean untested on most lanes, and "unanswerable must not
    /// read as answered" is the single property the whole redesign turns on.
    #[test]
    fn an_unanswerable_placement_is_never_reported_as_placed() {
        use crate::decode_affinity::ObservedAffinity;
        let cores = crate::core_topology::host();
        let worker = |observed: ObservedAffinity| WorkerPlacement {
            tid: Some(1),
            attempted_cpu: Some(0),
            attempt: PinAttempt::Applied,
            observed,
        };

        for blind in [
            ObservedAffinity::Unsupported,
            ObservedAffinity::QueryFailed("injected".to_string()),
        ] {
            let verdict = realized_placement_of(&[worker(blind.clone())], cores);
            assert!(
                !verdict.is_one_worker_per_physical_core(),
                "an unreadable mask was reported as a realized placement: {verdict:?}"
            );
            assert!(
                matches!(verdict, RealizedPlacement::Unobservable(_)),
                "an unreadable mask must be reported as a blind spot, not as a defect and \
                 not as success: {verdict:?}"
            );
        }

        // An undetectable topology is the third blind spot, and it must not
        // collapse into either of the others.
        assert_eq!(
            realized_placement_of(&[worker(ObservedAffinity::Cpus(vec![0]))], None),
            RealizedPlacement::Unobservable(PlacementBlindSpot::TopologyUndetected)
        );

        // Every blind spot, plus the two real verdicts, checked against the one
        // predicate callers use. A future variant added without thought lands
        // in the `match` below and has to be classified deliberately.
        for placement in [
            RealizedPlacement::OneWorkerPerPhysicalCore,
            RealizedPlacement::SharedCore,
            RealizedPlacement::Unpinned,
            RealizedPlacement::Unobservable(PlacementBlindSpot::AffinityQueryUnsupported),
            RealizedPlacement::Unobservable(PlacementBlindSpot::AffinityQueryFailed),
            RealizedPlacement::Unobservable(PlacementBlindSpot::TopologyUndetected),
        ] {
            let expected = match placement {
                RealizedPlacement::OneWorkerPerPhysicalCore => true,
                RealizedPlacement::SharedCore
                | RealizedPlacement::Unpinned
                | RealizedPlacement::Unobservable(_) => false,
            };
            assert_eq!(
                placement.is_one_worker_per_physical_core(),
                expected,
                "{placement:?} is classified wrongly"
            );
        }
    }

    /// Some workers readable, some not -- the case a real pool cannot produce.
    ///
    /// Readability is a property of the *target*, so every worker in a real
    /// pool answers the same way and this mixture is unreachable through
    /// `build`. It is also exactly where round-one review found the aggregator
    /// wrong: it answered `Some(true)` on the strength of the one worker it
    /// could check and threw away the unverified claim next to it. Injected as
    /// evidence for that reason -- unreachable through a real pool would have
    /// meant untested.
    #[test]
    fn an_unreadable_worker_mask_withdraws_the_whole_honesty_verdict() {
        use crate::decode_affinity::ObservedAffinity;
        let honest = WorkerPlacement {
            tid: Some(1),
            attempted_cpu: Some(0),
            attempt: PinAttempt::Applied,
            observed: ObservedAffinity::Cpus(vec![0]),
        };
        let unverifiable = WorkerPlacement {
            tid: Some(2),
            attempted_cpu: Some(1),
            attempt: PinAttempt::Applied,
            observed: ObservedAffinity::QueryFailed("injected".to_string()),
        };
        let lying = WorkerPlacement {
            tid: Some(3),
            attempted_cpu: Some(2),
            attempt: PinAttempt::Applied,
            observed: ObservedAffinity::Cpus(vec![7]),
        };
        let claims_nothing = WorkerPlacement {
            tid: Some(4),
            attempted_cpu: None,
            attempt: PinAttempt::NotRequested,
            observed: ObservedAffinity::Cpus(vec![0, 1, 2, 3]),
        };

        assert_eq!(
            placement_report_is_honest_of(std::slice::from_ref(&honest)),
            Some(true),
            "a single verified worker is a verified pool"
        );
        assert_eq!(
            placement_report_is_honest_of(&[honest.clone(), unverifiable.clone()]),
            None,
            "one worker verified and one unverifiable is not a verified pool -- this is \
             the exact shape that read as `Some(true)` before"
        );
        assert_eq!(
            placement_report_is_honest_of(&[unverifiable.clone(), honest.clone()]),
            None,
            "the verdict must not depend on which worker the loop reaches first"
        );

        // A definite contradiction outranks an unreadable mask: it stays true
        // regardless of how many peers went unchecked, so it must not be
        // downgraded to "cannot tell".
        assert_eq!(
            placement_report_is_honest_of(&[unverifiable.clone(), lying.clone()]),
            Some(false),
            "a worker demonstrably not where it claims is a dishonest report, not an \
             unanswerable one"
        );
        assert_eq!(
            placement_report_is_honest_of(&[honest.clone(), lying]),
            Some(false)
        );

        // The other direction, and the reason the two `None`s are kept apart: a
        // worker that never asked for a pin claims nothing, so it cannot
        // withdraw a verdict its peers support. Collapsing both `None`s would
        // make every partially pinned pool permanently unanswerable.
        assert_eq!(
            placement_report_is_honest_of(&[honest, claims_nothing.clone()]),
            Some(true),
            "a worker that claims no pin must not sink a verdict the pinned workers earned"
        );
        assert_eq!(
            placement_report_is_honest_of(&[claims_nothing]),
            None,
            "claiming nothing is not honest success"
        );
        assert_eq!(
            placement_report_is_honest_of(&[]),
            None,
            "no evidence is not success"
        );
        assert_eq!(
            placement_report_is_honest_of(&[unverifiable]),
            None,
            "an unverified claim on its own is unanswerable"
        );
    }

    /// A worker that was never asked to pin claims nothing, and "claims
    /// nothing" must not be scored as honest success -- otherwise an entirely
    /// unpinned pool would report a clean honesty verdict.
    #[test]
    fn an_unpinned_pool_is_observed_to_make_no_placement_claim() {
        let pools = SpmdDecodePools::build_with_schedule(
            &[NodeShard {
                index: 0,
                cpus: Vec::new(),
                workers: 2,
            }],
            DecodeSchedule::Fixed,
            false,
        );
        assert!(
            pools
                .worker_placements()
                .iter()
                .all(|placement| placement.attempt == PinAttempt::NotRequested),
            "an empty CPU set must attempt no pins ({:?})",
            pools.worker_placements()
        );
        assert_eq!(pools.realized_placement(), RealizedPlacement::Unpinned);
        assert_eq!(
            pools.placement_report_is_honest(),
            None,
            "a pool that claims no placement has no honesty verdict to give"
        );
        pools.shutdown();
    }

    /// Read the calling *thread's* actual affinity mask from `/proc`.
    ///
    /// `/proc/thread-self` is the calling thread's own task directory -- i.e.
    /// `/proc/self/task/<tid>` -- so this is per-TID kernel state, not the
    /// process-wide mask in `/proc/self/status`. The distinction is the whole
    /// point of this control and is asserted below rather than assumed.
    #[cfg(target_os = "linux")]
    fn actual_mask_of_calling_thread() -> Option<(String, Vec<usize>)> {
        let tid = std::fs::read_link("/proc/thread-self")
            .ok()?
            .to_string_lossy()
            .into_owned();
        let status = std::fs::read_to_string("/proc/thread-self/status").ok()?;
        let list = status
            .lines()
            .find_map(|line| line.strip_prefix("Cpus_allowed_list:"))?;
        Some((tid, onnx_runtime_hostmon::parse_cpu_list(list)?))
    }

    /// Pin one real thread per CPU and return what the kernel says each one is
    /// actually confined to.
    #[cfg(target_os = "linux")]
    fn actual_masks_of_threads_pinned_to(cpus: &[usize]) -> Option<Vec<(String, Vec<usize>)>> {
        let handles: Vec<_> = cpus
            .iter()
            .map(|&cpu| {
                thread::spawn(move || {
                    crate::decode_affinity::pin_current_thread_to_cpu(cpu).ok()?;
                    actual_mask_of_calling_thread()
                })
            })
            .collect();
        let mut observed = Vec::with_capacity(cpus.len());
        for handle in handles {
            observed.push(handle.join().ok()??);
        }
        Some(observed)
    }

    /// The distinct-physical-core predicate, evaluated over masks the kernel
    /// reports rather than over what a pool claims. Mirrors
    /// `planned_placement_is_one_worker_per_physical_core`, which reads `worker_cpus`.
    #[cfg(target_os = "linux")]
    fn distinct_cores_from_actual_masks(
        masks: &[(String, Vec<usize>)],
        cores: &crate::core_topology::CoreTopology,
    ) -> Option<bool> {
        let mut seen = std::collections::BTreeSet::new();
        for (_, mask) in masks {
            // A pinned thread is confined to exactly one CPU. Anything wider
            // means the pin did not take, and the question is unanswerable.
            let [cpu] = mask.as_slice() else {
                return None;
            };
            let core: Vec<usize> = cores
                .siblings_of(*cpu)
                .map_or_else(|| vec![*cpu], <[usize]>::to_vec);
            if !seen.insert(core) {
                return Some(false);
            }
        }
        Some(true)
    }

    /// Negative control against **actual TID masks**, not reported ones.
    ///
    /// `the_planner_reports_a_shared_core_as_a_defect` above builds a pool and
    /// asks it about `worker_cpus` -- the pool's own claim. That is the exact
    /// quantity #1792 showed can be wrong: the EP printed
    /// `requested=16 realized=16 as_requested` while running 16 workers on 8
    /// physical cores, because the count was honest and placement was never
    /// reported at all. A control built on the claim cannot catch a wrong claim.
    ///
    /// This one pins real threads and reads what the kernel says about each
    /// TID, then runs the same distinct-core predicate over that. Both arms are
    /// present deliberately: the shared-core arm must report a defect, and the
    /// distinct-core arm must not, so a predicate that simply always answers
    /// `Some(false)` fails here rather than looking like a working detector.
    #[test]
    #[cfg(target_os = "linux")]
    fn actual_thread_masks_sharing_one_core_are_reported_as_a_defect() {
        let cores = match crate::core_topology::require_host_for_placement() {
            Ok(cores) => cores,
            Err(reason) => {
                eprintln!("skipping actual-mask control: {reason}");
                return;
            }
        };

        // Pick the sibling pair out of the CPUs this process may actually run
        // on. Taking `cores()[0]` blindly picks CPUs the host has but a
        // `taskset`ed or containerised runner cannot use, `environment_can_pin`
        // then refuses, and the control skips -- on the *most common* way this
        // suite is run. A skip that is invisible and environment-triggered is
        // the same fail-open shape this change exists to remove.
        let allowed: std::collections::BTreeSet<usize> = crate::decode_affinity::allowed_cpus()
            .unwrap_or_else(|| (0..cores.logical_count()).collect())
            .into_iter()
            .collect();
        let shared = cores.cores().iter().find_map(|group| {
            let mut inside = group.iter().copied().filter(|cpu| allowed.contains(cpu));
            match (inside.next(), inside.next()) {
                (Some(first), Some(second)) => Some([first, second]),
                _ => None,
            }
        });
        let Some(shared) = shared else {
            eprintln!(
                "skipping actual-mask control: no physical core has two SMT siblings inside \
                 this process's allowed set {allowed:?}, so the bad layout is unrepresentable \
                 here"
            );
            return;
        };
        if !environment_can_pin(&shared) {
            eprintln!(
                "skipping actual-mask control: this environment refuses to pin to {shared:?}"
            );
            return;
        }

        let observed = match actual_masks_of_threads_pinned_to(&shared) {
            Some(observed) => observed,
            // A refused pin is an environment fact, not a defect. It cannot be
            // silently tolerated further down, because an unpinned thread has a
            // wide mask and would answer `None`, so state it here.
            None => {
                eprintln!("skipping actual-mask control: threads could not be pinned");
                return;
            }
        };

        // This must be per-thread state. If `/proc/thread-self` were resolving
        // to process-wide data, every mask would equal the process mask and the
        // control would be measuring nothing -- the same "reading the wrong
        // thing" failure the control exists to catch, one level down.
        let process_mask = crate::decode_affinity::allowed_cpus().unwrap_or_default();
        for (tid, mask) in &observed {
            assert_eq!(
                mask.len(),
                1,
                "thread {tid} was pinned to a single CPU but the kernel reports mask {mask:?}"
            );
            assert_ne!(
                mask, &process_mask,
                "thread {tid} reports the process-wide mask, so this control is reading \
                 `/proc/self/status`-equivalent data and is not a per-TID check at all"
            );
        }

        assert_eq!(
            distinct_cores_from_actual_masks(&observed, cores),
            Some(false),
            "threads whose kernel-reported masks are {observed:?} occupy the SMT siblings \
             of one physical core, and the placement predicate must call that a defect"
        );

        // Positive arm: two genuinely distinct cores must not be reported as a
        // defect, otherwise the assertion above is satisfied by a predicate
        // that is simply always false.
        let distinct: Vec<usize> = cores
            .cores()
            .iter()
            .filter_map(|group| group.iter().copied().find(|cpu| allowed.contains(cpu)))
            .take(2)
            .collect();
        if distinct.len() < 2 || !environment_can_pin(&distinct) {
            eprintln!(
                "skipping the positive arm of the actual-mask control: two distinct physical \
                 cores are not pinnable inside {allowed:?}"
            );
            return;
        }
        let Some(observed) = actual_masks_of_threads_pinned_to(&distinct) else {
            eprintln!(
                "skipping the positive arm of the actual-mask control: threads could not be \
                 pinned to {distinct:?}"
            );
            return;
        };
        assert_eq!(
            distinct_cores_from_actual_masks(&observed, cores),
            Some(true),
            "threads on distinct physical cores {distinct:?} were reported as sharing \
             one, so the predicate answers `false` regardless of input and the negative \
             arm above proves nothing"
        );

        // Say what was actually checked. Under `--nocapture` -- or in the
        // captured output libtest replays for a *failing* test -- every exit
        // from this test is either this line or an explicit skip line, so a run
        // with neither did not execute.
        //
        // Under CI's default capture a passing test prints nothing either way,
        // so this does **not** distinguish "ran" from "skipped" in a green CI
        // log. Nothing can: the only capture-proof anti-vacuity signal is a
        // failure, which is why the *topology* branch panics instead of
        // printing. The two skips left here are genuinely environmental -- no
        // SMT sibling inside the allowed set, or a sandbox that refuses
        // `sched_setaffinity` -- and are stated rather than silent so they are
        // recoverable from a local `--nocapture` run.
        eprintln!(
            "actual-mask control ran: shared-core arm on {shared:?}, distinct-core arm on \
             {distinct:?}"
        );
    }

    /// A pool that pinned only half its workers has not placed one worker per
    /// core, and must not be scored on the half it did pin.
    ///
    /// This is the `.flatten()` trap: dropping the `None` slots and judging the
    /// remainder lets a pool with an entirely unplaced node report healthy --
    /// count honest, placement unexamined, the exact shape of #1729.
    #[test]
    fn a_partially_pinned_pool_is_not_scored_on_its_pinned_half() {
        let cores = match crate::core_topology::require_host_for_placement() {
            Ok(cores) => cores,
            Err(reason) => {
                eprintln!("skipping partial-pin scoring check: {reason}");
                return;
            }
        };
        if cores.core_count() < 2 {
            return;
        }
        let spread = crate::decode_affinity::order_pin_targets(
            &(0..cores.logical_count()).collect::<Vec<_>>(),
            Some(cores),
        );
        if !environment_can_pin(&spread[..2]) {
            return;
        }
        // One shard pinned to distinct cores, one shard deliberately free.
        let pools = SpmdDecodePools::build_with_schedule(
            &[
                NodeShard {
                    index: 0,
                    cpus: spread[..2].to_vec(),
                    workers: 2,
                },
                NodeShard {
                    index: 1,
                    cpus: Vec::new(),
                    workers: 1,
                },
            ],
            DecodeSchedule::Fixed,
            false,
        );
        // The property under test needs a *mixed* pool. Assert that premise
        // rather than an exact pinned count: how many pins the kernel grants is
        // an environment fact, and encoding it here would make the test report
        // on the sandbox instead of on the predicate.
        let placed = pools
            .worker_cpus()
            .iter()
            .filter(|cpu| cpu.is_some())
            .count();
        assert!(
            placed > 0 && placed < pools.worker_cpus().len(),
            "this test needs a partially pinned pool, but got {:?}",
            pools.worker_cpus()
        );
        assert_eq!(
            pools.planned_placement_is_one_worker_per_physical_core(),
            Some(false),
            "a pool with an unplaced worker must not report one-per-core on the \
             strength of the workers it did place"
        );
        pools.shutdown();
    }

    /// A pin target the kernel refuses must be retracted, not reported.
    ///
    /// Otherwise `worker_cpus()` reports intent as achievement -- a pool that
    /// failed to place a single worker would still present a tidy one-per-core
    /// layout. That is the unverified-label defect this mechanism exists to
    /// detect, reproduced inside the detector.
    #[test]
    fn a_pin_that_fails_is_retracted_rather_than_reported() {
        // A CPU id far past any plausible host: the pin call rejects it, so the
        // worker spawns and runs unpinned.
        let unpinnable = 1 << 20;
        if crate::decode_affinity::allowed_cpus().is_some_and(|cpus| cpus.contains(&unpinnable)) {
            return;
        }
        let pools = SpmdDecodePools::build_with_schedule(
            &[NodeShard {
                index: 0,
                cpus: vec![unpinnable],
                workers: 1,
            }],
            DecodeSchedule::Fixed,
            false,
        );
        assert_eq!(
            pools.worker_cpus(),
            &[None],
            "a worker whose pin was rejected must report no placement, not the \
             cpu it failed to reach"
        );
        assert_eq!(
            pools.planned_placement_is_one_worker_per_physical_core(),
            None,
            "a pool that placed nothing makes no placement claim"
        );
        pools.shutdown();
    }

    /// An unpinned pool has no placement claim to check, and must not be
    /// reported as a defect.
    #[test]
    fn an_unpinned_pool_makes_no_placement_claim() {
        let pools = SpmdDecodePools::build_with_schedule(
            &[NodeShard {
                index: 0,
                cpus: Vec::new(),
                workers: 2,
            }],
            DecodeSchedule::Fixed,
            false,
        );
        assert!(
            pools.worker_cpus().iter().all(Option::is_none),
            "an empty CPU set must leave every worker unpinned"
        );
        assert_eq!(
            pools.planned_placement_is_one_worker_per_physical_core(),
            None
        );
        pools.shutdown();
    }

    /// `node_shards` must *consult* the explicit request, not merely have a
    /// correct helper sitting next to it.
    ///
    /// This is the anti-vacuity guard for #1792. Every other test here also
    /// passes against the unfixed code, because the bug was never in the
    /// helpers -- it was that the builder never called them. Deleting the early
    /// return in `node_shards_with` fails this and only this.
    #[test]
    fn node_shards_consults_the_explicit_request_before_default_placement() {
        let sentinel = NodeShard {
            index: 7,
            cpus: vec![100, 101],
            workers: 2,
        };
        let shards = node_shards_with(16, |total| {
            assert_eq!(total, 16, "the worker count must reach the request");
            Some(vec![sentinel.clone()])
        });
        assert_eq!(
            shards.len(),
            1,
            "an honored request must replace default placement outright"
        );
        assert_eq!(shards[0].cpus, sentinel.cpus);
        assert_eq!(shards[0].index, 7);
        assert_eq!(shards[0].workers, 2);
    }

    /// ...and with no request, placement is what it was before this change.
    /// That is the regression that would matter most: routing everything
    /// through the parser would silently unpin the default pool, because
    /// `DecodeAffinity::parse(None)` is `Off`.
    #[test]
    fn no_explicit_request_leaves_default_placement_intact() {
        let defaulted = node_shards_with(16, |_| None);
        // Whatever this host's topology, the default path never returns an
        // empty schedule or a pool with no workers.
        assert!(!defaulted.is_empty());
        assert!(defaulted.iter().map(|s| s.workers).sum::<usize>() > 0);
        // Deferring must reach the real placement policy, not the unpinned
        // single shard `off` produces: on any host with a known allowed set the
        // default path pins.
        if crate::decode_affinity::allowed_cpus().is_some_and(|cpus| !cpus.is_empty()) {
            assert!(
                defaulted.iter().any(|shard| !shard.cpus.is_empty()),
                "default placement pins when the allowed set is known"
            );
        }
    }

    #[test]
    fn reserve_single_group_headroom_is_a_noop_when_headroom_exists_or_affinity_unknown() {
        // Requested workers < allowed CPUs: genuine headroom already exists, so
        // the count is unchanged (the numa-split / flat paths are untouched too).
        assert_eq!(reserve_single_group_headroom(16, 32, 0), 16);
        assert_eq!(reserve_single_group_headroom(31, 32, 0), 31);
        // allowed_count == 0 means the allowed set is unknown; workers run
        // unpinned and cannot occupy every core, so nothing is capped.
        assert_eq!(reserve_single_group_headroom(32, 0, 0), 32);
        assert_eq!(reserve_single_group_headroom(1, 0, 0), 1);
    }

    #[test]
    fn reserve_split_headroom_reserves_one_cpu_per_node_only_when_fully_subscribed() {
        // Both nodes fully subscribed (workers == node CPUs): each is reduced by
        // one so every socket keeps a free core for the dispatcher, whichever
        // node it lands on. A node with existing headroom is left untouched.
        let mut shards = vec![
            NodeShard {
                index: 0,
                cpus: (0..16).collect(),
                workers: 16,
            },
            NodeShard {
                index: 1,
                cpus: (16..32).collect(),
                workers: 10,
            },
        ];
        reserve_split_headroom(&mut shards);
        assert_eq!(shards[0].workers, 15);
        assert_eq!(shards[1].workers, 10);
        // CPU lists are preserved; only the pinned-worker count shrinks, so the
        // reserved (highest-index) CPU of a capped node stays unpinned.
        assert_eq!(shards[0].cpus.len(), 16);
    }

    #[test]
    fn reserve_split_headroom_floors_at_one_worker_per_node() {
        let mut shards = vec![NodeShard {
            index: 0,
            cpus: vec![7],
            workers: 1,
        }];
        reserve_split_headroom(&mut shards);
        assert_eq!(shards[0].workers, 1);
    }

    #[test]
    fn node_row_lengths_split_proportionally_and_cover_all_rows() {
        let pool = two_group_pool();
        assert_eq!(pool.node_row_lengths(100), vec![50, 50]);
        assert_eq!(pool.node_row_lengths(101), vec![50, 51]);
        assert_eq!(pool.node_row_lengths(1), vec![0, 1]);
        assert_eq!(pool.node_row_lengths(0), vec![0, 0]);
    }

    #[test]
    fn worker_row_segments_are_disjoint_and_cover_every_row() {
        let pool = two_group_pool();
        let n = 37usize;
        let segments = pool.worker_row_segments(n);
        assert_eq!(segments.len(), pool.total_workers());
        // Contiguous, non-overlapping, covering exactly 0..n.
        let mut expected_start = 0;
        for (start, len) in &segments {
            assert_eq!(*start, expected_start);
            expected_start += len;
        }
        assert_eq!(expected_start, n);
    }

    #[test]
    fn worker_row_segments_aligned_snaps_interior_boundaries_and_covers_every_row() {
        // Every interior boundary must be a multiple of `align`; the segments
        // must still be contiguous, disjoint, and cover exactly 0..n. This is
        // the invariant the MLAS SQNBit decode shard path relies on to keep each
        // N-tile whole (and thus bit-identical to the full-width call).
        let pool = single_group_pool(3);
        for &align in &[4usize, 16] {
            for &n in &[97usize, 128, 151936, 1, 0, 5, 17] {
                let segments = pool.worker_row_segments_aligned(n, align);
                assert_eq!(segments.len(), pool.total_workers());
                let mut expected_start = 0;
                for (index, &(start, len)) in segments.iter().enumerate() {
                    assert_eq!(start, expected_start, "n={n} align={align} seg {index}");
                    // Interior boundaries (every start past the first) must be
                    // align-aligned; the final segment may end at an unaligned n.
                    assert_eq!(
                        start % align,
                        0,
                        "n={n} align={align}: segment start {start} not aligned"
                    );
                    expected_start += len;
                }
                assert_eq!(expected_start, n, "n={n} align={align}: must cover 0..n");
            }
        }
    }

    #[test]
    fn worker_row_segments_aligned_is_identity_for_align_one() {
        let pool = two_group_pool();
        for &n in &[0usize, 1, 37, 100, 101] {
            assert_eq!(
                pool.worker_row_segments_aligned(n, 1),
                pool.worker_row_segments(n),
                "align=1 must reproduce the unaligned split (n={n})"
            );
        }
    }

    #[test]
    fn work_stealing_segments_preserve_coarse_default_tiles() {
        let pool = single_group_pool_with_schedule(4, DecodeSchedule::Steal);
        let segments = pool.output_column_segments(2048, 16);
        assert!(
            segments.len() >= pool.total_workers(),
            "work-stealing mode must expose at least one stealable tile per worker"
        );
        let mut expected_start = 0;
        for (start, len) in segments {
            assert_eq!(start, expected_start);
            assert_eq!(start % 16, 0);
            assert!(len >= MIN_STEAL_OUTPUTS_PER_TASK || start + len == 2048);
            expected_start += len;
        }
        assert_eq!(expected_start, 2048);
    }

    #[test]
    fn dispatch_output_rows_fans_out_under_a_narrow_ambient_rayon_pool() {
        // The regression guard for the call site itself: `dispatch_output_rows`
        // must decide serial-vs-parallel from *its own* `total_workers`, not
        // from whatever Rayon pool happens to be installed on the calling
        // thread. Sizing it from the ambient width means a narrow ambient pool
        // (`RAYON_NUM_THREADS=1`, or any caller inside a one-thread install)
        // silently collapses the whole dispatch onto the dispatcher thread
        // while every SPMD worker sits spinning -- measured at 14.2x on the
        // qwen int4 decode loop.
        //
        // Reverting the call site to the ambient-width rule turns this test
        // red; the `output_chunk_len_for` unit tests alone would not notice.
        let pool = single_group_pool(4);
        assert!(
            pool.total_workers > 1,
            "the test pool must be wide enough to fan out"
        );
        let n = 4096usize;
        let k = 1024usize;

        let threads: std::sync::Mutex<std::collections::HashSet<thread::ThreadId>> =
            std::sync::Mutex::new(std::collections::HashSet::new());
        let compute = |output_start: usize, outputs: &mut [f32]| {
            threads
                .lock()
                .expect("thread-id set")
                .insert(thread::current().id());
            for (offset, out) in outputs.iter_mut().enumerate() {
                *out = (output_start + offset) as f32;
            }
        };

        // The ambient pool must be genuinely one thread wide *on the thread that
        // calls the dispatch*, which is what `install` guarantees.
        let narrow = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("one-thread ambient pool");
        let mut sharded = vec![0.0f32; n];
        narrow.install(|| pool.dispatch_output_rows(&mut sharded, k, &compute));

        // A serial short-circuit runs `compute` exactly once, on one thread; a
        // fan-out runs it once per worker. Counting distinct threads therefore
        // separates the two regardless of which thread drove the dispatch.
        let observed = threads.lock().expect("thread-id set").len();
        assert!(
            observed > 1,
            "a {}-worker SPMD pool must still fan out when the ambient Rayon \
             pool is one thread wide, but `compute` ran on {observed} thread(s)",
            pool.total_workers
        );
        for (index, value) in sharded.iter().enumerate() {
            assert_eq!(*value, index as f32, "row {index} was not written once");
        }
    }

    /// The regression guard for #1746. `bound_process_to_decode_budget` confines
    /// the process to exactly `N` CPUs, so `reserve_single_group_headroom(N, N)`
    /// always fires and leaves `N-1` pinned workers. Without a dispatcher-owned
    /// shard the user's budget buys `N-1` compute lanes -- and at `N=2` that is
    /// one lane, which trips the `total_workers <= 1` serial short-circuit and
    /// makes `ONNX_GENAI_CPU_DECODE_THREADS=2` indistinguishable from `=1`.
    ///
    /// Deleting the dispatcher shard turns this red at every width.
    #[test]
    fn a_dispatcher_owned_shard_restores_the_requested_width() {
        for threads in 1..=4usize {
            let pool = dispatcher_shard_pool(threads);
            assert_eq!(
                pool.total_workers(),
                threads + 1,
                "{threads} pinned workers plus the dispatcher must be {} lanes",
                threads + 1
            );
            assert_eq!(
                pool.node_thread_counts.iter().sum::<usize>(),
                threads,
                "only {threads} threads may be spawned; the extra lane is the dispatcher"
            );
            pool.shutdown();
        }
    }

    /// The `N=2` case specifically: one pinned worker plus the dispatcher must
    /// actually execute on two threads, not take the serial short-circuit.
    #[test]
    fn a_two_lane_budget_fans_out_across_the_dispatcher_and_one_worker() {
        let pool = dispatcher_shard_pool(1);
        let n = 4096usize;
        let k = 1024usize;
        assert!(
            output_chunk_len_for(pool.total_workers(), n, k) < n,
            "the shape must be past the serial gate for this to test fan-out"
        );

        let threads: std::sync::Mutex<std::collections::HashSet<thread::ThreadId>> =
            std::sync::Mutex::new(std::collections::HashSet::new());
        let compute = |output_start: usize, outputs: &mut [f32]| {
            threads
                .lock()
                .expect("thread-id set")
                .insert(thread::current().id());
            for (offset, out) in outputs.iter_mut().enumerate() {
                *out = (output_start + offset) as f32;
            }
        };
        let mut sharded = vec![0.0f32; n];
        pool.dispatch_output_rows(&mut sharded, k, &compute);

        let observed = threads.lock().expect("thread-id set").len();
        assert_eq!(
            observed, 2,
            "a one-worker pool with a dispatcher shard must compute on two \
             threads, but `compute` ran on {observed}"
        );
        for (index, value) in sharded.iter().enumerate() {
            assert_eq!(*value, index as f32, "row {index} was not written once");
        }
        pool.shutdown();
    }

    /// The dispatcher's shard must cover its rows exactly once alongside the
    /// worker shards -- an off-by-one in the partition would silently drop or
    /// double-write rows rather than fail loudly.
    #[test]
    fn a_dispatcher_owned_shard_covers_its_rows_exactly_once() {
        for threads in 1..=4usize {
            for n in [1usize, 7, 64, 1000, 4096] {
                let pool = dispatcher_shard_pool(threads);
                let segments = pool.worker_row_segments(n);
                assert_eq!(
                    segments.len(),
                    threads + 1,
                    "one segment per lane including the dispatcher"
                );
                let mut covered = vec![0u32; n];
                for (start, len) in segments {
                    for hits in &mut covered[start..start + len] {
                        *hits += 1;
                    }
                }
                assert!(
                    covered.iter().all(|&hits| hits == 1),
                    "threads={threads} n={n}: rows must be covered exactly once"
                );
                pool.shutdown();
            }
        }
    }

    /// A panic in the *dispatcher's* shard must still complete the barrier
    /// before unwinding.
    ///
    /// The published `Job` borrows the closure off the dispatcher's stack frame
    /// and the workers are already running against it, so unwinding straight out
    /// of the inline shard would free the pointee while they still read it --
    /// a use-after-free, not merely a hang. The `catch_unwind` in `dispatch`
    /// exists for this; removing it makes this test fail under Miri.
    #[test]
    fn a_panic_in_the_dispatcher_shard_still_waits_for_the_workers() {
        let pool = dispatcher_shard_pool(2);
        let dispatcher_lane = pool.total_workers() - 1;
        let finished = Arc::new(AtomicUsize::new(0));

        let observed = {
            let finished = Arc::clone(&finished);
            let job = move |global_index: usize| {
                if global_index == dispatcher_lane {
                    panic!("dispatcher shard failed");
                }
                // Give the dispatcher every chance to unwind first.
                thread::sleep(Duration::from_millis(20));
                finished.fetch_add(1, Ordering::Release);
            };
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pool.dispatch(&job)))
        };

        assert!(
            observed.is_err(),
            "the dispatcher's panic must propagate to the caller"
        );
        assert_eq!(
            finished.load(Ordering::Acquire),
            dispatcher_lane,
            "every worker shard must have completed before the panic escaped"
        );
        pool.shutdown();
    }

    /// The dispatcher shard is only claimed on the single-group layout, and only
    /// when the headroom reservation actually took a worker away. A budget with
    /// spare CPUs must be unchanged, or it would over-subscribe the request.
    /// `decode_width` must never force the pool to build. Reading it is what a
    /// harness does *before* deciding whether the run is trustworthy, so if the
    /// read itself resolved the path it would both change process behaviour and
    /// destroy the "not resolved yet" signal that keeps an early assertion from
    /// passing vacuously.
    #[test]
    fn reading_the_decode_width_does_not_resolve_the_decode_path() {
        let before = POOLS.get().is_some();
        let width = decode_width();
        assert_eq!(
            POOLS.get().is_some(),
            before,
            "decode_width must not build the pool as a side effect"
        );
        if !before {
            assert_eq!(width.realized, None);
            assert!(
                !width.is_as_requested(),
                "an unresolved width must not report as satisfied, or a harness \
                 that asserts too early passes for the wrong reason"
            );
        }
    }

    /// The whole point of the read: a request that was silently reduced must not
    /// report as satisfied. Covers the reduction Roy's `t=N` rows could not see.
    #[test]
    fn a_reduced_width_does_not_report_as_requested() {
        let reduced = DecodeWidth {
            requested: Some(16),
            realized: Some(15),
            path: "spmd-pool",
        };
        assert!(
            !reduced.is_as_requested(),
            "16 requested and 15 realized is exactly the case a t=16 row would \
             otherwise label as a width-16 measurement"
        );

        let honored = DecodeWidth {
            requested: Some(16),
            realized: Some(16),
            path: "spmd-pool",
        };
        assert!(honored.is_as_requested());

        // Dropping to the flat path is the other silent reduction: the width
        // may even look larger, but it is not the pool that was asked for.
        let flat = DecodeWidth {
            requested: Some(2),
            realized: Some(32),
            path: "flat",
        };
        assert!(!flat.is_as_requested());
    }

    /// An unresolved width must stay unsatisfied no matter which half is
    /// missing, so a harness cannot pass by asserting before it has decoded.
    #[test]
    fn a_half_known_width_is_never_satisfied() {
        for (requested, realized) in [(Some(4), None), (None, Some(4)), (None, None)] {
            let width = DecodeWidth {
                requested,
                realized,
                path: "unresolved",
            };
            assert!(
                !width.is_as_requested(),
                "requested={requested:?} realized={realized:?} must not report satisfied"
            );
        }
    }

    /// Every op-observation is either a spin hit or a park-wake, exactly once
    /// per spawned worker per dispatch.
    ///
    /// This is the assertion the counters exist for: it pins the scheduler's
    /// accounting without measuring a single duration, so it cannot pass or fail
    /// because of what else is running on the runner. A miscounted branch --
    /// counting a park at sleep time instead of at wake time, or missing the
    /// spin-window return -- breaks the identity immediately, while a timing
    /// test would still pass.
    #[test]
    fn every_op_observation_is_counted_exactly_once_as_a_spin_hit_or_a_park() {
        let workers = 3usize;
        let pool = single_group_pool(workers);
        let ops = 16u64;
        for _ in 0..ops {
            pool.dispatch_index_tasks(workers, &|_| {});
        }
        // Read while no dispatch is in flight: `dispatch` returns only after
        // every worker has acknowledged, so all of this op's increments have
        // landed and none of the next op's can have.
        let counters = pool.counters();
        assert_eq!(
            counters.dispatches, ops,
            "every dispatch reached the barrier"
        );
        assert_eq!(
            counters.inline_dispatches, 0,
            "nothing should degrade inline"
        );
        assert_eq!(
            counters.spin_hits + counters.parks,
            ops * workers as u64,
            "each of {workers} workers must observe each of {ops} ops exactly \
             once, as either a spin hit or a park-wake, but got \
             {} spin hits and {} parks",
            counters.spin_hits,
            counters.parks,
        );
        pool.shutdown();
    }

    /// RAII scope for [`FORCE_WORKER_PROFILE`], restoring the previous value on
    /// drop -- including on unwind.
    ///
    /// The plain set/build/reset sequence this replaces leaks the override into
    /// every later test on the same thread if anything between the set and the
    /// reset panics, which is precisely the defect `EnvVarGuard` exists to
    /// eliminate for environment variables. The current body happens to reset
    /// before its first assertion, so the leak is unreachable *today*; that is
    /// an argument for making it structurally unreachable rather than for
    /// leaving a future edit to remember.
    struct ForcedWorkerProfile(Option<bool>);

    impl ForcedWorkerProfile {
        fn set(value: bool) -> Self {
            let previous = FORCE_WORKER_PROFILE.with(|cell| cell.replace(Some(value)));
            Self(previous)
        }

        /// Run `build` under the override and drop the guard immediately after.
        /// The gate is read once, on this thread, while the pool is built, so
        /// the override need not outlive the build.
        fn build<T>(self, build: impl FnOnce() -> T) -> T {
            build()
        }
    }

    impl Drop for ForcedWorkerProfile {
        fn drop(&mut self) {
            FORCE_WORKER_PROFILE.with(|cell| cell.set(self.0));
        }
    }

    /// Profiling is a per-pool decision, not a per-process one.
    ///
    /// The gate is read at every build rather than latched into a `OnceLock`,
    /// so this test is a real A/B: the two arms genuinely take different
    /// routes. Latch the gate and the second arm inherits the first's value,
    /// the assertion below compares a configuration against itself, and the
    /// test passes while measuring nothing -- #1736's exact shape.
    #[test]
    fn worker_timings_are_collected_only_when_the_pool_asked_for_them() {
        let workers = 2usize;
        let ops = 8u64;

        let off = ForcedWorkerProfile::set(false).build(|| single_group_pool(workers));
        let on = ForcedWorkerProfile::set(true).build(|| single_group_pool(workers));

        for _ in 0..ops {
            off.dispatch_index_tasks(workers, &|_| {});
            on.dispatch_index_tasks(workers, &|_| {});
        }

        for profile in off.worker_profiles() {
            assert_eq!(
                (profile.timed_ops, profile.wake_ns, profile.work_ns),
                (0, 0, 0),
                "worker {} must pay no clock reads with profiling off",
                profile.worker
            );
        }
        let on_profiles = on.worker_profiles();
        assert_eq!(on_profiles.len(), workers, "one row per spawned worker");
        for profile in on_profiles {
            assert_eq!(
                profile.timed_ops, ops,
                "worker {} ran every op and so must have timed every op",
                profile.worker
            );
            // Only a lower bound: a shard doing nothing can legitimately take
            // near-zero ns, so anything stronger would be a runner-speed
            // assertion. Zero, though, means the accumulator never ran.
            assert!(
                profile.work_ns > 0,
                "worker {} timed {} ops but accumulated no shard time",
                profile.worker,
                profile.timed_ops
            );
        }

        off.shutdown();
        on.shutdown();
    }

    /// Exactly one worker per node retires each op last, so the per-worker
    /// counts sum to the dispatch count.
    ///
    /// An identity, not a measurement -- it holds whatever the runner is doing.
    /// It is what makes an *uneven* distribution interpretable: the total is
    /// fixed, so a worker holding more than its `1/w` share holds it at another
    /// worker's expense.
    #[test]
    fn last_arrivals_sum_to_the_dispatch_count() {
        let workers = 4usize;
        let pool = single_group_pool(workers);
        let ops = 24u64;
        for _ in 0..ops {
            pool.dispatch_index_tasks(workers, &|_| {});
        }

        let profiles = pool.worker_profiles();
        let total: u64 = profiles.iter().map(|p| p.last_arrivals).sum();
        assert_eq!(
            total,
            ops * pool.node_count() as u64,
            "each of {ops} dispatches has exactly one last arriver per node, \
             but the per-worker counts sum to {total}: {:?}",
            profiles.iter().map(|p| p.last_arrivals).collect::<Vec<_>>()
        );
        pool.shutdown();
    }

    /// A shard that is slow on one worker is attributed to *that* worker.
    ///
    /// This is the whole point of the per-worker split: the aggregate counters
    /// cannot tell "worker 1 is late every op" from "all four workers are
    /// slightly slower", and those have different causes and different owners.
    /// Verified by mutation -- attributing to the wrong index, or to whichever
    /// worker happens to call in, fails this and passes
    /// `last_arrivals_sum_to_the_dispatch_count`.
    ///
    /// The sleep is 2 ms against shards that do nothing, and that margin is
    /// fixed -- so this deliberately does **not** assert a clean sweep. Last
    /// arrival is completion order, and an empty-shard worker preempted for
    /// longer than the sleep between its wake and its acknowledgement
    /// legitimately retires last. `cargo test` saturates every core with its
    /// own parallel tests, so on a small runner that is a real scheduling
    /// outcome rather than a defect, and demanding all 8 would fail a *correct*
    /// implementation -- the one thing a test must never do.
    ///
    /// A supermajority plus "is the busiest" still kills every mutation this
    /// test exists for: misattribution to a fixed index leaves the slow worker
    /// at zero, and attributing to whoever acknowledges leaves it at its even
    /// `1/w` share.
    #[test]
    fn the_slow_shard_is_attributed_to_the_worker_that_ran_it() {
        let workers = 4usize;
        let slow = 2usize;
        let pool = single_group_pool(workers);
        let ops = 8u64;
        for _ in 0..ops {
            pool.dispatch_index_tasks(workers, &|task| {
                if task == slow {
                    std::thread::sleep(Duration::from_millis(2));
                }
            });
        }

        let profiles = pool.worker_profiles();
        let counts: Vec<u64> = profiles.iter().map(|p| p.last_arrivals).collect();
        let busiest = profiles
            .iter()
            .enumerate()
            .max_by_key(|(_, p)| p.last_arrivals)
            .map_or(usize::MAX, |(index, _)| index);
        assert_eq!(
            busiest, slow,
            "the deliberately slow worker must own the most last arrivals, got {counts:?}"
        );
        assert!(
            profiles[slow].last_arrivals * 4 >= ops * 3,
            "the deliberately slow worker must own a supermajority of last \
             arrivals, not merely the plurality, got {counts:?}"
        );
        pool.shutdown();
    }

    #[test]
    fn worker_profile_gate_accepts_only_affirmative_values() {
        for raw in ["1", "on", "true", "TRUE", " on ", "True"] {
            assert!(
                parse_worker_profile(Some(raw)),
                "{raw:?} must enable per-worker timing"
            );
        }
        for raw in ["", "0", "off", "false", "yes", "2"] {
            assert!(
                !parse_worker_profile(Some(raw)),
                "{raw:?} must not enable per-worker timing"
            );
        }
        assert!(
            !parse_worker_profile(None),
            "unset must leave per-worker timing off"
        );
    }

    /// A re-entrant dispatch degrades to inline rather than racing the barrier,
    /// and the counter says so.    ///
    /// The pool has one job slot, so a shard closure that dispatches again --
    /// the nested fan-out shape, and the same shape two concurrent sessions
    /// produce -- must run every shard on the calling thread. That guarantee was
    /// previously only a comment; this makes it observable, which is what a
    /// concurrency harness needs to tell "the pool served both" from "one
    /// session ran serially".
    #[test]
    fn a_re_entrant_dispatch_is_counted_inline_rather_than_racing_the_barrier() {
        let workers = 2usize;
        let pool = single_group_pool(workers);
        let inner_runs = AtomicUsize::new(0);
        let outer = |_task: usize| {
            pool.dispatch_index_tasks(workers, &|_| {
                inner_runs.fetch_add(1, Ordering::Relaxed);
            });
        };
        pool.dispatch_index_tasks(workers, &outer);

        let counters = pool.counters();
        assert_eq!(
            counters.dispatches, 1,
            "only the outer dispatch may claim the slot"
        );
        assert_eq!(
            counters.inline_dispatches, workers as u64,
            "each worker's re-entrant dispatch must be declined back to it"
        );
        assert_eq!(
            inner_runs.load(Ordering::Relaxed),
            workers * workers,
            "an inline dispatch still runs every shard, on the caller"
        );
        pool.shutdown();
    }

    /// An idle gap far longer than the blocktime window must park the workers.
    ///
    /// Timing-dependent by nature -- parking *is* a timing behaviour -- but the
    /// margin is deliberate: the gap is an order of magnitude past the window,
    /// so for this to fail a runnable worker would have to be starved of a core
    /// for the whole gap, on a box where the rest of the test binary is
    /// evidently getting scheduled. The complementary direction (back-to-back
    /// dispatches never park) is *not* asserted, because that one really can
    /// fail on a loaded runner: a descheduled worker's window expires while it
    /// is off-CPU.
    #[test]
    fn an_idle_gap_far_longer_than_the_blocktime_parks_the_workers() {
        let workers = 2usize;
        let pool = single_group_pool(workers);
        // Prime, so the count below covers only gapped dispatches.
        pool.dispatch_index_tasks(workers, &|_| {});
        let before = pool.counters();

        let gap = decode_blocktime() * 20 + Duration::from_millis(5);
        let rounds = 4u64;
        for _ in 0..rounds {
            thread::sleep(gap);
            pool.dispatch_index_tasks(workers, &|_| {});
        }

        let after = pool.counters();
        let parks = after.parks - before.parks;
        assert!(
            parks >= rounds,
            "{rounds} gaps of {gap:?} against a {:?} blocktime window produced \
             only {parks} park-wakes across {workers} workers: the window is \
             not closing, so the pool holds cores through idle time",
            decode_blocktime(),
        );

        // Now go idle once more *without* a following dispatch, and re-check the
        // identity. This is the assertion that pins park-at-wake-time: counting
        // a park when the worker goes to sleep passes every check above (each
        // gap's park pairs 1:1 with the op that follows it) and breaks only
        // here, where the pool parks with nothing left to wake it. That is
        // exactly the state a real process sits in between requests, and the
        // state any snapshot of a served process is taken in.
        thread::sleep(gap);
        let idle = pool.counters();
        assert_eq!(
            idle.spin_hits + idle.parks,
            idle.dispatches * workers as u64,
            "a pool parked with no pending op must not have counted the sleep \
             as an observation: {} spin hits + {} parks against {} dispatches",
            idle.spin_hits,
            idle.parks,
            idle.dispatches,
        );
        pool.shutdown();
    }

    /// A futex wake that carries no new op must be counted as spurious and must
    /// not cost the pool the next op.
    ///
    /// The re-check under the futex guard is the pool's no-lost-wakeup argument,
    /// and it was previously untested because a spurious wake cannot be waited
    /// for -- so this manufactures one by waking the sense line without
    /// advancing it, which is precisely the shape a `wake_all` aimed at a
    /// sibling that had not re-armed produces.
    #[test]
    fn a_wake_that_carries_no_new_op_is_counted_spurious_and_loses_nothing() {
        let workers = 2usize;
        let pool = single_group_pool(workers);
        pool.dispatch_index_tasks(workers, &|_| {});

        // Let both workers exhaust the window and park.
        let settle = decode_blocktime() * 20 + Duration::from_millis(5);
        thread::sleep(settle);
        let shared = pool
            .shared
            .as_ref()
            .expect("the fixed schedule keeps state");
        atomic_wait::wake_all(&shared.node_sense[0].0.wake);
        thread::sleep(settle);

        let woken = pool.counters();
        assert!(
            woken.spurious_wakes >= 1,
            "waking the sense line without advancing it must be observed as a \
             spurious wake (expected up to {workers}, got {})",
            woken.spurious_wakes,
        );
        assert_eq!(
            woken.spin_hits + woken.parks,
            woken.dispatches * workers as u64,
            "a spurious wake is not an op-observation and must not be counted \
             as one"
        );

        // The property the re-check exists for: the next op still lands.
        let ran = AtomicUsize::new(0);
        pool.dispatch_index_tasks(workers, &|_| {
            ran.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(
            ran.load(Ordering::Relaxed),
            workers,
            "every shard must still run after a spurious wake"
        );
        pool.shutdown();
    }

    /// Dispatching *at* the blocktime boundary must keep the accounting exact.
    ///
    /// The regression guard for the hole this instrumentation was born with:
    /// when a publish lands between a worker's window expiring and the worker
    /// actually sleeping, the observation belongs to neither phase's obvious
    /// branch and was dropped. It survived the back-to-back and long-gap tests
    /// (both sit far from the boundary) and only showed up when the whole test
    /// binary ran in parallel and widened the race. A gap equal to the window is
    /// the shape that hits it deliberately -- and it is also where a park/wake
    /// tuning sweep spends most of its time, so the counters have to be exact
    /// there or the sweep is measuring its own instrument.
    #[test]
    fn dispatches_at_the_blocktime_boundary_keep_the_accounting_exact() {
        let workers = 2usize;
        let pool = single_group_pool(workers);
        let gap = decode_blocktime();
        let ops = 40u64;
        for _ in 0..ops {
            thread::sleep(gap);
            pool.dispatch_index_tasks(workers, &|_| {});
        }
        let counters = pool.counters();
        assert_eq!(counters.dispatches, ops);
        assert_eq!(
            counters.spin_hits + counters.parks,
            ops * workers as u64,
            "{ops} dispatches at exactly the {gap:?} window lost observations: \
             {} spin hits + {} parks",
            counters.spin_hits,
            counters.parks,
        );
        pool.shutdown();
    }

    #[test]
    fn the_dispatcher_only_claims_a_shard_when_headroom_was_reserved() {
        let group = |workers| {
            vec![NodeShard {
                index: 0,
                cpus: vec![],
                workers,
            }]
        };

        // Fully subscribed: `reserve_single_group_headroom` gave up a worker to
        // keep the dispatcher's CPU free, so the dispatcher computes that lane.
        // Four allowed CPUs over four physical cores (no SMT), so the core
        // budget and the logical budget agree and both reserve one.
        assert_eq!(reserve_single_group_headroom(4, 4, 4), 3);
        assert!(dispatcher_owns_a_shard(&group(3), 4));

        // Headroom already existed: no CPU was reserved, so claiming a shard
        // would make the pool one lane wider than the budget allows. Four
        // workers on a 32-logical/16-core host is far inside both budgets.
        assert_eq!(reserve_single_group_headroom(4, 32, 16), 4);
        assert!(!dispatcher_owns_a_shard(&group(4), 4));

        // A NUMA split keeps the previous behavior: the dispatcher's node is not
        // known at build time, so its shard could land cross-socket.
        let split = vec![
            NodeShard {
                index: 0,
                cpus: vec![],
                workers: 3,
            },
            NodeShard {
                index: 1,
                cpus: vec![],
                workers: 3,
            },
        ];
        assert!(!dispatcher_owns_a_shard(&split, 8));

        let pool = single_group_pool(4);
        assert_eq!(pool.total_workers(), 4);
        assert!(pool.dispatcher_shard.is_none());
        pool.shutdown();
    }

    const BUDGET_LANE_CHILD_ENV: &str = "ONNX_GENAI_TEST_BUDGET_LANE_CHILD";
    const BUDGET_LANE_CONFINE_ENV: &str = "ONNX_GENAI_TEST_BUDGET_LANE_CONFINE";
    const BUDGET_LANE_MARKER: &str = "BUDGET_LANE_RESULT=";
    const BUDGET_LANE_SKIP_MARKER: &str = "BUDGET_LANE_SKIP=";

    /// Total attempts for a single lane child before surfacing a failure. One
    /// nominal attempt plus two retries: the environmental crash is rare, so a
    /// small bound rides through it without masking a persistent problem.
    const BUDGET_LANE_CHILD_MAX_ATTEMPTS: u32 = 3;

    /// End-to-end guard for #1746, across the process boundary the defect lived
    /// in.
    ///
    /// The off-by-one was not in either half. `bound_process_to_decode_budget`
    /// correctly confines the process to `N` CPUs, and
    /// `reserve_single_group_headroom` correctly keeps one allowed CPU free for
    /// the dispatcher. It only appears when they *compose*: the confinement
    /// makes the group fully subscribed, so the reservation always fires and the
    /// budget silently buys `N-1` lanes.
    ///
    /// A subprocess per budget is required, not stylistic. Both halves latch --
    /// `PROCESS_BUDGET_BOUND` is a `OnceLock` and the pool is a `OnceLock` --
    /// and the child mutates process-wide CPU affinity, which would poison the
    /// test runner. Unit tests on either half in isolation cannot see this;
    /// that is precisely how it shipped.
    #[test]
    fn an_explicit_budget_buys_its_full_width_end_to_end() {
        if std::env::var_os(BUDGET_LANE_CHILD_ENV).is_some() {
            return; // The child arm runs as its own test below.
        }
        let available = std::thread::available_parallelism().map_or(1, |n| n.get());
        // A budget of 1 confines the process to one CPU, which `build_from_env`
        // deliberately declines (no core left for the dispatcher), so the
        // smallest budget that builds a pool is 2.
        let budgets: Vec<usize> = [2usize, 4, 8]
            .into_iter()
            .filter(|&n| n <= available)
            .collect();
        if budgets.is_empty() {
            eprintln!("skipped: needs >= 2 CPUs, host reports {available}");
            return;
        }

        for budget in budgets {
            let lane = run_budget_lane_child(budget, None).expect("the unconfined arm never skips");
            let (allowed, nodes, threads, lanes) =
                (lane.allowed, lane.nodes, lane.threads, lane.lanes);
            let (requested, realized, path) = (lane.requested, lane.realized, lane.path.as_str());
            if nodes != 1 {
                // A budget that straddles NUMA nodes takes the split path, which
                // keeps the previous per-node reservation. Not this test's case.
                eprintln!("budget {budget}: skipped, {nodes}-node split layout");
                continue;
            }
            // The introspection must agree with the pool it describes, on the
            // real production path. A `t=N` row is only a width-N measurement if
            // this holds, which is the whole reason the read exists.
            assert_eq!(
                (requested, realized, path),
                (Some(budget), Some(budget), "spmd-pool"),
                "budget {budget}: decode_width must report the realized width and \
                 path, got requested={requested:?} realized={realized:?} path={path}"
            );
            assert_eq!(
                lanes, budget,
                "ONNX_GENAI_CPU_DECODE_THREADS={budget} must buy {budget} compute lanes, \
                 got {lanes} ({threads} pinned threads on {allowed} allowed CPUs)"
            );
            assert!(
                threads >= 1 && threads <= budget,
                "budget {budget}: {threads} pinned threads is outside the budget"
            );
            if allowed == budget {
                // The confinement landed, so the reservation must have fired and
                // the dispatcher must be covering the lane it freed.
                assert_eq!(
                    threads,
                    budget - 1,
                    "budget {budget}: a fully-subscribed group must leave one CPU \
                     free for the dispatcher"
                );
            }
        }
    }

    /// One decoded `BUDGET_LANE_MARKER` line from a lane child.
    struct BudgetLane {
        allowed: usize,
        nodes: usize,
        threads: usize,
        lanes: usize,
        requested: Option<usize>,
        realized: Option<usize>,
        path: String,
    }

    /// Spawns the lane child at `budget`, optionally pre-confining it to
    /// `confine` CPUs, and decodes its one result line.
    /// Is this lane-child exit the known environmental Windows ARM64 crash?
    ///
    /// The child has two completion tokens, a result and a skip, and either one
    /// means it finished its work — so a fault raised after one was printed is
    /// not the environmental crash. Both must therefore be absent before a retry
    /// is justified, which is why this asks the shared classifier about each
    /// marker and requires them to agree.
    fn lane_child_crash_is_environmental(
        success: bool,
        exit_code: Option<i32>,
        stdout: &str,
        stderr: &str,
    ) -> bool {
        [BUDGET_LANE_MARKER, BUDGET_LANE_SKIP_MARKER]
            .iter()
            .all(|marker| {
                crate::test_support::is_environmental_access_violation_crash(
                    success, exit_code, stdout, stderr, marker,
                )
            })
    }

    /// Locks the lane-child retry to *exactly* the known environmental Windows
    /// ARM64 `STATUS_ACCESS_VIOLATION`, so a real regression can never be
    /// retried into a false pass.
    ///
    /// The retry exists because that crash failed a documentation-only PR
    /// (#1772), which no amount of correct code can prevent. The risk it
    /// introduces is the opposite one: a genuine assertion failure that is
    /// silently re-run until it passes. Every non-signature exit below must
    /// therefore be classified non-retryable.
    #[test]
    fn lane_child_retry_covers_only_the_environmental_crash() {
        const AV: Option<i32> = Some(crate::test_support::STATUS_ACCESS_VIOLATION);
        let result = format!("{BUDGET_LANE_MARKER}16,1,16,16,2,2,1");
        let skip = format!("{BUDGET_LANE_SKIP_MARKER}only one CPU online");

        // The signature: unsuccessful, no completion token, no panic, AV code.
        assert!(lane_child_crash_is_environmental(false, AV, "", ""));

        // A real assertion failure must fail fast even with an AV code.
        assert!(!lane_child_crash_is_environmental(
            false,
            AV,
            "",
            "thread 'main' panicked at src/foo.rs:1:1:\nassertion failed",
        ));

        // Success is never retryable, and neither is any other exit code.
        assert!(!lane_child_crash_is_environmental(true, AV, &result, ""));
        assert!(!lane_child_crash_is_environmental(false, Some(1), "", ""));
        assert!(!lane_child_crash_is_environmental(false, None, "", ""));

        // Both completion tokens mean the child finished its work, so a fault
        // after either one is not the environmental crash. This is the part the
        // shared classifier cannot express on its own: it takes a single marker,
        // and a lane child may legitimately emit either.
        assert!(!lane_child_crash_is_environmental(false, AV, &result, ""));
        assert!(!lane_child_crash_is_environmental(false, AV, &skip, ""));

        // Interleaved with the harness's own output, as `--nocapture` produces.
        let noisy = format!("running 1 test\n{skip}\ntest budget_lane_child ... ok");
        assert!(!lane_child_crash_is_environmental(false, AV, &noisy, ""));
    }

    fn run_budget_lane_child(budget: usize, confine: Option<usize>) -> Option<BudgetLane> {
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
        cmd.arg("--exact")
            .arg("decode_spmd::tests::budget_lane_child")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .arg("--ignored")
            .env(BUDGET_LANE_CHILD_ENV, budget.to_string())
            .env(
                crate::kernels::matmul_nbits::DECODE_THREADS_ENV,
                budget.to_string(),
            )
            .env(PERSISTENT_POOL_ENV, "1")
            // The child asserts an exact path label, so any ambient variable that
            // can change which path is chosen has to be cleared, not inherited.
            // `steal` would relabel the path "work-stealing-pool" and fail the
            // assertion for a reason unrelated to decode width.
            .env_remove(DECODE_SCHEDULE_ENV)
            .env_remove(crate::decode_affinity::DECODE_AFFINITY_ENV);
        if let Some(confine) = confine {
            cmd.env(BUDGET_LANE_CONFINE_ENV, confine.to_string());
        } else {
            cmd.env_remove(BUDGET_LANE_CONFINE_ENV);
        }
        let stdout = {
            // Native Windows ARM64 runners intermittently fault this child during
            // SPMD pool build/teardown with `STATUS_ACCESS_VIOLATION` and empty
            // stderr — not a Rust panic, not one of our assertions. It failed a
            // documentation-only PR (#1772), so it fails PRs that cannot possibly
            // have caused it. Retry only that exact signature; a real assertion
            // failure still fails fast on the first attempt, and on Linux the
            // signature never occurs, so behaviour there is unchanged.
            let mut attempt = 1;
            loop {
                let output = cmd.output().expect("run decode-budget lane child");
                let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
                let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
                let environmental = lane_child_crash_is_environmental(
                    output.status.success(),
                    output.status.code(),
                    &stdout,
                    &stderr,
                );
                if environmental && attempt < BUDGET_LANE_CHILD_MAX_ATTEMPTS {
                    eprintln!(
                        "note: retrying decode-budget lane child (budget={budget}) after \
                         environmental STATUS_ACCESS_VIOLATION crash, attempt {attempt}"
                    );
                    attempt += 1;
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    continue;
                }
                assert!(
                    output.status.success(),
                    "budget {budget} child failed ({}):\nstdout:\n{stdout}\nstderr:\n{stderr}",
                    crate::test_support::child_status_detail(&output.status),
                );
                break stdout;
            }
        };
        if let Some(reason) = stdout.lines().find_map(|line| {
            line.split_once(BUDGET_LANE_SKIP_MARKER)
                .map(|(_, r)| r.trim())
        }) {
            eprintln!("budget {budget}: child skipped: {reason}");
            return None;
        }
        let encoded = stdout
            .lines()
            .find_map(|line| {
                line.split_once(BUDGET_LANE_MARKER)
                    .map(|(_, rest)| rest.trim())
            })
            .unwrap_or_else(|| panic!("budget {budget} child emitted no result:\n{stdout}"));
        let fields: Vec<&str> = encoded.split(',').collect();
        assert_eq!(fields.len(), 7, "malformed lane result {encoded:?}");
        let num = |i: usize| fields[i].parse::<usize>().unwrap();
        // The child encodes an absent width as 0, which is never a valid width.
        let opt = |i: usize| Some(num(i)).filter(|&n| n != 0);
        Some(BudgetLane {
            allowed: num(0),
            nodes: num(1),
            threads: num(2),
            lanes: num(3),
            requested: opt(4),
            realized: opt(5),
            path: fields[6].trim().to_owned(),
        })
    }

    /// The counterpart to [`an_explicit_budget_buys_its_full_width_end_to_end`]:
    /// when the host *cannot* honour the request, the read must say so.
    ///
    /// This is the case that makes the introspection worth having. A published
    /// `t=N` row is a label, not a measurement, and the ways a request gets
    /// silently reduced -- `reserve_single_group_headroom`,
    /// `reserve_split_headroom`, and the single-CPU cpuset branch that drops
    /// decode to the flat path -- all report only through `report_spmd_fallback`,
    /// which is `tracing::debug!`/`NXRT_CALIB_DEBUG`-gated and therefore invisible
    /// in a default benchmark run.
    ///
    /// Requesting 8 lanes inside a 2-CPU cpuset is that situation made
    /// deterministic. The exact realized width is a policy detail this test
    /// deliberately does not pin; what it pins is that the read never claims the
    /// request was honoured when it was not.
    #[test]
    fn a_budget_the_host_cannot_honour_never_reports_as_requested() {
        let available = std::thread::available_parallelism().map_or(1, |n| n.get());
        const CONFINE: usize = 2;
        const REQUEST: usize = 8;
        if available < REQUEST {
            eprintln!("skipped: needs >= {REQUEST} CPUs, host reports {available}");
            return;
        }
        let Some(lane) = run_budget_lane_child(REQUEST, Some(CONFINE)) else {
            return; // No process-wide affinity mask on this host.
        };
        assert!(
            lane.allowed <= CONFINE,
            "the confinement must land, got {} allowed CPUs",
            lane.allowed
        );
        assert_eq!(
            lane.requested,
            Some(REQUEST),
            "the requested width must survive the reduction -- without it there is \
             nothing to compare the realized width against"
        );
        assert!(
            lane.realized.is_none_or(|r| r < REQUEST),
            "a {CONFINE}-CPU cpuset cannot deliver {REQUEST} lanes, but the read \
             reported {:?}",
            lane.realized
        );
        // The whole contract in one line: this must be false whenever the label
        // and the reality disagree.
        let width = DecodeWidth {
            requested: lane.requested,
            realized: lane.realized,
            path: "",
        };
        assert!(
            !width.is_as_requested(),
            "requested={:?} realized={:?} must not report as requested",
            lane.requested,
            lane.realized
        );
        // And the pool, if one was built, must agree with what the read said.
        if lane.path == "spmd-pool" {
            assert_eq!(
                lane.realized,
                Some(lane.lanes),
                "the read must report the pool's own lane count"
            );
        }
        eprintln!(
            "requested={REQUEST} in a {CONFINE}-CPU cpuset -> allowed={} realized={:?} path={}",
            lane.allowed, lane.realized, lane.path
        );
    }

    /// The child arm of [`an_explicit_budget_buys_its_full_width_end_to_end`].
    /// `#[ignore]`d so only the parent's explicit `--ignored --exact` run
    /// executes it; it confines the process's CPU affinity and must not run in
    /// the shared runner.
    #[test]
    #[ignore = "spawned by an_explicit_budget_buys_its_full_width_end_to_end"]
    fn budget_lane_child() {
        let Some(budget) = std::env::var_os(BUDGET_LANE_CHILD_ENV) else {
            return;
        };
        let budget: usize = budget.to_string_lossy().parse().expect("budget");
        // Optional pre-confinement, used by the reduced-width arm to make the
        // host unable to honour the request. Applied before
        // `bound_process_to_decode_budget` so the budget logic sees the smaller
        // cpuset, exactly as an external `taskset` or container limit would
        // present it.
        if let Some(confined) = std::env::var_os(BUDGET_LANE_CONFINE_ENV) {
            let confined: usize = confined.to_string_lossy().parse().expect("confine");
            let cpus: Vec<usize> = (0..confined).collect();
            if let Err(err) = crate::decode_affinity::set_current_thread_affinity(&cpus) {
                // Only Linux implements a process-wide mask, so on other hosts
                // there is no way to manufacture the reduction. Report a skip
                // rather than failing: the arm is about the read's behaviour
                // under reduction, not about affinity support.
                println!("{BUDGET_LANE_SKIP_MARKER}{err}");
                return;
            }
        }
        // `pools()` reads the budget from the environment, so confirm the env the
        // parent set agrees with the budget it asked for. Checked against the
        // *unclamped* accessor, since under confinement the resolved width is
        // deliberately smaller than the request.
        assert_eq!(
            crate::kernels::matmul_nbits::decode_thread_budget(),
            Some(budget),
            "the child's environment must carry the budget the parent requested"
        );
        // Exactly the production sequence: EP `initialize()` bounds the process,
        // then the pool is built lazily at first decode.
        crate::kernels::matmul_nbits::bound_process_to_decode_budget();
        let allowed = crate::decode_affinity::allowed_cpus().map_or(0, |c| c.len());
        // Deliberately the global `pools()` entry rather than `build_from_env`
        // directly: that is what a real decode reaches, and it is the only way
        // `decode_width` sees a resolved path -- so this arm checks the
        // introspection a harness will assert on, not a parallel construction of
        // it that could agree while production disagrees.
        let pool = pools();
        let (nodes, threads, lanes) = pool.map_or((0, 0, 0), |p| {
            (
                p.node_count(),
                p.node_thread_counts.iter().sum::<usize>(),
                p.total_workers(),
            )
        });
        let width = decode_width();
        println!(
            "{BUDGET_LANE_MARKER}{allowed},{nodes},{threads},{lanes},{},{},{}",
            width.requested.unwrap_or(0),
            width.realized.unwrap_or(0),
            width.path,
        );
        shutdown_pools();
    }

    #[test]
    fn dispatch_output_rows_matches_flat_computation() {
        let pool = two_group_pool();
        let n = 101usize;
        let compute = |output_start: usize, outputs: &mut [f32]| {
            for (offset, out) in outputs.iter_mut().enumerate() {
                *out = (output_start + offset) as f32 * 2.5 - 3.0;
            }
        };
        let mut sharded = vec![0.0f32; n];
        pool.dispatch_rows_across_workers(&mut sharded, &compute);
        let mut flat = vec![0.0f32; n];
        compute(0, &mut flat);
        assert_eq!(sharded, flat);
    }

    #[test]
    fn work_stealing_dispatch_output_rows_matches_flat_computation() {
        let pool = single_group_pool_with_schedule(4, DecodeSchedule::Steal);
        let n = 2048usize;
        let compute = |output_start: usize, outputs: &mut [f32]| {
            for (offset, out) in outputs.iter_mut().enumerate() {
                let row = output_start + offset;
                *out = row as f32 * 1.25 - 7.0;
            }
        };
        let mut sharded = vec![0.0f32; n];
        pool.dispatch_output_rows(&mut sharded, 1024, &compute);
        let mut flat = vec![0.0f32; n];
        compute(0, &mut flat);
        assert_eq!(sharded, flat);
    }

    #[test]
    fn dispatch_preserves_per_row_reduction_bit_for_bit() {
        // Mirror the real GEMV: each output row is a full-K f32 dot product. Row
        // sharding must not reorder the per-row accumulation, so the SPMD result
        // must be *byte-for-byte* identical to a single-threaded reference (this
        // is the parity invariant the greedy-token equality relies on).
        let pool = two_group_pool();
        let n = 257usize;
        let k = 320usize;
        // Deterministic pseudo-random-ish weights/activation, mixed signs/scales.
        let activation: Vec<f32> = (0..k)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.031_25)
            .collect();
        let weight = |row: usize, col: usize| -> f32 {
            (((row * 131 + col * 17) % 251) as f32 - 125.0) * 0.007_812_5
        };
        let compute = |output_start: usize, outputs: &mut [f32]| {
            for (offset, out) in outputs.iter_mut().enumerate() {
                let row = output_start + offset;
                let mut acc = 0.0f32;
                for (col, &a) in activation.iter().enumerate() {
                    acc += a * weight(row, col);
                }
                *out = acc;
            }
        };
        let mut sharded = vec![0.0f32; n];
        pool.dispatch_rows_across_workers(&mut sharded, &compute);
        let mut reference = vec![0.0f32; n];
        compute(0, &mut reference);
        assert_eq!(
            sharded.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "row-sharded dispatch must be bit-identical to the serial reference"
        );
    }

    #[test]
    fn dispatch_output_row_blocks_matches_flat_computation() {
        // Fixed-width row blocks (mirrors GroupQueryAttention's per-head output
        // rows): every `row_len`-element row is computed whole on one worker, so
        // the sharded result must equal the single-threaded reference row-for-row
        // and bit-for-bit (rows are independent).
        for (num_rows, row_len) in [
            (28usize, 128usize),
            (3, 128),
            (1, 64),
            (5, 3),
            (37, 1),
            (0, 8),
        ] {
            let pool = two_group_pool();
            let compute = |row_index: usize, row: &mut [f32]| {
                for (offset, out) in row.iter_mut().enumerate() {
                    // Order-sensitive accumulation to catch any row reordering.
                    let mut acc = 0.0f32;
                    for step in 0..=offset {
                        acc += (row_index * 7 + step) as f32 * 0.015_625 - 1.0;
                    }
                    *out = acc;
                }
            };
            let mut sharded = vec![0.0f32; num_rows * row_len];
            pool.dispatch_output_row_blocks(&mut sharded, row_len, num_rows, &compute);
            let mut reference = vec![0.0f32; num_rows * row_len];
            for row in 0..num_rows {
                compute(row, &mut reference[row * row_len..(row + 1) * row_len]);
            }
            assert_eq!(
                sharded.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "row-block dispatch must be bit-identical to the serial reference \
                 (num_rows={num_rows}, row_len={row_len})"
            );
        }
    }

    #[test]
    fn dispatch_is_reusable_across_many_ops() {
        // Exercises the barrier repeatedly: every worker must re-arm and the
        // dispatcher must observe completion each time (regression guard for
        // the sequence/pending protocol).
        let pool = single_group_pool(4);
        for round in 0..200usize {
            let n = 53usize;
            let compute = move |output_start: usize, outputs: &mut [f32]| {
                for (offset, out) in outputs.iter_mut().enumerate() {
                    *out = (round * 1000 + output_start + offset) as f32;
                }
            };
            let mut got = vec![0.0f32; n];
            pool.dispatch_rows_across_workers(&mut got, &compute);
            let mut want = vec![0.0f32; n];
            compute(0, &mut want);
            assert_eq!(got, want, "round {round}");
        }
    }

    #[test]
    fn build_then_immediate_dispatch_never_hangs() {
        // Regression guard: a dispatch issued right after `build` must not race
        // a not-yet-started worker (which would hang the barrier). Rebuild a
        // fresh pool and dispatch across all workers immediately, many times.
        for _ in 0..40usize {
            let pool = single_group_pool(6);
            let n = 61usize;
            let compute = |output_start: usize, outputs: &mut [f32]| {
                for (offset, out) in outputs.iter_mut().enumerate() {
                    *out = (output_start + offset) as f32;
                }
            };
            let mut got = vec![-1.0f32; n];
            pool.dispatch_rows_across_workers(&mut got, &compute);
            let mut want = vec![0.0f32; n];
            compute(0, &mut want);
            assert_eq!(got, want);
        }
    }

    #[test]
    fn place_rows_preserves_bytes() {
        let pool = two_group_pool();
        let n = 7usize;
        let stride = 4usize;
        let src: Vec<u8> = (0..(n * stride) as u8).collect();
        assert_eq!(pool.place_rows(&src, n), src);

        let scales: Vec<f32> = (0..n).map(|row| row as f32 * 0.5).collect();
        assert_eq!(pool.place_rows(&scales, n), scales);
    }

    #[test]
    fn tiny_ops_run_serially_but_correctly() {
        // Below the parallelization threshold the op runs on the dispatcher;
        // the result must still be correct.
        let pool = single_group_pool(8);
        let n = 3usize;
        let compute = |output_start: usize, outputs: &mut [f32]| {
            for (offset, out) in outputs.iter_mut().enumerate() {
                *out = (output_start + offset) as f32;
            }
        };
        let mut got = vec![0.0f32; n];
        pool.dispatch_output_rows(&mut got, 4096, &compute);
        assert_eq!(got, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn panicking_worker_poison_is_reported_without_hanging() {
        let pool = single_group_pool(4);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.dispatch(&|worker| {
                assert_ne!(worker, 2, "intentional SPMD worker panic");
            });
        }));
        let panic = result.expect_err("dispatcher must report a worker panic");
        let message = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or("");
        assert!(
            message.contains("persistent SPMD decode worker 2 panicked")
                && message.contains("pool is poisoned"),
            "unexpected dispatcher diagnostic: {message}"
        );
    }

    /// The last-arrival identity survives a worker panic.
    ///
    /// Every shard panics, so whichever worker retires last does so through
    /// `WorkerCompletion::drop` rather than `complete`. If only the happy path
    /// counted, the sum would be zero here and the documented identity would
    /// hold "except when a worker panicked" -- an exception a harness
    /// subtracting two snapshots has no way to learn about. Making every shard
    /// panic is what removes the timing dependence: it does not matter which
    /// worker is last, only that the last one took the unwind path.
    #[test]
    fn a_panicking_last_arriver_is_still_counted() {
        let pool = single_group_pool(4);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.dispatch(&|worker| panic!("intentional SPMD worker panic on {worker}"));
        }));
        assert!(result.is_err(), "the dispatcher must report the poison");

        let profiles = pool.worker_profiles();
        let total: u64 = profiles.iter().map(|p| p.last_arrivals).sum();
        assert_eq!(
            total,
            1,
            "the one dispatch must have exactly one last arriver even though it \
             unwound, got {:?}",
            profiles.iter().map(|p| p.last_arrivals).collect::<Vec<_>>()
        );
    }

    #[test]
    fn persistence_mode_parses_env_values() {
        // Mode parsing (pool is the default): unset -> On (pool), `0` -> Off
        // (flat path), `1` -> On (pool), `auto` -> Adaptive (calibrated).
        // Whitespace is trimmed; unrecognized values map to On (the pool default).
        assert_eq!(persistence_mode_from_raw(None), PersistenceMode::On);
        assert_eq!(persistence_mode_from_raw(Some("")), PersistenceMode::On);
        assert_eq!(persistence_mode_from_raw(Some("   ")), PersistenceMode::On);
        assert_eq!(persistence_mode_from_raw(Some("0")), PersistenceMode::Off);
        assert_eq!(persistence_mode_from_raw(Some(" 0 ")), PersistenceMode::Off);
        assert_eq!(persistence_mode_from_raw(Some("1")), PersistenceMode::On);
        assert_eq!(persistence_mode_from_raw(Some(" 1 ")), PersistenceMode::On);
        // `auto` (case-insensitive) enables load-adaptive calibration.
        assert_eq!(
            persistence_mode_from_raw(Some("auto")),
            PersistenceMode::Adaptive
        );
        assert_eq!(
            persistence_mode_from_raw(Some(" Auto ")),
            PersistenceMode::Adaptive
        );
        assert_eq!(
            persistence_mode_from_raw(Some("AUTO")),
            PersistenceMode::Adaptive
        );
        // Unknown values map to On (pool), never a surprise flat path.
        assert_eq!(persistence_mode_from_raw(Some("true")), PersistenceMode::On);
        assert_eq!(persistence_mode_from_raw(Some("2")), PersistenceMode::On);

        // The pool is BUILT for `On` (default/`=1`) and `Adaptive` (`=auto`);
        // only `Off` (`=0`) skips building. On is forced (no calibration).
        assert!(pool_mode_builds(persistence_mode_from_raw(Some("1"))));
        assert!(pool_mode_builds(persistence_mode_from_raw(None)));
        assert!(pool_mode_builds(persistence_mode_from_raw(Some("auto"))));
        assert!(!pool_mode_builds(persistence_mode_from_raw(Some("0"))));
        assert!(pool_mode_forces(persistence_mode_from_raw(Some("1"))));
        assert!(pool_mode_forces(persistence_mode_from_raw(None)));
        assert!(!pool_mode_forces(persistence_mode_from_raw(Some("auto"))));
    }

    #[test]
    fn default_selects_pool_without_probing() {
        // The default (unset) must deterministically select the pool with no
        // calibration — the core guarantee of the opt-in adaptive design.
        let mode = persistence_mode_from_raw(None);
        assert_eq!(mode, PersistenceMode::On, "default must be On (pool)");
        assert!(
            pool_mode_forces(mode),
            "default must force the pool (no calibration)"
        );
        assert!(pool_mode_builds(mode), "default must build the pool");
    }

    #[test]
    fn adaptive_flag_enables_calibration() {
        // `=auto` must enable adaptive calibration: build the pool but do NOT
        // force it — the calibrator decides.
        let mode = persistence_mode_from_raw(Some("auto"));
        assert_eq!(mode, PersistenceMode::Adaptive);
        assert!(
            pool_mode_builds(mode),
            "adaptive must build the pool for calibration"
        );
        assert!(
            !pool_mode_forces(mode),
            "adaptive must NOT force the pool — the calibrator picks"
        );
    }

    #[test]
    fn on_and_adaptive_build_the_pool_but_only_on_dispatches_unconditionally() {
        // The pool is built for `On` (the default) and `Adaptive` (`=auto`);
        // `Off` (`=0`) never builds it. `On` always dispatches (no calibration);
        // `Adaptive` lets the calibrator pick.
        assert!(pool_mode_builds(PersistenceMode::On));
        assert!(pool_mode_builds(PersistenceMode::Adaptive));
        assert!(!pool_mode_builds(PersistenceMode::Off));

        assert!(pool_mode_forces(PersistenceMode::On));
        assert!(!pool_mode_forces(PersistenceMode::Adaptive));
        assert!(!pool_mode_forces(PersistenceMode::Off));

        // The default env value (unset) maps to On: built, forced, not calibrated.
        assert!(pool_mode_builds(persistence_mode_from_raw(None)));
        assert!(pool_mode_forces(persistence_mode_from_raw(None)));
        assert!(!pool_mode_builds(persistence_mode_from_raw(Some("0"))));
        assert!(pool_mode_builds(persistence_mode_from_raw(Some("2"))));
        assert!(pool_mode_forces(persistence_mode_from_raw(Some("1"))));
    }

    /// Drive the calibrator from its current phase until it commits, feeding each
    /// step the per-path time it chose. Returns once the committed phase is reached.
    fn drive_to_commit(calib: &mut Calibrator, pool_ns: u64, flat_ns: u64) {
        for _ in 0..100_000 {
            if calib.phase == CalibPhase::Committed {
                return;
            }
            let path = calib.choose();
            let ns = match path {
                AutoPath::Pool => pool_ns,
                AutoPath::Flat => flat_ns,
            };
            calib.record(path, ns);
        }
        panic!("calibrator never reached the committed phase");
    }

    /// Fresh calibrator driven through warmup + one probe with the given per-path
    /// step times; returns the committed decision.
    fn run_one_probe(pool_ns: u64, flat_ns: u64) -> Calibrator {
        let mut calib = Calibrator::new();
        drive_to_commit(&mut calib, pool_ns, flat_ns);
        calib
    }

    #[test]
    fn calibrator_defaults_to_flat_before_any_measurement() {
        // The no-regression baseline: a fresh calibrator's committed path is the
        // flat path, so a host that never lets the pool win runs exactly today's
        // flat decode path.
        let calib = Calibrator::new();
        assert_eq!(calib.committed, AutoPath::Flat);
    }

    #[test]
    fn calibrator_probe_measures_flat_block_before_pool_block() {
        // Warmup runs on the pool (node-local placement), then the flat block is
        // measured first (pool quiesced), then the pool block -- never interleaved,
        // so a hot pool step cannot pollute a flat measurement.
        let mut calib = Calibrator::new();
        for _ in 0..CALIB_WARMUP_STEPS {
            assert_eq!(calib.choose(), AutoPath::Pool);
            assert_eq!(calib.phase, CalibPhase::Warmup);
            calib.record(AutoPath::Pool, 1_000);
        }
        assert_eq!(calib.phase, CalibPhase::ProbeFlat);
        // The whole flat block chooses Flat.
        while calib.phase == CalibPhase::ProbeFlat {
            assert_eq!(calib.choose(), AutoPath::Flat);
            calib.record(AutoPath::Flat, 100);
        }
        assert_eq!(calib.phase, CalibPhase::ProbePool);
        // The whole pool block chooses Pool.
        while calib.phase == CalibPhase::ProbePool {
            assert_eq!(calib.choose(), AutoPath::Pool);
            calib.record(AutoPath::Pool, 100);
        }
        assert_eq!(calib.phase, CalibPhase::Committed);
    }

    #[test]
    fn calibrator_probe_discards_the_transition_sample() {
        // The first sample of each block is discarded so the other path's
        // winding-down threads do not pollute it: only CALIB_PROBE_SAMPLES land.
        let mut calib = Calibrator::new();
        for _ in 0..CALIB_WARMUP_STEPS {
            calib.record(AutoPath::Pool, 1);
        }
        assert_eq!(calib.phase, CalibPhase::ProbeFlat);
        assert_eq!(calib.discard_left, CALIB_PROBE_DISCARD);
        while calib.phase == CalibPhase::ProbeFlat {
            calib.record(AutoPath::Flat, 100);
        }
        assert_eq!(calib.flat_ns.len(), CALIB_PROBE_SAMPLES);
    }

    #[test]
    fn calibrator_commits_pool_only_when_clearly_faster() {
        // Pool 20% faster than flat clears the 8% hysteresis margin -> adopt pool.
        let calib = run_one_probe(80, 100);
        assert_eq!(calib.phase, CalibPhase::Committed);
        assert_eq!(calib.committed, AutoPath::Pool);
        assert_eq!(calib.choose(), AutoPath::Pool);
    }

    #[test]
    fn calibrator_stays_flat_when_pool_slower_simulating_contention() {
        // Regression guard: under (simulated) co-tenant load the spinning pool is
        // slower, so its median probe time loses and the flat path stays
        // committed -- decode behaves exactly like today's flat path. This is the
        // property that makes "never regress under load" a measured guarantee.
        let calib = run_one_probe(200, 100);
        assert_eq!(calib.committed, AutoPath::Flat);
        assert_eq!(calib.choose(), AutoPath::Flat);
    }

    #[test]
    fn calibrator_stays_flat_within_the_hysteresis_margin() {
        // Pool only ~5% faster (< 8% margin): not worth switching, avoids flapping
        // and keeps the safe flat default.
        let calib = run_one_probe(95, 100);
        assert_eq!(calib.committed, AutoPath::Flat);
    }

    #[test]
    fn calibrator_stays_committed_permanently_after_initial_probe() {
        // Adopt the pool on a quiet probe, then verify the path stays frozen
        // indefinitely — no re-probe, no mid-generation switching.
        let mut calib = run_one_probe(80, 100);
        assert_eq!(calib.committed, AutoPath::Pool);
        // Feed many more steps than the old CALIB_RECAL_PERIOD (600):
        // the path must never leave Committed.
        for _ in 0..2000 {
            assert_eq!(calib.choose(), AutoPath::Pool);
            calib.record(AutoPath::Pool, 80);
            assert_eq!(calib.phase, CalibPhase::Committed);
        }
    }

    #[test]
    fn calibrator_probe_median_rejects_a_single_load_spike() {
        // The pool is genuinely faster (median 80) but one probe sample spikes
        // under a transient stall; the median ignores the outlier and still adopts
        // the pool, so a single blip does not cost the win.
        let mut calib = Calibrator::new();
        for _ in 0..CALIB_WARMUP_STEPS {
            calib.record(AutoPath::Pool, 80);
        }
        // Flat block: all 100.
        while calib.phase == CalibPhase::ProbeFlat {
            calib.record(AutoPath::Flat, 100);
        }
        // Pool block: mostly fast with one spike; the (discarded) transition
        // sample plus four 80s and one huge spike keep the median at 80.
        let mut pool_samples = [80u64, 80, 80, 80, 80, 100_000].into_iter();
        while calib.phase == CalibPhase::ProbePool {
            calib.record(AutoPath::Pool, pool_samples.next().unwrap_or(80));
        }
        assert_eq!(calib.committed, AutoPath::Pool);
    }

    /// Drive a fresh calibrator through warmup + a flat block (all `flat`) + a pool
    /// block whose *stored* samples are `pool_block` (a leading discard sample is
    /// fed automatically). Returns the calibrator after the single probe resolves
    /// (either committed or bounced back to `ProbeFlat` by a re-probe).
    fn probe_once(flat: u64, pool_block: &[u64]) -> Calibrator {
        let mut calib = Calibrator::new();
        for _ in 0..CALIB_WARMUP_STEPS {
            calib.record(AutoPath::Pool, 1_000);
        }
        while calib.phase == CalibPhase::ProbeFlat {
            calib.record(AutoPath::Flat, flat);
        }
        // The first pool record is the discarded transition sample; the rest are
        // the block's stored samples.
        let mut feed = std::iter::once(9_999u64).chain(pool_block.iter().copied());
        while calib.phase == CalibPhase::ProbePool {
            calib.record(
                AutoPath::Pool,
                feed.next().unwrap_or(pool_block[pool_block.len() - 1]),
            );
        }
        calib
    }

    #[test]
    fn block_contended_flags_a_nonuniform_block_only() {
        // Uniformly fast (idle) or uniformly slow (sustained load) blocks have a
        // tiny fast-half spread and are NOT flagged; a burst that inflates some
        // samples above the min pushes (median - min)/median over the threshold.
        assert!(!block_contended(100, &[100, 100, 100, 100, 100]));
        assert!(!block_contended(300, &[300, 300, 300, 300, 300])); // sustained load
        assert!(!block_contended(100, &[100, 100, 100, 100, 100_000])); // lone high outlier
        assert!(block_contended(420, &[300, 300, 420, 420, 420])); // 28% spread
        // Empty / single-sample blocks have no spread to measure.
        assert!(!block_contended(u64::MAX, &[]));
        assert!(!block_contended(50, &[50]));
    }

    #[test]
    fn calibrator_reprobes_a_burst_contaminated_pool_block_instead_of_committing() {
        // The observed failure: a transient co-tenant burst lands on part of the
        // pool block, inflating its median above the pool's true (fast) cost. The
        // calibrator must NOT lock in a decision from that block -- it re-probes.
        let mut calib = probe_once(100, &[300, 300, 420, 420, 420]);
        assert_eq!(
            calib.phase,
            CalibPhase::ProbeFlat,
            "must re-probe, not commit"
        );
        assert_eq!(calib.reprobes_left, CALIB_MAX_REPROBES - 1);
        // A clean re-probe now measures the pool as genuinely faster and adopts it,
        // recovering the throughput the poisoned probe would have thrown away.
        drive_to_commit(&mut calib, 80, 100);
        assert_eq!(calib.committed, AutoPath::Pool);
    }

    #[test]
    fn calibrator_commits_flat_under_uniform_sustained_load_without_reprobing() {
        // No-regression guarantee: genuine sustained load makes every pool step
        // uniformly slow (low spread), so the block is NOT treated as a burst -- the
        // calibrator commits flat immediately, exactly as before this robustness
        // change. Sustained load can never be misread as a transient burst.
        let calib = probe_once(100, &[300, 300, 300, 300, 300]);
        assert_eq!(calib.phase, CalibPhase::Committed);
        assert_eq!(calib.committed, AutoPath::Flat);
        assert_eq!(
            calib.reprobes_left, CALIB_MAX_REPROBES,
            "no re-probe was spent"
        );
    }

    #[test]
    fn calibrator_reprobe_budget_is_bounded_and_then_commits() {
        // A pathologically noisy host that never yields a clean probe must still
        // commit within CALIB_MAX_REPROBES re-probes (here on the median, which is
        // pool-slow -> flat), so re-probing can never loop or stall decode.
        let mut calib = Calibrator::new();
        for _ in 0..CALIB_WARMUP_STEPS {
            calib.record(AutoPath::Pool, 1_000);
        }
        let mut reprobes = 0u32;
        // Keep feeding contended blocks; count the re-probes until it commits.
        for _ in 0..(CALIB_MAX_REPROBES + 5) {
            while calib.phase == CalibPhase::ProbeFlat {
                calib.record(AutoPath::Flat, 100);
            }
            let mut feed = [9_999u64, 300, 300, 420, 420, 420].into_iter();
            while calib.phase == CalibPhase::ProbePool {
                calib.record(AutoPath::Pool, feed.next().unwrap_or(420));
            }
            if calib.phase == CalibPhase::Committed {
                break;
            }
            reprobes += 1;
        }
        assert_eq!(calib.phase, CalibPhase::Committed);
        assert_eq!(reprobes, CALIB_MAX_REPROBES);
        assert_eq!(calib.committed, AutoPath::Flat);
    }

    #[test]
    fn median_ns_picks_the_middle_and_guards_empty() {
        assert_eq!(median_ns(&mut []), u64::MAX);
        assert_eq!(median_ns(&mut [5]), 5);
        assert_eq!(median_ns(&mut [30, 10, 20]), 20);
        assert_eq!(median_ns(&mut [10, 40, 20, 30]), 30);
    }
}

/// The pool exposes one job slot and one barrier. Before the claim gate, two
/// threads dispatching at once overwrote each other's job pointer and each
/// other's pending counts. Both failure modes were observed by deleting the
/// gate and re-running these tests: shards running the *other* thread's
/// closure (silently wrong tensors, no crash), and -- more often -- the
/// pending counters never draining, hanging `wait` forever. The pool spins in
/// `wait`, so the hang burns every core until the process is killed.
///
/// Only the last two tests here are falsifiers for the gate itself: the first
/// three hold no contention, so they pass with or without it. They cover the
/// RAII guard's own contract (exclusivity, release-on-unwind) and the ordinary
/// single-dispatcher partition, which is worth pinning separately.
#[cfg(test)]
mod dispatch_claim_tests {
    use super::*;
    use std::sync::Barrier;
    use std::sync::atomic::AtomicU32;

    /// A dispatch whose shards all run, exactly once each, is the property the
    /// single job slot silently violated under a concurrent second dispatcher.
    fn assert_each_shard_ran_once(counts: &[AtomicU32]) {
        for (index, count) in counts.iter().enumerate() {
            assert_eq!(
                count.load(Ordering::Relaxed),
                1,
                "shard {index} ran {} times, expected exactly 1",
                count.load(Ordering::Relaxed)
            );
        }
    }

    /// A `SharedState` with only the barrier fields populated: one node per entry
    /// of `pending`, each with that many workers outstanding, no spawned threads
    /// and no job. Enough to drive the dispatcher-side publish/wait protocol
    /// directly. An empty slice builds a pool with no nodes at all, which is the
    /// degenerate shape the `DispatchClaim` tests want.
    fn barrier_only_shared_state(pending: &[usize]) -> SharedState {
        SharedState {
            node_sense: pending
                .iter()
                .map(|_| Padded(NodeSense::default()))
                .collect(),
            job: UnsafeCell::new(None),
            node_pending: pending
                .iter()
                .map(|count| Padded(AtomicUsize::new(*count)))
                .collect(),
            worker_node: Vec::new(),
            ready: AtomicUsize::new(0),
            poisoned_worker: AtomicUsize::new(0),
            dispatching: Padded(AtomicBool::new(false)),
            shutdown: AtomicBool::new(false),
            workers_exited: AtomicUsize::new(0),
            profile_workers: false,
            worker_counters: Vec::new(),
            dispatch_counters: Padded(DispatchCounters::default()),
            completion_sense: Padded(AtomicU32::new(0)),
            dispatcher_parked: Padded(AtomicBool::new(false)),
        }
    }

    /// A one-worker, one-node pool running the real `worker_loop`.
    fn one_worker_shared_state() -> Arc<SharedState> {
        Arc::new(SharedState {
            node_sense: vec![Padded(NodeSense::default())],
            job: UnsafeCell::new(None),
            node_pending: vec![Padded(AtomicUsize::new(0))],
            worker_node: vec![0],
            ready: AtomicUsize::new(0),
            poisoned_worker: AtomicUsize::new(0),
            dispatching: Padded(AtomicBool::new(false)),
            shutdown: AtomicBool::new(false),
            workers_exited: AtomicUsize::new(0),
            profile_workers: false,
            worker_counters: vec![Padded(WorkerCounters::default())],
            dispatch_counters: Padded(DispatchCounters::default()),
            completion_sense: Padded(AtomicU32::new(0)),
            dispatcher_parked: Padded(AtomicBool::new(false)),
        })
    }

    /// Counts calls so a re-run of a retired op is visible.
    ///
    /// The count is carried through the `Job`'s own `data` pointer rather than a
    /// static, so the two tests below cannot see each other's calls. They run
    /// concurrently in one process, and a shared counter made each one fail
    /// depending on the other's timing while both passed in isolation.
    ///
    /// SAFETY: `data` is the `&AtomicUsize` published alongside this fn pointer,
    /// which outlives the worker that reads it.
    unsafe fn counting_call(data: *const (), _global_index: usize) {
        unsafe { &*data.cast::<AtomicUsize>() }.fetch_add(1, Ordering::Release);
    }

    fn wait_for_ready(shared: &SharedState) {
        while shared.ready.load(Ordering::Acquire) < 1 {
            thread::sleep(Duration::from_millis(1));
        }
        // Past the spin window, so the worker is genuinely parked on the futex
        // rather than still spinning -- the park is the case that matters.
        thread::sleep(Duration::from_millis(50));
    }

    /// Publish one op through the *real* `SharedState::publish`.
    ///
    /// Deliberately not a local copy of it. A mirrored version would keep
    /// passing if `publish` itself regressed -- swapping its `ops` and `wake`
    /// bumps, for instance, lets a worker observe the wake with `ops` still
    /// stale, skip the op, and strand the dispatcher again -- which is exactly
    /// the class of bug these tests exist to catch.
    fn publish_one(shared: &SharedState, calls: &AtomicUsize) {
        shared.publish(
            Job {
                data: std::ptr::from_ref(calls).cast(),
                call: counting_call,
            },
            &[1],
        );
    }

    /// Shut down the way `SpmdDecodePools::shutdown` does once the pool is
    /// quiet: flag, then bump and wake every node's sense. Notably it does *not*
    /// touch `ops`.
    ///
    /// Mirrored rather than called because the real one needs a whole
    /// `SpmdDecodePools`. Its quiescence wait is covered separately by
    /// `shutdown_waits_for_an_in_flight_dispatch_before_stopping_workers`.
    fn shutdown_like_the_pool(shared: &SharedState) {
        shared.shutdown.store(true, Ordering::SeqCst);
        shared.node_sense[0].0.wake.fetch_add(1, Ordering::Release);
        atomic_wait::wake_all(&shared.node_sense[0].0.wake);
    }

    /// A shard published just before a concurrent shutdown must still be
    /// retired, because a dispatcher is already blocked waiting for it.
    ///
    /// The dispatcher holds the job closure on its own stack frame and does not
    /// return until every node's pending count reaches zero, so a worker that
    /// leaves without acknowledging does not merely lose work -- it strands that
    /// dispatcher permanently. Since #1801 the dispatcher *parks* rather than
    /// spins, so the symptom is a hang at zero CPU with no thread to attribute
    /// it to, which is materially harder to diagnose than a spin.
    ///
    /// Asserted on the pending count with a timeout rather than by joining the
    /// dispatcher, because the failure is a hang: joining would hang the suite
    /// instead of failing this test.
    #[test]
    fn a_dispatch_racing_shutdown_is_retired_not_abandoned() {
        let calls = AtomicUsize::new(0);
        let shared = one_worker_shared_state();
        let worker = {
            let shared = Arc::clone(&shared);
            thread::spawn(move || worker_loop(shared, 0))
        };
        wait_for_ready(&shared);

        // The interleaving: a dispatcher publishes, then a shutdown on another
        // thread lands before this worker has observed the op.
        publish_one(&shared, &calls);
        shutdown_like_the_pool(&shared);

        let deadline = Instant::now() + Duration::from_secs(5);
        while shared.node_pending[0].0.load(Ordering::Acquire) != 0 && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        let outstanding = shared.node_pending[0].0.load(Ordering::Acquire);
        let _ = worker.join();

        assert_eq!(
            outstanding, 0,
            "worker left without retiring a published shard, so a dispatcher \
             waiting on it can never make progress"
        );
        assert_eq!(
            calls.load(Ordering::Acquire),
            1,
            "the published op must actually run, not just be acknowledged"
        );
    }

    /// The shutdown bump on its own must never be mistaken for a new op.
    ///
    /// This is the hazard the old shutdown-first check existed to prevent, and
    /// it is the more dangerous of the two directions: re-running a retired
    /// `Job` dereferences a closure whose stack frame has already returned, so
    /// it is a use-after-free rather than a hang. Keeping both properties is the
    /// whole reason waking and having-work are two separate words.
    #[test]
    fn a_shutdown_bump_alone_never_re_runs_the_previous_op() {
        let calls = AtomicUsize::new(0);
        let shared = one_worker_shared_state();
        let worker = {
            let shared = Arc::clone(&shared);
            thread::spawn(move || worker_loop(shared, 0))
        };
        wait_for_ready(&shared);

        publish_one(&shared, &calls);
        let deadline = Instant::now() + Duration::from_secs(5);
        while shared.node_pending[0].0.load(Ordering::Acquire) != 0 && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(calls.load(Ordering::Acquire), 1, "first op must run");

        // Now shut down. The wake word advances, `ops` does not.
        shutdown_like_the_pool(&shared);
        let _ = worker.join();

        assert_eq!(
            calls.load(Ordering::Acquire),
            1,
            "the shutdown bump re-ran a retired op, whose closure no longer exists"
        );
    }

    /// Teardown must not begin while a dispatch holds the publish slot.
    ///
    /// `publish` commits each node's pending count before it bumps that node's
    /// `ops`, so inside that window a shard is outstanding but no worker can yet
    /// see it exists. A shutdown landing there is invisible to the `ops` gate --
    /// the worker correctly concludes it has nothing to run, sees the flag, and
    /// leaves without retiring a count the dispatcher is still waiting on.
    /// Splitting the sense words cannot close that, because the op has not been
    /// announced yet; waiting for the pool to go quiet is what closes it.
    #[test]
    fn shutdown_waits_for_an_in_flight_dispatch_before_stopping_workers() {
        let shared = one_worker_shared_state();

        // No dispatch in flight: quiescence is immediate.
        assert!(shared.await_quiescent_dispatch(Duration::from_millis(50)));

        let claim = DispatchClaim::try_claim(&shared).expect("uncontended claim");
        // With the slot held, it must refuse to report quiet rather than let a
        // teardown proceed into the middle of a publish.
        assert!(
            !shared.await_quiescent_dispatch(Duration::from_millis(50)),
            "reported quiet while a dispatch still held the publish slot"
        );

        // The property that matters is not that the helper exists but that the
        // stop flag stays down until the dispatch is done, so assert on the flag
        // through `begin_shutdown` -- the function the pool actually calls.
        let stopper = {
            let shared = Arc::clone(&shared);
            thread::spawn(move || shared.begin_shutdown(Duration::from_secs(5)))
        };
        thread::sleep(Duration::from_millis(50));
        assert!(
            !shared.shutdown.load(Ordering::SeqCst),
            "raised the stop flag while a dispatch was still in flight"
        );

        // And it must proceed once the dispatch releases, rather than block
        // teardown for the whole bound.
        drop(claim);
        assert!(
            stopper.join().expect("shutdown thread"),
            "did not observe the dispatch releasing the slot"
        );
        assert!(shared.shutdown.load(Ordering::SeqCst), "never stopped");
    }

    /// The dispatcher must be woken by the worker that retires the last shard.
    ///
    /// Asserted through a channel with a timeout rather than by simply calling
    /// `wait_with_yield_budget` on this thread: a lost wakeup is a *hang*, and a
    /// hang in CI reads as an infrastructure timeout rather than as this test
    /// failing. The budget is `0` so the dispatcher parks the moment its spin
    /// budget is exhausted, which is the path under test; the worker sleeps long
    /// enough that the dispatcher is reliably asleep before it signals, so the
    /// wake is genuinely required and not satisfied by the pre-park re-check.
    ///
    /// Verified by mutation: deleting the `wake_all` in `signal_completion` makes
    /// this time out. The `SeqCst` pair is *not* covered -- x86 is TSO, so a
    /// weakened ordering still passes here and would only fail on a weakly
    /// ordered target. It is argued for in `park_until_complete`, not asserted.
    #[test]
    fn a_parked_dispatcher_is_woken_by_the_last_worker_to_finish() {
        let shared = std::sync::Arc::new(barrier_only_shared_state(&[1]));
        let (tx, rx) = std::sync::mpsc::channel();
        let dispatcher = {
            let shared = std::sync::Arc::clone(&shared);
            thread::spawn(move || {
                shared.wait_with_yield_budget(0);
                let _ = tx.send(());
            })
        };
        thread::sleep(Duration::from_millis(50));
        assert!(
            !shared.all_workers_done(),
            "the fixture must still have the op outstanding, or the dispatcher \
             would return without ever parking and the test would pass vacuously"
        );
        if shared.node_pending[0].0.fetch_sub(1, Ordering::AcqRel) == 1 {
            shared.signal_completion();
        }
        rx.recv_timeout(Duration::from_secs(10))
            .expect("a parked dispatcher must be woken when the op completes");
        dispatcher.join().expect("dispatcher thread");
    }

    /// The park must survive many consecutive ops, with `completion_sense`
    /// advancing under it and `dispatcher_parked` correctly cleared each time.
    ///
    /// Its unique coverage is the two post-conditions, not the wake itself --
    /// that is already covered above. A dispatcher that failed to clear
    /// `dispatcher_parked` on the way out would leave every later op paying a
    /// `wake` syscall it does not need, which no pass/fail on timing would see.
    ///
    /// One rationale deliberately *not* claimed here, because I wrote it down and
    /// then falsified it: that a stale `observed` would make an iteration sleep
    /// forever. It would not. A stale `observed` fails the
    /// `completion_sense == observed` guard, so `park_until_complete` skips the
    /// `wait` entirely and the caller's loop still exits on `all_workers_done`.
    ///
    /// Every iteration sleeps before signalling for the same reason the single-op
    /// test does, and for the same duration -- a shorter one lets a loaded runner
    /// land the signal before the dispatcher has parked, which quietly drops that
    /// iteration onto the pre-park re-check. Verified: this test passed unchanged
    /// with `wake_all` deleted until the sleep was added.
    #[test]
    fn repeated_parks_never_lose_a_wakeup() {
        let shared = std::sync::Arc::new(barrier_only_shared_state(&[1]));
        for iteration in 0..16 {
            shared.node_pending[0].0.store(1, Ordering::Release);
            let (tx, rx) = std::sync::mpsc::channel();
            let dispatcher = {
                let shared = std::sync::Arc::clone(&shared);
                thread::spawn(move || {
                    shared.wait_with_yield_budget(0);
                    let _ = tx.send(());
                })
            };
            thread::sleep(Duration::from_millis(50));
            if shared.node_pending[0].0.fetch_sub(1, Ordering::AcqRel) == 1 {
                shared.signal_completion();
            }
            rx.recv_timeout(Duration::from_secs(10))
                .expect("every op must release the dispatcher");
            dispatcher.join().expect("dispatcher thread");
            assert_eq!(
                shared.completion_sense.0.load(Ordering::SeqCst),
                iteration + 1,
                "each op must advance the completion sense exactly once"
            );
            assert!(
                !shared.dispatcher_parked.0.load(Ordering::SeqCst),
                "a dispatcher that has returned must no longer advertise itself \
                 as parked, or every later op pays a wake syscall for nothing"
            );
        }
    }

    /// A dispatcher parked across a multi-node barrier must be released by
    /// whichever node's worker retires last, and must not be released by the
    /// first node to drain.
    ///
    /// This covers the control flow of the cross-node scan in
    /// `signal_completion` -- the path that exists only because more than one
    /// worker can reach it per op. It does *not* cover the ordering hazard that
    /// path's fence is there for: on x86 the `lock`-prefixed decrement is already
    /// a full barrier, so the fence can be deleted and this still passes. That
    /// argument is written out on `signal_completion` and is not asserted here.
    ///
    /// Verified by mutation: deleting the `wake_all` makes this time out.
    #[test]
    fn a_parked_dispatcher_waits_for_every_node_not_just_the_first() {
        let shared = std::sync::Arc::new(barrier_only_shared_state(&[1, 1]));
        let (tx, rx) = std::sync::mpsc::channel();
        let dispatcher = {
            let shared = std::sync::Arc::clone(&shared);
            thread::spawn(move || {
                shared.wait_with_yield_budget(0);
                let _ = tx.send(());
            })
        };
        thread::sleep(Duration::from_millis(50));

        if shared.node_pending[0].0.fetch_sub(1, Ordering::AcqRel) == 1 {
            shared.signal_completion();
        }
        assert!(
            matches!(
                rx.recv_timeout(Duration::from_millis(250)),
                Err(std::sync::mpsc::RecvTimeoutError::Timeout)
            ),
            "draining one node of two must not release the dispatcher"
        );

        if shared.node_pending[1].0.fetch_sub(1, Ordering::AcqRel) == 1 {
            shared.signal_completion();
        }
        rx.recv_timeout(Duration::from_secs(10))
            .expect("the last node to drain must release the dispatcher");
        dispatcher.join().expect("dispatcher thread");
    }

    #[test]
    fn an_uncontended_dispatch_releases_the_claim_for_the_next_one() {
        let shared = barrier_only_shared_state(&[]);
        {
            let claim = DispatchClaim::try_claim(&shared)
                .expect("an unclaimed pool must hand out the claim");
            assert!(
                DispatchClaim::try_claim(&shared).is_none(),
                "a held claim must be exclusive; a second claimant would \
                 publish into the same job slot"
            );
            drop(claim);
        }
        assert!(
            DispatchClaim::try_claim(&shared).is_some(),
            "a released claim must let the next dispatch use the pool; leaking \
             it would send every later dispatch inline forever"
        );
    }

    #[test]
    fn a_panic_inside_the_critical_section_still_releases_the_claim() {
        let shared = barrier_only_shared_state(&[]);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _claim = DispatchClaim::try_claim(&shared).expect("claim available");
            panic!("a worker poisoned the pool");
        }));
        assert!(result.is_err());
        assert!(
            DispatchClaim::try_claim(&shared).is_some(),
            "unwinding out of the critical section must release the claim"
        );
    }

    #[test]
    fn every_shard_of_a_dispatch_runs_exactly_once() {
        let Some(pools) = pools() else {
            return;
        };
        let counts: Vec<AtomicU32> = (0..pools.total_workers)
            .map(|_| AtomicU32::new(0))
            .collect();
        pools.dispatch(&|index: usize| {
            counts[index].fetch_add(1, Ordering::Relaxed);
        });
        assert_each_shard_ran_once(&counts);
    }

    /// The regression itself: two threads dispatching at the same instant. On
    /// the unguarded pool the loser's shards ran the winner's closure, so its
    /// own counters came back short (and the winner's over-counted). With the
    /// gate the loser runs inline and both see a complete, exact partition.
    #[test]
    fn two_threads_dispatching_at_once_each_get_their_own_complete_partition() {
        let Some(pools) = pools() else {
            return;
        };
        let workers = pools.total_workers;
        let start = Barrier::new(2);
        let left: Vec<AtomicU32> = (0..workers).map(|_| AtomicU32::new(0)).collect();
        let right: Vec<AtomicU32> = (0..workers).map(|_| AtomicU32::new(0)).collect();
        thread::scope(|scope| {
            for counts in [&left, &right] {
                scope.spawn(|| {
                    start.wait();
                    for _ in 0..64 {
                        pools.dispatch(&|index: usize| {
                            counts[index].fetch_add(1, Ordering::Relaxed);
                        });
                    }
                });
            }
        });
        for (side, counts) in [("left", &left), ("right", &right)] {
            for (index, count) in counts.iter().enumerate() {
                assert_eq!(
                    count.load(Ordering::Relaxed),
                    64,
                    "{side} shard {index} ran {} times across 64 dispatches, \
                     expected 64 -- a shard that ran the other thread's closure \
                     is the silent-corruption failure this gate prevents",
                    count.load(Ordering::Relaxed)
                );
            }
        }
    }

    /// A shard closure that dispatches again used to deadlock: the inner
    /// `publish` overwrote the job the outer barrier was still waiting on. The
    /// gate turns re-entrancy into inline execution.
    #[test]
    fn a_reentrant_dispatch_runs_inline_instead_of_deadlocking() {
        let Some(pools) = pools() else {
            return;
        };
        let workers = pools.total_workers;
        let inner: Vec<AtomicU32> = (0..workers).map(|_| AtomicU32::new(0)).collect();
        let outer = AtomicU32::new(0);
        pools.dispatch(&|index: usize| {
            if index != 0 {
                return;
            }
            outer.fetch_add(1, Ordering::Relaxed);
            pools.dispatch(&|nested: usize| {
                inner[nested].fetch_add(1, Ordering::Relaxed);
            });
        });
        assert_eq!(outer.load(Ordering::Relaxed), 1);
        assert_each_shard_ran_once(&inner);
    }
}

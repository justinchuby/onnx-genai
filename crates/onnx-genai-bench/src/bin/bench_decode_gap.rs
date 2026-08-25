//! Decode-shaped, gap-aware model-level benchmark for the CPU task runtime.
//!
//! `bench_generic` measures a model in a tight loop. That answers "how fast is
//! this graph when it is run back-to-back forever", which is the right question
//! for prefill and the wrong one for decode. Decode issues a parallel region,
//! does a stretch of serial host work, and issues the next one -- and the cost
//! of that pattern is dominated by whether the pool's workers were still
//! spinning when the next fan-out arrived. A tight loop never lets them park,
//! so it reports a best case that production never sees; a short loop never
//! leaves pool construction, so it reports a worst case that production also
//! never sees.
//!
//! This harness runs the model with a configurable gap between iterations, and
//! reports the steady-state distribution *plus* the counters that say which
//! regime it was in:
//!
//! * `vol-ctxt/iter` -- voluntary context switches per iteration. This is the
//!   park/wake count. Near zero means the pool spun through the gap; near the
//!   worker count means it parked and paid a futex wake on every dispatch.
//! * `cpu/wall` -- CPU-seconds burned per wall-second. A pool that spins
//!   through gaps converts idle time into CPU here.
//! * `rss` -- so a scheduler change that trades memory for latency is visible.
//!
//! Steady state is *detected*, not assumed: iterations are recorded from the
//! first one, and the reported window starts where the series stops trending
//! (see [`steady_state_start`]). A run whose series never settles says so
//! rather than quietly reporting its own warm-up.
//!
//! # Examples
//!
//! ```text
//! # A decode-shaped gap, native alone, steady state detected automatically.
//! bench_decode_gap --model m.onnx --gap-us 20 --iters 600 --native-threads 16
//!
//! # The A/A null control: the same binary twice, to size the noise band
//! # before believing any A/B ratio.
//! bench_decode_gap --model m.onnx --arm null --gap-us 20 --iters 600
//!
//! # Four concurrent sessions, the oversubscription shape.
//! bench_decode_gap --model m.onnx --sessions 4 --gap-us 50
//!
//! # Thread ownership census.
//! bench_decode_gap --model m.onnx --census --iters 64
//! ```
//!
//! Syscall-level attribution (futex and `sched_yield` counts) is not
//! self-reported, because Linux exposes no per-syscall counter to a process
//! about itself. The harness prints a ready-to-run `strace -c -f` line for the
//! exact invocation instead; voluntary context switches, which *are*
//! self-reported, already capture the park/wake behaviour those syscalls
//! implement.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Parser;
use onnx_genai_bench::decode_gap::{
    GapDistribution, GapKind, ProcessMetrics, Summary, ThreadInfo, census_by_name, census_delta,
    sample_process_metrics, spend_gap, steady_state_start, thread_census,
};
use onnx_genai_bench::model_io::{
    F16_EPSILON, build_arm, build_inputs, build_ort_inputs, compare_outputs, parse_shape,
};
use onnx_genai_ort::{Environment, Session, SessionOptions, ep_selection};
use onnx_runtime_ep_cpu::{dispatch_ledger, task_runtime};
use onnx_runtime_session::InferenceSession;

/// The native CPU EP's decode-pool width knob, read once into a `OnceLock` by
/// the EP, so it must be set before the first session is built.
const NATIVE_DECODE_THREADS_ENV: &str = "ONNX_GENAI_CPU_DECODE_THREADS";

#[derive(Debug, Parser)]
#[command(about = "Decode-shaped, gap-aware CPU benchmark with park/wake accounting")]
struct Args {
    /// ONNX model file.
    #[arg(long)]
    model: PathBuf,
    /// Which runtime to time. `null` runs the native arm twice and reports the
    /// second over the first -- the A/A control that sizes the host's noise
    /// band. No A/B ratio from this harness means anything until the null
    /// control has been read.
    #[arg(long, default_value = "native")]
    arm: String,
    /// Mean gap between iterations, in microseconds. `0` reproduces the
    /// tight-loop shape, which is useful only as a reference point.
    #[arg(long, default_value_t = 20)]
    gap_us: u64,
    /// Fractional half-width of the uniform spread around `--gap-us`. A fixed
    /// gap can sit permanently just inside or just outside the pool's spin
    /// window and report a clean number for a bimodal reality.
    #[arg(long, default_value_t = 0.25)]
    gap_jitter: f64,
    /// How the gap is spent: `busy` (spin, holding the core, like host-side
    /// compute), `sleep` (release the core, like blocking on a tokenizer), or
    /// `mixed` (alternate).
    #[arg(long, default_value = "busy")]
    gap_kind: String,
    /// Seed for the gap sequence. Printed with the results so a run can be
    /// reproduced exactly.
    #[arg(long, default_value_t = 0x5EB_A571_A2)]
    gap_seed: u64,
    /// Total iterations recorded per session, including the warm-up transient.
    /// Must be large enough to contain a transient *and* two steady windows;
    /// a 32-wide pool needs several hundred.
    #[arg(long, default_value_t = 600)]
    iters: usize,
    /// Force the steady-state window to start here instead of detecting it.
    /// `usize::MAX` sentinel is avoided: `--warmup-iters 0` means "detect".
    #[arg(long, default_value_t = 0)]
    warmup_iters: usize,
    /// Samples per window used by steady-state detection.
    #[arg(long, default_value_t = 32)]
    steady_window: usize,
    /// Relative tolerance for steady-state detection.
    #[arg(long, default_value_t = 0.05)]
    steady_tolerance: f64,
    /// Concurrent sessions, each with its own model instance and gap sequence.
    /// This is the oversubscription shape: N independent decoders sharing one
    /// host, which is what a server does.
    #[arg(long, default_value_t = 1)]
    sessions: usize,
    /// Native CPU decode-pool width. `0` leaves `ONNX_GENAI_CPU_DECODE_THREADS`
    /// exactly as inherited, which is what a user gets out of the box.
    #[arg(long, default_value_t = 0)]
    native_threads: usize,
    /// ORT `intra_op_num_threads` for the ORT arm. `0` keeps ORT's default.
    #[arg(long, default_value_t = 0)]
    ort_intra_threads: i32,
    /// Override the first model input shape, for example 1,3,416,416.
    #[arg(long)]
    input_shape: Option<String>,
    /// Skip the ORT parity check. Parity is checked once, before timing, and
    /// the ORT session is then dropped so it cannot spin against the timed
    /// native arm -- so this flag only matters when ORT cannot load the model.
    #[arg(long)]
    no_parity: bool,
    /// Print a per-thread census (name, CPU, context switches) after timing.
    #[arg(long)]
    census: bool,
    /// Print which threads each lifecycle phase created: native session load,
    /// ORT parity session, first inference, and the timed loop. This is how an
    /// unnamed thread gets an owner -- a census alone cannot tell an
    /// unnamed Rayon worker of ours from one of ORT's, but a delta can.
    #[arg(long)]
    census_phases: bool,
    /// Write every per-iteration sample to this CSV, warm-up included, so the
    /// transient can be inspected rather than inferred. Columns:
    /// `arm,iteration,wall_ms,in_steady_window`.
    #[arg(long)]
    dump_csv: Option<PathBuf>,
    /// Relative tolerance used for Float32 output parity.
    #[arg(long, default_value_t = 1e-3)]
    rel_tolerance: f32,
    /// Absolute tolerance used for Float32 output parity.
    #[arg(long, default_value_t = 1e-4)]
    abs_tolerance: f32,
}

/// Prints which threads each lifecycle phase created.
///
/// Thread `comm` truncates to 15 bytes and defaults to the parent's, so a
/// thread nobody named shows up wearing the *process* name. That makes a flat
/// census actively misleading: the anonymous workers look like they belong to
/// the binary. Bracketing each construction step tells you who really made
/// them.
struct PhaseCensus {
    enabled: bool,
    previous: Vec<ThreadInfo>,
}

impl PhaseCensus {
    fn new(enabled: bool) -> Self {
        Self {
            enabled,
            previous: if enabled { thread_census() } else { Vec::new() },
        }
    }

    fn mark(&mut self, phase: &str) {
        if !self.enabled {
            return;
        }
        let now = thread_census();
        let created = census_delta(&self.previous, &now);
        let removed = census_delta(&now, &self.previous);
        if created.is_empty() && removed.is_empty() {
            println!("phase[{phase}]: no thread change ({} live)", now.len());
        } else {
            let names = census_by_name(&created)
                .into_iter()
                .map(|(name, count, _)| format!("{count}x {name}"))
                .collect::<Vec<_>>()
                .join(", ");
            println!(
                "phase[{phase}]: +{} -{} threads ({} live){}",
                created.len(),
                removed.len(),
                now.len(),
                if names.is_empty() {
                    String::new()
                } else {
                    format!(" -> {names}")
                }
            );
        }
        self.previous = now;
    }
}

/// One point in time: process counters, the native pool's counters, and the
/// instant they were taken at.
#[derive(Clone, Copy)]
struct Snapshot {
    iteration: usize,
    metrics: ProcessMetrics,
    pool: task_runtime::PoolCounters,
    at: Instant,
}

fn snapshot(iteration: usize) -> Snapshot {
    Snapshot {
        iteration,
        metrics: sample_process_metrics(),
        pool: task_runtime::testing::counters(),
        at: Instant::now(),
    }
}

/// Field-wise difference of two pool counter snapshots.
fn pool_delta(
    after: task_runtime::PoolCounters,
    before: task_runtime::PoolCounters,
) -> task_runtime::PoolCounters {
    task_runtime::PoolCounters {
        dispatches: after.dispatches.saturating_sub(before.dispatches),
        tasks: after.tasks.saturating_sub(before.tasks),
        tasks_by_dispatcher: after
            .tasks_by_dispatcher
            .saturating_sub(before.tasks_by_dispatcher),
        slot_exhausted: after.slot_exhausted.saturating_sub(before.slot_exhausted),
        parks: after.parks.saturating_sub(before.parks),
        spin_hits: after.spin_hits.saturating_sub(before.spin_hits),
        panics: after.panics.saturating_sub(before.panics),
        straggler_waits: after.straggler_waits.saturating_sub(before.straggler_waits),
        straggler_yields: after
            .straggler_yields
            .saturating_sub(before.straggler_yields),
        spin_yields: after.spin_yields.saturating_sub(before.spin_yields),
    }
}

/// Which kernel families ran on which backend, and with what parallel degree.
///
/// The pool counters answer "how hard did `task_runtime` work"; they cannot
/// answer "did this model use `task_runtime` at all". Those two produce the
/// same output -- a row of zeroes -- and only one of them means the harness is
/// broken. Attribution is what separates them, so it is collected rather than
/// assumed: an fp32 `MatMul` fans out on rayon, not on the native task pool, so
/// a model built from fp32 `MatMul`s reports zero dispatches while saturating
/// eight cores, and that reading is correct.
///
/// Recording is on only for the single attribution inference, never during the
/// timed window, because `record_with` builds an `Observation` per dispatch.
fn attribute_routes(run: impl FnOnce() -> Result<()>) -> Result<Vec<RouteRow>> {
    dispatch_ledger::reset();
    dispatch_ledger::enable();
    let outcome = run();
    dispatch_ledger::disable();
    outcome?;

    let mut rows: Vec<RouteRow> = Vec::new();
    for observation in dispatch_ledger::snapshot() {
        let key = (
            format!("{:?}", observation.family),
            format!("{:?}", observation.backend),
            observation.dtype,
        );
        match rows.iter_mut().find(|row| row.key() == key) {
            Some(row) => {
                row.calls += 1;
                row.max_threads = row.max_threads.max(observation.threads);
            }
            None => rows.push(RouteRow {
                family: key.0,
                backend: key.1,
                dtype: observation.dtype,
                calls: 1,
                max_threads: observation.threads,
            }),
        }
    }
    rows.sort_by(|a, b| b.calls.cmp(&a.calls).then_with(|| a.family.cmp(&b.family)));
    Ok(rows)
}

/// One `(family, backend, dtype)` route observed during attribution.
struct RouteRow {
    family: String,
    backend: String,
    dtype: &'static str,
    calls: usize,
    max_threads: usize,
}

impl RouteRow {
    fn key(&self) -> (String, String, &'static str) {
        (self.family.clone(), self.backend.clone(), self.dtype)
    }
}

/// Whether any observed route is one the native task pool would drive.
///
/// Used only to phrase the zero-dispatch case: "this model does not use the
/// pool" is a different statement from "the pool did nothing this run", and the
/// report should not silently pick one.
fn any_route_observed(rows: &[RouteRow]) -> bool {
    !rows.is_empty()
}

/// What one timed arm produced.
struct ArmResult {
    /// Every iteration, in order, including the warm-up transient.
    samples_ms: Vec<f64>,
    /// Where steady state began, and how it was decided.
    steady_start: Option<usize>,
    detected: bool,
    /// Counters accumulated across the steady window only.
    steady_metrics: ProcessMetrics,
    /// Wall-clock seconds spanned by the steady window.
    steady_wall_s: f64,
    /// Counters accumulated across the whole run, warm-up included.
    total_metrics: ProcessMetrics,
    total_wall_s: f64,
    /// Native pool counters accumulated across the steady window.
    steady_pool: task_runtime::PoolCounters,
    /// Iterations the counter deltas above actually span.
    ///
    /// Counters are only sampled every `--steady-window` iterations, so the
    /// nearest snapshot at or before the steady boundary is generally *earlier*
    /// than the boundary. Dividing a counter delta by the sample-window length
    /// would then overstate every per-iteration figure, so per-iteration
    /// counters divide by this instead. It is reported, because a counter
    /// window wider than the sample window is exactly the kind of thing that
    /// silently turns a per-iteration number into a slightly wrong one.
    counter_iters: usize,
    /// Whether the counter window had to fall back to iteration zero, i.e.
    /// whether these "steady" counters in fact include the warm-up transient.
    counters_include_warmup: bool,
    /// Routes observed during the attribution inference, if one was taken.
    routes: Vec<RouteRow>,
}

impl ArmResult {
    fn steady_samples(&self) -> &[f64] {
        match self.steady_start {
            Some(start) => &self.samples_ms[start..],
            None => &self.samples_ms,
        }
    }

    fn summary(&self) -> Summary {
        Summary::from(self.steady_samples())
    }

    /// Voluntary context switches per iteration over the counter window: the
    /// park/wake count per dispatch.
    fn parks_per_iter(&self) -> f64 {
        self.steady_metrics.voluntary_ctxt_switches as f64 / self.counter_iters.max(1) as f64
    }

    /// CPU-seconds burned per wall-second over the steady window. A spinning
    /// pool converts the gap into CPU and shows up here.
    fn cpu_per_wall(&self) -> f64 {
        if self.steady_wall_s <= 0.0 {
            return 0.0;
        }
        (self.steady_metrics.cpu_us() as f64 / 1e6) / self.steady_wall_s
    }
}

/// Runs `iterations` of `body`, spending a gap from `gaps` before each one.
///
/// Metrics are sampled twice: once at the start, and once at the steady-state
/// boundary. Because the boundary is not known until the series is complete,
/// the loop records a metrics snapshot every `snapshot_every` iterations and
/// the caller picks the one nearest the detected boundary. That keeps the
/// per-iteration cost of measurement off the timed path -- reading
/// `/proc/self/status` costs tens of microseconds, which is the same order as
/// the gaps being measured.
fn timed_loop<F>(
    iterations: usize,
    gaps: &mut GapDistribution,
    snapshot_every: usize,
    mut body: F,
) -> Result<(Vec<f64>, Vec<Snapshot>)>
where
    F: FnMut() -> Result<()>,
{
    let mut samples = Vec::with_capacity(iterations);
    let mut snapshots = vec![snapshot(0)];
    for index in 0..iterations {
        let kind = gaps.next_kind();
        spend_gap(gaps.next_gap(), kind);
        let started = Instant::now();
        body()?;
        samples.push(started.elapsed().as_secs_f64() * 1_000.0);
        if snapshot_every > 0 && (index + 1).is_multiple_of(snapshot_every) {
            snapshots.push(snapshot(index + 1));
        }
    }
    snapshots.push(snapshot(iterations));
    Ok((samples, snapshots))
}

/// Turns a sample series and its metric snapshots into an [`ArmResult`].
fn assemble(samples: Vec<f64>, snapshots: Vec<Snapshot>, args: &Args) -> ArmResult {
    let detected = args.warmup_iters == 0;
    let steady_start = if detected {
        steady_state_start(&samples, args.steady_window, args.steady_tolerance)
    } else {
        (args.warmup_iters < samples.len()).then_some(args.warmup_iters)
    };
    let first = snapshots.first().expect("a snapshot at iteration zero");
    let last = snapshots.last().expect("a snapshot at the final iteration");
    // The latest snapshot at or before the steady boundary. Counters are only
    // sampled every `--steady-window` iterations, so this is generally
    // *earlier* than the boundary and the counter window is correspondingly
    // wider than the sample window -- and when steady state is reached before
    // the first interior snapshot there is no qualifying snapshot at all, so
    // the window falls back to iteration zero and the counters do include the
    // warm-up transient. Neither case is avoidable without sampling counters
    // every iteration (which would perturb what is being measured), so both are
    // recorded and reported instead of being asserted away.
    let boundary = steady_start
        .and_then(|start| {
            snapshots
                .iter()
                .rev()
                .find(|point| point.iteration <= start && point.iteration > 0)
        })
        .unwrap_or(first);
    let counter_iters = last.iteration.saturating_sub(boundary.iteration);
    let counters_include_warmup = boundary.iteration == 0 && steady_start.unwrap_or(0) > 0;
    ArmResult {
        samples_ms: samples,
        steady_start,
        detected,
        steady_metrics: last.metrics.since(&boundary.metrics),
        steady_wall_s: last.at.duration_since(boundary.at).as_secs_f64(),
        total_metrics: last.metrics.since(&first.metrics),
        total_wall_s: last.at.duration_since(first.at).as_secs_f64(),
        steady_pool: pool_delta(last.pool, boundary.pool),
        counter_iters,
        counters_include_warmup,
        routes: Vec::new(),
    }
}

fn run_native_arm(args: &Args, label: &str) -> Result<ArmResult> {
    let mut phases = PhaseCensus::new(args.census_phases);
    phases.mark("process-start");
    let mut sessions = (0..args.sessions.max(1))
        .map(|_| {
            InferenceSession::load(&args.model)
                .with_context(|| format!("load native session from {}", args.model.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    phases.mark("native-session-load");

    let input_shape = args
        .input_shape
        .as_deref()
        .map(parse_shape)
        .transpose()
        .map_err(anyhow::Error::msg)?;

    // Parity is judged once, against ORT, *before* the timed loop -- and the
    // ORT session is dropped immediately afterwards. A co-resident ORT session
    // spin-waits long after its last op and depresses a native arm measured
    // beside it by several times; building it, using it and dropping it makes
    // the timed window solo by construction rather than by remembering a flag.
    if !args.no_parity {
        let environment = Environment::new("bench-decode-gap")?;
        let options =
            SessionOptions::with_execution_provider(ep_selection("cpu")).with_intra_op_threads(1);
        let ort_session = Session::new(&environment, &args.model, options)
            .with_context(|| format!("load ORT CPU session from {}", args.model.display()))?;
        let inputs = build_inputs(&sessions[0], &ort_session, input_shape.as_deref())?;
        let native_inputs = inputs
            .iter()
            .map(|input| (input.name.as_str(), &input.native))
            .collect::<Vec<_>>();
        let ort_inputs = inputs
            .iter()
            .map(|input| (input.name.as_str(), &input.ort))
            .collect::<Vec<_>>();
        phases.mark("ort-parity-session-built");
        let native_reference = sessions[0]
            .run(&native_inputs)
            .context("native parity run")?;
        phases.mark("first-native-inference");
        let ort_reference = ort_session.run(&ort_inputs).context("ORT parity run")?;
        let diffs = compare_outputs(
            &native_reference,
            &ort_reference,
            args.abs_tolerance,
            args.rel_tolerance,
            F16_EPSILON,
            4.0 * F16_EPSILON,
        )?;
        let failures = diffs.iter().filter(|diff| !diff.pass).count();
        println!(
            "parity[{label}]: {} outputs checked, {failures} failed, max_rel={:.3e}",
            diffs.len(),
            diffs.iter().map(|diff| diff.max_rel).fold(0.0f32, f32::max)
        );
        if failures > 0 {
            bail!(
                "native/ORT parity failed on {failures} output(s); timing a wrong kernel is not a measurement"
            );
        }
    }
    // The ORT session is out of scope here. Anything it created that is still
    // alive shows up as a *negative* delta failing to appear.
    phases.mark("ort-session-dropped");

    // Inputs for the timed loop, built without ORT in the picture.
    let owned_inputs = build_timed_inputs(&sessions[0], &args.model, input_shape.as_deref())?;
    phases.mark("inputs-built");

    if args.sessions <= 1 {
        let mut session = sessions.pop().expect("at least one session");
        let refs = owned_inputs
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>();
        let mut gaps = GapDistribution::new(
            args.gap_us,
            args.gap_jitter,
            GapKind::parse(&args.gap_kind).map_err(anyhow::Error::msg)?,
            args.gap_seed,
        );
        let routes = attribute_routes(|| {
            std::hint::black_box(session.run(&refs).context("native warm run")?);
            Ok(())
        })?;
        phases.mark("timed-session-warm");
        let (samples, snapshots) =
            timed_loop(args.iters, &mut gaps, args.steady_window.max(1), || {
                std::hint::black_box(session.run(&refs).context("native measured run")?);
                Ok(())
            })?;
        phases.mark("timed-loop");
        let mut result = assemble(samples, snapshots, args);
        result.routes = routes;
        return Ok(result);
    }

    // Attribution runs on one session before the concurrent region, not inside
    // it: the ledger takes a mutex per observation, so recording across N
    // barrier-synchronised threads would serialise the very contention this arm
    // exists to measure. Routing does not depend on how many sessions are live,
    // so one session's routes describe them all -- and without this the
    // concurrent arm reports zero routes, which the report cannot distinguish
    // from a ledger that saw nothing and would wrongly call a valid run a dead
    // instrument.
    let routes = {
        let refs = owned_inputs
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
            .collect::<Vec<_>>();
        let session = sessions.first_mut().expect("at least one session");
        attribute_routes(|| {
            std::hint::black_box(session.run(&refs).context("native attribution run")?);
            Ok(())
        })?
    };
    let mut result = run_concurrent_native(args, sessions, owned_inputs)?;
    result.routes = routes;
    phases.mark("timed-loop");
    Ok(result)
}

/// The concurrent-session arm.
///
/// Each session gets its own thread, its own model instance and its own gap
/// sequence (seeded per session, so the sessions do not march in lockstep and
/// manufacture an artificial thundering herd). Process counters are sampled on
/// the parent thread around the whole barrier-synchronised region, because the
/// interesting quantity is what the *process* costs when N decoders share it.
fn run_concurrent_native(
    args: &Args,
    sessions: Vec<InferenceSession>,
    owned_inputs: Vec<(String, onnx_runtime_session::Tensor)>,
) -> Result<ArmResult> {
    use std::sync::{Arc, Barrier};

    let gap_kind = GapKind::parse(&args.gap_kind).map_err(anyhow::Error::msg)?;
    let barrier = Arc::new(Barrier::new(sessions.len() + 1));
    let inputs = Arc::new(owned_inputs);
    let iters = args.iters;
    let (gap_us, jitter, seed) = (args.gap_us, args.gap_jitter, args.gap_seed);

    let started_all = Instant::now();
    let before = snapshot(0);
    let mut handles = Vec::new();
    for (index, mut session) in sessions.into_iter().enumerate() {
        let barrier = Arc::clone(&barrier);
        let inputs = Arc::clone(&inputs);
        handles.push(
            std::thread::Builder::new()
                .name(format!("gapbench-{index}"))
                .spawn(move || -> Result<Vec<f64>> {
                    let refs = inputs
                        .iter()
                        .map(|(name, tensor)| (name.as_str(), tensor))
                        .collect::<Vec<_>>();
                    let mut gaps = GapDistribution::new(
                        gap_us,
                        jitter,
                        gap_kind,
                        seed.wrapping_add(index as u64 * 0x9E37_79B9),
                    );
                    barrier.wait();
                    let mut samples = Vec::with_capacity(iters);
                    for _ in 0..iters {
                        let kind = gaps.next_kind();
                        spend_gap(gaps.next_gap(), kind);
                        let started = Instant::now();
                        std::hint::black_box(session.run(&refs).context("native measured run")?);
                        samples.push(started.elapsed().as_secs_f64() * 1_000.0);
                    }
                    Ok(samples)
                })
                .context("spawn concurrent session thread")?,
        );
    }
    barrier.wait();

    let mut per_session = Vec::new();
    for handle in handles {
        per_session.push(
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("concurrent session thread panicked"))??,
        );
    }
    let after = snapshot(0);
    let elapsed = started_all.elapsed().as_secs_f64();

    for (index, samples) in per_session.iter().enumerate() {
        let summary = Summary::from(samples);
        println!(
            "  session[{index}]: p50={:.3} ms p90={:.3} ms n={}",
            summary.p50, summary.p90, summary.count
        );
    }

    // Pool the sessions' samples: the question a server asks is what a request
    // costs, not what a particular worker's stream of requests costs.
    let samples = per_session.concat();
    let snapshots = vec![
        Snapshot {
            iteration: 0,
            metrics: before.metrics,
            pool: before.pool,
            at: started_all,
        },
        Snapshot {
            iteration: samples.len(),
            metrics: after.metrics,
            pool: after.pool,
            at: Instant::now(),
        },
    ];
    let mut result = assemble(samples, snapshots, args);
    // Steady-state detection is meaningless across a concatenation of
    // independent series, so the concurrent arm reports the whole window and
    // says so rather than pretending it detected a boundary.
    result.steady_start = None;
    result.detected = false;
    result.steady_metrics = after.metrics.since(&before.metrics);
    result.steady_pool = pool_delta(after.pool, before.pool);
    result.steady_wall_s = elapsed;
    Ok(result)
}

/// Synthetic inputs for the native session alone.
///
/// Reuses the shared generators so the tensors the timed loop feeds are
/// byte-identical to the ones the parity check judged. The throwaway ORT
/// session exists only to supply declared dtypes to the shared builder; it is
/// dropped before it returns, so nothing of ORT's is alive during timing.
fn build_timed_inputs(
    session: &InferenceSession,
    model: &std::path::Path,
    override_shape: Option<&[usize]>,
) -> Result<Vec<(String, onnx_runtime_session::Tensor)>> {
    let environment = Environment::new("bench-decode-gap-inputs")?;
    let options =
        SessionOptions::with_execution_provider(ep_selection("cpu")).with_intra_op_threads(1);
    let probe = Session::new(&environment, model, options)
        .with_context(|| format!("load ORT CPU session from {}", model.display()))?;
    let pairs = build_inputs(session, &probe, override_shape)?;
    Ok(pairs
        .into_iter()
        .map(|pair| (pair.name, pair.native))
        .collect())
}

fn run_ort_arm(args: &Args) -> Result<ArmResult> {
    let environment = Environment::new("bench-decode-gap")?;
    let intra = if args.ort_intra_threads > 0 {
        args.ort_intra_threads
    } else {
        0
    };
    let options =
        SessionOptions::with_execution_provider(ep_selection("cpu")).with_intra_op_threads(intra);
    let session = Session::new(&environment, &args.model, options)
        .with_context(|| format!("load ORT CPU session from {}", args.model.display()))?;
    let input_shape = args
        .input_shape
        .as_deref()
        .map(parse_shape)
        .transpose()
        .map_err(anyhow::Error::msg)?;
    let inputs = build_ort_inputs(&session, input_shape.as_deref())?;
    let refs = inputs
        .iter()
        .map(|(name, value)| (name.as_str(), value))
        .collect::<Vec<_>>();
    let mut gaps = GapDistribution::new(
        args.gap_us,
        args.gap_jitter,
        GapKind::parse(&args.gap_kind).map_err(anyhow::Error::msg)?,
        args.gap_seed,
    );
    let (samples, snapshots) =
        timed_loop(args.iters, &mut gaps, args.steady_window.max(1), || {
            std::hint::black_box(session.run(&refs).context("ORT measured run")?);
            Ok(())
        })?;
    Ok(assemble(samples, snapshots, args))
}

fn report(label: &str, result: &ArmResult, args: &Args) {
    let summary = result.summary();
    let window = match (result.steady_start, result.detected) {
        (Some(start), true) => format!("detected at iter {start}"),
        (Some(start), false) => format!("forced at iter {start}"),
        (None, true) => "NOT REACHED (series never settled)".to_string(),
        (None, false) => "whole run".to_string(),
    };
    println!("\n== {label} ==");
    println!(
        "  steady window: {window} ({} of {} iters)",
        summary.count,
        result.samples_ms.len()
    );
    println!(
        "  wall/iter:     p50={:.4} ms  p90={:.4} ms  p99={:.4} ms  min={:.4} ms  spread={:.2}x",
        summary.p50,
        summary.p90,
        summary.p99,
        summary.min,
        summary.spread()
    );
    println!(
        "  cpu:           {:.3} s over steady window  ({:.2} cpu-s per wall-s)",
        result.steady_metrics.cpu_us() as f64 / 1e6,
        result.cpu_per_wall()
    );
    // Kernel time is where a scheduler's waiting shows up -- `sched_yield` and
    // futex traffic land here and nowhere else -- so a pool that converts an
    // idle gap into syscalls is visible as a sys share even when total CPU
    // looks reasonable. Both halves were already being collected from
    // /proc/self/stat and then discarded at the report.
    let cpu_us = result.steady_metrics.cpu_us();
    if cpu_us > 0 {
        println!(
            "  cpu split:     {:.3} s user  {:.3} s sys  ({:.1}% sys)",
            result.steady_metrics.user_us as f64 / 1e6,
            result.steady_metrics.sys_us as f64 / 1e6,
            100.0 * result.steady_metrics.sys_us as f64 / cpu_us as f64
        );
    }
    println!(
        "  ctxt switches: {:.1} vol/iter  {:.1} invol/iter  ({} vol total)",
        result.parks_per_iter(),
        result.steady_metrics.involuntary_ctxt_switches as f64 / result.counter_iters.max(1) as f64,
        result.steady_metrics.voluntary_ctxt_switches
    );
    println!(
        "  rss:           {} kB now, {} kB peak, {} threads",
        result.steady_metrics.rss_kb,
        result.steady_metrics.peak_rss_kb,
        result.steady_metrics.threads
    );
    let iterations = result.counter_iters.max(1) as f64;
    println!(
        "  native pool:   {:.2} dispatches/iter  {:.2} parks/iter  {:.2} spin-hits/iter  \
         {} slot-exhausted",
        result.steady_pool.dispatches as f64 / iterations,
        result.steady_pool.parks as f64 / iterations,
        result.steady_pool.spin_hits as f64 / iterations,
        result.steady_pool.slot_exhausted
    );
    // The #2075 quantity. Per-iteration alone is not interpretable: yields only
    // happen in a spin window that outlives the pure-spin phase, so the rate
    // that matters is per *window*, and windows end at a park or a spin hit.
    // Both denominators can be zero for reasons that are not "no yields", so
    // each gets its own verdict rather than a rate the reader has to discount
    // against the attribution line printed further down.
    let windows = result.steady_pool.parks + result.steady_pool.spin_hits;
    if result.steady_pool.dispatches == 0 {
        println!(
            "  spin yields:   n/a -- this model never dispatched to the pool, so its \
             {} yields are another workload's, not a #2075 datum",
            result.steady_pool.spin_yields
        );
    } else if windows == 0 {
        println!(
            "  spin yields:   {} total, but no spin window ended in this arm by \
             expiring or catching a dispatch, so there is no denominator to read \
             them against and this is not a #2075 datum",
            result.steady_pool.spin_yields
        );
    } else {
        println!(
            "  spin yields:   {:.2}/iter  {:.1}/window over {windows} windows (#2075)",
            result.steady_pool.spin_yields as f64 / iterations,
            result.steady_pool.spin_yields as f64 / windows as f64,
        );
    }
    let sample_iters = result.steady_samples().len();
    if result.counter_iters == sample_iters {
        println!(
            "  counter window: {} iters, same span as the samples above",
            result.counter_iters
        );
    } else {
        println!(
            "  counter window: {} iters, WIDER than the {} the samples above cover -- \
             counters are only sampled every --steady-window iters, so per-iter figures \
             here are divided by {}, not by {}",
            result.counter_iters, sample_iters, result.counter_iters, sample_iters
        );
    }
    if result.counters_include_warmup {
        println!(
            "  WARNING: steady state was reached before the first interior counter \
             snapshot, so every counter and cpu figure above spans the whole run and \
             INCLUDES the warm-up transient. Lower --steady-window to separate them."
        );
    }
    // A zero-dispatch row has two very different causes and the reader cannot
    // tell them apart from the row alone, so say which one it was.
    if result.steady_pool.dispatches == 0 {
        if any_route_observed(&result.routes) {
            println!(
                "  ATTRIBUTION:   this model never dispatched to task_runtime -- the pool \
                 row above is a true zero, not a dead counter. Routes below say what ran \
                 instead; pool numbers here describe a pool this model does not use."
            );
        } else {
            println!(
                "  ATTRIBUTION:   zero dispatches AND zero routes recorded. The ledger saw \
                 nothing, so this is a dead instrument, not a measurement -- do not quote \
                 the pool row."
            );
        }
    }
    for route in &result.routes {
        println!(
            "  route:         {} -> {} ({}, {} calls, up to {} threads)",
            route.family, route.backend, route.dtype, route.calls, route.max_threads
        );
    }
    if dispatch_ledger::dropped() > 0 {
        println!(
            "  route:         WARNING {} observations dropped (ledger full); the route list \
             is truncated and call counts are lower bounds",
            dispatch_ledger::dropped()
        );
    }
    println!(
        "  whole run:     {:.3} cpu-s over {:.3} wall-s (warm-up included)",
        result.total_metrics.cpu_us() as f64 / 1e6,
        result.total_wall_s
    );
    if result.steady_start.is_none() && result.detected {
        println!(
            "  WARNING: no steady state in {} iterations at gap {}us. Every number \
             above describes the warm-up transient. Raise --iters.",
            result.samples_ms.len(),
            args.gap_us
        );
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.iters == 0 {
        bail!("--iters must be greater than zero");
    }
    if args.sessions == 0 {
        bail!("--sessions must be greater than zero");
    }
    GapKind::parse(&args.gap_kind).map_err(anyhow::Error::msg)?;

    if args.native_threads > 0 {
        // SAFETY: single-threaded startup, before any session, thread pool or
        // other reader of the process environment exists.
        unsafe { std::env::set_var(NATIVE_DECODE_THREADS_ENV, args.native_threads.to_string()) };
    }
    let native_threads = std::env::var(NATIVE_DECODE_THREADS_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "default".to_string());

    println!("model:  {}", args.model.display());
    println!(
        "arm:    {} (build={}, decode_threads={native_threads}, sessions={})",
        args.arm,
        build_arm(),
        args.sessions
    );
    println!(
        "gap:    {}us +/-{:.0}% {} (seed {})",
        args.gap_us,
        args.gap_jitter * 100.0,
        args.gap_kind,
        args.gap_seed
    );

    match args.arm.as_str() {
        "native" => {
            let result = run_native_arm(&args, "native")?;
            report("native", &result, &args);
            emit_result_line("native", &result);
            if let Some(path) = &args.dump_csv {
                dump_csv(path, "native", &result)?;
            }
        }
        "ort" => {
            let result = run_ort_arm(&args)?;
            report("ort", &result, &args);
            emit_result_line("ort", &result);
            if let Some(path) = &args.dump_csv {
                dump_csv(path, "ort", &result)?;
            }
        }
        "null" => {
            let first = run_native_arm(&args, "null-a")?;
            report("null-a", &first, &args);
            let second = run_native_arm(&args, "null-b")?;
            report("null-b", &second, &args);
            if let Some(path) = &args.dump_csv {
                dump_csv(path, "null-a", &first)?;
                dump_csv(path, "null-b", &second)?;
            }
            let ratio = second.summary().p50 / first.summary().p50;
            println!(
                "\nresult: null_b_over_a={ratio:.3}x  (a={:.4} ms, b={:.4} ms)  \
                 -- any A/B ratio inside this band is noise",
                first.summary().p50,
                second.summary().p50
            );
        }
        other => bail!("unknown --arm '{other}'; expected native, ort or null"),
    }

    if args.census {
        let threads = thread_census();
        println!("\n== thread census ({} threads) ==", threads.len());
        println!("{:>28}  {:>6}  {:>12}", "name", "count", "cpu_ms");
        for (name, count, cpu_us) in census_by_name(&threads) {
            println!("{name:>28}  {count:>6}  {:>12.1}", cpu_us as f64 / 1000.0);
        }
    }

    println!(
        "\nfor futex/sched_yield attribution, re-run under:\n  strace -c -f -e trace=futex,sched_yield <this exact command>"
    );
    Ok(())
}

/// Appends this arm's whole sample series to `path`, creating it with a
/// header if it does not exist.
fn dump_csv(path: &std::path::Path, arm: &str, result: &ArmResult) -> Result<()> {
    use std::io::Write;
    let fresh = !path.exists();
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("open {} for append", path.display()))?;
    if fresh {
        writeln!(file, "arm,iteration,wall_ms,in_steady_window")?;
    }
    let steady_start = result.steady_start.unwrap_or(usize::MAX);
    for (index, sample) in result.samples_ms.iter().enumerate() {
        writeln!(
            file,
            "{arm},{index},{sample:.6},{}",
            u8::from(index >= steady_start)
        )?;
    }
    Ok(())
}

fn emit_result_line(arm: &str, result: &ArmResult) {
    let summary = result.summary();
    println!(
        "result: arm={arm} p50={:.4} ms p90={:.4} ms cpu_per_wall={:.2} sys_share={:.3} \
         vol_ctxt_per_iter={:.2} \
         parks_per_iter={:.2} spin_hits_per_iter={:.2} spin_yields_per_iter={:.2} \
         steady_iters={} rss_kb={}",
        summary.p50,
        summary.p90,
        result.cpu_per_wall(),
        {
            let cpu = result.steady_metrics.cpu_us();
            if cpu > 0 {
                result.steady_metrics.sys_us as f64 / cpu as f64
            } else {
                0.0
            }
        },
        result.parks_per_iter(),
        result.steady_pool.parks as f64 / summary.count.max(1) as f64,
        result.steady_pool.spin_hits as f64 / summary.count.max(1) as f64,
        result.steady_pool.spin_yields as f64 / summary.count.max(1) as f64,
        summary.count,
        result.steady_metrics.rss_kb
    );
}

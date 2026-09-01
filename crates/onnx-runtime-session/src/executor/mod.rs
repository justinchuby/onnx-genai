//! The sequential CPU executor (Track D, `docs/architecture/ORT2.md` §20, §11.3).
//!
//! Turns a loaded [`Graph`] plus its live [`WeightStore`] into a runnable plan:
//! resolve every value's concrete shape from the actual bound inputs at
//! `run`, size a device buffer per value from those *resolved* shapes, resolve
//! a kernel per node through the execution provider (keyed by the resolved
//! input shapes), then walk the topological order binding
//! [`TensorView`]/[`TensorMut`] windows over those buffers and invoking each
//! kernel. It is generic over any [`Graph`] and any [`ExecutionProvider`]; the
//! Phase-1 build wires in the CPU EP only, but nothing here is op- or
//! model-specific.
//!
//! ## Symbolic → concrete shape resolution (§3.2, §11)
//!
//! Real models carry *symbolic* input dims (e.g. `batch`, `max_seq_len`): the
//! loader produces a [`Shape`] whose dims are a mix of [`Dim::Static`] and
//! [`Dim::Symbolic`]. This executor is model-agnostic about them — a symbol is
//! whatever [`SymbolId`] the graph interned. At `run` it reads the actual shape
//! of each bound input, **binds** the graph's symbols to concrete sizes from
//! those inputs (conflicts across inputs are an error), and **substitutes**
//! those bindings into every value's loader shape to obtain a fully-concrete
//! shape. Buffers are sized from the resolved shapes and become run-scoped when
//! shapes are dynamic (reused when the resolved shape is unchanged, re-allocated
//! when it changes). A fully-static graph is simply the special case where
//! there are no symbols: resolution is a no-op and every buffer/kernel is
//! materialized once at build.
//!
//! The session does **not** infer op output shapes — that is the loader's job
//! (the loader runs `onnx-runtime-shape-inference` at load time). If a value's
//! loader shape still contains an unbound symbol after substitution, the
//! session resolves genuinely data-dependent extents just-in-time during
//! execution (see [`dynamic_output_shapes`]); anything it still cannot size is
//! reported as [`SessionError::UnresolvedShape`] naming the value and its
//! producing op, rather than guessing.
//!
//! ## Holden's precondition (ep-api safety review #1) — enforced here
//!
//! A [`TensorView`] carries no backing length, so it cannot self-check storage
//! bounds. This executor owns every buffer, so it is the layer that *can*: for
//! **every** input and output view of **every** node it calls
//! [`strided::view_in_bounds`] (or, for sub-byte dtypes, the `storage_bytes`
//! equivalent in [`view_bounds`]) against the **run-scoped resolved** buffer and
//! refuses to dispatch on failure. That check is the sole thing that makes
//! ep-cpu's unchecked pointer derefs sound.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use onnx_runtime_ep_api::{
    CaptureRegionShapeStatus, DeviceBuffer, DeviceGraphOwner, DeviceGraphSlot, DeviceGraphToken,
    DevicePtr, DevicePtrMut, DeviceValidationRegistration, DeviceValidationToken, EpError,
    ExecutionProvider, ExecutorArtifactFinalization, ExecutorArtifactPending,
    ExecutorArtifactReadinessEpoch, ExecutorInstanceId, ExecutorResidencyTelemetry,
    ExternalMmapRegion, FinalizedExpertBank, FinalizedExpertWeight, Kernel, KernelConstantInput,
    KernelInput, KernelMatch, LazyWeight, LazyWeightBoundary, ResidentWeight,
    StructuralCaptureDecline, TensorBacking, TensorMetadata, TensorMut, TensorView, WeightHandle,
    WorkspaceAllocation, WorkspaceLifetime, WorkspaceRequirement, WorkspaceView,
    expert_weight_groups, lazy_weight_candidates,
};
use smallvec::SmallVec;

type OptionalTensorSpecs = Vec<Option<(DataType, Vec<usize>)>>;
type ScopedOutputs = SmallVec<[Option<SessionOutput>; 16]>;
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ep_cpu::strided::view_in_bounds;
use onnx_runtime_ir::Attribute;
use onnx_runtime_ir::{
    DataType, DeviceType, Dim, Graph, Node, NodeId, Shape, SymbolId, TensorLayout, ValueId,
    WeightRef, as_static_shape, broadcast_shapes, compute_contiguous_strides, read_scalar_le,
};
use onnx_runtime_loader::WeightStore;
use onnx_runtime_memory::{PlanOptions, PlanStatus, ViewMap, plan_activations};
use onnx_runtime_optimizer::InitializerResolver;
use onnx_runtime_shape_inference::{
    DimExpr, InferenceRegistry, MAX_SHAPE_DATA_ELEMS, MergePolicy, NodeIo, ShapeData,
    SymbolInterner, TypeInfo,
};
use onnx_runtime_tracer::{Args, SpanGuard, TraceContext, annotate_current_span_with};

use crate::SessionOutput;
use crate::error::{Result, SessionError};
use crate::sequence::{
    ConcatPlan, SeqTensor, SequenceError, SequenceValue, SplitSpec, split_tensor, stack_new_axis,
};
use crate::tensor::{DeviceBindingSpec, DeviceIoBinding, SharedTensorBuffer, Tensor};

pub(super) struct DeviceValidationSubmission {
    ep: Arc<dyn ExecutionProvider>,
    token: DeviceValidationToken,
    active: bool,
}

impl DeviceValidationSubmission {
    pub(super) fn begin(
        ep: &Arc<dyn ExecutionProvider>,
        registration: &DeviceValidationRegistration,
    ) -> Result<Self> {
        let token = ep.begin_device_validation(registration)?;
        Ok(Self {
            ep: Arc::clone(ep),
            token,
            active: true,
        })
    }

    pub(super) fn token(&self) -> DeviceValidationToken {
        self.token
    }

    pub(super) fn add_recipient(&self, binding: &mut DeviceIoBinding) -> Result<()> {
        if binding.output_name().is_none() {
            return Ok(());
        }
        let token = self
            .ep
            .add_device_validation_recipient(self.token, binding.validation_registration())?;
        binding.set_device_validation(token);
        Ok(())
    }

    pub(super) fn activate(&self) -> Result<()> {
        self.ep.activate_device_validation(self.token)?;
        Ok(())
    }

    pub(super) fn disarm(&mut self) {
        self.active = false;
    }
}

impl Drop for DeviceValidationSubmission {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let result = self.ep.sync().map_err(SessionError::from).and_then(|()| {
            self.ep
                .abort_device_validation_submission(self.token)
                .map_err(SessionError::from)
        });
        match result {
            Ok(0) => {}
            Ok(flags) => eprintln!(
                "[onnx-runtime-session] recovered device validation failure while unwinding: \
                 provider={} flags=0x{flags:x}",
                self.ep.name()
            ),
            Err(error) => eprintln!(
                "[onnx-runtime-session] could not recover deferred device validation while \
                 unwinding provider={}: {error}",
                self.ep.name()
            ),
        }
    }
}

fn profile_ops_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("ONNX_GENAI_PROFILE_OPS")
            .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
    })
}

/// Low-overhead, env-gated (`NXRT_EXEC_PHASE_PROFILE=1`) phase profiler for the
/// executor's control-flow / subgraph-dispatch machinery. When disabled every
/// entry point is a single relaxed-atomic load and an early return, so the
/// production decode hot path pays no measurable cost. When enabled it
/// accumulates wall-clock nanoseconds and a call count per named phase, which
/// [`executor_phase_stats`] exposes. This exists to attribute the
/// per-decode-step control-flow overhead (`exec_if` / `run_subgraph` / child
/// setup) that the op-level profiler folds into the single `If` bucket.
mod phase_profile {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
    use std::sync::{Mutex, OnceLock};
    use std::time::Instant;

    pub(super) const UNKNOWN: u8 = 0;
    pub(super) const OFF: u8 = 1;
    pub(super) const ON: u8 = 2;

    static STATE: AtomicU8 = AtomicU8::new(UNKNOWN);
    static PRINTED: AtomicBool = AtomicBool::new(false);

    /// Publish an environment-derived gate value without overwriting a decision
    /// that is already in force, and report the value that actually is.
    ///
    /// The lazy initialiser below is a *writer*, not just a reader, and it runs
    /// on whatever thread first reaches the gate while holding no lock -
    /// `run.rs` consults the planner gate on every executor run, so under the
    /// parallel test runner that is any thread at all. Its load / read-env /
    /// store is not atomic: an unconditional `store` can land *after* a
    /// `force_*` call that did take [`globals_lock`], silently reverting it.
    /// That is not hypothetical - it is how
    /// `activation_memory_planner_reports_static_decode_graph_savings` failed
    /// on `Rust coverage (Windows x86_64)`: the gate it forced on was reset to
    /// `OFF` between the force and the run, so the run published no stats and
    /// the `expect` on them panicked.
    ///
    /// [`globals_lock`] cannot fix that, because the losing writer is a plain
    /// reader that must not be made to take a lock on the hot path. A
    /// `compare_exchange` from [`UNKNOWN`] fixes it instead, by making the
    /// initialiser lose the race rather than win it: it publishes only when
    /// nothing else has, and otherwise reports what is in force.
    pub(super) fn publish_env_derived(gate: &AtomicU8, on: bool) -> bool {
        let desired = if on { ON } else { OFF };
        match gate.compare_exchange(UNKNOWN, desired, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => on,
            Err(in_force) => in_force == ON,
        }
    }

    pub fn enabled() -> bool {
        match STATE.load(Ordering::Relaxed) {
            OFF => false,
            ON => true,
            _ => {
                let on = std::env::var("NXRT_EXEC_PHASE_PROFILE")
                    .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
                publish_env_derived(&STATE, on)
            }
        }
    }

    /// Test-only override of the env-derived enable state.
    #[cfg(test)]
    pub(super) fn force_enabled(on: bool) {
        STATE.store(if on { ON } else { OFF }, Ordering::Relaxed);
    }

    static PLAN_STATE: AtomicU8 = AtomicU8::new(UNKNOWN);

    /// Whether to run the activation-memory *planner* during a run.
    ///
    /// Deliberately **not** `enabled()`. The planner is a separate instrument:
    /// it rebuilds the view map and re-plans every activation on every run,
    /// work the shipped runtime never does. Its own span measures 1.9-6.0us
    /// per run, so charging it to `--phase-profile` meant the profiler
    /// perturbed the run it was measuring and reported its own cost back as a
    /// phase of that run. Removing it lowers `native_min` on a softmax decode
    /// from 0.065ms to 0.063ms, median of 15 interleaved repetitions - about
    /// 3%, in line with the span.
    ///
    /// Reading the environment directly, rather than `enabled()`, is what
    /// separates the two: `enable_for_process()` (what `--phase-profile` calls)
    /// no longer switches the planner on, while anyone who set
    /// `NXRT_EXEC_PHASE_PROFILE=1` in the environment - the CLI memory report's
    /// path to these stats - keeps getting them.
    pub fn activation_plan_enabled() -> bool {
        match PLAN_STATE.load(Ordering::Relaxed) {
            OFF => false,
            ON => true,
            _ => {
                let on = ["NXRT_ACTIVATION_MEMORY_PLAN", "NXRT_EXEC_PHASE_PROFILE"]
                    .iter()
                    .any(|key| {
                        std::env::var(key).is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                    });
                publish_env_derived(&PLAN_STATE, on)
            }
        }
    }

    /// Opt a process into activation-memory planning without turning on phase
    /// profiling, for callers that want the stats and are willing to pay.
    pub fn enable_activation_plan_for_process() {
        PLAN_STATE.store(ON, Ordering::Relaxed);
    }

    /// Test-only override of the env-derived planner state.
    #[cfg(test)]
    pub(super) fn force_activation_plan_enabled(on: bool) {
        PLAN_STATE.store(if on { ON } else { OFF }, Ordering::Relaxed);
    }

    /// The live planner gate, so a test can drive the real one rather than a
    /// copy of it.
    #[cfg(test)]
    pub(super) fn activation_plan_gate() -> &'static AtomicU8 {
        &PLAN_STATE
    }

    /// Holds the planner on for one test and clears it on drop, so a leaked
    /// gate cannot be observed by a later test that does not take the lock.
    ///
    /// The field is never read on purpose: it is the [`globals_lock`] guard,
    /// and holding it is the whole point.  `dead_code` cannot see that a
    /// `MutexGuard`'s value is its `Drop`, so it has to be told.
    #[cfg(test)]
    pub(super) struct ActivationPlanForTest(#[allow(dead_code)] std::sync::MutexGuard<'static, ()>);

    #[cfg(test)]
    impl ActivationPlanForTest {
        pub(super) fn on() -> Self {
            let guard = globals_lock();
            force_activation_plan_enabled(true);
            Self(guard)
        }
    }

    #[cfg(test)]
    impl Drop for ActivationPlanForTest {
        fn drop(&mut self) {
            force_activation_plan_enabled(false);
        }
    }

    /// Serialises the tests that write the two process-global gates above.
    /// Both are plain atomics with no per-test isolation, so without this the
    /// parallel runner lets one test's `enable`/`force` land inside another
    /// test's assertion window.
    ///
    /// It does **not** serialise every writer, and cannot: the lazy
    /// initialiser in `enabled`/`activation_plan_enabled` also writes, from any
    /// thread that reaches a gate first, and making a hot-path reader take a
    /// mutex is not an option. Holding this lock is therefore *not* sufficient
    /// to own a gate - see [`publish_env_derived`], which is what keeps that
    /// third writer from reverting a forced value.
    #[cfg(test)]
    pub(super) fn globals_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn enable_for_process() {
        STATE.store(ON, Ordering::Relaxed);
    }

    /// Test-only snapshot of a phase's accumulated `(total_ns, count)`.
    #[cfg(test)]
    pub(super) fn snapshot(phase: &'static str) -> Option<(u128, u64)> {
        registry()
            .lock()
            .ok()
            .and_then(|reg| reg.get(phase).map(|s| (s.total_ns, s.count)))
    }

    #[derive(Default, Clone, Copy)]
    struct PhaseStat {
        total_ns: u128,
        count: u64,
    }

    fn registry() -> &'static Mutex<BTreeMap<&'static str, PhaseStat>> {
        static REGISTRY: OnceLock<Mutex<BTreeMap<&'static str, PhaseStat>>> = OnceLock::new();
        REGISTRY.get_or_init(|| Mutex::new(BTreeMap::new()))
    }

    /// Accumulate `nanos` against `phase`. No-op unless the profiler is enabled.
    pub fn record(phase: &'static str, nanos: u128) {
        if !enabled() {
            return;
        }
        if let Ok(mut reg) = registry().lock() {
            let entry = reg.entry(phase).or_default();
            entry.total_ns += nanos;
            entry.count += 1;
        }
    }

    /// Every recorded phase, most expensive first.
    ///
    /// Exposed as data rather than rendered here: this crate has no dependency
    /// on the process-wide stage profiler that everything else reports through,
    /// and reversing that would point the native runtime at the ONNX Runtime
    /// crate. The caller that already depends on both does the merging.
    pub fn all_stats() -> Vec<(&'static str, u128, u64)> {
        let Ok(reg) = registry().lock() else {
            return Vec::new();
        };
        let mut rows = reg
            .iter()
            .map(|(phase, stat)| (*phase, stat.total_ns, stat.count))
            .collect::<Vec<_>>();
        rows.sort_by_key(|row| std::cmp::Reverse(row.1));
        rows
    }

    pub fn reset() {
        if let Ok(mut reg) = registry().lock() {
            reg.clear();
        }
        PRINTED.store(false, Ordering::Relaxed);
    }

    /// Scoped timer that records its lifetime to `phase` on drop.
    pub struct PhaseSpan {
        phase: &'static str,
        start: Option<Instant>,
    }

    impl PhaseSpan {
        pub fn new(phase: &'static str) -> Self {
            let active = enabled();
            Self {
                phase,
                // Avoid the clock read entirely on the disabled hot path.
                start: if active { Some(Instant::now()) } else { None },
            }
        }
    }

    impl Drop for PhaseSpan {
        fn drop(&mut self) {
            if let Some(start) = self.start {
                record(self.phase, start.elapsed().as_nanos());
            }
        }
    }

    /// Render and reset the accumulated per-phase table to stderr. Called once at
    /// process exit (or on demand) so a single line-oriented dump is available.
    pub fn report_to_stderr() {
        if !enabled() {
            return;
        }
        let rows: Vec<(&'static str, PhaseStat)> = match registry().lock() {
            Ok(reg) => reg.iter().map(|(n, s)| (*n, *s)).collect(),
            Err(_) => return,
        };
        if PRINTED.swap(true, Ordering::Relaxed) {
            return;
        }
        let mut rows = rows;
        rows.sort_by_key(|r| std::cmp::Reverse(r.1.total_ns));
        eprintln!("[nxrt-phase] phase,total_ms,calls,us/call");
        for (name, stat) in &rows {
            if name.ends_with("_bytes") {
                continue;
            }
            let total_ms = stat.total_ns as f64 / 1_000_000.0;
            let us_per_call = if stat.count > 0 {
                (stat.total_ns as f64 / 1_000.0) / stat.count as f64
            } else {
                0.0
            };
            eprintln!(
                "[nxrt-phase] {name},{total_ms:.3},{},{us_per_call:.2}",
                stat.count
            );
        }
        // Byte-valued counters (host traffic) are reported separately in MB.
        for (name, stat) in &rows {
            if !name.ends_with("_bytes") {
                continue;
            }
            let total_mb = stat.total_ns as f64 / (1024.0 * 1024.0);
            let mb_per_call = if stat.count > 0 {
                total_mb / stat.count as f64
            } else {
                0.0
            };
            eprintln!(
                "[nxrt-phase] {name},total_mb={total_mb:.1},calls={},mb/call={mb_per_call:.3}",
                stat.count
            );
        }
    }
}

/// Open an env-gated executor phase-profiling span (see [`phase_profile`]).
macro_rules! phase_span {
    ($phase:expr) => {
        phase_profile::PhaseSpan::new($phase)
    };
}

fn trace_span(name: &'static str, cat: &'static str) -> Option<SpanGuard> {
    onnx_runtime_tracer::global_context()
        .filter(|trace| trace.is_enabled())
        .map(|trace| trace.span(name, cat))
}

/// Public re-export so the bench/profile harness can dump the phase table.
/// Accumulated executor phase costs as `(phase, total_ns, calls)`, most
/// expensive first.
///
/// Empty unless `NXRT_EXEC_PHASE_PROFILE` is set. Kept separate from the
/// process-wide stage profiler because this crate cannot depend on it without
/// pointing the native runtime at the ONNX Runtime crate; the caller that
/// depends on both merges the two.
pub fn exec_phase_stats() -> Vec<(&'static str, u128, u64)> {
    phase_profile::all_stats()
}

pub fn print_exec_phase_profile() {
    phase_profile::report_to_stderr();
}

pub fn reset_exec_phase_profile() {
    phase_profile::reset();
}

/// Force executor phase profiling on for targeted per-step attribution.
///
/// This is deliberately process-wide because the phase profiler is itself
/// process-wide. Callers use it only under explicit diagnostic env knobs before
/// resetting and sampling a single decode step; production paths should leave
/// the cheap env-gated default untouched.
pub fn enable_exec_phase_profile_for_process() {
    phase_profile::enable_for_process();
}

/// Opt this process into the activation-memory planner, which runs on every
/// `Run` and is *not* enabled by [`enable_exec_phase_profile_for_process`]
/// because it costs enough to distort the run it would be reported against.
pub fn enable_activation_memory_plan_for_process() {
    phase_profile::enable_activation_plan_for_process();
}

/// Activation-memory planner metrics from the most recent measured top-level run.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActivationMemoryPlanStats {
    /// `true` when every activation owner had a concrete byte size and the
    /// planner produced a reusable-slot plan.
    pub complete: bool,
    /// Concurrent peak bytes after liveness-based slot sharing.
    pub peak_bytes: usize,
    /// Planner upper bound that counts one buffer-owner allocation per activation.
    /// This is not the executor's exact baseline: in-place aliases and sequence
    /// storage can make the current executor allocate less.
    pub naive_bytes: usize,
    /// Fraction saved vs. the naive baseline: `1 - peak / naive`.
    pub savings_ratio: f64,
    /// Number of reusable backing slots in the complete plan.
    pub num_slots: usize,
    /// Number of buffer-owner values assigned to slots.
    pub assignments: usize,
    /// Number of zero-copy view aliases folded into source-owner liveness.
    pub view_edges: usize,
    /// Number of activation owners still missing concrete sizes when deferred.
    pub unknown_sizes: usize,
}

pub(crate) fn host_dtype_alignment(dtype: DataType) -> usize {
    match dtype {
        DataType::Float16 | DataType::BFloat16 | DataType::Int16 | DataType::Uint16 => 2,
        DataType::Float32 | DataType::Int32 | DataType::Uint32 | DataType::Complex64 => 4,
        DataType::Float64 | DataType::Int64 | DataType::Uint64 | DataType::Complex128 => 8,
        _ => 1,
    }
}

fn print_op_profile(total: Duration, timings: HashMap<String, (Duration, usize)>) {
    let mut timings = timings.into_iter().collect::<Vec<_>>();
    timings.sort_unstable_by_key(|entry| std::cmp::Reverse(entry.1.0));
    let total_ms = total.as_secs_f64() * 1_000.0;
    eprintln!("[onnx-genai-profile] node execution: {total_ms:.3} ms");
    eprintln!("[onnx-genai-profile] op_type,total_ms,percent,calls");
    for (op_type, (elapsed, calls)) in timings {
        let elapsed_ms = elapsed.as_secs_f64() * 1_000.0;
        let percent = if total_ms == 0.0 {
            0.0
        } else {
            elapsed_ms / total_ms * 100.0
        };
        eprintln!("[onnx-genai-profile] {op_type},{elapsed_ms:.3},{percent:.2},{calls}");
    }
}

static DENSE_PREFETCH_GAP_JOINS: AtomicU64 = AtomicU64::new(0);
static DENSE_PREFETCH_GAP_NODES: AtomicU64 = AtomicU64::new(0);
static DENSE_PREFETCH_GAP_MAX: AtomicU64 = AtomicU64::new(0);

pub const DENSE_WEIGHT_PREFETCH_LOOKAHEAD_ENV: &str =
    "ONNX_GENAI_WEIGHT_OFFLOAD_PREFETCH_LOOKAHEAD_NODES";
const DEFAULT_DENSE_WEIGHT_PREFETCH_LOOKAHEAD_NODES: usize = 1;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DensePrefetchGapStats {
    pub joins: u64,
    pub nodes_between_sum: u64,
    pub nodes_between_max: u64,
}

pub fn dense_prefetch_gap_stats() -> DensePrefetchGapStats {
    DensePrefetchGapStats {
        joins: DENSE_PREFETCH_GAP_JOINS.load(Ordering::Relaxed),
        nodes_between_sum: DENSE_PREFETCH_GAP_NODES.load(Ordering::Relaxed),
        nodes_between_max: DENSE_PREFETCH_GAP_MAX.load(Ordering::Relaxed),
    }
}

pub fn dense_weight_prefetch_lookahead_nodes() -> usize {
    std::env::var(DENSE_WEIGHT_PREFETCH_LOOKAHEAD_ENV)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_DENSE_WEIGHT_PREFETCH_LOOKAHEAD_NODES)
}

pub fn reset_dense_prefetch_gap_stats() {
    DENSE_PREFETCH_GAP_JOINS.store(0, Ordering::Relaxed);
    DENSE_PREFETCH_GAP_NODES.store(0, Ordering::Relaxed);
    DENSE_PREFETCH_GAP_MAX.store(0, Ordering::Relaxed);
}

fn record_dense_prefetch_gap(nodes_between: u64) {
    DENSE_PREFETCH_GAP_JOINS.fetch_add(1, Ordering::Relaxed);
    DENSE_PREFETCH_GAP_NODES.fetch_add(nodes_between, Ordering::Relaxed);
    let mut current = DENSE_PREFETCH_GAP_MAX.load(Ordering::Relaxed);
    while nodes_between > current {
        match DENSE_PREFETCH_GAP_MAX.compare_exchange_weak(
            current,
            nodes_between,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
}

/// A per-node compiled entry: the structural facts the run loop needs without
/// re-deriving them from the graph. Shapes are **not** baked here — they are
/// resolved per run from the bound inputs (see module docs).
#[derive(Debug)]
pub(crate) struct NodePlan {
    pub node_id: NodeId,
    /// Positional input value ids in ONNX signature order. An omitted optional
    /// input (ONNX empty-string input name → `None` slot) is preserved as
    /// `None` so a later present input is never misread as the omitted one
    /// (e.g. `Slice(data, starts, ends, "", steps)`). Trailing `None`s are
    /// trimmed — a truly absent trailing optional simply lowers the arity.
    pub inputs: Vec<Option<ValueId>>,
    /// Output value ids, in positional order.
    pub outputs: Vec<ValueId>,
    /// Element types of the inputs, positional (matches `inputs`). An omitted
    /// optional (`None`) slot carries [`DataType::Undefined`] so EP claim-time
    /// validation can distinguish it from a supplied tensor.
    pub input_dtypes: Vec<DataType>,
    /// Element types of the outputs.
    pub output_dtypes: Vec<DataType>,
    /// Inputs consumed for the final time by this node and therefore eligible for
    /// a kernel-authorized in-place overwrite after additional runtime guards.
    pub inplace_dead_inputs: Vec<bool>,
    /// Lazy weight inputs this node may ask the EP to page at dispatch time.
    pub lazy_weight_inputs: Vec<ValueId>,
    /// Intermediate values whose last consumer is this node, and whose buffers
    /// may therefore be released once it has run.
    ///
    /// Distinct from [`NodePlan::inplace_dead_inputs`], which only ever lets a
    /// kernel overwrite an input whose shape and dtype already match its output.
    /// That covers elementwise chains and nothing else, so before this list
    /// existed every intermediate buffer a graph produced stayed resident for the
    /// whole run: a 2545-node vision encoder held all 2545 of them at once and
    /// spent ~20x the memory its live set needs.
    pub dead_after: Vec<ValueId>,
}

/// Map a [`crate::sequence::SequenceError`] into an actionable `SessionError`.
fn seq_err(e: crate::sequence::SequenceError) -> SessionError {
    e.into()
}

/// Normalize a possibly-negative ONNX `axis` against `rank`, returning the
/// non-negative axis or `None` when out of `[-rank, rank-1]`.
fn normalize_axis(axis: i64, rank: usize) -> Option<usize> {
    let r = rank as i64;
    let a = if axis < 0 { axis + r } else { axis };
    if a < 0 || a >= r {
        None
    } else {
        Some(a as usize)
    }
}

fn scan_list_attr(node: &Node, name: &str, count: usize, default: i64) -> Result<Vec<i64>> {
    match node.attr(name) {
        None => Ok(vec![default; count]),
        Some(attr) => {
            let values = attr.as_ints().ok_or_else(|| SessionError::ControlFlow {
                op: "Scan".to_string(),
                reason: format!("attribute '{name}' must be an INTS list"),
            })?;
            if values.len() != count {
                return Err(SessionError::ControlFlow {
                    op: "Scan".to_string(),
                    reason: format!(
                        "attribute '{name}' has {} value(s), expected {count}",
                        values.len()
                    ),
                });
            }
            Ok(values.to_vec())
        }
    }
}

/// Whether `(op_type, domain)` is one of the standard subgraph-bearing
/// control-flow ops the executor handles recursively (default `ai.onnx`
/// domain). Kept in lock-step with the loader's `validate_no_control_flow`
/// allow-list.
pub(crate) fn is_control_flow_op(op_type: &str, domain: &str) -> bool {
    domain.is_empty() && matches!(op_type, "If" | "Loop" | "Scan")
}

/// Whether `(op_type, domain)` is an ONNX **Sequence** op the executor handles
/// directly (default `ai.onnx` domain). Like control-flow ops these are handled
/// at the executor level rather than as leaf [`Kernel`](onnx_runtime_ep_api::Kernel)s
/// because a `Kernel` sees only tensor views, never a *sequence-of-tensors*
/// runtime value. Kept as a small self-contained routing predicate (mirroring
/// [`is_control_flow_op`]) so it never collides with the EP kernel registry.
pub(crate) fn is_sequence_op(op_type: &str, domain: &str) -> bool {
    domain.is_empty()
        && matches!(
            op_type,
            "SequenceEmpty"
                | "SequenceConstruct"
                | "SequenceInsert"
                | "SequenceErase"
                | "SequenceAt"
                | "SequenceLength"
                | "SplitToSequence"
                | "ConcatFromSequence"
        )
}

fn heterogeneous_api_error(operation: &str) -> SessionError {
    SessionError::HeterogeneousExecutionUnsupported {
        placement_summary: format!(
            "{operation} requires persistent external state or device-graph capture, which the \
             first heterogeneous execution slice deliberately rejects before execution"
        ),
    }
}

/// Whether a Sequence op yields a *sequence* value (vs. a tensor). Used at build
/// to mark sequence-typed values so they are excluded from tensor buffer sizing.
fn produces_sequence_output(op_type: &str, domain: &str) -> bool {
    domain.is_empty()
        && matches!(
            op_type,
            "SequenceEmpty"
                | "SequenceConstruct"
                | "SequenceInsert"
                | "SequenceErase"
                | "SplitToSequence"
        )
}

/// Read a single scalar `i64` element from a length-1 tensor (Loop's `M`).
fn tensor_scalar_i64(t: &Tensor) -> Option<i64> {
    if t.dtype != DataType::Int64 || t.numel() != 1 {
        return None;
    }
    read_scalar_le(t.as_bytes()).ok()
}

/// Read a single scalar bool from a length-1 `BOOL` tensor (a `BOOL` is one
/// byte; any nonzero is true, per ONNX).
fn tensor_scalar_bool(t: &Tensor) -> Option<bool> {
    if t.dtype != DataType::Bool || t.numel() != 1 {
        return None;
    }
    t.as_bytes().first().map(|&b| b != 0)
}

/// Build a length-1 `i64` scalar tensor (Loop's `iter_num` body input).
fn scalar_i64_tensor(v: i64) -> Result<Tensor> {
    Tensor::from_raw(DataType::Int64, vec![], &v.to_le_bytes())
}

/// Build a scalar `BOOL` tensor (Loop's `cond` body input).
fn scalar_bool_tensor(v: bool) -> Result<Tensor> {
    Tensor::from_raw(DataType::Bool, vec![], &[u8::from(v)])
}

fn missing_capture_error(attr_key: &str, name: &str) -> SessionError {
    SessionError::Internal(format!(
        "control-flow body '{attr_key}' captures free variable '{name}', but it is not \
         available in the enclosing scope. RULES #1: a subgraph may only reference outer \
         values that are graph inputs, initializers, or produced by an upstream node in an \
         enclosing graph; '{name}' matches none of these"
    ))
}

/// Names a graph or any nested body needs from its enclosing lexical scope.
/// A nested requirement stops propagating when this graph defines that name,
/// because the nested body will bind the local value at execution time.
fn required_outer_names(graph: &Graph) -> HashSet<String> {
    let formal_set: HashSet<ValueId> = graph.inputs.iter().copied().collect();
    let local_names: HashSet<&str> = graph
        .values
        .iter()
        .filter_map(|(_, value)| value.name.as_deref())
        .collect();
    let mut required = HashSet::new();
    for (vid, value) in graph.values.iter() {
        if value.producer.is_none()
            && !formal_set.contains(&vid)
            && !graph.initializers.contains_key(&vid)
            && let Some(name) = &value.name
        {
            required.insert(name.clone());
        }
    }
    for nested in graph.subgraphs.values() {
        for name in required_outer_names(nested) {
            if !local_names.contains(name.as_str()) {
                required.insert(name);
            }
        }
    }
    required
}

impl Drop for Executor {
    fn drop(&mut self) {
        // Observability (F5): a one-line memo activity summary when
        // `ONNX_GENAI_DECODE_MEMO_STATS=1`, so an on-model A/B can confirm the
        // memo actually fired (`replayed>0`) rather than being silently gated out.
        if self.decode_memo_enabled
            && std::env::var("ONNX_GENAI_DECODE_MEMO_STATS")
                .map(|v| matches!(v.as_str(), "1" | "true" | "on"))
                .unwrap_or(false)
        {
            let (primed, rebuilt, replayed, ineligible) = self.decode_memo_counts();
            let (views_reused, dispatch_elided) = self.decode_view_plan_counts();
            eprintln!(
                "[decode-memo] primed={primed} rebuilt={rebuilt} replayed={replayed} \
                 ineligible={ineligible} views_reused={views_reused} \
                 dispatch_elided={dispatch_elided}"
            );
        }
        // Evict the global weight-transpose cache to prevent address-reuse
        // staleness: if a subsequently loaded model's mmap recycles a virtual
        // address, the cache must not serve the old model's transposed weights.
        onnx_runtime_ep_cpu::kernels::matmul::clear_weight_transpose_caches();
        // Same lifetime boundary for the shared MLAS SQNBit packed weights
        // (#1056): they are keyed on the weight's address, so a later model whose
        // mmap recycles an address for a same-shaped weight must not inherit this
        // model's packed buffers.
        onnx_runtime_ep_cpu::kernels::matmul_nbits::clear_mlas_packed_caches();
        let mut safe_to_release = match self.ep.sync() {
            Ok(()) => true,
            Err(error) => {
                eprintln!(
                    "[onnx-runtime-session] executor drop could not synchronize deferred work: \
                     {error}"
                );
                false
            }
        };
        if safe_to_release && let Some(token) = self.pending_device_validation {
            match self.validation_registration.as_ref() {
                Some(registration) => {
                    match self.ep.consume_device_validation_error(registration, token) {
                        Ok(0) => {}
                        Ok(flags) => eprintln!(
                            "[onnx-runtime-session] executor drop consumed its deferred validation \
                             failure (flags=0x{flags:x})"
                        ),
                        Err(error) => {
                            safe_to_release = false;
                            eprintln!(
                                "[onnx-runtime-session] executor drop could not consume its \
                                 deferred validation: {error}"
                            );
                        }
                    }
                }
                None => {
                    safe_to_release = false;
                    eprintln!(
                        "[onnx-runtime-session] executor drop is missing its validation \
                         registration"
                    );
                }
            }
        }
        let mut graphs_reset = true;
        for cap in &mut self.slot_capture {
            if let Some(token) = cap.device_graph_token {
                match self.ep.reset_owned_device_graph(token) {
                    Ok(_) => cap.device_graph_token = None,
                    Err(error) => {
                        graphs_reset = false;
                        safe_to_release = false;
                        eprintln!(
                            "[onnx-runtime-session] executor drop could not reset graph \
                             {token:?}: {error}"
                        );
                    }
                }
            }
            cap.device_graph_signature = None;
        }
        if graphs_reset && let Err(error) = self.ep.retire_owned_device_graphs(self.graph_owner) {
            safe_to_release = false;
            eprintln!(
                "[onnx-runtime-session] executor drop could not retire graph owner {}: {error}",
                self.graph_owner.get()
            );
        }
        // Drain only this executor's provider-owned artifacts after its work,
        // validation generation, and exact graph tokens are retired. Sibling
        // and MTP executors sharing the EP retain their own artifact scopes.
        if let Err(error) = self.ep.drain_executor_artifacts(self.instance_id) {
            safe_to_release = false;
            eprintln!(
                "session: executor {} provider-artifact teardown was quarantined: {error}",
                self.instance_id.get()
            );
        }
        // Free every buffer via the owning EP (DeviceBuffer has no Drop).
        for (_, buf) in self.buffers.drain() {
            if safe_to_release {
                let _ = self.ep.deallocate(buf);
            } else {
                drop(buf);
            }
        }
        // An input buffer parked while its slot held a zero-copy borrow of the
        // caller's tensor is owned by this executor too. `unbind_borrowed_inputs`
        // returns them on every normal and error path, but a panic unwinding out
        // of a run drops the executor with them still parked.
        for (_, buf) in self.parked_input_buffers.drain(..) {
            if safe_to_release {
                let _ = self.ep.deallocate(buf);
            } else {
                drop(buf);
            }
        }
        if let Some(workspace) = self.persistent_workspace.take() {
            if safe_to_release {
                let _ = self.ep.deallocate_workspace(workspace.buffer);
            } else {
                drop(workspace);
            }
        }
        if let Some(workspace) = self.step_workspace.take() {
            if safe_to_release {
                let _ = self.ep.deallocate_workspace(workspace.buffer);
            } else {
                drop(workspace);
            }
        }
        self.shared_buffers.clear();
        if let Some(registration) = self.validation_registration.as_mut() {
            let owner = registration.owner();
            if let Err(error) = self.ep.unregister_device_validation_owner(registration) {
                eprintln!(
                    "[onnx-runtime-session] executor drop could not unregister validation owner \
                     {}: {error}",
                    owner.get()
                );
            } else {
                self.validation_registration = None;
            }
        }
    }
}

mod state;
use state::*;
pub(crate) use state::{ChildExecutor, ChildExecutorStats, Executor};
mod kernel_cache;
use build::*;
use capture::*;
pub use capture::{
    CaptureDecline, CaptureDeclineReport, CapturePathKind, ControlFlowStats,
    DeviceAllocationCounts, DeviceGraphCaptureResult, ExecutionProviderDecline,
    ExecutionProviderFallbackReport, SeamReason,
};
pub use kernel_cache::CacheStats;
pub(crate) use kernel_cache::KernelCache;
use kernel_cache::*;
#[cfg(test)]
pub(crate) use kernel_cache::{PREBIND_FALLBACK_TEST_HITS, PREBIND_FAST_PATH_TEST_HITS};
mod dynamic_shapes;
mod geometry;
use dynamic_shapes::*;
use geometry::*;
mod bindings;
#[cfg(test)]
use bindings::{AxisBound, PlannedInputShape};
mod build;
mod capture;
mod control_flow;
mod dispatch;
mod platform;
mod prefetch;
mod run;
mod sequence_ops;
pub(crate) use platform::auto_detect_cpu_ep;
pub use prefetch::{PrefetchStep, drive_double_buffer, plan_double_buffer};

#[cfg(test)]
mod tests;

//! Native CPU execution for schema-resolved ONNX `Einsum`.
//!
//! The equation is parsed exactly once, by [`EinsumShapePlan`], when the
//! shape-specialized kernel is built. Execution consumes only the plan's
//! structural classification and axis maps.

use std::alloc::Layout as AllocationLayout;
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};

use onnx_runtime_ep_api::{
    EpError, Kernel, KernelFactory, Result, TensorMetadata, TensorMut, TensorView, ViewOutput,
    WorkspaceLayout, WorkspaceRequirement, WorkspaceView,
};
use onnx_runtime_ir::{
    DataType, EinsumAxis, EinsumBinaryContractionPlan, EinsumContractionPlan,
    EinsumContractionTreePlan, EinsumContractionTreeStep, EinsumDimension, EinsumGenericNativePlan,
    EinsumInput, EinsumOperandPlan, EinsumPermutationPlan, EinsumPlan, EinsumPlannerQuality,
    EinsumPlanningClassification, EinsumSchema, EinsumShapePlan,
    EinsumSupportedContractionTreeCandidate, EinsumUnaryReductionPlan, EinsumValueId, Node, Shape,
    compute_contiguous_strides,
};
use rayon::prelude::*;

use super::{check_arity, matmul::MatMulKernel, to_dense_bytes, write_dense_bytes};
use crate::CpuCacheRetention;
use crate::dtype::{
    ComputeDomain, NumericElem, to_dense_f32_widen, write_dense, write_dense_f32_narrow,
};
use crate::strided::{elem_offset, next_index};

/// Diagnostic execution switch used by `benches/einsum.rs` and conformance.
///
/// `generic-native` forces the universal semantic index program and
/// `optimized` permits compatible view/reduction/MatMul/tree routes. `oracle`
/// retains the pre-existing high-precision diagnostic. The value is read only
/// when a kernel is constructed.
pub const EINSUM_MODE_ENV: &str = "NXRT_CPU_EINSUM_MODE";

/// CPU Einsum route policy captured immutably by a compiled kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EinsumExecutionMode {
    /// Preserve compatible native fast paths and use GenericNative as fallback.
    Optimized,
    /// Force the universal index program for every arithmetic expression.
    GenericNative,
    /// High-precision diagnostic retained for benchmark validation.
    Oracle,
}

#[cfg(test)]
type ExecutionMode = EinsumExecutionMode;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EinsumRoute {
    ViewCopy,
    Reduction,
    GenericNative,
    OptimizedDp,
    OptimizedHeuristic,
    Oracle,
    MatMulDirect,
    MatMulMaterialized,
    MatMulScalar,
}

impl EinsumRoute {
    fn label(self) -> &'static str {
        match self {
            Self::ViewCopy => "view-copy",
            Self::Reduction => "reduction-native",
            Self::GenericNative => "generic-native",
            Self::OptimizedDp => "optimized-dp",
            Self::OptimizedHeuristic => "optimized-heuristic",
            Self::Oracle => "oracle-diagnostic",
            Self::MatMulDirect => "matmul-direct",
            Self::MatMulMaterialized => "matmul-materialized",
            Self::MatMulScalar => "matmul-scalar",
        }
    }

    const fn telemetry_index(self) -> usize {
        match self {
            Self::ViewCopy => 0,
            Self::Reduction => 1,
            Self::GenericNative => 2,
            Self::OptimizedDp => 3,
            Self::OptimizedHeuristic => 4,
            Self::MatMulDirect => 5,
            Self::MatMulMaterialized => 6,
            Self::MatMulScalar => 7,
            Self::Oracle => 8,
        }
    }
}

const EINSUM_TELEMETRY_ROUTES: usize = 9;
static EINSUM_ROUTE_COUNTS: [AtomicUsize; EINSUM_TELEMETRY_ROUTES] =
    [const { AtomicUsize::new(0) }; EINSUM_TELEMETRY_ROUTES];

/// Reset process-local CPU Einsum route counters.
#[doc(hidden)]
pub fn reset_route_telemetry() {
    for count in &EINSUM_ROUTE_COUNTS {
        count.store(0, Ordering::Relaxed);
    }
}

/// Return the number of successful dispatches for a telemetry route index.
///
/// Indices are `0=view`, `1=reduction`, `2=generic-native`,
/// `3=optimized-dp`, `4=optimized-heuristic`, `5=matmul-direct`,
/// `6=matmul-materialized`, `7=matmul-scalar`, and `8=oracle`.
#[doc(hidden)]
pub fn route_telemetry_count(route: usize) -> usize {
    EINSUM_ROUTE_COUNTS
        .get(route)
        .map_or(0, |count| count.load(Ordering::Relaxed))
}

// Algorithm-selection ceiling, not an allocation grant or a retention budget.
// Larger trees use the lower-memory GenericNative route.
const EINSUM_TREE_WORKSPACE_LIMIT: usize = 32 * 1024 * 1024;

const CONCURRENCY_VIEW: usize = 0;
const CONCURRENCY_REDUCTION: usize = 1;
const CONCURRENCY_MATERIALIZED_GEMM: usize = 2;
const CONCURRENCY_GENERIC: usize = 3;

#[cfg(any(test, feature = "einsum_concurrency_probe"))]
mod concurrency_probe {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Condvar, Mutex, OnceLock};
    use std::time::{Duration, Instant};

    const ROUTES: usize = 4;
    const RENDEZVOUS_TIMEOUT: Duration = Duration::from_secs(10);
    static ARMED_GENERATION: AtomicU64 = AtomicU64::new(0);
    static STATE: OnceLock<(Mutex<State>, Condvar)> = OnceLock::new();

    struct State {
        generation: u64,
        route: Option<usize>,
        arrivals: usize,
        released: bool,
        completed: bool,
        active: [usize; ROUTES],
        max: [usize; ROUTES],
    }

    fn shared() -> &'static (Mutex<State>, Condvar) {
        STATE.get_or_init(|| {
            (
                Mutex::new(State {
                    generation: 0,
                    route: None,
                    arrivals: 0,
                    released: true,
                    completed: false,
                    active: [0; ROUTES],
                    max: [0; ROUTES],
                }),
                Condvar::new(),
            )
        })
    }

    fn lock_state() -> std::sync::MutexGuard<'static, State> {
        shared()
            .0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub(super) struct Entry {
        generation: u64,
        route: usize,
    }

    impl Entry {
        pub(super) fn enter(route: usize) -> Option<Self> {
            let generation = ARMED_GENERATION.load(Ordering::Acquire);
            if generation == 0 {
                return None;
            }

            let (_, ready) = shared();
            let mut state = lock_state();
            if state.generation != generation || state.route != Some(route) || state.released {
                return None;
            }
            state.active[route] += 1;
            state.max[route] = state.max[route].max(state.active[route]);
            state.arrivals += 1;
            ready.notify_all();
            if state.arrivals >= 2 {
                state.completed = true;
                state.released = true;
                ready.notify_all();
            } else {
                let deadline = Instant::now() + RENDEZVOUS_TIMEOUT;
                while state.generation == generation && !state.released {
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if remaining.is_zero() {
                        state.released = true;
                        ready.notify_all();
                        break;
                    }
                    let (next, timeout) = ready
                        .wait_timeout(state, remaining)
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    state = next;
                    if timeout.timed_out() && state.generation == generation && !state.released {
                        state.released = true;
                        ready.notify_all();
                    }
                }
            }
            Some(Self { generation, route })
        }
    }

    impl Drop for Entry {
        fn drop(&mut self) {
            let (_, ready) = shared();
            let mut state = lock_state();
            if state.generation == self.generation && state.active[self.route] != 0 {
                state.active[self.route] -= 1;
                ready.notify_all();
            }
        }
    }

    #[doc(hidden)]
    pub fn begin(route: usize) -> u64 {
        ARMED_GENERATION.store(0, Ordering::Release);
        let (_, ready) = shared();
        let mut state = lock_state();

        state.released = true;
        ready.notify_all();
        state.generation = state.generation.wrapping_add(1).max(1);
        state.route = (route < ROUTES).then_some(route);
        state.arrivals = 0;
        state.released = route >= ROUTES;
        state.completed = false;
        state.active = [0; ROUTES];
        state.max = [0; ROUTES];

        let generation = state.generation;
        if state.route.is_some() {
            ARMED_GENERATION.store(generation, Ordering::Release);
            generation
        } else {
            0
        }
    }

    #[doc(hidden)]
    pub fn finish(generation: u64) -> [usize; ROUTES] {
        if generation == 0 {
            return [0; ROUTES];
        }
        let _ =
            ARMED_GENERATION.compare_exchange(generation, 0, Ordering::AcqRel, Ordering::Acquire);
        let (_, ready) = shared();
        let mut state = lock_state();
        if state.generation != generation {
            return [0; ROUTES];
        }

        let mut result = state.max;
        if let Some(route) = state.route
            && !state.completed
        {
            result[route] = 0;
        }

        state.released = true;
        ready.notify_all();
        state.generation = state.generation.wrapping_add(1).max(1);
        state.route = None;
        state.arrivals = 0;
        state.completed = false;
        state.active = [0; ROUTES];
        state.max = [0; ROUTES];
        result
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::panic::{AssertUnwindSafe, catch_unwind};
        use std::sync::mpsc;

        static TEST_LOCK: Mutex<()> = Mutex::new(());

        fn lock_tests() -> std::sync::MutexGuard<'static, ()> {
            TEST_LOCK
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        }

        struct ScopedProbe {
            generation: u64,
        }

        impl ScopedProbe {
            fn new(route: usize) -> Self {
                Self {
                    generation: begin(route),
                }
            }

            fn finish(mut self) -> [usize; ROUTES] {
                let generation = std::mem::take(&mut self.generation);
                finish(generation)
            }
        }

        impl Drop for ScopedProbe {
            fn drop(&mut self) {
                let generation = std::mem::take(&mut self.generation);
                let _ = finish(generation);
            }
        }

        fn wait_for_arrival(generation: u64) {
            let (_, ready) = shared();
            let mut state = lock_state();
            while state.generation == generation && state.arrivals == 0 {
                state = ready
                    .wait(state)
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
            }
        }

        #[test]
        fn unwind_teardown_wakes_waiter_and_next_probe_starts_cleanly() {
            let _serial = lock_tests();
            let (done_tx, done_rx) = mpsc::channel();
            let unwind = catch_unwind(AssertUnwindSafe(|| {
                let probe = ScopedProbe::new(super::super::CONCURRENCY_REDUCTION);
                let generation = probe.generation;
                std::thread::spawn(move || {
                    let _entry = Entry::enter(super::super::CONCURRENCY_REDUCTION);
                    done_tx.send(()).unwrap();
                });
                wait_for_arrival(generation);
                panic!("deliberate armed-probe unwind");
            }));
            assert!(unwind.is_err());
            done_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("teardown did not wake the blocked probe participant");

            let probe = ScopedProbe::new(super::super::CONCURRENCY_REDUCTION);
            std::thread::scope(|scope| {
                for _ in 0..2 {
                    scope.spawn(|| {
                        let _entry = Entry::enter(super::super::CONCURRENCY_REDUCTION);
                    });
                }
            });
            assert!(probe.finish()[super::super::CONCURRENCY_REDUCTION] >= 2);
        }

        #[test]
        fn finish_is_idempotent_and_recovers_from_state_lock_poisoning() {
            let _serial = lock_tests();
            let _ = catch_unwind(|| {
                let _state = shared().0.lock().unwrap();
                panic!("deliberately poison concurrency probe state");
            });

            let probe = ScopedProbe::new(super::super::CONCURRENCY_MATERIALIZED_GEMM);
            let generation = probe.generation;
            std::thread::scope(|scope| {
                for _ in 0..2 {
                    scope.spawn(|| {
                        let _entry = Entry::enter(super::super::CONCURRENCY_MATERIALIZED_GEMM);
                    });
                }
            });
            assert!(probe.finish()[super::super::CONCURRENCY_MATERIALIZED_GEMM] >= 2);
            assert_eq!(finish(generation), [0; ROUTES]);
            assert_eq!(ARMED_GENERATION.load(Ordering::Acquire), 0);
            let state = lock_state();
            assert_eq!(state.route, None);
            assert_eq!(state.active, [0; ROUTES]);
            assert_eq!(state.max, [0; ROUTES]);
        }

        #[test]
        fn stale_finish_cannot_disarm_a_new_probe() {
            let _serial = lock_tests();
            assert_eq!(begin(ROUTES), 0, "invalid route must remain disarmed");

            let stale = begin(super::super::CONCURRENCY_REDUCTION);
            let current = begin(super::super::CONCURRENCY_MATERIALIZED_GEMM);
            assert_ne!(stale, current);
            assert_eq!(finish(stale), [0; ROUTES]);
            assert_eq!(ARMED_GENERATION.load(Ordering::Acquire), current);

            let probe = ScopedProbe {
                generation: current,
            };
            std::thread::scope(|scope| {
                for _ in 0..2 {
                    scope.spawn(|| {
                        let _entry = Entry::enter(super::super::CONCURRENCY_MATERIALIZED_GEMM);
                    });
                }
            });
            assert!(probe.finish()[super::super::CONCURRENCY_MATERIALIZED_GEMM] >= 2);
        }

        #[test]
        fn worker_panic_unwinds_owner_and_leaves_probe_disarmed() {
            let _serial = lock_tests();
            let unwind = catch_unwind(AssertUnwindSafe(|| {
                let _probe = ScopedProbe::new(super::super::CONCURRENCY_REDUCTION);
                std::thread::scope(|scope| {
                    let panicking = scope.spawn(|| {
                        let _entry = Entry::enter(super::super::CONCURRENCY_REDUCTION);
                        panic!("deliberate probe worker panic");
                    });
                    let peer = scope.spawn(|| {
                        let _entry = Entry::enter(super::super::CONCURRENCY_REDUCTION);
                    });
                    peer.join().unwrap();
                    panicking.join().unwrap();
                });
            }));
            assert!(unwind.is_err());
            assert_eq!(ARMED_GENERATION.load(Ordering::Acquire), 0);
            let state = lock_state();
            assert_eq!(state.route, None);
            assert_eq!(state.active, [0; ROUTES]);
            assert_eq!(state.max, [0; ROUTES]);
        }
    }
}

#[cfg(any(test, feature = "einsum_concurrency_probe"))]
use concurrency_probe::Entry as ConcurrencyProbeGuard;

#[cfg(not(any(test, feature = "einsum_concurrency_probe")))]
struct ConcurrencyProbeGuard;

#[cfg(not(any(test, feature = "einsum_concurrency_probe")))]
impl ConcurrencyProbeGuard {
    #[inline(always)]
    fn enter(_route: usize) -> Option<Self> {
        None
    }
}

#[cfg(any(test, feature = "einsum_concurrency_probe"))]
#[doc(hidden)]
pub use concurrency_probe::{begin as begin_concurrency_probe, finish as finish_concurrency_probe};

/// Shape-specialized CPU Einsum kernel.
pub struct EinsumKernel {
    plan: EinsumShapePlan,
    matmul: MatMulKernel,
    mode: EinsumExecutionMode,
    output_len: usize,
    // Concrete IR choices are compiled facts, indexed by accumulator width
    // (1/2/4/8 bytes). No mutable pool, dtype cache or per-run planning lock.
    tree_choices: [std::result::Result<Option<TreeChoice>, String>; 4],
    flops: Option<u64>,
    last_workspace_bytes: AtomicUsize,
    #[cfg(test)]
    last_route: std::sync::atomic::AtomicU8,
}

/// Factory for [`EinsumKernel`].
pub struct EinsumFactory {
    cache_retention: CpuCacheRetention,
    mode: Option<EinsumExecutionMode>,
}

impl EinsumFactory {
    pub fn new(cache_retention: CpuCacheRetention) -> Self {
        Self {
            cache_retention,
            mode: None,
        }
    }

    /// Construct a factory with an explicit immutable execution policy.
    ///
    /// This avoids process-global environment mutation in conformance and
    /// benchmark adapters while keeping the ordinary provider configuration
    /// compatible with [`EINSUM_MODE_ENV`].
    pub fn with_execution_mode(
        cache_retention: CpuCacheRetention,
        mode: EinsumExecutionMode,
    ) -> Self {
        Self {
            cache_retention,
            mode: Some(mode),
        }
    }
}

impl Default for EinsumFactory {
    fn default() -> Self {
        Self::new(CpuCacheRetention::default())
    }
}

impl KernelFactory for EinsumFactory {
    fn create(&self, node: &Node, input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let equation = equation(node)?;
        let input_shape_refs: Vec<_> = input_shapes.iter().map(Vec::as_slice).collect();
        let schema = EinsumSchema::resolve(node.local_opset().unwrap_or(12))
            .map_err(|error| EpError::KernelFailed(format!("Einsum: {error}")))?;
        let plan = EinsumShapePlan::build_for_schema(equation, &input_shape_refs, schema).map_err(
            |error| {
                EpError::KernelFailed(format!(
                    "Einsum: canonical planning failed for `{equation}`: {error}"
                ))
            },
        )?;
        let mode = self.mode.map_or_else(execution_mode, Ok)?;
        let flops = match plan.planning_classification() {
            EinsumPlanningClassification::Gemm(gemm) => gemm_flops(gemm),
            _ => None,
        };
        Ok(Box::new(EinsumKernel::new(
            plan,
            MatMulKernel::with_cache_retention(self.cache_retention.is_admitted()),
            mode,
            &input_shape_refs,
            flops,
        )?))
    }
}

fn equation(node: &Node) -> Result<&str> {
    let attribute = node.attr("equation").ok_or_else(|| {
        EpError::KernelFailed(
            "Einsum: missing required string attribute `equation`. HOW: export the node \
             with its ONNX equation attribute."
                .into(),
        )
    })?;
    attribute.as_str().ok_or_else(|| {
        EpError::KernelFailed(
            "Einsum: attribute `equation` must be valid UTF-8 STRING data. HOW: encode an ASCII \
             einsum equation such as `ik,kj->ij`."
                .into(),
        )
    })
}

fn execution_mode() -> Result<EinsumExecutionMode> {
    match std::env::var(EINSUM_MODE_ENV) {
        Ok(value) if value.eq_ignore_ascii_case("oracle") => Ok(EinsumExecutionMode::Oracle),
        Ok(value)
            if value.eq_ignore_ascii_case("generic")
                || value.eq_ignore_ascii_case("generic-native") =>
        {
            Ok(EinsumExecutionMode::GenericNative)
        }
        Ok(value) if value.eq_ignore_ascii_case("optimized") || value.trim().is_empty() => {
            Ok(EinsumExecutionMode::Optimized)
        }
        Ok(value) => Err(EpError::KernelFailed(format!(
            "Einsum: {EINSUM_MODE_ENV}={value:?} is invalid. HOW: use `optimized` (default), \
             `generic-native`, or `oracle` for a high-precision correctness diagnostic."
        ))),
        Err(std::env::VarError::NotPresent) => Ok(EinsumExecutionMode::Optimized),
        Err(std::env::VarError::NotUnicode(_)) => Err(EpError::KernelFailed(format!(
            "Einsum: {EINSUM_MODE_ENV} is not valid UTF-8. HOW: unset it or use `optimized`, \
             `generic-native`, or `oracle`."
        ))),
    }
}

/// Claim-time capability check shared with [`crate::CpuExecutionProvider`].
///
/// Returning the planner's structured rejection before ORT compiles the node
/// lets another CPU provider take legal but not-yet-native general
/// contractions instead of failing session creation after assignment.
pub fn unsupported_reason_for_opset(
    node: &Node,
    opset: u64,
    shapes: &[Shape],
    input_dtypes: &[DataType],
) -> Option<String> {
    let equation = match equation(node) {
        Ok(equation) => equation,
        Err(error) => return Some(error.to_string()),
    };
    if shapes.len() != input_dtypes.len() {
        return Some(format!(
            "Einsum `{equation}` received {} input shape(s) but {} input dtype(s); \
             HOW: finish graph type/shape inference before EP placement",
            shapes.len(),
            input_dtypes.len()
        ));
    }
    let inputs: Vec<_> = shapes
        .iter()
        .zip(input_dtypes)
        .map(|(shape, &dtype)| EinsumInput::new(dtype, shape))
        .collect();
    match EinsumPlan::build_for_opset(equation, &inputs, opset) {
        Ok(_) => None,
        Err(error) => Some(format!(
            "Einsum canonical planning rejected `{equation}`: {error}"
        )),
    }
}

impl Kernel for EinsumKernel {
    fn set_constant_inputs(&mut self, constant_inputs: &[bool]) {
        self.matmul.set_constant_inputs(constant_inputs);
    }

    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.execute_with_route(inputs, outputs, None).map(|_| ())
    }

    fn workspace_requirement(&self, inputs: &[TensorMetadata<'_>]) -> Result<WorkspaceRequirement> {
        let dtype = self.validate_workspace_metadata(inputs)?;
        let bytes_per_element = match self.mode {
            EinsumExecutionMode::GenericNative => accumulator_element_size(dtype),
            EinsumExecutionMode::Oracle
                if matches!(dtype, DataType::Float16 | DataType::Float32) =>
            {
                4
            }
            EinsumExecutionMode::Oracle => accumulator_element_size(dtype),
            EinsumExecutionMode::Optimized => match self.plan.planning_classification() {
                EinsumPlanningClassification::ViewOnlyPermutation(_)
                | EinsumPlanningClassification::DiagonalView(_) => {
                    return Ok(WorkspaceRequirement::NONE);
                }
                EinsumPlanningClassification::Gemm(_)
                    if matches!(
                        dtype,
                        DataType::Float16 | DataType::BFloat16 | DataType::Float32
                    ) =>
                {
                    4
                }
                EinsumPlanningClassification::Gemm(_) => return Ok(WorkspaceRequirement::NONE),
                EinsumPlanningClassification::ContractionTree(_)
                    if self.tree_choice(dtype)?.is_some() =>
                {
                    return Ok(WorkspaceRequirement::NONE);
                }
                _ => accumulator_element_size(dtype),
            },
        };
        let bytes = self
            .output_len
            .checked_mul(bytes_per_element)
            .ok_or_else(|| geometry_overflow(self.plan.equation(), "accumulator workspace"))?;
        let layout = AllocationLayout::from_size_align(bytes, bytes_per_element).map_err(|_| {
            geometry_overflow(self.plan.equation(), "addressable accumulator workspace")
        })?;
        if bytes == 0 {
            return Ok(WorkspaceRequirement::NONE);
        }
        Ok(WorkspaceRequirement {
            bytes: layout.size() as u64,
            alignment: layout.align(),
            ..WorkspaceRequirement::NONE
        })
    }

    fn execute_with_workspace(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        self.execute_with_route(inputs, outputs, workspace)
            .map(|_| ())
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }

    fn may_produce_views(&self) -> bool {
        self.mode == EinsumExecutionMode::Optimized
            && matches!(
                self.plan.planning_classification(),
                EinsumPlanningClassification::ViewOnlyPermutation(_)
                    | EinsumPlanningClassification::DiagonalView(_)
            )
    }

    fn view_outputs(
        &self,
        inputs: &[TensorView],
        output_shapes: &[Vec<usize>],
        num_outputs: usize,
    ) -> Option<Vec<ViewOutput>> {
        if num_outputs != 1 || output_shapes.len() != 1 {
            return None;
        }
        if self.mode != EinsumExecutionMode::Optimized {
            return None;
        }
        let dtype = inputs.first()?.dtype;
        if !self.plan.schema().supports_dtype(dtype)
            || inputs.iter().any(|input| input.dtype != dtype)
        {
            return None;
        }
        let permutation = match self.plan.planning_classification() {
            EinsumPlanningClassification::ViewOnlyPermutation(permutation)
            | EinsumPlanningClassification::DiagonalView(permutation) => permutation,
            _ => return None,
        };
        let input = inputs.get(permutation.input())?;
        if input.dtype.byte_size() == 0 {
            return None;
        }
        let shapes: Vec<_> = inputs.iter().map(|input| input.shape).collect();
        let resolved = self.plan.resolve_concrete_output_shape(&shapes).ok()?;
        if resolved != output_shapes[0] {
            return None;
        }
        let layout = permutation_layout(
            input,
            &self.plan.operands()[permutation.input()],
            permutation,
        )
        .ok()?;
        Some(vec![ViewOutput {
            input_index: permutation.input(),
            shape: layout.shape,
            strides: layout.strides,
            byte_offset: input.byte_offset,
        }])
    }

    fn estimated_flops(&self) -> Option<u64> {
        self.flops
    }
}

impl EinsumKernel {
    fn new(
        plan: EinsumShapePlan,
        matmul: MatMulKernel,
        mode: EinsumExecutionMode,
        input_shapes: &[&[usize]],
        flops: Option<u64>,
    ) -> Result<Self> {
        let output_shape = plan
            .resolve_concrete_output_shape(input_shapes)
            .map_err(|error| {
                EpError::KernelFailed(format!("Einsum workspace output planning failed: {error}"))
            })?;
        let output_len = checked_numel("compiled output", &output_shape)?;
        let tree_choices = std::array::from_fn(|width| {
            if mode == EinsumExecutionMode::Optimized
                && let EinsumPlanningClassification::ContractionTree(tree) =
                    plan.planning_classification()
            {
                select_tree_choice(&plan, tree, input_shapes, &output_shape, 1 << width)
                    .map_err(|error| error.to_string())
            } else {
                Ok(None)
            }
        });
        Ok(Self {
            plan,
            matmul,
            mode,
            output_len,
            tree_choices,
            flops,
            last_workspace_bytes: AtomicUsize::new(0),
            #[cfg(test)]
            last_route: std::sync::atomic::AtomicU8::new(0),
        })
    }

    fn validate_workspace_metadata(&self, inputs: &[TensorMetadata<'_>]) -> Result<DataType> {
        let dtype = inputs.first().map(|input| input.dtype).ok_or_else(|| EpError::KernelFailed("Einsum workspace needs at least one present numeric input; provide the compiled operand metadata".into()))?;
        if inputs.len() != self.plan.operands().len() || !self.plan.schema().supports_dtype(dtype) {
            return Err(EpError::KernelFailed(format!(
                "Einsum `{}` workspace received {} operands with dtype {dtype:?}; supply {} homogeneous operands supported by {}",
                self.plan.equation(),
                inputs.len(),
                self.plan.operands().len(),
                self.plan.schema()
            )));
        }
        for (index, (input, operand)) in inputs.iter().zip(self.plan.operands()).enumerate() {
            if !input.present || input.dtype != dtype || input.shape.len() != operand.shape().len() || input.shape.iter().zip(operand.shape()).any(|(actual, planned)| !matches!(planned, EinsumDimension::Static(expected) if expected == actual)) {
                return Err(EpError::KernelFailed(format!("Einsum `{}` workspace input {index} has present={}, dtype={:?}, shape={:?}; provide the compiled homogeneous {dtype:?} operand shape {:?}", self.plan.equation(), input.present, input.dtype, input.shape, operand.shape())));
            }
        }
        Ok(dtype)
    }

    fn tree_choice(&self, dtype: DataType) -> Result<Option<TreeChoice>> {
        let width = accumulator_element_size(dtype);
        if !width.is_power_of_two() || width > 8 {
            return Err(EpError::KernelFailed(format!(
                "Einsum `{}` cannot select a contraction tree for accumulator width {width}; use a supported numeric dtype",
                self.plan.equation()
            )));
        }
        self.tree_choices[width.trailing_zeros() as usize]
            .as_ref()
            .copied()
            .map_err(|error| {
                EpError::KernelFailed(format!(
                    "Einsum `{}` compiled tree selection failed for {dtype:?}: {error}",
                    self.plan.equation()
                ))
            })
    }

    fn execute_with_route(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<EinsumRoute> {
        self.last_workspace_bytes.store(0, Ordering::Relaxed);
        self.validate_execution(inputs, outputs)?;
        if self.mode == EinsumExecutionMode::GenericNative {
            let _probe = ConcurrencyProbeGuard::enter(CONCURRENCY_GENERIC);
            self.execute_generic_native(inputs, outputs, workspace)?;
            return Ok(self.finish_route(EinsumRoute::GenericNative));
        }
        if self.mode == EinsumExecutionMode::Oracle {
            let _probe = ConcurrencyProbeGuard::enter(CONCURRENCY_GENERIC);
            if matches!(inputs[0].dtype, DataType::Float32 | DataType::Float16) {
                self.execute_oracle(inputs, outputs, workspace)?;
            } else {
                self.execute_generic_native(inputs, outputs, workspace)?;
            }
            return Ok(self.finish_route(EinsumRoute::Oracle));
        }

        match self.plan.planning_classification() {
            EinsumPlanningClassification::ViewOnlyPermutation(permutation)
            | EinsumPlanningClassification::DiagonalView(permutation) => {
                let _probe = ConcurrencyProbeGuard::enter(CONCURRENCY_VIEW);
                self.execute_view_copy(inputs, outputs, permutation)?;
                Ok(self.finish_route(EinsumRoute::ViewCopy))
            }
            EinsumPlanningClassification::ReductionOrElementwise(_) => {
                let _probe = ConcurrencyProbeGuard::enter(CONCURRENCY_REDUCTION);
                self.execute_generic_native(inputs, outputs, workspace)?;
                Ok(self.finish_route(EinsumRoute::Reduction))
            }
            EinsumPlanningClassification::Gemm(gemm)
                if matches!(
                    inputs[0].dtype,
                    DataType::Float16 | DataType::BFloat16 | DataType::Float32
                ) =>
            {
                let route = self.execute_gemm(inputs, outputs, workspace, gemm)?;
                Ok(self.finish_route(route))
            }
            EinsumPlanningClassification::Gemm(gemm) => {
                let _probe = ConcurrencyProbeGuard::enter(CONCURRENCY_GENERIC);
                crate::dispatch_arith!(inputs[0].dtype, "Einsum", T => {
                    self.execute_scalar_gemm_typed::<T>(inputs, outputs, gemm)
                })?;
                Ok(self.finish_route(EinsumRoute::MatMulScalar))
            }
            EinsumPlanningClassification::ContractionTree(tree) => {
                let _probe = ConcurrencyProbeGuard::enter(CONCURRENCY_GENERIC);
                let route =
                    self.execute_contraction_tree_or_generic(inputs, outputs, workspace, tree)?;
                Ok(self.finish_route(route))
            }
            _ => {
                let _probe = ConcurrencyProbeGuard::enter(CONCURRENCY_GENERIC);
                self.execute_generic_native(inputs, outputs, workspace)?;
                Ok(self.finish_route(EinsumRoute::GenericNative))
            }
        }
    }

    fn finish_route(&self, route: EinsumRoute) -> EinsumRoute {
        EINSUM_ROUTE_COUNTS[route.telemetry_index()].fetch_add(1, Ordering::Relaxed);
        self.record_route(route);
        route
    }

    fn validate_execution(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity(
            "Einsum",
            inputs,
            outputs,
            self.plan.operands().len(),
            self.plan.operands().len(),
            1,
        )?;
        for (index, input) in inputs.iter().enumerate() {
            input.validate()?;
            if !input.device.is_host_accessible() {
                return Err(EpError::KernelFailed(format!(
                    "Einsum `{}` input #{index} is on {:?}; the native CPU kernel requires \
                     host-accessible inputs. HOW: place or copy the tensor to CPU before this node.",
                    self.plan.equation(),
                    input.device
                )));
            }
            if !self.plan.schema().supports_dtype(input.dtype) {
                return Err(EpError::KernelFailed(format!(
                    "Einsum `{}` input #{index} has dtype {:?}, which is not admitted by {}. HOW: \
                     use a homogeneous numeric dtype supported by that schema{}.",
                    self.plan.equation(),
                    input.dtype,
                    self.plan.schema(),
                    if input.dtype == DataType::BFloat16 {
                        " or import ai.onnx opset 28+ for BFloat16"
                    } else {
                        ""
                    }
                )));
            }
            if input.dtype != inputs[0].dtype {
                return Err(EpError::KernelFailed(format!(
                    "Einsum `{}` input #{index} has dtype {:?}, but input #0 has {:?}; ONNX \
                     Einsum operands must share one dtype",
                    self.plan.equation(),
                    input.dtype,
                    inputs[0].dtype
                )));
            }
        }
        outputs[0].validate()?;
        if !outputs[0].device.is_host_accessible() {
            return Err(EpError::KernelFailed(format!(
                "Einsum `{}` output is on {:?}; the native CPU kernel requires a host-accessible \
                 output",
                self.plan.equation(),
                outputs[0].device
            )));
        }
        if outputs[0].dtype != inputs[0].dtype {
            return Err(EpError::KernelFailed(format!(
                "Einsum `{}` output dtype {:?} must match input dtype {:?}",
                self.plan.equation(),
                outputs[0].dtype,
                inputs[0].dtype
            )));
        }
        let shapes: Vec<_> = inputs.iter().map(|input| input.shape).collect();
        let expected = self
            .plan
            .resolve_concrete_output_shape(&shapes)
            .map_err(|error| {
                EpError::KernelFailed(format!(
                    "Einsum `{}` runtime shape validation failed: {error}",
                    self.plan.equation()
                ))
            })?;
        if outputs[0].shape != expected {
            return Err(EpError::KernelFailed(format!(
                "Einsum `{}` output shape {:?} does not match canonical resolved shape \
                 {expected:?}",
                self.plan.equation(),
                outputs[0].shape
            )));
        }
        Ok(())
    }

    fn execute_view_copy(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        permutation: &EinsumPermutationPlan,
    ) -> Result<()> {
        let input = &inputs[permutation.input()];
        let layout = permutation_layout(
            input,
            &self.plan.operands()[permutation.input()],
            permutation,
        )?;
        let view = TensorView::new(
            input.data,
            input.dtype,
            &layout.shape,
            &layout.strides,
            input.device,
        )
        .with_byte_offset(input.byte_offset);
        let dense = to_dense_bytes(&view)?;
        self.last_workspace_bytes
            .store(dense.len(), Ordering::Relaxed);
        write_dense_bytes(&mut outputs[0], &dense)
    }

    fn execute_oracle(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        let mut iteration_axes = self.plan.output_axes().to_vec();
        iteration_axes.extend_from_slice(self.plan.reduction_axes());
        let output_rank = self.plan.output_axes().len();
        let mappings: Vec<Vec<usize>> = self
            .plan
            .operands()
            .iter()
            .map(|operand| {
                operand
                    .unique_axes()
                    .iter()
                    .map(|operand_axis| {
                        iteration_axes
                            .iter()
                            .position(|axis| axis == &operand_axis.axis())
                            .expect("validated supported plan maps every operand axis")
                    })
                    .collect()
            })
            .collect();
        self.execute_generic(
            inputs,
            outputs,
            workspace,
            &iteration_axes,
            output_rank,
            &mappings,
            true,
        )
    }

    fn execute_generic_native(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        crate::dispatch_arith!(inputs[0].dtype, "Einsum", T => {
            self.execute_generic_native_typed::<T>(
                inputs,
                outputs,
                workspace,
                self.plan.generic_native(),
            )
        })
    }

    fn execute_generic_native_typed<T: EinsumElement>(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
        generic: &EinsumGenericNativePlan,
    ) -> Result<()> {
        let program = generic.index_program();
        let iteration_shape = axes_shape(&self.plan, program.iteration_axes())?;
        let output_rank = program.output_rank();
        let output_shape = iteration_shape.get(..output_rank).ok_or_else(|| {
            EpError::KernelFailed(format!(
                "Einsum `{}` GenericNative output rank {output_rank} exceeds iteration rank {}",
                self.plan.equation(),
                iteration_shape.len()
            ))
        })?;
        let reduction_shape = &iteration_shape[output_rank..];
        let output_len = checked_numel("GenericNative output", output_shape)?;
        let reduction_len = checked_numel("GenericNative reduction", reduction_shape)?;
        if output_len == 0 {
            return Ok(());
        }
        let prepared =
            prepare_generic_inputs::<T>(self.plan.equation(), inputs, generic, &iteration_shape)?;
        let work = (output_len as u128)
            .checked_mul(reduction_len.max(1) as u128)
            .and_then(|value| value.checked_mul(prepared.len() as u128))
            .ok_or_else(|| {
                geometry_overflow(self.plan.equation(), "GenericNative work accounting")
            })?;
        let tile = generic_output_tile(reduction_len, prepared.len(), output_len);

        // SAFETY: CPU dispatch lends exclusive live workspace for this call;
        // the closure writes every element before exposing initialized values.
        unsafe {
            with_accumulator::<<T as NumericElem>::Acc, _>(workspace, output_len, |accumulators| {
                let evaluate =
                    |first_tile: usize, output: &mut [MaybeUninit<<T as NumericElem>::Acc>]| {
                        let first_output = first_tile * tile;
                        let mut coordinates = vec![0usize; iteration_shape.len()];
                        for (local, destination) in output.iter_mut().enumerate() {
                            let output_linear = first_output + local;
                            decode_row_major(
                                output_linear,
                                output_shape,
                                &mut coordinates[..output_rank],
                            );
                            let mut sum = <T as NumericElem>::Acc::default();
                            for reduction_linear in 0..reduction_len {
                                decode_row_major(
                                    reduction_linear,
                                    reduction_shape,
                                    &mut coordinates[output_rank..],
                                );
                                let mut product = T::one();
                                for input in &prepared {
                                    product = product.c_mul(input.read(&coordinates));
                                }
                                sum = if reduction_linear == 0 {
                                    product
                                } else {
                                    sum.c_add(product)
                                };
                            }
                            destination.write(sum);
                        }
                    };

                const PARALLEL_GENERIC_WORK_ITEMS: u128 = 128 * 1024;
                if output_len > 1 && work >= PARALLEL_GENERIC_WORK_ITEMS {
                    crate::task_runtime::chunk_runs_mut(accumulators, tile, 1, evaluate);
                } else {
                    evaluate(0, accumulators);
                }
                self.last_workspace_bytes.store(
                    output_len.saturating_mul(std::mem::size_of::<T::Acc>()),
                    Ordering::Relaxed,
                );
                // Every output position above writes a final sum, including zero reduction.
                T::write_accumulators(&mut outputs[0], initialized(accumulators))
            })
        }
    }

    fn execute_contraction_tree_or_generic(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
        tree: &EinsumContractionTreePlan,
    ) -> Result<EinsumRoute> {
        let Some(choice) = self.tree_choice(inputs[0].dtype)? else {
            self.execute_generic_native(inputs, outputs, workspace)?;
            return Ok(EinsumRoute::GenericNative);
        };
        let candidate = tree.candidates()[choice.candidate]
            .supported()
            .ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "Einsum `{}` compiled tree choice {} is unsupported; rebuild the kernel",
                    self.plan.equation(),
                    choice.candidate
                ))
            })?;
        let actual_workspace_bytes = crate::dispatch_arith!(inputs[0].dtype, "Einsum", T => {
            self.execute_contraction_tree_typed::<T>(
                inputs,
                outputs,
                candidate,
                choice.bytes,
                EINSUM_TREE_WORKSPACE_LIMIT,
            )
        })?;
        self.last_workspace_bytes
            .store(actual_workspace_bytes, Ordering::Relaxed);
        Ok(match tree.quality() {
            EinsumPlannerQuality::ExactSubsetDp => EinsumRoute::OptimizedDp,
            EinsumPlannerQuality::DeterministicGreedy => EinsumRoute::OptimizedHeuristic,
            EinsumPlannerQuality::GenericNativeFallback => unreachable!("handled above"),
        })
    }

    fn execute_scalar_gemm_typed<T: EinsumElement>(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        gemm: &EinsumContractionPlan,
    ) -> Result<()> {
        let left = TreeValue::Leaf(prepare_tree_leaf::<T>(
            self.plan.equation(),
            &inputs[0],
            &self.plan.operands()[0],
        )?);
        let right = TreeValue::Leaf(prepare_tree_leaf::<T>(
            self.plan.equation(),
            &inputs[1],
            &self.plan.operands()[1],
        )?);
        let mut canonical_axes = Vec::new();
        canonical_axes.extend_from_slice(gemm.batch_axes());
        canonical_axes.extend_from_slice(gemm.left_free_axes());
        canonical_axes.extend_from_slice(gemm.right_free_axes());
        let mut iteration_axes = canonical_axes.clone();
        iteration_axes.extend_from_slice(gemm.contract_axes());
        let iteration_shape = axes_shape(&self.plan, &iteration_axes)?;
        let output_rank = canonical_axes.len();
        let canonical_shape = iteration_shape[..output_rank].to_vec();
        let canonical_len = checked_numel("scalar GEMM output", &canonical_shape)?;
        let mut canonical = Vec::new();
        resize_scratch(&mut canonical, canonical_len)?;
        let left = prepare_tree_accessor(
            self.plan.equation(),
            &left,
            &iteration_axes,
            &iteration_shape,
        )?;
        let right = prepare_tree_accessor(
            self.plan.equation(),
            &right,
            &iteration_axes,
            &iteration_shape,
        )?;
        evaluate_tree_product::<T>(
            &mut canonical,
            &iteration_shape[..output_rank],
            &iteration_shape[output_rank..],
            &[left, right],
        )?;
        let final_value = DenseTreeValue {
            axes: canonical_axes,
            shape: canonical_shape,
            data: canonical,
        };
        let workspace_bytes = final_value
            .data
            .capacity()
            .checked_mul(std::mem::size_of::<T::Acc>())
            .ok_or_else(|| geometry_overflow(self.plan.equation(), "scalar GEMM workspace"))?;
        self.last_workspace_bytes
            .store(workspace_bytes, Ordering::Relaxed);
        write_tree_output::<T>(
            self.plan.equation(),
            &self.plan,
            &final_value,
            gemm.output_permutation(),
            &mut outputs[0],
        )
    }

    fn execute_contraction_tree_typed<T: EinsumElement>(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        candidate: &EinsumSupportedContractionTreeCandidate,
        admitted_workspace_bytes: usize,
        workspace_limit: usize,
    ) -> Result<usize> {
        let max_value = candidate
            .steps()
            .iter()
            .map(EinsumContractionTreeStep::output)
            .map(EinsumValueId::index)
            .chain(std::iter::once(candidate.final_output().index()))
            .max()
            .unwrap_or(inputs.len().saturating_sub(1));
        let mut values: Vec<Option<TreeValue<T>>> = (0..=max_value).map(|_| None).collect();
        for (input_index, input) in inputs.iter().enumerate() {
            let operand = self.plan.operands().get(input_index).ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "Einsum `{}` contraction tree references missing input #{input_index}",
                    self.plan.equation()
                ))
            })?;
            values[input_index] = Some(TreeValue::Leaf(prepare_tree_leaf::<T>(
                self.plan.equation(),
                input,
                operand,
            )?));
        }

        let mut temporary_slots: Vec<Option<Vec<T::Acc>>> =
            (0..candidate.cost().slot_count()).map(|_| None).collect();
        let mut temporary_plans = vec![None; values.len()];
        for temporary in candidate.temporaries() {
            let value = temporary.value().index();
            if value >= temporary_plans.len() || temporary.slot() >= temporary_slots.len() {
                return Err(EpError::KernelFailed(format!(
                    "Einsum `{}` contraction candidate has an invalid temporary schedule for {}",
                    self.plan.equation(),
                    temporary.value()
                )));
            }
            temporary_plans[value] = Some((temporary.slot(), temporary.last_use_step()));
        }

        for (step_index, step) in candidate.steps().iter().enumerate() {
            let (output_id, output_axes, input_ids, mut buffer) = match step {
                EinsumContractionTreeStep::UnaryReduction(unary) => {
                    let output_id = unary.output();
                    let buffer = take_tree_output_buffer(
                        self.plan.equation(),
                        output_id,
                        unary.output_axes(),
                        candidate.final_output(),
                        &temporary_plans,
                        &mut temporary_slots,
                        &self.plan,
                    )?;
                    (output_id, unary.output_axes(), vec![unary.input()], buffer)
                }
                EinsumContractionTreeStep::BinaryContraction(binary) => {
                    let output_id = binary.output();
                    let buffer = take_tree_output_buffer(
                        self.plan.equation(),
                        output_id,
                        binary.canonical_output_axes(),
                        candidate.final_output(),
                        &temporary_plans,
                        &mut temporary_slots,
                        &self.plan,
                    )?;
                    (
                        output_id,
                        binary.canonical_output_axes(),
                        vec![binary.left(), binary.right()],
                        buffer,
                    )
                }
                _ => {
                    return Err(EpError::KernelFailed(format!(
                        "Einsum `{}` contraction candidate contains a newer unsupported step",
                        self.plan.equation()
                    )));
                }
            };
            if values.get(output_id.index()).is_some_and(Option::is_some) {
                return Err(EpError::KernelFailed(format!(
                    "Einsum `{}` contraction candidate produces {} more than once",
                    self.plan.equation(),
                    output_id
                )));
            }

            match step {
                EinsumContractionTreeStep::UnaryReduction(unary) => {
                    execute_tree_unary(
                        &self.plan,
                        unary,
                        value_at(self.plan.equation(), &values, unary.input())?,
                        &mut buffer,
                    )?;
                }
                EinsumContractionTreeStep::BinaryContraction(binary) => {
                    let left = value_at(self.plan.equation(), &values, binary.left())?;
                    let right = value_at(self.plan.equation(), &values, binary.right())?;
                    execute_tree_binary(&self.plan, binary, left, right, &mut buffer)?;
                }
                _ => {
                    return Err(EpError::KernelFailed(format!(
                        "Einsum `{}` contraction candidate contains a newer unsupported step",
                        self.plan.equation()
                    )));
                }
            }
            let shape = axes_shape(&self.plan, output_axes)?;
            values[output_id.index()] = Some(TreeValue::Dense(DenseTreeValue {
                axes: output_axes.to_vec(),
                shape,
                data: buffer,
            }));

            for input_id in input_ids {
                let index = input_id.index();
                if temporary_plans
                    .get(index)
                    .and_then(|plan| *plan)
                    .is_some_and(|(_, last_use)| last_use == step_index)
                    && let Some(TreeValue::Dense(value)) = values[index].take()
                {
                    let slot = temporary_plans[index]
                        .expect("temporary plan was checked above")
                        .0;
                    if temporary_slots[slot].is_some() {
                        return Err(EpError::KernelFailed(format!(
                            "Einsum `{}` contraction temporary slot {slot} was released twice at \
                             step {step_index}",
                            self.plan.equation()
                        )));
                    }
                    temporary_slots[slot] = Some(value.data);
                }
            }
        }

        let final_value = values
            .get_mut(candidate.final_output().index())
            .and_then(Option::take)
            .ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "Einsum `{}` contraction candidate did not produce final value {}",
                    self.plan.equation(),
                    candidate.final_output()
                ))
            })?;
        let TreeValue::Dense(final_value) = final_value else {
            return Err(EpError::KernelFailed(format!(
                "Einsum `{}` contraction candidate ended at an input leaf",
                self.plan.equation()
            )));
        };
        let actual_workspace_bytes = contraction_tree_capacity_bytes(
            self.plan.equation(),
            &final_value,
            &values,
            &temporary_slots,
        )?;
        if actual_workspace_bytes > admitted_workspace_bytes
            || actual_workspace_bytes > workspace_limit
        {
            return Err(EpError::KernelFailed(format!(
                "Einsum `{}` contraction-tree accumulator workspace retained \
                 {actual_workspace_bytes} bytes after admitting {admitted_workspace_bytes} bytes \
                 against the {workspace_limit}-byte limit. HOW: use the GenericNative route or \
                 reduce the output/intermediate extents.",
                self.plan.equation()
            )));
        }
        drop(values);
        drop(temporary_slots);
        write_tree_output::<T>(
            self.plan.equation(),
            &self.plan,
            &final_value,
            candidate.final_output_permutation(),
            &mut outputs[0],
        )?;
        Ok(actual_workspace_bytes)
    }

    fn execute_generic(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
        iteration_axes: &[EinsumAxis],
        output_rank: usize,
        mappings: &[Vec<usize>],
        high_precision: bool,
    ) -> Result<()> {
        let iteration_shape = axes_shape(&self.plan, iteration_axes)?;
        let output_shape = &iteration_shape[..output_rank];
        let reduction_shape = &iteration_shape[output_rank..];
        let output_len = checked_numel("output", output_shape)?;
        let reduction_len = checked_numel("reduction", reduction_shape)?;

        let layouts: Vec<_> = inputs
            .iter()
            .zip(self.plan.operands())
            .map(|(input, operand)| unique_operand_layout(input, operand))
            .collect::<Result<_>>()?;
        let views: Vec<_> = inputs
            .iter()
            .zip(&layouts)
            .map(|(input, layout)| {
                TensorView::new(
                    input.data,
                    input.dtype,
                    &layout.shape,
                    &layout.strides,
                    input.device,
                )
                .with_byte_offset(input.byte_offset)
            })
            .collect();
        let dense: Vec<Cow<'_, [f32]>> = views
            .iter()
            .map(|view| to_dense_f32_widen("Einsum", view))
            .collect::<Result<_>>()?;
        let operand_iteration_strides: Vec<Vec<usize>> = layouts
            .iter()
            .zip(mappings)
            .map(|(layout, mapping)| {
                let dense_strides = compute_contiguous_strides(&layout.shape);
                let mut strides = vec![0usize; iteration_axes.len()];
                for (unique_axis, &iteration_axis) in mapping.iter().enumerate() {
                    let iter_extent = iteration_shape[iteration_axis];
                    strides[iteration_axis] = if layout.shape[unique_axis] == 1 && iter_extent != 1
                    {
                        0
                    } else {
                        usize::try_from(dense_strides[unique_axis]).map_err(|_| {
                            EpError::KernelFailed(format!(
                                "Einsum `{}` produced a negative dense stride for operand axis \
                                 {unique_axis}",
                                self.plan.equation()
                            ))
                        })?
                    };
                }
                Ok(strides)
            })
            .collect::<Result<_>>()?;

        // SAFETY: this call exclusively borrows live CPU workspace. Initialize
        // before forming readable f32, as the original oracle also did.
        unsafe {
            with_zeroed_accumulator::<f32, _>(workspace, output_len, |f32_output| {
                let identity_mappings = mappings
                    .iter()
                    .all(|mapping| mapping.iter().copied().eq(0..iteration_axes.len()));
                let aligned_dense = layouts.iter().zip(&dense).all(|(layout, data)| {
                    layout.shape == iteration_shape && data.len() == output_len
                });
                if !high_precision && reduction_len == 1 && identity_mappings && aligned_dense {
                    const PARALLEL_ELEMENTWISE_MIN_ELEMS: usize = 64 * 1024;
                    let evaluate = |index: usize| {
                        dense
                            .iter()
                            .fold(1.0f32, |product, operand| product * operand[index])
                    };
                    if output_len >= PARALLEL_ELEMENTWISE_MIN_ELEMS {
                        f32_output
                            .par_iter_mut()
                            .enumerate()
                            .for_each(|(index, output)| *output = evaluate(index));
                    } else {
                        f32_output
                            .iter_mut()
                            .enumerate()
                            .for_each(|(index, output)| *output = evaluate(index));
                    }
                    return write_dense_f32_narrow("Einsum", &mut outputs[0], f32_output);
                }
                if !high_precision
                    && dense.len() == 1
                    && reduction_len != 0
                    && identity_mappings
                    && layouts[0].shape == iteration_shape
                    && dense[0].len() == output_len.saturating_mul(reduction_len)
                {
                    const PARALLEL_REDUCTION_MIN_ELEMS: usize = 64 * 1024;
                    let data = dense[0].as_ref();
                    let reduce_one = |(output, values): (&mut f32, &[f32])| {
                        *output = values.iter().copied().sum();
                    };
                    if data.len() >= PARALLEL_REDUCTION_MIN_ELEMS && output_len > 1 {
                        f32_output
                            .par_iter_mut()
                            .zip(data.par_chunks(reduction_len))
                            .for_each(reduce_one);
                    } else {
                        f32_output
                            .iter_mut()
                            .zip(data.chunks(reduction_len))
                            .for_each(reduce_one);
                    }
                    return write_dense_f32_narrow("Einsum", &mut outputs[0], f32_output);
                }
                if output_len != 0 && reduction_len != 0 {
                    let mut output_index = vec![0usize; output_rank];
                    for (output_offset, output) in
                        f32_output.iter_mut().enumerate().take(output_len)
                    {
                        let mut reduction_index = vec![0usize; reduction_shape.len()];
                        let mut first = true;
                        let mut sum_f32 = 0.0f32;
                        let mut sum_f64 = 0.0f64;
                        while first || next_index(reduction_shape, &mut reduction_index) {
                            first = false;
                            let mut product_f32 = 1.0f32;
                            let mut product_f64 = 1.0f64;
                            for ((data, strides), _operand) in dense
                                .iter()
                                .zip(&operand_iteration_strides)
                                .zip(self.plan.operands())
                            {
                                let mut offset = 0usize;
                                for axis in 0..iteration_axes.len() {
                                    let index = if axis < output_rank {
                                        output_index[axis]
                                    } else {
                                        reduction_index[axis - output_rank]
                                    };
                                    offset = offset
                                        .checked_add(index.checked_mul(strides[axis]).ok_or_else(
                                            || {
                                                geometry_overflow(
                                                    self.plan.equation(),
                                                    "operand offset",
                                                )
                                            },
                                        )?)
                                        .ok_or_else(|| {
                                            geometry_overflow(
                                                self.plan.equation(),
                                                "operand offset",
                                            )
                                        })?;
                                }
                                let value = *data.get(offset).ok_or_else(|| {
                                    EpError::KernelFailed(format!(
                                        "Einsum `{}` canonical operand offset {offset} exceeded a \
                                     dense operand with {} element(s)",
                                        self.plan.equation(),
                                        data.len()
                                    ))
                                })?;
                                product_f32 *= value;
                                product_f64 *= f64::from(value);
                            }
                            if high_precision {
                                sum_f64 += product_f64;
                            } else {
                                sum_f32 += product_f32;
                            }
                            if reduction_shape.is_empty() {
                                break;
                            }
                        }
                        *output = if high_precision {
                            sum_f64 as f32
                        } else {
                            sum_f32
                        };
                        if output_offset + 1 < output_len {
                            let advanced = next_index(output_shape, &mut output_index);
                            debug_assert!(advanced);
                        }
                    }
                }
                write_dense_f32_narrow("Einsum", &mut outputs[0], f32_output)
            })
        }
    }

    fn execute_gemm(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
        gemm: &EinsumContractionPlan,
    ) -> Result<EinsumRoute> {
        let geometry = self
            .plan
            .resolve_concrete_gemm_geometry(
                &inputs.iter().map(|input| input.shape).collect::<Vec<_>>(),
            )
            .map_err(|error| {
                EpError::KernelFailed(format!(
                    "Einsum `{}` could not resolve GEMM geometry: {error}",
                    self.plan.equation()
                ))
            })?
            .ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "Einsum `{}` canonical classification changed before GEMM execution",
                    self.plan.equation()
                ))
            })?;
        let left_ordered =
            ordered_operand_layout(&inputs[0], &self.plan.operands()[0], gemm.left_axis_order())?;
        let right_ordered = ordered_operand_layout(
            &inputs[1],
            &self.plan.operands()[1],
            gemm.right_axis_order(),
        )?;
        let left = collapse_operand_layout(
            &left_ordered,
            gemm.batch_axes().len(),
            gemm.left_free_axes().len(),
            gemm.contract_axes().len(),
        );
        let right = collapse_operand_layout(
            &right_ordered,
            gemm.batch_axes().len(),
            gemm.contract_axes().len(),
            gemm.right_free_axes().len(),
        );
        let output = collapsed_output_layout(&self.plan, gemm, &outputs[0]);

        let output_aliases_input = inputs
            .iter()
            .any(|input| views_may_overlap(input, &outputs[0]));
        if !output_aliases_input
            && let (Some(left), Some(right), Some(output)) = (left, right, output)
        {
            let left_view = TensorView::new(
                inputs[0].data,
                inputs[0].dtype,
                &left.shape,
                &left.strides,
                inputs[0].device,
            )
            .with_byte_offset(inputs[0].byte_offset);
            let right_view = TensorView::new(
                inputs[1].data,
                inputs[1].dtype,
                &right.shape,
                &right.strides,
                inputs[1].device,
            )
            .with_byte_offset(inputs[1].byte_offset);
            let output_view = TensorMut::new(
                outputs[0].data,
                outputs[0].dtype,
                &output.shape,
                &output.strides,
                outputs[0].device,
            )
            .with_byte_offset(outputs[0].byte_offset);
            self.matmul
                .execute(&[left_view, right_view], &mut [output_view])?;
            return Ok(EinsumRoute::MatMulDirect);
        }

        let left_view = TensorView::new(
            inputs[0].data,
            inputs[0].dtype,
            &left_ordered.shape,
            &left_ordered.strides,
            inputs[0].device,
        )
        .with_byte_offset(inputs[0].byte_offset);
        let right_view = TensorView::new(
            inputs[1].data,
            inputs[1].dtype,
            &right_ordered.shape,
            &right_ordered.strides,
            inputs[1].device,
        )
        .with_byte_offset(inputs[1].byte_offset);
        let left_dense = to_dense_f32_widen("Einsum", &left_view)?;
        let right_dense = to_dense_f32_widen("Einsum", &right_view)?;
        let materialized_input_bytes = [&left_dense, &right_dense]
            .into_iter()
            .map(|data| match data {
                Cow::Borrowed(_) => Some(0),
                Cow::Owned(data) => data.len().checked_mul(std::mem::size_of::<f32>()),
            })
            .try_fold(0usize, |total, bytes| {
                bytes.and_then(|bytes| total.checked_add(bytes))
            })
            .ok_or_else(|| geometry_overflow(self.plan.equation(), "GEMM input workspace"))?;
        let batch_rank = gemm.batch_axes().len();
        let left_shape = flattened_gemm_shape(
            &left_ordered.shape[..batch_rank],
            geometry.m(),
            geometry.k(),
        );
        let right_shape = flattened_gemm_shape(
            &right_ordered.shape[..batch_rank],
            geometry.k(),
            geometry.n(),
        );
        let left_strides = compute_contiguous_strides(&left_shape);
        let right_strides = compute_contiguous_strides(&right_shape);
        let left_f32 = TensorView::new(
            onnx_runtime_ep_api::DevicePtr(left_dense.as_ptr().cast()),
            DataType::Float32,
            &left_shape,
            &left_strides,
            onnx_runtime_ir::DeviceId::cpu(),
        );
        let right_f32 = TensorView::new(
            onnx_runtime_ep_api::DevicePtr(right_dense.as_ptr().cast()),
            DataType::Float32,
            &right_shape,
            &right_strides,
            onnx_runtime_ir::DeviceId::cpu(),
        );
        let canonical_shape =
            flattened_gemm_shape(geometry.batch_shape(), geometry.m(), geometry.n());
        let canonical_strides = compute_contiguous_strides(&canonical_shape);
        let canonical_len = checked_numel("GEMM output", &canonical_shape)?;
        let workspace_bytes = canonical_len
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|bytes| bytes.checked_add(materialized_input_bytes))
            .ok_or_else(|| geometry_overflow(self.plan.equation(), "GEMM workspace"))?;
        self.last_workspace_bytes
            .store(workspace_bytes, Ordering::Relaxed);
        let _probe = ConcurrencyProbeGuard::enter(CONCURRENCY_MATERIALIZED_GEMM);
        // SAFETY: exclusive CPU workspace; MatMul currently forms initialized
        // f32 references, so its destination must be initialized before entry.
        unsafe {
            with_zeroed_accumulator::<f32, _>(workspace, canonical_len, |f32_output| {
                let canonical_output = TensorMut::new(
                    onnx_runtime_ep_api::DevicePtrMut(f32_output.as_mut_ptr().cast()),
                    DataType::Float32,
                    &canonical_shape,
                    &canonical_strides,
                    onnx_runtime_ir::DeviceId::cpu(),
                );
                self.matmul
                    .execute(&[left_f32, right_f32], &mut [canonical_output])?;
                write_canonical_output(&self.plan, gemm, f32_output, &mut outputs[0])
            })
        }?;
        Ok(EinsumRoute::MatMulMaterialized)
    }

    #[cfg(test)]
    fn record_route(&self, route: EinsumRoute) {
        self.last_route.store(
            u8::try_from(route.telemetry_index() + 1).expect("route index fits u8"),
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    #[cfg(not(test))]
    fn record_route(&self, _route: EinsumRoute) {}
}

trait EinsumElement: NumericElem<Acc: Send + Sync + 'static> + Send + Sync + 'static {
    fn one() -> Self::Acc {
        Self::from_f32_scalar(1.0).to_acc()
    }

    fn write_accumulators(output: &mut TensorMut, data: &[Self::Acc]) -> Result<()>;
}

macro_rules! impl_einsum_float32_accumulator {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl EinsumElement for $ty {
                fn write_accumulators(output: &mut TensorMut, data: &[f32]) -> Result<()> {
                    write_dense_f32_narrow("Einsum", output, data)
                }
            }
        )+
    };
}

impl_einsum_float32_accumulator!(f32, half::f16, half::bf16);

impl EinsumElement for f64 {
    fn write_accumulators(output: &mut TensorMut, data: &[f64]) -> Result<()> {
        write_dense::<f64>(output, data)
    }
}

macro_rules! impl_einsum_integer {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl EinsumElement for $ty {
                fn write_accumulators(output: &mut TensorMut, data: &[$ty]) -> Result<()> {
                    write_dense::<$ty>(output, data)
                }
            }
        )+
    };
}

impl_einsum_integer!(i8, i16, i32, i64, u8, u16, u32, u64);

#[derive(Clone, Copy)]
struct ReadPtr<T>(*const T);

impl<T> ReadPtr<T> {
    fn get(self) -> *const T {
        self.0
    }
}

// SAFETY: GenericNative only constructs this wrapper from immutable TensorView
// inputs. The call owns no mutable reference to their storage until all
// parallel reads have completed; an aliased output is written only afterwards.
unsafe impl<T: Sync> Send for ReadPtr<T> {}
// SAFETY: See the `Send` implementation above.
unsafe impl<T: Sync> Sync for ReadPtr<T> {}

struct PreparedGenericInput<T> {
    origin: ReadPtr<T>,
    iteration_strides: Vec<i128>,
}

impl<T: EinsumElement> PreparedGenericInput<T> {
    #[inline]
    fn read(&self, coordinates: &[usize]) -> T::Acc {
        let offset = self
            .iteration_strides
            .iter()
            .zip(coordinates)
            .fold(0i128, |sum, (&stride, &index)| sum + stride * index as i128);
        // `prepare_generic_inputs` proves the complete addressed range fits
        // `isize`, and every coordinate here is inside that range.
        let offset = offset as isize;
        // SAFETY: the executor validates the backing view bounds before kernel
        // dispatch. `offset` is derived only from in-range coordinates and the
        // plan's physical-axis map, including diagonal stride summation and
        // ellipsis broadcast zero-strides. Reads are unaligned because
        // `byte_offset` need only satisfy dtype alignment, not Rust `T`.
        unsafe { self.origin.get().offset(offset).read_unaligned().to_acc() }
    }
}

struct TreeLeaf<T> {
    origin: ReadPtr<T>,
    axes: Vec<EinsumAxis>,
    shape: Vec<usize>,
    strides: Vec<i128>,
}

struct DenseTreeValue<A> {
    axes: Vec<EinsumAxis>,
    shape: Vec<usize>,
    data: Vec<A>,
}

enum TreeValue<T: EinsumElement> {
    Leaf(TreeLeaf<T>),
    Dense(DenseTreeValue<T::Acc>),
}

enum TreeReadSource<T: EinsumElement> {
    Leaf(ReadPtr<T>),
    Accumulator(ReadPtr<T::Acc>),
}

struct TreeAccessor<T: EinsumElement> {
    source: TreeReadSource<T>,
    iteration_strides: Vec<i128>,
}

impl<T: EinsumElement> TreeAccessor<T> {
    #[inline]
    fn read(&self, coordinates: &[usize]) -> T::Acc {
        let offset = self
            .iteration_strides
            .iter()
            .zip(coordinates)
            .fold(0i128, |sum, (&stride, &index)| sum + stride * index as i128)
            as isize;
        // SAFETY: `prepare_tree_accessor` proves the complete addressed range
        // fits `isize`; the source leaf or temporary remains alive and
        // immutable until the blocking tile dispatch returns.
        unsafe {
            match self.source {
                TreeReadSource::Leaf(pointer) => {
                    pointer.get().offset(offset).read_unaligned().to_acc()
                }
                TreeReadSource::Accumulator(pointer) => {
                    pointer.get().offset(offset).read_unaligned()
                }
            }
        }
    }
}

fn prepare_tree_leaf<T: EinsumElement>(
    equation: &str,
    input: &TensorView,
    operand: &EinsumOperandPlan,
) -> Result<TreeLeaf<T>> {
    if input.dtype != T::DTYPE {
        return Err(EpError::KernelFailed(format!(
            "Einsum `{equation}` contraction tree expected {:?}, but input #{} has {:?}",
            T::DTYPE,
            operand.input(),
            input.dtype
        )));
    }
    let mut axes = Vec::with_capacity(operand.unique_axes().len());
    let mut shape = Vec::with_capacity(operand.unique_axes().len());
    let mut strides = Vec::with_capacity(operand.unique_axes().len());
    for logical in operand.unique_axes() {
        let &first = logical.input_axes().first().ok_or_else(|| {
            EpError::KernelFailed(format!(
                "Einsum `{equation}` input #{} has a logical axis with no physical source axis",
                operand.input()
            ))
        })?;
        let stride = logical
            .input_axes()
            .iter()
            .try_fold(0i128, |sum, &physical| {
                sum.checked_add(i128::from(input.strides[physical]))
            })
            .ok_or_else(|| geometry_overflow(equation, "contraction-tree diagonal stride"))?;
        axes.push(logical.axis());
        shape.push(input.shape[first]);
        strides.push(stride);
    }
    validate_iteration_address_range(equation, input, &shape, &strides)?;
    Ok(TreeLeaf {
        origin: ReadPtr(input.data_ptr::<T>()),
        axes,
        shape,
        strides,
    })
}

fn value_at<'a, T: EinsumElement>(
    equation: &str,
    values: &'a [Option<TreeValue<T>>],
    id: EinsumValueId,
) -> Result<&'a TreeValue<T>> {
    values
        .get(id.index())
        .and_then(Option::as_ref)
        .ok_or_else(|| {
            EpError::KernelFailed(format!(
                "Einsum `{equation}` contraction candidate referenced unavailable value {id}"
            ))
        })
}

fn contraction_tree_workspace_bytes(
    equation: &str,
    plan: &EinsumShapePlan,
    candidate: &EinsumSupportedContractionTreeCandidate,
    output_shape: &[usize],
    accumulator_size: usize,
) -> Result<usize> {
    let output_elements = checked_numel("contraction-tree output", output_shape)?;
    let mut slot_elements = vec![0usize; candidate.cost().slot_count()];
    for temporary in candidate.temporaries() {
        let shape = axes_shape(plan, temporary.axes())?;
        let elements = checked_numel("contraction-tree retained temporary slot", &shape)?;
        let slot = slot_elements.get_mut(temporary.slot()).ok_or_else(|| {
            EpError::KernelFailed(format!(
                "Einsum `{equation}` contraction candidate references missing retained slot {}",
                temporary.slot()
            ))
        })?;
        *slot = (*slot).max(elements);
    }
    slot_elements
        .into_iter()
        .try_fold(output_elements, usize::checked_add)
        .and_then(|elements| elements.checked_mul(accumulator_size))
        .ok_or_else(|| geometry_overflow(equation, "contraction-tree retained workspace"))
}

fn contraction_tree_capacity_bytes<T: EinsumElement>(
    equation: &str,
    final_value: &DenseTreeValue<T::Acc>,
    values: &[Option<TreeValue<T>>],
    slots: &[Option<Vec<T::Acc>>],
) -> Result<usize> {
    let mut elements = final_value.data.capacity();
    for value in values.iter().flatten() {
        if let TreeValue::Dense(value) = value {
            elements = elements.checked_add(value.data.capacity()).ok_or_else(|| {
                geometry_overflow(equation, "contraction-tree live value capacity")
            })?;
        }
    }
    for slot in slots.iter().flatten() {
        elements = elements.checked_add(slot.capacity()).ok_or_else(|| {
            geometry_overflow(equation, "contraction-tree retained slot capacity")
        })?;
    }
    elements
        .checked_mul(std::mem::size_of::<T::Acc>())
        .ok_or_else(|| geometry_overflow(equation, "contraction-tree accumulator capacity"))
}

fn take_tree_output_buffer<A: Default>(
    equation: &str,
    output: EinsumValueId,
    output_axes: &[EinsumAxis],
    final_output: EinsumValueId,
    temporary_plans: &[Option<(usize, usize)>],
    slots: &mut [Option<Vec<A>>],
    plan: &EinsumShapePlan,
) -> Result<Vec<A>> {
    let shape = axes_shape(plan, output_axes)?;
    let len = checked_numel("contraction-tree temporary", &shape)?;
    let mut buffer = if output == final_output {
        Vec::new()
    } else {
        let (slot, _) = temporary_plans
            .get(output.index())
            .and_then(|entry| *entry)
            .ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "Einsum `{equation}` contraction candidate omitted storage for {output}"
                ))
            })?;
        slots
            .get_mut(slot)
            .ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "Einsum `{equation}` contraction candidate references missing slot {slot}"
                ))
            })?
            .take()
            .unwrap_or_default()
    };
    resize_scratch(&mut buffer, len)?;
    Ok(buffer)
}

fn prepare_tree_accessor<T: EinsumElement>(
    equation: &str,
    value: &TreeValue<T>,
    iteration_axes: &[EinsumAxis],
    iteration_shape: &[usize],
) -> Result<TreeAccessor<T>> {
    let (source, axes, shape, source_strides): (
        TreeReadSource<T>,
        &[EinsumAxis],
        &[usize],
        Vec<i128>,
    ) = match value {
        TreeValue::Leaf(value) => (
            TreeReadSource::Leaf(value.origin),
            &value.axes,
            &value.shape,
            value.strides.clone(),
        ),
        TreeValue::Dense(value) => (
            TreeReadSource::Accumulator(ReadPtr(value.data.as_ptr())),
            &value.axes,
            &value.shape,
            compute_contiguous_strides(&value.shape)
                .into_iter()
                .map(i128::from)
                .collect(),
        ),
    };
    if axes.len() != shape.len() || axes.len() != source_strides.len() {
        return Err(EpError::KernelFailed(format!(
            "Einsum `{equation}` contraction value has inconsistent axis/shape/stride ranks"
        )));
    }
    let mut iteration_strides = vec![0i128; iteration_axes.len()];
    for (source_axis, axis) in axes.iter().enumerate() {
        let iteration_axis = iteration_axes
            .iter()
            .position(|candidate| candidate == axis)
            .ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "Einsum `{equation}` contraction step omitted live {axis}"
                ))
            })?;
        iteration_strides[iteration_axis] =
            if shape[source_axis] == 1 && iteration_shape[iteration_axis] != 1 {
                0
            } else {
                source_strides[source_axis]
            };
    }
    validate_offset_range(
        equation,
        "contraction-tree value",
        iteration_shape,
        &iteration_strides,
    )?;
    Ok(TreeAccessor {
        source,
        iteration_strides,
    })
}

fn execute_tree_unary<T: EinsumElement>(
    shape_plan: &EinsumShapePlan,
    unary: &EinsumUnaryReductionPlan,
    input: &TreeValue<T>,
    output: &mut [T::Acc],
) -> Result<()> {
    let mut iteration_axes = unary.output_axes().to_vec();
    iteration_axes.extend_from_slice(unary.reduction_axes());
    let iteration_shape = axes_shape(shape_plan, &iteration_axes)?;
    let output_rank = unary.output_axes().len();
    let accessor = prepare_tree_accessor(
        shape_plan.equation(),
        input,
        &iteration_axes,
        &iteration_shape,
    )?;
    evaluate_tree_product::<T>(
        output,
        &iteration_shape[..output_rank],
        &iteration_shape[output_rank..],
        &[accessor],
    )
}

fn execute_tree_binary<T: EinsumElement>(
    shape_plan: &EinsumShapePlan,
    binary: &EinsumBinaryContractionPlan,
    left: &TreeValue<T>,
    right: &TreeValue<T>,
    output: &mut [T::Acc],
) -> Result<()> {
    let mut iteration_axes = binary.canonical_output_axes().to_vec();
    iteration_axes.extend_from_slice(binary.contract_axes());
    let iteration_shape = axes_shape(shape_plan, &iteration_axes)?;
    let output_rank = binary.canonical_output_axes().len();
    let left = prepare_tree_accessor(
        shape_plan.equation(),
        left,
        &iteration_axes,
        &iteration_shape,
    )?;
    let right = prepare_tree_accessor(
        shape_plan.equation(),
        right,
        &iteration_axes,
        &iteration_shape,
    )?;
    evaluate_tree_product::<T>(
        output,
        &iteration_shape[..output_rank],
        &iteration_shape[output_rank..],
        &[left, right],
    )
}

fn evaluate_tree_product<T: EinsumElement>(
    output: &mut [T::Acc],
    output_shape: &[usize],
    reduction_shape: &[usize],
    inputs: &[TreeAccessor<T>],
) -> Result<()> {
    let output_len = checked_numel("contraction-tree output", output_shape)?;
    if output.len() != output_len {
        return Err(EpError::KernelFailed(format!(
            "Einsum contraction-tree output buffer has {} elements, expected {output_len}",
            output.len()
        )));
    }
    if output_len == 0 {
        return Ok(());
    }
    let reduction_len = checked_numel("contraction-tree reduction", reduction_shape)?;
    let iteration_rank = output_shape.len() + reduction_shape.len();
    let work = (output_len as u128)
        .checked_mul(reduction_len.max(1) as u128)
        .and_then(|value| value.checked_mul(inputs.len() as u128))
        .ok_or_else(|| EpError::KernelFailed("Einsum contraction-tree work overflowed".into()))?;
    let tile = generic_output_tile(reduction_len, inputs.len(), output_len);
    let evaluate = |first_tile: usize, destination: &mut [T::Acc]| {
        let first_output = first_tile * tile;
        let mut coordinates = vec![0usize; iteration_rank];
        for (local, value) in destination.iter_mut().enumerate() {
            decode_row_major(
                first_output + local,
                output_shape,
                &mut coordinates[..output_shape.len()],
            );
            let mut sum = T::Acc::default();
            for reduction_linear in 0..reduction_len {
                decode_row_major(
                    reduction_linear,
                    reduction_shape,
                    &mut coordinates[output_shape.len()..],
                );
                let mut product = T::one();
                for input in inputs {
                    product = product.c_mul(input.read(&coordinates));
                }
                sum = if reduction_linear == 0 {
                    product
                } else {
                    sum.c_add(product)
                };
            }
            *value = sum;
        }
    };
    const PARALLEL_TREE_WORK_ITEMS: u128 = 128 * 1024;
    if output_len > 1 && work >= PARALLEL_TREE_WORK_ITEMS {
        crate::task_runtime::chunk_runs_mut(output, tile, 1, evaluate);
    } else {
        evaluate(0, output);
    }
    Ok(())
}

fn write_tree_output<T: EinsumElement>(
    equation: &str,
    plan: &EinsumShapePlan,
    final_value: &DenseTreeValue<T::Acc>,
    permutation: &[usize],
    output: &mut TensorMut,
) -> Result<()> {
    let requested_shape = axes_shape(plan, plan.output_axes())?;
    let requested_len = checked_numel("contraction-tree requested output", &requested_shape)?;
    if final_value.data.len() != requested_len {
        return Err(EpError::KernelFailed(format!(
            "Einsum `{equation}` contraction final element count mismatch: canonical={}, \
             output={requested_len}",
            final_value.data.len()
        )));
    }
    if final_value.axes == plan.output_axes()
        && (permutation.is_empty() || permutation.iter().copied().eq(0..permutation.len()))
    {
        return T::write_accumulators(output, &final_value.data);
    }
    if permutation.len() != requested_shape.len() || permutation.len() != final_value.shape.len() {
        return Err(EpError::KernelFailed(format!(
            "Einsum `{equation}` contraction final permutation rank {} does not match requested \
             rank {} and canonical rank {}",
            permutation.len(),
            requested_shape.len(),
            final_value.shape.len()
        )));
    }
    output.validate()?;
    if output.dtype != T::DTYPE {
        return Err(EpError::InvalidTensorView {
            reason: format!(
                "Einsum `{equation}` expected {:?} output, got {:?}",
                T::DTYPE,
                output.dtype
            ),
        });
    }
    if output.shape != requested_shape {
        return Err(EpError::InvalidTensorView {
            reason: format!(
                "Einsum `{equation}` expected output shape {requested_shape:?}, got {:?}",
                output.shape
            ),
        });
    }
    if requested_len == 0 {
        return Ok(());
    }
    let canonical_strides = compute_contiguous_strides(&final_value.shape);
    let mut requested_index = vec![0usize; requested_shape.len()];
    let output_origin = output.data_ptr_mut::<T>();
    for linear in 0..requested_len {
        decode_row_major(linear, &requested_shape, &mut requested_index);
        let mut canonical_offset = 0usize;
        for (requested_axis, &canonical_axis) in permutation.iter().enumerate() {
            let stride = usize::try_from(canonical_strides[canonical_axis]).map_err(|_| {
                EpError::KernelFailed(format!(
                    "Einsum `{equation}` contraction final stride is negative"
                ))
            })?;
            canonical_offset = canonical_offset
                .checked_add(
                    requested_index[requested_axis]
                        .checked_mul(stride)
                        .ok_or_else(|| {
                            geometry_overflow(equation, "contraction final output offset")
                        })?,
                )
                .ok_or_else(|| geometry_overflow(equation, "contraction final output offset"))?;
        }
        let value = *final_value.data.get(canonical_offset).ok_or_else(|| {
            EpError::KernelFailed(format!(
                "Einsum `{equation}` contraction final offset {canonical_offset} exceeds {} \
                 canonical element(s)",
                final_value.data.len()
            ))
        })?;
        let output_offset = elem_offset(output.strides, &requested_index);
        // SAFETY: `output` is validated, `requested_index` is in shape, and the
        // row-major walk visits each logical output once. The final accumulator
        // is an independent allocation, so no source element aliases output.
        unsafe {
            output_origin
                .offset(output_offset)
                .write_unaligned(T::from_acc(value));
        }
    }
    Ok(())
}

fn validate_offset_range(
    equation: &str,
    target: &str,
    shape: &[usize],
    strides: &[i128],
) -> Result<()> {
    let (mut minimum, mut maximum) = (0i128, 0i128);
    for (&extent, &stride) in shape.iter().zip(strides) {
        let span = i128::try_from(extent.saturating_sub(1))
            .ok()
            .and_then(|extent| extent.checked_mul(stride))
            .ok_or_else(|| geometry_overflow(equation, target))?;
        if span < 0 {
            minimum = minimum
                .checked_add(span)
                .ok_or_else(|| geometry_overflow(equation, target))?;
        } else {
            maximum = maximum
                .checked_add(span)
                .ok_or_else(|| geometry_overflow(equation, target))?;
        }
    }
    if minimum < isize::MIN as i128 || maximum > isize::MAX as i128 {
        return Err(geometry_overflow(equation, target));
    }
    Ok(())
}

fn prepare_generic_inputs<T: EinsumElement>(
    equation: &str,
    inputs: &[TensorView],
    generic: &EinsumGenericNativePlan,
    iteration_shape: &[usize],
) -> Result<Vec<PreparedGenericInput<T>>> {
    let programs = generic.index_program().operands();
    if programs.len() != inputs.len() {
        return Err(EpError::KernelFailed(format!(
            "Einsum `{equation}` GenericNative program has {} operand map(s), but execution \
             supplied {} input(s)",
            programs.len(),
            inputs.len()
        )));
    }
    programs
        .iter()
        .map(|program| {
            let input = inputs.get(program.input()).ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "Einsum `{equation}` GenericNative program references missing input #{}",
                    program.input()
                ))
            })?;
            if input.dtype != T::DTYPE {
                return Err(EpError::KernelFailed(format!(
                    "Einsum `{equation}` GenericNative typed dispatch expected {:?}, but input #{} \
                     has {:?}",
                    T::DTYPE,
                    program.input(),
                    input.dtype
                )));
            }
            let mappings = program.physical_axis_to_iteration_axis();
            let broadcasts = program.physical_axis_broadcasts_when_one();
            if mappings.len() != input.shape.len() || broadcasts.len() != input.shape.len() {
                return Err(EpError::KernelFailed(format!(
                    "Einsum `{equation}` GenericNative map for input #{} has {} axis entries and \
                     {} broadcast entries for runtime rank {}",
                    program.input(),
                    mappings.len(),
                    broadcasts.len(),
                    input.shape.len()
                )));
            }
            let mut iteration_strides = vec![0i128; iteration_shape.len()];
            for physical_axis in 0..input.shape.len() {
                let iteration_axis = mappings[physical_axis];
                let iteration_extent = *iteration_shape.get(iteration_axis).ok_or_else(|| {
                    EpError::KernelFailed(format!(
                        "Einsum `{equation}` input #{} physical axis {physical_axis} maps to \
                         missing iteration axis {iteration_axis}",
                        program.input()
                    ))
                })?;
                let stride = if broadcasts[physical_axis]
                    && input.shape[physical_axis] == 1
                    && iteration_extent != 1
                {
                    0
                } else {
                    i128::from(input.strides[physical_axis])
                };
                iteration_strides[iteration_axis] = iteration_strides[iteration_axis]
                    .checked_add(stride)
                    .ok_or_else(|| geometry_overflow(equation, "GenericNative diagonal stride"))?;
            }
            validate_iteration_address_range(equation, input, iteration_shape, &iteration_strides)?;
            Ok(PreparedGenericInput {
                origin: ReadPtr(input.data_ptr::<T>()),
                iteration_strides,
            })
        })
        .collect()
}

fn validate_iteration_address_range(
    equation: &str,
    input: &TensorView,
    shape: &[usize],
    strides: &[i128],
) -> Result<()> {
    let (mut minimum, mut maximum) = (0i128, 0i128);
    for (&extent, &stride) in shape.iter().zip(strides) {
        let span = i128::try_from(extent.saturating_sub(1))
            .ok()
            .and_then(|extent| extent.checked_mul(stride))
            .ok_or_else(|| geometry_overflow(equation, "GenericNative address range"))?;
        if span < 0 {
            minimum = minimum
                .checked_add(span)
                .ok_or_else(|| geometry_overflow(equation, "GenericNative minimum address"))?;
        } else {
            maximum = maximum
                .checked_add(span)
                .ok_or_else(|| geometry_overflow(equation, "GenericNative maximum address"))?;
        }
    }
    if minimum < isize::MIN as i128 || maximum > isize::MAX as i128 {
        return Err(geometry_overflow(equation, "GenericNative element address"));
    }
    let element_size = input.dtype.byte_size() as i128;
    let origin = i128::try_from(input.data.0 as usize)
        .ok()
        .and_then(|base| base.checked_add(input.byte_offset as i128))
        .ok_or_else(|| geometry_overflow(equation, "GenericNative byte origin"))?;
    let first = minimum
        .checked_mul(element_size)
        .and_then(|offset| origin.checked_add(offset))
        .ok_or_else(|| geometry_overflow(equation, "GenericNative minimum byte address"))?;
    let end = maximum
        .checked_mul(element_size)
        .and_then(|offset| origin.checked_add(offset))
        .and_then(|last| last.checked_add(element_size))
        .ok_or_else(|| geometry_overflow(equation, "GenericNative maximum byte address"))?;
    if first < 0 || end > usize::MAX as i128 {
        return Err(geometry_overflow(
            equation,
            "GenericNative host pointer range",
        ));
    }
    Ok(())
}

fn decode_row_major(mut linear: usize, shape: &[usize], coordinates: &mut [usize]) {
    debug_assert_eq!(shape.len(), coordinates.len());
    for axis in (0..shape.len()).rev() {
        coordinates[axis] = linear % shape[axis];
        linear /= shape[axis];
    }
}

fn generic_output_tile(reduction_len: usize, operand_count: usize, output_len: usize) -> usize {
    const TARGET_WORK_ITEMS_PER_TILE: usize = 32 * 1024;
    let work_per_output = reduction_len.saturating_mul(operand_count).max(1);
    TARGET_WORK_ITEMS_PER_TILE
        .div_ceil(work_per_output)
        .clamp(1, output_len)
}

fn accumulator_element_size(dtype: DataType) -> usize {
    match dtype {
        DataType::Float16 | DataType::BFloat16 => std::mem::size_of::<f32>(),
        _ => dtype.byte_size(),
    }
}

#[derive(Clone, Debug)]
struct Layout {
    shape: Vec<usize>,
    strides: Vec<i64>,
}

fn unique_operand_layout(input: &TensorView, operand: &EinsumOperandPlan) -> Result<Layout> {
    let mut shape = Vec::with_capacity(operand.unique_axes().len());
    let mut strides = Vec::with_capacity(operand.unique_axes().len());
    for axis in operand.unique_axes() {
        let &first = axis.input_axes().first().ok_or_else(|| {
            EpError::KernelFailed("Einsum canonical operand axis has no physical axis".into())
        })?;
        shape.push(input.shape[first]);
        let stride = axis.input_axes().iter().try_fold(0i64, |sum, &physical| {
            sum.checked_add(input.strides[physical]).ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "Einsum diagonal stride overflowed for input #{} axis {physical}",
                    operand.input()
                ))
            })
        })?;
        strides.push(stride);
    }
    Ok(Layout { shape, strides })
}

fn permutation_layout(
    input: &TensorView,
    operand: &EinsumOperandPlan,
    permutation: &EinsumPermutationPlan,
) -> Result<Layout> {
    let unique = unique_operand_layout(input, operand)?;
    let mut shape = Vec::with_capacity(permutation.output_to_operand_axis().len());
    let mut strides = Vec::with_capacity(permutation.output_to_operand_axis().len());
    for &axis in permutation.output_to_operand_axis() {
        shape.push(*unique.shape.get(axis).ok_or_else(|| {
            EpError::KernelFailed(format!(
                "Einsum `{}` permutation references missing operand axis {axis}",
                permutation.input()
            ))
        })?);
        strides.push(unique.strides[axis]);
    }
    Ok(Layout { shape, strides })
}

fn ordered_operand_layout(
    input: &TensorView,
    operand: &EinsumOperandPlan,
    order: &[Option<usize>],
) -> Result<Layout> {
    let unique = unique_operand_layout(input, operand)?;
    let mut shape = Vec::with_capacity(order.len());
    let mut strides = Vec::with_capacity(order.len());
    for entry in order {
        match entry {
            Some(axis) => {
                shape.push(*unique.shape.get(*axis).ok_or_else(|| {
                    EpError::KernelFailed(format!(
                        "Einsum input #{} GEMM order references missing unique axis {axis}",
                        operand.input()
                    ))
                })?);
                strides.push(unique.strides[*axis]);
            }
            None => {
                shape.push(1);
                strides.push(0);
            }
        }
    }
    Ok(Layout { shape, strides })
}

fn collapse_operand_layout(
    ordered: &Layout,
    batch_rank: usize,
    rows_rank: usize,
    columns_rank: usize,
) -> Option<Layout> {
    if ordered.shape.len() != batch_rank + rows_rank + columns_rank {
        return None;
    }
    let (rows, row_stride) = collapse_group(
        &ordered.shape[batch_rank..batch_rank + rows_rank],
        &ordered.strides[batch_rank..batch_rank + rows_rank],
    )?;
    let (columns, column_stride) = collapse_group(
        &ordered.shape[batch_rank + rows_rank..],
        &ordered.strides[batch_rank + rows_rank..],
    )?;
    let mut shape = ordered.shape[..batch_rank].to_vec();
    shape.extend([rows, columns]);
    let mut strides = ordered.strides[..batch_rank].to_vec();
    strides.extend([row_stride, column_stride]);
    Some(Layout { shape, strides })
}

fn collapse_group(shape: &[usize], strides: &[i64]) -> Option<(usize, i64)> {
    if shape.is_empty() {
        return Some((1, 0));
    }
    let extent = shape
        .iter()
        .try_fold(1usize, |value, &dim| value.checked_mul(dim))?;
    if extent == 0 {
        return Some((0, 1));
    }
    let mut base_stride = None;
    let mut inner_extent = 1usize;
    for (&dim, &stride) in shape.iter().zip(strides).rev() {
        if dim <= 1 {
            continue;
        }
        match base_stride {
            None => base_stride = Some(stride),
            Some(base) => {
                let expected = i128::from(base) * i128::try_from(inner_extent).ok()?;
                if expected != i128::from(stride) {
                    return None;
                }
            }
        }
        inner_extent = inner_extent.checked_mul(dim)?;
    }
    Some((extent, base_stride.unwrap_or(0)))
}

fn collapsed_output_layout(
    plan: &EinsumShapePlan,
    gemm: &EinsumContractionPlan,
    output: &TensorMut,
) -> Option<Layout> {
    let canonical_shape = canonical_output_shape(plan, gemm).ok()?;
    let canonical_strides = canonical_output_strides(plan, gemm, output).ok()?;
    collapse_operand_layout(
        &Layout {
            shape: canonical_shape,
            strides: canonical_strides,
        },
        gemm.batch_axes().len(),
        gemm.left_free_axes().len(),
        gemm.right_free_axes().len(),
    )
}

fn canonical_output_shape(
    plan: &EinsumShapePlan,
    gemm: &EinsumContractionPlan,
) -> Result<Vec<usize>> {
    let mut axes = Vec::new();
    axes.extend_from_slice(gemm.batch_axes());
    axes.extend_from_slice(gemm.left_free_axes());
    axes.extend_from_slice(gemm.right_free_axes());
    axes_shape(plan, &axes)
}

fn canonical_output_strides(
    plan: &EinsumShapePlan,
    gemm: &EinsumContractionPlan,
    output: &TensorMut,
) -> Result<Vec<i64>> {
    if gemm.output_permutation().len() != output.shape.len() {
        return Err(EpError::KernelFailed(format!(
            "Einsum `{}` output permutation rank {} does not match output rank {}",
            plan.equation(),
            gemm.output_permutation().len(),
            output.shape.len()
        )));
    }
    let mut canonical = vec![None; gemm.output_permutation().len()];
    for (requested, &canonical_axis) in gemm.output_permutation().iter().enumerate() {
        let slot = canonical.get_mut(canonical_axis).ok_or_else(|| {
            EpError::KernelFailed(format!(
                "Einsum `{}` output permutation references canonical axis {canonical_axis}",
                plan.equation()
            ))
        })?;
        if slot.replace(output.strides[requested]).is_some() {
            return Err(EpError::KernelFailed(format!(
                "Einsum `{}` output permutation repeats canonical axis {canonical_axis}",
                plan.equation()
            )));
        }
    }
    canonical
        .into_iter()
        .map(|stride| {
            stride.ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "Einsum `{}` output permutation omitted a canonical axis",
                    plan.equation()
                ))
            })
        })
        .collect()
}

fn write_canonical_output(
    plan: &EinsumShapePlan,
    gemm: &EinsumContractionPlan,
    data: &[f32],
    output: &mut TensorMut,
) -> Result<()> {
    let shape = canonical_output_shape(plan, gemm)?;
    let strides = canonical_output_strides(plan, gemm, output)?;
    let mut canonical = TensorMut::new(output.data, output.dtype, &shape, &strides, output.device)
        .with_byte_offset(output.byte_offset);
    write_dense_f32_narrow("Einsum", &mut canonical, data)
}

fn flattened_gemm_shape(batch_shape: &[usize], rows: usize, columns: usize) -> Vec<usize> {
    let mut shape = batch_shape.to_vec();
    shape.extend([rows, columns]);
    shape
}

fn axes_shape(plan: &EinsumShapePlan, axes: &[EinsumAxis]) -> Result<Vec<usize>> {
    axes.iter()
        .map(|axis| {
            plan.logical_axes()
                .iter()
                .find(|logical| logical.axis() == *axis)
                .and_then(|logical| logical.dimension().as_static())
                .ok_or_else(|| {
                    EpError::KernelFailed(format!(
                        "Einsum `{}` execution plan has no concrete extent for {axis}",
                        plan.equation()
                    ))
                })
        })
        .collect()
}

fn checked_numel(label: &str, shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |value, &dim| value.checked_mul(dim))
        .ok_or_else(|| EpError::KernelFailed(format!("Einsum {label} element count overflowed")))
}

fn geometry_overflow(equation: &str, target: &str) -> EpError {
    EpError::KernelFailed(format!(
        "Einsum `{equation}` {target} overflowed usize; HOW: use smaller concrete dimensions"
    ))
}

fn gemm_flops(gemm: &EinsumContractionPlan) -> Option<u64> {
    let geometry = gemm.geometry();
    [
        geometry.batch().as_static()?,
        geometry.m().as_static()?,
        geometry.k().as_static()?,
        geometry.n().as_static()?,
        2,
    ]
    .into_iter()
    .try_fold(1u64, |value, factor| {
        value.checked_mul(u64::try_from(factor).ok()?)
    })
}

#[derive(Clone, Copy)]
struct TreeChoice {
    candidate: usize,
    bytes: usize,
}

fn select_tree_choice(
    plan: &EinsumShapePlan,
    tree: &EinsumContractionTreePlan,
    input_shapes: &[&[usize]],
    output_shape: &[usize],
    accumulator_size: usize,
) -> Result<Option<TreeChoice>> {
    if tree.quality() == EinsumPlannerQuality::GenericNativeFallback {
        return Ok(None);
    }
    let Some(concrete) = plan
        .resolve_concrete_contraction_tree(input_shapes, accumulator_size)
        .map_err(|error| {
            EpError::KernelFailed(format!(
                "Einsum `{}` could not resolve its concrete contraction tree: {error}",
                plan.equation()
            ))
        })?
    else {
        return Ok(None);
    };
    let mut candidates = BTreeMap::new();
    for candidate in concrete
        .candidates()
        .iter()
        .filter(|candidate| candidate.cost().is_some())
    {
        let (index, supported) = tree.candidates().iter().enumerate()
            .filter(|(_, structural)| structural.id() == candidate.id())
            .find_map(|(index, structural)| structural.supported().map(|supported| (index, supported)))
            .ok_or_else(|| EpError::KernelFailed(format!(
                "Einsum `{}` concrete candidate `{}` has no supported structural plan; rebuild canonical planning",
                plan.equation(), candidate.id()
            )))?;
        let bytes = contraction_tree_workspace_bytes(
            plan.equation(),
            plan,
            supported,
            output_shape,
            accumulator_size,
        )?;
        candidates.insert(
            candidate.id().clone(),
            TreeChoice {
                candidate: index,
                bytes,
            },
        );
    }
    Ok(concrete
        .preferred_candidate_matching(|candidate| {
            candidates
                .get(candidate.id())
                .is_some_and(|choice| choice.bytes <= EINSUM_TREE_WORKSPACE_LIMIT)
        })
        .and_then(|selected| candidates.get(selected.id()).copied()))
}

/// Caller proves CPU accessibility, exclusive backing for the whole closure,
/// and a lifetime covering all accesses. Layout validation alone is not proof.
unsafe fn with_accumulator<T: Copy, R>(
    workspace: Option<WorkspaceView>,
    len: usize,
    execute: impl FnOnce(&mut [MaybeUninit<T>]) -> Result<R>,
) -> Result<R> {
    let region = AllocationLayout::array::<T>(len).map_err(|_| {
        EpError::KernelFailed(format!(
            "Einsum cannot address {len} accumulator elements of {} bytes; reduce the output shape",
            std::mem::size_of::<T>()
        ))
    })?;
    let [view] = WorkspaceLayout::new([region])?.bind(workspace)?;
    let pointer = if len == 0 {
        std::ptr::NonNull::<MaybeUninit<T>>::dangling().as_ptr()
    } else {
        view.ptr().0.cast::<MaybeUninit<T>>()
    };
    // SAFETY: caller supplies exclusive live backing; checked layout covers len.
    execute(unsafe { std::slice::from_raw_parts_mut(pointer, len) })
}

unsafe fn initialized<T>(storage: &[MaybeUninit<T>]) -> &[T] {
    // SAFETY: caller proves every element has been initialized before reading.
    unsafe { std::slice::from_raw_parts(storage.as_ptr().cast(), storage.len()) }
}

unsafe fn with_zeroed_accumulator<T: Default + Copy, R>(
    workspace: Option<WorkspaceView>,
    len: usize,
    execute: impl FnOnce(&mut [T]) -> Result<R>,
) -> Result<R> {
    // SAFETY: forwards the caller's backing/exclusivity proof.
    unsafe {
        with_accumulator(workspace, len, |storage: &mut [MaybeUninit<T>]| {
            for element in storage.iter_mut() {
                element.write(T::default());
            }
            // Every element now contains a valid initialized T.
            execute(std::slice::from_raw_parts_mut(
                storage.as_mut_ptr().cast(),
                storage.len(),
            ))
        })
    }
}

fn resize_scratch<T: Default>(buffer: &mut Vec<T>, len: usize) -> Result<()> {
    if len > buffer.len() {
        buffer
            .try_reserve_exact(len - buffer.len())
            .map_err(|error| {
                EpError::KernelFailed(format!(
                    "Einsum could not reserve {} bytes of typed temporary workspace: {error}",
                    len.saturating_mul(std::mem::size_of::<T>())
                ))
            })?;
    }
    buffer.resize_with(len, T::default);
    Ok(())
}

fn views_may_overlap(input: &TensorView, output: &TensorMut) -> bool {
    fn byte_range(
        base: usize,
        byte_offset: usize,
        shape: &[usize],
        strides: &[i64],
        element_size: usize,
    ) -> Option<(i128, i128)> {
        if checked_numel("alias-check", shape).ok()? == 0 {
            return Some((0, 0));
        }
        let (minimum, maximum) = crate::strided::addressed_elem_range(shape, strides);
        let origin = i128::try_from(base)
            .ok()?
            .checked_add(i128::try_from(byte_offset).ok()?)?;
        let element_size = i128::try_from(element_size).ok()?;
        let start = origin.checked_add(i128::from(minimum).checked_mul(element_size)?)?;
        let end = origin
            .checked_add(i128::from(maximum).checked_mul(element_size)?)?
            .checked_add(element_size)?;
        Some((start, end))
    }

    let element_size = input.dtype.byte_size();
    if element_size == 0 || output.dtype.byte_size() != element_size {
        return true;
    }
    let Some(input_range) = byte_range(
        input.data.0 as usize,
        input.byte_offset,
        input.shape,
        input.strides,
        element_size,
    ) else {
        return true;
    };
    let Some(output_range) = byte_range(
        output.data.0 as usize,
        output.byte_offset,
        output.shape,
        output.strides,
        element_size,
    ) else {
        return true;
    };
    input_range.0 < output_range.1 && output_range.0 < input_range.1
}

/// Execution workspace attributed by the most recent successful dispatch.
#[doc(hidden)]
pub fn benchmark_last_workspace_bytes(kernel: &dyn Kernel) -> Option<usize> {
    let kernel = kernel.as_any().downcast_ref::<EinsumKernel>()?;
    Some(kernel.last_workspace_bytes.load(Ordering::Relaxed))
}

/// Execute one validation dispatch and return the native branch that fired.
///
/// The timed benchmark calls [`Kernel::execute_with_workspace`]. This probe uses
/// the same dispatcher once before timing so route evidence is observed rather
/// than inferred from the equation or expected layout.
#[doc(hidden)]
pub fn benchmark_execute_route(
    kernel: &dyn Kernel,
    inputs: &[TensorView],
    outputs: &mut [TensorMut],
    workspace: Option<WorkspaceView>,
) -> Result<Option<&'static str>> {
    let Some(kernel) = kernel.as_any().downcast_ref::<EinsumKernel>() else {
        return Ok(None);
    };
    kernel
        .execute_with_route(inputs, outputs, workspace)
        .map(|route| Some(route.label()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::{Owned, PreparedTestKernel};
    use onnx_runtime_ep_api::abi::OrtGraphView;
    use onnx_runtime_ep_api::{DevicePtrMut, ExecutionProvider};
    use onnx_runtime_ir::{Attribute, DeviceId, Dim, FrozenGraph, Graph, NodeId, static_shape};

    fn kernel_with_retention(
        equation: &str,
        shapes: &[Vec<usize>],
        mode: ExecutionMode,
        cache_retention: CpuCacheRetention,
    ) -> Box<dyn Kernel> {
        let mut node = Node::new(NodeId(0), "Einsum", vec![], vec![]);
        node.attributes.insert(
            "equation".into(),
            Attribute::String(equation.as_bytes().to_vec()),
        );
        let input_shape_refs: Vec<_> = shapes.iter().map(Vec::as_slice).collect();
        let plan = EinsumShapePlan::build(equation, &input_shape_refs).unwrap();
        Box::new(
            EinsumKernel::new(
                plan,
                MatMulKernel::with_cache_retention(cache_retention.is_admitted()),
                mode,
                &input_shape_refs,
                None,
            )
            .unwrap(),
        )
    }

    fn kernel(equation: &str, shapes: &[Vec<usize>], mode: ExecutionMode) -> Box<dyn Kernel> {
        kernel_with_retention(equation, shapes, mode, CpuCacheRetention::default())
    }

    fn kernel_for_opset(
        equation: &str,
        shapes: &[Vec<usize>],
        opset: i64,
        mode: EinsumExecutionMode,
    ) -> Box<dyn Kernel> {
        let mut node = Node::new(NodeId(0), "Einsum", vec![], vec![]);
        node.version = Some(opset);
        node.attributes.insert(
            "equation".into(),
            Attribute::String(equation.as_bytes().to_vec()),
        );
        EinsumFactory::with_execution_mode(CpuCacheRetention::default(), mode)
            .create(&node, shapes)
            .unwrap()
    }

    fn route(kernel: &dyn Kernel) -> u8 {
        kernel
            .as_any()
            .downcast_ref::<EinsumKernel>()
            .unwrap()
            .last_route
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    #[test]
    fn workspace_bounds_follow_accumulator_width_and_compiled_route() {
        let reduction = kernel("ij->i", &[vec![2, 3]], ExecutionMode::Optimized);
        for (dtype, width) in [
            (DataType::Int8, 1),
            (DataType::Int16, 2),
            (DataType::Int32, 4),
            (DataType::Int64, 8),
            (DataType::Float16, 4),
            (DataType::Float32, 4),
            (DataType::Float64, 8),
        ] {
            let request = reduction
                .workspace_requirement(&[TensorMetadata::new(dtype, &[2, 3], true)])
                .unwrap();
            assert_eq!(
                (request.bytes, request.alignment),
                (2 * width as u64, width)
            );
        }
        let view = kernel("ij->ji", &[vec![2, 3]], ExecutionMode::Optimized);
        assert_eq!(
            view.workspace_requirement(&[TensorMetadata::new(DataType::Float32, &[2, 3], true)])
                .unwrap(),
            WorkspaceRequirement::NONE
        );
        let gemm = kernel(
            "ik,kj->ij",
            &[vec![2, 3], vec![3, 4]],
            ExecutionMode::Optimized,
        );
        for (dtype, bytes) in [
            (DataType::Float32, 32),
            (DataType::Float64, 0),
            (DataType::Int64, 0),
        ] {
            let request = gemm
                .workspace_requirement(&[
                    TensorMetadata::new(dtype, &[2, 3], true),
                    TensorMetadata::new(dtype, &[3, 4], true),
                ])
                .unwrap();
            assert_eq!(request.bytes, bytes);
        }
        let tree = kernel(
            "ab,bc,cd->ad",
            &[vec![2, 3], vec![3, 4], vec![4, 5]],
            ExecutionMode::Optimized,
        );
        for dtype in [
            DataType::Int8,
            DataType::Int16,
            DataType::Float32,
            DataType::Float64,
        ] {
            assert_eq!(
                tree.workspace_requirement(&[
                    TensorMetadata::new(dtype, &[2, 3], true),
                    TensorMetadata::new(dtype, &[3, 4], true),
                    TensorMetadata::new(dtype, &[4, 5], true)
                ])
                .unwrap(),
                WorkspaceRequirement::NONE,
                "a compiled tree must not reserve unused generic publication scratch"
            );
        }
        for metadata in [
            TensorMetadata::new(DataType::Float32, &[2, 3], false),
            TensorMetadata::new(DataType::Float32, &[3, 3], true),
        ] {
            let error = reduction
                .workspace_requirement(&[metadata])
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("input 0") && error.contains("compiled"),
                "{error}"
            );
        }
        let large = kernel(
            "i->i",
            &[vec![isize::MAX as usize / 8 + 1]],
            ExecutionMode::GenericNative,
        );
        let error = large
            .workspace_requirement(&[TensorMetadata::new(
                DataType::Float64,
                &[isize::MAX as usize / 8 + 1],
                true,
            )])
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("workspace") && error.contains("overflow"),
            "{error}"
        );
    }

    #[test]
    fn fresh_workspace_initializes_zero_reductions_and_refuses_missing_backing() {
        let reduction = kernel("ij->i", &[vec![2, 3]], ExecutionMode::GenericNative);
        let input = Owned::f64(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let mut output = Owned::f64(&[2], &[73.; 2]);
        let error = reduction
            .execute(&[input.view()], &mut [output.view_mut()])
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("16 bytes") && error.contains("prepare and pass"),
            "{error}"
        );
        assert_eq!(output.to_f64(), [73.; 2]);
        let mut scratch = [MaybeUninit::<u64>::uninit(); 2];
        let workspace = WorkspaceView::new(DevicePtrMut(scratch.as_mut_ptr().cast()), 16);
        reduction
            .execute_with_workspace(&[input.view()], &mut [output.view_mut()], Some(workspace))
            .unwrap();
        assert_eq!(output.to_f64(), [6., 15.]);
        for mode in [ExecutionMode::GenericNative, ExecutionMode::Oracle] {
            let zero = kernel("ij->i", &[vec![2, 0]], mode);
            let empty = Owned::f32(&[2, 0], &[]);
            let mut output = Owned::f32(&[2], &[73.; 2]);
            zero.execute_with_workspace(&[empty.view()], &mut [output.view_mut()], Some(workspace))
                .unwrap();
            assert_eq!(output.to_f32(), [0.; 2]);
        }
        let empty = kernel("ij->i", &[vec![0, 3]], ExecutionMode::GenericNative);
        assert_eq!(
            empty
                .workspace_requirement(&[TensorMetadata::new(DataType::Float64, &[0, 3], true)])
                .unwrap(),
            WorkspaceRequirement::NONE
        );
        empty
            .execute(
                &[Owned::f64(&[0, 3], &[]).view()],
                &mut [Owned::zeros(DataType::Float64, &[0]).view_mut()],
            )
            .unwrap();
    }

    #[test]
    fn workspace_reuse_is_funded_independent_of_weight_retention_and_has_no_warm_issuer_calls() {
        use crate::kernels::testutil::RecordingGovernor;
        use onnx_runtime_memory_api::Tier;
        use onnx_runtime_memory_governor::{
            HolderId, HostWorkspacePool, MemoryGovernor, ProcessMemoryManager,
        };
        use std::sync::Arc;
        let books = Arc::new(RecordingGovernor::new_host(16));
        let manager = ProcessMemoryManager::new().unwrap();
        let pool = HostWorkspacePool::governed(&manager, books.clone()).unwrap();
        let providers = [true, false].map(|admitted| {
            let resources = crate::CpuMemoryResources::new(
                books.weight_memory(HolderId::new(13)),
                Default::default(),
            )
            .unwrap()
            .with_cache_retention(CpuCacheRetention::new(admitted))
            .with_workspace_pool(pool.clone())
            .unwrap();
            crate::CpuExecutionProvider::initialized_with_resources(resources).unwrap()
        });
        let mut node = Node::new(NodeId(0), "Einsum", vec![], vec![]);
        node.attributes
            .insert("equation".into(), Attribute::String(b"ij->i".to_vec()));
        let kernels = providers
            .each_ref()
            .map(|provider| provider.get_kernel(&node, &[vec![2, 3]], 12).unwrap());
        let first = kernels[0]
            .workspace_requirement(&[TensorMetadata::new(DataType::Float64, &[2, 3], true)])
            .unwrap();
        let mut allocation = providers[0]
            .allocate_workspace(16, first.alignment, first.role)
            .unwrap();
        let identity = allocation.managed_owner().unwrap().identity();
        assert!(
            providers[1].allocate_workspace(8, 4, first.role).is_err(),
            "a second active lane must not execute without its own grant"
        );
        let input64 = Owned::f64(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let input32 = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let mut output64 = Owned::zeros(DataType::Float64, &[2]);
        let mut output32 = Owned::zeros_f32(&[2]);
        let workspace = Some(WorkspaceView::new(
            DevicePtrMut(allocation.as_mut_ptr()),
            allocation.len(),
        ));
        kernels[0]
            .execute_with_workspace(&[input64.view()], &mut [output64.view_mut()], workspace)
            .unwrap();
        providers[0].deallocate_workspace(allocation).unwrap();
        let calls = (books.claims(), books.releases());
        for iteration in 0..4096 {
            let (provider, kernel) = (&providers[iteration % 2], &kernels[iteration % 2]);
            let (input, output) = if iteration % 2 == 0 {
                (&input64, &mut output64)
            } else {
                (&input32, &mut output32)
            };
            let request = kernel
                .workspace_requirement(&[TensorMetadata::new(input.dtype, &input.shape, true)])
                .unwrap();
            let mut allocation = provider
                .allocate_workspace(request.bytes as usize, request.alignment, request.role)
                .unwrap();
            assert_eq!(allocation.managed_owner().unwrap().identity(), identity);
            let workspace = Some(WorkspaceView::new(
                DevicePtrMut(allocation.as_mut_ptr()),
                allocation.len(),
            ));
            kernel
                .execute_with_workspace(&[input.view()], &mut [output.view_mut()], workspace)
                .unwrap();
            provider.deallocate_workspace(allocation).unwrap();
        }
        assert_eq!(output64.to_f64(), [6., 15.]);
        assert_eq!(output32.to_f32(), [6., 15.]);
        assert_eq!((books.claims(), books.releases()), calls);
        assert_eq!(books.used(Tier::Host), 16);
        assert_eq!(manager.process_used(Tier::Host), 16);
        assert_eq!(manager.snapshot().unwrap().allocations.len(), 1);
        drop(kernels);
        drop(providers);
        pool.trim_idle(16).unwrap();
        assert_eq!(books.used(Tier::Host), 0);
    }

    #[test]
    fn compiled_kernel_is_naturally_sync_without_an_unsafe_assertion() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<EinsumKernel>();
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {index}: got {actual}, expected {expected}, tolerance {tolerance}"
            );
        }
    }

    #[test]
    fn benchmark_route_probe_reports_the_branch_that_executed() {
        let direct = kernel(
            "ik,kj->ij",
            &[vec![2, 3], vec![3, 2]],
            ExecutionMode::Optimized,
        );
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[3, 2], &[1., 0., 0., 1., 1., 1.]);
        let mut output = Owned::zeros_f32(&[2, 2]);
        let route = benchmark_execute_route(
            &*direct,
            &[a.view(), b.view()],
            &mut [output.view_mut()],
            None,
        )
        .unwrap();
        assert_eq!(route, Some("matmul-direct"));
        assert_eq!(output.to_f32(), [4., 5., 10., 11.]);
    }

    #[test]
    fn permutation_and_diagonal_are_zero_copy_views() {
        let permutation = kernel("abc->bca", &[vec![2, 3, 4]], ExecutionMode::Optimized);
        let input = Owned::f32(&[2, 3, 4], &(0..24).map(|x| x as f32).collect::<Vec<_>>());
        let specs = permutation
            .view_outputs(&[input.view()], &[vec![3, 4, 2]], 1)
            .expect("permutation must be a view");
        assert_eq!(specs[0].shape, [3, 4, 2]);
        assert_eq!(specs[0].strides, [4, 1, 12]);

        let diagonal = kernel("ii->i", &[vec![3, 3]], ExecutionMode::Optimized);
        let matrix = Owned::f32(&[3, 3], &(0..9).map(|x| x as f32).collect::<Vec<_>>());
        let specs = diagonal
            .view_outputs(&[matrix.view()], &[vec![3]], 1)
            .expect("diagonal must be a view");
        assert_eq!(specs[0].strides, [4]);

        let scalar = kernel("->", &[vec![]], ExecutionMode::Optimized);
        let value = Owned::f32(&[], &[7.5]);
        let specs = scalar
            .view_outputs(&[value.view()], &[vec![]], 1)
            .expect("rank-0 identity must be a view");
        assert!(specs[0].shape.is_empty());
        assert!(specs[0].strides.is_empty());
    }

    #[test]
    fn reduction_elementwise_and_outer_product_follow_plan_mappings() {
        let reduce = kernel("ij->i", &[vec![2, 3]], ExecutionMode::Optimized);
        let x = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let mut out = Owned::zeros_f32(&[2]);
        reduce
            .execute_prepared(&[x.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), [6., 15.]);
        assert_eq!(route(&*reduce), 2);

        let outer = kernel("i,j->ij", &[vec![2], vec![3]], ExecutionMode::Optimized);
        let left = Owned::f32(&[2], &[2., 3.]);
        let right = Owned::f32(&[3], &[5., 7., 11.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        outer
            .execute_prepared(&[left.view(), right.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), [10., 14., 22., 15., 21., 33.]);
    }

    #[test]
    fn gemm_and_transpose_required_bmm_use_matmul_lowering() {
        let gemm = kernel(
            "ik,kj->ij",
            &[vec![2, 3], vec![3, 2]],
            ExecutionMode::Optimized,
        );
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[3, 2], &[1., 0., 0., 1., 1., 0.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        gemm.execute_prepared(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), [4., 2., 10., 5.]);
        assert_eq!(route(&*gemm), 6);

        let bmm = kernel(
            "bik,bjk->bij",
            &[vec![2, 2, 3], vec![2, 4, 3]],
            ExecutionMode::Optimized,
        );
        let a = Owned::f32(
            &[2, 2, 3],
            &[1., 2., 3., 4., 5., 6., 1., 0., 1., 2., 1., 0.],
        );
        let b = Owned::f32(
            &[2, 4, 3],
            &[
                1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 1., 1., 2., 3., 3., 2., 1., 2., 0., 1.,
                0., 1., 2.,
            ],
        );
        let mut out = Owned::zeros_f32(&[2, 2, 4]);
        bmm.execute_prepared(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(
            out.to_f32(),
            [
                1., 2., 3., 6., 4., 5., 6., 15., 4., 4., 3., 2., 4., 8., 4., 1.
            ]
        );
        assert_eq!(route(&*bmm), 6);
    }

    #[test]
    fn broadcast_bmm_rank_one_and_zero_dimensions_are_supported() {
        let bmm = kernel(
            "mk,...kn->...mn",
            &[vec![2, 3], vec![2, 3, 2]],
            ExecutionMode::Optimized,
        );
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(
            &[2, 3, 2],
            &[1., 0., 0., 1., 1., 0., 2., 1., 1., 0., 0., 2.],
        );
        let mut out = Owned::zeros_f32(&[2, 2, 2]);
        bmm.execute_prepared(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), [4., 2., 10., 5., 4., 7., 13., 16.]);

        let dot = kernel("i,i->", &[vec![3], vec![3]], ExecutionMode::Optimized);
        let x = Owned::f32(&[3], &[1., 2., 3.]);
        let y = Owned::f32(&[3], &[4., 5., 6.]);
        let mut scalar = Owned::zeros_f32(&[]);
        dot.execute_prepared(&[x.view(), y.view()], &mut [scalar.view_mut()])
            .unwrap();
        assert_eq!(scalar.to_f32(), [32.]);

        let zero = kernel(
            "ik,kj->ij",
            &[vec![0, 3], vec![3, 4]],
            ExecutionMode::Optimized,
        );
        let a = Owned::f32(&[0, 3], &[]);
        let b = Owned::f32(&[3, 4], &[1.; 12]);
        let mut out = Owned::zeros_f32(&[0, 4]);
        zero.execute_prepared(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert!(out.to_f32().is_empty());
    }

    #[test]
    fn multi_axis_output_permutation_and_diagonal_contraction_are_correct() {
        let multi = kernel(
            "abxy,xycd->dcab",
            &[vec![2, 1, 2, 2], vec![2, 2, 2, 2]],
            ExecutionMode::Optimized,
        );
        let left = Owned::f32(&[2, 1, 2, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let right = Owned::f32(
            &[2, 2, 2, 2],
            &[
                1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ],
        );
        let mut out = Owned::zeros_f32(&[2, 2, 2, 1]);
        multi
            .execute_prepared(&[left.view(), right.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(
            out.to_f32(),
            [90., 202., 110., 254., 100., 228., 120., 280.]
        );
        assert_eq!(route(&*multi), 7);

        let diagonal = kernel(
            "iik,kj->ij",
            &[vec![2, 2, 3], vec![3, 2]],
            ExecutionMode::Optimized,
        );
        let left = Owned::f32(
            &[2, 2, 3],
            &[1., 2., 3., 99., 99., 99., 99., 99., 99., 4., 5., 6.],
        );
        let right = Owned::f32(&[3, 2], &[1., 0., 0., 1., 1., 1.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        diagonal
            .execute_prepared(&[left.view(), right.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), [4., 5., 10., 11.]);

        // The output permutation forces the materialized fallback while the
        // left operand has an inserted broadcast batch axis. Its materialized
        // shape must retain batch=1 rather than pretending it owns the resolved
        // batch=3 storage.
        let equation = "abxy,...xycd->d...cab";
        let shapes = [vec![2, 1, 2, 2], vec![3, 2, 2, 2, 2]];
        let broadcasted = kernel(equation, &shapes, ExecutionMode::Optimized);
        let oracle = kernel(equation, &shapes, ExecutionMode::Oracle);
        let left = Owned::f32(&shapes[0], &[1., 2., 3., 4., 5., 6., 7., 8.]);
        let right = Owned::f32(
            &shapes[1],
            &(1..=48).map(|value| value as f32 / 8.0).collect::<Vec<_>>(),
        );
        let mut actual = Owned::zeros_f32(&[2, 3, 2, 2, 1]);
        let mut expected = Owned::zeros_f32(&[2, 3, 2, 2, 1]);
        broadcasted
            .execute_prepared(&[left.view(), right.view()], &mut [actual.view_mut()])
            .unwrap();
        oracle
            .execute_prepared(&[left.view(), right.view()], &mut [expected.view_mut()])
            .unwrap();
        assert_close(&actual.to_f32(), &expected.to_f32(), 1e-5);
        assert_eq!(route(&*broadcasted), 7);
    }

    #[test]
    fn float16_and_noncontiguous_inputs_match_expected_values() {
        let gemm = kernel(
            "ik,kj->ij",
            &[vec![2, 3], vec![3, 2]],
            ExecutionMode::Optimized,
        );
        let a = Owned::f16(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f16(&[3, 2], &[1., 0., 0., 1., 1., 0.]);
        let mut out = Owned::zeros(DataType::Float16, &[2, 2]);
        gemm.execute_prepared(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_close(&out.to_f16_as_f32(), &[4., 2., 10., 5.], 0.05);

        let reduce = kernel("ij->i", &[vec![2, 3]], ExecutionMode::Optimized);
        let mut reduced = Owned::zeros(DataType::Float16, &[2]);
        reduce
            .execute_prepared(&[a.view()], &mut [reduced.view_mut()])
            .unwrap();
        assert_close(&reduced.to_f16_as_f32(), &[6., 15.], 0.05);

        let transpose = kernel("ij->ji", &[vec![2, 3]], ExecutionMode::Optimized);
        let mut transposed = Owned::zeros(DataType::Float16, &[3, 2]);
        transpose
            .execute_prepared(&[a.view()], &mut [transposed.view_mut()])
            .unwrap();
        assert_eq!(
            transposed.to_u16_bits(),
            [
                a.to_u16_bits()[0],
                a.to_u16_bits()[3],
                a.to_u16_bits()[1],
                a.to_u16_bits()[4],
                a.to_u16_bits()[2],
                a.to_u16_bits()[5],
            ],
            "view/copy semantics must preserve Float16 payload bits exactly"
        );

        let noncontiguous = kernel(
            "ik,kj->ij",
            &[vec![2, 3], vec![3, 2]],
            ExecutionMode::Optimized,
        );
        let a = Owned::f32(&[3, 2], &[1., 4., 2., 5., 3., 6.]).with_view(&[2, 3], &[1, 2]);
        let b = Owned::f32(&[3, 2], &[1., 0., 0., 1., 1., 0.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        noncontiguous
            .execute_prepared(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), [4., 2., 10., 5.]);
    }

    #[test]
    fn compute_fallback_is_alias_safe() {
        let permutation_kernel = kernel("ij->ji", &[vec![2, 2]], ExecutionMode::Optimized);
        let mut tensor = Owned::f32(&[2, 2], &[1., 2., 3., 4.]);
        let shape = tensor.shape.clone();
        let strides = tensor.strides.clone();
        let ptr = tensor.bytes.as_mut_ptr();
        let input = TensorView::new(
            onnx_runtime_ep_api::DevicePtr(ptr.cast_const().cast()),
            DataType::Float32,
            &shape,
            &strides,
            DeviceId::cpu(),
        );
        let output_shape = [2usize, 2];
        let output_strides = [2i64, 1];
        let mut output = TensorMut::new(
            DevicePtrMut(ptr.cast()),
            DataType::Float32,
            &output_shape,
            &output_strides,
            DeviceId::cpu(),
        );
        permutation_kernel
            .execute_prepared(&[input], std::slice::from_mut(&mut output))
            .unwrap();
        tensor.shape = shape;
        tensor.strides = strides;
        assert_eq!(tensor.to_f32(), [1., 3., 2., 4.]);

        let gemm = kernel(
            "ik,kj->ij",
            &[vec![2, 2], vec![2, 2]],
            ExecutionMode::Optimized,
        );
        let mut left = Owned::f32(&[2, 2], &[1., 2., 3., 4.]);
        let right = Owned::f32(&[2, 2], &[2., 0., 0., 3.]);
        let left_shape = left.shape.clone();
        let left_strides = left.strides.clone();
        let left_ptr = left.bytes.as_mut_ptr();
        let left_input = TensorView::new(
            onnx_runtime_ep_api::DevicePtr(left_ptr.cast_const().cast()),
            DataType::Float32,
            &left_shape,
            &left_strides,
            DeviceId::cpu(),
        );
        let mut aliased_output = TensorMut::new(
            DevicePtrMut(left_ptr.cast()),
            DataType::Float32,
            &left_shape,
            &left_strides,
            DeviceId::cpu(),
        );
        gemm.execute_prepared(
            &[left_input, right.view()],
            std::slice::from_mut(&mut aliased_output),
        )
        .unwrap();
        assert_eq!(left.to_f32(), [2., 6., 6., 12.]);
        assert_eq!(route(&*gemm), 7);
    }

    #[test]
    fn generic_native_is_alias_safe_and_reads_negative_strides() {
        let generic = kernel("ij->ji", &[vec![2, 2]], EinsumExecutionMode::GenericNative);
        let mut tensor = Owned::f32(&[2, 2], &[1., 2., 3., 4.]);
        let shape = tensor.shape.clone();
        let strides = tensor.strides.clone();
        let pointer = tensor.bytes.as_mut_ptr();
        let input = TensorView::new(
            onnx_runtime_ep_api::DevicePtr(pointer.cast_const().cast()),
            DataType::Float32,
            &shape,
            &strides,
            DeviceId::cpu(),
        );
        let mut output = TensorMut::new(
            DevicePtrMut(pointer.cast()),
            DataType::Float32,
            &shape,
            &strides,
            DeviceId::cpu(),
        );
        generic
            .execute_prepared(&[input], std::slice::from_mut(&mut output))
            .unwrap();
        assert_eq!(tensor.to_f32(), [1., 3., 2., 4.]);
        assert_eq!(route(&*generic), 3);

        let reverse = kernel("i->i", &[vec![3]], EinsumExecutionMode::GenericNative);
        let storage = Owned::f32(&[3], &[1., 2., 3.]);
        let reverse_shape = [3usize];
        let reverse_strides = [-1i64];
        let reversed = TensorView::new(
            onnx_runtime_ep_api::DevicePtr(storage.bytes.as_ptr().cast()),
            DataType::Float32,
            &reverse_shape,
            &reverse_strides,
            DeviceId::cpu(),
        )
        .with_byte_offset(2 * std::mem::size_of::<f32>());
        let mut output = Owned::zeros_f32(&[3]);
        reverse
            .execute_prepared(&[reversed], &mut [output.view_mut()])
            .unwrap();
        assert_eq!(output.to_f32(), [3., 2., 1.]);
    }

    #[test]
    fn empty_integer_reduction_is_zero_and_validation_failure_is_transactional() {
        let reduction = kernel("i->", &[vec![0]], EinsumExecutionMode::GenericNative);
        let input = Owned::i32(&[0], &[]);
        let mut output = Owned::zeros(DataType::Int32, &[]);
        reduction
            .execute_prepared(&[input.view()], &mut [output.view_mut()])
            .unwrap();
        assert_eq!(output.to_i32(), [0]);

        let identity = kernel("i->i", &[vec![2]], EinsumExecutionMode::GenericNative);
        let input = Owned::i32(&[2], &[7, 11]);
        let mut output = Owned::f32(&[2], &[123.0, 456.0]);
        let before = output.bytes.clone();
        let error = identity
            .execute_prepared(&[input.view()], &mut [output.view_mut()])
            .unwrap_err()
            .to_string();
        assert!(error.contains("output dtype"), "{error}");
        assert_eq!(
            output.bytes, before,
            "validation errors must not partially modify the output"
        );
    }

    #[test]
    fn oracle_mode_is_high_precision_and_non_vacuously_selected() {
        let optimized = kernel(
            "ik,kj->ij",
            &[vec![2, 3], vec![3, 2]],
            ExecutionMode::Optimized,
        );
        let oracle_kernel = kernel(
            "ik,kj->ij",
            &[vec![2, 3], vec![3, 2]],
            ExecutionMode::Oracle,
        );
        let a = Owned::f32(&[2, 3], &[1e10, 1., -1e10, 3., 4., 5.]);
        let b = Owned::f32(&[3, 2], &[1., 2., 1., 1., 1., 0.]);
        let mut fast = Owned::zeros_f32(&[2, 2]);
        let mut oracle = Owned::zeros_f32(&[2, 2]);
        optimized
            .execute_prepared(&[a.view(), b.view()], &mut [fast.view_mut()])
            .unwrap();
        oracle_kernel
            .execute_prepared(&[a.view(), b.view()], &mut [oracle.view_mut()])
            .unwrap();
        assert_eq!(route(&*optimized), 6);
        assert_eq!(route(&*oracle_kernel), 9);
        assert_eq!(oracle.to_f32(), [1., 2e10, 12., 10.]);
        assert_close(&fast.to_f32(), &oracle.to_f32(), 1024.0);
    }

    fn einsum_graph_with_equation(
        dtype: DataType,
        equation: &str,
        output_shape: &[usize],
    ) -> FrozenGraph {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 24);
        let left = graph.create_named_value("A", dtype, static_shape([2, 3]));
        let right = graph.create_named_value("B", dtype, static_shape([3, 2]));
        let output =
            graph.create_named_value("C", dtype, static_shape(output_shape.iter().copied()));
        graph.add_input(left);
        graph.add_input(right);
        let mut node = Node::new(
            NodeId(0),
            "Einsum",
            vec![Some(left), Some(right)],
            vec![output],
        );
        node.attributes.insert(
            "equation".into(),
            Attribute::String(equation.as_bytes().to_vec()),
        );
        graph.insert_node(node);
        graph.add_output(output);
        FrozenGraph::build(graph).unwrap()
    }

    fn einsum_graph(dtype: DataType) -> FrozenGraph {
        einsum_graph_with_equation(dtype, "ik,kj->ij", &[2, 2])
    }

    #[test]
    fn provider_placement_declines_bfloat16_and_reaches_float16_float32() {
        let provider = crate::CpuExecutionProvider::new();

        for dtype in [DataType::Float32, DataType::Float16] {
            let frozen = einsum_graph(dtype);
            let view = frozen.view();
            let node_index = view.nodes().next().expect("one Einsum node");
            let support = provider.supports_node(&view, node_index, 24);
            assert!(
                support.is_supported(),
                "{dtype:?} Einsum must be reachable through normal provider placement: {support:?}"
            );
            let claims = OrtGraphView::new(&view)
                .query_capabilities(&provider)
                .unwrap();
            assert_eq!(
                claims.len(),
                1,
                "{dtype:?} Einsum must produce one non-vacuous provider capability"
            );
            let kernel = provider
                .get_kernel(view.node(node_index), &[vec![2, 3], vec![3, 2]], 24)
                .unwrap();
            let (left, right) = match dtype {
                DataType::Float32 => (
                    Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]),
                    Owned::f32(&[3, 2], &[1., 0., 0., 1., 1., 0.]),
                ),
                DataType::Float16 => (
                    Owned::f16(&[2, 3], &[1., 2., 3., 4., 5., 6.]),
                    Owned::f16(&[3, 2], &[1., 0., 0., 1., 1., 0.]),
                ),
                _ => unreachable!(),
            };
            let mut output = Owned::zeros(dtype, &[2, 2]);
            kernel
                .execute_prepared(&[left.view(), right.view()], &mut [output.view_mut()])
                .unwrap();
            let actual = match dtype {
                DataType::Float32 => output.to_f32(),
                DataType::Float16 => output.to_f16_as_f32(),
                _ => unreachable!(),
            };
            assert_close(&actual, &[4., 2., 10., 5.], 0.05);
        }

        let frozen = einsum_graph(DataType::BFloat16);
        let view = frozen.view();
        let node_index = view.nodes().next().expect("one Einsum node");
        let support = provider.supports_node(&view, node_index, 24);
        assert!(!support.is_supported());
        let reason = support.reason().expect("BFloat16 decline must explain why");
        assert!(
            reason.contains("dtype BFloat16, which is not admitted by Einsum-12"),
            "{reason}"
        );
        assert!(reason.contains("import ai.onnx opset 28+"), "{reason}");
        assert!(reason.contains("HOW:"), "{reason}");
        assert!(
            OrtGraphView::new(&view)
                .query_capabilities(&provider)
                .unwrap()
                .is_empty(),
            "BFloat16 Einsum must not reach compilation through provider placement"
        );
    }

    #[test]
    fn mixed_case_implicit_equation_reaches_cpu_placement_and_executes() {
        let provider = crate::CpuExecutionProvider::new();
        let frozen = einsum_graph_with_equation(DataType::Float32, "Za,aB", &[2, 2]);
        let view = frozen.view();
        let node_index = view.nodes().next().expect("one Einsum node");
        let support = provider.supports_node(&view, node_index, 24);
        assert!(
            support.is_supported(),
            "mixed-case Einsum must be claimable: {support:?}"
        );
        assert_eq!(
            OrtGraphView::new(&view)
                .query_capabilities(&provider)
                .unwrap()
                .len(),
            1
        );
        let kernel = provider
            .get_kernel(view.node(node_index), &[vec![2, 3], vec![3, 2]], 24)
            .unwrap();
        let left = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let right = Owned::f32(&[3, 2], &[1., 0., 0., 1., 1., 0.]);
        let mut output = Owned::zeros_f32(&[2, 2]);
        kernel
            .execute_prepared(&[left.view(), right.view()], &mut [output.view_mut()])
            .unwrap();
        assert_eq!(
            output.to_f32(),
            [4., 10., 2., 5.],
            "implicit ASCII output order is [B, Z], the transpose of [Z, B]"
        );
    }

    #[test]
    fn general_contractions_and_planner_fallback_execute_natively() {
        let mut node = Node::new(NodeId(0), "Einsum", vec![], vec![]);
        node.attributes.insert(
            "equation".into(),
            Attribute::String(b"ij,jk,kl->il".to_vec()),
        );
        let shapes = [
            static_shape([2, 3]),
            static_shape([3, 2]),
            static_shape([2, 2]),
        ];
        let dtypes = [DataType::Float32; 3];
        assert!(unsupported_reason_for_opset(&node, 12, &shapes, &dtypes).is_none());

        let provider = crate::CpuExecutionProvider::new();
        let support = provider.supports_op(
            &node,
            12,
            &shapes,
            &dtypes,
            &[
                onnx_runtime_ir::TensorLayout::contiguous(),
                onnx_runtime_ir::TensorLayout::contiguous(),
                onnx_runtime_ir::TensorLayout::contiguous(),
            ],
        );
        assert!(support.is_supported(), "{:?}", support.reason());
        let contraction = provider
            .get_kernel(&node, &[vec![2, 3], vec![3, 2], vec![2, 2]], 12)
            .expect("general contraction compiles");
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let b = Owned::f32(&[3, 2], &[1., 0., 0., 1., 1., 1.]);
        let c = Owned::f32(&[2, 2], &[2., 0., 0., 3.]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        contraction
            .execute_prepared(&[a.view(), b.view(), c.view()], &mut [out.view_mut()])
            .expect("general contraction executes");
        assert_eq!(out.to_f32(), [8., 15., 20., 33.]);
        assert_eq!(route(&*contraction), 4);

        let mut mixed = Node::new(NodeId(1), "Einsum", vec![], vec![]);
        mixed
            .attributes
            .insert("equation".into(), Attribute::String(b"aik,kj->ij".to_vec()));
        let mixed_shapes = [static_shape([7, 2, 3]), static_shape([3, 4])];
        let mixed_dtypes = [DataType::Float16; 2];
        assert!(unsupported_reason_for_opset(&mixed, 12, &mixed_shapes, &mixed_dtypes).is_none());
        EinsumFactory::default()
            .create(&mixed, &[vec![7, 2, 3], vec![3, 4]])
            .expect("mixed local reduction compiles");

        let arity = 128;
        let equation = format!(
            "{}->",
            std::iter::repeat_n("i", arity)
                .collect::<Vec<_>>()
                .join(",")
        );
        let mut large = Node::new(NodeId(2), "Einsum", vec![], vec![]);
        large
            .attributes
            .insert("equation".into(), Attribute::String(equation.into_bytes()));
        let shapes = vec![static_shape([1]); arity];
        let dtypes = vec![DataType::Float32; arity];
        assert!(unsupported_reason_for_opset(&large, 12, &shapes, &dtypes).is_none());
        let kernel = EinsumFactory::with_execution_mode(
            CpuCacheRetention::default(),
            EinsumExecutionMode::Optimized,
        )
        .create(&large, &vec![vec![1]; arity])
        .expect("planner fallback compiles");
        let operands = (0..arity)
            .map(|_| Owned::f32(&[1], &[1.0]))
            .collect::<Vec<_>>();
        let views = operands.iter().map(Owned::view).collect::<Vec<_>>();
        let mut output = Owned::zeros_f32(&[]);
        kernel
            .execute_prepared(&views, &mut [output.view_mut()])
            .expect("128-input GenericNative fallback executes");
        assert_eq!(output.to_f32(), [1.0]);
        assert_eq!(route(&*kernel), 3);
    }

    #[test]
    fn contraction_tree_boundary_counts_slots_without_publication_scratch() {
        const ROWS: usize = 4094;
        const COLUMNS: usize = 2048;
        const OUTPUT_BYTES: usize = ROWS * COLUMNS * std::mem::size_of::<f32>();
        const RETAINED_SLOT_BYTES: usize = COLUMNS * std::mem::size_of::<f32>();
        const EXPECTED_WORKSPACE_BYTES: usize = OUTPUT_BYTES + RETAINED_SLOT_BYTES;

        assert_eq!(EXPECTED_WORKSPACE_BYTES, EINSUM_TREE_WORKSPACE_LIMIT - 8192);

        let retention = CpuCacheRetention::default();
        let kernel = kernel_with_retention(
            "ab,bc,cd->ad",
            &[vec![ROWS, 1], vec![1, 1], vec![1, COLUMNS]],
            ExecutionMode::Optimized,
            retention.clone(),
        );
        let left = Owned::f32(&[ROWS, 1], &vec![1.0; ROWS]);
        let middle = Owned::f32(&[1, 1], &[1.0]);
        let right = Owned::f32(&[1, COLUMNS], &vec![1.0; COLUMNS]);
        let mut output = Owned::zeros_f32(&[ROWS, COLUMNS]);
        kernel
            .execute_prepared(
                &[left.view(), middle.view(), right.view()],
                &mut [output.view_mut()],
            )
            .expect("the admitted near-limit contraction tree executes");

        assert_eq!(route(&*kernel), 4);
        assert_eq!(
            benchmark_last_workspace_bytes(&*kernel),
            Some(EXPECTED_WORKSPACE_BYTES),
            "telemetry must count the final accumulator plus the retained temporary-slot capacity"
        );
        assert_eq!(
            kernel
                .workspace_requirement(
                    &[left.view(), middle.view(), right.view()].map(|input| TensorMetadata::new(
                        input.dtype,
                        input.shape,
                        true
                    ))
                )
                .unwrap(),
            WorkspaceRequirement::NONE,
            "successful trees need no second publication accumulator"
        );
        assert_eq!(
            f32::from_ne_bytes(output.bytes[..4].try_into().unwrap()),
            1.0
        );
        assert_eq!(
            f32::from_ne_bytes(output.bytes[output.bytes.len() - 4..].try_into().unwrap()),
            1.0
        );
    }

    #[test]
    fn claim_and_factory_resolve_einsum_schema_before_backend_dtype_support() {
        let mut node = Node::new(NodeId(0), "Einsum", vec![], vec![]);
        node.attributes
            .insert("equation".into(), Attribute::String(b"i->i".to_vec()));
        let shape = [static_shape([2])];

        let opset11 = unsupported_reason_for_opset(&node, 11, &shape, &[DataType::Float32])
            .expect("must reject");
        assert!(opset11.contains("predates Einsum-12"), "{opset11}");

        let opset27 = unsupported_reason_for_opset(&node, 27, &shape, &[DataType::BFloat16])
            .expect("must reject");
        assert!(opset27.contains("not admitted by Einsum-12"), "{opset27}");

        assert!(unsupported_reason_for_opset(&node, 28, &shape, &[DataType::BFloat16]).is_none());

        node.version = Some(28);
        EinsumFactory::default()
            .create(&node, &[vec![2]])
            .expect("shape-only factory accepts an Einsum-28 semantic plan");
    }

    #[test]
    fn runtime_shape_and_dtype_errors_are_actionable() {
        let shape_kernel = kernel(
            "ik,kj->ij",
            &[vec![2, 3], vec![3, 2]],
            ExecutionMode::Optimized,
        );
        let a = Owned::f32(&[2, 3], &[1.; 6]);
        let b = Owned::f32(&[4, 2], &[1.; 8]);
        let mut out = Owned::zeros_f32(&[2, 2]);
        let error = shape_kernel
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("runtime shape validation failed"),
            "unexpected error: {error}"
        );
        assert!(
            error.contains("input #1") || error.contains("label `k`"),
            "error did not identify the rejected operand/axis: {error}"
        );

        let int_shape = vec![Dim::Static(2)];
        let mut node = Node::new(NodeId(0), "Einsum", vec![], vec![]);
        node.attributes
            .insert("equation".into(), Attribute::String(b"i->i".to_vec()));
        assert!(
            unsupported_reason_for_opset(&node, 12, &[int_shape], &[DataType::Int32]).is_none()
        );

        let direct = kernel("i->i", &[vec![2]], ExecutionMode::Optimized);
        let input = Owned::bf16(&[2], &[1.0, 2.0]);
        assert!(
            direct
                .view_outputs(&[input.view()], &[vec![2]], 1)
                .is_none(),
            "BFloat16 must not bypass kernel dtype validation through a view output"
        );
        let mut output = Owned::zeros(DataType::BFloat16, &[2]);
        let error = direct
            .execute(&[input.view()], &mut [output.view_mut()])
            .unwrap_err()
            .to_string();
        assert!(error.contains("not admitted by Einsum-12"), "{error}");
        assert!(error.contains("HOW:"), "{error}");

        let bf16 = kernel_for_opset("i->i", &[vec![2]], 28, EinsumExecutionMode::GenericNative);
        let mut output = Owned::zeros(DataType::BFloat16, &[2]);
        bf16.execute_prepared(&[input.view()], &mut [output.view_mut()])
            .expect("Einsum-28 BFloat16 executes");
        assert_eq!(output.to_u16_bits(), input.to_u16_bits());
    }
}

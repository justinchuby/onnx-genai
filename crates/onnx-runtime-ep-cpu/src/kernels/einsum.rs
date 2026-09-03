//! Native CPU execution for schema-resolved ONNX `Einsum`.
//!
//! The equation is parsed exactly once, by [`EinsumShapePlan`], when the
//! shape-specialized kernel is built. Execution consumes only the plan's
//! structural classification and axis maps.

use std::any::{Any, TypeId};
use std::borrow::Cow;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

use onnx_runtime_ep_api::{
    EpError, Kernel, KernelFactory, Result, TensorMut, TensorView, ViewOutput,
};
use onnx_runtime_ir::{
    DataType, EinsumAxis, EinsumBinaryContractionPlan, EinsumContractionPlan,
    EinsumContractionTreePlan, EinsumContractionTreeStep, EinsumGenericNativePlan, EinsumInput,
    EinsumOperandPlan, EinsumPermutationPlan, EinsumPlan, EinsumPlannerQuality,
    EinsumPlanningClassification, EinsumSchema, EinsumShapePlan,
    EinsumSupportedContractionTreeCandidate, EinsumUnaryReductionPlan, EinsumValueId, Node, Shape,
    compute_contiguous_strides,
};
use rayon::prelude::*;

use super::{check_arity, matmul::MatMulKernel, to_dense_bytes, write_dense_bytes};
use crate::dtype::{
    ComputeDomain, NumericElem, to_dense_f32_widen, write_dense, write_dense_f32_narrow,
};
use crate::kernels::governed_accumulator_budget::{
    DEFAULT_PER_THREAD_ACCUMULATOR_BYTES, DEFAULT_PROCESS_ACCUMULATOR_BYTES, GovernedAccumulator,
    GovernedAccumulatorBudget,
};
use crate::strided::next_index;

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

/// Process-wide byte accounting for typed Einsum scratch retained between
/// calls. Admission is deliberately absent here: each compiled session owns an
/// immutable [`EinsumScratchRetention`] verdict. The only process-global state
/// is the aggregate byte ceiling shared by those independent sessions.
static EINSUM_SCRATCH_BUDGET: GovernedAccumulatorBudget = GovernedAccumulatorBudget::new(
    DEFAULT_PER_THREAD_ACCUMULATOR_BYTES,
    DEFAULT_PROCESS_ACCUMULATOR_BYTES,
);

thread_local! {
    /// One reusable slot per worker thread, never one map entry per session.
    ///
    /// The slot itself is owned by the session retention token; TLS keeps only
    /// weak references. Dropping a session therefore frees inactive-worker
    /// buffers immediately, while switching sessions on one worker unregisters
    /// the previous slot before another owner can park there.
    static EINSUM_SCRATCH: RefCell<Option<EinsumTlsScratch>> =
        const { RefCell::new(None) };
}

struct EinsumScratchSlot {
    accumulator: Mutex<Box<dyn ErasedGovernedAccumulator>>,
}

trait ErasedGovernedAccumulator: Send {
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn capacity_bytes(&self) -> usize;
}

impl<T: Send + 'static> ErasedGovernedAccumulator for GovernedAccumulator<T> {
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn capacity_bytes(&self) -> usize {
        GovernedAccumulator::capacity_bytes(self)
    }
}

#[derive(Default)]
struct EinsumScratchSlots {
    next_registration: u64,
    active: Vec<(u64, Arc<EinsumScratchSlot>)>,
}

struct EinsumScratchRetentionInner {
    admitted: bool,
    budget: &'static GovernedAccumulatorBudget,
    // Every entry owns a non-empty, byte-reserved buffer. TLS has exactly one
    // weak entry and unregisters it on an owner switch or thread exit, so this
    // list is bounded by concurrently parked workers and the process byte cap;
    // it is not an ever-growing per-session thread map.
    slots: Mutex<EinsumScratchSlots>,
}

impl EinsumScratchRetentionInner {
    fn register<T: Send + 'static>(self: &Arc<Self>, buffer: Vec<T>) -> Option<EinsumTlsScratch> {
        let mut accumulator = GovernedAccumulator::new();
        if !accumulator.try_park(buffer, self.budget) {
            return None;
        }
        let slot = Arc::new(EinsumScratchSlot {
            accumulator: Mutex::new(Box::new(accumulator)),
        });
        let registration = {
            let mut slots = self
                .slots
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let registration = slots.next_registration;
            slots.next_registration = slots.next_registration.checked_add(1)?;
            slots.active.push((registration, Arc::clone(&slot)));
            registration
        };
        Some(EinsumTlsScratch {
            owner: Arc::downgrade(self),
            slot: Arc::downgrade(&slot),
            registration,
            element_type: TypeId::of::<T>(),
        })
    }

    fn unregister(&self, registration: u64) {
        let mut slots = self
            .slots
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(index) = slots
            .active
            .iter()
            .position(|(candidate, _)| *candidate == registration)
        {
            slots.active.swap_remove(index);
        }
    }

    #[cfg(test)]
    fn active_slots(&self) -> usize {
        self.slots
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .active
            .len()
    }
}

struct EinsumTlsScratch {
    owner: Weak<EinsumScratchRetentionInner>,
    slot: Weak<EinsumScratchSlot>,
    registration: u64,
    element_type: TypeId,
}

impl EinsumTlsScratch {
    fn belongs_to(&self, owner: &Arc<EinsumScratchRetentionInner>) -> bool {
        self.owner.ptr_eq(&Arc::downgrade(owner))
    }

    fn stores<T: 'static>(&self) -> bool {
        self.element_type == TypeId::of::<T>()
    }
}

impl Drop for EinsumTlsScratch {
    fn drop(&mut self) {
        if let Some(owner) = self.owner.upgrade() {
            owner.unregister(self.registration);
        }
    }
}

/// Immutable session-owned decision for retaining CPU Einsum scratch.
///
/// A CPU provider clones this handle into its Einsum factory and every compiled
/// kernel. Different providers therefore cannot overwrite one another's memory
/// strategy. The handle owns every parked slot strongly, while TLS keeps weak
/// references; dropping the last provider/session/kernel handle immediately
/// releases inactive-worker allocations and their process-budget reservations.
#[derive(Clone)]
pub struct EinsumScratchRetention {
    inner: Arc<EinsumScratchRetentionInner>,
}

impl std::fmt::Debug for EinsumScratchRetention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EinsumScratchRetention")
            .field("admitted", &self.inner.admitted)
            .finish_non_exhaustive()
    }
}

impl Default for EinsumScratchRetention {
    fn default() -> Self {
        Self::new(true)
    }
}

impl EinsumScratchRetention {
    /// Create one immutable retention verdict for one provider/session.
    pub fn new(admitted: bool) -> Self {
        Self::with_budget(admitted, &EINSUM_SCRATCH_BUDGET)
    }

    fn with_budget(admitted: bool, budget: &'static GovernedAccumulatorBudget) -> Self {
        Self {
            inner: Arc::new(EinsumScratchRetentionInner {
                admitted,
                budget,
                slots: Mutex::new(EinsumScratchSlots::default()),
            }),
        }
    }

    /// The memory plan verdict captured by this handle.
    pub fn is_admitted(&self) -> bool {
        self.inner.admitted
    }

    fn take<T: Send + 'static>(&self) -> Result<Vec<T>> {
        EINSUM_SCRATCH
            .try_with(|scratch| {
                let mut scratch = scratch.try_borrow_mut().map_err(|_| {
                    EpError::KernelFailed(
                        "Einsum: the per-thread scratch pool is already being checked out. \
                         HOW: do not hold the scratch-pool slot across kernel execution."
                            .into(),
                    )
                })?;
                if !self.inner.admitted {
                    scratch.take();
                    return Ok(Vec::new());
                }
                if !scratch
                    .as_ref()
                    .is_some_and(|entry| entry.belongs_to(&self.inner) && entry.stores::<T>())
                {
                    *scratch = None;
                }
                let Some(entry) = scratch.as_ref() else {
                    return Ok(Vec::new());
                };
                let Some(slot) = entry.slot.upgrade() else {
                    *scratch = None;
                    return Ok(Vec::new());
                };
                let mut accumulator = slot
                    .accumulator
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let accumulator = accumulator
                    .as_any_mut()
                    .downcast_mut::<GovernedAccumulator<T>>()
                    .ok_or_else(|| {
                        EpError::KernelFailed(
                            "Einsum: the typed scratch slot changed type while checked out".into(),
                        )
                    })?;
                Ok(accumulator.take())
            })
            .map_err(|_| {
                EpError::KernelFailed(
                    "Einsum: the per-thread scratch pool is unavailable during thread teardown"
                        .into(),
                )
            })?
    }

    fn park<T: Send + 'static>(&self, buffer: Vec<T>) {
        if !self.inner.admitted {
            let _ = EINSUM_SCRATCH.try_with(|scratch| {
                if let Ok(mut scratch) = scratch.try_borrow_mut() {
                    scratch.take();
                }
            });
            return;
        }
        let mut buffer = Some(buffer);
        let _ = EINSUM_SCRATCH.try_with(|scratch| {
            let Ok(mut scratch) = scratch.try_borrow_mut() else {
                return;
            };
            if !scratch
                .as_ref()
                .is_some_and(|entry| entry.belongs_to(&self.inner) && entry.stores::<T>())
            {
                *scratch = None;
            }
            if let Some(slot) = scratch.as_ref().and_then(|entry| entry.slot.upgrade()) {
                let mut erased = slot
                    .accumulator
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                let parked = erased
                    .as_any_mut()
                    .downcast_mut::<GovernedAccumulator<T>>()
                    .is_some_and(|accumulator| {
                        accumulator.try_park(
                            buffer
                                .take()
                                .expect("Einsum scratch buffer is parked at most once"),
                            self.inner.budget,
                        )
                    });
                if !parked {
                    *scratch = None;
                }
                return;
            }
            *scratch = None;
            if let Some(entry) = self.inner.register(
                buffer
                    .take()
                    .expect("Einsum scratch buffer is registered at most once"),
            ) {
                *scratch = Some(entry);
            }
        });
    }

    fn with_scratch<T: Default + Send + 'static, R>(
        &self,
        len: usize,
        execute: impl FnOnce(&mut Vec<T>) -> Result<R>,
    ) -> Result<R> {
        let mut buffer = self.take()?;
        resize_scratch(&mut buffer, len)?;
        let value = execute(&mut buffer)?;
        self.park(buffer);
        Ok(value)
    }

    fn with_f32_scratch<R>(
        &self,
        len: usize,
        execute: impl FnOnce(&mut Vec<f32>) -> Result<R>,
    ) -> Result<R> {
        self.with_scratch(len, execute)
    }

    fn current_thread_capacity_bytes(&self) -> usize {
        EINSUM_SCRATCH
            .try_with(|scratch| {
                let scratch = scratch.try_borrow().ok()?;
                let entry = scratch.as_ref()?;
                if !entry.belongs_to(&self.inner) {
                    return None;
                }
                let slot = entry.slot.upgrade()?;
                let bytes = slot
                    .accumulator
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .capacity_bytes();
                Some(bytes)
            })
            .ok()
            .flatten()
            .unwrap_or(0)
    }

    #[cfg(test)]
    fn active_slots(&self) -> usize {
        self.inner.active_slots()
    }
}

const CONCURRENCY_ROUTES: usize = 4;
const CONCURRENCY_VIEW: usize = 0;
const CONCURRENCY_REDUCTION: usize = 1;
const CONCURRENCY_MATERIALIZED_GEMM: usize = 2;
const CONCURRENCY_GENERIC: usize = 3;
static CONCURRENCY_PROBE_ENABLED: AtomicBool = AtomicBool::new(false);
static CONCURRENCY_ACTIVE: [AtomicUsize; CONCURRENCY_ROUTES] =
    [const { AtomicUsize::new(0) }; CONCURRENCY_ROUTES];
static CONCURRENCY_MAX: [AtomicUsize; CONCURRENCY_ROUTES] =
    [const { AtomicUsize::new(0) }; CONCURRENCY_ROUTES];

struct ConcurrencyProbeGuard {
    route: usize,
}

impl ConcurrencyProbeGuard {
    fn enter(route: usize) -> Option<Self> {
        if !CONCURRENCY_PROBE_ENABLED.load(Ordering::Relaxed) {
            return None;
        }
        let active = CONCURRENCY_ACTIVE[route].fetch_add(1, Ordering::AcqRel) + 1;
        CONCURRENCY_MAX[route].fetch_max(active, Ordering::Relaxed);
        Some(Self { route })
    }
}

impl Drop for ConcurrencyProbeGuard {
    fn drop(&mut self) {
        CONCURRENCY_ACTIVE[self.route].fetch_sub(1, Ordering::AcqRel);
    }
}

#[doc(hidden)]
pub fn reset_concurrency_probe() {
    for route in 0..CONCURRENCY_ROUTES {
        CONCURRENCY_ACTIVE[route].store(0, Ordering::Relaxed);
        CONCURRENCY_MAX[route].store(0, Ordering::Relaxed);
    }
    CONCURRENCY_PROBE_ENABLED.store(true, Ordering::Release);
}

#[doc(hidden)]
pub fn finish_concurrency_probe() -> [usize; CONCURRENCY_ROUTES] {
    CONCURRENCY_PROBE_ENABLED.store(false, Ordering::Release);
    std::array::from_fn(|route| CONCURRENCY_MAX[route].load(Ordering::Acquire))
}

/// Bytes currently parked between Einsum calls across all execution threads.
pub fn einsum_scratch_live_bytes() -> u64 {
    EINSUM_SCRATCH_BUDGET.live_bytes()
}

/// Current hard process-wide ceiling for retained Einsum scratch.
pub fn einsum_scratch_process_cap_bytes() -> u64 {
    EINSUM_SCRATCH_BUDGET.process_cap_bytes()
}

/// Predict the resident scratch budget to declare for a graph.
///
/// All Einsum kernels share one process-wide pool, so the prediction is one
/// process cap when the graph contains any default-domain `Einsum`, not one cap
/// per node. Reading the configured cap keeps admission aligned with the bytes
/// the runtime may actually retain.
pub fn einsum_scratch_budget_predicted_bytes(graph: &onnx_runtime_ir::Graph) -> u64 {
    let has_einsum = graph
        .nodes
        .values()
        .any(|node| node.is_default_domain() && node.op_type == "Einsum");
    if has_einsum {
        einsum_scratch_process_cap_bytes()
    } else {
        0
    }
}

/// Shape-specialized CPU Einsum kernel.
pub struct EinsumKernel {
    plan: EinsumShapePlan,
    matmul: MatMulKernel,
    scratch_retention: EinsumScratchRetention,
    mode: EinsumExecutionMode,
    flops: Option<u64>,
    last_workspace_bytes: AtomicUsize,
    #[cfg(test)]
    last_route: std::sync::atomic::AtomicU8,
}

/// Factory for [`EinsumKernel`].
pub struct EinsumFactory {
    scratch_retention: EinsumScratchRetention,
    mode: Option<EinsumExecutionMode>,
}

impl EinsumFactory {
    pub fn new(scratch_retention: EinsumScratchRetention) -> Self {
        Self {
            scratch_retention,
            mode: None,
        }
    }

    /// Construct a factory with an explicit immutable execution policy.
    ///
    /// This avoids process-global environment mutation in conformance and
    /// benchmark adapters while keeping the ordinary provider configuration
    /// compatible with [`EINSUM_MODE_ENV`].
    pub fn with_execution_mode(
        scratch_retention: EinsumScratchRetention,
        mode: EinsumExecutionMode,
    ) -> Self {
        Self {
            scratch_retention,
            mode: Some(mode),
        }
    }
}

impl Default for EinsumFactory {
    fn default() -> Self {
        Self::new(EinsumScratchRetention::default())
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
        Ok(Box::new(EinsumKernel {
            plan,
            matmul: MatMulKernel::default(),
            scratch_retention: self.scratch_retention.clone(),
            mode,
            flops,
            last_workspace_bytes: AtomicUsize::new(0),
            #[cfg(test)]
            last_route: std::sync::atomic::AtomicU8::new(0),
        }))
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

/// Claim-time capability check using the original Einsum-12 contract.
///
/// This compatibility wrapper intentionally does not inspect node metadata or
/// infer a schema from the operand dtypes. Model/provider paths that have an
/// effective opset must call [`unsupported_reason_for_opset`] instead.
pub fn unsupported_reason(
    node: &Node,
    shapes: &[Shape],
    input_dtypes: &[DataType],
) -> Option<String> {
    unsupported_reason_for_opset(node, 12, shapes, input_dtypes)
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
        self.execute_with_route(inputs, outputs).map(|_| ())
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
    fn with_scratch<T: Default + Send + 'static, R>(
        &self,
        len: usize,
        execute: impl FnOnce(&mut Vec<T>) -> Result<R>,
    ) -> Result<R> {
        self.scratch_retention.with_scratch(len, execute)
    }

    fn with_f32_scratch<R>(
        &self,
        len: usize,
        execute: impl FnOnce(&mut Vec<f32>) -> Result<R>,
    ) -> Result<R> {
        self.scratch_retention.with_f32_scratch(len, execute)
    }

    fn execute_with_route(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
    ) -> Result<EinsumRoute> {
        self.last_workspace_bytes.store(0, Ordering::Relaxed);
        self.validate_execution(inputs, outputs)?;
        if self.mode == EinsumExecutionMode::GenericNative {
            let _probe = ConcurrencyProbeGuard::enter(CONCURRENCY_GENERIC);
            self.execute_generic_native(inputs, outputs)?;
            return Ok(self.finish_route(EinsumRoute::GenericNative));
        }
        if self.mode == EinsumExecutionMode::Oracle {
            let _probe = ConcurrencyProbeGuard::enter(CONCURRENCY_GENERIC);
            if matches!(inputs[0].dtype, DataType::Float32 | DataType::Float16) {
                self.execute_oracle(inputs, outputs)?;
            } else {
                self.execute_generic_native(inputs, outputs)?;
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
                self.execute_generic_native(inputs, outputs)?;
                Ok(self.finish_route(EinsumRoute::Reduction))
            }
            EinsumPlanningClassification::Gemm(gemm)
                if matches!(
                    inputs[0].dtype,
                    DataType::Float16 | DataType::BFloat16 | DataType::Float32
                ) =>
            {
                let route = self.execute_gemm(inputs, outputs, gemm)?;
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
                let route = self.execute_contraction_tree_or_generic(inputs, outputs, tree)?;
                Ok(self.finish_route(route))
            }
            _ => {
                let _probe = ConcurrencyProbeGuard::enter(CONCURRENCY_GENERIC);
                self.execute_generic_native(inputs, outputs)?;
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

    fn execute_oracle(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
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
    ) -> Result<()> {
        crate::dispatch_arith!(inputs[0].dtype, "Einsum", T => {
            self.execute_generic_native_typed::<T>(
                inputs,
                outputs,
                self.plan.generic_native(),
            )
        })
    }

    fn execute_generic_native_typed<T: EinsumElement>(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
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

        self.with_scratch::<<T as NumericElem>::Acc, _>(output_len, |accumulators| {
            let evaluate = |first_tile: usize, output: &mut [<T as NumericElem>::Acc]| {
                let first_output = first_tile * tile;
                let mut coordinates = vec![0usize; iteration_shape.len()];
                for (local, destination) in output.iter_mut().enumerate() {
                    let output_linear = first_output + local;
                    decode_row_major(output_linear, output_shape, &mut coordinates[..output_rank]);
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
                    *destination = sum;
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
            T::write_accumulators(&mut outputs[0], accumulators)
        })
    }

    fn execute_contraction_tree_or_generic(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        tree: &EinsumContractionTreePlan,
    ) -> Result<EinsumRoute> {
        if tree.quality() == EinsumPlannerQuality::GenericNativeFallback {
            self.execute_generic_native(inputs, outputs)?;
            return Ok(EinsumRoute::GenericNative);
        }
        let input_shapes = inputs.iter().map(|input| input.shape).collect::<Vec<_>>();
        let accumulator_size = accumulator_element_size(inputs[0].dtype);
        let output_bytes = checked_numel("contraction-tree output", outputs[0].shape)?
            .checked_mul(accumulator_size)
            .ok_or_else(|| {
                geometry_overflow(self.plan.equation(), "contraction-tree output bytes")
            })?;
        let temporary_ceiling =
            u128::from(DEFAULT_PER_THREAD_ACCUMULATOR_BYTES).saturating_sub(output_bytes as u128);
        let concrete = self
            .plan
            .resolve_concrete_contraction_tree(&input_shapes, accumulator_size)
            .map_err(|error| {
                EpError::KernelFailed(format!(
                    "Einsum `{}` could not resolve its concrete contraction tree: {error}",
                    self.plan.equation()
                ))
            })?;
        let Some(selected) = concrete
            .as_ref()
            .and_then(|tree| tree.preferred_candidate_with_memory_ceiling(temporary_ceiling))
        else {
            self.execute_generic_native(inputs, outputs)?;
            return Ok(EinsumRoute::GenericNative);
        };
        let peak_temporary_bytes = selected
            .cost()
            .map(|cost| cost.peak_live_temporary_bytes())
            .unwrap_or(0);
        let workspace_bytes = peak_temporary_bytes
            .checked_add(output_bytes as u128)
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| {
                geometry_overflow(self.plan.equation(), "contraction-tree workspace bytes")
            })?;
        self.last_workspace_bytes
            .store(workspace_bytes, Ordering::Relaxed);
        let candidate = tree
            .candidates()
            .iter()
            .find(|candidate| candidate.id() == selected.id())
            .and_then(|candidate| candidate.supported())
            .ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "Einsum `{}` selected contraction candidate `{}` without a matching \
                     supported structural plan",
                    self.plan.equation(),
                    selected.id()
                ))
            })?;
        crate::dispatch_arith!(inputs[0].dtype, "Einsum", T => {
            self.execute_contraction_tree_typed::<T>(inputs, outputs, candidate)
        })?;
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
        let requested_len = checked_numel("scalar GEMM requested output", outputs[0].shape)?;
        let workspace_bytes = canonical_len
            .checked_add(requested_len)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<T::Acc>()))
            .ok_or_else(|| geometry_overflow(self.plan.equation(), "scalar GEMM workspace"))?;
        self.last_workspace_bytes
            .store(workspace_bytes, Ordering::Relaxed);
        self.with_scratch::<T::Acc, _>(requested_len, |requested| {
            permute_tree_output(
                self.plan.equation(),
                &self.plan,
                &final_value,
                gemm.output_permutation(),
                requested,
            )?;
            T::write_accumulators(&mut outputs[0], requested)
        })
    }

    fn execute_contraction_tree_typed<T: EinsumElement>(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        candidate: &EinsumSupportedContractionTreeCandidate,
    ) -> Result<()> {
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
        let output_shape = outputs[0].shape.to_vec();
        let output_len = checked_numel("contraction-tree final output", &output_shape)?;
        self.with_scratch::<T::Acc, _>(output_len, |requested| {
            permute_tree_output(
                self.plan.equation(),
                &self.plan,
                &final_value,
                candidate.final_output_permutation(),
                requested,
            )?;
            T::write_accumulators(&mut outputs[0], requested)
        })
    }

    fn execute_generic(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
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

        self.with_f32_scratch(output_len, |f32_output| {
            f32_output.fill(0.0);
            let identity_mappings = mappings
                .iter()
                .all(|mapping| mapping.iter().copied().eq(0..iteration_axes.len()));
            let aligned_dense = layouts
                .iter()
                .zip(&dense)
                .all(|(layout, data)| layout.shape == iteration_shape && data.len() == output_len);
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
                for (output_offset, output) in f32_output.iter_mut().enumerate().take(output_len) {
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
                                        geometry_overflow(self.plan.equation(), "operand offset")
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

    fn execute_gemm(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
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
        self.with_f32_scratch(canonical_len, |f32_output| {
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
        })?;
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

trait EinsumElement: NumericElem + Send + Sync + 'static
where
    Self::Acc: Send + Sync + 'static,
{
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

fn permute_tree_output<A: Copy>(
    equation: &str,
    plan: &EinsumShapePlan,
    final_value: &DenseTreeValue<A>,
    permutation: &[usize],
    requested: &mut [A],
) -> Result<()> {
    let requested_shape = axes_shape(plan, plan.output_axes())?;
    let requested_len = checked_numel("contraction-tree requested output", &requested_shape)?;
    if requested.len() != requested_len || final_value.data.len() != requested_len {
        return Err(EpError::KernelFailed(format!(
            "Einsum `{equation}` contraction final element count mismatch: canonical={}, \
             requested={}, output={requested_len}",
            final_value.data.len(),
            requested.len()
        )));
    }
    if requested_len == 0 {
        return Ok(());
    }
    if final_value.axes == plan.output_axes()
        && (permutation.is_empty() || permutation.iter().copied().eq(0..permutation.len()))
    {
        requested.copy_from_slice(&final_value.data);
        return Ok(());
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
    let canonical_strides = compute_contiguous_strides(&final_value.shape);
    let mut requested_index = vec![0usize; requested_shape.len()];
    for (linear, destination) in requested.iter_mut().enumerate() {
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
        *destination = *final_value.data.get(canonical_offset).ok_or_else(|| {
            EpError::KernelFailed(format!(
                "Einsum `{equation}` contraction final offset {canonical_offset} exceeds {} \
                 canonical element(s)",
                final_value.data.len()
            ))
        })?;
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

/// Current reusable typed scratch capacity held by an Einsum kernel.
///
/// Used by the benchmark to report steady-state workspace rather than infer it
/// from tensor shapes.
#[doc(hidden)]
pub fn benchmark_scratch_capacity_bytes(kernel: &dyn Kernel) -> Option<usize> {
    let kernel = kernel.as_any().downcast_ref::<EinsumKernel>()?;
    Some(kernel.scratch_retention.current_thread_capacity_bytes())
}

/// Execution workspace attributed by the most recent successful dispatch.
#[doc(hidden)]
pub fn benchmark_last_workspace_bytes(kernel: &dyn Kernel) -> Option<usize> {
    let kernel = kernel.as_any().downcast_ref::<EinsumKernel>()?;
    Some(kernel.last_workspace_bytes.load(Ordering::Relaxed))
}

/// Execute one validation dispatch and return the native branch that fired.
///
/// The timed benchmark continues to call [`Kernel::execute`]. This probe uses
/// the same dispatcher once before timing so route evidence is observed rather
/// than inferred from the equation or expected layout.
#[doc(hidden)]
pub fn benchmark_execute_route(
    kernel: &dyn Kernel,
    inputs: &[TensorView],
    outputs: &mut [TensorMut],
) -> Result<Option<&'static str>> {
    let Some(kernel) = kernel.as_any().downcast_ref::<EinsumKernel>() else {
        return Ok(None);
    };
    kernel
        .execute_with_route(inputs, outputs)
        .map(|route| Some(route.label()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ep_api::abi::OrtGraphView;
    use onnx_runtime_ep_api::{DevicePtrMut, ExecutionProvider};
    use onnx_runtime_ir::{Attribute, DeviceId, Dim, FrozenGraph, Graph, NodeId, static_shape};
    use std::sync::{Arc, Barrier, Mutex, MutexGuard};

    static SCRATCH_TEST_LOCK: Mutex<()> = Mutex::new(());
    static TEST_SCRATCH_BUDGET: GovernedAccumulatorBudget =
        GovernedAccumulatorBudget::new(128, 256);

    struct ScratchTestGuard {
        _guard: MutexGuard<'static, ()>,
    }

    impl Drop for ScratchTestGuard {
        fn drop(&mut self) {
            EINSUM_SCRATCH.with(|scratch| {
                scratch.borrow_mut().take();
            });
            TEST_SCRATCH_BUDGET.set_caps_for_test(128, 256);
            TEST_SCRATCH_BUDGET.reset_for_test();
        }
    }

    fn begin_scratch_test(per_thread_cap_bytes: u64, process_cap_bytes: u64) -> ScratchTestGuard {
        let guard = SCRATCH_TEST_LOCK
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        EINSUM_SCRATCH.with(|scratch| {
            scratch.borrow_mut().take();
        });
        TEST_SCRATCH_BUDGET.reset_for_test();
        TEST_SCRATCH_BUDGET.set_caps_for_test(per_thread_cap_bytes, process_cap_bytes);
        ScratchTestGuard { _guard: guard }
    }

    fn test_retention(admitted: bool) -> EinsumScratchRetention {
        EinsumScratchRetention::with_budget(admitted, &TEST_SCRATCH_BUDGET)
    }

    fn with_test_scratch<T>(
        retention: &EinsumScratchRetention,
        len: usize,
        execute: impl FnOnce(&mut Vec<f32>) -> Result<T>,
    ) -> Result<T> {
        retention.with_f32_scratch(len, execute)
    }

    fn kernel_with_retention(
        equation: &str,
        shapes: &[Vec<usize>],
        mode: ExecutionMode,
        scratch_retention: EinsumScratchRetention,
    ) -> Box<dyn Kernel> {
        let mut node = Node::new(NodeId(0), "Einsum", vec![], vec![]);
        node.attributes.insert(
            "equation".into(),
            Attribute::String(equation.as_bytes().to_vec()),
        );
        let input_shape_refs: Vec<_> = shapes.iter().map(Vec::as_slice).collect();
        let plan = EinsumShapePlan::build(equation, &input_shape_refs).unwrap();
        Box::new(EinsumKernel {
            flops: None,
            plan,
            matmul: MatMulKernel::default(),
            scratch_retention,
            mode,
            last_workspace_bytes: AtomicUsize::new(0),
            last_route: std::sync::atomic::AtomicU8::new(0),
        })
    }

    fn kernel(equation: &str, shapes: &[Vec<usize>], mode: ExecutionMode) -> Box<dyn Kernel> {
        kernel_with_retention(equation, shapes, mode, EinsumScratchRetention::default())
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
        EinsumFactory::with_execution_mode(EinsumScratchRetention::default(), mode)
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
    fn compiled_kernel_is_naturally_sync_without_an_unsafe_assertion() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<EinsumKernel>();
    }

    #[test]
    fn scratch_retention_is_bounded_by_the_process_aggregate_ceiling() {
        let _guard = begin_scratch_test(64, 128);
        const THREADS: usize = 4;
        let parked = Arc::new(Barrier::new(THREADS + 1));
        let release = Arc::new(Barrier::new(THREADS + 1));

        let workers = (0..THREADS)
            .map(|_| {
                let parked = Arc::clone(&parked);
                let release = Arc::clone(&release);
                std::thread::spawn(move || {
                    let retention = test_retention(true);
                    with_test_scratch(&retention, 16, |buffer| {
                        buffer.fill(1.0);
                        Ok(())
                    })
                    .expect("scratch call succeeds");
                    parked.wait();
                    release.wait();
                })
            })
            .collect::<Vec<_>>();

        parked.wait();
        assert_eq!(
            TEST_SCRATCH_BUDGET.live_bytes(),
            128,
            "two 64-byte buffers fit and the other workers must be declined"
        );
        release.wait();
        for worker in workers {
            worker.join().expect("scratch worker exits cleanly");
        }

        assert_eq!(
            TEST_SCRATCH_BUDGET.live_bytes(),
            0,
            "worker thread exit must release every parked reservation"
        );
    }

    #[test]
    fn scratch_reuses_an_admitted_allocation_on_the_same_thread() {
        let _guard = begin_scratch_test(128, 128);
        let retention = test_retention(true);
        let first = with_test_scratch(&retention, 16, |buffer| {
            Ok((buffer.as_ptr() as usize, buffer.capacity()))
        })
        .expect("first scratch call succeeds");
        let retained = TEST_SCRATCH_BUDGET.live_bytes();
        assert_eq!(retained, (first.1 * std::mem::size_of::<f32>()) as u64);
        assert_eq!(retention.current_thread_capacity_bytes(), retained as usize);

        let second = with_test_scratch(&retention, 16, |buffer| {
            Ok((buffer.as_ptr() as usize, buffer.capacity()))
        })
        .expect("second scratch call succeeds");
        assert_eq!(
            second, first,
            "the parked allocation must be checked out again"
        );
        assert_eq!(TEST_SCRATCH_BUDGET.live_bytes(), retained);
    }

    #[test]
    fn typed_scratch_switches_width_without_cross_type_reuse_or_accounting_leak() {
        let _guard = begin_scratch_test(128, 128);
        let retention = test_retention(true);
        retention
            .with_scratch::<f32, _>(16, |buffer| {
                buffer.fill(3.0);
                Ok(())
            })
            .unwrap();
        assert_eq!(TEST_SCRATCH_BUDGET.live_bytes(), 64);

        retention
            .with_scratch::<f64, _>(8, |buffer| {
                assert!(buffer.iter().all(|&value| value == 0.0));
                buffer.fill(5.0);
                Ok(())
            })
            .unwrap();
        assert_eq!(
            TEST_SCRATCH_BUDGET.live_bytes(),
            64,
            "changing accumulator dtype must release the old reservation before parking the new one"
        );
        assert_eq!(retention.active_slots(), 1);
    }

    #[test]
    fn oversized_and_declined_scratch_are_discarded_after_use() {
        let _guard = begin_scratch_test(32, 128);
        let admitted = test_retention(true);
        with_test_scratch(&admitted, 9, |buffer| {
            buffer.fill(2.0);
            Ok(())
        })
        .expect("oversized temporary scratch still computes");
        assert_eq!(TEST_SCRATCH_BUDGET.live_bytes(), 0);
        assert_eq!(
            admitted.current_thread_capacity_bytes(),
            0,
            "a buffer over the per-thread cap must not remain in TLS"
        );

        with_test_scratch(&admitted, 8, |_| Ok(())).expect("prime an admitted buffer");
        assert_eq!(TEST_SCRATCH_BUDGET.live_bytes(), 32);
        let declined = test_retention(false);
        with_test_scratch(&declined, 8, |buffer| {
            buffer.fill(3.0);
            Ok(())
        })
        .expect("declined retention still permits temporary scratch");
        assert_eq!(TEST_SCRATCH_BUDGET.live_bytes(), 0);
        assert_eq!(
            declined.current_thread_capacity_bytes(),
            0,
            "a declined buffer must be dropped rather than parked"
        );
        assert_eq!(
            admitted.active_slots(),
            0,
            "switching owners on one worker must unregister the old session slot"
        );
    }

    #[test]
    fn scratch_error_unwind_and_checkout_failure_leave_exact_accounting() {
        let _guard = begin_scratch_test(128, 128);
        let retention = test_retention(true);
        with_test_scratch(&retention, 16, |_| Ok(())).expect("prime retained scratch");
        assert_eq!(TEST_SCRATCH_BUDGET.live_bytes(), 64);

        let error = with_test_scratch(&retention, 16, |_buffer| -> Result<()> {
            Err(EpError::KernelFailed("injected scratch error".into()))
        })
        .unwrap_err();
        assert!(error.to_string().contains("injected scratch error"));
        assert_eq!(TEST_SCRATCH_BUDGET.live_bytes(), 0);
        assert_eq!(retention.current_thread_capacity_bytes(), 0);

        with_test_scratch(&retention, 16, |_| Ok(())).expect("prime scratch before unwind");
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = with_test_scratch(&retention, 16, |_buffer| -> Result<()> {
                panic!("injected scratch unwind")
            });
        }));
        assert!(unwind.is_err());
        assert_eq!(TEST_SCRATCH_BUDGET.live_bytes(), 0);
        assert_eq!(retention.current_thread_capacity_bytes(), 0);

        EINSUM_SCRATCH.with(|scratch| {
            let _borrow = scratch.borrow_mut();
            let error = with_test_scratch(&retention, 1, |_| Ok(())).unwrap_err();
            assert!(error.to_string().contains("already being checked out"));
        });
        assert_eq!(TEST_SCRATCH_BUDGET.live_bytes(), 0);
    }

    #[test]
    fn scratch_thread_exit_releases_its_reservation() {
        let _guard = begin_scratch_test(64, 64);
        let (parked_tx, parked_rx) = std::sync::mpsc::channel();
        let release = Arc::new(Barrier::new(2));
        let worker_release = Arc::clone(&release);
        let worker = std::thread::spawn(move || {
            let retention = test_retention(true);
            with_test_scratch(&retention, 16, |_| Ok(())).expect("worker parks scratch");
            parked_tx
                .send(TEST_SCRATCH_BUDGET.live_bytes())
                .expect("report parked bytes");
            worker_release.wait();
        });

        assert_eq!(parked_rx.recv().expect("worker reported"), 64);
        assert_eq!(TEST_SCRATCH_BUDGET.live_bytes(), 64);
        release.wait();
        worker.join().expect("scratch worker exits cleanly");
        assert_eq!(
            TEST_SCRATCH_BUDGET.live_bytes(),
            0,
            "TLS destruction must release the worker's reservation"
        );
    }

    #[test]
    fn zero_size_scratch_retains_and_accounts_nothing() {
        let _guard = begin_scratch_test(0, 0);
        let retention = test_retention(true);
        let len = with_test_scratch(&retention, 0, |buffer| Ok(buffer.len()))
            .expect("zero-sized scratch succeeds");
        assert_eq!(len, 0);
        assert_eq!(TEST_SCRATCH_BUDGET.live_bytes(), 0);
        assert_eq!(retention.current_thread_capacity_bytes(), 0);
    }

    #[test]
    fn zero_output_generic_execution_allocates_no_typed_scratch() {
        let _guard = begin_scratch_test(128, 128);
        let retention = test_retention(true);
        let kernel = kernel_with_retention(
            "i->i",
            &[vec![0]],
            EinsumExecutionMode::GenericNative,
            retention.clone(),
        );
        let input = Owned::f32(&[0], &[]);
        let mut output = Owned::zeros_f32(&[0]);
        kernel
            .execute(&[input.view()], &mut [output.view_mut()])
            .unwrap();
        assert_eq!(TEST_SCRATCH_BUDGET.live_bytes(), 0);
        assert_eq!(retention.current_thread_capacity_bytes(), 0);
    }

    #[test]
    fn opposite_session_verdicts_are_order_independent_on_one_worker() {
        let _guard = begin_scratch_test(64, 128);

        for admitted_created_first in [true, false] {
            EINSUM_SCRATCH.with(|scratch| {
                scratch.borrow_mut().take();
            });
            TEST_SCRATCH_BUDGET.reset_for_test();

            let first = test_retention(admitted_created_first);
            let second = test_retention(!admitted_created_first);
            let (admitted, declined) = if admitted_created_first {
                (&first, &second)
            } else {
                (&second, &first)
            };

            with_test_scratch(admitted, 16, |buffer| {
                buffer.fill(2.0);
                assert_eq!(buffer.iter().sum::<f32>(), 32.0);
                Ok(())
            })
            .expect("admitted session executes");
            assert_eq!(TEST_SCRATCH_BUDGET.live_bytes(), 64);
            assert_eq!(admitted.active_slots(), 1);

            with_test_scratch(declined, 16, |buffer| {
                buffer.fill(3.0);
                assert_eq!(buffer.iter().sum::<f32>(), 48.0);
                Ok(())
            })
            .expect("declined session executes without retaining");
            assert_eq!(TEST_SCRATCH_BUDGET.live_bytes(), 0);
            assert_eq!(admitted.active_slots(), 0);
            assert_eq!(declined.active_slots(), 0);

            with_test_scratch(admitted, 16, |buffer| {
                buffer.fill(4.0);
                assert_eq!(buffer.iter().sum::<f32>(), 64.0);
                Ok(())
            })
            .expect("the admitted session keeps its own verdict");
            assert_eq!(TEST_SCRATCH_BUDGET.live_bytes(), 64);
            assert_eq!(admitted.active_slots(), 1);
        }
    }

    #[test]
    fn admitted_sessions_do_not_cross_use_one_workers_slot() {
        let _guard = begin_scratch_test(128, 128);
        let first = test_retention(true);
        let second = test_retention(true);

        with_test_scratch(&first, 16, |buffer| {
            buffer.fill(11.0);
            Ok(())
        })
        .unwrap();
        assert_eq!(first.active_slots(), 1);
        assert_eq!(TEST_SCRATCH_BUDGET.live_bytes(), 64);

        with_test_scratch(&second, 8, |buffer| {
            assert!(
                buffer.iter().all(|&value| value == 0.0),
                "a new session must not receive another session's parked values"
            );
            buffer.fill(13.0);
            Ok(())
        })
        .unwrap();
        assert_eq!(first.active_slots(), 0);
        assert_eq!(second.active_slots(), 1);
        assert_eq!(
            TEST_SCRATCH_BUDGET.live_bytes(),
            32,
            "one worker retains at most its current session's buffer"
        );
    }

    #[test]
    fn admitted_and_declined_sessions_run_concurrently_without_sharing_retention() {
        let _guard = begin_scratch_test(64, 128);
        let admitted = test_retention(true);
        let declined = test_retention(false);
        let entered = Arc::new(Barrier::new(3));
        let release = Arc::new(Barrier::new(3));

        let admitted_worker = {
            let retention = admitted.clone();
            let entered = Arc::clone(&entered);
            let release = Arc::clone(&release);
            std::thread::spawn(move || {
                let sum = with_test_scratch(&retention, 16, |buffer| {
                    buffer.fill(5.0);
                    Ok(buffer.iter().sum::<f32>())
                })
                .expect("admitted session executes");
                assert_eq!(sum, 80.0);
                entered.wait();
                release.wait();
            })
        };
        let declined_worker = {
            let retention = declined.clone();
            let entered = Arc::clone(&entered);
            let release = Arc::clone(&release);
            std::thread::spawn(move || {
                let sum = with_test_scratch(&retention, 16, |buffer| {
                    buffer.fill(7.0);
                    Ok(buffer.iter().sum::<f32>())
                })
                .expect("declined session executes");
                assert_eq!(sum, 112.0);
                entered.wait();
                release.wait();
            })
        };

        entered.wait();
        assert_eq!(
            TEST_SCRATCH_BUDGET.live_bytes(),
            64,
            "only the admitted session may contribute retained bytes"
        );
        assert_eq!(admitted.active_slots(), 1);
        assert_eq!(declined.active_slots(), 0);
        release.wait();
        admitted_worker.join().unwrap();
        declined_worker.join().unwrap();
        assert_eq!(TEST_SCRATCH_BUDGET.live_bytes(), 0);
    }

    #[test]
    fn dropping_a_session_reclaims_inactive_worker_tls() {
        let _guard = begin_scratch_test(64, 64);
        let retention = test_retention(true);
        let mut node = Node::new(NodeId(0), "Einsum", vec![], vec![]);
        node.attributes
            .insert("equation".into(), Attribute::String(b"ij->i".to_vec()));
        let provider = crate::CpuExecutionProvider::with_einsum_scratch_retention(retention);
        let kernel = provider
            .get_kernel(&node, &[vec![2, 4]], 12)
            .expect("session compiles Einsum");
        drop(provider);

        let (parked_tx, parked_rx) = std::sync::mpsc::channel();
        let drop_kernel = Arc::new(Barrier::new(2));
        let keep_worker_alive = Arc::new(Barrier::new(2));
        let worker_drop_kernel = Arc::clone(&drop_kernel);
        let worker_keep_alive = Arc::clone(&keep_worker_alive);
        let worker = std::thread::spawn(move || {
            let input = Owned::f32(&[2, 4], &[1., 2., 3., 4., 5., 6., 7., 8.]);
            let mut output = Owned::zeros_f32(&[2]);
            kernel
                .execute(&[input.view()], &mut [output.view_mut()])
                .expect("worker executes compiled session kernel");
            assert_eq!(output.to_f32(), [10., 26.]);
            parked_tx
                .send(("parked", TEST_SCRATCH_BUDGET.live_bytes()))
                .expect("report parked scratch");
            worker_drop_kernel.wait();
            drop(kernel);
            parked_tx
                .send(("dropped", TEST_SCRATCH_BUDGET.live_bytes()))
                .expect("report dropped session");
            worker_keep_alive.wait();
        });

        assert_eq!(parked_rx.recv().expect("worker reported"), ("parked", 8));
        assert_eq!(TEST_SCRATCH_BUDGET.live_bytes(), 8);
        drop_kernel.wait();
        assert_eq!(parked_rx.recv().expect("worker reported"), ("dropped", 0));
        assert_eq!(
            TEST_SCRATCH_BUDGET.live_bytes(),
            0,
            "dropping the provider and its last compiled kernel must release \
             scratch while the worker and its TLS entry remain alive"
        );

        keep_worker_alive.wait();
        worker.join().expect("inactive worker exits cleanly");
        assert_eq!(TEST_SCRATCH_BUDGET.live_bytes(), 0);
    }

    #[test]
    fn scratch_prediction_is_one_configured_process_pool_per_graph() {
        let empty = Graph::new();
        assert_eq!(einsum_scratch_budget_predicted_bytes(&empty), 0);

        let mut graph = Graph::new();
        graph.insert_node(Node::new(NodeId(0), "Einsum", vec![], vec![]));
        assert_eq!(
            einsum_scratch_budget_predicted_bytes(&graph),
            einsum_scratch_process_cap_bytes(),
            "node count must not multiply the shared process pool"
        );
        graph.insert_node(Node::new(NodeId(1), "Einsum", vec![], vec![]));
        assert_eq!(
            einsum_scratch_budget_predicted_bytes(&graph),
            einsum_scratch_process_cap_bytes()
        );
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
        let route =
            benchmark_execute_route(&*direct, &[a.view(), b.view()], &mut [output.view_mut()])
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
        reduce.execute(&[x.view()], &mut [out.view_mut()]).unwrap();
        assert_eq!(out.to_f32(), [6., 15.]);
        assert_eq!(route(&*reduce), 2);

        let outer = kernel("i,j->ij", &[vec![2], vec![3]], ExecutionMode::Optimized);
        let left = Owned::f32(&[2], &[2., 3.]);
        let right = Owned::f32(&[3], &[5., 7., 11.]);
        let mut out = Owned::zeros_f32(&[2, 3]);
        outer
            .execute(&[left.view(), right.view()], &mut [out.view_mut()])
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
        gemm.execute(&[a.view(), b.view()], &mut [out.view_mut()])
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
        bmm.execute(&[a.view(), b.view()], &mut [out.view_mut()])
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
        bmm.execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_eq!(out.to_f32(), [4., 2., 10., 5., 4., 7., 13., 16.]);

        let dot = kernel("i,i->", &[vec![3], vec![3]], ExecutionMode::Optimized);
        let x = Owned::f32(&[3], &[1., 2., 3.]);
        let y = Owned::f32(&[3], &[4., 5., 6.]);
        let mut scalar = Owned::zeros_f32(&[]);
        dot.execute(&[x.view(), y.view()], &mut [scalar.view_mut()])
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
        zero.execute(&[a.view(), b.view()], &mut [out.view_mut()])
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
            .execute(&[left.view(), right.view()], &mut [out.view_mut()])
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
            .execute(&[left.view(), right.view()], &mut [out.view_mut()])
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
            .execute(&[left.view(), right.view()], &mut [actual.view_mut()])
            .unwrap();
        oracle
            .execute(&[left.view(), right.view()], &mut [expected.view_mut()])
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
        gemm.execute(&[a.view(), b.view()], &mut [out.view_mut()])
            .unwrap();
        assert_close(&out.to_f16_as_f32(), &[4., 2., 10., 5.], 0.05);

        let reduce = kernel("ij->i", &[vec![2, 3]], ExecutionMode::Optimized);
        let mut reduced = Owned::zeros(DataType::Float16, &[2]);
        reduce
            .execute(&[a.view()], &mut [reduced.view_mut()])
            .unwrap();
        assert_close(&reduced.to_f16_as_f32(), &[6., 15.], 0.05);

        let transpose = kernel("ij->ji", &[vec![2, 3]], ExecutionMode::Optimized);
        let mut transposed = Owned::zeros(DataType::Float16, &[3, 2]);
        transpose
            .execute(&[a.view()], &mut [transposed.view_mut()])
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
            .execute(&[a.view(), b.view()], &mut [out.view_mut()])
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
            .execute(&[input], std::slice::from_mut(&mut output))
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
        gemm.execute(
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
            .execute(&[input], std::slice::from_mut(&mut output))
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
            .execute(&[reversed], &mut [output.view_mut()])
            .unwrap();
        assert_eq!(output.to_f32(), [3., 2., 1.]);
    }

    #[test]
    fn empty_integer_reduction_is_zero_and_validation_failure_is_transactional() {
        let reduction = kernel("i->", &[vec![0]], EinsumExecutionMode::GenericNative);
        let input = Owned::i32(&[0], &[]);
        let mut output = Owned::zeros(DataType::Int32, &[]);
        reduction
            .execute(&[input.view()], &mut [output.view_mut()])
            .unwrap();
        assert_eq!(output.to_i32(), [0]);

        let identity = kernel("i->i", &[vec![2]], EinsumExecutionMode::GenericNative);
        let input = Owned::i32(&[2], &[7, 11]);
        let mut output = Owned::f32(&[2], &[123.0, 456.0]);
        let before = output.bytes.clone();
        let error = identity
            .execute(&[input.view()], &mut [output.view_mut()])
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
            .execute(&[a.view(), b.view()], &mut [fast.view_mut()])
            .unwrap();
        oracle_kernel
            .execute(&[a.view(), b.view()], &mut [oracle.view_mut()])
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
            let claims = OrtGraphView::new(&view).query_capabilities(&provider);
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
                .execute(&[left.view(), right.view()], &mut [output.view_mut()])
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
            OrtGraphView::new(&view).query_capabilities(&provider).len(),
            1
        );
        let kernel = provider
            .get_kernel(view.node(node_index), &[vec![2, 3], vec![3, 2]], 24)
            .unwrap();
        let left = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let right = Owned::f32(&[3, 2], &[1., 0., 0., 1., 1., 0.]);
        let mut output = Owned::zeros_f32(&[2, 2]);
        kernel
            .execute(&[left.view(), right.view()], &mut [output.view_mut()])
            .unwrap();
        assert_eq!(
            output.to_f32(),
            [4., 10., 2., 5.],
            "implicit ASCII output order is [B, Z], the transpose of [Z, B]"
        );
    }

    #[test]
    fn two_session_opposite_verdicts_survive_both_load_orders_and_compute() {
        let _guard = begin_scratch_test(128, 128);
        let mut node = Node::new(NodeId(0), "Einsum", vec![], vec![]);
        node.attributes
            .insert("equation".into(), Attribute::String(b"ij->i".to_vec()));
        let input = Owned::f32(&[2, 4], &[1., 2., 3., 4., 5., 6., 7., 8.]);

        for admitted_first in [true, false] {
            EINSUM_SCRATCH.with(|scratch| {
                scratch.borrow_mut().take();
            });
            TEST_SCRATCH_BUDGET.reset_for_test();

            let first_retention = test_retention(admitted_first);
            let second_retention = test_retention(!admitted_first);
            let first =
                crate::CpuExecutionProvider::with_einsum_scratch_retention(first_retention.clone());
            let second = crate::CpuExecutionProvider::with_einsum_scratch_retention(
                second_retention.clone(),
            );
            let (admitted_provider, admitted_retention, declined_provider, declined_retention) =
                if admitted_first {
                    (&first, &first_retention, &second, &second_retention)
                } else {
                    (&second, &second_retention, &first, &first_retention)
                };
            let admitted_kernel = admitted_provider
                .get_kernel(&node, &[vec![2, 4]], 12)
                .expect("admitted provider compiles Einsum");
            let declined_kernel = declined_provider
                .get_kernel(&node, &[vec![2, 4]], 12)
                .expect("declined provider compiles Einsum");

            let mut admitted_output = Owned::zeros_f32(&[2]);
            admitted_kernel
                .execute(&[input.view()], &mut [admitted_output.view_mut()])
                .expect("admitted kernel executes");
            assert_eq!(admitted_output.to_f32(), [10., 26.]);
            assert!(TEST_SCRATCH_BUDGET.live_bytes() > 0);
            assert_eq!(admitted_retention.active_slots(), 1);

            let mut declined_output = Owned::zeros_f32(&[2]);
            declined_kernel
                .execute(&[input.view()], &mut [declined_output.view_mut()])
                .expect("declined kernel executes");
            assert_eq!(declined_output.to_f32(), [10., 26.]);
            assert_eq!(TEST_SCRATCH_BUDGET.live_bytes(), 0);
            assert_eq!(admitted_retention.active_slots(), 0);
            assert_eq!(declined_retention.active_slots(), 0);

            admitted_kernel
                .execute(&[input.view()], &mut [admitted_output.view_mut()])
                .expect("admitted kernel keeps its original verdict");
            assert_eq!(admitted_output.to_f32(), [10., 26.]);
            assert!(TEST_SCRATCH_BUDGET.live_bytes() > 0);
        }
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
            .execute(&[a.view(), b.view(), c.view()], &mut [out.view_mut()])
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
            EinsumScratchRetention::default(),
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
            .execute(&views, &mut [output.view_mut()])
            .expect("128-input GenericNative fallback executes");
        assert_eq!(output.to_f32(), [1.0]);
        assert_eq!(route(&*kernel), 3);
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
        assert!(unsupported_reason(&node, &[int_shape], &[DataType::Int32]).is_none());

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
        bf16.execute(&[input.view()], &mut [output.view_mut()])
            .expect("Einsum-28 BFloat16 executes");
        assert_eq!(output.to_u16_bits(), input.to_u16_bits());
    }
}

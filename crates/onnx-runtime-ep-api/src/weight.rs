//! Capability-negotiated lazy weight handles for executor-to-EP delivery.

use std::collections::BTreeSet;
use std::sync::Arc;

use onnx_runtime_ir::{DataType, Graph, NodeId, ValueId};

use crate::ExternalMmapRegion;

/// Capability flag advertised by paging-aware execution providers.
pub const NXRT_WEIGHT_PAGING_CAPABILITY: &str = "nxrt";

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExecutionProviderCapabilities {
    flags: BTreeSet<String>,
}

impl ExecutionProviderCapabilities {
    pub fn stock() -> Self {
        Self::default()
    }

    pub fn nxrt_weight_paging() -> Self {
        Self::from_flags([NXRT_WEIGHT_PAGING_CAPABILITY])
    }

    pub fn from_flags(flags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            flags: flags.into_iter().map(Into::into).collect(),
        }
    }

    pub fn advertises(&self, capability: &str) -> bool {
        self.flags.contains(capability)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentWeight {
    pub dtype: DataType,
    pub shape: Vec<usize>,
    bytes: Arc<[u8]>,
}

impl ResidentWeight {
    pub fn new(
        dtype: DataType,
        shape: Vec<usize>,
        bytes: impl Into<Arc<[u8]>>,
    ) -> Result<Self, WeightHandleError> {
        let elements = checked_shape_product(&shape)?;
        let expected = dtype.checked_storage_bytes(elements).ok_or_else(|| {
            WeightHandleError::InvalidResident("resident weight byte count overflow".into())
        })?;
        if expected > isize::MAX as usize {
            return Err(WeightHandleError::InvalidResident(
                "resident weight byte count exceeds isize::MAX".into(),
            ));
        }
        let bytes = bytes.into();
        if bytes.len() != expected {
            return Err(WeightHandleError::InvalidResident(format!(
                "resident weight has {} bytes, expected {expected}",
                bytes.len()
            )));
        }
        Ok(Self {
            dtype,
            shape,
            bytes,
        })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

fn checked_shape_product(shape: &[usize]) -> Result<usize, WeightHandleError> {
    let mut product = 1usize;
    let mut has_zero = false;
    for &dimension in shape {
        if dimension == 0 {
            has_zero = true;
        } else {
            product = product.checked_mul(dimension).ok_or_else(|| {
                WeightHandleError::InvalidResident("resident weight element count overflow".into())
            })?;
        }
    }
    Ok(if has_zero { 0 } else { product })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LazyWeightBoundary {
    /// `ai.onnx::MatMul`, the ordinary dense GEMM boundary used by unquantized
    /// decoder exports.
    MatMul,
    /// `pkg.nxrt::BlockQuantizedMoE`, the Phase-3 offload binding boundary.
    BlockQuantizedMoe,
    /// `com.microsoft::QMoE`, the boundary real DeepSeek/GLM/Qwen exports use.
    QMoe,
    /// `com.microsoft::MatMulNBits`, the packed INT4/INT8 GEMV boundary that
    /// dominates dense-model (e.g. Qwen2.5) VRAM.
    MatMulNBits,
}

impl LazyWeightBoundary {
    /// Every op boundary at which a lazy weight may be device-paged.
    pub const ALL: [LazyWeightBoundary; 4] = [
        Self::MatMul,
        Self::BlockQuantizedMoe,
        Self::QMoe,
        Self::MatMulNBits,
    ];

    /// Canonical (domain, op_type) this boundary binds at.
    fn identity(self) -> (&'static str, &'static str) {
        match self {
            Self::MatMul => ("", "MatMul"),
            Self::BlockQuantizedMoe => ("pkg.nxrt", "BlockQuantizedMoE"),
            Self::QMoe => ("com.microsoft", "QMoE"),
            Self::MatMulNBits => ("com.microsoft", "MatMulNBits"),
        }
    }

    pub fn matches(self, domain: &str, op_type: &str) -> bool {
        let (want_domain, want_op) = self.identity();
        domain == want_domain && op_type == want_op
    }

    /// The offload boundary that binds `(domain, op_type)`, if any.
    pub fn for_op(domain: &str, op_type: &str) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|boundary| boundary.matches(domain, op_type))
    }

    /// Whether any offload boundary binds `(domain, op_type)`.
    pub fn matches_any(domain: &str, op_type: &str) -> bool {
        Self::for_op(domain, op_type).is_some()
    }

    /// Whether this boundary's route-telemetry producer is published by
    /// resolved kernel compilation and can therefore be absent temporarily.
    ///
    /// A missing producer is readiness-dependent only for these boundaries.
    /// For every other boundary, absence is a terminal unsupported capability,
    /// not permission to remain pending forever.
    pub const fn route_telemetry_producer_may_appear_after_compilation(self) -> bool {
        matches!(self, Self::QMoe)
    }
}

/// An initializer the executor may expose as a lazy weight handle.
///
/// Strategy inference and executor construction share this classifier so the
/// reported pageable geometry cannot drift from the weights the runtime pages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LazyWeightCandidate {
    pub value: ValueId,
    pub boundary: LazyWeightBoundary,
    pub first_consumer: NodeId,
}

pub fn lazy_weight_candidates(graph: &Graph) -> Vec<LazyWeightCandidate> {
    let mut candidates = Vec::new();
    for &value in graph.initializers.keys() {
        let graph_value = graph.value(value);
        let consumers = graph.consumers(value);
        let Some(&first_consumer) = consumers.first() else {
            continue;
        };
        let mut boundary = None;
        let lazy_only = graph_value.producer.is_none()
            && !graph.outputs.contains(&value)
            && consumers.into_iter().all(|consumer| {
                let node = graph.node(consumer);
                match LazyWeightBoundary::for_op(&node.domain, &node.op_type) {
                    Some(found) => {
                        boundary.get_or_insert(found);
                        true
                    }
                    None => false,
                }
            });
        if let Some(boundary) = boundary.filter(|_| lazy_only) {
            candidates.push(LazyWeightCandidate {
                value,
                boundary,
                first_consumer,
            });
        }
    }
    candidates
}

/// A logical MoE expert-weight group: every initializer `ValueId` consumed
/// as a weight/scale/zero-point/bias input by the *same* fused-routing
/// (`QMoE`/`BlockQuantizedMoE`) node.
///
/// Derived purely from graph structure (which initializers feed which node's
/// input slots) — **never** from tensor names. A single QMoE node's
/// `fc1_experts_weights`/`fc1_scales`/`fc2_experts_weights`/`fc2_scales`/
/// `fc3_experts_weights`/`fc3_scales`/bias/zero-point inputs together
/// describe *one* logical expert bank; any residency policy or boundary
/// application that would tier some of these `ValueId`s but not others is
/// silently splitting one logical expert across two residency states, which
/// this type exists to make structurally visible and preventable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertWeightGroup {
    /// The node whose inputs define this group (the `QMoE`/`BlockQuantizedMoE`
    /// call site).
    pub node: NodeId,
    /// The offload boundary this node binds at.
    pub boundary: LazyWeightBoundary,
    /// Every initializer `ValueId` consumed as an input to `node`, in input
    /// order, deduplicated. All members must transition atomically: either
    /// every member's residency decision is honored, or the whole group
    /// falls back to whole-bank-resident before any physical mutation.
    pub members: Vec<ValueId>,
}

impl ExpertWeightGroup {
    /// Whether `value` is a member of this group.
    pub fn contains(&self, value: ValueId) -> bool {
        self.members.contains(&value)
    }
}

/// Derive every [`ExpertWeightGroup`] in `graph`: one group per node whose
/// `op_type`/`domain` binds a fused-routing [`LazyWeightBoundary`]
/// (`QMoe`/`BlockQuantizedMoe`) and which consumes at least one initializer
/// input.
///
/// Dense boundaries (`MatMul`/`MatMulNBits`) are intentionally excluded: they
/// have no multi-tensor "logical expert" concept to group — each such
/// initializer is already its own atomic unit. Only fused-routing boundaries,
/// whose kernel call reads several distinct initializers (weights, scales,
/// optional zero-points/bias) for what is semantically one expert bank, need
/// grouping.
///
/// This is graph-structural, not name-based: membership is exactly "which
/// initializer `ValueId`s are input operands of this node", read directly
/// from [`Node::inputs`](onnx_runtime_ir::Node::inputs).
pub fn expert_weight_groups(graph: &Graph) -> Vec<ExpertWeightGroup> {
    let mut groups = Vec::new();
    for (node_id, node) in graph.nodes.iter() {
        let boundary = match LazyWeightBoundary::for_op(&node.domain, &node.op_type) {
            Some(LazyWeightBoundary::QMoe) => LazyWeightBoundary::QMoe,
            Some(LazyWeightBoundary::BlockQuantizedMoe) => LazyWeightBoundary::BlockQuantizedMoe,
            _ => continue,
        };
        let mut members = Vec::new();
        for input in &node.inputs {
            let Some(value) = input else { continue };
            if graph.initializers.contains_key(value) && !members.contains(value) {
                members.push(*value);
            }
        }
        if !members.is_empty() {
            groups.push(ExpertWeightGroup {
                node: node_id,
                boundary,
                members,
            });
        }
    }
    groups
}

pub trait ResidentWeightMaterializer: Send + Sync {
    fn materialize(&self) -> Result<ResidentWeight, WeightHandleError>;
}

impl<F> ResidentWeightMaterializer for F
where
    F: Fn() -> Result<ResidentWeight, WeightHandleError> + Send + Sync,
{
    fn materialize(&self) -> Result<ResidentWeight, WeightHandleError> {
        self()
    }
}

#[derive(Clone)]
pub struct LazyWeight {
    pub boundary: LazyWeightBoundary,
    /// Canonical element type of the backing tensor.
    pub dtype: DataType,
    /// Canonical shape of the backing tensor.
    pub shape: Vec<usize>,
    /// Validated external mmap ranges that back this initializer, in binding
    /// order. Their lengths sum to the canonical byte size of the tensor.
    pub regions: Vec<ExternalMmapRegion>,
    resident_materializer: Arc<dyn ResidentWeightMaterializer>,
}

impl std::fmt::Debug for LazyWeight {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LazyWeight")
            .field("boundary", &self.boundary)
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .field("regions", &self.regions)
            .field("resident_materializer", &"<deferred>")
            .finish()
    }
}

impl LazyWeight {
    /// Build a lazy weight bound at an arbitrary offload boundary.
    pub fn new<M>(
        boundary: LazyWeightBoundary,
        dtype: DataType,
        shape: Vec<usize>,
        regions: Vec<ExternalMmapRegion>,
        resident_materializer: M,
    ) -> Result<Self, WeightHandleError>
    where
        M: ResidentWeightMaterializer + 'static,
    {
        if regions.is_empty() {
            return Err(WeightHandleError::MissingRegions);
        }
        Ok(Self {
            boundary,
            dtype,
            shape,
            regions,
            resident_materializer: Arc::new(resident_materializer),
        })
    }

    pub fn block_quantized_moe<M>(
        dtype: DataType,
        shape: Vec<usize>,
        regions: Vec<ExternalMmapRegion>,
        resident_materializer: M,
    ) -> Result<Self, WeightHandleError>
    where
        M: ResidentWeightMaterializer + 'static,
    {
        Self::new(
            LazyWeightBoundary::BlockQuantizedMoe,
            dtype,
            shape,
            regions,
            resident_materializer,
        )
    }

    /// Total canonical byte size of the backing tensor, summed across regions.
    pub fn region_bytes_len(&self) -> usize {
        self.regions.iter().map(|region| region.len).sum()
    }

    /// Materialize the unchanged stock-EP resident behavior.
    pub fn materialize(&self) -> Result<ResidentWeight, WeightHandleError> {
        self.resident_materializer.materialize()
    }
}

/// General executor weight input: resident today, lazy when an EP opts in.
#[derive(Clone, Debug)]
pub enum WeightHandle {
    Resident(ResidentWeight),
    Lazy(LazyWeight),
}

impl WeightHandle {
    pub fn negotiate(
        &self,
        capabilities: &ExecutionProviderCapabilities,
    ) -> Result<NegotiatedWeight, WeightHandleError> {
        match self {
            Self::Resident(weight) => Ok(NegotiatedWeight::Resident(weight.clone())),
            Self::Lazy(weight) if capabilities.advertises(NXRT_WEIGHT_PAGING_CAPABILITY) => {
                Ok(NegotiatedWeight::Lazy(weight.clone()))
            }
            Self::Lazy(weight) => Ok(NegotiatedWeight::Resident(weight.materialize()?)),
        }
    }

    pub fn is_lazy_for(&self, capabilities: &ExecutionProviderCapabilities) -> bool {
        matches!(self, Self::Lazy(_)) && capabilities.advertises(NXRT_WEIGHT_PAGING_CAPABILITY)
    }

    /// Borrow the inner [`LazyWeight`] when this handle is lazy.
    pub fn as_lazy(&self) -> Option<&LazyWeight> {
        match self {
            Self::Lazy(weight) => Some(weight),
            Self::Resident(_) => None,
        }
    }
}

#[derive(Clone, Debug)]
pub enum NegotiatedWeight {
    Resident(ResidentWeight),
    Lazy(LazyWeight),
}

impl NegotiatedWeight {
    /// Explicit host route available to every executor and EP.
    pub fn materialize_host_fallback(&self) -> Result<ResidentWeight, WeightHandleError> {
        match self {
            Self::Resident(weight) => Ok(weight.clone()),
            Self::Lazy(weight) => weight.materialize(),
        }
    }

    /// Phase 3b will call this at `pkg.nxrt::BlockQuantizedMoE` binding time.
    pub fn try_bind_device<B: LazyDeviceWeightBinder>(
        &self,
        binder: &B,
    ) -> Result<B::Binding, WeightHandleError> {
        match self {
            Self::Resident(_) => Err(WeightHandleError::Unsupported(
                "resident weights do not require lazy device binding".into(),
            )),
            Self::Lazy(weight) => binder.bind_block_quantized_moe(weight),
        }
    }
}

/// EP seam for Phase-3b live device paging.
pub trait LazyDeviceWeightBinder {
    type Binding;

    fn bind_block_quantized_moe(
        &self,
        weight: &LazyWeight,
    ) -> Result<Self::Binding, WeightHandleError>;
}

/// Resolves the live host bytes backing a validated external mmap region.
///
/// A paging-capable device binder uses this to copy only the selected region
/// bytes host→device, rather than materializing the whole resident tensor on the
/// host first (WEIGHT_OFFLOAD §9 invariant 5: never allocate an unbudgeted full
/// expansion). The executor's weight store owns the live mappings and implements
/// this; the returned slice must stay valid for the duration of the copy.
pub trait MmapRegionSource {
    fn region_bytes(&self, region: &ExternalMmapRegion) -> Result<&[u8], WeightHandleError>;

    /// Return the whole live mapping identified by `mapping_id`, if available.
    ///
    /// Used by the zero-copy hybrid weight path to page-lock and device-map an
    /// entire mapping in a single `cuMemHostRegister`, guaranteeing that every
    /// weight's device pointer is contiguous over its full length. Returning
    /// `None` disables zero-copy for that mapping (the caller falls back to a
    /// copy). The returned slice must stay valid for the mapping's lifetime.
    fn full_mapping_bytes(&self, _mapping_id: usize) -> Option<&[u8]> {
        None
    }
}

/// A lazy weight paged into device memory by an EP, ready for a kernel to read.
///
/// The executor substitutes its [`device_ptr`](Self::device_ptr) into the input
/// `TensorView` for the weight and holds this value for the kernel's lifetime.
/// `keep_alive` owns whatever device residency the EP allocated (e.g. a VRAM
/// page), so the memory stays resident until the executor drops the binding —
/// after the kernel has run.
pub struct PagedWeight {
    device_ptr: *const std::ffi::c_void,
    device: onnx_runtime_ir::DeviceId,
    len: usize,
    keep_alive: Arc<dyn std::any::Any + Send + Sync>,
}

impl PagedWeight {
    pub fn new(
        device_ptr: *const std::ffi::c_void,
        device: onnx_runtime_ir::DeviceId,
        len: usize,
        keep_alive: Arc<dyn std::any::Any + Send + Sync>,
    ) -> Self {
        Self {
            device_ptr,
            device,
            len,
            keep_alive,
        }
    }

    /// Opaque device pointer to the paged weight bytes.
    pub fn device_ptr(&self) -> *const std::ffi::c_void {
        self.device_ptr
    }

    /// Device the paged bytes live on.
    pub fn device(&self) -> onnx_runtime_ir::DeviceId {
        self.device
    }

    /// Number of bytes resident in the paged allocation.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the paged allocation is empty (never true for a valid page).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Borrow the residency keep-alive (for downcasting / observability).
    pub fn keep_alive(&self) -> &Arc<dyn std::any::Any + Send + Sync> {
        &self.keep_alive
    }
}

impl std::fmt::Debug for PagedWeight {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PagedWeight")
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

/// CPU-only Phase-3a binder: callers must use the host materialization route.
#[derive(Clone, Copy, Debug, Default)]
pub struct Phase3aHostOnlyBinder;

impl LazyDeviceWeightBinder for Phase3aHostOnlyBinder {
    type Binding = ();

    fn bind_block_quantized_moe(
        &self,
        _weight: &LazyWeight,
    ) -> Result<Self::Binding, WeightHandleError> {
        Err(WeightHandleError::Unsupported(
            "live device weight paging is deferred to WEIGHT_OFFLOAD Phase 3b".into(),
        ))
    }
}

#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
pub enum WeightHandleError {
    #[error("invalid resident weight: {0}")]
    InvalidResident(String),
    #[error("lazy weight requires at least one external mmap region")]
    MissingRegions,
    #[error("unsupported: {0}")]
    Unsupported(String),
    #[error("device weight binding failed: {0}")]
    DeviceBinding(String),
}

/// Why a candidate expert region did not enter a [`ResidencyPlan`] as
/// resident.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResidencyDegradationReason {
    /// The catalog itself was non-pageable (see
    /// [`onnx_runtime_loader::NonPageableReason`]) — the plan cannot promise
    /// per-expert placement for this value at all.
    NonPageableCatalog(onnx_runtime_loader::NonPageableReason),
    /// The policy chose to keep this value fully resident rather than emit
    /// per-expert placement (the default/only shipped behavior today).
    PolicyDeclinedSplit,
    /// The policy asked for a plan this build-time seam cannot safely honor
    /// (e.g. it referenced an expert index out of range, or overlapping
    /// ranges) — the executor falls back to whole-bank residency and records
    /// why.
    RejectedByValidation(String),
}

/// One entry in a [`ResidencyPlan`]: what an execution boundary should do for
/// one lazy-weight value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResidencyDecision {
    /// Keep the whole initializer resident, exactly like today's shipped
    /// behavior. `reason` is `None` for the default policy's decisions and
    /// `Some(..)` when a catalog/validation problem forced this fallback.
    WholeBankResident {
        reason: Option<ResidencyDegradationReason>,
    },
    /// Placement/admission is unconstrained per expert (informational only —
    /// this slice does not execute prefetch/eviction from this decision).
    /// `experts` lists the candidate expert indices in ascending order (the
    /// deterministic ordering requirement).
    PerExpertCandidate { experts: Vec<usize> },
}

/// A validated, typed residency decision set for one build's lazy-weight
/// candidates.
///
/// A `ResidencyPlan` is pure data: it names *which* value gets *which*
/// decision. It never allocates, copies bytes, opens a CUDA stream, or owns a
/// VA range — those stay the exclusive responsibility of
/// `onnx-runtime-ep-cuda::weight_paging`'s `CudaWeightResidency` (or the
/// executor's existing whole-bank materialization path on non-CUDA EPs).
/// Building a plan and *acting* on it are two different lifecycle steps by
/// construction: nothing in this type can mutate memory.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResidencyPlan {
    decisions: std::collections::HashMap<ValueId, ResidencyDecision>,
    /// Name of the policy that produced this plan, for telemetry.
    policy_name: &'static str,
}

impl ResidencyPlan {
    pub fn policy_name(&self) -> &'static str {
        self.policy_name
    }

    pub fn decision(&self, value: ValueId) -> Option<&ResidencyDecision> {
        self.decisions.get(&value)
    }

    /// Values in deterministic (ascending [`ValueId`]) order, for stable
    /// telemetry/log output and reproducible tests.
    pub fn ordered_values(&self) -> impl Iterator<Item = ValueId> + '_ {
        let mut values: Vec<ValueId> = self.decisions.keys().copied().collect();
        values.sort_unstable_by_key(|value| value.0);
        values.into_iter()
    }

    pub fn len(&self) -> usize {
        self.decisions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.decisions.is_empty()
    }

    /// Count of values with a resident decision that carries no degradation
    /// reason (i.e. the policy's ordinary output, not a validation fallback).
    /// Disjoint from [`Self::degraded_count`]: every entry is counted by
    /// exactly one of the two.
    pub fn resident_count(&self) -> usize {
        self.decisions
            .values()
            .filter(|decision| {
                matches!(
                    decision,
                    ResidencyDecision::WholeBankResident { reason: None }
                )
            })
            .count()
    }

    /// Count of values whose decision carries an explicit degradation reason
    /// (nonpageable catalog, policy decline, or validation rejection).
    pub fn degraded_count(&self) -> usize {
        self.decisions
            .values()
            .filter(|decision| {
                matches!(
                    decision,
                    ResidencyDecision::WholeBankResident { reason: Some(_) }
                )
            })
            .count()
    }
}

/// Structural inputs a [`ResidencyPolicy`] may use to decide placement.
///
/// Deliberately excludes model names, op allowlists, and quantization-format
/// branches: a policy may only look at [`LazyWeightBoundary`], the
/// [`onnx_runtime_loader::WeightRegionCatalog`] (byte layout/pageability), and
/// the caller-supplied budget/capability context.
pub struct ResidencyPolicyInput<'a> {
    /// The [`ValueId`] of the weight initializer this decision covers. Used
    /// by profile-based policies (e.g. [`StaticProfileResidencyPolicy`]) to
    /// look up the per-value hot-expert set. Policies that don't need it
    /// (like [`WholeBankResidentPolicy`]) simply ignore it.
    pub value_id: ValueId,
    pub boundary: LazyWeightBoundary,
    pub catalog: &'a onnx_runtime_loader::WeightRegionCatalog,
    /// Advisory device byte budget available for expert-bank residency, if
    /// the caller has one. `None` means "unconstrained" (today's default).
    pub budget_bytes: Option<u64>,
}

/// Eviction-order class a policy assigns to spans admitted at one
/// [`LazyWeightBoundary`]. This names *which* churn population an admitted
/// span joins; it never touches an `Arc`, a page, or a byte — the executing
/// cache (e.g. `onnx-runtime-ep-cuda::weight_paging::CudaWeightResidency`)
/// still owns victim selection, admission, and eviction mechanics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvictionClass {
    /// Ordinary least-recently-used churn population.
    Lru,
    /// Retained ahead of LRU churn (still evictable to avoid OOM, but never
    /// chosen ahead of an ordinary LRU victim) — today's scan-resistant dense
    /// path.
    StableResident,
}

/// Structural, per-admission inputs for the static hot-set pin decision.
/// Deliberately carries only byte-length and *caller-supplied* live-state
/// snapshots (already-pinned membership, bytes pinned so far) rather than
/// letting the policy reach into cache internals directly — the policy
/// remains a pure decision function of these inputs, never a second owner of
/// the pin set itself.
#[derive(Clone, Copy, Debug)]
pub struct AdmissionPolicyInput {
    /// Opaque per-weight identity key (stable within one cache instance).
    pub key: u64,
    /// Byte length of the span being admitted.
    pub len_bytes: u64,
    /// Whether `key` is already in the executing cache's pinned set.
    pub already_pinned: bool,
    /// Total bytes already committed to the pinned set (excluding `key`).
    pub pinned_bytes_used: u64,
}

/// A pluggable placement/admission/eviction/prefetch/resize *decision*
/// surface. A policy answers "which experts (if any) does this value split
/// into for planning purposes", "which churn population does this boundary's
/// spans join", and "should this specific span enter the static hot set",
/// nothing else.
///
/// Implementations must not allocate, copy, synchronize, or hold VA/pointer
/// state — see the module-level `ResidencyPlan` doc for why that split is
/// enforced by type rather than convention.
pub trait ResidencyPolicy: Send + Sync {
    /// Stable identifier for telemetry (e.g. `"whole_bank_resident"`).
    fn name(&self) -> &'static str;

    /// Decide one value's residency for planning purposes only.
    fn decide(&self, input: &ResidencyPolicyInput<'_>) -> ResidencyDecision;

    /// Eviction-order class for spans admitted at `boundary`. Default: always
    /// ordinary LRU (today's non-scan-resistant behavior) — a policy opts
    /// into stable-resident treatment explicitly.
    fn eviction_class(&self, boundary: LazyWeightBoundary) -> EvictionClass {
        let _ = boundary;
        EvictionClass::Lru
    }

    /// Whether to admit this span into the static hot-set pin, once and for
    /// the life of the cache. Default: never pin (today's byte-identical
    /// default with no pin configuration engaged).
    fn should_pin(&self, input: &AdmissionPolicyInput) -> bool {
        let _ = input;
        false
    }
}

/// The only shipped policy today: always keep every candidate value fully
/// resident, exactly matching pre-#82 behavior byte-for-byte/handle-for-handle.
#[derive(Clone, Copy, Debug, Default)]
pub struct WholeBankResidentPolicy;

impl ResidencyPolicy for WholeBankResidentPolicy {
    fn name(&self) -> &'static str {
        "whole_bank_resident"
    }

    fn decide(&self, input: &ResidencyPolicyInput<'_>) -> ResidencyDecision {
        let reason = if input.catalog.is_pageable() {
            None
        } else {
            match input.catalog.pageability() {
                onnx_runtime_loader::Pageability::NonPageable(reason) => Some(
                    ResidencyDegradationReason::NonPageableCatalog(reason.clone()),
                ),
                onnx_runtime_loader::Pageability::Pageable => None,
            }
        };
        ResidencyDecision::WholeBankResident { reason }
    }
}

/// A [`ResidencyPolicy`] that consults a static offline profile: a
/// `HashMap<ValueId, Vec<usize>>` mapping each value to its "hot" expert
/// indices (the set to keep Device-resident). Values absent from the profile
/// fall back to [`WholeBankResidentPolicy`] behavior — byte-for-byte
/// unchanged from today's default.
///
/// The profile is pure data: no model-name inspection, no env-var allowlist,
/// no runtime trace. All entries are validated at `decide()` time by the
/// existing `plan_residency` validation layer; out-of-range or duplicate
/// indices degrade to `WholeBankResident` there, never here.
///
/// This is the sole new production policy in Slice 5 of #1810.
#[derive(Clone, Debug, Default)]
pub struct StaticProfileResidencyPolicy {
    /// Hot expert indices per value. An entry with an empty `Vec` is
    /// treated identically to a missing entry
    /// (→ `WholeBankResident`).
    profile: std::collections::HashMap<ValueId, Vec<usize>>,
}

impl StaticProfileResidencyPolicy {
    /// Build from an explicit profile map. Entries with empty expert lists
    /// are silently dropped (equivalent to a missing entry).
    pub fn new(profile: std::collections::HashMap<ValueId, Vec<usize>>) -> Self {
        let profile = profile
            .into_iter()
            .filter(|(_, experts)| !experts.is_empty())
            .collect();
        Self { profile }
    }

    /// Add or replace the hot-expert set for one value. An empty list
    /// removes any existing entry.
    pub fn with_entry(mut self, value: ValueId, experts: Vec<usize>) -> Self {
        if experts.is_empty() {
            self.profile.remove(&value);
        } else {
            self.profile.insert(value, experts);
        }
        self
    }

    /// How many values have an explicit hot-expert profile.
    pub fn profile_len(&self) -> usize {
        self.profile.len()
    }
}

impl ResidencyPolicy for StaticProfileResidencyPolicy {
    fn name(&self) -> &'static str {
        "static_profile"
    }

    fn decide(&self, input: &ResidencyPolicyInput<'_>) -> ResidencyDecision {
        match self.profile.get(&input.value_id) {
            Some(experts) if !experts.is_empty() => {
                let mut sorted = experts.clone();
                sorted.sort_unstable();
                sorted.dedup();
                ResidencyDecision::PerExpertCandidate { experts: sorted }
            }
            _ => WholeBankResidentPolicy.decide(input),
        }
    }
}

/// Direction of a policy-issued elastic resize request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResizeDirection {
    /// Ask the executing residency cache to make more bytes available
    /// (mirrors the existing over-budget `MemoryLease::grow` path).
    Grow,
    /// Ask the executing residency cache to give bytes back (mirrors the
    /// existing `ReclaimableMappedHolder::reclaim_mapped` path).
    Shrink,
}

/// A policy-issued elastic resize *intent*: "I would like the budget to move
/// by this many bytes, in this direction." This type never allocates, frees,
/// copies, or relocates a virtual address — it is pure data describing a
/// desired outcome. Only the existing Resource Governor / PMM / VMM / weight
/// paging authorities (`MemoryLease::grow`/`shrink`,
/// `ReclaimableMappedHolder::reclaim_mapped`) execute a resize; this type is
/// the seam a [`ResidencyPolicy`] uses to ask for one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResidencyResizeRequest {
    pub direction: ResizeDirection,
    /// Byte delta requested, always measured as a magnitude regardless of
    /// direction (i.e. never negative-encoded).
    pub target_bytes: u64,
    /// Higher priority requests may preempt lower ones when several policies
    /// or callers contend for one resize opportunity; opaque to the executor
    /// beyond ordering. `0` is the default/lowest priority.
    pub priority: u32,
}

/// Everything a resize commit must be true about before any state mutates.
/// Constructed by the caller from the *existing* signals this slice reuses
/// rather than duplicates: [`CudaRuntime::is_capturing`]-equivalent capture
/// state, the provider deferred-release queue's pending count, and whether a
/// page admission is currently in flight. A resize may only be validated
/// against a snapshot that reports every field `false`/`0`.
///
/// [`CudaRuntime::is_capturing`]: <../onnx_runtime_ep_cuda/struct.CudaRuntime.html#method.is_capturing>
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct ResizeSafePoint {
    /// `true` if any graph slot in use is mid-capture/replay.
    pub capturing: bool,
    /// Count of deferred releases the provider's queue has not yet settled.
    pub pending_deferred_releases: u64,
    /// `true` if a page admission (page-in) is currently in flight and has
    /// not yet reached a terminal state.
    pub admission_in_flight: bool,
    /// `true` when this call is running under multi-device/tensor-parallel
    /// execution. No existing barrier/authority coordinates a resize across
    /// devices, so a resize under TP must fail closed rather than invent
    /// distributed synchronization.
    pub multi_device: bool,
    /// Count of live [`RoutedResidencyProof`] guards. Any live guard —
    /// whole-bank or exact-set — promised its holder that the covered region
    /// set stays resident and unrelocated for the guard's lifetime; a resize
    /// that committed while one is alive could break that promise, so a
    /// nonzero count fails the safe-point check closed.
    pub routed_guards_active: u64,
}

impl ResizeSafePoint {
    /// `true` only when every unsafe condition is absent.
    pub fn is_safe(&self) -> bool {
        !self.capturing
            && self.pending_deferred_releases == 0
            && !self.admission_in_flight
            && !self.multi_device
            && self.routed_guards_active == 0
    }

    /// Human-readable reason the current snapshot is not a safe point, or
    /// `None` when [`Self::is_safe`] is `true`. Checked in a fixed priority
    /// order so the reason reported is deterministic across otherwise-tied
    /// snapshots.
    pub fn blocking_reason(&self) -> Option<&'static str> {
        if self.capturing {
            return Some("a CUDA graph is currently capturing or replaying");
        }
        if self.pending_deferred_releases > 0 {
            return Some("deferred weight-page releases have not settled");
        }
        if self.admission_in_flight {
            return Some("a page admission is currently in flight");
        }
        if self.multi_device {
            return Some(
                "no existing barrier/authority coordinates a resize across devices/TP; \
                 failing closed rather than inventing distributed synchronization",
            );
        }
        if self.routed_guards_active > 0 {
            return Some(
                "a RoutedResidencyProof guard is alive and promised its covered region \
                 set stays resident and unrelocated for its lifetime",
            );
        }
        None
    }
}

/// Why a [`ResidencyResizeRequest`] was rejected before any state mutated, or
/// why a resize that began committing had to roll back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResizeRejection {
    /// [`ResizeSafePoint::is_safe`] was `false`; see the wrapped reason.
    NotSafePoint(&'static str),
    /// The request would relocate or expose a currently cold/unmapped
    /// (potentially routed) expert; refused rather than risk correctness.
    WouldExposeColdExpert,
    /// The executor tried to commit and the underlying governor/lease/reclaim
    /// primitive refused or could not fully satisfy the request; the message
    /// is that primitive's own error text.
    ExecutionFailed(String),
    /// `target_bytes` was zero — nothing to do; not an error, but not a plan
    /// either.
    NoOp,
}

/// A validated, *not-yet-executed* resize decision: pure data naming what
/// would happen, produced by [`plan_resize`]. Executing it (calling
/// `reclaim_mapped`/`lease.grow`/`lease.shrink`) is a distinct later step
/// owned entirely by `onnx-runtime-ep-cuda::weight_paging` /
/// `onnx-runtime-memory-governor`, exactly like [`ResidencyPlan`] vs its
/// execution.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResidencyResizePlan {
    /// Safe to attempt; not yet committed.
    Accepted(ResidencyResizeRequest),
    /// Rejected before any state mutated. Keeps the original `request` (not
    /// just the reason) so an execution/telemetry layer can still report
    /// which direction/byte count was asked for, even though nothing
    /// happened.
    Rejected {
        request: ResidencyResizeRequest,
        reason: ResizeRejection,
    },
}

/// Outcome of executing an accepted [`ResidencyResizePlan`]. Always reports
/// what actually happened, even a partial grow/shrink or an outright failure,
/// so telemetry and tests can observe exact before/after state rather than
/// trusting the request was fully honored.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidencyResizeOutcome {
    pub direction: ResizeDirection,
    pub requested_bytes: u64,
    /// Bytes the executor actually moved the budget by. May be less than
    /// `requested_bytes` on a best-effort partial shrink/grow, or `0` on
    /// rejection.
    pub accepted_bytes: u64,
    pub before_bytes: u64,
    pub after_bytes: u64,
    /// `None` on full success; `Some(reason)` on rejection or a failed
    /// commit that rolled back.
    pub rejection: Option<ResizeRejection>,
    /// How many times the commit attempt rolled back before returning.
    pub rollback_count: u32,
    pub safe_point: ResizeSafePoint,
}

impl ResidencyResizeOutcome {
    pub fn is_success(&self) -> bool {
        self.rejection.is_none()
    }
}

/// Validate a [`ResidencyResizeRequest`] against a [`ResizeSafePoint`]
/// snapshot before any state mutates. This is the one place a resize
/// *request* becomes a resize *plan*; nothing here allocates, copies,
/// synchronizes, or touches a lease/allowance/VA. The executor (CUDA weight
/// paging + governor) is solely responsible for turning an
/// [`ResidencyResizePlan::Accepted`] into bytes actually moved.
pub fn plan_resize(
    request: ResidencyResizeRequest,
    safe_point: ResizeSafePoint,
) -> ResidencyResizePlan {
    if request.target_bytes == 0 {
        return ResidencyResizePlan::Rejected {
            request,
            reason: ResizeRejection::NoOp,
        };
    }
    if let Some(reason) = safe_point.blocking_reason() {
        return ResidencyResizePlan::Rejected {
            request,
            reason: ResizeRejection::NotSafePoint(reason),
        };
    }
    ResidencyResizePlan::Accepted(request)
}

/// What a dispatch is asking to be proven resident before it launches.
///
/// Fused-routing kernels (today, every shipped `QMoE`/`BlockQuantizedMoE`
/// CUDA kernel) select experts entirely on-device inside the same launch that
/// consumes their weights — the host never sees `selected_experts` before or
/// during dispatch. [`RoutedResidencyRequirement::FusedRoutingUnknown`] names
/// that honestly. [`RoutedResidencyRequirement::HostKnownExperts`] exists for
/// a future (or non-fused) kernel that computes routing host-side, or on a
/// separate pre-pass, before the weight-touching sub-kernels launch; no
/// kernel in this codebase constructs it today.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RoutedResidencyRequirement {
    /// The caller cannot name the routed set before launch. The only lawful
    /// proof for this requirement is [`RoutedResidencyCoverage::WholeBank`].
    FusedRoutingUnknown,
    /// The caller already knows exactly which experts this dispatch will
    /// touch (e.g. a host-side or pre-pass routing step ran first).
    /// `experts` need not be sorted or deduplicated; [`prove_routed_residency`]
    /// normalizes and validates it.
    HostKnownExperts { experts: Vec<usize> },
}

/// Why a [`RoutedResidencyProof`] covers the whole bank instead of an exact
/// expert set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WholeBankReason {
    /// The requirement was [`RoutedResidencyRequirement::FusedRoutingUnknown`]
    /// — routing is fused into the kernel and no host pre-launch visibility
    /// exists. This is the only reachable reason for QMoE/BlockQuantizedMoE
    /// as shipped today.
    FusedRoutingHasNoHostVisibility,
    /// Host-known expert IDs were supplied but failed validation (an
    /// out-of-range index, or the catalog itself is non-pageable), so the
    /// proof safely degrades to whole-bank rather than risk exposing a cold
    /// expert.
    InvalidExactSet(String),
}

/// What a [`RoutedResidencyProof`] actually covers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RoutedResidencyCoverage {
    /// Every expert region in the bank is proven resident for the guard's
    /// lifetime. `reason` records why an exact set was not used.
    WholeBank { reason: WholeBankReason },
    /// Exactly these experts (sorted ascending, deduplicated) are proven
    /// resident; no other expert region is guaranteed resident by this proof.
    ExactExperts { experts: Vec<usize> },
}

/// An unforgeable, RAII-shaped proof that a QMoE/BlockQuantizedMoE dispatch's
/// expert-bank residency requirement has been satisfied for the lifetime of
/// this value.
///
/// Only [`prove_routed_residency`] can construct one (the private `_sealed`
/// field has no public constructor and no field a caller outside this module
/// can set), so a kernel or policy cannot fabricate a proof to bypass the
/// residency owner. While a guard is alive, [`RoutedResidencyProof::blocks_resize`]
/// tells the resize seam (`plan_resize`/`ResizeSafePoint`) to fail closed
/// rather than relocate or unmap a region this proof promised was resident —
/// dropping the guard (stream completion / deferred release settled) is what
/// releases that constraint.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoutedResidencyProof {
    coverage: RoutedResidencyCoverage,
    _sealed: (),
}

impl RoutedResidencyProof {
    pub fn coverage(&self) -> &RoutedResidencyCoverage {
        &self.coverage
    }

    /// `true` for every proof: a live guard of any coverage (whole-bank or
    /// exact-set) blocks a concurrent resize, because a resize that succeeded
    /// mid-dispatch could relocate or unmap a region this proof promised was
    /// resident for the launch's lifetime.
    pub fn blocks_resize(&self) -> bool {
        true
    }
}

/// Object-safe handle for a live per-EP routed-residency guard (e.g.
/// `onnx-runtime-ep-cuda::weight_paging::RoutedResidencyGuard`), so
/// [`crate::provider::ExecutionProvider::acquire_routed_residency`] can
/// return one without this crate depending on any specific EP backend.
/// Dropping the box is what releases the guard; `proof` exposes the typed
/// attestation it holds.
pub trait RoutedResidencyGuardHandle: Send + Sync {
    fn proof(&self) -> &RoutedResidencyProof;
}

/// The one function that turns a [`RoutedResidencyRequirement`] into a
/// [`RoutedResidencyProof`].
///
/// This is pure validation over `catalog`; it never allocates, copies, pages,
/// or synchronizes. The residency owner (`CudaWeightResidency`) is what
/// already guarantees every expert's bytes are resident before this is
/// called (today: whole-initializer paging, so any coverage this returns is
/// already true by construction); this function's job is only to produce the
/// typed, unforgeable attestation of *which* coverage claim dispatch may rely
/// on, and to refuse (degrading to whole-bank) rather than certify an
/// exact-set claim that fails validation.
///
/// * [`RoutedResidencyRequirement::FusedRoutingUnknown`] always yields
///   [`RoutedResidencyCoverage::WholeBank`] with
///   [`WholeBankReason::FusedRoutingHasNoHostVisibility`] — the only lawful
///   proof when the host has no pre-launch visibility into the routed set.
/// * [`RoutedResidencyRequirement::HostKnownExperts`] yields
///   [`RoutedResidencyCoverage::ExactExperts`] (sorted, deduplicated) only
///   when the catalog is pageable and every index is in range; otherwise it
///   degrades to [`RoutedResidencyCoverage::WholeBank`] with
///   [`WholeBankReason::InvalidExactSet`].
pub fn prove_routed_residency(
    requirement: RoutedResidencyRequirement,
    catalog: &onnx_runtime_loader::WeightRegionCatalog,
) -> RoutedResidencyProof {
    let coverage = match requirement {
        RoutedResidencyRequirement::FusedRoutingUnknown => RoutedResidencyCoverage::WholeBank {
            reason: WholeBankReason::FusedRoutingHasNoHostVisibility,
        },
        RoutedResidencyRequirement::HostKnownExperts { experts } => {
            if !catalog.is_pageable() {
                RoutedResidencyCoverage::WholeBank {
                    reason: WholeBankReason::InvalidExactSet(
                        "catalog is not pageable; cannot certify an exact expert set".into(),
                    ),
                }
            } else {
                let mut sorted = experts;
                sorted.sort_unstable();
                sorted.dedup();
                match sorted
                    .iter()
                    .find(|&&expert| catalog.region(expert).is_none())
                {
                    Some(&bad) => RoutedResidencyCoverage::WholeBank {
                        reason: WholeBankReason::InvalidExactSet(format!(
                            "expert index {bad} is out of range for this catalog"
                        )),
                    },
                    None => RoutedResidencyCoverage::ExactExperts { experts: sorted },
                }
            }
        }
    };
    RoutedResidencyProof {
        coverage,
        _sealed: (),
    }
}

/// Build a validated [`ResidencyPlan`] from the per-expert region candidates
/// gathered at build time, running `policy` over each and validating its
/// output before accepting it.
///
/// This is the one call site where a plan is *created*; nothing here
/// allocates, copies, or synchronizes. Validation failures (an expert index
/// referencing an expert count the catalog does not have, or a duplicate
/// expert index) degrade that single value to `WholeBankResident` with
/// [`ResidencyDegradationReason::RejectedByValidation`] rather than failing
/// the whole plan or the model load — correctness always wins over honoring a
/// policy's request.
pub fn plan_residency<'a>(
    candidates: impl IntoIterator<
        Item = (
            ValueId,
            LazyWeightBoundary,
            &'a onnx_runtime_loader::WeightRegionCatalog,
        ),
    >,
    policy: &dyn ResidencyPolicy,
    budget_bytes: Option<u64>,
) -> ResidencyPlan {
    let mut decisions = std::collections::HashMap::new();
    for (value, boundary, catalog) in candidates {
        let input = ResidencyPolicyInput {
            value_id: value,
            boundary,
            catalog,
            budget_bytes,
        };
        let decision = policy.decide(&input);
        let validated = validate_decision(catalog, decision);
        decisions.insert(value, validated);
    }
    ResidencyPlan {
        decisions,
        policy_name: policy.name(),
    }
}

/// Validate a policy's decision against the catalog it was derived from.
/// Never panics: any inconsistency degrades to `WholeBankResident` with an
/// explicit `RejectedByValidation` reason.
fn validate_decision(
    catalog: &onnx_runtime_loader::WeightRegionCatalog,
    decision: ResidencyDecision,
) -> ResidencyDecision {
    match decision {
        ResidencyDecision::PerExpertCandidate { experts } => {
            if !catalog.is_pageable() {
                return ResidencyDecision::WholeBankResident {
                    reason: Some(ResidencyDegradationReason::RejectedByValidation(
                        "policy proposed per-expert placement over a nonpageable catalog".into(),
                    )),
                };
            }
            let expert_count = catalog.layout().experts;
            let mut seen = std::collections::HashSet::with_capacity(experts.len());
            for &expert in &experts {
                if expert >= expert_count {
                    return ResidencyDecision::WholeBankResident {
                        reason: Some(ResidencyDegradationReason::RejectedByValidation(format!(
                            "expert index {expert} out of range for a {expert_count}-expert bank"
                        ))),
                    };
                }
                if !seen.insert(expert) {
                    return ResidencyDecision::WholeBankResident {
                        reason: Some(ResidencyDegradationReason::RejectedByValidation(format!(
                            "expert index {expert} appears more than once in one plan entry"
                        ))),
                    };
                }
                if catalog.region(expert).is_none() {
                    return ResidencyDecision::WholeBankResident {
                        reason: Some(ResidencyDegradationReason::RejectedByValidation(format!(
                            "expert {expert} has no validated byte range in its catalog"
                        ))),
                    };
                }
            }
            let mut ordered = experts;
            ordered.sort_unstable();
            ResidencyDecision::PerExpertCandidate { experts: ordered }
        }
        whole_bank @ ResidencyDecision::WholeBankResident { .. } => whole_bank,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    fn resident() -> ResidentWeight {
        ResidentWeight::new(DataType::Uint8, vec![4], vec![1, 2, 3, 4]).unwrap()
    }

    fn region() -> ExternalMmapRegion {
        ExternalMmapRegion {
            mapping_id: 7,
            offset: 100,
            len: 4,
        }
    }

    fn lazy() -> WeightHandle {
        WeightHandle::Lazy(
            LazyWeight::block_quantized_moe(DataType::Uint8, vec![4], vec![region()], || {
                Ok(resident())
            })
            .unwrap(),
        )
    }

    #[test]
    fn stock_ep_materializes_the_resident_fallback() {
        let NegotiatedWeight::Resident(weight) = lazy()
            .negotiate(&ExecutionProviderCapabilities::stock())
            .unwrap()
        else {
            panic!("stock EP must receive resident materialization");
        };
        assert_eq!(weight.bytes(), &[1, 2, 3, 4]);
    }

    #[test]
    fn nxrt_capability_preserves_lazy_block_quantized_moe_handle() {
        let materializations = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&materializations);
        let lazy = WeightHandle::Lazy(
            LazyWeight::block_quantized_moe(DataType::Uint8, vec![4], vec![region()], move || {
                counter.fetch_add(1, Ordering::Relaxed);
                Ok(resident())
            })
            .unwrap(),
        );
        let NegotiatedWeight::Lazy(weight) = lazy
            .negotiate(&ExecutionProviderCapabilities::nxrt_weight_paging())
            .unwrap()
        else {
            panic!("nxrt EP must receive lazy weight handle");
        };
        assert_eq!(weight.boundary, LazyWeightBoundary::BlockQuantizedMoe);
        assert_eq!(weight.regions, vec![region()]);
        assert_eq!(materializations.load(Ordering::Relaxed), 0);
        assert_eq!(weight.materialize().unwrap().bytes(), &[1, 2, 3, 4]);
        assert_eq!(materializations.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn offload_boundary_recognizes_dense_and_moe_boundaries() {
        assert_eq!(
            LazyWeightBoundary::for_op("", "MatMul"),
            Some(LazyWeightBoundary::MatMul)
        );
        assert_eq!(
            LazyWeightBoundary::for_op("pkg.nxrt", "BlockQuantizedMoE"),
            Some(LazyWeightBoundary::BlockQuantizedMoe)
        );
        assert_eq!(
            LazyWeightBoundary::for_op("com.microsoft", "QMoE"),
            Some(LazyWeightBoundary::QMoe)
        );
        assert!(LazyWeightBoundary::matches_any("com.microsoft", "QMoE"));
        assert!(LazyWeightBoundary::matches_any(
            "pkg.nxrt",
            "BlockQuantizedMoE"
        ));
        assert!(LazyWeightBoundary::matches_any("", "MatMul"));
        // Wrong domain/op pairings and unrelated ops are not offload boundaries.
        assert_eq!(LazyWeightBoundary::for_op("pkg.nxrt", "QMoE"), None);
        assert_eq!(
            LazyWeightBoundary::for_op("com.microsoft", "BlockQuantizedMoE"),
            None
        );
        assert!(!LazyWeightBoundary::matches_any("ai.onnx", "MatMul"));
    }

    #[test]
    fn only_qmoe_has_a_deferred_route_telemetry_producer() {
        assert!(LazyWeightBoundary::QMoe.route_telemetry_producer_may_appear_after_compilation());
        for boundary in [
            LazyWeightBoundary::MatMul,
            LazyWeightBoundary::BlockQuantizedMoe,
            LazyWeightBoundary::MatMulNBits,
        ] {
            assert!(
                !boundary.route_telemetry_producer_may_appear_after_compilation(),
                "{boundary:?} has no producer publication path"
            );
        }
    }

    fn shape1(n: usize) -> onnx_runtime_ir::Shape {
        onnx_runtime_ir::static_shape([n])
    }

    fn inline_initializer(graph: &mut Graph, name: &str) -> ValueId {
        let value = graph.create_named_value(name, DataType::Uint8, shape1(4));
        graph.set_initializer(
            value,
            onnx_runtime_ir::WeightRef::Inline(onnx_runtime_ir::TensorData::from_raw(
                DataType::Uint8,
                vec![4],
                vec![0u8; 4],
            )),
        );
        value
    }

    #[test]
    fn expert_weight_groups_groups_every_qmoe_input_initializer() {
        let mut graph = Graph::new();
        let input = graph.create_named_value("input", DataType::Float32, shape1(4));
        let router = graph.create_named_value("router_probs", DataType::Float32, shape1(4));
        let fc1_w = inline_initializer(&mut graph, "fc1_experts_weights");
        let fc1_s = inline_initializer(&mut graph, "fc1_scales");
        let fc1_b = inline_initializer(&mut graph, "fc1_experts_bias");
        let fc2_w = inline_initializer(&mut graph, "fc2_experts_weights");
        let fc2_s = inline_initializer(&mut graph, "fc2_scales");
        let fc3_w = inline_initializer(&mut graph, "fc3_experts_weights");
        let fc3_s = inline_initializer(&mut graph, "fc3_scales");
        let output = graph.create_named_value("output", DataType::Float32, shape1(4));

        let mut node = onnx_runtime_ir::Node::new(
            NodeId(0),
            "QMoE",
            vec![
                Some(input),
                Some(router),
                Some(fc1_w),
                Some(fc1_s),
                Some(fc1_b),
                Some(fc2_w),
                Some(fc2_s),
                None,
                Some(fc3_w),
                Some(fc3_s),
            ],
            vec![output],
        );
        node.domain = "com.microsoft".to_string();
        let node_id = graph.insert_node(node);

        let groups = expert_weight_groups(&graph);
        assert_eq!(groups.len(), 1, "exactly one QMoE node -> one group");
        let group = &groups[0];
        assert_eq!(group.node, node_id);
        assert_eq!(group.boundary, LazyWeightBoundary::QMoe);
        // Only initializer-backed inputs are members: `input`/`router_probs`
        // are graph values, not initializers, and must not be included.
        assert_eq!(
            group.members,
            vec![fc1_w, fc1_s, fc1_b, fc2_w, fc2_s, fc3_w, fc3_s]
        );
        assert!(group.contains(fc1_w));
        assert!(group.contains(fc3_s));
        assert!(!group.contains(input));
        assert!(!group.contains(router));
    }

    #[test]
    fn expert_weight_groups_ignores_dense_matmul_and_empty_moe() {
        let mut graph = Graph::new();
        // A dense MatMul boundary: not grouped (each initializer is already
        // its own atomic unit; MatMul has no multi-tensor logical-expert
        // concept).
        let dense_w = inline_initializer(&mut graph, "dense_weight");
        let dense_in = graph.create_named_value("x", DataType::Float32, shape1(4));
        let dense_out = graph.create_named_value("y", DataType::Float32, shape1(4));
        graph.insert_node(onnx_runtime_ir::Node::new(
            NodeId(0),
            "MatMul",
            vec![Some(dense_in), Some(dense_w)],
            vec![dense_out],
        ));

        // A QMoE node with zero initializer inputs (e.g. all inputs are
        // graph values, not weights) yields no group.
        let a = graph.create_named_value("a", DataType::Float32, shape1(4));
        let b = graph.create_named_value("b", DataType::Float32, shape1(4));
        let out2 = graph.create_named_value("out2", DataType::Float32, shape1(4));
        let mut empty_qmoe =
            onnx_runtime_ir::Node::new(NodeId(0), "QMoE", vec![Some(a), Some(b)], vec![out2]);
        empty_qmoe.domain = "com.microsoft".to_string();
        graph.insert_node(empty_qmoe);

        assert!(expert_weight_groups(&graph).is_empty());
    }

    #[test]
    fn phase3a_device_binding_is_explicitly_unsupported_with_host_route() {
        let negotiated = lazy()
            .negotiate(&ExecutionProviderCapabilities::nxrt_weight_paging())
            .unwrap();
        assert_eq!(
            negotiated.try_bind_device(&Phase3aHostOnlyBinder),
            Err(WeightHandleError::Unsupported(
                "live device weight paging is deferred to WEIGHT_OFFLOAD Phase 3b".into()
            ))
        );
        assert_eq!(
            negotiated.materialize_host_fallback().unwrap().bytes(),
            &[1, 2, 3, 4]
        );
    }

    // -- ResidencyPlan / ResidencyPolicy tests -----------------------------

    use onnx_runtime_ir::WeightRef;
    use onnx_runtime_loader::{
        ExpertQuantization, ExpertStorageOrder, ExpertTensorLayout, WeightRegionCatalog,
    };

    fn expert_layout() -> ExpertTensorLayout {
        ExpertTensorLayout {
            version: 1,
            experts: 3,
            rows_per_expert: 2,
            storage_elements_per_row: 4,
            order: ExpertStorageOrder::ExpertMajor,
            quantization: Some(ExpertQuantization {
                bits: 4,
                block_size: 16,
                blocks_per_row: 1,
            }),
        }
    }

    fn pageable_catalog() -> WeightRegionCatalog {
        let layout = expert_layout();
        let weight = WeightRef::External {
            path: std::path::PathBuf::from("/nonexistent/weights.bin"),
            offset: 16,
            length: layout.experts * layout.rows_per_expert * layout.storage_elements_per_row,
            dtype: DataType::Uint8,
            dims: vec![
                layout.experts,
                layout.rows_per_expert,
                layout.storage_elements_per_row,
            ],
        };
        WeightRegionCatalog::classify(&weight, layout)
    }

    fn non_pageable_catalog() -> WeightRegionCatalog {
        let mut layout = expert_layout();
        layout.order = ExpertStorageOrder::Interleaved;
        let weight = WeightRef::External {
            path: std::path::PathBuf::from("/nonexistent/weights.bin"),
            offset: 16,
            length: layout.experts * layout.rows_per_expert * layout.storage_elements_per_row,
            dtype: DataType::Uint8,
            dims: vec![
                layout.experts,
                layout.rows_per_expert,
                layout.storage_elements_per_row,
            ],
        };
        WeightRegionCatalog::classify(&weight, layout)
    }

    /// Test-only alternate policy proving the trait boundary is
    /// substitutable: it proposes splitting every pageable catalog into
    /// per-expert candidates. This is never shipped as a default and must
    /// not ship LRU/static-hot/q* behavior.
    struct AlwaysSplitPolicy;

    impl ResidencyPolicy for AlwaysSplitPolicy {
        fn name(&self) -> &'static str {
            "test_always_split"
        }

        fn decide(&self, input: &ResidencyPolicyInput<'_>) -> ResidencyDecision {
            if input.catalog.is_pageable() {
                ResidencyDecision::PerExpertCandidate {
                    experts: (0..input.catalog.layout().experts).collect(),
                }
            } else {
                ResidencyDecision::WholeBankResident { reason: None }
            }
        }
    }

    struct OutOfRangePolicy;

    impl ResidencyPolicy for OutOfRangePolicy {
        fn name(&self) -> &'static str {
            "test_out_of_range"
        }

        fn decide(&self, _input: &ResidencyPolicyInput<'_>) -> ResidencyDecision {
            ResidencyDecision::PerExpertCandidate {
                experts: vec![9999],
            }
        }
    }

    struct DuplicatePolicy;

    impl ResidencyPolicy for DuplicatePolicy {
        fn name(&self) -> &'static str {
            "test_duplicate"
        }

        fn decide(&self, _input: &ResidencyPolicyInput<'_>) -> ResidencyDecision {
            ResidencyDecision::PerExpertCandidate {
                experts: vec![0, 0],
            }
        }
    }

    fn value(id: u32) -> ValueId {
        ValueId(id)
    }

    #[test]
    fn whole_bank_resident_policy_matches_default_behavior_for_pageable_catalog() {
        let catalog = pageable_catalog();
        let plan = plan_residency(
            [(value(1), LazyWeightBoundary::QMoe, &catalog)],
            &WholeBankResidentPolicy,
            None,
        );
        assert_eq!(plan.policy_name(), "whole_bank_resident");
        assert_eq!(plan.len(), 1);
        assert_eq!(plan.resident_count(), 1);
        assert_eq!(plan.degraded_count(), 0);
        assert_eq!(
            plan.decision(value(1)),
            Some(&ResidencyDecision::WholeBankResident { reason: None })
        );
    }

    #[test]
    fn whole_bank_resident_policy_surfaces_non_pageable_reason() {
        let catalog = non_pageable_catalog();
        let plan = plan_residency(
            [(value(1), LazyWeightBoundary::QMoe, &catalog)],
            &WholeBankResidentPolicy,
            None,
        );
        assert_eq!(
            plan.decision(value(1)),
            Some(&ResidencyDecision::WholeBankResident {
                reason: Some(ResidencyDegradationReason::NonPageableCatalog(
                    onnx_runtime_loader::NonPageableReason::NotExpertMajor
                ))
            })
        );
    }

    #[test]
    fn alternate_policy_is_substitutable_and_produces_per_expert_candidates() {
        let catalog = pageable_catalog();
        let plan = plan_residency(
            [(value(1), LazyWeightBoundary::QMoe, &catalog)],
            &AlwaysSplitPolicy,
            None,
        );
        assert_eq!(plan.policy_name(), "test_always_split");
        assert_eq!(
            plan.decision(value(1)),
            Some(&ResidencyDecision::PerExpertCandidate {
                experts: vec![0, 1, 2]
            })
        );
    }

    #[test]
    fn out_of_range_expert_index_degrades_to_whole_bank_with_reason() {
        let catalog = pageable_catalog();
        let plan = plan_residency(
            [(value(1), LazyWeightBoundary::QMoe, &catalog)],
            &OutOfRangePolicy,
            None,
        );
        assert_eq!(plan.resident_count(), 0);
        match plan.decision(value(1)) {
            Some(ResidencyDecision::WholeBankResident {
                reason: Some(ResidencyDegradationReason::RejectedByValidation(_)),
            }) => {}
            other => panic!("expected rejected validation, got {other:?}"),
        }
        assert_eq!(plan.degraded_count(), 1);
    }

    #[test]
    fn duplicate_expert_index_degrades_to_whole_bank_with_reason() {
        let catalog = pageable_catalog();
        let plan = plan_residency(
            [(value(1), LazyWeightBoundary::QMoe, &catalog)],
            &DuplicatePolicy,
            None,
        );
        match plan.decision(value(1)) {
            Some(ResidencyDecision::WholeBankResident {
                reason: Some(ResidencyDegradationReason::RejectedByValidation(_)),
            }) => {}
            other => panic!("expected rejected validation, got {other:?}"),
        }
    }

    #[test]
    fn per_expert_candidate_over_non_pageable_catalog_is_rejected() {
        let catalog = non_pageable_catalog();
        let plan = plan_residency(
            [(value(1), LazyWeightBoundary::QMoe, &catalog)],
            &AlwaysSplitPolicy,
            None,
        );
        // AlwaysSplitPolicy itself checks is_pageable(), so this exercises
        // the policy's own fallback rather than validation -- assert it is
        // still whole-bank resident, never a candidate over a non-pageable
        // catalog.
        assert!(matches!(
            plan.decision(value(1)),
            Some(ResidencyDecision::WholeBankResident { .. })
        ));
    }

    #[test]
    fn plan_residency_orders_values_deterministically() {
        let catalog = pageable_catalog();
        let plan = plan_residency(
            [
                (value(5), LazyWeightBoundary::QMoe, &catalog),
                (value(1), LazyWeightBoundary::QMoe, &catalog),
                (value(3), LazyWeightBoundary::QMoe, &catalog),
            ],
            &WholeBankResidentPolicy,
            None,
        );
        let ordered: Vec<u32> = plan.ordered_values().map(|v| v.0).collect();
        assert_eq!(ordered, vec![1, 3, 5]);
    }

    // -- ResidencyResizeRequest / ResizeSafePoint / plan_resize --

    fn grow_request(bytes: u64) -> ResidencyResizeRequest {
        ResidencyResizeRequest {
            direction: ResizeDirection::Grow,
            target_bytes: bytes,
            priority: 0,
        }
    }

    fn shrink_request(bytes: u64) -> ResidencyResizeRequest {
        ResidencyResizeRequest {
            direction: ResizeDirection::Shrink,
            target_bytes: bytes,
            priority: 0,
        }
    }

    #[test]
    fn plan_resize_accepts_at_a_safe_point() {
        let plan = plan_resize(grow_request(1024), ResizeSafePoint::default());
        assert_eq!(plan, ResidencyResizePlan::Accepted(grow_request(1024)));
    }

    #[test]
    fn plan_resize_rejects_zero_byte_request_as_noop() {
        let plan = plan_resize(grow_request(0), ResizeSafePoint::default());
        assert_eq!(
            plan,
            ResidencyResizePlan::Rejected {
                request: grow_request(0),
                reason: ResizeRejection::NoOp,
            }
        );
    }

    #[test]
    fn plan_resize_rejects_during_capture() {
        let unsafe_point = ResizeSafePoint {
            capturing: true,
            ..Default::default()
        };
        let plan = plan_resize(shrink_request(512), unsafe_point);
        assert!(matches!(
            plan,
            ResidencyResizePlan::Rejected {
                reason: ResizeRejection::NotSafePoint(_),
                ..
            }
        ));
    }

    #[test]
    fn plan_resize_rejects_with_pending_deferred_releases() {
        let unsafe_point = ResizeSafePoint {
            pending_deferred_releases: 3,
            ..Default::default()
        };
        let plan = plan_resize(shrink_request(512), unsafe_point);
        assert!(matches!(
            plan,
            ResidencyResizePlan::Rejected {
                reason: ResizeRejection::NotSafePoint(_),
                ..
            }
        ));
    }

    #[test]
    fn plan_resize_rejects_with_admission_in_flight() {
        let unsafe_point = ResizeSafePoint {
            admission_in_flight: true,
            ..Default::default()
        };
        let plan = plan_resize(grow_request(512), unsafe_point);
        assert!(matches!(
            plan,
            ResidencyResizePlan::Rejected {
                reason: ResizeRejection::NotSafePoint(_),
                ..
            }
        ));
    }

    #[test]
    fn plan_resize_fails_closed_under_multi_device() {
        let unsafe_point = ResizeSafePoint {
            multi_device: true,
            ..Default::default()
        };
        let plan = plan_resize(shrink_request(512), unsafe_point);
        let ResidencyResizePlan::Rejected {
            reason: ResizeRejection::NotSafePoint(reason),
            ..
        } = plan
        else {
            panic!("multi-device must fail closed, not silently coordinate a resize");
        };
        assert!(reason.contains("barrier"));
    }

    #[test]
    fn safe_point_blocking_reason_is_deterministic_when_several_conditions_hold() {
        // Capture always wins first, regardless of what else is also unsafe.
        let point = ResizeSafePoint {
            capturing: true,
            pending_deferred_releases: 5,
            admission_in_flight: true,
            multi_device: true,
            routed_guards_active: 1,
        };
        assert!(!point.is_safe());
        assert_eq!(
            point.blocking_reason(),
            Some("a CUDA graph is currently capturing or replaying")
        );
    }

    #[test]
    fn resize_outcome_reports_success_only_without_rejection() {
        let success = ResidencyResizeOutcome {
            direction: ResizeDirection::Grow,
            requested_bytes: 100,
            accepted_bytes: 100,
            before_bytes: 0,
            after_bytes: 100,
            rejection: None,
            rollback_count: 0,
            safe_point: ResizeSafePoint::default(),
        };
        assert!(success.is_success());

        let failure = ResidencyResizeOutcome {
            rejection: Some(ResizeRejection::WouldExposeColdExpert),
            ..success
        };
        assert!(!failure.is_success());
    }

    #[test]
    fn fused_routing_unknown_always_yields_whole_bank() {
        let catalog = pageable_catalog();
        let proof =
            prove_routed_residency(RoutedResidencyRequirement::FusedRoutingUnknown, &catalog);
        assert_eq!(
            proof.coverage(),
            &RoutedResidencyCoverage::WholeBank {
                reason: WholeBankReason::FusedRoutingHasNoHostVisibility
            }
        );
        assert!(proof.blocks_resize());
    }

    #[test]
    fn host_known_experts_over_a_pageable_catalog_yields_exact_sorted_deduped_set() {
        let catalog = pageable_catalog();
        let proof = prove_routed_residency(
            RoutedResidencyRequirement::HostKnownExperts {
                experts: vec![2, 0, 0, 1],
            },
            &catalog,
        );
        assert_eq!(
            proof.coverage(),
            &RoutedResidencyCoverage::ExactExperts {
                experts: vec![0, 1, 2]
            }
        );
    }

    #[test]
    fn host_known_experts_over_a_non_pageable_catalog_degrades_to_whole_bank_with_reason() {
        let catalog = non_pageable_catalog();
        let proof = prove_routed_residency(
            RoutedResidencyRequirement::HostKnownExperts { experts: vec![0] },
            &catalog,
        );
        let RoutedResidencyCoverage::WholeBank {
            reason: WholeBankReason::InvalidExactSet(reason),
        } = proof.coverage()
        else {
            panic!("a non-pageable catalog must never certify an exact expert set");
        };
        assert!(reason.contains("not pageable"));
    }

    #[test]
    fn host_known_experts_with_an_out_of_range_index_degrades_to_whole_bank_with_reason() {
        let catalog = pageable_catalog();
        // `expert_layout()` has 3 experts (indices 0..=2); 99 is out of range.
        let proof = prove_routed_residency(
            RoutedResidencyRequirement::HostKnownExperts {
                experts: vec![0, 99],
            },
            &catalog,
        );
        let RoutedResidencyCoverage::WholeBank {
            reason: WholeBankReason::InvalidExactSet(reason),
        } = proof.coverage()
        else {
            panic!("an out-of-range expert index must never be certified");
        };
        assert!(reason.contains("99"));
    }

    #[test]
    fn resize_safe_point_fails_closed_while_a_routed_guard_is_active() {
        let point = ResizeSafePoint {
            routed_guards_active: 1,
            ..Default::default()
        };
        assert!(!point.is_safe());
        assert_eq!(
            point.blocking_reason(),
            Some(
                "a RoutedResidencyProof guard is alive and promised its covered region \
                 set stays resident and unrelocated for its lifetime"
            )
        );
    }

    #[test]
    fn resize_safe_point_is_safe_with_no_routed_guards() {
        let point = ResizeSafePoint {
            routed_guards_active: 0,
            ..Default::default()
        };
        assert!(point.is_safe());
    }

    // -- StaticProfileResidencyPolicy tests -------------------------------

    #[test]
    fn static_profile_policy_emits_per_expert_candidate_for_profiled_value() {
        let catalog = pageable_catalog();
        let mut profile = std::collections::HashMap::new();
        profile.insert(value(7), vec![2, 0]);
        let policy = StaticProfileResidencyPolicy::new(profile);
        assert_eq!(policy.profile_len(), 1);
        let plan = plan_residency(
            [(value(7), LazyWeightBoundary::QMoe, &catalog)],
            &policy,
            None,
        );
        assert_eq!(plan.policy_name(), "static_profile");
        assert_eq!(
            plan.decision(value(7)),
            Some(&ResidencyDecision::PerExpertCandidate {
                experts: vec![0, 2]
            })
        );
    }

    #[test]
    fn static_profile_policy_missing_value_falls_back_to_whole_bank() {
        let catalog = pageable_catalog();
        let policy = StaticProfileResidencyPolicy::new(std::collections::HashMap::new());
        let plan = plan_residency(
            [(value(1), LazyWeightBoundary::QMoe, &catalog)],
            &policy,
            None,
        );
        assert_eq!(
            plan.decision(value(1)),
            Some(&ResidencyDecision::WholeBankResident { reason: None })
        );
    }

    #[test]
    fn static_profile_policy_empty_entry_is_treated_as_missing() {
        let catalog = pageable_catalog();
        let policy = StaticProfileResidencyPolicy::default()
            .with_entry(value(3), vec![])
            .with_entry(value(4), vec![1]);
        assert_eq!(policy.profile_len(), 1);
        let plan = plan_residency(
            [
                (value(3), LazyWeightBoundary::QMoe, &catalog),
                (value(4), LazyWeightBoundary::QMoe, &catalog),
            ],
            &policy,
            None,
        );
        assert!(matches!(
            plan.decision(value(3)),
            Some(&ResidencyDecision::WholeBankResident { .. })
        ));
        assert!(matches!(
            plan.decision(value(4)),
            Some(&ResidencyDecision::PerExpertCandidate { .. })
        ));
    }

    #[test]
    fn static_profile_policy_out_of_range_expert_degrades_via_validation() {
        let catalog = pageable_catalog();
        let policy = StaticProfileResidencyPolicy::default().with_entry(value(9), vec![9999]);
        let plan = plan_residency(
            [(value(9), LazyWeightBoundary::QMoe, &catalog)],
            &policy,
            None,
        );
        assert!(matches!(
            plan.decision(value(9)),
            Some(&ResidencyDecision::WholeBankResident {
                reason: Some(ResidencyDegradationReason::RejectedByValidation(_))
            })
        ));
    }
}

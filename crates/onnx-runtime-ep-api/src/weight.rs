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
}

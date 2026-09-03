//! Bounded deterministic binary contraction-tree planning for general Einsum.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use super::{
    EinsumAxis, EinsumConcreteGemmGeometry, EinsumDimension, EinsumGemmGeometry, EinsumLogicalAxis,
    EinsumOperandPlan,
};

/// A candidate-local value identifier.
///
/// Leaf values use their input index. Derived values have stable identifiers
/// within one candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EinsumValueId(usize);

impl EinsumValueId {
    /// Numeric identifier.
    pub const fn index(self) -> usize {
        self.0
    }
}

impl fmt::Display for EinsumValueId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

/// A stable lexicographic contraction-tree candidate identifier.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EinsumContractionTreeCandidateId(String);

impl EinsumContractionTreeCandidateId {
    /// Stable textual form, for example `((0,1),2)`.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for EinsumContractionTreeCandidateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A checked planning-time cost bound.
///
/// Unknown symbolic dimensions have no fabricated finite bound. They compare
/// as an infinite upper bound until concrete runtime shapes are supplied.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EinsumCostBound {
    /// Exact checked value.
    Exact(u128),
    /// Unknown value with an unbounded/infinite upper bound.
    UnknownUpperBound,
}

impl EinsumCostBound {
    /// Exact value, when statically known.
    pub const fn exact(self) -> Option<u128> {
        match self {
            Self::Exact(value) => Some(value),
            Self::UnknownUpperBound => None,
        }
    }

    /// Whether concrete runtime shapes are required to obtain a finite value.
    pub const fn requires_concrete_shape(self) -> bool {
        matches!(self, Self::UnknownUpperBound)
    }

    /// Scale an element count to bytes without wrapping.
    pub fn checked_scale(self, multiplier: usize) -> Option<Self> {
        match (self, multiplier) {
            (_, 0) | (Self::Exact(0), _) => Some(Self::Exact(0)),
            (Self::Exact(value), multiplier) => {
                Some(Self::Exact(value.checked_mul(multiplier as u128)?))
            }
            (Self::UnknownUpperBound, _) => Some(Self::UnknownUpperBound),
        }
    }
}

fn compare_bound(left: EinsumCostBound, right: EinsumCostBound) -> Ordering {
    match (left, right) {
        (EinsumCostBound::Exact(left), EinsumCostBound::Exact(right)) => left.cmp(&right),
        (EinsumCostBound::Exact(_), EinsumCostBound::UnknownUpperBound) => Ordering::Less,
        (EinsumCostBound::UnknownUpperBound, EinsumCostBound::Exact(_)) => Ordering::Greater,
        (EinsumCostBound::UnknownUpperBound, EinsumCostBound::UnknownUpperBound) => Ordering::Equal,
    }
}

/// A cost component whose checked arithmetic can reject one candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EinsumCostMetric {
    /// Total scalar arithmetic operations.
    Flops,
    /// Unary reductions and non-contracting products.
    UnaryOrProductWork,
    /// Sum of all temporary value element counts.
    IntermediateElements,
    /// Maximum simultaneously live temporary elements.
    PeakLiveTemporaryElements,
    /// Temporary writes plus their consuming reads.
    TotalIntermediateTraffic,
    /// Estimated permutation/diagonal packing traffic.
    LayoutOrPackingTraffic,
    /// Extra logical operand elements caused by batch broadcasting.
    BroadcastAmplification,
    /// A binary step's static B/M/K/N geometry.
    Geometry,
    /// A byte-scaled cost component.
    Bytes,
}

/// Explicit deterministic bounds for contraction planning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EinsumPlannerBudget {
    /// Largest arity eligible for exact subset-DP enumeration.
    pub exact_operand_limit: usize,
    /// Maximum subset states retained by exact planning.
    pub max_states: usize,
    /// Maximum candidate trees constructed by one planning pass.
    pub max_candidates: usize,
    /// Maximum logical axes eligible for exact planning.
    pub max_exact_axes: usize,
    /// Maximum total greedy work units. The planner reserves worst-case tree,
    /// leaf-set, stable-ID, and lowering metadata first; the remainder pays
    /// for pair scoring.
    pub max_heuristic_candidates: usize,
}

impl EinsumPlannerBudget {
    /// Hard structural ceiling for exact subset enumeration, independent of a
    /// caller raising `exact_operand_limit`.
    pub const MAX_EXACT_TREE_OPERANDS: usize = 8;
    /// Hard ceiling for a materialized contraction tree.
    pub const MAX_CONTRACTION_TREE_DEPTH: usize = 64;
    /// Conservative retained-metadata allowance charged per exact candidate.
    pub const EXACT_METADATA_UNITS_PER_CANDIDATE: usize = 1024;

    /// Exact planner metadata ceiling derived from `max_candidates`.
    pub fn exact_metadata_units_limit(self) -> Option<usize> {
        self.max_candidates
            .checked_mul(Self::EXACT_METADATA_UNITS_PER_CANDIDATE)
    }
}

impl Default for EinsumPlannerBudget {
    fn default() -> Self {
        Self {
            exact_operand_limit: 5,
            max_states: 64,
            max_candidates: 4096,
            max_exact_axes: 64,
            max_heuristic_candidates: 4096,
        }
    }
}

/// Quality of the bounded planner result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EinsumPlannerQuality {
    /// Every ordered binary tree was enumerated through subset DP.
    ExactSubsetDp,
    /// A deterministic bounded greedy tree was selected.
    DeterministicGreedy,
    /// Tree construction exceeded a work, metadata, or depth bound, so the
    /// semantic plan intentionally retains only its universal index program.
    GenericNativeFallback,
}

/// Why bounded tree planning retained only the universal index program.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EinsumPlannerFallbackReason {
    /// Conservative tree/lowering metadata or total heuristic work exceeded
    /// `max_heuristic_candidates`.
    WorkOrMetadataBudgetExceeded,
    /// The selected tree would exceed the explicit depth ceiling.
    MaximumDepthExceeded,
    /// Checked planner bookkeeping arithmetic could not represent its bounds.
    PlanningArithmeticOverflow,
}

impl fmt::Display for EinsumPlannerFallbackReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkOrMetadataBudgetExceeded => {
                f.write_str("the configured work/metadata budget was exceeded")
            }
            Self::MaximumDepthExceeded => {
                f.write_str("the explicit contraction-tree depth ceiling was exceeded")
            }
            Self::PlanningArithmeticOverflow => {
                f.write_str("checked planner bookkeeping arithmetic overflowed")
            }
        }
    }
}

/// Actual state/candidate/axis consumption for one planning pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EinsumPlannerUsage {
    states: usize,
    candidates: usize,
    axes: usize,
    work: usize,
    metadata_units: usize,
    max_depth: usize,
    candidate_id_bytes: usize,
    budget: EinsumPlannerBudget,
}

impl EinsumPlannerUsage {
    /// Subset or forest states considered.
    pub const fn states(self) -> usize {
        self.states
    }

    /// Candidate trees or pair merges considered.
    pub const fn candidates(self) -> usize {
        self.candidates
    }

    /// Logical axes in the equation.
    pub const fn axes(self) -> usize {
        self.axes
    }

    /// Total bounded planning work units. Heuristic planning includes pair
    /// scoring plus the conservative metadata reservation; exact planning
    /// includes candidate construction and retained subset metadata.
    pub const fn work(self) -> usize {
        self.work
    }

    /// Conservative units reserved for tree nodes, leaf membership, lowering
    /// metadata, and stable candidate identifiers.
    pub const fn metadata_units(self) -> usize {
        self.metadata_units
    }

    /// Deepest selected binary tree node. Generic fallback reports zero.
    pub const fn max_depth(self) -> usize {
        self.max_depth
    }

    /// Bytes in the materialized public candidate identifiers.
    pub const fn candidate_id_bytes(self) -> usize {
        self.candidate_id_bytes
    }

    /// Configured deterministic bounds.
    pub const fn budget(self) -> EinsumPlannerBudget {
        self.budget
    }
}

impl fmt::Display for EinsumCostMetric {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Flops => "FLOPs",
            Self::UnaryOrProductWork => "unary/product work",
            Self::IntermediateElements => "intermediate elements",
            Self::PeakLiveTemporaryElements => "peak live temporary elements",
            Self::TotalIntermediateTraffic => "total intermediate traffic",
            Self::LayoutOrPackingTraffic => "layout/packing traffic",
            Self::BroadcastAmplification => "broadcast amplification",
            Self::Geometry => "B/M/K/N geometry",
            Self::Bytes => "byte cost",
        };
        f.write_str(name)
    }
}

/// Why one semantically legal ordered tree cannot be costed/lowered safely.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EinsumContractionTreeCandidateUnsupportedReason {
    /// Checked arithmetic overflowed.
    CostOverflow {
        /// Rejected metric.
        metric: EinsumCostMetric,
    },
    /// A reduced axis survived below the lowest node containing all of its
    /// occurrences. This is an internal fail-closed guard.
    UnloweredLocalReduction {
        /// Axis that should already have been reduced.
        axis: EinsumAxis,
    },
}

impl fmt::Display for EinsumContractionTreeCandidateUnsupportedReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CostOverflow { metric } => {
                write!(f, "{metric} exceeds the checked u128 planning bound")
            }
            Self::UnloweredLocalReduction { axis } => write!(
                f,
                "{axis} remained on only one child after its lowest legal reduction node"
            ),
        }
    }
}

/// Deterministic EP-neutral cost for one supported candidate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EinsumContractionCost {
    flops: EinsumCostBound,
    unary_or_product_work: EinsumCostBound,
    intermediate_elements: EinsumCostBound,
    peak_live_temporary_elements: EinsumCostBound,
    total_intermediate_traffic_elements: EinsumCostBound,
    layout_or_packing_traffic_elements: EinsumCostBound,
    broadcast_amplification_elements: EinsumCostBound,
    slot_count: usize,
}

impl EinsumContractionCost {
    /// Total scalar arithmetic operations.
    pub const fn flops(&self) -> EinsumCostBound {
        self.flops
    }

    /// Subset of arithmetic due to leaf-local reductions or K-free products.
    pub const fn unary_or_product_work(&self) -> EinsumCostBound {
        self.unary_or_product_work
    }

    /// Sum of all values stored in temporary slots.
    pub const fn intermediate_elements(&self) -> EinsumCostBound {
        self.intermediate_elements
    }

    /// Intermediate bytes for a caller-supplied common element width.
    pub fn intermediate_bytes(&self, element_size: usize) -> Option<EinsumCostBound> {
        self.intermediate_elements.checked_scale(element_size)
    }

    /// Maximum simultaneously live temporary elements.
    pub const fn peak_live_temporary_elements(&self) -> EinsumCostBound {
        self.peak_live_temporary_elements
    }

    /// Maximum simultaneously live temporary bytes.
    pub fn peak_live_temporary_bytes(&self, element_size: usize) -> Option<EinsumCostBound> {
        self.peak_live_temporary_elements
            .checked_scale(element_size)
    }

    /// Temporary writes plus consuming reads, in elements.
    pub const fn total_intermediate_traffic_elements(&self) -> EinsumCostBound {
        self.total_intermediate_traffic_elements
    }

    /// Temporary writes plus consuming reads, in bytes.
    pub fn total_intermediate_traffic_bytes(&self, element_size: usize) -> Option<EinsumCostBound> {
        self.total_intermediate_traffic_elements
            .checked_scale(element_size)
    }

    /// Estimated layout/packing traffic, in elements.
    pub const fn layout_or_packing_traffic_elements(&self) -> EinsumCostBound {
        self.layout_or_packing_traffic_elements
    }

    /// Estimated layout/packing traffic, in bytes.
    pub fn layout_or_packing_traffic_bytes(&self, element_size: usize) -> Option<EinsumCostBound> {
        self.layout_or_packing_traffic_elements
            .checked_scale(element_size)
    }

    /// Extra logical operand elements caused by batch broadcasting.
    pub const fn broadcast_amplification_elements(&self) -> EinsumCostBound {
        self.broadcast_amplification_elements
    }

    /// Number of reusable temporary slots in the linear-scan schedule.
    pub const fn slot_count(&self) -> usize {
        self.slot_count
    }

    /// Whether any cost component still has an unknown/infinite upper bound.
    pub fn requires_concrete_rescore(&self) -> bool {
        [
            self.flops,
            self.unary_or_product_work,
            self.intermediate_elements,
            self.peak_live_temporary_elements,
            self.total_intermediate_traffic_elements,
            self.layout_or_packing_traffic_elements,
            self.broadcast_amplification_elements,
        ]
        .into_iter()
        .any(EinsumCostBound::requires_concrete_shape)
    }

    fn compare(&self, other: &Self) -> Ordering {
        [
            (self.flops, other.flops),
            (self.unary_or_product_work, other.unary_or_product_work),
            (self.intermediate_elements, other.intermediate_elements),
            (
                self.peak_live_temporary_elements,
                other.peak_live_temporary_elements,
            ),
            (
                self.total_intermediate_traffic_elements,
                other.total_intermediate_traffic_elements,
            ),
            (
                self.layout_or_packing_traffic_elements,
                other.layout_or_packing_traffic_elements,
            ),
            (
                self.broadcast_amplification_elements,
                other.broadcast_amplification_elements,
            ),
        ]
        .into_iter()
        .map(|(left, right)| compare_bound(left, right))
        .find(|ordering| *ordering != Ordering::Equal)
        .unwrap_or_else(|| self.slot_count.cmp(&other.slot_count))
    }
}

/// Exact cost after resolving concrete runtime shapes and an element width.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EinsumResolvedContractionCost {
    flops: u128,
    unary_or_product_work: u128,
    intermediate_elements: u128,
    intermediate_bytes: u128,
    peak_live_temporary_bytes: u128,
    total_intermediate_traffic_bytes: u128,
    layout_or_packing_traffic_bytes: u128,
    broadcast_amplification_elements: u128,
    slot_count: usize,
}

impl EinsumResolvedContractionCost {
    /// Total scalar arithmetic operations.
    pub const fn flops(&self) -> u128 {
        self.flops
    }

    /// Leaf-local reduction and K-free product work.
    pub const fn unary_or_product_work(&self) -> u128 {
        self.unary_or_product_work
    }

    /// Sum of temporary element counts.
    pub const fn intermediate_elements(&self) -> u128 {
        self.intermediate_elements
    }

    /// Sum of temporary value bytes.
    pub const fn intermediate_bytes(&self) -> u128 {
        self.intermediate_bytes
    }

    /// Maximum simultaneously live temporary bytes.
    pub const fn peak_live_temporary_bytes(&self) -> u128 {
        self.peak_live_temporary_bytes
    }

    /// Temporary writes plus reads.
    pub const fn total_intermediate_traffic_bytes(&self) -> u128 {
        self.total_intermediate_traffic_bytes
    }

    /// Estimated permutation/diagonal materialization traffic.
    pub const fn layout_or_packing_traffic_bytes(&self) -> u128 {
        self.layout_or_packing_traffic_bytes
    }

    /// Extra logical elements caused by batch broadcasting.
    pub const fn broadcast_amplification_elements(&self) -> u128 {
        self.broadcast_amplification_elements
    }

    /// Reusable temporary slot count.
    pub const fn slot_count(&self) -> usize {
        self.slot_count
    }

    fn compare(&self, other: &Self) -> Ordering {
        [
            self.flops.cmp(&other.flops),
            self.unary_or_product_work.cmp(&other.unary_or_product_work),
            self.intermediate_elements.cmp(&other.intermediate_elements),
            self.intermediate_bytes.cmp(&other.intermediate_bytes),
            self.peak_live_temporary_bytes
                .cmp(&other.peak_live_temporary_bytes),
            self.total_intermediate_traffic_bytes
                .cmp(&other.total_intermediate_traffic_bytes),
            self.layout_or_packing_traffic_bytes
                .cmp(&other.layout_or_packing_traffic_bytes),
            self.broadcast_amplification_elements
                .cmp(&other.broadcast_amplification_elements),
            self.slot_count.cmp(&other.slot_count),
        ]
        .into_iter()
        .find(|ordering| *ordering != Ordering::Equal)
        .unwrap_or(Ordering::Equal)
    }
}

/// A leaf-local unary reduction performed at the lowest node containing every
/// occurrence of its axes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EinsumUnaryReductionPlan {
    input: EinsumValueId,
    output: EinsumValueId,
    reduction_axes: Vec<EinsumAxis>,
    input_axes: Vec<EinsumAxis>,
    output_axes: Vec<EinsumAxis>,
    input_elements: EinsumCostBound,
    output_elements: EinsumCostBound,
    reduction_elements: EinsumCostBound,
}

impl EinsumUnaryReductionPlan {
    /// Source value.
    pub const fn input(&self) -> EinsumValueId {
        self.input
    }

    /// Reduced value.
    pub const fn output(&self) -> EinsumValueId {
        self.output
    }

    /// Locally reduced axes.
    pub fn reduction_axes(&self) -> &[EinsumAxis] {
        &self.reduction_axes
    }

    /// Source logical-axis order.
    pub fn input_axes(&self) -> &[EinsumAxis] {
        &self.input_axes
    }

    /// Result logical-axis order.
    pub fn output_axes(&self) -> &[EinsumAxis] {
        &self.output_axes
    }

    /// Source elements after diagonal folding.
    pub const fn input_elements(&self) -> EinsumCostBound {
        self.input_elements
    }

    /// Result elements.
    pub const fn output_elements(&self) -> EinsumCostBound {
        self.output_elements
    }

    /// Product of locally reduced extents.
    pub const fn reduction_elements(&self) -> EinsumCostBound {
        self.reduction_elements
    }
}

/// One ordered binary contraction/product node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EinsumBinaryLowering {
    /// General index-program product/contraction.
    GenericNative,
    /// Named-axis contraction that may be optimized as GEMM/BMM.
    GemmCompatible,
}

/// One ordered binary contraction/product node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EinsumBinaryContractionPlan {
    left: EinsumValueId,
    right: EinsumValueId,
    output: EinsumValueId,
    left_leaf_inputs: Vec<usize>,
    right_leaf_inputs: Vec<usize>,
    left_value_axes: Vec<EinsumAxis>,
    right_value_axes: Vec<EinsumAxis>,
    batch_axes: Vec<EinsumAxis>,
    left_free_axes: Vec<EinsumAxis>,
    contract_axes: Vec<EinsumAxis>,
    right_free_axes: Vec<EinsumAxis>,
    left_axis_order: Vec<Option<usize>>,
    right_axis_order: Vec<Option<usize>>,
    left_virtual_singleton_axes: Vec<EinsumAxis>,
    right_virtual_singleton_axes: Vec<EinsumAxis>,
    canonical_output_axes: Vec<EinsumAxis>,
    output_permutation: Vec<usize>,
    geometry: EinsumGemmGeometry,
    left_requires_packing: bool,
    right_requires_packing: bool,
    lowering: EinsumBinaryLowering,
    left_elements: EinsumCostBound,
    right_elements: EinsumCostBound,
    output_elements: EinsumCostBound,
}

impl EinsumBinaryContractionPlan {
    /// Left candidate value.
    pub const fn left(&self) -> EinsumValueId {
        self.left
    }

    /// Right candidate value.
    pub const fn right(&self) -> EinsumValueId {
        self.right
    }

    /// Produced candidate value.
    pub const fn output(&self) -> EinsumValueId {
        self.output
    }

    /// Original input leaves represented by the left value.
    pub fn left_leaf_inputs(&self) -> &[usize] {
        &self.left_leaf_inputs
    }

    /// Original input leaves represented by the right value.
    pub fn right_leaf_inputs(&self) -> &[usize] {
        &self.right_leaf_inputs
    }

    /// Left value's logical-axis order before canonicalization.
    pub fn left_value_axes(&self) -> &[EinsumAxis] {
        &self.left_value_axes
    }

    /// Right value's logical-axis order before canonicalization.
    pub fn right_value_axes(&self) -> &[EinsumAxis] {
        &self.right_value_axes
    }

    /// Canonical B/BMM batch axes.
    pub fn batch_axes(&self) -> &[EinsumAxis] {
        &self.batch_axes
    }

    /// Axes flattened into M.
    pub fn left_free_axes(&self) -> &[EinsumAxis] {
        &self.left_free_axes
    }

    /// Axes eliminated at this exact tree node and flattened into K.
    pub fn contract_axes(&self) -> &[EinsumAxis] {
        &self.contract_axes
    }

    /// Axes flattened into N.
    pub fn right_free_axes(&self) -> &[EinsumAxis] {
        &self.right_free_axes
    }

    /// Target left layout `[batch..., M..., K...]`.
    pub fn left_axis_order(&self) -> &[Option<usize>] {
        &self.left_axis_order
    }

    /// Target right layout `[batch..., K..., N...]`.
    pub fn right_axis_order(&self) -> &[Option<usize>] {
        &self.right_axis_order
    }

    /// Batch axes represented by virtual singleton dimensions on the left.
    pub fn left_virtual_singleton_axes(&self) -> &[EinsumAxis] {
        &self.left_virtual_singleton_axes
    }

    /// Batch axes represented by virtual singleton dimensions on the right.
    pub fn right_virtual_singleton_axes(&self) -> &[EinsumAxis] {
        &self.right_virtual_singleton_axes
    }

    /// Canonical result order `[batch..., M..., N...]`.
    pub fn canonical_output_axes(&self) -> &[EinsumAxis] {
        &self.canonical_output_axes
    }

    /// Requested result axis to canonical-result axis. Intermediate nodes use
    /// the identity permutation.
    pub fn output_permutation(&self) -> &[usize] {
        &self.output_permutation
    }

    /// Statically known B/M/K/N geometry.
    pub const fn geometry(&self) -> &EinsumGemmGeometry {
        &self.geometry
    }

    /// Whether the EP-neutral model charges a left-side pack/materialization.
    pub const fn left_requires_packing(&self) -> bool {
        self.left_requires_packing
    }

    /// Whether the EP-neutral model charges a right-side pack/materialization.
    pub const fn right_requires_packing(&self) -> bool {
        self.right_requires_packing
    }

    /// Semantic generic lowering or optional GEMM-compatible subtype.
    pub const fn lowering(&self) -> EinsumBinaryLowering {
        self.lowering
    }

    /// Left value elements before virtual broadcast.
    pub const fn left_elements(&self) -> EinsumCostBound {
        self.left_elements
    }

    /// Right value elements before virtual broadcast.
    pub const fn right_elements(&self) -> EinsumCostBound {
        self.right_elements
    }

    /// Canonical result elements.
    pub const fn output_elements(&self) -> EinsumCostBound {
        self.output_elements
    }
}

/// One step in a candidate's evaluation schedule.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EinsumContractionTreeStep {
    /// Leaf-local reduction.
    UnaryReduction(EinsumUnaryReductionPlan),
    /// Ordered binary contraction/product.
    BinaryContraction(Box<EinsumBinaryContractionPlan>),
}

impl EinsumContractionTreeStep {
    /// Produced value.
    pub const fn output(&self) -> EinsumValueId {
        match self {
            Self::UnaryReduction(plan) => plan.output,
            Self::BinaryContraction(plan) => plan.output,
        }
    }

    fn inputs(&self) -> Vec<EinsumValueId> {
        match self {
            Self::UnaryReduction(plan) => vec![plan.input],
            Self::BinaryContraction(plan) => vec![plan.left, plan.right],
        }
    }
}

/// Storage policy for a non-final contraction value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EinsumTemporaryStoragePolicy {
    /// Store in the typed plan's accumulator/intermediate dtype. In
    /// particular, f16/bf16 inputs use f32 intermediates and narrow only once
    /// at the final output.
    Accumulator,
}

/// One temporary value and its reusable-slot liveness interval.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EinsumTemporaryValuePlan {
    value: EinsumValueId,
    slot: usize,
    birth_step: usize,
    last_use_step: usize,
    axes: Vec<EinsumAxis>,
    leaf_inputs: Vec<usize>,
    global_iteration_axis_indices: Vec<usize>,
    storage_policy: EinsumTemporaryStoragePolicy,
    elements: EinsumCostBound,
}

impl EinsumTemporaryValuePlan {
    /// Candidate value stored in the slot.
    pub const fn value(&self) -> EinsumValueId {
        self.value
    }

    /// Reusable slot number.
    pub const fn slot(&self) -> usize {
        self.slot
    }

    /// Step that produces the value.
    pub const fn birth_step(&self) -> usize {
        self.birth_step
    }

    /// Last step that consumes the value.
    pub const fn last_use_step(&self) -> usize {
        self.last_use_step
    }

    /// Logical axes stored in the temporary.
    pub fn axes(&self) -> &[EinsumAxis] {
        &self.axes
    }

    /// Original leaves represented by this temporary.
    pub fn leaf_inputs(&self) -> &[usize] {
        &self.leaf_inputs
    }

    /// Map each live axis to its canonical global iteration-axis index.
    pub fn global_iteration_axis_indices(&self) -> &[usize] {
        &self.global_iteration_axis_indices
    }

    /// Backend-neutral intermediate storage policy.
    pub const fn storage_policy(&self) -> EinsumTemporaryStoragePolicy {
        self.storage_policy
    }

    /// Statically known element count.
    pub const fn elements(&self) -> EinsumCostBound {
        self.elements
    }
}

/// A fully planned, supported candidate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EinsumSupportedContractionTreeCandidate {
    steps: Vec<EinsumContractionTreeStep>,
    temporaries: Vec<EinsumTemporaryValuePlan>,
    final_output: EinsumValueId,
    final_output_permutation: Vec<usize>,
    cost: EinsumContractionCost,
}

impl EinsumSupportedContractionTreeCandidate {
    /// Evaluation steps in execution order.
    pub fn steps(&self) -> &[EinsumContractionTreeStep] {
        &self.steps
    }

    /// Linear-scan temporary-slot/liveness schedule.
    pub fn temporaries(&self) -> &[EinsumTemporaryValuePlan] {
        &self.temporaries
    }

    /// Final produced value.
    pub const fn final_output(&self) -> EinsumValueId {
        self.final_output
    }

    /// Requested output axis to root canonical-result axis.
    pub fn final_output_permutation(&self) -> &[usize] {
        &self.final_output_permutation
    }

    /// Deterministic planning-time cost.
    pub const fn cost(&self) -> &EinsumContractionCost {
        &self.cost
    }
}

/// Supported or fail-closed status for one enumerated ordered tree.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EinsumContractionTreeCandidatePlan {
    /// Candidate is structurally lowerable and has checked cost bounds.
    Supported(Box<EinsumSupportedContractionTreeCandidate>),
    /// Candidate is semantically legal but cannot be represented safely.
    Unsupported(EinsumContractionTreeCandidateUnsupportedReason),
}

/// One semantically legal ordered binary tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EinsumContractionTreeCandidate {
    id: EinsumContractionTreeCandidateId,
    plan: EinsumContractionTreeCandidatePlan,
}

impl EinsumContractionTreeCandidate {
    /// Stable lexicographic candidate identifier.
    pub const fn id(&self) -> &EinsumContractionTreeCandidateId {
        &self.id
    }

    /// Supported or fail-closed candidate plan.
    pub const fn plan(&self) -> &EinsumContractionTreeCandidatePlan {
        &self.plan
    }

    /// Supported candidate details.
    pub fn supported(&self) -> Option<&EinsumSupportedContractionTreeCandidate> {
        match &self.plan {
            EinsumContractionTreeCandidatePlan::Supported(plan) => Some(plan.as_ref()),
            EinsumContractionTreeCandidatePlan::Unsupported(_) => None,
        }
    }

    /// Structured candidate refusal.
    pub const fn unsupported_reason(
        &self,
    ) -> Option<&EinsumContractionTreeCandidateUnsupportedReason> {
        match &self.plan {
            EinsumContractionTreeCandidatePlan::Supported(_) => None,
            EinsumContractionTreeCandidatePlan::Unsupported(reason) => Some(reason),
        }
    }
}

/// Canonical bounded ordered-tree plan for a multi-input einsum.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EinsumContractionTreePlan {
    arity: usize,
    leaf_values: Vec<EinsumValueId>,
    candidates: Vec<EinsumContractionTreeCandidate>,
    preferred_candidate: Option<usize>,
    requires_concrete_rescore: bool,
    quality: EinsumPlannerQuality,
    fallback_reason: Option<EinsumPlannerFallbackReason>,
    usage: EinsumPlannerUsage,
    logical_order: Vec<EinsumAxis>,
    output_axes: Vec<EinsumAxis>,
    reduction_axes: BTreeSet<EinsumAxis>,
    occurrence_inputs: BTreeMap<EinsumAxis, BTreeSet<usize>>,
}

impl EinsumContractionTreePlan {
    /// Number of original operands.
    pub const fn arity(&self) -> usize {
        self.arity
    }

    /// Stable leaf value IDs, equal to input indices.
    pub fn leaf_values(&self) -> &[EinsumValueId] {
        &self.leaf_values
    }

    /// Materialized bounded candidates, sorted by stable ID. Exact planning
    /// contains every ordered tree; greedy planning contains one; generic
    /// fallback contains none.
    pub fn candidates(&self) -> &[EinsumContractionTreeCandidate] {
        &self.candidates
    }

    /// Preferred candidate under the planning-time EP-neutral cost model.
    pub fn preferred_candidate(&self) -> Option<&EinsumContractionTreeCandidate> {
        self.preferred_candidate
            .and_then(|index| self.candidates.get(index))
    }

    /// Whether unknown symbolic dimensions require concrete runtime re-score.
    pub const fn requires_concrete_rescore(&self) -> bool {
        self.requires_concrete_rescore
    }

    /// Whether planning was exact subset DP or bounded deterministic greedy.
    pub const fn quality(&self) -> EinsumPlannerQuality {
        self.quality
    }

    /// Structured reason for [`EinsumPlannerQuality::GenericNativeFallback`].
    pub const fn fallback_reason(&self) -> Option<EinsumPlannerFallbackReason> {
        self.fallback_reason
    }

    /// Actual bounded-planner resource use.
    pub const fn usage(&self) -> EinsumPlannerUsage {
        self.usage
    }

    pub(super) fn resolve(
        &self,
        dimensions: &BTreeMap<EinsumAxis, usize>,
        input_shapes: &[&[usize]],
        operands: &[EinsumOperandPlan],
        element_size: usize,
    ) -> EinsumConcreteContractionTreePlan {
        let replanned;
        let structural_candidates = if self.quality == EinsumPlannerQuality::DeterministicGreedy {
            let concrete_dimensions = dimensions
                .iter()
                .map(|(&axis, &size)| (axis, EinsumDimension::Static(size)))
                .collect::<BTreeMap<_, _>>();
            let planned = greedy_tree(
                self.arity,
                &self.logical_order,
                &concrete_dimensions,
                &self.occurrence_inputs,
                &self.reduction_axes,
                self.usage.budget,
            );
            let output_axis_set = self.output_axes.iter().copied().collect();
            replanned = planned
                .trees
                .into_iter()
                .map(|tree| {
                    let id = EinsumContractionTreeCandidateId(tree.id().to_owned());
                    let plan = match build_candidate(
                        &tree,
                        operands,
                        &self.logical_order,
                        &concrete_dimensions,
                        &self.occurrence_inputs,
                        &output_axis_set,
                        &self.reduction_axes,
                        &self.output_axes,
                    ) {
                        Ok(plan) => EinsumContractionTreeCandidatePlan::Supported(Box::new(plan)),
                        Err(reason) => EinsumContractionTreeCandidatePlan::Unsupported(reason),
                    };
                    EinsumContractionTreeCandidate { id, plan }
                })
                .collect::<Vec<_>>();
            &replanned
        } else {
            &self.candidates
        };
        let mut candidates = Vec::with_capacity(structural_candidates.len());
        for candidate in structural_candidates {
            let concrete = match candidate.supported() {
                Some(supported) => {
                    match concretize_candidate(supported, dimensions, input_shapes, operands)
                        .and_then(|(steps, temporaries)| {
                            score_candidate(&steps, &temporaries, element_size)
                                .map(|score| (steps, score))
                        }) {
                        Ok((steps, (_, Some(resolved_cost)))) => {
                            let geometries = steps
                                .iter()
                                .filter_map(|step| match step {
                                    EinsumContractionTreeStep::BinaryContraction(binary) => {
                                        Some(resolve_binary_geometry(binary, dimensions))
                                    }
                                    EinsumContractionTreeStep::UnaryReduction(_) => None,
                                })
                                .collect::<Result<Vec<_>, _>>();
                            match geometries {
                                Ok(binary_geometries) => EinsumConcreteContractionTreeCandidate {
                                    id: candidate.id.clone(),
                                    cost: Some(resolved_cost),
                                    binary_geometries,
                                    unsupported_reason: None,
                                },
                                Err(reason) => EinsumConcreteContractionTreeCandidate {
                                    id: candidate.id.clone(),
                                    cost: None,
                                    binary_geometries: Vec::new(),
                                    unsupported_reason: Some(reason),
                                },
                            }
                        }
                        Ok((_, (_, None))) => {
                            unreachable!("concrete dimensions produce exact costs")
                        }
                        Err(reason) => EinsumConcreteContractionTreeCandidate {
                            id: candidate.id.clone(),
                            cost: None,
                            binary_geometries: Vec::new(),
                            unsupported_reason: Some(reason),
                        },
                    }
                }
                None => EinsumConcreteContractionTreeCandidate {
                    id: candidate.id.clone(),
                    cost: None,
                    binary_geometries: Vec::new(),
                    unsupported_reason: candidate.unsupported_reason().cloned(),
                },
            };
            candidates.push(concrete);
        }
        let preferred_candidate = candidates
            .iter()
            .enumerate()
            .filter(|(_, candidate)| candidate.cost.is_some())
            .min_by(|(_, left), (_, right)| {
                left.cost
                    .as_ref()
                    .expect("filtered")
                    .compare(right.cost.as_ref().expect("filtered"))
                    .then_with(|| left.id.cmp(&right.id))
            })
            .map(|(index, _)| index);
        EinsumConcreteContractionTreePlan {
            candidates,
            preferred_candidate,
        }
    }
}

/// One concrete runtime candidate re-score.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EinsumConcreteContractionTreeCandidate {
    id: EinsumContractionTreeCandidateId,
    cost: Option<EinsumResolvedContractionCost>,
    binary_geometries: Vec<EinsumConcreteGemmGeometry>,
    unsupported_reason: Option<EinsumContractionTreeCandidateUnsupportedReason>,
}

impl EinsumConcreteContractionTreeCandidate {
    /// Stable candidate identifier.
    pub const fn id(&self) -> &EinsumContractionTreeCandidateId {
        &self.id
    }

    /// Exact checked cost, or `None` when concrete arithmetic overflowed.
    pub const fn cost(&self) -> Option<&EinsumResolvedContractionCost> {
        self.cost.as_ref()
    }

    /// Concrete B/M/K/N geometry for each binary step in execution order.
    pub fn binary_geometries(&self) -> &[EinsumConcreteGemmGeometry] {
        &self.binary_geometries
    }

    /// Concrete refusal, if any.
    pub const fn unsupported_reason(
        &self,
    ) -> Option<&EinsumContractionTreeCandidateUnsupportedReason> {
        self.unsupported_reason.as_ref()
    }
}

/// Concrete runtime re-score for every candidate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EinsumConcreteContractionTreePlan {
    candidates: Vec<EinsumConcreteContractionTreeCandidate>,
    preferred_candidate: Option<usize>,
}

impl EinsumConcreteContractionTreePlan {
    /// Concrete candidates in the same stable order as the structural plan.
    pub fn candidates(&self) -> &[EinsumConcreteContractionTreeCandidate] {
        &self.candidates
    }

    /// Lowest-cost concrete candidate, including the stable ID tie-break.
    pub fn preferred_candidate(&self) -> Option<&EinsumConcreteContractionTreeCandidate> {
        self.preferred_candidate
            .and_then(|index| self.candidates.get(index))
    }

    /// Lowest-cost candidate whose peak live temporary bytes fit `ceiling`.
    ///
    /// `None` is not a semantic rejection: the caller must select the
    /// mandatory generic-native/tiled plan instead.
    pub fn preferred_candidate_with_memory_ceiling(
        &self,
        ceiling: u128,
    ) -> Option<&EinsumConcreteContractionTreeCandidate> {
        self.candidates
            .iter()
            .filter(|candidate| {
                candidate
                    .cost()
                    .is_some_and(|cost| cost.peak_live_temporary_bytes() <= ceiling)
            })
            .min_by(|left, right| {
                left.cost()
                    .expect("filtered")
                    .compare(right.cost().expect("filtered"))
                    .then_with(|| left.id().cmp(right.id()))
            })
    }
}

#[derive(Clone)]
struct ValueDescriptor {
    id: EinsumValueId,
    leaves: BTreeSet<usize>,
    axes: Vec<EinsumAxis>,
    axis_dimensions: Vec<EinsumDimension>,
    elements: EinsumCostBound,
    requires_materialization: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum TreeExprKind {
    Leaf(usize),
    Merge(Box<TreeExpr>, Box<TreeExpr>),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TreeExpr {
    kind: TreeExprKind,
    id: String,
    leaves: BTreeSet<usize>,
    depth: usize,
    metadata_units: usize,
}

impl TreeExpr {
    fn leaf(input: usize) -> Self {
        Self {
            kind: TreeExprKind::Leaf(input),
            id: input.to_string(),
            leaves: BTreeSet::from([input]),
            depth: 0,
            metadata_units: 2,
        }
    }

    fn merge(left: Self, right: Self) -> Result<Self, EinsumPlannerFallbackReason> {
        let depth = left
            .depth
            .max(right.depth)
            .checked_add(1)
            .ok_or(EinsumPlannerFallbackReason::PlanningArithmeticOverflow)?;
        if depth > EinsumPlannerBudget::MAX_CONTRACTION_TREE_DEPTH {
            return Err(EinsumPlannerFallbackReason::MaximumDepthExceeded);
        }
        let id_capacity = left
            .id
            .len()
            .checked_add(right.id.len())
            .and_then(|value| value.checked_add(3))
            .ok_or(EinsumPlannerFallbackReason::PlanningArithmeticOverflow)?;
        let mut id = String::with_capacity(id_capacity);
        id.push('(');
        id.push_str(&left.id);
        id.push(',');
        id.push_str(&right.id);
        id.push(')');
        let leaves: BTreeSet<_> = left.leaves.union(&right.leaves).copied().collect();
        let metadata_units = left
            .metadata_units
            .checked_add(right.metadata_units)
            .and_then(|value| value.checked_add(id.len()))
            .and_then(|value| value.checked_add(leaves.len()))
            .and_then(|value| value.checked_add(1))
            .ok_or(EinsumPlannerFallbackReason::PlanningArithmeticOverflow)?;
        Ok(Self {
            kind: TreeExprKind::Merge(Box::new(left), Box::new(right)),
            id,
            leaves,
            depth,
            metadata_units,
        })
    }

    fn id(&self) -> &str {
        &self.id
    }
}

struct PlannedTrees {
    trees: Vec<TreeExpr>,
    quality: EinsumPlannerQuality,
    states: usize,
    candidates: usize,
    work: usize,
    metadata_units: usize,
    max_depth: usize,
    candidate_id_bytes: usize,
    fallback_reason: Option<EinsumPlannerFallbackReason>,
}

impl PlannedTrees {
    fn generic_native_fallback(reason: EinsumPlannerFallbackReason) -> Self {
        Self {
            trees: Vec::new(),
            quality: EinsumPlannerQuality::GenericNativeFallback,
            states: 0,
            candidates: 0,
            work: 0,
            metadata_units: 0,
            max_depth: 0,
            candidate_id_bytes: 0,
            fallback_reason: Some(reason),
        }
    }
}

pub(super) fn build_contraction_tree(
    operands: &[EinsumOperandPlan],
    logical_axes: &[EinsumLogicalAxis],
    output_axes: &[EinsumAxis],
    reduction_axes: &[EinsumAxis],
    budget: EinsumPlannerBudget,
) -> EinsumContractionTreePlan {
    debug_assert!(operands.len() >= 2);
    let dimensions: BTreeMap<_, _> = logical_axes
        .iter()
        .map(|logical| (logical.axis(), logical.dimension()))
        .collect();
    let occurrence_inputs: BTreeMap<_, BTreeSet<_>> = logical_axes
        .iter()
        .map(|logical| {
            (
                logical.axis(),
                logical
                    .occurrences()
                    .iter()
                    .map(|reference| reference.input())
                    .collect(),
            )
        })
        .collect();
    let logical_order: Vec<_> = logical_axes.iter().map(EinsumLogicalAxis::axis).collect();
    let reduction_axes: BTreeSet<_> = reduction_axes.iter().copied().collect();
    let output_axis_set: BTreeSet<_> = output_axes.iter().copied().collect();

    let planned = plan_trees(
        operands.len(),
        &logical_order,
        &dimensions,
        &occurrence_inputs,
        &reduction_axes,
        budget,
    );
    let mut candidates = planned
        .trees
        .into_iter()
        .map(|tree| {
            let id = EinsumContractionTreeCandidateId(tree.id().to_owned());
            let plan = match build_candidate(
                &tree,
                operands,
                &logical_order,
                &dimensions,
                &occurrence_inputs,
                &output_axis_set,
                &reduction_axes,
                output_axes,
            ) {
                Ok(plan) => EinsumContractionTreeCandidatePlan::Supported(Box::new(plan)),
                Err(reason) => EinsumContractionTreeCandidatePlan::Unsupported(reason),
            };
            EinsumContractionTreeCandidate { id, plan }
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| left.id.cmp(&right.id));
    let preferred_candidate = candidates
        .iter()
        .enumerate()
        .filter_map(|(index, candidate)| {
            candidate
                .supported()
                .map(|supported| (index, supported.cost()))
        })
        .min_by(|(left_index, left), (right_index, right)| {
            left.compare(right)
                .then_with(|| candidates[*left_index].id.cmp(&candidates[*right_index].id))
        })
        .map(|(index, _)| index);
    let requires_concrete_rescore = candidates.iter().any(|candidate| {
        candidate
            .supported()
            .is_some_and(|candidate| candidate.cost.requires_concrete_rescore())
    });
    let axis_count = logical_order.len();
    let (stored_logical_order, stored_output_axes, stored_reduction_axes, stored_occurrence_inputs) =
        if planned.quality == EinsumPlannerQuality::GenericNativeFallback {
            (Vec::new(), Vec::new(), BTreeSet::new(), BTreeMap::new())
        } else {
            (
                logical_order,
                output_axes.to_vec(),
                reduction_axes,
                occurrence_inputs,
            )
        };

    EinsumContractionTreePlan {
        arity: operands.len(),
        leaf_values: (0..operands.len()).map(EinsumValueId).collect(),
        candidates,
        preferred_candidate,
        requires_concrete_rescore,
        quality: planned.quality,
        fallback_reason: planned.fallback_reason,
        usage: EinsumPlannerUsage {
            states: planned.states,
            candidates: planned.candidates,
            axes: axis_count,
            work: planned.work,
            metadata_units: planned.metadata_units,
            max_depth: planned.max_depth,
            candidate_id_bytes: planned.candidate_id_bytes,
            budget,
        },
        logical_order: stored_logical_order,
        output_axes: stored_output_axes,
        reduction_axes: stored_reduction_axes,
        occurrence_inputs: stored_occurrence_inputs,
    }
}

fn plan_trees(
    arity: usize,
    logical_order: &[EinsumAxis],
    dimensions: &BTreeMap<EinsumAxis, EinsumDimension>,
    occurrence_inputs: &BTreeMap<EinsumAxis, BTreeSet<usize>>,
    reduction_axes: &BTreeSet<EinsumAxis>,
    budget: EinsumPlannerBudget,
) -> PlannedTrees {
    let exact_state_count = 1usize
        .checked_shl(arity as u32)
        .and_then(|n| n.checked_sub(1));
    if arity
        <= budget
            .exact_operand_limit
            .min(EinsumPlannerBudget::MAX_EXACT_TREE_OPERANDS)
        && logical_order.len() <= budget.max_exact_axes
        && exact_state_count.is_some_and(|states| states <= budget.max_states)
        && let Some(exact) = enumerate_exact_trees(arity, budget)
    {
        return exact;
    }

    greedy_tree(
        arity,
        logical_order,
        dimensions,
        occurrence_inputs,
        reduction_axes,
        budget,
    )
}

fn enumerate_exact_trees(arity: usize, budget: EinsumPlannerBudget) -> Option<PlannedTrees> {
    let full = 1u64.checked_shl(arity as u32)?.checked_sub(1)?;
    let mut states = BTreeMap::<u64, Vec<TreeExpr>>::new();
    let metadata_limit = budget.exact_metadata_units_limit()?;
    let mut metadata_units = 0usize;
    for input in 0..arity {
        let leaf = TreeExpr::leaf(input);
        metadata_units = metadata_units.checked_add(leaf.metadata_units)?;
        if metadata_units > metadata_limit {
            return None;
        }
        states.insert(1u64 << input, vec![leaf]);
    }
    let mut candidate_count = 0usize;
    for size in 2..=arity {
        for subset in 1..=full {
            if subset.count_ones() as usize != size {
                continue;
            }
            if states.len() >= budget.max_states {
                return None;
            }
            let mut trees = Vec::new();
            let mut left = (subset - 1) & subset;
            while left != 0 {
                let right = subset ^ left;
                if right != 0
                    && let (Some(left_trees), Some(right_trees)) =
                        (states.get(&left), states.get(&right))
                {
                    for left_tree in left_trees {
                        for right_tree in right_trees {
                            candidate_count = candidate_count.checked_add(1)?;
                            if candidate_count > budget.max_candidates {
                                return None;
                            }
                            let tree =
                                TreeExpr::merge(left_tree.clone(), right_tree.clone()).ok()?;
                            metadata_units = metadata_units.checked_add(tree.metadata_units)?;
                            if metadata_units > metadata_limit {
                                return None;
                            }
                            trees.push(tree);
                        }
                    }
                }
                left = (left - 1) & subset;
            }
            trees.sort_by(|left, right| left.id.cmp(&right.id));
            trees.dedup_by(|left, right| left == right);
            states.insert(subset, trees);
        }
    }
    let trees = states.remove(&full).unwrap_or_default();
    let candidate_id_bytes = trees
        .iter()
        .try_fold(0usize, |sum, tree| sum.checked_add(tree.id.len()))?;
    let max_depth = trees.iter().map(|tree| tree.depth).max().unwrap_or(0);
    let work = candidate_count.checked_add(metadata_units)?;
    Some(PlannedTrees {
        trees,
        quality: EinsumPlannerQuality::ExactSubsetDp,
        states: states.len() + 1,
        candidates: candidate_count,
        work,
        metadata_units,
        max_depth,
        candidate_id_bytes,
        fallback_reason: None,
    })
}

fn heuristic_metadata_bound(arity: usize, axes: usize) -> Option<usize> {
    let square = arity.checked_mul(arity)?;
    let digits = arity.max(1).ilog10() as usize + 1;
    let identifiers = square.checked_mul(digits.checked_add(3)?)?;
    let leaf_membership = arity
        .checked_mul(arity.checked_add(1)?)?
        .checked_div(2)?
        .checked_mul(4)?;
    let steps = arity.checked_mul(2)?.checked_sub(1)?;
    let lowering_axes = steps.checked_mul(axes.max(1))?.checked_mul(16)?;
    identifiers
        .checked_add(leaf_membership)?
        .checked_add(lowering_axes)
}

fn heuristic_score_units(
    logical_order: &[EinsumAxis],
    occurrence_inputs: &BTreeMap<EinsumAxis, BTreeSet<usize>>,
) -> Option<usize> {
    occurrence_inputs
        .values()
        .try_fold(logical_order.len().checked_add(1)?, |sum, inputs| {
            sum.checked_add(inputs.len())
        })
}

fn greedy_tree(
    arity: usize,
    logical_order: &[EinsumAxis],
    dimensions: &BTreeMap<EinsumAxis, EinsumDimension>,
    occurrence_inputs: &BTreeMap<EinsumAxis, BTreeSet<usize>>,
    reduction_axes: &BTreeSet<EinsumAxis>,
    budget: EinsumPlannerBudget,
) -> PlannedTrees {
    let Some(metadata_units) = heuristic_metadata_bound(arity, logical_order.len()) else {
        return PlannedTrees::generic_native_fallback(
            EinsumPlannerFallbackReason::PlanningArithmeticOverflow,
        );
    };
    if metadata_units > budget.max_heuristic_candidates {
        return PlannedTrees::generic_native_fallback(
            EinsumPlannerFallbackReason::WorkOrMetadataBudgetExceeded,
        );
    }
    let Some(score_units) = heuristic_score_units(logical_order, occurrence_inputs) else {
        return PlannedTrees::generic_native_fallback(
            EinsumPlannerFallbackReason::PlanningArithmeticOverflow,
        );
    };
    let score_work_limit = budget.max_heuristic_candidates - metadata_units;
    let mut score_work = 0usize;
    let mut forest = (0..arity).map(TreeExpr::leaf).collect::<Vec<_>>();
    let mut candidates = 0usize;
    let mut states = 1usize;
    while forest.len() > 1 {
        forest.sort_by(|left, right| {
            left.depth
                .cmp(&right.depth)
                .then_with(|| left.id.cmp(&right.id))
        });
        let mut best: Option<(GreedyScore, usize, usize)> = None;
        'pairs: for left in 0..forest.len() {
            for right in 0..forest.len() {
                if left == right {
                    continue;
                }
                let Some(next_work) = score_work.checked_add(score_units) else {
                    break 'pairs;
                };
                if next_work > score_work_limit {
                    break 'pairs;
                }
                score_work = next_work;
                candidates += 1;
                let score = greedy_merge_score(
                    &forest[left],
                    &forest[right],
                    logical_order,
                    dimensions,
                    occurrence_inputs,
                    reduction_axes,
                );
                if best
                    .as_ref()
                    .is_none_or(|(current, current_left, current_right)| {
                        compare_greedy_score(
                            &score,
                            &forest[left],
                            &forest[right],
                            current,
                            &forest[*current_left],
                            &forest[*current_right],
                        )
                        .is_lt()
                    })
                {
                    best = Some((score, left, right));
                }
            }
        }
        let (_, left, right) = best.unwrap_or((
            GreedyScore {
                output: EinsumCostBound::UnknownUpperBound,
                reduction: EinsumCostBound::UnknownUpperBound,
                depth: forest[0].depth.max(forest[1].depth) + 1,
            },
            0,
            1,
        ));
        let (high, low) = if left > right {
            (left, right)
        } else {
            (right, left)
        };
        let right_tree = forest.remove(high);
        let left_tree = forest.remove(low);
        let merged = if low == left {
            TreeExpr::merge(left_tree, right_tree)
        } else {
            TreeExpr::merge(right_tree, left_tree)
        };
        let merged = match merged {
            Ok(merged) => merged,
            Err(reason) => return PlannedTrees::generic_native_fallback(reason),
        };
        forest.push(merged);
        states += 1;
    }
    let max_depth = forest.first().map_or(0, |tree| tree.depth);
    let candidate_id_bytes = forest.first().map_or(0, |tree| tree.id.len());
    PlannedTrees {
        trees: forest,
        quality: EinsumPlannerQuality::DeterministicGreedy,
        states,
        candidates,
        work: metadata_units + score_work,
        metadata_units,
        max_depth,
        candidate_id_bytes,
        fallback_reason: None,
    }
}

#[derive(Clone, Copy)]
struct GreedyScore {
    output: EinsumCostBound,
    reduction: EinsumCostBound,
    depth: usize,
}

fn greedy_merge_score(
    left: &TreeExpr,
    right: &TreeExpr,
    logical_order: &[EinsumAxis],
    dimensions: &BTreeMap<EinsumAxis, EinsumDimension>,
    occurrence_inputs: &BTreeMap<EinsumAxis, BTreeSet<usize>>,
    reduction_axes: &BTreeSet<EinsumAxis>,
) -> GreedyScore {
    let contained = |inputs: &BTreeSet<usize>| {
        inputs
            .iter()
            .all(|input| left.leaves.contains(input) || right.leaves.contains(input))
    };
    let intersects = |inputs: &BTreeSet<usize>| {
        inputs
            .iter()
            .any(|input| left.leaves.contains(input) || right.leaves.contains(input))
    };
    let axis_product = |reduced: bool| {
        logical_order
            .iter()
            .copied()
            .filter(|axis| {
                let occurrences = &occurrence_inputs[axis];
                let is_reduced = reduction_axes.contains(axis) && contained(occurrences);
                if reduced {
                    is_reduced
                } else {
                    intersects(occurrences) && !is_reduced
                }
            })
            .try_fold(EinsumCostBound::Exact(1), |product, axis| {
                mul_bound(
                    product,
                    dimension_bound(dimensions[&axis]),
                    EinsumCostMetric::Geometry,
                )
            })
            .unwrap_or(EinsumCostBound::UnknownUpperBound)
    };
    GreedyScore {
        output: axis_product(false),
        reduction: axis_product(true),
        depth: left.depth.max(right.depth) + 1,
    }
}

fn compare_greedy_score(
    left: &GreedyScore,
    left_tree: &TreeExpr,
    right_tree: &TreeExpr,
    current: &GreedyScore,
    current_left: &TreeExpr,
    current_right: &TreeExpr,
) -> Ordering {
    compare_bound(left.output, current.output)
        .then_with(|| compare_bound(left.reduction, current.reduction))
        .then_with(|| left.depth.cmp(&current.depth))
        .then_with(|| left_tree.id.cmp(&current_left.id))
        .then_with(|| right_tree.id.cmp(&current_right.id))
}

#[allow(clippy::too_many_arguments)]
fn build_candidate(
    tree: &TreeExpr,
    operands: &[EinsumOperandPlan],
    logical_order: &[EinsumAxis],
    dimensions: &BTreeMap<EinsumAxis, EinsumDimension>,
    occurrence_inputs: &BTreeMap<EinsumAxis, BTreeSet<usize>>,
    output_axis_set: &BTreeSet<EinsumAxis>,
    reduction_axis_set: &BTreeSet<EinsumAxis>,
    output_axes: &[EinsumAxis],
) -> Result<EinsumSupportedContractionTreeCandidate, EinsumContractionTreeCandidateUnsupportedReason>
{
    let arity = operands.len();
    let mut descriptors = BTreeMap::new();
    for (input, operand) in operands.iter().enumerate() {
        let axes = operand
            .unique_axes()
            .iter()
            .map(|axis| axis.axis())
            .collect::<Vec<_>>();
        let axis_dimensions = operand
            .unique_axes()
            .iter()
            .map(|axis| axis.dimension())
            .collect::<Vec<_>>();
        let elements = dimension_values_product(&axis_dimensions)?;
        descriptors.insert(
            EinsumValueId(input),
            ValueDescriptor {
                id: EinsumValueId(input),
                leaves: BTreeSet::from([input]),
                axes,
                axis_dimensions,
                elements,
                requires_materialization: !operand.diagonal_axis_indices().is_empty(),
            },
        );
    }
    let mut steps = Vec::new();
    let mut next_value = arity;
    let final_value = build_tree_expr(
        tree,
        arity,
        &mut next_value,
        &mut steps,
        &mut descriptors,
        logical_order,
        dimensions,
        occurrence_inputs,
        output_axis_set,
        reduction_axis_set,
        output_axes,
    )?;
    let final_output_permutation = match steps.last() {
        Some(EinsumContractionTreeStep::BinaryContraction(binary)) => {
            binary.output_permutation.clone()
        }
        _ => Vec::new(),
    };
    let temporaries = schedule_temporaries(&steps, final_value.id, &descriptors, logical_order);
    let (cost, _) = score_candidate(&steps, &temporaries, 1)?;
    Ok(EinsumSupportedContractionTreeCandidate {
        steps,
        temporaries,
        final_output: final_value.id,
        final_output_permutation,
        cost,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_tree_expr(
    tree: &TreeExpr,
    arity: usize,
    next_value: &mut usize,
    steps: &mut Vec<EinsumContractionTreeStep>,
    descriptors: &mut BTreeMap<EinsumValueId, ValueDescriptor>,
    logical_order: &[EinsumAxis],
    dimensions: &BTreeMap<EinsumAxis, EinsumDimension>,
    occurrence_inputs: &BTreeMap<EinsumAxis, BTreeSet<usize>>,
    output_axis_set: &BTreeSet<EinsumAxis>,
    reduction_axis_set: &BTreeSet<EinsumAxis>,
    requested_output_axes: &[EinsumAxis],
) -> Result<ValueDescriptor, EinsumContractionTreeCandidateUnsupportedReason> {
    let mut traversal = vec![(tree, false)];
    let mut values = Vec::<ValueDescriptor>::new();
    while let Some((node, visited)) = traversal.pop() {
        match &node.kind {
            TreeExprKind::Leaf(input) => {
                let leaf = descriptors[&EinsumValueId(*input)].clone();
                let local_reductions = logical_order
                    .iter()
                    .copied()
                    .filter(|axis| {
                        reduction_axis_set.contains(axis)
                            && occurrence_inputs[axis].is_subset(&leaf.leaves)
                    })
                    .collect::<BTreeSet<_>>();
                if local_reductions.is_empty() {
                    values.push(leaf);
                    continue;
                }
                let output_id = EinsumValueId(*next_value);
                *next_value += 1;
                let output_axes = leaf
                    .axes
                    .iter()
                    .copied()
                    .filter(|axis| !local_reductions.contains(axis))
                    .collect::<Vec<_>>();
                let output_axis_dimensions = leaf
                    .axes
                    .iter()
                    .zip(&leaf.axis_dimensions)
                    .filter_map(|(axis, dimension)| {
                        (!local_reductions.contains(axis)).then_some(*dimension)
                    })
                    .collect::<Vec<_>>();
                let reduction_dimensions = leaf
                    .axes
                    .iter()
                    .zip(&leaf.axis_dimensions)
                    .filter_map(|(axis, dimension)| {
                        local_reductions.contains(axis).then_some(*dimension)
                    })
                    .collect::<Vec<_>>();
                let output_elements = dimension_values_product(&output_axis_dimensions)?;
                let reduction_elements = dimension_values_product(&reduction_dimensions)?;
                steps.push(EinsumContractionTreeStep::UnaryReduction(
                    EinsumUnaryReductionPlan {
                        input: leaf.id,
                        output: output_id,
                        reduction_axes: logical_order
                            .iter()
                            .copied()
                            .filter(|axis| local_reductions.contains(axis))
                            .collect(),
                        input_axes: leaf.axes.clone(),
                        output_axes: output_axes.clone(),
                        input_elements: leaf.elements,
                        output_elements,
                        reduction_elements,
                    },
                ));
                let output = ValueDescriptor {
                    id: output_id,
                    leaves: leaf.leaves,
                    axes: output_axes,
                    axis_dimensions: output_axis_dimensions,
                    elements: output_elements,
                    requires_materialization: false,
                };
                descriptors.insert(output_id, output.clone());
                values.push(output);
            }
            TreeExprKind::Merge(left, right) if !visited => {
                traversal.push((node, true));
                traversal.push((right, false));
                traversal.push((left, false));
            }
            TreeExprKind::Merge(_, _) => {
                let right = values
                    .pop()
                    .expect("iterative post-order lowering produced the right child");
                let left = values
                    .pop()
                    .expect("iterative post-order lowering produced the left child");
                let output_id = EinsumValueId(*next_value);
                *next_value += 1;
                let final_node = left.leaves.len() + right.leaves.len() == arity;
                let (binary, output) = build_binary(
                    left,
                    right,
                    output_id,
                    final_node,
                    logical_order,
                    dimensions,
                    occurrence_inputs,
                    output_axis_set,
                    reduction_axis_set,
                    requested_output_axes,
                )?;
                steps.push(EinsumContractionTreeStep::BinaryContraction(Box::new(
                    binary,
                )));
                descriptors.insert(output_id, output.clone());
                values.push(output);
            }
        }
    }
    debug_assert_eq!(values.len(), 1);
    Ok(values
        .pop()
        .expect("a contraction tree always produces one root value"))
}

#[allow(clippy::too_many_arguments)]
fn build_binary(
    left: ValueDescriptor,
    right: ValueDescriptor,
    output: EinsumValueId,
    final_node: bool,
    logical_order: &[EinsumAxis],
    dimensions: &BTreeMap<EinsumAxis, EinsumDimension>,
    occurrence_inputs: &BTreeMap<EinsumAxis, BTreeSet<usize>>,
    output_axis_set: &BTreeSet<EinsumAxis>,
    reduction_axis_set: &BTreeSet<EinsumAxis>,
    requested_output_axes: &[EinsumAxis],
) -> Result<
    (EinsumBinaryContractionPlan, ValueDescriptor),
    EinsumContractionTreeCandidateUnsupportedReason,
> {
    let subtree: BTreeSet<_> = left.leaves.union(&right.leaves).copied().collect();
    let left_axes: BTreeSet<_> = left.axes.iter().copied().collect();
    let right_axes: BTreeSet<_> = right.axes.iter().copied().collect();
    let union_axes: BTreeSet<_> = left_axes.union(&right_axes).copied().collect();
    let eliminable = |axis: EinsumAxis| {
        reduction_axis_set.contains(&axis) && occurrence_inputs[&axis].is_subset(&subtree)
    };
    for &axis in &union_axes {
        if eliminable(axis) && left_axes.contains(&axis) != right_axes.contains(&axis) {
            return Err(
                EinsumContractionTreeCandidateUnsupportedReason::UnloweredLocalReduction { axis },
            );
        }
    }
    let contract_axes = logical_order
        .iter()
        .copied()
        .filter(|axis| left_axes.contains(axis) && right_axes.contains(axis) && eliminable(*axis))
        .collect::<Vec<_>>();
    let retained: BTreeSet<_> = union_axes
        .iter()
        .copied()
        .filter(|axis| !eliminable(*axis))
        .collect();
    let batch_axes = logical_order
        .iter()
        .copied()
        .filter(|axis| {
            retained.contains(axis)
                && ((left_axes.contains(axis) && right_axes.contains(axis))
                    || (matches!(axis, EinsumAxis::Ellipsis(_)) && output_axis_set.contains(axis)))
        })
        .collect::<Vec<_>>();
    let batch_set: BTreeSet<_> = batch_axes.iter().copied().collect();
    let left_free_axes = left
        .axes
        .iter()
        .copied()
        .filter(|axis| retained.contains(axis) && !batch_set.contains(axis))
        .collect::<Vec<_>>();
    let right_free_axes = right
        .axes
        .iter()
        .copied()
        .filter(|axis| retained.contains(axis) && !batch_set.contains(axis))
        .collect::<Vec<_>>();
    let mut canonical_output_axes = batch_axes.clone();
    canonical_output_axes.extend_from_slice(&left_free_axes);
    canonical_output_axes.extend_from_slice(&right_free_axes);

    let axis_index = |axes: &[EinsumAxis], axis: EinsumAxis| {
        axes.iter().position(|candidate| *candidate == axis)
    };
    let left_axis_order = batch_axes
        .iter()
        .chain(&left_free_axes)
        .chain(&contract_axes)
        .map(|axis| axis_index(&left.axes, *axis))
        .collect::<Vec<_>>();
    let right_axis_order = batch_axes
        .iter()
        .chain(&contract_axes)
        .chain(&right_free_axes)
        .map(|axis| axis_index(&right.axes, *axis))
        .collect::<Vec<_>>();
    let left_virtual_singleton_axes = batch_axes
        .iter()
        .zip(&left_axis_order)
        .filter_map(|(&axis, mapping)| mapping.is_none().then_some(axis))
        .collect();
    let right_virtual_singleton_axes = batch_axes
        .iter()
        .zip(&right_axis_order)
        .filter_map(|(&axis, mapping)| mapping.is_none().then_some(axis))
        .collect();
    let output_permutation = if final_node {
        requested_output_axes
            .iter()
            .map(|axis| {
                canonical_output_axes
                    .iter()
                    .position(|candidate| candidate == axis)
                    .expect("root retains exactly the requested output axes")
            })
            .collect()
    } else {
        (0..canonical_output_axes.len()).collect()
    };
    let geometry = EinsumGemmGeometry {
        batch_shape: batch_axes.iter().map(|axis| dimensions[axis]).collect(),
        batch: dimension_product(&batch_axes, dimensions)?,
        m: dimension_product(&left_free_axes, dimensions)?,
        k: dimension_product(&contract_axes, dimensions)?,
        n: dimension_product(&right_free_axes, dimensions)?,
    };
    let left_requires_packing =
        left.requires_materialization || !is_identity_mapping(&left_axis_order, left.axes.len());
    let right_requires_packing =
        right.requires_materialization || !is_identity_mapping(&right_axis_order, right.axes.len());
    let output_axes = if final_node {
        requested_output_axes.to_vec()
    } else {
        canonical_output_axes.clone()
    };
    let output_axis_dimensions = output_axes
        .iter()
        .map(|axis| dimensions[axis])
        .collect::<Vec<_>>();
    let output_elements = dimension_values_product(&output_axis_dimensions)?;
    let descriptor = ValueDescriptor {
        id: output,
        leaves: subtree,
        axes: output_axes,
        axis_dimensions: output_axis_dimensions,
        elements: output_elements,
        requires_materialization: false,
    };
    let mut left_leaf_inputs = left.leaves.iter().copied().collect::<Vec<_>>();
    let mut right_leaf_inputs = right.leaves.iter().copied().collect::<Vec<_>>();
    left_leaf_inputs.sort_unstable();
    right_leaf_inputs.sort_unstable();
    let lowering = if !contract_axes.is_empty()
        && contract_axes
            .iter()
            .all(|axis| matches!(axis, EinsumAxis::Label(_)))
    {
        EinsumBinaryLowering::GemmCompatible
    } else {
        EinsumBinaryLowering::GenericNative
    };
    Ok((
        EinsumBinaryContractionPlan {
            left: left.id,
            right: right.id,
            output,
            left_leaf_inputs,
            right_leaf_inputs,
            left_value_axes: left.axes,
            right_value_axes: right.axes,
            batch_axes,
            left_free_axes,
            contract_axes,
            right_free_axes,
            left_axis_order,
            right_axis_order,
            left_virtual_singleton_axes,
            right_virtual_singleton_axes,
            canonical_output_axes,
            output_permutation,
            geometry,
            left_requires_packing,
            right_requires_packing,
            lowering,
            left_elements: left.elements,
            right_elements: right.elements,
            output_elements,
        },
        descriptor,
    ))
}

fn dimension_product(
    axes: &[EinsumAxis],
    dimensions: &BTreeMap<EinsumAxis, EinsumDimension>,
) -> Result<EinsumDimension, EinsumContractionTreeCandidateUnsupportedReason> {
    if axes
        .iter()
        .any(|axis| dimensions[axis] == EinsumDimension::Static(0))
    {
        return Ok(EinsumDimension::Static(0));
    }
    let mut product = 1usize;
    for axis in axes {
        let EinsumDimension::Static(value) = dimensions[axis] else {
            return Ok(EinsumDimension::Dynamic);
        };
        product = product.checked_mul(value).ok_or(
            EinsumContractionTreeCandidateUnsupportedReason::CostOverflow {
                metric: EinsumCostMetric::Geometry,
            },
        )?;
    }
    Ok(EinsumDimension::Static(product))
}

fn dimension_values_product(
    dimensions: &[EinsumDimension],
) -> Result<EinsumCostBound, EinsumContractionTreeCandidateUnsupportedReason> {
    if dimensions.contains(&EinsumDimension::Static(0)) {
        return Ok(EinsumCostBound::Exact(0));
    }
    let mut product = 1u128;
    for dimension in dimensions {
        let EinsumDimension::Static(value) = dimension else {
            return Ok(EinsumCostBound::UnknownUpperBound);
        };
        product = product.checked_mul(*value as u128).ok_or(
            EinsumContractionTreeCandidateUnsupportedReason::CostOverflow {
                metric: EinsumCostMetric::Geometry,
            },
        )?;
    }
    Ok(EinsumCostBound::Exact(product))
}

fn dimension_bound(dimension: EinsumDimension) -> EinsumCostBound {
    match dimension {
        EinsumDimension::Static(value) => EinsumCostBound::Exact(value as u128),
        EinsumDimension::Dynamic => EinsumCostBound::UnknownUpperBound,
    }
}

fn is_identity_mapping(order: &[Option<usize>], axis_count: usize) -> bool {
    order
        .iter()
        .filter_map(|mapping| *mapping)
        .eq(0..axis_count)
}

fn schedule_temporaries(
    steps: &[EinsumContractionTreeStep],
    final_output: EinsumValueId,
    descriptors: &BTreeMap<EinsumValueId, ValueDescriptor>,
    logical_order: &[EinsumAxis],
) -> Vec<EinsumTemporaryValuePlan> {
    let mut last_use = BTreeMap::new();
    for (step_index, step) in steps.iter().enumerate() {
        for input in step.inputs() {
            last_use.insert(input, step_index);
        }
    }
    let mut next_slot = 0usize;
    let mut free_slots = BTreeSet::new();
    let mut live_slots = BTreeMap::<EinsumValueId, usize>::new();
    let mut temporaries = Vec::new();
    for (step_index, step) in steps.iter().enumerate() {
        let output = step.output();
        if output != final_output {
            let slot = free_slots.pop_first().unwrap_or_else(|| {
                let slot = next_slot;
                next_slot += 1;
                slot
            });
            live_slots.insert(output, slot);
            let descriptor = &descriptors[&output];
            temporaries.push(EinsumTemporaryValuePlan {
                value: output,
                slot,
                birth_step: step_index,
                last_use_step: last_use.get(&output).copied().unwrap_or(step_index),
                axes: descriptor.axes.clone(),
                leaf_inputs: descriptor.leaves.iter().copied().collect(),
                global_iteration_axis_indices: descriptor
                    .axes
                    .iter()
                    .map(|axis| {
                        logical_order
                            .iter()
                            .position(|candidate| candidate == axis)
                            .expect("temporary axis belongs to global logical order")
                    })
                    .collect(),
                storage_policy: EinsumTemporaryStoragePolicy::Accumulator,
                elements: descriptor.elements,
            });
        }
        for input in step.inputs() {
            if last_use.get(&input) == Some(&step_index)
                && let Some(slot) = live_slots.remove(&input)
            {
                free_slots.insert(slot);
            }
        }
    }
    temporaries
}

fn score_candidate(
    steps: &[EinsumContractionTreeStep],
    temporaries: &[EinsumTemporaryValuePlan],
    element_size: usize,
) -> Result<
    (EinsumContractionCost, Option<EinsumResolvedContractionCost>),
    EinsumContractionTreeCandidateUnsupportedReason,
> {
    let mut flops = EinsumCostBound::Exact(0);
    let mut unary_or_product = EinsumCostBound::Exact(0);
    let mut packing = EinsumCostBound::Exact(0);
    let mut broadcast = EinsumCostBound::Exact(0);
    for step in steps {
        match step {
            EinsumContractionTreeStep::UnaryReduction(unary) => {
                let output = unary.output_elements;
                let reduction = unary.reduction_elements;
                let work = reduction_work(output, reduction)?;
                flops = add_bound(flops, work, EinsumCostMetric::Flops)?;
                unary_or_product =
                    add_bound(unary_or_product, work, EinsumCostMetric::UnaryOrProductWork)?;
            }
            EinsumContractionTreeStep::BinaryContraction(binary) => {
                let output = binary.output_elements;
                let k = dimension_bound(binary.geometry.k);
                let work = binary_work(output, k, binary.contract_axes.is_empty())?;
                flops = add_bound(flops, work, EinsumCostMetric::Flops)?;
                if binary.contract_axes.is_empty() {
                    unary_or_product =
                        add_bound(unary_or_product, work, EinsumCostMetric::UnaryOrProductWork)?;
                }
                if binary.left_requires_packing {
                    let elements = binary.left_elements;
                    packing = add_bound(
                        packing,
                        mul_bound(
                            elements,
                            EinsumCostBound::Exact(2),
                            EinsumCostMetric::LayoutOrPackingTraffic,
                        )?,
                        EinsumCostMetric::LayoutOrPackingTraffic,
                    )?;
                }
                if binary.right_requires_packing {
                    let elements = binary.right_elements;
                    packing = add_bound(
                        packing,
                        mul_bound(
                            elements,
                            EinsumCostBound::Exact(2),
                            EinsumCostMetric::LayoutOrPackingTraffic,
                        )?,
                        EinsumCostMetric::LayoutOrPackingTraffic,
                    )?;
                }
                if binary
                    .output_permutation
                    .iter()
                    .copied()
                    .ne(0..binary.output_permutation.len())
                {
                    packing = add_bound(
                        packing,
                        mul_bound(
                            output,
                            EinsumCostBound::Exact(2),
                            EinsumCostMetric::LayoutOrPackingTraffic,
                        )?,
                        EinsumCostMetric::LayoutOrPackingTraffic,
                    )?;
                }
                broadcast = add_bound(
                    broadcast,
                    binary_broadcast_amplification(binary)?,
                    EinsumCostMetric::BroadcastAmplification,
                )?;
            }
        }
    }

    let intermediate_elements =
        temporaries
            .iter()
            .try_fold(EinsumCostBound::Exact(0), |sum, temporary| {
                add_bound(
                    sum,
                    temporary.elements,
                    EinsumCostMetric::IntermediateElements,
                )
            })?;
    let total_intermediate_traffic = mul_bound(
        intermediate_elements,
        EinsumCostBound::Exact(2),
        EinsumCostMetric::TotalIntermediateTraffic,
    )?;
    let peak_live = peak_live_elements(temporaries)?;
    let slot_count = temporaries
        .iter()
        .map(|temporary| temporary.slot + 1)
        .max()
        .unwrap_or(0);
    let cost = EinsumContractionCost {
        flops,
        unary_or_product_work: unary_or_product,
        intermediate_elements,
        peak_live_temporary_elements: peak_live,
        total_intermediate_traffic_elements: total_intermediate_traffic,
        layout_or_packing_traffic_elements: packing,
        broadcast_amplification_elements: broadcast,
        slot_count,
    };
    if cost.requires_concrete_rescore() {
        return Ok((cost, None));
    }
    let exact = |bound: EinsumCostBound| match bound {
        EinsumCostBound::Exact(value) => Ok(value),
        EinsumCostBound::UnknownUpperBound => Err(
            EinsumContractionTreeCandidateUnsupportedReason::CostOverflow {
                metric: EinsumCostMetric::Geometry,
            },
        ),
    };
    let bytes = |bound: EinsumCostBound| {
        let value = exact(bound)?;
        EinsumCostBound::Exact(value)
            .checked_scale(element_size)
            .and_then(EinsumCostBound::exact)
            .ok_or(
                EinsumContractionTreeCandidateUnsupportedReason::CostOverflow {
                    metric: EinsumCostMetric::Bytes,
                },
            )
    };
    let intermediate = exact(intermediate_elements)?;
    let resolved = EinsumResolvedContractionCost {
        flops: exact(flops)?,
        unary_or_product_work: exact(unary_or_product)?,
        intermediate_elements: intermediate,
        intermediate_bytes: bytes(intermediate_elements)?,
        peak_live_temporary_bytes: bytes(peak_live)?,
        total_intermediate_traffic_bytes: bytes(total_intermediate_traffic)?,
        layout_or_packing_traffic_bytes: bytes(packing)?,
        broadcast_amplification_elements: exact(broadcast)?,
        slot_count,
    };
    Ok((cost, Some(resolved)))
}

fn add_bound(
    left: EinsumCostBound,
    right: EinsumCostBound,
    metric: EinsumCostMetric,
) -> Result<EinsumCostBound, EinsumContractionTreeCandidateUnsupportedReason> {
    match (left, right) {
        (EinsumCostBound::Exact(left), EinsumCostBound::Exact(right)) => left
            .checked_add(right)
            .map(EinsumCostBound::Exact)
            .ok_or(EinsumContractionTreeCandidateUnsupportedReason::CostOverflow { metric }),
        _ => Ok(EinsumCostBound::UnknownUpperBound),
    }
}

fn mul_bound(
    left: EinsumCostBound,
    right: EinsumCostBound,
    metric: EinsumCostMetric,
) -> Result<EinsumCostBound, EinsumContractionTreeCandidateUnsupportedReason> {
    match (left, right) {
        (EinsumCostBound::Exact(0), _) | (_, EinsumCostBound::Exact(0)) => {
            Ok(EinsumCostBound::Exact(0))
        }
        (EinsumCostBound::Exact(left), EinsumCostBound::Exact(right)) => left
            .checked_mul(right)
            .map(EinsumCostBound::Exact)
            .ok_or(EinsumContractionTreeCandidateUnsupportedReason::CostOverflow { metric }),
        _ => Ok(EinsumCostBound::UnknownUpperBound),
    }
}

fn product_bounds<const N: usize>(
    factors: [EinsumCostBound; N],
    metric: EinsumCostMetric,
) -> Result<EinsumCostBound, EinsumContractionTreeCandidateUnsupportedReason> {
    if factors.contains(&EinsumCostBound::Exact(0)) {
        return Ok(EinsumCostBound::Exact(0));
    }
    factors
        .into_iter()
        .try_fold(EinsumCostBound::Exact(1), |product, factor| {
            mul_bound(product, factor, metric)
        })
}

fn reduction_work(
    output: EinsumCostBound,
    reduction: EinsumCostBound,
) -> Result<EinsumCostBound, EinsumContractionTreeCandidateUnsupportedReason> {
    if output == EinsumCostBound::Exact(0) {
        return Ok(EinsumCostBound::Exact(0));
    }
    match reduction {
        EinsumCostBound::Exact(0 | 1) => Ok(EinsumCostBound::Exact(0)),
        EinsumCostBound::Exact(value) => mul_bound(
            output,
            EinsumCostBound::Exact(value - 1),
            EinsumCostMetric::UnaryOrProductWork,
        ),
        EinsumCostBound::UnknownUpperBound => {
            if output == EinsumCostBound::Exact(0) {
                Ok(EinsumCostBound::Exact(0))
            } else {
                Ok(EinsumCostBound::UnknownUpperBound)
            }
        }
    }
}

fn binary_work(
    output: EinsumCostBound,
    k: EinsumCostBound,
    product_only: bool,
) -> Result<EinsumCostBound, EinsumContractionTreeCandidateUnsupportedReason> {
    if output == EinsumCostBound::Exact(0) {
        return Ok(EinsumCostBound::Exact(0));
    }
    if product_only {
        return Ok(output);
    }
    match k {
        EinsumCostBound::Exact(0) => Ok(EinsumCostBound::Exact(0)),
        EinsumCostBound::Exact(value) => {
            let operations = value
                .checked_mul(2)
                .and_then(|value| value.checked_sub(1))
                .ok_or(
                    EinsumContractionTreeCandidateUnsupportedReason::CostOverflow {
                        metric: EinsumCostMetric::Flops,
                    },
                )?;
            mul_bound(
                output,
                EinsumCostBound::Exact(operations),
                EinsumCostMetric::Flops,
            )
        }
        EinsumCostBound::UnknownUpperBound => {
            if output == EinsumCostBound::Exact(0) {
                Ok(EinsumCostBound::Exact(0))
            } else {
                Ok(EinsumCostBound::UnknownUpperBound)
            }
        }
    }
}

fn binary_broadcast_amplification(
    binary: &EinsumBinaryContractionPlan,
) -> Result<EinsumCostBound, EinsumContractionTreeCandidateUnsupportedReason> {
    let difference = |expanded: EinsumCostBound, physical: EinsumCostBound| -> EinsumCostBound {
        match (expanded, physical) {
            (EinsumCostBound::Exact(expanded), EinsumCostBound::Exact(physical)) => {
                EinsumCostBound::Exact(expanded.saturating_sub(physical))
            }
            _ => EinsumCostBound::UnknownUpperBound,
        }
    };
    let batch = dimension_bound(binary.geometry.batch);
    let m = dimension_bound(binary.geometry.m);
    let k = dimension_bound(binary.geometry.k);
    let n = dimension_bound(binary.geometry.n);
    let left = difference(
        product_bounds([batch, m, k], EinsumCostMetric::BroadcastAmplification)?,
        binary.left_elements,
    );
    let right = difference(
        product_bounds([batch, k, n], EinsumCostMetric::BroadcastAmplification)?,
        binary.right_elements,
    );
    add_bound(left, right, EinsumCostMetric::BroadcastAmplification)
}

fn peak_live_elements(
    temporaries: &[EinsumTemporaryValuePlan],
) -> Result<EinsumCostBound, EinsumContractionTreeCandidateUnsupportedReason> {
    let max_step = temporaries
        .iter()
        .map(|temporary| temporary.last_use_step)
        .max()
        .unwrap_or(0);
    let mut peak = EinsumCostBound::Exact(0);
    for step in 0..=max_step {
        let live = temporaries
            .iter()
            .filter(|temporary| temporary.birth_step <= step && step <= temporary.last_use_step)
            .try_fold(EinsumCostBound::Exact(0), |sum, temporary| {
                add_bound(
                    sum,
                    temporary.elements,
                    EinsumCostMetric::PeakLiveTemporaryElements,
                )
            })?;
        if compare_bound(live, peak) == Ordering::Greater {
            peak = live;
        }
    }
    Ok(peak)
}

fn concretize_candidate(
    candidate: &EinsumSupportedContractionTreeCandidate,
    dimensions: &BTreeMap<EinsumAxis, usize>,
    input_shapes: &[&[usize]],
    operands: &[EinsumOperandPlan],
) -> Result<
    (
        Vec<EinsumContractionTreeStep>,
        Vec<EinsumTemporaryValuePlan>,
    ),
    EinsumContractionTreeCandidateUnsupportedReason,
> {
    let mut steps = candidate.steps.clone();
    for step in &mut steps {
        match step {
            EinsumContractionTreeStep::UnaryReduction(unary) => {
                let input = unary.input.index();
                unary.input_elements =
                    concrete_leaf_elements(input, &unary.input_axes, input_shapes, operands)?;
                unary.output_elements =
                    concrete_leaf_elements(input, &unary.output_axes, input_shapes, operands)?;
                unary.reduction_elements =
                    concrete_leaf_elements(input, &unary.reduction_axes, input_shapes, operands)?;
            }
            EinsumContractionTreeStep::BinaryContraction(binary) => {
                binary.left_elements = concrete_value_elements(
                    &binary.left_leaf_inputs,
                    &binary.left_value_axes,
                    dimensions,
                    input_shapes,
                    operands,
                )?;
                binary.right_elements = concrete_value_elements(
                    &binary.right_leaf_inputs,
                    &binary.right_value_axes,
                    dimensions,
                    input_shapes,
                    operands,
                )?;
                binary.output_elements =
                    concrete_global_elements(&binary.canonical_output_axes, dimensions)?;
                let geometry = resolve_binary_geometry(binary, dimensions)?;
                binary.geometry = EinsumGemmGeometry {
                    batch_shape: geometry
                        .batch_shape()
                        .iter()
                        .copied()
                        .map(EinsumDimension::Static)
                        .collect(),
                    batch: EinsumDimension::Static(geometry.batch()),
                    m: EinsumDimension::Static(geometry.m()),
                    k: EinsumDimension::Static(geometry.k()),
                    n: EinsumDimension::Static(geometry.n()),
                };
            }
        }
    }
    let mut temporaries = candidate.temporaries.clone();
    for temporary in &mut temporaries {
        temporary.elements = concrete_value_elements(
            &temporary.leaf_inputs,
            &temporary.axes,
            dimensions,
            input_shapes,
            operands,
        )?;
    }
    Ok((steps, temporaries))
}

fn concrete_value_elements(
    leaf_inputs: &[usize],
    axes: &[EinsumAxis],
    dimensions: &BTreeMap<EinsumAxis, usize>,
    input_shapes: &[&[usize]],
    operands: &[EinsumOperandPlan],
) -> Result<EinsumCostBound, EinsumContractionTreeCandidateUnsupportedReason> {
    if let [input] = leaf_inputs {
        concrete_leaf_elements(*input, axes, input_shapes, operands)
    } else {
        concrete_global_elements(axes, dimensions)
    }
}

fn concrete_leaf_elements(
    input: usize,
    axes: &[EinsumAxis],
    input_shapes: &[&[usize]],
    operands: &[EinsumOperandPlan],
) -> Result<EinsumCostBound, EinsumContractionTreeCandidateUnsupportedReason> {
    let operand = &operands[input];
    let shape = input_shapes[input];
    let mut values = Vec::with_capacity(axes.len());
    for axis in axes {
        let operand_axis = operand
            .unique_axes()
            .iter()
            .find(|candidate| candidate.axis() == *axis)
            .expect("candidate leaf axis belongs to its source operand");
        values.push(shape[operand_axis.input_axes()[0]]);
    }
    concrete_product(&values)
}

fn concrete_global_elements(
    axes: &[EinsumAxis],
    dimensions: &BTreeMap<EinsumAxis, usize>,
) -> Result<EinsumCostBound, EinsumContractionTreeCandidateUnsupportedReason> {
    let values = axes.iter().map(|axis| dimensions[axis]).collect::<Vec<_>>();
    concrete_product(&values)
}

fn concrete_product(
    values: &[usize],
) -> Result<EinsumCostBound, EinsumContractionTreeCandidateUnsupportedReason> {
    if values.contains(&0) {
        return Ok(EinsumCostBound::Exact(0));
    }
    values
        .iter()
        .try_fold(1u128, |product, value| {
            product.checked_mul(*value as u128).ok_or(
                EinsumContractionTreeCandidateUnsupportedReason::CostOverflow {
                    metric: EinsumCostMetric::Geometry,
                },
            )
        })
        .map(EinsumCostBound::Exact)
}

fn resolve_binary_geometry(
    binary: &EinsumBinaryContractionPlan,
    dimensions: &BTreeMap<EinsumAxis, usize>,
) -> Result<EinsumConcreteGemmGeometry, EinsumContractionTreeCandidateUnsupportedReason> {
    let product =
        |axes: &[EinsumAxis]| -> Result<usize, EinsumContractionTreeCandidateUnsupportedReason> {
            if axes.iter().any(|axis| dimensions[axis] == 0) {
                return Ok(0);
            }
            axes.iter().try_fold(1usize, |product, axis| {
                product.checked_mul(dimensions[axis]).ok_or(
                    EinsumContractionTreeCandidateUnsupportedReason::CostOverflow {
                        metric: EinsumCostMetric::Geometry,
                    },
                )
            })
        };
    Ok(EinsumConcreteGemmGeometry {
        batch_shape: binary
            .batch_axes
            .iter()
            .map(|axis| dimensions[axis])
            .collect(),
        batch: product(&binary.batch_axes)?,
        m: product(&binary.left_free_axes)?,
        k: product(&binary.contract_axes)?,
        n: product(&binary.right_free_axes)?,
    })
}

#[cfg(test)]
mod cost_tests {
    use super::*;

    #[test]
    fn exact_zero_annihilates_work_before_irrelevant_boundary_arithmetic() {
        let zero = EinsumCostBound::Exact(0);
        let maximum = EinsumCostBound::Exact(u128::MAX);
        let unknown = EinsumCostBound::UnknownUpperBound;

        for (name, result) in [
            ("contracted binary", binary_work(zero, maximum, false)),
            ("symbolic binary", binary_work(zero, unknown, false)),
            ("product binary", binary_work(zero, maximum, true)),
            ("unary reduction", reduction_work(zero, maximum)),
            ("symbolic unary reduction", reduction_work(zero, unknown)),
        ] {
            assert_eq!(result.unwrap(), zero, "{name}");
        }
    }

    #[test]
    fn exact_zero_annihilates_cost_products_in_every_factor_position() {
        let metrics = [
            EinsumCostMetric::UnaryOrProductWork,
            EinsumCostMetric::TotalIntermediateTraffic,
            EinsumCostMetric::LayoutOrPackingTraffic,
            EinsumCostMetric::BroadcastAmplification,
        ];
        for metric in metrics {
            assert_eq!(
                mul_bound(EinsumCostBound::Exact(0), EinsumCostBound::Exact(2), metric,).unwrap(),
                EinsumCostBound::Exact(0),
                "{metric}"
            );

            for zero_index in 0..4 {
                let mut factors = [
                    EinsumCostBound::Exact(u128::MAX),
                    EinsumCostBound::Exact(2),
                    EinsumCostBound::UnknownUpperBound,
                    EinsumCostBound::Exact(u128::MAX),
                ];
                factors[zero_index] = EinsumCostBound::Exact(0);
                assert_eq!(
                    product_bounds(factors, metric).unwrap(),
                    EinsumCostBound::Exact(0),
                    "{metric}, zero factor {zero_index}"
                );
            }
        }
    }

    #[test]
    fn byte_scaling_preserves_zero_and_unknown_bounds() {
        for (bound, multiplier, expected) in [
            (
                EinsumCostBound::Exact(0),
                usize::MAX,
                Some(EinsumCostBound::Exact(0)),
            ),
            (
                EinsumCostBound::UnknownUpperBound,
                0,
                Some(EinsumCostBound::Exact(0)),
            ),
            (
                EinsumCostBound::UnknownUpperBound,
                usize::MAX,
                Some(EinsumCostBound::UnknownUpperBound),
            ),
            (EinsumCostBound::Exact(u128::MAX), 2, None),
        ] {
            assert_eq!(bound.checked_scale(multiplier), expected);
        }
    }
}

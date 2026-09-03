//! Ordered binary contraction-tree planning for coupled two- and three-input
//! einsums.

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
    Exact(u64),
    /// Unknown value with an unbounded/infinite upper bound.
    UnknownUpperBound,
}

impl EinsumCostBound {
    /// Exact value, when statically known.
    pub const fn exact(self) -> Option<u64> {
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
        match self {
            Self::Exact(value) => Some(Self::Exact(
                value.checked_mul(u64::try_from(multiplier).ok()?)?,
            )),
            Self::UnknownUpperBound => Some(Self::UnknownUpperBound),
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
                write!(f, "{metric} exceeds the checked u64 planning bound")
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
    flops: u64,
    unary_or_product_work: u64,
    intermediate_elements: u64,
    intermediate_bytes: u64,
    peak_live_temporary_bytes: u64,
    total_intermediate_traffic_bytes: u64,
    layout_or_packing_traffic_bytes: u64,
    broadcast_amplification_elements: u64,
    slot_count: usize,
}

impl EinsumResolvedContractionCost {
    /// Total scalar arithmetic operations.
    pub const fn flops(&self) -> u64 {
        self.flops
    }

    /// Leaf-local reduction and K-free product work.
    pub const fn unary_or_product_work(&self) -> u64 {
        self.unary_or_product_work
    }

    /// Sum of temporary element counts.
    pub const fn intermediate_elements(&self) -> u64 {
        self.intermediate_elements
    }

    /// Sum of temporary value bytes.
    pub const fn intermediate_bytes(&self) -> u64 {
        self.intermediate_bytes
    }

    /// Maximum simultaneously live temporary bytes.
    pub const fn peak_live_temporary_bytes(&self) -> u64 {
        self.peak_live_temporary_bytes
    }

    /// Temporary writes plus reads.
    pub const fn total_intermediate_traffic_bytes(&self) -> u64 {
        self.total_intermediate_traffic_bytes
    }

    /// Estimated permutation/diagonal materialization traffic.
    pub const fn layout_or_packing_traffic_bytes(&self) -> u64 {
        self.layout_or_packing_traffic_bytes
    }

    /// Extra logical elements caused by batch broadcasting.
    pub const fn broadcast_amplification_elements(&self) -> u64 {
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

/// One temporary value and its reusable-slot liveness interval.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EinsumTemporaryValuePlan {
    value: EinsumValueId,
    slot: usize,
    birth_step: usize,
    last_use_step: usize,
    axes: Vec<EinsumAxis>,
    leaf_inputs: Vec<usize>,
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
    Supported(EinsumSupportedContractionTreeCandidate),
    /// Candidate is semantically legal but cannot be represented safely.
    Unsupported(EinsumContractionTreeCandidateUnsupportedReason),
}

/// One semantically legal ordered binary tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EinsumContractionTreeCandidate {
    id: EinsumContractionTreeCandidateId,
    first_pair: [usize; 2],
    root_intermediate_on_left: bool,
    plan: EinsumContractionTreeCandidatePlan,
}

impl EinsumContractionTreeCandidate {
    /// Stable lexicographic candidate identifier.
    pub const fn id(&self) -> &EinsumContractionTreeCandidateId {
        &self.id
    }

    /// Ordered original-input pair combined by the first binary node.
    pub const fn first_pair(&self) -> [usize; 2] {
        self.first_pair
    }

    /// Whether the first intermediate is the left operand of the root node.
    pub const fn root_intermediate_on_left(&self) -> bool {
        self.root_intermediate_on_left
    }

    /// Supported or fail-closed candidate plan.
    pub const fn plan(&self) -> &EinsumContractionTreeCandidatePlan {
        &self.plan
    }

    /// Supported candidate details.
    pub const fn supported(&self) -> Option<&EinsumSupportedContractionTreeCandidate> {
        match &self.plan {
            EinsumContractionTreeCandidatePlan::Supported(plan) => Some(plan),
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

/// Canonical ordered-tree plan for a coupled two- or three-input einsum.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EinsumContractionTreePlan {
    arity: usize,
    leaf_values: Vec<EinsumValueId>,
    candidates: Vec<EinsumContractionTreeCandidate>,
    preferred_candidate: Option<usize>,
    requires_concrete_rescore: bool,
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

    /// Every semantically legal ordered binary tree, sorted by stable ID.
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

    pub(super) fn resolve(
        &self,
        dimensions: &BTreeMap<EinsumAxis, usize>,
        input_shapes: &[&[usize]],
        operands: &[EinsumOperandPlan],
        element_size: usize,
    ) -> EinsumConcreteContractionTreePlan {
        let mut candidates = Vec::with_capacity(self.candidates.len());
        for candidate in &self.candidates {
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

#[derive(Clone, Copy)]
struct CandidateShape {
    first_pair: [usize; 2],
    remaining: Option<usize>,
    root_intermediate_on_left: bool,
}

pub(super) fn build_contraction_tree(
    operands: &[EinsumOperandPlan],
    logical_axes: &[EinsumLogicalAxis],
    output_axes: &[EinsumAxis],
    reduction_axes: &[EinsumAxis],
) -> EinsumContractionTreePlan {
    debug_assert!((2..=3).contains(&operands.len()));
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
    let local_reductions: Vec<BTreeSet<_>> = (0..operands.len())
        .map(|input| {
            reduction_axes
                .iter()
                .copied()
                .filter(|axis| occurrence_inputs[axis].len() == 1)
                .filter(|axis| occurrence_inputs[axis].contains(&input))
                .collect()
        })
        .collect();

    let shapes = enumerate_shapes(operands.len());
    let mut candidates = shapes
        .into_iter()
        .map(|shape| {
            let id = candidate_id(shape);
            let plan = match build_candidate(
                shape,
                operands,
                &logical_order,
                &dimensions,
                &occurrence_inputs,
                &output_axis_set,
                &reduction_axes,
                output_axes,
                &local_reductions,
            ) {
                Ok(plan) => EinsumContractionTreeCandidatePlan::Supported(plan),
                Err(reason) => EinsumContractionTreeCandidatePlan::Unsupported(reason),
            };
            EinsumContractionTreeCandidate {
                id,
                first_pair: shape.first_pair,
                root_intermediate_on_left: shape.root_intermediate_on_left,
                plan,
            }
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

    EinsumContractionTreePlan {
        arity: operands.len(),
        leaf_values: (0..operands.len()).map(EinsumValueId).collect(),
        candidates,
        preferred_candidate,
        requires_concrete_rescore,
    }
}

fn enumerate_shapes(arity: usize) -> Vec<CandidateShape> {
    if arity == 2 {
        return vec![
            CandidateShape {
                first_pair: [0, 1],
                remaining: None,
                root_intermediate_on_left: true,
            },
            CandidateShape {
                first_pair: [1, 0],
                remaining: None,
                root_intermediate_on_left: true,
            },
        ];
    }
    let mut shapes = Vec::with_capacity(12);
    for left in 0..3 {
        for right in 0..3 {
            if left == right {
                continue;
            }
            let remaining = (0..3)
                .find(|input| *input != left && *input != right)
                .expect("three inputs leave one remaining input");
            for root_intermediate_on_left in [true, false] {
                shapes.push(CandidateShape {
                    first_pair: [left, right],
                    remaining: Some(remaining),
                    root_intermediate_on_left,
                });
            }
        }
    }
    shapes
}

fn candidate_id(shape: CandidateShape) -> EinsumContractionTreeCandidateId {
    let [left, right] = shape.first_pair;
    let text = if let Some(remaining) = shape.remaining {
        if shape.root_intermediate_on_left {
            format!("(({left},{right}),{remaining})")
        } else {
            format!("({remaining},({left},{right}))")
        }
    } else {
        format!("({left},{right})")
    };
    EinsumContractionTreeCandidateId(text)
}

#[allow(clippy::too_many_arguments)]
fn build_candidate(
    shape: CandidateShape,
    operands: &[EinsumOperandPlan],
    logical_order: &[EinsumAxis],
    dimensions: &BTreeMap<EinsumAxis, EinsumDimension>,
    occurrence_inputs: &BTreeMap<EinsumAxis, BTreeSet<usize>>,
    output_axis_set: &BTreeSet<EinsumAxis>,
    reduction_axis_set: &BTreeSet<EinsumAxis>,
    output_axes: &[EinsumAxis],
    local_reductions: &[BTreeSet<EinsumAxis>],
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

    let prepare_leaf =
        |input: usize,
         steps: &mut Vec<EinsumContractionTreeStep>,
         descriptors: &mut BTreeMap<EinsumValueId, ValueDescriptor>|
         -> Result<ValueDescriptor, EinsumContractionTreeCandidateUnsupportedReason> {
            let leaf = descriptors[&EinsumValueId(input)].clone();
            if local_reductions[input].is_empty() {
                return Ok(leaf);
            }
            let output_id = EinsumValueId(arity + input);
            let output_axes = leaf
                .axes
                .iter()
                .copied()
                .filter(|axis| !local_reductions[input].contains(axis))
                .collect::<Vec<_>>();
            let output_axis_dimensions = leaf
                .axes
                .iter()
                .zip(&leaf.axis_dimensions)
                .filter_map(|(axis, dimension)| {
                    (!local_reductions[input].contains(axis)).then_some(*dimension)
                })
                .collect::<Vec<_>>();
            let reduction_dimensions = leaf
                .axes
                .iter()
                .zip(&leaf.axis_dimensions)
                .filter_map(|(axis, dimension)| {
                    local_reductions[input].contains(axis).then_some(*dimension)
                })
                .collect::<Vec<_>>();
            let output_elements = dimension_values_product(&output_axis_dimensions)?;
            let reduction_elements = dimension_values_product(&reduction_dimensions)?;
            let plan = EinsumUnaryReductionPlan {
                input: leaf.id,
                output: output_id,
                reduction_axes: logical_order
                    .iter()
                    .copied()
                    .filter(|axis| local_reductions[input].contains(axis))
                    .collect(),
                input_axes: leaf.axes.clone(),
                output_axes: output_axes.clone(),
                input_elements: leaf.elements,
                output_elements,
                reduction_elements,
            };
            steps.push(EinsumContractionTreeStep::UnaryReduction(plan));
            let output = ValueDescriptor {
                id: output_id,
                leaves: leaf.leaves,
                axes: output_axes,
                axis_dimensions: output_axis_dimensions,
                elements: output_elements,
                requires_materialization: false,
            };
            descriptors.insert(output_id, output.clone());
            Ok(output)
        };

    let first_left = prepare_leaf(shape.first_pair[0], &mut steps, &mut descriptors)?;
    let first_right = prepare_leaf(shape.first_pair[1], &mut steps, &mut descriptors)?;
    let first_output_id = EinsumValueId(arity * 2);
    let first_is_final = shape.remaining.is_none();
    let (first_binary, first_output) = build_binary(
        first_left,
        first_right,
        first_output_id,
        first_is_final,
        logical_order,
        dimensions,
        occurrence_inputs,
        output_axis_set,
        reduction_axis_set,
        output_axes,
    )?;
    let final_output_permutation = if first_is_final {
        first_binary.output_permutation.clone()
    } else {
        Vec::new()
    };
    steps.push(EinsumContractionTreeStep::BinaryContraction(Box::new(
        first_binary,
    )));
    descriptors.insert(first_output_id, first_output.clone());

    let final_output = if let Some(remaining) = shape.remaining {
        let remaining = prepare_leaf(remaining, &mut steps, &mut descriptors)?;
        let final_output_id = EinsumValueId(arity * 2 + 1);
        let (left, right) = if shape.root_intermediate_on_left {
            (first_output, remaining)
        } else {
            (remaining, first_output)
        };
        let (root, output) = build_binary(
            left,
            right,
            final_output_id,
            true,
            logical_order,
            dimensions,
            occurrence_inputs,
            output_axis_set,
            reduction_axis_set,
            output_axes,
        )?;
        let final_output_permutation = root.output_permutation.clone();
        steps.push(EinsumContractionTreeStep::BinaryContraction(Box::new(root)));
        descriptors.insert(final_output_id, output);
        (final_output_id, final_output_permutation)
    } else {
        (first_output_id, final_output_permutation)
    };

    let temporaries = schedule_temporaries(&steps, final_output.0, &descriptors);
    let (cost, _) = score_candidate(&steps, &temporaries, 1)?;
    Ok(EinsumSupportedContractionTreeCandidate {
        steps,
        temporaries,
        final_output: final_output.0,
        final_output_permutation: final_output.1,
        cost,
    })
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
    let mut product = 1u64;
    for dimension in dimensions {
        let EinsumDimension::Static(value) = dimension else {
            return Ok(EinsumCostBound::UnknownUpperBound);
        };
        product = product
            .checked_mul(u64::try_from(*value).map_err(|_| {
                EinsumContractionTreeCandidateUnsupportedReason::CostOverflow {
                    metric: EinsumCostMetric::Geometry,
                }
            })?)
            .ok_or(
                EinsumContractionTreeCandidateUnsupportedReason::CostOverflow {
                    metric: EinsumCostMetric::Geometry,
                },
            )?;
    }
    Ok(EinsumCostBound::Exact(product))
}

fn dimension_bound(dimension: EinsumDimension) -> EinsumCostBound {
    match dimension {
        EinsumDimension::Static(value) => u64::try_from(value)
            .map(EinsumCostBound::Exact)
            .unwrap_or(EinsumCostBound::UnknownUpperBound),
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
    let element_size = u64::try_from(element_size).map_err(|_| {
        EinsumContractionTreeCandidateUnsupportedReason::CostOverflow {
            metric: EinsumCostMetric::Bytes,
        }
    })?;
    let intermediate = exact(intermediate_elements)?;
    let resolved = EinsumResolvedContractionCost {
        flops: exact(flops)?,
        unary_or_product_work: exact(unary_or_product)?,
        intermediate_elements: intermediate,
        intermediate_bytes: intermediate.checked_mul(element_size).ok_or(
            EinsumContractionTreeCandidateUnsupportedReason::CostOverflow {
                metric: EinsumCostMetric::Bytes,
            },
        )?,
        peak_live_temporary_bytes: exact(peak_live)?.checked_mul(element_size).ok_or(
            EinsumContractionTreeCandidateUnsupportedReason::CostOverflow {
                metric: EinsumCostMetric::Bytes,
            },
        )?,
        total_intermediate_traffic_bytes: exact(total_intermediate_traffic)?
            .checked_mul(element_size)
            .ok_or(
                EinsumContractionTreeCandidateUnsupportedReason::CostOverflow {
                    metric: EinsumCostMetric::Bytes,
                },
            )?,
        layout_or_packing_traffic_bytes: exact(packing)?.checked_mul(element_size).ok_or(
            EinsumContractionTreeCandidateUnsupportedReason::CostOverflow {
                metric: EinsumCostMetric::Bytes,
            },
        )?,
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

fn reduction_work(
    output: EinsumCostBound,
    reduction: EinsumCostBound,
) -> Result<EinsumCostBound, EinsumContractionTreeCandidateUnsupportedReason> {
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
        mul_bound(
            mul_bound(batch, m, EinsumCostMetric::BroadcastAmplification)?,
            k,
            EinsumCostMetric::BroadcastAmplification,
        )?,
        binary.left_elements,
    );
    let right = difference(
        mul_bound(
            mul_bound(batch, k, EinsumCostMetric::BroadcastAmplification)?,
            n,
            EinsumCostMetric::BroadcastAmplification,
        )?,
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
        .try_fold(1u64, |product, value| {
            product
                .checked_mul(u64::try_from(*value).map_err(|_| {
                    EinsumContractionTreeCandidateUnsupportedReason::CostOverflow {
                        metric: EinsumCostMetric::Geometry,
                    }
                })?)
                .ok_or(
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

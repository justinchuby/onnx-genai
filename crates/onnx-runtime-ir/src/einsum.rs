//! Canonical, execution-provider-neutral planning for ONNX `Einsum`.
//!
//! [`EinsumPlan`] is the typed, parsed, and validated representation used by
//! shape inference and typed execution-provider admission. [`EinsumShapePlan`]
//! carries the same structural contract for factories that receive shapes but
//! no dtype. Both carry an explicit resolved schema proof, record logical axes,
//! diagonal groups, equality/broadcast constraints, output order, reductions,
//! a universal generic index program, and bounded contraction-planner output.
//! Consumers never need to reparse an equation, infer schema from a dtype, or
//! treat GEMM compatibility as the semantic boundary.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::{DataType, Dim};

mod tree;

pub use tree::{
    EinsumBinaryContractionPlan, EinsumBinaryLowering, EinsumConcreteContractionTreeCandidate,
    EinsumConcreteContractionTreePlan, EinsumContractionCost, EinsumContractionTreeCandidate,
    EinsumContractionTreeCandidateId, EinsumContractionTreeCandidatePlan,
    EinsumContractionTreeCandidateUnsupportedReason, EinsumContractionTreePlan,
    EinsumContractionTreeStep, EinsumCostBound, EinsumCostMetric, EinsumPlannerBudget,
    EinsumPlannerFallbackReason, EinsumPlannerQuality, EinsumPlannerUsage,
    EinsumResolvedContractionCost, EinsumSupportedContractionTreeCandidate,
    EinsumTemporaryStoragePolicy, EinsumTemporaryValuePlan, EinsumUnaryReductionPlan,
    EinsumValueId,
};

/// The ONNX `Einsum` schema selected by the imported `ai.onnx` opset.
///
/// This is a proof value: model-facing code resolves it from the effective
/// imported opset once, then passes it through validation, shape inference, and
/// execution-provider planning. Dtype legality is never inferred from the
/// dtype itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EinsumSchema {
    /// `Einsum-12`, selected by imported opsets 12 through 27.
    V12,
    /// `Einsum-28`, selected by imported opset 28 and later.
    V28,
}

impl EinsumSchema {
    /// Resolve the applicable schema from the effective imported `ai.onnx`
    /// opset.
    pub fn resolve(imported_opset: u64) -> Result<Self, EinsumSchemaError> {
        match imported_opset {
            0..=11 => Err(EinsumSchemaError { imported_opset }),
            12..=27 => Ok(Self::V12),
            28.. => Ok(Self::V28),
        }
    }

    /// The schema's ONNX `since_version`.
    pub const fn since_version(self) -> u64 {
        match self {
            Self::V12 => 12,
            Self::V28 => 28,
        }
    }

    /// Whether `dtype` is admitted by this schema's `T` constraint.
    pub const fn supports_dtype(self, dtype: DataType) -> bool {
        is_base_numeric_dtype(dtype) || matches!((self, dtype), (Self::V28, DataType::BFloat16))
    }
}

impl fmt::Display for EinsumSchema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Einsum-{}", self.since_version())
    }
}

/// An imported opset that cannot select an ONNX `Einsum` schema.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EinsumSchemaError {
    imported_opset: u64,
}

impl EinsumSchemaError {
    /// Rejected effective `ai.onnx` opset.
    pub const fn imported_opset(self) -> u64 {
        self.imported_opset
    }
}

impl fmt::Display for EinsumSchemaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ai.onnx opset {} cannot select Einsum: the operator was introduced in opset 12",
            self.imported_opset
        )
    }
}

impl std::error::Error for EinsumSchemaError {}

/// A dimension value that can expose a statically known extent to the planner.
///
/// Dynamic/symbolic dimensions return `None`. The plan preserves their runtime
/// equality or broadcast requirements as [`EinsumLogicalAxis`] constraints.
pub trait EinsumDimensionValue {
    /// Return the non-negative static extent, when known.
    fn einsum_static_size(&self) -> Option<usize>;
}

impl EinsumDimensionValue for usize {
    fn einsum_static_size(&self) -> Option<usize> {
        Some(*self)
    }
}

impl EinsumDimensionValue for Dim {
    fn einsum_static_size(&self) -> Option<usize> {
        self.as_static()
    }
}

/// One input supplied to [`EinsumPlan::build`].
///
/// Optional dtype/shape fields let callers faithfully represent incomplete
/// graph metadata. The planner rejects either absence explicitly; permissive
/// shape inference may then choose to leave its output unresolved.
#[derive(Clone, Copy, Debug)]
pub struct EinsumInput<'a, D> {
    dtype: Option<DataType>,
    shape: Option<&'a [D]>,
}

impl<'a, D> EinsumInput<'a, D> {
    /// An input whose dtype and shape are both available.
    pub const fn new(dtype: DataType, shape: &'a [D]) -> Self {
        Self {
            dtype: Some(dtype),
            shape: Some(shape),
        }
    }

    /// An input assembled from possibly incomplete graph metadata.
    pub const fn from_optional(dtype: Option<DataType>, shape: Option<&'a [D]>) -> Self {
        Self { dtype, shape }
    }

    /// The input dtype, when available.
    pub const fn dtype(self) -> Option<DataType> {
        self.dtype
    }

    /// The input shape, when available.
    pub const fn shape(self) -> Option<&'a [D]> {
        self.shape
    }
}

/// A normalized ASCII einsum label.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EinsumLabel(u8);

impl EinsumLabel {
    /// The label as an ASCII character.
    pub const fn as_char(self) -> char {
        self.0 as char
    }
}

impl fmt::Display for EinsumLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.as_char().to_string())
    }
}

/// A canonical logical axis in an einsum expression.
///
/// Ellipsis axes are numbered left-to-right after expanding every explicit
/// ellipsis to the common rank required by the ONNX Einsum schema. Operands without an
/// ellipsis simply do not contain those axes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EinsumAxis {
    /// A broadcast axis expanded from `...`.
    Ellipsis(usize),
    /// A named equation label.
    Label(EinsumLabel),
}

impl fmt::Display for EinsumAxis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ellipsis(index) => write!(f, "ellipsis axis #{index}"),
            Self::Label(label) => write!(f, "label `{label}`"),
        }
    }
}

/// One physical input axis participating in a logical einsum axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EinsumAxisRef {
    input: usize,
    axis: usize,
}

impl EinsumAxisRef {
    /// The zero-based input index.
    pub const fn input(self) -> usize {
        self.input
    }

    /// The zero-based physical axis in that input.
    pub const fn axis(self) -> usize {
        self.axis
    }
}

/// The statically known portion of an einsum extent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EinsumDimension {
    /// A concrete extent.
    Static(usize),
    /// A symbolic or otherwise unknown extent.
    Dynamic,
}

impl EinsumDimension {
    /// The concrete extent, when statically known.
    pub const fn as_static(self) -> Option<usize> {
        match self {
            Self::Static(value) => Some(value),
            Self::Dynamic => None,
        }
    }
}

/// The compatibility rule for all occurrences of a logical axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EinsumDimensionRule {
    /// Named labels must have exactly equal extents.
    Equal,
    /// Right-aligned ellipsis dimensions follow NumPy broadcast rules.
    Broadcast,
}

/// The canonical description of one logical axis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EinsumLogicalAxis {
    axis: EinsumAxis,
    occurrences: Vec<EinsumAxisRef>,
    dimension: EinsumDimension,
    rule: EinsumDimensionRule,
    requires_runtime_check: bool,
    representative: EinsumAxisRef,
    output_position: Option<usize>,
}

impl EinsumLogicalAxis {
    /// The logical label or expanded ellipsis axis.
    pub const fn axis(&self) -> EinsumAxis {
        self.axis
    }

    /// All physical axes governed by this logical axis.
    pub fn occurrences(&self) -> &[EinsumAxisRef] {
        &self.occurrences
    }

    /// The statically resolved extent, or [`EinsumDimension::Dynamic`].
    pub const fn dimension(&self) -> EinsumDimension {
        self.dimension
    }

    /// Equality for labels, broadcasting for ellipsis axes.
    pub const fn rule(&self) -> EinsumDimensionRule {
        self.rule
    }

    /// Whether concrete execution must re-check this dynamic constraint.
    pub const fn requires_runtime_check(&self) -> bool {
        self.requires_runtime_check
    }

    /// The physical dimension selected for a named output label.
    ///
    /// A concrete occurrence wins over a dynamic one; otherwise the last
    /// occurrence wins, matching the prior shape-inference representative
    /// behavior.
    pub const fn representative(&self) -> EinsumAxisRef {
        self.representative
    }

    /// This axis's position in the output, or `None` when it is reduced.
    pub const fn output_position(&self) -> Option<usize> {
        self.output_position
    }
}

/// A unique logical axis in one operand, after repeated-label axes are folded
/// into a diagonal view.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EinsumOperandAxis {
    axis: EinsumAxis,
    input_axes: Vec<usize>,
    dimension: EinsumDimension,
}

impl EinsumOperandAxis {
    /// The logical axis.
    pub const fn axis(&self) -> EinsumAxis {
        self.axis
    }

    /// Physical axes combined into this axis. More than one means diagonal
    /// extraction; a strided view uses the sum of those axes' strides.
    pub fn input_axes(&self) -> &[usize] {
        &self.input_axes
    }

    /// The statically resolved extent within this operand.
    pub const fn dimension(&self) -> EinsumDimension {
        self.dimension
    }
}

/// The canonical mapping for one input operand.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EinsumOperandPlan {
    input: usize,
    rank: usize,
    has_ellipsis: bool,
    ellipsis_rank: usize,
    shape: Vec<EinsumDimension>,
    axes: Vec<EinsumAxis>,
    unique_axes: Vec<EinsumOperandAxis>,
    diagonal_axis_indices: Vec<usize>,
    static_numel: Option<usize>,
}

impl EinsumOperandPlan {
    /// The input index.
    pub const fn input(&self) -> usize {
        self.input
    }

    /// The original input rank.
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// Whether the source term contained `...`, including a zero-rank
    /// ellipsis in a scalar/fully-named input.
    pub const fn has_ellipsis(&self) -> bool {
        self.has_ellipsis
    }

    /// Number of physical axes consumed by this operand's ellipsis.
    pub const fn ellipsis_rank(&self) -> usize {
        self.ellipsis_rank
    }

    /// Static/dynamic shape summary used to admit this operand.
    pub fn shape(&self) -> &[EinsumDimension] {
        &self.shape
    }

    /// Logical axis for every physical input axis.
    pub fn axes(&self) -> &[EinsumAxis] {
        &self.axes
    }

    /// Unique logical axes in first-physical-occurrence order.
    pub fn unique_axes(&self) -> &[EinsumOperandAxis] {
        &self.unique_axes
    }

    /// Indices into [`Self::unique_axes`] whose physical-axis list extracts a
    /// diagonal.
    pub fn diagonal_axis_indices(&self) -> &[usize] {
        &self.diagonal_axis_indices
    }

    /// Static element count when every dimension is known.
    pub const fn static_numel(&self) -> Option<usize> {
        self.static_numel
    }
}

/// A pure view/permutation of one input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EinsumPermutationPlan {
    input: usize,
    output_to_operand_axis: Vec<usize>,
}

impl EinsumPermutationPlan {
    /// The source input.
    pub const fn input(&self) -> usize {
        self.input
    }

    /// For each output axis, the source index in
    /// [`EinsumOperandPlan::unique_axes`].
    pub fn output_to_operand_axis(&self) -> &[usize] {
        &self.output_to_operand_axis
    }

    /// Whether no axis reorder is required.
    pub fn is_identity(&self) -> bool {
        self.output_to_operand_axis
            .iter()
            .copied()
            .eq(0..self.output_to_operand_axis.len())
    }
}

/// An elementwise product and/or reduction lowering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EinsumReductionPlan {
    iteration_axes: Vec<EinsumAxis>,
    output_rank: usize,
    operand_axis_mappings: Vec<Vec<usize>>,
}

impl EinsumReductionPlan {
    /// Iteration order: output axes first, followed by reduction axes.
    pub fn iteration_axes(&self) -> &[EinsumAxis] {
        &self.iteration_axes
    }

    /// The number of leading iteration axes retained in the output.
    pub const fn output_rank(&self) -> usize {
        self.output_rank
    }

    /// Per input, map each unique operand axis to an iteration-axis index.
    pub fn operand_axis_mappings(&self) -> &[Vec<usize>] {
        &self.operand_axis_mappings
    }
}

/// Overflow semantics for integer Einsum arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EinsumIntegerOverflowSemantics {
    /// Every multiply/add is defined modulo `2^width`, matching the homogeneous
    /// fixed-width tensor dtype without invoking host-language signed-overflow
    /// undefined behavior.
    WrappingModuloPowerOfTwo,
}

/// Backend-neutral accumulation and intermediate-storage policy.
///
/// Fast-path selection is intentionally absent. An EP may choose scalar,
/// vector, GEMM, or tiled execution, but it must preserve these precision and
/// final-rounding boundaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EinsumPrecisionPolicy {
    input_output_dtype: DataType,
    accumulator_dtype: DataType,
    intermediate_dtype: DataType,
    narrow_once_at_output: bool,
    integer_overflow: Option<EinsumIntegerOverflowSemantics>,
}

impl EinsumPrecisionPolicy {
    fn for_dtype(dtype: DataType) -> Self {
        match dtype {
            DataType::Float16 | DataType::BFloat16 => Self {
                input_output_dtype: dtype,
                accumulator_dtype: DataType::Float32,
                intermediate_dtype: DataType::Float32,
                narrow_once_at_output: true,
                integer_overflow: None,
            },
            DataType::Float32 | DataType::Float64 => Self {
                input_output_dtype: dtype,
                accumulator_dtype: dtype,
                intermediate_dtype: dtype,
                narrow_once_at_output: false,
                integer_overflow: None,
            },
            DataType::Uint8
            | DataType::Uint16
            | DataType::Uint32
            | DataType::Uint64
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64 => Self {
                input_output_dtype: dtype,
                accumulator_dtype: dtype,
                intermediate_dtype: dtype,
                narrow_once_at_output: false,
                integer_overflow: Some(EinsumIntegerOverflowSemantics::WrappingModuloPowerOfTwo),
            },
            _ => unreachable!("schema validation admits only Einsum numeric dtypes"),
        }
    }

    /// Homogeneous ONNX input/output dtype.
    pub const fn input_output_dtype(self) -> DataType {
        self.input_output_dtype
    }

    /// Dtype used for multiplication and accumulation.
    pub const fn accumulator_dtype(self) -> DataType {
        self.accumulator_dtype
    }

    /// Dtype used for every non-final materialized intermediate.
    pub const fn intermediate_dtype(self) -> DataType {
        self.intermediate_dtype
    }

    /// Whether the accumulator/intermediate is narrowed exactly once, when the
    /// final output tensor is written.
    pub const fn narrow_once_at_output(self) -> bool {
        self.narrow_once_at_output
    }

    /// Defined fixed-width integer overflow behavior, when applicable.
    pub const fn integer_overflow(self) -> Option<EinsumIntegerOverflowSemantics> {
        self.integer_overflow
    }

    /// Fixed byte width charged for materialized intermediates.
    pub fn intermediate_element_size(self) -> usize {
        self.intermediate_dtype.byte_size()
    }
}

/// Index mapping for one source operand in the universal generic program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EinsumOperandIndexProgram {
    input: usize,
    physical_axis_to_iteration_axis: Vec<usize>,
    physical_axis_broadcasts_when_one: Vec<bool>,
}

impl EinsumOperandIndexProgram {
    /// Source operand index.
    pub const fn input(&self) -> usize {
        self.input
    }

    /// For every physical source axis, the generic iteration-axis index.
    ///
    /// Repeated labels map multiple physical axes to the same index, which is
    /// the diagonal constraint. Missing axes are implicit broadcast axes.
    pub fn physical_axis_to_iteration_axis(&self) -> &[usize] {
        &self.physical_axis_to_iteration_axis
    }

    /// Per physical axis, whether extent `1` means use source index zero while
    /// the logical ellipsis iteration axis expands to a larger extent.
    pub fn physical_axis_broadcasts_when_one(&self) -> &[bool] {
        &self.physical_axis_broadcasts_when_one
    }
}

/// Universal, backend-neutral loop/index program for a legal equation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EinsumIndexProgram {
    iteration_axes: Vec<EinsumAxis>,
    output_rank: usize,
    operands: Vec<EinsumOperandIndexProgram>,
}

impl EinsumIndexProgram {
    /// Iteration order: requested output axes first, followed by every reduced
    /// axis in canonical order.
    pub fn iteration_axes(&self) -> &[EinsumAxis] {
        &self.iteration_axes
    }

    /// Number of leading iteration axes retained in the output.
    pub const fn output_rank(&self) -> usize {
        self.output_rank
    }

    /// Physical-to-logical index maps for every operand.
    pub fn operands(&self) -> &[EinsumOperandIndexProgram] {
        &self.operands
    }
}

/// Mandatory generic-native fallback for every schema-valid equation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EinsumGenericNativePlan {
    index_program: EinsumIndexProgram,
}

impl EinsumGenericNativePlan {
    /// Generic loop/index program. EPs may tile it to obey a memory ceiling.
    pub const fn index_program(&self) -> &EinsumIndexProgram {
        &self.index_program
    }
}

/// Complete semantic plan independent of any one fast-path family.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EinsumSemanticPlan {
    generic_native: EinsumGenericNativePlan,
    contraction_tree: Option<EinsumContractionTreePlan>,
}

impl EinsumSemanticPlan {
    /// Universal fallback that exists for every legal equation.
    pub const fn generic_native(&self) -> &EinsumGenericNativePlan {
        &self.generic_native
    }

    /// Bounded binary contraction plan for multi-operand equations.
    ///
    /// This is available even when a simpler classification (for example an
    /// outer product) is also present, so GEMM is an optimization subtype
    /// rather than the semantic boundary.
    pub const fn contraction_tree(&self) -> Option<&EinsumContractionTreePlan> {
        self.contraction_tree.as_ref()
    }
}

/// Memory-aware semantic execution choice.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EinsumExecutionSelection {
    /// Execute the named bounded contraction-tree candidate.
    ContractionTree(EinsumContractionTreeCandidateId),
    /// Execute/tile the mandatory universal index program.
    GenericNative,
}

/// Flattened GEMM/BMM geometry known at planning time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EinsumGemmGeometry {
    batch_shape: Vec<EinsumDimension>,
    batch: EinsumDimension,
    m: EinsumDimension,
    k: EinsumDimension,
    n: EinsumDimension,
}

/// Fully concrete checked GEMM/BMM geometry resolved from a canonical plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EinsumConcreteGemmGeometry {
    batch_shape: Vec<usize>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
}

impl EinsumConcreteGemmGeometry {
    /// Concrete broadcast batch shape.
    pub fn batch_shape(&self) -> &[usize] {
        &self.batch_shape
    }

    /// Product of batch dimensions.
    pub const fn batch(&self) -> usize {
        self.batch
    }

    /// Flattened GEMM `M`.
    pub const fn m(&self) -> usize {
        self.m
    }

    /// Flattened GEMM `K`.
    pub const fn k(&self) -> usize {
        self.k
    }

    /// Flattened GEMM `N`.
    pub const fn n(&self) -> usize {
        self.n
    }
}

impl EinsumGemmGeometry {
    /// Canonical broadcast batch shape.
    pub fn batch_shape(&self) -> &[EinsumDimension] {
        &self.batch_shape
    }

    /// Product of the batch extents.
    pub const fn batch(&self) -> EinsumDimension {
        self.batch
    }

    /// Product of left-free extents, or `1` for a vector/dot-product side.
    pub const fn m(&self) -> EinsumDimension {
        self.m
    }

    /// Product of contracted extents.
    pub const fn k(&self) -> EinsumDimension {
        self.k
    }

    /// Product of right-free extents, or `1` for a vector/dot-product side.
    pub const fn n(&self) -> EinsumDimension {
        self.n
    }
}

/// A binary contraction directly lowerable to GEMM/BMM after diagonal views,
/// singleton insertion, permutation, and flattening.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EinsumContractionPlan {
    batch_axes: Vec<EinsumAxis>,
    left_free_axes: Vec<EinsumAxis>,
    contract_axes: Vec<EinsumAxis>,
    right_free_axes: Vec<EinsumAxis>,
    left_axis_order: Vec<Option<usize>>,
    right_axis_order: Vec<Option<usize>>,
    output_permutation: Vec<usize>,
    geometry: EinsumGemmGeometry,
}

impl EinsumContractionPlan {
    /// Canonical B/BMM batch axes. Named axes are equal; ellipsis axes may
    /// broadcast and may be absent (`1`) on either operand.
    pub fn batch_axes(&self) -> &[EinsumAxis] {
        &self.batch_axes
    }

    /// Axes flattened into GEMM `M`.
    pub fn left_free_axes(&self) -> &[EinsumAxis] {
        &self.left_free_axes
    }

    /// Axes flattened into GEMM `K`.
    pub fn contract_axes(&self) -> &[EinsumAxis] {
        &self.contract_axes
    }

    /// Axes flattened into GEMM `N`.
    pub fn right_free_axes(&self) -> &[EinsumAxis] {
        &self.right_free_axes
    }

    /// Target left layout `[batch..., M..., K...]`. Each entry is an index into
    /// input 0's unique axes; `None` inserts a broadcast singleton.
    pub fn left_axis_order(&self) -> &[Option<usize>] {
        &self.left_axis_order
    }

    /// Target right layout `[batch..., K..., N...]`. Each entry is an index into
    /// input 1's unique axes; `None` inserts a broadcast singleton.
    pub fn right_axis_order(&self) -> &[Option<usize>] {
        &self.right_axis_order
    }

    /// For every requested output axis, its index in the canonical GEMM result
    /// `[batch..., M..., N...]`.
    pub fn output_permutation(&self) -> &[usize] {
        &self.output_permutation
    }

    /// Statically known flattened geometry.
    pub const fn geometry(&self) -> &EinsumGemmGeometry {
        &self.geometry
    }
}

/// Structural semantics of a validated equation.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EinsumClassification {
    /// One-input permutation/identity with no arithmetic.
    ViewOnlyPermutation(EinsumPermutationPlan),
    /// One-input diagonal extraction, optionally followed by a permutation.
    DiagonalView(EinsumPermutationPlan),
    /// Elementwise product and/or reduction without a coupled contraction.
    ReductionOrElementwise(EinsumReductionPlan),
    /// Binary GEMM/BMM-compatible contraction.
    Gemm(EinsumContractionPlan),
    /// Bounded ordered binary-tree planning for a general coupled contraction.
    ContractionTree(EinsumContractionTreePlan),
}

/// Which checked static product overflowed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EinsumOverflowTarget {
    /// Full element count for an input.
    Input(usize),
    /// Full output element count.
    Output,
    /// Flattened BMM batch product.
    GemmBatch,
    /// Flattened GEMM `M`.
    GemmM,
    /// Flattened GEMM `K`.
    GemmK,
    /// Flattened GEMM `N`.
    GemmN,
}

impl fmt::Display for EinsumOverflowTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(input) => write!(f, "input #{input} element count"),
            Self::Output => f.write_str("output element count"),
            Self::GemmBatch => f.write_str("flattened GEMM batch"),
            Self::GemmM => f.write_str("flattened GEMM M"),
            Self::GemmK => f.write_str("flattened GEMM K"),
            Self::GemmN => f.write_str("flattened GEMM N"),
        }
    }
}

/// Side of the normalized equation containing a syntax error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EinsumEquationSide {
    /// A left-hand-side input term.
    Input(usize),
    /// The explicit output term.
    Output,
}

impl fmt::Display for EinsumEquationSide {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(input) => write!(f, "input term #{input}"),
            Self::Output => f.write_str("output term"),
        }
    }
}

/// Structured cause of a planning failure.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EinsumPlanErrorKind {
    /// Einsum requires at least one input.
    NoInputs,
    /// Equation term count does not match node input count.
    InputCount {
        /// Terms in the equation.
        equation_terms: usize,
        /// Inputs supplied by the caller.
        inputs: usize,
    },
    /// More than one explicit-output arrow was present.
    MultipleOutputArrows,
    /// A term contained a character outside ASCII letters and one `...`.
    InvalidCharacter {
        /// Input or output term.
        side: EinsumEquationSide,
        /// Byte offset in the normalized term.
        offset: usize,
        /// Rejected character.
        found: char,
    },
    /// A term contained multiple ellipses.
    MultipleEllipses {
        /// Input or output term.
        side: EinsumEquationSide,
    },
    /// The effective imported `ai.onnx` opset predates `Einsum-12`.
    UnsupportedOpset {
        /// Effective imported opset.
        imported_opset: u64,
    },
    /// An input has no dtype metadata.
    MissingInputDtype {
        /// Input index.
        input: usize,
    },
    /// An input has no shape/rank metadata.
    MissingInputShape {
        /// Input index.
        input: usize,
    },
    /// An input uses a dtype outside the resolved schema's type constraint.
    UnsupportedInputDtype {
        /// Input index.
        input: usize,
        /// Rejected dtype.
        dtype: DataType,
        /// Resolved schema proof.
        schema: EinsumSchema,
    },
    /// Inputs do not share the single variadic type parameter `T`.
    InputDtypeMismatch {
        /// Input index.
        input: usize,
        /// Dtype established by the first available input type.
        expected: DataType,
        /// Rejected dtype.
        actual: DataType,
    },
    /// A term's named labels cannot describe its input rank.
    InputRankMismatch {
        /// Input index.
        input: usize,
        /// Actual rank.
        rank: usize,
        /// Number of named labels.
        named_labels: usize,
        /// Whether the term contains an ellipsis.
        has_ellipsis: bool,
    },
    /// Two explicit ellipses expand to different numbers of dimensions.
    EllipsisRankMismatch {
        /// First input term containing an ellipsis.
        first_input: usize,
        /// Expansion rank established by the first ellipsis.
        first_rank: usize,
        /// Later input term containing an incompatible ellipsis.
        input: usize,
        /// Expansion rank of the incompatible ellipsis.
        rank: usize,
    },
    /// Concrete shapes supplied to an existing plan have the wrong count.
    ResolvedInputCountMismatch {
        /// Count used to build the plan.
        expected: usize,
        /// Count supplied for execution.
        found: usize,
    },
    /// A concrete execution shape has the wrong rank.
    ResolvedInputRankMismatch {
        /// Input index.
        input: usize,
        /// Rank used to build the plan.
        expected: usize,
        /// Rank supplied for execution.
        found: usize,
    },
    /// A concrete execution shape disagrees with a statically admitted axis.
    ResolvedInputDimensionMismatch {
        /// Input index.
        input: usize,
        /// Physical input axis.
        axis: usize,
        /// Extent used to build the plan.
        expected: usize,
        /// Extent supplied for execution.
        found: usize,
    },
    /// A named label's occurrence counter overflowed.
    LabelMultiplicityOverflow {
        /// Label being counted.
        label: EinsumLabel,
    },
    /// The explicit output repeats a label.
    DuplicateOutputLabel {
        /// Repeated label.
        label: EinsumLabel,
    },
    /// The explicit output names a label absent from all inputs.
    OutputLabelMissingFromInputs {
        /// Missing label.
        label: EinsumLabel,
    },
    /// Equal-label dimensions are statically incompatible.
    LabelDimensionMismatch {
        /// Repeated/shared label.
        label: EinsumLabel,
        /// First conflicting physical axis.
        first: EinsumAxisRef,
        /// First extent.
        first_size: usize,
        /// Second conflicting physical axis.
        second: EinsumAxisRef,
        /// Second extent.
        second_size: usize,
    },
    /// Ellipsis dimensions are statically broadcast-incompatible.
    EllipsisDimensionMismatch {
        /// Canonical left-to-right ellipsis axis.
        axis: usize,
        /// First conflicting physical axis.
        first: EinsumAxisRef,
        /// First extent.
        first_size: usize,
        /// Second conflicting physical axis.
        second: EinsumAxisRef,
        /// Second extent.
        second_size: usize,
    },
    /// A fully static geometry product exceeded `usize`.
    GeometryOverflow {
        /// Product that overflowed.
        target: EinsumOverflowTarget,
    },
    /// Runtime tree re-scoring requires a non-zero fixed-width element size.
    InvalidElementSize {
        /// Rejected byte width.
        element_size: usize,
    },
}

/// An actionable Einsum planning error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EinsumPlanError {
    equation: String,
    kind: EinsumPlanErrorKind,
}

impl EinsumPlanError {
    fn new(equation: &str, kind: EinsumPlanErrorKind) -> Self {
        Self {
            equation: equation.to_owned(),
            kind,
        }
    }

    /// The normalized equation that failed.
    pub fn equation(&self) -> &str {
        &self.equation
    }

    /// The structured failure cause.
    pub const fn kind(&self) -> &EinsumPlanErrorKind {
        &self.kind
    }

    /// Whether this error represents incomplete graph metadata rather than a
    /// malformed or incompatible einsum.
    pub const fn is_incomplete_metadata(&self) -> bool {
        matches!(
            self.kind,
            EinsumPlanErrorKind::MissingInputDtype { .. }
                | EinsumPlanErrorKind::MissingInputShape { .. }
        )
    }
}

impl fmt::Display for EinsumPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Einsum equation `{}`: ", self.equation)?;
        match &self.kind {
            EinsumPlanErrorKind::NoInputs => f.write_str("expected at least one input, found none"),
            EinsumPlanErrorKind::InputCount {
                equation_terms,
                inputs,
            } => write!(
                f,
                "equation has {equation_terms} input terms but the node has {inputs} inputs"
            ),
            EinsumPlanErrorKind::MultipleOutputArrows => {
                f.write_str("contains more than one `->` output separator")
            }
            EinsumPlanErrorKind::InvalidCharacter {
                side,
                offset,
                found,
            } => write!(
                f,
                "{side} has invalid character `{found}` at normalized byte offset {offset}; expected case-sensitive ASCII letters or one `...`"
            ),
            EinsumPlanErrorKind::MultipleEllipses { side } => {
                write!(f, "{side} contains more than one ellipsis")
            }
            EinsumPlanErrorKind::UnsupportedOpset { imported_opset } => write!(
                f,
                "effective ai.onnx opset {imported_opset} predates Einsum-12; HOW: export this node with ai.onnx opset >= 12"
            ),
            EinsumPlanErrorKind::MissingInputDtype { input } => {
                write!(f, "input #{input} has no available tensor dtype")
            }
            EinsumPlanErrorKind::MissingInputShape { input } => {
                write!(f, "input #{input} has no available tensor shape/rank")
            }
            EinsumPlanErrorKind::UnsupportedInputDtype {
                input,
                dtype,
                schema,
            } => write!(
                f,
                "input #{input} has dtype {dtype:?}, which is not admitted by {schema}; expected \
                 Uint8, Uint16, Uint32, Uint64, Int8, Int16, Int32, Int64, Float16, Float32, or \
                 Float64{}. HOW: cast every operand to a schema-supported homogeneous dtype or \
                 import ai.onnx opset 28+ when BFloat16 is required",
                if *schema == EinsumSchema::V28 {
                    ", or BFloat16"
                } else {
                    ""
                }
            ),
            EinsumPlanErrorKind::InputDtypeMismatch {
                input,
                expected,
                actual,
            } => write!(
                f,
                "input #{input} has dtype {actual:?}, but another known input established homogeneous dtype {expected:?}"
            ),
            EinsumPlanErrorKind::InputRankMismatch {
                input,
                rank,
                named_labels,
                has_ellipsis,
            } => {
                if *has_ellipsis {
                    write!(
                        f,
                        "input #{input} rank {rank} is smaller than its {named_labels} named labels, so `...` cannot expand"
                    )
                } else {
                    write!(
                        f,
                        "input #{input} rank {rank} does not match its {named_labels} equation labels"
                    )
                }
            }
            EinsumPlanErrorKind::EllipsisRankMismatch {
                first_input,
                first_rank,
                input,
                rank,
            } => write!(
                f,
                "input term #{input} explicit ellipsis has expansion rank {rank}, but input term #{first_input} explicit ellipsis has expansion rank {first_rank}; ONNX Einsum requires every explicit ellipsis to represent the same number of dimensions"
            ),
            EinsumPlanErrorKind::ResolvedInputCountMismatch { expected, found } => write!(
                f,
                "plan was built for {expected} inputs, but execution supplied {found}"
            ),
            EinsumPlanErrorKind::ResolvedInputRankMismatch {
                input,
                expected,
                found,
            } => write!(
                f,
                "plan was built for input #{input} rank {expected}, but execution supplied rank {found}"
            ),
            EinsumPlanErrorKind::ResolvedInputDimensionMismatch {
                input,
                axis,
                expected,
                found,
            } => write!(
                f,
                "plan fixed input #{input} axis {axis} at {expected}, but execution supplied {found}"
            ),
            EinsumPlanErrorKind::LabelMultiplicityOverflow { label } => {
                write!(f, "occurrence count for label `{label}` overflowed")
            }
            EinsumPlanErrorKind::DuplicateOutputLabel { label } => {
                write!(f, "output label `{label}` appears more than once")
            }
            EinsumPlanErrorKind::OutputLabelMissingFromInputs { label } => {
                write!(f, "output label `{label}` does not appear in any input")
            }
            EinsumPlanErrorKind::LabelDimensionMismatch {
                label,
                first,
                first_size,
                second,
                second_size,
            } => write!(
                f,
                "label `{label}` requires equal dimensions, but input #{} axis {} is {first_size} and input #{} axis {} is {second_size}",
                first.input, first.axis, second.input, second.axis
            ),
            EinsumPlanErrorKind::EllipsisDimensionMismatch {
                axis,
                first,
                first_size,
                second,
                second_size,
            } => write!(
                f,
                "ellipsis axis #{axis} cannot broadcast input #{} axis {} ({first_size}) with input #{} axis {} ({second_size})",
                first.input, first.axis, second.input, second.axis
            ),
            EinsumPlanErrorKind::GeometryOverflow { target } => {
                write!(f, "{target} overflows `usize`; use smaller dimensions")
            }
            EinsumPlanErrorKind::InvalidElementSize { element_size } => write!(
                f,
                "runtime contraction-tree re-score received element size {element_size}; provide a non-zero fixed-width dtype byte size"
            ),
        }
    }
}

impl std::error::Error for EinsumPlanError {}

/// Failure while resolving a plan's output shape with caller-owned symbolic
/// dimensions.
#[derive(Debug, PartialEq, Eq)]
pub enum EinsumResolveError<E> {
    /// The caller did not provide the same number of shapes used to build the
    /// plan.
    InputCount {
        /// Expected shape count.
        expected: usize,
        /// Supplied shape count.
        found: usize,
    },
    /// A supplied rank differs from the rank used to build the plan.
    InputRank {
        /// Input index.
        input: usize,
        /// Planned rank.
        expected: usize,
        /// Supplied rank.
        found: usize,
    },
    /// The caller's symbolic broadcast resolver rejected an ellipsis axis.
    Broadcast {
        /// Logical ellipsis axis.
        axis: EinsumAxis,
        /// Resolver-specific cause.
        source: E,
    },
}

/// A normalized, validated, immutable typed Einsum plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EinsumPlan {
    dtype: DataType,
    precision: EinsumPrecisionPolicy,
    shape: EinsumShapePlan,
}

impl EinsumPlan {
    /// Parse and validate using the `Einsum-12` contract.
    ///
    /// This source-compatible entry point remains explicitly v12-compatible.
    /// Model-facing paths must resolve the imported opset and call
    /// [`Self::build_for_schema`] or [`Self::build_for_opset`] instead.
    pub fn build<D: EinsumDimensionValue>(
        equation: &str,
        inputs: &[EinsumInput<'_, D>],
    ) -> Result<Self, EinsumPlanError> {
        Self::build_for_schema(equation, inputs, EinsumSchema::V12)
    }

    /// Resolve the applicable schema from `imported_opset`, then build.
    pub fn build_for_opset<D: EinsumDimensionValue>(
        equation: &str,
        inputs: &[EinsumInput<'_, D>],
        imported_opset: u64,
    ) -> Result<Self, EinsumPlanError> {
        let schema = EinsumSchema::resolve(imported_opset).map_err(|_| {
            EinsumPlanError::new(
                equation,
                EinsumPlanErrorKind::UnsupportedOpset { imported_opset },
            )
        })?;
        Self::build_for_schema(equation, inputs, schema)
    }

    /// Parse and validate against an already-resolved schema proof.
    pub fn build_for_schema<D: EinsumDimensionValue>(
        equation: &str,
        inputs: &[EinsumInput<'_, D>],
        schema: EinsumSchema,
    ) -> Result<Self, EinsumPlanError> {
        Self::build_for_schema_with_budget(equation, inputs, schema, EinsumPlannerBudget::default())
    }

    /// Build with explicit bounded-planner budgets.
    pub fn build_for_schema_with_budget<D: EinsumDimensionValue>(
        equation: &str,
        inputs: &[EinsumInput<'_, D>],
        schema: EinsumSchema,
        planner_budget: EinsumPlannerBudget,
    ) -> Result<Self, EinsumPlanError> {
        let normalized = normalize_equation(equation);
        let parsed = ParsedEquation::parse(&normalized, inputs.len())?;
        if inputs.iter().any(|input| input.shape.is_none()) {
            validate_partial_input_shapes(&normalized, &parsed, inputs, schema, planner_budget)?;
        }
        let (dtype, input_meta) = validate_inputs(&normalized, inputs, schema)?;
        let precision = EinsumPrecisionPolicy::for_dtype(dtype);
        let shape = build_shape_plan(normalized, parsed, input_meta, schema, planner_budget)?;
        Ok(Self {
            dtype,
            precision,
            shape,
        })
    }

    /// Shared input/output dtype.
    pub const fn dtype(&self) -> DataType {
        self.dtype
    }

    /// Resolved ONNX schema proof.
    pub const fn schema(&self) -> EinsumSchema {
        self.shape.schema()
    }

    /// Explicit accumulator/intermediate/final-rounding policy.
    pub const fn precision_policy(&self) -> EinsumPrecisionPolicy {
        self.precision
    }

    /// Complete semantic plan, including the mandatory generic fallback.
    pub const fn semantic_plan(&self) -> &EinsumSemanticPlan {
        self.shape.semantic_plan()
    }

    /// Mandatory generic-native fallback.
    pub const fn generic_native(&self) -> &EinsumGenericNativePlan {
        self.shape.generic_native()
    }

    /// The structural plan, which carries no dtype claim.
    pub const fn shape_plan(&self) -> &EinsumShapePlan {
        &self.shape
    }

    /// Whitespace-free canonical equation.
    pub fn equation(&self) -> &str {
        self.shape.equation()
    }

    /// Whether the source equation supplied an explicit `->` output.
    pub const fn has_explicit_output(&self) -> bool {
        self.shape.has_explicit_output()
    }

    /// Canonical operand mappings.
    pub fn operands(&self) -> &[EinsumOperandPlan] {
        self.shape.operands()
    }

    /// Canonical logical axes, in ellipsis-then-ASCII-label order.
    pub fn logical_axes(&self) -> &[EinsumLogicalAxis] {
        self.shape.logical_axes()
    }

    /// Requested output axes in exact output order.
    pub fn output_axes(&self) -> &[EinsumAxis] {
        self.shape.output_axes()
    }

    /// Statically known output extents.
    pub fn output_shape(&self) -> &[EinsumDimension] {
        self.shape.output_shape()
    }

    /// Logical axes summed out by the equation.
    pub fn reduction_axes(&self) -> &[EinsumAxis] {
        self.shape.reduction_axes()
    }

    /// Structural lowering class and its complete axis mappings.
    pub const fn classification(&self) -> &EinsumClassification {
        self.shape.classification()
    }

    /// Static output element count when every output dimension is known.
    pub const fn static_output_numel(&self) -> Option<usize> {
        self.shape.static_output_numel()
    }

    /// Resolve the output using the exact symbolic dimensions from which this
    /// plan was built.
    pub fn resolve_output_shape<D: Clone, E>(
        &self,
        input_shapes: &[&[D]],
        broadcast: impl FnMut(&D, &D) -> Result<D, E>,
    ) -> Result<Vec<D>, EinsumResolveError<E>> {
        self.shape.resolve_output_shape(input_shapes, broadcast)
    }

    /// Validate concrete runtime shapes against this already-parsed plan and
    /// return the exact output shape without reparsing the equation.
    pub fn resolve_concrete_output_shape(
        &self,
        input_shapes: &[&[usize]],
    ) -> Result<Vec<usize>, EinsumPlanError> {
        self.shape.resolve_concrete_output_shape(input_shapes)
    }

    /// Resolve checked concrete B/M/K/N geometry for a GEMM/BMM-classified
    /// plan. Returns `Ok(None)` for every other classification.
    pub fn resolve_concrete_gemm_geometry(
        &self,
        input_shapes: &[&[usize]],
    ) -> Result<Option<EinsumConcreteGemmGeometry>, EinsumPlanError> {
        self.shape.resolve_concrete_gemm_geometry(input_shapes)
    }

    /// Re-score/replan bounded contraction-tree candidates for concrete
    /// runtime shapes using the semantic intermediate dtype.
    pub fn resolve_concrete_contraction_tree(
        &self,
        input_shapes: &[&[usize]],
    ) -> Result<Option<EinsumConcreteContractionTreePlan>, EinsumPlanError> {
        self.shape.resolve_concrete_contraction_tree(
            input_shapes,
            self.precision.intermediate_element_size(),
        )
    }

    /// Select a concrete contraction tree under `memory_ceiling_bytes`, or the
    /// mandatory generic/tiled program when no candidate fits.
    pub fn select_concrete_execution(
        &self,
        input_shapes: &[&[usize]],
        memory_ceiling_bytes: u128,
    ) -> Result<EinsumExecutionSelection, EinsumPlanError> {
        let Some(tree) = self.resolve_concrete_contraction_tree(input_shapes)? else {
            return Ok(EinsumExecutionSelection::GenericNative);
        };
        Ok(tree
            .preferred_candidate_with_memory_ceiling(memory_ceiling_bytes)
            .map(|candidate| EinsumExecutionSelection::ContractionTree(candidate.id().clone()))
            .unwrap_or(EinsumExecutionSelection::GenericNative))
    }
}

/// A normalized, validated, immutable Einsum equation/shape plan with no dtype.
///
/// This representation is for consumers whose construction API receives
/// shapes but not tensor dtypes. It cannot be mistaken for a typed
/// [`EinsumPlan`] and therefore cannot claim a fabricated runtime dtype.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EinsumShapePlan {
    schema: EinsumSchema,
    equation: String,
    explicit_output: bool,
    operands: Vec<EinsumOperandPlan>,
    logical_axes: Vec<EinsumLogicalAxis>,
    output_axes: Vec<EinsumAxis>,
    output_shape: Vec<EinsumDimension>,
    reduction_axes: Vec<EinsumAxis>,
    classification: EinsumClassification,
    semantic: EinsumSemanticPlan,
    static_output_numel: Option<usize>,
}

impl EinsumShapePlan {
    /// Parse and validate shapes using the `Einsum-12` contract.
    ///
    /// This entry point intentionally performs no dtype validation. It is for
    /// kernel factories whose API receives concrete shapes but not tensor
    /// dtypes; their provider capability gate and runtime kernel must validate
    /// the real dtype separately. Using a fabricated dtype here would let a
    /// direct factory call bypass the canonical ONNX type contract.
    pub fn build<D: EinsumDimensionValue>(
        equation: &str,
        input_shapes: &[&[D]],
    ) -> Result<Self, EinsumPlanError> {
        Self::build_for_schema(equation, input_shapes, EinsumSchema::V12)
    }

    /// Resolve the applicable schema from `imported_opset`, then build.
    pub fn build_for_opset<D: EinsumDimensionValue>(
        equation: &str,
        input_shapes: &[&[D]],
        imported_opset: u64,
    ) -> Result<Self, EinsumPlanError> {
        let schema = EinsumSchema::resolve(imported_opset).map_err(|_| {
            EinsumPlanError::new(
                equation,
                EinsumPlanErrorKind::UnsupportedOpset { imported_opset },
            )
        })?;
        Self::build_for_schema(equation, input_shapes, schema)
    }

    /// Parse and validate shapes against an already-resolved schema proof.
    pub fn build_for_schema<D: EinsumDimensionValue>(
        equation: &str,
        input_shapes: &[&[D]],
        schema: EinsumSchema,
    ) -> Result<Self, EinsumPlanError> {
        Self::build_for_schema_with_budget(
            equation,
            input_shapes,
            schema,
            EinsumPlannerBudget::default(),
        )
    }

    /// Shape-only build with explicit bounded-planner budgets.
    pub fn build_for_schema_with_budget<D: EinsumDimensionValue>(
        equation: &str,
        input_shapes: &[&[D]],
        schema: EinsumSchema,
        planner_budget: EinsumPlannerBudget,
    ) -> Result<Self, EinsumPlanError> {
        let normalized = normalize_equation(equation);
        let parsed = ParsedEquation::parse(&normalized, input_shapes.len())?;
        let input_meta = input_shapes
            .iter()
            .map(|shape| InputMeta {
                shape: shape
                    .iter()
                    .map(|dimension| {
                        dimension
                            .einsum_static_size()
                            .map_or(EinsumDimension::Dynamic, EinsumDimension::Static)
                    })
                    .collect(),
            })
            .collect();
        build_shape_plan(normalized, parsed, input_meta, schema, planner_budget)
    }

    /// Resolved ONNX schema proof.
    pub const fn schema(&self) -> EinsumSchema {
        self.schema
    }

    /// Complete semantic plan, including the mandatory generic fallback.
    pub const fn semantic_plan(&self) -> &EinsumSemanticPlan {
        &self.semantic
    }

    /// Mandatory generic-native fallback.
    pub const fn generic_native(&self) -> &EinsumGenericNativePlan {
        self.semantic.generic_native()
    }

    /// Whitespace-free canonical equation.
    pub fn equation(&self) -> &str {
        &self.equation
    }

    /// Whether the source equation supplied an explicit `->` output.
    pub const fn has_explicit_output(&self) -> bool {
        self.explicit_output
    }

    /// Canonical operand mappings.
    pub fn operands(&self) -> &[EinsumOperandPlan] {
        &self.operands
    }

    /// Canonical logical axes, in ellipsis-then-ASCII-label order.
    pub fn logical_axes(&self) -> &[EinsumLogicalAxis] {
        &self.logical_axes
    }

    /// Requested output axes in exact output order.
    pub fn output_axes(&self) -> &[EinsumAxis] {
        &self.output_axes
    }

    /// Statically known output extents.
    pub fn output_shape(&self) -> &[EinsumDimension] {
        &self.output_shape
    }

    /// Logical axes summed out by the equation.
    pub fn reduction_axes(&self) -> &[EinsumAxis] {
        &self.reduction_axes
    }

    /// Structural lowering class and its complete axis mappings.
    pub const fn classification(&self) -> &EinsumClassification {
        &self.classification
    }

    /// Static output element count when every output dimension is known.
    pub const fn static_output_numel(&self) -> Option<usize> {
        self.static_output_numel
    }

    /// Resolve the output using the exact symbolic dimensions from which this
    /// plan was built.
    ///
    /// Named axes select the plan's validated representative. Expanded
    /// ellipsis axes are folded in input order through `broadcast`, allowing
    /// shape inference to preserve its symbol-unification lineage.
    pub fn resolve_output_shape<D: Clone, E>(
        &self,
        input_shapes: &[&[D]],
        mut broadcast: impl FnMut(&D, &D) -> Result<D, E>,
    ) -> Result<Vec<D>, EinsumResolveError<E>> {
        if input_shapes.len() != self.operands.len() {
            return Err(EinsumResolveError::InputCount {
                expected: self.operands.len(),
                found: input_shapes.len(),
            });
        }
        for (input, (shape, operand)) in input_shapes.iter().zip(&self.operands).enumerate() {
            if shape.len() != operand.rank {
                return Err(EinsumResolveError::InputRank {
                    input,
                    expected: operand.rank,
                    found: shape.len(),
                });
            }
        }

        let by_axis: BTreeMap<_, _> = self
            .logical_axes
            .iter()
            .map(|logical| (logical.axis, logical))
            .collect();
        // Reproduce the shape-inference handler's historical input-by-input
        // broadcast order exactly. This is observable for symbolic dimensions:
        // each callback may mint a representative and record lineage.
        let mut resolved_ellipsis: Option<Vec<D>> = None;
        for (shape, operand) in input_shapes.iter().zip(&self.operands) {
            if !operand.has_ellipsis {
                continue;
            }
            let current: Vec<_> = operand
                .axes
                .iter()
                .enumerate()
                .filter(|(_, logical)| matches!(logical, EinsumAxis::Ellipsis(_)))
                .map(|(axis, _)| shape[axis].clone())
                .collect();
            resolved_ellipsis = Some(match resolved_ellipsis {
                None => current,
                Some(existing) => {
                    debug_assert_eq!(
                        existing.len(),
                        current.len(),
                        "validated explicit ellipses have one common expansion rank"
                    );
                    let mut result = Vec::with_capacity(existing.len());
                    for (axis, (left, right)) in
                        existing.into_iter().zip(current.iter()).enumerate()
                    {
                        result.push(broadcast(&left, right).map_err(|source| {
                            EinsumResolveError::Broadcast {
                                axis: EinsumAxis::Ellipsis(axis),
                                source,
                            }
                        })?);
                    }
                    result
                }
            });
        }
        let resolved_ellipsis = resolved_ellipsis.unwrap_or_default();
        let mut output = Vec::with_capacity(self.output_axes.len());
        for axis in &self.output_axes {
            let logical = by_axis[axis];
            let resolved = match logical.rule {
                EinsumDimensionRule::Equal => {
                    let representative = logical.representative;
                    input_shapes[representative.input][representative.axis].clone()
                }
                EinsumDimensionRule::Broadcast => resolved_ellipsis[match axis {
                    EinsumAxis::Ellipsis(index) => *index,
                    EinsumAxis::Label(_) => unreachable!("broadcast rule belongs to ellipsis"),
                }]
                .clone(),
            };
            output.push(resolved);
        }
        Ok(output)
    }

    /// Validate concrete runtime shapes against this already-parsed plan and
    /// return the exact output shape without reparsing the equation.
    pub fn resolve_concrete_output_shape(
        &self,
        input_shapes: &[&[usize]],
    ) -> Result<Vec<usize>, EinsumPlanError> {
        let dimensions = self.resolve_concrete_axes(input_shapes)?;
        let output: Vec<_> = self
            .output_axes
            .iter()
            .map(|axis| dimensions[axis])
            .collect();
        checked_usize_product(&output).ok_or_else(|| {
            EinsumPlanError::new(
                &self.equation,
                EinsumPlanErrorKind::GeometryOverflow {
                    target: EinsumOverflowTarget::Output,
                },
            )
        })?;
        Ok(output)
    }

    /// Resolve checked concrete B/M/K/N geometry for a GEMM/BMM-classified
    /// plan. Returns `Ok(None)` for every other classification.
    pub fn resolve_concrete_gemm_geometry(
        &self,
        input_shapes: &[&[usize]],
    ) -> Result<Option<EinsumConcreteGemmGeometry>, EinsumPlanError> {
        let EinsumClassification::Gemm(gemm) = &self.classification else {
            return Ok(None);
        };
        let dimensions = self.resolve_concrete_axes(input_shapes)?;
        let output: Vec<_> = self
            .output_axes
            .iter()
            .map(|axis| dimensions[axis])
            .collect();
        checked_usize_product(&output).ok_or_else(|| {
            EinsumPlanError::new(
                &self.equation,
                EinsumPlanErrorKind::GeometryOverflow {
                    target: EinsumOverflowTarget::Output,
                },
            )
        })?;
        let batch_shape: Vec<_> = gemm
            .batch_axes
            .iter()
            .map(|axis| dimensions[axis])
            .collect();
        Ok(Some(EinsumConcreteGemmGeometry {
            batch: concrete_axis_product(
                &self.equation,
                &gemm.batch_axes,
                &dimensions,
                EinsumOverflowTarget::GemmBatch,
            )?,
            m: concrete_axis_product(
                &self.equation,
                &gemm.left_free_axes,
                &dimensions,
                EinsumOverflowTarget::GemmM,
            )?,
            k: concrete_axis_product(
                &self.equation,
                &gemm.contract_axes,
                &dimensions,
                EinsumOverflowTarget::GemmK,
            )?,
            n: concrete_axis_product(
                &self.equation,
                &gemm.right_free_axes,
                &dimensions,
                EinsumOverflowTarget::GemmN,
            )?,
            batch_shape,
        }))
    }

    /// Re-score/replan the semantic contraction tree for concrete runtime
    /// shapes and a caller-supplied intermediate-storage element width.
    ///
    /// Shape-only consumers must pass the width required by their typed
    /// precision policy (for example 4 for f16/bf16 f32 intermediates); this
    /// API never fabricates one.
    pub fn resolve_concrete_contraction_tree(
        &self,
        input_shapes: &[&[usize]],
        element_size: usize,
    ) -> Result<Option<EinsumConcreteContractionTreePlan>, EinsumPlanError> {
        let Some(tree) = self.semantic.contraction_tree() else {
            return Ok(None);
        };
        if element_size == 0 {
            return Err(EinsumPlanError::new(
                &self.equation,
                EinsumPlanErrorKind::InvalidElementSize { element_size },
            ));
        }
        let dimensions = self.resolve_concrete_axes(input_shapes)?;
        Ok(Some(tree.resolve(
            &dimensions,
            input_shapes,
            &self.operands,
            element_size,
        )))
    }

    fn resolve_concrete_axes(
        &self,
        input_shapes: &[&[usize]],
    ) -> Result<BTreeMap<EinsumAxis, usize>, EinsumPlanError> {
        if input_shapes.len() != self.operands.len() {
            return Err(EinsumPlanError::new(
                &self.equation,
                EinsumPlanErrorKind::ResolvedInputCountMismatch {
                    expected: self.operands.len(),
                    found: input_shapes.len(),
                },
            ));
        }
        for (input, (shape, operand)) in input_shapes.iter().zip(&self.operands).enumerate() {
            if shape.len() != operand.rank {
                return Err(EinsumPlanError::new(
                    &self.equation,
                    EinsumPlanErrorKind::ResolvedInputRankMismatch {
                        input,
                        expected: operand.rank,
                        found: shape.len(),
                    },
                ));
            }
            for (axis, (planned, &found)) in operand.shape.iter().zip(*shape).enumerate() {
                if let EinsumDimension::Static(expected) = planned
                    && *expected != found
                {
                    return Err(EinsumPlanError::new(
                        &self.equation,
                        EinsumPlanErrorKind::ResolvedInputDimensionMismatch {
                            input,
                            axis,
                            expected: *expected,
                            found,
                        },
                    ));
                }
            }
            checked_usize_product(shape).ok_or_else(|| {
                EinsumPlanError::new(
                    &self.equation,
                    EinsumPlanErrorKind::GeometryOverflow {
                        target: EinsumOverflowTarget::Input(input),
                    },
                )
            })?;
        }

        let mut dimensions = BTreeMap::new();
        for logical in &self.logical_axes {
            let first = logical.occurrences[0];
            let mut size = input_shapes[first.input][first.axis];
            let mut representative = first;
            for occurrence in logical.occurrences.iter().copied().skip(1) {
                let candidate = input_shapes[occurrence.input][occurrence.axis];
                match logical.rule {
                    EinsumDimensionRule::Equal => {
                        if size != candidate {
                            let EinsumAxis::Label(label) = logical.axis else {
                                unreachable!("equal rule belongs to a named label")
                            };
                            return Err(EinsumPlanError::new(
                                &self.equation,
                                EinsumPlanErrorKind::LabelDimensionMismatch {
                                    label,
                                    first: representative,
                                    first_size: size,
                                    second: occurrence,
                                    second_size: candidate,
                                },
                            ));
                        }
                    }
                    EinsumDimensionRule::Broadcast => {
                        if size == candidate || candidate == 1 {
                            continue;
                        }
                        if size == 1 {
                            size = candidate;
                            representative = occurrence;
                            continue;
                        }
                        let EinsumAxis::Ellipsis(axis) = logical.axis else {
                            unreachable!("broadcast rule belongs to ellipsis")
                        };
                        return Err(EinsumPlanError::new(
                            &self.equation,
                            EinsumPlanErrorKind::EllipsisDimensionMismatch {
                                axis,
                                first: representative,
                                first_size: size,
                                second: occurrence,
                                second_size: candidate,
                            },
                        ));
                    }
                }
            }
            dimensions.insert(logical.axis, size);
        }
        Ok(dimensions)
    }
}

#[derive(Clone, Debug)]
struct ParsedTerm {
    before: Vec<EinsumLabel>,
    has_ellipsis: bool,
    after: Vec<EinsumLabel>,
}

impl ParsedTerm {
    fn named_count(&self) -> Option<usize> {
        self.before.len().checked_add(self.after.len())
    }

    fn labels(&self) -> impl Iterator<Item = EinsumLabel> + '_ {
        self.before.iter().chain(&self.after).copied()
    }

    fn parse(
        equation: &str,
        term: &str,
        side: EinsumEquationSide,
    ) -> Result<Self, EinsumPlanError> {
        let bytes = term.as_bytes();
        let mut before = Vec::new();
        let mut after = Vec::new();
        let mut has_ellipsis = false;
        let mut offset = 0;
        while offset < bytes.len() {
            let byte = bytes[offset];
            if byte.is_ascii_alphabetic() {
                let label = EinsumLabel(byte);
                if has_ellipsis {
                    after.push(label);
                } else {
                    before.push(label);
                }
                offset += 1;
                continue;
            }
            if bytes[offset..].starts_with(b"...") {
                if has_ellipsis {
                    return Err(EinsumPlanError::new(
                        equation,
                        EinsumPlanErrorKind::MultipleEllipses { side },
                    ));
                }
                has_ellipsis = true;
                offset += 3;
                continue;
            }
            let found = term[offset..].chars().next().unwrap_or('\0');
            return Err(EinsumPlanError::new(
                equation,
                EinsumPlanErrorKind::InvalidCharacter {
                    side,
                    offset,
                    found,
                },
            ));
        }
        Ok(Self {
            before,
            has_ellipsis,
            after,
        })
    }
}

#[derive(Clone, Debug)]
struct ParsedEquation {
    inputs: Vec<ParsedTerm>,
    output: Option<ParsedTerm>,
}

impl ParsedEquation {
    fn parse(equation: &str, input_count: usize) -> Result<Self, EinsumPlanError> {
        if input_count == 0 {
            return Err(EinsumPlanError::new(
                equation,
                EinsumPlanErrorKind::NoInputs,
            ));
        }
        let mut arrows = equation.match_indices("->");
        let arrow = arrows.next().map(|(index, _)| index);
        if arrows.next().is_some() {
            return Err(EinsumPlanError::new(
                equation,
                EinsumPlanErrorKind::MultipleOutputArrows,
            ));
        }
        let (left, right) = match arrow {
            Some(index) => (&equation[..index], Some(&equation[index + 2..])),
            None => (equation, None),
        };
        let input_terms: Vec<_> = left.split(',').collect();
        if input_terms.len() != input_count {
            return Err(EinsumPlanError::new(
                equation,
                EinsumPlanErrorKind::InputCount {
                    equation_terms: input_terms.len(),
                    inputs: input_count,
                },
            ));
        }
        let inputs = input_terms
            .into_iter()
            .enumerate()
            .map(|(input, term)| {
                ParsedTerm::parse(equation, term, EinsumEquationSide::Input(input))
            })
            .collect::<Result<_, _>>()?;
        let output = right
            .map(|term| ParsedTerm::parse(equation, term, EinsumEquationSide::Output))
            .transpose()?;
        Ok(Self { inputs, output })
    }
}

#[derive(Clone, Debug)]
struct InputMeta {
    shape: Vec<EinsumDimension>,
}

fn validate_partial_input_shapes<D: EinsumDimensionValue>(
    equation: &str,
    parsed: &ParsedEquation,
    inputs: &[EinsumInput<'_, D>],
    schema: EinsumSchema,
    planner_budget: EinsumPlannerBudget,
) -> Result<(), EinsumPlanError> {
    let mut known_ellipsis_rank = None;
    for (input, (term, metadata)) in parsed.inputs.iter().zip(inputs).enumerate() {
        let Some(shape) = metadata.shape else {
            continue;
        };
        let named_labels = term.named_count().ok_or_else(|| {
            EinsumPlanError::new(
                equation,
                EinsumPlanErrorKind::GeometryOverflow {
                    target: EinsumOverflowTarget::Input(input),
                },
            )
        })?;
        let rank = shape.len();
        let ellipsis_rank = if term.has_ellipsis {
            rank.checked_sub(named_labels)
        } else if rank == named_labels {
            Some(0)
        } else {
            None
        }
        .ok_or_else(|| {
            EinsumPlanError::new(
                equation,
                EinsumPlanErrorKind::InputRankMismatch {
                    input,
                    rank,
                    named_labels,
                    has_ellipsis: term.has_ellipsis,
                },
            )
        })?;
        if term.has_ellipsis {
            if let Some((first_input, first_rank)) = known_ellipsis_rank {
                if ellipsis_rank != first_rank {
                    return Err(EinsumPlanError::new(
                        equation,
                        EinsumPlanErrorKind::EllipsisRankMismatch {
                            first_input,
                            first_rank,
                            input,
                            rank: ellipsis_rank,
                        },
                    ));
                }
            } else {
                known_ellipsis_rank = Some((input, ellipsis_rank));
            }
        }
    }

    let assumed_ellipsis_rank = known_ellipsis_rank.map_or(0, |(_, rank)| rank);
    let input_meta = parsed
        .inputs
        .iter()
        .zip(inputs)
        .enumerate()
        .map(|(input, (term, metadata))| {
            if let Some(shape) = metadata.shape {
                return Ok(InputMeta {
                    shape: shape
                        .iter()
                        .map(|dimension| {
                            dimension
                                .einsum_static_size()
                                .map_or(EinsumDimension::Dynamic, EinsumDimension::Static)
                        })
                        .collect(),
                });
            }
            let named_labels = term.named_count().ok_or_else(|| {
                EinsumPlanError::new(
                    equation,
                    EinsumPlanErrorKind::GeometryOverflow {
                        target: EinsumOverflowTarget::Input(input),
                    },
                )
            })?;
            let rank = named_labels
                .checked_add(if term.has_ellipsis {
                    assumed_ellipsis_rank
                } else {
                    0
                })
                .ok_or_else(|| {
                    EinsumPlanError::new(
                        equation,
                        EinsumPlanErrorKind::GeometryOverflow {
                            target: EinsumOverflowTarget::Input(input),
                        },
                    )
                })?;
            Ok(InputMeta {
                shape: vec![EinsumDimension::Dynamic; rank],
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    build_shape_plan(
        equation.to_owned(),
        parsed.clone(),
        input_meta,
        schema,
        planner_budget,
    )
    .map(drop)
}

fn validate_inputs<D: EinsumDimensionValue>(
    equation: &str,
    inputs: &[EinsumInput<'_, D>],
    schema: EinsumSchema,
) -> Result<(DataType, Vec<InputMeta>), EinsumPlanError> {
    let mut expected_dtype = None;
    let mut missing_dtype = None;
    for (input, metadata) in inputs.iter().enumerate() {
        let Some(dtype) = metadata.dtype else {
            missing_dtype.get_or_insert(input);
            continue;
        };
        if !schema.supports_dtype(dtype) {
            return Err(EinsumPlanError::new(
                equation,
                EinsumPlanErrorKind::UnsupportedInputDtype {
                    input,
                    dtype,
                    schema,
                },
            ));
        }
        if let Some(expected) = expected_dtype {
            if dtype != expected {
                return Err(EinsumPlanError::new(
                    equation,
                    EinsumPlanErrorKind::InputDtypeMismatch {
                        input,
                        expected,
                        actual: dtype,
                    },
                ));
            }
        } else {
            expected_dtype = Some(dtype);
        }
    }
    if let Some(input) = missing_dtype {
        return Err(EinsumPlanError::new(
            equation,
            EinsumPlanErrorKind::MissingInputDtype { input },
        ));
    }

    let mut result = Vec::with_capacity(inputs.len());
    for (input, metadata) in inputs.iter().enumerate() {
        let shape = metadata.shape.ok_or_else(|| {
            EinsumPlanError::new(equation, EinsumPlanErrorKind::MissingInputShape { input })
        })?;
        result.push(InputMeta {
            shape: shape
                .iter()
                .map(|dimension| {
                    dimension
                        .einsum_static_size()
                        .map_or(EinsumDimension::Dynamic, EinsumDimension::Static)
                })
                .collect(),
        });
    }
    let dtype = expected_dtype
        .ok_or_else(|| EinsumPlanError::new(equation, EinsumPlanErrorKind::NoInputs))?;
    Ok((dtype, result))
}

fn normalize_equation(equation: &str) -> String {
    equation
        .chars()
        .filter(|&character| character != ' ')
        .collect()
}

const fn is_base_numeric_dtype(dtype: DataType) -> bool {
    matches!(
        dtype,
        DataType::Uint8
            | DataType::Uint16
            | DataType::Uint32
            | DataType::Uint64
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Float16
            | DataType::Float32
            | DataType::Float64
    )
}

fn build_shape_plan(
    equation: String,
    parsed: ParsedEquation,
    input_meta: Vec<InputMeta>,
    schema: EinsumSchema,
    planner_budget: EinsumPlannerBudget,
) -> Result<EinsumShapePlan, EinsumPlanError> {
    let mut ellipsis_ranks = Vec::with_capacity(input_meta.len());
    let mut explicit_ellipsis = None;
    for (input, (term, metadata)) in parsed.inputs.iter().zip(&input_meta).enumerate() {
        let named_labels = term.named_count().ok_or_else(|| {
            EinsumPlanError::new(
                &equation,
                EinsumPlanErrorKind::GeometryOverflow {
                    target: EinsumOverflowTarget::Input(input),
                },
            )
        })?;
        let rank = metadata.shape.len();
        let ellipsis_rank = if term.has_ellipsis {
            rank.checked_sub(named_labels)
        } else if rank == named_labels {
            Some(0)
        } else {
            None
        }
        .ok_or_else(|| {
            EinsumPlanError::new(
                &equation,
                EinsumPlanErrorKind::InputRankMismatch {
                    input,
                    rank,
                    named_labels,
                    has_ellipsis: term.has_ellipsis,
                },
            )
        })?;
        if term.has_ellipsis {
            if let Some((first_input, first_rank)) = explicit_ellipsis {
                if ellipsis_rank != first_rank {
                    return Err(EinsumPlanError::new(
                        &equation,
                        EinsumPlanErrorKind::EllipsisRankMismatch {
                            first_input,
                            first_rank,
                            input,
                            rank: ellipsis_rank,
                        },
                    ));
                }
            } else {
                explicit_ellipsis = Some((input, ellipsis_rank));
            }
        }
        ellipsis_ranks.push(ellipsis_rank);
    }
    let ellipsis_rank = explicit_ellipsis.map_or(0, |(_, rank)| rank);

    let mut occurrences: BTreeMap<EinsumAxis, Vec<EinsumAxisRef>> = BTreeMap::new();
    let mut operand_axes = Vec::with_capacity(input_meta.len());
    for (input, ((term, metadata), input_ellipsis_rank)) in parsed
        .inputs
        .iter()
        .zip(&input_meta)
        .zip(ellipsis_ranks.iter().copied())
        .enumerate()
    {
        let rank = metadata.shape.len();
        let mut axes = Vec::with_capacity(rank);
        axes.extend(term.before.iter().copied().map(EinsumAxis::Label));
        let ellipsis_start = ellipsis_rank - input_ellipsis_rank;
        axes.extend(
            (ellipsis_start..ellipsis_rank)
                .map(EinsumAxis::Ellipsis)
                .take(input_ellipsis_rank),
        );
        axes.extend(term.after.iter().copied().map(EinsumAxis::Label));
        debug_assert_eq!(axes.len(), rank);

        for (axis, logical) in axes.iter().copied().enumerate() {
            let list = occurrences.entry(logical).or_default();
            if let EinsumAxis::Label(label) = logical
                && list.len().checked_add(1).is_none()
            {
                return Err(EinsumPlanError::new(
                    &equation,
                    EinsumPlanErrorKind::LabelMultiplicityOverflow { label },
                ));
            }
            list.push(EinsumAxisRef { input, axis });
        }

        let static_numel = checked_dimension_product(&metadata.shape).ok_or_else(|| {
            EinsumPlanError::new(
                &equation,
                EinsumPlanErrorKind::GeometryOverflow {
                    target: EinsumOverflowTarget::Input(input),
                },
            )
        })?;
        let mut unique_axes = Vec::<EinsumOperandAxis>::new();
        let mut unique_indices = BTreeMap::<EinsumAxis, usize>::new();
        for (axis, logical) in axes.iter().copied().enumerate() {
            if let Some(index) = unique_indices.get(&logical).copied() {
                unique_axes[index].input_axes.push(axis);
                unique_axes[index].dimension =
                    merge_equal_dimension(unique_axes[index].dimension, metadata.shape[axis]);
            } else {
                unique_indices.insert(logical, unique_axes.len());
                unique_axes.push(EinsumOperandAxis {
                    axis: logical,
                    input_axes: vec![axis],
                    dimension: metadata.shape[axis],
                });
            }
        }
        let diagonal_axis_indices = unique_axes
            .iter()
            .enumerate()
            .filter_map(|(index, axis)| (axis.input_axes.len() > 1).then_some(index))
            .collect();
        operand_axes.push(EinsumOperandPlan {
            input,
            rank,
            has_ellipsis: term.has_ellipsis,
            ellipsis_rank: input_ellipsis_rank,
            shape: metadata.shape.clone(),
            axes,
            unique_axes,
            diagonal_axis_indices,
            static_numel,
        });
    }

    let mut label_counts = BTreeMap::<EinsumLabel, usize>::new();
    for term in &parsed.inputs {
        for label in term.labels() {
            let count = label_counts.entry(label).or_default();
            *count = count.checked_add(1).ok_or_else(|| {
                EinsumPlanError::new(
                    &equation,
                    EinsumPlanErrorKind::LabelMultiplicityOverflow { label },
                )
            })?;
        }
    }

    let explicit_output = parsed.output.is_some();
    let output_axes = if let Some(output) = &parsed.output {
        let mut seen = BTreeSet::new();
        let mut axes = Vec::new();
        for label in &output.before {
            validate_output_label(&equation, *label, &occurrences, &mut seen)?;
            axes.push(EinsumAxis::Label(*label));
        }
        if output.has_ellipsis {
            axes.extend((0..ellipsis_rank).map(EinsumAxis::Ellipsis));
        }
        for label in &output.after {
            validate_output_label(&equation, *label, &occurrences, &mut seen)?;
            axes.push(EinsumAxis::Label(*label));
        }
        axes
    } else {
        let mut axes: Vec<_> = (0..ellipsis_rank).map(EinsumAxis::Ellipsis).collect();
        axes.extend(
            label_counts
                .iter()
                .filter_map(|(&label, &count)| (count == 1).then_some(EinsumAxis::Label(label))),
        );
        axes
    };
    let output_positions: BTreeMap<_, _> = output_axes
        .iter()
        .copied()
        .enumerate()
        .map(|(position, axis)| (axis, position))
        .collect();

    let mut logical_axes = Vec::with_capacity(occurrences.len());
    for (axis, axis_occurrences) in &occurrences {
        let (dimension, representative, requires_runtime_check, rule) = match axis {
            EinsumAxis::Label(label) => {
                let summary =
                    summarize_equal_axis(&equation, *label, axis_occurrences, &input_meta)?;
                (
                    summary.dimension,
                    summary.representative,
                    summary.requires_runtime_check,
                    EinsumDimensionRule::Equal,
                )
            }
            EinsumAxis::Ellipsis(index) => {
                let summary =
                    summarize_broadcast_axis(&equation, *index, axis_occurrences, &input_meta)?;
                (
                    summary.dimension,
                    summary.representative,
                    summary.requires_runtime_check,
                    EinsumDimensionRule::Broadcast,
                )
            }
        };
        logical_axes.push(EinsumLogicalAxis {
            axis: *axis,
            occurrences: axis_occurrences.clone(),
            dimension,
            rule,
            requires_runtime_check,
            representative,
            output_position: output_positions.get(axis).copied(),
        });
    }
    let logical_by_axis: BTreeMap<_, _> = logical_axes
        .iter()
        .map(|logical| (logical.axis, logical))
        .collect();
    let output_shape: Vec<_> = output_axes
        .iter()
        .map(|axis| logical_by_axis[axis].dimension)
        .collect();
    let static_output_numel = checked_dimension_product(&output_shape).ok_or_else(|| {
        EinsumPlanError::new(
            &equation,
            EinsumPlanErrorKind::GeometryOverflow {
                target: EinsumOverflowTarget::Output,
            },
        )
    })?;
    let reduction_axes: Vec<_> = logical_axes
        .iter()
        .filter_map(|logical| logical.output_position.is_none().then_some(logical.axis))
        .collect();
    let contraction_tree = (operand_axes.len() > 1).then(|| {
        tree::build_contraction_tree(
            &operand_axes,
            &logical_axes,
            &output_axes,
            &reduction_axes,
            planner_budget,
        )
    });
    let classification = classify(
        &equation,
        &operand_axes,
        &logical_axes,
        &output_axes,
        &reduction_axes,
        contraction_tree.as_ref(),
    )?;
    let mut iteration_axes = output_axes.clone();
    iteration_axes.extend_from_slice(&reduction_axes);
    let iteration_positions: BTreeMap<_, _> = iteration_axes
        .iter()
        .copied()
        .enumerate()
        .map(|(index, axis)| (axis, index))
        .collect();
    let generic_native = EinsumGenericNativePlan {
        index_program: EinsumIndexProgram {
            iteration_axes,
            output_rank: output_axes.len(),
            operands: operand_axes
                .iter()
                .map(|operand| EinsumOperandIndexProgram {
                    input: operand.input,
                    physical_axis_to_iteration_axis: operand
                        .axes
                        .iter()
                        .map(|axis| iteration_positions[axis])
                        .collect(),
                    physical_axis_broadcasts_when_one: operand
                        .axes
                        .iter()
                        .map(|axis| matches!(axis, EinsumAxis::Ellipsis(_)))
                        .collect(),
                })
                .collect(),
        },
    };

    Ok(EinsumShapePlan {
        schema,
        equation,
        explicit_output,
        operands: operand_axes,
        logical_axes,
        output_axes,
        output_shape,
        reduction_axes,
        classification,
        semantic: EinsumSemanticPlan {
            generic_native,
            contraction_tree,
        },
        static_output_numel,
    })
}

fn validate_output_label(
    equation: &str,
    label: EinsumLabel,
    occurrences: &BTreeMap<EinsumAxis, Vec<EinsumAxisRef>>,
    seen: &mut BTreeSet<EinsumLabel>,
) -> Result<(), EinsumPlanError> {
    if !seen.insert(label) {
        return Err(EinsumPlanError::new(
            equation,
            EinsumPlanErrorKind::DuplicateOutputLabel { label },
        ));
    }
    if !occurrences.contains_key(&EinsumAxis::Label(label)) {
        return Err(EinsumPlanError::new(
            equation,
            EinsumPlanErrorKind::OutputLabelMissingFromInputs { label },
        ));
    }
    Ok(())
}

struct AxisSummary {
    dimension: EinsumDimension,
    representative: EinsumAxisRef,
    requires_runtime_check: bool,
}

fn summarize_equal_axis(
    equation: &str,
    label: EinsumLabel,
    occurrences: &[EinsumAxisRef],
    inputs: &[InputMeta],
) -> Result<AxisSummary, EinsumPlanError> {
    let mut representative = occurrences[0];
    let mut dimension = input_dimension(inputs, representative);
    let mut first_static = dimension.as_static().map(|size| (representative, size));
    let mut has_dynamic = matches!(dimension, EinsumDimension::Dynamic);
    for occurrence in occurrences.iter().copied().skip(1) {
        let candidate = input_dimension(inputs, occurrence);
        has_dynamic |= matches!(candidate, EinsumDimension::Dynamic);
        if let Some(size) = candidate.as_static() {
            if let Some((first, first_size)) = first_static {
                if size != first_size {
                    return Err(EinsumPlanError::new(
                        equation,
                        EinsumPlanErrorKind::LabelDimensionMismatch {
                            label,
                            first,
                            first_size,
                            second: occurrence,
                            second_size: size,
                        },
                    ));
                }
            } else {
                first_static = Some((occurrence, size));
            }
        }
        if dimension.as_static().is_none() {
            representative = occurrence;
            dimension = candidate;
        }
    }
    if let Some((static_ref, size)) = first_static {
        representative = static_ref;
        dimension = EinsumDimension::Static(size);
    }
    Ok(AxisSummary {
        dimension,
        representative,
        requires_runtime_check: occurrences.len() > 1 && has_dynamic,
    })
}

fn summarize_broadcast_axis(
    equation: &str,
    axis: usize,
    occurrences: &[EinsumAxisRef],
    inputs: &[InputMeta],
) -> Result<AxisSummary, EinsumPlanError> {
    let mut representative = occurrences[0];
    let mut dimension = input_dimension(inputs, representative);
    let mut requires_runtime_check = false;
    for occurrence in occurrences.iter().copied().skip(1) {
        let candidate = input_dimension(inputs, occurrence);
        match (dimension, candidate) {
            (EinsumDimension::Static(left), EinsumDimension::Static(right)) => {
                if left == right || right == 1 {
                    continue;
                }
                if left == 1 {
                    representative = occurrence;
                    dimension = candidate;
                    continue;
                }
                return Err(EinsumPlanError::new(
                    equation,
                    EinsumPlanErrorKind::EllipsisDimensionMismatch {
                        axis,
                        first: representative,
                        first_size: left,
                        second: occurrence,
                        second_size: right,
                    },
                ));
            }
            (EinsumDimension::Static(1), EinsumDimension::Dynamic) => {
                representative = occurrence;
                dimension = EinsumDimension::Dynamic;
            }
            (EinsumDimension::Dynamic, EinsumDimension::Static(1)) => {}
            (EinsumDimension::Static(_), EinsumDimension::Dynamic) => {
                requires_runtime_check = true;
            }
            (EinsumDimension::Dynamic, EinsumDimension::Static(right)) => {
                if right != 1 {
                    representative = occurrence;
                    dimension = candidate;
                    requires_runtime_check = true;
                }
            }
            (EinsumDimension::Dynamic, EinsumDimension::Dynamic) => {
                requires_runtime_check = true;
            }
        }
    }
    Ok(AxisSummary {
        dimension,
        representative,
        requires_runtime_check,
    })
}

fn input_dimension(inputs: &[InputMeta], reference: EinsumAxisRef) -> EinsumDimension {
    inputs[reference.input].shape[reference.axis]
}

fn merge_equal_dimension(current: EinsumDimension, candidate: EinsumDimension) -> EinsumDimension {
    match (current, candidate) {
        (static_dimension @ EinsumDimension::Static(_), _) => static_dimension,
        (_, static_dimension @ EinsumDimension::Static(_)) => static_dimension,
        _ => candidate,
    }
}

fn checked_dimension_product(dimensions: &[EinsumDimension]) -> Option<Option<usize>> {
    if dimensions.contains(&EinsumDimension::Static(0)) {
        return Some(Some(0));
    }
    let mut product = 1usize;
    for dimension in dimensions {
        let EinsumDimension::Static(value) = dimension else {
            return Some(None);
        };
        product = product.checked_mul(*value)?;
    }
    Some(Some(product))
}

fn checked_usize_product(dimensions: &[usize]) -> Option<usize> {
    if dimensions.contains(&0) {
        return Some(0);
    }
    dimensions
        .iter()
        .try_fold(1usize, |product, dimension| product.checked_mul(*dimension))
}

fn concrete_axis_product(
    equation: &str,
    axes: &[EinsumAxis],
    dimensions: &BTreeMap<EinsumAxis, usize>,
    target: EinsumOverflowTarget,
) -> Result<usize, EinsumPlanError> {
    let values: Vec<_> = axes.iter().map(|axis| dimensions[axis]).collect();
    checked_usize_product(&values).ok_or_else(|| {
        EinsumPlanError::new(equation, EinsumPlanErrorKind::GeometryOverflow { target })
    })
}

fn product_for_axes(
    equation: &str,
    axes: &[EinsumAxis],
    logical: &BTreeMap<EinsumAxis, &EinsumLogicalAxis>,
    target: EinsumOverflowTarget,
) -> Result<EinsumDimension, EinsumPlanError> {
    let dimensions: Vec<_> = axes.iter().map(|axis| logical[axis].dimension).collect();
    match checked_dimension_product(&dimensions) {
        Some(Some(value)) => Ok(EinsumDimension::Static(value)),
        Some(None) => Ok(EinsumDimension::Dynamic),
        None => Err(EinsumPlanError::new(
            equation,
            EinsumPlanErrorKind::GeometryOverflow { target },
        )),
    }
}

fn classify(
    equation: &str,
    operands: &[EinsumOperandPlan],
    logical_axes: &[EinsumLogicalAxis],
    output_axes: &[EinsumAxis],
    reduction_axes: &[EinsumAxis],
    contraction_tree: Option<&EinsumContractionTreePlan>,
) -> Result<EinsumClassification, EinsumPlanError> {
    if operands.len() == 1 && reduction_axes.is_empty() {
        let permutation = permutation_plan(&operands[0], output_axes);
        return Ok(if operands[0].diagonal_axis_indices.is_empty() {
            EinsumClassification::ViewOnlyPermutation(permutation)
        } else {
            EinsumClassification::DiagonalView(permutation)
        });
    }

    let distinct_input_count = |axis: EinsumAxis| {
        logical_axes
            .iter()
            .find(|logical| logical.axis == axis)
            .map(|logical| {
                logical
                    .occurrences
                    .iter()
                    .map(|reference| reference.input)
                    .collect::<BTreeSet<_>>()
                    .len()
            })
            .unwrap_or(0)
    };
    let cross_reduction_axes: Vec<_> = reduction_axes
        .iter()
        .copied()
        .filter(|axis| distinct_input_count(*axis) > 1)
        .collect();

    if cross_reduction_axes.is_empty() {
        let mut iteration_axes = output_axes.to_vec();
        iteration_axes.extend_from_slice(reduction_axes);
        let operand_axis_mappings = operands
            .iter()
            .map(|operand| {
                operand
                    .unique_axes
                    .iter()
                    .map(|axis| {
                        iteration_axes
                            .iter()
                            .position(|candidate| *candidate == axis.axis)
                            .expect("every logical operand axis is retained or reduced")
                    })
                    .collect()
            })
            .collect();
        return Ok(EinsumClassification::ReductionOrElementwise(
            EinsumReductionPlan {
                iteration_axes,
                output_rank: output_axes.len(),
                operand_axis_mappings,
            },
        ));
    }

    if operands.len() == 2 {
        let local_reduction_axes: Vec<_> = reduction_axes
            .iter()
            .copied()
            .filter(|axis| distinct_input_count(*axis) == 1)
            .collect();
        let reduced_ellipsis = cross_reduction_axes
            .iter()
            .any(|axis| matches!(axis, EinsumAxis::Ellipsis(_)));
        if !local_reduction_axes.is_empty() || reduced_ellipsis {
            return Ok(EinsumClassification::ContractionTree(
                contraction_tree
                    .expect("multi-operand semantic plans always include a tree")
                    .clone(),
            ));
        }
        return build_contraction(
            equation,
            operands,
            logical_axes,
            output_axes,
            cross_reduction_axes,
        )
        .map(EinsumClassification::Gemm);
    }

    Ok(EinsumClassification::ContractionTree(
        contraction_tree
            .expect("multi-operand semantic plans always include a tree")
            .clone(),
    ))
}

fn permutation_plan(
    operand: &EinsumOperandPlan,
    output_axes: &[EinsumAxis],
) -> EinsumPermutationPlan {
    EinsumPermutationPlan {
        input: operand.input,
        output_to_operand_axis: output_axes
            .iter()
            .map(|axis| {
                operand
                    .unique_axes
                    .iter()
                    .position(|candidate| candidate.axis == *axis)
                    .expect("view output retains every unique operand axis")
            })
            .collect(),
    }
}

fn build_contraction(
    equation: &str,
    operands: &[EinsumOperandPlan],
    logical_axes: &[EinsumLogicalAxis],
    output_axes: &[EinsumAxis],
    contract_axes: Vec<EinsumAxis>,
) -> Result<EinsumContractionPlan, EinsumPlanError> {
    let logical: BTreeMap<_, _> = logical_axes.iter().map(|axis| (axis.axis, axis)).collect();
    let present_in = |axis: EinsumAxis, input: usize| {
        logical[&axis]
            .occurrences
            .iter()
            .any(|reference| reference.input == input)
    };
    let batch_axes: Vec<_> = output_axes
        .iter()
        .copied()
        .filter(|axis| {
            matches!(axis, EinsumAxis::Ellipsis(_))
                || (present_in(*axis, 0) && present_in(*axis, 1))
        })
        .collect();
    let left_free_axes: Vec<_> = output_axes
        .iter()
        .copied()
        .filter(|axis| !batch_axes.contains(axis) && present_in(*axis, 0) && !present_in(*axis, 1))
        .collect();
    let right_free_axes: Vec<_> = output_axes
        .iter()
        .copied()
        .filter(|axis| !batch_axes.contains(axis) && present_in(*axis, 1) && !present_in(*axis, 0))
        .collect();

    let operand_axis_index = |input: usize, axis: EinsumAxis| {
        operands[input]
            .unique_axes
            .iter()
            .position(|candidate| candidate.axis == axis)
    };
    let left_axis_order = batch_axes
        .iter()
        .chain(&left_free_axes)
        .chain(&contract_axes)
        .map(|axis| operand_axis_index(0, *axis))
        .collect();
    let right_axis_order = batch_axes
        .iter()
        .chain(&contract_axes)
        .chain(&right_free_axes)
        .map(|axis| operand_axis_index(1, *axis))
        .collect();

    let mut canonical_output = batch_axes.clone();
    canonical_output.extend_from_slice(&left_free_axes);
    canonical_output.extend_from_slice(&right_free_axes);
    let output_permutation = output_axes
        .iter()
        .map(|axis| {
            canonical_output
                .iter()
                .position(|candidate| candidate == axis)
                .expect("GEMM output axis belongs to batch, M, or N")
        })
        .collect();

    let batch_shape = batch_axes
        .iter()
        .map(|axis| logical[axis].dimension)
        .collect::<Vec<_>>();
    let geometry = EinsumGemmGeometry {
        batch: product_for_axes(
            equation,
            &batch_axes,
            &logical,
            EinsumOverflowTarget::GemmBatch,
        )?,
        m: product_for_axes(
            equation,
            &left_free_axes,
            &logical,
            EinsumOverflowTarget::GemmM,
        )?,
        k: product_for_axes(
            equation,
            &contract_axes,
            &logical,
            EinsumOverflowTarget::GemmK,
        )?,
        n: product_for_axes(
            equation,
            &right_free_axes,
            &logical,
            EinsumOverflowTarget::GemmN,
        )?,
        batch_shape,
    };

    Ok(EinsumContractionPlan {
        batch_axes,
        left_free_axes,
        contract_axes,
        right_free_axes,
        left_axis_order,
        right_axis_order,
        output_permutation,
        geometry,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SymbolId;

    fn plan(equation: &str, shapes: &[&[usize]]) -> EinsumPlan {
        let inputs: Vec<_> = shapes
            .iter()
            .map(|shape| EinsumInput::new(DataType::Float32, shape))
            .collect();
        EinsumPlan::build(equation, &inputs).unwrap()
    }

    fn axis_label(axis: EinsumAxis) -> char {
        match axis {
            EinsumAxis::Label(label) => label.as_char(),
            EinsumAxis::Ellipsis(_) => '.',
        }
    }

    fn output_labels(plan: &EinsumPlan) -> String {
        plan.output_axes().iter().copied().map(axis_label).collect()
    }

    fn static_shape(plan: &EinsumPlan) -> Vec<Option<usize>> {
        plan.output_shape()
            .iter()
            .map(|dimension| dimension.as_static())
            .collect()
    }

    fn error_kind<D: EinsumDimensionValue>(
        equation: &str,
        inputs: &[EinsumInput<'_, D>],
    ) -> EinsumPlanErrorKind {
        EinsumPlan::build(equation, inputs).unwrap_err().kind
    }

    fn assert_ellipsis_rank_mismatch(
        equation: &str,
        left: &[usize],
        right: &[usize],
        first_rank: usize,
        rank: usize,
    ) {
        let inputs = [
            EinsumInput::new(DataType::Float32, left),
            EinsumInput::new(DataType::Float32, right),
        ];
        let error = EinsumPlan::build(equation, &inputs).unwrap_err();
        assert!(matches!(
            error.kind(),
            EinsumPlanErrorKind::EllipsisRankMismatch {
                first_input: 0,
                first_rank: actual_first_rank,
                input: 1,
                rank: actual_rank,
            } if *actual_first_rank == first_rank && *actual_rank == rank
        ));
        assert!(
            error
                .to_string()
                .contains("ONNX Einsum requires every explicit ellipsis")
        );
    }

    #[test]
    fn explicit_and_implicit_equations_normalize_and_order_output() {
        let explicit = plan("  i k , k j  ->  j i ", &[&[2, 3], &[3, 4]]);
        assert_eq!(explicit.equation(), "ik,kj->ji");
        assert!(explicit.has_explicit_output());
        assert_eq!(output_labels(&explicit), "ji");
        assert_eq!(static_shape(&explicit), vec![Some(4), Some(2)]);

        let implicit = plan("za,ay", &[&[2, 3], &[3, 5]]);
        assert!(!implicit.has_explicit_output());
        assert_eq!(output_labels(&implicit), "yz");
        assert_eq!(static_shape(&implicit), vec![Some(5), Some(2)]);
    }

    #[test]
    fn mixed_case_labels_are_distinct_and_implicitly_sorted_by_ascii() {
        let implicit = plan("Za,aB", &[&[2, 3], &[3, 4]]);
        assert_eq!(output_labels(&implicit), "BZ");
        assert_eq!(static_shape(&implicit), vec![Some(4), Some(2)]);

        let explicit = plan("aA->Aa", &[&[2, 3]]);
        assert_eq!(output_labels(&explicit), "Aa");
        assert_eq!(static_shape(&explicit), vec![Some(3), Some(2)]);
        assert_eq!(
            explicit
                .logical_axes()
                .iter()
                .filter(|axis| matches!(axis.axis(), EinsumAxis::Label(_)))
                .count(),
            2,
            "upper- and lower-case labels are different logical axes"
        );
    }

    #[test]
    fn only_ascii_space_is_ignored() {
        let spaced = plan(" i k , k j -> j i ", &[&[2, 3], &[3, 4]]);
        assert_eq!(spaced.equation(), "ik,kj->ji");

        for invalid in ['\t', '\n', '\u{00a0}', '\u{2003}', '\u{2028}'] {
            let equation = format!("i{invalid}->i");
            let shape = [7usize];
            let inputs = [EinsumInput::new(DataType::Float32, &shape)];
            let error = EinsumPlan::build(&equation, &inputs).unwrap_err();
            assert_eq!(error.equation(), equation);
            assert!(matches!(
                error.kind(),
                EinsumPlanErrorKind::InvalidCharacter {
                    side: EinsumEquationSide::Input(0),
                    offset: 1,
                    found,
                } if *found == invalid
            ));
        }
    }

    #[test]
    fn scalar_rank_one_and_zero_dimensions_are_preserved() {
        for equation in ["", "->", "   ->   "] {
            let scalar = plan(equation, &[&[]]);
            assert!(scalar.output_axes().is_empty());
            assert_eq!(scalar.static_output_numel(), Some(1));
            assert!(matches!(
                scalar.classification(),
                EinsumClassification::ViewOnlyPermutation(view) if view.is_identity()
            ));
        }

        let rank_one = plan("i->i", &[&[7]]);
        assert_eq!(static_shape(&rank_one), vec![Some(7)]);
        let zero = plan("ij->ji", &[&[0, usize::MAX]]);
        assert_eq!(static_shape(&zero), vec![Some(usize::MAX), Some(0)]);
        assert_eq!(zero.static_output_numel(), Some(0));
    }

    #[test]
    fn explicit_ellipses_expand_to_one_fixed_rank() {
        let plan = plan("...ij,j...k->...ik", &[&[6, 5, 2, 3], &[3, 6, 5, 4]]);
        assert_eq!(
            static_shape(&plan),
            vec![Some(6), Some(5), Some(2), Some(4)]
        );
        assert_eq!(
            plan.output_axes(),
            &[
                EinsumAxis::Ellipsis(0),
                EinsumAxis::Ellipsis(1),
                EinsumAxis::Label(EinsumLabel(b'i')),
                EinsumAxis::Label(EinsumLabel(b'k')),
            ]
        );
        assert_eq!(
            plan.operands()[0].axes(),
            &[
                EinsumAxis::Ellipsis(0),
                EinsumAxis::Ellipsis(1),
                EinsumAxis::Label(EinsumLabel(b'i')),
                EinsumAxis::Label(EinsumLabel(b'j')),
            ]
        );
        assert_eq!(
            plan.operands()[1].axes(),
            &[
                EinsumAxis::Label(EinsumLabel(b'j')),
                EinsumAxis::Ellipsis(0),
                EinsumAxis::Ellipsis(1),
                EinsumAxis::Label(EinsumLabel(b'k')),
            ]
        );
    }

    #[test]
    fn unequal_explicit_ellipsis_ranks_are_rejected() {
        assert_ellipsis_rank_mismatch("...ij,j...k->...ik", &[5, 2, 3], &[3, 6, 5, 4], 1, 2);
        assert_ellipsis_rank_mismatch("...ij,...jk->...ik", &[2, 3], &[6, 5, 3, 4], 0, 2);
    }

    #[test]
    fn terms_without_ellipsis_do_not_acquire_broadcast_axes() {
        let explicit = plan("ij,...jk->...ik", &[&[2, 3], &[6, 5, 3, 4]]);
        assert_eq!(
            static_shape(&explicit),
            vec![Some(6), Some(5), Some(2), Some(4)]
        );
        assert!(!explicit.operands()[0].has_ellipsis());
        assert_eq!(explicit.operands()[0].ellipsis_rank(), 0);
        assert_eq!(explicit.operands()[0].axes().len(), 2);

        let implicit = plan("ij,...jk", &[&[2, 3], &[6, 5, 3, 4]]);
        assert_eq!(
            static_shape(&implicit),
            vec![Some(6), Some(5), Some(2), Some(4)]
        );

        let zero_rank = plan("...ij,...jk->...ik", &[&[2, 3], &[3, 4]]);
        assert_eq!(static_shape(&zero_rank), vec![Some(2), Some(4)]);
        assert!(
            zero_rank
                .operands()
                .iter()
                .all(EinsumOperandPlan::has_ellipsis)
        );
        assert!(
            zero_rank
                .operands()
                .iter()
                .all(|operand| operand.ellipsis_rank() == 0)
        );
        assert!(
            zero_rank
                .output_axes()
                .iter()
                .all(|axis| !matches!(axis, EinsumAxis::Ellipsis(_)))
        );
    }

    #[test]
    fn symbolic_ellipsis_retains_runtime_broadcast_constraints() {
        let left = [Dim::Symbolic(SymbolId(1)), Dim::Static(2), Dim::Static(3)];
        let right = [Dim::Static(6), Dim::Static(3), Dim::Static(4)];
        let inputs = [
            EinsumInput::new(DataType::Float32, &left),
            EinsumInput::new(DataType::Float32, &right),
        ];
        let plan = EinsumPlan::build("...ij,...jk->...ik", &inputs).unwrap();
        assert_eq!(static_shape(&plan), vec![Some(6), Some(2), Some(4)]);
        let batch = plan
            .logical_axes()
            .iter()
            .find(|logical| logical.axis() == EinsumAxis::Ellipsis(0))
            .unwrap();
        assert_eq!(batch.rule(), EinsumDimensionRule::Broadcast);
        assert!(batch.requires_runtime_check());
        assert_eq!(batch.occurrences().len(), 2);
    }

    #[test]
    fn repeated_labels_form_diagonal_views() {
        let diagonal = plan("...iii->...i", &[&[2, 5, 5, 5]]);
        let EinsumClassification::DiagonalView(view) = diagonal.classification() else {
            panic!("expected diagonal view");
        };
        assert!(view.is_identity());
        let operand = &diagonal.operands()[0];
        assert_eq!(operand.diagonal_axis_indices(), &[1]);
        assert_eq!(operand.unique_axes()[1].input_axes(), &[1, 2, 3]);
        assert_eq!(static_shape(&diagonal), vec![Some(2), Some(5)]);

        let reduced = plan("ii", &[&[4, 4]]);
        assert!(matches!(
            reduced.classification(),
            EinsumClassification::ReductionOrElementwise(_)
        ));
        assert_eq!(reduced.reduction_axes().len(), 1);
    }

    #[test]
    fn every_rank_three_output_permutation_is_a_view() {
        let cases = [
            ("abc->abc", [0, 1, 2]),
            ("abc->acb", [0, 2, 1]),
            ("abc->bac", [1, 0, 2]),
            ("abc->bca", [1, 2, 0]),
            ("abc->cab", [2, 0, 1]),
            ("abc->cba", [2, 1, 0]),
        ];
        for (equation, expected) in cases {
            let plan = plan(equation, &[&[2, 3, 4]]);
            let EinsumClassification::ViewOnlyPermutation(view) = plan.classification() else {
                panic!("{equation} was not a view");
            };
            assert_eq!(view.output_to_operand_axis(), expected);
        }
    }

    #[test]
    fn elementwise_outer_product_and_nary_local_reduction_share_one_class() {
        for (equation, shapes, output) in [
            ("ij,ij->ij", vec![&[2, 3][..], &[2, 3][..]], "ij"),
            ("i,j->ij", vec![&[2][..], &[3][..]], "ij"),
            ("ij,i,i->i", vec![&[2, 3][..], &[2][..], &[2][..]], "i"),
        ] {
            let plan = plan(equation, &shapes);
            let EinsumClassification::ReductionOrElementwise(reduction) = plan.classification()
            else {
                panic!("{equation} was not reduction/elementwise");
            };
            assert_eq!(output_labels(&plan), output);
            assert_eq!(reduction.output_rank(), output.len());
            assert_eq!(reduction.operand_axis_mappings().len(), shapes.len());
        }
    }

    #[test]
    fn matrix_contraction_carries_complete_gemm_layout() {
        let plan = plan("ik,kj->ij", &[&[2, 3], &[3, 4]]);
        let EinsumClassification::Gemm(gemm) = plan.classification() else {
            panic!("expected GEMM");
        };
        assert!(gemm.batch_axes().is_empty());
        assert_eq!(
            gemm.left_free_axes()
                .iter()
                .map(|axis| axis_label(*axis))
                .collect::<String>(),
            "i"
        );
        assert_eq!(
            gemm.contract_axes()
                .iter()
                .map(|axis| axis_label(*axis))
                .collect::<String>(),
            "k"
        );
        assert_eq!(
            gemm.right_free_axes()
                .iter()
                .map(|axis| axis_label(*axis))
                .collect::<String>(),
            "j"
        );
        assert_eq!(gemm.left_axis_order(), &[Some(0), Some(1)]);
        assert_eq!(gemm.right_axis_order(), &[Some(0), Some(1)]);
        assert_eq!(gemm.output_permutation(), &[0, 1]);
        assert_eq!(gemm.geometry().batch(), EinsumDimension::Static(1));
        assert_eq!(gemm.geometry().m(), EinsumDimension::Static(2));
        assert_eq!(gemm.geometry().k(), EinsumDimension::Static(3));
        assert_eq!(gemm.geometry().n(), EinsumDimension::Static(4));
    }

    #[test]
    fn bmm_layout_inserts_missing_batch_axes_and_records_output_permutation() {
        let plan = plan("mk,...kn->n...m", &[&[2, 3], &[6, 5, 3, 4]]);
        let EinsumClassification::Gemm(gemm) = plan.classification() else {
            panic!("expected BMM");
        };
        assert_eq!(gemm.left_axis_order(), &[None, None, Some(0), Some(1)]);
        assert_eq!(
            gemm.right_axis_order(),
            &[Some(0), Some(1), Some(2), Some(3)]
        );
        assert_eq!(gemm.output_permutation(), &[3, 0, 1, 2]);
        assert_eq!(
            gemm.geometry().batch_shape(),
            &[EinsumDimension::Static(6), EinsumDimension::Static(5)]
        );
        assert_eq!(gemm.geometry().batch(), EinsumDimension::Static(30));
    }

    #[test]
    fn vector_and_multi_axis_contractions_flatten_without_shape_dispatch() {
        let dot = plan("i,i->", &[&[8], &[8]]);
        let EinsumClassification::Gemm(dot) = dot.classification() else {
            panic!("expected dot-product GEMM");
        };
        assert_eq!(dot.geometry().m(), EinsumDimension::Static(1));
        assert_eq!(dot.geometry().k(), EinsumDimension::Static(8));
        assert_eq!(dot.geometry().n(), EinsumDimension::Static(1));

        let multi = plan("abxy,xycd->dcab", &[&[2, 3, 5, 7], &[5, 7, 11, 13]]);
        let EinsumClassification::Gemm(multi) = multi.classification() else {
            panic!("expected flattened GEMM");
        };
        assert_eq!(multi.geometry().m(), EinsumDimension::Static(6));
        assert_eq!(multi.geometry().k(), EinsumDimension::Static(35));
        assert_eq!(multi.geometry().n(), EinsumDimension::Static(143));
        assert_eq!(multi.output_permutation(), &[2, 3, 0, 1]);
    }

    #[test]
    fn general_contractions_and_reduced_ellipsis_have_bounded_trees() {
        let nary = plan("ij,jk,kl->il", &[&[2, 3], &[3, 4], &[4, 5]]);
        assert!(matches!(
            nary.classification(),
            EinsumClassification::ContractionTree(tree)
                if tree.arity() == 3 && tree.candidates().len() == 12
        ));

        let mixed = plan("aik,kj->ij", &[&[7, 2, 3], &[3, 4]]);
        assert!(matches!(
            mixed.classification(),
            EinsumClassification::ContractionTree(tree)
                if tree.arity() == 2 && tree.candidates().len() == 2
        ));

        let ellipsis = plan("...i,...i->i", &[&[5, 3], &[1, 3]]);
        assert!(matches!(
            ellipsis.classification(),
            EinsumClassification::ContractionTree(tree)
                if tree.arity() == 2 && tree.preferred_candidate().is_some()
        ));
        assert!(ellipsis.generic_native().index_program().output_rank() == 1);
    }

    #[test]
    fn parser_rejects_bad_counts_characters_arrows_and_ellipses() {
        let shape = [2usize];
        let one = [EinsumInput::new(DataType::Float32, &shape)];
        let none: [EinsumInput<'_, usize>; 0] = [];
        assert!(matches!(
            error_kind("i", &none),
            EinsumPlanErrorKind::NoInputs
        ));
        assert!(matches!(
            error_kind("i,j", &one),
            EinsumPlanErrorKind::InputCount {
                equation_terms: 2,
                inputs: 1
            }
        ));
        assert!(matches!(
            error_kind("i->i->i", &one),
            EinsumPlanErrorKind::MultipleOutputArrows
        ));
        assert!(matches!(
            error_kind("i$", &one),
            EinsumPlanErrorKind::InvalidCharacter {
                side: EinsumEquationSide::Input(0),
                found: '$',
                ..
            }
        ));
        assert!(matches!(
            error_kind("......i", &one),
            EinsumPlanErrorKind::MultipleEllipses {
                side: EinsumEquationSide::Input(0)
            }
        ));
    }

    #[test]
    fn output_and_rank_legality_is_validated() {
        let matrix = [2usize, 3];
        let input = [EinsumInput::new(DataType::Float32, &matrix)];
        assert!(matches!(
            error_kind("i->i", &input),
            EinsumPlanErrorKind::InputRankMismatch {
                input: 0,
                rank: 2,
                named_labels: 1,
                has_ellipsis: false
            }
        ));
        assert!(matches!(
            error_kind("ij->ii", &input),
            EinsumPlanErrorKind::DuplicateOutputLabel { label }
                if label.as_char() == 'i'
        ));
        assert!(matches!(
            error_kind("ij->ik", &input),
            EinsumPlanErrorKind::OutputLabelMissingFromInputs { label }
                if label.as_char() == 'k'
        ));
        assert!(matches!(
            error_kind("ijk", &input),
            EinsumPlanErrorKind::InputRankMismatch {
                has_ellipsis: false,
                ..
            }
        ));
    }

    #[test]
    fn label_and_ellipsis_dimension_conflicts_name_inputs_axes_and_labels() {
        let left = [2usize, 3];
        let wrong_label = [4usize, 5];
        let inputs = [
            EinsumInput::new(DataType::Float32, &left),
            EinsumInput::new(DataType::Float32, &wrong_label),
        ];
        let error = EinsumPlan::build("ij,ik->jk", &inputs).unwrap_err();
        assert!(matches!(
            error.kind(),
            EinsumPlanErrorKind::LabelDimensionMismatch {
                label,
                first_size: 2,
                second_size: 4,
                ..
            } if label.as_char() == 'i'
        ));
        assert!(error.to_string().contains("input #0 axis 0"));
        assert!(error.to_string().contains("input #1 axis 0"));

        let left = [2usize, 3, 4];
        let right = [5usize, 4, 6];
        let inputs = [
            EinsumInput::new(DataType::Float32, &left),
            EinsumInput::new(DataType::Float32, &right),
        ];
        let error = EinsumPlan::build("...ij,...jk->...ik", &inputs).unwrap_err();
        assert!(matches!(
            error.kind(),
            EinsumPlanErrorKind::EllipsisDimensionMismatch {
                axis: 0,
                first_size: 2,
                second_size: 5,
                ..
            }
        ));
    }

    #[test]
    fn dtype_and_metadata_validation_is_explicit() {
        let shape = [2usize];
        let missing_dtype = [EinsumInput::from_optional(None, Some(&shape))];
        let error = EinsumPlan::build("i->i", &missing_dtype).unwrap_err();
        assert!(error.is_incomplete_metadata());
        assert!(matches!(
            error.kind(),
            EinsumPlanErrorKind::MissingInputDtype { input: 0 }
        ));

        let missing_shape = [EinsumInput::<usize>::from_optional(
            Some(DataType::Float32),
            None,
        )];
        assert!(matches!(
            error_kind("i->i", &missing_shape),
            EinsumPlanErrorKind::MissingInputShape { input: 0 }
        ));

        let known_invalid_after_unknown = [
            EinsumInput::<usize>::from_optional(None, None),
            EinsumInput::<usize>::from_optional(Some(DataType::Bool), None),
        ];
        assert!(matches!(
            error_kind("i,i->i", &known_invalid_after_unknown),
            EinsumPlanErrorKind::UnsupportedInputDtype {
                input: 1,
                dtype: DataType::Bool,
                ..
            }
        ));

        let invalid_known_shape_before_unknown = [
            EinsumInput::from_optional(Some(DataType::Float32), Some(&[2usize, 3][..])),
            EinsumInput::<usize>::from_optional(Some(DataType::Float32), None),
        ];
        assert!(matches!(
            error_kind("i,j->ij", &invalid_known_shape_before_unknown),
            EinsumPlanErrorKind::InputRankMismatch {
                input: 0,
                rank: 2,
                named_labels: 1,
                ..
            }
        ));

        let unsupported = [EinsumInput::new(DataType::Bool, &shape)];
        assert!(matches!(
            error_kind("i->i", &unsupported),
            EinsumPlanErrorKind::UnsupportedInputDtype {
                input: 0,
                dtype: DataType::Bool,
                schema: EinsumSchema::V12,
            }
        ));

        let bfloat16 = [EinsumInput::new(DataType::BFloat16, &shape)];
        let error = EinsumPlan::build("i->i", &bfloat16).unwrap_err();
        assert!(matches!(
            error.kind(),
            EinsumPlanErrorKind::UnsupportedInputDtype {
                input: 0,
                dtype: DataType::BFloat16,
                schema: EinsumSchema::V12,
            }
        ));
        assert!(error.to_string().contains("not admitted by Einsum-12"));
        assert!(EinsumPlan::build_for_schema("i->i", &bfloat16, EinsumSchema::V28).is_ok());

        let integer = [3usize];
        let mismatch = [
            EinsumInput::new(DataType::Float32, &shape),
            EinsumInput::new(DataType::Int32, &integer),
        ];
        assert!(matches!(
            error_kind("i,i->i", &mismatch),
            EinsumPlanErrorKind::InputDtypeMismatch {
                input: 1,
                expected: DataType::Float32,
                actual: DataType::Int32
            }
        ));
    }

    #[test]
    fn shape_only_planning_does_not_fabricate_a_dtype() {
        let left = [2usize, 3];
        let right = [3usize, 4];
        let plan = EinsumShapePlan::build("ik,kj->ij", &[&left, &right]).unwrap();
        assert!(matches!(
            plan.classification(),
            EinsumClassification::Gemm(_)
        ));
        assert_eq!(
            plan.output_shape()
                .iter()
                .map(|dimension| dimension.as_static())
                .collect::<Vec<_>>(),
            [Some(2), Some(4)]
        );
    }

    #[test]
    fn checked_geometry_reports_input_output_and_gemm_group_overflow() {
        let overflow = [usize::MAX, 2];
        let input = [EinsumInput::new(DataType::Float32, &overflow)];
        assert!(matches!(
            error_kind("ij->ij", &input),
            EinsumPlanErrorKind::GeometryOverflow {
                target: EinsumOverflowTarget::Input(0)
            }
        ));

        let output_overflow = [usize::MAX, 2, 0];
        let input = [EinsumInput::new(DataType::Float32, &output_overflow)];
        assert!(matches!(
            error_kind("abc->ab", &input),
            EinsumPlanErrorKind::GeometryOverflow {
                target: EinsumOverflowTarget::Output
            }
        ));

        let left = [usize::MAX, 2, 0];
        let right = [0usize, 0];
        let inputs = [
            EinsumInput::new(DataType::Float32, &left),
            EinsumInput::new(DataType::Float32, &right),
        ];
        assert!(matches!(
            error_kind("abk,kn->abn", &inputs),
            EinsumPlanErrorKind::GeometryOverflow {
                target: EinsumOverflowTarget::GemmM
            }
        ));

        let left = [0usize, usize::MAX, 2];
        let right = [usize::MAX, 2, 0];
        let inputs = [
            EinsumInput::new(DataType::Float32, &left),
            EinsumInput::new(DataType::Float32, &right),
        ];
        assert!(matches!(
            error_kind("mab,abn->mn", &inputs),
            EinsumPlanErrorKind::GeometryOverflow {
                target: EinsumOverflowTarget::GemmK
            }
        ));

        let left = [usize::MAX, 2, 0, 1];
        let right = [usize::MAX, 2, 1, 0];
        let inputs = [
            EinsumInput::new(DataType::Float32, &left),
            EinsumInput::new(DataType::Float32, &right),
        ];
        assert!(matches!(
            error_kind("abmk,abkn->abmn", &inputs),
            EinsumPlanErrorKind::GeometryOverflow {
                target: EinsumOverflowTarget::GemmBatch
            }
        ));
    }

    #[test]
    fn output_resolution_uses_the_planned_representative_and_broadcast_order() {
        let left = [1usize, 2, 3];
        let right = [6usize, 3, 4];
        let inputs = [
            EinsumInput::new(DataType::Float32, &left),
            EinsumInput::new(DataType::Float32, &right),
        ];
        let plan = EinsumPlan::build("...ij,...jk->...ik", &inputs).unwrap();
        let shapes: [&[usize]; 2] = [&left, &right];
        let mut calls = 0;
        let output = plan
            .resolve_output_shape(&shapes, |left, right| {
                calls += 1;
                if *left == 1 {
                    Ok::<_, ()>(*right)
                } else if *right == 1 || left == right {
                    Ok(*left)
                } else {
                    Err(())
                }
            })
            .unwrap();
        assert_eq!(output, vec![6, 2, 4]);
        assert_eq!(calls, 1);

        assert!(matches!(
            plan.resolve_output_shape(&shapes[..1], |left, _| Ok::<_, ()>(*left)),
            Err(EinsumResolveError::InputCount {
                expected: 2,
                found: 1
            })
        ));
        let short = [1usize, 2];
        let wrong_shapes: [&[usize]; 2] = [&short, &right];
        assert!(matches!(
            plan.resolve_output_shape(&wrong_shapes, |left, _| Ok::<_, ()>(*left)),
            Err(EinsumResolveError::InputRank {
                input: 0,
                expected: 3,
                found: 2
            })
        ));
    }

    #[test]
    fn symbolic_plan_validates_concrete_shapes_and_resolves_gemm_geometry() {
        let left = [
            Dim::Symbolic(SymbolId(1)),
            Dim::Static(2),
            Dim::Symbolic(SymbolId(2)),
        ];
        let right = [
            Dim::Symbolic(SymbolId(3)),
            Dim::Symbolic(SymbolId(2)),
            Dim::Static(4),
        ];
        let inputs = [
            EinsumInput::new(DataType::Float32, &left),
            EinsumInput::new(DataType::Float32, &right),
        ];
        let plan = EinsumPlan::build("...mk,...kn->...mn", &inputs).unwrap();
        let concrete_left = [1usize, 2, 3];
        let concrete_right = [6usize, 3, 4];
        let concrete: [&[usize]; 2] = [&concrete_left, &concrete_right];
        assert_eq!(
            plan.resolve_concrete_output_shape(&concrete).unwrap(),
            vec![6, 2, 4]
        );
        let geometry = plan
            .resolve_concrete_gemm_geometry(&concrete)
            .unwrap()
            .unwrap();
        assert_eq!(geometry.batch_shape(), &[6]);
        assert_eq!(geometry.batch(), 6);
        assert_eq!(geometry.m(), 2);
        assert_eq!(geometry.k(), 3);
        assert_eq!(geometry.n(), 4);

        let wrong_contract = [6usize, 5, 4];
        let wrong: [&[usize]; 2] = [&concrete_left, &wrong_contract];
        assert!(matches!(
            plan.resolve_concrete_output_shape(&wrong).unwrap_err().kind(),
            EinsumPlanErrorKind::LabelDimensionMismatch {
                label,
                first_size: 3,
                second_size: 5,
                ..
            } if label.as_char() == 'k'
        ));

        let wrong_static = [1usize, 7, 3];
        let wrong: [&[usize]; 2] = [&wrong_static, &concrete_right];
        assert!(matches!(
            plan.resolve_concrete_output_shape(&wrong)
                .unwrap_err()
                .kind(),
            EinsumPlanErrorKind::ResolvedInputDimensionMismatch {
                input: 0,
                axis: 1,
                expected: 2,
                found: 7
            }
        ));
    }
}

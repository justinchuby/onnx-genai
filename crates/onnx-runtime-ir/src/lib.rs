//! # `onnx-runtime-ir`
//!
//! The Graph intermediate representation (IR) for the ORT 2.0 runtime.
//!
//! This crate is the **stable contract** that every downstream runtime crate
//! (`onnx-runtime-loader`, `onnx-runtime-ep-api`, `onnx-runtime-session`, …)
//! builds against. It is intentionally pure, safe Rust with no FFI and no
//! device dependencies so it compiles standalone on any target.
//!
//! It is a Rust port of the design captured in `docs/architecture/ORT2.md` §3 (Graph IR),
//! §5 (Striding & Layout) and §11 (Dynamic Shape), itself inspired by the
//! Python [`onnx-ir`](https://github.com/onnx/ir-py) package.
//!
//! ## What lives here
//!
//! | Concept | Type |
//! |---------|------|
//! | Element type | [`DataType`] |
//! | Symbolic / static shapes | [`Shape`], [`Dim`], [`SymbolConstraints`] |
//! | Physical strided layout | [`TensorLayout`], [`MemoryFormat`] |
//! | Canonical Einsum planning | [`EinsumPlan`], [`EinsumShapePlan`], [`EinsumPlanningClassification`] |
//! | Device placement | [`DeviceType`], [`DeviceId`] |
//! | Graph values (SSA edges) | [`Value`], [`ValueId`] |
//! | Graph operations | [`Node`], [`NodeId`], [`Attribute`] |
//! | Constant / weight storage | [`TensorData`], [`SparseTensorData`], [`WeightRef`] |
//! | The graph itself | [`Graph`] |
//! | Errors | [`IrError`], [`GraphError`] |
//!
//! ## Design guarantees
//!
//! * **SSA-like:** every [`Value`] has at most one producer [`Node`]; node
//!   outputs are unique.
//! * **First-class layout & device:** every value carries a [`TensorLayout`]
//!   and an optional [`DeviceId`], unlike upstream ONNX / `onnx-ir`.
//! * **Mutable during optimization:** the [`Graph`] mutation API keeps
//!   producer/consumer edges consistent so optimization passes can rewrite it,
//!   then it is shared immutably via `Arc` once frozen.
//!
//! Per-op graph shape inference remains in `onnx-runtime-shape-inference`.
//! Shared semantic contracts needed by both inference and execution providers,
//! such as [`EinsumPlan`] and [`EinsumShapePlan`], live here so every consumer
//! validates and classifies an operator exactly once. The `Graph` operations that are cheap and
//! foundational (topological ordering, validation, edge rewiring, broadcasting,
//! stride arithmetic) are likewise fully implemented and unit-tested.

#![forbid(unsafe_code)]

mod arena;
mod device;
mod domain;
mod dtype;
mod einsum;
mod error;
mod graph;
mod graph_view;
mod layout;
mod node;
mod scan_inline;
mod shape;
mod tensor;
mod value;

pub use arena::{Arena, ArenaKey};
pub use device::{DeviceId, DeviceType};
pub use domain::{AI_ONNX_DOMAIN, is_default_domain, normalize_domain};
pub use dtype::DataType;
pub use einsum::{
    EinsumAxis, EinsumAxisRef, EinsumBinaryContractionPlan, EinsumBinaryLowering,
    EinsumClassification, EinsumConcreteContractionTreeCandidate,
    EinsumConcreteContractionTreePlan, EinsumConcreteGemmGeometry, EinsumConcretePlanError,
    EinsumContractionCost, EinsumContractionPlan, EinsumContractionTreeCandidate,
    EinsumContractionTreeCandidateId, EinsumContractionTreeCandidatePlan,
    EinsumContractionTreeCandidateUnsupportedReason, EinsumContractionTreePlan,
    EinsumContractionTreeStep, EinsumCostBound, EinsumCostMetric, EinsumDimension,
    EinsumDimensionRule, EinsumDimensionValue, EinsumEquationSide, EinsumExecutionSelection,
    EinsumGemmGeometry, EinsumGenericNativePlan, EinsumIndexProgram, EinsumInput,
    EinsumIntegerOverflowSemantics, EinsumLabel, EinsumLogicalAxis, EinsumOperandAxis,
    EinsumOperandIndexProgram, EinsumOperandPlan, EinsumOpsetPlanError, EinsumOverflowTarget,
    EinsumPermutationPlan, EinsumPlan, EinsumPlanError, EinsumPlanErrorKind, EinsumPlannerBudget,
    EinsumPlannerFallbackReason, EinsumPlannerQuality, EinsumPlannerUsage,
    EinsumPlanningClassification, EinsumPrecisionPolicy, EinsumReductionPlan, EinsumResolveError,
    EinsumResolvedContractionCost, EinsumSchema, EinsumSchemaError, EinsumSemanticPlan,
    EinsumShapePlan, EinsumSupportedContractionTreeCandidate, EinsumTemporaryStoragePolicy,
    EinsumTemporaryValuePlan, EinsumUnaryReductionPlan, EinsumUnsupportedReason, EinsumValueId,
};
pub use error::{GraphError, IrError, Result};
pub use graph::{Graph, ModelFunction, ModelFunctionKey};
pub use graph_view::{ConsumerUse, FrozenGraph, GraphView, GraphViewCache, NodeIndex, ValueIndex};
pub use layout::{
    MemoryFormat, TensorLayout, broadcast_shapes, compute_contiguous_strides, is_contiguous,
    is_dense,
};
pub use node::{Attribute, Node, NodeId, RUNTIME_DOMAIN};
pub use scan_inline::inline_single_trip_scan_bodies;
pub use shape::{
    Dim, Shape, SymbolConstraints, SymbolId, as_static_shape, is_fully_static, static_shape,
};
pub use tensor::{
    FromLeBytes, RawBytesError, SparseTensorData, TensorData, TypeProto, WeightRef,
    checked_expected_bytes, checked_numel, read_scalar_le, read_vec_le,
};
pub use value::{Consumers, Usage, Value, ValueId};

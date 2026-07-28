//! # `onnx-runtime-shape-inference`
//!
//! Symbolic shape inference over the [`onnx_runtime_ir::Graph`] IR.
//!
//! This crate is the general, extensible successor to the bounded shape-
//! inference stopgaps elsewhere in the runtime (the loader's `const-fold-lite`
//! pass and the session's just-in-time data-dependent resolution). Its design
//! mirrors the reference implementation
//! [`justinchuby/onnx-shape-inference`](https://github.com/justinchuby/onnx-shape-inference):
//!
//! 1. **Extensible per-op registry** keyed by `(domain, op_type, opset)` with
//!    range-based version matching ([`InferenceRegistry`]). Unregistered ops
//!    leave their outputs unresolved rather than failing.
//! 2. **Symbolic dimension arithmetic** ([`DimExpr`]) — a small canonical
//!    integer polynomial that captures the affine/product forms the op set
//!    produces (`d0*d1`, `d0+k`, `d0/k`, reshape `-1` cancellation), lowered
//!    back to IR [`Dim`](onnx_runtime_ir::Dim)s only when writing results.
//! 3. **Shape-DATA propagation** ([`ShapeData`]) — tracks the known element
//!    values of the small integer tensors in `Shape → Slice → Concat → Gather
//!    → Unsqueeze → Reshape` chains, so computed shapes resolve without
//!    executing the graph. This is what lets transformer graphs infer
//!    statically.
//! 4. **Merge policies** ([`MergePolicy`]) — [`Strict`](MergePolicy::Strict)
//!    (concrete disagreements are errors) and
//!    [`Permissive`](MergePolicy::Permissive) (prefer the more specific dim and
//!    keep going; the robust default).
//!
//! ## Usage
//!
//! ```no_run
//! use onnx_runtime_shape_inference::{InferenceRegistry, MergePolicy};
//! # fn demo(graph: &mut onnx_runtime_ir::Graph) {
//! let registry = InferenceRegistry::default_registry();
//! let opsets = graph.opset_imports.clone();
//! let report = registry
//!     .infer_graph(graph, &opsets, MergePolicy::Permissive)
//!     .expect("inference");
//! assert!(report.fully_resolved());
//! # }
//! ```
//!
//! Single-node inference (for testing or custom passes) is available via
//! [`InferenceRegistry::infer_node`].
//!
//! ## Design invariants
//!
//! * **Model-agnostic.** Rules dispatch purely on `(domain, op_type, opset)` and
//!   tensor metadata — never on model names or op counts.
//! * **The IR contract is not modified.** Derived dimensions live in this
//!   crate's [`DimExpr`] and are lowered to a fresh symbol when they cannot be
//!   expressed as an IR [`Dim`](onnx_runtime_ir::Dim).
//! * **Permissive by default, never panics on unknown input.** Errors are
//!   reserved for genuine contract violations (see [`ShapeInferError`]).
//!
//! ## Control flow and the container-type limitation
//!
//! Control-flow ops that carry subgraph bodies are inferred by propagating
//! shapes *through* the body: `If` reconciles its two branch
//! outputs, while `Loop` and `Scan` seed the body's formal inputs from the
//! node's operands, infer the body, then map the body outputs back (stacking a
//! trip-count / scan axis where the op requires one).
//!
//! The **Sequence** family (`SequenceEmpty`/`Construct`/`Insert`/`Erase`/`At`/
//! `Length`/`ConcatFromSequence`/`SplitToSequence`), **Optional**
//! (`Optional`/`OptionalHasElement`/`OptionalGetElement`), and **Map** ops are
//! deliberately **deferred**: an SSA [`Value`](onnx_runtime_ir::Value) and this
//! crate's [`TypeInfo`] carry only a tensor `dtype` + `shape`, with no way to
//! represent a *container element type*. Correctly propagating, say, the
//! element shape produced by `SequenceAt` would require an IR/type-model
//! extension (a sum type over tensor/sequence/optional/map element types).
//! Rather than fabricate a bogus tensor shape for a sequence-typed edge, these
//! ops are left unregistered — their outputs stay *unresolved* (the permissive
//! outcome) instead of *wrong*. See issue #355.

#![forbid(unsafe_code)]

pub mod context;
pub mod dim_expr;
mod error;
mod handlers;
mod infer;
mod registry;
mod report;
pub mod shape_data;

pub use context::{
    InferenceContext, MergePolicy, NodeIo, SymbolInterner, TypeInfo, TypedShape, merge_shapes,
};
pub use dim_expr::DimExpr;
pub use error::ShapeInferError;
pub use registry::{InferenceFn, InferenceRegistry};
pub use report::InferenceReport;
pub use shape_data::{MAX_SHAPE_DATA_ELEMS, ShapeData};

//! Errors raised by shape inference.

use onnx_runtime_ir::ValueId;

/// An error produced while inferring shapes.
///
/// Inference is *permissive by default*: an unsupported operator or an
/// under-specified input never errors — the affected outputs are simply left
/// unresolved (see [`crate::InferenceReport`]). These variants are raised only
/// for genuine contract violations: a malformed graph (cycle), an operator used
/// with the wrong arity/rank, or — under [`MergePolicy::Strict`] — a concrete
/// dimension conflict between an inferred and a declared shape.
///
/// [`MergePolicy::Strict`]: crate::MergePolicy::Strict
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum ShapeInferError {
    /// The graph could not be topologically ordered (it contains a cycle).
    #[error("graph has a cycle; cannot order nodes for shape inference")]
    CycleDetected,

    /// An operator was invoked with the wrong number of inputs.
    #[error("op `{op}`: expected {expected} inputs, found {found}")]
    Arity {
        op: String,
        expected: String,
        found: usize,
    },

    /// An input had a rank the operator cannot accept.
    #[error("op `{op}`: input #{index} has invalid rank {rank} ({detail})")]
    InvalidRank {
        op: String,
        index: usize,
        rank: usize,
        detail: String,
    },

    /// An attribute required by the operator was missing or the wrong type.
    #[error("op `{op}`: attribute `{attr}` is missing or has the wrong type")]
    MissingAttribute { op: String, attr: String },

    /// A structural inconsistency detected while applying an op rule (e.g. a
    /// contraction-dimension mismatch in `MatMul`, or an out-of-range axis).
    #[error("op `{op}`: {detail}")]
    Invalid { op: String, detail: String },

    /// Under [`MergePolicy::Strict`](crate::MergePolicy::Strict), an inferred
    /// dimension disagreed with the value's declared dimension. Only concrete
    /// (static) disagreements are reported; symbolic differences are treated as
    /// naming and never conflict (see [`crate::context::merge_shapes`]).
    #[error(
        "value {value:?}: inferred dim {inferred} conflicts with declared dim {declared} at axis {axis}"
    )]
    ShapeConflict {
        value: ValueId,
        axis: usize,
        inferred: i64,
        declared: i64,
    },

    /// Under [`MergePolicy::Strict`](crate::MergePolicy::Strict), an inferred
    /// rank disagreed with the value's declared rank.
    #[error("value {value:?}: inferred rank {inferred} conflicts with declared rank {declared}")]
    RankConflict {
        value: ValueId,
        inferred: usize,
        declared: usize,
    },
}

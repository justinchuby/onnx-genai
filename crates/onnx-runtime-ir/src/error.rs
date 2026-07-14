//! Error types for the IR crate (see `docs/ORT2.md` §22 and §3.3).
//!
//! The IR uses two error types:
//! * [`GraphError`] — a single structural defect found while validating or
//!   ordering a [`Graph`](crate::Graph).
//! * [`IrError`] — the crate-level error returned by fallible IR operations,
//!   a subset of the runtime's top-level `Error` (§22) restricted to what the
//!   IR itself can produce.

use crate::node::NodeId;
use crate::value::ValueId;

/// A convenience `Result` alias for IR operations.
pub type Result<T> = std::result::Result<T, IrError>;

/// A single structural defect in a [`Graph`](crate::Graph).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphError {
    /// A node references a value id that is not live in the value arena.
    DanglingValue(ValueId),
    /// A referenced node id is not live in the node arena.
    DanglingNode(NodeId),
    /// The graph contains a cycle (no valid topological order).
    CycleDetected,
    /// A graph output (or interior value) has no producing node.
    MissingProducer(ValueId),
    /// The same value is produced as an output more than once (SSA violation).
    DuplicateOutput(ValueId),
    /// A graph input has a producer node (inputs must be sources).
    InputHasProducer(ValueId),
    /// A value's `producer` link disagrees with the node's `outputs`.
    ProducerLinkMismatch(ValueId),
    /// A value's `consumers` list disagrees with a node's `inputs`.
    ConsumerLinkMismatch(ValueId),
    /// An opset import is malformed.
    InvalidOpsetImport { domain: String, version: u64 },
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphError::DanglingValue(v) => write!(f, "dangling value id {v:?}"),
            GraphError::DanglingNode(n) => write!(f, "dangling node id {n:?}"),
            GraphError::CycleDetected => write!(f, "cycle detected in graph"),
            GraphError::MissingProducer(v) => write!(f, "value {v:?} has no producer"),
            GraphError::DuplicateOutput(v) => write!(f, "value {v:?} produced more than once"),
            GraphError::InputHasProducer(v) => write!(f, "graph input {v:?} has a producer"),
            GraphError::ProducerLinkMismatch(v) => {
                write!(f, "producer link inconsistent for value {v:?}")
            }
            GraphError::ConsumerLinkMismatch(v) => {
                write!(f, "consumer link inconsistent for value {v:?}")
            }
            GraphError::InvalidOpsetImport { domain, version } => {
                write!(f, "invalid opset import: domain={domain} version={version}")
            }
        }
    }
}

impl std::error::Error for GraphError {}

/// Crate-level error for fallible IR operations.
#[derive(Debug, thiserror::Error)]
pub enum IrError {
    /// One or more structural defects were found by
    /// [`Graph::validate`](crate::Graph::validate).
    #[error("graph validation failed: {0:?}")]
    GraphInvalid(Vec<GraphError>),

    /// A cycle was detected while computing a topological order.
    #[error("cycle detected in graph")]
    CycleDetected,

    /// A referenced value id is not live in the graph.
    #[error("unknown value id: {0:?}")]
    UnknownValue(ValueId),

    /// A referenced node id is not live in the graph.
    #[error("unknown node id: {0:?}")]
    UnknownNode(NodeId),

    /// Two shapes that were required to match did not.
    #[error("shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// Two shapes could not be broadcast together.
    #[error("broadcast incompatible: {a:?} vs {b:?}")]
    BroadcastIncompatible { a: Vec<usize>, b: Vec<usize> },

    /// An opset version the IR does not model.
    #[error("unsupported opset: domain={domain}, version={version}")]
    UnsupportedOpset { domain: String, version: u64 },
}

impl From<Vec<GraphError>> for IrError {
    fn from(errors: Vec<GraphError>) -> Self {
        IrError::GraphInvalid(errors)
    }
}

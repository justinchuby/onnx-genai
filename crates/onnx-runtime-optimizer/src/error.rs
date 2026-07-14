//! Error types for the optimizer crate.
//!
//! Mirrors the two-layer error style of `onnx-runtime-ir`: structural graph
//! defects surface as [`GraphError`](onnx_runtime_ir::GraphError)s, wrapped
//! here in the crate-level [`OptimizerError`].

use onnx_runtime_ir::{GraphError, IrError};

/// A convenience `Result` alias for optimizer operations.
pub type Result<T> = std::result::Result<T, OptimizerError>;

/// Crate-level error for the optimization pipeline.
#[derive(Debug, thiserror::Error)]
pub enum OptimizerError {
    /// A pass left the graph in a structurally invalid state. Raised by the
    /// default [`OptimizationPass::postconditions`](crate::OptimizationPass::postconditions)
    /// check (debug builds).
    #[error("graph invariant violated after pass '{pass}': {errors:?}")]
    PostconditionFailed {
        /// The pass whose postcondition failed.
        pass: String,
        /// The structural defects found by `graph.validate()`.
        errors: Vec<GraphError>,
    },

    /// A rewrite requested an operation the IR rejected.
    #[error(transparent)]
    Ir(#[from] IrError),

    /// A fusion could not be applied safely (e.g. malformed match).
    #[error("fusion '{0}' could not be applied")]
    Fusion(String),
}

impl From<Vec<GraphError>> for OptimizerError {
    fn from(errors: Vec<GraphError>) -> Self {
        OptimizerError::Ir(IrError::GraphInvalid(errors))
    }
}

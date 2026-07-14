//! The result summary returned by whole-graph inference.

use onnx_runtime_ir::ValueId;

/// A summary of what whole-graph inference resolved.
///
/// A value is "resolved" once it has a known dtype and a known-rank shape (every
/// dimension is concrete or symbolic — never unknown). Values an op rule could
/// not resolve (unregistered op, data-dependent extent without shape-data) are
/// listed in [`unresolved`](InferenceReport::unresolved).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InferenceReport {
    /// Total number of live values in the graph.
    pub total_values: usize,
    /// Values that ended with a resolved shape.
    pub resolved: Vec<ValueId>,
    /// Values left without a resolved shape.
    pub unresolved: Vec<ValueId>,
    /// Number of fresh symbolic dimensions minted during inference.
    pub fresh_symbols: usize,
}

impl InferenceReport {
    /// The number of resolved values.
    pub fn num_resolved(&self) -> usize {
        self.resolved.len()
    }

    /// The number of unresolved values.
    pub fn num_unresolved(&self) -> usize {
        self.unresolved.len()
    }

    /// Whether every live value in the graph resolved to a known shape.
    pub fn fully_resolved(&self) -> bool {
        self.unresolved.is_empty()
    }
}

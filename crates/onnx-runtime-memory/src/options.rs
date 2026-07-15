//! Planner configuration knobs.

/// Options controlling what the activation planner considers part of the arena.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct PlanOptions {
    /// Include graph **inputs** as buffer owners in the activation arena.
    ///
    /// Defaults to `false`: graph inputs are supplied by the caller (feeds) and
    /// are not part of the executor-owned scratch arena. Set to `true` when the
    /// caller wants the planner to also account for input buffers (e.g. a
    /// self-contained arena that copies feeds in).
    pub include_graph_inputs: bool,
}

impl PlanOptions {
    /// The default options (graph inputs excluded).
    pub fn new() -> Self {
        Self::default()
    }

    /// Also treat graph inputs as arena buffer owners.
    pub fn with_graph_inputs(mut self, include: bool) -> Self {
        self.include_graph_inputs = include;
        self
    }
}

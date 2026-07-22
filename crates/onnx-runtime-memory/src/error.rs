//! Error types for planning and validation.

use onnx_runtime_ir::ValueId;

use crate::plan::SlotId;

/// A failure that prevents the planner from producing any plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PlanError {
    /// The graph is not schedulable (contains a cycle), so no execution order
    /// — and therefore no liveness — exists.
    #[error("graph is not schedulable (cycle detected); cannot compute liveness")]
    Cycle,
}

/// A violated correctness invariant found by [`crate::validate`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ValidateError {
    /// Two values with overlapping live intervals were assigned the same slot —
    /// one would clobber the other's still-live data.
    #[error("values {a:?} and {b:?} have overlapping live intervals but share slot {slot:?}")]
    SlotConflict {
        a: ValueId,
        b: ValueId,
        slot: SlotId,
    },
    /// A value is assigned to a slot too small to hold it.
    #[error("value {value:?} needs {needed} bytes but slot {slot:?} has capacity {capacity}")]
    UndersizedSlot {
        value: ValueId,
        slot: SlotId,
        needed: usize,
        capacity: usize,
    },
    /// A buffer owner has no slot assignment in the plan.
    #[error("buffer owner {value:?} has no slot assignment")]
    MissingAssignment { value: ValueId },
    /// A value is assigned to a slot id that does not exist in the plan.
    #[error("value {value:?} references unknown slot {slot:?}")]
    UnknownSlot { value: ValueId, slot: SlotId },
    /// A zero-copy view outlives the source buffer it aliases (fold error): the
    /// source could be recycled while the view still points into it.
    #[error(
        "view {view:?} is used at node {view_use} but its source {source_owner:?} is retired at node {source_end}"
    )]
    ViewOutlivesSource {
        view: ValueId,
        source_owner: ValueId,
        view_use: usize,
        source_end: usize,
    },
    /// The graph became unschedulable between planning and validation.
    #[error("graph is not schedulable (cycle detected)")]
    Cycle,
}

//! The **size oracle**: byte size of an activation value.
//!
//! Sizes may be unknown at build time when a shape has symbolic (dynamic)
//! dimensions. The planner is generic over any `Fn(ValueId) -> Option<usize>`
//! so the *same* algorithm serves two callers:
//!
//! * **build-time** planning from fully-static shapes ([`static_size_oracle`]),
//!   which returns `None` for any symbolic-shaped value and drives the planner
//!   to a [`crate::PlanStatus::Deferred`] result; and
//! * **run-time** planning, where the executor supplies a closure backed by the
//!   resolved concrete shapes for the current run.

use onnx_runtime_ir::{as_static_shape, Graph, ValueId};

/// Byte size of a value from its *static* shape, or `None` if any dimension is
/// symbolic (unknown until run time) or the element count overflows `usize`.
///
/// Uses [`onnx_runtime_ir::DataType::checked_storage_bytes`] so sub-byte packed
/// types (`int4`/`uint4`/`float4`) are sized correctly and an overflowing
/// element count becomes `None` rather than a wrapped under-count.
pub fn static_size(graph: &Graph, value: ValueId) -> Option<usize> {
    let val = graph.try_value(value)?;
    let dims = as_static_shape(&val.shape)?;
    let mut numel: usize = 1;
    for d in dims {
        numel = numel.checked_mul(d)?;
    }
    val.dtype.checked_storage_bytes(numel)
}

/// A size oracle closure that sizes values from their fully-static shapes.
///
/// Returns `None` for any symbolic-shaped value, which the planner reports as
/// [`crate::PlanStatus::Deferred`] so the executor can re-plan once shapes
/// resolve.
pub fn static_size_oracle(graph: &Graph) -> impl Fn(ValueId) -> Option<usize> + '_ {
    move |value| static_size(graph, value)
}

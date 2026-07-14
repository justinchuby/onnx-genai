//! Best-effort static/symbolic shape inference (§19.3).
//!
//! Walks the graph in topological order and dispatches to per-op shape rules.
//! The per-op rule table is a substantial downstream task; this skeleton
//! provides the driver signature only.

use onnx_runtime_ir::{Graph, Shape};

use crate::LoaderError;

/// Run shape inference over the whole graph, populating value shapes in place.
pub fn run_shape_inference(graph: Graph) -> Result<Graph, LoaderError> {
    let _order = graph.topological_order().map_err(|_| {
        LoaderError::GraphBuild("cycle detected during shape inference".into())
    })?;
    todo!("ort2-loader: per-op shape inference dispatch (see docs/ORT2.md §19.3)")
}

/// Infer output shapes for a single op given its input shapes and attributes.
///
/// The full rule table (MatMul, Conv, broadcasting elementwise, Reshape,
/// Gather, …) is a downstream task.
pub fn infer_op_shapes(
    op_type: &str,
    domain: &str,
    input_shapes: &[Shape],
) -> Result<Vec<Shape>, LoaderError> {
    let _ = (op_type, domain, input_shapes);
    todo!("ort2-loader: per-op shape inference rule")
}

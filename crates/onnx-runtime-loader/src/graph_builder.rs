//! Build an [`onnx_runtime_ir::Graph`] from a decoded `ModelProto` (§19.1).
//!
//! Responsible for the graph-construction invariants of `docs/ORT2.md` §3.5:
//! stable value ids, unique node outputs (SSA), source values for inputs and
//! initializers, and interning symbolic dims that share a protobuf name.

use onnx_runtime_ir::Graph;

use crate::proto::ModelProto;
use crate::LoaderError;

/// Build the IR graph (without weights or shape inference yet).
pub fn build_graph(model: &ModelProto) -> Result<Graph, LoaderError> {
    let _ = model;
    todo!("ort2-loader: build IR Graph from GraphProto (nodes, values, symbols, opsets)")
}

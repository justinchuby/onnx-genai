//! Recognize a single-row FP32 projection whose immutable RHS widens from FLOAT6.

use crate::{Attribute, DataType, Dim, Graph, Node, NodeId, ValueId};
use std::collections::HashSet;

pub struct Float6Projection {
    pub activation: ValueId,
    pub packed: ValueId,
    pub output: ValueId,
    pub cast: NodeId,
}

impl Float6Projection {
    pub fn recognize(graph: &Graph, matmul: &Node, captured: &HashSet<ValueId>) -> Option<Self> {
        if !matmul.is_default_domain() || matmul.op_type != "MatMul" {
            return None;
        }
        let [Some(a), Some(w)] = matmul.inputs.as_slice() else {
            return None;
        };
        let [Some(output)] = matmul.outputs.as_slice() else {
            return None;
        };
        let widened = graph.value(*w);
        if widened.is_graph_output || graph.num_uses(*w) != 1 || captured.contains(w) {
            return None;
        }
        let cast = graph.node(widened.producer?);
        if !cast.is_default_domain()
            || cast.op_type != "Cast"
            || cast.attr("to").and_then(Attribute::as_int) != Some(1)
            || graph.effective_opset(cast).unwrap_or(0) < 28
        {
            return None;
        }
        let [Some(packed)] = cast.inputs.as_slice() else {
            return None;
        };
        let source = graph.initializers.get(packed)?;
        let [k, n] = source.dims() else { return None };
        let activation = graph.value(*a);
        if !source.dtype().is_float6()
            || graph.value(*packed).is_graph_input
            || activation.dtype != DataType::Float32
            || activation.shape.as_slice() != [Dim::Static(1), Dim::Static(*k)]
            || *k == 0
            || *n == 0
            || !n.is_multiple_of(4)
        {
            return None;
        }
        Some(Self {
            activation: *a,
            packed: *packed,
            output: *output,
            cast: cast.id,
        })
    }
}

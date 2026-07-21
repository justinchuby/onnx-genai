use onnx_runtime_ir::{Attribute, DataType, Graph, NodeId};
use onnx_runtime_optimizer::{
    OptimizationPass, OptimizerError, PassContext, Result as OptimizerResult,
};

pub(crate) const SILU_MUL_FUSION_ATTR: &str = "_cuda_silu_mul";
const MICROSOFT_DOMAIN: &str = "com.microsoft";

/// Fuse `Mul(Silu(gate), up)` into CUDA's tagged two-input `Mul` variant.
///
/// Keeping the node as standard `Mul` preserves ordinary binary shape
/// inference. The private marker is consumed only by the CUDA kernel factory;
/// the session restores the pre-pass graph before falling back to another EP.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CudaSwiGluFusion;

pub(crate) fn cuda_optimization_passes() -> Vec<Box<dyn OptimizationPass>> {
    vec![Box::new(CudaSwiGluFusion)]
}

impl OptimizationPass for CudaSwiGluFusion {
    fn name(&self) -> &str {
        "CudaSwiGluFusion"
    }

    fn run(&self, graph: &mut Graph, _ctx: &PassContext) -> OptimizerResult<()> {
        let silu_nodes: Vec<NodeId> = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| {
                (node.op_type == "Silu"
                    && node.domain == MICROSOFT_DOMAIN
                    && node.inputs.len() == 1
                    && node.outputs.len() == 1)
                    .then_some(id)
            })
            .collect();

        let mut changed = false;
        for silu_id in silu_nodes {
            let Some(silu) = graph.try_node(silu_id) else {
                continue;
            };
            let Some(gate) = silu.inputs[0] else {
                continue;
            };
            let silu_output = silu.outputs[0];
            if graph.outputs.contains(&silu_output) {
                continue;
            }
            let consumers = graph.consumers(silu_output);
            if consumers.len() != 1 {
                continue;
            }

            let mul_id = consumers[0];
            let mul = graph.node(mul_id);
            if mul.op_type != "Mul"
                || !matches!(mul.domain.as_str(), "" | "ai.onnx")
                || mul.inputs.len() != 2
                || mul.outputs.len() != 1
                || !mul.attributes.is_empty()
            {
                continue;
            }
            let up = if mul.inputs[0] == Some(silu_output) {
                mul.inputs[1]
            } else if mul.inputs[1] == Some(silu_output) {
                mul.inputs[0]
            } else {
                None
            };
            let Some(up) = up else {
                continue;
            };

            let gate_value = graph.value(gate);
            let up_value = graph.value(up);
            if gate_value.dtype != up_value.dtype
                || gate_value.shape != up_value.shape
                || !matches!(
                    gate_value.dtype,
                    DataType::Float16 | DataType::Float32 | DataType::BFloat16
                )
            {
                continue;
            }

            let mut fused = mul.clone();
            fused.inputs = vec![Some(gate), Some(up)];
            fused
                .attributes
                .insert(SILU_MUL_FUSION_ATTR.into(), Attribute::Int(1));
            graph.replace_node(mul_id, fused);
            graph.remove_node(silu_id);
            changed = true;
        }

        if changed {
            graph.validate().map_err(OptimizerError::from)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{Dim, Node, NodeId, ValueId};

    fn value(graph: &mut Graph, name: &str, dtype: DataType, width: usize) -> ValueId {
        graph.create_named_value(name, dtype, vec![Dim::Static(1), Dim::Static(width)])
    }

    fn swiglu_graph(dtype: DataType, gate_width: usize, up_width: usize) -> Graph {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        graph.opset_imports.insert(MICROSOFT_DOMAIN.into(), 1);
        let gate = value(&mut graph, "gate", dtype, gate_width);
        let up = value(&mut graph, "up", dtype, up_width);
        let silu_output = value(&mut graph, "silu", dtype, gate_width);
        let output = value(&mut graph, "output", dtype, gate_width);
        graph.add_input(gate);
        graph.add_input(up);
        let mut silu = Node::new(NodeId(0), "Silu", vec![Some(gate)], vec![silu_output]);
        silu.domain = MICROSOFT_DOMAIN.into();
        graph.insert_node(silu);
        graph.insert_node(Node::new(
            NodeId(0),
            "Mul",
            vec![Some(silu_output), Some(up)],
            vec![output],
        ));
        graph.add_output(output);
        graph
    }

    #[test]
    fn fuses_equal_shape_silu_mul() {
        let mut graph = swiglu_graph(DataType::Float16, 7, 7);
        CudaSwiGluFusion
            .run(&mut graph, &PassContext::new())
            .unwrap();

        assert_eq!(graph.num_nodes(), 1);
        let fused = graph.nodes.values().next().unwrap();
        assert_eq!(fused.op_type, "Mul");
        assert_eq!(
            fused.attr(SILU_MUL_FUSION_ATTR).and_then(Attribute::as_int),
            Some(1)
        );
        assert_eq!(fused.inputs.len(), 2);
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn leaves_broadcast_silu_mul_separate() {
        let mut graph = swiglu_graph(DataType::Float16, 7, 1);
        CudaSwiGluFusion
            .run(&mut graph, &PassContext::new())
            .unwrap();

        assert_eq!(graph.num_nodes(), 2);
        assert!(
            graph
                .nodes
                .values()
                .all(|node| node.attr(SILU_MUL_FUSION_ATTR).is_none())
        );
    }
}

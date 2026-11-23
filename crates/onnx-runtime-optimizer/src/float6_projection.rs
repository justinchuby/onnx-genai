use crate::{OptimizationPass, PassContext, Result};
use onnx_runtime_ir::{Graph, Node, NodeId, float6_projection::Float6Projection};

pub struct Float6ProjectionFusion;

impl OptimizationPass for Float6ProjectionFusion {
    fn name(&self) -> &str {
        "Float6ProjectionFusion"
    }

    fn run(&self, graph: &mut Graph, _: &PassContext) -> Result<()> {
        let candidates: Vec<_> = graph
            .nodes
            .iter()
            .filter_map(|(id, node)| (node.op_type == "MatMul").then_some(id))
            .collect();
        let captured = graph
            .captured_inputs_by_node()
            .into_values()
            .flatten()
            .collect();
        for id in candidates {
            let Some(p) = Float6Projection::recognize(graph, graph.node(id), &captured) else {
                continue;
            };
            let mut fused = Node::new(
                NodeId(0),
                "Float6Projection",
                vec![Some(p.activation), Some(p.packed)],
                vec![p.output],
            );
            fused.domain = "pkg.nxrt".into();
            graph.opset_imports.entry(fused.domain.clone()).or_insert(1);
            graph.replace_node(id, fused);
            graph.remove_node(p.cast);
        }
        Ok(())
    }
}

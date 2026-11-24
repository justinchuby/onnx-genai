use std::collections::HashSet;

use onnx_runtime_ir::{Graph, RUNTIME_DOMAIN, dequantize_packed_gather};

use crate::{OptimizationPass, PassContext, Result};

pub struct DequantizePackedGatherFusion;

impl OptimizationPass for DequantizePackedGatherFusion {
    fn name(&self) -> &str {
        "DequantizePackedGatherFusion"
    }

    fn run(&self, graph: &mut Graph, _: &PassContext) -> Result<()> {
        let captured: HashSet<_> = graph
            .captured_inputs_by_node()
            .into_values()
            .flatten()
            .collect();
        let mut changed = false;
        for id in graph.topological_order().map_err(|error| vec![error])? {
            if let Ok((dq, fused)) =
                dequantize_packed_gather::recognize(graph, graph.node(id), &|value| {
                    captured.contains(&value)
                })
            {
                graph.replace_node(id, fused);
                graph.remove_node(dq);
                changed = true;
            }
        }
        if changed {
            graph
                .opset_imports
                .entry(RUNTIME_DOMAIN.into())
                .or_insert(1);
        }
        Ok(())
    }
}

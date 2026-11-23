use std::collections::HashSet;

use onnx_runtime_ir::{Graph, RUNTIME_DOMAIN, affine_gather};

use crate::{OptimizationPass, PassContext, Result};

/// Run only on providers registering the corresponding native kernel.
pub struct AffineGatherFusion;

impl OptimizationPass for AffineGatherFusion {
    fn name(&self) -> &str {
        "AffineGatherFusion"
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
                affine_gather::recognize(graph, graph.node(id), &|v| captured.contains(&v))
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

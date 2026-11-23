use std::collections::{HashMap, HashSet};

use onnx_runtime_ir::{Attribute, DataType, Graph, RUNTIME_DOMAIN, ValueId, quantized_grouped};

use crate::{OptimizationPass, PassContext, Result};

pub struct QuantizedGroupedFusion;

impl OptimizationPass for QuantizedGroupedFusion {
    fn name(&self) -> &str {
        "QuantizedGroupedFusion"
    }

    fn run(&self, graph: &mut Graph, ctx: &PassContext) -> Result<()> {
        let captured: HashSet<_> = graph
            .captured_inputs_by_node()
            .into_values()
            .flatten()
            .collect();
        let order = graph.topological_order().map_err(|error| vec![error])?;
        let positions: HashMap<_, _> = order
            .iter()
            .copied()
            .enumerate()
            .map(|(i, id)| (id, i))
            .collect();
        for id in order {
            if graph.node(id).op_type != "GroupedMatMul" {
                continue;
            }
            if let Ok((mut dead, replacement)) = quantized_grouped::recognize(
                graph,
                graph.node(id),
                &|v| literal(graph, ctx, v),
                &|v| captured.contains(&v),
            ) {
                graph.replace_node(id, replacement);
                // Consumers must be disconnected before their producers are removed.
                dead.sort_unstable_by_key(|id| std::cmp::Reverse(positions[id]));
                dead.dedup();
                graph.remove_nodes(&dead);
                graph
                    .opset_imports
                    .entry(RUNTIME_DOMAIN.into())
                    .or_insert(1);
            }
        }
        Ok(())
    }
}

fn literal(graph: &Graph, ctx: &PassContext, id: ValueId) -> Option<Vec<i64>> {
    let value = graph.value(id);
    if value
        .tensor_shape()?
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(d.as_static()?))?
        > 3
    {
        return None;
    }
    let bytes = if graph.is_constant_initializer(id) {
        ctx.initializer_bytes(graph.initializers.get(&id)?)?
    } else {
        let node = graph.node(value.producer?);
        if node.op_type != "Constant"
            || !onnx_runtime_ir::is_default_domain(&node.domain)
            || !node.overload.is_empty()
            || !node.inputs.is_empty()
            || node.attributes.len() != 1
        {
            return None;
        }
        match node.attr("value") {
            Some(Attribute::Tensor(t)) if t.dtype == value.dtype => t.data.as_slice(),
            _ => match (node.attr("value_int"), node.attr("value_ints")) {
                (Some(Attribute::Int(i)), None) if value.dtype == DataType::Int64 => {
                    return Some(vec![*i]);
                }
                (None, Some(Attribute::Ints(i))) if value.dtype == DataType::Int64 => {
                    return Some(i.clone());
                }
                _ => return None,
            },
        }
    };
    match value.dtype {
        DataType::Uint8 => Some(bytes.iter().map(|&v| i64::from(v)).collect()),
        DataType::Int64 if bytes.len().is_multiple_of(8) => Some(
            bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|x| i64::from_le_bytes(*x))
                .collect(),
        ),
        _ => None,
    }
}

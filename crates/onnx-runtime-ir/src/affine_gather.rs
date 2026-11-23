//! Recognition of a row gather that can dequantize only the selected elements.

use crate::{DataType, Graph, Node, NodeId, RUNTIME_DOMAIN, ValueId, is_default_domain};

pub const OP: &str = "AffineQuantizedGather";

/// Restrict this path to the common native DQ contract: scalar FLOAT scale,
/// scalar optional zero point, and INT8/UINT8 input. Moving per-tensor DQ
/// across Gather does not change the scale/zero-point coordinate.
pub fn recognize(
    graph: &Graph,
    gather: &Node,
    captured: &impl Fn(ValueId) -> bool,
) -> Result<(NodeId, Node), &'static str> {
    if gather.op_type != "Gather"
        || !is_default_domain(&gather.domain)
        || !gather.overload.is_empty()
        || gather.outputs.len() != 1
        || gather.attributes.keys().any(|key| key != "axis")
    {
        return Err("not a standard single-output Gather");
    }
    let [Some(data), Some(indices)] = gather.inputs.as_slice() else {
        return Err("Gather requires data and indices");
    };
    let data_value = graph.value(*data);
    let rank = data_value.rank().ok_or("unknown data rank")?;
    let axis = match gather.attr("axis") {
        None => 0,
        Some(a) => a.as_int().ok_or("Gather axis is not an integer")?,
    };
    if rank == 0 || !(-(rank as i64)..rank as i64).contains(&axis) {
        return Err("Gather axis is out of range");
    }
    if graph.outputs.contains(data) || data_value.consumers.len() != 1 || captured(*data) {
        return Err("decoded data has other uses");
    }
    let dq_id = data_value.producer.ok_or("Gather data has no producer")?;
    let dq = graph.node(dq_id);
    if dq.op_type != "DequantizeLinear"
        || !is_default_domain(&dq.domain)
        || !dq.overload.is_empty()
        || dq.outputs.as_slice() != [Some(*data)]
        || !matches!(graph.effective_opset(dq), Some(13..=29))
        || dq
            .attributes
            .iter()
            .any(|(key, attr)| key != "axis" || attr.as_int().is_none())
    {
        return Err("not a supported standard DequantizeLinear");
    }
    let (codes, scale, zero) = match dq.inputs.as_slice() {
        [Some(c), Some(s)] => (*c, *s, None),
        [Some(c), Some(s), Some(z)] => (*c, *s, Some(*z)),
        _ => return Err("DequantizeLinear requires codes and scale, with an optional zero point"),
    };
    let codes_value = graph.value(codes);
    let scale_value = graph.value(scale);
    if !matches!(codes_value.dtype, DataType::Int8 | DataType::Uint8)
        || codes_value.tensor_shape() != data_value.tensor_shape()
        || scale_value.dtype != DataType::Float32
        || scale_value.rank() != Some(0)
        || data_value.dtype != DataType::Float32
        || !matches!(
            graph.value(*indices).dtype,
            DataType::Int32 | DataType::Int64
        )
        || zero.is_some_and(|id| {
            let z = graph.value(id);
            z.dtype != codes_value.dtype || z.rank() != Some(0)
        })
    {
        return Err(
            "requires matching INT8/UINT8 codes and zero point, scalar FLOAT scale, and integer indices",
        );
    }
    let mut fused = gather.clone();
    fused.domain = RUNTIME_DOMAIN.into();
    fused.op_type = OP.into();
    fused.version = Some(1);
    fused.inputs = vec![Some(codes), Some(*indices), Some(scale)];
    if let Some(z) = zero {
        fused.inputs.push(Some(z));
    }
    Ok((dq_id, fused))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::static_shape;

    #[test]
    fn declines_captured_and_shared_decode_and_non_scalar_scale() {
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 29);
        let w = g.create_named_value("w", DataType::Uint8, static_shape([3, 4]));
        let s = g.create_named_value("s", DataType::Float32, vec![]);
        let i = g.create_named_value("i", DataType::Int32, vec![]);
        let decoded = g.create_named_value("decoded", DataType::Float32, static_shape([3, 4]));
        let y = g.create_named_value("y", DataType::Float32, static_shape([4]));
        for id in [w, s, i] {
            g.add_input(id);
        }
        g.insert_node(Node::new(
            NodeId(0),
            "DequantizeLinear",
            vec![Some(w), Some(s)],
            vec![decoded],
        ));
        let root = g.insert_node(Node::new(
            NodeId(0),
            "Gather",
            vec![Some(decoded), Some(i)],
            vec![y],
        ));
        g.add_output(y);
        assert!(recognize(&g, g.node(root), &|_| false).is_ok());
        assert_eq!(
            recognize(&g, g.node(root), &|v| v == decoded).unwrap_err(),
            "decoded data has other uses",
        );
        let mut per_axis = g.clone();
        per_axis.value_mut(s).shape = static_shape([4]);
        assert!(recognize(&per_axis, per_axis.node(root), &|_| false).is_err());
        let y2 = g.create_named_value("y2", DataType::Float32, static_shape([4]));
        g.insert_node(Node::new(
            NodeId(0),
            "Gather",
            vec![Some(decoded), Some(i)],
            vec![y2],
        ));
        assert_eq!(
            recognize(&g, g.node(root), &|_| false).unwrap_err(),
            "decoded data has other uses",
        );
    }
}

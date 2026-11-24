//! Recognition and runtime geometry for `DequantizePacked -> Gather(axis=0)`.

use crate::dequantize_packed::Metadata;
use crate::{DataType, Graph, Node, NodeId, RUNTIME_DOMAIN, ValueId, is_default_domain};

pub const OP: &str = "DequantizePackedGather";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Layout {
    pub rows: usize,
    pub row_elements: usize,
    pub indices: usize,
    pub lanes: usize,
}

pub fn layout(
    metadata: &Metadata,
    packed_dtype: DataType,
    packed_shape: &[usize],
    indices_dtype: DataType,
    indices_shape: &[usize],
    output_dtype: DataType,
    output_shape: &[usize],
) -> Result<Layout, String> {
    metadata.validate_input(packed_dtype, packed_shape)?;
    if !matches!(indices_dtype, DataType::Int32 | DataType::Int64) {
        return Err(format!(
            "indices must be INT32 or INT64, got {indices_dtype:?}"
        ));
    }
    let (&rows, tail) = metadata
        .shape()
        .split_first()
        .ok_or("decoded data must have rank at least one")?;
    let product = |shape: &[usize], what: &str| {
        shape
            .iter()
            .try_fold(1usize, |n, &dim| n.checked_mul(dim))
            .ok_or_else(|| format!("{what} shape {shape:?} overflows host indexing"))
    };
    let row_elements = product(tail, "decoded row")?;
    let indices = product(indices_shape, "indices")?;
    let lanes = indices
        .checked_mul(row_elements)
        .ok_or("output element count overflows host indexing")?;
    let expected_shape: Vec<_> = indices_shape.iter().chain(tail.iter()).copied().collect();
    if output_dtype != metadata.dtype() || output_shape != expected_shape {
        return Err(format!(
            "output is {output_dtype:?}{output_shape:?}; expected {:?}{expected_shape:?}",
            metadata.dtype()
        ));
    }
    if lanes != product(output_shape, "output")? {
        return Err("output element count does not match Gather geometry".into());
    }
    let (width, _) = metadata.format().block();
    if row_elements % width != 0 {
        return Err(format!(
            "decoded row has {row_elements} elements, which is not a whole number of \
             {}-element {} blocks",
            width,
            metadata.format().name()
        ));
    }
    Ok(Layout {
        rows,
        row_elements,
        indices,
        lanes,
    })
}

pub fn recognize(
    graph: &Graph,
    gather: &Node,
    captured: &impl Fn(ValueId) -> bool,
) -> Result<(NodeId, Node), String> {
    if gather.op_type != "Gather"
        || !is_default_domain(&gather.domain)
        || !gather.overload.is_empty()
        || gather.outputs.len() != 1
        || gather.attributes.keys().any(|key| key != "axis")
    {
        return Err("not a standard single-output Gather".into());
    }
    let [Some(decoded), Some(indices)] = gather.inputs.as_slice() else {
        return Err("Gather requires data and indices".into());
    };
    let decoded_value = graph.value(*decoded);
    let rank = decoded_value
        .rank()
        .ok_or("decoded Gather data rank is unknown")?;
    let axis = gather.attr("axis").map_or(Ok(0), |attr| {
        attr.as_int()
            .ok_or("Gather axis is not an integer".to_string())
    })?;
    if rank == 0 || !(-(rank as i64)..rank as i64).contains(&axis) {
        return Err(format!(
            "Gather axis {axis} is out of range for decoded rank {rank}"
        ));
    }
    if axis.rem_euclid(rank as i64) != 0 {
        return Err(format!(
            "only Gather axis 0 can preserve packed row blocks, got axis {axis}"
        ));
    }
    if graph.outputs.contains(decoded) || decoded_value.consumers.len() != 1 || captured(*decoded) {
        return Err("decoded data has another use, graph output, or subgraph capture".into());
    }

    let dq_id = decoded_value
        .producer
        .ok_or("Gather data has no producer")?;
    let dq = graph.node(dq_id);
    if dq.op_type != "DequantizePacked"
        || !is_default_domain(&dq.domain)
        || !dq.overload.is_empty()
        || dq.inputs.len() != 1
        || dq.inputs[0].is_none()
        || dq.outputs.as_slice() != [Some(*decoded)]
        || !matches!(graph.effective_opset(dq), Some(29..))
        || dq
            .attributes
            .keys()
            .any(|key| !matches!(key.as_str(), "format" | "shape" | "output_dtype"))
    {
        return Err("not a supported standard DequantizePacked-29".into());
    }
    let metadata = Metadata::from_node(dq)?;
    if metadata.shape().is_empty() {
        return Err("decoded data must have rank at least one".into());
    }
    let packed = dq.inputs[0].unwrap();
    let packed_value = graph.value(packed);
    metadata.validate_input_dtype(packed_value.dtype)?;
    if packed_value.rank() != Some(1) {
        return Err("packed input must have rank one".into());
    }
    if let Some(shape) = packed_value.tensor_shape() {
        if let Some(bytes) = shape[0].as_static() {
            metadata.validate_packed_bytes(bytes)?;
        }
    }
    let decoded_shape_matches = decoded_value.tensor_shape().is_some_and(|shape| {
        shape.len() == metadata.shape().len()
            && shape
                .iter()
                .zip(metadata.shape())
                .all(|(actual, &expected)| actual.as_static() == Some(expected))
    });
    if decoded_value.dtype != metadata.dtype()
        || !decoded_shape_matches
        || !matches!(
            graph.value(*indices).dtype,
            DataType::Int32 | DataType::Int64
        )
    {
        return Err(
            "decoded value must match DequantizePacked metadata and indices must be integer".into(),
        );
    }

    let mut fused = gather.clone();
    fused.domain = RUNTIME_DOMAIN.into();
    fused.op_type = OP.into();
    fused.version = Some(1);
    fused.inputs = vec![Some(packed), Some(*indices)];
    fused.attributes = dq.attributes.clone();
    Ok((dq_id, fused))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Attribute, Graph, NodeId, static_shape};

    fn graph() -> (Graph, NodeId, ValueId) {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 29);
        let packed = graph.create_named_value("packed", DataType::Uint8, static_shape([72]));
        let indices = graph.create_named_value("indices", DataType::Int64, static_shape([2]));
        let decoded = graph.create_named_value("decoded", DataType::Float32, static_shape([2, 64]));
        let output = graph.create_named_value("output", DataType::Float32, static_shape([2, 64]));
        graph.add_input(packed);
        graph.add_input(indices);
        let mut dq = Node::new(
            NodeId(0),
            "DequantizePacked",
            vec![Some(packed)],
            vec![decoded],
        );
        dq.attributes
            .insert("format".into(), Attribute::String(b"gguf:nvfp4".to_vec()));
        dq.attributes
            .insert("shape".into(), Attribute::Ints(vec![2, 64]));
        graph.insert_node(dq);
        let gather = graph.insert_node(Node::new(
            NodeId(0),
            "Gather",
            vec![Some(decoded), Some(indices)],
            vec![output],
        ));
        graph.add_output(output);
        (graph, gather, decoded)
    }

    #[test]
    fn recognizes_private_axis_zero_decode() {
        let (graph, gather, _) = graph();
        let (_, fused) = recognize(&graph, graph.node(gather), &|_| false).unwrap();
        assert_eq!(fused.op_type, OP);
        assert_eq!(fused.inputs.len(), 2);
    }

    #[test]
    fn declines_nonzero_axis_and_visible_shared_or_captured_decode() {
        let (mut graph, gather, decoded) = graph();
        graph
            .node_mut(gather)
            .attributes
            .insert("axis".into(), Attribute::Int(1));
        assert!(recognize(&graph, graph.node(gather), &|_| false).is_err());
        graph
            .node_mut(gather)
            .attributes
            .insert("axis".into(), Attribute::Int(0));
        graph.add_output(decoded);
        assert!(recognize(&graph, graph.node(gather), &|_| false).is_err());
        graph.outputs.retain(|&value| value != decoded);
        assert!(recognize(&graph, graph.node(gather), &|v| v == decoded).is_err());
        let extra = graph.create_named_value("extra", DataType::Float32, static_shape([2, 64]));
        graph.insert_node(Node::new(
            NodeId(0),
            "Identity",
            vec![Some(decoded)],
            vec![extra],
        ));
        assert!(recognize(&graph, graph.node(gather), &|_| false).is_err());
    }
}

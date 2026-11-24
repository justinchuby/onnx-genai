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

/// Validate and return `(rows, row_elements)` for an axis-0 packed row gather.
///
/// Each gathered axis-0 slice must address an integral number of self-contained
/// GGUF blocks (`row_elements % width == 0`). Standard `DequantizePacked` only
/// constrains its innermost dimension, so rank-1 `[N]` tensors are valid for
/// standalone `DequantizePacked` (`row_elements == 1`) but cannot be gathered
/// as whole packed blocks along axis 0.
pub fn row_geometry(metadata: &Metadata) -> Result<(usize, usize), String> {
    let (&rows, tail) = metadata
        .shape()
        .split_first()
        .ok_or("decoded data must have rank at least one")?;
    let row_elements = tail
        .iter()
        .try_fold(1usize, |n, &dim| n.checked_mul(dim))
        .ok_or_else(|| format!("decoded row shape {tail:?} overflows host indexing"))?;
    let (width, _) = metadata.format().block();
    if !row_elements.is_multiple_of(width) {
        return Err(format!(
            "decoded row has {row_elements} elements, which is not a whole number of \
             {}-element {} blocks",
            width,
            metadata.format().name()
        ));
    }
    Ok((rows, row_elements))
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
    let (rows, row_elements) = row_geometry(metadata)?;
    let tail = &metadata.shape()[1..];
    let product = |shape: &[usize], what: &str| {
        shape
            .iter()
            .try_fold(1usize, |n, &dim| n.checked_mul(dim))
            .ok_or_else(|| format!("{what} shape {shape:?} overflows host indexing"))
    };
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
        || gather.attributes.keys().any(|key| key != "axis")
    {
        return Err("not a standard single-output Gather".into());
    }
    let [Some(output)] = gather.outputs.as_slice() else {
        return Err("not a standard single-output Gather".into());
    };
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
    row_geometry(&metadata)?;
    let packed = dq.inputs[0].unwrap();
    let packed_value = graph.value(packed);
    let packed_dtype = packed_value
        .tensor_dtype()
        .ok_or("packed input dtype is unknown")?;
    metadata.validate_input_dtype(packed_dtype)?;
    if packed_value.rank() != Some(1) {
        return Err("packed input must have rank one".into());
    }
    if let Some(shape) = packed_value.tensor_shape()
        && let Some(bytes) = shape[0].as_static()
    {
        metadata.validate_packed_bytes(bytes)?;
    }
    let decoded_shape_matches = decoded_value.tensor_shape().is_some_and(|shape| {
        shape.len() == metadata.shape().len()
            && shape
                .iter()
                .zip(metadata.shape())
                .all(|(actual, &expected)| actual.as_static() == Some(expected))
    });
    if decoded_value.tensor_dtype() != Some(metadata.dtype())
        || !decoded_shape_matches
        || !matches!(
            graph.value(*indices).tensor_dtype(),
            Some(DataType::Int32 | DataType::Int64)
        )
        || graph
            .value(*output)
            .tensor_dtype()
            .is_some_and(|dtype| dtype != metadata.dtype())
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

    #[test]
    fn declines_rank_one_and_overflowing_zero_row_decode() {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 29);
        let packed = graph.create_named_value("packed", DataType::Uint8, static_shape([36]));
        let indices = graph.create_named_value("indices", DataType::Int64, static_shape([2]));
        let decoded = graph.create_named_value("decoded", DataType::Float32, static_shape([64]));
        let output = graph.create_named_value("output", DataType::Float32, static_shape([2]));
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
            .insert("shape".into(), Attribute::Ints(vec![64]));
        let dq_id = graph.insert_node(dq);
        let gather = graph.insert_node(Node::new(
            NodeId(0),
            "Gather",
            vec![Some(decoded), Some(indices)],
            vec![output],
        ));
        graph.add_output(output);

        let err = recognize(&graph, graph.node(gather), &|_| false).unwrap_err();
        assert!(err.contains("whole number of 64-element"), "{err}");

        let big = 1usize << 30;
        let big_attr = 1i64 << 30;
        graph
            .value_mut(packed)
            .set_tensor_facts(DataType::Uint8, static_shape([0]));
        graph
            .value_mut(decoded)
            .set_tensor_facts(DataType::Float32, static_shape([0, big, big, 64]));
        graph
            .value_mut(output)
            .set_tensor_facts(DataType::Float32, static_shape([2, big, big, 64]));
        graph.node_mut(dq_id).attributes.insert(
            "shape".into(),
            Attribute::Ints(vec![0, big_attr, big_attr, 64]),
        );
        let err = recognize(&graph, graph.node(gather), &|_| false).unwrap_err();
        assert!(err.contains("overflows host indexing"), "{err}");
    }

    #[test]
    fn declines_missing_or_mismatched_gather_output() {
        let (mut graph, gather, _) = graph();
        let output = graph.node(gather).outputs[0].unwrap();
        graph
            .value_mut(output)
            .set_tensor_facts(DataType::Float16, static_shape([2, 64]));
        assert!(recognize(&graph, graph.node(gather), &|_| false).is_err());

        graph
            .value_mut(output)
            .set_tensor_facts(DataType::Float32, static_shape([2, 64]));
        graph.node_mut(gather).outputs = vec![None];
        assert!(recognize(&graph, graph.node(gather), &|_| false).is_err());
    }
}

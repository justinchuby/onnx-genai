//! Shared operator names and GGUF grouped-projection eligibility.

use crate::{Attribute, DataType, Graph, Node, RUNTIME_DOMAIN, as_static_shape, is_default_domain};

pub const OP: &str = "AffineQuantizedGroupedMatMul";
pub const PACKED_OP: &str = "PackedGroupedMatMul";

/// Validate a direct GGUF producer. The caller must prove that the decoded
/// bank does not escape the region before removing the producer.
pub fn recognize_packed(
    g: &Graph,
    root: &Node,
) -> Result<(Vec<crate::NodeId>, Node), &'static str> {
    let [Some(a), Some(w), Some(idx)] = root.inputs.as_slice() else {
        return Err("expected unbiased grouped projection");
    };
    let [Some(y)] = root.outputs.as_slice() else {
        return Err("expected one grouped projection output");
    };
    if root.op_type != "GroupedMatMul"
        || !is_default_domain(&root.domain)
        || !root.overload.is_empty()
        || g.effective_opset(root) != Some(29)
        || root.attributes.len() != 1
        || root.attr("transB").and_then(Attribute::as_int) != Some(1)
        || g.value(*a).dtype != DataType::Float32
        || g.value(*y).dtype != DataType::Float32
        || !matches!(g.value(*idx).dtype, DataType::Int32 | DataType::Int64)
    {
        return Err("expected transposed FLOAT GroupedMatMul with integer routes and no bias");
    }
    let dq = g
        .value(*w)
        .producer
        .map(|id| g.node(id))
        .ok_or("missing packed producer")?;
    if dq.op_type != "DequantizePacked"
        || !is_default_domain(&dq.domain)
        || !dq.overload.is_empty()
        || dq.outputs.as_slice() != [Some(*w)]
        || g.effective_opset(dq) != Some(29)
        || !dq
            .attributes
            .keys()
            .all(|name| matches!(name.as_str(), "format" | "shape" | "output_dtype"))
    {
        return Err("expected standard DequantizePacked-29 attributes");
    }
    let [Some(packed)] = dq.inputs.as_slice() else {
        return Err("DequantizePacked requires one packed input");
    };
    let metadata = crate::dequantize_packed::Metadata::from_node(dq)
        .map_err(|_| "invalid DequantizePacked metadata")?;
    let [groups, n, k] = *metadata.shape() else {
        return Err("decoded packed expert bank must have shape [groups,N,K]");
    };
    let shape = |id| {
        g.value(id)
            .tensor_shape()
            .and_then(|dims| as_static_shape(dims))
    };
    let Some([m, depth]) = shape(*a).and_then(|dims| dims.try_into().ok()) else {
        return Err("expected static grouped activation shape [M,K]");
    };
    let Some([route_rows, lanes]) = shape(*idx).and_then(|dims| dims.try_into().ok()) else {
        return Err("expected static grouped route shape [M,lanes]");
    };
    if groups == 0
        || n == 0
        || k == 0
        || !k.is_multiple_of(metadata.format().block().0)
        || m == 0
        || depth != k
        || route_rows != m
        || metadata.dtype() != DataType::Float32
        || g.value(*w).dtype != DataType::Float32
        || shape(*w).as_deref() != Some(metadata.shape())
        || g.value(*packed).dtype != DataType::Uint8
        || shape(*packed).as_deref() != Some(&[metadata.packed_bytes()])
        || shape(*y).as_deref() != Some(&[m, lanes, n])
    {
        return Err("packed expert bank or grouped projection shapes are incompatible");
    }
    let mut replacement = root.clone();
    replacement.domain = RUNTIME_DOMAIN.into();
    replacement.op_type = PACKED_OP.into();
    replacement.version = Some(1);
    replacement.inputs = vec![Some(*a), Some(*packed), Some(*idx)];
    replacement.attributes = dq.attributes.clone();
    Ok((vec![dq.id], replacement))
}

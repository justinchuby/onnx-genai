//! Recognize packed unsigned nibbles feeding a blocked expert projection.

use crate::{
    Attribute, DataType, Graph, Node, NodeId, RUNTIME_DOMAIN, ValueId, is_default_domain,
    packed_routed,
};

pub const OP: &str = "AffineQuantizedGroupedMatMul";

pub fn recognize(
    g: &Graph,
    root: &Node,
    literal: &impl Fn(ValueId) -> Option<Vec<i64>>,
    captured: &impl Fn(ValueId) -> bool,
) -> Result<(Vec<NodeId>, Node), &'static str> {
    let [Some(a), Some(w), Some(idx)] = root.inputs.as_slice() else {
        return Err("expected unbiassed grouped projection");
    };
    if root.op_type != "GroupedMatMul"
        || !is_default_domain(&root.domain)
        || !root.overload.is_empty()
        || g.effective_opset(root) != Some(29)
        || root.attributes.len() != 1
        || root.attr("transB").and_then(Attribute::as_int) != Some(1)
        || !matches!(root.outputs.as_slice(), [Some(y)] if g.value(*y).dtype == DataType::Float32)
        || g.value(*a).dtype != DataType::Float32
        || !matches!(g.value(*idx).dtype, DataType::Int32 | DataType::Int64)
        || g.value(*a).rank() != Some(2)
        || g.value(*idx).rank() != Some(2)
    {
        return Err("expected transposed FLOAT GroupedMatMul with integer routes and no bias");
    }
    let dq = standard_producer(g, *w, "DequantizeLinear")?;
    let (codes, scale, zero) = match dq.inputs.as_slice() {
        [Some(c), Some(s)] | [Some(c), Some(s), None] => (*c, *s, None),
        [Some(c), Some(s), Some(z)] => (*c, *s, Some(*z)),
        _ => return Err("unexpected dequantize operands"),
    };
    let shape = |id| {
        g.value(id)
            .tensor_shape()
            .and_then(|s| s.iter().map(|d| d.as_static()).collect::<Option<Vec<_>>>())
    };
    let bank = shape(*w).ok_or("unknown expert bank shape")?;
    let [e, n, k] = bank.as_slice() else {
        return Err("expected rank-three bank");
    };
    let block = dq
        .attr("block_size")
        .and_then(Attribute::as_int)
        .and_then(|v| usize::try_from(v).ok())
        .filter(|&v| v > 0)
        .ok_or("missing block size")?;
    let blocks = k.div_ceil(block);
    if !matches!(g.effective_opset(dq), Some(21..=29))
        || dq.attributes.len() != 2
        || !matches!(dq.attr("axis").and_then(Attribute::as_int), Some(2 | -1))
        || *e == 0
        || *n == 0
        || *k == 0
        || !k.is_multiple_of(block)
        || g.value(*w).dtype != DataType::Float32
        || g.value(scale).dtype != DataType::Float32
        || shape(scale).as_deref() != Some(&[*e, *n, blocks])
        || shape(codes).as_deref() != Some(bank.as_slice())
    {
        return Err("requires static, row-wise blocked affine weights and FLOAT scales");
    }
    let strip_cast = |id| -> Result<(ValueId, Option<NodeId>), &'static str> {
        if g.value(id).dtype == DataType::Uint8 {
            return Ok((id, None));
        }
        let cast = standard_producer(g, id, "Cast")?;
        if g.value(id).dtype != DataType::Uint4
            || cast.attributes.len() != 1
            || cast.attr("to").and_then(Attribute::as_int) != Some(21)
        {
            return Err("expected unsigned four-bit cast");
        }
        match cast.inputs.as_slice() {
            [Some(input)] if shape(*input) == shape(id) => Ok((*input, Some(cast.id))),
            _ => Err("unexpected nibble cast input"),
        }
    };
    let (unpacked, cast) = strip_cast(codes)?;
    let (packed, mut remove) = nibble_source(g, unpacked, *k, literal)?;
    remove.extend(cast);
    remove.push(dq.id);
    let mut implicit_zero = 0;
    let packed_zero = if let Some(z) = zero {
        if g.value(z).dtype != g.value(codes).dtype
            || shape(z).as_deref() != Some(&[*e, *n, blocks])
        {
            return Err("zero points must have the blocked scale geometry and code dtype");
        }
        let (unpacked_z, cast) = strip_cast(z)?;
        remove.extend(cast);
        if let Ok((packed, chain)) = zero_source(g, unpacked_z, blocks, literal) {
            remove.extend(chain);
            Some(packed)
        } else {
            let fill = standard_producer(g, unpacked_z, "ConstantOfShape")?;
            let [Some(dims)] = fill.inputs.as_slice() else {
                return Err("invalid static zero points");
            };
            if fill.attributes.len() != 1
                || !matches!(fill.attr("value"), Some(Attribute::Tensor(t))
                    if t.dtype == DataType::Uint8 && t.dims == [1] && t.data.as_slice() == [8])
                || !(literal(*dims).as_deref() == Some(&[*e as i64, *n as i64, blocks as i64])
                    || standard_producer(g, *dims, "Shape").is_ok_and(|n| {
                        n.inputs.as_slice() == [Some(scale)] && n.attributes.is_empty()
                    }))
            {
                return Err("unsupported zero-point source");
            }
            implicit_zero = 8;
            remove.push(fill.id);
            None
        }
    } else {
        None
    };
    let owned: std::collections::HashSet<_> = remove.iter().copied().chain([root.id]).collect();
    for &id in &remove {
        for output in g.node(id).output_values() {
            if g.outputs.contains(&output)
                || captured(output)
                || g.value(output)
                    .consumers
                    .nodes()
                    .iter()
                    .any(|id| !owned.contains(id))
            {
                return Err("decoded weights or packing intermediates have other uses");
            }
        }
    }
    let mut replacement = root.clone();
    replacement.domain = RUNTIME_DOMAIN.into();
    replacement.op_type = OP.into();
    replacement.version = Some(1);
    replacement.attributes.clear();
    replacement
        .attributes
        .insert("depth".into(), Attribute::Int(*k as i64));
    replacement
        .attributes
        .insert("block_size".into(), Attribute::Int(block as i64));
    replacement
        .attributes
        .insert("zero".into(), Attribute::Int(implicit_zero));
    replacement.inputs = vec![Some(*a), Some(packed), Some(*idx), Some(scale)];
    if let Some(z) = packed_zero {
        replacement.inputs.push(Some(z));
    }
    Ok((remove, replacement))
}

fn zero_source(
    g: &Graph,
    v: ValueId,
    blocks: usize,
    literal: &impl Fn(ValueId) -> Option<Vec<i64>>,
) -> Result<(ValueId, Vec<NodeId>), &'static str> {
    if let Ok(slice) = standard_producer(g, v, "Slice") {
        let [Some(data), Some(start), Some(end), Some(axis)] = slice.inputs.as_slice() else {
            return Err("unsupported zero-point Slice inputs");
        };
        let exact = |id, val| {
            g.value(id).dtype == DataType::Int64
                && g.value(id).tensor_shape() == Some(&crate::static_shape([1]))
                && literal(id).as_deref() == Some(&[val])
        };
        let before = g
            .value(*data)
            .tensor_shape()
            .ok_or("unknown Slice input geometry")?;
        let after = g
            .value(v)
            .tensor_shape()
            .ok_or("unknown Slice output geometry")?;
        let padded = blocks
            .div_ceil(2)
            .checked_mul(2)
            .ok_or("block count overflows")?;
        if !slice.attributes.is_empty()
            || before.len() != 3
            || after.len() != 3
            || before[..2] != after[..2]
            || !exact(*start, 0)
            || !exact(*end, blocks as i64)
            || !exact(*axis, 2)
        {
            return Err("Slice must remove only the optional trailing padding nibble in each row");
        }
        let (source, mut dead) = nibble_source(g, *data, padded, literal)?;
        dead.push(slice.id);
        Ok((source, dead))
    } else {
        nibble_source(g, v, blocks, literal)
    }
}

/// Recover a packed UINT8 source for an unsigned, low-nibble-first tensor.
/// The returned node set can be erased only after proving every intermediate
/// output is private to the fusion. The logical count is per row, not global.
pub fn nibble_source(
    g: &Graph,
    output: ValueId,
    count: usize,
    literal: &impl Fn(ValueId) -> Option<Vec<i64>>,
) -> Result<(ValueId, Vec<NodeId>), &'static str> {
    let node = |v, op| standard_producer(g, v, op);
    let input = |n: &Node, i: usize| n.inputs.get(i).copied().flatten().ok_or("missing input");
    let one_attr = |n: &Node, attr: &str, value| {
        n.attributes.len() == 1 && n.attr(attr).and_then(|a| a.as_int()) == Some(value)
    };
    let mut nodes = Vec::new();
    let source = if let Ok(unpack) = node(output, "Unpack") {
        let packed_routed::Op::Unpack(p) =
            packed_routed::Op::from_node(unpack).map_err(|_| "invalid Unpack packing")?
        else {
            return Err("not an Unpack");
        };
        if g.effective_opset(unpack) != Some(29)
            || unpack.inputs.len() != 2
            || unpack
                .attributes
                .keys()
                .any(|k| !matches!(k.as_str(), "bits" | "radix" | "group_size" | "axis"))
            || p.radix != 16
            || p.group != 1
            || p.axis(3).ok() != Some(2)
            || literal(input(unpack, 1)?).as_deref() != Some(&[count as i64])
            || g.value(input(unpack, 1)?).rank() != Some(0)
            || g.value(input(unpack, 1)?).dtype != DataType::Int64
        {
            return Err("expected last-axis Unpack with a scalar logical nibble count");
        }
        nodes.push(unpack.id);
        input(unpack, 0)?
    } else {
        let reshape = node(output, "Reshape")?;
        let concat = node(input(reshape, 0)?, "Concat")?;
        let low_axis = node(input(concat, 0)?, "Unsqueeze")?;
        let high_axis = node(input(concat, 1)?, "Unsqueeze")?;
        let low = node(input(low_axis, 0)?, "BitwiseAnd")?;
        let high = node(input(high_axis, 0)?, "BitShift")?;
        let source = input(low, 0)?;
        let out_shape = g
            .value(output)
            .tensor_shape()
            .ok_or("unknown output shape")?;
        let [experts, rows, width] = out_shape.as_slice() else {
            return Err("expected rank-three nibble output");
        };
        let dims = [
            experts.as_static().ok_or("unknown expert count")? as i64,
            rows.as_static().ok_or("unknown row count")? as i64,
            width.as_static().ok_or("unknown row width")? as i64,
        ];
        let scalar_byte = |v, expected| {
            g.value(v).dtype == DataType::Uint8
                && g.value(v).rank() == Some(0)
                && literal(v).as_deref() == Some(&[expected])
        };
        let last_axis = |n: &Node| -> Result<bool, &'static str> {
            let v = input(n, 1)?;
            Ok(n.inputs.len() == 2
                && n.attributes.is_empty()
                && g.value(v).dtype == DataType::Int64
                && g.value(v).tensor_shape() == Some(&crate::static_shape([1]))
                && literal(v).as_deref() == Some(&[-1]))
        };
        let shape_val = input(reshape, 1)?;
        let reshape_dims = literal(shape_val).ok_or("dynamic reshape target")?;
        if reshape.inputs.len() != 2
            || !reshape.attributes.is_empty()
            || dims[2] != count as i64
            || !count.is_multiple_of(2)
            || g.value(shape_val).dtype != DataType::Int64
            || g.value(shape_val).tensor_shape() != Some(&crate::static_shape([3]))
            || (reshape_dims != dims && reshape_dims != [dims[0], dims[1], -1])
            || concat.inputs.len() != 2
            || !one_attr(concat, "axis", -1)
            || !last_axis(low_axis)?
            || !last_axis(high_axis)?
            || low.inputs.len() != 2
            || !low.attributes.is_empty()
            || high.inputs.len() != 2
            || high.attributes.len() != 1
            || high.attr("direction").and_then(|a| a.as_str()) != Some("RIGHT")
            || input(high, 0)? != source
            || !scalar_byte(input(low, 1)?, 15)
            || !scalar_byte(input(high, 1)?, 4)
        {
            return Err("not a low-nibble-then-high-nibble expansion");
        }
        nodes.extend([
            reshape.id,
            concat.id,
            low_axis.id,
            high_axis.id,
            low.id,
            high.id,
        ]);
        source
    };
    let packed = g.value(source);
    let packed_shape = packed.tensor_shape().ok_or("unknown packed shape")?;
    let decoded_shape = g
        .value(output)
        .tensor_shape()
        .ok_or("unknown nibble shape")?;
    if packed.dtype != DataType::Uint8
        || g.value(output).dtype != DataType::Uint8
        || packed_shape.len() != 3
        || decoded_shape.len() != 3
        || packed_shape[..2] != decoded_shape[..2]
        || decoded_shape[2].as_static() != Some(count)
        || packed_shape[2].as_static() != Some(count.div_ceil(2))
    {
        return Err("packed rows do not match the logical nibble geometry");
    }
    Ok((source, nodes))
}

fn standard_producer<'g>(g: &'g Graph, v: ValueId, op: &str) -> Result<&'g Node, &'static str> {
    let id = g.value(v).producer.ok_or("value has no producer")?;
    let n = g.node(id);
    if n.op_type != op
        || !is_default_domain(&n.domain)
        || !n.overload.is_empty()
        || n.outputs.as_slice() != [Some(v)]
    {
        return Err("unexpected producer");
    }
    Ok(n)
}

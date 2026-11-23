use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, RUNTIME_DOMAIN, ValueId, is_default_domain,
    packed_routed, quantized_grouped::OP, static_shape,
};

use super::Rejection;

/// Validate the operators and operands selected by the affine topology rule.
pub(super) fn replacement(
    g: &Graph,
    root: &Node,
    literal: &impl Fn(ValueId) -> Option<Vec<i64>>,
    decode: &impl Fn(ValueId, &mut Vec<NodeId>) -> Result<(), Rejection>,
    region: &mut Vec<NodeId>,
) -> Result<Node, Rejection> {
    let [Some(a), Some(w), Some(idx)] = root.inputs.as_slice() else {
        return Err(Rejection::Structure {
            value: None,
            reason: "expected three present GroupedMatMul operands without bias",
        });
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
        return Err(
            "expected opset-29 transposed FLOAT GroupedMatMul with integer routes and no bias"
                .into(),
        );
    }
    let dq = standard_producer(g, *w, "DequantizeLinear").map_err(|_| Rejection::Structure {
        value: Some(*w),
        reason: "expected one standard-domain DequantizeLinear producer for grouped weights",
    })?;
    region.push(dq.id);
    let (codes, scale, zero) = match dq.inputs.as_slice() {
        [Some(c), Some(s)] | [Some(c), Some(s), None] => (*c, *s, None),
        [Some(c), Some(s), Some(z)] => (*c, *s, Some(*z)),
        _ => return Err("unexpected dequantize operands".into()),
    };
    let shape = |id| {
        g.value(id)
            .tensor_shape()
            .and_then(|s| s.iter().map(|d| d.as_static()).collect::<Option<Vec<_>>>())
    };
    let bank = shape(*w).ok_or("unknown expert bank shape")?;
    let [e, n, k] = bank.as_slice() else {
        return Err("expected rank-three bank".into());
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
        return Err("requires opset-21–29 DequantizeLinear with static, row-wise blocked affine weights and FLOAT scales".into());
    }
    let strip_cast = |id, region: &mut Vec<NodeId>| -> Result<ValueId, &'static str> {
        if g.value(id).dtype == DataType::Uint8 {
            return Ok(id);
        }
        let cast = standard_producer(g, id, "Cast")?;
        if g.value(id).dtype != DataType::Uint4
            || g.effective_opset(cast) != Some(29)
            || cast.attributes.len() != 1
            || cast.attr("to").and_then(Attribute::as_int) != Some(21)
        {
            return Err("expected unsigned four-bit cast");
        }
        match cast.inputs.as_slice() {
            [Some(input)] if shape(*input) == shape(id) => {
                region.push(cast.id);
                Ok(*input)
            }
            _ => Err("unexpected nibble cast input"),
        }
    };
    let unpacked = strip_cast(codes, region)?;
    decode(unpacked, region)?;
    let packed = nibble_source(g, unpacked, *k, literal)?;
    let mut implicit_zero = 0;
    let packed_zero = if let Some(z) = zero {
        if g.value(z).dtype != g.value(codes).dtype
            || shape(z).as_deref() != Some(&[*e, *n, blocks])
        {
            return Err("zero points must have the blocked scale geometry and code dtype".into());
        }
        let unpacked_z = strip_cast(z, region)?;
        if let Ok(fill) = standard_producer(g, unpacked_z, "ConstantOfShape") {
            let [Some(dims)] = fill.inputs.as_slice() else {
                return Err("invalid static zero points".into());
            };
            if g.effective_opset(fill) != Some(29)
                || fill.attributes.len() != 1
                || !matches!(fill.attr("value"), Some(Attribute::Tensor(t))
                    if t.dtype == DataType::Uint8 && t.dims == [1] && t.data.as_slice() == [8])
                || !(literal(*dims).as_deref() == Some(&[*e as i64, *n as i64, blocks as i64])
                    || standard_producer(g, *dims, "Shape").is_ok_and(|n| {
                        g.effective_opset(n) == Some(29)
                            && n.inputs.as_slice() == [Some(scale)]
                            && n.attributes.is_empty()
                    }))
            {
                return Err("unsupported zero-point source".into());
            }
            implicit_zero = 8;
            region.push(fill.id);
            None
        } else {
            Some(zero_source(g, unpacked_z, blocks, literal, decode, region)?)
        }
    } else {
        None
    };
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
    Ok(replacement)
}

fn zero_source(
    g: &Graph,
    v: ValueId,
    blocks: usize,
    literal: &impl Fn(ValueId) -> Option<Vec<i64>>,
    decode: &impl Fn(ValueId, &mut Vec<NodeId>) -> Result<(), Rejection>,
    region: &mut Vec<NodeId>,
) -> Result<ValueId, Rejection> {
    if let Ok(slice) = standard_producer(g, v, "Slice") {
        let [Some(data), Some(start), Some(end), Some(axis)] = slice.inputs.as_slice() else {
            return Err("unsupported zero-point Slice inputs".into());
        };
        let exact = |id, val| {
            g.value(id).dtype == DataType::Int64
                && g.value(id).tensor_shape() == Some(&static_shape([1]))
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
        if g.effective_opset(slice) != Some(29)
            || !slice.attributes.is_empty()
            || before.len() != 3
            || after.len() != 3
            || before[..2] != after[..2]
            || !exact(*start, 0)
            || !exact(*end, blocks as i64)
            || !exact(*axis, 2)
        {
            return Err(
                "Slice must remove only the optional trailing padding nibble in each row".into(),
            );
        }
        region.push(slice.id);
        decode(*data, region)?;
        Ok(nibble_source(g, *data, padded, literal)?)
    } else {
        decode(v, region)?;
        Ok(nibble_source(g, v, blocks, literal)?)
    }
}

/// Recover a packed UINT8 source for an unsigned, low-nibble-first tensor.
/// The logical count is per row, not global.
fn nibble_source(
    g: &Graph,
    output: ValueId,
    count: usize,
    literal: &impl Fn(ValueId) -> Option<Vec<i64>>,
) -> Result<ValueId, &'static str> {
    let node = |v| {
        let n = producer(g, v);
        if g.effective_opset(n) != Some(29) {
            return Err("unpacked codes must use the opset-29 operator forms");
        }
        Ok(n)
    };
    let one_attr = |n: &Node, attr: &str, value| {
        n.attributes.len() == 1 && n.attr(attr).and_then(|a| a.as_int()) == Some(value)
    };
    let decode = node(output)?;
    let source = if decode.op_type == "Unpack" {
        let unpack = decode;
        let packed_routed::Op::Unpack(p) =
            packed_routed::Op::from_node(unpack).map_err(|_| "invalid Unpack packing")?
        else {
            return Err("not an Unpack");
        };
        let count_value = input(unpack, 1);
        if unpack
            .attributes
            .keys()
            .any(|k| !matches!(k.as_str(), "bits" | "radix" | "group_size" | "axis"))
            || p.radix != 16
            || p.group != 1
            || p.axis(3).ok() != Some(2)
            || literal(count_value).as_deref() != Some(&[count as i64])
            || g.value(count_value).rank() != Some(0)
            || g.value(count_value).dtype != DataType::Int64
        {
            return Err("expected last-axis Unpack with a scalar logical nibble count");
        }
        input(unpack, 0)
    } else {
        let reshape = decode;
        let concat = node(input(reshape, 0))?;
        let low_axis = node(input(concat, 0))?;
        let high_axis = node(input(concat, 1))?;
        let low = node(input(low_axis, 0))?;
        let high = node(input(high_axis, 0))?;
        let source = input(low, 0);
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
        let last_axis = |n: &Node| {
            let v = input(n, 1);
            n.attributes.is_empty()
                && g.value(v).dtype == DataType::Int64
                && g.value(v).tensor_shape() == Some(&static_shape([1]))
                && literal(v).as_deref() == Some(&[-1])
        };
        let shape_val = input(reshape, 1);
        let reshape_dims = literal(shape_val).ok_or("dynamic reshape target")?;
        if !reshape.attributes.is_empty()
            || dims[2] != count as i64
            || !count.is_multiple_of(2)
            || g.value(shape_val).dtype != DataType::Int64
            || g.value(shape_val).tensor_shape() != Some(&static_shape([3]))
            || (reshape_dims != dims && reshape_dims != [dims[0], dims[1], -1])
            || !one_attr(concat, "axis", -1)
            || !last_axis(low_axis)
            || !last_axis(high_axis)
            || !low.attributes.is_empty()
            || high.attributes.len() != 1
            || high.attr("direction").and_then(|a| a.as_str()) != Some("RIGHT")
            || !scalar_byte(input(low, 1), 15)
            || !scalar_byte(input(high, 1), 4)
        {
            return Err("not a low-nibble-then-high-nibble expansion");
        }
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
    Ok(source)
}

fn producer(g: &Graph, v: ValueId) -> &Node {
    g.node(g.value(v).producer.expect("matched operator output"))
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

fn input(n: &Node, slot: usize) -> ValueId {
    n.inputs[slot].expect("matched operand")
}

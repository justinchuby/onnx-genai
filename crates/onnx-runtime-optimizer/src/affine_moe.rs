//! Full MoE region over the affine projections recognized by QuantizedGroupedFusion.

use onnx_runtime_ir::pattern::MatchSession;
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, RUNTIME_DOMAIN, ValueId, as_static_shape,
};

use crate::{OptimizationPass, PassContext, Result, quantized_grouped::literal};

pub const OP: &str = "AffineRoutedMoE";
pub const THRESHOLD: f32 = 1e-8;

pub struct AffineMoeFusion;

impl OptimizationPass for AffineMoeFusion {
    fn name(&self) -> &str {
        "AffineMoeFusion"
    }

    fn run(&self, g: &mut Graph, ctx: &PassContext) -> Result<()> {
        if g.opset_imports
            .get("")
            .or_else(|| g.opset_imports.get("ai.onnx"))
            != Some(&29)
        {
            return Ok(());
        }
        let order = g.topological_order().map_err(|error| vec![error])?;
        // Reverse traversal keeps the original topological removal order valid.
        // Rebuild the boundary index only after a structural candidate matches.
        for &id in order.iter().rev() {
            let Some(root) = g.try_node(id) else { continue };
            if root.op_type != "Reshape" {
                continue;
            }
            let Ok((region, replacement)) = candidate(g, ctx, root) else {
                continue;
            };
            let out = root.outputs[0].expect("candidate verified its output");
            let mut closed: Vec<_> = order
                .iter()
                .rev()
                .copied()
                .filter(|n| region.contains(n))
                .collect();
            if MatchSession::new(g)?
                .first_escaping_use(&closed, &[out])
                .is_some()
            {
                continue;
            }
            closed.retain(|&n| n != id);
            g.replace_node(id, replacement);
            g.remove_nodes(&closed);
        }
        Ok(())
    }
}

use std::collections::HashSet;

fn candidate(
    g: &Graph,
    ctx: &PassContext,
    root: &Node,
) -> std::result::Result<(HashSet<NodeId>, Node), &'static str> {
    let mut region = HashSet::from([root.id]);
    let mut get = |v, op, attrs: &[(&str, i64)]| {
        let n = g
            .value(v)
            .producer
            .and_then(|id| g.try_node(id))
            .ok_or("missing producer")?;
        require(n, op, attrs)?;
        region.insert(n.id);
        Ok(n)
    };
    let require = |n: &Node, attrs: &[(&str, i64)]| require(n, "Reshape", attrs);
    require(root, &[("allowzero", 1)])?;
    let [Some(out)] = root.outputs.as_slice() else {
        return Err("expected one MoE output");
    };
    let [sum, shape] = inputs(root)?;
    let [masked, reduce_axis] = inputs(get(sum, "ReduceSum", &[("keepdims", 0)])?)?;
    let [active, weighted, zero] = inputs(get(masked, "Where", &[])?)?;
    let [down, routing3] = inputs(get(weighted, "Mul", &[])?)?;
    let [routing, axis3] = inputs(get(routing3, "Unsqueeze", &[])?)?;
    let [active2, mask_axis] = inputs(get(active, "Unsqueeze", &[])?)?;
    let [mask_routing, threshold] = inputs(get(active2, "Greater", &[])?)?;
    let [logits] = inputs(get(routing, "Softmax", &[("axis", 1)])?)?;
    let top = get(logits, "TopK", &[("axis", 1), ("sorted", 1)])?;
    let [Some(vals), Some(indices)] = top.outputs.as_slice() else {
        return Err("missing TopK outputs");
    };
    let [reversed, k_value] = inputs(top)?;
    let [router, rstart, rend, raxis, rstep] = inputs(get(reversed, "Slice", &[])?)?;
    let [proj2, down_shape] = inputs(get(down, "Reshape", &[])?)?;
    let p2 = projection(g, proj2)?;
    let [activated, act_shape] = inputs(get(p2.activation, "Reshape", &[])?)?;
    let [selected, selection_shape] = inputs(get(p2.routes, "Reshape", &[])?)?;
    let [last_expert, top_indices] = inputs(get(selected, "Sub", &[])?)?;
    let activation = g
        .value(activated)
        .producer
        .and_then(|id| g.try_node(id))
        .ok_or("missing SwiGLU")?;
    let Some(Attribute::Float(alpha)) = activation.attr("alpha") else {
        return Err("expected explicit SwiGLU alpha");
    };
    if !alpha.is_finite()
        || activation.attributes.len() != 1
        || activation.op_type != "SwiGLU"
        || !activation.is_default_domain()
        || !activation.overload.is_empty()
        || g.effective_opset(activation) != Some(29)
    {
        return Err("expected standard SwiGLU-29 with finite alpha");
    }
    let [clamped_gate, up_plus_beta] = inputs(activation)?;
    let [gate, max] = inputs(get(clamped_gate, "Min", &[])?)?;
    let [clamped_up, beta] = inputs(get(up_plus_beta, "Add", &[])?)?;
    let [up, min, upper] = inputs(get(clamped_up, "Clip", &[])?)?;
    let [proj1, gstart, gend, gaxis, gstep] = inputs(get(gate, "Slice", &[])?)?;
    let [up_proj, ustart, uend, uaxis, ustep] = inputs(get(up, "Slice", &[])?)?;
    let p1 = projection(g, proj1)?;
    let [x, flat_shape] = inputs(get(p1.activation, "Reshape", &[])?)?;
    let shape_source = g.value(shape).producer.and_then(|id| g.try_node(id));
    if shape_source.is_some_and(|n| n.op_type == "Shape") {
        let [shape_x] = inputs(get(shape, "Shape", &[])?)?;
        if shape_x != x {
            return Err("result shape must come from the input activation");
        }
    }
    region.extend([activation.id, p1.node.id, p2.node.id]);

    let dims = |v| g.value(v).tensor_shape().and_then(|s| as_static_shape(s));
    let xd = dims(x).ok_or("expected static activation geometry")?;
    let &[tokens, experts] = dims(router)
        .as_deref()
        .ok_or("expected static router geometry")?
    else {
        return Err("router must be [tokens, experts]");
    };
    let hidden = p1.depth;
    let intermediate = p2.depth;
    let inter2 = intermediate
        .checked_mul(2)
        .ok_or("intermediate width overflows")?;
    let k = literal(g, ctx, k_value)
        .and_then(|v| usize::try_from(*v.first()?).ok())
        .ok_or("expected static top-k")?;
    if !(2..=3).contains(&xd.len())
        || xd.last() != Some(&hidden)
        || xd[..xd.len() - 1]
            .iter()
            .try_fold(1usize, |a, b| a.checked_mul(*b))
            != Some(tokens)
        || tokens == 0
        || !(1..=experts).contains(&k)
        || dims(*out).as_deref() != Some(xd.as_slice())
        || [x, router, *out]
            .iter()
            .any(|&v| g.value(v).dtype != DataType::Float32)
        || dims(p1.codes).as_deref() != Some(&[experts, inter2, hidden / 2])
        || dims(p2.codes).as_deref() != Some(&[experts, hidden, intermediate / 2])
        || !hidden.is_multiple_of(2)
        || !intermediate.is_multiple_of(2)
    {
        return Err("incompatible FLOAT activation, router or affine expert shapes");
    }
    let eq = |v, expected: &[i64]| {
        g.value(v).dtype == DataType::Int64
            && dims(v).as_deref() == Some(&[expected.len()])
            && literal(g, ctx, v).as_deref() == Some(expected)
    };
    let e = i64::try_from(experts).map_err(|_| "expert count exceeds INT64")?;
    let k = i64::try_from(k).map_err(|_| "top-k exceeds INT64")?;
    let h = i64::try_from(hidden).map_err(|_| "hidden width exceeds INT64")?;
    let i = i64::try_from(intermediate).map_err(|_| "intermediate width exceeds INT64")?;
    let i2 = i64::try_from(inter2).map_err(|_| "interleaved width exceeds INT64")?;
    let expected_shape = xd
        .iter()
        .map(|&d| i64::try_from(d))
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|_| "input dimension exceeds INT64")?;
    if *vals != logits
        || *indices != top_indices
        || mask_routing != routing
        || p1.routes != selected
        || up_proj != proj1
        || (!shape_source.is_some_and(|n| n.op_type == "Shape") && !eq(shape, &expected_shape))
        || ![
            (reduce_axis, &[1][..]),
            (axis3, &[2]),
            (mask_axis, &[2]),
            (k_value, &[k]),
            (rstart, &[e - 1]),
            (rend, &[-e - 1]),
            (raxis, &[1]),
            (rstep, &[-1]),
            (last_expert, &[e - 1]),
            (down_shape, &[-1, k, h]),
            (act_shape, &[-1, i]),
            (selection_shape, &[-1, 1]),
            (gstart, &[0]),
            (gend, &[i2]),
            (gaxis, &[2]),
            (gstep, &[2]),
            (ustart, &[1]),
            (uend, &[i2]),
            (uaxis, &[2]),
            (ustep, &[2]),
            (flat_shape, &[-1, h]),
        ]
        .iter()
        .all(|(v, a)| eq(*v, a))
    {
        return Err("routing, interleaving or flattening do not match the affine MoE region");
    }
    let float = |v| scalar(g, ctx, v);
    let limit = float(max)
        .filter(|v| v.is_finite() && *v > 0.0)
        .ok_or("expected positive finite gate limit")?;
    let beta = float(beta)
        .filter(|v| v.is_finite())
        .ok_or("expected finite SwiGLU beta")?;
    if float(min) != Some(-limit)
        || float(upper) != Some(limit)
        || float(zero).map(f32::to_bits) != Some(0)
        || float(threshold) != Some(THRESHOLD)
    {
        return Err("activation clip or inactive-route masking differs");
    }
    for (p, width) in [(&p1, inter2), (&p2, hidden)] {
        let blocks = p.depth / p.block;
        if g.value(p.scale).dtype != DataType::Float32
            || dims(p.scale).as_deref() != Some(&[experts, width, blocks])
            || p.zeros.is_some_and(|z| {
                g.value(z).dtype != DataType::Uint8
                    || dims(z).as_deref() != Some(&[experts, width, blocks.div_ceil(2)])
            })
        {
            return Err("incompatible affine scale or zero-point shape");
        }
    }
    let mut n = root.clone();
    n.op_type = OP.into();
    n.domain = RUNTIME_DOMAIN.into();
    n.version = Some(1);
    n.inputs = vec![
        Some(x),
        Some(router),
        Some(p1.codes),
        Some(p1.scale),
        p1.zeros,
        Some(p2.codes),
        Some(p2.scale),
        p2.zeros,
    ];
    n.attributes = [
        ("k", Attribute::Int(k)),
        ("alpha", Attribute::Float(*alpha)),
        ("beta", Attribute::Float(beta)),
        ("limit", Attribute::Float(limit)),
        ("first_block", Attribute::Int(p1.block as i64)),
        ("first_zero", Attribute::Int(p1.zero)),
        ("down_block", Attribute::Int(p2.block as i64)),
        ("down_zero", Attribute::Int(p2.zero)),
    ]
    .into_iter()
    .map(|(key, attr)| (key.into(), attr))
    .collect();
    if n.inputs
        .iter()
        .flatten()
        .any(|v| g.value(*v).producer.is_some_and(|id| region.contains(&id)))
    {
        return Err("fused input is produced by a removed node");
    }
    Ok((region, n))
}

fn require(n: &Node, op: &str, attrs: &[(&str, i64)]) -> std::result::Result<(), &'static str> {
    if n.op_type != op
        || !n.is_default_domain()
        || !n.overload.is_empty()
        || n.version.is_some_and(|v| v != 29)
        || n.attributes.len() != attrs.len()
        || attrs
            .iter()
            .any(|(key, val)| n.attr(key).and_then(Attribute::as_int) != Some(*val))
    {
        return Err("unexpected operator, version or attribute in affine MoE");
    }
    Ok(())
}

fn inputs<const N: usize>(n: &Node) -> std::result::Result<[ValueId; N], &'static str> {
    let slots: &[Option<ValueId>; N] = n
        .inputs
        .as_slice()
        .try_into()
        .map_err(|_| "wrong input count")?;
    if slots.iter().any(Option::is_none) {
        return Err("missing input");
    }
    Ok(slots.map(|v| v.expect("all slots present")))
}

struct Projection<'a> {
    node: &'a Node,
    activation: ValueId,
    routes: ValueId,
    codes: ValueId,
    scale: ValueId,
    zeros: Option<ValueId>,
    depth: usize,
    block: usize,
    zero: i64,
}

fn projection(g: &Graph, v: ValueId) -> std::result::Result<Projection<'_>, &'static str> {
    let node = g
        .value(v)
        .producer
        .and_then(|id| g.try_node(id))
        .ok_or("expected affine projection")?;
    if node.domain != RUNTIME_DOMAIN
        || node.op_type != onnx_runtime_ir::quantized_grouped::OP
        || !node.overload.is_empty()
        || g.effective_opset(node) != Some(1)
        || node.outputs.as_slice() != [Some(v)]
        || node.attributes.len() != 3
    {
        return Err("expected one recognized affine grouped projection");
    }
    let (a, c, r, s, zeros) = match node.inputs.as_slice() {
        [Some(a), Some(c), Some(r), Some(s)] => (*a, *c, *r, *s, None),
        [Some(a), Some(c), Some(r), Some(s), Some(z)] => (*a, *c, *r, *s, Some(*z)),
        _ => return Err("malformed affine grouped projection inputs"),
    };
    let attr = |name| node.attr(name).and_then(Attribute::as_int);
    let size = |name| {
        attr(name)
            .and_then(|v| usize::try_from(v).ok())
            .filter(|&v| v != 0)
    };
    let depth = size("depth").ok_or("invalid affine depth")?;
    let block = size("block_size")
        .filter(|b| depth.is_multiple_of(*b))
        .ok_or("invalid affine block")?;
    let zero = attr("zero")
        .filter(|z| (0..=15).contains(z))
        .ok_or("invalid affine zero")?;
    if g.value(c).dtype != DataType::Uint8
        || g.value(r).dtype != DataType::Int64
        || [a, v]
            .iter()
            .any(|&id| g.value(id).dtype != DataType::Float32)
    {
        return Err("unsupported affine projection dtypes");
    }
    Ok(Projection {
        node,
        activation: a,
        routes: r,
        codes: c,
        scale: s,
        zeros,
        depth,
        block,
        zero,
    })
}

fn scalar(g: &Graph, ctx: &PassContext, v: ValueId) -> Option<f32> {
    let value = g.value(v);
    if value.dtype != DataType::Float32 || value.rank() != Some(0) {
        return None;
    }
    if g.is_constant_initializer(v) {
        return Some(f32::from_le_bytes(
            ctx.initializer_bytes(g.initializers.get(&v)?)?
                .try_into()
                .ok()?,
        ));
    }
    let node = g.node(value.producer?);
    if node.op_type == "Constant"
        && node.is_default_domain()
        && node.overload.is_empty()
        && node.attributes.len() == 1
        && node.inputs.is_empty()
    {
        match node.attr("value_float") {
            Some(Attribute::Float(f)) => return Some(*f),
            _ => {
                if let Some(Attribute::Tensor(t)) = node.attr("value")
                    && t.dtype == DataType::Float32
                    && t.dims.is_empty()
                {
                    return Some(f32::from_le_bytes(t.data.as_slice().try_into().ok()?));
                }
            }
        }
    }
    None
}

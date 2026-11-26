//! Bias-free FLOAT split-bank expert dispatch from Mobius's opset-29 grouped lowering.
//! Keep the standard full-axis Softmax materialized: TopK must rank its rounded
//! probabilities, not its logits (distinct logits can round to the same probability).

use super::*;

pub const OP: &str = "AffineSoftmaxFirstMoE";
/// The native combine maps a token's selected routes onto one CUDA warp.
pub const MAX_ROUTES_PER_TOKEN: usize = 32;

pub struct Fusion;

impl OptimizationPass for Fusion {
    fn name(&self) -> &str {
        "AffineSoftmaxFirstMoeFusion"
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
        for &id in order.iter().rev() {
            let Some(root) = g.try_node(id).filter(|n| n.op_type == "ReduceSum") else {
                continue;
            };
            let Ok((region, replacement, out)) = candidate(g, ctx, root) else {
                continue;
            };
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

/// Geometry shared by rewrite admission and CUDA workspace preparation.
/// All products used for element/byte addresses fit on the host and device.
pub fn route_extent(
    tokens: usize,
    k: usize,
    experts: usize,
    hidden: usize,
    inter: usize,
) -> Option<usize> {
    if tokens == 0 || k == 0 || k > experts || k > MAX_ROUTES_PER_TOKEN || hidden == 0 || inter == 0
    {
        return None;
    }
    let routes = tokens.checked_mul(k)?;
    let activation = routes.checked_mul(inter)?;
    let output_bytes = tokens.checked_mul(hidden)?.checked_mul(4)?;
    let bytes = routes
        .checked_mul(12)?
        .checked_add(activation.checked_mul(4)?)?
        .checked_add(output_bytes)?
        .checked_add(4)?;
    for (a, b) in [
        (tokens, experts),
        (experts, hidden.checked_mul(2)?.checked_mul(inter)?),
    ] {
        if a.checked_mul(b)?.checked_mul(4)? > isize::MAX as usize {
            return None;
        }
    }
    let alignment = align_of::<i64>().max(align_of::<f32>());
    (bytes.checked_next_multiple_of(alignment)? <= isize::MAX as usize).then_some(routes)
}

fn producer<'a>(g: &'a Graph, v: ValueId, op: &str) -> std::result::Result<&'a Node, &'static str> {
    let n = g
        .value(v)
        .producer
        .and_then(|id| g.try_node(id))
        .ok_or("missing producer")?;
    if n.op_type != op
        || !n.is_default_domain()
        || !n.overload.is_empty()
        || n.version.is_some_and(|v| v != 29)
    {
        return Err("unexpected operator, domain or version");
    }
    Ok(n)
}

fn axis(n: &Node, rank: i64, target: i64, default: Option<i64>) -> bool {
    let value = match n.attr("axis") {
        Some(a) => a.as_int(),
        None => default,
    };
    matches!(value, Some(a) if a == target || a == target - rank)
}

fn candidate(
    g: &Graph,
    ctx: &PassContext,
    root: &Node,
) -> std::result::Result<(HashSet<NodeId>, Node, ValueId), &'static str> {
    require(root, "ReduceSum", &[("keepdims", 0)])?;
    let [Some(out)] = root.outputs.as_slice() else {
        return Err("expected one combined output");
    };
    let [weighted, sum_axis] = inputs(root)?;
    let dims = |v| g.value(v).tensor_shape().and_then(|s| as_static_shape(s));
    let eq = |v, expected: &[i64]| {
        g.value(v).dtype == DataType::Int64
            && dims(v).as_deref() == Some(&[expected.len()])
            && literal(g, ctx, v).as_deref() == Some(expected)
    };
    let float = |v| g.value(v).dtype == DataType::Float32;
    let mut region = HashSet::from([root.id]);
    let mut get = |v, op| {
        let n = producer(g, v, op)?;
        region.insert(n.id);
        Ok::<_, &'static str>(n)
    };
    let [down, routing3] = inputs(get(weighted, "Mul")?)?;
    let expand = get(routing3, "Unsqueeze")?;
    require(expand, "Unsqueeze", &[])?;
    let [routing, route_axis] = inputs(expand)?;
    let strip_cast = |v, n: &Node| {
        require(n, "Cast", &[("to", 1)])?;
        let [source] = inputs(n)?;
        if !float(v) || !float(source) || dims(v) != dims(source) {
            return Err("routing and activation casts must preserve FLOAT");
        }
        Ok(source)
    };
    let without_cast = |v| {
        if g.value(v)
            .producer
            .and_then(|id| g.try_node(id))
            .is_some_and(|n| n.op_type == "Cast")
        {
            strip_cast(v, producer(g, v, "Cast")?)
        } else {
            Ok(v)
        }
    };
    let routing = if without_cast(routing)? != routing {
        strip_cast(routing, get(routing, "Cast")?)?
    } else {
        routing
    };
    let top = get(routing, "TopK")?;
    let [Some(weights), Some(selected)] = top.outputs.as_slice() else {
        return Err("expected TopK values and indices");
    };
    let default_one = |key| top.attr(key).is_none_or(|a| a.as_int() == Some(1));
    if *weights != routing
        || !axis(top, 2, 1, Some(-1))
        || !default_one("sorted")
        || !default_one("largest")
        || top
            .attributes
            .keys()
            .any(|key| !matches!(key.as_str(), "axis" | "sorted" | "largest"))
    {
        return Err("expected largest, sorted TopK over the full expert axis");
    }
    let [probabilities, kv] = inputs(top)?;
    let &[tokens, experts] = dims(probabilities)
        .as_deref()
        .ok_or("expected static router geometry")?
    else {
        return Err("TopK probabilities must be [tokens, experts]");
    };
    let e = i64::try_from(experts).map_err(|_| "expert count exceeds INT64")?;
    // The pinned export flattens the router and may surround it with no-op FLOAT casts.
    // These nodes remain live, preserving the exact values read by the standard TopK.
    let prob_source = without_cast(probabilities)?;
    let source = if let Ok(reshape) = producer(g, prob_source, "Reshape") {
        require(reshape, "Reshape", &[])?;
        let [source, shape] = inputs(reshape)?;
        if !eq(shape, &[-1, e]) {
            return Err("router reshape must preserve the expert axis");
        }
        without_cast(source)?
    } else {
        prob_source
    };
    let softmax = producer(g, source, "Softmax")?;
    let [logits] = inputs(softmax)?;
    let sd = dims(source).ok_or("expected static Softmax geometry")?;
    let Some((&axis_width, leading)) = sd.split_last() else {
        return Err("missing Softmax expert axis");
    };
    if !(1..=2).contains(&leading.len())
        || axis_width != experts
        || leading.iter().try_fold(1usize, |a, b| a.checked_mul(*b)) != Some(tokens)
        || !axis(softmax, sd.len() as i64, sd.len() as i64 - 1, Some(-1))
        || softmax.attributes.keys().any(|key| key != "axis")
        || !float(logits)
        || !float(source)
        || !float(probabilities)
    {
        return Err("router must use FLOAT Softmax over all experts before selection");
    }
    let down_reshape = get(down, "Reshape")?;
    require(down_reshape, "Reshape", &[])?;
    let [down_proj, down_shape] = inputs(down_reshape)?;
    let p2 = projection(g, down_proj)?;
    let act_reshape = get(p2.activation, "Reshape")?;
    require(act_reshape, "Reshape", &[])?;
    let [act, act_shape] = inputs(act_reshape)?;
    let select_reshape = get(p2.routes, "Reshape")?;
    require(select_reshape, "Reshape", &[])?;
    let [down_ids, select_shape] = inputs(select_reshape)?;

    // Match the explicit FLOAT SiLU arithmetic, including its rounding points.
    let activation = get(act, "Mul")?;
    require(activation, "Mul", &[])?;
    let [silu, up] = inputs(activation)?;
    let silu = if without_cast(silu)? != silu {
        strip_cast(silu, get(silu, "Cast")?)?
    } else {
        silu
    };
    let div = get(silu, "Div")?;
    require(div, "Div", &[])?;
    let [gate32, denom] = inputs(div)?;
    let add = get(denom, "Add")?;
    require(add, "Add", &[])?;
    let [one, exp] = inputs(add)?;
    let exponent = get(exp, "Exp")?;
    require(exponent, "Exp", &[])?;
    let [neg] = inputs(exponent)?;
    let negate = get(neg, "Neg")?;
    require(negate, "Neg", &[])?;
    let [neg_gate] = inputs(negate)?;
    let gate = if without_cast(gate32)? != gate32 {
        strip_cast(gate32, get(gate32, "Cast")?)?
    } else {
        gate32
    };
    let split = get(gate, "Split")?;
    let [Some(gate_out), Some(up_out)] = split.outputs.as_slice() else {
        return Err("expected two Split outputs");
    };
    let [first_proj, split_sizes] = inputs(split)?;
    let p1 = projection(g, first_proj)?;
    let x = p1.activation;
    let h = p1.depth;
    let i = p2.depth;
    let inter2 = i.checked_mul(2).ok_or("split width overflows")?;
    let k = literal(g, ctx, kv)
        .and_then(|v| usize::try_from(*v.first()?).ok())
        .ok_or("expected static TopK K")?;
    let routes = route_extent(tokens, k, experts, h, i)
        .ok_or("invalid or unaddressable CUDA MoE geometry")?;
    let (kh, hh, ih) = (
        i64::try_from(k).map_err(|_| "TopK K exceeds INT64")?,
        i64::try_from(h).map_err(|_| "hidden exceeds INT64")?,
        i64::try_from(i).map_err(|_| "intermediate exceeds INT64")?,
    );
    if !eq(kv, &[kh])
        || !eq(sum_axis, &[1])
        || !eq(route_axis, &[2])
        || !eq(split_sizes, &[ih, ih])
        || !axis(split, 3, 2, None)
        || split.attributes.len() != 1
        || !eq(down_shape, &[-1, kh, hh])
        || !eq(act_shape, &[-1, ih])
        || !eq(select_shape, &[-1, 1])
        || down_ids != *selected
        || p1.routes != *selected
        || *gate_out != gate
        || *up_out != up
        || neg_gate != gate32
        || scalar(g, ctx, one) != Some(1.0)
    {
        return Err("expert dispatch, activation or unmasked combine topology differs");
    }
    for (v, expected) in [
        (x, &[tokens, h][..]),
        (*out, &[tokens, h]),
        (routing, &[tokens, k]),
        (first_proj, &[tokens, k, inter2]),
        (down_proj, &[routes, 1, h]),
        (gate, &[tokens, k, i]),
        (up, &[tokens, k, i]),
        (act, &[tokens, k, i]),
    ] {
        if !float(v) || dims(v).as_deref() != Some(expected) {
            return Err("incompatible FLOAT geometry in affine expert dispatch");
        }
    }
    if dims(*selected).as_deref() != Some(&[tokens, k]) {
        return Err("incompatible TopK expert IDs");
    }
    for (p, width) in [(&p1, inter2), (&p2, h)] {
        let blocks = p.depth / p.block;
        if !p.depth.is_multiple_of(2)
            || dims(p.codes).as_deref() != Some(&[experts, width, p.depth / 2])
            || !float(p.scale)
            || dims(p.scale).as_deref() != Some(&[experts, width, blocks])
            || p.zeros.is_some_and(|z| {
                g.value(z).dtype != DataType::Uint8
                    || dims(z).as_deref() != Some(&[experts, width, blocks.div_ceil(2)])
            })
        {
            return Err("incompatible affine UINT4 bank, scales or zero points");
        }
    }
    region.extend([p1.node.id, p2.node.id]);
    let mut node = root.clone();
    node.op_type = OP.into();
    node.domain = RUNTIME_DOMAIN.into();
    node.version = Some(1);
    node.inputs = vec![
        Some(x),
        Some(probabilities),
        Some(p1.codes),
        Some(p1.scale),
        p1.zeros,
        Some(p2.codes),
        Some(p2.scale),
        p2.zeros,
    ];
    node.attributes = [
        ("k", kh),
        ("first_block", p1.block as i64),
        ("first_zero", p1.zero),
        ("down_block", p2.block as i64),
        ("down_zero", p2.zero),
    ]
    .into_iter()
    .map(|(key, attr)| (key.into(), Attribute::Int(attr)))
    .collect();
    if node
        .inputs
        .iter()
        .flatten()
        .any(|v| g.value(*v).producer.is_some_and(|id| region.contains(&id)))
    {
        return Err("fused input is produced by a removed node");
    }
    Ok((region, node, *out))
}

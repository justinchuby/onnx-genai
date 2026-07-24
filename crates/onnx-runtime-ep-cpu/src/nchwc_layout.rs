//! `CpuNchwcLayoutPropagation`: a graph-level NCHWc layout-propagation pass that
//! mirrors ONNX Runtime's `NchwcTransformer`.
//!
//! The per-op `Conv` kernel reorders NCHW -> NCHWc on its input and back on its
//! output for *every* convolution. Across a CNN backbone that per-op reorder is
//! the dominant residual overhead versus ORT, which transforms the activation to
//! NCHWc *once* at graph entry and keeps the whole backbone blocked, reordering
//! back only at the exit.
//!
//! This pass identifies maximal regions of NCHWc-capable ops (Conv, Max/Average/
//! GlobalAverage pooling, and the layout-agnostic element-wise `Add`, `Relu`,
//! `Clip`) connected by 4-D NCHW tensors, inserts a single
//! `NchwcReorderToBlocked` at each region entry and `NchwcReorderToNchw` at each
//! exit, and rewrites the interior Conv/Pool nodes to the blocked kernels in
//! `kernels::nchwc`. Element-wise interior ops are left as their standard
//! kernels because they operate identically on the blocked buffer.
//!
//! The pass is purely structural and general: it keys only off op type, static
//! 4-D shapes, and blocked-conv eligibility (never model identity). Convs the
//! pass does not fold stay on the per-op NCHWc/im2col fallback in
//! `kernels::conv`.

use std::collections::HashMap;

use onnx_runtime_ir::{Attribute, DataType, Dim, Graph, Node, NodeId, ValueId, as_static_shape};
use onnx_runtime_optimizer::{OptimizationPass, OptimizerError, PassContext, Result};

use crate::kernels::nchwc::{
    NCHWC_AVERAGE_POOL_OP, NCHWC_CONV_OP, NCHWC_DOMAIN, NCHWC_GLOBAL_AVERAGE_POOL_OP,
    NCHWC_MAX_POOL_OP, REORDER_TO_BLOCKED_OP, REORDER_TO_NCHW_OP, round_up,
};

/// Always-on NCHWc layout-propagation pass. Runs after Conv+BN(+Relu) fusion so
/// the interior activation ops it sees are already folded where possible.
pub struct NchwcLayoutPropagation;

impl NchwcLayoutPropagation {
    pub fn new() -> Self {
        Self
    }
}

impl Default for NchwcLayoutPropagation {
    fn default() -> Self {
        Self::new()
    }
}

const DEFAULT_DOMAINS: [&str; 2] = ["", "ai.onnx"];

#[derive(Clone, Copy)]
enum AutoPad {
    NotSet,
    SameUpper,
    SameLower,
    Valid,
}

/// The kind of blocked pool a standard pool node maps to.
#[derive(Clone, Copy)]
enum PoolOp {
    Max,
    Average,
    GlobalAverage,
}

/// Everything the blocked `NchwcConv` kernel needs, resolved from a `Conv` node.
struct ConvPlan {
    input_blocked: bool,
    filter_bibo: bool,
    group_count: usize,
    kernel: [i64; 2],
    strides: [i64; 2],
    dilations: [i64; 2],
    pads: [i64; 4],
    in_channels: usize,
    out_channels: usize,
    output_h: usize,
    output_w: usize,
    weight_shape: [i64; 4],
    activation: Option<String>,
}

/// What a region member node is, after classification.
enum Member {
    Conv(ConvPlan),
    Pool(PoolOp),
    /// Element-wise op kept on its standard kernel; the listed input indices are
    /// the 4-D data operands that must be fed blocked.
    Elementwise {
        data_inputs: Vec<usize>,
    },
}

impl OptimizationPass for NchwcLayoutPropagation {
    fn name(&self) -> &str {
        "CpuNchwcLayoutPropagation"
    }

    fn run(&self, graph: &mut Graph, _ctx: &PassContext) -> Result<()> {
        if std::env::var_os("NXRT_DISABLE_NCHWC_LAYOUT").is_some() {
            // Escape hatch for A/B profiling and debugging: skip propagation and
            // leave every Conv on the per-op NCHWc/im2col path.
            return Ok(());
        }
        let block = mlas_sys::nchwc_block_size();
        if block < 8 {
            // Host has no blocked-convolution kernel; the per-op path already
            // falls back to im2col, so there is nothing to propagate.
            return Ok(());
        }

        let order = match graph.topological_order() {
            Ok(order) => order,
            Err(_) => return Ok(()),
        };

        // Recompute interior 4-D shapes: the target models carry no `value_info`
        // and this pass runs before the executor's shape inference, so channel
        // counts and output geometry are otherwise invisible to classification.
        let shapes = propagate_shapes(graph);

        // Phase 1: classify region membership in topological order so a node's
        // producers are decided before it.
        let mut members: HashMap<NodeId, Member> = HashMap::new();
        for &nid in &order {
            if let Some(member) = classify(graph, nid, block, &members, &shapes) {
                members.insert(nid, member);
            }
        }
        if members.is_empty() {
            if std::env::var_os("NXRT_NCHWC_DEBUG").is_some() {
                let mut counts: HashMap<&str, usize> = HashMap::new();
                for &nid in &order {
                    if let Some(n) = graph.try_node(nid) {
                        *counts.entry(n.op_type.as_str()).or_default() += 1;
                    }
                }
                eprintln!("[nchwc-layout] no members. op histogram: {counts:?}");
                if let Some(&first_conv) = order
                    .iter()
                    .find(|&&nid| graph.try_node(nid).is_some_and(|n| n.op_type == "Conv"))
                {
                    let n = graph.node(first_conv);
                    let xs = n
                        .inputs
                        .first()
                        .copied()
                        .flatten()
                        .and_then(|v| resolve4(graph, &shapes, v))
                        .map(|s| [s.c, s.h, s.w]);
                    let ws = n
                        .inputs
                        .get(1)
                        .copied()
                        .flatten()
                        .and_then(|v| value_shape4_all_static(graph, v));
                    let os = resolve4(graph, &shapes, n.outputs[0]).map(|s| [s.c, s.h, s.w]);
                    eprintln!(
                        "[nchwc-layout] first Conv x(chw)={xs:?} w={ws:?} out(chw)={os:?} domain={:?}",
                        n.domain
                    );
                }
            }
            return Ok(());
        }

        // Phase 2: materialize blocked twins and rewrite member nodes.
        // `blocked` maps a NCHW value to its blocked (NCHWc) twin, covering both
        // region-produced outputs and entry-reorder inputs.
        let mut blocked: HashMap<ValueId, ValueId> = HashMap::new();
        // Region-produced (original output -> blocked twin) pairs, for the exit
        // phase; ordered for deterministic reorder-node creation.
        let mut region_outputs: Vec<(ValueId, ValueId)> = Vec::new();

        for &nid in &order {
            let Some(member) = members.get(&nid) else {
                continue;
            };
            let node = graph.node(nid).clone();
            let out = node.outputs[0];
            let out_nchw = resolve4(graph, &shapes, out).expect("member output must be static 4-D");
            let out_blocked = graph.create_named_value(
                format!("__nxrt_nchwc_{}", nid.0),
                DataType::Float32,
                blocked_shape(out_nchw, block),
            );

            let new_node = match member {
                Member::Conv(plan) => {
                    let input0 = node.inputs[0].expect("Conv input present");
                    let conv_input = if plan.input_blocked {
                        blocked_input(graph, &mut blocked, &shapes, input0, block)
                    } else {
                        // First-layer 3-channel algorithm consumes NCHW directly.
                        input0
                    };
                    let mut inputs = vec![Some(conv_input), node.inputs[1]];
                    if let Some(bias) = node.inputs.get(2).copied().flatten() {
                        inputs.push(Some(bias));
                    }
                    build_conv_node(nid, plan, inputs, out_blocked)
                }
                Member::Pool(kind) => {
                    let input0 = node.inputs[0].expect("Pool input present");
                    let bin = blocked_input(graph, &mut blocked, &shapes, input0, block);
                    build_pool_node(nid, *kind, &node, bin, out_blocked)
                }
                Member::Elementwise { data_inputs } => {
                    let mut new = node.clone();
                    for &idx in data_inputs {
                        let iv = node.inputs[idx].expect("elementwise data input present");
                        let bin = blocked_input(graph, &mut blocked, &shapes, iv, block);
                        new.inputs[idx] = Some(bin);
                    }
                    new.outputs = vec![out_blocked];
                    new
                }
            };

            graph.replace_node(nid, new_node);
            blocked.insert(out, out_blocked);
            region_outputs.push((out, out_blocked));
        }

        // Phase 3: insert exit reorders for any region-produced value still
        // consumed in NCHW form (by a non-member node or as a graph output).
        for (orig, blocked_twin) in region_outputs {
            let consumers = graph.consumers(orig);
            let is_output = graph.value(orig).is_graph_output;
            if !consumers.is_empty() || is_output {
                let out_shape =
                    resolve4(graph, &shapes, orig).expect("region output must be static 4-D");
                let channels = out_shape.c;
                let nchw = graph.create_named_value(
                    format!("__nxrt_nchwc_out_{}", orig.0),
                    DataType::Float32,
                    nchw_shape(out_shape),
                );
                let mut reorder = Node::new(
                    NodeId(0),
                    REORDER_TO_NCHW_OP,
                    vec![Some(blocked_twin)],
                    vec![nchw],
                );
                reorder.domain = NCHWC_DOMAIN.to_string();
                reorder
                    .attributes
                    .insert("channels".into(), Attribute::Int(channels as i64));
                graph.insert_node(reorder);
                graph.replace_all_uses(orig, nchw);
            }
            // The original NCHW value is now producerless and unused; drop it.
            remove_if_orphan(graph, orig);
        }

        graph
            .opset_imports
            .entry(NCHWC_DOMAIN.to_string())
            .or_insert(1);
        if std::env::var_os("NXRT_NCHWC_DEBUG").is_some() {
            let mut n_conv = 0usize;
            let mut n_pool = 0usize;
            let mut n_elt = 0usize;
            let mut n_to_blocked = 0usize;
            let mut n_to_nchw = 0usize;
            for nid in graph.topological_order().unwrap_or_default() {
                let op = graph.node(nid).op_type.as_str();
                match op {
                    NCHWC_CONV_OP => n_conv += 1,
                    NCHWC_MAX_POOL_OP | NCHWC_AVERAGE_POOL_OP | NCHWC_GLOBAL_AVERAGE_POOL_OP => {
                        n_pool += 1
                    }
                    REORDER_TO_BLOCKED_OP => n_to_blocked += 1,
                    REORDER_TO_NCHW_OP => n_to_nchw += 1,
                    _ => {}
                }
            }
            n_elt += members
                .values()
                .filter(|m| matches!(m, Member::Elementwise { .. }))
                .count();
            eprintln!(
                "[nchwc-layout] members: conv={n_conv} pool={n_pool} elementwise={n_elt} | \
                 entry_reorders(to_blocked)={n_to_blocked} exit_reorders(to_nchw)={n_to_nchw}"
            );
        }
        graph.validate().map_err(OptimizerError::from)?;
        Ok(())
    }
}

/// Fetch (or lazily create an entry reorder for) the blocked twin of `value`.
fn blocked_input(
    graph: &mut Graph,
    blocked: &mut HashMap<ValueId, ValueId>,
    shapes: &HashMap<ValueId, Shape4>,
    value: ValueId,
    block: usize,
) -> ValueId {
    if let Some(&twin) = blocked.get(&value) {
        return twin;
    }
    let nchw = resolve4(graph, shapes, value).expect("blocked input must be static 4-D");
    let twin = graph.create_named_value(
        format!("__nxrt_nchwc_in_{}", value.0),
        DataType::Float32,
        blocked_shape(nchw, block),
    );
    let mut reorder = Node::new(
        NodeId(0),
        REORDER_TO_BLOCKED_OP,
        vec![Some(value)],
        vec![twin],
    );
    reorder.domain = NCHWC_DOMAIN.to_string();
    reorder
        .attributes
        .insert("channels".into(), Attribute::Int(nchw.c as i64));
    graph.insert_node(reorder);
    blocked.insert(value, twin);
    twin
}

fn build_conv_node(
    nid: NodeId,
    plan: &ConvPlan,
    inputs: Vec<Option<ValueId>>,
    output: ValueId,
) -> Node {
    let mut node = Node::new(nid, NCHWC_CONV_OP, inputs, vec![output]);
    node.domain = NCHWC_DOMAIN.to_string();
    let attrs = &mut node.attributes;
    attrs.insert("kernel_shape".into(), Attribute::Ints(plan.kernel.to_vec()));
    attrs.insert("strides".into(), Attribute::Ints(plan.strides.to_vec()));
    attrs.insert("dilations".into(), Attribute::Ints(plan.dilations.to_vec()));
    attrs.insert("pads".into(), Attribute::Ints(plan.pads.to_vec()));
    attrs.insert(
        "group_count".into(),
        Attribute::Int(plan.group_count as i64),
    );
    attrs.insert(
        "in_channels".into(),
        Attribute::Int(plan.in_channels as i64),
    );
    attrs.insert(
        "out_channels".into(),
        Attribute::Int(plan.out_channels as i64),
    );
    attrs.insert("output_h".into(), Attribute::Int(plan.output_h as i64));
    attrs.insert("output_w".into(), Attribute::Int(plan.output_w as i64));
    attrs.insert(
        "input_blocked".into(),
        Attribute::Int(i64::from(plan.input_blocked)),
    );
    attrs.insert(
        "filter_bibo".into(),
        Attribute::Int(i64::from(plan.filter_bibo)),
    );
    attrs.insert(
        "weight_shape".into(),
        Attribute::Ints(plan.weight_shape.to_vec()),
    );
    if let Some(activation) = &plan.activation {
        attrs.insert(
            "activation".into(),
            Attribute::String(activation.clone().into_bytes()),
        );
    }
    node
}

fn build_pool_node(
    nid: NodeId,
    kind: PoolOp,
    original: &Node,
    input: ValueId,
    output: ValueId,
) -> Node {
    let op = match kind {
        PoolOp::Max => NCHWC_MAX_POOL_OP,
        PoolOp::Average => NCHWC_AVERAGE_POOL_OP,
        PoolOp::GlobalAverage => NCHWC_GLOBAL_AVERAGE_POOL_OP,
    };
    let mut node = Node::new(nid, op, vec![Some(input)], vec![output]);
    node.domain = NCHWC_DOMAIN.to_string();
    if matches!(kind, PoolOp::Max | PoolOp::Average) {
        for name in ["kernel_shape", "strides", "pads", "dilations"] {
            if let Some(attr) = original.attr(name) {
                node.attributes.insert(name.into(), attr.clone());
            }
        }
        if matches!(kind, PoolOp::Average)
            && let Some(attr) = original.attr("count_include_pad")
        {
            node.attributes
                .insert("count_include_pad".into(), attr.clone());
        }
    }
    node
}

/// Classify whether `nid` is an NCHWc region member.
fn classify(
    graph: &Graph,
    nid: NodeId,
    block: usize,
    members: &HashMap<NodeId, Member>,
    shapes: &HashMap<ValueId, Shape4>,
) -> Option<Member> {
    let node = graph.try_node(nid)?;
    if !DEFAULT_DOMAINS.contains(&node.domain.as_str()) {
        return None;
    }
    match node.op_type.as_str() {
        "Conv" => conv_plan(graph, node, block, shapes).map(Member::Conv),
        "MaxPool" => pool_ok(graph, node, shapes)
            .then_some(())
            .filter(|_| input_from_member(graph, node, &[0], members))
            .map(|_| Member::Pool(PoolOp::Max)),
        "AveragePool" => pool_ok(graph, node, shapes)
            .then_some(())
            .filter(|_| input_from_member(graph, node, &[0], members))
            .map(|_| Member::Pool(PoolOp::Average)),
        "GlobalAveragePool" => (unary_shapes_ok(graph, node, shapes)
            && input_from_member(graph, node, &[0], members))
        .then_some(Member::Pool(PoolOp::GlobalAverage)),
        "Relu" | "Clip" => (unary_shapes_ok(graph, node, shapes)
            && input_from_member(graph, node, &[0], members))
        .then_some(Member::Elementwise {
            data_inputs: vec![0],
        }),
        "Add" => add_ok(graph, node, shapes)
            .then_some(())
            .filter(|_| input_from_member(graph, node, &[0, 1], members))
            .map(|_| Member::Elementwise {
                data_inputs: vec![0, 1],
            }),
        _ => None,
    }
}

/// Whether at least one of the listed 4-D data inputs is produced by a member.
fn input_from_member(
    graph: &Graph,
    node: &Node,
    indices: &[usize],
    members: &HashMap<NodeId, Member>,
) -> bool {
    indices.iter().any(|&idx| {
        node.inputs
            .get(idx)
            .copied()
            .flatten()
            .and_then(|v| graph.value(v).producer)
            .is_some_and(|producer| members.contains_key(&producer))
    })
}

fn unary_shapes_ok(graph: &Graph, node: &Node, shapes: &HashMap<ValueId, Shape4>) -> bool {
    if node.outputs.len() != 1 {
        return false;
    }
    let Some(input) = node.inputs.first().copied().flatten() else {
        return false;
    };
    let (Some(i), Some(o)) = (
        resolve4(graph, shapes, input),
        resolve4(graph, shapes, node.outputs[0]),
    ) else {
        return false;
    };
    i.c.is_multiple_of(4) && o.c.is_multiple_of(4)
}

fn add_ok(graph: &Graph, node: &Node, shapes: &HashMap<ValueId, Shape4>) -> bool {
    if node.inputs.len() != 2 || node.outputs.len() != 1 {
        return false;
    }
    let (Some(a), Some(b)) = (
        node.inputs[0].and_then(|v| resolve4(graph, shapes, v)),
        node.inputs[1].and_then(|v| resolve4(graph, shapes, v)),
    ) else {
        return false;
    };
    // Identical channel/spatial extents only: no broadcasting, so element-wise
    // Add over the blocked buffer equals the NCHW result. Channels must reorder.
    a.c == b.c && a.h == b.h && a.w == b.w && a.c.is_multiple_of(4)
}

/// Pooling nodes are only NCHWc-capable when they map cleanly onto `MlasNchwcPool`
/// (2-D, single output, no dilation/ceil-mode/storage-order surprises).
fn pool_ok(graph: &Graph, node: &Node, shapes: &HashMap<ValueId, Shape4>) -> bool {
    if node.outputs.len() != 1 {
        return false;
    }
    let Some(input) = node.inputs.first().copied().flatten() else {
        return false;
    };
    let (Some(i), Some(o)) = (
        resolve4(graph, shapes, input),
        resolve4(graph, shapes, node.outputs[0]),
    ) else {
        return false;
    };
    if !i.c.is_multiple_of(4) || !o.c.is_multiple_of(4) {
        return false;
    }
    // 2-D kernel.
    if node
        .attr("kernel_shape")
        .and_then(Attribute::as_ints)
        .is_none_or(|k| k.len() != 2)
    {
        return false;
    }
    // No dilation, ceil-mode, storage-order, or SAME padding: MLAS's blocked
    // pool computes classic floor-mode output geometry.
    if node
        .attr("dilations")
        .and_then(Attribute::as_ints)
        .is_some_and(|d| d.iter().any(|&v| v != 1))
    {
        return false;
    }
    if node
        .attr("ceil_mode")
        .and_then(Attribute::as_int)
        .unwrap_or(0)
        != 0
    {
        return false;
    }
    if node
        .attr("storage_order")
        .and_then(Attribute::as_int)
        .unwrap_or(0)
        != 0
    {
        return false;
    }
    matches!(auto_pad(node), Some(AutoPad::NotSet) | Some(AutoPad::Valid))
}

/// Blocked-conv eligibility, mirroring `kernels::conv`'s `select_impl`.
fn conv_plan(
    graph: &Graph,
    node: &Node,
    block: usize,
    shapes: &HashMap<ValueId, Shape4>,
) -> Option<ConvPlan> {
    if node.outputs.len() != 1 {
        return None;
    }
    let x = node.inputs.first().copied().flatten()?;
    let w = node.inputs.get(1).copied().flatten()?;
    let x_shape = resolve4(graph, shapes, x)?;
    let out_shape = resolve4(graph, shapes, node.outputs[0])?;
    let w_shape = value_shape4_all_static(graph, w)?;

    let group = node
        .attr("group")
        .and_then(Attribute::as_int)
        .unwrap_or(1)
        .max(1) as usize;
    let input_channels = x_shape.c;
    let output_channels = w_shape[0];
    let ic_per_group = w_shape[1];
    if !input_channels.is_multiple_of(group)
        || !output_channels.is_multiple_of(group)
        || ic_per_group != input_channels / group
    {
        return None;
    }

    let kernel = [w_shape[2] as i64, w_shape[3] as i64];
    let dilations = pair_attr(node, "dilations", 1)?;
    let strides = pair_attr(node, "strides", 1)?;
    let (pads, _) = resolve_pads(
        [x_shape.h, x_shape.w],
        [w_shape[2], w_shape[3]],
        [dilations[0] as usize, dilations[1] as usize],
        [strides[0] as usize, strides[1] as usize],
        explicit_pads(node)?,
        auto_pad(node)?,
    )?;

    // 4-byte channel alignment: MlasReorderInputNchw reads channels four at a time.
    const CHANNEL_ALIGNMENT: usize = 4;
    let (input_blocked, filter_bibo, group_count) = if group == 1 {
        if input_channels < block {
            (false, false, 1)
        } else if input_channels.is_multiple_of(CHANNEL_ALIGNMENT) {
            (true, true, 1)
        } else {
            return None;
        }
    } else if ic_per_group == 1
        && output_channels == group
        && output_channels.is_multiple_of(CHANNEL_ALIGNMENT)
    {
        (true, false, round_up(output_channels, block))
    } else {
        return None;
    };

    let activation = node
        .attr("activation")
        .and_then(Attribute::as_str)
        .map(str::to_string);

    Some(ConvPlan {
        input_blocked,
        filter_bibo,
        group_count,
        kernel,
        strides,
        dilations,
        pads,
        in_channels: input_channels,
        out_channels: output_channels,
        output_h: out_shape.h,
        output_w: out_shape.w,
        weight_shape: [
            w_shape[0] as i64,
            w_shape[1] as i64,
            w_shape[2] as i64,
            w_shape[3] as i64,
        ],
        activation,
    })
}

fn pair_attr(node: &Node, name: &str, default: i64) -> Option<[i64; 2]> {
    match node.attr(name).and_then(Attribute::as_ints) {
        None => Some([default; 2]),
        Some(values) if values.len() == 2 && values.iter().all(|&v| v > 0) => {
            Some([values[0], values[1]])
        }
        Some(_) => None,
    }
}

fn explicit_pads(node: &Node) -> Option<[usize; 4]> {
    match node.attr("pads").and_then(Attribute::as_ints) {
        None => Some([0; 4]),
        Some(values) if values.len() == 4 && values.iter().all(|&v| v >= 0) => Some([
            values[0] as usize,
            values[1] as usize,
            values[2] as usize,
            values[3] as usize,
        ]),
        Some(_) => None,
    }
}

fn auto_pad(node: &Node) -> Option<AutoPad> {
    match node.attr("auto_pad").and_then(Attribute::as_str) {
        None | Some("NOTSET") => Some(AutoPad::NotSet),
        Some("SAME_UPPER") => Some(AutoPad::SameUpper),
        Some("SAME_LOWER") => Some(AutoPad::SameLower),
        Some("VALID") => Some(AutoPad::Valid),
        Some(_) => None,
    }
}

/// Resolve explicit pad values, matching `kernels::conv::output_geometry`.
fn resolve_pads(
    input: [usize; 2],
    kernel: [usize; 2],
    dilations: [usize; 2],
    strides: [usize; 2],
    mut pads: [usize; 4],
    auto_pad: AutoPad,
) -> Option<([i64; 4], [usize; 2])> {
    let mut output = [0usize; 2];
    for axis in 0..2 {
        let effective = dilations[axis]
            .checked_mul(kernel[axis] - 1)?
            .checked_add(1)?;
        match auto_pad {
            AutoPad::SameUpper | AutoPad::SameLower => {
                output[axis] = input[axis].div_ceil(strides[axis]);
                let total = output[axis]
                    .saturating_sub(1)
                    .checked_mul(strides[axis])?
                    .checked_add(effective)?
                    .saturating_sub(input[axis]);
                let begin = if matches!(auto_pad, AutoPad::SameUpper) {
                    total / 2
                } else {
                    total - total / 2
                };
                pads[axis] = begin;
                pads[axis + 2] = total - begin;
            }
            AutoPad::Valid => {
                pads[axis] = 0;
                pads[axis + 2] = 0;
                output[axis] = if input[axis] < effective {
                    0
                } else {
                    (input[axis] - effective) / strides[axis] + 1
                };
            }
            AutoPad::NotSet => {
                let padded = input[axis]
                    .checked_add(pads[axis])?
                    .checked_add(pads[axis + 2])?;
                output[axis] = if padded < effective {
                    0
                } else {
                    (padded - effective) / strides[axis] + 1
                };
            }
        }
    }
    Some((
        [
            pads[0] as i64,
            pads[1] as i64,
            pads[2] as i64,
            pads[3] as i64,
        ],
        output,
    ))
}

/// A 4-D `[N, C, H, W]` shape where the batch dimension may stay symbolic while
/// the channel/spatial extents are statically known. Layout eligibility and
/// blocked geometry depend only on `C`/`H`/`W`; the batch `Dim` is carried
/// verbatim so blocked/reorder values we synthesize keep the graph's symbol and
/// resolve to the concrete batch at runtime.
#[derive(Clone, Copy)]
struct Shape4 {
    n: Dim,
    c: usize,
    h: usize,
    w: usize,
}

/// The `[N, C, H, W]` shape of `value` with static `C`/`H`/`W`, if known.
fn value_shape4(graph: &Graph, value: ValueId) -> Option<Shape4> {
    let shape = &graph.try_value(value)?.shape;
    if shape.len() != 4 {
        return None;
    }
    Some(Shape4 {
        n: shape[0],
        c: shape[1].as_static()?,
        h: shape[2].as_static()?,
        w: shape[3].as_static()?,
    })
}

/// Resolve a value's 4-D shape, preferring the pass-local propagated cache and
/// falling back to any statically-declared graph shape (inputs, initializers,
/// pre-existing `value_info`).
fn resolve4(graph: &Graph, shapes: &HashMap<ValueId, Shape4>, value: ValueId) -> Option<Shape4> {
    shapes
        .get(&value)
        .copied()
        .or_else(|| value_shape4(graph, value))
}

/// Forward-propagate `[N, C, H, W]` shapes through the graph in topological order.
///
/// The vision models we target ship no `value_info`, and this pass runs *before*
/// the executor's post-pass shape inference, so interior activation shapes are
/// otherwise unknown. We recompute them for the layout-relevant op set (Conv,
/// pooling, and 4-D-preserving element-wise ops) so region classification can
/// see real channel counts and output geometry.
fn propagate_shapes(graph: &Graph) -> HashMap<ValueId, Shape4> {
    let mut shapes: HashMap<ValueId, Shape4> = HashMap::new();
    let Ok(order) = graph.topological_order() else {
        return shapes;
    };
    for nid in order {
        let Some(node) = graph.try_node(nid) else {
            continue;
        };
        if node.outputs.len() != 1 {
            continue;
        }
        if let Some(out) = compute_output_shape(graph, &shapes, node) {
            shapes.insert(node.outputs[0], out);
        }
    }
    shapes
}

/// Best-effort `[N, C, H, W]` output shape for the layout-relevant op set.
fn compute_output_shape(
    graph: &Graph,
    shapes: &HashMap<ValueId, Shape4>,
    node: &Node,
) -> Option<Shape4> {
    if !DEFAULT_DOMAINS.contains(&node.domain.as_str()) {
        return None;
    }
    let input0 = node.inputs.first().copied().flatten();
    match node.op_type.as_str() {
        "Conv" => {
            let x = resolve4(graph, shapes, input0?)?;
            let w = value_shape4_all_static(graph, node.inputs.get(1).copied().flatten()?)?;
            let strides = pair_attr(node, "strides", 1)?;
            let dilations = pair_attr(node, "dilations", 1)?;
            let (_, out_hw) = resolve_pads(
                [x.h, x.w],
                [w[2], w[3]],
                [dilations[0] as usize, dilations[1] as usize],
                [strides[0] as usize, strides[1] as usize],
                explicit_pads(node)?,
                auto_pad(node)?,
            )?;
            Some(Shape4 {
                n: x.n,
                c: w[0],
                h: out_hw[0],
                w: out_hw[1],
            })
        }
        "MaxPool" | "AveragePool" => {
            let x = resolve4(graph, shapes, input0?)?;
            let kernel = node.attr("kernel_shape").and_then(Attribute::as_ints)?;
            if kernel.len() != 2 {
                return None;
            }
            let strides = pair_attr(node, "strides", 1)?;
            let dilations = pair_attr(node, "dilations", 1)?;
            let (_, out_hw) = resolve_pads(
                [x.h, x.w],
                [kernel[0] as usize, kernel[1] as usize],
                [dilations[0] as usize, dilations[1] as usize],
                [strides[0] as usize, strides[1] as usize],
                explicit_pads(node)?,
                auto_pad(node)?,
            )?;
            Some(Shape4 {
                n: x.n,
                c: x.c,
                h: out_hw[0],
                w: out_hw[1],
            })
        }
        "GlobalAveragePool" => {
            let x = resolve4(graph, shapes, input0?)?;
            Some(Shape4 {
                n: x.n,
                c: x.c,
                h: 1,
                w: 1,
            })
        }
        // 4-D-preserving element-wise ops: output shape equals the (first) 4-D
        // data input. `Add` requires identical operand shapes elsewhere.
        "Relu" | "Clip" | "BatchNormalization" | "Add" | "Mul" | "Sub" | "Div" | "LeakyRelu"
        | "PRelu" => resolve4(graph, shapes, input0?),
        _ => None,
    }
}

/// The fully-static `[N, C, H, W]` of a weight/initializer value.
fn value_shape4_all_static(graph: &Graph, value: ValueId) -> Option<[usize; 4]> {
    let shape = &graph.try_value(value)?.shape;
    let dims = as_static_shape(shape)?;
    (dims.len() == 4).then(|| [dims[0], dims[1], dims[2], dims[3]])
}

/// Blocked `[N, round_up(C, block), H, W]` shape, preserving the batch `Dim`
/// (possibly symbolic) so the value resolves to the concrete batch at runtime.
fn blocked_shape(nchw: Shape4, block: usize) -> Vec<Dim> {
    vec![
        nchw.n,
        Dim::Static(round_up(nchw.c, block)),
        Dim::Static(nchw.h),
        Dim::Static(nchw.w),
    ]
}

/// NCHW `[N, C, H, W]` shape, preserving the batch `Dim`.
fn nchw_shape(nchw: Shape4) -> Vec<Dim> {
    vec![
        nchw.n,
        Dim::Static(nchw.c),
        Dim::Static(nchw.h),
        Dim::Static(nchw.w),
    ]
}

fn remove_if_orphan(graph: &mut Graph, value: ValueId) {
    // Routes through the IR's orphan GC so the freed arena slot also clears its
    // unknown-type/shape flags; a raw `values.remove` would leave stale flags
    // that a later slot reuse (our own blocked twins) inherits, defeating shape
    // inference's seeding of those values.
    graph.gc_value_if_orphan(value);
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::static_shape;
    use onnx_runtime_optimizer::PassContext;

    fn count_op(graph: &Graph, op: &str) -> usize {
        graph.nodes.values().filter(|n| n.op_type == op).count()
    }

    fn conv_node(name: &str, x: ValueId, w: ValueId, out: ValueId) -> Node {
        let mut node = Node::new(NodeId(0), "Conv", vec![Some(x), Some(w)], vec![out]);
        node.attributes
            .insert("kernel_shape".into(), Attribute::Ints(vec![3, 3]));
        node.attributes
            .insert("pads".into(), Attribute::Ints(vec![1, 1, 1, 1]));
        node.attributes
            .insert("strides".into(), Attribute::Ints(vec![1, 1]));
        node.name = name.to_string();
        node
    }

    fn run_pass(graph: &mut Graph) {
        NchwcLayoutPropagation::new()
            .run(graph, &PassContext::new())
            .expect("layout propagation pass");
    }

    /// A 3x3-same Conv weight value with `[out, in, 3, 3]`.
    fn weight(graph: &mut Graph, name: &str, out: usize, inn: usize) -> ValueId {
        graph.create_named_value(name, DataType::Float32, static_shape([out, inn, 3, 3]))
    }

    /// `X -> Conv -> Relu -> Conv -> Relu(out)`: the whole chain must collapse
    /// into one NCHWc region — two blocked Convs, both Relus kept, a single
    /// entry reorder and a single exit reorder, and no interior reorders.
    #[test]
    fn linear_conv_relu_chain_forms_single_region() {
        // 16 channels is a multiple of every supported block size (8/16).
        const C: usize = 16;
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 13);
        let x = graph.create_named_value("x", DataType::Float32, static_shape([1, C, 8, 8]));
        graph.add_input(x);

        let w1 = weight(&mut graph, "w1", C, C);
        let c1 = graph.create_named_value("c1", DataType::Float32, static_shape([1, C, 8, 8]));
        graph.insert_node(conv_node("conv1", x, w1, c1));
        let r1 = graph.create_named_value("r1", DataType::Float32, static_shape([1, C, 8, 8]));
        graph.insert_node(Node::new(NodeId(0), "Relu", vec![Some(c1)], vec![r1]));

        let w2 = weight(&mut graph, "w2", C, C);
        let c2 = graph.create_named_value("c2", DataType::Float32, static_shape([1, C, 8, 8]));
        graph.insert_node(conv_node("conv2", r1, w2, c2));
        let r2 = graph.create_named_value("r2", DataType::Float32, static_shape([1, C, 8, 8]));
        graph.insert_node(Node::new(NodeId(0), "Relu", vec![Some(c2)], vec![r2]));
        graph.add_output(r2);

        run_pass(&mut graph);

        assert_eq!(count_op(&graph, NCHWC_CONV_OP), 2, "both Convs blocked");
        assert_eq!(count_op(&graph, "Conv"), 0, "no plain Conv remains");
        assert_eq!(count_op(&graph, "Relu"), 2, "Relus stay element-wise");
        assert_eq!(
            count_op(&graph, REORDER_TO_BLOCKED_OP),
            1,
            "exactly one entry reorder for the whole region"
        );
        assert_eq!(
            count_op(&graph, REORDER_TO_NCHW_OP),
            1,
            "exactly one exit reorder at the region output"
        );
        assert!(graph.validate().is_ok());
    }

    /// A residual `Add` joining two blocked producers must stay a plain `Add`
    /// whose operands are the producers' blocked twins — proving element-wise
    /// ops need no per-op reorder and residuals fuse into the region.
    #[test]
    fn residual_add_joins_blocked_paths_without_reorder() {
        const C: usize = 16;
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 13);
        let x = graph.create_named_value("x", DataType::Float32, static_shape([1, C, 8, 8]));
        graph.add_input(x);

        let w1 = weight(&mut graph, "w1", C, C);
        let c1 = graph.create_named_value("c1", DataType::Float32, static_shape([1, C, 8, 8]));
        graph.insert_node(conv_node("conv1", x, w1, c1));

        let w2 = weight(&mut graph, "w2", C, C);
        let c2 = graph.create_named_value("c2", DataType::Float32, static_shape([1, C, 8, 8]));
        graph.insert_node(conv_node("conv2", c1, w2, c2));

        // Residual: Add(conv1_out, conv2_out) — both inside the region.
        let sum = graph.create_named_value("sum", DataType::Float32, static_shape([1, C, 8, 8]));
        let add_id = graph.insert_node(Node::new(
            NodeId(0),
            "Add",
            vec![Some(c1), Some(c2)],
            vec![sum],
        ));
        graph.add_output(sum);

        run_pass(&mut graph);

        assert_eq!(count_op(&graph, NCHWC_CONV_OP), 2);
        assert_eq!(count_op(&graph, "Add"), 1, "Add stays element-wise");
        // A single entry reorder (conv1 consumes blocked X) and a single exit
        // reorder (the Add output leaves the region). No reorder feeds the Add.
        assert_eq!(count_op(&graph, REORDER_TO_BLOCKED_OP), 1);
        assert_eq!(count_op(&graph, REORDER_TO_NCHW_OP), 1);

        // The Add's operands are exactly the two blocked Conv outputs, so the
        // residual is consumed directly in NCHWc with no intervening reorder.
        let add = graph.node(add_id);
        let conv_outputs: std::collections::HashSet<ValueId> = graph
            .nodes
            .values()
            .filter(|n| n.op_type == NCHWC_CONV_OP)
            .map(|n| n.outputs[0])
            .collect();
        for slot in &add.inputs {
            let v = slot.expect("Add operand present");
            assert!(
                conv_outputs.contains(&v),
                "Add operand should be a blocked Conv output"
            );
        }
        assert!(graph.validate().is_ok());
    }

    /// A first layer with fewer channels than the block (RGB stem) stays on the
    /// NCHW-direct blocked-conv algorithm: it consumes NCHW directly (no entry
    /// reorder) yet still produces a blocked output that the next Conv reuses.
    #[test]
    fn rgb_stem_needs_no_entry_reorder() {
        const C: usize = 16;
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 13);
        let x = graph.create_named_value("x", DataType::Float32, static_shape([1, 3, 8, 8]));
        graph.add_input(x);

        let w1 = weight(&mut graph, "w1", C, 3);
        let c1 = graph.create_named_value("c1", DataType::Float32, static_shape([1, C, 8, 8]));
        graph.insert_node(conv_node("conv1", x, w1, c1));

        let w2 = weight(&mut graph, "w2", C, C);
        let c2 = graph.create_named_value("c2", DataType::Float32, static_shape([1, C, 8, 8]));
        graph.insert_node(conv_node("conv2", c1, w2, c2));
        graph.add_output(c2);

        run_pass(&mut graph);

        assert_eq!(count_op(&graph, NCHWC_CONV_OP), 2);
        assert_eq!(
            count_op(&graph, REORDER_TO_BLOCKED_OP),
            0,
            "3-channel stem consumes NCHW directly"
        );
        assert_eq!(count_op(&graph, REORDER_TO_NCHW_OP), 1);
        assert!(graph.validate().is_ok());
    }
}

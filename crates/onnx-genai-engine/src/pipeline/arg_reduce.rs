//! Exact tiling of a degenerate last-axis `ArgMax` / `ArgMin`.
//!
//! ORT's CUDA `arg_min_max_last_axis_kernel` parallelizes across reduced *rows*
//! and walks each row's reduced extent with a single lane. A greedy token
//! sampler reduces one row of the whole vocabulary, so the kernel degenerates to
//! a serial scan: measured on an H200 with stock ORT 1.28 it costs 4.46 ms for
//! `[1, 202048]` float32, more than half of the entire decode step, and it is
//! insensitive to cuDNN availability because ORT never routes arg-reductions
//! through cuDNN.
//!
//! This pass rewrites such a node into the identical reduction expressed over a
//! `[rows, tile]` factorization of the same extent, which gives the same kernel
//! `rows` lanes to work with:
//!
//! ```text
//! tiled  = Reshape(x, [0, rows, tile])            # (B, rows, tile)
//! inner  = ArgMax(tiled, axis=-1, keepdims=1)     # (B, rows, 1)
//! best   = GatherElements(tiled, inner, axis=-1)  # (B, rows, 1) row extrema
//! outer  = ArgMax(best', axis=-1, keepdims=1)     # (B, 1)
//! index  = outer * tile + inner'[outer]           # (B, 1)
//! ```
//!
//! For every finite row the result is bit-exact, including tie-breaking.
//! `ArgMax` with the default `select_last_index=0` returns the first maximal
//! index in a row; the outer reduction then selects the first tile holding the
//! running maximum, so the recombined index is the first maximal index of the
//! flat row. With `select_last_index=1` both stages select the last, which
//! recombines to the last maximal flat index. `ArgMin` follows the same argument
//! on minima.
//!
//! A NaN is the one input this does not reproduce: it can win its tile and hide
//! that tile's real extremum, so the recombined index may differ from the flat
//! scan's. ONNX leaves `ArgMax`/`ArgMin` NaN behaviour undefined and the flat
//! kernel's own answer is already arbitrary, so this is a difference between two
//! unspecified results rather than a regression — but it is not exactness, and
//! callers that must match a specific kernel on NaN rows cannot use this pass.
//!
//! Only a statically known reduced extent is rewritten, so the emitted shapes
//! stay fully static and the island remains capturable as a single CUDA graph.

use std::collections::HashMap;

use onnx_runtime_loader::proto::onnx::{
    GraphProto, NodeProto, TensorProto, tensor_proto, tensor_shape_proto, type_proto,
};

/// Reduced extents below this never degenerate badly enough to be worth extra nodes.
const MINIMUM_TILED_EXTENT: i64 = 4_096;

/// Require the tiled scan length to be a small fraction of the flat scan it replaces.
const MINIMUM_SCAN_REDUCTION: i64 = 8;

/// Domain of the optional fused CUDA kernel, mirrored from
/// `onnx_genai_ort::fused_argmax::DOMAIN` so this pass does not depend on the
/// CUDA feature being enabled to know the name it would emit.
pub const FUSED_DOMAIN: &str = "com.github.onnx_genai";

/// Op name of the optional fused CUDA kernel.
pub const FUSED_OP: &str = "ArgMaxLastAxis";

/// What each rewritten node was replaced with.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArgReduceRewrites {
    /// Nodes replaced by the portable `rows x tile` ONNX expansion.
    pub tiled: usize,
    /// Nodes replaced by the single fused custom-op node.
    pub fused: usize,
}

impl ArgReduceRewrites {
    pub fn total(self) -> usize {
        self.tiled + self.fused
    }
}

/// Rewrite every degenerate last-axis arg-reduction in `graph`.
///
/// When `fused` is set, a node the fused CUDA kernel can serve exactly becomes a
/// single [`FUSED_OP`] node; everything else — `ArgMin`, `select_last_index=1`,
/// a non-float input, or any graph running without the kernel — falls back to
/// the portable tiling, which computes the same result using only standard ONNX
/// ops.
pub fn tile_degenerate_arg_reductions(graph: &mut GraphProto, fused: bool) -> ArgReduceRewrites {
    let shapes = static_shapes(graph);
    let element_types = if fused {
        element_types(graph)
    } else {
        HashMap::new()
    };
    let mut rewritten = ArgReduceRewrites::default();
    let mut index = 0;
    while index < graph.node.len() {
        let Some(plan) = plan_rewrite(&graph.node[index], &shapes) else {
            index += 1;
            continue;
        };
        let replacement = if fused && fusable(&graph.node[index], &element_types) {
            rewritten.fused += 1;
            fuse(&graph.node[index], &plan, &mut graph.initializer)
        } else {
            rewritten.tiled += 1;
            expand(&graph.node[index], &plan, &mut graph.initializer)
        };
        let inserted = replacement.len();
        graph.node.splice(index..=index, replacement);
        index += inserted;
    }
    rewritten
}

/// Element types the fused kernel implements. `ArgMax` is defined over every
/// numeric tensor type, but the kernel dispatches only these, and an unsupported
/// type would fail at the first `Run` rather than at session creation, where the
/// island has no fallback left. Anything else takes the portable expansion,
/// which is type agnostic.
const FUSABLE_ELEMENT_TYPES: [i32; 3] = [
    tensor_proto::DataType::Float as i32,
    tensor_proto::DataType::Float16 as i32,
    tensor_proto::DataType::Bfloat16 as i32,
];

/// Whether the fused kernel reproduces this node exactly.
///
/// The kernel is a maximum reduction over float data that keeps the lowest index
/// on ties, so `ArgMin`, `select_last_index=1`, a non-float input and an input
/// whose type the graph does not declare all stay on the portable expansion
/// rather than being served by something that would answer a different question
/// or not run at all.
fn fusable(node: &NodeProto, element_types: &HashMap<String, i32>) -> bool {
    if node.op_type != "ArgMax" || attribute_int(node, "select_last_index").unwrap_or(0) != 0 {
        return false;
    }
    node.input
        .first()
        .and_then(|input| element_types.get(input))
        .is_some_and(|elem| FUSABLE_ELEMENT_TYPES.contains(elem))
}

/// Declared element type of every value the graph types.
///
/// `Cast` is followed through its `to` attribute rather than its input, because
/// changing the element type is the whole point of the node; `Identity` keeps
/// its input's type. A value this cannot type is simply absent, which callers
/// treat as "not known to be fusable".
fn element_types(graph: &GraphProto) -> HashMap<String, i32> {
    let mut types = HashMap::new();
    for value in graph
        .input
        .iter()
        .chain(&graph.value_info)
        .chain(&graph.output)
    {
        if let Some(type_proto::Value::TensorType(tensor)) =
            value.r#type.as_ref().and_then(|kind| kind.value.as_ref())
        {
            types.insert(value.name.clone(), tensor.elem_type);
        }
    }
    for initializer in &graph.initializer {
        types.insert(initializer.name.clone(), initializer.data_type);
    }
    let mut changed = true;
    while changed {
        changed = false;
        for node in &graph.node {
            let (Some(input), Some(output)) = (node.input.first(), node.output.first()) else {
                continue;
            };
            if types.contains_key(output) {
                continue;
            }
            let produced = match node.op_type.as_str() {
                "Cast" => attribute_int(node, "to").map(|to| to as i32),
                "Identity" => types.get(input).copied(),
                _ => None,
            };
            if let Some(elem) = produced {
                types.insert(output.clone(), elem);
                changed = true;
            }
        }
    }
    types
}

/// Replace a degenerate `ArgMax` with the fused custom op.
///
/// The kernel always drops the reduced axis, so `keepdims=1` keeps its original
/// shape through a trailing `Reshape` rather than changing the kernel contract.
fn fuse(node: &NodeProto, plan: &TilePlan, initializers: &mut Vec<TensorProto>) -> Vec<NodeProto> {
    let base = if node.name.is_empty() {
        format!("fused_{}_{}", node.op_type, node.output[0])
    } else {
        format!("fused_{}", node.name)
    };
    let result = node.output[0].clone();
    if !plan.keepdims {
        return vec![
            make_node(FUSED_OP, &[&node.input[0]], &[&result], &base).with_domain(FUSED_DOMAIN),
        ];
    }
    let flat = format!("{base}__flat");
    let keep_shape = format!("{base}__keep_shape");
    initializers.push(int64_initializer(&keep_shape, &[0, 1]));
    vec![
        make_node(FUSED_OP, &[&node.input[0]], &[&flat], &base).with_domain(FUSED_DOMAIN),
        make_node(
            "Reshape",
            &[&flat, &keep_shape],
            &[&result],
            &format!("{base}_keepdims"),
        ),
    ]
}

struct TilePlan {
    rows: i64,
    tile: i64,
    keepdims: bool,
}

/// Factor `extent` into `rows * tile` minimizing the longest serial scan any lane
/// performs, which is `max(rows, tile)` across the two stages.
fn plan_tiles(extent: i64) -> Option<(i64, i64)> {
    if extent < MINIMUM_TILED_EXTENT {
        return None;
    }
    let mut best: Option<(i64, i64)> = None;
    let mut divisor = 1i64;
    while divisor.saturating_mul(divisor) <= extent {
        if extent % divisor == 0 {
            let candidate = (divisor, extent / divisor);
            if best.is_none_or(|(rows, tile)| candidate.0 + candidate.1 < rows + tile) {
                best = Some(candidate);
            }
        }
        divisor += 1;
    }
    let (rows, tile) = best?;
    (rows > 1 && tile > 1 && rows.saturating_add(tile) <= extent / MINIMUM_SCAN_REDUCTION)
        .then_some((rows, tile))
}

fn plan_rewrite(node: &NodeProto, shapes: &HashMap<String, Vec<Option<i64>>>) -> Option<TilePlan> {
    if !matches!(node.op_type.as_str(), "ArgMax" | "ArgMin") {
        return None;
    }
    let shape = shapes.get(node.input.first()?)?;
    // Only the two-dimensional `(batch, reduced)` form is tiled: the leading axis
    // stays the serving row axis so a stable binding still keys on one batch.
    if shape.len() != 2 {
        return None;
    }
    let axis = attribute_int(node, "axis").unwrap_or(0);
    if axis != 1 && axis != -1 {
        return None;
    }
    let keepdims = attribute_int(node, "keepdims").unwrap_or(1) != 0;
    let extent = shape[1]?;
    let (rows, tile) = plan_tiles(extent)?;
    Some(TilePlan {
        rows,
        tile,
        keepdims,
    })
}

fn expand(
    node: &NodeProto,
    plan: &TilePlan,
    initializers: &mut Vec<TensorProto>,
) -> Vec<NodeProto> {
    let base = if node.name.is_empty() {
        format!("tiled_{}_{}", node.op_type, node.output[0])
    } else {
        format!("tiled_{}", node.name)
    };
    let source = node.input[0].clone();
    let result = node.output[0].clone();
    let select_last = attribute_int(node, "select_last_index").unwrap_or(0);

    let tiled_shape = format!("{base}__tiled_shape");
    let rows_shape = format!("{base}__rows_shape");
    let tile_scale = format!("{base}__tile");
    initializers.push(int64_initializer(&tiled_shape, &[0, plan.rows, plan.tile]));
    initializers.push(int64_initializer(&rows_shape, &[0, plan.rows]));
    initializers.push(int64_initializer(&tile_scale, &[plan.tile]));

    let tiled = format!("{base}__tiled");
    let inner = format!("{base}__inner");
    let extrema = format!("{base}__extrema");
    let rows_extrema = format!("{base}__rows_extrema");
    let rows_inner = format!("{base}__rows_inner");
    let outer = format!("{base}__outer");
    let selected = format!("{base}__selected");
    let scaled = format!("{base}__scaled");
    let flat = if plan.keepdims {
        result.clone()
    } else {
        format!("{base}__flat")
    };

    let mut nodes = vec![
        // (B, extent) -> (B, rows, tile); axis 0 is copied so the row axis stays dynamic.
        make_node(
            "Reshape",
            &[&source, &tiled_shape],
            &[&tiled],
            &format!("{base}_reshape"),
        ),
        // Per-tile extremum position, one lane per (row, tile) pair.
        arg_node(
            &node.op_type,
            &tiled,
            &inner,
            &format!("{base}_inner"),
            -1,
            1,
            select_last,
        ),
        // Gather each tile's extreme value so the outer stage reduces `rows` values.
        make_node(
            "GatherElements",
            &[&tiled, &inner],
            &[&extrema],
            &format!("{base}_gather_extrema"),
        )
        .with_int_attribute("axis", -1),
        make_node(
            "Reshape",
            &[&extrema, &rows_shape],
            &[&rows_extrema],
            &format!("{base}_reshape_extrema"),
        ),
        make_node(
            "Reshape",
            &[&inner, &rows_shape],
            &[&rows_inner],
            &format!("{base}_reshape_inner"),
        ),
        arg_node(
            &node.op_type,
            &rows_extrema,
            &outer,
            &format!("{base}_outer"),
            -1,
            1,
            select_last,
        ),
        make_node(
            "GatherElements",
            &[&rows_inner, &outer],
            &[&selected],
            &format!("{base}_gather_inner"),
        )
        .with_int_attribute("axis", -1),
        // Recombine the flat index: tile origin plus the offset inside that tile.
        make_node(
            "Mul",
            &[&outer, &tile_scale],
            &[&scaled],
            &format!("{base}_scale"),
        ),
        make_node(
            "Add",
            &[&scaled, &selected],
            &[&flat],
            &format!("{base}_combine"),
        ),
    ];
    if !plan.keepdims {
        let squeeze_shape = format!("{base}__squeeze_shape");
        initializers.push(int64_initializer(&squeeze_shape, &[0]));
        nodes.push(make_node(
            "Reshape",
            &[&flat, &squeeze_shape],
            &[&result],
            &format!("{base}_squeeze"),
        ));
    }
    nodes
}

/// Collect statically known tensor shapes declared by graph inputs, value info,
/// and initializers. A dimension that is symbolic or absent stays `None`.
fn static_shapes(graph: &GraphProto) -> HashMap<String, Vec<Option<i64>>> {
    let mut shapes = HashMap::new();
    for value in graph
        .input
        .iter()
        .chain(&graph.value_info)
        .chain(&graph.output)
    {
        let Some(type_proto::Value::TensorType(tensor)) =
            value.r#type.as_ref().and_then(|kind| kind.value.as_ref())
        else {
            continue;
        };
        let Some(shape) = tensor.shape.as_ref() else {
            continue;
        };
        shapes.insert(
            value.name.clone(),
            shape
                .dim
                .iter()
                .map(|dimension| match dimension.value {
                    Some(tensor_shape_proto::dimension::Value::DimValue(extent)) if extent > 0 => {
                        Some(extent)
                    }
                    _ => None,
                })
                .collect(),
        );
    }
    for initializer in &graph.initializer {
        shapes.insert(
            initializer.name.clone(),
            initializer
                .dims
                .iter()
                .map(|extent| Some(*extent))
                .collect(),
        );
    }
    // Shape-transparent producers let a rewrite see through the casts a policy
    // artifact commonly places in front of its reduction.
    let mut changed = true;
    while changed {
        changed = false;
        for node in &graph.node {
            if !matches!(node.op_type.as_str(), "Cast" | "Identity") {
                continue;
            }
            let (Some(input), Some(output)) = (node.input.first(), node.output.first()) else {
                continue;
            };
            if shapes.contains_key(output) {
                continue;
            }
            let Some(shape) = shapes.get(input).cloned() else {
                continue;
            };
            shapes.insert(output.clone(), shape);
            changed = true;
        }
    }
    shapes
}

fn attribute_int(node: &NodeProto, name: &str) -> Option<i64> {
    node.attribute
        .iter()
        .find(|attribute| attribute.name == name)
        .map(|attribute| attribute.i)
}

fn int64_initializer(name: &str, values: &[i64]) -> TensorProto {
    TensorProto {
        name: name.to_string(),
        data_type: tensor_proto::DataType::Int64 as i32,
        dims: vec![values.len() as i64],
        int64_data: values.to_vec(),
        ..Default::default()
    }
}

fn make_node(op_type: &str, inputs: &[&str], outputs: &[&str], name: &str) -> NodeProto {
    NodeProto {
        op_type: op_type.to_string(),
        name: name.to_string(),
        input: inputs.iter().map(|value| (*value).to_string()).collect(),
        output: outputs.iter().map(|value| (*value).to_string()).collect(),
        ..Default::default()
    }
}

fn arg_node(
    op_type: &str,
    input: &str,
    output: &str,
    name: &str,
    axis: i64,
    keepdims: i64,
    select_last_index: i64,
) -> NodeProto {
    make_node(op_type, &[input], &[output], name)
        .with_int_attribute("axis", axis)
        .with_int_attribute("keepdims", keepdims)
        .with_int_attribute("select_last_index", select_last_index)
}

trait WithIntAttribute {
    fn with_int_attribute(self, name: &str, value: i64) -> Self;
    /// Place the node in a non-default operator domain.
    fn with_domain(self, domain: &str) -> Self;
}

impl WithIntAttribute for NodeProto {
    fn with_int_attribute(mut self, name: &str, value: i64) -> Self {
        self.attribute
            .push(onnx_runtime_loader::proto::onnx::AttributeProto {
                name: name.to_string(),
                r#type: onnx_runtime_loader::proto::onnx::attribute_proto::AttributeType::Int
                    as i32,
                i: value,
                ..Default::default()
            });
        self
    }

    fn with_domain(mut self, domain: &str) -> Self {
        self.domain = domain.to_string();
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_loader::proto::onnx::{TensorShapeProto, TypeProto, ValueInfoProto};

    pub(super) fn tensor_input(name: &str, dims: &[Option<i64>]) -> ValueInfoProto {
        ValueInfoProto {
            name: name.to_string(),
            r#type: Some(TypeProto {
                denotation: String::new(),
                value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                    elem_type: tensor_proto::DataType::Float as i32,
                    shape: Some(TensorShapeProto {
                        dim: dims
                            .iter()
                            .map(|extent| tensor_shape_proto::Dimension {
                                value: Some(match extent {
                                    Some(extent) => {
                                        tensor_shape_proto::dimension::Value::DimValue(*extent)
                                    }
                                    None => tensor_shape_proto::dimension::Value::DimParam(
                                        "batch".to_string(),
                                    ),
                                }),
                                ..Default::default()
                            })
                            .collect(),
                    }),
                })),
            }),
            ..Default::default()
        }
    }

    pub(super) fn sampler_graph(dims: &[Option<i64>], keepdims: i64) -> GraphProto {
        GraphProto {
            name: "sampler".to_string(),
            input: vec![tensor_input("logits", dims)],
            node: vec![
                make_node("ArgMax", &["logits"], &["token"], "argmax")
                    .with_int_attribute("axis", -1)
                    .with_int_attribute("keepdims", keepdims),
            ],
            ..Default::default()
        }
    }

    #[test]
    fn tiling_factors_minimize_the_longest_serial_scan() {
        // 202048 = 2^6 * 7 * 11 * 41; the balanced split is 448 x 451.
        assert_eq!(plan_tiles(202_048), Some((448, 451)));
        assert_eq!(plan_tiles(1 << 16), Some((256, 256)));
    }

    #[test]
    fn small_or_unfactorable_extents_are_left_alone() {
        assert_eq!(plan_tiles(1_024), None);
        // A prime extent only factors as 1 x extent, which reduces nothing.
        assert_eq!(plan_tiles(200_003), None);
    }

    #[test]
    fn degenerate_sampler_argmax_is_tiled_into_two_stages() {
        let mut graph = sampler_graph(&[None, Some(202_048)], 0);
        assert_eq!(tile_degenerate_arg_reductions(&mut graph, false).tiled, 1);
        let ops = graph
            .node
            .iter()
            .map(|node| node.op_type.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            ops,
            [
                "Reshape",
                "ArgMax",
                "GatherElements",
                "Reshape",
                "Reshape",
                "ArgMax",
                "GatherElements",
                "Mul",
                "Add",
                "Reshape",
            ]
        );
        assert_eq!(graph.node.last().expect("output node").output, ["token"]);
        assert!(
            graph
                .initializer
                .iter()
                .any(|initializer| initializer.int64_data == [0, 448, 451])
        );
    }

    #[test]
    fn keepdims_output_skips_the_trailing_reshape() {
        let mut graph = sampler_graph(&[None, Some(202_048)], 1);
        assert_eq!(tile_degenerate_arg_reductions(&mut graph, false).tiled, 1);
        assert_eq!(graph.node.last().expect("output node").op_type, "Add");
        assert_eq!(graph.node.last().expect("output node").output, ["token"]);
    }

    #[test]
    fn symbolic_or_small_reductions_are_not_rewritten() {
        let mut symbolic = sampler_graph(&[None, None], 0);
        assert_eq!(
            tile_degenerate_arg_reductions(&mut symbolic, false).tiled,
            0
        );
        let mut small = sampler_graph(&[None, Some(1_024)], 0);
        assert_eq!(tile_degenerate_arg_reductions(&mut small, false).tiled, 0);
    }

    #[test]
    fn a_cast_in_front_of_the_reduction_stays_visible() {
        let mut graph = GraphProto {
            name: "sampler".to_string(),
            input: vec![tensor_input("raw", &[None, Some(202_048)])],
            node: vec![
                make_node("Cast", &["raw"], &["logits"], "cast").with_int_attribute("to", 1),
                make_node("ArgMax", &["logits"], &["token"], "argmax")
                    .with_int_attribute("axis", -1)
                    .with_int_attribute("keepdims", 0),
            ],
            ..Default::default()
        };
        assert_eq!(tile_degenerate_arg_reductions(&mut graph, false).tiled, 1);
    }
}

/// Behaviour of the optional fused-kernel substitution.
#[cfg(test)]
mod fused {
    use super::tests::{sampler_graph, tensor_input};
    use super::*;

    #[test]
    fn degenerate_argmax_becomes_a_single_custom_node() {
        let mut graph = sampler_graph(&[None, Some(202_048)], 0);
        let rewrites = tile_degenerate_arg_reductions(&mut graph, true);
        assert_eq!(rewrites, ArgReduceRewrites { tiled: 0, fused: 1 });
        assert_eq!(graph.node.len(), 1, "the fused form is one node, not nine");
        assert_eq!(graph.node[0].op_type, FUSED_OP);
        assert_eq!(graph.node[0].domain, FUSED_DOMAIN);
        assert_eq!(graph.node[0].input, vec!["logits".to_string()]);
        assert_eq!(graph.node[0].output, vec!["token".to_string()]);
        // The output keeps the original name, so the rest of the island is
        // untouched by the substitution.
        assert!(graph.initializer.is_empty());
    }

    #[test]
    fn keepdims_is_restored_by_a_reshape() {
        let mut graph = sampler_graph(&[None, Some(202_048)], 1);
        assert_eq!(tile_degenerate_arg_reductions(&mut graph, true).fused, 1);
        let ops = graph
            .node
            .iter()
            .map(|node| node.op_type.as_str())
            .collect::<Vec<_>>();
        assert_eq!(ops, vec![FUSED_OP, "Reshape"]);
        assert_eq!(graph.node[1].output, vec!["token".to_string()]);
        let shape = graph
            .initializer
            .iter()
            .find(|init| init.name.ends_with("__keep_shape"))
            .expect("keepdims reshape target");
        assert_eq!(shape.int64_data, vec![0, 1]);
    }

    #[test]
    fn argmin_stays_on_the_portable_expansion() {
        // The kernel is a maximum reduction, so a minimum must not be fused.
        let mut graph = GraphProto {
            name: "sampler".to_string(),
            input: vec![tensor_input("logits", &[None, Some(202_048)])],
            node: vec![
                make_node("ArgMin", &["logits"], &["token"], "argmin")
                    .with_int_attribute("axis", -1)
                    .with_int_attribute("keepdims", 0),
            ],
            ..Default::default()
        };
        let rewrites = tile_degenerate_arg_reductions(&mut graph, true);
        assert_eq!(rewrites, ArgReduceRewrites { tiled: 1, fused: 0 });
        assert!(graph.node.iter().all(|node| node.domain.is_empty()));
    }

    #[test]
    fn last_index_tie_breaking_stays_on_the_portable_expansion() {
        // The kernel keeps the lowest index on ties.
        let mut graph = GraphProto {
            name: "sampler".to_string(),
            input: vec![tensor_input("logits", &[None, Some(202_048)])],
            node: vec![
                make_node("ArgMax", &["logits"], &["token"], "argmax")
                    .with_int_attribute("axis", -1)
                    .with_int_attribute("keepdims", 0)
                    .with_int_attribute("select_last_index", 1),
            ],
            ..Default::default()
        };
        let rewrites = tile_degenerate_arg_reductions(&mut graph, true);
        assert_eq!(rewrites, ArgReduceRewrites { tiled: 1, fused: 0 });
    }

    #[test]
    fn narrow_reductions_are_left_alone_either_way() {
        // Below the tiling threshold ORT's own kernel is not degenerate, so
        // neither rewrite applies and the graph is untouched.
        let mut graph = sampler_graph(&[None, Some(1_024)], 0);
        assert_eq!(
            tile_degenerate_arg_reductions(&mut graph, true),
            ArgReduceRewrites::default()
        );
        assert_eq!(graph.node[0].op_type, "ArgMax");
    }
}

/// The element types the fused substitution accepts.
#[cfg(test)]
mod fused_element_types {
    use super::tests::tensor_input;
    use super::*;

    fn graph_with(
        elem: tensor_proto::DataType,
        cast_to: Option<tensor_proto::DataType>,
    ) -> GraphProto {
        let mut input = tensor_input("raw", &[None, Some(202_048)]);
        if let Some(type_proto::Value::TensorType(tensor)) =
            input.r#type.as_mut().and_then(|kind| kind.value.as_mut())
        {
            tensor.elem_type = elem as i32;
        }
        let mut node = vec![];
        let reduced = match cast_to {
            Some(to) => {
                node.push(
                    make_node("Cast", &["raw"], &["logits"], "cast")
                        .with_int_attribute("to", to as i64),
                );
                "logits"
            }
            None => "raw",
        };
        node.push(
            make_node("ArgMax", &[reduced], &["token"], "argmax")
                .with_int_attribute("axis", -1)
                .with_int_attribute("keepdims", 0),
        );
        GraphProto {
            name: "sampler".to_string(),
            input: vec![input],
            node,
            ..Default::default()
        }
    }

    #[test]
    fn float_inputs_are_fused() {
        for elem in [
            tensor_proto::DataType::Float,
            tensor_proto::DataType::Float16,
            tensor_proto::DataType::Bfloat16,
        ] {
            let mut graph = graph_with(elem, None);
            assert_eq!(
                tile_degenerate_arg_reductions(&mut graph, true).fused,
                1,
                "{elem:?} is a type the kernel implements"
            );
        }
    }

    #[test]
    fn other_numeric_inputs_take_the_portable_expansion() {
        // ArgMax is defined over every numeric type, but the kernel is not, and
        // an unsupported type would fail at the first run rather than at
        // session creation, where the island has no fallback left.
        for elem in [
            tensor_proto::DataType::Int32,
            tensor_proto::DataType::Int64,
            tensor_proto::DataType::Double,
            tensor_proto::DataType::Uint8,
        ] {
            let mut graph = graph_with(elem, None);
            assert_eq!(
                tile_degenerate_arg_reductions(&mut graph, true),
                ArgReduceRewrites { tiled: 1, fused: 0 },
                "{elem:?} must not be fused"
            );
        }
    }

    #[test]
    fn a_cast_retypes_the_reduction_input() {
        // The tiling pass deliberately sees through a Cast to find the reduced
        // shape, so the type check has to follow the Cast's target type rather
        // than the type it consumed - in both directions.
        let mut to_float = graph_with(
            tensor_proto::DataType::Float16,
            Some(tensor_proto::DataType::Float),
        );
        assert_eq!(tile_degenerate_arg_reductions(&mut to_float, true).fused, 1);
        let mut to_int = graph_with(
            tensor_proto::DataType::Float,
            Some(tensor_proto::DataType::Int32),
        );
        assert_eq!(
            tile_degenerate_arg_reductions(&mut to_int, true),
            ArgReduceRewrites { tiled: 1, fused: 0 }
        );
    }

    #[test]
    fn an_untyped_input_is_not_fused() {
        let mut graph = graph_with(tensor_proto::DataType::Float, None);
        graph.input[0].r#type = None;
        // Without a declared type the pass cannot know the kernel applies, and
        // the tiling needs the shape, so nothing is rewritten at all.
        assert_eq!(
            tile_degenerate_arg_reductions(&mut graph, true),
            ArgReduceRewrites::default()
        );
    }
}

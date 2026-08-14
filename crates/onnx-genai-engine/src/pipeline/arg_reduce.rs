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
//! The result is bit-exact, including tie-breaking. `ArgMax` with the default
//! `select_last_index=0` returns the first maximal index in a row; the outer
//! reduction then selects the first tile holding the running maximum, so the
//! recombined index is the first maximal index of the flat row. With
//! `select_last_index=1` both stages select the last, which recombines to the
//! last maximal flat index. `ArgMin` follows the same argument on minima.
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

/// Rewrite every degenerate last-axis arg-reduction in `graph`, returning how many
/// nodes were replaced.
pub(crate) fn tile_degenerate_arg_reductions(graph: &mut GraphProto) -> usize {
    let shapes = static_shapes(graph);
    let mut rewritten = 0;
    let mut index = 0;
    while index < graph.node.len() {
        let Some(plan) = plan_rewrite(&graph.node[index], &shapes) else {
            index += 1;
            continue;
        };
        let replacement = expand(&graph.node[index], &plan, &mut graph.initializer);
        let inserted = replacement.len();
        graph.node.splice(index..=index, replacement);
        index += inserted;
        rewritten += 1;
    }
    rewritten
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_loader::proto::onnx::{TensorShapeProto, TypeProto, ValueInfoProto};

    fn tensor_input(name: &str, dims: &[Option<i64>]) -> ValueInfoProto {
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

    fn sampler_graph(dims: &[Option<i64>], keepdims: i64) -> GraphProto {
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
        assert_eq!(tile_degenerate_arg_reductions(&mut graph), 1);
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
        assert_eq!(tile_degenerate_arg_reductions(&mut graph), 1);
        assert_eq!(graph.node.last().expect("output node").op_type, "Add");
        assert_eq!(graph.node.last().expect("output node").output, ["token"]);
    }

    #[test]
    fn symbolic_or_small_reductions_are_not_rewritten() {
        let mut symbolic = sampler_graph(&[None, None], 0);
        assert_eq!(tile_degenerate_arg_reductions(&mut symbolic), 0);
        let mut small = sampler_graph(&[None, Some(1_024)], 0);
        assert_eq!(tile_degenerate_arg_reductions(&mut small), 0);
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
        assert_eq!(tile_degenerate_arg_reductions(&mut graph), 1);
    }
}

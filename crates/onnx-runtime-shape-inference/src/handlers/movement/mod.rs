//! Data-movement rules: `Reshape`, `Transpose`, `Flatten`, `Squeeze`,
//! `Unsqueeze`, `Expand`, `Concat`, `Slice`, `Split`, `Gather`,
//! `GatherElements`, `ScatterND`, `ScatterElements`, `Scatter`, `Trilu`,
//! `DepthToSpace`, and `SpaceToDepth`.
//!
//! Several of these are *shape-data consumers* (`Reshape`/`Expand`/`Slice` read
//! a computed shape vector) and/or *shape-data transformers* (`Gather`/`Slice`/
//! `Concat`/`Squeeze`/`Unsqueeze` on a shape vector), which is what keeps a
//! `Shape → … → Reshape` chain resolvable without executing the graph.

use onnx_runtime_ir::Attribute;

use crate::context::InferenceContext;
use crate::error::ShapeInferError;
use crate::registry::InferenceRegistry;

/// Read a shape-data operand (input `i`) as concrete `i64`s, if every element is
/// a constant.
fn const_ints(ctx: &InferenceContext, i: usize) -> Option<Vec<i64>> {
    ctx.input_shape_data(i)?
        .elems
        .iter()
        .map(|e| e.as_const())
        .collect()
}

fn validate_vector_input(
    ctx: &InferenceContext,
    index: usize,
    op: &str,
) -> Result<(), ShapeInferError> {
    if ctx.has_input(index)
        && let Some(rank) = ctx.input_rank(index)
        && rank != 1
    {
        return Err(ShapeInferError::InvalidRank {
            op: op.into(),
            index,
            rank,
            detail: "input must be a 1-D tensor".into(),
        });
    }
    Ok(())
}

fn checked_extent(op: &str, value: i128) -> Result<i64, ShapeInferError> {
    if !(0..=isize::MAX as i128).contains(&value) {
        return Err(ShapeInferError::Invalid {
            op: op.into(),
            detail: format!("inferred extent {value} is outside 0..=isize::MAX"),
        });
    }
    Ok(value as i64)
}

mod concat_slice;
mod resize;
mod scatter;
mod space_depth;
mod split_gather;
mod transform;

use concat_slice::*;
use resize::*;
use scatter::*;
use space_depth::*;
use split_gather::*;
use transform::*;

/// Read an integer-list attribute.
fn attr_ints(ctx: &InferenceContext, name: &str) -> Option<Vec<i64>> {
    ctx.node
        .attr(name)
        .and_then(Attribute::as_ints)
        .map(<[i64]>::to_vec)
}

/// Register the data-movement family.
pub fn register(reg: &mut InferenceRegistry) {
    reg.register("", "Transpose", 1, transpose);
    reg.register("", "Reshape", 1, reshape);
    reg.register("", "Flatten", 1, flatten);
    // Squeeze/Unsqueeze moved axes from attribute to input at opset 13 — a real
    // range-based dispatch.
    reg.register("", "Squeeze", 1, squeeze_v1);
    reg.register("", "Squeeze", 13, squeeze_v13);
    reg.register("", "Unsqueeze", 1, unsqueeze_v1);
    reg.register("", "Unsqueeze", 13, unsqueeze_v13);
    reg.register("", "Expand", 8, expand);
    reg.register("", "Resize", 10, resize_v10);
    reg.register("", "Resize", 11, resize);
    reg.register("", "Concat", 1, concat);
    reg.register("", "Slice", 1, slice);
    reg.register("", "Split", 1, split);
    reg.register("", "Gather", 1, gather);
    reg.register("", "GatherElements", 1, gather_elements);
    reg.register("", "GatherND", 11, gather_nd);
    reg.register("", "GatherND", 12, gather_nd);
    reg.register("", "GatherND", 13, gather_nd);
    reg.register("", "Scatter", 9, scatter_elements);
    reg.register("", "ScatterElements", 11, scatter_elements);
    reg.register("", "ScatterElements", 13, scatter_elements);
    reg.register("", "ScatterElements", 16, scatter_elements);
    reg.register("", "ScatterND", 11, scatter_nd);
    reg.register("", "ScatterND", 13, scatter_nd);
    reg.register("", "ScatterND", 16, scatter_nd);
    reg.register("", "ScatterND", 18, scatter_nd);
    reg.register("", "Trilu", 14, trilu);
    // `ReverseSequence` (opset 10) permutes elements but preserves the input's
    // shape and dtype, so it reuses the elementwise same-shape rule.
    reg.register(
        "",
        "ReverseSequence",
        10,
        crate::handlers::elementwise::unary,
    );
    reg.register("", "DepthToSpace", 1, depth_to_space);
    reg.register("", "DepthToSpace", 11, depth_to_space);
    reg.register("", "DepthToSpace", 13, depth_to_space);
    reg.register("", "SpaceToDepth", 1, space_to_depth);
    reg.register("", "SpaceToDepth", 13, space_to_depth);
    // `Col2Im` (opset 18) folds columns back into an image; `CenterCropPad`
    // (opset 18) center-crops/pads selected axes to a target `shape`.
    reg.register("", "Col2Im", 18, col2im);
    reg.register("", "CenterCropPad", 18, center_crop_pad);
}

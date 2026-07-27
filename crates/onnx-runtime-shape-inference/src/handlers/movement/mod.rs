//! Data-movement rules: `Reshape`, `Transpose`, `Flatten`, `Squeeze`,
//! `Unsqueeze`, `Expand`, `Concat`, `Slice`, `Split`, `Gather`,
//! `GatherElements`, `ScatterND`, `ScatterElements`, `Scatter`, `Trilu`,
//! `DepthToSpace`, and `SpaceToDepth`.
//!
//! Several of these are *shape-data consumers* (`Reshape`/`Expand`/`Slice` read
//! a computed shape vector) and/or *shape-data transformers* (`Gather`/`Slice`/
//! `Concat`/`Squeeze`/`Unsqueeze` on a shape vector), which is what keeps a
//! `Shape → … → Reshape` chain resolvable without executing the graph.

use onnx_runtime_ir::{Attribute, DataType};

use crate::context::{InferenceContext, TypeInfo};
use crate::dim_expr::DimExpr;
use crate::error::ShapeInferError;
use crate::handlers::checked_axis;
use crate::registry::InferenceRegistry;
use crate::shape_data::ShapeData;

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
mod split_gather;
mod transform;

use concat_slice::*;
use resize::*;
use scatter::*;
use split_gather::*;
use transform::*;

fn blocksize(ctx: &InferenceContext, op: &str) -> Result<i64, ShapeInferError> {
    let value = ctx
        .node
        .attr("blocksize")
        .and_then(Attribute::as_int)
        .ok_or_else(|| ShapeInferError::MissingAttribute {
            op: op.into(),
            attr: "blocksize".into(),
        })?;
    if value < 1 {
        return Err(ShapeInferError::Invalid {
            op: op.into(),
            detail: format!("blocksize must be positive, found {value}"),
        });
    }
    Ok(value)
}

fn spatial_input(ctx: &InferenceContext, op: &str) -> Result<Option<TypeInfo>, ShapeInferError> {
    let Some(input) = ctx.input_type(0).cloned() else {
        return Ok(None);
    };
    if input.rank() != 4 {
        return Err(ShapeInferError::InvalidRank {
            op: op.into(),
            index: 0,
            rank: input.rank(),
            detail: "input must have shape [N, C, H, W]".into(),
        });
    }
    Ok(Some(input))
}

/// `DepthToSpace`: `[N,C,H,W]` becomes
/// `[N,C/(blocksize²),H*blocksize,W*blocksize]`.
pub fn depth_to_space(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(input) = spatial_input(ctx, "DepthToSpace")? else {
        return Ok(());
    };
    let blocksize = blocksize(ctx, "DepthToSpace")?;
    let block_area = blocksize
        .checked_mul(blocksize)
        .ok_or_else(|| ShapeInferError::Invalid {
            op: "DepthToSpace".into(),
            detail: format!("blocksize² overflows i64 for blocksize {blocksize}"),
        })?;
    if let Some(mode) = ctx.node.attr("mode") {
        let mode = mode.as_str().ok_or_else(|| ShapeInferError::Invalid {
            op: "DepthToSpace".into(),
            detail: "mode must be the string DCR or CRD".into(),
        })?;
        if !matches!(mode, "DCR" | "CRD") {
            return Err(ShapeInferError::Invalid {
                op: "DepthToSpace".into(),
                detail: format!("mode must be DCR or CRD, found {mode}"),
            });
        }
    }

    let channel = if let Some(channel) = input.shape[1].as_const() {
        if channel % block_area != 0 {
            return Err(ShapeInferError::Invalid {
                op: "DepthToSpace".into(),
                detail: format!(
                    "channel dimension {channel} is not divisible by blocksize² ({block_area})"
                ),
            });
        }
        DimExpr::constant(channel / block_area)
    } else {
        input.shape[1]
            .checked_div(&DimExpr::constant(block_area))
            .unwrap_or_else(|| ctx.fresh_dim())
    };
    let scale = DimExpr::constant(blocksize);
    let output = vec![
        input.shape[0].clone(),
        channel,
        input.shape[2].mul(&scale),
        input.shape[3].mul(&scale),
    ];
    ctx.set_output(0, input.dtype, output);
    Ok(())
}

/// `SpaceToDepth`: `[N,C,H,W]` becomes
/// `[N,C*blocksize²,H/blocksize,W/blocksize]`.
pub fn space_to_depth(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(input) = spatial_input(ctx, "SpaceToDepth")? else {
        return Ok(());
    };
    let blocksize = blocksize(ctx, "SpaceToDepth")?;
    let block_area = blocksize
        .checked_mul(blocksize)
        .ok_or_else(|| ShapeInferError::Invalid {
            op: "SpaceToDepth".into(),
            detail: format!("blocksize² overflows i64 for blocksize {blocksize}"),
        })?;
    for (axis, name) in [(2, "height"), (3, "width")] {
        if let Some(extent) = input.shape[axis].as_const()
            && extent % blocksize != 0
        {
            return Err(ShapeInferError::Invalid {
                op: "SpaceToDepth".into(),
                detail: format!(
                    "{name} dimension {extent} is not divisible by blocksize ({blocksize})"
                ),
            });
        }
    }

    let divisor = DimExpr::constant(blocksize);
    let divide = |dim: &DimExpr, ctx: &mut InferenceContext| {
        dim.checked_div(&divisor).unwrap_or_else(|| ctx.fresh_dim())
    };
    let height = divide(&input.shape[2], ctx);
    let width = divide(&input.shape[3], ctx);
    let output = vec![
        input.shape[0].clone(),
        input.shape[1].mul(&DimExpr::constant(block_area)),
        height,
        width,
    ];
    ctx.set_output(0, input.dtype, output);
    Ok(())
}

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
    reg.register("", "DepthToSpace", 1, depth_to_space);
    reg.register("", "DepthToSpace", 11, depth_to_space);
    reg.register("", "DepthToSpace", 13, depth_to_space);
    reg.register("", "SpaceToDepth", 1, space_to_depth);
    reg.register("", "SpaceToDepth", 13, space_to_depth);
}

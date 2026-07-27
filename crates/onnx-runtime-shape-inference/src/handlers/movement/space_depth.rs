use onnx_runtime_ir::Attribute;

use crate::context::{InferenceContext, TypeInfo};
use crate::dim_expr::DimExpr;
use crate::error::ShapeInferError;

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

use onnx_runtime_ir::space_depth::{AxisScale, SpaceDepth};

use crate::context::InferenceContext;
use crate::dim_expr::DimExpr;
use crate::error::ShapeInferError;

pub fn infer_space_depth(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let invalid = |detail| ShapeInferError::Invalid {
        op: ctx.node.op_type.clone(),
        detail,
    };
    let params = SpaceDepth::parse(ctx.node, ctx.opset(&ctx.node.domain)).map_err(invalid)?;
    let Some(input) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    if input.rank() != 4 {
        return Err(ShapeInferError::InvalidRank {
            op: ctx.node.op_type.clone(),
            index: 0,
            rank: input.rank(),
            detail: SpaceDepth::rank_error(input.rank()),
        });
    }

    let mut output = Vec::with_capacity(4);
    for (axis, (dim, scale)) in input
        .shape
        .into_iter()
        .zip(params.axis_scales())
        .enumerate()
    {
        let dim = if let Some(value) = dim.as_const() {
            DimExpr::constant(scale.apply(axis, value).map_err(invalid)?)
        } else {
            match scale {
                AxisScale::Keep => dim,
                AxisScale::Multiply(factor) => dim.mul(&DimExpr::constant(factor)),
                AxisScale::Divide(divisor) => dim
                    .checked_div(&DimExpr::constant(divisor))
                    .unwrap_or_else(|| ctx.fresh_dim()),
            }
        };
        output.push(dim);
    }
    ctx.set_output(0, input.dtype, output);
    Ok(())
}

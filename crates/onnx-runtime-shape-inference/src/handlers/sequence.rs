//! Sequence-construction and scan rules: `Tile`, `Range`, and `CumSum`.

use onnx_runtime_ir::Attribute;

use crate::context::InferenceContext;
use crate::dim_expr::DimExpr;
use crate::error::ShapeInferError;
use crate::handlers::norm_axis;
use crate::registry::InferenceRegistry;

fn const_ints(ctx: &InferenceContext, input: usize) -> Option<Vec<i64>> {
    ctx.input_shape_data(input)?
        .elems
        .iter()
        .map(DimExpr::as_const)
        .collect()
}

/// Legacy `Tile` (opset 1) repeats `tiles` times along a single `axis`.
fn tile_v1(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(input) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    if input.shape.is_empty() {
        return Ok(());
    }
    let axis = ctx
        .node
        .attr("axis")
        .and_then(Attribute::as_int)
        .unwrap_or(0);
    let axis = norm_axis(axis, input.rank());
    let tiles = ctx.node.attr("tiles").and_then(Attribute::as_int);
    let mut output = input.shape;
    let input_dim = output[axis].clone();
    output[axis] = tiles
        .filter(|&n| n >= 0)
        .map(|n| input_dim.mul(&DimExpr::constant(n)))
        .unwrap_or_else(|| ctx.fresh_dim());
    ctx.set_output(0, input.dtype, output);
    Ok(())
}

/// `Tile`: multiply every input extent by the corresponding static repeat.
fn tile(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(input) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    let repeats = const_ints(ctx, 1);
    let mut output = Vec::with_capacity(input.rank());
    for (i, dim) in input.shape.iter().enumerate() {
        output.push(
            repeats
                .as_ref()
                .filter(|r| r.len() == input.rank())
                .and_then(|r| r.get(i))
                .filter(|&&n| n >= 0)
                .map(|&n| dim.mul(&DimExpr::constant(n)))
                .unwrap_or_else(|| ctx.fresh_dim()),
        );
    }
    ctx.set_output(0, input.dtype, output);
    Ok(())
}

/// Compute a `Range` length for integer scalar inputs, if all values are known.
fn range_len(start: i64, limit: i64, delta: i64) -> Option<i64> {
    if delta == 0 {
        return None;
    }
    let start = i128::from(start);
    let limit = i128::from(limit);
    let delta = i128::from(delta);
    let distance = limit - start;
    let count = if (distance > 0 && delta > 0) || (distance < 0 && delta < 0) {
        let distance = distance.abs();
        let step = delta.abs();
        (distance + step - 1) / step
    } else {
        0
    };
    i64::try_from(count).ok()
}

/// `Range`: output is a one-dimensional tensor with a static length only when
/// all scalar operands are statically known.
fn range(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(dtype) = ctx.input_dtype(0) else {
        return Ok(());
    };
    let length = match (const_ints(ctx, 0), const_ints(ctx, 1), const_ints(ctx, 2)) {
        (Some(start), Some(limit), Some(delta))
            if start.len() == 1 && limit.len() == 1 && delta.len() == 1 =>
        {
            range_len(start[0], limit[0], delta[0])
        }
        _ => None,
    };
    let dim = length
        .map(DimExpr::constant)
        .unwrap_or_else(|| ctx.fresh_dim());
    ctx.set_output(0, dtype, vec![dim]);
    Ok(())
}

/// `CumSum` does not change the input tensor's shape or dtype.
fn cum_sum(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    if let Some(input) = ctx.input_type(0).cloned() {
        ctx.set_output_type(0, input);
    }
    Ok(())
}

/// Register sequence-construction and scan rules.
pub fn register(reg: &mut InferenceRegistry) {
    reg.register("", "Tile", 1, tile_v1);
    reg.register("", "Tile", 6, tile);
    reg.register("", "Tile", 13, tile);
    reg.register("", "Range", 11, range);
    reg.register("", "CumSum", 11, cum_sum);
    reg.register("", "CumSum", 14, cum_sum);
}

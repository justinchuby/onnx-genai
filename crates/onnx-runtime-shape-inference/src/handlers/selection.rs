//! Value-selection rules: `ArgMax`, `ArgMin`, `TopK`, and `NonZero`.

use onnx_runtime_ir::{Attribute, DataType};

use crate::context::InferenceContext;
use crate::dim_expr::DimExpr;
use crate::error::ShapeInferError;
use crate::handlers::norm_axis;
use crate::registry::InferenceRegistry;

/// `ArgMax`/`ArgMin`: replace or remove the selected axis and return int64.
fn arg_reduce(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(input) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    if input.is_empty() {
        return Ok(());
    }
    let axis = ctx
        .node
        .attr("axis")
        .and_then(Attribute::as_int)
        .unwrap_or(0);
    let axis = norm_axis(axis, input.len());
    let keepdims = ctx
        .node
        .attr("keepdims")
        .and_then(Attribute::as_int)
        .unwrap_or(1)
        != 0;
    let mut output = input;
    if keepdims {
        output[axis] = DimExpr::constant(1);
    } else {
        output.remove(axis);
    }
    ctx.set_output(0, DataType::Int64, output);
    Ok(())
}

/// Read a statically-known scalar integer operand.
fn scalar_int(ctx: &InferenceContext, input: usize) -> Option<i64> {
    let data = ctx.input_shape_data(input)?;
    if data.elems.len() != 1 {
        return None;
    }
    data.elems.first()?.as_const()
}

/// `TopK`: values retain the data dtype and indices are int64, with the selected
/// axis replaced by `K`.
fn top_k_v1(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let k = ctx.node.attr("k").and_then(Attribute::as_int);
    top_k(ctx, k)
}

fn top_k_v10(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    top_k(ctx, scalar_int(ctx, 1))
}

fn top_k(ctx: &mut InferenceContext, k: Option<i64>) -> Result<(), ShapeInferError> {
    let Some(input) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    let Some(dtype) = ctx.input_dtype(0) else {
        return Ok(());
    };
    if input.is_empty() {
        return Ok(());
    }
    let axis = ctx
        .node
        .attr("axis")
        .and_then(Attribute::as_int)
        .unwrap_or(-1);
    let axis = norm_axis(axis, input.len());
    let mut output = input;
    output[axis] = k
        .filter(|&k| k >= 0)
        .map(DimExpr::constant)
        .unwrap_or_else(|| ctx.fresh_dim());
    ctx.set_output(0, dtype, output.clone());
    ctx.set_output(1, DataType::Int64, output);
    Ok(())
}

/// `NonZero`: its second dimension is data-dependent, while the first is the
/// statically-known rank of the input.
fn non_zero(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(rank) = ctx.input_rank(0) else {
        return Ok(());
    };
    let nnz = ctx.fresh_dim();
    ctx.set_output(
        0,
        DataType::Int64,
        vec![DimExpr::constant(rank as i64), nnz],
    );
    Ok(())
}

/// Register selection-family rules.
pub fn register(reg: &mut InferenceRegistry) {
    for op in ["ArgMax", "ArgMin"] {
        reg.register("", op, 1, arg_reduce);
        reg.register("", op, 11, arg_reduce);
        reg.register("", op, 12, arg_reduce);
        reg.register("", op, 13, arg_reduce);
    }
    reg.register("", "TopK", 1, top_k_v1);
    reg.register("", "TopK", 10, top_k_v10);
    reg.register("", "TopK", 11, top_k_v10);
    reg.register("", "NonZero", 9, non_zero);
    reg.register("", "NonZero", 13, non_zero);
}

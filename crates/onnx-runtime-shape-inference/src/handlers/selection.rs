//! Value-selection rules: `ArgMax`, `ArgMin`, `TopK`, `NonZero`, `OneHot`,
//! and `Compress`.

use onnx_runtime_ir::{Attribute, DataType};

use crate::context::InferenceContext;
use crate::dim_expr::DimExpr;
use crate::error::ShapeInferError;
use crate::handlers::checked_axis;
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
    let rank = input.len();
    let axis = checked_axis(axis, rank).ok_or_else(|| ShapeInferError::Invalid {
        op: ctx.op().into(),
        detail: format!("axis {axis} is out of range for rank {rank}"),
    })?;
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
    let rank = input.len();
    let axis = checked_axis(axis, rank).ok_or_else(|| ShapeInferError::Invalid {
        op: "TopK".into(),
        detail: format!("axis {axis} is out of range for rank {rank}"),
    })?;
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
    let rank = i64::try_from(rank).map_err(|_| ShapeInferError::Invalid {
        op: "NonZero".into(),
        detail: "input rank exceeds the supported integer range".into(),
    })?;
    let nnz = ctx.fresh_dim();
    ctx.set_output(0, DataType::Int64, vec![DimExpr::constant(rank), nnz]);
    Ok(())
}

/// `NonMaxSuppression` emits one data-dependent `[batch, class, box]` row per
/// selected box.
fn non_max_suppression(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let selected = ctx.fresh_dim();
    ctx.set_output(0, DataType::Int64, vec![selected, DimExpr::constant(3)]);
    Ok(())
}

/// `OneHot`: insert the scalar `depth` extent into the indices shape. The
/// output element type comes from the two-element `values` input.
fn one_hot(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(indices) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    let Some(dtype) = ctx.input_dtype(2) else {
        return Ok(());
    };
    if let Some(depth_shape) = ctx.input_shape(1)
        && !depth_shape.is_empty()
    {
        return Err(ShapeInferError::InvalidRank {
            op: "OneHot".into(),
            index: 1,
            rank: depth_shape.len(),
            detail: "depth must be a scalar".into(),
        });
    }
    if let Some(values_shape) = ctx.input_shape(2) {
        if values_shape.len() != 1 {
            return Err(ShapeInferError::InvalidRank {
                op: "OneHot".into(),
                index: 2,
                rank: values_shape.len(),
                detail: "values must be a rank-1 tensor of length 2".into(),
            });
        }
        if let Some(len) = values_shape[0].as_const()
            && len != 2
        {
            return Err(ShapeInferError::Invalid {
                op: "OneHot".into(),
                detail: format!("values length must be 2, found {len}"),
            });
        }
    }

    let depth = match ctx
        .input_shape_data(1)
        .filter(|data| data.is_scalar())
        .and_then(|data| data.elems.first())
        .cloned()
    {
        Some(depth) => {
            if let Some(value) = depth.as_const()
                && value < 1
            {
                return Err(ShapeInferError::Invalid {
                    op: "OneHot".into(),
                    detail: format!("depth must be positive, found {value}"),
                });
            }
            depth
        }
        None => ctx.fresh_dim(),
    };

    let output_rank = indices.len() + 1;
    let axis_attr = ctx
        .node
        .attr("axis")
        .and_then(Attribute::as_int)
        .unwrap_or(-1);
    let axis = if axis_attr < 0 {
        axis_attr + output_rank as i64
    } else {
        axis_attr
    };
    if !(0..output_rank as i64).contains(&axis) {
        return Err(ShapeInferError::Invalid {
            op: "OneHot".into(),
            detail: format!(
                "axis {axis_attr} is outside [-{output_rank}, {})",
                output_rank
            ),
        });
    }

    let mut output = indices;
    output.insert(axis as usize, depth);
    ctx.set_output(0, dtype, output);
    Ok(())
}

/// `Compress`: the selected element count is data-dependent. With an axis, only
/// that extent becomes unknown; without one, the input is flattened first.
fn compress(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(data) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    if let Some(condition) = ctx.input_shape(1)
        && condition.len() != 1
    {
        return Err(ShapeInferError::InvalidRank {
            op: "Compress".into(),
            index: 1,
            rank: condition.len(),
            detail: "condition must be rank 1".into(),
        });
    }

    let Some(axis_attr) = ctx.node.attr("axis").and_then(Attribute::as_int) else {
        let selected = ctx.fresh_dim();
        ctx.set_output(0, data.dtype, vec![selected]);
        return Ok(());
    };
    let rank = data.rank();
    let axis = if axis_attr < 0 {
        axis_attr + rank as i64
    } else {
        axis_attr
    };
    if !(0..rank as i64).contains(&axis) {
        return Err(ShapeInferError::Invalid {
            op: "Compress".into(),
            detail: format!("axis {axis_attr} is outside [-{rank}, {rank})"),
        });
    }
    let mut output = data.shape;
    output[axis as usize] = ctx.fresh_dim();
    ctx.set_output(0, data.dtype, output);
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
    reg.register("", "NonMaxSuppression", 10, non_max_suppression);
    reg.register("", "OneHot", 9, one_hot);
    reg.register("", "OneHot", 11, one_hot);
    reg.register("", "Compress", 9, compress);
    reg.register("", "Compress", 11, compress);
}

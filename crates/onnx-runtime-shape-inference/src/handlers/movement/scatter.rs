use onnx_runtime_ir::Attribute;

use crate::context::InferenceContext;
use crate::dim_expr::DimExpr;
use crate::error::ShapeInferError;
use crate::handlers::checked_axis;

/// `ScatterElements` and deprecated `Scatter`: output type and shape are those
/// of the data input. Axis and reduction attributes do not affect inference.
pub fn scatter_elements(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(data) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    if data.rank() == 0 {
        return Err(ShapeInferError::InvalidRank {
            op: ctx.op().into(),
            index: 0,
            rank: 0,
            detail: "data must have rank at least 1".into(),
        });
    }
    let axis = ctx
        .node
        .attr("axis")
        .and_then(Attribute::as_int)
        .unwrap_or(0);
    checked_axis(axis, data.rank()).ok_or_else(|| ShapeInferError::Invalid {
        op: ctx.op().into(),
        detail: format!("axis {axis} is out of range for rank {}", data.rank()),
    })?;
    for index in [1, 2] {
        if let Some(rank) = ctx.input_rank(index)
            && rank != data.rank()
        {
            return Err(ShapeInferError::InvalidRank {
                op: ctx.op().into(),
                index,
                rank,
                detail: format!("input must have the same rank {} as data", data.rank()),
            });
        }
    }
    ctx.set_output_type(0, data);
    Ok(())
}

/// `ScatterND`: output type and shape are those of the data input.
pub fn scatter_nd(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(data) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    if data.rank() == 0 {
        return Err(ShapeInferError::InvalidRank {
            op: "ScatterND".into(),
            index: 0,
            rank: 0,
            detail: "data must have rank at least 1".into(),
        });
    }
    let Some(indices) = ctx.input_shape(1) else {
        ctx.set_output_type(0, data);
        return Ok(());
    };
    if indices.is_empty() {
        return Err(ShapeInferError::InvalidRank {
            op: "ScatterND".into(),
            index: 1,
            rank: 0,
            detail: "indices must have rank at least 1".into(),
        });
    }
    if let Some(index_depth) = indices.last().and_then(DimExpr::as_const) {
        let index_depth = usize::try_from(index_depth).map_err(|_| ShapeInferError::Invalid {
            op: "ScatterND".into(),
            detail: format!("indices last dimension must be non-negative, found {index_depth}"),
        })?;
        if index_depth > data.rank() {
            return Err(ShapeInferError::Invalid {
                op: "ScatterND".into(),
                detail: format!(
                    "indices last dimension {index_depth} exceeds data rank {}",
                    data.rank()
                ),
            });
        }
        if let Some(updates_rank) = ctx.input_rank(2) {
            let expected = indices
                .len()
                .checked_add(data.rank())
                .and_then(|rank| rank.checked_sub(index_depth))
                .and_then(|rank| rank.checked_sub(1))
                .filter(|&rank| rank <= isize::MAX as usize)
                .ok_or_else(|| ShapeInferError::Invalid {
                    op: "ScatterND".into(),
                    detail: "updates rank arithmetic overflowed".into(),
                })?;
            if updates_rank != expected {
                return Err(ShapeInferError::InvalidRank {
                    op: "ScatterND".into(),
                    index: 2,
                    rank: updates_rank,
                    detail: format!("updates rank must be {expected}"),
                });
            }
        }
    }
    ctx.set_output_type(0, data);
    Ok(())
}

/// `Trilu`: selecting a triangular region does not change the input type.
pub fn trilu(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(input) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    if input.rank() < 2 {
        return Err(ShapeInferError::InvalidRank {
            op: "Trilu".into(),
            index: 0,
            rank: input.rank(),
            detail: "input must be a matrix or a batch of matrices".into(),
        });
    }
    if ctx.has_input(1)
        && let Some(k_shape) = ctx.input_shape(1)
        && !k_shape.is_empty()
    {
        return Err(ShapeInferError::InvalidRank {
            op: "Trilu".into(),
            index: 1,
            rank: k_shape.len(),
            detail: "k must be a scalar".into(),
        });
    }
    ctx.set_output_type(0, input);
    Ok(())
}

/// `TensorScatter` (opset 24): the cache keeps its shape, so the output type
/// and shape are those of `past_cache`. `update` differs only in the sequence
/// dimension named by `axis`, which lets us validate the static parts of that
/// contract here rather than at execute time.
pub fn tensor_scatter(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(past_cache) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    // The op needs a batch dimension and a sequence dimension after it, and
    // `write_indices` is indexed by the batch coordinate.
    if past_cache.rank() < 2 {
        return Err(ShapeInferError::InvalidRank {
            op: "TensorScatter".into(),
            index: 0,
            rank: past_cache.rank(),
            detail: "past_cache must have rank at least 2 (batch and sequence)".into(),
        });
    }
    let axis = ctx
        .node
        .attr("axis")
        .and_then(Attribute::as_int)
        .unwrap_or(-2);
    let axis = checked_axis(axis, past_cache.rank()).ok_or_else(|| ShapeInferError::Invalid {
        op: "TensorScatter".into(),
        detail: format!("axis {axis} is out of range for rank {}", past_cache.rank()),
    })?;
    if axis == 0 {
        return Err(ShapeInferError::Invalid {
            op: "TensorScatter".into(),
            detail: "axis must not select the batch dimension".into(),
        });
    }
    if let Some(update_rank) = ctx.input_rank(1)
        && update_rank != past_cache.rank()
    {
        return Err(ShapeInferError::InvalidRank {
            op: "TensorScatter".into(),
            index: 1,
            rank: update_rank,
            detail: format!(
                "update must have the same rank {} as past_cache",
                past_cache.rank()
            ),
        });
    }
    if let Some(update) = ctx.input_shape(1).map(<[DimExpr]>::to_vec)
        && update.len() == past_cache.rank()
    {
        let cache_shape = past_cache.shape.clone();
        for dimension in 0..past_cache.rank() {
            let (Some(cache_dim), Some(update_dim)) = (
                cache_shape[dimension].as_const(),
                update[dimension].as_const(),
            ) else {
                continue;
            };
            if dimension == axis {
                // The update writes a window into a fixed-capacity buffer.
                if update_dim > cache_dim {
                    return Err(ShapeInferError::Invalid {
                        op: "TensorScatter".into(),
                        detail: format!(
                            "update sequence length {update_dim} exceeds cache capacity \
                             {cache_dim} at axis {axis}"
                        ),
                    });
                }
            } else if update_dim != cache_dim {
                return Err(ShapeInferError::Invalid {
                    op: "TensorScatter".into(),
                    detail: format!(
                        "update dimension {update_dim} at index {dimension} must match \
                         past_cache dimension {cache_dim}; only the sequence axis {axis} may differ"
                    ),
                });
            }
        }
    }
    ctx.set_output_type(0, past_cache);
    Ok(())
}

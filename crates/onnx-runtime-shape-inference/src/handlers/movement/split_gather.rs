use onnx_runtime_ir::{Attribute, DataType};

use crate::context::InferenceContext;
use crate::dim_expr::DimExpr;
use crate::error::ShapeInferError;
use crate::handlers::checked_axis;
use crate::shape_data::ShapeData;

use super::const_ints;

/// `Split`: divide the input along `axis` into the requested sizes (or equally).
pub fn split(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(t) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    let rank = t.rank();
    if rank == 0 {
        return Err(ShapeInferError::InvalidRank {
            op: "Split".into(),
            index: 0,
            rank,
            detail: "input must have rank at least 1".into(),
        });
    }
    let axis = ctx
        .node
        .attr("axis")
        .and_then(Attribute::as_int)
        .unwrap_or(0);
    let axis = checked_axis(axis, rank).ok_or_else(|| ShapeInferError::Invalid {
        op: "Split".into(),
        detail: format!("axis {axis} is outside [-{rank}, {rank})"),
    })?;
    let n_out = ctx.num_outputs();
    let raw_num_outputs = ctx.node.attr("num_outputs").and_then(Attribute::as_int);
    if let Some(n) = raw_num_outputs
        && n <= 0
    {
        return Err(ShapeInferError::Invalid {
            op: "Split".into(),
            detail: format!("num_outputs must be positive, got {n}"),
        });
    }
    let num_outputs = raw_num_outputs.and_then(|n| usize::try_from(n).ok());

    let sizes: Option<Vec<i64>> = ctx
        .node
        .attr("split")
        .and_then(Attribute::as_ints)
        .map(<[i64]>::to_vec)
        .or_else(|| const_ints(ctx, 1));
    let has_dynamic_split = sizes.is_none() && ctx.has_input(1);
    if let (Some(sizes), Some(num_outputs)) = (&sizes, num_outputs) {
        if sizes.len() != n_out {
            return Err(ShapeInferError::Invalid {
                op: "Split".into(),
                detail: format!(
                    "split provides {} sizes but the node has {n_out} outputs",
                    sizes.len()
                ),
            });
        }
        if num_outputs != n_out {
            return Err(ShapeInferError::Invalid {
                op: "Split".into(),
                detail: format!("num_outputs is {num_outputs} but the node has {n_out} outputs"),
            });
        }
        return Err(ShapeInferError::Invalid {
            op: "Split".into(),
            detail: "split input and num_outputs cannot both be specified".into(),
        });
    }
    if let Some(sizes) = &sizes {
        if sizes.len() != n_out {
            return Err(ShapeInferError::Invalid {
                op: "Split".into(),
                detail: format!(
                    "split provides {} sizes but the node has {n_out} outputs",
                    sizes.len()
                ),
            });
        }
        let total = sizes.iter().try_fold(0_i128, |total, &size| {
            if size < 0 {
                None
            } else {
                total.checked_add(i128::from(size))
            }
        });
        if let (Some(total), Some(extent)) = (total, t.shape[axis].as_const())
            && total != i128::from(extent)
        {
            return Err(ShapeInferError::Invalid {
                op: "Split".into(),
                detail: format!("split sizes sum to {total}, but axis extent is {extent}"),
            });
        }
    }
    if let Some(num_outputs) = num_outputs
        && num_outputs != n_out
    {
        return Err(ShapeInferError::Invalid {
            op: "Split".into(),
            detail: format!("num_outputs is {num_outputs} but the node has {n_out} outputs"),
        });
    }

    for i in 0..n_out {
        let mut shape = t.shape.clone();
        shape[axis] = match (&sizes, has_dynamic_split) {
            (Some(s), _) => match s.get(i).copied() {
                Some(v) if v < 0 => {
                    return Err(ShapeInferError::Invalid {
                        op: "Split".into(),
                        detail: format!("split size at index {i} is negative: {v}"),
                    });
                }
                Some(v) if usize::try_from(v).is_err() || v as u128 > isize::MAX as u128 => {
                    return Err(ShapeInferError::Invalid {
                        op: "Split".into(),
                        detail: format!("split size at index {i} exceeds isize::MAX: {v}"),
                    });
                }
                Some(v) => DimExpr::constant(v),
                None => ctx.fresh_dim(),
            },
            (None, true) => ctx.fresh_dim(),
            (None, false) => {
                match (num_outputs, t.shape[axis].as_const()) {
                    // With opset-18 `num_outputs`, ONNX gives every output but
                    // the last ceil(dim / n) elements; the last gets the
                    // remainder. This differs from the older equal-split path.
                    (Some(n), Some(d)) if i < n => {
                        let n = i64::try_from(n).map_err(|_| ShapeInferError::Invalid {
                            op: "Split".into(),
                            detail: "num_outputs exceeds the supported integer range".into(),
                        })?;
                        let chunk = d
                            .checked_add(n - 1)
                            .and_then(|numerator| numerator.checked_div(n))
                            .ok_or_else(|| ShapeInferError::Invalid {
                                op: "Split".into(),
                                detail: "split chunk arithmetic overflowed".into(),
                            })?;
                        let remainder = (n - 1)
                            .checked_mul(chunk)
                            .and_then(|used| d.checked_sub(used))
                            .ok_or_else(|| ShapeInferError::Invalid {
                                op: "Split".into(),
                                detail: "split remainder arithmetic overflowed".into(),
                            })?;
                        if remainder < 0 {
                            return Err(ShapeInferError::Invalid {
                                op: "Split".into(),
                                detail: format!(
                                    "cannot split axis extent {d} into {n} parts: \
                                     the even chunk size {chunk} leaves a negative final remainder"
                                ),
                            });
                        }
                        let size = if i + 1 == n as usize {
                            remainder
                        } else {
                            chunk
                        };
                        DimExpr::constant(size)
                    }
                    // The legacy no-`split` form is only exact when divisible.
                    (None, Some(d)) if n_out > 0 => {
                        let n_out = i64::try_from(n_out).map_err(|_| ShapeInferError::Invalid {
                            op: "Split".into(),
                            detail: "output count exceeds the supported integer range".into(),
                        })?;
                        if d % n_out == 0 {
                            DimExpr::constant(d / n_out)
                        } else {
                            ctx.fresh_dim()
                        }
                    }
                    _ => ctx.fresh_dim(),
                }
            }
        };
        ctx.set_output(i, t.dtype, shape);
    }
    Ok(())
}

/// `Gather`: `data[:axis] + indices.shape + data[axis+1:]`.
pub fn gather(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(data) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    let dtype = ctx.input_dtype(0).unwrap_or(DataType::Float32);
    let rank = data.len();
    if rank == 0 {
        return Err(ShapeInferError::InvalidRank {
            op: "Gather".into(),
            index: 0,
            rank,
            detail: "Gather data must have rank ≥ 1".into(),
        });
    }
    let axis = ctx
        .node
        .attr("axis")
        .and_then(Attribute::as_int)
        .unwrap_or(0);
    let axis = checked_axis(axis, rank).ok_or_else(|| ShapeInferError::Invalid {
        op: "Gather".into(),
        detail: format!("axis {axis} is out of range for rank {rank}"),
    })?;
    let idx_shape = ctx
        .input_shape(1)
        .map(<[DimExpr]>::to_vec)
        .unwrap_or_default();

    let mut out = Vec::with_capacity(rank - 1 + idx_shape.len());
    out.extend_from_slice(&data[..axis]);
    out.extend(idx_shape.iter().cloned());
    out.extend_from_slice(&data[axis + 1..]);
    ctx.set_output(0, dtype, out);

    // Shape-data: gathering elements of a 1-D shape vector on axis 0.
    if axis == 0
        && let Some(sd) = gather_shape_data(ctx)
    {
        ctx.set_output_shape_data(0, sd);
    }
    Ok(())
}

/// Gather elements of a 1-D shape-data vector at concrete indices.
fn gather_shape_data(ctx: &InferenceContext) -> Option<ShapeData> {
    let sd = ctx.input_shape_data(0)?;
    if sd.is_scalar() {
        return None;
    }
    let idx = ctx.input_shape_data(1)?;
    let idx_ints: Vec<i64> = idx
        .elems
        .iter()
        .map(|e| e.as_const())
        .collect::<Option<_>>()?;
    let n = sd.elems.len() as i64;
    let pick = |i: i64| -> Option<DimExpr> {
        let i = if i < 0 { i + n } else { i };
        sd.elems.get(i as usize).cloned()
    };
    let elems: Vec<DimExpr> = idx_ints.iter().map(|&i| pick(i)).collect::<Option<_>>()?;
    if idx.is_scalar() {
        Some(ShapeData::scalar(sd.dtype, elems.into_iter().next()?))
    } else {
        Some(ShapeData::vector(sd.dtype, elems))
    }
}

/// `GatherElements`: the output shape follows the indices tensor; dtype of the
/// data. (Not `GatherND` — this is the elementwise gather whose output rank
/// equals the indices' rank.)
pub fn gather_elements(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let data_rank = ctx.input_rank(0);
    let indices_rank = ctx.input_rank(1);
    if let Some(rank) = data_rank {
        if rank == 0 {
            return Err(ShapeInferError::InvalidRank {
                op: "GatherElements".into(),
                index: 0,
                rank,
                detail: "data must have rank at least 1".into(),
            });
        }
        let axis = ctx
            .node
            .attr("axis")
            .and_then(Attribute::as_int)
            .unwrap_or(0);
        if checked_axis(axis, rank).is_none() {
            return Err(ShapeInferError::Invalid {
                op: "GatherElements".into(),
                detail: format!("axis {axis} is outside [-{rank}, {rank})"),
            });
        }
    }
    if let (Some(data_rank), Some(indices_rank)) = (data_rank, indices_rank)
        && data_rank != indices_rank
    {
        return Err(ShapeInferError::InvalidRank {
            op: "GatherElements".into(),
            index: 1,
            rank: indices_rank,
            detail: format!("indices rank must equal data rank {data_rank}"),
        });
    }
    let dtype = ctx.input_dtype(0);
    let idx_shape = ctx.input_shape(1).map(<[DimExpr]>::to_vec);
    if let (Some(dtype), Some(shape)) = (dtype, idx_shape) {
        ctx.set_output(0, dtype, shape);
    }
    Ok(())
}

/// `GatherND`: `data[:batch_dims] + indices[batch_dims:-1] +
/// data[batch_dims + indices[-1]:]`.
pub fn gather_nd(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(data) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    let Some(indices) = ctx.input_shape(1).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    let Some(dtype) = ctx.input_dtype(0) else {
        return Ok(());
    };
    if data.is_empty() {
        return Err(ShapeInferError::InvalidRank {
            op: "GatherND".into(),
            index: 0,
            rank: 0,
            detail: "data must have rank at least 1".into(),
        });
    }
    if indices.is_empty() {
        return Err(ShapeInferError::InvalidRank {
            op: "GatherND".into(),
            index: 1,
            rank: 0,
            detail: "indices must have rank at least 1".into(),
        });
    }
    let Some(index_depth) = indices.last().and_then(DimExpr::as_const) else {
        // The index-tuple depth determines the output rank. Without it, retain
        // the crate's unknown-rank representation (no TypeInfo).
        return Ok(());
    };
    let batch_dims = ctx
        .node
        .attr("batch_dims")
        .and_then(Attribute::as_int)
        .unwrap_or(0);
    let batch_dims = usize::try_from(batch_dims).map_err(|_| ShapeInferError::Invalid {
        op: "GatherND".into(),
        detail: format!("batch_dims must be non-negative, found {batch_dims}"),
    })?;
    let index_depth = usize::try_from(index_depth).map_err(|_| ShapeInferError::Invalid {
        op: "GatherND".into(),
        detail: format!("index tuple depth must be non-negative, found {index_depth}"),
    })?;
    if batch_dims > data.len()
        || batch_dims >= indices.len()
        || index_depth > data.len().saturating_sub(batch_dims)
    {
        return Err(ShapeInferError::Invalid {
            op: "GatherND".into(),
            detail: format!(
                "batch_dims {batch_dims} and index depth {index_depth} are incompatible with data rank {} and indices rank {}",
                data.len(),
                indices.len()
            ),
        });
    }

    let capacity = data
        .len()
        .checked_add(indices.len())
        .and_then(|rank| rank.checked_sub(index_depth))
        .and_then(|rank| rank.checked_sub(1))
        .filter(|&rank| rank <= isize::MAX as usize)
        .ok_or_else(|| ShapeInferError::Invalid {
            op: "GatherND".into(),
            detail: "output rank arithmetic overflowed".into(),
        })?;
    let mut out = Vec::with_capacity(capacity);
    out.extend_from_slice(&data[..batch_dims]);
    out.extend(indices[batch_dims..indices.len() - 1].iter().cloned());
    out.extend_from_slice(&data[batch_dims + index_depth..]);
    ctx.set_output(0, dtype, out);
    Ok(())
}

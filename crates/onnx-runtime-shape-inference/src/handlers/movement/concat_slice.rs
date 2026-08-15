use onnx_runtime_ir::{Attribute, DataType};

use crate::context::InferenceContext;
use crate::dim_expr::DimExpr;
use crate::error::ShapeInferError;
use crate::handlers::checked_axis;
use crate::shape_data::ShapeData;

use super::{attr_ints, checked_extent, const_ints, validate_vector_input};

/// `Concat`: sum the concat axis across inputs; other dims from input 0.
pub fn concat(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(axis_attr) = ctx.node.attr("axis").and_then(Attribute::as_int) else {
        return Err(ShapeInferError::MissingAttribute {
            op: "Concat".into(),
            attr: "axis".into(),
        });
    };
    let present: Vec<usize> = (0..ctx.num_inputs())
        .filter(|&i| ctx.has_input(i))
        .collect();
    let Some(&first) = present.first() else {
        return Ok(());
    };
    if present.iter().any(|&i| ctx.input_shape(i).is_none()) {
        return Ok(());
    }
    let Some(base) = ctx.input_shape(first).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    let dtype = ctx.input_dtype(first).unwrap_or(DataType::Float32);
    let rank = base.len();
    let axis = checked_axis(axis_attr, rank).ok_or_else(|| ShapeInferError::Invalid {
        op: "Concat".into(),
        detail: format!("axis {axis_attr} is out of range for rank {rank}"),
    })?;

    let mut out = base.clone();
    let mut sum = 0i128;
    let mut all_known = true;
    for &i in &present {
        match ctx.input_shape(i).map(<[DimExpr]>::to_vec) {
            Some(shape) if shape.len() == rank => {
                if let Some(extent) = shape[axis].as_const() {
                    sum = sum.checked_add(i128::from(extent)).ok_or_else(|| {
                        ShapeInferError::Invalid {
                            op: "Concat".into(),
                            detail: "concat-axis extent sum overflowed".into(),
                        }
                    })?;
                    if sum > isize::MAX as i128 {
                        return Err(ShapeInferError::Invalid {
                            op: "Concat".into(),
                            detail: format!(
                                "known concat-axis extent sum {sum} exceeds isize::MAX"
                            ),
                        });
                    }
                } else {
                    all_known = false;
                }
                for non_concat_axis in 0..rank {
                    if non_concat_axis == axis {
                        continue;
                    }
                    let current = &out[non_concat_axis];
                    let incoming = &shape[non_concat_axis];
                    match (current.as_const(), incoming.as_const()) {
                        (Some(a), Some(b)) if a != b => {
                            return Err(ShapeInferError::Invalid {
                                op: "Concat".into(),
                                detail: format!(
                                    "non-concat dimension {non_concat_axis} differs: {a} != {b}"
                                ),
                            });
                        }
                        (None, Some(_)) => out[non_concat_axis] = incoming.clone(),
                        (None, None) if current != incoming => {
                            out[non_concat_axis] = ctx.broadcast_dim(current, incoming)?;
                        }
                        _ => {}
                    }
                }
            }
            Some(shape) => {
                return Err(ShapeInferError::InvalidRank {
                    op: "Concat".into(),
                    index: i,
                    rank: shape.len(),
                    detail: format!("all inputs must have rank {rank}"),
                });
            }
            None => all_known = false,
        }
    }
    if all_known {
        out[axis] = DimExpr::constant(checked_extent("Concat", sum)?);
    } else {
        out[axis] = ctx.fresh_dim();
    }
    ctx.set_output(0, dtype, out);

    // Shape-data: concatenation of shape vectors / scalars.
    if let Some(sd) = concat_shape_data(ctx, &present) {
        ctx.set_output_shape_data(0, sd);
    }
    Ok(())
}

/// Concatenate shape-data operands (scalars contribute one element each).
fn concat_shape_data(ctx: &InferenceContext, present: &[usize]) -> Option<ShapeData> {
    let mut elems = Vec::new();
    // Carry the operands' actual integer dtype rather than assuming Int64: a
    // shape-computation chain may run on Int32 dims.
    let mut dtype = DataType::Int64;
    for (k, &i) in present.iter().enumerate() {
        let sd = ctx.input_shape_data(i)?;
        if k == 0 {
            dtype = sd.dtype;
        }
        elems.extend(sd.elems.iter().cloned());
    }
    Some(ShapeData::vector(dtype, elems))
}

/// `Slice` (opset ≥ 10 input-driven, with an opset < 10 attribute fallback).
pub fn slice(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(data) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    let dtype = ctx.input_dtype(0).unwrap_or(DataType::Float32);
    let rank = data.len();

    let input_driven = ctx.opset("") >= 10;
    if input_driven {
        for index in 1..ctx.num_inputs().min(5) {
            validate_vector_input(ctx, index, "Slice")?;
        }
    }
    let starts = if input_driven {
        const_ints(ctx, 1)
    } else {
        attr_ints(ctx, "starts")
    };
    let ends = if input_driven {
        const_ints(ctx, 2)
    } else {
        attr_ints(ctx, "ends")
    };
    let axes_present = input_driven && ctx.has_input(3);
    let steps_present = input_driven && ctx.has_input(4);
    let axes = if input_driven {
        const_ints(ctx, 3)
    } else {
        attr_ints(ctx, "axes")
    };
    let steps = if input_driven {
        const_ints(ctx, 4)
    } else {
        attr_ints(ctx, "steps")
    };

    let mut out = data.clone();
    let mut propagate_shape_data = false;
    match (starts.as_ref(), ends.as_ref()) {
        (Some(starts), Some(ends)) => {
            if starts.len() != ends.len() {
                return Err(ShapeInferError::Invalid {
                    op: "Slice".into(),
                    detail: format!(
                        "starts and ends lengths differ: {} != {}",
                        starts.len(),
                        ends.len()
                    ),
                });
            }
            if axes_present && axes.is_none() {
                for dim in &mut out {
                    *dim = ctx.fresh_dim();
                }
                ctx.set_output(0, dtype, out);
                return Ok(());
            }
            if steps_present && steps.is_none() {
                for axis in dynamic_slice_axes(rank, starts.len(), axes.as_deref())? {
                    out[axis] = ctx.fresh_dim();
                }
                ctx.set_output(0, dtype, out);
                return Ok(());
            }
            let axes: Vec<usize> = match axes.as_deref() {
                Some(raw_axes) => {
                    if raw_axes.len() != starts.len() {
                        return Err(ShapeInferError::Invalid {
                            op: "Slice".into(),
                            detail: format!(
                                "axes has {} entries but starts has {}",
                                raw_axes.len(),
                                starts.len()
                            ),
                        });
                    }
                    checked_unique_axes(raw_axes, rank, "Slice")?
                }
                None => checked_default_axes(starts.len(), rank, "Slice")?,
            };
            let steps = match steps.as_deref() {
                Some(steps) if steps.len() != axes.len() => {
                    return Err(ShapeInferError::Invalid {
                        op: "Slice".into(),
                        detail: format!(
                            "steps has {} entries but axes has {}",
                            steps.len(),
                            axes.len()
                        ),
                    });
                }
                Some(steps) => steps,
                None => &[],
            };
            for (k, &ax) in axes.iter().enumerate() {
                let step = steps.get(k).copied().unwrap_or(1);
                out[ax] = slice_dim(
                    &data[ax],
                    starts.get(k).copied(),
                    ends.get(k).copied(),
                    step,
                )?
                .unwrap_or_else(|| ctx.fresh_dim());
            }
            propagate_shape_data = true;
        }
        _ => {
            let known_len = starts
                .as_ref()
                .or(ends.as_ref())
                .map(Vec::len)
                .or_else(|| vector_length(ctx, 1))
                .or_else(|| vector_length(ctx, 2));
            let dynamic_axes = if axes_present && axes.is_none() {
                (0..rank).collect()
            } else if let Some(raw_axes) = axes.as_deref() {
                checked_unique_axes(raw_axes, rank, "Slice")?
            } else if let Some(length) = known_len {
                checked_default_axes(length, rank, "Slice")?
            } else {
                (0..rank).collect()
            };
            for axis in dynamic_axes {
                out[axis] = ctx.fresh_dim();
            }
        }
    }
    ctx.set_output(0, dtype, out);

    // Shape-data: slicing a 1-D shape vector on axis 0 with concrete bounds.
    if propagate_shape_data && let Some(sd) = slice_shape_data(ctx, rank) {
        ctx.set_output_shape_data(0, sd);
    }
    Ok(())
}

pub(super) fn vector_length(ctx: &InferenceContext, index: usize) -> Option<usize> {
    let shape = ctx.input_shape(index)?;
    (shape.len() == 1)
        .then(|| shape[0].as_const())
        .flatten()
        .and_then(|length| usize::try_from(length).ok())
}

fn dynamic_slice_axes(
    rank: usize,
    length: usize,
    axes: Option<&[i64]>,
) -> Result<Vec<usize>, ShapeInferError> {
    match axes {
        Some(axes) => checked_unique_axes(axes, rank, "Slice"),
        None => checked_default_axes(length, rank, "Slice"),
    }
}

fn checked_default_axes(
    length: usize,
    rank: usize,
    op: &str,
) -> Result<Vec<usize>, ShapeInferError> {
    if length > rank {
        return Err(ShapeInferError::Invalid {
            op: op.into(),
            detail: format!("{length} implicit axes exceed input rank {rank}"),
        });
    }
    Ok((0..length).collect())
}

fn checked_unique_axes(axes: &[i64], rank: usize, op: &str) -> Result<Vec<usize>, ShapeInferError> {
    let mut normalized = Vec::with_capacity(axes.len());
    for &axis in axes {
        let axis = checked_axis(axis, rank).ok_or_else(|| ShapeInferError::Invalid {
            op: op.into(),
            detail: format!("axis {axis} is out of range for rank {rank}"),
        })?;
        if normalized.contains(&axis) {
            return Err(ShapeInferError::Invalid {
                op: op.into(),
                detail: format!("axis {axis} appears more than once"),
            });
        }
        normalized.push(axis);
    }
    Ok(normalized)
}

/// A concrete sliced extent, or `None` when any of the bounds/dim are symbolic.
fn slice_dim(
    dim: &DimExpr,
    start: Option<i64>,
    end: Option<i64>,
    step: i64,
) -> Result<Option<DimExpr>, ShapeInferError> {
    if step == 0 {
        return Err(ShapeInferError::Invalid {
            op: "Slice".into(),
            detail: "step cannot be 0".into(),
        });
    }
    let Some(d) = dim.as_const() else {
        return Ok(None);
    };
    let (Some(start), Some(end)) = (start, end) else {
        return Ok(None);
    };
    let d = i128::from(d);
    if d == 0 {
        return Ok(Some(DimExpr::constant(0)));
    }
    let step = i128::from(step);
    let norm = |v: i64| -> i128 {
        let v = i128::from(v);
        let v = if v < 0 { v + d } else { v };
        v.clamp(0, d)
    };
    let len = if step > 0 {
        let s = norm(start);
        let e = norm(end);
        ((e - s).max(0) + step - 1) / step
    } else {
        let start = i128::from(start);
        let end = i128::from(end);
        let s = if start < 0 {
            (start + d).clamp(0, d - 1)
        } else {
            start.min(d - 1)
        };
        let e = if end < 0 {
            (end + d).clamp(-1, d - 1)
        } else {
            end.min(d)
        };
        ((s - e).max(0) + (-step) - 1) / (-step)
    };
    Ok(Some(DimExpr::constant(checked_extent("Slice", len)?)))
}

/// Slice a 1-D shape-data vector on axis 0 with concrete bounds.
fn slice_shape_data(ctx: &InferenceContext, _rank: usize) -> Option<ShapeData> {
    let sd = ctx.input_shape_data(0)?;
    if sd.is_scalar() {
        return None;
    }
    let starts = const_ints(ctx, 1).or_else(|| attr_ints(ctx, "starts"))?;
    let ends = const_ints(ctx, 2).or_else(|| attr_ints(ctx, "ends"))?;
    let axes = const_ints(ctx, 3).or_else(|| attr_ints(ctx, "axes"));
    // Only handle a single axis-0 slice of the vector.
    if let Some(ax) = &axes
        && ax.as_slice() != [0]
    {
        return None;
    }
    let steps = const_ints(ctx, 4).or_else(|| attr_ints(ctx, "steps"));
    let step = steps.as_ref().and_then(|s| s.first()).copied().unwrap_or(1);
    if step != 1 {
        return None;
    }
    let n = sd.elems.len() as i64;
    let norm = |v: i64| -> usize {
        let v = if v < 0 { v + n } else { v };
        v.clamp(0, n) as usize
    };
    let s = norm(*starts.first()?);
    let e = norm(*ends.first()?);
    let elems = sd.elems.get(s..e.max(s)).unwrap_or(&[]).to_vec();
    Some(ShapeData::vector(sd.dtype, elems))
}

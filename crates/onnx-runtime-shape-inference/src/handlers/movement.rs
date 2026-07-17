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
use crate::handlers::{checked_axis, norm_axis};
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

/// `Transpose`: permute dimensions by `perm` (default: reverse).
pub fn transpose(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(t) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    let rank = t.rank();
    let perm: Vec<usize> = match ctx.node.attr("perm").and_then(Attribute::as_ints) {
        Some(p) => p.iter().map(|&a| norm_axis(a, rank)).collect(),
        None => (0..rank).rev().collect(),
    };
    if perm.len() != rank {
        return Err(ShapeInferError::Invalid {
            op: "Transpose".into(),
            detail: format!("perm length {} != rank {rank}", perm.len()),
        });
    }
    let out: Vec<DimExpr> = perm.iter().map(|&p| t.shape[p].clone()).collect();
    ctx.set_output(0, t.dtype, out);
    Ok(())
}

/// `Reshape`: output shape from the (shape-data) target vector, resolving `0`
/// (copy) and `-1` (infer) dims symbolically.
pub fn reshape(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(input) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    let dtype = ctx.input_dtype(0).unwrap_or(DataType::Float32);
    let allowzero = ctx
        .node
        .attr("allowzero")
        .and_then(Attribute::as_int)
        .unwrap_or(0)
        != 0;

    let Some(target) = ctx.input_shape_data(1).map(ShapeData::as_shape) else {
        // No resolved target: produce a fresh-symbol shape of the known rank.
        if let Some(rank) = target_rank(ctx) {
            let out = (0..rank).map(|_| ctx.fresh_dim()).collect();
            ctx.set_output(0, dtype, out);
        }
        return Ok(());
    };

    let total = DimExpr::product(&input);
    let mut out: Vec<DimExpr> = Vec::with_capacity(target.len());
    let mut product = DimExpr::constant(1);
    let mut neg1: Option<usize> = None;
    for (i, t) in target.iter().enumerate() {
        match t.as_const() {
            Some(-1) => {
                neg1 = Some(i);
                out.push(DimExpr::constant(1)); // placeholder, fixed below
            }
            Some(0) if !allowzero => {
                let d = input
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| DimExpr::constant(1));
                product = product.mul(&d);
                out.push(d);
            }
            _ => {
                product = product.mul(t);
                out.push(t.clone());
            }
        }
    }
    if let Some(idx) = neg1 {
        out[idx] = total
            .checked_div(&product)
            .unwrap_or_else(|| ctx.fresh_dim());
    }
    ctx.set_output(0, dtype, out);
    Ok(())
}

/// The rank of a `Reshape`/`Expand` target when its values are unknown but its
/// length is a concrete 1-D shape.
fn target_rank(ctx: &InferenceContext) -> Option<usize> {
    let s = ctx.input_shape(1)?;
    if s.len() == 1 {
        s[0].as_const().map(|n| n.max(0) as usize)
    } else {
        None
    }
}

/// `Flatten`: collapse to `[prod(dims[..axis]), prod(dims[axis..])]`.
pub fn flatten(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(t) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    let rank = t.rank();
    let axis = ctx
        .node
        .attr("axis")
        .and_then(Attribute::as_int)
        .unwrap_or(1);
    let axis = if axis < 0 {
        (axis + rank as i64).max(0) as usize
    } else {
        (axis as usize).min(rank)
    };
    let outer = DimExpr::product(&t.shape[..axis]);
    let inner = DimExpr::product(&t.shape[axis..]);
    ctx.set_output(0, t.dtype, vec![outer, inner]);
    Ok(())
}

/// `Squeeze` with axes taken from an attribute (opset < 13).
pub fn squeeze_v1(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let axes = ctx
        .node
        .attr("axes")
        .and_then(Attribute::as_ints)
        .map(<[i64]>::to_vec);
    squeeze_common(ctx, axes)
}

/// `Squeeze` with axes taken from input 1 (opset ≥ 13).
pub fn squeeze_v13(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let axes = const_ints(ctx, 1);
    squeeze_common(ctx, axes)
}

fn squeeze_common(
    ctx: &mut InferenceContext,
    axes: Option<Vec<i64>>,
) -> Result<(), ShapeInferError> {
    let Some(t) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    let rank = t.rank();
    let out: Vec<DimExpr> = match axes {
        Some(axes) => {
            let norm: Vec<usize> = axes.iter().map(|&a| norm_axis(a, rank)).collect();
            t.shape
                .iter()
                .enumerate()
                .filter(|(i, _)| !norm.contains(i))
                .map(|(_, d)| d.clone())
                .collect()
        }
        // No axes: drop every statically-size-1 dim.
        None => t
            .shape
            .iter()
            .filter(|d| d.as_const() != Some(1))
            .cloned()
            .collect(),
    };
    // Squeeze on a shape-data vector (drops nothing structurally for a 1-D
    // shape vector, but keep the data flowing for downstream ops).
    if let Some(sd) = ctx.input_shape_data(0).cloned() {
        ctx.set_output_shape_data(0, sd);
    }
    ctx.set_output(0, t.dtype, out);
    Ok(())
}

/// `Unsqueeze` with axes from an attribute (opset < 13).
pub fn unsqueeze_v1(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let axes = ctx
        .node
        .attr("axes")
        .and_then(Attribute::as_ints)
        .map(<[i64]>::to_vec);
    unsqueeze_common(ctx, axes)
}

/// `Unsqueeze` with axes from input 1 (opset ≥ 13).
pub fn unsqueeze_v13(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let axes = const_ints(ctx, 1);
    unsqueeze_common(ctx, axes)
}

fn unsqueeze_common(
    ctx: &mut InferenceContext,
    axes: Option<Vec<i64>>,
) -> Result<(), ShapeInferError> {
    let Some(t) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    let Some(axes) = axes else {
        return Ok(());
    };
    let out_rank = t.rank() + axes.len();
    // ONNX Unsqueeze axes index positions in the *output* tensor (accepted
    // range `[-output_rank, output_rank-1]`), so normalize against `out_rank`,
    // not the input rank — otherwise a high axis (e.g. 3 into a rank-2 input)
    // is wrongly clamped and the size-1 dims land in the wrong slots.
    let norm: Vec<usize> = axes
        .iter()
        .map(|&a| {
            let a = if a < 0 { a + out_rank as i64 } else { a };
            a.clamp(0, out_rank as i64 - 1) as usize
        })
        .collect();
    let mut out = Vec::with_capacity(out_rank);
    let mut src = t.shape.iter();
    for i in 0..out_rank {
        if norm.contains(&i) {
            out.push(DimExpr::constant(1));
        } else if let Some(d) = src.next() {
            out.push(d.clone());
        }
    }
    // Shape-data: a scalar unsqueezed to a 1-vector keeps its value (common in
    // shape-computation chains that build a dim list).
    if let Some(sd) = ctx.input_shape_data(0).cloned()
        && sd.is_scalar()
        && norm == [0]
    {
        ctx.set_output_shape_data(0, ShapeData::vector(sd.dtype, sd.elems));
    }
    ctx.set_output(0, t.dtype, out);
    Ok(())
}

/// `Expand` (opset 8+): bidirectionally broadcast the input shape against the
/// values of the shape-tensor input.
pub fn expand(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(input) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    validate_vector_input(ctx, 1, "Expand")?;
    if let Some(target) = ctx.input_shape_data(1).map(ShapeData::as_shape) {
        for dim in &target {
            if let Some(value) = dim.as_const()
                && !(0..=isize::MAX as i64).contains(&value)
            {
                return Err(ShapeInferError::Invalid {
                    op: "Expand".into(),
                    detail: format!("target extent {value} is outside 0..=isize::MAX"),
                });
            }
        }
        let shape = bidirectional_broadcast(ctx, &input.shape, &target)?;
        ctx.set_output(0, input.dtype, shape);
    }
    Ok(())
}

fn bidirectional_broadcast(
    ctx: &mut InferenceContext,
    input: &[DimExpr],
    target: &[DimExpr],
) -> Result<Vec<DimExpr>, ShapeInferError> {
    let rank = input.len().max(target.len());
    let mut output = Vec::with_capacity(rank);
    for axis in 0..rank {
        let input_offset = rank - input.len();
        let target_offset = rank - target.len();
        let a = if axis < input_offset {
            DimExpr::constant(1)
        } else {
            input[axis - input_offset].clone()
        };
        let b = if axis < target_offset {
            DimExpr::constant(1)
        } else {
            target[axis - target_offset].clone()
        };
        if let (Some(a), Some(b)) = (a.as_const(), b.as_const())
            && a != 1
            && b != 1
            && a != b
        {
            return Err(ShapeInferError::Invalid {
                op: "Expand".into(),
                detail: format!("incompatible broadcast dims {a} and {b} at axis {axis}"),
            });
        }
        output.push(ctx.broadcast_dim(&a, &b)?);
    }
    Ok(output)
}

fn resize_axes(ctx: &InferenceContext, rank: usize) -> Result<Vec<usize>, ShapeInferError> {
    let Some(raw_axes) = ctx.node.attr("axes").and_then(Attribute::as_ints) else {
        return Ok((0..rank).collect());
    };
    if raw_axes.is_empty() {
        return Ok((0..rank).collect());
    }
    let mut axes = Vec::with_capacity(raw_axes.len());
    for &axis in raw_axes {
        let axis = checked_axis(axis, rank).ok_or_else(|| ShapeInferError::Invalid {
            op: "Resize".into(),
            detail: format!("axis {axis} is out of range for rank {rank}"),
        })?;
        if axes.contains(&axis) {
            return Err(ShapeInferError::Invalid {
                op: "Resize".into(),
                detail: format!("axis {axis} appears more than once"),
            });
        }
        axes.push(axis);
    }
    Ok(axes)
}

fn known_empty_vector(ctx: &InferenceContext, input: usize) -> bool {
    ctx.input_shape(input)
        .is_some_and(|shape| shape.len() == 1 && shape[0].as_const() == Some(0))
}

fn resize_extent_from_scale(input: i64, scale: f64) -> Result<i64, ShapeInferError> {
    if !scale.is_finite() || scale <= 0.0 || input < 0 {
        return Err(ShapeInferError::Invalid {
            op: "Resize".into(),
            detail: format!("invalid scale {scale}"),
        });
    }
    if input == 0 {
        return Ok(0);
    }

    // Apply the exact binary value of the scale in integer space. Converting
    // isize::MAX to either f32 or f64 rounds it up to 2^63 on 64-bit targets.
    let bits = scale.to_bits();
    let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    let (significand, exponent) = if exponent_bits == 0 {
        (fraction, -1074)
    } else {
        ((1_u64 << 52) | fraction, exponent_bits - 1023 - 52)
    };
    let product = (input as u128) * u128::from(significand);
    let maximum = isize::MAX as u128;
    let output = if exponent >= 0 {
        let shift = exponent as u32;
        if shift >= 128 || product > (maximum >> shift) {
            return Err(ShapeInferError::Invalid {
                op: "Resize".into(),
                detail: "inferred extent exceeds isize::MAX".into(),
            });
        }
        product << shift
    } else {
        let shift = exponent.unsigned_abs();
        if shift >= 128 { 0 } else { product >> shift }
    };
    if output > maximum {
        return Err(ShapeInferError::Invalid {
            op: "Resize".into(),
            detail: format!("inferred extent {output} exceeds isize::MAX"),
        });
    }
    Ok(output as i64)
}

fn resize_extent_from_ratio(
    input: i64,
    numerator: i64,
    denominator: i64,
) -> Result<i64, ShapeInferError> {
    let product = i128::from(input) * i128::from(numerator);
    let denominator = i128::from(denominator);
    let quotient = product / denominator;
    let remainder = product % denominator;
    let rounded = quotient + i128::from(remainder * 2 >= denominator);
    if rounded > isize::MAX as i128 {
        return Err(ShapeInferError::Invalid {
            op: "Resize".into(),
            detail: format!("inferred extent {rounded} exceeds isize::MAX"),
        });
    }
    Ok(rounded as i64)
}

/// `Resize` (opset 13/18/19): infer from a constant `sizes` or `scales`
/// vector. Runtime-computed vectors preserve only the output rank.
pub fn resize(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(input) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    let coordinate_mode = ctx
        .node
        .attr("coordinate_transformation_mode")
        .and_then(Attribute::as_str)
        .unwrap_or("half_pixel");
    if coordinate_mode == "tf_crop_and_resize" {
        validate_vector_input(ctx, 1, "Resize")?;
    }
    validate_vector_input(ctx, 2, "Resize")?;
    validate_vector_input(ctx, 3, "Resize")?;

    let scales = ctx
        .input_shape_data(2)
        .and_then(ShapeData::as_float_vector)
        .map(<[f64]>::to_vec);
    let sizes = const_ints(ctx, 3);
    let has_scales = ctx.has_input(2)
        && !known_empty_vector(ctx, 2)
        && scales.as_ref().is_none_or(|values| !values.is_empty());
    let has_sizes = ctx.has_input(3)
        && !known_empty_vector(ctx, 3)
        && sizes.as_ref().is_none_or(|values| !values.is_empty());
    if has_scales && has_sizes {
        return Err(ShapeInferError::Invalid {
            op: "Resize".into(),
            detail: "scales and sizes cannot both be provided".into(),
        });
    }

    let axes = resize_axes(ctx, input.rank())?;
    if !has_scales && !has_sizes {
        let output = (0..input.rank()).map(|_| ctx.fresh_dim()).collect();
        ctx.set_output(0, input.dtype, output);
        return Ok(());
    }
    if has_sizes {
        let Some(mut sizes) = sizes else {
            let output = (0..input.rank()).map(|_| ctx.fresh_dim()).collect();
            ctx.set_output(0, input.dtype, output);
            return Ok(());
        };
        if sizes.len() != axes.len() {
            return Err(ShapeInferError::Invalid {
                op: "Resize".into(),
                detail: format!(
                    "sizes has {} values but {} resize axes were selected",
                    sizes.len(),
                    axes.len()
                ),
            });
        }

        let policy = ctx
            .node
            .attr("keep_aspect_ratio_policy")
            .and_then(Attribute::as_str)
            .unwrap_or("stretch");
        if policy != "stretch" {
            if !matches!(policy, "not_larger" | "not_smaller") {
                return Err(ShapeInferError::Invalid {
                    op: "Resize".into(),
                    detail: format!("unknown keep_aspect_ratio_policy {policy}"),
                });
            }
            let input_extents = axes
                .iter()
                .map(|&axis| input.shape[axis].as_const())
                .collect::<Option<Vec<_>>>();
            let Some(input_extents) = input_extents else {
                let output = (0..input.rank()).map(|_| ctx.fresh_dim()).collect();
                ctx.set_output(0, input.dtype, output);
                return Ok(());
            };
            if sizes
                .iter()
                .zip(&input_extents)
                .any(|(&size, &extent)| size <= 0 || extent <= 0)
            {
                let output = (0..input.rank()).map(|_| ctx.fresh_dim()).collect();
                ctx.set_output(0, input.dtype, output);
                return Ok(());
            }
            let (scale_numerator, scale_denominator) = sizes
                .iter()
                .copied()
                .zip(input_extents.iter().copied())
                .reduce(|left, right| {
                    let ordering = (i128::from(left.0) * i128::from(right.1))
                        .cmp(&(i128::from(right.0) * i128::from(left.1)));
                    if (policy == "not_larger" && ordering.is_le())
                        || (policy == "not_smaller" && ordering.is_ge())
                    {
                        left
                    } else {
                        right
                    }
                })
                .unwrap_or((1, 1));
            for (size, extent) in sizes.iter_mut().zip(input_extents) {
                *size = resize_extent_from_ratio(extent, scale_numerator, scale_denominator)?;
            }
        }

        let mut output = input.shape;
        for (&axis, size) in axes.iter().zip(sizes) {
            output[axis] = if size > 0 {
                if i128::from(size) > isize::MAX as i128 {
                    return Err(ShapeInferError::Invalid {
                        op: "Resize".into(),
                        detail: format!("inferred extent {size} exceeds isize::MAX"),
                    });
                }
                DimExpr::constant(size)
            } else {
                ctx.fresh_dim()
            };
        }
        ctx.set_output(0, input.dtype, output);
        return Ok(());
    }

    let policy = ctx
        .node
        .attr("keep_aspect_ratio_policy")
        .and_then(Attribute::as_str)
        .unwrap_or("stretch");
    if policy != "stretch" {
        return Err(ShapeInferError::Invalid {
            op: "Resize".into(),
            detail: "scales requires keep_aspect_ratio_policy=stretch".into(),
        });
    }
    let Some(scales) = scales else {
        let output = (0..input.rank()).map(|_| ctx.fresh_dim()).collect();
        ctx.set_output(0, input.dtype, output);
        return Ok(());
    };
    if scales.len() != axes.len() {
        return Err(ShapeInferError::Invalid {
            op: "Resize".into(),
            detail: format!(
                "scales has {} values but {} resize axes were selected",
                scales.len(),
                axes.len()
            ),
        });
    }
    let mut output = input.shape;
    for (&axis, scale) in axes.iter().zip(scales) {
        output[axis] = match output[axis].as_const() {
            Some(extent) => DimExpr::constant(resize_extent_from_scale(extent, scale)?),
            None => ctx.fresh_dim(),
        };
    }
    ctx.set_output(0, input.dtype, output);
    Ok(())
}

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
                for axis in 0..rank {
                    out[axis] = ctx.fresh_dim();
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

fn vector_length(ctx: &InferenceContext, index: usize) -> Option<usize> {
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
    let num_outputs = ctx
        .node
        .attr("num_outputs")
        .and_then(Attribute::as_int)
        .and_then(|n| usize::try_from(n).ok())
        .filter(|&n| n > 0);

    let sizes: Option<Vec<i64>> = ctx
        .node
        .attr("split")
        .and_then(Attribute::as_ints)
        .map(<[i64]>::to_vec)
        .or_else(|| const_ints(ctx, 1));
    let has_dynamic_split = sizes.is_none() && ctx.has_input(1);

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
    let axis = norm_axis(axis, rank);
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
    reg.register("", "Resize", 13, resize);
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

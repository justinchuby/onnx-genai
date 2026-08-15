use onnx_runtime_ir::Attribute;

use crate::context::InferenceContext;
use crate::dim_expr::DimExpr;
use crate::error::ShapeInferError;
use crate::handlers::checked_axis;
use crate::shape_data::ShapeData;

use super::{const_ints, validate_vector_input};

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

/// Legacy `Resize` opset 10: infer from the required `scales` input.
pub fn resize_v10(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(input) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    validate_vector_input(ctx, 1, "Resize")?;
    let Some(scales) = ctx
        .input_shape_data(1)
        .and_then(ShapeData::as_float_vector)
        .map(<[f64]>::to_vec)
    else {
        let output = (0..input.rank()).map(|_| ctx.fresh_dim()).collect();
        ctx.set_output(0, input.dtype, output);
        return Ok(());
    };
    if scales.len() != input.rank() {
        return Err(ShapeInferError::Invalid {
            op: "Resize".into(),
            detail: format!(
                "scales has {} values but input rank is {}",
                scales.len(),
                input.rank()
            ),
        });
    }
    let mut output = input.shape;
    for (extent, scale) in output.iter_mut().zip(scales) {
        *extent = match extent.as_const() {
            Some(extent) => DimExpr::constant(resize_extent_from_scale(extent, scale)?),
            None => ctx.fresh_dim(),
        };
    }
    ctx.set_output(0, input.dtype, output);
    Ok(())
}

/// `Resize` (opset 11+): infer from a constant `sizes` or `scales`
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

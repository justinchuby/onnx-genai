//! Spatial rules: `Conv`, pooling operators, and `Pad`.
//!
//! These use the standard spatial output formula
//! `floor((D + pad_begin + pad_end - dilation*(kernel-1) - 1) / stride) + 1`
//! (ceil when `ceil_mode` is set). A concrete spatial dim is computed exactly;
//! a symbolic one degrades to a fresh symbol so the output keeps a known rank.

use onnx_runtime_ir::{Attribute, DataType};

use crate::context::InferenceContext;
use crate::dim_expr::DimExpr;
use crate::error::ShapeInferError;
use crate::handlers::checked_axis;
use crate::registry::InferenceRegistry;

/// Auto-pad handling per the ONNX spec.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AutoPad {
    NotSet,
    SameUpper,
    SameLower,
    Valid,
}

fn auto_pad(ctx: &InferenceContext) -> AutoPad {
    match ctx.node.attr("auto_pad").and_then(Attribute::as_str) {
        Some("SAME_UPPER") => AutoPad::SameUpper,
        Some("SAME_LOWER") => AutoPad::SameLower,
        Some("VALID") => AutoPad::Valid,
        _ => AutoPad::NotSet,
    }
}

/// Per-axis parameters for the spatial output formula.
struct SpatialParams {
    kernel: i64,
    stride: i64,
    dilation: i64,
    pad_begin: i64,
    pad_end: i64,
    auto: AutoPad,
    ceil_mode: bool,
}

/// Compute one spatial output extent.
fn spatial_out(ctx: &mut InferenceContext, dim: &DimExpr, p: &SpatialParams) -> DimExpr {
    let Some(d) = dim.as_const() else {
        return ctx.fresh_dim();
    };
    if p.stride <= 0 {
        return ctx.fresh_dim();
    }
    let out = match p.auto {
        AutoPad::SameUpper | AutoPad::SameLower => (d + p.stride - 1) / p.stride,
        AutoPad::Valid => {
            let eff = p.dilation * (p.kernel - 1) + 1;
            (d - eff) / p.stride + 1
        }
        AutoPad::NotSet => {
            let eff = p.dilation * (p.kernel - 1) + 1;
            let numer = d + p.pad_begin + p.pad_end - eff;
            if p.ceil_mode {
                // ceil division for a non-negative denominator.
                (numer + p.stride - 1) / p.stride + 1
            } else {
                numer / p.stride + 1
            }
        }
    };
    DimExpr::constant(out.max(0))
}

/// Shared Conv/Pool spatial-shape computation.
fn conv_pool(
    ctx: &mut InferenceContext,
    channels: DimExpr,
    is_conv: bool,
) -> Result<(), ShapeInferError> {
    let Some(x) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    let dtype = ctx.input_dtype(0).unwrap_or(DataType::Float32);
    if x.len() < 3 {
        return Err(ShapeInferError::InvalidRank {
            op: ctx.op().to_string(),
            index: 0,
            rank: x.len(),
            detail: "expected [N, C, D1, …]".into(),
        });
    }
    let n_spatial = x.len() - 2;
    let batch = x[0].clone();

    // kernel_shape: attribute, or (for Conv) the trailing dims of W.
    let kernel: Vec<i64> = match ctx.node.attr("kernel_shape").and_then(Attribute::as_ints) {
        Some(k) => k.to_vec(),
        None if is_conv => match ctx.input_shape(1) {
            Some(w) if w.len() == n_spatial + 2 => {
                w[2..].iter().map(|d| d.as_const().unwrap_or(1)).collect()
            }
            _ => vec![1; n_spatial],
        },
        None => vec![1; n_spatial],
    };
    let strides = int_list(ctx, "strides", n_spatial, 1);
    let dilations = int_list(ctx, "dilations", n_spatial, 1);
    let pads = int_list(ctx, "pads", n_spatial * 2, 0);
    let auto = auto_pad(ctx);
    let ceil_mode = ctx
        .node
        .attr("ceil_mode")
        .and_then(Attribute::as_int)
        .unwrap_or(0)
        != 0;

    let mut out = Vec::with_capacity(x.len());
    out.push(batch);
    out.push(channels);
    for d in 0..n_spatial {
        let params = SpatialParams {
            kernel: *kernel.get(d).unwrap_or(&1),
            stride: *strides.get(d).unwrap_or(&1),
            dilation: *dilations.get(d).unwrap_or(&1),
            pad_begin: *pads.get(d).unwrap_or(&0),
            pad_end: *pads.get(d + n_spatial).unwrap_or(&0),
            auto,
            ceil_mode,
        };
        let dim = spatial_out(ctx, &x[d + 2], &params);
        out.push(dim);
    }
    ctx.set_output(0, dtype, out);
    Ok(())
}

/// `Conv`: output channels come from `W`'s first dim.
pub fn conv(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let channels = ctx
        .input_shape(1)
        .and_then(|w| w.first().cloned())
        .unwrap_or_else(|| ctx.fresh_dim());
    conv_pool(ctx, channels, true)
}

fn pool_list(
    ctx: &InferenceContext,
    name: &str,
    len: usize,
    default: i64,
    required: bool,
) -> Result<Vec<i64>, ShapeInferError> {
    match ctx.node.attr(name).and_then(Attribute::as_ints) {
        Some(values) if values.len() == len => Ok(values.to_vec()),
        Some(values) => Err(ShapeInferError::Invalid {
            op: ctx.op().into(),
            detail: format!(
                "attribute {name} has {} values but spatial rank is {len}",
                values.len()
            ),
        }),
        None if required => Err(ShapeInferError::MissingAttribute {
            op: ctx.op().into(),
            attr: name.into(),
        }),
        None => Ok(vec![default; len]),
    }
}

fn checked_pool_extent(
    op: &str,
    input: i128,
    kernel: i64,
    stride: i64,
    dilation: i64,
    pad_begin: i64,
    pad_end: i64,
    ceil_mode: bool,
) -> Result<i128, ShapeInferError> {
    let effective_kernel = i128::from(dilation)
        .checked_mul(i128::from(kernel) - 1)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| ShapeInferError::Invalid {
            op: op.into(),
            detail: "effective kernel arithmetic overflowed".into(),
        })?;
    let numerator = input
        .checked_add(i128::from(pad_begin))
        .and_then(|value| value.checked_add(i128::from(pad_end)))
        .and_then(|value| value.checked_sub(effective_kernel))
        .ok_or_else(|| ShapeInferError::Invalid {
            op: op.into(),
            detail: "pooling extent arithmetic overflowed".into(),
        })?;
    let stride = i128::from(stride);
    let quotient = if ceil_mode {
        numerator.div_euclid(stride) + i128::from(numerator.rem_euclid(stride) != 0)
    } else {
        numerator.div_euclid(stride)
    };
    let mut output = quotient
        .checked_add(1)
        .ok_or_else(|| ShapeInferError::Invalid {
            op: op.into(),
            detail: "pooling output arithmetic overflowed".into(),
        })?
        .max(0);
    if ceil_mode && output > 0 {
        let last_start =
            (output - 1)
                .checked_mul(stride)
                .ok_or_else(|| ShapeInferError::Invalid {
                    op: op.into(),
                    detail: "pooling window arithmetic overflowed".into(),
                })?;
        let right_padding_start =
            input
                .checked_add(i128::from(pad_begin))
                .ok_or_else(|| ShapeInferError::Invalid {
                    op: op.into(),
                    detail: "pooling padding arithmetic overflowed".into(),
                })?;
        if last_start >= right_padding_start {
            output -= 1;
        }
    }
    Ok(output)
}

fn validate_pool_partial_extents(
    op: &str,
    input: Option<i64>,
    kernel: i64,
    dilation: i64,
    pad_begin: i64,
    pad_end: i64,
) -> Result<(), ShapeInferError> {
    let maximum = isize::MAX as i128;
    let effective_kernel = i128::from(dilation) * (i128::from(kernel) - 1) + 1;
    if effective_kernel > maximum {
        return Err(ShapeInferError::Invalid {
            op: op.into(),
            detail: format!("effective kernel {effective_kernel} exceeds isize::MAX"),
        });
    }
    let pad_sum = i128::from(pad_begin) + i128::from(pad_end);
    if pad_sum > maximum {
        return Err(ShapeInferError::Invalid {
            op: op.into(),
            detail: format!("padding extent {pad_sum} exceeds isize::MAX"),
        });
    }
    if let Some(input) = input {
        let padded_input = i128::from(input) + pad_sum;
        if padded_input > maximum {
            return Err(ShapeInferError::Invalid {
                op: op.into(),
                detail: format!("padded input extent {padded_input} exceeds isize::MAX"),
            });
        }
    }
    Ok(())
}

fn pool_spatial_dim(
    ctx: &mut InferenceContext,
    input: &DimExpr,
    kernel: i64,
    stride: i64,
    dilation: i64,
    pad_begin: i64,
    pad_end: i64,
    auto_pad: AutoPad,
    ceil_mode: bool,
) -> Result<DimExpr, ShapeInferError> {
    if kernel <= 0 || stride <= 0 || dilation <= 0 {
        return Err(ShapeInferError::Invalid {
            op: ctx.op().into(),
            detail: "kernel_shape, strides, and dilations must be positive".into(),
        });
    }

    let known = input.as_const();
    validate_pool_partial_extents(ctx.op(), known, kernel, dilation, pad_begin, pad_end)?;
    let output = match auto_pad {
        AutoPad::SameUpper | AutoPad::SameLower => known.map(|extent| {
            let extent = i128::from(extent);
            extent.div_euclid(i128::from(stride))
                + i128::from(extent.rem_euclid(i128::from(stride)) != 0)
        }),
        AutoPad::Valid => known
            .map(i128::from)
            .map(|extent| {
                checked_pool_extent(ctx.op(), extent, kernel, stride, dilation, 0, 0, ceil_mode)
            })
            .transpose()?,
        AutoPad::NotSet => known
            .map(i128::from)
            .map(|extent| {
                checked_pool_extent(
                    ctx.op(),
                    extent,
                    kernel,
                    stride,
                    dilation,
                    pad_begin,
                    pad_end,
                    ceil_mode,
                )
            })
            .transpose()?,
    };

    if let Some(output) = output {
        if output > isize::MAX as i128 {
            return Err(ShapeInferError::Invalid {
                op: ctx.op().into(),
                detail: format!("inferred extent {output} exceeds isize::MAX"),
            });
        }
        return Ok(DimExpr::constant(output as i64));
    }

    if auto_pad == AutoPad::NotSet {
        let lower_bound = checked_pool_extent(
            ctx.op(),
            0,
            kernel,
            stride,
            dilation,
            pad_begin,
            pad_end,
            ceil_mode,
        )?;
        if lower_bound > isize::MAX as i128 {
            return Err(ShapeInferError::Invalid {
                op: ctx.op().into(),
                detail: format!(
                    "guaranteed pooling extent lower bound {lower_bound} exceeds isize::MAX"
                ),
            });
        }
    }
    Ok(ctx.fresh_dim())
}

/// `MaxPool`/`AveragePool`: preserve N/C and infer each spatial extent.
pub fn pool(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(input) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    if input.rank() < 2 {
        return Err(ShapeInferError::InvalidRank {
            op: ctx.op().into(),
            index: 0,
            rank: input.rank(),
            detail: "expected [N, C, D1, …]".into(),
        });
    }
    let spatial_rank = input.rank() - 2;
    let kernels = pool_list(ctx, "kernel_shape", spatial_rank, 0, true)?;
    let strides = pool_list(ctx, "strides", spatial_rank, 1, false)?;
    let dilations = pool_list(ctx, "dilations", spatial_rank, 1, false)?;
    let explicit_pads = ctx.node.attr("pads").and_then(Attribute::as_ints).is_some();
    let pads = pool_list(ctx, "pads", spatial_rank * 2, 0, false)?;
    let auto_pad = if explicit_pads {
        AutoPad::NotSet
    } else {
        auto_pad(ctx)
    };
    let ceil_mode = ctx
        .node
        .attr("ceil_mode")
        .and_then(Attribute::as_int)
        .unwrap_or(0)
        != 0;

    let mut output = Vec::with_capacity(input.rank());
    output.extend_from_slice(&input.shape[..2]);
    for axis in 0..spatial_rank {
        output.push(pool_spatial_dim(
            ctx,
            &input.shape[axis + 2],
            kernels[axis],
            strides[axis],
            dilations[axis],
            pads[axis],
            pads[axis + spatial_rank],
            auto_pad,
            ceil_mode,
        )?);
    }
    ctx.set_output(0, input.dtype, output.clone());
    if ctx.num_outputs() > 1 {
        ctx.set_output(1, DataType::Int64, output);
    }
    Ok(())
}

/// `Pad`: each selected dim grows by its begin+end pad.
pub fn pad(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(x) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    let dtype = ctx.input_dtype(0).unwrap_or(DataType::Float32);
    let rank = x.len();

    if ctx.has_input(1)
        && let Some(pads_rank) = ctx.input_rank(1)
        && pads_rank != 1
    {
        return Err(ShapeInferError::InvalidRank {
            op: "Pad".into(),
            index: 1,
            rank: pads_rank,
            detail: "pads must be a 1-D tensor".into(),
        });
    }

    // pads: attribute (opset < 11) or input 1 shape-data (opset ≥ 11).
    let pads: Option<Vec<i64>> = ctx
        .node
        .attr("pads")
        .and_then(Attribute::as_ints)
        .map(<[i64]>::to_vec)
        .or_else(|| {
            ctx.input_shape_data(1)
                .and_then(|sd| sd.elems.iter().map(|e| e.as_const()).collect())
        });

    // Opset 18 added the optional `axes` input. When present, the pads are
    // indexed by that subset rather than by every input axis.
    let has_axes = ctx.has_input(3);
    let axes: Vec<usize> = if has_axes {
        if let Some(axes_rank) = ctx.input_rank(3)
            && axes_rank != 1
        {
            return Err(ShapeInferError::InvalidRank {
                op: "Pad".into(),
                index: 3,
                rank: axes_rank,
                detail: "axes must be a 1-D tensor".into(),
            });
        }
        let Some(raw_axes) = ctx.input_shape_data(3).and_then(|sd| {
            sd.elems
                .iter()
                .map(|axis| axis.as_const())
                .collect::<Option<Vec<_>>>()
        }) else {
            let out = (0..rank).map(|_| ctx.fresh_dim()).collect();
            ctx.set_output(0, dtype, out);
            return Ok(());
        };
        let mut normalized = Vec::with_capacity(raw_axes.len());
        for axis in raw_axes {
            let axis = checked_axis(axis, rank).ok_or_else(|| ShapeInferError::Invalid {
                op: "Pad".into(),
                detail: format!("axis {axis} is out of range for rank {rank}"),
            })?;
            if normalized.contains(&axis) {
                return Err(ShapeInferError::Invalid {
                    op: "Pad".into(),
                    detail: format!("axis {axis} appears more than once"),
                });
            }
            normalized.push(axis);
        }
        normalized
    } else {
        (0..rank).collect()
    };

    let Some(pads) = pads else {
        let mut out = x;
        for axis in axes {
            out[axis] = ctx.fresh_dim();
        }
        ctx.set_output(0, dtype, out);
        return Ok(());
    };

    let expected_pads = axes
        .len()
        .checked_mul(2)
        .ok_or_else(|| ShapeInferError::Invalid {
            op: "Pad".into(),
            detail: "pads length arithmetic overflowed".into(),
        })?;
    if pads.len() != expected_pads {
        return Err(ShapeInferError::Invalid {
            op: "Pad".into(),
            detail: format!(
                "pads has {} entries but {} selected axes require {}",
                pads.len(),
                axes.len(),
                expected_pads
            ),
        });
    }

    let mut out = x;
    for (i, axis) in axes.into_iter().enumerate() {
        let total_pad = i128::from(pads[i])
            .checked_add(i128::from(pads[pads.len() / 2 + i]))
            .ok_or_else(|| ShapeInferError::Invalid {
                op: "Pad".into(),
                detail: "total padding arithmetic overflowed".into(),
            })?;
        if total_pad > isize::MAX as i128 {
            return Err(ShapeInferError::Invalid {
                op: "Pad".into(),
                detail: format!("total padding {total_pad} exceeds isize::MAX"),
            });
        }
        out[axis] = match out[axis].as_const() {
            Some(extent) => {
                let output_extent = i128::from(extent).checked_add(total_pad).ok_or_else(|| {
                    ShapeInferError::Invalid {
                        op: "Pad".into(),
                        detail: "output extent arithmetic overflowed".into(),
                    }
                })?;
                if !(0..=isize::MAX as i128).contains(&output_extent) {
                    return Err(ShapeInferError::Invalid {
                        op: "Pad".into(),
                        detail: format!(
                            "inferred extent {output_extent} is outside 0..=isize::MAX"
                        ),
                    });
                }
                DimExpr::constant(output_extent as i64)
            }
            None if total_pad == 0 => out[axis].clone(),
            None => ctx.fresh_dim(),
        };
    }
    ctx.set_output(0, dtype, out);
    Ok(())
}

/// Read an integer-list attribute of a given length, defaulting missing entries.
fn int_list(ctx: &InferenceContext, name: &str, len: usize, default: i64) -> Vec<i64> {
    let mut v = ctx
        .node
        .attr(name)
        .and_then(Attribute::as_ints)
        .map(<[i64]>::to_vec)
        .unwrap_or_default();
    v.resize(len, default);
    v
}

/// Register the spatial family.
pub fn register(reg: &mut InferenceRegistry) {
    reg.register("", "Conv", 1, conv);
    reg.register("", "MaxPool", 1, pool);
    reg.register("", "AveragePool", 1, pool);
    reg.register("", "GlobalAveragePool", 1, global_pool);
    reg.register("", "GlobalMaxPool", 1, global_pool);
    reg.register("", "Pad", 1, pad);
}

/// `GlobalAveragePool`/`GlobalMaxPool`: spatial dims collapse to 1.
fn global_pool(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(x) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    if x.len() < 2 {
        return Ok(());
    }
    let dtype = ctx.input_dtype(0).unwrap_or(DataType::Float32);
    let mut out = x.clone();
    for d in out.iter_mut().skip(2) {
        *d = DimExpr::constant(1);
    }
    ctx.set_output(0, dtype, out);
    Ok(())
}

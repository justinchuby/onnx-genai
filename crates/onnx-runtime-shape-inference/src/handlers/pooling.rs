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
use crate::shape_data::ShapeData;

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
    /// Extra `ConvTranspose` trailing padding; ignored by the forward formula.
    output_padding: i64,
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
            output_padding: 0,
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

#[derive(Clone, Copy)]
struct PoolParams {
    kernel: i64,
    stride: i64,
    dilation: i64,
    pad_begin: i64,
    pad_end: i64,
    ceil_mode: bool,
}

fn checked_pool_extent(op: &str, input: i128, params: PoolParams) -> Result<i128, ShapeInferError> {
    let effective_kernel = i128::from(params.dilation)
        .checked_mul(i128::from(params.kernel) - 1)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| ShapeInferError::Invalid {
            op: op.into(),
            detail: "effective kernel arithmetic overflowed".into(),
        })?;
    let numerator = input
        .checked_add(i128::from(params.pad_begin))
        .and_then(|value| value.checked_add(i128::from(params.pad_end)))
        .and_then(|value| value.checked_sub(effective_kernel))
        .ok_or_else(|| ShapeInferError::Invalid {
            op: op.into(),
            detail: "pooling extent arithmetic overflowed".into(),
        })?;
    let stride = i128::from(params.stride);
    let quotient = if params.ceil_mode {
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
    if params.ceil_mode && output > 0 {
        let last_start =
            (output - 1)
                .checked_mul(stride)
                .ok_or_else(|| ShapeInferError::Invalid {
                    op: op.into(),
                    detail: "pooling window arithmetic overflowed".into(),
                })?;
        let right_padding_start =
            input
                .checked_add(i128::from(params.pad_begin))
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
    params: PoolParams,
    auto_pad: AutoPad,
) -> Result<DimExpr, ShapeInferError> {
    if params.kernel <= 0 || params.stride <= 0 || params.dilation <= 0 {
        return Err(ShapeInferError::Invalid {
            op: ctx.op().into(),
            detail: "kernel_shape, strides, and dilations must be positive".into(),
        });
    }

    let known = input.as_const();
    validate_pool_partial_extents(
        ctx.op(),
        known,
        params.kernel,
        params.dilation,
        params.pad_begin,
        params.pad_end,
    )?;
    let output = match auto_pad {
        AutoPad::SameUpper | AutoPad::SameLower => known.map(|extent| {
            let extent = i128::from(extent);
            extent.div_euclid(i128::from(params.stride))
                + i128::from(extent.rem_euclid(i128::from(params.stride)) != 0)
        }),
        AutoPad::Valid => known
            .map(i128::from)
            .map(|extent| {
                checked_pool_extent(
                    ctx.op(),
                    extent,
                    PoolParams {
                        pad_begin: 0,
                        pad_end: 0,
                        ..params
                    },
                )
            })
            .transpose()?,
        AutoPad::NotSet => known
            .map(i128::from)
            .map(|extent| checked_pool_extent(ctx.op(), extent, params))
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
        let lower_bound = checked_pool_extent(ctx.op(), 0, params)?;
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
            PoolParams {
                kernel: kernels[axis],
                stride: strides[axis],
                dilation: dilations[axis],
                pad_begin: pads[axis],
                pad_end: pads[axis + spatial_rank],
                ceil_mode,
            },
            auto_pad,
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

/// One `MaxUnpool` spatial output extent.
///
/// `MaxUnpool` is the partial inverse of `MaxPool`, so it applies the transpose
/// formula `stride*(input-1) - pad_begin - pad_end + kernel` (there is no
/// dilation or output padding). A symbolic input, a non-positive stride/kernel,
/// or an out-of-range result degrades to a fresh symbol.
fn max_unpool_dim(
    ctx: &mut InferenceContext,
    input: &DimExpr,
    kernel: i64,
    stride: i64,
    pad_begin: i64,
    pad_end: i64,
) -> DimExpr {
    let Some(d) = input.as_const() else {
        return ctx.fresh_dim();
    };
    if stride <= 0 || kernel <= 0 {
        return ctx.fresh_dim();
    }
    let value =
        i128::from(stride) * (i128::from(d) - 1) - i128::from(pad_begin) - i128::from(pad_end)
            + i128::from(kernel);
    if !(0..=isize::MAX as i128).contains(&value) {
        return ctx.fresh_dim();
    }
    DimExpr::constant(value as i64)
}

/// `MaxUnpool` (opset 9): scatter `X` `[N, C, D1, …]` back into a larger tensor.
///
/// When the optional `output_shape` input (slot 2, a 1-D tensor of the *full*
/// output dims) resolves to constants it is used verbatim; otherwise the
/// spatial extents follow the transpose formula from `kernel_shape`, `strides`,
/// and `pads`. `N`/`C` are always copied from `X`.
pub fn max_unpool(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(x) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    let dtype = ctx.input_dtype(0).unwrap_or(DataType::Float32);
    if x.len() < 3 {
        return Err(ShapeInferError::InvalidRank {
            op: "MaxUnpool".into(),
            index: 0,
            rank: x.len(),
            detail: "expected [N, C, D1, …]".into(),
        });
    }
    let n_spatial = x.len() - 2;
    // The optional `output_shape` input overrides the computed extents.
    if ctx.has_input(2) {
        let dims = ctx.input_shape_data(2).map(ShapeData::as_shape);
        let out = match dims {
            Some(dims) if dims.len() == x.len() => {
                for dim in &dims {
                    if let Some(extent) = dim.as_const()
                        && !(0..=isize::MAX as i64).contains(&extent)
                    {
                        return Err(ShapeInferError::Invalid {
                            op: "MaxUnpool".into(),
                            detail: format!(
                                "output_shape extent {extent} is outside 0..=isize::MAX"
                            ),
                        });
                    }
                }
                dims
            }
            // The dims are supplied at runtime but not statically known: keep
            // the rank (N, C, and one fresh symbol per spatial axis).
            _ => {
                let mut out = vec![x[0].clone(), x[1].clone()];
                out.extend((0..n_spatial).map(|_| ctx.fresh_dim()));
                out
            }
        };
        ctx.set_output(0, dtype, out);
        return Ok(());
    }
    let kernel = pool_list(ctx, "kernel_shape", n_spatial, 0, true)?;
    let strides = pool_list(ctx, "strides", n_spatial, 1, false)?;
    let pads = pool_list(ctx, "pads", n_spatial * 2, 0, false)?;
    let mut out = vec![x[0].clone(), x[1].clone()];
    for axis in 0..n_spatial {
        out.push(max_unpool_dim(
            ctx,
            &x[axis + 2],
            kernel[axis],
            strides[axis],
            pads[axis],
            pads[axis + n_spatial],
        ));
    }
    ctx.set_output(0, dtype, out);
    Ok(())
}

/// Register the spatial family.
pub fn register(reg: &mut InferenceRegistry) {
    reg.register("", "Conv", 1, conv);
    reg.register("", "ConvTranspose", 1, conv_transpose);
    reg.register("", "MaxPool", 1, pool);
    reg.register("", "AveragePool", 1, pool);
    // `LpPool` shares `MaxPool`/`AveragePool`'s windowed spatial formula (it
    // has the same `kernel_shape`/`strides`/`pads`/`auto_pad` attributes, plus
    // `dilations`/`ceil_mode` from opset 18), so it reuses `pool`.
    reg.register("", "LpPool", 1, pool);
    reg.register("", "GlobalAveragePool", 1, global_pool);
    reg.register("", "GlobalMaxPool", 1, global_pool);
    // `GlobalLpPool` collapses every spatial dim to 1, exactly like the other
    // global pools.
    reg.register("", "GlobalLpPool", 1, global_pool);
    reg.register("", "MaxUnpool", 9, max_unpool);
    reg.register("", "MaxUnpool", 11, max_unpool);
    reg.register("", "Pad", 1, pad);
    reg.register("", "GridSample", 16, grid_sample);
    reg.register("", "AffineGrid", 20, affine_grid);
}

/// One `ConvTranspose` spatial output extent.
///
/// With an explicit (non-auto) pad the deconvolution formula is
/// `stride*(input-1) + output_padding + effective_kernel - pad_begin - pad_end`.
/// Under `SAME_UPPER`/`SAME_LOWER` the spec fixes the extent at `input*stride`.
/// A symbolic input, a non-positive stride, or an overflowing/out-of-range
/// result degrades to a fresh symbol — an honest "unknown" rather than a
/// fabricated extent.
fn conv_transpose_dim(ctx: &mut InferenceContext, input: &DimExpr, p: &SpatialParams) -> DimExpr {
    let Some(d) = input.as_const() else {
        return ctx.fresh_dim();
    };
    if p.stride <= 0 {
        return ctx.fresh_dim();
    }
    let value = match p.auto {
        AutoPad::SameUpper | AutoPad::SameLower => i128::from(d) * i128::from(p.stride),
        AutoPad::Valid | AutoPad::NotSet => {
            let effective_kernel = i128::from(p.dilation) * (i128::from(p.kernel) - 1) + 1;
            i128::from(p.stride) * (i128::from(d) - 1)
                + i128::from(p.output_padding)
                + effective_kernel
                - i128::from(p.pad_begin)
                - i128::from(p.pad_end)
        }
    };
    if !(0..=isize::MAX as i128).contains(&value) {
        return ctx.fresh_dim();
    }
    DimExpr::constant(value as i64)
}

/// `ConvTranspose` (opset 1): the transpose (fractionally-strided) convolution.
///
/// Output channels are `W.shape[1] * group`; each spatial extent follows the
/// deconvolution formula, or the explicit `output_shape` attribute when given.
pub fn conv_transpose(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(x) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    let dtype = ctx.input_dtype(0).unwrap_or(DataType::Float32);
    if x.len() < 3 {
        return Err(ShapeInferError::InvalidRank {
            op: "ConvTranspose".into(),
            index: 0,
            rank: x.len(),
            detail: "expected [N, C, D1, …]".into(),
        });
    }
    let n_spatial = x.len() - 2;
    let group = ctx
        .node
        .attr("group")
        .and_then(Attribute::as_int)
        .unwrap_or(1)
        .max(1);
    // Output channels come from `W`'s second dimension (M / group) times group.
    let channels = match ctx.input_shape(1) {
        Some(w) if w.len() >= 2 => w[1].mul(&DimExpr::constant(group)),
        _ => ctx.fresh_dim(),
    };
    let kernel: Vec<i64> = match ctx.node.attr("kernel_shape").and_then(Attribute::as_ints) {
        Some(k) => k.to_vec(),
        None => match ctx.input_shape(1) {
            Some(w) if w.len() == n_spatial + 2 => {
                w[2..].iter().map(|d| d.as_const().unwrap_or(1)).collect()
            }
            _ => vec![1; n_spatial],
        },
    };
    let strides = int_list(ctx, "strides", n_spatial, 1);
    let dilations = int_list(ctx, "dilations", n_spatial, 1);
    let pads = int_list(ctx, "pads", n_spatial * 2, 0);
    let output_padding = int_list(ctx, "output_padding", n_spatial, 0);
    let output_shape = ctx
        .node
        .attr("output_shape")
        .and_then(Attribute::as_ints)
        .map(<[i64]>::to_vec);
    let auto = auto_pad(ctx);

    let mut out = Vec::with_capacity(x.len());
    out.push(x[0].clone());
    out.push(channels);
    for d in 0..n_spatial {
        let dim = if let Some(shape) = &output_shape {
            match shape.get(d).copied() {
                Some(value) if value >= 0 => DimExpr::constant(value),
                _ => ctx.fresh_dim(),
            }
        } else {
            let params = SpatialParams {
                kernel: *kernel.get(d).unwrap_or(&1),
                stride: *strides.get(d).unwrap_or(&1),
                dilation: *dilations.get(d).unwrap_or(&1),
                pad_begin: *pads.get(d).unwrap_or(&0),
                pad_end: *pads.get(d + n_spatial).unwrap_or(&0),
                auto,
                ceil_mode: false,
                output_padding: *output_padding.get(d).unwrap_or(&0),
            };
            conv_transpose_dim(ctx, &x[d + 2], &params)
        };
        out.push(dim);
    }
    ctx.set_output(0, dtype, out);
    Ok(())
}

/// `GridSample` (opset 16): sample `X` `[N, C, D1, …]` at the locations in
/// `grid` `[N, D1_out, …, spatial]`. The output keeps `N`/`C` from `X` and takes
/// each spatial extent from the matching `grid` dimension.
pub fn grid_sample(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(x) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    if x.len() < 3 {
        return Err(ShapeInferError::InvalidRank {
            op: "GridSample".into(),
            index: 0,
            rank: x.len(),
            detail: "expected [N, C, D1, …]".into(),
        });
    }
    let dtype = ctx.input_dtype(0).unwrap_or(DataType::Float32);
    let n_spatial = x.len() - 2;
    let mut out = vec![x[0].clone(), x[1].clone()];
    match ctx.input_shape(1) {
        Some(grid) if grid.len() == n_spatial + 2 => {
            out.extend(grid[1..=n_spatial].iter().cloned());
        }
        _ => out.extend((0..n_spatial).map(|_| ctx.fresh_dim())),
    }
    ctx.set_output(0, dtype, out);
    Ok(())
}

/// `AffineGrid` (opset 20): generate a sampling grid from a batch of affine
/// matrices `theta` and a target output `size`. For 2-D (`size = [N, C, H, W]`)
/// the grid is `[N, H, W, 2]`; for 3-D (`size = [N, C, D, H, W]`) it is
/// `[N, D, H, W, 3]`. The extents come from the resolved `size` vector, so the
/// rule needs `size`'s shape-data; the element type follows `theta`.
pub fn affine_grid(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(dtype) = ctx.input_dtype(0) else {
        return Ok(());
    };
    let Some(size) = ctx
        .input_shape_data(1)
        .filter(|data| !data.is_scalar())
        .map(|data| data.elems.clone())
    else {
        return Ok(());
    };
    let out = match size.len() {
        // 2-D: size = [N, C, H, W]  ->  grid = [N, H, W, 2].
        4 => vec![
            size[0].clone(),
            size[2].clone(),
            size[3].clone(),
            DimExpr::constant(2),
        ],
        // 3-D: size = [N, C, D, H, W]  ->  grid = [N, D, H, W, 3].
        5 => vec![
            size[0].clone(),
            size[2].clone(),
            size[3].clone(),
            size[4].clone(),
            DimExpr::constant(3),
        ],
        _ => return Ok(()),
    };
    ctx.set_output(0, dtype, out);
    Ok(())
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

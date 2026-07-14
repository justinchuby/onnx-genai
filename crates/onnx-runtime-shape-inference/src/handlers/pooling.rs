//! Spatial rules (stretch coverage): `Conv`, `MaxPool`, `AveragePool`, `Pad`.
//!
//! These use the standard spatial output formula
//! `floor((D + pad_begin + pad_end - dilation*(kernel-1) - 1) / stride) + 1`
//! (ceil when `ceil_mode` is set). A concrete spatial dim is computed exactly;
//! a symbolic one degrades to a fresh symbol so the output keeps a known rank.

use onnx_runtime_ir::{Attribute, DataType};

use crate::context::InferenceContext;
use crate::dim_expr::DimExpr;
use crate::error::ShapeInferError;
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

/// `MaxPool`/`AveragePool`: channels are preserved from the input.
pub fn pool(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let channels = ctx
        .input_shape(0)
        .and_then(|x| x.get(1).cloned())
        .unwrap_or_else(|| ctx.fresh_dim());
    conv_pool(ctx, channels, false)
}

/// `Pad`: each dim grows by its begin+end pad.
pub fn pad(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(x) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    let dtype = ctx.input_dtype(0).unwrap_or(DataType::Float32);
    let rank = x.len();

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

    let Some(pads) = pads else {
        // Unknown pads: keep the rank, refresh dims.
        let out = (0..rank).map(|_| ctx.fresh_dim()).collect();
        ctx.set_output(0, dtype, out);
        return Ok(());
    };
    let mut out = Vec::with_capacity(rank);
    for (i, d) in x.iter().enumerate() {
        let begin = pads.get(i).copied().unwrap_or(0);
        let end = pads.get(i + rank).copied().unwrap_or(0);
        out.push(d.add(&DimExpr::constant(begin + end)));
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
    let dtype = ctx.input_dtype(0).unwrap_or(DataType::Float32);
    let mut out = x.clone();
    for d in out.iter_mut().skip(2) {
        *d = DimExpr::constant(1);
    }
    ctx.set_output(0, dtype, out);
    Ok(())
}

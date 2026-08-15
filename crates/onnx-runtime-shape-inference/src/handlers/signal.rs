//! Signal-processing rules: the discrete Fourier transform family (`DFT`,
//! `STFT`), the mel-filterbank generator (`MelWeightMatrix`), and the cosine-sum
//! window generators (`HannWindow`, `HammingWindow`, `BlackmanWindow`).
//!
//! Every output extent these operators need — a transform length, a onesided
//! bin count, a window size — comes from a *scalar or vector integer input*
//! whose value flows in as shape-data. When that value is statically known the
//! rule emits a concrete extent; otherwise it degrades to a fresh symbol rather
//! than fabricating a constant, matching the crate's permissive philosophy. The
//! output rank is always fixed by the specification, so these rules resolve the
//! rank (and the always-known dimensions, such as the trailing complex `2`)
//! even when the data-dependent extents are unknown.

use onnx_runtime_ir::{Attribute, DataType};

use crate::context::InferenceContext;
use crate::dim_expr::DimExpr;
use crate::error::ShapeInferError;
use crate::registry::InferenceRegistry;

/// Read input slot `index` as a statically-known integer scalar, if its
/// shape-data resolved to a rank-0 constant.
fn scalar_int(ctx: &InferenceContext, index: usize) -> Option<i64> {
    let data = ctx.input_shape_data(index)?;
    if !data.is_scalar() {
        return None;
    }
    data.elems.first()?.as_const()
}

/// The `output_datatype` attribute as a [`DataType`] (default Float32), used by
/// the window and mel-matrix generators to type their output.
fn output_datatype(ctx: &InferenceContext) -> DataType {
    ctx.node
        .attr("output_datatype")
        .and_then(Attribute::as_int)
        .and_then(|raw| i32::try_from(raw).ok())
        .and_then(DataType::from_onnx)
        .unwrap_or(DataType::Float32)
}

/// Whether integer attribute `name` is set and non-zero (a boolean flag).
fn flag(ctx: &InferenceContext, name: &str) -> bool {
    ctx.node.attr(name).and_then(Attribute::as_int).unwrap_or(0) != 0
}

/// The size of the onesided (nonredundant) half-spectrum of an `n`-point DFT:
/// `floor(n / 2) + 1`.
fn onesided_bins(n: i64) -> i64 {
    (n >> 1) + 1
}

/// Normalize and validate a DFT `axis`. The last dimension carries the
/// real/imaginary components and is never a valid signal axis, so the accepted
/// range is `[-rank, -2] ∪ [0, rank - 2]`. Returns the non-negative index, or
/// `None` when the axis is out of range.
fn dft_axis_index(axis: i64, rank: usize) -> Option<usize> {
    let r = rank as i64;
    if axis < -r || axis == -1 || axis >= r - 1 {
        return None;
    }
    let normalized = if axis >= 0 { axis } else { axis + r };
    usize::try_from(normalized).ok()
}

/// `DFT` (opset 17 and 20): the discrete Fourier transform along one signal
/// axis.
///
/// The output matches the input rank. The trailing dimension is always coerced
/// to `2` (real and imaginary parts). The signal axis becomes `dft_length` when
/// that optional scalar input is given, and is halved to `floor(n / 2) + 1` when
/// `onesided` is set. In opset 20 the axis moved from an attribute (default `1`)
/// to an optional scalar input (default `-2`); when that input is present but
/// not statically known, only the rank and the trailing `2` can be resolved.
fn dft(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(input) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    let dtype = input.dtype;
    let rank = input.shape.len();
    if rank < 2 {
        return Err(ShapeInferError::InvalidRank {
            op: "DFT".into(),
            index: 0,
            rank,
            detail: "input must have rank >= 2 (including the complex dimension)".into(),
        });
    }

    let onesided = flag(ctx, "onesided");
    let has_dft_length = ctx.has_input(1);
    let last = rank - 1;

    // Opset 20 carries the axis as an optional input; earlier opsets use an
    // attribute. The default differs accordingly: `-2` for the input form,
    // `1` for the attribute form.
    let axis_is_input = ctx.opset("") >= 20 && ctx.has_input(2);
    let axis_value: Option<i64> = if axis_is_input {
        scalar_int(ctx, 2)
    } else {
        let default_axis = if ctx.opset("") >= 20 { -2 } else { 1 };
        Some(
            ctx.node
                .attr("axis")
                .and_then(Attribute::as_int)
                .unwrap_or(default_axis),
        )
    };

    let mut out = input.shape.clone();

    let Some(axis) = axis_value else {
        // Opset-20 axis input present but not statically known. When onesided or
        // dft_length would rewrite the signal axis we cannot know which axis
        // that is, so every signal extent becomes a fresh symbol; otherwise the
        // input shape is preserved. Either way the trailing dimension is 2.
        if onesided || has_dft_length {
            out = (0..rank).map(|_| ctx.fresh_dim()).collect();
        }
        out[last] = DimExpr::constant(2);
        ctx.set_output(0, dtype, out);
        return Ok(());
    };

    let Some(axis_idx) = dft_axis_index(axis, rank) else {
        return Err(ShapeInferError::Invalid {
            op: "DFT".into(),
            detail: format!("axis {axis} is invalid for a tensor of rank {rank}"),
        });
    };

    if has_dft_length {
        out[axis_idx] = match scalar_int(ctx, 1) {
            Some(length) => DimExpr::constant(length),
            None => ctx.fresh_dim(),
        };
    }
    if onesided {
        out[axis_idx] = match out[axis_idx].as_const() {
            Some(n) => DimExpr::constant(onesided_bins(n)),
            None => ctx.fresh_dim(),
        };
    }
    out[last] = DimExpr::constant(2);
    ctx.set_output(0, dtype, out);
    Ok(())
}

/// `STFT` (opset 17): the short-time Fourier transform.
///
/// The signal is `[batch, signal_length, 1|2]`; the output is
/// `[batch, frames, dft_unique_bins, 2]`. The transform size comes from the
/// optional `frame_length` scalar or, failing that, the length of the optional
/// `window` vector. `dft_unique_bins` is `floor(size / 2) + 1` when `onesided`
/// (the spec default) is set, else the full `size`. The frame count is
/// `floor((signal_length - size) / frame_step) + 1`. Any extent that cannot be
/// resolved statically degrades to a fresh symbol; the rank-4 shape and the
/// trailing complex `2` are always emitted.
fn stft(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(signal) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    let dtype = signal.dtype;
    let rank = signal.shape.len();
    if rank < 2 {
        return Err(ShapeInferError::InvalidRank {
            op: "STFT".into(),
            index: 0,
            rank,
            detail: "signal must have rank >= 2".into(),
        });
    }

    let batch = signal.shape[0].clone();
    let signal_length = signal.shape[1].as_const();
    let frame_step = scalar_int(ctx, 1);

    // The transform size is the frame_length scalar if present, else the length
    // of the window vector.
    let frame_length = ctx.has_input(3).then(|| scalar_int(ctx, 3)).flatten();
    let window_length = ctx.has_input(2).then(|| {
        ctx.input_shape(2)
            .and_then(<[DimExpr]>::first)
            .and_then(DimExpr::as_const)
    });
    let dft_size = frame_length.or(window_length.flatten());

    // STFT's `onesided` defaults to 1, unlike DFT.
    let onesided = ctx
        .node
        .attr("onesided")
        .and_then(Attribute::as_int)
        .unwrap_or(1)
        != 0;

    let bins = dft_size.map(|size| if onesided { onesided_bins(size) } else { size });
    let frames = match (signal_length, dft_size, frame_step) {
        (Some(length), Some(size), Some(step)) if step != 0 => Some((length - size) / step + 1),
        _ => None,
    };

    let frames_dim = frames.map_or_else(|| ctx.fresh_dim(), DimExpr::constant);
    let bins_dim = bins.map_or_else(|| ctx.fresh_dim(), DimExpr::constant);
    ctx.set_output(
        0,
        dtype,
        vec![batch, frames_dim, bins_dim, DimExpr::constant(2)],
    );
    Ok(())
}

/// `MelWeightMatrix` (opset 17): a `[floor(dft_length / 2) + 1, num_mel_bins]`
/// filterbank whose element type is `output_datatype` (default Float32). Both
/// extents come from scalar integer inputs; an unresolved one degrades to a
/// fresh symbol.
fn mel_weight_matrix(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let dtype = output_datatype(ctx);
    let rows = match scalar_int(ctx, 1) {
        Some(dft_length) if dft_length > 0 => DimExpr::constant(onesided_bins(dft_length)),
        _ => ctx.fresh_dim(),
    };
    let cols = match scalar_int(ctx, 0) {
        Some(num_mel_bins) if num_mel_bins > 0 => DimExpr::constant(num_mel_bins),
        _ => ctx.fresh_dim(),
    };
    ctx.set_output(0, dtype, vec![rows, cols]);
    Ok(())
}

/// `HannWindow`/`HammingWindow`/`BlackmanWindow` (opset 17): a 1-D window of
/// length `size` (a scalar integer input) whose element type is
/// `output_datatype` (default Float32). An unknown `size` degrades to a fresh
/// symbol; the rank-1 shape is always emitted.
fn window(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let dtype = output_datatype(ctx);
    let length = match scalar_int(ctx, 0) {
        Some(size) if size > 0 => DimExpr::constant(size),
        _ => ctx.fresh_dim(),
    };
    ctx.set_output(0, dtype, vec![length]);
    Ok(())
}

/// Register the signal-processing family.
pub fn register(reg: &mut InferenceRegistry) {
    // `DFT` gains an optional `axis` input at opset 20; the rule handles both
    // schema forms, so it is registered at each boundary.
    reg.register("", "DFT", 17, dft);
    reg.register("", "DFT", 20, dft);
    reg.register("", "STFT", 17, stft);
    reg.register("", "MelWeightMatrix", 17, mel_weight_matrix);
    reg.register("", "HannWindow", 17, window);
    reg.register("", "HammingWindow", 17, window);
    reg.register("", "BlackmanWindow", 17, window);
}

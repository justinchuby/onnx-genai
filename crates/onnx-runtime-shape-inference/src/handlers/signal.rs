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

/// The size of the onesided (nonredundant) half-spectrum of an `n`-point DFT:
/// `floor(n / 2) + 1`.
fn onesided_bins(n: i64) -> i64 {
    (n >> 1) + 1
}

/// `DFT` (opset 17 and 20): the discrete Fourier transform along one signal
/// axis.
///
/// The output matches the input rank. Full/RFFT output has two components;
/// IRFFT has one real component and a default length of twice (input bins - 1). The signal axis becomes `dft_length` when
/// that optional scalar input is given. Only forward onesided output is halved
/// to `floor(n / 2) + 1`. In opset 20 the axis moved from an attribute (default `1`)
/// to an optional scalar input (default `-2`); when that input is present but
/// not statically known, the component width still follows the transform mode.
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

    let invalid = |detail| ShapeInferError::Invalid {
        op: "DFT".into(),
        detail,
    };
    let mode = onnx_runtime_ir::dft::DftMode::parse(ctx.node).map_err(invalid)?;
    let has_dft_length = ctx.has_input(1);
    let last = rank - 1;
    if let Some(components) = input.shape[last].as_const() {
        let components = usize::try_from(components)
            .map_err(|_| invalid(format!("invalid input component dimension {components}")))?;
        mode.validate_components(components).map_err(invalid)?;
    }

    // Opset 20 carries the axis as an optional input; earlier opsets use an
    // attribute. The default differs accordingly: `-2` for the input form,
    // `1` for the attribute form.
    let axis_is_input = ctx.opset("") >= 20 && ctx.has_input(2);
    let axis_value: Option<i64> = if axis_is_input {
        scalar_int(ctx, 2)
    } else {
        Some(onnx_runtime_ir::dft::default_axis(ctx.node, ctx.opset("")))
    };

    let mut out = input.shape.clone();

    let Some(axis) = axis_value else {
        // Opset-20 axis input present but not statically known. When onesided or
        // dft_length would rewrite the signal axis we cannot know which axis
        // that is, so every signal extent becomes a fresh symbol; otherwise the
        // input shape is preserved. Component width follows the transform mode.
        if matches!(
            mode,
            onnx_runtime_ir::dft::DftMode::RealForward | onnx_runtime_ir::dft::DftMode::RealInverse
        ) || has_dft_length
        {
            out = (0..rank).map(|_| ctx.fresh_dim()).collect();
        }
        out[last] = DimExpr::constant(mode.output_components() as i64);
        ctx.set_output(0, dtype, out);
        return Ok(());
    };

    let axis_idx = onnx_runtime_ir::dft::normalize_axis(axis, rank).map_err(invalid)?;

    let length = if has_dft_length {
        scalar_int(ctx, 1)
            .map(onnx_runtime_ir::dft::explicit_length)
            .transpose()
            .map_err(invalid)?
    } else {
        input.shape[axis_idx]
            .as_const()
            .map(|n| {
                let n = usize::try_from(n)
                    .map_err(|_| invalid(format!("invalid signal-axis extent {n}")))?;
                mode.default_length(n).map_err(invalid)
            })
            .transpose()?
    };
    out[axis_idx] = if !has_dft_length
        && matches!(
            mode,
            onnx_runtime_ir::dft::DftMode::Forward | onnx_runtime_ir::dft::DftMode::Inverse
        ) {
        input.shape[axis_idx].clone()
    } else {
        match length {
            Some(n) => DimExpr::constant(i64::try_from(mode.output_length(n)).map_err(|_| {
                invalid(format!(
                    "DFT output extent for length {n} exceeds i64; reduce the transform length"
                ))
            })?),
            None => ctx.fresh_dim(),
        }
    };
    out[last] = DimExpr::constant(mode.output_components() as i64);
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
    if rank != 3 {
        return Err(ShapeInferError::InvalidRank {
            op: "STFT".into(),
            index: 0,
            rank,
            detail: "signal must have rank 3: [batch, signal_length, 1|2]".into(),
        });
    }

    if let Some(components) = signal.shape[2].as_const()
        && components != 1
        && components != 2
    {
        return Err(ShapeInferError::Invalid {
            op: "STFT".into(),
            detail: format!(
                "signal's last dimension must be 1 (real) or 2 (complex), got {components}"
            ),
        });
    }

    let batch = signal.shape[0].clone();
    let signal_length = signal.shape[1].as_const();
    if let Some(shape) = ctx.input_shape(1)
        && !shape.is_empty()
    {
        return Err(ShapeInferError::InvalidRank {
            op: "STFT".into(),
            index: 1,
            rank: shape.len(),
            detail: "frame_step must be a scalar".into(),
        });
    }
    let frame_step = scalar_int(ctx, 1);
    if let Some(step) = frame_step
        && step <= 0
    {
        return Err(ShapeInferError::Invalid {
            op: "STFT".into(),
            detail: format!("frame_step must be greater than zero, got {step}"),
        });
    }

    let has_window = ctx.has_input(2);
    let has_frame_length = ctx.has_input(3);
    if !has_window && !has_frame_length {
        return Err(ShapeInferError::Invalid {
            op: "STFT".into(),
            detail: "either optional window or frame_length must be provided".into(),
        });
    }

    let window_length = if has_window {
        let shape = ctx.input_shape(2).ok_or_else(|| ShapeInferError::Invalid {
            op: "STFT".into(),
            detail: "window input has no tensor shape".into(),
        })?;
        if shape.len() != 1 {
            return Err(ShapeInferError::InvalidRank {
                op: "STFT".into(),
                index: 2,
                rank: shape.len(),
                detail: "window must have rank 1".into(),
            });
        }
        let length = shape[0].as_const();
        if let Some(length) = length
            && length <= 0
        {
            return Err(ShapeInferError::Invalid {
                op: "STFT".into(),
                detail: format!("window length must be greater than zero, got {length}"),
            });
        }
        length
    } else {
        None
    };

    let frame_length = if has_frame_length {
        if let Some(shape) = ctx.input_shape(3)
            && !shape.is_empty()
        {
            return Err(ShapeInferError::InvalidRank {
                op: "STFT".into(),
                index: 3,
                rank: shape.len(),
                detail: "frame_length must be a scalar".into(),
            });
        }
        let length = scalar_int(ctx, 3);
        if let Some(length) = length
            && length <= 0
        {
            return Err(ShapeInferError::Invalid {
                op: "STFT".into(),
                detail: format!("frame_length must be greater than zero, got {length}"),
            });
        }
        length
    } else {
        None
    };
    if let (Some(window), Some(frame)) = (window_length, frame_length)
        && window != frame
    {
        return Err(ShapeInferError::Invalid {
            op: "STFT".into(),
            detail: format!("window length {window} must equal frame_length {frame}"),
        });
    }
    // When frame_length is present but value-unknown, the transform size stays
    // unknown even if the window length is known: equality still has to be
    // validated once the scalar value arrives.
    let dft_size = if has_frame_length {
        frame_length
    } else {
        window_length
    };

    // STFT's `onesided` defaults to 1, unlike DFT.
    let onesided = ctx
        .node
        .attr("onesided")
        .and_then(Attribute::as_int)
        .unwrap_or(1)
        != 0;
    if onesided && signal.shape[2].as_const() == Some(2) {
        return Err(ShapeInferError::Invalid {
            op: "STFT".into(),
            detail: "onesided=1 requires a real signal (last dimension 1)".into(),
        });
    }

    let bins = dft_size.map(|size| if onesided { onesided_bins(size) } else { size });
    let frames = match (signal_length, dft_size, frame_step) {
        (Some(length), Some(size), Some(_)) if size > length => {
            return Err(ShapeInferError::Invalid {
                op: "STFT".into(),
                detail: format!(
                    "frame length {size} exceeds signal length {length}; STFT uses complete unpadded frames"
                ),
            });
        }
        (Some(length), Some(size), Some(step)) => Some((length - size) / step + 1),
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

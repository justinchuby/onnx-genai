//! Sequence-construction and scan rules: `Tile`, `Range`, and `CumSum`.

use crate::context::InferenceContext;
use crate::dim_expr::DimExpr;
use crate::error::ShapeInferError;
use crate::registry::InferenceRegistry;

fn const_ints(ctx: &InferenceContext, input: usize) -> Option<Vec<i64>> {
    ctx.input_shape_data(input)?
        .elems
        .iter()
        .map(DimExpr::as_const)
        .collect()
}

/// A statically-known floating-point scalar constant at input `input`, if any.
fn const_float_scalar(ctx: &InferenceContext, input: usize) -> Option<f64> {
    ctx.input_shape_data(input)?.as_float_scalar()
}

/// `Tile` (opset 6/13): multiply every input extent by the corresponding static
/// repeat. Takes two inputs — `input` and a 1-D `repeats` (int64, one entry per
/// input dimension); output dim[i] == input dim[i] × repeats[i]. The rank is
/// always known (== rank(input)); extents with an unknown repeat degrade to a
/// fresh symbol.
fn tile(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(input) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    let repeats = const_ints(ctx, 1);
    let mut output = Vec::with_capacity(input.rank());
    for (i, dim) in input.shape.iter().enumerate() {
        output.push(
            repeats
                .as_ref()
                .filter(|r| r.len() == input.rank())
                .and_then(|r| r.get(i))
                .filter(|&&n| n >= 0)
                .map(|&n| dim.mul(&DimExpr::constant(n)))
                .unwrap_or_else(|| ctx.fresh_dim()),
        );
    }
    ctx.set_output(0, input.dtype, output);
    Ok(())
}

enum RangeLength {
    Known(i64),
    Unknown,
    TooLarge,
}

/// Compute a `Range` length for integer scalar inputs, if all values are known.
fn range_len(start: i64, limit: i64, delta: i64) -> RangeLength {
    if delta == 0 {
        return RangeLength::Unknown;
    }
    let start = i128::from(start);
    let limit = i128::from(limit);
    let delta = i128::from(delta);
    let distance = limit - start;
    let count = if (distance > 0 && delta > 0) || (distance < 0 && delta < 0) {
        let distance = distance.abs();
        let step = delta.abs();
        (distance + step - 1) / step
    } else {
        0
    };
    if count > isize::MAX as i128 {
        RangeLength::TooLarge
    } else {
        RangeLength::Known(count as i64)
    }
}

/// Compute a Float32 `Range` length with the same arithmetic as the CPU
/// kernel's `float_range_count`.
fn range_len_f32(start: f64, limit: f64, delta: f64) -> RangeLength {
    let start = start as f32;
    let limit = limit as f32;
    let delta = delta as f32;
    if delta == 0.0 || !start.is_finite() || !limit.is_finite() || !delta.is_finite() {
        return RangeLength::Unknown;
    }
    let count = ((limit - start) / delta).ceil().max(0.0);
    // `isize::MAX` rounds up to the next power of two in f32. Equality is
    // therefore already outside the representable positive-isize range.
    if !count.is_finite() || count >= isize::MAX as f32 {
        return RangeLength::TooLarge;
    }
    RangeLength::Known(count as i64)
}

/// Compute a Float64 `Range` length. Handles negative `delta` (descending
/// ranges) and rejects a zero/non-finite `delta` or a count that overflows an
/// `i64`.
fn range_len_f64(start: f64, limit: f64, delta: f64) -> RangeLength {
    if delta == 0.0 || !start.is_finite() || !limit.is_finite() || !delta.is_finite() {
        return RangeLength::Unknown;
    }
    let count = ((limit - start) / delta).ceil().max(0.0);
    // `isize::MAX` rounds up to the next power of two in f64. Equality is
    // therefore already outside the representable positive-isize range.
    if !count.is_finite() || count >= isize::MAX as f64 {
        return RangeLength::TooLarge;
    }
    RangeLength::Known(count as i64)
}

/// `Range`: output is a one-dimensional tensor with a static length only when
/// all scalar operands are statically known. Supports both integer and
/// floating-point (`Float32`/`Float64`) constant operands.
fn range(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(dtype) = ctx.input_dtype(0) else {
        return Ok(());
    };
    let length = range_int_len(ctx)
        .or_else(|| range_float_len(ctx, dtype))
        .unwrap_or(RangeLength::Unknown);
    let dim = match length {
        RangeLength::Known(length) => DimExpr::constant(length),
        RangeLength::Unknown => ctx.fresh_dim(),
        RangeLength::TooLarge => {
            return Err(ShapeInferError::Invalid {
                op: "Range".into(),
                detail: "output length exceeds isize::MAX".into(),
            });
        }
    };
    ctx.set_output(0, dtype, vec![dim]);
    Ok(())
}

/// The `Range` length when all three operands are integer scalar constants.
fn range_int_len(ctx: &InferenceContext) -> Option<RangeLength> {
    match (const_ints(ctx, 0), const_ints(ctx, 1), const_ints(ctx, 2)) {
        (Some(start), Some(limit), Some(delta))
            if start.len() == 1 && limit.len() == 1 && delta.len() == 1 =>
        {
            Some(range_len(start[0], limit[0], delta[0]))
        }
        _ => None,
    }
}

/// The `Range` length when all three operands are floating-point scalar
/// constants.
fn range_float_len(
    ctx: &InferenceContext,
    dtype: onnx_runtime_ir::DataType,
) -> Option<RangeLength> {
    let start = const_float_scalar(ctx, 0)?;
    let limit = const_float_scalar(ctx, 1)?;
    let delta = const_float_scalar(ctx, 2)?;
    match dtype {
        onnx_runtime_ir::DataType::Float32 => Some(range_len_f32(start, limit, delta)),
        onnx_runtime_ir::DataType::Float64 => Some(range_len_f64(start, limit, delta)),
        _ => None,
    }
}

/// `CumSum` does not change the input tensor's shape or dtype.
fn cum_sum(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    if let Some(input) = ctx.input_type(0).cloned() {
        ctx.set_output_type(0, input);
    }
    Ok(())
}

/// Register sequence-construction and scan rules.
pub fn register(reg: &mut InferenceRegistry) {
    // The CPU kernel implements only the modern two-input `(input, repeats)`
    // form, registered at opset 6; there is no attribute-based opset-1 path.
    reg.register("", "Tile", 6, tile);
    reg.register("", "Range", 11, range);
    reg.register("", "CumSum", 11, cum_sum);
    reg.register("", "CumSum", 14, cum_sum);
}

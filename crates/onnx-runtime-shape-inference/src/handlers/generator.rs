//! Random-generator rules: `RandomNormal`, `RandomUniform`, their `*Like`
//! variants, `Bernoulli`, and `Multinomial`.
//!
//! The shape-generating ops read their extents from the `shape` attribute; the
//! `*Like`/`Bernoulli` ops mirror the input tensor's shape. In every case the
//! element type comes from the optional `dtype` attribute (an ONNX
//! `TensorProto.DataType` integer), falling back to a spec-defined default.

use onnx_runtime_ir::{Attribute, DataType};

use crate::context::InferenceContext;
use crate::dim_expr::DimExpr;
use crate::error::ShapeInferError;
use crate::registry::InferenceRegistry;

/// The `dtype` attribute as a [`DataType`], if present and recognised.
fn dtype_attr(ctx: &InferenceContext) -> Option<DataType> {
    let raw = ctx.node.attr("dtype").and_then(Attribute::as_int)?;
    i32::try_from(raw).ok().and_then(DataType::from_onnx)
}

/// `RandomNormal`/`RandomUniform` (opset 1): a fresh tensor whose extents are
/// the `shape` attribute and whose dtype is `dtype` (default Float32).
fn random(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let dtype = dtype_attr(ctx).unwrap_or(DataType::Float32);
    let Some(shape) = ctx
        .node
        .attr("shape")
        .and_then(Attribute::as_ints)
        .map(<[i64]>::to_vec)
    else {
        return Ok(());
    };
    let dims = shape
        .into_iter()
        .map(|extent| {
            if extent >= 0 {
                DimExpr::constant(extent)
            } else {
                ctx.fresh_dim()
            }
        })
        .collect();
    ctx.set_output(0, dtype, dims);
    Ok(())
}

/// `RandomNormalLike`/`RandomUniformLike`/`Bernoulli`: same shape as the input;
/// dtype is the `dtype` attribute when given, else the input's own dtype.
fn random_like(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(input) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    let dtype = dtype_attr(ctx).unwrap_or(input.dtype);
    ctx.set_output(0, dtype, input.shape);
    Ok(())
}

/// `Multinomial` (opset 7): draws `sample_size` samples per batch row. Input is
/// `[batch, classes]`; output is `[batch, sample_size]` with an integer dtype
/// (`dtype`, default Int32).
fn multinomial(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(input) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    if input.len() != 2 {
        return Err(ShapeInferError::InvalidRank {
            op: "Multinomial".into(),
            index: 0,
            rank: input.len(),
            detail: "expected [batch, classes]".into(),
        });
    }
    let dtype = dtype_attr(ctx).unwrap_or(DataType::Int32);
    let sample_size = ctx
        .node
        .attr("sample_size")
        .and_then(Attribute::as_int)
        .unwrap_or(1);
    let samples = if sample_size >= 0 {
        DimExpr::constant(sample_size)
    } else {
        ctx.fresh_dim()
    };
    ctx.set_output(0, dtype, vec![input[0].clone(), samples]);
    Ok(())
}

/// Register the random-generator family.
pub fn register(reg: &mut InferenceRegistry) {
    reg.register("", "RandomNormal", 1, random);
    reg.register("", "RandomUniform", 1, random);
    reg.register("", "RandomNormalLike", 1, random_like);
    reg.register("", "RandomUniformLike", 1, random_like);
    reg.register("", "Bernoulli", 15, random_like);
    reg.register("", "Multinomial", 7, multinomial);
}

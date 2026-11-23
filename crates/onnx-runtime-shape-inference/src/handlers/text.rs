//! Standard ONNX text operators (opset 20).

use onnx_runtime_ir::DataType;

use crate::{context::InferenceContext, error::ShapeInferError, registry::InferenceRegistry};

fn regex_full_match(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    if let Some(shape) = ctx.input_shape(0).map(<[_]>::to_vec) {
        ctx.set_output(0, DataType::Bool, shape);
    }
    Ok(())
}

fn string_concat(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    if let (Some(a), Some(b)) = (
        ctx.input_shape(0).map(<[_]>::to_vec),
        ctx.input_shape(1).map(<[_]>::to_vec),
    ) {
        let shape = ctx.broadcast(&a, &b)?;
        ctx.set_output(0, DataType::String, shape);
    }
    Ok(())
}

fn string_split(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    if let Some(shape) = ctx.input_shape(0).map(<[_]>::to_vec) {
        let mut substrings = shape.clone();
        substrings.push(ctx.fresh_dim());
        ctx.set_output(0, DataType::String, substrings);
        ctx.set_output(1, DataType::Int64, shape);
    }
    Ok(())
}

pub fn register(reg: &mut InferenceRegistry) {
    reg.register("", "RegexFullMatch", 20, regex_full_match);
    reg.register("", "StringConcat", 20, string_concat);
    reg.register("", "StringSplit", 20, string_split);
}

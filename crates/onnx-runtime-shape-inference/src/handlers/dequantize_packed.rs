use onnx_runtime_ir::dequantize_packed::Metadata;

use crate::{context::InferenceContext, dim_expr::DimExpr, error::ShapeInferError};

pub(super) fn infer(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let invalid = |detail| ShapeInferError::Invalid {
        op: "DequantizePacked".into(),
        detail,
    };
    let metadata = Metadata::from_node(ctx.node).map_err(invalid)?;
    if let Some(dtype) = ctx.input_dtype(0) {
        metadata.validate_input_dtype(dtype).map_err(invalid)?;
    }
    if let Some(shape) = ctx.input_shape(0) {
        metadata.validate_input_rank(shape.len()).map_err(invalid)?;
        if let Some(n) = shape[0].as_const() {
            let len = usize::try_from(n).map_err(|_| {
                invalid(format!(
                    "packed input length {n} must be nonnegative and fit usize"
                ))
            })?;
            metadata.validate_packed_bytes(len).map_err(invalid)?;
        }
    }
    ctx.set_output(
        0,
        metadata.dtype(),
        metadata
            .shape()
            .iter()
            .map(|&n| DimExpr::constant(n as i64))
            .collect(),
    );
    Ok(())
}

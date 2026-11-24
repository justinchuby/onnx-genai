use onnx_runtime_ir::roi_align::RoiAlign;

use crate::{DimExpr, InferenceContext, ShapeInferError};

pub fn infer(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let invalid = |detail| ShapeInferError::Invalid {
        op: "RoiAlign".into(),
        detail,
    };
    let params = RoiAlign::parse(ctx.node, ctx.opset(&ctx.node.domain)).map_err(invalid)?;
    let dtype = ctx.input_dtype(0);
    params
        .validate_dtypes(dtype, ctx.input_dtype(1), ctx.input_dtype(2))
        .map_err(invalid)?;
    let x = ctx.input_shape(0);
    let rois = ctx.input_shape(1);
    let indices = ctx.input_shape(2);
    RoiAlign::validate_shapes(x, rois, indices, |d| {
        d.as_const().and_then(|n| usize::try_from(n).ok())
    })
    .map_err(invalid)?;
    let Some(dtype) = dtype else { return Ok(()) };
    let count = [indices, rois]
        .into_iter()
        .flatten()
        .map(|shape| &shape[0])
        .max_by_key(|d| d.as_const().is_some())
        .cloned();
    let channels = x.map(|shape| shape[1].clone());
    let count = count.unwrap_or_else(|| ctx.fresh_dim());
    let channels = channels.unwrap_or_else(|| ctx.fresh_dim());
    ctx.set_output(
        0,
        dtype,
        vec![
            count,
            channels,
            DimExpr::constant(params.output_height as i64),
            DimExpr::constant(params.output_width as i64),
        ],
    );
    Ok(())
}

//! Loss rules: `NegativeLogLikelihoodLoss` and `SoftmaxCrossEntropyLoss`.
//!
//! Both reduce a per-element loss to either a per-sample tensor (when
//! `reduction` is `none`) or a scalar (for `mean`/`sum`, the default). The
//! output element type is that of the floating-point score/input tensor. The
//! reduced (scalar) form is a rank-0 tensor, represented here by an empty shape.

use onnx_runtime_ir::Attribute;

use crate::context::InferenceContext;
use crate::error::ShapeInferError;
use crate::registry::InferenceRegistry;

/// The `reduction` attribute, defaulting to `mean`. Only the `none` value
/// changes the output shape; `mean` and `sum` both reduce to a scalar.
fn is_reduction_none(ctx: &InferenceContext) -> bool {
    ctx.node
        .attr("reduction")
        .and_then(Attribute::as_str)
        .unwrap_or("mean")
        == "none"
}

/// `NegativeLogLikelihoodLoss` (opset 12+): input `(N, C, d1, …, dk)` and target
/// `(N, d1, …, dk)`. With `reduction = none` the loss is `(N, d1, …, dk)` — the
/// batch dimension followed by the spatial dimensions, dropping the class axis.
/// Otherwise the loss is a scalar.
fn nll_loss(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(input) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    let dtype = input.dtype;
    let rank = input.shape.len();
    if rank < 2 {
        return Err(ShapeInferError::InvalidRank {
            op: "NegativeLogLikelihoodLoss".into(),
            index: 0,
            rank,
            detail: "input rank must be >= 2 (N, C, ...)".into(),
        });
    }

    if is_reduction_none(ctx) {
        let mut shape = Vec::with_capacity(rank - 1);
        shape.push(input.shape[0].clone());
        shape.extend(input.shape[2..].iter().cloned());
        ctx.set_output(0, dtype, shape);
    } else {
        ctx.set_output(0, dtype, Vec::new());
    }
    Ok(())
}

/// `SoftmaxCrossEntropyLoss` (opset 12+): scores `(N, C)` or `(N, C, d1, …, dk)`
/// and labels `(N)` or `(N, d1, …, dk)`. With `reduction = none` the loss has
/// the labels' shape; otherwise it is a scalar. The optional second output
/// `log_prob` always mirrors the scores' type and shape.
fn softmax_cross_entropy_loss(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(scores) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    let dtype = scores.dtype;

    if is_reduction_none(ctx) {
        // The loss follows the labels' shape; leave it unresolved when the
        // labels' shape is unknown.
        if let Some(labels) = ctx.input_shape(1).map(<[_]>::to_vec) {
            ctx.set_output(0, dtype, labels);
        }
    } else {
        ctx.set_output(0, dtype, Vec::new());
    }

    if ctx.num_outputs() >= 2 {
        ctx.set_output(1, dtype, scores.shape.clone());
    }
    Ok(())
}

/// Register the loss family.
pub fn register(reg: &mut InferenceRegistry) {
    reg.register("", "NegativeLogLikelihoodLoss", 12, nll_loss);
    reg.register(
        "",
        "SoftmaxCrossEntropyLoss",
        12,
        softmax_cross_entropy_loss,
    );
}

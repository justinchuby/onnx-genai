//! Recurrent neural-network rules: `RNN`, `GRU`, and `LSTM`.

use onnx_runtime_ir::Attribute;

use crate::context::InferenceContext;
use crate::dim_expr::DimExpr;
use crate::error::ShapeInferError;
use crate::registry::InferenceRegistry;

fn recurrent(
    ctx: &mut InferenceContext,
    state_outputs: usize,
    honor_layout: bool,
) -> Result<(), ShapeInferError> {
    let Some(x) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    if x.shape.len() != 3 {
        return Ok(());
    }
    let Some(hidden_size) = ctx
        .node
        .attr("hidden_size")
        .and_then(Attribute::as_int)
        .filter(|&size| size > 0)
    else {
        return Ok(());
    };
    let num_directions = match ctx
        .node
        .attr("direction")
        .and_then(Attribute::as_str)
        .unwrap_or("forward")
    {
        "forward" | "reverse" => 1,
        "bidirectional" => 2,
        _ => return Ok(()),
    };
    let batch_major = honor_layout
        && ctx
            .node
            .attr("layout")
            .and_then(Attribute::as_int)
            .unwrap_or(0)
            == 1;
    let (sequence, batch) = if batch_major {
        (x.shape[1].clone(), x.shape[0].clone())
    } else {
        (x.shape[0].clone(), x.shape[1].clone())
    };
    let directions = DimExpr::constant(num_directions);
    let hidden = DimExpr::constant(hidden_size);
    let y_shape = if batch_major {
        vec![batch.clone(), sequence, directions.clone(), hidden.clone()]
    } else {
        vec![sequence, directions.clone(), batch.clone(), hidden.clone()]
    };
    let state_shape = if batch_major {
        vec![batch, directions, hidden]
    } else {
        vec![directions, batch, hidden]
    };

    if ctx.num_outputs() > 0 {
        ctx.set_output(0, x.dtype, y_shape);
    }
    for output in 1..=state_outputs.min(ctx.num_outputs().saturating_sub(1)) {
        ctx.set_output(output, x.dtype, state_shape.clone());
    }
    Ok(())
}

fn rnn_pre_14(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    recurrent(ctx, 1, false)
}

fn rnn_14(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    recurrent(ctx, 1, true)
}

fn gru_pre_14(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    recurrent(ctx, 1, false)
}

fn gru_14(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    recurrent(ctx, 1, true)
}

fn lstm_pre_14(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    recurrent(ctx, 2, false)
}

fn lstm_14(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    recurrent(ctx, 2, true)
}

/// Register recurrent neural-network rules at their layout schema boundary.
pub fn register(reg: &mut InferenceRegistry) {
    reg.register("", "RNN", 1, rnn_pre_14);
    reg.register("", "RNN", 14, rnn_14);
    reg.register("", "GRU", 1, gru_pre_14);
    reg.register("", "GRU", 14, gru_14);
    reg.register("", "LSTM", 1, lstm_pre_14);
    reg.register("", "LSTM", 14, lstm_14);
}

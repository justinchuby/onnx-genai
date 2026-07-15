//! Normalisation and reduction rules: `LayerNormalization`, `Softmax`,
//! `ReduceMean`/`ReduceSum` (and related reductions).

use crate::context::InferenceContext;
use crate::dim_expr::DimExpr;
use crate::error::ShapeInferError;
use crate::handlers::norm_axis;
use crate::registry::InferenceRegistry;

/// `LayerNormalization`: output 0 is the input shape; optional `Mean` and
/// `InvStdDev` outputs are the input shape with the normalised axes set to `1`.
pub fn layer_norm(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(x) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    ctx.set_output_type(0, x.clone());
    let rank = x.rank();
    let axis = ctx.node.attr("axis").and_then(|a| a.as_int()).unwrap_or(-1);
    let axis = norm_axis(axis, rank.max(1));
    let mut reduced = x.shape.clone();
    for d in reduced.iter_mut().skip(axis) {
        *d = DimExpr::constant(1);
    }
    if ctx.num_outputs() > 1 {
        ctx.set_output(1, x.dtype, reduced.clone());
    }
    if ctx.num_outputs() > 2 {
        ctx.set_output(2, x.dtype, reduced);
    }
    Ok(())
}

/// `SkipLayerNormalization` (com.microsoft): output 0 is the input shape;
/// optional `mean` (1) and `inv_std_var` (2) are the input shape with the
/// normalised last axis set to `1`; optional `input_skip_bias_sum` (3) is the
/// full input shape (`X + skip + bias`). Only the outputs the node requests are
/// emitted.
pub fn skip_layer_norm(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(x) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    ctx.set_output_type(0, x.clone());
    let rank = x.rank();
    let axis = norm_axis(-1, rank.max(1));
    let mut reduced = x.shape.clone();
    for d in reduced.iter_mut().skip(axis) {
        *d = DimExpr::constant(1);
    }
    if ctx.num_outputs() > 1 {
        ctx.set_output(1, x.dtype, reduced.clone());
    }
    if ctx.num_outputs() > 2 {
        ctx.set_output(2, x.dtype, reduced);
    }
    if ctx.num_outputs() > 3 {
        ctx.set_output_type(3, x);
    }
    Ok(())
}

/// `RMSNormalization`: output 0 is the input shape (single output, no optional
/// mean/inv-std). Shape- and dtype-preserving like other norm ops.
pub fn rms_norm(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    if let Some(x) = ctx.input_type(0).cloned() {
        ctx.set_output_type(0, x);
    }
    Ok(())
}

/// `RotaryEmbedding`: output shape and dtype equal the input `X` (input 0).
pub fn rotary_embedding(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    if let Some(x) = ctx.input_type(0).cloned() {
        ctx.set_output_type(0, x);
    }
    Ok(())
}

/// `Softmax`/`LogSoftmax`: shape- and dtype-preserving.
pub fn softmax(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    if let Some(t) = ctx.input_type(0).cloned() {
        ctx.set_output_type(0, t);
    }
    Ok(())
}

/// `ReduceMean`/`ReduceSum`/… — reduce over `axes` (from the attribute pre-opset
/// 18, or input 1 as shape-data from opset 18), honouring `keepdims`.
pub fn reduce(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(x) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    let rank = x.rank();
    let keepdims = ctx
        .node
        .attr("keepdims")
        .and_then(|a| a.as_int())
        .unwrap_or(1)
        != 0;
    let noop_empty = ctx
        .node
        .attr("noop_with_empty_axes")
        .and_then(|a| a.as_int())
        .unwrap_or(0)
        != 0;

    // Axes: attribute first (opset ≤17), then a shape-data input (opset-18).
    let axes_attr: Option<Vec<i64>> = ctx
        .node
        .attr("axes")
        .and_then(|a| a.as_ints())
        .map(<[i64]>::to_vec);
    let axes_raw: Option<Vec<i64>> = axes_attr.clone().or_else(|| {
        ctx.input_shape_data(1).and_then(|sd| {
            sd.elems
                .iter()
                .map(|e| e.as_const())
                .collect::<Option<Vec<i64>>>()
        })
    });

    // Distinguish a genuinely-absent/empty axes list from an axes *input* that
    // is present but unresolved (missing or symbolic shape-data). In the latter
    // case we cannot know which axes are reduced — nor, under
    // `noop_with_empty_axes`, whether this node is a no-op — so leaving the
    // output unresolved is more honest than fabricating a reduce-all shape.
    if axes_raw.is_none() && axes_attr.is_none() && ctx.has_input(1) {
        return Ok(());
    }

    let axes: Vec<usize> = match axes_raw {
        Some(a) if !a.is_empty() => a.into_iter().map(|ax| norm_axis(ax, rank.max(1))).collect(),
        Some(_) if noop_empty => {
            // Explicitly empty axes with noop flag: identity.
            ctx.set_output_type(0, x);
            return Ok(());
        }
        // No axes given (or empty without noop): reduce all axes.
        _ => (0..rank).collect(),
    };

    let mut out = Vec::with_capacity(rank);
    for (i, d) in x.shape.iter().enumerate() {
        if axes.contains(&i) {
            if keepdims {
                out.push(DimExpr::constant(1));
            }
        } else {
            out.push(d.clone());
        }
    }
    ctx.set_output(0, x.dtype, out);
    Ok(())
}

/// Register the normalisation/reduction family.
pub fn register(reg: &mut InferenceRegistry) {
    reg.register("", "LayerNormalization", 1, layer_norm);
    // The optimizer emits fused `LayerNormalization` in the `com.microsoft`
    // contrib domain; register the same rule there so the fused op's shape
    // still resolves (identical output-shape semantics as the standard op).
    reg.register("com.microsoft", "LayerNormalization", 1, layer_norm);
    reg.register("com.microsoft", "SkipLayerNormalization", 1, skip_layer_norm);
    reg.register(
        "com.microsoft",
        "SimplifiedLayerNormalization",
        1,
        layer_norm,
    );
    reg.register("", "Softmax", 1, softmax);
    reg.register("", "LogSoftmax", 1, softmax);
    // Standard LLM/transformer norm primitives (ai.onnx): both are
    // shape-preserving (output == input X).
    reg.register("", "RMSNormalization", 23, rms_norm);
    reg.register("", "RotaryEmbedding", 23, rotary_embedding);

    for op in [
        "ReduceMean",
        "ReduceSum",
        "ReduceMax",
        "ReduceMin",
        "ReduceProd",
        "ReduceL2",
        "ReduceSumSquare",
    ] {
        reg.register("", op, 1, reduce);
    }
}

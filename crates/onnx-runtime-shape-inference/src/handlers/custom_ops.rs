//! Shape rules for runtime custom operators.

use onnx_runtime_ir::{Attribute, DataType};

use crate::context::InferenceContext;
use crate::dim_expr::DimExpr;
use crate::error::ShapeInferError;
use crate::registry::InferenceRegistry;

/// `MoE`/`QMoE`: the single output preserves the activation tensor.
pub fn moe(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    require_outputs(ctx, 1)?;
    if let Some(input) = ctx.input_type(0).cloned() {
        ctx.set_output_type(0, input);
    }
    Ok(())
}

/// `SparseKvGather`: cache `[B,G,C,D]` and indices `[B,G,Q,K]` produce
/// selected KV `[B,G,Q,K,D]`.
pub fn sparse_kv_gather(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    require_outputs(ctx, 1)?;
    let Some(cache) = rank_shape(ctx, 0, 4)? else {
        return Ok(());
    };
    let Some(indices) = rank_shape(ctx, 1, 4)? else {
        return Ok(());
    };
    let dtype = ctx.input_dtype(0).unwrap_or(DataType::Float32);
    ctx.set_output(
        0,
        dtype,
        vec![
            cache[0].clone(),
            cache[1].clone(),
            indices[2].clone(),
            indices[3].clone(),
            cache[3].clone(),
        ],
    );
    Ok(())
}

/// Frozen stateful `CompressedSparseAttention` v1 output shapes.
pub fn compressed_sparse_attention(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(query) = rank_shape(ctx, 0, 4)? else {
        return Ok(());
    };
    let ratio = positive_attr(ctx, "compression_ratio")?;
    if !matches!(ratio, 4 | 128) {
        return Err(ShapeInferError::Invalid {
            op: ctx.op().into(),
            detail: format!("compression_ratio must be 4 or 128, got {ratio}"),
        });
    }
    let _num_heads = positive_attr(ctx, "num_heads")?;
    let head_dim = positive_attr(ctx, "head_dim")?;
    let rope_dim = nonnegative_attr(ctx, "qk_rope_head_dim", 0)?;
    let cache_format = ctx
        .node
        .attr("cache_format")
        .map(|attr| {
            attr.as_str().ok_or_else(|| ShapeInferError::Invalid {
                op: ctx.op().into(),
                detail: "cache_format must be a string".into(),
            })
        })
        .transpose()?
        .unwrap_or("f32");
    let (cache_dtype, stored_width) = stored_width(ctx, cache_format, head_dim, rope_dim)?;
    let records = ctx
        .input_shape_data(9)
        .filter(|data| data.is_scalar())
        .and_then(|data| data.elems.first())
        .and_then(DimExpr::as_const)
        .filter(|&total| total >= 0)
        .map(|total| DimExpr::constant(total / ratio))
        .unwrap_or_else(|| ctx.fresh_dim());

    ctx.set_output(0, DataType::Float32, query.clone());
    ctx.set_output(
        1,
        cache_dtype,
        vec![query[0].clone(), records.clone(), c(stored_width)],
    );

    if ratio == 128 {
        if ctx.num_outputs() != 3 {
            return Err(ShapeInferError::Arity {
                op: ctx.op().into(),
                expected: "exactly 3 outputs for compression_ratio=128".into(),
                found: ctx.num_outputs(),
            });
        }
        ctx.set_output(
            2,
            DataType::Float32,
            vec![query[0].clone(), c(ratio), c(2), c(head_dim)],
        );
        return Ok(());
    }

    if !(5..=6).contains(&ctx.num_outputs()) {
        return Err(ShapeInferError::Arity {
            op: ctx.op().into(),
            expected: "5 or 6 outputs for compression_ratio=4".into(),
            found: ctx.num_outputs(),
        });
    }
    if cache_format != "fp8_e4m3_block64" {
        return Err(ShapeInferError::Invalid {
            op: ctx.op().into(),
            detail: "compression_ratio=4 requires cache_format='fp8_e4m3_block64'".into(),
        });
    }
    let index_heads = positive_attr(ctx, "index_num_heads")?;
    let index_dim = positive_attr(ctx, "index_head_dim")?;
    let index_topk = positive_attr(ctx, "index_topk")?;
    let index_width = fp4_width(ctx, index_dim)?;
    let compressor_width = doubled(ctx, "head_dim", head_dim)?;
    let index_compressor_width = doubled(ctx, "index_head_dim", index_dim)?;
    ctx.set_output(
        2,
        DataType::Float32,
        vec![query[0].clone(), c(8), c(2), c(compressor_width)],
    );
    ctx.set_output(
        3,
        DataType::Uint8,
        vec![query[0].clone(), records.clone(), c(index_width)],
    );
    ctx.set_output(
        4,
        DataType::Float32,
        vec![query[0].clone(), c(8), c(2), c(index_compressor_width)],
    );
    if ctx.num_outputs() == 6 {
        let selections = records
            .as_const()
            .map(|count| DimExpr::constant(count.min(index_topk)))
            .unwrap_or_else(|| ctx.fresh_dim());
        ctx.set_output(
            5,
            DataType::Int32,
            vec![
                query[0].clone(),
                c(index_heads),
                query[1].clone(),
                selections,
            ],
        );
    }
    Ok(())
}

fn rank_shape(
    ctx: &InferenceContext,
    index: usize,
    expected: usize,
) -> Result<Option<Vec<DimExpr>>, ShapeInferError> {
    let Some(shape) = ctx.input_shape(index) else {
        return Ok(None);
    };
    if shape.len() != expected {
        return Err(ShapeInferError::InvalidRank {
            op: ctx.op().into(),
            index,
            rank: shape.len(),
            detail: format!("expected rank {expected}"),
        });
    }
    Ok(Some(shape.to_vec()))
}

fn require_outputs(ctx: &InferenceContext, expected: usize) -> Result<(), ShapeInferError> {
    if ctx.num_outputs() != expected {
        return Err(ShapeInferError::Arity {
            op: ctx.op().into(),
            expected: format!("exactly {expected} output(s)"),
            found: ctx.num_outputs(),
        });
    }
    Ok(())
}

fn int_attr(ctx: &InferenceContext, name: &str, default: i64) -> Result<i64, ShapeInferError> {
    match ctx.node.attr(name) {
        Some(attr) => attr.as_int().ok_or_else(|| ShapeInferError::Invalid {
            op: ctx.op().into(),
            detail: format!("{name} must be an integer"),
        }),
        None => Ok(default),
    }
}

fn positive_attr(ctx: &InferenceContext, name: &str) -> Result<i64, ShapeInferError> {
    let value = ctx
        .node
        .attr(name)
        .and_then(Attribute::as_int)
        .ok_or_else(|| ShapeInferError::MissingAttribute {
            op: ctx.op().into(),
            attr: name.into(),
        })?;
    if value <= 0 {
        return Err(ShapeInferError::Invalid {
            op: ctx.op().into(),
            detail: format!("{name} must be positive, got {value}"),
        });
    }
    Ok(value)
}

fn nonnegative_attr(
    ctx: &InferenceContext,
    name: &str,
    default: i64,
) -> Result<i64, ShapeInferError> {
    let value = int_attr(ctx, name, default)?;
    if value < 0 {
        return Err(ShapeInferError::Invalid {
            op: ctx.op().into(),
            detail: format!("{name} must be non-negative, got {value}"),
        });
    }
    Ok(value)
}

fn stored_width(
    ctx: &InferenceContext,
    format: &str,
    head_dim: i64,
    rope_dim: i64,
) -> Result<(DataType, i64), ShapeInferError> {
    match format {
        "f32" => Ok((DataType::Float32, head_dim)),
        "fp8_e4m3_block64" => {
            let non_rope = head_dim.checked_sub(rope_dim).filter(|dim| dim % 64 == 0);
            let Some(non_rope) = non_rope else {
                return Err(ShapeInferError::Invalid {
                    op: ctx.op().into(),
                    detail: "non-RoPE head width must be non-negative and divisible by 64".into(),
                });
            };
            let width = (non_rope / 64)
                .checked_mul(65)
                .and_then(|width| rope_dim.checked_mul(2)?.checked_add(width))
                .ok_or_else(|| ShapeInferError::Invalid {
                    op: ctx.op().into(),
                    detail: "FP8 cache width overflow".into(),
                })?;
            Ok((DataType::Uint8, width))
        }
        "fp4_e2m1_block32" => Ok((DataType::Uint8, fp4_width(ctx, head_dim)?)),
        other => Err(ShapeInferError::Invalid {
            op: ctx.op().into(),
            detail: format!("unsupported cache_format '{other}'"),
        }),
    }
}

fn fp4_width(ctx: &InferenceContext, logical_width: i64) -> Result<i64, ShapeInferError> {
    if logical_width % 32 != 0 {
        return Err(ShapeInferError::Invalid {
            op: ctx.op().into(),
            detail: format!("FP4 logical width {logical_width} must be divisible by 32"),
        });
    }
    logical_width
        .checked_div(32)
        .and_then(|blocks| blocks.checked_mul(17))
        .ok_or_else(|| ShapeInferError::Invalid {
            op: ctx.op().into(),
            detail: "FP4 cache width overflow".into(),
        })
}

fn doubled(ctx: &InferenceContext, name: &str, value: i64) -> Result<i64, ShapeInferError> {
    value
        .checked_mul(2)
        .ok_or_else(|| ShapeInferError::Invalid {
            op: ctx.op().into(),
            detail: format!("{name} doubled width overflow"),
        })
}

fn c(value: i64) -> DimExpr {
    DimExpr::constant(value)
}

/// `com.microsoft::CausalConvWithState`: output 0 preserves the `[B, C, L]`
/// input activation; output 1 (present conv state) preserves the past-state
/// `[B, C, K-1]` cache shape supplied as input 3.
pub fn causal_conv_with_state(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    if let Some(input) = ctx.input_type(0).cloned() {
        ctx.set_output_type(0, input);
    }
    if ctx.num_outputs() >= 2
        && ctx.has_input(3)
        && let Some(state) = ctx.input_type(3).cloned()
    {
        ctx.set_output_type(1, state);
    }
    Ok(())
}

/// `com.microsoft::LinearAttention`: output 0 is `[B, T, max(H_q, H_kv)·d_v]`
/// where `d_v = value_hidden / H_kv`; output 1 (present recurrent state)
/// preserves the past-state `[B, H_kv, d_k, d_v]` cache supplied as input 3.
pub fn linear_attention(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let q_num_heads = int_attr(ctx, "q_num_heads", 0)?;
    let kv_num_heads = int_attr(ctx, "kv_num_heads", 0)?;
    if let (Some(query), Some(query_shape), Some(value_shape)) = (
        ctx.input_type(0).map(|t| t.dtype),
        ctx.input_shape(0).map(<[_]>::to_vec),
        ctx.input_shape(2).map(<[_]>::to_vec),
    ) && query_shape.len() == 3
        && value_shape.len() == 3
        && kv_num_heads > 0
        && q_num_heads > 0
    {
        let out_heads = q_num_heads.max(kv_num_heads);
        let d_v = value_shape[2]
            .checked_div(&c(kv_num_heads))
            .unwrap_or_else(DimExpr::overflow);
        let out_hidden = d_v.mul(&c(out_heads));
        ctx.set_output(
            0,
            query,
            vec![query_shape[0].clone(), query_shape[1].clone(), out_hidden],
        );
    }
    if ctx.num_outputs() >= 2
        && ctx.has_input(3)
        && let Some(state) = ctx.input_type(3).cloned()
    {
        ctx.set_output_type(1, state);
    }
    Ok(())
}

/// `com.microsoft::RotaryEmbedding`: the rotation is shape-preserving, so the
/// output mirrors the `X` activation (input 0).
pub fn rotary_embedding(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    if let Some(input) = ctx.input_type(0).cloned() {
        ctx.set_output_type(0, input);
    }
    Ok(())
}

/// `com.microsoft::GatherBlockQuantized`: output preserves the scales dtype and
/// has shape `indices.shape ++ data.shape[1:]` (for `gather_axis == 0`), with
/// the last axis scaled by `8 / bits` when several elements are packed per byte.
pub fn gather_block_quantized(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let gather_axis = int_attr(ctx, "gather_axis", 0)?;
    let bits = int_attr(ctx, "bits", 4)?;
    if gather_axis != 0 || bits <= 0 {
        return Ok(());
    }
    let components = 8 / bits;
    if let (Some(dtype), Some(data_shape), Some(indices_shape)) = (
        ctx.input_type(2).map(|t| t.dtype),
        ctx.input_shape(0).map(<[_]>::to_vec),
        ctx.input_shape(1).map(<[_]>::to_vec),
    ) && data_shape.len() >= 2
    {
        let last = data_shape.len() - 1;
        let mut dims = indices_shape;
        for (axis, dim) in data_shape.iter().enumerate().skip(1) {
            if axis == last && components > 1 {
                dims.push(dim.mul(&c(components)));
            } else {
                dims.push(dim.clone());
            }
        }
        ctx.set_output(0, dtype, dims);
    }
    Ok(())
}

/// Register custom runtime and ORT contrib operators.
pub fn register(reg: &mut InferenceRegistry) {
    reg.register("com.microsoft", "MoE", 1, moe);
    reg.register("com.microsoft", "QMoE", 1, moe);
    reg.register(
        "com.microsoft",
        "CausalConvWithState",
        1,
        causal_conv_with_state,
    );
    reg.register("com.microsoft", "LinearAttention", 1, linear_attention);
    reg.register(
        "com.microsoft",
        "GatherBlockQuantized",
        1,
        gather_block_quantized,
    );
    reg.register("com.microsoft", "RotaryEmbedding", 1, rotary_embedding);
    reg.register("pkg.nxrt", "BlockQuantizedMoE", 1, moe);
    reg.register("pkg.nxrt", "SparseKvGather", 1, sparse_kv_gather);
    reg.register(
        "pkg.nxrt",
        "CompressedSparseAttention",
        1,
        compressed_sparse_attention,
    );
    reg.register(
        "com.microsoft",
        "CompressedSparseAttention",
        1,
        compressed_sparse_attention,
    );
}

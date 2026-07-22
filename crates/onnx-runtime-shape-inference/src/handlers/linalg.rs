//! Linear-algebra rules: `MatMul` (NumPy semantics) and `Gemm`.

use onnx_runtime_ir::Attribute;

use crate::context::InferenceContext;
use crate::dim_expr::DimExpr;
use crate::error::ShapeInferError;
use crate::registry::InferenceRegistry;

/// `MatMul` with full NumPy broadcasting semantics.
pub fn matmul(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let a = ctx.input_shape(0).map(<[DimExpr]>::to_vec);
    let b = ctx.input_shape(1).map(<[DimExpr]>::to_vec);
    let dtype = ctx.input_dtype(0);
    if let (Some(a), Some(b), Some(dtype)) = (a, b, dtype) {
        let shape = matmul_shape(ctx, &a, &b, "MatMul")?;
        ctx.set_output(0, dtype, shape);
    }
    Ok(())
}

/// Quantized matmul with packed weights and output width supplied by `N`.
pub fn quantized_matmul(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(mut shape) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    let Some(dtype) = ctx.input_dtype(0) else {
        return Ok(());
    };
    let n = ctx
        .node
        .attr("N")
        .and_then(Attribute::as_int)
        .ok_or_else(|| ShapeInferError::MissingAttribute {
            op: ctx.node.op_type.clone(),
            attr: "N".into(),
        })?;
    let Some(last) = shape.last_mut() else {
        return Err(ShapeInferError::InvalidRank {
            op: ctx.node.op_type.clone(),
            index: 0,
            rank: 0,
            detail: "activation input must have rank ≥ 1".into(),
        });
    };
    *last = DimExpr::constant(n);
    ctx.set_output(0, dtype, shape);
    Ok(())
}

/// `com.microsoft.FusedMatMul`: MatMul with optional pre-transposition of the
/// last two dims (`transA`/`transB`) and batch-axis relocation
/// (`transBatchA`/`transBatchB`), plus a shape-neutral `alpha` scale.
///
/// ORT emits this op throughout optimized transformer graphs (MatMul+Transpose
/// fusion in attention QKᵀ and post-transpose FC layers). We reorder each
/// operand's dims into the plain `[batch…, row, col]` layout per the contrib
/// spec, then reuse [`matmul_shape`] so all the 1-D promotion, batch broadcast,
/// and contraction-mismatch checking that plain MatMul does still applies.
/// `alpha` only scales values, so it has no shape effect and is ignored.
pub fn fused_matmul(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let a = ctx.input_shape(0).map(<[DimExpr]>::to_vec);
    let b = ctx.input_shape(1).map(<[DimExpr]>::to_vec);
    let dtype = ctx.input_dtype(0);
    let (Some(a), Some(b), Some(dtype)) = (a, b, dtype) else {
        return Ok(());
    };
    let flag = |name: &str| ctx.node.attr(name).and_then(Attribute::as_int).unwrap_or(0) != 0;
    let a_eff = apply_fused_trans(&a, flag("transA"), flag("transBatchA"));
    let b_eff = apply_fused_trans(&b, flag("transB"), flag("transBatchB"));
    let shape = matmul_shape(ctx, &a_eff, &b_eff, "FusedMatMul")?;
    ctx.set_output(0, dtype, shape);
    Ok(())
}

/// Reorder a FusedMatMul operand's dims into the plain-MatMul `[batch…, row,
/// col]` layout given its `trans` / `trans_batch` flags.
///
/// Mirrors ORT's `FusedMatMulShapeInference` (contrib_defs.cc): `trans` swaps
/// the trailing row/col pair; `trans_batch` relocates the *leading* batch axis
/// into the row slot (moving the remaining leading dims `1..rank-1` into the
/// batch prefix). A rank-≤1 (vector) operand is returned unchanged — numpy
/// transpose of a vector is a no-op, matching ORT which forces the flags off.
fn apply_fused_trans(raw: &[DimExpr], trans: bool, trans_batch: bool) -> Vec<DimExpr> {
    let rank = raw.len();
    if rank < 2 {
        return raw.to_vec();
    }
    let mut out = Vec::with_capacity(rank);
    // Batch prefix: dims `1..rank-1` when relocating the leading batch axis,
    // else the leading `0..rank-2`.
    let (start, end) = if trans_batch {
        (1, rank - 1)
    } else {
        (0, rank - 2)
    };
    out.extend_from_slice(&raw[start..end]);
    // Row (M) then col (K) indices, per ORT's dim-selection expressions.
    let row_idx = if trans {
        rank - 1
    } else if trans_batch {
        0
    } else {
        rank - 2
    };
    let col_idx = if trans {
        if trans_batch { 0 } else { rank - 2 }
    } else {
        rank - 1
    };
    out.push(raw[row_idx].clone());
    out.push(raw[col_idx].clone());
    out
}

/// Compute the output shape of `A @ B` following `numpy.matmul`.
fn matmul_shape(
    ctx: &mut InferenceContext,
    a: &[DimExpr],
    b: &[DimExpr],
    op: &str,
) -> Result<Vec<DimExpr>, ShapeInferError> {
    if a.is_empty() || b.is_empty() {
        return Err(ShapeInferError::Invalid {
            op: op.into(),
            detail: "operands must have rank ≥ 1".into(),
        });
    }
    let a_is_1d = a.len() == 1;
    let b_is_1d = b.len() == 1;

    // Promote 1-D operands: A -> [1, K], B -> [K, 1].
    let a2: Vec<DimExpr> = if a_is_1d {
        vec![DimExpr::constant(1), a[0].clone()]
    } else {
        a.to_vec()
    };
    let b2: Vec<DimExpr> = if b_is_1d {
        vec![b[0].clone(), DimExpr::constant(1)]
    } else {
        b.to_vec()
    };

    let (a_batch, a_mk) = a2.split_at(a2.len() - 2);
    let (b_batch, b_kn) = b2.split_at(b2.len() - 2);
    let m = a_mk[0].clone();
    let ka = a_mk[1].clone();
    let kb = b_kn[0].clone();
    let n = b_kn[1].clone();

    // A concrete contraction-dim mismatch is a malformed graph, not just an
    // under-specified shape — report it regardless of policy.
    if let (Some(ka), Some(kb)) = (ka.as_const(), kb.as_const())
        && ka != kb
    {
        return Err(ShapeInferError::Invalid {
            op: op.into(),
            detail: format!("contraction mismatch: {ka} vs {kb}"),
        });
    }

    let mut shape = ctx.broadcast(a_batch, b_batch)?;
    if !a_is_1d {
        shape.push(m);
    }
    if !b_is_1d {
        shape.push(n);
    }
    Ok(shape)
}

/// `com.microsoft::FusedMatMulBias`: the optimizer's `MatMul(A, B) + bias`
/// fusion. The bias is numpy-broadcast onto the matmul result, so the output
/// shape is exactly `MatMul(A, B)`'s shape — the plain-MatMul rule applies.
pub fn fused_matmul_bias(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let a = ctx.input_shape(0).map(<[DimExpr]>::to_vec);
    let b = ctx.input_shape(1).map(<[DimExpr]>::to_vec);
    let dtype = ctx.input_dtype(0);
    if let (Some(a), Some(b), Some(dtype)) = (a, b, dtype) {
        let shape = matmul_shape(ctx, &a, &b, "FusedMatMulBias")?;
        ctx.set_output(0, dtype, shape);
    }
    Ok(())
}

/// `com.microsoft::FusedGemm`: the optimizer's `Relu(MatMul(A, B) + bias)`
/// fusion. The bias is numpy-broadcast onto the matmul result and `Relu` is
/// elementwise, so the output shape is exactly `MatMul(A, B)`'s shape — the
/// plain-MatMul rule applies.
pub fn fused_gemm(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let a = ctx.input_shape(0).map(<[DimExpr]>::to_vec);
    let b = ctx.input_shape(1).map(<[DimExpr]>::to_vec);
    let dtype = ctx.input_dtype(0);
    if let (Some(a), Some(b), Some(dtype)) = (a, b, dtype) {
        let shape = matmul_shape(ctx, &a, &b, "FusedGemm")?;
        ctx.set_output(0, dtype, shape);
    }
    Ok(())
}

/// `com.microsoft::FusedAttention`: the optimizer's SDPA-core fusion
/// (`MatMul(Q, Kᵀ) → scale → [+mask] → Softmax → MatMul(·, V)`). The output
/// shape is exactly that of the final `MatMul(probs, V)`: `Q`'s leading/batch
/// dims and `[seq_q, head_dim_v]`.
///
/// We reproduce the two-matmul shape flow symbolically so symbolic batch/seq
/// dims propagate and the contraction dims are checked. The `k_transposed`
/// attribute mirrors the kernel: when unset (`0`) the `K` input is
/// `[…, seq_k, head_dim]` and is transposed to form `Kᵀ`; when set (`1`) `K` is
/// already `[…, head_dim, seq_k]` and used as-is. `scale` only rescales values
/// (no shape effect) and the optional additive `mask` broadcasts into the
/// scores, so neither changes the output shape.
pub fn fused_attention(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let q = ctx.input_shape(0).map(<[DimExpr]>::to_vec);
    let k = ctx.input_shape(1).map(<[DimExpr]>::to_vec);
    let v = ctx.input_shape(2).map(<[DimExpr]>::to_vec);
    let dtype = ctx.input_dtype(0);
    let (Some(q), Some(k), Some(v), Some(dtype)) = (q, k, v, dtype) else {
        return Ok(());
    };
    let k_transposed = ctx
        .node
        .attr("k_transposed")
        .and_then(Attribute::as_int)
        .unwrap_or(0)
        != 0;
    // K^T: when K is not pre-transposed, swap its trailing two dims.
    let k_eff = if k_transposed {
        k
    } else if k.len() >= 2 {
        let mut kt = k.clone();
        let r = kt.len();
        kt.swap(r - 2, r - 1);
        kt
    } else {
        k
    };
    let scores = matmul_shape(ctx, &q, &k_eff, "FusedAttention")?;
    let shape = matmul_shape(ctx, &scores, &v, "FusedAttention")?;
    ctx.set_output(0, dtype, shape);
    Ok(())
}

/// Standard `ai.onnx::Attention` (opset 23–26): scaled dot-product attention
/// with 3D/4D inputs, GQA/MQA head sharing, a KV cache, and up to four outputs.
///
/// The executor sizes each value's buffer from resolved shapes, so every
/// produced output must be inferable:
/// * `Y` — matches Q's rank: 4D `(batch, q_heads, q_seq, v_head_size)`, or 3D
///   `(batch, q_seq, q_heads·v_head_size)`.
/// * `present_key` — `(batch, kv_heads, total_seq, head_size)`.
/// * `present_value` — `(batch, kv_heads, total_seq, v_head_size)`.
/// * `qk_matmul_output` — `(batch, q_heads, q_seq, total_seq)`.
///
/// `total_seq = past_seq + kv_seq` (past comes from the optional `past_key`
/// input 4). For 3D inputs the per-head sizes come from the `q_num_heads` /
/// `kv_num_heads` attributes (`head_size = hidden / num_heads`).
pub fn attention(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let q = ctx.input_shape(0).map(<[DimExpr]>::to_vec);
    let k = ctx.input_shape(1).map(<[DimExpr]>::to_vec);
    let v = ctx.input_shape(2).map(<[DimExpr]>::to_vec);
    let dtype = ctx.input_dtype(0);
    let (Some(q), Some(k), Some(v), Some(dtype)) = (q, k, v, dtype) else {
        return Ok(());
    };
    let q_rank = q.len();
    if !(q_rank == 3 || q_rank == 4) || k.len() != q_rank || v.len() != q_rank {
        return Err(ShapeInferError::Invalid {
            op: "Attention".into(),
            detail: format!(
                "Q, K, V must all be rank 3 or 4 (got Q={q_rank}, K={}, V={})",
                k.len(),
                v.len()
            ),
        });
    }

    let attr_dim = |name: &str| -> Option<DimExpr> {
        ctx.node
            .attr(name)
            .and_then(Attribute::as_int)
            .map(DimExpr::constant)
    };

    // Resolve batch, per-head dims, and head counts for both ranks.
    let batch = q[0].clone();
    let (q_heads, q_seq, head_size, kv_heads, kv_seq, v_head_size) = if q_rank == 4 {
        (
            q[1].clone(),
            q[2].clone(),
            q[3].clone(),
            k[1].clone(),
            k[2].clone(),
            v[3].clone(),
        )
    } else {
        // 3D: (batch, seq, hidden). Split hidden by the num_heads attributes.
        let q_heads = attr_dim("q_num_heads").ok_or_else(|| ShapeInferError::Invalid {
            op: "Attention".into(),
            detail: "3D inputs require the `q_num_heads` attribute".into(),
        })?;
        let kv_heads = attr_dim("kv_num_heads").ok_or_else(|| ShapeInferError::Invalid {
            op: "Attention".into(),
            detail: "3D inputs require the `kv_num_heads` attribute".into(),
        })?;
        let head_size = q[2]
            .checked_div(&q_heads)
            .unwrap_or_else(|| ctx.fresh_dim());
        let v_head_size = v[2]
            .checked_div(&kv_heads)
            .unwrap_or_else(|| ctx.fresh_dim());
        (
            q_heads,
            q[1].clone(),
            head_size,
            kv_heads,
            k[1].clone(),
            v_head_size,
        )
    };

    // total_seq = past_seq + kv_seq (past_key is input 4, when present).
    let total_seq = match ctx.input_shape(4) {
        Some(pk) if pk.len() == 4 => pk[2].add(&kv_seq),
        _ => kv_seq.clone(),
    };

    // Y: 4D (batch, q_heads, q_seq, v_head_size) or 3D (batch, q_seq, hidden).
    let y_shape = if q_rank == 4 {
        vec![
            batch.clone(),
            q_heads.clone(),
            q_seq.clone(),
            v_head_size.clone(),
        ]
    } else {
        vec![batch.clone(), q_seq.clone(), q_heads.mul(&v_head_size)]
    };
    ctx.set_output(0, dtype, y_shape);

    // present_key / present_value (4D), when those outputs exist.
    if ctx.num_outputs() > 1 {
        ctx.set_output(
            1,
            dtype,
            vec![
                batch.clone(),
                kv_heads.clone(),
                total_seq.clone(),
                head_size.clone(),
            ],
        );
    }
    if ctx.num_outputs() > 2 {
        let v_dtype = ctx.input_dtype(2).unwrap_or(dtype);
        ctx.set_output(
            2,
            v_dtype,
            vec![
                batch.clone(),
                kv_heads.clone(),
                total_seq.clone(),
                v_head_size.clone(),
            ],
        );
    }
    // qk_matmul_output: (batch, q_heads, q_seq, total_seq).
    if ctx.num_outputs() > 3 {
        ctx.set_output(3, dtype, vec![batch, q_heads, q_seq, total_seq]);
    }
    Ok(())
}

/// `Gemm(A, B, C?)`: `Y = alpha * A' * B' + beta * C`, output `[M, N]`.
pub fn gemm(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let a = ctx.input_shape(0).map(<[DimExpr]>::to_vec);
    let b = ctx.input_shape(1).map(<[DimExpr]>::to_vec);
    let dtype = ctx.input_dtype(0);
    let (Some(a), Some(b), Some(dtype)) = (a, b, dtype) else {
        return Ok(());
    };
    if a.len() != 2 || b.len() != 2 {
        return Err(ShapeInferError::InvalidRank {
            op: "Gemm".into(),
            index: if a.len() != 2 { 0 } else { 1 },
            rank: if a.len() != 2 { a.len() } else { b.len() },
            detail: "Gemm operands must be rank-2".into(),
        });
    }
    let trans_a = ctx
        .node
        .attr("transA")
        .and_then(|x| x.as_int())
        .unwrap_or(0)
        != 0;
    let trans_b = ctx
        .node
        .attr("transB")
        .and_then(|x| x.as_int())
        .unwrap_or(0)
        != 0;
    let m = if trans_a { a[1].clone() } else { a[0].clone() };
    let n = if trans_b { b[0].clone() } else { b[1].clone() };
    ctx.set_output(0, dtype, vec![m, n]);
    Ok(())
}

/// Register the linear-algebra family.
pub fn register(reg: &mut InferenceRegistry) {
    reg.register("", "MatMul", 1, matmul);
    reg.register("", "Gemm", 1, gemm);
    reg.register("pkg.nxrt", "BlockQuantizedMatMul", 1, quantized_matmul);
    reg.register("com.microsoft", "MatMulNBits", 1, quantized_matmul);
    // com.microsoft fused matmul honors transA/transB/transBatch attributes.
    reg.register("com.microsoft", "FusedMatMul", 1, fused_matmul);
    // The optimizer's MatMul+Add(bias) fusion: output shape == MatMul's.
    reg.register("com.microsoft", "FusedMatMulBias", 1, fused_matmul_bias);
    // The optimizer's MatMul+Add(bias)+Relu fusion: output shape == MatMul's
    // (bias broadcasts, Relu is elementwise).
    reg.register("com.microsoft", "FusedGemm", 1, fused_gemm);
    // The optimizer's SDPA-core fusion: output shape == MatMul(probs, V)'s.
    reg.register("com.microsoft", "FusedAttention", 1, fused_attention);
    // Standard ai.onnx::Attention. Added at opset 23; the shape contract is
    // unchanged through opset 26 (the opset-24 revision only adds the
    // `nonpad_kv_seqlen` external-cache input, which never concatenates a past
    // cache — total_seq stays kv_seq). The registry resolves the highest
    // `min_opset <= version`, so this single rule serves opsets 23–26.
    reg.register("", "Attention", 23, attention);
}

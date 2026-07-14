//! Linear-algebra rules: `MatMul` (NumPy semantics) and `Gemm`.

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
        let shape = matmul_shape(ctx, &a, &b)?;
        ctx.set_output(0, dtype, shape);
    }
    Ok(())
}

/// Compute the output shape of `A @ B` following `numpy.matmul`.
fn matmul_shape(
    ctx: &mut InferenceContext,
    a: &[DimExpr],
    b: &[DimExpr],
) -> Result<Vec<DimExpr>, ShapeInferError> {
    if a.is_empty() || b.is_empty() {
        return Err(ShapeInferError::Invalid {
            op: "MatMul".into(),
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
            op: "MatMul".into(),
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
    // com.microsoft fused matmul variants share MatMul's output shape.
    reg.register("com.microsoft", "FusedMatMul", 1, matmul);
}

use crate::{DimExpr, InferenceContext, InferenceRegistry, ShapeInferError};
use onnx_runtime_ir::{
    DataType,
    packed_routed::{OPS, Op},
};

fn infer(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let name = ctx.op().to_owned();
    let invalid = |detail| ShapeInferError::Invalid {
        op: name.clone(),
        detail,
    };
    let op = Op::from_node(ctx.node).map_err(&invalid)?;
    let dtypes = (0..ctx.node.inputs.len())
        .map(|i| ctx.input_dtype(i).unwrap_or(DataType::Undefined))
        .collect::<Vec<_>>();
    op.validate_dtypes(&dtypes).map_err(&invalid)?;
    let equal = |a: &DimExpr, b: &DimExpr, label: &str| {
        if let (Some(a), Some(b)) = (a.as_const(), b.as_const())
            && a != b
        {
            return Err(invalid(format!(
                "{label} dimensions must match, got {a} and {b}"
            )));
        }
        Ok(())
    };
    let check_rank = |i, rank| {
        if let Some(actual) = ctx.input_rank(i)
            && actual != rank
        {
            return Err(invalid(format!(
                "input {i} must have rank {rank}, got {actual}"
            )));
        }
        Ok(())
    };
    match op {
        Op::Pack(packing, dtype) => {
            if let Some(mut out) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) {
                let i = packing.axis(out.len()).map_err(&invalid)?;
                out[i] = if let Some(n) = out[i].as_const() {
                    let n = usize::try_from(n)
                        .map_err(|_| invalid(format!("invalid input extent {n}")))?;
                    let words = packing.words(n, dtype.byte_size() * 8).map_err(&invalid)?;
                    DimExpr::constant(
                        i64::try_from(words)
                            .map_err(|_| invalid("packed extent exceeds INT64_MAX".into()))?,
                    )
                } else {
                    ctx.fresh_dim()
                };
                ctx.set_output(0, dtype, out);
            }
        }
        Op::Unpack(packing) => {
            check_rank(1, 0)?;
            if let Some(mut out) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) {
                let i = packing.axis(out.len()).map_err(&invalid)?;
                let count = ctx
                    .input_shape_data(1)
                    .filter(|d| d.is_scalar())
                    .and_then(|d| d.elems.first())
                    .cloned();
                if let Some(n) = count.as_ref().and_then(DimExpr::as_const) {
                    let n = usize::try_from(n).map_err(|_| {
                        invalid(format!("count must be a nonnegative axis extent, got {n}"))
                    })?;
                    if let Some(dtype) = ctx.input_dtype(0).filter(|dt| *dt != DataType::Undefined)
                        && let Some(words) = out[i].as_const()
                    {
                        let expected = packing.words(n, dtype.byte_size() * 8).map_err(&invalid)?;
                        if usize::try_from(words) != Ok(expected) {
                            return Err(invalid(format!(
                                "packed axis {i} must contain exactly {expected} words for {n} codes, got {words}"
                            )));
                        }
                    }
                }
                out[i] = count.unwrap_or_else(|| ctx.fresh_dim());
                ctx.set_output(0, DataType::Uint8, out);
            }
        }
        Op::GroupedMatMul { transposed } => {
            for (i, rank) in [(0, 2), (1, 3), (2, 2), (3, 2)] {
                check_rank(i, rank)?;
            }
            if let (Some([m, k]), Some([g, w1, w2]), Some([rows, selections])) =
                (ctx.input_shape(0), ctx.input_shape(1), ctx.input_shape(2))
            {
                let (wk, n) = if transposed { (w2, w1) } else { (w1, w2) };
                equal(m, rows, "token")?;
                equal(k, wk, "contraction")?;
                if let Some([bg, bn]) = ctx.input_shape(3) {
                    equal(g, bg, "bias group")?;
                    equal(n, bn, "bias output width")?;
                }
                let output = vec![m.clone(), selections.clone(), n.clone()];
                if let Some(dtype) = ctx.input_dtype(0).or_else(|| ctx.input_dtype(1)) {
                    ctx.set_output(0, dtype, output);
                }
            }
        }
    }
    Ok(())
}

pub fn register(reg: &mut InferenceRegistry) {
    for name in OPS {
        reg.register("", name, 29, infer);
    }
    reg.register(
        onnx_runtime_ir::RUNTIME_DOMAIN,
        onnx_runtime_ir::quantized_grouped::OP,
        1,
        |ctx| {
            if let (Some([m, _]), Some([_, n, _]), Some([_, selections])) =
                (ctx.input_shape(0), ctx.input_shape(1), ctx.input_shape(2))
            {
                ctx.set_output(
                    0,
                    DataType::Float32,
                    vec![m.clone(), selections.clone(), n.clone()],
                );
            }
            Ok(())
        },
    );
    reg.register(
        onnx_runtime_ir::RUNTIME_DOMAIN,
        onnx_runtime_ir::quantized_grouped::PACKED_OP,
        1,
        |ctx| {
            let metadata = onnx_runtime_ir::dequantize_packed::Metadata::from_node(ctx.node)
                .map_err(|detail| ShapeInferError::Invalid {
                    op: ctx.node.op_type.clone(),
                    detail,
                })?;
            if let ([_, n, _], Some([m, _]), Some([_, selections])) =
                (metadata.shape(), ctx.input_shape(0), ctx.input_shape(2))
            {
                ctx.set_output(
                    0,
                    metadata.dtype(),
                    vec![m.clone(), selections.clone(), DimExpr::constant(*n as i64)],
                );
            }
            Ok(())
        },
    );
}

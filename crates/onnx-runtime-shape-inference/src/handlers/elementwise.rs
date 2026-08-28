//! Elementwise rules: unary activations, broadcasting binary/variadic ops, and
//! `Where`. Binary/variadic integer ops also propagate shape-data so arithmetic
//! *on* shape vectors (e.g. `Concat`-of-dims + 1) resolves symbolically.

use crate::context::InferenceContext;
use crate::dim_expr::DimExpr;
use crate::error::ShapeInferError;
use crate::registry::InferenceRegistry;
use crate::shape_data::ShapeData;

/// Shape- and dtype-preserving unary op (`Relu`, `Gelu`, `Erf`, `Tanh`,
/// `Sigmoid`, `Sqrt`, …).
pub fn unary(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    if let Some(t) = ctx.input_type(0).cloned() {
        ctx.set_output_type(0, t);
    }
    Ok(())
}

/// Broadcasting binary op (`Add`, `Sub`, `Mul`, `Div`, `Pow`, …).
pub fn binary(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let a = ctx.input_shape(0).map(<[DimExpr]>::to_vec);
    let b = ctx.input_shape(1).map(<[DimExpr]>::to_vec);
    let dtype = ctx.input_dtype(0).or_else(|| ctx.input_dtype(1));
    if let (Some(a), Some(b), Some(dtype)) = (a, b, dtype) {
        let shape = ctx.broadcast(&a, &b)?;
        ctx.set_output(0, dtype, shape);
    }
    // Shape-data: arithmetic on small integer shape vectors.
    if let Some(sd) = binary_shape_data(ctx.op(), ctx.input_shape_data(0), ctx.input_shape_data(1))
    {
        ctx.set_output_shape_data(0, sd);
    }
    Ok(())
}

/// Shape-preserving logical negation (`Not`): output is always boolean.
pub fn logical_not(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    if let Some(shape) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) {
        ctx.set_output(0, onnx_runtime_ir::DataType::Bool, shape);
    }
    Ok(())
}

/// Shape-preserving unary predicate (`IsInf`, `IsNaN`): output is boolean.
pub fn unary_predicate(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    if let Some(shape) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) {
        ctx.set_output(0, onnx_runtime_ir::DataType::Bool, shape);
    }
    Ok(())
}

/// Broadcasting comparison or binary logical op: output is always boolean.
pub fn boolean_binary(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let a = ctx.input_shape(0).map(<[DimExpr]>::to_vec);
    let b = ctx.input_shape(1).map(<[DimExpr]>::to_vec);
    if let (Some(a), Some(b)) = (a, b) {
        let shape = ctx.broadcast(&a, &b)?;
        ctx.set_output(0, onnx_runtime_ir::DataType::Bool, shape);
    }
    Ok(())
}

/// Variadic broadcasting op (`Min`, `Max`, `Sum`, `Mean`).
pub fn variadic(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let n = ctx.num_inputs();
    let mut acc: Option<Vec<DimExpr>> = None;
    let mut dtype = None;
    for i in 0..n {
        if let Some(s) = ctx.input_shape(i).map(<[DimExpr]>::to_vec) {
            dtype = dtype.or_else(|| ctx.input_dtype(i));
            acc = Some(match acc {
                None => s,
                Some(prev) => ctx.broadcast(&prev, &s)?,
            });
        }
    }
    if let (Some(shape), Some(dtype)) = (acc, dtype) {
        ctx.set_output(0, dtype, shape);
    }
    // Shape-data for two-operand Min/Max on shape vectors.
    if n == 2
        && let Some(sd) =
            binary_shape_data(ctx.op(), ctx.input_shape_data(0), ctx.input_shape_data(1))
    {
        ctx.set_output_shape_data(0, sd);
    }
    Ok(())
}

/// `Where(cond, x, y)`: broadcast of all three; dtype from the branches.
pub fn where_op(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let cond = ctx.input_shape(0).map(<[DimExpr]>::to_vec);
    let x = ctx.input_shape(1).map(<[DimExpr]>::to_vec);
    let y = ctx.input_shape(2).map(<[DimExpr]>::to_vec);
    let dtype = ctx.input_dtype(1).or_else(|| ctx.input_dtype(2));
    if let (Some(cond), Some(x), Some(y), Some(dtype)) = (cond, x, y, dtype) {
        let xy = ctx.broadcast(&x, &y)?;
        let shape = ctx.broadcast(&cond, &xy)?;
        ctx.set_output(0, dtype, shape);
    }
    Ok(())
}

/// Elementwise arithmetic on two shape-data operands, with scalar broadcasting.
/// Returns `None` unless both operands are present, integer, and the op yields
/// an exactly-representable result for every element.
fn binary_shape_data(op: &str, a: Option<&ShapeData>, b: Option<&ShapeData>) -> Option<ShapeData> {
    let (a, b) = (a?, b?);
    if a.float_elems.is_some() || b.float_elems.is_some() {
        return None;
    }
    let apply = |x: &DimExpr, y: &DimExpr| -> Option<DimExpr> {
        match op {
            "Add" => Some(x.add(y)),
            "Sub" => Some(x.sub(y)),
            "Mul" => Some(x.mul(y)),
            "Div" => x.checked_div(y),
            "Min" => const_binop(x, y, i64::min),
            "Max" => const_binop(x, y, i64::max),
            _ => None,
        }
    };
    let elems = match (a.is_scalar(), b.is_scalar()) {
        (false, false) => {
            if a.elems.len() != b.elems.len() {
                return None;
            }
            a.elems
                .iter()
                .zip(&b.elems)
                .map(|(x, y)| apply(x, y))
                .collect::<Option<Vec<_>>>()?
        }
        (false, true) => {
            let y = b.elems.first()?;
            a.elems
                .iter()
                .map(|x| apply(x, y))
                .collect::<Option<Vec<_>>>()?
        }
        (true, false) => {
            let x = a.elems.first()?;
            b.elems
                .iter()
                .map(|y| apply(x, y))
                .collect::<Option<Vec<_>>>()?
        }
        (true, true) => vec![apply(a.elems.first()?, b.elems.first()?)?],
    };
    let dims = if a.is_scalar() && b.is_scalar() {
        Vec::new()
    } else {
        vec![elems.len()]
    };
    // Carry the operands' integer dtype rather than assuming Int64 (a shape
    // chain may be Int32); the values are identical, only the label differs.
    Some(ShapeData {
        dtype: a.dtype,
        dims,
        elems,
        float_elems: None,
    })
}

/// Apply `f` only when both operands are concrete constants.
fn const_binop(x: &DimExpr, y: &DimExpr, f: fn(i64, i64) -> i64) -> Option<DimExpr> {
    match (x.as_const(), y.as_const()) {
        (Some(a), Some(b)) => Some(DimExpr::constant(f(a, b))),
        _ => None,
    }
}

/// Register the elementwise family.
pub fn register(reg: &mut InferenceRegistry) {
    for op in [
        "Relu",
        "Erf",
        "Tanh",
        "Sigmoid",
        "Sqrt",
        "Exp",
        "Log",
        "Neg",
        "Abs",
        "Sin",
        "Cos",
        "Reciprocal",
        "Softplus",
        "Softsign",
        "Floor",
        "Ceil",
        "Round",
        "Sign",
    ] {
        reg.register("", op, 1, unary);
    }
    for op in ["Acos", "Asin", "Atan", "Tan"] {
        reg.register("", op, 7, unary);
    }
    for op in ["Acosh", "Asinh", "Atanh", "Cosh", "Sinh"] {
        reg.register("", op, 9, unary);
    }
    for op in ["Clip", "Elu", "HardSigmoid", "LeakyRelu"] {
        reg.register("", op, 1, unary);
    }
    reg.register("", "Selu", 6, unary);
    reg.register("", "ThresholdedRelu", 10, unary);
    // Additional shape- and dtype-preserving activations, gated at their
    // `since_version`: `Shrink` (9), `Celu` (12), `HardSwish` (14), `Mish` (18).
    reg.register("", "Shrink", 9, unary);
    reg.register("", "Celu", 12, unary);
    reg.register("", "HardSwish", 14, unary);
    reg.register("", "Mish", 18, unary);
    reg.register("", "Hardmax", 13, unary);
    reg.register("", "LpNormalization", 1, unary);
    reg.register("", "GroupNormalization", 18, unary);
    reg.register("", "GroupNormalization", 21, unary);
    // `ai.onnx::Gelu` (opset 20): same-shape elementwise activation. Registered
    // at its since_version so shape-inference membership agrees with the CPU
    // kernel (also opset 20); the contrib `com.microsoft::Gelu` is separate.
    reg.register("", "Gelu", 20, unary);
    // `ai.onnx::Swish` (opset 24): elementwise x·sigmoid(alpha·x), same-shape.
    reg.register("", "Swish", 24, unary);
    for op in ["Add", "Sub", "Mul", "Div", "Pow"] {
        reg.register("", op, 1, binary);
    }
    reg.register("", "Mod", 10, binary);
    reg.register("", "BitShift", 11, binary);
    for op in ["BitwiseAnd", "BitwiseOr", "BitwiseXor"] {
        reg.register("", op, 18, binary);
    }
    reg.register("", "PRelu", 16, binary);
    for op in ["Less", "Greater", "Equal", "And", "Or", "Xor"] {
        reg.register("", op, 1, boolean_binary);
    }
    for op in ["LessOrEqual", "GreaterOrEqual"] {
        reg.register("", op, 12, boolean_binary);
    }
    reg.register("", "Not", 1, logical_not);
    reg.register("", "BitwiseNot", 18, unary);
    reg.register("", "IsNaN", 9, unary_predicate);
    reg.register("", "IsInf", 10, unary_predicate);
    for op in ["Min", "Max", "Sum", "Mean"] {
        reg.register("", op, 1, variadic);
    }
    reg.register("", "Where", 1, where_op);

    // com.microsoft elementwise activations (shape-preserving).
    for op in ["Gelu", "FastGelu", "BiasGelu", "QuickGelu", "Silu"] {
        reg.register("com.microsoft", op, 1, unary);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::DataType;

    #[test]
    fn shape_data_arithmetic_ignores_float_side_channels() {
        let float = ShapeData::float_scalar(DataType::Float32, 1.0);
        let integer = ShapeData::scalar(DataType::Int64, DimExpr::constant(2));

        assert_eq!(binary_shape_data("Add", Some(&float), Some(&integer)), None);
        assert_eq!(binary_shape_data("Add", Some(&integer), Some(&float)), None);
    }
}

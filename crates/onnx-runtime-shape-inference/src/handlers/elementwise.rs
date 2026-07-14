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
            let y = &b.elems[0];
            a.elems
                .iter()
                .map(|x| apply(x, y))
                .collect::<Option<Vec<_>>>()?
        }
        (true, false) => {
            let x = &a.elems[0];
            b.elems
                .iter()
                .map(|y| apply(x, y))
                .collect::<Option<Vec<_>>>()?
        }
        (true, true) => vec![apply(&a.elems[0], &b.elems[0])?],
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
        "Gelu",
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
        "Floor",
        "Ceil",
        "Round",
        "Not",
        "Sign",
    ] {
        reg.register("", op, 1, unary);
    }
    for op in ["Add", "Sub", "Mul", "Div", "Pow"] {
        reg.register("", op, 1, binary);
    }
    for op in ["Min", "Max", "Sum", "Mean"] {
        reg.register("", op, 1, variadic);
    }
    reg.register("", "Where", 1, where_op);

    // com.microsoft elementwise activations (shape-preserving).
    for op in ["Gelu", "FastGelu", "BiasGelu", "QuickGelu"] {
        reg.register("com.microsoft", op, 1, unary);
    }
}

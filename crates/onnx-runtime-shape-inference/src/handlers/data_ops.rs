//! Data/shape source rules: `Shape`, `Size`, `Constant`, `ConstantOfShape`,
//! `Cast`, `Identity`. These are the primary *producers* of shape-data — the
//! head of every shape-computation chain.

use onnx_runtime_ir::{Attribute, DataType};

use crate::context::InferenceContext;
use crate::dim_expr::DimExpr;
use crate::error::ShapeInferError;
use crate::registry::InferenceRegistry;
use crate::shape_data::ShapeData;

/// `Shape`: emit the input's dims as a 1-D int64 tensor, and record those dims
/// as shape-data so downstream `Gather`/`Slice`/`Concat`/`Reshape` resolve.
pub fn shape(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(input) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    let rank = input.len() as i64;
    // opset-15 start/end slicing of the dim list.
    let clamp = |v: i64| -> usize {
        let v = if v < 0 { v + rank } else { v };
        v.clamp(0, rank) as usize
    };
    let start = clamp(
        ctx.node
            .attr("start")
            .and_then(Attribute::as_int)
            .unwrap_or(0),
    );
    let end = clamp(
        ctx.node
            .attr("end")
            .and_then(Attribute::as_int)
            .unwrap_or(rank),
    );
    let elems: Vec<DimExpr> = input.get(start..end.max(start)).unwrap_or(&[]).to_vec();

    ctx.set_output(
        0,
        DataType::Int64,
        vec![DimExpr::constant(elems.len() as i64)],
    );
    ctx.set_output_shape_data(0, ShapeData::vector(DataType::Int64, elems));
    Ok(())
}

/// `Size`: the total element count as a rank-0 int64 scalar (symbolic when any
/// dim is symbolic).
pub fn size(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(input) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    let total = DimExpr::product(&input);
    ctx.set_output(0, DataType::Int64, Vec::new());
    ctx.set_output_shape_data(0, ShapeData::scalar(DataType::Int64, total));
    Ok(())
}

/// `Constant`: derive shape, dtype, and (for small integer tensors) shape-data
/// from the node's attribute.
pub fn constant(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let node = ctx.node;
    if let Some(Attribute::Tensor(t)) = node.attr("value") {
        let shape = t
            .dims
            .iter()
            .map(|&d| DimExpr::constant(d as i64))
            .collect();
        ctx.set_output(0, t.dtype, shape);
        if let Some(sd) = ShapeData::from_tensor(t.dtype, &t.dims, &t.data) {
            ctx.set_output_shape_data(0, sd);
        }
        return Ok(());
    }
    if let Some(i) = node.attr("value_int").and_then(Attribute::as_int) {
        ctx.set_output(0, DataType::Int64, Vec::new());
        ctx.set_output_shape_data(0, ShapeData::scalar(DataType::Int64, DimExpr::constant(i)));
        return Ok(());
    }
    if let Some(v) = node.attr("value_ints").and_then(Attribute::as_ints) {
        let elems: Vec<DimExpr> = v.iter().map(|&x| DimExpr::constant(x)).collect();
        ctx.set_output(0, DataType::Int64, vec![DimExpr::constant(v.len() as i64)]);
        ctx.set_output_shape_data(0, ShapeData::vector(DataType::Int64, elems));
        return Ok(());
    }
    if node.attr("value_float").is_some() {
        ctx.set_output(0, DataType::Float32, Vec::new());
        return Ok(());
    }
    if let Some(Attribute::Floats(v)) = node.attr("value_floats") {
        ctx.set_output(
            0,
            DataType::Float32,
            vec![DimExpr::constant(v.len() as i64)],
        );
        return Ok(());
    }
    Ok(())
}

/// `ConstantOfShape`: output shape is the (shape-data) input vector; dtype from
/// the `value` attribute (default float32).
pub fn constant_of_shape(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let (dtype, fill) = value_attr_scalar(ctx.node);
    if let Some(sd) = ctx.input_shape_data(0).cloned() {
        let shape = sd.as_shape();
        let numel_known = shape.iter().all(|d| d.as_const().is_some());
        ctx.set_output(0, dtype, shape.clone());
        // Small all-constant integer fills can themselves seed a chain.
        if let (Some(fill), true) = (
            fill,
            numel_known && (dtype.is_int() || dtype == DataType::Bool),
        ) {
            // Overflow-safe product: a pathological shape whose element count
            // exceeds i64 degrades to "no shape-data" rather than panicking or
            // wrapping (mirrors the DimExpr overflow contract).
            let numel: Option<i64> = shape
                .iter()
                .filter_map(|d| d.as_const())
                .try_fold(1i64, |acc, d| acc.checked_mul(d));
            if let Some(numel) = numel
                && (0..=1024).contains(&numel)
            {
                let elems = vec![fill; numel as usize];
                ctx.set_output_shape_data(0, ShapeData::vector(dtype, elems));
            }
        }
        return Ok(());
    }
    // No shape-data: fall back to a fresh-symbol shape of the known rank.
    if let Some(shape) = ctx.input_shape(0).map(<[DimExpr]>::to_vec)
        && shape.len() == 1
        && let Some(rank) = shape[0].as_const()
    {
        let out = (0..rank).map(|_| ctx.fresh_dim()).collect();
        ctx.set_output(0, dtype, out);
    }
    Ok(())
}

/// `Cast`: shape-preserving dtype change; shape-data is cast through if the
/// target type stays integral.
pub fn cast(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(to) = ctx.node.attr("to").and_then(Attribute::as_int) else {
        return Err(ShapeInferError::MissingAttribute {
            op: "Cast".into(),
            attr: "to".into(),
        });
    };
    let Some(dtype) = DataType::from_onnx(to as i32) else {
        return Err(ShapeInferError::Invalid {
            op: "Cast".into(),
            detail: format!("unknown target dtype {to}"),
        });
    };
    if let Some(shape) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) {
        ctx.set_output(0, dtype, shape);
    }
    if (dtype.is_int() || dtype == DataType::Bool)
        && let Some(sd) = ctx.input_shape_data(0).cloned()
    {
        ctx.set_output_shape_data(
            0,
            ShapeData {
                dtype,
                dims: sd.dims,
                elems: sd.elems,
                float_elems: None,
            },
        );
    }
    Ok(())
}

/// `Identity`: pass the input's type and shape-data straight through.
pub fn identity(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    if let Some(t) = ctx.input_type(0).cloned() {
        ctx.set_output_type(0, t);
    }
    if let Some(sd) = ctx.input_shape_data(0).cloned() {
        ctx.set_output_shape_data(0, sd);
    }
    Ok(())
}

/// Read the `value` attribute of a `ConstantOfShape` as `(dtype, fill_scalar)`.
fn value_attr_scalar(node: &onnx_runtime_ir::Node) -> (DataType, Option<DimExpr>) {
    if let Some(Attribute::Tensor(t)) = node.attr("value") {
        let fill = ShapeData::from_tensor(t.dtype, &t.dims, &t.data)
            .and_then(|sd| sd.elems.into_iter().next());
        return (t.dtype, fill);
    }
    (DataType::Float32, None)
}

/// Register the data/shape source family.
pub fn register(reg: &mut InferenceRegistry) {
    reg.register("", "Shape", 1, shape);
    reg.register("", "Size", 1, size);
    reg.register("", "Constant", 1, constant);
    reg.register("", "ConstantOfShape", 1, constant_of_shape);
    reg.register("", "Cast", 1, cast);
    reg.register("", "CastLike", 1, cast_like);
    reg.register("", "Identity", 1, identity);
    for version in [10, 13, 19, 21, 23, 25] {
        reg.register("", "QuantizeLinear", version, quantize_linear);
        reg.register("", "DequantizeLinear", version, dequantize_linear);
    }
    reg.register("", "DynamicQuantizeLinear", 11, dynamic_quantize_linear);
}

/// `CastLike(input, target_type)`: shape from input, dtype from the second
/// operand.
fn cast_like(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let dtype = ctx.input_dtype(1);
    if let (Some(shape), Some(dtype)) = (ctx.input_shape(0).map(<[DimExpr]>::to_vec), dtype) {
        ctx.set_output(0, dtype, shape);
    }
    Ok(())
}

/// `QuantizeLinear`: output shape follows x; its type follows zero_point, or
/// defaults to uint8 when zero_point is omitted.
fn quantize_linear(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let dtype = if ctx.has_input(2) {
        ctx.input_dtype(2)
    } else {
        Some(DataType::Uint8)
    };
    if let (Some(shape), Some(dtype)) = (ctx.input_shape(0).map(<[DimExpr]>::to_vec), dtype) {
        ctx.set_output(0, dtype, shape);
    }
    Ok(())
}

/// `DequantizeLinear`: output shape follows x and dtype follows scale.
fn dequantize_linear(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    if let (Some(shape), Some(dtype)) = (
        ctx.input_shape(0).map(<[DimExpr]>::to_vec),
        ctx.input_dtype(1),
    ) {
        ctx.set_output(0, dtype, shape);
    }
    Ok(())
}

/// `DynamicQuantizeLinear`: uint8 data with scalar float32 scale and uint8
/// zero_point.
fn dynamic_quantize_linear(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    if let Some(shape) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) {
        ctx.set_output(0, DataType::Uint8, shape);
        ctx.set_output(1, DataType::Float32, Vec::new());
        ctx.set_output(2, DataType::Uint8, Vec::new());
    }
    Ok(())
}

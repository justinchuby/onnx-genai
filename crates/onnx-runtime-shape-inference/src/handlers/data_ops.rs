//! Data/shape source rules: `Shape`, `Size`, `Constant`, `ConstantOfShape`,
//! `Cast`, `Identity`. These are the primary *producers* of shape-data — the
//! head of every shape-computation chain.

use onnx_runtime_ir::{Attribute, DataType};

use crate::context::InferenceContext;
use crate::dim_expr::DimExpr;
use crate::error::ShapeInferError;
use crate::handlers::checked_axis;
use crate::registry::InferenceRegistry;
use crate::shape_data::ShapeData;

/// `Shape`: emit the input's dims as a 1-D int64 tensor, and record those dims
/// as shape-data so downstream `Gather`/`Slice`/`Concat`/`Reshape` resolve.
pub fn shape(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(input) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    let rank = i64::try_from(input.len()).map_err(|_| ShapeInferError::Invalid {
        op: "Shape".into(),
        detail: "input rank exceeds the supported integer range".into(),
    })?;
    // opset-15 start/end slicing of the dim list.
    let clamp = |v: i64| -> usize {
        let v = if v < 0 {
            v.checked_add(rank).unwrap_or(i64::MIN)
        } else {
            v
        };
        usize::try_from(v.clamp(0, rank)).expect("clamped shape axis fits usize")
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
    if input.iter().all(DimExpr::is_const) {
        let total = input.iter().try_fold(1usize, |total, dim| {
            let extent = usize::try_from(dim.as_const().expect("checked constant")).ok()?;
            total.checked_mul(extent)
        });
        if total.is_none_or(|total| total > isize::MAX as usize) {
            return Err(ShapeInferError::Invalid {
                op: "Size".into(),
                detail: "input element count exceeds isize::MAX".into(),
            });
        }
    }
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
    if let Some(rank) = ctx.input_rank(0)
        && rank != 1
    {
        return Err(ShapeInferError::InvalidRank {
            op: "ConstantOfShape".into(),
            index: 0,
            rank,
            detail: "shape input must be a 1-D tensor".into(),
        });
    }
    if let Some(sd) = ctx.input_shape_data(0).cloned() {
        let shape = sd.as_shape();
        for dim in &shape {
            if let Some(extent) = dim.as_const()
                && !(0..=isize::MAX as i64).contains(&extent)
            {
                return Err(ShapeInferError::Invalid {
                    op: "ConstantOfShape".into(),
                    detail: format!("shape extent {extent} is outside 0..=isize::MAX"),
                });
            }
        }
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

fn quantized_output_dtype(ctx: &InferenceContext) -> Result<Option<DataType>, ShapeInferError> {
    let Some(raw) = ctx
        .node
        .attr("output_dtype")
        .and_then(Attribute::as_int)
        .filter(|&raw| raw != 0)
    else {
        return Ok(None);
    };
    DataType::from_onnx(raw as i32)
        .map(Some)
        .ok_or_else(|| ShapeInferError::Invalid {
            op: ctx.op().into(),
            detail: format!("unknown output_dtype {raw}"),
        })
}

fn validate_quantization_granularity(
    ctx: &InferenceContext,
    data_input: usize,
    scale_input: usize,
    zero_point_input: usize,
) -> Result<(), ShapeInferError> {
    let (Some(data), Some(scale)) = (ctx.input_type(data_input), ctx.input_type(scale_input))
    else {
        return Ok(());
    };
    let data_rank = data.rank();
    let scale_rank = scale.rank();
    let block_size = if ctx.opset("") >= 21 {
        ctx.node
            .attr("block_size")
            .and_then(Attribute::as_int)
            .unwrap_or(0)
    } else {
        0
    };
    if block_size == 0 && scale_rank == 0 {
        return Ok(());
    }
    if data_rank == 0 {
        return Err(ShapeInferError::Invalid {
            op: ctx.op().into(),
            detail: "non-scalar scale cannot quantize a scalar input".into(),
        });
    }
    let raw_axis = ctx
        .node
        .attr("axis")
        .and_then(Attribute::as_int)
        .unwrap_or(1);
    let axis = checked_axis(raw_axis, data_rank).ok_or_else(|| ShapeInferError::Invalid {
        op: ctx.op().into(),
        detail: format!("axis {raw_axis} is out of range for rank {data_rank}"),
    })?;

    if block_size != 0 {
        if scale_rank != data_rank {
            return Err(ShapeInferError::Invalid {
                op: ctx.op().into(),
                detail: format!(
                    "blocked scale rank {scale_rank} must match input rank {data_rank}"
                ),
            });
        }
        if block_size <= 0 {
            return Err(ShapeInferError::Invalid {
                op: ctx.op().into(),
                detail: "blocked quantization requires a positive block_size".into(),
            });
        }
        let zero_point = ctx.input_type(zero_point_input);
        if let Some(zero_point) = zero_point
            && zero_point.rank() != data_rank
        {
            return Err(ShapeInferError::Invalid {
                op: ctx.op().into(),
                detail: format!(
                    "blocked zero-point rank {} must match input rank {data_rank}",
                    zero_point.rank()
                ),
            });
        }
        for dimension in 0..data_rank {
            let Some(data_extent) = data.shape[dimension].as_const() else {
                continue;
            };
            let expected = if dimension == axis {
                i128::from(data_extent)
                    .checked_add(i128::from(block_size) - 1)
                    .ok_or_else(|| ShapeInferError::Invalid {
                        op: ctx.op().into(),
                        detail: "blocked quantization extent arithmetic overflowed".into(),
                    })?
                    / i128::from(block_size)
            } else {
                i128::from(data_extent)
            };
            for (name, extent) in [
                ("scale", scale.shape[dimension].as_const()),
                (
                    "zero-point",
                    zero_point.and_then(|value| value.shape[dimension].as_const()),
                ),
            ] {
                if let Some(extent) = extent
                    && i128::from(extent) != expected
                {
                    return Err(ShapeInferError::Invalid {
                        op: ctx.op().into(),
                        detail: format!(
                            "blocked {name} dimension {dimension} is {extent}, expected {expected}"
                        ),
                    });
                }
            }
        }
        return Ok(());
    }

    if scale_rank == 1 {
        if let (Some(scale_extent), Some(data_extent)) =
            (scale.shape[0].as_const(), data.shape[axis].as_const())
            && scale_extent != data_extent
        {
            return Err(ShapeInferError::Invalid {
                op: ctx.op().into(),
                detail: format!(
                    "per-axis scale length {scale_extent} does not match input axis {axis} extent {data_extent}"
                ),
            });
        }
        return Ok(());
    }
    if ctx.opset("") < 21 || scale_rank != data_rank {
        return Err(ShapeInferError::Invalid {
            op: ctx.op().into(),
            detail: format!("scale rank {scale_rank} must be 0, 1, or the input rank {data_rank}"),
        });
    }

    Err(ShapeInferError::Invalid {
        op: ctx.op().into(),
        detail: "blocked quantization requires a positive block_size".into(),
    })
}

/// `QuantizeLinear`: output shape follows x; its type follows zero_point,
/// `output_dtype` (opset 21+), or defaults to uint8.
fn quantize_linear(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    validate_quantization_granularity(ctx, 0, 1, 2)?;
    let attribute_dtype = quantized_output_dtype(ctx)?;
    let dtype = if ctx.has_input(2) {
        let Some(zero_point_dtype) = ctx.input_dtype(2) else {
            return Ok(());
        };
        if attribute_dtype.is_some_and(|dtype| dtype != zero_point_dtype) {
            return Err(ShapeInferError::Invalid {
                op: "QuantizeLinear".into(),
                detail: "output_dtype does not match y_zero_point dtype".into(),
            });
        }
        Some(zero_point_dtype)
    } else {
        Some(attribute_dtype.unwrap_or(DataType::Uint8))
    };
    if let (Some(shape), Some(dtype)) = (ctx.input_shape(0).map(<[DimExpr]>::to_vec), dtype) {
        ctx.set_output(0, dtype, shape);
    }
    Ok(())
}

/// `DequantizeLinear`: output shape follows x and dtype follows scale.
fn dequantize_linear(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    validate_quantization_granularity(ctx, 0, 1, 2)?;
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

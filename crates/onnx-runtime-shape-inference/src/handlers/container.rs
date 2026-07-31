//! Container-type rules — the first proven slice of the `Sequence` family.
//!
//! These rules exercise the additive [`ValueType`] layer: they read and produce
//! container element types instead of (only) tensor [`TypeInfo`]. Tensor
//! operators are untouched. See issue #449 for the multi-PR roadmap; this module
//! implements `SequenceEmpty`, `SequenceConstruct`, `SequenceLength`, and
//! `SequenceAt`.

use onnx_runtime_ir::{Attribute, DataType};

use crate::context::{InferenceContext, TensorType, TypeInfo, TypedShape, ValueType};
use crate::dim_expr::DimExpr;
use crate::error::ShapeInferError;
use crate::registry::InferenceRegistry;

/// The `dtype` attribute as a [`DataType`], if present and recognised.
fn dtype_attr(ctx: &InferenceContext) -> Option<DataType> {
    let raw = ctx.node.attr("dtype").and_then(Attribute::as_int)?;
    i32::try_from(raw).ok().and_then(DataType::from_onnx)
}

/// `SequenceEmpty` (opset 11): an empty sequence whose element is a tensor of
/// the optional `dtype` attribute (default `Float32`). The element shape is
/// unknown — an empty sequence carries no exemplar — so the element is a
/// dtype-only tensor.
fn sequence_empty(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let dtype = dtype_attr(ctx).unwrap_or(DataType::Float32);
    let element = ValueType::Tensor(TensorType::dtype_only(dtype));
    ctx.set_output_value_type(0, ValueType::sequence(element));
    Ok(())
}

/// `SequenceConstruct` (opset 11): a sequence of the common element type of the
/// tensor inputs. ONNX requires the inputs to be homogeneous, so a dtype
/// disagreement is an error. The element shape is the per-dimension agreement of
/// the inputs (equal dims — including symbolic — are preserved; disagreements
/// degrade to a fresh symbol; differing ranks yield an unknown element shape).
fn sequence_construct(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let mut element: Option<TensorType> = None;
    let mut shape_known = true;
    for i in 0..ctx.num_inputs() {
        if !ctx.has_input(i) {
            continue;
        }
        let Some(input) = ctx.input_type(i).cloned() else {
            // A present-but-untyped input: we cannot confirm the element shape
            // agrees, so the shape becomes unknown while the dtype (from typed
            // siblings) is still recovered.
            shape_known = false;
            continue;
        };
        element = Some(match element.take() {
            None => TensorType::from(input),
            Some(acc) => merge_element(ctx, acc, input)?,
        });
    }
    let Some(mut element) = element else {
        // No typed inputs: nothing to constrain the element type.
        return Ok(());
    };
    if !shape_known {
        element.shape = None;
    }
    ctx.set_output_value_type(0, ValueType::sequence(ValueType::Tensor(element)));
    Ok(())
}

/// Merge an accumulated element tensor type with another input's type: dtypes
/// must match (ONNX homogeneity), shapes agree per dimension.
fn merge_element(
    ctx: &mut InferenceContext,
    acc: TensorType,
    input: TypeInfo,
) -> Result<TensorType, ShapeInferError> {
    if acc.dtype != input.dtype {
        return Err(ShapeInferError::Invalid {
            op: "SequenceConstruct".into(),
            detail: format!(
                "sequence elements must share a dtype, found {:?} and {:?}",
                acc.dtype, input.dtype
            ),
        });
    }
    let shape = match acc.shape {
        Some(acc_shape) => merge_shape(ctx, &acc_shape, &input.shape),
        None => None,
    };
    Ok(TensorType {
        dtype: acc.dtype,
        shape,
    })
}

/// Per-dimension agreement of two element shapes. Differing ranks yield `None`
/// (unknown); within a matching rank, structurally-equal dims (including
/// symbolic ones) are preserved and disagreements degrade to a fresh symbol.
fn merge_shape(ctx: &mut InferenceContext, a: &[DimExpr], b: &[DimExpr]) -> Option<TypedShape> {
    if a.len() != b.len() {
        return None;
    }
    let merged = a
        .iter()
        .zip(b.iter())
        .map(|(da, db)| {
            if da == db {
                da.clone()
            } else {
                ctx.fresh_dim()
            }
        })
        .collect();
    Some(merged)
}

/// `SequenceLength` (opset 11): the number of elements in a sequence, an `i64`
/// scalar tensor. The result is independent of the element type.
fn sequence_length(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    ctx.set_output(0, DataType::Int64, Vec::new());
    Ok(())
}

/// `SequenceAt` (opset 11): the element of a sequence at a given index; its type
/// is the sequence's element type. Recovers the full tensor type when the
/// element shape is known (preserving symbolic dims); a dtype-only element stays
/// unresolved at the tensor level (pending the unknown-rank follow-up).
fn sequence_at(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(element) = ctx
        .input_value_type(0)
        .and_then(ValueType::as_sequence_element)
    else {
        return Ok(());
    };
    if let Some(tensor) = element.as_tensor()
        && let Some(type_info) = tensor.to_type_info()
    {
        ctx.set_output_type(0, type_info);
    }
    Ok(())
}

/// Register the container-type rules.
pub fn register(reg: &mut InferenceRegistry) {
    reg.register("", "SequenceEmpty", 11, sequence_empty);
    reg.register("", "SequenceConstruct", 11, sequence_construct);
    reg.register("", "SequenceLength", 11, sequence_length);
    reg.register("", "SequenceAt", 11, sequence_at);
}

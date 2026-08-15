//! Container-type rules — the first proven slice of the `Sequence` family.
//!
//! These rules exercise the additive [`ValueType`] layer: they read and produce
//! container element types instead of (only) tensor [`TypeInfo`]. Tensor
//! operators are untouched. See issue #449 for the multi-PR roadmap.
//!
//! Increment 1 (foundation) implemented `SequenceEmpty`, `SequenceConstruct`,
//! `SequenceLength`, and `SequenceAt`. Increment 2 adds the sequence *mutation*
//! ops (`SequenceInsert`, `SequenceErase`) and the tensor⇔sequence *conversion*
//! ops (`SplitToSequence` for tensor→sequence, `ConcatFromSequence` for the
//! sequence→tensor direction of the seam).

use onnx_runtime_ir::{Attribute, DataType};

use crate::context::{InferenceContext, TensorType, ValueType, unify_tensor_type};
use crate::dim_expr::DimExpr;
use crate::error::ShapeInferError;
use crate::handlers::checked_axis;
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
            Some(acc) => {
                unify_tensor_type(ctx.interner_mut(), "SequenceConstruct", acc, input.into())?
            }
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

/// The tensor leaf of input `i`'s sequence element type, when `i` is a
/// sequence-of-tensor value. Shared by every rule that reads an element type
/// out of a sequence input.
fn sequence_element_tensor(ctx: &InferenceContext, i: usize) -> Option<TensorType> {
    ctx.input_value_type(i)?
        .as_sequence_element()?
        .as_tensor()
        .cloned()
}

/// `SequenceInsert` (opset 11): insert `tensor` (input 1) into `input_sequence`
/// (input 0) at an optional `position` (input 2). The result's element type
/// unifies the existing element type with the inserted tensor under the same
/// homogeneity rules as `SequenceConstruct`; `position` never affects the type.
fn sequence_insert(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let existing = sequence_element_tensor(ctx, 0);
    let inserted = ctx.input_type(1).cloned().map(TensorType::from);
    let element = match (existing, inserted) {
        (Some(acc), Some(ins)) => {
            unify_tensor_type(ctx.interner_mut(), "SequenceInsert", acc, ins)?
        }
        // Only one side is typed: keep its dtype but drop the shape, since we
        // cannot confirm the two element shapes agree.
        (Some(mut acc), None) => {
            acc.shape = None;
            acc
        }
        (None, Some(ins)) => ins,
        (None, None) => return Ok(()),
    };
    ctx.set_output_value_type(0, ValueType::sequence(ValueType::Tensor(element)));
    Ok(())
}

/// `SequenceErase` (opset 11): remove the element at an optional `position`. The
/// element type is unchanged — only the length (which the type layer does not
/// track) decreases.
fn sequence_erase(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    if let Some(element) = ctx
        .input_value_type(0)
        .and_then(ValueType::as_sequence_element)
        .cloned()
    {
        ctx.set_output_value_type(0, ValueType::sequence(element));
    }
    Ok(())
}

/// `SplitToSequence` (opset 11): split a tensor (input 0) into a sequence along
/// `axis`. The element dtype is the input dtype. When an explicit `split` input
/// (input 1) is present the per-chunk extent along `axis` varies, so it degrades
/// to a fresh symbol; otherwise each chunk is size 1 and `keepdims` decides
/// whether the split axis is kept (extent 1) or removed.
fn split_to_sequence(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(input) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    let rank = input.rank();
    if rank == 0 {
        return Err(ShapeInferError::InvalidRank {
            op: "SplitToSequence".into(),
            index: 0,
            rank,
            detail: "input must have rank at least 1".into(),
        });
    }
    let axis_attr = ctx
        .node
        .attr("axis")
        .and_then(Attribute::as_int)
        .unwrap_or(0);
    let axis = checked_axis(axis_attr, rank).ok_or_else(|| ShapeInferError::Invalid {
        op: "SplitToSequence".into(),
        detail: format!("axis {axis_attr} is outside [-{rank}, {rank})"),
    })?;
    let keepdims = ctx
        .node
        .attr("keepdims")
        .and_then(Attribute::as_int)
        .unwrap_or(1)
        != 0;
    let mut shape = input.shape.clone();
    if ctx.has_input(1) {
        // Explicit split (scalar chunk size or 1-D sizes): chunk extents vary
        // along the axis, so a single element type cannot pin it down.
        shape[axis] = ctx.fresh_dim();
    } else if keepdims {
        shape[axis] = DimExpr::constant(1);
    } else {
        shape.remove(axis);
    }
    let element = TensorType::new(input.dtype, shape);
    ctx.set_output_value_type(0, ValueType::sequence(ValueType::Tensor(element)));
    Ok(())
}

/// `ConcatFromSequence` (opset 11): recover a tensor from a sequence by
/// concatenating its elements along `axis` (or stacking on a new axis when
/// `new_axis=1`). This is the sequence→tensor direction of the container seam:
/// the output dtype is the sequence element dtype, and the output shape is the
/// element shape with the concat axis made symbolic (its extent is the unknown
/// total across the sequence) or, for `new_axis=1`, a fresh symbolic dimension
/// inserted at `axis`. A dtype-only element leaves the output unresolved (its
/// rank is unknown), the same honest degradation as `SequenceAt`.
fn concat_from_sequence(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let axis_attr = ctx
        .node
        .attr("axis")
        .and_then(Attribute::as_int)
        .ok_or_else(|| ShapeInferError::Invalid {
            op: "ConcatFromSequence".into(),
            detail: "requires the mandatory 'axis' attribute".into(),
        })?;
    let new_axis = ctx
        .node
        .attr("new_axis")
        .and_then(Attribute::as_int)
        .unwrap_or(0)
        != 0;
    let Some(element) = sequence_element_tensor(ctx, 0) else {
        return Ok(());
    };
    let Some(mut shape) = element.shape else {
        return Ok(());
    };
    let output_rank = shape.len() + usize::from(new_axis);
    let axis = checked_axis(axis_attr, output_rank).ok_or_else(|| ShapeInferError::Invalid {
        op: "ConcatFromSequence".into(),
        detail: format!("axis {axis_attr} is outside [-{output_rank}, {output_rank})"),
    })?;
    if new_axis {
        shape.insert(axis, ctx.fresh_dim());
    } else {
        shape[axis] = ctx.fresh_dim();
    }
    ctx.set_output(0, element.dtype, shape);
    Ok(())
}

/// Register the container-type rules.
pub fn register(reg: &mut InferenceRegistry) {
    reg.register("", "SequenceEmpty", 11, sequence_empty);
    reg.register("", "SequenceConstruct", 11, sequence_construct);
    reg.register("", "SequenceLength", 11, sequence_length);
    reg.register("", "SequenceAt", 11, sequence_at);
    reg.register("", "SequenceInsert", 11, sequence_insert);
    reg.register("", "SequenceErase", 11, sequence_erase);
    reg.register("", "SplitToSequence", 11, split_to_sequence);
    reg.register("", "ConcatFromSequence", 11, concat_from_sequence);
}

//! `Einsum` (opset 12) shape inference through the shared canonical planner.
//!
//! Parsing, validation, ellipsis expansion, diagonals, reductions, and output
//! ordering live in `onnx-runtime-ir::EinsumPlan`. This handler only supplies
//! shape-inference dimensions and resolves ellipsis broadcasts through
//! [`InferenceContext::broadcast_dim`], preserving the context's symbolic
//! representative and lineage bookkeeping.

use onnx_runtime_ir::{Attribute, EinsumInput, EinsumPlan, EinsumResolveError};

use crate::context::{InferenceContext, TypeInfo};
use crate::dim_expr::DimExpr;
use crate::error::ShapeInferError;
use crate::registry::InferenceRegistry;

fn einsum(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(equation) = ctx.node.attr("equation").and_then(Attribute::as_str) else {
        return Ok(());
    };

    // Clone before planning so output resolution can mutably use the context's
    // broadcast chokepoint without retaining immutable borrows into `ctx`.
    let input_types: Vec<Option<TypeInfo>> = (0..ctx.num_inputs())
        .map(|i| ctx.input_type(i).cloned())
        .collect();
    let inputs: Vec<_> = input_types
        .iter()
        .map(|input| {
            EinsumInput::from_optional(
                input.as_ref().map(|type_info| type_info.dtype),
                input.as_ref().map(|type_info| type_info.shape.as_slice()),
            )
        })
        .collect();
    let plan = match EinsumPlan::build(equation, &inputs) {
        Ok(plan) => plan,
        Err(error) if error.is_incomplete_metadata() => return Ok(()),
        Err(error) => {
            return Err(ShapeInferError::Invalid {
                op: ctx.op().to_owned(),
                detail: error.to_string(),
            });
        }
    };

    let input_shapes: Vec<&[DimExpr]> = input_types
        .iter()
        .enumerate()
        .map(|(input_index, input)| {
            input
                .as_ref()
                .ok_or_else(|| ShapeInferError::Invalid {
                    op: ctx.op().to_owned(),
                    detail: format!(
                        "shared Einsum plan admitted input #{input_index} without resolved type metadata"
                    ),
                })
                .map(|type_info| type_info.shape.as_slice())
        })
        .collect::<Result<_, _>>()?;
    let output_shape = match plan
        .resolve_output_shape(&input_shapes, |left, right| ctx.broadcast_dim(left, right))
    {
        Ok(shape) => shape,
        Err(EinsumResolveError::Broadcast { source, .. }) => return Err(source),
        Err(EinsumResolveError::InputCount { expected, found }) => {
            return Err(ShapeInferError::Invalid {
                op: ctx.op().to_owned(),
                detail: format!(
                    "shared Einsum plan expected {expected} resolved input shapes, found {found}"
                ),
            });
        }
        Err(EinsumResolveError::InputRank {
            input,
            expected,
            found,
        }) => {
            return Err(ShapeInferError::Invalid {
                op: ctx.op().to_owned(),
                detail: format!(
                    "shared Einsum plan expected input #{input} rank {expected}, found {found}"
                ),
            });
        }
    };

    ctx.set_output(0, plan.dtype(), output_shape);
    Ok(())
}

/// Register the `Einsum` rule.
pub fn register(reg: &mut InferenceRegistry) {
    reg.register("", "Einsum", 12, einsum);
}

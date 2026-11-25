use onnx_runtime_ir::{Attribute, DataType};

use crate::context::InferenceContext;
use crate::dim_expr::DimExpr;
use crate::error::ShapeInferError;
use crate::handlers::checked_axis;
use crate::shape_data::ShapeData;

use super::concat_slice::vector_length;
use super::{checked_extent, const_ints, validate_vector_input};

/// `Transpose`: permute dimensions by `perm` (default: reverse).
pub fn transpose(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(t) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    let rank = t.rank();
    let perm: Vec<usize> = match ctx.node.attr("perm").and_then(Attribute::as_ints) {
        Some(p) => p
            .iter()
            .map(|&axis| {
                checked_axis(axis, rank).ok_or_else(|| ShapeInferError::Invalid {
                    op: "Transpose".into(),
                    detail: format!("axis {axis} is out of range for rank {rank}"),
                })
            })
            .collect::<Result<_, _>>()?,
        None => (0..rank).rev().collect(),
    };
    if perm.len() != rank {
        return Err(ShapeInferError::Invalid {
            op: "Transpose".into(),
            detail: format!("perm length {} != rank {rank}", perm.len()),
        });
    }
    if perm
        .iter()
        .enumerate()
        .any(|(i, axis)| perm[..i].contains(axis))
    {
        return Err(ShapeInferError::Invalid {
            op: "Transpose".into(),
            detail: "perm must contain each axis exactly once".into(),
        });
    }
    let out: Vec<DimExpr> = perm.iter().map(|&p| t.shape[p].clone()).collect();
    ctx.set_output(0, t.dtype, out);
    Ok(())
}

/// `Reshape`: output shape from the (shape-data) target vector, resolving `0`
/// (copy) and `-1` (infer) dims symbolically.
pub fn reshape(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(input) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    let dtype = ctx.input_dtype(0).unwrap_or(DataType::Float32);
    let allowzero = ctx
        .node
        .attr("allowzero")
        .and_then(Attribute::as_int)
        .unwrap_or(0)
        != 0;

    let Some(target) = ctx.input_shape_data(1).map(ShapeData::as_shape) else {
        // A runtime target has unknown values, so even a known target-vector
        // length cannot establish the output dimensions.
        return Ok(());
    };

    let total = DimExpr::product(&input);
    let mut out: Vec<DimExpr> = Vec::with_capacity(target.len());
    let mut product = DimExpr::constant(1);
    let mut neg1: Option<usize> = None;
    let mut neg1_count = 0;
    for (i, t) in target.iter().enumerate() {
        match t.as_const() {
            Some(-1) => {
                neg1_count += 1;
                neg1 = Some(i);
                out.push(DimExpr::constant(1)); // placeholder, fixed below
            }
            Some(0) if !allowzero => {
                let d = input
                    .get(i)
                    .cloned()
                    .ok_or_else(|| ShapeInferError::Invalid {
                        op: "Reshape".into(),
                        detail: format!(
                            "0 at target index {i} has no corresponding input dimension"
                        ),
                    })?;
                product = product.mul(&d);
                out.push(d);
            }
            Some(value) if value < 0 => {
                return Err(ShapeInferError::Invalid {
                    op: "Reshape".into(),
                    detail: format!("target dimension {value} is invalid"),
                });
            }
            _ => {
                product = product.mul(t);
                out.push(t.clone());
            }
        }
    }
    if neg1_count > 1 {
        return Err(ShapeInferError::Invalid {
            op: "Reshape".into(),
            detail: format!("at most one dimension may be -1, found {neg1_count}"),
        });
    }
    if allowzero && neg1.is_some() && target.iter().any(|t| t.as_const() == Some(0)) {
        return Err(ShapeInferError::Invalid {
            op: "Reshape".into(),
            detail: "allowzero=1 does not permit 0 and -1 in the same target shape".into(),
        });
    }
    if let Some(idx) = neg1 {
        if product.as_const() == Some(0) {
            return Err(ShapeInferError::Invalid {
                op: "Reshape".into(),
                detail: "cannot infer -1 dimension when the remaining target product is zero"
                    .into(),
            });
        }
        out[idx] = match total.checked_div(&product) {
            Some(inferred) => inferred,
            None if total.is_const() && product.is_const() => {
                return Err(ShapeInferError::Invalid {
                    op: "Reshape".into(),
                    detail: "input element count is not divisible by the known target dimensions"
                        .into(),
                });
            }
            None => ctx.fresh_dim(),
        };
    } else if let (Some(input_elements), Some(output_elements)) =
        (total.as_const(), product.as_const())
        && input_elements != output_elements
    {
        return Err(ShapeInferError::Invalid {
            op: "Reshape".into(),
            detail: format!(
                "input element count {input_elements} does not match target element count {output_elements}"
            ),
        });
    }
    ctx.set_output(0, dtype, out);
    Ok(())
}

/// `Flatten`: collapse to `[prod(dims[..axis]), prod(dims[axis..])]`.
pub fn flatten(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(t) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    let rank = t.rank();
    let axis = ctx
        .node
        .attr("axis")
        .and_then(Attribute::as_int)
        .unwrap_or(1);
    let axis = if axis < 0 {
        (axis + rank as i64).max(0) as usize
    } else {
        (axis as usize).min(rank)
    };
    let outer = DimExpr::product(&t.shape[..axis]);
    let inner = DimExpr::product(&t.shape[axis..]);
    ctx.set_output(0, t.dtype, vec![outer, inner]);
    Ok(())
}

/// `Squeeze` with axes taken from an attribute (opset < 13).
pub fn squeeze_v1(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let axes = ctx
        .node
        .attr("axes")
        .and_then(Attribute::as_ints)
        .map(<[i64]>::to_vec);
    squeeze_common(ctx, axes)
}

/// `Squeeze` with axes taken from input 1 (opset ≥ 13).
pub fn squeeze_v13(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    // A runtime axes tensor can remove a variable number of dimensions, so its
    // output rank is not statically knowable.
    if ctx.has_input(1) && const_ints(ctx, 1).is_none() {
        return Ok(());
    }
    let axes = const_ints(ctx, 1);
    squeeze_common(ctx, axes)
}

fn squeeze_common(
    ctx: &mut InferenceContext,
    axes: Option<Vec<i64>>,
) -> Result<(), ShapeInferError> {
    let Some(t) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    let rank = t.rank();
    let out: Vec<DimExpr> = match axes {
        Some(axes) => {
            let mut norm = Vec::with_capacity(axes.len());
            for axis in axes {
                let axis = checked_axis(axis, rank).ok_or_else(|| ShapeInferError::Invalid {
                    op: ctx.op().into(),
                    detail: format!("axis {axis} is out of range for rank {rank}"),
                })?;
                if norm.contains(&axis) {
                    return Err(ShapeInferError::Invalid {
                        op: ctx.op().into(),
                        detail: format!("axis {axis} is specified more than once"),
                    });
                }
                norm.push(axis);
            }
            for &axis in &norm {
                match t.shape[axis].as_const() {
                    Some(1) => {}
                    Some(extent) => {
                        return Err(ShapeInferError::Invalid {
                            op: ctx.op().into(),
                            detail: format!(
                                "cannot squeeze axis {axis} with non-singleton extent {extent}"
                            ),
                        });
                    }
                    None => return Ok(()),
                }
            }
            t.shape
                .iter()
                .enumerate()
                .filter(|(i, _)| !norm.contains(i))
                .map(|(_, d)| d.clone())
                .collect()
        }
        // Without axes, a dynamic extent might be 1 at runtime, so no output
        // shape can be inferred.
        None => {
            if t.shape.iter().any(|d| d.as_const().is_none()) {
                return Ok(());
            }
            t.shape
                .iter()
                .filter(|d| d.as_const() != Some(1))
                .cloned()
                .collect()
        }
    };
    // Squeeze on a shape-data vector (drops nothing structurally for a 1-D
    // shape vector, but keep the data flowing for downstream ops).
    if let Some(sd) = ctx.input_shape_data(0).cloned() {
        ctx.set_output_shape_data(0, sd);
    }
    ctx.set_output(0, t.dtype, out);
    Ok(())
}

/// `Unsqueeze` with axes from an attribute (opset < 13).
pub fn unsqueeze_v1(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let axes = ctx
        .node
        .attr("axes")
        .and_then(Attribute::as_ints)
        .map(<[i64]>::to_vec);
    unsqueeze_common(ctx, axes)
}

/// `Unsqueeze` with axes from input 1 (opset ≥ 13).
pub fn unsqueeze_v13(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    validate_vector_input(ctx, 1, "Unsqueeze")?;
    let axes = const_ints(ctx, 1);
    if axes.is_none() {
        let Some(t) = ctx.input_type(0).cloned() else {
            return Ok(());
        };
        let Some(axis_count) = vector_length(ctx, 1) else {
            return Ok(());
        };
        let out_rank =
            t.rank()
                .checked_add(axis_count)
                .ok_or_else(|| ShapeInferError::Invalid {
                    op: "Unsqueeze".into(),
                    detail: format!(
                        "input rank {} plus {axis_count} axes exceeds the supported rank",
                        t.rank()
                    ),
                })?;
        let out = (0..out_rank).map(|_| ctx.fresh_dim()).collect();
        ctx.set_output(0, t.dtype, out);
        return Ok(());
    }
    unsqueeze_common(ctx, axes)
}

fn unsqueeze_common(
    ctx: &mut InferenceContext,
    axes: Option<Vec<i64>>,
) -> Result<(), ShapeInferError> {
    let Some(t) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    let Some(axes) = axes else {
        return Ok(());
    };
    let out_rank = t.rank() + axes.len();
    // ONNX Unsqueeze axes index positions in the *output* tensor (accepted
    // range `[-output_rank, output_rank-1]`), so normalize against `out_rank`,
    // not the input rank — otherwise a high axis (e.g. 3 into a rank-2 input)
    // is wrongly clamped and the size-1 dims land in the wrong slots.
    let mut norm = Vec::with_capacity(axes.len());
    for &axis in &axes {
        let axis = checked_axis(axis, out_rank).ok_or_else(|| ShapeInferError::Invalid {
            op: ctx.op().into(),
            detail: format!("axis {axis} is out of range for output rank {out_rank}"),
        })?;
        if norm.contains(&axis) {
            return Err(ShapeInferError::Invalid {
                op: ctx.op().into(),
                detail: format!("axis {axis} is specified more than once"),
            });
        }
        norm.push(axis);
    }
    let mut out = Vec::with_capacity(out_rank);
    let mut src = t.shape.iter();
    for i in 0..out_rank {
        if norm.contains(&i) {
            out.push(DimExpr::constant(1));
        } else if let Some(d) = src.next() {
            out.push(d.clone());
        }
    }
    // Shape-data: a scalar unsqueezed to a 1-vector keeps its value (common in
    // shape-computation chains that build a dim list).
    if let Some(sd) = ctx.input_shape_data(0).cloned()
        && sd.is_scalar()
        && norm == [0]
    {
        ctx.set_output_shape_data(0, ShapeData::vector(sd.dtype, sd.elems));
    }
    ctx.set_output(0, t.dtype, out);
    Ok(())
}

/// `Expand` (opset 8+): bidirectionally broadcast the input shape against the
/// values of the shape-tensor input.
pub fn expand(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(input) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    validate_vector_input(ctx, 1, "Expand")?;
    if let Some(target) = ctx.input_shape_data(1).map(ShapeData::as_shape) {
        for dim in &target {
            if let Some(value) = dim.as_const()
                && !(0..=isize::MAX as i64).contains(&value)
            {
                return Err(ShapeInferError::Invalid {
                    op: "Expand".into(),
                    detail: format!("target extent {value} is outside 0..=isize::MAX"),
                });
            }
        }
        let shape = bidirectional_broadcast(ctx, &input.shape, &target)?;
        ctx.set_output(0, input.dtype, shape);
    }
    Ok(())
}

fn bidirectional_broadcast(
    ctx: &mut InferenceContext,
    input: &[DimExpr],
    target: &[DimExpr],
) -> Result<Vec<DimExpr>, ShapeInferError> {
    let rank = input.len().max(target.len());
    let mut output = Vec::with_capacity(rank);
    for axis in 0..rank {
        let input_offset = rank - input.len();
        let target_offset = rank - target.len();
        let a = if axis < input_offset {
            DimExpr::constant(1)
        } else {
            input[axis - input_offset].clone()
        };
        let b = if axis < target_offset {
            DimExpr::constant(1)
        } else {
            target[axis - target_offset].clone()
        };
        if let (Some(a), Some(b)) = (a.as_const(), b.as_const())
            && a != 1
            && b != 1
            && a != b
        {
            return Err(ShapeInferError::Invalid {
                op: "Expand".into(),
                detail: format!("incompatible broadcast dims {a} and {b} at axis {axis}"),
            });
        }
        output.push(ctx.broadcast_dim(&a, &b)?);
    }
    Ok(output)
}

/// `Col2Im` (opset 18): rearrange column blocks back into a multi-dimensional
/// image — the inverse of an im2col unfold.
///
/// The data input has shape `[N, C * prod(block_shape), L]`. The `image_shape`
/// input (slot 1) gives the spatial extents of the reconstructed image and the
/// `block_shape` input (slot 2) gives the sliding-block extents. The output is
/// `[N, C, *image_shape]`, where `C = data.shape[1] / prod(block_shape)`.
/// Unknown operands degrade to fresh symbols rather than fabricated extents.
pub fn col2im(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(data) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    let dtype = ctx.input_dtype(0).unwrap_or(DataType::Float32);
    if data.len() != 3 {
        return Err(ShapeInferError::InvalidRank {
            op: "Col2Im".into(),
            index: 0,
            rank: data.len(),
            detail: "expected [N, C * prod(block_shape), L]".into(),
        });
    }
    validate_vector_input(ctx, 1, "Col2Im")?;
    validate_vector_input(ctx, 2, "Col2Im")?;

    // Channels: divide the folded second dimension by the block volume. When
    // the block shape is not statically known, the channel count is unknown.
    let channels = match const_ints(ctx, 2) {
        Some(blocks) => {
            let volume = blocks
                .iter()
                .copied()
                .try_fold(1i64, |acc, b| acc.checked_mul(b));
            match volume {
                Some(volume) if volume > 0 => data[1]
                    .checked_div(&DimExpr::constant(volume))
                    .unwrap_or_else(|| ctx.fresh_dim()),
                _ => ctx.fresh_dim(),
            }
        }
        None => ctx.fresh_dim(),
    };

    // The spatial extents are exactly the `image_shape` values; a symbolic
    // element stays symbolic. Without shape-data the spatial rank is unknown,
    // so the output is left unresolved.
    let Some(image) = ctx.input_shape_data(1).map(ShapeData::as_shape) else {
        return Ok(());
    };
    let mut out = vec![data[0].clone(), channels];
    for dim in image {
        if let Some(extent) = dim.as_const() {
            out.push(DimExpr::constant(checked_extent(
                "Col2Im",
                i128::from(extent),
            )?));
        } else {
            out.push(dim);
        }
    }
    ctx.set_output(0, dtype, out);
    Ok(())
}

/// `CenterCropPad` (opset 18): center-crop or center-pad `input_data` so that
/// the extents along `axes` match the `shape` input, leaving every other axis
/// untouched.
///
/// The output has the input's rank. Each targeted axis takes its extent from
/// the corresponding `shape` element (symbolic when the value is not statically
/// known); untargeted axes are copied through. When `axes` is omitted it
/// defaults to every axis of the input.
pub fn center_crop_pad(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(input) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    let dtype = ctx.input_dtype(0).unwrap_or(DataType::Float32);
    let rank = input.len();
    validate_vector_input(ctx, 1, "CenterCropPad")?;

    let axes: Vec<usize> = match ctx.node.attr("axes").and_then(Attribute::as_ints) {
        Some(raw) => raw
            .iter()
            .map(|&axis| {
                checked_axis(axis, rank).ok_or_else(|| ShapeInferError::Invalid {
                    op: "CenterCropPad".into(),
                    detail: format!("axis {axis} is out of range for rank {rank}"),
                })
            })
            .collect::<Result<_, _>>()?,
        None => (0..rank).collect(),
    };

    let mut out = input;
    // The target extents come from the `shape` input; when a value is not
    // statically known the corresponding axis degrades to a fresh symbol.
    let target = ctx.input_shape_data(1).map(ShapeData::as_shape);
    for (i, &axis) in axes.iter().enumerate() {
        let dim = match target.as_ref().and_then(|t| t.get(i)) {
            Some(dim) if dim.as_const().is_some() => {
                let extent = dim.as_const().unwrap();
                DimExpr::constant(checked_extent("CenterCropPad", i128::from(extent))?)
            }
            _ => ctx.fresh_dim(),
        };
        out[axis] = dim;
    }
    ctx.set_output(0, dtype, out);
    Ok(())
}

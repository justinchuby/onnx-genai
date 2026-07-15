//! Data-movement rules: `Reshape`, `Transpose`, `Flatten`, `Squeeze`,
//! `Unsqueeze`, `Expand`, `Concat`, `Slice`, `Split`, `Gather`,
//! `GatherElements`.
//!
//! Several of these are *shape-data consumers* (`Reshape`/`Expand`/`Slice` read
//! a computed shape vector) and/or *shape-data transformers* (`Gather`/`Slice`/
//! `Concat`/`Squeeze`/`Unsqueeze` on a shape vector), which is what keeps a
//! `Shape → … → Reshape` chain resolvable without executing the graph.

use onnx_runtime_ir::{Attribute, DataType};

use crate::context::InferenceContext;
use crate::dim_expr::DimExpr;
use crate::error::ShapeInferError;
use crate::handlers::norm_axis;
use crate::registry::InferenceRegistry;
use crate::shape_data::ShapeData;

/// Read a shape-data operand (input `i`) as concrete `i64`s, if every element is
/// a constant.
fn const_ints(ctx: &InferenceContext, i: usize) -> Option<Vec<i64>> {
    ctx.input_shape_data(i)?
        .elems
        .iter()
        .map(|e| e.as_const())
        .collect()
}

/// `Transpose`: permute dimensions by `perm` (default: reverse).
pub fn transpose(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(t) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    let rank = t.rank();
    let perm: Vec<usize> = match ctx.node.attr("perm").and_then(Attribute::as_ints) {
        Some(p) => p.iter().map(|&a| norm_axis(a, rank)).collect(),
        None => (0..rank).rev().collect(),
    };
    if perm.len() != rank {
        return Err(ShapeInferError::Invalid {
            op: "Transpose".into(),
            detail: format!("perm length {} != rank {rank}", perm.len()),
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
        // No resolved target: produce a fresh-symbol shape of the known rank.
        if let Some(rank) = target_rank(ctx) {
            let out = (0..rank).map(|_| ctx.fresh_dim()).collect();
            ctx.set_output(0, dtype, out);
        }
        return Ok(());
    };

    let total = DimExpr::product(&input);
    let mut out: Vec<DimExpr> = Vec::with_capacity(target.len());
    let mut product = DimExpr::constant(1);
    let mut neg1: Option<usize> = None;
    for (i, t) in target.iter().enumerate() {
        match t.as_const() {
            Some(-1) => {
                neg1 = Some(i);
                out.push(DimExpr::constant(1)); // placeholder, fixed below
            }
            Some(0) if !allowzero => {
                let d = input
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| DimExpr::constant(1));
                product = product.mul(&d);
                out.push(d);
            }
            _ => {
                product = product.mul(t);
                out.push(t.clone());
            }
        }
    }
    if let Some(idx) = neg1 {
        out[idx] = total
            .checked_div(&product)
            .unwrap_or_else(|| ctx.fresh_dim());
    }
    ctx.set_output(0, dtype, out);
    Ok(())
}

/// The rank of a `Reshape`/`Expand` target when its values are unknown but its
/// length is a concrete 1-D shape.
fn target_rank(ctx: &InferenceContext) -> Option<usize> {
    let s = ctx.input_shape(1)?;
    if s.len() == 1 {
        s[0].as_const().map(|n| n.max(0) as usize)
    } else {
        None
    }
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
            let norm: Vec<usize> = axes.iter().map(|&a| norm_axis(a, rank)).collect();
            t.shape
                .iter()
                .enumerate()
                .filter(|(i, _)| !norm.contains(i))
                .map(|(_, d)| d.clone())
                .collect()
        }
        // No axes: drop every statically-size-1 dim.
        None => t
            .shape
            .iter()
            .filter(|d| d.as_const() != Some(1))
            .cloned()
            .collect(),
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
    let axes = const_ints(ctx, 1);
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
    let norm: Vec<usize> = axes
        .iter()
        .map(|&a| {
            let a = if a < 0 { a + out_rank as i64 } else { a };
            a.clamp(0, out_rank as i64 - 1) as usize
        })
        .collect();
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

/// `Expand`: broadcast the input against the (shape-data) target shape.
pub fn expand(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(input) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    let dtype = ctx.input_dtype(0).unwrap_or(DataType::Float32);
    if let Some(target) = ctx.input_shape_data(1).map(ShapeData::as_shape) {
        let shape = ctx.broadcast(&input, &target)?;
        ctx.set_output(0, dtype, shape);
    } else if let Some(rank) = target_rank(ctx) {
        // Unknown target values: rank is max(input rank, target length), dims
        // are fresh symbols.
        let out_rank = rank.max(input.len());
        let out = (0..out_rank).map(|_| ctx.fresh_dim()).collect();
        ctx.set_output(0, dtype, out);
    }
    Ok(())
}

/// `Concat`: sum the concat axis across inputs; other dims from input 0.
pub fn concat(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(axis_attr) = ctx.node.attr("axis").and_then(Attribute::as_int) else {
        return Err(ShapeInferError::MissingAttribute {
            op: "Concat".into(),
            attr: "axis".into(),
        });
    };
    let present: Vec<usize> = (0..ctx.num_inputs())
        .filter(|&i| ctx.has_input(i))
        .collect();
    let Some(&first) = present.first() else {
        return Ok(());
    };
    let Some(base) = ctx.input_shape(first).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    let dtype = ctx.input_dtype(first).unwrap_or(DataType::Float32);
    let rank = base.len();
    let axis = norm_axis(axis_attr, rank);

    let mut out = base.clone();
    let mut sum = DimExpr::constant(0);
    let mut all_known = true;
    for &i in &present {
        match ctx.input_shape(i) {
            Some(s) if s.len() == rank => sum = sum.add(&s[axis]),
            _ => all_known = false,
        }
    }
    if all_known {
        out[axis] = sum;
    } else {
        out[axis] = ctx.fresh_dim();
    }
    ctx.set_output(0, dtype, out);

    // Shape-data: concatenation of shape vectors / scalars.
    if let Some(sd) = concat_shape_data(ctx, &present) {
        ctx.set_output_shape_data(0, sd);
    }
    Ok(())
}

/// Concatenate shape-data operands (scalars contribute one element each).
fn concat_shape_data(ctx: &InferenceContext, present: &[usize]) -> Option<ShapeData> {
    let mut elems = Vec::new();
    // Carry the operands' actual integer dtype rather than assuming Int64: a
    // shape-computation chain may run on Int32 dims.
    let mut dtype = DataType::Int64;
    for (k, &i) in present.iter().enumerate() {
        let sd = ctx.input_shape_data(i)?;
        if k == 0 {
            dtype = sd.dtype;
        }
        elems.extend(sd.elems.iter().cloned());
    }
    Some(ShapeData::vector(dtype, elems))
}

/// `Slice` (opset ≥ 10 input-driven, with an opset < 10 attribute fallback).
pub fn slice(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(data) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    let dtype = ctx.input_dtype(0).unwrap_or(DataType::Float32);
    let rank = data.len();

    let starts = const_ints(ctx, 1).or_else(|| attr_ints(ctx, "starts"));
    let ends = const_ints(ctx, 2).or_else(|| attr_ints(ctx, "ends"));
    let axes = const_ints(ctx, 3).or_else(|| attr_ints(ctx, "axes"));
    let steps = const_ints(ctx, 4).or_else(|| attr_ints(ctx, "steps"));

    let mut out = data.clone();
    match (starts, ends) {
        (Some(starts), Some(ends)) => {
            let axes: Vec<usize> = match axes {
                Some(a) => a.iter().map(|&x| norm_axis(x, rank)).collect(),
                None => (0..starts.len()).collect(),
            };
            for (k, &ax) in axes.iter().enumerate() {
                let step = steps.as_ref().and_then(|s| s.get(k)).copied().unwrap_or(1);
                out[ax] = slice_dim(
                    &data[ax],
                    starts.get(k).copied(),
                    ends.get(k).copied(),
                    step,
                )
                .unwrap_or_else(|| ctx.fresh_dim());
            }
        }
        // Bounds unknown (data-dependent): the sliced extents become fresh
        // symbols; other axes are untouched. We do not know which axes are
        // sliced, so conservatively refresh none unless axes are known.
        _ => {
            if let Some(axes) = axes {
                for ax in axes {
                    let ax = norm_axis(ax, rank);
                    out[ax] = ctx.fresh_dim();
                }
            } else {
                // Fully data-dependent: refresh all axes to keep a known rank.
                for d in out.iter_mut() {
                    *d = ctx.fresh_dim();
                }
            }
        }
    }
    ctx.set_output(0, dtype, out);

    // Shape-data: slicing a 1-D shape vector on axis 0 with concrete bounds.
    if let Some(sd) = slice_shape_data(ctx, rank) {
        ctx.set_output_shape_data(0, sd);
    }
    Ok(())
}

/// A concrete sliced extent, or `None` when any of the bounds/dim are symbolic.
fn slice_dim(dim: &DimExpr, start: Option<i64>, end: Option<i64>, step: i64) -> Option<DimExpr> {
    let d = dim.as_const()?;
    let (start, end) = (start?, end?);
    if step == 0 {
        return None;
    }
    let norm = |v: i64| -> i64 {
        let v = if v < 0 { v + d } else { v };
        v.clamp(0, d)
    };
    let len = if step > 0 {
        let s = norm(start);
        let e = norm(end);
        ((e - s).max(0) + step - 1) / step
    } else {
        // Negative step: clamp differently.
        let s = if start < 0 {
            (start + d).clamp(0, d - 1)
        } else {
            start.min(d - 1)
        };
        let e = if end < 0 {
            (end + d).clamp(-1, d - 1)
        } else {
            end.min(d)
        };
        ((s - e).max(0) + (-step) - 1) / (-step)
    };
    Some(DimExpr::constant(len.max(0)))
}

/// Slice a 1-D shape-data vector on axis 0 with concrete bounds.
fn slice_shape_data(ctx: &InferenceContext, _rank: usize) -> Option<ShapeData> {
    let sd = ctx.input_shape_data(0)?;
    if sd.is_scalar() {
        return None;
    }
    let starts = const_ints(ctx, 1).or_else(|| attr_ints(ctx, "starts"))?;
    let ends = const_ints(ctx, 2).or_else(|| attr_ints(ctx, "ends"))?;
    let axes = const_ints(ctx, 3).or_else(|| attr_ints(ctx, "axes"));
    // Only handle a single axis-0 slice of the vector.
    if let Some(ax) = &axes
        && ax.as_slice() != [0]
    {
        return None;
    }
    let steps = const_ints(ctx, 4).or_else(|| attr_ints(ctx, "steps"));
    let step = steps.as_ref().and_then(|s| s.first()).copied().unwrap_or(1);
    if step != 1 {
        return None;
    }
    let n = sd.elems.len() as i64;
    let norm = |v: i64| -> usize {
        let v = if v < 0 { v + n } else { v };
        v.clamp(0, n) as usize
    };
    let s = norm(*starts.first()?);
    let e = norm(*ends.first()?);
    let elems = sd.elems.get(s..e.max(s)).unwrap_or(&[]).to_vec();
    Some(ShapeData::vector(sd.dtype, elems))
}

/// `Split`: divide the input along `axis` into the requested sizes (or equally).
pub fn split(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(t) = ctx.input_type(0).cloned() else {
        return Ok(());
    };
    let rank = t.rank();
    let axis = ctx
        .node
        .attr("axis")
        .and_then(Attribute::as_int)
        .unwrap_or(0);
    let axis = norm_axis(axis, rank);
    let n_out = ctx.num_outputs();

    let sizes: Option<Vec<i64>> = ctx
        .node
        .attr("split")
        .and_then(Attribute::as_ints)
        .map(<[i64]>::to_vec)
        .or_else(|| const_ints(ctx, 1));

    for i in 0..n_out {
        let mut shape = t.shape.clone();
        shape[axis] = match &sizes {
            Some(s) => s
                .get(i)
                .map(|&v| DimExpr::constant(v))
                .unwrap_or_else(|| ctx.fresh_dim()),
            None => {
                // Equal split: only exact when the axis is a known multiple.
                match t.shape[axis].as_const() {
                    Some(d) if n_out > 0 && d % (n_out as i64) == 0 => {
                        DimExpr::constant(d / n_out as i64)
                    }
                    _ => ctx.fresh_dim(),
                }
            }
        };
        ctx.set_output(i, t.dtype, shape);
    }
    Ok(())
}

/// `Gather`: `data[:axis] + indices.shape + data[axis+1:]`.
pub fn gather(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(data) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    let dtype = ctx.input_dtype(0).unwrap_or(DataType::Float32);
    let rank = data.len();
    if rank == 0 {
        return Err(ShapeInferError::InvalidRank {
            op: "Gather".into(),
            index: 0,
            rank,
            detail: "Gather data must have rank ≥ 1".into(),
        });
    }
    let axis = ctx
        .node
        .attr("axis")
        .and_then(Attribute::as_int)
        .unwrap_or(0);
    let axis = norm_axis(axis, rank);
    let idx_shape = ctx
        .input_shape(1)
        .map(<[DimExpr]>::to_vec)
        .unwrap_or_default();

    let mut out = Vec::with_capacity(rank - 1 + idx_shape.len());
    out.extend_from_slice(&data[..axis]);
    out.extend(idx_shape.iter().cloned());
    out.extend_from_slice(&data[axis + 1..]);
    ctx.set_output(0, dtype, out);

    // Shape-data: gathering elements of a 1-D shape vector on axis 0.
    if axis == 0
        && let Some(sd) = gather_shape_data(ctx)
    {
        ctx.set_output_shape_data(0, sd);
    }
    Ok(())
}

/// Gather elements of a 1-D shape-data vector at concrete indices.
fn gather_shape_data(ctx: &InferenceContext) -> Option<ShapeData> {
    let sd = ctx.input_shape_data(0)?;
    if sd.is_scalar() {
        return None;
    }
    let idx = ctx.input_shape_data(1)?;
    let idx_ints: Vec<i64> = idx
        .elems
        .iter()
        .map(|e| e.as_const())
        .collect::<Option<_>>()?;
    let n = sd.elems.len() as i64;
    let pick = |i: i64| -> Option<DimExpr> {
        let i = if i < 0 { i + n } else { i };
        sd.elems.get(i as usize).cloned()
    };
    let elems: Vec<DimExpr> = idx_ints.iter().map(|&i| pick(i)).collect::<Option<_>>()?;
    if idx.is_scalar() {
        Some(ShapeData::scalar(sd.dtype, elems.into_iter().next()?))
    } else {
        Some(ShapeData::vector(sd.dtype, elems))
    }
}

/// `GatherElements`: the output shape follows the indices tensor; dtype of the
/// data. (Not `GatherND` — this is the elementwise gather whose output rank
/// equals the indices' rank.)
pub fn gather_elements(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let dtype = ctx.input_dtype(0);
    let idx_shape = ctx.input_shape(1).map(<[DimExpr]>::to_vec);
    if let (Some(dtype), Some(shape)) = (dtype, idx_shape) {
        ctx.set_output(0, dtype, shape);
    }
    Ok(())
}

/// `GatherND`: `data[:batch_dims] + indices[batch_dims:-1] +
/// data[batch_dims + indices[-1]:]`.
pub fn gather_nd(ctx: &mut InferenceContext) -> Result<(), ShapeInferError> {
    let Some(data) = ctx.input_shape(0).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    let Some(indices) = ctx.input_shape(1).map(<[DimExpr]>::to_vec) else {
        return Ok(());
    };
    let Some(dtype) = ctx.input_dtype(0) else {
        return Ok(());
    };
    let Some(index_depth) = indices.last().and_then(DimExpr::as_const) else {
        // The index-tuple depth determines the output rank. Without it, retain
        // the crate's unknown-rank representation (no TypeInfo).
        return Ok(());
    };
    let batch_dims = ctx
        .node
        .attr("batch_dims")
        .and_then(Attribute::as_int)
        .unwrap_or(0);
    if batch_dims < 0 {
        return Ok(());
    }
    let batch_dims = batch_dims as usize;
    let Ok(index_depth) = usize::try_from(index_depth) else {
        return Ok(());
    };
    if batch_dims > data.len()
        || batch_dims >= indices.len()
        || index_depth > data.len().saturating_sub(batch_dims)
    {
        return Ok(());
    }

    let mut out = Vec::with_capacity(data.len() + indices.len() - index_depth - 1);
    out.extend_from_slice(&data[..batch_dims]);
    out.extend(indices[batch_dims..indices.len() - 1].iter().cloned());
    out.extend_from_slice(&data[batch_dims + index_depth..]);
    ctx.set_output(0, dtype, out);
    Ok(())
}

/// Read an integer-list attribute.
fn attr_ints(ctx: &InferenceContext, name: &str) -> Option<Vec<i64>> {
    ctx.node
        .attr(name)
        .and_then(Attribute::as_ints)
        .map(<[i64]>::to_vec)
}

/// Register the data-movement family.
pub fn register(reg: &mut InferenceRegistry) {
    reg.register("", "Transpose", 1, transpose);
    reg.register("", "Reshape", 1, reshape);
    reg.register("", "Flatten", 1, flatten);
    // Squeeze/Unsqueeze moved axes from attribute to input at opset 13 — a real
    // range-based dispatch.
    reg.register("", "Squeeze", 1, squeeze_v1);
    reg.register("", "Squeeze", 13, squeeze_v13);
    reg.register("", "Unsqueeze", 1, unsqueeze_v1);
    reg.register("", "Unsqueeze", 13, unsqueeze_v13);
    reg.register("", "Expand", 1, expand);
    reg.register("", "Concat", 1, concat);
    reg.register("", "Slice", 1, slice);
    reg.register("", "Split", 1, split);
    reg.register("", "Gather", 1, gather);
    reg.register("", "GatherElements", 1, gather_elements);
    reg.register("", "GatherND", 11, gather_nd);
    reg.register("", "GatherND", 12, gather_nd);
    reg.register("", "GatherND", 13, gather_nd);
}

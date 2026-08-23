//! Constant folding: replace a node whose inputs are *all* constant
//! (initializers) with a precomputed initializer (see `docs/architecture/ORT2.md` §18.1).
//!
//! ## Boundary (deliberately conservative)
//!
//! Fully general constant folding needs a kernel executor — the optimizer has
//! none — so this pass folds only what the IR can compute *directly and
//! exactly*, reusing the lessons of the loader's "const-fold-lite"
//! (`crates/onnx-runtime-loader/src/shape_inference.rs`):
//!
//! * **`Constant`** nodes are materialized into initializers (always safe).
//! * **`Shape`** on a fully-static input becomes an `int64` initializer.
//! * **Elementwise integer `Add`/`Sub`/`Mul`** on two *same-shape* constant
//!   `int32`/`int64` tensors are evaluated with **checked** arithmetic; any
//!   overflow aborts the fold rather than emit a wrong constant.
//! * **`Concat`** of constant tensors along a static axis (used to assemble a
//!   `Reshape` shape from a literal prefix plus a folded `Shape` suffix).
//! * **`Reshape`/`Transpose`** of a constant tensor are pure data relayouts —
//!   no arithmetic, no precision loss, output size never exceeds input size —
//!   so they are folded **regardless of tensor size** (see
//!   [`MAX_WEIGHT_FOLD_ELEMS`]). Model builders emit these around quantized
//!   MoE expert weights to reorder HF's `gate_up_proj` layout into the
//!   interleaved layout `QMoE`'s CPU/CUDA kernels require (e.g. mobius's
//!   `_interleave_gate_up_rows`), relying on the runtime to fold them into a
//!   literal initializer at load time — exactly what stock ORT's own
//!   constant-folding does. Without this, downstream weight-placement
//!   analysis (which requires `QMoE`'s expert-weight inputs to be literal
//!   initializers) fails on a semantically-valid graph.
//!
//! Everything else is left untouched. `Constant`/`Shape`/`Add`/`Sub`/`Mul`/
//! `Concat` folding is bounded to [`MAX_FOLD_ELEMS`] elements — they exist
//! for shape computation, so a larger operand indicates something other than
//! a shape value. `Reshape`/`Transpose` use the much larger
//! [`MAX_WEIGHT_FOLD_ELEMS`] instead, since folding them is just a bounded
//! memcpy/permute with no combinatorial cost. Dispatch is purely on op type —
//! no model-specific names. The invariant is: **never produce a wrong
//! constant.** When in doubt, do not fold.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use onnx_runtime_ir::{
    Attribute, DataType, Graph, NodeId, TensorData, ValueId, WeightRef, as_static_shape,
    checked_numel, is_fully_static, read_vec_le, static_shape,
};

use crate::error::Result;
use crate::pass::{OptimizationPass, PassContext};

/// Upper bound on the number of elements this pass will materialize for
/// `Shape`/`Add`/`Sub`/`Mul`/`Concat`. Keeps folding limited to
/// shape-computation-sized tensors.
const MAX_FOLD_ELEMS: usize = 1024;

/// Upper bound on the number of elements `Reshape`/`Transpose` will
/// materialize. These ops never grow data (output size == input size) and
/// perform no arithmetic, so they are safe to fold at weight scale; this is
/// only a sanity ceiling against a corrupt/adversarial shape, not a
/// performance-motivated limit like [`MAX_FOLD_ELEMS`].
///
/// Correctness-wise this bound is deliberately generous. It has no *load-time
/// cost* budget attached: `"basic"` optimization (this pass plus dead-node
/// elimination) now runs unconditionally on the production native-decode load
/// path (`onnx-genai-engine::native_decode::{load,proposer}`), so a real
/// (non-tiny) model with a long `Reshape`/`Transpose` weight-relayout chain
/// over large expert tensors folds every time that model loads, not just
/// once. That is a legitimate follow-up (a byte/time budget, or scoping the
/// unconditional `"basic"` opt-in more narrowly) tracked separately — it does
/// not change the folds' correctness, which this pass still guarantees.
const MAX_WEIGHT_FOLD_ELEMS: usize = 1 << 30;

/// Folds constant-input nodes into initializers (bounded, integer/shape only).
#[derive(Clone, Copy, Debug, Default)]
pub struct ConstantFolding;

impl OptimizationPass for ConstantFolding {
    fn name(&self) -> &str {
        "ConstantFolding"
    }

    fn run(&self, graph: &mut Graph, _ctx: &PassContext) -> Result<()> {
        let candidates: Vec<NodeId> = graph
            .nodes
            .iter()
            .filter_map(|(nid, node)| is_candidate(node).then_some(nid))
            .collect();
        let mut unresolved = HashMap::with_capacity(candidates.len());
        let mut dependents: HashMap<ValueId, Vec<NodeId>> =
            HashMap::with_capacity(candidates.len());
        // Match ascending fixpoint passes: higher IDs made ready during a wave
        // join it, while lower/equal IDs wait for the next wave.
        let mut current_wave = BinaryHeap::new();
        let mut next_wave = BinaryHeap::new();

        for nid in candidates {
            let node = graph.node(nid);
            let Some(inputs) = unresolved_inputs(graph, node) else {
                continue;
            };
            let count = inputs.len();
            unresolved.insert(nid, count);
            if count == 0 {
                current_wave.push(Reverse(nid.0));
            } else {
                for input in inputs {
                    dependents.entry(input).or_default().push(nid);
                }
            }
        }

        while !current_wave.is_empty() {
            while let Some(Reverse(raw_nid)) = current_wave.pop() {
                let nid = NodeId(raw_nid);
                if unresolved.remove(&nid).is_none() || !graph.nodes.contains(nid) {
                    continue;
                }
                let (out, folded) = {
                    let node = graph.node(nid);
                    let folded = match node.op_type.as_str() {
                        "Constant" => eval_constant(node),
                        "Shape" => fold_shape(graph, node),
                        "Add" | "Sub" | "Mul" => fold_binary_int(graph, node),
                        "Concat" => fold_concat(graph, node),
                        "Reshape" => fold_reshape(graph, node),
                        "Transpose" => fold_transpose(graph, node),
                        _ => None,
                    };
                    (node.outputs[0], folded)
                };
                let Some(tensor) = folded else { continue };

                // Only fold outputs that are still needed (have a consumer or
                // are graph outputs); dead outputs are DCE's job and folding
                // them would leave a stale initializer referencing a GC'd id.
                let needed = graph.outputs.contains(&out)
                    || graph.try_value(out).is_some_and(|_| graph.has_uses(out));
                if !needed {
                    continue;
                }

                graph.remove_node(nid);
                // The output survives because it is needed; retype it to the
                // folded tensor and back it with an inline initializer.
                if graph.try_value(out).is_none() {
                    continue;
                }
                let dims = tensor.dims.clone();
                let dtype = tensor.dtype;
                let v = graph.value_mut(out);
                v.dtype = dtype;
                v.shape = static_shape(dims);
                graph.set_initializer(out, WeightRef::Inline(tensor));

                for consumer in dependents.remove(&out).unwrap_or_default() {
                    let Some(count) = unresolved.get_mut(&consumer) else {
                        continue;
                    };
                    *count -= 1;
                    if *count == 0 {
                        let wave = if consumer.0 > nid.0 {
                            &mut current_wave
                        } else {
                            &mut next_wave
                        };
                        wave.push(Reverse(consumer.0));
                    }
                }
            }
            std::mem::swap(&mut current_wave, &mut next_wave);
        }
        Ok(())
    }
}

fn is_candidate(node: &onnx_runtime_ir::Node) -> bool {
    matches!(node.domain.as_str(), "" | "ai.onnx")
        && node.outputs.len() == 1
        && matches!(
            node.op_type.as_str(),
            "Constant" | "Shape" | "Add" | "Sub" | "Mul" | "Concat" | "Reshape" | "Transpose"
        )
}

fn unresolved_inputs(graph: &Graph, node: &onnx_runtime_ir::Node) -> Option<Vec<ValueId>> {
    match node.op_type.as_str() {
        "Constant" => Some(Vec::new()),
        "Shape" => {
            let input = node.inputs.first().copied().flatten()?;
            let shape = &graph.try_value(input)?.shape;
            if shape.len() <= MAX_FOLD_ELEMS && is_fully_static(shape) {
                Some(Vec::new())
            } else {
                Some(vec![input])
            }
        }
        "Add" | "Sub" | "Mul" => {
            if node.inputs.len() != 2 {
                return None;
            }
            let inputs = [node.inputs[0]?, node.inputs[1]?];
            Some(
                inputs
                    .into_iter()
                    .filter(|&input| inline_const(graph, input).is_none())
                    .collect(),
            )
        }
        "Concat" => {
            if node.inputs.is_empty() {
                return None;
            }
            let inputs: Vec<ValueId> = node.inputs.iter().copied().collect::<Option<_>>()?;
            Some(
                inputs
                    .into_iter()
                    .filter(|&input| inline_const(graph, input).is_none())
                    .collect(),
            )
        }
        "Reshape" => {
            if node.attr("allowzero").and_then(Attribute::as_int) == Some(1) {
                return None; // rare `allowzero=1` semantics: bail conservatively
            }
            let data = node.inputs.first().copied().flatten()?;
            let shape = node.inputs.get(1).copied().flatten()?;
            Some(
                [data, shape]
                    .into_iter()
                    .filter(|&input| inline_const(graph, input).is_none())
                    .collect(),
            )
        }
        "Transpose" => {
            let data = node.inputs.first().copied().flatten()?;
            Some(
                [data]
                    .into_iter()
                    .filter(|&input| inline_const(graph, input).is_none())
                    .collect(),
            )
        }
        _ => None,
    }
}

/// The inline constant tensor backing `value`, if any (external weights, which
/// are large, are never folded).
fn inline_const(graph: &Graph, value: ValueId) -> Option<&TensorData> {
    match graph.initializers.get(&value)? {
        WeightRef::Inline(t) => Some(t),
        WeightRef::External { .. } => None,
    }
}

/// Materialize a `Constant` node's value into a concrete [`TensorData`].
fn eval_constant(node: &onnx_runtime_ir::Node) -> Option<TensorData> {
    if let Some(Attribute::Tensor(t)) = node.attr("value") {
        return Some(t.clone());
    }
    if let Some(ints) = node.attr("value_ints").and_then(Attribute::as_ints) {
        let mut data = Vec::with_capacity(ints.len() * 8);
        for &i in ints {
            data.extend_from_slice(&i.to_le_bytes());
        }
        return Some(TensorData::from_raw(
            DataType::Int64,
            vec![ints.len()],
            data,
        ));
    }
    if let Some(i) = node.attr("value_int").and_then(Attribute::as_int) {
        return Some(TensorData::from_raw(
            DataType::Int64,
            Vec::new(),
            i.to_le_bytes().to_vec(),
        ));
    }
    None
}

/// Fold `Shape(x)` when `x` has a fully-static shape into an `int64` vector,
/// honoring the optional `start`/`end` slice attributes (Python-style
/// slicing semantics: negative indices count from the end, out-of-range
/// values clamp).
fn fold_shape(graph: &Graph, node: &onnx_runtime_ir::Node) -> Option<TensorData> {
    let input = node.inputs.first().copied().flatten()?;
    let shape = &graph.try_value(input)?.shape;
    let dims = as_static_shape(shape)?;
    if dims.len() > MAX_FOLD_ELEMS {
        return None;
    }
    let rank = dims.len() as i64;
    let clamp = |v: i64| -> usize { v.clamp(0, rank) as usize };
    let start = node
        .attr("start")
        .and_then(Attribute::as_int)
        .map_or(0, |v| clamp(if v < 0 { v + rank } else { v }));
    let end = node
        .attr("end")
        .and_then(Attribute::as_int)
        .map_or(dims.len(), |v| clamp(if v < 0 { v + rank } else { v }));
    let sliced = if start < end { &dims[start..end] } else { &[] };
    let mut data = Vec::with_capacity(sliced.len() * 8);
    for &d in sliced {
        data.extend_from_slice(&(d as i64).to_le_bytes());
    }
    Some(TensorData::from_raw(
        DataType::Int64,
        vec![sliced.len()],
        data,
    ))
}

/// Fold elementwise integer `Add`/`Sub`/`Mul` on two same-shape constant
/// tensors. Uses checked arithmetic; overflow aborts (returns `None`).
fn fold_binary_int(graph: &Graph, node: &onnx_runtime_ir::Node) -> Option<TensorData> {
    if node.inputs.len() != 2 {
        return None;
    }
    let a = inline_const(graph, node.inputs[0]?)?;
    let b = inline_const(graph, node.inputs[1]?)?;
    if a.dtype != b.dtype || a.dims != b.dims {
        return None; // no broadcasting / mixed dtype in v1
    }
    if !matches!(a.dtype, DataType::Int32 | DataType::Int64) {
        return None;
    }
    let numel = a.numel();
    if numel > MAX_FOLD_ELEMS {
        return None;
    }
    let op = node.op_type.as_str();
    let apply = |x: i64, y: i64| -> Option<i64> {
        match op {
            "Add" => x.checked_add(y),
            "Sub" => x.checked_sub(y),
            "Mul" => x.checked_mul(y),
            _ => None,
        }
    };

    match a.dtype {
        DataType::Int64 => {
            let (xs, ys) = (read_i64(a)?, read_i64(b)?);
            let mut data = Vec::with_capacity(numel * 8);
            for (x, y) in xs.into_iter().zip(ys) {
                data.extend_from_slice(&apply(x, y)?.to_le_bytes());
            }
            Some(TensorData::from_raw(DataType::Int64, a.dims.clone(), data))
        }
        DataType::Int32 => {
            let (xs, ys) = (read_i32(a)?, read_i32(b)?);
            let mut data = Vec::with_capacity(numel * 4);
            for (x, y) in xs.into_iter().zip(ys) {
                let r = apply(x as i64, y as i64)?;
                let r32: i32 = r.try_into().ok()?; // must fit back into i32
                data.extend_from_slice(&r32.to_le_bytes());
            }
            Some(TensorData::from_raw(DataType::Int32, a.dims.clone(), data))
        }
        _ => None,
    }
}

/// Fold `Concat` of same-dtype constant tensors along a static axis.
///
/// Bounded to [`MAX_FOLD_ELEMS`] like the other shape-value folds above —
/// `Concat` here exists only to assemble a `Reshape` shape from a literal
/// prefix plus a folded `Shape` suffix, never to concatenate model weights.
fn fold_concat(graph: &Graph, node: &onnx_runtime_ir::Node) -> Option<TensorData> {
    let axis_attr = node.attr("axis").and_then(Attribute::as_int)?;
    let inputs: Vec<&TensorData> = node
        .inputs
        .iter()
        .map(|slot| inline_const(graph, (*slot)?))
        .collect::<Option<_>>()?;
    let first = *inputs.first()?;
    if first.dtype == DataType::String || !first.strings.is_empty() {
        return None;
    }
    let elem_size = first.dtype.byte_size();
    if elem_size == 0 {
        return None; // sub-byte packed types unsupported here
    }
    let rank = first.dims.len();
    if rank == 0 {
        return None;
    }
    let axis = normalize_axis(axis_attr, rank)?;
    let mut axis_sum = 0usize;
    for t in &inputs {
        if t.dtype != first.dtype || t.dims.len() != rank {
            return None;
        }
        for (i, (&a, &b)) in t.dims.iter().zip(first.dims.iter()).enumerate() {
            if i != axis && a != b {
                return None;
            }
        }
        axis_sum = axis_sum.checked_add(t.dims[axis])?;
    }
    let mut out_dims = first.dims.clone();
    out_dims[axis] = axis_sum;
    let numel = checked_numel(&out_dims)?;
    if numel > MAX_FOLD_ELEMS {
        return None;
    }
    let outer: usize = out_dims[..axis].iter().product();
    let inner: usize = out_dims[axis + 1..].iter().product();
    let mut out_bytes = vec![0u8; numel.checked_mul(elem_size)?];
    let mut dst = 0usize;
    for o in 0..outer {
        for t in &inputs {
            let slab_len = t.dims[axis].checked_mul(inner)?.checked_mul(elem_size)?;
            let src_off = o.checked_mul(slab_len)?;
            let src = t.data.get(src_off..src_off + slab_len)?;
            out_bytes[dst..dst + slab_len].copy_from_slice(src);
            dst += slab_len;
        }
    }
    Some(TensorData::from_raw(first.dtype, out_dims, out_bytes))
}

fn normalize_axis(axis: i64, rank: usize) -> Option<usize> {
    let r = rank as i64;
    let a = if axis < 0 { axis + r } else { axis };
    (0..r).contains(&a).then_some(a as usize)
}

/// Fold `Reshape` of a constant tensor. A pure metadata change: the raw
/// bytes are copied byte-for-byte (no element reordering), so this is safe
/// for any dtype (including sub-byte packed ones) and any size up to
/// [`MAX_WEIGHT_FOLD_ELEMS`].
fn fold_reshape(graph: &Graph, node: &onnx_runtime_ir::Node) -> Option<TensorData> {
    let data_id = node.inputs.first().copied().flatten()?;
    let shape_id = node.inputs.get(1).copied().flatten()?;
    let data = inline_const(graph, data_id)?;
    let shape_tensor = inline_const(graph, shape_id)?;
    if shape_tensor.dtype != DataType::Int64 {
        return None;
    }
    if data.dtype == DataType::String || !data.strings.is_empty() {
        return None;
    }
    let numel = data.checked_numel()?;
    if numel > MAX_WEIGHT_FOLD_ELEMS {
        return None;
    }
    let shape_vals = read_i64(shape_tensor)?;
    let resolved = resolve_reshape_dims(&data.dims, &shape_vals)?;
    if checked_numel(&resolved)? != numel {
        return None;
    }
    Some(TensorData::from_raw(
        data.dtype,
        resolved,
        data.data.clone(),
    ))
}

/// Resolve ONNX `Reshape` target dims: `0` copies the input dim at that
/// position, at most one `-1` is inferred from the remaining element count,
/// anything else is taken literally.
fn resolve_reshape_dims(input_dims: &[usize], shape_vals: &[i64]) -> Option<Vec<usize>> {
    let mut resolved = Vec::with_capacity(shape_vals.len());
    let mut infer_at: Option<usize> = None;
    for (i, &v) in shape_vals.iter().enumerate() {
        if v == -1 {
            if infer_at.is_some() {
                return None; // at most one -1
            }
            infer_at = Some(i);
            resolved.push(0); // placeholder, filled in below
        } else if v == 0 {
            resolved.push(*input_dims.get(i)?);
        } else {
            resolved.push(usize::try_from(v).ok()?);
        }
    }
    if let Some(i) = infer_at {
        let known = resolved
            .iter()
            .enumerate()
            .filter(|&(idx, _)| idx != i)
            .try_fold(1usize, |acc, (_, &d)| acc.checked_mul(d))?;
        if known == 0 {
            return None; // ambiguous / would divide by zero
        }
        let total = checked_numel(input_dims)?;
        if total % known != 0 {
            return None;
        }
        resolved[i] = total / known;
    }
    Some(resolved)
}

/// Fold `Transpose` of a constant tensor by physically permuting its raw
/// bytes according to `perm` (or the default reversed-axis order). Bounded
/// to [`MAX_WEIGHT_FOLD_ELEMS`]; sub-byte packed dtypes are rejected since
/// permuting axes could split a packed byte across output elements.
fn fold_transpose(graph: &Graph, node: &onnx_runtime_ir::Node) -> Option<TensorData> {
    let input = node.inputs.first().copied().flatten()?;
    let data = inline_const(graph, input)?;
    if data.dtype == DataType::String || !data.strings.is_empty() {
        return None;
    }
    let elem_size = data.dtype.byte_size();
    if elem_size == 0 {
        return None;
    }
    let rank = data.dims.len();
    let perm: Vec<usize> = match node.attr("perm").and_then(Attribute::as_ints) {
        Some(ints) => {
            if ints.len() != rank {
                return None;
            }
            let mut p = Vec::with_capacity(rank);
            for &v in ints {
                let v = usize::try_from(v).ok()?;
                if v >= rank {
                    return None;
                }
                p.push(v);
            }
            p
        }
        None => (0..rank).rev().collect(),
    };
    let mut seen = vec![false; rank];
    for &p in &perm {
        if std::mem::replace(&mut seen[p], true) {
            return None; // not a permutation
        }
    }
    let numel = data.checked_numel()?;
    if numel > MAX_WEIGHT_FOLD_ELEMS {
        return None;
    }
    if data.data.len() != numel.checked_mul(elem_size)? {
        return None; // malformed tensor; be conservative
    }
    let out_dims: Vec<usize> = perm.iter().map(|&p| data.dims[p]).collect();
    let mut in_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        in_strides[i] = in_strides[i + 1].checked_mul(data.dims[i + 1])?;
    }
    let mut out_strides = vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1].checked_mul(out_dims[i + 1])?;
    }
    let mut out_bytes = vec![0u8; numel.checked_mul(elem_size)?];
    let mut idx = vec![0usize; rank];
    for o in 0..numel {
        let mut rem = o;
        for (k, stride) in out_strides.iter().enumerate() {
            idx[k] = rem / stride;
            rem %= stride;
        }
        let mut in_flat = 0usize;
        for (k, &p) in perm.iter().enumerate() {
            in_flat += idx[k] * in_strides[p];
        }
        let src = in_flat * elem_size;
        let dst = o * elem_size;
        out_bytes[dst..dst + elem_size].copy_from_slice(&data.data[src..src + elem_size]);
    }
    Some(TensorData::from_raw(data.dtype, out_dims, out_bytes))
}

fn read_i64(t: &TensorData) -> Option<Vec<i64>> {
    if t.data.len() != t.numel() * 8 {
        return None;
    }
    read_vec_le(&t.data).ok()
}

fn read_i32(t: &TensorData) -> Option<Vec<i32>> {
    if t.data.len() != t.numel() * 4 {
        return None;
    }
    read_vec_le(&t.data).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DeadNodeElimination;
    use onnx_runtime_ir::{Node, NodeId};
    use onnx_runtime_loader::{Model, encode_model};

    fn int64_tensor(dims: Vec<usize>, vals: &[i64]) -> TensorData {
        let mut data = Vec::new();
        for &v in vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        TensorData::from_raw(DataType::Int64, dims, data)
    }

    fn const_init(graph: &mut Graph, name: &str, dims: Vec<usize>, vals: &[i64]) -> ValueId {
        let shape = static_shape(dims.clone());
        let v = graph.create_named_value(name, DataType::Int64, shape);
        graph.set_initializer(v, WeightRef::Inline(int64_tensor(dims, vals)));
        v
    }

    fn run_reference_ascending_fixpoint(graph: &mut Graph) {
        loop {
            let mut changed = false;
            let node_ids: Vec<NodeId> = graph.nodes.keys().collect();
            for nid in node_ids {
                if !graph.nodes.contains(nid) {
                    continue;
                }
                let node = graph.node(nid).clone();
                if !matches!(node.domain.as_str(), "" | "ai.onnx") || node.outputs.len() != 1 {
                    continue;
                }
                let out = node.outputs[0];
                let folded = match node.op_type.as_str() {
                    "Constant" => eval_constant(&node),
                    "Shape" => fold_shape(graph, &node),
                    "Add" | "Sub" | "Mul" => fold_binary_int(graph, &node),
                    "Concat" => fold_concat(graph, &node),
                    "Reshape" => fold_reshape(graph, &node),
                    "Transpose" => fold_transpose(graph, &node),
                    _ => None,
                };
                let Some(tensor) = folded else { continue };
                let needed = graph.outputs.contains(&out)
                    || graph.try_value(out).is_some_and(|_| graph.has_uses(out));
                if !needed {
                    continue;
                }

                graph.remove_node(nid);
                if graph.try_value(out).is_some() {
                    let dims = tensor.dims.clone();
                    let dtype = tensor.dtype;
                    let v = graph.value_mut(out);
                    v.dtype = dtype;
                    v.shape = static_shape(dims);
                    graph.set_initializer(out, WeightRef::Inline(tensor));
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
    }

    fn serialized(graph: &Graph) -> Vec<u8> {
        encode_model(&Model::new(graph)).expect("serialize graph")
    }

    fn schedule_sensitive_chain() -> (Graph, ValueId) {
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let init = const_init(&mut g, "init", vec![1], &[2]);
        let a = g.create_named_value("a", DataType::Int64, static_shape([1]));
        let b = g.create_named_value("b", DataType::Int64, static_shape([1]));
        let out = g.create_named_value("out", DataType::Int64, static_shape([1]));

        let mut constant = Node::new(NodeId(0), "Constant", vec![], vec![a]);
        constant.attributes.insert(
            "value".into(),
            Attribute::Tensor(int64_tensor(vec![1], &[3])),
        );
        g.insert_node(constant);
        g.insert_node(Node::new(
            NodeId(0),
            "Add",
            vec![Some(a), Some(init)],
            vec![b],
        ));
        g.insert_node(Node::new(NodeId(0), "Shape", vec![Some(b)], vec![out]));
        g.add_output(out);
        (g, out)
    }

    #[test]
    fn ascending_wave_folds_constant_add_before_shape_consumer() {
        let (base, out) = schedule_sensitive_chain();
        let mut reference = base.clone();
        let mut worklist = base;
        run_reference_ascending_fixpoint(&mut reference);
        ConstantFolding
            .run(&mut worklist, &PassContext::new())
            .unwrap();

        assert_eq!(worklist.num_nodes(), 0, "Constant, Add, and Shape fold");
        assert!(!worklist.nodes.values().any(|node| node.op_type == "Add"));
        assert_eq!(
            read_i64(inline_const(&worklist, out).unwrap()),
            Some(vec![1])
        );
        assert_eq!(serialized(&worklist), serialized(&reference));
        assert!(worklist.validate().is_ok());
    }

    #[test]
    fn ascending_wave_leaves_lower_dead_producer_unfolded() {
        let mut base = Graph::new();
        base.opset_imports.insert(String::new(), 17);
        let init = const_init(&mut base, "init", vec![1], &[2]);
        let a = base.create_named_value("a", DataType::Int64, static_shape([1]));
        let b = base.create_named_value("b", DataType::Int64, static_shape([1]));
        let out = base.create_named_value("out", DataType::Int64, static_shape([1]));

        base.insert_node(Node::new(
            NodeId(0),
            "Add",
            vec![Some(a), Some(init)],
            vec![b],
        ));
        base.insert_node(Node::new(NodeId(0), "Shape", vec![Some(b)], vec![out]));
        let mut constant = Node::new(NodeId(0), "Constant", vec![], vec![a]);
        constant.attributes.insert(
            "value".into(),
            Attribute::Tensor(int64_tensor(vec![1], &[3])),
        );
        base.insert_node(constant);
        base.add_output(out);

        let mut reference = base.clone();
        let mut worklist = base;
        run_reference_ascending_fixpoint(&mut reference);
        ConstantFolding
            .run(&mut worklist, &PassContext::new())
            .unwrap();

        assert_eq!(worklist.num_nodes(), 1);
        assert_eq!(worklist.nodes.values().next().unwrap().op_type, "Add");
        assert!(inline_const(&worklist, b).is_none());
        assert_eq!(serialized(&worklist), serialized(&reference));
        assert!(worklist.validate().is_ok());
    }

    fn seeded_dag(mut seed: u64, nodes: usize) -> Graph {
        fn next(seed: &mut u64) -> u64 {
            *seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            *seed
        }

        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let init = const_init(&mut g, "init", vec![1], &[1]);
        let values: Vec<ValueId> = (0..nodes)
            .map(|i| g.create_named_value(format!("v{i}"), DataType::Int64, static_shape([1])))
            .collect();
        let mut definitions = Vec::with_capacity(nodes);
        for i in 0..nodes {
            let mut node = if i == 0 || next(&mut seed).is_multiple_of(4) {
                let mut constant = Node::new(NodeId(0), "Constant", vec![], vec![values[i]]);
                constant.attributes.insert(
                    "value".into(),
                    Attribute::Tensor(int64_tensor(vec![1], &[(next(&mut seed) % 8) as i64])),
                );
                constant
            } else if next(&mut seed).is_multiple_of(3) {
                let input = values[(next(&mut seed) as usize) % i];
                Node::new(NodeId(0), "Shape", vec![Some(input)], vec![values[i]])
            } else {
                let pick_input = |seed: &mut u64| {
                    if next(seed).is_multiple_of(4) {
                        init
                    } else {
                        values[(next(seed) as usize) % i]
                    }
                };
                Node::new(
                    NodeId(0),
                    "Add",
                    vec![Some(pick_input(&mut seed)), Some(pick_input(&mut seed))],
                    vec![values[i]],
                )
            };
            node.name = format!("node_{i}");
            definitions.push(node);
        }

        let mut order: Vec<usize> = (0..nodes).collect();
        for i in (1..nodes).rev() {
            order.swap(i, (next(&mut seed) as usize) % (i + 1));
        }
        for index in order {
            g.insert_node(definitions[index].clone());
        }
        for (i, &value) in values.iter().enumerate() {
            if i + 1 == nodes || (i > nodes / 2 && next(&mut seed).is_multiple_of(11)) {
                g.add_output(value);
            }
        }
        g
    }

    #[test]
    fn seeded_dags_are_byte_identical_to_ascending_fixpoint() {
        for seed in 0..32 {
            let base = seeded_dag(seed, 96);
            assert!(base.validate().is_ok(), "seed {seed}");
            let mut reference = base.clone();
            let mut worklist = base;
            run_reference_ascending_fixpoint(&mut reference);
            ConstantFolding
                .run(&mut worklist, &PassContext::new())
                .unwrap();
            assert_eq!(serialized(&worklist), serialized(&reference), "seed {seed}");
        }
    }

    #[test]
    fn folds_add_of_two_const_inputs() {
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let a = const_init(&mut g, "a", vec![3], &[1, 2, 3]);
        let b = const_init(&mut g, "b", vec![3], &[10, 20, 30]);
        let out = g.create_named_value("out", DataType::Int64, static_shape([3]));
        g.insert_node(Node::new(
            NodeId(0),
            "Add",
            vec![Some(a), Some(b)],
            vec![out],
        ));
        g.add_output(out);

        ConstantFolding.run(&mut g, &PassContext::new()).unwrap();

        assert_eq!(g.num_nodes(), 0, "Add should be folded away");
        let t = inline_const(&g, out).expect("out is now an initializer");
        assert_eq!(read_i64(t).unwrap(), vec![11, 22, 33]);
        assert!(g.validate().is_ok());
    }

    #[test]
    fn folds_sub_and_mul() {
        for (op, expect) in [("Sub", vec![9, 18, 27]), ("Mul", vec![10, 40, 90])] {
            let mut g = Graph::new();
            g.opset_imports.insert(String::new(), 17);
            let a = const_init(&mut g, "a", vec![3], &[10, 20, 30]);
            let b = const_init(&mut g, "b", vec![3], &[1, 2, 3]);
            let out = g.create_named_value("out", DataType::Int64, static_shape([3]));
            g.insert_node(Node::new(NodeId(0), op, vec![Some(a), Some(b)], vec![out]));
            g.add_output(out);

            ConstantFolding.run(&mut g, &PassContext::new()).unwrap();
            let t = inline_const(&g, out).unwrap();
            assert_eq!(read_i64(t).unwrap(), expect, "op {op}");
        }
    }

    #[test]
    fn does_not_fold_when_one_input_is_non_const() {
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let a = const_init(&mut g, "a", vec![3], &[1, 2, 3]);
        // `b` is a graph input, not a constant.
        let b = g.create_named_value("b", DataType::Int64, static_shape([3]));
        g.add_input(b);
        let out = g.create_named_value("out", DataType::Int64, static_shape([3]));
        g.insert_node(Node::new(
            NodeId(0),
            "Add",
            vec![Some(a), Some(b)],
            vec![out],
        ));
        g.add_output(out);

        ConstantFolding.run(&mut g, &PassContext::new()).unwrap();
        assert_eq!(g.num_nodes(), 1, "must not fold with a non-const input");
        assert!(inline_const(&g, out).is_none());
        assert!(g.validate().is_ok());
    }

    #[test]
    fn does_not_fold_mismatched_shapes() {
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let a = const_init(&mut g, "a", vec![3], &[1, 2, 3]);
        let b = const_init(&mut g, "b", vec![2], &[10, 20]);
        let out = g.create_named_value("out", DataType::Int64, static_shape([3]));
        g.insert_node(Node::new(
            NodeId(0),
            "Add",
            vec![Some(a), Some(b)],
            vec![out],
        ));
        g.add_output(out);

        ConstantFolding.run(&mut g, &PassContext::new()).unwrap();
        assert_eq!(g.num_nodes(), 1, "no broadcasting in v1");
    }

    #[test]
    fn does_not_fold_overflow() {
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let a = const_init(&mut g, "a", vec![1], &[i64::MAX]);
        let b = const_init(&mut g, "b", vec![1], &[1]);
        let out = g.create_named_value("out", DataType::Int64, static_shape([1]));
        g.insert_node(Node::new(
            NodeId(0),
            "Add",
            vec![Some(a), Some(b)],
            vec![out],
        ));
        g.add_output(out);

        ConstantFolding.run(&mut g, &PassContext::new()).unwrap();
        assert_eq!(g.num_nodes(), 1, "overflow must abort the fold");
    }

    #[test]
    fn folds_constant_node_to_initializer() {
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let out = g.create_named_value("c", DataType::Int64, static_shape([2]));
        let mut node = Node::new(NodeId(0), "Constant", vec![], vec![out]);
        node.attributes.insert(
            "value".into(),
            Attribute::Tensor(int64_tensor(vec![2], &[7, 8])),
        );
        g.insert_node(node);
        // Keep `out` alive with a consumer.
        let sink = g.create_named_value("sink", DataType::Int64, static_shape([2]));
        g.insert_node(Node::new(
            NodeId(0),
            "Identity",
            vec![Some(out)],
            vec![sink],
        ));
        g.add_output(sink);

        ConstantFolding.run(&mut g, &PassContext::new()).unwrap();
        assert!(g.try_node(NodeId(0)).is_none(), "Constant folded away");
        let t = inline_const(&g, out).unwrap();
        assert_eq!(read_i64(t).unwrap(), vec![7, 8]);
        assert!(g.validate().is_ok());
    }

    #[test]
    fn folds_shape_of_static_input() {
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let x = g.create_named_value("x", DataType::Float32, static_shape([2, 3, 4]));
        g.add_input(x);
        let out = g.create_named_value("s", DataType::Int64, static_shape([3]));
        g.insert_node(Node::new(NodeId(0), "Shape", vec![Some(x)], vec![out]));
        g.add_output(out);

        ConstantFolding.run(&mut g, &PassContext::new()).unwrap();
        let t = inline_const(&g, out).expect("Shape folded to initializer");
        assert_eq!(read_i64(t).unwrap(), vec![2, 3, 4]);
        assert!(g.validate().is_ok());
    }

    #[test]
    fn folds_transitively_to_fixpoint() {
        // Constant c1, Constant c2, then Add(c1, c2) -> out. All should fold.
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);

        let c1 = g.create_named_value("c1", DataType::Int64, static_shape([2]));
        let mut n1 = Node::new(NodeId(0), "Constant", vec![], vec![c1]);
        n1.attributes.insert(
            "value".into(),
            Attribute::Tensor(int64_tensor(vec![2], &[1, 2])),
        );
        g.insert_node(n1);

        let c2 = g.create_named_value("c2", DataType::Int64, static_shape([2]));
        let mut n2 = Node::new(NodeId(0), "Constant", vec![], vec![c2]);
        n2.attributes.insert(
            "value".into(),
            Attribute::Tensor(int64_tensor(vec![2], &[3, 4])),
        );
        g.insert_node(n2);

        let out = g.create_named_value("out", DataType::Int64, static_shape([2]));
        g.insert_node(Node::new(
            NodeId(0),
            "Add",
            vec![Some(c1), Some(c2)],
            vec![out],
        ));
        g.add_output(out);

        ConstantFolding.run(&mut g, &PassContext::new()).unwrap();
        assert_eq!(g.num_nodes(), 0, "both constants and the Add fold away");
        let t = inline_const(&g, out).unwrap();
        assert_eq!(read_i64(t).unwrap(), vec![4, 6]);
        assert!(g.validate().is_ok());
    }

    fn constant_chain(nodes: usize, reverse_node_ids: bool) -> (Graph, ValueId) {
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let zero = const_init(&mut g, "zero", vec![1], &[0]);
        let one = const_init(&mut g, "one", vec![1], &[1]);
        let mut values = Vec::with_capacity(nodes + 1);
        values.push(zero);
        for _ in 0..nodes {
            values.push(g.create_value(DataType::Int64, static_shape([1])));
        }

        if reverse_node_ids {
            for i in (1..=nodes).rev() {
                g.insert_node(Node::new(
                    NodeId(0),
                    "Add",
                    vec![Some(values[i - 1]), Some(one)],
                    vec![values[i]],
                ));
            }
        } else {
            for i in 1..=nodes {
                g.insert_node(Node::new(
                    NodeId(0),
                    "Add",
                    vec![Some(values[i - 1]), Some(one)],
                    vec![values[i]],
                ));
            }
        }

        let out = values[nodes];
        g.add_output(out);
        (g, out)
    }

    #[test]
    fn reverse_node_id_chain_matches_forward_order() {
        let (mut forward, forward_out) = constant_chain(64, false);
        let (mut reverse, reverse_out) = constant_chain(64, true);

        let reverse_ids = reverse.topological_order().unwrap();
        assert!(
            reverse_ids.windows(2).all(|ids| ids[0].0 > ids[1].0),
            "test graph must have reverse dependency NodeIds"
        );

        ConstantFolding
            .run(&mut forward, &PassContext::new())
            .unwrap();
        ConstantFolding
            .run(&mut reverse, &PassContext::new())
            .unwrap();

        assert_eq!(forward.num_nodes(), 0);
        assert_eq!(reverse.num_nodes(), 0);
        assert_eq!(
            inline_const(&forward, forward_out),
            inline_const(&reverse, reverse_out)
        );
        assert_eq!(
            read_i64(inline_const(&reverse, reverse_out).unwrap()),
            Some(vec![64])
        );
        assert!(forward.validate().is_ok());
        assert!(reverse.validate().is_ok());
    }

    #[test]
    fn does_not_fold_float_binary() {
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let mk = |g: &mut Graph, name: &str| {
            let v = g.create_named_value(name, DataType::Float32, static_shape([2]));
            g.set_initializer(
                v,
                WeightRef::Inline(TensorData::from_raw(
                    DataType::Float32,
                    vec![2],
                    vec![0u8; 8],
                )),
            );
            v
        };
        let a = mk(&mut g, "a");
        let b = mk(&mut g, "b");
        let out = g.create_named_value("out", DataType::Float32, static_shape([2]));
        g.insert_node(Node::new(
            NodeId(0),
            "Mul",
            vec![Some(a), Some(b)],
            vec![out],
        ));
        g.add_output(out);

        ConstantFolding.run(&mut g, &PassContext::new()).unwrap();
        assert_eq!(g.num_nodes(), 1, "float folding is out of scope in v1");
    }

    fn raw_const_init(
        graph: &mut Graph,
        name: &str,
        dtype: DataType,
        dims: Vec<usize>,
        data: Vec<u8>,
    ) -> ValueId {
        let shape = static_shape(dims.clone());
        let v = graph.create_named_value(name, dtype, shape);
        graph.set_initializer(
            v,
            WeightRef::Inline(TensorData::from_raw(dtype, dims, data)),
        );
        v
    }

    fn ints_attr_node(op_type: &str, inputs: Vec<Option<ValueId>>, outputs: Vec<ValueId>) -> Node {
        Node::new(NodeId(0), op_type, inputs, outputs)
    }

    #[test]
    fn folds_reshape_with_literal_shape() {
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let data = raw_const_init(&mut g, "x", DataType::Uint8, vec![2, 3], (0u8..6).collect());
        let shape = const_init(&mut g, "shape", vec![1], &[6]);
        let out = g.create_named_value("out", DataType::Uint8, static_shape([6]));
        g.insert_node(ints_attr_node(
            "Reshape",
            vec![Some(data), Some(shape)],
            vec![out],
        ));
        g.add_output(out);

        ConstantFolding.run(&mut g, &PassContext::new()).unwrap();
        let t = inline_const(&g, out).expect("Reshape folded to initializer");
        assert_eq!(t.dims, vec![6]);
        assert_eq!(t.data, (0u8..6).collect::<Vec<_>>());
        assert!(g.validate().is_ok());
    }

    #[test]
    fn folds_reshape_with_inferred_dim() {
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let data = raw_const_init(&mut g, "x", DataType::Uint8, vec![2, 3], (0u8..6).collect());
        let shape = const_init(&mut g, "shape", vec![2], &[3, -1]);
        let out = g.create_named_value("out", DataType::Uint8, static_shape([3, 2]));
        g.insert_node(ints_attr_node(
            "Reshape",
            vec![Some(data), Some(shape)],
            vec![out],
        ));
        g.add_output(out);

        ConstantFolding.run(&mut g, &PassContext::new()).unwrap();
        let t = inline_const(&g, out).expect("Reshape folded to initializer");
        assert_eq!(t.dims, vec![3, 2]);
        assert_eq!(t.data, (0u8..6).collect::<Vec<_>>());
    }

    #[test]
    fn folds_reshape_beyond_shape_fold_bound() {
        // Reshape/Transpose must fold at weight scale, well past MAX_FOLD_ELEMS
        // (the bound reserved for shape-computation-sized Add/Sub/Mul/Concat).
        let numel = MAX_FOLD_ELEMS * 4;
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let data = raw_const_init(&mut g, "x", DataType::Uint8, vec![numel], vec![0u8; numel]);
        let shape = const_init(&mut g, "shape", vec![2], &[2, (numel / 2) as i64]);
        let out = g.create_named_value("out", DataType::Uint8, static_shape([2, numel / 2]));
        g.insert_node(ints_attr_node(
            "Reshape",
            vec![Some(data), Some(shape)],
            vec![out],
        ));
        g.add_output(out);

        ConstantFolding.run(&mut g, &PassContext::new()).unwrap();
        let t = inline_const(&g, out).expect("large Reshape must still fold");
        assert_eq!(t.dims, vec![2, numel / 2]);
    }

    #[test]
    fn folds_transpose_permutes_bytes() {
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        // 2x3 tensor: rows [0,1,2] and [3,4,5]; perm=[1,0] transposes to
        // 3x2: [0,3, 1,4, 2,5].
        let data = raw_const_init(
            &mut g,
            "x",
            DataType::Uint8,
            vec![2, 3],
            vec![0, 1, 2, 3, 4, 5],
        );
        let out = g.create_named_value("out", DataType::Uint8, static_shape([3, 2]));
        let mut node = ints_attr_node("Transpose", vec![Some(data)], vec![out]);
        node.attributes
            .insert("perm".into(), Attribute::Ints(vec![1, 0]));
        g.insert_node(node);
        g.add_output(out);

        ConstantFolding.run(&mut g, &PassContext::new()).unwrap();
        let t = inline_const(&g, out).expect("Transpose folded to initializer");
        assert_eq!(t.dims, vec![3, 2]);
        assert_eq!(t.data, vec![0, 3, 1, 4, 2, 5]);
    }

    #[test]
    fn folds_concat_along_axis_zero() {
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let a = const_init(&mut g, "a", vec![2], &[1, 2]);
        let b = const_init(&mut g, "b", vec![3], &[3, 4, 5]);
        let out = g.create_named_value("out", DataType::Int64, static_shape([5]));
        let mut node = ints_attr_node("Concat", vec![Some(a), Some(b)], vec![out]);
        node.attributes.insert("axis".into(), Attribute::Int(0));
        g.insert_node(node);
        g.add_output(out);

        ConstantFolding.run(&mut g, &PassContext::new()).unwrap();
        let t = inline_const(&g, out).expect("Concat folded to initializer");
        assert_eq!(read_i64(t).unwrap(), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn folds_gate_up_interleave_chain_to_a_literal_initializer() {
        // Mirrors mobius's `_interleave_gate_up_rows`: reorder fc1 gate/up
        // rows from HF-concatenated `[E, 2*inter, ...]` layout to QMoE's
        // interleaved `[g_0, u_0, g_1, u_1, ...]` layout, entirely at
        // constant-fold time (relying on Shape(start=2)+Concat+Reshape+
        // Transpose+Concat+Reshape all folding into one literal initializer).
        let num_experts = 2usize;
        let half = 2usize;
        let fc1_out = 2 * half;
        let trailing = 3usize;
        let numel = num_experts * fc1_out * trailing;

        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let tensor = raw_const_init(
            &mut g,
            "fc1_experts_weights",
            DataType::Uint8,
            vec![num_experts, fc1_out, trailing],
            (0u8..numel as u8).collect(),
        );

        let trailing_shape = g.create_named_value("trailing", DataType::Int64, static_shape([1]));
        let mut shape_node = ints_attr_node("Shape", vec![Some(tensor)], vec![trailing_shape]);
        shape_node
            .attributes
            .insert("start".into(), Attribute::Int(2));
        g.insert_node(shape_node);

        let split_const = g.create_named_value("split_const", DataType::Int64, static_shape([3]));
        let mut split_const_node = ints_attr_node("Constant", vec![], vec![split_const]);
        split_const_node.attributes.insert(
            "value_ints".into(),
            Attribute::Ints(vec![num_experts as i64, 2, half as i64]),
        );
        g.insert_node(split_const_node);

        let split_shape = g.create_named_value("split_shape", DataType::Int64, static_shape([4]));
        let mut split_concat = ints_attr_node(
            "Concat",
            vec![Some(split_const), Some(trailing_shape)],
            vec![split_shape],
        );
        split_concat
            .attributes
            .insert("axis".into(), Attribute::Int(0));
        g.insert_node(split_concat);

        let reshaped = g.create_named_value(
            "reshaped",
            DataType::Uint8,
            static_shape([num_experts, 2, half, trailing]),
        );
        g.insert_node(ints_attr_node(
            "Reshape",
            vec![Some(tensor), Some(split_shape)],
            vec![reshaped],
        ));

        let transposed = g.create_named_value(
            "transposed",
            DataType::Uint8,
            static_shape([num_experts, half, 2, trailing]),
        );
        let mut transpose_node =
            ints_attr_node("Transpose", vec![Some(reshaped)], vec![transposed]);
        transpose_node
            .attributes
            .insert("perm".into(), Attribute::Ints(vec![0, 2, 1, 3]));
        g.insert_node(transpose_node);

        let merge_const = g.create_named_value("merge_const", DataType::Int64, static_shape([2]));
        let mut merge_const_node = ints_attr_node("Constant", vec![], vec![merge_const]);
        merge_const_node.attributes.insert(
            "value_ints".into(),
            Attribute::Ints(vec![num_experts as i64, fc1_out as i64]),
        );
        g.insert_node(merge_const_node);

        let merge_shape = g.create_named_value("merge_shape", DataType::Int64, static_shape([3]));
        let mut merge_concat = ints_attr_node(
            "Concat",
            vec![Some(merge_const), Some(trailing_shape)],
            vec![merge_shape],
        );
        merge_concat
            .attributes
            .insert("axis".into(), Attribute::Int(0));
        g.insert_node(merge_concat);

        let result = g.create_named_value(
            "result",
            DataType::Uint8,
            static_shape([num_experts, fc1_out, trailing]),
        );
        g.insert_node(ints_attr_node(
            "Reshape",
            vec![Some(transposed), Some(merge_shape)],
            vec![result],
        ));
        g.add_output(result);

        ConstantFolding.run(&mut g, &PassContext::new()).unwrap();
        DeadNodeElimination
            .run(&mut g, &PassContext::new())
            .unwrap();

        assert_eq!(
            g.num_nodes(),
            0,
            "the entire interleave chain must fold to a literal initializer"
        );
        let t = inline_const(&g, result).expect("result must be a literal initializer");
        assert_eq!(t.dims, vec![num_experts, fc1_out, trailing]);
        // Expert 0 rows: g0=[0,1,2] g1=[3,4,5] u0=[6,7,8] u1=[9,10,11].
        // Expert 1 rows: g0=[12,13,14] g1=[15,16,17] u0=[18,19,20] u1=[21,22,23].
        // Interleaved as [g0,u0,g1,u1] per expert.
        assert_eq!(
            t.data,
            vec![
                0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11, //
                12, 13, 14, 18, 19, 20, 15, 16, 17, 21, 22, 23,
            ]
        );
        assert!(g.validate().is_ok());
    }
}

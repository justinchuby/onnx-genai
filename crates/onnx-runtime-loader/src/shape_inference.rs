//! Best-effort static/symbolic shape inference (§19.3).
//!
//! Walks the graph in topological order and dispatches to a per-op rule table.
//! The table is **scoped to the BERT op set** (MatMul, Gemm, Add and other
//! broadcast elementwise ops, LayerNormalization, Reshape, Transpose, Gather,
//! Concat, Softmax, Erf/Gelu, Shape, Unsqueeze/Squeeze, reductions, …). Ops
//! outside the table are skipped, leaving their outputs' declared shapes
//! untouched — see [`infer_op_shapes`] and the `TODO(deferred-ops)` note below.

use std::collections::HashMap;

use onnx_runtime_ir::{Attribute, DataType, Dim, Graph, Node, Shape, ValueId};

use crate::LoaderError;

/// Run shape inference over the whole graph, populating value shapes (and, where
/// derivable, dtypes) in place.
pub fn run_shape_inference(mut graph: Graph) -> Result<Graph, LoaderError> {
    let order = graph
        .topological_order()
        .map_err(|_| LoaderError::GraphBuild("cycle detected during shape inference".into()))?;

    for nid in order {
        let node = graph.node(nid).clone();
        let input_shapes: Vec<Shape> = node
            .inputs
            .iter()
            .map(|slot| match slot {
                Some(id) => graph.value(*id).shape.clone(),
                None => Vec::new(),
            })
            .collect();
        let input_dtypes: Vec<DataType> = node
            .inputs
            .iter()
            .map(|slot| match slot {
                Some(id) => graph.value(*id).dtype,
                None => DataType::Float32,
            })
            .collect();

        // Ops whose output shape depends on the *values* of constant inputs
        // (initializers) are handled by the driver, which can read them.
        let inferred = infer_with_constants(&graph, &node, &input_shapes)
            .or_else(|| infer_op_shapes(&node.op_type, &node.domain, &input_shapes, &node.attributes));

        if let Some(shapes) = inferred {
            for (out, shape) in node.outputs.iter().zip(shapes) {
                graph.value_mut(*out).shape = shape;
            }
        }

        if let Some(dtypes) = infer_output_dtypes(&node, &input_dtypes) {
            for (out, dt) in node.outputs.iter().zip(dtypes) {
                graph.value_mut(*out).dtype = dt;
            }
        }
    }

    Ok(graph)
}

/// Read a constant `int64`/`int32` initializer as `Vec<i64>`, if `value` is an
/// inline initializer.
fn const_ints(graph: &Graph, value: ValueId) -> Option<Vec<i64>> {
    let w = graph.initializers.get(&value)?;
    let t = match w {
        onnx_runtime_ir::WeightRef::Inline(t) => t,
        _ => return None,
    };
    match t.dtype {
        DataType::Int64 => Some(
            t.data
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect(),
        ),
        DataType::Int32 => Some(
            t.data
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64)
                .collect(),
        ),
        _ => None,
    }
}

/// Handle ops whose output shape depends on constant input *values*.
fn infer_with_constants(graph: &Graph, node: &Node, inputs: &[Shape]) -> Option<Vec<Shape>> {
    match (node.domain.as_str(), node.op_type.as_str()) {
        ("" | "ai.onnx", "Reshape") => {
            let target = const_ints(graph, node.inputs.get(1).copied().flatten()?)?;
            Some(vec![reshape(inputs.first()?, &target)])
        }
        ("" | "ai.onnx", "Unsqueeze") => {
            // opset>=13: axes is input 1 (constant); else attribute.
            let axes = node
                .inputs
                .get(1)
                .copied()
                .flatten()
                .and_then(|v| const_ints(graph, v))
                .or_else(|| node.attr("axes").and_then(|a| a.as_ints()).map(<[i64]>::to_vec))?;
            Some(vec![unsqueeze(inputs.first()?, &axes)])
        }
        ("" | "ai.onnx", "Squeeze") => {
            let axes = node
                .inputs
                .get(1)
                .copied()
                .flatten()
                .and_then(|v| const_ints(graph, v))
                .or_else(|| node.attr("axes").and_then(|a| a.as_ints()).map(<[i64]>::to_vec));
            Some(vec![squeeze(inputs.first()?, axes.as_deref())])
        }
        _ => None,
    }
}

/// Infer output shapes for a single op from its input shapes and attributes.
///
/// Returns `None` for ops outside the BERT rule table (the driver then leaves
/// the declared output shapes in place).
///
/// TODO(deferred-ops): ops requiring value-dependent or op-specific rules not
/// yet modelled here (Slice, Pad, Conv, Split, NonZero, TopK, Range, Tile,
/// Expand, ScatterND, Resize, control-flow If/Loop/Scan, …) are intentionally
/// skipped for Phase 1 and should be added as the supported model set grows.
pub fn infer_op_shapes(
    op_type: &str,
    domain: &str,
    inputs: &[Shape],
    attrs: &HashMap<String, Attribute>,
) -> Option<Vec<Shape>> {
    if !matches!(domain, "" | "ai.onnx" | "com.microsoft") {
        return None;
    }
    match op_type {
        // Unary, shape-preserving elementwise.
        "Relu" | "Gelu" | "Erf" | "Sqrt" | "Tanh" | "Sigmoid" | "Exp" | "Log" | "Neg"
        | "Abs" | "Reciprocal" | "Sin" | "Cos" | "Identity" | "Softmax" | "LogSoftmax"
        | "Cast" | "Clip" | "Elu" | "LeakyRelu" | "BiasGelu" | "FastGelu" | "QuickGelu" => {
            Some(vec![inputs.first()?.clone()])
        }
        // Dropout: first output mirrors input; mask (2nd) ignored here.
        "Dropout" => Some(vec![inputs.first()?.clone()]),
        // Binary/N-ary broadcast elementwise.
        "Add" | "Sub" | "Mul" | "Div" | "Pow" | "Max" | "Min" | "Mean" | "Equal"
        | "Greater" | "Less" | "GreaterOrEqual" | "LessOrEqual" | "And" | "Or" | "Xor" => {
            Some(vec![broadcast_many(inputs)?])
        }
        "Where" => Some(vec![broadcast_many(inputs)?]),
        "MatMul" => Some(vec![matmul(inputs.first()?, inputs.get(1)?)?]),
        "Gemm" => Some(vec![gemm(inputs, attrs)?]),
        "LayerNormalization" | "SkipLayerNormalization" | "SimplifiedLayerNormalization" => {
            Some(vec![inputs.first()?.clone()])
        }
        "Transpose" => Some(vec![transpose(inputs.first()?, attrs)?]),
        "Gather" => Some(vec![gather(inputs.first()?, inputs.get(1)?, attrs)?]),
        "Concat" => Some(vec![concat(inputs, attrs)?]),
        "Shape" => Some(vec![vec![Dim::Static(inputs.first()?.len())]]),
        "Size" => Some(vec![Vec::new()]),
        "Flatten" => Some(vec![flatten(inputs.first()?, attrs)?]),
        "ReduceMean" | "ReduceSum" | "ReduceMax" | "ReduceMin" | "ReduceProd"
        | "ReduceL2" | "ReduceLogSumExp" => Some(vec![reduce(inputs.first()?, attrs)?]),
        _ => None,
    }
}

/// Best-effort output dtype propagation (the doc driver only sets shapes, but
/// interior values start with a placeholder dtype, so this improves fidelity).
fn infer_output_dtypes(node: &Node, input_dtypes: &[DataType]) -> Option<Vec<DataType>> {
    let out_n = node.outputs.len();
    match node.op_type.as_str() {
        "Cast" => {
            let to = node.attr("to")?.as_int()?;
            Some(vec![DataType::from_onnx(to as i32)?])
        }
        "Shape" | "Size" | "NonZero" | "ArgMax" | "ArgMin" => {
            Some(vec![DataType::Int64; out_n])
        }
        "Equal" | "Greater" | "Less" | "GreaterOrEqual" | "LessOrEqual" | "And" | "Or"
        | "Xor" | "Not" | "IsNaN" | "IsInf" => Some(vec![DataType::Bool; out_n.max(1)]),
        _ => {
            // Default: outputs share the first input's dtype.
            let dt = *input_dtypes.first()?;
            Some(vec![dt; out_n])
        }
    }
}

// === Shape rule helpers ===

fn axis_attr(attrs: &HashMap<String, Attribute>, name: &str, default: i64) -> i64 {
    attrs.get(name).and_then(Attribute::as_int).unwrap_or(default)
}

fn norm_axis(axis: i64, rank: usize) -> usize {
    if axis < 0 {
        (axis + rank as i64).max(0) as usize
    } else {
        (axis as usize).min(rank)
    }
}

/// Broadcast two dims (numpy rules), symbol-aware.
fn broadcast_dim(a: Dim, b: Dim) -> Dim {
    match (a, b) {
        (x, y) if x == y => x,
        (Dim::Static(1), y) => y,
        (x, Dim::Static(1)) => x,
        // Prefer a static extent over a symbol when both are non-1.
        (Dim::Static(n), Dim::Symbolic(_)) => Dim::Static(n),
        (Dim::Symbolic(_), Dim::Static(n)) => Dim::Static(n),
        // Two differing statics is ill-formed; keep the left as best effort.
        (x, _) => x,
    }
}

fn broadcast_two(a: &[Dim], b: &[Dim]) -> Shape {
    let rank = a.len().max(b.len());
    let mut out = vec![Dim::Static(1); rank];
    for i in 0..rank {
        let da = if i < rank - a.len() {
            Dim::Static(1)
        } else {
            a[i - (rank - a.len())]
        };
        let db = if i < rank - b.len() {
            Dim::Static(1)
        } else {
            b[i - (rank - b.len())]
        };
        out[i] = broadcast_dim(da, db);
    }
    out
}

fn broadcast_many(inputs: &[Shape]) -> Option<Shape> {
    let mut it = inputs.iter().filter(|s| !s.is_empty() || inputs.len() == 1);
    let mut acc = it.next().cloned().or_else(|| inputs.first().cloned())?;
    for s in inputs.iter().skip(1) {
        acc = broadcast_two(&acc, s);
    }
    Some(acc)
}

fn matmul(a: &[Dim], b: &[Dim]) -> Option<Shape> {
    match (a.len(), b.len()) {
        (0, _) | (_, 0) => None,
        (1, 1) => Some(Vec::new()), // dot product -> scalar
        (1, _) => {
            // a is [k]; treated as [1,k], result drops the leading 1.
            let mut r = matmul(&[Dim::Static(1), a[0]], b)?;
            r.remove(r.len() - 2);
            Some(r)
        }
        (_, 1) => {
            let mut r = matmul(a, &[b[0], Dim::Static(1)])?;
            r.pop();
            Some(r)
        }
        (na, nb) => {
            let batch = broadcast_two(&a[..na - 2], &b[..nb - 2]);
            let m = a[na - 2];
            let n = b[nb - 1];
            let mut out = batch;
            out.push(m);
            out.push(n);
            Some(out)
        }
    }
}

fn gemm(inputs: &[Shape], attrs: &HashMap<String, Attribute>) -> Option<Shape> {
    let a = inputs.first()?;
    let b = inputs.get(1)?;
    if a.len() != 2 || b.len() != 2 {
        return None;
    }
    let trans_a = axis_attr(attrs, "transA", 0) != 0;
    let trans_b = axis_attr(attrs, "transB", 0) != 0;
    let m = if trans_a { a[1] } else { a[0] };
    let n = if trans_b { b[0] } else { b[1] };
    Some(vec![m, n])
}

fn transpose(input: &[Dim], attrs: &HashMap<String, Attribute>) -> Option<Shape> {
    let rank = input.len();
    let perm: Vec<usize> = match attrs.get("perm").and_then(Attribute::as_ints) {
        Some(p) => p.iter().map(|&x| x as usize).collect(),
        None => (0..rank).rev().collect(),
    };
    if perm.len() != rank {
        return None;
    }
    Some(perm.iter().map(|&p| input[p]).collect())
}

fn gather(data: &[Dim], indices: &[Dim], attrs: &HashMap<String, Attribute>) -> Option<Shape> {
    if data.is_empty() {
        return None;
    }
    let axis = norm_axis(axis_attr(attrs, "axis", 0), data.len());
    let mut out = Vec::with_capacity(data.len() + indices.len() - 1);
    out.extend_from_slice(&data[..axis]);
    out.extend_from_slice(indices);
    out.extend_from_slice(&data[axis + 1..]);
    Some(out)
}

fn concat(inputs: &[Shape], attrs: &HashMap<String, Attribute>) -> Option<Shape> {
    let first = inputs.iter().find(|s| !s.is_empty())?;
    let rank = first.len();
    let axis = norm_axis(axis_attr(attrs, "axis", 0), rank);
    let mut out = first.clone();
    let mut sum = 0usize;
    let mut all_static = true;
    for s in inputs {
        if s.len() != rank {
            return None;
        }
        match s[axis] {
            Dim::Static(n) => sum += n,
            Dim::Symbolic(_) => all_static = false,
        }
    }
    out[axis] = if all_static {
        Dim::Static(sum)
    } else {
        Dim::Symbolic(u32_symbol())
    };
    Some(out)
}

fn flatten(input: &[Dim], attrs: &HashMap<String, Attribute>) -> Option<Shape> {
    let rank = input.len();
    let axis = norm_axis(axis_attr(attrs, "axis", 1), rank);
    let outer = mul_dims(&input[..axis]);
    let inner = mul_dims(&input[axis..]);
    Some(vec![outer, inner])
}

fn reduce(input: &[Dim], attrs: &HashMap<String, Attribute>) -> Option<Shape> {
    let rank = input.len();
    let keepdims = axis_attr(attrs, "keepdims", 1) != 0;
    let axes: Vec<usize> = match attrs.get("axes").and_then(Attribute::as_ints) {
        Some(a) => a.iter().map(|&x| norm_axis(x, rank)).collect(),
        None => (0..rank).collect(), // reduce all
    };
    let mut out = Vec::new();
    for (i, d) in input.iter().enumerate() {
        if axes.contains(&i) {
            if keepdims {
                out.push(Dim::Static(1));
            }
        } else {
            out.push(*d);
        }
    }
    Some(out)
}

fn reshape(input: &[Dim], target: &[i64]) -> Shape {
    let known: Option<usize> = if input.iter().all(|d| d.is_static()) {
        Some(input.iter().map(|d| d.as_static().unwrap()).product())
    } else {
        None
    };
    let mut out: Vec<Dim> = Vec::with_capacity(target.len());
    let mut product: usize = 1;
    let mut neg1: Option<usize> = None;
    for (i, &t) in target.iter().enumerate() {
        match t {
            -1 => {
                neg1 = Some(i);
                out.push(Dim::Static(1)); // placeholder, fixed below
            }
            0 => {
                // Copy the corresponding input dim.
                let d = input.get(i).copied().unwrap_or(Dim::Static(1));
                if let Dim::Static(n) = d {
                    product *= n;
                }
                out.push(d);
            }
            n => {
                product *= n as usize;
                out.push(Dim::Static(n as usize));
            }
        }
    }
    if let (Some(idx), Some(total)) = (neg1, known) {
        let rem = total.checked_div(product).unwrap_or(0);
        out[idx] = Dim::Static(rem);
    } else if let Some(idx) = neg1 {
        out[idx] = Dim::Symbolic(u32_symbol());
    }
    out
}

fn unsqueeze(input: &[Dim], axes: &[i64]) -> Shape {
    let out_rank = input.len() + axes.len();
    let norm: Vec<usize> = axes
        .iter()
        .map(|&a| if a < 0 { (a + out_rank as i64) as usize } else { a as usize })
        .collect();
    let mut out = Vec::with_capacity(out_rank);
    let mut src = input.iter();
    for i in 0..out_rank {
        if norm.contains(&i) {
            out.push(Dim::Static(1));
        } else if let Some(d) = src.next() {
            out.push(*d);
        }
    }
    out
}

fn squeeze(input: &[Dim], axes: Option<&[i64]>) -> Shape {
    let rank = input.len();
    match axes {
        Some(axes) => {
            let norm: Vec<usize> = axes.iter().map(|&a| norm_axis(a, rank)).collect();
            input
                .iter()
                .enumerate()
                .filter(|(i, _)| !norm.contains(i))
                .map(|(_, d)| *d)
                .collect()
        }
        None => input
            .iter()
            .filter(|d| !matches!(d, Dim::Static(1)))
            .copied()
            .collect(),
    }
}

fn mul_dims(dims: &[Dim]) -> Dim {
    if dims.iter().all(|d| d.is_static()) {
        Dim::Static(dims.iter().map(|d| d.as_static().unwrap()).product::<usize>().max(1))
    } else {
        Dim::Symbolic(u32_symbol())
    }
}

/// A fresh, un-interned symbol placeholder for a derived dynamic dim.
///
/// These come from computed extents (e.g. an all-symbolic Concat axis) that are
/// not tied to a graph-level dim-param; they are deliberately anonymous.
fn u32_symbol() -> onnx_runtime_ir::SymbolId {
    use std::sync::atomic::{AtomicU32, Ordering};
    // High range to avoid colliding with graph-interned symbol ids.
    static NEXT: AtomicU32 = AtomicU32::new(0x8000_0000);
    onnx_runtime_ir::SymbolId(NEXT.fetch_add(1, Ordering::Relaxed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::SymbolId;

    fn s(dims: &[usize]) -> Shape {
        dims.iter().map(|&d| Dim::Static(d)).collect()
    }

    #[test]
    fn matmul_2d() {
        let out = matmul(&s(&[3, 4]), &s(&[4, 5])).unwrap();
        assert_eq!(out, s(&[3, 5]));
    }

    #[test]
    fn matmul_batched_broadcast() {
        // [2,3,4] x [4,5] -> [2,3,5]
        let out = matmul(&s(&[2, 3, 4]), &s(&[4, 5])).unwrap();
        assert_eq!(out, s(&[2, 3, 5]));
    }

    #[test]
    fn matmul_symbolic_rows() {
        let batch = Dim::Symbolic(SymbolId(7));
        let a = vec![batch, Dim::Static(4)];
        let out = matmul(&a, &s(&[4, 8])).unwrap();
        assert_eq!(out, vec![batch, Dim::Static(8)]);
    }

    #[test]
    fn broadcast_bias() {
        // [batch, 8] + [8] -> [batch, 8]
        let batch = Dim::Symbolic(SymbolId(1));
        let out = broadcast_two(&[batch, Dim::Static(8)], &s(&[8]));
        assert_eq!(out, vec![batch, Dim::Static(8)]);
    }

    #[test]
    fn transpose_default_and_perm() {
        let mut attrs = HashMap::new();
        assert_eq!(transpose(&s(&[2, 3, 4]), &attrs).unwrap(), s(&[4, 3, 2]));
        attrs.insert("perm".to_string(), Attribute::Ints(vec![0, 2, 1]));
        assert_eq!(transpose(&s(&[2, 3, 4]), &attrs).unwrap(), s(&[2, 4, 3]));
    }

    #[test]
    fn gather_axis0() {
        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), Attribute::Int(0));
        // data [V, H], indices [B, S] -> [B, S, H]
        let out = gather(&s(&[100, 16]), &s(&[2, 5]), &attrs).unwrap();
        assert_eq!(out, s(&[2, 5, 16]));
    }

    #[test]
    fn reshape_infers_neg1() {
        assert_eq!(reshape(&s(&[2, 3, 4]), &[-1, 4]), s(&[6, 4]));
        assert_eq!(reshape(&s(&[2, 3, 4]), &[0, -1]), s(&[2, 12]));
    }

    #[test]
    fn concat_static_axis() {
        let mut attrs = HashMap::new();
        attrs.insert("axis".to_string(), Attribute::Int(1));
        let out = concat(&[s(&[2, 3]), s(&[2, 5])], &attrs).unwrap();
        assert_eq!(out, s(&[2, 8]));
    }

    #[test]
    fn reduce_mean_keepdims() {
        let mut attrs = HashMap::new();
        attrs.insert("axes".to_string(), Attribute::Ints(vec![-1]));
        attrs.insert("keepdims".to_string(), Attribute::Int(1));
        assert_eq!(reduce(&s(&[2, 3, 4]), &attrs).unwrap(), s(&[2, 3, 1]));
        attrs.insert("keepdims".to_string(), Attribute::Int(0));
        assert_eq!(reduce(&s(&[2, 3, 4]), &attrs).unwrap(), s(&[2, 3]));
    }

    #[test]
    fn unsqueeze_squeeze_roundtrip() {
        let up = unsqueeze(&s(&[2, 3]), &[0]);
        assert_eq!(up, s(&[1, 2, 3]));
        assert_eq!(squeeze(&up, Some(&[0])), s(&[2, 3]));
    }

    #[test]
    fn dispatch_unknown_op_returns_none() {
        assert!(infer_op_shapes("SomeCustomOp", "", &[s(&[2, 2])], &HashMap::new()).is_none());
    }
}

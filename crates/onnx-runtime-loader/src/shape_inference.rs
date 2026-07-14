//! Best-effort static/symbolic shape inference (§19.3).
//!
//! Walks the graph in topological order and dispatches to a per-op rule table.
//! Alongside the shape rules runs a bounded partial evaluator
//! ("const-fold-lite", see [`fold_value`]) that computes concrete `i64` values
//! — with symbolic passthrough — for the small integer tensors that make up
//! shape-computation subgraphs (`Constant → Shape → Gather/Slice → Concat →
//! Unsqueeze → Reshape/Expand`). Those folded values feed the value-dependent
//! shape rules, so the classic dynamic-shape chain resolves to concrete or
//! symbolic dims instead of being left unresolved. The evaluator is generic
//! over ONNX op semantics — there is no model-specific casing.
//!
//! The rule table covers the BERT op set (MatMul, Gemm, broadcast elementwise,
//! LayerNormalization, Reshape, Transpose, Gather, Concat, Softmax, Shape,
//! Slice, Expand, Unsqueeze/Squeeze, reductions, …). Ops outside the table are
//! skipped, leaving their outputs' declared shapes untouched.

use std::collections::HashMap;

use onnx_runtime_ir::{
    Attribute, DataType, Dim, Graph, Node, Shape, SymbolId, TensorData, ValueId, WeightRef,
};

use crate::LoaderError;

/// Upper bound on the element count of an integer tensor that the partial
/// evaluator ([`ConstEnv`]) is willing to fold. Shape-computation tensors are
/// tiny (a handful of dims); large weight tensors are never folded. This keeps
/// the const-fold "lite" and bounded (see the module docs / decision note).
const MAX_FOLD_ELEMS: usize = 1024;

/// One element of a folded integer shape-vector: either a concrete `i64` or a
/// symbolic dimension carried over from a dynamic input shape (e.g. the output
/// of `Shape` on a tensor with a symbolic `batch`/`seq` dim).
#[derive(Clone, Copy, Debug, PartialEq)]
enum IntElem {
    Const(i64),
    Sym(SymbolId),
}

/// A partially-evaluated integer tensor value (rank-0 scalar or rank-1 vector)
/// flowing through a shape-computation subgraph. Only small integer tensors are
/// represented; anything else stays `None` in the [`ConstEnv`].
#[derive(Clone, Debug)]
struct KnownVal {
    dtype: DataType,
    /// Static dims of the tensor. Empty == rank-0 scalar; `[n]` == rank-1.
    dims: Vec<usize>,
    /// Row-major elements. `len == 1` for a scalar.
    elems: Vec<IntElem>,
}

impl KnownVal {
    fn scalar(dtype: DataType, e: IntElem) -> Self {
        Self { dtype, dims: Vec::new(), elems: vec![e] }
    }
    fn vector(dtype: DataType, elems: Vec<IntElem>) -> Self {
        Self { dtype, dims: vec![elems.len()], elems }
    }
    fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }
    /// The shape implied by this value (concrete dims from its `dims`).
    fn shape(&self) -> Shape {
        self.dims.iter().map(|&d| Dim::Static(d)).collect()
    }
}

/// The environment of known constant integer values, keyed by value id. Filled
/// incrementally in topological order by the partial evaluator.
type ConstEnv = HashMap<ValueId, KnownVal>;

/// Run shape inference over the whole graph, populating value shapes (and, where
/// derivable, dtypes) in place.
///
/// A lightweight partial evaluator ("const-fold-lite") runs alongside the shape
/// rules: it computes concrete `i64` values (with symbolic passthrough) for the
/// small integer tensors that make up shape-computation subgraphs
/// (`Shape → Gather/Slice → Concat → Reshape/Expand`, …). Those folded values
/// feed the value-dependent shape rules (Reshape/Slice/Expand/…), which is what
/// lets the classic dynamic-shape chain resolve to concrete-or-symbolic dims.
pub fn run_shape_inference(mut graph: Graph) -> Result<Graph, LoaderError> {
    let order = graph
        .topological_order()
        .map_err(|_| LoaderError::GraphBuild("cycle detected during shape inference".into()))?;

    // Seed the const environment with foldable integer initializers.
    let mut env: ConstEnv = ConstEnv::new();
    let init_ids: Vec<ValueId> = graph.initializers.keys().copied().collect();
    for vid in init_ids {
        if let Some(kv) = known_from_weight(&graph.initializers[&vid]) {
            env.insert(vid, kv);
        }
    }

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

        // 1. Constant: derive shape + dtype (+ maybe a folded value) directly
        //    from the node's attribute.
        if node.op_type == "Constant" && matches!(node.domain.as_str(), "" | "ai.onnx") {
            if let (Some((shape, dtype, known)), Some(&out)) =
                (eval_constant(&node), node.outputs.first())
            {
                graph.value_mut(out).shape = shape;
                graph.value_mut(out).dtype = dtype;
                if let Some(kv) = known {
                    env.insert(out, kv);
                }
            }
            continue;
        }

        // 2. Partial evaluation: fold this node's (single) integer output value,
        //    if all its inputs are known small integer tensors. A folded value
        //    is authoritative — it fixes both the value and the concrete shape.
        if let Some(kv) = fold_value(&node, &input_shapes, &env) {
            if let Some(out) = node.outputs.first() {
                graph.value_mut(*out).shape = kv.shape();
                graph.value_mut(*out).dtype = kv.dtype;
                env.insert(*out, kv);
            }
            continue;
        }

        // 3. Shape rules. Value-dependent ops read folded values from `env`;
        //    the rest use pure input-shape rules. Inferred shapes only fill
        //    gaps (empty/unresolved outputs), never clobber a hint that a
        //    producer (e.g. ONNX `value_info`) already provided.
        let inferred = infer_value_dependent(&node, &input_shapes, &env)
            .or_else(|| infer_op_shapes(&node.op_type, &node.domain, &input_shapes, &node.attributes));

        if let Some(shapes) = inferred {
            for (out, shape) in node.outputs.iter().zip(shapes) {
                if graph.value(*out).shape.is_empty() {
                    graph.value_mut(*out).shape = shape;
                }
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

// === Constant-value propagation ("const-fold-lite") ===

/// Map an IR [`Dim`] to a folded integer element (static → concrete, symbolic →
/// passthrough symbol).
fn dim_to_elem(d: Dim) -> IntElem {
    match d {
        Dim::Static(n) => IntElem::Const(n as i64),
        Dim::Symbolic(s) => IntElem::Sym(s),
    }
}

/// Map a folded integer element back to an IR [`Dim`]. Non-negative constants
/// become static extents; symbols pass through; a stray negative constant
/// (should not occur in a real dim position) degrades to a fresh symbol.
fn elem_to_dim(e: IntElem) -> Dim {
    match e {
        IntElem::Const(n) if n >= 0 => Dim::Static(n as usize),
        IntElem::Const(_) => Dim::Symbolic(u32_symbol()),
        IntElem::Sym(s) => Dim::Symbolic(s),
    }
}

/// Decode a [`TensorData`]'s raw bytes as `i64`s, for the integer/bool element
/// types that shape math uses. Returns `None` for non-integer payloads.
fn tensor_ints(t: &TensorData) -> Option<Vec<i64>> {
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
        DataType::Uint8 | DataType::Int8 | DataType::Bool => {
            Some(t.data.iter().map(|&b| b as i64).collect())
        }
        _ => None,
    }
}

/// Build a [`KnownVal`] from a constant tensor, if it is a small (≤ [`MAX_FOLD_ELEMS`])
/// integer tensor of rank ≤ 1. Larger or non-integer tensors are not folded.
fn known_from_tensor(t: &TensorData) -> Option<KnownVal> {
    if t.dims.len() > 1 {
        return None;
    }
    let numel: usize = t.dims.iter().product::<usize>().max(1);
    if numel > MAX_FOLD_ELEMS {
        return None;
    }
    let ints = tensor_ints(t)?;
    if ints.len() != numel {
        return None;
    }
    Some(KnownVal {
        dtype: t.dtype,
        dims: t.dims.clone(),
        elems: ints.into_iter().map(IntElem::Const).collect(),
    })
}

/// Build a [`KnownVal`] from an inline integer initializer (external weights are
/// large floats and never folded).
fn known_from_weight(w: &WeightRef) -> Option<KnownVal> {
    match w {
        WeightRef::Inline(t) => known_from_tensor(t),
        WeightRef::External { .. } => None,
    }
}

/// Look up the folded value of a node input by slot index.
fn input_val<'a>(node: &Node, env: &'a ConstEnv, idx: usize) -> Option<&'a KnownVal> {
    node.inputs.get(idx).copied().flatten().and_then(|v| env.get(&v))
}

/// Read a node input as a concrete `i64` slice (all elements must be
/// [`IntElem::Const`]), from either a folded value input or an attribute
/// fallback.
fn const_ints_of(node: &Node, env: &ConstEnv, idx: usize, attr: &str) -> Option<Vec<i64>> {
    let from_input = input_val(node, env, idx).and_then(all_const);
    from_input.or_else(|| node.attr(attr).and_then(|a| a.as_ints()).map(<[i64]>::to_vec))
}

/// If every element of `kv` is a concrete constant, return them as `i64`s.
fn all_const(kv: &KnownVal) -> Option<Vec<i64>> {
    kv.elems
        .iter()
        .map(|e| match e {
            IntElem::Const(n) => Some(*n),
            IntElem::Sym(_) => None,
        })
        .collect()
}

/// Derive `(shape, dtype, folded value)` for a `Constant` node from its
/// attribute (the `value` tensor or the opset-12 scalar/list forms).
fn eval_constant(node: &Node) -> Option<(Shape, DataType, Option<KnownVal>)> {
    if let Some(Attribute::Tensor(t)) = node.attr("value") {
        let shape: Shape = t.dims.iter().map(|&d| Dim::Static(d)).collect();
        return Some((shape, t.dtype, known_from_tensor(t)));
    }
    if let Some(i) = node.attr("value_int").and_then(Attribute::as_int) {
        let kv = KnownVal::scalar(DataType::Int64, IntElem::Const(i));
        return Some((Vec::new(), DataType::Int64, Some(kv)));
    }
    if let Some(v) = node.attr("value_ints").and_then(Attribute::as_ints) {
        let elems = v.iter().map(|&x| IntElem::Const(x)).collect();
        let kv = KnownVal::vector(DataType::Int64, elems);
        let shape = vec![Dim::Static(v.len())];
        return Some((shape, DataType::Int64, Some(kv)));
    }
    if node.attr("value_float").is_some() {
        return Some((Vec::new(), DataType::Float32, None));
    }
    if let Some(Attribute::Floats(v)) = node.attr("value_floats") {
        return Some((vec![Dim::Static(v.len())], DataType::Float32, None));
    }
    None
}

/// Partially evaluate a node's single integer output value from folded inputs.
///
/// Handles the ops that make up shape-computation subgraphs. Returns `None` when
/// the value is not foldable (missing/symbolic operands, unsupported op, or a
/// non-integer output) — the caller then falls back to pure shape rules.
fn fold_value(node: &Node, input_shapes: &[Shape], env: &ConstEnv) -> Option<KnownVal> {
    if !matches!(node.domain.as_str(), "" | "ai.onnx") {
        return None;
    }
    let out = match node.op_type.as_str() {
        "Shape" => {
            let shape = input_shapes.first()?;
            let mut elems: Vec<IntElem> = shape.iter().map(|&d| dim_to_elem(d)).collect();
            // opset-15 start/end slicing of the dim list (attributes).
            let rank = elems.len() as i64;
            let clamp = |v: i64| -> usize {
                let v = if v < 0 { v + rank } else { v };
                v.clamp(0, rank) as usize
            };
            let start = clamp(node.attr("start").and_then(Attribute::as_int).unwrap_or(0));
            let end = clamp(node.attr("end").and_then(Attribute::as_int).unwrap_or(rank));
            elems = elems.get(start..end.max(start)).unwrap_or(&[]).to_vec();
            KnownVal::vector(DataType::Int64, elems)
        }
        "Identity" => input_val(node, env, 0)?.clone(),
        "Cast" => {
            let src = input_val(node, env, 0)?;
            let to = node.attr("to").and_then(Attribute::as_int)?;
            let dt = DataType::from_onnx(to as i32)?;
            // Only keep folding while the value stays integral (shape math).
            if !(dt.is_int() || dt == DataType::Bool) {
                return None;
            }
            KnownVal { dtype: dt, dims: src.dims.clone(), elems: src.elems.clone() }
        }
        "Unsqueeze" => {
            let src = input_val(node, env, 0)?;
            let axes = const_ints_of(node, env, 1, "axes")?;
            // Shape math only unsqueezes a scalar to a 1-vector.
            if !src.is_scalar() || axes != [0] {
                return None;
            }
            KnownVal::vector(src.dtype, src.elems.clone())
        }
        "Squeeze" => {
            let src = input_val(node, env, 0)?;
            if src.elems.len() != 1 {
                return None;
            }
            KnownVal::scalar(src.dtype, src.elems[0])
        }
        "Concat" => {
            let mut elems = Vec::new();
            let mut dtype = None;
            for slot in &node.inputs {
                let id = (*slot)?;
                let kv = env.get(&id)?;
                dtype.get_or_insert(kv.dtype);
                elems.extend_from_slice(&kv.elems);
                if elems.len() > MAX_FOLD_ELEMS {
                    return None;
                }
            }
            KnownVal::vector(dtype?, elems)
        }
        "Gather" => fold_gather(node, env)?,
        "Slice" => fold_slice(node, env)?,
        "Reshape" => {
            // 1-D integer reshape: element order is preserved.
            let src = input_val(node, env, 0)?;
            let target = const_ints_of(node, env, 1, "")?;
            let numel = src.elems.len();
            let concrete: usize = target
                .iter()
                .filter(|&&t| t > 0)
                .map(|&t| t as usize)
                .product::<usize>()
                .max(1);
            // Only a plain 1-D reshuffle (or [-1]) of a small vector is folded.
            if target.len() != 1 || (target[0] >= 0 && concrete != numel) {
                return None;
            }
            KnownVal::vector(src.dtype, src.elems.clone())
        }
        "Add" | "Sub" | "Mul" | "Div" | "Min" | "Max" => fold_binop(node, env)?,
        _ => return None,
    };
    Some(out)
}

/// Fold `Gather` on a 1-D integer vector along axis 0 (the shape-vector case).
fn fold_gather(node: &Node, env: &ConstEnv) -> Option<KnownVal> {
    let data = input_val(node, env, 0)?;
    let indices = input_val(node, env, 1)?;
    let axis = node.attr("axis").and_then(Attribute::as_int).unwrap_or(0);
    if data.is_scalar() || axis != 0 {
        return None;
    }
    let n = data.elems.len() as i64;
    let pick = |idx: IntElem| -> Option<IntElem> {
        match idx {
            IntElem::Const(i) => {
                let i = if i < 0 { i + n } else { i };
                data.elems.get(usize::try_from(i).ok()?).copied()
            }
            IntElem::Sym(_) => None,
        }
    };
    if indices.is_scalar() {
        Some(KnownVal::scalar(data.dtype, pick(indices.elems[0])?))
    } else {
        let elems: Option<Vec<IntElem>> = indices.elems.iter().map(|&i| pick(i)).collect();
        Some(KnownVal::vector(data.dtype, elems?))
    }
}

/// Fold `Slice` on a 1-D integer vector along axis 0 (opset-10 input form).
fn fold_slice(node: &Node, env: &ConstEnv) -> Option<KnownVal> {
    let data = input_val(node, env, 0)?;
    if data.is_scalar() {
        return None;
    }
    let starts = all_const(input_val(node, env, 1)?)?;
    let ends = all_const(input_val(node, env, 2)?)?;
    let axes = match input_val(node, env, 3) {
        Some(kv) => all_const(kv)?,
        None => (0..starts.len() as i64).collect(),
    };
    let steps = match input_val(node, env, 4) {
        Some(kv) => all_const(kv)?,
        None => vec![1; starts.len()],
    };
    if axes != [0] || starts.len() != 1 {
        return None;
    }
    let n = data.elems.len() as i64;
    let norm = |v: i64| -> i64 {
        let v = if v < 0 { v + n } else { v };
        v.clamp(0, n)
    };
    let (start, end, step) = (norm(starts[0]), norm(ends[0]), steps[0]);
    if step <= 0 {
        return None;
    }
    let mut elems = Vec::new();
    let mut i = start;
    while i < end {
        elems.push(*data.elems.get(i as usize)?);
        i += step;
    }
    Some(KnownVal::vector(data.dtype, elems))
}

/// Fold an element-wise binary integer op with numpy-style scalar broadcasting.
/// Any symbolic operand makes the corresponding output element a fresh symbol
/// (the value is genuinely unknown until runtime).
fn fold_binop(node: &Node, env: &ConstEnv) -> Option<KnownVal> {
    let a = input_val(node, env, 0)?;
    let b = input_val(node, env, 1)?;
    let len = a.elems.len().max(b.elems.len());
    // Only equal-length or scalar-broadcast shapes.
    let ea = |i: usize| a.elems[if a.elems.len() == 1 { 0 } else { i }];
    let eb = |i: usize| b.elems[if b.elems.len() == 1 { 0 } else { i }];
    if a.elems.len() != 1 && b.elems.len() != 1 && a.elems.len() != b.elems.len() {
        return None;
    }
    let op = node.op_type.as_str();
    let mut elems = Vec::with_capacity(len);
    for i in 0..len {
        let e = match (ea(i), eb(i)) {
            (IntElem::Const(x), IntElem::Const(y)) => {
                let r = match op {
                    "Add" => x.checked_add(y),
                    "Sub" => x.checked_sub(y),
                    "Mul" => x.checked_mul(y),
                    "Div" => (y != 0).then(|| x / y),
                    "Min" => Some(x.min(y)),
                    "Max" => Some(x.max(y)),
                    _ => None,
                };
                match r {
                    Some(v) => IntElem::Const(v),
                    None => IntElem::Sym(u32_symbol()),
                }
            }
            // A symbol on either side ⇒ unknown until runtime.
            _ => IntElem::Sym(u32_symbol()),
        };
        elems.push(e);
    }
    let dtype = if a.dtype.is_int() { a.dtype } else { b.dtype };
    if a.is_scalar() && b.is_scalar() {
        Some(KnownVal::scalar(dtype, elems.remove(0)))
    } else {
        Some(KnownVal::vector(dtype, elems))
    }
}

/// Handle ops whose output *shape* depends on constant input *values* but whose
/// data payload is not itself folded (e.g. `Reshape`/`Slice`/`Expand` of a
/// float activation tensor). Reads folded shape-vectors from `env`.
fn infer_value_dependent(node: &Node, inputs: &[Shape], env: &ConstEnv) -> Option<Vec<Shape>> {
    if !matches!(node.domain.as_str(), "" | "ai.onnx") {
        return None;
    }
    match node.op_type.as_str() {
        "Reshape" => {
            let target = input_val(node, env, 1)?;
            Some(vec![reshape_elems(inputs.first()?, &target.elems)])
        }
        "Unsqueeze" => {
            let axes = const_ints_of(node, env, 1, "axes")?;
            Some(vec![unsqueeze(inputs.first()?, &axes)])
        }
        "Squeeze" => {
            let axes = node
                .inputs
                .get(1)
                .copied()
                .flatten()
                .and_then(|v| env.get(&v))
                .and_then(all_const)
                .or_else(|| node.attr("axes").and_then(|a| a.as_ints()).map(<[i64]>::to_vec));
            Some(vec![squeeze(inputs.first()?, axes.as_deref())])
        }
        "Slice" => Some(vec![slice_shape(node, inputs.first()?, env)?]),
        "Expand" => {
            let target = input_val(node, env, 1)?;
            let target_dims: Shape = target.elems.iter().map(|&e| elem_to_dim(e)).collect();
            Some(vec![broadcast_two(inputs.first()?, &target_dims)])
        }
        _ => None,
    }
}

/// Infer a `Slice` output shape (opset-10 input form) from the data shape and
/// the folded `starts`/`ends`/`axes`/`steps` value inputs. Sliced axes whose
/// bounds are symbolic or unknown (data-dependent) yield a fresh symbolic
/// extent rather than a guess; unsliced axes keep their input extent.
fn slice_shape(node: &Node, input: &[Dim], env: &ConstEnv) -> Option<Shape> {
    let rank = input.len();
    let starts = input_val(node, env, 1);
    let ends = input_val(node, env, 2);
    let steps = input_val(node, env, 4);
    // Which axes are sliced: explicit `axes` input, else the default
    // `0..len(starts)`. We need at least one of these to know the mapping.
    let axes: Vec<i64> = match input_val(node, env, 3) {
        Some(kv) => all_const(kv)?,
        None => {
            let k = starts
                .map(|s| s.elems.len())
                .or_else(|| ends.map(|e| e.elems.len()))?;
            (0..k as i64).collect()
        }
    };
    let mut out = input.to_vec();
    for (k, &ax) in axes.iter().enumerate() {
        let ax = norm_axis(ax, rank);
        let s = starts.and_then(|kv| kv.elems.get(k)).copied();
        let e = ends.and_then(|kv| kv.elems.get(k)).copied();
        let st = steps
            .and_then(|kv| kv.elems.get(k))
            .copied()
            .unwrap_or(IntElem::Const(1));
        out[ax] = match (input[ax], s, e, st) {
            (
                Dim::Static(n),
                Some(IntElem::Const(s)),
                Some(IntElem::Const(e)),
                IntElem::Const(st),
            ) if st > 0 => {
                let n = n as i64;
                let clamp = |v: i64| if v < 0 { (v + n).max(0) } else { v.min(n) };
                let (s, e) = (clamp(s), clamp(e));
                let span = (e - s).max(0);
                Dim::Static(((span + st - 1) / st) as usize)
            }
            // Data-dependent / unknown bound: leave the extent symbolic.
            _ => Dim::Symbolic(u32_symbol()),
        };
    }
    Some(out)
}

/// Infer output shapes for a single op from its input shapes and attributes.
///
/// Returns `None` for ops outside the BERT rule table (the driver then leaves
/// the declared output shapes in place).
///
/// Value-dependent ops (`Constant`, `Reshape`, `Slice`, `Expand`,
/// `Unsqueeze`/`Squeeze`) are handled in [`run_shape_inference`] /
/// [`infer_value_dependent`] because they need folded input *values*, not just
/// shapes.
///
/// TODO(deferred-ops): ops requiring value-dependent or op-specific rules not
/// yet modelled here (Pad, Conv, Split, NonZero, TopK, Range, Tile, ScatterND,
/// Resize, control-flow If/Loop/Scan, …) are intentionally skipped for Phase 1
/// and should be added as the supported model set grows.
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

/// Normalise a signed `axis` attribute to a `usize` index into a shape of
/// `rank` dimensions.
///
/// Negative values wrap around (e.g., -1 → rank-1).  Positive values are
/// clamped to `rank.saturating_sub(1)` — not `rank` — so that callers such as
/// `gather` and `concat` that index `shape[axis]` or `shape[axis+1..]` cannot
/// panic on a malformed `axis == rank` value from the model proto.
fn norm_axis(axis: i64, rank: usize) -> usize {
    if axis < 0 {
        (axis + rank as i64).max(0) as usize
    } else {
        (axis as usize).min(rank.saturating_sub(1))
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

#[cfg(test)]
fn reshape(input: &[Dim], target: &[i64]) -> Shape {
    let elems: Vec<IntElem> = target.iter().map(|&t| IntElem::Const(t)).collect();
    reshape_elems(input, &elems)
}

/// Reshape rule over folded target elements, so a symbolic target dim (from a
/// data-dependent shape chain) propagates as a symbolic output dim.
fn reshape_elems(input: &[Dim], target: &[IntElem]) -> Shape {
    let known: Option<usize> = if input.iter().all(|d| d.is_static()) {
        Some(input.iter().map(|d| d.as_static().unwrap()).product())
    } else {
        None
    };
    let mut out: Vec<Dim> = Vec::with_capacity(target.len());
    let mut product: usize = 1;
    let mut neg1: Option<usize> = None;
    // A symbolic target dim makes the total product unknown, so a `-1` entry
    // cannot be resolved to a concrete extent.
    let mut has_symbolic = false;
    for (i, &t) in target.iter().enumerate() {
        match t {
            IntElem::Const(-1) => {
                neg1 = Some(i);
                out.push(Dim::Static(1)); // placeholder, fixed below
            }
            IntElem::Const(0) => {
                // Copy the corresponding input dim.
                let d = input.get(i).copied().unwrap_or(Dim::Static(1));
                match d {
                    Dim::Static(n) => product *= n,
                    Dim::Symbolic(_) => has_symbolic = true,
                }
                out.push(d);
            }
            IntElem::Const(n) => {
                product *= n as usize;
                out.push(Dim::Static(n as usize));
            }
            IntElem::Sym(s) => {
                has_symbolic = true;
                out.push(Dim::Symbolic(s));
            }
        }
    }
    if let (Some(idx), Some(total), false) = (neg1, known, has_symbolic) {
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
    use onnx_runtime_ir::{NodeId, SymbolId};

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

    #[test]
    fn reshape_elems_propagates_symbolic_target() {
        // A symbolic target dim (from a dynamic shape chain) must pass through,
        // and a `-1` alongside it cannot be resolved so it stays symbolic.
        let sym = SymbolId(42);
        let out = reshape_elems(
            &s(&[2, 12]),
            &[IntElem::Sym(sym), IntElem::Const(-1), IntElem::Const(4)],
        );
        assert!(matches!(out[0], Dim::Symbolic(id) if id == sym));
        assert_eq!(out[2], Dim::Static(4));
        assert!(matches!(out[1], Dim::Symbolic(_)), "-1 unresolved -> symbolic");
    }

    #[test]
    fn reshape_elems_copy_and_neg1_static() {
        // 0 copies the input dim; -1 resolves from the known total.
        let out = reshape_elems(&s(&[2, 3, 4]), &[IntElem::Const(0), IntElem::Const(-1)]);
        assert_eq!(out, s(&[2, 12]));
    }

    #[test]
    fn fold_binop_symbol_stays_symbolic() {
        // Min([Sym, Sym], [512, 512]) must yield symbols, not a wrong constant.
        let node = Node {
            id: NodeId(0),
            op_type: "Min".into(),
            domain: String::new(),
            inputs: vec![Some(ValueId(0)), Some(ValueId(1))],
            outputs: vec![ValueId(2)],
            attributes: HashMap::new(),
            doc_string: None,
            device: None,
            exec_order: None,
        };
        let mut env = ConstEnv::new();
        env.insert(
            ValueId(0),
            KnownVal::vector(DataType::Int64, vec![IntElem::Sym(SymbolId(1)), IntElem::Sym(SymbolId(2))]),
        );
        env.insert(
            ValueId(1),
            KnownVal::vector(DataType::Int64, vec![IntElem::Const(512), IntElem::Const(512)]),
        );
        let out = fold_binop(&node, &env).unwrap();
        assert!(out.elems.iter().all(|e| matches!(e, IntElem::Sym(_))));
    }

    #[test]
    fn fold_binop_concrete_computes() {
        let node = Node {
            id: NodeId(0),
            op_type: "Add".into(),
            domain: String::new(),
            inputs: vec![Some(ValueId(0)), Some(ValueId(1))],
            outputs: vec![ValueId(2)],
            attributes: HashMap::new(),
            doc_string: None,
            device: None,
            exec_order: None,
        };
        let mut env = ConstEnv::new();
        env.insert(ValueId(0), KnownVal::scalar(DataType::Int64, IntElem::Const(3)));
        env.insert(ValueId(1), KnownVal::scalar(DataType::Int64, IntElem::Const(4)));
        let out = fold_binop(&node, &env).unwrap();
        assert_eq!(out.elems, vec![IntElem::Const(7)]);
        assert!(out.is_scalar());
    }

    #[test]
    fn norm_axis_clamps_at_rank_minus_one() {
        // A malformed axis == rank used to clamp to `rank`, which would cause
        // an index panic in `gather` / `concat`. It must now clamp to rank-1.
        assert_eq!(norm_axis(3, 3), 2, "axis == rank should clamp to rank-1");
        assert_eq!(norm_axis(10, 3), 2, "axis >> rank should clamp to rank-1");
        // Negative axes still wrap correctly.
        assert_eq!(norm_axis(-1, 4), 3);
        assert_eq!(norm_axis(-4, 4), 0);
        // rank == 0 edge: saturating_sub avoids underflow.
        assert_eq!(norm_axis(0, 0), 0);
    }
}

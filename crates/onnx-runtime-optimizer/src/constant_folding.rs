//! Constant folding: replace a node whose inputs are *all* constant
//! (initializers) with a precomputed initializer (see `docs/ORT2.md` §18.1).
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
//!
//! Everything else is left untouched. Folding is bounded to
//! [`MAX_FOLD_ELEMS`] elements so large weight tensors are never materialized,
//! and dispatch is purely on op type — no model-specific names. The invariant
//! is: **never produce a wrong constant.** When in doubt, do not fold.

use onnx_runtime_ir::{
    Attribute, DataType, Graph, NodeId, TensorData, ValueId, WeightRef, as_static_shape,
    static_shape,
};

use crate::error::Result;
use crate::pass::{OptimizationPass, PassContext};

/// Upper bound on the number of elements this pass will materialize. Keeps
/// folding limited to shape-computation-sized tensors, never model weights.
const MAX_FOLD_ELEMS: usize = 1024;

/// Folds constant-input nodes into initializers (bounded, integer/shape only).
#[derive(Clone, Copy, Debug, Default)]
pub struct ConstantFolding;

impl OptimizationPass for ConstantFolding {
    fn name(&self) -> &str {
        "ConstantFolding"
    }

    fn run(&self, graph: &mut Graph, _ctx: &PassContext) -> Result<()> {
        // Iterate to a fixpoint: folding one node may make its consumers
        // foldable in the next round (e.g. Constant -> Add).
        loop {
            let mut changed = false;
            let node_ids: Vec<NodeId> = graph.nodes.keys().collect();
            for nid in node_ids {
                if !graph.nodes.contains(nid) {
                    continue;
                }
                let node = graph.node(nid).clone();
                if !matches!(node.domain.as_str(), "" | "ai.onnx") {
                    continue;
                }
                if node.outputs.len() != 1 {
                    continue;
                }
                let out = node.outputs[0];

                let folded: Option<TensorData> = match node.op_type.as_str() {
                    "Constant" => eval_constant(&node),
                    "Shape" => fold_shape(graph, &node),
                    "Add" | "Sub" | "Mul" => fold_binary_int(graph, &node),
                    _ => None,
                };

                let Some(tensor) = folded else { continue };

                // Only fold outputs that are still needed (have a consumer or
                // are graph outputs); dead outputs are DCE's job and folding
                // them would leave a stale initializer referencing a GC'd id.
                let needed = graph.outputs.contains(&out)
                    || graph
                        .try_value(out)
                        .is_some_and(|v| !v.consumers.is_empty());
                if !needed {
                    continue;
                }

                graph.remove_node(nid);
                // The output survives because it is needed; retype it to the
                // folded tensor and back it with an inline initializer.
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
        Ok(())
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
        return Some(TensorData::from_raw(DataType::Int64, vec![ints.len()], data));
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

/// Fold `Shape(x)` when `x` has a fully-static shape into an `int64` vector.
///
/// Conservative: bails if `start`/`end` attributes are present (a slice of the
/// shape) so we never emit a partial result.
fn fold_shape(graph: &Graph, node: &onnx_runtime_ir::Node) -> Option<TensorData> {
    if node.attr("start").is_some() || node.attr("end").is_some() {
        return None;
    }
    let input = node.inputs.first().copied().flatten()?;
    let shape = &graph.try_value(input)?.shape;
    let dims = as_static_shape(shape)?;
    if dims.len() > MAX_FOLD_ELEMS {
        return None;
    }
    let mut data = Vec::with_capacity(dims.len() * 8);
    for &d in &dims {
        data.extend_from_slice(&(d as i64).to_le_bytes());
    }
    Some(TensorData::from_raw(DataType::Int64, vec![dims.len()], data))
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

fn read_i64(t: &TensorData) -> Option<Vec<i64>> {
    if t.data.len() != t.numel() * 8 {
        return None;
    }
    Some(
        t.data
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect(),
    )
}

fn read_i32(t: &TensorData) -> Option<Vec<i32>> {
    if t.data.len() != t.numel() * 4 {
        return None;
    }
    Some(
        t.data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{Node, NodeId};

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

    #[test]
    fn folds_add_of_two_const_inputs() {
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let a = const_init(&mut g, "a", vec![3], &[1, 2, 3]);
        let b = const_init(&mut g, "b", vec![3], &[10, 20, 30]);
        let out = g.create_named_value("out", DataType::Int64, static_shape([3]));
        g.insert_node(Node::new(NodeId(0), "Add", vec![Some(a), Some(b)], vec![out]));
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
        g.insert_node(Node::new(NodeId(0), "Add", vec![Some(a), Some(b)], vec![out]));
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
        g.insert_node(Node::new(NodeId(0), "Add", vec![Some(a), Some(b)], vec![out]));
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
        g.insert_node(Node::new(NodeId(0), "Add", vec![Some(a), Some(b)], vec![out]));
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
        node.attributes
            .insert("value".into(), Attribute::Tensor(int64_tensor(vec![2], &[7, 8])));
        g.insert_node(node);
        // Keep `out` alive with a consumer.
        let sink = g.create_named_value("sink", DataType::Int64, static_shape([2]));
        g.insert_node(Node::new(NodeId(0), "Identity", vec![Some(out)], vec![sink]));
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
        n1.attributes
            .insert("value".into(), Attribute::Tensor(int64_tensor(vec![2], &[1, 2])));
        g.insert_node(n1);

        let c2 = g.create_named_value("c2", DataType::Int64, static_shape([2]));
        let mut n2 = Node::new(NodeId(0), "Constant", vec![], vec![c2]);
        n2.attributes
            .insert("value".into(), Attribute::Tensor(int64_tensor(vec![2], &[3, 4])));
        g.insert_node(n2);

        let out = g.create_named_value("out", DataType::Int64, static_shape([2]));
        g.insert_node(Node::new(NodeId(0), "Add", vec![Some(c1), Some(c2)], vec![out]));
        g.add_output(out);

        ConstantFolding.run(&mut g, &PassContext::new()).unwrap();
        assert_eq!(g.num_nodes(), 0, "both constants and the Add fold away");
        let t = inline_const(&g, out).unwrap();
        assert_eq!(read_i64(t).unwrap(), vec![4, 6]);
        assert!(g.validate().is_ok());
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
        g.insert_node(Node::new(NodeId(0), "Mul", vec![Some(a), Some(b)], vec![out]));
        g.add_output(out);

        ConstantFolding.run(&mut g, &PassContext::new()).unwrap();
        assert_eq!(g.num_nodes(), 1, "float folding is out of scope in v1");
    }
}

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

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use onnx_runtime_ir::{
    Attribute, DataType, Graph, NodeId, TensorData, ValueId, WeightRef, as_static_shape,
    is_fully_static, static_shape,
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
            "Constant" | "Shape" | "Add" | "Sub" | "Mul"
        )
}

fn unresolved_inputs(graph: &Graph, node: &onnx_runtime_ir::Node) -> Option<Vec<ValueId>> {
    match node.op_type.as_str() {
        "Constant" => Some(Vec::new()),
        "Shape" => {
            if node.attr("start").is_some() || node.attr("end").is_some() {
                return None;
            }
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
    Some(TensorData::from_raw(
        DataType::Int64,
        vec![dims.len()],
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
}

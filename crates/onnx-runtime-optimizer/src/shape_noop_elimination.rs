//! Eliminate shape/metadata nodes whose value is provably identical to input 0.

use std::collections::{HashMap, HashSet};

use onnx_runtime_ir::{Graph, Node, NodeId, ValueId};

use crate::error::{OptimizerError, Result};
use crate::pass::{OptimizationPass, PassContext};

/// Removes no-op `Identity` / reshape-class nodes from the executable graph.
///
/// This pass is deliberately conservative: it rewires consumers only when the
/// output tensor has the same known dtype, shape, and layout as data input 0.
/// Shape-changing view ops still need a distinct SSA value to carry the changed
/// runtime geometry, so they are left to the executor's zero-copy view path.
#[derive(Clone, Copy, Debug, Default)]
pub struct ShapeNoOpElimination;

impl OptimizationPass for ShapeNoOpElimination {
    fn name(&self) -> &str {
        "ShapeNoOpElimination"
    }

    fn run(&self, graph: &mut Graph, _ctx: &PassContext) -> Result<()> {
        let changed = eliminate_in_graph(graph);
        if changed {
            graph.validate().map_err(OptimizerError::from)?;
        }
        Ok(())
    }
}

fn eliminate_in_graph(graph: &mut Graph) -> bool {
    let mut changed = false;

    loop {
        let captured_values = values_captured_by_subgraphs(graph);
        let removable = graph
            .topological_order()
            .unwrap_or_else(|_| graph.nodes.keys().collect())
            .into_iter()
            .find(|&nid| removable_node(graph, nid, &captured_values).is_some());

        let Some(nid) = removable else {
            break;
        };
        let (input, output) = removable_node(graph, nid, &captured_values)
            .expect("candidate was just checked removable");
        graph.replace_all_uses(output, input);
        graph.remove_node(nid);
        changed = true;
    }

    for subgraph in graph.subgraphs.values_mut() {
        changed |= eliminate_in_graph(subgraph);
    }

    changed
}

fn removable_node(
    graph: &Graph,
    nid: NodeId,
    captured_values: &HashSet<ValueId>,
) -> Option<(ValueId, ValueId)> {
    let node = graph.try_node(nid)?;
    if !is_noop_shape_node(node) || node.outputs.len() != 1 {
        return None;
    }
    let input = node.inputs.first().copied().flatten()?;
    let output = node.outputs[0];
    if input == output
        || graph.outputs.contains(&output)
        || graph.initializers.contains_key(&output)
        || captured_values.contains(&output)
        || !graph.values.contains(input)
        || !graph.values.contains(output)
        || !graph.value_type_is_known(input)
        || !graph.value_type_is_known(output)
        || !graph.value_shape_is_known(input)
        || !graph.value_shape_is_known(output)
    {
        return None;
    }

    let in_value = graph.value(input);
    let out_value = graph.value(output);
    (in_value.dtype == out_value.dtype
        && in_value.shape == out_value.shape
        && in_value.layout == out_value.layout)
        .then_some((input, output))
}

fn is_noop_shape_node(node: &Node) -> bool {
    node.is_default_domain()
        && matches!(
            node.op_type.as_str(),
            "Identity" | "Reshape" | "Squeeze" | "Unsqueeze"
        )
}

fn values_captured_by_subgraphs(graph: &Graph) -> HashSet<ValueId> {
    if graph.subgraphs.is_empty() {
        return HashSet::new();
    }
    let name_index: HashMap<&str, ValueId> = graph
        .values
        .iter()
        .filter_map(|(vid, value)| value.name.as_deref().map(|name| (name, vid)))
        .collect();
    let mut captured = HashSet::new();
    for body in graph.subgraphs.values() {
        collect_outer_names(body, &name_index, &mut captured);
    }
    captured
}

fn collect_outer_names(
    graph: &Graph,
    outer_names: &HashMap<&str, ValueId>,
    captured: &mut HashSet<ValueId>,
) {
    for &input in &graph.inputs {
        let value = graph.value(input);
        if value.producer.is_none()
            && !graph.initializers.contains_key(&input)
            && let Some(name) = value.name.as_deref()
            && let Some(&outer) = outer_names.get(name)
        {
            captured.insert(outer);
        }
    }
    for body in graph.subgraphs.values() {
        collect_outer_names(body, outer_names, captured);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{Attribute, DataType, Dim, Node, NodeId, static_shape};

    fn named(g: &mut Graph, name: &str, shape: impl IntoIterator<Item = usize>) -> ValueId {
        g.create_named_value(name, DataType::Float32, static_shape(shape))
    }

    #[test]
    fn removes_identity_and_rewires_consumers() {
        let mut g = Graph::new();
        let input = named(&mut g, "input", [2, 3]);
        let bias = named(&mut g, "bias", [2, 3]);
        g.add_input(input);
        g.add_input(bias);
        let id_out = named(&mut g, "id_out", [2, 3]);
        let identity = g.insert_node(Node::new(
            NodeId(0),
            "Identity",
            vec![Some(input)],
            vec![id_out],
        ));
        let out = named(&mut g, "out", [2, 3]);
        let add = g.insert_node(Node::new(
            NodeId(0),
            "Add",
            vec![Some(id_out), Some(bias)],
            vec![out],
        ));
        g.add_output(out);

        ShapeNoOpElimination
            .run(&mut g, &PassContext::new())
            .unwrap();

        assert!(g.try_node(identity).is_none());
        assert_eq!(g.node(add).inputs[0], Some(input));
        assert!(g.try_value(id_out).is_none());
        assert!(g.validate().is_ok());
    }

    #[test]
    fn removes_equal_shape_reshape_but_keeps_shape_changing_view() {
        let mut g = Graph::new();
        let input = named(&mut g, "input", [2, 3]);
        let shape = g.create_named_value("shape", DataType::Int64, static_shape([2]));
        g.add_input(input);
        let same = named(&mut g, "same", [2, 3]);
        let same_reshape = g.insert_node(Node::new(
            NodeId(0),
            "Reshape",
            vec![Some(input), Some(shape)],
            vec![same],
        ));
        let changed = named(&mut g, "changed", [6]);
        let changed_reshape = g.insert_node(Node::new(
            NodeId(0),
            "Reshape",
            vec![Some(same), Some(shape)],
            vec![changed],
        ));
        g.add_output(changed);

        ShapeNoOpElimination
            .run(&mut g, &PassContext::new())
            .unwrap();

        assert!(g.try_node(same_reshape).is_none());
        assert!(g.try_node(changed_reshape).is_some());
        assert_eq!(g.node(changed_reshape).inputs[0], Some(input));
        assert!(g.validate().is_ok());
    }

    #[test]
    fn removes_noop_squeeze_and_unsqueeze() {
        let mut g = Graph::new();
        let input = named(&mut g, "input", [2, 3]);
        let axes = g.create_named_value("axes", DataType::Int64, static_shape([0]));
        g.add_input(input);
        let squeezed = named(&mut g, "squeezed", [2, 3]);
        let squeeze = g.insert_node(Node::new(
            NodeId(0),
            "Squeeze",
            vec![Some(input), Some(axes)],
            vec![squeezed],
        ));
        let unsqueezed = named(&mut g, "unsqueezed", [2, 3]);
        let unsqueeze = g.insert_node(Node::new(
            NodeId(0),
            "Unsqueeze",
            vec![Some(squeezed), Some(axes)],
            vec![unsqueezed],
        ));
        let out = named(&mut g, "out", [2, 3]);
        g.insert_node(Node::new(
            NodeId(0),
            "Add",
            vec![Some(unsqueezed), Some(input)],
            vec![out],
        ));
        g.add_output(out);

        ShapeNoOpElimination
            .run(&mut g, &PassContext::new())
            .unwrap();

        assert!(g.try_node(squeeze).is_none());
        assert!(g.try_node(unsqueeze).is_none());
        assert!(g.validate().is_ok());
    }

    #[test]
    fn preserves_graph_output_identity_name() {
        let mut g = Graph::new();
        let input = named(&mut g, "input", [4]);
        g.add_input(input);
        let output = named(&mut g, "public_output", [4]);
        let identity = g.insert_node(Node::new(
            NodeId(0),
            "Identity",
            vec![Some(input)],
            vec![output],
        ));
        g.add_output(output);

        ShapeNoOpElimination
            .run(&mut g, &PassContext::new())
            .unwrap();

        assert!(g.try_node(identity).is_some());
        assert_eq!(g.outputs, vec![output]);
        assert_eq!(g.value(output).name.as_deref(), Some("public_output"));
    }

    #[test]
    fn skips_unknown_shapes() {
        let mut g = Graph::new();
        let input = named(&mut g, "input", [4]);
        g.mark_value_shape_unknown(input);
        g.add_input(input);
        let output = named(&mut g, "identity_out", [4]);
        let identity = g.insert_node(Node::new(
            NodeId(0),
            "Identity",
            vec![Some(input)],
            vec![output],
        ));
        let final_out = named(&mut g, "out", [4]);
        g.insert_node(Node::new(
            NodeId(0),
            "Add",
            vec![Some(output), Some(output)],
            vec![final_out],
        ));
        g.add_output(final_out);

        ShapeNoOpElimination
            .run(&mut g, &PassContext::new())
            .unwrap();

        assert!(g.try_node(identity).is_some());
    }

    #[test]
    fn preserves_values_captured_by_control_flow_subgraphs() {
        let mut g = Graph::new();
        let input = named(&mut g, "input", [1]);
        g.add_input(input);
        let captured = named(&mut g, "captured", [1]);
        let identity = g.insert_node(Node::new(
            NodeId(0),
            "Identity",
            vec![Some(input)],
            vec![captured],
        ));
        let cond = g.create_named_value("cond", DataType::Bool, Vec::<Dim>::new());
        g.add_input(cond);
        let if_out = named(&mut g, "if_out", [1]);
        let if_node = g.insert_node(Node::new(NodeId(0), "If", vec![Some(cond)], vec![if_out]));
        let mut then_branch = Graph::new();
        let then_capture = named(&mut then_branch, "captured", [1]);
        then_branch.add_input(then_capture);
        then_branch.add_output(then_capture);
        g.node_mut(if_node).attributes.insert(
            "then_branch".to_string(),
            Attribute::Graph(Box::new(then_branch.clone())),
        );
        g.subgraphs
            .insert((if_node, "then_branch".to_string()), then_branch);
        let else_branch = {
            let mut branch = Graph::new();
            let else_capture = named(&mut branch, "captured", [1]);
            branch.add_input(else_capture);
            branch.add_output(else_capture);
            branch
        };
        g.node_mut(if_node).attributes.insert(
            "else_branch".to_string(),
            Attribute::Graph(Box::new(else_branch.clone())),
        );
        g.subgraphs
            .insert((if_node, "else_branch".to_string()), else_branch);
        g.add_output(if_out);

        ShapeNoOpElimination
            .run(&mut g, &PassContext::new())
            .unwrap();

        assert!(g.try_node(identity).is_some());
    }
}

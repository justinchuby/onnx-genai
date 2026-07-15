//! Built-in validation rules for this first wave (ONNX_RS §8.2).
//!
//! Three rules ship now, chosen to exercise the model-, node-, and value-level
//! [`ViolationLocation`] variants:
//!
//! * [`MissingOpsetImportRule`] — every operator domain used by a node must have
//!   a matching `opset_import` (§8.2 "IR rules").
//! * [`DuplicateValueNameRule`] — no two values may share a name (§8.2
//!   "structural rules": unique value names / single producer).
//! * [`GraphAcyclicRule`] — the dataflow graph must be acyclic (§8.2).

use std::collections::HashMap;

use onnx_runtime_ir::Graph;

use super::{Severity, ValidationContext, ValidationRule, Violation, ViolationLocation};
use crate::model::Model;

/// Normalise a node/opset domain: the empty string and `"ai.onnx"` both denote
/// the default ONNX domain.
fn normalize_domain(domain: &str) -> &str {
    if domain.is_empty() { "ai.onnx" } else { domain }
}

/// Every operator domain referenced by a node must be declared in
/// `opset_imports` (ONNX_RS §8.2, IR rule `OpsetImportPresentRule`).
pub struct MissingOpsetImportRule;

impl ValidationRule for MissingOpsetImportRule {
    fn id(&self) -> &str {
        "ir.opset_import_present"
    }

    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn check(&self, model: &Model, _ctx: &ValidationContext) -> Vec<Violation> {
        // Pre-normalise the declared import domains once.
        let declared: std::collections::HashSet<&str> = model
            .graph
            .opset_imports
            .keys()
            .map(|d| normalize_domain(d))
            .collect();
        check_graph_opset_imports(
            &model.graph,
            &model.metadata.graph_name,
            &declared,
            self.id(),
        )
    }
}

/// No two distinct values may carry the same (non-empty) name — ONNX graphs are
/// SSA, so a name identifies a unique edge (ONNX_RS §8.2 `UniqueValueNamesRule`).
pub struct DuplicateValueNameRule;

impl ValidationRule for DuplicateValueNameRule {
    fn id(&self) -> &str {
        "structure.duplicate_value_name"
    }

    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn check(&self, model: &Model, _ctx: &ValidationContext) -> Vec<Violation> {
        check_graph_duplicate_names(&model.graph, self.id())
    }
}

fn check_graph_opset_imports(
    graph: &Graph,
    graph_name: &str,
    declared: &std::collections::HashSet<&str>,
    rule_id: &str,
) -> Vec<Violation> {
    let mut violations = Vec::new();
    for (_nid, node) in graph.nodes.iter() {
        let domain = normalize_domain(&node.domain);
        if !declared.contains(domain) {
            violations.push(Violation {
                rule_id: rule_id.to_string(),
                severity: Severity::Error,
                message: format!(
                    "node '{}' ({}) uses domain '{}' but no matching opset_import is declared",
                    node_label(node),
                    node.op_type,
                    if node.domain.is_empty() {
                        "ai.onnx"
                    } else {
                        &node.domain
                    },
                ),
                location: ViolationLocation::Node {
                    graph_name: graph_name.to_string(),
                    node_name: node_label(node),
                },
            });
        }
    }
    for subgraph in graph.subgraphs.values() {
        violations.extend(check_graph_opset_imports(
            subgraph, graph_name, declared, rule_id,
        ));
    }
    violations
}

fn check_graph_duplicate_names(graph: &Graph, rule_id: &str) -> Vec<Violation> {
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for (_vid, value) in graph.values.iter() {
        if let Some(name) = value.name.as_deref()
            && !name.is_empty()
        {
            *counts.entry(name).or_insert(0) += 1;
        }
    }

    let mut dups: Vec<(&str, usize)> = counts.into_iter().filter(|(_, count)| *count > 1).collect();
    dups.sort_by(|a, b| a.0.cmp(b.0));

    let mut violations: Vec<Violation> = dups
        .into_iter()
        .map(|(name, count)| Violation {
            rule_id: rule_id.to_string(),
            severity: Severity::Error,
            message: format!("value name '{}' is used by {} distinct values", name, count),
            location: ViolationLocation::Value {
                value_name: name.to_string(),
            },
        })
        .collect();
    for subgraph in graph.subgraphs.values() {
        violations.extend(check_graph_duplicate_names(subgraph, rule_id));
    }
    violations
}

/// The dataflow graph must be acyclic (ONNX_RS §8.2 `GraphAcyclicRule`). Wraps
/// the IR's own topological-order check.
pub struct GraphAcyclicRule;

impl ValidationRule for GraphAcyclicRule {
    fn id(&self) -> &str {
        "structure.graph_acyclic"
    }

    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn check(&self, model: &Model, _ctx: &ValidationContext) -> Vec<Violation> {
        check_graph_acyclic(&model.graph, &model.metadata.graph_name, self.id())
    }
}

/// Recursively check `graph` (and its control-flow subgraphs) for cycles.
fn check_graph_acyclic(graph: &Graph, graph_name: &str, rule_id: &str) -> Vec<Violation> {
    let mut violations = Vec::new();
    if graph.topological_order().is_err() {
        violations.push(Violation {
            rule_id: rule_id.to_string(),
            severity: Severity::Error,
            message: "graph contains a cycle (no valid topological order)".to_string(),
            location: ViolationLocation::Graph {
                graph_name: graph_name.to_string(),
            },
        });
    }
    for sub in graph.subgraphs.values() {
        violations.extend(check_graph_acyclic(sub, graph_name, rule_id));
    }
    violations
}

/// A stable display label for a node: its name, or `<op_type#id>` if anonymous.
fn node_label(node: &onnx_runtime_ir::Node) -> String {
    if node.name.is_empty() {
        format!("<{}#{}>", node.op_type, node.id.0)
    } else {
        node.name.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{Attribute, DataType, Node, NodeId, static_shape};

    fn nested_model(subgraph: Graph) -> Model {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 21);
        let cond = graph.create_named_value("cond", DataType::Bool, static_shape([]));
        let out = graph.create_named_value("out", DataType::Float32, static_shape([1]));
        graph.add_input(cond);
        let mut node = Node::new(NodeId(0), "If", vec![Some(cond)], vec![out]);
        node.attributes.insert(
            "then_branch".into(),
            Attribute::Graph(Box::new(subgraph.clone())),
        );
        let node_id = graph.insert_node(node);
        graph
            .subgraphs
            .insert((node_id, "then_branch".into()), subgraph);
        graph.add_output(out);
        Model::new(graph)
    }

    #[test]
    fn missing_opset_import_is_flagged() {
        let mut g = Graph::new();
        // Deliberately declare no opset import.
        let x = g.create_named_value("X", DataType::Float32, static_shape([2]));
        let y = g.create_named_value("Y", DataType::Float32, static_shape([2]));
        g.add_input(x);
        let mut node = Node::new(NodeId(0), "Relu", vec![Some(x)], vec![y]);
        node.name = "r".to_string();
        g.insert_node(node);
        g.add_output(y);

        let result = Model::new(g).validate();
        assert!(!result.is_valid());
        assert!(
            result
                .violations
                .iter()
                .any(|v| v.rule_id == "ir.opset_import_present")
        );
    }

    #[test]
    fn present_opset_import_passes() {
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 21);
        let x = g.create_named_value("X", DataType::Float32, static_shape([2]));
        let y = g.create_named_value("Y", DataType::Float32, static_shape([2]));
        g.add_input(x);
        let node = Node::new(NodeId(0), "Relu", vec![Some(x)], vec![y]);
        g.insert_node(node);
        g.add_output(y);

        let result = Model::new(g).validate();
        assert!(result.is_valid(), "{:?}", result.violations);
    }

    #[test]
    fn duplicate_value_name_is_flagged() {
        let rule = DuplicateValueNameRule;
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 21);
        // Two distinct values that share the name "dup".
        g.create_named_value("dup", DataType::Float32, static_shape([1]));
        g.create_named_value("dup", DataType::Float32, static_shape([1]));
        let model = Model::new(g);
        let violations = rule.check(&model, &ValidationContext::default());
        assert_eq!(violations.len(), 1);
        assert_eq!(
            violations[0].location,
            ViolationLocation::Value {
                value_name: "dup".to_string()
            }
        );
    }

    #[test]
    fn duplicate_value_name_inside_subgraph_is_flagged() {
        let mut subgraph = Graph::new();
        subgraph.create_named_value("nested_dup", DataType::Float32, static_shape([1]));
        subgraph.create_named_value("nested_dup", DataType::Float32, static_shape([1]));

        let result = nested_model(subgraph).validate();
        assert!(result.violations.iter().any(|violation| {
            violation.rule_id == "structure.duplicate_value_name"
                && violation.location
                    == ViolationLocation::Value {
                        value_name: "nested_dup".into(),
                    }
        }));
    }

    #[test]
    fn missing_opset_import_inside_subgraph_is_flagged() {
        let mut subgraph = Graph::new();
        let x = subgraph.create_named_value("x", DataType::Float32, static_shape([1]));
        let y = subgraph.create_named_value("y", DataType::Float32, static_shape([1]));
        subgraph.add_input(x);
        let mut node = Node::new(NodeId(0), "Custom", vec![Some(x)], vec![y]);
        node.domain = "example.custom".into();
        subgraph.insert_node(node);
        subgraph.add_output(y);

        let result = nested_model(subgraph).validate();
        assert!(
            result
                .violations
                .iter()
                .any(|violation| violation.rule_id == "ir.opset_import_present")
        );
    }

    #[test]
    fn valid_nested_model_passes() {
        let mut subgraph = Graph::new();
        let x = subgraph.create_named_value("x", DataType::Float32, static_shape([1]));
        let y = subgraph.create_named_value("y", DataType::Float32, static_shape([1]));
        subgraph.add_input(x);
        subgraph.insert_node(Node::new(NodeId(0), "Relu", vec![Some(x)], vec![y]));
        subgraph.add_output(y);

        let result = nested_model(subgraph).validate();
        assert!(result.is_valid(), "{:?}", result.violations);
    }
}

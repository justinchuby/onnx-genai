//! Built-in validation rules for this validation family.

use std::collections::{HashMap, HashSet};

use onnx_runtime_ir::{Graph, ValueId};

use super::super::{Severity, ValidationContext, ValidationRule, Violation, ViolationLocation};
use super::{node_label, node_violation, normalize_domain, value_label};
use crate::model::Model;

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

/// Graph inputs and outputs must reference live, named values (ONNX_RS §8.2
/// `InputOutputDeclaredRule`). The shared IR always carries dtype and shape for
/// a live value, so liveness plus a non-empty ONNX name establishes declaration.
pub struct InputOutputDeclaredRule;

impl ValidationRule for InputOutputDeclaredRule {
    fn id(&self) -> &str {
        "structure.input_output_declared"
    }

    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn check(&self, model: &Model, _ctx: &ValidationContext) -> Vec<Violation> {
        check_graph_io_declared(&model.graph, &model.metadata.graph_name, self.id())
    }
}

/// Every present node input must have a source, and every named node output must
/// be consumed, exported, or captured by a subgraph (ONNX_RS §8.2).
pub struct NoUnconnectedNodesRule;

impl ValidationRule for NoUnconnectedNodesRule {
    fn id(&self) -> &str {
        "structure.no_unconnected_nodes"
    }

    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn check(&self, model: &Model, _ctx: &ValidationContext) -> Vec<Violation> {
        check_graph_connections(
            &model.graph,
            &model.metadata.graph_name,
            self.id(),
            &HashSet::new(),
        )
    }
}

fn check_graph_io_declared(graph: &Graph, graph_name: &str, rule_id: &str) -> Vec<Violation> {
    let mut violations = Vec::new();
    for (kind, ids) in [("input", &graph.inputs), ("output", &graph.outputs)] {
        for &value_id in ids {
            match graph.try_value(value_id) {
                None => violations.push(Violation {
                    rule_id: rule_id.to_string(),
                    severity: Severity::Error,
                    message: format!("graph {kind} references missing value id {}", value_id.0),
                    location: ViolationLocation::Graph {
                        graph_name: graph_name.to_string(),
                    },
                }),
                Some(value) if value.name.as_deref().is_none_or(str::is_empty) => {
                    violations.push(Violation {
                        rule_id: rule_id.to_string(),
                        severity: Severity::Error,
                        message: format!(
                            "graph {kind} value id {} has no declared name",
                            value_id.0
                        ),
                        location: ViolationLocation::Value {
                            value_name: value_label(value_id, value.name.as_deref()),
                        },
                    });
                }
                Some(_) => {}
            }
        }
    }
    for subgraph in graph.subgraphs.values() {
        violations.extend(check_graph_io_declared(subgraph, graph_name, rule_id));
    }
    violations
}

fn check_graph_connections(
    graph: &Graph,
    graph_name: &str,
    rule_id: &str,
    outer_scope: &HashSet<String>,
) -> Vec<Violation> {
    let mut violations = Vec::new();
    let inputs: HashSet<ValueId> = graph.inputs.iter().copied().collect();
    let graph_outputs: HashSet<ValueId> = graph.outputs.iter().copied().collect();
    for (_node_id, node) in graph.nodes.iter() {
        for (index, value_id) in node.inputs.iter().enumerate() {
            let Some(value_id) = value_id else { continue };
            let problem = match graph.try_value(*value_id) {
                None => Some(format!(
                    "input {index} references missing value id {}",
                    value_id.0
                )),
                Some(value)
                    if value.producer.is_none()
                        && !inputs.contains(value_id)
                        && !graph.initializers.contains_key(value_id) =>
                {
                    (!value
                        .name
                        .as_ref()
                        .is_some_and(|name| outer_scope.contains(name)))
                    .then(|| {
                        format!(
                            "input {index} references '{}' which is neither a graph input, initializer, node output, nor outer-scope value",
                            value_label(*value_id, value.name.as_deref())
                        )
                    })
                }
                Some(value)
                    if value
                        .producer
                        .is_some_and(|producer| !graph.nodes.contains(producer)) =>
                {
                    Some(format!(
                        "input {index} references '{}' whose producer is missing",
                        value_label(*value_id, value.name.as_deref())
                    ))
                }
                Some(_) => None,
            };
            if let Some(message) = problem {
                violations.push(node_violation(
                    rule_id,
                    graph_name,
                    node,
                    format!("node '{}' {message}", node_label(node)),
                ));
            }
        }
        for (index, &value_id) in node.outputs.iter().enumerate() {
            let problem = match graph.try_value(value_id) {
                None => Some(format!(
                    "output {index} references missing value id {}",
                    value_id.0
                )),
                Some(value) if value.name.as_deref().is_none_or(str::is_empty) => None,
                Some(value)
                    if graph_outputs.contains(&value_id)
                        || graph.has_uses(value_id)
                        || value
                            .name
                            .as_deref()
                            .is_some_and(|name| subgraphs_capture_name(graph, name)) =>
                {
                    None
                }
                Some(value) => Some(format!(
                    "output {index} '{}' is neither consumed, a graph output, nor captured by a subgraph",
                    value_label(value_id, value.name.as_deref())
                )),
            };
            if let Some(message) = problem {
                violations.push(node_violation(
                    rule_id,
                    graph_name,
                    node,
                    format!("node '{}' {message}", node_label(node)),
                ));
            }
        }
    }
    let mut visible_names = outer_scope.clone();
    visible_names.extend(
        graph
            .values
            .values()
            .filter_map(|value| value.name.as_ref())
            .filter(|name| !name.is_empty())
            .cloned(),
    );
    for subgraph in graph.subgraphs.values() {
        violations.extend(check_graph_connections(
            subgraph,
            graph_name,
            rule_id,
            &visible_names,
        ));
    }
    violations
}

fn subgraphs_capture_name(graph: &Graph, name: &str) -> bool {
    graph.subgraphs.values().any(|subgraph| {
        let locally_defined = subgraph.values.values().any(|value| {
            value.name.as_deref() == Some(name)
                && (value.producer.is_some()
                    || subgraph.inputs.contains(&value.id)
                    || subgraph.initializers.contains_key(&value.id))
        });
        let directly_captured = subgraph.nodes.values().any(|node| {
            node.input_values().any(|value_id| {
                subgraph.try_value(value_id).is_some_and(|value| {
                    value.name.as_deref() == Some(name)
                        && value.producer.is_none()
                        && !subgraph.inputs.contains(&value_id)
                        && !subgraph.initializers.contains_key(&value_id)
                })
            })
        });
        directly_captured || (!locally_defined && subgraphs_capture_name(subgraph, name))
    })
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
            message: format!("value name '{name}' is used by {count} distinct values"),
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

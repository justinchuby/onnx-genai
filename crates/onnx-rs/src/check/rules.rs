//! Built-in validation rules for this first wave (ONNX_RS §8.2).
//!
//! * [`MissingOpsetImportRule`] — every operator domain used by a node must have
//!   a matching `opset_import` (§8.2 "IR rules").
//! * [`DuplicateValueNameRule`] — no two values may share a name (§8.2
//!   "structural rules": unique value names / single producer).
//! * [`GraphAcyclicRule`] — the dataflow graph must be acyclic (§8.2).
//! * [`SchemaNodeConformsRule`] — nodes match their resolved operator schema.

use std::collections::HashMap;

use onnx_runtime_ir::{Attribute, Graph};

use super::{Severity, ValidationContext, ValidationRule, Violation, ViolationLocation};
use crate::model::Model;
use crate::schema::{AttributeType, OpSchema};

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

/// Each node must resolve to an opset-compatible schema and conform to its
/// input/output and attribute declarations (ONNX_RS §8.2).
pub struct SchemaNodeConformsRule;

impl ValidationRule for SchemaNodeConformsRule {
    fn id(&self) -> &str {
        "schema.node_conforms"
    }

    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn check(&self, model: &Model, ctx: &ValidationContext) -> Vec<Violation> {
        check_graph_schemas(
            &model.graph,
            &model.graph.opset_imports,
            &model.metadata.graph_name,
            ctx,
            self.id(),
        )
    }
}

fn check_graph_schemas(
    graph: &Graph,
    opset_imports: &HashMap<String, u64>,
    graph_name: &str,
    ctx: &ValidationContext,
    rule_id: &str,
) -> Vec<Violation> {
    let mut violations = Vec::new();
    for (_node_id, node) in graph.nodes.iter() {
        let Some(opset) = imported_opset(opset_imports, &node.domain) else {
            continue;
        };
        let Some(schema) = ctx.schemas().lookup(&node.op_type, &node.domain, opset) else {
            let reason = if ctx.schemas().contains_operator(&node.op_type, &node.domain) {
                format!("has no schema valid at opset {opset}")
            } else {
                "is not present in the schema registry".to_string()
            };
            violations.push(node_violation(
                rule_id,
                graph_name,
                node,
                format!(
                    "node '{}' ({}::{}) {reason}",
                    node_label(node),
                    normalize_domain(&node.domain),
                    node.op_type
                ),
            ));
            continue;
        };
        check_node_arity(schema, node, graph_name, rule_id, &mut violations);
        check_node_attributes(schema, node, graph_name, rule_id, &mut violations);
    }
    for subgraph in graph.subgraphs.values() {
        violations.extend(check_graph_schemas(
            subgraph,
            opset_imports,
            graph_name,
            ctx,
            rule_id,
        ));
    }
    violations
}

fn imported_opset(opsets: &HashMap<String, u64>, domain: &str) -> Option<u64> {
    if domain.is_empty() || domain == "ai.onnx" {
        opsets.get("").or_else(|| opsets.get("ai.onnx")).copied()
    } else {
        opsets.get(domain).copied()
    }
}

fn check_node_arity(
    schema: &OpSchema,
    node: &onnx_runtime_ir::Node,
    graph_name: &str,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    let min_inputs = schema
        .inputs
        .iter()
        .filter(|spec| !spec.optional && !spec.variadic)
        .count();
    let max_inputs =
        (!schema.inputs.iter().any(|spec| spec.variadic)).then_some(schema.inputs.len());
    if node.inputs.len() < min_inputs || max_inputs.is_some_and(|max| node.inputs.len() > max) {
        violations.push(node_violation(
            rule_id,
            graph_name,
            node,
            arity_message("input", node.inputs.len(), min_inputs, max_inputs),
        ));
    }
    for (index, spec) in schema.inputs.iter().enumerate() {
        if !spec.optional && !spec.variadic && node.inputs.get(index).is_some_and(Option::is_none) {
            violations.push(node_violation(
                rule_id,
                graph_name,
                node,
                format!(
                    "required input '{}' at position {index} is omitted",
                    spec.name
                ),
            ));
        }
    }

    let min_outputs = schema
        .outputs
        .iter()
        .filter(|spec| !spec.optional && !spec.variadic)
        .count();
    let max_outputs =
        (!schema.outputs.iter().any(|spec| spec.variadic)).then_some(schema.outputs.len());
    if node.outputs.len() < min_outputs || max_outputs.is_some_and(|max| node.outputs.len() > max) {
        violations.push(node_violation(
            rule_id,
            graph_name,
            node,
            arity_message("output", node.outputs.len(), min_outputs, max_outputs),
        ));
    }
}

fn arity_message(kind: &str, actual: usize, min: usize, max: Option<usize>) -> String {
    match max {
        Some(max) if min == max => {
            format!("has {actual} {kind}s but schema requires exactly {min}")
        }
        Some(max) => format!("has {actual} {kind}s but schema permits {min}..={max}"),
        None => format!("has {actual} {kind}s but schema requires at least {min}"),
    }
}

fn check_node_attributes(
    schema: &OpSchema,
    node: &onnx_runtime_ir::Node,
    graph_name: &str,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    for spec in &schema.attributes {
        match node.attributes.get(&spec.name) {
            None if spec.required && spec.default.is_none() => {
                violations.push(node_violation(
                    rule_id,
                    graph_name,
                    node,
                    format!("required attribute '{}' is missing", spec.name),
                ));
            }
            Some(value) if !attribute_matches(value, spec.attr_type) => {
                violations.push(node_violation(
                    rule_id,
                    graph_name,
                    node,
                    format!(
                        "attribute '{}' has type {} but schema requires {:?}",
                        spec.name,
                        attribute_name(value),
                        spec.attr_type
                    ),
                ));
            }
            _ => {}
        }
    }
    for (name, value) in &node.attributes {
        if !schema.attributes.iter().any(|spec| spec.name == *name) {
            violations.push(node_violation(
                rule_id,
                graph_name,
                node,
                format!(
                    "attribute '{name}' of type {} is not declared by the schema",
                    attribute_name(value)
                ),
            ));
        }
    }
}

fn attribute_matches(value: &Attribute, expected: AttributeType) -> bool {
    matches!(
        (value, expected),
        (Attribute::Int(_), AttributeType::Int)
            | (Attribute::Float(_), AttributeType::Float)
            | (Attribute::String(_), AttributeType::String)
            | (Attribute::Tensor(_), AttributeType::Tensor)
            | (Attribute::Graph(_), AttributeType::Graph)
            | (Attribute::SparseTensor(_), AttributeType::SparseTensor)
            | (Attribute::TypeProto(_), AttributeType::TypeProto)
            | (Attribute::Ints(_), AttributeType::Ints)
            | (Attribute::Floats(_), AttributeType::Floats)
            | (Attribute::Strings(_), AttributeType::Strings)
            | (Attribute::Graphs(_), AttributeType::Graphs)
    )
}

fn attribute_name(value: &Attribute) -> &'static str {
    match value {
        Attribute::Int(_) => "int",
        Attribute::Float(_) => "float",
        Attribute::String(_) => "string",
        Attribute::Tensor(_) => "tensor",
        Attribute::SparseTensor(_) => "sparse_tensor",
        Attribute::Graph(_) => "graph",
        Attribute::Graphs(_) => "graphs",
        Attribute::TypeProto(_) => "type_proto",
        Attribute::Ints(_) => "ints",
        Attribute::Floats(_) => "floats",
        Attribute::Strings(_) => "strings",
    }
}

fn node_violation(
    rule_id: &str,
    graph_name: &str,
    node: &onnx_runtime_ir::Node,
    message: String,
) -> Violation {
    Violation {
        rule_id: rule_id.to_string(),
        severity: Severity::Error,
        message,
        location: ViolationLocation::Node {
            graph_name: graph_name.to_string(),
            node_name: node_label(node),
        },
    }
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
    use crate::schema::SchemaRegistry;
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

    fn one_node_model(op_type: &str, inputs: usize, outputs: usize) -> Model {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 21);
        let input_ids = (0..inputs)
            .map(|index| {
                graph.create_named_value(
                    format!("input{index}"),
                    DataType::Float32,
                    static_shape([1]),
                )
            })
            .collect::<Vec<_>>();
        let output_ids = (0..outputs)
            .map(|index| {
                graph.create_named_value(
                    format!("output{index}"),
                    DataType::Float32,
                    static_shape([1]),
                )
            })
            .collect::<Vec<_>>();
        graph.insert_node(Node::new(
            NodeId(0),
            op_type,
            input_ids.iter().copied().map(Some).collect(),
            output_ids,
        ));
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

    #[test]
    fn schema_rule_accepts_conforming_node() {
        let result = one_node_model("Add", 2, 1).validate();
        assert!(result.is_valid(), "{:?}", result.violations);
    }

    #[test]
    fn schema_rule_checks_arity_required_and_typed_attributes() {
        let mut model = one_node_model("Gemm", 1, 2);
        let node_id = model.graph.nodes.iter().next().unwrap().0;
        let node = model.graph.node_mut(node_id);
        node.attributes.insert("alpha".into(), Attribute::Int(1));
        node.attributes.insert("unknown".into(), Attribute::Int(1));
        let result = model.validate();
        let messages = result
            .violations
            .iter()
            .filter(|violation| violation.rule_id == "schema.node_conforms")
            .map(|violation| violation.message.as_str())
            .collect::<Vec<_>>();
        assert!(messages.iter().any(|message| message.contains("1 inputs")));
        assert!(messages.iter().any(|message| message.contains("2 outputs")));
        assert!(
            messages
                .iter()
                .any(|message| message.contains("requires Float"))
        );
        assert!(
            messages
                .iter()
                .any(|message| message.contains("not declared"))
        );
    }

    #[test]
    fn schema_rule_checks_required_attributes_and_opset_range() {
        let yaml = r#"
domain: ""
name: NeedsAxis
since_version: 13
until_version: 20
attributes:
  - { name: axis, type: int, required: true }
inputs: [{ name: X, type_str: T }]
outputs: [{ name: Y, type_str: T }]
"#;
        let mut schemas = SchemaRegistry::new();
        schemas.load_yaml(yaml).unwrap();
        let mut model = one_node_model("NeedsAxis", 1, 1);
        model.graph.opset_imports.insert(String::new(), 13);
        let checker = super::super::OnnxChecker::with_schema_registry(schemas.clone());
        let result = checker.check(&model);
        assert!(result.violations.iter().any(|violation| {
            violation.rule_id == "schema.node_conforms"
                && violation.message.contains("required attribute 'axis'")
        }));

        model.graph.opset_imports.insert(String::new(), 21);
        let result = super::super::OnnxChecker::with_schema_registry(schemas).check(&model);
        assert!(result.violations.iter().any(|violation| {
            violation.rule_id == "schema.node_conforms"
                && violation.message.contains("no schema valid at opset 21")
        }));
    }

    #[test]
    fn schema_rule_reports_unregistered_operator() {
        let result = one_node_model("NotARealOp", 1, 1).validate();
        assert!(result.violations.iter().any(|violation| {
            violation.rule_id == "schema.node_conforms"
                && violation
                    .message
                    .contains("not present in the schema registry")
        }));
    }
}

//! Built-in validation rules (ONNX_RS §8.2).
//!
//! * [`MissingOpsetImportRule`] — every operator domain used by a node must have
//!   a matching `opset_import` (§8.2 "IR rules").
//! * [`DuplicateValueNameRule`] — no two values may share a name (§8.2
//!   "structural rules": unique value names / single producer).
//! * [`GraphAcyclicRule`] — the dataflow graph must be acyclic (§8.2).
//! * [`SchemaNodeConformsRule`] — nodes match their resolved operator schema.
//! * [`InputOutputDeclaredRule`] — graph inputs and outputs are named, live values.
//! * [`NoUnconnectedNodesRule`] — node inputs resolve to graph sources or producers.
//! * [`TypeConstraintSatisfiedRule`] — node value types satisfy schema constraints.
//! * [`InitializerTypeMatchesDeclaredRule`] — initializer and value dtypes agree.
//! * [`IrVersionSupportedRule`] — the model declares a valid ONNX IR version.
//! * [`MultiDeviceConfigurationRule`] — IR v11+ distributed annotations are
//!   internally consistent.

use std::collections::{HashMap, HashSet};

use onnx_runtime_ir::{Attribute, Graph, ValueId};
use onnx_runtime_loader::proto::onnx::{
    DeviceConfigurationProto, GraphProto, ModelProto, NodeProto, TensorShapeProto, type_proto,
};

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
                        || !value.consumers.is_empty()
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

/// Node input/output element types must satisfy and consistently bind the type
/// variables declared by the resolved operator schema (ONNX_RS §8.2).
pub struct TypeConstraintSatisfiedRule;

impl ValidationRule for TypeConstraintSatisfiedRule {
    fn id(&self) -> &str {
        "schema.type_constraint_satisfied"
    }

    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn check(&self, model: &Model, ctx: &ValidationContext) -> Vec<Violation> {
        check_graph_type_constraints(
            &model.graph,
            &model.graph.opset_imports,
            &model.metadata.graph_name,
            ctx,
            self.id(),
        )
    }
}

fn check_graph_type_constraints(
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
            continue;
        };
        let mut bindings = HashMap::new();
        for (type_param, value_id) in schema_value_bindings(schema, node) {
            let Some(value) = graph.try_value(value_id) else {
                continue;
            };
            if !graph.value_type_is_known(value_id) {
                continue;
            }
            let Some(constraint) = schema
                .type_constraints
                .iter()
                .find(|constraint| constraint.type_param == type_param)
            else {
                continue;
            };
            if !constraint.allowed.contains(&value.dtype) {
                violations.push(node_violation(
                    rule_id,
                    graph_name,
                    node,
                    format!(
                        "type parameter '{}' does not allow {:?} for value '{}'",
                        type_param,
                        value.dtype,
                        value_label(value_id, value.name.as_deref())
                    ),
                ));
                continue;
            }
            match bindings.entry(type_param) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(value.dtype);
                }
                std::collections::hash_map::Entry::Occupied(entry)
                    if *entry.get() != value.dtype =>
                {
                    violations.push(node_violation(
                        rule_id,
                        graph_name,
                        node,
                        format!(
                            "type parameter '{}' is bound to both {:?} and {:?}",
                            type_param,
                            entry.get(),
                            value.dtype
                        ),
                    ));
                }
                std::collections::hash_map::Entry::Occupied(_) => {}
            }
        }
    }
    for subgraph in graph.subgraphs.values() {
        violations.extend(check_graph_type_constraints(
            subgraph,
            opset_imports,
            graph_name,
            ctx,
            rule_id,
        ));
    }
    violations
}

fn schema_value_bindings<'a>(
    schema: &'a OpSchema,
    node: &'a onnx_runtime_ir::Node,
) -> Vec<(&'a str, ValueId)> {
    let mut bindings = Vec::new();
    for (index, spec) in schema.inputs.iter().enumerate() {
        let slots = if spec.variadic {
            node.inputs.get(index..).unwrap_or_default()
        } else {
            node.inputs
                .get(index..index.saturating_add(1))
                .unwrap_or_default()
        };
        bindings.extend(
            slots
                .iter()
                .filter_map(|slot| *slot)
                .map(|value_id| (spec.type_str.as_str(), value_id)),
        );
    }
    for (index, spec) in schema.outputs.iter().enumerate() {
        let values = if spec.variadic {
            node.outputs.get(index..).unwrap_or_default()
        } else {
            node.outputs
                .get(index..index.saturating_add(1))
                .unwrap_or_default()
        };
        bindings.extend(
            values
                .iter()
                .copied()
                .map(|value_id| (spec.type_str.as_str(), value_id)),
        );
    }
    bindings
}

/// Initializer tensor element types must match their graph value declarations
/// (ONNX_RS §8.2 `InitializerTypeMatchesDeclaredRule`).
pub struct InitializerTypeMatchesDeclaredRule;

impl ValidationRule for InitializerTypeMatchesDeclaredRule {
    fn id(&self) -> &str {
        "type.initializer_matches_declared"
    }

    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn check(&self, model: &Model, _ctx: &ValidationContext) -> Vec<Violation> {
        check_graph_initializer_types(&model.graph, self.id())
    }
}

fn check_graph_initializer_types(graph: &Graph, rule_id: &str) -> Vec<Violation> {
    let mut violations = Vec::new();
    for (&value_id, initializer) in &graph.initializers {
        match graph.try_value(value_id) {
            None => violations.push(Violation {
                rule_id: rule_id.to_string(),
                severity: Severity::Error,
                message: format!("initializer references missing value id {}", value_id.0),
                location: ViolationLocation::Value {
                    value_name: format!("<value#{}>", value_id.0),
                },
            }),
            Some(value) if value.dtype != initializer.dtype() => violations.push(Violation {
                rule_id: rule_id.to_string(),
                severity: Severity::Error,
                message: format!(
                    "initializer '{}' has dtype {:?} but its declared value type is {:?}",
                    value_label(value_id, value.name.as_deref()),
                    initializer.dtype(),
                    value.dtype
                ),
                location: ViolationLocation::Value {
                    value_name: value_label(value_id, value.name.as_deref()),
                },
            }),
            Some(value)
                if graph.value_shape_is_known(value_id)
                    && initializer_shape_mismatch(&value.shape, initializer.dims()) =>
            {
                violations.push(Violation {
                    rule_id: rule_id.to_string(),
                    severity: Severity::Error,
                    message: format!(
                        "initializer '{}' has shape {:?} but its declared value shape is {:?}",
                        value_label(value_id, value.name.as_deref()),
                        initializer.dims(),
                        value.shape
                    ),
                    location: ViolationLocation::Value {
                        value_name: value_label(value_id, value.name.as_deref()),
                    },
                })
            }
            Some(_) => {}
        }
    }
    for subgraph in graph.subgraphs.values() {
        violations.extend(check_graph_initializer_types(subgraph, rule_id));
    }
    violations
}

fn initializer_shape_mismatch(declared: &[onnx_runtime_ir::Dim], actual: &[usize]) -> bool {
    declared.len() != actual.len()
        || declared
            .iter()
            .zip(actual)
            .any(|(declared, &actual)| declared.as_static().is_some_and(|dim| dim != actual))
}

/// The ONNX IR version is required and starts at 1. There is deliberately no
/// upper ceiling: newer IR versions are accepted unless a concrete incompatibility
/// is known (ONNX_RS §8.2 `IrVersionSupportedRule`).
pub struct IrVersionSupportedRule;

impl ValidationRule for IrVersionSupportedRule {
    fn id(&self) -> &str {
        "ir.version_supported"
    }

    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn check(&self, model: &Model, _ctx: &ValidationContext) -> Vec<Violation> {
        if model.metadata.ir_version >= 1 {
            return Vec::new();
        }
        vec![Violation {
            rule_id: self.id().to_string(),
            severity: self.severity(),
            message: format!(
                "ir_version {} is invalid; ONNX IR versions start at 1",
                model.metadata.ir_version
            ),
            location: ViolationLocation::Model,
        }]
    }
}

/// Validate the IR v11+ multi-device configuration and sharding annotations.
///
/// This rule operates on the retained protobuf because the execution IR
/// deliberately treats distributed annotations as backend hints and does not
/// project them onto runtime nodes.
pub struct MultiDeviceConfigurationRule;

impl ValidationRule for MultiDeviceConfigurationRule {
    fn id(&self) -> &str {
        "multidevice.configuration_valid"
    }

    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn check(&self, model: &Model, _ctx: &ValidationContext) -> Vec<Violation> {
        let Some(proto) = model.retained_proto() else {
            return Vec::new();
        };
        check_multi_device_model(proto, self.id())
    }
}

fn check_multi_device_model(model: &ModelProto, rule_id: &str) -> Vec<Violation> {
    let mut violations = Vec::new();
    if model.ir_version < 11 && model_has_multi_device_annotations(model) {
        violations.push(model_violation(
            rule_id,
            format!(
                "multi-device annotations require IR version 11 or newer, but model declares {}",
                model.ir_version
            ),
        ));
    }
    let mut configurations = HashMap::new();
    for configuration in &model.configuration {
        if configuration.name.is_empty() {
            violations.push(model_violation(
                rule_id,
                "device configuration name must be present",
            ));
        } else if configurations
            .insert(configuration.name.as_str(), configuration)
            .is_some()
        {
            violations.push(model_violation(
                rule_id,
                format!(
                    "device configuration name '{}' is not unique",
                    configuration.name
                ),
            ));
        }
        if configuration.num_devices <= 0 {
            violations.push(model_violation(
                rule_id,
                format!(
                    "device configuration '{}' must declare a positive num_devices",
                    configuration.name
                ),
            ));
        }
        if !configuration.device.is_empty()
            && configuration.device.len() != configuration.num_devices.max(0) as usize
        {
            violations.push(model_violation(
                rule_id,
                format!(
                    "device configuration '{}' names {} devices but num_devices is {}",
                    configuration.name,
                    configuration.device.len(),
                    configuration.num_devices
                ),
            ));
        }
    }

    if let Some(graph) = &model.graph {
        check_multi_device_graph(graph, &configurations, rule_id, &mut violations);
    }
    for training in &model.training_info {
        if let Some(graph) = &training.initialization {
            check_multi_device_graph(graph, &configurations, rule_id, &mut violations);
        }
        if let Some(graph) = &training.algorithm {
            check_multi_device_graph(graph, &configurations, rule_id, &mut violations);
        }
    }
    for function in &model.functions {
        for node in &function.node {
            check_multi_device_node(
                node,
                &function.name,
                None,
                &configurations,
                rule_id,
                &mut violations,
            );
        }
    }
    violations
}

fn check_multi_device_graph<'a>(
    graph: &GraphProto,
    configurations: &HashMap<&'a str, &'a DeviceConfigurationProto>,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    let graph_name = graph.name.as_str();
    for node in &graph.node {
        check_multi_device_node(
            node,
            graph_name,
            Some(graph),
            configurations,
            rule_id,
            violations,
        );
        for attribute in &node.attribute {
            if let Some(subgraph) = &attribute.g {
                check_multi_device_graph(subgraph, configurations, rule_id, violations);
            }
            for subgraph in &attribute.graphs {
                check_multi_device_graph(subgraph, configurations, rule_id, violations);
            }
        }
    }
}

fn check_multi_device_node<'a>(
    node: &NodeProto,
    graph_name: &str,
    graph: Option<&GraphProto>,
    configurations: &HashMap<&'a str, &'a DeviceConfigurationProto>,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    for device_configuration in &node.device_configurations {
        if !configurations.contains_key(device_configuration.configuration_id.as_str()) {
            violations.push(proto_node_violation(
                rule_id,
                graph_name,
                node,
                format!(
                    "device configuration id '{}' does not match a ModelProto.configuration name",
                    device_configuration.configuration_id
                ),
            ));
        }

        for sharding in &device_configuration.sharding_spec {
            if sharding.tensor_name.is_empty()
                || !node
                    .input
                    .iter()
                    .chain(&node.output)
                    .any(|name| name == &sharding.tensor_name)
            {
                violations.push(proto_node_violation(
                    rule_id,
                    graph_name,
                    node,
                    format!(
                        "sharding tensor '{}' is not a named node input or output",
                        sharding.tensor_name
                    ),
                ));
            }

            let rank = graph.and_then(|graph| tensor_rank(graph, &sharding.tensor_name));
            for sharded_dim in &sharding.sharded_dim {
                if let Some(rank) = rank {
                    let rank = rank as i64;
                    if sharded_dim.axis < -rank || sharded_dim.axis >= rank {
                        violations.push(proto_node_violation(
                            rule_id,
                            graph_name,
                            node,
                            format!(
                                "sharding axis {} for tensor '{}' is outside [-{}, {}]",
                                sharded_dim.axis,
                                sharding.tensor_name,
                                rank,
                                rank - 1
                            ),
                        ));
                    }
                }
                for simple in &sharded_dim.simple_sharding {
                    if simple.num_shards <= 0 {
                        violations.push(proto_node_violation(
                            rule_id,
                            graph_name,
                            node,
                            format!(
                                "sharded axis {} for tensor '{}' must declare a positive num_shards",
                                sharded_dim.axis, sharding.tensor_name
                            ),
                        ));
                    }
                }
            }
        }
    }
}

fn tensor_rank(graph: &GraphProto, name: &str) -> Option<usize> {
    graph
        .initializer
        .iter()
        .find(|tensor| tensor.name == name)
        .map(|tensor| tensor.dims.len())
        .or_else(|| {
            graph
                .sparse_initializer
                .iter()
                .find(|tensor| {
                    tensor
                        .values
                        .as_ref()
                        .is_some_and(|values| values.name == name)
                })
                .map(|tensor| tensor.dims.len())
        })
        .or_else(|| {
            graph
                .input
                .iter()
                .chain(&graph.output)
                .chain(&graph.value_info)
                .find(|value| value.name == name)
                .and_then(|value| value.r#type.as_ref())
                .and_then(type_rank)
        })
}

fn model_has_multi_device_annotations(model: &ModelProto) -> bool {
    !model.configuration.is_empty()
        || model
            .graph
            .as_ref()
            .is_some_and(graph_has_device_annotations)
        || model.training_info.iter().any(|training| {
            training
                .initialization
                .as_ref()
                .is_some_and(graph_has_device_annotations)
                || training
                    .algorithm
                    .as_ref()
                    .is_some_and(graph_has_device_annotations)
        })
        || model
            .functions
            .iter()
            .any(|function| function.node.iter().any(node_has_device_annotations))
}

fn graph_has_device_annotations(graph: &GraphProto) -> bool {
    graph.node.iter().any(node_has_device_annotations)
}

fn node_has_device_annotations(node: &NodeProto) -> bool {
    !node.device_configurations.is_empty()
        || node.attribute.iter().any(|attribute| {
            attribute
                .g
                .as_ref()
                .is_some_and(graph_has_device_annotations)
                || attribute.graphs.iter().any(graph_has_device_annotations)
        })
}

fn type_rank(value: &onnx_runtime_loader::proto::onnx::TypeProto) -> Option<usize> {
    match value.value.as_ref()? {
        type_proto::Value::TensorType(tensor) => tensor.shape.as_ref().map(shape_rank),
        type_proto::Value::SparseTensorType(tensor) => tensor.shape.as_ref().map(shape_rank),
        _ => None,
    }
}

fn shape_rank(shape: &TensorShapeProto) -> usize {
    shape.dim.len()
}

fn model_violation(rule_id: &str, message: impl Into<String>) -> Violation {
    Violation {
        rule_id: rule_id.to_string(),
        severity: Severity::Error,
        message: message.into(),
        location: ViolationLocation::Model,
    }
}

fn proto_node_violation(
    rule_id: &str,
    graph_name: &str,
    node: &NodeProto,
    message: impl Into<String>,
) -> Violation {
    Violation {
        rule_id: rule_id.to_string(),
        severity: Severity::Error,
        message: message.into(),
        location: ViolationLocation::Node {
            graph_name: graph_name.to_string(),
            node_name: if node.name.is_empty() {
                format!("<{}>", node.op_type)
            } else {
                node.name.clone()
            },
        },
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
        .map(|spec| {
            if spec.variadic {
                spec.min_arity
            } else {
                usize::from(!spec.optional)
            }
        })
        .sum();
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
        .map(|spec| {
            if spec.variadic {
                spec.min_arity
            } else {
                usize::from(!spec.optional)
            }
        })
        .sum();
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
            | (Attribute::Tensors(_), AttributeType::Tensors)
            | (Attribute::SparseTensors(_), AttributeType::SparseTensors)
            | (Attribute::TypeProtos(_), AttributeType::TypeProtos)
    )
}

fn attribute_name(value: &Attribute) -> &'static str {
    match value {
        Attribute::Int(_) => "int",
        Attribute::Float(_) => "float",
        Attribute::String(_) => "string",
        Attribute::Tensor(_) => "tensor",
        Attribute::Tensors(_) => "tensors",
        Attribute::SparseTensor(_) => "sparse_tensor",
        Attribute::SparseTensors(_) => "sparse_tensors",
        Attribute::Graph(_) => "graph",
        Attribute::Graphs(_) => "graphs",
        Attribute::TypeProto(_) => "type_proto",
        Attribute::TypeProtos(_) => "type_protos",
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

fn value_label(value_id: ValueId, name: Option<&str>) -> String {
    match name {
        Some(name) if !name.is_empty() => name.to_string(),
        _ => format!("<value#{}>", value_id.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::SchemaRegistry;
    use onnx_runtime_ir::{
        Attribute, DataType, Dim, Node, NodeId, TensorData, WeightRef, static_shape,
    };

    fn if_model(subgraph: Graph, output_count: usize, include_else_branch: bool) -> Model {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 21);
        let cond = graph.create_named_value("cond", DataType::Bool, static_shape([]));
        let outputs = (0..output_count)
            .map(|index| {
                graph.create_named_value(
                    format!("out{index}"),
                    DataType::Float32,
                    static_shape([1]),
                )
            })
            .collect::<Vec<_>>();
        graph.add_input(cond);
        let mut node = Node::new(NodeId(0), "If", vec![Some(cond)], outputs.clone());
        node.attributes.insert(
            "then_branch".into(),
            Attribute::Graph(Box::new(subgraph.clone())),
        );
        if include_else_branch {
            node.attributes.insert(
                "else_branch".into(),
                Attribute::Graph(Box::new(subgraph.clone())),
            );
        }
        let node_id = graph.insert_node(node);
        graph
            .subgraphs
            .insert((node_id, "then_branch".into()), subgraph.clone());
        if include_else_branch {
            graph
                .subgraphs
                .insert((node_id, "else_branch".into()), subgraph);
        }
        for output in outputs {
            graph.add_output(output);
        }
        Model::new(graph)
    }

    fn nested_model(subgraph: Graph) -> Model {
        if_model(subgraph, 1, true)
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
            output_ids.clone(),
        ));
        for input in input_ids {
            graph.add_input(input);
        }
        for output in output_ids {
            graph.add_output(output);
        }
        Model::new(graph)
    }

    fn assert_error(rule_id: &str, violations: &[Violation], location: ViolationLocation) {
        assert_eq!(violations.len(), 1, "{violations:?}");
        assert_eq!(violations[0].rule_id, rule_id);
        assert_eq!(violations[0].severity, Severity::Error);
        assert_eq!(violations[0].location, location);
    }

    #[test]
    fn declared_graph_io_passes() {
        let rule = InputOutputDeclaredRule;
        let mut graph = Graph::new();
        let input = graph.create_named_value("X", DataType::Float32, static_shape([1]));
        let output = graph.create_named_value("Y", DataType::Float32, static_shape([1]));
        graph.add_input(input);
        graph.add_output(output);

        assert!(
            rule.check(&Model::new(graph), &ValidationContext::default())
                .is_empty()
        );
    }

    #[test]
    fn unnamed_graph_io_is_flagged() {
        let rule = InputOutputDeclaredRule;
        let mut graph = Graph::new();
        let input = graph.create_value(DataType::Float32, static_shape([1]));
        graph.add_input(input);

        let violations = rule.check(&Model::new(graph), &ValidationContext::default());
        assert_error(
            rule.id(),
            &violations,
            ViolationLocation::Value {
                value_name: format!("<value#{}>", input.0),
            },
        );
    }

    #[test]
    fn connected_node_inputs_pass() {
        let rule = NoUnconnectedNodesRule;
        let mut graph = Graph::new();
        let input = graph.create_named_value("X", DataType::Float32, static_shape([1]));
        let output = graph.create_named_value("Y", DataType::Float32, static_shape([1]));
        let final_output = graph.create_named_value("Z", DataType::Float32, static_shape([1]));
        graph.add_input(input);
        let node_id = graph.insert_node(Node::new(
            NodeId(0),
            "Relu",
            vec![Some(input)],
            vec![output],
        ));
        graph.insert_node(Node::new(
            NodeId(0),
            "Identity",
            vec![Some(output)],
            vec![final_output],
        ));
        let mut subgraph = Graph::new();
        let capture = subgraph.create_named_value("X", DataType::Float32, static_shape([1]));
        let subgraph_output =
            subgraph.create_named_value("subgraph_output", DataType::Float32, static_shape([1]));
        subgraph.insert_node(Node::new(
            NodeId(0),
            "Relu",
            vec![Some(capture)],
            vec![subgraph_output],
        ));
        subgraph.add_output(subgraph_output);
        graph.subgraphs.insert((node_id, "body".into()), subgraph);
        graph.add_output(final_output);

        assert!(
            rule.check(&Model::new(graph), &ValidationContext::default())
                .is_empty()
        );
    }

    #[test]
    fn omitted_optional_node_output_passes() {
        let rule = NoUnconnectedNodesRule;
        let mut graph = Graph::new();
        let input = graph.create_named_value("X", DataType::Float32, static_shape([1]));
        let omitted = graph.create_value(DataType::Float32, Vec::new());
        graph.add_input(input);
        graph.insert_node(Node::new(
            NodeId(0),
            "OptionalOutput",
            vec![Some(input)],
            vec![omitted],
        ));

        assert!(
            rule.check(&Model::new(graph), &ValidationContext::default())
                .is_empty()
        );
    }

    #[test]
    fn captured_node_output_passes() {
        let rule = NoUnconnectedNodesRule;
        let mut graph = Graph::new();
        let input = graph.create_named_value("X", DataType::Float32, static_shape([1]));
        let captured = graph.create_named_value("captured", DataType::Float32, static_shape([1]));
        graph.add_input(input);
        let node_id = graph.insert_node(Node::new(
            NodeId(0),
            "Relu",
            vec![Some(input)],
            vec![captured],
        ));

        let mut subgraph = Graph::new();
        let capture = subgraph.create_named_value("captured", DataType::Float32, static_shape([1]));
        let output = subgraph.create_named_value("Y", DataType::Float32, static_shape([1]));
        subgraph.insert_node(Node::new(
            NodeId(0),
            "Relu",
            vec![Some(capture)],
            vec![output],
        ));
        subgraph.add_output(output);
        graph.subgraphs.insert((node_id, "body".into()), subgraph);

        assert!(
            rule.check(&Model::new(graph), &ValidationContext::default())
                .is_empty()
        );
    }

    #[test]
    fn dangling_node_output_is_flagged() {
        let rule = NoUnconnectedNodesRule;
        let mut graph = Graph::new();
        let input = graph.create_named_value("X", DataType::Float32, static_shape([1]));
        let dangling = graph.create_named_value("dangling", DataType::Float32, static_shape([1]));
        graph.add_input(input);
        let mut node = Node::new(NodeId(0), "Relu", vec![Some(input)], vec![dangling]);
        node.name = "relu".into();
        graph.insert_node(node);

        let violations = rule.check(&Model::new(graph), &ValidationContext::default());
        assert_error(
            rule.id(),
            &violations,
            ViolationLocation::Node {
                graph_name: String::new(),
                node_name: "relu".into(),
            },
        );
    }

    #[test]
    fn undefined_node_input_is_flagged() {
        let rule = NoUnconnectedNodesRule;
        let mut graph = Graph::new();
        let dangling = graph.create_named_value("dangling", DataType::Float32, static_shape([1]));
        let output = graph.create_named_value("Y", DataType::Float32, static_shape([1]));
        let mut node = Node::new(NodeId(0), "Relu", vec![Some(dangling)], vec![output]);
        node.name = "relu".into();
        graph.insert_node(node);
        graph.add_output(output);

        let violations = rule.check(&Model::new(graph), &ValidationContext::default());
        assert_error(
            rule.id(),
            &violations,
            ViolationLocation::Node {
                graph_name: String::new(),
                node_name: "relu".into(),
            },
        );
    }

    #[test]
    fn schema_type_constraints_accept_consistent_allowed_types() {
        let rule = TypeConstraintSatisfiedRule;
        let model = one_node_model("Add", 2, 1);

        assert!(rule.check(&model, &ValidationContext::default()).is_empty());
    }

    #[test]
    fn schema_type_constraints_skip_unknown_placeholder_types() {
        let rule = TypeConstraintSatisfiedRule;
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 21);
        let input = graph.create_named_value("X", DataType::Int64, static_shape([1]));
        let output = graph.create_named_value("Y", DataType::Float32, Vec::new());
        graph.mark_value_type_unknown(output);
        graph.mark_value_shape_unknown(output);
        graph.add_input(input);
        graph.insert_node(Node::new(
            NodeId(0),
            "Identity",
            vec![Some(input)],
            vec![output],
        ));
        graph.add_output(output);

        assert!(
            rule.check(&Model::new(graph), &ValidationContext::default())
                .is_empty()
        );
    }

    #[test]
    fn schema_type_constraint_violation_is_flagged() {
        let rule = TypeConstraintSatisfiedRule;
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 21);
        let input = graph.create_named_value("X", DataType::Int64, static_shape([1]));
        let output = graph.create_named_value("Y", DataType::Int64, static_shape([1]));
        graph.add_input(input);
        let mut node = Node::new(NodeId(0), "Relu", vec![Some(input)], vec![output]);
        node.name = "relu".into();
        graph.insert_node(node);
        graph.add_output(output);

        let violations = rule.check(&Model::new(graph), &ValidationContext::default());
        assert!(
            violations.iter().all(|violation| {
                violation.rule_id == rule.id()
                    && violation.severity == Severity::Error
                    && violation.location
                        == ViolationLocation::Node {
                            graph_name: String::new(),
                            node_name: "relu".into(),
                        }
            }),
            "{violations:?}"
        );
        assert_eq!(violations.len(), 2);
    }

    #[test]
    fn matching_initializer_type_passes() {
        let rule = InitializerTypeMatchesDeclaredRule;
        let mut graph = Graph::new();
        let value = graph.create_named_value("W", DataType::Float32, static_shape([1]));
        graph.set_initializer(
            value,
            WeightRef::Inline(TensorData::from_raw(DataType::Float32, vec![1], vec![0; 4])),
        );

        assert!(
            rule.check(&Model::new(graph), &ValidationContext::default())
                .is_empty()
        );
    }

    #[test]
    fn initializer_symbolic_and_unknown_declared_shapes_pass() {
        let rule = InitializerTypeMatchesDeclaredRule;
        let mut graph = Graph::new();
        let batch = graph.create_symbol(Some("batch".into()));
        let symbolic = graph.create_named_value(
            "symbolic",
            DataType::Float32,
            vec![Dim::Symbolic(batch), Dim::Static(3)],
        );
        graph.set_initializer(
            symbolic,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float32,
                vec![2, 3],
                vec![0; 24],
            )),
        );
        let unknown = graph.create_named_value("unknown", DataType::Float32, Vec::new());
        graph.mark_value_shape_unknown(unknown);
        graph.set_initializer(
            unknown,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float32,
                vec![4],
                vec![0; 16],
            )),
        );

        assert!(
            rule.check(&Model::new(graph), &ValidationContext::default())
                .is_empty()
        );
    }

    #[test]
    fn mismatched_initializer_shape_is_flagged() {
        let rule = InitializerTypeMatchesDeclaredRule;
        for dims in [vec![2, 4], vec![2, 3, 1]] {
            let mut graph = Graph::new();
            let value = graph.create_named_value("W", DataType::Float32, static_shape([2, 3]));
            graph.set_initializer(
                value,
                WeightRef::Inline(TensorData::from_raw(
                    DataType::Float32,
                    dims.clone(),
                    Vec::new(),
                )),
            );

            let violations = rule.check(&Model::new(graph), &ValidationContext::default());
            assert_error(
                rule.id(),
                &violations,
                ViolationLocation::Value {
                    value_name: "W".into(),
                },
            );
        }
    }

    #[test]
    fn mismatched_initializer_type_is_flagged() {
        let rule = InitializerTypeMatchesDeclaredRule;
        let mut graph = Graph::new();
        let value = graph.create_named_value("W", DataType::Float32, static_shape([1]));
        graph.set_initializer(
            value,
            WeightRef::Inline(TensorData::from_raw(DataType::Int64, vec![1], vec![0; 8])),
        );

        let violations = rule.check(&Model::new(graph), &ValidationContext::default());
        assert_error(
            rule.id(),
            &violations,
            ViolationLocation::Value {
                value_name: "W".into(),
            },
        );
    }

    #[test]
    fn present_and_future_ir_versions_pass() {
        let rule = IrVersionSupportedRule;
        for ir_version in [1, 10, 999] {
            let mut model = Model::new(Graph::new());
            model.metadata.ir_version = ir_version;
            assert!(
                rule.check(&model, &ValidationContext::default()).is_empty(),
                "ir_version {ir_version}"
            );
        }
    }

    #[test]
    fn absent_ir_version_is_flagged() {
        let rule = IrVersionSupportedRule;
        let mut model = Model::new(Graph::new());
        model.metadata.ir_version = 0;

        let violations = rule.check(&model, &ValidationContext::default());
        assert_error(rule.id(), &violations, ViolationLocation::Model);
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
    fn if_requires_else_branch() {
        let result = if_model(Graph::new(), 1, false).validate();
        assert!(result.violations.iter().any(|violation| {
            violation.rule_id == "schema.node_conforms"
                && violation
                    .message
                    .contains("required attribute 'else_branch'")
        }));
    }

    #[test]
    fn if_variadic_outputs_require_at_least_one_value() {
        let result = if_model(Graph::new(), 0, true).validate();
        assert!(result.violations.iter().any(|violation| {
            violation.rule_id == "schema.node_conforms"
                && violation
                    .message
                    .contains("has 0 outputs but schema requires at least 1")
        }));

        let result = if_model(Graph::new(), 1, true).validate();
        assert!(result.is_valid(), "{:?}", result.violations);
    }

    #[test]
    fn schema_rule_accepts_conforming_node() {
        let result = one_node_model("Add", 2, 1).validate();
        assert!(result.is_valid(), "{:?}", result.violations);
    }

    #[test]
    fn common_builtin_arity_boundaries_pass() {
        for (op_type, inputs, outputs) in [
            ("MatMul", 2, 1),
            ("Gemm", 2, 1),
            ("Gemm", 3, 1),
            ("Add", 2, 1),
            ("Relu", 1, 1),
            ("Conv", 2, 1),
            ("Conv", 3, 1),
            ("Mul", 2, 1),
            ("Identity", 1, 1),
        ] {
            let result = one_node_model(op_type, inputs, outputs).validate();
            assert!(
                result.is_valid(),
                "{op_type}({inputs}, {outputs}): {:?}",
                result.violations
            );
        }
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

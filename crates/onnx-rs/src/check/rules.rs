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
    AttributeProto, DeviceConfigurationProto, FunctionProto, GraphProto, ModelProto, NodeProto,
    SparseTensorProto, StringStringEntryProto, TensorProto, TensorShapeProto, TypeProto,
    ValueInfoProto, attribute_proto, tensor_proto, type_proto,
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
            if let Some(expected) = concrete_tensor_dtype(type_param) {
                if value.dtype != expected {
                    violations.push(node_violation(
                        rule_id,
                        graph_name,
                        node,
                        format!(
                            "concrete type '{}' requires {:?} for value '{}', found {:?}",
                            type_param,
                            expected,
                            value_label(value_id, value.name.as_deref()),
                            value.dtype
                        ),
                    ));
                }
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

fn concrete_tensor_dtype(type_str: &str) -> Option<onnx_runtime_ir::DataType> {
    let dtype = type_str.strip_prefix("tensor(")?.strip_suffix(')')?;
    Some(match dtype {
        "float" => onnx_runtime_ir::DataType::Float32,
        "double" => onnx_runtime_ir::DataType::Float64,
        "float16" => onnx_runtime_ir::DataType::Float16,
        "bfloat16" => onnx_runtime_ir::DataType::BFloat16,
        "uint8" => onnx_runtime_ir::DataType::Uint8,
        "uint16" => onnx_runtime_ir::DataType::Uint16,
        "uint32" => onnx_runtime_ir::DataType::Uint32,
        "uint64" => onnx_runtime_ir::DataType::Uint64,
        "int8" => onnx_runtime_ir::DataType::Int8,
        "int16" => onnx_runtime_ir::DataType::Int16,
        "int32" => onnx_runtime_ir::DataType::Int32,
        "int64" => onnx_runtime_ir::DataType::Int64,
        "bool" => onnx_runtime_ir::DataType::Bool,
        "string" => onnx_runtime_ir::DataType::String,
        "complex64" => onnx_runtime_ir::DataType::Complex64,
        "complex128" => onnx_runtime_ir::DataType::Complex128,
        "float8e4m3fn" => onnx_runtime_ir::DataType::Float8E4M3FN,
        "float8e4m3fnuz" => onnx_runtime_ir::DataType::Float8E4M3FNUZ,
        "float8e5m2" => onnx_runtime_ir::DataType::Float8E5M2,
        "float8e5m2fnuz" => onnx_runtime_ir::DataType::Float8E5M2FNUZ,
        "uint4" => onnx_runtime_ir::DataType::Uint4,
        "int4" => onnx_runtime_ir::DataType::Int4,
        "float4e2m1" => onnx_runtime_ir::DataType::Float4E2M1,
        "float8e8m0" => onnx_runtime_ir::DataType::Float8E8M0,
        "uint2" => onnx_runtime_ir::DataType::Uint2,
        "int2" => onnx_runtime_ir::DataType::Int2,
        _ => return None,
    })
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

/// The ONNX IR version is required and must not exceed the version implemented
/// by this v1.20-bound checker (IR 13).
pub struct IrVersionSupportedRule;

impl ValidationRule for IrVersionSupportedRule {
    fn id(&self) -> &str {
        "ir.version_supported"
    }

    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn check(&self, model: &Model, _ctx: &ValidationContext) -> Vec<Violation> {
        if (1..=13).contains(&model.metadata.ir_version) {
            return Vec::new();
        }
        let message = if model.metadata.ir_version < 1 {
            format!(
                "ir_version {} is invalid; ONNX IR versions start at 1",
                model.metadata.ir_version
            )
        } else {
            format!(
                "ir_version {} is newer than this ONNX v1.20 checker (IR 13)",
                model.metadata.ir_version
            )
        };
        vec![Violation {
            rule_id: self.id().to_string(),
            severity: self.severity(),
            message,
            location: ViolationLocation::Model,
        }]
    }
}

/// Validate fields and invariants whose legality changed at a particular ONNX
/// IR version. Training-only fields are intentionally ignored.
pub struct IrVersionFeatureRule;

impl ValidationRule for IrVersionFeatureRule {
    fn id(&self) -> &str {
        "ir.version_gated_features"
    }

    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn check(&self, model: &Model, _ctx: &ValidationContext) -> Vec<Violation> {
        let ir_version = model.metadata.ir_version;
        let mut violations = Vec::new();
        if ir_version >= 3 && model.graph.opset_imports.is_empty() {
            violations.push(model_violation(
                self.id(),
                format!("model with IR version {ir_version} must specify opset_import"),
            ));
        } else if ir_version < 3 && !model.graph.opset_imports.is_empty() {
            violations.push(model_violation(
                self.id(),
                "models with IR version below 3 must not specify opset_import",
            ));
        }
        let Some(proto) = model.retained_proto() else {
            return violations;
        };
        if ir_version < 8 && !proto.functions.is_empty() {
            violations.push(model_violation(
                self.id(),
                "ModelProto.functions requires IR version 8 or newer",
            ));
        }
        if let Some(graph) = &proto.graph {
            check_graph_ir_features(graph, ir_version, self.id(), &mut violations);
        }
        for function in &proto.functions {
            check_function_ir_features(function, ir_version, self.id(), &mut violations);
        }
        violations
    }
}

fn check_graph_ir_features(
    graph: &GraphProto,
    ir_version: i64,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    let location = ViolationLocation::Graph {
        graph_name: graph.name.clone(),
    };
    if ir_version <= 3 {
        let inputs = graph
            .input
            .iter()
            .map(|value| value.name.as_str())
            .collect::<HashSet<_>>();
        for initializer in &graph.initializer {
            if !inputs.contains(initializer.name.as_str()) {
                violations.push(Violation {
                    rule_id: rule_id.to_string(),
                    severity: Severity::Error,
                    message: format!(
                        "initializer '{}' must also be a graph input for IR version {ir_version}",
                        initializer.name
                    ),
                    location: location.clone(),
                });
            }
        }
    }
    if ir_version < 5 && !graph.quantization_annotation.is_empty() {
        violations.push(Violation {
            rule_id: rule_id.to_string(),
            severity: Severity::Error,
            message: "GraphProto.quantization_annotation requires IR version 5 or newer".into(),
            location: location.clone(),
        });
    }
    if ir_version < 6 && !graph.sparse_initializer.is_empty() {
        violations.push(Violation {
            rule_id: rule_id.to_string(),
            severity: Severity::Error,
            message: "GraphProto.sparse_initializer requires IR version 6 or newer".into(),
            location: location.clone(),
        });
    }
    if ir_version < 10 && !graph.metadata_props.is_empty() {
        violations.push(Violation {
            rule_id: rule_id.to_string(),
            severity: Severity::Error,
            message: "GraphProto.metadata_props requires IR version 10 or newer".into(),
            location: location.clone(),
        });
    }
    for value in graph
        .input
        .iter()
        .chain(&graph.output)
        .chain(&graph.value_info)
    {
        check_value_ir_features(value, ir_version, rule_id, violations);
    }
    for tensor in &graph.initializer {
        check_tensor_ir_features(tensor, ir_version, rule_id, violations);
    }
    for sparse in &graph.sparse_initializer {
        check_sparse_ir_features(sparse, ir_version, rule_id, violations);
    }
    for node in &graph.node {
        check_node_ir_features(node, ir_version, &graph.name, rule_id, violations);
    }
}

fn check_function_ir_features(
    function: &FunctionProto,
    ir_version: i64,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    if ir_version < 9 && !function.attribute_proto.is_empty() {
        violations.push(model_violation(
            rule_id,
            format!(
                "FunctionProto '{}' attribute_proto requires IR version 9 or newer",
                function.name
            ),
        ));
    }
    if ir_version < 10 {
        if !function.overload.is_empty() {
            violations.push(model_violation(
                rule_id,
                format!(
                    "FunctionProto '{}' overload requires IR version 10 or newer",
                    function.name
                ),
            ));
        }
        if !function.value_info.is_empty() {
            violations.push(model_violation(
                rule_id,
                format!(
                    "FunctionProto '{}' value_info requires IR version 10 or newer",
                    function.name
                ),
            ));
        }
        if !function.metadata_props.is_empty() {
            violations.push(model_violation(
                rule_id,
                format!(
                    "FunctionProto '{}' metadata_props requires IR version 10 or newer",
                    function.name
                ),
            ));
        }
    }
    for value in &function.value_info {
        check_value_ir_features(value, ir_version, rule_id, violations);
    }
    for attribute in &function.attribute_proto {
        check_attribute_ir_features(
            attribute,
            ir_version,
            ViolationLocation::Model,
            rule_id,
            violations,
        );
    }
    for node in &function.node {
        check_node_ir_features(node, ir_version, &function.name, rule_id, violations);
    }
}

fn check_node_ir_features(
    node: &NodeProto,
    ir_version: i64,
    graph_name: &str,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    let location = ViolationLocation::Node {
        graph_name: graph_name.to_string(),
        node_name: proto_node_label(node),
    };
    if ir_version < 10 && (!node.overload.is_empty() || !node.metadata_props.is_empty()) {
        violations.push(Violation {
            rule_id: rule_id.to_string(),
            severity: Severity::Error,
            message: "NodeProto overload and metadata_props require IR version 10 or newer".into(),
            location: location.clone(),
        });
    }
    for attribute in &node.attribute {
        check_attribute_ir_features(attribute, ir_version, location.clone(), rule_id, violations);
        if let Some(graph) = &attribute.g {
            check_graph_ir_features(graph, ir_version, rule_id, violations);
        }
        for graph in &attribute.graphs {
            check_graph_ir_features(graph, ir_version, rule_id, violations);
        }
    }
    if ir_version < 11 && !node.device_configurations.is_empty() {
        violations.push(Violation {
            rule_id: rule_id.to_string(),
            severity: Severity::Error,
            message: "NodeProto.device_configurations requires IR version 11 or newer".into(),
            location,
        });
    }
}

fn check_attribute_ir_features(
    attribute: &AttributeProto,
    ir_version: i64,
    location: ViolationLocation,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    if ir_version < 6 && (attribute.sparse_tensor.is_some() || !attribute.sparse_tensors.is_empty())
    {
        violations.push(Violation {
            rule_id: rule_id.to_string(),
            severity: Severity::Error,
            message: "sparse tensor attributes require IR version 6 or newer".into(),
            location: location.clone(),
        });
    }
    if let Some(tensor) = &attribute.t {
        check_tensor_ir_features(tensor, ir_version, rule_id, violations);
    }
    for tensor in &attribute.tensors {
        check_tensor_ir_features(tensor, ir_version, rule_id, violations);
    }
    if let Some(sparse) = &attribute.sparse_tensor {
        check_sparse_ir_features(sparse, ir_version, rule_id, violations);
    }
    for sparse in &attribute.sparse_tensors {
        check_sparse_ir_features(sparse, ir_version, rule_id, violations);
    }
    if let Some(value) = &attribute.tp {
        check_type_ir_features(value, ir_version, location.clone(), rule_id, violations);
    }
    for value in &attribute.type_protos {
        check_type_ir_features(value, ir_version, location.clone(), rule_id, violations);
    }
}

fn check_value_ir_features(
    value: &ValueInfoProto,
    ir_version: i64,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    let location = ViolationLocation::Value {
        value_name: value.name.clone(),
    };
    if ir_version < 10 && !value.metadata_props.is_empty() {
        violations.push(Violation {
            rule_id: rule_id.to_string(),
            severity: Severity::Error,
            message: "ValueInfoProto.metadata_props requires IR version 10 or newer".into(),
            location: location.clone(),
        });
    }
    if let Some(value_type) = &value.r#type {
        check_type_ir_features(value_type, ir_version, location, rule_id, violations);
    }
}

fn check_type_ir_features(
    value: &TypeProto,
    ir_version: i64,
    location: ViolationLocation,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    match value.value.as_ref() {
        Some(type_proto::Value::TensorType(tensor)) => {
            check_dtype_ir_feature(tensor.elem_type, ir_version, location, rule_id, violations);
        }
        Some(type_proto::Value::SparseTensorType(tensor)) => {
            if ir_version < 8 {
                violations.push(Violation {
                    rule_id: rule_id.to_string(),
                    severity: Severity::Error,
                    message: "TypeProto.SparseTensor requires IR version 8 or newer".into(),
                    location: location.clone(),
                });
            }
            check_dtype_ir_feature(tensor.elem_type, ir_version, location, rule_id, violations);
        }
        Some(type_proto::Value::OptionalType(optional)) => {
            if ir_version < 8 {
                violations.push(Violation {
                    rule_id: rule_id.to_string(),
                    severity: Severity::Error,
                    message: "TypeProto.Optional requires IR version 8 or newer".into(),
                    location: location.clone(),
                });
            }
            if let Some(elem_type) = &optional.elem_type {
                check_type_ir_features(elem_type, ir_version, location, rule_id, violations);
            }
        }
        Some(type_proto::Value::SequenceType(sequence)) => {
            if let Some(elem_type) = &sequence.elem_type {
                check_type_ir_features(elem_type, ir_version, location, rule_id, violations);
            }
        }
        Some(type_proto::Value::MapType(map)) => {
            if let Some(value_type) = &map.value_type {
                check_type_ir_features(value_type, ir_version, location, rule_id, violations);
            }
        }
        Some(type_proto::Value::OpaqueType(_)) | None => {}
    }
}

fn check_sparse_ir_features(
    sparse: &SparseTensorProto,
    ir_version: i64,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    if let Some(values) = &sparse.values {
        check_tensor_ir_features(values, ir_version, rule_id, violations);
    }
    if let Some(indices) = &sparse.indices {
        check_tensor_ir_features(indices, ir_version, rule_id, violations);
    }
}

fn check_tensor_ir_features(
    tensor: &TensorProto,
    ir_version: i64,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    let location = ViolationLocation::Value {
        value_name: tensor.name.clone(),
    };
    if ir_version < 10 && !tensor.metadata_props.is_empty() {
        violations.push(Violation {
            rule_id: rule_id.to_string(),
            severity: Severity::Error,
            message: "TensorProto.metadata_props requires IR version 10 or newer".into(),
            location: location.clone(),
        });
    }
    check_dtype_ir_feature(tensor.data_type, ir_version, location, rule_id, violations);
}

fn check_dtype_ir_feature(
    dtype: i32,
    ir_version: i64,
    location: ViolationLocation,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    let required = match tensor_proto::DataType::try_from(dtype).ok() {
        Some(tensor_proto::DataType::Bfloat16) => 4,
        Some(
            tensor_proto::DataType::Float8e4m3fn
            | tensor_proto::DataType::Float8e4m3fnuz
            | tensor_proto::DataType::Float8e5m2
            | tensor_proto::DataType::Float8e5m2fnuz,
        ) => 9,
        Some(tensor_proto::DataType::Uint4 | tensor_proto::DataType::Int4) => 10,
        Some(tensor_proto::DataType::Float4e2m1) => 11,
        Some(tensor_proto::DataType::Float8e8m0) => 12,
        Some(tensor_proto::DataType::Uint2 | tensor_proto::DataType::Int2) => 13,
        _ => return,
    };
    if ir_version < required {
        violations.push(Violation {
            rule_id: rule_id.to_string(),
            severity: Severity::Error,
            message: format!(
                "TensorProto data type {dtype} requires IR version {required} or newer"
            ),
            location,
        });
    }
}

/// Validate model-local functions using the v1.20 checker topology/import
/// rules plus signature, default-attribute, and unique-ID rules.
pub struct FunctionProtoValidityRule;

impl ValidationRule for FunctionProtoValidityRule {
    fn id(&self) -> &str {
        "proto.function_valid"
    }

    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn check(&self, model: &Model, ctx: &ValidationContext) -> Vec<Violation> {
        let Some(proto) = model.retained_proto() else {
            return Vec::new();
        };
        if proto.ir_version < 8 {
            return Vec::new();
        }
        check_model_functions(proto, ctx, self.id())
    }
}

type FunctionKey = (String, String, String);

fn function_key(function: &FunctionProto) -> FunctionKey {
    (
        normalize_domain(&function.domain).to_string(),
        function.name.clone(),
        function.overload.clone(),
    )
}

fn node_function_key(node: &NodeProto) -> FunctionKey {
    (
        normalize_domain(&node.domain).to_string(),
        node.op_type.clone(),
        node.overload.clone(),
    )
}

fn check_model_functions(
    model: &ModelProto,
    ctx: &ValidationContext,
    rule_id: &str,
) -> Vec<Violation> {
    let mut violations = Vec::new();
    let mut functions = HashMap::new();
    for function in &model.functions {
        let key = function_key(function);
        if functions.insert(key.clone(), function).is_some() {
            violations.push(model_violation(
                rule_id,
                format!(
                    "model-local function '{}::{}' overload '{}' is not unique",
                    key.0, key.1, key.2
                ),
            ));
        }
    }

    let mut model_opsets = opset_map(&model.opset_import);
    for function in &model.functions {
        for import in &function.opset_import {
            model_opsets
                .entry(normalize_domain(&import.domain).to_string())
                .or_insert(import.version);
        }
    }

    for function in &model.functions {
        check_function_proto(function, &model_opsets, ctx, rule_id, &mut violations);
    }
    check_function_recursion(&functions, rule_id, &mut violations);
    violations
}

fn opset_map(
    imports: &[onnx_runtime_loader::proto::onnx::OperatorSetIdProto],
) -> HashMap<String, i64> {
    imports
        .iter()
        .map(|import| (normalize_domain(&import.domain).to_string(), import.version))
        .collect()
}

fn check_function_proto(
    function: &FunctionProto,
    model_opsets: &HashMap<String, i64>,
    ctx: &ValidationContext,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    if function.name.is_empty() {
        violations.push(model_violation(
            rule_id,
            "FunctionProto.name must be non-empty",
        ));
    }
    check_unique_function_names(
        &function.input,
        "input",
        &function.name,
        rule_id,
        violations,
    );
    check_unique_function_names(
        &function.output,
        "output",
        &function.name,
        rule_id,
        violations,
    );
    check_unique_function_names(
        &function.attribute,
        "attribute",
        &function.name,
        rule_id,
        violations,
    );

    let required_attrs = function
        .attribute
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    let mut default_attrs = HashSet::new();
    for attribute in &function.attribute_proto {
        if !default_attrs.insert(attribute.name.as_str()) {
            violations.push(model_violation(
                rule_id,
                format!(
                    "function '{}' has duplicate default attribute '{}'",
                    function.name, attribute.name
                ),
            ));
        }
        if required_attrs.contains(attribute.name.as_str()) {
            violations.push(model_violation(
                rule_id,
                format!(
                    "function '{}' attribute '{}' appears in both attribute and attribute_proto",
                    function.name, attribute.name
                ),
            ));
        }
    }
    let formal_attrs = required_attrs
        .union(&default_attrs)
        .copied()
        .collect::<HashSet<_>>();
    for node in &function.node {
        check_function_attribute_refs(node, &formal_attrs, &function.name, rule_id, violations);
    }

    let function_opsets = opset_map(&function.opset_import);
    let mut local = HashSet::new();
    for input in &function.input {
        local.insert(input.clone());
    }
    check_proto_nodes(
        &function.node,
        &mut local,
        &HashSet::new(),
        &function.name,
        model_opsets,
        &function_opsets,
        ctx,
        rule_id,
        violations,
    );
}

fn check_unique_function_names(
    names: &[String],
    kind: &str,
    function_name: &str,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    let mut seen = HashSet::new();
    for name in names {
        if !seen.insert(name.as_str()) {
            violations.push(model_violation(
                rule_id,
                format!("function '{function_name}' has duplicate {kind} '{name}'"),
            ));
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn check_proto_nodes(
    nodes: &[NodeProto],
    local: &mut HashSet<String>,
    outer: &HashSet<String>,
    graph_name: &str,
    model_opsets: &HashMap<String, i64>,
    function_opsets: &HashMap<String, i64>,
    ctx: &ValidationContext,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    for node in nodes {
        let location = ViolationLocation::Node {
            graph_name: graph_name.to_string(),
            node_name: proto_node_label(node),
        };
        for input in &node.input {
            if !input.is_empty() && !local.contains(input) && !outer.contains(input) {
                violations.push(Violation {
                    rule_id: rule_id.to_string(),
                    severity: Severity::Error,
                    message: format!(
                        "input '{input}' is neither a function/graph input nor an output of a previous node"
                    ),
                    location: location.clone(),
                });
            }
        }
        if node.op_type.is_empty() {
            violations.push(Violation {
                rule_id: rule_id.to_string(),
                severity: Severity::Error,
                message: "NodeProto.op_type must be non-empty".into(),
                location: location.clone(),
            });
        }
        if node.input.is_empty() && node.output.is_empty() {
            violations.push(Violation {
                rule_id: rule_id.to_string(),
                severity: Severity::Error,
                message: "function-body node must have at least one input or output".into(),
                location: location.clone(),
            });
        }
        check_function_node_schema(
            node,
            graph_name,
            model_opsets,
            function_opsets,
            ctx,
            rule_id,
            violations,
        );
        let mut visible = outer.clone();
        visible.extend(local.iter().cloned());
        for attribute in &node.attribute {
            if let Some(graph) = &attribute.g {
                check_function_subgraph(
                    graph,
                    &visible,
                    model_opsets,
                    function_opsets,
                    ctx,
                    rule_id,
                    violations,
                );
            }
            for graph in &attribute.graphs {
                check_function_subgraph(
                    graph,
                    &visible,
                    model_opsets,
                    function_opsets,
                    ctx,
                    rule_id,
                    violations,
                );
            }
        }
        for output in &node.output {
            if output.is_empty() {
                continue;
            }
            if local.contains(output) || outer.contains(output) {
                violations.push(Violation {
                    rule_id: rule_id.to_string(),
                    severity: Severity::Error,
                    message: format!(
                        "output '{output}' violates single static assignment in function body"
                    ),
                    location: location.clone(),
                });
            } else {
                local.insert(output.clone());
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn check_function_subgraph(
    graph: &GraphProto,
    outer: &HashSet<String>,
    model_opsets: &HashMap<String, i64>,
    function_opsets: &HashMap<String, i64>,
    ctx: &ValidationContext,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    if graph.name.is_empty() {
        violations.push(Violation {
            rule_id: rule_id.to_string(),
            severity: Severity::Error,
            message: "function-body subgraph name must be non-empty".into(),
            location: ViolationLocation::Graph {
                graph_name: graph.name.clone(),
            },
        });
    }
    let mut local = HashSet::new();
    for input in &graph.input {
        if !local.insert(input.name.clone()) {
            violations.push(Violation {
                rule_id: rule_id.to_string(),
                severity: Severity::Error,
                message: format!("subgraph input '{}' is not unique", input.name),
                location: ViolationLocation::Graph {
                    graph_name: graph.name.clone(),
                },
            });
        }
    }
    for initializer in &graph.initializer {
        local.insert(initializer.name.clone());
    }
    for sparse in &graph.sparse_initializer {
        if let Some(values) = &sparse.values {
            local.insert(values.name.clone());
        }
    }
    check_proto_nodes(
        &graph.node,
        &mut local,
        outer,
        &graph.name,
        model_opsets,
        function_opsets,
        ctx,
        rule_id,
        violations,
    );
    for output in &graph.output {
        if !local.contains(&output.name) {
            violations.push(Violation {
                rule_id: rule_id.to_string(),
                severity: Severity::Error,
                message: format!(
                    "subgraph output '{}' is not defined in the subgraph",
                    output.name
                ),
                location: ViolationLocation::Graph {
                    graph_name: graph.name.clone(),
                },
            });
        }
    }
}

fn check_function_node_schema(
    node: &NodeProto,
    graph_name: &str,
    model_opsets: &HashMap<String, i64>,
    function_opsets: &HashMap<String, i64>,
    ctx: &ValidationContext,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    let domain = normalize_domain(&node.domain);
    let location = ViolationLocation::Node {
        graph_name: graph_name.to_string(),
        node_name: proto_node_label(node),
    };
    let Some(&function_version) = function_opsets.get(domain) else {
        violations.push(Violation {
            rule_id: rule_id.to_string(),
            severity: Severity::Error,
            message: format!("no opset import for function-body domain '{domain}'"),
            location,
        });
        return;
    };
    if let Some(&model_version) = model_opsets.get(domain)
        && model_version != function_version
    {
        let function_schema = u64::try_from(function_version)
            .ok()
            .and_then(|version| ctx.schemas().lookup(&node.op_type, domain, version));
        let model_schema = u64::try_from(model_version)
            .ok()
            .and_then(|version| ctx.schemas().lookup(&node.op_type, domain, version));
        if !(function_schema.is_none() && model_schema.is_none())
            && function_schema.map(|schema| schema.since_version)
                != model_schema.map(|schema| schema.since_version)
        {
            violations.push(Violation {
                rule_id: rule_id.to_string(),
                severity: Severity::Error,
                message: format!(
                    "function opset {function_version} for '{domain}::{}' is incompatible with model opset {model_version}",
                    node.op_type
                ),
                location: location.clone(),
            });
        }
    }

    let Some(version) = u64::try_from(function_version).ok() else {
        violations.push(Violation {
            rule_id: rule_id.to_string(),
            severity: Severity::Error,
            message: format!(
                "function-body domain '{domain}' has invalid negative opset version {function_version}"
            ),
            location,
        });
        return;
    };
    let Some(schema) = ctx.schemas().lookup(&node.op_type, domain, version) else {
        if domain == "ai.onnx" || domain == "ai.onnx.ml" {
            violations.push(Violation {
                rule_id: rule_id.to_string(),
                severity: Severity::Error,
                message: format!(
                    "no schema registered for function-body op '{domain}::{}' at opset {version}",
                    node.op_type
                ),
                location,
            });
        }
        return;
    };
    check_proto_node_arity(schema, node, location.clone(), rule_id, violations);
    check_proto_node_attributes(schema, node, location, rule_id, violations);
}

fn check_proto_node_arity(
    schema: &OpSchema,
    node: &NodeProto,
    location: ViolationLocation,
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
    if node.input.len() < min_inputs || max_inputs.is_some_and(|max| node.input.len() > max) {
        violations.push(Violation {
            rule_id: rule_id.to_string(),
            severity: Severity::Error,
            message: arity_message("input", node.input.len(), min_inputs, max_inputs),
            location: location.clone(),
        });
    }
    for (index, spec) in schema.inputs.iter().enumerate() {
        if !spec.optional && !spec.variadic && node.input.get(index).is_some_and(String::is_empty) {
            violations.push(Violation {
                rule_id: rule_id.to_string(),
                severity: Severity::Error,
                message: format!(
                    "required input '{}' at position {index} is omitted",
                    spec.name
                ),
                location: location.clone(),
            });
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
    if node.output.len() < min_outputs || max_outputs.is_some_and(|max| node.output.len() > max) {
        violations.push(Violation {
            rule_id: rule_id.to_string(),
            severity: Severity::Error,
            message: arity_message("output", node.output.len(), min_outputs, max_outputs),
            location,
        });
    }
}

fn check_proto_node_attributes(
    schema: &OpSchema,
    node: &NodeProto,
    location: ViolationLocation,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    for spec in &schema.attributes {
        match node
            .attribute
            .iter()
            .find(|attribute| attribute.name == spec.name)
        {
            None if spec.required && spec.default.is_none() => violations.push(Violation {
                rule_id: rule_id.to_string(),
                severity: Severity::Error,
                message: format!("required attribute '{}' is missing", spec.name),
                location: location.clone(),
            }),
            Some(attribute) if attribute_proto_type(attribute.r#type) != Some(spec.attr_type) => {
                violations.push(Violation {
                    rule_id: rule_id.to_string(),
                    severity: Severity::Error,
                    message: format!(
                        "attribute '{}' has discriminator {} but schema requires {:?}",
                        spec.name, attribute.r#type, spec.attr_type
                    ),
                    location: location.clone(),
                });
            }
            _ => {}
        }
    }
    for attribute in &node.attribute {
        if !schema
            .attributes
            .iter()
            .any(|spec| spec.name == attribute.name)
        {
            violations.push(Violation {
                rule_id: rule_id.to_string(),
                severity: Severity::Error,
                message: format!(
                    "attribute '{}' is not declared by the schema",
                    attribute.name
                ),
                location: location.clone(),
            });
        }
    }
}

fn attribute_proto_type(value: i32) -> Option<AttributeType> {
    Some(
        match attribute_proto::AttributeType::try_from(value).ok()? {
            attribute_proto::AttributeType::Float => AttributeType::Float,
            attribute_proto::AttributeType::Int => AttributeType::Int,
            attribute_proto::AttributeType::String => AttributeType::String,
            attribute_proto::AttributeType::Tensor => AttributeType::Tensor,
            attribute_proto::AttributeType::Graph => AttributeType::Graph,
            attribute_proto::AttributeType::SparseTensor => AttributeType::SparseTensor,
            attribute_proto::AttributeType::TypeProto => AttributeType::TypeProto,
            attribute_proto::AttributeType::Floats => AttributeType::Floats,
            attribute_proto::AttributeType::Ints => AttributeType::Ints,
            attribute_proto::AttributeType::Strings => AttributeType::Strings,
            attribute_proto::AttributeType::Tensors => AttributeType::Tensors,
            attribute_proto::AttributeType::Graphs => AttributeType::Graphs,
            attribute_proto::AttributeType::SparseTensors => AttributeType::SparseTensors,
            attribute_proto::AttributeType::TypeProtos => AttributeType::TypeProtos,
            attribute_proto::AttributeType::Undefined => return None,
        },
    )
}

fn check_function_attribute_refs(
    node: &NodeProto,
    formal_attrs: &HashSet<&str>,
    function_name: &str,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    for attribute in &node.attribute {
        if !attribute.ref_attr_name.is_empty()
            && !formal_attrs.contains(attribute.ref_attr_name.as_str())
        {
            violations.push(model_violation(
                rule_id,
                format!(
                    "function '{function_name}' node attribute '{}' references undeclared function attribute '{}'",
                    attribute.name, attribute.ref_attr_name
                ),
            ));
        }
        if let Some(graph) = &attribute.g {
            for nested in &graph.node {
                check_function_attribute_refs(
                    nested,
                    formal_attrs,
                    function_name,
                    rule_id,
                    violations,
                );
            }
        }
        for graph in &attribute.graphs {
            for nested in &graph.node {
                check_function_attribute_refs(
                    nested,
                    formal_attrs,
                    function_name,
                    rule_id,
                    violations,
                );
            }
        }
    }
}

fn check_function_recursion(
    functions: &HashMap<FunctionKey, &FunctionProto>,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    fn callees(
        node: &NodeProto,
        functions: &HashMap<FunctionKey, &FunctionProto>,
        out: &mut Vec<FunctionKey>,
    ) {
        let key = node_function_key(node);
        if functions.contains_key(&key) {
            out.push(key);
        }
        for attribute in &node.attribute {
            if let Some(graph) = &attribute.g {
                for nested in &graph.node {
                    callees(nested, functions, out);
                }
            }
            for graph in &attribute.graphs {
                for nested in &graph.node {
                    callees(nested, functions, out);
                }
            }
        }
    }

    fn visit(
        key: &FunctionKey,
        functions: &HashMap<FunctionKey, &FunctionProto>,
        states: &mut HashMap<FunctionKey, u8>,
        stack: &mut Vec<FunctionKey>,
        rule_id: &str,
        violations: &mut Vec<Violation>,
    ) {
        match states.get(key).copied().unwrap_or(0) {
            2 => return,
            1 => {
                let start = stack.iter().position(|item| item == key).unwrap_or(0);
                let mut chain = stack[start..]
                    .iter()
                    .map(|item| format!("{}::{}:{}", item.0, item.1, item.2))
                    .collect::<Vec<_>>();
                chain.push(format!("{}::{}:{}", key.0, key.1, key.2));
                violations.push(model_violation(
                    rule_id,
                    format!(
                        "model-local functions are recursive: {}",
                        chain.join(" -> ")
                    ),
                ));
                return;
            }
            _ => {}
        }
        states.insert(key.clone(), 1);
        stack.push(key.clone());
        let mut next = Vec::new();
        if let Some(function) = functions.get(key) {
            for node in &function.node {
                callees(node, functions, &mut next);
            }
        }
        for callee in next {
            visit(&callee, functions, states, stack, rule_id, violations);
        }
        stack.pop();
        states.insert(key.clone(), 2);
    }

    let mut states = HashMap::new();
    let mut stack = Vec::new();
    for key in functions.keys() {
        visit(key, functions, &mut states, &mut stack, rule_id, violations);
    }
}

fn proto_node_label(node: &NodeProto) -> String {
    if node.name.is_empty() {
        format!("<{}>", node.op_type)
    } else {
        node.name.clone()
    }
}

/// Metadata maps are encoded as repeated entries and therefore require an
/// explicit distinct-key check at every protobuf scope.
pub struct MetadataKeysUniqueRule;

impl ValidationRule for MetadataKeysUniqueRule {
    fn id(&self) -> &str {
        "proto.metadata_keys_unique"
    }

    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn check(&self, model: &Model, _ctx: &ValidationContext) -> Vec<Violation> {
        let Some(proto) = model.retained_proto() else {
            return Vec::new();
        };
        let mut violations = Vec::new();
        check_unique_entries(
            &proto.metadata_props,
            "ModelProto.metadata_props",
            ViolationLocation::Model,
            self.id(),
            &mut violations,
        );
        if let Some(graph) = &proto.graph {
            check_graph_metadata(graph, self.id(), &mut violations);
        }
        for function in &proto.functions {
            check_unique_entries(
                &function.metadata_props,
                "FunctionProto.metadata_props",
                ViolationLocation::Model,
                self.id(),
                &mut violations,
            );
            for value in &function.value_info {
                check_value_metadata(value, self.id(), &mut violations);
            }
            for node in &function.node {
                check_node_metadata(node, &function.name, self.id(), &mut violations);
            }
        }
        violations
    }
}

/// Validate attribute names, discriminators, union payloads, and per-node
/// attribute-name uniqueness.
pub struct AttributeProtoValidityRule;

impl ValidationRule for AttributeProtoValidityRule {
    fn id(&self) -> &str {
        "proto.attribute_valid"
    }

    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn check(&self, model: &Model, _ctx: &ValidationContext) -> Vec<Violation> {
        let Some(proto) = model.retained_proto() else {
            return Vec::new();
        };
        let mut violations = Vec::new();
        if let Some(graph) = &proto.graph {
            check_graph_attributes(graph, proto.ir_version, self.id(), &mut violations);
        }
        for function in &proto.functions {
            check_attribute_list(
                &function.attribute_proto,
                ViolationLocation::Model,
                proto.ir_version,
                self.id(),
                &mut violations,
            );
            for node in &function.node {
                check_node_attributes_proto(
                    node,
                    &function.name,
                    proto.ir_version,
                    self.id(),
                    &mut violations,
                );
            }
        }
        violations
    }
}

/// Validate every retained `TypeProto`, including container requirements and
/// ONNX-ML opaque types.
pub struct ProtoTypeValidityRule;

impl ValidationRule for ProtoTypeValidityRule {
    fn id(&self) -> &str {
        "proto.type_valid"
    }

    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn check(&self, model: &Model, _ctx: &ValidationContext) -> Vec<Violation> {
        let Some(proto) = model.retained_proto() else {
            return Vec::new();
        };
        let mut violations = Vec::new();
        if let Some(graph) = &proto.graph {
            check_graph_types(graph, true, self.id(), &mut violations);
        }
        for function in &proto.functions {
            for value in &function.value_info {
                check_value_type(value, false, self.id(), &mut violations);
            }
            for node in &function.node {
                check_node_types(node, &function.name, self.id(), &mut violations);
            }
            for attribute in &function.attribute_proto {
                check_attribute_types(
                    attribute,
                    ViolationLocation::Model,
                    self.id(),
                    &mut violations,
                );
            }
        }
        violations
    }
}

/// Validate dense tensor dimensions, storage-field selection, payload size,
/// segment bounds, and external-data constraints.
pub struct TensorPayloadValidityRule;

impl ValidationRule for TensorPayloadValidityRule {
    fn id(&self) -> &str {
        "proto.tensor_payload_valid"
    }

    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn check(&self, model: &Model, _ctx: &ValidationContext) -> Vec<Violation> {
        let Some(proto) = model.retained_proto() else {
            return Vec::new();
        };
        let mut violations = Vec::new();
        if let Some(graph) = &proto.graph {
            visit_graph_tensors(graph, self.id(), &mut violations);
        }
        for function in &proto.functions {
            for node in &function.node {
                visit_node_tensors(node, &function.name, self.id(), &mut violations);
            }
            for attribute in &function.attribute_proto {
                visit_attribute_tensors(
                    attribute,
                    ViolationLocation::Model,
                    self.id(),
                    &mut violations,
                );
            }
        }
        violations
    }
}

/// Validate sparse tensor COO structure, index type/shape, bounds, ordering,
/// and uniqueness.
pub struct SparseTensorValidityRule;

impl ValidationRule for SparseTensorValidityRule {
    fn id(&self) -> &str {
        "proto.sparse_tensor_valid"
    }

    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn check(&self, model: &Model, _ctx: &ValidationContext) -> Vec<Violation> {
        let Some(proto) = model.retained_proto() else {
            return Vec::new();
        };
        let mut violations = Vec::new();
        if let Some(graph) = &proto.graph {
            visit_graph_sparse_tensors(graph, self.id(), &mut violations);
        }
        for function in &proto.functions {
            for node in &function.node {
                visit_node_sparse_tensors(node, &function.name, self.id(), &mut violations);
            }
            for attribute in &function.attribute_proto {
                visit_attribute_sparse_tensors(
                    attribute,
                    ViolationLocation::Model,
                    self.id(),
                    &mut violations,
                );
            }
        }
        violations
    }
}

fn check_graph_attributes(
    graph: &GraphProto,
    ir_version: i64,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    for node in &graph.node {
        check_node_attributes_proto(node, &graph.name, ir_version, rule_id, violations);
    }
}

fn check_node_attributes_proto(
    node: &NodeProto,
    graph_name: &str,
    ir_version: i64,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    let location = ViolationLocation::Node {
        graph_name: graph_name.to_string(),
        node_name: if node.name.is_empty() {
            format!("<{}>", node.op_type)
        } else {
            node.name.clone()
        },
    };
    if node.op_type.is_empty() {
        violations.push(Violation {
            rule_id: rule_id.to_string(),
            severity: Severity::Error,
            message: "NodeProto.op_type must be present".into(),
            location: location.clone(),
        });
    }
    check_attribute_list(
        &node.attribute,
        location.clone(),
        ir_version,
        rule_id,
        violations,
    );
    for attribute in &node.attribute {
        if let Some(graph) = &attribute.g {
            check_graph_attributes(graph, ir_version, rule_id, violations);
        }
        for graph in &attribute.graphs {
            check_graph_attributes(graph, ir_version, rule_id, violations);
        }
    }
}

fn check_attribute_list(
    attributes: &[AttributeProto],
    location: ViolationLocation,
    ir_version: i64,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    let mut names = HashSet::new();
    for attribute in attributes {
        if attribute.name.is_empty() {
            violations.push(Violation {
                rule_id: rule_id.to_string(),
                severity: Severity::Error,
                message: "AttributeProto.name must be present".into(),
                location: location.clone(),
            });
        } else if !names.insert(attribute.name.as_str()) {
            violations.push(Violation {
                rule_id: rule_id.to_string(),
                severity: Severity::Error,
                message: format!("attribute name '{}' is not unique", attribute.name),
                location: location.clone(),
            });
        }
        check_attribute_proto(attribute, location.clone(), ir_version, rule_id, violations);
    }
}

fn check_attribute_proto(
    attribute: &AttributeProto,
    location: ViolationLocation,
    ir_version: i64,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    let expected = attribute_proto::AttributeType::try_from(attribute.r#type)
        .ok()
        .filter(|value| *value != attribute_proto::AttributeType::Undefined);
    if ir_version >= 2 && expected.is_none() {
        violations.push(Violation {
            rule_id: rule_id.to_string(),
            severity: Severity::Error,
            message: format!(
                "attribute '{}' has invalid or undefined discriminator {}",
                attribute.name, attribute.r#type
            ),
            location,
        });
        return;
    }
    let populated = [
        (attribute.f != 0.0, attribute_proto::AttributeType::Float),
        (attribute.i != 0, attribute_proto::AttributeType::Int),
        (
            !attribute.s.is_empty(),
            attribute_proto::AttributeType::String,
        ),
        (
            attribute.t.is_some(),
            attribute_proto::AttributeType::Tensor,
        ),
        (attribute.g.is_some(), attribute_proto::AttributeType::Graph),
        (
            !attribute.floats.is_empty(),
            attribute_proto::AttributeType::Floats,
        ),
        (
            !attribute.ints.is_empty(),
            attribute_proto::AttributeType::Ints,
        ),
        (
            !attribute.strings.is_empty(),
            attribute_proto::AttributeType::Strings,
        ),
        (
            !attribute.tensors.is_empty(),
            attribute_proto::AttributeType::Tensors,
        ),
        (
            !attribute.graphs.is_empty(),
            attribute_proto::AttributeType::Graphs,
        ),
        (
            attribute.tp.is_some(),
            attribute_proto::AttributeType::TypeProto,
        ),
        (
            !attribute.type_protos.is_empty(),
            attribute_proto::AttributeType::TypeProtos,
        ),
        (
            attribute.sparse_tensor.is_some(),
            attribute_proto::AttributeType::SparseTensor,
        ),
        (
            !attribute.sparse_tensors.is_empty(),
            attribute_proto::AttributeType::SparseTensors,
        ),
    ]
    .into_iter()
    .filter_map(|(present, kind)| present.then_some(kind))
    .collect::<Vec<_>>();
    if !attribute.ref_attr_name.is_empty() {
        if !populated.is_empty() {
            violations.push(Violation {
                rule_id: rule_id.to_string(),
                severity: Severity::Error,
                message: format!(
                    "referenced attribute '{}' must not contain a value payload",
                    attribute.name
                ),
                location,
            });
        }
        return;
    }
    if let Some(expected) = expected {
        if let Some(actual) = populated.iter().find(|&&actual| actual != expected) {
            violations.push(Violation {
                rule_id: rule_id.to_string(),
                severity: Severity::Error,
                message: format!(
                    "attribute '{}' discriminator {:?} conflicts with populated {:?} payload",
                    attribute.name, expected, actual
                ),
                location: location.clone(),
            });
        }
        let required_message_missing = matches!(
            expected,
            attribute_proto::AttributeType::Tensor
                | attribute_proto::AttributeType::Graph
                | attribute_proto::AttributeType::SparseTensor
                | attribute_proto::AttributeType::TypeProto
        ) && !populated.contains(&expected);
        if required_message_missing {
            violations.push(Violation {
                rule_id: rule_id.to_string(),
                severity: Severity::Error,
                message: format!(
                    "attribute '{}' discriminator {:?} requires its message payload",
                    attribute.name, expected
                ),
                location,
            });
        }
    }
}

fn check_unique_entries(
    entries: &[StringStringEntryProto],
    field: &str,
    location: ViolationLocation,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    let mut keys = HashSet::new();
    for entry in entries {
        if !keys.insert(entry.key.as_str()) {
            violations.push(Violation {
                rule_id: rule_id.to_string(),
                severity: Severity::Error,
                message: format!("{field} contains duplicate key '{}'", entry.key),
                location: location.clone(),
            });
        }
    }
}

fn check_graph_metadata(graph: &GraphProto, rule_id: &str, violations: &mut Vec<Violation>) {
    let location = ViolationLocation::Graph {
        graph_name: graph.name.clone(),
    };
    check_unique_entries(
        &graph.metadata_props,
        "GraphProto.metadata_props",
        location.clone(),
        rule_id,
        violations,
    );
    for value in graph
        .input
        .iter()
        .chain(&graph.output)
        .chain(&graph.value_info)
    {
        check_value_metadata(value, rule_id, violations);
    }
    for tensor in &graph.initializer {
        check_tensor_metadata(tensor, rule_id, violations);
    }
    for sparse in &graph.sparse_initializer {
        if let Some(values) = &sparse.values {
            check_tensor_metadata(values, rule_id, violations);
        }
        if let Some(indices) = &sparse.indices {
            check_tensor_metadata(indices, rule_id, violations);
        }
    }
    for annotation in &graph.quantization_annotation {
        check_unique_entries(
            &annotation.quant_parameter_tensor_names,
            "TensorAnnotation.quant_parameter_tensor_names",
            ViolationLocation::Value {
                value_name: annotation.tensor_name.clone(),
            },
            rule_id,
            violations,
        );
    }
    for node in &graph.node {
        check_node_metadata(node, &graph.name, rule_id, violations);
    }
}

fn check_node_metadata(
    node: &NodeProto,
    graph_name: &str,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    let location = ViolationLocation::Node {
        graph_name: graph_name.to_string(),
        node_name: if node.name.is_empty() {
            format!("<{}>", node.op_type)
        } else {
            node.name.clone()
        },
    };
    check_unique_entries(
        &node.metadata_props,
        "NodeProto.metadata_props",
        location,
        rule_id,
        violations,
    );
    for attribute in &node.attribute {
        if let Some(graph) = &attribute.g {
            check_graph_metadata(graph, rule_id, violations);
        }
        for graph in &attribute.graphs {
            check_graph_metadata(graph, rule_id, violations);
        }
        if let Some(tensor) = &attribute.t {
            check_tensor_metadata(tensor, rule_id, violations);
        }
        for tensor in &attribute.tensors {
            check_tensor_metadata(tensor, rule_id, violations);
        }
        if let Some(sparse) = &attribute.sparse_tensor {
            if let Some(values) = &sparse.values {
                check_tensor_metadata(values, rule_id, violations);
            }
            if let Some(indices) = &sparse.indices {
                check_tensor_metadata(indices, rule_id, violations);
            }
        }
        for sparse in &attribute.sparse_tensors {
            if let Some(values) = &sparse.values {
                check_tensor_metadata(values, rule_id, violations);
            }
            if let Some(indices) = &sparse.indices {
                check_tensor_metadata(indices, rule_id, violations);
            }
        }
    }
}

fn check_value_metadata(value: &ValueInfoProto, rule_id: &str, violations: &mut Vec<Violation>) {
    check_unique_entries(
        &value.metadata_props,
        "ValueInfoProto.metadata_props",
        ViolationLocation::Value {
            value_name: value.name.clone(),
        },
        rule_id,
        violations,
    );
}

fn check_tensor_metadata(tensor: &TensorProto, rule_id: &str, violations: &mut Vec<Violation>) {
    check_unique_entries(
        &tensor.metadata_props,
        "TensorProto.metadata_props",
        ViolationLocation::Value {
            value_name: tensor.name.clone(),
        },
        rule_id,
        violations,
    );
}

fn check_graph_types(
    graph: &GraphProto,
    top_level: bool,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    for value in &graph.input {
        check_value_type(value, top_level, rule_id, violations);
    }
    for value in &graph.output {
        check_value_type(value, top_level, rule_id, violations);
    }
    for value in &graph.value_info {
        check_value_type(value, false, rule_id, violations);
    }
    for node in &graph.node {
        check_node_types(node, &graph.name, rule_id, violations);
    }
}

fn check_node_types(
    node: &NodeProto,
    graph_name: &str,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    let location = ViolationLocation::Node {
        graph_name: graph_name.to_string(),
        node_name: if node.name.is_empty() {
            format!("<{}>", node.op_type)
        } else {
            node.name.clone()
        },
    };
    for attribute in &node.attribute {
        check_attribute_types(attribute, location.clone(), rule_id, violations);
        if let Some(graph) = &attribute.g {
            check_graph_types(graph, false, rule_id, violations);
        }
        for graph in &attribute.graphs {
            check_graph_types(graph, false, rule_id, violations);
        }
    }
}

fn check_attribute_types(
    attribute: &AttributeProto,
    location: ViolationLocation,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    if let Some(value) = &attribute.tp {
        check_type_proto(value, location.clone(), false, rule_id, violations);
    }
    for value in &attribute.type_protos {
        check_type_proto(value, location.clone(), false, rule_id, violations);
    }
}

fn check_value_type(
    value: &ValueInfoProto,
    require_type: bool,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    let location = ViolationLocation::Value {
        value_name: value.name.clone(),
    };
    if value.name.is_empty() {
        violations.push(Violation {
            rule_id: rule_id.to_string(),
            severity: Severity::Error,
            message: "ValueInfoProto.name must be present".into(),
            location: location.clone(),
        });
    }
    match &value.r#type {
        Some(r#type) => check_type_proto(r#type, location, require_type, rule_id, violations),
        None if require_type => violations.push(Violation {
            rule_id: rule_id.to_string(),
            severity: Severity::Error,
            message: "top-level graph inputs and outputs must declare a TypeProto".into(),
            location,
        }),
        None => {}
    }
}

fn check_type_proto(
    value: &TypeProto,
    location: ViolationLocation,
    require_shape: bool,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    let invalid = |message: String, violations: &mut Vec<Violation>| {
        violations.push(Violation {
            rule_id: rule_id.to_string(),
            severity: Severity::Error,
            message,
            location: location.clone(),
        });
    };
    match value.value.as_ref() {
        None => invalid("TypeProto must select a value variant".into(), violations),
        Some(type_proto::Value::TensorType(tensor)) => {
            check_tensor_type(
                tensor.elem_type,
                tensor.shape.as_ref(),
                require_shape,
                &invalid,
                violations,
            );
        }
        Some(type_proto::Value::SparseTensorType(tensor)) => {
            check_tensor_type(
                tensor.elem_type,
                tensor.shape.as_ref(),
                require_shape,
                &invalid,
                violations,
            );
        }
        Some(type_proto::Value::SequenceType(sequence)) => match &sequence.elem_type {
            Some(elem_type) => {
                check_type_proto(elem_type, location.clone(), false, rule_id, violations)
            }
            None => invalid(
                "TypeProto.Sequence.elem_type must be present".into(),
                violations,
            ),
        },
        Some(type_proto::Value::MapType(map)) => {
            let key = tensor_proto::DataType::try_from(map.key_type).ok();
            if !matches!(
                key,
                Some(
                    tensor_proto::DataType::Uint8
                        | tensor_proto::DataType::Int8
                        | tensor_proto::DataType::Uint16
                        | tensor_proto::DataType::Int16
                        | tensor_proto::DataType::Int32
                        | tensor_proto::DataType::Int64
                        | tensor_proto::DataType::String
                        | tensor_proto::DataType::Uint32
                        | tensor_proto::DataType::Uint64
                )
            ) {
                invalid(
                    format!(
                        "TypeProto.Map.key_type {} must be an integral or string dtype",
                        map.key_type
                    ),
                    violations,
                );
            }
            match &map.value_type {
                Some(value_type) => {
                    check_type_proto(value_type, location.clone(), false, rule_id, violations)
                }
                None => invalid(
                    "TypeProto.Map.value_type must be present".into(),
                    violations,
                ),
            }
        }
        Some(type_proto::Value::OptionalType(optional)) => match &optional.elem_type {
            Some(elem_type) => {
                if !matches!(
                    elem_type.value,
                    Some(
                        type_proto::Value::TensorType(_)
                            | type_proto::Value::SequenceType(_)
                            | type_proto::Value::MapType(_)
                    )
                ) {
                    invalid(
                        "TypeProto.Optional.elem_type must be a tensor, sequence, or map".into(),
                        violations,
                    );
                }
                check_type_proto(elem_type, location.clone(), false, rule_id, violations);
            }
            None => invalid(
                "TypeProto.Optional.elem_type must be present".into(),
                violations,
            ),
        },
        Some(type_proto::Value::OpaqueType(_)) => {}
    }
}

fn check_tensor_type(
    elem_type: i32,
    shape: Option<&TensorShapeProto>,
    require_shape: bool,
    invalid: &impl Fn(String, &mut Vec<Violation>),
    violations: &mut Vec<Violation>,
) {
    if tensor_proto::DataType::try_from(elem_type)
        .ok()
        .is_none_or(|dtype| dtype == tensor_proto::DataType::Undefined)
    {
        invalid(
            format!("tensor elem_type {elem_type} must be a defined ONNX dtype"),
            violations,
        );
    }
    if require_shape && shape.is_none() {
        invalid(
            "top-level tensor and sparse tensor types must declare a shape".into(),
            violations,
        );
    }
    if let Some(shape) = shape {
        for (index, dim) in shape.dim.iter().enumerate() {
            if matches!(
                dim.value,
                Some(onnx_runtime_loader::proto::onnx::tensor_shape_proto::dimension::Value::DimValue(value))
                    if value < 0
            ) {
                invalid(
                    format!("tensor shape dimension {index} must not be negative"),
                    violations,
                );
            }
        }
    }
}

fn visit_graph_tensors(graph: &GraphProto, rule_id: &str, violations: &mut Vec<Violation>) {
    for tensor in &graph.initializer {
        check_tensor_payload(tensor, rule_id, violations);
    }
    for sparse in &graph.sparse_initializer {
        if let Some(values) = &sparse.values {
            check_tensor_payload(values, rule_id, violations);
        }
        if let Some(indices) = &sparse.indices {
            check_tensor_payload(indices, rule_id, violations);
        }
    }
    for node in &graph.node {
        visit_node_tensors(node, &graph.name, rule_id, violations);
    }
}

fn visit_node_tensors(
    node: &NodeProto,
    graph_name: &str,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    let location = ViolationLocation::Node {
        graph_name: graph_name.to_string(),
        node_name: if node.name.is_empty() {
            format!("<{}>", node.op_type)
        } else {
            node.name.clone()
        },
    };
    for attribute in &node.attribute {
        visit_attribute_tensors(attribute, location.clone(), rule_id, violations);
        if let Some(graph) = &attribute.g {
            visit_graph_tensors(graph, rule_id, violations);
        }
        for graph in &attribute.graphs {
            visit_graph_tensors(graph, rule_id, violations);
        }
    }
}

fn visit_attribute_tensors(
    attribute: &AttributeProto,
    _location: ViolationLocation,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    if let Some(tensor) = &attribute.t {
        check_tensor_payload(tensor, rule_id, violations);
    }
    for tensor in &attribute.tensors {
        check_tensor_payload(tensor, rule_id, violations);
    }
    if let Some(sparse) = &attribute.sparse_tensor {
        if let Some(values) = &sparse.values {
            check_tensor_payload(values, rule_id, violations);
        }
        if let Some(indices) = &sparse.indices {
            check_tensor_payload(indices, rule_id, violations);
        }
    }
    for sparse in &attribute.sparse_tensors {
        if let Some(values) = &sparse.values {
            check_tensor_payload(values, rule_id, violations);
        }
        if let Some(indices) = &sparse.indices {
            check_tensor_payload(indices, rule_id, violations);
        }
    }
}

fn check_tensor_payload(tensor: &TensorProto, rule_id: &str, violations: &mut Vec<Violation>) {
    let location = ViolationLocation::Value {
        value_name: tensor.name.clone(),
    };
    let mut report = |message: String| {
        violations.push(Violation {
            rule_id: rule_id.to_string(),
            severity: Severity::Error,
            message,
            location: location.clone(),
        });
    };
    let Some(dtype) = tensor_proto::DataType::try_from(tensor.data_type)
        .ok()
        .filter(|dtype| *dtype != tensor_proto::DataType::Undefined)
    else {
        report(format!(
            "TensorProto.data_type {} must be a defined ONNX dtype",
            tensor.data_type
        ));
        return;
    };
    let Some(full_count) = checked_numel(&tensor.dims) else {
        report(
            "TensorProto dimensions must be non-negative and their product must fit usize".into(),
        );
        return;
    };
    let count = if let Some(segment) = &tensor.segment {
        if segment.begin < 0 || segment.end < segment.begin {
            report("TensorProto.segment must satisfy 0 <= begin <= end".into());
            return;
        }
        let Ok(begin) = usize::try_from(segment.begin) else {
            report("TensorProto.segment.begin does not fit usize".into());
            return;
        };
        let Ok(end) = usize::try_from(segment.end) else {
            report("TensorProto.segment.end does not fit usize".into());
            return;
        };
        if end > full_count {
            report(format!(
                "TensorProto.segment end {end} exceeds element count {full_count}"
            ));
        }
        end.saturating_sub(begin)
    } else {
        full_count
    };

    let populated = [
        !tensor.float_data.is_empty(),
        !tensor.int32_data.is_empty(),
        !tensor.string_data.is_empty(),
        !tensor.int64_data.is_empty(),
        !tensor.raw_data.is_empty(),
        !tensor.double_data.is_empty(),
        !tensor.uint64_data.is_empty(),
    ]
    .into_iter()
    .filter(|present| *present)
    .count();
    let data_location = tensor_proto::DataLocation::try_from(tensor.data_location).ok();
    if data_location.is_none() {
        report(format!(
            "TensorProto.data_location {} is not valid",
            tensor.data_location
        ));
        return;
    }
    if data_location == Some(tensor_proto::DataLocation::External) {
        if populated != 0 {
            report("external TensorProto must not contain embedded payload fields".into());
        }
        let mut external = HashMap::new();
        for entry in &tensor.external_data {
            if external
                .insert(entry.key.as_str(), entry.value.as_str())
                .is_some()
            {
                report(format!(
                    "TensorProto.external_data contains duplicate key '{}'",
                    entry.key
                ));
            }
        }
        if external
            .get("location")
            .is_none_or(|value| value.is_empty())
        {
            report("external TensorProto requires a non-empty location entry".into());
        }
        let offset = parse_external_usize(external.get("offset").copied(), "offset", &mut report);
        let length = parse_external_usize(external.get("length").copied(), "length", &mut report);
        if let (Some(offset), Some(length)) = (offset, length)
            && offset.checked_add(length).is_none()
        {
            report("external TensorProto offset + length overflows usize".into());
        }
        return;
    }
    if !tensor.external_data.is_empty() {
        report("inline TensorProto must not contain external_data entries".into());
    }
    if populated > 1 {
        report("TensorProto must use exactly one embedded payload field".into());
        return;
    }
    if count > 0 && populated == 0 {
        report("non-empty TensorProto is missing its payload".into());
        return;
    }
    if populated == 0 {
        return;
    }
    let actual_expected = match dtype {
        _ if !tensor.raw_data.is_empty() => storage_bytes(dtype, count)
            .map(|expected| (tensor.raw_data.len(), expected, "raw_data")),
        tensor_proto::DataType::Float => Some((tensor.float_data.len(), count, "float_data")),
        tensor_proto::DataType::Complex64 => count
            .checked_mul(2)
            .map(|expected| (tensor.float_data.len(), expected, "float_data")),
        tensor_proto::DataType::Double => Some((tensor.double_data.len(), count, "double_data")),
        tensor_proto::DataType::Complex128 => count
            .checked_mul(2)
            .map(|expected| (tensor.double_data.len(), expected, "double_data")),
        tensor_proto::DataType::Int64 => Some((tensor.int64_data.len(), count, "int64_data")),
        tensor_proto::DataType::String => Some((tensor.string_data.len(), count, "string_data")),
        tensor_proto::DataType::Uint32 | tensor_proto::DataType::Uint64 => {
            Some((tensor.uint64_data.len(), count, "uint64_data"))
        }
        tensor_proto::DataType::Uint4
        | tensor_proto::DataType::Int4
        | tensor_proto::DataType::Float4e2m1 => count
            .checked_add(1)
            .map(|value| (tensor.int32_data.len(), value / 2, "int32_data")),
        tensor_proto::DataType::Uint2 | tensor_proto::DataType::Int2 => count
            .checked_add(3)
            .map(|value| (tensor.int32_data.len(), value / 4, "int32_data")),
        _ => Some((tensor.int32_data.len(), count, "int32_data")),
    };
    match actual_expected {
        Some((actual, expected, field)) if actual != expected => report(format!(
            "TensorProto.{field} contains {actual} entries/bytes but {expected} are required"
        )),
        None => report("TensorProto payload size arithmetic overflowed usize".into()),
        _ => {}
    }
}

fn parse_external_usize(
    value: Option<&str>,
    key: &str,
    report: &mut impl FnMut(String),
) -> Option<usize> {
    let value = value?;
    match value.parse::<usize>() {
        Ok(value) => Some(value),
        Err(_) => {
            report(format!(
                "TensorProto.external_data '{key}' must be an unsigned integer"
            ));
            None
        }
    }
}

fn checked_numel(dims: &[i64]) -> Option<usize> {
    dims.iter().try_fold(1usize, |count, &dim| {
        let dim = usize::try_from(dim).ok()?;
        count.checked_mul(dim)
    })
}

fn storage_bytes(dtype: tensor_proto::DataType, count: usize) -> Option<usize> {
    let bits = match dtype {
        tensor_proto::DataType::Uint2 | tensor_proto::DataType::Int2 => 2,
        tensor_proto::DataType::Uint4
        | tensor_proto::DataType::Int4
        | tensor_proto::DataType::Float4e2m1 => 4,
        tensor_proto::DataType::Uint8
        | tensor_proto::DataType::Int8
        | tensor_proto::DataType::Bool
        | tensor_proto::DataType::Float8e4m3fn
        | tensor_proto::DataType::Float8e4m3fnuz
        | tensor_proto::DataType::Float8e5m2
        | tensor_proto::DataType::Float8e5m2fnuz
        | tensor_proto::DataType::Float8e8m0 => 8,
        tensor_proto::DataType::Uint16
        | tensor_proto::DataType::Int16
        | tensor_proto::DataType::Float16
        | tensor_proto::DataType::Bfloat16 => 16,
        tensor_proto::DataType::Float
        | tensor_proto::DataType::Int32
        | tensor_proto::DataType::Uint32 => 32,
        tensor_proto::DataType::Double
        | tensor_proto::DataType::Int64
        | tensor_proto::DataType::Uint64
        | tensor_proto::DataType::Complex64 => 64,
        tensor_proto::DataType::Complex128 => 128,
        tensor_proto::DataType::String | tensor_proto::DataType::Undefined => return None,
    };
    count
        .checked_mul(bits)
        .and_then(|bits| bits.checked_add(7))
        .map(|bits| bits / 8)
}

fn visit_graph_sparse_tensors(graph: &GraphProto, rule_id: &str, violations: &mut Vec<Violation>) {
    for sparse in &graph.sparse_initializer {
        check_sparse_tensor(sparse, true, rule_id, violations);
    }
    for node in &graph.node {
        visit_node_sparse_tensors(node, &graph.name, rule_id, violations);
    }
}

fn visit_node_sparse_tensors(
    node: &NodeProto,
    graph_name: &str,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    let location = ViolationLocation::Node {
        graph_name: graph_name.to_string(),
        node_name: if node.name.is_empty() {
            format!("<{}>", node.op_type)
        } else {
            node.name.clone()
        },
    };
    for attribute in &node.attribute {
        visit_attribute_sparse_tensors(attribute, location.clone(), rule_id, violations);
        if let Some(graph) = &attribute.g {
            visit_graph_sparse_tensors(graph, rule_id, violations);
        }
        for graph in &attribute.graphs {
            visit_graph_sparse_tensors(graph, rule_id, violations);
        }
    }
}

fn visit_attribute_sparse_tensors(
    attribute: &AttributeProto,
    _location: ViolationLocation,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    if let Some(sparse) = &attribute.sparse_tensor {
        check_sparse_tensor(sparse, false, rule_id, violations);
    }
    for sparse in &attribute.sparse_tensors {
        check_sparse_tensor(sparse, false, rule_id, violations);
    }
}

fn check_sparse_tensor(
    sparse: &SparseTensorProto,
    require_name: bool,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    let name = sparse
        .values
        .as_ref()
        .map(|values| values.name.clone())
        .unwrap_or_default();
    let location = ViolationLocation::Value { value_name: name };
    let mut report = |message: String| {
        violations.push(Violation {
            rule_id: rule_id.to_string(),
            severity: Severity::Error,
            message,
            location: location.clone(),
        });
    };
    let Some(values) = &sparse.values else {
        report("SparseTensorProto.values must be present".into());
        return;
    };
    if require_name && values.name.is_empty() {
        report("sparse initializer values must have a non-empty name".into());
    }
    if sparse.dims.is_empty() || sparse.dims.iter().any(|&dim| dim <= 0) {
        report("SparseTensorProto must have positive rank and dimensions".into());
        return;
    }
    let Some(dense_count) = checked_numel(&sparse.dims) else {
        report("SparseTensorProto dimensions must fit usize".into());
        return;
    };
    if values.dims.len() != 1 {
        report("SparseTensorProto.values must have shape [NNZ]".into());
        return;
    }
    let Ok(nnz) = usize::try_from(values.dims[0]) else {
        report("SparseTensorProto NNZ must be non-negative".into());
        return;
    };
    if nnz > dense_count {
        report(format!(
            "SparseTensorProto NNZ {nnz} exceeds dense element count {dense_count}"
        ));
    }
    let Some(indices) = &sparse.indices else {
        if nnz != 0 {
            report("SparseTensorProto.indices must be present when NNZ is nonzero".into());
        }
        return;
    };
    if indices.data_type != tensor_proto::DataType::Int64 as i32 {
        report("SparseTensorProto.indices must have INT64 dtype".into());
        return;
    }
    let rank = sparse.dims.len();
    let coordinate = match indices.dims.as_slice() {
        [count] if usize::try_from(*count).ok() == Some(nnz) => false,
        [count, width]
            if usize::try_from(*count).ok() == Some(nnz)
                && usize::try_from(*width).ok() == Some(rank) =>
        {
            true
        }
        _ => {
            report(format!(
                "SparseTensorProto.indices shape must be [{nnz}] or [{nnz}, {rank}]"
            ));
            return;
        }
    };
    let Some(index_values) = tensor_i64_values(indices) else {
        report("SparseTensorProto.indices must use int64_data or raw_data".into());
        return;
    };
    let expected_indices = if coordinate {
        nnz.checked_mul(rank)
    } else {
        Some(nnz)
    };
    if expected_indices != Some(index_values.len()) {
        report("SparseTensorProto.indices payload count does not match its shape".into());
        return;
    }
    if coordinate {
        let mut previous: Option<&[i64]> = None;
        for tuple in index_values.chunks(rank.max(1)) {
            if tuple
                .iter()
                .zip(&sparse.dims)
                .any(|(&index, &dim)| index < 0 || index >= dim)
            {
                report("SparseTensorProto coordinate index is out of bounds".into());
            }
            if previous.is_some_and(|prior| prior >= tuple) {
                report(
                    "SparseTensorProto coordinate indices must be lexicographically increasing"
                        .into(),
                );
            }
            previous = Some(tuple);
        }
    } else {
        let Ok(dense_count) = i64::try_from(dense_count) else {
            report("SparseTensorProto dense element count does not fit i64".into());
            return;
        };
        let mut previous = None;
        for &index in &index_values {
            if index < 0 || index >= dense_count {
                report("SparseTensorProto linear index is out of bounds".into());
            }
            if previous.is_some_and(|prior| prior >= index) {
                report("SparseTensorProto linear indices must be strictly increasing".into());
            }
            previous = Some(index);
        }
    }
}

fn tensor_i64_values(tensor: &TensorProto) -> Option<Vec<i64>> {
    if !tensor.int64_data.is_empty() {
        return Some(tensor.int64_data.clone());
    }
    if tensor.raw_data.len() % 8 != 0 {
        return None;
    }
    Some(
        tensor
            .raw_data
            .chunks_exact(8)
            .map(|bytes| i64::from_le_bytes(bytes.try_into().expect("chunk size is eight")))
            .collect(),
    )
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
        let configuration = configurations
            .get(device_configuration.configuration_id.as_str())
            .copied();
        if configuration.is_none() {
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
            if let Some(configuration) = configuration {
                check_sharding_devices(
                    sharding,
                    configuration,
                    graph_name,
                    node,
                    rule_id,
                    violations,
                );
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

fn check_sharding_devices(
    sharding: &onnx_runtime_loader::proto::onnx::ShardingSpecProto,
    configuration: &DeviceConfigurationProto,
    graph_name: &str,
    node: &NodeProto,
    rule_id: &str,
    violations: &mut Vec<Violation>,
) {
    let num_devices = i64::from(configuration.num_devices);
    let mut groups = HashMap::new();
    for group in &sharding.index_to_device_group_map {
        if groups.insert(group.key, group).is_some() {
            violations.push(proto_node_violation(
                rule_id,
                graph_name,
                node,
                format!("device group key {} is not unique", group.key),
            ));
        }
        if !sharding.device.contains(&group.key) {
            violations.push(proto_node_violation(
                rule_id,
                graph_name,
                node,
                format!(
                    "device group key {} is not referenced by ShardingSpecProto.device",
                    group.key
                ),
            ));
        }
        if group.value.is_empty() {
            violations.push(proto_node_violation(
                rule_id,
                graph_name,
                node,
                format!(
                    "device group {} must contain at least one device",
                    group.key
                ),
            ));
        }
        let mut members = HashSet::new();
        for &member in &group.value {
            if member < 0 || member >= num_devices {
                violations.push(proto_node_violation(
                    rule_id,
                    graph_name,
                    node,
                    format!(
                        "device group {} member {} is outside [0, {})",
                        group.key, member, configuration.num_devices
                    ),
                ));
            } else if !members.insert(member) {
                violations.push(proto_node_violation(
                    rule_id,
                    graph_name,
                    node,
                    format!(
                        "device group {} contains duplicate member {}",
                        group.key, member
                    ),
                ));
            }
        }
    }
    for &device in &sharding.device {
        if !groups.contains_key(&device) && (device < 0 || device >= num_devices) {
            violations.push(proto_node_violation(
                rule_id,
                graph_name,
                node,
                format!(
                    "sharding device {} is neither a group key nor within [0, {})",
                    device, configuration.num_devices
                ),
            ));
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
    use onnx_runtime_loader::proto::onnx::{
        FunctionProto, OperatorSetIdProto, StringStringEntryProto,
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

    fn retained_model(proto: ModelProto) -> Model {
        retained_model_at(proto, 13)
    }

    fn retained_model_at(mut proto: ModelProto, ir_version: i64) -> Model {
        proto.ir_version = ir_version;
        if ir_version >= 3 && proto.opset_import.is_empty() {
            proto.opset_import.push(OperatorSetIdProto {
                domain: String::new(),
                version: 24,
            });
        }
        if proto.graph.is_none() {
            proto.graph = Some(GraphProto {
                name: "graph".into(),
                ..Default::default()
            });
        }
        Model::from_proto(proto).unwrap()
    }

    fn function(name: &str, nodes: Vec<NodeProto>) -> FunctionProto {
        FunctionProto {
            name: name.into(),
            input: vec!["X".into()],
            output: vec!["Y".into()],
            node: nodes,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 24,
            }],
            domain: "local.test".into(),
            ..Default::default()
        }
    }

    fn local_function_call_model(function: FunctionProto, call: NodeProto) -> Model {
        let graph_inputs = call
            .input
            .iter()
            .filter(|name| !name.is_empty())
            .map(|name| ValueInfoProto {
                name: name.clone(),
                ..Default::default()
            })
            .collect();
        let graph_outputs = call
            .output
            .iter()
            .filter(|name| !name.is_empty())
            .map(|name| ValueInfoProto {
                name: name.clone(),
                ..Default::default()
            })
            .collect();
        retained_model(ModelProto {
            opset_import: vec![
                OperatorSetIdProto {
                    domain: String::new(),
                    version: 24,
                },
                OperatorSetIdProto {
                    domain: "local.test".into(),
                    version: 1,
                },
            ],
            graph: Some(GraphProto {
                name: "graph".into(),
                input: graph_inputs,
                output: graph_outputs,
                node: vec![call],
                ..Default::default()
            }),
            functions: vec![function],
            ..Default::default()
        })
    }

    #[test]
    fn metadata_rule_rejects_duplicate_keys() {
        let model = retained_model(ModelProto {
            metadata_props: vec![
                StringStringEntryProto {
                    key: "owner".into(),
                    value: "one".into(),
                },
                StringStringEntryProto {
                    key: "owner".into(),
                    value: "two".into(),
                },
            ],
            ..Default::default()
        });
        let violations = MetadataKeysUniqueRule.check(&model, &ValidationContext::default());
        assert_eq!(violations.len(), 1);
        assert!(violations[0].message.contains("duplicate key 'owner'"));
    }

    #[test]
    fn attribute_rule_checks_discriminator_union_and_references() {
        let location = ViolationLocation::Model;
        let mut violations = Vec::new();
        check_attribute_proto(
            &AttributeProto {
                name: "weight".into(),
                r#type: attribute_proto::AttributeType::Tensor as i32,
                i: 7,
                ..Default::default()
            },
            location.clone(),
            13,
            "test",
            &mut violations,
        );
        assert_eq!(violations.len(), 2, "{violations:?}");

        violations.clear();
        check_attribute_proto(
            &AttributeProto {
                name: "alpha".into(),
                ref_attr_name: "alpha".into(),
                r#type: attribute_proto::AttributeType::Float as i32,
                ..Default::default()
            },
            location,
            13,
            "test",
            &mut violations,
        );
        assert!(violations.is_empty(), "{violations:?}");
    }

    #[test]
    fn attribute_type_is_required_starting_with_ir_v2() {
        let attribute = AttributeProto {
            name: "axis".into(),
            i: 1,
            ..Default::default()
        };
        let mut violations = Vec::new();
        check_attribute_proto(
            &attribute,
            ViolationLocation::Model,
            1,
            "test",
            &mut violations,
        );
        assert!(violations.is_empty(), "{violations:?}");

        check_attribute_proto(
            &attribute,
            ViolationLocation::Model,
            2,
            "test",
            &mut violations,
        );
        assert_eq!(violations.len(), 1, "{violations:?}");
        assert!(violations[0].message.contains("undefined discriminator"));
    }

    #[test]
    fn type_rule_accepts_opaque_and_rejects_invalid_containers() {
        let mut violations = Vec::new();
        check_type_proto(
            &TypeProto {
                value: Some(type_proto::Value::OpaqueType(type_proto::Opaque {
                    domain: "example".into(),
                    name: "State".into(),
                })),
                ..Default::default()
            },
            ViolationLocation::Model,
            false,
            "test",
            &mut violations,
        );
        assert!(violations.is_empty());

        check_type_proto(
            &TypeProto {
                value: Some(type_proto::Value::MapType(Box::new(type_proto::Map {
                    key_type: tensor_proto::DataType::Float as i32,
                    value_type: None,
                }))),
                ..Default::default()
            },
            ViolationLocation::Model,
            false,
            "test",
            &mut violations,
        );
        assert_eq!(violations.len(), 2, "{violations:?}");
    }

    #[test]
    fn top_level_tensor_requires_shape_but_nested_tensor_does_not() {
        let tensor_without_shape = TypeProto {
            value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                elem_type: tensor_proto::DataType::Float as i32,
                shape: None,
            })),
            ..Default::default()
        };
        let mut violations = Vec::new();
        check_value_type(
            &ValueInfoProto {
                name: "input".into(),
                r#type: Some(tensor_without_shape.clone()),
                ..Default::default()
            },
            true,
            "test",
            &mut violations,
        );
        assert_eq!(violations.len(), 1, "{violations:?}");
        assert!(violations[0].message.contains("must declare a shape"));

        violations.clear();
        check_value_type(
            &ValueInfoProto {
                name: "sequence".into(),
                r#type: Some(TypeProto {
                    value: Some(type_proto::Value::SequenceType(Box::new(
                        type_proto::Sequence {
                            elem_type: Some(Box::new(tensor_without_shape)),
                        },
                    ))),
                    ..Default::default()
                }),
                ..Default::default()
            },
            true,
            "test",
            &mut violations,
        );
        assert!(violations.is_empty(), "{violations:?}");
    }

    #[test]
    fn tensor_payload_rule_checks_sizes_and_external_offsets() {
        let mut violations = Vec::new();
        check_tensor_payload(
            &TensorProto {
                dims: vec![2],
                data_type: tensor_proto::DataType::Float as i32,
                raw_data: vec![0; 4],
                name: "short".into(),
                ..Default::default()
            },
            "test",
            &mut violations,
        );
        assert_eq!(violations.len(), 1);
        assert!(violations[0].message.contains("8 are required"));

        violations.clear();
        check_tensor_payload(
            &TensorProto {
                dims: vec![1],
                data_type: tensor_proto::DataType::Float as i32,
                data_location: tensor_proto::DataLocation::External as i32,
                external_data: vec![
                    StringStringEntryProto {
                        key: "location".into(),
                        value: "weights.bin".into(),
                    },
                    StringStringEntryProto {
                        key: "offset".into(),
                        value: usize::MAX.to_string(),
                    },
                    StringStringEntryProto {
                        key: "length".into(),
                        value: "1".into(),
                    },
                ],
                ..Default::default()
            },
            "test",
            &mut violations,
        );
        assert!(
            violations
                .iter()
                .any(|violation| violation.message.contains("offset + length overflows"))
        );
    }

    #[test]
    fn tensor_payload_rule_accepts_unused_packed_bits_but_checks_length() {
        let mut tensor = TensorProto {
            dims: vec![3],
            data_type: tensor_proto::DataType::Uint4 as i32,
            raw_data: vec![0x21, 0xf3],
            ..Default::default()
        };
        let mut violations = Vec::new();
        check_tensor_payload(&tensor, "test", &mut violations);
        assert!(violations.is_empty(), "{violations:?}");

        tensor.raw_data.pop();
        check_tensor_payload(&tensor, "test", &mut violations);
        assert!(
            violations
                .iter()
                .any(|violation| violation.message.contains("but 2 are required")),
            "{violations:?}"
        );
    }

    #[test]
    fn sparse_tensor_rule_checks_order_and_bounds() {
        let sparse = SparseTensorProto {
            values: Some(TensorProto {
                dims: vec![2],
                data_type: tensor_proto::DataType::Float as i32,
                float_data: vec![1.0, 2.0],
                name: "sparse".into(),
                ..Default::default()
            }),
            indices: Some(TensorProto {
                dims: vec![2],
                data_type: tensor_proto::DataType::Int64 as i32,
                int64_data: vec![1, 1],
                ..Default::default()
            }),
            dims: vec![2, 2],
        };
        let mut violations = Vec::new();
        check_sparse_tensor(&sparse, true, "test", &mut violations);
        assert_eq!(violations.len(), 1, "{violations:?}");
        assert!(violations[0].message.contains("strictly increasing"));

        let mut valid = sparse;
        valid.indices.as_mut().unwrap().int64_data = vec![0, 3];
        violations.clear();
        check_sparse_tensor(&valid, true, "test", &mut violations);
        assert!(violations.is_empty(), "{violations:?}");
    }

    #[test]
    fn sparse_tensor_indices_are_optional_only_for_zero_nnz() {
        let mut sparse = SparseTensorProto {
            values: Some(TensorProto {
                dims: vec![0],
                data_type: tensor_proto::DataType::Float as i32,
                name: "empty_sparse".into(),
                ..Default::default()
            }),
            indices: None,
            dims: vec![2, 3],
        };
        let mut violations = Vec::new();
        check_sparse_tensor(&sparse, true, "test", &mut violations);
        assert!(violations.is_empty(), "{violations:?}");

        sparse.values.as_mut().unwrap().dims = vec![1];
        check_sparse_tensor(&sparse, true, "test", &mut violations);
        assert_eq!(violations.len(), 1, "{violations:?}");
        assert!(violations[0].message.contains("when NNZ is nonzero"));
    }

    #[test]
    fn sparse_tensor_requires_positive_rank_and_dimensions() {
        let mut sparse = SparseTensorProto {
            values: Some(TensorProto {
                dims: vec![0],
                data_type: tensor_proto::DataType::Float as i32,
                name: "empty_sparse".into(),
                ..Default::default()
            }),
            indices: None,
            dims: Vec::new(),
        };
        let mut violations = Vec::new();
        check_sparse_tensor(&sparse, true, "test", &mut violations);
        assert_eq!(violations.len(), 1, "{violations:?}");
        assert!(
            violations[0]
                .message
                .contains("positive rank and dimensions")
        );

        sparse.dims = vec![2, 0];
        violations.clear();
        check_sparse_tensor(&sparse, true, "test", &mut violations);
        assert_eq!(violations.len(), 1, "{violations:?}");
        assert!(
            violations[0]
                .message
                .contains("positive rank and dimensions")
        );
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
    fn concrete_schema_input_type_is_enforced() {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 24);
        let data = graph.create_named_value("data", DataType::Float32, static_shape([2]));
        let shape = graph.create_named_value("shape", DataType::Float32, static_shape([1]));
        let output = graph.create_named_value("output", DataType::Float32, static_shape([2]));
        graph.add_input(data);
        graph.add_input(shape);
        graph.insert_node(Node::new(
            NodeId(0),
            "Reshape",
            vec![Some(data), Some(shape)],
            vec![output],
        ));
        graph.add_output(output);
        let mut model = Model::new(graph);
        let violations = TypeConstraintSatisfiedRule.check(&model, &ValidationContext::default());
        assert_eq!(violations.len(), 1, "{violations:?}");
        assert!(violations[0].message.contains("tensor(int64)"));

        model.graph.value_mut(shape).dtype = DataType::Int64;
        assert!(
            TypeConstraintSatisfiedRule
                .check(&model, &ValidationContext::default())
                .is_empty()
        );
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
    fn supported_ir_versions_pass_and_future_version_fails() {
        let rule = IrVersionSupportedRule;
        for ir_version in [1, 10, 13] {
            let mut model = Model::new(Graph::new());
            model.metadata.ir_version = ir_version;
            assert!(
                rule.check(&model, &ValidationContext::default()).is_empty(),
                "ir_version {ir_version}"
            );
        }
        let mut model = Model::new(Graph::new());
        model.metadata.ir_version = 14;
        let violations = rule.check(&model, &ValidationContext::default());
        assert_eq!(violations.len(), 1);
        assert!(violations[0].message.contains("newer than"));
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
    fn ir_feature_rule_checks_opset_initializer_and_dtype_gates() {
        let rule = IrVersionFeatureRule;
        let model = retained_model_at(
            ModelProto {
                opset_import: vec![OperatorSetIdProto {
                    domain: String::new(),
                    version: 1,
                }],
                ..Default::default()
            },
            2,
        );
        assert!(
            rule.check(&model, &ValidationContext::default())
                .iter()
                .any(|violation| violation.message.contains("must not specify opset_import"))
        );

        let initializer = TensorProto {
            dims: vec![1],
            data_type: tensor_proto::DataType::Float as i32,
            float_data: vec![1.0],
            name: "W".into(),
            ..Default::default()
        };
        let model = retained_model_at(
            ModelProto {
                graph: Some(GraphProto {
                    name: "graph".into(),
                    initializer: vec![initializer],
                    ..Default::default()
                }),
                ..Default::default()
            },
            3,
        );
        assert!(
            rule.check(&model, &ValidationContext::default())
                .iter()
                .any(|violation| violation.message.contains("must also be a graph input"))
        );

        let model = retained_model_at(
            ModelProto {
                graph: Some(GraphProto {
                    name: "graph".into(),
                    initializer: vec![TensorProto {
                        dims: vec![1],
                        data_type: tensor_proto::DataType::Uint2 as i32,
                        raw_data: vec![0],
                        name: "packed".into(),
                        ..Default::default()
                    }],
                    ..Default::default()
                }),
                ..Default::default()
            },
            12,
        );
        assert!(
            rule.check(&model, &ValidationContext::default())
                .iter()
                .any(|violation| violation.message.contains("requires IR version 13"))
        );
    }

    #[test]
    fn ir_feature_rule_accepts_model_metadata_before_ir10() {
        let model = retained_model_at(
            ModelProto {
                metadata_props: vec![StringStringEntryProto {
                    key: "owner".into(),
                    value: "onnx".into(),
                }],
                ..Default::default()
            },
            7,
        );
        let violations = IrVersionFeatureRule.check(&model, &ValidationContext::default());
        assert!(violations.is_empty(), "{violations:?}");
    }

    #[test]
    fn function_rule_accepts_valid_signature_and_body() {
        let model = retained_model(ModelProto {
            functions: vec![function(
                "Pass",
                vec![NodeProto {
                    input: vec!["X".into()],
                    output: vec!["Y".into()],
                    op_type: "Identity".into(),
                    ..Default::default()
                }],
            )],
            ..Default::default()
        });
        let violations = FunctionProtoValidityRule.check(&model, &ValidationContext::default());
        assert!(violations.is_empty(), "{violations:?}");
    }

    #[test]
    fn function_rule_accepts_empty_default_domain() {
        let mut default_domain = function(
            "Pass",
            vec![NodeProto {
                input: vec!["X".into()],
                output: vec!["Y".into()],
                op_type: "Identity".into(),
                ..Default::default()
            }],
        );
        default_domain.domain.clear();
        let model = retained_model(ModelProto {
            functions: vec![default_domain],
            ..Default::default()
        });
        let violations = FunctionProtoValidityRule.check(&model, &ValidationContext::default());
        assert!(violations.is_empty(), "{violations:?}");
    }

    #[test]
    fn function_rule_checks_defaults_topology_imports_and_unique_ids() {
        let mut invalid = function(
            "Broken",
            vec![NodeProto {
                input: vec!["missing".into()],
                output: vec!["Y".into()],
                op_type: "Identity".into(),
                ..Default::default()
            }],
        );
        invalid.attribute = vec!["alpha".into()];
        invalid.attribute_proto = vec![AttributeProto {
            name: "alpha".into(),
            r#type: attribute_proto::AttributeType::Float as i32,
            ..Default::default()
        }];
        invalid.opset_import.clear();
        let model = retained_model(ModelProto {
            functions: vec![invalid.clone(), invalid],
            ..Default::default()
        });
        let violations = FunctionProtoValidityRule.check(&model, &ValidationContext::default());
        for expected in [
            "is not unique",
            "both attribute and attribute_proto",
            "neither a function/graph input",
            "no opset import",
        ] {
            assert!(
                violations
                    .iter()
                    .any(|violation| violation.message.contains(expected)),
                "{expected}: {violations:?}"
            );
        }
    }

    #[test]
    fn function_rule_rejects_undeclared_attribute_refs_and_recursion() {
        let make_call = |callee: &str| NodeProto {
            input: vec!["X".into()],
            output: vec!["Y".into()],
            op_type: callee.into(),
            domain: "local.test".into(),
            attribute: vec![AttributeProto {
                name: "alpha".into(),
                ref_attr_name: "undeclared".into(),
                r#type: attribute_proto::AttributeType::Float as i32,
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut first = function("First", vec![make_call("Second")]);
        let mut second = function("Second", vec![make_call("First")]);
        for function in [&mut first, &mut second] {
            function.opset_import.push(OperatorSetIdProto {
                domain: "local.test".into(),
                version: 1,
            });
        }
        let model = retained_model(ModelProto {
            functions: vec![first, second],
            ..Default::default()
        });
        let violations = FunctionProtoValidityRule.check(&model, &ValidationContext::default());
        assert!(
            violations
                .iter()
                .any(|violation| violation.message.contains("undeclared function attribute"))
        );
        assert!(
            violations
                .iter()
                .any(|violation| violation.message.contains("are recursive"))
        );
    }

    #[test]
    fn function_rule_accepts_consistent_call_with_omitted_default_attribute() {
        let mut callee = function(
            "Affine",
            vec![NodeProto {
                input: vec!["X".into()],
                output: vec!["Y".into()],
                op_type: "Identity".into(),
                ..Default::default()
            }],
        );
        callee.attribute = vec!["alpha".into()];
        callee.attribute_proto = vec![AttributeProto {
            name: "beta".into(),
            r#type: attribute_proto::AttributeType::Float as i32,
            f: 1.0,
            ..Default::default()
        }];
        let model = local_function_call_model(
            callee,
            NodeProto {
                input: vec!["X".into()],
                output: vec!["Y".into()],
                op_type: "Affine".into(),
                domain: "local.test".into(),
                attribute: vec![AttributeProto {
                    name: "alpha".into(),
                    r#type: attribute_proto::AttributeType::Float as i32,
                    ..Default::default()
                }],
                ..Default::default()
            },
        );
        let violations = FunctionProtoValidityRule.check(&model, &ValidationContext::default());
        assert!(violations.is_empty(), "{violations:?}");
    }

    #[test]
    fn function_rule_accepts_call_site_arity_and_attribute_mismatches() {
        let callee = function(
            "Affine",
            vec![NodeProto {
                input: vec!["X".into()],
                output: vec!["Y".into()],
                op_type: "Identity".into(),
                ..Default::default()
            }],
        );
        let mut caller = function(
            "Caller",
            vec![NodeProto {
                input: vec!["X".into(), "X".into()],
                output: vec!["Y".into(), "unused".into()],
                op_type: "Affine".into(),
                domain: "local.test".into(),
                attribute: vec![AttributeProto {
                    name: "gamma".into(),
                    r#type: attribute_proto::AttributeType::Float as i32,
                    ..Default::default()
                }],
                ..Default::default()
            }],
        );
        caller.opset_import.push(OperatorSetIdProto {
            domain: "local.test".into(),
            version: 1,
        });
        let mut callee = callee;
        callee.attribute = vec!["alpha".into()];
        callee.attribute_proto = vec![AttributeProto {
            name: "beta".into(),
            r#type: attribute_proto::AttributeType::Float as i32,
            f: 1.0,
            ..Default::default()
        }];
        let model = retained_model(ModelProto {
            functions: vec![callee, caller],
            ..Default::default()
        });
        let violations = FunctionProtoValidityRule.check(&model, &ValidationContext::default());
        assert!(violations.is_empty(), "{violations:?}");
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
    fn round_four_optional_inputs_and_attributes_may_be_omitted() {
        for (op_type, inputs) in [
            ("ReduceMax", 1),
            ("ReduceMin", 1),
            ("ReduceProd", 1),
            ("ReduceL1", 1),
            ("ReduceL2", 1),
            ("ReduceLogSum", 1),
            ("ReduceLogSumExp", 1),
            ("ReduceSumSquare", 1),
            ("ArgMax", 1),
            ("ArgMin", 1),
            ("LogSoftmax", 1),
            ("RMSNormalization", 2),
        ] {
            let mut model = one_node_model(op_type, inputs, 1);
            model.graph.opset_imports.insert(String::new(), 24);
            if matches!(op_type, "ArgMax" | "ArgMin") {
                let output = model.graph.outputs[0];
                model.graph.value_mut(output).dtype = DataType::Int64;
            }
            let result = model.validate();
            assert!(
                result.is_valid(),
                "{op_type}: optional fields should be omittable: {:?}",
                result.violations
            );
        }
    }

    #[test]
    fn round_five_schemas_accept_official_minimal_forms() {
        for (op_type, inputs, outputs, int64_inputs, bool_inputs, int64_outputs, bool_outputs) in [
            ("GatherElements", 2, 1, vec![1], vec![], vec![], vec![]),
            ("GatherND", 2, 1, vec![1], vec![], vec![], vec![]),
            ("Equal", 2, 1, vec![], vec![], vec![], vec![0]),
            ("Greater", 2, 1, vec![], vec![], vec![], vec![0]),
            ("Less", 2, 1, vec![], vec![], vec![], vec![0]),
            ("And", 2, 1, vec![], vec![0, 1], vec![], vec![0]),
            ("Or", 2, 1, vec![], vec![0, 1], vec![], vec![0]),
            ("Not", 1, 1, vec![], vec![0], vec![], vec![0]),
            ("Shape", 1, 1, vec![], vec![], vec![0], vec![]),
            ("Size", 1, 1, vec![], vec![], vec![0], vec![]),
            ("NonZero", 1, 1, vec![], vec![], vec![0], vec![]),
            ("Range", 3, 1, vec![], vec![], vec![], vec![]),
            ("Split", 1, 2, vec![], vec![], vec![], vec![]),
        ] {
            let mut model = one_node_model(op_type, inputs, outputs);
            model.graph.opset_imports.insert(String::new(), 25);
            for index in int64_inputs {
                let value = model.graph.inputs[index];
                model.graph.value_mut(value).dtype = DataType::Int64;
            }

            for index in bool_inputs {
                let value = model.graph.inputs[index];
                model.graph.value_mut(value).dtype = DataType::Bool;
            }
            for index in int64_outputs {
                let value = model.graph.outputs[index];
                model.graph.value_mut(value).dtype = DataType::Int64;
            }
            for index in bool_outputs {
                let value = model.graph.outputs[index];
                model.graph.value_mut(value).dtype = DataType::Bool;
            }

            let result = model.validate();
            assert!(
                result.is_valid(),
                "{op_type}: optional fields should be omittable: {:?}",
                result.violations
            );
        }

        let mut cast = one_node_model("Cast", 1, 1);
        cast.graph.opset_imports.insert(String::new(), 25);
        let output = cast.graph.outputs[0];
        cast.graph.value_mut(output).dtype = DataType::Int64;
        let node = cast.graph.nodes.keys().next().unwrap();
        cast.graph.node_mut(node).attributes.insert(
            "to".into(),
            Attribute::Int(i64::from(DataType::Int64.to_onnx())),
        );
        assert!(cast.validate().is_valid());

        cast.graph.node_mut(node).attributes.remove("to");
        assert!(cast.validate().violations.iter().any(|violation| {
            violation.rule_id == "schema.node_conforms"
                && violation.message.contains("required attribute 'to'")
        }));
    }

    #[test]
    fn round_six_schemas_accept_official_minimal_forms() {
        for (op_type, inputs, int64_inputs) in [
            ("Tile", 2, vec![1]),
            ("Pad", 2, vec![1]),
            ("ScatterND", 3, vec![1]),
            ("ScatterElements", 3, vec![1]),
            ("ConstantOfShape", 1, vec![0]),
        ] {
            let mut model = one_node_model(op_type, inputs, 1);
            model.graph.opset_imports.insert(String::new(), 25);
            for index in int64_inputs {
                let value = model.graph.inputs[index];
                model.graph.value_mut(value).dtype = DataType::Int64;
            }
            let result = model.validate();
            assert!(
                result.is_valid(),
                "{op_type}: official minimal form should pass: {:?}",
                result.violations
            );
        }

        let mut pad_with_axes = one_node_model("Pad", 4, 1);
        pad_with_axes.graph.opset_imports.insert(String::new(), 25);
        let pads = pad_with_axes.graph.inputs[1];
        let axes = pad_with_axes.graph.inputs[3];
        pad_with_axes.graph.value_mut(pads).dtype = DataType::Int64;
        pad_with_axes.graph.value_mut(axes).dtype = DataType::Int32;
        let node = pad_with_axes.graph.nodes.keys().next().unwrap();
        pad_with_axes.graph.node_mut(node).inputs[2] = None;
        assert!(
            pad_with_axes.validate().is_valid(),
            "optional constant_value may be omitted while axes is present"
        );
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

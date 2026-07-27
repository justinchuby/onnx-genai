//! Built-in validation rules for this validation family.

use std::collections::{HashMap, HashSet};

use onnx_runtime_ir::{Attribute, Graph, ValueId};
use onnx_runtime_loader::proto::onnx::{
    AttributeProto, GraphProto, NodeProto, StringStringEntryProto, TensorProto, TensorShapeProto,
    TypeProto, ValueInfoProto, attribute_proto, tensor_proto, type_proto,
};

use super::super::{Severity, ValidationContext, ValidationRule, Violation, ViolationLocation};
use super::{arity_message, node_label, node_violation, normalize_domain, value_label};
use crate::model::Model;
use crate::schema::{AttributeType, OpSchema};

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

pub(super) fn check_attribute_proto(
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

pub(super) fn check_value_type(
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

pub(super) fn check_type_proto(
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

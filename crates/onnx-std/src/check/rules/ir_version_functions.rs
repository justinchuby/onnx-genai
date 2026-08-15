//! Built-in validation rules for this validation family.

use std::collections::{HashMap, HashSet};

use onnx_runtime_loader::proto::onnx::{
    AttributeProto, FunctionProto, GraphProto, ModelProto, NodeProto, SparseTensorProto,
    TensorProto, TypeProto, ValueInfoProto, attribute_proto, tensor_proto, type_proto,
};

use super::super::{Severity, ValidationContext, ValidationRule, Violation, ViolationLocation};
use super::{arity_message, model_violation, normalize_domain};
use crate::model::Model;
use crate::schema::{AttributeType, OpSchema};

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

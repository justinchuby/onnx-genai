//! Built-in validation rules for this validation family.

use std::collections::{HashMap, HashSet};

use onnx_runtime_loader::proto::onnx::{
    DeviceConfigurationProto, GraphProto, ModelProto, NodeProto, TensorShapeProto, type_proto,
};

use super::super::{Severity, ValidationContext, ValidationRule, Violation};
use super::{model_violation, proto_node_violation};
use crate::model::Model;

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

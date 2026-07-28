//! Tests for the `onnx_runtime.*` metadata hint scanner.

use super::*;
use crate::model::Model;
use onnx_runtime_ir::{DataType, Graph, Node, NodeId, static_shape};
use onnx_runtime_loader::ModelMetadata;

/// Build an `onnx_runtime.*` entry attached to a node.
fn node_entry(node: &str, key: &str, value: &str, source: HintSource) -> HintEntry {
    HintEntry {
        scope: HintScope::Node {
            node_name: node.to_string(),
        },
        source,
        key: key.to_string(),
        value: value.to_string(),
    }
}

/// Build a model-level `onnx_runtime.*` entry.
fn model_entry(key: &str, value: &str, source: HintSource) -> HintEntry {
    HintEntry {
        scope: HintScope::Model,
        source,
        key: key.to_string(),
        value: value.to_string(),
    }
}

#[test]
fn empty_input_yields_defaults_and_no_warnings() {
    let hints = MetadataHints::scan([]);
    assert_eq!(hints, MetadataHints::default());
    assert!(hints.nodes.is_empty());
    assert!(hints.warnings.is_empty());
    assert!(!hints.has_errors());
}

#[test]
fn valid_node_hints_are_consumed_with_correct_types() {
    let hints = MetadataHints::scan([
        node_entry(
            "attn",
            "onnx_runtime.device",
            "gpu:0",
            HintSource::OnnxMetadata,
        ),
        node_entry(
            "attn",
            "onnx_runtime.device.strength",
            "force",
            HintSource::OnnxMetadata,
        ),
        node_entry("attn", "onnx_runtime.layer", "7", HintSource::OnnxMetadata),
        node_entry(
            "attn",
            "onnx_runtime.offloadable",
            "true",
            HintSource::OnnxMetadata,
        ),
        node_entry(
            "attn",
            "onnx_runtime.kernel",
            "flash_attention",
            HintSource::OnnxMetadata,
        ),
        node_entry(
            "attn",
            "onnx_runtime.memory.priority",
            "low",
            HintSource::OnnxMetadata,
        ),
    ]);

    assert!(hints.warnings.is_empty());
    let node = hints.nodes.get("attn").expect("node hints present");
    assert_eq!(node.device.as_deref(), Some("gpu:0"));
    assert_eq!(node.device_strength, Some(PlacementStrength::Force));
    assert_eq!(node.layer, Some(7));
    assert_eq!(node.offloadable, Some(true));
    assert_eq!(node.kernel.as_deref(), Some("flash_attention"));
    assert_eq!(node.memory_priority.as_deref(), Some("low"));
}

#[test]
fn valid_model_and_graph_hints_are_consumed() {
    let hints = MetadataHints::scan([
        model_entry(
            "onnx_runtime.model.num_layers",
            "32",
            HintSource::OnnxMetadata,
        ),
        model_entry(
            "onnx_runtime.model.layer_pattern",
            "model.layers.{}",
            HintSource::OnnxMetadata,
        ),
        HintEntry {
            scope: HintScope::Graph {
                graph_name: "main".to_string(),
            },
            source: HintSource::OnnxMetadata,
            key: "onnx_runtime.memory.arena_gpu_mb".to_string(),
            value: "4096".to_string(),
        },
    ]);

    assert!(hints.warnings.is_empty());
    assert_eq!(hints.model.num_layers, Some(32));
    assert_eq!(
        hints.model.layer_pattern.as_deref(),
        Some("model.layers.{}")
    );
    assert_eq!(hints.model.arena_gpu_mb, Some(4096));
}

#[test]
fn unknown_namespace_key_warns_but_is_not_an_error() {
    let hints = MetadataHints::scan([node_entry(
        "attn",
        "onnx_runtime.devcie", // typo
        "gpu",
        HintSource::OnnxMetadata,
    )]);

    assert_eq!(
        hints.warnings,
        vec![MetadataWarning::UnknownKey {
            node: "attn".to_string(),
            key: "onnx_runtime.devcie".to_string(),
        }]
    );
    assert!(!hints.has_errors());
    assert!(!hints.nodes.contains_key("attn"));
}

#[test]
fn keys_outside_namespace_are_ignored() {
    let hints = MetadataHints::scan([
        model_entry("author", "deckard", HintSource::OnnxMetadata),
        model_entry("com.microsoft.foo", "bar", HintSource::OnnxMetadata),
    ]);
    assert!(hints.warnings.is_empty());
    assert_eq!(hints.model, ModelHints::default());
}

#[test]
fn node_level_key_at_model_scope_is_unknown() {
    // `onnx_runtime.kernel` is a node key; at model scope it is unrecognised.
    let hints = MetadataHints::scan([model_entry(
        "onnx_runtime.kernel",
        "cutlass",
        HintSource::OnnxMetadata,
    )]);
    assert_eq!(
        hints.warnings,
        vec![MetadataWarning::UnknownKey {
            node: String::new(),
            key: "onnx_runtime.kernel".to_string(),
        }]
    );
}

#[test]
fn malformed_integer_value_warns_with_expected_type() {
    let hints = MetadataHints::scan([node_entry(
        "attn",
        "onnx_runtime.layer",
        "not-a-number",
        HintSource::OnnxMetadata,
    )]);
    assert_eq!(
        hints.warnings,
        vec![MetadataWarning::InvalidValue {
            node: "attn".to_string(),
            key: "onnx_runtime.layer".to_string(),
            value: "not-a-number".to_string(),
            expected: "an integer",
        }]
    );
    assert!(!hints.nodes.contains_key("attn"));
}

#[test]
fn malformed_boolean_and_enum_values_warn() {
    let hints = MetadataHints::scan([
        node_entry(
            "n",
            "onnx_runtime.offloadable",
            "yes",
            HintSource::OnnxMetadata,
        ),
        node_entry(
            "n",
            "onnx_runtime.memory.priority",
            "urgent",
            HintSource::OnnxMetadata,
        ),
    ]);
    assert_eq!(hints.warnings.len(), 2);
    assert!(hints.warnings.iter().any(|w| matches!(
        w,
        MetadataWarning::InvalidValue { key, expected, .. }
            if key == "onnx_runtime.offloadable" && *expected == "a boolean (\"true\" or \"false\")"
    )));
    assert!(hints.warnings.iter().any(|w| matches!(
        w,
        MetadataWarning::InvalidValue { key, expected, .. }
            if key == "onnx_runtime.memory.priority" && *expected == "one of: high, low, normal"
    )));
}

#[test]
fn boolean_parsing_is_case_insensitive() {
    let hints = MetadataHints::scan([
        node_entry(
            "n",
            "onnx_runtime.scheduling.cuda_graph",
            "TRUE",
            HintSource::OnnxMetadata,
        ),
        node_entry(
            "n",
            "onnx_runtime.scheduling.overlap",
            "False",
            HintSource::OnnxMetadata,
        ),
    ]);
    assert!(hints.warnings.is_empty());
    let node = hints.nodes.get("n").unwrap();
    assert_eq!(node.cuda_graph, Some(true));
    assert_eq!(node.overlap, Some(false));
}

#[test]
fn higher_priority_source_wins_for_the_same_key() {
    // Feed lowest priority last to prove ordering is by priority, not insertion.
    let hints = MetadataHints::scan([
        model_entry(
            "onnx_runtime.model.architecture",
            "llama",
            HintSource::ProgrammaticBuilder,
        ),
        model_entry(
            "onnx_runtime.model.architecture",
            "phi",
            HintSource::OnnxMetadata,
        ),
    ]);
    assert!(hints.warnings.is_empty());
    assert_eq!(hints.model.architecture.as_deref(), Some("llama"));
}

#[test]
fn higher_priority_source_wins_for_node_scalar_hint() {
    let hints = MetadataHints::scan([
        node_entry("n", "onnx_runtime.layer", "3", HintSource::OnnxMetadata),
        node_entry(
            "n",
            "onnx_runtime.layer",
            "9",
            HintSource::ExecutionHintsJson,
        ),
    ]);
    assert_eq!(hints.nodes.get("n").unwrap().layer, Some(9));
}

#[test]
fn duplicate_key_from_same_source_keeps_first() {
    let hints = MetadataHints::scan([
        node_entry(
            "n",
            "onnx_runtime.group",
            "attn_0",
            HintSource::OnnxMetadata,
        ),
        node_entry(
            "n",
            "onnx_runtime.group",
            "attn_9",
            HintSource::OnnxMetadata,
        ),
    ]);
    assert!(hints.warnings.is_empty());
    assert_eq!(
        hints.nodes.get("n").unwrap().group.as_deref(),
        Some("attn_0")
    );
}

#[test]
fn force_overrides_prefer_regardless_of_source_priority() {
    // A low-priority forced device beats a high-priority preferred one.
    let hints = MetadataHints::scan([
        node_entry(
            "n",
            "onnx_runtime.device",
            "cpu",
            HintSource::ProgrammaticBuilder,
        ),
        node_entry("n", "onnx_runtime.device", "gpu", HintSource::OnnxMetadata),
        node_entry(
            "n",
            "onnx_runtime.device.strength",
            "force",
            HintSource::OnnxMetadata,
        ),
    ]);
    assert!(!hints.has_errors());
    let node = hints.nodes.get("n").unwrap();
    assert_eq!(node.device.as_deref(), Some("gpu"));
    assert_eq!(node.device_strength, Some(PlacementStrength::Force));
}

#[test]
fn contradicting_force_devices_are_a_hard_error() {
    let hints = MetadataHints::scan([
        node_entry("n", "onnx_runtime.device", "gpu", HintSource::OnnxMetadata),
        node_entry(
            "n",
            "onnx_runtime.device.strength",
            "force",
            HintSource::OnnxMetadata,
        ),
        node_entry(
            "n",
            "onnx_runtime.device",
            "cpu",
            HintSource::ExecutionHintsJson,
        ),
        node_entry(
            "n",
            "onnx_runtime.device.strength",
            "force",
            HintSource::ExecutionHintsJson,
        ),
    ]);
    assert!(hints.has_errors());
    assert!(hints.warnings.iter().any(|w| matches!(
        w,
        MetadataWarning::ConflictingForce { node, .. } if node == "n"
    )));
}

#[test]
fn preferred_device_defaults_strength_to_prefer() {
    let hints = MetadataHints::scan([node_entry(
        "n",
        "onnx_runtime.device",
        "npu",
        HintSource::OnnxMetadata,
    )]);
    let node = hints.nodes.get("n").unwrap();
    assert_eq!(node.device.as_deref(), Some("npu"));
    assert_eq!(node.device_strength, Some(PlacementStrength::Prefer));
}

#[test]
fn from_model_reads_model_level_metadata_props() {
    let metadata = ModelMetadata {
        metadata_props: vec![
            ("onnx_runtime.version".to_string(), "1".to_string()),
            (
                "onnx_runtime.model.num_layers".to_string(),
                "32".to_string(),
            ),
            ("author".to_string(), "ignored".to_string()),
        ],
        ..Default::default()
    };
    let model = Model::with_metadata(add_graph(), metadata);
    let hints = MetadataHints::from_model(&model);
    assert!(hints.warnings.is_empty());
    assert_eq!(hints.model.version.as_deref(), Some("1"));
    assert_eq!(hints.model.num_layers, Some(32));
}

#[test]
fn from_model_reads_node_level_metadata_props() {
    let model = Model::with_metadata(add_graph(), ModelMetadata::default());
    let mut proto = model.to_proto().expect("encode proto");
    let graph = proto.graph.as_mut().expect("graph present");
    let node = graph
        .node
        .iter_mut()
        .find(|n| n.name == "add0")
        .expect("named node present");
    node.metadata_props = vec![
        string_entry("onnx_runtime.device", "gpu"),
        string_entry("onnx_runtime.layer", "0"),
        string_entry("onnx_runtime.unknown_key", "x"),
    ];

    let reloaded = Model::from_proto(proto).expect("model from proto");
    let hints = MetadataHints::from_model(&reloaded);

    let node_hints = hints.nodes.get("add0").expect("node hints present");
    assert_eq!(node_hints.device.as_deref(), Some("gpu"));
    assert_eq!(node_hints.layer, Some(0));
    assert!(hints.warnings.iter().any(|w| matches!(
        w,
        MetadataWarning::UnknownKey { key, .. } if key == "onnx_runtime.unknown_key"
    )));
}

#[test]
fn from_model_with_no_metadata_is_safe() {
    let model = Model::new(add_graph());
    let hints = MetadataHints::from_model(&model);
    assert_eq!(hints, MetadataHints::default());
}

fn string_entry(
    key: &str,
    value: &str,
) -> onnx_runtime_loader::proto::onnx::StringStringEntryProto {
    onnx_runtime_loader::proto::onnx::StringStringEntryProto {
        key: key.to_string(),
        value: value.to_string(),
    }
}

/// Build a tiny `Z = Add(X, Y)` graph with a named node.
fn add_graph() -> Graph {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 21);
    let x = g.create_named_value("X", DataType::Float32, static_shape([2, 3]));
    let y = g.create_named_value("Y", DataType::Float32, static_shape([2, 3]));
    let z = g.create_named_value("Z", DataType::Float32, static_shape([2, 3]));
    g.add_input(x);
    g.add_input(y);
    let mut node = Node::new(NodeId(0), "Add", vec![Some(x), Some(y)], vec![z]);
    node.name = "add0".to_string();
    g.insert_node(node);
    g.add_output(z);
    g
}

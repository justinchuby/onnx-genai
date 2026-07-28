//! Tests for the `onnx_runtime.*` metadata hint scanner.

use super::*;
use crate::model::Model;
use onnx_runtime_ir::{DataType, Graph, Node, NodeId, static_shape};
use onnx_runtime_loader::ModelMetadata;
use onnx_runtime_loader::proto::onnx::{AttributeProto, GraphProto, NodeProto, attribute_proto};

/// Build an `onnx_runtime.*` entry attached to a node by name (external source).
fn node_entry(node: &str, key: &str, value: &str, source: HintSource) -> HintEntry {
    HintEntry {
        scope: HintScope::NamedNode {
            name: node.to_string(),
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
fn control_flow_subgraph_hints_are_scanned_with_qualified_paths() {
    for (op_type, owner_name, attribute_name) in [
        ("Loop", "loop", "body"),
        ("If", "if", "then_branch"),
        ("If", "if", "else_branch"),
        ("Scan", "scan", "body"),
    ] {
        let subgraph = GraphProto {
            metadata_props: vec![string_entry("onnx_runtime.model.architecture", "nested")],
            node: vec![proto_node("inner", [("onnx_runtime.device", "gpu:1")])],
            ..Default::default()
        };
        let hints = scan_proto_graph(GraphProto {
            node: vec![proto_node_with_graph(
                op_type,
                owner_name,
                attribute_name,
                subgraph,
            )],
            ..Default::default()
        });

        assert!(hints.warnings.is_empty(), "{op_type}/{attribute_name}");
        assert_eq!(hints.model.architecture.as_deref(), Some("nested"));
        assert_eq!(
            hints
                .nodes
                .get(&format!("{owner_name}/{attribute_name}/inner"))
                .and_then(|node| node.device.as_deref()),
            Some("gpu:1"),
            "{op_type}/{attribute_name}"
        );
    }
}

#[test]
fn unknown_key_inside_subgraph_warns_with_qualified_node() {
    let hints = scan_proto_graph(graph_with_nested_node(
        "loop",
        "body",
        proto_node("inner", [("onnx_runtime.devcie", "gpu")]),
    ));

    assert_eq!(
        hints.warnings,
        vec![MetadataWarning::UnknownKey {
            node: "loop/body/inner".to_string(),
            key: "onnx_runtime.devcie".to_string(),
        }]
    );
}

#[test]
fn invalid_value_inside_subgraph_warns_with_qualified_node() {
    let hints = scan_proto_graph(graph_with_nested_node(
        "scan",
        "body",
        proto_node("inner", [("onnx_runtime.layer", "first")]),
    ));

    assert_eq!(
        hints.warnings,
        vec![MetadataWarning::InvalidValue {
            node: "scan/body/inner".to_string(),
            key: "onnx_runtime.layer".to_string(),
            value: "first".to_string(),
            expected: "an integer",
        }]
    );
}

#[test]
fn deeply_nested_graph_attributes_are_scanned() {
    let leaf = proto_node("leaf", [("onnx_runtime.offloadable", "true")]);
    let loop_node = proto_node_with_graph(
        "Loop",
        "loop",
        "body",
        GraphProto {
            node: vec![leaf],
            ..Default::default()
        },
    );
    let hints = scan_proto_graph(GraphProto {
        node: vec![proto_node_with_graph(
            "If",
            "if",
            "then_branch",
            GraphProto {
                node: vec![loop_node],
                ..Default::default()
            },
        )],
        ..Default::default()
    });

    assert_eq!(
        hints
            .nodes
            .get("if/then_branch/loop/body/leaf")
            .and_then(|node| node.offloadable),
        Some(true)
    );
}

#[test]
fn top_level_node_identity_is_unchanged_when_subgraphs_are_scanned() {
    let mut top = proto_node("shared", [("onnx_runtime.layer", "1")]);
    top.attribute.push(graph_attribute(
        "body",
        GraphProto {
            node: vec![proto_node("shared", [("onnx_runtime.layer", "2")])],
            ..Default::default()
        },
    ));
    let hints = scan_proto_graph(GraphProto {
        node: vec![top],
        ..Default::default()
    });

    assert_eq!(
        hints.nodes.get("shared").and_then(|node| node.layer),
        Some(1)
    );
    assert_eq!(
        hints
            .nodes
            .get("shared/body/shared")
            .and_then(|node| node.layer),
        Some(2)
    );
}

#[test]
fn nested_node_identity_does_not_collide_with_slash_named_top_level_node() {
    let nested_path = NodePath::root_node("owner".to_string(), 1)
        .with_attribute("body".to_string(), 0, None)
        .with_node("inner".to_string(), 0);
    let hints = scan_proto_graph(GraphProto {
        node: vec![
            proto_node("owner/body/inner", [("onnx_runtime.layer", "1")]),
            proto_node_with_graph(
                "Loop",
                "owner",
                "body",
                GraphProto {
                    node: vec![proto_node("inner", [("onnx_runtime.layer", "2")])],
                    ..Default::default()
                },
            ),
        ],
        ..Default::default()
    });

    assert_eq!(hints.nodes.len(), 2);
    assert_eq!(
        hints
            .nodes
            .get("owner/body/inner")
            .and_then(|node| node.layer),
        Some(1)
    );
    assert_eq!(
        hints
            .nodes
            .get_path(&nested_path)
            .and_then(|node| node.layer),
        Some(2)
    );
}

#[test]
fn graph_and_graphs_paths_that_render_the_same_stay_distinct() {
    let singular_path = NodePath::root_node("owner".to_string(), 0)
        .with_attribute("branches[0]".to_string(), 0, None)
        .with_node("inner".to_string(), 0);
    let repeated_path = NodePath::root_node("owner".to_string(), 0)
        .with_attribute("branches".to_string(), 1, Some(0))
        .with_node("inner".to_string(), 0);
    let owner = NodeProto {
        name: "owner".to_string(),
        attribute: vec![
            graph_attribute(
                "branches[0]",
                GraphProto {
                    node: vec![proto_node("inner", [("onnx_runtime.layer", "10")])],
                    ..Default::default()
                },
            ),
            AttributeProto {
                name: "branches".to_string(),
                r#type: attribute_proto::AttributeType::Graphs as i32,
                graphs: vec![GraphProto {
                    node: vec![proto_node("inner", [("onnx_runtime.layer", "20")])],
                    ..Default::default()
                }],
                ..Default::default()
            },
        ],
        ..Default::default()
    };

    let hints = scan_proto_graph(GraphProto {
        node: vec![owner],
        ..Default::default()
    });

    assert_eq!(singular_path.display_name(), repeated_path.display_name());
    assert_eq!(hints.nodes.len(), 2);
    assert_eq!(
        hints
            .nodes
            .get_path(&singular_path)
            .and_then(|node| node.layer),
        Some(10)
    );
    assert_eq!(
        hints
            .nodes
            .get_path(&repeated_path)
            .and_then(|node| node.layer),
        Some(20)
    );
}

#[test]
fn from_model_with_no_metadata_is_safe() {
    let model = Model::new(add_graph());
    let hints = MetadataHints::from_model(&model);
    assert_eq!(hints, MetadataHints::default());
}

// --- Node-identity collision classes ------------------------------------
//
// Node names in ONNX are optional and non-unique, so they must never key a
// node. Each test below pins one collision class shut. The `_path` helpers
// rebuild the structural key the scan produces; if any node were keyed by its
// raw name string again, the `len()`/per-node assertions would regress.

/// Structural key of the top-level node at `index` named `name`.
fn top_level_path(name: &str, index: usize) -> NodePath {
    NodePath::root_node(name.to_string(), index)
}

#[test]
fn two_anonymous_top_level_nodes_stay_distinct() {
    // The round-3 reviewer probe: two unnamed top-level nodes must not merge
    // under an empty-string key. Each keeps its own hints.
    let hints = scan_proto_graph(GraphProto {
        node: vec![
            proto_node("", [("onnx_runtime.layer", "1")]),
            proto_node("", [("onnx_runtime.layer", "2")]),
        ],
        ..Default::default()
    });

    assert_eq!(hints.nodes.len(), 2);
    assert_eq!(
        hints
            .nodes
            .get_path(&top_level_path("", 0))
            .and_then(|node| node.layer),
        Some(1)
    );
    assert_eq!(
        hints
            .nodes
            .get_path(&top_level_path("", 1))
            .and_then(|node| node.layer),
        Some(2)
    );
}

#[test]
fn two_same_named_top_level_nodes_stay_distinct() {
    // ONNX does not require node names to be unique; duplicate names must not
    // collide.
    let hints = scan_proto_graph(GraphProto {
        node: vec![
            proto_node("dup", [("onnx_runtime.layer", "1")]),
            proto_node("dup", [("onnx_runtime.layer", "2")]),
        ],
        ..Default::default()
    });

    assert_eq!(hints.nodes.len(), 2);
    assert_eq!(
        hints
            .nodes
            .get_path(&top_level_path("dup", 0))
            .and_then(|node| node.layer),
        Some(1)
    );
    assert_eq!(
        hints
            .nodes
            .get_path(&top_level_path("dup", 1))
            .and_then(|node| node.layer),
        Some(2)
    );
}

#[test]
fn top_level_name_matching_a_subgraph_display_path_stays_distinct() {
    // A top-level node literally named "owner/body/inner" must not merge with a
    // nested node whose display path renders to the same string.
    let nested_path = NodePath::root_node("owner".to_string(), 1)
        .with_attribute("body".to_string(), 0, None)
        .with_node("inner".to_string(), 0);
    let hints = scan_proto_graph(GraphProto {
        node: vec![
            proto_node("owner/body/inner", [("onnx_runtime.layer", "1")]),
            proto_node_with_graph(
                "Loop",
                "owner",
                "body",
                GraphProto {
                    node: vec![proto_node("inner", [("onnx_runtime.layer", "2")])],
                    ..Default::default()
                },
            ),
        ],
        ..Default::default()
    });

    assert_eq!(hints.nodes.len(), 2);
    assert_eq!(
        hints
            .nodes
            .get_path(&top_level_path("owner/body/inner", 0))
            .and_then(|node| node.layer),
        Some(1)
    );
    assert_eq!(
        hints
            .nodes
            .get_path(&nested_path)
            .and_then(|node| node.layer),
        Some(2)
    );
}

#[test]
fn external_named_hint_resolves_onto_the_structural_node_it_names() {
    // External sources address nodes by name; the name must resolve to the
    // model's structural node so priorities merge, not create a second entry.
    let hints = MetadataHints::scan([
        // Model-embedded structural hint on top-level node "attn" (index 0).
        HintEntry {
            scope: HintScope::Node {
                path: top_level_path("attn", 0),
            },
            source: HintSource::OnnxMetadata,
            key: "onnx_runtime.layer".to_string(),
            value: "1".to_string(),
        },
        // External JSON references the same node by name with higher priority.
        node_entry(
            "attn",
            "onnx_runtime.layer",
            "9",
            HintSource::ExecutionHintsJson,
        ),
    ]);

    assert_eq!(hints.nodes.len(), 1);
    assert_eq!(hints.nodes.get("attn").and_then(|node| node.layer), Some(9));
    assert_eq!(
        hints
            .nodes
            .get_path(&top_level_path("attn", 0))
            .and_then(|node| node.layer),
        Some(9)
    );
}

#[test]
fn external_named_hint_for_unknown_node_is_kept_without_colliding() {
    // A name that matches no scanned node stays in its own external keyspace and
    // never merges with an anonymous structural node.
    let hints = MetadataHints::scan([
        HintEntry {
            scope: HintScope::Node {
                path: top_level_path("", 0),
            },
            source: HintSource::OnnxMetadata,
            key: "onnx_runtime.layer".to_string(),
            value: "1".to_string(),
        },
        node_entry(
            "ghost",
            "onnx_runtime.layer",
            "7",
            HintSource::ProgrammaticBuilder,
        ),
    ]);

    assert_eq!(hints.nodes.len(), 2);
    assert_eq!(
        hints
            .nodes
            .get_path(&top_level_path("", 0))
            .and_then(|node| node.layer),
        Some(1)
    );
    assert_eq!(
        hints.nodes.get("ghost").and_then(|node| node.layer),
        Some(7)
    );
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

fn proto_node<const N: usize>(name: &str, metadata: [(&str, &str); N]) -> NodeProto {
    NodeProto {
        name: name.to_string(),
        op_type: "Identity".to_string(),
        metadata_props: metadata
            .into_iter()
            .map(|(key, value)| string_entry(key, value))
            .collect(),
        ..Default::default()
    }
}

fn proto_node_with_graph(
    op_type: &str,
    name: &str,
    attribute_name: &str,
    graph: GraphProto,
) -> NodeProto {
    NodeProto {
        name: name.to_string(),
        op_type: op_type.to_string(),
        attribute: vec![graph_attribute(attribute_name, graph)],
        ..Default::default()
    }
}

fn graph_attribute(name: &str, graph: GraphProto) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: attribute_proto::AttributeType::Graph as i32,
        g: Some(graph),
        ..Default::default()
    }
}

fn graph_with_nested_node(owner: &str, attribute: &str, node: NodeProto) -> GraphProto {
    GraphProto {
        node: vec![proto_node_with_graph(
            "CustomControlFlow",
            owner,
            attribute,
            GraphProto {
                node: vec![node],
                ..Default::default()
            },
        )],
        ..Default::default()
    }
}

fn scan_proto_graph(graph: GraphProto) -> MetadataHints {
    let mut entries = Vec::new();
    collect_graph_hint_entries(&graph, &mut entries);
    MetadataHints::scan(entries)
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

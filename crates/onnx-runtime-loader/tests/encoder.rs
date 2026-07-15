//! Round-trip tests for the ONNX **encoder** (`encode → decode` fidelity).
//!
//! Each test builds a synthetic `ModelProto`, loads it through the full loader
//! pipeline into an IR `Graph`, re-encodes that graph with the encoder, decodes
//! the produced bytes again, and asserts structural + byte-exact equality. This
//! is the inverse of `graph_builder`/`weights` and the foundation of the §55.4
//! EPContext dump path.

use std::collections::HashMap;

use prost::Message;

use onnx_runtime_ir::{Attribute, Dim, Graph};
use onnx_runtime_loader::proto::onnx;
use onnx_runtime_loader::{
    Model, ModelMetadata, WeightStore, encode_model, encode_model_proto,
    load_model_bytes_with_weights, write_model,
};

// ── proto construction helpers ────────────────────────────────────────────────

enum Dimlike {
    Static(i64),
    Param(&'static str),
}

fn tensor_type(elem_type: i32, dims: &[Dimlike]) -> onnx::TypeProto {
    use onnx::tensor_shape_proto::{Dimension, dimension::Value as DV};
    let dim = dims
        .iter()
        .map(|d| Dimension {
            value: Some(match d {
                Dimlike::Static(n) => DV::DimValue(*n),
                Dimlike::Param(p) => DV::DimParam(p.to_string()),
            }),
            ..Default::default()
        })
        .collect();
    onnx::TypeProto {
        value: Some(onnx::type_proto::Value::TensorType(
            onnx::type_proto::Tensor {
                elem_type,
                shape: Some(onnx::TensorShapeProto { dim }),
            },
        )),
        ..Default::default()
    }
}

fn value_info(name: &str, elem_type: i32, dims: &[Dimlike]) -> onnx::ValueInfoProto {
    onnx::ValueInfoProto {
        name: name.to_string(),
        r#type: Some(tensor_type(elem_type, dims)),
        ..Default::default()
    }
}

fn raw_initializer(name: &str, data_type: i32, dims: &[i64], raw: Vec<u8>) -> onnx::TensorProto {
    onnx::TensorProto {
        name: name.to_string(),
        data_type,
        dims: dims.to_vec(),
        raw_data: raw,
        ..Default::default()
    }
}

fn node(op: &str, inputs: &[&str], outputs: &[&str]) -> onnx::NodeProto {
    onnx::NodeProto {
        op_type: op.to_string(),
        input: inputs.iter().map(|s| s.to_string()).collect(),
        output: outputs.iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    }
}

fn float_attr(name: &str, v: f32) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.into(),
        r#type: onnx::attribute_proto::AttributeType::Float as i32,
        f: v,
        ..Default::default()
    }
}

fn int_attr(name: &str, v: i64) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.into(),
        r#type: onnx::attribute_proto::AttributeType::Int as i32,
        i: v,
        ..Default::default()
    }
}

fn ints_attr(name: &str, v: &[i64]) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.into(),
        r#type: onnx::attribute_proto::AttributeType::Ints as i32,
        ints: v.to_vec(),
        ..Default::default()
    }
}

fn floats_attr(name: &str, v: &[f32]) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.into(),
        r#type: onnx::attribute_proto::AttributeType::Floats as i32,
        floats: v.to_vec(),
        ..Default::default()
    }
}

fn str_attr(name: &str, v: &str) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.into(),
        r#type: onnx::attribute_proto::AttributeType::String as i32,
        s: v.as_bytes().to_vec(),
        ..Default::default()
    }
}

fn strings_attr(name: &str, v: &[&str]) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.into(),
        r#type: onnx::attribute_proto::AttributeType::Strings as i32,
        strings: v.iter().map(|s| s.as_bytes().to_vec()).collect(),
        ..Default::default()
    }
}

/// A STRING attribute carrying *raw* (possibly non-UTF-8) bytes — the shape of
/// an `ep_cache_context` opaque blob.
fn bytes_attr(name: &str, bytes: &[u8]) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.into(),
        r#type: onnx::attribute_proto::AttributeType::String as i32,
        s: bytes.to_vec(),
        ..Default::default()
    }
}

fn epctx_node(
    inputs: &[&str],
    outputs: &[&str],
    attrs: Vec<onnx::AttributeProto>,
) -> onnx::NodeProto {
    onnx::NodeProto {
        op_type: "EPContext".into(),
        domain: "com.microsoft".into(),
        input: inputs.iter().map(|s| s.to_string()).collect(),
        output: outputs.iter().map(|s| s.to_string()).collect(),
        attribute: attrs,
        ..Default::default()
    }
}

// ── model under test ──────────────────────────────────────────────────────────

/// An opaque binary blob including bytes that are NOT valid UTF-8 (0x00, 0x80,
/// 0xFF, a lone 0xC3 continuation) — proves the encoder preserves it byte-exact.
fn opaque_blob() -> Vec<u8> {
    vec![
        0x00, 0x01, 0x80, 0xFE, 0xFF, b'v', b'1', 0x00, 0xC3, 0x28, 0x7F,
    ]
}

/// Build a diverse model exercising: symbolic dims, float/int64 initializers
/// with raw bytes, standard nodes with shape inference (MatMul → Add), and a
/// `com.microsoft::EPContext` node carrying an opaque blob plus every scalar and
/// list attribute kind.
fn build_model_bytes() -> Vec<u8> {
    // W: float32 [4,3] with distinct byte pattern; B: float32 [3]; idx: int64 [2]
    // (unused, present to exercise int64 initializer byte-exactness).
    let w_raw: Vec<u8> = (0..12u32)
        .flat_map(|i| (i as f32 * 0.25).to_le_bytes())
        .collect();
    let b_raw: Vec<u8> = [1.5f32, -2.5, 3.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let idx_raw: Vec<u8> = [7i64, -9].iter().flat_map(|v| v.to_le_bytes()).collect();

    let graph = onnx::GraphProto {
        name: "roundtrip".into(),
        input: vec![value_info(
            "X",
            1,
            &[Dimlike::Param("batch"), Dimlike::Static(4)],
        )],
        output: vec![value_info(
            "Y",
            1,
            &[Dimlike::Param("batch"), Dimlike::Static(3)],
        )],
        initializer: vec![
            raw_initializer("W", 1, &[4, 3], w_raw),
            raw_initializer("B", 1, &[3], b_raw),
            raw_initializer("idx", 7, &[2], idx_raw),
        ],
        node: vec![
            node("MatMul", &["X", "W"], &["M"]),
            node("Add", &["M", "B"], &["A"]),
            epctx_node(
                &["A"],
                &["Y"],
                vec![
                    int_attr("main_context", 1),
                    int_attr("embed_mode", 1),
                    bytes_attr("ep_cache_context", &opaque_blob()),
                    str_attr("source", "VendorEP"),
                    str_attr("ep_sdk_version", "sdk-2.1"),
                    str_attr("partition_name", "part_0"),
                    float_attr("alpha", 0.25),
                    int_attr("k", 42),
                    ints_attr("shape_hint", &[2, 3, 5]),
                    floats_attr("scales", &[0.5, 1.5, -2.0]),
                    strings_attr("tags", &["alpha", "beta"]),
                ],
            ),
        ],
        ..Default::default()
    };

    let m = onnx::ModelProto {
        ir_version: 8,
        opset_import: vec![
            onnx::OperatorSetIdProto {
                domain: String::new(),
                version: 17,
            },
            onnx::OperatorSetIdProto {
                domain: "com.microsoft".into(),
                version: 1,
            },
        ],
        producer_name: "roy".into(),
        producer_version: "1.2.3".into(),
        domain: "com.example".into(),
        model_version: 5,
        doc_string: "round-trip fixture".into(),
        graph: Some(graph),
        ..Default::default()
    };
    m.encode_to_vec()
}

// ── assertions ────────────────────────────────────────────────────────────────

/// Map initializer name → raw bytes, resolved through the weight store.
fn initializer_bytes(graph: &Graph, store: &WeightStore) -> HashMap<String, Vec<u8>> {
    graph
        .initializers
        .iter()
        .map(|(&vid, w)| {
            let name = graph.value(vid).name.clone().expect("initializer named");
            let bytes = store.bytes(w).expect("weight bytes live").to_vec();
            (name, bytes)
        })
        .collect()
}

/// The op_types of a graph's nodes in node-id order.
fn op_types(graph: &Graph) -> Vec<(String, String)> {
    let mut ids: Vec<_> = graph.nodes.keys().collect();
    ids.sort_by_key(|n| n.0);
    ids.into_iter()
        .map(|id| {
            let n = graph.node(id);
            (n.op_type.clone(), n.domain.clone())
        })
        .collect()
}

fn find_ep_node(graph: &Graph) -> onnx_runtime_ir::NodeId {
    graph
        .nodes
        .iter()
        .find(|(_, n)| n.op_type == "EPContext")
        .map(|(id, _)| id)
        .expect("EPContext node present")
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[test]
fn round_trip_structural_and_byte_exact() {
    let bytes1 = build_model_bytes();
    let (graph1, store1) = load_model_bytes_with_weights(&bytes1, ".").expect("decode 1");

    // Encode the loaded graph back to bytes (with model metadata mirrored).
    let model = Model::new(&graph1)
        .with_weights(&store1)
        .with_metadata(ModelMetadata {
            ir_version: 8,
            producer_name: "roy".into(),
            producer_version: "1.2.3".into(),
            domain: "com.example".into(),
            model_version: 5,
            doc_string: Some("round-trip fixture".into()),
            graph_name: "roundtrip".into(),
            metadata_props: vec![("author".into(), "roy".into())],
        });
    let bytes2 = encode_model(&model).expect("encode");

    // Decode the produced bytes again.
    let (graph2, store2) = load_model_bytes_with_weights(&bytes2, ".").expect("decode 2");

    // Node count + op_types (with domains) identical.
    assert_eq!(graph1.num_nodes(), graph2.num_nodes());
    assert_eq!(op_types(&graph1), op_types(&graph2));

    // Graph I/O names identical.
    let io_names = |g: &Graph, ids: &[onnx_runtime_ir::ValueId]| -> Vec<String> {
        ids.iter()
            .map(|&v| g.value(v).name.clone().unwrap())
            .collect()
    };
    assert_eq!(
        io_names(&graph1, &graph1.inputs),
        io_names(&graph2, &graph2.inputs)
    );
    assert_eq!(
        io_names(&graph1, &graph1.outputs),
        io_names(&graph2, &graph2.outputs)
    );

    // Opset imports identical.
    assert_eq!(graph1.opset_imports, graph2.opset_imports);

    // Initializer bytes byte-exact for every named initializer (float + int64).
    let init1 = initializer_bytes(&graph1, &store1);
    let init2 = initializer_bytes(&graph2, &store2);
    assert_eq!(init1.keys().len(), 3);
    assert_eq!(init1, init2, "initializer bytes must survive round-trip");

    // A symbolic input dim ("batch") survives as symbolic; the static dim stays.
    let x1 = graph2.value(graph2.inputs[0]);
    assert!(
        matches!(x1.shape[0], Dim::Symbolic(_)),
        "batch stays symbolic"
    );
    assert_eq!(x1.shape[1], Dim::Static(4));

    // EPContext attributes: opaque blob byte-exact, and every attribute kind
    // survives with its value.
    let ep1 = graph1.node(find_ep_node(&graph1));
    let ep2 = graph2.node(find_ep_node(&graph2));

    // ep_cache_context is preserved as a raw-bytes STRING attribute, byte-exact
    // — through the generic string-bytes path, with no op-specific handling.
    match ep2.attr("ep_cache_context") {
        Some(Attribute::String(bytes)) => {
            assert_eq!(bytes, &opaque_blob(), "opaque blob must be byte-exact");
        }
        other => panic!("ep_cache_context should be a STRING attribute, got {other:?}"),
    }

    assert_eq!(ep2.attr("alpha").and_then(Attribute::as_float), Some(0.25));
    assert_eq!(ep2.attr("k").and_then(Attribute::as_int), Some(42));
    assert_eq!(
        ep2.attr("shape_hint").and_then(Attribute::as_ints),
        Some(&[2, 3, 5][..])
    );
    assert_eq!(
        ep2.attr("source").and_then(Attribute::as_str),
        Some("VendorEP")
    );
    match ep2.attr("scales") {
        Some(Attribute::Floats(v)) => assert_eq!(v, &vec![0.5, 1.5, -2.0]),
        other => panic!("scales should be Floats, got {other:?}"),
    }
    match ep2.attr("tags") {
        Some(Attribute::Strings(v)) => assert_eq!(
            v,
            &vec![b"alpha".to_vec(), b"beta".to_vec()],
            "string-list bytes must survive round-trip"
        ),
        other => panic!("tags should be Strings, got {other:?}"),
    }

    // The first decode's EPContext opaque payload matches the second decode's
    // exactly (proves no drift across the round-trip).
    match (ep1.attr("ep_cache_context"), ep2.attr("ep_cache_context")) {
        (Some(Attribute::String(a)), Some(Attribute::String(b))) => assert_eq!(a, b),
        _ => panic!("ep_cache_context missing"),
    }
}

#[test]
fn encoded_ep_cache_context_is_a_string_attribute_on_the_wire() {
    // The opaque UINT8 blob must be emitted as an ONNX STRING attribute (bytes in
    // `AttributeProto.s`), matching upstream ORT and the load path's expectation.
    let bytes1 = build_model_bytes();
    let (graph1, store1) = load_model_bytes_with_weights(&bytes1, ".").expect("decode 1");
    let model = Model::new(&graph1).with_weights(&store1);
    let proto = encode_model_proto(&model).expect("encode proto");

    let g = proto.graph.expect("graph");
    let ep = g
        .node
        .iter()
        .find(|n| n.op_type == "EPContext")
        .expect("EPContext node");
    let attr = ep
        .attribute
        .iter()
        .find(|a| a.name == "ep_cache_context")
        .expect("ep_cache_context attribute");

    assert_eq!(
        attr.r#type,
        onnx::attribute_proto::AttributeType::String as i32,
        "ep_cache_context must be a STRING attribute on the wire"
    );
    assert_eq!(
        attr.s,
        opaque_blob(),
        "STRING attribute bytes must be byte-exact"
    );
    assert!(
        attr.t.is_none(),
        "must not be emitted as a tensor attribute"
    );
}

#[test]
fn model_metadata_round_trips_through_proto() {
    let bytes1 = build_model_bytes();
    let (graph1, store1) = load_model_bytes_with_weights(&bytes1, ".").expect("decode 1");
    let meta = ModelMetadata {
        ir_version: 9,
        producer_name: "roy".into(),
        producer_version: "9.9".into(),
        domain: "com.example".into(),
        model_version: 42,
        doc_string: Some("hello".into()),
        graph_name: "g".into(),
        metadata_props: vec![("k1".into(), "v1".into()), ("k2".into(), "v2".into())],
    };
    let model = Model::new(&graph1)
        .with_weights(&store1)
        .with_metadata(meta);
    let bytes2 = encode_model(&model).expect("encode");

    let proto = onnx::ModelProto::decode(&bytes2[..]).expect("decode proto");
    assert_eq!(proto.ir_version, 9);
    assert_eq!(proto.producer_name, "roy");
    assert_eq!(proto.producer_version, "9.9");
    assert_eq!(proto.domain, "com.example");
    assert_eq!(proto.model_version, 42);
    assert_eq!(proto.doc_string, "hello");
    assert_eq!(proto.graph.as_ref().unwrap().name, "g");
    let props: HashMap<_, _> = proto
        .metadata_props
        .iter()
        .map(|e| (e.key.clone(), e.value.clone()))
        .collect();
    assert_eq!(props.get("k1"), Some(&"v1".to_string()));
    assert_eq!(props.get("k2"), Some(&"v2".to_string()));
}

#[test]
fn real_fixture_round_trips_if_present() {
    // Decode a real model, re-encode it, decode again, and assert structural +
    // initializer-byte equality. Skips gracefully when fixtures are absent.
    let manifest = env!("CARGO_MANIFEST_DIR");
    let root = std::path::Path::new(manifest)
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    let candidates = [
        "crates/onnx-runtime-session/tests/fixtures/bert_toy/model.onnx",
        "tests/fixtures/tiny-whisper/encoder.onnx",
    ];

    let mut ran = false;
    for rel in candidates {
        let path = root.join(rel);
        if !path.exists() {
            continue;
        }
        let bytes1 = std::fs::read(&path).expect("read fixture");
        let base = path.parent().unwrap();
        let (graph1, store1) =
            load_model_bytes_with_weights(&bytes1, base).expect("decode fixture");

        let model = Model::new(&graph1).with_weights(&store1);
        let bytes2 = encode_model(&model).expect("encode fixture");
        let (graph2, store2) =
            load_model_bytes_with_weights(&bytes2, base).expect("re-decode fixture");

        assert_eq!(
            graph1.num_nodes(),
            graph2.num_nodes(),
            "node count mismatch for {rel}"
        );
        assert_eq!(
            op_types(&graph1),
            op_types(&graph2),
            "op_types mismatch for {rel}"
        );
        assert_eq!(
            initializer_bytes(&graph1, &store1),
            initializer_bytes(&graph2, &store2),
            "initializer bytes mismatch for {rel}"
        );
        ran = true;
    }
    if !ran {
        eprintln!("real_fixture_round_trips_if_present: no fixtures found, skipping");
    }
}

#[test]
fn write_model_produces_reloadable_file() {
    let bytes1 = build_model_bytes();
    let (graph1, store1) = load_model_bytes_with_weights(&bytes1, ".").expect("decode 1");
    let model = Model::new(&graph1).with_weights(&store1);

    // CARGO_TARGET_TMPDIR is provided by cargo for integration-test scratch
    // files (never /tmp).
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
    let path = dir.join("roundtrip_model.onnx");
    write_model(&model, &path).expect("write");

    let (graph2, _store2) =
        load_model_bytes_with_weights(&std::fs::read(&path).expect("read back"), ".")
            .expect("reload");
    assert_eq!(graph1.num_nodes(), graph2.num_nodes());
    assert_eq!(op_types(&graph1), op_types(&graph2));

    let _ = std::fs::remove_file(&path);
}

// ── boundary / generic-path tests ─────────────────────────────────────────────

/// A `UINT8` opaque tensor **initializer** must round-trip byte-exact through the
/// generic tensor path (no op knowledge). Complements the STRING-attribute path.
#[test]
fn uint8_opaque_initializer_round_trips_byte_exact() {
    use onnx_runtime_ir::{DataType, Node, NodeId, TensorData, WeightRef};

    let payload = opaque_blob();
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let blob = graph.create_named_value("blob", DataType::Uint8, vec![payload.len().into()]);
    graph.set_initializer(
        blob,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Uint8,
            vec![payload.len()],
            payload.clone(),
        )),
    );
    let out = graph.create_named_value("out", DataType::Uint8, vec![payload.len().into()]);
    graph.insert_node(Node::new(
        NodeId(0),
        "Identity",
        vec![Some(blob)],
        vec![out],
    ));
    graph.add_output(out);

    let bytes = encode_model(&Model::new(&graph)).expect("encode");
    let (graph2, store2) = load_model_bytes_with_weights(&bytes, ".").expect("decode");

    let init = initializer_bytes(&graph2, &store2);
    assert_eq!(
        init.get("blob"),
        Some(&payload),
        "UINT8 opaque initializer must be byte-exact"
    );
}

/// An empty graph (no nodes, no initializers, no I/O) encodes and reloads.
#[test]
fn empty_graph_round_trips() {
    let graph = Graph::new();
    let bytes = encode_model(&Model::new(&graph)).expect("encode empty");
    let (graph2, _store2) = load_model_bytes_with_weights(&bytes, ".").expect("decode empty");
    assert_eq!(graph2.num_nodes(), 0);
    assert!(graph2.initializers.is_empty());
}

/// A graph whose only computation forwards an initializer round-trips byte-exact
/// (the initializer is the sole data source).
#[test]
fn initializer_only_graph_round_trips() {
    use onnx_runtime_ir::{DataType, Node, NodeId, TensorData, WeightRef};

    let raw: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let w = graph.create_named_value("W", DataType::Float32, vec![2usize.into(), 2usize.into()]);
    graph.set_initializer(
        w,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Float32,
            vec![2, 2],
            raw.clone(),
        )),
    );
    let out =
        graph.create_named_value("out", DataType::Float32, vec![2usize.into(), 2usize.into()]);
    graph.insert_node(Node::new(NodeId(0), "Identity", vec![Some(w)], vec![out]));
    graph.add_output(out);

    let bytes = encode_model(&Model::new(&graph)).expect("encode");
    let (graph2, store2) = load_model_bytes_with_weights(&bytes, ".").expect("decode");
    assert_eq!(graph2.num_nodes(), 1);
    assert_eq!(initializer_bytes(&graph2, &store2).get("W"), Some(&raw));
}

#[test]
fn control_flow_subgraphs_round_trip() {
    let branch = |name: &str, op: &str| onnx::GraphProto {
        name: name.into(),
        output: vec![value_info("branch_out", 1, &[Dimlike::Static(2)])],
        node: vec![
            node("Relu", &["x"], &["activated"]),
            node(op, &["activated"], &["branch_out"]),
        ],
        ..Default::default()
    };
    let graph = onnx::GraphProto {
        name: "control_flow".into(),
        input: vec![
            value_info("cond", 9, &[]),
            value_info("x", 1, &[Dimlike::Static(2)]),
        ],
        output: vec![value_info("y", 1, &[Dimlike::Static(2)])],
        node: vec![onnx::NodeProto {
            name: "choose".into(),
            op_type: "If".into(),
            input: vec!["cond".into()],
            output: vec!["y".into()],
            attribute: vec![
                onnx::AttributeProto {
                    name: "then_branch".into(),
                    r#type: onnx::attribute_proto::AttributeType::Graph as i32,
                    g: Some(branch("then", "Identity")),
                    ..Default::default()
                },
                onnx::AttributeProto {
                    name: "else_branch".into(),
                    r#type: onnx::attribute_proto::AttributeType::Graph as i32,
                    g: Some(branch("else", "Neg")),
                    ..Default::default()
                },
            ],
            ..Default::default()
        }],
        ..Default::default()
    };
    let proto = onnx::ModelProto {
        ir_version: 10,
        opset_import: vec![onnx::OperatorSetIdProto {
            domain: String::new(),
            version: 21,
        }],
        graph: Some(graph),
        ..Default::default()
    };

    let (graph1, store1) =
        load_model_bytes_with_weights(&proto.encode_to_vec(), ".").expect("decode control flow");
    let bytes =
        encode_model(&Model::new(&graph1).with_weights(&store1)).expect("encode control flow");
    let (graph2, _) = load_model_bytes_with_weights(&bytes, ".").expect("re-decode control flow");

    let if_node = graph2.nodes.iter().next().expect("If node").1;
    for attr_name in ["then_branch", "else_branch"] {
        let subgraph = graph2
            .subgraphs
            .get(&(if_node.id, attr_name.to_string()))
            .expect("indexed branch");
        assert!(subgraph.inputs.is_empty(), "{attr_name} input signature");
        assert_eq!(subgraph.outputs.len(), 1, "{attr_name} output signature");
        assert_eq!(subgraph.num_nodes(), 2, "{attr_name} body");
    }
}

#[test]
fn graphs_attribute_encodes_every_indexed_body() {
    use onnx_runtime_ir::{DataType, Node, NodeId};

    let body = |op: &str| {
        let mut graph = Graph::new();
        let input = graph.create_named_value("in", DataType::Float32, vec![1usize.into()]);
        let output = graph.create_named_value("out", DataType::Float32, vec![1usize.into()]);
        graph.add_input(input);
        graph.insert_node(Node::new(NodeId(0), op, vec![Some(input)], vec![output]));
        graph.add_output(output);
        graph
    };

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 21);
    let x = graph.create_named_value("x", DataType::Float32, vec![1usize.into()]);
    let y = graph.create_named_value("y", DataType::Float32, vec![1usize.into()]);
    graph.add_input(x);
    let mut node = Node::new(NodeId(0), "Custom", vec![Some(x)], vec![y]);
    node.attributes.insert(
        "bodies".into(),
        Attribute::Graphs(vec![Graph::new(), Graph::new()]),
    );
    let node_id = graph.insert_node(node);
    graph
        .subgraphs
        .insert((node_id, "bodies[0]".into()), body("Relu"));
    graph
        .subgraphs
        .insert((node_id, "bodies[1]".into()), body("Neg"));
    graph.add_output(y);

    let proto = encode_model_proto(&Model::new(&graph)).expect("encode Graphs attribute");
    let attribute = &proto.graph.unwrap().node[0].attribute[0];
    assert_eq!(
        attribute.r#type,
        onnx::attribute_proto::AttributeType::Graphs as i32
    );
    assert_eq!(attribute.graphs.len(), 2);
    assert_eq!(attribute.graphs[0].node[0].op_type, "Relu");
    assert_eq!(attribute.graphs[1].node[0].op_type, "Neg");
}

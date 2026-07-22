//! Loader tests for `com.microsoft::EPContext` nodes — the load path (§55.3).
//!
//! Builds synthetic `ModelProto`s (via the loader's own prost types, matching
//! the style of `tests/loader.rs`) and exercises:
//! * recognition + typed-view attribute parsing (defaults and explicit values),
//! * embedded-blob resolution (opaque binary bytes round-trip losslessly),
//! * external-file resolution (relative path → mmap, bytes match),
//! * path-safety (traversal / absolute paths rejected),
//! * opaque shape handling (declared value_info shapes survive load).

use std::path::{Path, PathBuf};

use prost::Message;

use onnx_runtime_ir::{DataType, Dim};
use onnx_runtime_loader::proto::onnx;
use onnx_runtime_loader::{EmbedMode, EpContextBlob, ep_context_nodes, resolve_ep_context};

// ── proto construction helpers ────────────────────────────────────────────────

fn tensor_type(elem_type: i32, dims: &[i64]) -> onnx::TypeProto {
    use onnx::tensor_shape_proto::{Dimension, dimension::Value as DV};
    let dim = dims
        .iter()
        .map(|d| Dimension {
            value: Some(DV::DimValue(*d)),
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

fn value_info(name: &str, elem_type: i32, dims: &[i64]) -> onnx::ValueInfoProto {
    onnx::ValueInfoProto {
        name: name.to_string(),
        r#type: Some(tensor_type(elem_type, dims)),
        ..Default::default()
    }
}

fn int_attr(name: &str, v: i64) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.to_string(),
        r#type: onnx::attribute_proto::AttributeType::Int as i32,
        i: v,
        ..Default::default()
    }
}

fn str_attr(name: &str, v: &str) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.to_string(),
        r#type: onnx::attribute_proto::AttributeType::String as i32,
        s: v.as_bytes().to_vec(),
        ..Default::default()
    }
}

/// A STRING attribute carrying *raw* bytes (used for `ep_cache_context`, whose
/// embedded payload is arbitrary binary — not necessarily valid UTF-8).
fn bytes_attr(name: &str, bytes: &[u8]) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.to_string(),
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
        op_type: "EPContext".to_string(),
        domain: "com.microsoft".to_string(),
        input: inputs.iter().map(|s| s.to_string()).collect(),
        output: outputs.iter().map(|s| s.to_string()).collect(),
        attribute: attrs,
        ..Default::default()
    }
}

/// Serialize a `GraphProto` into a `ModelProto` importing both the default and
/// `com.microsoft` opsets.
fn model_ms(graph: onnx::GraphProto) -> Vec<u8> {
    let m = onnx::ModelProto {
        ir_version: 8,
        opset_import: vec![
            onnx::OperatorSetIdProto {
                domain: String::new(),
                version: 17,
            },
            onnx::OperatorSetIdProto {
                domain: "com.microsoft".to_string(),
                version: 1,
            },
        ],
        graph: Some(graph),
        ..Default::default()
    };
    m.encode_to_vec()
}

/// A minimal single-EPContext graph: `X → EPContext → Y`, with the given
/// EPContext attributes and an optional declared output shape.
fn single_epctx_graph(attrs: Vec<onnx::AttributeProto>, out_dims: &[i64]) -> onnx::GraphProto {
    onnx::GraphProto {
        name: "epctx".into(),
        input: vec![value_info("X", 1, &[2, 4])],
        output: vec![value_info("Y", 1, out_dims)],
        node: vec![epctx_node(&["X"], &["Y"], attrs)],
        ..Default::default()
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[test]
fn embedded_blob_resolves_to_matching_bytes() {
    // Opaque payload including bytes that are NOT valid UTF-8 (0x00, 0x80, 0xFF)
    // to prove the loader preserves the blob losslessly.
    let payload: Vec<u8> = vec![0x00, 0x01, 0x80, 0xFE, 0xFF, b'v', b'1', 0x00, 0xC3, 0x28];

    let g = single_epctx_graph(
        vec![
            int_attr("embed_mode", 1),
            bytes_attr("ep_cache_context", &payload),
            str_attr("source", "SomeVendorEP"),
        ],
        &[2, 8],
    );
    let bytes = model_ms(g);

    let (graph, _store) =
        onnx_runtime_loader::load_model_bytes_with_weights(&bytes, ".").expect("load");

    let nodes: Vec<_> = ep_context_nodes(&graph).collect();
    assert_eq!(nodes.len(), 1, "exactly one EPContext node recognized");
    let n = &nodes[0];
    assert_eq!(n.embed_mode, EmbedMode::Embedded);
    assert_eq!(n.source, Some("SomeVendorEP"));

    let blob = resolve_ep_context(Path::new("."), n).expect("resolve embedded");
    match blob {
        EpContextBlob::Embedded(b) => assert_eq!(b, payload, "embedded bytes must match exactly"),
        EpContextBlob::External { .. } => panic!("expected Embedded, got External"),
    }
}

#[test]
fn external_blob_mmaps_and_matches() {
    // Per-test temp dir under target/ (never /tmp), provided by Cargo.
    let base: PathBuf = Path::new(env!("CARGO_TARGET_TMPDIR")).join("epctx_external");
    std::fs::create_dir_all(&base).expect("mkdir model dir");

    let payload: Vec<u8> = (0..=255u8).cycle().take(1024).collect();
    let bin_path = base.join("ctx_blob.bin");
    std::fs::write(&bin_path, &payload).expect("write external blob");

    let g = single_epctx_graph(
        vec![
            int_attr("embed_mode", 0),
            // Relative path, resolved against the model directory (§55.3).
            str_attr("ep_cache_context", "ctx_blob.bin"),
            str_attr("source", "SomeVendorEP"),
        ],
        &[2, 8],
    );
    let bytes = model_ms(g);

    let (graph, _store) =
        onnx_runtime_loader::load_model_bytes_with_weights(&bytes, &base).expect("load");

    let nodes: Vec<_> = ep_context_nodes(&graph).collect();
    assert_eq!(nodes.len(), 1);
    let n = &nodes[0];
    assert_eq!(n.embed_mode, EmbedMode::ExternalFile);

    let blob = resolve_ep_context(&base, n).expect("resolve external");
    match blob {
        EpContextBlob::External { path, map } => {
            assert_eq!(path, bin_path, "resolved path joins model dir + relative");
            assert_eq!(&map[..], payload.as_slice(), "mmap'd bytes must match file");
        }
        EpContextBlob::Embedded(_) => panic!("expected External, got Embedded"),
    }

    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn attribute_parsing_uses_defaults_when_absent() {
    // Only `ep_cache_context` present — everything else must default (§55.2).
    let g = single_epctx_graph(vec![bytes_attr("ep_cache_context", b"blob")], &[2, 8]);
    let bytes = model_ms(g);
    let (graph, _store) =
        onnx_runtime_loader::load_model_bytes_with_weights(&bytes, ".").expect("load");

    let n = ep_context_nodes(&graph).next().expect("one EPContext node");
    assert!(n.main_context, "main_context defaults to true");
    assert_eq!(
        n.embed_mode,
        EmbedMode::Embedded,
        "embed_mode defaults to Embedded"
    );
    assert_eq!(n.source, None);
    assert_eq!(n.sdk_version, None);
    assert_eq!(n.partition_name, None);
}

#[test]
fn attribute_parsing_reads_explicit_values() {
    let g = single_epctx_graph(
        vec![
            int_attr("main_context", 0),
            int_attr("embed_mode", 0),
            str_attr("ep_cache_context", "blob.bin"),
            str_attr("source", "MyVendorEP"),
            str_attr("ep_sdk_version", "1.2.3"),
            str_attr("partition_name", "part_0"),
        ],
        &[2, 8],
    );
    let bytes = model_ms(g);
    let (graph, _store) =
        onnx_runtime_loader::load_model_bytes_with_weights(&bytes, ".").expect("load");

    let n = ep_context_nodes(&graph).next().expect("one EPContext node");
    assert!(!n.main_context, "main_context=0 → false");
    assert_eq!(n.embed_mode, EmbedMode::ExternalFile);
    assert_eq!(n.source, Some("MyVendorEP"));
    assert_eq!(n.sdk_version, Some("1.2.3"));
    assert_eq!(n.partition_name, Some("part_0"));
    // Variadic i/o come straight from the node.
    assert_eq!(n.inputs().len(), 1);
    assert_eq!(n.outputs().len(), 1);
}

#[test]
fn external_parent_traversal_is_rejected() {
    let g = single_epctx_graph(
        vec![
            int_attr("embed_mode", 0),
            str_attr("ep_cache_context", "../evil.bin"),
        ],
        &[2, 8],
    );
    let bytes = model_ms(g);
    let (graph, _store) =
        onnx_runtime_loader::load_model_bytes_with_weights(&bytes, ".").expect("load");

    let n = ep_context_nodes(&graph).next().expect("one EPContext node");
    let err = resolve_ep_context(Path::new("."), &n).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("traversal") && msg.contains("evil"),
        "traversal must be rejected with a clear error, got: {msg}"
    );
}

#[test]
fn external_absolute_path_is_rejected() {
    let g = single_epctx_graph(
        vec![
            int_attr("embed_mode", 0),
            str_attr("ep_cache_context", "/etc/passwd"),
        ],
        &[2, 8],
    );
    let bytes = model_ms(g);
    let (graph, _store) =
        onnx_runtime_loader::load_model_bytes_with_weights(&bytes, ".").expect("load");

    let n = ep_context_nodes(&graph).next().expect("one EPContext node");
    let err = resolve_ep_context(Path::new("."), &n).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("absolute"),
        "absolute path must be rejected with a clear error, got: {msg}"
    );
}

#[test]
fn epcontext_output_shape_is_opaque_and_preserved() {
    // EPContext is not a registered shape-inference op. Loading (which runs
    // shape inference) must NOT error and must leave the declared value_info
    // output shape [2, 8] intact — no op-specific inference (§55.3).
    let g = single_epctx_graph(
        vec![
            int_attr("embed_mode", 1),
            bytes_attr("ep_cache_context", b"opaque"),
        ],
        &[2, 8],
    );
    let bytes = model_ms(g);
    let graph = onnx_runtime_loader::load_model_bytes(&bytes).expect("load with shape inference");

    // Find Y and confirm its declared shape survived.
    let y = graph
        .values
        .iter()
        .find(|(_, v)| v.name.as_deref() == Some("Y"))
        .map(|(id, _)| id)
        .expect("output Y present");
    assert_eq!(
        graph.value(y).shape,
        vec![Dim::Static(2), Dim::Static(8)],
        "EPContext output must keep its declared value_info shape"
    );
    assert_eq!(graph.value(y).dtype, DataType::Float32);
}

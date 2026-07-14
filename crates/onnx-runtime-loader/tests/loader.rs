//! End-to-end loader tests: hand-built `ModelProto` → IR `Graph`.
//!
//! Exercises graph construction (edges, SSA, source values), symbolic-dim
//! interning by name, opset imports, and shape inference over a
//! MatMul → Add → LayerNormalization chain.

use prost::Message;

use onnx_runtime_ir::{Dim, Graph, WeightRef};
use onnx_runtime_loader::proto::onnx;

// --- proto construction helpers ---

enum Dimlike {
    Static(i64),
    Param(&'static str),
}

fn tensor_type(elem_type: i32, dims: &[Dimlike]) -> onnx::TypeProto {
    use onnx::tensor_shape_proto::{dimension::Value as DV, Dimension};
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
        value: Some(onnx::type_proto::Value::TensorType(onnx::type_proto::Tensor {
            elem_type,
            shape: Some(onnx::TensorShapeProto { dim }),
        })),
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

fn f32_initializer(name: &str, dims: &[i64]) -> onnx::TensorProto {
    let numel: i64 = dims.iter().product();
    onnx::TensorProto {
        name: name.to_string(),
        data_type: 1, // FLOAT
        dims: dims.to_vec(),
        raw_data: vec![0u8; numel as usize * 4],
        ..Default::default()
    }
}

fn i64_initializer(name: &str, values: &[i64]) -> onnx::TensorProto {
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    onnx::TensorProto {
        name: name.to_string(),
        data_type: 7, // INT64
        dims: vec![values.len() as i64],
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

fn model(graph: onnx::GraphProto, opset: i64) -> Vec<u8> {
    let m = onnx::ModelProto {
        ir_version: 8,
        opset_import: vec![onnx::OperatorSetIdProto {
            domain: String::new(),
            version: opset,
        }],
        graph: Some(graph),
        ..Default::default()
    };
    m.encode_to_vec()
}

fn find(graph: &Graph, name: &str) -> onnx_runtime_ir::ValueId {
    graph
        .values
        .iter()
        .find(|(_, v)| v.name.as_deref() == Some(name))
        .map(|(id, _)| id)
        .unwrap_or_else(|| panic!("value {name} not found"))
}

// --- tests ---

#[test]
fn matmul_add_layernorm_chain() {
    // X[batch, 4] -> MatMul(W[4,8]) -> H -> Add(B[8]) -> A -> LayerNorm -> Y
    let g = onnx::GraphProto {
        name: "bert_like".into(),
        input: vec![value_info(
            "X",
            1,
            &[Dimlike::Param("batch"), Dimlike::Static(4)],
        )],
        output: vec![value_info(
            "Y",
            1,
            &[Dimlike::Param("batch"), Dimlike::Static(8)],
        )],
        initializer: vec![
            f32_initializer("W", &[4, 8]),
            f32_initializer("B", &[8]),
            f32_initializer("Scale", &[8]),
            f32_initializer("Bias", &[8]),
        ],
        node: vec![
            node("MatMul", &["X", "W"], &["H"]),
            node("Add", &["H", "B"], &["A"]),
            node("LayerNormalization", &["A", "Scale", "Bias"], &["Y"]),
        ],
        ..Default::default()
    };

    let bytes = model(g, 17);
    let graph = onnx_runtime_loader::load_model_bytes(&bytes).expect("load");

    // Opset imports populated.
    assert_eq!(graph.opset_imports.get(""), Some(&17));

    // Structure: 3 nodes; X is the only graph input; Y the only output.
    assert_eq!(graph.num_nodes(), 3);
    assert_eq!(graph.inputs.len(), 1);
    assert_eq!(graph.outputs.len(), 1);

    // Initializers are source values (no producer) and recorded as weights.
    let w = find(&graph, "W");
    assert!(graph.value(w).producer.is_none());
    assert!(graph.initializers.contains_key(&w));
    match &graph.initializers[&w] {
        WeightRef::Inline(t) => {
            assert_eq!(t.dims, vec![4, 8]);
            assert_eq!(t.data.len(), 4 * 8 * 4);
        }
        _ => panic!("expected inline weight"),
    }

    // Edge consistency: H is produced by the MatMul node and consumed by Add.
    let h = find(&graph, "H");
    let matmul_nid = graph.value(h).producer.expect("H has producer");
    assert_eq!(graph.node(matmul_nid).op_type, "MatMul");
    assert_eq!(graph.value(h).consumers.len(), 1);

    // X is a graph input with no producer.
    let x = find(&graph, "X");
    assert!(graph.value(x).producer.is_none());
    assert!(graph.inputs.contains(&x));

    // Shape inference: batch is symbolic and shared; feature dim is 8.
    let y = find(&graph, "Y");
    let yshape = &graph.value(y).shape;
    assert_eq!(yshape.len(), 2);
    assert_eq!(yshape[1], Dim::Static(8));
    let batch_sym = match graph.value(x).shape[0] {
        Dim::Symbolic(id) => id,
        _ => panic!("X batch dim should be symbolic"),
    };
    assert_eq!(yshape[0], Dim::Symbolic(batch_sym));

    // H shape propagated to [batch, 8].
    assert_eq!(
        graph.value(h).shape,
        vec![Dim::Symbolic(batch_sym), Dim::Static(8)]
    );

    // The built graph upholds all structural invariants.
    graph.validate().expect("graph valid");
}

#[test]
fn symbolic_dims_interned_by_name() {
    // Two inputs share dim_param "seq"; they must resolve to the same SymbolId.
    let g = onnx::GraphProto {
        input: vec![
            value_info("A", 1, &[Dimlike::Param("seq"), Dimlike::Static(2)]),
            value_info("B", 1, &[Dimlike::Param("seq"), Dimlike::Static(2)]),
        ],
        output: vec![value_info(
            "C",
            1,
            &[Dimlike::Param("seq"), Dimlike::Static(2)],
        )],
        node: vec![node("Add", &["A", "B"], &["C"])],
        ..Default::default()
    };
    let bytes = model(g, 17);
    let graph = onnx_runtime_loader::load_model_bytes(&bytes).expect("load");

    let a = find(&graph, "A");
    let b = find(&graph, "B");
    let sa = match graph.value(a).shape[0] {
        Dim::Symbolic(id) => id,
        _ => panic!("A[0] symbolic"),
    };
    let sb = match graph.value(b).shape[0] {
        Dim::Symbolic(id) => id,
        _ => panic!("B[0] symbolic"),
    };
    assert_eq!(sa, sb, "same dim_param must intern to same SymbolId");
}

#[test]
fn reshape_uses_constant_shape_initializer() {
    // Reshape(X[2,3,4], shape=[-1, 4]) -> [6, 4], shape from a constant init.
    let g = onnx::GraphProto {
        input: vec![value_info(
            "X",
            1,
            &[Dimlike::Static(2), Dimlike::Static(3), Dimlike::Static(4)],
        )],
        output: vec![value_info("Y", 1, &[Dimlike::Static(6), Dimlike::Static(4)])],
        initializer: vec![i64_initializer("shape", &[-1, 4])],
        node: vec![node("Reshape", &["X", "shape"], &["Y"])],
        ..Default::default()
    };
    let bytes = model(g, 17);
    let graph = onnx_runtime_loader::load_model_bytes(&bytes).expect("load");

    let y = find(&graph, "Y");
    assert_eq!(
        graph.value(y).shape,
        vec![Dim::Static(6), Dim::Static(4)],
        "reshape -1 should resolve to 6"
    );
}

#[test]
fn smoke_load_real_fixture_if_present() {
    // Repo fixtures live at <workspace>/tests/fixtures/*/model.onnx. They are
    // model-specific but the loader is generic, so a successful load + validate
    // is a good real-world smoke check. Skips gracefully if fixtures are absent.
    let manifest = env!("CARGO_MANIFEST_DIR");
    let root = std::path::Path::new(manifest)
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");
    let candidates = [
        "tests/fixtures/tiny-eagle3/model.onnx",     // external data
        "tests/fixtures/tiny-whisper/encoder.onnx",  // inline
    ];
    let mut loaded_any = false;
    for rel in candidates {
        let path = root.join(rel);
        if !path.exists() {
            continue;
        }
        let graph = onnx_runtime_loader::load_model(&path)
            .unwrap_or_else(|e| panic!("failed to load {}: {e}", path.display()));
        graph
            .validate()
            .unwrap_or_else(|e| panic!("invalid graph from {}: {e:?}", path.display()));
        assert!(graph.num_nodes() > 0);
        loaded_any = true;
    }
    if !loaded_any {
        eprintln!("smoke_load_real_fixture_if_present: no fixtures found, skipping");
    }
}

// ── load_*_with_weights: bytes survive after load, work for inline + external ──

/// Build a tiny model with inline weights and verify that the Arc<WeightStore>
/// keeps the bytes accessible after `load_model_bytes_with_weights` returns.
#[test]
fn load_bytes_with_weights_inline_survives() {
    let g = onnx::GraphProto {
        name: "inline_weights".into(),
        input: vec![value_info("X", 1, &[Dimlike::Static(2), Dimlike::Static(4)])],
        output: vec![value_info("Y", 1, &[Dimlike::Static(2), Dimlike::Static(8)])],
        initializer: vec![f32_initializer("W", &[4, 8])],
        node: vec![node("MatMul", &["X", "W"], &["Y"])],
        ..Default::default()
    };
    let bytes = model(g, 17);

    let (graph, store) =
        onnx_runtime_loader::load_model_bytes_with_weights(&bytes, ".")
            .expect("load_model_bytes_with_weights");

    // The store must be usable to get bytes for every initializer in the graph.
    let mut found_inline = false;
    for weight_ref in graph.initializers.values() {
        match weight_ref {
            WeightRef::Inline(_) => {
                let raw = store.bytes(weight_ref).expect("inline bytes present");
                // W is [4,8] f32 → 128 bytes of zeros
                assert_eq!(raw.len(), 4 * 8 * 4, "W byte count");
                assert!(raw.iter().all(|&b| b == 0), "W should be all-zeros");
                found_inline = true;
            }
            WeightRef::External { .. } => {}
        }
    }
    assert!(found_inline, "expected at least one inline initializer");

    // Drop the graph; the Arc alone must keep bytes valid.
    drop(graph);
    // Re-query via a clone of the Arc — bytes still live.
    let store2 = std::sync::Arc::clone(&store);
    // We can't re-query without the WeightRef, but we can verify the Arc
    // has the right ref-count and the store isn't dropped.
    assert_eq!(std::sync::Arc::strong_count(&store2), 2);
}

/// Load a real fixture that has an external-data file and verify that the
/// `Arc<WeightStore>` exposes non-empty byte slices for External WeightRefs.
/// Skips gracefully if the fixture directory is absent.
#[test]
fn load_with_weights_external_data_fixture() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let root = std::path::Path::new(manifest)
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root");

    // Pick any fixture that ships with external data.
    let candidates = [
        "tests/fixtures/tiny-eagle3/model.onnx",
        "tests/fixtures/tiny-llm/model.onnx",
        "tests/fixtures/tiny-llm-scatter/model.onnx",
    ];

    let mut tested_external = false;
    for rel in candidates {
        let path = root.join(rel);
        if !path.exists() {
            continue;
        }

        let (graph, store) = onnx_runtime_loader::load_model_with_weights(&path)
            .unwrap_or_else(|e| panic!("load_model_with_weights({rel}): {e}"));

        graph
            .validate()
            .unwrap_or_else(|e| panic!("invalid graph from {rel}: {e:?}"));

        // For every initializer, store.bytes() must return Some with len > 0.
        for (vid, weight_ref) in &graph.initializers {
            let raw = store
                .bytes(weight_ref)
                .unwrap_or_else(|| panic!("store.bytes() returned None for value {vid:?}"));
            assert!(!raw.is_empty(), "weight bytes must be non-empty");
        }

        // Verify external refs specifically yield bytes even after we drop the
        // graph (mmap still alive via Arc).
        let externals: Vec<_> = graph
            .initializers
            .values()
            .filter(|w| matches!(w, WeightRef::External { .. }))
            .cloned()
            .collect();

        drop(graph); // graph gone — Arc<WeightStore> must keep mmaps alive

        for w in &externals {
            let raw = store.bytes(w).expect("external bytes live after graph drop");
            assert!(!raw.is_empty());
            tested_external = true;
        }

        break; // one fixture is enough
    }

    if !tested_external {
        eprintln!("load_with_weights_external_data_fixture: no external-data fixture found, skipping");
    }
}

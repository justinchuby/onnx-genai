//! External initializer path-safety and mmap tests (§19.2).

use std::path::Path;

use onnx_runtime_ir::WeightRef;
use onnx_runtime_loader::proto::onnx;
use onnx_runtime_loader::{LoaderError, load_model_bytes_with_weights};
use prost::Message;

fn external_model(location: &str, length: usize) -> Vec<u8> {
    let initializer = onnx::TensorProto {
        name: "weight".to_string(),
        data_type: 2, // UINT8
        dims: vec![length as i64],
        external_data: vec![
            onnx::StringStringEntryProto {
                key: "location".to_string(),
                value: location.to_string(),
            },
            onnx::StringStringEntryProto {
                key: "length".to_string(),
                value: length.to_string(),
            },
        ],
        data_location: onnx::tensor_proto::DataLocation::External as i32,
        ..Default::default()
    };
    onnx::ModelProto {
        ir_version: 8,
        opset_import: vec![onnx::OperatorSetIdProto {
            domain: String::new(),
            version: 21,
        }],
        graph: Some(onnx::GraphProto {
            name: "external_weight".to_string(),
            initializer: vec![initializer],
            ..Default::default()
        }),
        ..Default::default()
    }
    .encode_to_vec()
}

fn assert_path_rejected(location: &str) {
    let error = load_model_bytes_with_weights(
        &external_model(location, 1),
        Path::new(env!("CARGO_TARGET_TMPDIR")),
    )
    .expect_err("unsafe external-data path must be rejected");

    assert!(matches!(
        error,
        LoaderError::ExternalDataPath { path, .. } if path == location
    ));
}

#[test]
fn external_data_rejects_parent_traversal() {
    assert_path_rejected("../escape.bin");
}

#[cfg(unix)]
#[test]
fn external_data_rejects_absolute_path() {
    assert_path_rejected("/etc/passwd");
}

#[test]
fn external_data_rejects_embedded_parent_traversal() {
    assert_path_rejected("subdir/nested/../../../escape.bin");
}

#[test]
fn external_data_mmaps_safe_relative_paths() {
    let model_dir =
        Path::new(env!("CARGO_TARGET_TMPDIR")).join("external_data_mmaps_safe_relative_paths");
    let _ = std::fs::remove_dir_all(&model_dir);
    std::fs::create_dir_all(model_dir.join("subdir")).expect("create external-data test directory");

    for (location, payload) in [
        ("weights.bin", b"top-level".as_slice()),
        ("subdir/weights.bin", b"nested".as_slice()),
    ] {
        std::fs::write(model_dir.join(location), payload).expect("write external-data test file");
        let (graph, store) =
            load_model_bytes_with_weights(&external_model(location, payload.len()), &model_dir)
                .expect("safe relative external-data path must resolve");
        let weight = graph
            .initializers
            .values()
            .next()
            .expect("external initializer must be present");

        assert_eq!(store.bytes(weight), Some(payload));
        assert!(matches!(
            weight,
            WeightRef::External { path, .. } if path == &model_dir.join(location)
        ));
    }
}

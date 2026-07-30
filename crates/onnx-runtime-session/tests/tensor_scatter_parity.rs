//! End-to-end native-session parity with the committed ONNX Runtime CPU golden.
//!
//! Regenerate the fixture and ORT reference with:
//! `python3 tests/fixtures/tensor_scatter/generate.py`

use std::path::{Path, PathBuf};

use onnx_runtime_session::{InferenceSession, Tensor};

fn fixture_directory() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tensor_scatter")
}

fn read_f32(name: &str) -> Vec<f32> {
    std::fs::read(fixture_directory().join(name))
        .unwrap()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn read_i64(name: &str) -> Vec<i64> {
    std::fs::read(fixture_directory().join(name))
        .unwrap()
        .chunks_exact(8)
        .map(|bytes| i64::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

#[test]
fn tensor_scatter_native_session_matches_onnxruntime_exactly() {
    let mut session =
        InferenceSession::load(fixture_directory().join("model.onnx.textproto")).unwrap();
    let cache = Tensor::from_f32(&[2, 5, 2, 2], &read_f32("cache.f32.bin")).unwrap();
    let updates = Tensor::from_f32(&[2, 2, 2, 2], &read_f32("updates.f32.bin")).unwrap();
    let write_indices = Tensor::from_i64(&[2], &read_i64("write_indices.i64.bin")).unwrap();

    let outputs = session
        .run(&[
            ("cache", &cache),
            ("updates", &updates),
            ("write_indices", &write_indices),
        ])
        .unwrap();

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].shape, [2, 5, 2, 2]);
    assert_eq!(
        outputs[0].to_vec_f32(),
        read_f32("updated_cache.ort.f32.bin")
    );
}

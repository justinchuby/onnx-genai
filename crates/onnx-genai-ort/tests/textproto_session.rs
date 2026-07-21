//! ORT-side textproto loading: a git-friendly `*.onnx.textproto` fixture must
//! create a working session and run inference identically to its binary
//! counterpart.
//!
//! ORT cannot read protobuf TextFormat from disk, so [`Session::new`] detects
//! the `.textproto` suffix, converts the model to binary bytes (via onnx-std),
//! and creates the session from memory with `CreateSessionFromArray`. Because
//! that path has no model-directory context, textproto fixtures inline all
//! weights.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use onnx_genai_ort::{Environment, Session, SessionOptions, Value};

fn tiny_whisper_encoder_textproto() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-whisper/encoder.onnx.textproto")
}

fn test_environment() -> &'static Environment {
    static ENVIRONMENT: OnceLock<Environment> = OnceLock::new();
    ENVIRONMENT.get_or_init(|| Environment::new("textproto-session-test").expect("env"))
}

#[test]
fn loads_and_runs_textproto_fixture() {
    let path = tiny_whisper_encoder_textproto();
    if !path.exists() {
        eprintln!("loads_and_runs_textproto_fixture: fixture absent, skipping");
        return;
    }

    let session = Session::new(
        test_environment(),
        &path,
        SessionOptions::default().with_intra_op_threads(1),
    )
    .expect("session created from textproto fixture");

    assert_eq!(session.input_names(), &["input_features".to_string()]);
    assert_eq!(
        session.output_names(),
        &["encoder_hidden_states".to_string()]
    );

    let features = Value::from_slice_f32(&vec![0.0f32; 80 * 8], &[1, 80, 8]).expect("input");
    let outputs = session
        .run(&[("input_features", &features)])
        .expect("run textproto session");
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].shape(), &[1, 4, 4]);
}

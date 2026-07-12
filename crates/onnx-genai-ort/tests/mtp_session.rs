use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use onnx_genai_ort::{
    Environment, MtpDecodeOptions, MtpDecodeSession, MtpDraftKvMode, Session, SessionOptions,
};

const HIDDEN: usize = 16;

fn tiny_mtp() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen35-mtp/model.onnx")
}

fn deterministic_session_options() -> SessionOptions {
    // Single-threaded intra-op for exact-equality determinism.
    SessionOptions::default().with_intra_op_threads(1)
}

fn test_environment() -> &'static Environment {
    static ENVIRONMENT: OnceLock<Environment> = OnceLock::new();
    ENVIRONMENT.get_or_init(|| Environment::new("mtp-session-test").expect("env"))
}

fn load_head() -> Session {
    Session::new(
        test_environment(),
        &tiny_mtp(),
        deterministic_session_options(),
    )
    .expect("session")
}

/// Deterministic pseudo-random f32 in roughly [-1, 1) from a linear-congruential seed.
fn lcg_weights(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(len);
    for _ in 0..len {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (state >> 33) as u32;
        out.push((bits as f32 / u32::MAX as f32) * 2.0 - 1.0);
    }
    out
}

fn target_hidden() -> Vec<f32> {
    lcg_weights(0xA5A5_1234, HIDDEN)
}

#[test]
fn detects_mtp_head_signature() {
    let session = load_head();
    let signature = MtpDecodeSession::detect(&session)
        .expect("detect")
        .expect("fixture is an MTP head");
    assert_eq!(signature.hidden_size, HIDDEN);
    assert_eq!(signature.kv_heads, 1);
    assert_eq!(signature.head_dim, 8);
    assert_eq!(signature.layers, 1);
}

#[test]
fn single_step_produces_hidden_state() {
    let session = load_head();
    let mut mtp =
        MtpDecodeSession::new(&session, MtpDecodeOptions::default()).expect("mtp session");
    assert_eq!(mtp.mode(), MtpDraftKvMode::HiddenThreaded);

    let embeds = lcg_weights(0xDEAD_BEEF, HIDDEN);
    let hidden = target_hidden();
    let mtp_hidden = mtp.step(&embeds, &hidden, 0).expect("step");
    assert_eq!(mtp_hidden.len(), HIDDEN);
    assert!(mtp_hidden.iter().all(|value| value.is_finite()));
    // HiddenThreaded never grows the head cache.
    assert_eq!(mtp.past_len(), 0);
}

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use onnx_genai_ort::{
    Environment, LinearEmbedder, LinearLmHead, LmHead, MtpDecodeOptions, MtpDecodeSession,
    MtpDraftKvMode, Session, SessionOptions, TokenEmbedder, argmax,
};

const HIDDEN: usize = 16;
const VOCAB: usize = 32;

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

fn embedder() -> LinearEmbedder {
    LinearEmbedder::new(lcg_weights(0x1111_2222, VOCAB * HIDDEN), VOCAB, HIDDEN).expect("embedder")
}

fn lm_head() -> LinearLmHead {
    LinearLmHead::new(lcg_weights(0x3333_4444, HIDDEN * VOCAB), HIDDEN, VOCAB).expect("lm head")
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

#[test]
fn proposes_k_draft_tokens_from_real_fixture() {
    let session = load_head();
    let mut mtp =
        MtpDecodeSession::new(&session, MtpDecodeOptions::default()).expect("mtp session");

    let embedder = embedder();
    let lm_head = lm_head();
    let hidden = target_hidden();

    // The main model's own next-token (from logits over the target hidden) is
    // the guaranteed token that MTP speculates beyond.
    let mut guaranteed_logits = vec![0.0f32; VOCAB];
    lm_head
        .logits(&hidden, &mut guaranteed_logits)
        .expect("guaranteed logits");
    let guaranteed = argmax(&guaranteed_logits).expect("argmax") as u32;

    let k = 4;
    let proposal = mtp
        .propose(&hidden, guaranteed, k, &embedder, &lm_head)
        .expect("propose");

    assert_eq!(proposal.guaranteed_token, guaranteed);
    assert_eq!(proposal.draft_tokens.len(), k);
    assert_eq!(proposal.draft_hiddens.len(), k);
    for &token in &proposal.draft_tokens {
        assert!((token as usize) < VOCAB, "draft token {token} out of vocab");
    }
    for hidden_state in &proposal.draft_hiddens {
        assert_eq!(hidden_state.len(), HIDDEN);
        assert!(hidden_state.iter().all(|value| value.is_finite()));
    }

    // Determinism: a second proposal (fresh reset) yields identical tokens.
    let repeat = mtp
        .propose(&hidden, guaranteed, k, &embedder, &lm_head)
        .expect("propose repeat");
    assert_eq!(proposal.draft_tokens, repeat.draft_tokens);
    assert_eq!(proposal.draft_hiddens, repeat.draft_hiddens);

    // The guaranteed token is a stable function of the target hidden.
    let mut recomputed = vec![0.0f32; VOCAB];
    lm_head.logits(&hidden, &mut recomputed).expect("recompute");
    assert_eq!(argmax(&recomputed).expect("argmax") as u32, guaranteed);
}

#[test]
fn manual_chain_matches_propose() {
    // The propose loop is just: embed(prev) -> step(hidden) -> argmax(lm_head).
    let session = load_head();
    let mut mtp =
        MtpDecodeSession::new(&session, MtpDecodeOptions::default()).expect("mtp session");
    let embedder = embedder();
    let lm_head = lm_head();
    let hidden = target_hidden();

    let mut guaranteed_logits = vec![0.0f32; VOCAB];
    lm_head
        .logits(&hidden, &mut guaranteed_logits)
        .expect("logits");
    let guaranteed = argmax(&guaranteed_logits).expect("argmax") as u32;

    let k = 3;
    let proposal = mtp
        .propose(&hidden, guaranteed, k, &embedder, &lm_head)
        .expect("propose");

    // Reproduce the chain by hand using the low-level step API.
    mtp.reset();
    let mut running = hidden.clone();
    let mut prev = guaranteed;
    let mut expected = Vec::new();
    let mut embed_buf = vec![0.0f32; HIDDEN];
    let mut logits_buf = vec![0.0f32; VOCAB];
    for i in 0..k {
        embedder.embed(prev, &mut embed_buf).expect("embed");
        let mtp_hidden = mtp.step(&embed_buf, &running, i as i64).expect("step");
        lm_head
            .logits(&mtp_hidden, &mut logits_buf)
            .expect("logits");
        let token = argmax(&logits_buf).expect("argmax") as u32;
        expected.push(token);
        prev = token;
        running = mtp_hidden;
    }
    assert_eq!(proposal.draft_tokens, expected);
}

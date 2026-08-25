#![cfg(feature = "bench-native")]
//! Stage 2a (#750) guard tests for the batch-N fused forward
//! ([`NativeDecodeSession::run_fused_batch_prefill`]).
//!
//! These run entirely on CPU against the architecture-representative synthetic
//! cached decoder, so they are deterministic and need no GPU. They validate the
//! two properties the batch-N fused forward rests on, in isolation from the
//! CUDA weight-streaming residency (measured separately on hardware) and from
//! the batched KV *layout* (stage 2b):
//!
//! 1. **Row independence** — one fused `[N, 1]` forward with an empty past
//!    computes each row as an independent length-1 sequence.
//! 2. **Batch-1 byte-identical guard** (#844) — row `i` of a batch-N forward is
//!    *bit-for-bit* identical to a batch-1 forward of `token_ids[i]`.

use onnx_genai_bench::synthetic_decoder;
use onnx_genai_engine::NativeDecodeSession;
use onnx_genai_metadata::{DecoderAbi, KvOwnership, SequenceInputKind};
use onnx_runtime_session::InferenceSession;
use std::collections::BTreeMap;

/// Explicit I/O contract for the synthetic decoder. Its `input_ids`,
/// `attention_mask`, and `position_ids` ports are all `[-1, -1]` Int64, so the
/// shape/dtype auto-resolution is ambiguous and a declared spec is required.
fn synthetic_io() -> DecoderAbi {
    DecoderAbi {
        sequence_source: Some(SequenceInputKind::TokenIds),
        kv_ownership: Some(KvOwnership::Owned),
        kv_layout: None,
        token_input: Some("input_ids".into()),
        inputs_embeds_input: None,
        attention_mask_input: Some("attention_mask".into()),
        position_ids_input: Some("position_ids".into()),
        logits_output: Some("logits".into()),
        hidden_output: None,
        kv_inputs: Some(vec![
            "past_key_values.0.key".into(),
            "past_key_values.0.value".into(),
            "past_key_values.1.key".into(),
            "past_key_values.1.value".into(),
        ]),
        kv_outputs: Some(vec![
            "present_key_values.0.key".into(),
            "present_key_values.0.value".into(),
            "present_key_values.1.key".into(),
            "present_key_values.1.value".into(),
        ]),
        encoder_hidden_states_input: None,
        audio_features_input: None,
        cross_kv_inputs: None,
        cross_kv_outputs: None,
        kv_update: None,
        state_pairs: None,
        optional_inputs: BTreeMap::new(),
        static_cache: None,
        csa_state_groups: None,
    }
}

fn synthetic_session() -> NativeDecodeSession {
    let graph = synthetic_decoder::build_synthetic_decoder();
    let session = InferenceSession::from_graph(graph).expect("build synthetic session");
    NativeDecodeSession::from_session_with_io(session, &synthetic_io())
        .expect("wrap synthetic native decoder")
}

/// Bit pattern of a logits row, so the comparison is byte-identical rather than
/// float-approximate.
fn bits(row: &[f32]) -> Vec<u32> {
    row.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn native_fused_batch_prefill_row_identical_to_batch_one() {
    // Tokens must be < VOCAB_SIZE (32) for the synthetic embedding table.
    const TOKENS: [u32; 4] = [3, 17, 5, 29];

    let mut session = synthetic_session();

    // Batch-N: one fused forward over N independent single-token rows.
    let batched = session
        .run_fused_batch_prefill(&TOKENS)
        .expect("batch-N fused forward");
    assert_eq!(batched.len(), TOKENS.len());

    // Batch-1: each token through the same code path, one row at a time.
    for (index, &token) in TOKENS.iter().enumerate() {
        let single = session
            .run_fused_batch_prefill(&[token])
            .expect("batch-1 fused forward");
        assert_eq!(single.len(), 1);
        assert_eq!(
            bits(&batched[index]),
            bits(&single[0]),
            "row {index} (token {token}) of the batch-{} forward must be byte-identical to its batch-1 forward",
            TOKENS.len()
        );
    }
}

#[test]
fn native_fused_batch_prefill_is_stateless() {
    // The probe must not disturb an in-progress decode: current_len stays 0 and
    // a subsequent ordinary decode is unaffected.
    let mut session = synthetic_session();
    assert_eq!(session.current_len(), 0);

    let _ = session
        .run_fused_batch_prefill(&[1, 2, 3])
        .expect("fused batch forward");
    assert_eq!(
        session.current_len(),
        0,
        "probe must not advance decode state"
    );

    // A normal single-token decode still works after the probe.
    let logits = session.decode(&[7], 0).expect("decode after probe");
    assert_eq!(logits.len(), 1);
    assert_eq!(session.current_len(), 1);
}

#[test]
fn native_fused_batch_prefill_rejects_empty() {
    let mut session = synthetic_session();
    assert!(session.run_fused_batch_prefill(&[]).is_err());
}

#[test]
fn native_fused_batch_forward_with_past_row_identical_to_batch_one() {
    // Stage 2b: the batch-N fused forward over a *non-empty* length-L batched KV
    // past must keep rows independent — row `i` bit-for-bit equal to a batch-1
    // forward of the same token at the same past length. This exercises the ONNX
    // attention batch coupling across QKV / mask / past-KV at N > 1 with real
    // past content, which the stage 2a empty-past probe did not.
    const TOKENS: [u32; 4] = [3, 17, 5, 29];
    const PAST_LEN: usize = 6;

    let mut session = synthetic_session();

    let batched = session
        .run_fused_batch_forward(&TOKENS, PAST_LEN)
        .expect("batch-N fused forward with past");
    assert_eq!(batched.len(), TOKENS.len());

    for (index, &token) in TOKENS.iter().enumerate() {
        let single = session
            .run_fused_batch_forward(&[token], PAST_LEN)
            .expect("batch-1 fused forward with past");
        assert_eq!(single.len(), 1);
        assert_eq!(
            bits(&batched[index]),
            bits(&single[0]),
            "row {index} (token {token}) of the batch-{} forward at past_len {PAST_LEN} must be byte-identical to its batch-1 forward",
            TOKENS.len()
        );
    }
}

#[test]
fn native_fused_batch_forward_with_past_is_stateless() {
    // A non-empty batched past must still leave the persistent decode state
    // untouched (cuda.rs KV governance is not engaged).
    let mut session = synthetic_session();
    assert_eq!(session.current_len(), 0);

    let _ = session
        .run_fused_batch_forward(&[1, 2, 3], 8)
        .expect("fused batch forward with past");
    assert_eq!(
        session.current_len(),
        0,
        "probe must not advance decode state even with a non-empty past"
    );

    let logits = session.decode(&[7], 0).expect("decode after probe");
    assert_eq!(logits.len(), 1);
    assert_eq!(session.current_len(), 1);
}

#[test]
fn native_fused_batch_forward_past_zero_matches_prefill() {
    // past_len == 0 must be byte-identical to the stage 2a empty-past prefill.
    const TOKENS: [u32; 3] = [2, 11, 30];
    let mut session = synthetic_session();

    let prefill = session
        .run_fused_batch_prefill(&TOKENS)
        .expect("fused batch prefill");
    let forward = session
        .run_fused_batch_forward(&TOKENS, 0)
        .expect("fused batch forward past_len=0");

    assert_eq!(prefill.len(), forward.len());
    for (index, (a, b)) in prefill.iter().zip(&forward).enumerate() {
        assert_eq!(
            bits(a),
            bits(b),
            "row {index}: run_fused_batch_forward(_, 0) must equal run_fused_batch_prefill"
        );
    }
}

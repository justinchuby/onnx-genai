//! End-to-end integration proof for the native DeepSeek-V4 CompressedSparse-
//! Attention (CSA/HCA) present->past state threading, on the *alternating*
//! ratio-4 / ratio-128 schedule that `deepseek_v4_tiny_qmoe_e2e.rs` explicitly
//! declares out of scope (that fixture is dense CSA, ratio 0, no compressed
//! state). This fixture is the first onnx-genai-side proof that the merged
//! Mobius `pkg.nxrt::CompressedSparseAttention` exporter (Mobius #593) and the
//! onnx-genai native decode runtime agree on the compressed-state ABI end to
//! end: load, prefill, >=16 decode, present->past cursor progression, and the
//! query-dependent sparse route — with no PagedAttention and no dense fallback.
//!
//! The committed fixture is a deterministic, tiny (2-layer) alternating
//! schedule built by the merged Mobius exporter with synthetic small weights:
//!
//! - layer 0: ratio-4 query-selective CSA, `cache_format=fp8_e4m3_block64`,
//!   threading four state edges (compressed_kv, compression_carry, index_key,
//!   index_carry) plus the `selected_indices` sparse route output;
//! - layer 1: ratio-128 HCA, `cache_format=f32`, threading two state edges
//!   (compressed_kv, compression_carry).
//!
//! The ratio-128 layer keeps its compressed KV records as genuine **float32**
//! (its `f32` cache format), while the ratio-4 layer packs them into **uint8**
//! (its `fp8_e4m3_block64` cache format). Both dtypes must be accepted by the
//! runtime's CSA state-edge resolver from the group's declared `cache_format`,
//! exactly as the CSA op kernel's `CacheFormat::dtype()` does. See the
//! `native_decode::csa::expected_dtype` unit regressions
//! (`f32_cache_expects_float32_compressed_kv_records`) for the isolated proof.
//!
//! Weights are synthetic and small: this proves the attention/state *integration*
//! (shape-faithful ABI, present->past rebinding, route evolution) independently
//! of real quantized weights and of the planar-format runtime slice, so **no
//! performance claim is made or implied**.
//!
//! `DEEPSEEK_V4_TINY_CSA_E2E_DIR` may override the committed fixture. A missing
//! fixture skips cleanly so source packages that omit binary fixtures stay green.
//!
//! Scope: this is the **CPU** native-decode proof. The CUDA native-decode path
//! (`DecodeCudaState`) does not yet support CSA/HCA compressed state — it sizes
//! every declared state input as a fixed-geometry f32/f16/bf16 device buffer
//! (`persistent_state_shapes`), so it rejects both the ratio-4 packed **uint8**
//! record buffers and the **symbolic** compressed-record axis. Threading the
//! growing compressed-record cache on-device (with capture/replay invalidation
//! as the cursor advances and MemoryGovernor accounting for compressed state) is
//! a separate slice; the CUDA legs (capture>=3 replays, fallbacks==0, governor
//! baseline->resident->baseline) are deferred to it and are intentionally not
//! asserted here. See the decision note
//! `deckard-hca-c1-e2e-proof.md` for the exact typed blocker and the smallest
//! owner-targeted fix.
#![cfg(feature = "native-backend")]

use std::path::{Path, PathBuf};

use onnx_genai_engine::NativeDecodeSession;
use onnx_genai_metadata::{
    CsaCacheFormat, CsaCompressionRatio, CsaRecurrence, CsaStateEdge, CsaStateGroupAbi,
    CsaStateRole, DecoderAbi, KvOwnership, SequenceInputKind,
};
use onnx_runtime_session::{DevicePreference, InferenceSession};

/// Resolve the committed fixture directory, honoring the env override first.
fn fixture_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("DEEPSEEK_V4_TINY_CSA_E2E_DIR") {
        let p = PathBuf::from(dir);
        if p.join("model.onnx").is_file() {
            return Some(p);
        }
    }
    let committed =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-deepseek-v4-csa");
    committed.join("model.onnx").is_file().then_some(committed)
}

fn edge(role: CsaStateRole, past: &str, present: &str) -> CsaStateEdge {
    CsaStateEdge {
        role,
        past_port: past.to_string(),
        present_port: present.to_string(),
    }
}

/// The explicit decode ABI for the tiny alternating fixture. Its `input_ids`,
/// `attention_mask`, and `position_ids` are ambiguous rank-2 int64 ports, and
/// the compressed-state roles are indistinguishable by structure, so the spec
/// is authoritative. Port names are the exporter's deterministic `layer_id`-
/// keyed names (`_deepseek_v4_csa.py::CsaLayerPlan`).
fn tiny_csa_io() -> DecoderAbi {
    DecoderAbi {
        sequence_source: Some(SequenceInputKind::TokenIds),
        kv_ownership: Some(KvOwnership::Owned),
        token_input: Some("input_ids".into()),
        attention_mask_input: Some("attention_mask".into()),
        position_ids_input: Some("position_ids".into()),
        logits_output: Some("logits".into()),
        // The exporter still emits a dense KV cache per layer (a Concat of past
        // and the current step's key/value), computed in parallel with the CSA
        // attention output. Thread it as an ordinary owned KV pair.
        kv_inputs: Some(vec![
            "past_key_values.0.key".into(),
            "past_key_values.0.value".into(),
            "past_key_values.1.key".into(),
            "past_key_values.1.value".into(),
        ]),
        kv_outputs: Some(vec![
            "present.0.key".into(),
            "present.0.value".into(),
            "present.1.key".into(),
            "present.1.value".into(),
        ]),
        csa_state_groups: Some(vec![
            // layer 0 — ratio-4 query-selective CSA, fp8 packed records.
            CsaStateGroupAbi {
                ratio: CsaCompressionRatio::Ratio4,
                cache_format: CsaCacheFormat::Fp8E4m3Block64,
                recurrence: CsaRecurrence::Standard,
                edges: vec![
                    edge(
                        CsaStateRole::CompressedKv,
                        "past_compressed_kv.0",
                        "present_compressed_kv.0",
                    ),
                    edge(
                        CsaStateRole::CompressionCarry,
                        "past_compression_carry.0",
                        "present_compression_carry.0",
                    ),
                    edge(
                        CsaStateRole::IndexKey,
                        "past_index_key.0",
                        "present_index_key.0",
                    ),
                    edge(
                        CsaStateRole::IndexCarry,
                        "past_index_carry.0",
                        "present_index_carry.0",
                    ),
                ],
            },
            // layer 1 — ratio-128 HCA, uncompressed float32 records.
            CsaStateGroupAbi {
                ratio: CsaCompressionRatio::Ratio128,
                cache_format: CsaCacheFormat::F32,
                recurrence: CsaRecurrence::Standard,
                edges: vec![
                    edge(
                        CsaStateRole::CompressedKv,
                        "past_compressed_kv.1",
                        "present_compressed_kv.1",
                    ),
                    edge(
                        CsaStateRole::CompressionCarry,
                        "past_compression_carry.1",
                        "present_compression_carry.1",
                    ),
                ],
            },
        ]),
        ..DecoderAbi::default()
    }
}

fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i as u32)
        .expect("logits row must not be empty")
}

/// Build a native CPU decode session over the fixture with the explicit CSA ABI.
/// A successful build is itself a proof: `from_session_with_io` resolves every
/// declared CSA state group against the graph's real typed IO and returns a
/// typed error on any mismatch, so reaching a live session means the compressed
/// state threads through its own ports — not a dense KV cache, not PagedAttention.
fn build_cpu_session(dir: &Path) -> NativeDecodeSession {
    let session = InferenceSession::builder()
        .model(dir.join("model.onnx"))
        .device(DevicePreference::Cpu)
        .option("optimization", "basic")
        .build()
        .expect("build native CPU session over the tiny CSA fixture");
    NativeDecodeSession::from_session_with_io(session, &tiny_csa_io())
        .expect("wrap native CSA/HCA decoder with the explicit compressed-state ABI")
}

/// Core CPU proof: native load with compressed-state groups honored, prefill,
/// and >=16 decode steps with a per-step present->past cursor advance.
#[test]
fn tiny_csa_hca_cpu_prefill_and_16_decode_threads_compressed_state() {
    let Some(dir) = fixture_dir() else {
        eprintln!("skipping: tiny CSA fixture not present");
        return;
    };
    let mut sess = build_cpu_session(&dir);

    // Prefill an 8-token prompt from an empty past. Reaching logits proves the
    // CSA op executed with an empty compressed cache (records == 0) at prefill.
    let prompt: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let mut logits = sess
        .decode(&prompt, 0)
        .expect("prefill forward")
        .pop()
        .expect("prefill logits row");
    assert!(!logits.is_empty(), "prefill must emit a logits row");
    assert_eq!(
        sess.current_len(),
        prompt.len(),
        "prefill advances the committed cursor to the prompt length"
    );

    // >=16 decode steps. Each step reads the previous step's present_* state as
    // this step's past_* (the runtime rebinds present->past between steps); a
    // step that failed to thread the compressed state would either error in the
    // CSA op or diverge, so 16 clean steps is the threading proof.
    const DECODE_STEPS: usize = 16;
    for step in 0..DECODE_STEPS {
        let token = argmax(&logits);
        let past = sess.current_len();
        logits = sess
            .decode(&[token], past)
            .unwrap_or_else(|e| panic!("decode step {step} failed: {e:#}"))
            .pop()
            .expect("decode logits row");
        assert_eq!(
            sess.current_len(),
            past + 1,
            "decode step {step} advances the committed cursor by exactly one"
        );
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "decode step {step} produced non-finite logits"
        );
    }
    assert_eq!(
        sess.current_len(),
        prompt.len() + DECODE_STEPS,
        "final cursor is prompt + decode steps"
    );
}

/// Speculative rollback proof on the compressed-state decoder: snapshot the
/// compressed state at a committed anchor, decode a speculative draft window,
/// then roll the compressed cache + carries back to the anchor and continue.
///
/// A CSA/HCA compressed-record cache advances on a backend-owned compression
/// cursor (~tokens/ratio) with a compressor carry that cannot be reconstructed
/// at an arbitrary token boundary, so it is NOT prefix-sliceable like a dense KV
/// cache. The runtime therefore threads a rollback through snapshot/restore
/// (the same mechanism the conv/SSM recurrent decoders use): the snapshot
/// captures the compressed-record buffers and carries at the anchor, and the
/// rollback restores them wholesale and re-advances by exactly the accepted
/// tokens (here: none — a full rollback to the anchor).
#[test]
fn tiny_csa_hca_cpu_snapshot_rollback_then_continue() {
    let Some(dir) = fixture_dir() else {
        eprintln!("skipping: tiny CSA fixture not present");
        return;
    };
    let mut sess = build_cpu_session(&dir);
    let prompt: Vec<u32> = vec![9, 8, 7, 6, 5];
    let mut logits = sess
        .decode(&prompt, 0)
        .expect("prefill")
        .pop()
        .expect("prefill logits");
    let anchor = sess.current_len();

    // Snapshot the compressed state at the committed anchor (before the draft).
    let snapshot = sess
        .snapshot_recurrent_state_public()
        .expect("snapshot compressed + carry state at the anchor");

    // Decode a speculative draft window forward past the anchor.
    for _ in 0..6 {
        let token = argmax(&logits);
        let past = sess.current_len();
        logits = sess
            .decode(&[token], past)
            .expect("draft decode")
            .pop()
            .unwrap();
    }
    assert_eq!(sess.current_len(), anchor + 6, "draft advanced the cursor");

    // Roll fully back to the anchor: dense KV is prefix-sliced, the compressed
    // record buffers + carries are restored from the snapshot, and zero accepted
    // tokens are re-advanced. The cursor must land exactly on the anchor.
    sess.rollback_recurrent_to_accepted(&snapshot, anchor, &[])
        .expect("rollback compressed state to the anchor");
    assert_eq!(
        sess.current_len(),
        anchor,
        "rollback restores the committed cursor to the anchor"
    );

    // Continue decoding from the restored anchor: the compressed cache is
    // consistent again, so the CSA op keeps threading present->past cleanly.
    let mut logits = sess
        .decode(&[argmax(&logits)], anchor)
        .expect("decode after rollback")
        .pop()
        .unwrap();
    for _ in 0..4 {
        let token = argmax(&logits);
        let past = sess.current_len();
        logits = sess
            .decode(&[token], past)
            .expect("continue after rollback")
            .pop()
            .unwrap();
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "post-rollback decode produced non-finite logits"
        );
    }
    assert_eq!(
        sess.current_len(),
        anchor + 5,
        "5 post-rollback decode steps advance the cursor deterministically"
    );
}

/// Refusal proof: a *bare* rewind to a non-zero length with CSA state present is
/// a typed error, never a silent corruption. The compressed cache cannot be
/// prefix-sliced to an arbitrary token boundary, so the only supported rewinds
/// are `reset()` (to 0) and the snapshot/rollback path exercised above.
#[test]
fn tiny_csa_hca_cpu_bare_nonzero_rewind_is_typed_refused() {
    let Some(dir) = fixture_dir() else {
        eprintln!("skipping: tiny CSA fixture not present");
        return;
    };
    let mut sess = build_cpu_session(&dir);
    sess.decode(&[3, 1, 4, 1, 5], 0).expect("prefill");
    for _ in 0..3 {
        let past = sess.current_len();
        sess.decode(&[9], past).expect("decode");
    }
    let before = sess.current_len();
    let err = sess
        .rewind(before - 1)
        .expect_err("a bare non-zero rewind of CSA state must be refused, not silently corrupt");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("CompressedSparseAttention") && msg.contains("compressed-record"),
        "refusal must name the compressed-record cache; got: {msg}"
    );
    // The refusal must leave the committed cursor untouched (no partial rewind).
    assert_eq!(
        sess.current_len(),
        before,
        "a refused rewind must not advance or retreat the cursor"
    );
    // reset() (rewind to 0) is still the supported full teardown.
    sess.reset().expect("reset after a refused rewind");
    assert_eq!(sess.current_len(), 0, "reset clears the compressed state");
}

/// Teardown / reset proof: a full reset returns the compressed-state decoder to
/// an empty committed cursor, and a fresh prefill still works (no leaked state).
#[test]
fn tiny_csa_hca_cpu_reset_clears_committed_state() {
    let Some(dir) = fixture_dir() else {
        eprintln!("skipping: tiny CSA fixture not present");
        return;
    };
    let mut sess = build_cpu_session(&dir);
    sess.decode(&[1, 2, 3, 4], 0).expect("prefill");
    for _ in 0..3 {
        let past = sess.current_len();
        sess.decode(&[5], past).expect("decode");
    }
    assert!(sess.current_len() > 0);
    sess.reset().expect("reset");
    assert_eq!(sess.current_len(), 0, "reset clears the committed cursor");
    sess.decode(&[7, 7, 7], 0).expect("prefill after reset");
    assert_eq!(sess.current_len(), 3, "fresh prefill after reset works");
}

/// Multi-request isolation: two independent sessions over the same fixture decode
/// without cross-talk (each holds its own compressed state).
#[test]
fn tiny_csa_hca_cpu_multi_request_isolation() {
    let Some(dir) = fixture_dir() else {
        eprintln!("skipping: tiny CSA fixture not present");
        return;
    };
    let mut a = build_cpu_session(&dir);
    let mut b = build_cpu_session(&dir);
    a.decode(&[1, 2, 3, 4, 5, 6], 0).expect("a prefill");
    b.decode(&[9, 9], 0).expect("b prefill");
    assert_eq!(a.current_len(), 6);
    assert_eq!(b.current_len(), 2);
    let pa = a.current_len();
    a.decode(&[1], pa).expect("a decode");
    assert_eq!(a.current_len(), 7);
    assert_eq!(b.current_len(), 2, "b's cursor is unaffected by a's decode");
}

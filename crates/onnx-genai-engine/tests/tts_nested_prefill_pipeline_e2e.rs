//! End-to-end test for the **prefill + trailing-text pre-embedder-driven
//! nested-autoregressive** (multi-decoder TTS) pipeline shape — the real
//! Qwen3-TTS talker where the outer decoder is (1) PREFILLED on frame 0 with a
//! real multi-position embedding sequence and (2) conditioned on one
//! trailing-text embedding per subsequent frame (DESIGN.md §20.3, optional
//! `prefill_embedder` extension of the `nested_autoregressive` contract).
//!
//! Runs `Engine::from_pipeline_dir` + `PipelineEngine::synthesize` on the
//! deterministic fixture built by `scripts/build_tiny_tts_nested_prefill.py`:
//!
//!   * `talker_prefill_embedder` (`prompt_only`): `text_ids[1, L] ->
//!     prefill_embeds[1, P, HIDDEN]` (every position holds `TS == Σ text_ids`,
//!     the talker's frame-0 seed) `+ trailing_text_embeds[1, T, HIDDEN]` (row `k`
//!     holds `k + 1`, the pre-embedder's `text_embed` on frame `k + 1`).
//!   * `talker_step_embedder` (pre-embedder): `frame_codes + text_embed ->
//!     inputs_embeds == Σ_i frame_codes[i] + text_embed`.
//!   * `talker` (outer AR): `argmax logits == round(S)`, `last_hidden_state == S`,
//!     `S == mean_over_hidden(inputs_embeds)`.
//!   * `code_predictor` (inner AR): `code[f][g] == S_f + g + 1`.
//!   * `vocoder` (`final_only`): `audio == 2 * flatten(codes)`.
//!
//! With `num_code_groups = 4`, `max_frames = 2`, `P = 2`, prompt `[8]`
//! (`TS == 8`) the recurrence yields pool codes `[[9,10,11,12],[43,44,45,46]]`:
//!   * frame 0 PREFILL: `S_0 == TS == 8` (prompt-derived, impossible under the
//!     zero-seed path) -> inner codes `[9,10,11,12]`, `code_0 == 8`.
//!   * frame 1: `frame_codes=[8,10,11,12]` (Σ=41) `+ t_0=1` -> `S_1 == 42` ->
//!     inner codes `[43,44,45,46]`.
//!
//! This is distinct from the ZERO-SEED (no-prefill) path
//! `[[1,2,3,4],[10,11,12,13]]`, proving the prefill + trailing-text path is
//! exercised. The test asserts the exact codes, shape, and closed-form waveform.

use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest, PipelineGenerateRequest,
};

const FRAMES: usize = 2;
const GROUPS: usize = 4;
const PREFILL_LEN: usize = 2;
const TRAILING_LEN: usize = 2;
const PROMPT_ID: u32 = 8;

fn tiny_tts_nested_prefill_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-tts-nested-prefill")
}

/// Replicate the engine's prefill + trailing-text recurrence. Frame 0's talker
/// mean is the prompt-derived `TS == Σ prompt_ids` (multi-position PREFILL feed);
/// frame `k >= 1` sums the previous tuple and adds `trailing[k-1] == k`.
fn expected_pool_codes(prompt_ids: &[i64]) -> Vec<i64> {
    let ts: i64 = prompt_ids.iter().sum();
    let trailing: Vec<i64> = (0..TRAILING_LEN as i64).map(|k| k + 1).collect();
    let mut codes = vec![0i64; FRAMES * GROUPS];
    let mut prev_tuple: Option<Vec<i64>> = None;
    for f in 0..FRAMES {
        let s = if f == 0 {
            ts
        } else {
            let frame_codes = prev_tuple.clone().expect("frames >= 1 have a prev tuple");
            let t = trailing.get(f - 1).copied().unwrap_or(0);
            frame_codes.iter().sum::<i64>() + t
        };
        let outer_code_0 = s;
        for g in 0..GROUPS {
            codes[f * GROUPS + g] = s + g as i64 + 1;
        }
        let mut tuple = Vec::with_capacity(GROUPS);
        tuple.push(outer_code_0);
        for g in 1..GROUPS {
            tuple.push(codes[f * GROUPS + g]);
        }
        prev_tuple = Some(tuple);
    }
    codes
}

/// The ZERO-SEED (no prefill embedder) path: frame 0 seeds `S_0 == 0` and every
/// `text_embed` is zero. The prefill path must diverge from it.
fn zero_seed_path_codes() -> Vec<i64> {
    let mut codes = vec![0i64; FRAMES * GROUPS];
    let mut prev_tuple: Option<Vec<i64>> = None;
    for f in 0..FRAMES {
        let frame_codes = prev_tuple.clone().unwrap_or_else(|| vec![0i64; GROUPS]);
        let s: i64 = frame_codes.iter().sum();
        let outer_code_0 = s;
        for g in 0..GROUPS {
            codes[f * GROUPS + g] = s + g as i64 + 1;
        }
        let mut tuple = Vec::with_capacity(GROUPS);
        tuple.push(outer_code_0);
        for g in 1..GROUPS {
            tuple.push(codes[f * GROUPS + g]);
        }
        prev_tuple = Some(tuple);
    }
    codes
}

#[test]
fn nested_ar_prefill_pipeline_prefills_and_threads_trailing_text() -> anyhow::Result<()> {
    let mut engine =
        Engine::from_pipeline_dir(&tiny_tts_nested_prefill_dir(), EngineConfig::default())?;

    // The prompt tokens ARE consumed here: they seed the prefill embedder's
    // `text_ids`, so `TS == 8` drives the talker's frame-0 prefill.
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![PROMPT_ID]));
    request.options = GenerateOptions {
        max_new_tokens: 4,
        temperature: 0.0,
        stop_on_eos: false,
        ..GenerateOptions::default()
    };

    let synthesis = engine.synthesize(PipelineGenerateRequest::new(request))?;

    let expected = expected_pool_codes(&[PROMPT_ID as i64]);
    let expected_tokens: Vec<u32> = expected.iter().map(|&c| c as u32).collect();
    assert_eq!(synthesis.generation.token_ids, expected_tokens);

    // (a) The talker is PREFILLED on frame 0: its frame-0 mean is the
    // prompt-derived `TS == 8` (not `0`), so the first frame's inner codes are
    // `[TS + g + 1] == [9,10,11,12]`. A single zero-seed position could never
    // yield these — this is only reachable by feeding the multi-position
    // `prefill_embeds` (length `P == 2`) directly to the talker's `inputs_embeds`.
    let ts = PROMPT_ID as i64;
    for (g, &code) in expected[..GROUPS].iter().enumerate() {
        assert_eq!(
            code,
            ts + g as i64 + 1,
            "frame-0 code {g} must reflect the prompt-derived prefill mean TS == {ts}"
        );
    }
    assert_eq!(PREFILL_LEN, 2, "the prefill sequence is multi-position");

    // (b) The prefill + trailing-text stream must diverge from the zero-seed path.
    assert_ne!(
        expected,
        zero_seed_path_codes(),
        "prefill codes must differ from the zero-seed (no-prefill) path"
    );

    // (c) The assembled codes are published as `talker.output_codes` [1, F, G].
    let codes_value = synthesis
        .tensors
        .get("talker.output_codes")
        .expect("assembled codes published as talker.output_codes");
    assert_eq!(codes_value.shape(), &[1, FRAMES as i64, GROUPS as i64]);
    assert_eq!(codes_value.to_vec_i64().expect("codes int64"), expected);

    // (c) The post-decode vocoder reconstructs the waveform: 2 * flatten(codes).
    let audio = synthesis
        .tensors
        .get("vocoder.audio")
        .expect("post-decode vocoder stage output present in the shared pool")
        .to_vec_f32()?;
    let expected_audio: Vec<f32> = expected.iter().map(|&c| (2 * c) as f32).collect();
    assert_eq!(audio.len(), expected_audio.len());
    for (got, want) in audio.iter().zip(&expected_audio) {
        assert!((got - want).abs() < 1e-5, "audio {got} != {want}");
    }

    Ok(())
}

#[test]
fn generate_on_nested_ar_prefill_pipeline_returns_codes_only() -> anyhow::Result<()> {
    let mut engine =
        Engine::from_pipeline_dir(&tiny_tts_nested_prefill_dir(), EngineConfig::default())?;

    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![PROMPT_ID]));
    request.options = GenerateOptions {
        max_new_tokens: 4,
        temperature: 0.0,
        stop_on_eos: false,
        ..GenerateOptions::default()
    };

    let result = engine.generate(request)?;
    let expected: Vec<u32> = expected_pool_codes(&[PROMPT_ID as i64])
        .iter()
        .map(|&c| c as u32)
        .collect();
    assert_eq!(result.token_ids, expected);
    Ok(())
}

//! End-to-end test for the **pre-embedder-driven nested-autoregressive**
//! (multi-decoder TTS) pipeline shape — the real Qwen3-TTS talker where the
//! outer decoder is driven by `inputs_embeds` materialized each frame from the
//! PREVIOUS frame's codes, NOT by `input_ids` (DESIGN.md §20.3, optional
//! `pre_embedder` extension of the `nested_autoregressive` contract).
//!
//! Runs `Engine::from_pipeline_dir` + `PipelineEngine::synthesize` on the
//! deterministic fixture built by `scripts/build_tiny_tts_nested_preembed.py`:
//!
//!   * `talker_step_embedder` (pre-embedder): `frame_codes[1, G] (+text_embed) ->
//!     inputs_embeds[1, 1, HIDDEN]`, summing a tiny identity codec table over the
//!     group axis so `inputs_embeds == Σ_i frame_codes[i]` (text_embed fed zeros).
//!   * `talker` (outer AR, inputs_embeds-driven): `argmax logits == round(S)` and
//!     `last_hidden_state == S` broadcast, where `S == Σ_i frame_codes[i]`.
//!   * `code_predictor` (inner AR): `code == mean(inputs_embeds) + 1`, seeded by
//!     the talker hidden state, so `code[f][g] == S_f + g + 1`.
//!   * `vocoder` (`final_only`): `codes[1, F, G] -> audio == 2 * flatten(codes)`.
//!
//! The engine assembles the next frame's `frame_codes` from the previous frame's
//! tuple `[outer_code_0, inner_code_1, ..., inner_code_{G-1}]`, so with
//! `num_code_groups = 4`, `max_frames = 3` the recurrence yields pool codes
//! `[[1,2,3,4],[10,11,12,13],[46,47,48,49]]` — distinct from the `input_ids`-driven
//! `tiny-tts-nested` fixture (`[[1,2,3,4],[2,3,4,5],[3,4,5,6]]`), proving the
//! pre-embedder path is exercised. The test asserts the exact codes, shape, and
//! the closed-form waveform.

use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest, PipelineGenerateRequest,
};

const FRAMES: usize = 3;
const GROUPS: usize = 4;

fn tiny_tts_nested_preembed_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-tts-nested-preembed")
}

/// Replicate the engine's pre-embedder-driven recurrence to derive the expected
/// per-frame pool codes: `frame_codes` sums to `S`; the talker code (group 0 of
/// the next tuple) is `round(S) == S`; the inner codes are `S + g + 1`.
fn expected_pool_codes() -> Vec<i64> {
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

/// The `input_ids`-driven `tiny-tts-nested` fixture would yield `f + g + 1`; the
/// pre-embedder path must diverge from it (else the new path is not exercised).
fn input_ids_path_codes() -> Vec<i64> {
    let mut codes = vec![0i64; FRAMES * GROUPS];
    for f in 0..FRAMES {
        for g in 0..GROUPS {
            codes[f * GROUPS + g] = (f + g + 1) as i64;
        }
    }
    codes
}

#[test]
fn nested_ar_preembed_pipeline_drives_talker_through_step_embedder() -> anyhow::Result<()> {
    let mut engine =
        Engine::from_pipeline_dir(&tiny_tts_nested_preembed_dir(), EngineConfig::default())?;

    // A single-token prompt (id irrelevant: the talker is driven by the
    // pre-embedder, not by these tokens).
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![8]));
    request.options = GenerateOptions {
        max_new_tokens: 4,
        temperature: 0.0,
        stop_on_eos: false,
        ..GenerateOptions::default()
    };

    let synthesis = engine.synthesize(PipelineGenerateRequest::new(request))?;

    let expected = expected_pool_codes();
    let expected_tokens: Vec<u32> = expected.iter().map(|&c| c as u32).collect();
    assert_eq!(synthesis.generation.token_ids, expected_tokens);

    // The assembled codes are published as `talker.output_codes` [1, F, G].
    let codes_value = synthesis
        .tensors
        .get("talker.output_codes")
        .expect("assembled codes published as talker.output_codes");
    assert_eq!(codes_value.shape(), &[1, FRAMES as i64, GROUPS as i64]);
    assert_eq!(codes_value.to_vec_i64().expect("codes int64"), expected);

    // The pre-embedder path must diverge from the input_ids-driven path.
    assert_ne!(
        expected,
        input_ids_path_codes(),
        "pre-embedder codes must differ from the input_ids-driven path"
    );

    // The post-decode vocoder stage reconstructs the waveform: 2 * flatten(codes).
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
fn generate_on_nested_ar_preembed_pipeline_returns_codes_only() -> anyhow::Result<()> {
    // `generate` drives the pre-embedder-fed nested loops and returns the
    // flattened code tokens, without running the post-decode vocoder stage.
    let mut engine =
        Engine::from_pipeline_dir(&tiny_tts_nested_preembed_dir(), EngineConfig::default())?;

    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![8]));
    request.options = GenerateOptions {
        max_new_tokens: 4,
        temperature: 0.0,
        stop_on_eos: false,
        ..GenerateOptions::default()
    };

    let result = engine.generate(request)?;
    let expected: Vec<u32> = expected_pool_codes().iter().map(|&c| c as u32).collect();
    assert_eq!(result.token_ids, expected);
    Ok(())
}

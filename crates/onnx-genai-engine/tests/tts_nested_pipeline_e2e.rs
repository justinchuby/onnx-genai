//! End-to-end test for the **nested-autoregressive** (multi-decoder TTS)
//! composite pipeline shape — the Qwen3-TTS-style dual, hierarchically-nested AR
//! loop that the flat composite contract could not express (DESIGN.md §20.3).
//!
//! Runs `Engine::from_pipeline_dir` + `PipelineEngine::synthesize` on the
//! deterministic fixture built by `scripts/build_tiny_tts_nested.py`:
//!
//!   * `talker` (outer AR): logits `-(v - position)^2` (argmax first-code group
//!     == position) plus a per-frame `last_hidden_state == position` broadcast
//!     across the hidden dim. From a 1-token prompt, `seed_f == f`.
//!   * `code_predictor` (inner AR): `code == mean(inputs_embeds) + 1`, emitting
//!     `code_embeds` that the engine threads back as the next inner step's
//!     `inputs_embeds`. Inner step 0 is seeded by the talker hidden state, so
//!     `code[f][g] == f + g + 1`.
//!   * `vocoder` (`final_only` single_pass): `codes[1, F, G] -> audio[1, F*G]`
//!     with `audio == 2 * flatten(codes)`.
//!
//! With `num_code_groups = 4` and `max_frames = 3` the assembled per-frame code
//! groups are `[[1,2,3,4],[2,3,4,5],[3,4,5,6]]`, published into the shared pool
//! as `talker.output_codes` and routed to the vocoder by the dataflow edge
//! `talker.output_codes -> vocoder.codes`. The test asserts the exact codes AND
//! the closed-form waveform `[2,4,6,8,4,6,8,10,6,8,10,12]`.

use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest, PipelineGenerateRequest,
};

const FRAMES: usize = 3;
const GROUPS: usize = 4;

fn tiny_tts_nested_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-tts-nested")
}

fn expected_codes() -> Vec<i64> {
    let mut codes = Vec::with_capacity(FRAMES * GROUPS);
    for f in 0..FRAMES {
        for g in 0..GROUPS {
            codes.push((f + g + 1) as i64);
        }
    }
    codes
}

#[test]
fn nested_ar_pipeline_decodes_code_groups_then_vocodes_waveform() -> anyhow::Result<()> {
    let mut engine = Engine::from_pipeline_dir(&tiny_tts_nested_dir(), EngineConfig::default())?;

    // A single-token prompt (id irrelevant: the talker's argmax depends only on
    // position). Three outer frames at positions 0..2, each expanding into four
    // residual code groups.
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![8]));
    request.options = GenerateOptions {
        max_new_tokens: 4,
        temperature: 0.0,
        stop_on_eos: false,
        ..GenerateOptions::default()
    };

    let synthesis = engine.synthesize(PipelineGenerateRequest::new(request))?;

    // Flattened per-frame code groups are returned as the generated token ids.
    let expected: Vec<u32> = expected_codes().iter().map(|&c| c as u32).collect();
    assert_eq!(synthesis.generation.token_ids, expected);

    // The assembled codes are published as `talker.output_codes` [1, F, G].
    let codes_value = synthesis
        .tensors
        .get("talker.output_codes")
        .expect("assembled codes published as talker.output_codes");
    assert_eq!(codes_value.shape(), &[1, FRAMES as i64, GROUPS as i64]);
    assert_eq!(
        codes_value.to_vec_i64().expect("codes int64"),
        expected_codes()
    );

    // The post-decode vocoder stage reconstructs the waveform: 2 * flatten(codes).
    let audio = synthesis
        .tensors
        .get("vocoder.audio")
        .expect("post-decode vocoder stage output present in the shared pool")
        .to_vec_f32()?;
    let expected_audio: Vec<f32> = expected_codes().iter().map(|&c| (2 * c) as f32).collect();
    assert_eq!(audio.len(), expected_audio.len());
    for (got, want) in audio.iter().zip(&expected_audio) {
        assert!((got - want).abs() < 1e-5, "audio {got} != {want}");
    }

    Ok(())
}

#[test]
fn generate_on_nested_ar_pipeline_returns_codes_only() -> anyhow::Result<()> {
    // `generate` drives the nested loops and returns the flattened code tokens,
    // without running the post-decode vocoder stage. Callers who want the
    // waveform use `synthesize`.
    let mut engine = Engine::from_pipeline_dir(&tiny_tts_nested_dir(), EngineConfig::default())?;

    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![8]));
    request.options = GenerateOptions {
        max_new_tokens: 4,
        temperature: 0.0,
        stop_on_eos: false,
        ..GenerateOptions::default()
    };

    let result = engine.generate(request)?;
    let expected: Vec<u32> = expected_codes().iter().map(|&c| c as u32).collect();
    assert_eq!(result.token_ids, expected);
    Ok(())
}

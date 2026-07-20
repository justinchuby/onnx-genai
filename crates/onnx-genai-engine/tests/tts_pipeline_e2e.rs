//! End-to-end test for the **text-to-speech (TTS)** composite pipeline shape —
//! an autoregressive decoder that emits audio *code* tokens, followed by a
//! **post-decode `final_only` single_pass vocoder stage** (DESIGN.md §20). This
//! is the one composite structure the AR path could not previously express.
//!
//! Runs `Engine::from_pipeline_dir` + `PipelineEngine::synthesize` on the
//! deterministic fixture built by `scripts/build_tiny_tts.py`:
//!
//!   * `decoder` (autoregressive): logits `-(vocab_index - position)^2`, so
//!     `argmax == position`; from a 1-token prompt (position 0) the greedy code
//!     sequence is `[0, 1, 2, 3]`.
//!   * `vocoder` (`final_only` single_pass): `codes[1,T] -> audio[1, T*2]` with
//!     `audio[i*2 + j] = codes[i] * 2`.
//!
//! The generated codes are published into the shared pool as the synthetic
//! tensor `decoder.output_ids` and routed to the vocoder by the pipeline
//! `dataflow` edge `decoder.output_ids -> vocoder.codes`. The test asserts the
//! exact generated code ids AND the closed-form waveform
//! `[0, 0, 2, 2, 4, 4, 6, 6]`.

use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest, PipelineGenerateRequest,
};

fn tiny_tts_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-tts")
}

#[test]
fn tts_composite_decodes_codes_then_vocodes_waveform() -> anyhow::Result<()> {
    let mut engine = Engine::from_pipeline_dir(&tiny_tts_dir(), EngineConfig::default())?;

    // A single-token prompt (id is irrelevant: the decoder's argmax depends only
    // on position). Four decode steps at positions 0..3 emit codes [0, 1, 2, 3].
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![8]));
    request.options = GenerateOptions {
        max_new_tokens: 4,
        temperature: 0.0,
        stop_on_eos: false,
        ..GenerateOptions::default()
    };

    let synthesis = engine.synthesize(PipelineGenerateRequest::new(request))?;

    // Exact generated code token ids (also asserted by the Python builder).
    assert_eq!(synthesis.generation.token_ids, vec![0, 1, 2, 3]);

    // The generated codes are published into the shared pool as a tensor.
    let codes = synthesis
        .tensors
        .get("decoder.output_ids")
        .expect("generated codes published as decoder.output_ids")
        .to_vec_i64()
        .expect("codes tensor is int64");
    assert_eq!(codes, vec![0, 1, 2, 3]);

    // The post-decode vocoder stage reconstructs the waveform: each code scaled
    // by 2 and repeated twice -> audio[i*2 + j] = codes[i] * 2.
    let audio = synthesis
        .tensors
        .get("vocoder.audio")
        .expect("post-decode vocoder stage output present in the shared pool")
        .to_vec_f32()?;
    let expected: Vec<f32> = vec![0.0, 0.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0];
    assert_eq!(audio.len(), expected.len());
    for (got, want) in audio.iter().zip(&expected) {
        assert!((got - want).abs() < 1e-5, "audio {got} != {want}");
    }

    Ok(())
}

#[test]
fn generate_on_tts_pipeline_returns_codes_only() -> anyhow::Result<()> {
    // `generate` still works on a TTS pipeline: it drives the AR loop and returns
    // the code tokens, without running the post-decode vocoder stage. Callers who
    // want the waveform use `synthesize`.
    let mut engine = Engine::from_pipeline_dir(&tiny_tts_dir(), EngineConfig::default())?;

    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![8]));
    request.options = GenerateOptions {
        max_new_tokens: 4,
        temperature: 0.0,
        stop_on_eos: false,
        ..GenerateOptions::default()
    };

    let result = engine.generate(request)?;
    assert_eq!(result.token_ids, vec![0, 1, 2, 3]);
    Ok(())
}

//! Static cross-attention KV binding regression for encoder-decoder pipelines.
//!
//! Drives the synthetic `tiny-whisper-cross-kv` fixture, whose shape mirrors the
//! real Foundry / ORT-genai Whisper export: the ENCODER emits cross-attention KV
//! once (`present_*_cross_%d`) and the DECODER consumes it as STATIC
//! `past_*_cross_%d` inputs plus its own GROWING self-attention KV. The decoder
//! graph genuinely requires the cross-KV inputs, so a passing run proves the
//! engine ran the encoder prologue once, captured its cross-KV outputs, and
//! re-bound them to the decoder on every autoregressive step. `input_ids` is
//! INT32, also exercising the Int32 token input path.

use std::path::{Path, PathBuf};

use onnx_genai_engine::pipeline::PipelineGenerateRequest;
use onnx_genai_engine::{Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest};
use onnx_genai_ort::Value;
use onnx_genai_preprocess::audio::{LogMelExtractor, WHISPER_SAMPLE_RATE, decode_wav_pcm16};

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-whisper-cross-kv")
}

#[test]
#[ignore = "synthetic Whisper static-cross-KV contract test; run explicitly with ORT runtime libs"]
fn static_cross_kv_binds_encoder_outputs_to_decoder() -> anyhow::Result<()> {
    let model_dir = fixture_dir();
    let audio = decode_wav_pcm16(&std::fs::read(model_dir.join("tiny.wav"))?)?;
    let features = LogMelExtractor::new(80, WHISPER_SAMPLE_RATE)?
        .extract(&audio.samples, audio.sample_rate)?;
    assert_eq!(features.shape(), [1, 80, 8]);
    let audio_features = Value::from_vec_f32(features.data, &[1, 80, 8])?;

    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![2]));
    request.options = GenerateOptions {
        max_new_tokens: 3,
        temperature: 0.0,
        stop_on_eos: false,
        ..GenerateOptions::default()
    };

    let mut engine = Engine::from_pipeline_dir(&model_dir, EngineConfig::default())?;
    let result = engine.generate_with_pipeline_request(
        PipelineGenerateRequest::new(request).with_input("encoder.audio_features", audio_features),
    )?;

    // Reaching a deterministic transcription requires the static cross-KV inputs
    // (`past_*_cross_0`) to have been bound from the encoder prologue; the decoder
    // graph errors on a missing input otherwise. The token bias fixes argmax to 4.
    assert_eq!(result.token_ids, vec![4, 4, 4]);
    Ok(())
}

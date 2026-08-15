use std::path::{Path, PathBuf};

use onnx_genai_engine::pipeline::PipelineGenerateRequest;
use onnx_genai_engine::{Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest};
use onnx_genai_ort::Value;
use onnx_genai_preprocess::audio::{LogMelExtractor, WHISPER_SAMPLE_RATE, decode_wav_pcm16};

fn tiny_whisper_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-whisper")
}

#[test]
#[ignore = "synthetic Whisper-contract smoke test; run explicitly for audio pipeline validation"]
fn tiny_wav_runs_through_whisper_pipeline() -> anyhow::Result<()> {
    let model_dir = tiny_whisper_dir();
    let audio = decode_wav_pcm16(&std::fs::read(model_dir.join("tiny.wav"))?)?;
    let features = LogMelExtractor::new(80, WHISPER_SAMPLE_RATE)?
        .extract(&audio.samples, audio.sample_rate)?;
    assert_eq!(features.shape(), [1, 80, 8]);

    let input_features = Value::from_vec_f32(features.data, &[1, 80, 8])?;
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![2]));
    request.options = GenerateOptions {
        max_new_tokens: 2,
        temperature: 0.0,
        stop_on_eos: false,
        ..GenerateOptions::default()
    };

    let mut engine = Engine::from_pipeline_dir(&model_dir, EngineConfig::default())?;
    let result = engine.generate_with_pipeline_request(
        PipelineGenerateRequest::new(request).with_input("encoder.input_features", input_features),
    )?;
    assert_eq!(result.token_ids, vec![4, 4]);
    Ok(())
}

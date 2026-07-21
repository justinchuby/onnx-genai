//! Optional end-to-end regression harness for a **real** Whisper package running
//! through the onnx-genai composite ASR pipeline (encoder → cross-attention AR
//! decoder). Unlike the synthetic `tiny-whisper` contract test, this drives a
//! real Mobius-built package on real audio and asserts a real transcription.
//!
//! It is env-gated so it never runs in CI without inputs. Build a package and
//! run it explicitly:
//!
//! ```sh
//! python -m mobius build --model openai/whisper-tiny --task speech-to-text \
//!   --runtime onnx-genai /tmp/wsp
//! curl -sL https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav \
//!   -o /tmp/jfk.wav
//! WHISPER_PKG_DIR=/tmp/wsp WHISPER_WAV=/tmp/jfk.wav \
//!   cargo test -p onnx-genai-engine --test whisper_real_e2e -- --ignored --nocapture
//! ```

use onnx_genai_engine::pipeline::PipelineGenerateRequest;
use onnx_genai_engine::{Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest};
use onnx_genai_ort::Value;
use onnx_genai_preprocess::audio::{decode_wav_pcm16, LogMelExtractor, WHISPER_SAMPLE_RATE};

// Whisper multilingual forced decoder prompt: <|startoftranscript|> <|en|>
// <|transcribe|> <|notimestamps|>.
const WHISPER_SOT_PROMPT: [u32; 4] = [50258, 50259, 50359, 50363];

#[test]
#[ignore = "requires a real Whisper package (WHISPER_PKG_DIR) + audio (WHISPER_WAV)"]
fn real_whisper_transcribes_audio_through_engine() -> anyhow::Result<()> {
    let (Ok(pkg_dir), Ok(wav_path)) = (
        std::env::var("WHISPER_PKG_DIR"),
        std::env::var("WHISPER_WAV"),
    ) else {
        eprintln!("skipping: set WHISPER_PKG_DIR and WHISPER_WAV to run this harness");
        return Ok(());
    };

    let audio = decode_wav_pcm16(&std::fs::read(&wav_path)?)?;
    let features = LogMelExtractor::new(80, WHISPER_SAMPLE_RATE)?
        .extract_padded(&audio.samples, audio.sample_rate)?;
    assert_eq!(features.shape(), [1, 80, 3000], "Whisper encoder input");
    let input_features = Value::from_vec_f32(features.data, &[1, 80, 3000])?;

    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(WHISPER_SOT_PROMPT.to_vec()));
    request.options = GenerateOptions {
        max_new_tokens: 64,
        temperature: 0.0,
        ..GenerateOptions::default()
    };

    let mut engine =
        Engine::from_pipeline_dir(std::path::Path::new(&pkg_dir), EngineConfig::default())?;
    let result = engine.generate_with_pipeline_request(
        PipelineGenerateRequest::new(request).with_input("encoder.input_features", input_features),
    )?;

    eprintln!("transcription: {:?}", result.text);
    assert!(!result.token_ids.is_empty(), "expected generated tokens");
    let text = result.text.to_lowercase();
    assert!(
        text.contains("country") || text.contains("americans"),
        "unexpected transcription: {:?}",
        result.text
    );
    Ok(())
}

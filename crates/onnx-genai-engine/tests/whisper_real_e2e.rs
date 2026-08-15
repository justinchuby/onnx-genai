//! Optional end-to-end regression harness for a **real** Whisper package running
//! through the onnx-genai composite ASR pipeline (encoder → cross-attention AR
//! decoder). It drives a real package on real audio and asserts a real
//! transcription, exercising the static cross-attention KV binding wired for
//! encoder-decoder pipelines (the encoder prologue's `present_*_cross_%d`
//! outputs are bound to the decoder's `past_*_cross_%d` inputs every step).
//!
//! It is skip-if-absent so it never fails CI without inputs. It accepts either a
//! Foundry / ORT-genai Whisper cache package (encoder + decoder `.onnx` +
//! `genai_config.json`, loaded through the ORT compatibility path) or a native
//! Mobius-built package. Provide inputs explicitly:
//!
//! ```sh
//! export LD_LIBRARY_PATH=$(find target -type d -path '*onnx-genai-ort-sys*/out/ort-prebuilt/lib' | head -1):$LD_LIBRARY_PATH
//! WHISPER_PKG_DIR=/path/to/whisper-tiny/v4 WHISPER_WAV=/path/to/audio_16k.wav \
//!   ONNX_GENAI_BACKEND=ort \
//!   cargo test -p onnx-genai-engine --test whisper_real_e2e -- --ignored --nocapture
//! ```
//!
//! The encoder prompt input port defaults to `audio_features` (Foundry Whisper)
//! and can be overridden with `WHISPER_FEATURES_INPUT` (e.g. `input_features`).

use onnx_genai_engine::pipeline::PipelineGenerateRequest;
use onnx_genai_engine::{Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest};
use onnx_genai_ort::Value;
use onnx_genai_preprocess::audio::{LogMelExtractor, WHISPER_SAMPLE_RATE, decode_wav_pcm16};

// Whisper multilingual forced decoder prompt: <|startoftranscript|> <|en|>
// <|transcribe|> <|notimestamps|>.
const WHISPER_SOT_PROMPT: [u32; 4] = [50258, 50259, 50359, 50363];

#[test]
#[ignore = "requires a real Whisper package (WHISPER_PKG_DIR) + 16 kHz mono PCM audio (WHISPER_WAV)"]
fn real_whisper_transcribes_audio_through_engine() -> anyhow::Result<()> {
    let (Ok(pkg_dir), Ok(wav_path)) = (
        std::env::var("WHISPER_PKG_DIR"),
        std::env::var("WHISPER_WAV"),
    ) else {
        eprintln!("skipping: set WHISPER_PKG_DIR and WHISPER_WAV to run this harness");
        return Ok(());
    };
    if !std::path::Path::new(&pkg_dir).exists() || !std::path::Path::new(&wav_path).exists() {
        eprintln!("skipping: WHISPER_PKG_DIR or WHISPER_WAV path does not exist");
        return Ok(());
    }
    let features_input =
        std::env::var("WHISPER_FEATURES_INPUT").unwrap_or_else(|_| "audio_features".to_string());

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
        PipelineGenerateRequest::new(request)
            .with_input(format!("encoder.{features_input}"), input_features),
    )?;

    eprintln!("transcription: {:?}", result.text);
    eprintln!("token_ids: {:?}", result.token_ids);
    assert!(!result.token_ids.is_empty(), "expected generated tokens");
    let text = result.text.to_lowercase();
    assert!(
        text.contains("country") || text.contains("americans"),
        "unexpected transcription: {:?}",
        result.text
    );
    Ok(())
}

//! End-to-end test for the text-to-speech renderer that backs
//! `onnx-genai generate --output-audio` and `POST /v1/audio/speech`.
//!
//! Uses the deterministic fixture built by `scripts/build_tiny_tts.py`:
//!
//!   * `decoder` (autoregressive): argmax equals the position, so four decode
//!     steps emit codes `[0, 1, 2, 3]`.
//!   * `vocoder` (`final_only`): `codes[1, T] -> audio[1, T * 2]` with
//!     `audio[i * 2 + j] = codes[i] * 2`, giving `[0, 0, 2, 2, 4, 4, 6, 6]`.
//!
//! The fixture declares `pipeline.audio.sample_rate: 16000`, so the renderer
//! must carry that rate through to the encoded WAV rather than guessing one.

use std::path::{Path, PathBuf};

use onnx_genai::engine::{EngineConfig, PipelineEngine};
use onnx_genai::ort::Tokenizer;
use onnx_genai::text_to_audio::{self, SynthesizedAudio, TextToAudioRequest};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-tts")
}

fn synthesize(request: &TextToAudioRequest) -> SynthesizedAudio {
    let dir = fixture();
    let mut engine = PipelineEngine::from_dir_with_config(&dir, EngineConfig::default())
        .expect("the tiny TTS fixture must load as a pipeline");
    let tokenizer =
        Tokenizer::from_file(dir.join("tokenizer.json")).expect("the fixture ships a tokenizer");
    text_to_audio::synthesize(&mut engine, &tokenizer, request).expect("synthesis must succeed")
}

fn base_request() -> TextToAudioRequest {
    TextToAudioRequest {
        text: "hello".to_string(),
        max_new_tokens: Some(4),
        ..TextToAudioRequest::default()
    }
}

#[test]
fn synthesizes_the_declared_waveform_at_the_declared_sample_rate() {
    let audio = synthesize(&base_request());

    assert_eq!(audio.sample_rate, 16_000, "the declared rate must be used");
    assert_eq!(audio.channels, 1);
    // Closed form from the fixture: each code scaled by 2 and repeated twice.
    assert_eq!(audio.samples, vec![0.0, 0.0, 2.0, 2.0, 4.0, 4.0, 6.0, 6.0]);
    assert!((audio.duration_secs() - 8.0 / 16_000.0).abs() < 1e-9);
}

#[test]
fn the_waveform_encodes_to_a_readable_wav() {
    let audio = synthesize(&base_request());

    let wav = audio.to_wav().expect("encoding must succeed");
    assert_eq!(&wav[..4], b"RIFF");

    let decoded = onnx_genai::preprocess::audio::decode_wav_pcm16(&wav)
        .expect("the encoding must be readable PCM16");
    assert_eq!(decoded.sample_rate, 16_000);
    assert_eq!(decoded.samples.len(), 8);
    // The fixture's samples sit far outside [-1, 1], so PCM encoding clamps
    // them to full scale. That is the documented behavior, and the renderer
    // warns about it rather than failing.
    assert!(audio.peak() > 1.5);
    assert!(decoded.samples.iter().all(|sample| sample.abs() <= 1.0));
}

#[test]
fn raw_pcm_carries_two_bytes_per_sample_and_no_container() {
    let audio = synthesize(&base_request());

    let pcm = audio.to_pcm16();

    assert_eq!(pcm.len(), audio.samples.len() * 2);
    assert_ne!(&pcm[..4], b"RIFF");
}

#[test]
fn a_shorter_budget_produces_a_shorter_waveform() {
    let short = synthesize(&TextToAudioRequest {
        max_new_tokens: Some(2),
        ..base_request()
    });
    let long = synthesize(&base_request());

    assert!(
        short.samples.len() < long.samples.len(),
        "{} vs {}",
        short.samples.len(),
        long.samples.len()
    );
}

#[test]
fn an_explicit_sample_rate_overrides_the_declared_one() {
    let audio = synthesize(&TextToAudioRequest {
        sample_rate: Some(22_050),
        ..base_request()
    });

    assert_eq!(audio.sample_rate, 22_050);
}

#[test]
fn a_package_with_no_waveform_stage_is_rejected() {
    // The diffusion fixture has a final stage but no autoregressive decoder, so
    // it is not a speech package.
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-txt2img");
    let mut engine = PipelineEngine::from_dir_with_config(&dir, EngineConfig::default())
        .expect("the diffusion fixture loads");
    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");

    let error = text_to_audio::synthesize(&mut engine, &tokenizer, &base_request())
        .expect_err("a diffusion package must not be synthesized to audio");

    let message = error.to_string();
    assert!(message.contains("What:"), "message: {message}");
    assert!(message.contains("How:"), "message: {message}");
}

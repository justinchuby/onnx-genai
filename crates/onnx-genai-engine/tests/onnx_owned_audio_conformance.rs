//! ONNX-owned, self-contained audio conformance.
//!
//! These tests give onnx-genai its own nested-audio execution and buffered
//! PCM16 WAV audio conformance using only fixtures checked into this
//! repository. They deliberately do not depend on any producer-supplied
//! package (such as a `hierarchical_audio` fixture), so they never skip: the
//! authoritative runtime is exercised regardless of what an external producer
//! chooses to sync into `onnx_genai_workflow_conformance.rs`.
//!
//! The runtime consumes canonical metadata fields generically here. The speech
//! package is driven purely through its declared `prompt_tokens` input and its
//! `audio` output's `media` contract; nothing keys off a model family.

use onnx_genai_engine::{Engine, EngineConfig};
use onnx_genai_metadata::{WorkflowStep, parse_metadata};
use std::path::PathBuf;

#[path = "common/hermetic_workflows.rs"]
mod hermetic_workflows;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/onnx_genai_workflows")
        .join(name)
}

fn contains_nested_loop(steps: &[WorkflowStep], inside_loop: bool) -> bool {
    steps.iter().any(|step| match step {
        WorkflowStep::Loop { setup, steps, .. } => {
            inside_loop || contains_nested_loop(setup, true) || contains_nested_loop(steps, true)
        }
        WorkflowStep::Branch { cases, default, .. } => {
            cases
                .values()
                .any(|case| contains_nested_loop(std::slice::from_ref(case), inside_loop))
                || default.as_ref().is_some_and(|default| {
                    contains_nested_loop(std::slice::from_ref(default.as_ref()), inside_loop)
                })
        }
        _ => false,
    })
}

/// Read a little-endian `u16` from a canonical PCM WAV header offset.
fn header_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

/// Read a little-endian `u32` from a canonical PCM WAV header offset.
fn header_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

/// The tiny in-tree speech package executes and its audio output resamples and
/// PCM16-encodes into a valid buffered WAV that honours the declared `media`
/// contract (24 kHz, two channels, 16-bit).
#[test]
fn onnx_owned_speech_workflow_encodes_buffered_pcm16_wav() -> anyhow::Result<()> {
    let mut engine = Engine::from_dir(&fixture("speech_wav"), EngineConfig::default())?;
    let outputs =
        engine.run_pipeline_outputs(hermetic_workflows::speech_request(&[2, 6, 7, 8, 9, 3], 8)?)?;

    // The pre-adapter audio output is planar [1, channels, samples] float32.
    let audio = &outputs["audio"];
    assert_eq!(audio.shape()[..2], [1, 2]);
    assert!(audio.shape()[2] > 0);
    assert!(audio.to_vec_f32()?.iter().all(|sample| sample.is_finite()));

    // The engine encodes the declared media contract into a canonical WAV.
    let encoded = engine.encode_audio_output(&outputs, "audio")?;
    assert_eq!(encoded.content_type, "audio/wav");
    assert_eq!(encoded.sample_rate_hz, 24000);
    assert_eq!(encoded.channels, 2);

    let wav = &encoded.bytes;
    assert!(wav.len() > 44, "WAV must carry a header plus PCM samples");
    assert_eq!(&wav[0..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    assert_eq!(&wav[12..16], b"fmt ");
    assert_eq!(header_u16(wav, 20), 1, "PCM format tag");
    assert_eq!(header_u16(wav, 22), 2, "channel count from media contract");
    assert_eq!(
        header_u32(wav, 24),
        24000,
        "sample rate from media contract"
    );
    assert_eq!(header_u16(wav, 34), 16, "pcm_s16_le is 16-bit");
    // The 48 kHz -> 24 kHz resample halves the 64 emitted samples per channel.
    let data_bytes = header_u32(wav, 40) as usize;
    assert_eq!(
        data_bytes,
        32 * 2 * 2,
        "resampled stereo 16-bit sample bytes"
    );

    // Encoding binds to the exact declared output name, not "the first audio
    // output": an unknown or non-audio name fails closed.
    assert!(
        engine.encode_audio_output(&outputs, "waveform").is_err(),
        "an undeclared output name must not be encoded"
    );
    Ok(())
}

/// The in-tree nested TTS package (nested control flow with a carried,
/// autoregressive frame loop) executes on the authoritative runtime and emits a
/// finite waveform. This owns nested-audio execution conformance without any
/// producer-supplied fixture.
#[test]
fn onnx_owned_nested_audio_workflow_executes_nested_generation() -> anyhow::Result<()> {
    let package = fixture("tts");
    let metadata_source = std::fs::read_to_string(package.join("inference_metadata.yaml"))?;
    let metadata = parse_metadata(&metadata_source, Some("yaml"))?;
    let workflow = &metadata
        .pipeline
        .as_ref()
        .expect("TTS fixture declares a workflow")
        .workflow;
    assert!(
        contains_nested_loop(&workflow.steps, false),
        "the nested-audio package must structurally contain a nested generation loop"
    );

    let mut engine = Engine::from_dir(&package, EngineConfig::default())?;
    let outputs = engine.run_pipeline_outputs(hermetic_workflows::tts_request(&[1, 2], 1)?)?;
    let waveform = &outputs["waveform"];
    assert_eq!(waveform.shape()[..2], [1, 1]);
    let samples = waveform.to_vec_f32()?;
    assert!(!samples.is_empty());
    assert!(samples.iter().all(|sample| sample.is_finite()));
    Ok(())
}

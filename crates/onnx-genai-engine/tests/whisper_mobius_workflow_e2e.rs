//! End-to-end execution of a natively exported Whisper package whose audio
//! preprocessing, encoder, and decode loop are all declared in workflow metadata.
//!
//! Unlike the imported-`genai_config` harness, this package carries a declared
//! `preprocessing.audio` program, so the workflow input is *encoded audio bytes*
//! rather than a precomputed feature tensor. A passing run therefore proves the
//! whole chain is expressible and executable as metadata:
//!
//! 1. the audio adapter decodes, resamples, windows, and log-mels the clip,
//! 2. the encoder runs once in the loop prologue,
//! 3. its hidden states become loop-invariant request-aligned cross state,
//! 4. the decoder autoregressively consumes them beside its own growing cache,
//! 5. each row terminates on its own end-of-transcript token.
//!
//! ```sh
//! export LD_LIBRARY_PATH=$(find target -type d -path '*onnx-genai-ort-sys*/out/ort-prebuilt/lib' | head -1):$LD_LIBRARY_PATH
//! MOBIUS_WHISPER_DIR=/abs/path/to/mobius-whisper-tiny \
//! WHISPER_WAV=/abs/path/to/audio_16k.wav \
//! WHISPER_WAV_SHORT=/abs/path/to/audio_short_16k.wav \
//!   cargo test -p onnx-genai-engine --test whisper_mobius_workflow_e2e -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use onnx_genai_engine::pipeline::{PipelineEngine, PipelineGenerateRequest, WorkflowOutputRole};
use onnx_genai_engine::{Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest};
use onnx_genai_ort::{DataType, Value};
use onnx_genai_preprocess::audio::{LogMelExtractor, WHISPER_SAMPLE_RATE, decode_wav_pcm16};

/// Whisper multilingual forced decoder prompt:
/// `<|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>`.
const SOT_PROMPT: [i64; 4] = [50258, 50259, 50359, 50363];
const EOS: i64 = 50257;

fn package_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("MOBIUS_WHISPER_DIR")?);
    dir.join("inference_metadata.yaml").is_file().then_some(dir)
}

/// The same export published without a declared audio program, so the encoder
/// feature tensor is a request input. Batch-leading features are what make a
/// genuine B=2 submission expressible; a single encoded-bytes input is not
/// rectangular across clips of different length.
fn features_package_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("MOBIUS_WHISPER_FEATURES_DIR")?);
    dir.join("inference_metadata.yaml").is_file().then_some(dir)
}

fn log_mel(path: &str) -> anyhow::Result<Vec<f32>> {
    let audio = decode_wav_pcm16(&std::fs::read(path)?)?;
    let features = LogMelExtractor::new(80, WHISPER_SAMPLE_RATE)?
        .extract_padded(&audio.samples, audio.sample_rate)?;
    anyhow::ensure!(
        features.shape() == [1, 80, 3000],
        "whisper encoder expects a 30 s padded log-mel window, got {:?}",
        features.shape()
    );
    Ok(features.data)
}

/// Bind the feature-input variant for an arbitrary batch of clips.
fn batched_features_request(
    features: &[f32],
    batch: usize,
    max_new_tokens: usize,
) -> anyhow::Result<PipelineGenerateRequest> {
    let rows = i64::try_from(batch)?;
    let prompt_len = i64::try_from(SOT_PROMPT.len())?;
    let input_ids = SOT_PROMPT
        .repeat(batch)
        .iter()
        .flat_map(|token| token.to_le_bytes())
        .collect::<Vec<u8>>();
    let slot_ids = (0..rows).collect::<Vec<_>>();

    Ok(PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(SOT_PROMPT.iter().map(|token| *token as u32).collect()),
        options: GenerateOptions {
            max_new_tokens,
            temperature: 0.0,
            ..GenerateOptions::default()
        },
    })
    .with_input(
        "request.decoder_input_ids",
        Value::from_raw_bytes(input_ids, &[rows, prompt_len], DataType::Int64)?,
    )
    .with_input(
        "encoder.input.input_features",
        Value::from_slice_f32(features, &[rows, 80, 3000])?,
    )
    .with_input(
        "request.max_iterations",
        Value::from_slice_i64(&[i64::try_from(max_new_tokens)?], &[1])?,
    )
    .with_input("package.eos_ids", Value::from_slice_i64(&[EOS], &[1])?)
    .with_input(
        "package.slot_ids",
        Value::from_slice_i64(&slot_ids, &[rows])?,
    )
    .with_input(
        "request.prompt_lengths",
        Value::from_slice_i64(&vec![prompt_len; batch], &[rows])?,
    )
    .with_input(
        "request.eos_ids",
        Value::from_slice_i64(&vec![EOS; batch], &[rows, 1])?,
    )
    .with_input(
        "request.eos_lengths",
        Value::from_slice_i64(&vec![1_i64; batch], &[rows])?,
    )
    .with_input(
        "request.row_max_iterations",
        Value::from_slice_i64(&vec![-1_i64; batch], &[rows])?,
    )
    .with_input(
        "package.active",
        Value::from_raw_bytes(vec![1; batch], &[rows], DataType::Bool)?,
    )
    .with_input(
        "package.not_done",
        Value::from_raw_bytes(vec![0; batch], &[rows], DataType::Bool)?,
    )
    .with_input(
        "package.one_token",
        Value::from_slice_i64(&vec![1_i64; batch], &[rows])?,
    )
    .with_input(
        "package.cache_lengths",
        Value::from_slice_i64(&vec![0_i64; batch], &[rows])?,
    )
    .with_input(
        "package.zero_batch",
        Value::from_slice_i64(&vec![0_i64; batch], &[rows])?,
    )
    .with_input(
        "request.temperature",
        Value::from_slice_f32(&vec![1.0_f32; batch], &[rows])?,
    )
    .with_input(
        "request.top_k",
        Value::from_slice_i64(&vec![1_i64; batch], &[rows])?,
    )
    .with_input(
        "request.top_p",
        Value::from_slice_f32(&vec![1.0_f32; batch], &[rows])?,
    )
    .with_input(
        "request.min_p",
        Value::from_slice_f32(&vec![0.0_f32; batch], &[rows])?,
    )
    .with_input("request.seed", Value::from_slice_i64(&slot_ids, &[rows])?)
    .with_input(
        "request.rng_counter",
        Value::from_slice_i64(&vec![0_i64; batch], &[rows])?,
    ))
}

/// Bind every request-scoped port the exported workflow declares.
///
/// `request.audio` carries the encoded clip: the package's declared audio
/// program is what turns it into the encoder's feature tensor, so nothing in
/// this harness knows the mel-bin count or the window length.
fn transcribe_request(
    wav_bytes: Vec<u8>,
    max_new_tokens: usize,
) -> anyhow::Result<PipelineGenerateRequest> {
    let prompt_len = i64::try_from(SOT_PROMPT.len())?;
    let encoded_len = i64::try_from(wav_bytes.len())?;
    let input_ids = SOT_PROMPT
        .iter()
        .flat_map(|token| token.to_le_bytes())
        .collect::<Vec<u8>>();

    Ok(PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(SOT_PROMPT.iter().map(|token| *token as u32).collect()),
        options: GenerateOptions {
            max_new_tokens,
            temperature: 0.0,
            ..GenerateOptions::default()
        },
    })
    .with_input(
        "request.decoder_input_ids",
        Value::from_raw_bytes(input_ids, &[1, prompt_len], DataType::Int64)?,
    )
    .with_input(
        "request.audio",
        Value::from_raw_bytes(wav_bytes, &[encoded_len], DataType::Uint8)?,
    )
    .with_input(
        "request.max_iterations",
        Value::from_slice_i64(&[i64::try_from(max_new_tokens)?], &[1])?,
    )
    .with_input("package.eos_ids", Value::from_slice_i64(&[EOS], &[1])?)
    .with_input("package.slot_ids", Value::from_slice_i64(&[0], &[1])?)
    .with_input(
        "request.prompt_lengths",
        Value::from_slice_i64(&[prompt_len], &[1])?,
    )
    .with_input("request.eos_ids", Value::from_slice_i64(&[EOS], &[1, 1])?)
    .with_input("request.eos_lengths", Value::from_slice_i64(&[1], &[1])?)
    .with_input(
        "request.row_max_iterations",
        Value::from_slice_i64(&[-1], &[1])?,
    )
    .with_input(
        "package.active",
        Value::from_raw_bytes(vec![1], &[1], DataType::Bool)?,
    )
    .with_input(
        "package.not_done",
        Value::from_raw_bytes(vec![0], &[1], DataType::Bool)?,
    )
    .with_input("package.one_token", Value::from_slice_i64(&[1], &[1])?)
    .with_input("package.cache_lengths", Value::from_slice_i64(&[0], &[1])?)
    .with_input("package.zero_batch", Value::from_slice_i64(&[0], &[1])?)
    .with_input("request.temperature", Value::from_slice_f32(&[1.0], &[1])?)
    .with_input("request.top_k", Value::from_slice_i64(&[1], &[1])?)
    .with_input("request.top_p", Value::from_slice_f32(&[1.0], &[1])?)
    .with_input("request.min_p", Value::from_slice_f32(&[0.0], &[1])?)
    .with_input("request.seed", Value::from_slice_i64(&[0], &[1])?)
    .with_input("request.rng_counter", Value::from_slice_i64(&[0], &[1])?))
}

fn rows_of(
    engine: &PipelineEngine,
    outputs: &onnx_genai_engine::pipeline::PipelineOutputs,
) -> anyhow::Result<Vec<Vec<i64>>> {
    engine
        .output_rows_for_role(outputs, WorkflowOutputRole::Tokens)
        .into_iter()
        .map(|(_, value)| Ok(value.to_vec_i64()?))
        .collect()
}

#[test]
#[ignore = "requires a Mobius-exported Whisper workflow package (MOBIUS_WHISPER_DIR) and 16 kHz mono PCM audio (WHISPER_WAV)"]
fn exported_whisper_workflow_transcribes_encoded_audio() -> anyhow::Result<()> {
    let (Some(dir), Ok(wav)) = (package_dir(), std::env::var("WHISPER_WAV")) else {
        eprintln!("skipping: set MOBIUS_WHISPER_DIR and WHISPER_WAV to run this harness");
        return Ok(());
    };
    let bytes = std::fs::read(&wav)?;
    let mut engine = Engine::from_pipeline_dir(&dir, EngineConfig::default())?;
    let output = engine.run_pipeline_outputs(transcribe_request(bytes, 64)?)?;
    let tokens = engine
        .structured_output_for_role(&output, WorkflowOutputRole::Tokens)
        .expect("workflow must emit tokens")
        .to_vec_i64()?;
    eprintln!("tokens: {tokens:?}");
    assert!(!tokens.is_empty(), "expected generated tokens");
    assert_eq!(
        tokens.last().copied(),
        Some(EOS),
        "greedy decoding must stop at end-of-transcript"
    );
    assert_eq!(
        tokens.iter().filter(|token| **token == EOS).count(),
        1,
        "decoding must not continue past end-of-transcript"
    );
    Ok(())
}

#[test]
#[ignore = "requires a Mobius-exported Whisper workflow package and two clips of different length"]
fn exported_whisper_workflow_separates_clips_of_different_length() -> anyhow::Result<()> {
    let (Some(dir), Ok(long_wav), Ok(short_wav)) = (
        package_dir(),
        std::env::var("WHISPER_WAV"),
        std::env::var("WHISPER_WAV_SHORT"),
    ) else {
        eprintln!("skipping: set MOBIUS_WHISPER_DIR, WHISPER_WAV and WHISPER_WAV_SHORT");
        return Ok(());
    };
    let long = std::fs::read(&long_wav)?;
    let short = std::fs::read(&short_wav)?;
    let mut engine = Engine::from_pipeline_dir(&dir, EngineConfig::default())?;

    // Both clips pad to the same declared analysis window, so the difference in
    // duration must survive only through the encoder states and the per-row
    // termination — not through tensor shapes.
    let long_output = engine.run_pipeline_outputs(transcribe_request(long, 96)?)?;
    let long_rows = rows_of(&engine, &long_output)?;
    let short_output = engine.run_pipeline_outputs(transcribe_request(short, 96)?)?;
    let short_rows = rows_of(&engine, &short_output)?;

    eprintln!("long  {:?}", long_rows[0]);
    eprintln!("short {:?}", short_rows[0]);
    assert_ne!(
        long_rows[0], short_rows[0],
        "different audio must not collapse to the same transcript"
    );
    assert!(
        long_rows[0].len() > short_rows[0].len(),
        "the longer clip must produce the longer transcript"
    );
    for (label, tokens) in [("long", &long_rows[0]), ("short", &short_rows[0])] {
        assert_eq!(
            tokens.last().copied(),
            Some(EOS),
            "the {label} clip must end at end-of-transcript"
        );
        assert_eq!(
            tokens.iter().filter(|token| **token == EOS).count(),
            1,
            "the {label} clip must not keep decoding past end-of-transcript"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires the feature-input Mobius Whisper package (MOBIUS_WHISPER_FEATURES_DIR) and two clips of different length"]
fn exported_whisper_workflow_keeps_rows_aligned_across_lengths() -> anyhow::Result<()> {
    let (Some(dir), Ok(long_wav), Ok(short_wav)) = (
        features_package_dir(),
        std::env::var("WHISPER_WAV"),
        std::env::var("WHISPER_WAV_SHORT"),
    ) else {
        eprintln!("skipping: set MOBIUS_WHISPER_FEATURES_DIR, WHISPER_WAV and WHISPER_WAV_SHORT");
        return Ok(());
    };
    let long = log_mel(&long_wav)?;
    let short = log_mel(&short_wav)?;
    let mut engine = Engine::from_pipeline_dir(&dir, EngineConfig::default())?;

    let short_alone_output =
        engine.run_pipeline_outputs(batched_features_request(&short, 1, 96)?)?;
    let short_alone = rows_of(&engine, &short_alone_output)?;
    let long_alone_output = engine.run_pipeline_outputs(batched_features_request(&long, 1, 96)?)?;
    let long_alone = rows_of(&engine, &long_alone_output)?;

    // Both clips pad to the same 30 s window, so batching them is what exercises
    // row alignment: each decoder row must keep reading the cross state its own
    // encoder row produced, and each must terminate on its own EOS.
    let mut long_then_short = long.clone();
    long_then_short.extend_from_slice(&short);
    let mut short_then_long = short.clone();
    short_then_long.extend_from_slice(&long);
    let forward_output =
        engine.run_pipeline_outputs(batched_features_request(&long_then_short, 2, 96)?)?;
    let forward = rows_of(&engine, &forward_output)?;
    let reversed_output =
        engine.run_pipeline_outputs(batched_features_request(&short_then_long, 2, 96)?)?;
    let reversed = rows_of(&engine, &reversed_output)?;

    assert_eq!(forward.len(), 2, "one result row per submitted clip");
    assert_eq!(reversed.len(), 2, "one result row per submitted clip");
    eprintln!("long alone   {:?}", long_alone[0]);
    eprintln!("short alone  {:?}", short_alone[0]);
    eprintln!("forward row0 {:?}", forward[0]);
    eprintln!("forward row1 {:?}", forward[1]);

    // A float32 export is numerically stable across batch sizes, so the
    // batched rows must reproduce the standalone runs exactly.
    assert_eq!(
        forward[0], long_alone[0],
        "the long clip's batched row must match its standalone transcript"
    );
    assert_eq!(
        forward[1], short_alone[0],
        "the short clip's batched row must match its standalone transcript"
    );
    // Permuting the submitted clips must permute the results the same way.
    assert_eq!(forward[0], reversed[1], "row permutation must be exact");
    assert_eq!(forward[1], reversed[0], "row permutation must be exact");

    for (row, tokens) in forward.iter().enumerate() {
        assert_eq!(
            tokens.last().copied(),
            Some(EOS),
            "row {row} must end at end-of-transcript"
        );
        assert_eq!(
            tokens.iter().filter(|token| **token == EOS).count(),
            1,
            "row {row} must not keep decoding past end-of-transcript"
        );
    }
    // Early EOS on one row must not truncate the other.
    assert!(
        forward[0].len() > forward[1].len(),
        "the longer clip must keep decoding after the shorter row finishes"
    );
    Ok(())
}

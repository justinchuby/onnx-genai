//! End-to-end execution of an encoder-conditioned ASR package expressed purely
//! as workflow metadata.
//!
//! The package under test is produced by importing an existing
//! `genai_config.json` Whisper export into the native workflow IR: the encoder
//! runs once in the loop prologue, its cross-attention KV outputs become
//! loop-invariant request-aligned state, and the decoder consumes them on every
//! autoregressive step alongside its own growing self-attention cache. A
//! passing run therefore proves three separate things at once: the metadata
//! expresses the encoder-conditioned loop, the runtime executes it, and the
//! cross state stays bound across steps (the decoder graph errors on a missing
//! input otherwise).
//!
//! ```sh
//! export LD_LIBRARY_PATH=$(find target -type d -path '*onnx-genai-ort-sys*/out/ort-prebuilt/lib' | head -1):$LD_LIBRARY_PATH
//! WHISPER_METADATA_DIR=/path/to/imported-package \
//! WHISPER_WAV=/path/to/audio_16k.wav \
//! WHISPER_WAV_SHORT=/path/to/audio_short_16k.wav \
//!   cargo test -p onnx-genai-engine --test whisper_metadata_e2e -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use onnx_genai_engine::pipeline::{PipelineGenerateRequest, WorkflowOutputRole};
use onnx_genai_engine::{Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest};
use onnx_genai_ort::{DataType, Value};
use onnx_genai_preprocess::audio::{LogMelExtractor, WHISPER_SAMPLE_RATE, decode_wav_pcm16};

/// Whisper multilingual forced decoder prompt:
/// `<|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>`.
const SOT_PROMPT: [i64; 4] = [50258, 50259, 50359, 50363];
const EOS: i64 = 50257;

fn package_dir() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("WHISPER_METADATA_DIR")?);
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
    if let Some(dir) = std::env::var_os("WHISPER_FEATURE_DUMP") {
        let stem = std::path::Path::new(path)
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned())
            .unwrap_or_else(|| "features".to_string());
        let bytes = features
            .data
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>();
        std::fs::write(
            std::path::Path::new(&dir).join(format!("{stem}.rust_mel.f32")),
            bytes,
        )?;
    }
    Ok(features.data)
}

/// Bind every request-scoped port the imported workflow declares.
///
/// Rows are batch positions: the package carries no row identity of its own, so
/// the caller's row order is the only association between an audio clip, an
/// encoder state, and a decoder row.
fn transcribe_request(
    features: &[f32],
    batch: usize,
    max_new_tokens: usize,
) -> anyhow::Result<PipelineGenerateRequest> {
    let rows = i64::try_from(batch)?;
    let prompt_len = i64::try_from(SOT_PROMPT.len())?;
    // The decoder graph takes INT32 token ids, so the request port is int32 too.
    let input_ids = SOT_PROMPT
        .repeat(batch)
        .iter()
        .flat_map(|token| (*token as i32).to_le_bytes())
        .collect::<Vec<u8>>();
    let slot_ids = (0..rows).collect::<Vec<_>>();

    Ok(PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(SOT_PROMPT.iter().map(|t| *t as u32).collect()),
        options: GenerateOptions {
            max_new_tokens,
            temperature: 0.0,
            ..GenerateOptions::default()
        },
    })
    .with_input(
        "request.input_ids",
        Value::from_raw_bytes(input_ids, &[rows, prompt_len], DataType::Int32)?,
    )
    .with_input(
        "encoder.input.audio_features",
        Value::from_slice_f32(features, &[rows, 80, 3000])?,
    )
    .with_input(
        "request.max_iterations",
        Value::from_slice_i64(&[i64::try_from(max_new_tokens)?], &[1])?,
    )
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

/// Read the emitted token rows, one vector per submitted request row.
fn rows_of(
    engine: &Engine,
    outputs: &onnx_genai_engine::pipeline::PipelineOutputs,
) -> anyhow::Result<Vec<Vec<i64>>> {
    engine
        .output_rows_for_role(outputs, WorkflowOutputRole::Tokens)
        .into_iter()
        .map(|(_, value)| Ok(value.to_vec_i64()?))
        .collect()
}

#[test]
#[ignore = "requires an imported Whisper workflow package (WHISPER_METADATA_DIR) and 16 kHz mono PCM audio (WHISPER_WAV)"]
fn imported_whisper_workflow_transcribes_audio() -> anyhow::Result<()> {
    let (Some(dir), Ok(wav)) = (package_dir(), std::env::var("WHISPER_WAV")) else {
        eprintln!("skipping: set WHISPER_METADATA_DIR and WHISPER_WAV to run this harness");
        return Ok(());
    };
    let features = log_mel(&wav)?;
    let mut engine = Engine::from_pipeline_dir(&dir, EngineConfig::default())?;
    let output = engine.run_pipeline_outputs(transcribe_request(&features, 1, 64)?)?;
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
    Ok(())
}

#[test]
#[ignore = "requires an imported Whisper workflow package and two clips of different length"]
fn imported_whisper_workflow_keeps_rows_aligned_across_lengths() -> anyhow::Result<()> {
    let (Some(dir), Ok(long_wav), Ok(short_wav)) = (
        package_dir(),
        std::env::var("WHISPER_WAV"),
        std::env::var("WHISPER_WAV_SHORT"),
    ) else {
        eprintln!("skipping: set WHISPER_METADATA_DIR, WHISPER_WAV and WHISPER_WAV_SHORT");
        return Ok(());
    };
    let long = log_mel(&long_wav)?;
    let short = log_mel(&short_wav)?;
    let mut engine = Engine::from_pipeline_dir(&dir, EngineConfig::default())?;

    let short_output = engine.run_pipeline_outputs(transcribe_request(&short, 1, 64)?)?;
    let short_alone = rows_of(&engine, &short_output)?;

    // Both clips pad to the same 30 s window but transcribe to different
    // lengths, so batching them is what exercises row alignment: each decoder
    // row must keep reading the cross state its own encoder row produced, and
    // each must terminate on its own end-of-transcript token.
    let mut long_then_short = long.clone();
    long_then_short.extend_from_slice(&short);
    let mut short_then_long = short.clone();
    short_then_long.extend_from_slice(&long);
    let forward_output =
        engine.run_pipeline_outputs(transcribe_request(&long_then_short, 2, 64)?)?;
    let forward = rows_of(&engine, &forward_output)?;
    let reversed_output =
        engine.run_pipeline_outputs(transcribe_request(&short_then_long, 2, 64)?)?;
    let reversed = rows_of(&engine, &reversed_output)?;

    assert_eq!(forward.len(), 2, "one result row per submitted clip");
    assert_eq!(reversed.len(), 2, "one result row per submitted clip");
    eprintln!("forward row0 {:?}", forward[0]);
    eprintln!("forward row1 {:?}", forward[1]);
    eprintln!("reversed row0 {:?}", reversed[0]);
    eprintln!("reversed row1 {:?}", reversed[1]);

    // Permuting the submitted clips must permute the results the same way.
    // This is the alignment claim itself, and unlike a comparison against a
    // differently sized batch it does not depend on the quantized graph
    // producing bit-identical logits at another batch size.
    assert_eq!(
        forward[0], reversed[1],
        "the long clip must transcribe the same whichever row it occupies"
    );
    assert_eq!(
        forward[1], reversed[0],
        "the short clip must transcribe the same whichever row it occupies"
    );
    assert_eq!(
        forward[1], short_alone[0],
        "the short clip's row must not be perturbed by the row batched beside it"
    );
    assert_ne!(
        forward[0], forward[1],
        "different audio must not collapse to the same transcript"
    );

    // Per-row termination: each row stops at its own end-of-transcript token,
    // and the shorter clip stops first without truncating the longer one.
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
    assert!(
        forward[0].len() > forward[1].len(),
        "the longer clip must produce the longer transcript"
    );
    Ok(())
}

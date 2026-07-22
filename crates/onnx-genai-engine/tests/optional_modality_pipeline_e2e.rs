use std::fs;
use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest, PipelineGenerateRequest,
};
use onnx_genai_ort::{DataType, Value};

fn fixture_root(name: &str) -> anyhow::Result<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/test-fixtures")
        .join(name);
    fs::create_dir_all(&root)?;
    Ok(root)
}

fn autoregressive_fixture(name: &str, metadata: &str) -> anyhow::Result<PathBuf> {
    let source =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-vlm-multibinding");
    let root = fixture_root(name)?;
    fs::copy(
        source.join("decoder.onnx.textproto"),
        root.join("decoder.onnx.textproto"),
    )?;
    fs::copy(source.join("tokenizer.json"), root.join("tokenizer.json"))?;

    let original = fs::read_to_string(source.join("embedding.onnx.textproto"))?;
    let audio_encoder = r#"ir_version: 8
graph {
  node {
    input: "audio_features"
    output: "encoded"
    op_type: "Identity"
  }
  name: "optional_audio_encoder"
  input {
    name: "audio_features"
    type {
      tensor_type {
        elem_type: 10
        shape {
          dim { dim_param: "num_audio_tokens" }
          dim { dim_value: 8 }
        }
      }
    }
  }
  output {
    name: "encoded"
    type {
      tensor_type {
        elem_type: 10
        shape {
          dim { dim_param: "num_audio_tokens" }
          dim { dim_value: 8 }
        }
      }
    }
  }
}
opset_import { domain: "" version: 13 }
"#;
    fs::write(root.join("audio_encoder.onnx.textproto"), audio_encoder)?;
    let marker = "  output {\n    name: \"inputs_embeds\"";
    let audio_input = "\
  input {
    name: \"audio_features\"
    type {
      tensor_type {
        elem_type: 10
        shape {
          dim { dim_param: \"num_audio_tokens\" }
          dim { dim_value: 8 }
        }
      }
    }
  }
  output {
    name: \"inputs_embeds\"";
    let embedding = original.replacen(marker, audio_input, 1);
    assert_ne!(embedding, original, "embedding fixture marker must match");
    fs::write(root.join("embedding.onnx.textproto"), embedding)?;
    fs::write(root.join("inference_metadata.yaml"), metadata)?;
    Ok(root)
}

fn generate_request() -> PipelineGenerateRequest {
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![1, 4]));
    request.options = GenerateOptions {
        max_new_tokens: 2,
        temperature: 0.0,
        stop_on_eos: false,
        ..GenerateOptions::default()
    };
    PipelineGenerateRequest::new(request)
}

fn autoregressive_metadata(optional: bool, routed_producer: bool) -> String {
    let optional_input = if optional {
        r#"        optional_inputs:
          audio_features:
            presence: audio
            absent:
              kind: zeros
              shape: [0, 8]
"#
    } else {
        ""
    };
    let producer_model = if routed_producer {
        r#"    audio_encoder:
      filename: audio_encoder.onnx.textproto
      type: encoder
"#
    } else {
        ""
    };
    let producer_edge = if routed_producer {
        r#"    - from: audio_encoder.encoded
      to: embedding.audio_features
"#
    } else {
        ""
    };
    let producer_phase = if routed_producer {
        r#"    audio_encoder:
      run_on: prompt_only
      when_present: audio
"#
    } else {
        ""
    };
    format!(
        r#"pipeline:
  models:
{producer_model}    embedding:
      filename: embedding.onnx.textproto
      type: encoder
      io:
        token_input: input_ids
{optional_input}    decoder:
      filename: decoder.onnx.textproto
      type: decoder
      tokenizer: tokenizer.json
  dataflow:
{producer_edge}    - from: embedding.inputs_embeds
      to: decoder.inputs_embeds
    - from: embedding.aux
      to: decoder.aux
  strategy:
    kind: autoregressive
    decoder: decoder
    max_tokens: 2
  phases:
{producer_phase}    embedding:
      run_on: every_step
    decoder:
      run_on: every_step
"#
    )
}

#[test]
fn image_only_skips_audio_producer_and_reuses_fp16_empty_fallback() -> anyhow::Result<()> {
    let metadata = autoregressive_metadata(true, true);
    let dir = autoregressive_fixture("optional-audio-absent", &metadata)?;
    let mut engine = Engine::from_pipeline_dir(&dir, EngineConfig::default())?;

    let result = engine.synthesize(generate_request())?;

    assert_eq!(
        result.generation.token_ids.len(),
        2,
        "prefill and decode complete"
    );
    assert!(
        !result.tensors.contains_key("audio_encoder.encoded"),
        "presence-gated producer must not run"
    );
    let fallback = result
        .tensors
        .get("embedding.audio_features")
        .expect("absent destination fallback remains cached in the pool");
    assert_eq!(fallback.dtype(), DataType::Float16);
    assert_eq!(fallback.shape(), &[0, 8]);
    assert_eq!(fallback.numel(), 0);
    assert!(
        result.tensors.contains_key("embedding.inputs_embeds"),
        "every-step consumer ran successfully with the cached fallback"
    );
    Ok(())
}

#[test]
fn present_audio_uses_direct_rank2_features_without_fallback() -> anyhow::Result<()> {
    let metadata = autoregressive_metadata(true, true);
    let dir = autoregressive_fixture("optional-audio-present", &metadata)?;
    let mut engine = Engine::from_pipeline_dir(&dir, EngineConfig::default())?;
    let audio = vec![half::f16::from_f32(1.0).to_bits(); 16];
    let request = generate_request()
        .with_presence("audio")
        .with_input(
            "audio_encoder.audio_features",
            Value::from_slice_f16_bits(&audio, &[2, 8])?,
        )
        .with_input(
            "embedding.audio_features",
            Value::from_slice_f16_bits(&audio, &[2, 8])?,
        );

    let result = engine.synthesize(request)?;

    assert_eq!(result.generation.token_ids.len(), 2);
    assert!(
        result.tensors.contains_key("audio_encoder.encoded"),
        "guarded producer runs when its presence key is present"
    );
    let supplied = result.tensors.get("embedding.audio_features").unwrap();
    assert_eq!(supplied.shape(), &[2, 8]);
    assert!(
        supplied
            .to_vec_f32_lossy()?
            .iter()
            .all(|value| *value == 1.0)
    );
    Ok(())
}

#[test]
fn present_audio_without_active_path_fails_before_component_run() -> anyhow::Result<()> {
    let metadata = autoregressive_metadata(true, false);
    let dir = autoregressive_fixture("optional-audio-present-missing", &metadata)?;
    let mut engine = Engine::from_pipeline_dir(&dir, EngineConfig::default())?;

    let error = engine
        .synthesize(generate_request().with_presence("audio"))
        .err()
        .expect("present optional input without a source must fail");

    assert!(
        error
            .to_string()
            .contains("missing optional-but-present pipeline input 'embedding.audio_features'"),
        "unexpected error: {error:#}"
    );
    Ok(())
}

#[test]
fn absent_audio_rejects_caller_supplied_associated_tensor() -> anyhow::Result<()> {
    let metadata = autoregressive_metadata(true, false);
    let dir = autoregressive_fixture("optional-audio-absent-supplied", &metadata)?;
    let mut engine = Engine::from_pipeline_dir(&dir, EngineConfig::default())?;
    let request = generate_request().with_input(
        "embedding.audio_features",
        Value::from_slice_f16_bits(&[0; 8], &[1, 8])?,
    );

    let error = engine
        .synthesize(request)
        .err()
        .expect("caller data for an absent key must be rejected");

    assert!(
        error.to_string().contains(
            "pipeline input 'embedding.audio_features' is associated with presence key 'audio' \
             but that key was declared absent"
        ),
        "unexpected error: {error:#}"
    );
    Ok(())
}

#[test]
fn unresolved_fallback_shape_symbol_is_reported_before_execution() -> anyhow::Result<()> {
    let metadata =
        autoregressive_metadata(true, false).replace("shape: [0, 8]", "shape: [audio_len, 8]");
    let dir = autoregressive_fixture("optional-audio-unresolved-symbol", &metadata)?;
    let mut engine = Engine::from_pipeline_dir(&dir, EngineConfig::default())?;

    let error = engine
        .synthesize(generate_request())
        .err()
        .expect("unresolved fallback symbol must fail");

    assert!(
        error.to_string().contains(
            "unresolved fallback shape symbol 'audio_len' for optional pipeline input \
             'embedding.audio_features'"
        ),
        "unexpected error: {error:#}"
    );
    Ok(())
}

#[test]
fn undeclared_required_audio_input_never_receives_a_fallback() -> anyhow::Result<()> {
    let metadata = autoregressive_metadata(false, false);
    let dir = autoregressive_fixture("required-audio-missing", &metadata)?;
    let mut engine = Engine::from_pipeline_dir(&dir, EngineConfig::default())?;

    let error = engine
        .synthesize(generate_request())
        .err()
        .expect("required graph input must remain required");

    assert!(
        error
            .to_string()
            .contains("missing required pipeline input 'embedding.audio_features'"),
        "unexpected error: {error:#}"
    );
    Ok(())
}

fn prompt_only_fixture(name: &str, optional: bool) -> anyhow::Result<PathBuf> {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-codec");
    let root = fixture_root(name)?;
    fs::copy(
        source.join("encoder.onnx.textproto"),
        root.join("encoder.onnx.textproto"),
    )?;
    fs::copy(
        source.join("vocoder.onnx.textproto"),
        root.join("vocoder.onnx.textproto"),
    )?;
    let optional_input = if optional {
        r#"      io:
        optional_inputs:
          waveform:
            presence: audio
            absent:
              kind: zeros
              shape: [1, 16]
"#
    } else {
        ""
    };
    let metadata = format!(
        r#"pipeline:
  models:
    encoder:
      filename: encoder.onnx.textproto
      type: audio_encoder
{optional_input}    vocoder:
      filename: vocoder.onnx.textproto
      type: vocoder
  dataflow:
    - from: encoder.codes
      to: vocoder.codes
  strategy:
    kind: single_pass
    model: vocoder
  phases:
    encoder:
      run_on: prompt_only
"#
    );
    fs::write(root.join("inference_metadata.yaml"), metadata)?;
    Ok(root)
}

#[test]
fn prompt_only_consumer_receives_zero_fallback() -> anyhow::Result<()> {
    let dir = prompt_only_fixture("optional-prompt-only", true)?;
    let mut engine = Engine::from_pipeline_dir(&dir, EngineConfig::default())?;
    let request =
        PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![])));

    let tensors = engine.run_pipeline(request)?;

    let fallback = tensors.get("encoder.waveform").expect("fallback cached");
    assert_eq!(fallback.shape(), &[1, 16]);
    assert!(fallback.to_vec_f32()?.iter().all(|value| *value == 0.0));
    assert!(
        tensors
            .get("encoder.codes")
            .expect("prompt component ran")
            .to_vec_f32()?
            .iter()
            .all(|value| *value == 0.0)
    );
    assert!(tensors.contains_key("vocoder.audio"));
    Ok(())
}

#[test]
fn prompt_only_required_input_still_errors() -> anyhow::Result<()> {
    let dir = prompt_only_fixture("required-prompt-only", false)?;
    let mut engine = Engine::from_pipeline_dir(&dir, EngineConfig::default())?;
    let request =
        PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![])));

    let error = engine
        .run_pipeline(request)
        .err()
        .expect("required prompt-only input must not be synthesized");

    assert!(
        error
            .to_string()
            .contains("missing required pipeline input 'encoder.waveform'"),
        "unexpected error: {error:#}"
    );
    Ok(())
}

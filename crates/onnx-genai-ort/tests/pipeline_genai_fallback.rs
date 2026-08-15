use std::path::{Path, PathBuf};

use onnx_genai_ort::PipelineModelDirectory;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../onnx-genai-genai-config/tests/fixtures")
        .join(name)
}

#[test]
fn complete_genai_metadata_still_rejects_non_executable_edge_rank() {
    let error = PipelineModelDirectory::load(fixture("vlm-complete"))
        .expect_err("rank-mismatched compatibility package must fail admission")
        .to_string();

    assert!(error.contains("embedding.image_features"), "{error}");
    assert!(error.contains("incompatible ranks"), "{error}");
    assert!(
        error.contains("producer rank 2, consumer rank 3"),
        "{error}"
    );
    assert!(error.contains("regenerate the native sidecar"), "{error}");
}

#[test]
fn incomplete_genai_package_fails_with_regeneration_guidance() {
    let error = PipelineModelDirectory::load(fixture("vlm-incomplete"))
        .expect_err("incomplete compatibility package must fail")
        .to_string();

    assert!(error.contains("missing required semantics"));
    assert!(error.contains("mrope_section"));
    assert!(error.contains("Why:"));
    assert!(error.contains("never guesses from model.type"));
    assert!(error.contains("How to fix:"));
    assert!(error.contains("native inference_metadata.json"));
}

#[test]
fn real_foundry_package_fails_loudly_when_preprocessing_is_not_executable() {
    let model_dir =
        Path::new("/home/justinchu/.foundry/cache/models/Microsoft/qwen3.5-9b-generic-cpu-2/v2");
    if !model_dir.is_dir() {
        eprintln!("real Foundry Qwen3.5 package is not installed; validation is deferred");
        return;
    }

    let error = PipelineModelDirectory::load(model_dir)
        .expect_err("unsupported smart resize must not load")
        .to_string();
    assert!(error.contains("smart_resize=false"));
    assert!(error.contains("Why:"));
    assert!(error.contains("How to fix:"));
}

#[test]
fn real_foundry_whisper_encoder_decoder_package_loads_as_pipeline() {
    // The Foundry Whisper caches are ORT-genai encoder-decoder packages (separate
    // encoder/decoder ONNX + genai_config.json, no native inference_metadata).
    // The compat loader must recognize the encoder-decoder shape and synthesize a
    // valid pipeline spec purely from the declared config + authoritative ONNX
    // graph ports.
    let model_dir = Path::new(
        "/home/justinchu/.foundry/cache/models/Microsoft/openai-whisper-tiny-generic-cpu-4/v4",
    );
    if !model_dir.is_dir() {
        eprintln!("real Foundry Whisper tiny package is not installed; validation is deferred");
        return;
    }

    // The package structurally declares a pipeline (encoder + decoder).
    assert!(
        PipelineModelDirectory::load_if_declared(model_dir)
            .expect("encoder-decoder package is a recognized pipeline")
            .is_some()
    );

    let directory = PipelineModelDirectory::load(model_dir)
        .expect("encoder-decoder compatibility package must load as a pipeline");
    let spec = &directory.spec;

    // Encoder + decoder components resolve to the real ONNX files.
    assert_eq!(spec.models.len(), 2);
    assert_eq!(spec.models["encoder"].role, "encoder");
    assert_eq!(spec.models["decoder"].role, "decoder");
    assert!(
        directory.model_paths["encoder"].ends_with("whisper-tiny_encoder_int8.onnx"),
        "{:?}",
        directory.model_paths["encoder"]
    );
    assert!(
        directory.model_paths["decoder"].ends_with("whisper-tiny_decoder_int8.onnx"),
        "{:?}",
        directory.model_paths["decoder"]
    );

    // Audio prompt input surfaced on the encoder; logits on the decoder.
    let encoder_io = spec.models["encoder"].io.as_ref().expect("encoder io");
    assert_eq!(
        encoder_io.audio_features_input.as_deref(),
        Some("audio_features")
    );
    let decoder_io = spec.models["decoder"].io.as_ref().expect("decoder io");
    assert_eq!(decoder_io.logits_output.as_deref(), Some("logits"));

    // whisper-tiny has 4 decoder layers: self-KV (grows) and cross-KV (static),
    // each with a key and a value port per layer.
    assert_eq!(decoder_io.kv_inputs.as_deref().map(<[_]>::len), Some(8));
    assert_eq!(decoder_io.kv_outputs.as_deref().map(<[_]>::len), Some(8));
    assert_eq!(decoder_io.kv_update.as_deref(), Some("append"));
    assert_eq!(
        decoder_io.cross_kv_inputs.as_deref().map(<[_]>::len),
        Some(8)
    );
    assert_eq!(
        decoder_io.cross_kv_outputs.as_deref().map(<[_]>::len),
        Some(8)
    );

    // Cross-attention KV static routing is declared by the positional pairing of
    // the decoder's cross_kv_inputs (past_*_cross) and cross_kv_outputs (the
    // encoder-produced present_*_cross), computed once by the encoder. It is
    // stateful routing, not per-step dataflow, so no cross edges are emitted.
    let cross_inputs = decoder_io
        .cross_kv_inputs
        .as_deref()
        .expect("cross kv inputs");
    let cross_outputs = decoder_io
        .cross_kv_outputs
        .as_deref()
        .expect("cross kv outputs");
    for layer in 0..4 {
        assert!(
            cross_inputs.contains(&format!("past_key_cross_{layer}"))
                && cross_inputs.contains(&format!("past_value_cross_{layer}")),
            "missing decoder cross-KV input for layer {layer}"
        );
        assert!(
            cross_outputs.contains(&format!("present_key_cross_{layer}"))
                && cross_outputs.contains(&format!("present_value_cross_{layer}")),
            "missing encoder cross-KV output for layer {layer}"
        );
    }
    assert!(
        !spec
            .dataflow
            .iter()
            .any(|edge| edge.from.contains("_cross_")),
        "cross-KV must be stateful routing, not per-step dataflow edges"
    );
}

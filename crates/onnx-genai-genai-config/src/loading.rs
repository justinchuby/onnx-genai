use std::path::{Path, PathBuf};

use onnx_genai_metadata::InferenceMetadata;

use crate::{
    EncoderDecoderGraphInfo, GENAI_CONFIG_FILE, GenAiConfig, GenAiConfigError, ModelGraphInfo,
    ModelShape, PipelineGraphInfo, transducer_unsupported,
};

/// Path to a `genai_config.json` inside `model_dir`, if one exists.
pub fn find_in_dir(model_dir: &Path) -> Option<PathBuf> {
    let path = model_dir.join(GENAI_CONFIG_FILE);
    path.is_file().then_some(path)
}

/// Load and parse a `genai_config.json` from an explicit path.
pub fn load(path: &Path) -> Result<GenAiConfig, GenAiConfigError> {
    let content = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&content)?)
}

/// Best-effort compatibility metadata for a model directory.
///
/// Returns `Ok(None)` when the directory has no `genai_config.json`.
pub fn inference_metadata_from_dir(
    model_dir: &Path,
    kv_native_dtype: Option<&str>,
) -> Result<Option<InferenceMetadata>, GenAiConfigError> {
    let Some(path) = find_in_dir(model_dir) else {
        return Ok(None);
    };
    let config = load(&path)?;
    Ok(Some(config.to_inference_metadata(kv_native_dtype)?))
}

/// Compatibility metadata from an explicitly resolved configuration path.
pub fn inference_metadata_from_path(
    path: &Path,
    kv_native_dtype: Option<&str>,
) -> Result<InferenceMetadata, GenAiConfigError> {
    load(path)?.to_inference_metadata(kv_native_dtype)
}

/// Like [`inference_metadata_from_dir`], but derives the single-decoder KV/state
/// topology from the decoder's actual ONNX graph inventory (`decoder_graph`)
/// rather than expanding KV name patterns over a uniform per-layer count.
///
/// This is the entry point runtime loaders use once the decoder session is
/// available: it lets hybrid SSM/attention decoders load by declaring only the
/// graph's real ports (sparse dense KV plus fixed recurrent `state_pairs`).
pub fn inference_metadata_from_dir_with_graph(
    model_dir: &Path,
    kv_native_dtype: Option<&str>,
    decoder_graph: &ModelGraphInfo,
) -> Result<Option<InferenceMetadata>, GenAiConfigError> {
    let Some(path) = find_in_dir(model_dir) else {
        return Ok(None);
    };
    let config = load(&path)?;
    Ok(Some(config.to_inference_metadata_with_graph(
        kv_native_dtype,
        decoder_graph,
    )?))
}

/// Graph-aware compatibility metadata from an explicitly resolved configuration path.
pub fn inference_metadata_from_path_with_graph(
    path: &Path,
    kv_native_dtype: Option<&str>,
    decoder_graph: &ModelGraphInfo,
) -> Result<InferenceMetadata, GenAiConfigError> {
    load(path)?.to_inference_metadata_with_graph(kv_native_dtype, decoder_graph)
}

/// Strict compatibility conversion for an existing multimodal ORT-GenAI package.
///
/// Unlike [`inference_metadata_from_dir`], this entry point never fills missing
/// VLM semantics from conventions or layer counts. The JSON files provide the
/// semantic contract while `graphs` provides the authoritative ONNX port list,
/// rank, shape, and dtype facts.
pub fn pipeline_inference_metadata_from_dir(
    model_dir: &Path,
    graphs: &PipelineGraphInfo,
) -> Result<Option<InferenceMetadata>, GenAiConfigError> {
    let Some(path) = find_in_dir(model_dir) else {
        return Ok(None);
    };
    let config = load(&path)?;
    if config.shape() != ModelShape::Multimodal {
        return Ok(None);
    }
    match config.to_strict_pipeline_metadata(model_dir, graphs) {
        Ok(metadata) => Ok(Some(metadata)),
        // A split embedding+decoder package whose image preprocessing is not
        // representable cannot drive the vision path, but text decode never
        // touches vision: fall back to the text-only decode pipeline so the
        // package still loads for text generation. This is modality-driven, not
        // model-specific — any image-unusable multimodal package synthesizes
        // text decode the same way.
        Err(GenAiConfigError::UnrepresentablePreprocessing { .. }) => {
            Ok(Some(config.to_strict_text_only_pipeline_metadata(graphs)?))
        }
        Err(error) => Err(error),
    }
}

/// Strict compatibility conversion for an existing encoder-decoder ORT-GenAI
/// package (audio/text sequence-to-sequence, e.g. Whisper).
///
/// Like [`pipeline_inference_metadata_from_dir`], the JSON files provide the
/// semantic contract while `graphs` provides the authoritative ONNX port list,
/// rank, shape, and dtype facts. Nothing is inferred from `model.type` or a
/// model name: the encoder-decoder shape is recognized only from the declared
/// `model.encoder` section, and an RNN-T transducer (which also declares an
/// encoder) is declined with [`GenAiConfigError::UnsupportedPipelineFamily`]
/// rather than mis-bound. Returns `Ok(None)` when the directory has no
/// `genai_config.json` or the config does not describe an encoder-decoder model.
pub fn encoder_decoder_pipeline_inference_metadata_from_dir(
    model_dir: &Path,
    graphs: &EncoderDecoderGraphInfo,
) -> Result<Option<InferenceMetadata>, GenAiConfigError> {
    let Some(path) = find_in_dir(model_dir) else {
        return Ok(None);
    };
    let config = load(&path)?;
    // A transducer also declares `model.encoder`; decline it explicitly with the
    // honest family error rather than returning `Ok(None)` (which would surface a
    // misleading "not an encoder-decoder" fall-through) or fabricating a spec.
    if config.is_transducer() {
        return Err(transducer_unsupported());
    }
    if config.shape() != ModelShape::EncoderDecoder {
        return Ok(None);
    }
    Ok(Some(
        config.to_strict_encoder_decoder_pipeline_metadata(graphs)?,
    ))
}

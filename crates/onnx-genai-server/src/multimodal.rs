//! Public multimodal (image + audio) input plumbing for pipeline models.
//!
//! The OpenAI-compatible server and the `onnx-genai` CLI both need to turn a
//! user-supplied image or audio clip into the tensors a pipeline package
//! declares, and to expand image placeholder tokens in the prompt. That
//! contract is derived entirely from typed metadata (`preprocessing.image`,
//! `pipeline.vision`, and the components' declared graph inputs) — never from a
//! model, vendor, or architecture name.
//!
//! This module is the single home for that derivation so both front ends stay
//! behaviorally identical.

use std::path::{Path, PathBuf};

use anyhow::Context;
use onnx_genai_metadata::PipelineStrategy;
use onnx_genai_ort::{DataType, PipelineModelDirectory, PipelineModels, TensorInfo, Tokenizer};

use crate::audio_input::AudioInputSpec;
use crate::image_input::{VisionInputSpec, VisionOutputBinding, metadata_dtype};

pub use crate::audio_input::{AudioTensor, preprocess_wav};
pub use crate::image_input::{
    ImageBundle, ImageTensor, MAX_EXPANDED_PROMPT_TOKENS, VisionInputSpec as ImageInputSpec,
    image_tensor_value, preprocess_encoded_images,
};

/// Image and audio input contracts declared by one pipeline package.
///
/// Both fields are `None` for a pipeline that declares neither modality.
#[derive(Debug, Clone)]
pub struct MultimodalSpecs {
    pub vision: Option<VisionInputSpec>,
    pub audio: Option<AudioInputSpec>,
}

impl MultimodalSpecs {
    /// Returns true when the package accepts neither image nor audio input.
    pub fn is_empty(&self) -> bool {
        self.vision.is_none() && self.audio.is_none()
    }
}

/// Everything an in-process front end needs to drive a pipeline package's
/// prompt and multimodal inputs.
#[derive(Debug, Clone)]
pub struct PipelineSetup {
    /// Tokenizer used for the pipeline's decoder prompt.
    pub tokenizer_path: PathBuf,
    /// Declared image and audio input contracts.
    pub multimodal: MultimodalSpecs,
}

/// Resolve the prompt tokenizer and multimodal input contracts for `model_dir`,
/// or `None` when the package does not structurally declare a pipeline.
///
/// This opens the pipeline's ONNX components to read their declared input
/// dtypes and shapes, then releases them; callers load the execution engine
/// separately.
pub fn load(model_dir: &Path) -> anyhow::Result<Option<PipelineSetup>> {
    let Some(directory) =
        PipelineModelDirectory::load_if_declared(model_dir).with_context(|| {
            format!(
                "What: the pipeline package at {} could not be inspected. \
             Why: its declared metadata could not be resolved. \
             How: verify inference_metadata.yaml (or genai_config.json) is present and valid.",
                model_dir.display()
            )
        })?
    else {
        return Ok(None);
    };
    let models = PipelineModels::load(model_dir).map_err(|error| {
        anyhow::anyhow!(
            "What: the pipeline components at {} could not be inspected. \
             Why: {error}. \
             How: verify every component file named in pipeline.models exists and is a valid ONNX model.",
            model_dir.display()
        )
    })?;
    let multimodal = build(&directory, &models)?;
    Ok(Some(PipelineSetup {
        tokenizer_path: tokenizer_path(model_dir, &directory)?,
        multimodal,
    }))
}

/// Resolve a pipeline package's prompt tokenizer: the decoder component's own
/// tokenizer when declared, otherwise the package's shared tokenizer.
pub fn tokenizer_path(
    model_dir: &Path,
    directory: &PipelineModelDirectory,
) -> anyhow::Result<PathBuf> {
    directory
        .spec
        .models
        .values()
        .find(|component| component.role == "decoder")
        .and_then(|component| component.tokenizer.as_ref())
        .map(|path| model_dir.join(path))
        .or_else(|| directory.tokenizer_paths.shared.clone())
        .with_context(|| {
            format!(
                "What: the pipeline package at {} has no prompt tokenizer. \
                 Why: no component declares role 'decoder' with a tokenizer, and the package ships no shared tokenizer.json. \
                 How: add tokenizer.json to the package or declare a tokenizer on the decoder component.",
                model_dir.display()
            )
        })
}

/// Build the decoder prompt for an audio (speech-to-text) pipeline.
///
/// The transcription prompt is a token sequence, not user text: the audio
/// features carry the content. Optional `language` selects a language token
/// when the tokenizer declares one.
pub fn audio_decoder_prompt(
    tokenizer: &Tokenizer,
    language: Option<&str>,
) -> anyhow::Result<Vec<u32>> {
    let mut token_ids = vec![
        tokenizer
            .token_id("<|startoftranscript|>")
            .or_else(|| tokenizer.eos_token_id())
            .unwrap_or(0),
    ];
    if let Some(language) = language.filter(|value| !value.is_empty()) {
        let token = format!("<|{}|>", language.to_ascii_lowercase());
        token_ids.push(tokenizer.token_id(&token).with_context(|| {
            format!(
                "What: the requested transcription language was rejected. \
                 Why: this model's tokenizer declares no '{token}' token. \
                 How: omit the language or choose one the model supports."
            )
        })?);
    }
    for token in ["<|transcribe|>", "<|notimestamps|>"] {
        if let Some(token_id) = tokenizer.token_id(token) {
            token_ids.push(token_id);
        }
    }
    Ok(token_ids)
}

/// Derive the image and audio contracts from an already-resolved pipeline
/// directory and its loaded component graphs.
pub fn build(
    directory: &PipelineModelDirectory,
    models: &PipelineModels,
) -> anyhow::Result<MultimodalSpecs> {
    Ok(MultimodalSpecs {
        vision: build_vision(directory, models)?,
        audio: build_audio(directory, models)?,
    })
}

fn build_vision(
    directory: &PipelineModelDirectory,
    models: &PipelineModels,
) -> anyhow::Result<Option<VisionInputSpec>> {
    let Some(program) = directory
        .preprocessing
        .as_ref()
        .and_then(|preprocessing| preprocessing.image.as_ref())
    else {
        return Ok(None);
    };
    let vision = directory.spec.vision.as_ref().context(
        "What: typed image processor binding discovery failed. \
         Why: preprocessing.image is declared, but pipeline.vision has no placeholder expansion contract. \
         How: add typed pipeline.vision metadata before serving image chat requests.",
    )?;
    let mut bindings = Vec::new();
    let mut pixel_shape = None;
    let mut endpoints = std::collections::HashSet::new();
    for output in &program.outputs {
        let resolved = resolve_image_output(models, &output.name)?;
        let Some((endpoint, input)) = resolved else {
            if output.optional.unwrap_or(false) {
                continue;
            }
            anyhow::bail!(
                "What: image processor endpoint '{}' is missing. \
                 Why: preprocessing.image.outputs declares content '{}' dtype '{}' but no ONNX component input matches that metadata endpoint. \
                 How: name the exact component.input endpoint or add the missing graph input operation.",
                output.name,
                output.content,
                output.dtype
            );
        };
        let declared_dtype = metadata_dtype(&output.dtype)?;
        if input.dtype != declared_dtype {
            anyhow::bail!(
                "What: image processor endpoint '{endpoint}' has an incompatible dtype. \
                 Why: typed metadata declares {:?}, but the ONNX input expects {:?} shape {:?}. \
                 How: correct preprocessing.image.outputs '{}' dtype or the graph input type.",
                declared_dtype,
                input.dtype,
                input.shape,
                output.name
            );
        }
        if !endpoints.insert(endpoint.clone()) {
            anyhow::bail!(
                "What: image processor endpoint '{endpoint}' is bound more than once. \
                 Why: multiple preprocessing.image.outputs resolve to the same ONNX input. \
                 How: give every typed output a unique component.input endpoint."
            );
        }
        if output.content == "pixels" && pixel_shape.is_none() {
            pixel_shape = Some(input.shape.clone());
        }
        bindings.push(VisionOutputBinding {
            metadata_name: output.name.clone(),
            endpoint,
            content: output.content.clone(),
            dtype: declared_dtype,
            shape: input.shape.clone(),
        });
    }
    let pixel_shape = pixel_shape.context(
        "What: image processor initialization failed. \
         Why: preprocessing.image.outputs has no resolved content='pixels' endpoint with an expected dtype/shape. \
         How: declare a pixels output bound to the pipeline's primary image tensor input.",
    )?;
    VisionInputSpec::from_program(bindings, &pixel_shape, program, vision).map(Some)
}

fn build_audio(
    directory: &PipelineModelDirectory,
    models: &PipelineModels,
) -> anyhow::Result<Option<AudioInputSpec>> {
    let audio_inputs = models
        .sessions
        .iter()
        .flat_map(|(component, session)| {
            session.inputs().iter().filter_map(move |input| {
                (input.name == "input_features")
                    .then_some((format!("{component}.{}", input.name), input))
            })
        })
        .collect::<Vec<_>>();
    let pipeline_max_tokens = strategy_max_tokens(&directory.spec.strategy);
    match audio_inputs.as_slice() {
        [] => Ok(None),
        [(endpoint, input)] => {
            if input.dtype != DataType::Float32 {
                anyhow::bail!(
                    "audio input '{endpoint}' must be Float32, but the model declares {:?}",
                    input.dtype
                );
            }
            AudioInputSpec::from_input(endpoint.clone(), &input.shape, pipeline_max_tokens)
                .map(Some)
        }
        _ => anyhow::bail!("pipeline declares multiple input_features inputs"),
    }
}

/// Maximum tokens declared by a strategy, searching composite child stages.
pub(crate) fn strategy_max_tokens(strategy: &PipelineStrategy) -> Option<usize> {
    strategy.max_tokens.or_else(|| {
        strategy
            .stages
            .iter()
            .find_map(|stage| strategy_max_tokens(&stage.strategy))
    })
}

fn resolve_image_output<'a>(
    models: &'a PipelineModels,
    metadata_endpoint: &str,
) -> anyhow::Result<Option<(String, &'a TensorInfo)>> {
    if let Some((component, input_name)) = metadata_endpoint.split_once('.')
        && let Some(session) = models.sessions.get(component)
    {
        return Ok(session
            .inputs()
            .iter()
            .find(|input| input.name == input_name)
            .map(|input| (metadata_endpoint.to_string(), input)));
    }

    let matches = models
        .sessions
        .iter()
        .flat_map(|(component, session)| {
            session
                .inputs()
                .iter()
                .filter(move |input| input.name == metadata_endpoint)
                .map(move |input| (format!("{component}.{}", input.name), input))
        })
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [] => Ok(None),
        [(endpoint, input)] => Ok(Some((endpoint.clone(), *input))),
        _ => anyhow::bail!(
            "What: image processor endpoint '{metadata_endpoint}' is ambiguous. \
             Why: {} ONNX component inputs share that unqualified name. \
             How: use an exact component.input name in preprocessing.image.outputs.",
            matches.len()
        ),
    }
}

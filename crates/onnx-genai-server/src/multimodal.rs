//! Generic media binding for workflow package inputs.

use std::path::{Path, PathBuf};

use anyhow::Context;
use onnx_genai_engine::PipelineGenerateRequest;
use onnx_genai_metadata::{RuntimeInputRole, SemanticInputRole};
use onnx_genai_ort::{DataType, PipelineModelDirectory, PipelineModels, Tokenizer, Value};
use onnx_genai_preprocess::audio::WHISPER_N_FRAMES;

pub use crate::audio_input::{AudioInputSpec, AudioTensor, preprocess_samples, preprocess_wav};
pub use crate::image_input::MAX_EXPANDED_PROMPT_TOKENS;

#[derive(Debug, Clone)]
pub struct VisionInputSpec {
    input: String,
}

impl VisionInputSpec {
    pub fn placeholder_token_id(&self) -> Option<u32> {
        None
    }
}

#[derive(Debug, Clone)]
pub struct MultimodalSpecs {
    pub vision: Option<VisionInputSpec>,
    pub audio: Option<AudioInputSpec>,
}

impl MultimodalSpecs {
    pub fn is_empty(&self) -> bool {
        self.vision.is_none() && self.audio.is_none()
    }

    pub fn accepted_modalities(&self) -> String {
        let mut modalities = vec!["text"];
        if self.vision.is_some() {
            modalities.push("image");
        }
        if self.audio.is_some() {
            modalities.push("audio");
        }
        modalities.join(" + ")
    }

    pub fn sole_modality(&self) -> Option<&'static str> {
        match (self.vision.is_some(), self.audio.is_some()) {
            (true, false) => Some("image"),
            (false, true) => Some("audio"),
            _ => None,
        }
    }
}

pub fn admit_attachments(
    specs: Option<&MultimodalSpecs>,
    model: &str,
    images: usize,
    audio: usize,
) -> anyhow::Result<()> {
    if images > 1 {
        anyhow::bail!(
            "workflow package '{model}' accepts one encoded media tensor per request; bind a \
             multi-image application tensor explicitly when the package declares one"
        );
    }
    if audio > 1 || (images > 0 && audio > 0) {
        anyhow::bail!("workflow package '{model}' accepts one media attachment per request");
    }
    if images > 0 && specs.is_none_or(|specs| specs.vision.is_none()) {
        anyhow::bail!("workflow package '{model}' declares no typed image media input");
    }
    if audio > 0 && specs.is_none_or(|specs| specs.audio.is_none()) {
        anyhow::bail!("workflow package '{model}' declares no typed audio media input");
    }
    Ok(())
}

#[derive(Debug)]
pub struct MultimodalInput {
    tensors: Vec<PreparedTensor>,
}

#[derive(Debug)]
enum PreparedTensor {
    EncodedImage {
        name: String,
        bytes: Vec<u8>,
    },
    EncodedAudio {
        name: String,
        bytes: Vec<u8>,
    },
    Audio {
        name: String,
        data: Vec<f32>,
        shape: Vec<i64>,
    },
}

impl MultimodalInput {
    pub fn from_images(
        spec: &VisionInputSpec,
        images: &[Vec<u8>],
        prompt_token_ids: &mut [u32],
        max_prompt_tokens: usize,
    ) -> anyhow::Result<Self> {
        if images.len() != 1 {
            anyhow::bail!("workflow image binding requires exactly one encoded image tensor");
        }
        if prompt_token_ids.len() > max_prompt_tokens {
            anyhow::bail!("prompt exceeds the declared context budget");
        }
        Ok(Self {
            tensors: vec![PreparedTensor::EncodedImage {
                name: spec.input.clone(),
                bytes: images[0].clone(),
            }],
        })
    }

    pub fn from_wav(spec: &AudioInputSpec, bytes: &[u8]) -> anyhow::Result<Self> {
        if spec.encoded {
            return Ok(Self {
                tensors: vec![PreparedTensor::EncodedAudio {
                    name: spec.endpoint.clone(),
                    bytes: bytes.to_vec(),
                }],
            });
        }
        let tensor = preprocess_wav(bytes, spec)?;
        Ok(Self {
            tensors: vec![PreparedTensor::Audio {
                name: tensor.endpoint,
                data: tensor.data,
                shape: tensor.shape,
            }],
        })
    }

    pub fn from_samples(
        spec: &AudioInputSpec,
        samples: &[f32],
        sample_rate: u32,
    ) -> anyhow::Result<Self> {
        if spec.encoded {
            // The package's own program decodes a container, so a caller that
            // holds bare samples has to hand one back.
            let bytes = crate::audio_input::encode_samples_wav(samples, sample_rate)?;
            return Ok(Self {
                tensors: vec![PreparedTensor::EncodedAudio {
                    name: spec.endpoint.clone(),
                    bytes,
                }],
            });
        }
        let tensor = preprocess_samples(samples, sample_rate, spec)?;
        Ok(Self {
            tensors: vec![PreparedTensor::Audio {
                name: tensor.endpoint,
                data: tensor.data,
                shape: tensor.shape,
            }],
        })
    }

    pub fn bind(
        self,
        mut request: PipelineGenerateRequest,
    ) -> anyhow::Result<PipelineGenerateRequest> {
        for tensor in self.tensors {
            let (name, value) = match tensor {
                PreparedTensor::EncodedImage { name, bytes } => {
                    let shape = [i64::try_from(bytes.len()).context("encoded image is too large")?];
                    let value = Value::from_raw_bytes(bytes, &shape, DataType::Uint8)?;
                    (name, value)
                }

                PreparedTensor::EncodedAudio { name, bytes } => {
                    let shape = [i64::try_from(bytes.len()).context("encoded audio is too large")?];
                    let value = Value::from_raw_bytes(bytes, &shape, DataType::Uint8)?;
                    (name, value)
                }

                PreparedTensor::Audio { name, data, shape } => {
                    (name, Value::from_vec_f32(data, &shape)?)
                }
            };
            request = request.with_input(name, value);
        }
        Ok(request)
    }
}

pub fn audio_window_seconds(spec: &AudioInputSpec) -> f32 {
    spec.n_frames as f32 * 160.0 / 16_000.0
}

pub fn expansion_token_budget(
    model_max_context: Option<usize>,
    max_tokens: usize,
) -> anyhow::Result<usize> {
    let limit = model_max_context
        .map(|context| {
            context
                .checked_sub(max_tokens)
                .context("max_tokens exceeds model context")
        })
        .transpose()?
        .unwrap_or(MAX_EXPANDED_PROMPT_TOKENS);
    Ok(limit.min(MAX_EXPANDED_PROMPT_TOKENS))
}

#[derive(Debug, Clone)]
pub struct PipelineSetup {
    pub tokenizer_path: PathBuf,
    pub multimodal: MultimodalSpecs,
    /// Sampling regime the package declares for itself, if any.
    ///
    /// Carried here because the package's own metadata is the only place that
    /// states it; the execution engine is loaded separately and never sees it.
    pub generation_defaults: Option<onnx_genai_metadata::GenerationDefaults>,
}

/// Resolve the prompt tokenizer and multimodal input contracts for `model_dir`,
/// or `None` when the package does not structurally declare a pipeline.
///
/// This reads the pipeline components' declared graph I/O without creating ORT
/// sessions or materializing weights; callers load the execution engine
/// separately with their selected backend.
pub fn load(model_dir: &Path) -> anyhow::Result<Option<PipelineSetup>> {
    let Some(directory) = PipelineModelDirectory::load_if_declared(model_dir)? else {
        return Ok(None);
    };
    let models = PipelineModels::load_with_ort_session_filter(
        model_dir,
        onnx_genai_ort::SessionOptions::default(),
        |_| false,
    )
    .map_err(|error| {
        anyhow::anyhow!(
            "What: the pipeline components at {} could not be inspected. \
             Why: {error}. \
             How: verify every component file named in pipeline.workflow.components exists and is a valid ONNX model.",
            model_dir.display()
        )
    })?;
    Ok(Some(PipelineSetup {
        generation_defaults: directory
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.generation.as_ref())
            .and_then(|generation| generation.defaults.clone()),
        tokenizer_path: tokenizer_path(model_dir, &directory)?,
        multimodal: build(&directory, &models)?,
    }))
}

pub fn tokenizer_path(
    model_dir: &Path,
    directory: &PipelineModelDirectory,
) -> anyhow::Result<PathBuf> {
    directory
        .tokenizer_paths
        .shared
        .clone()
        .or_else(|| {
            let path = model_dir.join("tokenizer.json");
            path.is_file().then_some(path)
        })
        .context("workflow package requires tokenizer.json for text convenience APIs")
}

pub fn audio_decoder_prompt(
    tokenizer: &Tokenizer,
    language: Option<&str>,
) -> anyhow::Result<Vec<u32>> {
    let mut tokens = vec![
        tokenizer
            .token_id("<|startoftranscript|>")
            .or_else(|| tokenizer.eos_token_id())
            .unwrap_or(0),
    ];
    if let Some(language) = language.filter(|value| !value.is_empty()) {
        let token = format!("<|{}|>", language.to_ascii_lowercase());
        tokens.push(
            tokenizer
                .token_id(&token)
                .with_context(|| format!("missing token '{token}'"))?,
        );
    }
    for token in ["<|transcribe|>", "<|notimestamps|>"] {
        if let Some(id) = tokenizer.token_id(token) {
            tokens.push(id);
        }
    }
    Ok(tokens)
}

pub fn build(
    directory: &PipelineModelDirectory,
    _models: &PipelineModels,
) -> anyhow::Result<MultimodalSpecs> {
    let media_inputs = directory
        .spec
        .workflow
        .inputs
        .iter()
        .filter(|(_, input)| {
            matches!(
                &input.role,
                SemanticInputRole::Runtime {
                    role: RuntimeInputRole::Media,
                    ..
                }
            )
        })
        .collect::<Vec<_>>();
    let encoded_media = media_inputs
        .iter()
        .find(|(_, input)| input.contract.dtype == "uint8" && input.contract.rank == 1);
    let vision = if directory
        .preprocessing
        .as_ref()
        .and_then(|p| p.image.as_ref())
        .is_some()
    {
        let (name, _input) = encoded_media
            .context("preprocessing.image requires a uint8 rank-1 media workflow input")?;
        Some(VisionInputSpec {
            input: (*name).clone(),
        })
    } else {
        None
    };
    // An audio program declares its own feature geometry, so the server reads
    // the mel bins and window length off the program's output contract instead
    // of guessing them from a model input it is no longer given.
    let audio = match directory
        .preprocessing
        .as_ref()
        .and_then(|p| p.audio.as_ref())
    {
        Some(program) if vision.is_none() => {
            let (name, _input) = encoded_media
                .context("preprocessing.audio requires a uint8 rank-1 media workflow input")?;
            let features = program
                .outputs
                .iter()
                .find(|output| output.content == "audio_features")
                .context("preprocessing.audio must declare an audio_features output")?;
            let contract = features.contract.as_ref().with_context(|| {
                format!(
                    "preprocessing.audio output '{}' must declare a TensorContract",
                    features.name
                )
            })?;
            if contract.rank != 3 {
                anyhow::bail!(
                    "preprocessing.audio output '{}' must be rank 3 [batch, mel, frames]",
                    features.name
                );
            }
            let dimension = |index: usize| {
                contract
                    .shape
                    .as_ref()
                    .and_then(|shape| shape.get(index))
                    .and_then(|value| match value {
                        onnx_genai_metadata::TensorDimension::Fixed(size) => {
                            usize::try_from(*size).ok()
                        }
                        onnx_genai_metadata::TensorDimension::Symbol(_) => None,
                    })
            };
            let n_mels = dimension(1).context(
                "preprocessing.audio must declare a concrete mel-bin count in its feature contract",
            )?;
            let n_frames = dimension(2).unwrap_or(WHISPER_N_FRAMES);
            Some(AudioInputSpec::from_encoded_input(
                (*name).clone(),
                n_mels,
                n_frames,
                None,
            ))
        }
        _ => None,
    };
    Ok(MultimodalSpecs { vision, audio })
}

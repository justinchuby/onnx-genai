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
use onnx_genai_engine::PipelineGenerateRequest;
use onnx_genai_metadata::PipelineStrategy;
use onnx_genai_ort::{
    DataType, PipelineModelDirectory, PipelineModels, TensorInfo, Tokenizer, Value,
};
use onnx_genai_preprocess::image::packed::ImageExpansionSummary;

use crate::image_input::{VisionOutputBinding, metadata_dtype};

pub use crate::audio_input::{AudioInputSpec, AudioTensor, preprocess_samples, preprocess_wav};
pub use crate::image_input::{
    ImageBundle, ImageTensor, MAX_EXPANDED_PROMPT_TOKENS, VisionInputSpec, image_tensor_value,
    preprocess_encoded_images,
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

    /// Human-readable list of the modalities this package accepts, e.g.
    /// `"text + image"`. Quoted in rejection messages so a caller learns the
    /// model's actual capability from the error alone.
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

    /// The single non-text modality this package declares, if exactly one.
    ///
    /// Lets a front end turn a bare "missing required pipeline input" into
    /// advice about the attachment the caller most likely forgot.
    pub fn sole_modality(&self) -> Option<&'static str> {
        match (self.vision.is_some(), self.audio.is_some()) {
            (true, false) => Some("image"),
            (false, true) => Some("audio"),
            _ => None,
        }
    }
}

/// Reject an attachment set the model cannot consume.
///
/// `specs` is `None` for a single decoder graph, which is text-only by
/// construction. This is the one definition of the admission policy: the CLI
/// and the HTTP API both call it, so a caller gets the same answer and the same
/// wording whichever front end they use.
pub fn admit_attachments(
    specs: Option<&MultimodalSpecs>,
    model: &str,
    images: usize,
    audio: usize,
) -> anyhow::Result<()> {
    if images > 0 && audio > 0 {
        anyhow::bail!(
            "What: image and audio input were combined for {model}. \
             Why: a pipeline consumes one non-text modality per generation, so both cannot be bound at once. \
             How: send them separately."
        );
    }
    if audio > 1 {
        anyhow::bail!(
            "What: {audio} audio clips were rejected for {model}. \
             Why: an audio pipeline transcribes one clip per generation. \
             How: send one clip at a time."
        );
    }
    let accepted = specs.map_or_else(|| "text".to_string(), MultimodalSpecs::accepted_modalities);
    if images > 0 && specs.is_none_or(|specs| specs.vision.is_none()) {
        anyhow::bail!(
            "What: image input was rejected for {model}. \
             Why: it accepts {accepted} input; {}. \
             How: use a vision-language package, or send the prompt as text only.",
            match specs {
                None =>
                    "it is a single decoder graph, not a multi-component pipeline, so it has no image encoder to bind",
                Some(_) =>
                    "its package declares no image preprocessing program (`preprocessing.image` bound to a component input, plus `pipeline.vision`)",
            }
        );
    }
    if audio > 0 && specs.is_none_or(|specs| specs.audio.is_none()) {
        anyhow::bail!(
            "What: audio input was rejected for {model}. \
             Why: it accepts {accepted} input; {}. \
             How: use a speech package, or send the prompt as text only.",
            match specs {
                None =>
                    "it is a single decoder graph, not a multi-component pipeline, so it has no audio encoder to bind",
                Some(_) => "no component of its package declares an `input_features` audio input",
            }
        );
    }
    Ok(())
}

/// Preprocessed, model-ready non-text input for one generation.
///
/// This is the single implementation of "attachments become pipeline inputs".
/// The CLI builds one from local files and binds it immediately; the server
/// builds one from fetched URLs or base64 payloads and sends it to the engine
/// thread, which binds it there. Both get identical preprocessing, identical
/// placeholder expansion, and identical validation.
#[derive(Debug)]
pub struct MultimodalInput {
    tensors: Vec<PreparedTensor>,
    image_summaries: Vec<ImageExpansionSummary>,
    presence_keys: Vec<String>,
}

#[derive(Debug)]
enum PreparedTensor {
    Image(ImageTensor),
    Audio(AudioTensor),
}

impl PreparedTensor {
    fn endpoint(&self) -> &str {
        match self {
            Self::Image(tensor) => &tensor.endpoint,
            Self::Audio(tensor) => &tensor.endpoint,
        }
    }

    fn into_value(self) -> anyhow::Result<Value> {
        match self {
            Self::Image(tensor) => image_tensor_value(tensor),
            Self::Audio(tensor) => {
                Value::from_vec_f32(tensor.data, &tensor.shape).with_context(|| {
                    format!(
                        "What: audio endpoint '{}' tensor construction failed. \
                     Why: the extracted features did not fill the declared shape. \
                     How: report this as an audio preprocessing bug.",
                        tensor.endpoint
                    )
                })
            }
        }
    }
}

impl MultimodalInput {
    /// Preprocess `images` (encoded bytes, in prompt order) and expand every
    /// image placeholder in `prompt_token_ids` into its declared token run.
    ///
    /// `prompt_token_ids` is replaced with the expanded sequence, because the
    /// expansion is what the decoder must actually see.
    pub fn from_images(
        spec: &VisionInputSpec,
        images: &[Vec<u8>],
        prompt_token_ids: &mut Vec<u32>,
        max_prompt_tokens: usize,
    ) -> anyhow::Result<Self> {
        // Check the prompt before decoding pixels: a placeholder mistake is
        // cheap to report and the caller can fix it without touching the files.
        ensure_placeholders(spec, prompt_token_ids, images.len())?;
        let bundle = preprocess_encoded_images(images, spec)?;
        *prompt_token_ids = spec.expand_prompt(prompt_token_ids, &bundle, max_prompt_tokens)?;
        Ok(Self {
            tensors: bundle
                .tensors
                .into_iter()
                .map(PreparedTensor::Image)
                .collect(),
            image_summaries: bundle.images,
            presence_keys: spec.presence_keys().to_vec(),
        })
    }

    /// Preprocess one PCM16 WAV clip into the package's declared audio input.
    pub fn from_wav(spec: &AudioInputSpec, bytes: &[u8]) -> anyhow::Result<Self> {
        Ok(Self {
            tensors: vec![PreparedTensor::Audio(preprocess_wav(bytes, spec)?)],
            image_summaries: Vec::new(),
            presence_keys: Vec::new(),
        })
    }

    /// Preprocess raw `[-1, 1]` mono samples, for callers streaming audio.
    pub fn from_samples(
        spec: &AudioInputSpec,
        samples: &[f32],
        sample_rate: u32,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            tensors: vec![PreparedTensor::Audio(preprocess_samples(
                samples,
                sample_rate,
                spec,
            )?)],
            image_summaries: Vec::new(),
            presence_keys: Vec::new(),
        })
    }

    /// Bind every prepared tensor onto `request`, failing closed on a duplicate
    /// endpoint or an image ordering that no longer matches the prompt.
    pub fn bind(
        self,
        mut request: PipelineGenerateRequest,
    ) -> anyhow::Result<PipelineGenerateRequest> {
        let has_images = !self.image_summaries.is_empty();
        for (prompt_index, summary) in self.image_summaries.iter().enumerate() {
            if summary.image_index != prompt_index {
                anyhow::bail!(
                    "What: pipeline image admission ordering is inconsistent. \
                     Why: prompt image {prompt_index} carries expansion summary index {}. \
                     How: preserve image content parts, tensor packing, and summaries in prompt order.",
                    summary.image_index
                );
            }
        }
        let mut endpoints = std::collections::HashSet::with_capacity(self.tensors.len());
        for tensor in self.tensors {
            let endpoint = tensor.endpoint().to_string();
            if !endpoints.insert(endpoint.clone()) {
                anyhow::bail!(
                    "What: pipeline tensor injection rejected a duplicate endpoint. \
                     Why: '{endpoint}' was supplied more than once. \
                     How: declare each preprocessing output endpoint exactly once."
                );
            }
            request = request.with_input(endpoint, tensor.into_value()?);
        }
        if has_images {
            for key in self.presence_keys {
                request = request.with_presence(key);
            }
        }
        Ok(request)
    }
}

/// Longest audio, in seconds, the model's declared input window accepts.
///
/// Derived from the declared frame count at Whisper's 10 ms hop, so the caller
/// never has to assume a 30-second window.
pub fn audio_window_seconds(spec: &AudioInputSpec) -> f32 {
    spec.n_frames as f32 * onnx_genai_preprocess::audio::WHISPER_HOP_LENGTH as f32
        / onnx_genai_preprocess::audio::WHISPER_SAMPLE_RATE as f32
}

/// Give the prompt one image placeholder per attached image.
///
/// Expansion needs exactly one placeholder token per image, in prompt order.
/// Requiring the caller to type a model's private placeholder spelling (`<image>`,
/// `<|image_pad|>`, …) is a bad contract: they would have to read the package's
/// metadata to write a prompt. So a prompt that positions the placeholders
/// itself is honored, and a prompt that mentions none gets them prepended — the
/// conventional "images, then the question about them" order.
///
/// A partial set is rejected rather than topped up: the caller clearly meant to
/// position them, and guessing where the rest belong would silently change which
/// image a sentence refers to.
fn ensure_placeholders(
    spec: &VisionInputSpec,
    prompt_token_ids: &mut Vec<u32>,
    images: usize,
) -> anyhow::Result<()> {
    let Some(placeholder) = spec.placeholder_token_id() else {
        return Ok(());
    };
    let present = prompt_token_ids
        .iter()
        .filter(|&&token| token == placeholder)
        .count();
    match placeholder_action(placeholder, present, images) {
        PlaceholderAction::Keep => Ok(()),
        PlaceholderAction::Prepend(count) => {
            let mut prompt = vec![placeholder; count];
            prompt.append(prompt_token_ids);
            *prompt_token_ids = prompt;
            Ok(())
        }
        PlaceholderAction::Mismatch => anyhow::bail!(
            "What: the prompt's image placeholders do not match the {images} image(s) supplied. \
             Why: it positions {present} of them, so the images and the text cannot be lined up. \
             How: write exactly one placeholder per image, or none at all to have them prepended in order."
        ),
    }
}

/// What to do about a prompt's image placeholders.
#[derive(Debug, PartialEq, Eq)]
enum PlaceholderAction {
    /// The prompt already positions them; honor it.
    Keep,
    /// The prompt positions none; prepend this many.
    Prepend(usize),
    /// A partial set: refuse rather than guess where the rest belong.
    Mismatch,
}

fn placeholder_action(_placeholder: u32, present: usize, images: usize) -> PlaceholderAction {
    match (present, images) {
        (present, images) if present == images => PlaceholderAction::Keep,
        (0, images) => PlaceholderAction::Prepend(images),
        _ => PlaceholderAction::Mismatch,
    }
}

/// Token budget available to image placeholder expansion.
///
/// Bounded by the model's context minus the tokens the caller reserved for the
/// response, and never above [`MAX_EXPANDED_PROMPT_TOKENS`].
pub fn expansion_token_budget(
    model_max_context: Option<usize>,
    max_tokens: usize,
) -> anyhow::Result<usize> {
    let limit = match model_max_context {
        Some(max_context) => max_context.checked_sub(max_tokens).with_context(|| {
            format!(
                "What: an image request cannot fit within the model context. \
                 Why: max_tokens ({max_tokens}) already exceeds the model context limit ({max_context}) before prompt and image tokens are counted. \
                 How: reduce max_tokens below the model context limit."
            )
        })?,
        None => MAX_EXPANDED_PROMPT_TOKENS,
    };
    Ok(limit.min(MAX_EXPANDED_PROMPT_TOKENS))
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
    let presence_keys = image_presence_keys(directory, &bindings)?;
    VisionInputSpec::from_program(bindings, &pixel_shape, program, vision, presence_keys).map(Some)
}

fn image_presence_keys(
    directory: &PipelineModelDirectory,
    bindings: &[VisionOutputBinding],
) -> anyhow::Result<Vec<String>> {
    let mut keys = std::collections::BTreeSet::new();
    for binding in bindings {
        let (component, input) = binding.endpoint.split_once('.').with_context(|| {
            format!(
                "image preprocessing endpoint '{}' is not component.input",
                binding.endpoint
            )
        })?;
        if let Some(key) = directory
            .spec
            .phases
            .get(component)
            .and_then(|phase| phase.when_present.as_ref())
        {
            keys.insert(key.clone());
        }
        if let Some(key) = directory
            .spec
            .models
            .get(component)
            .and_then(|model| model.io.as_ref())
            .and_then(|io| io.optional_inputs.get(input))
            .map(|optional| &optional.presence)
        {
            keys.insert(key.clone());
        }
    }
    Ok(keys.into_iter().collect())
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

#[cfg(test)]
mod tests {
    use super::*;

    fn audio_only() -> MultimodalSpecs {
        MultimodalSpecs {
            vision: None,
            audio: Some(
                AudioInputSpec::from_input(
                    "encoder.input_features".to_string(),
                    &[1, 80, 3000],
                    None,
                )
                .expect("a valid audio contract"),
            ),
        }
    }

    fn vision_only() -> MultimodalSpecs {
        MultimodalSpecs {
            vision: Some(
                VisionInputSpec::from_input("encoder.pixel_values".to_string(), &[1, 3, 4, 4])
                    .expect("a valid vision contract"),
            ),
            audio: None,
        }
    }

    fn expect_rejection(result: anyhow::Result<()>) -> String {
        let message = result
            .expect_err("the attachment must be rejected")
            .to_string();
        assert!(message.contains("What:"), "message: {message}");
        assert!(message.contains("Why:"), "message: {message}");
        assert!(message.contains("How:"), "message: {message}");
        message
    }

    #[test]
    fn a_single_decoder_graph_accepts_text_only() {
        assert!(admit_attachments(None, "the model", 0, 0).is_ok());

        let message = expect_rejection(admit_attachments(None, "the model", 1, 0));
        assert!(message.contains("single decoder graph"), "{message}");
        assert!(message.contains("it accepts text input"), "{message}");

        let message = expect_rejection(admit_attachments(None, "the model", 0, 1));
        assert!(message.contains("single decoder graph"), "{message}");
    }

    #[test]
    fn a_pipeline_admits_only_the_modalities_it_declares() {
        let vision = vision_only();
        assert!(admit_attachments(Some(&vision), "the model", 2, 0).is_ok());
        let message = expect_rejection(admit_attachments(Some(&vision), "the model", 0, 1));
        assert!(message.contains("input_features"), "{message}");
        assert!(
            message.contains("it accepts text + image input"),
            "{message}"
        );

        let audio = audio_only();
        assert!(admit_attachments(Some(&audio), "the model", 0, 1).is_ok());
        let message = expect_rejection(admit_attachments(Some(&audio), "the model", 1, 0));
        assert!(message.contains("preprocessing.image"), "{message}");
        assert!(
            message.contains("it accepts text + audio input"),
            "{message}"
        );
    }

    #[test]
    fn modalities_cannot_be_mixed_and_audio_is_one_clip_at_a_time() {
        let both = MultimodalSpecs {
            vision: vision_only().vision,
            audio: audio_only().audio,
        };
        assert_eq!(both.accepted_modalities(), "text + image + audio");
        assert_eq!(both.sole_modality(), None);

        let message = expect_rejection(admit_attachments(Some(&both), "the model", 1, 1));
        assert!(message.contains("send them separately"), "{message}");

        let message = expect_rejection(admit_attachments(Some(&both), "the model", 0, 2));
        assert!(message.contains("one clip"), "{message}");
    }

    #[test]
    fn sole_modality_names_the_attachment_a_caller_likely_forgot() {
        assert_eq!(vision_only().sole_modality(), Some("image"));
        assert_eq!(audio_only().sole_modality(), Some("audio"));
        assert_eq!(
            MultimodalSpecs {
                vision: None,
                audio: None
            }
            .sole_modality(),
            None
        );
    }

    #[test]
    fn placeholders_are_prepended_when_the_prompt_writes_none() {
        let spec = vision_only().vision.expect("a vision contract");
        // The test spec carries no expansion contract, so exercise the rule
        // directly against a known placeholder id.
        let mut prompt = vec![10_u32, 11];
        assert!(ensure_placeholders(&spec, &mut prompt, 2).is_ok());
        // Without a declared placeholder the prompt is left untouched.
        assert_eq!(prompt, vec![10, 11]);
    }

    #[test]
    fn binding_images_activates_the_image_presence_key() {
        let input = MultimodalInput {
            tensors: Vec::new(),
            image_summaries: vec![ImageExpansionSummary {
                image_index: 0,
                original_size: (1, 1),
                tile_grid: onnx_genai_preprocess::image::TileGrid {
                    columns: 1,
                    rows: 1,
                },
                tile_count: 1,
                expansion_count: 1,
                patch_grid: None,
                spatial_merge_size: 1,
                tensor_offset: 0,
                tensor_length: 1,
            }],
            presence_keys: vec!["image_features".to_string()],
        };
        let request = input
            .bind(PipelineGenerateRequest::new(
                onnx_genai::GenerateRequest::new(onnx_genai::GeneratePrompt::TokenIds(vec![0])),
            ))
            .expect("image presence is bound");

        assert!(request.present.contains("image_features"));
        assert!(!request.present.contains("image"));
    }

    #[test]
    fn placeholder_rules_accept_a_full_set_and_reject_a_partial_one() {
        // Exercised through the pure rule so it does not need a package that
        // declares a multi-image tensor axis.
        assert_eq!(placeholder_action(3, 0, 2), PlaceholderAction::Prepend(2));
        assert_eq!(placeholder_action(3, 2, 2), PlaceholderAction::Keep);
        assert_eq!(placeholder_action(3, 1, 2), PlaceholderAction::Mismatch);
        assert_eq!(placeholder_action(3, 3, 2), PlaceholderAction::Mismatch);
        // No images: a prompt that mentions none is fine.
        assert_eq!(placeholder_action(3, 0, 0), PlaceholderAction::Keep);
    }

    #[test]
    fn the_expansion_budget_reserves_room_for_the_response() {
        assert_eq!(expansion_token_budget(Some(4096), 512).unwrap(), 3584);
        // No declared context: fall back to the hard cap.
        assert_eq!(
            expansion_token_budget(None, 512).unwrap(),
            MAX_EXPANDED_PROMPT_TOKENS
        );
        // A context smaller than the reservation is a user error, not a clamp.
        let message = expansion_token_budget(Some(128), 256)
            .expect_err("an impossible reservation must fail")
            .to_string();
        assert!(message.contains("What:"), "{message}");
        assert!(message.contains("How:"), "{message}");
    }
}

//! Generic media binding for workflow package inputs.

use std::path::{Path, PathBuf};

use anyhow::Context;
use onnx_genai_engine::PipelineGenerateRequest;
use onnx_genai_metadata::{RuntimeInputRole, SemanticInputRole, TensorDimension};
use onnx_genai_ort::{DataType, PipelineModelDirectory, PipelineModels, Value};

pub use crate::audio_input::{AudioInputSpec, AudioTensor, preprocess_audio, preprocess_samples};
pub use crate::media_input::MAX_EXPANDED_PROMPT_TOKENS;

#[derive(Debug, Clone)]
pub struct VisionInputSpec {
    input: String,
    /// The prompt token the package expands into one image's token run.
    ///
    /// Declared as the `image_placeholder` special token. Without it the
    /// package states no place in the token stream for its image features, and
    /// binding an image is refused rather than silently ignored.
    placeholder_token_id: Option<u32>,
    /// The package's own image program, used to size each placeholder run.
    program: Option<Box<onnx_genai_metadata::VisionPreprocessingProgram>>,
}

impl VisionInputSpec {
    pub fn placeholder_token_id(&self) -> Option<u32> {
        self.placeholder_token_id
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
        prompt_token_ids: &mut Vec<u32>,
        max_prompt_tokens: usize,
    ) -> anyhow::Result<Self> {
        if images.len() != 1 {
            anyhow::bail!("workflow image binding requires exactly one encoded image tensor");
        }
        expand_image_placeholders(spec, &images[0], prompt_token_ids)?;
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

    pub fn from_audio_bytes(
        spec: &AudioInputSpec,
        bytes: &[u8],
        limits: onnx_genai_preprocess::audio::AudioDecodeLimits,
    ) -> anyhow::Result<Self> {
        if spec.is_encoded() {
            onnx_genai_preprocess::audio::decode_audio(bytes, limits)
                .context("validate encoded audio input before workflow admission")?;
            return Ok(Self {
                tensors: vec![PreparedTensor::EncodedAudio {
                    name: spec.endpoint.clone(),
                    bytes: bytes.to_vec(),
                }],
            });
        }
        let tensor = preprocess_audio(bytes, spec, limits)?;
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
        if spec.is_encoded() {
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
    // A single decoder declares a workflow like every other package, but it is
    // not a composed pipeline: it has no components to feed and no multimodal
    // inputs to describe. This reads layer 1 of the shared classification; the
    // engine's executor choice reads layer 2 of the same one, which is layer 1
    // plus the decode contract, so the two cannot drift apart.
    if directory
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.pipeline.as_ref())
        .is_some_and(|pipeline| onnx_genai_metadata::is_single_decoder_workflow(&pipeline.workflow))
    {
        return Ok(None);
    }
    let _models = PipelineModels::load_with_ort_session_filter(
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
        multimodal: build(&directory)?,
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

pub fn build(directory: &PipelineModelDirectory) -> anyhow::Result<MultimodalSpecs> {
    if let Some(metadata) = directory.metadata.as_ref()
        && onnx_genai_metadata::transcription::transcription_contract(metadata)
            .map_err(anyhow::Error::msg)?
            .is_some()
    {
        // A frame-decoding task owns this encoded input. It is not permission
        // to feed audio into the generative chat convenience path.
        return Ok(MultimodalSpecs {
            vision: None,
            audio: None,
        });
    }
    let mut specs = derive_specs(
        &directory.spec.workflow,
        directory.preprocessing.as_ref(),
        image_placeholder_token_id(directory.metadata.as_ref()),
    )?;
    if let Some(audio) = &mut specs.audio {
        audio.transcription =
            onnx_genai_engine::transcription::generative::GenerativeTranscription::load(
                &directory.root,
            )?;
    }
    Ok(specs)
}

/// The token a package declares as its expandable image placeholder.
fn image_placeholder_token_id(
    metadata: Option<&onnx_genai_metadata::InferenceMetadata>,
) -> Option<u32> {
    metadata?
        .package
        .as_ref()?
        .tokenizer
        .as_ref()?
        .special_tokens
        .as_ref()?
        .image_token_id
}

/// Replace each declared image placeholder with that image's token run.
///
/// The run length is a property of the preprocessed image, not of the package:
/// the package's own image program decides the patch grid, and the patch grid
/// decides how many image tokens the prompt must reserve. Running the program
/// here is what lets a caller state one placeholder and get a correctly sized
/// prompt, which is the only way the embedding component can scatter the vision
/// features into the token stream.
fn expand_image_placeholders(
    spec: &VisionInputSpec,
    encoded: &[u8],
    prompt_token_ids: &mut Vec<u32>,
) -> anyhow::Result<()> {
    let placeholder = spec.placeholder_token_id.with_context(|| {
        "What: this package accepts an image but cannot place it in the prompt. \
         Why: it declares no numeric image placeholder token, so there is no \
         token for the image's features to replace, and the encoded image would be \
         preprocessed and then ignored. \
         How: declare `package.tokenizer.special_tokens.image_token_id`; keep its text \
         spelling only in tokenizer assets."
            .to_string()
    })?;
    let program = spec
        .program
        .as_ref()
        .context("image binding requires the package's preprocessing.image program")?;
    let occurrences = prompt_token_ids
        .iter()
        .filter(|token| **token == placeholder)
        .count();
    if occurrences != 1 {
        anyhow::bail!(
            "What: the prompt carries {occurrences} image placeholder token(s) but one image \
             was attached. Why: each attached image is expanded from exactly one placeholder. \
             How: include the model's image placeholder token once per image."
        );
    }
    let pixels = program
        .outputs
        .iter()
        .find(|output| output.content == "pixels")
        .context("preprocessing.image must declare a pixels output")?;
    let shape = pixels
        .contract
        .as_ref()
        .context("preprocessing.image pixels output must declare a TensorContract")?
        .shape
        .iter()
        .map(|dim| match dim {
            TensorDimension::Fixed(value) => *value,
            TensorDimension::Symbol(_) | TensorDimension::Any => -1,
        })
        .collect::<Vec<_>>();
    let bundle =
        onnx_genai_preprocess::image::ImagePreprocessor::from_input_and_program(&shape, program)?
            .preprocess_encoded([encoded])?;
    let summary = bundle
        .images
        .first()
        .context("image preprocessing produced no image summary")?;
    let tokens = summary.image_token_count()?.context(
        "What: the image's prompt token run could not be sized. \
         Why: this package's image program emits tiles rather than a patch grid, and the \
         tokens-per-tile count is a package fact preprocessing cannot recover. \
         How: use a patchifying image program, or expand the placeholder before calling.",
    )?;
    let position = prompt_token_ids
        .iter()
        .position(|token| *token == placeholder)
        .expect("placeholder occurrence counted above");
    // One placeholder becomes `tokens` copies of itself: the embedding component
    // matches on the token id to decide where the vision features land.
    prompt_token_ids.splice(
        position..position + 1,
        std::iter::repeat_n(placeholder, tokens),
    );
    Ok(())
}

/// Bind the package's media workflow inputs to the server's media front ends.
///
/// Both front ends are derived from the same `media` runtime inputs and are
/// told apart by their declared contract, never by a port name.
fn derive_specs(
    workflow: &onnx_genai_metadata::WorkflowSpec,
    preprocessing: Option<&onnx_genai_metadata::PreprocessingSpec>,
    placeholder_token_id: Option<u32>,
) -> anyhow::Result<MultimodalSpecs> {
    let media_inputs = workflow
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
    // A preprocessing program is executable workflow data, not authority for
    // the chat/audio frontends. Task profiles may own application-sourced
    // images independently; only runtime Media roles grant these bindings.
    if media_inputs.is_empty() {
        return Ok(MultimodalSpecs {
            vision: None,
            audio: None,
        });
    }
    let encoded_media = media_inputs
        .iter()
        .find(|(_, input)| input.contract.dtype == "uint8" && input.contract.rank() == 1);
    let vision = if let Some(program) = preprocessing.and_then(|p| p.image.as_ref()) {
        let (name, _input) = encoded_media
            .context("preprocessing.image requires a uint8 rank-1 media workflow input")?;
        Some(VisionInputSpec {
            input: (*name).clone(),
            placeholder_token_id,
            program: Some(Box::new(program.clone())),
        })
    } else {
        None
    };
    // Encoded input programs execute inside the workflow; feature input programs
    // execute at this boundary. Both use the same authored DSP and clip admission.
    let feature_inputs = media_inputs
        .iter()
        .filter(|(_, input)| input.contract.dtype == "float32" && input.contract.rank() == 3)
        .collect::<Vec<_>>();
    anyhow::ensure!(
        feature_inputs.len() <= 1,
        "multiple rank-3 float32 media inputs are ambiguous for one audio attachment; declare one audio feature endpoint or bind media explicitly"
    );
    let audio = if let Some((name, input)) = feature_inputs.first() {
        let program = preprocessing.and_then(|p|p.audio.as_ref()).with_context(||
            format!("audio feature input '{name}' requires preprocessing.audio; declare its sample rate, analysis transforms and named output instead of relying on a default recipe"))?;
        let shape = input
            .contract
            .shape
            .iter()
            .map(|dim| match dim {
                TensorDimension::Fixed(value) => *value,
                _ => -1,
            })
            .collect::<Vec<_>>();
        Some(AudioInputSpec::from_input(
            (*name).clone(),
            &shape,
            program,
        )?)
    } else if let Some(program) = preprocessing.and_then(|p| p.audio.as_ref()) {
        anyhow::ensure!(
            vision.is_none(),
            "one encoded media endpoint cannot own both image and audio preprocessing; declare unambiguous inputs"
        );
        let (name, _) = encoded_media.context("preprocessing.audio requires a uint8 rank-1 media input or a declared float32 feature input")?;
        Some(AudioInputSpec::from_encoded_input(
            (*name).clone(),
            program,
        )?)
    } else {
        None
    };

    Ok(MultimodalSpecs { vision, audio })
}

#[cfg(test)]
mod media_binding_tests {
    use onnx_genai_metadata::{InferenceMetadata, PreprocessingSpec, WorkflowSpec};

    use super::{derive_specs, image_placeholder_token_id};

    /// A workflow declaring exactly the given `media` runtime inputs.
    fn workflow(inputs: serde_json::Value) -> WorkflowSpec {
        serde_json::from_value(serde_json::json!({
            "manifest": {},
            "inputs": inputs,
            "outputs": {},
            "components": {},
            "steps": [],
        }))
        .expect("workflow spec")
    }

    fn media(contract: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "role": { "kind": "runtime", "version": "1", "role": "media" },
            "source": { "kind": "request" },
            "contract": contract,
        })
    }

    fn feature_program(endpoint: &str, bins: usize, frames: usize) -> PreprocessingSpec {
        serde_json::from_value(serde_json::json!({"audio":{
            "transforms":[{"op":"decode","outputs":["raw"]},
                {"op":"resample","sample_rate":8000,"outputs":["resampled"]},
                {"op":"pad","mode":"fixed_window","target_length":frames*64,"outputs":["padded"]},
                {"op":"log_mel","sample_rate":8000,"num_mel_bins":bins,"n_fft":128,"hop_length":64,"win_length":96,"outputs":["mel"]}],
            "outputs":[{"name":endpoint,"source":"mel","content":"audio_features","dtype":"float32"}]
        }})).unwrap()
    }
    #[test]
    fn image_placeholder_comes_from_package_tokenizer_facts() {
        let metadata: InferenceMetadata = serde_json::from_value(serde_json::json!({
            "schema_version": "v1.2",
            "package": {
                "tokenizer": {
                    "special_tokens": {
                        "image_token_id": 151655,
                    },
                },
            },
        }))
        .expect("metadata");

        assert_eq!(image_placeholder_token_id(Some(&metadata)), Some(151655));
    }

    #[test]
    fn a_rank_3_float_media_input_binds_the_audio_front_end() {
        // Shape binds the result; the authored program owns the analysis math.
        let workflow = workflow(serde_json::json!({
            "audio_features": media(serde_json::json!({
                "dtype": "float32",
                "shape": [1, 80, 3000],
            })),
        }));
        let specs = derive_specs(
            &workflow,
            Some(&feature_program("audio_features", 80, 3000)),
            None,
        )
        .expect("specs");
        assert!(
            derive_specs(&workflow, None, None)
                .unwrap_err()
                .to_string()
                .contains("preprocessing.audio")
        );
        let audio = specs.audio.expect("audio spec");
        assert_eq!(audio.endpoint, "audio_features");
        assert_eq!(audio.feature_shape(), Some([1, 80, 3000]));
        assert!(specs.vision.is_none());
    }

    #[test]
    fn encoded_audio_admission_accepts_waveform_outputs_and_uses_program_windows() {
        let workflow = workflow(
            serde_json::json!({"recording":media(serde_json::json!({"dtype":"uint8","shape":["bytes"]}))}),
        );
        let preprocessing:PreprocessingSpec=serde_json::from_value(serde_json::json!({"audio":{"transforms":[{"op":"decode","outputs":["raw"]},{"op":"resample","sample_rate":8000,"outputs":["resampled"]},{"op":"trim","target_length":256,"outputs":["wave"]}],"outputs":[{"name":"wave","source":"wave","content":"waveform","dtype":"float32"}]}})).unwrap();
        let audio = derive_specs(&workflow, Some(&preprocessing), None)
            .unwrap()
            .audio
            .unwrap();
        assert!(audio.is_encoded());
        assert!(audio.feature_shape().is_none());
        assert_eq!(
            audio.maximum_input_samples(16000).unwrap().unwrap().get(),
            512
        );
    }
    #[test]
    fn feature_audio_accepts_arbitrary_positive_mel_counts_and_rejects_missing_geometry() {
        let declaration = feature_program("features", 7, 4).audio.unwrap();
        let spec =
            super::AudioInputSpec::from_input("features".into(), &[1, 7, 4], &declaration).unwrap();
        let silent = super::preprocess_samples(&[0.0; 512], 16000, &spec).unwrap();
        let tone = super::preprocess_samples(&[0.5; 512], 16000, &spec).unwrap();
        assert_eq!(tone.shape, [1, 7, 4]);
        assert_eq!(tone.data.len(), 28);
        assert_eq!(
            spec.maximum_input_samples(44100).unwrap().unwrap().get(),
            1413
        );
        assert!(tone.data.iter().all(|v| v.is_finite()));
        assert_ne!(tone.data, silent.data);
        let encoded = onnx_genai_preprocess::audio::encode_wav_f32(&[0.5; 512], 16000, 1).unwrap();
        let expected = onnx_genai_preprocess::audio::AudioProgram::from_program(&declaration)
            .unwrap()
            .run(
                &[&encoded],
                onnx_genai_preprocess::audio::AudioDecodeLimits::APPLICATION_POLICY,
            )
            .unwrap();
        let onnx_genai_preprocess::audio::AudioTensorData::Fp32(data) = &expected[0].data else {
            panic!("features")
        };
        assert_eq!(&tone.data, data);
        let mut wrong = declaration.clone();
        wrong.outputs[0].name = "unbound".into();
        assert!(
            super::AudioInputSpec::from_input("features".into(), &[1, 7, 4], &wrong)
                .unwrap_err()
                .to_string()
                .contains("same name")
        );
        let mut wrong = declaration.clone();
        wrong.outputs[0].content = "waveform".into();
        assert!(super::AudioInputSpec::from_input("features".into(), &[1, 7, 4], &wrong).is_err());
        let mut wrong = declaration.clone();
        wrong.transforms[2].target_length = Some(384);
        let wrong =
            super::AudioInputSpec::from_input("features".into(), &[1, 7, 4], &wrong).unwrap();
        let error = super::preprocess_samples(&[0.5; 512], 16000, &wrong)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("features") && error.contains("shape") && error.contains("geometry"),
            "{error}"
        );
        let mut branched = declaration.clone();
        branched.transforms.push(serde_json::from_value(serde_json::json!({"op":"trim","target_length":8,"inputs":["resampled"],"outputs":["short"]})).unwrap());
        branched.outputs.push(serde_json::from_value(serde_json::json!({"name":"unused","source":"short","content":"waveform","dtype":"float32"})).unwrap());
        let branched =
            super::AudioInputSpec::from_input("features".into(), &[1, 7, 4], &branched).unwrap();
        assert_eq!(
            branched
                .maximum_input_samples(44100)
                .unwrap()
                .unwrap()
                .get(),
            1413
        );
        assert_eq!(
            super::preprocess_samples(&[0.5; 512], 16000, &branched)
                .unwrap()
                .data,
            tone.data
        );
        for shape in [[1, 0, 4], [1, 7, -1], [2, 7, 4]] {
            assert!(
                super::AudioInputSpec::from_input("features".into(), &shape, &declaration).is_err()
            );
        }
    }
    #[test]
    fn capability_free_media_workflow_parses_with_crlf() {
        let authored = workflow(serde_json::json!({
            "audio_features": media(serde_json::json!({
                "dtype": "float32",
                "shape": [1, 80, 3000],
            })),
        }));
        let yaml = serde_yaml::to_string(&authored)
            .expect("a structurally declared media workflow serializes");
        assert!(
            !yaml.contains("capabilities"),
            "media eligibility is structural, not a manifest capability list"
        );
        let crlf = yaml.replace('\n', "\r\n");
        let reparsed: WorkflowSpec =
            serde_yaml::from_str(&crlf).expect("CRLF capability-free workflow parses");
        assert!(
            derive_specs(
                &reparsed,
                Some(&feature_program("audio_features", 80, 3000)),
                None
            )
            .expect("media contract is sufficient")
            .audio
            .is_some(),
            "the runtime media input and tensor contract, not a list flag, select audio"
        );
    }

    #[test]
    fn an_audio_contract_with_unconstrained_shape_is_refused_by_name() {
        // Unconstrained extents cannot state the mel-bin count, and guessing 80
        // would make a 128-mel model produce silent garbage.
        let workflow = workflow(serde_json::json!({
            "audio_features": media(serde_json::json!({
                "dtype": "float32",
                "shape": ["Any", "Any", "Any"],
            })),
        }));
        let error = derive_specs(
            &workflow,
            Some(&feature_program("audio_features", 7, 4)),
            None,
        )
        .expect_err("must refuse");
        let message = format!("{error:#}");
        assert!(
            message.contains("audio_features")
                && message.contains("positive concrete feature dimension")
                && message.contains("-1"),
            "{message}"
        );
    }

    #[test]
    fn an_image_package_binds_vision_and_leaves_audio_unbound() {
        let workflow = workflow(serde_json::json!({
            "image_bytes": media(serde_json::json!({ "dtype": "uint8", "shape": ["Any"] })),
        }));
        let preprocessing: PreprocessingSpec = serde_json::from_value(serde_json::json!({
            "image": {
                "transforms": [{ "op": "decode_rgb", "outputs": ["pixels"] }],
                "outputs": [{
                    "source": "pixels",
                    "name": "image.pixel_values",
                    "content": "pixels",
                    "dtype": "float32",
                }],
            },
        }))
        .expect("preprocessing");
        let specs = derive_specs(&workflow, Some(&preprocessing), None).expect("specs");
        assert!(specs.vision.is_some());
        assert!(specs.audio.is_none());
    }

    #[test]
    fn a_workflow_with_no_media_inputs_binds_nothing() {
        let specs = derive_specs(&workflow(serde_json::json!({})), None, None).expect("specs");
        assert!(specs.is_empty());
    }

    /// A patchifying image program sized like the real Qwen-VL processor.
    fn patchifying_preprocessing() -> PreprocessingSpec {
        serde_json::from_value(serde_json::json!({
            "image": {
                "transforms": [
                    { "op": "decode_rgb", "outputs": ["t0"] },
                    {
                        "op": "resize",
                        "mode": "pixel_area",
                        "min_pixels": 65536,
                        "max_pixels": 16777216,
                        "size_multiple": 32,
                        "inputs": ["t0"],
                        "outputs": ["t1"],
                    },
                    {
                        "op": "patchify",
                        "patch_size": 16,
                        "flatten": true,
                        "temporal_patch_size": 2,
                        "merge_size": 2,
                        "channel_order": "channels_first",
                        "inputs": ["t1"],
                        "outputs": ["t2"],
                    },
                ],
                "outputs": [{
                    "source": "t2",
                    "name": "image.pixel_values",
                    "content": "pixels",
                    "dtype": "float32",
                    "contract": { "dtype": "float32", "shape": ["num_patches", 1536] },
                }],
            },
        }))
        .expect("preprocessing")
    }

    fn image_workflow() -> WorkflowSpec {
        workflow(serde_json::json!({
            "image_bytes": media(serde_json::json!({ "dtype": "uint8", "shape": ["Any"] })),
        }))
    }

    /// A 64x64 solid PNG, small enough to inline and large enough to patchify.
    fn encoded_png() -> Vec<u8> {
        let mut raw = Vec::new();
        for _ in 0..64 {
            raw.push(0u8);
            raw.extend(std::iter::repeat_n(128u8, 64 * 3));
        }
        let mut png = vec![137, 80, 78, 71, 13, 10, 26, 10];
        let mut chunk = |kind: &[u8], data: Vec<u8>| {
            png.extend((data.len() as u32).to_be_bytes());
            let mut body = kind.to_vec();
            body.extend(&data);
            png.extend(&body);
            png.extend(crc32(&body).to_be_bytes());
        };
        let mut ihdr = Vec::new();
        ihdr.extend(64u32.to_be_bytes());
        ihdr.extend(64u32.to_be_bytes());
        ihdr.extend([8, 2, 0, 0, 0]);
        chunk(b"IHDR", ihdr);
        chunk(b"IDAT", deflate_stored(&raw));
        chunk(b"IEND", Vec::new());
        png
    }

    fn crc32(data: &[u8]) -> u32 {
        let mut crc = 0xffff_ffffu32;
        for byte in data {
            crc ^= u32::from(*byte);
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (0xedb8_8320 & mask);
            }
        }
        !crc
    }

    /// zlib stream with stored (uncompressed) deflate blocks.
    fn deflate_stored(data: &[u8]) -> Vec<u8> {
        let mut out = vec![0x78, 0x01];
        for (index, block) in data.chunks(65535).enumerate() {
            let last = u8::from((index + 1) * 65535 >= data.len());
            out.push(last);
            out.extend((block.len() as u16).to_le_bytes());
            out.extend((!(block.len() as u16)).to_le_bytes());
            out.extend(block);
        }
        let mut a = 1u32;
        let mut b = 0u32;
        for byte in data {
            a = (a + u32::from(*byte)) % 65521;
            b = (b + a) % 65521;
        }
        out.extend(((b << 16) | a).to_be_bytes());
        out
    }

    #[test]
    fn a_declared_placeholder_expands_to_the_image_token_run() {
        // The run length is a property of the preprocessed image, so the server
        // runs the package's own program to size it. Without this the vision
        // features have nowhere to land and the image is silently ignored.
        let preprocessing = patchifying_preprocessing();
        let specs = derive_specs(&image_workflow(), Some(&preprocessing), Some(7)).expect("specs");
        let vision = specs.vision.expect("vision spec");
        assert_eq!(vision.placeholder_token_id(), Some(7));
        let mut prompt = vec![1u32, 7, 2];
        super::MultimodalInput::from_images(&vision, &[encoded_png()], &mut prompt, 4096)
            .expect("bind");
        // 64x64 is below the program's 65536 min_pixels, so it is scaled up to
        // 256x256: a 16x16 grid of 16px patches, merged 2x2 into 64 tokens.
        assert_eq!(prompt.len(), 2 + 64, "prompt: {prompt:?}");
        assert_eq!(prompt[0], 1);
        assert_eq!(prompt[prompt.len() - 1], 2);
        assert!(prompt[1..prompt.len() - 1].iter().all(|token| *token == 7));
    }

    #[test]
    fn an_image_without_a_declared_placeholder_is_refused_by_name() {
        // Silently dropping the image is worse than refusing it: the model would
        // answer confidently from text alone and the caller could not tell.
        let preprocessing = patchifying_preprocessing();
        let specs = derive_specs(&image_workflow(), Some(&preprocessing), None).expect("specs");
        let vision = specs.vision.expect("vision spec");
        let mut prompt = vec![1u32, 2];
        let error =
            super::MultimodalInput::from_images(&vision, &[encoded_png()], &mut prompt, 4096)
                .expect_err("must refuse");
        assert!(
            format!("{error:#}").contains("tokens.image_token_id"),
            "{error:#}"
        );
    }

    #[test]
    fn a_prompt_without_the_placeholder_is_refused() {
        let preprocessing = patchifying_preprocessing();
        let specs = derive_specs(&image_workflow(), Some(&preprocessing), Some(7)).expect("specs");
        let vision = specs.vision.expect("vision spec");
        let mut prompt = vec![1u32, 2];
        let error =
            super::MultimodalInput::from_images(&vision, &[encoded_png()], &mut prompt, 4096)
                .expect_err("must refuse");
        assert!(
            format!("{error:#}").contains("0 image placeholder token(s)"),
            "{error:#}"
        );
    }

    #[test]
    fn an_expanded_prompt_over_budget_is_refused() {
        // The budget must be checked after expansion: one placeholder token can
        // become hundreds, and the pre-expansion length says nothing about it.
        let preprocessing = patchifying_preprocessing();
        let specs = derive_specs(&image_workflow(), Some(&preprocessing), Some(7)).expect("specs");
        let vision = specs.vision.expect("vision spec");
        let mut prompt = vec![7u32];
        let error = super::MultimodalInput::from_images(&vision, &[encoded_png()], &mut prompt, 3)
            .expect_err("must refuse");
        assert!(format!("{error:#}").contains("context budget"), "{error:#}");
    }
}

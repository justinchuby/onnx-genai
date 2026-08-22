use std::collections::BTreeMap;
use std::io::Cursor;

use axum::{Json, extract::State};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use image::{DynamicImage, ImageFormat, Rgb, RgbImage};
use onnx_genai::{GenerateOptions, GeneratePrompt, GenerateRequest};
use onnx_genai_metadata::ImageOutputValueRange;
use onnx_genai_ort::DataType;
use serde::{Deserialize, Serialize};
use serde_json::{Value as JsonValue, json};

use super::{ApiError, ApiJson, AppState, map_generate_submit_error, now_unix, resolve_model};
use crate::image_generation::{
    ImageExecutionRequest, ImageInputValue, ImagePipelineSpec, ProducedImage, scalar_f32,
    scalar_i64, token_input,
};

const MAX_IMAGES_PER_REQUEST: usize = 8;
const IMAGE_RANGE_RELATIVE_EPSILON: f32 = 1.0e-5;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OpenAiImageRequest {
    model: String,
    prompt: String,
    #[serde(default = "one")]
    n: usize,
    #[serde(default)]
    size: Option<String>,
    #[serde(default)]
    quality: Option<OpenAiQuality>,
    #[serde(default)]
    style: Option<OpenAiStyle>,
    #[serde(default)]
    response_format: OpenAiResponseFormat,
    #[serde(default)]
    output_format: Option<OpenAiOutputFormat>,
    #[serde(default)]
    background: Option<OpenAiBackground>,
    #[serde(default)]
    moderation: Option<OpenAiModeration>,
    #[serde(default)]
    user: Option<String>,
    /// Non-standard bridge for metadata-declared application inputs that the
    /// caller prepared outside the server (for example multimodal encoders).
    #[serde(default)]
    application_inputs: BTreeMap<String, ApplicationTensor>,
}

fn one() -> usize {
    1
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum OpenAiQuality {
    Standard,
    Hd,
    Low,
    Medium,
    High,
    Auto,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum OpenAiStyle {
    Natural,
    Vivid,
}

#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum OpenAiResponseFormat {
    #[default]
    B64Json,
    Url,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum OpenAiOutputFormat {
    Png,
    Jpeg,
    Webp,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum OpenAiBackground {
    Auto,
    Opaque,
    Transparent,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
enum OpenAiModeration {
    Auto,
    Low,
}

#[derive(Debug, Serialize)]
pub(crate) struct OpenAiImageResponse {
    created: u64,
    data: Vec<OpenAiImageData>,
}

#[derive(Debug, Serialize)]
struct OpenAiImageData {
    b64_json: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct A1111ImageRequest {
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    negative_prompt: String,
    #[serde(default = "default_seed")]
    seed: i64,
    #[serde(default = "default_subseed")]
    subseed: i64,
    #[serde(default)]
    subseed_strength: f32,
    #[serde(default)]
    steps: Option<usize>,
    #[serde(default)]
    cfg_scale: Option<f32>,
    #[serde(default)]
    width: Option<u32>,
    #[serde(default)]
    height: Option<u32>,
    #[serde(default)]
    sampler_name: Option<String>,
    #[serde(default)]
    sampler_index: Option<String>,
    #[serde(default = "one")]
    batch_size: usize,
    #[serde(default = "one")]
    n_iter: usize,
    #[serde(default)]
    init_images: Vec<String>,
    #[serde(default)]
    denoising_strength: Option<f32>,
    #[serde(default = "default_true")]
    send_images: bool,
    #[serde(default)]
    save_images: bool,
    /// Model-agnostic typed inputs for metadata sources with kind
    /// `application`. These are explicit rather than silently inferred from
    /// A1111 fields when a package has no server-side preprocessor.
    #[serde(default)]
    application_inputs: BTreeMap<String, ApplicationTensor>,
    #[serde(default, flatten)]
    extra: BTreeMap<String, JsonValue>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ApplicationTensor {
    dtype: String,
    shape: Vec<i64>,
    data_b64: String,
}

impl ApplicationTensor {
    fn lower(
        &self,
        contract: &onnx_genai_metadata::TensorContract,
    ) -> Result<ImageInputValue, ApiError> {
        if self.dtype != contract.dtype {
            return Err(ApiError::bad_request(format!(
                "application input dtype '{}' does not match metadata dtype '{}'",
                self.dtype, contract.dtype
            )));
        }
        if self.shape.len() != contract.rank {
            return Err(ApiError::bad_request(format!(
                "application input rank {} does not match metadata rank {}",
                self.shape.len(),
                contract.rank
            )));
        }
        let dtype = parse_application_dtype(&self.dtype)?;
        if self.shape.iter().any(|dimension| *dimension < 0) {
            return Err(ApiError::bad_request(
                "application input shapes must contain only non-negative dimensions",
            ));
        }
        let elements = self.shape.iter().try_fold(1_usize, |total, dimension| {
            let dimension = usize::try_from(*dimension)
                .map_err(|_| ApiError::bad_request("application input dimension is too large"))?;
            total
                .checked_mul(dimension)
                .ok_or_else(|| ApiError::bad_request("application input element count overflowed"))
        })?;
        let expected = elements
            .checked_mul(dtype.size_of())
            .ok_or_else(|| ApiError::bad_request("application input byte count overflowed"))?;
        let bytes = STANDARD.decode(&self.data_b64).map_err(|error| {
            ApiError::bad_request(format!("application input data_b64 is invalid: {error}"))
        })?;
        if bytes.len() != expected {
            return Err(ApiError::bad_request(format!(
                "application input contains {} decoded bytes, expected {expected} for dtype '{}' and shape {:?}",
                bytes.len(),
                self.dtype,
                self.shape
            )));
        }
        Ok(ImageInputValue::Raw {
            bytes,
            shape: self.shape.clone(),
            dtype,
        })
    }
}

fn default_seed() -> i64 {
    -1
}

fn default_subseed() -> i64 {
    -1
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Serialize)]
pub(crate) struct A1111ImageResponse {
    images: Vec<String>,
    parameters: JsonValue,
    info: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct A1111Model {
    title: String,
    model_name: String,
    hash: Option<String>,
    sha256: Option<String>,
    filename: String,
    config: Option<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct A1111Sampler {
    name: String,
    aliases: Vec<String>,
    options: BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
struct NormalizedImageRequest {
    prompt: String,
    negative_prompt: String,
    seed: u64,
    steps: usize,
    guidance_scale: f32,
    width: Option<u32>,
    height: Option<u32>,
    sampler: Option<String>,
    init_image: Option<Vec<u8>>,
    denoising_strength: Option<f32>,
    application_inputs: BTreeMap<String, ApplicationTensor>,
}

pub(crate) async fn openai_images(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<OpenAiImageRequest>,
) -> Result<Json<OpenAiImageResponse>, ApiError> {
    validate_count(request.n)?;
    if request.quality.is_some() {
        return Err(ApiError::bad_request(
            "the loaded workflow declares no image quality runtime role; `quality` cannot be honored",
        ));
    }
    if request.style.is_some() {
        return Err(ApiError::bad_request(
            "the loaded workflow declares no image style runtime role; `style` cannot be honored",
        ));
    }
    if matches!(request.response_format, OpenAiResponseFormat::Url) {
        return Err(ApiError::bad_request(
            "`response_format: url` is unsupported because this server has no persistent image asset store; use `b64_json`",
        ));
    }
    if matches!(
        request.output_format,
        Some(OpenAiOutputFormat::Jpeg | OpenAiOutputFormat::Webp)
    ) {
        return Err(ApiError::bad_request(
            "only `output_format: png` is supported by the loaded image encoder",
        ));
    }
    if matches!(request.background, Some(OpenAiBackground::Transparent)) {
        return Err(ApiError::bad_request(
            "`background: transparent` cannot be honored because the workflow emits RGB without alpha",
        ));
    }
    if request.moderation.is_some() {
        return Err(ApiError::bad_request(
            "`moderation` cannot be honored because no image moderation pipeline is configured",
        ));
    }
    if let Some(user) = request.user.as_deref() {
        tracing::info!(user, "OpenAI image generation request attribution");
    }

    let handle = resolve_model(&state.registry, &request.model).await?;
    let spec = image_spec(&handle, request.application_inputs.is_empty())?;
    let (width, height) = request
        .size
        .as_deref()
        .map(parse_size)
        .transpose()?
        .map_or(Ok((None, None)), |(width, height)| {
            resolve_dimensions(spec, Some(width), Some(height))
        })?;
    let base_seed = request_seed(spec);
    let mut data = Vec::with_capacity(request.n);
    for index in 0..request.n {
        let normalized = NormalizedImageRequest {
            prompt: request.prompt.clone(),
            negative_prompt: String::new(),
            seed: base_seed.wrapping_add(index as u64),
            steps: request_steps(spec),
            guidance_scale: request_guidance(spec),
            width,
            height,
            sampler: spec.samplers.first().cloned(),
            init_image: None,
            denoising_strength: None,
            application_inputs: request.application_inputs.clone(),
        };
        let png = execute(&handle, spec, &normalized).await?;
        data.push(OpenAiImageData {
            b64_json: STANDARD.encode(png),
        });
    }
    Ok(Json(OpenAiImageResponse {
        created: now_unix(),
        data,
    }))
}

pub(crate) async fn a1111_txt2img(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<A1111ImageRequest>,
) -> Result<Json<A1111ImageResponse>, ApiError> {
    if !request.init_images.is_empty() || request.denoising_strength.is_some() {
        return Err(ApiError::bad_request(
            "txt2img does not accept `init_images` or `denoising_strength`; use /sdapi/v1/img2img",
        ));
    }
    execute_a1111(state, request, false).await
}

pub(crate) async fn a1111_img2img(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<A1111ImageRequest>,
) -> Result<Json<A1111ImageResponse>, ApiError> {
    execute_a1111(state, request, true).await
}

async fn execute_a1111(
    state: AppState,
    request: A1111ImageRequest,
    img2img: bool,
) -> Result<Json<A1111ImageResponse>, ApiError> {
    reject_a1111_unsupported(&request)?;
    let count = request
        .batch_size
        .checked_mul(request.n_iter)
        .ok_or_else(|| ApiError::bad_request("batch_size * n_iter overflowed"))?;
    validate_count(count)?;
    let handle = resolve_model(&state.registry, "").await?;
    let uses_application_inputs = !request.application_inputs.is_empty();
    let spec = image_spec(&handle, !uses_application_inputs)?;
    if spec.guidance_scale.is_none() && !uses_application_inputs {
        return Err(ApiError::bad_request(
            "the loaded workflow exposes no guidance_scale role required by the A1111 cfg_scale contract",
        ));
    }
    let sampler = resolve_sampler(
        spec,
        request.sampler_name.as_deref(),
        request.sampler_index.as_deref(),
    )?;
    let steps = resolve_steps(spec, request.steps)?;
    let guidance = if uses_application_inputs && spec.guidance_scale.is_none() {
        request.cfg_scale.unwrap_or(7.0)
    } else {
        resolve_guidance(spec, request.cfg_scale)?
    };
    let (width, height) =
        if uses_application_inputs && spec.width.is_none() && spec.height.is_none() {
            (request.width, request.height)
        } else {
            resolve_dimensions(spec, request.width, request.height)?
        };
    let init_image = if img2img {
        if spec.media.is_none() && !uses_application_inputs {
            return Err(ApiError::bad_request(
                "the loaded workflow declares no semantic image/media input, so img2img is unavailable",
            ));
        }
        if request.init_images.len() != 1 {
            return Err(ApiError::bad_request(
                "img2img requires exactly one base64 entry in `init_images`",
            ));
        }
        let bytes = decode_base64_image(&request.init_images[0])?;
        spec.media.as_ref().map(|_| bytes)
    } else {
        None
    };
    let denoising_strength = if uses_application_inputs && spec.denoising_strength.is_none() {
        None
    } else {
        resolve_denoising_strength(spec, img2img, request.denoising_strength)?
    };
    let first_seed = resolve_seed(request.seed)?;
    let mut images = Vec::with_capacity(count);
    let mut all_seeds = Vec::with_capacity(count);
    let mut actual_width = width;
    let mut actual_height = height;
    for index in 0..count {
        let seed = first_seed.wrapping_add(index as u64);
        let normalized = NormalizedImageRequest {
            prompt: request.prompt.clone(),
            negative_prompt: request.negative_prompt.clone(),
            seed,
            steps,
            guidance_scale: guidance,
            width,
            height,
            sampler: sampler.clone(),
            init_image: init_image.clone(),
            denoising_strength,
            application_inputs: request.application_inputs.clone(),
        };
        let png = execute(&handle, spec, &normalized).await?;
        let decoded =
            image::load_from_memory_with_format(&png, ImageFormat::Png).map_err(|error| {
                ApiError::internal(format!("generated PNG validation failed: {error}"))
            })?;
        actual_width = Some(decoded.width());
        actual_height = Some(decoded.height());
        if request.send_images {
            images.push(STANDARD.encode(png));
        }
        all_seeds.push(seed);
    }
    let parameters = json!({
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "seed": first_seed,
        "subseed": request.subseed,
        "subseed_strength": request.subseed_strength,
        "steps": steps,
        "cfg_scale": guidance,
        "width": actual_width,
        "height": actual_height,
        "sampler_name": sampler,
        "batch_size": request.batch_size,
        "n_iter": request.n_iter,
        "denoising_strength": denoising_strength,
        "send_images": request.send_images,
        "save_images": request.save_images,
    });
    let info = json!({
        "seed": first_seed,
        "all_seeds": all_seeds,
        "width": actual_width,
        "height": actual_height,
        "sampler_name": sampler,
        "steps": steps,
        "cfg_scale": guidance,
    })
    .to_string();
    Ok(Json(A1111ImageResponse {
        images,
        parameters,
        info,
    }))
}

fn reject_a1111_unsupported(request: &A1111ImageRequest) -> Result<(), ApiError> {
    if request.save_images {
        return Err(ApiError::bad_request(
            "`save_images: true` is unsupported because the server has no configured image asset store",
        ));
    }
    if request.subseed != -1 || request.subseed_strength != 0.0 {
        return Err(ApiError::bad_request(
            "subseed variation is not exposed by the loaded workflow",
        ));
    }
    if !request.extra.is_empty() {
        return Err(ApiError::bad_request(format!(
            "unsupported A1111 fields (scripts/extensions/control inputs are never silently ignored): {}",
            request.extra.keys().cloned().collect::<Vec<_>>().join(", ")
        )));
    }
    Ok(())
}

fn image_spec(
    handle: &crate::registry::ModelHandle,
    require_http_roles: bool,
) -> Result<&ImagePipelineSpec, ApiError> {
    let spec = handle.image_pipeline.as_ref().ok_or_else(|| {
        ApiError::bad_request(
            "the selected model does not declare a pipeline image output in inference metadata",
        )
    })?;
    if require_http_roles
        && (spec.prompt_tokens.is_none() || spec.steps.is_none() || spec.seed.is_none())
    {
        return Err(ApiError::bad_request(
            "the image workflow must declare prompt_tokens, max_iterations, and seed runtime roles for HTTP generation",
        ));
    }
    Ok(spec)
}

async fn execute(
    handle: &crate::registry::ModelHandle,
    spec: &ImagePipelineSpec,
    normalized: &NormalizedImageRequest,
) -> Result<Vec<u8>, ApiError> {
    let pipeline_request = lower(handle, spec, normalized)?;
    let image = handle
        .engine
        .generate_image(pipeline_request)
        .await
        .map_err(map_generate_submit_error)?
        .map_err(|error| ApiError::internal(format!("image generation failed: {error:#}")))?;
    encode_png(image, spec.output_value_range)
}

fn lower(
    handle: &crate::registry::ModelHandle,
    spec: &ImagePipelineSpec,
    request: &NormalizedImageRequest,
) -> Result<ImageExecutionRequest, ApiError> {
    if let Some(sampler) = request.sampler.as_deref()
        && !spec.samplers.iter().any(|declared| declared == sampler)
    {
        return Err(ApiError::bad_request(format!(
            "sampler '{sampler}' is not the solver declared by the loaded workflow"
        )));
    }
    let mut prompt = Vec::new();
    let mut inputs = Vec::with_capacity(request.application_inputs.len() + 8);
    for (name, binding) in &spec.application_inputs {
        if binding.required && !request.application_inputs.contains_key(name) {
            return Err(ApiError::bad_request(format!(
                "required metadata application input '{name}' was not provided"
            )));
        }
    }
    for (name, tensor) in &request.application_inputs {
        let binding = spec.application_inputs.get(name).ok_or_else(|| {
            ApiError::bad_request(format!(
                "application input '{name}' is not declared by workflow metadata"
            ))
        })?;
        inputs.push((name.clone(), tensor.lower(&binding.contract)?));
    }
    let prompt_binding = spec.prompt_tokens.as_ref();
    if let Some(binding) = prompt_binding {
        prompt = handle.tokenizer.encode(&request.prompt).map_err(|error| {
            ApiError::bad_request(format!("failed to tokenize prompt: {error}"))
        })?;
        inputs.push((
            binding.name.clone(),
            token_input(binding, &prompt)
                .map_err(|error| ApiError::bad_request(format!("{error:#}")))?,
        ));
    } else if request.application_inputs.is_empty() {
        return Err(ApiError::bad_request(
            "the image workflow declares no prompt_tokens runtime input that the HTTP API can bind",
        ));
    }
    if !request.negative_prompt.is_empty()
        && spec.negative_prompt_tokens.is_none()
        && request.application_inputs.is_empty()
    {
        return Err(ApiError::bad_request(
            "the loaded workflow has no negative_prompt_tokens input; omit `negative_prompt`",
        ));
    }
    let mut negative_tokens = if let Some(binding) = spec.negative_prompt_tokens.as_ref() {
        if !request.negative_prompt.is_empty() || binding.required {
            let tokens = handle
                .tokenizer
                .encode(&request.negative_prompt)
                .map_err(|error| {
                    ApiError::bad_request(format!("failed to tokenize negative prompt: {error}"))
                })?;
            Some((binding, tokens))
        } else {
            None
        }
    } else {
        None
    };
    if let Some((_, negative)) = negative_tokens.as_mut() {
        let target = prompt.len().max(negative.len());
        let pad = ["<pad>", "[PAD]", "<|pad|>"]
            .into_iter()
            .find_map(|token| handle.tokenizer.token_id(token))
            .unwrap_or(0);
        prompt.resize(target, pad);
        negative.resize(target, pad);
    }
    let options = GenerateOptions {
        max_new_tokens: request.steps,
        seed: Some(request.seed),
        ..GenerateOptions::default()
    };
    let mut lowered = ImageExecutionRequest {
        request: GenerateRequest {
            prompt: GeneratePrompt::TokenIds(prompt.clone()),
            options,
        },
        inputs,
    };
    if let Some((binding, tokens)) = negative_tokens {
        lowered.inputs.push((
            binding.name.clone(),
            token_input(binding, &tokens)
                .map_err(|error| ApiError::bad_request(format!("{error:#}")))?,
        ));
    }
    if let Some(binding) = &spec.seed {
        lowered.inputs.push((
            binding.name.clone(),
            scalar_i64(binding, request.seed as i64)
                .map_err(|error| ApiError::bad_request(format!("{error:#}")))?,
        ));
    }
    if let Some(binding) = &spec.steps {
        lowered.inputs.push((
            binding.name.clone(),
            scalar_i64(binding, request.steps as i64)
                .map_err(|error| ApiError::bad_request(format!("{error:#}")))?,
        ));
    }
    if let Some(binding) = &spec.guidance_scale {
        lowered.inputs.push((
            binding.name.clone(),
            scalar_f32(binding, request.guidance_scale)
                .map_err(|error| ApiError::bad_request(format!("{error:#}")))?,
        ));
    }
    for (binding, value) in [
        (spec.width.as_ref(), request.width.map(i64::from)),
        (spec.height.as_ref(), request.height.map(i64::from)),
    ] {
        if let (Some(binding), Some(value)) = (binding, value) {
            lowered.inputs.push((
                binding.name.clone(),
                scalar_i64(binding, value)
                    .map_err(|error| ApiError::bad_request(format!("{error:#}")))?,
            ));
        }
    }
    if let Some(bytes) = &request.init_image {
        let binding = spec.media.as_ref().ok_or_else(|| {
            ApiError::bad_request("the loaded workflow declares no image/media input")
        })?;
        lowered
            .inputs
            .push((binding.name.clone(), ImageInputValue::Bytes(bytes.clone())));
    }
    if let (Some(binding), Some(value)) = (&spec.denoising_strength, request.denoising_strength) {
        lowered.inputs.push((
            binding.name.clone(),
            scalar_f32(binding, value)
                .map_err(|error| ApiError::bad_request(format!("{error:#}")))?,
        ));
    }
    Ok(lowered)
}

fn parse_application_dtype(dtype: &str) -> Result<DataType, ApiError> {
    match dtype {
        "float32" => Ok(DataType::Float32),
        "float16" => Ok(DataType::Float16),
        "bfloat16" => Ok(DataType::BFloat16),
        "int8" => Ok(DataType::Int8),
        "int16" => Ok(DataType::Int16),
        "int32" => Ok(DataType::Int32),
        "int64" => Ok(DataType::Int64),
        "uint8" => Ok(DataType::Uint8),
        "uint16" => Ok(DataType::Uint16),
        "uint32" => Ok(DataType::Uint32),
        "uint64" => Ok(DataType::Uint64),
        "bool" => Ok(DataType::Bool),
        _ => Err(ApiError::bad_request(format!(
            "unsupported application input dtype '{dtype}'"
        ))),
    }
}

fn encode_png(
    image: ProducedImage,
    value_range: ImageOutputValueRange,
) -> Result<Vec<u8>, ApiError> {
    let (minimum, maximum) = image
        .values
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), value| {
            (min.min(*value), max.max(*value))
        });
    let (range_minimum, range_maximum): (f32, f32) = match value_range {
        ImageOutputValueRange::ZeroToOne => (0.0, 1.0),
        ImageOutputValueRange::NegativeOneToOne => (-1.0, 1.0),
        ImageOutputValueRange::ZeroTo255 => (0.0, 255.0),
    };
    // GPU kernels can overshoot a mathematically bounded activation by a few
    // ulps. Validate against the declared range with a small scale-relative
    // tolerance, then clamp only that numerical noise during quantization.
    let tolerance = IMAGE_RANGE_RELATIVE_EPSILON * (range_maximum - range_minimum).abs().max(1.0);
    if !minimum.is_finite()
        || !maximum.is_finite()
        || minimum < range_minimum - tolerance
        || maximum > range_maximum + tolerance
    {
        return Err(ApiError::internal(format!(
            "image output violates declared {value_range:?} pixel range: [{minimum}, {maximum}]"
        )));
    }
    let (height, width, channel_first) = match image.shape.as_slice() {
        [1, 3, height, width] => (*height, *width, true),
        [1, 3, 1, height, width] => (*height, *width, true),
        [1, height, width, 3] => (*height, *width, false),
        [1, 1, height, width, 3] => (*height, *width, false),
        shape => {
            return Err(ApiError::internal(format!(
                "image output must contain one RGB image in NCHW, NTHWC, or NCTHW layout, got {shape:?}"
            )));
        }
    };
    let width = u32::try_from(width)
        .map_err(|_| ApiError::internal("generated image width is outside u32"))?;
    let height = u32::try_from(height)
        .map_err(|_| ApiError::internal("generated image height is outside u32"))?;
    let expected = width as usize * height as usize * 3;
    if image.values.len() != expected {
        return Err(ApiError::internal(format!(
            "image output has {} values, expected {expected}",
            image.values.len()
        )));
    }
    let mut rgb = RgbImage::new(width, height);
    for y in 0..height as usize {
        for x in 0..width as usize {
            let pixel = if channel_first {
                let plane = width as usize * height as usize;
                [
                    image.values[y * width as usize + x],
                    image.values[plane + y * width as usize + x],
                    image.values[2 * plane + y * width as usize + x],
                ]
            } else {
                let offset = (y * width as usize + x) * 3;
                [
                    image.values[offset],
                    image.values[offset + 1],
                    image.values[offset + 2],
                ]
            };
            rgb.put_pixel(
                x as u32,
                y as u32,
                Rgb(pixel.map(|value| {
                    let normalized = match value_range {
                        ImageOutputValueRange::ZeroToOne => value,
                        ImageOutputValueRange::NegativeOneToOne => (value + 1.0) * 0.5,
                        ImageOutputValueRange::ZeroTo255 => value / 255.0,
                    };
                    (normalized.clamp(0.0, 1.0) * 255.0).round() as u8
                })),
            );
        }
    }
    let mut bytes = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(rgb)
        .write_to(&mut bytes, ImageFormat::Png)
        .map_err(|error| ApiError::internal(format!("failed to encode PNG: {error}")))?;
    Ok(bytes.into_inner())
}

fn validate_count(count: usize) -> Result<(), ApiError> {
    if !(1..=MAX_IMAGES_PER_REQUEST).contains(&count) {
        return Err(ApiError::bad_request(format!(
            "image count must be between 1 and {MAX_IMAGES_PER_REQUEST}"
        )));
    }

    Ok(())
}

fn parse_size(size: &str) -> Result<(u32, u32), ApiError> {
    let (width, height) = size.split_once('x').ok_or_else(|| {
        ApiError::bad_request(
            "size must use the OpenAI `WIDTHxHEIGHT` form, for example `1024x1024`",
        )
    })?;
    let width = width
        .parse()
        .map_err(|_| ApiError::bad_request("size width must be a positive integer"))?;
    let height = height
        .parse()
        .map_err(|_| ApiError::bad_request("size height must be a positive integer"))?;
    Ok((width, height))
}

fn request_seed(spec: &ImagePipelineSpec) -> u64 {
    ImagePipelineSpec::default_i64(spec.seed.as_ref())
        .unwrap_or(0)
        .max(0) as u64
}

fn request_steps(spec: &ImagePipelineSpec) -> usize {
    ImagePipelineSpec::default_i64(spec.steps.as_ref())
        .unwrap_or(20)
        .max(1) as usize
}

fn request_guidance(spec: &ImagePipelineSpec) -> f32 {
    ImagePipelineSpec::default_f32(spec.guidance_scale.as_ref()).unwrap_or(7.0)
}

fn resolve_seed(seed: i64) -> Result<u64, ApiError> {
    if seed >= 0 {
        return Ok(seed as u64);
    }
    if seed != -1 {
        return Err(ApiError::bad_request(
            "seed must be -1 or a non-negative integer",
        ));
    }
    let mut bytes = [0_u8; 8];
    getrandom::fill(&mut bytes)
        .map_err(|error| ApiError::internal(format!("failed to generate random seed: {error}")))?;
    Ok(u64::from_le_bytes(bytes) & i64::MAX as u64)
}

fn resolve_steps(spec: &ImagePipelineSpec, steps: Option<usize>) -> Result<usize, ApiError> {
    if steps.is_some() && spec.steps.is_none() {
        return Err(ApiError::bad_request(
            "the loaded workflow exposes no max_iterations role, so requested `steps` cannot be bound",
        ));
    }
    let steps = steps.unwrap_or_else(|| request_steps(spec));
    if steps == 0 {
        return Err(ApiError::bad_request("steps must be greater than zero"));
    }
    Ok(steps)
}

fn resolve_guidance(spec: &ImagePipelineSpec, value: Option<f32>) -> Result<f32, ApiError> {
    if value.is_some() && spec.guidance_scale.is_none() {
        return Err(ApiError::bad_request(
            "the loaded workflow exposes no guidance_scale role, so requested `cfg_scale` cannot be bound",
        ));
    }
    let value = value.unwrap_or_else(|| request_guidance(spec));
    if !value.is_finite() || value < 0.0 {
        return Err(ApiError::bad_request(
            "cfg_scale must be finite and non-negative",
        ));
    }
    Ok(value)
}

fn resolve_dimensions(
    spec: &ImagePipelineSpec,
    width: Option<u32>,
    height: Option<u32>,
) -> Result<(Option<u32>, Option<u32>), ApiError> {
    match (width, height) {
        (None, None) => Ok((None, None)),
        (Some(width), Some(height)) if spec.width.is_some() && spec.height.is_some() => {
            if width == 0 || height == 0 {
                Err(ApiError::bad_request(
                    "width and height must be greater than zero",
                ))
            } else {
                Ok((Some(width), Some(height)))
            }
        }
        _ => Err(ApiError::bad_request(
            "requested width/height cannot be bound because the loaded workflow does not expose both semantic dimension roles",
        )),
    }
}

fn resolve_sampler(
    spec: &ImagePipelineSpec,
    sampler_name: Option<&str>,
    sampler_index: Option<&str>,
) -> Result<Option<String>, ApiError> {
    if let (Some(name), Some(index)) = (sampler_name, sampler_index)
        && normalize_sampler(name) != normalize_sampler(index)
    {
        return Err(ApiError::bad_request(
            "`sampler_name` and `sampler_index` request different samplers",
        ));
    }
    let requested = sampler_name.or(sampler_index);
    let Some(requested) = requested else {
        return Ok(spec.samplers.first().cloned());
    };
    spec.samplers
        .iter()
        .find(|sampler| normalize_sampler(sampler) == normalize_sampler(requested))
        .cloned()
        .map(Some)
        .ok_or_else(|| {
            ApiError::bad_request(format!(
                "sampler '{requested}' is unsupported; available samplers: {}",
                spec.samplers.join(", ")
            ))
        })
}

fn normalize_sampler(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn resolve_denoising_strength(
    spec: &ImagePipelineSpec,
    img2img: bool,
    value: Option<f32>,
) -> Result<Option<f32>, ApiError> {
    if !img2img {
        return Ok(None);
    }
    let value = value.unwrap_or(0.75);
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(ApiError::bad_request(
            "denoising_strength must be between 0 and 1",
        ));
    }
    if spec.denoising_strength.is_none() {
        return Err(ApiError::bad_request(
            "the loaded workflow declares an image input but no denoising_strength runtime role",
        ));
    }
    Ok(Some(value))
}

fn decode_base64_image(input: &str) -> Result<Vec<u8>, ApiError> {
    let encoded = input.split_once(',').map_or(input, |(_, payload)| payload);
    let bytes = STANDARD.decode(encoded).map_err(|error| {
        ApiError::bad_request(format!("init image is not valid base64: {error}"))
    })?;
    image::load_from_memory(&bytes)
        .map_err(|error| ApiError::bad_request(format!("init image is not decodable: {error}")))?;
    Ok(bytes)
}

pub(crate) async fn a1111_models(
    State(state): State<AppState>,
) -> Result<Json<Vec<A1111Model>>, ApiError> {
    let handle = resolve_model(&state.registry, "").await?;
    Ok(Json(vec![A1111Model {
        title: handle.id.clone(),
        model_name: handle.id.clone(),
        hash: None,
        sha256: None,
        filename: handle.id.clone(),
        config: None,
    }]))
}

pub(crate) async fn a1111_samplers(
    State(state): State<AppState>,
) -> Result<Json<Vec<A1111Sampler>>, ApiError> {
    let handle = resolve_model(&state.registry, "").await?;
    let spec = image_spec(&handle, false)?;
    Ok(Json(
        spec.samplers
            .iter()
            .map(|name| A1111Sampler {
                name: name.clone(),
                aliases: Vec::new(),
                options: BTreeMap::new(),
            })
            .collect(),
    ))
}

pub(crate) async fn a1111_options(
    State(state): State<AppState>,
) -> Result<Json<JsonValue>, ApiError> {
    let handle = resolve_model(&state.registry, "").await?;
    let spec = image_spec(&handle, false)?;
    Ok(Json(json!({
        "sd_model_checkpoint": handle.id,
        "samples_format": "png",
        "save_images": false,
        "sampler_name": spec.samplers.first(),
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode_pixel(png: &[u8]) -> [u8; 3] {
        image::load_from_memory_with_format(png, ImageFormat::Png)
            .unwrap()
            .into_rgb8()
            .get_pixel(0, 0)
            .0
    }

    #[test]
    fn channel_first_tensor_encodes_as_png() {
        let png = encode_png(
            ProducedImage {
                values: vec![1.0, 0.0, 0.0],
                shape: vec![1, 3, 1, 1],
            },
            ImageOutputValueRange::ZeroToOne,
        )
        .unwrap();
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
    }

    #[test]
    fn single_frame_video_tensor_encodes_as_png() {
        let png = encode_png(
            ProducedImage {
                values: vec![1.0, 0.0, 0.0],
                shape: vec![1, 3, 1, 1, 1],
            },
            ImageOutputValueRange::ZeroToOne,
        )
        .unwrap();
        assert_eq!(decode_pixel(&png), [255, 0, 0]);
    }

    #[test]
    fn declared_value_range_controls_pixel_conversion() {
        let image = || ProducedImage {
            values: vec![0.0, 0.5, 1.0],
            shape: vec![1, 3, 1, 1],
        };
        let zero_to_one = encode_png(image(), ImageOutputValueRange::ZeroToOne).unwrap();
        let negative_one_to_one =
            encode_png(image(), ImageOutputValueRange::NegativeOneToOne).unwrap();
        let zero_to_255 = encode_png(
            ProducedImage {
                values: vec![0.0, 127.5, 255.0],
                shape: vec![1, 3, 1, 1],
            },
            ImageOutputValueRange::ZeroTo255,
        )
        .unwrap();

        assert_eq!(decode_pixel(&zero_to_one), [0, 128, 255]);
        assert_eq!(decode_pixel(&negative_one_to_one), [128, 191, 255]);
        assert_eq!(decode_pixel(&zero_to_255), [0, 128, 255]);
    }

    #[test]
    fn pixels_outside_the_declared_range_are_rejected() {
        let error = encode_png(
            ProducedImage {
                values: vec![-0.1, 0.0, 1.0],
                shape: vec![1, 3, 1, 1],
            },
            ImageOutputValueRange::ZeroToOne,
        )
        .unwrap_err();
        assert!(error.message.contains("violates declared ZeroToOne"));
    }

    #[test]
    fn numerical_noise_at_declared_range_boundaries_is_clamped() {
        let zero_to_one = encode_png(
            ProducedImage {
                values: vec![-5.0e-6, 0.5, 1.0 + 5.0e-6],
                shape: vec![1, 3, 1, 1],
            },
            ImageOutputValueRange::ZeroToOne,
        )
        .unwrap();
        let negative_one_to_one = encode_png(
            ProducedImage {
                values: vec![-1.0 - 5.0e-6, 0.0, 1.0 + 5.0e-6],
                shape: vec![1, 3, 1, 1],
            },
            ImageOutputValueRange::NegativeOneToOne,
        )
        .unwrap();

        assert_eq!(decode_pixel(&zero_to_one), [0, 128, 255]);
        assert_eq!(decode_pixel(&negative_one_to_one), [0, 128, 255]);
    }

    #[test]
    fn material_range_violations_are_not_treated_as_numerical_noise() {
        for (values, value_range) in [
            (vec![-1.0e-3, 0.0, 1.0], ImageOutputValueRange::ZeroToOne),
            (
                vec![-1.001, 0.0, 1.0],
                ImageOutputValueRange::NegativeOneToOne,
            ),
        ] {
            let error = encode_png(
                ProducedImage {
                    values,
                    shape: vec![1, 3, 1, 1],
                },
                value_range,
            )
            .unwrap_err();
            assert!(error.message.contains("violates declared"));
        }
    }

    #[test]
    fn application_tensor_decodes_typed_raw_bytes() {
        let tensor = ApplicationTensor {
            dtype: "bfloat16".to_string(),
            shape: vec![1, 2],
            data_b64: STANDARD.encode([0_u8, 1, 2, 3]),
        };

        assert!(matches!(
            tensor
                .lower(&onnx_genai_metadata::TensorContract {
                    dtype: "bfloat16".to_string(),
                    rank: 2,
                    shape: None,
                    optional: false,
                    batch_layout: Default::default(),
                })
                .unwrap(),
            ImageInputValue::Raw {
                bytes,
                shape,
                dtype: DataType::BFloat16,
            } if bytes == [0, 1, 2, 3] && shape == [1, 2]
        ));
    }

    #[test]
    fn application_tensor_rejects_wrong_byte_count() {
        let tensor = ApplicationTensor {
            dtype: "float32".to_string(),
            shape: vec![2],
            data_b64: STANDARD.encode([0_u8; 4]),
        };

        let error = tensor
            .lower(&onnx_genai_metadata::TensorContract {
                dtype: "float32".to_string(),
                rank: 1,
                shape: None,
                optional: false,
                batch_layout: Default::default(),
            })
            .unwrap_err();
        assert!(error.message.contains("expected 8"));
    }
}

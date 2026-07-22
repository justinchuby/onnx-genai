use std::time::Duration;

use anyhow::Context;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use onnx_genai_metadata::ImagePreprocessingProgram;
use onnx_genai_ort::DataType;
use onnx_genai_preprocess::image::{
    ImagePreprocessor, ThumbnailPosition,
    packed::{ImageExpansionSummary, ImageTensorDType, ImageTensorData},
};

const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;
const MAX_IMAGE_BASE64_BYTES: usize = MAX_IMAGE_BYTES.div_ceil(3) * 4;
pub(crate) const MAX_EXPANDED_PROMPT_TOKENS: usize = 1024 * 1024;

#[derive(Debug)]
pub(crate) struct ImageTensor {
    pub(crate) endpoint: String,
    pub(crate) expected_dtype: DataType,
    pub(crate) expected_shape: Vec<i64>,
    pub(crate) shape: Vec<i64>,
    pub(crate) data: ImageTensorData,
}

#[derive(Debug)]
pub(crate) struct ImageBundle {
    pub(crate) tensors: Vec<ImageTensor>,
    pub(crate) images: Vec<ImageExpansionSummary>,
    /// Compatibility view of the first fp32 tensor for existing unit tests.
    #[cfg(test)]
    pub(crate) shape: Vec<i64>,
    /// Compatibility view of the first fp32 tensor for existing unit tests.
    #[cfg(test)]
    pub(crate) data: Vec<f32>,
    /// Total number of preprocessed tiles in this tensor batch.
    #[cfg(test)]
    pub(crate) num_tiles: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct VisionOutputBinding {
    pub(crate) metadata_name: String,
    pub(crate) endpoint: String,
    pub(crate) content: String,
    pub(crate) dtype: DataType,
    pub(crate) shape: Vec<i64>,
}

#[derive(Debug, Clone, Copy)]
enum ImageTokenCountSource {
    PerTile,
    PerPatch,
}

#[derive(Debug, Clone)]
struct VisionExpansionSpec {
    placeholder_token_id: i64,
    image_token_id: i64,
    count_source: ImageTokenCountSource,
    tokens_per_unit: usize,
    row_separator_token_id: Option<i64>,
    column_separator_token_id: Option<i64>,
    thumbnail_position: ThumbnailPosition,
}

#[derive(Debug, Clone)]
pub(crate) struct VisionInputSpec {
    bindings: Vec<VisionOutputBinding>,
    preprocessor: ImagePreprocessor,
    expansion: Option<VisionExpansionSpec>,
}

impl VisionInputSpec {
    #[cfg(test)]
    pub(crate) fn from_input(endpoint: String, shape: &[i64]) -> anyhow::Result<Self> {
        let preprocessor = ImagePreprocessor::from_input(shape)
            .with_context(|| format!("invalid preprocessing for vision input '{endpoint}'"))?;
        Ok(Self {
            bindings: vec![VisionOutputBinding {
                metadata_name: "pixels".to_string(),
                endpoint,
                content: "pixels".to_string(),
                dtype: DataType::Float32,
                shape: shape.to_vec(),
            }],
            preprocessor,
            expansion: None,
        })
    }

    pub(crate) fn from_program(
        bindings: Vec<VisionOutputBinding>,
        pixel_shape: &[i64],
        program: &ImagePreprocessingProgram,
        vision: &onnx_genai_metadata::PipelineVisionConfig,
    ) -> anyhow::Result<Self> {
        if bindings.is_empty() {
            anyhow::bail!(
                "What: image processor binding discovery produced no pipeline endpoints. \
                 Why: preprocessing.image.outputs did not resolve to any ONNX input. \
                 How: bind each required image output to an exact component.input endpoint in typed metadata."
            );
        }
        let preprocessor = ImagePreprocessor::from_input_and_program(pixel_shape, program)
            .context("What: the typed image preprocessing program could not be initialized. Why: its declared transforms or pixel shape are invalid. How: correct preprocessing.image transforms and outputs in inference metadata.")?;
        let expansion = Some(VisionExpansionSpec::from_metadata(vision)?);
        Ok(Self {
            bindings,
            preprocessor,
            expansion,
        })
    }

    pub(crate) fn expand_prompt(
        &self,
        prompt_token_ids: &[u32],
        bundle: &ImageBundle,
        max_prompt_tokens: usize,
    ) -> anyhow::Result<Vec<u32>> {
        let expansion = self.expansion.as_ref().context(
            "What: image placeholder expansion could not run. \
             Why: the server has image tensors but no typed pipeline.vision expansion contract. \
             How: declare image_placeholder_token_id, image_token_id, token_count_source, and its required count field.",
        )?;
        expansion.expand(prompt_token_ids, &bundle.images, max_prompt_tokens)
    }
}

impl VisionExpansionSpec {
    fn from_metadata(vision: &onnx_genai_metadata::PipelineVisionConfig) -> anyhow::Result<Self> {
        let placeholder_token_id = vision.image_placeholder_token_id.context(
            "What: image placeholder expansion metadata is incomplete. \
             Why: pipeline.vision.image_placeholder_token_id is missing. \
             How: declare the token ID that marks each image position in the tokenized prompt.",
        )?;
        let image_token_id = vision.image_token_id.unwrap_or(placeholder_token_id);
        if vision.placeholder_per_image == Some(false) {
            anyhow::bail!(
                "What: image placeholder expansion metadata is unsupported. \
                 Why: placeholder_per_image=false does not define how multiple prompt images map to placeholders. \
                 How: declare placeholder_per_image=true or add a typed grouping operation before serving this package."
            );
        }
        let (count_source, tokens_per_unit) = match vision.token_count_source.as_deref() {
            None | Some("per_tile") => (
                ImageTokenCountSource::PerTile,
                vision.tokens_per_tile.context(
                    "What: per-tile image expansion metadata is incomplete. \
                     Why: pipeline.vision.tokens_per_tile is missing. \
                     How: declare the number of image tokens emitted for each preprocessed tile.",
                )?,
            ),
            Some("per_patch") => (
                ImageTokenCountSource::PerPatch,
                vision.tokens_per_patch.context(
                    "What: per-patch image expansion metadata is incomplete. \
                     Why: pipeline.vision.tokens_per_patch is missing. \
                     How: declare the number of image tokens emitted for each real patch.",
                )?,
            ),
            Some("from_grid") => anyhow::bail!(
                "What: grid-derived image expansion metadata is incomplete. \
                 Why: token_count_source='from_grid' does not name the processor-summary endpoint whose dtype/shape supplies each image grid. \
                 How: add a typed grid-summary endpoint operation to pipeline.vision, then bind it to the declared preprocessing output."
            ),
            Some(other) => anyhow::bail!(
                "What: image placeholder expansion metadata is unsupported. \
                 Why: token_count_source '{other}' has no registered server operation. \
                 How: use per_tile, per_patch, or from_grid, or add that typed metadata operation."
            ),
        };
        if tokens_per_unit == 0 {
            anyhow::bail!(
                "What: image placeholder expansion metadata is invalid. \
                 Why: the declared tokens-per-unit value is zero. \
                 How: set tokens_per_tile or tokens_per_patch to at least 1."
            );
        }
        let thumbnail_position = match vision.thumbnail_order.as_deref() {
            None | Some("none") => ThumbnailPosition::None,
            Some("prepend") => ThumbnailPosition::Prepend,
            Some("append") => ThumbnailPosition::Append,
            Some(other) => anyhow::bail!(
                "What: image thumbnail ordering metadata is unsupported. \
                 Why: thumbnail_order '{other}' has no registered preprocessing operation. \
                 How: use none, prepend, or append."
            ),
        };
        if !matches!(count_source, ImageTokenCountSource::PerTile)
            && (vision.row_separator_token_id.is_some()
                || vision.column_separator_token_id.is_some())
        {
            anyhow::bail!(
                "What: image separator expansion metadata is incomplete. \
                 Why: {:?} expansion declares row/column separators, but the per-image summary has no patch-grid row/column operation. \
                 How: emit a typed patch-grid summary operation or remove the separator token IDs.",
                vision.token_count_source
            );
        }
        Ok(Self {
            placeholder_token_id,
            image_token_id,
            count_source,
            tokens_per_unit,
            row_separator_token_id: vision.row_separator_token_id,
            column_separator_token_id: vision.column_separator_token_id,
            thumbnail_position,
        })
    }

    fn expand(
        &self,
        prompt_token_ids: &[u32],
        images: &[ImageExpansionSummary],
        max_prompt_tokens: usize,
    ) -> anyhow::Result<Vec<u32>> {
        let placeholder = u32::try_from(self.placeholder_token_id).with_context(|| {
            format!(
                "What: image placeholder expansion could not start. Why: placeholder token ID {} is outside u32. How: declare a non-negative tokenizer token ID.",
                self.placeholder_token_id
            )
        })?;
        let image_token = u32::try_from(self.image_token_id).with_context(|| {
            format!(
                "What: image placeholder expansion could not start. Why: emitted image token ID {} is outside u32. How: declare a non-negative tokenizer token ID.",
                self.image_token_id
            )
        })?;
        let placeholder_count = prompt_token_ids
            .iter()
            .filter(|&&token| token == placeholder)
            .count();
        if placeholder_count != images.len() {
            anyhow::bail!(
                "What: image placeholder expansion count mismatch. \
                 Why: the tokenized prompt contains {placeholder_count} placeholder(s) for endpoint token ID {placeholder}, but preprocessing produced {} image summary record(s). \
                 How: include exactly one matching placeholder for every image content part, in prompt order.",
                images.len()
            );
        }

        let row_separator = self
            .row_separator_token_id
            .map(u32::try_from)
            .transpose()
            .with_context(|| {
                format!(
                    "What: image row separator expansion failed. Why: token ID {:?} is outside u32. How: declare a valid tokenizer token ID.",
                    self.row_separator_token_id
                )
            })?;
        let column_separator = self
            .column_separator_token_id
            .map(u32::try_from)
            .transpose()
            .with_context(|| {
                format!(
                    "What: image column separator expansion failed. Why: token ID {:?} is outside u32. How: declare a valid tokenizer token ID.",
                    self.column_separator_token_id
                )
            })?;

        let mut replacement_len = 0usize;
        for (prompt_image_index, image) in images.iter().enumerate() {
            if image.image_index != prompt_image_index {
                anyhow::bail!(
                    "What: image prompt ordering validation failed. \
                     Why: prompt image {prompt_image_index} received preprocessing summary index {}. \
                     How: preserve image content parts and preprocessing summaries in prompt order.",
                    image.image_index
                );
            }
            let image_len = self.replacement_len(image)?;
            replacement_len = replacement_len.checked_add(image_len).with_context(|| {
                format!(
                    "What: image placeholder expansion length overflowed. Why: adding image {prompt_image_index}'s {image_len} tokens made the aggregate replacement length exceed usize. How: reduce image count, tiles, patches, or tokens-per-unit metadata."
                )
            })?;
        }

        let output_len = prompt_token_ids
            .len()
            .checked_sub(placeholder_count)
            .and_then(|base| base.checked_add(replacement_len))
            .context(
                "What: image placeholder expansion overflowed. Why: the final prefill length does not fit usize. How: reduce image count, tiles, patches, or expansion tokens.",
            )?;
        if output_len > max_prompt_tokens {
            anyhow::bail!(
                "What: image placeholder expansion exceeds the allowed prompt length. \
                Why: the final prefill length would be {output_len} tokens, above the pre-allocation limit of {max_prompt_tokens} tokens derived from the model context and the server safety bound ({MAX_EXPANDED_PROMPT_TOKENS}). \
                How: reduce the prompt, image count, tiles, patches, or tokens-per-unit metadata."
            );
        }
        let mut expanded = Vec::new();
        expanded.try_reserve_exact(output_len).context(
            "What: image placeholder expansion allocation failed. Why: memory could not be reserved for the validated final prefill token bundle. How: reduce the prompt, image count, or image expansion size.",
        )?;
        let mut image_index = 0usize;
        for &token in prompt_token_ids {
            if token == placeholder {
                self.append_replacement(
                    &images[image_index],
                    image_token,
                    row_separator,
                    column_separator,
                    &mut expanded,
                );
                image_index += 1;
            } else {
                expanded.push(token);
            }
        }
        Ok(expanded)
    }

    fn replacement_len(&self, image: &ImageExpansionSummary) -> anyhow::Result<usize> {
        match self.count_source {
            ImageTokenCountSource::PerPatch => image
                .expansion_count
                .checked_mul(self.tokens_per_unit)
                .context(
                    "What: per-image patch expansion length overflowed. Why: expansion_count multiplied by tokens_per_patch exceeds usize. How: reduce the patch count or tokens_per_patch metadata.",
                ),
            ImageTokenCountSource::PerTile => {
                let columns = usize::try_from(image.tile_grid.columns)
                    .context("What: image tile expansion length could not be computed. Why: the tile-grid column count does not fit usize. How: reduce the declared preprocessing tile grid.")?;
                let rows = usize::try_from(image.tile_grid.rows)
                    .context("What: image tile expansion length could not be computed. Why: the tile-grid row count does not fit usize. How: reduce the declared preprocessing tile grid.")?;
                let local_tiles = rows.checked_mul(columns).context(
                    "What: image tile expansion length overflowed. Why: tile-grid rows multiplied by columns exceeds usize. How: reduce the preprocessing tile grid.",
                )?;
                let thumbnail_tiles =
                    usize::from(self.thumbnail_position != ThumbnailPosition::None);
                let expected_tiles = local_tiles
                    .checked_add(thumbnail_tiles)
                    .context(
                        "What: image tile expansion length overflowed. Why: adding the thumbnail to the local tile count exceeds usize. How: reduce the preprocessing tile grid or remove the thumbnail.",
                    )?;
                if image.tile_count != expected_tiles {
                    anyhow::bail!(
                        "What: image tile expansion summary is inconsistent. \
                         Why: image {} reports {} tile(s), but its {}x{} grid and {:?} thumbnail metadata require {expected_tiles}. \
                         How: make preprocessing.image tile/thumbnail operations match pipeline.vision.thumbnail_order.",
                        image.image_index,
                        image.tile_count,
                        image.tile_grid.columns,
                        image.tile_grid.rows,
                        self.thumbnail_position
                    );
                }
                let image_tokens = expected_tiles
                    .checked_mul(self.tokens_per_unit)
                    .context(
                        "What: per-image tile expansion length overflowed. Why: tile count multiplied by tokens_per_tile exceeds usize. How: reduce the tile count or tokens_per_tile metadata.",
                    )?;
                let column_separators = if self.column_separator_token_id.is_some() {
                    rows.checked_mul(columns.saturating_sub(1)).context(
                        "What: image column-separator length overflowed. Why: rows multiplied by separators-per-row exceeds usize. How: reduce the preprocessing tile grid.",
                    )?
                } else {
                    0
                };
                let row_separators = if self.row_separator_token_id.is_some() {
                    rows.saturating_sub(1)
                } else {
                    0
                };
                image_tokens
                    .checked_add(column_separators)
                    .and_then(|len| len.checked_add(row_separators))
                    .context(
                        "What: per-image tile expansion length overflowed. Why: image tokens plus row/column separators exceeds usize. How: reduce the tile grid, tokens_per_tile, or separator usage.",
                    )
            }
        }
    }

    fn append_replacement(
        &self,
        image: &ImageExpansionSummary,
        image_token: u32,
        row_separator: Option<u32>,
        column_separator: Option<u32>,
        tokens: &mut Vec<u32>,
    ) {
        let emit_unit = |tokens: &mut Vec<u32>| {
            tokens.extend(std::iter::repeat_n(image_token, self.tokens_per_unit));
        };
        match self.count_source {
            ImageTokenCountSource::PerPatch => {
                for _ in 0..image.expansion_count {
                    emit_unit(tokens);
                }
            }
            ImageTokenCountSource::PerTile => {
                if self.thumbnail_position == ThumbnailPosition::Prepend {
                    emit_unit(tokens);
                }
                for row in 0..image.tile_grid.rows {
                    for column in 0..image.tile_grid.columns {
                        emit_unit(tokens);
                        if column + 1 < image.tile_grid.columns
                            && let Some(separator) = column_separator
                        {
                            tokens.push(separator);
                        }
                    }
                    if row + 1 < image.tile_grid.rows
                        && let Some(separator) = row_separator
                    {
                        tokens.push(separator);
                    }
                }
                if self.thumbnail_position == ThumbnailPosition::Append {
                    emit_unit(tokens);
                }
            }
        }
    }
}

pub(crate) async fn load_and_preprocess(
    urls: &[String],
    spec: &VisionInputSpec,
) -> anyhow::Result<ImageBundle> {
    let mut images = Vec::with_capacity(urls.len());
    for url in urls {
        images.push(load_image_bytes(url).await?);
    }
    let bundle = spec
        .preprocessor
        .preprocess_encoded(&images)
        .context("What: image preprocessing failed. Why: encoded image data did not satisfy the typed preprocessing program. How: inspect preprocessing.image transforms, output dtypes, and endpoint shapes.")?;

    let mut tensors = Vec::with_capacity(spec.bindings.len());
    for binding in &spec.bindings {
        let tensor = bundle
            .tensor(&binding.metadata_name)
            .with_context(|| {
                format!(
                    "What: image processor endpoint '{}' was not produced. Why: preprocessing.image.outputs declared '{}' ({}, expected {:?} {:?}), but its operation emitted no tensor. How: add or fix the '{}' output operation in typed metadata.",
                    binding.endpoint,
                    binding.metadata_name,
                    binding.content,
                    binding.dtype,
                    binding.shape,
                    binding.content
                )
            })?;
        let actual_dtype = ort_dtype(tensor.dtype);
        if actual_dtype != binding.dtype || !shape_matches(&binding.shape, &tensor.shape) {
            anyhow::bail!(
                "What: image processor endpoint '{}' has an incompatible tensor. \
                 Why: expected dtype {:?} shape {:?}, got dtype {:?} shape {:?}. \
                 How: correct preprocessing.image.outputs '{}' and its typed transform operation.",
                binding.endpoint,
                binding.dtype,
                binding.shape,
                actual_dtype,
                tensor.shape,
                binding.metadata_name
            );
        }
        tensors.push(ImageTensor {
            endpoint: binding.endpoint.clone(),
            expected_dtype: binding.dtype,
            expected_shape: binding.shape.clone(),
            shape: tensor.shape.clone(),
            data: tensor.data.clone(),
        });
    }

    #[cfg(test)]
    let (shape, data) = tensors
        .iter()
        .find_map(|tensor| match &tensor.data {
            ImageTensorData::Fp32(data) => Some((tensor.shape.clone(), data.clone())),
            _ => None,
        })
        .unwrap_or_default();

    Ok(ImageBundle {
        tensors,
        images: bundle.images,
        #[cfg(test)]
        shape,
        #[cfg(test)]
        data,
        #[cfg(test)]
        num_tiles: bundle.num_tiles,
    })
}

pub(crate) fn metadata_dtype(value: &str) -> anyhow::Result<DataType> {
    match value {
        "float32" | "fp32" => Ok(DataType::Float32),
        "float16" | "fp16" | "half" => Ok(DataType::Float16),
        "bfloat16" | "bf16" => Ok(DataType::BFloat16),
        "int64" => Ok(DataType::Int64),
        "int32" => Ok(DataType::Int32),
        "int8" => Ok(DataType::Int8),
        "uint8" => Ok(DataType::Uint8),
        "bool" => Ok(DataType::Bool),
        other => anyhow::bail!(
            "What: image output dtype metadata is unsupported. Why: '{other}' has no typed tensor mapping. How: declare float32, float16, bfloat16, int64, int32, int8, uint8, or bool."
        ),
    }
}

fn ort_dtype(dtype: ImageTensorDType) -> DataType {
    match dtype {
        ImageTensorDType::Fp32 => DataType::Float32,
        ImageTensorDType::Fp16 => DataType::Float16,
        ImageTensorDType::Bf16 => DataType::BFloat16,
        ImageTensorDType::Int64 => DataType::Int64,
        ImageTensorDType::Int32 => DataType::Int32,
        ImageTensorDType::Int8 => DataType::Int8,
        ImageTensorDType::Uint8 => DataType::Uint8,
        ImageTensorDType::Bool => DataType::Bool,
    }
}

fn shape_matches(expected: &[i64], actual: &[i64]) -> bool {
    expected.len() == actual.len()
        && expected
            .iter()
            .zip(actual)
            .all(|(&expected, &actual)| expected < 0 || expected == actual)
}

async fn load_image_bytes(url: &str) -> anyhow::Result<Vec<u8>> {
    if let Some(data) = url.strip_prefix("data:image/") {
        let (_, encoded) = data
            .split_once(";base64,")
            .context(
                "What: image data URI parsing failed. Why: the URI is missing the required ';base64,' marker after its image media type. How: send data:image/<type>;base64,<standard-base64-payload>.",
            )?;
        if encoded.len() > MAX_IMAGE_BASE64_BYTES {
            anyhow::bail!(
                "What: image data URI exceeds the input limit. \
                 Why: its base64 payload is {} bytes, above the maximum encoded length of {MAX_IMAGE_BASE64_BYTES} bytes for a {MAX_IMAGE_BYTES}-byte image. \
                 How: resize or recompress the image so its decoded bytes are at most {MAX_IMAGE_BYTES}.",
                encoded.len()
            );
        }
        let bytes = STANDARD.decode(encoded).with_context(|| {
            "What: image data URI decoding failed. Why: the payload after ';base64,' is not valid standard base64. How: encode the image bytes with standard base64 and send data:image/<type>;base64,<payload>."
        })?;
        if bytes.len() > MAX_IMAGE_BYTES {
            anyhow::bail!(
                "What: decoded image data exceeds the input limit. \
                 Why: the data URI decoded to {} bytes, above the {MAX_IMAGE_BYTES}-byte limit. \
                 How: resize or recompress the image before encoding it.",
                bytes.len()
            );
        }
        return Ok(bytes);
    }

    if url.starts_with("http://") || url.starts_with("https://") {
        let parsed_url = reqwest::Url::parse(url).context(
            "What: image URL parsing failed. Why: the supplied HTTP(S) value is not a valid absolute URL. How: provide a complete URL such as https://images.example/image.png without credentials in the URL.",
        )?;
        if parsed_url.host_str().is_none() {
            anyhow::bail!(
                "What: image URL parsing failed. \
                 Why: the supplied HTTP(S) URL has no host. \
                 How: provide a complete URL such as https://images.example/image.png."
            );
        }
        let safe_url = safe_http_url_context(&parsed_url);
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .redirect(reqwest::redirect::Policy::limited(3))
            .build()
            .context(
                "What: image URL fetching could not start. Why: the server could not initialize its bounded HTTP client. How: retry the request or contact the server operator if the failure persists.",
            )?;
        let mut response = client
            .get(parsed_url)
            .send()
            .await
            .with_context(|| {
                format!(
                    "What: image URL fetch failed for {safe_url}. Why: the remote host could not be reached within the timeout or redirect limit. How: verify the URL is reachable over HTTP(S), uses at most 3 redirects, and responds within 10 seconds."
                )
            })?;
        if !response.status().is_success() {
            anyhow::bail!(
                "What: image URL fetch failed for {safe_url}. \
                 Why: the remote host returned HTTP status {} instead of a successful response. \
                 How: provide a URL that returns the image directly with a 2xx status.",
                response.status()
            );
        }
        if response
            .content_length()
            .is_some_and(|length| length > MAX_IMAGE_BYTES as u64)
        {
            anyhow::bail!(
                "What: image URL response exceeds the input limit for {safe_url}. \
                 Why: Content-Length is {} bytes, above the {MAX_IMAGE_BYTES}-byte limit. \
                 How: resize or recompress the remote image, or provide a smaller image URL.",
                response.content_length().unwrap_or_default()
            );
        }
        let initial_capacity = response.content_length().unwrap_or_default() as usize;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(initial_capacity).with_context(|| {
            format!(
                "What: image URL response allocation failed for {safe_url}. Why: memory could not be reserved for the declared {initial_capacity}-byte body. How: provide a smaller image."
            )
        })?;
        while let Some(chunk) = response.chunk().await.with_context(|| {
            format!(
                "What: image URL response could not be read from {safe_url}. Why: the HTTP body ended with a transport or decoding error. How: verify the remote server returns a complete image response."
            )
        })? {
            let next_len = bytes.len().checked_add(chunk.len()).context(
                "What: image URL response length overflowed. Why: accumulated body length does not fit usize. How: provide an image no larger than the documented byte limit.",
            )?;
            if next_len > MAX_IMAGE_BYTES {
                anyhow::bail!(
                    "What: image URL response exceeds the input limit for {safe_url}. \
                     Why: the streamed body reached at least {next_len} bytes, above the {MAX_IMAGE_BYTES}-byte limit. \
                     How: resize or recompress the remote image, or provide a smaller image URL."
                );
            }
            bytes.try_reserve_exact(chunk.len()).with_context(|| {
                format!(
                    "What: image URL response allocation failed for {safe_url}. Why: memory could not be extended for a validated body of {next_len} bytes. How: provide a smaller image."
                )
            })?;
            bytes.extend_from_slice(&chunk);
        }
        return Ok(bytes);
    }

    let scheme = safe_url_scheme(url);
    anyhow::bail!(
        "What: image URL scheme is unsupported. \
         Why: the request used scheme '{scheme}', but image_url accepts only data:image/...;base64, http, or https inputs. \
         How: embed the image as a base64 data URI or host it at an HTTP(S) URL."
    )
}

fn safe_http_url_context(url: &reqwest::Url) -> String {
    url.host_str()
        .map(|host| {
            let port = url
                .port()
                .map(|port| format!(":{port}"))
                .unwrap_or_default();
            format!("{}://{host}{port}", url.scheme())
        })
        .unwrap_or_else(|| "the supplied HTTP(S) origin".to_string())
}

fn safe_url_scheme(url: &str) -> &str {
    url.split_once(':')
        .map(|(scheme, _)| scheme)
        .filter(|scheme| {
            !scheme.is_empty()
                && scheme.len() <= 32
                && scheme
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'-' | b'.'))
        })
        .unwrap_or("<missing or invalid>")
}

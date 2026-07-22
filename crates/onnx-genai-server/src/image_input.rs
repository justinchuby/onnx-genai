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
    ) -> anyhow::Result<Vec<u32>> {
        let expansion = self.expansion.as_ref().context(
            "What: image placeholder expansion could not run. \
             Why: the server has image tensors but no typed pipeline.vision expansion contract. \
             How: declare image_placeholder_token_id, image_token_id, token_count_source, and its required count field.",
        )?;
        expansion.expand(prompt_token_ids, &bundle.images)
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

        let mut replacements = Vec::with_capacity(images.len());
        for (prompt_image_index, image) in images.iter().enumerate() {
            if image.image_index != prompt_image_index {
                anyhow::bail!(
                    "What: image prompt ordering validation failed. \
                     Why: prompt image {prompt_image_index} received preprocessing summary index {}. \
                     How: preserve image content parts and preprocessing summaries in prompt order.",
                    image.image_index
                );
            }
            replacements.push(self.replacement(image, image_token)?);
        }

        let replacement_len = replacements.iter().try_fold(0usize, |total, tokens| {
            total
                .checked_add(tokens.len())
                .context("expanded image token count overflowed")
        })?;
        let output_len = prompt_token_ids
            .len()
            .checked_sub(placeholder_count)
            .and_then(|base| base.checked_add(replacement_len))
            .context(
                "What: image placeholder expansion overflowed. Why: the final prefill length does not fit usize. How: reduce image count, tiles, patches, or expansion tokens.",
            )?;
        let mut expanded = Vec::new();
        expanded.try_reserve_exact(output_len).context(
            "What: image placeholder expansion allocation failed. Why: the final prefill token bundle is too large. How: reduce image count or image expansion size.",
        )?;
        let mut image_index = 0usize;
        for &token in prompt_token_ids {
            if token == placeholder {
                expanded.extend_from_slice(&replacements[image_index]);
                image_index += 1;
            } else {
                expanded.push(token);
            }
        }
        Ok(expanded)
    }

    fn replacement(
        &self,
        image: &ImageExpansionSummary,
        image_token: u32,
    ) -> anyhow::Result<Vec<u32>> {
        match self.count_source {
            ImageTokenCountSource::PerPatch => {
                let count = image
                    .expansion_count
                    .checked_mul(self.tokens_per_unit)
                    .context(
                        "What: per-image expansion count overflowed. Why: expansion_count * tokens_per_unit is too large. How: reduce patch count or expansion metadata.",
                    )?;
                Ok(std::iter::repeat_n(image_token, count).collect())
            }
            ImageTokenCountSource::PerTile => {
                let thumbnail_tiles =
                    usize::from(self.thumbnail_position != ThumbnailPosition::None);
                let local_tiles = usize::try_from(
                    image
                        .tile_grid
                        .columns
                        .checked_mul(image.tile_grid.rows)
                        .context("image tile grid multiplication overflowed")?,
                )
                .context("image tile grid does not fit usize")?;
                let expected_tiles = local_tiles
                    .checked_add(thumbnail_tiles)
                    .context("image tile count overflowed")?;
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
                let mut tokens = Vec::new();
                let emit_tile = |tokens: &mut Vec<u32>| {
                    tokens.extend(std::iter::repeat_n(image_token, self.tokens_per_unit));
                };
                if self.thumbnail_position == ThumbnailPosition::Prepend {
                    emit_tile(&mut tokens);
                }
                for row in 0..image.tile_grid.rows {
                    for column in 0..image.tile_grid.columns {
                        emit_tile(&mut tokens);
                        if column + 1 < image.tile_grid.columns
                            && let Some(separator) = self.column_separator_token_id
                        {
                            tokens.push(u32::try_from(separator).with_context(|| {
                                format!(
                                    "What: image column separator expansion failed. Why: token ID {separator} is outside u32. How: declare a valid tokenizer token ID."
                                )
                            })?);
                        }
                    }
                    if row + 1 < image.tile_grid.rows
                        && let Some(separator) = self.row_separator_token_id
                    {
                        tokens.push(u32::try_from(separator).with_context(|| {
                            format!(
                                "What: image row separator expansion failed. Why: token ID {separator} is outside u32. How: declare a valid tokenizer token ID."
                            )
                        })?);
                    }
                }
                if self.thumbnail_position == ThumbnailPosition::Append {
                    emit_tile(&mut tokens);
                }
                Ok(tokens)
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
            .context("image data URI must use base64 encoding")?;
        let bytes = STANDARD
            .decode(encoded)
            .context("image data URI contains invalid base64")?;
        if bytes.len() > MAX_IMAGE_BYTES {
            anyhow::bail!("image exceeds the {MAX_IMAGE_BYTES} byte limit");
        }
        return Ok(bytes);
    }

    if url.starts_with("http://") || url.starts_with("https://") {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .redirect(reqwest::redirect::Policy::limited(3))
            .build()
            .context("failed to initialize image HTTP client")?;
        let response = client
            .get(url)
            .send()
            .await
            .with_context(|| format!("failed to fetch image URL {url}"))?
            .error_for_status()
            .with_context(|| format!("image URL returned an error: {url}"))?;
        if response
            .content_length()
            .is_some_and(|length| length > MAX_IMAGE_BYTES as u64)
        {
            anyhow::bail!("image exceeds the {MAX_IMAGE_BYTES} byte limit");
        }
        let bytes = response
            .bytes()
            .await
            .context("failed to read image body")?;
        if bytes.len() > MAX_IMAGE_BYTES {
            anyhow::bail!("image exceeds the {MAX_IMAGE_BYTES} byte limit");
        }
        return Ok(bytes.to_vec());
    }

    anyhow::bail!("image_url must be a data:image/...;base64 URI or an http(s) URL")
}

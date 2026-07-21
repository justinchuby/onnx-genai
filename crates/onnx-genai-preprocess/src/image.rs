//! Metadata-driven RGB image preprocessing.

pub mod packed;

use std::path::Path;

use anyhow::Context;
use image::{DynamicImage, Rgb, RgbImage, imageops::FilterType};
use serde::Deserialize;

pub use packed::{
    ImageExpansionSummary, ImageTensorBundle, ImageTensorDType, ImageTensorData, NamedImageTensor,
};
use packed::{OutputSpec, PackSpec, PreparedImage};

const CHANNELS: usize = 3;

/// Tensor channel layout declared by the model input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageLayout {
    Nchw,
    Nhwc,
}

/// Image resizing strategy selected by §35 metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeMode {
    ShortestEdgeCenterCrop,
    Fixed,
    LongestEdgePad,
}

/// Resize interpolation selected by §35 metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    Bicubic,
    Bilinear,
    Lanczos3,
}

/// Image tiling strategy selected by §35 metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TilingMode {
    None,
    FixedGrid,
    DynamicAnyres,
}

/// A tile grid expressed as columns × rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileGrid {
    pub columns: u32,
    pub rows: u32,
}

impl TileGrid {
    fn tile_count(self) -> anyhow::Result<usize> {
        let count = self
            .columns
            .checked_mul(self.rows)
            .context("image tile grid is too large")?;
        usize::try_from(count).context("image tile grid is too large")
    }
}

/// Placement of an optional global-thumbnail token segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThumbnailPosition {
    None,
    Prepend,
    Append,
}

/// Configuration for expanding one prompt image placeholder per preprocessed image.
///
/// Each local tile emits `tokens_per_tile` copies of `image_token_id`. Optional
/// column separators are emitted between tiles in a row, and optional row
/// separators are emitted between rows. A global thumbnail emits one additional
/// tile-sized segment before or after the local grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenExpansionConfig {
    pub image_placeholder_token_id: i64,
    pub image_token_id: i64,
    pub tokens_per_tile: usize,
    pub thumbnail_position: ThumbnailPosition,
    pub row_separator_token_id: Option<i64>,
    pub column_separator_token_id: Option<i64>,
}

/// Tile metadata required to expand image placeholders without image tensor data.
#[derive(Debug, Clone, Copy)]
pub struct ImageTilingSummary<'a> {
    pub num_tiles: usize,
    pub tiles_per_image: &'a [usize],
    /// Local grids corresponding one-to-one with `tiles_per_image`.
    pub tile_grids: &'a [TileGrid],
    /// Thumbnail position as stored in the image tensor.
    ///
    /// This is the authoritative ordering: the thumbnail tile appears at this
    /// position within each image's tile slice of the tensor. Token expansion
    /// must use the same ordering so that token indices line up with tile
    /// (embedding) indices. Must match `TokenExpansionConfig::thumbnail_position`.
    pub thumbnail_position: ThumbnailPosition,
}

/// Resolved image tiling parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageTilingConfig {
    pub mode: TilingMode,
    pub tile_size: u32,
    /// Maximum local grid tiles; an enabled global thumbnail is additional.
    pub max_tiles: usize,
    pub aspect_ratios: Vec<TileGrid>,
    pub include_thumbnail: bool,
}

/// Pixel normalization selected by §35 metadata.
#[derive(Debug, Clone, PartialEq)]
pub enum Normalization {
    ZeroToOne,
    MeanStd { mean: [f32; 3], std: [f32; 3] },
}

/// Resolved image preprocessing parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct ImagePreprocessConfig {
    pub width: u32,
    pub height: u32,
    pub resize_mode: ResizeMode,
    pub interpolation: Interpolation,
    pub tiling: ImageTilingConfig,
    pub normalization: Normalization,
}

/// Replaces each image placeholder in a prompt with its image's tile token sequence.
///
/// Placeholders are matched to images in prompt order. The number of placeholder
/// occurrences must exactly match `tiling.tiles_per_image.len()`. The returned
/// token IDs are ready for the caller to pass to its scheduler/decoder; wiring
/// this function between tokenization and sequence-length/KV allocation is the
/// responsibility of the engine or server.
pub fn expand_image_placeholders(
    prompt_token_ids: &[i64],
    tiling: ImageTilingSummary<'_>,
    config: &TokenExpansionConfig,
) -> anyhow::Result<Vec<i64>> {
    validate_token_expansion(tiling, config)?;

    let placeholder_count = prompt_token_ids
        .iter()
        .filter(|token_id| **token_id == config.image_placeholder_token_id)
        .count();
    if placeholder_count != tiling.tiles_per_image.len() {
        anyhow::bail!(
            "prompt contains {placeholder_count} image placeholder(s), but preprocessing produced {} image(s)",
            tiling.tiles_per_image.len()
        );
    }

    let mut replacements = Vec::with_capacity(tiling.tile_grids.len());
    let mut replacement_tokens = 0usize;
    for grid in tiling.tile_grids {
        let replacement = expanded_image_tokens(*grid, tiling.thumbnail_position, config)?;
        replacement_tokens = replacement_tokens
            .checked_add(replacement.len())
            .context("expanded image token sequence is too large")?;
        replacements.push(replacement);
    }

    let output_len = prompt_token_ids
        .len()
        .checked_sub(placeholder_count)
        .and_then(|length| length.checked_add(replacement_tokens))
        .context("expanded prompt token sequence is too large")?;
    let mut expanded = Vec::new();
    expanded
        .try_reserve_exact(output_len)
        .context("failed to allocate expanded prompt token sequence")?;
    let mut image_index = 0usize;
    for token_id in prompt_token_ids {
        if *token_id == config.image_placeholder_token_id {
            expanded.extend_from_slice(&replacements[image_index]);
            image_index += 1;
        } else {
            expanded.push(*token_id);
        }
    }
    Ok(expanded)
}

fn validate_token_expansion(
    tiling: ImageTilingSummary<'_>,
    config: &TokenExpansionConfig,
) -> anyhow::Result<()> {
    if config.tokens_per_tile == 0 {
        anyhow::bail!("tokens_per_tile must be greater than zero");
    }
    if tiling.tiles_per_image.len() != tiling.tile_grids.len() {
        anyhow::bail!(
            "tiles_per_image has {} entries, but tile_grids has {}",
            tiling.tiles_per_image.len(),
            tiling.tile_grids.len()
        );
    }
    // The config thumbnail position must match the actual tensor layout so that
    // emitted token indices align with tile (embedding) indices in the tensor.
    if config.thumbnail_position != tiling.thumbnail_position {
        anyhow::bail!(
            "config thumbnail_position {:?} does not match tensor thumbnail_position {:?}; \
             token order must match the tile order stored in the image tensor",
            config.thumbnail_position,
            tiling.thumbnail_position,
        );
    }

    let thumbnail_tiles = usize::from(tiling.thumbnail_position != ThumbnailPosition::None);
    let mut total_tiles = 0usize;
    for (image_index, (&actual_tiles, grid)) in tiling
        .tiles_per_image
        .iter()
        .zip(tiling.tile_grids)
        .enumerate()
    {
        if grid.columns == 0 || grid.rows == 0 {
            anyhow::bail!("image {image_index} tile grid dimensions must be greater than zero");
        }
        let expected_tiles = grid
            .tile_count()?
            .checked_add(thumbnail_tiles)
            .context("image tile count is too large")?;
        if actual_tiles != expected_tiles {
            anyhow::bail!(
                "image {image_index} reports {actual_tiles} tile(s), but its {}x{} grid and thumbnail configuration require {expected_tiles}",
                grid.columns,
                grid.rows
            );
        }
        total_tiles = total_tiles
            .checked_add(actual_tiles)
            .context("total image tile count is too large")?;
    }
    if total_tiles != tiling.num_tiles {
        anyhow::bail!(
            "tiling summary reports {} total tile(s), but tiles_per_image sums to {total_tiles}",
            tiling.num_tiles
        );
    }
    Ok(())
}

fn expanded_image_tokens(
    grid: TileGrid,
    thumbnail_position: ThumbnailPosition,
    config: &TokenExpansionConfig,
) -> anyhow::Result<Vec<i64>> {
    let local_tiles = grid.tile_count()?;
    let thumbnail_tiles = usize::from(thumbnail_position != ThumbnailPosition::None);
    let separator_count = usize::from(config.column_separator_token_id.is_some())
        .checked_mul(local_tiles.saturating_sub(grid.rows as usize))
        .and_then(|count| {
            usize::from(config.row_separator_token_id.is_some())
                .checked_mul((grid.rows as usize).saturating_sub(1))
                .and_then(|rows| count.checked_add(rows))
        })
        .context("expanded image separator count is too large")?;
    let capacity = local_tiles
        .checked_add(thumbnail_tiles)
        .and_then(|tiles| tiles.checked_mul(config.tokens_per_tile))
        .and_then(|tokens| tokens.checked_add(separator_count))
        .context("expanded image token sequence is too large")?;
    let mut tokens = Vec::new();
    tokens
        .try_reserve_exact(capacity)
        .context("failed to allocate expanded image token sequence")?;

    let emit_tile = |tokens: &mut Vec<i64>| {
        tokens.extend(std::iter::repeat_n(
            config.image_token_id,
            config.tokens_per_tile,
        ));
    };
    if thumbnail_position == ThumbnailPosition::Prepend {
        emit_tile(&mut tokens);
    }
    for row in 0..grid.rows {
        for column in 0..grid.columns {
            emit_tile(&mut tokens);
            if column + 1 < grid.columns
                && let Some(separator) = config.column_separator_token_id
            {
                tokens.push(separator);
            }
        }
        if row + 1 < grid.rows
            && let Some(separator) = config.row_separator_token_id
        {
            tokens.push(separator);
        }
    }
    if thumbnail_position == ThumbnailPosition::Append {
        emit_tile(&mut tokens);
    }
    Ok(tokens)
}

/// Reusable image preprocessor resolved from a model input and §35 metadata.
#[derive(Debug, Clone)]
pub struct ImagePreprocessor {
    shape: Vec<i64>,
    layout: ImageLayout,
    config: ImagePreprocessConfig,
    program: ImageProgram,
}

#[derive(Debug, Deserialize)]
struct MetadataDocument {
    preprocessing: Option<PreprocessingMetadata>,
}

#[derive(Debug, Deserialize)]
struct PreprocessingMetadata {
    image: Option<ImageMetadata>,
}

#[derive(Debug, Deserialize)]
struct ImageMetadata {
    resize: Option<ResizeMetadata>,
    tiling: Option<TilingMetadata>,
    normalize: Option<NormalizeMetadata>,
    #[serde(default)]
    transforms: Vec<ImageTransformMetadata>,
    #[serde(default)]
    outputs: Vec<ImageOutputMetadata>,
}

#[derive(Debug, Deserialize)]
struct ResizeMetadata {
    mode: Option<String>,
    size: Option<ImageSize>,
    interpolation: Option<String>,
    crop: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ImageSize {
    Square(u32),
    Dimensions { width: u32, height: u32 },
}

#[derive(Debug, Deserialize)]
struct NormalizeMetadata {
    mean: [f32; 3],
    std: [f32; 3],
}

#[derive(Debug, Deserialize)]
struct TilingMetadata {
    mode: Option<String>,
    tile_size: Option<u32>,
    max_tiles: Option<usize>,
    aspect_ratios: Option<Vec<[u32; 2]>>,
    include_thumbnail: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct ImageTransformMetadata {
    op: String,
    size: Option<ImageSize>,
    mode: Option<String>,
    interpolation: Option<String>,
    scale: Option<f64>,
    mean: Option<Vec<f32>>,
    std: Option<Vec<f32>>,
    tile_size: Option<usize>,
    max_tiles: Option<usize>,
    include_thumbnail: Option<bool>,
    patch_size: Option<usize>,
    flatten: Option<bool>,
    pad_value: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct ImageOutputMetadata {
    name: String,
    content: String,
    dtype: String,
    pad_value: Option<f64>,
    optional: Option<bool>,
}

#[derive(Debug, Clone)]
struct ImageProgram {
    value_ops: Vec<ValueOp>,
    patch_size: Option<usize>,
    pad_value: Option<f64>,
    outputs: Vec<OutputSpec>,
}

#[derive(Debug, Clone)]
enum ValueOp {
    Rescale(f32),
    Normalize { mean: [f32; 3], std: [f32; 3] },
}

impl ImagePreprocessor {
    /// Resolves preprocessing from a model pixel input and optional metadata file.
    pub fn from_input_and_metadata(
        shape: &[i64],
        metadata_path: Option<&Path>,
    ) -> anyhow::Result<Self> {
        let document = metadata_path
            .map(std::fs::read_to_string)
            .transpose()
            .context("failed to read preprocessing metadata")?
            .map(|content| {
                serde_yaml::from_str::<MetadataDocument>(&content)
                    .context("failed to parse preprocessing metadata")
            })
            .transpose()?;
        Self::from_metadata_document(shape, document)
    }

    fn from_metadata_document(
        shape: &[i64],
        document: Option<MetadataDocument>,
    ) -> anyhow::Result<Self> {
        if shape.is_empty() {
            anyhow::bail!("vision pixel input shape must not be empty");
        }
        if shape
            .iter()
            .any(|dimension| *dimension == 0 || *dimension < -1)
        {
            anyhow::bail!("vision pixel input shape contains an invalid dimension: {shape:?}");
        }
        let metadata = document
            .and_then(|document| document.preprocessing)
            .and_then(|preprocessing| preprocessing.image);
        let is_typed_program = metadata
            .as_ref()
            .is_some_and(|image| !image.transforms.is_empty() || !image.outputs.is_empty());
        let (layout, model_width, model_height) = if shape.len() == 4 {
            let layout = match (shape[1], shape[3]) {
                (3, _) => ImageLayout::Nchw,
                (_, 3) => ImageLayout::Nhwc,
                _ if is_typed_program => ImageLayout::Nchw,
                _ => anyhow::bail!(
                    "vision input must declare an RGB channel dimension, but the model declares {shape:?}"
                ),
            };
            let (height, width) = match layout {
                ImageLayout::Nchw => (shape[2], shape[3]),
                ImageLayout::Nhwc => (shape[1], shape[2]),
            };
            (layout, width, height)
        } else if is_typed_program {
            (ImageLayout::Nchw, -1, -1)
        } else {
            anyhow::bail!(
                "legacy image preprocessing requires a rank-4 vision input, but the model declares {shape:?}; packed inputs require preprocessing.image.transforms and outputs"
            );
        };
        let (config, program) = if is_typed_program {
            typed_program_from_metadata(
                metadata.context("typed image preprocessing metadata is missing")?,
                model_width,
                model_height,
            )?
        } else {
            let config = preprocessing_from_metadata(metadata, model_width, model_height)?;
            let program = legacy_program(&config)?;
            (config, program)
        };
        let mut resolved_shape = shape.to_vec();
        if resolved_shape.len() == 4 {
            match layout {
                ImageLayout::Nchw => {
                    resolved_shape[2] = i64::from(config.height);
                    resolved_shape[3] = i64::from(config.width);
                }
                ImageLayout::Nhwc => {
                    resolved_shape[1] = i64::from(config.height);
                    resolved_shape[2] = i64::from(config.width);
                }
            }
        }
        Ok(Self {
            shape: resolved_shape,
            layout,
            config,
            program,
        })
    }

    /// Resolves preprocessing using model dimensions and default §35 behavior.
    pub fn from_input(shape: &[i64]) -> anyhow::Result<Self> {
        Self::from_input_and_metadata(shape, None)
    }

    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    pub fn layout(&self) -> ImageLayout {
        self.layout
    }

    pub fn config(&self) -> &ImagePreprocessConfig {
        &self.config
    }

    /// Decodes encoded images and preprocesses them into a named tensor bundle.
    pub fn preprocess_encoded<I, B>(&self, images: I) -> anyhow::Result<ImageTensorBundle>
    where
        I: IntoIterator<Item = B>,
        B: AsRef<[u8]>,
    {
        let images = images
            .into_iter()
            .map(|bytes| image::load_from_memory(bytes.as_ref()).context("failed to decode image"))
            .collect::<anyhow::Result<Vec<_>>>()?;
        self.preprocess(&images)
    }

    /// Preprocesses decoded images into a named typed tensor bundle.
    pub fn preprocess(&self, images: &[DynamicImage]) -> anyhow::Result<ImageTensorBundle> {
        if images.is_empty() {
            anyhow::bail!("at least one image is required");
        }
        let width = self.config.width as usize;
        let height = self.config.height as usize;
        let mut prepared = Vec::with_capacity(images.len());
        for image in images {
            let (grid, image_tiles) = tile_image(image, &self.config)?;
            let tiles = image_tiles
                .iter()
                .map(|tile| normalize_tile(tile, width, height, &self.program.value_ops))
                .collect();
            prepared.push(PreparedImage {
                original_size: (image.width(), image.height()),
                tile_grid: grid,
                tiles,
            });
        }
        let thumbnail_position = if self.config.tiling.include_thumbnail {
            ThumbnailPosition::Prepend
        } else {
            ThumbnailPosition::None
        };
        packed::build_bundle(
            prepared,
            &PackSpec {
                width,
                height,
                layout: self.layout,
                patch_size: self.program.patch_size,
                pad_value: self.program.pad_value,
                outputs: self.program.outputs.clone(),
                declared_pixel_shape: self.shape.clone(),
            },
            thumbnail_position,
        )
    }
}

fn legacy_program(config: &ImagePreprocessConfig) -> anyhow::Result<ImageProgram> {
    let value_ops = match &config.normalization {
        Normalization::ZeroToOne => vec![ValueOp::Rescale(1.0 / 255.0)],
        Normalization::MeanStd { mean, std } => vec![
            ValueOp::Rescale(1.0 / 255.0),
            ValueOp::Normalize {
                mean: *mean,
                std: *std,
            },
        ],
    };
    Ok(ImageProgram {
        value_ops,
        patch_size: None,
        pad_value: None,
        outputs: vec![OutputSpec {
            name: "pixels".to_owned(),
            content: "pixels".to_owned(),
            dtype: ImageTensorDType::Fp32,
            pad_value: None,
            optional: false,
        }],
    })
}

fn typed_program_from_metadata(
    metadata: ImageMetadata,
    model_width: i64,
    model_height: i64,
) -> anyhow::Result<(ImagePreprocessConfig, ImageProgram)> {
    if metadata.transforms.is_empty() {
        anyhow::bail!(
            "preprocessing.image.transforms must not be empty when typed image outputs are declared"
        );
    }
    if metadata.outputs.is_empty() {
        anyhow::bail!(
            "preprocessing.image.outputs must not be empty when typed image transforms are declared"
        );
    }
    if metadata.resize.is_some() || metadata.tiling.is_some() || metadata.normalize.is_some() {
        anyhow::bail!(
            "preprocessing.image cannot mix legacy resize/tiling/normalize fields with typed transforms"
        );
    }

    let mut resize = None;
    let mut tiling = None;
    let mut value_ops = Vec::new();
    let mut patch_size = None;
    let mut pad_value = None;
    let mut decoded = false;
    let mut patchified = false;
    let mut padded = false;
    for transform in metadata.transforms {
        match transform.op.as_str() {
            "decode_rgb" => {
                if decoded || resize.is_some() || !value_ops.is_empty() || patchified || padded {
                    anyhow::bail!("decode_rgb must be the first image transform");
                }
                decoded = true;
            }
            "resize" => {
                if resize.is_some()
                    || tiling.is_some()
                    || !value_ops.is_empty()
                    || patchified
                    || padded
                {
                    anyhow::bail!(
                        "resize must occur once and before tile, rescale, normalize, patchify, or pad"
                    );
                }
                let size = transform
                    .size
                    .context("image resize transform requires size metadata")?;
                let mode = match transform.mode.as_deref().unwrap_or("stretch") {
                    "stretch" | "fixed" | "fixed_size" => ResizeMode::Fixed,
                    "crop" | "shortest_edge" | "shortest_edge_center_crop" => {
                        ResizeMode::ShortestEdgeCenterCrop
                    }
                    "pad" | "longest_edge_pad" => ResizeMode::LongestEdgePad,
                    other => anyhow::bail!(
                        "unsupported image resize transform mode '{other}'; expected stretch, crop, or pad"
                    ),
                };
                let interpolation = parse_interpolation(transform.interpolation.as_deref())?;
                resize = Some((size, mode, interpolation));
            }
            "rescale" => {
                if patchified || padded {
                    anyhow::bail!("rescale must occur before patchify or pad");
                }
                let scale = transform
                    .scale
                    .context("image rescale transform requires scale metadata")?;
                let scale = scale as f32;
                if !scale.is_finite() {
                    anyhow::bail!("image rescale scale must be finite and representable as fp32");
                }
                value_ops.push(ValueOp::Rescale(scale));
            }
            "normalize" => {
                if patchified || padded {
                    anyhow::bail!("normalize must occur before patchify or pad");
                }
                let mean = channel_values("mean", transform.mean)?;
                let std = channel_values("std", transform.std)?;
                if mean.iter().any(|value| !value.is_finite())
                    || std.iter().any(|value| !value.is_finite() || *value <= 0.0)
                {
                    anyhow::bail!(
                        "image normalization mean/std values must be finite and std must be greater than zero"
                    );
                }
                value_ops.push(ValueOp::Normalize { mean, std });
            }
            "tile" => {
                if tiling.is_some() || !value_ops.is_empty() || patchified || padded {
                    anyhow::bail!(
                        "tile must occur once and before rescale, normalize, patchify, or pad"
                    );
                }
                let tile_size = transform
                    .tile_size
                    .context("image tile transform requires tile_size metadata")?;
                if tile_size == 0 {
                    anyhow::bail!("image tile transform tile_size must be greater than zero");
                }
                let tile_size = u32::try_from(tile_size).context("image tile_size is too large")?;
                let max_tiles = transform.max_tiles.unwrap_or(6);
                if max_tiles == 0 {
                    anyhow::bail!("image tile transform max_tiles must be greater than zero");
                }
                tiling = Some(ImageTilingConfig {
                    mode: TilingMode::DynamicAnyres,
                    tile_size,
                    max_tiles,
                    aspect_ratios: default_anyres_grids(),
                    include_thumbnail: transform.include_thumbnail.unwrap_or(false),
                });
            }
            "patchify" => {
                if patchified || padded {
                    anyhow::bail!("patchify must occur once and before pad");
                }
                if transform.flatten == Some(false) {
                    anyhow::bail!(
                        "image patchify flatten=false is not supported; declare flatten=true for packed patch outputs"
                    );
                }
                let size = transform
                    .patch_size
                    .context("image patchify transform requires patch_size metadata")?;
                if size == 0 {
                    anyhow::bail!("image patchify patch_size must be greater than zero");
                }
                patch_size = Some(size);
                patchified = true;
            }
            "pad" => {
                if !patchified {
                    anyhow::bail!("image pad transform requires a preceding patchify transform");
                }
                if padded {
                    anyhow::bail!("image pad transform may occur only once");
                }
                let value = transform.pad_value.unwrap_or(0.0);
                if !value.is_finite() {
                    anyhow::bail!("image pad transform pad_value must be finite");
                }
                pad_value = Some(value);
                padded = true;
            }
            other => anyhow::bail!(
                "unsupported required image transform '{other}'; supported operations are decode_rgb, resize, rescale, normalize, tile, patchify, and pad"
            ),
        }
    }

    let (width, height, resize_mode, interpolation) = match resize {
        Some((ImageSize::Square(size), mode, interpolation)) => (size, size, mode, interpolation),
        Some((ImageSize::Dimensions { width, height }, mode, interpolation)) => {
            (width, height, mode, interpolation)
        }
        None => (
            resolve_dimension("width", model_width, None)?,
            resolve_dimension("height", model_height, None)?,
            ResizeMode::Fixed,
            Interpolation::Bicubic,
        ),
    };
    if width == 0 || height == 0 {
        anyhow::bail!("image resize dimensions must be greater than zero");
    }
    let tiling = match tiling {
        Some(tiling) => {
            if tiling.tile_size != width || tiling.tile_size != height {
                anyhow::bail!(
                    "image tile_size {} must match resized dimensions {width}x{height}",
                    tiling.tile_size
                );
            }
            tiling
        }
        None => ImageTilingConfig {
            mode: TilingMode::None,
            tile_size: width,
            max_tiles: 1,
            aspect_ratios: vec![TileGrid {
                columns: 1,
                rows: 1,
            }],
            include_thumbnail: false,
        },
    };
    let outputs = metadata
        .outputs
        .into_iter()
        .map(|output| {
            Ok(OutputSpec {
                name: output.name,
                content: output.content,
                dtype: ImageTensorDType::parse(&output.dtype)?,
                pad_value: output.pad_value,
                optional: output.optional.unwrap_or(false),
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok((
        ImagePreprocessConfig {
            width,
            height,
            resize_mode,
            interpolation,
            tiling,
            // Typed programs execute value transforms in declared order.
            normalization: Normalization::ZeroToOne,
        },
        ImageProgram {
            value_ops,
            patch_size,
            pad_value,
            outputs,
        },
    ))
}

fn parse_interpolation(value: Option<&str>) -> anyhow::Result<Interpolation> {
    match value.unwrap_or("bicubic") {
        "bicubic" => Ok(Interpolation::Bicubic),
        "bilinear" => Ok(Interpolation::Bilinear),
        "lanczos" | "lanczos3" => Ok(Interpolation::Lanczos3),
        other => anyhow::bail!("unsupported image interpolation '{other}'"),
    }
}

fn channel_values(name: &str, values: Option<Vec<f32>>) -> anyhow::Result<[f32; 3]> {
    let values = values.with_context(|| format!("image normalize transform requires {name}"))?;
    values.try_into().map_err(|values: Vec<f32>| {
        anyhow::anyhow!(
            "image normalize transform {name} must contain 3 RGB values, got {}",
            values.len()
        )
    })
}

fn preprocessing_from_metadata(
    metadata: Option<ImageMetadata>,
    model_width: i64,
    model_height: i64,
) -> anyhow::Result<ImagePreprocessConfig> {
    let declared_size = metadata
        .as_ref()
        .and_then(|image| image.resize.as_ref())
        .and_then(|resize| resize.size.as_ref())
        .map(|size| match size {
            ImageSize::Square(size) => (*size, *size),
            ImageSize::Dimensions { width, height } => (*width, *height),
        });
    let width = resolve_dimension("width", model_width, declared_size.map(|size| size.0))?;
    let height = resolve_dimension("height", model_height, declared_size.map(|size| size.1))?;

    let resize = metadata.as_ref().and_then(|image| image.resize.as_ref());
    let mode = resize.and_then(|resize| resize.mode.as_deref());
    let crop = resize.and_then(|resize| resize.crop.as_deref());
    let resize_mode = match mode.unwrap_or("shortest_edge") {
        "shortest_edge" => match crop.unwrap_or("center") {
            "center" | "center_crop" => ResizeMode::ShortestEdgeCenterCrop,
            other => anyhow::bail!("unsupported shortest_edge crop mode '{other}'"),
        },
        "fixed" | "fixed_size" => {
            if crop.is_some_and(|crop| crop != "none") {
                anyhow::bail!("fixed resize only supports crop mode 'none'");
            }
            ResizeMode::Fixed
        }
        "longest_edge_pad" => {
            if crop.is_some_and(|crop| crop != "none") {
                anyhow::bail!("longest_edge_pad only supports crop mode 'none'");
            }
            ResizeMode::LongestEdgePad
        }
        other => anyhow::bail!("unsupported image resize mode '{other}'"),
    };
    let interpolation = match resize
        .and_then(|resize| resize.interpolation.as_deref())
        .unwrap_or("bicubic")
    {
        "bicubic" => Interpolation::Bicubic,
        "bilinear" => Interpolation::Bilinear,
        "lanczos" | "lanczos3" => Interpolation::Lanczos3,
        other => anyhow::bail!("unsupported image interpolation '{other}'"),
    };
    let tiling = tiling_from_metadata(
        metadata.as_ref().and_then(|image| image.tiling.as_ref()),
        width,
        height,
    )?;
    let normalization = match metadata.and_then(|image| image.normalize) {
        Some(normalize) => {
            if normalize.std.iter().any(|value| *value <= 0.0) {
                anyhow::bail!("image normalization std values must be greater than zero");
            }
            Normalization::MeanStd {
                mean: normalize.mean,
                std: normalize.std,
            }
        }
        None => Normalization::ZeroToOne,
    };

    Ok(ImagePreprocessConfig {
        width,
        height,
        resize_mode,
        interpolation,
        tiling,
        normalization,
    })
}

fn tiling_from_metadata(
    metadata: Option<&TilingMetadata>,
    width: u32,
    height: u32,
) -> anyhow::Result<ImageTilingConfig> {
    let mode = match metadata.and_then(|tiling| tiling.mode.as_deref()) {
        None | Some("none") => TilingMode::None,
        Some("fixed_grid") => TilingMode::FixedGrid,
        Some("dynamic_anyres") => TilingMode::DynamicAnyres,
        Some(other) => anyhow::bail!("unsupported image tiling mode '{other}'"),
    };
    if mode == TilingMode::None {
        return Ok(ImageTilingConfig {
            mode,
            tile_size: width,
            max_tiles: 1,
            aspect_ratios: vec![TileGrid {
                columns: 1,
                rows: 1,
            }],
            include_thumbnail: false,
        });
    }

    let tile_size = match metadata.and_then(|tiling| tiling.tile_size) {
        Some(0) => anyhow::bail!("image tiling tile_size must be greater than zero"),
        Some(tile_size) => tile_size,
        None if width == height => width,
        None => anyhow::bail!("non-square tiled image inputs require tiling.tile_size metadata"),
    };
    if width != tile_size || height != tile_size {
        anyhow::bail!(
            "tiling tile_size {tile_size} must match model tile dimensions {width}x{height}"
        );
    }
    let max_tiles = metadata.and_then(|tiling| tiling.max_tiles).unwrap_or(6);
    if max_tiles == 0 {
        anyhow::bail!("image tiling max_tiles must be greater than zero");
    }

    let configured_ratios = metadata.and_then(|tiling| tiling.aspect_ratios.as_ref());
    let aspect_ratios = match (mode, configured_ratios) {
        (TilingMode::FixedGrid, None) => vec![TileGrid {
            columns: 1,
            rows: 1,
        }],
        (TilingMode::DynamicAnyres, None) => default_anyres_grids(),
        (_, Some(ratios)) => ratios
            .iter()
            .map(|[columns, rows]| TileGrid {
                columns: *columns,
                rows: *rows,
            })
            .collect(),
        (TilingMode::None, _) => unreachable!("none returned above"),
    };
    if aspect_ratios.is_empty() {
        anyhow::bail!("image tiling aspect_ratios must not be empty");
    }
    for grid in &aspect_ratios {
        if grid.columns == 0 || grid.rows == 0 {
            anyhow::bail!("image tiling aspect ratios must contain positive grid dimensions");
        }
        grid.tile_count()?;
    }
    if mode == TilingMode::FixedGrid {
        if aspect_ratios.len() != 1 {
            anyhow::bail!("fixed_grid tiling requires exactly one aspect_ratios entry");
        }
        let count = aspect_ratios[0].tile_count()?;
        if count > max_tiles {
            anyhow::bail!(
                "fixed_grid produces {count} local tiles, exceeding max_tiles {max_tiles}"
            );
        }
    } else if !aspect_ratios
        .iter()
        .any(|grid| grid.tile_count().is_ok_and(|count| count <= max_tiles))
    {
        anyhow::bail!("no dynamic_anyres aspect ratio fits max_tiles {max_tiles}");
    }

    Ok(ImageTilingConfig {
        mode,
        tile_size,
        max_tiles,
        aspect_ratios,
        include_thumbnail: metadata
            .and_then(|tiling| tiling.include_thumbnail)
            .unwrap_or(true),
    })
}

fn default_anyres_grids() -> Vec<TileGrid> {
    [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (2, 2)]
        .into_iter()
        .map(|(columns, rows)| TileGrid { columns, rows })
        .collect()
}

fn resolve_dimension(name: &str, model: i64, configured: Option<u32>) -> anyhow::Result<u32> {
    if model == 0 || model < -1 {
        anyhow::bail!("vision input has invalid {name} dimension {model}");
    }
    match (model, configured) {
        (model, Some(configured)) if model > 0 && model as u32 != configured => anyhow::bail!(
            "preprocessing {name} {configured} does not match model input {name} {model}"
        ),
        (_, Some(0)) => anyhow::bail!("preprocessing {name} must be greater than zero"),
        (_, Some(configured)) => Ok(configured),
        (model, None) if model > 0 => Ok(model as u32),
        (_, None) => anyhow::bail!(
            "dynamic vision input {name} requires preprocessing.image.resize.size metadata"
        ),
    }
}

fn tile_image(
    image: &DynamicImage,
    config: &ImagePreprocessConfig,
) -> anyhow::Result<(TileGrid, Vec<RgbImage>)> {
    match config.tiling.mode {
        TilingMode::None => Ok((
            TileGrid {
                columns: 1,
                rows: 1,
            },
            vec![resize_image(image, config)],
        )),
        TilingMode::FixedGrid => {
            let grid = config.tiling.aspect_ratios[0];
            Ok((grid, tiled_image_for_grid(image, config, grid)?))
        }
        TilingMode::DynamicAnyres => {
            let grid = select_best_grid(
                image.width(),
                image.height(),
                config.tiling.tile_size,
                config.tiling.max_tiles,
                &config.tiling.aspect_ratios,
            )?;
            Ok((grid, tiled_image_for_grid(image, config, grid)?))
        }
    }
}

fn tiled_image_for_grid(
    image: &DynamicImage,
    config: &ImagePreprocessConfig,
    grid: TileGrid,
) -> anyhow::Result<Vec<RgbImage>> {
    let tile_size = config.tiling.tile_size;
    let width = grid
        .columns
        .checked_mul(tile_size)
        .context("tiled image width is too large")?;
    let height = grid
        .rows
        .checked_mul(tile_size)
        .context("tiled image height is too large")?;
    let resized = resize_image_to(image, config, width, height);
    let local_count = grid.tile_count()?;
    let mut tiles = Vec::with_capacity(local_count + usize::from(config.tiling.include_thumbnail));
    // Encoder conventions place the global view before row-major local tiles.
    if config.tiling.include_thumbnail {
        tiles.push(resize_image_to(image, config, tile_size, tile_size));
    }
    for row in 0..grid.rows {
        for column in 0..grid.columns {
            tiles.push(
                image::imageops::crop_imm(
                    &resized,
                    column * tile_size,
                    row * tile_size,
                    tile_size,
                    tile_size,
                )
                .to_image(),
            );
        }
    }
    Ok(tiles)
}

/// Selects the LLaVA-style best resolution.
///
/// Candidates exceeding `max_tiles` are ignored. Remaining candidates maximize
/// effective source pixels after aspect-preserving fit, then minimize padded or
/// cropped canvas pixels. Configuration order breaks any remaining tie.
fn select_best_grid(
    image_width: u32,
    image_height: u32,
    tile_size: u32,
    max_tiles: usize,
    grids: &[TileGrid],
) -> anyhow::Result<TileGrid> {
    let original_area = u64::from(image_width) * u64::from(image_height);
    let mut best = None;
    for grid in grids.iter().copied() {
        let Some((effective, wasted)) = (|| {
            let tile_count = grid.tile_count().ok()?;
            if tile_count > max_tiles {
                return None;
            }
            let candidate_width = grid.columns.checked_mul(tile_size)?;
            let candidate_height = grid.rows.checked_mul(tile_size)?;
            let scale = (candidate_width as f64 / image_width as f64)
                .min(candidate_height as f64 / image_height as f64);
            let fitted_width = (image_width as f64 * scale).floor() as u64;
            let fitted_height = (image_height as f64 * scale).floor() as u64;
            let effective = (fitted_width * fitted_height).min(original_area);
            let candidate_area = u64::from(candidate_width) * u64::from(candidate_height);
            let wasted = candidate_area.saturating_sub(effective);
            Some((effective, wasted))
        })() else {
            continue;
        };
        if best.is_none_or(|(_, best_effective, best_wasted)| {
            effective > best_effective || (effective == best_effective && wasted < best_wasted)
        }) {
            best = Some((grid, effective, wasted));
        }
    }
    best.map(|(grid, _, _)| grid)
        .context("no image tiling aspect ratio fits max_tiles")
}

fn resize_image(image: &DynamicImage, config: &ImagePreprocessConfig) -> RgbImage {
    resize_image_to(image, config, config.width, config.height)
}

fn resize_image_to(
    image: &DynamicImage,
    config: &ImagePreprocessConfig,
    width: u32,
    height: u32,
) -> RgbImage {
    let rgb = image.to_rgb8();
    let filter = match config.interpolation {
        Interpolation::Bicubic => FilterType::CatmullRom,
        Interpolation::Bilinear => FilterType::Triangle,
        Interpolation::Lanczos3 => FilterType::Lanczos3,
    };
    match config.resize_mode {
        ResizeMode::Fixed => image::imageops::resize(&rgb, width, height, filter),
        ResizeMode::ShortestEdgeCenterCrop => {
            let scale =
                (width as f64 / rgb.width() as f64).max(height as f64 / rgb.height() as f64);
            let resized_width = ((rgb.width() as f64 * scale).round() as u32).max(width);
            let resized_height = ((rgb.height() as f64 * scale).round() as u32).max(height);
            let resized = image::imageops::resize(&rgb, resized_width, resized_height, filter);
            image::imageops::crop_imm(
                &resized,
                (resized_width - width) / 2,
                (resized_height - height) / 2,
                width,
                height,
            )
            .to_image()
        }
        ResizeMode::LongestEdgePad => {
            let scale =
                (width as f64 / rgb.width() as f64).min(height as f64 / rgb.height() as f64);
            let resized_width = ((rgb.width() as f64 * scale).round() as u32).clamp(1, width);
            let resized_height = ((rgb.height() as f64 * scale).round() as u32).clamp(1, height);
            let resized = image::imageops::resize(&rgb, resized_width, resized_height, filter);
            let mut padded = RgbImage::from_pixel(width, height, Rgb([0, 0, 0]));
            image::imageops::replace(
                &mut padded,
                &resized,
                i64::from((width - resized_width) / 2),
                i64::from((height - resized_height) / 2),
            );
            padded
        }
    }
}

fn normalize_tile(
    image: &RgbImage,
    width: usize,
    height: usize,
    operations: &[ValueOp],
) -> Vec<f32> {
    let mut values = Vec::with_capacity(CHANNELS * width * height);
    for channel in 0..CHANNELS {
        values.extend(image.pixels().map(|pixel| {
            operations.iter().fold(
                f32::from(pixel[channel]),
                |value, operation| match operation {
                    ValueOp::Rescale(scale) => value * scale,
                    ValueOp::Normalize { mean, std } => (value - mean[channel]) / std[channel],
                },
            )
        }));
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token_expansion_config() -> TokenExpansionConfig {
        TokenExpansionConfig {
            image_placeholder_token_id: 99,
            image_token_id: 7,
            tokens_per_tile: 2,
            thumbnail_position: ThumbnailPosition::None,
            row_separator_token_id: None,
            column_separator_token_id: None,
        }
    }

    #[test]
    fn expands_single_untiled_image_placeholder() {
        let config = token_expansion_config();
        let tiles_per_image = [1];
        let grids = [TileGrid {
            columns: 1,
            rows: 1,
        }];

        let expanded = expand_image_placeholders(
            &[1, 99, 2],
            ImageTilingSummary {
                num_tiles: 1,
                tiles_per_image: &tiles_per_image,
                tile_grids: &grids,
                thumbnail_position: ThumbnailPosition::None,
            },
            &config,
        )
        .unwrap();

        assert_eq!(expanded, [1, 7, 7, 2]);
    }

    #[test]
    fn expands_single_image_local_tiles_in_row_major_order() {
        let config = token_expansion_config();
        let tiles_per_image = [6];
        let grids = [TileGrid {
            columns: 3,
            rows: 2,
        }];

        let expanded = expand_image_placeholders(
            &[99],
            ImageTilingSummary {
                num_tiles: 6,
                tiles_per_image: &tiles_per_image,
                tile_grids: &grids,
                thumbnail_position: ThumbnailPosition::None,
            },
            &config,
        )
        .unwrap();

        assert_eq!(expanded, [7; 12]);
    }

    #[test]
    fn expands_tiles_with_appended_global_thumbnail() {
        let mut config = token_expansion_config();
        config.thumbnail_position = ThumbnailPosition::Append;
        config.column_separator_token_id = Some(8);
        let tiles_per_image = [3];
        let grids = [TileGrid {
            columns: 2,
            rows: 1,
        }];

        // tiling.thumbnail_position must match config; here both say Append so
        // that this test exercises the Append code path in expanded_image_tokens.
        let expanded = expand_image_placeholders(
            &[99],
            ImageTilingSummary {
                num_tiles: 3,
                tiles_per_image: &tiles_per_image,
                tile_grids: &grids,
                thumbnail_position: ThumbnailPosition::Append,
            },
            &config,
        )
        .unwrap();

        assert_eq!(expanded, [7, 7, 8, 7, 7, 7, 7]);
    }

    #[test]
    fn inserts_column_and_row_separators_between_local_tiles() {
        let mut config = token_expansion_config();
        config.tokens_per_tile = 1;
        config.thumbnail_position = ThumbnailPosition::Prepend;
        config.column_separator_token_id = Some(8);
        config.row_separator_token_id = Some(9);
        let tiles_per_image = [5];
        let grids = [TileGrid {
            columns: 2,
            rows: 2,
        }];

        let expanded = expand_image_placeholders(
            &[99],
            ImageTilingSummary {
                num_tiles: 5,
                tiles_per_image: &tiles_per_image,
                tile_grids: &grids,
                thumbnail_position: ThumbnailPosition::Prepend,
            },
            &config,
        )
        .unwrap();

        assert_eq!(expanded, [7, 7, 8, 7, 9, 7, 8, 7]);
    }

    #[test]
    fn matches_multiple_placeholders_to_images_in_prompt_order() {
        let mut config = token_expansion_config();
        config.tokens_per_tile = 1;
        let tiles_per_image = [2, 3];
        let grids = [
            TileGrid {
                columns: 2,
                rows: 1,
            },
            TileGrid {
                columns: 1,
                rows: 3,
            },
        ];

        let expanded = expand_image_placeholders(
            &[10, 99, 11, 99, 12],
            ImageTilingSummary {
                num_tiles: 5,
                tiles_per_image: &tiles_per_image,
                tile_grids: &grids,
                thumbnail_position: ThumbnailPosition::None,
            },
            &config,
        )
        .unwrap();

        assert_eq!(expanded, [10, 7, 7, 11, 7, 7, 7, 12]);
    }

    #[test]
    fn rejects_inconsistent_token_expansion_inputs() {
        let grids = [TileGrid {
            columns: 2,
            rows: 1,
        }];

        let mut config = token_expansion_config();
        config.tokens_per_tile = 0;
        let error = expand_image_placeholders(
            &[99],
            ImageTilingSummary {
                num_tiles: 2,
                tiles_per_image: &[2],
                tile_grids: &grids,
                thumbnail_position: ThumbnailPosition::None,
            },
            &config,
        )
        .unwrap_err();
        assert!(error.to_string().contains("tokens_per_tile"));

        config.tokens_per_tile = 1;
        let error = expand_image_placeholders(
            &[99, 99],
            ImageTilingSummary {
                num_tiles: 2,
                tiles_per_image: &[2],
                tile_grids: &grids,
                thumbnail_position: ThumbnailPosition::None,
            },
            &config,
        )
        .unwrap_err();
        assert!(error.to_string().contains("2 image placeholder"));

        let error = expand_image_placeholders(
            &[99],
            ImageTilingSummary {
                num_tiles: 3,
                tiles_per_image: &[2],
                tile_grids: &grids,
                thumbnail_position: ThumbnailPosition::None,
            },
            &config,
        )
        .unwrap_err();
        assert!(error.to_string().contains("reports 3 total tile"));

        let error = expand_image_placeholders(
            &[99],
            ImageTilingSummary {
                num_tiles: 1,
                tiles_per_image: &[1],
                tile_grids: &grids,
                thumbnail_position: ThumbnailPosition::None,
            },
            &config,
        )
        .unwrap_err();
        assert!(error.to_string().contains("require 2"));
    }

    #[test]
    fn bicubic_shortest_edge_resize_center_crops_to_target_dimensions() {
        let config = ImagePreprocessConfig {
            width: 4,
            height: 4,
            resize_mode: ResizeMode::ShortestEdgeCenterCrop,
            interpolation: Interpolation::Bicubic,
            tiling: ImageTilingConfig {
                mode: TilingMode::None,
                tile_size: 4,
                max_tiles: 1,
                aspect_ratios: vec![TileGrid {
                    columns: 1,
                    rows: 1,
                }],
                include_thumbnail: false,
            },
            normalization: Normalization::ZeroToOne,
        };
        let image = DynamicImage::ImageRgb8(RgbImage::from_fn(12, 6, |x, _| {
            if x < 6 {
                Rgb([255, 0, 0])
            } else {
                Rgb([0, 0, 255])
            }
        }));
        assert_eq!(resize_image(&image, &config).dimensions(), (4, 4));
    }

    #[test]
    fn clip_mean_std_normalization_matches_known_pixel() {
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(1, 1, Rgb([255, 128, 0])));
        let preprocessor = ImagePreprocessor {
            shape: vec![1, 3, 1, 1],
            layout: ImageLayout::Nchw,
            config: ImagePreprocessConfig {
                width: 1,
                height: 1,
                resize_mode: ResizeMode::Fixed,
                interpolation: Interpolation::Bicubic,
                tiling: ImageTilingConfig {
                    mode: TilingMode::None,
                    tile_size: 1,
                    max_tiles: 1,
                    aspect_ratios: vec![TileGrid {
                        columns: 1,
                        rows: 1,
                    }],
                    include_thumbnail: false,
                },
                normalization: Normalization::MeanStd {
                    mean: [0.48145466, 0.4578275, 0.40821073],
                    std: [0.26862954, 0.261_302_6, 0.275_777_1],
                },
            },
            program: ImageProgram {
                value_ops: vec![
                    ValueOp::Rescale(1.0 / 255.0),
                    ValueOp::Normalize {
                        mean: [0.48145466, 0.4578275, 0.40821073],
                        std: [0.26862954, 0.261_302_6, 0.275_777_1],
                    },
                ],
                patch_size: None,
                pad_value: None,
                outputs: vec![OutputSpec {
                    name: "pixels".to_owned(),
                    content: "pixels".to_owned(),
                    dtype: ImageTensorDType::Fp32,
                    pad_value: None,
                    optional: false,
                }],
            },
        };
        let tensor = preprocessor.preprocess(&[image]).unwrap();
        let expected = [
            (1.0 - 0.48145466) / 0.26862954,
            (128.0 / 255.0 - 0.4578275) / 0.261_302_6,
            (0.0 - 0.40821073) / 0.275_777_1,
        ];
        let pixels = tensor.tensor_by_content("pixels").unwrap();
        let actual = pixels.data.as_f32_slice().unwrap();
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn metadata_selects_target_resize_and_normalization() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/image_preprocessing.yaml");
        let preprocessor =
            ImagePreprocessor::from_input_and_metadata(&[-1, 3, -1, -1], Some(&path)).unwrap();
        assert_eq!(preprocessor.shape(), &[-1, 3, 2, 2]);
        assert_eq!(preprocessor.config().resize_mode, ResizeMode::Fixed);
        assert_eq!(preprocessor.config().interpolation, Interpolation::Bicubic);
        assert_eq!(
            preprocessor.config().normalization,
            Normalization::MeanStd {
                mean: [0.1, 0.2, 0.3],
                std: [0.4, 0.5, 0.6],
            }
        );
        assert_eq!(preprocessor.config().tiling.mode, TilingMode::None);
    }

    #[test]
    fn metadata_selects_dynamic_anyres_tiling() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/image_tiling.yaml");
        let preprocessor =
            ImagePreprocessor::from_input_and_metadata(&[-1, 3, 2, 2], Some(&path)).unwrap();

        assert_eq!(
            preprocessor.config().tiling,
            ImageTilingConfig {
                mode: TilingMode::DynamicAnyres,
                tile_size: 2,
                max_tiles: 4,
                aspect_ratios: vec![
                    TileGrid {
                        columns: 1,
                        rows: 1
                    },
                    TileGrid {
                        columns: 2,
                        rows: 1
                    },
                    TileGrid {
                        columns: 1,
                        rows: 2
                    },
                    TileGrid {
                        columns: 2,
                        rows: 2
                    },
                ],
                include_thumbnail: true,
            }
        );
    }

    #[test]
    fn metadata_selects_fixed_grid_tiling() {
        let document = serde_yaml::from_str::<MetadataDocument>(
            r#"
preprocessing:
  image:
    resize:
      mode: fixed
      size: 2
      crop: none
    tiling:
      mode: fixed_grid
      tile_size: 2
      max_tiles: 6
      aspect_ratios: [[3, 2]]
"#,
        )
        .unwrap();
        let config = preprocessing_from_metadata(
            document
                .preprocessing
                .and_then(|preprocessing| preprocessing.image),
            2,
            2,
        )
        .unwrap();

        assert_eq!(config.tiling.mode, TilingMode::FixedGrid);
        assert_eq!(
            config.tiling.aspect_ratios,
            [TileGrid {
                columns: 3,
                rows: 2
            }]
        );
        assert!(config.tiling.include_thumbnail);
    }

    #[test]
    fn missing_metadata_uses_bicubic_center_crop_and_zero_to_one() {
        let preprocessor = ImagePreprocessor::from_input(&[1, 3, 4, 4]).unwrap();
        assert_eq!(
            (preprocessor.config().width, preprocessor.config().height),
            (4, 4)
        );
        assert_eq!(
            preprocessor.config().resize_mode,
            ResizeMode::ShortestEdgeCenterCrop
        );
        assert_eq!(
            preprocessor.config().normalization,
            Normalization::ZeroToOne
        );
    }

    fn tiled_preprocessor(
        mode: TilingMode,
        grids: Vec<TileGrid>,
        max_tiles: usize,
    ) -> ImagePreprocessor {
        ImagePreprocessor {
            shape: vec![-1, 3, 2, 2],
            layout: ImageLayout::Nchw,
            config: ImagePreprocessConfig {
                width: 2,
                height: 2,
                resize_mode: ResizeMode::Fixed,
                interpolation: Interpolation::Bicubic,
                tiling: ImageTilingConfig {
                    mode,
                    tile_size: 2,
                    max_tiles,
                    aspect_ratios: grids,
                    include_thumbnail: true,
                },
                normalization: Normalization::ZeroToOne,
            },
            program: ImageProgram {
                value_ops: vec![ValueOp::Rescale(1.0 / 255.0)],
                patch_size: None,
                pad_value: None,
                outputs: vec![OutputSpec {
                    name: "pixels".to_owned(),
                    content: "pixels".to_owned(),
                    dtype: ImageTensorDType::Fp32,
                    pad_value: None,
                    optional: false,
                }],
            },
        }
    }

    #[test]
    fn none_tiling_preserves_one_output_per_image() {
        let preprocessor = ImagePreprocessor::from_input(&[-1, 3, 2, 2]).unwrap();
        let images = [
            DynamicImage::ImageRgb8(RgbImage::from_pixel(3, 2, Rgb([255, 0, 0]))),
            DynamicImage::ImageRgb8(RgbImage::from_pixel(2, 3, Rgb([0, 0, 255]))),
        ];
        let tensor = preprocessor.preprocess(&images).unwrap();
        let pixels = tensor.tensor_by_content("pixels").unwrap();

        assert_eq!(pixels.shape, [2, 3, 2, 2]);
        assert_eq!(tensor.num_tiles, 2);
        assert_eq!(tensor.tiles_per_image, [1, 1]);
        assert_eq!(
            tensor.tile_grids,
            [
                TileGrid {
                    columns: 1,
                    rows: 1
                },
                TileGrid {
                    columns: 1,
                    rows: 1
                }
            ]
        );
        assert_eq!(
            tensor
                .images
                .iter()
                .map(|image| image.original_size)
                .collect::<Vec<_>>(),
            [(3, 2), (2, 3)]
        );
        assert_eq!(pixels.data.len(), 2 * 3 * 2 * 2);
    }

    #[test]
    fn fixed_grid_produces_grid_tiles_and_global_thumbnail() {
        let preprocessor = tiled_preprocessor(
            TilingMode::FixedGrid,
            vec![TileGrid {
                columns: 3,
                rows: 2,
            }],
            6,
        );
        let image = DynamicImage::ImageRgb8(RgbImage::from_fn(6, 4, |x, y| {
            Rgb([(x * 20) as u8, (y * 30) as u8, 0])
        }));
        let tensor = preprocessor.preprocess(&[image]).unwrap();
        let pixels = tensor.tensor_by_content("pixels").unwrap();

        assert_eq!(pixels.shape, [7, 3, 2, 2]);
        assert_eq!(tensor.num_tiles, 7);
        assert_eq!(tensor.tiles_per_image, [7]);
        assert_eq!(
            tensor.tile_grids,
            [TileGrid {
                columns: 3,
                rows: 2
            }]
        );
        assert_eq!(pixels.data.len(), 7 * 3 * 2 * 2);
        assert_eq!(tensor.tile_data(0).unwrap().len(), 3 * 2 * 2);
        assert_eq!(tensor.tile_data(6).unwrap().len(), 3 * 2 * 2);
        assert!(tensor.tile_data(7).is_none());
    }

    #[test]
    fn dynamic_anyres_selects_expected_representative_grids() {
        let grids = default_anyres_grids();
        assert_eq!(
            select_best_grid(1200, 400, 336, 6, &grids).unwrap(),
            TileGrid {
                columns: 3,
                rows: 1
            }
        );
        assert_eq!(
            select_best_grid(400, 1200, 336, 6, &grids).unwrap(),
            TileGrid {
                columns: 1,
                rows: 3
            }
        );
        assert_eq!(
            select_best_grid(800, 800, 336, 6, &grids).unwrap(),
            TileGrid {
                columns: 2,
                rows: 2
            }
        );
    }

    #[test]
    fn dynamic_anyres_respects_max_tiles_and_adds_thumbnail() {
        let preprocessor = tiled_preprocessor(
            TilingMode::DynamicAnyres,
            vec![
                TileGrid {
                    columns: 3,
                    rows: 2,
                },
                TileGrid {
                    columns: 2,
                    rows: 2,
                },
                TileGrid {
                    columns: 2,
                    rows: 1,
                },
            ],
            4,
        );
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(800, 800, Rgb([64, 128, 255])));
        let tensor = preprocessor.preprocess(&[image]).unwrap();
        let pixels = tensor.tensor_by_content("pixels").unwrap();

        assert_eq!(pixels.shape, [5, 3, 2, 2]);
        assert_eq!(tensor.num_tiles, 5);
        assert_eq!(tensor.tiles_per_image, [5]);
        assert_eq!(
            tensor.tile_grids,
            [TileGrid {
                columns: 2,
                rows: 2
            }]
        );
    }

    // --- Tests specifically for thumbnail position / tile ordering alignment ---

    /// Regression test for the bug reported by Gaff: when the preprocessor
    /// includes a thumbnail it is always placed FIRST in the tensor
    /// (`ThumbnailPosition::Prepend`).  `tiling_summary()` must report this so
    /// that callers can drive token expansion with the correct ordering.
    #[test]
    fn tiling_summary_reports_prepend_thumbnail_position_matching_tensor_layout() {
        let preprocessor = tiled_preprocessor(
            TilingMode::FixedGrid,
            vec![TileGrid {
                columns: 2,
                rows: 1,
            }],
            2,
        );
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(4, 2, Rgb([100, 150, 200])));
        let tensor = preprocessor.preprocess(&[image]).unwrap();

        // The pipeline stores thumbnail first (index 0) then local tiles.
        assert_eq!(
            tensor.thumbnail_position,
            ThumbnailPosition::Prepend,
            "tensor thumbnail_position must be Prepend to match tiled_image_for_grid layout"
        );
        assert_eq!(
            tensor.tiling_summary().thumbnail_position,
            ThumbnailPosition::Prepend,
        );
        // tiles_per_image = [thumbnail + 2 local] = 3
        assert_eq!(tensor.tiles_per_image, [3]);
    }

    /// Token order must match tile order when thumbnail is first in the tensor.
    ///
    /// With tokens_per_tile=1 and a 2×1 grid + prepended thumbnail the expected
    /// token sequence is [thumbnail, local(0,0), local(0,1)].  Previously this
    /// would be silently wrong if a caller accidentally used `Append` in config.
    #[test]
    fn prepend_thumbnail_token_order_matches_tensor_tile_order() {
        let mut config = token_expansion_config();
        config.tokens_per_tile = 1;
        config.thumbnail_position = ThumbnailPosition::Prepend;
        config.column_separator_token_id = Some(8);
        let tiles_per_image = [3];
        let grids = [TileGrid {
            columns: 2,
            rows: 1,
        }];
        // tiling.thumbnail_position=Prepend matches actual tensor layout.
        let expanded = expand_image_placeholders(
            &[99],
            ImageTilingSummary {
                num_tiles: 3,
                tiles_per_image: &tiles_per_image,
                tile_grids: &grids,
                thumbnail_position: ThumbnailPosition::Prepend,
            },
            &config,
        )
        .unwrap();
        // Expected: thumbnail first, then local tile 0, col_sep, local tile 1.
        assert_eq!(expanded, [7, 7, 8, 7]);
    }

    /// Token expansion must reject a config whose thumbnail_position contradicts
    /// the tensor layout reported by the tiling summary.  This is the exact
    /// failure mode described in Gaff's review: tensor has thumbnail FIRST but
    /// config says LAST, silently producing misaligned embeddings.
    #[test]
    fn mismatched_thumbnail_position_config_vs_tiling_is_rejected() {
        let mut config = token_expansion_config();
        config.thumbnail_position = ThumbnailPosition::Append; // wrong for a Prepend tensor
        let tiles_per_image = [3];
        let grids = [TileGrid {
            columns: 2,
            rows: 1,
        }];
        let error = expand_image_placeholders(
            &[99],
            ImageTilingSummary {
                num_tiles: 3,
                tiles_per_image: &tiles_per_image,
                tile_grids: &grids,
                thumbnail_position: ThumbnailPosition::Prepend, // actual tensor layout
            },
            &config,
        )
        .unwrap_err();
        let msg = error.to_string();
        assert!(
            msg.contains("thumbnail_position"),
            "error should mention thumbnail_position mismatch, got: {msg}"
        );
    }

    /// Verify that token expansion driven by the real ImageTensor tiling summary
    /// (thumbnail_position=Prepend) produces token order [thumbnail, local…],
    /// which aligns with how tiled_image_for_grid lays out pixels in the tensor.
    #[test]
    fn token_expansion_from_real_tensor_summary_matches_tile_layout() {
        let preprocessor = tiled_preprocessor(
            TilingMode::FixedGrid,
            vec![TileGrid {
                columns: 2,
                rows: 1,
            }],
            2,
        );
        let image = DynamicImage::ImageRgb8(RgbImage::from_pixel(4, 2, Rgb([10, 20, 30])));
        let tensor = preprocessor.preprocess(&[image]).unwrap();
        // 3 tiles: thumbnail (index 0), local (index 1), local (index 2).
        assert_eq!(tensor.num_tiles, 3);
        assert_eq!(tensor.thumbnail_position, ThumbnailPosition::Prepend);

        let summary = tensor.tiling_summary();
        let mut config = token_expansion_config();
        config.tokens_per_tile = 1;
        // Config must match the tensor layout reported by tiling_summary.
        config.thumbnail_position = summary.thumbnail_position;

        let expanded = expand_image_placeholders(&[99], summary, &config).unwrap();
        // 3 tokens total: first corresponds to thumbnail (tensor index 0),
        // then the two local tiles in row-major order.
        assert_eq!(expanded.len(), 3);
        assert_eq!(expanded, [7, 7, 7]);
    }

    fn typed_preprocessor(shape: &[i64], image_yaml: &str) -> ImagePreprocessor {
        let document = serde_yaml::from_str::<MetadataDocument>(image_yaml).unwrap();
        ImagePreprocessor::from_metadata_document(shape, Some(document)).unwrap()
    }

    fn packed_test_images() -> [DynamicImage; 2] {
        [
            DynamicImage::ImageRgb8(RgbImage::from_pixel(4, 2, Rgb([255, 0, 0]))),
            DynamicImage::ImageRgb8(RgbImage::from_pixel(2, 2, Rgb([0, 0, 255]))),
        ]
    }

    const PADDED_PROGRAM: &str = r#"
preprocessing:
  image:
    transforms:
      - op: decode_rgb
      - op: resize
        size: 2
        mode: stretch
        interpolation: bilinear
      - op: tile
        tile_size: 2
        max_tiles: 2
      - op: rescale
        scale: 0.00392156862745098
      - op: patchify
        patch_size: 1
        flatten: true
      - op: pad
        pad_value: 0
    outputs:
      - name: image_pixels
        content: pixels
        dtype: fp32
      - name: image_coordinates
        content: patch_coordinates
        dtype: int64
        pad_value: -1
"#;

    // Small checked-in vectors generated once from equivalent HF processor
    // operations (RGB conversion, resize, rescale, CHW patchify, and padding).
    const HF_PADDED_PIXELS: [f32; 48] = [
        1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    const HF_PADDED_COORDINATES: [i64; 32] = [
        0, 0, 0, 1, 1, 0, 1, 1, 2, 0, 2, 1, 3, 0, 3, 1, 0, 0, 0, 1, 1, 0, 1, 1, -1, -1, -1, -1, -1,
        -1, -1, -1,
    ];

    #[test]
    fn gemma4_shaped_padded_patches_and_sentinel_coordinates_match_fixture() {
        let preprocessor = typed_preprocessor(&[2, 8, 3], PADDED_PROGRAM);
        let bundle = preprocessor.preprocess(&packed_test_images()).unwrap();
        let pixels = bundle.tensor("image_pixels").unwrap();
        let coordinates = bundle.tensor("image_coordinates").unwrap();

        assert_eq!(pixels.shape, [2, 8, 3]);
        assert_eq!(
            pixels.data.as_f32_slice().unwrap(),
            HF_PADDED_PIXELS.as_slice()
        );
        assert_eq!(coordinates.shape, [2, 8, 2]);
        assert_eq!(
            coordinates.data,
            ImageTensorData::Int64(HF_PADDED_COORDINATES.to_vec())
        );
        assert_eq!(
            bundle
                .images
                .iter()
                .map(|summary| (
                    summary.image_index,
                    summary.expansion_count,
                    summary.tensor_offset,
                    summary.tensor_length,
                ))
                .collect::<Vec<_>>(),
            [(0, 8, 0, 8), (1, 4, 8, 8)]
        );
    }

    #[test]
    fn qwen_shaped_concatenated_patches_emit_per_image_grid() {
        let program = PADDED_PROGRAM
            .replace(
                "      - op: tile\n        tile_size: 2\n        max_tiles: 2\n",
                "",
            )
            .replace(
                "      - op: pad\n        pad_value: 0\n",
                "",
            )
            .replace(
                "      - name: image_coordinates\n        content: patch_coordinates\n        dtype: int64\n        pad_value: -1\n",
                "      - name: image_grid\n        content: grid_dimensions\n        dtype: int64\n",
            );
        let preprocessor = typed_preprocessor(&[8, 3], &program);
        let bundle = preprocessor.preprocess(&packed_test_images()).unwrap();
        let pixels = bundle.tensor("image_pixels").unwrap();
        let grid = bundle.tensor("image_grid").unwrap();

        assert_eq!(pixels.shape, [8, 3]);
        assert_eq!(grid.shape, [2, 3]);
        assert_eq!(grid.data, ImageTensorData::Int64(vec![1, 2, 2, 1, 2, 2]));
        assert_eq!(
            bundle
                .images
                .iter()
                .map(|summary| summary.tensor_offset)
                .collect::<Vec<_>>(),
            [0, 4]
        );
    }

    #[test]
    fn phi_shaped_outputs_include_original_sizes_and_patch_validity() {
        let program = PADDED_PROGRAM
            .replace("dtype: fp32", "dtype: bf16")
            .replace(
                "      - name: image_coordinates\n",
                "      - name: image_pixels_fp16\n        content: pixels\n        dtype: fp16\n      - name: image_coordinates\n",
            )
            .replace(
                "      - name: image_coordinates\n        content: patch_coordinates\n        dtype: int64\n        pad_value: -1\n",
                "      - name: image_sizes\n        content: original_size\n        dtype: int64\n      - name: patch_mask\n        content: validity_mask\n        dtype: bool\n",
            );
        let preprocessor = typed_preprocessor(&[2, 8, 3], &program);
        let bundle = preprocessor.preprocess(&packed_test_images()).unwrap();

        let pixels = bundle.tensor("image_pixels").unwrap();
        assert_eq!(pixels.dtype, ImageTensorDType::Bf16);
        assert!(matches!(pixels.data, ImageTensorData::Bf16(_)));
        let fp16_pixels = bundle.tensor("image_pixels_fp16").unwrap();
        assert_eq!(fp16_pixels.dtype, ImageTensorDType::Fp16);
        assert_eq!(
            fp16_pixels.data,
            ImageTensorData::Fp16(
                HF_PADDED_PIXELS
                    .iter()
                    .map(|value| if *value == 1.0 { 0x3c00 } else { 0 })
                    .collect()
            )
        );
        assert_eq!(
            bundle.tensor("image_sizes").unwrap().data,
            ImageTensorData::Int64(vec![2, 4, 2, 2])
        );
        assert_eq!(
            bundle.tensor("patch_mask").unwrap().data,
            ImageTensorData::Bool(vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
        );
    }

    #[test]
    fn rank4_nchw_values_remain_unchanged_in_bundle() {
        let preprocessor = ImagePreprocessor::from_input(&[1, 3, 1, 2]).unwrap();
        let image = DynamicImage::ImageRgb8(
            RgbImage::from_raw(2, 1, vec![255, 0, 128, 0, 64, 255]).unwrap(),
        );
        let bundle = preprocessor.preprocess(&[image]).unwrap();
        let pixels = bundle.tensor("pixels").unwrap();

        assert_eq!(pixels.shape, [1, 3, 1, 2]);
        assert_eq!(
            pixels.data.as_f32_slice().unwrap(),
            [1.0, 0.0, 0.0, 64.0 / 255.0, 128.0 / 255.0, 1.0]
        );
    }
}

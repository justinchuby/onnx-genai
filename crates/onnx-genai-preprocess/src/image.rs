//! Metadata-driven RGB image preprocessing.

use std::path::Path;

use anyhow::Context;
use image::{DynamicImage, Rgb, RgbImage, imageops::FilterType};
use serde::Deserialize;

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

/// A contiguous image tensor whose batch dimension contains all produced tiles.
#[derive(Debug)]
pub struct ImageTensor {
    pub shape: Vec<i64>,
    pub data: Vec<f32>,
    /// Total tiles across all input images.
    pub num_tiles: usize,
    /// Tile counts corresponding to each input image.
    pub tiles_per_image: Vec<usize>,
    pub original_sizes: Vec<(u32, u32)>,
}

impl ImageTensor {
    /// Returns one normalized tile in the tensor's declared channel layout.
    pub fn tile_data(&self, index: usize) -> Option<&[f32]> {
        let values_per_tile = self.data.len().checked_div(self.num_tiles)?;
        let start = index.checked_mul(values_per_tile)?;
        let end = start.checked_add(values_per_tile)?;
        self.data.get(start..end)
    }
}

/// Reusable image preprocessor resolved from a model input and §35 metadata.
#[derive(Debug, Clone)]
pub struct ImagePreprocessor {
    shape: Vec<i64>,
    layout: ImageLayout,
    config: ImagePreprocessConfig,
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

impl ImagePreprocessor {
    /// Resolves preprocessing from a rank-4 model input and optional metadata file.
    pub fn from_input_and_metadata(
        shape: &[i64],
        metadata_path: Option<&Path>,
    ) -> anyhow::Result<Self> {
        if shape.len() != 4 {
            anyhow::bail!("vision input must be rank 4, but the model declares {shape:?}");
        }
        let layout = match (shape[1], shape[3]) {
            (3, _) => ImageLayout::Nchw,
            (_, 3) => ImageLayout::Nhwc,
            _ => anyhow::bail!(
                "vision input must declare an RGB channel dimension, but the model declares {shape:?}"
            ),
        };
        let (height, width) = match layout {
            ImageLayout::Nchw => (shape[2], shape[3]),
            ImageLayout::Nhwc => (shape[1], shape[2]),
        };
        if shape[0] == 0 || shape[0] < -1 {
            anyhow::bail!("vision input has invalid batch dimension {}", shape[0]);
        }
        let config = load_preprocessing(metadata_path, width, height)?;
        let mut resolved_shape = shape.to_vec();
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
        Ok(Self {
            shape: resolved_shape,
            layout,
            config,
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

    /// Decodes encoded images and preprocesses them into one batched tensor.
    pub fn preprocess_encoded<I, B>(&self, images: I) -> anyhow::Result<ImageTensor>
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

    /// Preprocesses decoded images into one batched tensor.
    pub fn preprocess(&self, images: &[DynamicImage]) -> anyhow::Result<ImageTensor> {
        if images.is_empty() {
            anyhow::bail!("at least one image is required");
        }
        let pixels_per_image = self.config.width as usize * self.config.height as usize;
        let mut tiles = Vec::new();
        let mut tiles_per_image = Vec::with_capacity(images.len());
        let mut original_sizes = Vec::with_capacity(images.len());
        for image in images {
            original_sizes.push((image.width(), image.height()));
            let image_tiles = tile_image(image, &self.config)?;
            tiles_per_image.push(image_tiles.len());
            tiles.extend(image_tiles);
        }
        if self.shape[0] > 0 && self.shape[0] as usize != tiles.len() {
            anyhow::bail!(
                "this model expects {} image tile(s) per request, but preprocessing produced {}",
                self.shape[0],
                tiles.len()
            );
        }

        let num_tiles = tiles.len();
        let mut data = Vec::with_capacity(num_tiles * CHANNELS * pixels_per_image);
        for rgb in &tiles {
            match self.layout {
                ImageLayout::Nchw => {
                    for channel in 0..CHANNELS {
                        data.extend(
                            rgb.pixels()
                                .map(|pixel| normalize(pixel[channel], channel, &self.config)),
                        );
                    }
                }
                ImageLayout::Nhwc => {
                    for pixel in rgb.pixels() {
                        data.extend(
                            pixel
                                .0
                                .iter()
                                .enumerate()
                                .map(|(channel, value)| normalize(*value, channel, &self.config)),
                        );
                    }
                }
            }
        }

        let mut shape = self.shape.clone();
        shape[0] = i64::try_from(num_tiles).context("image tile batch is too large")?;
        Ok(ImageTensor {
            shape,
            data,
            num_tiles,
            tiles_per_image,
            original_sizes,
        })
    }
}

fn load_preprocessing(
    metadata_path: Option<&Path>,
    model_width: i64,
    model_height: i64,
) -> anyhow::Result<ImagePreprocessConfig> {
    let image_metadata = metadata_path
        .map(std::fs::read_to_string)
        .transpose()
        .context("failed to read preprocessing metadata")?
        .map(|content| {
            serde_yaml::from_str::<MetadataDocument>(&content)
                .context("failed to parse preprocessing metadata")
        })
        .transpose()?
        .and_then(|document| document.preprocessing)
        .and_then(|preprocessing| preprocessing.image);
    preprocessing_from_metadata(image_metadata, model_width, model_height)
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
) -> anyhow::Result<Vec<RgbImage>> {
    match config.tiling.mode {
        TilingMode::None => Ok(vec![resize_image(image, config)]),
        TilingMode::FixedGrid => {
            let grid = config.tiling.aspect_ratios[0];
            tiled_image_for_grid(image, config, grid)
        }
        TilingMode::DynamicAnyres => {
            let grid = select_best_grid(
                image.width(),
                image.height(),
                config.tiling.tile_size,
                config.tiling.max_tiles,
                &config.tiling.aspect_ratios,
            )?;
            tiled_image_for_grid(image, config, grid)
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

fn normalize(value: u8, channel: usize, config: &ImagePreprocessConfig) -> f32 {
    let value = f32::from(value) / 255.0;
    match &config.normalization {
        Normalization::ZeroToOne => value,
        Normalization::MeanStd { mean, std } => (value - mean[channel]) / std[channel],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        };
        let tensor = preprocessor.preprocess(&[image]).unwrap();
        let expected = [
            (1.0 - 0.48145466) / 0.26862954,
            (128.0 / 255.0 - 0.4578275) / 0.261_302_6,
            (0.0 - 0.40821073) / 0.275_777_1,
        ];
        for (actual, expected) in tensor.data.iter().zip(expected) {
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

        assert_eq!(tensor.shape, [2, 3, 2, 2]);
        assert_eq!(tensor.num_tiles, 2);
        assert_eq!(tensor.tiles_per_image, [1, 1]);
        assert_eq!(tensor.original_sizes, [(3, 2), (2, 3)]);
        assert_eq!(tensor.data.len(), 2 * 3 * 2 * 2);
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

        assert_eq!(tensor.shape, [7, 3, 2, 2]);
        assert_eq!(tensor.num_tiles, 7);
        assert_eq!(tensor.tiles_per_image, [7]);
        assert_eq!(tensor.data.len(), 7 * 3 * 2 * 2);
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

        assert_eq!(tensor.shape, [5, 3, 2, 2]);
        assert_eq!(tensor.num_tiles, 5);
        assert_eq!(tensor.tiles_per_image, [5]);
    }
}

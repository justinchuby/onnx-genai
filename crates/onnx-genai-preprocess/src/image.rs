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
    pub normalization: Normalization,
}

/// A contiguous image tensor in the model's declared layout.
#[derive(Debug)]
pub struct ImageTensor {
    pub shape: Vec<i64>,
    pub data: Vec<f32>,
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
        if self.shape[0] > 0 && self.shape[0] as usize != images.len() {
            anyhow::bail!(
                "this model expects {} image(s) per request, but {} were provided",
                self.shape[0],
                images.len()
            );
        }

        let pixels_per_image = self.config.width as usize * self.config.height as usize;
        let mut data = Vec::with_capacity(images.len() * CHANNELS * pixels_per_image);
        for image in images {
            let rgb = resize_image(image, &self.config);
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
        shape[0] = i64::try_from(images.len()).context("image batch is too large")?;
        Ok(ImageTensor { shape, data })
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
        normalization,
    })
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

fn resize_image(image: &DynamicImage, config: &ImagePreprocessConfig) -> RgbImage {
    let rgb = image.to_rgb8();
    let filter = match config.interpolation {
        Interpolation::Bicubic => FilterType::CatmullRom,
        Interpolation::Bilinear => FilterType::Triangle,
        Interpolation::Lanczos3 => FilterType::Lanczos3,
    };
    match config.resize_mode {
        ResizeMode::Fixed => image::imageops::resize(&rgb, config.width, config.height, filter),
        ResizeMode::ShortestEdgeCenterCrop => {
            let scale = (config.width as f64 / rgb.width() as f64)
                .max(config.height as f64 / rgb.height() as f64);
            let resized_width = ((rgb.width() as f64 * scale).round() as u32).max(config.width);
            let resized_height = ((rgb.height() as f64 * scale).round() as u32).max(config.height);
            let resized = image::imageops::resize(&rgb, resized_width, resized_height, filter);
            image::imageops::crop_imm(
                &resized,
                (resized_width - config.width) / 2,
                (resized_height - config.height) / 2,
                config.width,
                config.height,
            )
            .to_image()
        }
        ResizeMode::LongestEdgePad => {
            let scale = (config.width as f64 / rgb.width() as f64)
                .min(config.height as f64 / rgb.height() as f64);
            let resized_width =
                ((rgb.width() as f64 * scale).round() as u32).clamp(1, config.width);
            let resized_height =
                ((rgb.height() as f64 * scale).round() as u32).clamp(1, config.height);
            let resized = image::imageops::resize(&rgb, resized_width, resized_height, filter);
            let mut padded = RgbImage::from_pixel(config.width, config.height, Rgb([0, 0, 0]));
            image::imageops::replace(
                &mut padded,
                &resized,
                i64::from((config.width - resized_width) / 2),
                i64::from((config.height - resized_height) / 2),
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
}

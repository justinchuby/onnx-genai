use std::time::Duration;

use anyhow::Context;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use image::{DynamicImage, imageops::FilterType};

const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;

#[derive(Debug, Clone)]
pub(crate) struct VisionInputSpec {
    pub(crate) endpoint: String,
    pub(crate) shape: Vec<i64>,
    pub(crate) layout: ImageLayout,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ImageLayout {
    Nchw,
    Nhwc,
}

#[derive(Debug)]
pub(crate) struct ImageTensor {
    pub(crate) endpoint: String,
    pub(crate) shape: Vec<i64>,
    pub(crate) data: Vec<f32>,
}

impl VisionInputSpec {
    pub(crate) fn from_input(endpoint: String, shape: &[i64]) -> anyhow::Result<Self> {
        if shape.len() != 4 {
            anyhow::bail!(
                "vision input '{endpoint}' must be rank 4, but the model declares {shape:?}"
            );
        }
        let layout = match (shape[1], shape[3]) {
            (3, _) => ImageLayout::Nchw,
            (_, 3) => ImageLayout::Nhwc,
            _ => anyhow::bail!(
                "vision input '{endpoint}' must declare an RGB channel dimension, but the model declares {shape:?}"
            ),
        };
        let (height, width) = match layout {
            ImageLayout::Nchw => (shape[2], shape[3]),
            ImageLayout::Nhwc => (shape[1], shape[2]),
        };
        if height <= 0 || width <= 0 {
            anyhow::bail!(
                "vision input '{endpoint}' must declare fixed image dimensions, but the model declares {shape:?}"
            );
        }
        if shape[0] == 0 || shape[0] < -1 {
            anyhow::bail!(
                "vision input '{endpoint}' has invalid batch dimension {}",
                shape[0]
            );
        }
        Ok(Self {
            endpoint,
            shape: shape.to_vec(),
            layout,
        })
    }

    fn dimensions(&self) -> (u32, u32) {
        match self.layout {
            ImageLayout::Nchw => (self.shape[3] as u32, self.shape[2] as u32),
            ImageLayout::Nhwc => (self.shape[2] as u32, self.shape[1] as u32),
        }
    }
}

pub(crate) async fn load_and_preprocess(
    urls: &[String],
    spec: &VisionInputSpec,
) -> anyhow::Result<ImageTensor> {
    if urls.is_empty() {
        anyhow::bail!("at least one image is required");
    }
    if spec.shape[0] > 0 && spec.shape[0] as usize != urls.len() {
        anyhow::bail!(
            "this model expects {} image(s) per request, but {} were provided",
            spec.shape[0],
            urls.len()
        );
    }

    let mut images = Vec::with_capacity(urls.len());
    for url in urls {
        let bytes = load_image_bytes(url).await?;
        images.push(
            image::load_from_memory(&bytes)
                .with_context(|| format!("failed to decode image from {url}"))?,
        );
    }
    preprocess_images(&images, spec)
}

pub(crate) fn preprocess_images(
    images: &[DynamicImage],
    spec: &VisionInputSpec,
) -> anyhow::Result<ImageTensor> {
    let (width, height) = spec.dimensions();
    let pixels_per_image = width as usize * height as usize;
    let mut data = Vec::with_capacity(images.len() * 3 * pixels_per_image);

    for image in images {
        let rgb = image
            .resize_exact(width, height, FilterType::Triangle)
            .to_rgb8();
        // The generic pipeline metadata exposes shape and dtype but no processor
        // statistics, so use RGB values normalized to [0, 1].
        match spec.layout {
            ImageLayout::Nchw => {
                for channel in 0..3 {
                    data.extend(rgb.pixels().map(|pixel| pixel[channel] as f32 / 255.0));
                }
            }
            ImageLayout::Nhwc => {
                data.extend(
                    rgb.pixels()
                        .flat_map(|pixel| pixel.0.map(|value| value as f32 / 255.0)),
                );
            }
        }
    }

    let mut shape = spec.shape.clone();
    shape[0] = i64::try_from(images.len()).context("image batch is too large")?;
    Ok(ImageTensor {
        endpoint: spec.endpoint.clone(),
        shape,
        data,
    })
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

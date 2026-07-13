use std::{path::Path, time::Duration};

use anyhow::Context;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use onnx_genai_preprocess::image::ImagePreprocessor;

const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;

#[derive(Debug)]
pub(crate) struct ImageTensor {
    pub(crate) endpoint: String,
    pub(crate) shape: Vec<i64>,
    pub(crate) data: Vec<f32>,
    /// Total number of preprocessed tiles in this tensor batch.
    pub(crate) num_tiles: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct VisionInputSpec {
    pub(crate) endpoint: String,
    preprocessor: ImagePreprocessor,
}

impl VisionInputSpec {
    #[cfg(test)]
    pub(crate) fn from_input(endpoint: String, shape: &[i64]) -> anyhow::Result<Self> {
        Self::from_input_and_metadata(endpoint, shape, None)
    }

    pub(crate) fn from_input_and_metadata(
        endpoint: String,
        shape: &[i64],
        metadata_path: Option<&Path>,
    ) -> anyhow::Result<Self> {
        let preprocessor = ImagePreprocessor::from_input_and_metadata(shape, metadata_path)
            .with_context(|| format!("invalid preprocessing for vision input '{endpoint}'"))?;
        Ok(Self {
            endpoint,
            preprocessor,
        })
    }
}

pub(crate) async fn load_and_preprocess(
    urls: &[String],
    spec: &VisionInputSpec,
) -> anyhow::Result<ImageTensor> {
    let mut images = Vec::with_capacity(urls.len());
    for url in urls {
        images.push(load_image_bytes(url).await?);
    }
    let tensor = spec
        .preprocessor
        .preprocess_encoded(&images)
        .with_context(|| format!("failed to preprocess image for {}", spec.endpoint))?;
    Ok(ImageTensor {
        endpoint: spec.endpoint.clone(),
        shape: tensor.shape,
        data: tensor.data,
        num_tiles: tensor.num_tiles,
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

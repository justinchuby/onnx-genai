use std::time::Duration;

use anyhow::Context;
use base64::{Engine as _, engine::general_purpose::STANDARD};

const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;
const MAX_IMAGE_BASE64_BYTES: usize = MAX_IMAGE_BYTES.div_ceil(3) * 4;
pub const MAX_EXPANDED_PROMPT_TOKENS: usize = 1024 * 1024;

pub(crate) async fn fetch_images(urls: &[String]) -> anyhow::Result<Vec<Vec<u8>>> {
    let mut images = Vec::with_capacity(urls.len());
    for url in urls {
        images.push(load_image_bytes(url).await?);
    }
    Ok(images)
}

async fn load_image_bytes(url: &str) -> anyhow::Result<Vec<u8>> {
    if let Some(data) = url.strip_prefix("data:image/") {
        let (_, encoded) = data
            .split_once(";base64,")
            .context("image data URI must contain ';base64,'")?;
        if encoded.len() > MAX_IMAGE_BASE64_BYTES {
            anyhow::bail!("encoded image exceeds the {MAX_IMAGE_BYTES}-byte input limit");
        }
        let bytes = STANDARD
            .decode(encoded)
            .context("image data URI contains invalid base64")?;
        if bytes.len() > MAX_IMAGE_BYTES {
            anyhow::bail!("decoded image exceeds the {MAX_IMAGE_BYTES}-byte input limit");
        }
        return Ok(bytes);
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        let parsed = reqwest::Url::parse(url).context("invalid image URL")?;
        if parsed.host_str().is_none()
            || !parsed.username().is_empty()
            || parsed.password().is_some()
        {
            anyhow::bail!("image URL must have a host and must not contain credentials");
        }
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .redirect(reqwest::redirect::Policy::limited(3))
            .build()?;
        let mut response = client.get(parsed).send().await?.error_for_status()?;
        if response
            .content_length()
            .is_some_and(|length| length > MAX_IMAGE_BYTES as u64)
        {
            anyhow::bail!("remote image exceeds the {MAX_IMAGE_BYTES}-byte input limit");
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await? {
            if bytes.len().saturating_add(chunk.len()) > MAX_IMAGE_BYTES {
                anyhow::bail!("remote image exceeds the {MAX_IMAGE_BYTES}-byte input limit");
            }
            bytes.extend_from_slice(&chunk);
        }
        return Ok(bytes);
    }
    anyhow::bail!("image input must be an HTTP(S) URL or image data URI")
}

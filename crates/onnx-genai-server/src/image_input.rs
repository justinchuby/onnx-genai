use std::time::Duration;

use anyhow::Context;
use base64::{Engine as _, engine::general_purpose::STANDARD};

const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;
pub const MAX_EXPANDED_PROMPT_TOKENS: usize = 1024 * 1024;

pub(crate) async fn fetch_images(urls: &[String]) -> anyhow::Result<Vec<Vec<u8>>> {
    let mut images = Vec::with_capacity(urls.len());
    for url in urls {
        images.push(fetch_image(url, std::num::NonZeroUsize::new(MAX_IMAGE_BYTES).unwrap()).await?);
    }
    Ok(images)
}

pub(crate) async fn fetch_image(
    url: &str,
    max_bytes: std::num::NonZeroUsize,
) -> anyhow::Result<Vec<u8>> {
    let max_bytes = max_bytes.get().min(MAX_IMAGE_BYTES);
    if let Some(data) = url.strip_prefix("data:image/") {
        let (_, encoded) = data
            .split_once(";base64,")
            .context("image data URI must contain ';base64,'")?;
        return decode_base64_image(encoded, max_bytes);
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
            .is_some_and(|length| length > max_bytes as u64)
        {
            anyhow::bail!("remote image exceeds the {max_bytes}-byte input limit");
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response.chunk().await? {
            if bytes.len().saturating_add(chunk.len()) > max_bytes {
                anyhow::bail!("remote image exceeds the {max_bytes}-byte input limit");
            }
            bytes.extend_from_slice(&chunk);
        }
        return Ok(bytes);
    }
    if url.contains("://") || url.starts_with("data:") {
        anyhow::bail!("image input must be an HTTP(S) URL, image data URI or raw base64");
    }
    decode_base64_image(url, max_bytes)
}

fn decode_base64_image(encoded: &str, max_bytes: usize) -> anyhow::Result<Vec<u8>> {
    if encoded.len() > max_bytes.div_ceil(3) * 4 {
        anyhow::bail!("encoded image exceeds the {max_bytes}-byte input limit");
    }
    let bytes = STANDARD
        .decode(encoded)
        .context("image input must be an HTTP(S) URL, image data URI or valid raw base64")?;
    if bytes.len() > max_bytes {
        anyhow::bail!("decoded image exceeds the {max_bytes}-byte input limit");
    }
    Ok(bytes)
}

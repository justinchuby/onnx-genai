use std::time::Duration;

use anyhow::Context;
use base64::{Engine as _, engine::general_purpose::STANDARD};

const MAX_ENCODED_MEDIA_BYTES: usize = 20 * 1024 * 1024;
pub const MAX_EXPANDED_PROMPT_TOKENS: usize = 1024 * 1024;

pub(crate) async fn fetch_images(urls: &[String]) -> anyhow::Result<Vec<Vec<u8>>> {
    let mut images = Vec::with_capacity(urls.len());
    for url in urls {
        images.push(
            fetch_image(
                url,
                std::num::NonZeroUsize::new(MAX_ENCODED_MEDIA_BYTES).unwrap(),
            )
            .await?,
        );
    }
    Ok(images)
}

pub(crate) async fn fetch_image(
    url: &str,
    max_bytes: std::num::NonZeroUsize,
) -> anyhow::Result<Vec<u8>> {
    let max_bytes = max_bytes.get().min(MAX_ENCODED_MEDIA_BYTES);
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

/// Bounded request-local ownership shared across typed media and questions.
pub(crate) struct MediaRequest {
    remaining: usize,
    budget_hint: &'static str,
    media: std::collections::HashMap<
        onnx_genai_metadata::EncodedMediaKind,
        std::collections::HashMap<String, std::sync::Arc<[u8]>>,
    >,
}
impl MediaRequest {
    pub(crate) fn new(limit: std::num::NonZeroUsize, budget_hint: &'static str) -> Self {
        Self {
            remaining: limit.get(),
            budget_hint,
            media: std::collections::HashMap::new(),
        }
    }
    pub(crate) async fn load(
        &mut self,
        kind: onnx_genai_metadata::EncodedMediaKind,
        urls: &[String],
        field: &str,
    ) -> Result<Vec<std::sync::Arc<[u8]>>, crate::routes::ApiError> {
        let mut media = Vec::with_capacity(urls.len());
        for (index, url) in urls.iter().enumerate() {
            media.push(
                self.load_one(kind, url, &format!("{field}[{index}]"))
                    .await?,
            );
        }
        Ok(media)
    }
    pub(crate) async fn load_one(
        &mut self,
        kind: onnx_genai_metadata::EncodedMediaKind,
        url: &str,
        field: &str,
    ) -> Result<std::sync::Arc<[u8]>, crate::routes::ApiError> {
        use crate::routes::ApiError;
        if let Some(bytes) = self.media.get(&kind).and_then(|cache| cache.get(url)) {
            return Ok(std::sync::Arc::clone(bytes));
        }
        let limit = std::num::NonZeroUsize::new(self.remaining).ok_or_else(|| {
            ApiError::payload_too_large(format!(
                "{field} exceeds the request media budget; reduce media or raise {}",
                self.budget_hint
            ))
        })?;
        let bytes=match kind {
            onnx_genai_metadata::EncodedMediaKind::Image=>fetch_image(url,limit).await,
            onnx_genai_metadata::EncodedMediaKind::Audio=>decode_audio(url,limit),
        }.map_err(|error|ApiError::invalid_request(format!("{field} could not be loaded within the remaining {limit}-byte media budget ({}): {error:#}",self.budget_hint)))?;
        let bytes=tokio::task::spawn_blocking(move ||->anyhow::Result<Vec<u8>> {
            match kind {
                onnx_genai_metadata::EncodedMediaKind::Image=>{ image::ImageReader::new(std::io::Cursor::new(&bytes)).with_guessed_format()?.decode()?; }
                onnx_genai_metadata::EncodedMediaKind::Audio=>{onnx_genai_preprocess::audio::decode_wav_pcm16(&bytes)?;}
            }
            Ok(bytes)
        }).await.map_err(|error|ApiError::internal(format!("validate {field}: media decoding task stopped: {error}; retry or inspect server logs")))?
            .map_err(|error|ApiError::invalid_request({ let (label,guidance)=match kind {onnx_genai_metadata::EncodedMediaKind::Image=>("image","valid PNG or JPEG data"),onnx_genai_metadata::EncodedMediaKind::Audio=>("audio","valid PCM16 WAV data")}; format!("{field} is not a supported encoded {label}: {error:#}; provide {guidance}") }))?;
        self.remaining -= bytes.len();
        let bytes: std::sync::Arc<[u8]> = bytes.into();
        self.media
            .entry(kind)
            .or_default()
            .insert(url.to_owned(), std::sync::Arc::clone(&bytes));
        Ok(bytes)
    }
}
fn decode_audio(source: &str, limit: std::num::NonZeroUsize) -> anyhow::Result<Vec<u8>> {
    let encoded = if let Some(uri) = source.strip_prefix("data:audio/") {
        uri.split_once(";base64,")
            .context("audio data URI must contain ';base64,'")?
            .1
    } else {
        anyhow::ensure!(
            !source.contains("://") && !source.starts_with("data:") && !source.starts_with("file:"),
            "audio must be base64 PCM16 WAV bytes or an audio data URI; remote/file URLs are not fetched"
        );
        source
    };
    let max = limit.get().min(MAX_ENCODED_MEDIA_BYTES);
    anyhow::ensure!(
        encoded.len() <= max.div_ceil(3) * 4,
        "encoded audio exceeds its {max}-byte input limit; reduce the clip"
    );
    let bytes = STANDARD
        .decode(encoded)
        .context("audio must contain valid base64 PCM16 WAV bytes")?;
    anyhow::ensure!(
        bytes.len() <= max,
        "decoded audio exceeds its {max}-byte input limit; reduce the clip"
    );
    Ok(bytes)
}

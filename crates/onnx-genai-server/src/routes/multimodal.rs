use super::*;

pub(crate) async fn audio_transcriptions(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Response, ApiError> {
    let mut file = None;
    let mut filename = None;
    let mut language = None;
    let mut response_format = "json".to_string();
    let mut model_name = String::new();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|err| ApiError::bad_request(format!("invalid multipart form: {err}")))?
    {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "file" => {
                filename = field.file_name().map(ToString::to_string);
                file = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|err| {
                            ApiError::bad_request(format!("failed to read audio file: {err}"))
                        })?
                        .to_vec(),
                );
            }
            "language" => {
                language = Some(field.text().await.map_err(|err| {
                    ApiError::bad_request(format!("invalid language field: {err}"))
                })?);
            }
            "response_format" => {
                response_format = field.text().await.map_err(|err| {
                    ApiError::bad_request(format!("invalid response_format field: {err}"))
                })?;
            }
            "model" => {
                model_name = field
                    .text()
                    .await
                    .map_err(|err| ApiError::bad_request(format!("invalid model field: {err}")))?;
            }
            _ => {}
        }
    }

    let handle = resolve_model(&state.registry, &model_name).await?;

    let bytes = file.ok_or_else(|| ApiError::bad_request("multipart field 'file' is required"))?;
    if !matches!(response_format.as_str(), "json" | "text") {
        return Err(ApiError::bad_request(format!(
            "unsupported response_format '{response_format}'; expected 'json' or 'text'"
        )));
    }
    if filename
        .as_deref()
        .is_some_and(|name| name.to_ascii_lowercase().ends_with(".mp3"))
    {
        return Err(ApiError::bad_request(
            "MP3 audio is not supported yet; provide a PCM16 WAV file",
        ));
    }
    let spec = handle
        .multimodal
        .as_ref()
        .and_then(|multimodal| multimodal.audio.as_ref())
        .ok_or_else(|| {
            ApiError::bad_request(format!(
                "What: audio transcription was rejected for model '{}'. \
                 Why: no component of its package declares an `input_features` audio input. \
                 How: serve a speech package.",
                handle.id
            ))
        })?;
    let input = MultimodalInput::from_wav(spec, &bytes)
        .map_err(|err| ApiError::bad_request(format!("invalid audio input: {err}")))?;
    let max_tokens = spec
        .max_tokens
        .unwrap_or(state.config.max_output_tokens)
        .min(state.config.max_output_tokens);
    let token_ids = audio_decoder_prompt(&handle.tokenizer, language.as_deref())?;
    let prompt_tokens = token_ids.len();
    let request = GenerateRequest {
        prompt: GeneratePrompt::TokenIds(token_ids),
        options: GenerateOptions {
            max_new_tokens: max_tokens,
            temperature: 0.0,
            max_context: handle.model_max_context,
            ..GenerateOptions::default()
        },
    };
    let generation = handle
        .engine
        .generate_pipeline(request, Some(input))
        .await
        .map_err(map_generate_submit_error)?;
    let result = collect_generation_result(generation.events)
        .await
        .map_err(generation_failure)?;
    crate::metrics::add_prompt_tokens(prompt_tokens);

    match response_format.as_str() {
        "json" => Ok(Json(AudioTranscriptionResponse { text: result.text }).into_response()),
        "text" => Ok((
            [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            result.text,
        )
            .into_response()),
        _ => unreachable!("response format validated before generation"),
    }
}

/// `POST /v1/images/generations` — OpenAI-compatible text-to-image.
///
/// Renders through the model package's own declared denoise loop; the sampler,
/// scheduler, and component wiring all come from its metadata. Only
/// `b64_json` is offered because this server stores nothing and therefore has
/// no URL to hand back.
/// `POST /v1/audio/speech` — OpenAI-compatible text-to-speech.
///
/// Synthesizes through the package's own declared vocoder stage and returns the
/// audio bytes directly, as OpenAI does. Only container formats this server can
/// encode are offered; a compressed format is refused rather than silently
/// substituted, because a caller that asked for MP3 must not receive WAV under
/// an MP3 content type.
pub(crate) async fn audio_speech(
    State(_state): State<AppState>,
    ApiJson(request): ApiJson<SpeechRequest>,
) -> Result<Response, ApiError> {
    let handle = resolve_model(&_state.registry, &request.model).await?;
    if !handle.text_to_audio {
        return Err(ApiError::bad_request(format!(
            "What: speech synthesis was rejected for model '{}'. \
             Why: its package declares no waveform stage (a `run_on: final_only` component fed by an autoregressive decoder), so it cannot produce audio. \
             How: serve a text-to-speech package, or call /v1/chat/completions for text.",
            handle.id
        )));
    }
    if request.input.trim().is_empty() {
        return Err(ApiError::bad_request(
            "What: an empty `input` was rejected. \
             Why: there is nothing to speak. \
             How: send the text to synthesize in `input`.",
        ));
    }
    let format = request.response_format.unwrap_or_default();
    let content_type = format.content_type().ok_or_else(|| {
        ApiError::bad_request(format!(
            "What: response_format \"{}\" was rejected. \
             Why: this server encodes only uncompressed audio, and returning WAV under an {} content type would mislabel the body. \
             How: request response_format \"wav\" or \"pcm\".",
            format.label(),
            format.label()
        ))
    })?;
    if let Some(speed) = request.speed
        && (speed - 1.0).abs() > f32::EPSILON
    {
        return Err(ApiError::bad_request(format!(
            "What: speed {speed} was rejected. \
             Why: this server does not resample synthesized audio, so it cannot change playback rate. \
             How: omit `speed`, or resample the returned audio yourself."
        )));
    }

    let synthesis_request = TextToAudioRequest {
        text: request.input.clone(),
        max_new_tokens: request.max_tokens,
        temperature: request.temperature,
        seed: request.seed,
        sample_rate: request.sample_rate,
    };
    let audio = handle
        .engine
        .synthesize_speech(handle.tokenizer.clone(), synthesis_request)
        .await
        .map_err(map_generate_submit_error)?
        .map_err(|error| ApiError::bad_request(format!("speech synthesis failed: {error:#}")))?;

    let body = match format {
        SpeechResponseFormat::Pcm => audio.to_pcm16(),
        _ => audio
            .to_wav()
            .map_err(|error| ApiError::internal(format!("audio encoding failed: {error:#}")))?,
    };
    Ok((
        [
            (header::CONTENT_TYPE, content_type.to_string()),
            // The rate is not in the body for raw PCM, so advertise it.
            (
                header::HeaderName::from_static("x-sample-rate"),
                audio.sample_rate.to_string(),
            ),
        ],
        body,
    )
        .into_response())
}

pub(crate) async fn image_generations(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<ImageGenerationRequest>,
) -> Result<Response, ApiError> {
    let handle = resolve_model(&state.registry, &request.model).await?;
    if !handle.text_to_image {
        return Err(ApiError::bad_request(format!(
            "What: image generation was rejected for model '{}'. \
             Why: its package declares no denoise loop (`pipeline.strategy.denoiser`), so it cannot produce images. \
             How: serve a diffusion package, or call /v1/chat/completions for text.",
            handle.id
        )));
    }
    if let Some(ImageResponseFormat::Url) = request.response_format {
        return Err(ApiError::bad_request(
            "What: response_format \"url\" was rejected. \
             Why: this server does not host generated images, so it has no URL to return. \
             How: request response_format \"b64_json\" and decode the base64 PNG yourself.",
        ));
    }
    let count = request.n.unwrap_or(1);
    // The bound is the renderer's policy, so the CLI and this endpoint agree.
    onnx_genai::text_to_image::validate_batch_size(count)
        .map_err(|error| ApiError::bad_request(format!("{error:#} (field: n)")))?;
    let (width, height) = parse_image_size(request.size.as_deref())?;
    // Bounded before anything is allocated: size and steps are caller-supplied.
    onnx_genai::text_to_image::validate_image_size(width, height)
        .map_err(|error| ApiError::bad_request(format!("{error:#} (field: size)")))?;
    if let Some(steps) = request.steps {
        onnx_genai::text_to_image::validate_steps(steps)
            .map_err(|error| ApiError::bad_request(format!("{error:#} (field: steps)")))?;
    }

    let render_request = TextToImageRequest {
        prompt: request.prompt.clone(),
        negative_prompt: request.negative_prompt.clone().unwrap_or_default(),
        steps: request.steps,
        guidance_scale: request.guidance_scale,
        start_step: None,
        seed: request.seed.unwrap_or(0),
        width,
        height,
        batch_size: count,
        tokenizer_path: None,
        text_encoder_path: None,
        vae_decoder: None,
        source_image: None,
        vae_encoder: None,
    };
    let images = handle
        .engine
        .render_images(handle.model_dir.clone(), render_request)
        .await
        .map_err(map_generate_submit_error)?
        .map_err(|error| ApiError::bad_request(format!("image generation failed: {error:#}")))?;

    let data = images
        .iter()
        .map(|image| encode_png_base64(image).map(|b64_json| ImageData { b64_json }))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error: anyhow::Error| {
            ApiError::internal(format!("image encoding failed: {error:#}"))
        })?;
    if data.is_empty() {
        return Err(ApiError::internal(
            "What: image generation returned nothing. \
             Why: the pipeline produced fewer images than the requested batch. \
             How: retry with n=1, or report this as a pipeline output-shape bug.",
        ));
    }

    Ok(Json(ImageGenerationResponse {
        created: now_unix(),
        data,
    })
    .into_response())
}

/// Parse OpenAI's `"<width>x<height>"` size, defaulting to 512x512.
fn parse_image_size(size: Option<&str>) -> Result<(usize, usize), ApiError> {
    const DEFAULT_SIDE: usize = 512;
    let Some(size) = size
        .map(str::trim)
        .filter(|size| !size.is_empty() && *size != "auto")
    else {
        return Ok((DEFAULT_SIDE, DEFAULT_SIDE));
    };
    let reject = || {
        ApiError::bad_request(format!(
            "What: size \"{size}\" was rejected. \
             Why: it is not a \"<width>x<height>\" pair of positive integers. \
             How: send a size such as \"512x512\" or \"768x512\", or omit it for 512x512."
        ))
    };
    let (width, height) = size.split_once(['x', 'X']).ok_or_else(reject)?;
    let width: usize = width.trim().parse().map_err(|_| reject())?;
    let height: usize = height.trim().parse().map_err(|_| reject())?;
    if width == 0 || height == 0 {
        return Err(reject());
    }
    Ok((width, height))
}

/// Encode one rendered image as a base64 PNG.
fn encode_png_base64(image: &onnx_genai::text_to_image::RenderedImage) -> anyhow::Result<String> {
    let mut png = Vec::new();
    onnx_genai::text_to_image::write_png(image, &mut png)?;
    Ok(base64::engine::general_purpose::STANDARD.encode(png))
}

#[cfg(test)]
mod transcription_failure_tests {
    use super::*;

    #[test]
    fn kv_admission_refusal_maps_to_overload() {
        let error = generation_failure(DriverFailure {
            message: "internal scheduler details".to_string(),
            kind: DriverFailureKind::MemoryOverload,
        });

        assert_eq!(error.status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(error.kind, "resource_limit_error");
        assert_eq!(error.message, MEMORY_OVERLOAD_MESSAGE);
    }

    #[test]
    fn unrelated_transcription_failure_remains_internal() {
        let error = generation_failure(DriverFailure::internal("decoder failure"));

        assert_eq!(error.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(error.kind, "server_error");
    }
}

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

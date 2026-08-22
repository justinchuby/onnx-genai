use axum::{
    extract::State,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use onnx_genai::{GenerateOptions, GeneratePrompt, GenerateRequest};

use super::{ApiError, ApiJson, map_generate_submit_error, resolve_model};
use crate::{state::AppState, types::AudioSpeechRequest};

pub(crate) async fn audio_speech(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<AudioSpeechRequest>,
) -> Result<Response, ApiError> {
    if request.stream {
        return Err(ApiError::bad_request(
            "stream=true is not supported for buffered audio workflows",
        ));
    }
    if !request.response_format.eq_ignore_ascii_case("wav") {
        return Err(ApiError::bad_request(
            "only response_format=wav is supported",
        ));
    }
    let handle = resolve_model(&state.registry, &request.model).await?;
    if !handle.pipeline {
        return Err(ApiError::bad_request(format!(
            "model '{}' is not an executable workflow package",
            handle.id
        )));
    }
    let processor = handle.speech_prompt.as_ref().ok_or_else(|| {
        ApiError::bad_request(format!(
            "model '{}' does not declare a text-assembly speech adapter",
            handle.id
        ))
    })?;
    let prompt = processor
        .assemble(&request.input, &request.instructions)
        .map_err(|error| ApiError::bad_request(format!("invalid speech input: {error:#}")))?;
    let token_ids = handle.tokenizer.encode(&prompt).map_err(|error| {
        ApiError::bad_request(format!("failed to tokenize speech input: {error}"))
    })?;
    if token_ids.len() > processor.max_input_tokens {
        return Err(ApiError::bad_request(format!(
            "speech prompt contains {} tokens, exceeding the declared maximum of {}",
            token_ids.len(),
            processor.max_input_tokens
        )));
    }
    let token_rows = processor.token_rows(token_ids).map_err(|error| {
        ApiError::bad_request(format!("invalid speech guidance rows: {error:#}"))
    })?;
    let generation = GenerateRequest {
        prompt: GeneratePrompt::TokenRows(token_rows),
        options: GenerateOptions {
            max_new_tokens: processor
                .max_output_units
                .saturating_add(processor.state_advance_units),
            max_context: handle.model_max_context,
            ..GenerateOptions::default()
        },
    };
    let audio = handle
        .engine
        .synthesize_speech(generation)
        .await
        .map_err(map_generate_submit_error)?;
    Ok((
        StatusCode::OK,
        [(header::CONTENT_TYPE, "audio/wav")],
        audio.bytes,
    )
        .into_response())
}

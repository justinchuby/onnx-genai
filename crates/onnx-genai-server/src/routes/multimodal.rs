use super::completions::await_driver_admission;
use super::*;

pub(crate) async fn audio_transcriptions(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Response, ApiError> {
    let mut file = None;
    let mut language = None;
    let mut prompt = None;
    let mut response_format = "json".to_string();
    let mut model_name = String::new();
    let mut stream = false;
    let mut requested_tokens = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|err| ApiError::invalid_request(format!("invalid multipart form: {err}")))?
    {
        let name = field.name().unwrap_or_default().to_string();
        match name.as_str() {
            "file" => {
                file = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|err| {
                            ApiError::invalid_request(format!("failed to read audio file: {err}"))
                        })?
                        .to_vec(),
                );
            }
            "stream" => {
                let value = field.text().await.map_err(|err| {
                    ApiError::invalid_request(format!("invalid stream field: {err}"))
                })?;
                stream = value
                    .parse::<bool>()
                    .map_err(|_| ApiError::invalid_request("stream must be true or false"))?;
            }
            "max_tokens" => {
                let value = field.text().await.map_err(|err| {
                    ApiError::invalid_request(format!("invalid max_tokens field: {err}"))
                })?;
                requested_tokens = Some(value.parse::<usize>().map_err(|_| {
                    ApiError::invalid_request("max_tokens must be a positive integer")
                })?);
            }
            "prompt" => {
                prompt = Some(field.text().await.map_err(|err| {
                    ApiError::invalid_request(format!("invalid prompt field: {err}"))
                })?);
            }
            "language" => {
                language = Some(field.text().await.map_err(|err| {
                    ApiError::invalid_request(format!("invalid language field: {err}"))
                })?);
            }
            "response_format" => {
                response_format = field.text().await.map_err(|err| {
                    ApiError::invalid_request(format!("invalid response_format field: {err}"))
                })?;
            }
            "model" => {
                model_name = field.text().await.map_err(|err| {
                    ApiError::invalid_request(format!("invalid model field: {err}"))
                })?;
            }
            _ => {}
        }
    }

    let handle = resolve_model(&state.registry, &model_name).await?;

    let bytes =
        file.ok_or_else(|| ApiError::invalid_request("multipart field 'file' is required"))?;
    if !matches!(response_format.as_str(), "json" | "text") {
        return Err(ApiError::invalid_request(format!(
            "unsupported response_format '{response_format}'; expected 'json' or 'text'"
        )));
    }
    if stream && response_format != "json" {
        return Err(ApiError::invalid_request(
            "stream=true requires response_format=json for transcript SSE events",
        ));
    }
    let output_limit =
        std::num::NonZeroUsize::new(requested_tokens.unwrap_or(state.config.max_output_tokens))
            .filter(|limit| limit.get() <= state.config.max_output_tokens)
            .ok_or_else(|| {
                ApiError::invalid_request(format!(
                    "max_tokens must be 1..={}; lower max_tokens or raise --max-output-tokens",
                    state.config.max_output_tokens
                ))
            })?;
    if let Some(contract) = &handle.engine.transcription {
        if language.as_ref().is_some_and(|value| !value.is_empty())
            || prompt.as_ref().is_some_and(|value| !value.is_empty())
        {
            return Err(ApiError::invalid_request(
                "this frame-decoding transcription profile declares no language/prompt conditioning; omit those fields or select a conditioned speech package",
            ));
        }
        let bytes=tokio::task::spawn_blocking(move ||onnx_genai_preprocess::audio::decode_audio(&bytes).map(|_|bytes)).await
            .map_err(|error|ApiError::internal(format!("validate transcription file: decoding task stopped: {error}; retry or inspect server logs")))?
            .map_err(|error|ApiError::invalid_request(format!("file is not supported audio: {error}; provide a supported encoded audio recording")))?;
        let limits=onnx_genai_engine::transcription::TranscriptionLimits {
            max_frames:state.config.max_transcription_frames,
            max_tokens:std::num::NonZeroUsize::new(state.config.max_output_tokens).ok_or_else(||ApiError::internal("max_output_tokens is zero; configure a positive transcription output token allowance"))?,
            max_text_bytes:state.config.max_transcription_text_bytes,
        };
        let prepared = onnx_genai_engine::transcription::PreparedTranscription::prepare(
            Arc::clone(contract),
            bytes.into(),
            limits,
        )
        .and_then(|request| request.with_output_limit(output_limit))
        .map_err(|error| {
            ApiError::invalid_request(format!("file transcription preparation: {error:#}"))
        })?;
        let result = handle
            .engine
            .transcribe(prepared)
            .await
            .map_err(map_generate_submit_error)?;
        let usage = TranscriptionUsage {
            input: result.frames,
            output: result.token_ids.len(),
            cached: 0,
            audio_frames: Some(result.frames),
        };
        return transcription_response(result.text, usage, &response_format, stream);
    }
    if prompt.as_ref().is_some_and(|value| !value.is_empty()) {
        return Err(ApiError::invalid_request(
            "prompt conditioning is not implemented by this generative speech input contract; omit prompt",
        ));
    }
    let spec = handle
        .multimodal
        .as_ref()
        .and_then(|multimodal| multimodal.audio.as_ref())
        .ok_or_else(|| {
            ApiError {status:StatusCode::NOT_IMPLEMENTED,kind:"not_supported_error",retry_after_secs:None,message:format!(
                "What: audio transcription was rejected for model '{}'. Why: its package declares neither a frame-transcription profile nor a generative audio input. How: serve a speech package with an explicit audio input contract.",handle.id)}
        })?;
    let input = MultimodalInput::from_audio_bytes(spec, &bytes)
        .map_err(|err| ApiError::invalid_request(format!("invalid audio input: {err:#}")))?;
    let max_tokens = output_limit.get();
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
        .submit_generation(
            None,
            request,
            onnx_genai_engine::ToolCallPolicy::Disabled,
            Some(input),
        )
        .await
        .map_err(map_generate_submit_error)?;
    await_driver_admission(generation.admission).await?;
    crate::metrics::add_prompt_tokens(prompt_tokens);
    if stream {
        return Ok(transcription_stream(
            generation.events,
            prompt_tokens,
            state.config.max_transcription_text_bytes.get(),
        ));
    }
    let result = collect_generation_result(generation.events)
        .await
        .map_err(generation_failure)?;
    if result.text.len() > state.config.max_transcription_text_bytes.get() {
        return Err(ApiError::invalid_request(
            "transcription text exceeds --max-transcription-text-bytes; shorten audio or raise the render-buffer limit",
        ));
    }
    let usage = TranscriptionUsage {
        input: prompt_tokens,
        output: result.token_ids.len(),
        cached: result.prefix_cache_hit_len,
        audio_frames: None,
    };
    transcription_response(result.text, usage, &response_format, false)
}
#[derive(Clone, Copy)]
struct TranscriptionUsage {
    input: usize,
    output: usize,
    cached: usize,
    audio_frames: Option<usize>,
}
impl TranscriptionUsage {
    fn done(self, text: &str) -> Result<serde_json::Value, ApiError> {
        let total = self.input.checked_add(self.output).ok_or_else(|| {
            ApiError::internal(
                "transcription usage overflows; shorten the audio or reduce the output budget",
            )
        })?;
        transcript_value(crate::types::TranscriptResponse::Done {
            text: text.to_owned(),
            usage: crate::types::TranscriptTokenUsage {
                input_tokens: self.input,
                output_tokens: self.output,
                total_tokens: total,
                input_tokens_details: crate::types::TranscriptInputTokenDetails {
                    cached_tokens: self.cached,
                    audio_frames: self.audio_frames,
                },
            },
        })
    }
}
fn transcript_value(
    event: crate::types::TranscriptResponse,
) -> Result<serde_json::Value, ApiError> {
    serde_json::to_value(event).map_err(|error| {
        ApiError::internal(format!(
            "serialize transcription event: {error}; inspect server logs"
        ))
    })
}
fn transcript_delta(delta: &str) -> Result<serde_json::Value, ApiError> {
    transcript_value(crate::types::TranscriptResponse::Delta {
        delta: delta.to_owned(),
    })
}
fn transcription_response(
    text: String,
    usage: TranscriptionUsage,
    response_format: &str,
    stream: bool,
) -> Result<Response, ApiError> {
    if stream {
        let mut events: Vec<Result<Event, Infallible>> = Vec::new();
        if !text.is_empty() {
            events.push(Ok(
                Event::default().data(transcript_delta(&text)?.to_string())
            ));
        }
        events.push(Ok(Event::default().data(usage.done(&text)?.to_string())));
        events.push(Ok(Event::default().data("[DONE]")));
        return Ok(Sse::new(tokio_stream::iter(events)).into_response());
    }
    match response_format {
        "json" => Ok(Json(usage.done(&text)?).into_response()),
        "text" => Ok(([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], text).into_response()),
        _ => unreachable!("response format validated before transcription"),
    }
}
async fn transcript_event(
    tx: &mpsc::Sender<Result<Event, Infallible>>,
    value: serde_json::Value,
) -> bool {
    tx.send(Ok(Event::default().data(value.to_string())))
        .await
        .is_ok()
}
fn transcription_stream(
    mut events: mpsc::Receiver<DriverEvent>,
    input_tokens: usize,
    max_bytes: usize,
) -> Response {
    let (tx, rx) = mpsc::channel(16);
    tokio::spawn(async move {
        let outcome=async {
            let mut published=String::new();
            loop {
                let event=tokio::select! { biased; _=tx.closed()=>return Ok(()), event=events.recv()=>event };
                match event {
                    Some(DriverEvent::Token(token))=>{
                        let size=published.len().checked_add(token.text.len()).ok_or_else(||ApiError::internal("transcription delta size overflows; shorten audio"))?;
                        if size>max_bytes {return Err(ApiError::invalid_request("transcription text exceeds --max-transcription-text-bytes; shorten audio or raise the render-buffer limit"));}
                        if !token.text.is_empty() {
                            if !transcript_event(&tx,transcript_delta(&token.text)?).await {return Ok(());}
                            published.push_str(&token.text);
                        }
                    }
                    Some(DriverEvent::WorkflowPublications(_))=>{}
                    Some(DriverEvent::Finished(result))=>{
                        if result.text.len()>max_bytes {return Err(ApiError::invalid_request("transcription text exceeds --max-transcription-text-bytes; shorten audio or raise the render-buffer limit"));}
                        let tail=result.text.strip_prefix(&published).ok_or_else(||ApiError::internal("committed transcription does not extend its published deltas; repair the tokenizer's incremental output contract"))?;
                        if !tail.is_empty() && !transcript_event(&tx,transcript_delta(tail)?).await {return Ok(());}
                        let usage=TranscriptionUsage{input:input_tokens,output:result.token_ids.len(),cached:result.prefix_cache_hit_len,audio_frames:None};
                        let _=transcript_event(&tx,usage.done(&result.text)?).await;
                        let _=tx.send(Ok(Event::default().data("[DONE]"))).await;
                        return Ok(());
                    }
                    Some(DriverEvent::Error(error))=>return Err(generation_failure(error)),
                    None=>return Err(ApiError::internal("transcription stream ended without a final result; inspect the engine logs and retry")),
                }
            }
        }.await;
        if let Err(error) = outcome {
            let _ = transcript_event(
                &tx,
                serde_json::json!({"error":{"type":error.kind,"message":error.message}}),
            )
            .await;
            let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
        }
    });
    Sse::new(ReceiverStream::new(rx)).into_response()
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio_stream::StreamExt;
    #[tokio::test]
    async fn transcript_deltas_publish_before_completion_and_disconnect_releases_driver_receiver() {
        let (driver, events) = mpsc::channel(1);
        let response = transcription_stream(events, 3, 1024);
        let mut body = response.into_body().into_data_stream();
        driver
            .send(DriverEvent::Token(onnx_genai_engine::GenerateToken {
                token_id: 1,
                text: "first".into(),
                finish_reason: None,
                logprob: None,
            }))
            .await
            .unwrap();
        let chunk = tokio::time::timeout(Duration::from_secs(1), body.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let value = std::str::from_utf8(&chunk).unwrap();
        assert!(
            value.contains("transcript.text.delta") && value.contains("first"),
            "{value}"
        );
        drop(body);
        tokio::time::timeout(Duration::from_secs(1), driver.closed())
            .await
            .unwrap();
    }
}

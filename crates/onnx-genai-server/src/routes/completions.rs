use super::*;

pub(crate) async fn completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    ApiJson(request): ApiJson<CompletionRequest>,
) -> Result<Response, ApiError> {
    let handle = resolve_model(&state.registry, &request.model).await?;
    if handle.pipeline {
        return Err(ApiError::bad_request(
            "/v1/completions is not supported by pipeline models",
        ));
    }
    validate_completion_request(&request, &state.config)?;
    let session_id = session_id_from_headers(&headers)?;
    if request.suffix.is_some() && handle.fim_config.is_none() {
        return Err(ApiError::bad_request(
            "FIM is not supported by this model because its tokenizer configuration does not declare recognized FIM tokens",
        ));
    }
    if request.suffix.is_some() && session_id.is_some() {
        return Err(ApiError::bad_request(
            "X-Session-Id is not supported for FIM completions",
        ));
    }

    if request.stream {
        Ok(stream_completion(state, handle, request, session_id)
            .await?
            .into_response())
    } else {
        Ok(Json(run_completion(state, handle, request, session_id).await?).into_response())
    }
}

pub(crate) async fn embeddings(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, ApiError> {
    let handle = resolve_model(&state.registry, &request.model).await?;
    validate_embedding_request(&request, &handle.tokenizer)?;

    let encoding_format = request.encoding_format;
    let model = request.model.clone();

    let inputs: Vec<Vec<u32>> = match request.input {
        EmbeddingInput::String(text) => {
            let tokens = handle
                .tokenizer
                .encode(&text)
                .map_err(|err| ApiError::internal(format!("input tokenization failed: {err}")))?;
            vec![tokens]
        }
        EmbeddingInput::Strings(texts) => {
            let mut all = Vec::with_capacity(texts.len());
            for text in &texts {
                let tokens = handle.tokenizer.encode(text).map_err(|err| {
                    ApiError::internal(format!("input tokenization failed: {err}"))
                })?;
                all.push(tokens);
            }
            all
        }
        EmbeddingInput::TokenArrays(arrays) => arrays,
    };

    let total_tokens: usize = inputs.iter().map(|ids| ids.len()).sum();

    let mut data = Vec::with_capacity(inputs.len());
    for (index, input_ids) in inputs.into_iter().enumerate() {
        let vector = handle
            .engine
            .embed(input_ids, EmbeddingOptions::default())
            .await
            .map_err(|err| ApiError::internal(format!("embedding failed: {err}")))?;
        data.push(EmbeddingData {
            object: "embedding",
            embedding: EmbeddingVector::from_floats(vector, encoding_format),
            index,
        });
    }

    Ok(Json(EmbeddingResponse {
        object: "list",
        data,
        model,
        usage: EmbeddingUsage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    }))
}

fn validate_embedding_request(
    request: &EmbeddingRequest,
    tokenizer: &Tokenizer,
) -> Result<(), ApiError> {
    if request.dimensions == Some(0) {
        return Err(ApiError::bad_request(
            "dimensions must be greater than zero",
        ));
    }

    let validate_tokens = |tokens: &[u32]| {
        if tokens.is_empty() {
            Err(ApiError::bad_request(
                "embedding input must contain at least one token",
            ))
        } else {
            Ok(())
        }
    };
    match &request.input {
        EmbeddingInput::String(input) => {
            let tokens = tokenizer.encode(input).map_err(|err| {
                ApiError::bad_request(format!("input tokenization failed: {err}"))
            })?;
            validate_tokens(&tokens)
        }
        EmbeddingInput::Strings(inputs) => {
            if inputs.is_empty() {
                return Err(ApiError::bad_request(
                    "embedding input array must not be empty",
                ));
            }
            for input in inputs {
                let tokens = tokenizer.encode(input).map_err(|err| {
                    ApiError::bad_request(format!("input tokenization failed: {err}"))
                })?;
                validate_tokens(&tokens)?;
            }
            Ok(())
        }
        EmbeddingInput::TokenArrays(inputs) => {
            if inputs.is_empty() {
                return Err(ApiError::bad_request(
                    "embedding input array must not be empty",
                ));
            }
            for tokens in inputs {
                validate_tokens(tokens)?;
            }
            Ok(())
        }
    }
}

async fn run_completion(
    state: AppState,
    handle: Arc<ModelHandle>,
    request: CompletionRequest,
    client_session_id: Option<String>,
) -> Result<CompletionResponse, ApiError> {
    let id = text_completion_id();
    let created = now_unix();
    let model = request.model.clone();
    let requested_logprobs = request.logprobs;
    let tokenizer = handle.tokenizer.clone();
    let prepared = prepare_completion(&request, &handle)?;
    enforce_context_cap(
        prepared.prompt_tokens,
        request.max_tokens,
        handle.model_max_context,
    )?;
    let generation = submit_completion(
        &handle,
        &state.sessions,
        prepared.generation,
        client_session_id.as_deref(),
    )
    .await?;
    let result = collect_generation_result(generation.events)
        .await
        .map_err(generation_failure)?;
    crate::metrics::add_prompt_tokens(prepared.prompt_tokens);
    let completion_tokens = result.token_ids.len();
    let logprobs = completion_logprobs(&result, &tokenizer, requested_logprobs)
        .map_err(|err| ApiError::internal(format!("logprobs conversion failed: {err}")))?;

    Ok(CompletionResponse {
        id,
        object: "text_completion",
        created,
        model,
        choices: vec![CompletionChoice {
            text: result.text,
            index: 0,
            finish_reason: finish_reason_label(&result.finish_reason),
            logprobs,
        }],
        usage: Usage {
            prompt_tokens: prepared.prompt_tokens,
            completion_tokens,
            total_tokens: prepared.prompt_tokens + completion_tokens,
        },
    })
}

async fn stream_completion(
    state: AppState,
    handle: Arc<ModelHandle>,
    request: CompletionRequest,
    client_session_id: Option<String>,
) -> Result<Sse<ReceiverStream<Result<Event, Infallible>>>, ApiError> {
    let id = text_completion_id();
    let created = now_unix();
    let model = request.model.clone();
    let requested_logprobs = request.logprobs;
    let tokenizer = handle.tokenizer.clone();
    let user_stop_sequences = request
        .stop
        .clone()
        .map(StopInput::into_texts)
        .unwrap_or_default();
    let prepared = prepare_completion(&request, &handle)?;
    enforce_context_cap(
        prepared.prompt_tokens,
        request.max_tokens,
        handle.model_max_context,
    )?;
    let generation = submit_completion(
        &handle,
        &state.sessions,
        prepared.generation,
        client_session_id.as_deref(),
    )
    .await?;
    await_driver_admission(generation.admission).await?;
    let mut driver_rx = generation.events;
    crate::metrics::add_prompt_tokens(prepared.prompt_tokens);
    let (tx, rx) = mpsc::channel(16);

    tokio::spawn(async move {
        let mut stop_buffer = StopBoundaryBuffer::new(user_stop_sequences.clone());
        let mut emitted_text = false;
        let result = loop {
            match driver_rx.recv().await {
                Some(DriverEvent::Token(token)) => {
                    if requested_logprobs.is_some() {
                        continue;
                    }
                    let finish_reason = token.finish_reason.clone();
                    let text = stop_buffer.push(&token.text);
                    if !text.is_empty() {
                        emitted_text = true;
                        send_completion_stream_chunk(
                            &tx,
                            completion_chunk(&id, created, &model, text, None),
                        )
                        .await?;
                    }
                    if matches!(finish_reason, Some(FinishReason::StopSequence { .. })) {
                        stop_buffer.pending.clear();
                    }
                }
                Some(DriverEvent::Finished(result)) => break Ok(result),
                Some(DriverEvent::Error(error)) => break Err(error),
                None => {
                    break Err(DriverFailure::internal(
                        "generation stream ended before result",
                    ));
                }
            }
        };

        match result {
            Ok(result) => {
                if let Some(requested_logprobs) = requested_logprobs {
                    send_completion_logprob_chunks(
                        &tx,
                        (&id, created, &model),
                        &result,
                        &tokenizer,
                        requested_logprobs,
                        &user_stop_sequences,
                    )
                    .await?;
                } else if !emitted_text && !result.text.is_empty() {
                    send_completion_stream_chunk(
                        &tx,
                        completion_chunk(&id, created, &model, result.text, None),
                    )
                    .await?;
                } else if !matches!(result.finish_reason, FinishReason::StopSequence { .. }) {
                    let text = stop_buffer.flush();
                    if !text.is_empty() {
                        send_completion_stream_chunk(
                            &tx,
                            completion_chunk(&id, created, &model, text, None),
                        )
                        .await?;
                    }
                }
                send_completion_stream_chunk(
                    &tx,
                    completion_done_chunk(
                        &id,
                        created,
                        &model,
                        finish_reason_label(&result.finish_reason),
                    ),
                )
                .await?;
            }
            Err(err) => {
                let error = generation_failure(err);
                tx.send(Ok(Event::default().event("error").data(
                    serde_json::to_string(&ErrorResponse {
                        error: ErrorBody {
                            message: error.message,
                            kind: error.kind,
                        },
                    })?,
                )))
                .await
                .context("stream receiver closed")?;
            }
        }

        tx.send(Ok(Event::default().data("[DONE]")))
            .await
            .context("stream receiver closed")?;
        Ok::<(), anyhow::Error>(())
    });

    Ok(Sse::new(ReceiverStream::new(rx)))
}

pub(crate) async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    ApiJson(request): ApiJson<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    let handle = resolve_model(&state.registry, &request.model).await?;
    validate_request(&request, &state.config)?;
    let requested_session_id = session_id_from_headers(&headers)?;
    // OpenAI-compatible clients such as OpenCode attach their own session key
    // while still resending the complete message history. Pipeline engines
    // already retain and rewind their one device-resident context internally,
    // so ignore the transport hint instead of rejecting an otherwise valid
    // stateless request.
    let session_id = (!handle.pipeline).then_some(requested_session_id).flatten();
    let image_urls = request.image_urls();
    let input_audio = request.input_audio();
    // One admission policy, shared with the CLI, so both front ends accept and
    // reject the same inputs with the same explanation.
    crate::multimodal::admit_attachments(
        handle.multimodal.as_ref(),
        &format!("model '{}'", handle.id),
        image_urls.len(),
        input_audio.len(),
    )
    .map_err(|error| ApiError::bad_request(format!("{error:#}")))?;

    if request.stream {
        Ok(
            stream_chat_completion(state, handle, request, session_id, image_urls, input_audio)
                .await?
                .into_response(),
        )
    } else {
        let response =
            run_chat_completion(state, handle, request, session_id, image_urls, input_audio)
                .await?;
        Ok(Json(response).into_response())
    }
}

async fn run_chat_completion(
    state: AppState,
    handle: Arc<ModelHandle>,
    request: ChatCompletionRequest,
    client_session_id: Option<String>,
    image_urls: Vec<String>,
    input_audio: Vec<InputAudio>,
) -> Result<ChatCompletionResponse, ApiError> {
    let id = completion_id();
    let created = now_unix();
    let model = request.model.clone();
    let requested_top_logprobs = request
        .logprobs
        .then_some(request.top_logprobs.unwrap_or(0));
    let tokenizer = handle.tokenizer.clone();
    let placeholder = positional_image_placeholder(&request, &handle);
    let mut prepared = prepare_generate_request(
        &request,
        &handle.tokenizer,
        handle.chat_template.as_deref(),
        client_session_id.is_some(),
        placeholder.as_deref(),
        handle.generation_defaults.as_ref(),
    )
    .map_err(|err| ApiError::internal(format!("prompt tokenization failed: {err}")))?;
    if !input_audio.is_empty() {
        prepared = prepare_audio_generate_request(&request, &handle.tokenizer)?;
    }
    let pipeline_input = if !image_urls.is_empty() {
        Some(
            preprocess_chat_images(
                &image_urls,
                handle
                    .multimodal
                    .as_ref()
                    .and_then(|multimodal| multimodal.vision.as_ref())
                    .ok_or_else(|| {
                        ApiError::bad_request(
                            "image input requires a model with a declared vision input",
                        )
                    })?,
                &mut prepared,
                handle.model_max_context,
                request.max_tokens,
            )
            .await?,
        )
    } else if let Some(audio) = input_audio.first() {
        Some(preprocess_chat_audio(audio, &handle)?)
    } else {
        None
    };
    enforce_context_cap(
        prepared.prompt_tokens,
        request.max_tokens,
        handle.model_max_context,
    )?;
    let prompt_tokens = prepared.prompt_tokens;
    let mut generation_request = prepared.request;
    generation_request.options.max_context = handle.model_max_context;
    let session_lookup = if let Some(id) = client_session_id.as_deref() {
        Some(get_or_create_session(&handle.engine, &state.sessions, id).await?)
    } else {
        None
    };

    let session_for_count = session_lookup;
    let wants_constrained_json = request.wants_constrained_json();
    let generation = if handle.pipeline {
        handle
            .engine
            .generate_pipeline(generation_request, pipeline_input)
            .await
            .map_err(map_generate_submit_error)?
    } else {
        handle
            .engine
            .generate(session_lookup, generation_request)
            .await
            .map_err(map_generate_submit_error)?
    };
    let result = collect_generation_result(generation.events)
        .await
        .map_err(generation_failure);
    crate::metrics::add_prompt_tokens(prompt_tokens);

    let session_token_count = if let Some(engine_session_id) = session_for_count {
        Some(
            handle
                .engine
                .session_token_count(engine_session_id)
                .await
                .map_err(|err| ApiError::internal(format!("session token count failed: {err}")))?,
        )
    } else {
        None
    };

    let (content, tool_calls, completion_tokens, finish_reason, logprobs) = match result {
        Ok(result) => {
            let default_finish_reason = finish_reason_label(&result.finish_reason);
            let logprobs = chat_logprobs(&result, &tokenizer, requested_top_logprobs)
                .map_err(|err| ApiError::internal(format!("logprobs conversion failed: {err}")))?;
            let parsed = if tools_parseable_from_output(&request) {
                parse_assistant_output(
                    assistant_output_text(&result, &tokenizer),
                    default_finish_reason,
                )
            } else {
                ParsedAssistantOutput {
                    content: Some(result.text),
                    tool_calls: None,
                    finish_reason: default_finish_reason,
                }
            };
            (
                parsed.content,
                parsed.tool_calls,
                result.token_ids.len(),
                parsed.finish_reason,
                logprobs,
            )
        }
        Err(err)
            if wants_constrained_json
                && json_constraint_stopped_incomplete_message(&err.message) =>
        {
            (Some("{}".to_string()), None, 0, "stop", None)
        }
        Err(err) => return Err(err),
    };
    let total_tokens = prompt_tokens + completion_tokens;
    Ok(ChatCompletionResponse {
        id,
        object: "chat.completion",
        created,
        model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: content.map(ChatMessageContent::Text),
                tool_calls,
                tool_call_id: None,
                name: None,
            },
            finish_reason,
            logprobs,
        }],
        usage: Some(Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens,
        }),
        session_id: client_session_id,
        session_token_count,
    })
}

async fn stream_chat_completion(
    state: AppState,
    handle: Arc<ModelHandle>,
    request: ChatCompletionRequest,
    client_session_id: Option<String>,
    image_urls: Vec<String>,
    input_audio: Vec<InputAudio>,
) -> Result<Sse<ReceiverStream<Result<Event, Infallible>>>, ApiError> {
    let id = completion_id();
    let created = now_unix();
    let model = request.model.clone();
    let requested_top_logprobs = request
        .logprobs
        .then_some(request.top_logprobs.unwrap_or(0));
    let tokenizer = handle.tokenizer.clone();
    let user_stop_sequences = request
        .stop
        .clone()
        .map(StopInput::into_texts)
        .unwrap_or_default();
    let placeholder = positional_image_placeholder(&request, &handle);
    let mut prepared = prepare_generate_request(
        &request,
        &handle.tokenizer,
        handle.chat_template.as_deref(),
        client_session_id.is_some(),
        placeholder.as_deref(),
        handle.generation_defaults.as_ref(),
    )
    .map_err(|err| ApiError::internal(format!("prompt tokenization failed: {err}")))?;
    if !input_audio.is_empty() {
        prepared = prepare_audio_generate_request(&request, &handle.tokenizer)?;
    }
    let pipeline_input = if !image_urls.is_empty() {
        Some(
            preprocess_chat_images(
                &image_urls,
                handle
                    .multimodal
                    .as_ref()
                    .and_then(|multimodal| multimodal.vision.as_ref())
                    .ok_or_else(|| {
                        ApiError::bad_request(
                            "image input requires a model with a declared vision input",
                        )
                    })?,
                &mut prepared,
                handle.model_max_context,
                request.max_tokens,
            )
            .await?,
        )
    } else if let Some(audio) = input_audio.first() {
        Some(preprocess_chat_audio(audio, &handle)?)
    } else {
        None
    };
    enforce_context_cap(
        prepared.prompt_tokens,
        request.max_tokens,
        handle.model_max_context,
    )?;
    let wants_constrained_json = request.wants_constrained_json();
    let mut generation_request = prepared.request;
    generation_request.options.max_context = handle.model_max_context;
    let (tx, rx) = mpsc::channel(16);
    let session_lookup = if let Some(id) = client_session_id.as_deref() {
        Some(get_or_create_session(&handle.engine, &state.sessions, id).await?)
    } else {
        None
    };
    let generation = if handle.pipeline {
        handle
            .engine
            .generate_pipeline(generation_request, pipeline_input)
            .await
            .map_err(map_generate_submit_error)?
    } else {
        handle
            .engine
            .generate(session_lookup, generation_request)
            .await
            .map_err(map_generate_submit_error)?
    };
    await_driver_admission(generation.admission).await?;
    let mut driver_rx = generation.events;
    crate::metrics::add_prompt_tokens(prepared.prompt_tokens);

    tokio::spawn(async move {
        send_stream_chunk(&tx, role_chunk(&id, created, &model)).await?;

        let mut stop_buffer = StopBoundaryBuffer::new(user_stop_sequences.clone());
        let mut buffered_text = String::new();
        let buffer_for_tool_detection =
            request.has_tool_context() && tools_parseable_from_output(&request);
        let result = loop {
            match driver_rx.recv().await {
                Some(DriverEvent::Token(token)) => {
                    if requested_top_logprobs.is_some() {
                        continue;
                    }
                    let finish_reason = token.finish_reason.clone();
                    let content = stop_buffer.push(&token.text);
                    if buffer_for_tool_detection {
                        buffered_text.push_str(&content);
                    } else if !wants_constrained_json && !content.is_empty() {
                        send_stream_chunk(&tx, content_chunk(&id, created, &model, content, None))
                            .await?;
                    }
                    if matches!(finish_reason, Some(FinishReason::StopSequence { .. })) {
                        stop_buffer.pending.clear();
                    }
                }
                Some(DriverEvent::Finished(result)) => break Ok(result),
                Some(DriverEvent::Error(error)) => break Err(error),
                None => {
                    break Err(DriverFailure::internal(
                        "generation stream ended before result",
                    ));
                }
            }
        };

        match result {
            Ok(result) => {
                if let Some(requested_top_logprobs) = requested_top_logprobs {
                    let tool_calls = if buffer_for_tool_detection {
                        parse_tool_calls(&assistant_output_text(&result, &tokenizer))
                    } else {
                        Vec::new()
                    };
                    if tool_calls.is_empty() {
                        send_chat_logprob_chunks(
                            &tx,
                            (&id, created, &model),
                            &result,
                            &tokenizer,
                            requested_top_logprobs,
                            &user_stop_sequences,
                        )
                        .await?;
                        send_stream_chunk(
                            &tx,
                            done_chunk(
                                &id,
                                created,
                                &model,
                                finish_reason_label(&result.finish_reason),
                            ),
                        )
                        .await?;
                    } else {
                        send_tool_call_deltas(&tx, (&id, created, &model), tool_calls).await?;
                    }
                } else if buffer_for_tool_detection {
                    let parsed = parse_assistant_output(
                        assistant_output_text(&result, &tokenizer),
                        finish_reason_label(&result.finish_reason),
                    );
                    if let Some(tool_calls) = parsed.tool_calls {
                        send_tool_call_deltas(&tx, (&id, created, &model), tool_calls).await?;
                    } else {
                        let content = parsed.content.unwrap_or_else(|| {
                            if !matches!(result.finish_reason, FinishReason::StopSequence { .. }) {
                                buffered_text.push_str(&stop_buffer.flush());
                            }
                            buffered_text
                        });
                        if !content.is_empty() {
                            send_stream_chunk(
                                &tx,
                                content_chunk(&id, created, &model, content, None),
                            )
                            .await?;
                        }
                        send_stream_chunk(
                            &tx,
                            done_chunk(&id, created, &model, parsed.finish_reason),
                        )
                        .await?;
                    }
                } else if wants_constrained_json {
                    if !result.text.is_empty() {
                        send_stream_chunk(
                            &tx,
                            content_chunk(&id, created, &model, result.text, None),
                        )
                        .await?;
                    }
                    send_stream_chunk(
                        &tx,
                        done_chunk(
                            &id,
                            created,
                            &model,
                            finish_reason_label(&result.finish_reason),
                        ),
                    )
                    .await?;
                } else {
                    if !matches!(result.finish_reason, FinishReason::StopSequence { .. }) {
                        let content = stop_buffer.flush();
                        if !content.is_empty() {
                            send_stream_chunk(
                                &tx,
                                content_chunk(&id, created, &model, content, None),
                            )
                            .await?;
                        }
                    }
                    send_stream_chunk(
                        &tx,
                        done_chunk(
                            &id,
                            created,
                            &model,
                            finish_reason_label(&result.finish_reason),
                        ),
                    )
                    .await?;
                }
            }
            Err(err)
                if wants_constrained_json
                    && json_constraint_stopped_incomplete_message(&err.message) =>
            {
                send_stream_chunk(
                    &tx,
                    content_chunk(&id, created, &model, "{}".to_string(), None),
                )
                .await?;
                send_stream_chunk(&tx, done_chunk(&id, created, &model, "stop")).await?;
            }
            Err(err) => {
                let error = generation_failure(err);
                tx.send(Ok(Event::default().event("error").data(
                    serde_json::to_string(&ErrorResponse {
                        error: ErrorBody {
                            message: error.message,
                            kind: error.kind,
                        },
                    })?,
                )))
                .await
                .context("stream receiver closed")?;
            }
        }

        tx.send(Ok(Event::default().data("[DONE]")))
            .await
            .context("stream receiver closed")?;
        Ok::<(), anyhow::Error>(())
    });

    Ok(Sse::new(ReceiverStream::new(rx)))
}

async fn await_driver_admission(
    admission: oneshot::Receiver<Result<(), DriverFailure>>,
) -> Result<(), ApiError> {
    match admission.await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => Err(generation_failure(error)),
        Err(_) => Err(ApiError::internal(
            "generation driver stopped before admission completed",
        )),
    }
}

#[cfg(test)]
mod stream_admission_tests {
    use super::*;
    use axum::body::to_bytes;

    fn memory_overload() -> DriverFailure {
        DriverFailure {
            message: "request_id=7 seq_id=9 available=1024 shortfall=2048".to_string(),
            kind: DriverFailureKind::MemoryOverload,
        }
    }

    async fn assert_streaming_overload_response() {
        let (tx, rx) = oneshot::channel();
        tx.send(Err(memory_overload())).unwrap();

        let error = match await_driver_admission(rx).await {
            Err(error) => error,
            Ok(()) => panic!("memory refusal must happen before constructing SSE"),
        };
        let response = error.into_response();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers()[header::RETRY_AFTER], "1");
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(body["error"]["type"], "resource_limit_error");
        assert_eq!(body["error"]["message"], MEMORY_OVERLOAD_MESSAGE);
        assert!(!body.to_string().contains("request_id"));
    }

    #[tokio::test]
    async fn streaming_completion_refusal_is_http_overload_before_sse() {
        assert_streaming_overload_response().await;
    }

    #[tokio::test]
    async fn streaming_chat_refusal_is_http_overload_before_role_chunk() {
        assert_streaming_overload_response().await;
    }

    #[tokio::test]
    async fn accepted_stream_does_not_wait_for_first_token() {
        let (admission_tx, admission_rx) = oneshot::channel();
        let (_events_tx, mut events_rx) = mpsc::channel::<DriverEvent>(1);
        admission_tx.send(Ok(())).unwrap();

        tokio::time::timeout(
            std::time::Duration::from_millis(50),
            await_driver_admission(admission_rx),
        )
        .await
        .expect("admission must not wait for output")
        .unwrap();
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(10), events_rx.recv())
                .await
                .is_err(),
            "the test driver deliberately delays its first event"
        );
    }

    #[tokio::test]
    async fn dropped_admission_sender_fails_without_hanging() {
        let (tx, rx) = oneshot::channel::<Result<(), DriverFailure>>();
        drop(tx);

        let error = tokio::time::timeout(
            std::time::Duration::from_millis(50),
            await_driver_admission(rx),
        )
        .await
        .expect("dropped sender must resolve promptly")
        .unwrap_err();
        assert_eq!(error.status, StatusCode::INTERNAL_SERVER_ERROR);
    }

    #[tokio::test]
    async fn continuous_row_refusal_is_internal_not_memory_overload() {
        let (tx, rx) = oneshot::channel();
        tx.send(Err(DriverFailure::internal(
            "continuous decode row assignment failed",
        )))
        .unwrap();

        let error = await_driver_admission(rx).await.unwrap_err();
        assert_eq!(error.status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(error.kind, "server_error");
        assert_eq!(error.retry_after_secs, None);
    }
}

pub(crate) async fn collect_generation_result(
    mut rx: mpsc::Receiver<DriverEvent>,
) -> Result<GenerateResult, DriverFailure> {
    while let Some(event) = rx.recv().await {
        match event {
            DriverEvent::Token(_) => {}
            DriverEvent::Finished(result) => return Ok(result),
            DriverEvent::Error(error) => return Err(error),
        }
    }
    Err(DriverFailure::internal(
        "generation stream ended before result",
    ))
}

fn preprocess_chat_audio(
    input: &InputAudio,
    handle: &ModelHandle,
) -> Result<MultimodalInput, ApiError> {
    let bytes = crate::audio_input::decode_chat_audio(input)
        .map_err(|err| ApiError::bad_request(format!("invalid audio input: {err}")))?;
    let spec = handle
        .multimodal
        .as_ref()
        .and_then(|multimodal| multimodal.audio.as_ref())
        .ok_or_else(|| {
            ApiError::bad_request("audio input requires a model with a declared audio input")
        })?;
    MultimodalInput::from_wav(spec, &bytes)
        .map_err(|err| ApiError::bad_request(format!("invalid audio input: {err}")))
}

async fn preprocess_chat_images(
    image_urls: &[String],
    spec: &crate::multimodal::VisionInputSpec,
    prepared: &mut PreparedGenerateRequest,
    model_max_context: Option<usize>,
    max_tokens: usize,
) -> Result<MultimodalInput, ApiError> {
    let max_prompt_tokens =
        crate::multimodal::expansion_token_budget(model_max_context, max_tokens)
            .map_err(|err| ApiError::bad_request(format!("{err:#}")))?;
    let images = crate::image_input::fetch_images(image_urls)
        .await
        .map_err(|err| ApiError::bad_request(format!("invalid image input: {err:#}")))?;
    let GeneratePrompt::TokenIds(mut token_ids) = std::mem::replace(
        &mut prepared.request.prompt,
        GeneratePrompt::TokenIds(Vec::new()),
    ) else {
        return Err(ApiError::internal(
            "What: image placeholder expansion received an untokenized prompt. Why: the server preprocessing order was violated. How: tokenize before preprocessing and expansion.",
        ));
    };
    let input = MultimodalInput::from_images(spec, &images, &mut token_ids, max_prompt_tokens)
        .map_err(|err| ApiError::bad_request(format!("invalid image input: {err:#}")))?;
    prepared.prompt_tokens = token_ids.len();
    prepared.request.prompt = GeneratePrompt::TokenIds(token_ids);
    Ok(input)
}

fn prepare_audio_generate_request(
    request: &ChatCompletionRequest,
    tokenizer: &Tokenizer,
) -> Result<PreparedGenerateRequest, ApiError> {
    let token_ids = audio_decoder_prompt(tokenizer, None)?;
    let prompt_tokens = token_ids.len();
    Ok(PreparedGenerateRequest {
        request: GenerateRequest {
            prompt: GeneratePrompt::TokenIds(token_ids),
            options: build_generate_options_with_tokenizer(request, tokenizer),
        },
        prompt_tokens,
    })
}

fn validate_request(
    request: &ChatCompletionRequest,
    config: &ServerConfig,
) -> Result<(), ApiError> {
    if request.messages.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty"));
    }
    if request.max_tokens == 0 {
        return Err(ApiError::bad_request(
            "max_tokens must be greater than zero",
        ));
    }
    if request.max_tokens > config.max_output_tokens {
        return Err(ApiError::bad_request(format!(
            "max_tokens must be less than or equal to the server cap of {}",
            config.max_output_tokens
        )));
    }
    if !request.temperature.is_finite() || request.temperature < 0.0 {
        return Err(ApiError::bad_request(
            "temperature must be finite and non-negative",
        ));
    }
    if !request.top_p.is_finite() || request.top_p < 0.0 {
        return Err(ApiError::bad_request(
            "top_p must be finite and non-negative",
        ));
    }
    if !request.min_p.is_finite() || !(0.0..=1.0).contains(&request.min_p) {
        return Err(ApiError::bad_request(
            "min_p must be finite and between 0 and 1",
        ));
    }
    if !request.top_a.is_finite() || !(0.0..=1.0).contains(&request.top_a) {
        return Err(ApiError::bad_request(
            "top_a must be finite and between 0 and 1",
        ));
    }
    if !request.typical_p.is_finite() || !(0.0..=1.0).contains(&request.typical_p) {
        return Err(ApiError::bad_request(
            "typical_p must be finite and between 0 and 1",
        ));
    }
    if !request.repetition_penalty.is_finite() || request.repetition_penalty <= 0.0 {
        return Err(ApiError::bad_request(
            "repetition_penalty must be finite and greater than zero",
        ));
    }
    if !request.frequency_penalty.is_finite() {
        return Err(ApiError::bad_request("frequency_penalty must be finite"));
    }
    if !request.presence_penalty.is_finite() {
        return Err(ApiError::bad_request("presence_penalty must be finite"));
    }
    if !request.dry_multiplier.is_finite() || request.dry_multiplier < 0.0 {
        return Err(ApiError::bad_request(
            "dry_multiplier must be finite and non-negative",
        ));
    }
    if !request.dry_base.is_finite() || request.dry_base < 1.0 {
        return Err(ApiError::bad_request(
            "dry_base must be finite and at least 1",
        ));
    }
    if request.dry_allowed_length == 0 {
        return Err(ApiError::bad_request(
            "dry_allowed_length must be greater than zero",
        ));
    }
    if request.mirostat > 2 {
        return Err(ApiError::bad_request("mirostat must be 0, 1, or 2"));
    }
    if !request.mirostat_tau.is_finite() || request.mirostat_tau <= 0.0 {
        return Err(ApiError::bad_request(
            "mirostat_tau must be finite and greater than zero",
        ));
    }
    if !request.mirostat_eta.is_finite() || request.mirostat_eta <= 0.0 {
        return Err(ApiError::bad_request(
            "mirostat_eta must be finite and greater than zero",
        ));
    }
    if !request.xtc_probability.is_finite() || !(0.0..=1.0).contains(&request.xtc_probability) {
        return Err(ApiError::bad_request(
            "xtc_probability must be finite and between 0 and 1",
        ));
    }
    if !request.xtc_threshold.is_finite() || !(0.0..=1.0).contains(&request.xtc_threshold) {
        return Err(ApiError::bad_request(
            "xtc_threshold must be finite and between 0 and 1",
        ));
    }
    if request
        .top_logprobs
        .is_some_and(|count| count > MAX_CHAT_TOP_LOGPROBS)
    {
        return Err(ApiError::bad_request(format!(
            "top_logprobs must be between 0 and {MAX_CHAT_TOP_LOGPROBS}"
        )));
    }
    if request.top_logprobs.is_some() && !request.logprobs {
        return Err(ApiError::bad_request(
            "top_logprobs requires logprobs to be true",
        ));
    }
    validate_tool_choice(request)?;
    Ok(())
}

fn validate_completion_request(
    request: &CompletionRequest,
    config: &ServerConfig,
) -> Result<(), ApiError> {
    if request.max_tokens == 0 {
        return Err(ApiError::bad_request(
            "max_tokens must be greater than zero",
        ));
    }
    if request.max_tokens > config.max_output_tokens {
        return Err(ApiError::bad_request(format!(
            "max_tokens must be less than or equal to the server cap of {}",
            config.max_output_tokens
        )));
    }
    if !request.temperature.is_finite() || request.temperature < 0.0 {
        return Err(ApiError::bad_request(
            "temperature must be finite and non-negative",
        ));
    }
    if !request.top_p.is_finite() || request.top_p < 0.0 {
        return Err(ApiError::bad_request(
            "top_p must be finite and non-negative",
        ));
    }
    if !request.min_p.is_finite() || !(0.0..=1.0).contains(&request.min_p) {
        return Err(ApiError::bad_request(
            "min_p must be finite and between 0 and 1",
        ));
    }
    if !request.frequency_penalty.is_finite() {
        return Err(ApiError::bad_request("frequency_penalty must be finite"));
    }
    if !request.presence_penalty.is_finite() {
        return Err(ApiError::bad_request("presence_penalty must be finite"));
    }
    if request
        .logprobs
        .is_some_and(|count| count > MAX_COMPLETION_LOGPROBS)
    {
        return Err(ApiError::bad_request(format!(
            "logprobs must be between 0 and {MAX_COMPLETION_LOGPROBS}"
        )));
    }
    Ok(())
}

fn enforce_context_cap(
    prompt_tokens: usize,
    max_tokens: usize,
    model_max_context: Option<usize>,
) -> Result<(), ApiError> {
    let Some(model_max_context) = model_max_context else {
        return Ok(());
    };
    let total = prompt_tokens
        .checked_add(max_tokens)
        .ok_or_else(|| {
            ApiError::bad_request(
                "What: request admission length overflowed. Why: final prefill length plus max_tokens does not fit usize. How: reduce the prompt, image expansion size, or max_tokens.",
            )
        })?;
    if total > model_max_context {
        return Err(ApiError::bad_request(format!(
            "What: request admission exceeded the model context limit. \
             Why: final prefill length ({prompt_tokens}) after placeholder expansion plus max_tokens ({max_tokens}) is {total}, above {model_max_context}. \
             How: reduce the prompt, image count/expansion size, or max_tokens."
        )));
    }
    Ok(())
}

fn validate_tool_choice(request: &ChatCompletionRequest) -> Result<(), ApiError> {
    let Some(tool_choice) = &request.tool_choice else {
        return Ok(());
    };
    match tool_choice {
        ToolChoice::Mode(ToolChoiceMode::Required) => {
            if !request
                .tools
                .as_ref()
                .is_some_and(|tools| tools.iter().any(|tool| tool.kind == "function"))
            {
                return Err(ApiError::bad_request(
                    "tool_choice required requires at least one function tool",
                ));
            }
        }
        ToolChoice::Specific(choice) => {
            if choice.kind != "function" {
                return Err(ApiError::bad_request(
                    "specific tool_choice type must be function",
                ));
            }
            if !request.tools.as_ref().is_some_and(|tools| {
                tools.iter().any(|tool| {
                    tool.kind == "function" && tool.function.name == choice.function.name
                })
            }) {
                return Err(ApiError::bad_request(format!(
                    "tool_choice function '{}' was not provided in tools",
                    choice.function.name
                )));
            }
        }
        ToolChoice::Mode(ToolChoiceMode::Auto | ToolChoiceMode::None) => {}
    }
    Ok(())
}

fn session_id_from_headers(headers: &HeaderMap) -> Result<Option<String>, ApiError> {
    let Some(value) = headers.get(SESSION_ID_HEADER) else {
        return Ok(None);
    };
    let session_id = value
        .to_str()
        .map_err(|_| ApiError::bad_request("X-Session-Id must be valid UTF-8"))?
        .trim();
    if session_id.is_empty() {
        return Err(ApiError::bad_request("X-Session-Id must not be empty"));
    }
    if session_id.len() > MAX_SESSION_ID_LEN {
        return Err(ApiError::bad_request(format!(
            "X-Session-Id must be at most {MAX_SESSION_ID_LEN} bytes"
        )));
    }
    Ok(Some(session_id.to_string()))
}

async fn get_or_create_session(
    engine: &EngineDriver,
    sessions: &SessionRegistry,
    client_id: &str,
) -> Result<SessionId, ApiError> {
    if let Some(engine_session_id) = sessions
        .get(client_id)
        .map_err(|err| ApiError::internal(format!("session registry failed: {err}")))?
    {
        return Ok(engine_session_id);
    }

    let engine_session_id = engine
        .create_session()
        .await
        .map_err(|err| ApiError::internal(format!("session create failed: {err}")))?;
    let evicted = sessions
        .insert(client_id.to_string(), engine_session_id)
        .map_err(|err| ApiError::internal(format!("session registry failed: {err}")))?;
    close_evicted_session(engine, evicted).await?;
    Ok(engine_session_id)
}

pub fn build_generate_request(request: &ChatCompletionRequest) -> GenerateRequest {
    GenerateRequest {
        prompt: GeneratePrompt::Text(build_prompt(request)),
        options: build_generate_options(request),
    }
}

pub(crate) fn prepare_completion(
    request: &CompletionRequest,
    handle: &ModelHandle,
) -> Result<PreparedCompletion, ApiError> {
    let mut options = build_completion_options(request, &handle.tokenizer);
    options.max_context = handle.model_max_context;
    // Honor the model's declared sampling regime (e.g. a reasoning model that
    // ships do_sample=true); explicit request fields still win.
    options.resolve_sampling_defaults(
        handle.generation_defaults.as_ref(),
        &completion_sampling_overrides(request),
    );
    if let Some(suffix) = request.suffix.as_ref() {
        let fim_config = handle
            .fim_config
            .as_ref()
            .ok_or_else(|| ApiError::bad_request("FIM is not supported by this model"))?;
        let prompt = fim_config.format_prompt(&request.prompt, suffix);
        let prompt_tokens = tokenize_prompt(&handle.tokenizer, &prompt)?;
        Ok(PreparedCompletion {
            generation: CompletionGeneration::Fim {
                prefix: request.prompt.clone(),
                suffix: suffix.clone(),
                options,
            },
            prompt_tokens,
        })
    } else {
        let token_ids = handle
            .tokenizer
            .encode(&request.prompt)
            .map_err(|err| ApiError::internal(format!("prompt tokenization failed: {err}")))?;
        let prompt_tokens = token_ids.len();
        Ok(PreparedCompletion {
            generation: CompletionGeneration::Plain(GenerateRequest {
                prompt: GeneratePrompt::TokenIds(token_ids),
                options,
            }),
            prompt_tokens,
        })
    }
}

async fn submit_completion(
    handle: &ModelHandle,
    sessions: &SessionRegistry,
    generation: CompletionGeneration,
    client_session_id: Option<&str>,
) -> Result<DriverGeneration, ApiError> {
    match generation {
        CompletionGeneration::Plain(request) => {
            let session_id = if let Some(id) = client_session_id {
                Some(get_or_create_session(&handle.engine, sessions, id).await?)
            } else {
                None
            };
            handle
                .engine
                .generate(session_id, request)
                .await
                .map_err(map_generate_submit_error)
        }
        CompletionGeneration::Fim {
            prefix,
            suffix,
            options,
        } => {
            let fim_config = handle
                .fim_config
                .clone()
                .ok_or_else(|| ApiError::bad_request("FIM is not supported by this model"))?;
            handle
                .engine
                .generate_fim(prefix, suffix, fim_config, options)
                .await
                .map_err(map_generate_submit_error)
        }
    }
}

fn tokenize_prompt(tokenizer: &Tokenizer, prompt: &str) -> Result<usize, ApiError> {
    tokenizer
        .encode(prompt)
        .map(|tokens| tokens.len())
        .map_err(|err| ApiError::internal(format!("prompt tokenization failed: {err}")))
}

fn build_completion_options(request: &CompletionRequest, tokenizer: &Tokenizer) -> GenerateOptions {
    let mut options = GenerateOptions {
        max_new_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        min_p: request.min_p,
        frequency_penalty: request.frequency_penalty,
        presence_penalty: request.presence_penalty,
        top_logprobs: request.logprobs,
        ..GenerateOptions::default()
    };
    if let Some(stop) = request.stop.clone() {
        options.stop_sequences = stop.into_sequences();
    }
    add_tokenizer_stop_sequences(&mut options, tokenizer);
    options
}

/// The caller's *explicit* sampling selections for a text completion, feeding
/// [`GenerateOptions::resolve_sampling_defaults`].
///
/// Completions historically decoded greedily by default. `temperature: 0`
/// keeps that as an explicit greedy request; `min_p > 0` (the only stochastic
/// knob the completions schema exposes) is an explicit request to sample.
/// Otherwise the greedy decision is deferred (`None`) so a model that declares
/// `do_sample: true` samples instead of looping under forced greedy.
///
/// `temperature` and `top_p` are passed as explicit values because the
/// OpenAI-compatible schema always supplies them (with documented defaults) and
/// has no "unspecified" state, so the API defaults win over any model-declared
/// temperature/top_p; only the greedy decision defers to the model.
fn completion_sampling_overrides(request: &CompletionRequest) -> SamplingOverrides {
    let greedy = if request.temperature == 0.0 {
        Some(true)
    } else if request.min_p > 0.0 {
        Some(false)
    } else {
        None
    };
    SamplingOverrides {
        greedy,
        temperature: Some(request.temperature),
        top_p: Some(request.top_p),
        top_k: None,
    }
}
/// The text spelling of this model's image placeholder, if it declares one.
///
/// Decoded from the declared token id rather than hardcoded, so the prompt
/// carries whatever token this package actually uses. It re-tokenizes to the
/// same id because a placeholder is always a distinct vocabulary entry.
pub(crate) fn image_placeholder_text(handle: &ModelHandle) -> Option<String> {
    let token = handle
        .multimodal
        .as_ref()
        .and_then(|multimodal| multimodal.vision.as_ref())
        .and_then(|vision| vision.placeholder_token_id())?;
    handle.tokenizer.decode_with_special_tokens(&[token]).ok()
}

/// The placeholder to render at each image part's position, or `None` when the
/// request positions its images some other way.
///
/// A caller who writes the placeholder into the text has already said where the
/// images go; rendering another one per image part would double the count and
/// the request would be rejected. So a hand-written placeholder wins, and
/// automatic positioning applies only when the text contains none.
fn positional_image_placeholder(
    request: &ChatCompletionRequest,
    handle: &ModelHandle,
) -> Option<String> {
    let placeholder = image_placeholder_text(handle)?;
    let already_positioned = request.messages.iter().any(|message| {
        message
            .content
            .as_ref()
            .is_some_and(|content| content.render(None).contains(&placeholder))
    });
    (!already_positioned).then_some(placeholder)
}

fn prepare_generate_request(
    request: &ChatCompletionRequest,
    tokenizer: &Tokenizer,
    chat_template: Option<&ChatTemplate>,
    session: bool,
    image_placeholder: Option<&str>,
    generation_defaults: Option<&GenerationDefaults>,
) -> anyhow::Result<PreparedGenerateRequest> {
    let prompt = if session && !request.has_tool_context() {
        build_session_prompt(&request.messages, image_placeholder)
    } else {
        render_prompt(request, chat_template, image_placeholder)?
    };
    let token_ids = tokenizer
        .encode(&prompt)
        .map_err(|e| anyhow::anyhow!("Failed to tokenize prompt: {e}"))?;
    let prompt_tokens = token_ids.len();
    let mut options = build_generate_options_with_tokenizer(request, tokenizer);
    // Honor the model's declared sampling regime (e.g. a reasoning model that
    // ships do_sample=true); explicit request fields still win.
    options.resolve_sampling_defaults(generation_defaults, &chat_sampling_overrides(request));
    Ok(PreparedGenerateRequest {
        request: GenerateRequest {
            prompt: GeneratePrompt::TokenIds(token_ids),
            options,
        },
        prompt_tokens,
    })
}

fn build_generate_options(request: &ChatCompletionRequest) -> GenerateOptions {
    // Chat historically used greedy decoding even with its default temperature
    // and top-p fields. Preserve that default while treating the newly exposed
    // stochastic controls as an explicit request to sample.
    let stochastic_sampling_requested = request.seed.is_some()
        || request.top_k > 0
        || request.min_p > 0.0
        || request.top_a > 0.0
        || request.typical_p < 1.0
        || request.mirostat > 0
        || request.xtc_probability > 0.0;
    let mut options = GenerateOptions {
        max_new_tokens: request.max_tokens,
        temperature: request.temperature,
        top_p: request.top_p,
        top_k: request.top_k,
        min_p: request.min_p,
        top_a: request.top_a,
        typical_p: request.typical_p,
        repetition_penalty: request.repetition_penalty,
        frequency_penalty: request.frequency_penalty,
        presence_penalty: request.presence_penalty,
        greedy: request.temperature == 0.0 || !stochastic_sampling_requested,
        seed: request.seed,
        dry: (request.dry_multiplier > 0.0).then(|| DryConfig {
            multiplier: request.dry_multiplier,
            base: request.dry_base,
            allowed_length: request.dry_allowed_length,
            sequence_breakers: request.dry_sequence_breakers.clone(),
        }),
        mirostat: match request.mirostat {
            1 => Some(MirostatConfig {
                tau: request.mirostat_tau,
                eta: request.mirostat_eta,
                version: MirostatVersion::V1,
            }),
            2 => Some(MirostatConfig {
                tau: request.mirostat_tau,
                eta: request.mirostat_eta,
                version: MirostatVersion::V2,
            }),
            _ => None,
        },
        xtc: (request.xtc_probability > 0.0).then_some(XtcConfig {
            probability: request.xtc_probability,
            threshold: request.xtc_threshold,
        }),
        top_logprobs: request
            .logprobs
            .then_some(request.top_logprobs.unwrap_or(0)),
        ..GenerateOptions::default()
    };
    if let Some(stop) = request.stop.clone() {
        options.stop_sequences = stop.into_sequences();
    }
    if let Some(constraint) = response_format_constraint(request) {
        options.constraint = Some(constraint);
    }
    if let Some(constraint) = forced_tool_choice_constraint(request) {
        options.constraint = Some(constraint);
    }
    options
}

/// The caller's *explicit* sampling selections for a chat completion, feeding
/// [`GenerateOptions::resolve_sampling_defaults`].
///
/// Mirrors [`build_generate_options`]'s greedy heuristic as the *explicit* half
/// of the precedence contract: `temperature: 0` forces greedy, any exposed
/// stochastic control (seed, top_k, min_p, top_a, typical_p, mirostat, xtc) is
/// an explicit request to sample, and a request that carries none of these
/// defers the greedy decision (`None`) to the model's declared `do_sample`.
///
/// `temperature`/`top_p`/`top_k` are passed as explicit values: the
/// OpenAI-compatible schema always supplies them with documented defaults and
/// cannot express "unspecified", so the API defaults win over any model-declared
/// temperature/top_p/top_k, and only the greedy (do_sample) decision defers to
/// the model.
fn chat_sampling_overrides(request: &ChatCompletionRequest) -> SamplingOverrides {
    let requests_sampling = request.seed.is_some()
        || request.top_k > 0
        || request.min_p > 0.0
        || request.top_a > 0.0
        || request.typical_p < 1.0
        || request.mirostat > 0
        || request.xtc_probability > 0.0;
    let greedy = if request.temperature == 0.0 {
        Some(true)
    } else if requests_sampling {
        Some(false)
    } else {
        None
    };
    SamplingOverrides {
        greedy,
        temperature: Some(request.temperature),
        top_p: Some(request.top_p),
        // `top_k` is an extension the OpenAI schema never carries, and 0 is its
        // "disabled" sentinel rather than a caller's choice. Treat an absent
        // `top_k` as unspecified so a package that declares one keeps it,
        // instead of every OpenAI client silently widening the model to the
        // full vocabulary.
        top_k: (request.top_k > 0).then_some(request.top_k),
    }
}

fn response_format_constraint(request: &ChatCompletionRequest) -> Option<GenerateConstraint> {
    match request.response_format.as_ref()? {
        ResponseFormat::JsonObject => Some(GenerateConstraint::Json),
        ResponseFormat::JsonSchema { json_schema } => serde_json::to_string(&json_schema.schema)
            .ok()
            .map(GenerateConstraint::JsonSchema),
        ResponseFormat::Text => None,
    }
}

fn forced_tool_choice_constraint(request: &ChatCompletionRequest) -> Option<GenerateConstraint> {
    let schemas = forced_tool_choice_schemas(request)?;
    let schema = if schemas.len() == 1 {
        schemas.into_iter().next()?
    } else {
        serde_json::json!({ "anyOf": schemas })
    };
    let schema = serde_json::to_string(&schema).ok()?;
    Some(GenerateConstraint::Lark(format!(
        "start: \"<tool_call>\\n\" tool \"\\n</tool_call>\"\ntool: %json {schema}\n"
    )))
}

fn forced_tool_choice_schemas(request: &ChatCompletionRequest) -> Option<Vec<serde_json::Value>> {
    let tools = request
        .tools
        .as_ref()?
        .iter()
        .filter(|tool| tool.kind == "function");
    let selected = match request.tool_choice.as_ref()? {
        ToolChoice::Mode(ToolChoiceMode::Required) => tools.collect::<Vec<_>>(),
        ToolChoice::Specific(choice) if choice.kind == "function" => tools
            .filter(|tool| tool.function.name == choice.function.name)
            .collect::<Vec<_>>(),
        ToolChoice::Mode(ToolChoiceMode::Auto | ToolChoiceMode::None) | ToolChoice::Specific(_) => {
            Vec::new()
        }
    };

    let schemas = selected
        .into_iter()
        .map(tool_call_schema_for_tool)
        .collect::<Vec<_>>();
    (!schemas.is_empty()).then_some(schemas)
}

fn tool_call_schema_for_tool(tool: &ChatTool) -> serde_json::Value {
    let arguments_schema = tool
        .function
        .parameters
        .clone()
        .unwrap_or_else(|| serde_json::json!({ "type": "object" }));
    serde_json::json!({
        "type": "object",
        "properties": {
            "name": { "enum": [tool.function.name.clone()] },
            "arguments": arguments_schema
        },
        "required": ["name", "arguments"],
        "additionalProperties": false
    })
}

fn build_generate_options_with_tokenizer(
    request: &ChatCompletionRequest,
    tokenizer: &Tokenizer,
) -> GenerateOptions {
    let mut options = build_generate_options(request);
    add_tokenizer_stop_sequences(&mut options, tokenizer);
    options
}

fn add_tokenizer_stop_sequences(options: &mut GenerateOptions, tokenizer: &Tokenizer) {
    let eos_token_ids = tokenizer.eos_token_ids();
    if let Some(first) = eos_token_ids.first().copied() {
        options.eos_token_id = Some(first);
    }
    for eos_token_id in eos_token_ids {
        let eos_sequence = StopSequence::Tokens(vec![eos_token_id]);
        if !options.stop_sequences.contains(&eos_sequence) {
            options.stop_sequences.push(eos_sequence);
        }
    }
    if let Some(im_end_id) = tokenizer.token_id("<|im_end|>") {
        let im_end_sequence = StopSequence::Tokens(vec![im_end_id]);
        if !options.stop_sequences.contains(&im_end_sequence) {
            options.stop_sequences.push(im_end_sequence);
        }
    }
}

fn json_constraint_stopped_incomplete_message(message: &str) -> bool {
    message.contains("JSON constrained decoding stopped before a complete JSON value")
}

fn tools_parseable_from_output(request: &ChatCompletionRequest) -> bool {
    !matches!(
        request.tool_choice,
        Some(ToolChoice::Mode(ToolChoiceMode::None))
    )
}

fn tools_offered_to_model(request: &ChatCompletionRequest) -> Option<&Vec<ChatTool>> {
    if matches!(
        request.tool_choice,
        Some(ToolChoice::Mode(ToolChoiceMode::None))
    ) {
        return None;
    }
    request.tools.as_ref().filter(|tools| !tools.is_empty())
}

fn build_session_prompt(messages: &[ChatMessage], image_placeholder: Option<&str>) -> String {
    messages
        .last()
        .and_then(|message| message.content.as_ref())
        .map(|content| content.render(image_placeholder))
        .unwrap_or_default()
}

/// Translate one OpenAI chat message into the value a chat template renders.
///
/// Tool identity travels with the message because a tool-calling template names
/// each result from the `name`, or resolves it from the `tool_call_id` against
/// the assistant call it answers; dropping either renders an unnamed result.
fn template_message(
    message: &ChatMessage,
    image_placeholder: Option<&str>,
) -> anyhow::Result<TemplateChatMessage> {
    let mut template_message = TemplateChatMessage::new(
        message.role.as_str(),
        message
            .content
            .as_ref()
            .map(|content| content.render(image_placeholder))
            .unwrap_or_default(),
    )
    .with_tool_result(message.name.clone(), message.tool_call_id.clone());
    if let Some(tool_calls) = &message.tool_calls {
        template_message = template_message.with_tool_calls(template_tool_calls(tool_calls)?);
    }
    Ok(template_message)
}

/// Assistant tool calls in the shape Hugging Face chat templates expect.
///
/// OpenAI carries `function.arguments` as a JSON *string*, while chat templates
/// index it as a mapping (`args.items()`), and the jinja sandbox cannot parse a
/// string back into one. So decode the arguments here, where the wire format is
/// known, and leave anything that is not a JSON object as the original string
/// rather than guessing at a shape the caller did not send.
fn template_tool_calls(tool_calls: &[ChatMessageToolCall]) -> anyhow::Result<serde_json::Value> {
    let mut value = serde_json::to_value(tool_calls)?;
    for call in value.as_array_mut().into_iter().flatten() {
        let Some(arguments) = call
            .pointer("/function/arguments")
            .and_then(serde_json::Value::as_str)
        else {
            continue;
        };
        let Ok(decoded @ serde_json::Value::Object(_)) =
            serde_json::from_str::<serde_json::Value>(arguments)
        else {
            continue;
        };
        if let Some(slot) = call.pointer_mut("/function/arguments") {
            *slot = decoded;
        }
    }
    Ok(value)
}

fn render_prompt(
    request: &ChatCompletionRequest,
    chat_template: Option<&ChatTemplate>,
    image_placeholder: Option<&str>,
) -> anyhow::Result<String> {
    if let Some(chat_template) = chat_template {
        let messages = request
            .messages
            .iter()
            .map(|message| template_message(message, image_placeholder))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let tools_json = tools_offered_to_model(request)
            .map(serde_json::to_string)
            .transpose()?;
        return chat_template
            .render_with_reasoning_effort(
                &messages,
                tools_json.as_deref(),
                true,
                request.reasoning_effort.map(ReasoningEffort::as_str),
            )
            .map_err(|err| anyhow::anyhow!("chat template render failed: {err}"));
    }
    Ok(build_prompt(request))
}

/// Build the Phase 2 chat prompt with a simple role-tagged template:
/// `<|role|>\n{content}\n` for every message, followed by `<|assistant|>\n`.
/// Model-specific templates will replace this once tokenizer chat templates are wired.
pub fn build_prompt(request: &ChatCompletionRequest) -> String {
    let mut prompt = String::new();
    if let Some(tools) = tools_offered_to_model(request) {
        prompt.push_str("<|tools|>\n");
        prompt.push_str(&serde_json::to_string(tools).unwrap_or_else(|_| "[]".to_string()));
        prompt.push('\n');
    }
    if let Some(tool_choice) = &request.tool_choice {
        prompt.push_str("<|tool_choice|>\n");
        prompt.push_str(&tool_choice_prompt(tool_choice));
        prompt.push('\n');
    }
    for message in &request.messages {
        prompt.push_str("<|");
        prompt.push_str(message.role.trim());
        prompt.push_str("|>\n");
        if let Some(tool_call_id) = &message.tool_call_id {
            prompt.push_str("tool_call_id: ");
            prompt.push_str(tool_call_id);
            prompt.push('\n');
        }
        if let Some(content) = &message.content {
            prompt.push_str(&content.text());
        }
        if let Some(tool_calls) = &message.tool_calls {
            if message.content.is_some() {
                prompt.push('\n');
            }
            prompt
                .push_str(&serde_json::to_string(tool_calls).unwrap_or_else(|_| "[]".to_string()));
        }
        prompt.push('\n');
    }
    prompt.push_str("<|assistant|>\n");
    prompt
}

fn tool_choice_prompt(tool_choice: &ToolChoice) -> String {
    match tool_choice {
        ToolChoice::Mode(mode) => match mode {
            ToolChoiceMode::Auto => "auto".to_string(),
            ToolChoiceMode::None => "none".to_string(),
            ToolChoiceMode::Required => "required".to_string(),
        },
        ToolChoice::Specific(choice) => format!("function: {}", choice.function.name),
    }
}

pub fn parse_tool_calls(output: &str) -> Vec<ChatMessageToolCall> {
    // Model families do not normally mix formats. When they do, use ATEM,
    // Qwen, Llama, then Mistral order so generated call IDs remain deterministic.
    let parsed_calls = extract_atem_tool_calls(output)
        .into_iter()
        .chain(extract_qwen_tool_calls(output))
        .into_iter()
        .chain(extract_llama_tool_calls(output))
        .chain(extract_mistral_tool_calls(output));
    let mut calls = Vec::new();
    for value in parsed_calls {
        if let Some(call) = parsed_tool_call_to_openai(calls.len(), value) {
            calls.push(call);
        }
    }
    calls
}

fn extract_atem_tool_calls(output: &str) -> Vec<serde_json::Value> {
    const INVOKE: &str = "<atem:invoke";
    const INVOKE_CLOSE: &str = "</atem:invoke>";
    const PARAMETER: &str = "<atem:parameter";
    const PARAMETER_CLOSE: &str = "</atem:parameter>";

    let mut calls = Vec::new();
    let mut rest = output;
    while let Some(start) = rest.find(INVOKE) {
        rest = &rest[start + INVOKE.len()..];
        let Some(tag_end) = rest.find('>') else {
            break;
        };
        let Some(name) = tag_attribute(&rest[..tag_end], "name") else {
            rest = &rest[tag_end + 1..];
            continue;
        };
        rest = &rest[tag_end + 1..];
        let Some(invoke_end) = rest.find(INVOKE_CLOSE) else {
            break;
        };
        let body = &rest[..invoke_end];
        let mut arguments = serde_json::Map::new();
        let mut parameters = body;
        while let Some(parameter_start) = parameters.find(PARAMETER) {
            parameters = &parameters[parameter_start + PARAMETER.len()..];
            let Some(parameter_tag_end) = parameters.find('>') else {
                break;
            };
            let Some(key) = tag_attribute(&parameters[..parameter_tag_end], "name") else {
                parameters = &parameters[parameter_tag_end + 1..];
                continue;
            };
            parameters = &parameters[parameter_tag_end + 1..];
            let Some(parameter_end) = parameters.find(PARAMETER_CLOSE) else {
                break;
            };
            let raw_value = &parameters[..parameter_end];
            let value = serde_json::from_str(raw_value.trim())
                .unwrap_or_else(|_| serde_json::Value::String(raw_value.to_string()));
            arguments.insert(key, value);
            parameters = &parameters[parameter_end + PARAMETER_CLOSE.len()..];
        }
        calls.push(serde_json::json!({
            "name": name,
            "arguments": arguments,
        }));
        rest = &rest[invoke_end + INVOKE_CLOSE.len()..];
    }
    calls
}

fn tag_attribute(tag: &str, attribute: &str) -> Option<String> {
    let marker = format!("{attribute}=\"");
    let value = tag.split_once(&marker)?.1;
    Some(value.split_once('"')?.0.to_string())
}

fn extract_qwen_tool_calls(output: &str) -> Vec<serde_json::Value> {
    let mut values = Vec::new();
    let mut rest = output;
    while let Some(start) = rest.find("<tool_call>") {
        rest = &rest[start + "<tool_call>".len()..];
        let Some(end) = rest.find("</tool_call>") else {
            break;
        };
        let body = rest[..end].trim();
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(body) {
            values.push(value);
        }
        rest = &rest[end + "</tool_call>".len()..];
    }
    values
}

fn extract_llama_tool_calls(output: &str) -> Vec<serde_json::Value> {
    const MARKER: &str = "<|python_tag|>";
    let mut values = Vec::new();
    let mut rest = output;
    while let Some(start) = rest.find(MARKER) {
        rest = &rest[start + MARKER.len()..];
        let mut json = rest;
        loop {
            json = json.trim_start();
            if let Some(after_separator) = json.strip_prefix(';') {
                json = after_separator;
                continue;
            }
            if json.is_empty() || json.starts_with("<|") {
                break;
            }
            let Some((value, consumed)) = parse_json_value_prefix(json) else {
                break;
            };
            values.push(value);
            json = &json[consumed..];
        }
    }
    values
}

fn extract_mistral_tool_calls(output: &str) -> Vec<serde_json::Value> {
    const MARKER: &str = "[TOOL_CALLS]";
    let mut values = Vec::new();
    let mut rest = output;
    while let Some(start) = rest.find(MARKER) {
        rest = &rest[start + MARKER.len()..];
        if let Some((serde_json::Value::Array(calls), _)) =
            parse_json_value_prefix(rest.trim_start())
        {
            values.extend(calls);
        }
    }
    values
}

fn parse_json_value_prefix(input: &str) -> Option<(serde_json::Value, usize)> {
    let mut stream = serde_json::Deserializer::from_str(input).into_iter::<serde_json::Value>();
    let value = stream.next()?.ok()?;
    Some((value, stream.byte_offset()))
}

#[derive(Debug, Clone)]
pub struct ParsedAssistantOutput {
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ChatMessageToolCall>>,
    pub finish_reason: &'static str,
}

pub fn parse_assistant_output(
    output: String,
    default_finish_reason: &'static str,
) -> ParsedAssistantOutput {
    // OpenAI tool calls end the assistant turn. The batch row finishes normally
    // with finish_reason=tool_calls; role=tool follow-up messages are submitted
    // as a new turn rather than pausing and resuming mid-token.
    let tool_calls = parse_tool_calls(&output);
    if tool_calls.is_empty() {
        ParsedAssistantOutput {
            content: Some(atem_visible_content(&output).unwrap_or(output)),
            tool_calls: None,
            finish_reason: default_finish_reason,
        }
    } else {
        ParsedAssistantOutput {
            content: None,
            tool_calls: Some(tool_calls),
            finish_reason: "tool_calls",
        }
    }
}

/// Private ATEM reasoning channel, which a client must never be shown.
const ATEM_REASONING_CHANNEL: &str = "to=self<|message|>";
/// The ATEM channel addressed to the caller, i.e. the visible answer.
const ATEM_USER_CHANNEL: &str = "to=user<|message|>";

/// The part of an ATEM turn a client may see, or `None` for output that carries
/// no ATEM channel at all.
///
/// A turn is a sequence of addressed channels, and only the one addressed to the
/// user is an answer. A turn that ends before reaching it — truncated by the
/// token budget, say — therefore produced no answer, and yields empty content
/// rather than leaking the model's private reasoning as if it were one.
fn atem_visible_content(output: &str) -> Option<String> {
    if let Some((_, answer)) = output.rsplit_once(ATEM_USER_CHANNEL) {
        let end = ["<|eot|>", "<|eom|>"]
            .into_iter()
            .filter_map(|marker| answer.find(marker))
            .min()
            .unwrap_or(answer.len());
        return Some(answer[..end].to_string());
    }
    output.contains(ATEM_REASONING_CHANNEL).then(String::new)
}

fn assistant_output_text(result: &GenerateResult, tokenizer: &Tokenizer) -> String {
    let Ok(with_special_tokens) = tokenizer.decode_with_special_tokens(&result.token_ids) else {
        return result.text.clone();
    };
    if with_special_tokens.contains(ATEM_USER_CHANNEL)
        || with_special_tokens.contains(ATEM_REASONING_CHANNEL)
        || with_special_tokens.contains("<atem:invoke")
    {
        with_special_tokens
    } else {
        result.text.clone()
    }
}

fn parsed_tool_call_to_openai(
    index: usize,
    value: serde_json::Value,
) -> Option<ChatMessageToolCall> {
    let name = value.get("name")?.as_str()?.to_string();
    let arguments = value
        .get("arguments")
        .or_else(|| value.get("parameters"))
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    Some(ChatMessageToolCall {
        id: format!("call_{index}"),
        kind: "function".to_string(),
        function: ChatMessageToolCallFunction {
            name,
            arguments: serde_json::to_string(&arguments).ok()?,
        },
    })
}

fn chat_logprobs(
    result: &GenerateResult,
    tokenizer: &Tokenizer,
    requested_top_logprobs: Option<usize>,
) -> anyhow::Result<Option<ChatLogprobs>> {
    let Some(requested_top_logprobs) = requested_top_logprobs else {
        return Ok(None);
    };
    let logprobs = result
        .logprobs
        .as_deref()
        .context("engine did not return requested logprobs")?;
    if logprobs.len() != result.token_ids.len() {
        anyhow::bail!(
            "engine returned {} logprob records for {} generated tokens",
            logprobs.len(),
            result.token_ids.len()
        );
    }
    let content = logprobs
        .iter()
        .map(|entry| chat_token_logprob(tokenizer, entry, requested_top_logprobs))
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(Some(ChatLogprobs { content }))
}

fn chat_token_logprob(
    tokenizer: &Tokenizer,
    entry: &TokenLogprob,
    requested_top_logprobs: usize,
) -> anyhow::Result<ChatTokenLogprob> {
    let token = decode_logprob_token(tokenizer, entry.token_id)?;
    let top_logprobs = entry
        .top
        .iter()
        .take(requested_top_logprobs)
        .map(|&(token_id, logprob)| {
            let token = decode_logprob_token(tokenizer, token_id)?;
            Ok(ChatTopLogprob {
                bytes: token.as_bytes().to_vec(),
                token,
                logprob,
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(ChatTokenLogprob {
        bytes: token.as_bytes().to_vec(),
        token,
        logprob: entry.logprob,
        top_logprobs,
    })
}

fn completion_logprobs(
    result: &GenerateResult,
    tokenizer: &Tokenizer,
    requested_top_logprobs: Option<usize>,
) -> anyhow::Result<Option<CompletionLogprobs>> {
    let Some(requested_top_logprobs) = requested_top_logprobs else {
        return Ok(None);
    };
    let logprobs = result
        .logprobs
        .as_deref()
        .context("engine did not return requested logprobs")?;
    if logprobs.len() != result.token_ids.len() {
        anyhow::bail!(
            "engine returned {} logprob records for {} generated tokens",
            logprobs.len(),
            result.token_ids.len()
        );
    }

    let mut tokens = Vec::with_capacity(logprobs.len());
    let mut token_logprobs = Vec::with_capacity(logprobs.len());
    let mut top_logprobs = Vec::with_capacity(logprobs.len());
    let mut text_offset = Vec::with_capacity(logprobs.len());
    let mut offset = 0;
    for entry in logprobs {
        let token = decode_logprob_token(tokenizer, entry.token_id)?;
        text_offset.push(offset);
        offset += token.len();
        tokens.push(token);
        token_logprobs.push(entry.logprob);
        top_logprobs.push(
            entry
                .top
                .iter()
                .take(requested_top_logprobs)
                .map(|&(token_id, logprob)| {
                    Ok((decode_logprob_token(tokenizer, token_id)?, logprob))
                })
                .collect::<anyhow::Result<_>>()?,
        );
    }
    Ok(Some(CompletionLogprobs {
        tokens,
        token_logprobs,
        top_logprobs,
        text_offset,
    }))
}

fn decode_logprob_token(tokenizer: &Tokenizer, token_id: u32) -> anyhow::Result<String> {
    let decoded = tokenizer
        .decode(&[token_id])
        .with_context(|| format!("failed to decode token id {token_id}"))?;
    if !decoded.is_empty() {
        return Ok(decoded);
    }
    tokenizer
        .inner()
        .id_to_token(token_id)
        .with_context(|| format!("token id {token_id} is not in the tokenizer vocabulary"))
}

async fn send_completion_logprob_chunks(
    tx: &mpsc::Sender<Result<Event, Infallible>>,
    response: (&str, u64, &str),
    result: &GenerateResult,
    tokenizer: &Tokenizer,
    requested_top_logprobs: usize,
    stop_sequences: &[String],
) -> anyhow::Result<()> {
    let (id, created, model) = response;
    let logprobs = completion_logprobs(result, tokenizer, Some(requested_top_logprobs))?
        .context("requested completion logprobs were not built")?;
    let stream_text = result
        .token_ids
        .iter()
        .map(|&token_id| tokenizer.decode(&[token_id]).map_err(anyhow::Error::from))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let visible_text = truncate_tokens_at_stop(&stream_text, stop_sequences);
    for (index, text) in visible_text.into_iter().enumerate() {
        if text.is_empty() {
            continue;
        }
        send_completion_stream_chunk(
            tx,
            completion_chunk(
                id,
                created,
                model,
                text,
                Some(CompletionLogprobs {
                    tokens: vec![logprobs.tokens[index].clone()],
                    token_logprobs: vec![logprobs.token_logprobs[index]],
                    top_logprobs: vec![logprobs.top_logprobs[index].clone()],
                    text_offset: vec![logprobs.text_offset[index]],
                }),
            ),
        )
        .await?;
    }
    Ok(())
}

async fn send_chat_logprob_chunks(
    tx: &mpsc::Sender<Result<Event, Infallible>>,
    response: (&str, u64, &str),
    result: &GenerateResult,
    tokenizer: &Tokenizer,
    requested_top_logprobs: usize,
    stop_sequences: &[String],
) -> anyhow::Result<()> {
    let (id, created, model) = response;
    let logprobs = chat_logprobs(result, tokenizer, Some(requested_top_logprobs))?
        .context("requested chat logprobs were not built")?;
    let stream_text = result
        .token_ids
        .iter()
        .map(|&token_id| tokenizer.decode(&[token_id]).map_err(anyhow::Error::from))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let visible_text = truncate_tokens_at_stop(&stream_text, stop_sequences);
    for (index, content) in visible_text.into_iter().enumerate() {
        if content.is_empty() {
            continue;
        }
        send_stream_chunk(
            tx,
            content_chunk(
                id,
                created,
                model,
                content,
                Some(ChatLogprobs {
                    content: vec![logprobs.content[index].clone()],
                }),
            ),
        )
        .await?;
    }
    Ok(())
}

fn truncate_tokens_at_stop(tokens: &[String], stop_sequences: &[String]) -> Vec<String> {
    let text = tokens.concat();
    let cutoff = stop_sequences
        .iter()
        .filter(|stop| !stop.is_empty())
        .filter_map(|stop| text.find(stop))
        .min()
        .unwrap_or(text.len());
    let mut cursor = 0;
    let mut visible = Vec::new();
    for token in tokens {
        if cursor >= cutoff {
            break;
        }
        let mut end = (cutoff - cursor).min(token.len());
        while !token.is_char_boundary(end) {
            end -= 1;
        }
        visible.push(token[..end].to_string());
        cursor += token.len();
    }
    visible
}

fn finish_reason_label(reason: &FinishReason) -> &'static str {
    match reason {
        FinishReason::MaxTokens | FinishReason::Length => "length",
        FinishReason::EosToken | FinishReason::StopSequence { .. } => "stop",
    }
}

async fn send_tool_call_deltas(
    tx: &mpsc::Sender<Result<Event, Infallible>>,
    response: (&str, u64, &str),
    tool_calls: Vec<ChatMessageToolCall>,
) -> anyhow::Result<()> {
    let (id, created, model) = response;
    for chunk in tool_call_delta_chunks(id, created, model, tool_calls) {
        send_stream_chunk(tx, chunk).await?;
    }
    send_stream_chunk(tx, done_chunk(id, created, model, "tool_calls")).await
}

fn completion_id() -> String {
    format!("chatcmpl-{}", now_unix())
}

fn text_completion_id() -> String {
    format!("cmpl-{}", now_unix())
}

#[cfg(test)]
mod prompt_rendering_tests {
    use super::*;
    use serde_json::json;

    // A tool result reaches the template with the identity the caller sent, so
    // a tool-calling template can name the function it answers.
    #[test]
    fn tool_result_identity_reaches_the_template() {
        let message: ChatMessage = serde_json::from_value(json!({
            "role": "tool",
            "content": "42",
            "tool_call_id": "call_0",
            "name": "get_weather"
        }))
        .unwrap();

        let rendered = template_message(&message, None).unwrap();

        assert_eq!(rendered.name.as_deref(), Some("get_weather"));
        assert_eq!(rendered.tool_call_id.as_deref(), Some("call_0"));
    }

    // Chat templates index `function.arguments` as a mapping, so the JSON
    // string OpenAI puts on the wire is decoded before rendering.
    #[test]
    fn assistant_tool_call_arguments_render_as_a_mapping() {
        let message: ChatMessage = serde_json::from_value(json!({
            "role": "assistant",
            "tool_calls": [{
                "id": "call_0",
                "type": "function",
                "function": {"name": "bash", "arguments": "{\"command\":\"ls\"}"}
            }, {
                "id": "call_1",
                "type": "function",
                "function": {"name": "echo", "arguments": "not json"}
            }]
        }))
        .unwrap();

        let rendered = template_message(&message, None).unwrap();
        let calls = rendered.tool_calls.expect("tool calls");

        assert_eq!(calls[0]["function"]["arguments"]["command"], "ls");
        assert_eq!(calls[1]["function"]["arguments"], "not json");
    }

    // A reasoning model's template reads the caller's effort; without it the
    // template stays on its own default, which is often maximal.
    #[test]
    fn reasoning_effort_reaches_the_template() {
        let template = ChatTemplate::from_source(
            "{{ reasoning_strength if reasoning_strength is defined and reasoning_strength else 'default' }}",
        );
        let request = |extra: serde_json::Value| -> ChatCompletionRequest {
            let mut body = json!({ "model": "m", "messages": [{"role": "user", "content": "hi"}] });
            let object = body.as_object_mut().unwrap();
            for (key, value) in extra.as_object().unwrap() {
                object.insert(key.clone(), value.clone());
            }
            serde_json::from_value(body).unwrap()
        };

        assert_eq!(
            render_prompt(
                &request(json!({ "reasoning_effort": "low" })),
                Some(&template),
                None
            )
            .unwrap(),
            "low"
        );
        assert_eq!(
            render_prompt(&request(json!({})), Some(&template), None).unwrap(),
            "default"
        );
    }
}

#[cfg(test)]
mod sampling_resolution_tests {
    use super::*;
    use serde_json::json;

    fn declared(do_sample: Option<bool>, temperature: Option<f32>) -> GenerationDefaults {
        GenerationDefaults {
            do_sample,
            temperature,
            top_k: None,
            top_p: None,
            repetition_penalty: None,
            num_beams: None,
            num_return_sequences: None,
            min_length: None,
            max_length: None,
            length_penalty: None,
            no_repeat_ngram_size: None,
            diversity_penalty: None,
            early_stopping: None,
        }
    }

    fn declared_top_k(top_k: usize) -> GenerationDefaults {
        GenerationDefaults {
            top_k: Some(top_k),
            ..declared(Some(true), None)
        }
    }

    fn chat_request(extra: serde_json::Value) -> ChatCompletionRequest {
        let mut body = json!({ "model": "m", "messages": [{"role": "user", "content": "hi"}] });
        let object = body.as_object_mut().unwrap();
        for (key, value) in extra.as_object().unwrap() {
            object.insert(key.clone(), value.clone());
        }
        serde_json::from_value(body).unwrap()
    }

    // A silent chat request against a model that declares do_sample=true now
    // samples instead of being forced greedy — the server inherits the fix.
    #[test]
    fn silent_chat_request_honors_model_do_sample() {
        let request = chat_request(json!({}));
        let mut options = build_generate_options(&request);
        assert!(options.greedy, "the base chat default is greedy");
        options.resolve_sampling_defaults(
            Some(&declared(Some(true), Some(0.6))),
            &chat_sampling_overrides(&request),
        );
        assert!(!options.greedy, "model do_sample=true must disable greedy");
        // The OpenAI-compatible temperature default wins over the model's
        // declared temperature (the schema always supplies a value).
        assert_eq!(options.temperature, 1.0);
    }

    // An explicit sampling control keeps its meaning against a greedy model, and
    // temperature 0 still forces greedy regardless of the model.
    #[test]
    fn explicit_chat_controls_win_over_model() {
        let seeded = chat_request(json!({ "seed": 7 }));
        let mut options = build_generate_options(&seeded);
        options.resolve_sampling_defaults(
            Some(&declared(Some(false), None)),
            &chat_sampling_overrides(&seeded),
        );
        assert!(!options.greedy, "an explicit seed requests sampling");

        let cold = chat_request(json!({ "temperature": 0.0 }));
        let mut options = build_generate_options(&cold);
        options.resolve_sampling_defaults(
            Some(&declared(Some(true), Some(0.6))),
            &chat_sampling_overrides(&cold),
        );
        assert!(options.greedy, "temperature 0 forces greedy over the model");
    }

    // `top_k` is absent from the OpenAI schema, so a client that never mentions
    // it must not disable a model's declared top_k with the 0 sentinel.
    #[test]
    fn silent_chat_request_keeps_model_top_k() {
        let request = chat_request(json!({}));
        let mut options = build_generate_options(&request);
        options.resolve_sampling_defaults(
            Some(&declared_top_k(64)),
            &chat_sampling_overrides(&request),
        );
        assert_eq!(options.top_k, 64, "a silent request keeps declared top_k");

        let explicit = chat_request(json!({ "top_k": 5 }));
        let mut options = build_generate_options(&explicit);
        options.resolve_sampling_defaults(
            Some(&declared_top_k(64)),
            &chat_sampling_overrides(&explicit),
        );
        assert_eq!(options.top_k, 5, "an explicit top_k still wins");
    }

    // A silent completion request likewise adopts the model's declared regime.
    #[test]
    fn silent_completion_request_honors_model_do_sample() {
        let request: CompletionRequest =
            serde_json::from_value(json!({ "model": "m", "prompt": "hi" })).unwrap();
        let mut options = GenerateOptions {
            temperature: request.temperature,
            top_p: request.top_p,
            ..GenerateOptions::default()
        };
        assert!(options.greedy, "completions default to greedy");
        options.resolve_sampling_defaults(
            Some(&declared(Some(true), Some(0.6))),
            &completion_sampling_overrides(&request),
        );
        assert!(!options.greedy, "model do_sample=true must disable greedy");
    }
}

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
    let output_budget = request.output_budget(state.config.max_output_tokens);
    let mut prepared = prepare_generate_request(
        &request,
        &handle.tokenizer,
        client_session_id.is_some(),
        &PromptContext {
            chat_template: handle.chat_template.as_deref(),
            image_placeholder: placeholder.as_deref(),
            generation_defaults: handle.generation_defaults.as_ref(),
            default_reasoning_effort: state.config.default_reasoning_effort,
        },
        output_budget,
    )
    .map_err(|err| ApiError::internal(format!("prompt tokenization failed: {err}")))?;
    if !input_audio.is_empty() {
        prepared = prepare_audio_generate_request(&request, &handle.tokenizer, output_budget)?;
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
                reserved_output_tokens(&request, state.config.max_output_tokens),
            )
            .await?,
        )
    } else if let Some(audio) = input_audio.first() {
        Some(preprocess_chat_audio(audio, &handle)?)
    } else {
        None
    };
    let output_budget = admit_output_budget(
        &request,
        prepared.prompt_tokens,
        output_budget,
        handle.model_max_context,
    )?;
    let prompt_tokens = prepared.prompt_tokens;
    let mut generation_request = prepared.request;
    generation_request.options.max_new_tokens = output_budget;
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

    let (content, tool_calls, completion_tokens, finish_reason, logprobs, reasoning) = match result
    {
        Ok(result) => {
            let default_finish_reason = finish_reason_label(&result.finish_reason);
            let logprobs = chat_logprobs(&result, &tokenizer, requested_top_logprobs)
                .map_err(|err| ApiError::internal(format!("logprobs conversion failed: {err}")))?;
            let output = assistant_output_text(&result, &tokenizer);
            let reasoning = atem_reasoning_content(&output).filter(|thought| !thought.is_empty());
            let parsed = if tools_parseable_from_output(&request) {
                parse_assistant_output(output, default_finish_reason).aligned_to(&request)
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
                reasoning,
            )
        }
        Err(err)
            if wants_constrained_json
                && json_constraint_stopped_incomplete_message(&err.message) =>
        {
            (Some("{}".to_string()), None, 0, "stop", None, None)
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
                reasoning_content: reasoning,
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
    let output_budget = request.output_budget(state.config.max_output_tokens);
    let mut prepared = prepare_generate_request(
        &request,
        &handle.tokenizer,
        client_session_id.is_some(),
        &PromptContext {
            chat_template: handle.chat_template.as_deref(),
            image_placeholder: placeholder.as_deref(),
            generation_defaults: handle.generation_defaults.as_ref(),
            default_reasoning_effort: state.config.default_reasoning_effort,
        },
        output_budget,
    )
    .map_err(|err| ApiError::internal(format!("prompt tokenization failed: {err}")))?;
    if !input_audio.is_empty() {
        prepared = prepare_audio_generate_request(&request, &handle.tokenizer, output_budget)?;
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
                reserved_output_tokens(&request, state.config.max_output_tokens),
            )
            .await?,
        )
    } else if let Some(audio) = input_audio.first() {
        Some(preprocess_chat_audio(audio, &handle)?)
    } else {
        None
    };
    let output_budget = admit_output_budget(
        &request,
        prepared.prompt_tokens,
        output_budget,
        handle.model_max_context,
    )?;
    let wants_constrained_json = request.wants_constrained_json();
    let mut generation_request = prepared.request;
    generation_request.options.max_new_tokens = output_budget;
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
        let mut channel_gate = PrivateChannelGate::new(handle.private_channels);
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
                    // The special-token spelling is only read by an armed gate; a
                    // model with no private channel discards it, so skip the extra
                    // per-token decode on that common path.
                    let spelled = channel_gate
                        .armed()
                        .then(|| tokenizer.decode_with_special_tokens(&[token.token_id]).ok())
                        .flatten();
                    let revealed = channel_gate.push(spelled.as_deref(), &token.text);
                    if !revealed.reasoning.is_empty() {
                        send_stream_chunk(
                            &tx,
                            reasoning_chunk(&id, created, &model, revealed.reasoning),
                        )
                        .await?;
                    }
                    let content = stop_buffer.push(&revealed.content);
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
                    let mut tool_calls = if buffer_for_tool_detection {
                        parse_tool_calls(&assistant_output_text(&result, &tokenizer))
                    } else {
                        Vec::new()
                    };
                    align_tool_calls(&mut tool_calls, &request);
                    if tool_calls.is_empty() {
                        send_chat_logprob_chunks(
                            &tx,
                            (&id, created, &model),
                            &result,
                            &tokenizer,
                            requested_top_logprobs,
                            &user_stop_sequences,
                            handle.private_channels,
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
                    )
                    .aligned_to(&request);
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
                    let content = visible_assistant_text(&result, &tokenizer);
                    if !content.is_empty() {
                        send_stream_chunk(&tx, content_chunk(&id, created, &model, content, None))
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
                    // The gate's withheld text is ordinary content rather than a
                    // partial stop sequence, so it is drained either way; only the
                    // stop buffer's own pending text is dropped on a stop match.
                    let remaining = channel_gate.flush();
                    if !remaining.reasoning.is_empty() {
                        send_stream_chunk(
                            &tx,
                            reasoning_chunk(&id, created, &model, remaining.reasoning),
                        )
                        .await?;
                    }
                    let mut content = stop_buffer.push(&remaining.content);
                    if !matches!(result.finish_reason, FinishReason::StopSequence { .. }) {
                        content.push_str(&stop_buffer.flush());
                    }
                    if !content.is_empty() {
                        send_stream_chunk(&tx, content_chunk(&id, created, &model, content, None))
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
    output_budget: usize,
) -> Result<PreparedGenerateRequest, ApiError> {
    let token_ids = audio_decoder_prompt(tokenizer, None)?;
    let prompt_tokens = token_ids.len();
    Ok(PreparedGenerateRequest {
        request: GenerateRequest {
            prompt: GeneratePrompt::TokenIds(token_ids),
            options: build_generate_options_with_tokenizer(request, tokenizer, output_budget),
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
    if let Some((field, requested)) = request.requested_output_budget() {
        if requested == 0 {
            return Err(ApiError::bad_request(format!(
                "{field} must be greater than zero"
            )));
        }
        if requested > config.max_output_tokens {
            return Err(ApiError::bad_request(format!(
                "{field} must be less than or equal to the server cap of {}",
                config.max_output_tokens
            )));
        }
    }
    validate_sampling_range(request.temperature, "temperature")?;
    validate_sampling_range(request.top_p, "top_p")?;
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
    validate_sampling_range(request.temperature, "temperature")?;
    validate_sampling_range(request.top_p, "top_p")?;
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

/// The response budget this request may actually decode with, once the final
/// prompt length is known.
///
/// A caller that named a budget is told when it does not fit, because quietly
/// returning less than it asked for answers a question it did not ask. A caller
/// that named none asked for whatever fits, so its budget shrinks to the
/// context the prompt left behind rather than turning into a rejection — the
/// server's cap is a ceiling on generosity, not a reservation the caller made.
fn admit_output_budget(
    request: &ChatCompletionRequest,
    prompt_tokens: usize,
    budget: usize,
    model_max_context: Option<usize>,
) -> Result<usize, ApiError> {
    if request.requested_output_budget().is_some() {
        enforce_context_cap(prompt_tokens, budget, model_max_context)?;
        return Ok(budget);
    }
    let Some(model_max_context) = model_max_context else {
        return Ok(budget);
    };
    let remaining = model_max_context
        .checked_sub(prompt_tokens)
        .filter(|remaining| *remaining > 0)
        .ok_or_else(|| {
            ApiError::bad_request(format!(
                "What: request admission exceeded the model context limit. \
                 Why: final prefill length ({prompt_tokens}) after placeholder expansion leaves no room to decode within {model_max_context}. \
                 How: shorten the prompt or reduce the image count/expansion size."
            ))
        })?;
    Ok(budget.min(remaining))
}

/// Tokens the caller explicitly reserved for its response, which is what image
/// placeholder expansion must leave free. An unnamed budget reserves nothing,
/// because it yields to whatever the prompt turns out to need.
fn reserved_output_tokens(request: &ChatCompletionRequest, cap: usize) -> usize {
    request
        .requested_output_budget()
        .map_or(0, |_| request.output_budget(cap))
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

/// Builds a generation request without a server in hand, so an unspecified
/// output budget falls back to the default cap a server would have applied.
pub fn build_generate_request(request: &ChatCompletionRequest) -> GenerateRequest {
    GenerateRequest {
        prompt: GeneratePrompt::Text(build_prompt(request)),
        options: build_generate_options(request, request.output_budget(DEFAULT_MAX_OUTPUT_TOKENS)),
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
        temperature: request.temperature.unwrap_or(NEUTRAL_SAMPLING),
        top_p: request.top_p.unwrap_or(NEUTRAL_SAMPLING),
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
fn completion_sampling_overrides(request: &CompletionRequest) -> SamplingOverrides {
    let greedy = if request.temperature == Some(0.0) {
        Some(true)
    } else if request.min_p > 0.0 {
        Some(false)
    } else {
        None
    };
    SamplingOverrides {
        greedy,
        temperature: request.temperature,
        top_p: request.top_p,
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

/// What the loaded model and the server's configuration contribute to a prompt,
/// as opposed to what the request itself carries.
///
/// These travel together to every prompt that is built, and a request that
/// dropped one of them would render against the wrong defaults rather than
/// fail, so they are passed as one value instead of four positional arguments.
struct PromptContext<'a> {
    chat_template: Option<&'a ChatTemplate>,
    image_placeholder: Option<&'a str>,
    generation_defaults: Option<&'a GenerationDefaults>,
    /// Applied when the request omits `reasoning_effort`.
    default_reasoning_effort: Option<ReasoningEffort>,
}

fn prepare_generate_request(
    request: &ChatCompletionRequest,
    tokenizer: &Tokenizer,
    session: bool,
    context: &PromptContext<'_>,
    output_budget: usize,
) -> anyhow::Result<PreparedGenerateRequest> {
    let prompt = if session && !request.has_tool_context() {
        build_session_prompt(&request.messages, context.image_placeholder)
    } else {
        render_prompt(request, context)?
    };
    let token_ids = tokenizer
        .encode(&prompt)
        .map_err(|e| anyhow::anyhow!("Failed to tokenize prompt: {e}"))?;
    let prompt_tokens = token_ids.len();
    let mut options = build_generate_options_with_tokenizer(request, tokenizer, output_budget);
    // Honor the model's declared sampling regime (e.g. a reasoning model that
    // ships do_sample=true); explicit request fields still win.
    options.resolve_sampling_defaults(
        context.generation_defaults,
        &chat_sampling_overrides(request),
    );
    Ok(PreparedGenerateRequest {
        request: GenerateRequest {
            prompt: GeneratePrompt::TokenIds(token_ids),
            options,
        },
        prompt_tokens,
    })
}

fn build_generate_options(
    request: &ChatCompletionRequest,
    output_budget: usize,
) -> GenerateOptions {
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
        max_new_tokens: output_budget,
        temperature: request.temperature.unwrap_or(NEUTRAL_SAMPLING),
        top_p: request.top_p.unwrap_or(NEUTRAL_SAMPLING),
        top_k: request.top_k,
        min_p: request.min_p,
        top_a: request.top_a,
        typical_p: request.typical_p,
        repetition_penalty: request.repetition_penalty,
        frequency_penalty: request.frequency_penalty,
        presence_penalty: request.presence_penalty,
        greedy: request.temperature == Some(0.0) || !stochastic_sampling_requested,
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
    let greedy = if request.temperature == Some(0.0) {
        Some(true)
    } else if requests_sampling {
        Some(false)
    } else {
        None
    };
    SamplingOverrides {
        greedy,
        // An absent field is a caller with no opinion, not a caller choosing
        // the schema's documented default, so a package that declares its own
        // temperature or top-p keeps it. An agent client typically sends
        // neither; overriding both with 1.0 widened every such model to its
        // full distribution, which is how a long agent turn degenerates.
        temperature: request.temperature,
        top_p: request.top_p,
        // `top_k` is an extension the OpenAI schema never carries, and 0 is its
        // "disabled" sentinel rather than a caller's choice. Treat an absent
        // `top_k` as unspecified so a package that declares one keeps it,
        // instead of every OpenAI client silently widening the model to the
        // full vocabulary.
        top_k: (request.top_k > 0).then_some(request.top_k),
    }
}

/// Neutral value for a sampling control the caller left unset, used only where
/// options are built without resolving them against the package's declaration.
const NEUTRAL_SAMPLING: f32 = 1.0;

/// Reject a sampling control the caller actually sent; an absent one is the
/// package's business rather than a value to judge.
fn validate_sampling_range(value: Option<f32>, field: &str) -> Result<(), ApiError> {
    match value {
        Some(value) if !value.is_finite() || value < 0.0 => Err(ApiError::bad_request(format!(
            "{field} must be finite and non-negative"
        ))),
        _ => Ok(()),
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
    output_budget: usize,
) -> GenerateOptions {
    let mut options = build_generate_options(request, output_budget);
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
    context: &PromptContext<'_>,
) -> anyhow::Result<String> {
    if let Some(chat_template) = context.chat_template {
        let messages = request
            .messages
            .iter()
            .map(|message| template_message(message, context.image_placeholder))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let tools_json = tools_offered_to_model(request)
            .map(serde_json::to_string)
            .transpose()?;
        return chat_template
            .render_with_reasoning_effort(
                &messages,
                tools_json.as_deref(),
                true,
                request
                    .reasoning_effort
                    .or(context.default_reasoning_effort)
                    .map(ReasoningEffort::as_str),
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
            // A parameter that opens another parameter before it closes was
            // opened twice: a value cannot contain a parameter, so the outer
            // tag is a stray and the inner one is the real parameter. Taking
            // the outer at its word would swallow the stray tag into the value
            // and lose the argument the model meant to pass.
            if let Some(nested) = parameters.find(PARAMETER)
                && nested < parameter_end
            {
                parameters = &parameters[nested..];
                continue;
            }
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

impl ParsedAssistantOutput {
    /// Point every parsed call at a tool the caller actually offered.
    fn aligned_to(mut self, request: &ChatCompletionRequest) -> Self {
        if let Some(calls) = self.tool_calls.as_deref_mut() {
            align_tool_calls(calls, request);
        }
        self
    }
}

/// Resolve a namespaced call back to the tool the caller offered.
///
/// This package's own tool instructions carry a namespaced example
/// (`example_tool_name.example_function_name`), so a model offered a bare name
/// sometimes answers with a qualified one. The namespace is the model's
/// spelling, not the caller's, and a client can only dispatch a name it
/// offered — it rejects anything else as an unavailable tool. So when the
/// qualified name is not on offer but its final segment is, the call names that
/// tool and is resolved to it. Anything else is left untouched for the caller
/// to reject, because inventing a target it never offered would be worse than
/// the error it already knows how to report.
fn align_tool_calls(calls: &mut [ChatMessageToolCall], request: &ChatCompletionRequest) {
    let Some(offered) = tools_offered_to_model(request) else {
        return;
    };
    let names_a_tool =
        |name: &str| -> bool { offered.iter().any(|tool| tool.function.name == name) };
    for call in calls {
        if names_a_tool(&call.function.name) {
            continue;
        }
        let Some((_, suffix)) = call.function.name.rsplit_once('.') else {
            continue;
        };
        if names_a_tool(suffix) {
            call.function.name = suffix.to_string();
        }
    }
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

/// Opens an ATEM channel, ending the address that names its recipient.
const ATEM_MESSAGE: &str = "<|message|>";
/// Begins the address that names an ATEM channel's recipient.
const ATEM_ADDRESS: &str = "to=";
/// The ATEM channel addressed to the caller, i.e. the visible answer.
const ATEM_USER_CHANNEL: &str = "to=user<|message|>";

/// The part of an ATEM turn a client may see, or `None` for output that carries
/// no ATEM channel at all.
///
/// A turn is a sequence of channels, each addressed to a recipient: the model
/// itself for private reasoning, a tool for a call, or the user. Only the one
/// addressed to the user is an answer, so a turn that ends before reaching it —
/// truncated by the token budget, say — produced no answer, and yields empty
/// content rather than leaking the model's reasoning as if it were one.
fn atem_visible_content(output: &str) -> Option<String> {
    if let Some((_, answer)) = output.rsplit_once(ATEM_USER_CHANNEL) {
        let end = ["<|eot|>", "<|eom|>"]
            .into_iter()
            .filter_map(|marker| answer.find(marker))
            .min()
            .unwrap_or(answer.len());
        return Some(answer[..end].to_string());
    }
    atem_addresses_a_channel(output).then(String::new)
}

/// The ATEM channel the model addresses to itself, i.e. its private thinking.
const ATEM_SELF_CHANNEL: &str = "to=self<|message|>";

/// What an ATEM turn thought before answering, or `None` for output that
/// carries no private channel.
///
/// A turn may think more than once — between tool calls, say — and the whole of
/// it is one train of thought, so the segments are joined. This is not part of
/// the answer and is never merged into it; it is offered separately so a client
/// can show that the model is working rather than watching silence.
fn atem_reasoning_content(output: &str) -> Option<String> {
    let mut thought = String::new();
    let mut addressed = false;
    for (at, marker) in output.match_indices(ATEM_SELF_CHANNEL) {
        addressed = true;
        let rest = &output[at + marker.len()..];
        let end = ["<|eot|>", "<|eom|>"]
            .into_iter()
            .filter_map(|marker| rest.find(marker))
            .min()
            .unwrap_or(rest.len());
        thought.push_str(&rest[..end]);
    }
    addressed.then_some(thought)
}

/// Whether the output has opened a channel addressed to someone.
///
/// The recipient is matched by shape rather than by name because a model may
/// address any tool it was offered, not only itself and the user, and a channel
/// addressed to a tool is no more visible than one addressed to the model.
fn atem_addresses_a_channel(output: &str) -> bool {
    output.match_indices(ATEM_ADDRESS).any(|(at, _)| {
        output[at..]
            .split_once(ATEM_MESSAGE)
            .is_some_and(|(address, _)| !address.contains('<'))
    })
}

/// Streams only the part of a channelled turn a client may see.
///
/// A streamed token's text has its special tokens stripped, so the channel
/// markers that decide what is visible are invisible token by token and a
/// streaming turn would otherwise send private reasoning as it is produced. The
/// gate keeps the turn's special-token spelling and asks [`atem_visible_content`]
/// what is visible so far, emitting only what has grown since the previous
/// token, so channel semantics stay in one place and streaming agrees with the
/// buffered path by construction.
///
/// The gate is armed from the package's own declaration that it has a private
/// channel, not from guessing at the shape of a turn, so a model that declares
/// none is never gated and streams exactly what it generated. An armed gate
/// then fails closed: nothing it cannot place in the channel addressed to the
/// caller is served as an answer, because the cost of withholding an answer is
/// a retry and the cost of releasing reasoning as one is a disclosure that
/// cannot be taken back. Private thinking is not discarded, though — it is
/// reported separately, so a client can show the model working instead of
/// waiting on silence.
#[derive(Debug)]
struct PrivateChannelGate {
    /// The turn so far, spelled with the special tokens that mark channels.
    transcript: String,
    /// How much of the answer has already been streamed.
    content: ChannelStream,
    /// How much of the thinking has already been streamed.
    reasoning: ChannelStream,
    /// Whether the model declares a channel the caller must not be shown.
    armed: bool,
}

/// What one token added to each channel.
#[derive(Debug, Default, PartialEq, Eq)]
struct ChannelDelta {
    /// Text belonging to the answer.
    content: String,
    /// Text belonging to the model's private thinking.
    reasoning: String,
}

/// Tracks how much of one channel has been sent, so each emission carries only
/// what has appeared since the last one.
#[derive(Debug, Default)]
struct ChannelStream {
    sent: String,
}

impl ChannelStream {
    /// What `revealed` adds to what was already sent.
    ///
    /// A channel that is reopened replaces its text rather than extending it,
    /// so text that is no longer a continuation is sent whole.
    fn growth(&mut self, revealed: String) -> String {
        let grown = match revealed.strip_prefix(&self.sent) {
            Some(grown) => grown.to_string(),
            None => revealed.clone(),
        };
        self.sent = revealed;
        grown
    }
}

impl PrivateChannelGate {
    fn new(armed: bool) -> Self {
        Self {
            transcript: String::new(),
            content: ChannelStream::default(),
            reasoning: ChannelStream::default(),
            armed,
        }
    }

    /// Whether the gate withholds private channels, in which case the caller
    /// must supply each token's special-token spelling for it to read.
    fn armed(&self) -> bool {
        self.armed
    }

    /// The visible text this token added, which is empty while the token
    /// belongs to a private channel.
    ///
    /// `spelled` is the token with its special tokens intact and `plain` is the
    /// same token as a client would see it. A token that cannot be spelled
    /// leaves the gate unable to place it in a channel, so an armed gate
    /// withholds it.
    fn push(&mut self, spelled: Option<&str>, plain: &str) -> ChannelDelta {
        if !self.armed {
            return ChannelDelta {
                content: plain.to_string(),
                reasoning: String::new(),
            };
        }
        let Some(spelled) = spelled else {
            return ChannelDelta::default();
        };
        self.transcript.push_str(spelled);
        self.growth()
    }

    /// Whatever the gate is still holding once the turn ends.
    ///
    /// A turn that never reached the caller's channel produced no answer, so
    /// nothing is released as one: the model spent the turn thinking, and
    /// saying so is the honest empty completion a `length` finish reason
    /// already reports. The thinking itself is still returned, in the channel
    /// that is for thinking.
    fn flush(&mut self) -> ChannelDelta {
        if !self.armed {
            return ChannelDelta::default();
        }
        self.growth()
    }

    /// What each channel has revealed since the last emission.
    fn growth(&mut self) -> ChannelDelta {
        ChannelDelta {
            content: self
                .content
                .growth(atem_visible_content(&self.transcript).unwrap_or_default()),
            reasoning: self
                .reasoning
                .growth(atem_reasoning_content(&self.transcript).unwrap_or_default()),
        }
    }
}

/// The part of a finished turn a caller may see.
///
/// The buffered paths all reduce to this, so no path can disagree with the
/// streaming gate about which channel carried the answer.
fn visible_assistant_text(result: &GenerateResult, tokenizer: &Tokenizer) -> String {
    let output = assistant_output_text(result, tokenizer);
    atem_visible_content(&output).unwrap_or(output)
}

fn assistant_output_text(result: &GenerateResult, tokenizer: &Tokenizer) -> String {
    let Ok(with_special_tokens) = tokenizer.decode_with_special_tokens(&result.token_ids) else {
        return result.text.clone();
    };
    if atem_addresses_a_channel(&with_special_tokens)
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
    private_channels: bool,
) -> anyhow::Result<()> {
    let (id, created, model) = response;
    let logprobs = chat_logprobs(result, tokenizer, Some(requested_top_logprobs))?
        .context("requested chat logprobs were not built")?;
    // Logprobs are per token, so the visible text is gated per token too: a
    // token inside a private channel contributes an empty string and keeps its
    // logprob entry aligned by index, rather than being dropped here.
    let mut gate = PrivateChannelGate::new(private_channels);
    let stream_text = result
        .token_ids
        .iter()
        .map(|&token_id| {
            let spelled = tokenizer.decode_with_special_tokens(&[token_id]).ok();
            let plain = tokenizer.decode(&[token_id])?;
            Ok(gate.push(spelled.as_deref(), &plain).content)
        })
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
    /// A prompt context carrying only a template and an optional server-side
    /// effort default, which is all these cases vary.
    fn context<'a>(
        template: &'a ChatTemplate,
        default_reasoning_effort: Option<ReasoningEffort>,
    ) -> PromptContext<'a> {
        PromptContext {
            chat_template: Some(template),
            image_placeholder: None,
            generation_defaults: None,
            default_reasoning_effort,
        }
    }

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
                &context(&template, None),
            )
            .unwrap(),
            "low"
        );
        assert_eq!(
            render_prompt(&request(json!({})), &context(&template, None)).unwrap(),
            "default"
        );
    }

    // An agent client that never sends `reasoning_effort` would otherwise leave
    // the model on its own default and can burn a whole token budget thinking
    // without emitting an answer, so the operator can supply a floor. A client
    // that does ask still wins.
    #[test]
    fn server_default_reasoning_effort_fills_in_only_when_the_request_is_silent() {
        let template = ChatTemplate::from_source(
            "{{ reasoning_strength if reasoning_strength is defined and reasoning_strength else 'model-default' }}",
        );
        let request = |extra: serde_json::Value| -> ChatCompletionRequest {
            let mut body = json!({ "model": "m", "messages": [{"role": "user", "content": "hi"}] });
            let object = body.as_object_mut().unwrap();
            for (key, value) in extra.as_object().unwrap() {
                object.insert(key.clone(), value.clone());
            }
            serde_json::from_value(body).unwrap()
        };

        // Silent request takes the server default.
        assert_eq!(
            render_prompt(
                &request(json!({})),
                &context(&template, Some(ReasoningEffort::Low)),
            )
            .unwrap(),
            "low"
        );
        // An explicit request value outranks the server default.
        assert_eq!(
            render_prompt(
                &request(json!({ "reasoning_effort": "high" })),
                &context(&template, Some(ReasoningEffort::Low)),
            )
            .unwrap(),
            "high"
        );
        // No server default configured leaves the model on its own default.
        assert_eq!(
            render_prompt(&request(json!({})), &context(&template, None)).unwrap(),
            "model-default"
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
        let mut options = build_generate_options(&request, DEFAULT_MAX_OUTPUT_TOKENS);
        assert!(options.greedy, "the base chat default is greedy");
        options.resolve_sampling_defaults(
            Some(&declared(Some(true), Some(0.6))),
            &chat_sampling_overrides(&request),
        );
        assert!(!options.greedy, "model do_sample=true must disable greedy");
        assert_eq!(
            options.temperature, 0.6,
            "a silent request keeps the declared temperature"
        );
    }

    // An explicit sampling control keeps its meaning against a greedy model, and
    // temperature 0 still forces greedy regardless of the model.
    #[test]
    fn explicit_chat_controls_win_over_model() {
        let seeded = chat_request(json!({ "seed": 7 }));
        let mut options = build_generate_options(&seeded, DEFAULT_MAX_OUTPUT_TOKENS);
        options.resolve_sampling_defaults(
            Some(&declared(Some(false), None)),
            &chat_sampling_overrides(&seeded),
        );
        assert!(!options.greedy, "an explicit seed requests sampling");

        let cold = chat_request(json!({ "temperature": 0.0 }));
        let mut options = build_generate_options(&cold, DEFAULT_MAX_OUTPUT_TOKENS);
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
        let mut options = build_generate_options(&request, DEFAULT_MAX_OUTPUT_TOKENS);
        options.resolve_sampling_defaults(
            Some(&declared_top_k(64)),
            &chat_sampling_overrides(&request),
        );
        assert_eq!(options.top_k, 64, "a silent request keeps declared top_k");

        let explicit = chat_request(json!({ "top_k": 5 }));
        let mut options = build_generate_options(&explicit, DEFAULT_MAX_OUTPUT_TOKENS);
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
            temperature: request.temperature.unwrap_or(NEUTRAL_SAMPLING),
            top_p: request.top_p.unwrap_or(NEUTRAL_SAMPLING),
            ..GenerateOptions::default()
        };
        assert!(options.greedy, "completions default to greedy");
        options.resolve_sampling_defaults(
            Some(&declared(Some(true), Some(0.6))),
            &completion_sampling_overrides(&request),
        );
        assert!(!options.greedy, "model do_sample=true must disable greedy");
        assert_eq!(
            options.temperature, 0.6,
            "a silent request keeps the declared temperature"
        );
    }

    // An agent client sends neither field, and the schema's documented 1.0
    // defaults used to be forwarded as the caller's choice, silently widening
    // every such model to its full distribution.
    #[test]
    fn a_silent_request_keeps_the_declared_temperature_and_top_p() {
        let silent = chat_sampling_overrides(&chat_request(json!({})));
        assert_eq!(silent.temperature, None);
        assert_eq!(silent.top_p, None);

        let explicit = chat_sampling_overrides(&chat_request(json!({
            "temperature": 0.2,
            "top_p": 0.5
        })));
        assert_eq!(explicit.temperature, Some(0.2));
        assert_eq!(explicit.top_p, Some(0.5));
    }

    // Zero temperature is still how OpenAI clients ask for determinism, and it
    // has to keep winning over a package that declares sampling.
    #[test]
    fn an_explicit_zero_temperature_still_forces_greedy() {
        let request = chat_request(json!({ "temperature": 0.0 }));
        let mut options = build_generate_options(&request, DEFAULT_MAX_OUTPUT_TOKENS);
        options.resolve_sampling_defaults(
            Some(&declared(Some(true), Some(0.6))),
            &chat_sampling_overrides(&request),
        );

        assert!(options.greedy, "an explicit zero temperature is greedy");
        assert_eq!(options.temperature, 0.0);
    }

    // The subtle case the Option distinction exists for: a caller that sends the
    // schema's own default value (1.0) has still made an explicit choice, so it
    // must override a package that declares a different temperature — absent is
    // "no opinion", a sent 1.0 is an opinion that happens to equal the default.
    #[test]
    fn an_explicit_default_valued_temperature_still_overrides_the_package() {
        let request = chat_request(json!({ "temperature": 1.0 }));
        assert_eq!(
            chat_sampling_overrides(&request).temperature,
            Some(1.0),
            "an explicitly sent 1.0 is a choice, not absence"
        );
        let mut options = build_generate_options(&request, DEFAULT_MAX_OUTPUT_TOKENS);
        options.resolve_sampling_defaults(
            Some(&declared(Some(true), Some(0.6))),
            &chat_sampling_overrides(&request),
        );
        assert_eq!(
            options.temperature, 1.0,
            "a caller's explicit 1.0 wins over the package's declared 0.6"
        );
    }
}

#[cfg(test)]
mod output_budget_tests {
    use super::*;
    use serde_json::json;

    fn request(budget: serde_json::Value) -> ChatCompletionRequest {
        let mut body = json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
        });
        for (field, value) in budget.as_object().expect("budget fields") {
            body[field] = value.clone();
        }
        serde_json::from_value(body).expect("request")
    }

    // A reasoning model spends the same budget on thinking as on its answer, so
    // a client that names no budget must get the server's cap rather than a
    // small fixed default that truncates the turn mid-thought.
    #[test]
    fn an_unspecified_budget_is_the_server_cap() {
        assert_eq!(request(json!({})).output_budget(4096), 4096);
        assert_eq!(request(json!({})).requested_output_budget(), None);
    }

    // OpenAI deprecated max_tokens for chat and accepts only
    // max_completion_tokens for reasoning models, so the newer field wins.
    #[test]
    fn max_completion_tokens_supersedes_max_tokens() {
        let both = request(json!({"max_tokens": 16, "max_completion_tokens": 2048}));
        assert_eq!(both.output_budget(4096), 2048);
        assert_eq!(
            both.requested_output_budget(),
            Some(("max_completion_tokens", 2048))
        );
        assert_eq!(request(json!({"max_tokens": 16})).output_budget(4096), 16);
    }

    // The server cap bounds whatever the client asked for.
    #[test]
    fn the_server_cap_bounds_a_requested_budget() {
        assert_eq!(
            request(json!({"max_completion_tokens": 99_999})).output_budget(4096),
            4096
        );
    }

    // A rejection names the field the client actually sent, so the caller can
    // find it in its own request.
    #[test]
    fn a_rejected_budget_names_the_field_the_client_sent() {
        let config = ServerConfig {
            max_output_tokens: 128,
            ..ServerConfig::default()
        };
        let error = validate_request(&request(json!({"max_completion_tokens": 4096})), &config)
            .expect_err("over cap");
        assert!(
            error.message.contains("max_completion_tokens"),
            "{}",
            error.message
        );
        let error =
            validate_request(&request(json!({"max_tokens": 0})), &config).expect_err("zero");
        assert!(error.message.contains("max_tokens"), "{}", error.message);
        validate_request(&request(json!({})), &config).expect("an unspecified budget is valid");
    }

    // The cap-sized fallback is a ceiling, not a reservation: a short-context
    // model must still answer a request that named no budget, instead of
    // rejecting every one of them because cap plus prompt exceeds its context.
    #[test]
    fn an_unspecified_budget_yields_to_the_prompt() {
        let budget = admit_output_budget(&request(json!({})), 3000, 4096, Some(4096))
            .expect("an unspecified budget never rejects a prompt that fits");

        assert_eq!(budget, 1096);
    }

    // A caller that named a budget asked a question, and returning less than it
    // asked for would answer a different one, so it is told instead.
    #[test]
    fn a_requested_budget_that_does_not_fit_is_rejected() {
        let error = admit_output_budget(
            &request(json!({"max_completion_tokens": 2048})),
            3000,
            2048,
            Some(4096),
        )
        .expect_err("2048 requested tokens do not fit after a 3000 token prompt");

        assert!(error.message.contains("2048"), "{}", error.message);
    }

    // A prompt that fills the context leaves nothing to decode, which is a real
    // rejection rather than a budget of zero that would return an empty turn.
    #[test]
    fn a_prompt_that_fills_the_context_is_rejected() {
        let error = admit_output_budget(&request(json!({})), 4096, 4096, Some(4096))
            .expect_err("a full context leaves no room to decode");

        assert!(error.message.contains("4096"), "{}", error.message);
    }

    // Image expansion has to leave room for the response the caller reserved,
    // and an unnamed budget reserves nothing because it yields to the prompt.
    #[test]
    fn only_a_named_budget_reserves_room_for_expansion() {
        assert_eq!(reserved_output_tokens(&request(json!({})), 4096), 0);
        assert_eq!(
            reserved_output_tokens(&request(json!({"max_tokens": 512})), 4096),
            512
        );
    }
}

#[cfg(test)]
mod atem_tool_call_parsing_tests {
    use super::*;

    fn arguments(output: &str) -> serde_json::Value {
        let calls = parse_tool_calls(output);
        assert_eq!(calls.len(), 1, "expected one call from {output:?}");
        serde_json::from_str(&calls[0].function.arguments).expect("arguments")
    }

    // Observed from OpenCode: the model opened the same parameter twice, and
    // taking the outer tag at its word swallowed the stray tag into the value,
    // so the call reached the client with the argument it needed missing.
    #[test]
    fn a_parameter_opened_twice_keeps_the_inner_one() {
        assert_eq!(
            arguments(concat!(
                "<atem:invoke name=\"bash\">\n",
                "<atem:parameter name=\"command\">\n",
                "<atem:parameter name=\"command\">ls -la</atem:parameter>\n",
                "</atem:invoke>",
            )),
            serde_json::json!({"command": "ls -la"})
        );
    }

    // The repair is confined to the parameter that was opened twice; the ones
    // around it are read exactly as they were written.
    #[test]
    fn a_stray_open_tag_does_not_disturb_its_neighbours() {
        assert_eq!(
            arguments(concat!(
                "<atem:invoke name=\"edit\">\n",
                "<atem:parameter name=\"path\">a.py</atem:parameter>\n",
                "<atem:parameter name=\"text\">",
                "<atem:parameter name=\"text\">hi</atem:parameter>\n",
                "<atem:parameter name=\"count\">2</atem:parameter>\n",
                "</atem:invoke>",
            )),
            serde_json::json!({"path": "a.py", "text": "hi", "count": 2})
        );
    }

    // A well-formed call is unaffected, including a value that legitimately
    // carries angle brackets.
    #[test]
    fn a_well_formed_call_is_read_as_written() {
        assert_eq!(
            arguments(concat!(
                "<atem:invoke name=\"bash\">\n",
                "<atem:parameter name=\"command\">echo \"<b>hi</b>\"</atem:parameter>\n",
                "</atem:invoke>",
            )),
            serde_json::json!({"command": "echo \"<b>hi</b>\""})
        );
    }
}

#[cfg(test)]
mod tool_name_alignment_tests {
    use super::*;
    use serde_json::json;

    fn request_offering(names: &[&str]) -> ChatCompletionRequest {
        let tools: Vec<_> = names
            .iter()
            .map(|name| json!({"type": "function", "function": {"name": name}}))
            .collect();
        serde_json::from_value(json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": tools
        }))
        .expect("request")
    }

    fn call(name: &str) -> ChatMessageToolCall {
        serde_json::from_value(json!({
            "id": "call_0",
            "type": "function",
            "function": {"name": name, "arguments": "{}"}
        }))
        .expect("call")
    }

    fn aligned(name: &str, offered: &[&str]) -> String {
        let mut calls = vec![call(name)];
        align_tool_calls(&mut calls, &request_offering(offered));
        calls.remove(0).function.name
    }

    // The package's tool instructions show a namespaced example, so the model
    // sometimes qualifies a bare name. The client only knows the name it
    // offered, and rejects anything else as an unavailable tool.
    #[test]
    fn a_namespaced_call_resolves_to_the_offered_tool() {
        assert_eq!(aligned("glob.glob", &["glob", "read"]), "glob");
        assert_eq!(aligned("functions.read", &["glob", "read"]), "read");
    }

    // A name the caller offered is never rewritten, even when it contains a dot.
    #[test]
    fn an_offered_name_is_left_alone() {
        assert_eq!(aligned("glob", &["glob"]), "glob");
        assert_eq!(aligned("fs.read", &["fs.read", "read"]), "fs.read");
    }

    // A suffix that names nothing on offer stays as the model spelled it, so
    // the caller reports the unavailable tool it already knows how to report
    // rather than dispatching one it never offered.
    #[test]
    fn an_unknown_call_is_left_for_the_caller_to_reject() {
        assert_eq!(aligned("shell.exec", &["glob", "read"]), "shell.exec");
        assert_eq!(aligned("wander", &["glob", "read"]), "wander");
    }

    // Suffix resolution is safe only because it matches a call's final segment
    // against an offered tool's *whole* name, so the resolved target is always a
    // single offered name and never a choice between two. When two offered tools
    // merely share a final segment (`fs.read`, `net.read`) and the model writes a
    // third namespace (`svc.read`), the bare segment `read` is offered by
    // neither, so the call is left alone for the caller to reject rather than
    // being resolved arbitrarily to one of them.
    #[test]
    fn a_shared_final_segment_is_not_resolved_arbitrarily() {
        assert_eq!(aligned("svc.read", &["fs.read", "net.read"]), "svc.read");
        // An exact offer still wins and is never rewritten to the other.
        assert_eq!(aligned("net.read", &["fs.read", "net.read"]), "net.read");
    }

    // The whole path a real turn travels: an ATEM tool call the model spelled
    // with a namespace it was never offered is parsed and then resolved to the
    // offered tool, so the client dispatches it instead of rejecting `glob.glob`.
    #[test]
    fn a_namespaced_atem_call_dispatches_through_the_full_parse_path() {
        let output = "to=functions.glob<|message|>\
             <atem:invoke name=\"glob.glob\">\
             <atem:parameter name=\"pattern\">*.rs</atem:parameter>\
             </atem:invoke>"
            .to_string();
        let parsed =
            parse_assistant_output(output, "stop").aligned_to(&request_offering(&["glob", "read"]));
        let calls = parsed.tool_calls.expect("an ATEM invoke is a tool call");
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0].function.name, "glob",
            "the namespaced call is dispatched to the offered flat tool"
        );
    }
}

#[cfg(test)]
mod channel_gate_tests {
    use super::*;

    /// Feeds a turn to an armed gate one token at a time and returns what
    /// streamed.
    ///
    /// Each token is given as `(spelled, plain)` so the test can state exactly
    /// where the special tokens fall, which is what the gate reads.
    fn stream(tokens: &[(&str, &str)]) -> String {
        channels(tokens).0
    }

    /// The private thinking an armed gate reports for the same turn.
    fn thinking(tokens: &[(&str, &str)]) -> String {
        channels(tokens).1
    }

    /// What an armed gate routes to each channel, as `(answer, thinking)`.
    fn channels(tokens: &[(&str, &str)]) -> (String, String) {
        let mut gate = PrivateChannelGate::new(true);
        let mut answer = String::new();
        let mut thought = String::new();
        for (spelled, plain) in tokens {
            let revealed = gate.push(Some(spelled), plain);
            answer.push_str(&revealed.content);
            thought.push_str(&revealed.reasoning);
        }
        let remaining = gate.flush();
        answer.push_str(&remaining.content);
        thought.push_str(&remaining.reasoning);
        (answer, thought)
    }

    // Private reasoning must never reach a streaming client, and a turn cut off
    // before it addresses the user streams nothing rather than the thinking.
    #[test]
    fn reasoning_never_streams() {
        assert_eq!(
            stream(&[
                (" to=self", " to=self"),
                ("<|message|>", ""),
                ("secret", "secret"),
                (" plan", " plan"),
            ]),
            ""
        );
    }

    // Once the turn addresses the user, the answer streams incrementally and
    // the reasoning that preceded it stays withheld.
    #[test]
    fn only_the_user_channel_streams() {
        assert_eq!(
            stream(&[
                (" to=self", " to=self"),
                ("<|message|>", ""),
                ("secret", "secret"),
                ("<|end|> to=user<|message|>", "<|end|> to=user"),
                ("Hello", "Hello"),
                (" there", " there"),
                ("<|eot|>", ""),
            ]),
            "Hello there"
        );
    }

    // A model may address any tool it was offered, and that channel carries a
    // call rather than an answer, so it must not stream as content either.
    #[test]
    fn a_tool_channel_never_streams() {
        assert_eq!(
            stream(&[
                (" to=functions.bash", " to=functions.bash"),
                ("<|message|>", ""),
                ("<atem:invoke name=\"bash\">", "<atem:invoke name=\"bash\">"),
            ]),
            ""
        );
    }

    // An armed gate fails closed: a turn from a model that declares a private
    // channel is withheld until it names the channel the caller may see, even
    // when its opening tokens look like ordinary prose.
    #[test]
    fn an_armed_gate_withholds_an_unaddressed_turn() {
        assert_eq!(stream(&[("Hello", "Hello"), (" world", " world")]), "");
    }

    // A session prompt is not wrapped in the chat template, so the model spells
    // the turn header itself. The gate must not mistake that for prose and let
    // the reasoning through.
    #[test]
    fn a_self_opened_turn_still_hides_reasoning() {
        assert_eq!(
            stream(&[
                ("<|start|>", ""),
                ("assistant", "assistant"),
                (" to=self<|message|>", " to=self"),
                ("secret", "secret"),
                (
                    "<|end|><|start|>assistant to=user<|message|>",
                    "<|end|>assistant to=user"
                ),
                ("Hi", "Hi"),
            ]),
            "Hi"
        );
    }

    // A token the tokenizer cannot spell leaves channels undecidable. An armed
    // gate withholds it, because releasing a token that might be reasoning is
    // the one mistake it cannot take back.
    #[test]
    fn an_unspellable_token_is_withheld() {
        let mut gate = PrivateChannelGate::new(true);
        assert_eq!(gate.push(None, "raw"), ChannelDelta::default());
        assert_eq!(
            gate.push(Some(" to=self<|message|>"), " to=self").content,
            ""
        );
        assert_eq!(gate.push(Some("secret"), "secret").content, "");
    }

    // A model that declares no private channel is every ordinary model, and it
    // must stream exactly what it generated.
    #[test]
    fn an_unarmed_gate_streams_everything() {
        let mut gate = PrivateChannelGate::new(false);
        assert_eq!(
            gate.push(Some(" to=self<|message|>"), " to=self").content,
            " to=self"
        );
        assert_eq!(gate.push(None, "raw").content, "raw");
        assert_eq!(gate.flush(), ChannelDelta::default());
    }

    // Thinking is withheld from the answer but not thrown away: it is reported
    // on its own channel as it is produced, so a client can show the model
    // working instead of waiting on silence.
    #[test]
    fn thinking_is_reported_on_its_own_channel_as_it_is_produced() {
        let tokens = [
            (" to=self", " to=self"),
            ("<|message|>", ""),
            ("weigh", "weigh"),
            (" it", " it"),
            (
                "<|eom|><|start|>assistant to=user<|message|>",
                "assistant to=user",
            ),
            ("Hi", "Hi"),
        ];
        assert_eq!(stream(&tokens), "Hi");
        assert_eq!(thinking(&tokens), "weigh it");
    }

    // A turn that never reaches the user still reports what it was thinking,
    // which is the whole of what a truncated turn produced.
    #[test]
    fn a_turn_that_never_answers_still_reports_its_thinking() {
        let tokens = [
            (" to=self", " to=self"),
            ("<|message|>", ""),
            ("still", "still"),
            (" going", " going"),
        ];
        assert_eq!(stream(&tokens), "");
        assert_eq!(thinking(&tokens), "still going");
    }

    // Thinking resumed after a tool call is one train of thought, so the
    // segments join rather than the later one replacing the earlier.
    #[test]
    fn thinking_resumed_after_a_tool_call_extends_it() {
        let tokens = [
            (" to=self<|message|>", " to=self"),
            ("first", "first"),
            (
                "<|eom|><|start|>assistant to=self<|message|>",
                "assistant to=self",
            ),
            ("second", "second"),
        ];
        assert_eq!(thinking(&tokens), "firstsecond");
    }

    // An unarmed gate has no private channel to report, so it never invents one.
    #[test]
    fn an_unarmed_gate_reports_no_thinking() {
        let mut gate = PrivateChannelGate::new(false);
        assert_eq!(gate.push(Some(" to=self<|message|>"), "x").reasoning, "");
    }

    // The streaming loop skips the per-token special-token decode when the gate
    // is unarmed. That is sound only if an unarmed gate ignores `spelled`, so
    // pin it: `armed()` reports the state, and an unarmed gate yields the same
    // delta whether or not the spelling is supplied.
    #[test]
    fn an_unarmed_gate_ignores_spelling_so_the_decode_can_be_skipped() {
        assert!(!PrivateChannelGate::new(false).armed());
        assert!(PrivateChannelGate::new(true).armed());
        let mut with = PrivateChannelGate::new(false);
        let mut without = PrivateChannelGate::new(false);
        for plain in [" to=self", "weigh", " it", "Hi"] {
            let spelled = format!("{plain}<|message|>");
            assert_eq!(with.push(Some(&spelled), plain), without.push(None, plain));
        }
    }
}

// The buffered (non-streamed) path does not go through the gate: it composes
// `atem_reasoning_content` for the reasoning and `parse_assistant_output` for
// the answer, exactly as `run_chat_completion` does. These assert that a
// buffered turn reports its thinking on `reasoning_content` while keeping it out
// of `content`, so the two paths agree instead of the buffered one silently
// stripping reasoning as it did before #1197 was reversed.
#[cfg(test)]
mod buffered_reasoning_tests {
    use super::*;

    // The final buffered message reports the thinking beside the answer, and the
    // answer is only the user channel — reasoning is never spliced into content.
    #[test]
    fn a_buffered_message_reports_reasoning_beside_the_answer() {
        let output =
            " to=self<|message|>weigh it<|eom|><|start|>assistant to=user<|message|>Hi<|eot|>"
                .to_string();
        assert_eq!(atem_reasoning_content(&output).as_deref(), Some("weigh it"));
        assert_eq!(
            parse_assistant_output(output, "stop").content.as_deref(),
            Some("Hi")
        );
    }

    // A buffered turn that never reaches the user still reports what it thought,
    // and its answer is empty rather than the reasoning leaking out as one.
    #[test]
    fn a_buffered_turn_that_never_answers_still_reports_its_thinking() {
        let output = " to=self<|message|>still going<|eot|>".to_string();
        assert_eq!(
            atem_reasoning_content(&output).as_deref(),
            Some("still going")
        );
        assert_eq!(
            parse_assistant_output(output, "stop").content.as_deref(),
            Some("")
        );
    }
}

//! llama.cpp's native completion protocol uses the same admitted driver and
//! sampler preparation as chat. Only prompt construction and wire events differ.
use super::*;
use serde_json::{Map, Value, json};

struct NativeRequest {
    sampling: ChatCompletionRequest,
    prompt: GeneratePrompt,
    suffix: Option<String>,
    return_tokens: bool,
}

fn parse(mut body: Map<String, Value>, infill: bool) -> Result<NativeRequest, ApiError> {
    let prompt = match body.remove("prompt") {
        Some(prompt) => prompt,
        None if infill => json!(""),
        None => {
            return Err(ApiError::invalid_request(
                "prompt is required for native completions; send text or token ids",
            ));
        }
    };
    let prompt = match prompt {
        Value::String(text) => GeneratePrompt::Text(text),
        Value::Array(ids) => GeneratePrompt::TokenIds(ids.into_iter().enumerate().map(|(index, id)| {
            id.as_u64().and_then(|id| u32::try_from(id).ok()).ok_or_else(||
                ApiError::invalid_request(format!("prompt[{index}] must be an unsigned token id; send text or one token-id array")))
        }).collect::<Result<_, _>>()?),
        _ => return Err(ApiError::invalid_request("prompt must be text or an array of unsigned token ids")),
    };
    let suffix = if infill {
        let prefix = body
            .remove("input_prefix")
            .ok_or_else(|| ApiError::invalid_request("input_prefix is required for /infill"))?;
        let suffix = body
            .remove("input_suffix")
            .ok_or_else(|| ApiError::invalid_request("input_suffix is required for /infill"))?;
        let prefix = prefix
            .as_str()
            .ok_or_else(|| ApiError::invalid_request("input_prefix must be a string"))?
            .to_string();
        let suffix = suffix
            .as_str()
            .ok_or_else(|| ApiError::invalid_request("input_suffix must be a string"))?
            .to_string();
        if let Some(extra) = body.remove("input_extra")
            && extra != json!([])
        {
            return Err(ApiError::invalid_request(
                "input_extra is not declared by this model's FIM contract; omit it or send []",
            ));
        }
        match &prompt {
            GeneratePrompt::Text(text) if text.is_empty() => {}
            _ => {
                return Err(ApiError::invalid_request(
                    "additional infill prompt text is not declared by this model's FIM contract; omit prompt",
                ));
            }
        }
        Some((prefix, suffix))
    } else {
        None
    };
    if let Some(budget) = body.remove("n_predict")
        && body.insert("max_tokens".into(), budget).is_some()
    {
        return Err(ApiError::invalid_request(
            "send n_predict or max_tokens, not both",
        ));
    }
    let return_tokens = match body.remove("return_tokens") {
        None => false,
        Some(Value::Bool(value)) => value,
        Some(_) => return Err(ApiError::invalid_request("return_tokens must be a boolean")),
    };
    if let Some(schema) = body.remove("json_schema")
        && body.insert("response_format".into(), json!({"type": "json_schema", "json_schema": {"name":"completion", "schema":schema}})).is_some() {
            return Err(ApiError::invalid_request("send json_schema or response_format, not both"));
    }
    body.insert("messages".into(), json!([{"role":"user","content":""}]));
    let sampling: ChatCompletionRequest =
        serde_json::from_value(Value::Object(body)).map_err(|err| {
            ApiError::invalid_request(format!(
                "invalid native generation parameters: {err}; use the fields advertised by /props"
            ))
        })?;
    if sampling.has_tool_context() {
        return Err(ApiError::invalid_request(
            "native completions do not render tools; use /v1/chat/completions for tool calling",
        ));
    }
    let (prompt, suffix) = match suffix {
        Some((prefix, suffix)) => (GeneratePrompt::Text(prefix), Some(suffix)),
        None => (prompt, None),
    };
    Ok(NativeRequest {
        sampling,
        prompt,
        suffix,
        return_tokens,
    })
}

pub(crate) async fn completions(
    State(state): State<AppState>,
    ApiJson(body): ApiJson<Map<String, Value>>,
) -> Result<Response, ApiError> {
    run(state, parse(body, false)?).await
}
pub(crate) async fn infill(
    State(state): State<AppState>,
    ApiJson(body): ApiJson<Map<String, Value>>,
) -> Result<Response, ApiError> {
    run(state, parse(body, true)?).await
}

async fn run(state: AppState, request: NativeRequest) -> Result<Response, ApiError> {
    super::completions::validate_request(&request.sampling, &state.config)?;
    super::llama::validate_extensions(&request.sampling)?;
    let handle = resolve_model(&state.registry, &request.sampling.model).await?;
    let tokens = match &request.prompt {
        GeneratePrompt::Text(text) if request.suffix.is_some() => {
            let fim = handle.fim_config.as_ref().ok_or_else(|| {
                ApiError::invalid_request(format!(
                    "model '{}' declares no FIM tokens; select an infill-capable model",
                    handle.id
                ))
            })?;
            handle
                .tokenizer
                .encode(&fim.format_prompt(text, request.suffix.as_deref().unwrap_or_default()))
                .map_err(|err| {
                    ApiError::invalid_request(format!("infill tokenization failed: {err}"))
                })?
        }
        GeneratePrompt::Text(text) => handle.tokenizer.encode(text).map_err(|err| {
            ApiError::invalid_request(format!("prompt tokenization failed: {err}"))
        })?,
        GeneratePrompt::TokenRows(_) => {
            return Err(ApiError::invalid_request(
                "native completion accepts one prompt token row; submit batches as separate requests",
            ));
        }
        GeneratePrompt::TokenIds(ids) => {
            for id in ids {
                if handle.tokenizer.inner().id_to_token(*id).is_none() {
                    return Err(ApiError::invalid_request(format!(
                        "prompt token id {id} is absent from model '{}' vocabulary",
                        handle.id
                    )));
                }
            }
            ids.clone()
        }
    };
    let prompt_tokens = tokens.len();
    if tokens.is_empty() {
        return Err(ApiError::invalid_request(
            "prompt must encode to at least one token; send nonempty text or token ids",
        ));
    }
    let output_budget = super::completions::admit_output_budget(
        &request.sampling,
        prompt_tokens,
        request
            .sampling
            .output_budget(state.config.max_output_tokens),
        handle.model_max_context,
    )?;
    let mut options = super::completions::build_generate_options_with_tokenizer(
        &request.sampling,
        &handle.tokenizer,
        output_budget,
    )
    .map_err(|err| ApiError::invalid_request(err.to_string()))?;
    options.resolve_sampling_defaults(
        handle.generation_defaults.as_ref(),
        &super::completions::chat_sampling_overrides(&request.sampling),
    );
    options.max_context = handle.model_max_context;
    let generation = if let Some(suffix) = &request.suffix {
        let GeneratePrompt::Text(prefix) = &request.prompt else {
            return Err(ApiError::invalid_request(
                "infill input_prefix must be text",
            ));
        };
        handle
            .engine
            .generate_fim(
                prefix.clone(),
                suffix.clone(),
                handle.fim_config.clone().ok_or_else(|| {
                    ApiError::invalid_request("infill requires a declared FIM contract")
                })?,
                options,
            )
            .await
    } else {
        handle
            .engine
            .generate(
                None,
                GenerateRequest {
                    prompt: GeneratePrompt::TokenIds(tokens),
                    options,
                },
                onnx_genai_engine::ToolCallPolicy::Disabled,
                None,
            )
            .await
    }
    .map_err(map_generate_submit_error)?;
    super::completions::await_driver_admission(generation.admission).await?;
    crate::metrics::add_prompt_tokens(prompt_tokens);
    let model = handle.id.clone();
    let return_tokens = request.return_tokens;
    if !request.sampling.stream {
        let result = super::completions::collect_generation_result(generation.events)
            .await
            .map_err(generation_failure)?;
        return Ok(Json(final_value(
            &model,
            prompt_tokens,
            &result,
            return_tokens,
            false,
        ))
        .into_response());
    }
    let mut source = generation.events;
    let stops = request
        .sampling
        .stop
        .map(StopInput::into_texts)
        .unwrap_or_default();
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(16);
    tokio::spawn(async move {
        let mut stop_buffer = StopBoundaryBuffer::new(stops);
        let mut emitted = false;
        loop {
            let event = tokio::select! { _ = tx.closed() => break, event = source.recv() => event };
            let value = match event {
                Some(DriverEvent::Token(token)) => {
                    let content = stop_buffer.push(&token.text);
                    emitted |= !content.is_empty();
                    if matches!(token.finish_reason, Some(FinishReason::StopSequence { .. })) {
                        stop_buffer.pending.clear();
                    }
                    json!({"content":content,"tokens":if return_tokens {vec![token.token_id]} else {vec![]},"stop":false,"index":0})
                }
                Some(DriverEvent::WorkflowPublications(_)) => continue,
                Some(DriverEvent::Finished(result)) => {
                    let mut value =
                        final_value(&model, prompt_tokens, &result, return_tokens, true);
                    value["content"] = json!(if !emitted {
                        result.text.clone()
                    } else if matches!(result.finish_reason, FinishReason::StopSequence { .. }) {
                        String::new()
                    } else {
                        stop_buffer.flush()
                    });
                    let _ = tx.send(Ok(Event::default().data(value.to_string()))).await;
                    break;
                }
                Some(DriverEvent::Error(error)) => {
                    let error = generation_failure(error);
                    let _ = tx
                        .send(Ok(Event::default().event("error").data(
                            json!({"error":{"message":error.message,"type":error.kind}})
                                .to_string(),
                        )))
                        .await;
                    break;
                }
                None => {
                    let _ = tx.send(Ok(Event::default().event("error").data(json!({"error":{"message":"native completion driver ended without a result; retry the request","type":"server_error"}}).to_string()))).await;
                    break;
                }
            };
            if tx
                .send(Ok(Event::default().data(value.to_string())))
                .await
                .is_err()
            {
                break;
            }
        }
    });
    Ok(Sse::new(ReceiverStream::new(rx)).into_response())
}
fn final_value(
    model: &str,
    prompt_tokens: usize,
    result: &GenerateResult,
    return_tokens: bool,
    streaming: bool,
) -> Value {
    let stop_type = match result.finish_reason {
        FinishReason::MaxTokens | FinishReason::Length => "limit",
        FinishReason::StopSequence { .. } => "word",
        FinishReason::EosToken | FinishReason::ToolCalls => "eos",
    };
    json!({"content":result.text,"tokens":if return_tokens && !streaming { result.token_ids.clone() } else {vec![]},
        "stop":true,"index":0,"model":model,"tokens_predicted":result.token_ids.len(),
        "tokens_evaluated":prompt_tokens,"tokens_cached":result.prefix_cache_hit_len,
        "stop_type":stop_type,"truncated":matches!(result.finish_reason, FinishReason::Length)})
}

pub(crate) async fn embeddings(
    State(state): State<AppState>,
    ApiJson(mut body): ApiJson<Map<String, Value>>,
) -> Result<Json<Value>, ApiError> {
    let content = body.remove("content").ok_or_else(|| {
        ApiError::invalid_request(
            "content is required for native embeddings; send text, a text batch, or token arrays",
        )
    })?;
    let model = body.remove("model").unwrap_or_else(|| json!(""));
    let norm = match body.remove("embd_normalize") {
        None => 2,
        Some(value) => value.as_i64().filter(|value| *value >= -1).ok_or_else(|| {
            ApiError::invalid_request(
                "embd_normalize must be an integer >= -1; use -1 for none or 2 for L2",
            )
        })?,
    };
    if !body.is_empty() {
        return Err(ApiError::invalid_request(format!(
            "unsupported native embedding fields {:?}; send content and model",
            body.keys().collect::<Vec<_>>()
        )));
    }
    let request: EmbeddingRequest = serde_json::from_value(json!({"model":model,"input":content}))
        .map_err(|err| ApiError::invalid_request(format!("invalid embedding content: {err}")))?;
    let response = super::completions::embeddings(State(state), ApiJson(request))
        .await?
        .0;
    let result: Vec<_> = response
        .data
        .into_iter()
        .map(|mut data| {
            match &mut data.embedding {
                EmbeddingVector::Float(values) => normalize_embedding(values, norm),
                EmbeddingVector::Base64(_) => return Err(ApiError::internal("native embedding conversion requires float vectors; check the internal embedding encoding")),
            }
            Ok(json!({"index":data.index,"embedding":[data.embedding]}))
        })
        .collect::<Result<_, ApiError>>()?;
    Ok(Json(json!(result)))
}

// Scale before exponentiation so large p and finite f32 inputs stay finite.
fn normalize_embedding(vector: &mut [f32], norm: i64) {
    if norm == -1 {
        return;
    }
    let scale = vector
        .iter()
        .map(|value| f64::from(*value).abs())
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return;
    }
    let divisor = if norm == 0 {
        scale / 32760.0
    } else {
        let power = norm as f64;
        scale
            * vector
                .iter()
                .map(|value| (f64::from(*value).abs() / scale).powf(power))
                .sum::<f64>()
                .powf(1.0 / power)
    };
    for value in vector {
        *value = (f64::from(*value) / divisor) as f32;
    }
}

#[cfg(test)]
mod normalization_tests {
    use super::normalize_embedding;
    #[test]
    fn native_norms_are_finite_and_match_their_contract() {
        for (norm, expected) in [
            (-1, vec![3.0, 4.0]),
            (1, vec![3.0 / 7.0, 4.0 / 7.0]),
            (2, vec![0.6, 0.8]),
            (0, vec![24570.0, 32760.0]),
        ] {
            let mut vector = vec![3.0, 4.0];
            normalize_embedding(&mut vector, norm);
            for (actual, expected) in vector.iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-5);
            }
        }
        let mut vector = vec![f32::MAX, f32::MAX];
        normalize_embedding(&mut vector, 1000);
        assert!(
            vector
                .iter()
                .all(|value| value.is_finite() && *value <= 1.0)
        );
    }
}

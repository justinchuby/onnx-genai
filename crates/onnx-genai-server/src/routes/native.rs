//! llama.cpp's native completion protocol uses the same admitted driver and
//! sampler preparation as chat. Only prompt construction and wire events differ.
use super::prompt::MixedPrompt;
use super::*;
use serde_json::{Map, Value, json};

struct NativeRequest {
    sampling: Arc<ChatCompletionRequest>,
    prompt: MixedPrompt,
    suffix: Option<String>,
    return_tokens: bool,
    prefill_only: bool,
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
    let prompt = MixedPrompt::parse_completion(prompt)
        .map_err(|error| ApiError::invalid_request(format!("invalid prompt: {error}")))?;
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
            MixedPrompt::Text(text) if text.is_empty() => {}
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
    let prefill_only = body.get("max_tokens").and_then(Value::as_u64) == Some(0);
    if prefill_only {
        // Sampling validation still runs through the shared positive-output
        // schema; the operation below never submits that placeholder budget.
        body.insert("max_tokens".into(), json!(1));
        if body.contains_key("max_completion_tokens") {
            return Err(ApiError::invalid_request(
                "send n_predict=0 or max_tokens=0 without max_completion_tokens",
            ));
        }
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
    let sampling: ChatCompletionRequest = serde_path_to_error::deserialize(Value::Object(body))
        .map_err(|err| {
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
        Some((prefix, suffix)) => (MixedPrompt::Text(prefix), Some(suffix)),
        None => (prompt, None),
    };
    Ok(NativeRequest {
        sampling: Arc::new(sampling),
        prompt,
        suffix,
        return_tokens,
        prefill_only,
    })
}

enum NativeOutput {
    Buffered(Value),
    Stream {
        receiver: mpsc::Receiver<Result<Event, Infallible>>,
        completed: oneshot::Receiver<()>,
    },
}
impl IntoResponse for NativeOutput {
    fn into_response(self) -> Response {
        match self {
            Self::Buffered(value) => Json(value).into_response(),
            Self::Stream { receiver, .. } => {
                Sse::new(ReceiverStream::new(receiver)).into_response()
            }
        }
    }
}
struct PreparedNative {
    request: NativeRequest,
    handle: Arc<ModelHandle>,
    operation: NativeOperation,
}

enum NativeOperation {
    Generate {
        tokens: Vec<u32>,
        options: Box<GenerateOptions>,
    },
    Prefill(onnx_genai_engine::PrefillRequest),
}

pub(crate) async fn completions(
    State(state): State<AppState>,
    ApiJson(mut body): ApiJson<Map<String, Value>>,
) -> Result<Response, ApiError> {
    let prompt = body.remove("prompt").ok_or_else(|| {
        ApiError::invalid_request(
            "prompt is required for native completions; send text, token ids or a prompt batch",
        )
    })?;
    let super::prompt::PromptBatch { first, rest } =
        super::prompt::PromptBatch::parse(prompt, state.config.max_completion_prompts)?;
    let prompts = rest.into_iter();
    body.insert("prompt".into(), first);
    let first = parse(body, false)?;
    let mut requests = Vec::with_capacity(prompts.len() + 1);
    for prompt in prompts {
        requests.push(NativeRequest {
            sampling: Arc::clone(&first.sampling),
            prompt: MixedPrompt::parse_completion(prompt)
                .map_err(|error| ApiError::invalid_request(format!("invalid prompt: {error}")))?,
            suffix: None,
            return_tokens: first.return_tokens,
            prefill_only: first.prefill_only,
        });
    }
    validate_native_request(&first, &state.config)?;
    // One batch retains one admitted package version across admin reloads.
    let handle = resolve_model(&state.registry, &first.sampling.model).await?;
    let mut prepared = Vec::with_capacity(requests.len() + 1);
    prepared.push(prepare(&state, first, Arc::clone(&handle))?);
    for request in requests {
        prepared.push(prepare(&state, request, Arc::clone(&handle))?);
    }
    let mut inputs = prepared.into_iter().enumerate();
    let (index, first) = inputs
        .next()
        .expect("nonempty prompt batch validated above");
    let first = run(first, index).await?;
    match first {
        NativeOutput::Buffered(value) => {
            let mut values = vec![value];
            for (index, input) in inputs {
                let NativeOutput::Buffered(value) = run(input, index).await? else {
                    return Err(ApiError::internal(
                        "native batch changed response mode; retry with a single stream setting",
                    ));
                };
                values.push(value);
            }
            Ok(Json(if values.len() == 1 {
                values.remove(0)
            } else {
                json!(values)
            })
            .into_response())
        }
        NativeOutput::Stream {
            receiver: source,
            completed: source_completed,
        } => {
            let (tx, rx) = mpsc::channel(16);
            let (completed, outcome) = oneshot::channel();
            tokio::spawn(forward_native_batch(
                source,
                source_completed,
                inputs,
                tx,
                completed,
                run,
            ));
            Ok(NativeOutput::Stream {
                receiver: rx,
                completed: outcome,
            }
            .into_response())
        }
    }
}
async fn forward_native_batch<T, I, Start, Started>(
    mut source: mpsc::Receiver<Result<Event, Infallible>>,
    mut source_completed: oneshot::Receiver<()>,
    mut inputs: I,
    tx: mpsc::Sender<Result<Event, Infallible>>,
    completed: oneshot::Sender<()>,
    mut start: Start,
) where
    I: Iterator<Item = (usize, T)>,
    Start: FnMut(T, usize) -> Started,
    Started: std::future::Future<Output = Result<NativeOutput, ApiError>>,
{
    loop {
        loop {
            let event =
                tokio::select! { _ = tx.closed() => return, event = source.recv() => event };
            let Some(event) = event else {
                break;
            };
            if tx.send(event).await.is_err() {
                return;
            }
        }
        if (&mut source_completed).await.is_err() {
            return;
        }
        if tx.is_closed() {
            return;
        }
        let Some((index, input)) = inputs.next() else {
            let _ = completed.send(());
            break;
        };
        let output = tokio::select! { biased; _ = tx.closed() => return, output = start(input, index) => output };
        match output {
            Ok(NativeOutput::Stream {
                receiver,
                completed,
            }) => {
                source = receiver;
                source_completed = completed;
            }
            Ok(NativeOutput::Buffered(_)) => break,
            Err(error) => {
                let _ = tx.send(Ok(Event::default().event("error").data(
                                json!({"error":{"message":error.message,"type":error.kind,"code":error.status.as_u16()},"index":index}).to_string()
                            ))).await;
                break;
            }
        }
    }
}
pub(crate) async fn infill(
    State(state): State<AppState>,
    ApiJson(body): ApiJson<Map<String, Value>>,
) -> Result<Response, ApiError> {
    let request = parse(body, true)?;
    validate_native_request(&request, &state.config)?;
    let handle = resolve_model(&state.registry, &request.sampling.model).await?;
    Ok(run(prepare(&state, request, handle)?, 0)
        .await?
        .into_response())
}
fn validate_native_request(request: &NativeRequest, config: &ServerConfig) -> Result<(), ApiError> {
    super::completions::validate_request(&request.sampling, config)?;
    super::llama::validate_extensions(&request.sampling)
}
fn prepare(
    state: &AppState,
    request: NativeRequest,
    handle: Arc<ModelHandle>,
) -> Result<PreparedNative, ApiError> {
    let tokens = match &request.prompt {
        MixedPrompt::Text(text) if request.suffix.is_some() => {
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
        prompt => prompt.encode(
            &handle.tokenizer,
            onnx_genai_ort::TokenizationOptions {
                add_special: true,
                parse_special: true,
            },
        )?,
    };
    let prompt_tokens = tokens.len();
    if tokens.is_empty() {
        return Err(ApiError::invalid_request(
            "prompt must encode to at least one token; send nonempty text or token ids",
        ));
    }
    if request.prefill_only {
        super::completions::build_generate_options_with_tokenizer(
            &request.sampling,
            &handle.tokenizer,
            1,
        )
        .map_err(|error| ApiError::invalid_request(error.to_string()))?;
        if handle
            .model_max_context
            .is_some_and(|limit| prompt_tokens > limit)
        {
            return Err(ApiError::invalid_request(format!(
                "prefill prompt contains {prompt_tokens} tokens, exceeding model context {:?}; shorten the prompt",
                handle.model_max_context
            )));
        }
        let cold_start = request
            .sampling
            .extensions
            .get("cache_prompt")
            .and_then(Value::as_bool)
            == Some(false);
        return Ok(PreparedNative {
            request,
            operation: NativeOperation::Prefill(onnx_genai_engine::PrefillRequest {
                prompt: GeneratePrompt::TokenIds(tokens),
                max_context: handle.model_max_context,
                cold_start,
            }),
            handle,
        });
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
    Ok(PreparedNative {
        request,
        handle,
        operation: NativeOperation::Generate {
            tokens,
            options: Box::new(options),
        },
    })
}
async fn run(prepared: PreparedNative, index: usize) -> Result<NativeOutput, ApiError> {
    let PreparedNative {
        request,
        handle,
        operation,
    } = prepared;
    let (tokens, options) = match operation {
        NativeOperation::Generate { tokens, options } => (tokens, *options),
        NativeOperation::Prefill(input) => {
            let result = handle
                .engine
                .prefill(input)
                .await
                .map_err(map_generate_submit_error)?;
            let value = json!({"content":"","tokens":[],"stop":true,"index":index,"model":handle.id,
                "tokens_predicted":0,"tokens_evaluated":result.prompt_tokens,
                "tokens_cached":result.prefix_cache_hit_len,"stop_type":"limit","truncated":false});
            if !request.sampling.stream {
                return Ok(NativeOutput::Buffered(value));
            }
            let (tx, receiver) = mpsc::channel(1);
            let (completed, outcome) = oneshot::channel();
            let _ = tx.send(Ok(Event::default().data(value.to_string()))).await;
            let _ = completed.send(());
            return Ok(NativeOutput::Stream {
                receiver,
                completed: outcome,
            });
        }
    };
    let prompt_tokens = tokens.len();
    let generation = if let Some(suffix) = &request.suffix {
        let MixedPrompt::Text(prefix) = &request.prompt else {
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
    let stops = request
        .sampling
        .stop
        .clone()
        .map(StopInput::into_texts)
        .unwrap_or_default();
    if !request.sampling.stream {
        let result = super::completions::collect_generation_result(generation.events)
            .await
            .map_err(generation_failure)?;
        return Ok(NativeOutput::Buffered(final_value(
            &model,
            prompt_tokens,
            &result,
            return_tokens,
            false,
            index,
            &stops,
        )));
    }
    let mut source = generation.events;
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(16);
    let (completed, outcome) = oneshot::channel();
    tokio::spawn(async move {
        let mut stop_buffer = StopBoundaryBuffer::new(stops.clone());
        let mut published = String::new();
        loop {
            let event = tokio::select! { _ = tx.closed() => break, event = source.recv() => event };
            let value = match event {
                Some(DriverEvent::Token(token)) => {
                    let content = stop_buffer.push(&token.text);
                    published.push_str(&content);
                    if matches!(token.finish_reason, Some(FinishReason::StopSequence { .. })) {
                        stop_buffer.discard_pending();
                    }
                    json!({"content":content,"tokens":if return_tokens {vec![token.token_id]} else {vec![]},"stop":false,"index":index})
                }
                Some(DriverEvent::WorkflowPublications(_)) => continue,
                Some(DriverEvent::Finished(result)) => {
                    let mut value = final_value(
                        &model,
                        prompt_tokens,
                        &result,
                        return_tokens,
                        true,
                        index,
                        &stops,
                    );
                    let Some(tail) =
                        text_before_stop(&result.text, &stops).strip_prefix(&published)
                    else {
                        let _ = tx.send(Ok(Event::default().event("error").data(json!({
                            "error":{"message":"native tokenizer's final text does not preserve the published stream prefix; repair its incremental decoder or retry with stream=false","type":"server_error","code":500},"index":index
                        }).to_string()))).await;
                        break;
                    };
                    value["content"] = json!(tail);
                    if tx
                        .send(Ok(Event::default().data(value.to_string())))
                        .await
                        .is_ok()
                    {
                        let _ = completed.send(());
                    }
                    break;
                }
                Some(DriverEvent::Error(error)) => {
                    let error = generation_failure(error);
                    let _ = tx
                        .send(Ok(Event::default().event("error").data(
                            json!({"error":{"message":error.message,"type":error.kind,"code":error.status.as_u16()},"index":index})
                                .to_string(),
                        )))
                        .await;
                    break;
                }
                None => {
                    let _ = tx.send(Ok(Event::default().event("error").data(json!({"error":{"message":"native completion driver ended without a result; retry the request","type":"server_error","code":500},"index":index}).to_string()))).await;
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
    Ok(NativeOutput::Stream {
        receiver: rx,
        completed: outcome,
    })
}
fn final_value(
    model: &str,
    prompt_tokens: usize,
    result: &GenerateResult,
    return_tokens: bool,
    streaming: bool,
    index: usize,
    stops: &[String],
) -> Value {
    let stop_type = match result.finish_reason {
        FinishReason::MaxTokens | FinishReason::Length => "limit",
        FinishReason::StopSequence { .. } => "word",
        FinishReason::EosToken | FinishReason::ToolCalls => "eos",
    };
    json!({"content":text_before_stop(&result.text, stops),"tokens":if return_tokens && !streaming { result.token_ids.clone() } else {vec![]},
        "stop":true,"index":index,"model":model,"tokens_predicted":result.token_ids.len(),
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

/// Native speech shares the package-declared text assembly and PCM contract.
pub(crate) async fn tts(
    State(state): State<AppState>,
    ApiJson(mut body): ApiJson<Map<String, Value>>,
) -> Result<Response, ApiError> {
    if let Some(prompt) = body.remove("prompt")
        && body.insert("input".into(), prompt).is_some()
    {
        return Err(ApiError::invalid_request(
            "send input or prompt to /tts, not both",
        ));
    }
    body.entry("model").or_insert(json!(""));
    for (key, neutral) in [
        ("speed", json!(1.0)),
        ("top_k", json!(0)),
        ("top_p", json!(0)),
        ("seed", json!(-1)),
        ("loudness", json!(0)),
        ("cfg_scale", json!(0)),
        ("flow_steps", json!(0)),
    ] {
        if let Some(value) = body.remove(key) {
            let accepted = value == neutral
                || value
                    .as_f64()
                    .zip(neutral.as_f64())
                    .is_some_and(|(a, b)| a == b);
            if !accepted {
                return Err(ApiError::invalid_request(format!(
                    "/tts field '{key}' requires {neutral} for this package's speech contract; omit it to use declared model settings"
                )));
            }
        }
    }
    let request = serde_json::from_value(Value::Object(body)).map_err(|err| ApiError::invalid_request(format!("invalid native speech request: {err}; send input/prompt and optional model; voice cloning and undeclared synthesis controls require a capable package contract")))?;
    super::audio_speech(State(state), ApiJson(request)).await
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

#[cfg(test)]
mod batch_lifecycle_tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[tokio::test]
    async fn native_batch_does_not_submit_after_failed_stream_or_disconnected_consumer() {
        for disconnect in [false, true] {
            let (source, receiver) = mpsc::channel(2);
            let (source_done, source_completed) = oneshot::channel();
            let (tx, rx) = mpsc::channel(2);
            let (done, outcome) = oneshot::channel();
            let calls = AtomicUsize::new(0);
            let mut retained_consumer = Some(rx);
            if disconnect {
                drop(retained_consumer.take());
            } else {
                source
                    .send(Ok(Event::default().event("error").data("failure")))
                    .await
                    .unwrap();
                drop(source_done);
                drop(source);
                // A closed stream without its completion proof is failure,
                // even if the error frame itself was delivered successfully.
                forward_native_batch(
                    receiver,
                    source_completed,
                    [(1, ())].into_iter(),
                    tx,
                    done,
                    |(), _| {
                        calls.fetch_add(1, Ordering::SeqCst);
                        async { Err(ApiError::internal("later input must not be submitted")) }
                    },
                )
                .await;
                assert_eq!(calls.load(Ordering::SeqCst), 0);
                assert!(outcome.await.is_err());
                assert!(retained_consumer.as_mut().unwrap().recv().await.is_some());
                continue;
            }
            forward_native_batch(
                receiver,
                source_completed,
                [(1, ())].into_iter(),
                tx,
                done,
                |(), _| {
                    calls.fetch_add(1, Ordering::SeqCst);
                    async { Err(ApiError::internal("later input must not be submitted")) }
                },
            )
            .await;
            assert_eq!(calls.load(Ordering::SeqCst), 0);
            assert!(source.is_closed());
            assert!(outcome.await.is_err());
        }
    }

    #[tokio::test]
    async fn native_batch_submits_next_input_only_after_explicit_completion() {
        let (source, receiver) = mpsc::channel(2);
        let (source_done, source_completed) = oneshot::channel();
        source
            .send(Ok(Event::default().data("first")))
            .await
            .unwrap();
        source_done.send(()).unwrap();
        drop(source);
        let (tx, mut rx) = mpsc::channel(4);
        let (done, outcome) = oneshot::channel();
        let calls = AtomicUsize::new(0);
        forward_native_batch(
            receiver,
            source_completed,
            [(1, ())].into_iter(),
            tx,
            done,
            |(), index| {
                calls.fetch_add(1, Ordering::SeqCst);
                assert_eq!(index, 1);
                async {
                    let (source, receiver) = mpsc::channel(2);
                    let (done, completed) = oneshot::channel();
                    source
                        .send(Ok(Event::default().data("second")))
                        .await
                        .unwrap();
                    done.send(()).unwrap();
                    Ok(NativeOutput::Stream {
                        receiver,
                        completed,
                    })
                }
            },
        )
        .await;
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        outcome.await.unwrap();
        assert!(rx.recv().await.is_some());
        assert!(rx.recv().await.is_some());
        assert!(rx.recv().await.is_none());
    }
}

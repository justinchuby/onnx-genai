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
    probabilities: NativeProbabilities,
    response_fields: Arc<ResponseFields>,
}

/// Pinned paths traverse objects and retain the entire path as the output key.
#[derive(Default)]
struct ResponseFields(Vec<String>);
impl ResponseFields {
    fn project(&self, value: Value) -> Value {
        if self.0.is_empty() {
            return value;
        }
        let mut output = Map::new();
        for path in &self.0 {
            let selected = path
                .split('/')
                .try_fold(&value, |current, key| current.as_object()?.get(key));
            if let Some(selected) = selected {
                output.insert(path.clone(), selected.clone());
            }
        }
        Value::Object(output)
    }
}

#[derive(Clone, Copy, Default)]
struct NativeProbabilities {
    top: usize,
    post: bool,
}

fn parse(mut body: Map<String, Value>, infill: bool) -> Result<NativeRequest, ApiError> {
    let response_fields=Arc::new(ResponseFields(match body.remove("response_fields") {
        None=>Vec::new(),
        Some(value)=>serde_json::from_value::<Vec<String>>(value).map_err(|error| ApiError::invalid_request(format!("response_fields must be an array of string paths; use [] to return every field: {error}")))?,
    }));
    let top = match body.remove("n_probs") {
        None => 0,
        Some(value) => value
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                ApiError::invalid_request(
                    "n_probs must be a nonnegative integer; use 0 to disable token probabilities",
                )
            })?,
    };
    let post = match body.remove("post_sampling_probs") {
        None => false,
        Some(Value::Bool(value)) => value,
        Some(_) => {
            return Err(ApiError::invalid_request(
                "post_sampling_probs must be a boolean; use false for model probabilities or true for processed probabilities",
            ));
        }
    };
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
    // The pinned native sampler chooses the grammar field when it is present,
    // including an empty grammar; chat instead rejects conflicting constraints.
    if body.contains_key("grammar") {
        body.remove("json_schema");
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
        probabilities: NativeProbabilities { top, post },
        response_fields,
    })
}

enum NativeOutput {
    Buffered(Value),
    Stream {
        receiver: mpsc::Receiver<Result<Event, Infallible>>,
        completed: oneshot::Receiver<()>,
    },
}
impl NativeOutput {
    fn response(self, ping: crate::SsePingInterval) -> Response {
        match self {
            Self::Buffered(value) => Json(value).into_response(),
            Self::Stream { receiver, .. } => ping.response(ReceiverStream::new(receiver)),
        }
    }
}
struct PreparedNative {
    request: NativeRequest,
    handle: Arc<ModelHandle>,
    operation: NativeOperation,
    metadata: NativeFinalMetadata,
}

struct NativeFinalMetadata {
    tokenizer: Arc<onnx_genai_ort::Tokenizer>,
    minimum_indent: usize,
    model: String,
    prompt: String,
    slot: Option<usize>,
    settings: Value,
    stops: Vec<String>,
}
impl NativeFinalMetadata {
    fn new(
        handle: &ModelHandle,
        tokens: &[u32],
        options: &GenerateOptions,
        request: &NativeRequest,
        budget: usize,
    ) -> Result<Self, ApiError> {
        let stops = options
            .stop_sequences
            .iter()
            .filter_map(|stop| match stop {
                onnx_genai_engine::logits::StopSequence::Text(text) => Some(text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>();
        let stages = options
            .sampler_order
            .as_ref()
            .map_or(onnx_genai_engine::SamplerSequence::DEFAULT, |sequence| {
                sequence.stages()
            });
        let mut settings = json!({
            "temperature":options.temperature,"top_k":options.top_k,"top_p":options.top_p,
            "min_p":options.min_p,"top_a":options.top_a,"typical_p":options.typical_p,
            "top_n_sigma":options.top_n_sigma,"repeat_penalty":options.repetition_penalty,
            "repeat_last_n":options.penalty_window.map_or(json!(-1), |n| json!(n)),
            "presence_penalty":options.presence_penalty,"frequency_penalty":options.frequency_penalty,
            "adaptive_target":options.adaptive_p.target,"adaptive_decay":options.adaptive_p.decay,
            "n_predict":budget,"max_tokens":budget,"n_indent":options.minimum_indent,"ignore_eos":!options.stop_on_eos,
            "stream":request.sampling.stream,"stop":stops,"samplers":stages,
            "grammar_trigger_collation":request.sampling.grammar_trigger_collation.unwrap_or_default(),
            "n_probs":options.top_logprobs.unwrap_or(0),"post_sampling_probs":request.probabilities.post,
            "cache_prompt":!options.cold_start,"preserved_tokens":options.effective_preserved_tokens(),
            "dynatemp_range":options.dynamic_temperature.map_or(0.0, |c| c.range),
            "dry_multiplier":options.dry.as_ref().map_or(0.0, |c| c.multiplier),
            "xtc_probability":options.xtc.map_or(0.0, |c| c.probability),
            "mirostat":options.mirostat.map_or(0, |c| match c.version {
                onnx_genai_engine::MirostatVersion::V1 => 1,
                onnx_genai_engine::MirostatVersion::V2 => 2,
            }),
            "timings_per_token":request.sampling.extensions.get("timings_per_token").and_then(Value::as_bool).unwrap_or(false),
            "logit_bias":options.logit_bias.iter().map(|(token,bias)| json!({"token":token,"bias":match bias {
                onnx_genai_engine::logits::LogitBias::Add(value) => json!(value),
                onnx_genai_engine::logits::LogitBias::Ban => Value::Null,
            }})).collect::<Vec<_>>(),
        });
        if let Some(seed) = options.seed {
            settings["seed"] = json!(seed);
        }
        if let Some(config) = options.dynamic_temperature {
            settings["dynatemp_exponent"] = json!(config.exponent);
        }
        if let Some(config) = options.xtc {
            settings["xtc_threshold"] = json!(config.threshold);
        }
        if let Some(config) = options.mirostat {
            settings["mirostat_tau"] = json!(config.tau);
            settings["mirostat_eta"] = json!(config.eta);
        }
        if let Some(config) = &options.dry {
            settings["dry_base"] = json!(config.base);
            settings["dry_allowed_length"] = json!(config.allowed_length);
            settings["dry_penalty_last_n"] = config.window.map_or(json!(-1), |n| json!(n));
            settings["dry_sequence_breakers"] = json!(request.sampling.dry_sequence_breakers);
        }
        Ok(Self {
            tokenizer:handle.tokenizer.clone(), minimum_indent:options.minimum_indent,
            model:handle.id.clone(),
            prompt:handle.tokenizer.decode_with_special_tokens(tokens).map_err(|error| ApiError::invalid_request(format!("native response prompt detokenization failed: {error}; repair the package tokenizer or send valid token IDs")))?,
            slot:handle.engine.native_slot.as_ref().map(|_| 0),
            settings,stops,
        })
    }
    fn decorate(&self, value: &mut Value, has_new_line: bool, stopping_word: &str) {
        value["prompt"] = json!(self.prompt);
        value["generation_settings"] = self.settings.clone();
        value["has_new_line"] = json!(has_new_line);
        value["stopping_word"] = json!(stopping_word);
        if let Some(slot) = self.slot {
            value["id_slot"] = json!(slot);
        }
    }
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
            probabilities: first.probabilities,
            response_fields: Arc::clone(&first.response_fields),
        });
    }
    validate_native_request(&first, &state.config)?;
    let ping = crate::SsePingInterval::resolve(&first.sampling, state.config.sse_ping_interval)?;
    // One batch retains one admitted package version across admin reloads.
    let handle = resolve_model(&state.registry, &first.sampling.model).await?;
    let mut prepared = Vec::with_capacity(requests.len() + 1);
    prepared.push(prepare(&state, first, Arc::clone(&handle))?);
    for request in requests {
        prepared.push(prepare(&state, request, Arc::clone(&handle))?);
    }
    let adapter_selection = match &prepared[0].operation {
        NativeOperation::Generate { options, .. } => options.adapter_selection.clone(),
        NativeOperation::Prefill(_) => None,
    };
    for input in prepared.iter_mut().skip(1) {
        if let NativeOperation::Generate { options, .. } = &mut input.operation {
            options.adapter_selection = adapter_selection.clone();
        }
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
            .response(ping))
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
    let ping = crate::SsePingInterval::resolve(&request.sampling, state.config.sse_ping_interval)?;
    let handle = resolve_model(&state.registry, &request.sampling.model).await?;
    Ok(run(prepare(&state, request, handle)?, 0)
        .await?
        .response(ping))
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
    super::llama::validate_slot_selection(&request.sampling, &handle)?;
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
        let mut prefill_options = super::completions::build_generate_options_with_tokenizer(
            &request.sampling,
            &handle.tokenizer,
            1,
        )
        .map_err(|error| ApiError::invalid_request(error.to_string()))?;
        crate::lora::apply_request(&request.sampling, &handle, &mut prefill_options, false)?;
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
        prefill_options.resolve_sampling_defaults(
            handle.generation_defaults.as_ref(),
            &super::completions::chat_sampling_overrides(&request.sampling),
        );
        prefill_options.cold_start = cold_start;
        let metadata = NativeFinalMetadata::new(&handle, &tokens, &prefill_options, &request, 0)?;
        return Ok(PreparedNative {
            metadata,
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
    .map_err(|err| ApiError::invalid_request(format!("{err:#}")))?;
    crate::lora::apply_request(&request.sampling, &handle, &mut options, false)?;
    if request.probabilities.top > 0 {
        options.top_logprobs = Some(
            request
                .probabilities
                .top
                .min(handle.tokenizer.inner().get_vocab_size(true)),
        );
        options.logprob_source = if request.probabilities.post {
            onnx_genai_engine::LogprobSource::Sampling
        } else {
            onnx_genai_engine::LogprobSource::Model
        };
    }
    options.max_constraint_bytes = state.config.replay_buffer_bytes;
    options.resolve_sampling_defaults(
        handle.generation_defaults.as_ref(),
        &super::completions::chat_sampling_overrides(&request.sampling),
    );
    options.max_context = handle.model_max_context;
    crate::reasoning_request::ReasoningRequest::extract(&request.sampling)
        .and_then(|controls| {
            controls.apply(
                &mut options,
                &handle.tokenizer,
                crate::reasoning_request::BudgetContext::Raw,
            )
        })
        .map_err(|error| {
            ApiError::invalid_request(format!("invalid native reasoning controls: {error:#}"))
        })?;
    if options.constraint.is_some() || options.reasoning_budget.is_some() {
        options
            .validate_processors(&handle.tokenizer)
            .map_err(|error| {
                ApiError::invalid_request(format!(
                    "invalid native completion constraint: {error:#}"
                ))
            })?;
    }
    let metadata = NativeFinalMetadata::new(&handle, &tokens, &options, &request, output_budget)?;
    Ok(PreparedNative {
        metadata,
        request,
        handle,
        operation: NativeOperation::Generate {
            tokens,
            options: Box::new(options),
        },
    })
}
fn prefill_value(
    metadata: &NativeFinalMetadata,
    index: usize,
    result: onnx_genai_engine::PrefillResult,
) -> Value {
    let mut value = json!({"content":"","tokens":[],"stop":true,"index":index,"model":metadata.model,
        "tokens_predicted":0,"tokens_evaluated":result.prompt_tokens,
        "tokens_cached":result.retained_cache_tokens,"stop_type":"limit","truncated":false,
        "timings":crate::timings::render(result.timings)});
    metadata.decorate(&mut value, false, "");
    value
}

async fn run(prepared: PreparedNative, index: usize) -> Result<NativeOutput, ApiError> {
    let PreparedNative {
        request,
        handle,
        operation,
        metadata,
    } = prepared;
    let response_fields = Arc::clone(&request.response_fields);
    let (tokens, options) = match operation {
        NativeOperation::Generate { tokens, options } => (tokens, *options),
        NativeOperation::Prefill(input) => {
            let mut prefill = handle
                .engine
                .prefill(input)
                .await
                .map_err(map_generate_submit_error)?;
            super::completions::await_driver_admission(prefill.admission).await?;
            let return_progress = request
                .sampling
                .extensions
                .get("return_progress")
                .and_then(Value::as_bool)
                == Some(true);
            if !request.sampling.stream {
                let mut open = true;
                loop {
                    tokio::select! {
                        biased;
                        progress = prefill.progress.recv(), if open => { open = progress.is_some(); }
                        result = &mut prefill.result => {
                            let result = result.map_err(|_| ApiError::internal("prompt-only worker stopped before returning its committed result; inspect server logs and retry"))?
                                .map_err(generation_failure)?;
                            return Ok(NativeOutput::Buffered(response_fields.project(prefill_value(&metadata, index, result))));
                        }
                    }
                }
            }
            let (tx, receiver) = mpsc::channel(16);
            let (completed, outcome) = oneshot::channel();
            tokio::spawn(async move {
                let mut open = true;
                loop {
                    tokio::select! {
                        biased;
                        _ = tx.closed() => break,
                        progress = prefill.progress.recv(), if open => {
                            match progress {
                                Some(DriverEvent::PromptProgress(progress)) if return_progress => {
                                    let value = json!({"content":"","tokens":[],"stop":false,"index":index,
                                        "tokens_predicted":0,"tokens_evaluated":progress.total,
                                        "prompt_progress":crate::timings::render_prompt_progress(progress)});
                                    if tx.send(Ok(Event::default().data(value.to_string()))).await.is_err() { break; }
                                }
                                None => open = false,
                                _ => {}
                            }
                        }
                        result = &mut prefill.result => {
                            let result = result.map_err(|_| ApiError::internal("prompt-only worker stopped before returning its committed result; inspect server logs and retry"))
                                .and_then(|result| result.map_err(generation_failure));
                            match result {
                                Ok(result) => {
                                    let value = response_fields.project(prefill_value(&metadata, index, result));
                                    if tx.send(Ok(Event::default().data(value.to_string()))).await.is_ok() {
                                        let _ = completed.send(());
                                    }
                                }
                                Err(error) => {
                                    let value = json!({"error":{"message":error.message,"type":error.kind,"code":500},"index":index});
                                    let _ = tx.send(Ok(Event::default().event("error").data(value.to_string()))).await;
                                }
                            }
                            break;
                        }
                    }
                }
            });
            return Ok(NativeOutput::Stream {
                receiver,
                completed: outcome,
            });
        }
    };
    let minimum_indent = options.minimum_indent;
    let probabilities = ProbabilityOutput::new(request.probabilities, &handle.tokenizer)?;
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
    let return_tokens = request.return_tokens;
    let return_progress = request
        .sampling
        .extensions
        .get("return_progress")
        .and_then(Value::as_bool)
        == Some(true);
    let timings_per_token = request
        .sampling
        .extensions
        .get("timings_per_token")
        .and_then(Value::as_bool)
        == Some(true);
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
        let mut value = final_value(
            &metadata,
            prompt_tokens,
            &result,
            return_tokens,
            false,
            index,
        )?;
        probabilities.decorate_result(&mut value, &result, &handle.tokenizer, &stops)?;
        return Ok(NativeOutput::Buffered(response_fields.project(value)));
    }
    let mut source = generation.events;
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(16);
    let (completed, outcome) = oneshot::channel();
    tokio::spawn(async move {
        let mut stop_buffer =
            StopBoundaryBuffer::new(stops.clone()).with_minimum_indent(minimum_indent);
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
                    let mut value = json!({"content":content,"tokens":if return_tokens {vec![token.token_id]} else {vec![]},"stop":false,"index":index});
                    if timings_per_token && let Some(sample) = token.timings {
                        value["timings"] = crate::timings::render(sample);
                    }
                    if probabilities.config.top > 0 {
                        let probability = token.logprob.as_ref().ok_or_else(|| ApiError::internal("native probability stream is missing a committed token distribution; select a decoder with probability reporting"))
                            .and_then(|probability| probabilities.render(probability, content.as_bytes()));
                        match probability {
                            Ok(probability) => {
                                value["completion_probabilities"] = json!([probability])
                            }
                            Err(error) => {
                                let _ = tx.send(Ok(Event::default().event("error").data(json!({"error":{"message":error.message,"type":error.kind,"code":500},"index":index}).to_string()))).await;
                                break;
                            }
                        }
                    }
                    value
                }
                Some(DriverEvent::PromptProgress(progress)) => {
                    if !return_progress {
                        continue;
                    }
                    json!({"content":"","tokens":[],"stop":false,"index":index,
                        "tokens_predicted":0,"tokens_evaluated":prompt_tokens,
                        "prompt_progress":crate::timings::render_prompt_progress(progress)})
                }
                Some(DriverEvent::WorkflowPublications(_)) => continue,
                Some(DriverEvent::Finished(result)) => {
                    let mut value = match final_value(
                        &metadata,
                        prompt_tokens,
                        &result,
                        return_tokens,
                        true,
                        index,
                    ) {
                        Ok(value) => value,
                        Err(error) => {
                            let _ = tx.send(Ok(Event::default().event("error").data(json!({
                                "error":{"message":error.message,"type":error.kind,"code":500},"index":index
                            }).to_string()))).await;
                            break;
                        }
                    };
                    let Some(tail) =
                        text_before_stop(&result.text, &stops).strip_prefix(&published)
                    else {
                        let _ = tx.send(Ok(Event::default().event("error").data(json!({
                            "error":{"message":"native tokenizer's final text does not preserve the published stream prefix; repair its incremental decoder or retry with stream=false","type":"server_error","code":500},"index":index
                        }).to_string()))).await;
                        break;
                    };
                    value["content"] = json!(tail);
                    let value = response_fields.project(value);
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
    metadata: &NativeFinalMetadata,
    prompt_tokens: usize,
    result: &GenerateResult,
    return_tokens: bool,
    streaming: bool,
    index: usize,
) -> Result<Value, ApiError> {
    let stops = &metadata.stops;
    let model = &metadata.model;
    let stop_type = match result.finish_reason {
        FinishReason::MaxTokens
        | FinishReason::Length
        | FinishReason::IndentationLimit
        | FinishReason::PredictionTimeLimit => "limit",
        FinishReason::StopSequence { .. } => "word",
        FinishReason::EosToken | FinishReason::ToolCalls => "eos",
    };
    let mut value = json!({"content":text_before_stop(&result.text, stops),"tokens":if return_tokens && !streaming { result.token_ids.clone() } else {vec![]},
        "stop":true,"index":index,"model":model,"tokens_predicted":result.token_ids.len(),
        "tokens_evaluated":prompt_tokens,"tokens_cached":result.retained_cache_tokens,
        "stop_type":stop_type,"truncated":matches!(result.finish_reason, FinishReason::Length)});
    if let Some(sample) = result.timings {
        value["timings"] = crate::timings::render(sample);
    }
    let content = text_before_stop(&result.text, stops);
    // The upstream diagnostic survives a later time/EOS stop-type override.
    let raw_text = (metadata.minimum_indent > 0).then(|| {
        onnx_genai_engine::tokenization::decode_output(
            &metadata.tokenizer, &result.token_ids, &result.preserved_tokens,
        ).map_err(|error| ApiError::internal(format!("decode native stop-word diagnostic before indentation clipping: {error:#}; repair the package tokenizer")))
    }).transpose()?;
    let stopping_word = onnx_genai_engine::logits::first_text_stop(
        raw_text.as_deref().unwrap_or(&result.text),
        stops
            .iter()
            .enumerate()
            .map(|(index, stop)| (index, stop.as_str())),
        0,
    )
    .and_then(|(_, index)| stops.get(index))
    .map(String::as_str)
    .unwrap_or("");
    metadata.decorate(&mut value, content.contains('\n'), stopping_word);
    Ok(value)
}

fn utf8_prefix(bytes: &[u8]) -> &str {
    match std::str::from_utf8(bytes) {
        Ok(text) => text,
        Err(error) => std::str::from_utf8(&bytes[..error.valid_up_to()]).unwrap_or_default(),
    }
}

/// Resolve byte pieces once per request, preserving incomplete UTF-8 in bytes.
struct ProbabilityOutput {
    config: NativeProbabilities,
    pieces: Vec<Vec<u8>>,
}
impl ProbabilityOutput {
    fn new(
        config: NativeProbabilities,
        tokenizer: &onnx_genai_ort::Tokenizer,
    ) -> Result<Self, ApiError> {
        let ids = if config.top == 0 {
            Vec::new()
        } else {
            (0..tokenizer.inner().get_vocab_size(true)).map(|id| u32::try_from(id)
                .map_err(|_| ApiError::invalid_request("probability vocabulary exceeds u32 token IDs; select a smaller tokenizer")))
                .collect::<Result<Vec<_>,_>>()?
        };
        let pieces =
            onnx_genai_engine::tokenization::token_pieces(tokenizer, &ids).map_err(|error| {
                ApiError::invalid_request(format!(
                    "native probability token bytes cannot be resolved: {error:#}"
                ))
            })?;
        Ok(Self { config, pieces })
    }
    fn render(
        &self,
        probability: &onnx_genai_engine::TokenLogprob,
        text: &[u8],
    ) -> Result<Value, ApiError> {
        let score_key = if self.config.post { "prob" } else { "logprob" };
        let score = |logprob: f32| {
            if self.config.post {
                logprob.exp()
            } else if logprob.is_finite() && logprob.exp() != 0.0 {
                logprob
            } else {
                f32::MIN
            }
        };
        let mut top = Vec::new();
        for &(id, logprob) in probability.top.iter().take(self.config.top) {
            if self.config.post && logprob.exp() == 0.0 {
                continue;
            }
            let bytes = self.pieces.get(id as usize).ok_or_else(|| ApiError::internal(format!("probability token ID {id} is outside the resolved vocabulary; repair decoder output")))?;
            top.push(json!({"id":id,"token":utf8_prefix(bytes),"bytes":bytes,(score_key):score(logprob)}));
        }
        Ok(
            json!({"id":probability.token_id,"token":utf8_prefix(text),"bytes":text,
            (score_key):score(probability.logprob),
            (if self.config.post {"top_probs"} else {"top_logprobs"}):top}),
        )
    }
    fn decorate_result(
        &self,
        value: &mut Value,
        result: &GenerateResult,
        tokenizer: &onnx_genai_ort::Tokenizer,
        stops: &[String],
    ) -> Result<(), ApiError> {
        if self.config.top == 0 {
            return Ok(());
        }
        let values = result.logprobs.as_ref().filter(|values| values.len() == result.token_ids.len())
            .ok_or_else(|| ApiError::internal("native completion did not return one probability per generated token; select a decoder with probability reporting"))?;
        let texts = onnx_genai_engine::tokenization::token_texts(
            Some(tokenizer),
            &result.token_ids,
            &result.preserved_tokens,
        )
        .map_err(|error| {
            ApiError::internal(format!(
                "native probability text decoding failed: {error:#}"
            ))
        })?;
        let stop_tokens = match result.finish_reason {
            FinishReason::StopSequence { index } => match stops.get(index) {
                Some(stop) => MixedPrompt::Text(stop.clone())
                    .encode(
                        tokenizer,
                        onnx_genai_ort::TokenizationOptions {
                            add_special: false,
                            parse_special: true,
                        },
                    )?
                    .len(),
                None => 0,
            },
            _ => 0,
        };
        let visible = values.len().saturating_sub(stop_tokens);
        value["completion_probabilities"] = Value::Array(
            values[..visible]
                .iter()
                .zip(texts)
                .map(|(probability, text)| self.render(probability, text.as_bytes()))
                .collect::<Result<Vec<_>, _>>()?,
        );
        Ok(())
    }
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
    let lora = body.remove("lora");
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
    let mut mapped = json!({"model":model,"input":content});
    if let Some(lora) = lora {
        mapped["lora"] = lora;
    }
    let request: EmbeddingRequest = serde_json::from_value(mapped)
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
    let request: crate::types::AudioSpeechRequest = request;
    let format = match request.response_format.to_ascii_lowercase().as_str() {
        "wav" => onnx_genai_engine::AudioOutputFormat::Wav,
        "pcm" => onnx_genai_engine::AudioOutputFormat::PcmF32Le,
        "pcm16" => onnx_genai_engine::AudioOutputFormat::PcmS16Le,
        _ => {
            return Err(ApiError::invalid_request(
                "/tts response_format must be wav, pcm (float32 little-endian), or pcm16 (signed 16-bit little-endian)",
            ));
        }
    };
    super::speech::synthesize(state, request, format).await
}

#[cfg(test)]
mod probability_tests {
    use super::*;
    #[test]
    fn native_probability_bytes_preserve_partial_utf8_and_finite_zero_scores() {
        let output = ProbabilityOutput {
            config: NativeProbabilities {
                top: 1,
                post: false,
            },
            pieces: vec![vec![0xc3]],
        };
        let probability = onnx_genai_engine::TokenLogprob {
            token_id: 0,
            logprob: f32::NEG_INFINITY,
            top: vec![(0, f32::NEG_INFINITY)],
        };
        let value = output.render(&probability, &[0xc3]).unwrap();
        assert_eq!(value["token"], "");
        assert_eq!(value["bytes"], json!([195]));
        assert_eq!(value["logprob"], json!(f32::MIN));
        assert_eq!(value["top_logprobs"][0]["token"], "");
        assert_eq!(value["top_logprobs"][0]["bytes"], json!([195]));
        let underflow = onnx_genai_engine::TokenLogprob {
            logprob: -1000.0,
            top: vec![(0, -1000.0)],
            ..probability.clone()
        };
        assert_eq!(
            output.render(&underflow, b"x").unwrap()["logprob"],
            json!(f32::MIN)
        );
        let output = ProbabilityOutput {
            config: NativeProbabilities { top: 1, post: true },
            ..output
        };
        let value = output.render(&probability, &[0xc3]).unwrap();
        assert_eq!(value["prob"], 0.0);
        assert_eq!(value["top_probs"], json!([]));
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

#[cfg(test)]
mod response_field_tests {
    use super::*;
    #[test]
    fn paths_match_pinned_object_traversal_and_full_path_keys() {
        let source = json!({"a":{"b":{"c":1},"":2},"x":3,"list":[4],"":5,"tilde~1key":6});
        let fields = ResponseFields(
            vec![
                "a/b/c",
                "x",
                "a/b",
                "a/missing",
                "x/y",
                "list/0",
                "a/",
                "",
                "tilde~1key",
                "a/b/c",
            ]
            .into_iter()
            .map(str::to_string)
            .collect(),
        );
        assert_eq!(
            fields.project(source.clone()),
            json!({"a/b/c":1,"x":3,"a/b":{"c":1},"a/":2,"":5,"tilde~1key":6})
        );
        assert_eq!(ResponseFields::default().project(source.clone()), source);
    }
}

#[cfg(test)]
mod json_schema_tests {
    use super::*;
    #[test]
    fn native_grammar_presence_takes_precedence_over_legacy_schema() {
        for grammar in ["", "root ::= \"hello\""] {
            let request = parse(
                json!({"prompt":"hello","grammar":grammar,"json_schema":["ignored"]})
                    .as_object()
                    .unwrap()
                    .clone(),
                false,
            )
            .unwrap();
            assert_eq!(request.sampling.grammar.as_deref(), Some(grammar));
            assert!(request.sampling.response_schema().is_none());
            request.sampling.validate_grammar().unwrap();
        }
    }
    #[test]
    fn native_empty_schema_uses_shared_object_domain_without_grammar() {
        let request = parse(
            json!({"prompt":"hello","json_schema":{}})
                .as_object()
                .unwrap()
                .clone(),
            false,
        )
        .unwrap();
        assert_eq!(
            request.sampling.response_schema().unwrap().resolved_value(),
            json!({"type":"object"})
        );
    }
}

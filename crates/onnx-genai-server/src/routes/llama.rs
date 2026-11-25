//! llama.cpp's discovery and tokenizer protocol, backed by the admitted package.
use super::*;
use axum::extract::Query;
use serde_json::{Value, json};

#[derive(Default, Deserialize)]
pub(crate) struct ModelQuery {
    #[serde(default)]
    model: String,
    #[serde(default = "autoload_default")]
    autoload: bool,
}
fn autoload_default() -> bool {
    true
}

pub(super) async fn queried_model(
    state: &AppState,
    query: &ModelQuery,
) -> Result<Arc<ModelHandle>, ApiError> {
    if let Some(handle) = state
        .registry
        .lookup_loaded(&query.model)
        .map_err(map_registry_error)?
    {
        return Ok(handle);
    }
    if query.autoload {
        resolve_model(&state.registry, &query.model).await
    } else {
        Err(ApiError::not_found(format!(
            "model '{}' is not loaded; load it first or use autoload=true",
            query.model
        )))
    }
}

pub(crate) async fn props(
    State(state): State<AppState>,
    query: Result<Query<ModelQuery>, axum::extract::rejection::QueryRejection>,
) -> Result<Json<Value>, ApiError> {
    let Query(query) = query.map_err(|err| {
        ApiError::invalid_request(format!(
            "invalid /props query: {err}; use model=<model-id> and autoload=true or autoload=false"
        ))
    })?;
    if query.model.is_empty()
        && (state.model_manager.is_some()
            || state.registry.is_router().map_err(map_registry_error)?)
    {
        return Ok(Json(json!({
            "role": "router", "ui": true, "ui_settings": {},
            "nxrt_capabilities": {"tool_protocol_per_model": true, "reasoning_control": true, "resumable_streams": true, "server_tools": false, "model_management": state.model_manager.is_some(), "managed_model_formats": if state.model_manager.is_some() {vec!["nxrt_package"]} else {Vec::<&str>::new()}},
            "cors_proxy_enabled": false, "endpoint_slots": true,
            "endpoint_props": state.config.endpoint_props, "endpoint_metrics": cfg!(feature = "metrics"),
            "build_info": concat!("nxrt ", env!("CARGO_PKG_VERSION"))
        })));
    }
    let handle = queried_model(&state, &query).await?;
    let mut options = GenerateOptions::default();
    options.resolve_sampling_defaults(
        handle.generation_defaults.as_ref(),
        &SamplingOverrides::default(),
    );
    let modalities = handle.multimodal.as_ref();
    let mut properties = json!({
        "role": "model",
        "model_alias": handle.id,
        "default_generation_settings": {
            "n_ctx": handle.model_max_context,
            "params": {
                "n_predict": state.config.max_output_tokens,
                "t_max_predict_ms": -1,
                "seed": -1,
                "temperature": if options.greedy { 0.0 } else { options.temperature },
                "top_k": options.top_k, "top_p": options.top_p, "min_p": options.min_p,
                "typ_p": options.typical_p, "repeat_penalty": options.repetition_penalty,
                "repeat_last_n": crate::types::DEFAULT_REPEAT_LAST_N, "presence_penalty": options.presence_penalty,
                "frequency_penalty": options.frequency_penalty,
                "dry_multiplier": 0.0, "dry_base": 1.75, "dry_allowed_length": 2,
                "dry_penalty_last_n": -1, "dry_sequence_breakers": [],
                "mirostat": 0, "mirostat_tau": 5.0, "mirostat_eta": 0.1,
                "xtc_probability": 0.0, "xtc_threshold": 0.1,
                "dynatemp_range": 0.0, "dynatemp_exponent": 1.0,
                "top_n_sigma": -1.0, "adaptive_target": -1.0, "adaptive_decay": 0.9,
                "samplers": onnx_genai_engine::SamplerSequence::DEFAULT.iter().filter(|&&stage| stage != onnx_genai_engine::SamplerStage::TopA).collect::<Vec<_>>(),
                "backend_sampling": false, "timings_per_token": false,
                "sse_ping_interval":state.config.sse_ping_interval.seconds(),
                "stop": []
            }
        },
        "modalities": {
            "vision": modalities.is_some_and(|m| m.vision.is_some()),
            "audio": handle.engine.transcription.is_some() || modalities.is_some_and(|m| m.audio.is_some()), "video": false
        },
        "chat_template": handle.chat_template.as_ref().map_or("", |t| t.source()),
        "chat_template_caps": {},
        "endpoint_slots": handle.engine.native_slot.is_some(), "endpoint_props": state.config.endpoint_props,
        "endpoint_metrics": cfg!(feature = "metrics"),
        "ui": true, "ui_settings": {}, "is_sleeping": false,
        "cors_proxy_enabled": false,
        "build_info": concat!("nxrt ", env!("CARGO_PKG_VERSION")),
        "nxrt_capabilities": {
            "server_tools": false,
            "model_management": state.model_manager.is_some(), "managed_model_formats": if state.model_manager.is_some() {vec!["nxrt_package"]} else {Vec::<&str>::new()},
            "resumable_streams": true, "reasoning_control": true,
            "tools": handle.tool_protocol.is_some(), "custom_sampler_order": handle.engine.runtime_sampling,
            "lora": handle.engine.adapters.is_some(),
            "lora_embeddings": handle.engine.adapters.as_ref().is_some_and(|state|state.capabilities.embeddings),
            "lora_sessions": handle.engine.adapters.as_ref().is_some_and(|state|state.capabilities.sessions),
            "frame_transcription": handle.engine.transcription.is_some(),
            "generative_transcription": handle.multimodal.as_ref().and_then(|specs|specs.audio.as_ref()).is_some_and(|audio|audio.transcription().is_some()),
            "decision": handle.engine.decision.is_some(),
            "decision_image_limit": handle.engine.decision.as_ref().map(|contract| contract.media().iter().filter(|m| m.kind==onnx_genai_metadata::EncodedMediaKind::Image).count()),
            "decision_media_order": handle.engine.decision.as_ref().map(|contract|contract.media().iter().map(|m|m.kind).collect::<Vec<_>>()),
            "decision_audio_limit": handle.engine.decision.as_ref().map(|contract|contract.media().iter().filter(|m|m.kind==onnx_genai_metadata::EncodedMediaKind::Audio).count()),
            "decision_spoken_instructions": handle.engine.decision.as_ref().is_some_and(|contract|contract.spoken_input().is_some()),
            "reranking": handle.engine.reranking.is_some(),
            "reranking_image_limits": handle.engine.reranking.as_ref().map(|contract|json!({"query":contract.query_images().len(),"document":contract.document_images().len()}))
        }
    });
    let lora = match &handle.engine.adapters {
        Some(state) => state.list().map_err(crate::lora::api_error)?,
        None => Vec::new(),
    };
    properties["default_generation_settings"]["params"]["lora"] = json!(
        lora.into_iter()
            .map(|entry| crate::lora::Activation {
                id: entry.id,
                scale: entry.scale
            })
            .collect::<Vec<_>>()
    );
    if let Some(source) = handle
        .chat_template
        .as_ref()
        .and_then(|template| template.tool_source())
    {
        properties["chat_template_tool_use"] = json!(source);
    }
    if !handle.engine.runtime_sampling
        && let Some(params) = properties["default_generation_settings"]["params"].as_object_mut()
    {
        params.remove("samplers");
    }
    Ok(Json(properties))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TokenizeRequest {
    #[serde(default)]
    model: String,
    content: super::prompt::MixedPrompt,
    #[serde(default)]
    add_special: bool,
    #[serde(default)]
    parse_special: Option<bool>,
    #[serde(default)]
    with_pieces: bool,
}

pub(crate) async fn tokenize(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<TokenizeRequest>,
) -> Result<Json<Value>, ApiError> {
    let handle = resolve_model(&state.registry, &request.model).await?;
    let encoding = request.content.encode(
        &handle.tokenizer,
        onnx_genai_ort::TokenizationOptions {
            add_special: request.add_special,
            parse_special: request.parse_special.unwrap_or(true),
        },
    )?;
    let tokens: Vec<Value> = if request.with_pieces {
        let pieces = onnx_genai_engine::tokenization::token_pieces(&handle.tokenizer, &encoding)
            .map_err(|err| {
                ApiError::invalid_request(format!("token pieces could not be decoded: {err:#}"))
            })?;
        encoding
            .iter()
            .zip(pieces)
            .map(|(id, piece)| {
                let piece = match std::str::from_utf8(&piece) {
                    Ok(text) => json!(text),
                    Err(_) => json!(piece),
                };
                json!({"id":id,"piece":piece})
            })
            .collect()
    } else {
        encoding.iter().map(|id| json!(id)).collect()
    };
    Ok(Json(json!({"tokens": tokens})))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DetokenizeRequest {
    #[serde(default)]
    model: String,
    tokens: Vec<u32>,
}
pub(crate) async fn detokenize(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<DetokenizeRequest>,
) -> Result<Json<Value>, ApiError> {
    let handle = resolve_model(&state.registry, &request.model).await?;
    for id in &request.tokens {
        if handle.tokenizer.inner().id_to_token(*id).is_none() {
            return Err(ApiError::invalid_request(format!(
                "tokens contains id {id}, absent from model '{}' vocabulary; tokenize with the same model",
                handle.id
            )));
        }
    }
    let content = handle
        .tokenizer
        .decode_with_special_tokens(&request.tokens)
        .map_err(|err| ApiError::invalid_request(format!("tokens could not be decoded: {err}")))?;
    Ok(Json(json!({"content": content})))
}

pub(crate) async fn apply_template(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<ChatCompletionRequest>,
) -> Result<Json<Value>, ApiError> {
    let handle = resolve_model(&state.registry, &request.model).await?;
    validate_extensions(&request)?;
    let prompt = super::completions::render_model_prompt(
        &request,
        &handle,
        state.config.default_reasoning_effort,
    )
    .map_err(|err| ApiError::invalid_request(format!("apply-template failed: {err}")))?;
    Ok(Json(json!({"prompt": prompt})))
}

/// Controls that do not change inference may be safely accepted. Every other
/// unknown field is rejected, rather than reporting success with ignored settings.
pub(super) fn validate_extensions(request: &ChatCompletionRequest) -> Result<(), ApiError> {
    request
        .validate_grammar()
        .map_err(|error| ApiError::invalid_request(format!("{error:#}")))?;
    if request.repeat_last_n != -1 && usize::try_from(request.repeat_last_n).is_err() {
        return Err(ApiError::invalid_request(
            "repeat_last_n must be -1 for full history, 0 to disable all penalties, or a non-negative token count supported by this host",
        ));
    }
    ChatTemplate::validate_kwargs(&request.chat_template_kwargs)
        .map_err(|err| ApiError::invalid_request(err.to_string()))?;
    crate::reasoning_request::ReasoningRequest::extract(request).map_err(|error| {
        ApiError::invalid_request(format!("invalid reasoning controls: {error:#}"))
    })?;
    for (key, value) in &request.extensions {
        if crate::reasoning_request::ReasoningRequest::is_field(key) {
            continue;
        }
        let accepted = match key.as_str() {
            "reasoning_format" => value == "auto" || value == "deepseek",
            // These opt into optional progress/control facilities, not a different decoding algorithm.
            "return_progress" | "timings_per_token" => value.is_boolean(),
            "sse_ping_interval" => {
                let _ = crate::SsePingInterval::parse(value)?;
                true
            }
            "cache_prompt" | "ignore_eos" => value.is_boolean(),
            "logit_bias" => value.is_object() || value.is_array(),
            "dry_penalty_last_n" => value
                .as_i64()
                .is_some_and(|n| n == -1 || usize::try_from(n).is_ok()),
            "backend_sampling" => value == false,
            "lora" => value.is_array(),
            "id_slot" => matches!(value.as_i64(), Some(-1 | 0)),
            "n" => value == 1,
            "user" => value.is_string(),
            _ => false,
        };
        if !accepted {
            return Err(ApiError::invalid_request(format!(
                "generation request field '{key}' value {value} is unsupported; remove it or use a supported setting (see /props)"
            )));
        }
    }
    request
        .validate_continuation()
        .map_err(|error| ApiError::invalid_request(format!("{error:#}")))?;
    Ok(())
}

/// Upstream currently acknowledges this request without reading a mutation schema.
/// Keep it independent of model resolution and do not pretend to change defaults.
pub(crate) async fn post_props(State(state): State<AppState>) -> Result<Json<Value>, ApiError> {
    if !state.config.endpoint_props {
        return Err(ApiError {
            status: StatusCode::NOT_IMPLEMENTED,
            kind: "not_supported_error",
            message: "POST /props is disabled; start nxrt serve with --props to enable the upstream acknowledgement (no properties are changed)".into(),
            retry_after_secs: None,
        });
    }
    Ok(Json(json!({"success": true})))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ManageModelRequest {
    model: String,
    #[serde(default)]
    extra_args: Vec<String>,
}

pub(crate) async fn load_model(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<ManageModelRequest>,
) -> Result<Json<Value>, ApiError> {
    if !request.extra_args.is_empty() {
        return Err(ApiError::invalid_request(
            "extra_args cannot configure an in-process nxrt model; configure the model in --models-config instead",
        ));
    }
    if let Some(manager) = &state.model_manager
        && manager.has_transfer(&request.model)?
    {
        return Err(ApiError::unavailable(
            "model has an incomplete download; POST /models with its model id to resume it before loading",
        ));
    }
    let id = request.model;
    let _ = super::admin::admin_load_model(State(state.clone()), AxumPath(id.clone())).await?;
    if let Some(manager) = &state.model_manager {
        manager.loaded_changed(&id, true);
    }
    Ok(Json(json!({"success": true})))
}

pub(crate) async fn unload_model(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<ManageModelRequest>,
) -> Result<Json<Value>, ApiError> {
    if !request.extra_args.is_empty() {
        return Err(ApiError::invalid_request(
            "extra_args is not valid when unloading a model; remove the field",
        ));
    }
    if let Some(manager) = &state.model_manager
        && manager.pause(&request.model).await?
    {
        return Ok(Json(json!({"success":true})));
    }
    let id = request.model;
    super::admin::admin_unload_model(State(state.clone()), AxumPath(id.clone())).await?;
    if let Some(manager) = &state.model_manager {
        manager.loaded_changed(&id, false);
    }
    Ok(Json(json!({"success": true})))
}

async fn send_model_event(tx: &mpsc::Sender<Result<Event, Infallible>>, value: Value) -> bool {
    tx.send(Ok(Event::default().data(value.to_string())))
        .await
        .is_ok()
}
async fn poll_model_statuses(
    state: &AppState,
    last: &mut std::collections::BTreeMap<String, bool>,
    tx: &mpsc::Sender<Result<Event, Infallible>>,
) -> bool {
    let statuses = match state.registry.statuses() {
        Ok(statuses) => statuses,
        Err(error) => {
            tracing::warn!(%error,"model status stream stopped");
            return false;
        }
    };
    let mut present = std::collections::BTreeSet::new();
    for status in statuses {
        if status.transfer.is_some() {
            continue;
        }
        present.insert(status.id.clone());
        if last.insert(status.id.clone(), status.loaded) == Some(status.loaded) {
            continue;
        }
        if !send_model_event(tx,json!({"model":status.id,"event":"model_status","data":{"status":if status.loaded {"loaded"} else {"unloaded"}}})).await {return false;}
    }
    last.retain(|id, _| present.contains(id));
    true
}
pub(crate) async fn model_events(State(state): State<AppState>) -> Result<Response, ApiError> {
    use crate::model_management::ManagementEvent;
    state.registry.statuses().map_err(map_registry_error)?;
    let mut incoming = state
        .model_manager
        .as_ref()
        .map(|manager| manager.subscribe());
    let initial = state
        .model_manager
        .as_ref()
        .map(|manager| manager.initial_events())
        .transpose()?
        .unwrap_or_default();
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(1);
    tokio::spawn(async move {
        for value in initial {
            if !send_model_event(&tx, value).await {
                return;
            }
        }
        let mut last = std::collections::BTreeMap::new();
        let mut seen = std::collections::HashMap::new();
        let mut pending =
            std::collections::BTreeMap::<String, Arc<crate::model_management::TransferJob>>::new();
        let mut statuses = tokio::time::interval(std::time::Duration::from_secs(1));
        let mut progress = tokio::time::interval(std::time::Duration::from_millis(100));
        progress.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                _=tx.closed()=>return,
                _=statuses.tick()=>{if !poll_model_statuses(&state,&mut last,&tx).await {return;}},
                _=progress.tick(),if !pending.is_empty()=>{
                    for (_,job) in std::mem::take(&mut pending) {
                        if let Some(value)=ManagementEvent::Progress(job).materialize(&mut seen)
                            && !send_model_event(&tx,value).await {return;}
                    }
                },
                event=async {match incoming.as_mut() {Some(receiver)=>receiver.recv().await,None=>std::future::pending().await}}=>{
                    match event {
                        Ok(ManagementEvent::Progress(job))=>{pending.insert(job.id().into(),job);},
                        Ok(event@ManagementEvent::Protocol(_))=>{
                            // Flush this model's latest bytes before its lifecycle event.
                            if let ManagementEvent::Protocol(value)=&event && let Some(model)=value["model"].as_str() && let Some(job)=pending.remove(model)
                                && let Some(value)=ManagementEvent::Progress(job).materialize(&mut seen) && !send_model_event(&tx,value).await {return;}
                            if let Some(value)=event.materialize(&mut seen) && !send_model_event(&tx,value).await {return;}
                        },
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(_))=>{
                            pending.clear();seen.clear();
                            if !send_model_event(&tx,json!({"model":"*","event":"models_reload","data":{}})).await {return;}
                            if let Some(manager)=&state.model_manager {
                                match manager.initial_events() {Ok(values)=>for value in values {if !send_model_event(&tx,value).await {return;}},Err(error)=>{tracing::warn!(error=%error.message,"model status snapshot failed");return;}}
                            }
                        },
                        Err(tokio::sync::broadcast::error::RecvError::Closed)=>return,
                    }
                }
            }
        }
    });
    Ok(Sse::new(ReceiverStream::new(rx))
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response())
}

/// Count the exact prompt prepared for inference, including declared media, without spending a generation.
pub(crate) async fn input_tokens(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<ChatCompletionRequest>,
) -> Result<Json<Value>, ApiError> {
    Ok(Json(
        json!({"object":"response.input_tokens", "input_tokens": count_input_tokens(&state, &request).await?}),
    ))
}

pub(super) async fn count_input_tokens(
    state: &AppState,
    request: &ChatCompletionRequest,
) -> Result<usize, ApiError> {
    validate_extensions(request)?;
    let handle = resolve_model(&state.registry, &request.model).await?;
    super::completions::count_model_input_tokens(request, &handle, &state.config).await
}

/// An explicit slot selects a physical executor, independently of conversation IDs.
pub(super) fn validate_slot_selection(
    request: &ChatCompletionRequest,
    handle: &ModelHandle,
) -> Result<(), ApiError> {
    if request.extensions.get("id_slot").is_some_and(|v| v == 0)
        && handle.engine.native_slot.is_none()
    {
        return Err(ApiError::invalid_request(format!(
            "id_slot=0 is unavailable for model '{}'; select a serial Native executor, use id_slot=-1 for automatic selection, or omit id_slot",
            handle.id
        )));
    }
    Ok(())
}

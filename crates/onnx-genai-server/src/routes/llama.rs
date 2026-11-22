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

async fn queried_model(state: &AppState, query: &ModelQuery) -> Result<Arc<ModelHandle>, ApiError> {
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
        && (state.registry.statuses().map_err(map_registry_error)?.len() > 1
            || state
                .registry
                .lookup_loaded("")
                .map_err(map_registry_error)?
                .is_none())
    {
        return Ok(Json(json!({
            "role": "router", "ui": true, "ui_settings": {},
            "nxrt_capabilities": {"tool_protocol_per_model": true, "reasoning_control": false, "resumable_streams": true},
            "cors_proxy_enabled": false, "endpoint_slots": false,
            "endpoint_props": false, "endpoint_metrics": cfg!(feature = "metrics"),
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
    Ok(Json(json!({
        "role": "model",
        "model_alias": handle.id,
        "default_generation_settings": {
            "n_ctx": handle.model_max_context,
            "params": {
                "n_predict": state.config.max_output_tokens,
                "seed": -1,
                "temperature": if options.greedy { 0.0 } else { options.temperature },
                "top_k": options.top_k, "top_p": options.top_p, "min_p": options.min_p,
                "typ_p": options.typical_p, "repeat_penalty": options.repetition_penalty,
                "repeat_last_n": -1, "presence_penalty": options.presence_penalty,
                "frequency_penalty": options.frequency_penalty,
                "dry_multiplier": 0.0, "dry_base": 1.75, "dry_allowed_length": 2,
                "dry_penalty_last_n": -1, "dry_sequence_breakers": [],
                "mirostat": 0, "mirostat_tau": 5.0, "mirostat_eta": 0.1,
                "xtc_probability": 0.0, "xtc_threshold": 0.1,
                "dynatemp_range": 0.0, "dynatemp_exponent": 1.0,
                "backend_sampling": false, "timings_per_token": false,
                "stop": []
            }
        },
        "modalities": {
            "vision": modalities.is_some_and(|m| m.vision.is_some()),
            "audio": modalities.is_some_and(|m| m.audio.is_some()), "video": false
        },
        "chat_template": handle.chat_template.as_ref().map_or("", |t| t.source()),
        "chat_template_caps": {},
        "endpoint_slots": false, "endpoint_props": false,
        "endpoint_metrics": cfg!(feature = "metrics"),
        "ui": true, "ui_settings": {}, "is_sleeping": false,
        "cors_proxy_enabled": false,
        "build_info": concat!("nxrt ", env!("CARGO_PKG_VERSION")),
        "nxrt_capabilities": {
            "resumable_streams": true, "reasoning_control": false,
            "tools": handle.tool_protocol.is_some(), "custom_sampler_order": false,
            "reranking": handle.engine.reranking.is_some()
        }
    })))
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
    ChatTemplate::validate_kwargs(&request.chat_template_kwargs)
        .map_err(|err| ApiError::invalid_request(err.to_string()))?;
    for (key, value) in &request.extensions {
        let accepted = match key.as_str() {
            "repeat_last_n" => value
                .as_i64()
                .is_some_and(|n| n == -1 || usize::try_from(n).is_ok()),
            "reasoning_format" => value == "auto" || value == "deepseek",
            // These opt into optional progress/control facilities, not a different decoding algorithm.
            "reasoning_control" | "return_progress" | "timings_per_token" => value.is_boolean(),
            "sse_ping_interval" => value.as_f64().is_some_and(|n| n >= 0.0),
            "cache_prompt" | "ignore_eos" => value.is_boolean(),
            "logit_bias" => value.is_object() || value.is_array(),
            "dynatemp_range" => value.as_f64() == Some(0.0),
            "dynatemp_exponent" => value.as_f64() == Some(1.0),
            "dry_penalty_last_n" => value
                .as_i64()
                .is_some_and(|n| n == -1 || usize::try_from(n).is_ok()),
            "backend_sampling" => value == false,
            "n" => value == 1,
            "user" => value.is_string(),
            _ => false,
        };
        if !accepted {
            return Err(ApiError::invalid_request(format!(
                "chat request field '{key}' value {value} is unsupported; remove it or use a supported setting (see /props)"
            )));
        }
    }
    if request.continue_final_message {
        return Err(ApiError::invalid_request(
            "continue_final_message=true is not supported yet; send a new user turn instead",
        ));
    }
    Ok(())
}

pub(crate) async fn disabled_feature() -> Response {
    (StatusCode::FORBIDDEN, Json(json!({"error": {"type": "feature_disabled", "message": "this feature is unavailable in nxrt; use the capabilities reported by /props"}}))).into_response()
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
    let _ = super::admin::admin_load_model(State(state), AxumPath(request.model)).await?;
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
    super::admin::admin_unload_model(State(state), AxumPath(request.model)).await?;
    Ok(Json(json!({"success": true})))
}

pub(crate) async fn model_events(State(state): State<AppState>) -> Result<Response, ApiError> {
    // Use committed registry state, never invented load percentages. Polling also
    // sees model changes made through the existing nxrt administration API.
    state.registry.statuses().map_err(map_registry_error)?;
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(16);
    tokio::spawn(async move {
        let mut last = std::collections::BTreeMap::new();
        loop {
            let statuses = match state.registry.statuses() {
                Ok(statuses) => statuses,
                Err(error) => {
                    tracing::warn!(%error, "model status stream stopped");
                    return;
                }
            };
            for status in statuses {
                if last.insert(status.id.clone(), status.loaded) == Some(status.loaded) {
                    continue;
                }
                let event = Event::default().data(
                    json!({
                        "model": status.id, "event": "status_update",
                        "data": {"status": if status.loaded {"loaded"} else {"unloaded"}}
                    })
                    .to_string(),
                );
                if tx.send(Ok(event)).await.is_err() {
                    return;
                }
            }
            tokio::select! {
                _ = tx.closed() => return,
                _ = tokio::time::sleep(std::time::Duration::from_secs(1)) => {}
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

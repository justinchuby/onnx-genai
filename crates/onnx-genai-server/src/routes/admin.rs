use super::*;

pub(crate) async fn health(
    State(state): State<AppState>,
) -> Result<Json<HealthResponse>, ApiError> {
    Ok(Json(HealthResponse {
        status: "ok",
        model: state
            .registry
            .default_id()
            .map_err(map_registry_error)?
            .unwrap_or_default(),
    }))
}

pub(crate) async fn models(
    State(state): State<AppState>,
) -> Result<Json<ModelsResponse>, ApiError> {
    Ok(Json(ModelsResponse {
        object: "list",
        data: state
            .registry
            .ids()
            .map_err(map_registry_error)?
            .into_iter()
            .map(|id| ModelObject {
                id,
                object: "model",
                created: now_unix(),
                owned_by: "onnx-genai",
            })
            .collect(),
    }))
}

/// `GET /v1/status` — node-status contract for the cluster router (§34.8).
///
/// Real values: `queue_depth` (admission queue), `active_sessions` (session
/// registry), `healthy`, `node_id`. Everything else is a documented placeholder
/// because the underlying getter does not exist yet — see per-field comments.
pub(crate) async fn status(State(state): State<AppState>) -> Result<Json<NodeStatus>, ApiError> {
    let snapshot = crate::metrics::snapshot();
    Ok(Json(NodeStatus {
        // Node-level id from server config; independent of any loaded model.
        node_id: state.config.node_id.clone(),
        // Healthy while the node has a default model registered to serve.
        healthy: state
            .registry
            .default_id()
            .map_err(map_registry_error)?
            .is_some(),
        // KV page statistics: the engine does not yet expose paged-KV
        // introspection (see /v1/debug/kv), so these stay 0 until a getter exists.
        kv_usage: 0.0,      // not yet tracked
        kv_pages_used: 0,   // not yet tracked
        kv_pages_total: 0,  // not yet tracked
        kv_pages_shared: 0, // not yet tracked
        // Real: admission/backpressure queue depth (§36).
        queue_depth: u32::try_from(snapshot.pending_requests).unwrap_or(u32::MAX),
        // Real: aggregate active sessions across the node.
        active_sessions: u32::try_from(snapshot.active_sessions).unwrap_or(u32::MAX),
        paused_sessions: 0, // not yet tracked (no preemption/pause state exposed)
        tokens_per_second: 0.0, // not yet tracked (only cumulative token totals recorded)
        batch_utilization: 0.0, // not yet tracked (max batch size not surfaced to the server)
        // Per-session detail: session ids are real (redacted, since full ids are
        // bearer tokens — see session.rs). priority/kv_pages/state are not yet
        // tracked, so they carry documented placeholders rather than invented values.
        sessions: state
            .sessions
            .client_ids_redacted()
            .unwrap_or_default()
            .into_iter()
            .map(|id| SessionStatus {
                id,
                priority: "unknown".to_string(), // not yet tracked
                kv_pages: 0,                     // not yet tracked
                state: "unknown".to_string(),    // not yet tracked
            })
            .collect(),
        // System-prompt prefix hashes are not yet surfaced by the engine.
        prefix_hashes: Vec::new(),
    }))
}

pub(crate) async fn debug_config(
    State(state): State<AppState>,
) -> Result<Json<DebugConfigResponse>, ApiError> {
    let handle = state
        .registry
        .resolve("")
        .map_err(map_registry_error)?
        .ok_or_else(|| ApiError::internal("no model loaded"))?;
    Ok(Json(DebugConfigResponse {
        model_id: handle.id.clone(),
        pipeline: handle.pipeline,
        max_output_tokens: state.config.max_output_tokens,
        max_sessions: state.config.max_sessions,
        max_queue_depth: state.config.max_queue_depth,
        model_max_context: handle.model_max_context,
    }))
}

pub(crate) async fn debug_sessions(
    State(state): State<AppState>,
) -> Result<Json<DebugSessionsResponse>, ApiError> {
    let snapshot = crate::metrics::snapshot();
    let sessions = state
        .sessions
        .client_ids_redacted()
        .map_err(|err| ApiError::internal(format!("session registry failed: {err}")))?;
    Ok(Json(DebugSessionsResponse {
        active_sessions: snapshot.active_sessions,
        max_sessions: state.sessions.max_sessions(),
        sessions,
    }))
}

pub(crate) async fn debug_kv(
    State(state): State<AppState>,
) -> Result<Json<DebugKvResponse>, ApiError> {
    let handle = state
        .registry
        .resolve("")
        .map_err(map_registry_error)?
        .ok_or_else(|| ApiError::internal("no model loaded"))?;
    let snapshot = crate::metrics::snapshot();
    let batching = handle.engine.batching();
    let prefix_cache_hit_rate = if snapshot.prefix_cache_lookups == 0 {
        0.0
    } else {
        snapshot.prefix_cache_hits as f64 / snapshot.prefix_cache_lookups as f64
    };
    Ok(Json(DebugKvResponse {
        prefix_cache_hits: snapshot.prefix_cache_hits,
        prefix_cache_lookups: snapshot.prefix_cache_lookups,
        prefix_cache_hit_rate,
        active_batch_size: snapshot.current_batch_size,
        pending_queue_depth: snapshot.pending_requests,
        available_admission_slots: handle.engine.generation_capacity.available_permits(),
        rejected_requests: snapshot.rejections,
        engine_kv_introspection: "unavailable: engine does not yet expose KV page statistics",
        batch_supported: batching.supported,
        effective_max_batch: batching.effective_max_batch as u64,
    }))
}

pub(crate) async fn resources(
    State(state): State<AppState>,
) -> Result<Json<ResourcesResponse>, ApiError> {
    let handle = state
        .registry
        .resolve("")
        .map_err(map_registry_error)?
        .ok_or_else(|| ApiError::internal("no model loaded"))?;
    let snapshot = handle
        .engine
        .resource_snapshot()
        .await
        .map_err(|err| ApiError::internal(format!("resource snapshot failed: {err}")))?;
    Ok(Json(
        ResourcesResponse::from(snapshot)
            .with_batching(handle.engine.batching())
            .with_memory_strategy(&handle.engine.memory_strategy_plan()),
    ))
}

pub(crate) async fn admin_set_vram_limit(
    State(state): State<AppState>,
    Json(request): Json<SetVramLimitRequest>,
) -> Result<Response, ApiError> {
    let limit = parse_resource_limit(&request.limit)
        .map_err(|err| ApiError::bad_request(err.to_string()))?;
    let snapshot = state.registry.set_vram_limit(limit).await.map_err(|err| {
        if matches!(
            err.downcast_ref::<EngineGovernorError>(),
            Some(EngineGovernorError::RuntimeOverrideDisabled)
        ) {
            ApiError::forbidden(err.to_string())
        } else {
            ApiError::conflict(format!("resource override failed: {err}"))
        }
    })?;
    Ok(match snapshot {
        Some(snapshot) => {
            // Include the batching report for parity with `GET /v1/resources`.
            // Resolving the default handle is best-effort here: a vram override
            // succeeded, so a handle exists; if it cannot be resolved we still
            // return the governor snapshot rather than failing the override.
            let response = ResourcesResponse::from(snapshot);
            let response = match state.registry.resolve("") {
                Ok(Some(handle)) => response.with_batching(handle.engine.batching()),
                _ => response,
            };
            Json(response).into_response()
        }
        None => StatusCode::NO_CONTENT.into_response(),
    })
}

/// `GET /v1/debug/profile` — where the server's decode time went.
///
/// The aggregate counterpart to the Perfetto export: a trace answers "what
/// happened, when", which needs a viewer and a full timeline, while this answers
/// "which stages cost what", which is the question you ask a running server.
///
/// Stages are whatever the active decode path recorded — `ort.*` under ONNX
/// Runtime, `native.*` under the native runtime — so the shape of the answer
/// follows the backend without this endpoint knowing which one is in use.
/// Empty until `ONNX_GENAI_PROFILE` is set, and empty is reported as empty
/// rather than fabricated.
pub(crate) async fn debug_profile(
    State(state): State<AppState>,
) -> Result<Json<DebugProfileResponse>, ApiError> {
    let stages = onnx_genai_ort::profile::snapshot()
        .into_iter()
        .map(|stage| ProfileStage {
            stage: stage.stage,
            total_ms: stage.total_ns as f64 / 1e6,
            calls: stage.calls,
            us_per_call: if stage.calls > 0 {
                (stage.total_ns as f64 / 1e3) / stage.calls as f64
            } else {
                0.0
            },
        })
        .collect::<Vec<_>>();
    let memory_strategy_plans = state
        .registry
        .memory_strategy_plans()
        .map_err(map_registry_error)?
        .into_iter()
        .map(|(model_id, plan)| ModelMemoryStrategyPlan {
            model_id,
            plan: (*plan).clone(),
        })
        .collect();
    Ok(Json(DebugProfileResponse {
        collecting: onnx_genai_ort::profile::enabled(),
        note: "Stage totals accumulate across every request this process has served. Run with ONNX_GENAI_PROFILE=1 to collect them.",
        stages,
        memory_strategy_plans,
    }))
}

pub(crate) async fn debug_trace() -> Json<DebugTraceResponse> {
    let latest_trace_id = crate::metrics::latest_trace_id();
    let recorded_events = perfetto_event_count();
    let collecting = onnx_genai_ort::profile::tracing_enabled();
    Json(DebugTraceResponse {
        tracing_span: "http.request",
        latest_trace_id: format!("{latest_trace_id:016x}"),
        perfetto_export: PerfettoExportInfo {
            endpoint: PERFETTO_EXPORT_PATH,
            recorded_events,
            collecting,
            note: "GET the endpoint for a Chrome Trace Event Format document (open in https://ui.perfetto.dev). Run with ONNX_GENAI_TRACE set to collect decode spans.",
        },
        otlp_export: OTLP_EXPORT_STATUS,
        aggregate_profile: "/v1/debug/profile",
    })
}

/// `GET /v1/debug/trace/perfetto` — download the accumulated decode-timeline as
/// a Chrome Trace Event Format (Perfetto) JSON document.
///
/// The document combines the process-global engine profiler with native-runtime
/// and execution-provider spans, so a native `session_run` is no longer an
/// opaque block. When no spans have been recorded the response is a well-formed
/// but empty trace (`traceEvents: []`) — never fabricated events. The recorded
/// events carry only static stage names and timings (no session IDs, paths, or
/// user data), so no redaction is required.
pub(crate) async fn debug_trace_perfetto() -> Response {
    let document = perfetto_trace_document();
    let body = match serde_json::to_vec(&document) {
        Ok(body) => body,
        Err(err) => {
            return ApiError::internal(format!("failed to serialize Perfetto trace: {err}"))
                .into_response();
        }
    };
    (
        [
            (
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/json"),
            ),
            (
                header::CONTENT_DISPOSITION,
                HeaderValue::from_static("attachment; filename=\"onnx-genai-trace.json\""),
            ),
        ],
        body,
    )
        .into_response()
}

fn perfetto_trace_document() -> serde_json::Value {
    let mut document = onnx_genai_ort::profile::trace_document();
    let runtime_events = runtime_trace_events();
    if !runtime_events.is_empty()
        && let Some(events) = document
            .get_mut("traceEvents")
            .and_then(serde_json::Value::as_array_mut)
    {
        events.extend(runtime_events);
    }
    document
}

fn perfetto_event_count() -> usize {
    onnx_genai_ort::profile::trace_event_count() + runtime_trace_events().len()
}

/// Native runtime and execution-provider spans share the engine timeline when
/// the native backend is present. The server remains usable without it.
fn runtime_trace_events() -> Vec<serde_json::Value> {
    #[cfg(feature = "native-backend")]
    {
        onnx_genai::engine::runtime_trace::collected_events()
            .into_iter()
            .filter_map(|event| serde_json::to_value(event).ok())
            .collect()
    }
    #[cfg(not(feature = "native-backend"))]
    {
        Vec::new()
    }
}

/// `GET /v1/admin/models` — list every configured model with loaded/available
/// status and, for loaded models, the last-request timestamp.
pub(crate) async fn admin_list_models(
    State(state): State<AppState>,
) -> Result<Json<AdminModelsResponse>, ApiError> {
    let data = state
        .registry
        .statuses()
        .map_err(map_registry_error)?
        .into_iter()
        .map(|status| AdminModelObject {
            id: status.id,
            loaded: status.loaded,
            is_default: status.is_default,
            last_request_at: status.last_request_at,
        })
        .collect();
    Ok(Json(AdminModelsResponse {
        object: "list",
        data,
    }))
}

/// `POST /v1/admin/models/{id}/load` — load a configured model.  404 if the id is
/// unknown, 500 if the model fails to build.
pub(crate) async fn admin_load_model(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<AdminLoadResponse>, ApiError> {
    if !state
        .registry
        .contains_available(&id)
        .map_err(map_registry_error)?
    {
        return Err(ApiError::not_found(format!("model '{id}' not found")));
    }
    state
        .registry
        .load(&id)
        .await
        .map_err(|err| ApiError::internal(format!("failed to load model '{id}': {err}")))?;
    Ok(Json(AdminLoadResponse { id, loaded: true }))
}

/// `POST /v1/admin/models/{id}/warm` — run a small generation to initialize a
/// loaded model's lazy runtime allocations. Repeated calls are idempotent.
pub(crate) async fn admin_warmup_model(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<AdminWarmupResponse>, ApiError> {
    let registry = state.registry.clone();
    let warmup_id = id.clone();
    let duration = tokio::task::spawn_blocking(move || registry.warmup(&warmup_id))
        .await
        .map_err(|_| ApiError::internal("model warmup task panicked"))?
        .map_err(|err| match err {
            crate::registry::WarmupError::Registry(err) => map_registry_error(err),
            crate::registry::WarmupError::NotLoaded => {
                ApiError::not_found(format!("model '{id}' is not loaded"))
            }
            crate::registry::WarmupError::Failed(err) => {
                ApiError::internal(format!("failed to warm model '{id}': {err}"))
            }
        })?;
    Ok(Json(AdminWarmupResponse {
        id,
        warmed: true,
        duration_ms: duration.as_millis(),
    }))
}

/// `DELETE /v1/admin/models/{id}` — unload a loaded model.  The spec is kept
/// available so the model can be lazily reloaded on a later request.  404 if the
/// model is not currently loaded.
pub(crate) async fn admin_unload_model(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<StatusCode, ApiError> {
    state.registry.unload(&id).map_err(|err| {
        if err
            .downcast_ref::<crate::registry::RegistryError>()
            .is_some()
        {
            map_registry_error(crate::registry::RegistryError)
        } else {
            ApiError::not_found(format!("model '{id}' is not loaded"))
        }
    })?;
    Ok(StatusCode::NO_CONTENT)
}

#[cfg(feature = "metrics")]
/// `GET /metrics` -- Prometheus exposition.
///
/// The resource-governor family is read through `resource_snapshot()`, which
/// since AC31 is served off the driver thread. It previously queued behind a
/// `DriverCommand`, which is why this endpoint was measured at 51,010 ms under
/// load while `/v1/status` answered in 26 ms; it now answers in 71 ms worst of
/// 1,067 polls under sustained load.
///
/// When the governor cannot be read the family is absent, so this ALWAYS emits
/// `onnx_genai_resource_governor_available` to say which case a scrape is in.
/// Silently dropping the gauges made an unreadable governor look exactly like
/// a scrape gap -- the one shape operators read as "nothing to see".
pub(crate) async fn prometheus_metrics(
    State(state): State<AppState>,
) -> Result<Response, ApiError> {
    let mut output = crate::metrics::encode_prometheus();
    let handle = state.registry.any_loaded().map_err(map_registry_error)?;

    // Read the KV mirror before awaiting the governor snapshot. The mirror is a
    // handful of relaxed loads and never touches the driver thread, so it
    // answers even while a generation is inline -- which is exactly when the
    // pool is worth reading, and exactly when the governor command below may
    // have to wait.
    if let Some(handle) = handle.as_ref() {
        let telemetry = handle.engine.kv_telemetry();
        output.push_str(&crate::metrics::encode_kv_telemetry(
            telemetry.applicability(),
            &telemetry.snapshot(),
        ));
    }

    let snapshot = state
        .registry
        .aggregate_resource_snapshot()
        .await
        .ok()
        .flatten();
    match snapshot {
        Some(snapshot) => output.push_str(&crate::metrics::encode_resource_governor(&snapshot)),
        None => output.push_str(&crate::metrics::encode_resource_governor_unavailable()),
    }
    if let Ok(Some(metrics)) = state.registry.aggregate_growth_metrics() {
        output.push_str(&crate::metrics::encode_mapped_growth(&metrics));
    }
    Ok((
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        output,
    )
        .into_response())
}

impl From<GovernorSnapshot> for ResourcesResponse {
    fn from(snapshot: GovernorSnapshot) -> Self {
        Self {
            configured_limits: ConfiguredResourceLimits {
                vram: format_resource_limit(snapshot.configured_limits.vram_limit),
                host_ram: format_resource_limit(snapshot.configured_limits.host_ram_limit),
                disk_spill: snapshot
                    .configured_limits
                    .disk_spill_limit
                    .map(format_resource_limit),
            },
            resolved_limits: ResolvedResourceLimits {
                vram_bytes: snapshot.resolved_limits.vram_bytes,
                host_ram_bytes: snapshot.resolved_limits.host_ram_bytes,
                disk_spill_bytes: snapshot.resolved_limits.disk_spill_bytes,
            },
            derived_kv_budget: DerivedKvBudget {
                bytes: snapshot.derived_budget.kv_bytes,
                total_pages: snapshot.derived_budget.total_pages,
                max_total_tokens: snapshot.derived_budget.max_total_tokens,
                reserved_bytes: snapshot.derived_budget.reserved_bytes,
            },
            vram: ResourceTier::from(snapshot.vram),
            host_ram: ResourceTier::from(snapshot.host_ram),
            disk_spill: snapshot.disk_spill.map(ResourceTier::from),
            batching: None,
            memory_strategy: None,
        }
    }
}

impl ResourcesResponse {
    /// Attach the model handle's resolved batching capability so `/v1/resources`
    /// reports `supported` / `effective_max_batch` directly (issue #750).
    fn with_batching(mut self, batching: &crate::driver::BatchingReport) -> Self {
        self.batching = Some(BatchingInfo {
            supported: batching.supported,
            requested_max_batch: batching.requested_max_batch as u64,
            effective_max_batch: batching.effective_max_batch as u64,
            reason: batching.reason.clone(),
        });
        self
    }

    /// Attach the model's resolved memory strategy so `/v1/resources` reports the
    /// chosen strategy, offload state, managed no-spill VMM state, and resolved
    /// budget directly, making the #755 managed VMM default observable.
    fn with_memory_strategy(mut self, plan: &onnx_genai_engine::MemoryStrategyPlan) -> Self {
        let application = plan.runtime_application();
        self.memory_strategy = Some(MemoryStrategyInfo {
            strategy: format!("{:?}", plan.strategy),
            weight_offload_enabled: application.weight_offload_enabled,
            managed_no_spill: application.managed_no_spill,
            auto_enabled: application.auto_enabled_from_vram_limit,
            resolved_device_budget_bytes: plan.resolved_device_budget_bytes,
            managed_limit_bytes: application.managed_limit_bytes,
            fits_resolved_device_budget: plan.fits_resolved_device_budget,
        });
        self
    }
}

impl From<onnx_genai::scheduler::TierSnapshot> for ResourceTier {
    fn from(snapshot: onnx_genai::scheduler::TierSnapshot) -> Self {
        Self {
            used: snapshot.used,
            limit: snapshot.limit,
            headroom: snapshot.headroom,
        }
    }
}

fn format_resource_limit(limit: ResourceLimit) -> String {
    match limit {
        ResourceLimit::Bytes(bytes) => bytes.to_string(),
        ResourceLimit::Fraction(fraction) => fraction.to_string(),
        ResourceLimit::Auto => "auto".to_string(),
    }
}

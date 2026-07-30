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

/// Directory mtime in epoch seconds, or `None` when it cannot be determined.
///
/// Returns `None` rather than a fallback timestamp: an unknown creation time is
/// omitted, never guessed. Any guess here would be indistinguishable from a
/// real one and would be wrong in a way no caller could detect.
fn directory_mtime_secs(path: &std::path::Path) -> Option<u64> {
    let modified = std::fs::metadata(path).ok()?.modified().ok()?;
    let since_epoch = modified.duration_since(std::time::UNIX_EPOCH).ok()?;
    Some(since_epoch.as_secs())
}

pub(crate) async fn models(
    State(state): State<AppState>,
) -> Result<Json<ModelsResponse>, ApiError> {
    Ok(Json(ModelsResponse {
        object: "list",
        // Built from `statuses()`, not `ids()`. `ids()` lists only *loaded*
        // models, so a configured-but-lazy model was absent from the only
        // ungated model endpoint -- and absence reads as "this model does not
        // exist", which is a stronger and falser claim than "not loaded".
        data: state
            .registry
            .statuses()
            .map_err(map_registry_error)?
            .into_iter()
            .map(|status| ModelObject {
                id: status.id,
                object: "model",
                created: directory_mtime_secs(&status.path),
                owned_by: "onnx-genai",
                loaded: status.loaded,
                is_default: status.is_default,
            })
            .collect(),
    }))
}

/// Fraction of the assemblable batch that is currently generating.
///
/// Clamped to 1.0 rather than allowed to exceed it. The numerator counts
/// in-flight generations across the whole node, so with several models loaded
/// -- each with its own driver and its own batch -- the sum can legitimately
/// exceed any single batch's capacity. Reporting 240% occupancy would be
/// arithmetically faithful and completely unreadable as a gauge; the honest
/// rendering of "more work in flight than one batch holds" is "full".
pub(crate) fn batch_utilization(in_flight: u64, capacity: usize) -> f32 {
    if capacity == 0 {
        return 0.0;
    }
    (in_flight as f32 / capacity as f32).min(1.0)
}

pub(crate) async fn status(State(state): State<AppState>) -> Result<Json<NodeStatus>, ApiError> {
    let snapshot = crate::metrics::snapshot();
    let default_id = state.registry.default_id().map_err(map_registry_error)?;

    // Every field this node cannot honestly measure, with the reason. Built
    // alongside the omissions so a field can never be dropped silently.
    let mut unavailable = BTreeMap::new();

    // Batch occupancy is read from the driver that owns the batch, not from
    // the process-wide HTTP gauge. `None` until a driver loop has published,
    // so an idle batch stays distinguishable from a build that never batched.
    //
    // The capacity is narrowed by admission per the standing ruling: a client
    // handed the raw row count would render "3 of 4" on a server whose real
    // ceiling is one. Admission also bounds the numerator, so the pair stays
    // commensurable after the narrowing.
    // Resolved from the same handle as the occupancy so the driver named here
    // is the driver that produced the numbers beside it.
    let handle = state.registry.resolve("").ok().flatten();
    let batch_driver = handle
        .as_ref()
        .map(|handle| handle.engine.batch_driver().clone());
    let occupancy = state
        .registry
        .resolve("")
        .ok()
        .flatten()
        .and_then(|handle| handle.engine.batch_telemetry().snapshot())
        .map(|mut o| {
            let admission = state.config.max_queue_depth as u64;
            o.capacity = o.capacity.min(admission);
            o.active = o.active.min(o.capacity);
            o
        });
    if occupancy.is_none() {
        for field in [
            "batch_utilization",
            "batch_in_flight",
            "batch_capacity",
            "batch_queued",
        ] {
            unavailable.insert(
                field,
                FieldUnavailable::pending(
                    "no generation has run on this node yet, so the driver has \
                     published no batch occupancy; a zero here would be \
                     indistinguishable from an idle batch",
                ),
            );
        }
    }

    // The driver selection is a property of a loaded engine. With no model
    // resolved there is no driver, and naming one would describe a machine
    // that is not running.
    if batch_driver.is_none() {
        for field in ["batch_driver", "batch_driver_detail"] {
            unavailable.insert(
                field,
                FieldUnavailable::pending(
                    "no model is loaded on this node, so no decode driver has \
                     been selected; naming one would describe an engine that \
                     does not exist",
                ),
            );
        }
    }

    // Paged-KV introspection is per-engine and this handler holds no engine
    // reference. The four KV fields travel together: reporting any one of them
    // while the others are absent would imply the pool is partially known.
    //
    // `kv_pages_total` is the trap here. It reads a real structure, so it
    // survives any "is this hardcoded?" audit -- but on a continuous-batching
    // model it describes a pool the decoder never uses, and it is *non-zero*,
    // which makes it invisible to every check built to catch fabricated zeros.
    // A non-zero value is not evidence that a mechanism is in play.
    const KV_DETAIL: &str = "paged-KV telemetry is served by the KV block-table \
                             endpoint; this node-status contract is model-agnostic \
                             and carries no engine reference";
    for field in [
        "kv_usage",
        "kv_pages_used",
        "kv_pages_total",
        "kv_pages_shared",
    ] {
        unavailable.insert(field, FieldUnavailable::unavailable(KV_DETAIL));
    }

    // Preemption exists nowhere in the driver: generation runs inline to
    // completion, so no session is ever in a paused state to count. This is
    // not-applicable rather than unavailable -- the number isn't missing, the
    // concept doesn't apply to this scheduler.
    unavailable.insert(
        "paused_sessions",
        FieldUnavailable::not_applicable(
            "the driver runs generations to completion without preemption, \
             so no session can be paused",
        ),
    );

    // Only a cumulative token total is recorded. Dividing it by uptime yields a
    // lifetime average, which on a live panel would be read as the current
    // rate -- lowest exactly when the node has been idle longest, and slowest
    // to move exactly when throughput changes most. Per-request latency
    // histograms on /metrics are the measured throughput signal.
    unavailable.insert(
        "tokens_per_second",
        FieldUnavailable::unavailable(
            "only a cumulative token count is recorded; a lifetime average \
             would misreport as a current rate -- see the latency histograms \
             on /metrics",
        ),
    );

    unavailable.insert(
        "prefix_hashes",
        FieldUnavailable::unavailable(
            "the engine does not surface system-prompt prefix hashes; an empty \
             list would claim none are cached",
        ),
    );

    Ok(Json(NodeStatus {
        // Node-level id from server config; independent of any loaded model.
        node_id: state.config.node_id.clone(),
        // Echoed so a captured payload is self-describing. Under the two-server
        // demo topology attribution is by origin, but a saved response should
        // not depend on remembering which port produced it.
        model_id: default_id.clone(),
        // Healthy while the node has a default model registered to serve.
        healthy: default_id.is_some(),
        kv_usage: None,
        kv_pages_used: None,
        kv_pages_total: None,
        kv_pages_shared: None,
        // Real: admission/backpressure queue depth (§36).
        queue_depth: u32::try_from(snapshot.pending_requests).unwrap_or(u32::MAX),
        // Real: aggregate active sessions across the node.
        active_sessions: u32::try_from(snapshot.active_sessions).unwrap_or(u32::MAX),
        paused_sessions: None,
        tokens_per_second: None,
        // Both terms come from the driver's own batch, read as one reading.
        //
        // They used to come from different populations: the numerator was
        // `metrics::REGISTRY.batch_size`, a per-HTTP-generation gauge bounded
        // by `max_queue_depth` (256 on the demo), while the denominator was
        // `effective_batch_capacity()` (4). Six concurrent requests therefore
        // published `batch_in_flight = 6, batch_capacity = 4`. The ratio was
        // clamped to 1.0 and looked correct; the raw pair renders "6 of 4".
        batch_utilization: occupancy.map(|o| batch_utilization(o.active, o.capacity as usize)),
        // The raw numerator, so the client never has to invert a saturating
        // ratio to recover a count the server already knows.
        batch_in_flight: occupancy.map(|o| u32::try_from(o.active).unwrap_or(u32::MAX)),
        // The denominator itself, so the client never hardcodes a capacity no
        // endpoint confirms. Same reading as the ratio above.
        batch_capacity: occupancy.map(|o| u32::try_from(o.capacity).unwrap_or(u32::MAX)),
        batch_queued: occupancy.map(|o| u32::try_from(o.queued).unwrap_or(u32::MAX)),
        // Named from the SAME handle the occupancy was read from, not from
        // `default_id`, so the id cannot drift from the numbers it labels if
        // the default changes between the two resolves.
        batch_model_id: handle.as_ref().map(|handle| handle.id.clone()),
        batch_driver: batch_driver.as_ref().map(|driver| driver.kind()),
        batch_driver_detail: batch_driver.as_ref().map(|driver| driver.explain()),
        // Session ids are real, and redacted because full ids are bearer
        // tokens (see session.rs). The per-session detail fields are omitted
        // rather than filled with "unknown".
        sessions: state
            .sessions
            .client_ids_redacted()
            .unwrap_or_default()
            .into_iter()
            .map(|id| SessionStatus {
                id,
                priority: None,
                kv_pages: None,
                state: None,
            })
            .collect(),
        prefix_hashes: None,
        unavailable,
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
    let generation_prefix_reuse_rate =
        crate::metrics::defined_ratio(snapshot.prefix_cache_hits, snapshot.prefix_cache_lookups);
    Ok(Json(DebugKvResponse {
        generations_with_prefix_reuse: snapshot.prefix_cache_hits,
        generations_completed: snapshot.prefix_cache_lookups,
        generation_prefix_reuse_rate,
        prefix_tokens_reused: snapshot.prefix_cache_hit_tokens,
        active_batch_size: snapshot.current_batch_size,
        pending_queue_depth: snapshot.pending_requests,
        available_admission_slots: handle.engine.generation_capacity.available_permits(),
        rejected_requests: snapshot.rejections,
        block_table_endpoint: "/v1/debug/kv/blocks",
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
    Ok(Json(snapshot.into()))
}

pub(crate) async fn admin_set_vram_limit(
    State(state): State<AppState>,
    Json(request): Json<SetVramLimitRequest>,
) -> Result<Json<ResourcesResponse>, ApiError> {
    let limit = parse_resource_limit(&request.limit)
        .map_err(|err| ApiError::bad_request(err.to_string()))?;
    let handle = state
        .registry
        .resolve("")
        .map_err(map_registry_error)?
        .ok_or_else(|| ApiError::internal("no model loaded"))?;
    let snapshot = handle
        .engine
        .set_vram_limit(limit)
        .await
        .map_err(|err| ApiError::internal(format!("resource override failed: {err}")))?
        .map_err(|err| match err {
            EngineGovernorError::RuntimeOverrideDisabled => ApiError::forbidden(err.to_string()),
            EngineGovernorError::Resource(_) => ApiError::conflict(err.to_string()),
        })?;
    Ok(Json(snapshot.into()))
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
pub(crate) async fn debug_profile() -> Json<DebugProfileResponse> {
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
    Json(DebugProfileResponse {
        collecting: onnx_genai_ort::profile::enabled(),
        note: "Stage totals accumulate across every request this process has served. Run with ONNX_GENAI_PROFILE=1 to collect them.",
        stages,
    })
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
/// `onnxgenai_resource_governor_available` to say which case a scrape is in.
/// Silently dropping the gauges made an unreadable governor look exactly like
/// a scrape gap -- the one shape operators read as "nothing to see".
pub(crate) async fn prometheus_metrics(
    State(state): State<AppState>,
) -> Result<Response, ApiError> {
    let mut output = crate::metrics::encode_prometheus();
    let snapshot = match state.registry.resolve("").map_err(map_registry_error)? {
        Some(handle) => handle.engine.resource_snapshot().await.ok(),
        None => None,
    };
    match snapshot {
        Some(snapshot) => output.push_str(&crate::metrics::encode_resource_governor(&snapshot)),
        None => output.push_str(&crate::metrics::encode_resource_governor_unavailable()),
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
        }
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

/// Per-page KV block table, read live during generation.
///
/// Served from the lock-free telemetry mirror rather than the page table
/// itself: the page table is owned by the driver thread and mutably borrowed
/// for the whole of a generation, and paged-KV behaviour is only worth watching
/// *during* one.
pub(crate) async fn debug_kv_blocks(
    State(state): State<AppState>,
    Query(query): Query<BlockWindowQuery>,
) -> Result<Json<BlockTableResponse>, ApiError> {
    let handle = state
        .registry
        .resolve("")
        .map_err(map_registry_error)?
        .ok_or_else(|| ApiError::internal("no model loaded"))?;
    let model_id = Some(handle.id.clone());
    let telemetry = handle.engine.kv_telemetry();

    // Enforced server-side rather than left to the client. Continuous batching
    // and paged KV are mutually exclusive, and on the batching path every
    // counter here is a truthful read of a pool the decoder never consults --
    // including the non-zero capacity, which is exactly what makes it
    // dangerous. A client cannot be expected to know that.
    match telemetry.applicability() {
        onnx_genai_engine::Applicability::NotApplicable => {
            // The wording comes from the reason the driver recorded, not from a
            // constant here. More than one fact reaches this state, and a
            // hardcoded sentence would name continuous batching even on a model
            // whose KV cache simply cannot page.
            let detail = telemetry
                .not_applicable_reason()
                .map_or("this model does not use paged KV", |reason| reason.detail());
            return Ok(Json(BlockTableResponse::not_applicable(model_id, detail)));
        }
        // The driver picks the decode path asynchronously at startup, so a poll
        // can genuinely arrive before the answer exists. Saying "pending" costs
        // one frame; saying "not applicable" would state the opposite of the
        // truth with total confidence on a paged model.
        onnx_genai_engine::Applicability::Unknown => {
            return Ok(Json(BlockTableResponse::pending(
                model_id,
                "the driver has not finished selecting a decode path yet",
            )));
        }
        onnx_genai_engine::Applicability::Applicable => {}
    }

    let count = BlockTableResponse::window_size(query.count);
    // `None` means no per-page mirror exists at all, which is NOT an empty
    // window. Rendering it as `live` with zero pages would draw a legitimately
    // empty grid for a pool we cannot see, and an empty grid is what a fully
    // unused pool also looks like.
    let Some(states) = telemetry.block_window(query.start, count) else {
        return Ok(Json(BlockTableResponse::pending(
            model_id,
            "the per-page block mirror has not been attached yet",
        )));
    };
    Ok(Json(BlockTableResponse::live(
        model_id,
        telemetry.snapshot(),
        query.start,
        telemetry.mirrored_block_capacity(),
        telemetry.pool_block_count(),
        states,
    )))
}

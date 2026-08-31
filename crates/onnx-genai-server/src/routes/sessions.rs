use super::*;

pub(crate) async fn create_session(
    State(state): State<AppState>,
) -> Result<Json<SessionResponse>, ApiError> {
    let handle = state
        .registry
        .resolve("")
        .map_err(map_registry_error)?
        .ok_or_else(|| ApiError::internal("no model loaded"))?;
    let client_id = state
        .sessions
        .next_client_id()
        .map_err(|err| ApiError::internal(format!("session id generation failed: {err}")))?;
    let placement = handle
        .engine
        .create_session()
        .await
        .map_err(session_create_failure)?;
    // Model-qualified from the moment it exists: the binding names the engine
    // that issued this placement, so every later decision about it — a turn, an
    // eviction, a close — reaches that engine and no other.
    let binding = handle.engine.binding(placement);

    let evicted = match state.sessions.insert(client_id.clone(), binding.clone()) {
        Ok(evicted) => evicted,
        Err(error) => {
            // The registry refused, so nothing names this session. Close it
            // rather than leaving the engine holding a conversation no client
            // can reach — its lease is uncontested, because nothing else has
            // ever seen this placement.
            if let Ok(lease) = state.sessions.acquire(binding, &client_id) {
                close_leased_session(&state.registry, lease).await?;
            }
            return Err(session_registry_failure(error));
        }
    };
    close_evicted_session(&state.registry, evicted).await?;

    Ok(Json(SessionResponse {
        id: client_id,
        object: "session",
    }))
}

pub(crate) async fn fork_session(
    State(state): State<AppState>,
    AxumPath(source_id): AxumPath<String>,
    ApiJson(request): ApiJson<SessionForkRequest>,
) -> Result<Json<SessionResponse>, ApiError> {
    let binding = state
        .sessions
        .get(&source_id)
        .map_err(session_registry_failure)?
        .ok_or_else(|| ApiError::not_found(format!("session {source_id} not found")))?;
    let lease = state
        .sessions
        .acquire(binding.clone(), &source_id)
        .map_err(package_execution_failure)?;
    let model = binding.model().as_str().to_string();
    let handle = state
        .registry
        .resolve(&model)
        .map_err(map_registry_error)?
        .ok_or_else(|| {
            ApiError::unavailable(format!(
                "session {source_id} belongs to unloaded model '{model}'"
            ))
        })?;
    let child_id = state.sessions.next_client_id().map_err(|error| {
        ApiError::internal(format!("forked session id generation failed: {error}"))
    })?;
    let placement = handle
        .engine
        .fork_session(
            lease,
            onnx_genai_engine::SessionPosition::new(request.position),
        )
        .await
        .map_err(session_fork_failure)?;
    let child_binding = handle.engine.binding(placement);
    let evicted = match state
        .sessions
        .insert(child_id.clone(), child_binding.clone())
    {
        Ok(evicted) => evicted,
        Err(error) => {
            if let Ok(lease) = state.sessions.acquire(child_binding, &child_id) {
                close_leased_session(&state.registry, lease).await?;
            }
            return Err(session_registry_failure(error));
        }
    };
    close_evicted_session(&state.registry, evicted).await?;
    Ok(Json(SessionResponse {
        id: child_id,
        object: "session",
    }))
}

/// Close a session.
///
/// Closing is a mutation of the conversation — the sharpest one there is — so it
/// takes the same exclusive lease a turn does, and takes it *before* the binding
/// is removed. Three things follow, and all three are the point:
///
/// - A close that races a live turn is refused with the same 409 an overlapping
///   turn gets, rather than destroying the state that turn is writing and
///   failing it for a reason its caller never asked about.
/// - The binding is removed while the lease is held, so the id cannot be
///   rebound to a new conversation until the engine has answered and the guard
///   has been released.
/// - The lease, the removal, and the close all name the *same* binding, because
///   the registry decides all three under one lock. Reading the binding here and
///   removing it afterwards would let a rebind slip between, and the close would
///   then destroy a conversation this request never leased.
///
/// The model comes from the binding, never from the default: a session belongs
/// to the engine that opened it, and `DELETE` on a two-model server routinely
/// names a session the default model has never heard of.
pub(crate) async fn delete_session(
    State(state): State<AppState>,
    AxumPath(client_id): AxumPath<String>,
) -> Result<StatusCode, ApiError> {
    let lease = state
        .sessions
        .take_for_close(&client_id)
        .map_err(|error| session_close_failure(&client_id, error))?;
    close_leased_session(&state.registry, lease).await?;

    Ok(StatusCode::NO_CONTENT)
}

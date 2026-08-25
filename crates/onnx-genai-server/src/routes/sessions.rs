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

    let evicted = state
        .sessions
        .insert(client_id.clone(), placement, handle.engine.session_leases())
        .map_err(|err| ApiError::internal(format!("session registry failed: {err}")))?;
    close_evicted_session(&handle.engine, evicted).await?;

    Ok(Json(SessionResponse {
        id: client_id,
        object: "session",
    }))
}

/// Close a session.
///
/// Closing is a mutation of the conversation — the sharpest one there is — so it
/// takes the same exclusive lease a turn does, and takes it *before* the binding
/// is removed. Two things follow, and both are the point:
///
/// - A close that races a live turn is refused with the same 409 an overlapping
///   turn gets, rather than destroying the state that turn is writing and
///   failing it for a reason its caller never asked about.
/// - The binding is removed while the lease is held, so the id cannot be
///   rebound to a new conversation until the engine has answered and the guard
///   has been released.
pub(crate) async fn delete_session(
    State(state): State<AppState>,
    AxumPath(client_id): AxumPath<String>,
) -> Result<StatusCode, ApiError> {
    let handle = state
        .registry
        .resolve("")
        .map_err(map_registry_error)?
        .ok_or_else(|| ApiError::internal("no model loaded"))?;
    let placement = state
        .sessions
        .get(&client_id)
        .map_err(|err| ApiError::internal(format!("session registry failed: {err}")))?
        .ok_or_else(|| ApiError::not_found(format!("session {client_id} not found")))?;
    let lease = handle
        .engine
        .acquire_session_lease(placement, &client_id)
        .map_err(package_capability_failure)?;
    // Another delete that got there first has already removed the binding and
    // closed the session; this one has nothing left to close.
    state
        .sessions
        .remove(&client_id)
        .map_err(|err| ApiError::internal(format!("session registry failed: {err}")))?
        .ok_or_else(|| ApiError::not_found(format!("session {client_id} not found")))?;

    handle
        .engine
        .close_session(lease)
        .await
        .map_err(|err| ApiError::internal(format!("session close failed: {err}")))?;

    Ok(StatusCode::NO_CONTENT)
}

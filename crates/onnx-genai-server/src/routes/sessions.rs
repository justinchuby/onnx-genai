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
    let engine_session_id = handle
        .engine
        .create_session()
        .await
        .map_err(|err| ApiError::internal(format!("session create failed: {err}")))?;

    let evicted = state
        .sessions
        .insert(client_id.clone(), engine_session_id)
        .map_err(|err| ApiError::internal(format!("session registry failed: {err}")))?;
    close_evicted_session(&handle.engine, evicted).await?;

    Ok(Json(SessionResponse {
        id: client_id,
        object: "session",
    }))
}

pub(crate) async fn delete_session(
    State(state): State<AppState>,
    AxumPath(client_id): AxumPath<String>,
) -> Result<StatusCode, ApiError> {
    let handle = state
        .registry
        .resolve("")
        .map_err(map_registry_error)?
        .ok_or_else(|| ApiError::internal("no model loaded"))?;
    let engine_session_id = state
        .sessions
        .remove(&client_id)
        .map_err(|err| ApiError::internal(format!("session registry failed: {err}")))?
        .ok_or_else(|| ApiError::not_found(format!("session {client_id} not found")))?;

    handle
        .engine
        .close_session(engine_session_id)
        .await
        .map_err(|err| ApiError::internal(format!("session close failed: {err}")))?;

    Ok(StatusCode::NO_CONTENT)
}

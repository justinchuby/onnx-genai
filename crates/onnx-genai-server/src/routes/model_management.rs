//! llama-server cache download/delete protocol. Files stay in an operator-owned root.
use super::*;
use axum::extract::Query;
use serde_json::{Value, json};
#[derive(Deserialize)]
pub(crate) struct DownloadRequest {
    #[serde(default)]
    model: String,
    #[serde(default = "main_revision")]
    revision: String,
    #[serde(default)]
    sidecars: Vec<String>,
}
fn main_revision() -> String {
    "main".into()
}
#[derive(Deserialize)]
pub(crate) struct DeleteQuery {
    model: String,
    sidecar: Option<String>,
}
fn manager(state: &AppState) -> Result<&crate::model_management::ModelManager, ApiError> {
    state.model_manager.as_ref().ok_or_else(||ApiError{status:StatusCode::FORBIDDEN,kind:"feature_disabled",message:"model downloads and deletion are disabled; start nxrt serve with --model-cache-dir <operator-owned-directory> to enable them".into(),retry_after_secs:None})
}
fn unsupported_sidecar() -> ApiError {
    ApiError{status:StatusCode::NOT_IMPLEMENTED,kind:"not_supported_error",message:"selective GGUF sidecars require an imported sidecar contract; use an nxrt package with its declared artifacts, without sidecars, or select a compatible source".into(),retry_after_secs:None}
}
pub(crate) async fn download(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<DownloadRequest>,
) -> Result<Json<Value>, ApiError> {
    let manager = manager(&state)?;
    if !request.sidecars.is_empty() {
        return Err(unsupported_sidecar());
    }
    let model = crate::model_hub::HubModel::new(request.model, request.revision)
        .map_err(|error| ApiError::invalid_request(format!("{error:#}")))?;
    manager.start(model).await?;
    Ok(Json(json!({"success":true})))
}
pub(crate) async fn delete(
    State(state): State<AppState>,
    query: Result<Query<DeleteQuery>, axum::extract::rejection::QueryRejection>,
) -> Result<Json<Value>, ApiError> {
    let manager = manager(&state)?;
    let Query(query) = query.map_err(|error| {
        ApiError::invalid_request(format!(
            "invalid model deletion query: {error}; supply ?model=<model-id>"
        ))
    })?;
    if query.model.trim().is_empty() {
        return Err(ApiError::invalid_request(
            "model must be a non-empty query string; supply ?model=<model-id>",
        ));
    }
    if query
        .sidecar
        .as_ref()
        .is_some_and(|value| !value.is_empty())
    {
        return Err(unsupported_sidecar());
    }
    manager.delete(&query.model).await?;
    Ok(Json(json!({"success":true})))
}

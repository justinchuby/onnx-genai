//! llama.cpp's model-qualified LoRA catalog and default scales.
use super::*;
use axum::extract::{Query, rejection::QueryRejection};
use serde_json::{Value, json};

async fn model(
    state: &AppState,
    query: Result<Query<super::llama::ModelQuery>, QueryRejection>,
) -> Result<Arc<ModelHandle>, ApiError> {
    let Query(query) = query.map_err(|e| {
        ApiError::invalid_request(format!(
            "invalid /lora-adapters query: {e}; use model=<id>&autoload=false"
        ))
    })?;
    let handle = super::llama::queried_model(state, &query).await?;
    handle.engine.workers.primary_sender().map_err(|error| {
        let mut failure=ApiError::unavailable(format!("LoRA model owner is unavailable: {error}; reload the model after resolving the worker failure"));
        failure.retry_after_secs=Some(1);
        failure
    })?;
    Ok(handle)
}
pub(crate) async fn list(
    State(state): State<AppState>,
    query: Result<Query<super::llama::ModelQuery>, QueryRejection>,
) -> Result<Json<Value>, ApiError> {
    let handle = model(&state, query).await?;
    Ok(Json(match &handle.engine.adapters {
        Some(adapters) => json!(adapters.list().map_err(crate::lora::api_error)?),
        None => json!([]),
    }))
}
pub(crate) async fn set(
    State(state): State<AppState>,
    query: Result<Query<super::llama::ModelQuery>, QueryRejection>,
    ApiJson(values): ApiJson<Vec<crate::lora::Activation>>,
) -> Result<Json<Value>, ApiError> {
    let handle = model(&state, query).await?;
    match &handle.engine.adapters {
        Some(adapters) => adapters.set(&values).map_err(crate::lora::api_error)?,
        None if !values.is_empty() => {
            return Err(ApiError::invalid_request(format!(
                "model '{}' has no admitted adapters; GET /lora-adapters lists the available IDs",
                handle.id
            )));
        }
        None => {}
    }
    Ok(Json(json!({"success":true})))
}

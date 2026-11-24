//! llama.cpp slots backed by an admitted single Native execution lane.
use super::*;
use crate::{
    slot_storage::{SlotFilename, SlotStorageInput, SlotStorageQuota},
    slots::SlotOperation,
};
use axum::extract::{
    Query,
    rejection::{PathRejection, QueryRejection},
};
use serde_json::{Value, json};

#[derive(Default, Deserialize)]
pub(crate) struct SlotQuery {
    #[serde(flatten)]
    model: super::llama::ModelQuery,
    #[serde(default)]
    fail_on_no_slot: String,
    #[serde(default)]
    action: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SlotBody {
    filename: Option<String>,
}
fn unsupported(message: impl Into<String>) -> ApiError {
    ApiError {
        status: StatusCode::NOT_IMPLEMENTED,
        message: message.into(),
        kind: "not_supported_error",
        retry_after_secs: None,
    }
}
fn operation_error(error: anyhow::Error) -> ApiError {
    let message = format!("{error:#}");
    let overloaded = matches!(
        DriverFailure::from_engine_error(&error).kind,
        DriverFailureKind::MemoryOverload
    ) || error.chain().any(|e| {
        e.is::<SlotStorageQuota>()
            || e.is::<tokio::sync::TryAcquireError>()
            || e.is::<crate::worker::WorkerUnavailable>()
    });
    if overloaded {
        let mut error = ApiError::unavailable(message);
        error.retry_after_secs = Some(1);
        return error;
    }
    if error.chain().any(|e| e.is::<SlotStorageInput>()) {
        return ApiError::invalid_request(message);
    }
    for cause in error.chain() {
        if let Some(io) = cause.downcast_ref::<std::io::Error>() {
            if io.kind() == std::io::ErrorKind::StorageFull {
                let mut error = ApiError::unavailable(message);
                error.retry_after_secs = Some(1);
                return error;
            }
            if io.kind() == std::io::ErrorKind::NotFound {
                return ApiError::not_found(message);
            }
            if io.kind() == std::io::ErrorKind::PermissionDenied {
                return ApiError::forbidden(message);
            }
        }
    }
    #[cfg(feature = "native-backend")]
    if error.is::<onnx_genai_engine::NativeStateFileReadError>() {
        return ApiError::invalid_request(message);
    }
    ApiError::internal(message)
}
fn query(value: Result<Query<SlotQuery>, QueryRejection>) -> Result<SlotQuery, ApiError> {
    value.map(|Query(q)|q).map_err(|e|ApiError::invalid_request(format!("invalid /slots query: {e}; use model=<id>, autoload=true|false, fail_on_no_slot=1, or action=save|restore|erase")))
}
async fn model(state: &AppState, query: &SlotQuery) -> Result<Arc<ModelHandle>, ApiError> {
    let handle = super::llama::queried_model(state, &query.model).await?;
    if handle.engine.native_slot.is_none() {
        return Err(unsupported(
            "this model executor has no admitted serial Native slot; select a single-sequence Native decoder",
        ));
    }
    Ok(handle)
}

pub(crate) async fn list(
    State(state): State<AppState>,
    input: Result<Query<SlotQuery>, QueryRejection>,
) -> Result<Json<Value>, ApiError> {
    let query = query(input)?;
    let model = model(&state, &query).await?;
    let slot = model.engine.slot_status().map_err(operation_error)?;
    if !query.fail_on_no_slot.is_empty() && slot.is_processing {
        return Err(ApiError::unavailable(
            "no slot available; retry after the active slot task finishes",
        ));
    }
    Ok(Json(
        json!([{"id":0,"is_processing":slot.is_processing,"n_past":slot.n_past,"n_ctx":slot.n_ctx.unwrap_or(0)}]),
    ))
}

pub(crate) async fn action(
    State(state): State<AppState>,
    slot: Result<AxumPath<u32>, PathRejection>,
    input: Result<Query<SlotQuery>, QueryRejection>,
    body: Result<Option<Json<SlotBody>>, JsonRejection>,
) -> Result<Json<Value>, ApiError> {
    let AxumPath(slot) = slot.map_err(|e| {
        ApiError::invalid_request(format!(
            "invalid slot id: {e}; list /slots for available IDs"
        ))
    })?;
    let query = query(input)?;
    if !matches!(query.action.as_str(), "save" | "restore" | "erase") {
        return Err(ApiError::invalid_request(
            "slot action must be save, restore or erase",
        ));
    }
    let storage = state.slot_storage.as_ref().ok_or_else(|| {
        unsupported(
            "slot state actions are disabled; start the server with --slot-save-path <directory>",
        )
    })?;
    let model = model(&state, &query).await?;
    if slot != 0 {
        return Err(ApiError::invalid_request(format!(
            "invalid slot id {slot}; this executor exposes slot 0, see GET /slots"
        )));
    }
    let status = model.engine.slot_status().map_err(operation_error)?;
    if !status.state_files {
        return Err(unsupported(
            "this slot has no admitted host-state persistence participant; select a host-state Native decoder",
        ));
    }
    let body=body.map_err(|e|ApiError::invalid_request(format!("invalid slot action JSON: {e}; provide {{\"filename\":\"cache.bin\"}} for save/restore")))?.map(|Json(b)|b);
    let operation = if query.action == "erase" {
        if body.as_ref().is_some_and(|b| b.filename.is_some()) {
            return Err(ApiError::invalid_request(
                "slot erase does not take filename; omit the body or send {}",
            ));
        }
        SlotOperation::Erase
    } else {
        let filename = body.and_then(|b| b.filename).ok_or_else(|| {
            ApiError::invalid_request(
                "slot save/restore requires JSON filename; provide {\"filename\":\"cache.bin\"}",
            )
        })?;
        let filename =
            SlotFilename::new(filename).map_err(|e| ApiError::invalid_request(format!("{e:#}")))?;
        if query.action == "save" {
            SlotOperation::Save {
                storage: Arc::clone(storage),
                filename,
            }
        } else {
            SlotOperation::Restore {
                storage: Arc::clone(storage),
                filename,
            }
        }
    };
    Ok(Json(
        model
            .engine
            .slot_operation(operation)
            .await
            .map_err(operation_error)?,
    ))
}

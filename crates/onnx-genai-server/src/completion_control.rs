//! Live completion controls. Registrations are owned by generation consumers,
//! including replay producers, rather than by individual HTTP readers.
use crate::{
    AppState,
    routes::{ApiError, ApiJson},
};
use axum::{Json, extract::State};
use onnx_genai_engine::reasoning::ReasoningControl;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

#[derive(Clone)]
pub(crate) struct CompletionControls(Arc<Inner>);
struct Inner {
    entries: Mutex<HashMap<String, Option<ReasoningControl>>>,
    limit: usize,
}
pub(crate) struct Registration {
    registry: CompletionControls,
    id: String,
}
impl CompletionControls {
    pub(crate) fn new(limit: usize) -> Self {
        Self(Arc::new(Inner {
            entries: Mutex::new(HashMap::new()),
            limit,
        }))
    }
    pub(crate) fn register_request(
        &self,
        id: &str,
        request: &crate::types::ChatCompletionRequest,
        options: &onnx_genai_engine::GenerateOptions,
    ) -> Result<Registration, ApiError> {
        let controls =
            crate::reasoning_request::ReasoningRequest::extract(request).map_err(|error| {
                ApiError::invalid_request(format!("invalid reasoning controls: {error:#}"))
            })?;
        let control = if controls.reasoning_control {
            options
                .reasoning_budget
                .as_ref()
                .map(|budget| budget.control())
        } else {
            None
        };
        self.register(id, control)
    }
    pub(crate) fn register(
        &self,
        id: &str,
        control: Option<ReasoningControl>,
    ) -> Result<Registration, ApiError> {
        let mut entries = self.0.entries.lock().map_err(|_| {
            ApiError::internal("live completion control registry lock poisoned; restart the server")
        })?;
        if entries.len() >= self.0.limit {
            return Err(ApiError::too_many_requests(format!(
                "live completion registry reached its {} request limit; retry after an active completion ends",
                self.0.limit
            )));
        }
        if entries.contains_key(id) {
            return Err(ApiError::internal(
                "completion ID collision; retry this request to generate a new identifier",
            ));
        }
        entries.insert(id.to_owned(), control);
        Ok(Registration {
            registry: self.clone(),
            id: id.to_owned(),
        })
    }
    fn end_reasoning(&self, id: &str) -> Result<Value, ApiError> {
        let entries = self.0.entries.lock().map_err(|_| {
            ApiError::internal("live completion control registry lock poisoned; restart the server")
        })?;
        match entries.get(id) {
            None => Ok(json!({"success":false,"message":"no active completion for this id"})),
            Some(None) => Ok(
                json!({"success":false,"message":"reasoning control not enabled for this completion"}),
            ),
            Some(Some(control)) => {
                control.request_end().map_err(|error| {
                    ApiError::internal(format!(
                        "request reasoning_end for completion {id:?}: {error:#}"
                    ))
                })?;
                Ok(json!({"success":true}))
            }
        }
    }
}
impl Drop for Registration {
    fn drop(&mut self) {
        // Cleanup remains possible after poison; admission still reports poison.
        let mut entries = self
            .registry
            .0
            .entries
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        entries.remove(&self.id);
    }
}

pub(crate) fn new_id(prefix: &str) -> Result<String, ApiError> {
    let mut bytes = [0_u8; 16];
    getrandom::fill(&mut bytes).map_err(|error| ApiError::internal(format!("generate completion ID from operating-system randomness: {error}; restore the host random source and retry")))?;
    use std::fmt::Write;
    let mut id = String::with_capacity(prefix.len() + 32);
    id.push_str(prefix);
    for byte in bytes {
        write!(&mut id, "{byte:02x}").expect("writing to String is infallible");
    }
    Ok(id)
}

#[derive(Deserialize)]
pub(crate) struct ControlRequest {
    id: String,
    action: String,
}
pub(crate) async fn control(
    State(state): State<AppState>,
    ApiJson(request): ApiJson<ControlRequest>,
) -> Result<Json<Value>, ApiError> {
    if request.action != "reasoning_end" {
        return Err(ApiError::invalid_request(format!(
            "unsupported control action {:?}; use action=reasoning_end",
            request.action
        )));
    }
    if request.id.is_empty() {
        return Err(ApiError::invalid_request(
            "control id must be a nonempty live completion identifier",
        ));
    }
    state
        .completion_controls
        .end_reasoning(&request.id)
        .map(Json)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn completion_control_lifetime_capacity_and_isolation() {
        let registry = CompletionControls::new(2);
        let first = registry
            .register("first", Some(ReasoningControl::default()))
            .unwrap();
        let second = registry.register("second", None).unwrap();
        assert!(registry.register("third", None).is_err());
        assert_eq!(
            registry.end_reasoning("first").unwrap(),
            json!({"success":true})
        );
        assert_eq!(
            registry.end_reasoning("first").unwrap(),
            json!({"success":true})
        );
        assert_eq!(
            registry.end_reasoning("second").unwrap(),
            json!({"success":false,"message":"reasoning control not enabled for this completion"})
        );
        assert_eq!(registry.end_reasoning("unknown").unwrap()["success"], false);
        drop(first);
        assert_eq!(
            registry.end_reasoning("first").unwrap()["message"],
            "no active completion for this id"
        );
        let third = registry.register("third", None).unwrap();
        assert!(registry.register("third", None).is_err());
        drop((second, third));
        assert!(registry.0.entries.lock().unwrap().is_empty());
    }
    #[test]
    fn completion_control_ids_are_unique_within_one_second() {
        let ids = (0..1024)
            .map(|_| new_id("chatcmpl-").unwrap())
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(ids.len(), 1024);
        assert!(
            ids.iter()
                .all(|id| id.starts_with("chatcmpl-") && id.len() == 41)
        );
    }
}

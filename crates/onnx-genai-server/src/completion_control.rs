//! Live completion controls. Registrations are owned by generation consumers,
//! including replay producers, rather than by individual HTTP readers.
use crate::{
    AppState,
    registry::ModelHandle,
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
    entries: Mutex<HashMap<String, ActiveCompletion>>,
    limit: usize,
}
/// A live control belongs to an admitted model, never a request-supplied name.
pub(crate) struct ActiveCompletion {
    model_id: String,
    control: Option<ReasoningControl>,
}
impl ActiveCompletion {
    #[cfg(test)]
    pub(crate) fn unarmed_for_test(model_id: &str) -> Self {
        Self {
            model_id: model_id.into(),
            control: None,
        }
    }
    pub(crate) fn new(model: &ModelHandle, control: Option<ReasoningControl>) -> Self {
        Self {
            model_id: model.id.clone(),
            control,
        }
    }
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
        model: &ModelHandle,
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
        self.register(id, ActiveCompletion::new(model, control))
    }
    pub(crate) fn register(
        &self,
        id: &str,
        completion: ActiveCompletion,
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
        entries.insert(id.to_owned(), completion);
        Ok(Registration {
            registry: self.clone(),
            id: id.to_owned(),
        })
    }
    fn end_reasoning(&self, id: &str, model: Option<&str>) -> Result<Value, ApiError> {
        let entries = self.0.entries.lock().map_err(|_| {
            ApiError::internal("live completion control registry lock poisoned; restart the server")
        })?;
        match entries
            .get(id)
            .filter(|entry| model.is_none_or(|model| entry.model_id == model))
        {
            None => Ok(json!({"success":false,"message":"no active completion for this id"})),
            Some(ActiveCompletion { control: None, .. }) => Ok(
                json!({"success":false,"message":"reasoning control not enabled for this completion"}),
            ),
            Some(ActiveCompletion {
                control: Some(control),
                ..
            }) => {
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
    model: Option<String>,
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
    let is_router = state.registry.is_router().map_err(|error| {
        ApiError::internal(format!("inspect model catalogue for reasoning control: {error}; restart the server to restore the registry"))
    })?;
    let model = if is_router {
        let model = request.model.as_deref().filter(|model| !model.trim().is_empty())
            .ok_or_else(|| ApiError::invalid_request("control model is required in router mode; send the configured model id with id and action=reasoning_end"))?;
        if !state.registry.contains_available(model).map_err(|error| ApiError::internal(format!("resolve reasoning control model {model:?}: {error}; restart the server to restore the registry")))? {
            return Err(ApiError::not_found(format!("reasoning control model {model:?} is not configured; choose an id from GET /v1/models")));
        }
        Some(model)
    } else {
        None
    };
    state
        .completion_controls
        .end_reasoning(&request.id, model)
        .map(Json)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn completion_control_lifetime_capacity_and_isolation() {
        let registry = CompletionControls::new(2);
        let first = registry
            .register(
                "first",
                ActiveCompletion {
                    model_id: "model-a".into(),
                    control: Some(ReasoningControl::default()),
                },
            )
            .unwrap();
        let second = registry
            .register(
                "second",
                ActiveCompletion {
                    model_id: "model-b".into(),
                    control: None,
                },
            )
            .unwrap();
        assert!(
            registry
                .register(
                    "third",
                    ActiveCompletion {
                        model_id: "model-b".into(),
                        control: None
                    }
                )
                .is_err()
        );
        assert_eq!(
            registry.end_reasoning("first", None).unwrap(),
            json!({"success":true})
        );
        assert_eq!(
            registry.end_reasoning("first", None).unwrap(),
            json!({"success":true})
        );
        assert_eq!(
            registry.end_reasoning("second", None).unwrap(),
            json!({"success":false,"message":"reasoning control not enabled for this completion"})
        );
        assert_eq!(
            registry.end_reasoning("unknown", None).unwrap()["success"],
            false
        );
        drop(first);
        assert_eq!(
            registry.end_reasoning("first", None).unwrap()["message"],
            "no active completion for this id"
        );
        let third = registry
            .register(
                "third",
                ActiveCompletion {
                    model_id: "model-b".into(),
                    control: None,
                },
            )
            .unwrap();
        assert!(
            registry
                .register(
                    "third",
                    ActiveCompletion {
                        model_id: "model-b".into(),
                        control: None
                    }
                )
                .is_err()
        );
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
    #[tokio::test]
    async fn canonical_control_routes_model_scopes_without_loading_idle_models()
    -> anyhow::Result<()> {
        use onnx_genai_engine::{
            GenerateOptions, GeneratePrompt, GenerateRequest, ToolCallPolicy,
            logits::LogitBias,
            reasoning::{ReasoningBudget, ReasoningBudgetSpec},
        };
        use tower::ServiceExt;
        let fixture =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm");
        let specs = ["model-a", "model-b", "idle-model"].map(|id| crate::ModelSpec {
            package_variant: None,
            id: id.into(),
            path: fixture.clone(),
            eager: id != "idle-model",
            warmup: false,
        });
        let state =
            crate::AppState::load_from_specs(specs.to_vec(), crate::ServerConfig::default())?;
        let handle = state.registry.lookup_loaded("model-a")?.unwrap();
        let budget = ReasoningBudget::resolve(
            &handle.tokenizer,
            ReasoningBudgetSpec {
                start_sequences: Vec::new(),
                initially_active: true,
                end_sequences: vec![vec![23]],
                forced_tokens: vec![23],
                prefix_tokens: Vec::new(),
                max_tokens: None,
            },
        )?;
        let _owner = state
            .completion_controls
            .register(
                "chatcmpl-a",
                ActiveCompletion::new(&handle, Some(budget.control())),
            )
            .unwrap();
        let router = crate::app(state.clone());
        async fn control(router: axum::Router, body: Value) -> (axum::http::StatusCode, Value) {
            let response = router
                .oneshot(
                    axum::http::Request::post("/v1/chat/completions/control")
                        .header("content-type", "application/json")
                        .body(axum::body::Body::from(body.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            let status = response.status();
            let body = axum::body::to_bytes(response.into_body(), 64 * 1024)
                .await
                .unwrap();
            (status, serde_json::from_slice(&body).unwrap())
        }
        for model in [Value::Null, json!(""), json!(" ")] {
            let (status, body) = control(
                router.clone(),
                json!({"id":"chatcmpl-a","action":"reasoning_end","model":model}),
            )
            .await;
            assert_eq!(status, axum::http::StatusCode::BAD_REQUEST, "{body}");
            assert!(
                body["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("model is required in router mode")
            );
        }
        let (status, body) = control(
            router.clone(),
            json!({"id":"chatcmpl-a","action":"reasoning_end","model":"unknown-model"}),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::NOT_FOUND);
        let error = body["error"]["message"].as_str().unwrap();
        assert!(error.contains("unknown-model") && error.contains("/v1/models"));
        for model in ["model-b", "idle-model"] {
            let (status, body) = control(
                router.clone(),
                json!({"id":"chatcmpl-a","action":"reasoning_end","model":model}),
            )
            .await;
            assert_eq!(status, axum::http::StatusCode::OK);
            assert_eq!(body["success"], false);
        }
        assert!(
            state.registry.lookup_loaded("idle-model")?.is_none(),
            "control must not autoload an idle model"
        );
        async fn sample(handle: &ModelHandle, budget: ReasoningBudget) -> Vec<u32> {
            let mut options = GenerateOptions {
                greedy: true,
                max_new_tokens: 2,
                stop_on_eos: false,
                reasoning_budget: Some(budget),
                ..Default::default()
            };
            options.logit_bias.insert(4, LogitBias::Add(1000.0));
            let generation = handle
                .engine
                .submit_generation(
                    None,
                    GenerateRequest {
                        prompt: GeneratePrompt::TokenIds(vec![4]),
                        options,
                    },
                    ToolCallPolicy::Disabled,
                    None,
                )
                .await
                .unwrap();
            generation.admission.await.unwrap().unwrap();
            let mut events = generation.events;
            loop {
                match events.recv().await.unwrap() {
                    crate::driver::DriverEvent::Finished(result) => return result.token_ids,
                    crate::driver::DriverEvent::Error(error) => panic!("{error:?}"),
                    _ => {}
                }
            }
        }
        // Wrong-model controls must not mutate the real model-a sampler signal.
        assert_eq!(sample(&handle, budget.clone()).await, [4, 4]);
        let (status, body) = control(
            router.clone(),
            json!({"id":"chatcmpl-a","action":"reasoning_end","model":"model-a"}),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::OK);
        assert_eq!(body["success"], true);
        assert_eq!(sample(&handle, budget).await, [23, 4]);
        let response = router
            .clone()
            .oneshot(axum::http::Request::post("/control").body(axum::body::Body::empty())?)
            .await?;
        assert_eq!(response.status(), axum::http::StatusCode::NOT_FOUND);
        state.registry.poison_for_test();
        let (status, body) = control(
            router,
            json!({"id":"chatcmpl-a","action":"reasoning_end","model":"model-a"}),
        )
        .await;
        assert_eq!(status, axum::http::StatusCode::INTERNAL_SERVER_ERROR);
        let error = body["error"]["message"].as_str().unwrap();
        assert!(
            error.contains("inspect model catalogue")
                && error.contains("registry lock poisoned")
                && error.contains("restart the server")
        );
        Ok(())
    }
}

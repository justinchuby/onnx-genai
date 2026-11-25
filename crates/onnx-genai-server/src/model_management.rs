//! Background model-transfer lifecycle and llama-server management events.
use crate::{
    model_cache::{ModelCache, PendingTransfer, TransferOutcome},
    model_hub::{HubClient, HubModel, HubProgress},
    registry::ModelRegistry,
    routes::ApiError,
};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashMap},
    sync::{Arc, Mutex, Weak},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore, broadcast, watch};

#[derive(Clone)]
pub(crate) struct ModelManager(Arc<Inner>);
struct Inner {
    cache: ModelCache,
    registry: ModelRegistry,
    hub: HubClient,
    permits: Arc<Semaphore>,

    operations: tokio::sync::Mutex<HashMap<String, Weak<tokio::sync::Mutex<()>>>>,
    events: broadcast::Sender<ManagementEvent>,
}
pub(crate) struct TransferJob {
    request: HubModel,
    state: Mutex<JobState>,
    progress: Mutex<ProgressState>,
}
#[derive(Default)]
struct ProgressState {
    files: BTreeMap<String, FileProgress>,
    version: u64,
    done: u64,
    total: u64,
}
#[derive(Clone)]
pub(crate) enum ManagementEvent {
    Protocol(Value),
    Progress(Arc<TransferJob>),
}
impl ManagementEvent {
    pub(crate) fn materialize(&self, seen: &mut HashMap<String, u64>) -> Option<Value> {
        match self {
            Self::Protocol(value) => {
                if (matches!(
                    value["event"].as_str(),
                    Some("model_remove" | "download_finished")
                ) || (value["event"] == "model_status"
                    && value["data"]["status"] == "downloading"))
                    && let Some(model) = value["model"].as_str()
                {
                    seen.remove(model);
                }
                Some(value.clone())
            }
            Self::Progress(job) => {
                let progress = lock(&job.progress);
                if seen.get(job.id()) == Some(&progress.version) {
                    return None;
                }
                seen.insert(job.id().into(), progress.version);
                Some(
                    json!({"model":job.id(),"event":"download_progress","data":{"progress":progress.files,"done":progress.done,"total":progress.total}}),
                )
            }
        }
    }
}
#[derive(Clone, serde::Serialize)]
struct FileProgress {
    done: u64,
    total: u64,
}
enum JobState {
    Running {
        pause: watch::Sender<bool>,
        done: watch::Receiver<bool>,
    },
    Incomplete {
        transfer: PendingTransfer,
        error: Option<String>,
    },
    Transition,
    Removed,
    RecoveryRequired(String),
}
#[derive(Debug, Clone)]
pub(crate) struct TransferSnapshot {
    pub(crate) id: String,
    pub(crate) running: bool,
    pub(crate) error: Option<String>,
}
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
fn internal(error: impl std::fmt::Display) -> ApiError {
    ApiError::internal(error.to_string())
}
impl ModelManager {
    pub(crate) fn new(
        cache: ModelCache,
        registry: ModelRegistry,
        pending: Vec<PendingTransfer>,
        downloads: usize,
    ) -> anyhow::Result<Self> {
        Self::with_hub(cache, registry, pending, downloads, HubClient::new()?)
    }
    fn with_hub(
        cache: ModelCache,
        registry: ModelRegistry,
        pending: Vec<PendingTransfer>,
        downloads: usize,
        hub: HubClient,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            downloads > 0 && downloads <= Semaphore::MAX_PERMITS,
            "model download concurrency must be 1..={}",
            Semaphore::MAX_PERMITS
        );
        let (events, _) = broadcast::channel(64);
        let manager = Self(Arc::new(Inner {
            cache,
            registry,
            hub,
            permits: Arc::new(Semaphore::new(downloads)),
            operations: tokio::sync::Mutex::new(HashMap::new()),
            events,
        }));
        for transfer in pending {
            let job = Arc::new(TransferJob {
                request: transfer.plan().model().clone(),
                state: Mutex::new(JobState::Incomplete {
                    transfer,
                    error: None,
                }),
                progress: Mutex::new(ProgressState::default()),
            });
            manager.0.registry.register_transfer(job)?;
        }
        Ok(manager)
    }
    async fn operation(&self, id: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut guards = self.0.operations.lock().await;
        if let Some(guard) = guards.get(id).and_then(Weak::upgrade) {
            return guard;
        }
        guards.retain(|_, guard| guard.strong_count() > 0);
        let guard = Arc::new(tokio::sync::Mutex::new(()));
        guards.insert(id.into(), Arc::downgrade(&guard));
        guard
    }
    fn event(&self, event: &str, model: &str, data: Value) {
        let _ = self.0.events.send(ManagementEvent::Protocol(
            json!({"model":model,"event":event,"data":data}),
        ));
    }
    pub(crate) fn subscribe(&self) -> broadcast::Receiver<ManagementEvent> {
        self.0.events.subscribe()
    }
    pub(crate) fn has_transfer(&self, id: &str) -> Result<bool, ApiError> {
        Ok(self.0.registry.transfer(id).map_err(internal)?.is_some())
    }
    pub(crate) fn initial_events(&self) -> Result<Vec<Value>, ApiError> {
        let mut events = Vec::new();
        for job in self.0.registry.transfers().map_err(internal)? {
            let snapshot = job.snapshot();
            events.push(json!({"event":"model_status","model":snapshot.id,"data":{"status":if snapshot.running {"downloading"} else {"incomplete"}}}));
            let progress = lock(&job.progress);
            if !progress.files.is_empty() {
                events.push(json!({"event":"download_progress","model":job.id(),"data":{"progress":progress.files,"done":progress.done,"total":progress.total}}));
            }
        }
        Ok(events)
    }
    fn permit(&self) -> Result<OwnedSemaphorePermit, ApiError> {
        Arc::clone(&self.0.permits).try_acquire_owned().map_err(|_|ApiError::too_many_requests("model download concurrency is full; pause an active download or retry after it finishes"))
    }
    pub(crate) async fn start(&self, request: HubModel) -> Result<(), ApiError> {
        let held = self.operation(request.id()).await.lock_owned().await;
        let manager = self.clone();
        tokio::spawn(async move {
            let _held = held;
            manager.start_inner(request).await
        })
        .await
        .map_err(|error| internal(format!("model download admission task stopped: {error}")))?
    }
    async fn start_inner(&self, request: HubModel) -> Result<(), ApiError> {
        if let Some(job) = self.0.registry.transfer(request.id()).map_err(internal)? {
            if job.request != request {
                return Err(ApiError::conflict(
                    "an incomplete download for this model uses another revision; cancel it before changing revisions",
                ));
            }
            let state = lock(&job.state);
            if let JobState::Running { done, .. } = &*state {
                if done.has_changed().is_err() {
                    return Err(ApiError::conflict(
                        "download worker stopped unexpectedly; restart the server to recover this persisted transfer",
                    ));
                }
                return Ok(());
            }
            drop(state);
            let permit = self.permit()?;
            let mut state = lock(&job.state);
            let previous = std::mem::replace(&mut *state, JobState::Transition);
            let transfer = match previous {
                JobState::Incomplete { transfer, .. } => transfer,
                JobState::RecoveryRequired(message) => {
                    let error = ApiError::conflict(message.clone());
                    *state = JobState::RecoveryRequired(message);
                    return Err(error);
                }
                other => {
                    *state = other;
                    return Err(ApiError::conflict(
                        "model download changed state; refresh /models and retry",
                    ));
                }
            };
            self.launch(Arc::clone(&job), transfer, permit, &mut state);
            return Ok(());
        }
        if self
            .0
            .registry
            .contains_available(request.id())
            .map_err(internal)?
        {
            if let Some(cached) = self
                .0
                .registry
                .cached_request(request.id())
                .map_err(internal)?
            {
                if cached != request {
                    return Err(ApiError::conflict(
                        "this cached model uses another revision; delete it before changing revisions",
                    ));
                }
                return Ok(());
            }
            return Err(ApiError::conflict(
                "model id belongs to an operator-configured source; choose another id or remove that configuration first",
            ));
        }
        let permit = self.permit()?;
        let plan = self
            .0
            .hub
            .plan(request.clone(), self.0.cache.hub_limits())
            .await
            .map_err(|error| {
                if error.downcast_ref::<reqwest::Error>().is_some_and(|error| {
                    error.status().is_none_or(|status| status.is_server_error())
                }) {
                    ApiError::unavailable(format!(
                        "resolve Hub model '{}': {error:#}",
                        request.id()
                    ))
                } else {
                    ApiError::invalid_request(format!(
                        "resolve Hub model '{}': {error:#}",
                        request.id()
                    ))
                }
            })?;
        let cache = self.0.cache.clone();
        let transfer = tokio::task::spawn_blocking(move || cache.begin(plan))
            .await
            .map_err(|error| internal(format!("create durable model transfer task: {error}")))?
            .map_err(|error| {
                if error.is::<crate::model_cache::CacheQuota>() {
                    ApiError::too_many_requests(format!("admit model download: {error:#}"))
                } else {
                    ApiError::invalid_request(format!("admit model download: {error:#}"))
                }
            })?;
        let job = Arc::new(TransferJob {
            request,
            state: Mutex::new(JobState::Transition),
            progress: Mutex::new(ProgressState::default()),
        });
        self.0
            .registry
            .register_transfer(Arc::clone(&job))
            .map_err(internal)?;
        {
            let mut state = lock(&job.state);
            self.launch(Arc::clone(&job), transfer, permit, &mut state);
        }
        Ok(())
    }
    fn launch(
        &self,
        job: Arc<TransferJob>,
        transfer: PendingTransfer,
        permit: OwnedSemaphorePermit,
        state: &mut JobState,
    ) {
        // Frontends aggregate per-file totals. Publish the complete plan before
        // any byte event, so total size cannot grow as later files start.
        {
            let mut progress = lock(&job.progress);
            for (name, total) in transfer.plan().files() {
                let url = self.0.hub.file_url(transfer.plan(), name);
                progress
                    .files
                    .entry(url.to_string())
                    .or_insert(FileProgress { done: 0, total });
            }
            progress.total = transfer.plan().bytes();
        }
        let (pause, mut control) = watch::channel(false);
        let (done, completion) = watch::channel(false);
        *state = JobState::Running {
            pause,
            done: completion,
        };
        self.event(
            "model_status",
            job.request.id(),
            json!({"status":"downloading"}),
        );
        let manager = self.clone();
        tokio::spawn(async move {
            let _permit = permit;
            let outcome = transfer
                .run(&manager.0.hub, &mut control, |progress| {
                    manager.progress(&job, progress)
                })
                .await;
            let outcome = match outcome {
                TransferOutcome::Complete(complete) => {
                    match tokio::task::spawn_blocking(move || complete.publish()).await {
                        Ok(Ok(package)) => {
                            match manager
                                .0
                                .registry
                                .register_cached(package, Some(Arc::clone(&job)))
                                .await
                            {
                                Ok(()) => {
                                    *lock(&job.state) = JobState::Removed;
                                    manager.event("download_finished", job.request.id(), json!({}));
                                    manager.event("models_reload", "*", json!({}));
                                    drop(_permit);
                                    let _ = done.send(true);
                                    return;
                                }
                                Err(failure) => {
                                    tracing::error!(model=job.request.id(),error=%failure,"published model could not be registered; restart to recover the committed cache entry");
                                    *lock(&job.state)=JobState::RecoveryRequired("cache publication committed but registration failed; restart the server to recover the published entry".into());
                                    manager.event(
                                        "download_failed",
                                        job.request.id(),
                                        json!({"message":format!("{failure:#}")}),
                                    );
                                    drop(_permit);
                                    let _ = done.send(true);
                                    return;
                                }
                            }
                        }
                        Ok(Err(failure)) => TransferOutcome::Failed(failure),
                        Err(error) => {
                            tracing::error!(model=job.request.id(),%error,"publication task stopped; restart to recover the durable cache record");
                            *lock(&job.state)=JobState::RecoveryRequired("publication task stopped; restart to recover the durable cache record".into());
                            manager.event("download_failed",job.request.id(),json!({"message":"publication task stopped; restart to recover the durable cache record"}));
                            return;
                        }
                    }
                }
                other => other,
            };
            match outcome {
                TransferOutcome::Paused(transfer) => {
                    *lock(&job.state) = JobState::Incomplete {
                        transfer,
                        error: None,
                    };
                    manager.event("download_failed", job.request.id(), json!({"stopped":true}));
                }
                TransferOutcome::Failed(failure) => {
                    let message = format!("{:#}", failure.cause);
                    *lock(&job.state) = JobState::Incomplete {
                        transfer: failure.transfer,
                        error: Some(message.clone()),
                    };
                    manager.event(
                        "download_failed",
                        job.request.id(),
                        json!({"message":message}),
                    );
                }
                TransferOutcome::Complete(_) => unreachable!("publication handled above"),
            }
            manager.event(
                "model_status",
                job.request.id(),
                json!({"status":"incomplete"}),
            );
            drop(_permit);
            let _ = done.send(true);
        });
    }
    fn progress(&self, job: &Arc<TransferJob>, state: HubProgress<'_>) {
        let mut progress = lock(&job.progress);
        if let Some(file) = progress.files.get_mut(state.url) {
            file.done = state.done;
            file.total = state.total;
        } else {
            progress.files.insert(
                state.url.into(),
                FileProgress {
                    done: state.done,
                    total: state.total,
                },
            );
        }
        progress.done = state.package_bytes;
        progress.total = state.package_total;
        progress.version = progress.version.wrapping_add(1);
        drop(progress);
        let _ = self
            .0
            .events
            .send(ManagementEvent::Progress(Arc::clone(job)));
    }
    pub(crate) fn loaded_changed(&self, id: &str, loaded: bool) {
        self.event(
            "status_change",
            id,
            json!({"status":if loaded {"loaded"} else {"unloaded"}}),
        );
    }
    pub(crate) async fn pause(&self, id: &str) -> Result<bool, ApiError> {
        let guard = self.operation(id).await;
        let _held = guard.lock().await;
        self.pause_inner(id).await
    }
    async fn pause_inner(&self, id: &str) -> Result<bool, ApiError> {
        let Some(job) = self.0.registry.transfer(id).map_err(internal)? else {
            return Ok(false);
        };
        let wait = {
            match &*lock(&job.state) {
                JobState::Running { pause, done } => {
                    let _ = pause.send(true);
                    Some(done.clone())
                }
                JobState::Incomplete { .. } => None,
                JobState::Transition => {
                    return Err(ApiError::conflict(
                        "model download is changing state; refresh and retry",
                    ));
                }
                JobState::Removed => return Ok(false),
                JobState::RecoveryRequired(message) => {
                    return Err(ApiError::conflict(message.clone()));
                }
            }
        };
        if let Some(mut done) = wait {
            done.wait_for(|finished|*finished).await.map_err(|_|ApiError::internal("download worker stopped unexpectedly; restart to recover its persisted transfer"))?;
        }
        self.has_transfer(id)
    }
    pub(crate) async fn delete(&self, id: &str) -> Result<(), ApiError> {
        let guard = self.operation(id).await;
        let held = guard.lock_owned().await;
        // An admitted delete outlives its HTTP waiter, just like registry deletion.
        let manager = self.clone();
        let id = id.to_owned();
        tokio::spawn(async move {
            let _held = held;
            manager.delete_inner(&id).await
        })
        .await
        .map_err(|error| internal(format!("model deletion task stopped: {error}")))?
    }
    async fn delete_inner(&self, id: &str) -> Result<(), ApiError> {
        self.pause_inner(id).await?;
        let job = self.0.registry.transfer(id).map_err(internal)?;
        if let Some(job) = job {
            let previous = {
                let mut state = lock(&job.state);
                std::mem::replace(&mut *state, JobState::Transition)
            };
            match previous {
                JobState::Incomplete {
                    transfer,
                    error: previous_error,
                } => {
                    match tokio::task::spawn_blocking(move || transfer.cancel())
                        .await
                        .map_err(|error| internal(format!("cancel model download task: {error}")))?
                    {
                        Ok(()) => {
                            *lock(&job.state) = JobState::Removed;
                            self.0
                                .registry
                                .unregister_transfer(&job)
                                .map_err(internal)?;
                        }
                        Err(failure) => {
                            *lock(&job.state) = JobState::Incomplete {
                                transfer: failure.transfer,
                                error: previous_error,
                            };
                            return Err(internal(format!(
                                "cancel model download: {:#}",
                                failure.cause
                            )));
                        }
                    }
                }
                other => {
                    *lock(&job.state) = other;
                    return Err(ApiError::conflict(
                        "download changed state during deletion; refresh /models and retry",
                    ));
                }
            }
        }
        self.0
            .registry
            .remove_cached(self.0.cache.clone(), id.into())
            .await
            .map_err(|error| ApiError::conflict(format!("delete model '{id}': {error:#}")))?;
        self.event("model_remove", id, json!({}));
        self.event("models_reload", "*", json!({}));
        Ok(())
    }
    #[cfg(test)]
    pub(crate) fn with_test_hub(
        cache: ModelCache,
        registry: ModelRegistry,
        pending: Vec<PendingTransfer>,
        downloads: usize,
        hub: HubClient,
    ) -> Self {
        Self::with_hub(cache, registry, pending, downloads, hub).unwrap()
    }
}

impl TransferJob {
    pub(crate) fn id(&self) -> &str {
        self.request.id()
    }
    pub(crate) fn snapshot(&self) -> TransferSnapshot {
        let (running, error) = match &*lock(&self.state) {
            JobState::Running { done, .. } => {
                let failed = done.has_changed().is_err();
                (!failed,failed.then(||"download worker stopped unexpectedly; restart the server to recover its persisted transfer".into()))
            }
            JobState::Incomplete { error, .. } => (false, error.clone()),
            JobState::RecoveryRequired(message) => (false, Some(message.clone())),
            JobState::Transition | JobState::Removed => (false, None),
        };
        TransferSnapshot {
            id: self.id().into(),
            running,
            error,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_hub::tests::{Transfer, hub};
    use axum::{
        Router,
        body::{Body, BodyDataStream},
        http::{Request, StatusCode},
    };
    use tokio_stream::StreamExt;
    use tower::ServiceExt;
    fn state(root: &std::path::Path, hub: HubClient) -> crate::AppState {
        let mut state = crate::AppState::load_from_specs(
            Vec::new(),
            crate::ServerConfig {
                model_cache_dir: Some(root.into()),
                model_downloads: std::num::NonZeroUsize::new(1).unwrap(),
                ..Default::default()
            },
        )
        .unwrap();
        let old = state.model_manager.take().unwrap();
        state.model_manager = Some(ModelManager::with_test_hub(
            old.0.cache.clone(),
            state.registry.clone(),
            Vec::new(),
            1,
            hub,
        ));
        state
    }
    async fn call(router: Router, method: &str, path: &str, body: Value) -> (StatusCode, Value) {
        let response = router
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(path)
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let bytes = axum::body::to_bytes(response.into_body(), 2 * 1024 * 1024)
            .await
            .unwrap();
        (status, serde_json::from_slice(&bytes).unwrap())
    }
    struct Feed {
        body: BodyDataStream,
        buffer: Vec<u8>,
    }
    impl Feed {
        async fn open(router: Router) -> Self {
            let response = router
                .oneshot(
                    Request::builder()
                        .uri("/models/sse")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            Self {
                body: response.into_body().into_data_stream(),
                buffer: Vec::new(),
            }
        }
        async fn until(&mut self, predicate: impl Fn(&Value) -> bool) -> Vec<Value> {
            tokio::time::timeout(std::time::Duration::from_secs(10), async {
                let mut values = Vec::new();
                loop {
                    if let Some(index) = self.buffer.windows(2).position(|part| part == b"\n\n") {
                        let frame = self.buffer.drain(..index + 2).collect::<Vec<_>>();
                        let frame = std::str::from_utf8(&frame).unwrap();
                        if let Some(data) =
                            frame.lines().find_map(|line| line.strip_prefix("data: "))
                        {
                            let value: Value = serde_json::from_str(data).unwrap();
                            let done = predicate(&value);
                            values.push(value);
                            if done {
                                return values;
                            }
                        }
                    } else {
                        let chunk = self
                            .body
                            .next()
                            .await
                            .expect("SSE connection ended")
                            .unwrap();
                        self.buffer.extend_from_slice(&chunk);
                        assert!(
                            self.buffer.len() < 1024 * 1024,
                            "fixture SSE frame must remain bounded"
                        );
                    }
                }
            })
            .await
            .expect("expected management event before timeout")
        }
    }
    fn request() -> Value {
        json!({"model":"owner/package","sidecars":[]})
    }
    #[tokio::test]
    async fn http_download_stream_load_native_inference_and_delete_form_one_workflow() {
        let root = tempfile::tempdir().unwrap();
        let (client, server) = hub(Transfer::Valid).await;
        let state = state(root.path(), client);
        let router = crate::app(state);
        let (_, props) = call(router.clone(), "GET", "/props", Value::Null).await;
        assert_eq!(props["role"], "router");
        assert_eq!(props["nxrt_capabilities"]["model_management"], true);
        assert_eq!(props["nxrt_capabilities"]["server_tools"], false);
        let mut feed = Feed::open(router.clone()).await;
        let (status, ack) = call(router.clone(), "POST", "/models", request()).await;
        assert_eq!(status, StatusCode::OK, "{ack}");
        assert_eq!(ack, json!({"success":true}));
        let events = feed
            .until(|event| event["event"] == "download_finished")
            .await;
        let progress = events
            .iter()
            .filter(|event| event["event"] == "download_progress")
            .collect::<Vec<_>>();
        assert!(!progress.is_empty(), "{events:?}");
        for event in progress {
            let files = event["data"]["progress"].as_object().unwrap();
            assert_eq!(
                files.len(),
                3,
                "every event includes the complete file plan"
            );
            assert_eq!(
                files
                    .values()
                    .map(|file| file["total"].as_u64().unwrap())
                    .sum::<u64>(),
                event["data"]["total"].as_u64().unwrap(),
                "the frontend's per-file sum must equal the package total"
            );
            for (url, file) in files {
                assert!(url.contains(&format!("/resolve/{}/", "a".repeat(40))));
                assert!(file["done"].as_u64().unwrap() <= file["total"].as_u64().unwrap());
            }
        }
        let (status, models) = call(router.clone(), "GET", "/models", Value::Null).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(models["data"][0]["in_cache"], true);
        assert_eq!(models["data"][0]["can_remove"], true);
        assert_eq!(models["data"][0]["source"], "cache");
        let (status, error) = call(
            router.clone(),
            "POST",
            "/models",
            json!({"model":"owner/package","revision":"another"}),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT, "{error}");
        let (status, reply) = call(
            router.clone(),
            "POST",
            "/models/load",
            json!({"model":"owner/package"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        let (status,answer)=call(router.clone(),"POST","/v1/systemone",json!({"model":"owner/package","state":"hello","questions":{"x":{"type":"noul","instructions":"the","criteria":{"false":"hello","true":"world"}}}})).await;
        assert_eq!(status, StatusCode::OK, "{answer}");
        let probability = answer["answers"]["x"]["noul"].as_f64().unwrap();
        assert!(probability > 0.2 && probability < 0.3, "{answer}");
        assert_eq!(answer["usage"]["output_tokens"], 0);
        let (status, reply) = call(
            router.clone(),
            "DELETE",
            "/models?model=owner%2Fpackage",
            Value::Null,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        let (_, models) = call(router.clone(), "GET", "/models", Value::Null).await;
        assert!(models["data"].as_array().unwrap().is_empty());
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 1);
        feed.until(|event| event["event"] == "model_remove").await;
        server.abort();
    }
    #[tokio::test]
    async fn http_failed_download_retries_its_original_commit_without_resolving_the_branch_again() {
        let root = tempfile::tempdir().unwrap();
        let (client, server) = hub(Transfer::FailOnce).await;
        let router = crate::app(state(root.path(), client));
        let mut feed = Feed::open(router.clone()).await;
        assert_eq!(
            call(router.clone(), "POST", "/models", request()).await.0,
            StatusCode::OK
        );
        let events = feed
            .until(|event| event["event"] == "download_failed")
            .await;
        assert!(
            events.last().unwrap()["data"]["message"]
                .as_str()
                .unwrap()
                .contains("ended at")
        );
        let (_, models) = call(router.clone(), "GET", "/models", Value::Null).await;
        assert_eq!(models["data"][0]["status"]["value"], "incomplete");
        assert_eq!(models["data"][0]["in_cache"], false);
        let (status, error) = call(
            router.clone(),
            "POST",
            "/models/load",
            json!({"model":"owner/package"}),
        )
        .await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{error}");
        let (status,error)=call(router.clone(),"POST","/v1/systemone",json!({"model":"owner/package","state":"hello","questions":{"x":{"type":"noul","instructions":"the"}}})).await;
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{error}");
        let (status, error) = call(
            router.clone(),
            "POST",
            "/models",
            json!({"model":"owner/package","revision":"another"}),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT, "{error}");
        assert_eq!(
            call(router.clone(), "POST", "/models", request()).await.0,
            StatusCode::OK
        );
        // The mock changes main to an unavailable commit on a second metadata request.
        feed.until(|event| event["event"] == "download_finished")
            .await;
        assert_eq!(
            call(
                router,
                "DELETE",
                "/models?model=owner%2Fpackage",
                Value::Null
            )
            .await
            .0,
            StatusCode::OK
        );
        server.abort();
    }
    #[tokio::test]
    async fn http_pause_restart_and_resume_preserve_incomplete_downloads() {
        let root = tempfile::tempdir().unwrap();
        let (client, server) = hub(Transfer::Slow).await;
        let original = state(root.path(), client);
        let router = crate::app(original.clone());
        let mut feed = Feed::open(router.clone()).await;
        assert_eq!(
            call(router.clone(), "POST", "/models", request()).await.0,
            StatusCode::OK
        );
        feed.until(|event| {
            event["event"] == "download_progress"
                && event["data"]["progress"].as_object().is_some_and(|files| {
                    files.iter().any(|(url, file)| {
                        url.ends_with("/tokenizer.json")
                            && file["done"].as_u64().unwrap() > 0
                            && file["done"].as_u64().unwrap() < file["total"].as_u64().unwrap()
                    })
                })
        })
        .await;
        let (status, error) = call(
            router.clone(),
            "POST",
            "/models",
            json!({"model":"owner/another","sidecars":[]}),
        )
        .await;
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS, "{error}");
        let (status, reply) = call(
            router.clone(),
            "POST",
            "/models/unload",
            json!({"model":"owner/package"}),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{reply}");
        let events = feed
            .until(|event| event["event"] == "download_failed")
            .await;
        assert_eq!(events.last().unwrap()["data"]["stopped"], true);
        drop(feed);
        drop(router);
        drop(original);
        server.abort();
        let (client, server) = hub(Transfer::Valid).await;
        let mut client = Some(client);
        // The old SSE owner releases its state asynchronously after the body closes.
        let recovered = tokio::time::timeout(std::time::Duration::from_secs(10), async {
            loop {
                match crate::AppState::load_from_specs(
                    Vec::new(),
                    crate::ServerConfig {
                        model_cache_dir: Some(root.path().into()),
                        ..Default::default()
                    },
                ) {
                    Ok(mut recovered) => {
                        let old = recovered.model_manager.take().unwrap();
                        recovered.model_manager = Some(ModelManager::with_test_hub(
                            old.0.cache.clone(),
                            recovered.registry.clone(),
                            Vec::new(),
                            2,
                            client.take().unwrap(),
                        ));
                        break recovered;
                    }
                    Err(error) => {
                        assert!(format!("{error:#}").contains("another server"), "{error:#}");
                        tokio::task::yield_now().await;
                    }
                }
            }
        })
        .await
        .expect("old state releases the process lock");
        let router = crate::app(recovered);
        let (_, models) = call(router.clone(), "GET", "/models", Value::Null).await;
        assert_eq!(models["data"][0]["status"]["value"], "incomplete");
        let mut feed = Feed::open(router.clone()).await;
        assert_eq!(
            call(router.clone(), "POST", "/models", request()).await.0,
            StatusCode::OK
        );
        feed.until(|event| event["event"] == "download_finished")
            .await;
        assert_eq!(
            call(
                router,
                "DELETE",
                "/models?model=owner%2Fpackage",
                Value::Null
            )
            .await
            .0,
            StatusCode::OK
        );
        server.abort();
    }
    #[tokio::test]
    async fn http_cache_opt_in_and_unsupported_sources_fail_before_work() {
        let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/onnx_genai_workflows/decision");
        let router = crate::app(crate::AppState::load(&fixture, None).unwrap());
        assert_eq!(
            call(router, "POST", "/models", request()).await.0,
            StatusCode::FORBIDDEN
        );
        let root = tempfile::tempdir().unwrap();
        let (client, server) = hub(Transfer::Valid).await;
        let router = crate::app(state(root.path(), client));
        for body in [
            json!({"model":"../escape"}),
            json!({"model":"https://host/repo"}),
            json!({"model":"owner/package:Q4"}),
        ] {
            assert_eq!(
                call(router.clone(), "POST", "/models", body).await.0,
                StatusCode::BAD_REQUEST
            );
        }
        assert_eq!(
            call(
                router.clone(),
                "POST",
                "/models",
                json!({"model":"owner/package","sidecars":["mmproj"]})
            )
            .await
            .0,
            StatusCode::NOT_IMPLEMENTED
        );
        assert_eq!(
            call(
                router,
                "DELETE",
                "/models?model=owner%2Fpackage&sidecar=mmproj",
                Value::Null
            )
            .await
            .0,
            StatusCode::NOT_IMPLEMENTED
        );
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 1);
        server.abort();
    }
    #[tokio::test]
    async fn http_cache_quota_rejects_before_creating_a_transfer() {
        let root = tempfile::tempdir().unwrap();
        let (hub, server) = hub(Transfer::Valid).await;
        let mut state = crate::AppState::load_from_specs(
            Vec::new(),
            crate::ServerConfig {
                model_cache_dir: Some(root.path().into()),
                model_cache_max_files: std::num::NonZeroUsize::new(3).unwrap(),
                ..Default::default()
            },
        )
        .unwrap();
        let old = state.model_manager.take().unwrap();
        state.model_manager = Some(ModelManager::with_test_hub(
            old.0.cache.clone(),
            state.registry.clone(),
            Vec::new(),
            1,
            hub,
        ));
        let router = crate::app(state);
        let (status, error) = call(router.clone(), "POST", "/models", request()).await;
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS, "{error}");
        assert!(
            error["error"]["message"]
                .as_str()
                .unwrap()
                .contains("quota exhausted")
        );
        let (_, models) = call(router, "GET", "/models", Value::Null).await;
        assert!(models["data"].as_array().unwrap().is_empty());
        assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 1);
        server.abort();
    }
    #[tokio::test]
    async fn configured_models_cannot_alias_or_be_deleted_as_managed_cache_files() {
        let root = tempfile::tempdir().unwrap();
        let config = crate::ServerConfig {
            model_cache_dir: Some(root.path().into()),
            ..Default::default()
        };
        let spec = crate::ModelSpec {
            id: "configured".into(),
            path: root.path().join("future"),
            eager: false,
            warmup: false,
        };
        let error = crate::AppState::load_from_specs(vec![spec], config.clone())
            .err()
            .expect("configured cache aliases must fail before loading");
        assert!(
            error.to_string().contains("inside the managed cache"),
            "{error:#}"
        );
        let external = tempfile::tempdir().unwrap();
        std::fs::write(external.path().join("operator-data"), "keep").unwrap();
        let state = crate::AppState::load_from_specs(
            vec![crate::ModelSpec {
                id: "configured".into(),
                path: external.path().into(),
                eager: false,
                warmup: false,
            }],
            config,
        )
        .unwrap();
        let router = crate::app(state);
        let (status, error) = call(router, "DELETE", "/models?model=configured", Value::Null).await;
        assert_eq!(status, StatusCode::CONFLICT, "{error}");
        assert!(
            error["error"]["message"]
                .as_str()
                .unwrap()
                .contains("operator-configured")
        );
        assert_eq!(
            std::fs::read_to_string(external.path().join("operator-data")).unwrap(),
            "keep"
        );
    }
}

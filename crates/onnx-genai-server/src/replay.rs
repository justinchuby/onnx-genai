//! Bounded, byte-exact replay for any opted-in SSE inference response.
//!
//! A conversation identifies one producer. HTTP readers do not own or cancel
//! that producer; replacement, explicit DELETE and failed admission do.
use crate::routes::{ApiError, ApiJson};
use axum::{
    Json,
    body::{Body, Bytes},
    extract::{Query, Request, State},
    http::{HeaderMap, Method, StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};
use serde::Deserialize;
use serde_json::json;
use std::{
    collections::{BTreeMap, VecDeque},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{Semaphore, mpsc, watch};
use tokio_stream::{StreamExt, wrappers::ReceiverStream};

const MAX_ID_BYTES: usize = 1024;
const REPLAY_TTL: Duration = Duration::from_secs(300);
const READ_CHUNK: usize = 64 * 1024;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct ByteOffset(usize);

struct Buffer {
    bytes: VecDeque<u8>,
    begin: ByteOffset,
    end: ByteOffset,
    completed: Option<(Instant, u64)>,
}
struct Replay {
    buffer: Mutex<Buffer>,
    changed: watch::Sender<()>,
    cancelled: watch::Sender<bool>,
    started_at: u64,
    capacity: usize,
    readers: Arc<Semaphore>,
}
impl Replay {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: Mutex::new(Buffer {
                bytes: VecDeque::new(),
                begin: ByteOffset(0),
                end: ByteOffset(0),
                completed: None,
            }),
            changed: watch::channel(()).0,
            cancelled: watch::channel(false).0,
            started_at: unix_seconds(),
            capacity,
            readers: Arc::new(Semaphore::new(8)),
        }
    }
    fn append(&self, bytes: &[u8]) -> Result<(), String> {
        let mut buffer = self
            .buffer
            .lock()
            .map_err(|_| "stream replay buffer lock is poisoned".to_string())?;
        let end = buffer
            .end
            .0
            .checked_add(bytes.len())
            .ok_or("stream replay byte offset overflow; start a new conversation")?;
        let keep = bytes.len().min(self.capacity);
        let old_keep = self.capacity.saturating_sub(keep);
        let discard = buffer.bytes.len().saturating_sub(old_keep);
        buffer.bytes.drain(..discard);
        buffer.bytes.extend(&bytes[bytes.len() - keep..]);
        buffer.end = ByteOffset(end);
        buffer.begin = ByteOffset(end - buffer.bytes.len());
        drop(buffer);
        self.changed.send_replace(());
        Ok(())
    }
    fn finish(&self) {
        if let Ok(mut buffer) = self.buffer.lock() {
            buffer
                .completed
                .get_or_insert_with(|| (Instant::now(), unix_seconds()));
        }
        self.changed.send_replace(());
    }
    fn cancel(&self) {
        self.cancelled.send_replace(true);
        if let Ok(mut buffer) = self.buffer.lock() {
            // Release retained payload promptly even when old HTTP readers exist.
            buffer.bytes = VecDeque::new();
            buffer.begin = buffer.end;
        }
        self.finish();
    }
    fn expired(&self) -> bool {
        self.buffer
            .lock()
            .map(|b| {
                b.completed
                    .is_some_and(|(at, _)| at.elapsed() >= REPLAY_TTL)
            })
            .unwrap_or(true)
    }
}

#[derive(Clone)]
pub(crate) struct ReplayRegistry {
    entries: Arc<Mutex<BTreeMap<String, Arc<Replay>>>>,
    max_streams: usize,
    buffer_bytes: usize,
    reaper_started: Arc<AtomicBool>,
}
impl ReplayRegistry {
    pub(crate) fn new(max_streams: usize, buffer_bytes: usize) -> Self {
        Self {
            entries: Arc::new(Mutex::new(BTreeMap::new())),
            max_streams,
            buffer_bytes,
            reaper_started: Arc::new(AtomicBool::new(false)),
        }
    }
    fn begin(&self, id: &str) -> Result<Arc<Replay>, ApiError> {
        validate_id(id)?;
        self.start_reaper();
        let mut entries = self.entries.lock().map_err(|_| replay_lock_error())?;
        entries.retain(|_, entry| !entry.expired());
        if !entries.contains_key(id) && entries.len() >= self.max_streams {
            let oldest = entries
                .iter()
                .filter_map(|(id, entry)| {
                    entry
                        .buffer
                        .lock()
                        .ok()?
                        .completed
                        .map(|(at, _)| (id.clone(), at))
                })
                .min_by_key(|(_, at)| *at)
                .map(|(id, _)| id);
            if let Some(id) = oldest {
                entries.remove(&id);
            } else {
                return Err(ApiError::too_many_requests(format!(
                    "replay stream capacity {} is occupied by active generations; stop an active conversation or retry without X-Conversation-Id",
                    self.max_streams
                )));
            }
        }
        let replay = Arc::new(Replay::new(self.buffer_bytes));
        if let Some(previous) = entries.insert(id.to_string(), replay.clone()) {
            previous.cancel();
        }
        Ok(replay)
    }
    fn start_reaper(&self) {
        if self.reaper_started.swap(true, Ordering::AcqRel) {
            return;
        }
        let entries = Arc::downgrade(&self.entries);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(REPLAY_TTL / 2).await;
                let Some(entries) = entries.upgrade() else {
                    return;
                };
                if let Ok(mut entries) = entries.lock() {
                    entries.retain(|_, entry| !entry.expired());
                }
            }
        });
    }
    fn get(&self, id: &str) -> Result<Arc<Replay>, ApiError> {
        validate_id(id)?;
        let mut entries = self.entries.lock().map_err(|_| replay_lock_error())?;
        entries.retain(|_, entry| !entry.expired());
        entries.get(id).cloned().ok_or_else(|| {
            ApiError::not_found(
                "stream not found or expired; start a new generation with X-Conversation-Id",
            )
        })
    }
    fn discard(&self, id: &str, replay: &Arc<Replay>) {
        if let Ok(mut entries) = self.entries.lock()
            && entries
                .get(id)
                .is_some_and(|entry| Arc::ptr_eq(entry, replay))
        {
            entries.remove(id);
        }
        replay.cancel();
    }
}

struct Admission {
    registry: ReplayRegistry,
    id: String,
    replay: Option<Arc<Replay>>,
}
impl Drop for Admission {
    fn drop(&mut self) {
        if let Some(replay) = self.replay.take() {
            self.registry.discard(&self.id, &replay);
        }
    }
}
struct Producer(Arc<Replay>);
impl Drop for Producer {
    fn drop(&mut self) {
        self.0.finish();
    }
}

pub(crate) async fn intercept(
    State(registry): State<ReplayRegistry>,
    request: Request,
    next: Next,
) -> Response {
    if request.method() != Method::POST || !request.headers().contains_key("x-conversation-id") {
        return next.run(request).await;
    }
    let id = match conversation_header(request.headers()) {
        Ok(id) => id,
        Err(error) => return error.into_response(),
    };
    let replay = match registry.begin(&id) {
        Ok(replay) => replay,
        Err(error) => return error.into_response(),
    };
    let mut admission = Admission {
        registry,
        id,
        replay: Some(replay.clone()),
    };
    let response = next.run(request).await;
    let is_sse = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("text/event-stream"));
    if !response.status().is_success() || !is_sse {
        return response;
    }
    if *replay.cancelled.borrow() {
        return ApiError::conflict("conversation was stopped before generation admission completed; submit a new turn to restart").into_response();
    }
    let (parts, body) = response.into_parts();
    let reader = match reader_body(replay.clone(), ByteOffset(0)) {
        Ok(body) => body,
        Err(error) => return error.into_response(),
    };
    let _ = admission.replay.take();
    tokio::spawn(async move {
        let producer = Producer(replay);
        let mut cancelled = producer.0.cancelled.subscribe();
        let mut source = body.into_data_stream();
        loop {
            if *cancelled.borrow() {
                break;
            }
            let next = tokio::select! {
                _ = cancelled.changed() => break,
                next = source.next() => next,
            };
            match next {
                Some(Ok(bytes)) => {
                    if let Err(error) = producer.0.append(&bytes) {
                        tracing::error!(%error, "stream replay append failed");
                        break;
                    }
                }
                Some(Err(error)) => {
                    tracing::warn!(%error, "replay source response failed");
                    break;
                }
                None => break,
            }
        }
        // Dropping the source body closes the original route's receiver, which
        // cancels its driver. Dropping Producer finalizes all remaining readers.
    });
    Response::from_parts(parts, reader)
}

fn reader_body(replay: Arc<Replay>, mut offset: ByteOffset) -> Result<Body, ApiError> {
    let reader_permit = replay.readers.clone().try_acquire_owned().map_err(|_| ApiError::too_many_requests("conversation already has eight replay readers; close another connection before reconnecting"))?;
    {
        let buffer = replay.buffer.lock().map_err(|_| replay_lock_error())?;
        if offset < buffer.begin {
            return Err(offset_lost(buffer.begin));
        }
    }
    let (tx, rx) = mpsc::channel::<Result<Bytes, std::io::Error>>(1);
    tokio::spawn(async move {
        let _reader_permit = reader_permit;
        let mut changed = replay.changed.subscribe();
        loop {
            let next = {
                match replay.buffer.lock() {
                    Err(_) => Err(std::io::Error::other(
                        "stream replay buffer lock is poisoned",
                    )),
                    Ok(buffer) => {
                        if *replay.cancelled.borrow() {
                            return;
                        }
                        if offset < buffer.begin {
                            Err(std::io::Error::other(
                                "stream offset lost while reading; reconnect from a retained offset or restart the generation",
                            ))
                        } else if offset < buffer.end {
                            let start = offset.0 - buffer.begin.0;
                            let bytes: Vec<_> = buffer
                                .bytes
                                .iter()
                                .skip(start)
                                .take(READ_CHUNK)
                                .copied()
                                .collect();
                            offset.0 += bytes.len();
                            Ok(Some(Bytes::from(bytes)))
                        } else if buffer.completed.is_some() {
                            return;
                        } else {
                            Ok(None)
                        }
                    }
                }
            };
            match next {
                Ok(Some(bytes)) => {
                    if tx.send(Ok(bytes)).await.is_err() {
                        return;
                    }
                }
                Err(error) => {
                    let _ = tx.send(Err(error)).await;
                    return;
                }
                Ok(None) => {
                    tokio::select! { _ = tx.closed() => return, result = changed.changed() => { if result.is_err() { return; } } }
                }
            }
        }
    });
    Ok(Body::from_stream(ReceiverStream::new(rx)))
}

#[derive(Deserialize)]
pub(crate) struct ReplayQuery {
    conv_id: String,
    #[serde(default, rename = "from")]
    offset: usize,
}
fn query(
    query: Result<Query<ReplayQuery>, axum::extract::rejection::QueryRejection>,
) -> Result<ReplayQuery, ApiError> {
    let Query(query) = query.map_err(|error| ApiError::invalid_request(format!(
        "invalid stream query: {error}; use conv_id=<conversation-id>&from=<non-negative byte offset>")))?;
    validate_id(&query.conv_id)?;
    Ok(query)
}
pub(crate) async fn get(
    State(state): State<crate::AppState>,
    query_input: Result<Query<ReplayQuery>, axum::extract::rejection::QueryRejection>,
) -> Result<Response, ApiError> {
    let query = query(query_input)?;
    let replay = state.replays.get(&query.conv_id)?;
    let body = reader_body(replay, ByteOffset(query.offset))?;
    Ok((
        [
            (header::CONTENT_TYPE, "text/event-stream"),
            (header::CACHE_CONTROL, "no-store"),
        ],
        body,
    )
        .into_response())
}
pub(crate) async fn delete(
    State(state): State<crate::AppState>,
    query_input: Result<Query<ReplayQuery>, axum::extract::rejection::QueryRejection>,
) -> Result<StatusCode, ApiError> {
    let query = query(query_input)?;
    let replay = state
        .replays
        .entries
        .lock()
        .map_err(|_| replay_lock_error())?
        .remove(&query.conv_id);
    if let Some(replay) = replay {
        replay.cancel();
    }
    Ok(StatusCode::NO_CONTENT)
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LookupRequest {
    conversation_ids: Vec<String>,
}
pub(crate) async fn lookup(
    State(state): State<crate::AppState>,
    ApiJson(request): ApiJson<LookupRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if request.conversation_ids.len() > 1024 {
        return Err(ApiError::invalid_request(
            "conversation_ids accepts at most 1024 identifiers per lookup; split the request",
        ));
    }
    for id in &request.conversation_ids {
        validate_id(id)?;
    }
    let mut entries = state
        .replays
        .entries
        .lock()
        .map_err(|_| replay_lock_error())?;
    entries.retain(|_, replay| !replay.expired());
    let mut result = Vec::new();
    for (id, replay) in entries.iter() {
        if !request.conversation_ids.iter().any(|requested| {
            id == requested
                || id
                    .strip_prefix(requested)
                    .is_some_and(|tail| tail.starts_with("::"))
        }) {
            continue;
        }
        let buffer = replay.buffer.lock().map_err(|_| replay_lock_error())?;
        result.push(json!({"conversation_id": id, "is_done": buffer.completed.is_some(), "total_bytes": buffer.end.0,
            "started_at": replay.started_at, "completed_at": buffer.completed.map_or(0, |(_, at)| at)}));
    }
    Ok(Json(json!(result)))
}
fn validate_id(id: &str) -> Result<(), ApiError> {
    if id.is_empty() || id.len() > MAX_ID_BYTES || id.chars().any(char::is_control) {
        return Err(ApiError::invalid_request(format!(
            "conversation id must contain 1..={MAX_ID_BYTES} bytes without control characters; supply X-Conversation-Id or conv_id"
        )));
    }
    Ok(())
}
fn conversation_header(headers: &HeaderMap) -> Result<String, ApiError> {
    let id = headers
        .get("x-conversation-id")
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| {
            ApiError::invalid_request("X-Conversation-Id must be a valid text header")
        })?;
    validate_id(id)?;
    Ok(id.to_string())
}
fn replay_lock_error() -> ApiError {
    ApiError::internal(
        "stream replay registry lock is poisoned; restart the server and retry the conversation",
    )
}
fn offset_lost(begin: ByteOffset) -> ApiError {
    ApiError::invalid_request(format!(
        "stream offset lost; retained bytes begin at {}; restart the generation or resume from a retained offset",
        begin.0
    ))
}
fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{Router, body::to_bytes, routing::post};
    use std::convert::Infallible;
    use tokio::time::timeout;
    use tower::ServiceExt;

    #[tokio::test]
    async fn replay_readers_are_bounded_and_release_admission_on_disconnect() {
        let replay = Arc::new(Replay::new(4096));
        let mut readers = Vec::new();
        for _ in 0..8 {
            readers.push(reader_body(replay.clone(), ByteOffset(0)).unwrap());
        }
        assert_eq!(
            reader_body(replay.clone(), ByteOffset(0))
                .err()
                .unwrap()
                .status,
            StatusCode::TOO_MANY_REQUESTS
        );
        readers.clear();
        tokio::time::timeout(Duration::from_secs(1), async {
            while replay.readers.available_permits() != 8 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(reader_body(replay, ByteOffset(0)).is_ok());
    }

    #[tokio::test]
    async fn replay_survives_reader_disconnect_and_preserves_exact_byte_offsets() {
        let registry = ReplayRegistry::new(2, 4096);
        let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(2);
        let source = Arc::new(Mutex::new(Some(rx)));
        let router = Router::new()
            .route(
                "/produce",
                post(move || {
                    let source = source.clone();
                    async move {
                        let receiver = source.lock().unwrap().take().unwrap();
                        (
                            [(header::CONTENT_TYPE, "text/event-stream")],
                            Body::from_stream(ReceiverStream::new(receiver)),
                        )
                    }
                }),
            )
            .layer(axum::middleware::from_fn_with_state(
                registry.clone(),
                intercept,
            ));
        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/produce")
                    .header("x-conversation-id", "chat::org/model")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        drop(response);
        let payload = "data: {\"content\":\"你好\"}\n\ndata: [DONE]\n\n".as_bytes();
        tx.send(Ok(Bytes::copy_from_slice(payload))).await.unwrap();
        drop(tx);
        let replay = registry.get("chat::org/model").unwrap();
        let actual = timeout(
            Duration::from_secs(5),
            to_bytes(
                reader_body(replay.clone(), ByteOffset(0)).unwrap(),
                usize::MAX,
            ),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(actual.as_ref(), payload);
        let offset = 7;
        let suffix = to_bytes(reader_body(replay, ByteOffset(offset)).unwrap(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(suffix.as_ref(), &payload[offset..]);
    }

    #[tokio::test]
    async fn replacing_a_conversation_cancels_the_original_source_without_another_token() {
        let registry = ReplayRegistry::new(1, 4096);
        let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(2);
        let source = Arc::new(Mutex::new(Some(rx)));
        let router = Router::new()
            .route(
                "/produce",
                post(move || {
                    let source = source.clone();
                    async move {
                        (
                            [(header::CONTENT_TYPE, "text/event-stream")],
                            Body::from_stream(ReceiverStream::new(
                                source.lock().unwrap().take().unwrap(),
                            )),
                        )
                    }
                }),
            )
            .layer(axum::middleware::from_fn_with_state(
                registry.clone(),
                intercept,
            ));
        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/produce")
                    .header("x-conversation-id", "chat")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        drop(response);
        let original = registry.get("chat").unwrap();
        let replacement = registry.begin("chat").unwrap();
        assert!(*original.cancelled.borrow());
        assert!(!*replacement.cancelled.borrow());
        timeout(Duration::from_secs(5), tx.closed())
            .await
            .expect("Stop/replacement must drop the source receiver even while no tokens arrive");
        registry.discard("chat", &original);
        assert!(
            Arc::ptr_eq(&registry.get("chat").unwrap(), &replacement),
            "stale producer cleanup must not remove the replacement"
        );
    }

    #[tokio::test]
    async fn replay_bounds_history_capacity_and_expiry_are_real() {
        let registry = ReplayRegistry::new(1, 4);
        let first = registry.begin("one").unwrap();
        first.append(b"abcdef").unwrap();
        let buffer = first.buffer.lock().unwrap();
        assert_eq!(buffer.bytes.iter().copied().collect::<Vec<_>>(), b"cdef");
        assert_eq!(buffer.begin, ByteOffset(2));
        assert_eq!(buffer.end, ByteOffset(6));
        drop(buffer);
        assert_eq!(
            registry.begin("two").err().unwrap().status,
            StatusCode::TOO_MANY_REQUESTS
        );
        first.finish();
        registry.begin("two").unwrap();
        assert_eq!(
            registry.get("one").err().unwrap().status,
            StatusCode::NOT_FOUND
        );
        let second = registry.get("two").unwrap();
        second.buffer.lock().unwrap().completed =
            Some((Instant::now() - REPLAY_TTL - Duration::from_secs(1), 1));
        assert_eq!(
            registry.get("two").err().unwrap().status,
            StatusCode::NOT_FOUND
        );
    }

    #[tokio::test]
    async fn evicted_offsets_fail_before_sse_headers_and_buffered_responses_leave_no_entry() {
        let replay = Arc::new(Replay::new(4));
        replay.append(b"abcdef").unwrap();
        let error = reader_body(replay, ByteOffset(1)).err().unwrap();
        assert_eq!(error.status, StatusCode::BAD_REQUEST);
        assert!(error.message.contains("retained bytes begin at 2"));
        let registry = ReplayRegistry::new(1, 4096);
        let router = Router::new()
            .route("/produce", post(|| async { Json(json!({"ok": true})) }))
            .layer(axum::middleware::from_fn_with_state(
                registry.clone(),
                intercept,
            ));
        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/produce")
                    .header("x-conversation-id", "chat")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            registry.get("chat").err().unwrap().status,
            StatusCode::NOT_FOUND
        );
    }
}

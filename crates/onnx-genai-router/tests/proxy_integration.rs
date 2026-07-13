//! End-to-end reverse-proxy integration tests.
//!
//! Spins up a stub "inference node" (a tiny axum server on an ephemeral port),
//! points a router at it, and drives requests through the router app so the
//! real `hyper` proxy path (TCP connect, header forwarding, streaming response,
//! session-affinity capture) is exercised — no mocks in the proxy itself.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::body::Body;
use axum::extract::State as AxumState;
use axum::http::{Request, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router as AxumRouter};
use tokio_stream::wrappers::ReceiverStream;
use tower::ServiceExt;

use onnx_genai_router::config::RoutingPolicy;
use onnx_genai_router::node::{NodeId, NodeState};
use onnx_genai_router::router::Router;
use onnx_genai_router::state::AppState;
use onnx_genai_router::{SharedState, build_app};

/// Shared counter so the stub node can report how many requests it received.
#[derive(Clone, Default)]
struct UpstreamState {
    hits: Arc<AtomicUsize>,
}

async fn upstream_chat(AxumState(state): AxumState<UpstreamState>, body: String) -> Response {
    state.hits.fetch_add(1, Ordering::SeqCst);
    // Echo the received body so the test can assert the proxy forwarded it.
    (StatusCode::OK, format!("echo:{body}")).into_response()
}

async fn upstream_stream(AxumState(state): AxumState<UpstreamState>) -> Response {
    state.hits.fetch_add(1, Ordering::SeqCst);
    // A multi-chunk streaming body (SSE-like) to exercise pass-through streaming.
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, std::convert::Infallible>>(4);
    tokio::spawn(async move {
        let _ = tx.send(Ok("data: a\n\n".to_string())).await;
        let _ = tx.send(Ok("data: b\n\n".to_string())).await;
    });
    let mut resp = Response::new(Body::from_stream(ReceiverStream::new(rx)));
    resp.headers_mut()
        .insert("content-type", "text/event-stream".parse().unwrap());
    resp
}

async fn upstream_create_session(AxumState(state): AxumState<UpstreamState>) -> Response {
    state.hits.fetch_add(1, Ordering::SeqCst);
    Json(serde_json::json!({ "id": "sess-xyz", "object": "session" })).into_response()
}

/// Bind a stub upstream node on an ephemeral port and return its `host:port`.
async fn spawn_upstream() -> (String, Arc<AtomicUsize>) {
    let state = UpstreamState::default();
    let hits = state.hits.clone();
    let app = AxumRouter::new()
        .route("/v1/chat/completions", post(upstream_chat))
        .route("/v1/stream", get(upstream_stream))
        .route("/v1/sessions", post(upstream_create_session))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (addr.to_string(), hits)
}

fn router_state(address: &str) -> SharedState {
    let node = NodeState::new("node-a", address);
    let router = Router::new(vec![node], RoutingPolicy::AffinityThenLoad);
    AppState::new(router, 1000)
}

#[tokio::test]
async fn proxies_chat_request_and_forwards_body() {
    let (address, hits) = spawn_upstream().await;
    let app = build_app(router_state(&address));

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(r#"{"messages":[{"role":"user","content":"hi"}]}"#))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(text.starts_with("echo:"));
    assert!(text.contains("\"role\":\"user\""));
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn streams_sse_response_through_proxy() {
    let (address, _hits) = spawn_upstream().await;
    let app = build_app(router_state(&address));

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/stream")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers().get("content-type").unwrap(),
        "text/event-stream"
    );
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    assert_eq!(text, "data: a\n\ndata: b\n\n");
}

#[tokio::test]
async fn records_affinity_on_create_session() {
    let (address, _hits) = spawn_upstream().await;
    let state = router_state(&address);
    let app = build_app(state.clone());

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/sessions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // The server-assigned id ("sess-xyz") is now pinned to the routed node.
    let router = state.router.lock().unwrap();
    assert_eq!(
        router.session_map().get("sess-xyz"),
        Some(&NodeId::new("node-a"))
    );
}

#[tokio::test]
async fn returns_503_when_no_healthy_nodes() {
    let (address, _hits) = spawn_upstream().await;
    let mut node = NodeState::new("node-a", &address);
    node.healthy = false;
    let router = Router::new(vec![node], RoutingPolicy::AffinityThenLoad);
    let state = AppState::new(router, 1000);
    let app = build_app(state);

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

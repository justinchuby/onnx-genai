//! The router's own HTTP API under `/router/*` (see `docs/DESIGN.md` §34.7).
//!
//! Everything here is served by axum. Requests that do not match a `/router/*`
//! route fall through to the [`crate::proxy`] reverse-proxy fallback.

use axum::Json;
use axum::Router as AxumRouter;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde::Serialize;

use crate::node::NodeId;
use crate::proxy::proxy_handler;
use crate::state::SharedState;

/// Build the axum app: the `/router/*` API plus the proxy fallback.
pub fn build_app(state: SharedState) -> AxumRouter {
    AxumRouter::new()
        .route("/router/status", get(router_status))
        .route("/router/sessions", get(router_sessions))
        .route("/router/metrics", get(router_metrics))
        .route("/router/drain/{node_id}", post(drain_node))
        .route("/router/rebalance", post(rebalance))
        .fallback(proxy_handler)
        .with_state(state)
}

/// One node's state in the `/router/status` response.
#[derive(Serialize)]
struct NodeView {
    id: String,
    address: String,
    healthy: bool,
    draining: bool,
    kv_usage: f32,
    queue_depth: u32,
    active_sessions: u32,
    tokens_per_second: f64,
    consecutive_misses: u32,
}

/// `GET /router/status` — router health + per-node state.
#[derive(Serialize)]
struct StatusResponse {
    healthy: bool,
    healthy_nodes: usize,
    total_nodes: usize,
    nodes: Vec<NodeView>,
}

async fn router_status(State(state): State<SharedState>) -> Json<StatusResponse> {
    let router = state.router.lock().expect("router mutex poisoned");
    let nodes: Vec<NodeView> = router
        .nodes()
        .iter()
        .map(|n| NodeView {
            id: n.id.0.clone(),
            address: n.address.clone(),
            healthy: n.healthy,
            draining: router.is_draining(&n.id),
            kv_usage: n.kv_usage,
            queue_depth: n.queue_depth,
            active_sessions: n.active_sessions,
            tokens_per_second: n.tokens_per_second,
            consecutive_misses: n.consecutive_misses,
        })
        .collect();
    let healthy_nodes = router.healthy_node_count();
    Json(StatusResponse {
        healthy: healthy_nodes > 0,
        healthy_nodes,
        total_nodes: nodes.len(),
        nodes,
    })
}

/// `GET /router/sessions` — the session → node affinity table.
#[derive(Serialize)]
struct SessionsResponse {
    count: usize,
    sessions: std::collections::BTreeMap<String, String>,
}

async fn router_sessions(State(state): State<SharedState>) -> Json<SessionsResponse> {
    let router = state.router.lock().expect("router mutex poisoned");
    let sessions: std::collections::BTreeMap<String, String> = router
        .session_map()
        .iter()
        .map(|(session, node)| (session.clone(), node.0.clone()))
        .collect();
    Json(SessionsResponse {
        count: sessions.len(),
        sessions,
    })
}

/// `GET /router/metrics` — Prometheus text exposition.
async fn router_metrics(State(state): State<SharedState>) -> Response {
    let body = {
        let router = state.router.lock().expect("router mutex poisoned");
        state.metrics.encode(&router)
    };
    (
        StatusCode::OK,
        [("content-type", "text/plain; version=0.0.4")],
        body,
    )
        .into_response()
}

/// `POST /router/drain/{node_id}` — mark a node draining (stop new sessions,
/// keep serving existing affinity).
#[derive(Serialize)]
struct DrainResponse {
    node: String,
    draining: bool,
}

async fn drain_node(State(state): State<SharedState>, Path(node_id): Path<String>) -> Response {
    let id = NodeId::new(node_id.clone());
    let mut router = state.router.lock().expect("router mutex poisoned");
    if router.set_draining(&id, true) {
        Json(DrainResponse {
            node: node_id,
            draining: true,
        })
        .into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            format!("unknown node id `{node_id}`"),
        )
            .into_response()
    }
}

/// `POST /router/rebalance` — trigger session rebalancing across nodes.
#[derive(Serialize)]
struct RebalanceResponse {
    migrated: usize,
}

async fn rebalance(State(state): State<SharedState>) -> Json<RebalanceResponse> {
    let mut router = state.router.lock().expect("router mutex poisoned");
    let migrated = router.rebalance();
    Json(RebalanceResponse { migrated })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt; // oneshot

    use crate::config::RoutingPolicy;
    use crate::node::NodeState;
    use crate::router::{RouteRequest, Router};
    use crate::state::AppState;

    fn test_state() -> SharedState {
        let mut n0 = NodeState::new("gpu-0", "10.0.0.1:8000");
        n0.kv_usage = 0.5;
        n0.queue_depth = 3;
        let n1 = NodeState::new("gpu-1", "10.0.0.2:8000");
        let router = Router::new(vec![n0, n1], RoutingPolicy::AffinityThenLoad);
        AppState::new(router, 1000)
    }

    async fn body_string(resp: Response) -> String {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    #[tokio::test]
    async fn status_returns_200_with_node_shape() {
        let app = build_app(test_state());
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/router/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let json: serde_json::Value = serde_json::from_str(&body_string(resp).await).unwrap();
        assert_eq!(json["healthy"], true);
        assert_eq!(json["total_nodes"], 2);
        assert_eq!(json["nodes"][0]["id"], "gpu-0");
        assert_eq!(json["nodes"][0]["draining"], false);
    }

    #[tokio::test]
    async fn metrics_returns_200_prometheus_text() {
        let app = build_app(test_state());
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/router/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let text = body_string(resp).await;
        assert!(text.contains("onnx_genai_router_node_healthy{node=\"gpu-0\"}"));
        assert!(text.contains("onnx_genai_router_session_map_size"));
    }

    #[tokio::test]
    async fn sessions_reflects_affinity_table() {
        let state = test_state();
        {
            let mut router = state.router.lock().unwrap();
            router.record_session_affinity("s1", NodeId::new("gpu-0"));
        }
        let app = build_app(state);
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/router/sessions")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&body_string(resp).await).unwrap();
        assert_eq!(json["count"], 1);
        assert_eq!(json["sessions"]["s1"], "gpu-0");
    }

    #[tokio::test]
    async fn drain_marks_node_and_reroutes_new_sessions() {
        let state = test_state();
        // Pin s1 to gpu-1 (least loaded: kv 0.0 vs 0.5).
        let pinned = {
            let mut router = state.router.lock().unwrap();
            router
                .route(&RouteRequest {
                    session_id: Some("s1".into()),
                    system_prompt_hash: None,
                })
                .unwrap()
        };
        assert_eq!(pinned, NodeId::new("gpu-1"));

        let app = build_app(state.clone());
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/router/drain/gpu-1")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let mut router = state.router.lock().unwrap();
        assert!(router.is_draining(&NodeId::new("gpu-1")));
        // Existing session keeps its affinity to the draining node.
        let existing = router
            .route(&RouteRequest {
                session_id: Some("s1".into()),
                system_prompt_hash: None,
            })
            .unwrap();
        assert_eq!(existing, NodeId::new("gpu-1"));
        // A new session avoids the draining node.
        let fresh = router
            .route(&RouteRequest {
                session_id: Some("s2".into()),
                system_prompt_hash: None,
            })
            .unwrap();
        assert_eq!(fresh, NodeId::new("gpu-0"));
    }

    #[tokio::test]
    async fn drain_unknown_node_returns_404() {
        let app = build_app(test_state());
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/router/drain/ghost")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn rebalance_returns_migrated_count() {
        let state = test_state();
        {
            let mut router = state.router.lock().unwrap();
            router
                .route(&RouteRequest {
                    session_id: Some("s1".into()),
                    system_prompt_hash: None,
                })
                .unwrap();
            router.set_draining(&NodeId::new("gpu-1"), true);
        }
        let app = build_app(state);
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/router/rebalance")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_str(&body_string(resp).await).unwrap();
        assert_eq!(json["migrated"], 1);
    }
}

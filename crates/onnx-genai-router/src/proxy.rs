//! Reverse proxy: forwards non-`/router/*` requests to the routed inference
//! node and streams the response back (see `docs/DESIGN.md` §34.7).
//!
//! **Model-agnostic.** The proxy parses the request body *only* to extract an
//! opaque `session_id` (for affinity) and to hash the system prompt / first
//! message (for prefix co-location). It never inspects or branches on model
//! names. Bodies that are not JSON, or that carry neither signal, are forwarded
//! verbatim and routed by least-loaded fallback.
//!
//! Streaming: request bodies are buffered (they are small JSON and we must read
//! them to route). Response bodies are **streamed** — the upstream
//! `hyper::body::Incoming` is wrapped straight into an `axum` body, so SSE
//! token streams flow through without buffering. The sole exception is
//! `POST /v1/sessions`, whose (small, non-streamed) response is buffered so we
//! can read the server-assigned session id and record affinity.

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderName, Method, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use http_body_util::Full;
use hyper::Request as HyperRequest;
use serde_json::Value;

use crate::node::NodeId;
use crate::prefix_map::hash_system_prompt;
use crate::router::RouteRequest;
use crate::state::SharedState;

/// Max request body we will buffer before routing (16 MiB). Larger uploads are
/// rejected; the router is not a bulk data path.
const MAX_REQUEST_BODY: usize = 16 * 1024 * 1024;

/// Hop-by-hop headers that must not be forwarded across a proxy (RFC 7230
/// §6.1), plus `host` which we let hyper set for the upstream authority.
fn is_hop_by_hop(name: &HeaderName) -> bool {
    matches!(
        name.as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailers"
            | "transfer-encoding"
            | "upgrade"
            | "host"
    )
}

/// axum fallback handler: everything not matched by the `/router/*` API.
pub async fn proxy_handler(State(state): State<SharedState>, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let method = parts.method.clone();
    let uri = parts.uri.clone();

    let body_bytes = match axum::body::to_bytes(body, MAX_REQUEST_BODY).await {
        Ok(b) => b,
        Err(_) => {
            return (StatusCode::PAYLOAD_TOO_LARGE, "request body too large").into_response();
        }
    };

    // Model-agnostic extraction of affinity + prefix signals.
    let route_request = extract_route_fields(&body_bytes);

    // Route under a short synchronous lock; drop the guard before any await.
    let decision = {
        let mut router = state.router.lock().expect("router mutex poisoned");
        router.route_decision(&route_request)
    };
    let (node_id, decision) = match decision {
        Some(x) => x,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "no healthy nodes available to route request",
            )
                .into_response();
        }
    };

    // Resolve the upstream address (short lock).
    let address = {
        let router = state.router.lock().expect("router mutex poisoned");
        match router.node_by_id(&node_id) {
            Some(node) => node.address.clone(),
            None => {
                return (StatusCode::SERVICE_UNAVAILABLE, "routed node disappeared")
                    .into_response();
            }
        }
    };

    state.metrics.record_request(node_id.as_str(), decision);
    tracing::debug!(
        node = %node_id,
        decision = ?decision,
        method = %method,
        path = uri.path(),
        "proxying request"
    );

    // `POST /v1/sessions` establishes a session whose id the *server* assigns;
    // buffer its response so we can read that id and record affinity.
    let capture_session = method == Method::POST && uri.path() == "/v1/sessions";

    match forward(
        &state,
        &address,
        &parts.method,
        &uri,
        &parts.headers,
        body_bytes,
    )
    .await
    {
        Ok(resp) => {
            if capture_session {
                capture_session_affinity(&state, &node_id, resp).await
            } else {
                resp
            }
        }
        Err(err) => {
            tracing::warn!(node = %node_id, address = %address, error = %err, "upstream proxy error");
            (StatusCode::BAD_GATEWAY, format!("upstream error: {err}")).into_response()
        }
    }
}

/// Forward one request to `address`, returning a streaming [`Response`].
async fn forward(
    state: &SharedState,
    address: &str,
    method: &Method,
    uri: &Uri,
    headers: &HeaderMap,
    body: Bytes,
) -> Result<Response, Box<dyn std::error::Error + Send + Sync>> {
    let path_and_query = uri.path_and_query().map(|pq| pq.as_str()).unwrap_or("/");
    let upstream_uri = format!("http://{address}{path_and_query}");

    let mut builder = HyperRequest::builder()
        .method(method.clone())
        .uri(upstream_uri);
    for (name, value) in headers.iter() {
        if !is_hop_by_hop(name) {
            builder = builder.header(name, value);
        }
    }
    let outbound = builder.body(Full::new(body))?;

    let upstream = state.client.request(outbound).await?;
    let (parts, incoming) = upstream.into_parts();

    // Stream the upstream body straight through (SSE-friendly, no buffering).
    let mut response = Response::new(Body::new(incoming));
    *response.status_mut() = parts.status;
    for (name, value) in parts.headers.iter() {
        if !is_hop_by_hop(name) {
            response.headers_mut().insert(name, value.clone());
        }
    }
    Ok(response)
}

/// Buffer a `/v1/sessions` response, record affinity for the server-assigned
/// session id, then return the (rebuilt) response to the client.
///
/// The buffer is capped at [`MAX_REQUEST_BODY`] (symmetric with the request
/// path). A session-creation response is small JSON. If the upstream advertises
/// a `Content-Length` larger than the cap we **stream the response through
/// untouched without capturing affinity** rather than failing the client's
/// request. If the body has no `Content-Length` (e.g. chunked) and turns out to
/// exceed the cap while buffering, we can no longer stream it (it has been
/// consumed), so we surface a `502` — but this is not the streaming path a
/// well-behaved session endpoint takes.
async fn capture_session_affinity(
    state: &SharedState,
    node_id: &NodeId,
    response: Response,
) -> Response {
    let (parts, body) = response.into_parts();

    // If the upstream declares an oversize body up front, don't buffer it:
    // stream it straight through and skip affinity capture.
    let declared_len = content_length(&parts.headers);
    if exceeds_cap(declared_len, MAX_REQUEST_BODY) {
        tracing::warn!(
            node = %node_id,
            content_length = ?declared_len,
            "/v1/sessions response exceeds buffer cap; streaming through without capturing affinity"
        );
        return Response::from_parts(parts, body);
    }

    // Buffer within the cap (symmetric with the request-path upload cap).
    let bytes = match axum::body::to_bytes(body, MAX_REQUEST_BODY).await {
        Ok(bytes) => bytes,
        Err(err) => {
            tracing::warn!(
                node = %node_id,
                error = %err,
                "failed reading /v1/sessions response within buffer cap; skipping affinity capture"
            );
            return (StatusCode::BAD_GATEWAY, "failed reading upstream response").into_response();
        }
    };

    if let Some(session_id) = extract_response_session_id(&bytes) {
        let mut router = state.router.lock().expect("router mutex poisoned");
        router.record_session_affinity(session_id, node_id.clone());
    }

    let mut rebuilt = Response::from_parts(parts, Body::from(bytes));
    // A buffered body has a known length; drop any stale transfer-encoding.
    rebuilt.headers_mut().remove(header::TRANSFER_ENCODING);
    rebuilt
}

/// Parse a `Content-Length` header into bytes, if present and valid.
fn content_length(headers: &HeaderMap) -> Option<u64> {
    headers
        .get(header::CONTENT_LENGTH)?
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
}

/// Whether a declared body length is known to exceed the buffering cap.
fn exceeds_cap(declared_len: Option<u64>, cap: usize) -> bool {
    matches!(declared_len, Some(len) if len > cap as u64)
}

/// Extract the (opaque) affinity + prefix signals from a request body.
///
/// Generic and model-agnostic:
/// - `session_id`: the top-level `session_id` or `session` string field.
/// - prefix hash: the first `system`-role message content in a `messages`
///   array, else a top-level `system` string, else a top-level `prompt`
///   string. The chosen text is hashed with [`hash_system_prompt`].
pub fn extract_route_fields(body: &[u8]) -> RouteRequest {
    let value: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(_) => return RouteRequest::default(),
    };
    RouteRequest {
        session_id: extract_session_id(body),
        system_prompt_hash: extract_prefix_text(&value).map(|text| hash_system_prompt(&text)),
    }
}

/// Pull an opaque session id from a JSON object (`session_id` or `session`).
pub fn extract_session_id(body: &[u8]) -> Option<String> {
    session_id_from_keys(body, &["session_id", "session"])
}

/// Pull the session id from a `POST /v1/sessions` *response*. A session-creation
/// response conventionally returns the new id in an `id` field, so we also
/// accept that (generically — no model-specific knowledge).
fn extract_response_session_id(body: &[u8]) -> Option<String> {
    session_id_from_keys(body, &["id", "session_id", "session"])
}

fn session_id_from_keys(body: &[u8], keys: &[&str]) -> Option<String> {
    let value: Value = serde_json::from_slice(body).ok()?;
    let obj = value.as_object()?;
    for key in keys {
        if let Some(Value::String(s)) = obj.get(*key)
            && !s.is_empty()
        {
            return Some(s.clone());
        }
    }
    None
}

/// Extract the text to hash for prefix co-location, generically.
fn extract_prefix_text(value: &Value) -> Option<String> {
    let obj = value.as_object()?;

    if let Some(Value::Array(messages)) = obj.get("messages") {
        // First system-role message, else the very first message.
        let chosen = messages
            .iter()
            .find(|m| m.get("role").and_then(Value::as_str) == Some("system"))
            .or_else(|| messages.first());
        if let Some(message) = chosen
            && let Some(text) = message_content_text(message)
            && !text.is_empty()
        {
            return Some(text);
        }
    }

    for key in ["system", "prompt"] {
        if let Some(Value::String(s)) = obj.get(key)
            && !s.is_empty()
        {
            return Some(s.clone());
        }
    }
    None
}

/// Flatten a chat message's `content`, which may be a string or an array of
/// typed parts, into plain text (text parts concatenated).
fn message_content_text(message: &Value) -> Option<String> {
    match message.get("content")? {
        Value::String(s) => Some(s.clone()),
        Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                if let Some(s) = part.get("text").and_then(Value::as_str) {
                    text.push_str(s);
                }
            }
            Some(text)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_session_id_from_session_id_field() {
        let body = br#"{"session_id":"abc-123","messages":[]}"#;
        let req = extract_route_fields(body);
        assert_eq!(req.session_id.as_deref(), Some("abc-123"));
    }

    #[test]
    fn extracts_session_id_from_session_field() {
        let body = br#"{"session":"s-9"}"#;
        assert_eq!(extract_session_id(body).as_deref(), Some("s-9"));
    }

    #[test]
    fn hashes_system_message_from_chat_body() {
        let body = br#"{
            "messages": [
                {"role":"system","content":"You are a helpful assistant."},
                {"role":"user","content":"hi"}
            ]
        }"#;
        let req = extract_route_fields(body);
        assert_eq!(
            req.system_prompt_hash,
            Some(hash_system_prompt("You are a helpful assistant."))
        );
    }

    #[test]
    fn hashes_first_message_when_no_system_role() {
        let body = br#"{"messages":[{"role":"user","content":"hello world"}]}"#;
        let req = extract_route_fields(body);
        assert_eq!(
            req.system_prompt_hash,
            Some(hash_system_prompt("hello world"))
        );
    }

    #[test]
    fn flattens_array_content_parts() {
        let body = br#"{
            "messages": [
                {"role":"system","content":[
                    {"type":"text","text":"part-a "},
                    {"type":"image_url","image_url":{"url":"ignored"}},
                    {"type":"text","text":"part-b"}
                ]}
            ]
        }"#;
        let req = extract_route_fields(body);
        assert_eq!(
            req.system_prompt_hash,
            Some(hash_system_prompt("part-a part-b"))
        );
    }

    #[test]
    fn hashes_prompt_field_for_completions() {
        let body = br#"{"prompt":"complete this"}"#;
        let req = extract_route_fields(body);
        assert_eq!(
            req.system_prompt_hash,
            Some(hash_system_prompt("complete this"))
        );
        assert!(req.session_id.is_none());
    }

    #[test]
    fn non_json_body_yields_empty_route_request() {
        let req = extract_route_fields(b"not json at all");
        assert!(req.session_id.is_none());
        assert!(req.system_prompt_hash.is_none());
    }

    #[test]
    fn empty_session_id_is_ignored() {
        let body = br#"{"session_id":""}"#;
        assert!(extract_session_id(body).is_none());
    }

    #[test]
    fn hop_by_hop_headers_are_recognized() {
        assert!(is_hop_by_hop(&HeaderName::from_static("connection")));
        assert!(is_hop_by_hop(&HeaderName::from_static("host")));
        assert!(!is_hop_by_hop(&HeaderName::from_static("content-type")));
    }

    #[test]
    fn oversize_content_length_exceeds_cap() {
        // A declared length above the cap is oversize; at/below is not.
        assert!(exceeds_cap(
            Some(MAX_REQUEST_BODY as u64 + 1),
            MAX_REQUEST_BODY
        ));
        assert!(!exceeds_cap(
            Some(MAX_REQUEST_BODY as u64),
            MAX_REQUEST_BODY
        ));
        assert!(!exceeds_cap(Some(0), MAX_REQUEST_BODY));
        // Unknown length is not treated as oversize (buffer-with-cap path).
        assert!(!exceeds_cap(None, MAX_REQUEST_BODY));
    }

    #[test]
    fn content_length_parses_valid_header() {
        let mut headers = HeaderMap::new();
        headers.insert(header::CONTENT_LENGTH, "1234".parse().unwrap());
        assert_eq!(content_length(&headers), Some(1234));

        headers.insert(header::CONTENT_LENGTH, "not-a-number".parse().unwrap());
        assert_eq!(content_length(&headers), None);

        assert_eq!(content_length(&HeaderMap::new()), None);
    }

    #[tokio::test]
    async fn oversize_session_response_streams_through_without_capturing() {
        use crate::config::RoutingPolicy;
        use crate::node::NodeState;
        use crate::router::Router;
        use crate::state::AppState;

        let router = Router::new(
            vec![NodeState::new("gpu-0", "10.0.0.1:8000")],
            RoutingPolicy::AffinityThenLoad,
        );
        let state = AppState::new(router, 10);
        let node_id = NodeId::new("gpu-0");

        // Build a response that advertises an oversize Content-Length but whose
        // body would parse as a session id if it were (wrongly) captured.
        let body = br#"{"id":"should-not-be-captured"}"#.to_vec();
        let mut response = Response::new(Body::from(body));
        response.headers_mut().insert(
            header::CONTENT_LENGTH,
            ((MAX_REQUEST_BODY + 1) as u64).into(),
        );

        let out = capture_session_affinity(&state, &node_id, response).await;

        // Streamed through unchanged (200, still declares the oversize length)
        // and, crucially, NO affinity was recorded.
        assert_eq!(out.status(), StatusCode::OK);
        let router = state.router.lock().unwrap();
        assert!(router.session_map().get("should-not-be-captured").is_none());
    }
}

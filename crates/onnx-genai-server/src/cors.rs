//! Browser transport policy. This does not fetch URLs or execute tools.
use anyhow::{Context, ensure};
use axum::{
    extract::{Request, State},
    http::{HeaderMap, HeaderValue, Method, StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::{net::SocketAddr, sync::Arc};

/// Actual local address of an accepted socket, for custom in-process adapters.
/// `serve` uses `CorsConnectionInfo` instead, resolving each accepted connection.
#[derive(Clone, Copy, Debug)]
pub struct CorsListeningAddress(pub SocketAddr);

/// Accepted TCP connection context. Wildcard listener addresses are deliberately
/// not used: a socket accepted on 0.0.0.0 can have a loopback local address.
/// Custom axum TCP adapters can use `into_make_service_with_connect_info` with
/// this type. Resolution failures are retained and fail closed in middleware.
#[derive(Clone, Debug)]
pub struct CorsConnectionInfo(Result<SocketAddr, Arc<str>>);
impl
    axum::extract::connect_info::Connected<axum::serve::IncomingStream<'_, tokio::net::TcpListener>>
    for CorsConnectionInfo
{
    fn connect_info(stream: axum::serve::IncomingStream<'_, tokio::net::TcpListener>) -> Self {
        Self(stream.io().local_addr().map_err(|error| {
            Arc::from(format!("resolve accepted TCP socket local address for CORS policy: {error}; verify the accepted connection is valid and retry"))
        }))
    }
}

fn local_address(request: &Request) -> Result<Option<SocketAddr>, &str> {
    if let Some(info) = request
        .extensions()
        .get::<axum::extract::ConnectInfo<CorsConnectionInfo>>()
    {
        return match &info.0.0 {
            Ok(address) => Ok(Some(*address)),
            Err(message) => Err(message.as_ref()),
        };
    }
    Ok(request
        .extensions()
        .get::<CorsListeningAddress>()
        .map(|address| address.0))
}

#[derive(Clone, Debug)]
enum Origins {
    Any,
    SameOrigin,
    Listed(Vec<Origin>),
}
#[derive(Clone, Debug)]
struct Origin {
    value: HeaderValue,
    authority: Option<String>,
}
impl Origin {
    fn parse(value: &str) -> anyhow::Result<Self> {
        let header = HeaderValue::from_str(value).context(
            "invalid --cors-origins HTTP value; supply an origin without control characters",
        )?;
        if value == "null" {
            return Ok(Self {
                value: header,
                authority: None,
            });
        }
        let url = reqwest::Url::parse(value).with_context(|| {
            format!("invalid --cors-origins origin '{value}'; use scheme://host[:port]")
        })?;
        ensure!(
            matches!(url.scheme(), "http" | "https")
                && url.host_str().is_some()
                && url.username().is_empty()
                && url.password().is_none()
                && url.path() == "/"
                && url.query().is_none()
                && url.fragment().is_none()
                && !value.ends_with('/'),
            "invalid --cors-origins origin '{value}'; use an HTTP(S) origin without credentials, path, query or fragment"
        );
        let authority = value
            .split_once("://")
            .map(|(_, authority)| authority.to_owned());
        Ok(Self {
            value: header,
            authority,
        })
    }
}

/// Validated immutable CORS configuration, shared by all model routes.
#[derive(Clone, Debug)]
pub struct CorsPolicy(Arc<Policy>);
#[derive(Debug)]
struct Policy {
    origins: Origins,
    methods: HeaderValue,
    headers: HeaderValue,
    any_headers: bool,
    credentials: bool,
}
impl Default for CorsPolicy {
    fn default() -> Self {
        Self::new("*", "GET, POST, DELETE, OPTIONS", "*", true)
            .expect("valid built-in CORS defaults")
    }
}
impl CorsPolicy {
    pub fn credentials_enabled(&self) -> bool {
        self.0.credentials
    }

    pub fn new(
        origins: &str,
        methods: &str,
        headers: &str,
        credentials: bool,
    ) -> anyhow::Result<Self> {
        HeaderValue::from_str(origins)
            .context("invalid --cors-origins header; remove control characters")?;
        let origins = match origins {
            "*" => Origins::Any,
            "localhost" => Origins::SameOrigin,
            _ => {
                let mut values = Vec::new();
                for origin in origins.split(',').map(str::trim) {
                    ensure!(
                        !origin.is_empty(),
                        "--cors-origins contains an empty origin; supply *, localhost, or comma-separated HTTP(S) origins"
                    );
                    values.push(Origin::parse(origin)?);
                }
                Origins::Listed(values)
            }
        };
        for method in methods.split(',').map(str::trim) {
            ensure!(
                !method.is_empty() && Method::from_bytes(method.as_bytes()).is_ok(),
                "invalid --cors-methods entry '{method}'; supply comma-separated HTTP method tokens"
            );
        }
        let any_headers = headers == "*";
        if !any_headers {
            for name in headers.split(',').map(str::trim) {
                ensure!(
                    !name.is_empty() && axum::http::HeaderName::from_bytes(name.as_bytes()).is_ok(),
                    "invalid --cors-headers entry '{name}'; supply * or comma-separated HTTP header names"
                );
            }
        }
        Ok(Self(Arc::new(Policy {
            origins,
            methods: HeaderValue::from_str(methods)
                .context("invalid --cors-methods header; remove control characters")?,
            headers: HeaderValue::from_str(headers)
                .context("invalid --cors-headers header; remove control characters")?,
            any_headers,
            credentials,
        })))
    }
}

fn denied(reason: &str) -> Response {
    (StatusCode::FORBIDDEN, axum::Json(serde_json::json!({"error": {
        "message": format!("Cross-origin request not allowed: {reason}; configure --cors-origins for this browser origin"),
        "type": "permission_error", "code": 403
    }}))).into_response()
}
fn authority(origin: &str) -> Option<&str> {
    origin.split_once("://").map(|(_, authority)| authority)
}
fn loopback(ip: std::net::IpAddr) -> bool {
    ip.is_loopback()
        || matches!(ip, std::net::IpAddr::V6(ip) if ip.to_ipv4_mapped().is_some_and(|ip| ip.is_loopback()))
}
fn loopback_or_unspecified(ip: std::net::IpAddr) -> bool {
    loopback(ip) || ip.is_unspecified()
}
fn host_allowed(policy: &Origins, request: &Request) -> bool {
    let local = match local_address(request) {
        Ok(Some(local)) => local,
        Ok(None) => return true, // In-process routers have no network listener.
        Err(_) => return false,  // Middleware reports the connection-context failure first.
    };
    if !loopback(local.ip()) {
        return true;
    }
    let Some(host) = request
        .headers()
        .get(header::HOST)
        .and_then(|h| h.to_str().ok())
    else {
        return false;
    };
    let Ok(parsed) = host.parse::<axum::http::uri::Authority>() else {
        return false;
    };
    let name = parsed.host().trim_matches(['[', ']']);
    if name.eq_ignore_ascii_case("localhost")
        || name
            .parse::<std::net::IpAddr>()
            .is_ok_and(loopback_or_unspecified)
    {
        return true;
    }
    matches!(policy, Origins::Listed(origins) if origins.iter().any(|origin| origin.authority.as_ref().is_some_and(|allowed| allowed.eq_ignore_ascii_case(host))))
}
fn allowed(policy: &Origins, request: &Request) -> bool {
    if matches!(policy, Origins::Any) {
        return true;
    }
    if !host_allowed(policy, request) {
        return false;
    }
    if let Some(origin) = request.headers().get(header::ORIGIN) {
        let Ok(origin) = origin.to_str() else {
            return false;
        };
        let same = authority(origin)
            .zip(
                request
                    .headers()
                    .get(header::HOST)
                    .and_then(|h| h.to_str().ok()),
            )
            .is_some_and(|(origin, host)| origin.eq_ignore_ascii_case(host));
        return same
            || matches!(policy, Origins::Listed(origins) if origins.iter().any(|allowed| allowed.value.as_bytes() == origin.as_bytes()));
    }
    let cross_site = request
        .headers()
        .get("sec-fetch-site")
        .is_some_and(|value| value == "cross-site" || value == "same-site");
    !cross_site
        || (request.method() == Method::GET
            && request
                .headers()
                .get("sec-fetch-mode")
                .is_some_and(|value| value == "navigate")
            && crate::webui::is_asset(request.uri().path()))
}
fn vary(headers: &mut HeaderMap, name: &'static str) {
    headers.append(header::VARY, HeaderValue::from_static(name));
}

pub(crate) async fn enforce(
    State(policy): State<CorsPolicy>,
    request: Request,
    next: Next,
) -> Response {
    if let Err(message) = local_address(&request) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            axum::Json(serde_json::json!({"error": {
                "message": message, "type": "internal_error", "code": 500
            }})),
        )
            .into_response();
    }
    let policy = &policy.0;
    let origin = request.headers().get(header::ORIGIN).cloned();
    let accepted = allowed(&policy.origins, &request);
    let preflight = request.method() == Method::OPTIONS;
    let requested_headers = request
        .headers()
        .get(header::ACCESS_CONTROL_REQUEST_HEADERS)
        .cloned();
    let mut response = if !accepted {
        denied("origin or local Host is outside the configured policy")
    } else if preflight {
        // Browsers omit Authorization here. Never enter auth, replay, routing or model admission.
        StatusCode::OK.into_response()
    } else {
        next.run(request).await
    };
    let headers = response.headers_mut();
    vary(headers, "Origin");
    if accepted {
        match &policy.origins {
            Origins::Any if !policy.credentials => {
                headers.insert(
                    header::ACCESS_CONTROL_ALLOW_ORIGIN,
                    HeaderValue::from_static("*"),
                );
            }
            Origins::SameOrigin => {}
            _ => {
                if let Some(origin) = origin {
                    headers.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, origin);
                }
            }
        }
        if policy.credentials {
            headers.insert(
                header::ACCESS_CONTROL_ALLOW_CREDENTIALS,
                HeaderValue::from_static("true"),
            );
        }
        if preflight {
            headers.insert(header::ACCESS_CONTROL_ALLOW_METHODS, policy.methods.clone());
            if policy.any_headers {
                // Credentialed requests and Authorization require explicit names, not literal '*'.
                if let Some(requested) = requested_headers {
                    headers.insert(header::ACCESS_CONTROL_ALLOW_HEADERS, requested);
                }
                vary(headers, "Access-Control-Request-Headers");
            } else {
                headers.insert(header::ACCESS_CONTROL_ALLOW_HEADERS, policy.headers.clone());
            }
            vary(headers, "Access-Control-Request-Method");
        }
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Router,
        body::{Body, to_bytes},
        middleware,
        routing::post,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tower::ServiceExt;

    fn router(policy: CorsPolicy, calls: Arc<AtomicUsize>) -> Router {
        let keys = crate::ApiKeys::from_keys(["test-key"]).unwrap();
        Router::new()
            .route(
                "/completion",
                post(move || {
                    calls.fetch_add(1, Ordering::SeqCst);
                    async { "actual protected route" }
                }),
            )
            .layer(middleware::from_fn_with_state(keys, crate::auth::enforce))
            .layer(middleware::from_fn_with_state(policy, enforce))
            .layer(axum::Extension(CorsListeningAddress(
                "127.0.0.1:8080".parse().unwrap(),
            )))
    }
    fn request(method: &str, origin: Option<&str>) -> axum::http::request::Builder {
        let request = axum::http::Request::builder()
            .uri("/completion")
            .method(method)
            .header(header::HOST, "127.0.0.1:8080");
        match origin {
            Some(origin) => request.header(header::ORIGIN, origin),
            None => request,
        }
    }
    #[tokio::test]
    async fn cors_preflight_skips_auth_and_route_but_actual_requests_remain_protected() {
        let calls = Arc::new(AtomicUsize::new(0));
        let router = router(CorsPolicy::default(), calls.clone());
        let preflight = router
            .clone()
            .oneshot(
                request("OPTIONS", Some("http://localhost:9999"))
                    .header(header::ACCESS_CONTROL_REQUEST_METHOD, "POST")
                    .header(
                        header::ACCESS_CONTROL_REQUEST_HEADERS,
                        "authorization, content-type",
                    )
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(preflight.status(), StatusCode::OK);
        assert_eq!(
            preflight.headers()[header::ACCESS_CONTROL_ALLOW_ORIGIN],
            "http://localhost:9999"
        );
        assert_eq!(
            preflight.headers()[header::ACCESS_CONTROL_ALLOW_HEADERS],
            "authorization, content-type"
        );
        assert_eq!(
            preflight.headers()[header::ACCESS_CONTROL_ALLOW_CREDENTIALS],
            "true"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        for (key, status) in [
            (None, StatusCode::UNAUTHORIZED),
            (Some("Bearer test-key"), StatusCode::OK),
        ] {
            let mut req = request("POST", Some("http://localhost:9999"));
            if let Some(key) = key {
                req = req.header(header::AUTHORIZATION, key);
            }
            let response = router
                .clone()
                .oneshot(req.body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), status);
            assert_eq!(
                response.headers()[header::ACCESS_CONTROL_ALLOW_ORIGIN],
                "http://localhost:9999"
            );
            assert!(
                response
                    .headers()
                    .get_all(header::VARY)
                    .iter()
                    .any(|v| v == "Origin")
            );
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
    #[tokio::test]
    async fn cors_restricted_origins_host_and_fetch_site_reject_before_execution() {
        let calls = Arc::new(AtomicUsize::new(0));
        let policy = CorsPolicy::new(
            "https://one.example, https://two.example",
            "POST",
            "content-type, authorization",
            true,
        )
        .unwrap();
        let router = router(policy, calls.clone());
        for origin in [
            "https://one.example",
            "https://two.example",
            "http://127.0.0.1:8080",
        ] {
            let response = router
                .clone()
                .oneshot(
                    request("POST", Some(origin))
                        .header(header::AUTHORIZATION, "Bearer test-key")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(
                response.headers()[header::ACCESS_CONTROL_ALLOW_ORIGIN],
                origin
            );
        }
        for method in ["POST", "OPTIONS"] {
            let response = router
                .clone()
                .oneshot(
                    request(method, Some("https://evil.example"))
                        .header(header::AUTHORIZATION, "Bearer test-key")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::FORBIDDEN);
            assert!(
                !response
                    .headers()
                    .contains_key(header::ACCESS_CONTROL_ALLOW_ORIGIN)
            );
            let body = to_bytes(response.into_body(), 4096).await.unwrap();
            assert!(String::from_utf8_lossy(&body).contains("--cors-origins"));
        }
        for malicious_host in [false, true] {
            let mut req = request("POST", None)
                .header(header::AUTHORIZATION, "Bearer test-key")
                .body(Body::empty())
                .unwrap();
            if malicious_host {
                req.headers_mut().insert(
                    header::HOST,
                    HeaderValue::from_static("rebinding.example:8080"),
                );
            } else {
                req.headers_mut()
                    .insert("sec-fetch-site", HeaderValue::from_static("cross-site"));
            }
            let response = router.clone().oneshot(req).await.unwrap();
            assert_eq!(response.status(), StatusCode::FORBIDDEN);
        }
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }
    #[tokio::test]
    async fn cors_same_origin_and_noncredentialed_wildcard_are_distinct() {
        for (policy, origin, status, expected) in [
            (
                CorsPolicy::new("localhost", "POST", "*", true).unwrap(),
                "http://127.0.0.1:8080",
                StatusCode::OK,
                None,
            ),
            (
                CorsPolicy::new("localhost", "POST", "*", true).unwrap(),
                "http://127.0.0.1:8081",
                StatusCode::FORBIDDEN,
                None,
            ),
            (
                CorsPolicy::new("*", "POST", "x-test", false).unwrap(),
                "http://other.example",
                StatusCode::OK,
                Some("*"),
            ),
        ] {
            let router = router(policy, Arc::new(AtomicUsize::new(0)));
            let response = router
                .oneshot(
                    request("OPTIONS", Some(origin))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), status);
            assert_eq!(
                response
                    .headers()
                    .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
                    .and_then(|v| v.to_str().ok()),
                expected
            );
            if expected == Some("*") {
                assert!(
                    !response
                        .headers()
                        .contains_key(header::ACCESS_CONTROL_ALLOW_CREDENTIALS)
                );
                assert_eq!(
                    response.headers()[header::ACCESS_CONTROL_ALLOW_HEADERS],
                    "x-test"
                );
                assert_eq!(
                    response.headers()[header::ACCESS_CONTROL_ALLOW_METHODS],
                    "POST"
                );
            }
        }
    }
    #[tokio::test]
    async fn cors_mapped_loopback_listener_rejects_rebinding_without_restricting_lan() {
        let calls = Arc::new(AtomicUsize::new(0));
        for (local, host, origin, expected) in [
            (
                "[::ffff:127.0.0.1]:8080",
                "rebinding.example:8080",
                "http://rebinding.example:8080",
                StatusCode::FORBIDDEN,
            ),
            (
                "[::ffff:127.0.0.2]:8080",
                "rebinding.example:8080",
                "http://rebinding.example:8080",
                StatusCode::FORBIDDEN,
            ),
            (
                "[::ffff:127.0.0.1]:8080",
                "localhost:8080",
                "http://localhost:8080",
                StatusCode::OK,
            ),
            (
                "0.0.0.0:8080",
                "proxy.example:8080",
                "http://proxy.example:8080",
                StatusCode::OK,
            ),
            (
                "[::]:8080",
                "proxy.example:8080",
                "http://proxy.example:8080",
                StatusCode::OK,
            ),
            (
                "192.0.2.1:8080",
                "proxy.example:8080",
                "http://proxy.example:8080",
                StatusCode::OK,
            ),
            (
                "[::ffff:192.0.2.1]:8080",
                "proxy.example:8080",
                "http://proxy.example:8080",
                StatusCode::OK,
            ),
        ] {
            let counter = calls.clone();
            let router = Router::new()
                .route(
                    "/completion",
                    post(move || {
                        counter.fetch_add(1, Ordering::SeqCst);
                        async { "executed" }
                    }),
                )
                .layer(middleware::from_fn_with_state(
                    CorsPolicy::new("localhost", "POST", "*", true).unwrap(),
                    enforce,
                ))
                .layer(axum::Extension(CorsListeningAddress(
                    local.parse().unwrap(),
                )));
            let before = calls.load(Ordering::SeqCst);
            let mut req = request("POST", Some(origin)).body(Body::empty()).unwrap();
            req.headers_mut()
                .insert(header::HOST, HeaderValue::from_str(host).unwrap());
            let response = router.oneshot(req).await.unwrap();
            assert_eq!(response.status(), expected, "listener {local}, host {host}");
            assert_eq!(
                calls.load(Ordering::SeqCst) - before,
                usize::from(expected == StatusCode::OK),
                "denied rebinding must not enter handler"
            );
        }
    }
    #[tokio::test]
    async fn cors_wildcard_tcp_listener_uses_each_accepted_socket_local_address() {
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = calls.clone();
        let router = Router::new()
            .route(
                "/probe",
                post(move || {
                    observed.fetch_add(1, Ordering::SeqCst);
                    async { "accepted protected origin" }
                }),
            )
            .layer(middleware::from_fn_with_state(
                CorsPolicy::new("https://frontend.example", "POST", "*", true).unwrap(),
                enforce,
            ))
            // The accepted-socket context must also take precedence over an adapter's
            // unrelated/non-loopback extension, never weaken it through fallback.
            .layer(axum::Extension(CorsListeningAddress(
                "192.0.2.1:8080".parse().unwrap(),
            )));
        let listener = tokio::net::TcpListener::bind("0.0.0.0:0").await.unwrap();
        let bound = listener.local_addr().unwrap();
        assert!(bound.ip().is_unspecified());
        let port = bound.port();
        let server = tokio::spawn(async move {
            crate::serve_listener(listener, router).await.unwrap();
        });
        let client = reqwest::Client::builder()
            .no_proxy()
            .timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap();
        let url = format!("http://127.0.0.1:{port}/probe");
        // Same Origin/Host would pass the ordinary Origin guard. Only the real
        // accepted loopback socket's Host guard rejects this rebinding request.
        let bad_host = format!("rebind.example:{port}");
        let rejected = client
            .post(&url)
            .header(header::HOST, &bad_host)
            .header(header::ORIGIN, format!("http://{bad_host}"))
            .send()
            .await
            .unwrap();
        assert_eq!(rejected.status(), StatusCode::FORBIDDEN);
        assert!(
            !rejected
                .headers()
                .contains_key(header::ACCESS_CONTROL_ALLOW_ORIGIN)
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        let accepted = client
            .post(&url)
            .header(header::ORIGIN, "https://frontend.example")
            .send()
            .await
            .unwrap();
        assert_eq!(accepted.status(), StatusCode::OK);
        assert_eq!(
            accepted.headers()[header::ACCESS_CONTROL_ALLOW_ORIGIN],
            "https://frontend.example"
        );
        assert_eq!(accepted.text().await.unwrap(), "accepted protected origin");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        server.abort();
        assert!(server.await.unwrap_err().is_cancelled());
    }

    #[tokio::test]
    async fn cors_socket_context_failure_fails_closed_and_nonloopback_does_not_gain_host_guard() {
        let calls = Arc::new(AtomicUsize::new(0));
        let router = router(
            CorsPolicy::new("localhost", "POST", "*", true).unwrap(),
            calls.clone(),
        );
        for (context, expected) in [
            (
                CorsConnectionInfo(Err(Arc::from(
                    "resolve accepted TCP socket local address failed; verify the connection and retry",
                ))),
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
            (
                CorsConnectionInfo(Ok("192.0.2.2:8080".parse().unwrap())),
                StatusCode::OK,
            ),
        ] {
            let mut req = request("POST", Some("http://rebind.example:8080"))
                .header(header::AUTHORIZATION, "Bearer test-key")
                .body(Body::empty())
                .unwrap();
            req.headers_mut().insert(
                header::HOST,
                HeaderValue::from_static("rebind.example:8080"),
            );
            req.extensions_mut()
                .insert(axum::extract::ConnectInfo(context));
            let response = router.clone().oneshot(req).await.unwrap();
            assert_eq!(response.status(), expected);
            if expected == StatusCode::INTERNAL_SERVER_ERROR {
                let body = to_bytes(response.into_body(), 4096).await.unwrap();
                let error = String::from_utf8_lossy(&body);
                assert!(
                    error.contains("accepted TCP socket local address") && error.contains("retry"),
                    "{error}"
                );
                assert_eq!(calls.load(Ordering::SeqCst), 0);
            }
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn cors_configuration_rejects_invalid_origins_methods_and_headers() {
        for origin in [
            "",
            "*,https://a.example",
            "https://a.example/",
            "https://a.example/path",
            "https://user@a.example",
            "ftp://a.example",
            "https://a.example?x",
            "https://a.example,",
            "https://a.example\n",
        ] {
            assert!(
                CorsPolicy::new(origin, "POST", "*", true).is_err(),
                "{origin:?}"
            );
        }
        for method in ["", "POST,", "bad method", "POST\r\n"] {
            assert!(
                CorsPolicy::new("*", method, "*", true).is_err(),
                "{method:?}"
            );
        }
        for header in ["", "authorization,", "bad header", "x-test\r\n"] {
            assert!(
                CorsPolicy::new("*", "POST", header, true).is_err(),
                "{header:?}"
            );
        }
        assert!(CorsPolicy::new("null", "POST", "*", true).is_ok());
    }
}

#[cfg(all(test, feature = "native-backend"))]
mod native_tests {
    use super::*;
    use axum::body::{Body, to_bytes};
    use onnx_genai_engine::{EngineConfig, EngineDecodeBackend};
    use serde_json::{Value, json};
    use tower::ServiceExt;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cors_native_json_sse_errors_and_recovery_preserve_auth() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm");
        let state = crate::AppState::load_with_config(
            &root,
            Some("cors-native".into()),
            crate::ServerConfig {
                api_keys: crate::ApiKeys::from_keys(["native-key"]).unwrap(),
                cors: CorsPolicy::new(
                    "http://127.0.0.1:9999",
                    "GET, POST, DELETE, OPTIONS",
                    "*",
                    true,
                )
                .unwrap(),
                engine_config: EngineConfig {
                    decode_backend: EngineDecodeBackend::Native,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();
        let router = crate::app(state).layer(axum::Extension(CorsListeningAddress(
            "127.0.0.1:8080".parse().unwrap(),
        )));
        for (origin, key, body, expected) in [
            (
                "http://127.0.0.1:9999",
                None,
                "{}",
                StatusCode::UNAUTHORIZED,
            ),
            (
                "http://127.0.0.1:9999",
                Some("native-key"),
                "not JSON",
                StatusCode::BAD_REQUEST,
            ),
            (
                "http://127.0.0.1:9998",
                Some("native-key"),
                "{}",
                StatusCode::FORBIDDEN,
            ),
        ] {
            let mut request = axum::http::Request::builder()
                .uri("/v1/completions")
                .method("POST")
                .header(header::HOST, "127.0.0.1:8080")
                .header(header::ORIGIN, origin)
                .header(header::CONTENT_TYPE, "application/json");
            if let Some(key) = key {
                request = request.header("x-api-key", key);
            }
            let response = router
                .clone()
                .oneshot(request.body(Body::from(body)).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), expected);
            assert_eq!(
                response
                    .headers()
                    .contains_key(header::ACCESS_CONTROL_ALLOW_ORIGIN),
                expected != StatusCode::FORBIDDEN
            );
        }
        let mut text = None;
        for stream in [false, true] {
            let response = router.clone().oneshot(axum::http::Request::builder().uri("/v1/completions").method("POST")
                .header(header::HOST, "127.0.0.1:8080").header(header::ORIGIN, "http://127.0.0.1:9999")
                .header("x-api-key", "native-key").header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(json!({"prompt":"hello","temperature":0,"logit_bias":{"5":1000},"max_tokens":2,"stream":stream}).to_string())).unwrap()).await.unwrap();
            assert_eq!(
                response.headers()[header::ACCESS_CONTROL_ALLOW_ORIGIN],
                "http://127.0.0.1:9999"
            );
            assert_eq!(
                response.headers()[header::ACCESS_CONTROL_ALLOW_CREDENTIALS],
                "true"
            );
            let status = response.status();
            let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
            assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
            if stream {
                let wire = String::from_utf8(body.to_vec()).unwrap();
                assert!(wire.contains("data: [DONE]"), "{wire}");
                let parts: Vec<Value> = wire
                    .lines()
                    .filter_map(|line| line.strip_prefix("data: "))
                    .filter(|data| *data != "[DONE]")
                    .map(|data| serde_json::from_str(data).unwrap())
                    .collect();
                let streamed: String = parts
                    .iter()
                    .filter_map(|part| part["choices"][0]["text"].as_str())
                    .collect();
                assert_eq!(Some(streamed), text);
                assert!(
                    parts
                        .iter()
                        .any(|part| part["choices"][0]["finish_reason"].as_str().is_some())
                );
            } else {
                let result: Value = serde_json::from_slice(&body).unwrap();
                let output = result["choices"][0]["text"].as_str().unwrap();
                assert!(!output.is_empty(), "{result}");
                assert_eq!(result["usage"]["completion_tokens"], 2);
                text = Some(output.to_owned());
            }
        }
    }
}

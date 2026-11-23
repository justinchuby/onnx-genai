//! Explicit, bounded HTTP transport for the official frontend's MCP clients.
use super::ApiError;
use crate::{AppState, McpProxyPolicy};
use axum::{
    body::{Body, Bytes},
    extract::{Query, Request, State},
    http::{HeaderMap, HeaderName, HeaderValue, StatusCode, header},
    response::Response,
};
use reqwest::Url;
use serde::Deserialize;
use std::{
    collections::HashSet,
    future::Future,
    io,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    pin::Pin,
    task::{Context, Poll},
    time::Duration,
};
use tokio::{
    sync::mpsc,
    time::{Instant, timeout, timeout_at},
};
use tokio_stream::{StreamExt as _, wrappers::ReceiverStream};

const FORWARD_PREFIX: &str = "x-llama-server-proxy-header-";

#[derive(Deserialize)]
pub(crate) struct ProxyQuery {
    url: String,
    /// Redirects crossing target origins must not retain explicit credentials
    /// merely because the browser is still contacting this proxy's origin.
    #[serde(default = "forward_by_default")]
    forward_headers: bool,
}
fn forward_by_default() -> bool {
    true
}

fn status_error(status: StatusCode, message: impl Into<String>) -> ApiError {
    let mut error = ApiError::invalid_request(message);
    error.status = status;
    if status.is_server_error() {
        error.kind = "upstream_error";
    } else if status == StatusCode::FORBIDDEN {
        error.kind = "feature_disabled";
    }
    error
}

fn target_url(raw: &str) -> Result<Url, ApiError> {
    let mut url = Url::parse(raw).map_err(|error| {
        ApiError::invalid_request(format!(
            "invalid /cors-proxy url: {error}; supply an absolute HTTP(S) target URL"
        ))
    })?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err(ApiError::invalid_request(
            "/cors-proxy url requires an HTTP(S) scheme and host",
        ));
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(ApiError::invalid_request(
            "/cors-proxy url must not contain authentication; supply explicit x-llama-server-proxy-header-authorization instead",
        ));
    }
    url.set_fragment(None);
    Ok(url)
}

fn public_v4(ip: Ipv4Addr) -> bool {
    let [a, b, c, _] = ip.octets();
    !(a == 0
        || a == 10
        || a == 127
        || a >= 224
        || (a == 100 && b & 0xc0 == 64)
        || (a == 169 && b == 254)
        || (a == 172 && b & 0xf0 == 16)
        || (a == 192 && b == 168)
        || (a == 192 && b == 0 && c == 0)
        || (a == 198 && b & 0xfe == 18)
        || ip.is_documentation())
}

fn public_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => public_v4(ip),
        IpAddr::V6(ip) => {
            if let Some(v4) = ip.to_ipv4_mapped() {
                return public_v4(v4);
            }
            let bytes = ip.octets();
            // The standard NAT64 prefix retains the embedded IPv4 scope.
            if bytes[..12] == [0, 0x64, 0xff, 0x9b, 0, 0, 0, 0, 0, 0, 0, 0] {
                return public_v4(Ipv4Addr::new(bytes[12], bytes[13], bytes[14], bytes[15]));
            }
            bytes[..12] != [0; 12]
                && bytes[0] & 0xfe != 0xfc
                && !(bytes[0] == 0xfe && bytes[1] & 0x80 == 0x80)
                && bytes[0] != 0xff
                && bytes[..4] != [0x20, 0x01, 0x0d, 0xb8]
        }
    }
}

fn checked_addresses(
    host: &str,
    addresses: Vec<SocketAddr>,
    policy: McpProxyPolicy,
) -> Result<Vec<SocketAddr>, ApiError> {
    if addresses.is_empty() {
        return Err(ApiError::invalid_request(format!(
            "/cors-proxy url host {host:?} resolved to no addresses; check its DNS configuration"
        )));
    }
    if policy != McpProxyPolicy::AllowPrivate
        && addresses.iter().any(|address| !public_ip(address.ip()))
    {
        return Err(ApiError::invalid_request(format!(
            "/cors-proxy url host {host:?} resolves to a non-public address; use a public target or explicitly enable --ui-mcp-proxy-allow-private-urls"
        )));
    }
    Ok(addresses)
}

fn hop_headers(headers: &HeaderMap) -> HashSet<HeaderName> {
    let mut names = [
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    ]
    .into_iter()
    .map(HeaderName::from_static)
    .collect::<HashSet<_>>();
    for value in headers.get_all(header::CONNECTION) {
        if let Ok(value) = value.to_str() {
            for name in value.split(',') {
                if let Ok(name) = HeaderName::from_bytes(name.trim().as_bytes()) {
                    names.insert(name);
                }
            }
        }
    }
    names
}

fn request_headers(headers: &HeaderMap, forward: bool) -> Result<HeaderMap, ApiError> {
    let mut explicit = HeaderMap::new();
    for (name, value) in headers {
        let Some(name) = name.as_str().strip_prefix(FORWARD_PREFIX) else {
            continue;
        };
        if name.is_empty() {
            continue;
        }
        let name = HeaderName::from_bytes(name.as_bytes()).map_err(|error| ApiError::invalid_request(format!(
            "invalid /cors-proxy forwarded header {name:?}: {error}; specify a valid HTTP header name after {FORWARD_PREFIX}")))?;
        explicit.append(name, value.clone());
    }
    let blocked = hop_headers(&explicit);
    let mut filtered = HeaderMap::new();
    for (name, value) in &explicit {
        if !blocked.contains(name)
            && name != header::HOST
            && name != header::CONTENT_LENGTH
            && (forward
                || matches!(
                    name.as_str(),
                    "accept" | "content-type" | "content-encoding"
                ))
        {
            filtered.append(name, value.clone());
        }
    }
    Ok(filtered)
}

fn response_headers(headers: &HeaderMap) -> HeaderMap {
    let blocked = hop_headers(headers);
    let mut result = HeaderMap::new();
    for (name, value) in headers {
        // The target controls its payload, not this server origin's browser
        // credentials, storage, CORS grants or alternative transport policy.
        let origin_policy = name.as_str().starts_with("access-control-")
            || matches!(
                name.as_str(),
                "set-cookie"
                    | "clear-site-data"
                    | "strict-transport-security"
                    | "alt-svc"
                    | "refresh"
                    | "service-worker-allowed"
                    | "content-security-policy-report-only"
            );
        if !blocked.contains(name) && !origin_policy {
            result.append(name, value.clone());
        }
    }
    // A remote HTML document must not execute with the inference server's origin.
    result.insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static("sandbox; default-src 'none'"),
    );
    result.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    result
}

fn redirect_location(
    source: &Url,
    location: &HeaderValue,
    forward: bool,
) -> Result<HeaderValue, ApiError> {
    let location = location.to_str().map_err(|error| {
        status_error(
            StatusCode::BAD_GATEWAY,
            format!("MCP target returned an invalid Location header: {error}; repair its redirect"),
        )
    })?;
    let next = source.join(location).map_err(|error| {
        status_error(
            StatusCode::BAD_GATEWAY,
            format!(
                "MCP target returned an invalid redirect URL: {error}; repair its Location header"
            ),
        )
    })?;
    let next = target_url(next.as_str()).map_err(|mut error| {
        error.status = StatusCode::BAD_GATEWAY;
        error.kind = "upstream_error";
        error.message = format!(
            "MCP target returned an invalid redirect: {}; repair its Location header",
            error.message
        );
        error
    })?;
    let mut encoded = Url::parse("http://proxy.invalid/").expect("static proxy query encoder URL");
    encoded.query_pairs_mut().append_pair("url", next.as_str());
    if !forward || source.origin() != next.origin() {
        encoded
            .query_pairs_mut()
            .append_pair("forward_headers", "false");
    }
    HeaderValue::from_str(&format!("?{}", encoded.query().unwrap_or_default())).map_err(|error| {
        status_error(
            StatusCode::BAD_GATEWAY,
            format!("cannot encode MCP redirect: {error}; shorten the target Location header"),
        )
    })
}

pub(crate) async fn forward(
    State(state): State<AppState>,
    query: Result<Query<ProxyQuery>, axum::extract::rejection::QueryRejection>,
    request: Request,
) -> Result<Response, ApiError> {
    let config = &state.config;
    if !config.mcp_proxy_policy.is_enabled() {
        return Err(status_error(
            StatusCode::FORBIDDEN,
            "/cors-proxy is disabled; explicitly enable --ui-mcp-proxy to proxy frontend MCP requests",
        ));
    }
    let connection = state.proxy_connections.clone().try_acquire_owned().map_err(|_| {
        let mut error = ApiError::unavailable("MCP proxy connection limit reached; retry when an active transfer finishes or configure --ui-mcp-proxy-max-connections");
        error.retry_after_secs = Some(1);
        error
    })?;
    let Query(query) = query.map_err(|error| {
        ApiError::invalid_request(format!(
            "invalid /cors-proxy query: {error}; provide url=<absolute HTTP(S) URL>"
        ))
    })?;
    let url = target_url(&query.url)?;
    let host = url
        .host_str()
        .expect("validated target host")
        .trim_matches(['[', ']']);
    let port = url.port_or_known_default().expect("HTTP(S) target port");
    let idle_timeout = Duration::from_secs(config.mcp_proxy_timeout_secs.get());
    let deadline = Instant::now().checked_add(idle_timeout)
        .ok_or_else(|| ApiError::invalid_request("--ui-mcp-proxy-timeout-secs exceeds the host clock range; configure a smaller transfer timeout"))?;
    let resolve_deadline = deadline.min(Instant::now() + Duration::from_secs(10));
    let addresses = if let Ok(ip) = host.parse::<IpAddr>() {
        vec![SocketAddr::new(ip, port)]
    } else {
        timeout_at(resolve_deadline, tokio::net::lookup_host((host, port)))
            .await
            .map_err(|_| {
                status_error(
                    StatusCode::GATEWAY_TIMEOUT,
                    format!(
                        "DNS lookup for MCP url host {host:?} timed out; check its DNS service"
                    ),
                )
            })?
            .map_err(|error| {
                ApiError::invalid_request(format!(
                    "cannot resolve MCP url host {host:?}: {error}; check its DNS configuration"
                ))
            })?
            .collect()
    };
    let addresses = checked_addresses(host, addresses, config.mcp_proxy_policy)?;
    let headers = request_headers(request.headers(), query.forward_headers)?;
    let method = request.method().clone();
    let limit = config.mcp_proxy_max_bytes.get();
    let mut incoming = request.into_body().into_data_stream();
    let mut body = Vec::new();
    while let Some(chunk) = timeout(idle_timeout,incoming.next()).await
        .map_err(|_| status_error(StatusCode::REQUEST_TIMEOUT,"reading /cors-proxy request body made no progress; check the sender or configure --ui-mcp-proxy-timeout-secs"))? {
        let chunk = chunk.map_err(|error| ApiError::invalid_request(format!("cannot read /cors-proxy request body: {:#}; resend a complete HTTP body",anyhow::Error::new(error))))?;
        if chunk.len() > limit.saturating_sub(body.len()) {
            return Err(status_error(StatusCode::PAYLOAD_TOO_LARGE,format!("/cors-proxy request body exceeds its {limit}-byte limit; reduce the body or configure --ui-mcp-proxy-max-bytes")));
        }
        body.extend_from_slice(&chunk);
    }
    // Disable environment proxies and automatic redirects: only these inspected
    // DNS results can be contacted, and every subsequent hop returns to admission.
    let client = reqwest::Client::builder()
        .no_proxy()
        .no_gzip()
        .no_brotli()
        .no_deflate()
        .no_zstd()
        .redirect(reqwest::redirect::Policy::none())
        .resolve_to_addrs(host, &addresses)
        .connect_timeout(idle_timeout.min(Duration::from_secs(10)))
        .read_timeout(idle_timeout)
        .build()
        .map_err(|error| {
            ApiError::internal(format!(
                "construct MCP HTTP transport: {error}; check TLS/runtime configuration"
            ))
        })?;
    let mut upstream = timeout(idle_timeout,client.request(method,url.clone()).headers(headers).body(body).send()).await
        .map_err(|_| status_error(StatusCode::GATEWAY_TIMEOUT,format!("MCP HTTP request to {host}:{port} timed out before response headers; check the target or configure --ui-mcp-proxy-timeout-secs")))?
        .map_err(|error| status_error(if error.is_timeout() { StatusCode::GATEWAY_TIMEOUT } else { StatusCode::BAD_GATEWAY },
            format!("MCP HTTP request to {host}:{port} failed: {:#}; check target reachability and TLS configuration",anyhow::Error::new(error.without_url()))))?;
    if upstream
        .content_length()
        .is_some_and(|length| length > limit as u64)
    {
        return Err(status_error(
            StatusCode::BAD_GATEWAY,
            format!(
                "MCP response exceeds the {limit}-byte proxy limit; reduce the response or configure --ui-mcp-proxy-max-bytes"
            ),
        ));
    }
    let status = upstream.status();
    let mut headers = response_headers(upstream.headers());
    if status.is_redirection()
        && let Some(location) = headers.get(header::LOCATION)
    {
        headers.insert(
            header::LOCATION,
            redirect_location(&url, location, query.forward_headers)?,
        );
    }
    let (sender, receiver) = mpsc::channel(1);
    let transfer = tokio::spawn(async move {
        let _connection = connection;
        let mut received = 0usize;
        loop {
            let chunk = tokio::select! {
                _ = sender.closed() => return,
                chunk = timeout(idle_timeout,upstream.chunk()) => chunk,
            };
            let chunk = match chunk {
                Ok(Ok(Some(bytes))) if bytes.len() <= limit.saturating_sub(received) => {
                    received += bytes.len();
                    bytes
                }
                Ok(Ok(None)) => return,
                result => {
                    let message = match result {
                        Ok(Ok(Some(_))) => format!("MCP response exceeded the {limit}-byte proxy limit; configure --ui-mcp-proxy-max-bytes or reduce target output"),
                        Ok(Err(error)) => format!("read MCP response body: {:#}; check target availability",anyhow::Error::new(error.without_url())),
                        Err(_) => "MCP response transfer timed out; configure --ui-mcp-proxy-timeout-secs or check the target".into(),
                        Ok(Ok(None)) => unreachable!(),
                    };
                    let _ =
                        timeout(idle_timeout, sender.send(Err(io::Error::other(message)))).await;
                    return;
                }
            };
            if !matches!(
                timeout(idle_timeout, sender.send(Ok(chunk))).await,
                Ok(Ok(()))
            ) {
                return;
            }
        }
    });
    let mut response = Response::new(Body::from_stream(ProxyStream {
        receiver: ReceiverStream::new(receiver),
        transfer,
        deadline: Box::pin(tokio::time::sleep(idle_timeout)),
        idle_timeout,
        ended: false,
    }));
    *response.status_mut() = status;
    *response.headers_mut() = headers;
    Ok(response)
}

/// Own the transfer lifetime as well as its bounded queue. A stalled consumer
/// cannot keep an upstream connection alive past the configured deadline, and
/// a later read observes a transfer error rather than a successful partial EOF.
struct ProxyStream {
    receiver: ReceiverStream<Result<Bytes, io::Error>>,
    transfer: tokio::task::JoinHandle<()>,
    deadline: Pin<Box<tokio::time::Sleep>>,
    idle_timeout: Duration,
    ended: bool,
}
impl tokio_stream::Stream for ProxyStream {
    type Item = Result<Bytes, io::Error>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.ended {
            return Poll::Ready(None);
        }
        if self.deadline.as_mut().poll(cx).is_ready() {
            self.ended = true;
            self.transfer.abort();
            return Poll::Ready(Some(Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "MCP response transfer timed out; configure --ui-mcp-proxy-timeout-secs or check the target",
            ))));
        }
        let result = Pin::new(&mut self.receiver).poll_next(cx);
        if matches!(result, Poll::Ready(Some(Ok(_)))) {
            let until = Instant::now()
                .checked_add(self.idle_timeout)
                .unwrap_or_else(Instant::now);
            self.deadline.as_mut().reset(until);
        }
        if matches!(result, Poll::Ready(None | Some(Err(_)))) {
            self.ended = true;
        }
        result
    }
}
impl Drop for ProxyStream {
    fn drop(&mut self) {
        self.transfer.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ModelSpec, ServerConfig, app};
    use axum::{
        Router,
        body::to_bytes,
        http::Method,
        routing::{get, post},
    };
    use std::{
        num::NonZeroUsize,
        path::PathBuf,
        sync::{Arc, Mutex},
    };
    use tower::ServiceExt;

    fn state(policy: McpProxyPolicy, bytes: usize) -> AppState {
        AppState::load_from_specs(
            vec![ModelSpec {
                id: "unloaded".into(),
                path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("../../tests/fixtures/tiny-llm"),
                eager: false,
                warmup: false,
            }],
            ServerConfig {
                mcp_proxy_policy: policy,
                mcp_proxy_max_bytes: NonZeroUsize::new(bytes).unwrap(),
                ..Default::default()
            },
        )
        .unwrap()
    }
    fn uri(target: &str) -> String {
        let mut encoder = Url::parse("http://proxy.invalid/cors-proxy").unwrap();
        encoder.query_pairs_mut().append_pair("url", target);
        format!("/cors-proxy?{}", encoder.query().unwrap())
    }
    async fn local(router: Router) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let task = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        (format!("http://{address}"), task)
    }

    #[test]
    fn mcp_proxy_checks_every_resolved_address_and_embedded_ipv4_scope() {
        for value in [
            "0.0.0.0",
            "10.1.2.3",
            "127.0.0.1",
            "100.64.0.1",
            "169.254.169.254",
            "172.16.0.1",
            "192.168.1.1",
            "192.0.0.1",
            "198.18.0.1",
            "224.0.0.1",
            "192.0.2.1",
            "::",
            "::1",
            "::127.0.0.1",
            "::ffff:127.0.0.1",
            "64:ff9b::a00:1",
            "fc00::1",
            "fe80::1",
            "fec0::1",
            "ff02::1",
            "2001:db8::1",
        ] {
            let ip: IpAddr = value.parse().unwrap();
            assert!(!public_ip(ip), "{value}");
            assert!(
                checked_addresses(
                    "target",
                    vec![SocketAddr::new(ip, 80)],
                    McpProxyPolicy::Public
                )
                .is_err()
            );
            assert!(
                checked_addresses(
                    "target",
                    vec![SocketAddr::new(ip, 80)],
                    McpProxyPolicy::AllowPrivate
                )
                .is_ok()
            );
        }
        for value in [
            "1.1.1.1",
            "8.8.8.8",
            "2606:4700:4700::1111",
            "::ffff:8.8.8.8",
            "64:ff9b::808:808",
        ] {
            assert!(public_ip(value.parse().unwrap()), "{value}");
        }
        assert!(
            checked_addresses(
                "mixed",
                vec![
                    "8.8.8.8:80".parse().unwrap(),
                    "127.0.0.1:80".parse().unwrap()
                ],
                McpProxyPolicy::Public
            )
            .is_err()
        );
        assert!(checked_addresses("empty", Vec::new(), McpProxyPolicy::Public).is_err());
    }

    #[test]
    fn mcp_proxy_rejects_invalid_targets_and_isolates_redirect_credentials() {
        for value in [
            "/relative",
            "file:///etc/passwd",
            "ftp://example.com/",
            "http://user@example.com/",
            "http://user:secret@example.com/",
            "http://[fe80::1%25en0]/",
        ] {
            assert!(target_url(value).is_err(), "{value}");
        }
        let source = target_url("https://example.com/start").unwrap();
        let same = redirect_location(&source, &HeaderValue::from_static("/next"), true).unwrap();
        assert_eq!(same, "?url=https%3A%2F%2Fexample.com%2Fnext");
        let other = redirect_location(
            &source,
            &HeaderValue::from_static("https://other.example/next"),
            true,
        )
        .unwrap();
        assert!(other.to_str().unwrap().ends_with("&forward_headers=false"));
        let again = redirect_location(&source, &HeaderValue::from_static("/next"), false).unwrap();
        assert!(again.to_str().unwrap().ends_with("&forward_headers=false"));
        assert!(
            redirect_location(
                &source,
                &HeaderValue::from_static("file:///etc/passwd"),
                true
            )
            .is_err()
        );
    }

    #[tokio::test]
    async fn mcp_proxy_disabled_and_public_only_admission_do_not_contact_private_targets() {
        let disabled = app(state(McpProxyPolicy::Disabled, 4096))
            .oneshot(Request::get("/cors-proxy").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(disabled.status(), StatusCode::FORBIDDEN);
        let router = app(state(McpProxyPolicy::Public, 4096));
        for value in [
            "http://127.0.0.1:9/",
            "http://[::1]:9/",
            "http://[::ffff:127.0.0.1]:9/",
            "http://localhost:9/",
        ] {
            let response = router
                .clone()
                .oneshot(Request::get(uri(value)).body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{value}");
            let body = to_bytes(response.into_body(), 4096).await.unwrap();
            assert!(String::from_utf8_lossy(&body).contains("non-public address"));
        }
        let response = router
            .oneshot(
                Request::get("/props?autoload=false")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 4096).await.unwrap()).unwrap();
        assert_eq!(body["cors_proxy_enabled"], true);
    }

    #[tokio::test]
    async fn mcp_proxy_forwards_explicit_headers_body_status_and_relative_redirect() {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let record = captured.clone();
        let target = Router::new()
            .route(
                "/capture",
                post(move |request: Request| {
                    let record = record.clone();
                    async move {
                        let method = request.method().clone();
                        let headers = request.headers().clone();
                        let body = to_bytes(request.into_body(), 4096).await.unwrap();
                        record.lock().unwrap().push((method, headers, body));
                        (
                            StatusCode::CREATED,
                            [
                                ("content-type", "application/json"),
                                ("set-cookie", "remote=secret"),
                                ("connection", "x-hop"),
                                ("x-hop", "remove"),
                            ],
                            "{\"ok\":true}",
                        )
                    }
                }),
            )
            .route("/empty", post(|| async { StatusCode::NO_CONTENT }))
            .route(
                "/start",
                get(|| async { (StatusCode::FOUND, [("location", "/next")]) }),
            )
            .route("/next", get(|| async { "final" }));
        let (base, task) = local(target).await;
        let router = app(state(McpProxyPolicy::AllowPrivate, 4096));
        let response = router
            .clone()
            .oneshot(
                Request::post(uri(&format!("{base}/capture")))
                    .header("authorization", "Bearer server-secret")
                    .header("cookie", "server=secret")
                    .header("x-api-key", "server-secret")
                    .header("proxy-authorization", "Basic server-secret")
                    .header(
                        "x-llama-server-proxy-header-authorization",
                        "Bearer explicit",
                    )
                    .header(
                        "x-llama-server-proxy-header-content-type",
                        "application/json",
                    )
                    .header("x-llama-server-proxy-header-connection", "x-private-hop")
                    .header("x-llama-server-proxy-header-x-private-hop", "remove")
                    .body(Body::from("{\"method\":\"ping\"}"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::CREATED);
        assert_eq!(response.headers()[header::CONTENT_TYPE], "application/json");
        assert!(!response.headers().contains_key(header::SET_COOKIE));
        assert!(!response.headers().contains_key("x-hop"));
        assert_eq!(
            to_bytes(response.into_body(), 4096).await.unwrap(),
            "{\"ok\":true}"
        );
        {
            let captured = captured.lock().unwrap();
            let (method, headers, body) = &captured[0];
            assert_eq!(method, Method::POST);
            assert_eq!(body, "{\"method\":\"ping\"}");
            assert_eq!(headers[header::AUTHORIZATION], "Bearer explicit");
            assert_eq!(headers[header::CONTENT_TYPE], "application/json");
            for name in [
                "cookie",
                "x-api-key",
                "proxy-authorization",
                "x-private-hop",
                "connection",
            ] {
                assert!(!headers.contains_key(name), "{name}");
            }
        }
        let response = router
            .clone()
            .oneshot(
                Request::post(uri(&format!("{base}/empty")))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NO_CONTENT);
        assert!(
            to_bytes(response.into_body(), 4096)
                .await
                .unwrap()
                .is_empty()
        );
        let response = router
            .clone()
            .oneshot(
                Request::get(uri(&format!("{base}/start")))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::FOUND);
        let location = response.headers()[header::LOCATION].to_str().unwrap();
        assert_eq!(
            format!("/cors-proxy{location}"),
            uri(&format!("{base}/next"))
        );
        let response = router
            .oneshot(
                Request::get(format!("/cors-proxy{location}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(to_bytes(response.into_body(), 4096).await.unwrap(), "final");
        task.abort();
    }

    #[tokio::test]
    async fn mcp_proxy_rejects_request_and_response_byte_budget_overruns() {
        let (base, task) =
            local(Router::new().route("/large", get(|| async { "response-is-too-large" }))).await;
        let router = app(state(McpProxyPolicy::AllowPrivate, 8));
        let response = router
            .clone()
            .oneshot(
                Request::post(uri(&base))
                    .body(Body::from("too-large-body"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let response = router
            .oneshot(
                Request::get(uri(&format!("{base}/large")))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
        task.abort();
    }
    struct DropSignalStream {
        receiver: ReceiverStream<Result<Bytes, io::Error>>,
        dropped: Option<tokio::sync::oneshot::Sender<()>>,
    }
    impl tokio_stream::Stream for DropSignalStream {
        type Item = Result<Bytes, io::Error>;
        fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
            Pin::new(&mut self.receiver).poll_next(cx)
        }
    }
    impl Drop for DropSignalStream {
        fn drop(&mut self) {
            if let Some(sender) = self.dropped.take() {
                let _ = sender.send(());
            }
        }
    }
    async fn streaming_target() -> (
        String,
        tokio::task::JoinHandle<()>,
        mpsc::Sender<Result<Bytes, io::Error>>,
        tokio::sync::oneshot::Receiver<()>,
    ) {
        let (sender, receiver) = mpsc::channel(2);
        let (dropped, observed) = tokio::sync::oneshot::channel();
        let stream = Arc::new(Mutex::new(Some(DropSignalStream {
            receiver: ReceiverStream::new(receiver),
            dropped: Some(dropped),
        })));
        let target = Router::new().route(
            "/events",
            get(move || {
                let stream = stream.clone();
                async move {
                    let stream = stream.lock().unwrap().take().unwrap();
                    (
                        [(header::CONTENT_TYPE, "text/event-stream")],
                        Body::from_stream(stream),
                    )
                }
            }),
        );
        let (base, task) = local(target).await;
        (base, task, sender, observed)
    }

    #[tokio::test]
    async fn mcp_proxy_streams_before_upstream_eof_and_cancels_on_consumer_drop() {
        use tokio_stream::StreamExt as _;
        let (base, task, sender, dropped) = streaming_target().await;
        sender
            .send(Ok(Bytes::from_static(b"data: ping\n\n")))
            .await
            .unwrap();
        let response = tokio::time::timeout(
            Duration::from_secs(2),
            app(state(McpProxyPolicy::AllowPrivate, 4096)).oneshot(
                Request::get(uri(&format!("{base}/events")))
                    .body(Body::empty())
                    .unwrap(),
            ),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "text/event-stream"
        );
        let mut body = response.into_body().into_data_stream();
        let first = tokio::time::timeout(Duration::from_secs(2), body.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert_eq!(first, "data: ping\n\n");
        // The upstream sender is still live: this was not an eagerly buffered response.
        drop(body);
        tokio::time::timeout(Duration::from_secs(2), dropped)
            .await
            .unwrap()
            .unwrap();
        drop(sender);
        task.abort();
    }

    #[tokio::test]
    async fn mcp_proxy_unknown_length_limit_is_a_body_error() {
        let (base, task, sender, dropped) = streaming_target().await;
        sender
            .send(Ok(Bytes::from_static(b"123456")))
            .await
            .unwrap();
        sender
            .send(Ok(Bytes::from_static(b"789abc")))
            .await
            .unwrap();
        let response = app(state(McpProxyPolicy::AllowPrivate, 8))
            .oneshot(
                Request::get(uri(&format!("{base}/events")))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(!response.headers().contains_key(header::CONTENT_LENGTH));
        let error = to_bytes(response.into_body(), 4096).await.unwrap_err();
        assert!(error.to_string().contains("8-byte proxy limit"), "{error}");
        tokio::time::timeout(Duration::from_secs(2), dropped)
            .await
            .unwrap()
            .unwrap();
        drop(sender);
        task.abort();
    }

    #[tokio::test]
    async fn mcp_proxy_stalled_consumer_times_out_without_retaining_the_upstream() {
        let (base, task, sender, dropped) = streaming_target().await;
        sender
            .send(Ok(Bytes::from_static(b"data: pending\n\n")))
            .await
            .unwrap();
        let mut config = state(McpProxyPolicy::AllowPrivate, 4096);
        config.config.mcp_proxy_timeout_secs = std::num::NonZeroU64::new(1).unwrap();
        let response = app(config)
            .oneshot(
                Request::get(uri(&format!("{base}/events")))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        // Deliberately retain the body without polling its bounded queue.
        tokio::time::sleep(Duration::from_millis(1100)).await;
        tokio::time::timeout(Duration::from_secs(2), dropped)
            .await
            .unwrap()
            .unwrap();
        let error = to_bytes(response.into_body(), 4096).await.unwrap_err();
        assert!(error.to_string().contains("timed out"), "{error}");
        drop(sender);
        task.abort();
    }

    #[tokio::test]
    async fn mcp_proxy_cross_origin_redirect_drops_credentials_and_preserves_negotiation() {
        let seen = Arc::new(Mutex::new(None));
        let record = seen.clone();
        let (target, target_task) = local(Router::new().route(
            "/capture",
            get(move |headers: HeaderMap| {
                *record.lock().unwrap() = Some(headers);
                async { "ok" }
            }),
        ))
        .await;
        let (source, source_task) = local(Router::new().route(
            "/redirect",
            get(move || {
                let target = target.clone();
                async move {
                    (
                        StatusCode::TEMPORARY_REDIRECT,
                        [(header::LOCATION, format!("{target}/capture"))],
                    )
                }
            }),
        ))
        .await;
        let router = app(state(McpProxyPolicy::AllowPrivate, 4096));
        let response = router
            .clone()
            .oneshot(
                Request::get(uri(&format!("{source}/redirect")))
                    .header("x-llama-server-proxy-header-authorization", "Bearer secret")
                    .header("x-llama-server-proxy-header-x-unknown-secret", "secret")
                    .header("x-llama-server-proxy-header-accept", "text/event-stream")
                    .header(
                        "x-llama-server-proxy-header-content-type",
                        "application/json",
                    )
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::TEMPORARY_REDIRECT);
        let location = response.headers()[header::LOCATION].to_str().unwrap();
        assert!(location.ends_with("&forward_headers=false"));
        let response = router
            .oneshot(
                Request::get(format!("/cors-proxy{location}"))
                    .header("x-llama-server-proxy-header-authorization", "Bearer secret")
                    .header("x-llama-server-proxy-header-x-unknown-secret", "secret")
                    .header("x-llama-server-proxy-header-accept", "text/event-stream")
                    .header(
                        "x-llama-server-proxy-header-content-type",
                        "application/json",
                    )
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(to_bytes(response.into_body(), 4096).await.unwrap(), "ok");
        let headers = seen.lock().unwrap().take().unwrap();
        assert!(!headers.contains_key(header::AUTHORIZATION));
        assert!(!headers.contains_key("x-unknown-secret"));
        assert_eq!(headers[header::ACCEPT], "text/event-stream");
        assert_eq!(headers[header::CONTENT_TYPE], "application/json");
        source_task.abort();
        target_task.abort();
    }
    #[tokio::test]
    async fn mcp_proxy_connection_limit_is_released_on_disconnect() {
        let (base, task, sender, dropped) = streaming_target().await;
        let mut state = state(McpProxyPolicy::AllowPrivate, 4096);
        state.proxy_connections = Arc::new(tokio::sync::Semaphore::new(1));
        let permits = state.proxy_connections.clone();
        let router = app(state);
        let response = router
            .clone()
            .oneshot(
                Request::get(uri(&format!("{base}/events")))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(permits.available_permits(), 0);
        let overload = router
            .oneshot(
                Request::get(uri(&format!("{base}/events")))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(overload.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(overload.headers()[header::RETRY_AFTER], "1");
        drop(response);
        tokio::time::timeout(Duration::from_secs(2), dropped)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(permits.available_permits(), 1);
        drop(sender);
        task.abort();
    }

    #[test]
    fn mcp_proxy_hop_headers_and_redirect_negotiation_have_independent_rules() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-llama-server-proxy-header-connection",
            HeaderValue::from_static("accept"),
        );
        headers.insert(
            "x-llama-server-proxy-header-accept",
            HeaderValue::from_static("application/json"),
        );
        headers.insert(
            "x-llama-server-proxy-header-content-type",
            HeaderValue::from_static("application/json"),
        );
        headers.insert(
            "x-llama-server-proxy-header-authorization",
            HeaderValue::from_static("secret"),
        );
        let forwarded = request_headers(&headers, false).unwrap();
        assert!(!forwarded.contains_key(header::ACCEPT));
        assert!(!forwarded.contains_key(header::AUTHORIZATION));
        assert_eq!(forwarded[header::CONTENT_TYPE], "application/json");
    }
    #[test]
    fn mcp_proxy_remote_response_cannot_modify_server_origin_policy() {
        let mut remote = HeaderMap::new();
        for name in [
            "set-cookie",
            "clear-site-data",
            "strict-transport-security",
            "alt-svc",
            "refresh",
            "service-worker-allowed",
            "content-security-policy-report-only",
            "access-control-allow-origin",
            "access-control-allow-credentials",
            "access-control-expose-headers",
        ] {
            remote.insert(
                HeaderName::from_static(name),
                HeaderValue::from_static("untrusted"),
            );
        }
        remote.insert(
            "content-security-policy",
            HeaderValue::from_static("default-src *"),
        );
        remote.insert("mcp-session-id", HeaderValue::from_static("target-session"));
        remote.insert(
            header::WWW_AUTHENTICATE,
            HeaderValue::from_static("Bearer resource_metadata=example"),
        );
        let filtered = response_headers(&remote);
        for name in remote.keys() {
            if !matches!(
                name.as_str(),
                "content-security-policy" | "mcp-session-id" | "www-authenticate"
            ) {
                assert!(!filtered.contains_key(name), "{name}");
            }
        }
        assert_eq!(
            filtered[header::CONTENT_SECURITY_POLICY],
            "sandbox; default-src 'none'"
        );
        assert_eq!(filtered["mcp-session-id"], "target-session");
        assert_eq!(
            filtered[header::WWW_AUTHENTICATE],
            remote[header::WWW_AUTHENTICATE]
        );
        let mut request = HeaderMap::new();
        request.insert(
            "x-llama-server-proxy-header-",
            HeaderValue::from_static("ignored"),
        );
        assert!(request_headers(&request, true).unwrap().is_empty());
    }
    #[tokio::test]
    async fn mcp_proxy_active_stream_survives_past_the_idle_timeout() {
        let (base, task, sender, dropped) = streaming_target().await;
        sender
            .send(Ok(Bytes::from_static(b"data: ping\n\n")))
            .await
            .unwrap();
        let mut state = state(McpProxyPolicy::AllowPrivate, 4096);
        state.config.mcp_proxy_timeout_secs = std::num::NonZeroU64::new(2).unwrap();
        let response = app(state)
            .oneshot(
                Request::get(uri(&format!("{base}/events")))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let mut body = response.into_body().into_data_stream();
        assert_eq!(body.next().await.unwrap().unwrap(), "data: ping\n\n");
        let start = Instant::now();
        for _ in 0..3 {
            tokio::time::sleep(Duration::from_millis(800)).await;
            sender
                .send(Ok(Bytes::from_static(b"data: ping\n\n")))
                .await
                .unwrap();
            let chunk = tokio::time::timeout(Duration::from_secs(2), body.next())
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            assert_eq!(chunk, "data: ping\n\n");
        }
        assert!(start.elapsed() > Duration::from_secs(2));
        drop(sender);
        assert!(
            tokio::time::timeout(Duration::from_secs(2), body.next())
                .await
                .unwrap()
                .is_none()
        );
        tokio::time::timeout(Duration::from_secs(2), dropped)
            .await
            .unwrap()
            .unwrap();
        task.abort();
    }
}

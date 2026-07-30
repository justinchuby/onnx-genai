// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! Cross-origin access control for the demo dashboard.
//!
//! The dashboard is served from a single origin but polls BOTH demo servers (the
//! scatter server for continuous batching, the dynamic server for paged KV), so
//! every request to the second server is cross-origin. Without CORS the browser
//! blocks them, and because the load-driving requests are `POST` with
//! `Content-Type: application/json` — not a CORS-safelisted content type — they
//! fail at the `OPTIONS` preflight, before any response header matters.
//!
//! That failure mode is indistinguishable from "the other server isn't running"
//! from JavaScript's point of view, so without this the dashboard's connection
//! error state would confidently tell a visitor to start a server that is
//! already running.
//!
//! Hand-rolled rather than `tower-http`'s `CorsLayer` for the same reason as
//! [`crate::demo_assets`]: `tower-http` is not a dependency of this crate, and
//! the policy we need is narrow enough to state exactly.
//!
//! # Policy
//!
//! Loopback origins are reflected automatically, on any port. Anything else must
//! be named explicitly with `--cors-allow-origin`. A wildcard is never sent: the
//! `Origin` is echoed back, so the allowed set stays exactly what was asked for.

use axum::{
    body::Body,
    extract::State,
    http::{HeaderValue, Method, Request, StatusCode, header},
    middleware::Next,
    response::Response,
};

use crate::state::AppState;

/// Methods the API actually exposes.
const ALLOWED_METHODS: &str = "GET, POST, DELETE, OPTIONS";

/// Request headers the dashboard sends. `content-type` is the load-bearing one:
/// it is what makes the demo's JSON POSTs non-simple and forces a preflight.
const ALLOWED_HEADERS: &str = "content-type, authorization, x-session-id";

/// Preflight cache lifetime. Ten minutes keeps a 4 Hz polling dashboard from
/// re-preflighting constantly without pinning a stale policy for long.
const MAX_AGE_SECONDS: &str = "600";

/// Returns `true` if `origin` points at the local machine.
///
/// Parsed structurally rather than by prefix matching: a `starts_with`
/// check would accept `http://localhost.evil.example`, which is not loopback.
fn is_loopback_origin(origin: &str) -> bool {
    let rest = match origin.split_once("://") {
        Some(("http" | "https", rest)) => rest,
        _ => return false,
    };
    // Reject anything with a path, query or credentials — a well-formed Origin
    // has none of them, and their presence means this is not what it claims.
    if rest.contains('/') || rest.contains('@') || rest.contains('?') {
        return false;
    }
    let host = match rest.rsplit_once(':') {
        // An IPv6 literal keeps its brackets; only strip a trailing :port.
        Some((host, port)) if port.chars().all(|c| c.is_ascii_digit()) && !port.is_empty() => host,
        _ => rest,
    };
    matches!(host, "localhost" | "127.0.0.1" | "[::1]" | "::1")
}

/// Returns the value to echo in `Access-Control-Allow-Origin`, or `None` when the
/// origin is not permitted.
fn allowed_origin<'a>(origin: &'a str, configured: &[String]) -> Option<&'a str> {
    if is_loopback_origin(origin) || configured.iter().any(|allowed| allowed == origin) {
        Some(origin)
    } else {
        None
    }
}

fn apply_cors_headers(response: &mut Response, origin: &str) {
    let headers = response.headers_mut();
    if let Ok(value) = HeaderValue::from_str(origin) {
        headers.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, value);
    }
    // The allowed origin varies per request, so any cache in front of the server
    // must key on Origin or it will serve one origin's headers to another.
    headers.insert(header::VARY, HeaderValue::from_static("Origin"));
}

/// Middleware applied to the whole router.
///
/// Applied at the router level so it also covers preflights, which never match a
/// route: `OPTIONS /v1/completions` would otherwise fall through to a `405` with
/// no CORS headers, and the browser would report only a generic network failure.
pub(crate) async fn cors_middleware(
    State(state): State<AppState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let origin = request
        .headers()
        .get(header::ORIGIN)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);

    let allowed = origin
        .as_deref()
        .and_then(|origin| allowed_origin(origin, &state.config.cors_allow_origins))
        .map(str::to_owned);

    if request.method() == Method::OPTIONS {
        let mut response = Response::new(Body::empty());
        *response.status_mut() = match allowed.as_deref() {
            Some(origin) => {
                apply_cors_headers(&mut response, origin);
                let headers = response.headers_mut();
                headers.insert(
                    header::ACCESS_CONTROL_ALLOW_METHODS,
                    HeaderValue::from_static(ALLOWED_METHODS),
                );
                headers.insert(
                    header::ACCESS_CONTROL_ALLOW_HEADERS,
                    HeaderValue::from_static(ALLOWED_HEADERS),
                );
                headers.insert(
                    header::ACCESS_CONTROL_MAX_AGE,
                    HeaderValue::from_static(MAX_AGE_SECONDS),
                );
                StatusCode::NO_CONTENT
            }
            // A disallowed preflight is answered without the approval headers;
            // the browser refuses the real request, which is the correct outcome.
            None => StatusCode::FORBIDDEN,
        };
        return response;
    }

    let mut response = next.run(request).await;
    if let Some(origin) = allowed {
        apply_cors_headers(&mut response, &origin);
    }
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loopback_origins_are_allowed_on_any_port() {
        for origin in [
            "http://localhost:8123",
            "http://localhost:8124",
            "http://127.0.0.1:3000",
            "http://[::1]:8123",
            "https://localhost:8443",
            "http://localhost",
        ] {
            assert!(is_loopback_origin(origin), "should allow {origin}");
        }
    }

    /// Prefix matching would accept these. Structural parsing must not.
    #[test]
    fn lookalike_origins_are_rejected() {
        for origin in [
            "http://localhost.evil.example",
            "http://127.0.0.1.evil.example",
            "http://evil.example",
            "http://notlocalhost",
            "file://",
            "null",
            "http://user@localhost:8123",
            "http://localhost:8123/path",
        ] {
            assert!(!is_loopback_origin(origin), "should reject {origin}");
        }
    }

    #[test]
    fn non_http_schemes_are_rejected() {
        assert!(!is_loopback_origin("ftp://localhost"));
        assert!(!is_loopback_origin("localhost:8123"));
    }

    #[test]
    fn configured_origins_are_allowed_verbatim() {
        let configured = vec!["https://demo.example".to_string()];
        assert_eq!(
            allowed_origin("https://demo.example", &configured),
            Some("https://demo.example")
        );
        assert_eq!(allowed_origin("https://other.example", &configured), None);
    }

    /// The response echoes the caller's origin rather than `*`, so the permitted
    /// set stays exactly what was configured.
    #[test]
    fn allowed_origin_is_echoed_never_wildcarded() {
        assert_eq!(
            allowed_origin("http://localhost:8124", &[]),
            Some("http://localhost:8124")
        );
        assert_ne!(allowed_origin("http://localhost:8124", &[]), Some("*"));
    }

    #[test]
    fn disallowed_origin_yields_no_header() {
        assert_eq!(allowed_origin("http://evil.example", &[]), None);
    }
}

//! Optional llama-server API-key admission, before replay or model routing.
use anyhow::{Context, ensure};
use axum::{
    extract::{Request, State},
    http::{StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};
use sha2::{Digest, Sha256};
use std::{fmt, path::Path, sync::Arc};
use subtle::{Choice, ConstantTimeEq};

/// Immutable key digests. Debug output never contains credentials or digests.
#[derive(Clone, Default)]
pub struct ApiKeys(Arc<[[u8; 32]]>);
impl fmt::Debug for ApiKeys {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ApiKeys")
            .field("configured_keys", &self.0.len())
            .finish_non_exhaustive()
    }
}
impl ApiKeys {
    pub fn from_keys<I, S>(keys: I) -> anyhow::Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut digests = Vec::new();
        for (index, key) in keys.into_iter().enumerate() {
            let key = key.as_ref();
            if key.is_empty() {
                continue;
            }
            ensure!(
                axum::http::HeaderValue::from_str(key).is_ok(),
                "API key {} cannot be sent as an HTTP header; remove invalid HTTP header control characters or line breaks",
                index + 1
            );
            let digest: [u8; 32] = Sha256::digest(key.as_bytes()).into();
            if !digests.contains(&digest) {
                digests.push(digest);
            }
        }
        Ok(Self(digests.into()))
    }
    pub fn from_sources(rows: &[String], file: Option<&Path>) -> anyhow::Result<Self> {
        let mut keys = Vec::new();
        for row in rows {
            keys.extend(csv_keys(row)?);
        }
        if let Some(path) = file {
            let contents = std::fs::read_to_string(path).with_context(|| {
                format!(
                    "read API key file '{}'; provide a readable UTF-8 file with one key per line",
                    path.display()
                )
            })?;
            keys.extend(
                contents
                    .lines()
                    .filter(|line| !line.is_empty() && !line.starts_with('#'))
                    .map(str::to_owned),
            );
        }
        Self::from_keys(keys)
    }
    fn accepts(&self, key: impl AsRef<[u8]>) -> bool {
        let digest: [u8; 32] = Sha256::digest(key.as_ref()).into();
        bool::from(self.0.iter().fold(Choice::from(0), |found, expected| {
            found | expected.ct_eq(&digest)
        }))
    }
}

fn csv_keys(row: &str) -> anyhow::Result<Vec<String>> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut closed = false;
    let mut chars = row.chars().peekable();
    while let Some(ch) = chars.next() {
        if quoted {
            if ch == '"' {
                if chars.peek() == Some(&'"') {
                    field.push('"');
                    chars.next();
                } else {
                    quoted = false;
                    closed = true;
                }
            } else {
                field.push(ch);
            }
        } else if ch == ',' {
            fields.push(std::mem::take(&mut field));
            closed = false;
        } else if ch == '"' && field.is_empty() && !closed {
            quoted = true;
        } else {
            ensure!(
                !closed && ch != '"',
                "--api-key has malformed CSV quoting; quote an entire key and double embedded quotes"
            );
            field.push(ch);
        }
    }
    ensure!(
        !quoted,
        "--api-key has an unterminated quoted key; close its CSV quote"
    );
    fields.push(field);
    Ok(fields)
}

pub(crate) async fn enforce(State(keys): State<ApiKeys>, request: Request, next: Next) -> Response {
    let path = request.uri().path();
    if keys.0.is_empty() || matches!(path, "/health" | "/v1/health") || crate::webui::is_asset(path)
    {
        return next.run(request).await;
    }
    let authorization = request
        .headers()
        .get(header::AUTHORIZATION)
        .map(|value| value.as_bytes());
    let key = authorization.filter(|value| !value.is_empty()).or_else(|| {
        request
            .headers()
            .get("x-api-key")
            .map(|value| value.as_bytes())
    });
    if key.is_some_and(|key| keys.accepts(key.strip_prefix(b"Bearer ").unwrap_or(key))) {
        return next.run(request).await;
    }
    (StatusCode::UNAUTHORIZED, axum::Json(serde_json::json!({"error":{"message":"Invalid API Key; provide a configured key in Authorization or X-Api-Key","type":"authentication_error","code":401}}))).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tower::ServiceExt;
    #[test]
    fn key_sources_support_csv_files_and_redacted_debug() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), "# comment\nfrom-file\r\n\n").unwrap();
        let keys = ApiKeys::from_sources(
            &["one,\"two,comma\",\"with\"\"quote\",,".into()],
            Some(file.path()),
        )
        .unwrap();
        for key in ["one", "two,comma", "with\"quote", "from-file"] {
            assert!(keys.accepts(key));
        }
        assert!(!keys.accepts("other"));
        assert!(!format!("{keys:?}").contains("from-file"));
        assert!(csv_keys("\"unfinished").is_err());
        assert!(ApiKeys::from_keys(["line\nbreak"]).is_err());
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn protected_application_admits_native_chat_and_keeps_health_and_ui_public() {
        use axum::body::{Body, to_bytes};
        use serde_json::{Value, json};
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm");
        let config = crate::ServerConfig {
            api_keys: ApiKeys::from_keys(["test-key"]).unwrap(),
            ..Default::default()
        };
        let state =
            crate::AppState::load_with_config(&root, Some("auth-native".into()), config).unwrap();
        let router = crate::app(state.clone());
        for (path, expected) in [
            ("/", 200),
            ("/health", 200),
            ("/props", 401),
            ("/v1/models", 401),
            ("/v1/realtime", 401),
            ("/v1/stream?conv_id=other", 401),
        ] {
            let response = router
                .clone()
                .oneshot(
                    axum::http::Request::builder()
                        .uri(path)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status().as_u16(), expected, "{path}");
        }
        let denied = router
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from("not JSON"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            denied.status(),
            StatusCode::UNAUTHORIZED,
            "authentication precedes request parsing"
        );
        let response = router
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .header("x-api-key", "test-key")
                    .body(Body::from(
                        json!({"messages":[{"role":"user","content":"hello"}],"max_tokens":1})
                            .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let output: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 65536).await.unwrap()).unwrap();
        assert_eq!(output["usage"]["completion_tokens"], 1);
        assert!(
            output["choices"][0]["message"]["content"]
                .as_str()
                .is_some_and(|text| !text.is_empty())
        );
        let model = state.registry.resolve("auth-native").unwrap().unwrap();
        tokio::task::spawn_blocking(move || model.engine.shutdown())
            .await
            .unwrap();
    }
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn realtime_upgrade_requires_a_valid_key_before_opening_native_session() {
        use futures_util::StreamExt;
        use tokio_tungstenite::tungstenite::client::IntoClientRequest;
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/onnx_genai_workflows/realtime");
        let state = crate::AppState::load_with_config(
            &root,
            Some("auth-realtime".into()),
            crate::ServerConfig {
                api_keys: ApiKeys::from_keys(["ws-key"]).unwrap(),
                ..Default::default()
            },
        )
        .unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let router = crate::app(state.clone());
        let task = tokio::spawn(async move {
            axum::serve(listener, router).await.unwrap();
        });
        let url = format!("ws://{address}/v1/realtime?model=auth-realtime");
        let denied = tokio_tungstenite::client_async(
            url.clone(),
            tokio::net::TcpStream::connect(address).await.unwrap(),
        )
        .await;
        match denied {
            Err(tokio_tungstenite::tungstenite::Error::Http(response)) => {
                assert_eq!(response.status(), StatusCode::UNAUTHORIZED)
            }
            _ => panic!("unauthenticated realtime upgrade must return HTTP401"),
        }
        let mut request = url.into_client_request().unwrap();
        request
            .headers_mut()
            .insert(header::AUTHORIZATION, "Bearer ws-key".parse().unwrap());
        let (mut client, response) = tokio_tungstenite::client_async(
            request,
            tokio::net::TcpStream::connect(address).await.unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(response.status(), StatusCode::SWITCHING_PROTOCOLS);
        let event = tokio::time::timeout(std::time::Duration::from_secs(5), client.next())
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        let output: serde_json::Value = serde_json::from_str(event.to_text().unwrap()).unwrap();
        assert_eq!(output["type"], "session.created");
        client.close(None).await.unwrap();
        drop(client);
        let model = state.registry.resolve("auth-realtime").unwrap().unwrap();
        tokio::task::spawn_blocking(move || model.engine.shutdown())
            .await
            .unwrap();
        task.abort();
        let _ = task.await;
    }
    #[tokio::test]
    async fn headers_public_assets_and_disabled_policy_follow_upstream_admission() {
        let keys = ApiKeys::from_keys(["first", "second", "café"]).unwrap();
        let router = axum::Router::new()
            .fallback(|| async { StatusCode::NO_CONTENT })
            .layer(axum::middleware::from_fn_with_state(keys, enforce));
        for (path, auth, xkey, expected) in [
            ("/v1/models", None, None, 401),
            ("/v1/models", Some("Bearer first"), None, 204),
            ("/v1/models", Some("second"), None, 204),
            ("/v1/models", Some("Bearer café"), None, 204),
            ("/v1/models", None, Some("first"), 204),
            ("/v1/models", Some("wrong"), Some("first"), 401),
            ("/v1/models", Some(""), Some("first"), 204),
            ("/health", None, None, 204),
            ("/v1/health", None, None, 204),
            ("/", None, None, 204),
            ("/index.html", None, None, 204),
            ("/missing.js", None, None, 401),
            ("/v1/realtime", None, None, 401),
            ("/v1/stream?conv_id=x", None, None, 401),
            ("/v1/admin/models", None, None, 401),
        ] {
            let mut request = axum::http::Request::builder().uri(path);
            if let Some(value) = auth {
                request = request.header(header::AUTHORIZATION, value);
            }
            if let Some(value) = xkey {
                request = request.header("x-api-key", value);
            }
            let response = router
                .clone()
                .oneshot(request.body(axum::body::Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status().as_u16(), expected, "{path}");
            if expected == 401 {
                let value: serde_json::Value = serde_json::from_slice(
                    &axum::body::to_bytes(response.into_body(), 4096)
                        .await
                        .unwrap(),
                )
                .unwrap();
                assert_eq!(value["error"]["type"], "authentication_error");
                assert_eq!(value["error"]["code"], 401);
            }
        }
        let invalid_header = router
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/models")
                    .header(
                        header::AUTHORIZATION,
                        axum::http::HeaderValue::from_bytes(&[255]).unwrap(),
                    )
                    .header("x-api-key", "first")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            invalid_header.status(),
            StatusCode::UNAUTHORIZED,
            "a malformed preferred header must not fall back to a different credential"
        );
        let open = axum::Router::new()
            .fallback(|| async { StatusCode::NO_CONTENT })
            .layer(axum::middleware::from_fn_with_state(
                ApiKeys::default(),
                enforce,
            ));
        assert_eq!(
            open.oneshot(
                axum::http::Request::builder()
                    .uri("/v1/models")
                    .body(axum::body::Body::empty())
                    .unwrap()
            )
            .await
            .unwrap()
            .status(),
            StatusCode::NO_CONTENT
        );
    }
}

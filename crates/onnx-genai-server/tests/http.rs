use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use onnx_genai_server::{AppState, app};
use serde_json::{Value, json};
use std::path::PathBuf;
use tower::ServiceExt;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm")
}

async fn test_app() -> axum::Router {
    let state = AppState::load(&fixture_dir(), Some("tiny-llm".to_string())).unwrap();
    app(state)
}

async fn create_http_session(app: axum::Router) -> String {
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/sessions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["object"], "session");
    json["id"].as_str().unwrap().to_string()
}

#[tokio::test]
async fn health_returns_loaded_model() {
    let app = test_app().await;
    let response = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "ok");
    assert_eq!(json["model"], "tiny-llm");
}

#[tokio::test]
async fn chat_completions_returns_openai_shape() {
    let app = test_app().await;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "tiny-llm",
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 1
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["object"], "chat.completion");
    assert_eq!(json["model"], "tiny-llm");
    assert_eq!(json["choices"][0]["index"], 0);
    assert_eq!(json["choices"][0]["message"]["role"], "assistant");
    assert!(json["choices"][0]["message"]["content"].is_string());
    assert!(json["choices"][0]["finish_reason"].is_string());
    let prompt_tokens = json["usage"]["prompt_tokens"].as_u64().unwrap();
    let completion_tokens = json["usage"]["completion_tokens"].as_u64().unwrap();
    let total_tokens = json["usage"]["total_tokens"].as_u64().unwrap();
    assert!(prompt_tokens > 0);
    assert_eq!(total_tokens, prompt_tokens + completion_tokens);
    assert!(json.get("session_id").is_none());
}

#[tokio::test]
async fn streaming_chat_completions_returns_sse_chunks() {
    let app = test_app().await;
    let session_id = create_http_session(app.clone()).await;
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .header("X-Session-Id", session_id)
                .body(Body::from(
                    json!({
                        "model": "tiny-llm",
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 1,
                        "stream": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers().get(header::CONTENT_TYPE).unwrap(),
        "text/event-stream"
    );
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(text.contains("chat.completion.chunk"), "{text}");
    assert!(text.contains("[DONE]"), "{text}");
}

#[tokio::test]
async fn chat_completions_reuses_persistent_session() {
    let app = test_app().await;
    let session_id = create_http_session(app.clone()).await;

    let first = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .header("X-Session-Id", &session_id)
                .body(Body::from(
                    json!({
                        "model": "tiny-llm",
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 1
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(first.status(), StatusCode::OK);
    let first_body = to_bytes(first.into_body(), usize::MAX).await.unwrap();
    let first_json: Value = serde_json::from_slice(&first_body).unwrap();
    assert_eq!(first_json["session_id"], session_id);
    let first_count = first_json["session_token_count"].as_u64().unwrap();
    assert!(first_count > 0);

    let second = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .header("X-Session-Id", &session_id)
                .body(Body::from(
                    json!({
                        "model": "tiny-llm",
                        "messages": [{"role": "user", "content": "world"}],
                        "max_tokens": 1
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(second.status(), StatusCode::OK);
    let second_body = to_bytes(second.into_body(), usize::MAX).await.unwrap();
    let second_json: Value = serde_json::from_slice(&second_body).unwrap();
    assert_eq!(second_json["session_id"], session_id);
    let second_count = second_json["session_token_count"].as_u64().unwrap();
    assert!(second_count > first_count);

    let deleted = app
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/v1/sessions/{session_id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(deleted.status(), StatusCode::NO_CONTENT);
}

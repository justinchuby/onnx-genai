use crate::{
    AppState, ChatCompletionRequest, CompletionRequest, ServerConfig, app, build_generate_request,
    driver::{DriverCommand, EngineDriver},
    routes::{CompletionGeneration, collect_generation_result, prepare_completion},
    sse::StopBoundaryBuffer,
};
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use onnx_genai::{Engine, EngineConfig};
use serde_json::{Value, json};
use std::{path::PathBuf, time::Duration};
use tokio::{sync::mpsc, time::timeout};
use tower::ServiceExt;

#[test]
fn completion_suffix_maps_to_fim_generation() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm");
    let mut state = AppState::load(&model_dir, Some("tiny-llm".to_string())).expect("load fixture");
    state.fim_config = Some(onnx_genai_engine::FimConfig {
        prefix_token: "<PRE>".to_string(),
        middle_token: "<MID>".to_string(),
        suffix_token: "<SUF>".to_string(),
        format: onnx_genai_engine::FimFormat::PSM,
    });
    let request: CompletionRequest = serde_json::from_value(json!({
        "model": "tiny-llm",
        "prompt": "prefix",
        "suffix": "suffix",
        "max_tokens": 7,
        "min_p": 0.2,
        "frequency_penalty": 0.3,
        "presence_penalty": 0.4
    }))
    .unwrap();

    let prepared = prepare_completion(&request, &state).unwrap();
    match prepared.generation {
        CompletionGeneration::Fim {
            prefix,
            suffix,
            options,
        } => {
            assert_eq!(prefix, "prefix");
            assert_eq!(suffix, "suffix");
            assert_eq!(options.max_new_tokens, 7);
            assert_eq!(options.min_p, 0.2);
            assert_eq!(options.frequency_penalty, 0.3);
            assert_eq!(options.presence_penalty, 0.4);
        }
        CompletionGeneration::Plain(_) => panic!("suffix must route to FIM generation"),
    }
}

#[tokio::test]
async fn completion_suffix_uses_fim_and_returns_text_completion() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm");
    let mut state = AppState::load(&model_dir, Some("tiny-llm".to_string())).expect("load fixture");
    state.fim_config = Some(onnx_genai_engine::FimConfig {
        prefix_token: "<PRE>".to_string(),
        middle_token: "<MID>".to_string(),
        suffix_token: "<SUF>".to_string(),
        format: onnx_genai_engine::FimFormat::PSM,
    });

    let response = app(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "tiny-llm",
                        "prompt": "prefix",
                        "suffix": "suffix",
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
    assert_eq!(json["object"], "text_completion");
    assert!(json["choices"][0]["text"].is_string());
    assert!(json["choices"][0]["logprobs"].is_null());
}

#[test]
fn stop_boundary_buffer_holds_partial_stop_sequence() {
    let mut buffer = StopBoundaryBuffer::new(vec!["tok20".to_string()]);
    assert_eq!(buffer.push("to"), "");
    assert_eq!(buffer.push("k"), "");
    assert_eq!(buffer.push("2"), "");
    assert_eq!(buffer.push("1"), "tok21");
    assert_eq!(buffer.flush(), "");
}

#[test]
fn stop_boundary_buffer_suppresses_matched_stop_sequence() {
    let mut buffer = StopBoundaryBuffer::new(vec!["tok20".to_string()]);
    assert_eq!(buffer.push("hello tok"), "hello ");
    assert_eq!(buffer.push("20"), "");
    assert_eq!(buffer.flush(), "");
}

#[tokio::test]
async fn generation_over_capacity_returns_429_with_retry_after() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm");
    let state = AppState::load_with_config(
        &model_dir,
        Some("tiny-llm".to_string()),
        ServerConfig {
            max_output_tokens: 16,
            max_sessions: 8,
            max_pending: 1,
        },
    )
    .unwrap();
    let _occupied = state
        .engine
        .generation_capacity
        .clone()
        .try_acquire_owned()
        .unwrap();

    let response = app(state)
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

    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(response.headers()[header::RETRY_AFTER], "1");
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body: Value = serde_json::from_slice(&body).unwrap();
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("generation capacity exceeded")
    );
}

#[tokio::test]
async fn stalled_output_route_does_not_block_another_completion() {
    let model_dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm-scatter");
    let engine = Engine::from_dir(&model_dir, EngineConfig::default()).unwrap();
    let driver = EngineDriver::start(engine, 2, 2);
    let slow_request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "tiny-llm-scatter",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8
    }))
    .unwrap();
    let fast_request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "tiny-llm-scatter",
        "messages": [{"role": "user", "content": "world"}],
        "max_tokens": 2
    }))
    .unwrap();
    let (slow_tx, _slow_rx) = mpsc::channel(1);
    let slow_permit = driver
        .generation_capacity
        .clone()
        .try_acquire_owned()
        .unwrap();
    driver
        .commands
        .send(DriverCommand::Generate {
            session_id: None,
            request: Box::new(build_generate_request(&slow_request)),
            events: slow_tx,
            permit: slow_permit,
        })
        .await
        .unwrap();
    let fast_rx = driver
        .generate(None, build_generate_request(&fast_request))
        .await
        .unwrap();

    let fast_result = timeout(Duration::from_secs(5), collect_generation_result(fast_rx))
        .await
        .expect("fast request timed out behind stalled consumer")
        .expect("fast request failed");
    assert_eq!(fast_result.token_ids.len(), 2);
}

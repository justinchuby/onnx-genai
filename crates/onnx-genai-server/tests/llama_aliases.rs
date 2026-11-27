//! Runtime acceptance for aliases that the static route audit cannot verify.
#![cfg(feature = "native-backend")]

use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use onnx_genai_engine::{EngineConfig, EngineDecodeBackend, NativeDecodeDevice};
use onnx_genai_server::{AppState, ServerConfig, app};
use serde_json::{Value, json};
use std::path::PathBuf;
use tower::ServiceExt;

fn native_app() -> axum::Router {
    native_fixture("tiny-llm")
}

fn native_fixture(name: &str) -> axum::Router {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name);
    let state = AppState::load_with_config(
        &fixture,
        Some("alias-native".into()),
        ServerConfig {
            engine_config: EngineConfig {
                decode_backend: EngineDecodeBackend::Native,
                native_device: Some(NativeDecodeDevice::Cpu),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    app(state)
}

async fn request(router: axum::Router, path: &str, body: Value) -> axum::response::Response {
    router
        .oneshot(
            Request::post(path)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap()
}

async fn json_response(response: axum::response::Response) -> Value {
    assert_eq!(response.status(), StatusCode::OK);
    serde_json::from_slice(&to_bytes(response.into_body(), 1024 * 1024).await.unwrap()).unwrap()
}

fn response_text(response: &Value) -> &str {
    response["output"][0]["content"][0]["text"]
        .as_str()
        .expect("native Responses result must contain generated output text")
}

#[tokio::test]
async fn responses_alias_executes_native_generation_and_preserves_usage_and_errors() {
    let router = native_app();
    let body = json!({"input":"hello","max_output_tokens":2,"temperature":0});
    let canonical =
        json_response(request(router.clone(), "/v1/responses", body.clone()).await).await;
    let alias = json_response(request(router.clone(), "/responses", body).await).await;
    assert_eq!(alias["object"], "response");
    assert!(!response_text(&alias).is_empty());
    assert_eq!(response_text(&alias), response_text(&canonical));
    assert_eq!(alias["status"], canonical["status"]);
    assert_eq!(alias["usage"], canonical["usage"]);
    assert!(alias["usage"]["input_tokens"].as_u64().unwrap() > 0);
    assert!(alias["usage"]["output_tokens"].as_u64().unwrap() > 0);

    for path in ["/v1/responses", "/responses"] {
        let response = request(
            router.clone(),
            path,
            json!({"input":42,"max_output_tokens":2}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{path}");
        let error: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 1024 * 1024).await.unwrap())
                .unwrap();
        assert_eq!(error["error"]["type"], "invalid_request_error", "{error}");
        assert!(
            error["error"]["message"]
                .as_str()
                .unwrap()
                .contains("input"),
            "{error}"
        );
    }
}

#[tokio::test]
async fn response_alias_stream_preserves_typed_events_text_and_terminal_usage() {
    let router = native_app();
    let mut results = Vec::new();
    for path in ["/v1/responses", "/responses"] {
        let response = request(
            router.clone(),
            path,
            json!({"input":"hello","max_output_tokens":2,"temperature":0,"stream":true}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK, "{path}");
        assert!(
            response.headers()[header::CONTENT_TYPE]
                .to_str()
                .unwrap()
                .starts_with("text/event-stream")
        );
        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let wire = std::str::from_utf8(&bytes).unwrap();
        let events = wire
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter(|line| *line != "[DONE]")
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        let wire_types = wire
            .lines()
            .filter_map(|line| line.strip_prefix("event: "))
            .collect::<Vec<_>>();
        assert_eq!(
            wire_types,
            events
                .iter()
                .map(|event| event["type"].as_str().unwrap())
                .collect::<Vec<_>>()
        );
        let mut kinds = Vec::<String>::new();
        for (sequence, event) in events.iter().enumerate() {
            assert_eq!(
                event["sequence_number"].as_u64(),
                Some(u64::try_from(sequence).unwrap()),
                "{path}: {event}"
            );
            let kind = event["type"].as_str().unwrap();
            // Only batching-dependent adjacent text deltas may collapse. Every
            // other typed event must occur in the declared position exactly once.
            if kind != "response.output_text.delta"
                || kinds.last().map(String::as_str) != Some(kind)
            {
                kinds.push(kind.to_owned());
            }
        }
        let terminal_type = kinds.last().unwrap().as_str();
        assert!(matches!(
            terminal_type,
            "response.completed" | "response.incomplete"
        ));
        assert_eq!(
            kinds.iter().map(String::as_str).collect::<Vec<_>>(),
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                terminal_type,
            ],
            "{path}: {events:?}"
        );
        assert_eq!(events.first().unwrap()["type"], "response.created");
        assert!(
            events.iter().all(|event| event["type"] != "error"),
            "{events:?}"
        );
        let terminals = events
            .iter()
            .filter(|event| {
                matches!(
                    event["type"].as_str(),
                    Some("response.completed" | "response.incomplete")
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(terminals.len(), 1, "{events:?}");
        assert_eq!(events.last().unwrap(), terminals[0]);
        let text = events
            .iter()
            .filter(|event| event["type"] == "response.output_text.delta")
            .map(|event| event["delta"].as_str().unwrap())
            .collect::<String>();
        assert!(!text.is_empty(), "{path}");
        let final_value = &terminals[0]["response"];
        assert_eq!(text, response_text(final_value));
        let response_id = events[0]["response"]["id"].as_str().unwrap();
        assert!(!response_id.is_empty());
        let output = final_value["output"].as_array().unwrap();
        assert_eq!(output.len(), 1);
        let item = &output[0];
        let item_id = item["id"].as_str().unwrap();
        assert!(!item_id.is_empty());
        assert_eq!(item["type"], "message");
        assert_eq!(item["role"], "assistant");
        for event in &events {
            match event["type"].as_str().unwrap() {
                "response.created"
                | "response.in_progress"
                | "response.completed"
                | "response.incomplete" => {
                    assert_eq!(event["response"]["id"], response_id);
                }
                "response.output_item.added" | "response.output_item.done" => {
                    assert_eq!(event["output_index"], 0);
                    assert_eq!(event["item"]["id"], item_id);
                    if event["type"] == "response.output_item.done" {
                        assert_eq!(&event["item"], item);
                    }
                }
                _ => {
                    assert_eq!(event["output_index"], 0);
                    assert_eq!(event["content_index"], 0);
                    assert_eq!(event["item_id"], item_id);
                    if event["type"] == "response.output_text.done" {
                        assert_eq!(event["text"], text);
                    }
                    if event["type"] == "response.content_part.done" {
                        assert_eq!(event["part"], item["content"][0]);
                    }
                }
            }
        }
        assert!(final_value["usage"]["input_tokens"].as_u64().unwrap() > 0);
        assert!(final_value["usage"]["output_tokens"].as_u64().unwrap() > 0);
        results.push((
            text,
            final_value["usage"].clone(),
            final_value["status"].clone(),
            kinds,
        ));
    }
    assert_eq!(results[0], results[1]);
}

#[tokio::test]
async fn input_count_aliases_match_native_inference_and_reject_malformed_input() {
    let router = native_app();
    let messages = json!([{"role":"user","content":"hello"}]);
    let chat = json_response(
        request(
            router.clone(),
            "/v1/chat/completions",
            json!({"messages":messages,"max_tokens":2,"temperature":0}),
        )
        .await,
    )
    .await;
    let responses = json_response(
        request(
            router.clone(),
            "/v1/responses",
            json!({"input":"hello","max_output_tokens":2,"temperature":0}),
        )
        .await,
    )
    .await;
    for (paths, body, malformed, expected) in [
        (
            [
                "/v1/chat/completions/input_tokens",
                "/chat/completions/input_tokens",
            ],
            json!({"messages":messages}),
            json!({"messages":"invalid"}),
            &chat["usage"]["prompt_tokens"],
        ),
        (
            ["/v1/responses/input_tokens", "/responses/input_tokens"],
            json!({"input":"hello"}),
            json!({"input":42}),
            &responses["usage"]["input_tokens"],
        ),
    ] {
        assert!(expected.as_u64().unwrap() > 0);
        for path in paths {
            let count = json_response(request(router.clone(), path, body.clone()).await).await;
            assert_eq!(&count["input_tokens"], expected, "{path}: {count}");
            let response = request(router.clone(), path, malformed.clone()).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{path}");
            let error: Value =
                serde_json::from_slice(&to_bytes(response.into_body(), 1024 * 1024).await.unwrap())
                    .unwrap();
            assert_eq!(
                error["error"]["type"], "invalid_request_error",
                "{path}: {error}"
            );
            assert!(!error["error"]["message"].as_str().unwrap().is_empty());
        }
    }
}

#[tokio::test]
async fn native_embedding_aliases_execute_pooled_values_and_normalization() {
    let router = native_fixture("tiny-mtp-full");
    for normalize in [-1, 2] {
        let body = json!({"content":"hello","embd_normalize":normalize});
        let canonical =
            json_response(request(router.clone(), "/embedding", body.clone()).await).await;
        let alias = json_response(request(router.clone(), "/embeddings", body).await).await;
        assert_eq!(alias, canonical);
        let vector = alias[0]["embedding"][0].as_array().unwrap();
        assert!(!vector.is_empty());
        let values = vector
            .iter()
            .map(|value| value.as_f64().unwrap())
            .collect::<Vec<_>>();
        assert!(values.iter().all(|value| value.is_finite()));
        assert!(values.iter().any(|value| *value != 0.0));
        if normalize == 2 {
            assert!((values.iter().map(|value| value * value).sum::<f64>() - 1.0).abs() < 1e-5);
        }
    }
    for path in ["/embedding", "/embeddings"] {
        let response = request(
            router.clone(),
            path,
            json!({"content":"hello","embd_normalize":-2}),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{path}");
        let error: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), 1024 * 1024).await.unwrap())
                .unwrap();
        assert!(
            error["error"]["message"]
                .as_str()
                .unwrap()
                .contains("embd_normalize")
        );
    }
}

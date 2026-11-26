//! Real Native decoded-content constraints; raw grammar keeps raw vocabulary semantics.
#![cfg(feature = "native-backend")]
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use onnx_genai_engine::{EngineConfig, EngineDecodeBackend, NativeDecodeDevice};
use onnx_genai_server::{AppState, ServerConfig, app};
use serde_json::{Value, json};
use std::path::Path;
use tower::ServiceExt;
mod decoder {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/build_literal_decoder.rs"
    ));
}
fn fixture(split: bool) -> (tempfile::TempDir, axum::Router) {
    let root = tempfile::tempdir().unwrap();
    decoder::build(
        root.path(),
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures"),
        split,
    );
    let router = app(AppState::load_with_config(
        root.path(),
        Some("literal".into()),
        ServerConfig {
            engine_config: EngineConfig {
                decode_backend: EngineDecodeBackend::Native,
                native_device: Some(NativeDecodeDevice::Cpu),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap());
    (root, router)
}
async fn request(router: axum::Router, body: Value) -> (StatusCode, String) {
    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let wire = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    (status, String::from_utf8(wire.to_vec()).unwrap())
}
#[tokio::test]
async fn native_literal_replace_masks_decoded_content_and_streams_canonical_output() {
    for split in [false, true] {
        let (_root, router) = fixture(split);
        for stream in [false, true] {
            let (status,wire)=request(router.clone(),json!({"model":"literal","messages":[{"role":"user","content":"hello"}],"response_format":{"type":"json_schema","json_schema":{"name":"seven","schema":{"const":7}}},"max_tokens":if split {2} else {1},"temperature":0,"logit_bias":if split {json!({"22":1000,"23":999,"24":1001,"3":1002})} else {json!({"22":1000,"23":1001,"3":1002})},"stream":stream,"stream_options":{"include_usage":true}})).await;
            assert_eq!(status, StatusCode::OK, "{wire}");
            if stream {
                assert!(wire.contains("data: [DONE]"), "{wire}");
                let mut text = String::new();
                let mut tokens = None;
                for line in wire
                    .lines()
                    .filter_map(|l| l.strip_prefix("data: "))
                    .filter(|l| *l != "[DONE]")
                {
                    let event: Value = serde_json::from_str(line).unwrap();
                    if let Some(delta) = event["choices"][0]["delta"]["content"].as_str() {
                        text.push_str(delta);
                    }
                    if let Some(n) = event["usage"]["completion_tokens"].as_u64() {
                        tokens = Some(n);
                    }
                }
                assert_eq!(text, "7");
                assert_eq!(tokens, Some(if split { 2 } else { 1 }));
            } else {
                let result: Value = serde_json::from_str(&wire).unwrap();
                assert_eq!(result["choices"][0]["message"]["content"], "7");
                assert_eq!(
                    result["usage"]["completion_tokens"],
                    if split { 2 } else { 1 }
                );
            }
        }
    }
}
#[tokio::test]
async fn native_literal_replace_preserves_raw_grammar_input_bytes() {
    let (_root, router) = fixture(false);
    let (status,wire)=request(router,json!({"model":"literal","messages":[{"role":"user","content":"hello"}],"grammar":"root ::= \"@source@\"","max_tokens":1,"temperature":0,"logit_bias":{"22":1000,"23":1001}})).await;
    assert_eq!(status, StatusCode::OK, "{wire}");
    let result: Value = serde_json::from_str(&wire).unwrap();
    assert_eq!(result["choices"][0]["message"]["content"], "7");
    assert_eq!(result["usage"]["completion_tokens"], 1);
}

#[tokio::test]
async fn native_literal_byte_runs_match_canonical_replacement_at_eof_in_buffered_and_sse() {
    for byte_level in [false, true] {
        let root = tempfile::tempdir().unwrap();
        decoder::invalid_group(
            root.path(),
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures"),
            byte_level,
        );
        let router = app(AppState::load_with_config(
            root.path(),
            Some("literal".into()),
            ServerConfig {
                engine_config: EngineConfig {
                    decode_backend: EngineDecodeBackend::Native,
                    native_device: Some(NativeDecodeDevice::Cpu),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap());
        for stream in [false, true] {
            let (status,wire)=request(router.clone(),json!({"model":"literal","messages":[{"role":"user","content":"hello"}],"response_format":{"type":"json_schema","json_schema":{"name":"seven","schema":{"const":7}}},"max_tokens":3,"temperature":0,"stream":stream,"stream_options":{"include_usage":true}})).await;
            assert_eq!(status, StatusCode::OK, "{wire}");
            if stream {
                assert!(wire.contains("data: [DONE]"), "{wire}");
                let mut content = String::new();
                let mut count = None;
                for data in wire
                    .lines()
                    .filter_map(|line| line.strip_prefix("data: "))
                    .filter(|data| *data != "[DONE]")
                {
                    let event: Value = serde_json::from_str(data).unwrap();
                    if let Some(text) = event["choices"][0]["delta"]["content"].as_str() {
                        content.push_str(text);
                    }
                    if let Some(n) = event["usage"]["completion_tokens"].as_u64() {
                        count = Some(n);
                    }
                }
                assert_eq!(content, "7");
                assert_eq!(count, Some(3));
            } else {
                let result: Value = serde_json::from_str(&wire).unwrap();
                assert_eq!(result["choices"][0]["message"]["content"], "7");
                assert_eq!(result["usage"]["completion_tokens"], 3);
            }
        }
    }
}

#[tokio::test]
async fn native_literal_workspace_denial_is_json_before_sse_and_valid_independent_request_recovers()
{
    let root = tempfile::tempdir().unwrap();
    decoder::build(
        root.path(),
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures"),
        false,
    );
    let load = |cap| {
        app(AppState::load_with_config(
            root.path(),
            Some("literal".into()),
            ServerConfig {
                engine_config: EngineConfig {
                    decode_backend: EngineDecodeBackend::Native,
                    native_device: Some(NativeDecodeDevice::Cpu),
                    limits: onnx_genai_engine::ResourceLimits {
                        host_ram_limit: onnx_genai_engine::ResourceLimit::Bytes(cap),
                        ..Default::default()
                    },
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap())
    };
    let body = json!({"model":"literal","messages":[{"role":"user","content":"hello"}],"response_format":{"type":"json_schema","json_schema":{"name":"seven","schema":{"const":7}}},"max_tokens":1,"temperature":0,"logit_bias":{"22":1000,"23":1001},"stream":true,"stream_options":{"include_usage":true}});
    let denied = load(160 * 1024 * 1024);
    for _ in 0..2 {
        let response = denied
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/chat/completions")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(response.headers()[header::RETRY_AFTER], "1");
        assert!(
            !response
                .headers()
                .get(header::CONTENT_TYPE)
                .unwrap()
                .to_str()
                .unwrap()
                .contains("text/event-stream")
        );
        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        let error: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(error["error"]["type"] == "resource_limit_error", "{error}");
    }
    let (status, wire) = request(load(512 * 1024 * 1024), body).await;
    assert_eq!(status, StatusCode::OK, "{wire}");
    assert!(wire.contains("data: [DONE]"));
    let mut text = String::new();
    let mut count = None;
    for data in wire
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
    {
        let event: Value = serde_json::from_str(data).unwrap();
        if let Some(delta) = event["choices"][0]["delta"]["content"].as_str() {
            text.push_str(delta);
        }
        if let Some(n) = event["usage"]["completion_tokens"].as_u64() {
            count = Some(n);
        }
    }
    assert_eq!(text, "7");
    assert_eq!(count, Some(1));
}

#[tokio::test]
async fn native_literal_deferred_semantic_errors_precede_sse_and_valid_retry_remains_usable() {
    let (_root, router) = fixture(false);
    let base = json!({"model":"literal","messages":[{"role":"user","content":"hello"}],"response_format":{"type":"json_schema","json_schema":{"name":"seven","schema":{"const":7}}},"max_tokens":1,"temperature":0,"logit_bias":{"22":1000,"23":1001},"stream":true,"stream_options":{"include_usage":true}});
    let mut invalid = base.clone();
    invalid["response_format"]["json_schema"]["schema"] = json!({"type":"not-a-json-schema-type"});
    let response = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(invalid.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(
        !response
            .headers()
            .get(header::CONTENT_TYPE)
            .unwrap()
            .to_str()
            .unwrap()
            .contains("text/event-stream")
    );
    let error = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    assert!(String::from_utf8_lossy(&error).contains("processor configuration"));
    let (status, wire) = request(router.clone(), base.clone()).await;
    assert_eq!(status, StatusCode::OK, "{wire}");
    assert!(wire.contains("data: [DONE]"));
    let bad = tempfile::tempdir().unwrap();
    decoder::build(
        bad.path(),
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures"),
        false,
    );
    let path = bad.path().join("tokenizer.json");
    let mut config: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    config["decoder"]["decoders"][2]["pattern"] = json!({"Regex":"@source@+"});
    std::fs::write(path, serde_json::to_vec(&config).unwrap()).unwrap();
    let bad_router = app(AppState::load_with_config(
        bad.path(),
        Some("literal".into()),
        ServerConfig {
            engine_config: EngineConfig {
                decode_backend: EngineDecodeBackend::Native,
                native_device: Some(NativeDecodeDevice::Cpu),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap());
    let mut unforced = base.clone();
    unforced.as_object_mut().unwrap().remove("response_format");
    unforced["stream"] = json!(false);
    unforced["logit_bias"] = json!({"22":1000});
    let (status, wire) = request(bad_router.clone(), unforced).await;
    assert_eq!(
        status,
        StatusCode::OK,
        "unforced Regex publication must retain canonical support: {wire}"
    );
    let result: Value = serde_json::from_str(&wire).unwrap();
    assert_eq!(result["choices"][0]["message"]["content"], "7");
    let response = bad_router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(base.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    assert!(
        !response
            .headers()
            .get(header::CONTENT_TYPE)
            .unwrap()
            .to_str()
            .unwrap()
            .contains("text/event-stream")
    );
    let error = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    assert!(String::from_utf8_lossy(&error).contains("regex Replace"));
    let (status, wire) = request(router, base).await;
    assert_eq!(status, StatusCode::OK, "{wire}");
    assert!(wire.contains("data: [DONE]"));
}

#[tokio::test]
async fn native_literal_tool_lark_masks_decoded_envelope_before_http_and_sse_publication() {
    let root = tempfile::tempdir().unwrap();
    decoder::build(
        root.path(),
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures"),
        false,
    );
    let envelope =
        "<tool_call>\n{\"name\":\"weather\",\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>";
    let path = root.path().join("tokenizer.json");
    let mut tokenizer: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    tokenizer["decoder"]["decoders"][2]["content"] = json!(envelope);
    std::fs::write(path, serde_json::to_vec(&tokenizer).unwrap()).unwrap();
    let path = root.path().join("inference_metadata.yaml");
    let mut metadata: serde_yaml::Value =
        serde_yaml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    metadata["schema_version"] = serde_yaml::Value::String("v1.3".into());
    metadata["package"]["tool_protocol"] =
        serde_yaml::to_value(json!({"identity":"tagged-json","version":"v1"})).unwrap();
    std::fs::write(path, serde_yaml::to_string(&metadata).unwrap()).unwrap();
    let router = app(AppState::load_with_config(
        root.path(),
        Some("literal".into()),
        ServerConfig {
            engine_config: EngineConfig {
                decode_backend: EngineDecodeBackend::Native,
                native_device: Some(NativeDecodeDevice::Cpu),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap());
    for stream in [false, true] {
        let body = json!({"model":"literal","messages":[{"role":"user","content":"hello"}],"max_tokens":1,"temperature":0,"logit_bias":{"22":1000,"23":1001},"stream":stream,"stream_options":{"include_usage":true},"tool_choice":"required","tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object","properties":{"city":{"const":"Paris"}},"required":["city"],"additionalProperties":false}}}]});
        for invalid in [false, true, false] {
            let mut current = body.clone();
            if invalid {
                current["tools"][0]["function"]["parameters"] = json!({"type":"not-a-schema-type"});
            }
            let (status, wire) = request(router.clone(), current).await;
            if invalid {
                assert_eq!(status, StatusCode::BAD_REQUEST, "{wire}");
                assert!(!wire.contains("data:"), "{wire}");
                continue;
            }
            assert_eq!(status, StatusCode::OK, "{wire}");
            let mut name = String::new();
            let mut arguments = String::new();
            let mut usage = None;
            if stream {
                assert!(wire.contains("data: [DONE]"), "{wire}");
                for line in wire
                    .lines()
                    .filter_map(|line| line.strip_prefix("data: "))
                    .filter(|line| *line != "[DONE]")
                {
                    let event: Value = serde_json::from_str(line).unwrap();
                    assert!(
                        event["choices"][0]["delta"]["content"]
                            .as_str()
                            .is_none_or(str::is_empty),
                        "raw envelope leaked: {event}"
                    );
                    if let Some(calls) = event["choices"][0]["delta"]["tool_calls"].as_array() {
                        for call in calls {
                            assert_eq!(call["index"], 0);
                            if let Some(value) = call["function"]["name"].as_str() {
                                name.push_str(value);
                            }
                            if let Some(value) = call["function"]["arguments"].as_str() {
                                arguments.push_str(value);
                            }
                        }
                    }
                    if let Some(value) = event["usage"]["completion_tokens"].as_u64() {
                        usage = Some(value);
                    }
                }
            } else {
                let event: Value = serde_json::from_str(&wire).unwrap();
                assert_eq!(event["choices"][0]["finish_reason"], "tool_calls");
                assert!(event["choices"][0]["message"]["content"].is_null());
                let call = &event["choices"][0]["message"]["tool_calls"][0];
                name = call["function"]["name"].as_str().unwrap().into();
                arguments = call["function"]["arguments"].as_str().unwrap().into();
                usage = event["usage"]["completion_tokens"].as_u64();
            }
            assert_eq!(name, "weather");
            assert_eq!(
                serde_json::from_str::<Value>(&arguments).unwrap(),
                json!({"city":"Paris"})
            );
            assert_eq!(usage, Some(1));
        }
    }
}

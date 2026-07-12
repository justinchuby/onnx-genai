use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use onnx_genai::{Engine, EngineConfig, GeneratePrompt};
use onnx_genai_engine::GenerateConstraint;
use onnx_genai_server::{
    AppState, ChatCompletionRequest, ServerConfig, app, build_generate_request,
    parse_assistant_output,
};
use serde_json::{Value, json};
use std::path::PathBuf;
use tower::ServiceExt;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm")
}

fn static_fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm-scatter")
}

async fn test_app() -> axum::Router {
    let state = AppState::load(&fixture_dir(), Some("tiny-llm".to_string())).unwrap();
    app(state)
}

async fn static_test_app() -> axum::Router {
    let state =
        AppState::load(&static_fixture_dir(), Some("tiny-llm-scatter".to_string())).unwrap();
    app(state)
}

async fn test_app_with_config(config: ServerConfig) -> axum::Router {
    let state =
        AppState::load_with_config(&fixture_dir(), Some("tiny-llm".to_string()), config).unwrap();
    app(state)
}

fn sse_data_lines(text: &str) -> Vec<&str> {
    text.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .collect()
}

fn sse_json_chunks(text: &str) -> Vec<Value> {
    sse_data_lines(text)
        .into_iter()
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect()
}

fn chat_request(body: Value) -> ChatCompletionRequest {
    serde_json::from_value(body).unwrap()
}

async fn post_chat_json(app: axum::Router, body: Value) -> Value {
    let response = app
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
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    serde_json::from_slice(&body).unwrap()
}

fn direct_static_completion(body: Value) -> String {
    let request = chat_request(body);
    let generate_request = build_generate_request(&request);
    let mut engine = Engine::from_dir(&static_fixture_dir(), EngineConfig::default()).unwrap();
    engine.generate(generate_request).unwrap().text
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
async fn chat_completions_rejects_max_tokens_over_server_cap() {
    let app = test_app_with_config(ServerConfig {
        max_output_tokens: 2,
        max_sessions: 8,
        max_pending: 8,
    })
    .await;
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
                        "max_tokens": 3
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert!(
        json["error"]["message"]
            .as_str()
            .unwrap()
            .contains("server cap of 2"),
        "{json}"
    );
}

#[tokio::test]
async fn session_ids_are_random_csprng_tokens() {
    let app = test_app().await;
    let mut ids = Vec::new();
    for _ in 0..8 {
        ids.push(create_http_session(app.clone()).await);
    }

    let mut values = Vec::new();
    for id in &ids {
        let token = id.strip_prefix("sess-").expect("session id prefix");
        assert_eq!(token.len(), 32, "{id}");
        assert!(token.chars().all(|ch| ch.is_ascii_hexdigit()), "{id}");
        values.push(u128::from_str_radix(token, 16).unwrap());
    }
    let unique = ids.iter().collect::<std::collections::HashSet<_>>();
    assert_eq!(unique.len(), ids.len(), "{ids:?}");
    assert!(
        values.windows(2).all(|pair| pair[0].abs_diff(pair[1]) != 1),
        "{ids:?}"
    );
}

#[tokio::test]
async fn sessions_are_lru_evicted_at_configured_cap() {
    let app = test_app_with_config(ServerConfig {
        max_output_tokens: 16,
        max_sessions: 2,
        max_pending: 8,
    })
    .await;
    let first = create_http_session(app.clone()).await;
    let second = create_http_session(app.clone()).await;

    let touch_first = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .header("X-Session-Id", &first)
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
    assert_eq!(touch_first.status(), StatusCode::OK);

    let third = create_http_session(app.clone()).await;

    let evicted = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/v1/sessions/{second}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(evicted.status(), StatusCode::NOT_FOUND);

    for id in [first, third] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri(format!("/v1/sessions/{id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NO_CONTENT);
    }
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
async fn concurrent_static_cache_chat_completions_share_batched_driver() {
    let app = static_test_app().await;
    let requests = [
        json!({"model": "tiny-llm-scatter", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 2}),
        json!({"model": "tiny-llm-scatter", "messages": [{"role": "user", "content": "world"}], "max_tokens": 3}),
        json!({"model": "tiny-llm-scatter", "messages": [{"role": "user", "content": "tok16"}], "max_tokens": 1}),
        json!({"model": "tiny-llm-scatter", "messages": [{"role": "user", "content": "tok17"}], "max_tokens": 4}),
    ];
    let expected = requests
        .iter()
        .cloned()
        .map(direct_static_completion)
        .collect::<Vec<_>>();

    let (a, b, c, d) = tokio::join!(
        post_chat_json(app.clone(), requests[0].clone()),
        post_chat_json(app.clone(), requests[1].clone()),
        post_chat_json(app.clone(), requests[2].clone()),
        post_chat_json(app.clone(), requests[3].clone()),
    );
    let responses = [a, b, c, d];

    for (response, expected_text) in responses.iter().zip(expected) {
        assert_eq!(response["object"], "chat.completion");
        assert_eq!(response["model"], "tiny-llm-scatter");
        assert_eq!(
            response["choices"][0]["message"]["content"]
                .as_str()
                .unwrap(),
            expected_text
        );
    }
}

#[test]
fn response_format_maps_to_generate_constraint_only_for_json_object() {
    let json_request = chat_request(json!({
        "model": "tiny-llm",
        "messages": [{"role": "user", "content": "hello"}],
        "response_format": {"type": "json_object"}
    }));
    let text_request = chat_request(json!({
        "model": "tiny-llm",
        "messages": [{"role": "user", "content": "hello"}],
        "response_format": {"type": "text"}
    }));
    let absent_request = chat_request(json!({
        "model": "tiny-llm",
        "messages": [{"role": "user", "content": "hello"}]
    }));

    assert_eq!(
        build_generate_request(&json_request).options.constraint,
        Some(GenerateConstraint::Json)
    );
    assert_eq!(
        build_generate_request(&text_request).options.constraint,
        None
    );
    assert_eq!(
        build_generate_request(&absent_request).options.constraint,
        None
    );
}

#[test]
fn forced_specific_tool_choice_builds_lark_tool_call_constraint() {
    let request = chat_request(json!({
        "model": "tiny-llm",
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                }
            }
        }],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
    }));

    let Some(GenerateConstraint::Lark(grammar)) =
        build_generate_request(&request).options.constraint
    else {
        panic!("expected forced tool_choice to build a Lark constraint");
    };
    assert!(
        grammar.contains("start: \"<tool_call>\\n\" tool \"\\n</tool_call>\""),
        "{grammar}"
    );
    assert!(grammar.contains("tool: %json"), "{grammar}");
    let schema_text = grammar.split_once("tool: %json ").unwrap().1.trim();
    let schema: Value = serde_json::from_str(schema_text).unwrap();
    assert_eq!(schema["properties"]["name"]["enum"][0], "get_weather");
    assert_eq!(schema["properties"]["arguments"]["required"][0], "location");
    assert_eq!(
        schema["properties"]["arguments"]["properties"]["location"]["type"],
        "string"
    );
}

#[test]
fn required_tool_choice_with_multiple_tools_allows_any_tool_schema() {
    let request = chat_request(json!({
        "model": "tiny-llm",
        "messages": [{"role": "user", "content": "pick a tool"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "parameters": {"type": "object", "properties": {"zone": {"type": "string"}}, "required": ["zone"]}
                }
            }
        ],
        "tool_choice": "required"
    }));

    let Some(GenerateConstraint::Lark(grammar)) =
        build_generate_request(&request).options.constraint
    else {
        panic!("expected forced tool_choice to build a Lark constraint");
    };
    let schema_text = grammar.split_once("tool: %json ").unwrap().1.trim();
    let schema: Value = serde_json::from_str(schema_text).unwrap();
    let any_of = schema["anyOf"].as_array().unwrap();
    assert_eq!(any_of.len(), 2);
    assert_eq!(any_of[0]["properties"]["name"]["enum"][0], "get_weather");
    assert_eq!(any_of[1]["properties"]["name"]["enum"][0], "get_time");
}

#[test]
fn auto_and_none_tool_choice_do_not_constrain_generation() {
    let tool = json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object"}
        }
    });
    let auto_request = chat_request(json!({
        "model": "tiny-llm",
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [tool.clone()],
        "tool_choice": "auto"
    }));
    let none_request = chat_request(json!({
        "model": "tiny-llm",
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [tool],
        "tool_choice": "none"
    }));

    assert_eq!(
        build_generate_request(&auto_request).options.constraint,
        None
    );
    assert_eq!(
        build_generate_request(&none_request).options.constraint,
        None
    );
    let GeneratePrompt::Text(prompt) = build_generate_request(&none_request).prompt else {
        panic!("expected text prompt");
    };
    assert!(!prompt.contains("<|tools|>"), "{prompt}");
}

#[test]
fn chat_request_with_tools_renders_tool_schema_in_prompt() {
    let request = chat_request(json!({
        "model": "tiny-llm",
        "messages": [
            {"role": "user", "content": "What is the weather?"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Seattle\"}"}
                }]
            },
            {"role": "tool", "tool_call_id": "call_0", "content": "{\"temp\":72}"}
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        }],
        "tool_choice": "auto"
    }));

    let generate_request = build_generate_request(&request);
    let GeneratePrompt::Text(prompt) = generate_request.prompt else {
        panic!("expected text prompt");
    };
    assert!(prompt.contains("<|tools|>"), "{prompt}");
    assert!(prompt.contains("get_weather"), "{prompt}");
    assert!(
        prompt.contains("\"city\":{\"type\":\"string\"}"),
        "{prompt}"
    );
    assert!(prompt.contains("<|tool|>"), "{prompt}");
    assert!(prompt.contains("tool_call_id: call_0"), "{prompt}");
}

#[test]
fn parser_converts_qwen_tool_call_blocks_to_openai_tool_calls() {
    let parsed = parse_assistant_output(
        r#"Thinking...
<tool_call>
{"name":"read_file","arguments":{"path":"src/lib.rs"}}
</tool_call>
<tool_call>
{"name":"write_file","arguments":{"path":"src/lib.rs","content":"ok"}}
</tool_call>"#
            .to_string(),
        "stop",
    );

    assert_eq!(parsed.finish_reason, "tool_calls");
    assert!(parsed.content.is_none());
    let calls = parsed.tool_calls.unwrap();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].id, "call_0");
    assert_eq!(calls[0].kind, "function");
    assert_eq!(calls[0].function.name, "read_file");
    assert_eq!(calls[0].function.arguments, r#"{"path":"src/lib.rs"}"#);
    assert_eq!(calls[1].id, "call_1");
    assert_eq!(calls[1].function.name, "write_file");
    let second_args: Value = serde_json::from_str(&calls[1].function.arguments).unwrap();
    assert_eq!(second_args["path"], "src/lib.rs");
    assert_eq!(second_args["content"], "ok");
}

#[tokio::test]
#[ignore = "requires gitignored models/qwen2.5-0.5b real model fixture"]
async fn qwen_real_model_tool_use_chain_end_to_end() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/qwen2.5-0.5b");
    assert!(
        model_dir.exists(),
        "build the real model fixture with scripts/build_qwen.sh"
    );
    let app = app(AppState::load(&model_dir, Some("qwen2.5-0.5b".to_string())).unwrap());
    let tool = json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    });
    let first_messages = json!([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather in Paris? Use the tool."}
    ]);

    let forced = post_chat_json(
        app.clone(),
        json!({
            "model": "qwen2.5-0.5b",
            "messages": first_messages,
            "tools": [tool.clone()],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
        }),
    )
    .await;
    assert_eq!(forced["choices"][0]["finish_reason"], "tool_calls");
    let tool_call = forced["choices"][0]["message"]["tool_calls"][0].clone();
    assert_eq!(tool_call["function"]["name"], "get_weather");
    let args: Value =
        serde_json::from_str(tool_call["function"]["arguments"].as_str().unwrap()).unwrap();
    assert!(args["location"].is_string(), "{args}");

    let final_response = post_chat_json(
        app,
        json!({
            "model": "qwen2.5-0.5b",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's the weather in Paris? Use the tool."},
                {"role": "assistant", "content": null, "tool_calls": [tool_call]},
                {"role": "tool", "tool_call_id": "call_0", "content": "{\"temp\":18,\"unit\":\"celsius\"}"}
            ],
            "tools": [tool],
            "tool_choice": "auto"
        }),
    )
    .await;
    assert_eq!(final_response["choices"][0]["finish_reason"], "stop");
    assert!(
        final_response["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .contains("18"),
        "{final_response}"
    );
}

#[tokio::test]
async fn chat_completions_response_format_json_object_returns_valid_json() {
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
                        "max_tokens": 14,
                        "response_format": {"type": "json_object"}
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["object"], "chat.completion");
    assert_eq!(json["choices"][0]["message"]["role"], "assistant");
    let content = json["choices"][0]["message"]["content"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(content).unwrap();
    assert!(parsed.is_object(), "{content}");
}

#[tokio::test]
async fn streaming_chat_completions_response_format_json_object_streams_valid_json() {
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
                        "max_tokens": 14,
                        "stream": true,
                        "response_format": {"type": "json_object"}
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
    let data_lines = sse_data_lines(&text);
    assert_eq!(data_lines.last(), Some(&"[DONE]"), "{text}");

    let chunks = sse_json_chunks(&text);
    let content: String = chunks
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
        .collect();
    let parsed: Value = serde_json::from_str(&content).unwrap();
    assert!(parsed.is_object(), "{content}");
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

    let data_lines = sse_data_lines(&text);
    assert_eq!(data_lines.last(), Some(&"[DONE]"));
    let chunks = sse_json_chunks(&text);
    let content_chunks = chunks
        .iter()
        .filter(|chunk| chunk["choices"][0]["delta"].get("content").is_some())
        .count();
    assert!(content_chunks <= 1, "{text}");
    assert_eq!(
        chunks.last().unwrap()["choices"][0]["finish_reason"],
        "length"
    );
}

#[tokio::test]
async fn streaming_chat_completions_stop_sequence_finishes_before_max_tokens() {
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
                        "max_tokens": 10,
                        "stream": true,
                        "stop": "tok22"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    let data_lines = sse_data_lines(&text);
    assert_eq!(data_lines.last(), Some(&"[DONE]"), "{text}");

    let chunks = sse_json_chunks(&text);
    let content: String = chunks
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
        .collect();
    let content_chunks = chunks
        .iter()
        .filter(|chunk| chunk["choices"][0]["delta"].get("content").is_some())
        .count();

    assert!(content_chunks < 10, "{text}");
    assert!(!content.contains("tok22"), "{text}");
    assert_eq!(
        chunks.last().unwrap()["choices"][0]["finish_reason"],
        "stop"
    );
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

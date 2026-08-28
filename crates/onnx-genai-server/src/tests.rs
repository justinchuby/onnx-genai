use crate::{
    AppState, ChatCompletionRequest, CompletionRequest, EmbeddingEncodingFormat, EmbeddingInput,
    EmbeddingRequest, EmbeddingResponse, EmbeddingUsage, EmbeddingVector, OrtSessionWorkerCount,
    ServerConfig, app, build_generate_request,
    driver::{DriverCommand, EngineDriver},
    models_config::ModelSpec,
    routes::{CompletionGeneration, collect_generation_result, prepare_completion},
    sse::StopBoundaryBuffer,
    sse::{content_chunk, done_chunk, tool_call_delta_chunks},
    types::{ChatMessageToolCall, ChatMessageToolCallFunction},
};
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
#[cfg(feature = "native-backend")]
use onnx_genai::engine::EngineDecodeBackend;
use onnx_genai::{Engine, EngineConfig};
use serde_json::{Value, json};
use std::{collections::BTreeMap, io::Cursor, path::PathBuf, sync::Arc, time::Duration};
use tokio::{sync::mpsc, time::timeout};
use tower::ServiceExt;

fn tiny_state() -> AppState {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm");
    AppState::load(&model_dir, Some("tiny-llm".to_string())).expect("load fixture")
}

fn tiny_state_with_debug() -> AppState {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm");
    AppState::load_with_config(
        &model_dir,
        Some("tiny-llm".to_string()),
        ServerConfig {
            enable_debug_endpoints: true,
            ..ServerConfig::default()
        },
    )
    .expect("load fixture with debug")
}

fn resource_state(allow_runtime_override: bool) -> AppState {
    let model_dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-mtp-full");
    let engine_config = EngineConfig {
        allow_runtime_override,
        ..EngineConfig::default()
    };
    AppState::load_with_config(
        &model_dir,
        Some("tiny-mtp-full".to_string()),
        ServerConfig {
            enable_admin_endpoints: true,
            engine_config,
            ..ServerConfig::default()
        },
    )
    .expect("load resource API fixture")
}

fn sse_json_events(body: &[u8]) -> Vec<Value> {
    std::str::from_utf8(body)
        .unwrap()
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect()
}

fn sse_events_from_chunks(
    chunks: impl IntoIterator<Item = crate::sse::ChatCompletionChunk>,
) -> Vec<Value> {
    let body = chunks
        .into_iter()
        .map(|chunk| format!("data: {}\n\n", serde_json::to_string(&chunk).unwrap()))
        .collect::<String>();
    sse_json_events(body.as_bytes())
}

fn test_tool_call(id: &str, name: &str, arguments: &str) -> ChatMessageToolCall {
    ChatMessageToolCall {
        id: id.to_string(),
        kind: "function".to_string(),
        function: ChatMessageToolCallFunction {
            name: name.to_string(),
            arguments: arguments.to_string(),
        },
    }
}

#[test]
fn tool_call_stream_deltas_emit_metadata_arguments_and_distinct_indices() {
    let calls = vec![
        test_tool_call("call_weather", "weather", r#"{"city":"Paris"}"#),
        test_tool_call("call_time", "time", r#"{"timezone":"UTC"}"#),
    ];
    let mut chunks = tool_call_delta_chunks("chatcmpl-test", 1, "test", calls.clone());
    chunks.push(done_chunk("chatcmpl-test", 1, "test", "tool_calls"));
    let events = sse_events_from_chunks(chunks);

    let tool_deltas = events
        .iter()
        .filter_map(|event| event["choices"][0]["delta"]["tool_calls"].as_array())
        .map(|calls| &calls[0])
        .collect::<Vec<_>>();
    let metadata = tool_deltas
        .iter()
        .filter(|call| call["function"]["name"].is_string())
        .collect::<Vec<_>>();
    assert_eq!(metadata.len(), calls.len());
    for (index, call) in calls.iter().enumerate() {
        assert_eq!(metadata[index]["index"], index);
        assert_eq!(metadata[index]["id"], call.id);
        assert_eq!(metadata[index]["type"], "function");
        assert_eq!(metadata[index]["function"]["name"], call.function.name);
        assert_eq!(metadata[index]["function"]["arguments"], "");
    }

    let mut arguments_by_index = BTreeMap::new();
    for call in tool_deltas
        .iter()
        .filter(|call| call["function"]["name"].is_null())
    {
        let index = call["index"].as_u64().unwrap() as usize;
        arguments_by_index
            .entry(index)
            .or_insert_with(String::new)
            .push_str(call["function"]["arguments"].as_str().unwrap());
        assert!(call.get("id").is_none());
        assert!(call.get("type").is_none());
    }
    for (index, call) in calls.iter().enumerate() {
        assert_eq!(arguments_by_index[&index], call.function.arguments);
    }
    assert_eq!(
        events.last().unwrap()["choices"][0]["finish_reason"],
        "tool_calls"
    );
}

#[test]
fn content_only_stream_deltas_remain_content_with_stop_finish_reason() {
    let events = sse_events_from_chunks([
        content_chunk(
            "chatcmpl-test",
            1,
            "test",
            "normal content".to_string(),
            None,
        ),
        done_chunk("chatcmpl-test", 1, "test", "stop"),
    ]);

    assert_eq!(
        events[0]["choices"][0]["delta"]["content"],
        "normal content"
    );
    assert!(events[0]["choices"][0]["delta"]["tool_calls"].is_null());
    assert_eq!(events[1]["choices"][0]["finish_reason"], "stop");
}

fn tiny_png_data_uri() -> String {
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    use image::{DynamicImage, ImageFormat, Rgb, RgbImage};

    let image = RgbImage::from_pixel(3, 4, Rgb([64, 128, 255]));
    let mut png = Cursor::new(Vec::new());
    DynamicImage::ImageRgb8(image)
        .write_to(&mut png, ImageFormat::Png)
        .unwrap();
    format!(
        "data:image/png;base64,{}",
        STANDARD.encode(png.into_inner())
    )
}

fn tiny_wav_bytes() -> Vec<u8> {
    let samples = [0_i16; 1_280];
    let data_len = (samples.len() * 2) as u32;
    let mut wav = Vec::with_capacity(44 + data_len as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16_u32.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes());
    wav.extend_from_slice(&1_u16.to_le_bytes());
    wav.extend_from_slice(&16_000_u32.to_le_bytes());
    wav.extend_from_slice(&32_000_u32.to_le_bytes());
    wav.extend_from_slice(&2_u16.to_le_bytes());
    wav.extend_from_slice(&16_u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    for sample in samples {
        wav.extend_from_slice(&sample.to_le_bytes());
    }
    wav
}

fn tiny_wav_base64() -> String {
    use base64::{Engine as _, engine::general_purpose::STANDARD};
    STANDARD.encode(tiny_wav_bytes())
}

fn multipart_audio_body(response_format: &str) -> (String, Vec<u8>) {
    multipart_audio_body_for_model("tiny-whisper", response_format)
}

fn multipart_audio_body_for_model(model: &str, response_format: &str) -> (String, Vec<u8>) {
    let boundary = "onnx-genai-audio-boundary";
    let mut body = Vec::new();
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{model}\r\n\
             --{boundary}\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\n{response_format}\r\n\
             --{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"tiny.wav\"\r\nContent-Type: audio/wav\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(&tiny_wav_bytes());
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    (boundary.to_string(), body)
}

#[test]
fn multimodal_message_parses_text_and_data_image_parts() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "tiny-vlm",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is shown?"},
                {"type": "image_url", "image_url": {"url": tiny_png_data_uri()}}
            ]
        }]
    }))
    .unwrap();

    assert_eq!(
        request.messages[0]
            .content
            .as_ref()
            .expect("content")
            .text(),
        "What is shown?"
    );
    assert_eq!(request.image_urls().len(), 1);
    assert!(request.image_urls()[0].starts_with("data:image/png;base64,"));
}

#[test]
fn multimodal_message_parses_base64_wav_input_audio_part() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "tiny-whisper",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe this"},
                {"type": "input_audio", "input_audio": {
                    "data": tiny_wav_base64(),
                    "format": "wav"
                }}
            ]
        }]
    }))
    .unwrap();

    assert_eq!(
        request.messages[0]
            .content
            .as_ref()
            .expect("content")
            .text(),
        "Transcribe this"
    );
    let audio = request.input_audio();
    assert_eq!(audio.len(), 1);
    assert_eq!(audio[0].format, "wav");
    assert!(!audio[0].data.is_empty());
}

#[test]
fn chat_sampling_controls_round_trip_into_generate_options() {
    let request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "tiny-llm",
        "messages": [{"role": "user", "content": "hello"}],
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 40,
        "min_p": 0.1,
        "top_a": 0.2,
        "typical_p": 0.7,
        "repetition_penalty": 1.15,
        "frequency_penalty": 0.3,
        "presence_penalty": 0.4,
        "seed": 1234,
        "dry_multiplier": 0.5,
        "dry_base": 1.75,
        "dry_allowed_length": 3,
        "dry_sequence_breakers": [13, 42],
        "mirostat": 2,
        "mirostat_tau": 4.5,
        "mirostat_eta": 0.2,
        "xtc_probability": 0.6,
        "xtc_threshold": 0.15
    }))
    .unwrap();

    let options = build_generate_request(&request).options;
    assert_eq!(options.temperature, 0.8);
    assert_eq!(options.top_p, 0.9);
    assert_eq!(options.top_k, 40);
    assert_eq!(options.min_p, 0.1);
    assert_eq!(options.top_a, 0.2);
    assert_eq!(options.typical_p, 0.7);
    assert_eq!(options.repetition_penalty, 1.15);
    assert_eq!(options.frequency_penalty, 0.3);
    assert_eq!(options.presence_penalty, 0.4);
    assert!(!options.greedy);
    assert_eq!(options.seed, Some(1234));

    let dry = options.dry.expect("DRY enabled");
    assert_eq!(dry.multiplier, 0.5);
    assert_eq!(dry.base, 1.75);
    assert_eq!(dry.allowed_length, 3);
    assert_eq!(dry.sequence_breakers, vec![13, 42]);

    let mirostat = options.mirostat.expect("Mirostat enabled");
    assert_eq!(mirostat.tau, 4.5);
    assert_eq!(mirostat.eta, 0.2);
    assert_eq!(mirostat.version, onnx_genai_engine::MirostatVersion::V2);

    let xtc = options.xtc.expect("XTC enabled");
    assert_eq!(xtc.probability, 0.6);
    assert_eq!(xtc.threshold, 0.15);
}

#[test]
fn chat_sampling_defaults_stay_greedy_while_seed_enables_sampling() {
    let request = |seed| {
        serde_json::from_value::<ChatCompletionRequest>(json!({
            "model": "tiny-llm",
            "messages": [{"role": "user", "content": "hello"}],
            "seed": seed
        }))
        .unwrap()
    };

    assert!(build_generate_request(&request(Value::Null)).options.greedy);
    let seeded = build_generate_request(&request(json!(17))).options;
    assert!(!seeded.greedy);
    assert_eq!(seeded.seed, Some(17));
}

#[test]
fn transcription_json_response_has_openai_shape() {
    let response = crate::types::AudioTranscriptionResponse {
        text: "hello".to_string(),
    };
    assert_eq!(
        serde_json::to_value(response).unwrap(),
        json!({"text": "hello"})
    );
}

#[test]
fn embedding_request_accepts_openai_input_variants_and_defaults_to_float() {
    let single: EmbeddingRequest = serde_json::from_value(json!({
        "model": "embedder",
        "input": "hello"
    }))
    .unwrap();
    assert!(matches!(single.input, EmbeddingInput::String(_)));
    assert_eq!(single.encoding_format, EmbeddingEncodingFormat::Float);

    let strings: EmbeddingRequest = serde_json::from_value(json!({
        "model": "embedder",
        "input": ["hello", "world"],
        "encoding_format": "base64",
        "dimensions": 64
    }))
    .unwrap();
    assert!(matches!(strings.input, EmbeddingInput::Strings(_)));
    assert_eq!(strings.encoding_format, EmbeddingEncodingFormat::Base64);
    assert_eq!(strings.dimensions, Some(64));

    let tokens: EmbeddingRequest = serde_json::from_value(json!({
        "model": "embedder",
        "input": [[1, 2], [3, 4]]
    }))
    .unwrap();
    assert!(matches!(tokens.input, EmbeddingInput::TokenArrays(_)));

    assert!(
        serde_json::from_value::<EmbeddingRequest>(json!({
            "model": "embedder",
            "input": "hello",
            "encoding_format": "hex"
        }))
        .is_err()
    );
}

#[tokio::test]
async fn chat_logprobs_match_openai_shape_and_are_opt_in() {
    let router = app(tiny_state());
    let response = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "tiny-llm",
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 2,
                        "temperature": 0.0,
                        "logprobs": true,
                        "top_logprobs": 2
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    let content = body["choices"][0]["logprobs"]["content"]
        .as_array()
        .unwrap();
    assert_eq!(
        content.len(),
        body["usage"]["completion_tokens"].as_u64().unwrap() as usize
    );
    for token in content {
        let token_text = token["token"].as_str().unwrap();
        let bytes = token["bytes"].as_array().unwrap();
        assert_eq!(
            bytes
                .iter()
                .map(|byte| byte.as_u64().unwrap() as u8)
                .collect::<Vec<_>>(),
            token_text.as_bytes()
        );
        assert!(token["logprob"].is_number());
        let top_logprobs = token["top_logprobs"].as_array().unwrap();
        assert!(top_logprobs.len() <= 2);
        for alternative in top_logprobs {
            let token_text = alternative["token"].as_str().unwrap();
            assert_eq!(
                alternative["bytes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|byte| byte.as_u64().unwrap() as u8)
                    .collect::<Vec<_>>(),
                token_text.as_bytes()
            );
        }
    }

    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "tiny-llm",
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 1,
                        "temperature": 0.0
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert!(body["choices"][0]["logprobs"].is_null());
}

#[tokio::test]
async fn completion_logprobs_match_legacy_openai_shape() {
    let response = app(tiny_state())
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "tiny-llm",
                        "prompt": "hello",
                        "max_tokens": 3,
                        "temperature": 0.0,
                        "logprobs": 2
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    let logprobs = &body["choices"][0]["logprobs"];
    let tokens = logprobs["tokens"].as_array().unwrap();
    let token_logprobs = logprobs["token_logprobs"].as_array().unwrap();
    let top_logprobs = logprobs["top_logprobs"].as_array().unwrap();
    let offsets = logprobs["text_offset"].as_array().unwrap();
    assert_eq!(tokens.len(), 3);
    assert_eq!(token_logprobs.len(), tokens.len());
    assert_eq!(top_logprobs.len(), tokens.len());
    assert_eq!(offsets.len(), tokens.len());
    let mut expected_offset = 0;
    for index in 0..tokens.len() {
        assert_eq!(offsets[index].as_u64().unwrap() as usize, expected_offset);
        expected_offset += tokens[index].as_str().unwrap().len();
        assert!(top_logprobs[index].as_object().unwrap().len() <= 2);
    }
}

#[tokio::test]
async fn buffered_and_sse_routes_deliver_only_committed_workflow_publications() {
    let request = |stream| {
        Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                json!({
                    "model": "tiny-llm",
                    "prompt": "hello",
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "stream": stream
                })
                .to_string(),
            ))
            .unwrap()
    };

    let buffered = app(tiny_state()).oneshot(request(false)).await.unwrap();
    assert_eq!(buffered.status(), StatusCode::OK);
    let buffered: Value =
        serde_json::from_slice(&to_bytes(buffered.into_body(), usize::MAX).await.unwrap()).unwrap();
    let publications = buffered["workflow_outputs"]
        .as_array()
        .expect("buffered route exposes committed workflow publications");
    assert!(!publications.is_empty());
    assert!(
        publications
            .iter()
            .all(|publication| publication["finality"] == "final"),
        "{publications:#?}"
    );

    let streamed = app(tiny_state()).oneshot(request(true)).await.unwrap();
    assert_eq!(streamed.status(), StatusCode::OK);
    let body = to_bytes(streamed.into_body(), usize::MAX).await.unwrap();
    let text = std::str::from_utf8(&body).unwrap();
    let streamed_publications = text
        .split("\n\n")
        .filter(|event| event.lines().any(|line| line == "event: workflow_output"))
        .map(|event| {
            let data = event
                .lines()
                .find_map(|line| line.strip_prefix("data: "))
                .expect("workflow output event has data");
            serde_json::from_str::<Value>(data).expect("workflow output event is JSON")
        })
        .collect::<Vec<_>>();
    assert_eq!(streamed_publications.as_slice(), publications.as_slice());
}

#[tokio::test]
async fn streaming_chat_and_completion_chunks_include_logprobs() {
    let chat = app(tiny_state())
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "tiny-llm",
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 2,
                        "temperature": 0.0,
                        "stream": true,
                        "logprobs": true,
                        "top_logprobs": 1
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let events = sse_json_events(&to_bytes(chat.into_body(), usize::MAX).await.unwrap());
    let content_events = events
        .iter()
        .filter(|event| event["choices"][0]["delta"]["content"].is_string())
        .collect::<Vec<_>>();
    assert_eq!(content_events.len(), 2);
    for event in content_events {
        let record = &event["choices"][0]["logprobs"]["content"][0];
        assert_eq!(event["choices"][0]["delta"]["content"], record["token"]);
        assert_eq!(record["top_logprobs"].as_array().unwrap().len(), 1);
        assert!(record["bytes"].is_array());
    }

    let completion = app(tiny_state())
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "tiny-llm",
                        "prompt": "hello",
                        "max_tokens": 2,
                        "temperature": 0.0,
                        "stream": true,
                        "logprobs": 1
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let events = sse_json_events(&to_bytes(completion.into_body(), usize::MAX).await.unwrap());
    let token_events = events
        .iter()
        .filter(|event| {
            !event["choices"][0]["text"]
                .as_str()
                .unwrap_or_default()
                .is_empty()
        })
        .collect::<Vec<_>>();
    assert_eq!(token_events.len(), 2);
    for event in token_events {
        let logprobs = &event["choices"][0]["logprobs"];
        assert_eq!(logprobs["tokens"].as_array().unwrap().len(), 1);
        assert_eq!(logprobs["token_logprobs"].as_array().unwrap().len(), 1);
        assert_eq!(logprobs["top_logprobs"].as_array().unwrap().len(), 1);
        assert_eq!(logprobs["text_offset"].as_array().unwrap().len(), 1);
    }
}

#[tokio::test]
async fn accepted_streams_preserve_first_chunk_and_chat_protocol_order() {
    let chat = app(tiny_state())
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "tiny-llm",
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 2,
                        "temperature": 0.0,
                        "stream": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(chat.status(), StatusCode::OK);
    let chat_events = sse_json_events(&to_bytes(chat.into_body(), usize::MAX).await.unwrap());
    assert_eq!(
        chat_events[0]["choices"][0]["delta"]["role"], "assistant",
        "the role chunk must remain first after admission"
    );
    assert_eq!(
        chat_events
            .iter()
            .filter(|event| event["choices"][0]["delta"]["content"].is_string())
            .count(),
        2,
        "the admission handshake must not consume the first content token"
    );

    let completion = app(tiny_state())
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "tiny-llm",
                        "prompt": "hello",
                        "max_tokens": 2,
                        "temperature": 0.0,
                        "stream": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(completion.status(), StatusCode::OK);
    let completion_events =
        sse_json_events(&to_bytes(completion.into_body(), usize::MAX).await.unwrap());
    assert_eq!(
        completion_events
            .iter()
            .filter(|event| {
                !event["choices"][0]["text"]
                    .as_str()
                    .unwrap_or_default()
                    .is_empty()
            })
            .count(),
        2,
        "the admission handshake must not consume the first completion token"
    );
}

#[tokio::test]
async fn logprobs_validation_enforces_openai_limits() {
    for body in [
        json!({
            "model": "tiny-llm",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1,
            "top_logprobs": 1
        }),
        json!({
            "model": "tiny-llm",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1,
            "logprobs": true,
            "top_logprobs": 21
        }),
    ] {
        let response = app(tiny_state())
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
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    let response = app(tiny_state())
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "tiny-llm",
                        "prompt": "hello",
                        "max_tokens": 1,
                        "logprobs": 6
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn chat_sampling_controls_reject_invalid_ranges() {
    for (field, value) in [
        ("min_p", json!(1.1)),
        ("top_a", json!(-0.1)),
        ("typical_p", json!(1.1)),
        ("repetition_penalty", json!(0.0)),
        ("dry_base", json!(0.5)),
        ("mirostat", json!(3)),
        ("xtc_probability", json!(1.1)),
        ("xtc_threshold", json!(-0.1)),
    ] {
        let mut body = json!({
            "model": "tiny-llm",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1
        });
        body[field] = value;
        let response = app(tiny_state())
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
        assert_eq!(
            response.status(),
            StatusCode::BAD_REQUEST,
            "{field} should be rejected"
        );
    }
}

#[test]
fn embedding_response_serializes_float_and_base64_vectors() {
    use base64::{Engine as _, engine::general_purpose::STANDARD};

    let float = EmbeddingResponse {
        object: "list",
        data: vec![crate::EmbeddingData {
            object: "embedding",
            embedding: EmbeddingVector::from_floats(
                vec![1.0, -2.0],
                EmbeddingEncodingFormat::Float,
            ),
            index: 0,
        }],
        model: "embedder".to_string(),
        usage: EmbeddingUsage {
            prompt_tokens: 2,
            total_tokens: 2,
        },
    };
    let float = serde_json::to_value(float).unwrap();
    assert_eq!(float["object"], "list");
    assert_eq!(float["data"][0]["object"], "embedding");
    assert_eq!(float["data"][0]["embedding"], json!([1.0, -2.0]));
    assert_eq!(float["data"][0]["index"], 0);
    assert_eq!(float["model"], "embedder");
    assert_eq!(
        float["usage"],
        json!({"prompt_tokens": 2, "total_tokens": 2})
    );

    let base64 = EmbeddingVector::from_floats(vec![1.0, -2.0], EmbeddingEncodingFormat::Base64);
    let encoded = serde_json::to_value(base64).unwrap();
    let expected = STANDARD.encode(
        [1.0_f32, -2.0_f32]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    );
    assert_eq!(encoded, expected);
}

#[tokio::test]
async fn embeddings_valid_inputs_fail_on_logits_only_model() {
    let router = app(tiny_state());
    for input in [
        json!("hello"),
        json!(["hello", "world"]),
        json!([[1, 2], [3, 4]]),
    ] {
        let response = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/embeddings")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        json!({
                            "model": "tiny-llm",
                            "input": input,
                            "encoding_format": "base64"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        // tiny-llm is a logits-only model; the engine rejects embedding requests
        // with a descriptive error rather than NOT_IMPLEMENTED.
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("hidden-state output"),
            "{body}"
        );
    }

    for (body, message) in [
        (
            json!({"model": "tiny-llm", "input": []}),
            "embedding input array must not be empty",
        ),
        (
            json!({"model": "tiny-llm", "input": "hello", "dimensions": 0}),
            "dimensions must be greater than zero",
        ),
    ] {
        let response = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/embeddings")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["error"]["message"], message);
    }
}

#[tokio::test]
async fn embeddings_success_path_returns_openai_compatible_response() {
    let model_dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-mtp-full");
    let state = AppState::load(&model_dir, Some("tiny-mtp-full".to_string()))
        .expect("load tiny-mtp-full fixture");
    let router = app(state);

    // Single string input → one embedding entry
    let response = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "tiny-mtp-full",
                        "input": "hello"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["object"], "list");
    assert_eq!(body["model"], "tiny-mtp-full");
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["object"], "embedding");
    assert_eq!(data[0]["index"], 0);
    assert!(data[0]["embedding"].is_array());
    assert!(!data[0]["embedding"].as_array().unwrap().is_empty());
    let usage = &body["usage"];
    assert!(usage["prompt_tokens"].as_u64().unwrap() > 0);
    assert_eq!(usage["prompt_tokens"], usage["total_tokens"]);

    // Array of strings → one entry per input, indices in order, correct token count
    let response = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "tiny-mtp-full",
                        "input": ["hello", "world"]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);
    assert_eq!(data[0]["index"], 0);
    assert_eq!(data[1]["index"], 1);

    // base64 encoding format → embedding is a string, not an array
    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "tiny-mtp-full",
                        "input": "hello",
                        "encoding_format": "base64"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert!(body["data"][0]["embedding"].is_string());
}

#[tokio::test]
async fn transcription_multipart_against_non_audio_model_returns_400() {
    // Send the correct model name (tiny-llm) so routing succeeds, then the handler
    // returns 400 because tiny-llm has no audio input spec.
    let (boundary, body) = multipart_audio_body_for_model("tiny-llm", "json");
    let response = app(tiny_state())
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/transcriptions")
                .header(
                    header::CONTENT_TYPE,
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body: Value = serde_json::from_slice(&body).unwrap();
    let message = body["error"]["message"].as_str().unwrap();
    assert!(message.contains("What:"), "message: {message}");
    assert!(message.contains("How:"), "message: {message}");
    assert!(
        message.contains("input_features"),
        "the missing contract must be named: {message}"
    );
}

#[tokio::test]
async fn speech_endpoint_rejects_streaming_before_execution() {
    let response = app(tiny_state())
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "tiny-llm",
                        "input": "lyrics",
                        "instructions": "music description",
                        "response_format": "wav",
                        "stream": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body: Value = serde_json::from_slice(&body).unwrap();
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("stream=true")
    );
}

#[tokio::test]
async fn speech_endpoint_accepts_only_wav_delivery() {
    let response = app(tiny_state())
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "tiny-llm",
                        "input": "lyrics",
                        "response_format": "mp3"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body: Value = serde_json::from_slice(&body).unwrap();
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("response_format=wav")
    );
}

#[tokio::test]
async fn non_speech_registry_entry_does_not_expose_speech_route() {
    let response = app(tiny_state())
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header("content-type", "application/json")
                .body(Body::from(
                    serde_json::json!({
                        "model": "tiny-llm",
                        "input": "lyrics",
                        "instructions": "music",
                        "response_format": "wav",
                        "stream": false
                    })
                    .to_string(),
                ))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body = String::from_utf8(body.to_vec()).expect("UTF-8");
    assert!(
        body.contains("compatible buffered PCM16 WAV output"),
        "{body}"
    );
}

fn speech_state() -> AppState {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/onnx_genai_workflows/speech_wav");
    AppState::load(&model_dir, Some("speech-wav".to_string())).expect("load speech fixture")
}

/// Raw `/v1/audio/speech` conformance against an ONNX-owned, self-contained
/// workflow package: a text prompt assembled by a generic text-assembly adapter
/// is synthesized into a buffered PCM16 WAV that honours the package's declared
/// `media` contract (audio/wav, two channels, 24 kHz, 16-bit). The runtime only
/// consumes canonical metadata fields, so this holds for any package that
/// declares the same contract.
#[tokio::test]
async fn speech_endpoint_synthesizes_buffered_pcm16_wav() {
    let response = app(speech_state())
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "speech-wav",
                        "input": "hello world",
                        "instructions": "the quick brown fox",
                        "response_format": "wav",
                        "stream": false
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("audio/wav")
    );

    let wav = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert!(wav.len() > 44, "WAV must carry a header plus PCM samples");
    assert_eq!(&wav[0..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    assert_eq!(&wav[12..16], b"fmt ");
    // Audio format tag 1 == PCM.
    assert_eq!(u16::from_le_bytes([wav[20], wav[21]]), 1);
    // Channel count and sample rate come from the declared media contract.
    assert_eq!(u16::from_le_bytes([wav[22], wav[23]]), 2);
    assert_eq!(
        u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]),
        24000
    );
    // pcm_s16_le is 16-bit.
    assert_eq!(u16::from_le_bytes([wav[34], wav[35]]), 16);
    assert_eq!(&wav[36..40], b"data");
}

/// The optional `max_output_units` budget is honoured against the package's
/// declared ceiling. The value is read from the canonical text-assembly
/// contract (`speech_processor.json`), not from any model-family default, so an
/// explicit in-range budget still renders a valid buffered WAV.
#[tokio::test]
async fn speech_endpoint_honors_explicit_max_output_units() {
    let response = app(speech_state())
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "speech-wav",
                        "input": "hello world",
                        "instructions": "the quick brown fox",
                        "response_format": "wav",
                        "stream": false,
                        "max_output_units": 3
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("audio/wav")
    );
    let wav = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(&wav[0..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
}

/// The raw speech route fails closed when `max_output_units` falls outside the
/// canonical `[1, max_output_units]` budget declared by the package. Both the
/// zero and over-ceiling cases are rejected before any execution, and the error
/// names the declared ceiling generically (no model-specific constant).
#[tokio::test]
async fn speech_endpoint_rejects_out_of_range_max_output_units() {
    for units in [0_usize, 9999_usize] {
        let response = app(speech_state())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/audio/speech")
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(
                        json!({
                            "model": "speech-wav",
                            "input": "hello world",
                            "response_format": "wav",
                            "stream": false,
                            "max_output_units": units
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(
            response.status(),
            StatusCode::BAD_REQUEST,
            "max_output_units={units} must be rejected"
        );
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("max_output_units must be between 1 and"),
            "message: {}",
            body["error"]["message"]
        );
    }
}

/// The declared ceiling itself is exactly the inclusive upper bound: a request
/// for the full budget (64) renders a valid WAV, while the first value beyond it
/// (65) fails closed with an error that names the declared ceiling verbatim.
#[tokio::test]
async fn speech_endpoint_enforces_declared_max_output_units_ceiling() {
    // The package declares `max_output_units: 64`; 64 is admitted.
    let response = app(speech_state())
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "speech-wav",
                        "input": "hello world",
                        "response_format": "wav",
                        "stream": false,
                        "max_output_units": 64
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "the declared ceiling of 64 must be admitted"
    );
    let wav = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(&wav[0..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");

    // 65 is the first value beyond the ceiling and must be rejected.
    let response = app(speech_state())
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "speech-wav",
                        "input": "hello world",
                        "response_format": "wav",
                        "stream": false,
                        "max_output_units": 65
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body: Value = serde_json::from_slice(&body).unwrap();
    let message = body["error"]["message"].as_str().unwrap();
    assert!(
        message.contains("64"),
        "error must name the declared ceiling of 64: {message}"
    );
}

fn speech_state_from(fixture: &str, id: &str) -> AppState {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/onnx_genai_workflows")
        .join(fixture);
    AppState::load(&model_dir, Some(id.to_string())).expect("load speech fixture")
}

fn try_load_speech_fixture(fixture: &str, id: &str) -> anyhow::Result<AppState> {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/onnx_genai_workflows")
        .join(fixture);
    AppState::load(&model_dir, Some(id.to_string()))
}

/// Admission and execution must bind the *exact* compatible output resolved at
/// load time, not merely "any" role: audio output. The mixed fixture declares
/// two role: audio outputs where the map-first key (`audio`) is an incompatible
/// raw float stream and only the second (`waveform`) is a compatible buffered
/// PCM16 WAV (mono, 16 kHz, 64 samples => 128 bytes of PCM). A correct binding
/// encodes `waveform`; the old "first role: audio" behaviour would have selected
/// the incompatible `audio` output instead.
#[tokio::test]
async fn speech_endpoint_encodes_resolved_output_not_map_first_role_audio() {
    let response = app(speech_state_from("speech_wav_mixed_audio", "speech-mixed"))
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/speech")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "speech-mixed",
                        "input": "hello world",
                        "instructions": "the quick brown fox",
                        "response_format": "wav",
                        "stream": false
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok()),
        Some("audio/wav")
    );
    let wav = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    assert_eq!(&wav[0..4], b"RIFF");
    assert_eq!(&wav[8..12], b"WAVE");
    assert_eq!(&wav[12..16], b"fmt ");
    // The resolved `waveform` output is mono at 16 kHz; the incompatible,
    // map-first `audio` output would have been stereo at 24 kHz.
    assert_eq!(
        u16::from_le_bytes([wav[22], wav[23]]),
        1,
        "must encode the resolved mono waveform output, not the stereo audio output"
    );
    assert_eq!(
        u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]),
        16000,
        "must encode the resolved output's 16 kHz sample rate"
    );
    assert_eq!(&wav[36..40], b"data");
    // 64 samples * 1 channel * 2 bytes (pcm_s16_le) == 128 bytes of PCM data.
    assert_eq!(
        u32::from_le_bytes([wav[40], wav[41], wav[42], wav[43]]),
        128,
        "must carry exactly the resolved mono output's PCM payload"
    );
}

/// Fail closed when more than one workflow output declares a compatible buffered
/// PCM16 WAV audio contract: the load cannot decide which output the speech
/// adapter binds to, so it rejects with a clear error naming both candidates.
#[test]
fn speech_load_rejects_ambiguous_audio_outputs() {
    let error = match try_load_speech_fixture("speech_wav_two_audio", "speech-two-audio") {
        Ok(_) => panic!("ambiguous compatible audio outputs must be rejected at load"),
        Err(error) => format!("{error:#}"),
    };
    assert!(
        error.contains("exactly one is required to bind the speech text-assembly adapter"),
        "error must explain the ambiguity: {error}"
    );
}

/// Fail closed when more than one component implements the text-assembly
/// contract: the speech processor cannot be resolved unambiguously, so the load
/// rejects with a clear error.
#[test]
fn speech_load_rejects_multiple_text_assembly_adapters() {
    let error = match try_load_speech_fixture("speech_wav_two_adapters", "speech-two-adapters") {
        Ok(_) => panic!("multiple text-assembly adapters must be rejected at load"),
        Err(error) => format!("{error:#}"),
    };
    assert!(
        error.contains("exactly one is required for speech synthesis"),
        "error must explain the adapter ambiguity: {error}"
    );
}

#[tokio::test]
#[ignore = "synthetic Whisper-contract smoke test; run explicitly for audio server validation"]
async fn audio_endpoints_route_through_tiny_whisper_pipeline() {
    let model_dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-whisper");
    let router =
        app(AppState::load(&model_dir, Some("tiny-whisper".to_string()))
            .expect("load Whisper fixture"));

    let chat_response = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "tiny-whisper",
                        "messages": [{
                            "role": "user",
                            "content": [{
                                "type": "input_audio",
                                "input_audio": {
                                    "data": tiny_wav_base64(),
                                    "format": "wav"
                                }
                            }]
                        }],
                        "max_tokens": 2,
                        "temperature": 0.0
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(chat_response.status(), StatusCode::OK);
    let chat_body = to_bytes(chat_response.into_body(), usize::MAX)
        .await
        .unwrap();
    let chat_body: Value = serde_json::from_slice(&chat_body).unwrap();
    assert!(chat_body["choices"][0]["message"]["content"].is_string());

    let (boundary, body) = multipart_audio_body("json");
    let transcription_response = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/transcriptions")
                .header(
                    header::CONTENT_TYPE,
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(transcription_response.status(), StatusCode::OK);
    let transcription_body = to_bytes(transcription_response.into_body(), usize::MAX)
        .await
        .unwrap();
    let transcription_body: Value = serde_json::from_slice(&transcription_body).unwrap();
    assert!(transcription_body["text"].is_string());

    let (boundary, body) = multipart_audio_body("text");
    let text_response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/audio/transcriptions")
                .header(
                    header::CONTENT_TYPE,
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(text_response.status(), StatusCode::OK);
    assert_eq!(
        text_response.headers()[header::CONTENT_TYPE],
        "text/plain; charset=utf-8"
    );
    let text_body = to_bytes(text_response.into_body(), usize::MAX)
        .await
        .unwrap();
    assert!(!text_body.is_empty());
}

#[tokio::test]
async fn status_reports_node_status_contract() {
    let response = app(tiny_state())
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body: Value = serde_json::from_slice(&body).unwrap();

    // node_id is present, non-empty, and NOT the model id (model-agnostic).
    let node_id = body["node_id"].as_str().expect("node_id is a string");
    assert!(!node_id.is_empty(), "node_id must not be empty");
    assert_ne!(node_id, "tiny-llm", "node_id must not be the model id");

    assert_eq!(body["healthy"], true);
    // Real metrics serialize with the right JSON types.
    assert!(body["queue_depth"].is_u64());
    assert!(body["active_sessions"].is_u64());
    assert!(body["paused_sessions"].is_u64());
    assert!(body["kv_usage"].is_number());
    assert!(body["kv_pages_used"].is_u64());
    assert!(body["kv_pages_total"].is_u64());
    assert!(body["kv_pages_shared"].is_u64());
    assert!(body["tokens_per_second"].is_number());
    assert!(body["batch_utilization"].is_number());
    assert!(body["sessions"].is_array());
    assert!(body["prefix_hashes"].is_array());
    let workers = body["workers"].as_array().expect("workers is an array");
    assert_eq!(workers.len(), 1);
    assert_eq!(workers[0]["worker_id"], 0);
    assert!(workers[0]["active_turns"].is_u64());
    assert!(workers[0]["live_sessions"].is_u64());
    assert_eq!(workers[0]["health"], "healthy");
}

#[tokio::test]
async fn status_reports_each_configured_ort_worker() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm");
    let state = AppState::load_with_config(
        &model_dir,
        Some("tiny-llm".to_string()),
        ServerConfig {
            ort_session_workers: OrtSessionWorkerCount::new(2).unwrap(),
            ..ServerConfig::default()
        },
    )
    .expect("load fixture with two ORT workers");

    let response = app(state)
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body: Value = serde_json::from_slice(&body).unwrap();
    let workers = body["workers"].as_array().expect("workers is an array");
    assert_eq!(workers.len(), 2);
    assert_eq!(workers[0]["worker_id"], 0);
    assert_eq!(workers[1]["worker_id"], 1);
    assert!(workers.iter().all(|worker| worker["health"] == "healthy"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn resources_and_metrics_track_worker1_host_release_without_worker0_command() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm");
    let state = AppState::load_with_config(
        &model_dir,
        Some("tiny-llm".to_string()),
        ServerConfig {
            ort_session_workers: OrtSessionWorkerCount::new(2).unwrap(),
            ..ServerConfig::default()
        },
    )
    .expect("load fixture with two ORT workers");
    let handle = state.registry.resolve("").unwrap().unwrap();
    let worker0_at_ready = handle
        .engine
        .resource_snapshot
        .lock()
        .unwrap()
        .clone()
        .unwrap()
        .host_ram
        .used;
    let router = app(state.clone());

    let before = router
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/resources")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let before = json_body(before).await;
    let both_workers = before["host_ram"]["used"].as_u64().unwrap();
    assert!(
        both_workers > worker0_at_ready,
        "worker 1 allocation must be visible through /v1/resources"
    );

    handle
        .engine
        .workers
        .sender_for(crate::worker::WorkerId::new(1))
        .unwrap()
        .send(DriverCommand::Panic)
        .await
        .unwrap();
    timeout(Duration::from_secs(5), async {
        loop {
            if handle.engine.worker_statuses()[1].worker.health
                == crate::worker::WorkerHealth::Failed
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("worker 1 reports failure");

    let after = router
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/resources")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let after = json_body(after).await;
    let worker0_only = after["host_ram"]["used"].as_u64().unwrap();
    assert_eq!(worker0_only, worker0_at_ready);

    let metrics = router
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let metrics = String::from_utf8(
        to_bytes(metrics.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    assert!(
        metrics.contains(&format!("onnx_genai_host_ram_used_bytes {worker0_only}\n")),
        "metrics must use the same live shared-ledger snapshot:\n{metrics}"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn multiple_workers_still_refuse_a_second_turn_on_one_session() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm");
    let state = AppState::load_with_config(
        &model_dir,
        Some("tiny-llm".to_string()),
        ServerConfig {
            ort_session_workers: OrtSessionWorkerCount::new(2).unwrap(),
            ..ServerConfig::default()
        },
    )
    .expect("load fixture with two ORT workers");
    let router = app(state.clone());
    let session = "sess-two-workers";
    let (status, first) = chat_turn_for(router.clone(), "tiny-llm", Some(session), "hello").await;
    assert_eq!(status, StatusCode::OK, "{first}");

    let held = state
        .sessions
        .acquire(binding_of(&state, session), session)
        .expect("an idle session is leasable");
    let (status, conflict) = chat_turn_for(router, "tiny-llm", Some(session), "again").await;
    assert_eq!(status, StatusCode::CONFLICT, "{conflict}");
    assert_eq!(conflict["error"]["type"], "conflict_error");
    drop(held);
}

#[test]
fn multiple_ort_workers_fail_closed_for_composite_workflows() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/onnx_genai_workflows/vlm");
    let error = match AppState::load_with_config(
        &model_dir,
        Some("composite".to_string()),
        ServerConfig {
            ort_session_workers: OrtSessionWorkerCount::new(2).unwrap(),
            ..ServerConfig::default()
        },
    ) {
        Ok(_) => panic!("composite execution must not silently fall back to one worker"),
        Err(error) => error,
    };
    let message = format!("{error:#}");
    assert!(message.contains("composite pipeline"), "{message}");
    assert!(message.contains("--ort-session-workers 1"), "{message}");
}

#[cfg(feature = "native-backend")]
#[test]
fn multiple_ort_workers_fail_closed_for_native_decode() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-native-scalar-gqa");
    let error = match AppState::load_with_config(
        &model_dir,
        Some("native".to_string()),
        ServerConfig {
            ort_session_workers: OrtSessionWorkerCount::new(2).unwrap(),
            engine_config: EngineConfig {
                decode_backend: EngineDecodeBackend::Native,
                ..EngineConfig::default()
            },
            ..ServerConfig::default()
        },
    ) {
        Ok(_) => panic!("native decode must not silently fall back to one worker"),
        Err(error) => error,
    };
    let message = format!("{error:#}");
    assert!(
        message.contains("require the ORT decode backend"),
        "{message}"
    );
}

#[tokio::test]
async fn status_node_id_reflects_configured_value() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm");
    let state = AppState::load_with_config(
        &model_dir,
        Some("tiny-llm".to_string()),
        ServerConfig {
            node_id: "gpu-7".to_string(),
            ..ServerConfig::default()
        },
    )
    .expect("load fixture with node id");

    let response = app(state)
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(body["node_id"], "gpu-7");
}

#[tokio::test]
async fn status_active_sessions_reflect_real_state() {
    let state = tiny_state();
    let handle = state.registry.resolve("").unwrap().unwrap();
    // Create a real engine session and register it, mirroring the session route.
    let engine_session = handle
        .engine
        .create_session()
        .await
        .expect("create engine session");
    let client_id = state.sessions.next_client_id().unwrap();
    state
        .sessions
        .insert(client_id, handle.engine.binding(engine_session))
        .expect("register session");

    let response = app(state)
        .oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body: Value = serde_json::from_slice(&body).unwrap();
    assert!(
        body["active_sessions"].as_u64().unwrap() >= 1,
        "active_sessions must reflect the registered session"
    );
    let sessions = body["sessions"].as_array().unwrap();
    assert!(
        !sessions.is_empty(),
        "sessions list must include the registered session"
    );
    assert!(sessions[0]["id"].as_str().unwrap().starts_with("sess-"));
}

#[tokio::test]
async fn debug_endpoints_expose_config_sessions_cache_and_trace_state() {
    let router = app(tiny_state_with_debug());

    let config = router
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/debug/config")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(config.status(), StatusCode::OK);
    let config: Value =
        serde_json::from_slice(&to_bytes(config.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(config["model_id"], "tiny-llm");
    assert_eq!(config["max_queue_depth"], 256);
    // Facts about the package's serialized workflow, not about which executor
    // the runtime chose for a declared step.
    assert_eq!(config["workflow_components"], 2);
    assert_eq!(config["workflow_graph_components"], 1);
    assert_eq!(config["workflow_declares_generation_loop"], true);

    let created = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/sessions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(created.status(), StatusCode::OK);
    let created: Value =
        serde_json::from_slice(&to_bytes(created.into_body(), usize::MAX).await.unwrap()).unwrap();

    let sessions = router
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/debug/sessions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(sessions.status(), StatusCode::OK);
    let sessions: Value =
        serde_json::from_slice(&to_bytes(sessions.into_body(), usize::MAX).await.unwrap()).unwrap();
    // The list must contain a redacted entry for the created session, but must NOT
    // contain the raw bearer credential (full capability ID).
    let raw_id = created["id"].as_str().unwrap();
    let session_list = sessions["sessions"].as_array().unwrap();
    assert!(
        !session_list.iter().any(|v| v.as_str() == Some(raw_id)),
        "raw session ID must not appear in debug/sessions response"
    );
    // Redacted form starts with "sess-" and ends with "…"
    assert!(
        session_list.iter().any(|v| v
            .as_str()
            .is_some_and(|s| s.starts_with("sess-") && s.ends_with('…'))),
        "expected a redacted session entry (sess-<prefix>…) in debug/sessions"
    );

    let cache = router
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/debug/kv")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(cache.status(), StatusCode::OK);
    let cache: Value =
        serde_json::from_slice(&to_bytes(cache.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert!(cache["prefix_cache_hits"].is_u64());
    assert!(cache["pending_queue_depth"].is_u64());
    assert!(cache["available_admission_slots"].is_u64());

    let trace = router
        .oneshot(
            Request::builder()
                .uri("/v1/debug/trace")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(trace.status(), StatusCode::OK);
    let trace: Value =
        serde_json::from_slice(&to_bytes(trace.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(trace["tracing_span"], "http.request");
    assert!(trace["latest_trace_id"].is_string());
    assert_eq!(
        trace["perfetto_export"]["endpoint"],
        "/v1/debug/trace/perfetto"
    );
    assert!(trace["perfetto_export"]["recorded_events"].is_u64());
    assert!(trace["perfetto_export"]["collecting"].is_boolean());
    assert!(
        trace["otlp_export"].as_str().unwrap().contains("deferred"),
        "OTLP export must be reported as deferred"
    );
}

#[tokio::test]
async fn debug_endpoints_report_no_loaded_model_without_panicking() {
    let state = lazy_state(ServerConfig {
        enable_debug_endpoints: true,
        ..ServerConfig::default()
    });
    state
        .registry
        .unload("model-a")
        .expect("unload the only loaded model");
    let router = app(state);

    for path in ["/v1/debug/config", "/v1/debug/kv"] {
        let response = router
            .clone()
            .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let body: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(error_message(&body), "no model loaded");
    }
}

#[tokio::test]
async fn debug_trace_perfetto_returns_well_formed_chrome_trace() {
    let router = app(tiny_state_with_debug());

    let resp = router
        .oneshot(
            Request::builder()
                .uri("/v1/debug/trace/perfetto")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok()),
        Some("application/json")
    );
    assert!(
        resp.headers()
            .get(header::CONTENT_DISPOSITION)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.contains("onnx-genai-trace.json")),
        "Perfetto export must be served as a downloadable attachment"
    );
    let doc: Value =
        serde_json::from_slice(&to_bytes(resp.into_body(), usize::MAX).await.unwrap()).unwrap();
    // Chrome Trace Event Format: an object with a `traceEvents` array. It may be
    // empty (no spans recorded in this process) but must be well-formed so it
    // opens directly in https://ui.perfetto.dev.
    assert!(
        doc["traceEvents"].is_array(),
        "Perfetto document must contain a traceEvents array"
    );
    assert_eq!(doc["displayTimeUnit"], "ms");
}

#[tokio::test]
async fn debug_endpoints_return_404_when_gate_is_off() {
    // Default state has enable_debug_endpoints = false; routes must not be registered.
    let router = app(tiny_state());
    for path in &[
        "/v1/debug/config",
        "/v1/debug/sessions",
        "/v1/debug/kv",
        "/v1/debug/trace",
        "/v1/debug/trace/perfetto",
    ] {
        let resp = router
            .clone()
            .oneshot(Request::builder().uri(*path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::NOT_FOUND,
            "{path} must return 404 when debug endpoints are disabled"
        );
    }
}

#[cfg(feature = "metrics")]
#[tokio::test]
async fn metrics_exposes_prometheus_families_and_request_counter_increments() {
    let router = app(tiny_state());
    let before = router
        .clone()
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let before = to_bytes(before.into_body(), usize::MAX).await.unwrap();
    let before = String::from_utf8(before.to_vec()).unwrap();
    let before_health = prometheus_sample(
        &before,
        "onnx_genai_requests_total{endpoint=\"/health\",status=\"200\"}",
    );

    for _ in 0..2 {
        let response = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    let response = router
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        response.headers()[header::CONTENT_TYPE],
        "text/plain; version=0.0.4; charset=utf-8"
    );
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body = String::from_utf8(body.to_vec()).unwrap();
    assert!(body.contains("# TYPE onnx_genai_requests_total counter"));
    assert!(body.contains("# TYPE onnx_genai_time_to_first_token_seconds histogram"));
    assert!(body.contains("# TYPE onnx_genai_e2e_request_latency_seconds histogram"));
    assert!(body.contains("onnx_genai_sessions_active"));
    assert!(body.contains("onnx_genai_requests_waiting"));
    assert!(body.contains("onnx_genai_batch_size_current"));
    assert!(body.contains("onnx_genai_prefix_cache_hit_rate"));
    assert!(body.contains("onnx_genai_rejections_total"));
    assert!(body.contains("onnx_genai_vram_used_bytes"));
    assert!(body.contains("onnx_genai_vram_limit_bytes"));
    assert!(body.contains("onnx_genai_host_ram_used_bytes"));
    assert!(body.contains("onnx_genai_host_ram_limit_bytes"));
    assert!(body.contains("onnx_genai_kv_budget_bytes"));
    let after_health = prometheus_sample(
        &body,
        "onnx_genai_requests_total{endpoint=\"/health\",status=\"200\"}",
    );
    assert!(after_health >= before_health + 2);
}

/// `/metrics` must say WHICH case a scrape is in when the governor family is
/// absent.
///
/// The handler previously used `if let Ok(snapshot) = ...`, so an unreadable
/// governor dropped the whole `onnx_genai_*` family with no trace. In Prometheus
/// a series that simply stops is indistinguishable from a scrape gap, a
/// restart, or a relabel -- the graph just ends, which is the one shape an
/// operator reads as "nothing to see". An absent resource ceiling is exactly
/// the condition you most need to alert on, so it must be published, not
/// omitted.
///
/// This asserts the INVARIANT BINDING the marker to the family, not merely that
/// a string is present: the marker must be 1 exactly when the gauges are there.
///
/// ⚠️ WHAT THIS TEST CANNOT PROVE, stated because I measured it rather than
/// assumed it: `tiny_state`'s governor ALWAYS resolves, so this exercises only
/// the success branch. Reverting the handler to the old silent-omission form
/// leaves this test GREEN -- verified by mutation, not by reasoning. The
/// absent branch is covered by
/// `an_unreadable_governor_is_published_as_absent_not_omitted`, which drives
/// the encoder directly because no route-level fixture can make the governor
/// fail on demand.
///
/// Note this deliberately does NOT use `prometheus_sample`, whose `unwrap_or(0)`
/// renders an ABSENT metric as the value `0` -- the precise absent/zero
/// conflation this test exists to detect. It would report success on a handler
/// that emitted nothing at all.
#[cfg(feature = "metrics")]
#[tokio::test]
async fn metrics_states_governor_availability_rather_than_dropping_the_family() {
    let response = app(tiny_state())
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body = String::from_utf8(body.to_vec()).unwrap();

    let marker = crate::metrics::RESOURCE_GOVERNOR_AVAILABLE;
    let available = body
        .lines()
        .find_map(|line| line.strip_prefix(marker)?.trim().parse::<u64>().ok())
        .unwrap_or_else(|| {
            panic!(
                "{marker} is absent. Every scrape must state whether the \
                 resource-governor gauges are real; without it an unreadable \
                 governor is indistinguishable from a scrape gap.\n{body}"
            )
        });

    // The family and its marker must agree in BOTH directions, which is what
    // makes this a guard rather than a spelling check.
    let family_present = body.contains("onnx_genai_vram_limit_bytes");
    match available {
        1 => assert!(
            family_present,
            "{marker} is 1 but the governor gauges are absent; the marker \
             promises readings that are not there"
        ),
        0 => assert!(
            !family_present,
            "{marker} is 0 but the governor gauges are present; the marker \
             disclaims readings that ARE there, so operators would discard \
             good data"
        ),
        other => panic!("{marker} must be 0 or 1, got {other}"),
    }
}

/// The unavailable branch, driven directly because no route fixture reaches it.
///
/// This is the half that carries the fix. It is mutation-verified: deleting the
/// `RESOURCE_GOVERNOR_AVAILABLE` gauge from `encode_resource_governor_unavailable`
/// (i.e. returning the empty string, which is what the old handler effectively
/// did) reddens this and nothing else in the suite.
#[cfg(feature = "metrics")]
#[test]
fn an_unreadable_governor_is_published_as_absent_not_omitted() {
    let marker = crate::metrics::RESOURCE_GOVERNOR_AVAILABLE;
    let output = crate::metrics::encode_resource_governor_unavailable();

    assert!(
        output.contains(&format!("{marker} 0\n")),
        "an unreadable governor must publish {marker} 0. Emitting nothing \
         makes a broken governor look identical to a scrape gap, and the \
         resource ceilings are exactly what an operator needs to alert on \
         when they cannot be read.\ngot: {output:?}"
    );
    assert!(
        output.contains(&format!("# TYPE {marker} gauge")),
        "the marker must carry its TYPE line or Prometheus will not scrape \
         it as a gauge:\n{output}"
    );

    // The disclaimer must not be accompanied by the readings it disclaims.
    assert!(
        !output.contains("onnx_genai_vram_limit_bytes"),
        "the unavailable path emitted governor gauges; a 0 marker beside real \
         readings would make operators discard good data:\n{output}"
    );
}

#[tokio::test]
async fn resources_get_and_admin_vram_override_report_governor_state() {
    let router = app(resource_state(true));
    let response = router
        .clone()
        .oneshot(
            Request::builder()
                .uri("/v1/resources")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = json_body(response).await;
    for key in [
        "configured_limits",
        "resolved_limits",
        "derived_kv_budget",
        "vram",
        "host_ram",
        "disk_spill",
    ] {
        assert!(body.get(key).is_some(), "missing resource key {key}");
    }
    for key in ["used", "limit", "headroom"] {
        assert!(body["vram"].get(key).is_some(), "missing VRAM key {key}");
        assert!(
            body["host_ram"].get(key).is_some(),
            "missing host RAM key {key}"
        );
    }

    // #755: the resolved memory strategy (strategy, offload state, managed
    // no-spill VMM state, resolved budget) must be observable over /v1/resources.
    let memory_strategy = body
        .get("memory_strategy")
        .expect("resources must report the resolved memory strategy");
    for key in [
        "strategy",
        "weight_offload_enabled",
        "managed_no_spill",
        "auto_enabled",
    ] {
        assert!(
            memory_strategy.get(key).is_some(),
            "missing memory_strategy key {key}"
        );
    }
    assert!(
        memory_strategy["strategy"].is_string(),
        "memory strategy must be named: {body}"
    );
    // #874: the reported strategy must be the one the platform default actually
    // selects, not merely a well-formed string. The keys above were already
    // asserted before #874 changed the *value* on Windows, so a silent revert of
    // that decision would not have failed any test — exactly the shape of gap
    // that let `device_policy` go unnoticed for months (#678).
    //
    // This model fits, so both platforms must infer `FullResident` with offload
    // off. The platform split only appears for an over-budget model: Windows/WDDM
    // prefers the OS shared-memory path (`Compatibility`), while Linux keeps
    // managed streaming because there is no fallback there and the alternative is
    // "does not run" (#783 — do not inherit a WDDM-specific conclusion).
    assert_eq!(
        memory_strategy["strategy"], "FullResident",
        "a fitting model must stay fully resident on every platform: {body}"
    );
    assert_eq!(
        memory_strategy["weight_offload_enabled"], false,
        "a fitting model must not enable weight offload: {body}"
    );
    assert_eq!(
        memory_strategy["auto_enabled"], false,
        "nothing should be auto-enabled from a vram limit for a fitting model: {body}"
    );
    let impossible = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/resources/vram-limit")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(json!({"limit": "1"}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(impossible.status(), StatusCode::CONFLICT);
    let impossible = json_body(impossible).await;
    assert!(
        impossible["error"]["message"]
            .as_str()
            .unwrap()
            .contains("cannot satisfy lowered resource limit")
    );
    // A 409 is never a fault: nothing broke and the caller can act on it.
    assert_eq!(
        impossible["error"]["type"].as_str(),
        Some("conflict_error"),
        "{impossible}"
    );

    let valid = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/resources/vram-limit")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(json!({"limit": "auto"}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(valid.status(), StatusCode::OK);
    assert!(json_body(valid).await["vram"]["limit"].is_number());
}

#[tokio::test]
async fn explicit_max_batch_above_one_is_refused_on_non_batching_backend() {
    // tiny-llm is a plain past/present model with no shared KV buffer on the CPU
    // EP, so it cannot batch. An explicit `--max-batch 4` must be refused at
    // startup (issue #750), not silently accepted and served one-at-a-time.
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm");
    let error = match AppState::load_with_config(
        &model_dir,
        Some("tiny-llm".to_string()),
        ServerConfig {
            max_batch: Some(4),
            ..ServerConfig::default()
        },
    ) {
        Ok(_) => panic!("explicit --max-batch 4 must be refused on a non-batching backend"),
        Err(error) => format!("{error:#}"),
    };
    assert!(
        error.contains("failed to start engine worker 0")
            && error.contains("engine worker initialization failed")
            && error.contains("--max-batch 4")
            && error.contains("cannot be honored"),
        "the worker's typed initialization failure must reach model loading with context: {error}"
    );
}

#[tokio::test]
async fn default_max_batch_is_silently_clamped_on_non_batching_backend() {
    // With no explicit `--max-batch`, the same non-batching model must still load
    // (the default width is clamped, not refused), and `/v1/resources` must
    // report the honest capability so an operator is not left guessing.
    let router = app(tiny_state());
    let response = router
        .oneshot(
            Request::builder()
                .uri("/v1/resources")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = json_body(response).await;
    assert_eq!(
        body["batching"]["supported"], false,
        "tiny-llm cannot batch on the CPU EP"
    );
    assert_eq!(
        body["batching"]["effective_max_batch"], 1,
        "non-batching backend must clamp the effective width to 1"
    );
    assert!(
        body["batching"]["reason"]
            .as_str()
            .is_some_and(|r| !r.is_empty()),
        "batching report must carry a non-empty reason"
    );
}

#[tokio::test]
async fn admin_vram_override_requires_engine_runtime_override() {
    let response = app(resource_state(false))
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/resources/vram-limit")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(json!({"limit": "auto"}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    assert!(
        json_body(response).await["error"]["message"]
            .as_str()
            .unwrap()
            .contains("allow_runtime_override")
    );
}

#[cfg(feature = "metrics")]
fn prometheus_sample(body: &str, metric: &str) -> u64 {
    body.lines()
        .find_map(|line| {
            line.strip_prefix(metric)
                .and_then(|value| value.trim().parse().ok())
        })
        .unwrap_or(0)
}

#[tokio::test]
#[ignore = "requires gitignored models/tiny-vlm; run scripts/build_tiny_vlm.py first"]
async fn vision_request_routes_through_tiny_vlm_pipeline() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-vlm");
    if !model_dir.is_dir() {
        eprintln!("skipping tiny VLM server test: fixture is absent");
        return;
    }
    let state = AppState::load(&model_dir, Some("tiny-vlm".to_string())).expect("load fixture");
    let response = app(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "tiny-vlm",
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "describe"},
                                {"type": "image_url", "image_url": {"url": tiny_png_data_uri()}}
                            ]
                        }],
                        "max_tokens": 1,
                        "temperature": 0.0
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body: Value = serde_json::from_slice(&body).unwrap();
    assert!(body["choices"][0]["message"]["content"].is_string());
}

#[test]
fn completion_suffix_maps_to_fim_generation() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm");
    let state = AppState::load(&model_dir, Some("tiny-llm".to_string()))
        .expect("load fixture")
        .with_default_fim_config(Some(onnx_genai_engine::FimConfig {
            prefix_token: "<PRE>".to_string(),
            middle_token: "<MID>".to_string(),
            suffix_token: "<SUF>".to_string(),
            format: onnx_genai_engine::FimFormat::PSM,
        }));
    let handle = state.registry.resolve("").unwrap().unwrap();
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

    let prepared = prepare_completion(&request, &handle).unwrap();
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
    let state = AppState::load(&model_dir, Some("tiny-llm".to_string()))
        .expect("load fixture")
        .with_default_fim_config(Some(onnx_genai_engine::FimConfig {
            prefix_token: "<PRE>".to_string(),
            middle_token: "<MID>".to_string(),
            suffix_token: "<SUF>".to_string(),
            format: onnx_genai_engine::FimFormat::PSM,
        }));

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
async fn fim_stream_returns_headers_before_generation_finishes() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm");
    let state = AppState::load(&model_dir, Some("tiny-llm".to_string()))
        .expect("load fixture")
        .with_default_fim_config(Some(onnx_genai_engine::FimConfig {
            prefix_token: "<PRE>".to_string(),
            middle_token: "<MID>".to_string(),
            suffix_token: "<SUF>".to_string(),
            format: onnx_genai_engine::FimFormat::PSM,
        }));
    let request = Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({
                "model": "tiny-llm",
                "prompt": "prefix",
                "suffix": "suffix",
                "max_tokens": 1,
                "temperature": 0.0,
                "stream": true
            })
            .to_string(),
        ))
        .unwrap();

    // The property is that headers follow *admission* rather than a completed
    // generation — the bug this guards against withheld them until the whole
    // FIM response was decoded, which is seconds on a real request.
    //
    // The budget is deliberately far wider than the thing being measured. At
    // 100 ms it was measuring the machine: on a saturated box under
    // `--test-threads=$(nproc)` it fails while the request itself takes
    // milliseconds. A whole probe run — load plus six 64-token generations —
    // measures 115 ms, so two seconds separates "after admission" from "after
    // generation" with room the scheduler cannot eat.
    let response = timeout(Duration::from_secs(2), app(state).oneshot(request))
        .await
        .expect("SSE headers must follow admission, not completed FIM generation")
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = timeout(
        Duration::from_secs(10),
        to_bytes(response.into_body(), usize::MAX),
    )
    .await
    .expect("FIM stream did not terminate")
    .unwrap();
    assert!(
        std::str::from_utf8(&body)
            .unwrap()
            .ends_with("data: [DONE]\n\n")
    );
}

#[tokio::test]
async fn accepted_zero_visible_output_stream_returns_headers_and_terminates() {
    let response = app(tiny_state())
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "tiny-llm",
                        "prompt": "hello",
                        "max_tokens": 1,
                        "temperature": 0.0,
                        "stop": "tok22",
                        "logprobs": 1,
                        "stream": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let events = sse_json_events(&body);
    assert!(
        events.iter().all(|event| event["choices"][0]["text"]
            .as_str()
            .unwrap_or_default()
            .is_empty()),
        "the stop sequence must suppress all visible token text: {}",
        std::str::from_utf8(&body).unwrap()
    );
    assert_eq!(
        events.last().unwrap()["choices"][0]["finish_reason"],
        "stop"
    );
    assert!(
        std::str::from_utf8(&body)
            .unwrap()
            .ends_with("data: [DONE]\n\n")
    );
}

#[tokio::test]
async fn loaded_server_model_reports_nonzero_device_ledger_usage() {
    let state = tiny_state();
    let handle = state.registry.resolve("").unwrap().unwrap();
    let snapshot = handle.engine.resource_snapshot().await.unwrap();

    assert!(
        snapshot.breakdown.model_weights_bytes > 0,
        "fixture must charge a quantity, not just emit a load event"
    );
    assert!(
        snapshot.vram.used > 0,
        "server metrics must read the engine ledger after load; a zero here recreates #706"
    );
}

#[tokio::test]
async fn queue_depth_admission_limit_returns_429_with_retry_after() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm");
    let state = AppState::load_with_config(
        &model_dir,
        Some("tiny-llm".to_string()),
        ServerConfig {
            max_output_tokens: 16,
            max_sessions: 8,
            max_queue_depth: 1,
            enable_debug_endpoints: false,
            ..ServerConfig::default()
        },
    )
    .unwrap();
    let handle = state.registry.resolve("").unwrap().unwrap();
    let _occupied = handle
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
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm");
    let (driver, ()) = EngineDriver::start(
        move || Ok((Engine::from_dir(&model_dir, EngineConfig::default())?, ())),
        2,
        2,
    )
    .unwrap();
    let slow_request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "tiny-llm",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 8
    }))
    .unwrap();
    let fast_request: ChatCompletionRequest = serde_json::from_value(json!({
        "model": "tiny-llm",
        "messages": [{"role": "user", "content": "world"}],
        "max_tokens": 2
    }))
    .unwrap();
    let (slow_tx, _slow_rx) = mpsc::channel(1);
    let (slow_admission, _slow_admission_rx) = tokio::sync::oneshot::channel();
    let slow_permit = driver
        .generation_capacity
        .clone()
        .try_acquire_owned()
        .unwrap();
    driver
        .workers
        .primary_sender()
        .expect("driver worker is running")
        .send(DriverCommand::Generate {
            input: None,
            session_id: None,
            lease: None,
            request: Box::new(build_generate_request(&slow_request)),
            admission: slow_admission,
            events: slow_tx,
            permit: crate::driver::WorkerPermit::untracked(slow_permit),
        })
        .await
        .unwrap();
    let fast_rx = driver
        .generate(None, build_generate_request(&fast_request), None)
        .await
        .unwrap();

    let fast_result = timeout(
        Duration::from_secs(5),
        collect_generation_result(fast_rx.events),
    )
    .await
    .expect("fast request timed out behind stalled consumer")
    .expect("fast request failed");
    assert_eq!(fast_result.token_ids.len(), 2);
}

#[cfg(feature = "native-backend")]
#[tokio::test]
async fn native_driver_sessions_generate_through_server_path() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-native-sub4-engine");
    let (driver, ()) = EngineDriver::start(
        move || {
            Ok((
                Engine::from_dir(
                    &model_dir,
                    EngineConfig {
                        decode_backend: EngineDecodeBackend::Native,
                        ..EngineConfig::default()
                    },
                )?,
                (),
            ))
        },
        2,
        4,
    )
    .unwrap();
    let session_id = driver.create_session().await.unwrap();
    let leases = crate::lease::SessionLeases::with_shards(1);

    let mut request =
        onnx_genai::GenerateRequest::new(onnx_genai::GeneratePrompt::TokenIds(vec![0]));
    request.options.max_new_tokens = 2;
    request.options.temperature = 0.0;
    request.options.stop_on_eos = false;
    let generation = driver
        .generate(
            Some(
                leases
                    .acquire(driver.binding(session_id), "native-session")
                    .expect("a session with no turn in flight is leasable"),
            ),
            request,
            None,
        )
        .await
        .unwrap();
    let result = timeout(
        Duration::from_secs(5),
        collect_generation_result(generation.events),
    )
    .await
    .expect("native session generation timed out")
    .expect("native session generation failed");

    assert_eq!(result.token_ids, vec![1, 1]);
    assert_eq!(driver.session_token_count(session_id).await.unwrap(), 3);
    // The route that holds the turn's lease is dropped by the driver *after*
    // it sends `Finished`, so a returned result does not by itself imply a
    // released lease — it only implies the release is imminent. Poll rather
    // than assume: this still fails if the release never happens, without
    // racing the driver thread when it does.
    let mut lease = None;
    for _ in 0..200 {
        match leases.acquire(driver.binding(session_id), "native-session-close") {
            Ok(guard) => {
                lease = Some(guard);
                break;
            }
            Err(_) => tokio::time::sleep(Duration::from_millis(25)).await,
        }
    }
    let lease = lease.expect("the finished turn never released its lease");
    driver.close_session(lease).await.unwrap();
}

// ── KV cache dtype CLI/env surface tests ─────────────────────────────────────

/// Parse each accepted KV-cache dtype string using the same function that the
/// server binary uses (`parse_kv_cache_dtype`), and verify that the result is
/// threaded through `ServerConfig.engine_config.kv_cache_dtype`.
#[test]
fn kv_cache_dtype_parses_all_accepted_values() {
    use crate::state::parse_kv_cache_dtype;
    use onnx_genai_engine::KvDType;

    for (input, expected) in [
        ("f32", KvDType::F32),
        ("fp32", KvDType::F32),
        ("float32", KvDType::F32),
        ("int8", KvDType::Int8),
        ("fp8_e4m3fn", KvDType::Fp8E4M3Fn),
        ("float8_e4m3fn", KvDType::Fp8E4M3Fn),
        ("fp8_e5m2", KvDType::Fp8E5M2),
        ("float8_e5m2", KvDType::Fp8E5M2),
    ] {
        let parsed = parse_kv_cache_dtype(input)
            .unwrap_or_else(|_| panic!("expected '{input}' to parse successfully"));
        assert_eq!(parsed, expected, "'{input}' should parse to {expected:?}");
    }
}

#[test]
fn kv_cache_dtype_rejects_garbage_values() {
    use crate::state::parse_kv_cache_dtype;

    for bad in ["fp4", "nope", "", "int4", "float64"] {
        assert!(
            parse_kv_cache_dtype(bad).is_err(),
            "'{bad}' should be rejected as an invalid KV dtype"
        );
    }
}

#[cfg(feature = "native-cuda")]
#[test]
fn native_device_parser_accepts_cuda_index() {
    use crate::state::parse_native_device;
    use onnx_genai_engine::NativeDecodeDevice;

    assert_eq!(parse_native_device("cpu").unwrap(), NativeDecodeDevice::Cpu);
    assert_eq!(
        parse_native_device("cuda").unwrap(),
        NativeDecodeDevice::Cuda { index: None }
    );
    assert_eq!(
        parse_native_device("cuda:3").unwrap(),
        NativeDecodeDevice::Cuda { index: Some(3) }
    );
    assert!(parse_native_device("webgpu").is_err());
}

#[cfg(all(feature = "native-backend", not(feature = "native-cuda")))]
#[test]
fn native_device_parser_rejects_cuda_without_cuda_feature() {
    use crate::state::parse_native_device;

    assert!(parse_native_device("cpu").is_ok());
    // Assert the facts, not the sentence: the message must name the feature that
    // actually gates this path and give a usable rebuild command. The previous
    // assertion pinned `'cuda' feature`, a name that no longer exists, so it kept
    // passing after the rename while the message became unactionable.
    let message = parse_native_device("cuda:0").unwrap_err();
    assert!(
        message.contains("native-cuda"),
        "message must name the real gating feature: {message}"
    );
    assert!(
        message.contains("--features native-cuda"),
        "message must give a usable rebuild command: {message}"
    );
}

#[test]
fn server_config_engine_config_kv_cache_dtype_defaults_to_f32() {
    use onnx_genai_engine::KvDType;
    let config = ServerConfig::default();
    assert_eq!(
        config.engine_config.kv_cache_dtype,
        KvDType::F32,
        "default ServerConfig must use F32 KV storage"
    );
}

#[test]
fn server_config_engine_config_kv_cache_dtype_can_be_set() {
    use onnx_genai_engine::KvDType;
    let config = ServerConfig {
        engine_config: EngineConfig {
            kv_cache_dtype: KvDType::Fp8E4M3Fn,
            ..EngineConfig::default()
        },
        ..ServerConfig::default()
    };
    assert_eq!(config.engine_config.kv_cache_dtype, KvDType::Fp8E4M3Fn);
}

// ── M2: multi-model routing tests ────────────────────────────────────────────

/// Load the tiny-llm fixture twice under two different ids to exercise
/// multi-model routing without requiring a second distinct fixture.
fn two_model_state() -> AppState {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm");
    let specs = vec![
        ModelSpec {
            id: "model-a".to_string(),
            path: path.clone(),
            eager: true,
            warmup: false,
        },
        ModelSpec {
            id: "model-b".to_string(),
            path: path.clone(),
            eager: true,
            warmup: false,
        },
    ];
    AppState::load_from_specs(specs, ServerConfig::default()).expect("load two tiny-llm fixtures")
}

#[tokio::test]
async fn named_model_routes_to_the_correct_handle() {
    let router = app(two_model_state());
    // Request for model-a returns 200 and echoes model-a in the response.
    let resp = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "model-a",
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 1,
                        "temperature": 0.0
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(resp.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["model"], "model-a");
}

#[tokio::test]
async fn unknown_named_model_returns_404() {
    let router = app(two_model_state());
    let resp = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "does-not-exist",
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 1
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let body: Value =
        serde_json::from_slice(&to_bytes(resp.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("does-not-exist"),
        "error should name the unknown model: {body}"
    );
}

#[tokio::test]
async fn empty_model_field_falls_back_to_default() {
    let router = app(two_model_state());
    // Sending an empty string for model should resolve to the first loaded model.
    let resp = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "",
                        "messages": [{"role": "user", "content": "hello"}],
                        "max_tokens": 1,
                        "temperature": 0.0
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    // Should succeed (200) – empty model falls back to the default, not 404.
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn models_endpoint_lists_all_loaded_models() {
    let router = app(two_model_state());
    let resp = router
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body: Value =
        serde_json::from_slice(&to_bytes(resp.into_body(), usize::MAX).await.unwrap()).unwrap();
    let ids: Vec<&str> = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|obj| obj["id"].as_str().unwrap())
        .collect();
    assert!(
        ids.contains(&"model-a"),
        "model-a not in /v1/models: {body}"
    );
    assert!(
        ids.contains(&"model-b"),
        "model-b not in /v1/models: {body}"
    );
    assert_eq!(ids.len(), 2);
}

#[tokio::test]
async fn unknown_model_returns_404_on_completions_endpoint() {
    let router = app(tiny_state());
    let resp = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "no-such-model",
                        "prompt": "hello",
                        "max_tokens": 1
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn single_model_startup_still_works_via_load_with_config() {
    // Regression guard: the existing load_with_config / single-model path must
    // behave identically to M1.
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm");
    let state = AppState::load_with_config(
        &model_dir,
        Some("tiny-llm".to_string()),
        ServerConfig::default(),
    )
    .expect("single-model load must still work");
    // Registry has exactly one entry with the expected id.
    assert_eq!(state.registry.ids().unwrap().len(), 1);
    assert_eq!(
        state.registry.default_id().unwrap().as_deref(),
        Some("tiny-llm")
    );

    let resp = app(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "tiny-llm",
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 1,
                        "temperature": 0.0
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

// ── M3: runtime load/unload + LRU eviction + lazy load tests ─────────────────

/// Build a two-model state where `model-a` is eager and `model-b` is lazy.
/// Both are backed by the tiny-llm fixture. `config` lets callers toggle admin
/// endpoints and the loaded-model cap.
fn lazy_state(config: ServerConfig) -> AppState {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm");
    let specs = vec![
        ModelSpec {
            id: "model-a".to_string(),
            path: path.clone(),
            eager: true,
            warmup: false,
        },
        ModelSpec {
            id: "model-b".to_string(),
            path: path.clone(),
            eager: false,
            warmup: false,
        },
    ];
    AppState::load_from_specs(specs, config).expect("load lazy two-model state")
}

#[cfg(feature = "metrics")]
#[tokio::test]
async fn metrics_keep_shared_vram_after_default_model_unload() {
    let state = lazy_state(ServerConfig::default());
    state.registry.load("model-b").await.unwrap();
    state.registry.unload("model-a").unwrap();

    let response = app(state)
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body = String::from_utf8(body.to_vec()).unwrap();
    let used = prometheus_sample(&body, "onnx_genai_vram_used_bytes");
    let limit = prometheus_sample(&body, "onnx_genai_vram_limit_bytes");
    let headroom = prometheus_sample(&body, "onnx_genai_vram_headroom_bytes");
    assert!(used > 0);
    assert!(limit > used);
    assert_eq!(headroom, limit - used);
}

#[tokio::test]
async fn admin_vram_update_with_no_loaded_models_sets_future_policy() {
    let state = lazy_state(ServerConfig {
        enable_admin_endpoints: true,
        engine_config: EngineConfig {
            allow_runtime_override: true,
            ..EngineConfig::default()
        },
        ..ServerConfig::default()
    });
    state.registry.unload("model-a").unwrap();

    let response = app(state.clone())
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/resources/vram-limit")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(json!({"limit": "7GiB"}).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NO_CONTENT);

    let loaded = state.registry.load("model-b").await.unwrap();
    assert_eq!(
        loaded.engine.resource_snapshot().await.unwrap().vram.limit,
        7 << 30
    );
}

fn chat_request(model: &str) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            json!({
                "model": model,
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 1,
                "temperature": 0.0
            })
            .to_string(),
        ))
        .unwrap()
}

async fn json_body(resp: axum::response::Response) -> Value {
    serde_json::from_slice(&to_bytes(resp.into_body(), usize::MAX).await.unwrap()).unwrap()
}

#[tokio::test]
async fn lazy_model_is_loaded_on_first_request() {
    let state = lazy_state(ServerConfig::default());
    // Only the eager model is loaded at startup.
    assert_eq!(state.registry.ids().unwrap(), vec!["model-a"]);
    assert!(state.registry.contains_available("model-b").unwrap());

    // Routing to the lazy model triggers a load and succeeds.
    let resp = app(state.clone())
        .oneshot(chat_request("model-b"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json_body(resp).await;
    assert_eq!(body["model"], "model-b");

    // The shared registry now has both models loaded.
    let mut ids = state.registry.ids().unwrap();
    ids.sort();
    assert_eq!(ids, vec!["model-a", "model-b"]);
}

#[tokio::test]
async fn admin_load_then_route_to_lazy_model() {
    let state = lazy_state(ServerConfig {
        enable_admin_endpoints: true,
        ..ServerConfig::default()
    });

    // Admin-load the lazy model.
    let resp = app(state.clone())
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/models/model-b/load")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert!(state.registry.resolve("model-b").unwrap().is_some());

    // Subsequent routing works without re-loading.
    let resp = app(state.clone())
        .oneshot(chat_request("model-b"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn admin_warmup_loaded_model_returns_success_and_is_idempotent() {
    let state = lazy_state(ServerConfig {
        enable_admin_endpoints: true,
        ..ServerConfig::default()
    });

    let first_response = app(state.clone())
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/models/model-a/warm")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(first_response.status(), StatusCode::OK);
    let first_body = json_body(first_response).await;
    assert_eq!(first_body["id"], "model-a");
    assert_eq!(first_body["warmed"], true);
    assert!(first_body["duration_ms"].is_number());

    let second_response = app(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/models/model-a/warm")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(second_response.status(), StatusCode::OK);
    let second_body = json_body(second_response).await;
    assert_eq!(second_body["duration_ms"], 0);
}

#[tokio::test]
async fn admin_warmup_unknown_model_returns_404() {
    let state = lazy_state(ServerConfig {
        enable_admin_endpoints: true,
        ..ServerConfig::default()
    });
    let response = app(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/models/no-such-model/warm")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn admin_warmup_loaded_model_generation_failure_returns_500() {
    let state = lazy_state(ServerConfig {
        enable_admin_endpoints: true,
        max_queue_depth: 1,
        ..ServerConfig::default()
    });
    let handle = state.registry.resolve("model-a").unwrap().unwrap();
    let _occupied = handle
        .engine
        .generation_capacity
        .clone()
        .try_acquire_owned()
        .unwrap();

    let response = app(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/models/model-a/warm")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
}

#[tokio::test]
async fn configured_warmup_runs_when_an_eager_model_loads() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm");
    let state = AppState::load_from_specs(
        vec![ModelSpec {
            id: "model-a".to_string(),
            path,
            eager: true,
            warmup: true,
        }],
        ServerConfig::default(),
    )
    .expect("load and warm tiny model");
    assert!(state.registry.is_warmed_for_test("model-a"));
}

#[tokio::test]
async fn admin_unload_then_lazy_reload() {
    let state = lazy_state(ServerConfig {
        enable_admin_endpoints: true,
        ..ServerConfig::default()
    });
    // Unload the eager, default model.
    let resp = app(state.clone())
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/v1/admin/models/model-a")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NO_CONTENT);
    assert!(state.registry.resolve("model-a").unwrap().is_none());
    // The spec is retained for lazy reload.
    assert!(state.registry.contains_available("model-a").unwrap());

    // A subsequent request for the default (empty model) lazily reloads it.
    let resp = app(state.clone()).oneshot(chat_request("")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert!(state.registry.resolve("model-a").unwrap().is_some());
}

#[tokio::test]
async fn admin_unload_unknown_model_returns_404() {
    let state = lazy_state(ServerConfig {
        enable_admin_endpoints: true,
        ..ServerConfig::default()
    });
    // model-b is available but not loaded → unload is a 404.
    let resp = app(state)
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/v1/admin/models/model-b")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn admin_unload_distinguishes_poisoned_registry_from_absent_model() {
    let absent_model_state = lazy_state(ServerConfig {
        enable_admin_endpoints: true,
        ..ServerConfig::default()
    });
    let absent_model_response = app(absent_model_state)
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/v1/admin/models/model-b")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(absent_model_response.status(), StatusCode::NOT_FOUND);
    let body: Value = serde_json::from_slice(
        &to_bytes(absent_model_response.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_well_formed_error_without_internal_path(&body);
    assert_eq!(error_message(&body), "model 'model-b' is not loaded");

    let poisoned_registry_state = lazy_state(ServerConfig {
        enable_admin_endpoints: true,
        ..ServerConfig::default()
    });
    poisoned_registry_state.registry.poison_for_test();
    let poisoned_registry_response = app(poisoned_registry_state)
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/v1/admin/models/model-a")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("poisoned registry must produce an HTTP response");
    assert_eq!(
        poisoned_registry_response.status(),
        StatusCode::INTERNAL_SERVER_ERROR
    );
    let body: Value = serde_json::from_slice(
        &to_bytes(poisoned_registry_response.into_body(), usize::MAX)
            .await
            .unwrap(),
    )
    .unwrap();
    assert_well_formed_error_without_internal_path(&body);
    assert_eq!(error_message(&body), "model registry failed");
}

#[tokio::test]
async fn admin_load_unknown_model_returns_404() {
    let state = lazy_state(ServerConfig {
        enable_admin_endpoints: true,
        ..ServerConfig::default()
    });
    let resp = app(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/models/no-such-model/load")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn max_loaded_models_evicts_least_recently_used() {
    let state = lazy_state(ServerConfig {
        enable_admin_endpoints: true,
        max_loaded_models: Some(1),
        ..ServerConfig::default()
    });
    // model-a is loaded at startup (cap = 1).
    assert_eq!(state.registry.ids().unwrap(), vec!["model-a"]);

    // Loading model-b must evict model-a to respect the cap.
    let resp = app(state.clone())
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/models/model-b/load")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        state.registry.ids().unwrap(),
        vec!["model-b"],
        "model-a should be evicted"
    );
    assert!(state.registry.contains_available("model-a").unwrap());
}

#[tokio::test]
async fn admin_list_reports_loaded_and_available() {
    let state = lazy_state(ServerConfig {
        enable_admin_endpoints: true,
        ..ServerConfig::default()
    });
    let resp = app(state)
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/admin/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json_body(resp).await;
    let entries = body["data"].as_array().unwrap();
    assert_eq!(entries.len(), 2);
    let a = entries.iter().find(|e| e["id"] == "model-a").unwrap();
    let b = entries.iter().find(|e| e["id"] == "model-b").unwrap();
    assert_eq!(a["loaded"], true);
    assert_eq!(a["is_default"], true);
    assert!(a["last_request_at"].is_number());
    assert_eq!(b["loaded"], false);
    assert_eq!(b["is_default"], false);
    assert!(b["last_request_at"].is_null());
}

#[tokio::test]
async fn admin_endpoints_return_404_when_gate_is_off() {
    // Admin endpoints disabled (default): routes are not mounted.
    let state = lazy_state(ServerConfig::default());
    for (method, uri) in [
        ("GET", "/v1/admin/models"),
        ("POST", "/v1/admin/models/model-b/load"),
        ("POST", "/v1/admin/models/model-a/warm"),
        ("DELETE", "/v1/admin/models/model-a"),
        ("POST", "/v1/admin/resources/vram-limit"),
    ] {
        let resp = app(state.clone())
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(uri)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            resp.status(),
            StatusCode::NOT_FOUND,
            "{method} {uri} must 404 when admin endpoints are disabled"
        );
    }
}

#[tokio::test]
async fn empty_model_field_falls_back_to_default_on_embeddings() {
    // An empty `model` field on /v1/embeddings must resolve to the registry's
    // default model and return 200, not 400.
    let model_dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-mtp-full");
    let state = AppState::load(&model_dir, Some("tiny-mtp-full".to_string()))
        .expect("load tiny-mtp-full fixture");
    let resp = app(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "",
                        "input": "hello"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn unknown_model_returns_404_on_embeddings_endpoint() {
    let model_dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-mtp-full");
    let state = AppState::load(&model_dir, Some("tiny-mtp-full".to_string()))
        .expect("load tiny-mtp-full fixture");
    let resp = app(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/embeddings")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "no-such-model",
                        "input": "hello"
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn concurrent_lazy_loads_of_same_id_load_once() {
    let state = lazy_state(ServerConfig::default());
    // Fire many concurrent requests for the same lazy model.
    let mut handles = Vec::new();
    for _ in 0..8 {
        let state = state.clone();
        handles.push(tokio::spawn(async move {
            app(state)
                .oneshot(chat_request("model-b"))
                .await
                .unwrap()
                .status()
        }));
    }
    for handle in handles {
        assert_eq!(handle.await.unwrap(), StatusCode::OK);
    }
    // Exactly one loaded instance of model-b exists in the registry.
    assert!(state.registry.resolve("model-b").unwrap().is_some());
    let mut ids = state.registry.ids().unwrap();
    ids.sort();
    assert_eq!(ids, vec!["model-a", "model-b"]);
}

async fn post_json(state: AppState, uri: &str, body: Value) -> (StatusCode, Value) {
    let response = app(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, json)
}

fn error_message(body: &Value) -> String {
    body["error"]["message"]
        .as_str()
        .unwrap_or_default()
        .to_string()
}

fn assert_well_formed_error_without_internal_path(body: &Value) {
    let message = body["error"]["message"]
        .as_str()
        .expect("error response must contain a message");
    assert!(!message.is_empty());
    assert_eq!(body["error"]["type"], "server_error");
    assert!(
        !message.contains(env!("CARGO_MANIFEST_DIR")),
        "error must not expose the crate path: {body}"
    );
    assert!(
        !message.contains("tests/fixtures"),
        "error must not expose an internal fixture path: {body}"
    );
}

fn assert_actionable(message: &str) {
    assert!(message.contains("What:"), "message: {message}");
    assert!(message.contains("Why:"), "message: {message}");
    assert!(message.contains("How:"), "message: {message}");
}

#[tokio::test]
async fn unknown_content_part_types_are_rejected_by_name() {
    let (status, body) = post_json(
        tiny_state(),
        "/v1/chat/completions",
        json!({
            "model": "tiny-llm",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "input_video", "input_video": {"data": "..."}}
                ]
            }],
            "max_tokens": 1
        }),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    let message = error_message(&body);
    assert_actionable(&message);
    assert!(
        message.contains("input_video") && message.contains("content part 1"),
        "the offending part and its index must be named: {message}"
    );
    assert!(
        message.contains("image_url"),
        "the accepted types must be listed: {message}"
    );
}

#[tokio::test]
async fn a_malformed_image_url_part_names_the_part_and_the_expected_shape() {
    // A common client mistake: image_url sent as a bare string.
    let (status, body) = post_json(
        tiny_state(),
        "/v1/chat/completions",
        json!({
            "model": "tiny-llm",
            "messages": [{
                "role": "user",
                "content": [{"type": "image_url", "image_url": "https://example.invalid/cat.png"}]
            }],
            "max_tokens": 1
        }),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    let message = error_message(&body);
    assert_actionable(&message);
    assert!(
        message.contains("content part 0") && message.contains("image_url"),
        "message: {message}"
    );
}

#[tokio::test]
async fn content_parts_must_be_objects_with_a_type() {
    for (content, expected) in [
        (json!(["just a string"]), "content part 0"),
        (json!([{"text": "no type here"}]), "`type`"),
        (json!(42), "content"),
    ] {
        let (status, body) = post_json(
            tiny_state(),
            "/v1/chat/completions",
            json!({
                "model": "tiny-llm",
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 1
            }),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "body: {body}");
        let message = error_message(&body);
        assert_actionable(&message);
        assert!(message.contains(expected), "message: {message}");
    }
}

#[tokio::test]
async fn poisoned_model_registry_returns_http_500() {
    let state = tiny_state();
    state.registry.poison_for_test();

    let response = app(state)
        .oneshot(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .expect("poisoned registry must produce an HTTP response");
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_well_formed_error_without_internal_path(&body);
    assert_eq!(error_message(&body), "model registry failed");
}

#[tokio::test]
async fn a_json_syntax_error_is_reported_as_an_actionable_400() {
    let response = app(tiny_state())
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from("{\"model\": "))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_actionable(&error_message(&body));
}

#[test]
fn serde_position_suffixes_are_stripped_from_rejection_messages() {
    use crate::routes::strip_serde_position;

    assert_eq!(
        strip_serde_position("What: bad part. How: fix it. at line 1 column 74"),
        "What: bad part. How: fix it."
    );
    // A message that merely mentions a line is left alone.
    assert_eq!(
        strip_serde_position("What: something at line boundaries"),
        "What: something at line boundaries"
    );
}

#[test]
fn content_parts_render_images_where_they_were_written() {
    use crate::types::ChatMessageContent;

    let content: ChatMessageContent = serde_json::from_value(json!([
        {"type": "text", "text": "compare "},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
        {"type": "text", "text": " with "},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,BB"}},
        {"type": "text", "text": "."}
    ]))
    .expect("content parts parse");

    assert_eq!(
        content.render(Some("<image>")),
        "compare <image> with <image>.",
        "each image must be written where it appeared"
    );
    // Without an image contract the parts are dropped, as before; such a
    // request is rejected moments later by the admission check.
    assert_eq!(content.render(None), "compare  with .");
}

async fn get_body(uri: &str) -> Value {
    let response = app(tiny_state_with_debug())
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK, "GET {uri}");
    serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap()
}

#[tokio::test]
async fn debug_profile_reports_stage_totals() {
    // Asking a running server where its time went should not require a trace
    // viewer, so the aggregate is served as data.
    let body = get_body("/v1/debug/profile").await;

    assert!(body["stages"].is_array(), "body: {body}");
    assert!(
        body["collecting"].is_boolean(),
        "whether anything is being collected must be stated, not inferred from \
         an empty list: {body}"
    );
    assert!(
        body["note"]
            .as_str()
            .is_some_and(|note| note.contains("ONNX_GENAI_PROFILE")),
        "an empty profile must say how to fill it: {body}"
    );
    let plans = body["memory_strategy_plans"]
        .as_array()
        .expect("profile must expose model memory strategy plans");
    assert_eq!(plans.len(), 1, "body: {body}");
    assert_eq!(plans[0]["model_id"], "tiny-llm");
    assert!(
        plans[0]["plan"]["strategy"].is_string(),
        "the effective strategy must be present: {body}"
    );
    assert!(
        plans[0]["plan"]["decisions"]
            .as_array()
            .is_some_and(|decisions| decisions.iter().all(|decision| {
                decision["source"].is_string()
                    && decision["reason"].is_string()
                    && decision["evidence"].is_string()
            })),
        "every strategy decision must include provenance: {body}"
    );
}

#[tokio::test]
async fn an_uncollected_profile_is_empty_rather_than_invented() {
    // Tests run without ONNX_GENAI_PROFILE, so this is the real state.
    let body = get_body("/v1/debug/profile").await;
    assert_eq!(body["collecting"], false, "nothing enabled it: {body}");
    assert_eq!(
        body["stages"].as_array().map(Vec::len),
        Some(0),
        "an uncollected profile must be empty, not fabricated: {body}"
    );
}

#[tokio::test]
async fn the_trace_endpoint_points_at_the_aggregate_profile() {
    // A caller who found the trace endpoint should be able to discover the
    // cheaper question from it.
    let body = get_body("/v1/debug/trace").await;
    assert_eq!(
        body["aggregate_profile"], "/v1/debug/profile",
        "body: {body}"
    );
}

/// `/v1/resources` reports on a running batch, so parking its snapshot until the
/// batch drains makes it readable only when there is nothing to report. This is
/// asserted against `handle_or_defer_during_batch` rather than through the HTTP
/// stack on purpose: the test fixtures generate in under a millisecond, so a
/// racing integration test cannot land a request inside the batch window and
/// passes whether or not the bug is present.
#[tokio::test]
async fn resource_snapshots_are_answered_during_a_batch_not_deferred() {
    let model_dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm-scatter");
    let engine = Engine::from_dir(&model_dir, EngineConfig::default()).unwrap();

    // The mirror is what the driver refreshes between steps; a batch in flight
    // holds `&mut Engine`, so the answer has to come from here.
    let mirror = std::sync::Mutex::new(Some(engine.resource_snapshot()));

    let (reply, rx) = tokio::sync::oneshot::channel();
    let deferred = crate::driver::handle_or_defer_during_batch(
        &mirror,
        crate::driver::DriverCommand::ResourceSnapshot(reply),
    );

    assert!(
        deferred.is_none(),
        "a resource snapshot was pushed to the deferred queue; /v1/resources will \
         appear to hang until every in-flight generation completes"
    );
    rx.await
        .expect("the snapshot reply channel was dropped without an answer")
        .expect("the snapshot itself failed");
}

/// Before the driver has ever refreshed the mirror there is nothing truthful to
/// report, so the snapshot must be deferred rather than answered with a
/// fabricated or default value.
#[tokio::test]
async fn an_empty_snapshot_mirror_defers_rather_than_fabricating() {
    let mirror: std::sync::Mutex<Option<onnx_genai_engine::GovernorSnapshot>> =
        std::sync::Mutex::new(None);

    let (reply, _rx) = tokio::sync::oneshot::channel();
    let deferred = crate::driver::handle_or_defer_during_batch(
        &mirror,
        crate::driver::DriverCommand::ResourceSnapshot(reply),
    );

    assert!(
        matches!(
            deferred,
            Some(crate::driver::DriverCommand::ResourceSnapshot(_))
        ),
        "an unpopulated mirror must defer, not invent a snapshot"
    );
}

/// The complement: commands that *reconfigure* the engine must still be parked until the/// batch drains. This pins the helper's contract to "answer read-only observability",
/// not "answer anything".
#[tokio::test]
async fn mutating_commands_are_still_deferred() {
    let model_dir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm-scatter");
    let engine = Engine::from_dir(&model_dir, EngineConfig::default()).unwrap();

    let mirror = std::sync::Mutex::new(Some(engine.resource_snapshot()));

    let (reply, _rx) = tokio::sync::oneshot::channel();
    let deferred = crate::driver::handle_or_defer_during_batch(
        &mirror,
        crate::driver::DriverCommand::SetVramLimit {
            limit: onnx_genai_engine::ResourceLimit::Auto,
            reply,
        },
    );

    assert!(
        matches!(
            deferred,
            Some(crate::driver::DriverCommand::SetVramLimit { .. })
        ),
        "a command requiring &mut Engine must stay deferred"
    );
}

/// Sessions work for a package whose components the interpreter invokes.
///
/// `/v1/sessions` used to reject these outright, on the grounds that they own
/// no decode-core KV sequence. That conflated two different things: a session
/// is the conversation a client is having, and where the runtime keeps it —
/// a paged KV sequence, or the `scope: session` cells the workflow declares —
/// is not something a client can act on. A client that opened a session against
/// one package and got a 400 against another would have to know which kind it
/// was talking to, which is exactly the caller-side split this removes.
#[tokio::test]
async fn sessions_open_and_close_for_an_interpreted_package() {
    let state = speech_state_from("speech_wav", "workflow-sessions");
    let router = app(state);

    let created = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/sessions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        created.status(),
        StatusCode::OK,
        "a package the interpreter drives still has conversations"
    );
    let created: Value =
        serde_json::from_slice(&to_bytes(created.into_body(), usize::MAX).await.unwrap()).unwrap();
    let id = created["id"].as_str().expect("session id").to_string();

    let deleted = router
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/v1/sessions/{id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(deleted.status(), StatusCode::NO_CONTENT);
}

/// An interpreted package's conversation survives the HTTP boundary.
///
/// A session is what a client holds across requests, so the property has to be
/// checked where a client actually is: three `POST /v1/completions` carrying the
/// same `X-Session-Id`, compared against one request carrying the whole
/// conversation. The engine-level test pins the same property; this one pins
/// that nothing between the socket and the interpreter drops the session.
fn workflow_session_package(scratch: &tempfile::TempDir) -> PathBuf {
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
    let source = fixtures.join("onnx_genai_workflows/decoder");
    let destination = scratch.path().join("package");
    fn copy_tree(from: &std::path::Path, to: &std::path::Path) {
        std::fs::create_dir_all(to).expect("create package directory");
        for entry in std::fs::read_dir(from).expect("read fixture") {
            let entry = entry.expect("fixture entry");
            let target = to.join(entry.file_name());
            if entry.file_type().expect("file type").is_dir() {
                copy_tree(&entry.path(), &target);
            } else {
                std::fs::copy(entry.path(), target).expect("copy fixture file");
            }
        }
    }
    copy_tree(&source, &destination);
    // The conformance fixture ships no tokenizer because its conformance is
    // about the workflow. An HTTP completion is text, so this borrows the tiny
    // decoder's tokenizer — 32 ids, all inside the fixture's 128-wide vocab.
    std::fs::copy(
        fixtures.join("tiny-llm/tokenizer.json"),
        destination.join("tokenizer.json"),
    )
    .expect("copy tokenizer");
    destination
}

/// The same package with its declared conversation removed — what every
/// migrated interpreted decoder package looked like before this.
fn workflow_package_without_conversation(scratch: &tempfile::TempDir) -> PathBuf {
    let destination = workflow_session_package(scratch);
    let metadata = destination.join("inference_metadata.yaml");
    let mut document: Value =
        serde_yaml::from_str(&std::fs::read_to_string(&metadata).expect("read metadata"))
            .expect("parse metadata");
    document["pipeline"]["workflow"]["state"]
        .as_object_mut()
        .expect("workflow declares state")
        .remove("conversation")
        .expect("the fixture declares a conversation");
    let capabilities = document["pipeline"]["workflow"]["manifest"]["capabilities"]
        .as_array_mut()
        .expect("the manifest declares capabilities");
    capabilities.retain(|capability| capability.as_str() != Some("session_state_lease"));
    std::fs::write(
        &metadata,
        serde_yaml::to_string(&document).expect("serialize metadata"),
    )
    .expect("write metadata");
    destination
}

async fn chat_turn(
    router: axum::Router,
    session: Option<&str>,
    prompt: &str,
) -> (StatusCode, Value) {
    chat_turn_for(router, "workflow-multi-turn", session, prompt).await
}

async fn chat_turn_for(
    router: axum::Router,
    model: &str,
    session: Option<&str>,
    prompt: &str,
) -> (StatusCode, Value) {
    let mut builder = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json");
    if let Some(session) = session {
        builder = builder.header("x-session-id", session);
    }
    let response = router
        .oneshot(
            builder
                .body(Body::from(
                    json!({
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 3,
                        "temperature": 0.0
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    (status, body)
}

fn session_tokens(body: &Value) -> u64 {
    body["session_token_count"]
        .as_u64()
        .unwrap_or_else(|| panic!("a session response reports its token count: {body}"))
}

/// Three turns in one session accumulate one conversation; a fourth id does not
/// see it; deleting the session releases it.
///
/// The arithmetic is what makes this non-vacuous. `usage.prompt_tokens` is what
/// this turn was actually prefilled with — the conversation plus the request —
/// and `session_token_count` is what the session holds afterwards, so the two
/// differ by exactly this turn's generation. A turn that restarted would be
/// prefilled with its own request alone, and both numbers would be short by
/// everything said before it.
#[tokio::test]
async fn chat_completions_continue_a_conversation_across_requests() {
    let scratch = tempfile::tempdir().expect("scratch directory");
    let package = workflow_session_package(&scratch);
    let state = AppState::load(&package, Some("workflow-multi-turn".to_string()))
        .expect("load interpreted package");
    let router = app(state);

    let session = "sess-multi-turn";
    let mut turns = Vec::new();
    let mut previous = 0u64;
    for (index, prompt) in ["hello world", "the quick", "brown fox"].iter().enumerate() {
        let (status, body) = chat_turn(router.clone(), Some(session), prompt).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let prefill = body["usage"]["prompt_tokens"]
            .as_u64()
            .expect("prompt tokens");
        let generated = body["usage"]["completion_tokens"]
            .as_u64()
            .expect("completion tokens");
        assert!(
            prefill > previous,
            "turn {} is prefilled with the conversation ({previous}) and its own request: {body}",
            index + 1
        );
        assert_eq!(
            session_tokens(&body),
            prefill + generated,
            "the conversation after turn {} is what it was prefilled with plus what it \
             generated: {body}",
            index + 1
        );
        previous = prefill + generated;
        turns.push(body);
    }
    let third = turns.pop().expect("three turns");
    assert_eq!(third["session_id"].as_str(), Some(session));

    // A different id is a different conversation, from its first turn.
    let (status, isolated) = chat_turn(router.clone(), Some("sess-other"), "brown fox").await;
    assert_eq!(status, StatusCode::OK, "{isolated}");
    assert!(
        session_tokens(&isolated) < session_tokens(&third),
        "an independent session starts its own conversation"
    );
    assert_eq!(
        session_tokens(&isolated),
        isolated["usage"]["prompt_tokens"].as_u64().unwrap()
            + isolated["usage"]["completion_tokens"].as_u64().unwrap(),
        "a first turn is prefilled with its own request and nothing else"
    );

    // And the session the server handed out is released when it is deleted.
    let created = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/sessions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(created.status(), StatusCode::OK);
    let created: Value =
        serde_json::from_slice(&to_bytes(created.into_body(), usize::MAX).await.unwrap()).unwrap();
    let id = created["id"].as_str().expect("session id").to_string();
    let deleted = router
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
    assert_eq!(deleted.status(), StatusCode::NO_CONTENT);
    let missing = router
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/v1/sessions/{id}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        missing.status(),
        StatusCode::NOT_FOUND,
        "a deleted session is gone, not emptied"
    );
}

/// A request with no session is unchanged by the package declaring one.
#[tokio::test]
async fn chat_completions_without_a_session_stay_stateless() {
    let scratch = tempfile::tempdir().expect("scratch directory");
    let package = workflow_session_package(&scratch);
    let state = AppState::load(&package, Some("workflow-multi-turn".to_string()))
        .expect("load interpreted package");
    let router = app(state);

    let (first_status, first) = chat_turn(router.clone(), None, "hello world").await;
    let (second_status, second) = chat_turn(router.clone(), None, "hello world").await;
    assert_eq!(first_status, StatusCode::OK, "{first}");
    assert_eq!(second_status, StatusCode::OK, "{second}");
    assert!(first["session_token_count"].is_null());
    assert_eq!(
        first["usage"], second["usage"],
        "a stateless request leaves nothing behind for the next one"
    );
    assert_eq!(
        first["choices"][0]["message"]["content"],
        second["choices"][0]["message"]["content"]
    );
}

/// A package that cannot continue a conversation says so with a status a client
/// can act on.
///
/// 500 told a caller the server had failed and to retry something that will
/// never succeed. 409 says the request and the loaded package disagree, which
/// is what happened.
#[tokio::test]
async fn sessions_are_refused_with_a_conflict_for_a_package_that_cannot_continue_one() {
    let scratch = tempfile::tempdir().expect("scratch directory");
    let package = workflow_package_without_conversation(&scratch);
    let state = AppState::load(&package, Some("workflow-multi-turn".to_string()))
        .expect("load interpreted package");
    let router = app(state);

    // The direct session endpoint.
    let created = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/sessions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(created.status(), StatusCode::CONFLICT);
    let body: Value =
        serde_json::from_slice(&to_bytes(created.into_body(), usize::MAX).await.unwrap()).unwrap();
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(
        message.contains("scope: session"),
        "the refusal names what the package has to declare: {body}"
    );

    // And a completion that carries a session id, which creates one implicitly.
    let (status, chat) = chat_turn(router.clone(), Some("sess-refused"), "hello world").await;
    assert_eq!(status, StatusCode::CONFLICT, "{chat}");

    let completion = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .header("x-session-id", "sess-refused")
                .body(Body::from(
                    json!({
                        "model": "workflow-multi-turn",
                        "prompt": "hello world",
                        "max_tokens": 3,
                        "temperature": 0.0
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(completion.status(), StatusCode::CONFLICT);

    // Stateless generation against the same package is unaffected: a package
    // with no conversation answers one question at a time, which is a fact
    // about it rather than a fault.
    let (status, stateless) = chat_turn(router, None, "hello world").await;
    assert_eq!(status, StatusCode::OK, "{stateless}");
}

/// A non-token workflow keeps its session handle.
///
/// A speech package has no conversation to lose, so refusing it a session would
/// make "sessions" a property of which package shape was loaded — which is the
/// caller-side split this runtime does not have.
#[tokio::test]
async fn sessions_still_open_for_a_package_that_publishes_no_tokens() {
    let router = app(speech_state_from("speech_wav", "workflow-sessions-speech"));
    let created = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/sessions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(created.status(), StatusCode::OK);
}

/// Concurrent turns of one conversation cannot lose an update.
///
/// Two turns that both read the conversation before either writes it would
/// leave the loser's prompt and generation nowhere, and nothing would report
/// that they were lost. The routing lease settles it by refusing, so each
/// racing turn either ran or said out loud that it did not: the conversation is
/// exactly what the *admitted* turns produce in series, and every turn that was
/// not admitted answered with a typed conflict naming the session.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_turns_on_one_session_do_not_lose_a_conversation() {
    let scratch = tempfile::tempdir().expect("scratch directory");
    let package = workflow_session_package(&scratch);
    let state = AppState::load(&package, Some("workflow-multi-turn".to_string()))
        .expect("load interpreted package");
    let router = app(state.clone());
    let session = "sess-concurrent";

    // The same prompt every time, so each admitted turn contributes an
    // identical number of request tokens and the expected total is exact
    // arithmetic rather than a bound.
    const TURNS: usize = 4;
    // A barrier rather than four spawns in a row: every task is released into
    // the router at the same instant, on a runtime with real threads, so the
    // race the lease exists to settle is actually run.
    let start = Arc::new(tokio::sync::Barrier::new(TURNS));
    let mut inflight = Vec::new();
    for _ in 0..TURNS {
        let router = router.clone();
        let start = start.clone();
        inflight.push(tokio::spawn(async move {
            start.wait().await;
            chat_turn(router, Some(session), "hello world").await
        }));
    }
    let mut generated = 0u64;
    let mut first_turn_prefill = u64::MAX;
    let mut conversation = 0u64;
    let mut admitted = 0u64;
    let mut refused = 0u64;
    for handle in inflight {
        let (status, body) = handle.await.expect("turn completed");
        match status {
            StatusCode::OK => {
                admitted += 1;
                generated += body["usage"]["completion_tokens"]
                    .as_u64()
                    .expect("generated");
                // Whichever turn ran first was prefilled with its own request
                // alone.
                first_turn_prefill = first_turn_prefill
                    .min(body["usage"]["prompt_tokens"].as_u64().expect("prefill"));
                conversation = conversation.max(session_tokens(&body));
            }
            StatusCode::CONFLICT => {
                refused += 1;
                assert_eq!(body["error"]["type"], "conflict_error", "{body}");
                assert!(
                    body["error"]["message"]
                        .as_str()
                        .expect("a refusal explains itself")
                        .contains(session),
                    "a refusal names the conversation it is about: {body}"
                );
            }
            other => panic!("a racing turn is admitted or refused, not {other}: {body}"),
        }
    }
    assert!(admitted >= 1, "somebody has to win the race");
    assert_eq!(admitted + refused, TURNS as u64);

    // Every admitted turn's request is in the conversation exactly once, and so
    // is every token it generated. A lost update leaves the total short by the
    // turn whose write was overwritten; a silently queued turn leaves it long.
    assert_eq!(
        conversation,
        first_turn_prefill * admitted + generated,
        "the conversation is every admitted turn's request and generation, once each"
    );

    // The race is over, so nothing may still be holding a lease.
    assert_eq!(leases_held(&state), 0, "a finished race leaves no lease");
    assert!(
        state.sessions.get(session).expect("registry").is_some(),
        "a refused turn does not take the conversation with it"
    );
}

/// How many sessions currently hold a turn lease, across every loaded model.
///
/// One map for the whole server, so this is the honest total rather than one
/// engine's view of it.
fn leases_held(state: &AppState) -> usize {
    state.sessions.leases().held()
}

/// The model-qualified binding a client id currently resolves to.
fn binding_of(state: &AppState, client_id: &str) -> crate::lease::ModelSessionPlacement {
    state
        .sessions
        .get(client_id)
        .expect("registry")
        .unwrap_or_else(|| panic!("{client_id} is registered"))
}

/// A turn already in flight refuses the next one instead of queueing it.
///
/// This is the whole of Phase 2 in one assertion. The lease a live turn holds
/// is taken here directly, which is exactly the guard a live turn carries, and
/// a second turn is raced against it on another thread: it must come back 409
/// *while the first is still holding*, not later and not 200. Queueing it
/// behind the first would be the pre-Phase-2 behaviour — a slow success that
/// reads a conversation the first turn is part-way through replacing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_second_turn_on_a_busy_session_is_refused_rather_than_queued() {
    let scratch = tempfile::tempdir().expect("scratch directory");
    let package = workflow_session_package(&scratch);
    let state = AppState::load(&package, Some("workflow-multi-turn".to_string()))
        .expect("load interpreted package");
    let router = app(state.clone());
    let session = "sess-busy";

    // One sequential turn establishes the conversation and its placement.
    let (status, first) = chat_turn(router.clone(), Some(session), "hello world").await;
    assert_eq!(status, StatusCode::OK, "{first}");
    let established = session_tokens(&first);

    let engine = state
        .registry
        .resolve("")
        .expect("resolve default model")
        .expect("a model is loaded")
        .engine
        .clone();
    // Stand in for a turn in flight: this is the same guard `generate` carries.
    let held = state
        .sessions
        .acquire(binding_of(&state, session), session)
        .expect("an idle session is leasable");

    let start = Arc::new(tokio::sync::Barrier::new(2));
    let racer = tokio::spawn({
        let router = router.clone();
        let start = start.clone();
        async move {
            start.wait().await;
            chat_turn(router, Some(session), "hello world").await
        }
    });
    start.wait().await;
    let (status, body) = racer.await.expect("the racing turn answered");
    assert_eq!(
        status,
        StatusCode::CONFLICT,
        "a busy session refuses, it does not queue: {body}"
    );
    assert_eq!(body["error"]["type"], "conflict_error", "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .expect("a refusal explains itself")
            .contains(session),
        "{body}"
    );

    // And the refused turn did no work on its way to being refused: it never
    // charged an admission permit, which is what distinguishes a refusal taken
    // before enqueue from one taken after the request was already queued.
    assert_eq!(
        engine.generation_capacity.available_permits(),
        usize::try_from(engine.generation_capacity_size).expect("capacity fits"),
        "a refused turn occupies no queue slot"
    );

    // The refusal cost the conversation nothing.
    assert_eq!(
        state
            .sessions
            .get(session)
            .expect("registry")
            .map(|binding| binding.placement().engine_session_id),
        Some(held.placement().engine_session_id),
        "a refused turn leaves the binding exactly where it was"
    );

    // And when the turn that held it ends, the session is ordinary again.
    drop(held);
    assert_eq!(leases_held(&state), 0);
    let (status, resumed) = chat_turn(router, Some(session), "hello world").await;
    assert_eq!(status, StatusCode::OK, "{resumed}");
    assert!(
        session_tokens(&resumed) > established,
        "the conversation continued from where it was: {resumed}"
    );
}

/// Two different conversations do not refuse each other.
///
/// The lease is keyed by session, so distinct sessions are behaviour-identical
/// to before: both are admitted. They may still execute one after the other —
/// this phase adds no parallelism — but neither is ever told it conflicts.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn turns_on_distinct_sessions_are_all_admitted() {
    let scratch = tempfile::tempdir().expect("scratch directory");
    let package = workflow_session_package(&scratch);
    let state = AppState::load(&package, Some("workflow-multi-turn".to_string()))
        .expect("load interpreted package");
    let router = app(state.clone());

    const SESSIONS: [&str; 4] = ["sess-a", "sess-b", "sess-c", "sess-d"];
    let start = Arc::new(tokio::sync::Barrier::new(SESSIONS.len()));
    let mut inflight = Vec::new();
    for session in SESSIONS {
        let router = router.clone();
        let start = start.clone();
        inflight.push(tokio::spawn(async move {
            start.wait().await;
            chat_turn(router, Some(session), "hello world").await
        }));
    }
    for handle in inflight {
        let (status, body) = handle.await.expect("turn completed");
        assert_eq!(
            status,
            StatusCode::OK,
            "distinct conversations never conflict: {body}"
        );
    }
    assert_eq!(leases_held(&state), 0);
    for session in SESSIONS {
        assert!(
            state.sessions.get(session).expect("registry").is_some(),
            "{session} kept its binding"
        );
    }
}

/// A stateless request is untouched by any of this.
///
/// Requests that carry no session take no lease, so racing them cannot refuse
/// them — including while a session on the same engine is mid-turn.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stateless_requests_take_no_lease_and_are_never_refused() {
    let scratch = tempfile::tempdir().expect("scratch directory");
    let package = workflow_session_package(&scratch);
    let state = AppState::load(&package, Some("workflow-multi-turn".to_string()))
        .expect("load interpreted package");
    let router = app(state.clone());

    let (status, first) =
        chat_turn(router.clone(), Some("sess-stateless-peer"), "hello world").await;
    assert_eq!(status, StatusCode::OK, "{first}");
    let held = state
        .sessions
        .acquire(
            binding_of(&state, "sess-stateless-peer"),
            "sess-stateless-peer",
        )
        .expect("an idle session is leasable");

    let start = Arc::new(tokio::sync::Barrier::new(3));
    let mut inflight = Vec::new();
    for _ in 0..2 {
        let router = router.clone();
        let start = start.clone();
        inflight.push(tokio::spawn(async move {
            start.wait().await;
            chat_turn(router, None, "hello world").await
        }));
    }
    start.wait().await;
    for handle in inflight {
        let (status, body) = handle.await.expect("turn completed");
        assert_eq!(
            status,
            StatusCode::OK,
            "a stateless turn is never refused: {body}"
        );
    }
    assert_eq!(leases_held(&state), 1, "only the held session has a lease");
    drop(held);
    assert_eq!(leases_held(&state), 0);
}

/// Deleting a session mid-turn is refused, not raced.
///
/// Close is a mutation of the conversation, so it takes the same lease a turn
/// does. A delete that raced a live turn would free the engine session the turn
/// is still writing, and would let the client id be handed to a new
/// conversation while the old one was still running.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deleting_a_session_during_a_turn_is_refused_and_the_session_survives() {
    let scratch = tempfile::tempdir().expect("scratch directory");
    let package = workflow_session_package(&scratch);
    let state = AppState::load(&package, Some("workflow-multi-turn".to_string()))
        .expect("load interpreted package");
    let router = app(state.clone());
    let session = "sess-delete-race";

    let (status, first) = chat_turn(router.clone(), Some(session), "hello world").await;
    assert_eq!(status, StatusCode::OK, "{first}");
    let held = state
        .sessions
        .acquire(binding_of(&state, session), session)
        .expect("an idle session is leasable");

    let start = Arc::new(tokio::sync::Barrier::new(2));
    let deleter = tokio::spawn({
        let router = router.clone();
        let start = start.clone();
        async move {
            start.wait().await;
            let response = router
                .oneshot(
                    Request::builder()
                        .method("DELETE")
                        .uri(format!("/v1/sessions/{session}"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            let status = response.status();
            let body: Value =
                serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                    .unwrap();
            (status, body)
        }
    });
    start.wait().await;
    let (status, body) = deleter.await.expect("the delete answered");
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["error"]["type"], "conflict_error", "{body}");
    assert!(
        state.sessions.get(session).expect("registry").is_some(),
        "a refused delete does not remove the binding"
    );

    // Once the turn ends the delete goes through, and the id is free again.
    drop(held);
    let response = router
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/v1/sessions/{session}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NO_CONTENT);
    assert!(state.sessions.get(session).expect("registry").is_none());
    assert_eq!(leases_held(&state), 0);
}

// ── Phase 2: one lease map, across every loaded model ───────────────────────

fn handle_for(state: &AppState, model: &str) -> Arc<crate::registry::ModelHandle> {
    state
        .registry
        .resolve(model)
        .expect("registry")
        .unwrap_or_else(|| panic!("{model} is loaded"))
}

/// Two loaded models, one real engine session opened and bound in each.
///
/// The fixture exists to reproduce the collision that makes a model-blind lease
/// key wrong. Every engine numbers its sessions from its own counter and every
/// worker pool starts at worker 0, so `model-a`'s first session and
/// `model-b`'s first session have the *identical* placement while naming two
/// entirely unrelated conversations. The returned placement is that shared
/// value: a key that is not model-qualified cannot tell these two apart, and
/// every test below is a way of asking what that costs.
async fn colliding_two_model_sessions(
    config: ServerConfig,
) -> (AppState, crate::worker::SessionPlacement) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm");
    let specs = vec![
        ModelSpec {
            id: "model-a".to_string(),
            path: path.clone(),
            eager: true,
            warmup: false,
        },
        ModelSpec {
            id: "model-b".to_string(),
            path,
            eager: true,
            warmup: false,
        },
    ];
    let state = AppState::load_from_specs(specs, config).expect("load two tiny-llm fixtures");
    let a = handle_for(&state, "model-a");
    let b = handle_for(&state, "model-b");
    let placement_a = a.engine.create_session().await.expect("model-a session");
    let placement_b = b.engine.create_session().await.expect("model-b session");
    assert_eq!(
        placement_a, placement_b,
        "the fixture is only interesting if the two engines collide, and they do: \
         each numbers its own sessions and each pool starts at worker 0",
    );
    state
        .sessions
        .insert("sess-on-a".to_string(), a.engine.binding(placement_a))
        .expect("bind model-a's session");
    state
        .sessions
        .insert("sess-on-b".to_string(), b.engine.binding(placement_b))
        .expect("bind model-b's session");
    (state, placement_a)
}

/// `DELETE` closes the conversation on the model that owns it.
///
/// The route used to resolve the *default* model and close there. With two
/// models loaded and their placements collided, that closed model-a's
/// conversation whenever a client deleted model-b's — destroying a live
/// conversation nobody asked about while leaving the one that was asked about
/// running, and orphaned.
#[tokio::test]
async fn deleting_a_session_closes_it_on_the_model_that_owns_it() {
    let (state, placement) = colliding_two_model_sessions(ServerConfig::default()).await;
    let router = app(state.clone());
    let a = handle_for(&state, "model-a");
    let b = handle_for(&state, "model-b");

    let response = router
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/v1/sessions/sess-on-b")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NO_CONTENT);
    assert!(state.sessions.get("sess-on-b").expect("registry").is_none());
    assert_eq!(leases_held(&state), 0, "the close released its lease");

    // model-b's engine no longer has the session, because that is the one that
    // was closed.
    let lease = state
        .sessions
        .acquire(b.engine.binding(placement), "sess-on-b")
        .expect("nothing holds it");
    let error = b
        .engine
        .close_session(lease)
        .await
        .expect_err("the DELETE already closed it on model-b");
    assert!(error.to_string().contains("not found"), "{error}");

    // model-a's identically placed conversation was never touched.
    let lease = state
        .sessions
        .acquire(a.engine.binding(placement), "sess-on-a")
        .expect("idle");
    a.engine
        .close_session(lease)
        .await
        .expect("model-a's session is exactly where it was");
}

/// A lease taken on one model cannot be spent on another.
///
/// The guard names the model it leased, and the driver checks that name before
/// it destroys anything. This is the backstop for the whole model-qualified
/// key: even if a future caller resolved the wrong engine, the close is refused
/// rather than performed on the wrong conversation.
#[tokio::test]
async fn a_lease_from_one_model_cannot_close_another_models_session() {
    let (state, placement) = colliding_two_model_sessions(ServerConfig::default()).await;
    let a = handle_for(&state, "model-a");
    let b = handle_for(&state, "model-b");

    let lease = state
        .sessions
        .acquire(a.engine.binding(placement), "sess-on-a")
        .expect("idle");
    let error = b
        .engine
        .close_session(lease)
        .await
        .expect_err("model-b must refuse a lease it did not issue");
    assert!(
        error
            .to_string()
            .contains("cannot be closed on model 'model-b'"),
        "{error}"
    );

    // And the conversation the refused close named is still there.
    let lease = state
        .sessions
        .acquire(a.engine.binding(placement), "sess-on-a")
        .expect("the refused close released its lease");
    a.engine
        .close_session(lease)
        .await
        .expect("model-a's session survived the refusal");
}

/// Eviction never takes another model's live conversation, refuses when every
/// binding is busy, and closes what it does take on the right engine.
///
/// Three properties in one arc, because they are one arc: a full registry has
/// to choose a victim, that choice has to skip whatever has a turn in flight
/// regardless of which model it belongs to, and when there is nothing to choose
/// the answer is a refusal rather than a quiet overshoot of `max_sessions`.
#[tokio::test]
async fn eviction_across_models_skips_live_conversations_and_refuses_when_all_are_busy() {
    let (state, placement) = colliding_two_model_sessions(ServerConfig {
        max_sessions: 2,
        ..ServerConfig::default()
    })
    .await;
    let a = handle_for(&state, "model-a");
    let b = handle_for(&state, "model-b");

    // A turn in flight on model-b's session, and model-b's binding made the
    // least recently accessed one: the obvious victim, and the wrong one.
    let turn_on_b = state
        .sessions
        .acquire(b.engine.binding(placement), "sess-on-b")
        .expect("a turn on model-b");
    state
        .sessions
        .get("sess-on-a")
        .expect("registry")
        .expect("still bound");

    let second_on_a = a
        .engine
        .create_session()
        .await
        .expect("another model-a session");
    let evicted = state
        .sessions
        .insert("sess-on-a-2".to_string(), a.engine.binding(second_on_a))
        .expect("room can be made")
        .expect("something had to go");
    assert_eq!(
        evicted.model().as_str(),
        "model-a",
        "the live model-b conversation is skipped, however old it is",
    );
    assert_eq!(evicted.placement(), placement);
    // Closed on the model the guard names — which for a collided placement is
    // the only thing that distinguishes it from destroying model-b's.
    a.engine
        .close_session(evicted)
        .await
        .expect("the evicted conversation is closed on its own engine");
    assert_eq!(state.sessions.len(), 2, "the bound holds");

    // Now every bound conversation has a turn in flight, so there is no victim.
    let turn_on_a = state
        .sessions
        .acquire(a.engine.binding(second_on_a), "sess-on-a-2")
        .expect("a turn on model-a");
    let second_on_b = b
        .engine
        .create_session()
        .await
        .expect("another model-b session");
    let refused = state
        .sessions
        .insert("sess-on-b-2".to_string(), b.engine.binding(second_on_b))
        .expect_err("a bound that yields under load is not a bound");
    assert!(matches!(
        refused,
        crate::session::SessionRegistryError::AtCapacity { bound: 2 }
    ));
    assert_eq!(
        state.sessions.len(),
        2,
        "the registry never went over its bound, even transiently",
    );

    // The refusal is transient: model-b's turn ends, and its binding becomes
    // the victim it always was — closed on model-b's engine.
    drop(turn_on_b);
    let evicted = state
        .sessions
        .insert("sess-on-b-2".to_string(), b.engine.binding(second_on_b))
        .expect("room can be made again")
        .expect("something had to go");
    assert_eq!(evicted.model().as_str(), "model-b");
    assert_eq!(evicted.placement(), placement);
    b.engine
        .close_session(evicted)
        .await
        .expect("model-b's conversation is closed on model-b");
    assert_eq!(state.sessions.len(), 2);

    drop(turn_on_a);
    assert_eq!(leases_held(&state), 0, "nothing leaked a lease");
}

/// A session id from one model is refused on another, not silently continued
/// into whatever conversation shares its placement there.
#[tokio::test]
async fn a_session_bound_to_one_model_is_refused_on_another() {
    let (state, _placement) = colliding_two_model_sessions(ServerConfig::default()).await;
    let router = app(state.clone());

    let (status, body) = chat_turn_for(router, "model-a", Some("sess-on-b"), "hello").await;
    assert_eq!(
        status,
        StatusCode::CONFLICT,
        "model-a must not generate into model-b's conversation: {body}"
    );
    assert_eq!(body["error"]["type"], "conflict_error", "{body}");
    let message = body["error"]["message"].as_str().unwrap_or_default();
    assert!(message.contains("model-b"), "{body}");
    assert!(message.contains("model-a"), "{body}");
    assert_eq!(leases_held(&state), 0, "a refusal takes no lease with it");
}

/// A full registry whose every conversation is busy refuses a new session
/// instead of admitting one over the bound.
///
/// `max_sessions` is what an operator sized the server's session memory
/// against. Admitting one anyway left the registry permanently at `max + k` —
/// nothing ever walks it back down, because the next insert evicts one and adds
/// one. The refusal is a 429 for the same reason every other "at capacity"
/// answer here is: the request succeeds unchanged once any turn ends.
#[tokio::test]
async fn creating_a_session_when_every_conversation_is_busy_is_refused_not_overshot() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm");
    let state = AppState::load_with_config(
        &model_dir,
        Some("tiny-llm".to_string()),
        ServerConfig {
            max_sessions: 1,
            ..ServerConfig::default()
        },
    )
    .expect("load fixture with a one-session bound");
    let router = app(state.clone());

    let first = create_http_session(&router).await;
    assert_eq!(state.sessions.len(), 1);
    let turn = state
        .sessions
        .acquire(binding_of(&state, &first), &first)
        .expect("a turn on the only session");

    let response = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/sessions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert!(
        response.headers().contains_key("retry-after"),
        "a transient capacity refusal says when to come back",
    );
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(body["error"]["type"], "resource_limit_error", "{body}");
    assert_eq!(
        state.sessions.len(),
        1,
        "the refused create left the registry at its bound, not over it",
    );
    assert!(
        state.sessions.get(&first).expect("registry").is_some(),
        "and it did not close the live conversation to make room",
    );

    // Once the turn ends the same request succeeds, by evicting — and the
    // registry is still at exactly its bound.
    drop(turn);
    let second = create_http_session(&router).await;
    assert_ne!(second, first);
    assert_eq!(state.sessions.len(), 1);
    assert!(state.sessions.get(&first).expect("registry").is_none());
    assert_eq!(leases_held(&state), 0);
}

async fn create_http_session(router: &axum::Router) -> String {
    let response = router
        .clone()
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
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    body["id"].as_str().expect("a session id").to_string()
}

/// A request that fails admission must not open a session, and must not evict
/// somebody else's.
///
/// Counting the conversation before admission is right; *creating* the session
/// before admission was not. A client whose requests are all rejected would
/// otherwise destroy one live conversation per rejection once the registry is
/// full, and strand an engine session it never used.
#[tokio::test]
async fn a_rejected_request_neither_opens_nor_evicts_a_session() {
    let scratch = tempfile::tempdir().expect("scratch directory");
    let package = workflow_session_package(&scratch);
    let state = AppState::load(&package, Some("workflow-multi-turn".to_string()))
        .expect("load interpreted package");
    let router = app(state.clone());

    // A live conversation.
    let (status, first) = chat_turn(router.clone(), Some("sess-live"), "hello world").await;
    assert_eq!(status, StatusCode::OK, "{first}");
    let live = state
        .sessions
        .get("sess-live")
        .expect("registry")
        .expect("the live session is registered");

    // A request that cannot be admitted: more output than the model's context.
    let rejected = router
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .header("x-session-id", "sess-rejected")
                .body(Body::from(
                    json!({
                        "model": "workflow-multi-turn",
                        "messages": [{"role": "user", "content": "hello world"}],
                        "max_tokens": 1_000_000,
                        "temperature": 0.0
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(rejected.status(), StatusCode::BAD_REQUEST);

    assert_eq!(
        state.sessions.get("sess-rejected").expect("registry"),
        None,
        "a rejected request opens no session"
    );
    assert_eq!(
        state.sessions.get("sess-live").expect("registry"),
        Some(live),
        "a rejected request evicts nobody"
    );

    // And the live conversation still continues.
    let (status, second) = chat_turn(router, Some("sess-live"), "the quick").await;
    assert_eq!(status, StatusCode::OK, "{second}");
    assert!(session_tokens(&second) > session_tokens(&first));
}

/// A fill-in-the-middle completion is not a turn in a conversation, and never
/// opens one.
///
/// The route refuses the combination outright, and `conversational_session_id`
/// is the second answer to the same question for any caller that reaches
/// `run_completion` another way: the FIM submit path takes no session, so
/// opening one would claim an LRU slot — closing another client's live
/// conversation once the registry is full — for a request that never touches it.
#[tokio::test]
async fn a_fim_completion_carrying_a_session_id_opens_no_session() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm");
    let state = AppState::load(&model_dir, Some("tiny-llm".to_string()))
        .expect("load fixture")
        .with_default_fim_config(Some(onnx_genai_engine::FimConfig {
            prefix_token: "<PRE>".to_string(),
            middle_token: "<MID>".to_string(),
            suffix_token: "<SUF>".to_string(),
            format: onnx_genai_engine::FimFormat::PSM,
        }));
    let router = app(state.clone());

    let response = router
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .header("x-session-id", "sess-fim")
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
    assert_eq!(
        response.status(),
        StatusCode::BAD_REQUEST,
        "a FIM completion cannot be a turn in a conversation"
    );
    assert_eq!(
        state.sessions.get("sess-fim").expect("registry"),
        None,
        "and it opens no session on its way to being refused"
    );

    // Without the header the same request is an ordinary FIM completion.
    let plain = app(state.clone())
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
    assert_eq!(plain.status(), StatusCode::OK);
}

/// An ORT decode-core session appends each request behind its retained KV.
#[tokio::test]
async fn an_ort_decode_core_session_is_charged_for_what_it_attends() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm");
    let state = AppState::load(&model_dir, Some("tiny-llm".to_string())).expect("load fixture");
    let router = app(state);
    let session = "sess-decode-core";

    let (status, first) = chat_turn_for(router.clone(), "tiny-llm", Some(session), "hello").await;
    assert_eq!(status, StatusCode::OK, "{first}");
    let first_prompt = first["usage"]["prompt_tokens"]
        .as_u64()
        .expect("prompt tokens");
    let retained = session_tokens(&first);
    assert_eq!(
        retained,
        first_prompt
            + first["usage"]["completion_tokens"]
                .as_u64()
                .expect("generated"),
        "the session retains this turn's prompt and generation: {first}"
    );

    let (status, second) = chat_turn_for(router, "tiny-llm", Some(session), "hello").await;
    assert_eq!(status, StatusCode::OK, "{second}");
    assert_eq!(
        second["usage"]["prompt_tokens"]
            .as_u64()
            .expect("prompt tokens"),
        first_prompt + retained,
        "the continuing turn is charged the sequence it is appended to: {second}"
    );
}

/// A conversation beyond the model window is refused rather than returning an
/// empty successful completion.
#[tokio::test]
async fn a_decode_core_conversation_past_the_window_is_refused_not_answered_empty() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm");
    let state = AppState::load(&model_dir, Some("tiny-llm".to_string())).expect("load fixture");
    let router = app(state);
    let session = "sess-decode-core-window";
    let message = "the quick brown fox jumps over the lazy dog";

    let (status, first) = chat_turn_for(router.clone(), "tiny-llm", Some(session), message).await;
    assert_eq!(status, StatusCode::OK, "{first}");
    assert!(
        first["usage"]["completion_tokens"]
            .as_u64()
            .expect("generated")
            > 0,
        "the first turn generates: {first}"
    );

    let (status, second) = chat_turn_for(router, "tiny-llm", Some(session), message).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{second}");
    assert!(
        second["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("context limit"),
        "the refusal names the limit it hit: {second}"
    );
}

/// A lease the graph carries is not in front of the prompt, so it is not
/// charged.
///
/// This is the half of the accounting that is genuinely zero: a loop-carried or
/// group-held lease lives in a cache the package bounds itself. Charging a
/// request for it would refuse turns for context they do not occupy.
#[tokio::test]
async fn a_graph_carried_lease_is_not_charged_against_the_prompt() {
    let scratch = tempfile::tempdir().expect("scratch directory");
    let package = workflow_session_package(&scratch);
    // Drop the prompt continuation and session-scope a loop-carried cache cell,
    // so the session is carried inside the graph instead.
    let metadata = package.join("inference_metadata.yaml");
    let mut document: Value =
        serde_yaml::from_str(&std::fs::read_to_string(&metadata).expect("read metadata"))
            .expect("parse metadata");
    let state_cells = document["pipeline"]["workflow"]["state"]
        .as_object_mut()
        .expect("workflow declares state");
    state_cells.remove("conversation").expect("a conversation");
    let cache = state_cells.get_mut("cache_0").expect("a cache cell");
    cache["scope"] = Value::String("session".into());
    cache["release_boundary"] = Value::String("session".into());
    std::fs::write(
        &metadata,
        serde_yaml::to_string(&document).expect("serialize metadata"),
    )
    .expect("write metadata");

    let state = AppState::load(&package, Some("workflow-multi-turn".to_string()))
        .expect("load interpreted package");
    let router = app(state);
    let session = "sess-graph-carried";

    let (status, first) = chat_turn(router.clone(), Some(session), "hello world").await;
    assert_eq!(status, StatusCode::OK, "{first}");
    let first_prompt = first["usage"]["prompt_tokens"]
        .as_u64()
        .expect("prompt tokens");

    let (status, second) = chat_turn(router, Some(session), "hello world").await;
    assert_eq!(status, StatusCode::OK, "{second}");
    assert_eq!(
        second["usage"]["prompt_tokens"]
            .as_u64()
            .expect("prompt tokens"),
        first_prompt,
        "a lease carried inside the graph is not in front of the prompt: {second}"
    );
}

/// A prompt-continuation package *is* charged for the conversation, because it
/// really is prefilled again.
///
/// The same two-turn shape as the decode-core case, and the opposite answer:
/// here the second turn's prompt is its own request *plus* everything the first
/// turn left behind, because that is what the runtime puts in front of it.
#[tokio::test]
async fn a_prompt_continuation_turn_is_charged_for_what_is_prepended() {
    let scratch = tempfile::tempdir().expect("scratch directory");
    let package = workflow_session_package(&scratch);
    let state = AppState::load(&package, Some("workflow-multi-turn".to_string()))
        .expect("load interpreted package");
    let router = app(state);
    let session = "sess-prepended";

    let (status, first) = chat_turn(router.clone(), Some(session), "hello world").await;
    assert_eq!(status, StatusCode::OK, "{first}");
    let first_prompt = first["usage"]["prompt_tokens"]
        .as_u64()
        .expect("prompt tokens");
    let conversation = session_tokens(&first);

    let (status, second) = chat_turn(router, Some(session), "hello world").await;
    assert_eq!(status, StatusCode::OK, "{second}");
    let second_prompt = second["usage"]["prompt_tokens"]
        .as_u64()
        .expect("prompt tokens");

    assert_eq!(
        second_prompt,
        first_prompt + conversation,
        "a prepended conversation is part of what the turn is prefilled with"
    );
    assert_eq!(
        session_tokens(&second),
        second_prompt
            + second["usage"]["completion_tokens"]
                .as_u64()
                .expect("generated"),
        "and the conversation it leaves is what it was prefilled with plus what it generated"
    );
}

/// A capability refusal is not a server error, and its body says so.
///
/// `error.type` is what a client branches on. Reporting a package that will
/// never serve a request under the same `server_error` as a crash told callers
/// to retry something that cannot succeed.
#[tokio::test]
async fn a_capability_refusal_body_names_the_disagreement_not_a_fault() {
    let scratch = tempfile::tempdir().expect("scratch directory");
    let package = workflow_package_without_conversation(&scratch);
    let state = AppState::load(&package, Some("workflow-multi-turn".to_string()))
        .expect("load interpreted package");

    let created = app(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/sessions")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(created.status(), StatusCode::CONFLICT);
    let body: Value =
        serde_json::from_slice(&to_bytes(created.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(
        body["error"]["type"].as_str(),
        Some("conflict_error"),
        "a capability refusal is a conflict, not a server fault: {body}"
    );
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("scope: session"),
        "and it names what the package has to declare: {body}"
    );
}

/// A conversation past the bound its package declares is refused 4xx, typed.
///
/// The status is read off the engine's own error variant, so it cannot drift
/// with the wording of a message — which is how it used to be decided.
#[tokio::test]
async fn a_conversation_past_its_bound_is_a_typed_client_error() {
    let scratch = tempfile::tempdir().expect("scratch directory");
    let package = workflow_session_package(&scratch);
    // Narrow only the conversation's own bound, so the caches keep theirs.
    let metadata = package.join("inference_metadata.yaml");
    let mut document: Value =
        serde_yaml::from_str(&std::fs::read_to_string(&metadata).expect("read metadata"))
            .expect("parse metadata");
    document["pipeline"]["workflow"]["inputs"]["package.conversation_limit"] = json!({
        "contract": {"dtype": "int64", "shape": [1]},
        "role": {"kind": "opaque"},
        "source": {"kind": "literal"},
        "required": false,
        "default": 6
    });
    document["pipeline"]["workflow"]["state"]["conversation"]["recurrence"]["max"] =
        Value::String("package.conversation_limit".into());
    std::fs::write(
        &metadata,
        serde_yaml::to_string(&document).expect("serialize metadata"),
    )
    .expect("write metadata");

    let state = AppState::load(&package, Some("workflow-multi-turn".to_string()))
        .expect("load interpreted package");
    let router = app(state);
    let session = "sess-over-bound";

    // The first turn fits; the second cannot.
    let (status, first) = chat_turn(router.clone(), Some(session), "hello").await;
    assert_eq!(status, StatusCode::OK, "{first}");
    let (status, second) = chat_turn(router, Some(session), "hello").await;
    assert_eq!(
        status,
        StatusCode::BAD_REQUEST,
        "a conversation past its declared bound is the caller's to shorten: {second}"
    );
    assert_eq!(
        second["error"]["type"].as_str(),
        Some("invalid_request_error"),
        "{second}"
    );
    assert!(
        second["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("declares a bound of 6"),
        "the refusal names the bound: {second}"
    );
}

/// A turn that fails gives the lease back.
///
/// The refusal a failed turn deserves is the one it failed for. If the guard
/// leaked on the error path, the *next* turn on that session would be answered
/// 409 forever, and the conversation would be unusable without a delete.
#[tokio::test]
async fn a_failed_turn_releases_its_lease() {
    let scratch = tempfile::tempdir().expect("scratch directory");
    let package = workflow_session_package(&scratch);
    // Narrow the conversation's own bound so the second turn cannot fit.
    let metadata = package.join("inference_metadata.yaml");
    let mut document: Value =
        serde_yaml::from_str(&std::fs::read_to_string(&metadata).expect("read metadata"))
            .expect("parse metadata");
    document["pipeline"]["workflow"]["inputs"]["package.conversation_limit"] = json!({
        "contract": {"dtype": "int64", "shape": [1]},
        "role": {"kind": "opaque"},
        "source": {"kind": "literal"},
        "required": false,
        "default": 6
    });
    document["pipeline"]["workflow"]["state"]["conversation"]["recurrence"]["max"] =
        Value::String("package.conversation_limit".into());
    std::fs::write(
        &metadata,
        serde_yaml::to_string(&document).expect("serialize metadata"),
    )
    .expect("write metadata");

    let state = AppState::load(&package, Some("workflow-multi-turn".to_string()))
        .expect("load interpreted package");
    let router = app(state.clone());
    let session = "sess-failing";

    let (status, first) = chat_turn(router.clone(), Some(session), "hello").await;
    assert_eq!(status, StatusCode::OK, "{first}");
    assert_eq!(leases_held(&state), 0, "a finished turn holds nothing");

    for attempt in 0..3 {
        let (status, body) = chat_turn(router.clone(), Some(session), "hello").await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "attempt {attempt} must fail for the bound it broke, not for a leaked lease: {body}"
        );
        assert_eq!(
            body["error"]["type"].as_str(),
            Some("invalid_request_error"),
            "{body}"
        );
        assert_eq!(
            leases_held(&state),
            0,
            "attempt {attempt} left a lease behind"
        );
    }
}

/// A client that walks away does not lock its own conversation out.
///
/// Dropping the request future is the cancellation path: the guard is not held
/// by that future, it rides in the command, so the worker that is part-way
/// through the turn still releases it exactly once when the turn ends. A guard
/// held by the route future instead would either leak here or — worse — be
/// released while the engine was still writing the conversation.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_cancelled_client_does_not_leak_its_session_lease() {
    let scratch = tempfile::tempdir().expect("scratch directory");
    let package = workflow_session_package(&scratch);
    let state = AppState::load(&package, Some("workflow-multi-turn".to_string()))
        .expect("load interpreted package");
    let router = app(state.clone());
    let session = "sess-cancelled";

    // Establish the conversation so the abandoned turn is a continuation.
    let (status, first) = chat_turn(router.clone(), Some(session), "hello world").await;
    assert_eq!(status, StatusCode::OK, "{first}");

    // The abort has to land while the lease is held, which is the window the
    // guard exists to survive. Hitting that window from outside is racy — a turn
    // this small can finish before `abort` is delivered — so wait for the lease
    // to appear before aborting and retry when the turn wins anyway. Retrying is
    // what makes the test deterministic without making it a timing assertion:
    // the invariant below is checked on every attempt, and the loop only exists
    // to guarantee at least one attempt was a real mid-turn cancellation.
    let mut cancelled_mid_turn = false;
    let mut admitted = None;
    for attempt in 0..32 {
        let abandoned = tokio::spawn({
            let router = router.clone();
            async move { chat_turn(router, Some(session), "hello world").await }
        });

        for _ in 0..1_000 {
            if leases_held(&state) == 1 {
                break;
            }
            tokio::task::yield_now().await;
        }
        abandoned.abort();

        match abandoned.await {
            Err(joined) if joined.is_cancelled() => cancelled_mid_turn = true,
            Err(joined) => std::panic::resume_unwind(joined.into_panic()),
            // The turn beat the abort. Not the case under test, but the lease
            // still has to have come back, so fall through and check it.
            Ok((status, body)) => {
                assert_eq!(status, StatusCode::OK, "attempt {attempt}: {body}");
            }
        }

        // The turn that was abandoned may still be running on the worker. What
        // must be true is that it ends, and that ending gives the lease back —
        // so the next turn is admitted rather than refused forever.
        let mut next = None;
        for _ in 0..200 {
            let (status, body) = chat_turn(router.clone(), Some(session), "hello world").await;
            match status {
                StatusCode::OK => {
                    next = Some(body);
                    break;
                }
                StatusCode::CONFLICT => tokio::time::sleep(Duration::from_millis(25)).await,
                other => panic!("attempt {attempt}: unexpected {other}: {body}"),
            }
        }
        let next = next.unwrap_or_else(|| {
            panic!("attempt {attempt}: the abandoned turn never released its lease")
        });
        assert_eq!(
            leases_held(&state),
            0,
            "attempt {attempt} left a lease behind",
        );
        admitted = Some(next);

        if cancelled_mid_turn {
            break;
        }
    }

    assert!(
        cancelled_mid_turn,
        "no attempt managed to abort a turn while its lease was held",
    );
    let admitted = admitted.expect("at least one attempt ran");
    assert!(session_tokens(&admitted) > 0, "{admitted}");
    assert_eq!(leases_held(&state), 0, "nothing is left leased");
    assert!(
        state.sessions.get(session).expect("registry").is_some(),
        "and the conversation is still registered"
    );
}

/// A busy exclusive session is a 409 a client can retry; an over-bound
/// conversation is a 400 it cannot.
///
/// Both are the same engine type and the same driver failure kind, so the status
/// has to come from the variant. Pinning them together is what stops a later
/// edit collapsing them onto one code.
#[test]
fn capability_refusals_map_to_the_status_their_variant_means() {
    use onnx_genai_engine::PackageCapabilityError;

    let no_state = crate::driver::DriverFailure::from_engine_error(&anyhow::Error::from(
        PackageCapabilityError::NoSessionState,
    ));
    let response = crate::routes::generation_failure(no_state);
    assert_eq!(response.status, StatusCode::CONFLICT);
    assert_eq!(response.kind, "conflict_error");

    let busy = crate::driver::DriverFailure::from_engine_error(&anyhow::Error::from(
        PackageCapabilityError::ExclusiveLeaseConflict {
            session: "shared".to_string(),
        },
    ));
    let response = crate::routes::generation_failure(busy);
    assert_eq!(response.status, StatusCode::CONFLICT);
    assert_eq!(response.kind, "conflict_error");
    // The structured field, not the sentence: nothing may depend on wording.
    assert!(response.message.contains("shared"));

    let over_bound = crate::driver::DriverFailure::from_engine_error(&anyhow::Error::from(
        PackageCapabilityError::ConversationOverBound {
            cell: "conversation".to_string(),
            requested: 12,
            bound: 6,
        },
    ));
    let response = crate::routes::generation_failure(over_bound);
    assert_eq!(response.status, StatusCode::BAD_REQUEST);
    assert_eq!(response.kind, "invalid_request_error");
    assert_eq!(
        response.message,
        PackageCapabilityError::ConversationOverBound {
            cell: "conversation".to_string(),
            requested: 12,
            bound: 6,
        }
        .to_string()
    );

    // An ordinary failure is still a server error, so the new kind cannot
    // swallow a real fault.
    let internal =
        crate::driver::DriverFailure::from_engine_error(&anyhow::anyhow!("forward pass failed"));
    let response = crate::routes::generation_failure(internal);
    assert_eq!(response.status, StatusCode::INTERNAL_SERVER_ERROR);
    assert_eq!(response.kind, "server_error");
}

/// All three session refusals, on both paths that can raise them, asserted on
/// status *and* on the body a client branches on.
///
/// One test, because the three are only meaningful against each other: a client
/// has to tell "this package will never do that" (409, and it is `NoSessionState`
/// on `/v1/sessions` or on an `X-Session-Id` request) from "shorten this turn"
/// (400) from "try again in a moment" (409). Statuses come from the typed
/// variant, never from a message.
#[tokio::test]
async fn every_session_refusal_reports_a_status_and_a_type_a_client_can_branch_on() {
    // 1. NoSessionState — the package declares no conversation at all.
    let scratch = tempfile::tempdir().expect("scratch directory");
    let package = workflow_package_without_conversation(&scratch);
    let state = AppState::load(&package, Some("workflow-multi-turn".to_string()))
        .expect("load interpreted package");
    let router = app(state);

    for (path, response) in [
        (
            "/v1/sessions",
            router
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/v1/sessions")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap(),
        ),
        (
            "x-session-id chat",
            router
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/v1/chat/completions")
                        .header(header::CONTENT_TYPE, "application/json")
                        .header("x-session-id", "sess-no-state")
                        .body(Body::from(
                            json!({
                                "model": "workflow-multi-turn",
                                "messages": [{"role": "user", "content": "hello world"}],
                                "max_tokens": 2
                            })
                            .to_string(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap(),
        ),
        (
            "x-session-id completions",
            router
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/v1/completions")
                        .header(header::CONTENT_TYPE, "application/json")
                        .header("x-session-id", "sess-no-state")
                        .body(Body::from(
                            json!({
                                "model": "workflow-multi-turn",
                                "prompt": "hello world",
                                "max_tokens": 2
                            })
                            .to_string(),
                        ))
                        .unwrap(),
                )
                .await
                .unwrap(),
        ),
    ] {
        assert_eq!(response.status(), StatusCode::CONFLICT, "{path}");
        let body: Value =
            serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap())
                .unwrap();
        assert_eq!(
            body["error"]["type"].as_str(),
            Some("conflict_error"),
            "{path}: {body}"
        );
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap_or_default()
                .contains("scope: session"),
            "{path}: {body}"
        );
    }

    // 2. ConversationOverBound — a turn the package's own bound refuses.
    let scratch = tempfile::tempdir().expect("scratch directory");
    let package = workflow_session_package(&scratch);
    let metadata = package.join("inference_metadata.yaml");
    let mut document: Value =
        serde_yaml::from_str(&std::fs::read_to_string(&metadata).expect("read metadata"))
            .expect("parse metadata");
    document["pipeline"]["workflow"]["inputs"]["package.conversation_limit"] = json!({
        "contract": {"dtype": "int64", "shape": [1]},
        "role": {"kind": "opaque"},
        "source": {"kind": "literal"},
        "required": false,
        "default": 6
    });
    document["pipeline"]["workflow"]["state"]["conversation"]["recurrence"]["max"] =
        Value::String("package.conversation_limit".into());
    std::fs::write(
        &metadata,
        serde_yaml::to_string(&document).expect("serialize metadata"),
    )
    .expect("write metadata");
    let state = AppState::load(&package, Some("workflow-multi-turn".to_string()))
        .expect("load interpreted package");
    let router = app(state.clone());

    let (status, first) = chat_turn(router.clone(), Some("sess-bound"), "hello").await;
    assert_eq!(status, StatusCode::OK, "{first}");
    let (status, second) = chat_turn(router.clone(), Some("sess-bound"), "hello").await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{second}");
    assert_eq!(
        second["error"]["type"].as_str(),
        Some("invalid_request_error"),
        "{second}"
    );
    assert!(
        second["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("declares a bound of 6"),
        "{second}"
    );

    // 3. ExclusiveLeaseConflict — a real turn on a session that already has
    //    one, answered over HTTP rather than constructed by hand. This is the
    //    variant that used to be unreachable, and the same session is used on
    //    purpose: an over-bound conversation answers 400 when it is idle and
    //    409 when it is busy, because the lease is decided in the routing layer
    //    before the turn is ever evaluated against the package.
    let busy = state
        .sessions
        .acquire(binding_of(&state, "sess-bound"), "sess-bound")
        .expect("an idle session is leasable");
    let (status, conflict) = chat_turn(router, Some("sess-bound"), "hello").await;
    assert_eq!(status, StatusCode::CONFLICT, "{conflict}");
    assert_eq!(
        conflict["error"]["type"].as_str(),
        Some("conflict_error"),
        "{conflict}"
    );
    assert!(
        conflict["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("sess-bound"),
        "{conflict}"
    );
    drop(busy);
}

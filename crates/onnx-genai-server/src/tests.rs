use crate::{
    AppState, ChatCompletionRequest, CompletionRequest, EmbeddingEncodingFormat, EmbeddingInput,
    EmbeddingRequest, EmbeddingResponse, EmbeddingUsage, EmbeddingVector, ServerConfig, app,
    build_generate_request,
    driver::{DriverCommand, EngineDriver},
    models_config::ModelSpec,
    routes::{CompletionGeneration, collect_generation_result, prepare_completion},
    sse::StopBoundaryBuffer,
};
use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use onnx_genai::{Engine, EngineConfig};
use serde_json::{Value, json};
use std::{io::Cursor, path::PathBuf, time::Duration};
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
    assert_eq!(
        body["error"]["message"],
        "this model does not support audio transcription"
    );
}

#[tokio::test]
async fn audio_chat_against_non_audio_model_returns_400() {
    let response = app(tiny_state())
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "tiny-llm",
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
                        "max_tokens": 1
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
    assert_eq!(
        body["error"]["message"],
        "this model does not support audio input"
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
async fn image_decode_and_preprocessing_use_pipeline_tensor_shape() {
    use onnx_genai_ort::{DataType, PipelineModels};

    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-vlm");
    if !model_dir.is_dir() {
        eprintln!("skipping image preprocessing test: tiny-vlm fixture is absent");
        return;
    }

    let models = PipelineModels::load(&model_dir).unwrap();
    let encoder = models.session("encoder").expect("encoder");
    let input = encoder
        .inputs()
        .iter()
        .find(|input| input.name == "pixel_values")
        .expect("pixel_values");
    assert_eq!(input.dtype, DataType::Float32);
    let spec = crate::image_input::VisionInputSpec::from_input(
        "encoder.pixel_values".to_string(),
        &input.shape,
    )
    .unwrap();
    let tensor = crate::image_input::load_and_preprocess(&[tiny_png_data_uri()], &spec)
        .await
        .unwrap();

    assert_eq!(tensor.shape, input.shape);
    assert_eq!(
        tensor.data.len(),
        input.shape.iter().product::<i64>() as usize
    );
    assert!(tensor.data.iter().all(|value| (0.0..=1.0).contains(value)));
}

#[tokio::test]
async fn sidecar_free_compatibility_package_builds_server_pipeline_and_preprocesses_image() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../onnx-genai-genai-config/tests/fixtures/vlm-complete");
    let handle = crate::state::build_handle(
        &ModelSpec {
            id: "compat-vlm".to_owned(),
            path: model_dir,
            eager: true,
        },
        &ServerConfig::default(),
    )
    .expect("the real server model-loading path accepts the compatibility package");

    assert!(handle.pipeline);
    let vision = handle
        .vision_input
        .as_ref()
        .expect("server constructed executable vision preprocessing");
    let tensor = crate::image_input::load_and_preprocess(&[tiny_png_data_uri()], vision)
        .await
        .expect("server preprocessing executes");
    assert_eq!(tensor.tensors[0].endpoint, "vision_encoder.pixel_values");
    assert!(!tensor.data.is_empty());
    assert!(tensor.num_tiles > 0);
}

#[tokio::test]
async fn vision_request_against_non_pipeline_model_returns_400() {
    let model_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm");
    let state = AppState::load(&model_dir, Some("tiny-llm".to_string())).expect("load fixture");
    let response = app(state)
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(
                    json!({
                        "model": "tiny-llm",
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "describe"},
                                {"type": "image_url", "image_url": {"url": tiny_png_data_uri()}}
                            ]
                        }],
                        "max_tokens": 1
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
    assert_eq!(
        body["error"]["message"],
        "this model does not support image input"
    );
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
    let handle = state.registry.resolve("").unwrap();
    // Create a real engine session and register it, mirroring the session route.
    let engine_session = handle
        .engine
        .create_session()
        .await
        .expect("create engine session");
    let client_id = state.sessions.next_client_id().unwrap();
    state
        .sessions
        .insert(client_id, engine_session)
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
    assert_eq!(config["pipeline"], false);

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
    assert!(body.contains("onnxgenai_vram_used_bytes"));
    assert!(body.contains("onnxgenai_vram_limit_bytes"));
    assert!(body.contains("onnxgenai_host_ram_used_bytes"));
    assert!(body.contains("onnxgenai_host_ram_limit_bytes"));
    assert!(body.contains("onnxgenai_kv_budget_bytes"));
    let after_health = prometheus_sample(
        &body,
        "onnx_genai_requests_total{endpoint=\"/health\",status=\"200\"}",
    );
    assert!(after_health >= before_health + 2);
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
    let handle = state.registry.resolve("").unwrap();
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
    let handle = state.registry.resolve("").unwrap();
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

#[test]
fn pipeline_input_tensor_carries_num_tiles_for_image() {
    use crate::driver::PipelineInputTensor;

    let tensor = PipelineInputTensor {
        endpoint: "encoder.pixel_values".to_string(),
        data: vec![0.0; 12],
        shape: vec![1, 3, 2, 2],
        num_tiles: Some(4),
    };
    assert_eq!(tensor.num_tiles, Some(4));
}

#[test]
fn pipeline_input_tensor_num_tiles_none_for_audio() {
    use crate::driver::PipelineInputTensor;

    let tensor = PipelineInputTensor {
        endpoint: "encoder.audio_features".to_string(),
        data: vec![0.0; 8],
        shape: vec![1, 2, 4],
        num_tiles: None,
    };
    assert_eq!(tensor.num_tiles, None);
}

#[tokio::test]
async fn image_load_and_preprocess_populates_num_tiles() {
    let spec = crate::image_input::VisionInputSpec::from_input(
        "encoder.pixel_values".to_string(),
        // shape [N, C, H, W] — N is the batch/tile dimension
        &[1, 3, 4, 4],
    )
    .expect("valid spec");
    let tensor = crate::image_input::load_and_preprocess(&[tiny_png_data_uri()], &spec)
        .await
        .expect("preprocess succeeds");
    // The preprocessor always produces at least one tile.
    assert!(tensor.num_tiles >= 1, "num_tiles must be at least 1");
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

#[cfg(all(feature = "native-backend", feature = "cuda"))]
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

#[cfg(all(feature = "native-backend", not(feature = "cuda")))]
#[test]
fn native_device_parser_rejects_cuda_without_cuda_feature() {
    use crate::state::parse_native_device;

    assert!(parse_native_device("cpu").is_ok());
    assert!(
        parse_native_device("cuda:0")
            .unwrap_err()
            .contains("'cuda' feature")
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
        },
        ModelSpec {
            id: "model-b".to_string(),
            path: path.clone(),
            eager: true,
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
    assert_eq!(state.registry.ids().len(), 1);
    assert_eq!(state.registry.default_id().as_deref(), Some("tiny-llm"));

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
        },
        ModelSpec {
            id: "model-b".to_string(),
            path: path.clone(),
            eager: false,
        },
    ];
    AppState::load_from_specs(specs, config).expect("load lazy two-model state")
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
    assert_eq!(state.registry.ids(), vec!["model-a"]);
    assert!(state.registry.contains_available("model-b"));

    // Routing to the lazy model triggers a load and succeeds.
    let resp = app(state.clone())
        .oneshot(chat_request("model-b"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = json_body(resp).await;
    assert_eq!(body["model"], "model-b");

    // The shared registry now has both models loaded.
    let mut ids = state.registry.ids();
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
    assert!(state.registry.resolve("model-b").is_some());

    // Subsequent routing works without re-loading.
    let resp = app(state.clone())
        .oneshot(chat_request("model-b"))
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
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
    assert!(state.registry.resolve("model-a").is_none());
    // The spec is retained for lazy reload.
    assert!(state.registry.contains_available("model-a"));

    // A subsequent request for the default (empty model) lazily reloads it.
    let resp = app(state.clone()).oneshot(chat_request("")).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert!(state.registry.resolve("model-a").is_some());
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
    assert_eq!(state.registry.ids(), vec!["model-a"]);

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
        state.registry.ids(),
        vec!["model-b"],
        "model-a should be evicted"
    );
    assert!(state.registry.contains_available("model-a"));
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
    assert!(state.registry.resolve("model-b").is_some());
    let mut ids = state.registry.ids();
    ids.sort();
    assert_eq!(ids, vec!["model-a", "model-b"]);
}

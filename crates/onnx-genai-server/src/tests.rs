use crate::{
    AppState, ChatCompletionRequest, CompletionRequest, EmbeddingEncodingFormat, EmbeddingInput,
    EmbeddingRequest, EmbeddingResponse, EmbeddingUsage, EmbeddingVector, ServerConfig, app,
    build_generate_request,
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
    let boundary = "onnx-genai-audio-boundary";
    let mut body = Vec::new();
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\ntiny-whisper\r\n\
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
async fn embeddings_validate_input_then_report_engine_api_gap() {
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

        assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body: Value = serde_json::from_slice(&body).unwrap();
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("single-pass hidden-state or pooled-output API"),
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
async fn transcription_multipart_against_non_audio_model_returns_400() {
    let (boundary, body) = multipart_audio_body("json");
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
async fn status_reports_sane_server_fields() {
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
    assert_eq!(body["status"], "ready");
    assert_eq!(body["model_id"], "tiny-llm");
    assert_eq!(body["version"], env!("CARGO_PKG_VERSION"));
    assert!(body["uptime_seconds"].is_u64());
    assert!(body["active_sessions"].is_u64());
    assert!(body["pending_queue_depth"].is_u64());
    assert!(body["total_requests"].is_u64());
    assert!(body["total_tokens"].is_u64());
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
        session_list
            .iter()
            .any(|v| v.as_str().is_some_and(|s| s.starts_with("sess-") && s.ends_with('…'))),
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
    assert!(
        trace["perfetto_export"]
            .as_str()
            .unwrap()
            .contains("not implemented")
    );
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
    ] {
        let resp = router
            .clone()
            .oneshot(
                Request::builder()
                    .uri(*path)
                    .body(Body::empty())
                    .unwrap(),
            )
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
    let after_health = prometheus_sample(
        &body,
        "onnx_genai_requests_total{endpoint=\"/health\",status=\"200\"}",
    );
    assert!(after_health >= before_health + 2);
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

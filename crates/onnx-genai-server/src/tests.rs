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
use std::{io::Cursor, path::PathBuf, time::Duration};
use tokio::{sync::mpsc, time::timeout};
use tower::ServiceExt;

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

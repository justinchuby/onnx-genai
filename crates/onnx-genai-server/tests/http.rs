use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use onnx_genai::GeneratePrompt;
use onnx_genai_engine::GenerateConstraint;
use onnx_genai_metadata::ToolProtocolDeclaration;
use onnx_genai_server::{
    AppState, ChatCompletionRequest, OrtSessionWorkerCount, ServerConfig, ToolCallStream,
    ToolParseOutcome, ToolProtocol, app, build_generate_request,
    build_generate_request_with_protocol, parse_assistant_output,
};
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

async fn test_app_with_config(config: ServerConfig) -> axum::Router {
    let state =
        AppState::load_with_config(&fixture_dir(), Some("tiny-llm".to_string()), config).unwrap();
    app(state)
}

fn diffusion_fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/comfyui_workflows/txt2img_sd15")
}

async fn image_app() -> axum::Router {
    let state =
        AppState::load(&diffusion_fixture_dir(), Some("tiny-diffusion".to_string())).unwrap();
    app(state)
}

fn copy_tree(source: &std::path::Path, destination: &std::path::Path) {
    std::fs::create_dir_all(destination).unwrap();
    for entry in std::fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        let target = destination.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            std::fs::copy(entry.path(), target).unwrap();
        }
    }
}

fn image_package_without_guidance_role() -> tempfile::TempDir {
    let package = tempfile::tempdir().unwrap();
    copy_tree(&diffusion_fixture_dir(), package.path());
    let metadata_path = package.path().join("inference_metadata.yaml");
    let metadata = std::fs::read_to_string(&metadata_path)
        .unwrap()
        .replace("\r\n", "\n");
    let runtime_role = "        role:\n          kind: runtime\n          version: '1.0'\n          role: guidance_scale\n        source:\n          kind: application\n          name: guidance_scale";
    let application_role = "        role:\n          kind: opaque\n        source:\n          kind: application\n          name: guidance_scale";
    let updated = metadata.replace(runtime_role, application_role);
    assert_ne!(
        updated, metadata,
        "fixture must contain the guidance runtime role"
    );
    std::fs::write(metadata_path, updated).unwrap();
    package
}

async fn image_admin_app() -> axum::Router {
    let state = AppState::load_with_config(
        &diffusion_fixture_dir(),
        Some("tiny-diffusion".to_string()),
        ServerConfig {
            enable_admin_endpoints: true,
            ..ServerConfig::default()
        },
    )
    .unwrap();
    app(state)
}

async fn post_json(app: axum::Router, uri: &str, body: Value) -> axum::response::Response {
    app.oneshot(
        Request::builder()
            .method("POST")
            .uri(uri)
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap(),
    )
    .await
    .unwrap()
}

async fn response_json(response: axum::response::Response) -> (StatusCode, Value) {
    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (
        status,
        serde_json::from_slice(&body)
            .unwrap_or_else(|_| panic!("non-JSON response: {}", String::from_utf8_lossy(&body))),
    )
}

fn assert_png_base64(encoded: &str) {
    let bytes = STANDARD.decode(encoded).unwrap();
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
    let decoded = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png).unwrap();
    assert_eq!((decoded.width(), decoded.height()), (4, 4));
}

async fn pending_queue_depth(app: axum::Router) -> u64 {
    let (_, body) = response_json(
        app.oneshot(
            Request::builder()
                .uri("/v1/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap(),
    )
    .await;
    body["queue_depth"].as_u64().unwrap()
}

async fn wait_for_pending_queue_depth(app: axum::Router, expected: u64) {
    for _ in 0..300 {
        if pending_queue_depth(app.clone()).await == expected {
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    assert_eq!(pending_queue_depth(app).await, expected);
}

fn sse_data_lines(text: &str) -> Vec<&str> {
    text.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .collect()
}

fn sse_json_chunks(text: &str) -> Vec<Value> {
    text.split("\n\n")
        .filter(|event| !event.lines().any(|line| line.starts_with("event: ")))
        .filter_map(|event| event.lines().find_map(|line| line.strip_prefix("data: ")))
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

async fn post_completion(app: axum::Router, body: Value) -> axum::response::Response {
    app.oneshot(
        Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap(),
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn openai_images_executes_metadata_diffusion_pipeline_and_returns_png() {
    let (status, body) = response_json(
        post_json(
            image_app().await,
            "/v1/images/generations",
            json!({
                "model": "tiny-diffusion",
                "prompt": "a tiny lighthouse",
                "n": 1,
                "response_format": "b64_json",
                "output_format": "png",
                "background": "opaque"
            }),
        )
        .await,
    )
    .await;

    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body["created"].as_u64().is_some(), "{body}");
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
    assert_png_base64(data[0]["b64_json"].as_str().unwrap());
    assert!(data[0].get("revised_prompt").is_none(), "{body}");
}

#[tokio::test]
async fn admin_warmup_executes_image_only_pipeline_output() {
    let app = image_admin_app().await;
    let (status, body) = response_json(
        app.oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/admin/models/tiny-diffusion/warm")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["id"], "tiny-diffusion");
    assert_eq!(body["warmed"], true);
}

#[tokio::test]
async fn openai_and_a1111_lower_to_equivalent_pipeline_inputs() {
    let app = image_app().await;
    let (_, openai) = response_json(
        post_json(
            app.clone(),
            "/v1/images/generations",
            json!({
                "model": "tiny-diffusion",
                "prompt": "equivalent request"
            }),
        )
        .await,
    )
    .await;
    let (status, a1111) = response_json(
        post_json(
            app,
            "/sdapi/v1/txt2img",
            json!({
                "prompt": "equivalent request",
                "negative_prompt": "",
                "seed": 20260821,
                "steps": 4,
                "cfg_scale": 7.5,
                "sampler_name": "Euler"
            }),
        )
        .await,
    )
    .await;

    assert_eq!(status, StatusCode::OK, "{a1111}");
    assert_eq!(
        openai["data"][0]["b64_json"], a1111["images"][0],
        "both schemas must bind the same semantic workflow inputs"
    );
}

#[tokio::test]
async fn a1111_seed_is_deterministic_and_reported() {
    async fn generate(app: axum::Router, seed: i64) -> Value {
        let (status, body) = response_json(
            post_json(
                app,
                "/sdapi/v1/txt2img",
                json!({
                    "prompt": "seed proof",
                    "seed": seed,
                    "steps": 4,
                    "cfg_scale": 7.5
                }),
            )
            .await,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        body
    }

    let app = image_app().await;
    let first = generate(app.clone(), 11).await;
    let again = generate(app.clone(), 11).await;
    let other = generate(app, 12).await;
    assert_eq!(first["images"][0], again["images"][0]);
    assert_ne!(first["images"][0], other["images"][0]);
    assert_eq!(first["parameters"]["seed"], 11);
    let info: Value = serde_json::from_str(first["info"].as_str().unwrap()).unwrap();
    assert_eq!(info["all_seeds"], json!([11]));
    assert_png_base64(first["images"][0].as_str().unwrap());
}

#[tokio::test]
async fn repeated_image_generations_restore_pending_metrics_to_baseline() {
    let app = image_app().await;
    wait_for_pending_queue_depth(app.clone(), 0).await;
    let baseline = 0;
    for seed in 100..103 {
        let (status, body) = response_json(
            post_json(
                app.clone(),
                "/sdapi/v1/txt2img",
                json!({
                    "prompt": "metrics lifecycle",
                    "seed": seed,
                    "steps": 1,
                    "cfg_scale": 7.5
                }),
            )
            .await,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }
    wait_for_pending_queue_depth(app, baseline).await;
}

#[tokio::test]
async fn image_apis_fail_closed_for_unbindable_or_fake_behavior() {
    let app = image_app().await;
    for (uri, request, expected) in [
        (
            "/v1/images/generations",
            json!({
                "model": "tiny-diffusion",
                "prompt": "x",
                "response_format": "url"
            }),
            "asset store",
        ),
        (
            "/v1/images/generations",
            json!({
                "model": "tiny-diffusion",
                "prompt": "x",
                "size": "1024x1024"
            }),
            "dimension roles",
        ),
        (
            "/v1/images/generations",
            json!({
                "model": "tiny-diffusion",
                "prompt": "x",
                "output_format": "webp"
            }),
            "only `output_format: png`",
        ),
        (
            "/v1/images/generations",
            json!({
                "model": "tiny-diffusion",
                "prompt": "x",
                "background": "transparent"
            }),
            "emits RGB without alpha",
        ),
        (
            "/v1/images/generations",
            json!({
                "model": "tiny-diffusion",
                "prompt": "x",
                "moderation": "auto"
            }),
            "no image moderation pipeline",
        ),
        (
            "/sdapi/v1/txt2img",
            json!({
                "prompt": "x",
                "alwayson_scripts": {"controlnet": {"args": []}}
            }),
            "never silently ignored",
        ),
        (
            "/sdapi/v1/img2img",
            json!({
                "prompt": "x",
                "init_images": ["aGVsbG8="],
                "denoising_strength": 0.5
            }),
            "no semantic image/media input",
        ),
    ] {
        let (status, body) = response_json(post_json(app.clone(), uri, request).await).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains(expected),
            "{body}"
        );
    }
}

#[tokio::test]
async fn img2img_accepts_media_sized_json_and_preserves_payload_too_large() {
    let app = image_app().await;
    let accepted_base64 = "A".repeat(3 * 1024 * 1024);
    let (status, body) = response_json(
        post_json(
            app.clone(),
            "/sdapi/v1/img2img",
            json!({
                "prompt": "large source image",
                "init_images": [accepted_base64],
                "denoising_strength": 0.5
            }),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("no semantic image/media input"),
        "{body}"
    );

    let oversized_base64 = "A".repeat(26 * 1024 * 1024);
    let (status, body) = response_json(
        post_json(
            app,
            "/sdapi/v1/img2img",
            json!({
                "prompt": "oversized source image",
                "init_images": [oversized_base64],
                "denoising_strength": 0.5
            }),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{body}");
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn txt2img_accepts_realistic_application_inputs_and_rejects_oversized_json() {
    let app = image_app().await;
    let latent = STANDARD.encode(vec![0_u8; 4 * 384 * 384 * 4]);
    let (status, body) = response_json(
        post_json(
            app.clone(),
            "/sdapi/v1/txt2img",
            json!({
                "application_inputs": {
                    "latent": {
                        "dtype": "float32",
                        "shape": [1, 4, 384, 384],
                        "data_b64": latent
                    }
                }
            }),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("not declared by workflow metadata"),
        "{body}"
    );

    let oversized = "A".repeat(26 * 1024 * 1024);
    let (status, body) = response_json(
        post_json(
            app,
            "/sdapi/v1/txt2img",
            json!({
                "application_inputs": {
                    "latent": {
                        "dtype": "float32",
                        "shape": [1],
                        "data_b64": oversized
                    }
                }
            }),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{body}");
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn a1111_application_inputs_do_not_bypass_legacy_control_roles() {
    let package = image_package_without_guidance_role();
    let state = AppState::load(package.path(), Some("application-image".to_string())).unwrap();
    let app = app(state);
    let scalar = STANDARD.encode(7.5_f32.to_le_bytes());
    let (status, body) = response_json(
        post_json(
            app,
            "/sdapi/v1/txt2img",
            json!({
                "cfg_scale": 8.0,
                "application_inputs": {
                    "guidance_scale": {
                        "dtype": "float32",
                        "shape": [1],
                        "data_b64": scalar
                    }
                }
            }),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("requested `cfg_scale` cannot be bound"),
        "{body}"
    );

    let (status, body) = response_json(
        post_json(
            image_app().await,
            "/sdapi/v1/img2img",
            json!({
                "init_images": ["not-base64"],
                "denoising_strength": 0.5,
                "application_inputs": {
                    "latent": {
                        "dtype": "float32",
                        "shape": [1, 4, 8, 8],
                        "data_b64": STANDARD.encode(vec![0_u8; 4 * 8 * 8 * 4])
                    }
                }
            }),
        )
        .await,
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("no semantic image/media input"),
        "{body}"
    );
}

#[tokio::test]
async fn a1111_discovery_reports_loaded_metadata_capabilities() {
    let app = image_app().await;
    for (uri, key, expected) in [
        ("/sdapi/v1/sd-models", "model_name", "tiny-diffusion"),
        ("/sdapi/v1/samplers", "name", "euler"),
    ] {
        let response = app
            .clone()
            .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
            .await
            .unwrap();
        let (status, body) = response_json(response).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body[0][key], expected);
    }
    let response = app
        .oneshot(
            Request::builder()
                .uri("/sdapi/v1/options")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let (status, body) = response_json(response).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["samples_format"], "png");
    assert_eq!(body["save_images"], false);
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
        max_queue_depth: 8,
        enable_debug_endpoints: false,
        ..ServerConfig::default()
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
async fn session_fork_endpoint_returns_an_independent_child() {
    let app = test_app().await;
    let source = create_http_session(app.clone()).await;
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/v1/sessions/{source}/fork"))
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(json!({ "position": 0 }).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let (status, body) = response_json(response).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["object"], "session");
    let child = body["id"].as_str().unwrap().to_string();
    assert_ne!(source, child);

    for id in [source, child] {
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn session_fork_routes_worker_local_id_collisions_without_leaking_failed_children() {
    let app = test_app_with_config(ServerConfig {
        ort_session_workers: OrtSessionWorkerCount::new(2).unwrap(),
        ..ServerConfig::default()
    })
    .await;
    let first = create_http_session(app.clone()).await;
    let second = create_http_session(app.clone()).await;

    async fn live_sessions(app: axum::Router) -> Vec<u64> {
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let (_, body) = response_json(response).await;
        body["workers"]
            .as_array()
            .unwrap()
            .iter()
            .map(|worker| worker["live_sessions"].as_u64().unwrap())
            .collect()
    }

    assert_eq!(
        live_sessions(app.clone()).await,
        vec![1, 1],
        "the two client sessions must own colliding worker-local engine ids"
    );

    let rejected = post_json(
        app.clone(),
        &format!("/v1/sessions/{first}/fork"),
        json!({ "position": 1 }),
    )
    .await;
    let (status, body) = response_json(rejected).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert!(body.get("id").is_none(), "{body}");
    assert_eq!(
        live_sessions(app.clone()).await,
        vec![1, 1],
        "failed publication must not leak a worker child or reservation"
    );

    let first_child = post_json(
        app.clone(),
        &format!("/v1/sessions/{first}/fork"),
        json!({ "position": 0 }),
    )
    .await;
    let (status, first_child) = response_json(first_child).await;
    assert_eq!(status, StatusCode::OK, "{first_child}");
    assert_eq!(
        live_sessions(app.clone()).await,
        vec![2, 1],
        "the child must stay on its source worker"
    );

    let second_child = post_json(
        app.clone(),
        &format!("/v1/sessions/{second}/fork"),
        json!({ "position": 0 }),
    )
    .await;
    let (status, second_child) = response_json(second_child).await;
    assert_eq!(status, StatusCode::OK, "{second_child}");
    assert_eq!(
        live_sessions(app.clone()).await,
        vec![2, 2],
        "matching engine-local ids must route through distinct source owners"
    );

    for id in [
        first,
        second,
        first_child["id"].as_str().unwrap().to_string(),
        second_child["id"].as_str().unwrap().to_string(),
    ] {
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
async fn session_fork_endpoint_reports_the_unsupported_participant_before_child() {
    let app = test_app().await;
    let source = create_http_session(app.clone()).await;
    let generated = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .header("X-Session-Id", &source)
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
    assert_eq!(generated.status(), StatusCode::OK);

    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(format!("/v1/sessions/{source}/fork"))
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(json!({ "position": 0 }).to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let (status, body) = response_json(response).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("runner-owned decoder KV"),
        "{body}"
    );

    let closed = app
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(format!("/v1/sessions/{source}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(closed.status(), StatusCode::NO_CONTENT);
}

#[tokio::test]
async fn sessions_are_lru_evicted_at_configured_cap() {
    let app = test_app_with_config(ServerConfig {
        max_output_tokens: 16,
        max_sessions: 2,
        max_queue_depth: 8,
        enable_debug_endpoints: false,
        ..ServerConfig::default()
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
async fn completions_returns_openai_text_completion_shape() {
    let response = post_completion(
        test_app().await,
        json!({
            "model": "tiny-llm",
            "prompt": "hello",
            "max_tokens": 1,
            "min_p": 0.05,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.2
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    assert!(json["id"].as_str().unwrap().starts_with("cmpl-"));
    assert_eq!(json["object"], "text_completion");
    assert_eq!(json["model"], "tiny-llm");
    assert_eq!(json["choices"][0]["index"], 0);
    assert!(json["choices"][0]["text"].is_string());
    assert!(json["choices"][0]["finish_reason"].is_string());
    assert!(json["choices"][0]["logprobs"].is_null());
    let prompt_tokens = json["usage"]["prompt_tokens"].as_u64().unwrap();
    let completion_tokens = json["usage"]["completion_tokens"].as_u64().unwrap();
    assert_eq!(
        json["usage"]["total_tokens"].as_u64().unwrap(),
        prompt_tokens + completion_tokens
    );
}

#[tokio::test]
async fn completions_with_suffix_rejects_model_without_fim_tokens() {
    let response = post_completion(
        test_app().await,
        json!({
            "model": "tiny-llm",
            "prompt": "fn main() {",
            "suffix": "}",
            "max_tokens": 1
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap();
    let message = json["error"]["message"].as_str().unwrap();
    assert!(
        message.contains("FIM is not supported by this model"),
        "{json}"
    );
    assert!(message.contains("recognized FIM tokens"), "{json}");
}

#[test]
fn response_format_maps_to_the_requested_generate_constraint() {
    let json_request = chat_request(json!({
        "model": "tiny-llm",
        "messages": [{"role": "user", "content": "hello"}],
        "response_format": {"type": "json_object"}
    }));
    let schema = json!({
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"]
    });
    let json_schema_request = chat_request(json!({
        "model": "tiny-llm",
        "messages": [{"role": "user", "content": "hello"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": schema, "strict": true}
        }
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
        build_generate_request(&json_schema_request)
            .options
            .constraint,
        Some(GenerateConstraint::JsonSchema(schema.to_string()))
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

#[tokio::test]
async fn chat_completions_rejects_malformed_json_schema() {
    let app = test_app().await;
    for json_schema in [
        json!({"name": "answer"}),
        json!({"name": "answer", "schema": "not an object"}),
    ] {
        let response = app
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
                            "response_format": {"type": "json_schema", "json_schema": json_schema}
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let error: Value = serde_json::from_slice(&body).unwrap();
        assert!(
            error["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("schema")),
            "{error}"
        );
    }
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
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "weather",
                "schema": {"type": "object"},
                "strict": true
            }
        }
    }));

    let Some(GenerateConstraint::Lark(grammar)) =
        build_generate_request_with_protocol(&request, &declared_protocol("tagged-json"))
            .unwrap()
            .options
            .constraint
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
        build_generate_request_with_protocol(&request, &declared_protocol("tagged-json"))
            .unwrap()
            .options
            .constraint
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
        build_generate_request_with_protocol(&auto_request, &declared_protocol("tagged-json"))
            .unwrap()
            .options
            .constraint,
        None
    );
    assert_eq!(
        build_generate_request_with_protocol(&none_request, &declared_protocol("atem-xml"))
            .unwrap()
            .options
            .constraint,
        None
    );
    let GeneratePrompt::Text(prompt) = build_generate_request(&none_request).prompt else {
        panic!("expected text prompt");
    };
    assert!(!prompt.contains("<|tools|>"), "{prompt}");
}

#[test]
fn generic_request_builder_does_not_guess_a_tool_protocol() {
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
    assert!(!prompt.contains("<|tools|>"), "{prompt}");
    assert!(!prompt.contains("<atem:tools>"), "{prompt}");
    assert!(prompt.contains("<|tool|>"), "{prompt}");
    assert!(prompt.contains("tool_call_id: call_0"), "{prompt}");
}

fn declared_protocol(identity: &str) -> ToolProtocol {
    ToolProtocol::from_declaration(&ToolProtocolDeclaration {
        identity: identity.to_string(),
        version: "v1".to_string(),
    })
    .unwrap()
}

#[tokio::test]
async fn tool_request_without_package_declaration_fails_before_generation() {
    let response = post_json(
        test_app().await,
        "/v1/chat/completions",
        json!({
            "model": "tiny-llm",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [{"type": "function", "function": {"name": "weather"}}]
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    let message = body["error"]["message"].as_str().unwrap();
    assert!(message.contains("before inference"), "{message}");
    assert!(message.contains("package.tool_protocol"), "{message}");
}

#[test]
fn declared_tagged_json_protocol_converts_multiple_calls_to_openai() {
    let protocol = declared_protocol("tagged-json");
    let parsed = parse_assistant_output(
        Some(&protocol),
        None,
        r#"<tool_call>
{"name":"read_file","arguments":{"path":"src/lib.rs"}}
</tool_call>
<tool_call>
{"name":"write_file","arguments":{"path":"src/lib.rs","content":"ok"}}
</tool_call>"#
            .to_string(),
        "stop",
    )
    .expect("valid tagged JSON call");

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

#[test]
fn declared_atem_xml_protocol_converts_escaped_values_to_openai() {
    let protocol = declared_protocol("atem-xml");
    let mut stream = ToolCallStream::default();
    let outcome = stream.push(
        protocol.parser(),
        r#"<atem:invoke name="bash">
<atem:parameter name="command">{"cmd":"printf ok"}</atem:parameter>
<atem:parameter name="description">"run a &lt; safe command"</atem:parameter>
</atem:invoke>"#,
    );
    let ToolParseOutcome::Complete(calls) = outcome else {
        panic!("expected a complete declared ATEM envelope");
    };
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "bash");
    let arguments: Value = serde_json::from_str(&calls[0].arguments).unwrap();
    assert_eq!(arguments["command"]["cmd"], "printf ok");
    assert_eq!(arguments["description"], "run a < safe command");
}

#[test]
fn declared_atem_xml_request_uses_its_adapter_across_generation_and_parse() {
    let request = chat_request(json!({
        "model": "declared-atem-package",
        "messages": [{"role": "user", "content": "run the weather tool"}],
        "tools": [{
            "type": "function",
            "function": {
                "name": "weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }
        }],
        "tool_choice": {"type": "function", "function": {"name": "weather"}},
        "response_format": {"type": "json_object"}
    }));
    let protocol = declared_protocol("atem-xml");

    let generation = build_generate_request_with_protocol(&request, &protocol)
        .expect("the declared package selects its adapter");
    assert_eq!(
        generation.options.constraint, None,
        "ATEM explicitly declares no JSON grammar rather than inheriting tagged-json"
    );
    let GeneratePrompt::Text(prompt) = generation.prompt else {
        panic!("the fallback production prompt is text");
    };
    assert!(prompt.contains("<atem:tools>"), "{prompt}");
    assert!(!prompt.contains("<|tools|>"), "{prompt}");

    let parsed = parse_assistant_output(
        Some(&protocol),
        Some(&request),
        "<atem:invoke name=\"weather\"><atem:parameter name=\"city\">\"Paris\"</atem:parameter></atem:invoke>"
            .to_string(),
        "stop",
    )
    .expect("the generated ATEM envelope must parse through the same adapter");
    assert_eq!(parsed.finish_reason, "tool_calls");
    assert_eq!(parsed.tool_calls.unwrap()[0].function.name, "weather");
}

#[test]
fn parser_returns_only_atem_user_channel_content() {
    let parsed = parse_assistant_output(
        None,
        None,
        "<|start|>assistant to=self<|message|>private reasoning<|eom|>\
         <|start|>assistant to=user<|message|>final answer<|eot|>"
            .to_string(),
        "stop",
    )
    .expect("ordinary ATEM user content");

    assert_eq!(parsed.content.as_deref(), Some("final answer"));
    assert!(parsed.tool_calls.is_none());
}

// A turn truncated inside the private reasoning channel produced no answer, so
// the client sees empty content rather than the model's reasoning.
#[test]
fn parser_withholds_atem_reasoning_without_a_user_channel() {
    let parsed = parse_assistant_output(
        None,
        None,
        "<|start|>assistant to=self<|message|>private reasoning that ran long".to_string(),
        "length",
    )
    .expect("ordinary truncated ATEM reasoning");

    assert_eq!(parsed.content.as_deref(), Some(""));
    assert!(parsed.tool_calls.is_none());
    assert_eq!(parsed.finish_reason, "length");
}

#[test]
fn declared_protocol_reports_incomplete_and_malformed_envelopes() {
    let protocol = declared_protocol("tagged-json");
    let mut stream = ToolCallStream::default();
    assert!(matches!(
        stream.push(protocol.parser(), "<tool_call>{\"name\":\"read\"}"),
        ToolParseOutcome::Incomplete
    ));
    let mut stream = ToolCallStream::default();
    assert!(matches!(
        stream.push(protocol.parser(), "<tool_call>{\"name\":}</tool_call>"),
        ToolParseOutcome::Malformed(_)
    ));
}

#[test]
fn declared_protocol_envelope_failures_are_not_assistant_content() {
    for (identity, output, failure) in [
        (
            "tagged-json",
            "<tool_call>{\"name\":\"read\"}",
            "incomplete",
        ),
        (
            "atem-xml",
            "<atem:invoke name=\"read\"><atem:parameter name=\"path\">",
            "incomplete",
        ),
        (
            "tagged-json",
            "<tool_call>{\"name\":}</tool_call>",
            "malformed",
        ),
        (
            "atem-xml",
            "<atem:invoke><atem:parameter name=\"path\">x</atem:parameter></atem:invoke>",
            "malformed",
        ),
    ] {
        let error = parse_assistant_output(
            Some(&declared_protocol(identity)),
            None,
            output.to_string(),
            "stop",
        )
        .expect_err("declared malformed or incomplete envelopes must fail closed")
        .to_string();
        assert!(error.contains(&format!("{identity}@v1")), "{error}");
        assert!(error.contains(failure), "{error}");
        assert!(error.contains("buffered generation boundary"), "{error}");
    }
}

#[test]
fn buffered_atem_route_rejects_text_outside_the_envelope_sequence() {
    let protocol = declared_protocol("atem-xml");
    let call = r#"<atem:invoke name="read"></atem:invoke>"#;
    for (output, reason) in [
        (format!("junk{call}"), "before the first invoke envelope"),
        (
            format!("{call}junk"),
            "trailing text after an invoke envelope",
        ),
        (
            format!("junk{call}junk"),
            "before the first invoke envelope",
        ),
    ] {
        let error = parse_assistant_output(Some(&protocol), None, output, "stop")
            .expect_err("text outside a declared ATEM envelope sequence must fail closed")
            .to_string();
        assert!(error.contains("atem-xml@v1"), "{error}");
        assert!(error.contains("malformed envelope"), "{error}");
        assert!(error.contains("buffered generation boundary"), "{error}");
        assert!(error.contains(reason), "{error}");
    }
}

#[test]
fn buffered_atem_route_accepts_surrounding_whitespace_and_multiple_calls() {
    let protocol = declared_protocol("atem-xml");
    let output = " \n<atem:invoke name=\"read\"></atem:invoke>\t\
                  <atem:invoke name=\"write\"><atem:parameter name=\"path\">\
                  \"src/lib.rs\"</atem:parameter></atem:invoke>\r\n"
        .to_string();
    let parsed = parse_assistant_output(Some(&protocol), None, output, "stop")
        .expect("whitespace around and between ATEM envelopes is legal");
    let calls = parsed.tool_calls.expect("two parsed tool calls");
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].function.name, "read");
    assert_eq!(calls[1].function.name, "write");
}

#[test]
fn declared_protocol_preserves_ordinary_no_call_assistant_text() {
    for identity in ["tagged-json", "atem-xml"] {
        let output = "ordinary assistant text".to_string();
        let parsed = parse_assistant_output(
            Some(&declared_protocol(identity)),
            None,
            output.clone(),
            "stop",
        )
        .expect("a non-envelope is ordinary assistant text");
        assert_eq!(parsed.content, Some(output));
        assert!(matches!(parsed.tool_parse, ToolParseOutcome::NoCall));
    }
}

#[test]
fn buffered_declared_protocol_enforces_required_and_specific_tool_choice() {
    for identity in ["tagged-json", "atem-xml"] {
        let protocol = declared_protocol(identity);
        let required = chat_request(json!({
            "model": "fixture",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [{"type": "function", "function": {"name": "weather"}}],
            "tool_choice": "required"
        }));
        let error = parse_assistant_output(
            Some(&protocol),
            Some(&required),
            "ordinary assistant text".to_string(),
            "stop",
        )
        .expect_err("required tool choice must reject a terminal no-call")
        .to_string();
        assert!(error.contains(&format!("{identity}@v1")), "{error}");
        assert!(error.contains("tool_choice required"), "{error}");
        assert!(error.contains("no tool call"), "{error}");

        let specific = chat_request(json!({
            "model": "fixture",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": [
                {"type": "function", "function": {"name": "weather"}},
                {"type": "function", "function": {"name": "calendar"}}
            ],
            "tool_choice": {"type": "function", "function": {"name": "weather"}}
        }));
        let error = parse_assistant_output(
            Some(&protocol),
            Some(&specific),
            "ordinary assistant text".to_string(),
            "stop",
        )
        .expect_err("specific tool choice must reject a terminal no-call")
        .to_string();
        assert!(error.contains(&format!("{identity}@v1")), "{error}");
        assert!(error.contains("weather"), "{error}");
        assert!(error.contains("no tool call"), "{error}");

        let mismatched = match identity {
            "tagged-json" => {
                r#"<tool_call>{"name":"calendar","arguments":{}}</tool_call>"#.to_string()
            }
            "atem-xml" => r#"<atem:invoke name="calendar"></atem:invoke>"#.to_string(),
            _ => unreachable!(),
        };
        let error = parse_assistant_output(Some(&protocol), Some(&specific), mismatched, "stop")
            .expect_err("specific tool choice must reject a different parsed function")
            .to_string();
        assert!(error.contains(&format!("{identity}@v1")), "{error}");
        assert!(error.contains("weather"), "{error}");
        assert!(error.contains("calendar"), "{error}");
    }
}

#[test]
fn buffered_declared_protocol_preserves_auto_and_none_no_call_behavior() {
    for identity in ["tagged-json", "atem-xml"] {
        for mode in ["auto", "none"] {
            let protocol = declared_protocol(identity);
            let request = chat_request(json!({
                "model": "fixture",
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": [{"type": "function", "function": {"name": "weather"}}],
                "tool_choice": mode
            }));
            let parsed = parse_assistant_output(
                Some(&protocol),
                Some(&request),
                "ordinary assistant text".to_string(),
                "stop",
            )
            .expect("auto and none permit ordinary assistant content");
            assert!(matches!(parsed.tool_parse, ToolParseOutcome::NoCall));
        }
    }
}

#[test]
fn plain_assistant_output_preserves_content() {
    let output = "ordinary assistant text".to_string();
    let parsed = parse_assistant_output(None, None, output.clone(), "stop");
    let parsed = parsed.expect("ordinary assistant content");

    assert_eq!(parsed.content, Some(output));
    assert!(parsed.tool_calls.is_none());
    assert_eq!(parsed.finish_reason, "stop");
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
                        "max_tokens": 13,
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
                        "max_tokens": 13,
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
async fn streaming_completions_returns_text_completion_chunks() {
    let response = post_completion(
        test_app().await,
        json!({
            "model": "tiny-llm",
            "prompt": "hello",
            "max_tokens": 1,
            "stream": true
        }),
    )
    .await;

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
    assert!(!chunks.is_empty(), "{text}");
    assert!(
        chunks
            .iter()
            .all(|chunk| chunk["object"] == "text_completion"),
        "{text}"
    );
    assert!(
        chunks
            .iter()
            .all(|chunk| chunk["choices"][0]["logprobs"].is_null()),
        "{text}"
    );
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

use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use onnx_genai::GeneratePrompt;
use onnx_genai_engine::GenerateConstraint;
use onnx_genai_metadata::ToolProtocolDeclaration;
use onnx_genai_server::{
    AppState, ChatCompletionRequest, OrtSessionWorkerCount, ServerConfig, ToolProtocol, app,
    build_generate_request, build_generate_request_with_protocol,
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

async fn json_test_app() -> (tempfile::TempDir, axum::Router) {
    let package = tempfile::tempdir().unwrap();
    copy_tree(&fixture_dir(), package.path());
    let path = package.path().join("tokenizer.json");
    let mut tokenizer: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let vocab = tokenizer["model"]["vocab"].as_object_mut().unwrap();
    vocab.remove("tok22");
    vocab.insert("{}".into(), json!(22));
    tokenizer["added_tokens"] = json!([{"id":3,"content":"<eos>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}]);
    std::fs::write(path, serde_json::to_vec(&tokenizer).unwrap()).unwrap();
    let router = app(AppState::load(package.path(), Some("tiny-llm".into())).unwrap());
    (package, router)
}

async fn fixture_app(name: &str) -> axum::Router {
    let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name);
    app(AppState::load(&directory, Some(name.to_string())).unwrap())
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
    let app = test_app_with_config(ServerConfig {
        engine_config: onnx_genai_engine::EngineConfig {
            decode_backend: onnx_genai_engine::EngineDecodeBackend::Ort,
            ..Default::default()
        },
        ..Default::default()
    })
    .await;
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
        engine_config: onnx_genai_engine::EngineConfig {
            decode_backend: onnx_genai_engine::EngineDecodeBackend::Ort,
            ..Default::default()
        },
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
    let app = test_app_with_config(ServerConfig {
        engine_config: onnx_genai_engine::EngineConfig {
            decode_backend: onnx_genai_engine::EngineDecodeBackend::Ort,
            ..Default::default()
        },
        ..Default::default()
    })
    .await;
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
        message.contains("suffix requires a model declaring FIM tokens"),
        "{json}"
    );
    assert!(
        message.contains("choose a FIM-capable package or omit suffix"),
        "{json}"
    );
    assert_eq!(json["error"]["type"], "invalid_request_error");
    assert_eq!(json["error"]["code"], 400);
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
        build_generate_request(&json_request, &fixture_tokenizer())
            .unwrap()
            .options
            .constraint,
        Some(GenerateConstraint::JsonSchema {
            schema: json!({"type":"object"}).to_string(),
            prefix: String::new()
        })
    );
    assert_eq!(
        build_generate_request(&json_schema_request, &fixture_tokenizer())
            .unwrap()
            .options
            .constraint,
        Some(GenerateConstraint::JsonSchema {
            schema: schema.to_string(),
            prefix: String::new()
        })
    );
    for (format, expected) in [
        (
            json!({"type":"json_schema","json_schema":{"name":"answer"}}),
            json!({"type":"object"}),
        ),
        (json!({"type":"json_schema","schema":true}), json!(true)),
    ] {
        let request = chat_request(
            json!({"messages":[{"role":"user","content":"hello"}],"response_format":format}),
        );
        assert_eq!(
            build_generate_request(&request, &fixture_tokenizer())
                .unwrap()
                .options
                .constraint,
            Some(GenerateConstraint::JsonSchema {
                schema: expected.to_string(),
                prefix: String::new()
            })
        );
    }
    assert_eq!(
        build_generate_request(&text_request, &fixture_tokenizer())
            .unwrap()
            .options
            .constraint,
        None
    );
    assert_eq!(
        build_generate_request(&absent_request, &fixture_tokenizer())
            .unwrap()
            .options
            .constraint,
        None
    );
}

#[tokio::test]
async fn chat_completions_rejects_malformed_json_schema() {
    let app = test_app().await;
    for json_schema in [
        json!({"name": "answer", "schema": []}),
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

#[cfg(feature = "native-backend")]
#[tokio::test]
async fn native_required_tool_choice_skips_an_impossible_alternative() {
    let package = tempfile::tempdir().unwrap();
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-tool-call");
    copy_tree(&source, package.path());
    let path = package.path().join("tokenizer.json");
    let mut tokenizer: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let vocab = tokenizer["model"]["vocab"].as_object_mut().unwrap();
    let previous = vocab
        .iter()
        .find(|(_, id)| id.as_u64() == Some(22))
        .unwrap()
        .0
        .clone();
    vocab.remove(&previous);
    vocab.insert(
        "<tool_call>\n{\"name\":\"weather\",\"arguments\":{\"city\":\"Paris\"}}\n</tool_call>"
            .into(),
        json!(22),
    );
    std::fs::write(path, serde_json::to_vec(&tokenizer).unwrap()).unwrap();
    std::fs::write(
        package.path().join("tokenizer_config.json"),
        r#"{"chat_template":"assistant"}"#,
    )
    .unwrap();
    let config = ServerConfig {
        engine_config: onnx_genai_engine::EngineConfig {
            decode_backend: onnx_genai_engine::EngineDecodeBackend::Native,
            ..Default::default()
        },
        ..Default::default()
    };
    let router =
        app(
            AppState::load_with_config(package.path(), Some("native-tools".into()), config)
                .unwrap(),
        );
    let response=post_json(router,"/v1/chat/completions",json!({
        "messages":[{"role":"user","content":"hello"}],"max_tokens":1,"temperature":0,"logit_bias":{"22":1000},"tool_choice":"required",
        "tools":[{"type":"function","function":{"name":"impossible","parameters":false}},
                 {"type":"function","function":{"name":"weather","parameters":{"type":"object","properties":{"city":{"const":"Paris"}},"required":["city"],"additionalProperties":false}}}]
    })).await;
    let status = response.status();
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    assert_eq!(status, StatusCode::OK, "{body}");
    let call = &body["choices"][0]["message"]["tool_calls"][0]["function"];
    assert_eq!(call["name"], "weather", "{body}");
    assert_eq!(
        serde_json::from_str::<Value>(call["arguments"].as_str().unwrap()).unwrap(),
        json!({"city":"Paris"})
    );
    assert_eq!(body["usage"]["completion_tokens"], 1);
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

    let Some(GenerateConstraint::Lark { grammar, prefix }) = build_generate_request_with_protocol(
        &request,
        &declared_protocol("tagged-json"),
        &fixture_tokenizer(),
    )
    .unwrap()
    .options
    .constraint
    else {
        panic!("expected forced tool_choice to build a Lark constraint");
    };
    assert!(prefix.is_empty());
    use onnx_genai_engine::byte_grammar::{GrammarByteCursor, GrammarCandidate};
    let cursor = GrammarByteCursor::lark(&grammar, 4096).unwrap();
    for (body, valid) in [
        (
            r#"{"name":"get_weather","arguments":{"location":"Paris"}}"#,
            true,
        ),
        (
            r#"{"arguments":{"location":"Paris"},"id":"call_1","name":"get_weather"}"#,
            true,
        ),
        (
            r#"{"name":"get_time","arguments":{"location":"Paris"}}"#,
            false,
        ),
        (r#"{"name":"get_weather","arguments":{}}"#, false),
        (
            r#"{"name":"get_weather","arguments":{"location":3}}"#,
            false,
        ),
        (
            r#"{"name":"get_weather","arguments":{"location":"Paris"},"extra":1}"#,
            false,
        ),
    ] {
        let text = format!("<tool_call>\n{body}\n</tool_call>");
        assert_eq!(
            matches!(
                cursor.preview_root(text.as_bytes()).unwrap(),
                GrammarCandidate::Viable { complete: true }
            ),
            valid,
            "{text}"
        );
    }
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

    let Some(GenerateConstraint::Lark { grammar, prefix }) = build_generate_request_with_protocol(
        &request,
        &declared_protocol("tagged-json"),
        &fixture_tokenizer(),
    )
    .unwrap()
    .options
    .constraint
    else {
        panic!("expected forced tool_choice to build a Lark constraint");
    };
    assert!(prefix.is_empty());
    use onnx_genai_engine::byte_grammar::{GrammarByteCursor, GrammarCandidate};
    let cursor = GrammarByteCursor::lark(&grammar, 4096).unwrap();
    for (body, valid) in [
        (
            r#"{"name":"get_weather","arguments":{"location":"Paris"}}"#,
            true,
        ),
        (r#"{"name":"get_time","arguments":{"zone":"UTC"}}"#, true),
        (
            r#"{"name":"get_weather","arguments":{"zone":"UTC"}}"#,
            false,
        ),
        (
            r#"{"name":"get_time","arguments":{"location":"Paris"}}"#,
            false,
        ),
    ] {
        let text = format!("<tool_call>\n{body}\n</tool_call>");
        assert_eq!(
            matches!(
                cursor.preview_root(text.as_bytes()).unwrap(),
                GrammarCandidate::Viable { complete: true }
            ),
            valid,
            "{text}"
        );
    }
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
        build_generate_request_with_protocol(
            &auto_request,
            &declared_protocol("tagged-json"),
            &fixture_tokenizer()
        )
        .unwrap()
        .options
        .constraint,
        None
    );
    assert_eq!(
        build_generate_request_with_protocol(
            &none_request,
            &declared_protocol("atem-xml"),
            &fixture_tokenizer()
        )
        .unwrap()
        .options
        .constraint,
        None
    );
    let GeneratePrompt::Text(prompt) = build_generate_request(&none_request, &fixture_tokenizer())
        .unwrap()
        .prompt
    else {
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

    let generate_request = build_generate_request(&request, &fixture_tokenizer()).unwrap();
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

#[tokio::test]
async fn tool_result_chain_commits_typed_calls_and_accepts_reordered_results_over_http() {
    let tools = json!([
        {
            "type": "function",
            "function": {
                "name": "weather",
                "parameters": {"type": "object"}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "time",
                "parameters": {"type": "object"}
            }
        }
    ]);
    let body = json!({
        "model": "tiny-tool-call",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": 2
    });
    let buffered = post_json(
        fixture_app("tiny-tool-call").await,
        "/v1/chat/completions",
        body.clone(),
    )
    .await;
    let status = buffered.status();
    let buffered_bytes = to_bytes(buffered.into_body(), usize::MAX).await.unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&buffered_bytes)
    );
    let buffered: Value = serde_json::from_slice(&buffered_bytes).unwrap();
    let choice = &buffered["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls");
    assert!(choice["message"]["content"].is_null());
    let tool_calls = choice["message"]["tool_calls"]
        .as_array()
        .expect("typed tool calls");
    assert_eq!(tool_calls.len(), 2);
    assert_eq!(tool_calls[0]["id"], "call_weather");
    assert_eq!(tool_calls[0]["function"]["name"], "weather");
    assert_eq!(
        tool_calls[0]["function"]["arguments"],
        r#"{"city":"Paris"}"#
    );
    assert_eq!(tool_calls[1]["id"], "call_time");
    assert_eq!(tool_calls[1]["function"]["name"], "time");
    assert_eq!(
        tool_calls[1]["function"]["arguments"],
        r#"{"timezone":"UTC"}"#
    );

    let mut streamed_body = body;
    streamed_body["stream"] = Value::Bool(true);
    let streamed = post_json(
        fixture_app("tiny-tool-call").await,
        "/v1/chat/completions",
        streamed_body,
    )
    .await;
    assert_eq!(streamed.status(), StatusCode::OK);
    let streamed = String::from_utf8(
        to_bytes(streamed.into_body(), usize::MAX)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    let streamed_chunks = sse_json_chunks(&streamed);
    let streamed_calls = streamed_chunks
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["delta"]["tool_calls"].as_array())
        .map(|calls| &calls[0])
        .collect::<Vec<_>>();
    assert_eq!(streamed_calls.len(), 4, "{streamed}");
    assert_eq!(streamed_calls[0]["index"], 0);
    assert_eq!(streamed_calls[0]["id"], "call_weather");
    assert_eq!(streamed_calls[0]["function"]["name"], "weather");
    assert_eq!(streamed_calls[1]["index"], 0);
    assert_eq!(
        streamed_calls[1]["function"]["arguments"],
        tool_calls[0]["function"]["arguments"]
    );
    assert_eq!(streamed_calls[2]["index"], 1);
    assert_eq!(streamed_calls[2]["id"], "call_time");
    assert_eq!(streamed_calls[2]["function"]["name"], "time");
    assert_eq!(streamed_calls[3]["index"], 1);
    assert_eq!(
        streamed_calls[3]["function"]["arguments"],
        tool_calls[1]["function"]["arguments"]
    );
    assert_eq!(
        streamed_chunks
            .iter()
            .filter(|chunk| chunk["choices"][0]["finish_reason"] == "tool_calls")
            .count(),
        1,
        "{streamed}"
    );
    assert!(!streamed.contains("<tool_call>"), "{streamed}");
    assert!(streamed.contains("data: [DONE]"), "{streamed}");

    let final_response = post_chat_json(
        fixture_app("tiny-tool-call").await,
        json!({
            "model": "tiny-tool-call",
            "messages": [
                {"role": "assistant", "content": null, "tool_calls": tool_calls},
                {
                    "role": "tool",
                    "tool_call_id": "call_time",
                    "name": "time",
                    "content": "UTC"
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_weather",
                    "name": "weather",
                    "content": "Paris"
                }
            ],
            "max_tokens": 2
        }),
    )
    .await;
    assert_eq!(final_response["choices"][0]["finish_reason"], "stop");
    assert_eq!(
        final_response["choices"][0]["message"]["content"],
        "Results accepted."
    );
    assert!(
        final_response["choices"][0]["message"]
            .get("tool_calls")
            .is_none()
    );
}

#[tokio::test]
async fn required_no_call_fails_before_buffered_success_publication() {
    let response = post_json(
        fixture_app("tiny-tool-call").await,
        "/v1/chat/completions",
        json!({
            "model": "tiny-tool-call",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "weather",
                    "parameters": {"type": "object"}
                }
            }],
            "tool_choice": "required",
            "max_tokens": 1
        }),
    )
    .await;
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body: Value =
        serde_json::from_slice(&to_bytes(response.into_body(), usize::MAX).await.unwrap()).unwrap();
    let message = body["error"]["message"].as_str().unwrap();
    assert!(message.contains("at least one was required"), "{message}");
    assert!(!body.to_string().contains("\"tool_calls\""), "{body}");
}

#[tokio::test]
async fn tool_result_history_rejections_are_actionable_before_generation() {
    let calls = json!([
        {
            "id": "call_weather",
            "type": "function",
            "function": {"name": "weather", "arguments": "{\"city\":\"Paris\"}"}
        },
        {
            "id": "call_time",
            "type": "function",
            "function": {"name": "time", "arguments": "{\"timezone\":\"UTC\"}"}
        }
    ]);
    let base = json!({
        "model": "tiny-tool-call",
        "tools": [
            {"type": "function", "function": {"name": "weather"}},
            {"type": "function", "function": {"name": "time"}}
        ],
        "tool_choice": "auto"
    });
    let cases = [
        (
            json!([
                {"role": "assistant", "content": null, "tool_calls": calls.clone()},
                {"role": "tool", "tool_call_id": "call_unknown", "content": "nope"}
            ]),
            "does not match any assistant tool call",
        ),
        (
            json!([
                {"role": "assistant", "content": null, "tool_calls": calls.clone()},
                {"role": "tool", "tool_call_id": "call_weather", "content": "Paris"},
                {"role": "tool", "tool_call_id": "call_weather", "content": "Paris"},
                {"role": "tool", "tool_call_id": "call_time", "content": "UTC"}
            ]),
            "duplicates the result already supplied",
        ),
        (
            json!([
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [
                        {
                            "id": "call_weather",
                            "type": "function",
                            "function": {"name": "weather", "arguments": "{\"city\":\"Paris\"}"}
                        },
                        {
                            "id": "call_weather",
                            "type": "function",
                            "function": {"name": "weather", "arguments": "{\"city\":\"Lyon\"}"}
                        }
                    ]
                }
            ]),
            "duplicates messages[0].tool_calls[0].id",
        ),
        (
            json!([
                {"role": "assistant", "content": null, "tool_calls": calls.clone()},
                {"role": "tool", "tool_call_id": "call_weather", "content": "Paris"}
            ]),
            "missing tool result(s) for 'call_time'",
        ),
        (
            json!([
                {"role": "assistant", "content": null, "tool_calls": calls.clone()},
                {
                    "role": "tool",
                    "tool_call_id": "call_weather",
                    "name": "time",
                    "content": "Paris"
                },
                {"role": "tool", "tool_call_id": "call_time", "content": "UTC"}
            ]),
            "does not match tool_call_id 'call_weather' function 'weather'",
        ),
        (
            json!([
                {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_weather",
                        "type": "function",
                        "function": {"name": "weather", "arguments": "not JSON"}
                    }]
                },
                {"role": "tool", "tool_call_id": "call_weather", "content": "Paris"}
            ]),
            "function.arguments must be a JSON object",
        ),
        (
            json!([
                {"role": "assistant", "content": null, "tool_calls": calls.clone()},
                {
                    "role": "tool",
                    "tool_call_id": "call_weather",
                    "content": [{"type": "text", "text": "Paris"}]
                },
                {"role": "tool", "tool_call_id": "call_time", "content": "UTC"}
            ]),
            "must be a text string",
        ),
    ];

    for (messages, expected) in cases {
        let mut body = base.clone();
        body["messages"] = messages;
        let response = post_json(
            fixture_app("tiny-tool-call").await,
            "/v1/chat/completions",
            body,
        )
        .await;
        let (status, error) = response_json(response).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{error}");
        let message = error["error"]["message"].as_str().unwrap();
        assert!(message.contains("before inference"), "{message}");
        assert!(message.contains(expected), "{message}");
    }
}

#[tokio::test]
async fn no_tools_and_none_do_not_publish_typed_tool_calls() {
    let no_tools = post_chat_json(
        fixture_app("tiny-tool-call").await,
        json!({
            "model": "tiny-tool-call",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 1
        }),
    )
    .await;
    let none = post_chat_json(
        fixture_app("tiny-tool-call").await,
        json!({
            "model": "tiny-tool-call",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "function": {"name": "weather"}}],
            "tool_choice": "none",
            "max_tokens": 1
        }),
    )
    .await;

    for response in [&no_tools, &none] {
        assert_ne!(response["choices"][0]["finish_reason"], "tool_calls");
        assert!(
            response["choices"][0]["message"]
                .get("tool_calls")
                .is_none()
        );
    }
}

#[tokio::test]
async fn chat_completions_response_format_json_object_returns_valid_json() {
    let (_package, app) = json_test_app().await;
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
                        "response_format": {"type": "json_object"}, "logit_bias":{"22":1000}
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
    assert!(
        json["usage"]["completion_tokens"].as_u64().unwrap() > 0,
        "must report actual generated JSON tokens: {json}"
    );
    assert_eq!(json["choices"][0]["message"]["role"], "assistant");
    let content = json["choices"][0]["message"]["content"].as_str().unwrap();
    let parsed: Value = serde_json::from_str(content).unwrap();
    assert!(parsed.is_object(), "{content}");
}

#[tokio::test]
async fn streaming_chat_completions_response_format_json_object_streams_valid_json() {
    let (_package, app) = json_test_app().await;
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
                        "stream": true,"stream_options":{"include_usage":true},
                        "response_format": {"type": "json_object"}, "logit_bias":{"22":1000}
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
    assert!(
        chunks
            .iter()
            .any(|chunk| chunk["usage"]["completion_tokens"]
                .as_u64()
                .is_some_and(|count| count > 0)),
        "must report actual generated JSON tokens: {text}"
    );
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
async fn ort_chat_completions_reuses_persistent_session() {
    let app = test_app_with_config(ServerConfig {
        engine_config: onnx_genai_engine::EngineConfig {
            decode_backend: onnx_genai_engine::EngineDecodeBackend::Ort,
            ..Default::default()
        },
        ..Default::default()
    })
    .await;
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

fn fixture_tokenizer() -> onnx_genai_ort::Tokenizer {
    onnx_genai_ort::Tokenizer::from_file(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm/tokenizer.json"),
    )
    .unwrap()
}

#[cfg(feature = "native-backend")]
#[tokio::test]
async fn native_http_generation_does_not_load_ort() {
    const CHILD: &str = "NXRT_NATIVE_HTTP_NO_ORT_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "native_http_generation_does_not_load_ort",
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env_remove("ONNX_GENAI_BACKEND")
            .env("ONNX_GENAI_EP", "cpu")
            .env(
                "ONNX_GENAI_ORT_LIB",
                "/missing/nxrt-native-http-test/onnxruntime",
            )
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
        return;
    }
    let config = ServerConfig::default();
    assert_eq!(
        config.engine_config.decode_backend,
        onnx_genai_engine::EngineDecodeBackend::Native
    );
    let router = test_app_with_config(config).await;
    let response = post_json(
        router.clone(),
        "/completion",
        json!({"prompt":"hi", "n_predict":1, "temperature":0}),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(body["tokens_predicted"], 1, "{body}");
    let response = post_json(router, "/v1/chat/completions", json!({"messages":[{"role":"user","content":"hi"}], "max_tokens":1, "temperature":0, "stream":true})).await;
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    let stream = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(stream.contains("chat.completion.chunk"), "{stream}");
    assert!(stream.contains("data: [DONE]"), "{stream}");
    let mut finished = false;
    for data in stream
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
    {
        if data == "[DONE]" {
            continue;
        }
        let chunk: Value = serde_json::from_str(data).unwrap();
        assert!(chunk.get("error").is_none(), "{chunk}");
        finished |= chunk["choices"][0]["finish_reason"].is_string();
    }
    assert!(finished, "{stream}");
}

#[cfg(feature = "native-backend")]
mod infill_extra_context {
    use super::*;
    const TEMPLATE: &str =
        "{% for file in input_extra %}{{ file.filename }} {{ file.text }} {% endfor %}";

    fn package(template: Option<&str>) -> tempfile::TempDir {
        let package = tempfile::tempdir().unwrap();
        copy_tree(&fixture_dir(), package.path());
        let path = package.path().join("tokenizer.json");
        let mut tokenizer: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        for (id, spelling, special) in [
            (16, "alpha", false),
            (17, "beta", false),
            (20, "<CTX_PRE>", true),
            (21, "<CTX_SUF>", true),
            (22, "<CTX_MID>", true),
        ] {
            tokenizer["model"]["vocab"]
                .as_object_mut()
                .unwrap()
                .remove(&format!("tok{id}"));
            tokenizer["model"]["vocab"][spelling] = json!(id);
            tokenizer["added_tokens"].as_array_mut().unwrap().push(json!({"id":id,"content":spelling,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":special}));
        }
        std::fs::write(path, serde_json::to_vec(&tokenizer).unwrap()).unwrap();
        let path = package.path().join("inference_metadata.yaml");
        let mut metadata: serde_yaml::Value =
            serde_yaml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        metadata["schema_version"] = serde_yaml::Value::String("1.18".into());
        metadata["pipeline"]["workflow"]["publication_mode"] =
            serde_yaml::Value::String("commit_only".into());
        metadata["pipeline"]["workflow"]["outputs"]["tokens"]["family"] =
            serde_yaml::to_value(json!({"kind":"materialized"})).unwrap();
        metadata["package"]["tokenizer"]["special_tokens"] =
            serde_yaml::to_value(json!({"bos_token_id":2,"pad_token_id":0,"eos_token_id":[3]}))
                .unwrap();
        metadata["package"]["infill"] = serde_yaml::to_value(json!({"prefix_token":"<CTX_PRE>","middle_token":"<CTX_MID>","suffix_token":"<CTX_SUF>","order":"psm","extra_context_template":template})).unwrap();
        std::fs::write(path, serde_yaml::to_string(&metadata).unwrap()).unwrap();
        let path = package.path().join("model.onnx.textproto");
        let graph = std::fs::read_to_string(&path).unwrap();
        assert_eq!(graph.matches("output: \"logits\"").count(), 1);
        let mut graph = graph.replace("output: \"logits\"", "output: \"infill_base_logits\"");
        let a = (0..32)
            .map(|id| format!("float_data: {}", if id == 4 { 20 } else { -20 }))
            .collect::<Vec<_>>()
            .join(" ");
        let b = (0..32)
            .map(|id| format!("float_data: {}", if id == 5 { 20 } else { -20 }))
            .collect::<Vec<_>>()
            .join(" ");
        let nodes = format!(
            concat!(
                "  initializer {{ name: \"ctx_padding\" data_type: 7 dims: 1 dims: 3 int64_data: 0 int64_data: 0 int64_data: 0 }}\n",
                "  node {{ input: \"input_ids\" input: \"ctx_padding\" output: \"ctx_input\" op_type: \"Concat\" attribute {{ name: \"axis\" i: 1 type: INT }} }}\n",
                "  initializer {{ name: \"ctx_first\" data_type: 7 int64_data: 1 }}\n",
                "  initializer {{ name: \"ctx_second\" data_type: 7 int64_data: 2 }}\n",
                "  initializer {{ name: \"ctx_alpha\" data_type: 7 int64_data: 16 }}\n",
                "  initializer {{ name: \"ctx_hello\" data_type: 7 int64_data: 4 }}\n",
                "  initializer {{ name: \"ctx_axes\" data_type: 7 dims: 2 int64_data: 1 int64_data: 2 }}\n",
                "  initializer {{ name: \"ctx_zero\" data_type: 1 float_data: 0 }}\n",
                "  initializer {{ name: \"ctx_good\" data_type: 1 dims: 1 dims: 1 dims: 32 {a} }}\n",
                "  initializer {{ name: \"ctx_other\" data_type: 1 dims: 1 dims: 1 dims: 32 {b} }}\n",
                "  node {{ input: \"ctx_input\" input: \"ctx_first\" output: \"ctx_name\" op_type: \"Gather\" attribute {{ name: \"axis\" i: 1 type: INT }} }}\n",
                "  node {{ input: \"ctx_input\" input: \"ctx_second\" output: \"ctx_text\" op_type: \"Gather\" attribute {{ name: \"axis\" i: 1 type: INT }} }}\n",
                "  node {{ input: \"ctx_name\" input: \"ctx_alpha\" output: \"ctx_name_ok\" op_type: \"Equal\" }}\n",
                "  node {{ input: \"ctx_text\" input: \"ctx_hello\" output: \"ctx_text_ok\" op_type: \"Equal\" }}\n",
                "  node {{ input: \"ctx_name_ok\" input: \"ctx_text_ok\" output: \"ctx_match\" op_type: \"And\" }}\n",
                "  node {{ input: \"ctx_match\" input: \"ctx_axes\" output: \"ctx_condition\" op_type: \"Unsqueeze\" }}\n",
                "  node {{ input: \"ctx_condition\" input: \"ctx_good\" input: \"ctx_other\" output: \"ctx_choice\" op_type: \"Where\" }}\n",
                "  node {{ input: \"infill_base_logits\" input: \"ctx_zero\" output: \"ctx_zeros\" op_type: \"Mul\" }}\n",
                "  node {{ input: \"ctx_zeros\" input: \"ctx_choice\" output: \"logits\" op_type: \"Add\" }}\n"
            ),
            a = a,
            b = b
        );
        let insert = graph.find("  initializer {").unwrap();
        graph.insert_str(insert, &nodes);
        std::fs::write(path, graph).unwrap();
        package
    }
    fn config() -> ServerConfig {
        ServerConfig {
            engine_config: onnx_genai_engine::EngineConfig {
                decode_backend: onnx_genai_engine::EngineDecodeBackend::Native,
                native_device: Some(onnx_genai_engine::NativeDecodeDevice::Cpu),
                ..Default::default()
            },
            ..Default::default()
        }
    }
    fn router(root: &std::path::Path, config: ServerConfig) -> axum::Router {
        app(AppState::load_with_config(root, Some("context-fim".into()), config).unwrap())
    }
    async fn body(response: axum::response::Response) -> Value {
        assert_eq!(response.status(), StatusCode::OK);
        serde_json::from_slice(&to_bytes(response.into_body(), 1024 * 1024).await.unwrap()).unwrap()
    }
    fn request(extra: Value, n: usize) -> Value {
        json!({"input_prefix":"hello","input_suffix":"world","input_extra":extra,"n_predict":n,"return_tokens":true,"cache_prompt":false,"temperature":0})
    }
    #[tokio::test]
    async fn native_infill_extra_context_drives_native_tokens_and_exact_prefill_accounting() {
        let package = package(Some(TEMPLATE));
        let router = router(package.path(), config());
        let tokenizer =
            onnx_genai_ort::Tokenizer::from_file(package.path().join("tokenizer.json")).unwrap();
        for (extra, rendered, expected) in [
            (
                json!([{"filename":"alpha","text":"hello"},{"filename":"beta","text":"world"}]),
                "alpha hello beta world ",
                4,
            ),
            (
                json!([{"filename":"beta","text":"world"},{"filename":"alpha","text":"hello"}]),
                "beta world alpha hello ",
                5,
            ),
            (
                json!([{"filename":"alpha","text":"world"}]),
                "alpha world ",
                5,
            ),
            (
                json!([{"filename":"beta","text":"hello"}]),
                "beta hello ",
                5,
            ),
            (json!([{"text":"hello"}]), "tmp hello ", 5),
            (json!([{"filename":"","text":"hello"}]), " hello ", 5),
            (json!([{"filename":"alpha","text":""}]), "alpha  ", 5),
            (
                json!([{"filename":"alpha","text":"hello","unknown":1}]),
                "alpha hello ",
                4,
            ),
            (
                json!([{"filename":"/missing","text":"hello"}]),
                "/missing hello ",
                5,
            ),
        ] {
            let prompt = format!("{rendered}<CTX_PRE>hello<CTX_SUF>world<CTX_MID>");
            let tokens = tokenizer.encode(&prompt).unwrap();
            let prefill =
                body(post_json(router.clone(), "/infill", request(extra.clone(), 0)).await).await;
            let generated =
                body(post_json(router.clone(), "/infill", request(extra, 1)).await).await;
            assert_eq!(prefill["tokens_evaluated"], tokens.len());
            assert_eq!(prefill["tokens_predicted"], 0);
            assert_eq!(generated["tokens_evaluated"], tokens.len());
            assert_eq!(generated["tokens_predicted"], 1);
            assert_eq!(generated["tokens"], json!([expected]));
            assert_eq!(
                generated["prompt"],
                tokenizer.decode_with_special_tokens(&tokens).unwrap()
            );
        }
    }
    #[tokio::test]
    async fn native_infill_extra_context_errors_precede_sse_and_keep_valid_requests_usable() {
        let package = package(Some(TEMPLATE));
        let router = router(
            package.path(),
            ServerConfig {
                max_infill_files: std::num::NonZeroUsize::new(2).unwrap(),
                max_infill_filename_bytes: std::num::NonZeroUsize::new(8).unwrap(),
                max_infill_context_bytes: std::num::NonZeroUsize::new(64).unwrap(),
                ..config()
            },
        );
        for extra in [
            json!({}),
            json!([{"filename":"alpha"}]),
            json!([{"filename":null,"text":"hello"}]),
            json!([{"filename":1,"text":"hello"}]),
            json!([{"filename":"longlabel","text":"hello"}]),
            json!([{"filename":"alpha","text":"x".repeat(80)}]),
            json!([{"filename":"alpha","text":1}]),
            json!([{"filename":"a","text":"x"},{"filename":"b","text":"x"},{"filename":"c","text":"x"}]),
        ] {
            let mut invalid = request(extra, 1);
            invalid["stream"] = json!(true);
            let response = post_json(router.clone(), "/infill", invalid).await;
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
            let valid = body(
                post_json(
                    router.clone(),
                    "/infill",
                    request(json!([{"filename":"alpha","text":"hello"}]), 1),
                )
                .await,
            )
            .await;
            assert_eq!(valid["tokens"], json!([4]));
        }
    }
    #[tokio::test]
    async fn native_infill_explicit_stop_roles_do_not_guess_file_separator_semantics() {
        for declared in [false, true] {
            let package = package(Some(TEMPLATE));
            let path = package.path().join("tokenizer.json");
            let mut tokenizer: Value =
                serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            tokenizer["model"]["vocab"]
                .as_object_mut()
                .unwrap()
                .remove("tok18");
            tokenizer["model"]["vocab"]["<|file_sep|>"] = json!(18);
            tokenizer["added_tokens"].as_array_mut().unwrap().push(json!({"id":18,"content":"<|file_sep|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}));
            std::fs::write(path, serde_json::to_vec(&tokenizer).unwrap()).unwrap();
            let path = package.path().join("model.onnx.textproto");
            let graph = std::fs::read_to_string(&path).unwrap();
            let old = graph
                .lines()
                .find(|line| line.contains("name: \"ctx_good\""))
                .unwrap();
            let logits = (0..32)
                .map(|id| format!("float_data: {}", if id == 18 { 20 } else { -20 }))
                .collect::<Vec<_>>()
                .join(" ");
            let replacement = format!(
                "  initializer {{ name: \"ctx_good\" data_type: 1 dims: 1 dims: 1 dims: 32 {logits} }}"
            );
            std::fs::write(path, graph.replace(old, &replacement)).unwrap();
            if declared {
                let path = package.path().join("inference_metadata.yaml");
                let mut metadata: serde_yaml::Value =
                    serde_yaml::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
                metadata["package"]["infill"]["additional_stop_tokens"] =
                    serde_yaml::to_value(vec!["<|file_sep|>"]).unwrap();
                std::fs::write(path, serde_yaml::to_string(&metadata).unwrap()).unwrap();
            }
            let result = body(
                post_json(
                    router(package.path(), config()),
                    "/infill",
                    request(json!([{"filename":"alpha","text":"hello"}]), 2),
                )
                .await,
            )
            .await;
            if declared {
                assert_eq!(result["tokens_predicted"], 1, "{result}");
                assert_eq!(result["stop_type"], "word", "{result}");
            } else {
                assert_eq!(result["tokens_predicted"], 2, "{result}");
                assert_eq!(result["tokens"], json!([18, 5]), "{result}");
                assert_eq!(result["stop_type"], "limit", "{result}");
            }
        }
    }
    #[tokio::test]
    async fn native_infill_extra_requires_authored_serialization_and_template_budgets() {
        let unsupported = package(None);
        let unsupported_router = router(unsupported.path(), config());
        let response = post_json(
            unsupported_router.clone(),
            "/infill",
            request(json!([{"filename":"alpha","text":"hello"}]), 1),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let valid =
            body(post_json(unsupported_router, "/infill", request(json!([]), 1)).await).await;
        assert_eq!(valid["tokens"], json!([5]));
        let fixed_header = package(Some("alpha hello "));
        let header_router = router(fixed_header.path(), config());
        for n in [0, 1] {
            let response =
                body(post_json(header_router.clone(), "/infill", request(json!([]), n)).await)
                    .await;
            let tokenizer =
                onnx_genai_ort::Tokenizer::from_file(fixed_header.path().join("tokenizer.json"))
                    .unwrap();
            let expected = tokenizer
                .encode("alpha hello <CTX_PRE>hello<CTX_SUF>world<CTX_MID>")
                .unwrap();
            assert_eq!(response["tokens_evaluated"], expected.len());
            if n == 1 {
                assert_eq!(response["tokens"], json!([4]));
            }
        }
        let authored = "{% for file in input_extra %}{% if file.filename == 'slow' %}{% for i in range(1000) %}x{% endfor %}{% else %}{{file.filename}} {{file.text}} {% endif %}{% endfor %}";
        let limited = package(Some(authored));
        let limited_router = router(
            limited.path(),
            ServerConfig {
                max_infill_template_operations: std::num::NonZeroU64::new(100).unwrap(),
                ..config()
            },
        );
        let response = post_json(
            limited_router.clone(),
            "/infill",
            request(json!([{"filename":"slow","text":"hello"}]), 1),
        )
        .await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let valid = body(
            post_json(
                limited_router,
                "/infill",
                request(json!([{"filename":"alpha","text":"hello"}]), 1),
            )
            .await,
        )
        .await;
        assert_eq!(valid["tokens"], json!([4]));
        let expanding = "{% for file in input_extra %}{% if file.filename == 'large' %}{{file.text}}{{file.text}}{{file.text}}{{file.text}}{% else %}{{file.filename}} {{file.text}} {% endif %}{% endfor %}";
        let package = package(Some(expanding));
        let bounded_router = router(
            package.path(),
            ServerConfig {
                max_infill_context_bytes: std::num::NonZeroUsize::new(16).unwrap(),
                ..config()
            },
        );
        let mut oversized = request(json!([{"filename":"large","text":"hello"}]), 1);
        oversized["stream"] = json!(true);
        let response = post_json(bounded_router.clone(), "/infill", oversized).await;
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
        let valid = body(
            post_json(
                bounded_router,
                "/infill",
                request(json!([{"filename":"alpha","text":"hello"}]), 1),
            )
            .await,
        )
        .await;
        assert_eq!(valid["tokens"], json!([4]));
    }
}

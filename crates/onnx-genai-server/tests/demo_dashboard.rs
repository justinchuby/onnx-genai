// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! Route-level tests for the demo dashboard served at `GET /demo`.
//!
//! These sit at the router level on purpose. The unit tests in `demo_assets`
//! cover directory resolution; what can only be verified here is that the
//! routes are wired, that the redirect layer is applied where it sees the bare
//! `/demo` path, and that the mount is reachable with every gate disabled.
//!
//! There is no cross-origin coverage because there are no cross-origin
//! requests: each demo server serves its own copy of the dashboard, and
//! switching scenarios navigates the browser to the other server's `/demo`, so
//! every fetch is same-origin with the server that served the page.

use axum::{
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use onnx_genai_server::{AppState, ServerConfig, app};
use std::path::{Path, PathBuf};
use tower::ServiceExt;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm")
}

/// A minimal stand-in for `examples/serving-dashboard`.
fn demo_fixture() -> tempfile::TempDir {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join("index.html"),
        "<!doctype html><script type=\"module\" src=\"app.js\"></script>",
    )
    .unwrap();
    std::fs::write(dir.path().join("app.js"), "export const ready = true;").unwrap();
    std::fs::create_dir(dir.path().join("styles")).unwrap();
    std::fs::write(dir.path().join("styles/tokens.css"), ":root{}").unwrap();
    dir
}

fn app_with_demo_dir(demo_assets_dir: Option<&Path>) -> axum::Router {
    let config = ServerConfig {
        demo_assets_dir: demo_assets_dir.map(Path::to_path_buf),
        ..Default::default()
    };
    let state =
        AppState::load_with_config(&fixture_dir(), Some("tiny-llm".to_string()), config).unwrap();
    app(state)
}

async fn body_string(response: axum::response::Response) -> String {
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    String::from_utf8(bytes.to_vec()).unwrap()
}

fn get(uri: &str) -> Request<Body> {
    Request::get(uri).body(Body::empty()).unwrap()
}

// --------------------------------------------------------------------------
// Static serving
// --------------------------------------------------------------------------

/// Without the trailing slash the browser resolves `src="app.js"` against `/`,
/// so every module 404s and the page renders blank.
#[tokio::test]
async fn demo_redirects_to_trailing_slash() {
    let dir = demo_fixture();
    let response = app_with_demo_dir(Some(dir.path()))
        .oneshot(get("/demo"))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::TEMPORARY_REDIRECT);
    assert_eq!(response.headers()[header::LOCATION], "/demo/");
}

#[tokio::test]
async fn demo_index_is_served() {
    let dir = demo_fixture();
    let response = app_with_demo_dir(Some(dir.path()))
        .oneshot(get("/demo/"))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response.headers()[header::CONTENT_TYPE].to_str().unwrap();
    assert!(content_type.starts_with("text/html"), "got: {content_type}");
    assert!(body_string(response).await.contains("<!doctype html>"));
}

/// A module served as anything other than a JavaScript type is rejected by the
/// browser's module loader, which shows up only as a console error.
#[tokio::test]
async fn demo_serves_modules_with_a_javascript_content_type() {
    let dir = demo_fixture();
    let response = app_with_demo_dir(Some(dir.path()))
        .oneshot(get("/demo/app.js"))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response.headers()[header::CONTENT_TYPE].to_str().unwrap();
    assert!(
        content_type.contains("javascript"),
        "a module served as anything else is refused by the browser's module \
         loader; got: {content_type}"
    );
    assert!(body_string(response).await.contains("export const ready"));
}

#[tokio::test]
async fn demo_serves_nested_assets() {
    let dir = demo_fixture();
    let response = app_with_demo_dir(Some(dir.path()))
        .oneshot(get("/demo/styles/tokens.css"))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let content_type = response.headers()[header::CONTENT_TYPE].to_str().unwrap();
    assert!(content_type.starts_with("text/css"), "got: {content_type}");
}

/// `ServeDir` owns traversal safety; this asserts we actually mounted it and
/// have not accidentally routed around it.
#[tokio::test]
async fn demo_refuses_path_traversal() {
    // The secret lives beside the served directory, inside our own tempdir, so
    // the test never writes a fixed name into a shared temp location.
    let root = tempfile::tempdir().expect("tempdir");
    let assets = root.path().join("assets");
    std::fs::create_dir(&assets).unwrap();
    std::fs::write(assets.join("index.html"), "<!doctype html>").unwrap();
    std::fs::create_dir(assets.join("styles")).unwrap();
    std::fs::write(root.path().join("escape-target.txt"), "secret").unwrap();

    for attempt in [
        "/demo/../escape-target.txt",
        "/demo/styles/../../escape-target.txt",
        "/demo/%2e%2e/escape-target.txt",
    ] {
        let response = app_with_demo_dir(Some(assets.as_path()))
            .oneshot(get(attempt))
            .await
            .unwrap();
        assert_ne!(
            response.status(),
            StatusCode::OK,
            "traversal must not succeed: {attempt}"
        );
        assert!(
            !body_string(response).await.contains("secret"),
            "traversal leaked file contents: {attempt}"
        );
    }
}

#[tokio::test]
async fn demo_returns_not_found_for_unknown_assets() {
    let dir = demo_fixture();
    let response = app_with_demo_dir(Some(dir.path()))
        .oneshot(get("/demo/does-not-exist.js"))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

/// Starting the server from the wrong directory is the first thing anyone will
/// do wrong, so the 404 has to name the flag and the expected location.
#[tokio::test]
async fn demo_without_configured_assets_explains_how_to_fix_it() {
    let response = app_with_demo_dir(None)
        .oneshot(get("/demo/"))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = body_string(response).await;
    assert!(body.contains("--demo-assets-dir"), "got: {body}");
    assert!(body.contains("examples/serving-dashboard"), "got: {body}");
}

/// The demo must work on a first run with no flags; gating it behind
/// `--enable-debug-endpoints` would defeat the point.
#[tokio::test]
async fn demo_is_reachable_with_debug_and_admin_endpoints_disabled() {
    let dir = demo_fixture();
    let config = ServerConfig {
        demo_assets_dir: Some(dir.path().to_path_buf()),
        enable_debug_endpoints: false,
        enable_admin_endpoints: false,
        ..Default::default()
    };
    let state =
        AppState::load_with_config(&fixture_dir(), Some("tiny-llm".to_string()), config).unwrap();

    let response = app(state).oneshot(get("/demo/")).await.unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

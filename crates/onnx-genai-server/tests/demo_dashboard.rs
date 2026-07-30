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
use onnx_genai_server::{AppState, ServerConfig, app, resolve_demo_assets_dir};
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

/// A request as a browser sends it. HTTP/1.1 requires `Host`, and the demo
/// policy is derived from it, so a test omitting it exercises the malformed
/// client path rather than the one every visitor takes.
fn get_from_host(uri: &str, host: &str) -> Request<Body> {
    Request::get(uri)
        .header(header::HOST, host)
        .body(Body::empty())
        .unwrap()
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

/// A 404 with an empty body is not a smaller error page, it is *no* error page:
/// the browser substitutes its own, in the operating system's locale, with the
/// full URL on screen. Measured in Chrome on a machine with a non-English
/// locale, that is what the audience sees. Any bytes at all take the page back,
/// so this asserts the floor across every way a `/demo` request can miss.
#[tokio::test]
async fn every_demo_not_found_carries_a_body_the_browser_will_not_replace() {
    let dir = demo_fixture();
    std::fs::write(dir.path().join("notes.md"), "internal notes").unwrap();

    let cases = [
        (
            "refused: on disk, not servable",
            Some(dir.path()),
            "/demo/notes.md",
        ),
        (
            "absent: servable extension",
            Some(dir.path()),
            "/demo/missing.js",
        ),
        (
            "absent: non-servable extension",
            Some(dir.path()),
            "/demo/missing.md",
        ),
        ("dotfile segment", Some(dir.path()), "/demo/.env"),
        (
            "nested dotfile directory",
            Some(dir.path()),
            "/demo/sub/.git/config",
        ),
        ("no asset directory configured", None, "/demo/"),
    ];

    for (label, assets, uri) in cases {
        let response = app_with_demo_dir(assets).oneshot(get(uri)).await.unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND, "{label}");
        let body = body_string(response).await;
        assert!(
            !body.is_empty(),
            "{label}: an empty 404 body hands the screen to the browser's own error page",
        );
    }

    // Positive control, deliberately after the loop: without it, a router that
    // 404s absolutely everything would satisfy every assertion above.
    let served = app_with_demo_dir(Some(dir.path()))
        .oneshot(get("/demo/app.js"))
        .await
        .unwrap();
    assert_eq!(
        served.status(),
        StatusCode::OK,
        "control: a servable asset must still be served, or the loop proves nothing",
    );
}

/// Refusing a file that exists must be indistinguishable from one that never
/// existed, or the 404 itself confirms what is on disk.
///
/// Both answers come from one constant, so the equality holds by construction.
/// This asserts it on the bytes anyway, because the previous version of this
/// property held only because BOTH bodies were empty -- giving either one a
/// body without the other would have leaked the difference as a length.
#[tokio::test]
async fn a_refused_demo_asset_is_byte_identical_to_an_absent_one() {
    let dir = demo_fixture();
    std::fs::write(dir.path().join("security-review.md"), "our own findings").unwrap();

    let refused = body_string(
        app_with_demo_dir(Some(dir.path()))
            .oneshot(get("/demo/security-review.md"))
            .await
            .unwrap(),
    )
    .await;
    let absent = body_string(
        app_with_demo_dir(Some(dir.path()))
            .oneshot(get("/demo/never-existed.md"))
            .await
            .unwrap(),
    )
    .await;

    assert_eq!(
        refused, absent,
        "a refused file is distinguishable from an absent one, which confirms it is on disk",
    );
    assert!(
        !refused.is_empty(),
        "two empty bodies satisfy the equality above vacuously",
    );
    assert!(
        !refused.contains("security-review"),
        "the refusal named the file it refused: {refused}",
    );

    // Both requests above carry a non-servable extension, so both are answered
    // by the refusal layer and agree by construction. The drift that actually
    // costs us is between the TWO mechanisms: a servable extension with no file
    // behind it is answered by `ServeDir`, which has its own not-found service.
    // Without this, removing that service leaves this test green.
    let from_serve_dir = body_string(
        app_with_demo_dir(Some(dir.path()))
            .oneshot(get("/demo/never-existed.js"))
            .await
            .unwrap(),
    )
    .await;
    assert_eq!(
        from_serve_dir, refused,
        "ServeDir's not-found body drifted from the refusal body; the two mechanisms \
         answering the same question must answer it with the same bytes",
    );
}

/// The error body is the surface most likely to be read by a stranger, and the
/// one we just finished clearing of filesystem paths. It must not reintroduce a
/// disclosure by quoting the request back, and must not name the asset
/// directory it searched.
#[tokio::test]
async fn no_demo_error_body_echoes_the_request_or_an_absolute_path() {
    let dir = demo_fixture();
    let absolute = dir.path().to_str().expect("utf-8 tempdir").to_string();

    for assets in [Some(dir.path()), None] {
        let response = app_with_demo_dir(assets)
            .oneshot(get("/demo/UNIQUEPATHMARKER.md?token=SECRETQUERYMARKER"))
            .await
            .unwrap();
        let body = body_string(response).await;

        assert!(
            !body.contains("UNIQUEPATHMARKER"),
            "the body echoed the requested path: {body}",
        );
        assert!(
            !body.contains("SECRETQUERYMARKER"),
            "the body echoed the query string: {body}",
        );
        assert!(
            !body.contains(&absolute),
            "the body disclosed the absolute asset directory: {body}",
        );
        // Absolute FILESYSTEM roots, not slashes. `/demo/` is a URL path and
        // must stay quotable -- banning `/` outright is the false positive that
        // makes a guard get loosened by whoever trips it.
        for root in ["/Users/", "/home/", "/private/", "/var/folders/", "C:\\"] {
            assert!(
                !body.contains(root),
                "the body contains an absolute filesystem path ({root}): {body}",
            );
        }
    }
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

// --------------------------------------------------------------------------
// What the demo origin publishes
// --------------------------------------------------------------------------

/// The asset directory is a working source tree, not a build output.
///
/// `examples/serving-dashboard` held 101 files when this was written: 15
/// markdown documents including an unpublished architecture and security
/// review, two shell scripts, three Python scripts, and 47 test files. All of
/// them answered 200 on the live demo origin, verified with curl against a
/// running server rather than inferred.
///
/// `ServeDir` is not at fault: those files really are inside the directory it
/// was pointed at. The mistaken assumption was that "where the assets live" and
/// "what the demo should publish" name the same set.
#[tokio::test]
async fn the_demo_origin_does_not_publish_the_source_tree() {
    let dir = demo_fixture();
    std::fs::write(dir.path().join("SECURITY-REVIEW.md"), "# internal findings").unwrap();
    std::fs::write(dir.path().join("run-demo.sh"), "#!/usr/bin/env bash").unwrap();
    std::fs::write(dir.path().join("format.test.js"), "// tests").unwrap();

    for path in [
        "/demo/SECURITY-REVIEW.md",
        "/demo/run-demo.sh",
        "/demo/format.test.js",
    ] {
        let response = app_with_demo_dir(Some(dir.path()))
            .oneshot(get(path))
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::NOT_FOUND,
            "{path} is readable on the demo origin"
        );
        // A body would confirm the file exists; the refusal must be
        // indistinguishable from absence.
        assert!(
            !body_string(response).await.contains("internal findings"),
            "{path} leaked its contents in the refusal"
        );
    }
}

/// The half that matters more. A restriction that blanks the dashboard on
/// stage is worse than the disclosure it prevents, and the unit tests check the
/// predicate rather than the mount -- which is exactly where a working
/// predicate and a broken page can coexist.
#[tokio::test]
async fn the_demo_page_still_loads_every_asset_it_needs() {
    let dir = demo_fixture();
    for path in [
        "/demo/",
        "/demo/index.html",
        "/demo/app.js",
        "/demo/styles/tokens.css",
    ] {
        let response = app_with_demo_dir(Some(dir.path()))
            .oneshot(get(path))
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "{path} is loaded by the page and stopped being served"
        );
    }
}

/// `connect-src` pinned to loopback is a browser-layer mitigation for the
/// origin-injection defect: the dashboard reads its server origins from a query
/// parameter, so a crafted link can aim the page at a third party and render
/// fabricated numbers under our own provenance badges. With this header the
/// parser AND the browser must both be defeated, not just the parser.
#[tokio::test]
async fn the_demo_page_carries_a_policy_that_confines_it_to_loopback() {
    let dir = demo_fixture();
    let response = app_with_demo_dir(Some(dir.path()))
        .oneshot(get_from_host("/demo/", "127.0.0.1:8123"))
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::OK);
    let policy = response.headers()[header::CONTENT_SECURITY_POLICY]
        .to_str()
        .unwrap()
        .to_string();
    assert!(
        policy.contains("connect-src 'self' http://127.0.0.1:* https://127.0.0.1:*"),
        "got: {policy}"
    );
    assert!(policy.contains("frame-ancestors 'none'"), "got: {policy}");
    assert_eq!(
        response.headers()[header::X_CONTENT_TYPE_OPTIONS],
        "nosniff"
    );
}

/// The port wildcard is the real topology, not laxity: the demo runs two
/// servers on one host and the page served from one legitimately polls the
/// other. CSP treats a differing port as a differing origin, so a bare `'self'`
/// would break the demo rather than secure it.
#[tokio::test]
async fn the_demo_policy_permits_the_second_server_of_the_two_server_topology() {
    let dir = demo_fixture();
    let response = app_with_demo_dir(Some(dir.path()))
        .oneshot(get_from_host("/demo/index.html", "127.0.0.1:8123"))
        .await
        .unwrap();
    let policy = response.headers()[header::CONTENT_SECURITY_POLICY]
        .to_str()
        .unwrap();

    assert!(
        policy.contains("http://127.0.0.1:*"),
        "a port-specific connect-src would block the sibling server the \
         dashboard polls: got {policy}"
    );
    assert!(
        !policy.contains(":8123"),
        "the policy must not pin the page's own port: got {policy}"
    );
}

/// A `Host` that is not a bare hostname must not widen the policy, and must not
/// be able to append a directive. The result is stricter than the browser case,
/// which is the correct direction to fail: the only clients that reach it are
/// ones that omitted a header HTTP/1.1 requires.
#[tokio::test]
async fn a_crafted_host_header_does_not_widen_the_demo_policy() {
    let dir = demo_fixture();
    let response = app_with_demo_dir(Some(dir.path()))
        .oneshot(get_from_host("/demo/", "evil.example.com"))
        .await
        .unwrap();
    let policy = response.headers()[header::CONTENT_SECURITY_POLICY]
        .to_str()
        .unwrap();

    assert!(
        policy.contains("http://evil.example.com:*"),
        "a well-formed host is honoured -- it only ever widens to the origin \
         family the visitor already navigated to: got {policy}"
    );
    assert!(
        !policy.contains("script-src *"),
        "the policy must never gain a directive from a header: got {policy}"
    );
}

/// Reproduces a measurement: before the dotfile rule, `.secret.json` and
/// `.vscode/settings.json` answered `200` on the demo origin WITH THEIR
/// CONTENTS, because the extension allowlist judges the extension and a dotfile
/// can carry an allowed one. `.env` and `.git/config` were refused only because
/// `env` and `config` happen not to be on the list.
///
/// Bodies are inspected rather than statuses, because the point of the defect
/// was the content that came back.
#[tokio::test]
async fn the_demo_origin_does_not_publish_dotfiles() {
    let dir = demo_fixture();
    let secret = "SUPER-SECRET-VALUE";
    std::fs::write(dir.path().join(".env"), format!("KEY={secret}")).unwrap();
    std::fs::write(
        dir.path().join(".secret.json"),
        format!("{{\"k\":\"{secret}\"}}"),
    )
    .unwrap();
    std::fs::create_dir(dir.path().join(".vscode")).unwrap();
    std::fs::write(
        dir.path().join(".vscode/settings.json"),
        format!("{{\"token\":\"{secret}\"}}"),
    )
    .unwrap();
    std::fs::create_dir(dir.path().join(".git")).unwrap();
    std::fs::write(dir.path().join(".git/config"), format!("url={secret}")).unwrap();

    for path in [
        "/demo/.env",
        "/demo/.secret.json",
        "/demo/.vscode/settings.json",
        "/demo/.git/config",
    ] {
        let response = app_with_demo_dir(Some(dir.path()))
            .oneshot(get(path))
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::NOT_FOUND,
            "{path} was served"
        );
        let body = body_string(response).await;
        assert!(
            !body.contains(secret),
            "{path} returned its contents on an unauthenticated origin"
        );
    }
}

/// A directory that exists but cannot serve the dashboard must not be mounted:
/// `--demo-assets-dir ~` would otherwise publish a home directory.
#[tokio::test]
async fn a_directory_without_the_entry_page_is_not_published() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("private.json"), "{\"pin\":\"1234\"}").unwrap();

    let resolved = resolve_demo_assets_dir(Some(dir.path().to_path_buf()));
    assert!(resolved.is_none(), "an arbitrary directory was accepted");

    // What a server started with that flag actually answers.
    let response = app_with_demo_dir(resolved.as_deref())
        .oneshot(get("/demo/private.json"))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = body_string(response).await;
    assert!(!body.contains("1234"), "the directory was published anyway");
    assert!(
        body.contains("index.html"),
        "the refusal must say what was required, not just 404: got {body:?}"
    );
}

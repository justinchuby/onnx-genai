// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! Static file serving for the demo dashboard at `GET /demo`.
//!
//! The files themselves are served by [`tower_http::services::ServeDir`], which
//! owns path-traversal safety, content-type detection and conditional requests.
//! This module supplies only the parts `ServeDir` does not:
//!
//! * locating the asset directory,
//! * an actionable response when it was not found,
//! * the `/demo` -> `/demo/` redirect.
//!
//! Serving the demo from the server itself is load-bearing rather than a
//! convenience: it makes the dashboard same-origin with the API, so a visitor
//! can open a plain running server and see the demo with no proxy, no build
//! step, and no cross-origin configuration.

use std::path::PathBuf;

use axum::{
    body::Body,
    http::{Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Redirect, Response},
};

/// Directory searched for demo assets when no explicit path is configured,
/// resolved relative to the current working directory.
const DEFAULT_DEMO_ASSETS_DIR: &str = "examples/serving-dashboard";

/// Shown when `/demo` is requested but no asset directory was found.
///
/// Names the flag, the environment variable AND the default location: starting
/// the server from the wrong working directory is the most likely first-run
/// mistake, and a bare `404` gives the visitor nothing to act on.
const MISSING_ASSETS_MESSAGE: &str = concat!(
    "The demo dashboard assets were not found.\n\n",
    "The server looks for them in this order:\n",
    "  1. --demo-assets-dir <DIR>\n",
    "  2. ONNX_GENAI_DEMO_ASSETS_DIR=<DIR>\n",
    "  3. ./examples/serving-dashboard (relative to the working directory)\n\n",
    "Either start the server from the repository root, or pass the directory \
     explicitly:\n",
    "  onnx-genai-server --model <MODEL_DIR> --demo-assets-dir examples/serving-dashboard\n",
);

/// Resolves the demo asset directory from an explicit override, the environment,
/// then a working-directory-relative default.
///
/// Returns `None` when nothing exists, so a server started outside the repo
/// still boots — `/demo` then reports the problem instead of the process
/// refusing to start over an optional feature.
pub fn resolve_demo_assets_dir(explicit: Option<PathBuf>) -> Option<PathBuf> {
    let candidate = explicit
        .or_else(|| std::env::var_os("ONNX_GENAI_DEMO_ASSETS_DIR").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from(DEFAULT_DEMO_ASSETS_DIR));
    candidate.is_dir().then_some(candidate)
}

/// Fallback for every `/demo` route when no asset directory was configured.
pub(crate) async fn missing_assets() -> Response {
    (StatusCode::NOT_FOUND, MISSING_ASSETS_MESSAGE).into_response()
}

/// Redirects exactly `/demo` to `/demo/`.
///
/// The trailing slash is not cosmetic. Without it the browser resolves the
/// page's relative `<script type="module" src="app.js">` against `/`, requests
/// `/app.js`, and every module 404s — presenting as a blank page whose only
/// symptom is a console error.
///
/// Implemented as a router-level layer rather than a route because `ServeDir` is
/// mounted with `nest_service("/demo", ..)`, which already claims the bare
/// `/demo` path; a competing route there would panic at startup.
///
/// Deliberately temporary rather than permanent: a permanent redirect is cached
/// by the browser and would outlive any change to this mount point.
pub(crate) async fn redirect_bare_demo(request: Request<Body>, next: Next) -> Response {
    if request.uri().path() == "/demo" {
        return Redirect::temporary("/demo/").into_response();
    }
    next.run(request).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_directory_resolves_to_none_rather_than_failing() {
        assert!(resolve_demo_assets_dir(Some(PathBuf::from("/nonexistent/demo/dir"))).is_none());
    }

    #[test]
    fn explicit_directory_is_used_when_it_exists() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert_eq!(
            resolve_demo_assets_dir(Some(dir.path().to_path_buf())).as_deref(),
            Some(dir.path())
        );
    }

    /// A file is not a directory: pointing `--demo-assets-dir` at `index.html`
    /// must fall through to the actionable message rather than half-working.
    #[test]
    fn a_file_is_not_an_asset_directory() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("index.html");
        std::fs::write(&file, "<!doctype html>").unwrap();
        assert!(resolve_demo_assets_dir(Some(file)).is_none());
    }

    /// The 404 is the first thing anyone hits when they start the server from
    /// the wrong directory, so it has to carry the fix.
    #[test]
    fn missing_assets_message_names_the_flag_and_the_default_path() {
        assert!(MISSING_ASSETS_MESSAGE.contains("--demo-assets-dir"));
        assert!(MISSING_ASSETS_MESSAGE.contains("ONNX_GENAI_DEMO_ASSETS_DIR"));
        assert!(MISSING_ASSETS_MESSAGE.contains(DEFAULT_DEMO_ASSETS_DIR));
    }
}

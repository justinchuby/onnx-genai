//! Exact-path static hosting of the pinned upstream Llama Chat build.
//! No filesystem access or network download occurs when a server starts.
use axum::{
    http::{Uri, header},
    response::{IntoResponse, Response},
};
include!(concat!(env!("OUT_DIR"), "/llama_chat_assets.rs"));

fn asset_path(path: &str) -> &str {
    let path = path.strip_prefix('/').unwrap_or(path);
    if path.is_empty() { "index.html" } else { path }
}
pub(crate) fn is_asset(path: &str) -> bool {
    asset(asset_path(path)).is_some()
}

pub(crate) async fn serve(uri: Uri) -> Response {
    let path = asset_path(uri.path());
    let Some(body) = asset(path) else {
        return crate::routes::ApiError::not_found(format!(
            "path '{}' is not served; open / for Llama Chat or /v1/models for model discovery",
            uri.path()
        ))
        .into_response();
    };
    let content_type = match path.rsplit('.').next().unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" => "application/json",
        "webmanifest" => "application/manifest+json",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "ico" => "image/x-icon",
        "woff" => "font/woff",
        "woff2" => "font/woff2",
        "ttf" => "font/ttf",
        "wasm" => "application/wasm",
        "txt" => "text/plain; charset=utf-8",
        _ => "application/octet-stream",
    };
    let cache = if path.starts_with("_app/immutable/") {
        "public, max-age=31536000, immutable"
    } else {
        "no-cache"
    };
    let mut response = (
        [
            (header::CONTENT_TYPE, content_type),
            (header::CACHE_CONTROL, cache),
        ],
        body,
    )
        .into_response();
    if path == "index.html" {
        response
            .headers_mut()
            .insert("cross-origin-opener-policy", "same-origin".parse().unwrap());
        response.headers_mut().insert(
            "cross-origin-embedder-policy",
            "require-corp".parse().unwrap(),
        );
    }
    response
}

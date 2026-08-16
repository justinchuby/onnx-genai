//! Shared ORT session-creation helper for the plugin-EP test harnesses.
//!
//! Fixtures are committed either as binary `model.onnx` or as git-friendly
//! `model.onnx.textproto` (ONNX protobuf TextFormat). ORT's on-disk model
//! parser only understands the binary protobuf wire format, so a textproto
//! fixture is parsed to binary in-memory (`onnx_std::textproto::to_binary`) and
//! loaded through `CreateSessionFromArray`. Binary fixtures continue to load
//! directly through `CreateSession` (preserving model-directory context for
//! any external-data models). This mirrors the production seam in
//! `onnx-genai-ort`'s `Session::new`.

use std::path::Path;

use onnx_genai_ort_sys as ort;

/// Returns true if `path` is an ONNX TextFormat fixture (`*.textproto`).
pub fn is_textproto_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("textproto"))
        .unwrap_or(false)
}

/// Create an ORT session from a fixture path, dispatching to the from-bytes API
/// for `*.textproto` fixtures and the from-path API for binary `*.onnx` files.
///
/// Returns the raw `OrtStatus` (null on success) exactly like the underlying
/// `CreateSession*` call, so callers keep using their existing `check_status`.
///
/// # Safety
///
/// `api`, `env`, and `options` must be valid ORT handles and `out_session` a
/// valid out-pointer, matching the raw ORT `CreateSession*` contract.
pub unsafe fn create_session(
    api: *const ort::OrtApi,
    env: *mut ort::OrtEnv,
    options: *mut ort::OrtSessionOptions,
    model_path: impl AsRef<Path>,
    out_session: *mut *mut ort::OrtSession,
) -> *mut ort::OrtStatus {
    let model_path = model_path.as_ref();
    if is_textproto_path(model_path) {
        let text = std::fs::read_to_string(model_path)
            .unwrap_or_else(|e| panic!("read textproto fixture {model_path:?}: {e}"));
        let bytes = onnx_std::textproto::to_binary(&text).unwrap_or_else(|e| {
            panic!("convert textproto fixture {model_path:?} to binary protobuf: {e}")
        });
        // SAFETY: caller guarantees `api` is a valid `OrtApi`; `bytes` outlives
        // the synchronous call and ORT copies/parses the model before returning.
        unsafe {
            let create = (*api)
                .CreateSessionFromArray
                .expect("OrtApi::CreateSessionFromArray unavailable");
            create(
                env,
                bytes.as_ptr() as *const std::ffi::c_void,
                bytes.len(),
                options,
                out_session,
            )
        }
    } else {
        let model_c = crate::ort_path::OrtPathBuf::new(model_path);
        // SAFETY: caller guarantees `api` is a valid `OrtApi`; `model_c` outlives
        // the synchronous call.
        unsafe {
            let create = (*api)
                .CreateSession
                .expect("OrtApi::CreateSession unavailable");
            create(env, model_c.as_ptr(), options, out_session)
        }
    }
}

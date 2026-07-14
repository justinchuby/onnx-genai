//! # `onnx-runtime-capi`
//!
//! The C ABI layer for the ORT 2.0 runtime — ultimately a drop-in
//! `libonnxruntime.so` (see `docs/ORT2.md` §21). Phase 1 targets the Tier 1
//! surface: `OrtGetApiBase` + `CreateSession` + `Run`.
//!
//! **Skeleton:** this crate is intentionally near-empty for now. It defines the
//! ORT-compatible status codes (§22) and a version string; the exported
//! `extern "C"` entry points and their `unsafe` marshalling land in the Phase 1
//! task `ort2-capi` (and continue through Phase 2).

use std::ffi::c_char;

/// ORT-compatible status code for the C API layer (`docs/ORT2.md` §22).
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OrtErrorCode {
    Ok = 0,
    Fail = 1,
    InvalidArgument = 2,
    NoSuchFile = 3,
    NoModel = 4,
    EngineMismatch = 5,
    InvalidProtobuf = 6,
    ModelLoaded = 7,
    NotImplemented = 8,
    InvalidGraph = 10,
    EpFail = 11,
}

/// The runtime version string, NUL-terminated for C consumers.
const VERSION: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();

/// Return the runtime version as a C string (`ORT`'s `GetVersionString`).
///
/// # Safety
/// The returned pointer is valid for the lifetime of the process and must not
/// be freed by the caller.
#[unsafe(no_mangle)]
pub extern "C" fn ort2_get_version_string() -> *const c_char {
    VERSION.as_ptr() as *const c_char
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_nul_terminated() {
        assert_eq!(*VERSION.last().unwrap(), 0);
        let ptr = ort2_get_version_string();
        assert!(!ptr.is_null());
    }

    #[test]
    fn status_codes_match_ort() {
        assert_eq!(OrtErrorCode::Ok as i32, 0);
        assert_eq!(OrtErrorCode::InvalidGraph as i32, 10);
        assert_eq!(OrtErrorCode::EpFail as i32, 11);
    }
}

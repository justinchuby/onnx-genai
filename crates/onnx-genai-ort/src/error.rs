//! ORT error handling.

use std::ffi::CStr;

#[derive(Debug, thiserror::Error)]
pub enum OrtError {
    #[error("ORT error: {message} (code: {code})")]
    Runtime { code: i32, message: String },
    #[error("Null pointer returned from ORT API")]
    NullPointer,
    #[error("ORT API function unavailable: {0}")]
    ApiUnavailable(&'static str),
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
    #[error("Session creation failed: {0}")]
    SessionCreation(String),
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, OrtError>;

pub(crate) fn api() -> Result<&'static onnx_genai_ort_sys::OrtApi> {
    // SAFETY: OrtGetApiBase is ORT's process-wide C API entry point. The returned
    // API table has static lifetime per ORT documentation and is never freed by us.
    unsafe {
        let base = onnx_genai_ort_sys::OrtGetApiBase();
        if base.is_null() {
            return Err(OrtError::NullPointer);
        }
        let get_api = (*base).GetApi.ok_or(OrtError::ApiUnavailable("GetApi"))?;
        let api = get_api(onnx_genai_ort_sys::ORT_API_VERSION);
        if api.is_null() {
            return Err(OrtError::NullPointer);
        }
        Ok(&*api)
    }
}

pub(crate) fn check_status(status: onnx_genai_ort_sys::OrtStatusPtr) -> Result<()> {
    if status.is_null() {
        return Ok(());
    }

    // SAFETY: A non-null OrtStatusPtr is owned by the caller and must be released
    // with ReleaseStatus after querying its immutable code/message fields.
    unsafe {
        let api = api()?;
        let get_code = api
            .GetErrorCode
            .ok_or(OrtError::ApiUnavailable("GetErrorCode"))?;
        let get_message = api
            .GetErrorMessage
            .ok_or(OrtError::ApiUnavailable("GetErrorMessage"))?;
        let code = get_code(status) as i32;
        let message_ptr = get_message(status);
        let message = if message_ptr.is_null() {
            "<no ORT error message>".to_string()
        } else {
            CStr::from_ptr(message_ptr).to_string_lossy().into_owned()
        };
        if let Some(release) = api.ReleaseStatus {
            release(status);
        }
        Err(OrtError::Runtime { code, message })
    }
}

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
    #[error("{0}")]
    RuntimeLibrary(String),
    #[error("{0}")]
    ApiVersionMismatch(String),
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
    #[error("Session creation failed: {0}")]
    SessionCreation(String),
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),
    #[cfg(feature = "cuda")]
    #[error("CUDA error: {0}")]
    Cuda(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    ThreadAffinity(#[from] crate::thread_affinity::ThreadAffinityError),
}

pub type Result<T> = std::result::Result<T, OrtError>;

pub(crate) fn api() -> Result<&'static onnx_genai_ort_sys::OrtApi> {
    // SAFETY: OrtGetApiBase is ORT's process-wide C API entry point. The returned
    // API table has static lifetime per ORT documentation and is never freed by us.
    unsafe {
        let base = onnx_genai_ort_sys::OrtGetApiBase();
        if base.is_null() {
            return Err(OrtError::RuntimeLibrary(
                onnx_genai_ort_sys::ort_load_error().unwrap_or_else(|| {
                    "Failed to load ONNX Runtime: OrtGetApiBase returned null".to_owned()
                }),
            ));
        }
        let get_api = (*base).GetApi.ok_or(OrtError::ApiUnavailable("GetApi"))?;
        let api = get_api(onnx_genai_ort_sys::ORT_API_VERSION);
        if api.is_null() {
            let loaded_path = onnx_genai_ort_sys::loaded_ort_path()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "<unknown path>".to_owned());
            let loaded_version =
                onnx_genai_ort_sys::loaded_ort_version().unwrap_or_else(|| "unknown".to_owned());
            let loaded_api = onnx_genai_ort_sys::loaded_ort_api_version()
                .map_or_else(|| "unknown".to_owned(), |version| version.to_string());
            let reason = onnx_genai_ort_sys::loaded_ort_reason()
                .unwrap_or_else(|| "dynamic loader search path".to_owned());
            return Err(OrtError::ApiVersionMismatch(format!(
                "ONNX Runtime API version mismatch: loaded {loaded_path} ({reason}), \
                 ORT version {loaded_version}, API {loaded_api}; onnx-genai was built \
                 against API {} (ORT 1.29.x).\n\
                 Fix: set ONNX_GENAI_ORT_LIB to the full path of the ORT 1.29 library \
                 (for example a conda env's onnxruntime.dll), set ONNX_GENAI_ORT_LIB_DIR \
                 to its containing directory, activate that conda env and put its library \
                 directory first on PATH/LD_LIBRARY_PATH/DYLD_LIBRARY_PATH, or rebuild so \
                 ort-sys downloads ORT 1.29.0.",
                onnx_genai_ort_sys::ORT_API_VERSION
            )));
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

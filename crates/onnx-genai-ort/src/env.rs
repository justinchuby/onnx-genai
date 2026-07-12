//! ORT Environment (global singleton).

use std::ffi::CString;
use std::ptr::NonNull;

use crate::{OrtError, Result};

/// ORT Environment — must be created before any sessions.
/// Typically one per process.
pub struct Environment {
    ptr: NonNull<onnx_genai_ort_sys::OrtEnv>,
    _name: String,
}

impl Environment {
    /// Create a new ORT environment.
    pub fn new(name: &str) -> Result<Self> {
        let log_id = CString::new(name)
            .map_err(|_| OrtError::InvalidArgument("environment name contains NUL".into()))?;
        let mut ptr = std::ptr::null_mut();
        let api = crate::error::api()?;
        let create = api.CreateEnv.ok_or(OrtError::ApiUnavailable("CreateEnv"))?;
        // SAFETY: `log_id` is NUL-terminated and lives for the call; `ptr` is a
        // valid out-parameter. This wrapper owns the returned environment handle.
        crate::error::check_status(unsafe {
            create(
                onnx_genai_ort_sys::ORT_LOGGING_LEVEL_WARNING,
                log_id.as_ptr(),
                &mut ptr,
            )
        })?;
        tracing::info!("Creating ORT environment: {}", name);
        Ok(Self {
            ptr: NonNull::new(ptr).ok_or(OrtError::NullPointer)?,
            _name: name.to_string(),
        })
    }

    pub(crate) fn as_ptr(&self) -> *const onnx_genai_ort_sys::OrtEnv {
        self.ptr.as_ptr()
    }
}

impl Drop for Environment {
    fn drop(&mut self) {
        if let Ok(api) = crate::error::api()
            && let Some(release) = api.ReleaseEnv
        {
            // SAFETY: `ptr` is owned by this wrapper and released exactly once here.
            unsafe { release(self.ptr.as_ptr()) };
        }
    }
}

// Safety: OrtEnv is thread-safe per ORT documentation
unsafe impl Send for Environment {}
unsafe impl Sync for Environment {}

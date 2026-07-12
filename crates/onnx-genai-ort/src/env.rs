//! ORT Environment (global singleton).

use crate::Result;

/// ORT Environment — must be created before any sessions.
/// Typically one per process.
pub struct Environment {
    // ptr: *mut ort_sys::OrtEnv,  // TODO: actual pointer when bindgen is wired
    _name: String,
}

impl Environment {
    /// Create a new ORT environment.
    pub fn new(name: &str) -> Result<Self> {
        // TODO: Call OrtCreateEnv via C API
        tracing::info!("Creating ORT environment: {}", name);
        Ok(Self {
            _name: name.to_string(),
        })
    }
}

impl Drop for Environment {
    fn drop(&mut self) {
        // TODO: Call OrtReleaseEnv
    }
}

// Safety: OrtEnv is thread-safe per ORT documentation
unsafe impl Send for Environment {}
unsafe impl Sync for Environment {}

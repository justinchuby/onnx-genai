//! ORT Environment (global singleton).

use std::ffi::CString;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use std::sync::Mutex;

use crate::{OrtError, Result};

/// ORT Environment — must be created before any sessions.
/// Typically one per process.
pub struct Environment {
    ptr: NonNull<onnx_genai_ort_sys::OrtEnv>,
    _name: String,
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    registered_ep_libraries: Mutex<std::collections::HashMap<String, PathBuf>>,
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
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            registered_ep_libraries: Mutex::new(std::collections::HashMap::new()),
        })
    }

    pub(crate) fn as_ptr(&self) -> *const onnx_genai_ort_sys::OrtEnv {
        self.ptr.as_ptr()
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    pub(crate) fn register_execution_provider_library(
        &self,
        registration_name: &str,
        path: &Path,
    ) -> Result<()> {
        let mut registered = self.registered_ep_libraries.lock().map_err(|_| {
            OrtError::InvalidArgument("execution provider registration lock was poisoned".into())
        })?;
        if let Some(registered_path) = registered.get(registration_name) {
            if registered_path == path {
                return Ok(());
            }
            return Err(OrtError::InvalidArgument(format!(
                "execution provider {registration_name} is already registered from {}",
                registered_path.display()
            )));
        }

        let name = CString::new(registration_name).map_err(|_| {
            OrtError::InvalidArgument("execution provider registration name contains NUL".into())
        })?;
        let path_c = CString::new(path.to_string_lossy().as_bytes()).map_err(|_| {
            OrtError::InvalidArgument("execution provider library path contains NUL".into())
        })?;
        let api = crate::error::api()?;
        let register = api
            .RegisterExecutionProviderLibrary
            .ok_or(OrtError::ApiUnavailable("RegisterExecutionProviderLibrary"))?;
        // SAFETY: the environment is live, and both C strings are NUL-terminated
        // and remain valid for the duration of the registration call.
        crate::error::check_status(unsafe {
            register(self.ptr.as_ptr(), name.as_ptr(), path_c.as_ptr())
        })?;
        registered.insert(registration_name.to_string(), path.to_path_buf());
        Ok(())
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

// SAFETY: ORT environments are process-level handles that the ORT C API permits
// to be used from multiple threads. This wrapper only shares the opaque handle,
// never mutates Rust-owned state through shared references, and releases it once
// from `Drop` after owning structs have dropped their sessions. This would stop
// being sound if ORT changed `OrtEnv` to require thread-affine access, or if a
// future owner dropped the environment while live sessions could still call ORT.
unsafe impl Send for Environment {}
// SAFETY: Same invariant as `Send`: shared references expose only the stable ORT
// environment handle, whose operations are thread-safe under ORT's contract.
unsafe impl Sync for Environment {}

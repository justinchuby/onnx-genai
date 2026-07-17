//! ORT Environment (global singleton).

use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::Mutex;

use crate::{OrtError, Result};

/// ORT Environment — must be created before any sessions.
/// Typically one per process.
pub struct Environment {
    ptr: NonNull<onnx_genai_ort_sys::OrtEnv>,
    _name: String,
    registered_ep_libraries: Mutex<std::collections::HashMap<String, PathBuf>>,
    /// Maps an EP-plugin registration handle to the provider name discovered on
    /// first registration, so later sessions sharing this environment can
    /// re-select the plugin's devices without re-running the (now impossible)
    /// before/after device diff.
    resolved_plugin_providers: Mutex<std::collections::HashMap<String, String>>,
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
            registered_ep_libraries: Mutex::new(std::collections::HashMap::new()),
            resolved_plugin_providers: Mutex::new(std::collections::HashMap::new()),
        })
    }

    pub(crate) fn as_ptr(&self) -> *const onnx_genai_ort_sys::OrtEnv {
        self.ptr.as_ptr()
    }

    /// Register an ORT execution-provider plugin shared library.
    ///
    /// Returns `Ok(true)` when this call performed the registration and
    /// `Ok(false)` when the same handle+path was already registered on this
    /// environment (ORT registration is process-global and must not be
    /// repeated). Callers use the flag to decide whether the plugin's devices
    /// can still be identified by a fresh before/after device diff.
    pub(crate) fn register_execution_provider_library(
        &self,
        registration_name: &str,
        path: &Path,
    ) -> Result<bool> {
        let mut registered = self.registered_ep_libraries.lock().map_err(|_| {
            OrtError::InvalidArgument("execution provider registration lock was poisoned".into())
        })?;
        if let Some(registered_path) = registered.get(registration_name) {
            if registered_path == path {
                return Ok(false);
            }
            return Err(OrtError::InvalidArgument(format!(
                "execution provider {registration_name} is already registered from {}",
                registered_path.display()
            )));
        }

        let name = CString::new(registration_name).map_err(|_| {
            OrtError::InvalidArgument("execution provider registration name contains NUL".into())
        })?;
        let api = crate::error::api()?;
        let register = api
            .RegisterExecutionProviderLibrary
            .ok_or(OrtError::ApiUnavailable("RegisterExecutionProviderLibrary"))?;

        // ORT's `RegisterExecutionProviderLibrary` takes the library path as
        // `ORTCHAR_T*`, which is `wchar_t` (UTF-16) on Windows and `char`
        // (UTF-8) elsewhere. Encode accordingly.
        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStrExt;
            let mut wide: Vec<u16> = path.as_os_str().encode_wide().collect();
            if wide.contains(&0) {
                return Err(OrtError::InvalidArgument(
                    "execution provider library path contains NUL".into(),
                ));
            }
            wide.push(0);
            // SAFETY: the environment is live, `name` is NUL-terminated, and
            // `wide` is a NUL-terminated UTF-16 buffer valid for the call.
            crate::error::check_status(unsafe {
                register(self.ptr.as_ptr(), name.as_ptr(), wide.as_ptr())
            })?;
        }
        #[cfg(not(windows))]
        {
            let path_c = CString::new(path.to_string_lossy().as_bytes()).map_err(|_| {
                OrtError::InvalidArgument("execution provider library path contains NUL".into())
            })?;
            // SAFETY: the environment is live, and both C strings are
            // NUL-terminated and valid for the duration of the call.
            crate::error::check_status(unsafe {
                register(self.ptr.as_ptr(), name.as_ptr(), path_c.as_ptr())
            })?;
        }
        registered.insert(registration_name.to_string(), path.to_path_buf());
        Ok(true)
    }

    /// Fetch the provider name previously discovered for a plugin registration
    /// handle, if any.
    pub(crate) fn cached_plugin_provider(&self, registration_name: &str) -> Option<String> {
        self.resolved_plugin_providers
            .lock()
            .ok()
            .and_then(|map| map.get(registration_name).cloned())
    }

    /// Record the provider name discovered for a plugin registration handle so
    /// subsequent sessions sharing this environment can re-select its devices.
    pub(crate) fn cache_plugin_provider(&self, registration_name: &str, provider_name: &str) {
        if let Ok(mut map) = self.resolved_plugin_providers.lock() {
            map.insert(registration_name.to_string(), provider_name.to_string());
        }
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

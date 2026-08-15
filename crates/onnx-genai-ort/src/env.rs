//! ORT Environment (global singleton).

use std::ffi::CString;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::{Mutex, MutexGuard, OnceLock};

use crate::{OrtError, Result};

#[derive(Default)]
struct PluginRegistration {
    path: PathBuf,
    provider_name: Option<String>,
}

#[derive(Default)]
struct EnvironmentLifecycle {
    active_environments: usize,
}

impl EnvironmentLifecycle {
    fn ensure_creation_capacity(&self) -> Result<()> {
        if self.active_environments == usize::MAX {
            return Err(OrtError::InvalidArgument(
                "ORT environment reference count overflowed".into(),
            ));
        }
        Ok(())
    }

    fn environment_created(&mut self) {
        self.active_environments += 1;
    }

    fn environment_released(&mut self) -> bool {
        if self.active_environments == 0 {
            debug_assert!(false, "released an untracked ORT environment");
            return false;
        }
        self.active_environments -= 1;
        self.active_environments == 0
    }
}

fn environment_lifecycle() -> &'static Mutex<EnvironmentLifecycle> {
    static LIFECYCLE: OnceLock<Mutex<EnvironmentLifecycle>> = OnceLock::new();
    LIFECYCLE.get_or_init(|| Mutex::new(EnvironmentLifecycle::default()))
}

fn registered_ep_libraries() -> &'static Mutex<std::collections::HashMap<String, PluginRegistration>>
{
    static REGISTERED: OnceLock<Mutex<std::collections::HashMap<String, PluginRegistration>>> =
        OnceLock::new();
    REGISTERED.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

fn release_environment(
    lifecycle: &mut EnvironmentLifecycle,
    registrations: &Mutex<std::collections::HashMap<String, PluginRegistration>>,
) {
    if lifecycle.environment_released() {
        registrations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
    }
}

fn plugin_registration_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

fn plugin_discovery_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// ORT Environment — must be created before any sessions.
/// Typically one per process.
pub struct Environment {
    ptr: NonNull<onnx_genai_ort_sys::OrtEnv>,
    _name: String,
}

impl Environment {
    /// Create a new ORT environment.
    pub fn new(name: &str) -> Result<Self> {
        // Serialize the ORT refcount transition with last-environment cache
        // clearing. A new OrtEnv cannot become visible between ReleaseEnv and
        // clearing registrations owned by the previous OrtEnv generation.
        let mut lifecycle = environment_lifecycle().lock().map_err(|_| {
            OrtError::InvalidArgument("ORT environment lifecycle lock was poisoned".into())
        })?;
        lifecycle.ensure_creation_capacity()?;
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
        let ptr = NonNull::new(ptr).ok_or(OrtError::NullPointer)?;
        lifecycle.environment_created();
        tracing::info!("Creating ORT environment: {}", name);
        Ok(Self {
            ptr,
            _name: name.to_string(),
        })
    }

    pub(crate) fn as_ptr(&self) -> *const onnx_genai_ort_sys::OrtEnv {
        self.ptr.as_ptr()
    }

    /// Register an ORT execution-provider plugin shared library.
    ///
    /// Returns `Ok(true)` when this call performed the registration and
    /// `Ok(false)` when the same handle+path was already registered in the
    /// current ORT environment generation. The registration is shared by
    /// concurrently live `Environment` wrappers.
    pub(crate) fn register_execution_provider_library(
        &self,
        registration_name: &str,
        path: &Path,
    ) -> Result<bool> {
        // Serialize registrations without retaining the cache mutex across the
        // ORT call. Environment create/drop may re-enter while ORT loads a
        // plugin, and last-environment teardown needs the cache mutex to clear
        // registrations without deadlocking.
        let _registration_guard = plugin_registration_lock().lock().map_err(|_| {
            OrtError::InvalidArgument(
                "execution provider registration call lock was poisoned".into(),
            )
        })?;
        {
            let registered = registered_ep_libraries().lock().map_err(|_| {
                OrtError::InvalidArgument(
                    "execution provider registration lock was poisoned".into(),
                )
            })?;
            if let Some(registration) = registered.get(registration_name) {
                if registration.path == path {
                    return Ok(false);
                }
                return Err(OrtError::InvalidArgument(format!(
                    "execution provider {registration_name} is already registered from {}",
                    registration.path.display()
                )));
            }
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
        let mut registered = registered_ep_libraries().lock().map_err(|_| {
            OrtError::InvalidArgument("execution provider registration lock was poisoned".into())
        })?;
        registered.insert(
            registration_name.to_string(),
            PluginRegistration {
                path: path.to_path_buf(),
                provider_name: None,
            },
        );
        Ok(true)
    }

    /// Serialize process-global plugin registration with provider-name
    /// discovery. This keeps another environment from observing a completed ORT
    /// registration before its provider name has been cached.
    pub(crate) fn lock_plugin_discovery(&self) -> Result<MutexGuard<'static, ()>> {
        plugin_discovery_lock().lock().map_err(|_| {
            OrtError::InvalidArgument("execution provider discovery lock was poisoned".into())
        })
    }

    /// Fetch the provider name discovered for a plugin registration handle in
    /// the current ORT environment generation, if any.
    pub(crate) fn cached_plugin_provider(&self, registration_name: &str) -> Result<Option<String>> {
        registered_ep_libraries()
            .lock()
            .map_err(|_| {
                OrtError::InvalidArgument(
                    "execution provider registration lock was poisoned".into(),
                )
            })
            .map(|map| {
                map.get(registration_name)
                    .and_then(|registration| registration.provider_name.clone())
            })
    }

    /// Record the provider name discovered for a plugin registration handle so
    /// sessions using any concurrently live environment can re-select its
    /// devices.
    pub(crate) fn cache_plugin_provider(
        &self,
        registration_name: &str,
        provider_name: &str,
    ) -> Result<()> {
        let mut registered = registered_ep_libraries().lock().map_err(|_| {
            OrtError::InvalidArgument("execution provider registration lock was poisoned".into())
        })?;
        let registration = registered.get_mut(registration_name).ok_or_else(|| {
            OrtError::InvalidArgument(format!(
                "execution provider {registration_name} was not registered"
            ))
        })?;
        if let Some(existing) = &registration.provider_name
            && existing != provider_name
        {
            return Err(OrtError::InvalidArgument(format!(
                "execution provider {registration_name} was already resolved as {existing}, not {provider_name}"
            )));
        }
        registration.provider_name = Some(provider_name.to_string());
        Ok(())
    }
}

impl Drop for Environment {
    fn drop(&mut self) {
        // Keep this guard through ReleaseEnv and cache clearing. Environment::new
        // takes the same lock before CreateEnv, so a fresh OrtEnv generation
        // cannot observe registrations owned by the generation being destroyed.
        let mut lifecycle = environment_lifecycle()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Ok(api) = crate::error::api()
            && let Some(release) = api.ReleaseEnv
        {
            // SAFETY: `ptr` is owned by this wrapper and released exactly once here.
            unsafe { release(self.ptr.as_ptr()) };
            release_environment(&mut lifecycle, registered_ep_libraries());
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

#[cfg(test)]
mod tests {
    use super::*;

    const REAL_LIFECYCLE_CHILD: &str = "ONNX_GENAI_REAL_ENV_LIFECYCLE_TEST";

    #[test]
    fn plugin_registration_cache_is_cleared_after_last_environment_drop() {
        if std::env::var_os(REAL_LIFECYCLE_CHILD).is_none() {
            // Run this assertion in an isolated test process so no parallel unit
            // test can keep another real Environment alive and mask the 1 -> 0
            // transition under test.
            let output = std::process::Command::new(
                std::env::current_exe().expect("current unit-test executable"),
            )
            .arg("--exact")
            .arg("env::tests::plugin_registration_cache_is_cleared_after_last_environment_drop")
            .arg("--nocapture")
            .env(REAL_LIFECYCLE_CHILD, "1")
            .output()
            .expect("run isolated real Environment lifecycle test");
            assert!(
                output.status.success(),
                "isolated real Environment lifecycle test failed\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return;
        }

        assert_eq!(
            environment_lifecycle()
                .lock()
                .expect("lifecycle lock")
                .active_environments,
            0
        );
        assert!(
            registered_ep_libraries()
                .lock()
                .expect("registration lock")
                .is_empty()
        );

        let first = Environment::new("plugin-lifecycle-first").expect("first environment");
        let second = Environment::new("plugin-lifecycle-second").expect("second environment");
        let registration_name = format!("onnx_genai_test_plugin_{}", std::process::id());
        let path = PathBuf::from("/onnx-genai/test/plugin.so");

        assert_eq!(
            environment_lifecycle()
                .lock()
                .expect("lifecycle lock")
                .active_environments,
            2
        );

        // There is no plugin shared library fixture in the unit-test suite.
        // Seed the real production cache, then exercise its production
        // registration/provider accessors and the real Environment lifecycle.
        registered_ep_libraries()
            .lock()
            .expect("registration lock")
            .insert(
                registration_name.clone(),
                PluginRegistration {
                    path: path.clone(),
                    provider_name: None,
                },
            );
        assert!(
            !second
                .register_execution_provider_library(&registration_name, &path)
                .expect("live environment should reuse registration")
        );
        first
            .cache_plugin_provider(&registration_name, "TestExecutionProvider")
            .expect("cache provider");
        assert_eq!(
            second
                .cached_plugin_provider(&registration_name)
                .expect("read provider")
                .as_deref(),
            Some("TestExecutionProvider")
        );

        drop(first);
        assert_eq!(
            environment_lifecycle()
                .lock()
                .expect("lifecycle lock")
                .active_environments,
            1
        );
        assert!(
            registered_ep_libraries()
                .lock()
                .expect("registration lock")
                .contains_key(&registration_name)
        );
        assert!(
            !second
                .register_execution_provider_library(&registration_name, &path)
                .expect("registration remains shared while an environment is live")
        );

        drop(second);
        assert_eq!(
            environment_lifecycle()
                .lock()
                .expect("lifecycle lock")
                .active_environments,
            0
        );
        assert!(
            registered_ep_libraries()
                .lock()
                .expect("registration lock")
                .is_empty()
        );

        let fresh = Environment::new("plugin-lifecycle-fresh").expect("fresh environment");
        assert_eq!(
            environment_lifecycle()
                .lock()
                .expect("lifecycle lock")
                .active_environments,
            1
        );
        assert_eq!(
            fresh
                .cached_plugin_provider(&registration_name)
                .expect("read provider after recreation"),
            None
        );
        assert!(
            fresh
                .register_execution_provider_library(&registration_name, &path)
                .is_err(),
            "a fresh environment must attempt the missing plugin registration, not return Ok(false) from stale cache"
        );
        drop(fresh);
        assert_eq!(
            environment_lifecycle()
                .lock()
                .expect("lifecycle lock")
                .active_environments,
            0
        );
    }
}

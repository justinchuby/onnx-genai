//! Dynamic library loading and symbol resolution for nxrt plugins.
//!
//! The [`NxrtPlugin`] struct owns an `Arc<libloading::Library>` so that any EP
//! instance or kernel obtained from the plugin structurally cannot outlive the
//! library — the `Arc` prevents unload while live references exist.

use std::ffi::CStr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use libloading::{Library, Symbol};

use crate::abi_contract::{
    NXRT_ABI_VERSION_MAJOR, NXRT_ABI_VERSION_MINOR, NxrtAbiVersionFn, NxrtCreateEpFn,
    NxrtDestroyEpFn, NxrtDeviceCountFn, NxrtEpHandle, NxrtEpNameFn,
    SYM_NXRT_ABI_VERSION, SYM_NXRT_CREATE_EP, SYM_NXRT_DESTROY_EP, SYM_NXRT_DEVICE_COUNT,
    SYM_NXRT_EP_NAME,
};
use crate::error::NxrtHostError;

/// A successfully loaded nxrt plugin library with validated ABI version.
///
/// Holds an `Arc<Library>` so clones and EP instances derived from this plugin
/// share ownership of the loaded library, preventing unload while any live
/// reference exists.
#[derive(Clone, Debug)]
pub struct NxrtPlugin {
    /// Shared ownership of the loaded dynamic library. Must outlive every
    /// symbol or handle obtained from it.
    pub(crate) library: Arc<Library>,
    /// Filesystem path used to load the library (for diagnostics).
    pub(crate) path: PathBuf,
    /// The EP name reported by the plugin.
    pub(crate) name: String,
    /// ABI version reported by the plugin (for diagnostics).
    pub(crate) abi_major: u32,
    pub(crate) abi_minor: u32,
}

impl NxrtPlugin {
    /// The EP name reported by the plugin.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The filesystem path used to load the plugin.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// ABI version reported by the plugin.
    pub fn abi_version(&self) -> (u32, u32) {
        (self.abi_major, self.abi_minor)
    }
}

/// Load an nxrt plugin from the given path, performing version negotiation.
///
/// # Errors
///
/// Returns an actionable error for each failure mode:
/// - Library not found or unloadable
/// - Missing required symbol
/// - Incompatible ABI major version
///
/// # Safety
///
/// Loading a dynamic library is inherently unsafe: a malicious library can
/// execute arbitrary code in its init routines. Callers must ensure the path
/// points to a trusted nxrt plugin.
pub fn load_nxrt_plugin(path: impl AsRef<Path>) -> Result<NxrtPlugin, NxrtHostError> {
    let path = path.as_ref().to_path_buf();

    // Phase 1: Load the library.
    let library = unsafe { Library::new(&path) }.map_err(|e| NxrtHostError::LibraryLoadFailed {
        path: path.clone(),
        reason: e.to_string(),
    })?;

    // Phase 2: Resolve version symbol and negotiate.
    let version_fn = resolve_symbol::<NxrtAbiVersionFn>(&library, SYM_NXRT_ABI_VERSION, &path)?;
    let (plugin_major, plugin_minor) = {
        let mut major: u32 = 0;
        let mut minor: u32 = 0;
        unsafe { version_fn(&mut major, &mut minor) };
        (major, minor)
    };

    if plugin_major != NXRT_ABI_VERSION_MAJOR {
        return Err(NxrtHostError::AbiVersionMismatch {
            path,
            host_major: NXRT_ABI_VERSION_MAJOR,
            host_minor: NXRT_ABI_VERSION_MINOR,
            plugin_major,
            plugin_minor,
        });
    }

    // Phase 3: Resolve remaining required symbols to fail fast on misspelled exports.
    resolve_symbol::<NxrtCreateEpFn>(&library, SYM_NXRT_CREATE_EP, &path)?;
    resolve_symbol::<NxrtDestroyEpFn>(&library, SYM_NXRT_DESTROY_EP, &path)?;
    resolve_symbol::<NxrtDeviceCountFn>(&library, SYM_NXRT_DEVICE_COUNT, &path)?;

    // Phase 4: Query EP name.
    let name_fn = resolve_symbol::<NxrtEpNameFn>(&library, SYM_NXRT_EP_NAME, &path)?;
    let name = unsafe {
        let ptr = name_fn();
        if ptr.is_null() {
            String::from("unknown")
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    Ok(NxrtPlugin {
        library: Arc::new(library),
        path,
        name,
        abi_major: plugin_major,
        abi_minor: plugin_minor,
    })
}

/// Resolve a single symbol from the library, producing an actionable error on failure.
fn resolve_symbol<'lib, T>(
    library: &'lib Library,
    sym_name: &[u8],
    path: &Path,
) -> Result<Symbol<'lib, T>, NxrtHostError> {
    // sym_name includes trailing NUL for C interop; strip it for the display name.
    let display_name = std::str::from_utf8(&sym_name[..sym_name.len() - 1])
        .unwrap_or("<invalid utf8>")
        .to_string();

    unsafe { library.get::<T>(sym_name) }.map_err(|e| NxrtHostError::SymbolNotFound {
        path: path.to_path_buf(),
        symbol: display_name,
        reason: e.to_string(),
    })
}

/// Create an EP instance from a loaded plugin.
///
/// This is separated from `load_nxrt_plugin` so that callers can inspect the
/// plugin metadata before committing to EP creation.
pub(crate) fn create_ep_instance(
    plugin: &NxrtPlugin,
    config_json: &CStr,
) -> Result<*mut NxrtEpHandle, NxrtHostError> {
    let create_fn: Symbol<NxrtCreateEpFn> =
        unsafe { plugin.library.get(SYM_NXRT_CREATE_EP) }.map_err(|e| {
            NxrtHostError::SymbolNotFound {
                path: plugin.path.clone(),
                symbol: "nxrt_create_ep".into(),
                reason: e.to_string(),
            }
        })?;

    let mut handle: *mut NxrtEpHandle = std::ptr::null_mut();
    let status = unsafe { create_fn(config_json.as_ptr(), &mut handle) };

    if !status.is_ok() {
        return Err(NxrtHostError::FactoryFailed {
            path: plugin.path.clone(),
            status: format!(
                "nxrt_create_ep returned {}: {}",
                status as i32,
                status.as_str()
            ),
        });
    }

    if handle.is_null() {
        return Err(NxrtHostError::FactoryFailed {
            path: plugin.path.clone(),
            status: "nxrt_create_ep returned Ok but handle is null".into(),
        });
    }

    // Validate device count > 0.
    let device_count_fn: Symbol<NxrtDeviceCountFn> =
        unsafe { plugin.library.get(SYM_NXRT_DEVICE_COUNT) }.map_err(|e| {
            NxrtHostError::SymbolNotFound {
                path: plugin.path.clone(),
                symbol: "nxrt_device_count".into(),
                reason: e.to_string(),
            }
        })?;

    let mut count: u32 = 0;
    let count_status = unsafe { device_count_fn(handle, &mut count) };
    if !count_status.is_ok() {
        // Destroy the handle before returning the error.
        destroy_ep_instance(plugin, handle);
        return Err(NxrtHostError::PluginCallFailed(format!(
            "nxrt_device_count failed: {}",
            count_status.as_str()
        )));
    }

    if count == 0 {
        destroy_ep_instance(plugin, handle);
        return Err(NxrtHostError::ZeroDevices {
            path: plugin.path.clone(),
        });
    }

    Ok(handle)
}

/// Destroy an EP handle. Best-effort; does not propagate errors.
pub(crate) fn destroy_ep_instance(plugin: &NxrtPlugin, handle: *mut NxrtEpHandle) {
    if handle.is_null() {
        return;
    }
    if let Ok(destroy_fn) =
        unsafe { plugin.library.get::<NxrtDestroyEpFn>(SYM_NXRT_DESTROY_EP) }
    {
        unsafe { destroy_fn(handle) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn load_missing_library_fails_with_actionable_error() {
        let result = load_nxrt_plugin("/nonexistent/path/libfake_nxrt.so");
        let err = result.unwrap_err();
        match &err {
            NxrtHostError::LibraryLoadFailed { path, reason } => {
                assert_eq!(path, &PathBuf::from("/nonexistent/path/libfake_nxrt.so"));
                assert!(!reason.is_empty(), "reason must be non-empty");
            }
            other => panic!("expected LibraryLoadFailed, got: {other:?}"),
        }
        // Verify the Display impl is actionable.
        let msg = err.to_string();
        assert!(msg.contains("libfake_nxrt.so"), "error must name the path");
    }

    #[test]
    fn abi_version_mismatch_error_is_actionable() {
        let err = NxrtHostError::AbiVersionMismatch {
            path: PathBuf::from("libtest.so"),
            host_major: 1,
            host_minor: 0,
            plugin_major: 2,
            plugin_minor: 3,
        };
        let msg = err.to_string();
        assert!(msg.contains("major 1"), "must name host version");
        assert!(msg.contains("major 2"), "must name plugin version");
        assert!(msg.contains("Rebuild"), "must suggest fix");
    }

    #[test]
    fn zero_devices_error_is_actionable() {
        let err = NxrtHostError::ZeroDevices {
            path: PathBuf::from("libtest.so"),
        };
        let msg = err.to_string();
        assert!(msg.contains("zero devices"));
        assert!(msg.contains("at least one"));
    }

    #[test]
    fn factory_failed_error_is_actionable() {
        let err = NxrtHostError::FactoryFailed {
            path: PathBuf::from("libtest.so"),
            status: "internal error".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("factory failed"));
        assert!(msg.contains("internal error"));
    }

    // Integration tests requiring a real cdylib are deferred to Pris's
    // cross-crate test suite. Here we validate the negative paths that don't
    // require a real plugin.
}

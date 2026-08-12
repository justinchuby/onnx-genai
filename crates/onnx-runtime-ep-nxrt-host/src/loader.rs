//! Dynamic library loading and symbol resolution for nxrt plugins.
//!
//! The [`NxrtPlugin`] struct owns an `Arc<libloading::Library>` so that any EP
//! instance or kernel obtained from the plugin structurally cannot outlive the
//! library — the `Arc` prevents unload while live references exist.
//!
//! # Load sequence (Nabil's ABI)
//!
//! 1. `dlopen` the library.
//! 2. Resolve `NxrtNegotiate` — call it with `NxrtNegotiateRequest::current()`.
//! 3. Validate the response: major mismatch → reject; agreed minor > host minor → reject;
//!    unknown capability bits → reject (fail closed).
//! 4. Resolve `NxrtCreateEpFactories` — call it to obtain factory vtables.
//! 5. The host owns all returned factories and must release them via vtable `release`.

use std::ffi::CStr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use libloading::{Library, Symbol};
use onnx_runtime_ep_nxrt_abi::{
    version::validate_negotiation, NxrtCreateEpFactoriesFn, NxrtEpFactoryVtable, NxrtNegotiateFn,
    NxrtNegotiateRequest, NxrtNegotiateResponse, NxrtVersionRange, NXRT_SYMBOL_CREATE_EP_FACTORIES,
    NXRT_SYMBOL_NEGOTIATE,
};

use crate::error::NxrtHostError;

/// Maximum number of factories we'll accept from a single plugin.
const MAX_FACTORIES: usize = 16;

/// A successfully loaded nxrt plugin library with validated ABI version.
///
/// Holds an `Arc<Library>` so clones and EP instances derived from this plugin
/// share ownership of the loaded library, preventing unload while any live
/// reference exists.
#[derive(Clone)]
pub struct NxrtPlugin {
    /// Shared ownership of the loaded dynamic library.
    pub(crate) library: Arc<Library>,
    /// Filesystem path used to load the library (for diagnostics).
    pub(crate) path: PathBuf,
    /// The EP name reported by the first factory (for diagnostics).
    pub(crate) name: String,
    /// ABI version agreed during negotiation.
    pub(crate) abi_major: u32,
    pub(crate) abi_minor: u32,
    /// Capability flags advertised by the plugin.
    #[allow(dead_code)]
    pub(crate) capability_flags: u64,
    /// Factory vtable pointers owned by this struct. Released on Drop.
    pub(crate) factories: Arc<FactorySet>,
}

impl std::fmt::Debug for NxrtPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NxrtPlugin")
            .field("path", &self.path)
            .field("name", &self.name)
            .field("abi_version", &(self.abi_major, self.abi_minor))
            .field("num_factories", &self.factories.count)
            .finish()
    }
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

    /// ABI version agreed during negotiation.
    pub fn abi_version(&self) -> (u32, u32) {
        (self.abi_major, self.abi_minor)
    }

    /// Number of EP factories provided by the plugin.
    pub fn num_factories(&self) -> usize {
        self.factories.count
    }

    /// Get a factory vtable by index.
    ///
    /// # Safety
    ///
    /// The returned pointer is valid as long as this `NxrtPlugin` (or a clone)
    /// is alive. The caller must not call `release` on it — that is done by
    /// `FactorySet::drop`.
    pub(crate) fn factory(&self, index: usize) -> Option<*mut NxrtEpFactoryVtable> {
        if index < self.factories.count {
            Some(self.factories.ptrs[index])
        } else {
            None
        }
    }
}

/// Owns the factory vtable pointers and releases them on drop.
pub(crate) struct FactorySet {
    ptrs: [*mut NxrtEpFactoryVtable; MAX_FACTORIES],
    count: usize,
    /// Keep library alive while factories exist.
    _library: Arc<Library>,
}

// SAFETY: Factory vtables are thread-safe per the nxrt ABI contract.
unsafe impl Send for FactorySet {}
unsafe impl Sync for FactorySet {}

impl Drop for FactorySet {
    fn drop(&mut self) {
        for i in 0..self.count {
            let factory = self.ptrs[i];
            if !factory.is_null() {
                // Catch panics at the plugin boundary.
                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    // Validate struct_size covers release+ctx before calling.
                    let release_end = std::mem::offset_of!(NxrtEpFactoryVtable, release)
                        + std::mem::size_of::<unsafe extern "C" fn(*mut std::ffi::c_void)>();
                    let ctx_end = std::mem::offset_of!(NxrtEpFactoryVtable, ctx)
                        + std::mem::size_of::<*mut std::ffi::c_void>();
                    let min_size = release_end.max(ctx_end);
                    let struct_size = unsafe { (*factory).struct_size } as usize;
                    if struct_size < min_size {
                        eprintln!(
                            "WARNING: factory[{i}] struct_size ({struct_size}) too small \
                             to contain release/ctx ({min_size}), skipping release"
                        );
                        return;
                    }
                    // SAFETY: We own these factories per the ABI contract.
                    unsafe { ((*factory).release)((*factory).ctx) };
                }));
            }
        }
    }
}

/// Load an nxrt plugin from the given path, performing version negotiation.
///
/// # Errors
///
/// Returns an actionable error for each failure mode:
/// - Library not found or unloadable
/// - Missing required symbol (`NxrtNegotiate` or `NxrtCreateEpFactories`)
/// - Incompatible ABI version (major mismatch, minor too new, unknown caps)
/// - Factory creation failure or zero factories
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
    let library = Arc::new(library);

    // Phase 2: Resolve NxrtNegotiate and perform version handshake.
    let negotiate_fn = resolve_symbol::<NxrtNegotiateFn>(&library, NXRT_SYMBOL_NEGOTIATE, &path)?;

    let request = NxrtNegotiateRequest::current();
    let mut response = NxrtNegotiateResponse::zeroed();

    let status = unsafe { negotiate_fn(&request, &mut response) };
    if !status.is_ok() {
        return Err(NxrtHostError::AbiVersionMismatch {
            path,
            host_major: request.host_range.major_max,
            host_minor: request.host_range.minor_max,
            plugin_major: response.plugin_range.major_max,
            plugin_minor: response.plugin_range.minor_max,
        });
    }

    // Phase 3: Host-side validation of the negotiation response.
    let host_range = NxrtVersionRange::current();
    if let Err(_reason) = validate_negotiation(&host_range, &response) {
        return Err(NxrtHostError::AbiVersionMismatch {
            path,
            host_major: host_range.major_max,
            host_minor: host_range.minor_max,
            plugin_major: response.agreed_major,
            plugin_minor: response.agreed_minor,
        });
    }

    // Phase 4: Resolve NxrtCreateEpFactories and call it.
    let create_fn = resolve_symbol::<NxrtCreateEpFactoriesFn>(
        &library,
        NXRT_SYMBOL_CREATE_EP_FACTORIES,
        &path,
    )?;

    let mut factory_ptrs: [*mut NxrtEpFactoryVtable; MAX_FACTORIES] =
        [std::ptr::null_mut(); MAX_FACTORIES];
    let mut num_factories: usize = 0;

    let create_status =
        unsafe { create_fn(factory_ptrs.as_mut_ptr(), MAX_FACTORIES, &mut num_factories) };

    if !create_status.is_ok() {
        let msg = create_status
            .message_str()
            .unwrap_or("(no message)")
            .to_owned();
        return Err(NxrtHostError::FactoryFailed { path, status: msg });
    }

    if num_factories == 0 {
        return Err(NxrtHostError::ZeroDevices { path });
    }

    // Clamp to actual array size for safety (untrusted plugin).
    if num_factories > MAX_FACTORIES {
        num_factories = MAX_FACTORIES;
    }

    // Validate all factory pointers are non-null.
    for i in 0..num_factories {
        if factory_ptrs[i].is_null() {
            // Release already-obtained factories.
            for item in factory_ptrs.iter().take(i) {
                if !item.is_null() {
                    let fptr = *item;
                    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let release_end = std::mem::offset_of!(NxrtEpFactoryVtable, release)
                            + std::mem::size_of::<unsafe extern "C" fn(*mut std::ffi::c_void)>();
                        let ctx_end = std::mem::offset_of!(NxrtEpFactoryVtable, ctx)
                            + std::mem::size_of::<*mut std::ffi::c_void>();
                        let min_size = release_end.max(ctx_end);
                        if (unsafe { (*fptr).struct_size } as usize) >= min_size {
                            unsafe { ((*fptr).release)((*fptr).ctx) };
                        }
                    }));
                }
            }
            return Err(NxrtHostError::FactoryFailed {
                path,
                status: format!("factory pointer at index {i} is null"),
            });
        }
    }

    // Read name from first factory (borrowed for factory lifetime — copy it).
    // Validate struct_size covers the `name` field before dereferencing it.
    // A newer host talking to an older plugin must not read past the end of
    // a smaller struct (mirrors provider_adapter.rs struct_size guards).
    let name = unsafe {
        let factory = &*factory_ptrs[0];
        let name_end =
            std::mem::offset_of!(NxrtEpFactoryVtable, name) + std::mem::size_of::<*const u8>();
        if (factory.struct_size as usize) < name_end {
            String::from("unknown")
        } else if factory.name.is_null() {
            String::from("unknown")
        } else {
            CStr::from_ptr(factory.name as *const std::os::raw::c_char)
                .to_string_lossy()
                .into_owned()
        }
    };

    let factory_set = Arc::new(FactorySet {
        ptrs: factory_ptrs,
        count: num_factories,
        _library: Arc::clone(&library),
    });

    Ok(NxrtPlugin {
        library,
        path,
        name,
        abi_major: response.agreed_major,
        abi_minor: response.agreed_minor,
        capability_flags: response.capability_flags,
        factories: factory_set,
    })
}

/// Resolve a single symbol from the library, producing an actionable error on failure.
fn resolve_symbol<'lib, T>(
    library: &'lib Library,
    sym_name: &[u8],
    path: &Path,
) -> Result<Symbol<'lib, T>, NxrtHostError> {
    let display_name = std::str::from_utf8(sym_name)
        .unwrap_or("<invalid utf8>")
        .to_string();

    unsafe { library.get::<T>(sym_name) }.map_err(|e| NxrtHostError::SymbolNotFound {
        path: path.to_path_buf(),
        symbol: display_name,
        reason: e.to_string(),
    })
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
        assert!(msg.contains("zero"));
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
}

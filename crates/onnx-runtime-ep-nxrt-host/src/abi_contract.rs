//! The nxrt ABI contract as expected by the host loader.
//!
//! This module defines the C ABI types and symbol names that an nxrt plugin
//! `.so`/`.dll` must export. When Nabil's `onnx-runtime-ep-nxrt-abi` crate
//! lands, this module should be replaced by a re-export from that crate.

use std::ffi::c_char;

// ─── Version constants ──────────────────────────────────────────────────────

/// The major ABI version this host requires. A plugin with a different major
/// version is incompatible and must not be loaded.
pub const NXRT_ABI_VERSION_MAJOR: u32 = 1;

/// The minor ABI version this host understands. A plugin with a higher minor
/// version is forward-compatible (new optional features only).
pub const NXRT_ABI_VERSION_MINOR: u32 = 0;

// ─── Symbol names ───────────────────────────────────────────────────────────

/// Entry-point symbol exported by every nxrt plugin for version negotiation.
pub const SYM_NXRT_ABI_VERSION: &[u8] = b"nxrt_abi_version\0";

/// Factory symbol that creates an EP instance.
pub const SYM_NXRT_CREATE_EP: &[u8] = b"nxrt_create_ep\0";

/// Symbol to destroy an EP instance returned by the factory.
pub const SYM_NXRT_DESTROY_EP: &[u8] = b"nxrt_destroy_ep\0";

/// Symbol that returns the plugin's human-readable name (null-terminated UTF-8).
pub const SYM_NXRT_EP_NAME: &[u8] = b"nxrt_ep_name\0";

/// Symbol that queries the number of available devices.
pub const SYM_NXRT_DEVICE_COUNT: &[u8] = b"nxrt_device_count\0";

// ─── ABI function signatures ────────────────────────────────────────────────

/// `nxrt_abi_version(out_major: *mut u32, out_minor: *mut u32)`
///
/// Writes the plugin's ABI major/minor version. Never fails.
pub type NxrtAbiVersionFn = unsafe extern "C" fn(out_major: *mut u32, out_minor: *mut u32);

/// Status code returned by fallible nxrt C functions.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NxrtStatus {
    Ok = 0,
    InvalidArgument = 1,
    InternalError = 2,
    Unsupported = 3,
    OutOfMemory = 4,
}

/// Opaque EP handle.
#[repr(C)]
pub struct NxrtEpHandle {
    _opaque: [u8; 0],
}

/// `nxrt_create_ep(config_json: *const c_char, out_handle: *mut *mut NxrtEpHandle) -> NxrtStatus`
///
/// Creates an EP instance. `config_json` is a null-terminated UTF-8 JSON
/// string with provider-specific configuration. On success, `*out_handle` is a
/// non-null opaque EP handle. On failure, `*out_handle` is null and the return
/// value indicates the error category.
pub type NxrtCreateEpFn = unsafe extern "C" fn(
    config_json: *const c_char,
    out_handle: *mut *mut NxrtEpHandle,
) -> NxrtStatus;

/// `nxrt_destroy_ep(handle: *mut NxrtEpHandle)`
///
/// Releases an EP instance. Must be called exactly once per successful
/// `nxrt_create_ep`. After this call the handle is invalid.
pub type NxrtDestroyEpFn = unsafe extern "C" fn(handle: *mut NxrtEpHandle);

/// `nxrt_ep_name() -> *const c_char`
///
/// Returns a pointer to a null-terminated UTF-8 string with the EP's
/// human-readable name. The pointer is valid for the lifetime of the library.
pub type NxrtEpNameFn = unsafe extern "C" fn() -> *const c_char;

/// `nxrt_device_count(handle: *mut NxrtEpHandle, out_count: *mut u32) -> NxrtStatus`
///
/// Writes the number of devices the EP can dispatch to. Must be ≥ 1 after
/// successful creation.
pub type NxrtDeviceCountFn =
    unsafe extern "C" fn(handle: *mut NxrtEpHandle, out_count: *mut u32) -> NxrtStatus;

// ─── Helpers ────────────────────────────────────────────────────────────────

impl NxrtStatus {
    /// Whether the status indicates success.
    pub fn is_ok(self) -> bool {
        self == Self::Ok
    }

    /// Human-readable description of the status code.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::InvalidArgument => "invalid argument",
            Self::InternalError => "internal error",
            Self::Unsupported => "unsupported",
            Self::OutOfMemory => "out of memory",
        }
    }
}

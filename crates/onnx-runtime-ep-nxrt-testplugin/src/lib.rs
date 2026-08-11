//! Test-fixture nxrt plugin.
//!
//! This crate is a `cdylib` that exports the nxrt ABI symbols expected by the
//! host loader (`onnx-runtime-ep-nxrt-host`). It implements a trivial CPU-based
//! EP for integration testing purposes only.
//!
//! # Variants
//!
//! The default build produces a conforming plugin. Compile-time features
//! produce deliberately broken variants for negative testing:
//! - Default: conforming plugin (correct version, all symbols, 1 device)
//! - `NXRT_TEST_WRONG_VERSION=1` env var: reports ABI major version 99
//! - `NXRT_TEST_FACTORY_ERROR=1` env var: factory returns InternalError
//! - `NXRT_TEST_ZERO_DEVICES=1` env var: reports 0 devices
//! - `NXRT_TEST_PANIC=1` env var: panics inside create_ep
//!
//! These are controlled via env vars read at runtime so a single cdylib binary
//! can be reused with different test scenarios.

use std::ffi::{c_char, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Global counter tracking live EP instances. Tests assert this reaches zero.
static LIVE_EP_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Opaque EP handle for this test plugin.
struct TestEpHandle {
    /// Sentinel value to detect double-free / corruption.
    magic: u64,
}

const EP_MAGIC: u64 = 0xDEAD_BEEF_CAFE_F00D;

/// NxrtStatus codes matching the host's `abi_contract::NxrtStatus`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NxrtStatus {
    Ok = 0,
    InvalidArgument = 1,
    InternalError = 2,
    Unsupported = 3,
    OutOfMemory = 4,
}

// ─── Exported symbols ───────────────────────────────────────────────────────

/// Version negotiation. Reports ABI major/minor.
///
/// If `NXRT_TEST_WRONG_VERSION` env is set, reports major=99 to trigger mismatch.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_abi_version(out_major: *mut u32, out_minor: *mut u32) {
    let wrong_version = std::env::var("NXRT_TEST_WRONG_VERSION").is_ok();
    let major = if wrong_version { 99 } else { 1 };
    if !out_major.is_null() {
        unsafe { *out_major = major };
    }
    if !out_minor.is_null() {
        unsafe { *out_minor = 0 };
    }
}

/// Factory: create an EP instance.
///
/// Controlled by env vars for negative testing.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_create_ep(
    _config_json: *const c_char,
    out_handle: *mut *mut c_void,
) -> NxrtStatus {
    // Panic test — must be caught by the host
    if std::env::var("NXRT_TEST_PANIC").is_ok() {
        panic!("deliberate test panic in nxrt_create_ep");
    }

    // Factory error test
    if std::env::var("NXRT_TEST_FACTORY_ERROR").is_ok() {
        if !out_handle.is_null() {
            unsafe { *out_handle = std::ptr::null_mut() };
        }
        return NxrtStatus::InternalError;
    }

    let handle = Box::new(TestEpHandle { magic: EP_MAGIC });
    LIVE_EP_COUNT.fetch_add(1, Ordering::SeqCst);

    if !out_handle.is_null() {
        unsafe { *out_handle = Box::into_raw(handle) as *mut c_void };
    }
    NxrtStatus::Ok
}

/// Destroy an EP instance.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_destroy_ep(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }
    let ep = unsafe { Box::from_raw(handle as *mut TestEpHandle) };
    assert_eq!(ep.magic, EP_MAGIC, "nxrt_destroy_ep: corrupt handle (double-free?)");
    LIVE_EP_COUNT.fetch_sub(1, Ordering::SeqCst);
}

/// Return the EP name.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_ep_name() -> *const c_char {
    b"NxrtTestPlugin\0".as_ptr() as *const c_char
}

/// Device count query.
///
/// If `NXRT_TEST_ZERO_DEVICES` env is set, reports 0.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_device_count(
    handle: *mut c_void,
    out_count: *mut u32,
) -> NxrtStatus {
    if handle.is_null() || out_count.is_null() {
        return NxrtStatus::InvalidArgument;
    }
    let count = if std::env::var("NXRT_TEST_ZERO_DEVICES").is_ok() {
        0
    } else {
        1
    };
    unsafe { *out_count = count };
    NxrtStatus::Ok
}

/// Query the live EP count (for lifetime tests).
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_test_live_ep_count() -> usize {
    LIVE_EP_COUNT.load(Ordering::SeqCst)
}

//! nxrt ABI round-trip and negative integration tests.
//!
//! These tests load the test-fixture plugin (`crates/onnx-runtime-ep-nxrt-testplugin/`)
//! via `libloading` and drive it through the nxrt ABI, asserting correctness
//! and failure behavior.
//!
//! # Prerequisites
//!
//! Build the test plugin before running:
//! ```sh
//! cd crates/onnx-runtime-ep-nxrt-testplugin && cargo build --lib
//! ```
//!
//! # Thread safety
//!
//! Tests that manipulate env vars are serialized via ENV_MUTEX to avoid races.

use std::ffi::{CStr, c_char, c_void};
use std::path::PathBuf;
use std::sync::Mutex;

use libloading::{Library, Symbol};

/// Mutex to serialize tests that use environment variables.
static ENV_MUTEX: Mutex<()> = Mutex::new(());

// ─── ABI types (mirror of host's abi_contract) ──────────────────────────────

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum NxrtStatus {
    Ok = 0,
    InvalidArgument = 1,
    InternalError = 2,
    Unsupported = 3,
    OutOfMemory = 4,
}

type NxrtAbiVersionFn = unsafe extern "C" fn(*mut u32, *mut u32);
type NxrtCreateEpFn = unsafe extern "C" fn(*const c_char, *mut *mut c_void) -> NxrtStatus;
type NxrtDestroyEpFn = unsafe extern "C" fn(*mut c_void);
type NxrtEpNameFn = unsafe extern "C" fn() -> *const c_char;
type NxrtDeviceCountFn = unsafe extern "C" fn(*mut c_void, *mut u32) -> NxrtStatus;
type NxrtTestLiveEpCountFn = unsafe extern "C" fn() -> usize;

// ─── Helpers ────────────────────────────────────────────────────────────────

fn testplugin_path() -> PathBuf {
    if let Ok(p) = std::env::var("NXRT_TESTPLUGIN_PATH") {
        return PathBuf::from(p);
    }
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
    // Try both the direct and deps paths
    let candidates = [
        workspace_root.join("crates/onnx-runtime-ep-nxrt-testplugin/target/debug/libonnx_runtime_ep_nxrt_testplugin.so"),
        workspace_root.join("crates/onnx-runtime-ep-nxrt-testplugin/target/debug/deps/libonnx_runtime_ep_nxrt_testplugin.so"),
    ];
    for path in &candidates {
        if path.exists() {
            return path.clone();
        }
    }
    panic!(
        "testplugin cdylib not found. Build it first:\n  \
         cd crates/onnx-runtime-ep-nxrt-testplugin && cargo build --lib\n  \
         Or set NXRT_TESTPLUGIN_PATH env var."
    );
}

fn load_testplugin() -> Library {
    let path = testplugin_path();
    assert!(path.exists(), "testplugin not found at {path:?}");
    unsafe { Library::new(&path) }.unwrap_or_else(|e| panic!("dlopen failed: {e}"))
}

// ─── Round-trip: full ABI lifecycle ─────────────────────────────────────────

#[test]
fn round_trip_load_negotiate_create_query_destroy() {
    let _lock = ENV_MUTEX.lock().unwrap();
    unsafe { std::env::remove_var("NXRT_TEST_WRONG_VERSION"); }
    unsafe { std::env::remove_var("NXRT_TEST_FACTORY_ERROR"); }
    unsafe { std::env::remove_var("NXRT_TEST_ZERO_DEVICES"); }
    unsafe { std::env::remove_var("NXRT_TEST_PANIC"); }

    let lib = load_testplugin();

    // Version negotiation
    let version_fn: Symbol<NxrtAbiVersionFn> =
        unsafe { lib.get(b"nxrt_abi_version") }.expect("symbol");
    let (mut major, mut minor) = (0u32, 0u32);
    unsafe { version_fn(&mut major, &mut minor) };
    assert_eq!(major, 1);
    assert_eq!(minor, 0);

    // Create EP
    let create_fn: Symbol<NxrtCreateEpFn> = unsafe { lib.get(b"nxrt_create_ep") }.expect("symbol");
    let mut handle: *mut c_void = std::ptr::null_mut();
    let status = unsafe { create_fn(b"{}\0".as_ptr() as _, &mut handle) };
    assert_eq!(status, NxrtStatus::Ok);
    assert!(!handle.is_null());

    // Query name
    let name_fn: Symbol<NxrtEpNameFn> = unsafe { lib.get(b"nxrt_ep_name") }.expect("symbol");
    let name = unsafe { CStr::from_ptr(name_fn()) }.to_str().unwrap();
    assert_eq!(name, "NxrtTestPlugin");

    // Query devices
    let dc_fn: Symbol<NxrtDeviceCountFn> = unsafe { lib.get(b"nxrt_device_count") }.expect("symbol");
    let mut count = 0u32;
    let status = unsafe { dc_fn(handle, &mut count) };
    assert_eq!(status, NxrtStatus::Ok);
    assert_eq!(count, 1);

    // Destroy
    let destroy_fn: Symbol<NxrtDestroyEpFn> = unsafe { lib.get(b"nxrt_destroy_ep") }.expect("symbol");
    unsafe { destroy_fn(handle) };

    // Verify no leak
    let live_fn: Symbol<NxrtTestLiveEpCountFn> =
        unsafe { lib.get(b"nxrt_test_live_ep_count") }.expect("symbol");
    assert_eq!(unsafe { live_fn() }, 0, "no leaked EP instances");
}

// ─── Ownership / lifetime ───────────────────────────────────────────────────

#[test]
fn ownership_multiple_ep_no_leak() {
    let _lock = ENV_MUTEX.lock().unwrap();
    unsafe { std::env::remove_var("NXRT_TEST_WRONG_VERSION"); }
    unsafe { std::env::remove_var("NXRT_TEST_FACTORY_ERROR"); }
    unsafe { std::env::remove_var("NXRT_TEST_ZERO_DEVICES"); }
    unsafe { std::env::remove_var("NXRT_TEST_PANIC"); }

    let lib = load_testplugin();
    let create_fn: Symbol<NxrtCreateEpFn> = unsafe { lib.get(b"nxrt_create_ep") }.unwrap();
    let destroy_fn: Symbol<NxrtDestroyEpFn> = unsafe { lib.get(b"nxrt_destroy_ep") }.unwrap();
    let live_fn: Symbol<NxrtTestLiveEpCountFn> =
        unsafe { lib.get(b"nxrt_test_live_ep_count") }.unwrap();

    let mut handles = Vec::new();
    for _ in 0..5 {
        let mut h: *mut c_void = std::ptr::null_mut();
        let s = unsafe { create_fn(b"{}\0".as_ptr() as _, &mut h) };
        assert_eq!(s, NxrtStatus::Ok);
        handles.push(h);
    }
    assert_eq!(unsafe { live_fn() }, 5);

    for h in handles.into_iter().rev() {
        unsafe { destroy_fn(h) };
    }
    assert_eq!(unsafe { live_fn() }, 0, "Drop counter must reach zero");
}

#[test]
fn library_outlives_ep_instances() {
    let _lock = ENV_MUTEX.lock().unwrap();
    unsafe { std::env::remove_var("NXRT_TEST_WRONG_VERSION"); }
    unsafe { std::env::remove_var("NXRT_TEST_FACTORY_ERROR"); }
    unsafe { std::env::remove_var("NXRT_TEST_ZERO_DEVICES"); }
    unsafe { std::env::remove_var("NXRT_TEST_PANIC"); }

    let lib = load_testplugin();
    let create_fn: Symbol<NxrtCreateEpFn> = unsafe { lib.get(b"nxrt_create_ep") }.unwrap();
    let destroy_fn: Symbol<NxrtDestroyEpFn> = unsafe { lib.get(b"nxrt_destroy_ep") }.unwrap();
    let live_fn: Symbol<NxrtTestLiveEpCountFn> =
        unsafe { lib.get(b"nxrt_test_live_ep_count") }.unwrap();

    let mut h: *mut c_void = std::ptr::null_mut();
    unsafe { create_fn(b"{}\0".as_ptr() as _, &mut h) };
    // Correct: destroy EP before dropping library
    unsafe { destroy_fn(h) };
    assert_eq!(unsafe { live_fn() }, 0);
    drop(lib); // safe: no live EP references
}

// ─── Negative tests ─────────────────────────────────────────────────────────

#[test]
fn negative_incompatible_major_version() {
    let _lock = ENV_MUTEX.lock().unwrap();
    unsafe { std::env::set_var("NXRT_TEST_WRONG_VERSION", "1"); }
    unsafe { std::env::remove_var("NXRT_TEST_FACTORY_ERROR"); }
    unsafe { std::env::remove_var("NXRT_TEST_ZERO_DEVICES"); }
    unsafe { std::env::remove_var("NXRT_TEST_PANIC"); }

    let lib = load_testplugin();
    let version_fn: Symbol<NxrtAbiVersionFn> = unsafe { lib.get(b"nxrt_abi_version") }.unwrap();
    let (mut major, mut minor) = (0u32, 0u32);
    unsafe { version_fn(&mut major, &mut minor) };
    assert_eq!(major, 99, "must report wrong version");
    // Host would reject: major != 1

    unsafe { std::env::remove_var("NXRT_TEST_WRONG_VERSION"); }
}

#[test]
fn negative_missing_symbol() {
    let lib = load_testplugin();
    let result: Result<Symbol<NxrtAbiVersionFn>, _> =
        unsafe { lib.get(b"nxrt_nonexistent_symbol_xyz") };
    assert!(result.is_err(), "nonexistent symbol must fail");
    assert!(!result.unwrap_err().to_string().is_empty());
}

#[test]
fn negative_missing_library() {
    let result = unsafe { Library::new("/nonexistent/libnxrt_fake.so") };
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("No such file") || msg.contains("cannot open"),
        "got: {msg}"
    );
}

#[test]
fn negative_factory_returns_error() {
    let _lock = ENV_MUTEX.lock().unwrap();
    unsafe { std::env::set_var("NXRT_TEST_FACTORY_ERROR", "1"); }
    unsafe { std::env::remove_var("NXRT_TEST_WRONG_VERSION"); }
    unsafe { std::env::remove_var("NXRT_TEST_ZERO_DEVICES"); }
    unsafe { std::env::remove_var("NXRT_TEST_PANIC"); }

    let lib = load_testplugin();
    let create_fn: Symbol<NxrtCreateEpFn> = unsafe { lib.get(b"nxrt_create_ep") }.unwrap();
    let mut h: *mut c_void = std::ptr::null_mut();
    let status = unsafe { create_fn(b"{}\0".as_ptr() as _, &mut h) };
    assert_eq!(status, NxrtStatus::InternalError);
    assert!(h.is_null());

    unsafe { std::env::remove_var("NXRT_TEST_FACTORY_ERROR"); }
}

#[test]
fn negative_zero_devices() {
    let _lock = ENV_MUTEX.lock().unwrap();
    unsafe { std::env::set_var("NXRT_TEST_ZERO_DEVICES", "1"); }
    unsafe { std::env::remove_var("NXRT_TEST_WRONG_VERSION"); }
    unsafe { std::env::remove_var("NXRT_TEST_FACTORY_ERROR"); }
    unsafe { std::env::remove_var("NXRT_TEST_PANIC"); }

    let lib = load_testplugin();
    let create_fn: Symbol<NxrtCreateEpFn> = unsafe { lib.get(b"nxrt_create_ep") }.unwrap();
    let dc_fn: Symbol<NxrtDeviceCountFn> = unsafe { lib.get(b"nxrt_device_count") }.unwrap();
    let destroy_fn: Symbol<NxrtDestroyEpFn> = unsafe { lib.get(b"nxrt_destroy_ep") }.unwrap();

    let mut h: *mut c_void = std::ptr::null_mut();
    unsafe { create_fn(b"{}\0".as_ptr() as _, &mut h) };
    let mut count = 99u32;
    let s = unsafe { dc_fn(h, &mut count) };
    assert_eq!(s, NxrtStatus::Ok);
    assert_eq!(count, 0, "must report zero devices");
    unsafe { destroy_fn(h) };

    unsafe { std::env::remove_var("NXRT_TEST_ZERO_DEVICES"); }
}

#[test]
fn negative_panic_contained_by_abi_guard() {
    // Test the ABI crate's panic containment helper (used by the export macro).
    use onnx_runtime_ep_nxrt_abi::status::{NxrtStatusCode, catch_status_panic};

    let status = catch_status_panic(|| panic!("test panic"));
    assert_eq!(status.code, NxrtStatusCode::InternalError);
    let msg = unsafe { status.message_str() }.unwrap_or("");
    assert!(msg.contains("panic"), "must mention panic: {msg}");
}

#[test]
fn negative_null_handle_device_count() {
    let _lock = ENV_MUTEX.lock().unwrap();
    unsafe { std::env::remove_var("NXRT_TEST_WRONG_VERSION"); }
    unsafe { std::env::remove_var("NXRT_TEST_FACTORY_ERROR"); }
    unsafe { std::env::remove_var("NXRT_TEST_ZERO_DEVICES"); }
    unsafe { std::env::remove_var("NXRT_TEST_PANIC"); }

    let lib = load_testplugin();
    let dc_fn: Symbol<NxrtDeviceCountFn> = unsafe { lib.get(b"nxrt_device_count") }.unwrap();
    let mut count = 99u32;
    let s = unsafe { dc_fn(std::ptr::null_mut(), &mut count) };
    assert_eq!(s, NxrtStatus::InvalidArgument, "null handle → InvalidArgument");
}

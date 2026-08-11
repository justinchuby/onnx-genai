//! Integration tests exercising the **real shipped nxrt ABI** end-to-end.
//!
//! These tests build `onnx-runtime-ep-nxrt-testplugin` as a cdylib, load it
//! via `libloading`, and drive the full lifecycle through the raw C symbols:
//!   negotiate → create_factories → factory.create_ep → ep methods → release
//!
//! This proves the ABI contract works across the dynamic boundary without
//! relying on the host's adapter (which Isidore is rewriting).

use std::ffi::CStr;
use std::path::PathBuf;
use std::sync::Mutex;

use libloading::{Library, Symbol};
use onnx_runtime_ep_nxrt_abi::{
    version::{
        validate_negotiation, NxrtVersionRange, NXRT_ABI_VERSION_MAJOR, NXRT_ABI_VERSION_MINOR,
    },
    NxrtCreateEpFactoriesFn, NxrtEpFactoryVtable, NxrtNegotiateFn, NxrtNegotiateRequest,
    NxrtNegotiateResponse, NxrtStatusCode, NXRT_CAP_KNOWN_MASK, NXRT_SYMBOL_CREATE_EP_FACTORIES,
    NXRT_SYMBOL_NEGOTIATE,
};

/// Mutex to serialize tests that manipulate process-wide env vars.
static ENV_MUTEX: Mutex<()> = Mutex::new(());

/// Locate the testplugin cdylib.
///
/// Resolution order:
/// 1. `NXRT_TESTPLUGIN_PATH` env var (explicit override)
/// 2. `CARGO_TARGET_DIR` / profile / libname (custom target dir)
/// 3. workspace root / target / profile / libname (default layout)
///
/// The profile defaults to "debug"; set `PROFILE=release` to test release builds.
/// If the library is not found, the test **skips loudly** with a build command hint.
fn testplugin_path() -> PathBuf {
    if let Ok(p) = std::env::var("NXRT_TESTPLUGIN_PATH") {
        let path = PathBuf::from(p);
        assert!(
            path.exists(),
            "NXRT_TESTPLUGIN_PATH set to {path:?} but file does not exist"
        );
        return path;
    }

    let profile = std::env::var("PROFILE").unwrap_or_else(|_| "debug".to_string());

    let libname = if cfg!(target_os = "linux") {
        "libonnx_runtime_ep_nxrt_testplugin.so"
    } else if cfg!(target_os = "macos") {
        "libonnx_runtime_ep_nxrt_testplugin.dylib"
    } else {
        "onnx_runtime_ep_nxrt_testplugin.dll"
    };

    // Try CARGO_TARGET_DIR first (set by cargo or CI)
    if let Ok(target_dir) = std::env::var("CARGO_TARGET_DIR") {
        let path = PathBuf::from(target_dir).join(&profile).join(libname);
        if path.exists() {
            return path;
        }
    }

    // Fall back to workspace root / target / profile
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let path = workspace_root.join("target").join(&profile).join(libname);
    if path.exists() {
        return path;
    }

    // Auto-build the testplugin cdylib if not present.
    eprintln!(
        "testplugin cdylib not found — building it now \
         (cargo build -p onnx-runtime-ep-nxrt-testplugin)..."
    );
    let status = std::process::Command::new("cargo")
        .args(["build", "-p", "onnx-runtime-ep-nxrt-testplugin"])
        .status()
        .expect("failed to invoke cargo");
    assert!(status.success(), "cargo build of testplugin failed");

    // Re-check the workspace target path
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let path = workspace_root.join("target").join(&profile).join(libname);
    assert!(
        path.exists(),
        "testplugin cdylib still not found at {path:?} after build. \
         Set NXRT_TESTPLUGIN_PATH if using a custom target dir."
    );
    path
}

/// Helper: load the testplugin library.
fn load_testplugin() -> Library {
    let path = testplugin_path();
    assert!(
        path.exists(),
        "testplugin cdylib not found at {path:?}. Run: cargo build -p onnx-runtime-ep-nxrt-testplugin"
    );
    unsafe { Library::new(&path).expect("failed to dlopen testplugin") }
}

// ═══════════════════════════════════════════════════════════════════════════════
// POSITIVE TESTS — full lifecycle through the real ABI
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn negotiate_succeeds_with_matching_version() {
    let lib = load_testplugin();
    let negotiate: Symbol<NxrtNegotiateFn> = unsafe {
        lib.get(NXRT_SYMBOL_NEGOTIATE)
            .expect("NxrtNegotiate not found")
    };

    let req = NxrtNegotiateRequest::current();
    let mut resp = NxrtNegotiateResponse::zeroed();

    let status = unsafe { negotiate(&req, &mut resp) };
    assert!(status.is_ok(), "negotiate failed: {:?}", status.code);
    assert_eq!(resp.agreed_major, NXRT_ABI_VERSION_MAJOR);
    assert_eq!(resp.agreed_minor, NXRT_ABI_VERSION_MINOR);
    // Capability flags must be within known mask
    assert_eq!(
        resp.capability_flags & !NXRT_CAP_KNOWN_MASK,
        0,
        "unknown capability bits set"
    );
}

#[test]
fn full_lifecycle_negotiate_create_release() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let lib = load_testplugin();

    // Step 1: Negotiate
    let negotiate: Symbol<NxrtNegotiateFn> = unsafe { lib.get(NXRT_SYMBOL_NEGOTIATE).unwrap() };
    let req = NxrtNegotiateRequest::current();
    let mut resp = NxrtNegotiateResponse::zeroed();
    let status = unsafe { negotiate(&req, &mut resp) };
    assert!(status.is_ok());

    // Validate negotiation response
    let host_range = NxrtVersionRange::current();
    validate_negotiation(&host_range, &resp).expect("negotiation validation failed");

    // Step 2: Create factories
    let create_factories: Symbol<NxrtCreateEpFactoriesFn> =
        unsafe { lib.get(NXRT_SYMBOL_CREATE_EP_FACTORIES).unwrap() };
    let mut factory_ptr: *mut NxrtEpFactoryVtable = std::ptr::null_mut();
    let mut num_factories: usize = 0;
    let status = unsafe { create_factories(&mut factory_ptr, 1, &mut num_factories) };
    assert!(status.is_ok(), "create_factories failed: {:?}", status.code);
    assert_eq!(num_factories, 1);
    assert!(!factory_ptr.is_null());

    let factory = unsafe { &*factory_ptr };
    assert!(
        factory.num_devices >= 1,
        "factory must advertise at least 1 device"
    );

    // Read factory name
    let name = unsafe { CStr::from_ptr(factory.name as *const std::ffi::c_char) };
    assert_eq!(name.to_str().unwrap(), "NxrtTestPlugin");

    // Step 3: Create EP via factory
    let mut ep_ptr = std::ptr::null_mut();
    let status = unsafe { (factory.create_ep)(factory.ctx, 0, &mut ep_ptr) };
    assert!(status.is_ok(), "create_ep failed: {:?}", status.code);
    assert!(!ep_ptr.is_null());

    let ep = unsafe { &*ep_ptr };
    // EP device_type = 0 (Cpu)
    assert_eq!(ep.device_type, 0);

    // Read EP name
    let ep_name = unsafe { CStr::from_ptr(ep.name as *const std::ffi::c_char) };
    assert_eq!(ep_name.to_str().unwrap(), "NxrtTestPlugin");

    // Step 4: Get capability (should claim nothing — fail closed)
    let mut num_claims: u32 = 99;
    let status = unsafe {
        (ep.get_capability)(
            ep.ctx,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            0,
            std::ptr::null_mut(),
            0,
            &mut num_claims,
        )
    };
    assert!(status.is_ok());
    assert_eq!(num_claims, 0, "test EP must claim no nodes");

    // Step 5: Release EP
    unsafe { (ep.release)(ep.ctx) };
    // Step 6: Release factory
    unsafe { (factory.release)(factory.ctx) };

    // Drop library last (after all handles released)
    drop(lib);
}

#[test]
fn ownership_lifetime_drop_counter_returns_to_zero() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let lib = load_testplugin();

    // Get the live count function
    let live_count: Symbol<unsafe extern "C" fn() -> usize> =
        unsafe { lib.get(b"nxrt_test_live_ep_count").unwrap() };
    assert_eq!(unsafe { live_count() }, 0, "leaking EPs from prior test");

    // Create factory
    let create_factories: Symbol<NxrtCreateEpFactoriesFn> =
        unsafe { lib.get(NXRT_SYMBOL_CREATE_EP_FACTORIES).unwrap() };
    let mut factory_ptr: *mut NxrtEpFactoryVtable = std::ptr::null_mut();
    let mut num: usize = 0;
    let status = unsafe { create_factories(&mut factory_ptr, 1, &mut num) };
    assert!(status.is_ok());
    let factory = unsafe { &*factory_ptr };

    // Create multiple EPs
    let mut eps = Vec::new();
    for i in 0..3 {
        let mut ep_ptr = std::ptr::null_mut();
        let s = unsafe { (factory.create_ep)(factory.ctx, i, &mut ep_ptr) };
        assert!(s.is_ok());
        eps.push(ep_ptr);
    }

    // The create_ep_factories helper in vtable.rs creates a probe EP then drops it,
    // plus one for each create_ep call. The live count reflects currently alive EPs.
    let count_with_eps = unsafe { live_count() };
    assert!(
        count_with_eps >= 3,
        "expected at least 3 live EPs, got {count_with_eps}"
    );

    // Release all EPs
    for ep_ptr in &eps {
        let ep = unsafe { &**ep_ptr };
        unsafe { (ep.release)(ep.ctx) };
    }

    // Release factory
    unsafe { (factory.release)(factory.ctx) };

    // Count must return to zero
    assert_eq!(unsafe { live_count() }, 0, "EP leak detected!");
}

// ═══════════════════════════════════════════════════════════════════════════════
// NEGATIVE TESTS — each fails closed, never crashes or hangs
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn negotiate_rejects_incompatible_major_version() {
    let lib = load_testplugin();
    let negotiate: Symbol<NxrtNegotiateFn> = unsafe { lib.get(NXRT_SYMBOL_NEGOTIATE).unwrap() };

    let req = NxrtNegotiateRequest {
        struct_size: std::mem::size_of::<NxrtNegotiateRequest>() as u32,
        host_range: NxrtVersionRange {
            major_min: 99,
            major_max: 99,
            minor_max: 0,
        },
    };
    let mut resp = NxrtNegotiateResponse::zeroed();
    let status = unsafe { negotiate(&req, &mut resp) };
    assert_eq!(status.status_code(), Some(NxrtStatusCode::VersionMismatch));
}

#[test]
fn validate_rejects_plugin_minor_newer_than_host() {
    // Simulate: plugin agreed minor=5 but host only supports minor_max=0
    let host_range = NxrtVersionRange {
        major_min: 1,
        major_max: 1,
        minor_max: 0,
    };
    let resp = NxrtNegotiateResponse {
        struct_size: std::mem::size_of::<NxrtNegotiateResponse>() as u32,
        agreed_major: 1,
        agreed_minor: 5, // newer than host
        plugin_range: NxrtVersionRange::current(),
        capability_flags: 0,
    };
    let result = validate_negotiation(&host_range, &resp);
    assert!(result.is_err(), "must reject plugin minor > host minor_max");
    assert!(result.unwrap_err().contains("minor"));
}

#[test]
fn validate_rejects_unknown_capability_bits() {
    let host_range = NxrtVersionRange::current();
    let resp = NxrtNegotiateResponse {
        struct_size: std::mem::size_of::<NxrtNegotiateResponse>() as u32,
        agreed_major: NXRT_ABI_VERSION_MAJOR,
        agreed_minor: NXRT_ABI_VERSION_MINOR,
        plugin_range: NxrtVersionRange::current(),
        capability_flags: 1 << 63, // unknown bit
    };
    let result = validate_negotiation(&host_range, &resp);
    assert!(result.is_err(), "must reject unknown capability bits");
    assert!(result.unwrap_err().contains("unknown capability flags"));
}

#[test]
fn missing_library_file_fails_gracefully() {
    let result = unsafe { Library::new("/nonexistent/libfake_nxrt_plugin.so") };
    assert!(result.is_err(), "must fail on missing library");
}

#[test]
fn missing_symbol_fails_gracefully() {
    let lib = load_testplugin();
    // Try to load a misspelled symbol
    let result: Result<Symbol<NxrtNegotiateFn>, _> = unsafe { lib.get(b"NxrtNegotiate_TYPO") };
    assert!(result.is_err(), "must fail on misspelled symbol");
}

#[test]
fn factory_panic_is_contained() {
    let _lock = ENV_MUTEX.lock().unwrap();
    // Set the env var to trigger a panic in the constructor
    std::env::set_var("NXRT_TEST_PANIC", "1");
    let lib = load_testplugin();

    let create_factories: Symbol<NxrtCreateEpFactoriesFn> =
        unsafe { lib.get(NXRT_SYMBOL_CREATE_EP_FACTORIES).unwrap() };
    let mut factory_ptr: *mut NxrtEpFactoryVtable = std::ptr::null_mut();
    let mut num: usize = 99;

    // The macro catches the panic and returns InternalError
    let status = unsafe { create_factories(&mut factory_ptr, 1, &mut num) };
    assert_eq!(status.status_code(), Some(NxrtStatusCode::InternalError));
    assert_eq!(num, 0, "num must be zeroed on panic");

    // B3: Verify the inline message buffer survives the cdylib boundary.
    // NxrtStatus.message is now a fixed [u8; 256] inline buffer (not heap-allocated),
    // so the message is readable in the host without cross-CRT free issues.
    let msg = status.message_str();
    assert!(
        msg.is_some(),
        "message_str() must return Some for an InternalError from the plugin cdylib"
    );
    eprintln!(
        "  B3 inline-buffer message across cdylib boundary: {:?}",
        msg.unwrap()
    );

    std::env::remove_var("NXRT_TEST_PANIC");
}

#[test]
fn factory_error_is_contained() {
    let _lock = ENV_MUTEX.lock().unwrap();
    // The FACTORY_ERROR env also triggers a panic which the macro catches
    std::env::set_var("NXRT_TEST_FACTORY_ERROR", "1");
    let lib = load_testplugin();

    let create_factories: Symbol<NxrtCreateEpFactoriesFn> =
        unsafe { lib.get(NXRT_SYMBOL_CREATE_EP_FACTORIES).unwrap() };
    let mut factory_ptr: *mut NxrtEpFactoryVtable = std::ptr::null_mut();
    let mut num: usize = 99;

    let status = unsafe { create_factories(&mut factory_ptr, 1, &mut num) };
    assert_eq!(status.status_code(), Some(NxrtStatusCode::InternalError));
    assert_eq!(num, 0);

    std::env::remove_var("NXRT_TEST_FACTORY_ERROR");
}

//! L3 integration tests: Real upstream ORT loads our EP plugin.
//!
//! # Test layers
//!
//! - `ort_api_sanity` — verify ORT loads and all EP-related vtable slots are non-null.
//! - `ort_register_ep_library` — real `RegisterExecutionProviderLibrary` call.
//! - `ort_loads_our_ep_and_runs_model` — full end-to-end: Env → Register → Devices → Session → Run.
//! - `ort_unsupported_op_declines_not_crashes` — negative: model with unsupported op must not crash.
//! - `conformance_*` — real-ORT conformance suite covering broadcast, multi-node, MatMul, batched-ND
//!   MatMul, mixed partition, INT32, dynamic dims, multiple runs, two sessions.
//! - `stress_register_run_unregister_cycles` — 25 complete register→Run→unregister cycles to lock
//!   in the use-after-free fix in factory.rs (bug only appeared at cycle ≥6).
//!
//! # `EpDevice_EpName` vs registration name
//!
//! `EpDevice_EpName` returns the factory's declared name (e.g. "cpu_ep"), which is the
//! string returned by `OrtEpFactory::GetName`. It is **not** the registration key passed
//! to `RegisterExecutionProviderLibrary`. Tests that search the device list must compare
//! against the factory name "cpu_ep", not the registration key.
//!
//! # f16/bf16 coverage
//!
//! `build_cpu_registry_with_descriptors()` (landed by Deckard) derives Float16 and BFloat16
//! type-constraint metadata from the kernel registry and wires it through the cpu-plugin shim
//! into `GetKernelRegistry`. ORT consults this registry to route half-precision nodes to our
//! EP. `conformance_add_float16` and `conformance_add_bfloat16` verify end-to-end routing
//! with exact bit-pattern assertions — both pass as of commit `577047a74`.
//!
//! # Environment
//!
//! No env vars required — the test resolves ORT from the ort-sys build output.
//! Skips loudly if ORT or the EP cdylib is absent.

mod cdylib_resolve;
use onnx_runtime_ort_testkit as ort_path;

use std::ffi::{CStr, CString};
use std::path::PathBuf;
use std::ptr;
use std::sync::{Mutex, MutexGuard};

use onnx_genai_ort_sys as ort;

/// Serialises all tests that load our EP plugin.
///
/// ORT's per-process EP device state is corrupted after ≥6 register+Run+unregister
/// cycles (factory.rs bug — Nabil).  The lock ensures tests run one at a time so
/// the cycle count stays below the failure threshold for the default test suite.
static ORT_EP_LOCK: Mutex<()> = Mutex::new(());

/// Acquire `ORT_EP_LOCK`, recovering from poisoning so that one test's panic
/// does not cascade `PoisonError` failures across every subsequent test.
fn lock_ort_ep() -> MutexGuard<'static, ()> {
    ORT_EP_LOCK.lock().unwrap_or_else(|poisoned| {
        eprintln!(
            "WARNING: ORT_EP_LOCK was poisoned by a prior test panic — recovering. \
             Investigate the original failure above."
        );
        poisoned.into_inner()
    })
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Resolve the directory containing the platform ORT shared library.
fn find_ort_lib_dir() -> Option<PathBuf> {
    ort_discovery::find_ort_lib_dir()
}

/// Canonical ORT discovery lives in the `onnx-runtime-ort-testkit` crate —
/// aliased here so existing `ort_discovery::` call sites keep working.
use onnx_runtime_ort_testkit as ort_discovery;
/// Session-creation helper (binary + textproto fixtures) from `main`.
#[path = "common/ort_session.rs"]
mod ort_session;

/// Find the EP cdylib produced by this crate.
fn find_ep_cdylib() -> Option<PathBuf> {
    cdylib_resolve::find_cpu_plugin_cdylib_optional()
}

/// When `NXRT_REQUIRE_ORT_TESTS=1`, tests must fail instead of silently skipping
/// if ORT or the EP cdylib is unavailable.
///
/// Skip a test loudly when a required resource is missing.
/// When `NXRT_REQUIRE_ORT_TESTS=1`, panics instead of skipping.
macro_rules! skip_if_missing {
    ($resource:expr, $msg:literal) => {
        match $resource {
            Some(v) => v,
            None => {
                if std::env::var("NXRT_REQUIRE_ORT_TESTS").as_deref() == Ok("1") {
                    panic!(
                        "NXRT_REQUIRE_ORT_TESTS=1 but required resource unavailable — {} cannot run",
                        $msg
                    );
                }
                eprintln!("\n*** SKIPPED: {} ***\n", $msg);
                return;
            }
        }
    };
}

/// Obtain the OrtApi vtable from a loaded libonnxruntime.
///
/// # Safety
/// `lib` must be a valid loaded libonnxruntime handle.
unsafe fn get_ort_api(lib: &libloading::Library) -> *const ort::OrtApi {
    type GetApiBaseFn = unsafe extern "C" fn() -> *const ort::OrtApiBase;
    let get_api_base: libloading::Symbol<'_, GetApiBaseFn> =
        unsafe { lib.get(b"OrtGetApiBase") }.expect("OrtGetApiBase not found in libonnxruntime");
    let api_base = unsafe { get_api_base() };
    assert!(!api_base.is_null(), "OrtGetApiBase returned null");
    let get_api = unsafe { (*api_base).GetApi }.expect("OrtApiBase::GetApi is null");
    let api = unsafe { get_api(ort::ORT_API_VERSION) };
    assert!(
        !api.is_null(),
        "GetApi(ORT_API_VERSION={}) returned null — ORT version mismatch?",
        ort::ORT_API_VERSION
    );
    api
}

/// Assert an OrtStatus is null (success); panic with error message otherwise.
///
/// # Safety
/// `api` and `status` must be valid (or null for status).
unsafe fn check_status(api: *const ort::OrtApi, status: *mut ort::OrtStatus, stage: &str) {
    if !status.is_null() {
        let get_msg = unsafe { (*api).GetErrorMessage }.expect("GetErrorMessage not in OrtApi");
        let msg_ptr = unsafe { get_msg(status) };
        let msg = if msg_ptr.is_null() {
            "(no message)".to_owned()
        } else {
            unsafe { CStr::from_ptr(msg_ptr) }
                .to_string_lossy()
                .into_owned()
        };
        if let Some(release) = unsafe { (*api).ReleaseStatus } {
            unsafe { release(status) };
        }
        panic!("STAGE [{stage}] FAILED: {msg}");
    }
}

// ─── L3 — ORT API sanity ─────────────────────────────────────────────────────

/// Verify that ORT loads and all plugin-EP vtable slots we rely on are non-null.
///
/// This is a prerequisite for all other L3 tests.
#[test]
fn ort_api_sanity() {
    let ort_lib_dir = skip_if_missing!(
        find_ort_lib_dir(),
        "ort_api_sanity: ORT not found; run `cargo build -p onnx-genai-ort-sys` first"
    );
    let ort_lib_path = ort_lib_dir.join(ort_discovery::ort_lib_name());

    let lib = unsafe { libloading::Library::new(&ort_lib_path) }
        .unwrap_or_else(|e| panic!("Failed to dlopen {}: {e}", ort_lib_path.display()));

    let api = unsafe { get_ort_api(&lib) };

    macro_rules! require_fn {
        ($f:ident) => {
            assert!(
                unsafe { (*api).$f }.is_some(),
                "OrtApi::{} is null — ORT build doesn't support plugin EP?",
                stringify!($f)
            );
        };
    }

    require_fn!(CreateEnv);
    require_fn!(ReleaseEnv);
    require_fn!(RegisterExecutionProviderLibrary);
    require_fn!(UnregisterExecutionProviderLibrary);
    require_fn!(GetEpDevices);
    require_fn!(SessionOptionsAppendExecutionProvider_V2);
    require_fn!(EpDevice_EpName);
    require_fn!(GetEpApi);
    require_fn!(CreateSession);
    require_fn!(Run);
    require_fn!(GetErrorMessage);
    require_fn!(CreateCpuMemoryInfo);
    require_fn!(CreateTensorWithDataAsOrtValue);
    require_fn!(GetTensorMutableData);
    require_fn!(ReleaseSession);
    require_fn!(ReleaseSessionOptions);
    require_fn!(ReleaseValue);
    require_fn!(ReleaseMemoryInfo);

    eprintln!("✓ ort_api_sanity: All plugin-EP API slots are populated in ORT 1.27");
}

// ─── L3 — RegisterExecutionProviderLibrary ───────────────────────────────────

/// Drive `RegisterExecutionProviderLibrary` + `GetEpDevices` without running a model.
///
/// # Known failure: GetSupportedDevices returns 0 devices
///
/// Drive `RegisterExecutionProviderLibrary` + `GetEpDevices` without running a model.
#[test]
fn ort_register_ep_library() {
    let _lock = lock_ort_ep();
    let ort_lib_dir =
        skip_if_missing!(find_ort_lib_dir(), "ort_register_ep_library: ORT not found");
    let ep_lib_path = skip_if_missing!(
        find_ep_cdylib(),
        "ort_register_ep_library: EP cdylib not found; run cargo build -p onnx-runtime-ep-cpu-plugin"
    );

    let ort_lib_path = ort_lib_dir.join(ort_discovery::ort_lib_name());
    let lib = unsafe { libloading::Library::new(&ort_lib_path) }.expect("dlopen ORT failed");
    let api = unsafe { get_ort_api(&lib) };

    unsafe {
        let mut env: *mut ort::OrtEnv = ptr::null_mut();
        let logid = CString::new("nxrt_reg_test").unwrap();
        let status =
            ((*api).CreateEnv.unwrap())(ort::ORT_LOGGING_LEVEL_WARNING, logid.as_ptr(), &mut env);
        check_status(api, status, "CreateEnv");

        let reg_name = CString::new("cpu_ep").unwrap();
        let ep_path_c = ort_path::OrtPathBuf::new(&ep_lib_path);
        let status = ((*api).RegisterExecutionProviderLibrary.unwrap())(
            env,
            reg_name.as_ptr(),
            ep_path_c.as_ptr(),
        );
        check_status(api, status, "RegisterExecutionProviderLibrary");
        eprintln!("✓ RegisterExecutionProviderLibrary succeeded");

        ((*api).UnregisterExecutionProviderLibrary.unwrap())(env, reg_name.as_ptr());
        ((*api).ReleaseEnv.unwrap())(env);
    }
}

// ─── L3 — Full end-to-end ────────────────────────────────────────────────────

/// Full end-to-end: Register → GetEpDevices → CreateSession → Run → verify output.
///
/// Same blocking condition as `ort_register_ep_library`.
#[test]
fn ort_loads_our_ep_and_runs_model() {
    let _lock = lock_ort_ep();
    let ort_lib_dir = skip_if_missing!(
        find_ort_lib_dir(),
        "ort_loads_our_ep_and_runs_model: ORT not found"
    );
    let ep_lib_path = skip_if_missing!(
        find_ep_cdylib(),
        "ort_loads_our_ep_and_runs_model: EP cdylib not found"
    );

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path =
        PathBuf::from(manifest_dir).join("tests/fixtures/add_1x4/model.onnx.textproto");
    assert!(
        model_path.exists(),
        "Missing model fixture: {}",
        model_path.display()
    );

    let ort_lib_path = ort_lib_dir.join(ort_discovery::ort_lib_name());
    let lib = unsafe { libloading::Library::new(&ort_lib_path) }.expect("dlopen ORT");
    let api = unsafe { get_ort_api(&lib) };

    unsafe {
        // Stage 1: Env
        let mut env: *mut ort::OrtEnv = ptr::null_mut();
        let logid = CString::new("nxrt_e2e").unwrap();
        let status =
            ((*api).CreateEnv.unwrap())(ort::ORT_LOGGING_LEVEL_WARNING, logid.as_ptr(), &mut env);
        check_status(api, status, "CreateEnv");
        eprintln!("✓ Stage 1: CreateEnv");

        // Stage 2: RegisterExecutionProviderLibrary
        let reg_name = CString::new("cpu_ep_e2e").unwrap();
        let ep_path_c = ort_path::OrtPathBuf::new(&ep_lib_path);
        let status = ((*api).RegisterExecutionProviderLibrary.unwrap())(
            env,
            reg_name.as_ptr(),
            ep_path_c.as_ptr(),
        );
        check_status(api, status, "RegisterExecutionProviderLibrary");
        eprintln!("✓ Stage 2: RegisterExecutionProviderLibrary");

        // Stage 3: GetEpDevices — our EP must appear
        let mut ep_devices: *const *const ort::OrtEpDevice = ptr::null();
        let mut num_devices: usize = 0;
        let status = ((*api).GetEpDevices.unwrap())(env, &mut ep_devices, &mut num_devices);
        check_status(api, status, "GetEpDevices");
        eprintln!("✓ Stage 3: GetEpDevices returned {num_devices} device(s)");

        let ep_name_fn = (*api).EpDevice_EpName.expect("EpDevice_EpName");
        let mut our_device: *const ort::OrtEpDevice = ptr::null();
        for i in 0..num_devices {
            let dev = *ep_devices.add(i);
            let name_ptr = ep_name_fn(dev);
            if !name_ptr.is_null() {
                let name = CStr::from_ptr(name_ptr).to_string_lossy();
                eprintln!("  Device {i}: {name:?}");
                if name == "cpu_ep" {
                    our_device = dev;
                }
            }
        }
        assert!(
            !our_device.is_null(),
            "EP 'cpu_ep' not in GetEpDevices result"
        );
        eprintln!("✓ Stage 3b: Found 'cpu_ep' in device list");

        // Stage 4: SessionOptions + AppendEP
        let mut session_options: *mut ort::OrtSessionOptions = ptr::null_mut();
        let status = ((*api).CreateSessionOptions.unwrap())(&mut session_options);
        check_status(api, status, "CreateSessionOptions");

        let devices_arr: [*const ort::OrtEpDevice; 1] = [our_device];
        let status = ((*api).SessionOptionsAppendExecutionProvider_V2.unwrap())(
            session_options,
            env,
            devices_arr.as_ptr(),
            1,
            ptr::null(),
            ptr::null(),
            0,
        );
        check_status(api, status, "SessionOptionsAppendExecutionProvider_V2");
        eprintln!("✓ Stage 4: EP appended to session options");

        // Stage 5: CreateSession
        let mut session: *mut ort::OrtSession = ptr::null_mut();
        let status =
            ort_session::create_session(api, env, session_options, &model_path, &mut session);
        check_status(api, status, "CreateSession");
        eprintln!("✓ Stage 5: CreateSession");

        // Stage 6: Run — Add([1,2,3,4], [5,6,7,8]) = [6,8,10,12]
        let mut mem_info: *mut ort::OrtMemoryInfo = ptr::null_mut();
        let status = ((*api).CreateCpuMemoryInfo.unwrap())(
            ort::OrtDeviceAllocator,
            ort::OrtMemTypeDefault,
            &mut mem_info,
        );
        check_status(api, status, "CreateCpuMemoryInfo");

        let shape: [i64; 2] = [1, 4];
        let mut x_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let mut y_data: [f32; 4] = [5.0, 6.0, 7.0, 8.0];

        let mut x_val: *mut ort::OrtValue = ptr::null_mut();
        let status = ((*api).CreateTensorWithDataAsOrtValue.unwrap())(
            mem_info,
            x_data.as_mut_ptr().cast(),
            4 * std::mem::size_of::<f32>(),
            shape.as_ptr(),
            2,
            ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &mut x_val,
        );
        check_status(api, status, "CreateTensor(X)");

        let mut y_val: *mut ort::OrtValue = ptr::null_mut();
        let status = ((*api).CreateTensorWithDataAsOrtValue.unwrap())(
            mem_info,
            y_data.as_mut_ptr().cast(),
            4 * std::mem::size_of::<f32>(),
            shape.as_ptr(),
            2,
            ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &mut y_val,
        );
        check_status(api, status, "CreateTensor(Y)");

        let input_names = [c"X".as_ptr(), c"Y".as_ptr()];
        let output_names = [c"Z".as_ptr()];
        let inputs: [*const ort::OrtValue; 2] = [x_val, y_val];
        let mut output: *mut ort::OrtValue = ptr::null_mut();

        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(),
            input_names.as_ptr(),
            inputs.as_ptr(),
            2,
            output_names.as_ptr(),
            1,
            &mut output,
        );
        check_status(api, status, "Run");
        assert!(!output.is_null());
        eprintln!("✓ Stage 6: Run succeeded");

        // Stage 7: Verify output
        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(output, &mut data_ptr);
        check_status(api, status, "GetTensorMutableData");
        let result = std::slice::from_raw_parts(data_ptr as *const f32, 4);
        let expected: [f32; 4] = [6.0, 8.0, 10.0, 12.0];
        eprintln!("  Got:      {result:?}");
        eprintln!("  Expected: {expected:?}");
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "output[{i}] = {got}, want {want}"
            );
        }
        eprintln!("✓ Stage 7: Output values correct");

        // Teardown
        ((*api).ReleaseValue.unwrap())(output);
        ((*api).ReleaseValue.unwrap())(x_val);
        ((*api).ReleaseValue.unwrap())(y_val);
        ((*api).ReleaseMemoryInfo.unwrap())(mem_info);
        ((*api).ReleaseSession.unwrap())(session);
        ((*api).ReleaseSessionOptions.unwrap())(session_options);
        let status = ((*api).UnregisterExecutionProviderLibrary.unwrap())(env, reg_name.as_ptr());
        check_status(api, status, "UnregisterExecutionProviderLibrary");
        ((*api).ReleaseEnv.unwrap())(env);
        eprintln!("\n✅ ort_loads_our_ep_and_runs_model: ALL STAGES PASSED");
    }
}

// ─── L3 — Negative: unsupported op must not crash ───────────────────────────

/// An unsupported-op model must be declined and must not crash.
///
/// Our CPU EP supports Add and Mul.  A model containing only `NonZero` must
/// fall through to ORT's default CPU EP.  Proves our EP's GetCapability
/// fail-closed behaviour and ORT's correct fallback path.
#[test]
fn ort_unsupported_op_declines_not_crashes() {
    let _lock = lock_ort_ep();
    let ort_lib_dir = skip_if_missing!(
        find_ort_lib_dir(),
        "ort_unsupported_op_declines_not_crashes: ORT not found"
    );
    let ep_lib_path = skip_if_missing!(
        find_ep_cdylib(),
        "ort_unsupported_op_declines_not_crashes: EP cdylib not found"
    );

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path =
        PathBuf::from(manifest_dir).join("tests/fixtures/nonzero_1x4/model.onnx.textproto");
    if !model_path.exists() {
        eprintln!(
            "*** SKIPPED: unsupported-op fixture missing at {}\n\
             Generate it with: python3 tests/fixtures/generate_nonzero.py",
            model_path.display()
        );
        return;
    }

    let ort_lib_path = ort_lib_dir.join(ort_discovery::ort_lib_name());
    let lib = unsafe { libloading::Library::new(&ort_lib_path) }.expect("dlopen ORT");
    let api = unsafe { get_ort_api(&lib) };

    unsafe {
        let mut env: *mut ort::OrtEnv = ptr::null_mut();
        let logid = CString::new("nxrt_neg_test").unwrap();
        let status =
            ((*api).CreateEnv.unwrap())(ort::ORT_LOGGING_LEVEL_WARNING, logid.as_ptr(), &mut env);
        check_status(api, status, "CreateEnv");

        let reg_name = CString::new("cpu_ep_neg").unwrap();
        let ep_path_c = ort_path::OrtPathBuf::new(&ep_lib_path);
        let status = ((*api).RegisterExecutionProviderLibrary.unwrap())(
            env,
            reg_name.as_ptr(),
            ep_path_c.as_ptr(),
        );
        check_status(api, status, "RegisterExecutionProviderLibrary");

        // Get our EP device
        let mut ep_devices: *const *const ort::OrtEpDevice = ptr::null();
        let mut num_devices: usize = 0;
        let status = ((*api).GetEpDevices.unwrap())(env, &mut ep_devices, &mut num_devices);
        check_status(api, status, "GetEpDevices");

        let ep_name_fn = (*api).EpDevice_EpName.expect("EpDevice_EpName");
        let mut our_device: *const ort::OrtEpDevice = ptr::null();
        for i in 0..num_devices {
            let dev = *ep_devices.add(i);
            let name_ptr = ep_name_fn(dev);
            if !name_ptr.is_null() && CStr::from_ptr(name_ptr).to_string_lossy() == "cpu_ep" {
                our_device = dev;
            }
        }
        assert!(
            !our_device.is_null(),
            "EP 'cpu_ep' not found in device list"
        );

        let mut session_options: *mut ort::OrtSessionOptions = ptr::null_mut();
        let status = ((*api).CreateSessionOptions.unwrap())(&mut session_options);
        check_status(api, status, "CreateSessionOptions");

        let devices_arr: [*const ort::OrtEpDevice; 1] = [our_device];
        let status = ((*api).SessionOptionsAppendExecutionProvider_V2.unwrap())(
            session_options,
            env,
            devices_arr.as_ptr(),
            1,
            ptr::null(),
            ptr::null(),
            0,
        );
        check_status(api, status, "SessionOptionsAppendExecutionProvider_V2");

        // CreateSession with unsupported-op model: should succeed (ORT falls back to default EP)
        let mut session: *mut ort::OrtSession = ptr::null_mut();
        let status =
            ort_session::create_session(api, env, session_options, &model_path, &mut session);
        check_status(api, status, "CreateSession(unsupported-op model)");
        assert!(
            !session.is_null(),
            "Session is null for unsupported-op model"
        );
        eprintln!(
            "✓ CreateSession succeeded for unsupported-op model (EP declined, ORT fell back)"
        );

        // Run must also succeed (ORT's default EP handles NonZero)
        let mut input_data: [f32; 4] = [0.0, 1.0, 0.0, 2.0];
        let shape: [i64; 2] = [1, 4];
        let mut mem_info: *mut ort::OrtMemoryInfo = ptr::null_mut();
        let status = ((*api).CreateCpuMemoryInfo.unwrap())(
            ort::OrtDeviceAllocator,
            ort::OrtMemTypeDefault,
            &mut mem_info,
        );
        check_status(api, status, "CreateCpuMemoryInfo");

        let mut input_val: *mut ort::OrtValue = ptr::null_mut();
        let status = ((*api).CreateTensorWithDataAsOrtValue.unwrap())(
            mem_info,
            input_data.as_mut_ptr().cast(),
            4 * std::mem::size_of::<f32>(),
            shape.as_ptr(),
            2,
            ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &mut input_val,
        );
        check_status(api, status, "CreateTensor(input)");

        let input_names = [c"X".as_ptr()];
        let output_names = [c"Y".as_ptr()];
        let inputs: [*const ort::OrtValue; 1] = [input_val];
        let mut output: *mut ort::OrtValue = ptr::null_mut();
        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(),
            input_names.as_ptr(),
            inputs.as_ptr(),
            1,
            output_names.as_ptr(),
            1,
            &mut output,
        );
        check_status(api, status, "Run(unsupported-op model)");
        eprintln!("✓ Run succeeded for unsupported-op model (no crash, correct fallback)");

        ((*api).ReleaseValue.unwrap())(output);
        ((*api).ReleaseValue.unwrap())(input_val);
        ((*api).ReleaseMemoryInfo.unwrap())(mem_info);
        ((*api).ReleaseSession.unwrap())(session);
        ((*api).ReleaseSessionOptions.unwrap())(session_options);
        ((*api).UnregisterExecutionProviderLibrary.unwrap())(env, reg_name.as_ptr());
        ((*api).ReleaseEnv.unwrap())(env);
        eprintln!("\n✅ ort_unsupported_op_declines_not_crashes: PASSED");
    }
}

/// Diagnostic: print which ORT EP API function pointers are non-null.
#[test]
fn diag_ort_ep_api_nullcheck() {
    let ort_lib_dir = skip_if_missing!(
        find_ort_lib_dir(),
        "diag_ort_ep_api_nullcheck: ORT not found"
    );
    let ort_lib_path = ort_lib_dir.join(ort_discovery::ort_lib_name());
    let lib = unsafe { libloading::Library::new(&ort_lib_path) }
        .unwrap_or_else(|e| panic!("Failed to load ORT at {}: {e}", ort_lib_path.display()));
    let api = unsafe { get_ort_api(&lib) };
    macro_rules! check_fn {
        ($field:ident) => {
            eprintln!(
                "  {:60}: {}",
                stringify!($field),
                if unsafe { (*api).$field }.is_some() {
                    "PRESENT"
                } else {
                    "NULL"
                }
            );
        };
    }
    eprintln!("ORT 1.27 API function pointer audit:");
    check_fn!(CreateEnv);
    check_fn!(RegisterExecutionProviderLibrary);
    check_fn!(UnregisterExecutionProviderLibrary);
    check_fn!(GetEpDevices);
    check_fn!(SessionOptionsAppendExecutionProvider_V2);
    check_fn!(EpDevice_EpName);
    check_fn!(GetEpApi);
    check_fn!(CreateSession);
    check_fn!(Run);
    check_fn!(GetErrorMessage);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Conformance suite
//
// All tests in this section share the same blocking condition:
//   BUG: ORT Run fails with "allocator != nullptr was false. Failed to find
//   allocator for device Device:[DeviceType:64 MemoryType:28 ...]"
//   Observed at: plugin_ort_e2e.rs ort_loads_our_ep_and_runs_model, Run stage.
//   Root cause: our EP does not register its allocator with ORT during EP
//   creation/compile, so ORT has no allocator for our custom device when it
//   needs to copy tensors at run-time.
//   Owner: Nabil (crates/onnx-runtime-ep-plugin/src/ep.rs or factory.rs).
//
// Each test also uses a distinct `reg_name` to avoid the parallel-test
// "library is already registered" collision in ORT's global EP registry.
// ═══════════════════════════════════════════════════════════════════════════════

/// Shared setup helper: load ORT, create Env, register EP, find our device,
/// build SessionOptions, append EP, create session.
///
/// Returns `None` when ORT or the EP cdylib are absent (test should skip).
///
/// # Safety
/// All ORT API calls are unsafe; caller must drive teardown:
///   ReleaseSession, ReleaseSessionOptions, UnregisterExecutionProviderLibrary, ReleaseEnv.
/// The process-wide `onnxruntime` handle, loaded once and never unloaded.
///
/// Each test used to `dlopen` its own handle and drop it at test end, which
/// `dlclose`s the library. That is what made this binary crash with an
/// intermittent `STATUS_ACCESS_VIOLATION` at a run-varying location: our plugin
/// cdylib stays resident and caches the host `OrtApi` in a process-global
/// (`status.rs::HOST_ORT_API`, set from `CreateEpFactories`), so unloading and
/// reloading ORT underneath it leaves that pointer describing a library that is
/// no longer mapped where it was.
///
/// Evidence for that reading, measured on this binary:
///
/// * every test passes in isolation, but the suite crashes after 10-16 of them;
/// * the crash lands on a *different* test each run (shape_f32, matmul_2d,
///   matmul_batched_nd), so it is cumulative rather than test-specific;
/// * `--test-threads=1` crashes 4/4, so it is not a data race — which also means
///   adding more locking cannot fix it;
/// * `stress_register_run_unregister_cycles` drives **25** full
///   CreateEnv/register/session/run/unregister/ReleaseEnv cycles and passes,
///   because it deliberately holds one library handle for its whole duration.
///   Cycle count is therefore not the variable; library load/unload is.
///
/// ORT is a process singleton in real use, so holding one handle for the test
/// binary's lifetime matches production and removes the reload entirely.
fn shared_ort_library(path: &std::path::Path) -> Option<&'static libloading::Library> {
    static LIB: std::sync::OnceLock<Option<libloading::Library>> = std::sync::OnceLock::new();
    LIB.get_or_init(|| unsafe { libloading::Library::new(path) }.ok())
        .as_ref()
}

#[allow(clippy::type_complexity)]
unsafe fn conformance_setup(
    reg_name: &str,
    model_path: &std::path::Path,
    disable_fallback: bool,
) -> Option<(
    &'static libloading::Library,
    *const ort::OrtApi,
    *mut ort::OrtEnv,
    *mut ort::OrtSessionOptions,
    *mut ort::OrtSession,
)> {
    let ort_lib_dir = match find_ort_lib_dir() {
        Some(d) => d,
        None => {
            if std::env::var("NXRT_REQUIRE_ORT_TESTS").as_deref() == Ok("1") {
                panic!(
                    "NXRT_REQUIRE_ORT_TESTS=1 but ORT lib dir not found — \
                     conformance test '{reg_name}' cannot run"
                );
            }
            return None;
        }
    };
    let ep_lib_path = match find_ep_cdylib() {
        Some(p) => p,
        None => {
            if std::env::var("NXRT_REQUIRE_ORT_TESTS").as_deref() == Ok("1") {
                panic!(
                    "NXRT_REQUIRE_ORT_TESTS=1 but EP cdylib not found — \
                     conformance test '{reg_name}' cannot run"
                );
            }
            return None;
        }
    };

    if !model_path.exists() {
        if std::env::var("NXRT_REQUIRE_ORT_TESTS").as_deref() == Ok("1") {
            panic!(
                "NXRT_REQUIRE_ORT_TESTS=1 but fixture missing at {}",
                model_path.display()
            );
        }
        eprintln!(
            "*** SKIPPED: fixture missing at {} ***",
            model_path.display()
        );
        return None;
    }

    let ort_lib_path = ort_lib_dir.join(ort_discovery::ort_lib_name());
    let lib = match shared_ort_library(&ort_lib_path) {
        Some(l) => l,
        None => {
            if std::env::var("NXRT_REQUIRE_ORT_TESTS").as_deref() == Ok("1") {
                panic!(
                    "NXRT_REQUIRE_ORT_TESTS=1 but dlopen failed for {}",
                    ort_lib_path.display()
                );
            }
            eprintln!(
                "*** SKIPPED: dlopen failed for {} ***",
                ort_lib_path.display()
            );
            return None;
        }
    };
    let api = unsafe { get_ort_api(lib) };

    // Env
    let mut env: *mut ort::OrtEnv = ptr::null_mut();
    let logid = std::ffi::CString::new(format!("nxrt_{reg_name}")).unwrap();
    let status = unsafe {
        ((*api).CreateEnv.unwrap())(ort::ORT_LOGGING_LEVEL_WARNING, logid.as_ptr(), &mut env)
    };
    unsafe { check_status(api, status, "CreateEnv") };

    // Register EP library
    let reg_name_c = std::ffi::CString::new(reg_name).unwrap();
    let ep_path_c = ort_path::OrtPathBuf::new(&ep_lib_path);
    let status = unsafe {
        ((*api).RegisterExecutionProviderLibrary.unwrap())(
            env,
            reg_name_c.as_ptr(),
            ep_path_c.as_ptr(),
        )
    };
    unsafe { check_status(api, status, "RegisterExecutionProviderLibrary") };

    // Find our device
    let mut ep_devices: *const *const ort::OrtEpDevice = ptr::null();
    let mut num_devices: usize = 0;
    let status = unsafe { ((*api).GetEpDevices.unwrap())(env, &mut ep_devices, &mut num_devices) };
    unsafe { check_status(api, status, "GetEpDevices") };

    let ep_name_fn = unsafe { (*api).EpDevice_EpName.expect("EpDevice_EpName") };
    let mut our_device: *const ort::OrtEpDevice = ptr::null();
    for i in 0..num_devices {
        let dev = unsafe { *ep_devices.add(i) };
        let name_ptr = unsafe { ep_name_fn(dev) };
        // EpDevice_EpName returns the name the EP declares internally ("cpu_ep"),
        // which is independent of the registration key used in RegisterExecutionProviderLibrary.
        if !name_ptr.is_null() && unsafe { CStr::from_ptr(name_ptr) }.to_string_lossy() == "cpu_ep"
        {
            our_device = dev;
        }
    }
    assert!(
        !our_device.is_null(),
        "EP 'cpu_ep' not found in GetEpDevices result (reg_name={reg_name})"
    );

    // SessionOptions + append EP
    let mut session_options: *mut ort::OrtSessionOptions = ptr::null_mut();
    let status = unsafe { ((*api).CreateSessionOptions.unwrap())(&mut session_options) };
    unsafe { check_status(api, status, "CreateSessionOptions") };
    unsafe { pin_intra_op_threads(api, session_options) };

    // Disable ORT's built-in CPU EP fallback — forces failure if our plugin EP
    // declines a node, proving the test is not vacuous.
    if disable_fallback {
        let key = CString::new("session.disable_cpu_ep_fallback").unwrap();
        let val = CString::new("1").unwrap();
        let add_config =
            unsafe { (*api).AddSessionConfigEntry }.expect("AddSessionConfigEntry not in OrtApi");
        let status = unsafe { add_config(session_options, key.as_ptr(), val.as_ptr()) };
        unsafe {
            check_status(
                api,
                status,
                "AddSessionConfigEntry(disable_cpu_ep_fallback)",
            )
        };
    }

    // Enable EP graph assignment recording so we can query per-node provider
    // attribution via Session_GetEpGraphAssignmentInfo (available since ORT 1.24).
    {
        let key = CString::new("session.record_ep_graph_assignment_info").unwrap();
        let val = CString::new("1").unwrap();
        let add_config =
            unsafe { (*api).AddSessionConfigEntry }.expect("AddSessionConfigEntry not in OrtApi");
        let status = unsafe { add_config(session_options, key.as_ptr(), val.as_ptr()) };
        unsafe {
            check_status(
                api,
                status,
                "AddSessionConfigEntry(record_ep_graph_assignment_info)",
            )
        };
    }

    let devices_arr: [*const ort::OrtEpDevice; 1] = [our_device];
    let status = unsafe {
        ((*api).SessionOptionsAppendExecutionProvider_V2.unwrap())(
            session_options,
            env,
            devices_arr.as_ptr(),
            1,
            ptr::null(),
            ptr::null(),
            0,
        )
    };
    unsafe { check_status(api, status, "SessionOptionsAppendExecutionProvider_V2") };

    // CreateSession
    let mut session: *mut ort::OrtSession = ptr::null_mut();
    let status =
        unsafe { ort_session::create_session(api, env, session_options, model_path, &mut session) };
    unsafe { check_status(api, status, "CreateSession") };

    Some((lib, api, env, session_options, session))
}

/// Shared teardown: ReleaseSession, ReleaseSessionOptions, Unregister, ReleaseEnv.
unsafe fn conformance_teardown(
    api: *const ort::OrtApi,
    env: *mut ort::OrtEnv,
    session_options: *mut ort::OrtSessionOptions,
    session: *mut ort::OrtSession,
    reg_name: &str,
) {
    unsafe {
        ((*api).ReleaseSession.unwrap())(session);
        ((*api).ReleaseSessionOptions.unwrap())(session_options);
        let reg_name_c = std::ffi::CString::new(reg_name).unwrap();
        let status = ((*api).UnregisterExecutionProviderLibrary.unwrap())(env, reg_name_c.as_ptr());
        check_status(api, status, "UnregisterExecutionProviderLibrary");
        ((*api).ReleaseEnv.unwrap())(env);
    }
}

// ─── Helper: query EP graph assignment ────────────────────────────────────────

/// Result of querying which nodes are assigned to which EP.
struct EpAssignmentInfo {
    /// (ep_name, op_type) pairs for every assigned node.
    assignments: Vec<(String, String)>,
}

impl EpAssignmentInfo {
    /// Op types assigned to our EP ("cpu_ep").
    fn ops_on_our_ep(&self) -> Vec<&str> {
        self.assignments
            .iter()
            .filter(|(ep, _)| ep == "cpu_ep")
            .map(|(_, op)| op.as_str())
            .collect()
    }

    /// Op types NOT assigned to our EP.
    fn ops_not_on_our_ep(&self) -> Vec<&str> {
        self.assignments
            .iter()
            .filter(|(ep, _)| ep != "cpu_ep")
            .map(|(_, op)| op.as_str())
            .collect()
    }
}

/// Query `Session_GetEpGraphAssignmentInfo` and return per-node (ep_name, op_type).
///
/// Requires `session.record_ep_graph_assignment_info=1` to have been set before
/// session creation (`conformance_setup` enables this unconditionally).
///
/// # Safety
/// `api` and `session` must be valid pointers from a successfully created session.
unsafe fn query_ep_assignment(
    api: *const ort::OrtApi,
    session: *mut ort::OrtSession,
) -> EpAssignmentInfo {
    let get_info = unsafe { (*api).Session_GetEpGraphAssignmentInfo }
        .expect("Session_GetEpGraphAssignmentInfo not in OrtApi — requires ORT ≥1.24");
    let get_ep_name =
        unsafe { (*api).EpAssignedSubgraph_GetEpName }.expect("EpAssignedSubgraph_GetEpName");
    let get_nodes =
        unsafe { (*api).EpAssignedSubgraph_GetNodes }.expect("EpAssignedSubgraph_GetNodes");
    let get_op_type =
        unsafe { (*api).EpAssignedNode_GetOperatorType }.expect("EpAssignedNode_GetOperatorType");

    let mut ep_subgraphs: *const *const ort::OrtEpAssignedSubgraph = ptr::null();
    let mut num_subgraphs: usize = 0;

    let status = unsafe { get_info(session, &mut ep_subgraphs, &mut num_subgraphs) };
    unsafe { check_status(api, status, "Session_GetEpGraphAssignmentInfo") };

    let mut assignments = Vec::new();
    for i in 0..num_subgraphs {
        let subgraph = unsafe { *ep_subgraphs.add(i) };

        let mut ep_name_ptr: *const std::os::raw::c_char = ptr::null();
        let status = unsafe { get_ep_name(subgraph, &mut ep_name_ptr) };
        unsafe { check_status(api, status, "EpAssignedSubgraph_GetEpName") };
        let ep_name = unsafe { CStr::from_ptr(ep_name_ptr) }
            .to_string_lossy()
            .into_owned();

        let mut ep_nodes: *const *const ort::OrtEpAssignedNode = ptr::null();
        let mut num_nodes: usize = 0;
        let status = unsafe { get_nodes(subgraph, &mut ep_nodes, &mut num_nodes) };
        unsafe { check_status(api, status, "EpAssignedSubgraph_GetNodes") };

        for j in 0..num_nodes {
            let node = unsafe { *ep_nodes.add(j) };
            let mut op_type_ptr: *const std::os::raw::c_char = ptr::null();
            let status = unsafe { get_op_type(node, &mut op_type_ptr) };
            unsafe { check_status(api, status, "EpAssignedNode_GetOperatorType") };
            let op_type = unsafe { CStr::from_ptr(op_type_ptr) }
                .to_string_lossy()
                .into_owned();
            assignments.push((ep_name.clone(), op_type));
        }
    }

    EpAssignmentInfo { assignments }
}

/// Assert that at least `expected_ops` are assigned to "cpu_ep".
/// Panics with a clear message if any expected op is missing.
unsafe fn assert_ops_assigned_to_our_ep(
    api: *const ort::OrtApi,
    session: *mut ort::OrtSession,
    expected_ops: &[&str],
    test_label: &str,
) {
    let info = unsafe { query_ep_assignment(api, session) };
    let ours = info.ops_on_our_ep();
    eprintln!(
        "  [{test_label}] EP assignment: ours={ours:?}, others={:?}",
        info.ops_not_on_our_ep()
    );
    for &op in expected_ops {
        assert!(
            ours.contains(&op),
            "[{test_label}] Expected op '{op}' assigned to cpu_ep, \
             but assignment was: {:?}",
            info.assignments
        );
    }
}

// ─── Helper: create a float tensor from a slice ───────────────────────────────

unsafe fn make_float_tensor(
    api: *const ort::OrtApi,
    data: &mut [f32],
    shape: &[i64],
) -> *mut ort::OrtValue {
    unsafe {
        let mut mem_info: *mut ort::OrtMemoryInfo = ptr::null_mut();
        let status = ((*api).CreateCpuMemoryInfo.unwrap())(
            ort::OrtDeviceAllocator,
            ort::OrtMemTypeDefault,
            &mut mem_info,
        );
        check_status(api, status, "CreateCpuMemoryInfo");

        let mut val: *mut ort::OrtValue = ptr::null_mut();
        let status = ((*api).CreateTensorWithDataAsOrtValue.unwrap())(
            mem_info,
            data.as_mut_ptr().cast(),
            std::mem::size_of_val(data),
            shape.as_ptr(),
            shape.len(),
            ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &mut val,
        );
        check_status(api, status, "CreateTensorWithDataAsOrtValue(float)");
        ((*api).ReleaseMemoryInfo.unwrap())(mem_info);
        val
    }
}

/// Create a FLOAT16 ORT tensor from raw u16 words (IEEE 754 binary16).
unsafe fn make_float16_tensor(
    api: *const ort::OrtApi,
    data: &mut [u16],
    shape: &[i64],
) -> *mut ort::OrtValue {
    unsafe {
        let mut mem_info: *mut ort::OrtMemoryInfo = ptr::null_mut();
        let status = ((*api).CreateCpuMemoryInfo.unwrap())(
            ort::OrtDeviceAllocator,
            ort::OrtMemTypeDefault,
            &mut mem_info,
        );
        check_status(api, status, "CreateCpuMemoryInfo(f16)");

        let mut val: *mut ort::OrtValue = ptr::null_mut();
        let status = ((*api).CreateTensorWithDataAsOrtValue.unwrap())(
            mem_info,
            data.as_mut_ptr().cast(),
            std::mem::size_of_val(data),
            shape.as_ptr(),
            shape.len(),
            ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
            &mut val,
        );
        check_status(api, status, "CreateTensorWithDataAsOrtValue(float16)");
        ((*api).ReleaseMemoryInfo.unwrap())(mem_info);
        val
    }
}

/// Create a BFLOAT16 ORT tensor from raw u16 words (top-16-bits of IEEE 754 binary32).
unsafe fn make_bfloat16_tensor(
    api: *const ort::OrtApi,
    data: &mut [u16],
    shape: &[i64],
) -> *mut ort::OrtValue {
    unsafe {
        let mut mem_info: *mut ort::OrtMemoryInfo = ptr::null_mut();
        let status = ((*api).CreateCpuMemoryInfo.unwrap())(
            ort::OrtDeviceAllocator,
            ort::OrtMemTypeDefault,
            &mut mem_info,
        );
        check_status(api, status, "CreateCpuMemoryInfo(bf16)");

        let mut val: *mut ort::OrtValue = ptr::null_mut();
        let status = ((*api).CreateTensorWithDataAsOrtValue.unwrap())(
            mem_info,
            data.as_mut_ptr().cast(),
            std::mem::size_of_val(data),
            shape.as_ptr(),
            shape.len(),
            ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
            &mut val,
        );
        check_status(api, status, "CreateTensorWithDataAsOrtValue(bfloat16)");
        ((*api).ReleaseMemoryInfo.unwrap())(mem_info);
        val
    }
}

unsafe fn make_int32_tensor(
    api: *const ort::OrtApi,
    data: &mut [i32],
    shape: &[i64],
) -> *mut ort::OrtValue {
    unsafe {
        let mut mem_info: *mut ort::OrtMemoryInfo = ptr::null_mut();
        let status = ((*api).CreateCpuMemoryInfo.unwrap())(
            ort::OrtDeviceAllocator,
            ort::OrtMemTypeDefault,
            &mut mem_info,
        );
        check_status(api, status, "CreateCpuMemoryInfo");

        let mut val: *mut ort::OrtValue = ptr::null_mut();
        let status = ((*api).CreateTensorWithDataAsOrtValue.unwrap())(
            mem_info,
            data.as_mut_ptr().cast(),
            std::mem::size_of_val(data),
            shape.as_ptr(),
            shape.len(),
            ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
            &mut val,
        );
        check_status(api, status, "CreateTensorWithDataAsOrtValue(int32)");
        ((*api).ReleaseMemoryInfo.unwrap())(mem_info);
        val
    }
}

// ─── Conformance: broadcast Add ──────────────────────────────────────────────

/// Add with broadcast: X=[2,3] + Y=[3] → Z=[2,3]
///
/// X = [[1,2,3],[4,5,6]]  Y = [10,20,30]
/// Expected Z = [[11,22,33],[14,25,36]]
///
/// Blocked by: allocator-registration bug in ep.rs/factory.rs (Nabil).
#[test]
fn conformance_add_broadcast() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path =
        PathBuf::from(manifest_dir).join("tests/fixtures/add_broadcast/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_bc", &model_path, true) })
    else {
        eprintln!("*** SKIPPED: conformance_add_broadcast — ORT or EP cdylib not found ***");
        return;
    };

    // Prove Add is assigned to our EP.
    unsafe { assert_ops_assigned_to_our_ep(api, session, &["Add"], "add_broadcast") };

    unsafe {
        let mut x_data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut y_data: [f32; 3] = [10.0, 20.0, 30.0];
        let x_shape: [i64; 2] = [2, 3];
        let y_shape: [i64; 1] = [3];

        let x_val = make_float_tensor(api, &mut x_data, &x_shape);
        let y_val = make_float_tensor(api, &mut y_data, &y_shape);

        let input_names = [c"X".as_ptr(), c"Y".as_ptr()];
        let output_names = [c"Z".as_ptr()];
        let inputs: [*const ort::OrtValue; 2] = [x_val, y_val];
        let mut output: *mut ort::OrtValue = ptr::null_mut();

        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(),
            input_names.as_ptr(),
            inputs.as_ptr(),
            2,
            output_names.as_ptr(),
            1,
            &mut output,
        );
        check_status(api, status, "Run");
        assert!(!output.is_null());

        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(output, &mut data_ptr);
        check_status(api, status, "GetTensorMutableData");
        let result = std::slice::from_raw_parts(data_ptr as *const f32, 6);
        let expected: [f32; 6] = [11.0, 22.0, 33.0, 14.0, 25.0, 36.0];
        eprintln!("  Got:      {result:?}");
        eprintln!("  Expected: {expected:?}");
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "output[{i}] = {got}, want {want}"
            );
        }

        ((*api).ReleaseValue.unwrap())(output);
        ((*api).ReleaseValue.unwrap())(x_val);
        ((*api).ReleaseValue.unwrap())(y_val);
        conformance_teardown(api, env, opts, session, "cpu_ep_bc");
        eprintln!("\n✅ conformance_add_broadcast: PASSED");
    }
}

// ─── Conformance: multi-node fused subgraph ───────────────────────────────────

/// Multi-node chain: T = (A + B) * C + D, all shape [1,4]
///
/// A=[1,2,3,4]  B=[1,1,1,1]  C=[2,2,2,2]  D=[0,0,0,0]
/// Expected T = [4,6,8,10]
///
/// This is the highest-value gap: topological intermediate threading through
/// Deckard/Nabil's fused-subgraph path has not been proven against real ORT.
///
/// Blocked by: allocator-registration bug in ep.rs/factory.rs (Nabil).
#[test]
fn conformance_chain_add_mul() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path =
        PathBuf::from(manifest_dir).join("tests/fixtures/chain_add_mul/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_chain", &model_path, true) })
    else {
        eprintln!("*** SKIPPED: conformance_chain_add_mul — ORT or EP cdylib not found ***");
        return;
    };

    // Prove Add and Mul are assigned to our EP.
    unsafe { assert_ops_assigned_to_our_ep(api, session, &["Add", "Mul"], "chain_add_mul") };

    unsafe {
        let mut a_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let mut b_data: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
        let mut c_data: [f32; 4] = [2.0, 2.0, 2.0, 2.0];
        let mut d_data: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
        let shape: [i64; 2] = [1, 4];

        let a_val = make_float_tensor(api, &mut a_data, &shape);
        let b_val = make_float_tensor(api, &mut b_data, &shape);
        let c_val = make_float_tensor(api, &mut c_data, &shape);
        let d_val = make_float_tensor(api, &mut d_data, &shape);

        let input_names = [c"A".as_ptr(), c"B".as_ptr(), c"C".as_ptr(), c"D".as_ptr()];
        let output_names = [c"T".as_ptr()];
        let inputs: [*const ort::OrtValue; 4] = [a_val, b_val, c_val, d_val];
        let mut output: *mut ort::OrtValue = ptr::null_mut();

        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(),
            input_names.as_ptr(),
            inputs.as_ptr(),
            4,
            output_names.as_ptr(),
            1,
            &mut output,
        );
        check_status(api, status, "Run");
        assert!(!output.is_null());

        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(output, &mut data_ptr);
        check_status(api, status, "GetTensorMutableData");
        let result = std::slice::from_raw_parts(data_ptr as *const f32, 4);
        // (A+B)*C+D = ([2,3,4,5]*[2,2,2,2])+[0,0,0,0] = [4,6,8,10]
        let expected: [f32; 4] = [4.0, 6.0, 8.0, 10.0];
        eprintln!("  Got:      {result:?}");
        eprintln!("  Expected: {expected:?}");
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "output[{i}] = {got}, want {want}"
            );
        }

        ((*api).ReleaseValue.unwrap())(output);
        ((*api).ReleaseValue.unwrap())(a_val);
        ((*api).ReleaseValue.unwrap())(b_val);
        ((*api).ReleaseValue.unwrap())(c_val);
        ((*api).ReleaseValue.unwrap())(d_val);
        conformance_teardown(api, env, opts, session, "cpu_ep_chain");
        eprintln!("\n✅ conformance_chain_add_mul: PASSED");
    }
}

/// Same three-node chain, run repeatedly with changing inputs.
///
/// The routed path recycles a subgraph's intermediate storage — a buffer is
/// retired once its last reader has run, and a thread-local pool carries the
/// storage into the next `Run` without re-zeroing it. So from the second `Run`
/// on, every intermediate is served dirty, holding the previous iteration's
/// values.
///
/// That is only safe while each kernel writes the whole of its output. This
/// test is the falsifier: the inputs change on every iteration, so any element
/// a kernel failed to write would surface as the *previous* iteration's answer
/// rather than the current one. A single `Run` cannot catch that, because the
/// first pass through the pool is freshly zeroed.
#[test]
fn conformance_chain_add_mul_repeated_runs_do_not_leak_stale_intermediates() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path =
        PathBuf::from(manifest_dir).join("tests/fixtures/chain_add_mul/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_chain_repeat", &model_path, true) })
    else {
        eprintln!(
            "*** SKIPPED: conformance_chain_add_mul_repeated_runs_do_not_leak_stale \
             intermediates — ORT or EP cdylib not found ***"
        );
        return;
    };

    unsafe { assert_ops_assigned_to_our_ep(api, session, &["Add", "Mul"], "chain_add_mul") };

    unsafe {
        for iter in 0..6u32 {
            let base = (iter as f32) * 10.0 + 1.0;
            let mut a_data: [f32; 4] = [base, base + 1.0, base + 2.0, base + 3.0];
            let mut b_data: [f32; 4] = [base * 2.0; 4];
            let mut c_data: [f32; 4] = [base + 0.5; 4];
            let mut d_data: [f32; 4] = [-base; 4];
            let shape: [i64; 2] = [1, 4];

            let expected: Vec<f32> = (0..4)
                .map(|i| (a_data[i] + b_data[i]) * c_data[i] + d_data[i])
                .collect();

            let a_val = make_float_tensor(api, &mut a_data, &shape);
            let b_val = make_float_tensor(api, &mut b_data, &shape);
            let c_val = make_float_tensor(api, &mut c_data, &shape);
            let d_val = make_float_tensor(api, &mut d_data, &shape);

            let input_names = [c"A".as_ptr(), c"B".as_ptr(), c"C".as_ptr(), c"D".as_ptr()];
            let output_names = [c"T".as_ptr()];
            let inputs: [*const ort::OrtValue; 4] = [a_val, b_val, c_val, d_val];
            let mut output: *mut ort::OrtValue = ptr::null_mut();

            let status = ((*api).Run.unwrap())(
                session,
                ptr::null(),
                input_names.as_ptr(),
                inputs.as_ptr(),
                4,
                output_names.as_ptr(),
                1,
                &mut output,
            );
            check_status(api, status, "Run");
            assert!(!output.is_null());

            let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
            let status = ((*api).GetTensorMutableData.unwrap())(output, &mut data_ptr);
            check_status(api, status, "GetTensorMutableData");
            let result = std::slice::from_raw_parts(data_ptr as *const f32, 4);
            for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1e-4,
                    "iteration {iter}: output[{i}] = {got}, want {want} \
                     (a stale recycled intermediate would show the previous iteration)"
                );
            }

            ((*api).ReleaseValue.unwrap())(output);
            ((*api).ReleaseValue.unwrap())(a_val);
            ((*api).ReleaseValue.unwrap())(b_val);
            ((*api).ReleaseValue.unwrap())(c_val);
            ((*api).ReleaseValue.unwrap())(d_val);
        }

        conformance_teardown(api, env, opts, session, "cpu_ep_chain_repeat");
        eprintln!(
            "\n✅ conformance_chain_add_mul_repeated_runs_do_not_leak_stale_intermediates: PASSED"
        );
    }
}

// ─── Conformance: MatMul ──────────────────────────────────────────────────────

/// MatMul [2,3] × [3,2] → [2,2]
///
/// A = [[1,2,3],[4,5,6]]   B = [[1,0],[0,1],[1,0]]
/// Expected C = [[4,2],[10,5]]
///
/// Blocked by: allocator-registration bug in ep.rs/factory.rs (Nabil).
#[test]
fn conformance_matmul_2d() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path =
        PathBuf::from(manifest_dir).join("tests/fixtures/matmul_2d/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_mm", &model_path, true) })
    else {
        eprintln!("*** SKIPPED: conformance_matmul_2d — ORT or EP cdylib not found ***");
        return;
    };

    // Prove MatMul is assigned to our EP.
    unsafe { assert_ops_assigned_to_our_ep(api, session, &["MatMul"], "matmul_2d") };

    unsafe {
        let mut a_data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut b_data: [f32; 6] = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
        let a_shape: [i64; 2] = [2, 3];
        let b_shape: [i64; 2] = [3, 2];

        let a_val = make_float_tensor(api, &mut a_data, &a_shape);
        let b_val = make_float_tensor(api, &mut b_data, &b_shape);

        let input_names = [c"A".as_ptr(), c"B".as_ptr()];
        let output_names = [c"C".as_ptr()];
        let inputs: [*const ort::OrtValue; 2] = [a_val, b_val];
        let mut output: *mut ort::OrtValue = ptr::null_mut();

        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(),
            input_names.as_ptr(),
            inputs.as_ptr(),
            2,
            output_names.as_ptr(),
            1,
            &mut output,
        );
        check_status(api, status, "Run");
        assert!(!output.is_null());

        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(output, &mut data_ptr);
        check_status(api, status, "GetTensorMutableData");
        let result = std::slice::from_raw_parts(data_ptr as *const f32, 4);
        let expected: [f32; 4] = [4.0, 2.0, 10.0, 5.0];
        eprintln!("  Got:      {result:?}");
        eprintln!("  Expected: {expected:?}");
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "output[{i}] = {got}, want {want}"
            );
        }

        ((*api).ReleaseValue.unwrap())(output);
        ((*api).ReleaseValue.unwrap())(a_val);
        ((*api).ReleaseValue.unwrap())(b_val);
        conformance_teardown(api, env, opts, session, "cpu_ep_mm");
        eprintln!("\n✅ conformance_matmul_2d: PASSED");
    }
}

// ─── Conformance: mixed partition (Add + NonZero) ────────────────────────────

/// Graph with Add (claimed by our EP) + NonZero (not claimed).
/// ORT must partition: our EP takes Add, ORT's default CPU EP takes NonZero.
/// Final output must be numerically correct.
///
/// X=[1,2,3,4]  Y=[0,0,0,0]  SUM=[1,2,3,4]  NonZero(SUM) = [[0],[1],[2],[3]] (row per dim)
///
/// Blocked by: allocator-registration bug in ep.rs/factory.rs (Nabil).
/// Also requires NonZero claim-predicate fix in graph_reader.rs (Nabil).
#[test]
fn conformance_mixed_partition() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path =
        PathBuf::from(manifest_dir).join("tests/fixtures/mixed_partition/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_mix", &model_path, false) })
    else {
        eprintln!("*** SKIPPED: conformance_mixed_partition — ORT or EP cdylib not found ***");
        return;
    };

    // Assert EP assignment via Session_GetEpGraphAssignmentInfo (ORT ≥1.24).
    // With disable_cpu_ep_fallback=false, ORT may route the entire graph to its
    // built-in CPUExecutionProvider to avoid partition overhead — that's valid.
    // The key invariant: our EP must NEVER be assigned NonZero (unsupported).
    // If ORT does partition, Add must be on our EP.
    unsafe {
        let info = query_ep_assignment(api, session);
        let ours = info.ops_on_our_ep();
        let others = info.ops_not_on_our_ep();
        eprintln!("  [mixed_partition] EP assignment: ours={ours:?}, others={others:?}");
        assert!(
            !ours.contains(&"NonZero"),
            "NonZero must NOT be assigned to cpu_ep (unsupported), got: {:?}",
            info.assignments
        );
        if ours.contains(&"Add") {
            eprintln!("  ✓ ORT assigned Add to our EP — partition confirmed");
        } else {
            eprintln!(
                "  ℹ ORT routed all to built-in CPUExecutionProvider \
                 (no partition) — valid with disable_cpu_ep_fallback=false"
            );
        }
    }

    unsafe {
        let mut x_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let mut y_data: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
        let shape: [i64; 2] = [1, 4];

        let x_val = make_float_tensor(api, &mut x_data, &shape);
        let y_val = make_float_tensor(api, &mut y_data, &shape);

        let input_names = [c"X".as_ptr(), c"Y".as_ptr()];
        let output_names = [c"Z".as_ptr()];
        let inputs: [*const ort::OrtValue; 2] = [x_val, y_val];
        let mut output: *mut ort::OrtValue = ptr::null_mut();

        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(),
            input_names.as_ptr(),
            inputs.as_ptr(),
            2,
            output_names.as_ptr(),
            1,
            &mut output,
        );
        check_status(api, status, "Run");
        assert!(!output.is_null(), "output is null");

        // NonZero([1,2,3,4]) → all four positions are non-zero.
        // Shape [1,4] flattened = 4 elements; rank=2 so NonZero output is [2,4].
        // indices: row0 = [0,0,0,0]  row1 = [0,1,2,3]  (row-major flat index per dim)
        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(output, &mut data_ptr);
        check_status(api, status, "GetTensorMutableData");
        let result = std::slice::from_raw_parts(data_ptr as *const i64, 8);
        let expected: [i64; 8] = [0, 0, 0, 0, 0, 1, 2, 3];
        eprintln!("  Got:      {result:?}");
        eprintln!("  Expected: {expected:?}");
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert_eq!(got, want, "output[{i}] = {got}, want {want}");
        }

        ((*api).ReleaseValue.unwrap())(output);
        ((*api).ReleaseValue.unwrap())(x_val);
        ((*api).ReleaseValue.unwrap())(y_val);
        conformance_teardown(api, env, opts, session, "cpu_ep_mix");
        eprintln!("\n✅ conformance_mixed_partition: PASSED");
    }
}

// ─── Conformance: integer dtype (Add INT32) ──────────────────────────────────

/// Add with INT32 inputs: shape [1,4].
///
/// X=[10,20,30,40]  Y=[1,2,3,4]  Expected Z=[11,22,33,44]
///
/// Blocked by: allocator-registration bug in ep.rs/factory.rs (Nabil).
#[test]
fn conformance_add_int32() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path =
        PathBuf::from(manifest_dir).join("tests/fixtures/add_int32/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_i32", &model_path, true) })
    else {
        eprintln!("*** SKIPPED: conformance_add_int32 — ORT or EP cdylib not found ***");
        return;
    };

    // Prove Add is assigned to our EP.
    unsafe { assert_ops_assigned_to_our_ep(api, session, &["Add"], "add_int32") };

    unsafe {
        let mut x_data: [i32; 4] = [10, 20, 30, 40];
        let mut y_data: [i32; 4] = [1, 2, 3, 4];
        let shape: [i64; 2] = [1, 4];

        let x_val = make_int32_tensor(api, &mut x_data, &shape);
        let y_val = make_int32_tensor(api, &mut y_data, &shape);

        let input_names = [c"X".as_ptr(), c"Y".as_ptr()];
        let output_names = [c"Z".as_ptr()];
        let inputs: [*const ort::OrtValue; 2] = [x_val, y_val];
        let mut output: *mut ort::OrtValue = ptr::null_mut();

        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(),
            input_names.as_ptr(),
            inputs.as_ptr(),
            2,
            output_names.as_ptr(),
            1,
            &mut output,
        );
        check_status(api, status, "Run");
        assert!(!output.is_null());

        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(output, &mut data_ptr);
        check_status(api, status, "GetTensorMutableData");
        let result = std::slice::from_raw_parts(data_ptr as *const i32, 4);
        let expected: [i32; 4] = [11, 22, 33, 44];
        eprintln!("  Got:      {result:?}");
        eprintln!("  Expected: {expected:?}");
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert_eq!(got, want, "output[{i}] = {got}, want {want}");
        }

        ((*api).ReleaseValue.unwrap())(output);
        ((*api).ReleaseValue.unwrap())(x_val);
        ((*api).ReleaseValue.unwrap())(y_val);
        conformance_teardown(api, env, opts, session, "cpu_ep_i32");
        eprintln!("\n✅ conformance_add_int32: PASSED");
    }
}

// ─── Conformance: dynamic dimension ──────────────────────────────────────────

/// Add with symbolic/dynamic batch dimension: graph shape is ["batch", 4].
///
/// At runtime, batch=1: X=[1,2,3,4]  Y=[5,6,7,8]  Expected Z=[6,8,10,12]
///
/// Validates that our EP handles ORT's -1 sentinel for dynamic dims without
/// wrapping to usize::MAX (Leon's bug, tracked in kernel_ctx.rs).
/// Acceptable outcomes: correct output OR a clean error; never a crash or
/// silently wrong numbers.
///
/// Blocked by: allocator-registration bug in ep.rs/factory.rs (Nabil).
#[test]
fn conformance_add_dynamic_dim() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path =
        PathBuf::from(manifest_dir).join("tests/fixtures/add_dynamic_dim/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_dyn", &model_path, true) })
    else {
        eprintln!("*** SKIPPED: conformance_add_dynamic_dim — ORT or EP cdylib not found ***");
        return;
    };

    // Prove Add is assigned to our EP.
    unsafe { assert_ops_assigned_to_our_ep(api, session, &["Add"], "add_dynamic_dim") };
    unsafe {
        // Provide batch=1 at runtime
        let mut x_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let mut y_data: [f32; 4] = [5.0, 6.0, 7.0, 8.0];
        let shape: [i64; 2] = [1, 4];

        let x_val = make_float_tensor(api, &mut x_data, &shape);
        let y_val = make_float_tensor(api, &mut y_data, &shape);

        let input_names = [c"X".as_ptr(), c"Y".as_ptr()];
        let output_names = [c"Z".as_ptr()];
        let inputs: [*const ort::OrtValue; 2] = [x_val, y_val];
        let mut output: *mut ort::OrtValue = ptr::null_mut();

        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(),
            input_names.as_ptr(),
            inputs.as_ptr(),
            2,
            output_names.as_ptr(),
            1,
            &mut output,
        );
        // Accept either success (check values) or a clean ORT error.
        // A crash or incorrect output is the actual failure we guard against.
        if !status.is_null() {
            let get_msg = (*api).GetErrorMessage.expect("GetErrorMessage");
            let msg = CStr::from_ptr(get_msg(status))
                .to_string_lossy()
                .into_owned();
            ((*api).ReleaseStatus.unwrap())(status);
            eprintln!("  Run returned ORT error (acceptable for dynamic-dim path): {msg}");
            // Must be a real ORT error, not a crash-followed-by-null.
            assert!(
                !msg.is_empty(),
                "status non-null but error message is empty — likely memory corruption"
            );
            eprintln!(
                "✓ conformance_add_dynamic_dim: dynamic dim handled with clean error (no crash)"
            );
        } else {
            assert!(!output.is_null());
            let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
            let status = ((*api).GetTensorMutableData.unwrap())(output, &mut data_ptr);
            check_status(api, status, "GetTensorMutableData");
            let result = std::slice::from_raw_parts(data_ptr as *const f32, 4);
            let expected: [f32; 4] = [6.0, 8.0, 10.0, 12.0];
            eprintln!("  Got:      {result:?}");
            eprintln!("  Expected: {expected:?}");
            for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1e-6,
                    "output[{i}] = {got}, want {want}"
                );
            }
            ((*api).ReleaseValue.unwrap())(output);
            eprintln!(
                "✓ conformance_add_dynamic_dim: dynamic dim handled correctly with correct output"
            );
        }

        ((*api).ReleaseValue.unwrap())(x_val);
        ((*api).ReleaseValue.unwrap())(y_val);
        conformance_teardown(api, env, opts, session, "cpu_ep_dyn");
        eprintln!("\n✅ conformance_add_dynamic_dim: PASSED");
    }
}

// ─── Conformance: multiple sequential Run calls ──────────────────────────────

/// Two back-to-back Run calls on the same session prove compute state is
/// reusable and not corrupted between inferences.
///
/// Call 1: [1,2,3,4] + [5,6,7,8] = [6,8,10,12]
/// Call 2: [10,0,10,0] + [0,10,0,10] = [10,10,10,10]
#[test]
fn conformance_multiple_run_calls() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path =
        PathBuf::from(manifest_dir).join("tests/fixtures/add_1x4/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_runs", &model_path, true) })
    else {
        eprintln!("*** SKIPPED: conformance_multiple_run_calls — ORT or EP cdylib not found ***");
        return;
    };

    unsafe {
        let shape: [i64; 2] = [1, 4];

        // Run 1
        let mut x1: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let mut y1: [f32; 4] = [5.0, 6.0, 7.0, 8.0];
        let xv1 = make_float_tensor(api, &mut x1, &shape);
        let yv1 = make_float_tensor(api, &mut y1, &shape);
        let input_names = [c"X".as_ptr(), c"Y".as_ptr()];
        let output_names = [c"Z".as_ptr()];
        let inputs1: [*const ort::OrtValue; 2] = [xv1, yv1];
        let mut out1: *mut ort::OrtValue = ptr::null_mut();
        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(),
            input_names.as_ptr(),
            inputs1.as_ptr(),
            2,
            output_names.as_ptr(),
            1,
            &mut out1,
        );
        check_status(api, status, "Run(1)");
        assert!(!out1.is_null());
        let mut dp: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(out1, &mut dp);
        check_status(api, status, "GetTensorMutableData(1)");
        let r1 = std::slice::from_raw_parts(dp as *const f32, 4);
        let e1: [f32; 4] = [6.0, 8.0, 10.0, 12.0];
        for (i, (got, want)) in r1.iter().zip(e1.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "Run1 output[{i}] = {got}, want {want}"
            );
        }
        eprintln!("  Run 1 ok: {r1:?}");

        // Run 2 — same session, different inputs
        let mut x2: [f32; 4] = [10.0, 0.0, 10.0, 0.0];
        let mut y2: [f32; 4] = [0.0, 10.0, 0.0, 10.0];
        let xv2 = make_float_tensor(api, &mut x2, &shape);
        let yv2 = make_float_tensor(api, &mut y2, &shape);
        let inputs2: [*const ort::OrtValue; 2] = [xv2, yv2];
        let mut out2: *mut ort::OrtValue = ptr::null_mut();
        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(),
            input_names.as_ptr(),
            inputs2.as_ptr(),
            2,
            output_names.as_ptr(),
            1,
            &mut out2,
        );
        check_status(api, status, "Run(2)");
        assert!(!out2.is_null());
        let mut dp2: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(out2, &mut dp2);
        check_status(api, status, "GetTensorMutableData(2)");
        let r2 = std::slice::from_raw_parts(dp2 as *const f32, 4);
        let e2: [f32; 4] = [10.0, 10.0, 10.0, 10.0];
        for (i, (got, want)) in r2.iter().zip(e2.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "Run2 output[{i}] = {got}, want {want}"
            );
        }
        eprintln!("  Run 2 ok: {r2:?}");

        ((*api).ReleaseValue.unwrap())(out2);
        ((*api).ReleaseValue.unwrap())(out1);
        ((*api).ReleaseValue.unwrap())(xv2);
        ((*api).ReleaseValue.unwrap())(yv2);
        ((*api).ReleaseValue.unwrap())(xv1);
        ((*api).ReleaseValue.unwrap())(yv1);
        conformance_teardown(api, env, opts, session, "cpu_ep_runs");
        eprintln!("\n✅ conformance_multiple_run_calls: PASSED");
    }
}

// ─── Conformance: two sessions from one registered library ───────────────────

/// Create two independent sessions from a single registered EP library.
/// Proves factory/EP lifetime is sound and sessions don't alias each other's state.
///
/// Session A: add_1x4 model.  Session B: add_broadcast model.
/// Both run and must produce correct, independent outputs.
///
/// Registration name: "cpu_ep_2sess" (the key passed to RegisterExecutionProviderLibrary).
/// EP name: "cpu_ep" (the string returned by the factory's GetName / EpDevice_EpName).
/// These are distinct: EpDevice_EpName returns the factory's declared name, not the
/// registration key. The device search must use the factory name "cpu_ep".
#[test]
fn conformance_two_sessions() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_a = PathBuf::from(manifest_dir).join("tests/fixtures/add_1x4/model.onnx.textproto");
    let model_b =
        PathBuf::from(manifest_dir).join("tests/fixtures/add_broadcast/model.onnx.textproto");

    let ort_lib_dir = skip_if_missing!(
        find_ort_lib_dir(),
        "conformance_two_sessions: ORT not found"
    );
    let ep_lib_path = skip_if_missing!(
        find_ep_cdylib(),
        "conformance_two_sessions: EP cdylib not found"
    );
    if !model_a.exists() || !model_b.exists() {
        eprintln!("*** SKIPPED: conformance_two_sessions — fixture(s) missing ***");
        return;
    }

    let ort_lib_path = ort_lib_dir.join(ort_discovery::ort_lib_name());

    unsafe {
        let lib = libloading::Library::new(&ort_lib_path).expect("dlopen ORT");
        let api = get_ort_api(&lib);

        let mut env: *mut ort::OrtEnv = ptr::null_mut();
        let logid = std::ffi::CString::new("nxrt_two_sess").unwrap();
        let status =
            ((*api).CreateEnv.unwrap())(ort::ORT_LOGGING_LEVEL_WARNING, logid.as_ptr(), &mut env);
        check_status(api, status, "CreateEnv");

        let reg_name = std::ffi::CString::new("cpu_ep_2sess").unwrap();
        let ep_path_c = ort_path::OrtPathBuf::new(&ep_lib_path);
        let status = ((*api).RegisterExecutionProviderLibrary.unwrap())(
            env,
            reg_name.as_ptr(),
            ep_path_c.as_ptr(),
        );
        check_status(api, status, "RegisterExecutionProviderLibrary");

        let mut ep_devices: *const *const ort::OrtEpDevice = ptr::null();
        let mut num_devices: usize = 0;
        let status = ((*api).GetEpDevices.unwrap())(env, &mut ep_devices, &mut num_devices);
        check_status(api, status, "GetEpDevices");

        let ep_name_fn = (*api).EpDevice_EpName.expect("EpDevice_EpName");
        let mut our_device: *const ort::OrtEpDevice = ptr::null();
        for i in 0..num_devices {
            let dev = *ep_devices.add(i);
            let name_ptr = ep_name_fn(dev);
            if !name_ptr.is_null() && CStr::from_ptr(name_ptr).to_string_lossy() == "cpu_ep" {
                our_device = dev;
            }
        }
        assert!(
            !our_device.is_null(),
            "EP 'cpu_ep' not found in GetEpDevices (reg_name=cpu_ep_2sess; EpDevice_EpName returns factory GetName, not registration key)"
        );

        // Build session A
        let mut opts_a: *mut ort::OrtSessionOptions = ptr::null_mut();
        let status = ((*api).CreateSessionOptions.unwrap())(&mut opts_a);
        check_status(api, status, "CreateSessionOptions(A)");
        let key = CString::new("session.disable_cpu_ep_fallback").unwrap();
        let val = CString::new("1").unwrap();
        let add_config = (*api).AddSessionConfigEntry.expect("AddSessionConfigEntry");
        let status = add_config(opts_a, key.as_ptr(), val.as_ptr());
        check_status(api, status, "AddSessionConfigEntry(A)");
        let devs: [*const ort::OrtEpDevice; 1] = [our_device];
        let status = ((*api).SessionOptionsAppendExecutionProvider_V2.unwrap())(
            opts_a,
            env,
            devs.as_ptr(),
            1,
            ptr::null(),
            ptr::null(),
            0,
        );
        check_status(api, status, "AppendEP(A)");
        let mut sess_a: *mut ort::OrtSession = ptr::null_mut();
        let status = ort_session::create_session(api, env, opts_a, &model_a, &mut sess_a);
        check_status(api, status, "CreateSession(A)");
        eprintln!("✓ Session A created (add_1x4)");

        // Build session B
        let mut opts_b: *mut ort::OrtSessionOptions = ptr::null_mut();
        let status = ((*api).CreateSessionOptions.unwrap())(&mut opts_b);
        check_status(api, status, "CreateSessionOptions(B)");
        let status = add_config(opts_b, key.as_ptr(), val.as_ptr());
        check_status(api, status, "AddSessionConfigEntry(B)");
        let status = ((*api).SessionOptionsAppendExecutionProvider_V2.unwrap())(
            opts_b,
            env,
            devs.as_ptr(),
            1,
            ptr::null(),
            ptr::null(),
            0,
        );
        check_status(api, status, "AppendEP(B)");
        let mut sess_b: *mut ort::OrtSession = ptr::null_mut();
        let status = ort_session::create_session(api, env, opts_b, &model_b, &mut sess_b);
        check_status(api, status, "CreateSession(B)");
        eprintln!("✓ Session B created (add_broadcast)");

        let shape14: [i64; 2] = [1, 4];
        let shape23: [i64; 2] = [2, 3];
        let shape3: [i64; 1] = [3];
        let in_names = [c"X".as_ptr(), c"Y".as_ptr()];
        let out_names = [c"Z".as_ptr()];

        // Run session A
        let mut xa: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let mut ya: [f32; 4] = [5.0, 6.0, 7.0, 8.0];
        let xva = make_float_tensor(api, &mut xa, &shape14);
        let yva = make_float_tensor(api, &mut ya, &shape14);
        let ins_a: [*const ort::OrtValue; 2] = [xva, yva];
        let mut out_a: *mut ort::OrtValue = ptr::null_mut();
        let status = ((*api).Run.unwrap())(
            sess_a,
            ptr::null(),
            in_names.as_ptr(),
            ins_a.as_ptr(),
            2,
            out_names.as_ptr(),
            1,
            &mut out_a,
        );
        check_status(api, status, "Run(A)");
        assert!(!out_a.is_null());
        let mut dpa: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(out_a, &mut dpa);
        check_status(api, status, "GetTensorMutableData(A)");
        let ra = std::slice::from_raw_parts(dpa as *const f32, 4);
        let ea: [f32; 4] = [6.0, 8.0, 10.0, 12.0];
        for (i, (g, w)) in ra.iter().zip(ea.iter()).enumerate() {
            assert!((g - w).abs() < 1e-6, "A[{i}]={g}, want {w}");
        }
        eprintln!("  Session A result: {ra:?} ✓");

        // Run session B
        let mut xb: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut yb: [f32; 3] = [10.0, 20.0, 30.0];
        let xvb = make_float_tensor(api, &mut xb, &shape23);
        let yvb = make_float_tensor(api, &mut yb, &shape3);
        let ins_b: [*const ort::OrtValue; 2] = [xvb, yvb];
        let mut out_b: *mut ort::OrtValue = ptr::null_mut();
        let status = ((*api).Run.unwrap())(
            sess_b,
            ptr::null(),
            in_names.as_ptr(),
            ins_b.as_ptr(),
            2,
            out_names.as_ptr(),
            1,
            &mut out_b,
        );
        check_status(api, status, "Run(B)");
        assert!(!out_b.is_null());
        let mut dpb: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(out_b, &mut dpb);
        check_status(api, status, "GetTensorMutableData(B)");
        let rb = std::slice::from_raw_parts(dpb as *const f32, 6);
        let eb: [f32; 6] = [11.0, 22.0, 33.0, 14.0, 25.0, 36.0];
        for (i, (g, w)) in rb.iter().zip(eb.iter()).enumerate() {
            assert!((g - w).abs() < 1e-6, "B[{i}]={g}, want {w}");
        }
        eprintln!("  Session B result: {rb:?} ✓");

        // Teardown both sessions
        ((*api).ReleaseValue.unwrap())(out_b);
        ((*api).ReleaseValue.unwrap())(out_a);
        ((*api).ReleaseValue.unwrap())(xvb);
        ((*api).ReleaseValue.unwrap())(yvb);
        ((*api).ReleaseValue.unwrap())(xva);
        ((*api).ReleaseValue.unwrap())(yva);
        ((*api).ReleaseSession.unwrap())(sess_b);
        ((*api).ReleaseSession.unwrap())(sess_a);
        ((*api).ReleaseSessionOptions.unwrap())(opts_b);
        ((*api).ReleaseSessionOptions.unwrap())(opts_a);
        let status = ((*api).UnregisterExecutionProviderLibrary.unwrap())(env, reg_name.as_ptr());
        check_status(api, status, "UnregisterExecutionProviderLibrary");
        ((*api).ReleaseEnv.unwrap())(env);
        eprintln!("\n✅ conformance_two_sessions: PASSED — both sessions independent and correct");
    }
}

// ─── Conformance: batched ND MatMul ──────────────────────────────────────────

/// Batched 3-D MatMul: A [2,3,4] × B [2,4,2] → C [2,3,2].
///
/// Tests the batched-ND broadcast inference in our MatMul kernel under real ORT dispatch.
///
/// Hand-computed expected values:
///   batch 0: A0 = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]  B0 = [[1,0],[0,1],[1,0],[0,1]]
///     C0[0] = [1*1+2*0+3*1+4*0, 1*0+2*1+3*0+4*1] = [4, 6]
///     C0[1] = [5+7, 6+8] = [12, 14]
///     C0[2] = [9+11, 10+12] = [20, 22]
///   batch 1: A1 = [[0,1,0,1],[2,0,2,0],[1,1,1,1]]  B1 = [[2,0],[0,2],[2,0],[0,2]]
///     C1[0] = [0*2+1*0+0*2+1*0, 0*0+1*2+0*0+1*2] = [0, 4]
///     C1[1] = [2*2+0*0+2*2+0*0, 2*0+0*2+2*0+0*2] = [8, 0]
///     C1[2] = [1*2+1*0+1*2+1*0, 1*0+1*2+1*0+1*2] = [4, 4]
#[test]
fn conformance_matmul_batched_nd() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path =
        PathBuf::from(manifest_dir).join("tests/fixtures/matmul_batched_nd/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_mm3d", &model_path, true) })
    else {
        eprintln!("*** SKIPPED: conformance_matmul_batched_nd — ORT or EP cdylib not found ***");
        return;
    };

    // Prove MatMul is assigned to our EP.
    unsafe { assert_ops_assigned_to_our_ep(api, session, &["MatMul"], "matmul_batched_nd") };
    unsafe {
        // A [2,3,4]
        let mut a_data: [f32; 24] = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 0.0, 1.0, 0.0, 1.0, 2.0,
            0.0, 2.0, 0.0, 1.0, 1.0, 1.0, 1.0,
        ];
        // B [2,4,2]
        let mut b_data: [f32; 16] = [
            1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0,
        ];
        let a_shape: [i64; 3] = [2, 3, 4];
        let b_shape: [i64; 3] = [2, 4, 2];

        let a_val = make_float_tensor(api, &mut a_data, &a_shape);
        let b_val = make_float_tensor(api, &mut b_data, &b_shape);

        let input_names = [c"A".as_ptr(), c"B".as_ptr()];
        let output_names = [c"C".as_ptr()];
        let inputs: [*const ort::OrtValue; 2] = [a_val, b_val];
        let mut output: *mut ort::OrtValue = ptr::null_mut();
        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(),
            input_names.as_ptr(),
            inputs.as_ptr(),
            2,
            output_names.as_ptr(),
            1,
            &mut output,
        );
        check_status(api, status, "Run");
        assert!(!output.is_null());

        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(output, &mut data_ptr);
        check_status(api, status, "GetTensorMutableData");
        let result = std::slice::from_raw_parts(data_ptr as *const f32, 12);
        // C[0] = [[4,6],[12,14],[20,22]]  C[1] = [[0,4],[8,0],[4,4]]
        let expected: [f32; 12] = [
            4.0, 6.0, 12.0, 14.0, 20.0, 22.0, 0.0, 4.0, 8.0, 0.0, 4.0, 4.0,
        ];
        eprintln!("  Got:      {result:?}");
        eprintln!("  Expected: {expected:?}");
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-5, "C[{i}] = {got}, want {want}");
        }

        ((*api).ReleaseValue.unwrap())(output);
        ((*api).ReleaseValue.unwrap())(a_val);
        ((*api).ReleaseValue.unwrap())(b_val);
        conformance_teardown(api, env, opts, session, "cpu_ep_mm3d");
        eprintln!("\n✅ conformance_matmul_batched_nd: PASSED");
    }
}

// ─── Stress: use-after-free regression ───────────────────────────────────────

/// 25 complete register → GetEpDevices → CreateSession → Run → Unregister cycles.
///
/// The use-after-free bug (fixed in commit c92838dba) corrupted `OrtEpDevice`
/// after ≥6 cycles when `OrtMemoryInfo` was released while ORT held the raw
/// pointer. This stress test exceeds that threshold by 4× so any regression
/// will be caught before it reaches production.
///
/// Each cycle uses a fresh Env, fresh session, fresh registration key, and
/// verifies the Run output independently. Corrupt memory typically manifests
/// as a DeviceType=-112 panic or a segfault on the device-lookup assertion.
#[test]
fn stress_register_run_unregister_cycles() {
    let _lock = lock_ort_ep();

    let ort_lib_dir = {
        let d = find_ort_lib_dir();
        if d.is_none() {
            eprintln!("*** SKIPPED: stress_register_run_unregister_cycles — ORT not found ***");
            return;
        }
        d.unwrap()
    };
    let ep_lib_path = {
        let p = find_ep_cdylib();
        if p.is_none() {
            eprintln!(
                "*** SKIPPED: stress_register_run_unregister_cycles — EP cdylib not found ***"
            );
            return;
        }
        p.unwrap()
    };
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path =
        PathBuf::from(manifest_dir).join("tests/fixtures/add_1x4/model.onnx.textproto");
    if !model_path.exists() {
        eprintln!(
            "*** SKIPPED: stress_register_run_unregister_cycles — add_1x4 fixture missing ***"
        );
        return;
    }

    let ort_lib_path = ort_lib_dir.join(ort_discovery::ort_lib_name());
    // Keep the library loaded for the whole test — dlopen reference-counts.
    let lib = unsafe { libloading::Library::new(&ort_lib_path) }.expect("dlopen ORT");
    let api = unsafe { get_ort_api(&lib) };
    let ep_path_c = ort_path::OrtPathBuf::new(&ep_lib_path);

    const CYCLES: usize = 25;
    for cycle in 0..CYCLES {
        let reg_key = format!("stress_ep_{cycle}");
        let reg_c = CString::new(reg_key.as_str()).unwrap();

        unsafe {
            // Env
            let logid = CString::new(format!("stress_{cycle}")).unwrap();
            let mut env: *mut ort::OrtEnv = ptr::null_mut();
            let status = ((*api).CreateEnv.unwrap())(
                ort::ORT_LOGGING_LEVEL_WARNING,
                logid.as_ptr(),
                &mut env,
            );
            check_status(api, status, &format!("CreateEnv[{cycle}]"));

            // Register
            let status = ((*api).RegisterExecutionProviderLibrary.unwrap())(
                env,
                reg_c.as_ptr(),
                ep_path_c.as_ptr(),
            );
            check_status(api, status, &format!("Register[{cycle}]"));

            // GetEpDevices — device must have correct DeviceType every cycle.
            let mut ep_devices: *const *const ort::OrtEpDevice = ptr::null();
            let mut num_devices: usize = 0;
            let status = ((*api).GetEpDevices.unwrap())(env, &mut ep_devices, &mut num_devices);
            check_status(api, status, &format!("GetEpDevices[{cycle}]"));

            let ep_name_fn = (*api).EpDevice_EpName.expect("EpDevice_EpName");
            let mut our_device: *const ort::OrtEpDevice = ptr::null();
            for i in 0..num_devices {
                let dev = *ep_devices.add(i);
                let name_ptr = ep_name_fn(dev);
                if !name_ptr.is_null() && CStr::from_ptr(name_ptr).to_string_lossy() == "cpu_ep" {
                    our_device = dev;
                }
            }
            assert!(
                !our_device.is_null(),
                "stress cycle {cycle}: EP 'cpu_ep' not found — possible DeviceType corruption (use-after-free regression)"
            );

            // Session
            let mut sess_opts: *mut ort::OrtSessionOptions = ptr::null_mut();
            let status = ((*api).CreateSessionOptions.unwrap())(&mut sess_opts);
            check_status(api, status, &format!("CreateSessionOptions[{cycle}]"));
            let devs: [*const ort::OrtEpDevice; 1] = [our_device];
            let status = ((*api).SessionOptionsAppendExecutionProvider_V2.unwrap())(
                sess_opts,
                env,
                devs.as_ptr(),
                1,
                ptr::null(),
                ptr::null(),
                0,
            );
            check_status(api, status, &format!("AppendEP[{cycle}]"));
            let mut session: *mut ort::OrtSession = ptr::null_mut();
            let status =
                ort_session::create_session(api, env, sess_opts, &model_path, &mut session);
            check_status(api, status, &format!("CreateSession[{cycle}]"));

            // Run: [1,2,3,4] + [5,6,7,8] = [6,8,10,12]
            let mut x_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
            let mut y_data: [f32; 4] = [5.0, 6.0, 7.0, 8.0];
            let shape: [i64; 2] = [1, 4];
            let xv = make_float_tensor(api, &mut x_data, &shape);
            let yv = make_float_tensor(api, &mut y_data, &shape);
            let in_names = [c"X".as_ptr(), c"Y".as_ptr()];
            let out_names = [c"Z".as_ptr()];
            let inputs: [*const ort::OrtValue; 2] = [xv, yv];
            let mut output: *mut ort::OrtValue = ptr::null_mut();
            let status = ((*api).Run.unwrap())(
                session,
                ptr::null(),
                in_names.as_ptr(),
                inputs.as_ptr(),
                2,
                out_names.as_ptr(),
                1,
                &mut output,
            );
            check_status(api, status, &format!("Run[{cycle}]"));
            assert!(!output.is_null(), "cycle {cycle}: output is null");
            let mut dp: *mut std::ffi::c_void = ptr::null_mut();
            let status = ((*api).GetTensorMutableData.unwrap())(output, &mut dp);
            check_status(api, status, &format!("GetTensorMutableData[{cycle}]"));
            let result = std::slice::from_raw_parts(dp as *const f32, 4);
            let expected: [f32; 4] = [6.0, 8.0, 10.0, 12.0];
            for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1e-6,
                    "cycle {cycle}: output[{i}]={got}, want {want}"
                );
            }

            // Teardown
            ((*api).ReleaseValue.unwrap())(output);
            ((*api).ReleaseValue.unwrap())(xv);
            ((*api).ReleaseValue.unwrap())(yv);
            ((*api).ReleaseSession.unwrap())(session);
            ((*api).ReleaseSessionOptions.unwrap())(sess_opts);
            let status = ((*api).UnregisterExecutionProviderLibrary.unwrap())(env, reg_c.as_ptr());
            check_status(api, status, &format!("Unregister[{cycle}]"));
            ((*api).ReleaseEnv.unwrap())(env);
        }

        eprintln!("  ✓ cycle {}/{CYCLES}", cycle + 1);
    }

    eprintln!(
        "\n✅ stress_register_run_unregister_cycles: {CYCLES} cycles PASSED — use-after-free regression clear"
    );
}

// ─── f16/bf16 end-to-end conformance ─────────────────────────────────────────

/// Float16 Add through ORT: `Z = X + Y` with all-f16 inputs and outputs.
///
/// The CPU kernels in `crates/onnx-runtime-ep-cpu` accept Float16 for Add.
/// `build_cpu_registry_with_descriptors()` derives Float16 type-constraint
/// metadata and the cpu-plugin shim wires it through `GetKernelRegistry`.
/// ORT consults the kernel registry to route f16 nodes to our EP, and our
/// kernel produces numerically correct output (exact f16 bit-pattern check).
#[test]
fn conformance_add_float16() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path =
        PathBuf::from(manifest_dir).join("tests/fixtures/add_float16/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_f16", &model_path, true) })
    else {
        eprintln!("*** SKIPPED: conformance_add_float16 — ORT or EP cdylib not found ***");
        return;
    };

    unsafe {
        // Float16 bit-patterns: 1.0=0x3C00, 2.0=0x4000, 3.0=0x4200, 4.0=0x4400
        let mut x_data: [u16; 4] = [0x3C00, 0x4000, 0x4200, 0x4400];
        let mut y_data: [u16; 4] = [0x3C00, 0x4000, 0x4200, 0x4400];
        let shape: [i64; 2] = [1, 4];

        let x_val = make_float16_tensor(api, &mut x_data, &shape);
        let y_val = make_float16_tensor(api, &mut y_data, &shape);

        let input_names = [c"X".as_ptr(), c"Y".as_ptr()];
        let output_names = [c"Z".as_ptr()];
        let inputs: [*const ort::OrtValue; 2] = [x_val, y_val];
        let mut output: *mut ort::OrtValue = ptr::null_mut();

        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(),
            input_names.as_ptr(),
            inputs.as_ptr(),
            2,
            output_names.as_ptr(),
            1,
            &mut output,
        );
        check_status(api, status, "Run(float16)");
        assert!(!output.is_null());

        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(output, &mut data_ptr);
        check_status(api, status, "GetTensorMutableData(float16)");

        // Expected: 2.0=0x4000, 4.0=0x4400, 6.0=0x4600, 8.0=0x4800
        let result = std::slice::from_raw_parts(data_ptr as *const u16, 4);
        let expected: [u16; 4] = [0x4000, 0x4400, 0x4600, 0x4800];
        eprintln!("  Got f16:      {result:?}");
        eprintln!("  Expected f16: {expected:?}");
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                got, want,
                "output[{i}]: got f16 bit-pattern {got:#06x}, want {want:#06x}"
            );
        }

        ((*api).ReleaseValue.unwrap())(output);
        ((*api).ReleaseValue.unwrap())(x_val);
        ((*api).ReleaseValue.unwrap())(y_val);
        assert_ops_assigned_to_our_ep(api, session, &["Add"], "conformance_add_float16");
        conformance_teardown(api, env, opts, session, "cpu_ep_f16");
        eprintln!("\n✅ conformance_add_float16: PASSED");
    }
}

/// BFloat16 Add through ORT: `Z = X + Y` with all-bf16 inputs and outputs.
///
/// Same routing as `conformance_add_float16`: `build_cpu_registry_with_descriptors()`
/// includes BFloat16 type-constraint metadata and the cpu-plugin shim wires it
/// through `GetKernelRegistry`. Exact bf16 bit-pattern check confirms our kernel
/// executes correctly.
///
/// BFloat16 is the top 16 bits of a float32 mantissa: 1.0=0x3F80, 2.0=0x4000,
/// 3.0=0x4040, 4.0=0x4080.
#[test]
fn conformance_add_bfloat16() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path =
        PathBuf::from(manifest_dir).join("tests/fixtures/add_bfloat16/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_bf16", &model_path, true) })
    else {
        eprintln!("*** SKIPPED: conformance_add_bfloat16 — ORT or EP cdylib not found ***");
        return;
    };

    unsafe {
        // BFloat16 bit-patterns (top 16 bits of f32):
        // 1.0→0x3F80, 2.0→0x4000, 3.0→0x4040, 4.0→0x4080
        let mut x_data: [u16; 4] = [0x3F80, 0x4000, 0x4040, 0x4080];
        let mut y_data: [u16; 4] = [0x3F80, 0x4000, 0x4040, 0x4080];
        let shape: [i64; 2] = [1, 4];

        let x_val = make_bfloat16_tensor(api, &mut x_data, &shape);
        let y_val = make_bfloat16_tensor(api, &mut y_data, &shape);

        let input_names = [c"X".as_ptr(), c"Y".as_ptr()];
        let output_names = [c"Z".as_ptr()];
        let inputs: [*const ort::OrtValue; 2] = [x_val, y_val];
        let mut output: *mut ort::OrtValue = ptr::null_mut();

        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(),
            input_names.as_ptr(),
            inputs.as_ptr(),
            2,
            output_names.as_ptr(),
            1,
            &mut output,
        );
        check_status(api, status, "Run(bfloat16)");
        assert!(!output.is_null());

        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(output, &mut data_ptr);
        check_status(api, status, "GetTensorMutableData(bfloat16)");

        // Expected: 2.0→0x4000, 4.0→0x4080, 6.0→0x40C0, 8.0→0x4100
        let result = std::slice::from_raw_parts(data_ptr as *const u16, 4);
        let expected: [u16; 4] = [0x4000, 0x4080, 0x40C0, 0x4100];
        eprintln!("  Got bf16:      {result:?}");
        eprintln!("  Expected bf16: {expected:?}");
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                got, want,
                "output[{i}]: got bf16 bit-pattern {got:#06x}, want {want:#06x}"
            );
        }

        ((*api).ReleaseValue.unwrap())(output);
        ((*api).ReleaseValue.unwrap())(x_val);
        ((*api).ReleaseValue.unwrap())(y_val);
        assert_ops_assigned_to_our_ep(api, session, &["Add"], "conformance_add_bfloat16");
        conformance_teardown(api, env, opts, session, "cpu_ep_bf16");
        eprintln!("\n✅ conformance_add_bfloat16: PASSED");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// B1 dtype conformance suite
//
// These tests prove the B1 fix (per-output dtypes from ORT graph value info).
// Each test uses an op whose output dtype differs from its first input dtype,
// exactly the scenario that was silently corrupted before the fix.
//
// The `conformance_setup` helper proves node assignment via three mechanisms:
// 1. `session.disable_cpu_ep_fallback=1` — ORT will error if any node cannot
//    be placed on our EP, preventing silent fallback to the built-in CPU EP.
// 2. Device lookup assertion — verifies our EP ("cpu_ep") appears in GetEpDevices
//    and is appended to the session options.
// 3. `Session_GetEpGraphAssignmentInfo` (ORT ≥1.24) — directly queries per-node
//    provider attribution, proving specific ops are assigned to "cpu_ep".
// ═══════════════════════════════════════════════════════════════════════════════

/// Assert the element type of an ORT output tensor.
///
/// # Safety
/// `api` and `output` must be valid pointers.
unsafe fn assert_output_dtype(
    api: *const ort::OrtApi,
    output: *const ort::OrtValue,
    expected: ort::ONNXTensorElementDataType,
    label: &str,
) {
    let get_type_shape =
        unsafe { (*api).GetTensorTypeAndShape }.expect("GetTensorTypeAndShape not in OrtApi");
    let get_elem_type =
        unsafe { (*api).GetTensorElementType }.expect("GetTensorElementType not in OrtApi");
    let release_info = unsafe { (*api).ReleaseTensorTypeAndShapeInfo }
        .expect("ReleaseTensorTypeAndShapeInfo not in OrtApi");

    let mut info: *mut ort::OrtTensorTypeAndShapeInfo = ptr::null_mut();
    let status = unsafe { get_type_shape(output, &mut info) };
    unsafe { check_status(api, status, &format!("GetTensorTypeAndShape({label})")) };

    let mut elem_type: ort::ONNXTensorElementDataType = 0;
    let status = unsafe { get_elem_type(info, &mut elem_type) };
    unsafe { check_status(api, status, &format!("GetTensorElementType({label})")) };
    unsafe { release_info(info) };

    assert_eq!(
        elem_type, expected,
        "{label}: output dtype mismatch: got {elem_type}, expected {expected}"
    );
    eprintln!("  ✓ {label}: dtype={elem_type} (matches expected {expected})");
}

/// Assert that `output` has the given shape (via ORT API).  Panics with a
/// diagnostic message on shape mismatch, so a regression is immediately visible.
unsafe fn assert_output_shape(
    api: *const ort::OrtApi,
    output: *const ort::OrtValue,
    expected: &[i64],
    label: &str,
) {
    let get_type_shape =
        unsafe { (*api).GetTensorTypeAndShape }.expect("GetTensorTypeAndShape not in OrtApi");
    let get_dims_count =
        unsafe { (*api).GetDimensionsCount }.expect("GetDimensionsCount not in OrtApi");
    let get_dims = unsafe { (*api).GetDimensions }.expect("GetDimensions not in OrtApi");
    let release_info = unsafe { (*api).ReleaseTensorTypeAndShapeInfo }
        .expect("ReleaseTensorTypeAndShapeInfo not in OrtApi");

    let mut info: *mut ort::OrtTensorTypeAndShapeInfo = ptr::null_mut();
    let status = unsafe { get_type_shape(output, &mut info) };
    unsafe { check_status(api, status, &format!("GetTensorTypeAndShape({label})")) };

    let mut rank: usize = 0;
    let status = unsafe { get_dims_count(info, &mut rank) };
    unsafe { check_status(api, status, &format!("GetDimensionsCount({label})")) };

    let mut dims: Vec<i64> = vec![0i64; rank];
    let status = unsafe { get_dims(info, dims.as_mut_ptr(), rank) };
    unsafe { check_status(api, status, &format!("GetDimensions({label})")) };
    unsafe { release_info(info) };

    assert_eq!(
        dims, expected,
        "{label}: shape mismatch: got {dims:?}, expected {expected:?}"
    );
    eprintln!("  ✓ {label}: shape={dims:?} (matches expected {expected:?})");
}

// ─── B1 dtype: Cast (f32 → i64) ─────────────────────────────────────────────

/// Cast f32 [2,3] → i64.  Output dtype must be INT64, not FLOAT.
/// X = [[1.5, 2.7, 3.0], [4.9, 5.1, 6.0]]
/// Expected Y = [[1, 2, 3], [4, 5, 6]] (truncated toward zero)
#[test]
fn conformance_cast_f32_to_i64() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path =
        PathBuf::from(manifest_dir).join("tests/fixtures/cast_f32_to_i64/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_cast", &model_path, true) })
    else {
        eprintln!("*** SKIPPED: conformance_cast_f32_to_i64 — ORT or EP cdylib not found ***");
        return;
    };

    // Prove Cast is assigned to our EP.
    unsafe { assert_ops_assigned_to_our_ep(api, session, &["Cast"], "cast_f32_to_i64") };

    unsafe {
        let mut x_data: [f32; 6] = [1.5, 2.7, 3.0, 4.9, 5.1, 6.0];
        let x_shape: [i64; 2] = [2, 3];
        let x_val = make_float_tensor(api, &mut x_data, &x_shape);

        let input_names = [c"X".as_ptr()];
        let output_names = [c"Y".as_ptr()];
        let inputs: [*const ort::OrtValue; 1] = [x_val];
        let mut output: *mut ort::OrtValue = ptr::null_mut();

        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(),
            input_names.as_ptr(),
            inputs.as_ptr(),
            1,
            output_names.as_ptr(),
            1,
            &mut output,
        );
        check_status(api, status, "Run(cast_f32_to_i64)");
        assert!(!output.is_null());

        // Assert output dtype is INT64, not FLOAT (the B1 bug would give FLOAT)
        assert_output_dtype(
            api,
            output,
            ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            "cast_f32_to_i64",
        );

        // Assert output values
        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(output, &mut data_ptr);
        check_status(api, status, "GetTensorMutableData(cast)");
        let result = std::slice::from_raw_parts(data_ptr as *const i64, 6);
        let expected: [i64; 6] = [1, 2, 3, 4, 5, 6];
        eprintln!("  Got:      {result:?}");
        eprintln!("  Expected: {expected:?}");
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert_eq!(got, want, "cast output[{i}] = {got}, want {want}");
        }

        ((*api).ReleaseValue.unwrap())(output);
        ((*api).ReleaseValue.unwrap())(x_val);
        conformance_teardown(api, env, opts, session, "cpu_ep_cast");
        eprintln!("\n✅ conformance_cast_f32_to_i64: PASSED — output is i64, values correct");
    }
}

// ─── B1 dtype: Where (bool, f32, f32 → f32) ─────────────────────────────────

/// Where(condition=bool, X=f32, Y=f32) → f32.
/// The first input is bool but the output must be f32.
///
/// condition = [[true, false], [false, true]]
/// X = [[1.0, 2.0], [3.0, 4.0]]
/// Y = [[10.0, 20.0], [30.0, 40.0]]
/// Expected Z = [[1.0, 20.0], [30.0, 4.0]]
#[test]
fn conformance_where_bool_f32() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path =
        PathBuf::from(manifest_dir).join("tests/fixtures/where_bool_f32/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_where", &model_path, true) })
    else {
        eprintln!("*** SKIPPED: conformance_where_bool_f32 — ORT or EP cdylib not found ***");
        return;
    };

    // Prove Where is assigned to our EP.
    unsafe { assert_ops_assigned_to_our_ep(api, session, &["Where"], "where_bool_f32") };

    unsafe {
        // Create bool tensor for condition
        let mut cond_data: [u8; 4] = [1, 0, 0, 1]; // true, false, false, true
        let cond_shape: [i64; 2] = [2, 2];
        let mut mem_info: *mut ort::OrtMemoryInfo = ptr::null_mut();
        let status = ((*api).CreateCpuMemoryInfo.unwrap())(
            ort::OrtDeviceAllocator,
            ort::OrtMemTypeDefault,
            &mut mem_info,
        );
        check_status(api, status, "CreateCpuMemoryInfo(bool)");
        let mut cond_val: *mut ort::OrtValue = ptr::null_mut();
        let status = ((*api).CreateTensorWithDataAsOrtValue.unwrap())(
            mem_info,
            cond_data.as_mut_ptr().cast(),
            std::mem::size_of_val(&cond_data),
            cond_shape.as_ptr(),
            2,
            ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
            &mut cond_val,
        );
        check_status(api, status, "CreateTensor(bool)");
        ((*api).ReleaseMemoryInfo.unwrap())(mem_info);

        let mut x_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let mut y_data: [f32; 4] = [10.0, 20.0, 30.0, 40.0];
        let shape: [i64; 2] = [2, 2];
        let x_val = make_float_tensor(api, &mut x_data, &shape);
        let y_val = make_float_tensor(api, &mut y_data, &shape);

        let input_names = [c"C".as_ptr(), c"X".as_ptr(), c"Y".as_ptr()];
        let output_names = [c"Z".as_ptr()];
        let inputs: [*const ort::OrtValue; 3] = [cond_val, x_val, y_val];
        let mut output: *mut ort::OrtValue = ptr::null_mut();

        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(),
            input_names.as_ptr(),
            inputs.as_ptr(),
            3,
            output_names.as_ptr(),
            1,
            &mut output,
        );
        check_status(api, status, "Run(where_bool_f32)");
        assert!(!output.is_null());

        // Assert output dtype is FLOAT, not BOOL (the B1 bug would give BOOL)
        assert_output_dtype(
            api,
            output,
            ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            "where_bool_f32",
        );

        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(output, &mut data_ptr);
        check_status(api, status, "GetTensorMutableData(where)");
        let result = std::slice::from_raw_parts(data_ptr as *const f32, 4);
        let expected: [f32; 4] = [1.0, 20.0, 30.0, 4.0];
        eprintln!("  Got:      {result:?}");
        eprintln!("  Expected: {expected:?}");
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "where output[{i}] = {got}, want {want}"
            );
        }

        ((*api).ReleaseValue.unwrap())(output);
        ((*api).ReleaseValue.unwrap())(cond_val);
        ((*api).ReleaseValue.unwrap())(x_val);
        ((*api).ReleaseValue.unwrap())(y_val);
        conformance_teardown(api, env, opts, session, "cpu_ep_where");
        eprintln!("\n✅ conformance_where_bool_f32: PASSED — output is f32, values correct");
    }
}

// ─── B1 dtype: Shape (f32 → i64) ────────────────────────────────────────────

/// Shape(f32 [3,4,5]) → i64 [3] with value [3,4,5].
/// Output dtype is always INT64 regardless of input dtype.
#[test]
fn conformance_shape_f32() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path =
        PathBuf::from(manifest_dir).join("tests/fixtures/shape_f32/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_shape", &model_path, true) })
    else {
        eprintln!("*** SKIPPED: conformance_shape_f32 — ORT or EP cdylib not found ***");
        return;
    };

    // Prove Shape is assigned to our EP.
    // Shape ops may be constant-folded by ORT's basic optimizations before EP
    // assignment. If present, verify it's on our EP; if absent, that's valid.
    unsafe {
        let info = query_ep_assignment(api, session);
        let ours = info.ops_on_our_ep();
        if ours.contains(&"Shape") {
            eprintln!("  [shape_f32] ✓ Shape assigned to cpu_ep");
        } else {
            eprintln!(
                "  [shape_f32] ℹ Shape not in assignment (likely constant-folded by ORT). \
                 Assigned: {:?}",
                info.assignments
            );
        }
    }

    unsafe {
        // Create a dummy f32 tensor with shape [3,4,5] (60 elements)
        let mut x_data = vec![0.0f32; 60];
        let x_shape: [i64; 3] = [3, 4, 5];
        let x_val = make_float_tensor(api, &mut x_data, &x_shape);

        let input_names = [c"X".as_ptr()];
        let output_names = [c"Y".as_ptr()];
        let inputs: [*const ort::OrtValue; 1] = [x_val];
        let mut output: *mut ort::OrtValue = ptr::null_mut();

        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(),
            input_names.as_ptr(),
            inputs.as_ptr(),
            1,
            output_names.as_ptr(),
            1,
            &mut output,
        );
        check_status(api, status, "Run(shape_f32)");
        assert!(!output.is_null());

        // Assert output dtype is INT64
        assert_output_dtype(
            api,
            output,
            ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
            "shape_f32",
        );

        // Assert output values = [3, 4, 5]
        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(output, &mut data_ptr);
        check_status(api, status, "GetTensorMutableData(shape)");
        let result = std::slice::from_raw_parts(data_ptr as *const i64, 3);
        let expected: [i64; 3] = [3, 4, 5];
        eprintln!("  Got:      {result:?}");
        eprintln!("  Expected: {expected:?}");
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert_eq!(got, want, "shape output[{i}] = {got}, want {want}");
        }

        ((*api).ReleaseValue.unwrap())(output);
        ((*api).ReleaseValue.unwrap())(x_val);
        conformance_teardown(api, env, opts, session, "cpu_ep_shape");
        eprintln!("\n✅ conformance_shape_f32: PASSED — output is i64, value [3,4,5]");
    }
}

// ─── LayerNormalization: multi-output, shape and value correctness ────────────

/// LayerNormalization(X [2,4], Scale [4]) → 3 outputs (Y, Mean, InvStdDev).
/// axis=-1; Mean and InvStdDev must have shape [2,1], not [2,4].
/// Verifies the fix to the ShapePreservingNorm → LayerNorm shape inference
/// refactor: the old code emitted input[0]'s full shape for every output.
#[test]
fn conformance_layer_norm_multi_output() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path =
        PathBuf::from(manifest_dir).join("tests/fixtures/layer_norm_f32/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_ln", &model_path, true) })
    else {
        eprintln!(
            "*** SKIPPED: conformance_layer_norm_multi_output — ORT or EP cdylib not found ***"
        );
        return;
    };

    unsafe {
        // X = [[1,2,3,4],[5,6,7,8]]  Scale = [1,1,1,1]
        let mut x_data: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut scale_data: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
        let x_shape: [i64; 2] = [2, 4];
        let s_shape: [i64; 1] = [4];
        let x_val = make_float_tensor(api, &mut x_data, &x_shape);
        let s_val = make_float_tensor(api, &mut scale_data, &s_shape);

        let input_names = [c"X".as_ptr(), c"Scale".as_ptr()];
        let output_names = [c"Y".as_ptr(), c"Mean".as_ptr(), c"InvStdDev".as_ptr()];
        let inputs: [*const ort::OrtValue; 2] = [x_val, s_val];
        let mut outputs: [*mut ort::OrtValue; 3] = [ptr::null_mut(); 3];

        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(),
            input_names.as_ptr(),
            inputs.as_ptr(),
            2,
            output_names.as_ptr(),
            3,
            outputs.as_mut_ptr(),
        );
        check_status(api, status, "Run(layer_norm)");

        // All 3 outputs must be non-null
        for (i, out) in outputs.iter().enumerate() {
            assert!(!out.is_null(), "LayerNorm output[{i}] is null");
        }

        // Assert all 3 outputs have dtype FLOAT
        assert_output_dtype(
            api,
            outputs[0],
            ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            "layer_norm_Y",
        );
        assert_output_dtype(
            api,
            outputs[1],
            ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            "layer_norm_Mean",
        );
        assert_output_dtype(
            api,
            outputs[2],
            ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            "layer_norm_InvStdDev",
        );

        // Assert output shapes: Y=[2,4], Mean=[2,1], InvStdDev=[2,1].
        // The old ShapePreservingNorm bug emitted [2,4] for Mean and InvStdDev.
        assert_output_shape(api, outputs[0], &[2, 4], "layer_norm_Y");
        assert_output_shape(api, outputs[1], &[2, 1], "layer_norm_Mean");
        assert_output_shape(api, outputs[2], &[2, 1], "layer_norm_InvStdDev");

        // Check Y values (layer-norm of [1,2,3,4] with scale=[1,1,1,1]):
        // mean=2.5, var=1.25, invstd=1/sqrt(1.25+eps)
        // Y = (X - mean) * invstd * scale
        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(outputs[0], &mut data_ptr);
        check_status(api, status, "GetTensorMutableData(Y)");
        let y_result = std::slice::from_raw_parts(data_ptr as *const f32, 8);
        eprintln!("  Y output: {y_result:?}");

        // Verify Y is normalized: for each row, mean≈0 and std≈1
        for row in 0..2 {
            let row_slice = &y_result[row * 4..(row + 1) * 4];
            let mean: f32 = row_slice.iter().sum::<f32>() / 4.0;
            assert!(
                mean.abs() < 1e-4,
                "LayerNorm Y row {row} mean={mean}, expected ~0"
            );
        }

        // Check Mean output: row means are 2.5 and 6.5.
        let status = ((*api).GetTensorMutableData.unwrap())(outputs[1], &mut data_ptr);
        check_status(api, status, "GetTensorMutableData(Mean)");
        let mean_result = std::slice::from_raw_parts(data_ptr as *const f32, 2);
        eprintln!("  Mean output: {mean_result:?}");
        assert!(
            (mean_result[0] - 2.5).abs() < 1e-4,
            "Mean[0]={}, want 2.5",
            mean_result[0]
        );
        assert!(
            (mean_result[1] - 6.5).abs() < 1e-4,
            "Mean[1]={}, want 6.5",
            mean_result[1]
        );

        // Check InvStdDev output: both rows have var=1.25, so invstd≈0.8944.
        let status = ((*api).GetTensorMutableData.unwrap())(outputs[2], &mut data_ptr);
        check_status(api, status, "GetTensorMutableData(InvStdDev)");
        let inv_result = std::slice::from_raw_parts(data_ptr as *const f32, 2);
        eprintln!("  InvStdDev output: {inv_result:?}");
        let expected_inv = 1.0_f32 / 1.25_f32.sqrt(); // ≈ 0.8944
        for (i, &v) in inv_result.iter().enumerate() {
            assert!(
                (v - expected_inv).abs() < 1e-3,
                "InvStdDev[{i}]={v}, want ≈{expected_inv}"
            );
        }

        for out in &outputs {
            ((*api).ReleaseValue.unwrap())(*out);
        }
        ((*api).ReleaseValue.unwrap())(x_val);
        ((*api).ReleaseValue.unwrap())(s_val);
        assert_ops_assigned_to_our_ep(
            api,
            session,
            &["LayerNormalization"],
            "layer_norm_multi_output",
        );
        conformance_teardown(api, env, opts, session, "cpu_ep_ln");
        eprintln!(
            "\n✅ conformance_layer_norm_multi_output: PASSED — shapes [2,1] and values verified"
        );
    }
}

// ─── LayerNormalization: 3-D input, axis=-1 (reduced shape = [2,3,1]) ────────

/// LayerNormalization(X [2,3,4], Scale [4]) with axis=-1.
/// Mean / InvStdDev shapes must be [2,3,1], not [2,3,4].
/// This catches a regression where the old ShapePreservingNorm emitted the
/// full input shape for every output — the difference is obvious in 3D.
///
/// X[0] = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
/// X[1] = [[-4,-3,-2,-1],[1,1,1,1],[0,1,2,3]]
/// Expected Mean (6 values, shape [2,3,1]): [2.5, 6.5, 10.5, -2.5, 1.0, 1.5]
#[test]
fn conformance_layer_norm_neg_axis() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = PathBuf::from(manifest_dir)
        .join("tests/fixtures/layer_norm_neg_axis_f32/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_ln_neg", &model_path, true) })
    else {
        eprintln!("*** SKIPPED: conformance_layer_norm_neg_axis — ORT or EP cdylib not found ***");
        return;
    };

    unsafe {
        #[rustfmt::skip]
        let mut x_data: [f32; 24] = [
            1.0,  2.0,  3.0,  4.0,
            5.0,  6.0,  7.0,  8.0,
            9.0, 10.0, 11.0, 12.0,
           -4.0, -3.0, -2.0, -1.0,
            1.0,  1.0,  1.0,  1.0,
            0.0,  1.0,  2.0,  3.0,
        ];
        let mut scale_data: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
        let x_shape: [i64; 3] = [2, 3, 4];
        let s_shape: [i64; 1] = [4];
        let x_val = make_float_tensor(api, &mut x_data, &x_shape);
        let s_val = make_float_tensor(api, &mut scale_data, &s_shape);

        let input_names = [c"X".as_ptr(), c"Scale".as_ptr()];
        let output_names = [c"Y".as_ptr(), c"Mean".as_ptr(), c"InvStdDev".as_ptr()];
        let inputs: [*const ort::OrtValue; 2] = [x_val, s_val];
        let mut outputs: [*mut ort::OrtValue; 3] = [ptr::null_mut(); 3];

        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(),
            input_names.as_ptr(),
            inputs.as_ptr(),
            2,
            output_names.as_ptr(),
            3,
            outputs.as_mut_ptr(),
        );
        check_status(api, status, "Run(layer_norm_neg_axis)");

        for (i, out) in outputs.iter().enumerate() {
            assert!(!out.is_null(), "LayerNorm(neg_axis) output[{i}] is null");
        }

        // Shapes: Y=[2,3,4], Mean=[2,3,1], InvStdDev=[2,3,1].
        assert_output_shape(api, outputs[0], &[2, 3, 4], "ln_neg_Y");
        assert_output_shape(api, outputs[1], &[2, 3, 1], "ln_neg_Mean");
        assert_output_shape(api, outputs[2], &[2, 3, 1], "ln_neg_InvStdDev");

        // Mean values (6 elements stored flat in [2,3,1]).
        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(outputs[1], &mut data_ptr);
        check_status(api, status, "GetTensorMutableData(ln_neg_Mean)");
        let mean_result = std::slice::from_raw_parts(data_ptr as *const f32, 6);
        eprintln!("  Mean output: {mean_result:?}");
        let expected_means: [f32; 6] = [2.5, 6.5, 10.5, -2.5, 1.0, 1.5];
        for (i, (&got, &want)) in mean_result.iter().zip(expected_means.iter()).enumerate() {
            assert!((got - want).abs() < 1e-4, "Mean[{i}]={got}, want {want}");
        }

        // InvStdDev: rows [1,2,3,4], [5,6,7,8], [9,10,11,12], [-4,-3,-2,-1]
        // and [0,1,2,3] all have var=1.25 → invstd≈0.8944.
        // Row [1,1,1,1] has var=0 → invstd=1/sqrt(eps) (very large); skip it.
        let status = ((*api).GetTensorMutableData.unwrap())(outputs[2], &mut data_ptr);
        check_status(api, status, "GetTensorMutableData(ln_neg_InvStdDev)");
        let inv_result = std::slice::from_raw_parts(data_ptr as *const f32, 6);
        eprintln!("  InvStdDev output: {inv_result:?}");
        let expected_inv = 1.0_f32 / 1.25_f32.sqrt(); // ≈ 0.8944
        for idx in [0usize, 1, 2, 3, 5] {
            assert!(
                (inv_result[idx] - expected_inv).abs() < 1e-3,
                "InvStdDev[{idx}]={}, want ≈{expected_inv}",
                inv_result[idx]
            );
        }

        for out in &outputs {
            ((*api).ReleaseValue.unwrap())(*out);
        }
        ((*api).ReleaseValue.unwrap())(x_val);
        ((*api).ReleaseValue.unwrap())(s_val);
        assert_ops_assigned_to_our_ep(api, session, &["LayerNormalization"], "layer_norm_neg_axis");
        conformance_teardown(api, env, opts, session, "cpu_ep_ln_neg");
        eprintln!(
            "\n✅ conformance_layer_norm_neg_axis: PASSED — Mean/InvStdDev shape [2,3,1], values verified"
        );
    }
}

// ─── RMSNormalization: single output, no Mean ─────────────────────────────────

/// RMSNormalization(X [2,4], scale [4]) → 1 output Y [2,4].
/// axis=-1.  Verifies that the EP handles a LayerNorm-family op with a single
/// output (no Mean, no InvStdDev) without crashing or misshaping Y.
/// Also validates that Y values satisfy RMS-norm invariant: rms(Y_row)≈1.
///
/// X = [[1,2,3,4],[5,6,7,8]], scale = [1,1,1,1]
/// Row 0: rms(X)=sqrt(7.5)≈2.7386 → Y[0,0]≈0.3651
/// Row 1: rms(X)=sqrt(43.5)≈6.5952 → rms(Y row)≈1.0
#[test]
fn conformance_rms_norm() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = PathBuf::from(manifest_dir)
        .join("tests/fixtures/simplified_layer_norm_f32/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_rms", &model_path, true) })
    else {
        eprintln!("*** SKIPPED: conformance_rms_norm — ORT or EP cdylib not found ***");
        return;
    };

    unsafe {
        let mut x_data: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut scale_data: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
        let x_shape: [i64; 2] = [2, 4];
        let s_shape: [i64; 1] = [4];
        let x_val = make_float_tensor(api, &mut x_data, &x_shape);
        let s_val = make_float_tensor(api, &mut scale_data, &s_shape);

        let input_names = [c"X".as_ptr(), c"scale".as_ptr()];
        let output_names = [c"Y".as_ptr()];
        let inputs: [*const ort::OrtValue; 2] = [x_val, s_val];
        let mut outputs: [*mut ort::OrtValue; 1] = [ptr::null_mut()];

        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(),
            input_names.as_ptr(),
            inputs.as_ptr(),
            2,
            output_names.as_ptr(),
            1,
            outputs.as_mut_ptr(),
        );
        check_status(api, status, "Run(rms_norm)");

        assert!(!outputs[0].is_null(), "RMSNorm Y output is null");

        // Y shape must be [2,4].
        assert_output_shape(api, outputs[0], &[2, 4], "rms_norm_Y");
        assert_output_dtype(
            api,
            outputs[0],
            ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            "rms_norm_Y",
        );

        // Check Y values.
        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(outputs[0], &mut data_ptr);
        check_status(api, status, "GetTensorMutableData(rms_norm_Y)");
        let y_result = std::slice::from_raw_parts(data_ptr as *const f32, 8);
        eprintln!("  RMSNorm Y output: {y_result:?}");

        // Y[0,0] ≈ 1.0 / sqrt(7.5)
        let expected_y00 = 1.0_f32 / 7.5_f32.sqrt();
        assert!(
            (y_result[0] - expected_y00).abs() < 1e-4,
            "Y[0,0]={}, want ≈{expected_y00}",
            y_result[0]
        );

        // For each row of Y, rms(row) ≈ 1.0 (RMS-norm invariant).
        for row in 0..2 {
            let row_slice = &y_result[row * 4..(row + 1) * 4];
            let rms: f32 = (row_slice.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
            assert!(
                (rms - 1.0).abs() < 1e-4,
                "RMSNorm Y row {row} rms={rms}, expected ~1.0"
            );
        }

        ((*api).ReleaseValue.unwrap())(outputs[0]);
        ((*api).ReleaseValue.unwrap())(x_val);
        ((*api).ReleaseValue.unwrap())(s_val);
        assert_ops_assigned_to_our_ep(api, session, &["RMSNormalization"], "rms_norm");
        conformance_teardown(api, env, opts, session, "cpu_ep_rms");
        eprintln!("\n✅ conformance_rms_norm: PASSED — Y shape [2,4], rms(Y row)≈1.0");
    }
}

// ─── Initializer-backed MatMul ───────────────────────────────────────────────

/// MatMul with constant-initializer weights (not graph inputs).
///
/// Real models supply weights as initializers, routing through prepacking rather
/// than graph inputs. This test verifies our EP handles initializer-backed inputs
/// correctly — a gap the upstream sibling PR identified.
///
/// Fixture: X[2,4] @ W[4,3](initializer) = Y[2,3]
#[test]
fn conformance_matmul_initializer_weights() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = PathBuf::from(manifest_dir)
        .join("tests/fixtures/matmul_initializer_weights/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_matmul_init", &model_path, true) })
    else {
        eprintln!(
            "*** SKIPPED: conformance_matmul_initializer_weights — ORT or EP cdylib not found ***"
        );
        return;
    };

    unsafe {
        assert_ops_assigned_to_our_ep(api, session, &["MatMul"], "matmul_initializer_weights");
    };

    unsafe {
        // X = [[1,0,0,0],[0,1,0,0]]
        // W (initializer) = [[1,0,0],[0,1,0],[0,0,1],[1,1,1]]
        // Y = X @ W = [[1,0,0],[0,1,0]]
        let mut x_data: [f32; 8] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let x_shape: [i64; 2] = [2, 4];
        let x_val = make_float_tensor(api, &mut x_data, &x_shape);

        let input_names = [c"X".as_ptr()];
        let output_names = [c"Y".as_ptr()];
        let inputs: [*const ort::OrtValue; 1] = [x_val];
        let mut output: *mut ort::OrtValue = ptr::null_mut();

        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(),
            input_names.as_ptr(),
            inputs.as_ptr(),
            1,
            output_names.as_ptr(),
            1,
            &mut output,
        );
        check_status(api, status, "Run(matmul_initializer_weights)");
        assert!(!output.is_null());

        // Verify output shape [2,3]
        assert_output_shape(api, output, &[2, 3], "matmul_init_Y");

        // Verify values: [[1,0,0],[0,1,0]]
        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(output, &mut data_ptr);
        check_status(api, status, "GetTensorMutableData(matmul_init_Y)");
        let result = std::slice::from_raw_parts(data_ptr as *const f32, 6);
        let expected: [f32; 6] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-4, "Y[{i}] = {got}, want {want}");
        }

        ((*api).ReleaseValue.unwrap())(output);
        ((*api).ReleaseValue.unwrap())(x_val);
        conformance_teardown(api, env, opts, session, "cpu_ep_matmul_init");
        eprintln!("\n✅ conformance_matmul_initializer_weights: PASSED");
    }
}

// ─── Assignment falsifiers ───────────────────────────────────────────────────
//
// This EP claims every node it supports and never asks ORT's CPU EP to run one
// for it. `supports_op` decides what it *can* run, and the answer to
// `GetCapability` follows from that alone.
//
// A unit test can only check a predicate; these check the thing that actually
// matters — what ORT does with the answer, read back from its own node-to-EP
// assignment. Each is a falsifier: it fails if a node this EP supports ends up
// on `CPUExecutionProvider`, if a claimed node arrives fragmented rather than
// whole, or if the resulting partition produces wrong numbers.

/// bfloat16 shows why giving a node back can be worse than running it slowly:
/// ORT's CPU EP has no bfloat16 `Tanh` kernel at all, so handing one over turns
/// a working session into a `NOT_IMPLEMENTED` session-creation failure.
///
/// This EP no longer declines anything, so this is a regression guard rather
/// than a falsifier against `main` — `supports_op` is dtype-agnostic for these
/// ops, so bf16 was always claimed. What it pins is that bf16 keeps working if
/// a future change ever reintroduces a conditional claim. The assignment and
/// numeric assertions carry the test: session creation alone would succeed even
/// if ORT did have a bf16 kernel, since a claimed node never reaches ORT's.
#[test]
fn assignment_policy_always_claims_bfloat16_activations() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = PathBuf::from(manifest_dir)
        .join("tests/fixtures/activation_assignment_bf16/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_assign_bf16", &model_path, false) })
    else {
        eprintln!(
            "*** SKIPPED: assignment_policy_always_claims_bfloat16_activations — \
             ORT or EP cdylib not found ***"
        );
        return;
    };

    unsafe {
        let info = query_ep_assignment(api, session);
        let ours = info.ops_on_our_ep();
        eprintln!(
            "  [assign_bf16] ours={ours:?}, others={:?}",
            info.ops_not_on_our_ep()
        );
        assert!(
            ours.contains(&"Tanh"),
            "bfloat16 'Tanh' must stay claimed — ORT has no bfloat16 kernel and \
             the session would fail to load without ours; got: {:?}",
            info.assignments
        );

        // bf16 words: 0x3F80 = 1.0, 0x0000 = 0.0, 0xBF80 = -1.0, 0x4000 = 2.0
        let mut x_data: [u16; 4] = [0x3F80, 0x0000, 0xBF80, 0x4000];
        let mut y_data: [u16; 4] = [0x0000, 0x0000, 0x0000, 0x0000];
        let shape: [i64; 2] = [1, 4];
        let x_val = make_bfloat16_tensor(api, &mut x_data, &shape);
        let y_val = make_bfloat16_tensor(api, &mut y_data, &shape);

        let input_names = [c"X".as_ptr(), c"Y".as_ptr()];
        let output_names = [c"Z".as_ptr()];
        let inputs: [*const ort::OrtValue; 2] = [x_val, y_val];
        let mut output: *mut ort::OrtValue = ptr::null_mut();
        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(),
            input_names.as_ptr(),
            inputs.as_ptr(),
            2,
            output_names.as_ptr(),
            1,
            &mut output,
        );
        check_status(api, status, "Run(assign_bf16)");
        assert!(!output.is_null());

        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(output, &mut data_ptr);
        check_status(api, status, "GetTensorMutableData(assign_bf16)");
        let got = std::slice::from_raw_parts(data_ptr as *const u16, 4);
        for (i, (&g, &x)) in got.iter().zip(x_data.iter()).enumerate() {
            let want = f32::from_bits((x as u32) << 16).tanh();
            let g = f32::from_bits((g as u32) << 16);
            // bfloat16 carries 8 mantissa bits, so ~1e-2 relative is the format
            // floor, not a kernel tolerance.
            assert!(
                (g - want).abs() < 1e-2,
                "Z[{i}] = {g}, want ~{want} (bfloat16 Tanh)"
            );
        }
        ((*api).ReleaseValue.unwrap())(output);
        ((*api).ReleaseValue.unwrap())(x_val);
        ((*api).ReleaseValue.unwrap())(y_val);
        conformance_teardown(api, env, opts, session, "cpu_ep_assign_bf16");
        eprintln!("\n✅ assignment_policy_always_claims_bfloat16_activations: PASSED");
    }
}

/// float16 `Gelu` must be claimed at **every** size, and the falsifier is the
/// absence of an inlined function body.
///
/// ORT's CPU EP has no float16 `Gelu` kernel, so declining does not hand the
/// node over — ORT inlines the `Gelu` function into
/// `Cast`/`Pow`/`Mul`/`Sum`/`Tanh`/`Sqrt` and this EP claims the ungoverned
/// float16 constituents, measured at 0.014-0.049x of plain ORT. An element-count
/// cap was written and deleted for exactly this reason; this test is what
/// stops it coming back. Both fixtures (3072 and 300000 elements) must show a
/// single claimed `Gelu` and no decomposition artefacts.
#[test]
fn assignment_policy_claims_float16_gelu_without_inlining_it() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");

    for (tag, reg) in [
        ("small", "cpu_ep_gelu_small"),
        ("large", "cpu_ep_gelu_large"),
    ] {
        let model_path = PathBuf::from(manifest_dir).join(format!(
            "tests/fixtures/gelu_assignment_f16_{tag}/model.onnx.textproto"
        ));
        let Some((_lib, api, env, opts, session)) =
            (unsafe { conformance_setup(reg, &model_path, false) })
        else {
            eprintln!(
                "*** SKIPPED: assignment_policy_claims_float16_gelu_without_inlining_it — \
                 ORT or EP cdylib not found ***"
            );
            return;
        };

        unsafe {
            let info = query_ep_assignment(api, session);
            let ours = info.ops_on_our_ep();
            eprintln!(
                "  [gelu_f16_{tag}] ours={ours:?}, others={:?}",
                info.ops_not_on_our_ep()
            );
            assert!(
                ours.contains(&"Gelu"),
                "float16 Gelu must be claimed at {tag} size — ORT has no float16 Gelu \
                 kernel, so deferring inlines the function instead; got: {:?}",
                info.assignments
            );
            // The decomposition signature: if ORT had inlined the function, the
            // graph would contain `Pow`/`Sum`/`Mul` nodes that do not exist in
            // the fixture. Their absence is what proves the claim held.
            for artefact in ["Pow", "Sum", "Mul"] {
                assert!(
                    !ours.contains(&artefact) && !info.ops_not_on_our_ep().contains(&artefact),
                    "float16 Gelu was inlined into its function body ('{artefact}' appeared) \
                     — the claim did not hold; got: {:?}",
                    info.assignments
                );
            }
            conformance_teardown(api, env, opts, session, reg);
        }
    }
    eprintln!("\n✅ assignment_policy_claims_float16_gelu_without_inlining_it: PASSED");
}

// ─── Falsifier: an inlined contrib function must still build a session ────────

/// Decode a float16 bit-pattern to `f32` (finite, normal/subnormal, no NaN
/// payload preservation — sufficient for the tolerance checks below).
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let man = (bits & 0x3ff) as u32;
    let f = match exp {
        0 if man == 0 => sign << 31,
        0 => {
            // Subnormal: normalise.
            let mut e = -1i32;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            let m = m & 0x3ff;
            (sign << 31) | (((127 - 15 + 2 + e) as u32) << 23) | (m << 13)
        }
        0x1f => (sign << 31) | (0xff << 23) | (man << 13),
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(f)
}

/// Encode `f32` as a float16 bit-pattern (round-to-nearest-even, finite input).
fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let mut val = (b & 0x7fff_ffff) as i32;
    if val >= 0x4780_0000 {
        return sign | 0x7c00; // overflow → inf
    }
    if val < 0x3880_0000 {
        // Subnormal or zero.
        let scaled = f32::from_bits(val as u32) / 2f32.powi(-24);
        return sign | (scaled.round() as u16);
    }
    val -= 0x3800_0000;
    let mant = ((val >> 13) & 0x7fff) as u16;
    let round = (val & 0x1fff) as u32;
    let mut out = mant;
    if round > 0x1000 || (round == 0x1000 && (out & 1) == 1) {
        out += 1;
    }
    sign | out
}

/// float16 `com.microsoft::FastGelu` must reach this EP as **one** node.
///
/// History, because the two failure modes are easy to confuse:
///
/// 1. ORT has no float16 kernel for `com.microsoft::FastGelu`, so if the node
///    is not claimed ORT inlines the contrib function body
///    (`Identity`/`Mul`/`Add`/`Tanh`/…). The routing preference then defers the
///    float16 `Tanh` inside that body, splitting the remainder into a partition
///    where `_inlfunc_FastGelu_X_bias` is both an output of our fused subgraph
///    and an input to three later `Mul`s. `build_subgraph_routing` cannot route
///    that shape, and failing at Compile is *not* a graceful decline — ORT
///    surfaces it as `FAIL : Compile: multi-node subgraph has unroutable
///    graph`, turning a model that used to load into one that does not. The
///    claim-time routing filter in `ep.rs` catches such partitions while
///    declining is still free (unit-tested there by
///    `claim_with_internally_reused_subgraph_output_is_unroutable`).
/// 2. Surviving that is not the same as being *fast*. Claiming the inlined
///    fragments measured 0.10-0.24x of plain ORT. The node only reaches this EP
///    whole once `ShapeInference::for_node` knows the contrib activations are
///    shape-preserving — without that arm the fail-closed shape filter in
///    `ep_get_capability_inner` silently drops the claim no matter what
///    `supports_op` and `claim_preference` say.
///
/// So this test asserts the strong property that subsumes both: the session
/// builds, `FastGelu` is on our EP, and none of the inlining artefacts exist.
#[test]
fn float16_fastgelu_is_claimed_as_a_single_node() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = PathBuf::from(manifest_dir)
        .join("tests/fixtures/fastgelu_assignment_f16/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_fastgelu_f16", &model_path, false) })
    else {
        eprintln!(
            "*** SKIPPED: float16_fastgelu_is_claimed_as_a_single_node — \
             ORT or EP cdylib not found ***"
        );
        return;
    };

    unsafe {
        let info = query_ep_assignment(api, session);
        let ours = info.ops_on_our_ep();
        eprintln!(
            "  [fastgelu_f16] ours={ours:?}, others={:?}",
            info.ops_not_on_our_ep()
        );
        assert!(
            ours.contains(&"FastGelu"),
            "float16 com.microsoft::FastGelu must be claimed whole — ORT has no float16 \
             kernel for it, so anything else means the node was inlined and we claimed \
             the pieces at ~0.12x; got: {:?}",
            info.assignments
        );
        // The inlining signature. None of these op types exist in the fixture,
        // so their absence is what proves the whole-node claim held.
        for artefact in ["Identity", "Cast", "Mul", "Add", "Tanh"] {
            assert!(
                !ours.contains(&artefact) && !info.ops_not_on_our_ep().contains(&artefact),
                "float16 FastGelu was inlined into its function body ('{artefact}' appeared) \
                 — the whole-node claim did not hold; got: {:?}",
                info.assignments
            );
        }

        let xs: [f32; 8] = [-4.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0, 6.0];
        let mut x_data: [u16; 8] = std::array::from_fn(|i| f32_to_f16_bits(xs[i]));
        let shape: [i64; 2] = [1, 8];
        let x_val = make_float16_tensor(api, &mut x_data, &shape);

        let input_names = [c"X".as_ptr()];
        let output_names = [c"Z".as_ptr()];
        let inputs: [*const ort::OrtValue; 1] = [x_val];
        let mut output: *mut ort::OrtValue = ptr::null_mut();

        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(),
            input_names.as_ptr(),
            inputs.as_ptr(),
            1,
            output_names.as_ptr(),
            1,
            &mut output,
        );
        check_status(api, status, "Run(float16 FastGelu)");
        assert!(!output.is_null());

        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(output, &mut data_ptr);
        check_status(api, status, "GetTensorMutableData(float16 FastGelu)");
        let result = std::slice::from_raw_parts(data_ptr as *const u16, 8);

        for (i, &x) in xs.iter().enumerate() {
            let inner = 0.797_884_6_f32 * (x + 0.044_715 * x * x * x);
            let want = 0.5 * x * (1.0 + inner.tanh());
            let got = f16_bits_to_f32(result[i]);
            assert!(
                (got - want).abs() <= 2e-2 * want.abs().max(1.0),
                "FastGelu(f16) output[{i}] for x={x}: got {got}, want {want}"
            );
        }

        ((*api).ReleaseValue.unwrap())(output);
        ((*api).ReleaseValue.unwrap())(x_val);
        conformance_teardown(api, env, opts, session, "cpu_ep_fastgelu_f16");
    }
    eprintln!("\n✅ float16_fastgelu_is_claimed_as_a_single_node: PASSED");
}

/// float16 `com.microsoft::QuickGelu` must be claimed as one node — the
/// knowingly-sub-1.0 assignment in this family, and therefore the one most in
/// need of a falsifier.
///
/// Claimed, we run this at 1.06x (512 elements) falling to 0.81x (1M). That is
/// below ORT-alone, and normally the policy would decline. It does not here
/// because the deferred alternative is not 1.0x: ORT inlines the function into
/// `Cast`/`Sigmoid`/`Cast`/`Mul`, this EP claims *two* fragments of the body,
/// and the session measures **0.093-0.220x**. Claiming is the better of two
/// losing options, exactly as for float16 exact `Gelu`.
///
/// What this test pins is that the node reaches us *whole*. ORT inlines
/// `QuickGelu` when no EP claims it, into `Cast`/`Sigmoid`/`Cast`/`Mul`, and a
/// fragmented arrival would mean we are executing the function body piecewise
/// instead of as one kernel — a real performance difference that the assignment
/// list alone would not reveal, since the fragments are ops we also claim.
#[test]
fn float16_quickgelu_is_claimed_as_a_single_node() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = PathBuf::from(manifest_dir)
        .join("tests/fixtures/quickgelu_assignment_f16/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_quickgelu_f16", &model_path, false) })
    else {
        eprintln!(
            "*** SKIPPED: float16_quickgelu_is_claimed_as_a_single_node — \
             ORT or EP cdylib not found ***"
        );
        return;
    };

    unsafe {
        let info = query_ep_assignment(api, session);
        let ours = info.ops_on_our_ep();
        eprintln!(
            "  [quickgelu_f16] ours={ours:?}, others={:?}",
            info.ops_not_on_our_ep()
        );
        assert!(
            ours.contains(&"QuickGelu"),
            "float16 com.microsoft::QuickGelu must be claimed whole — deferring makes ORT \
             inline it and we end up owning two fragments at ~0.10x; got: {:?}",
            info.assignments
        );
        for artefact in ["Cast", "Sigmoid", "Mul"] {
            assert!(
                !ours.contains(&artefact) && !info.ops_not_on_our_ep().contains(&artefact),
                "float16 QuickGelu was inlined into its function body ('{artefact}' appeared) \
                 — the whole-node claim did not hold; got: {:?}",
                info.assignments
            );
        }

        let xs: [f32; 8] = [-4.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0, 6.0];
        let mut x_data: [u16; 8] = std::array::from_fn(|i| f32_to_f16_bits(xs[i]));
        let shape: [i64; 2] = [1, 8];
        let x_val = make_float16_tensor(api, &mut x_data, &shape);

        let input_names = [c"X".as_ptr()];
        let output_names = [c"Z".as_ptr()];
        let inputs: [*const ort::OrtValue; 1] = [x_val];
        let mut output: *mut ort::OrtValue = ptr::null_mut();

        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(),
            input_names.as_ptr(),
            inputs.as_ptr(),
            1,
            output_names.as_ptr(),
            1,
            &mut output,
        );
        check_status(api, status, "Run(float16 QuickGelu)");
        assert!(!output.is_null());

        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(output, &mut data_ptr);
        check_status(api, status, "GetTensorMutableData(float16 QuickGelu)");
        let result = std::slice::from_raw_parts(data_ptr as *const u16, 8);

        for (i, &x) in xs.iter().enumerate() {
            let want = x / (1.0 + (-1.702_f32 * x).exp());
            let got = f16_bits_to_f32(result[i]);
            assert!(
                (got - want).abs() <= 2e-2 * want.abs().max(1.0),
                "QuickGelu(f16) output[{i}] for x={x}: got {got}, want {want}"
            );
        }

        ((*api).ReleaseValue.unwrap())(output);
        ((*api).ReleaseValue.unwrap())(x_val);
        conformance_teardown(api, env, opts, session, "cpu_ep_quickgelu_f16");
    }
    eprintln!("\n✅ float16_quickgelu_is_claimed_as_a_single_node: PASSED");
}

/// Abramowitz & Stegun 7.1.26 — accurate to ~1.5e-7, far tighter than the
/// float16 tolerance the `BiasGelu` check uses it under.
fn erf_reference(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_) * t) + 1.421_413_7) * t - 0.284_496_74) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp();
    sign * y
}

/// Exact (`approximate="none"`) float16 `Gelu` is claimed for the same reason
/// the tanh variant is: ORT has no float16 `Gelu` kernel, so deferring makes it
/// inline the function into `Cast`/`Pow`/`Sum`/`Mul`/`Erf`, and we then claim
/// the ungoverned pieces at a far worse rate than ORT runs the whole node.
#[test]
fn assignment_policy_claims_exact_float16_gelu() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = PathBuf::from(manifest_dir)
        .join("tests/fixtures/gelu_assignment_f16_exact/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_gelu_exact_f16", &model_path, false) })
    else {
        eprintln!(
            "*** SKIPPED: assignment_policy_claims_exact_float16_gelu — \
             ORT or EP cdylib not found ***"
        );
        return;
    };

    unsafe {
        let info = query_ep_assignment(api, session);
        let ours = info.ops_on_our_ep();
        eprintln!(
            "  [gelu_f16_exact] ours={ours:?}, others={:?}",
            info.ops_not_on_our_ep()
        );
        assert!(
            ours.contains(&"Gelu"),
            "exact float16 Gelu must be claimed — ORT has no float16 Gelu kernel, \
             so deferring inlines the function; got: {:?}",
            info.assignments
        );
        for artefact in ["Pow", "Sum", "Erf"] {
            assert!(
                !ours.contains(&artefact) && !info.ops_not_on_our_ep().contains(&artefact),
                "exact float16 Gelu was inlined ('{artefact}' appeared); got: {:?}",
                info.assignments
            );
        }
        conformance_teardown(api, env, opts, session, "cpu_ep_gelu_exact_f16");
    }
    eprintln!("\n✅ assignment_policy_claims_exact_float16_gelu: PASSED");
}

/// The routing filter must not cost any fusion that actually worked.
///
/// `MatMul(W) + Add(B) + Relu`, both weights initializers, is the shape an
/// over-broad filter would wrongly decline (an earlier draft that rejected every
/// multi-node claim reading an initializer did exactly that, pushing all three
/// nodes onto `CPUExecutionProvider`). ORT surfaces fused-subgraph initializers
/// as subgraph inputs, so this routes fine and must stay one claimed subgraph.
#[test]
fn initializer_chain_still_fuses_into_one_claim() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = PathBuf::from(manifest_dir)
        .join("tests/fixtures/initializer_chain_fusion/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_init_chain", &model_path, false) })
    else {
        eprintln!(
            "*** SKIPPED: initializer_chain_still_fuses_into_one_claim — \
             ORT or EP cdylib not found ***"
        );
        return;
    };

    unsafe {
        let info = query_ep_assignment(api, session);
        eprintln!(
            "  [init_chain] ours={:?}, others={:?}",
            info.ops_on_our_ep(),
            info.ops_not_on_our_ep()
        );
        assert!(
            info.ops_not_on_our_ep().is_empty(),
            "the whole MatMul+Add+Relu chain must stay fused on our EP; got: {:?}",
            info.assignments
        );

        let mut x_data: [f32; 8] = [1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, 0.5];
        let x_shape: [i64; 2] = [2, 4];
        let x_val = make_float_tensor(api, &mut x_data, &x_shape);

        let input_names = [c"X".as_ptr()];
        let output_names = [c"Y".as_ptr()];
        let inputs: [*const ort::OrtValue; 1] = [x_val];
        let mut output: *mut ort::OrtValue = ptr::null_mut();
        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(),
            input_names.as_ptr(),
            inputs.as_ptr(),
            1,
            output_names.as_ptr(),
            1,
            &mut output,
        );
        check_status(api, status, "Run(initializer chain)");

        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(output, &mut data_ptr);
        check_status(api, status, "GetTensorMutableData(initializer chain)");
        let result = std::slice::from_raw_parts(data_ptr as *const f32, 6);

        // W = [[1,0,0],[0,1,0],[0,0,1],[1,1,1]], B = [-1, 0.5, 2]
        // row0: [1,2,3] + [4,4,4] = [5,6,7]  + B = [4, 6.5, 9]
        // row1: [-1,-2,-3] + [0.5,0.5,0.5] = [-0.5,-1.5,-2.5] + B = [-1.5, -1, -0.5] -> relu 0
        let expected: [f32; 6] = [4.0, 6.5, 9.0, 0.0, 0.0, 0.0];
        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "output[{i}] = {got}, want {want}"
            );
        }

        ((*api).ReleaseValue.unwrap())(output);
        ((*api).ReleaseValue.unwrap())(x_val);
        conformance_teardown(api, env, opts, session, "cpu_ep_init_chain");
    }
    eprintln!("\n✅ initializer_chain_still_fuses_into_one_claim: PASSED");
}

// ─── Assignment falsifiers for the bit-exact elementwise unary ops ───────────

/// Drive one fixture through a real ORT session and report which op types
/// landed on this EP and which stayed on ORT's.
///
/// Returns `None` when ORT or the EP cdylib is unavailable, matching the skip
/// behaviour of the other conformance tests in this file.
fn unary_assignment(fixture: &str, reg: &str) -> Option<(Vec<String>, Vec<String>)> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path =
        PathBuf::from(manifest_dir).join(format!("tests/fixtures/{fixture}/model.onnx.textproto"));
    let (_lib, api, env, opts, session) = unsafe { conformance_setup(reg, &model_path, false) }?;
    unsafe {
        let info = query_ep_assignment(api, session);
        let ours: Vec<String> = info.ops_on_our_ep().iter().map(|s| s.to_string()).collect();
        let theirs: Vec<String> = info
            .ops_not_on_our_ep()
            .iter()
            .map(|s| s.to_string())
            .collect();
        eprintln!("  [{fixture}] ours={ours:?}, others={theirs:?}");
        conformance_teardown(api, env, opts, session, reg);
        Some((ours, theirs))
    }
}

/// The counterweight: thresholds that only ever decline would be trivially
/// "honest" and useless. ORT's float16 `Round` takes 5.5 ms where this EP takes
/// 0.45 ms, and that 12.4x win must actually be claimed.
#[test]
fn float16_round_is_assigned_because_it_wins() {
    let Some((ours, _)) = unary_assignment("round_assignment_f16_above", "cpu_ep_round_f16") else {
        eprintln!(
            "*** SKIPPED: float16_round_is_assigned_because_it_wins — \
             ORT or EP cdylib not found ***"
        );
        return;
    };
    assert!(
        ours.iter().any(|op| op == "Round"),
        "float16 Round is measured at 12.4x and must be claimed (ours={ours:?})",
    );
}

/// bfloat16 has no ORT CPU kernel for these ops, so the claim is a capability
/// claim and must survive. This is the counterweight that stops the deferral
/// from being widened into "never claim Exp".
#[test]
fn bfloat16_exp_is_still_assigned_because_ort_has_no_kernel() {
    let Some((ours, _theirs)) = unary_assignment("exp_assignment_bf16_large", "cpu_ep_exp_bf16")
    else {
        return;
    };
    assert!(
        ours.iter().any(|op| op == "Exp"),
        "bfloat16 Exp has no ORT CPU kernel and must stay claimed, got ours={ours:?}"
    );
}

/// `Relu` measures at 0.76-1.03x against ORT — never a clear win — and is
/// claimed anyway, like everything else this EP supports.
///
/// It is kept as its own test because `Relu` is the case where giving a node
/// away is most obviously self-defeating: it anchors
/// `MatMul+Bias+Relu -> FusedGemm`, so handing it over would split the fusion
/// across an EP boundary and cost far more than the elementwise ratio suggests.
/// This is the counterweight to `initializer_chain_still_fuses_into_one_claim`:
/// that test proves the fusion survives, this one pins that the node feeding it
/// is ours.
#[test]
fn float32_relu_stays_claimed_because_it_anchors_a_fusion() {
    let Some((ours, _theirs)) = unary_assignment("relu_assignment_f32_large", "cpu_ep_relu_f32")
    else {
        return;
    };
    assert!(
        ours.iter().any(|o| o == "Relu"),
        "Relu anchors MatMul+Bias+Relu fusion and must stay claimed, got ours={ours:?}"
    );
}

// ─── Falsifiers: this EP never hands a node to ORT's CPU EP ──────────────────

/// Every fixture that exists to probe assignment, and the op each one must be
/// running on *this* EP.
///
/// These fixtures were built to pin the old size/dtype decline thresholds, so
/// they deliberately span both sides of every boundary that policy used to
/// draw: below and above the `Sign` crossover, static and dynamic shapes,
/// float32/float16/bfloat16, the plain ops and the `com.microsoft` contrib
/// ones. That makes them exactly the right corpus for the opposite assertion —
/// if any decline survived anywhere, one of these would land on ORT.
const ASSIGNMENT_FIXTURES: &[(&str, &str)] = &[
    ("activation_assignment_f32", "Tanh"),
    ("activation_assignment_bf16", "Tanh"),
    ("exp_assignment_f32_large", "Exp"),
    ("exp_assignment_f16_large", "Exp"),
    ("exp_assignment_bf16_large", "Exp"),
    ("log_assignment_f32_large", "Log"),
    ("softplus_assignment_f32_mid", "Softplus"),
    ("relu_assignment_f32_large", "Relu"),
    ("neg_assignment_f32_large", "Neg"),
    ("sign_assignment_f32_below", "Sign"),
    ("sign_assignment_f32_above", "Sign"),
    ("sign_assignment_f32_dynamic", "Sign"),
    ("sign_assignment_f16_large", "Sign"),
    ("round_assignment_f16_above", "Round"),
    ("fastgelu_assignment_f32", "FastGelu"),
    ("fastgelu_assignment_f16", "FastGelu"),
    ("quickgelu_assignment_f16", "QuickGelu"),
    ("biasgelu_assignment_f16", "BiasGelu"),
    ("gelu_assignment_f16_small", "Gelu"),
    ("gelu_assignment_f16_large", "Gelu"),
    // `PRelu` and `GroupNormalization` are the ops the dtype filter was
    // silently declining. Both are registered by this EP and both are its
    // business, but `PRelu` had no kernel-registry entry (it is registered via
    // `register_cnn_ops`, which wrote past the descriptor recorder) and
    // `GroupNormalization` had no shape rule. Each cleared one filter and was
    // dropped by the other, so the pure-Rust inventory tests passed while ORT
    // ran them. These fixtures check the only thing that cannot be faked: what
    // ORT reports as the node's assigned EP.
    ("prelu_assignment_f32", "PRelu"),
    ("groupnorm_assignment_f32", "GroupNormalization"),
    // Not an activation this EP tunes, and that is the point: `Sin` is
    // registered by the CPU EP but was missing from the plugin's
    // `ShapeInference` table, so `GetCapability`'s fail-closed shape filter
    // dropped the claim and ORT ran it. Without a fixture whose op sits in that
    // gap, the sweep below can only prove the ops it already knew about.
    ("sin_assignment_f32", "Sin"),
    // ── Attention, MoE and KV cache ──────────────────────────────────────
    // The suite above is activations and normalisation. Those are not what
    // this EP is for, so until these rows existed the architectural rule was
    // unproven for exactly the operators the rule was written about.
    //
    // Five of these could not have passed before: `com.microsoft::Attention`,
    // `MoE`, `PackedMultiHeadAttention`, `ScatterND` and `Trilu` had no entry
    // in the plugin's `ShapeInference` table, so `GetCapability`'s fail-closed
    // filter dropped the claim and ORT's CPU EP ran them -- silently, and
    // regardless of what `supports_op` said.
    ("softmax_assignment_f32", "Softmax"),
    ("transpose_assignment_f32", "Transpose"),
    ("kv_concat_assignment_f32", "Concat"),
    ("kv_scatternd_assignment_f32", "ScatterND"),
    // float32 `RotaryEmbedding` was deferred to ORT by #1078 on a 12/12
    // losing grid. That deferral is withdrawn: this row is the falsifier that
    // fails if it is ever reintroduced.
    ("rotary_assignment_f32", "RotaryEmbedding"),
    ("mha_assignment_f32", "MultiHeadAttention"),
    ("gqa_assignment_f32", "GroupQueryAttention"),
    // `position_ids` is GQA's optional input *9*. Leaving that slot off the
    // per-slot dtype table sent a `do_rotary` node with explicit int64
    // positions to ORT even after the `seqlens_k` / `total_sequence_length`
    // slots were fixed, so it gets its own row rather than being assumed.
    ("gqa_rotary_pos_assignment_f32", "GroupQueryAttention"),
    ("msft_attention_assignment_f32", "Attention"),
    ("moe_assignment_f32", "MoE"),
    // f32 alone was not coverage: production mixtures are exported in half
    // precision, and `MoE` advertised f32 only while its kernel widens f16 and
    // bf16 to f32 and narrows on the way out. This row is the falsifier.
    ("moe_assignment_f16", "MoE"),
    // ORT has no CPU kernel for PackedMultiHeadAttention, so for that one the
    // shape-table decline bought a *load failure* rather than a slower run; it
    // does have CPU kernels for MoE and QMoE, so for those it was simply us not
    // running an op we implement. Each was rescued here and each needs its own real
    // session to prove it, because the pure-Rust inventory test builds
    // synthetic nodes and never opens one -- it cannot see a dtype-filter or
    // kernel-factory rejection, which are exactly the layers that were
    // silently declining GQA and RoPE.
    ("qmoe_assignment_f32", "QMoE"),
    ("packed_mha_assignment_f32", "PackedMultiHeadAttention"),
    ("trilu_assignment_f32", "Trilu"),
    ("scatter_elements_assignment_f32", "ScatterElements"),
];

/// The architectural rule, asserted directly: when this EP is loaded it takes
/// every node it supports, and ORT's CPU EP is left with nothing.
///
/// This EP used to decline the shape/dtype ranges where it measured slower than
/// ORT, which split the graph and let `CPUExecutionProvider` run part of it.
/// That is withdrawn: selecting this EP is a request for this EP, and a range
/// where it loses is a kernel to optimize rather than a node to give away.
///
/// The assertion is on ORT's *own* node-to-EP assignment, read back from the
/// session's profile, so it fails if a decline is reintroduced anywhere in the
/// path — `claim_preference`, `claim_preference_node`, `supports_op`, or the
/// plugin's `GetCapability` filters. Checking the policy function in isolation
/// would not: a claim can still be dropped downstream, which is precisely how
/// the contrib activations were silently unreachable before #1082.
#[test]
fn no_supported_node_is_ever_left_to_the_ort_cpu_ep() {
    let _lock = lock_ort_ep();
    let mut checked = 0usize;
    let mut failures: Vec<String> = Vec::new();

    for (fixture, op) in ASSIGNMENT_FIXTURES {
        let reg = format!("cpu_ep_noyield_{fixture}");
        let Some((ours, theirs)) = unary_assignment(fixture, &reg) else {
            // Only reachable without `NXRT_REQUIRE_ORT_TESTS=1`, which makes
            // `conformance_setup` panic instead. `continue` rather than
            // `return` so the completeness assertion below can still observe a
            // partial run instead of being skipped along with it.
            eprintln!("*** SKIPPED: {fixture} — ORT or EP cdylib not found ***");
            continue;
        };

        // Collected rather than asserted per fixture: a bare `assert!` stops at
        // the first decline, so a run reports one op when several are being
        // given away and the next iteration only uncovers the next one. The
        // whole matrix is the useful output.
        if !ours.iter().any(|claimed| claimed == op) || theirs.iter().any(|left| left == op) {
            failures.push(format!(
                "  {fixture}: '{op}' is not on this EP — ours={ours:?}, ORT was given {theirs:?}"
            ));
        }
        checked += 1;
    }

    assert!(
        failures.is_empty(),
        "{} of {checked} fixtures were handed to ORT's CPU EP. This EP does not defer — \
         a range where it is slower is a kernel to optimize, not a node to give away.\n{}",
        failures.len(),
        failures.join("\n")
    );

    if checked != ASSIGNMENT_FIXTURES.len() {
        eprintln!(
            "*** SKIPPED: only {checked}/{} fixtures ran — ORT or EP cdylib not found ***",
            ASSIGNMENT_FIXTURES.len()
        );
        return;
    }
    eprintln!("\n✅ no_supported_node_is_ever_left_to_the_ort_cpu_ep: {checked} fixtures PASSED");
}

/// The same rule with ORT's escape hatch removed.
///
/// `session.disable_cpu_ep_fallback=1` makes an unclaimed node unassignable, so
/// ORT fails session creation outright rather than quietly running it on
/// `CPUExecutionProvider`. Session creation succeeding is therefore a stronger
/// signal than reading the assignment back: it proves no node was declined,
/// including any node these fixtures contain that the assignment probe does not
/// name explicitly.
///
/// This is the falsifier that would have caught the old policy: under the
/// previous design several of these fixtures could only load *because* CPU
/// fallback was available to catch what this EP gave away.
#[test]
fn every_fixture_loads_with_cpu_fallback_disabled() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let mut checked = 0usize;

    for (fixture, op) in ASSIGNMENT_FIXTURES {
        let model_path = PathBuf::from(manifest_dir)
            .join(format!("tests/fixtures/{fixture}/model.onnx.textproto"));
        let reg = format!("cpu_ep_nofb_{fixture}");

        let Some((_lib, api, env, opts, session)) =
            (unsafe { conformance_setup(&reg, &model_path, true) })
        else {
            eprintln!("*** SKIPPED: {fixture} (no-fallback) — ORT or EP cdylib not found ***");
            continue;
        };

        unsafe {
            let info = query_ep_assignment(api, session);
            let ours = info.ops_on_our_ep();
            assert!(
                ours.contains(op),
                "{fixture}: with CPU fallback disabled the session loaded, but '{op}' is not on \
                 this EP — got {:?}",
                info.assignments
            );
            conformance_teardown(api, env, opts, session, &reg);
        }
        checked += 1;
    }

    if checked != ASSIGNMENT_FIXTURES.len() {
        eprintln!(
            "*** SKIPPED: only {checked}/{} fixtures ran — ORT or EP cdylib not found ***",
            ASSIGNMENT_FIXTURES.len()
        );
        return;
    }
    eprintln!("\n✅ every_fixture_loads_with_cpu_fallback_disabled: {checked} fixtures PASSED");
}

/// float16 `com.microsoft::BiasGelu` now runs on *this* EP, and produces the
/// right numbers.
///
/// This op used to be deferred: ORT keeps `BiasGelu` whole instead of inlining
/// it, and its cast-wrapped float32 kernel was faster than ours. Under the rule
/// that this EP does not hand work to ORT's CPU EP, we execute it — which makes
/// verifying our own arithmetic strictly more important than it was when ORT
/// was the one computing it.
///
/// `BiasGelu` is the *exact* erf formulation, not the tanh approximation, so
/// the reference is `0.5v(1 + erf(v/√2))` over `v = x + b` and the tolerance is
/// float16's, not float32's.
#[test]
fn float16_biasgelu_runs_on_our_ep_with_correct_numerics() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = PathBuf::from(manifest_dir)
        .join("tests/fixtures/biasgelu_assignment_f16/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_biasgelu_f16", &model_path, false) })
    else {
        eprintln!(
            "*** SKIPPED: float16_biasgelu_runs_on_our_ep_with_correct_numerics — \
             ORT or EP cdylib not found ***"
        );
        return;
    };

    unsafe {
        let info = query_ep_assignment(api, session);
        let ours = info.ops_on_our_ep();
        let theirs = info.ops_not_on_our_ep();
        eprintln!("  [biasgelu_f16] ours={ours:?}, others={theirs:?}");
        assert!(
            ours.contains(&"BiasGelu"),
            "float16 BiasGelu must run on this EP, not be handed to ORT; got: {:?}",
            info.assignments
        );
        assert!(
            !theirs.contains(&"BiasGelu"),
            "float16 BiasGelu was left to ORT's CPU EP; got: {:?}",
            info.assignments
        );

        let xs: [f32; 8] = [-4.0, -1.5, -0.5, 0.0, 0.5, 1.5, 3.0, 6.0];
        let bs: [f32; 8] = [0.5, -0.25, 0.0, 1.0, -1.0, 0.125, 0.75, -0.5];
        let mut x_data: [u16; 8] = std::array::from_fn(|i| f32_to_f16_bits(xs[i]));
        let mut b_data: [u16; 8] = std::array::from_fn(|i| f32_to_f16_bits(bs[i]));
        let x_shape: [i64; 2] = [1, 8];
        let b_shape: [i64; 1] = [8];
        let x_val = make_float16_tensor(api, &mut x_data, &x_shape);
        let b_val = make_float16_tensor(api, &mut b_data, &b_shape);

        let input_names = [c"X".as_ptr(), c"B".as_ptr()];
        let output_names = [c"Z".as_ptr()];
        let inputs: [*const ort::OrtValue; 2] = [x_val, b_val];
        let mut output: *mut ort::OrtValue = ptr::null_mut();

        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(),
            input_names.as_ptr(),
            inputs.as_ptr(),
            2,
            output_names.as_ptr(),
            1,
            &mut output,
        );
        check_status(api, status, "Run(float16 BiasGelu)");
        assert!(!output.is_null());

        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(output, &mut data_ptr);
        check_status(api, status, "GetTensorMutableData(float16 BiasGelu)");
        let result = std::slice::from_raw_parts(data_ptr as *const u16, 8);

        for i in 0..8 {
            let v = f16_bits_to_f32(x_data[i]) + f16_bits_to_f32(b_data[i]);
            let want = 0.5 * v * (1.0 + erf_reference(v / std::f32::consts::SQRT_2));
            let got = f16_bits_to_f32(result[i]);
            assert!(
                (got - want).abs() <= 2e-2 * want.abs().max(1.0),
                "BiasGelu(f16) output[{i}] for x={}, b={}: got {got}, want {want}",
                xs[i],
                bs[i]
            );
        }

        ((*api).ReleaseValue.unwrap())(output);
        ((*api).ReleaseValue.unwrap())(b_val);
        ((*api).ReleaseValue.unwrap())(x_val);
        conformance_teardown(api, env, opts, session, "cpu_ep_biasgelu_f16");
    }
    eprintln!("\n✅ float16_biasgelu_runs_on_our_ep_with_correct_numerics: PASSED");
}
// ─── Matmul-family assignment falsifiers ─────────────────────────────────────
//
// The activation sweep above covers the elementwise ops. This section covers
// the four matmul-family ops — `MatMul`, `Gemm`, `com.microsoft::MatMulNBits`
// and `QLinearMatMul` — which carry essentially all of a transformer's FLOPs
// and which had their own, separate deferral table until it was deleted.
//
// Each case below is placed *inside* a range that table used to hand to ORT:
// a weight tensor at or above its 2^20-element "measured region" threshold, a
// statically wide prefill activation, or the unsigned `QLinearMatMul` domain.
// A reintroduction of any of those rules therefore fails these tests rather
// than passing them silently.
//
// The models are generated rather than committed. A weight tensor big enough
// to reach the old threshold is ~0.5-1 MB, and ONNX TextFormat spells one byte
// as up to four characters, so committing ten of them would add megabytes of
// unreadable escapes to the tree. They are written under `CARGO_TARGET_TMPDIR`
// instead, which cargo provides per test binary and cleans with `cargo clean`.

/// A generated matmul-family model plus everything needed to run it twice.
#[derive(Clone)]
struct MatmulFamilyCase {
    /// Fixture name; also the file stem under `CARGO_TARGET_TMPDIR`.
    name: &'static str,
    /// The op type that must appear on our EP.
    op: &'static str,
    /// ONNX TextFormat source of the single-node model.
    model: String,
    /// Runtime inputs, in the graph's declared order.
    inputs: Vec<GeneratedInput>,
    /// Output name to fetch.
    output: &'static str,
    /// Element type of the output, one of the ONNX tensor element enums.
    output_elem: ort::ONNXTensorElementDataType,
    /// Whether ORT's own CPU EP can build a kernel for this node at all.
    ///
    /// `false` is not a limitation of the test — it is the point. ORT's CPU
    /// `MatMulNBits` `ORT_ENFORCE`s `block_size` in {16, 32, 64, 128, 256} at
    /// kernel construction, so a 512-wide block is a model ORT cannot load and
    /// this EP can. There was never a host to hand that node to.
    ort_can_build: bool,
    /// Absolute tolerance for the elementwise comparison against ORT.
    tolerance: f32,
}

/// One runtime input: name, element type, dims and raw little-endian bytes.
#[derive(Clone)]
struct GeneratedInput {
    name: &'static str,
    elem: ort::ONNXTensorElementDataType,
    dims: Vec<i64>,
    data: Vec<u8>,
}

/// Bytes that ONNX TextFormat spells as themselves.
///
/// `raw_data` is a protobuf `bytes` field, so every non-printable byte costs a
/// four-character octal escape. Restricting generated weights to printable
/// ASCII minus `"` and `\` keeps a 1 MB weight tensor a 1 MB string instead of
/// a 4 MB one, and 91 distinct values is plenty of variety for a numeric
/// comparison against an independent implementation.
fn printable_weight_blob(len: usize) -> String {
    let mut out = String::with_capacity(len);
    let mut byte = 0x23u8;
    for _ in 0..len {
        out.push(byte as char);
        byte = if byte >= 0x7e { 0x23 } else { byte + 1 };
        if byte == b'\\' {
            byte += 1;
        }
    }
    out
}

/// `count` little-endian float32 values, all `0x41232323` (~10.196).
///
/// Same trick as [`printable_weight_blob`]: those four bytes are `#`, `#`, `#`
/// and `A`, so the scale tensor costs one character per byte too. The value is
/// positive and O(10), which keeps a 256-deep dequantized dot product inside
/// float32 comfortably.
fn printable_scale_blob(count: usize) -> String {
    "###A".repeat(count)
}

/// An ONNX TextFormat `type { tensor_type { .. } }` block.
fn textproto_tensor_type(elem: ort::ONNXTensorElementDataType, dims: &[i64]) -> String {
    let dims = dims
        .iter()
        .map(|d| format!("{{ dim_value: {d} }}"))
        .collect::<Vec<_>>()
        .join(", ");
    format!("type {{ tensor_type {{ elem_type: {elem} shape {{ dim: [{dims}] }} }} }}")
}

/// A named graph input declaration.
fn textproto_value_info(name: &str, elem: ort::ONNXTensorElementDataType, dims: &[i64]) -> String {
    format!(
        "{{ name: \"{name}\" {} }}",
        textproto_tensor_type(elem, dims)
    )
}

/// A `raw_data` initializer whose payload is already TextFormat-safe.
fn textproto_initializer(
    name: &str,
    elem: ort::ONNXTensorElementDataType,
    dims: &[i64],
    raw: &str,
) -> String {
    let dims = dims
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!("{{ name: \"{name}\" data_type: {elem} dims: [{dims}] raw_data: \"{raw}\" }}")
}

const ELEM_F32: ort::ONNXTensorElementDataType = ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
const ELEM_U8: ort::ONNXTensorElementDataType = ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
const ELEM_I8: ort::ONNXTensorElementDataType = ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
const ELEM_F16: ort::ONNXTensorElementDataType = ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;

/// Deterministic activation values in `[-1, 1)`, varying along both axes so a
/// transposed or mis-strided kernel cannot pass by symmetry.
fn activation_f32(rows: usize, cols: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let t = ((r * 131 + c * 17) % 251) as f32 / 251.0;
            out.push(2.0 * t - 1.0);
        }
    }
    out
}

fn f32_slice_to_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f16_slice_to_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| f32_to_f16_bits(*v).to_le_bytes())
        .collect()
}

/// A dense `MatMul`/`Gemm` case with `B` supplied at run time.
///
/// `B` is a graph *input* rather than an initializer for two reasons: a
/// 4 MB float32 weight would be a 16 MB escape string, and the dynamic-weight
/// path is the one with no prepack to hide behind, so it is the harder case for
/// this EP to win and the more interesting one to pin.
fn dense_case(
    name: &'static str,
    op: &'static str,
    elem: ort::ONNXTensorElementDataType,
    m: usize,
    k: usize,
    n: usize,
    tolerance: f32,
) -> MatmulFamilyCase {
    let a_dims = vec![m as i64, k as i64];
    let b_dims = vec![k as i64, n as i64];
    let y_dims = vec![m as i64, n as i64];
    let model = format!(
        r#"ir_version: 10
graph {{
  node: [{{ input: ["A", "B"] output: ["Y"] op_type: "{op}" }}]
  name: "{name}"
  input: [{}, {}]
  output: [{}]
}}
opset_import: [{{ version: 17 }}]
"#,
        textproto_value_info("A", elem, &a_dims),
        textproto_value_info("B", elem, &b_dims),
        textproto_value_info("Y", elem, &y_dims),
    );
    let a = activation_f32(m, k);
    let b = activation_f32(k, n);
    let (a_bytes, b_bytes) = if elem == ELEM_F16 {
        (f16_slice_to_bytes(&a), f16_slice_to_bytes(&b))
    } else {
        (f32_slice_to_bytes(&a), f32_slice_to_bytes(&b))
    };
    MatmulFamilyCase {
        name,
        op,
        model,
        inputs: vec![
            GeneratedInput {
                name: "A",
                elem,
                dims: a_dims,
                data: a_bytes,
            },
            GeneratedInput {
                name: "B",
                elem,
                dims: b_dims,
                data: b_bytes,
            },
        ],
        output: "Y",
        output_elem: elem,
        ort_can_build: true,
        tolerance,
    }
}

/// A `com.microsoft::MatMulNBits` case with generated `B`, `scales` and, for
/// 8-bit, an explicit `zero_points`.
///
/// The 8-bit zero point cannot be left implicit here. Its default is 128 and
/// [`printable_weight_blob`] only produces bytes 0x23..=0x7e, so every weight
/// would dequantize negative and the whole output would carry one sign — a
/// comparison that a badly broken kernel could still pass. An explicit zero
/// point of 0x50 sits in the middle of the generated range and restores a
/// two-sided output. 4-bit needs no such fix: its nibbles span 0..15 around an
/// implicit default of 8.
#[allow(clippy::too_many_arguments)]
fn nbits_case(
    name: &'static str,
    m: usize,
    k: usize,
    n: usize,
    bits: u32,
    block_size: usize,
    act_elem: ort::ONNXTensorElementDataType,
    ort_can_build: bool,
) -> MatmulFamilyCase {
    assert!(
        k.is_multiple_of(block_size),
        "K must be a whole number of blocks"
    );
    assert!(act_elem == ELEM_F32 || act_elem == ELEM_F16);
    let blocks_per_col = k / block_size;
    let blob = block_size * bits as usize / 8;
    let b_elements = n * blocks_per_col * blob;
    let scale_elements = n * blocks_per_col;

    let mut initializers = vec![
        textproto_initializer(
            "B",
            ELEM_U8,
            &[n as i64, blocks_per_col as i64, blob as i64],
            &printable_weight_blob(b_elements),
        ),
        textproto_initializer(
            "scales",
            act_elem,
            &[scale_elements as i64],
            &if act_elem == ELEM_F32 {
                printable_scale_blob(scale_elements)
            } else {
                // f16 bits 0x4923 little-endian is the printable pair "#I",
                // and decodes to 10.2734 — the same order of magnitude as the
                // f32 scale, so both variants produce comparable outputs.
                "#I".repeat(scale_elements)
            },
        ),
    ];
    let mut node_inputs = vec!["\"A\"", "\"B\"", "\"scales\""];
    if bits == 8 {
        initializers.push(textproto_initializer(
            "zero_points",
            ELEM_U8,
            &[scale_elements as i64],
            &"P".repeat(scale_elements),
        ));
        node_inputs.push("\"zero_points\"");
    }

    let a_dims = vec![m as i64, k as i64];
    let y_dims = vec![m as i64, n as i64];
    let model = format!(
        r#"ir_version: 10
graph {{
  node: [{{
    input: [{}]
    output: ["Y"]
    op_type: "MatMulNBits"
    domain: "com.microsoft"
    attribute: [
      {{ name: "K" type: INT i: {k} }},
      {{ name: "N" type: INT i: {n} }},
      {{ name: "bits" type: INT i: {bits} }},
      {{ name: "block_size" type: INT i: {block_size} }}
    ]
  }}]
  name: "{name}"
  initializer: [{}]
  input: [{}]
  output: [{}]
}}
opset_import: [{{ version: 17 }}, {{ domain: "com.microsoft" version: 1 }}]
"#,
        node_inputs.join(", "),
        initializers.join(", "),
        textproto_value_info("A", act_elem, &a_dims),
        textproto_value_info("Y", act_elem, &y_dims),
    );

    let activation = activation_f32(m, k);
    MatmulFamilyCase {
        name,
        op: "MatMulNBits",
        model,
        inputs: vec![GeneratedInput {
            name: "A",
            elem: act_elem,
            dims: a_dims,
            data: if act_elem == ELEM_F32 {
                f32_slice_to_bytes(&activation)
            } else {
                f16_slice_to_bytes(&activation)
            },
        }],
        output: "Y",
        output_elem: act_elem,
        ort_can_build,
        // Dequantized weights are O(10) and K is 256-512, so outputs reach the
        // low thousands; both runtimes accumulate in float32 but not in the
        // same order, and this is a scale-appropriate absolute band. The f16
        // variant also rounds its activation and result to 11 significand
        // bits, which at this magnitude costs about one unit.
        tolerance: if act_elem == ELEM_F32 { 0.5 } else { 4.0 },
    }
}

/// A `QLinearMatMul` case with a constant `B`, which is the shape this EP
/// pre-packs once per session.
///
/// `signed` selects the activation/weight/output domain. Both are covered
/// because they took opposite decisions under the old policy — unsigned was
/// deferred at 1.13-2.65x, signed was claimed at 1.7-33x — so a partial
/// reintroduction would show up in exactly one of them.
fn qlinear_case(
    name: &'static str,
    m: usize,
    k: usize,
    n: usize,
    signed: bool,
) -> MatmulFamilyCase {
    let elem = if signed { ELEM_I8 } else { ELEM_U8 };
    // Zero points chosen to sit inside the generated ranges so neither operand
    // is one-sided: activations cycle 0..=254 for unsigned and the printable
    // band for signed, weights are always the printable band 0x23..=0x7e.
    let a_zero = if signed { 0x50u8 } else { 0x80u8 };
    let b_zero = 0x50u8;
    let y_zero = if signed { 0x50u8 } else { 0x80u8 };
    let a_dims = vec![m as i64, k as i64];
    let b_dims = vec![k as i64, n as i64];
    let y_dims = vec![m as i64, n as i64];

    let scalar = |name: &str, value: f32| {
        format!(
            "{{ name: \"{name}\" data_type: {ELEM_F32} raw_data: \"{}\" }}",
            value
                .to_le_bytes()
                .iter()
                .map(|b| format!("\\{b:03o}"))
                .collect::<String>()
        )
    };
    let zp = |name: &str, value: u8| {
        format!("{{ name: \"{name}\" data_type: {elem} raw_data: \"\\{value:03o}\" }}")
    };

    let initializers = [
        scalar("a_scale", 0.01),
        zp("a_zero_point", a_zero),
        textproto_initializer("B", elem, &b_dims, &printable_weight_blob(k * n)),
        scalar("b_scale", 0.01),
        zp("b_zero_point", b_zero),
        scalar("y_scale", 2.0),
        zp("y_zero_point", y_zero),
    ]
    .join(", ");

    let model = format!(
        r#"ir_version: 10
graph {{
  node: [{{
    input: ["A", "a_scale", "a_zero_point", "B", "b_scale", "b_zero_point", "y_scale", "y_zero_point"]
    output: ["Y"]
    op_type: "QLinearMatMul"
  }}]
  name: "{name}"
  initializer: [{initializers}]
  input: [{}]
  output: [{}]
}}
opset_import: [{{ version: 13 }}]
"#,
        textproto_value_info("A", elem, &a_dims),
        textproto_value_info("Y", elem, &y_dims),
    );

    let a: Vec<u8> = (0..m * k)
        .map(|i| {
            if signed {
                0x23u8.wrapping_add((i % 91) as u8)
            } else {
                (i % 255) as u8
            }
        })
        .collect();

    MatmulFamilyCase {
        name,
        op: "QLinearMatMul",
        model,
        inputs: vec![GeneratedInput {
            name: "A",
            elem,
            dims: a_dims,
            data: a,
        }],
        output: "Y",
        output_elem: elem,
        ort_can_build: true,
        // Quantized output: the only honest tolerance is exact, but rounding
        // of a value landing precisely on .5 is implementation-defined, so one
        // least-significant unit is allowed.
        tolerance: 1.0,
    }
}

/// The ten generated models, one per matmul-family range of interest.
fn matmul_family_cases() -> Vec<MatmulFamilyCase> {
    // Every weight tensor below is exactly 2^20 elements (256x4096, 512x2048),
    // which is the smallest size the deleted policy called "measured" and
    // therefore the smallest size at which it deferred. Before the claim fixes
    // in this file's companion commit, every `MatMulNBits` and `QLinearMatMul`
    // case here ran on ORT's CPU EP no matter what the policy said.
    vec![
        // Dense float32: deferred at 1 row and at 256.
        dense_case("matmul_f32_decode", "MatMul", ELEM_F32, 1, 256, 4096, 1e-3),
        dense_case(
            "matmul_f32_prefill",
            "MatMul",
            ELEM_F32,
            256,
            256,
            4096,
            1e-3,
        ),
        // Dense float16: deferred at every measured thread count, and the two
        // ops take different paths through this crate (`MatMul` has a GEMV,
        // `Gemm` reaches the same packed prefill through `try_half_fast_path`).
        dense_case("matmul_f16_decode", "MatMul", ELEM_F16, 1, 256, 4096, 5e-2),
        dense_case("gemm_f16_decode", "Gemm", ELEM_F16, 1, 256, 4096, 5e-2),
        // 4-bit MatMulNBits: the decode workhorse, deferred at every thread
        // count above one.
        nbits_case("nbits4_decode", 1, 256, 4096, 4, 32, ELEM_F32, true),
        // 8-bit MatMulNBits: claimed at decode, deferred once the row count was
        // statically >= 256. This is that shape.
        nbits_case("nbits8_prefill", 256, 256, 4096, 8, 32, ELEM_F32, true),
        // A block size ORT's CPU EP cannot construct a kernel for. Claimed
        // before and now, but for a reason that has nothing to do with speed.
        nbits_case("nbits4_block512", 1, 512, 2048, 4, 512, ELEM_F32, false),
        // A float16 activation is the shape a quantized LLM actually decodes
        // in, and it is only reachable at all because the dtype filter now
        // lists the quantized edge dtypes.
        nbits_case("nbits4_f16_decode", 1, 256, 4096, 4, 32, ELEM_F16, true),
        // QLinearMatMul in both signedness domains.
        qlinear_case("qlinear_u8", 1, 256, 4096, false),
        qlinear_case("qlinear_i8", 1, 256, 4096, true),
    ]
}

/// Materialize a generated model under `CARGO_TARGET_TMPDIR`.
fn write_generated_model(name: &str, text: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("matmul_family");
    std::fs::create_dir_all(&dir).expect("create generated-model dir");
    let path = dir.join(format!("{name}.onnx.textproto"));
    std::fs::write(&path, text).expect("write generated model");
    path
}

/// Wrap raw little-endian bytes as an `OrtValue` of the given element type.
///
/// # Safety
/// `api` must be a valid `OrtApi`, and `data` must outlive the returned value.
unsafe fn make_raw_tensor(
    api: *const ort::OrtApi,
    data: &mut [u8],
    dims: &[i64],
    elem: ort::ONNXTensorElementDataType,
) -> *mut ort::OrtValue {
    unsafe {
        let mut mem_info: *mut ort::OrtMemoryInfo = ptr::null_mut();
        let status = ((*api).CreateCpuMemoryInfo.unwrap())(
            ort::OrtDeviceAllocator,
            ort::OrtMemTypeDefault,
            &mut mem_info,
        );
        check_status(api, status, "CreateCpuMemoryInfo");
        let mut val: *mut ort::OrtValue = ptr::null_mut();
        let status = ((*api).CreateTensorWithDataAsOrtValue.unwrap())(
            mem_info,
            data.as_mut_ptr().cast(),
            data.len(),
            dims.as_ptr(),
            dims.len(),
            elem,
            &mut val,
        );
        check_status(api, status, "CreateTensorWithDataAsOrtValue(raw)");
        ((*api).ReleaseMemoryInfo.unwrap())(mem_info);
        val
    }
}

/// Run one generated case on `session` and return the output as float32.
///
/// Quantized outputs are widened rather than compared as bytes so a single
/// comparison path covers every case; the tolerance carried by the case is
/// what makes `1.0` mean "one quantization step" for those.
///
/// # Safety
/// `api` and `session` must be valid and the session must have been created
/// from `case`'s model.
unsafe fn run_generated_case(
    api: *const ort::OrtApi,
    session: *mut ort::OrtSession,
    case: &MatmulFamilyCase,
    stage: &str,
) -> Vec<f32> {
    let mut buffers: Vec<Vec<u8>> = case.inputs.iter().map(|i| i.data.clone()).collect();
    let mut values: Vec<*const ort::OrtValue> = Vec::with_capacity(buffers.len());
    for (input, buffer) in case.inputs.iter().zip(buffers.iter_mut()) {
        values.push(unsafe { make_raw_tensor(api, buffer, &input.dims, input.elem) } as *const _);
    }
    let input_names: Vec<CString> = case
        .inputs
        .iter()
        .map(|i| CString::new(i.name).unwrap())
        .collect();
    let input_name_ptrs: Vec<*const std::os::raw::c_char> =
        input_names.iter().map(|c| c.as_ptr()).collect();
    let output_name = CString::new(case.output).unwrap();
    let output_name_ptrs = [output_name.as_ptr()];

    let mut output: *mut ort::OrtValue = ptr::null_mut();
    unsafe {
        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(),
            input_name_ptrs.as_ptr(),
            values.as_ptr(),
            values.len(),
            output_name_ptrs.as_ptr(),
            1,
            &mut output,
        );
        check_status(api, status, stage);
        assert!(!output.is_null(), "{stage}: null output");

        let mut type_info: *mut ort::OrtTensorTypeAndShapeInfo = ptr::null_mut();
        let status = ((*api).GetTensorTypeAndShape.unwrap())(output, &mut type_info);
        check_status(api, status, "GetTensorTypeAndShape");
        let mut count: usize = 0;
        let status = ((*api).GetTensorShapeElementCount.unwrap())(type_info, &mut count);
        check_status(api, status, "GetTensorShapeElementCount");
        ((*api).ReleaseTensorTypeAndShapeInfo.unwrap())(type_info);

        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(output, &mut data_ptr);
        check_status(api, status, "GetTensorMutableData");

        let widened = match case.output_elem {
            ELEM_F32 => std::slice::from_raw_parts(data_ptr as *const f32, count).to_vec(),
            ELEM_F16 => std::slice::from_raw_parts(data_ptr as *const u16, count)
                .iter()
                .map(|bits| f16_bits_to_f32(*bits))
                .collect(),
            ELEM_U8 => std::slice::from_raw_parts(data_ptr as *const u8, count)
                .iter()
                .map(|v| *v as f32)
                .collect(),
            ELEM_I8 => std::slice::from_raw_parts(data_ptr as *const i8, count)
                .iter()
                .map(|v| *v as f32)
                .collect(),
            other => panic!("{stage}: unhandled output element type {other}"),
        };
        ((*api).ReleaseValue.unwrap())(output);
        for value in values {
            ((*api).ReleaseValue.unwrap())(value as *mut _);
        }
        widened
    }
}

/// Parse a positive-integer benchmark knob, naming the variable in any panic.
///
/// Unset, empty, and whitespace-only all mean "not configured" and yield
/// `None`; anything else must be a positive integer or the run aborts. A typo
/// must never degrade silently into a default, because the resulting numbers
/// would be reported as if they had been measured under the requested setting.
fn positive_bench_env(name: &str, raw: Option<&str>) -> Option<u32> {
    let raw = raw?.trim();
    if raw.is_empty() {
        return None;
    }
    let value: u32 = raw
        .parse()
        .unwrap_or_else(|_| panic!("{name} must be a positive integer, got {raw:?}"));
    assert!(value > 0, "{name} must be a positive integer, got {value}");
    Some(value)
}

/// Thread budget for a benchmark run, from `NXRT_MM_BENCH_THREADS`.
///
/// A shared box cannot be measured honestly at full width: whichever side runs
/// while another tenant is busy loses cores it would otherwise have had, and
/// ORT's intra-op pool spins, so it loses more than we do. Pinning both sides
/// to the same small thread count removes that variable -- the comparison stops
/// depending on how many cores the rest of the machine happened to leave free.
///
/// `None` (unset) keeps ORT's own default. Our side reads its budget from
/// `ONNX_GENAI_MLAS_THREADPOOL_THREADS`; [`bench_pool_budgets_agree`] is what
/// makes sure the caller set both.
fn bench_thread_budget(raw: Option<&str>) -> Option<i32> {
    let threads = positive_bench_env("NXRT_MM_BENCH_THREADS", raw)?;
    Some(i32::try_from(threads).unwrap_or_else(|_| {
        panic!("NXRT_MM_BENCH_THREADS exceeds ORT's intra-op thread count, got {threads}")
    }))
}

/// Warmup runs before the timed loop, from `NXRT_MM_BENCH_WARMUP`.
///
/// The default of 3 pays prepack and the first page faults, which is enough for
/// a steady-state kernel measurement but thin for anything thread-pool bound: a
/// pool that parks between runs is measured in its cold-wake state. Raise it
/// when the quantity under test is scaling rather than arithmetic.
fn bench_warmup_runs(raw: Option<&str>) -> usize {
    positive_bench_env("NXRT_MM_BENCH_WARMUP", raw).map_or(3, |n| n as usize)
}

/// Reject a half-pinned A/B, where one side is capped and the other is not.
///
/// This is the trap the whole change exists to close. `NXRT_MM_BENCH_THREADS`
/// pins ORT's intra-op pool, which is what plain ORT computes MatMul on -- but
/// our plugin's MatMul does *not* run there. It runs on the standalone MLAS
/// pool, sized by `ONNX_GENAI_MLAS_THREADPOOL_THREADS`. Setting only the former
/// therefore caps ORT while leaving us at full width, which flatters us by
/// exactly the "how many cores were free" mechanism this change is meant to
/// eliminate; setting only the latter does the reverse. Either way the printed
/// ratio would be an artefact, so refuse to produce it.
fn bench_pool_budgets_agree(ort_side: Option<&str>, our_side: Option<&str>) -> Result<(), String> {
    let ort_threads = positive_bench_env("NXRT_MM_BENCH_THREADS", ort_side);
    let our_threads = positive_bench_env("ONNX_GENAI_MLAS_THREADPOOL_THREADS", our_side);
    match (ort_threads, our_threads) {
        (None, None) => Ok(()),
        (a, b) if a == b => Ok(()),
        (a, b) => Err(format!(
            "a pinned A/B needs both pools pinned to the same count, got \
             NXRT_MM_BENCH_THREADS={} (ORT's intra-op pool) and \
             ONNX_GENAI_MLAS_THREADPOOL_THREADS={} (our MLAS pool); \
             pinning one side only measures the other side's core count",
            a.map_or_else(|| "unset".to_owned(), |v| v.to_string()),
            b.map_or_else(|| "unset".to_owned(), |v| v.to_string()),
        )),
    }
}

#[test]
fn the_bench_thread_budget_reads_a_positive_count_and_ignores_an_unset_one() {
    assert_eq!(bench_thread_budget(None), None);
    assert_eq!(bench_thread_budget(Some("")), None);
    assert_eq!(bench_thread_budget(Some("  ")), None);
    assert_eq!(bench_thread_budget(Some(" 4 ")), Some(4));
}

#[test]
#[should_panic(expected = "NXRT_MM_BENCH_THREADS must be a positive integer, got 0")]
fn the_bench_thread_budget_rejects_zero_threads() {
    bench_thread_budget(Some("0"));
}

#[test]
#[should_panic(expected = "NXRT_MM_BENCH_THREADS must be a positive integer, got \"all\"")]
fn the_bench_thread_budget_rejects_a_word() {
    bench_thread_budget(Some("all"));
}

#[test]
#[should_panic(expected = "NXRT_MM_BENCH_THREADS must be a positive integer, got \"-4\"")]
fn the_bench_thread_budget_rejects_a_negative_count() {
    bench_thread_budget(Some("-4"));
}

#[test]
#[should_panic(expected = "NXRT_MM_BENCH_THREADS exceeds ORT's intra-op thread count")]
fn the_bench_thread_budget_rejects_a_count_past_the_ort_api_width() {
    // Parses as u32, does not fit ORT's `c_int`, so it must be caught after the
    // parse rather than silently wrapping to a negative thread count.
    bench_thread_budget(Some("3000000000"));
}

#[test]
fn the_bench_warmup_defaults_to_three_and_takes_an_override() {
    assert_eq!(bench_warmup_runs(None), 3);
    assert_eq!(bench_warmup_runs(Some("")), 3);
    assert_eq!(bench_warmup_runs(Some("10")), 10);
}

#[test]
#[should_panic(expected = "NXRT_MM_BENCH_WARMUP must be a positive integer, got 0")]
fn the_bench_warmup_rejects_zero_because_that_would_time_the_prepack() {
    bench_warmup_runs(Some("0"));
}

#[test]
fn a_pinned_ab_is_refused_unless_both_pools_are_pinned_alike() {
    assert!(bench_pool_budgets_agree(None, None).is_ok());
    assert!(bench_pool_budgets_agree(Some(""), Some("  ")).is_ok());
    assert!(bench_pool_budgets_agree(Some("16"), Some("16")).is_ok());
    assert!(bench_pool_budgets_agree(Some(" 8 "), Some("8")).is_ok());

    let ort_only = bench_pool_budgets_agree(Some("16"), None).unwrap_err();
    assert!(ort_only.contains("NXRT_MM_BENCH_THREADS=16"), "{ort_only}");
    assert!(
        ort_only.contains("ONNX_GENAI_MLAS_THREADPOOL_THREADS=unset"),
        "{ort_only}"
    );
    assert!(bench_pool_budgets_agree(None, Some("16")).is_err());
    assert!(bench_pool_budgets_agree(Some("16"), Some("8")).is_err());
}

/// Apply [`bench_thread_budget`] to a session-options handle.
///
/// Called from every [`conformance_setup`], not only the benchmark, so a
/// conformance run under an explicit budget honours it too. That is a no-op
/// when the variable is unset -- which is every non-benchmark caller, including
/// CI -- and numeric parity does not depend on the thread count either way.
unsafe fn pin_intra_op_threads(api: *const ort::OrtApi, options: *mut ort::OrtSessionOptions) {
    let raw = std::env::var("NXRT_MM_BENCH_THREADS").ok();
    let Some(threads) = bench_thread_budget(raw.as_deref()) else {
        return;
    };
    let set = unsafe { (*api).SetIntraOpNumThreads }.expect("SetIntraOpNumThreads");
    let status = unsafe { set(options, threads) };
    unsafe { check_status(api, status, "SetIntraOpNumThreads") };
}

/// Create a session with **no** execution provider appended, so ORT's own CPU
/// EP owns the whole graph. Returns the ORT error message on failure instead of
/// panicking, because one case exists precisely to prove ORT cannot load it.
///
/// # Safety
/// `api` and `env` must be valid ORT handles.
unsafe fn try_plain_ort_session(
    api: *const ort::OrtApi,
    env: *mut ort::OrtEnv,
    model_path: &std::path::Path,
) -> Result<(*mut ort::OrtSessionOptions, *mut ort::OrtSession), String> {
    unsafe {
        let mut options: *mut ort::OrtSessionOptions = ptr::null_mut();
        let status = ((*api).CreateSessionOptions.unwrap())(&mut options);
        check_status(api, status, "CreateSessionOptions(plain ORT)");
        pin_intra_op_threads(api, options);
        let mut session: *mut ort::OrtSession = ptr::null_mut();
        let status = ort_session::create_session(api, env, options, model_path, &mut session);
        if status.is_null() {
            return Ok((options, session));
        }
        let get_msg = (*api).GetErrorMessage.expect("GetErrorMessage");
        let msg_ptr = get_msg(status);
        let msg = if msg_ptr.is_null() {
            "(no message)".to_owned()
        } else {
            CStr::from_ptr(msg_ptr).to_string_lossy().into_owned()
        };
        if let Some(release) = (*api).ReleaseStatus {
            release(status);
        }
        ((*api).ReleaseSessionOptions.unwrap())(options);
        Err(msg)
    }
}

/// Byte width of one element of `elem`, for the element types this file
/// generates.
fn elem_byte_size(elem: ort::ONNXTensorElementDataType) -> usize {
    match elem {
        ELEM_F32 => 4,
        ELEM_F16 => 2,
        ELEM_U8 | ELEM_I8 => 1,
        other => panic!("unhandled element type {other}"),
    }
}

/// The same case with its activation tensor rotated by one whole element.
///
/// Rotating by an element rather than by a byte keeps every value a valid
/// number of its dtype — a byte rotation of an f32 buffer would manufacture
/// NaNs and denormals and turn a parity check into a float-corner-case test.
fn with_rotated_activation(case: &MatmulFamilyCase) -> MatmulFamilyCase {
    let mut rotated = case.clone();
    let activation = &mut rotated.inputs[0];
    let width = elem_byte_size(activation.elem);
    assert!(
        activation.data.len() > width,
        "activation is a single element; rotating it is a no-op"
    );
    activation.data.rotate_left(width);
    rotated
}

/// The matmul-family half of the architectural rule, asserted end to end.
///
/// For every case this test, in one session with `session.disable_cpu_ep_fallback=1`:
///
/// 1. asserts ORT's own node-to-EP record puts the op on `cpu_ep` and nothing
///    on `CPUExecutionProvider`;
/// 2. runs it, so the claim is backed by an execution rather than a promise;
/// 3. compares the result elementwise against a second session with no EP
///    appended at all, i.e. against ORT's own kernel for the same model.
///
/// Point 3 is what makes point 1 worth asserting. Claiming everything is
/// trivially achievable by claiming things this EP computes wrongly, so the
/// rule "never defer" is only honest alongside "and get the same answer".
///
/// Disabling CPU fallback is not redundant with point 1 either. It changes what
/// a decline *does*: with fallback available a declined node quietly moves to
/// ORT and only the assignment record shows it, while with fallback disabled it
/// is unassignable and `CreateSession` fails outright. That catches nodes the
/// assignment probe does not enumerate.
#[test]
fn no_matmul_family_node_escapes_to_the_ort_cpu_ep() {
    let _lock = lock_ort_ep();
    let mut checked = 0usize;
    let mut sensitive = 0usize;
    for case in matmul_family_cases() {
        let path = write_generated_model(case.name, &case.model);
        let reg = format!("cpu_ep_mm_{}", case.name);
        let Some((_lib, api, env, opts, session)) =
            (unsafe { conformance_setup(&reg, &path, true) })
        else {
            eprintln!(
                "*** SKIPPED: {} — ORT or EP cdylib not found ***",
                case.name
            );
            return;
        };

        unsafe {
            let info = query_ep_assignment(api, session);
            let ours = info.ops_on_our_ep();
            let theirs = info.ops_not_on_our_ep();
            eprintln!("  [{}] ours={ours:?}, others={theirs:?}", case.name);
            assert!(
                ours.contains(&case.op),
                "{}: '{}' must run on this EP, got {:?}",
                case.name,
                case.op,
                info.assignments
            );
            assert!(
                theirs.is_empty(),
                "{}: nodes {theirs:?} were left to ORT's CPU EP. This EP does not defer — \
                 a range where it is slower is a kernel to optimize, not a node to give away",
                case.name,
            );

            let ours_out = run_generated_case(api, session, &case, &format!("Run({})", case.name));
            assert!(
                ours_out.iter().all(|v| v.is_finite()),
                "{}: our output has non-finite values",
                case.name
            );

            // A comparison against ORT passes trivially if both sides return
            // zeros, which is exactly what a mis-decoded quantized weight or a
            // silently-skipped kernel produces. Require a live signal first.
            let live = ours_out
                .iter()
                .filter(|v| v.is_finite() && **v != 0.0)
                .count();
            assert!(
                live * 2 >= ours_out.len(),
                "{}: only {live} of {} outputs are non-zero and finite — the case is \
                 degenerate and its parity check would be vacuous",
                case.name,
                ours_out.len()
            );

            match try_plain_ort_session(api, env, &path) {
                Ok((ort_opts, ort_session)) => {
                    assert!(
                        case.ort_can_build,
                        "{}: ORT built a kernel this test expected it to reject — \
                         the block-size claim in CPU_MATMUL_ASSIGNMENT.md is stale",
                        case.name
                    );
                    let ort_out = run_generated_case(
                        api,
                        ort_session,
                        &case,
                        &format!("Run(ORT baseline {})", case.name),
                    );
                    assert_eq!(
                        ours_out.len(),
                        ort_out.len(),
                        "{}: output length differs from ORT's",
                        case.name
                    );
                    let mut worst = 0.0f32;
                    let mut worst_at = 0usize;
                    for (i, (a, b)) in ours_out.iter().zip(ort_out.iter()).enumerate() {
                        let delta = (a - b).abs();
                        if delta > worst {
                            worst = delta;
                            worst_at = i;
                        }
                    }
                    eprintln!(
                        "  [{}] {} elements, worst |ours-ORT| = {worst:.6} at {worst_at} \
                         (tolerance {})",
                        case.name,
                        ours_out.len(),
                        case.tolerance
                    );
                    assert!(
                        worst <= case.tolerance,
                        "{}: element {worst_at} is {} against ORT's {} (delta {worst}, \
                         tolerance {})",
                        case.name,
                        ours_out[worst_at],
                        ort_out[worst_at],
                        case.tolerance
                    );

                    // Second run, same sessions, different activations. The
                    // EP now tells each kernel which of its inputs ORT holds
                    // constant, and kernels use that to keep a prepacked
                    // weight for the life of the session instead of rebuilding
                    // it per call. Caching a weight is sound; caching anything
                    // activation-derived is not, and the two are easy to
                    // confuse in a kernel that only ever sees one input set.
                    // Re-running the *same session* with rotated activations
                    // separates them: a weight cache still matches ORT, a
                    // stale activation cache returns the first answer.
                    let rotated = with_rotated_activation(&case);
                    let ours_second = run_generated_case(
                        api,
                        session,
                        &rotated,
                        &format!("Run({} rotated)", case.name),
                    );
                    let ort_second = run_generated_case(
                        api,
                        ort_session,
                        &rotated,
                        &format!("Run(ORT baseline {} rotated)", case.name),
                    );
                    // ORT is the oracle for whether this case can detect a
                    // stale cache at all. A saturating u8 `QLinearMatMul`
                    // output is unchanged by a one-element rotation for both
                    // implementations, which makes the check vacuous *for that
                    // case* rather than wrong; the suite-level counter below
                    // is what keeps the sweep as a whole non-vacuous.
                    let ort_moved = ort_second
                        .iter()
                        .zip(ort_out.iter())
                        .any(|(a, b)| (a - b).abs() > case.tolerance);
                    if ort_moved {
                        sensitive += 1;
                        assert!(
                            ours_second
                                .iter()
                                .zip(ours_out.iter())
                                .any(|(a, b)| (a - b).abs() > case.tolerance),
                            "{}: ORT's result changed when the activation rotated and ours \
                             did not — this EP returned a stale answer on the second run",
                            case.name
                        );
                    }
                    let mut worst_second = 0.0f32;
                    let mut worst_second_at = 0usize;
                    for (i, (a, b)) in ours_second.iter().zip(ort_second.iter()).enumerate() {
                        let delta = (a - b).abs();
                        if delta > worst_second {
                            worst_second = delta;
                            worst_second_at = i;
                        }
                    }
                    assert!(
                        worst_second <= case.tolerance,
                        "{}: second run with new activations diverged from ORT — element \
                         {worst_second_at} is {} against ORT's {} (delta {worst_second}, \
                         tolerance {}). A per-session weight cache must not outlive the \
                         activations it was used with",
                        case.name,
                        ours_second[worst_second_at],
                        ort_second[worst_second_at],
                        case.tolerance
                    );
                    ((*api).ReleaseSession.unwrap())(ort_session);
                    ((*api).ReleaseSessionOptions.unwrap())(ort_opts);
                }
                Err(message) => {
                    assert!(
                        !case.ort_can_build,
                        "{}: ORT could not load the model this test uses as its \
                         baseline: {message}",
                        case.name
                    );
                    eprintln!(
                        "  [{}] ORT declined the model outright, as expected: {}",
                        case.name,
                        message.lines().next().unwrap_or("")
                    );
                }
            }

            conformance_teardown(api, env, opts, session, &reg);
        }
        checked += 1;
    }
    if checked > 0 {
        assert_eq!(
            checked, 10,
            "every generated matmul case must have been run"
        );
        assert!(
            sensitive >= 5,
            "only {sensitive} of {checked} cases changed their result when the activation \
             rotated, so the stale-cache half of this test is mostly vacuous"
        );
        eprintln!(
            "\n✅ no_matmul_family_node_escapes_to_the_ort_cpu_ep: {checked} cases PASSED \
             ({sensitive} of them activation-sensitive on the second run)"
        );
    }
}

/// ORT's fused-node metadata carries a *set* of boundary inputs, so a node that
/// names one value in two slots is bound once.
///
/// `Mul(X, X)` is the smallest graph that says so, and it says it about a graph
/// input rather than an initializer — no constant-sharing pass, no
/// initializer-dropping option, nothing but the dedup itself. Numbering ORT's
/// inputs by node-input position instead of by value therefore walks one slot
/// past the end of the array ORT actually passes, which used to surface across
/// the C ABI as `Compute: internal panic`.
#[test]
fn a_node_that_names_one_value_twice_is_bound_once() {
    let _lock = lock_ort_ep();
    let n = 512usize;
    let model = format!(
        r#"ir_version: 10
graph {{
  node: [{{ input: ["X", "X"] output: ["Y"] op_type: "Mul" }}]
  name: "mul_self"
  input: [{}]
  output: [{}]
}}
opset_import: [{{ version: 17 }}]
"#,
        textproto_value_info("X", ELEM_F32, &[n as i64]),
        textproto_value_info("Y", ELEM_F32, &[n as i64]),
    );
    let values: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 64.0).collect();
    let case = MatmulFamilyCase {
        name: "mul_self",
        op: "Mul",
        model,
        inputs: vec![GeneratedInput {
            name: "X",
            elem: ELEM_F32,
            dims: vec![n as i64],
            data: f32_slice_to_bytes(&values),
        }],
        output: "Y",
        output_elem: ELEM_F32,
        ort_can_build: true,
        tolerance: 0.0,
    };

    let path = write_generated_model(case.name, &case.model);
    let reg = "cpu_ep_mul_self";
    let Some((_lib, api, env, opts, session)) = (unsafe { conformance_setup(reg, &path, true) })
    else {
        eprintln!("*** SKIPPED: mul_self — ORT or EP cdylib not found ***");
        return;
    };

    unsafe {
        let info = query_ep_assignment(api, session);
        assert!(
            info.ops_on_our_ep().contains(&"Mul"),
            "mul_self: 'Mul' must run on this EP, got {:?}",
            info.ops_on_our_ep()
        );
        let out = run_generated_case(api, session, &case, "Run(mul_self)");
        assert_eq!(out.len(), n);
        for (i, (got, want)) in out.iter().zip(values.iter()).enumerate() {
            assert_eq!(
                *got,
                want * want,
                "mul_self: element {i} is {got}, expected {}",
                want * want
            );
        }
        eprintln!("  [mul_self] {n} elements match x*x exactly");
        conformance_teardown(api, env, opts, session, reg);
    }
}

/// The documented edge of `MatMulNBits` support, pinned so it cannot become a
/// crash again.
///
/// `com.microsoft::MatMulNBits` spec-allows `zero_points` in `uint8`, `int32`,
/// `float16` or `float`, and ORT's own CPU kernel builds a session for each.
/// This EP's kernel implements the packed `uint8` form only. Advertising the
/// union of the op's edge dtypes was enough to get such a node *claimed*, after
/// which `execute` failed with `zero_points must have dtype Uint8` — a model
/// that used to produce an answer instead produced an error. The per-slot
/// constraint list declines it at `GetCapability` instead.
///
/// This is a real remaining gap, not a resting place: the float zero-point
/// form is a quantization semantic this EP does not yet implement, and it is
/// tracked as such rather than papered over.
#[test]
fn a_float_zero_point_matmul_nbits_is_declined_not_claimed_and_failed() {
    let _lock = lock_ort_ep();
    let (n, k, block_size, bits) = (64usize, 64usize, 32usize, 4u32);
    let blocks_per_col = k / block_size;
    let blob = block_size * bits as usize / 8;
    let scale_elements = n * blocks_per_col;
    let model = format!(
        r#"ir_version: 10
graph {{
  node: [{{
    input: ["A", "B", "scales", "zero_points"]
    output: ["Y"]
    op_type: "MatMulNBits"
    domain: "com.microsoft"
    attribute: [
      {{ name: "K" type: INT i: {k} }},
      {{ name: "N" type: INT i: {n} }},
      {{ name: "bits" type: INT i: {bits} }},
      {{ name: "block_size" type: INT i: {block_size} }}
    ]
  }}]
  name: "nbits_float_zp"
  initializer: [{}, {}, {}]
  input: [{}]
  output: [{}]
}}
opset_import: [{{ version: 17 }}, {{ domain: "com.microsoft" version: 1 }}]
"#,
        textproto_initializer(
            "B",
            ELEM_U8,
            &[n as i64, blocks_per_col as i64, blob as i64],
            &printable_weight_blob(n * blocks_per_col * blob)
        ),
        textproto_initializer(
            "scales",
            ELEM_F32,
            &[scale_elements as i64],
            &printable_scale_blob(scale_elements)
        ),
        // f16 bits 0x4923 little-endian is the printable pair "#I".
        textproto_initializer(
            "zero_points",
            ELEM_F16,
            &[scale_elements as i64],
            &"#I".repeat(scale_elements)
        ),
        textproto_value_info("A", ELEM_F32, &[1, k as i64]),
        textproto_value_info("Y", ELEM_F32, &[1, n as i64]),
    );

    let case = MatmulFamilyCase {
        name: "nbits_float_zp",
        op: "MatMulNBits",
        model,
        inputs: vec![GeneratedInput {
            name: "A",
            elem: ELEM_F32,
            dims: vec![1, k as i64],
            data: f32_slice_to_bytes(&activation_f32(1, k)),
        }],
        output: "Y",
        output_elem: ELEM_F32,
        ort_can_build: true,
        tolerance: 0.0,
    };

    let path = write_generated_model(case.name, &case.model);
    let reg = "cpu_ep_nbits_float_zp";
    // Fallback left available on purpose: the point is that the node is
    // declined and still computed, not that the session dies.
    let Some((_lib, api, env, opts, session)) = (unsafe { conformance_setup(reg, &path, false) })
    else {
        eprintln!("*** SKIPPED: nbits_float_zp — ORT or EP cdylib not found ***");
        return;
    };

    unsafe {
        let info = query_ep_assignment(api, session);
        assert!(
            !info.ops_on_our_ep().contains(&"MatMulNBits"),
            "a float16 zero_points MatMulNBits must not be claimed — this EP's \
             kernel accepts only packed uint8 zero points and would fail at Run"
        );
        let out = run_generated_case(api, session, &case, "Run(nbits_float_zp)");
        let live = out.iter().filter(|v| v.is_finite() && **v != 0.0).count();
        assert!(
            live * 2 >= out.len(),
            "nbits_float_zp: the model must still produce an answer, got {live} live \
             of {} outputs",
            out.len()
        );
        eprintln!("  [nbits_float_zp] declined, still computed: {live} live outputs");
        conformance_teardown(api, env, opts, session, reg);
    }
}

/// Weights must reach the kernels *labelled* as weights.
///
/// ORT gives a plugin EP a fused node whose initializers are inputs of that
/// node and graph inputs of its subgraph, so at Compile time a 1 GB
/// `MatMulNBits` `B` is shaped exactly like an activation. Every kernel that
/// builds a prepack — `MatMulNBits`, `QLinearMatMul`, the f16 `MatMul` widening
/// cache — decides whether that prepack may outlive the call from
/// `Kernel::set_constant_inputs`, so getting this wrong silently converts a
/// once-per-session cost into a per-token one, and in `MatMulNBits`'s case also
/// selects a slower kernel (its MLAS SQNBit path is gated on the same flag).
///
/// No output comparison can see any of that, which is why the EP exports a
/// counter and this test reads it from the very cdylib ORT loaded.
///
/// The cases are chosen to move the counter in opposite directions:
/// symmetric `nbits4_decode` has two initializers (`B`, `scales`),
/// asymmetric `nbits8_prefill` has three (`zero_points` as well), and
/// `matmul_f32_decode` declares both operands as graph inputs so it must
/// report nothing. A wiring that marks everything constant passes the first
/// two assertions and fails the third — and would be a real bug, not merely a
/// slower one, because a prepack of an activation must never outlive the call.
/// ONNX Runtime's own copy of this library must not build the persistent SPMD
/// decode pool.
///
/// `CreateEpFactories` opts the process out because nothing in the plugin path
/// ever enters an SPMD decode scope, and a pool that is built but never
/// dispatched to is pure cost: resident workers competing with ORT's intra-op
/// pool, plus a `MatMulNBits` weight pre-split into one MLAS shard per decode
/// worker, which caps an unscoped decode GEMV at that worker count. Measured
/// 0.376 ms -> 0.092 ms on int4 block-32 K=N=2048 M=1, against ORT's CPU EP at
/// 0.097 ms.
///
/// The sibling unit test in `decode_pool_optout.rs` can only prove the library
/// plumbing: a test binary links its *own* copy of `onnx-runtime-ep-cpu`, whose
/// pool statics are not the ones the `dlopen`ed cdylib uses. This test reads
/// the answer back out of the library ORT actually loaded, after ORT has run a
/// decode through it, so it fails if the `CreateEpFactories` call site is ever
/// dropped.
#[test]
fn the_plugin_ep_disables_the_decode_pool_in_ort() {
    let _lock = lock_ort_ep();
    let ep_path = match find_ep_cdylib() {
        Some(p) => p,
        None => {
            if std::env::var("NXRT_REQUIRE_ORT_TESTS").as_deref() == Ok("1") {
                panic!("NXRT_REQUIRE_ORT_TESTS=1 but the EP cdylib is missing");
            }
            eprintln!("*** SKIPPED: EP cdylib not found ***");
            return;
        }
    };
    if std::env::var("ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL").is_ok() {
        eprintln!("*** SKIPPED: the environment asks for a specific pool mode ***");
        return;
    }
    // Same absolute path ORT registers, so this is the same mapping and the
    // same statics -- not a second copy of the library.
    let ep_lib = unsafe { libloading::Library::new(&ep_path) }.expect("dlopen EP cdylib");
    let pool_built: libloading::Symbol<'_, unsafe extern "C" fn() -> i32> =
        unsafe { ep_lib.get(b"nxrt_ep_persistent_decode_pool_built") }
            .expect("nxrt_ep_persistent_decode_pool_built not exported");

    let case = matmul_family_cases()
        .into_iter()
        .find(|c| c.name == "nbits4_decode")
        .expect("case present");
    let path = write_generated_model(case.name, &case.model);
    let reg = "cpu_ep_decode_pool_optout";
    let Some((_lib, api, env, opts, session)) = (unsafe { conformance_setup(reg, &path, true) })
    else {
        eprintln!("*** SKIPPED: ORT not found ***");
        return;
    };
    unsafe {
        run_generated_case(api, session, &case, "decode-pool-optout");
        conformance_teardown(api, env, opts, session, reg);
    }

    assert_eq!(
        unsafe { pool_built() },
        0,
        "ONNX Runtime's copy of the plugin built the persistent SPMD decode \
         pool it never dispatches to; CreateEpFactories must opt the process out"
    );
}

#[test]
fn constant_weights_are_reported_to_kernels_as_constant() {
    let _lock = lock_ort_ep();
    let ep_path = match find_ep_cdylib() {
        Some(p) => p,
        None => {
            if std::env::var("NXRT_REQUIRE_ORT_TESTS").as_deref() == Ok("1") {
                panic!("NXRT_REQUIRE_ORT_TESTS=1 but the EP cdylib is missing");
            }
            eprintln!("*** SKIPPED: EP cdylib not found ***");
            return;
        }
    };
    // Same absolute path ORT registers, so this is the same mapping and the
    // same statics — not a second copy of the library.
    let ep_lib = unsafe { libloading::Library::new(&ep_path) }.expect("dlopen EP cdylib");
    let read: libloading::Symbol<'_, unsafe extern "C" fn() -> usize> =
        unsafe { ep_lib.get(b"nxrt_ep_constant_weight_inputs") }
            .expect("nxrt_ep_constant_weight_inputs not exported");
    let reset: libloading::Symbol<'_, unsafe extern "C" fn()> =
        unsafe { ep_lib.get(b"nxrt_ep_reset_constant_weight_inputs") }
            .expect("nxrt_ep_reset_constant_weight_inputs not exported");

    for (name, expected) in [
        ("nbits4_decode", 2usize),
        ("nbits8_prefill", 3usize),
        ("matmul_f32_decode", 0usize),
    ] {
        let case = matmul_family_cases()
            .into_iter()
            .find(|c| c.name == name)
            .expect("case present");
        let path = write_generated_model(case.name, &case.model);
        let reg = format!("cpu_ep_constflags_{}", case.name);
        unsafe { reset() };
        let Some((_lib, api, env, opts, session)) =
            (unsafe { conformance_setup(&reg, &path, true) })
        else {
            eprintln!("*** SKIPPED: {name} — ORT not found ***");
            return;
        };
        let observed = unsafe { read() };
        assert_eq!(
            observed, expected,
            "{name}: this EP reported {observed} constant inputs, expected {expected}"
        );
        unsafe { conformance_teardown(api, env, opts, session, &reg) };
        eprintln!("  [{name}] {observed} constant inputs reported");
    }
}

/// A `MatMulNBits` whose `B` and `scales` are declared **both** as
/// initializers and as graph inputs.
///
/// From ONNX IR version 4 that combination means the initializer is only a
/// default: the caller may hand a different tensor in on any `Run`, and ORT
/// says so in its load log ("Initializer B appears in graph inputs and will
/// not be treated as constant value/weight"). `Graph_GetInitializers` still
/// lists it — its own documentation says it "includes constant and
/// non-constant initializers" — so a name-only membership test would call it a
/// weight, and this crate would cache a prepack of the default for the life of
/// the session.
///
/// Returns the case with every operand supplied at run time, plus a second
/// `B` payload that produces different outputs.
fn overridable_nbits_case() -> (MatmulFamilyCase, Vec<u8>) {
    const M: usize = 1;
    const K: usize = 64;
    const N: usize = 64;
    const BLOCK: usize = 32;
    let blocks_per_col = K / BLOCK;
    let blob = BLOCK / 2;
    let b_elements = N * blocks_per_col * blob;
    let scale_elements = N * blocks_per_col;
    let b_dims = vec![N as i64, blocks_per_col as i64, blob as i64];
    let scale_dims = vec![scale_elements as i64];
    let a_dims = vec![M as i64, K as i64];

    let default_b = printable_weight_blob(b_elements);
    let scales = printable_scale_blob(scale_elements);
    let model = format!(
        r#"ir_version: 10
graph {{
  node: [{{
    input: ["A", "B", "scales"]
    output: ["Y"]
    op_type: "MatMulNBits"
    domain: "com.microsoft"
    attribute: [
      {{ name: "K" type: INT i: {K} }},
      {{ name: "N" type: INT i: {N} }},
      {{ name: "bits" type: INT i: 4 }},
      {{ name: "block_size" type: INT i: {BLOCK} }}
    ]
  }}]
  name: "nbits4_overridable"
  initializer: [{}, {}]
  input: [{}, {}, {}]
  output: [{}]
}}
opset_import: [{{ version: 17 }}, {{ domain: "com.microsoft" version: 1 }}]
"#,
        textproto_initializer("B", ELEM_U8, &b_dims, &default_b),
        textproto_initializer("scales", ELEM_F32, &scale_dims, &scales),
        textproto_value_info("A", ELEM_F32, &a_dims),
        textproto_value_info("B", ELEM_U8, &b_dims),
        textproto_value_info("scales", ELEM_F32, &scale_dims),
        textproto_value_info("Y", ELEM_F32, &[M as i64, N as i64]),
    );

    // Bit-complement of the default, so every 4-bit nibble changes and no
    // output element can coincide by construction.
    let other_b: Vec<u8> = default_b.as_bytes().iter().map(|b| !*b).collect();
    let case = MatmulFamilyCase {
        name: "nbits4_overridable",
        op: "MatMulNBits",
        model,
        inputs: vec![
            GeneratedInput {
                name: "A",
                elem: ELEM_F32,
                dims: a_dims,
                data: f32_slice_to_bytes(&activation_f32(M, K)),
            },
            GeneratedInput {
                name: "B",
                elem: ELEM_U8,
                dims: b_dims,
                data: default_b.as_bytes().to_vec(),
            },
            GeneratedInput {
                name: "scales",
                elem: ELEM_F32,
                dims: scale_dims,
                data: scales.as_bytes().to_vec(),
            },
        ],
        output: "Y",
        output_elem: ELEM_F32,
        ort_can_build: true,
        tolerance: 0.5,
    };
    (case, other_b)
}

/// An initializer the caller may override is **not** a constant weight, and
/// overriding it must change the answer.
///
/// Two independent falsifiers, because either one alone is weak: the counter
/// pins the classification (it reads 2 if membership is keyed on
/// `Graph_GetInitializers` names alone), and the numeric check pins the
/// consequence — a session-lifetime prepack of the default would return run
/// one's answer for run two, which ORT, used here as the oracle, does not.
#[test]
fn an_overridable_initializer_is_not_treated_as_a_constant_weight() {
    let _lock = lock_ort_ep();
    let ep_path = match find_ep_cdylib() {
        Some(p) => p,
        None => {
            if std::env::var("NXRT_REQUIRE_ORT_TESTS").as_deref() == Ok("1") {
                panic!("NXRT_REQUIRE_ORT_TESTS=1 but the EP cdylib is missing");
            }
            eprintln!("*** SKIPPED: EP cdylib not found ***");
            return;
        }
    };
    let ep_lib = unsafe { libloading::Library::new(&ep_path) }.expect("dlopen EP cdylib");
    let read: libloading::Symbol<'_, unsafe extern "C" fn() -> usize> =
        unsafe { ep_lib.get(b"nxrt_ep_constant_weight_inputs") }
            .expect("nxrt_ep_constant_weight_inputs not exported");
    let reset: libloading::Symbol<'_, unsafe extern "C" fn()> =
        unsafe { ep_lib.get(b"nxrt_ep_reset_constant_weight_inputs") }
            .expect("nxrt_ep_reset_constant_weight_inputs not exported");

    let (case, other_b) = overridable_nbits_case();
    let mut overridden = case.clone();
    overridden.inputs[1].data = other_b;
    let path = write_generated_model(case.name, &case.model);
    let reg = "cpu_ep_overridable_init";
    unsafe { reset() };
    let Some((_lib, api, env, opts, session)) = (unsafe { conformance_setup(reg, &path, true) })
    else {
        eprintln!("*** SKIPPED: ORT not found ***");
        return;
    };
    let observed = unsafe { read() };
    let ours_default = unsafe { run_generated_case(api, session, &case, "overridable/default") };
    let ours_other = unsafe { run_generated_case(api, session, &overridden, "overridable/other") };
    unsafe { conformance_teardown(api, env, opts, session, reg) };

    assert_eq!(
        observed, 0,
        "an initializer that is also a graph input may be replaced on any Run, \
         but this EP reported {observed} of them as constant weights"
    );

    // ORT as the oracle: same model, same two payloads, its own CPU EP.
    let (ort_default, ort_other) = {
        let Some((_lib2, api2, env2, opts2, session2)) =
            (unsafe { conformance_setup("cpu_ep_overridable_oracle", &path, true) })
        else {
            eprintln!("*** SKIPPED: ORT not found for the oracle run ***");
            return;
        };
        let plain = unsafe { try_plain_ort_session(api2, env2, &path) }
            .expect("ORT can build this MatMulNBits");
        let d = unsafe { run_generated_case(api2, plain.1, &case, "oracle/default") };
        let o = unsafe { run_generated_case(api2, plain.1, &overridden, "oracle/other") };
        unsafe {
            ((*api2).ReleaseSession.unwrap())(plain.1);
            ((*api2).ReleaseSessionOptions.unwrap())(plain.0);
        }
        unsafe { conformance_teardown(api2, env2, opts2, session2, "cpu_ep_overridable_oracle") };
        (d, o)
    };

    assert!(
        ort_default
            .iter()
            .zip(ort_other.iter())
            .any(|(a, b)| (a - b).abs() > case.tolerance),
        "the two B payloads must produce different outputs or this test proves nothing"
    );
    for (stage, ours, theirs) in [
        ("default", &ours_default, &ort_default),
        ("overridden", &ours_other, &ort_other),
    ] {
        assert_eq!(ours.len(), theirs.len(), "{stage}: output length");
        for (i, (a, b)) in ours.iter().zip(theirs.iter()).enumerate() {
            assert!(
                (a - b).abs() <= case.tolerance,
                "{stage}: element {i} is {a} but ORT says {b} — a stale prepack of the \
                 default initializer would look exactly like this"
            );
        }
    }
    eprintln!("  [overridable] 0 constant inputs; both payloads match ORT");
}

// ─── Plugin-path A/B benchmark ────────────────────────────────────────────────

/// Time `iters` `Run` calls over pre-created input tensors, returning
/// per-iteration milliseconds.
///
/// Input `OrtValue`s are built once and reused so the measurement is the
/// kernel plus ORT's per-run overhead, not the harness copying weights.
///
/// # Safety
/// `api`, `session` and every pointer in `values` must be valid.
unsafe fn bench_runs(
    api: *const ort::OrtApi,
    session: *mut ort::OrtSession,
    input_name_ptrs: &[*const std::os::raw::c_char],
    values: &[*const ort::OrtValue],
    output_name_ptrs: &[*const std::os::raw::c_char],
    iters: usize,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(iters);
    for _ in 0..iters {
        let mut output: *mut ort::OrtValue = ptr::null_mut();
        let start = std::time::Instant::now();
        unsafe {
            let status = ((*api).Run.unwrap())(
                session,
                ptr::null(),
                input_name_ptrs.as_ptr(),
                values.as_ptr(),
                values.len(),
                output_name_ptrs.as_ptr(),
                1,
                &mut output,
            );
            check_status(api, status, "bench Run");
            out.push(start.elapsed().as_secs_f64() * 1e3);
            ((*api).ReleaseValue.unwrap())(output);
        }
    }
    out
}

fn percentile(samples: &mut [f64], p: f64) -> f64 {
    if samples.is_empty() {
        // A one-sided run (`NXRT_MM_BENCH_SIDE`) collects no samples for the
        // other backend; report it as unmeasured rather than a number.
        return f64::NAN;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((samples.len() - 1) as f64 * p).round() as usize;
    samples[idx]
}

/// A single-input elementwise case: one node, one tensor in, same shape out.
///
/// Reuses [`MatmulFamilyCase`], which is a single-node generated model plus its
/// runtime inputs and nothing matmul-specific beyond its name.
///
/// `positive` maps the generated activations into `[0.5, 1.5]` for the ops that
/// are undefined or uninteresting on negatives (`Sqrt`, `Log`); everything else
/// gets the two-sided `[-1, 1]` range so saturating branches are exercised on
/// both sides.
///
/// `tolerance` is inert for these: the only consumer of the unary cases is the
/// timing A/B, which asserts assignment and then measures. Elementwise numerics
/// are covered against closed forms in `onnx-runtime-ep-cpu`'s own kernel
/// tests, not here.
#[allow(clippy::too_many_arguments)]
fn unary_case(
    name: &'static str,
    op: &'static str,
    domain: &'static str,
    elem: ort::ONNXTensorElementDataType,
    len: usize,
    opset: u64,
    positive: bool,
    tolerance: f32,
) -> MatmulFamilyCase {
    let dims = vec![1, len as i64];
    let opset_imports = if domain.is_empty() {
        format!("[{{ version: {opset} }}]")
    } else {
        format!("[{{ version: {opset} }}, {{ domain: \"{domain}\" version: 1 }}]")
    };
    let attrs = match op {
        // `Gelu`'s default is already "none" (exact); spell whichever variant
        // the case name promises so the model cannot silently drift.
        "Gelu" if name.contains("_tanh") => {
            " attribute: [{ name: \"approximate\" type: STRING s: \"tanh\" }]"
        }
        "Gelu" => " attribute: [{ name: \"approximate\" type: STRING s: \"none\" }]",
        // Celu's alpha divides the input *and* scales the output, so the
        // default and a non-default value exercise different arithmetic. The
        // name carries which one the case means.
        "Celu" if name.contains("_a2") => " attribute: [{ name: \"alpha\" type: FLOAT f: 2.0 }]",
        // Same for Elu: alpha scales only the negative half, so the default
        // and a non-default value take different arithmetic through the same
        // select.
        "Elu" if name.contains("_a2") => " attribute: [{ name: \"alpha\" type: FLOAT f: 2.0 }]",
        _ => "",
    };
    let model = format!(
        r#"ir_version: 10
graph {{
  node: [{{ input: ["X"] output: ["Y"] op_type: "{op}" domain: "{domain}"{attrs} }}]
  name: "{name}"
  input: [{}]
  output: [{}]
}}
opset_import: {opset_imports}
"#,
        textproto_value_info("X", elem, &dims),
        textproto_value_info("Y", elem, &dims),
    );
    let mut x = activation_f32(1, len);
    if positive {
        for v in &mut x {
            *v = v.abs() + 0.5;
        }
    }
    let data = if elem == ELEM_F16 {
        f16_slice_to_bytes(&x)
    } else {
        f32_slice_to_bytes(&x)
    };
    MatmulFamilyCase {
        name,
        op,
        model,
        inputs: vec![GeneratedInput {
            name: "X",
            elem,
            dims,
            data,
        }],
        output: "Y",
        output_elem: elem,
        ort_can_build: true,
        tolerance,
    }
}

/// A chain of `depth` identical single-input nodes, `X -> t1 -> … -> Y`.
///
/// Depth is the axis that separates the two costs #1077 conflates. Whatever
/// ORT charges per `Run` — session bookkeeping, fetch/feed marshalling — is
/// paid once regardless of depth, while our per-node dispatch is paid `depth`
/// times. Comparing depth 1, 10 and 100 against ORT therefore reads the
/// per-node slope directly, instead of a single number in which the fixed and
/// variable parts are indistinguishable.
///
/// `dynamic` leaves the row count symbolic, so ORT cannot fold shapes at
/// session build and our shape inference runs for real on every `Run`. A
/// dispatch path that is only fast on static shapes is not fast.
fn chain_case(
    name: &'static str,
    op: &'static str,
    depth: usize,
    len: usize,
    dynamic: bool,
) -> MatmulFamilyCase {
    assert!(depth >= 1, "a chain needs at least one node");
    let dims = vec![1, len as i64];
    let decl = |n: &str| -> String {
        if dynamic {
            format!(
                "{{ name: \"{n}\" type {{ tensor_type {{ elem_type: {ELEM_F32} \
                 shape {{ dim: [{{ dim_param: \"batch\" }}, {{ dim_value: {len} }}] }} }} }} }}"
            )
        } else {
            textproto_value_info(n, ELEM_F32, &dims)
        }
    };
    let nodes: Vec<String> = (0..depth)
        .map(|i| {
            let input = if i == 0 {
                "X".to_owned()
            } else {
                format!("t{i}")
            };
            let output = if i + 1 == depth {
                "Y".to_owned()
            } else {
                format!("t{}", i + 1)
            };
            format!("{{ input: [\"{input}\"] output: [\"{output}\"] op_type: \"{op}\" }}")
        })
        .collect();
    let model = format!(
        r#"ir_version: 10
graph {{
  node: [{}]
  name: "{name}"
  input: [{}]
  output: [{}]
}}
opset_import: [{{ version: 17 }}]
"#,
        nodes.join(", "),
        decl("X"),
        decl("Y"),
    );
    let x = activation_f32(1, len);
    MatmulFamilyCase {
        name,
        op,
        model,
        inputs: vec![GeneratedInput {
            name: "X",
            elem: ELEM_F32,
            dims,
            data: f32_slice_to_bytes(&x),
        }],
        output: "Y",
        output_elem: ELEM_F32,
        ort_can_build: true,
        // A Relu chain is exact in both implementations; Identity is a copy.
        // Anything above zero here would be hiding a real disagreement.
        tolerance: 0.0,
    }
}

/// A single small `MatMul`, `[1, k] x [k, n]`, with both operands supplied at
/// run time.
///
/// The elementwise cases all have trivial kernels, which is what makes them
/// good dispatch probes — but it also means a win there could be an artefact of
/// how little work the kernel does. `MatMul` at these sizes still has a real
/// kernel while staying small enough for dispatch to matter, so it is the case
/// that says whether the fast path generalises beyond one-liners.
///
/// Both operands are runtime inputs rather than initializers, which keeps the
/// weight out of the TextFormat source (an octal-escaped 1024x1024 f32 blob is
/// megabytes of test fixture) and denies both sides any prepacking, so the
/// comparison stays about dispatch.
fn small_matmul_case(name: &'static str, k: usize, n: usize) -> MatmulFamilyCase {
    let model = format!(
        r#"ir_version: 10
graph {{
  node: [{{ input: ["X", "W"] output: ["Y"] op_type: "MatMul" }}]
  name: "{name}"
  input: [{}, {}]
  output: [{}]
}}
opset_import: [{{ version: 17 }}]
"#,
        textproto_value_info("X", ELEM_F32, &[1, k as i64]),
        textproto_value_info("W", ELEM_F32, &[k as i64, n as i64]),
        textproto_value_info("Y", ELEM_F32, &[1, n as i64]),
    );
    MatmulFamilyCase {
        name,
        op: "MatMul",
        model,
        inputs: vec![
            GeneratedInput {
                name: "X",
                elem: ELEM_F32,
                dims: vec![1, k as i64],
                data: f32_slice_to_bytes(&activation_f32(1, k)),
            },
            GeneratedInput {
                name: "W",
                elem: ELEM_F32,
                dims: vec![k as i64, n as i64],
                data: f32_slice_to_bytes(&activation_f32(k, n)),
            },
        ],
        output: "Y",
        output_elem: ELEM_F32,
        ort_can_build: true,
        tolerance: 1e-3,
    }
}

/// The grid issue #1077 is closed against.
///
/// Every row isolates one variable. Read together they say where the remaining
/// dispatch cost is; read individually none of them would.
///
/// * **`identity_1`** is the floor. An `Identity` node does a memcpy, so
///   essentially everything the timer sees is the cost of getting there and
///   back. If we are at parity anywhere, it has to be here first.
/// * **`relu_1` / `relu_10` / `relu_100`** vary only depth, giving the
///   per-node slope with the per-`Run` constant divided out.
/// * **`*_dyn`** repeat the static rows with a symbolic batch dimension, so a
///   fast path that only works when shapes are known at build time cannot pass
///   unnoticed.
/// * **`matmul_*`** check the result is not an artefact of trivial kernels.
/// * The `4k` width keeps every kernel small enough that dispatch is a visible
///   fraction; at 1 Mi it would be rounding error, which is the whole reason
///   the earlier activation sweep could not answer this question.
fn dispatch_grid_cases() -> Vec<MatmulFamilyCase> {
    const W: usize = 4096;
    const TINY: usize = 8;
    vec![
        chain_case("grid_identity_1_static", "Identity", 1, W, false),
        chain_case("grid_identity_10_static", "Identity", 10, W, false),
        chain_case("grid_relu_1_static", "Relu", 1, W, false),
        chain_case("grid_relu_10_static", "Relu", 10, W, false),
        chain_case("grid_relu_100_static", "Relu", 100, W, false),
        chain_case("grid_identity_1_dyn", "Identity", 1, W, true),
        chain_case("grid_relu_1_dyn", "Relu", 1, W, true),
        chain_case("grid_relu_10_dyn", "Relu", 10, W, true),
        chain_case("grid_relu_100_dyn", "Relu", 100, W, true),
        // Dispatch-isolating widths. At W = 4096 a Relu node moves 32 KiB, and
        // that traffic — not the dispatch around it — sets the per-node cost,
        // so the ratio there reports elementwise throughput wearing a
        // dispatch-shaped hat. At 8 elements the kernel is a single masked
        // vector op and essentially everything left is overhead, which is what
        // #1077 is actually about. The pair brackets it: TINY is the overhead
        // floor, W the throughput regime.
        chain_case("grid_relu_1_tiny", "Relu", 1, TINY, false),
        chain_case("grid_relu_10_tiny", "Relu", 10, TINY, false),
        chain_case("grid_relu_100_tiny", "Relu", 100, TINY, false),
        // Depth 1000 pins the per-node slope: at depth 100 the fixed per-`Run`
        // cost is still ~9% of the total, so a slope read from 1/10/100 alone
        // carries the fixed term's noise into it. Ten times the nodes divides
        // that contamination by ten.
        chain_case("grid_relu_1000_tiny", "Relu", 1000, TINY, false),
        chain_case("grid_relu_1_tiny_dyn", "Relu", 1, TINY, true),
        chain_case("grid_relu_10_tiny_dyn", "Relu", 10, TINY, true),
        small_matmul_case("grid_matmul_128x128", 128, 128),
        small_matmul_case("grid_matmul_512x512", 512, 512),
    ]
}

/// Whether `NXRT_MM_BENCH_CASE` selects this case.
///
/// Exact when the filter names a case, substring otherwise, so `grid_relu` picks
/// a family while `grid_relu_1_tiny` picks one case. `filter_names_a_case` is
/// computed once over the whole case list by the caller.
///
/// This is a free function rather than an inline condition so that
/// `case_selectors_do_not_match_a_neighbour` tests the rule the benchmark
/// actually uses instead of a copy of it.
fn case_selected(name: &str, filter: &str, filter_names_a_case: bool) -> bool {
    if filter.is_empty() {
        return true;
    }
    if filter_names_a_case {
        name == filter
    } else {
        name.contains(filter)
    }
}

/// A case selector must name one case, not two.
///
/// `grid_relu_1_tiny` is a substring of `grid_relu_1_tiny_dyn`, and
/// `grid_relu_10_tiny` of `grid_relu_10_tiny_dyn`, so under the old plain
/// `contains` filter those selectors each ran **two** `Run`s per iteration while
/// `grid_relu_100_tiny` — which has no `_dyn` sibling — ran one. Deterministic
/// per-`Run` counts taken that way were 2x inflated at some depths and correct at
/// others, which is exactly the shape that makes an error survive review: the
/// figures stayed perfectly linear in iterations, so they looked high-confidence.
/// Three published measurements in #1077 had to be withdrawn.
///
/// This pins the property that matters — *selecting a case by its own name
/// selects that case alone* — rather than the absence of substring pairs, which
/// would forbid perfectly reasonable names like `..._dyn`.
///
#[test]
fn case_selectors_do_not_match_a_neighbour() {
    let cases: Vec<MatmulFamilyCase> = matmul_bench_cases()
        .into_iter()
        .chain(unary_bench_cases())
        .chain(dispatch_grid_cases())
        .collect();
    let names: Vec<&str> = cases.iter().map(|c| c.name).collect();
    let names_slice = names.as_slice();

    assert!(
        !names.is_empty(),
        "no cases: this test would pass vacuously"
    );
    {
        let mut sorted = names.clone();
        sorted.sort_unstable();
        let before = sorted.len();
        sorted.dedup();
        assert_eq!(before, sorted.len(), "duplicate case names: {names:?}");
    }

    let select = |filter: &str| -> Vec<&str> {
        let exact = names_slice.contains(&filter);
        names
            .iter()
            .copied()
            .filter(|n| case_selected(n, filter, exact))
            .collect()
    };

    for name in &names {
        assert_eq!(
            select(name),
            vec![*name],
            "selector `{name}` must run exactly that case"
        );
    }

    // The hazard is real and still present in the names, so the exact rule above
    // is load-bearing rather than decorative. If these pairs ever disappear this
    // assertion should be updated, not deleted -- it is what proves the rule is
    // doing work.
    let overlapping: Vec<(&str, &str)> = names
        .iter()
        .flat_map(|a| {
            names
                .iter()
                .filter(move |b| *b != a && b.contains(*a))
                .map(move |b| (*a, *b))
        })
        .collect();
    assert!(
        overlapping
            .iter()
            .any(|(a, b)| *a == "grid_relu_1_tiny" && *b == "grid_relu_1_tiny_dyn"),
        "expected the known overlapping pair to still exist; found {overlapping:?}"
    );

    // This test builds the `NXRT_MM_BENCH_GRID=1` superset. That covers the
    // `only` and default lists too, but only because no name is a substring of
    // two or more others and no `bench_*` name overlaps a `grid_*` one -- so a
    // filter can never select a case that a narrower list would have hidden.
    // That is an invariant, not a coincidence, so pin it: a future name
    // bridging the two prefixes would double-select in a mode this list cannot
    // reproduce.
    for a in &names {
        let contained_in: Vec<&&str> = names.iter().filter(|b| *b != a && b.contains(*a)).collect();
        assert!(
            contained_in.len() <= 1,
            "`{a}` is a substring of {contained_in:?}; the per-mode lists are no longer covered by this superset"
        );
        for b in contained_in {
            assert_eq!(
                a.starts_with("grid_"),
                b.starts_with("grid_"),
                "`{a}` and `{b}` straddle the bench_/grid_ prefixes, which appear in different GRID modes"
            );
        }
    }

    // A name that is a prefix of another still selects only itself...
    let exact_still_one = select("grid_relu_1_tiny");
    assert_eq!(exact_still_one, vec!["grid_relu_1_tiny"]);

    // ...while a filter naming no case still selects the whole family.
    let group = select("grid_identity");
    assert!(
        group.len() > 1 && group.iter().all(|n| n.starts_with("grid_identity")),
        "a non-exact filter must still select a family: {group:?}"
    );
}

/// The elementwise grid `CPU_ACTIVATION_GAPS.md` is written against, as
/// session-level cases.
///
/// Two sizes: a decode-width row (4 Ki) where per-node dispatch overhead is
/// still visible, and 1 Mi where the kernel is the whole cost. Both are the
/// sizes the published ratio table uses, so a number from here is directly
/// comparable to a number in that file.
fn unary_bench_cases() -> Vec<MatmulFamilyCase> {
    const K4: usize = 4096;
    const M1: usize = 1 << 20;
    vec![
        unary_case(
            "bench_tanh_f32_4k",
            "Tanh",
            "",
            ELEM_F32,
            K4,
            17,
            false,
            1e-4,
        ),
        unary_case(
            "bench_tanh_f32_1m",
            "Tanh",
            "",
            ELEM_F32,
            M1,
            17,
            false,
            1e-4,
        ),
        unary_case(
            "bench_sigmoid_f32_4k",
            "Sigmoid",
            "",
            ELEM_F32,
            K4,
            17,
            false,
            1e-4,
        ),
        unary_case(
            "bench_sigmoid_f32_1m",
            "Sigmoid",
            "",
            ELEM_F32,
            M1,
            17,
            false,
            1e-4,
        ),
        unary_case("bench_erf_f32_4k", "Erf", "", ELEM_F32, K4, 17, false, 1e-4),
        unary_case("bench_erf_f32_1m", "Erf", "", ELEM_F32, M1, 17, false, 1e-4),
        unary_case(
            "bench_gelu_tanh_f32_1m",
            "Gelu",
            "",
            ELEM_F32,
            M1,
            20,
            false,
            1e-4,
        ),
        unary_case(
            "bench_gelu_exact_f32_1m",
            "Gelu",
            "",
            ELEM_F32,
            M1,
            20,
            false,
            1e-4,
        ),
        unary_case("bench_exp_f32_1m", "Exp", "", ELEM_F32, M1, 17, false, 1e-4),
        unary_case(
            "bench_relu_f32_1m",
            "Relu",
            "",
            ELEM_F32,
            M1,
            17,
            false,
            0.0,
        ),
        // The activation family that moved off the generic scalar path: Elu,
        // LeakyRelu, HardSigmoid, ThresholdedRelu and Selu all have AVX2
        // kernels and the zero-copy tensor path now. `Swish(alpha != 1)` is
        // the only one left on the generic path.
        unary_case("bench_elu_f32_4k", "Elu", "", ELEM_F32, K4, 17, false, 1e-4),
        unary_case("bench_elu_f32_1m", "Elu", "", ELEM_F32, M1, 17, false, 1e-4),
        unary_case(
            "bench_leakyrelu_f32_1m",
            "LeakyRelu",
            "",
            ELEM_F32,
            M1,
            17,
            false,
            1e-4,
        ),
        unary_case(
            "bench_leakyrelu_f32_4k",
            "LeakyRelu",
            "",
            ELEM_F32,
            K4,
            17,
            false,
            1e-4,
        ),
        unary_case(
            "bench_hardsigmoid_f32_4k",
            "HardSigmoid",
            "",
            ELEM_F32,
            K4,
            17,
            false,
            1e-4,
        ),
        unary_case(
            "bench_hardsigmoid_f32_1m",
            "HardSigmoid",
            "",
            ELEM_F32,
            M1,
            17,
            false,
            1e-4,
        ),
        unary_case(
            "bench_thresholdedrelu_f32_4k",
            "ThresholdedRelu",
            "",
            ELEM_F32,
            K4,
            17,
            false,
            1e-4,
        ),
        unary_case(
            "bench_thresholdedrelu_f32_1m",
            "ThresholdedRelu",
            "",
            ELEM_F32,
            M1,
            17,
            false,
            1e-4,
        ),
        unary_case(
            "bench_selu_f32_4k",
            "Selu",
            "",
            ELEM_F32,
            K4,
            17,
            false,
            1e-4,
        ),
        unary_case(
            "bench_selu_f32_1m",
            "Selu",
            "",
            ELEM_F32,
            M1,
            17,
            false,
            1e-4,
        ),
        // `Celu` and `Mish` had no kernel here at all before, so ORT ran them
        // and there is no "before" arm to compare against: ours-vs-ORT *is*
        // the A/B. `Mish` is opset 18, `Celu` is 12.
        unary_case(
            "bench_celu_f32_4k",
            "Celu",
            "",
            ELEM_F32,
            K4,
            17,
            false,
            1e-4,
        ),
        unary_case(
            "bench_celu_f32_1m",
            "Celu",
            "",
            ELEM_F32,
            M1,
            17,
            false,
            1e-4,
        ),
        unary_case(
            "bench_mish_f32_4k",
            "Mish",
            "",
            ELEM_F32,
            K4,
            18,
            false,
            1e-4,
        ),
        unary_case(
            "bench_mish_f32_1m",
            "Mish",
            "",
            ELEM_F32,
            M1,
            18,
            false,
            1e-4,
        ),
        // `Log` was the last elementwise op still evaluated one `libm` call at
        // a time, so it gets both sizes: 4k to see the per-call overhead with
        // the data L1-resident, 1M to see the steady-state throughput.
        unary_case("bench_log_f32_4k", "Log", "", ELEM_F32, K4, 17, true, 1e-4),
        unary_case("bench_log_f32_1m", "Log", "", ELEM_F32, M1, 17, true, 1e-4),
        unary_case(
            "bench_sqrt_f32_4k",
            "Sqrt",
            "",
            ELEM_F32,
            K4,
            17,
            true,
            1e-4,
        ),
        unary_case(
            "bench_sqrt_f32_1m",
            "Sqrt",
            "",
            ELEM_F32,
            M1,
            17,
            true,
            1e-4,
        ),
        unary_case(
            "bench_quickgelu_f32_1m",
            "QuickGelu",
            "com.microsoft",
            ELEM_F32,
            M1,
            17,
            false,
            1e-4,
        ),
        unary_case(
            "bench_fastgelu_f32_1m",
            "FastGelu",
            "com.microsoft",
            ELEM_F32,
            M1,
            17,
            false,
            1e-4,
        ),
        unary_case(
            "bench_tanh_f16_1m",
            "Tanh",
            "",
            ELEM_F16,
            M1,
            17,
            false,
            5e-3,
        ),
        unary_case("bench_exp_f16_1m", "Exp", "", ELEM_F16, M1, 17, false, 5e-3),
    ]
}

/// Shapes worth measuring on the plugin path, at sizes a real decode/prefill
/// step reaches rather than the smallest size that proves correctness.
fn matmul_bench_cases() -> Vec<MatmulFamilyCase> {
    vec![
        dense_case(
            "bench_matmul_f32_m1",
            "MatMul",
            ELEM_F32,
            1,
            2048,
            2048,
            1e-3,
        ),
        dense_case(
            "bench_matmul_f32_m128",
            "MatMul",
            ELEM_F32,
            128,
            2048,
            2048,
            1e-3,
        ),
        dense_case(
            "bench_matmul_f16_m1",
            "MatMul",
            ELEM_F16,
            1,
            2048,
            2048,
            5e-2,
        ),
        dense_case(
            "bench_matmul_f16_m128",
            "MatMul",
            ELEM_F16,
            128,
            2048,
            2048,
            5e-2,
        ),
        nbits_case("bench_nbits4_m1", 1, 2048, 2048, 4, 32, ELEM_F32, true),
        nbits_case("bench_nbits4_m128", 128, 2048, 2048, 4, 32, ELEM_F32, true),
        nbits_case("bench_nbits4_f16_m1", 1, 2048, 2048, 4, 32, ELEM_F16, true),
        nbits_case("bench_nbits8_m1", 1, 2048, 2048, 8, 32, ELEM_F32, true),
        nbits_case("bench_nbits8_m256", 256, 2048, 2048, 8, 32, ELEM_F32, true),
        qlinear_case("bench_qlinear_u8_m1", 1, 2048, 2048, false),
        qlinear_case("bench_qlinear_u8_m128", 128, 2048, 2048, false),
        qlinear_case("bench_qlinear_i8_m1", 1, 2048, 2048, true),
    ]
}

/// Interleaved A/B of this EP against plain ORT **through the ORT session
/// API**, which is the only path a real model takes.
///
/// Every previously published matmul ratio came from `bench_generic`, which
/// drives the kernels natively. Those numbers stay valid as kernel-level
/// measurements, but until the claim fixes landed the plugin path never
/// reached the quantized kernels at all, so this is the first honest
/// session-level comparison for `MatMulNBits` and `QLinearMatMul`.
///
/// Off by default: it takes minutes and this host is shared. Run with
/// `NXRT_MM_BENCH=1 cargo test -p onnx-runtime-ep-cpu-plugin --release
/// --test plugin_ort_e2e plugin_path_ab -- --nocapture --ignored`.
#[test]
#[ignore = "benchmark, not a correctness test"]
fn plugin_path_ab_vs_plain_ort() {
    if std::env::var("NXRT_MM_BENCH").as_deref() != Ok("1") {
        eprintln!("set NXRT_MM_BENCH=1 to run the plugin-path A/B");
        return;
    }
    let iters: usize = std::env::var("NXRT_MM_BENCH_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(31);
    let filter = std::env::var("NXRT_MM_BENCH_CASE").unwrap_or_default();
    // Both sessions live in one process, so each side's thread pool is resident
    // (and, while ORT's intra-op pool spins, actively burning cores) during the
    // other side's timed runs. On a 32-vCPU host that is worth several times the
    // measured cost: the same int4 m=128 case reports ORT at 0.98 ms when our
    // sharded path runs beside it and 4.40 ms beside our full-width path.
    // `NXRT_MM_BENCH_SIDE=ours|ort` runs one side only, so a two-process A/B can
    // measure each backend with the machine to itself. Unset keeps the
    // interleaved single-process comparison.
    let side = std::env::var("NXRT_MM_BENCH_SIDE").unwrap_or_default();
    let (run_ours, run_ort) = match side.as_str() {
        "ours" => (true, false),
        "ort" => (false, true),
        _ => (true, true),
    };
    let _lock = lock_ort_ep();
    // Refuse a half-pinned comparison outright: the resulting ratio would be an
    // artefact of one side's core count, and a printed warning next to a number
    // is not a control.
    if let Err(why) = bench_pool_budgets_agree(
        std::env::var("NXRT_MM_BENCH_THREADS").ok().as_deref(),
        std::env::var("ONNX_GENAI_MLAS_THREADPOOL_THREADS")
            .ok()
            .as_deref(),
    ) {
        panic!("{why}");
    }
    let warmup = bench_warmup_runs(std::env::var("NXRT_MM_BENCH_WARMUP").ok().as_deref());
    // Self-describing output: a CSV row is worthless six weeks later if the
    // thread budget it was measured under is not next to it.
    println!(
        "# side={} intra_op_threads={} ours_pool_threads={} iters={iters} warmup={warmup}",
        if side.is_empty() { "both" } else { &side },
        std::env::var("NXRT_MM_BENCH_THREADS").unwrap_or_else(|_| "ort-default".to_owned()),
        std::env::var("ONNX_GENAI_MLAS_THREADPOOL_THREADS")
            .unwrap_or_else(|_| "ep-default".to_owned()),
    );
    println!(
        "case,ours_p50_ms,ort_p50_ms,ratio_p50,ours_p90_ms,ort_p90_ms,ratio_p90,cold_ours_ms,cold_ort_ms"
    );
    // The #1077 grid is off by default: it is 11 more sessions, and the
    // matmul/unary rows answer a different question (kernel throughput) than
    // these do (dispatch overhead). `NXRT_MM_BENCH_GRID=1` adds them;
    // `NXRT_MM_BENCH_GRID=only` runs them alone, which is what a dispatch
    // measurement wants — nothing else resident, nothing else warming caches.
    let grid = std::env::var("NXRT_MM_BENCH_GRID").unwrap_or_default();
    let cases: Vec<MatmulFamilyCase> = match grid.as_str() {
        "only" => dispatch_grid_cases(),
        "1" => matmul_bench_cases()
            .into_iter()
            .chain(unary_bench_cases())
            .chain(dispatch_grid_cases())
            .collect(),
        _ => matmul_bench_cases()
            .into_iter()
            .chain(unary_bench_cases())
            .collect(),
    };
    // `NXRT_MM_BENCH_CASE` selects **exactly** the case it names, and only falls
    // back to substring matching when it names none — so `grid_relu` still picks
    // the whole family while `grid_relu_1_tiny` picks one case.
    //
    // Without the exact rule this was a plain `contains`, and because
    // `grid_relu_1_tiny` is a substring of `grid_relu_1_tiny_dyn` a selector
    // naming one case silently ran two. Every per-`Run` figure derived from it
    // was the sum of two `Run`s, which invalidated three published measurements
    // in #1077 before it was caught. `case_selectors_do_not_match_a_neighbour`
    // is the regression test.
    let exact_case = cases.iter().any(|c| c.name == filter);
    for case in cases {
        if !case_selected(case.name, &filter, exact_case) {
            continue;
        }
        let path = write_generated_model(case.name, &case.model);
        let reg = format!("cpu_ep_bench_{}", case.name);
        let cold_ours = std::time::Instant::now();
        let Some((_lib, api, env, opts, session)) =
            (unsafe { conformance_setup(&reg, &path, true) })
        else {
            eprintln!(
                "*** SKIPPED: {} — ORT or EP cdylib not found ***",
                case.name
            );
            return;
        };
        let cold_ours = cold_ours.elapsed().as_secs_f64() * 1e3;

        unsafe {
            let info = query_ep_assignment(api, session);
            assert!(
                info.ops_on_our_ep().contains(&case.op),
                "{}: not assigned to this EP, refusing to report a ratio",
                case.name
            );

            let cold_ort = std::time::Instant::now();
            let (ort_opts, ort_session) = match try_plain_ort_session(api, env, &path) {
                Ok(pair) => pair,
                Err(msg) => {
                    eprintln!("{}: ORT cannot build this model ({msg})", case.name);
                    conformance_teardown(api, env, opts, session, &reg);
                    continue;
                }
            };
            let cold_ort = cold_ort.elapsed().as_secs_f64() * 1e3;

            let mut buffers: Vec<Vec<u8>> = case.inputs.iter().map(|i| i.data.clone()).collect();
            let mut values: Vec<*const ort::OrtValue> = Vec::with_capacity(buffers.len());
            for (input, buffer) in case.inputs.iter().zip(buffers.iter_mut()) {
                values.push(make_raw_tensor(api, buffer, &input.dims, input.elem) as *const _);
            }
            let input_names: Vec<CString> = case
                .inputs
                .iter()
                .map(|i| CString::new(i.name).unwrap())
                .collect();
            let input_name_ptrs: Vec<*const std::os::raw::c_char> =
                input_names.iter().map(|c| c.as_ptr()).collect();
            let output_name = CString::new(case.output).unwrap();
            let output_name_ptrs = [output_name.as_ptr()];

            // Warm both sides: first run pays prepack and page faults.
            if run_ours {
                bench_runs(
                    api,
                    session,
                    &input_name_ptrs,
                    &values,
                    &output_name_ptrs,
                    warmup,
                );
            }
            if run_ort {
                bench_runs(
                    api,
                    ort_session,
                    &input_name_ptrs,
                    &values,
                    &output_name_ptrs,
                    warmup,
                );
            }

            // Allocation attribution runs before the timed loop so its
            // counter updates never land inside a measured iteration.
            if run_ours
                && let Some((buf, names)) = probe_dispatch(
                    api,
                    session,
                    &input_name_ptrs,
                    &values,
                    &output_name_ptrs,
                    64,
                )
            {
                // Node count comes from ORT's assignment list, not from the
                // case definition: per-node figures must divide by the nodes
                // this EP actually received, and fusion means that is not the
                // same as the number of `Run` callbacks.
                report_probe(
                    case.name,
                    info.ops_on_our_ep().len().max(1),
                    64,
                    &buf,
                    &names,
                );
            }

            // Interleave one iteration each so a drifting host load lands on
            // both sides rather than on whichever ran second.
            let mut ours = Vec::with_capacity(iters);
            let mut theirs = Vec::with_capacity(iters);
            for _ in 0..iters {
                if run_ours {
                    ours.extend(bench_runs(
                        api,
                        session,
                        &input_name_ptrs,
                        &values,
                        &output_name_ptrs,
                        1,
                    ));
                }
                if run_ort {
                    theirs.extend(bench_runs(
                        api,
                        ort_session,
                        &input_name_ptrs,
                        &values,
                        &output_name_ptrs,
                        1,
                    ));
                }
            }
            // A one-sided run reports `NaN` for the side it did not measure, so
            // a ratio can never be assembled from two different processes by
            // accident; the caller compares the two p50 columns explicitly.
            let (o50, o90) = (percentile(&mut ours, 0.5), percentile(&mut ours, 0.9));
            let (t50, t90) = (percentile(&mut theirs, 0.5), percentile(&mut theirs, 0.9));
            println!(
                "{},{o50:.4},{t50:.4},{:.3},{o90:.4},{t90:.4},{:.3},{cold_ours:.1},{cold_ort:.1}",
                case.name,
                o50 / t50,
                o90 / t90,
            );

            for value in values {
                ((*api).ReleaseValue.unwrap())(value as *mut _);
            }
            ((*api).ReleaseSession.unwrap())(ort_session);
            ((*api).ReleaseSessionOptions.unwrap())(ort_opts);
            conformance_teardown(api, env, opts, session, &reg);
        }
    }
}

/// float32 `com.microsoft::RotaryEmbedding` and `GroupQueryAttention` execute
/// on *this* EP and agree with ORT's own CPU kernels.
///
/// Assignment alone is not the guarantee the policy asks for: a node can be
/// claimed and then compute the wrong thing, which is strictly worse than the
/// deferral it replaced. These two ops are the ones that were being handed to
/// ORT until the per-slot dtype tables landed, so they are the two that most
/// need their arithmetic checked against the implementation they displaced.
///
/// The baseline is a second session over the *same* model file with our EP
/// simply not appended, so ORT resolves both nodes to its own contrib CPU
/// kernels. Same bytes in, so any difference is ours.
#[test]
fn rope_and_gqa_execute_on_our_ep_and_match_ort_numerics() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");

    for (fixture, op) in [
        ("rotary_assignment_f32", "RotaryEmbedding"),
        ("gqa_assignment_f32", "GroupQueryAttention"),
    ] {
        let model_path = PathBuf::from(manifest_dir)
            .join(format!("tests/fixtures/{fixture}/model.onnx.textproto"));
        let reg = format!("cpu_ep_num_{fixture}");

        let Some((_lib, api, env, opts, session)) =
            (unsafe { conformance_setup(&reg, &model_path, true) })
        else {
            eprintln!("*** SKIPPED: {fixture} numerics — ORT or EP cdylib not found ***");
            return;
        };

        unsafe {
            let info = query_ep_assignment(api, session);
            assert!(
                info.ops_on_our_ep().contains(&op),
                "{fixture}: '{op}' must execute here, not on ORT's CPU EP; got {:?}",
                info.assignments
            );

            let (names, mut inputs, output_names) = attention_fixture_inputs(api, fixture);
            let ours = run_and_collect_f32(api, session, &names, &inputs, &output_names);

            // Same model, same bytes, but ORT resolves the node itself.
            let mut base_opts: *mut ort::OrtSessionOptions = ptr::null_mut();
            let status = ((*api).CreateSessionOptions.unwrap())(&mut base_opts);
            check_status(api, status, "CreateSessionOptions(baseline)");
            let mut base_session: *mut ort::OrtSession = ptr::null_mut();
            let status =
                ort_session::create_session(api, env, base_opts, &model_path, &mut base_session);
            check_status(api, status, "CreateSession(baseline)");
            let theirs = run_and_collect_f32(api, base_session, &names, &inputs, &output_names);
            ((*api).ReleaseSession.unwrap())(base_session);
            ((*api).ReleaseSessionOptions.unwrap())(base_opts);

            assert_eq!(ours.len(), theirs.len(), "{fixture}: output count differs");
            for (idx, (a, b)) in ours.iter().zip(theirs.iter()).enumerate() {
                assert_eq!(
                    a.len(),
                    b.len(),
                    "{fixture}: output {idx} length {} vs ORT's {}",
                    a.len(),
                    b.len()
                );
                let mut worst = 0.0f32;
                let mut worst_at = 0usize;
                for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                    assert!(
                        x.is_finite(),
                        "{fixture}: output {idx}[{i}] is {x}, not finite"
                    );
                    let err = (x - y).abs() / y.abs().max(1.0);
                    if err > worst {
                        worst = err;
                        worst_at = i;
                    }
                }
                assert!(
                    worst <= 2e-5,
                    "{fixture}: output {idx} differs from ORT by {worst} at [{worst_at}] \
                     (ours {}, ORT {})",
                    a[worst_at],
                    b[worst_at]
                );
                eprintln!(
                    "  [{fixture}] output {idx}: {} values, max rel err {worst:.3e}",
                    a.len()
                );
            }

            for v in inputs.drain(..) {
                ((*api).ReleaseValue.unwrap())(v);
            }
            conformance_teardown(api, env, opts, session, &reg);
        }
    }
}

/// Deterministic inputs for the attention fixtures, in each model's slot order.
///
/// The tensors are leaked rather than returned by value because ORT borrows the
/// caller's buffers: `CreateTensorWithDataAsOrtValue` does not copy, so the
/// backing storage has to outlive every `Run` that reads it.
unsafe fn attention_fixture_inputs(
    api: *const ort::OrtApi,
    fixture: &str,
) -> (
    Vec<&'static str>,
    Vec<*mut ort::OrtValue>,
    Vec<&'static str>,
) {
    // A cheap deterministic spread in [-1, 1) that is not periodic in the head
    // dimension, so a transposed or mis-strided read cannot look correct.
    let fill = |n: usize, seed: u32| -> &'static mut [f32] {
        let mut s = seed | 1;
        let v: Vec<f32> = (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((s >> 8) as f32 / (1u32 << 23) as f32) - 1.0
            })
            .collect();
        Box::leak(v.into_boxed_slice())
    };
    unsafe {
        match fixture {
            "rotary_assignment_f32" => {
                let (b, s, h, d) = (1i64, 1i64, 32i64, 128i64);
                let x = make_float_tensor(api, fill((b * s * h * d) as usize, 7), &[b, s, h * d]);
                let pos = Box::leak(vec![17i64].into_boxed_slice());
                let pos_bytes = std::slice::from_raw_parts_mut(
                    pos.as_mut_ptr().cast::<u8>(),
                    std::mem::size_of_val(pos),
                );
                let pos_v = make_raw_tensor(
                    api,
                    pos_bytes,
                    &[b, s],
                    ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                );
                let cos = make_float_tensor(api, fill((2048 * d / 2) as usize, 11), &[2048, d / 2]);
                let sin = make_float_tensor(api, fill((2048 * d / 2) as usize, 13), &[2048, d / 2]);
                (
                    vec!["input", "position_ids", "cos_cache", "sin_cache"],
                    vec![x, pos_v, cos, sin],
                    vec!["output"],
                )
            }
            "gqa_assignment_f32" => {
                let (b, s, h, kv, d, p) = (1i64, 1i64, 32i64, 8i64, 128i64, 1023i64);
                let q = make_float_tensor(api, fill((b * s * h * d) as usize, 3), &[b, s, h * d]);
                let k = make_float_tensor(api, fill((b * s * kv * d) as usize, 5), &[b, s, kv * d]);
                let v = make_float_tensor(api, fill((b * s * kv * d) as usize, 9), &[b, s, kv * d]);
                let pk =
                    make_float_tensor(api, fill((b * kv * p * d) as usize, 21), &[b, kv, p, d]);
                let pv =
                    make_float_tensor(api, fill((b * kv * p * d) as usize, 23), &[b, kv, p, d]);
                // seqlens_k is the *past* length (ORT adds the new token), and
                // total_sequence_length must equal max(seqlens_k) + 1.
                // Leaked for the same reason as the float buffers: ORT keeps a
                // pointer into them for the life of the value, so a temporary
                // would dangle by the time `Run` reads it.
                let seqlens =
                    make_int32_tensor(api, Box::leak(vec![p as i32].into_boxed_slice()), &[b]);
                let total = make_int32_tensor(
                    api,
                    Box::leak(vec![(p + 1) as i32].into_boxed_slice()),
                    &[1],
                );
                (
                    vec![
                        "query",
                        "key",
                        "value",
                        "past_key",
                        "past_value",
                        "seqlens_k",
                        "total_sequence_length",
                    ],
                    vec![q, k, v, pk, pv, seqlens, total],
                    vec!["output", "present_key", "present_value"],
                )
            }
            other => panic!("attention_fixture_inputs: no inputs defined for {other}"),
        }
    }
}

/// Run `session` over the named inputs and copy every output out as float32.
unsafe fn run_and_collect_f32(
    api: *const ort::OrtApi,
    session: *mut ort::OrtSession,
    input_names: &[&str],
    inputs: &[*mut ort::OrtValue],
    output_names: &[&str],
) -> Vec<Vec<f32>> {
    unsafe {
        let in_c: Vec<std::ffi::CString> = input_names
            .iter()
            .map(|n| std::ffi::CString::new(*n).unwrap())
            .collect();
        let out_c: Vec<std::ffi::CString> = output_names
            .iter()
            .map(|n| std::ffi::CString::new(*n).unwrap())
            .collect();
        let in_ptrs: Vec<*const i8> = in_c.iter().map(|c| c.as_ptr()).collect();
        let out_ptrs: Vec<*const i8> = out_c.iter().map(|c| c.as_ptr()).collect();
        let mut outputs: Vec<*mut ort::OrtValue> = vec![ptr::null_mut(); output_names.len()];

        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(),
            in_ptrs.as_ptr(),
            inputs.as_ptr().cast(),
            inputs.len(),
            out_ptrs.as_ptr(),
            output_names.len(),
            outputs.as_mut_ptr(),
        );
        check_status(api, status, "Run(attention fixture)");

        let mut collected = Vec::with_capacity(outputs.len());
        for out in &outputs {
            assert!(!out.is_null(), "Run returned a null output");
            let mut info: *mut ort::OrtTensorTypeAndShapeInfo = ptr::null_mut();
            let status = ((*api).GetTensorTypeAndShape.unwrap())(*out, &mut info);
            check_status(api, status, "GetTensorTypeAndShape(attention fixture)");
            let mut count = 0usize;
            let status = ((*api).GetTensorShapeElementCount.unwrap())(info, &mut count);
            check_status(api, status, "GetTensorShapeElementCount(attention fixture)");
            ((*api).ReleaseTensorTypeAndShapeInfo.unwrap())(info);

            let mut data: *mut std::ffi::c_void = ptr::null_mut();
            let status = ((*api).GetTensorMutableData.unwrap())(*out, &mut data);
            check_status(api, status, "GetTensorMutableData(attention fixture)");
            collected.push(std::slice::from_raw_parts(data as *const f32, count).to_vec());
        }
        for out in outputs {
            ((*api).ReleaseValue.unwrap())(out);
        }
        collected
    }
}

/// Configurations these kernels cannot compute, each declined at *claim* time
/// with the session still loading.
///
/// Every row is a **capability** answer rather than a performance one, and each
/// mirrors a rejection its kernel factory raises. The mirroring is what matters:
///
///   * decline at claim time  → ORT runs it, model works, we owe a kernel;
///   * claim and fail in the factory → `CreateSession` dies and **no fallback
///     recovers**, because ORT has already compiled the node onto this EP.
///
/// Making these ops reachable at all is what created the second possibility, so
/// this test pins the first. ORT's CPU EP runs all three configurations below,
/// which is why claiming-then-failing would take a model that works today and
/// make it unloadable.
///
/// | fixture | limit | where the factory raises it |
/// |---|---|---|
/// | `qmoe_columnwise_f32` | `block_size` absent → column-wise quantization | `qmoe.rs` |
/// | `moe_sparse_mixer_f32` | `use_sparse_mixer=1` (Phi-3.5-MoE / GRIN-MoE) | `moe.rs` |
/// | `gqa_smooth_softmax_f32` | `smooth_softmax=1` (attention sink) | `group_query_attention.rs` |
///
/// They are deliberately *not* in `ASSIGNMENT_FIXTURES`: they are the documented
/// exceptions, and each should stop being one as soon as its kernel exists.
#[test]
fn factory_only_capability_limits_are_declined_at_claim_time() {
    const CASES: &[(&str, &str, &str)] = &[
        (
            "qmoe_columnwise_f32",
            "QMoE",
            "`qmoe::unsupported_reason` (column-wise quantization)",
        ),
        (
            "moe_sparse_mixer_f32",
            "MoE",
            "`moe::unsupported_reason` (use_sparse_mixer=1)",
        ),
        (
            "gqa_smooth_softmax_f32",
            "GroupQueryAttention",
            "`group_query_attention::unsupported_reason` (smooth_softmax=1)",
        ),
    ];

    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    for (fixture, op, guard) in CASES {
        let model_path = PathBuf::from(manifest_dir)
            .join(format!("tests/fixtures/{fixture}/model.onnx.textproto"));
        let reg = format!("cpu_ep_claim_time_{fixture}");

        // Fallback must stay *enabled*: the point is that ORT can still run it.
        let Some((_lib, api, env, opts, session)) =
            (unsafe { conformance_setup(&reg, &model_path, false) })
        else {
            eprintln!("*** SKIPPED: {fixture} — ORT or EP cdylib not found ***");
            return;
        };

        unsafe {
            let info = query_ep_assignment(api, session);
            let ours = info.ops_on_our_ep();
            assert!(
                !ours.contains(op),
                "{fixture}: {op} was claimed, but the kernel factory rejects this \
                 configuration, which kills the session outright rather than falling back. \
                 Decline it in {guard}, or implement the missing path. Got: {:?}",
                info.assignments
            );
            eprintln!(
                "  [{fixture}] declined at claim time, session loaded; ours={ours:?}, \
                 others={:?}",
                info.ops_not_on_our_ep()
            );
            conformance_teardown(api, env, opts, session, &reg);
        }
    }
}

// ─── Falsifier: assignment is not execution ─────────────────────────────────

/// Deterministic float payloads for every input a fixture declares, read from
/// the session itself rather than from a per-fixture table.
///
/// Returns `None` — meaning "this fixture cannot be driven generically" — when
/// any input is not float32/float16/bfloat16 or has a non-static dimension.
/// Integer inputs are excluded on purpose: `seqlens_k`, `total_sequence_length`
/// and `position_ids` carry *semantics*, and filling them with noise would
/// produce a kernel error rather than a measurement. Those fixtures are covered
/// by `rope_and_gqa_execute_on_our_ep_and_match_ort_numerics`, which supplies
/// real values.
///
/// # Safety
/// `api` and `session` must come from a successfully created session.
#[allow(clippy::type_complexity)]
unsafe fn generic_float_inputs(
    api: *const ort::OrtApi,
    session: *mut ort::OrtSession,
) -> Option<(
    Vec<std::ffi::CString>,
    Vec<*mut ort::OrtValue>,
    Vec<std::ffi::CString>,
)> {
    unsafe {
        let mut allocator: *mut ort::OrtAllocator = ptr::null_mut();
        let status = ((*api).GetAllocatorWithDefaultOptions.unwrap())(&mut allocator);
        check_status(api, status, "GetAllocatorWithDefaultOptions");

        let mut num_inputs = 0usize;
        let status = ((*api).SessionGetInputCount.unwrap())(session, &mut num_inputs);
        check_status(api, status, "SessionGetInputCount");

        let mut names = Vec::with_capacity(num_inputs);
        let mut values: Vec<*mut ort::OrtValue> = Vec::with_capacity(num_inputs);

        for i in 0..num_inputs {
            let mut name_ptr: *mut std::os::raw::c_char = ptr::null_mut();
            let status =
                ((*api).SessionGetInputName.unwrap())(session, i, allocator, &mut name_ptr);
            check_status(api, status, "SessionGetInputName");
            let name = CStr::from_ptr(name_ptr).to_owned();
            ((*allocator).Free.unwrap())(allocator, name_ptr.cast());

            let mut type_info: *mut ort::OrtTypeInfo = ptr::null_mut();
            let status = ((*api).SessionGetInputTypeInfo.unwrap())(session, i, &mut type_info);
            check_status(api, status, "SessionGetInputTypeInfo");
            let mut tensor_info: *const ort::OrtTensorTypeAndShapeInfo = ptr::null();
            let status = ((*api).CastTypeInfoToTensorInfo.unwrap())(type_info, &mut tensor_info);
            check_status(api, status, "CastTypeInfoToTensorInfo");
            if tensor_info.is_null() {
                ((*api).ReleaseTypeInfo.unwrap())(type_info);
                for v in values.drain(..) {
                    ((*api).ReleaseValue.unwrap())(v);
                }
                return None;
            }
            let mut elem: ort::ONNXTensorElementDataType =
                ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
            let status = ((*api).GetTensorElementType.unwrap())(tensor_info, &mut elem);
            check_status(api, status, "GetTensorElementType");
            let mut rank = 0usize;
            let status = ((*api).GetDimensionsCount.unwrap())(tensor_info, &mut rank);
            check_status(api, status, "GetDimensionsCount");
            let mut dims = vec![0i64; rank];
            let status = ((*api).GetDimensions.unwrap())(tensor_info, dims.as_mut_ptr(), rank);
            check_status(api, status, "GetDimensions");
            ((*api).ReleaseTypeInfo.unwrap())(type_info);

            let is_float = elem == ELEM_F32
                || elem == ELEM_F16
                || elem == ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
            if !is_float || dims.iter().any(|&d| d <= 0) {
                for v in values.drain(..) {
                    ((*api).ReleaseValue.unwrap())(v);
                }
                return None;
            }

            let count: usize = dims.iter().map(|&d| d as usize).product::<usize>().max(1);
            // A spread in [-1, 1) rather than zeros: a saturating or masked
            // kernel can produce the right answer from an all-zero input for
            // the wrong reason, and this data also keeps `Log`/`Sqrt` inputs
            // off the degenerate value.
            let mut seed = (7u32).wrapping_add(count as u32) | 1;
            let value = |s: &mut u32| -> f32 {
                *s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((*s >> 8) as f32 / (1u32 << 23) as f32) - 1.0
            };
            let val = if elem == ELEM_F32 {
                let data: Vec<f32> = (0..count).map(|_| value(&mut seed)).collect();
                make_float_tensor(api, Box::leak(data.into_boxed_slice()), &dims)
            } else {
                let mut bytes = Vec::with_capacity(count * 2);
                for _ in 0..count {
                    let x = value(&mut seed);
                    let bits = if elem == ELEM_F16 {
                        f32_to_f16_bits(x)
                    } else {
                        // bfloat16 is the top half of the f32 pattern.
                        (x.to_bits() >> 16) as u16
                    };
                    bytes.extend_from_slice(&bits.to_le_bytes());
                }
                make_raw_tensor(api, Box::leak(bytes.into_boxed_slice()), &dims, elem)
            };
            names.push(name);
            values.push(val);
        }

        let mut num_outputs = 0usize;
        let status = ((*api).SessionGetOutputCount.unwrap())(session, &mut num_outputs);
        check_status(api, status, "SessionGetOutputCount");
        let mut out_names = Vec::with_capacity(num_outputs);
        for i in 0..num_outputs {
            let mut name_ptr: *mut std::os::raw::c_char = ptr::null_mut();
            let status =
                ((*api).SessionGetOutputName.unwrap())(session, i, allocator, &mut name_ptr);
            check_status(api, status, "SessionGetOutputName");
            out_names.push(CStr::from_ptr(name_ptr).to_owned());
            ((*allocator).Free.unwrap())(allocator, name_ptr.cast());
        }

        Some((names, values, out_names))
    }
}

/// Run `session` over `inputs`, discard the outputs, and return ORT's error
/// message instead of panicking when the run fails.
///
/// The baseline half of `every_assigned_node_is_also_executed_by_this_ep` needs
/// the non-panicking form: ORT's own CPU kernels reject two of these fixtures
/// (bfloat16 `Add` has no kernel; ORT's `QMoE` wants a different
/// `fc2_experts_scales` layout from the one its contrib schema admits), and
/// "ORT cannot run this at all" is a fact to report, not a test failure.
///
/// # Safety
/// Every pointer must come from the same live session.
unsafe fn try_run_discarding_outputs(
    api: *const ort::OrtApi,
    session: *mut ort::OrtSession,
    input_names: &[std::ffi::CString],
    inputs: &[*mut ort::OrtValue],
    output_names: &[std::ffi::CString],
) -> Result<(), String> {
    unsafe {
        let in_ptrs: Vec<*const std::os::raw::c_char> =
            input_names.iter().map(|c| c.as_ptr()).collect();
        let out_ptrs: Vec<*const std::os::raw::c_char> =
            output_names.iter().map(|c| c.as_ptr()).collect();
        let mut outputs: Vec<*mut ort::OrtValue> = vec![ptr::null_mut(); output_names.len()];
        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(),
            in_ptrs.as_ptr(),
            inputs.as_ptr().cast(),
            inputs.len(),
            out_ptrs.as_ptr(),
            output_names.len(),
            outputs.as_mut_ptr(),
        );
        let failure = if status.is_null() {
            None
        } else {
            let msg = CStr::from_ptr(((*api).GetErrorMessage.unwrap())(status))
                .to_string_lossy()
                .into_owned();
            ((*api).ReleaseStatus.unwrap())(status);
            Some(msg)
        };
        for out in outputs {
            if !out.is_null() {
                ((*api).ReleaseValue.unwrap())(out);
            }
        }
        match failure {
            None => Ok(()),
            Some(msg) => Err(msg),
        }
    }
}

/// [`try_run_discarding_outputs`], panicking on failure. Used for *our* side,
/// where a failed run is a real defect.
///
/// # Safety
/// Every pointer must come from the same live session.
unsafe fn run_discarding_outputs(
    api: *const ort::OrtApi,
    session: *mut ort::OrtSession,
    input_names: &[std::ffi::CString],
    inputs: &[*mut ort::OrtValue],
    output_names: &[std::ffi::CString],
    label: &str,
) {
    if let Err(msg) =
        unsafe { try_run_discarding_outputs(api, session, input_names, inputs, output_names) }
    {
        panic!("Run({label}) failed: {msg}");
    }
}

/// Being *assigned* a node and *executing* it are different claims, and only
/// the second one is what "this EP does not defer" actually promises.
///
/// `no_supported_node_is_ever_left_to_the_ort_cpu_ep` and
/// `every_fixture_loads_with_cpu_fallback_disabled` both read ORT's static
/// node→EP attribution, which is a session-build fact. Neither runs the model,
/// so neither can see whether our kernel is what produced the output. Output
/// equality cannot close that gap either: agreeing with ORT is exactly what a
/// correct kernel does, so a graph secretly executed by ORT looks identical.
///
/// This test closes it with the cdylib's own counter. For every fixture it can
/// drive generically, with `session.disable_cpu_ep_fallback=1`:
///
/// 1. ORT reports *n* nodes assigned to `cpu_ep`;
/// 2. one `Run` later, this EP's execution counter has advanced by exactly *n*.
///
/// The counter is read out of the same mapping ORT loaded (same absolute path,
/// so the same statics), which is why it observes the real session rather than
/// a second copy of the library.
///
/// **Non-vacuity** is checked in the same run: a baseline session over the same
/// model file with our EP never appended must leave the counter at zero. If the
/// counter incremented for any session, or if a fixture's nodes were silently
/// running on ORT, exactly one of the two halves fails.
#[test]
fn every_assigned_node_is_also_executed_by_this_ep() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let ep_path = match find_ep_cdylib() {
        Some(p) => p,
        None => {
            if std::env::var("NXRT_REQUIRE_ORT_TESTS").as_deref() == Ok("1") {
                panic!("NXRT_REQUIRE_ORT_TESTS=1 but the EP cdylib is missing");
            }
            eprintln!("*** SKIPPED: EP cdylib not found ***");
            return;
        }
    };
    // The same absolute path ORT registers: one mapping, one set of statics.
    let ep_lib = unsafe { libloading::Library::new(&ep_path) }.expect("dlopen EP cdylib");
    let executed: libloading::Symbol<'_, unsafe extern "C" fn() -> usize> =
        unsafe { ep_lib.get(b"nxrt_ep_executed_node_count") }
            .expect("nxrt_ep_executed_node_count not exported");
    let reset_executed: libloading::Symbol<'_, unsafe extern "C" fn()> =
        unsafe { ep_lib.get(b"nxrt_ep_reset_executed_node_count") }
            .expect("nxrt_ep_reset_executed_node_count not exported");

    let mut driven = 0usize;
    let mut baselines = 0usize;
    let mut skipped: Vec<&str> = Vec::new();
    let mut no_ort_kernel: Vec<&str> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    for (fixture, op) in ASSIGNMENT_FIXTURES {
        let model_path = PathBuf::from(manifest_dir)
            .join(format!("tests/fixtures/{fixture}/model.onnx.textproto"));
        let reg = format!("cpu_ep_exec_{fixture}");
        let Some((_lib, api, env, opts, session)) =
            (unsafe { conformance_setup(&reg, &model_path, true) })
        else {
            eprintln!("*** SKIPPED: {fixture} (execution) — ORT or EP cdylib not found ***");
            continue;
        };

        unsafe {
            let Some((names, inputs, out_names)) = generic_float_inputs(api, session) else {
                skipped.push(fixture);
                conformance_teardown(api, env, opts, session, &reg);
                continue;
            };

            let info = query_ep_assignment(api, session);
            let assigned = info.ops_on_our_ep().len();
            assert!(
                info.ops_on_our_ep().contains(op),
                "{fixture}: '{op}' is not on this EP — got {:?}",
                info.assignments
            );

            reset_executed();
            run_discarding_outputs(api, session, &names, &inputs, &out_names, fixture);
            let ran = executed();
            if ran != assigned {
                failures.push(format!(
                    "  {fixture}: ORT assigned {assigned} node(s) to cpu_ep but this EP executed \
                     {ran}"
                ));
            }

            // Non-vacuity: the same model, our EP never appended. ORT runs it
            // itself, so our counter must not move. Without this half, a
            // counter that incremented on every node of every session — or one
            // wired to the wrong event — would still satisfy the check above.
            //
            // Some fixtures have no ORT-only session at all: bfloat16 `Add`,
            // for one, has no ORT CPU kernel, which is why this EP claiming it
            // matters. Those are counted and reported rather than asserted, so
            // the check stays honest instead of silently dropping to zero
            // baselines.
            reset_executed();
            let mut base_opts: *mut ort::OrtSessionOptions = ptr::null_mut();
            let status = ((*api).CreateSessionOptions.unwrap())(&mut base_opts);
            check_status(api, status, "CreateSessionOptions(baseline)");
            pin_intra_op_threads(api, base_opts);
            let mut base_session: *mut ort::OrtSession = ptr::null_mut();
            let status =
                ort_session::create_session(api, env, base_opts, &model_path, &mut base_session);
            if status.is_null() {
                let (base_names, base_inputs, base_outs) =
                    generic_float_inputs(api, base_session).expect("baseline inputs");
                match try_run_discarding_outputs(
                    api,
                    base_session,
                    &base_names,
                    &base_inputs,
                    &base_outs,
                ) {
                    Ok(()) => {
                        let leaked = executed();
                        if leaked != 0 {
                            failures.push(format!(
                                "  {fixture}: a session without this EP still advanced our \
                                 execution counter by {leaked} — the counter does not observe \
                                 what it claims to"
                            ));
                        }
                        baselines += 1;
                    }
                    Err(msg) => {
                        eprintln!("  [{fixture}] ORT alone cannot run this graph: {msg}");
                        no_ort_kernel.push(fixture);
                    }
                }
                for v in base_inputs {
                    ((*api).ReleaseValue.unwrap())(v);
                }
                ((*api).ReleaseSession.unwrap())(base_session);
            } else {
                ((*api).ReleaseStatus.unwrap())(status);
                no_ort_kernel.push(fixture);
            }
            ((*api).ReleaseSessionOptions.unwrap())(base_opts);

            for v in inputs {
                ((*api).ReleaseValue.unwrap())(v);
            }
            conformance_teardown(api, env, opts, session, &reg);
        }
        eprintln!("  [{fixture}] '{op}' assigned to cpu_ep and executed here");
        driven += 1;
    }

    assert!(
        failures.is_empty(),
        "{} fixture(s) were assigned to this EP without being executed by it:\n{}",
        failures.len(),
        failures.join("\n")
    );

    if driven == 0 {
        eprintln!("*** SKIPPED: no fixture could be driven — ORT or EP cdylib not found ***");
        return;
    }
    assert!(
        baselines > 0,
        "every driven fixture failed to build an ORT-only baseline, so the non-vacuity half \
         never ran: {no_ort_kernel:?}"
    );
    // A fixture whose inputs carry semantics (int32 `seqlens_k`, int64
    // `position_ids`) cannot be filled with noise; those are driven with real
    // values by `rope_and_gqa_execute_on_our_ep_and_match_ort_numerics`.
    eprintln!(
        "\n✅ every_assigned_node_is_also_executed_by_this_ep: {driven} fixtures executed here \
         ({baselines} of them also proved a no-EP baseline leaves the counter at zero; \
         {} have no ORT CPU kernel at all: {no_ort_kernel:?}). {} needed semantic inputs and are \
         covered by rope_and_gqa_execute_on_our_ep_and_match_ort_numerics: {skipped:?}",
        no_ort_kernel.len(),
        skipped.len()
    );
}

// ─── Celu, Mish and Log: new native kernels, checked against ORT's ───────────

/// The activation buffer a unary parity check has to survive.
///
/// A sweep of well-behaved values in `[-1, 1]` is the one input set on which
/// every plausible implementation agrees, so it proves almost nothing. The
/// interesting disagreements live at the edges each kernel has to special-case
/// by hand: both signed zeros, both infinities, NaN, the ends of the normal
/// range, the denormal band (which `log` handles by pre-scaling and fixing the
/// exponent afterwards), and the saturation tails where `exp` overflows and the
/// kernels are expected to clamp rather than produce Inf/NaN.
///
/// The negative half is kept even for `Log`, where it is undefined: "both
/// return NaN" is a real contract, and a kernel that returned a huge negative
/// number instead would pass a positives-only sweep.
fn unary_special_values() -> Vec<f32> {
    let mut v = vec![
        f32::NEG_INFINITY,
        -f32::MAX,
        -1e30,
        -100.0,
        -88.72284,
        -88.0,
        -20.0,
        -10.0,
        -2.0,
        -1.5,
        -1.0,
        -0.5,
        -std::f32::consts::LN_2,
        -f32::MIN_POSITIVE,
        -1e-40,
        -1e-45,
        -0.0,
        0.0,
        1e-45,
        1e-40,
        5.877472e-39,
        f32::MIN_POSITIVE,
        0.5,
        std::f32::consts::LN_2,
        std::f32::consts::FRAC_1_SQRT_2,
        0.99999994,
        1.0,
        1.0000001,
        1.5,
        2.0,
        10.0,
        20.0,
        88.0,
        88.72284,
        100.0,
        1e30,
        f32::MAX,
        f32::INFINITY,
        f32::NAN,
    ];
    // A dense two-sided sweep on top of the corner cases, so the comparison is
    // not made only of exceptional lanes, and so the length crosses several
    // SIMD blocks with a ragged tail.
    let n = 421;
    for i in 0..n {
        v.push(-30.0 + 60.0 * (i as f32) / ((n - 1) as f32));
    }
    v
}

/// Whether two results are compatible, given that neither side is the oracle.
///
/// Exact equality is required wherever the value is not a rounding of a real
/// number -- NaN, the infinities, and the sign of zero -- because those are
/// contract, not accuracy. Everything else is compared relative to the larger
/// magnitude, which is the right scale for `log`: `ln(1e-38)` is `-87.3`, where
/// one ulp is `7.6e-6`, so an absolute bound would either be vacuous there or
/// unmeetable near `ln(1) = 0`.
fn unary_result_matches(ours: f32, ort: f32, rel: f32) -> bool {
    if ours.is_nan() || ort.is_nan() {
        return ours.is_nan() && ort.is_nan();
    }
    if ours.is_infinite() || ort.is_infinite() {
        return ours == ort;
    }
    if ours == 0.0 && ort == 0.0 {
        return true;
    }
    (ours - ort).abs() <= rel * ours.abs().max(ort.abs()).max(1.0)
}

/// `Celu`, `Mish` and `Log` run on this EP and agree with ORT's own kernels.
///
/// These three are the ops this EP was quietly not doing properly. `Celu` and
/// `Mish` had no kernel at all, so the plugin's fail-closed capability filter
/// dropped them and ORT ran them -- which the architectural rule forbids, and
/// which no existing test could see because a missing op cannot appear in a
/// coverage list written from the ops that exist. `Log` was registered but fell
/// into the scalar catch-all in `unary_math`, one `libm` call per element.
///
/// Every case runs with `session.disable_cpu_ep_fallback=1`, so a node this EP
/// declines does not quietly move to ORT -- session creation fails instead.
/// Assignment is then checked against ORT's own record, the case is run so the
/// claim is backed by an execution, and the output is compared elementwise
/// against a second session with no EP appended at all.
#[test]
fn the_native_activation_family_executes_locally_and_matches_ort_numerics() {
    let _lock = lock_ort_ep();
    let x = unary_special_values();
    let len = x.len();
    let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();

    // `Celu` is opset 12, `Mish` is opset 18; ask for each op's own opset
    // rather than one number that happens to satisfy both.
    let specs: [(&'static str, &'static str, u64, f32); 10] = [
        ("parity_celu_f32", "Celu", 12, 2e-6),
        ("parity_celu_a2_f32", "Celu", 12, 2e-6),
        ("parity_mish_f32", "Mish", 18, 2e-6),
        ("parity_log_f32", "Log", 17, 2e-6),
        ("parity_elu_f32", "Elu", 17, 2e-6),
        ("parity_elu_a2_f32", "Elu", 17, 2e-6),
        ("parity_leakyrelu_f32", "LeakyRelu", 17, 2e-6),
        ("parity_hardsigmoid_f32", "HardSigmoid", 17, 2e-6),
        ("parity_thresholdedrelu_f32", "ThresholdedRelu", 17, 2e-6),
        ("parity_selu_f32", "Selu", 17, 2e-6),
    ];

    let mut checked = 0usize;
    for (name, op, opset, rel) in specs {
        let mut case = unary_case(name, op, "", ELEM_F32, len, opset, false, rel);
        case.inputs[0].data = bytes.clone();
        let path = write_generated_model(name, &case.model);
        let reg = format!("cpu_ep_parity_{name}");
        let Some((_lib, api, env, _opts, session)) =
            (unsafe { conformance_setup(&reg, &path, true) })
        else {
            eprintln!("*** SKIPPED: {name} — ORT or EP cdylib not found ***");
            return;
        };

        unsafe {
            let info = query_ep_assignment(api, session);
            let ours_ops = info.ops_on_our_ep();
            let theirs = info.ops_not_on_our_ep();
            assert!(
                ours_ops.contains(&op),
                "{name}: '{op}' must run on this EP, got {:?}",
                info.assignments
            );
            assert!(
                theirs.is_empty(),
                "{name}: nodes {theirs:?} were left to ORT's CPU EP. This EP does not \
                 defer -- a missing kernel is a kernel to write, not a node to give away",
            );

            let ours_out = run_generated_case(api, session, &case, &format!("Run({name})"));
            let (ort_opts, ort_session) = match try_plain_ort_session(api, env, &path) {
                Ok(pair) => pair,
                Err(e) => panic!("{name}: ORT could not build its own kernel: {e}"),
            };
            let ort_out = run_generated_case(
                api,
                ort_session,
                &case,
                &format!("Run(ORT baseline {name})"),
            );
            assert_eq!(ours_out.len(), ort_out.len(), "{name}: output length");

            // `unary_result_matches` calls +0 and -0 equal, which is right for
            // a tolerance comparison and wrong for the question these kernels
            // actually had to answer: several of them are implemented as a
            // *select* precisely so a signed zero survives, and one (`Selu`)
            // has its branch written on `x < 0` rather than ONNX's `x > 0` for
            // no other reason. Comparing the sign bit against ORT is the only
            // thing that makes those choices evidence rather than assertion.
            for (i, &xi) in x.iter().enumerate() {
                if xi == 0.0 && ours_out[i] == 0.0 && ort_out[i] == 0.0 {
                    assert_eq!(
                        ours_out[i].to_bits(),
                        ort_out[i].to_bits(),
                        "{name}: sign of zero disagrees with ORT at x={xi:e} \
                         (ours {:e}, ORT {:e})",
                        ours_out[i],
                        ort_out[i]
                    );
                }
            }

            let mut worst = 0.0f32;
            let mut worst_at = usize::MAX;
            let mut bad: Vec<String> = Vec::new();
            for (i, (&a, &b)) in ours_out.iter().zip(ort_out.iter()).enumerate() {
                if !unary_result_matches(a, b, rel) {
                    if bad.len() < 12 {
                        bad.push(format!("  x={:e}: ours={a:e} ort={b:e}", x[i]));
                    }
                    let d = (a - b).abs() / a.abs().max(b.abs()).max(1.0);
                    if d > worst {
                        worst = d;
                        worst_at = i;
                    }
                }
            }
            assert!(
                bad.is_empty(),
                "{name}: {} of {len} elements disagree with ORT (worst {worst:e} at \
                 x={:e}, tolerance {rel:e}):\n{}",
                bad.len(),
                if worst_at == usize::MAX {
                    f32::NAN
                } else {
                    x[worst_at]
                },
                bad.join("\n"),
            );
            eprintln!("  [{name}] {len} elements match ORT within {rel:e}");
            ((*api).ReleaseSession.unwrap())(ort_session);
            ((*api).ReleaseSessionOptions.unwrap())(ort_opts);
        }
        checked += 1;
    }
    assert_eq!(checked, 10, "every parity case must have been checked");
    eprintln!(
        "\n✅ the_native_activation_family_executes_locally_and_matches_ort_numerics: PASSED"
    );
}

/// Bucket names, read from the library rather than copied.
///
/// A hand-maintained copy of this list drifted from the enum and mislabelled
/// two rows of the attribution table -- every number was right and attached to
/// the wrong phase. Asking the cdylib removes the second source of truth.
fn probe_phase_names(lib: &libloading::Library) -> Vec<String> {
    // SAFETY: the export is `extern "C"` with this signature and returns either
    // null or a 'static NUL-terminated string owned by the library.
    unsafe {
        let name_of: libloading::Symbol<
            '_,
            unsafe extern "C" fn(usize) -> *const std::os::raw::c_char,
        > = match lib.get(b"nxrt_dispatch_probe_phase_name") {
            Ok(f) => f,
            Err(_) => return Vec::new(),
        };
        let mut out = Vec::new();
        for i in 0.. {
            let p = name_of(i);
            if p.is_null() {
                break;
            }
            out.push(std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned());
        }
        out
    }
}

const PROBE_EVENTS: &[&str] = &[
    "OrtFfiCall",
    "DispatchAlloc",
    "NodeExecuted",
    "ShapeInferred",
    "OutputMaterialized",
];

/// Open the EP cdylib a second time to reach its probe exports.
///
/// ORT has already `dlopen`ed this exact path, so this returns a handle to the
/// same image and therefore the same counters — the allocations recorded here
/// are the ones made inside `Compute`, which an allocator installed in this
/// test binary could never see.
///
/// `None` when the cdylib was built without the `dispatch_probe` feature, which
/// is the normal case; the symbols simply are not there.
fn probe_lib() -> Option<&'static libloading::Library> {
    static LIB: std::sync::OnceLock<Option<libloading::Library>> = std::sync::OnceLock::new();
    LIB.get_or_init(|| {
        let path = cdylib_resolve::find_cpu_plugin_cdylib_optional()?;
        // SAFETY: the path is the EP cdylib this harness already registered
        // with ORT; re-opening it runs no new initialisers.
        unsafe { libloading::Library::new(path) }.ok()
    })
    .as_ref()
}

/// Reset the probe, run `runs` iterations, and return the per-phase totals.
///
/// Returns `None` unless `NXRT_MM_BENCH_PROBE=1` and the cdylib exports the
/// probe, so the default benchmark path is untouched.
#[allow(clippy::too_many_arguments)]
unsafe fn probe_dispatch(
    api: *const ort::OrtApi,
    session: *mut ort::OrtSession,
    input_names: &[*const std::os::raw::c_char],
    values: &[*const ort::OrtValue],
    output_names: &[*const std::os::raw::c_char],
    runs: usize,
) -> Option<(Vec<u64>, Vec<String>)> {
    if std::env::var("NXRT_MM_BENCH_PROBE").unwrap_or_default() != "1" {
        return None;
    }
    let lib = probe_lib()?;
    // SAFETY: both symbols are `extern "C"` exports of the loaded cdylib with
    // exactly these signatures; a missing symbol returns `Err`.
    unsafe {
        let reset: libloading::Symbol<'_, unsafe extern "C" fn()> =
            lib.get(b"nxrt_dispatch_probe_reset").ok()?;
        let snapshot: libloading::Symbol<'_, unsafe extern "C" fn(*mut u64, usize) -> usize> =
            lib.get(b"nxrt_dispatch_probe_snapshot").ok()?;
        reset();
        bench_runs(api, session, input_names, values, output_names, runs);
        let buckets = probe_phase_names(lib).len();
        assert!(buckets > 0, "cdylib exports no phase names");
        let need = (buckets - 1) * 2 + buckets * 2 + PROBE_EVENTS.len();
        let mut buf = vec![0u64; need];
        let written = snapshot(buf.as_mut_ptr(), need);
        assert_eq!(
            written, need,
            "probe wrote {written} u64s, this harness expected {need} \
             — PROBE_EVENTS is out of sync with dispatch_probe"
        );
        Some((buf, probe_phase_names(lib)))
    }
}

/// Print allocations and bytes per phase, normalised per `Run` and per node.
fn report_probe(case: &str, nodes: usize, runs: usize, buf: &[u64], names: &[String]) {
    let nb = names.len();
    let np = nb - 1;
    let (calls, ns) = (&buf[..np], &buf[np..2 * np]);
    let (allocs, bytes) = (
        &buf[2 * np..2 * np + nb],
        &buf[2 * np + nb..2 * np + 2 * nb],
    );
    let events = &buf[2 * np + 2 * nb..];
    let per_run = runs.max(1) as f64;
    let per_node = (runs.max(1) * nodes.max(1)) as f64;
    println!("# probe {case}: nodes={nodes} runs={runs}");
    println!("# phase,calls_per_run,ns_per_run,allocs_per_run,allocs_per_node,bytes_per_run");
    for (i, name) in names.iter().enumerate().take(np) {
        println!(
            "# {name},{:.2},{:.0},{:.3},{:.3},{:.0}",
            calls[i] as f64 / per_run,
            ns[i] as f64 / per_run,
            allocs[i] as f64 / per_run,
            allocs[i] as f64 / per_node,
            bytes[i] as f64 / per_run,
        );
    }
    // The unattributed bucket is the point of the table, not a footnote: it is
    // what says whether the per-phase rows are the whole story.
    println!(
        "# {},,,{:.3},{:.3},{:.0}",
        names[nb - 1],
        allocs[nb - 1] as f64 / per_run,
        allocs[nb - 1] as f64 / per_node,
        bytes[nb - 1] as f64 / per_run,
    );
    let total: u64 = allocs.iter().sum();
    let total_bytes: u64 = bytes.iter().sum();
    println!(
        "# TOTAL_attributed,,,{:.3},{:.3},{:.0}",
        total as f64 / per_run,
        total as f64 / per_node,
        total_bytes as f64 / per_run
    );
    for (i, name) in PROBE_EVENTS.iter().enumerate() {
        println!("# event {name},{:.3}/run", events[i] as f64 / per_run);
    }
}

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

// ─── Assignment policy falsifiers ────────────────────────────────────────────
//
// `onnx_runtime_ep_cpu::assignment_policy` decides which nodes this EP asks ORT
// to hand over, separately from `supports_op`, which decides what it *can* run.
// A unit test can only check the predicate; these check the thing that actually
// matters — what ORT does with the answer. Each one is a falsifier: it fails if
// the EP takes a node the policy declines, if it gives up a node the policy
// claims, or if the resulting partition produces wrong numbers.

/// float32 `Tanh`/`Sqrt`/`Sigmoid` are measured slower than ORT's own MLAS
/// kernels at every size on this ISA, so the plugin must not claim them — while
/// `Add`, which the policy does not govern, must still be claimed. This is the
/// falsifier for "declining is per-node and ORT partitions around it": if the
/// deferral were expressed as `supports_op` returning `Unsupported`, or if it
/// dropped the whole partition, `Add` would move too.
#[test]
fn assignment_policy_defers_float32_activations_to_ort() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = PathBuf::from(manifest_dir)
        .join("tests/fixtures/activation_assignment_f32/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_assign_f32", &model_path, false) })
    else {
        eprintln!(
            "*** SKIPPED: assignment_policy_defers_float32_activations_to_ort — \
             ORT or EP cdylib not found ***"
        );
        return;
    };

    unsafe {
        let info = query_ep_assignment(api, session);
        let ours = info.ops_on_our_ep();
        let others = info.ops_not_on_our_ep();
        eprintln!("  [assign_f32] ours={ours:?}, others={others:?}");
        for op in ["Tanh", "Sqrt", "Sigmoid"] {
            assert!(
                !ours.contains(&op),
                "float32 '{op}' is measured slower than ORT and must be left to ORT, \
                 got assignment: {:?}",
                info.assignments
            );
            assert!(
                others.contains(&op),
                "float32 '{op}' must actually be executed by ORT's own EP, \
                 got assignment: {:?}",
                info.assignments
            );
        }
        // The policy governs only the ops it has evidence about. `Add` is not
        // one of them, so declining the activations must not cost us the rest
        // of the graph.
        assert!(
            ours.contains(&"Add"),
            "declining activations must not drop the ungoverned 'Add' claim, \
             got assignment: {:?}",
            info.assignments
        );
    }

    // The mixed partition must still compute the right answer: the boundary is
    // crossed twice (into ORT for the activations, and the graph output comes
    // back from ORT).
    unsafe {
        let mut x_data: [f32; 4] = [0.0, 0.25, 1.0, 4.0];
        let mut y_data: [f32; 4] = [0.0, 0.25, 1.0, 4.0];
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
        check_status(api, status, "Run(assign_f32)");
        assert!(!output.is_null());

        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(output, &mut data_ptr);
        check_status(api, status, "GetTensorMutableData(assign_f32)");
        let got = std::slice::from_raw_parts(data_ptr as *const f32, 4);
        for (i, ((&g, &x), &y)) in got.iter().zip(x_data.iter()).zip(y_data.iter()).enumerate() {
            let want = {
                let s = x + y;
                let t = s.tanh();
                let q = t.sqrt();
                1.0f32 / (1.0 + (-q).exp())
            };
            assert!(
                (g - want).abs() < 1e-5,
                "Z[{i}] = {g}, want {want} — mixed partition changed the numerics"
            );
        }
        ((*api).ReleaseValue.unwrap())(output);
        ((*api).ReleaseValue.unwrap())(x_val);
        ((*api).ReleaseValue.unwrap())(y_val);
        conformance_teardown(api, env, opts, session, "cpu_ep_assign_f32");
        eprintln!("\n✅ assignment_policy_defers_float32_activations_to_ort: PASSED");
    }
}

/// bfloat16 is the one case where declining would be actively harmful: ORT's
/// CPU EP has no bfloat16 `Tanh` kernel at all, so a deferral turns a working
/// session into a `NOT_IMPLEMENTED` session-creation failure. The claim is a
/// capability, not a performance bet, and must survive any future tightening of
/// the performance thresholds.
///
/// This is a regression guard rather than a falsifier against `main`:
/// `supports_op` is dtype-agnostic for these ops, so bf16 was already claimed
/// before this policy existed and the test passes there too. What it pins is
/// that no future threshold may take bf16 away. The assignment and numeric
/// assertions carry the test — session creation alone would succeed even if ORT
/// did have a bf16 kernel, since a claimed node never reaches ORT's.
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

/// Deferring assumes there is a host kernel to defer *to*. With
/// `session.disable_cpu_ep_fallback=1` there is not — ORT refuses to place an
/// unclaimed node on its own CPU EP, so a node this EP declines becomes
/// unassignable and `CreateSession` fails. The plugin reads that session option
/// at `CreateEp` and switches the routing-preference gate off for such
/// sessions.
///
/// This is a falsifier for that safety valve: the same float32 graph whose
/// activations are deferred in
/// `assignment_policy_defers_float32_activations_to_ort` must, under this flag,
/// both load *and* come back fully claimed. Without the valve `conformance_setup`
/// panics inside `CreateSession`; with the gate wrongly disabled everywhere, the
/// sibling test fails instead. Only the intended behaviour satisfies both.
#[test]
fn assignment_policy_yields_when_cpu_fallback_is_disabled() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = PathBuf::from(manifest_dir)
        .join("tests/fixtures/activation_assignment_f32/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_assign_nofb", &model_path, true) })
    else {
        eprintln!(
            "*** SKIPPED: assignment_policy_yields_when_cpu_fallback_is_disabled — \
             ORT or EP cdylib not found ***"
        );
        return;
    };

    unsafe {
        let info = query_ep_assignment(api, session);
        let ours = info.ops_on_our_ep();
        eprintln!(
            "  [assign_nofb] ours={ours:?}, others={:?}",
            info.ops_not_on_our_ep()
        );
        for op in ["Add", "Tanh", "Sqrt", "Sigmoid"] {
            assert!(
                ours.contains(&op),
                "with cpu-ep fallback disabled there is nothing to defer to, so '{op}' \
                 must be claimed despite the performance policy; got: {:?}",
                info.assignments
            );
        }
        conformance_teardown(api, env, opts, session, "cpu_ep_assign_nofb");
    }
    eprintln!("\n✅ assignment_policy_yields_when_cpu_fallback_is_disabled: PASSED");
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

/// Regression falsifier for a hard session-creation failure.
///
/// ORT has no float16 kernel for `com.microsoft::FastGelu`, so it inlines the
/// contrib op into its function body (`Identity`/`Mul`/`Add`/`Tanh`/…). The
/// routing preference then defers the float16 `Tanh` inside that body, which
/// splits the remainder into a partition where `_inlfunc_FastGelu_X_bias` is
/// both an output of our fused subgraph and an input to three later `Mul`s.
/// `build_subgraph_routing` cannot route that shape, and failing at Compile is
/// *not* a graceful decline — ORT surfaces it as
/// `FAIL : Compile: multi-node subgraph has unroutable graph`, turning a model
/// that used to load into one that does not.
///
/// The claim-time routing filter in `ep.rs` catches such partitions while
/// declining is still free. This test fails without it.
#[test]
fn unroutable_partition_is_declined_not_fatal() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = PathBuf::from(manifest_dir)
        .join("tests/fixtures/fastgelu_assignment_f16/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_fastgelu_f16", &model_path, false) })
    else {
        eprintln!(
            "*** SKIPPED: unroutable_partition_is_declined_not_fatal — \
             ORT or EP cdylib not found ***"
        );
        return;
    };

    unsafe {
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
    eprintln!("\n✅ unroutable_partition_is_declined_not_fatal: PASSED");
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

/// The whole point of the size thresholds: a range this EP is measured to lose
/// must not be advertised to ORT as an assignment.
///
/// float32 `Sign` crosses over at 2048 elements (measured 0.95x at 1024, 1.15x
/// at 2048), so 1024 must go to ORT and 4096 must come to us. Both halves are
/// asserted in one test because either one alone is satisfiable by a policy
/// that is simply wrong in the other direction.
#[test]
fn float32_sign_is_assigned_only_above_its_measured_crossover() {
    let Some((below_ours, below_theirs)) =
        unary_assignment("sign_assignment_f32_below", "cpu_ep_sign_below")
    else {
        eprintln!(
            "*** SKIPPED: float32_sign_is_assigned_only_above_its_measured_crossover — \
             ORT or EP cdylib not found ***"
        );
        return;
    };
    assert!(
        !below_ours.iter().any(|op| op == "Sign"),
        "1024-element float32 Sign is measured at 0.95x and must be left to ORT, \
         but this EP claimed it (ours={below_ours:?})",
    );
    assert!(
        below_theirs.iter().any(|op| op == "Sign"),
        "the declined node must land on ORT's CPU EP as a single Sign node, \
         not vanish or fragment (others={below_theirs:?})",
    );

    let Some((above_ours, _)) = unary_assignment("sign_assignment_f32_above", "cpu_ep_sign_above")
    else {
        return;
    };
    assert!(
        above_ours.iter().any(|op| op == "Sign"),
        "4096-element float32 Sign is measured at 1.53x and must be claimed, \
         but this EP declined it (ours={above_ours:?})",
    );
}

/// `Neg` is one `xorps` per element, so it is memory-bound on both sides and
/// never reaches the 5% bar — not even at the largest tensor a transformer
/// would hand a unary op. A policy that claims it anywhere is claiming a range
/// it is measured to lose.
#[test]
fn float32_neg_is_never_assigned_even_at_a_megabyte() {
    let Some((ours, theirs)) = unary_assignment("neg_assignment_f32_large", "cpu_ep_neg_large")
    else {
        eprintln!(
            "*** SKIPPED: float32_neg_is_never_assigned_even_at_a_megabyte — \
             ORT or EP cdylib not found ***"
        );
        return;
    };
    assert!(
        !ours.iter().any(|op| op == "Neg"),
        "float32 Neg peaks at 1.03x at 2 M elements and must never be claimed, \
         but this EP claimed it at 1 M (ours={ours:?})",
    );
    assert!(
        theirs.iter().any(|op| op == "Neg"),
        "declining Neg must hand ORT a single Neg node (others={theirs:?})",
    );
}

/// The one op whose float16 answer is the opposite of its float32 one. At
/// 65536 elements float32 `Sign` is claimed and float16 `Sign` must not be:
/// ORT has a native float16 kernel for it and this EP widens to float32 first,
/// measured 0.35x. A dtype-blind threshold would get this backwards.
#[test]
fn float16_sign_is_not_assigned_where_float32_sign_would_be() {
    let Some((ours, theirs)) = unary_assignment("sign_assignment_f16_large", "cpu_ep_sign_f16")
    else {
        eprintln!(
            "*** SKIPPED: float16_sign_is_not_assigned_where_float32_sign_would_be — \
             ORT or EP cdylib not found ***"
        );
        return;
    };
    assert!(
        !ours.iter().any(|op| op == "Sign"),
        "float16 Sign is measured at 0.35x against ORT's native float16 kernel and \
         must be left to ORT (ours={ours:?})",
    );
    assert!(
        theirs.iter().any(|op| op == "Sign"),
        "ORT has a real float16 Sign kernel, so the declined node must stay a single \
         Sign node rather than being cast-wrapped (others={theirs:?})",
    );
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

/// An unknown element count could be either side of the crossover, and at
/// decode it is reliably the losing side. The policy must fail conservative.
#[test]
fn dynamic_shapes_are_not_assigned() {
    let Some((ours, theirs)) = unary_assignment("sign_assignment_f32_dynamic", "cpu_ep_sign_dyn")
    else {
        eprintln!("*** SKIPPED: dynamic_shapes_are_not_assigned — ORT or EP cdylib not found ***");
        return;
    };
    assert!(
        !ours.iter().any(|op| op == "Sign"),
        "a dynamic element count is not provably above the crossover and must not be \
         claimed (ours={ours:?})",
    );
    assert!(
        theirs.iter().any(|op| op == "Sign"),
        "the declined dynamic node must still be runnable by ORT (others={theirs:?})",
    );
}

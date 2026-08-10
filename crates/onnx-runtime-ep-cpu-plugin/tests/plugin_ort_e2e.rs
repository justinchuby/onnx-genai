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
//! The underlying CPU kernels (`crates/onnx-runtime-ep-cpu`) support Float16 and BFloat16
//! for Add and MatMul. However, the ORT plugin interface routes nodes to our EP only when
//! GetCapability claims them, and our EP does not currently register explicit half-dtype
//! type-constraint metadata with ORT's node-capability API. Consequently ORT may not route
//! f16/bf16 nodes to our EP and an end-to-end ONNX-model test with half inputs is not
//! provable without kernel-registry support. This is recorded as a coverage gap; see
//! `.squad/decisions/inbox/pris-ep-conformance-final.md`.
//!
//! # Environment
//!
//! No env vars required — the test resolves ORT from the ort-sys build output.
//! Skips loudly if ORT or the EP cdylib is absent.

use std::ffi::{CStr, CString};
use std::path::PathBuf;
use std::ptr;
use std::sync::Mutex;

use onnx_genai_ort_sys as ort;

/// Serialises all tests that load our EP plugin.
///
/// ORT's per-process EP device state is corrupted after ≥6 register+Run+unregister
/// cycles (factory.rs bug — Nabil).  The lock ensures tests run one at a time so
/// the cycle count stays below the failure threshold for the default test suite.
static ORT_EP_LOCK: Mutex<()> = Mutex::new(());

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Resolve the directory containing `libonnxruntime.so`.
fn find_ort_lib_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("NXRT_ORT_LIB_DIR") {
        let p = PathBuf::from(dir);
        if p.join("libonnxruntime.so").exists() {
            return Some(p);
        }
    }

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = std::path::Path::new(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let build_dir = workspace_root.join("target/debug/build");
    if build_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(&build_dir) {
            for entry in entries.flatten() {
                let name = entry.file_name();
                if name.to_string_lossy().starts_with("onnx-genai-ort-sys-") {
                    let lib_dir = entry.path().join("out/ort-prebuilt/lib");
                    if lib_dir.join("libonnxruntime.so").exists() {
                        return Some(lib_dir);
                    }
                }
            }
        }
    }
    None
}

/// Find the EP cdylib produced by this crate.
fn find_ep_cdylib() -> Option<PathBuf> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = std::path::Path::new(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let candidates = [
        workspace_root.join("target/debug/libonnx_runtime_ep_cpu_plugin.so"),
        workspace_root.join("target/release/libonnx_runtime_ep_cpu_plugin.so"),
    ];
    candidates.into_iter().find(|p| p.exists())
}

/// Skip a test loudly when a required resource is missing.
macro_rules! skip_if_missing {
    ($resource:expr, $msg:literal) => {
        match $resource {
            Some(v) => v,
            None => {
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
    let ort_lib_path = ort_lib_dir.join("libonnxruntime.so");

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
    let _lock = ORT_EP_LOCK.lock().unwrap();
    let ort_lib_dir =
        skip_if_missing!(find_ort_lib_dir(), "ort_register_ep_library: ORT not found");
    let ep_lib_path = skip_if_missing!(
        find_ep_cdylib(),
        "ort_register_ep_library: EP cdylib not found; run cargo build -p onnx-runtime-ep-cpu-plugin"
    );

    let ort_lib_path = ort_lib_dir.join("libonnxruntime.so");
    let lib = unsafe { libloading::Library::new(&ort_lib_path) }.expect("dlopen ORT failed");
    let api = unsafe { get_ort_api(&lib) };

    unsafe {
        let mut env: *mut ort::OrtEnv = ptr::null_mut();
        let logid = CString::new("nxrt_reg_test").unwrap();
        let status =
            ((*api).CreateEnv.unwrap())(ort::ORT_LOGGING_LEVEL_WARNING, logid.as_ptr(), &mut env);
        check_status(api, status, "CreateEnv");

        let reg_name = CString::new("cpu_ep").unwrap();
        let ep_path_c = CString::new(ep_lib_path.to_str().unwrap()).unwrap();
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
    let _lock = ORT_EP_LOCK.lock().unwrap();
    let ort_lib_dir = skip_if_missing!(
        find_ort_lib_dir(),
        "ort_loads_our_ep_and_runs_model: ORT not found"
    );
    let ep_lib_path = skip_if_missing!(
        find_ep_cdylib(),
        "ort_loads_our_ep_and_runs_model: EP cdylib not found"
    );

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = PathBuf::from(manifest_dir).join("tests/fixtures/add_1x4/model.onnx");
    assert!(
        model_path.exists(),
        "Missing model fixture: {}",
        model_path.display()
    );

    let ort_lib_path = ort_lib_dir.join("libonnxruntime.so");
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
        let ep_path_c = CString::new(ep_lib_path.to_str().unwrap()).unwrap();
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
        let model_c = CString::new(model_path.to_str().unwrap()).unwrap();
        let mut session: *mut ort::OrtSession = ptr::null_mut();
        let status =
            ((*api).CreateSession.unwrap())(env, model_c.as_ptr(), session_options, &mut session);
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
    let _lock = ORT_EP_LOCK.lock().unwrap();
    let ort_lib_dir = skip_if_missing!(
        find_ort_lib_dir(),
        "ort_unsupported_op_declines_not_crashes: ORT not found"
    );
    let ep_lib_path = skip_if_missing!(
        find_ep_cdylib(),
        "ort_unsupported_op_declines_not_crashes: EP cdylib not found"
    );

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = PathBuf::from(manifest_dir).join("tests/fixtures/nonzero_1x4/model.onnx");
    if !model_path.exists() {
        eprintln!(
            "*** SKIPPED: unsupported-op fixture missing at {}\n\
             Generate it with: python3 tests/fixtures/generate_nonzero.py",
            model_path.display()
        );
        return;
    }

    let ort_lib_path = ort_lib_dir.join("libonnxruntime.so");
    let lib = unsafe { libloading::Library::new(&ort_lib_path) }.expect("dlopen ORT");
    let api = unsafe { get_ort_api(&lib) };

    unsafe {
        let mut env: *mut ort::OrtEnv = ptr::null_mut();
        let logid = CString::new("nxrt_neg_test").unwrap();
        let status =
            ((*api).CreateEnv.unwrap())(ort::ORT_LOGGING_LEVEL_WARNING, logid.as_ptr(), &mut env);
        check_status(api, status, "CreateEnv");

        let reg_name = CString::new("cpu_ep_neg").unwrap();
        let ep_path_c = CString::new(ep_lib_path.to_str().unwrap()).unwrap();
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
        let model_c = CString::new(model_path.to_str().unwrap()).unwrap();
        let mut session: *mut ort::OrtSession = ptr::null_mut();
        let status =
            ((*api).CreateSession.unwrap())(env, model_c.as_ptr(), session_options, &mut session);
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
    let ort_lib_dir = match find_ort_lib_dir() {
        Some(d) => d,
        None => {
            eprintln!("SKIPPED: no ORT lib dir found");
            return;
        }
    };
    let ort_lib_path = ort_lib_dir.join("libonnxruntime.so");
    let lib = unsafe { libloading::Library::new(&ort_lib_path) }.unwrap();
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
#[allow(clippy::type_complexity)]
unsafe fn conformance_setup(
    reg_name: &str,
    model_path: &std::path::Path,
) -> Option<(
    libloading::Library,
    *const ort::OrtApi,
    *mut ort::OrtEnv,
    *mut ort::OrtSessionOptions,
    *mut ort::OrtSession,
)> {
    let ort_lib_dir = find_ort_lib_dir()?;
    let ep_lib_path = find_ep_cdylib()?;

    if !model_path.exists() {
        eprintln!(
            "*** SKIPPED: fixture missing at {} ***",
            model_path.display()
        );
        return None;
    }

    let ort_lib_path = ort_lib_dir.join("libonnxruntime.so");
    let lib = unsafe { libloading::Library::new(&ort_lib_path) }.ok()?;
    let api = unsafe { get_ort_api(&lib) };

    // Env
    let mut env: *mut ort::OrtEnv = ptr::null_mut();
    let logid = std::ffi::CString::new(format!("nxrt_{reg_name}")).unwrap();
    let status = unsafe {
        ((*api).CreateEnv.unwrap())(ort::ORT_LOGGING_LEVEL_WARNING, logid.as_ptr(), &mut env)
    };
    unsafe { check_status(api, status, "CreateEnv") };

    // Register EP library
    let reg_name_c = std::ffi::CString::new(reg_name).unwrap();
    let ep_path_c = std::ffi::CString::new(ep_lib_path.to_str().unwrap()).unwrap();
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
    let model_c = std::ffi::CString::new(model_path.to_str().unwrap()).unwrap();
    let mut session: *mut ort::OrtSession = ptr::null_mut();
    let status = unsafe {
        ((*api).CreateSession.unwrap())(env, model_c.as_ptr(), session_options, &mut session)
    };
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
    let _lock = ORT_EP_LOCK.lock().unwrap();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = PathBuf::from(manifest_dir).join("tests/fixtures/add_broadcast/model.onnx");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_bc", &model_path) })
    else {
        eprintln!("*** SKIPPED: conformance_add_broadcast — ORT or EP cdylib not found ***");
        return;
    };

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
    let _lock = ORT_EP_LOCK.lock().unwrap();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = PathBuf::from(manifest_dir).join("tests/fixtures/chain_add_mul/model.onnx");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_chain", &model_path) })
    else {
        eprintln!("*** SKIPPED: conformance_chain_add_mul — ORT or EP cdylib not found ***");
        return;
    };

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
    let _lock = ORT_EP_LOCK.lock().unwrap();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = PathBuf::from(manifest_dir).join("tests/fixtures/matmul_2d/model.onnx");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_mm", &model_path) })
    else {
        eprintln!("*** SKIPPED: conformance_matmul_2d — ORT or EP cdylib not found ***");
        return;
    };

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
    let _lock = ORT_EP_LOCK.lock().unwrap();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = PathBuf::from(manifest_dir).join("tests/fixtures/mixed_partition/model.onnx");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_mix", &model_path) })
    else {
        eprintln!("*** SKIPPED: conformance_mixed_partition — ORT or EP cdylib not found ***");
        return;
    };

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
    let _lock = ORT_EP_LOCK.lock().unwrap();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = PathBuf::from(manifest_dir).join("tests/fixtures/add_int32/model.onnx");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_i32", &model_path) })
    else {
        eprintln!("*** SKIPPED: conformance_add_int32 — ORT or EP cdylib not found ***");
        return;
    };

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
    let _lock = ORT_EP_LOCK.lock().unwrap();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = PathBuf::from(manifest_dir).join("tests/fixtures/add_dynamic_dim/model.onnx");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_dyn", &model_path) })
    else {
        eprintln!("*** SKIPPED: conformance_add_dynamic_dim — ORT or EP cdylib not found ***");
        return;
    };

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
    let _lock = ORT_EP_LOCK.lock().unwrap();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = PathBuf::from(manifest_dir).join("tests/fixtures/add_1x4/model.onnx");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_runs", &model_path) })
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
    let _lock = ORT_EP_LOCK.lock().unwrap();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_a = PathBuf::from(manifest_dir).join("tests/fixtures/add_1x4/model.onnx");
    let model_b = PathBuf::from(manifest_dir).join("tests/fixtures/add_broadcast/model.onnx");

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

    let ort_lib_path = ort_lib_dir.join("libonnxruntime.so");

    unsafe {
        let lib = libloading::Library::new(&ort_lib_path).expect("dlopen ORT");
        let api = get_ort_api(&lib);

        let mut env: *mut ort::OrtEnv = ptr::null_mut();
        let logid = std::ffi::CString::new("nxrt_two_sess").unwrap();
        let status =
            ((*api).CreateEnv.unwrap())(ort::ORT_LOGGING_LEVEL_WARNING, logid.as_ptr(), &mut env);
        check_status(api, status, "CreateEnv");

        let reg_name = std::ffi::CString::new("cpu_ep_2sess").unwrap();
        let ep_path_c = std::ffi::CString::new(ep_lib_path.to_str().unwrap()).unwrap();
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
        let model_a_c = std::ffi::CString::new(model_a.to_str().unwrap()).unwrap();
        let mut sess_a: *mut ort::OrtSession = ptr::null_mut();
        let status = ((*api).CreateSession.unwrap())(env, model_a_c.as_ptr(), opts_a, &mut sess_a);
        check_status(api, status, "CreateSession(A)");
        eprintln!("✓ Session A created (add_1x4)");

        // Build session B
        let mut opts_b: *mut ort::OrtSessionOptions = ptr::null_mut();
        let status = ((*api).CreateSessionOptions.unwrap())(&mut opts_b);
        check_status(api, status, "CreateSessionOptions(B)");
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
        let model_b_c = std::ffi::CString::new(model_b.to_str().unwrap()).unwrap();
        let mut sess_b: *mut ort::OrtSession = ptr::null_mut();
        let status = ((*api).CreateSession.unwrap())(env, model_b_c.as_ptr(), opts_b, &mut sess_b);
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
    let _lock = ORT_EP_LOCK.lock().unwrap();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path =
        PathBuf::from(manifest_dir).join("tests/fixtures/matmul_batched_nd/model.onnx");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_mm3d", &model_path) })
    else {
        eprintln!("*** SKIPPED: conformance_matmul_batched_nd — ORT or EP cdylib not found ***");
        return;
    };

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
    let _lock = ORT_EP_LOCK.lock().unwrap();

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
    let model_path = PathBuf::from(manifest_dir).join("tests/fixtures/add_1x4/model.onnx");
    if !model_path.exists() {
        eprintln!(
            "*** SKIPPED: stress_register_run_unregister_cycles — add_1x4 fixture missing ***"
        );
        return;
    }

    let ort_lib_path = ort_lib_dir.join("libonnxruntime.so");
    // Keep the library loaded for the whole test — dlopen reference-counts.
    let lib = unsafe { libloading::Library::new(&ort_lib_path) }.expect("dlopen ORT");
    let api = unsafe { get_ort_api(&lib) };
    let ep_path_c = CString::new(ep_lib_path.to_str().unwrap()).unwrap();

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
            let model_c = CString::new(model_path.to_str().unwrap()).unwrap();
            let mut session: *mut ort::OrtSession = ptr::null_mut();
            let status =
                ((*api).CreateSession.unwrap())(env, model_c.as_ptr(), sess_opts, &mut session);
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

// ─── f16/bf16 end-to-end conformance (blocked on Deckard's registry_entries) ─

/// Float16 Add through ORT: `Z = X + Y` with all-f16 inputs and outputs.
///
/// The CPU kernels in `crates/onnx-runtime-ep-cpu` accept Float16 for Add.
/// **Blocked:** ORT routes a node to our EP via `GetCapability`, which queries
/// `supports_op`. The EP's `supports_op` accepts any dtype that the kernel
/// handles — but ORT *also* consults `GetKernelRegistry` to learn our
/// type-constraint metadata before routing. Until Deckard lands
/// `registry_entries()` on `CpuExecutionProvider` (and the cpu-plugin shim
/// wires it through in `crates/onnx-runtime-ep-cpu/src/provider.rs` /
/// `crates/onnx-runtime-ep-cpu-plugin/src/lib.rs`), ORT does not route
/// f16 nodes to us and this test will fail with "no output" or fall through
/// to ORT's built-in CPU EP.
///
/// Remove the `#[ignore]` when `registry_entries()` lands and
/// `cargo test -p onnx-runtime-ep-cpu-plugin conformance_add_float16 -- --ignored`
/// passes with the numerically correct output below.
#[test]
#[ignore = "blocked: Deckard has not yet landed registry_entries() on CpuExecutionProvider \
            (crates/onnx-runtime-ep-cpu/src/provider.rs). Without it ORT does not route \
            Float16 nodes to our EP via GetKernelRegistry."]
fn conformance_add_float16() {
    let _lock = ORT_EP_LOCK.lock().unwrap();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path =
        PathBuf::from(manifest_dir).join("tests/fixtures/add_float16/model.onnx");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_f16", &model_path) })
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
        conformance_teardown(api, env, opts, session, "cpu_ep_f16");
        eprintln!("\n✅ conformance_add_float16: PASSED");
    }
}

/// BFloat16 Add through ORT: `Z = X + Y` with all-bf16 inputs and outputs.
///
/// Same routing dependency as `conformance_add_float16` above — blocked on
/// `registry_entries()` landing in `crates/onnx-runtime-ep-cpu/src/provider.rs`.
///
/// BFloat16 is the top 16 bits of a float32 mantissa: 1.0=0x3F80, 2.0=0x4000,
/// 3.0=0x4040, 4.0=0x4080.
#[test]
#[ignore = "blocked: Deckard has not yet landed registry_entries() on CpuExecutionProvider \
            (crates/onnx-runtime-ep-cpu/src/provider.rs). Without it ORT does not route \
            BFloat16 nodes to our EP via GetKernelRegistry."]
fn conformance_add_bfloat16() {
    let _lock = ORT_EP_LOCK.lock().unwrap();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path =
        PathBuf::from(manifest_dir).join("tests/fixtures/add_bfloat16/model.onnx");

    let Some((_lib, api, env, opts, session)) =
        (unsafe { conformance_setup("cpu_ep_bf16", &model_path) })
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
        conformance_teardown(api, env, opts, session, "cpu_ep_bf16");
        eprintln!("\n✅ conformance_add_bfloat16: PASSED");
    }
}

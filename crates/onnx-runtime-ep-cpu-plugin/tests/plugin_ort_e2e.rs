//! L3 integration tests: Real upstream ORT loads our EP plugin.
//!
//! # Test layers
//!
//! - `ort_api_sanity` — verify ORT loads and all EP-related vtable slots are non-null.
//! - `ort_register_ep_library` — real `RegisterExecutionProviderLibrary` call.
//! - `ort_loads_our_ep_and_runs_model` — full end-to-end: Env → Register → Devices → Session → Run.
//! - `ort_unsupported_op_declines_not_crashes` — negative: model with unsupported op must not crash.
//!
//! # Environment
//!
//! No env vars required — the test resolves ORT from the ort-sys build output.
//! Skips loudly if ORT or the EP cdylib is absent.

use std::ffi::{CStr, CString};
use std::path::PathBuf;
use std::ptr;

use onnx_genai_ort_sys as ort;

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
                eprintln!(
                    "\n*** SKIPPED: {} ***\n",
                    $msg
                );
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
        let get_msg = unsafe { (*api).GetErrorMessage }
            .expect("GetErrorMessage not in OrtApi");
        let msg_ptr = unsafe { get_msg(status) };
        let msg = if msg_ptr.is_null() {
            "(no message)".to_owned()
        } else {
            unsafe { CStr::from_ptr(msg_ptr) }.to_string_lossy().into_owned()
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
/// `factory.rs::factory_get_supported_devices` returns `*out_num = 0` (no devices).
/// ORT 1.27 calls `GetSupportedDevices` inside `RegisterExecutionProviderLibrary`
/// and segfaults when the factory reports zero devices.
///
/// Root cause: `GetSupportedDevices` must call `OrtEpApi::CreateEpDevice` to
/// create at least one `OrtEpDevice` for the CPU hardware device.
/// File: `crates/onnx-runtime-ep-plugin/src/factory.rs` — owned by Nabil.
///
/// This test is `#[ignore]`d until that bug is fixed.
#[test]
#[ignore = "BLOCKED: factory.rs::GetSupportedDevices returns 0 devices → ORT segfaults in \
            RegisterExecutionProviderLibrary. Fix: call OrtEpApi::CreateEpDevice in \
            crates/onnx-runtime-ep-plugin/src/factory.rs (Nabil's file)."]
fn ort_register_ep_library() {
    let ort_lib_dir = skip_if_missing!(
        find_ort_lib_dir(),
        "ort_register_ep_library: ORT not found"
    );
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
        let status = ((*api).CreateEnv.unwrap())(
            ort::ORT_LOGGING_LEVEL_WARNING,
            logid.as_ptr(),
            &mut env,
        );
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
#[ignore = "BLOCKED: factory.rs::GetSupportedDevices returns 0 devices → ORT segfaults in \
            RegisterExecutionProviderLibrary. Fix: call OrtEpApi::CreateEpDevice in \
            crates/onnx-runtime-ep-plugin/src/factory.rs (Nabil's file). \
            Additionally, compute.rs returns ORT_NOT_IMPLEMENTED (Deckard's fix pending)."]
fn ort_loads_our_ep_and_runs_model() {
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
        let status = ((*api).CreateEnv.unwrap())(
            ort::ORT_LOGGING_LEVEL_WARNING,
            logid.as_ptr(),
            &mut env,
        );
        check_status(api, status, "CreateEnv");
        eprintln!("✓ Stage 1: CreateEnv");

        // Stage 2: RegisterExecutionProviderLibrary
        let reg_name = CString::new("cpu_ep").unwrap();
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
        assert!(!our_device.is_null(), "EP 'cpu_ep' not in GetEpDevices result");
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
        let status = ((*api).CreateSession.unwrap())(env, model_c.as_ptr(), session_options, &mut session);
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
            shape.as_ptr(), 2,
            ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &mut x_val,
        );
        check_status(api, status, "CreateTensor(X)");

        let mut y_val: *mut ort::OrtValue = ptr::null_mut();
        let status = ((*api).CreateTensorWithDataAsOrtValue.unwrap())(
            mem_info,
            y_data.as_mut_ptr().cast(),
            4 * std::mem::size_of::<f32>(),
            shape.as_ptr(), 2,
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
            assert!((got - want).abs() < 1e-6, "output[{i}] = {got}, want {want}");
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

/// An unsupported-op model must be declined (not claimed + fail at Run) and must not crash.
///
/// Our CPU EP supports Add and Mul. A model containing only `NonZero` (not implemented)
/// must fall through to ORT's default CPU EP rather than our plugin EP accepting it.
///
/// Blocked by the same factory.rs bug as the main e2e test.
#[test]
#[ignore = "BLOCKED: factory.rs::GetSupportedDevices bug (same as ort_loads_our_ep_and_runs_model). \
            Once GetSupportedDevices is fixed, this test exercises GetCapability fail-closed behavior: \
            our EP should decline nodes it doesn't support so ORT falls back to its default CPU EP."]
fn ort_unsupported_op_declines_not_crashes() {
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
        let status = ((*api).CreateEnv.unwrap())(
            ort::ORT_LOGGING_LEVEL_WARNING,
            logid.as_ptr(),
            &mut env,
        );
        check_status(api, status, "CreateEnv");

        let reg_name = CString::new("cpu_ep").unwrap();
        let ep_path_c = CString::new(ep_lib_path.to_str().unwrap()).unwrap();
        let status = ((*api).RegisterExecutionProviderLibrary.unwrap())(
            env, reg_name.as_ptr(), ep_path_c.as_ptr(),
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
        assert!(!our_device.is_null(), "EP 'cpu_ep' not found in device list");

        let mut session_options: *mut ort::OrtSessionOptions = ptr::null_mut();
        let status = ((*api).CreateSessionOptions.unwrap())(&mut session_options);
        check_status(api, status, "CreateSessionOptions");

        let devices_arr: [*const ort::OrtEpDevice; 1] = [our_device];
        let status = ((*api).SessionOptionsAppendExecutionProvider_V2.unwrap())(
            session_options, env, devices_arr.as_ptr(), 1,
            ptr::null(), ptr::null(), 0,
        );
        check_status(api, status, "SessionOptionsAppendExecutionProvider_V2");

        // CreateSession with unsupported-op model: should succeed (ORT falls back to default EP)
        let model_c = CString::new(model_path.to_str().unwrap()).unwrap();
        let mut session: *mut ort::OrtSession = ptr::null_mut();
        let status = ((*api).CreateSession.unwrap())(env, model_c.as_ptr(), session_options, &mut session);
        check_status(api, status, "CreateSession(unsupported-op model)");
        assert!(!session.is_null(), "Session is null for unsupported-op model");
        eprintln!("✓ CreateSession succeeded for unsupported-op model (EP declined, ORT fell back)");

        // Run must also succeed (ORT's default EP handles NonZero)
        let mut input_data: [f32; 4] = [0.0, 1.0, 0.0, 2.0];
        let shape: [i64; 2] = [1, 4];
        let mut mem_info: *mut ort::OrtMemoryInfo = ptr::null_mut();
        let status = ((*api).CreateCpuMemoryInfo.unwrap())(
            ort::OrtDeviceAllocator, ort::OrtMemTypeDefault, &mut mem_info,
        );
        check_status(api, status, "CreateCpuMemoryInfo");

        let mut input_val: *mut ort::OrtValue = ptr::null_mut();
        let status = ((*api).CreateTensorWithDataAsOrtValue.unwrap())(
            mem_info, input_data.as_mut_ptr().cast(),
            4 * std::mem::size_of::<f32>(), shape.as_ptr(), 2,
            ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &mut input_val,
        );
        check_status(api, status, "CreateTensor(input)");

        let input_names = [c"X".as_ptr()];
        let output_names = [c"Y".as_ptr()];
        let inputs: [*const ort::OrtValue; 1] = [input_val];
        let mut output: *mut ort::OrtValue = ptr::null_mut();
        let status = ((*api).Run.unwrap())(
            session, ptr::null(),
            input_names.as_ptr(), inputs.as_ptr(), 1,
            output_names.as_ptr(), 1, &mut output,
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
            eprintln!("  {:60}: {}", stringify!($field),
                if unsafe { (*api).$field }.is_some() { "PRESENT" } else { "NULL" });
        }
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

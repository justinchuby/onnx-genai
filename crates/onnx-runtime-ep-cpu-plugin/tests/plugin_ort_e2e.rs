//! L3 integration test: Real upstream ORT loads our EP plugin and runs a model.
//!
//! This test exercises the full registration path:
//!   1. RegisterExecutionProviderLibrary — dlopen + dlsym CreateEpFactories
//!   2. GetEpDevices — enumerate our EP's devices
//!   3. SessionOptionsAppendExecutionProvider_V2 — attach our EP
//!   4. CreateSession + Run — inference on a tiny Add model
//!   5. UnregisterExecutionProviderLibrary — cleanup
//!
//! # Environment gating
//!
//! Set `NXRT_ORT_LIB_DIR` to the directory containing `libonnxruntime.so` (ORT 1.27+).
//! If unset, the test skips with a clear message. The EP cdylib must be pre-built
//! (cargo build -p onnx-runtime-ep-cpu-plugin).
//!
//! # Reasoning: ORT 1.27 vs 1.28
//!
//! We test against ORT 1.27.0 because our ort-sys bindings are compiled with
//! `ORT_API_VERSION = 27`. Using the matching version avoids any forward-compat
//! edge cases and matches the library ort-sys downloads during build.

use std::ffi::{CStr, CString};
use std::path::PathBuf;
use std::ptr;

/// Resolve the ORT library directory. Tries:
/// 1. `NXRT_ORT_LIB_DIR` env var
/// 2. The ort-sys build output (heuristic: find the most recent one)
fn find_ort_lib_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("NXRT_ORT_LIB_DIR") {
        let p = PathBuf::from(dir);
        if p.join("libonnxruntime.so").exists() {
            return Some(p);
        }
    }

    // Fallback: ort-sys build output
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = std::path::Path::new(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap();

    // Search for the ort-sys build output
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

/// Find our EP cdylib.
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

/// Helper: get the ORT API vtable from a loaded library.
unsafe fn get_ort_api(
    lib: &libloading::Library,
) -> *const onnx_genai_ort_sys::OrtApi {
    type GetApiBaseFn = unsafe extern "C" fn() -> *const onnx_genai_ort_sys::OrtApiBase;
    let get_api_base: libloading::Symbol<'_, GetApiBaseFn> =
        lib.get(b"OrtGetApiBase").expect("OrtGetApiBase not found");
    let api_base = get_api_base();
    assert!(!api_base.is_null(), "OrtGetApiBase returned null");
    let get_api = (*api_base).GetApi.unwrap();
    let api = get_api(onnx_genai_ort_sys::ORT_API_VERSION);
    assert!(!api.is_null(), "GetApi returned null for our API version");
    api
}

/// Helper: check an ORT status and panic with the error message if non-null.
unsafe fn check_status(api: *const onnx_genai_ort_sys::OrtApi, status: *mut onnx_genai_ort_sys::OrtStatus, stage: &str) {
    if !status.is_null() {
        let msg = ((*api).GetErrorMessage.unwrap())(status);
        let msg_str = CStr::from_ptr(msg).to_string_lossy();
        panic!("STAGE [{stage}] FAILED: {msg_str}");
    }
}

#[test]
fn ort_loads_our_ep_and_runs_model() {
    // --- Gate: find ORT library ---
    let ort_lib_dir = match find_ort_lib_dir() {
        Some(d) => d,
        None => {
            eprintln!(
                "\n\n*** SKIPPED: L3 ORT e2e test ***\n\
                 Set NXRT_ORT_LIB_DIR to a directory containing libonnxruntime.so (ORT 1.27+)\n\
                 or ensure `cargo build -p onnx-genai-ort-sys` has been run first.\n\n"
            );
            return;
        }
    };

    // --- Gate: find our EP cdylib ---
    let ep_lib_path = match find_ep_cdylib() {
        Some(p) => p,
        None => {
            eprintln!(
                "\n\n*** SKIPPED: L3 ORT e2e test ***\n\
                 Build the EP plugin first: cargo build -p onnx-runtime-ep-cpu-plugin\n\n"
            );
            return;
        }
    };

    // --- Gate: find the model fixture ---
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = PathBuf::from(manifest_dir).join("tests/fixtures/add_1x4/model.onnx");
    assert!(
        model_path.exists(),
        "Model fixture not found at {}",
        model_path.display()
    );

    eprintln!("ORT lib dir: {}", ort_lib_dir.display());
    eprintln!("EP cdylib: {}", ep_lib_path.display());
    eprintln!("Model: {}", model_path.display());

    let ort_lib_path = ort_lib_dir.join("libonnxruntime.so");

    // SAFETY: We control all the pointers and lifetimes below.
    unsafe {
        // --- Pre-flight: L2-style vtable check before handing to ORT ---
        // This catches missing vtable entries that would segfault inside ORT.
        {
            let plugin_lib = libloading::Library::new(&ep_lib_path)
                .expect("Pre-flight: failed to dlopen EP plugin");

            type CreateEpFactoriesFn = unsafe extern "C" fn(
                *const std::ffi::c_char,
                *const onnx_genai_ort_sys::OrtApiBase,
                *const onnx_genai_ort_sys::OrtLogger,
                *mut *mut onnx_genai_ort_sys::OrtEpFactory,
                usize,
                *mut usize,
            ) -> *mut onnx_genai_ort_sys::OrtStatus;

            let create: libloading::Symbol<'_, CreateEpFactoriesFn> =
                plugin_lib.get(b"CreateEpFactories").expect("CreateEpFactories not found");

            // Load ORT to get a real OrtApiBase for pre-flight
            let ort_lib_preflight = libloading::Library::new(&ort_lib_path)
                .expect("Pre-flight: failed to load ORT");
            let api_base = {
                type GetApiBaseFn = unsafe extern "C" fn() -> *const onnx_genai_ort_sys::OrtApiBase;
                let f: libloading::Symbol<'_, GetApiBaseFn> = ort_lib_preflight.get(b"OrtGetApiBase").unwrap();
                f()
            };

            let mut factories: [*mut onnx_genai_ort_sys::OrtEpFactory; 1] = [ptr::null_mut()];
            let mut num_factories = 0usize;
            let status = create(
                ptr::null(),
                api_base,
                ptr::null(),
                factories.as_mut_ptr(),
                1,
                &mut num_factories,
            );
            assert!(status.is_null(), "Pre-flight: CreateEpFactories returned error status");
            assert_eq!(num_factories, 1, "Pre-flight: expected 1 factory");
            let factory = factories[0];
            assert!(!factory.is_null(), "Pre-flight: factory pointer is null");

            // Check required vtable entries that ORT will call during registration
            let vtable = &*factory;
            let mut missing = Vec::new();
            if vtable.GetName.is_none() { missing.push("GetName"); }
            if vtable.GetVendor.is_none() { missing.push("GetVendor"); }
            if vtable.GetSupportedDevices.is_none() { missing.push("GetSupportedDevices"); }
            if vtable.GetVendorId.is_none() { missing.push("GetVendorId"); }
            if vtable.GetVersion.is_none() { missing.push("GetVersion"); }
            if vtable.CreateEp.is_none() { missing.push("CreateEp"); }
            if vtable.ReleaseEp.is_none() { missing.push("ReleaseEp"); }

            if !missing.is_empty() {
                // Release factory before panicking
                type ReleaseEpFactoryFn = unsafe extern "C" fn(*mut onnx_genai_ort_sys::OrtEpFactory) -> *mut onnx_genai_ort_sys::OrtStatus;
                if let Ok(release) = plugin_lib.get::<ReleaseEpFactoryFn>(b"ReleaseEpFactory") {
                    release(factory);
                }
                panic!(
                    "PRE-FLIGHT FAILED: OrtEpFactory vtable has None entries for required methods: {missing:?}\n\
                     ORT will segfault calling these. Fix required in crates/onnx-runtime-ep-plugin/src/factory.rs\n\
                     Stage reached: CreateEpFactories succeeds, but factory vtable is incomplete."
                );
            }
            eprintln!("✓ Pre-flight: All required vtable entries are populated");

            // Release
            type ReleaseEpFactoryFn = unsafe extern "C" fn(*mut onnx_genai_ort_sys::OrtEpFactory) -> *mut onnx_genai_ort_sys::OrtStatus;
            let release: libloading::Symbol<'_, ReleaseEpFactoryFn> =
                plugin_lib.get(b"ReleaseEpFactory").unwrap();
            release(factory);
        }

        // Load ORT
        let lib = libloading::Library::new(&ort_lib_path)
            .unwrap_or_else(|e| panic!("Failed to load ORT from {}: {e}", ort_lib_path.display()));

        let api = get_ort_api(&lib);

        // Stage 1: CreateEnv
        let mut env: *mut onnx_genai_ort_sys::OrtEnv = ptr::null_mut();
        let logid = CString::new("nxrt_l3_test").unwrap();
        let status = ((*api).CreateEnv.unwrap())(
            onnx_genai_ort_sys::ORT_LOGGING_LEVEL_WARNING,
            logid.as_ptr(),
            &mut env,
        );
        check_status(api, status, "CreateEnv");
        assert!(!env.is_null(), "STAGE [CreateEnv]: env is null");
        eprintln!("✓ Stage 1: CreateEnv succeeded");

        // Stage 2: RegisterExecutionProviderLibrary
        let reg_name = CString::new("cpu_ep").unwrap();
        let ep_path_str = CString::new(ep_lib_path.to_str().unwrap()).unwrap();
        let register_fn = (*api).RegisterExecutionProviderLibrary
            .expect("RegisterExecutionProviderLibrary not available in this ORT build");
        let status = register_fn(env, reg_name.as_ptr(), ep_path_str.as_ptr());
        check_status(api, status, "RegisterExecutionProviderLibrary");
        eprintln!("✓ Stage 2: RegisterExecutionProviderLibrary succeeded");

        // Stage 3: GetEpDevices — verify our EP is listed
        let get_ep_devices = (*api).GetEpDevices
            .expect("GetEpDevices not available in this ORT build");
        let mut ep_devices: *const *const onnx_genai_ort_sys::OrtEpDevice = ptr::null();
        let mut num_devices: usize = 0;
        let status = get_ep_devices(env, &mut ep_devices, &mut num_devices);
        check_status(api, status, "GetEpDevices");
        eprintln!("✓ Stage 3: GetEpDevices returned {num_devices} device(s)");

        // Find our EP device
        let ep_device_name_fn = (*api).EpDevice_EpName
            .expect("EpDevice_EpName not available");
        let mut our_device: *const onnx_genai_ort_sys::OrtEpDevice = ptr::null();
        for i in 0..num_devices {
            let device = *ep_devices.add(i);
            let name_ptr = ep_device_name_fn(device);
            if !name_ptr.is_null() {
                let name = CStr::from_ptr(name_ptr).to_string_lossy();
                eprintln!("  Device {i}: EP name = {name:?}");
                if name == "cpu_ep" {
                    our_device = device;
                }
            }
        }
        assert!(
            !our_device.is_null(),
            "STAGE [GetEpDevices]: our EP 'cpu_ep' not found among {num_devices} devices"
        );
        eprintln!("✓ Stage 3b: Found our EP device 'cpu_ep'");

        // Stage 4: Create session options + append our EP
        let mut session_options: *mut onnx_genai_ort_sys::OrtSessionOptions = ptr::null_mut();
        let status = ((*api).CreateSessionOptions.unwrap())(&mut session_options);
        check_status(api, status, "CreateSessionOptions");

        let append_ep_v2 = (*api).SessionOptionsAppendExecutionProvider_V2
            .expect("SessionOptionsAppendExecutionProvider_V2 not available");
        let devices_arr: [*const onnx_genai_ort_sys::OrtEpDevice; 1] = [our_device];
        let status = append_ep_v2(
            session_options,
            env,
            devices_arr.as_ptr(),
            1,
            ptr::null(),
            ptr::null(),
            0,
        );
        check_status(api, status, "SessionOptionsAppendExecutionProvider_V2");
        eprintln!("✓ Stage 4: SessionOptionsAppendExecutionProvider_V2 succeeded");

        // Stage 5: CreateSession
        let model_path_c = CString::new(model_path.to_str().unwrap()).unwrap();
        let mut session: *mut onnx_genai_ort_sys::OrtSession = ptr::null_mut();
        let status = ((*api).CreateSession.unwrap())(
            env,
            model_path_c.as_ptr(),
            session_options,
            &mut session,
        );
        check_status(api, status, "CreateSession");
        assert!(!session.is_null(), "STAGE [CreateSession]: session is null");
        eprintln!("✓ Stage 5: CreateSession succeeded");

        // Stage 6: Run — Add([1,2,3,4], [5,6,7,8]) = [6,8,10,12]
        let mut x_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let mut y_data: [f32; 4] = [5.0, 6.0, 7.0, 8.0];
        let shape: [i64; 2] = [1, 4];

        // Create memory info
        let mut mem_info: *mut onnx_genai_ort_sys::OrtMemoryInfo = ptr::null_mut();
        let status = ((*api).CreateCpuMemoryInfo.unwrap())(
            onnx_genai_ort_sys::OrtDeviceAllocator,
            onnx_genai_ort_sys::OrtMemTypeDefault,
            &mut mem_info,
        );
        check_status(api, status, "CreateCpuMemoryInfo");

        // Create input tensors
        let mut x_tensor: *mut onnx_genai_ort_sys::OrtValue = ptr::null_mut();
        let status = ((*api).CreateTensorWithDataAsOrtValue.unwrap())(
            mem_info,
            x_data.as_mut_ptr().cast(),
            (4 * std::mem::size_of::<f32>()),
            shape.as_ptr(),
            2,
            onnx_genai_ort_sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &mut x_tensor,
        );
        check_status(api, status, "CreateTensor(X)");

        let mut y_tensor: *mut onnx_genai_ort_sys::OrtValue = ptr::null_mut();
        let status = ((*api).CreateTensorWithDataAsOrtValue.unwrap())(
            mem_info,
            y_data.as_mut_ptr().cast(),
            (4 * std::mem::size_of::<f32>()),
            shape.as_ptr(),
            2,
            onnx_genai_ort_sys::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &mut y_tensor,
        );
        check_status(api, status, "CreateTensor(Y)");

        // Run
        let input_names = [
            c"X".as_ptr(),
            c"Y".as_ptr(),
        ];
        let output_names = [c"Z".as_ptr()];
        let inputs = [x_tensor as *const _, y_tensor as *const _];
        let mut output: *mut onnx_genai_ort_sys::OrtValue = ptr::null_mut();

        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(), // run_options
            input_names.as_ptr(),
            inputs.as_ptr(),
            2,
            output_names.as_ptr(),
            1,
            &mut output,
        );
        check_status(api, status, "Run");
        assert!(!output.is_null(), "STAGE [Run]: output is null");
        eprintln!("✓ Stage 6: Run succeeded");

        // Stage 7: Verify output values
        let mut output_data: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(output, &mut output_data);
        check_status(api, status, "GetTensorMutableData");

        let result = std::slice::from_raw_parts(output_data as *const f32, 4);
        let expected: [f32; 4] = [6.0, 8.0, 10.0, 12.0];
        eprintln!("  Output: {result:?}");
        eprintln!("  Expected: {expected:?}");

        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "STAGE [Verify]: output[{i}] = {got}, expected {want}"
            );
        }
        eprintln!("✓ Stage 7: Output values correct");

        // Cleanup
        ((*api).ReleaseValue.unwrap())(output);
        ((*api).ReleaseValue.unwrap())(x_tensor);
        ((*api).ReleaseValue.unwrap())(y_tensor);
        ((*api).ReleaseMemoryInfo.unwrap())(mem_info);
        ((*api).ReleaseSession.unwrap())(session);
        ((*api).ReleaseSessionOptions.unwrap())(session_options);

        let unregister_fn = (*api).UnregisterExecutionProviderLibrary
            .expect("UnregisterExecutionProviderLibrary not available");
        let status = unregister_fn(env, reg_name.as_ptr());
        check_status(api, status, "UnregisterExecutionProviderLibrary");
        eprintln!("✓ Stage 8: UnregisterExecutionProviderLibrary succeeded");

        ((*api).ReleaseEnv.unwrap())(env);
        eprintln!("\n✅ L3 END-TO-END TEST PASSED: All stages green.");
    }
}

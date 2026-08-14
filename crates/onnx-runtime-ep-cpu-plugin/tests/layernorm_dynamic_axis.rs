//! BL1 regression test: LayerNorm axis resolution with dynamic input dimensions.
//!
//! The model has inputs `X: [B, S, 4]` and `Scale: [4]` where B and S are
//! symbolic (dynamic) dimensions. LayerNorm `axis=-1` must produce:
//! - Y: same shape as X → [B, S, 4]
//! - Mean: [B, S, 1]  (dims from axis onward become 1)
//! - InvStdDev: [B, S, 1]
//!
//! **Pre-fix failure:** The old code resolved axis against the *static* shape
//! obtained by `filter_map(|d| d.as_static())`, which dropped symbolic dims.
//! For `[B, S, 4]` this collapsed to `[4]` (rank 1), so axis=-1 resolved to
//! index 0 — producing reduced shape `[1]` instead of `[B, S, 1]`. This test
//! asserts the correct runtime shapes and MUST FAIL against the pre-fix code.
//!
//! Real ORT 1.27, not a mock.

mod cdylib_resolve;
#[path = "common/ort_discovery.rs"]
mod ort_discovery;
mod ort_path;
#[path = "common/ort_session.rs"]
mod ort_session;

use std::ffi::{CStr, CString};
use std::path::PathBuf;
use std::ptr;
use std::sync::{Mutex, MutexGuard};

use onnx_genai_ort_sys as ort;

static ORT_EP_LOCK: Mutex<()> = Mutex::new(());

fn lock_ort_ep() -> MutexGuard<'static, ()> {
    ORT_EP_LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

fn find_ort_lib_dir() -> Option<PathBuf> {
    ort_discovery::find_ort_lib_dir()
}

fn find_ep_cdylib() -> Option<PathBuf> {
    cdylib_resolve::find_cpu_plugin_cdylib_optional()
}

unsafe fn get_ort_api(lib: &libloading::Library) -> *const ort::OrtApi {
    type GetApiBaseFn = unsafe extern "C" fn() -> *const ort::OrtApiBase;
    let get_api_base: libloading::Symbol<'_, GetApiBaseFn> =
        unsafe { lib.get(b"OrtGetApiBase") }.expect("OrtGetApiBase not found");
    let api_base = unsafe { get_api_base() };
    assert!(!api_base.is_null());
    let get_api = unsafe { (*api_base).GetApi }.expect("GetApi is null");
    let api = unsafe { get_api(ort::ORT_API_VERSION) };
    assert!(!api.is_null());
    api
}

unsafe fn check_status(api: *const ort::OrtApi, status: *mut ort::OrtStatus, stage: &str) {
    if !status.is_null() {
        let get_msg = unsafe { (*api).GetErrorMessage }.unwrap();
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
        check_status(api, status, "CreateTensorWithDataAsOrtValue");
        ((*api).ReleaseMemoryInfo.unwrap())(mem_info);
        val
    }
}

unsafe fn get_output_shape(api: *const ort::OrtApi, output: *const ort::OrtValue) -> Vec<i64> {
    let get_type_shape = unsafe { (*api).GetTensorTypeAndShape }.unwrap();
    let get_dims_count = unsafe { (*api).GetDimensionsCount }.unwrap();
    let get_dims = unsafe { (*api).GetDimensions }.unwrap();
    let release_info = unsafe { (*api).ReleaseTensorTypeAndShapeInfo }.unwrap();

    let mut info: *mut ort::OrtTensorTypeAndShapeInfo = ptr::null_mut();
    let status = unsafe { get_type_shape(output, &mut info) };
    unsafe { check_status(api, status, "GetTensorTypeAndShape") };

    let mut rank: usize = 0;
    let status = unsafe { get_dims_count(info, &mut rank) };
    unsafe { check_status(api, status, "GetDimensionsCount") };

    let mut dims = vec![0i64; rank];
    let status = unsafe { get_dims(info, dims.as_mut_ptr(), rank) };
    unsafe { check_status(api, status, "GetDimensions") };
    unsafe { release_info(info) };
    dims
}

/// Assert that a specific op type is assigned to "cpu_ep" via
/// `Session_GetEpGraphAssignmentInfo`. Requires `session.record_ep_graph_assignment_info=1`.
unsafe fn assert_op_assigned_to_our_ep(
    api: *const ort::OrtApi,
    session: *mut ort::OrtSession,
    expected_op: &str,
    test_label: &str,
) {
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

    let mut assignments: Vec<(String, String)> = Vec::new();
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

    let ours: Vec<&str> = assignments
        .iter()
        .filter(|(ep, _)| ep == "cpu_ep")
        .map(|(_, op)| op.as_str())
        .collect();
    eprintln!("  [{test_label}] EP assignment: ours={ours:?}");
    assert!(
        ours.contains(&expected_op),
        "[{test_label}] Expected op '{expected_op}' assigned to cpu_ep, \
         but assignment was: {assignments:?}"
    );
}

/// BL1 regression: LayerNorm axis=-1 on dynamic [B, S, H] must produce
/// Mean/InvStdDev with shape [B, S, 1], not a truncated form.
///
/// **This test fails against the pre-fix code** because the old code resolved
/// axis against the static shape `[4]` (rank 1, after dropping symbolic B, S),
/// yielding reduced shape `[1]` instead of `[B, S, 1]`.
#[test]
fn layernorm_dynamic_axis_mean_invstddev_shape() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = PathBuf::from(manifest_dir)
        .join("tests/fixtures/layer_norm_dynamic_axis/model.onnx.textproto");

    let Some(ort_lib_dir) = find_ort_lib_dir() else {
        if std::env::var("NXRT_REQUIRE_ORT_TESTS").as_deref() == Ok("1") {
            panic!("NXRT_REQUIRE_ORT_TESTS=1 but ORT not found");
        }
        eprintln!("*** SKIPPED: layernorm_dynamic_axis — ORT not found ***");
        return;
    };
    let Some(ep_lib_path) = find_ep_cdylib() else {
        if std::env::var("NXRT_REQUIRE_ORT_TESTS").as_deref() == Ok("1") {
            panic!("NXRT_REQUIRE_ORT_TESTS=1 but EP cdylib not found");
        }
        eprintln!("*** SKIPPED: layernorm_dynamic_axis — EP cdylib not found ***");
        return;
    };
    if !model_path.exists() {
        eprintln!(
            "*** SKIPPED: layernorm_dynamic_axis — fixture missing at {} ***",
            model_path.display()
        );
        return;
    }

    unsafe {
        let ort_lib_path = ort_lib_dir.join(ort_discovery::ort_lib_name());
        let lib = libloading::Library::new(&ort_lib_path).expect("load libonnxruntime");
        let api = get_ort_api(&lib);

        // Create env
        let mut env: *mut ort::OrtEnv = ptr::null_mut();
        let logid = c"nxrt_bl1_ln";
        let status =
            ((*api).CreateEnv.unwrap())(ort::ORT_LOGGING_LEVEL_WARNING, logid.as_ptr(), &mut env);
        check_status(api, status, "CreateEnv");

        // Register EP
        let reg_name = c"cpu_ep_bl1";
        let ep_path_c = ort_path::OrtPathBuf::new(&ep_lib_path);
        let status = ((*api).RegisterExecutionProviderLibrary.unwrap())(
            env,
            reg_name.as_ptr(),
            ep_path_c.as_ptr(),
        );
        check_status(api, status, "RegisterExecutionProviderLibrary");

        // Find our device
        let mut ep_devices: *const *const ort::OrtEpDevice = ptr::null();
        let mut num_devices: usize = 0;
        let status = ((*api).GetEpDevices.unwrap())(env, &mut ep_devices, &mut num_devices);
        check_status(api, status, "GetEpDevices");

        let ep_name_fn = (*api).EpDevice_EpName.unwrap();
        let mut our_device: *const ort::OrtEpDevice = ptr::null();
        for i in 0..num_devices {
            let dev = *ep_devices.add(i);
            let name_ptr = ep_name_fn(dev);
            if !name_ptr.is_null() && CStr::from_ptr(name_ptr).to_string_lossy() == "cpu_ep" {
                our_device = dev;
            }
        }
        assert!(!our_device.is_null(), "EP 'cpu_ep' not found");

        // Session
        let mut session_options: *mut ort::OrtSessionOptions = ptr::null_mut();
        let status = ((*api).CreateSessionOptions.unwrap())(&mut session_options);
        check_status(api, status, "CreateSessionOptions");

        // Disable CPU EP fallback — if our EP declines the node, the test must fail.
        let key = CString::new("session.disable_cpu_ep_fallback").unwrap();
        let val = CString::new("1").unwrap();
        let status =
            ((*api).AddSessionConfigEntry.unwrap())(session_options, key.as_ptr(), val.as_ptr());
        check_status(
            api,
            status,
            "AddSessionConfigEntry(disable_cpu_ep_fallback)",
        );

        // Record EP graph assignment info for direct assertion.
        let key = CString::new("session.record_ep_graph_assignment_info").unwrap();
        let val = CString::new("1").unwrap();
        let status =
            ((*api).AddSessionConfigEntry.unwrap())(session_options, key.as_ptr(), val.as_ptr());
        check_status(
            api,
            status,
            "AddSessionConfigEntry(record_ep_graph_assignment_info)",
        );

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

        let mut session: *mut ort::OrtSession = ptr::null_mut();
        let status =
            ort_session::create_session(api, env, session_options, &model_path, &mut session);
        check_status(api, status, "CreateSession");

        // Assert that LayerNormalization is owned by our EP, not the built-in CPU EP.
        assert_op_assigned_to_our_ep(api, session, "LayerNormalization", "layernorm_dynamic_axis");

        // Run with concrete shape [2, 3, 4] (B=2, S=3, H=4)
        let mut x_data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
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
        check_status(api, status, "Run(LayerNorm dynamic axis)");

        // All outputs must be non-null
        for (i, out) in outputs.iter().enumerate() {
            assert!(!out.is_null(), "LayerNorm output[{i}] is null");
        }

        // THE CRITICAL ASSERTION: shapes
        let y_shape = get_output_shape(api, outputs[0]);
        let mean_shape = get_output_shape(api, outputs[1]);
        let inv_shape = get_output_shape(api, outputs[2]);

        eprintln!("  Y shape: {y_shape:?}");
        eprintln!("  Mean shape: {mean_shape:?}");
        eprintln!("  InvStdDev shape: {inv_shape:?}");

        // Y must have the full input shape [2, 3, 4]
        assert_eq!(
            y_shape,
            vec![2, 3, 4],
            "Y shape should be [2, 3, 4], got {y_shape:?}"
        );

        // Mean and InvStdDev must have shape [2, 3, 1] — the last dim
        // (the normalized axis) becomes 1.
        // Pre-fix bug: axis resolved against truncated rank → wrong shape.
        assert_eq!(
            mean_shape,
            vec![2, 3, 1],
            "Mean shape should be [2, 3, 1] (axis=-1 on rank-3 input), got {mean_shape:?}. \
             BL1 bug: axis was resolved against truncated static shape."
        );
        assert_eq!(
            inv_shape,
            vec![2, 3, 1],
            "InvStdDev shape should be [2, 3, 1] (axis=-1 on rank-3 input), got {inv_shape:?}. \
             BL1 bug: axis was resolved against truncated static shape."
        );

        // Verify Mean values: row means for groups of 4
        // Row 0: mean([1,2,3,4])=2.5, Row 1: mean([5,6,7,8])=6.5, etc.
        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(outputs[1], &mut data_ptr);
        check_status(api, status, "GetTensorMutableData(Mean)");
        let mean_data = std::slice::from_raw_parts(data_ptr as *const f32, 6);
        eprintln!("  Mean values: {mean_data:?}");
        let expected_means: [f32; 6] = [2.5, 6.5, 10.5, 14.5, 18.5, 22.5];
        for (i, (&got, &exp)) in mean_data.iter().zip(expected_means.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-4, "Mean[{i}]={got}, expected {exp}");
        }

        // Teardown
        ((*api).ReleaseSession.unwrap())(session);
        ((*api).ReleaseSessionOptions.unwrap())(session_options);
        let status = ((*api).UnregisterExecutionProviderLibrary.unwrap())(env, reg_name.as_ptr());
        check_status(api, status, "UnregisterExecutionProviderLibrary");
        ((*api).ReleaseEnv.unwrap())(env);
    }
}

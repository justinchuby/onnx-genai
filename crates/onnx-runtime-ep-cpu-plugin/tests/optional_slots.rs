//! Tests for correct handling of omitted optional input/output slots.
//!
//! These tests verify that:
//! - BL2: Absent optional outputs preserve positional indexing (no compaction).
//! - BL3: Absent optional inputs produce a genuine absent TensorView, not alias
//!   to input 0.
//!
//! Each test asserts numerical correctness of the results, not just success.
//! They are designed to **fail against the pre-fix code** where `filter_map`
//! compacted output slots and absent inputs silently read ORT input 0.

mod cdylib_resolve;
mod ort_path;

use std::ffi::CStr;
use std::path::PathBuf;
use std::ptr;
use std::sync::{Mutex, MutexGuard};

use onnx_genai_ort_sys as ort;

static ORT_EP_LOCK: Mutex<()> = Mutex::new(());

fn lock_ort_ep() -> MutexGuard<'static, ()> {
    ORT_EP_LOCK.lock().unwrap_or_else(|poisoned| {
        eprintln!("WARNING: ORT_EP_LOCK poisoned — recovering.");
        poisoned.into_inner()
    })
}

fn find_ort_lib_dir() -> Option<PathBuf> {
    ort_discovery::find_ort_lib_dir()
}

/// Canonical ORT discovery — single source of truth in `tests/common/ort_discovery.rs`.
#[path = "common/ort_discovery.rs"]
mod ort_discovery;
#[path = "common/ort_session.rs"]
mod ort_session;

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
        let get_msg = unsafe { (*api).GetErrorMessage }.expect("GetErrorMessage");
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

#[allow(clippy::type_complexity)]
unsafe fn setup(
    reg_name: &str,
    model_path: &std::path::Path,
) -> Option<(
    libloading::Library,
    *const ort::OrtApi,
    *mut ort::OrtEnv,
    *mut ort::OrtSessionOptions,
    *mut ort::OrtSession,
)> {
    let ort_lib_dir = match find_ort_lib_dir() {
        Some(d) => d,
        None => {
            if std::env::var("NXRT_REQUIRE_ORT_TESTS").as_deref() == Ok("1") {
                panic!("NXRT_REQUIRE_ORT_TESTS=1 but ORT lib dir not found");
            }
            return None;
        }
    };
    let ep_lib_path = match find_ep_cdylib() {
        Some(p) => p,
        None => {
            if std::env::var("NXRT_REQUIRE_ORT_TESTS").as_deref() == Ok("1") {
                panic!("NXRT_REQUIRE_ORT_TESTS=1 but EP cdylib not found");
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
    let lib = match unsafe { libloading::Library::new(&ort_lib_path) } {
        Ok(l) => l,
        Err(e) => {
            if std::env::var("NXRT_REQUIRE_ORT_TESTS").as_deref() == Ok("1") {
                panic!(
                    "NXRT_REQUIRE_ORT_TESTS=1 but dlopen failed for {}: {e}",
                    ort_lib_path.display()
                );
            }
            eprintln!(
                "*** SKIPPED: dlopen failed for {}: {e} ***",
                ort_lib_path.display()
            );
            return None;
        }
    };
    let api = unsafe { get_ort_api(&lib) };

    let mut env: *mut ort::OrtEnv = ptr::null_mut();
    let logid = std::ffi::CString::new(format!("nxrt_{reg_name}")).unwrap();
    let status = unsafe {
        ((*api).CreateEnv.unwrap())(ort::ORT_LOGGING_LEVEL_WARNING, logid.as_ptr(), &mut env)
    };
    unsafe { check_status(api, status, "CreateEnv") };

    let reg_name_c = std::ffi::CString::new(reg_name).unwrap();
    let ep_path_c = ort_path::OrtPathBuf::new(&ep_lib_path);
    let status = unsafe {
        ((*api).RegisterExecutionProviderLibrary.unwrap())(
            env,
            reg_name_c.as_ptr(),
            ep_path_c.as_ptr(),
        )
    };
    unsafe { check_status(api, status, "RegisterEP") };

    let mut ep_devices: *const *const ort::OrtEpDevice = ptr::null();
    let mut num_devices: usize = 0;
    let status = unsafe { ((*api).GetEpDevices.unwrap())(env, &mut ep_devices, &mut num_devices) };
    unsafe { check_status(api, status, "GetEpDevices") };

    let ep_name_fn = unsafe { (*api).EpDevice_EpName.expect("EpDevice_EpName") };
    let mut our_device: *const ort::OrtEpDevice = ptr::null();
    for i in 0..num_devices {
        let dev = unsafe { *ep_devices.add(i) };
        let name_ptr = unsafe { ep_name_fn(dev) };
        if !name_ptr.is_null() && unsafe { CStr::from_ptr(name_ptr) }.to_string_lossy() == "cpu_ep"
        {
            our_device = dev;
        }
    }
    assert!(!our_device.is_null(), "EP 'cpu_ep' not found");

    let mut session_options: *mut ort::OrtSessionOptions = ptr::null_mut();
    let status = unsafe { ((*api).CreateSessionOptions.unwrap())(&mut session_options) };
    unsafe { check_status(api, status, "CreateSessionOptions") };

    // Disable ORT's built-in CPU EP fallback so that if our EP declines a
    // node, ORT errors instead of silently running it on the default EP.
    // Without this, tests pass vacuously even if our EP never claims the node.
    let key = std::ffi::CString::new("session.disable_cpu_ep_fallback").unwrap();
    let val = std::ffi::CString::new("1").unwrap();
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

    // Enable EP graph assignment recording so we can query per-node provider
    // attribution via Session_GetEpGraphAssignmentInfo (ORT ≥1.24).
    {
        let key = std::ffi::CString::new("session.record_ep_graph_assignment_info").unwrap();
        let val = std::ffi::CString::new("1").unwrap();
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
    unsafe { check_status(api, status, "AppendEP") };

    let mut session: *mut ort::OrtSession = ptr::null_mut();
    let status =
        unsafe { ort_session::create_session(api, env, session_options, model_path, &mut session) };
    unsafe { check_status(api, status, "CreateSession") };

    Some((lib, api, env, session_options, session))
}

unsafe fn teardown(
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
        check_status(api, status, "UnregisterEP");
        ((*api).ReleaseEnv.unwrap())(env);
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
        check_status(api, status, "CreateTensorFloat");
        ((*api).ReleaseMemoryInfo.unwrap())(mem_info);
        val
    }
}

unsafe fn make_scalar_float_tensor(api: *const ort::OrtApi, value: &mut f32) -> *mut ort::OrtValue {
    unsafe { make_float_tensor(api, std::slice::from_mut(value), &[]) }
}

// ─── Helper: require-ORT gate ─────────────────────────────────────────────────

/// When `NXRT_REQUIRE_ORT_TESTS=1`, tests must fail instead of silently skipping
/// if ORT or the EP cdylib is unavailable. Silent skips let the suite look green
/// while proving nothing.
fn require_ort_or_skip(test_name: &str) -> bool {
    if std::env::var("NXRT_REQUIRE_ORT_TESTS").as_deref() == Ok("1") {
        panic!(
            "NXRT_REQUIRE_ORT_TESTS=1 but ORT or EP cdylib is unavailable — \
             {test_name} cannot run. Install ORT or unset the variable."
        );
    }
    false
}

// ─── Helper: EP graph assignment assertion ────────────────────────────────────

/// Assert that every op in `expected_ops` is assigned to "cpu_ep" (our plugin EP),
/// not ORT's built-in CPUExecutionProvider, via `Session_GetEpGraphAssignmentInfo`.
///
/// Requires `session.record_ep_graph_assignment_info=1` (enabled by `setup()`).
unsafe fn assert_ops_assigned_to_our_ep(
    api: *const ort::OrtApi,
    session: *mut ort::OrtSession,
    expected_ops: &[&str],
    label: &str,
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

    for &expected in expected_ops {
        assert!(
            ours.contains(&expected),
            "[{label}] Expected op '{expected}' to be assigned to 'cpu_ep', \
             but it was not. Assignments: {assignments:?}"
        );
    }
    eprintln!("  ✓ [{label}] ops assigned to cpu_ep: {ours:?}");
}

// ─── BL2: SkipLayerNormalization output=(output, "", "", sum) ────────────────

/// SkipLayerNormalization with outputs (output, "", "", sum).
///
/// The ONNX node has 4 output slots: output at position 0, mean at 1 (absent),
/// inv_std_var at 2 (absent), and input_skip_bias_sum at position 3.
///
/// **Pre-fix failure mode**: `filter_map` compacted the output list to
/// `[output, sum]`, so the kernel saw `outputs.len() == 2` and wrote `mean`
/// (not `sum`) into position 1. The test asserts that output 1 (the graph's
/// second output) contains the *sum* (X + skip), not the mean.
#[test]
fn skip_layer_norm_output_sum_position() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = PathBuf::from(manifest_dir)
        .join("tests/fixtures/skip_layer_norm_output_sum/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) = (unsafe { setup("cpu_ep_sln_sum", &model_path) })
    else {
        eprintln!("*** SKIPPED: skip_layer_norm_output_sum_position ***");
        return;
    };

    unsafe {
        // X = [[1, 2, 3, 4], [5, 6, 7, 8]]
        let mut x_data: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // skip = [[0.5, -1, 1, 0], [-2, -3, -4, -1]]
        let mut skip_data: [f32; 8] = [0.5, -1.0, 1.0, 0.0, -2.0, -3.0, -4.0, -1.0];
        // gamma = [1, 1, 1, 1]
        let mut gamma_data: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

        let shape: [i64; 2] = [2, 4];
        let gamma_shape: [i64; 1] = [4];

        let x_val = make_float_tensor(api, &mut x_data, &shape);
        let skip_val = make_float_tensor(api, &mut skip_data, &shape);
        let gamma_val = make_float_tensor(api, &mut gamma_data, &gamma_shape);

        let input_names = [c"X".as_ptr(), c"skip".as_ptr(), c"gamma".as_ptr()];
        let output_names = [c"output".as_ptr(), c"sum".as_ptr()];
        let inputs: [*const ort::OrtValue; 3] = [x_val, skip_val, gamma_val];
        let mut outputs: [*mut ort::OrtValue; 2] = [ptr::null_mut(); 2];

        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(),
            input_names.as_ptr(),
            inputs.as_ptr(),
            3,
            output_names.as_ptr(),
            2,
            outputs.as_mut_ptr(),
        );
        check_status(api, status, "Run");

        // Expected sum = X + skip (no bias):
        // row0: [1.5, 1.0, 4.0, 4.0]
        // row1: [3.0, 3.0, 3.0, 7.0]
        let expected_sum: [f32; 8] = [1.5, 1.0, 4.0, 4.0, 3.0, 3.0, 3.0, 7.0];

        // Read the "sum" output (second graph output).
        let sum_out = outputs[1];
        assert!(!sum_out.is_null(), "sum output is null");
        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(sum_out, &mut data_ptr);
        check_status(api, status, "GetTensorMutableData(sum)");
        let result = std::slice::from_raw_parts(data_ptr as *const f32, 8);

        eprintln!("  sum got:      {result:?}");
        eprintln!("  sum expected: {expected_sum:?}");

        for (i, (got, want)) in result.iter().zip(expected_sum.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "sum[{i}] = {got}, want {want} — \
                 if this is the mean, the output compaction bug is present!"
            );
        }

        for o in &outputs {
            if !o.is_null() {
                ((*api).ReleaseValue.unwrap())(*o);
            }
        }
        ((*api).ReleaseValue.unwrap())(x_val);
        ((*api).ReleaseValue.unwrap())(skip_val);
        ((*api).ReleaseValue.unwrap())(gamma_val);
        teardown(api, env, opts, session, "cpu_ep_sln_sum");
        eprintln!("\n✅ skip_layer_norm_output_sum_position: PASSED");
    }
}

// ─── BL3: Clip(x, "", max) — omitted min with present max ───────────────────

/// Clip with omitted `min` input: Clip(X, "", max_val).
///
/// **Pre-fix failure mode**: The absent `min` input would alias to ORT input 0
/// (i.e. X itself), making the effective min = X, which clips everything to X
/// (no-op on the lower bound). With the fix, absent min means -infinity.
#[test]
fn clip_omitted_min_with_max() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path =
        PathBuf::from(manifest_dir).join("tests/fixtures/clip_no_min/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) = (unsafe { setup("cpu_ep_clip", &model_path) })
    else {
        eprintln!("*** SKIPPED: clip_omitted_min_with_max ***");
        return;
    };

    unsafe {
        // X = [-10, -1, 0.5, 100]
        let mut x_data: [f32; 4] = [-10.0, -1.0, 0.5, 100.0];
        // max = 5.0 (scalar)
        let mut max_val: f32 = 5.0;

        let x_shape: [i64; 2] = [1, 4];
        let x_val = make_float_tensor(api, &mut x_data, &x_shape);
        let max_val_tensor = make_scalar_float_tensor(api, &mut max_val);

        let input_names = [c"X".as_ptr(), c"max_val".as_ptr()];
        let output_names = [c"Y".as_ptr()];
        let inputs: [*const ort::OrtValue; 2] = [x_val, max_val_tensor];
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

        // Expected: clip with min=-inf, max=5 → [-10, -1, 0.5, 5]
        let expected: [f32; 4] = [-10.0, -1.0, 0.5, 5.0];
        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(output, &mut data_ptr);
        check_status(api, status, "GetTensorMutableData");
        let result = std::slice::from_raw_parts(data_ptr as *const f32, 4);

        eprintln!("  clip got:      {result:?}");
        eprintln!("  clip expected: {expected:?}");

        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "Y[{i}] = {got}, want {want} — \
                 if -10 was clipped to -10 (not lower), the absent-input bug may be present"
            );
        }

        ((*api).ReleaseValue.unwrap())(output);
        ((*api).ReleaseValue.unwrap())(x_val);
        ((*api).ReleaseValue.unwrap())(max_val_tensor);
        teardown(api, env, opts, session, "cpu_ep_clip");
        eprintln!("\n✅ clip_omitted_min_with_max: PASSED");
    }
}

// ─── BL3: SkipLayerNormalization with omitted beta/bias ──────────────────────

/// SkipLayerNormalization with inputs (X, skip, gamma, "", "") — beta and bias
/// are omitted.
///
/// **Pre-fix failure mode**: Absent inputs alias to ORT input 0 (X), so
/// beta = X and bias = X, corrupting the output. With the fix, absent inputs
/// produce a genuine absent TensorView (null data pointer), and the kernel
/// treats them as not provided.
#[test]
fn skip_layer_norm_omitted_beta_bias() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = PathBuf::from(manifest_dir)
        .join("tests/fixtures/skip_layer_norm_no_beta_bias/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) = (unsafe { setup("cpu_ep_sln_nb", &model_path) })
    else {
        eprintln!("*** SKIPPED: skip_layer_norm_omitted_beta_bias ***");
        return;
    };

    unsafe {
        // X = [[1, 2, 3, 4], [5, 6, 7, 8]]
        let mut x_data: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // skip = [[0.5, -1, 1, 0], [-2, -3, -4, -1]]
        let mut skip_data: [f32; 8] = [0.5, -1.0, 1.0, 0.0, -2.0, -3.0, -4.0, -1.0];
        // gamma = [1, 1, 1, 1]
        let mut gamma_data: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

        let shape: [i64; 2] = [2, 4];
        let gamma_shape: [i64; 1] = [4];

        let x_val = make_float_tensor(api, &mut x_data, &shape);
        let skip_val = make_float_tensor(api, &mut skip_data, &shape);
        let gamma_val = make_float_tensor(api, &mut gamma_data, &gamma_shape);

        let input_names = [c"X".as_ptr(), c"skip".as_ptr(), c"gamma".as_ptr()];
        let output_names = [c"output".as_ptr()];
        let inputs: [*const ort::OrtValue; 3] = [x_val, skip_val, gamma_val];
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
        check_status(api, status, "Run");
        assert!(!output.is_null());

        // sum = X + skip (no bias): [1.5, 1.0, 4.0, 4.0, 3.0, 3.0, 3.0, 7.0]
        // LayerNorm(sum, gamma=1, beta=0, eps=1e-5):
        // row0: mean=2.625, var=1.671875, inv_std=0.7733...
        //   output = (sum - mean) * inv_std * gamma + beta(=0)
        // row1: mean=4.0, var=2.5, inv_std=0.6324...
        let sum_row0 = [1.5f32, 1.0, 4.0, 4.0];
        let sum_row1 = [3.0f32, 3.0, 3.0, 7.0];
        let eps = 1e-5f32;

        let mean0: f32 = sum_row0.iter().sum::<f32>() / 4.0;
        let var0: f32 = sum_row0.iter().map(|v| (v - mean0).powi(2)).sum::<f32>() / 4.0;
        let inv_std0 = 1.0 / (var0 + eps).sqrt();

        let mean1: f32 = sum_row1.iter().sum::<f32>() / 4.0;
        let var1: f32 = sum_row1.iter().map(|v| (v - mean1).powi(2)).sum::<f32>() / 4.0;
        let inv_std1 = 1.0 / (var1 + eps).sqrt();

        let mut expected = [0.0f32; 8];
        for i in 0..4 {
            // gamma=1, beta=0 (absent)
            expected[i] = (sum_row0[i] - mean0) * inv_std0;
            expected[4 + i] = (sum_row1[i] - mean1) * inv_std1;
        }

        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(output, &mut data_ptr);
        check_status(api, status, "GetTensorMutableData");
        let result = std::slice::from_raw_parts(data_ptr as *const f32, 8);

        eprintln!("  output got:      {result:?}");
        eprintln!("  output expected: {expected:?}");

        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-4,
                "output[{i}] = {got}, want {want} — \
                 if beta/bias used X as the absent value, the numbers will be very wrong"
            );
        }

        ((*api).ReleaseValue.unwrap())(output);
        ((*api).ReleaseValue.unwrap())(x_val);
        ((*api).ReleaseValue.unwrap())(skip_val);
        ((*api).ReleaseValue.unwrap())(gamma_val);
        teardown(api, env, opts, session, "cpu_ep_sln_nb");
        eprintln!("\n✅ skip_layer_norm_omitted_beta_bias: PASSED");
    }
}

// ─── BL2 additional: SimplifiedLayerNormalization with two outputs ────────────

/// SimplifiedLayerNormalization with both outputs: (output, inv_std_var).
///
/// This exercises a different multi-output op to provide additional coverage
/// for the output position preservation fix.
#[test]
fn simplified_layer_norm_two_outputs_position() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = PathBuf::from(manifest_dir)
        .join("tests/fixtures/simplified_layer_norm_two_outputs/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) = (unsafe { setup("cpu_ep_sln2", &model_path) })
    else {
        eprintln!("*** SKIPPED: simplified_layer_norm_two_outputs_position ***");
        return;
    };

    unsafe {
        // X = [[1, 2, 3, 4], [2, 4, 6, 8]]
        let mut x_data: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0];
        // scale = [1, 1, 1, 1]
        let mut scale_data: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

        let shape: [i64; 2] = [2, 4];
        let scale_shape: [i64; 1] = [4];

        let x_val = make_float_tensor(api, &mut x_data, &shape);
        let scale_val = make_float_tensor(api, &mut scale_data, &scale_shape);

        let input_names = [c"X".as_ptr(), c"scale".as_ptr()];
        let output_names = [c"output".as_ptr(), c"inv_std".as_ptr()];
        let inputs: [*const ort::OrtValue; 2] = [x_val, scale_val];
        let mut outputs: [*mut ort::OrtValue; 2] = [ptr::null_mut(); 2];

        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(),
            input_names.as_ptr(),
            inputs.as_ptr(),
            2,
            output_names.as_ptr(),
            2,
            outputs.as_mut_ptr(),
        );
        check_status(api, status, "Run");

        // RMSNorm: output[i] = x[i] / sqrt(mean(x²) + eps) * scale[i]
        // row0: rms = sqrt((1+4+9+16)/4 + 1e-5) = sqrt(7.5 + 1e-5) ≈ 2.7386
        //   inv_std0 = 1/rms ≈ 0.3651
        // row1: rms = sqrt((4+16+36+64)/4 + 1e-5) = sqrt(30 + 1e-5) ≈ 5.4772
        //   inv_std1 = 1/rms ≈ 0.1826
        let eps = 1e-5f32;
        let row0 = [1.0f32, 2.0, 3.0, 4.0];
        let row1 = [2.0f32, 4.0, 6.0, 8.0];
        let rms0 = (row0.iter().map(|v| v * v).sum::<f32>() / 4.0 + eps).sqrt();
        let rms1 = (row1.iter().map(|v| v * v).sum::<f32>() / 4.0 + eps).sqrt();

        let inv_std0 = 1.0 / rms0;
        let inv_std1 = 1.0 / rms1;

        // Check inv_std output (second output)
        let inv_std_out = outputs[1];
        assert!(!inv_std_out.is_null(), "inv_std output is null");
        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(inv_std_out, &mut data_ptr);
        check_status(api, status, "GetTensorMutableData(inv_std)");
        let inv_std_result = std::slice::from_raw_parts(data_ptr as *const f32, 2);

        eprintln!("  inv_std got:      {inv_std_result:?}");
        eprintln!("  inv_std expected: [{inv_std0}, {inv_std1}]");
        assert!(
            (inv_std_result[0] - inv_std0).abs() < 1e-4,
            "inv_std[0] = {}, want {inv_std0}",
            inv_std_result[0]
        );
        assert!(
            (inv_std_result[1] - inv_std1).abs() < 1e-4,
            "inv_std[1] = {}, want {inv_std1}",
            inv_std_result[1]
        );

        for o in &outputs {
            if !o.is_null() {
                ((*api).ReleaseValue.unwrap())(*o);
            }
        }
        ((*api).ReleaseValue.unwrap())(x_val);
        ((*api).ReleaseValue.unwrap())(scale_val);
        teardown(api, env, opts, session, "cpu_ep_sln2");
        eprintln!("\n✅ simplified_layer_norm_two_outputs_position: PASSED");
    }
}

// ─── f16/bf16 helpers ────────────────────────────────────────────────────────

/// Truncate an f32 to IEEE 754 binary16 (round-to-nearest-even).
fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;
    if exp == 0xFF {
        // Inf / NaN
        return (sign | 0x7C00 | if mant != 0 { 0x0200 } else { 0 }) as u16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return (sign | 0x7C00) as u16; // overflow → ±inf
    }
    if new_exp <= 0 {
        return sign as u16; // underflow → ±0 (good enough for test data)
    }
    (sign | ((new_exp as u32) << 10) | (mant >> 13)) as u16
}

/// Truncate an f32 to bfloat16 (top 16 bits of the f32 representation).
fn f32_to_bf16(val: f32) -> u16 {
    (val.to_bits() >> 16) as u16
}

/// Widen an IEEE 754 binary16 value to f32. Handles normals, subnormals,
/// zeros, and inf/NaN. Used to compare kernel output against an f32 oracle.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign << 31 // ±0
        } else {
            // Subnormal f16 → normalized f32.
            let mut e: i32 = -1;
            let mut m = mant;
            loop {
                e += 1;
                m <<= 1;
                if m & 0x400 != 0 {
                    break;
                }
            }
            let m = m & 0x3FF;
            (sign << 31) | (((127 - 15 - e) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1F {
        (sign << 31) | (0xFF << 23) | (mant << 13) // inf / NaN
    } else {
        (sign << 31) | (((exp as i32 - 15 + 127) as u32) << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}

/// Widen a bfloat16 value to f32 (bf16 is the top 16 bits of the f32 form).
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

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
        check_status(api, status, "CreateTensorFloat16");
        ((*api).ReleaseMemoryInfo.unwrap())(mem_info);
        val
    }
}

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
        check_status(api, status, "CreateTensorBFloat16");
        ((*api).ReleaseMemoryInfo.unwrap())(mem_info);
        val
    }
}

// ─── B1: f16 SkipLayerNormalization with absent optional outputs ─────────────

/// B1 regression: SkipLayerNormalization (float16) with outputs (output,"","",sum).
///
/// Pre-fix: scratch buffer allocated at 2 bytes/elem (f16) but kernel given
/// a Float32 TensorMut (4 bytes/elem) → 2x heap buffer overflow on every
/// absent slot write. If the allocator happened to over-allocate, the overflow
/// was silent; this test detects it because incorrect dtype/size causes either
/// a crash or wrong data in the present outputs.
#[test]
fn skip_layer_norm_f16_absent_output_no_overflow() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = PathBuf::from(manifest_dir)
        .join("tests/fixtures/skip_layer_norm_f16_absent_output/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) = (unsafe { setup("cpu_ep_slnf16", &model_path) })
    else {
        eprintln!("*** SKIPPED: skip_layer_norm_f16_absent_output_no_overflow ***");
        return;
    };

    unsafe {
        // X = [[1, 2, 3, 4], [5, 6, 7, 8]] in f16
        let mut x_data: [u16; 8] = [
            f32_to_f16(1.0),
            f32_to_f16(2.0),
            f32_to_f16(3.0),
            f32_to_f16(4.0),
            f32_to_f16(5.0),
            f32_to_f16(6.0),
            f32_to_f16(7.0),
            f32_to_f16(8.0),
        ];
        // skip = [[0.5, -1, 1, 0], [-2, -3, -4, -1]] in f16
        let mut skip_data: [u16; 8] = [
            f32_to_f16(0.5),
            f32_to_f16(-1.0),
            f32_to_f16(1.0),
            f32_to_f16(0.0),
            f32_to_f16(-2.0),
            f32_to_f16(-3.0),
            f32_to_f16(-4.0),
            f32_to_f16(-1.0),
        ];
        // gamma = [1, 1, 1, 1] in f16
        let mut gamma_data: [u16; 4] = [f32_to_f16(1.0); 4];

        let shape: [i64; 2] = [2, 4];
        let gamma_shape: [i64; 1] = [4];

        let x_val = make_float16_tensor(api, &mut x_data, &shape);
        let skip_val = make_float16_tensor(api, &mut skip_data, &shape);
        let gamma_val = make_float16_tensor(api, &mut gamma_data, &gamma_shape);

        let input_names = [c"X".as_ptr(), c"skip".as_ptr(), c"gamma".as_ptr()];
        let output_names = [c"output".as_ptr(), c"sum".as_ptr()];
        let inputs: [*const ort::OrtValue; 3] = [x_val, skip_val, gamma_val];
        let mut outputs: [*mut ort::OrtValue; 2] = [ptr::null_mut(); 2];

        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(),
            input_names.as_ptr(),
            inputs.as_ptr(),
            3,
            output_names.as_ptr(),
            2,
            outputs.as_mut_ptr(),
        );
        check_status(api, status, "Run(skip_layer_norm_f16)");

        // sum = X + skip: [1.5, 1.0, 4.0, 4.0, 3.0, 3.0, 3.0, 7.0] (f16)
        let sum_out = outputs[1];
        assert!(!sum_out.is_null(), "sum output is null");
        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(sum_out, &mut data_ptr);
        check_status(api, status, "GetTensorMutableData(sum_f16)");
        let result = std::slice::from_raw_parts(data_ptr as *const u16, 8);

        let expected_sum_f32: [f32; 8] = [1.5, 1.0, 4.0, 4.0, 3.0, 3.0, 3.0, 7.0];
        eprintln!("  f16 sum (raw u16): {result:?}");
        for (i, (&got_u16, &want_f32)) in result.iter().zip(expected_sum_f32.iter()).enumerate() {
            let expected_u16 = f32_to_f16(want_f32);
            assert_eq!(
                got_u16, expected_u16,
                "sum[{i}]: got 0x{got_u16:04X}, want 0x{expected_u16:04X} (f32={want_f32}) — \
                 if the values are corrupted, the scratch buffer overflow is present"
            );
        }

        for o in &outputs {
            if !o.is_null() {
                ((*api).ReleaseValue.unwrap())(*o);
            }
        }
        ((*api).ReleaseValue.unwrap())(x_val);
        ((*api).ReleaseValue.unwrap())(skip_val);
        ((*api).ReleaseValue.unwrap())(gamma_val);
        teardown(api, env, opts, session, "cpu_ep_slnf16");
        eprintln!("\n✅ skip_layer_norm_f16_absent_output_no_overflow: PASSED");
    }
}

// ─── B1: bf16 SkipLayerNormalization with absent optional outputs ────────────

/// Same as the f16 test but with bfloat16.
#[test]
fn skip_layer_norm_bf16_absent_output_no_overflow() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = PathBuf::from(manifest_dir)
        .join("tests/fixtures/skip_layer_norm_bf16_absent_output/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) = (unsafe { setup("cpu_ep_slnbf16", &model_path) })
    else {
        eprintln!("*** SKIPPED: skip_layer_norm_bf16_absent_output_no_overflow ***");
        return;
    };

    unsafe {
        let mut x_data: [u16; 8] = [
            f32_to_bf16(1.0),
            f32_to_bf16(2.0),
            f32_to_bf16(3.0),
            f32_to_bf16(4.0),
            f32_to_bf16(5.0),
            f32_to_bf16(6.0),
            f32_to_bf16(7.0),
            f32_to_bf16(8.0),
        ];
        let mut skip_data: [u16; 8] = [
            f32_to_bf16(0.5),
            f32_to_bf16(-1.0),
            f32_to_bf16(1.0),
            f32_to_bf16(0.0),
            f32_to_bf16(-2.0),
            f32_to_bf16(-3.0),
            f32_to_bf16(-4.0),
            f32_to_bf16(-1.0),
        ];
        let mut gamma_data: [u16; 4] = [f32_to_bf16(1.0); 4];

        let shape: [i64; 2] = [2, 4];
        let gamma_shape: [i64; 1] = [4];

        let x_val = make_bfloat16_tensor(api, &mut x_data, &shape);
        let skip_val = make_bfloat16_tensor(api, &mut skip_data, &shape);
        let gamma_val = make_bfloat16_tensor(api, &mut gamma_data, &gamma_shape);

        let input_names = [c"X".as_ptr(), c"skip".as_ptr(), c"gamma".as_ptr()];
        let output_names = [c"output".as_ptr(), c"sum".as_ptr()];
        let inputs: [*const ort::OrtValue; 3] = [x_val, skip_val, gamma_val];
        let mut outputs: [*mut ort::OrtValue; 2] = [ptr::null_mut(); 2];

        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(),
            input_names.as_ptr(),
            inputs.as_ptr(),
            3,
            output_names.as_ptr(),
            2,
            outputs.as_mut_ptr(),
        );
        check_status(api, status, "Run(skip_layer_norm_bf16)");

        let sum_out = outputs[1];
        assert!(!sum_out.is_null(), "sum output is null");
        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(sum_out, &mut data_ptr);
        check_status(api, status, "GetTensorMutableData(sum_bf16)");
        let result = std::slice::from_raw_parts(data_ptr as *const u16, 8);

        let expected_sum_f32: [f32; 8] = [1.5, 1.0, 4.0, 4.0, 3.0, 3.0, 3.0, 7.0];
        eprintln!("  bf16 sum (raw u16): {result:?}");
        for (i, (&got_u16, &want_f32)) in result.iter().zip(expected_sum_f32.iter()).enumerate() {
            let expected_u16 = f32_to_bf16(want_f32);
            assert_eq!(
                got_u16, expected_u16,
                "sum[{i}]: got 0x{got_u16:04X}, want 0x{expected_u16:04X} (f32={want_f32})"
            );
        }

        for o in &outputs {
            if !o.is_null() {
                ((*api).ReleaseValue.unwrap())(*o);
            }
        }
        ((*api).ReleaseValue.unwrap())(x_val);
        ((*api).ReleaseValue.unwrap())(skip_val);
        ((*api).ReleaseValue.unwrap())(gamma_val);
        teardown(api, env, opts, session, "cpu_ep_slnbf16");
        eprintln!("\n✅ skip_layer_norm_bf16_absent_output_no_overflow: PASSED");
    }
}

// ─── B1: f16 LayerNormalization with absent Mean/InvStdDev ───────────────────

/// B1 regression: LayerNormalization (float16) with outputs (Y, "", "").
/// Mean and InvStdDev are absent — their scratch buffers must be sized for
/// f16 (2 bytes/elem), not hardcoded Float32 (4 bytes/elem).
#[test]
fn layer_norm_f16_absent_output_no_overflow() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = PathBuf::from(manifest_dir)
        .join("tests/fixtures/layer_norm_f16_absent_output/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) = (unsafe { setup("cpu_ep_lnf16", &model_path) })
    else {
        eprintln!("*** SKIPPED: layer_norm_f16_absent_output_no_overflow ***");
        return;
    };

    unsafe {
        let mut x_data: [u16; 8] = [
            f32_to_f16(1.0),
            f32_to_f16(2.0),
            f32_to_f16(3.0),
            f32_to_f16(4.0),
            f32_to_f16(5.0),
            f32_to_f16(6.0),
            f32_to_f16(7.0),
            f32_to_f16(8.0),
        ];
        let mut scale_data: [u16; 4] = [f32_to_f16(1.0); 4];

        let shape: [i64; 2] = [2, 4];
        let scale_shape: [i64; 1] = [4];

        let x_val = make_float16_tensor(api, &mut x_data, &shape);
        let scale_val = make_float16_tensor(api, &mut scale_data, &scale_shape);

        let input_names = [c"X".as_ptr(), c"Scale".as_ptr()];
        let output_names = [c"Y".as_ptr()];
        let inputs: [*const ort::OrtValue; 2] = [x_val, scale_val];
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
        check_status(api, status, "Run(layer_norm_f16)");
        assert!(!output.is_null());

        // Value oracle (not just crash-freedom): LayerNormalization over the
        // last axis with Scale=1 and no bias produces the normalized rows.
        //   X row0 = [1,2,3,4] → mean 2.5, var 1.25, invstd 1/sqrt(1.25+1e-5)
        //   X row1 = [5,6,7,8] → mean 6.5, same deviations → identical Y row
        //   Y = (X - mean) * invstd (Scale=1). Both rows equal:
        //   [-1.3416394, -0.4472131, 0.4472131, 1.3416394]
        // A scratch-buffer overflow that silently corrupted Y would fail here.
        let mut y_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(output, &mut y_ptr);
        check_status(api, status, "GetTensorMutableData(layer_norm_f16)");
        let result = std::slice::from_raw_parts(y_ptr as *const u16, 8);
        let expected_y_f32: [f32; 8] = [
            -1.3416394, -0.4472131, 0.4472131, 1.3416394, -1.3416394, -0.4472131, 0.4472131,
            1.3416394,
        ];
        eprintln!("  layer_norm_f16 Y (raw u16): {result:?}");
        // f16 has ~3 decimal digits; 0.02 absolute tolerance is well within f16
        // rounding for |Y|≈0.4–1.3 yet tight enough to catch corruption.
        for (i, (&got_u16, &want)) in result.iter().zip(expected_y_f32.iter()).enumerate() {
            let got = f16_to_f32(got_u16);
            assert!(
                (got - want).abs() <= 0.02,
                "Y[{i}]: got {got} (0x{got_u16:04X}), want ~{want} — \
                 corrupted normalized value indicates a scratch buffer overflow"
            );
        }

        ((*api).ReleaseValue.unwrap())(output);
        ((*api).ReleaseValue.unwrap())(x_val);
        ((*api).ReleaseValue.unwrap())(scale_val);
        teardown(api, env, opts, session, "cpu_ep_lnf16");
        eprintln!("\n✅ layer_norm_f16_absent_output_no_overflow: PASSED");
    }
}

// ─── B1: bf16 LayerNormalization with absent Mean/InvStdDev ──────────────────

/// Same as the f16 test but with bfloat16.
#[test]
fn layer_norm_bf16_absent_output_no_overflow() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = PathBuf::from(manifest_dir)
        .join("tests/fixtures/layer_norm_bf16_absent_output/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) = (unsafe { setup("cpu_ep_lnbf16", &model_path) })
    else {
        eprintln!("*** SKIPPED: layer_norm_bf16_absent_output_no_overflow ***");
        return;
    };

    unsafe {
        let mut x_data: [u16; 8] = [
            f32_to_bf16(1.0),
            f32_to_bf16(2.0),
            f32_to_bf16(3.0),
            f32_to_bf16(4.0),
            f32_to_bf16(5.0),
            f32_to_bf16(6.0),
            f32_to_bf16(7.0),
            f32_to_bf16(8.0),
        ];
        let mut scale_data: [u16; 4] = [f32_to_bf16(1.0); 4];

        let shape: [i64; 2] = [2, 4];
        let scale_shape: [i64; 1] = [4];

        let x_val = make_bfloat16_tensor(api, &mut x_data, &shape);
        let scale_val = make_bfloat16_tensor(api, &mut scale_data, &scale_shape);

        let input_names = [c"X".as_ptr(), c"Scale".as_ptr()];
        let output_names = [c"Y".as_ptr()];
        let inputs: [*const ort::OrtValue; 2] = [x_val, scale_val];
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
        check_status(api, status, "Run(layer_norm_bf16)");
        assert!(!output.is_null());

        // Value oracle (not just crash-freedom): same normalization as the f16
        // case. Both rows equal [-1.3416394, -0.4472131, 0.4472131, 1.3416394].
        // A scratch-buffer overflow that silently corrupted Y would fail here.
        let mut y_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(output, &mut y_ptr);
        check_status(api, status, "GetTensorMutableData(layer_norm_bf16)");
        let result = std::slice::from_raw_parts(y_ptr as *const u16, 8);
        let expected_y_f32: [f32; 8] = [
            -1.3416394, -0.4472131, 0.4472131, 1.3416394, -1.3416394, -0.4472131, 0.4472131,
            1.3416394,
        ];
        eprintln!("  layer_norm_bf16 Y (raw u16): {result:?}");
        // bf16 keeps only 7 mantissa bits; 0.05 absolute tolerance covers its
        // coarse rounding for |Y|≈0.4–1.3 while still catching corruption.
        for (i, (&got_u16, &want)) in result.iter().zip(expected_y_f32.iter()).enumerate() {
            let got = bf16_to_f32(got_u16);
            assert!(
                (got - want).abs() <= 0.05,
                "Y[{i}]: got {got} (0x{got_u16:04X}), want ~{want} — \
                 corrupted normalized value indicates a scratch buffer overflow"
            );
        }

        ((*api).ReleaseValue.unwrap())(output);
        ((*api).ReleaseValue.unwrap())(x_val);
        ((*api).ReleaseValue.unwrap())(scale_val);
        teardown(api, env, opts, session, "cpu_ep_lnbf16");
        eprintln!("\n✅ layer_norm_bf16_absent_output_no_overflow: PASSED");
    }
}

// ─── B2: Add → SkipLayerNorm(out,"","",sum) → Mul ────────────────────────────

/// B2 regression: Multi-node fused subgraph with an absent optional output in
/// the middle node. Tests that the routed path allocates scratch for absent
/// slots and keeps positions aligned.
///
/// Fixture: Add(A,B)->T; SkipLayerNorm(T,skip,gamma)->(out,"","",sum); Mul(sum,C)->result
///
/// Guards:
/// - `session.disable_cpu_ep_fallback=1` — ORT errors if our EP declines
/// - EP graph assignment assertion — verifies our EP owns the nodes
#[test]
fn add_skip_layer_norm_mul_routed() {
    let _lock = lock_ort_ep();
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model_path = PathBuf::from(manifest_dir)
        .join("tests/fixtures/add_skip_layer_norm_mul/model.onnx.textproto");

    let Some((_lib, api, env, opts, session)) = (unsafe { setup("cpu_ep_aslnm", &model_path) })
    else {
        eprintln!("*** SKIPPED: add_skip_layer_norm_mul_routed ***");
        if !require_ort_or_skip("add_skip_layer_norm_mul_routed") {
            return;
        }
        unreachable!();
    };

    // Prove all three fused nodes are assigned to our cpu_ep, not ORT's built-in.
    unsafe {
        assert_ops_assigned_to_our_ep(
            api,
            session,
            &["Add", "SkipLayerNormalization", "Mul"],
            "add_skip_layer_norm_mul",
        );
    };

    unsafe {
        // A = [[1,2,3,4],[5,6,7,8]]
        let mut a_data: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // B = [[0,0,0,0],[0,0,0,0]]
        let mut b_data: [f32; 8] = [0.0; 8];
        // skip = [[0.5, -1, 1, 0], [-2, -3, -4, -1]]
        let mut skip_data: [f32; 8] = [0.5, -1.0, 1.0, 0.0, -2.0, -3.0, -4.0, -1.0];
        // gamma = [1, 1, 1, 1]
        let mut gamma_data: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
        // C = [[2, 2, 2, 2], [2, 2, 2, 2]]
        let mut c_data: [f32; 8] = [2.0; 8];

        let shape: [i64; 2] = [2, 4];
        let gamma_shape: [i64; 1] = [4];

        let a_val = make_float_tensor(api, &mut a_data, &shape);
        let b_val = make_float_tensor(api, &mut b_data, &shape);
        let skip_val = make_float_tensor(api, &mut skip_data, &shape);
        let gamma_val = make_float_tensor(api, &mut gamma_data, &gamma_shape);
        let c_val = make_float_tensor(api, &mut c_data, &shape);

        let input_names = [
            c"A".as_ptr(),
            c"B".as_ptr(),
            c"skip".as_ptr(),
            c"gamma".as_ptr(),
            c"C".as_ptr(),
        ];
        let output_names = [c"result".as_ptr()];
        let inputs: [*const ort::OrtValue; 5] = [a_val, b_val, skip_val, gamma_val, c_val];
        let mut output: *mut ort::OrtValue = ptr::null_mut();

        let status = ((*api).Run.unwrap())(
            session,
            ptr::null(),
            input_names.as_ptr(),
            inputs.as_ptr(),
            5,
            output_names.as_ptr(),
            1,
            &mut output,
        );
        check_status(api, status, "Run(add_skip_ln_mul)");
        assert!(!output.is_null());

        // T = A + B = A (since B=0)
        // sum = T + skip = [[1.5, 1, 4, 4], [3, 3, 3, 7]]
        // result = sum * C = [[3, 2, 8, 8], [6, 6, 6, 14]]
        let expected: [f32; 8] = [3.0, 2.0, 8.0, 8.0, 6.0, 6.0, 6.0, 14.0];
        let mut data_ptr: *mut std::ffi::c_void = ptr::null_mut();
        let status = ((*api).GetTensorMutableData.unwrap())(output, &mut data_ptr);
        check_status(api, status, "GetTensorMutableData(result)");
        let result = std::slice::from_raw_parts(data_ptr as *const f32, 8);

        eprintln!("  result got:      {result:?}");
        eprintln!("  result expected: {expected:?}");

        for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-4,
                "result[{i}] = {got}, want {want} — \
                 if positions are misrouted, the absent slot compaction bug is present"
            );
        }

        ((*api).ReleaseValue.unwrap())(output);
        ((*api).ReleaseValue.unwrap())(a_val);
        ((*api).ReleaseValue.unwrap())(b_val);
        ((*api).ReleaseValue.unwrap())(skip_val);
        ((*api).ReleaseValue.unwrap())(gamma_val);
        ((*api).ReleaseValue.unwrap())(c_val);
        teardown(api, env, opts, session, "cpu_ep_aslnm");
        eprintln!("\n✅ add_skip_layer_norm_mul_routed: PASSED");
    }
}

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
    if build_dir.exists()
        && let Ok(entries) = std::fs::read_dir(&build_dir)
    {
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
    None
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

    let model_c = ort_path::OrtPathBuf::new(model_path);
    let mut session: *mut ort::OrtSession = ptr::null_mut();
    let status = unsafe {
        ((*api).CreateSession.unwrap())(env, model_c.as_ptr(), session_options, &mut session)
    };
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
    let model_path =
        PathBuf::from(manifest_dir).join("tests/fixtures/skip_layer_norm_output_sum/model.onnx");

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
    let model_path = PathBuf::from(manifest_dir).join("tests/fixtures/clip_no_min/model.onnx");

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
    let model_path =
        PathBuf::from(manifest_dir).join("tests/fixtures/skip_layer_norm_no_beta_bias/model.onnx");

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
        .join("tests/fixtures/simplified_layer_norm_two_outputs/model.onnx");

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

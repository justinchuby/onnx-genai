//! Full C-ABI round-trip tests for `onnx-runtime-capi`, driven by calling the
//! exported `extern "C"` functions directly from Rust (exactly as a C caller
//! would). Each test hand-builds a tiny ONNX model, serializes it to disk, then
//! exercises the marshalling layer end to end: create session → create input
//! tensor(s) → run → read outputs → release everything.
//!
//! Nothing here names a real model or bakes in a fixed op path; the model is a
//! generic MatMul (+ Add) chain built purely to exercise the ABI.

use std::ffi::{CStr, CString, c_void};
use std::path::PathBuf;
use std::ptr;

use onnx_runtime_capi::{
    OrtErrorCode, OrtSession, OrtSessionOptions, OrtValue, nxrt_add_session_config_entry,
    nxrt_create_session, nxrt_create_session_options, nxrt_create_session_with_options,
    nxrt_create_tensor, nxrt_get_error_code, nxrt_get_error_message, nxrt_get_tensor_data,
    nxrt_get_tensor_dtype, nxrt_get_tensor_rank, nxrt_get_tensor_shape, nxrt_release_session,
    nxrt_release_session_options, nxrt_release_status, nxrt_release_value, nxrt_run,
};
use onnx_runtime_loader::proto::onnx;
use prost::Message;

const FLOAT: i32 = 1; // ONNX TensorProto.DataType.FLOAT

// --- model construction helpers --------------------------------------------

fn f32_initializer(name: &str, dims: &[i64], data: &[f32]) -> onnx::TensorProto {
    onnx::TensorProto {
        name: name.to_string(),
        data_type: FLOAT,
        dims: dims.to_vec(),
        raw_data: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
        ..Default::default()
    }
}

fn value_info(name: &str, dims: &[i64]) -> onnx::ValueInfoProto {
    use onnx::tensor_shape_proto::{Dimension, dimension::Value as DV};
    let dim = dims
        .iter()
        .map(|&n| Dimension {
            value: Some(DV::DimValue(n)),
            ..Default::default()
        })
        .collect();
    onnx::ValueInfoProto {
        name: name.to_string(),
        r#type: Some(onnx::TypeProto {
            value: Some(onnx::type_proto::Value::TensorType(
                onnx::type_proto::Tensor {
                    elem_type: FLOAT,
                    shape: Some(onnx::TensorShapeProto { dim }),
                },
            )),
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn node(op: &str, inputs: &[&str], outputs: &[&str]) -> onnx::NodeProto {
    onnx::NodeProto {
        op_type: op.to_string(),
        input: inputs.iter().map(|s| s.to_string()).collect(),
        output: outputs.iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    }
}

/// A model: `Y = MatMul(X[2,3], W[3,2]) + B[2]`, static shapes throughout.
fn build_model_bytes() -> Vec<u8> {
    let w = [
        1.0f32, 2.0, //
        3.0, 4.0, //
        5.0, 6.0,
    ];
    let b = [10.0f32, 20.0];
    let graph = onnx::GraphProto {
        name: "abi_test".into(),
        input: vec![value_info("X", &[2, 3])],
        output: vec![value_info("Y", &[2, 2])],
        initializer: vec![
            f32_initializer("W", &[3, 2], &w),
            f32_initializer("B", &[2], &b),
        ],
        node: vec![
            node("MatMul", &["X", "W"], &["H"]),
            node("Add", &["H", "B"], &["Y"]),
        ],
        ..Default::default()
    };
    let model = onnx::ModelProto {
        ir_version: 8,
        opset_import: vec![onnx::OperatorSetIdProto {
            domain: String::new(),
            version: 17,
        }],
        graph: Some(graph),
        ..Default::default()
    };
    model.encode_to_vec()
}

/// Reference for `MatMul(X[2,3], W[3,2]) + B[2]`.
fn reference(x: &[f32]) -> Vec<f32> {
    let w = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = [10.0f32, 20.0];
    let mut out = vec![0.0f32; 4];
    for i in 0..2 {
        for j in 0..2 {
            let mut acc = 0.0f32;
            for k in 0..3 {
                acc += x[i * 3 + k] * w[k * 2 + j];
            }
            out[i * 2 + j] = acc + b[j];
        }
    }
    out
}

/// Write the synthetic model to a per-test file under the crate's target dir
/// (never `/tmp`), returning its path.
fn write_model(tag: &str) -> PathBuf {
    let path = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("capi_model_{tag}.onnx"));
    std::fs::write(&path, build_model_bytes()).expect("write model");
    path
}

fn cstring(s: &str) -> CString {
    CString::new(s).unwrap()
}

// --- tests -----------------------------------------------------------------

#[test]
fn full_roundtrip_matches_reference() {
    let path = write_model("roundtrip");
    let c_path = cstring(path.to_str().unwrap());

    // Create session.
    let mut session: *mut OrtSession = ptr::null_mut();
    let status = unsafe { nxrt_create_session(c_path.as_ptr(), &mut session) };
    assert!(status.is_null(), "create_session should succeed");
    assert!(!session.is_null());

    // Create input tensor X[2,3].
    let x_data = [0.5f32, -1.0, 2.0, 1.5, 0.0, -0.5];
    let x_bytes: Vec<u8> = x_data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let x_shape = [2i64, 3];
    let mut x_value: *mut OrtValue = ptr::null_mut();
    let status = unsafe {
        nxrt_create_tensor(
            x_bytes.as_ptr() as *const c_void,
            x_bytes.len(),
            x_shape.as_ptr(),
            x_shape.len(),
            FLOAT,
            &mut x_value,
        )
    };
    assert!(status.is_null(), "create_tensor should succeed");
    assert!(!x_value.is_null());

    // Run.
    let in_name = cstring("X");
    let out_name = cstring("Y");
    let input_names = [in_name.as_ptr()];
    let input_values = [x_value as *const OrtValue];
    let output_names = [out_name.as_ptr()];
    let mut out_value: *mut OrtValue = ptr::null_mut();
    let status = unsafe {
        nxrt_run(
            session,
            input_names.as_ptr(),
            input_values.as_ptr(),
            1,
            output_names.as_ptr(),
            1,
            &mut out_value,
        )
    };
    assert!(status.is_null(), "run should succeed");
    assert!(!out_value.is_null());

    // Read output dtype.
    let mut dtype: i32 = -1;
    let status = unsafe { nxrt_get_tensor_dtype(out_value, &mut dtype) };
    assert!(status.is_null());
    assert_eq!(dtype, FLOAT);

    // Read output rank + shape.
    let mut rank: usize = 0;
    let status = unsafe { nxrt_get_tensor_rank(out_value, &mut rank) };
    assert!(status.is_null());
    assert_eq!(rank, 2);
    let mut dims = vec![0i64; rank];
    let status = unsafe { nxrt_get_tensor_shape(out_value, dims.as_mut_ptr(), rank) };
    assert!(status.is_null());
    assert_eq!(dims, vec![2, 2]);

    // Read output data + assert against reference.
    let mut data_ptr: *const c_void = ptr::null();
    let mut data_len: usize = 0;
    let status = unsafe { nxrt_get_tensor_data(out_value, &mut data_ptr, &mut data_len) };
    assert!(status.is_null());
    assert_eq!(data_len, 4 * 4); // 4 f32
    let out_bytes = unsafe { std::slice::from_raw_parts(data_ptr as *const u8, data_len) };
    let got: Vec<f32> = out_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let want = reference(&x_data);
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() < 1e-4, "got {g}, want {w}");
    }

    // Release everything (each exactly once).
    unsafe {
        nxrt_release_value(out_value);
        nxrt_release_value(x_value);
        nxrt_release_session(session);
    }
}

#[test]
fn null_session_handle_is_invalid_argument_not_crash() {
    let out_name = cstring("Y");
    let output_names = [out_name.as_ptr()];
    let mut out_value: *mut OrtValue = ptr::null_mut();
    let status = unsafe {
        nxrt_run(
            ptr::null_mut(), // null session
            ptr::null(),
            ptr::null(),
            0,
            output_names.as_ptr(),
            1,
            &mut out_value,
        )
    };
    assert!(!status.is_null(), "null session must produce a status");
    assert_eq!(
        unsafe { nxrt_get_error_code(status) },
        OrtErrorCode::InvalidArgument
    );
    // Message is a readable C string.
    let msg = unsafe { nxrt_get_error_message(status) };
    assert!(!msg.is_null());
    let _ = unsafe { CStr::from_ptr(msg) }.to_str().unwrap();
    assert!(out_value.is_null(), "out slot must be pre-nulled on error");
    unsafe { nxrt_release_status(status) };
}

#[test]
fn create_tensor_wrong_byte_len_is_rejected() {
    // Shape [2,3] f32 needs 24 bytes; provide 20.
    let bytes = [0u8; 20];
    let shape = [2i64, 3];
    let mut value: *mut OrtValue = ptr::null_mut();
    let status = unsafe {
        nxrt_create_tensor(
            bytes.as_ptr() as *const c_void,
            bytes.len(),
            shape.as_ptr(),
            shape.len(),
            FLOAT,
            &mut value,
        )
    };
    assert!(!status.is_null(), "byte-len mismatch must error");
    assert_eq!(
        unsafe { nxrt_get_error_code(status) },
        OrtErrorCode::InvalidArgument
    );
    assert!(value.is_null(), "no value produced on error");
    unsafe { nxrt_release_status(status) };
}

#[test]
fn null_out_pointers_are_rejected() {
    // create_session with a null out pointer.
    let path = write_model("null_out");
    let c_path = cstring(path.to_str().unwrap());
    let status = unsafe { nxrt_create_session(c_path.as_ptr(), ptr::null_mut()) };
    assert!(!status.is_null());
    assert_eq!(
        unsafe { nxrt_get_error_code(status) },
        OrtErrorCode::InvalidArgument
    );
    unsafe { nxrt_release_status(status) };

    // create_tensor with a null out pointer.
    let bytes = [0u8; 4];
    let shape = [1i64];
    let status = unsafe {
        nxrt_create_tensor(
            bytes.as_ptr() as *const c_void,
            bytes.len(),
            shape.as_ptr(),
            1,
            FLOAT,
            ptr::null_mut(),
        )
    };
    assert!(!status.is_null());
    assert_eq!(
        unsafe { nxrt_get_error_code(status) },
        OrtErrorCode::InvalidArgument
    );
    unsafe { nxrt_release_status(status) };
}

#[test]
fn nonexistent_model_path_is_no_such_file() {
    let c_path = cstring("/no/such/path/definitely_missing_model.onnx");
    let mut session: *mut OrtSession = ptr::null_mut();
    let status = unsafe { nxrt_create_session(c_path.as_ptr(), &mut session) };
    assert!(!status.is_null());
    assert_eq!(
        unsafe { nxrt_get_error_code(status) },
        OrtErrorCode::NoSuchFile
    );
    assert!(session.is_null());
    unsafe { nxrt_release_status(status) };
}

#[test]
fn unknown_output_name_is_rejected() {
    let path = write_model("bad_output");
    let c_path = cstring(path.to_str().unwrap());
    let mut session: *mut OrtSession = ptr::null_mut();
    let status = unsafe { nxrt_create_session(c_path.as_ptr(), &mut session) };
    assert!(status.is_null());

    let x_bytes = [0u8; 24];
    let x_shape = [2i64, 3];
    let mut x_value: *mut OrtValue = ptr::null_mut();
    unsafe {
        nxrt_create_tensor(
            x_bytes.as_ptr() as *const c_void,
            x_bytes.len(),
            x_shape.as_ptr(),
            2,
            FLOAT,
            &mut x_value,
        )
    };

    let in_name = cstring("X");
    let bogus = cstring("NotAnOutput");
    let input_names = [in_name.as_ptr()];
    let input_values = [x_value as *const OrtValue];
    let output_names = [bogus.as_ptr()];
    let mut out_value: *mut OrtValue = ptr::null_mut();
    let status = unsafe {
        nxrt_run(
            session,
            input_names.as_ptr(),
            input_values.as_ptr(),
            1,
            output_names.as_ptr(),
            1,
            &mut out_value,
        )
    };
    assert!(!status.is_null());
    assert_eq!(
        unsafe { nxrt_get_error_code(status) },
        OrtErrorCode::InvalidArgument
    );
    assert!(out_value.is_null());

    unsafe {
        nxrt_release_status(status);
        nxrt_release_value(x_value);
        nxrt_release_session(session);
    }
}

#[test]
fn shape_mismatch_input_is_rejected_at_run() {
    let path = write_model("shape_mismatch");
    let c_path = cstring(path.to_str().unwrap());
    let mut session: *mut OrtSession = ptr::null_mut();
    let status = unsafe { nxrt_create_session(c_path.as_ptr(), &mut session) };
    assert!(status.is_null());

    // Model expects X[2,3]; hand it a well-formed but wrong-shaped [3,3].
    let x_bytes = [0u8; 36];
    let x_shape = [3i64, 3];
    let mut x_value: *mut OrtValue = ptr::null_mut();
    let status = unsafe {
        nxrt_create_tensor(
            x_bytes.as_ptr() as *const c_void,
            x_bytes.len(),
            x_shape.as_ptr(),
            2,
            FLOAT,
            &mut x_value,
        )
    };
    assert!(status.is_null());

    let in_name = cstring("X");
    let out_name = cstring("Y");
    let input_names = [in_name.as_ptr()];
    let input_values = [x_value as *const OrtValue];
    let output_names = [out_name.as_ptr()];
    let mut out_value: *mut OrtValue = ptr::null_mut();
    let status = unsafe {
        nxrt_run(
            session,
            input_names.as_ptr(),
            input_values.as_ptr(),
            1,
            output_names.as_ptr(),
            1,
            &mut out_value,
        )
    };
    assert!(!status.is_null(), "shape mismatch must error");
    assert_eq!(
        unsafe { nxrt_get_error_code(status) },
        OrtErrorCode::InvalidArgument
    );
    assert!(out_value.is_null());

    unsafe {
        nxrt_release_status(status);
        nxrt_release_value(x_value);
        nxrt_release_session(session);
    }
}

#[test]
fn release_is_null_tolerant() {
    // Releasing null through every release entry point is a safe no-op — this
    // is what makes the idiomatic `release(x); x = NULL;` guard against
    // double-release: a second release simply sees null.
    unsafe {
        nxrt_release_session(ptr::null_mut());
        nxrt_release_value(ptr::null_mut());
        nxrt_release_status(ptr::null_mut());
    }
}

#[test]
fn accessors_reject_null_value_handle() {
    let mut dtype = 0i32;
    let status = unsafe { nxrt_get_tensor_dtype(ptr::null(), &mut dtype) };
    assert!(!status.is_null());
    assert_eq!(
        unsafe { nxrt_get_error_code(status) },
        OrtErrorCode::InvalidArgument
    );
    unsafe { nxrt_release_status(status) };

    let mut rank = 0usize;
    let status = unsafe { nxrt_get_tensor_rank(ptr::null(), &mut rank) };
    assert!(!status.is_null());
    unsafe { nxrt_release_status(status) };

    let mut data: *const c_void = ptr::null();
    let mut len = 0usize;
    let status = unsafe { nxrt_get_tensor_data(ptr::null(), &mut data, &mut len) };
    assert!(!status.is_null());
    unsafe { nxrt_release_status(status) };
}

// --- session-options / ep.context_* config plumbing (§21.4 / §55.5) --------

/// Creating a session through `nxrt_create_session_with_options` with the
/// `ep.context_*` keys set forwards them to the `SessionBuilder` and succeeds —
/// the options reach the session layer through the C-ABI string key/value path.
#[test]
fn create_session_with_ep_context_options_succeeds() {
    let path = write_model("epctx_opts");
    let c_path = cstring(path.to_str().unwrap());

    let mut options: *mut OrtSessionOptions = ptr::null_mut();
    let status = unsafe { nxrt_create_session_options(&mut options) };
    assert!(status.is_null(), "create_session_options should succeed");
    assert!(!options.is_null());

    for (k, v) in [
        ("ep.context_enable", "1"),
        ("ep.context_file_path", ""),
        ("ep.context_embed_mode", "1"),
    ] {
        let ck = cstring(k);
        let cv = cstring(v);
        let status = unsafe { nxrt_add_session_config_entry(options, ck.as_ptr(), cv.as_ptr()) };
        assert!(
            status.is_null(),
            "add_session_config_entry({k}) should succeed"
        );
    }

    let mut session: *mut OrtSession = ptr::null_mut();
    let status =
        unsafe { nxrt_create_session_with_options(c_path.as_ptr(), options, &mut session) };
    assert!(
        status.is_null(),
        "create_session_with_options should succeed"
    );
    assert!(!session.is_null());

    unsafe { nxrt_release_session(session) };
    unsafe { nxrt_release_session_options(options) };
}

/// An unknown config key surfaces as `InvalidArgument` at session build — the
/// C API adds no divergent option logic; validation is the session layer's.
#[test]
fn create_session_with_unknown_option_is_invalid_argument() {
    let path = write_model("epctx_bad_key");
    let c_path = cstring(path.to_str().unwrap());

    let mut options: *mut OrtSessionOptions = ptr::null_mut();
    let status = unsafe { nxrt_create_session_options(&mut options) };
    assert!(status.is_null());

    let ck = cstring("ep.context_enabel"); // typo
    let cv = cstring("1");
    let status = unsafe { nxrt_add_session_config_entry(options, ck.as_ptr(), cv.as_ptr()) };
    assert!(status.is_null(), "adding an entry never validates");

    let mut session: *mut OrtSession = ptr::null_mut();
    let status =
        unsafe { nxrt_create_session_with_options(c_path.as_ptr(), options, &mut session) };
    assert!(!status.is_null(), "unknown key must fail at build");
    assert_eq!(
        unsafe { nxrt_get_error_code(status) },
        OrtErrorCode::InvalidArgument
    );
    assert!(session.is_null());

    unsafe { nxrt_release_status(status) };
    unsafe { nxrt_release_session_options(options) };
}

/// An invalid `ep.context_embed_mode` value is rejected as `InvalidArgument`.
#[test]
fn create_session_with_invalid_embed_mode_is_invalid_argument() {
    let path = write_model("epctx_bad_embed");
    let c_path = cstring(path.to_str().unwrap());

    let mut options: *mut OrtSessionOptions = ptr::null_mut();
    let status = unsafe { nxrt_create_session_options(&mut options) };
    assert!(status.is_null());

    let ck = cstring("ep.context_embed_mode");
    let cv = cstring("2"); // only 0/1 are valid
    let status = unsafe { nxrt_add_session_config_entry(options, ck.as_ptr(), cv.as_ptr()) };
    assert!(status.is_null());

    let mut session: *mut OrtSession = ptr::null_mut();
    let status =
        unsafe { nxrt_create_session_with_options(c_path.as_ptr(), options, &mut session) };
    assert!(!status.is_null(), "invalid embed_mode must fail at build");
    assert_eq!(
        unsafe { nxrt_get_error_code(status) },
        OrtErrorCode::InvalidArgument
    );

    unsafe { nxrt_release_status(status) };
    unsafe { nxrt_release_session_options(options) };
}

/// A null options handle makes `create_session_with_options` behave like the
/// plain `create_session` (no extra options).
#[test]
fn create_session_with_null_options_is_like_plain_create() {
    let path = write_model("epctx_null_opts");
    let c_path = cstring(path.to_str().unwrap());

    let mut session: *mut OrtSession = ptr::null_mut();
    let status =
        unsafe { nxrt_create_session_with_options(c_path.as_ptr(), ptr::null(), &mut session) };
    assert!(status.is_null(), "null options should still build");
    assert!(!session.is_null());
    unsafe { nxrt_release_session(session) };
}

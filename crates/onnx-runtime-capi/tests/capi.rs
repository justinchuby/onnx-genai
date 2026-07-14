//! Full C-ABI round-trip tests for `onnx-runtime-capi`, driven by calling the
//! exported `extern "C"` functions directly from Rust (exactly as a C caller
//! would). Each test hand-builds a tiny ONNX model, serializes it to disk, then
//! exercises the marshalling layer end to end: create session → create input
//! tensor(s) → run → read outputs → release everything.
//!
//! Nothing here names a real model or bakes in a fixed op path; the model is a
//! generic MatMul (+ Add) chain built purely to exercise the ABI.

use std::ffi::{c_void, CStr, CString};
use std::path::PathBuf;
use std::ptr;

use onnx_runtime_capi::{
    ort2_create_session, ort2_create_tensor, ort2_get_error_code, ort2_get_error_message,
    ort2_get_tensor_data, ort2_get_tensor_dtype, ort2_get_tensor_rank, ort2_get_tensor_shape,
    ort2_release_session, ort2_release_status, ort2_release_value, ort2_run, OrtErrorCode,
    OrtSession, OrtValue,
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
    use onnx::tensor_shape_proto::{dimension::Value as DV, Dimension};
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
            value: Some(onnx::type_proto::Value::TensorType(onnx::type_proto::Tensor {
                elem_type: FLOAT,
                shape: Some(onnx::TensorShapeProto { dim }),
            })),
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
    let status = unsafe { ort2_create_session(c_path.as_ptr(), &mut session) };
    assert!(status.is_null(), "create_session should succeed");
    assert!(!session.is_null());

    // Create input tensor X[2,3].
    let x_data = [0.5f32, -1.0, 2.0, 1.5, 0.0, -0.5];
    let x_bytes: Vec<u8> = x_data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let x_shape = [2i64, 3];
    let mut x_value: *mut OrtValue = ptr::null_mut();
    let status = unsafe {
        ort2_create_tensor(
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
        ort2_run(
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
    let status = unsafe { ort2_get_tensor_dtype(out_value, &mut dtype) };
    assert!(status.is_null());
    assert_eq!(dtype, FLOAT);

    // Read output rank + shape.
    let mut rank: usize = 0;
    let status = unsafe { ort2_get_tensor_rank(out_value, &mut rank) };
    assert!(status.is_null());
    assert_eq!(rank, 2);
    let mut dims = vec![0i64; rank];
    let status = unsafe { ort2_get_tensor_shape(out_value, dims.as_mut_ptr(), rank) };
    assert!(status.is_null());
    assert_eq!(dims, vec![2, 2]);

    // Read output data + assert against reference.
    let mut data_ptr: *const c_void = ptr::null();
    let mut data_len: usize = 0;
    let status = unsafe { ort2_get_tensor_data(out_value, &mut data_ptr, &mut data_len) };
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
        ort2_release_value(out_value);
        ort2_release_value(x_value);
        ort2_release_session(session);
    }
}

#[test]
fn null_session_handle_is_invalid_argument_not_crash() {
    let out_name = cstring("Y");
    let output_names = [out_name.as_ptr()];
    let mut out_value: *mut OrtValue = ptr::null_mut();
    let status = unsafe {
        ort2_run(
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
        unsafe { ort2_get_error_code(status) },
        OrtErrorCode::InvalidArgument
    );
    // Message is a readable C string.
    let msg = unsafe { ort2_get_error_message(status) };
    assert!(!msg.is_null());
    let _ = unsafe { CStr::from_ptr(msg) }.to_str().unwrap();
    assert!(out_value.is_null(), "out slot must be pre-nulled on error");
    unsafe { ort2_release_status(status) };
}

#[test]
fn create_tensor_wrong_byte_len_is_rejected() {
    // Shape [2,3] f32 needs 24 bytes; provide 20.
    let bytes = [0u8; 20];
    let shape = [2i64, 3];
    let mut value: *mut OrtValue = ptr::null_mut();
    let status = unsafe {
        ort2_create_tensor(
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
        unsafe { ort2_get_error_code(status) },
        OrtErrorCode::InvalidArgument
    );
    assert!(value.is_null(), "no value produced on error");
    unsafe { ort2_release_status(status) };
}

#[test]
fn null_out_pointers_are_rejected() {
    // create_session with a null out pointer.
    let path = write_model("null_out");
    let c_path = cstring(path.to_str().unwrap());
    let status = unsafe { ort2_create_session(c_path.as_ptr(), ptr::null_mut()) };
    assert!(!status.is_null());
    assert_eq!(
        unsafe { ort2_get_error_code(status) },
        OrtErrorCode::InvalidArgument
    );
    unsafe { ort2_release_status(status) };

    // create_tensor with a null out pointer.
    let bytes = [0u8; 4];
    let shape = [1i64];
    let status = unsafe {
        ort2_create_tensor(
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
        unsafe { ort2_get_error_code(status) },
        OrtErrorCode::InvalidArgument
    );
    unsafe { ort2_release_status(status) };
}

#[test]
fn nonexistent_model_path_is_no_such_file() {
    let c_path = cstring("/no/such/path/definitely_missing_model.onnx");
    let mut session: *mut OrtSession = ptr::null_mut();
    let status = unsafe { ort2_create_session(c_path.as_ptr(), &mut session) };
    assert!(!status.is_null());
    assert_eq!(
        unsafe { ort2_get_error_code(status) },
        OrtErrorCode::NoSuchFile
    );
    assert!(session.is_null());
    unsafe { ort2_release_status(status) };
}

#[test]
fn unknown_output_name_is_rejected() {
    let path = write_model("bad_output");
    let c_path = cstring(path.to_str().unwrap());
    let mut session: *mut OrtSession = ptr::null_mut();
    let status = unsafe { ort2_create_session(c_path.as_ptr(), &mut session) };
    assert!(status.is_null());

    let x_bytes = [0u8; 24];
    let x_shape = [2i64, 3];
    let mut x_value: *mut OrtValue = ptr::null_mut();
    unsafe {
        ort2_create_tensor(
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
        ort2_run(
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
        unsafe { ort2_get_error_code(status) },
        OrtErrorCode::InvalidArgument
    );
    assert!(out_value.is_null());

    unsafe {
        ort2_release_status(status);
        ort2_release_value(x_value);
        ort2_release_session(session);
    }
}

#[test]
fn shape_mismatch_input_is_rejected_at_run() {
    let path = write_model("shape_mismatch");
    let c_path = cstring(path.to_str().unwrap());
    let mut session: *mut OrtSession = ptr::null_mut();
    let status = unsafe { ort2_create_session(c_path.as_ptr(), &mut session) };
    assert!(status.is_null());

    // Model expects X[2,3]; hand it a well-formed but wrong-shaped [3,3].
    let x_bytes = [0u8; 36];
    let x_shape = [3i64, 3];
    let mut x_value: *mut OrtValue = ptr::null_mut();
    let status = unsafe {
        ort2_create_tensor(
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
        ort2_run(
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
        unsafe { ort2_get_error_code(status) },
        OrtErrorCode::InvalidArgument
    );
    assert!(out_value.is_null());

    unsafe {
        ort2_release_status(status);
        ort2_release_value(x_value);
        ort2_release_session(session);
    }
}

#[test]
fn release_is_null_tolerant() {
    // Releasing null through every release entry point is a safe no-op — this
    // is what makes the idiomatic `release(x); x = NULL;` guard against
    // double-release: a second release simply sees null.
    unsafe {
        ort2_release_session(ptr::null_mut());
        ort2_release_value(ptr::null_mut());
        ort2_release_status(ptr::null_mut());
    }
}

#[test]
fn accessors_reject_null_value_handle() {
    let mut dtype = 0i32;
    let status = unsafe { ort2_get_tensor_dtype(ptr::null(), &mut dtype) };
    assert!(!status.is_null());
    assert_eq!(
        unsafe { ort2_get_error_code(status) },
        OrtErrorCode::InvalidArgument
    );
    unsafe { ort2_release_status(status) };

    let mut rank = 0usize;
    let status = unsafe { ort2_get_tensor_rank(ptr::null(), &mut rank) };
    assert!(!status.is_null());
    unsafe { ort2_release_status(status) };

    let mut data: *const c_void = ptr::null();
    let mut len = 0usize;
    let status = unsafe { ort2_get_tensor_data(ptr::null(), &mut data, &mut len) };
    assert!(!status.is_null());
    unsafe { ort2_release_status(status) };
}

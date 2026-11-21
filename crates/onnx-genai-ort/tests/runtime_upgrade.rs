use std::path::Path;

use onnx_genai_ort::{Environment, Session, SessionOptions, Value};

#[test]
fn pinned_runtime_executes_cpu_identity_model() {
    let env = Environment::new("ort-runtime-upgrade").expect("ORT environment");
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/runtime_identity.onnx.textproto");
    let session = Session::new(
        &env,
        &path,
        SessionOptions::default().with_intra_op_threads(1),
    )
    .expect("CPU identity session");
    let input = Value::from_slice_f32(&[1.0, -2.5, 0.0], &[3]).expect("input tensor");
    let output = session.run(&[("x", &input)]).expect("CPU inference");
    assert_eq!(output.len(), 1);
    assert_eq!(output[0].shape(), &[3]);
    assert_eq!(
        output[0].to_vec_f32().expect("output data"),
        [1.0, -2.5, 0.0]
    );
    assert_eq!(onnx_genai_ort_sys::loaded_ort_api_version(), Some(32));
}

use std::path::Path;
use std::process::Command;

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA, Python ONNX packages, and the gpu-tests feature"
)]
#[test]
fn real_ort_session_executes_captures_and_rejects_v2() {
    let plugin = onnx_runtime_ort_testkit::find_plugin_cdylib_with_features(
        "onnx-runtime-ep-cuda-plugin",
        &["gpu-tests"],
    )
    .expect("build CUDA plugin cdylib with gpu-tests");
    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crate must be under <workspace>/crates");
    let script = workspace.join("scripts/validate_dsa_index_select_plugin.py");
    let status = Command::new("python3")
        .arg(script)
        .arg(plugin)
        .status()
        .expect("launch real ORT CUDA-plugin validation");
    assert!(status.success(), "real CUDA-plugin validation failed");
}

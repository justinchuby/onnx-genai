use std::path::{Path, PathBuf};

use onnx_genai_ort::{
    Environment, Session, SessionOptions, available_execution_providers, ep_selection,
};

fn tiny_llm() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm/model.onnx")
}

#[test]
fn reports_available_execution_providers() {
    let providers = available_execution_providers().expect("available providers");
    println!("available execution providers: {providers:?}");
    assert!(
        providers
            .iter()
            .any(|provider| provider == "CPUExecutionProvider"),
        "available providers: {providers:?}"
    );
}

#[test]
fn requested_gpu_execution_provider_loads_or_falls_back_to_cpu() {
    let env = Environment::new("execution-provider-fallback-test").expect("env");
    let options = SessionOptions::with_execution_provider(ep_selection("webgpu"));
    Session::new(&env, &tiny_llm(), options).expect("session falls back to CPU");
}

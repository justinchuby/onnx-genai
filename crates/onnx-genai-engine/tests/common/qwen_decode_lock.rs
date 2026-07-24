use std::path::PathBuf;

use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GeneratePrompt, GenerateRequest, NativeDecodeDevice,
};

pub fn assert_native_cuda_matches_ort_cuda(
    model_name: &str,
    model_dir_env: &str,
    expected_tokens: &[u32],
) -> anyhow::Result<()> {
    let Some(model_dir) = std::env::var_os(model_dir_env).map(PathBuf::from) else {
        eprintln!("skipping {model_name} decode lock: set {model_dir_env}");
        return Ok(());
    };
    if !model_dir.is_dir() {
        eprintln!(
            "skipping {model_name} decode lock: model is not installed at {}",
            model_dir.display()
        );
        return Ok(());
    }
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping {model_name} decode lock: CUDA unavailable: {error}");
        return Ok(());
    }

    unsafe {
        std::env::set_var("ONNX_GENAI_EP", "cuda");
    }
    let ort = generate(
        &model_dir,
        EngineDecodeBackend::Ort,
        None,
        expected_tokens.len(),
    )?;
    assert_eq!(
        ort, expected_tokens,
        "{model_name} ORT CUDA greedy sequence drifted"
    );

    let native = generate(
        &model_dir,
        EngineDecodeBackend::Native,
        Some(NativeDecodeDevice::Cuda { index: Some(0) }),
        expected_tokens.len(),
    )?;
    assert_eq!(
        native, expected_tokens,
        "{model_name} native CUDA greedy sequence drifted"
    );
    assert_eq!(
        native, ort,
        "{model_name} native and ORT CUDA greedy sequences diverged"
    );
    Ok(())
}

fn generate(
    model_dir: &std::path::Path,
    backend: EngineDecodeBackend,
    native_device: Option<NativeDecodeDevice>,
    token_count: usize,
) -> anyhow::Result<Vec<u32>> {
    let mut engine = Engine::from_dir(
        model_dir,
        EngineConfig {
            decode_backend: backend,
            native_device,
            ..EngineConfig::default()
        },
    )?;
    let mut request =
        GenerateRequest::new(GeneratePrompt::Text("The capital of France is".to_string()));
    request.options.max_new_tokens = token_count;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;
    Ok(engine.generate(request)?.token_ids)
}

use std::path::PathBuf;

use onnx_genai_engine::{
    DecodePrecision, Engine, EngineConfig, EngineDecodeBackend, GeneratePrompt, GenerateRequest,
    NativeDecodeDevice,
};

#[allow(dead_code)]
pub fn assert_native_matches_ort_greedy(
    model_dir_env: &str,
    prompt: &str,
    expected_tokens: &[u32],
) -> anyhow::Result<()> {
    let Some(model_dir) = cuda_model_dir(model_dir_env) else {
        return Ok(());
    };
    let ort = generate(
        &model_dir,
        EngineDecodeBackend::Ort,
        None,
        DecodePrecision::Model,
        prompt,
        expected_tokens.len(),
    )?;
    assert_eq!(
        ort, expected_tokens,
        "{model_dir_env} ORT CUDA greedy sequence drifted"
    );

    let native = generate(
        &model_dir,
        EngineDecodeBackend::Native,
        Some(NativeDecodeDevice::Cuda { index: Some(0) }),
        DecodePrecision::Model,
        prompt,
        expected_tokens.len(),
    )?;
    assert_eq!(
        native, expected_tokens,
        "{model_dir_env} native CUDA greedy sequence drifted"
    );
    assert_eq!(
        native, ort,
        "{model_dir_env} native and ORT CUDA greedy sequences diverged"
    );
    Ok(())
}

/// Assert that native CUDA decode with [`DecodePrecision::Fp16`] (the opt-in
/// fp32→fp16 decoder rewrite) reproduces the trusted ORT fp32 CUDA greedy stream
/// token-for-token. This locks the fp16-fused decode path for an fp32-activation
/// int4/block-32 (`accuracy_level=4`) model against its fp32 reference: any
/// divergence fails the test (no silent pass).
#[allow(dead_code)]
pub fn assert_native_fp16_matches_ort_greedy(
    model_dir_env: &str,
    prompt: &str,
    expected_tokens: &[u32],
) -> anyhow::Result<()> {
    let Some(model_dir) = cuda_model_dir(model_dir_env) else {
        return Ok(());
    };
    let ort = generate(
        &model_dir,
        EngineDecodeBackend::Ort,
        None,
        DecodePrecision::Model,
        prompt,
        expected_tokens.len(),
    )?;
    assert_eq!(
        ort, expected_tokens,
        "{model_dir_env} ORT CUDA fp32 greedy reference drifted"
    );

    let native_fp16 = generate(
        &model_dir,
        EngineDecodeBackend::Native,
        Some(NativeDecodeDevice::Cuda { index: Some(0) }),
        DecodePrecision::Fp16,
        prompt,
        expected_tokens.len(),
    )?;
    assert_eq!(
        native_fp16, expected_tokens,
        "{model_dir_env} native CUDA fp16-fused greedy sequence drifted from the locked stream"
    );
    assert_eq!(
        native_fp16, ort,
        "{model_dir_env} native CUDA fp16-fused decode diverged from the ORT fp32 reference"
    );
    Ok(())
}

#[allow(dead_code)]
pub fn assert_native_matches_golden(
    model_dir_env: &str,
    prompt: &str,
    expected_tokens: &[u32],
) -> anyhow::Result<()> {
    let Some(model_dir) = cuda_model_dir(model_dir_env) else {
        return Ok(());
    };
    let native = generate(
        &model_dir,
        EngineDecodeBackend::Native,
        Some(NativeDecodeDevice::Cuda { index: Some(0) }),
        DecodePrecision::Model,
        prompt,
        expected_tokens.len(),
    )?;
    assert_eq!(
        native, expected_tokens,
        "{model_dir_env} native CUDA greedy sequence drifted from its golden lock"
    );
    Ok(())
}

fn cuda_model_dir(model_dir_env: &str) -> Option<PathBuf> {
    let Some(model_dir) = std::env::var_os(model_dir_env).map(PathBuf::from) else {
        eprintln!("skipping decode lock: set {model_dir_env}");
        return None;
    };
    if !model_dir.is_dir() {
        eprintln!(
            "skipping decode lock for {model_dir_env}: model is not installed at {}",
            model_dir.display()
        );
        return None;
    }
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping decode lock for {model_dir_env}: CUDA unavailable: {error}");
        return None;
    }
    unsafe {
        std::env::set_var("ONNX_GENAI_EP", "cuda");
    }
    Some(model_dir)
}

fn generate(
    model_dir: &std::path::Path,
    backend: EngineDecodeBackend,
    native_device: Option<NativeDecodeDevice>,
    decode_precision: DecodePrecision,
    prompt: &str,
    token_count: usize,
) -> anyhow::Result<Vec<u32>> {
    let mut engine = Engine::from_dir(
        model_dir,
        EngineConfig {
            decode_backend: backend,
            native_device,
            decode_precision,
            ..EngineConfig::default()
        },
    )?;
    let mut request = GenerateRequest::new(GeneratePrompt::Text(prompt.to_string()));
    request.options.max_new_tokens = token_count;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;
    Ok(engine.generate(request)?.token_ids)
}

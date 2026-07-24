//! Native CUDA coherence and ORT parity lock for Qwen3-0.6B int4.
//!
//! Unlike the separate Qwen3 acc-4 divergence/oracle harness, this test locks a
//! prompt where native CUDA and ORT CUDA agree exactly for 32 greedy tokens. It
//! exercises Qwen3's per-head Q/K RMSNormalization-before-RoPE path.
//!
//! ```bash
//! QWEN3_0_6B_CUDA_E2E_DIR=/path/to/qwen3-0.6b-int4-cuda-postfix \
//! CUDA_VISIBLE_DEVICES=0 cargo test -p onnx-genai-engine \
//!   --features cuda,native-backend --test qwen3_0_6b_native_cuda_e2e \
//!   -- --ignored --nocapture
//! ```
#![cfg(all(feature = "cuda", feature = "native-backend"))]

use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GenerateRequest, GenerateResult, NativeDecodeDevice,
};

const DEFAULT_MODEL_DIR: &str = "/home/justinchu/mobius/.scratch/qwen3-0.6b-int4-cuda-postfix";
const PROMPT: &str = "The capital of France is";
const EXPECTED_TOKENS: [u32; 32] = [
    12095, 11, 323, 279, 6722, 315, 15344, 374, 21718, 13, 576, 6722, 315, 9625, 374, 1083, 279,
    6722, 315, 279, 3146, 429, 702, 279, 1429, 1251, 13, 576, 6722, 315, 15344, 374,
];

fn model_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("QWEN3_0_6B_CUDA_E2E_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_DIR));
    let required = [
        "model.onnx",
        "model.onnx.data",
        "inference_metadata.yaml",
        "tokenizer.json",
    ];
    let missing: Vec<_> = required
        .iter()
        .filter(|name| !dir.join(name).is_file())
        .collect();
    if missing.is_empty() {
        Some(dir)
    } else {
        eprintln!(
            "skipping Qwen3-0.6B native CUDA regression: model directory {} is missing {}",
            dir.display(),
            missing
                .iter()
                .map(|name| name.as_ref())
                .collect::<Vec<&str>>()
                .join(", ")
        );
        None
    }
}

fn generate(dir: &Path, backend: EngineDecodeBackend) -> anyhow::Result<GenerateResult> {
    let mut engine = Engine::from_dir(
        dir,
        EngineConfig {
            decode_backend: backend,
            native_device: Some(NativeDecodeDevice::Cuda { index: Some(0) }),
            ..EngineConfig::default()
        },
    )?;
    let mut request = GenerateRequest::new(PROMPT.to_string());
    request.options.max_new_tokens = EXPECTED_TOKENS.len();
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;
    engine.generate(request)
}

#[test]
#[ignore = "requires the real Qwen3-0.6B int4 postfix export and a CUDA device"]
fn qwen3_0_6b_native_cuda_is_coherent_and_matches_ort_for_32_tokens() -> anyhow::Result<()> {
    let Some(dir) = model_dir() else {
        return Ok(());
    };
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping Qwen3-0.6B native CUDA regression: CUDA unavailable: {error}");
        return Ok(());
    }

    unsafe {
        std::env::set_var("ONNX_GENAI_EP", "cuda");
    }
    let providers = onnx_genai_ort::available_execution_providers()?;
    assert!(
        providers
            .iter()
            .any(|provider| provider.eq_ignore_ascii_case("CUDAExecutionProvider")),
        "Qwen3 ORT parity lock requires CUDAExecutionProvider; available providers: {providers:?}"
    );

    let native = generate(&dir, EngineDecodeBackend::Native)?;
    assert_eq!(
        native.token_ids, EXPECTED_TOKENS,
        "native CUDA Qwen3 greedy stream drifted from the validated anchor"
    );
    assert!(
        native.text.contains("Paris") && native.text.contains("Rome"),
        "native CUDA output lost the validated coherent completion: {:?}",
        native.text
    );

    let ort = generate(&dir, EngineDecodeBackend::Ort)?;
    assert_eq!(
        ort.token_ids, EXPECTED_TOKENS,
        "ORT CUDA Qwen3 greedy stream drifted from the validated anchor"
    );
    assert_eq!(
        native.token_ids, ort.token_ids,
        "native CUDA diverged from ORT CUDA within the 32-token parity horizon"
    );
    eprintln!(
        "Qwen3-0.6B native CUDA lock OK: text={:?}, tokens={:?}",
        native.text, native.token_ids
    );
    Ok(())
}

//! Greedy-decode parity lock for Phi-4-mini int4.
//!
//! Native CUDA and ORT CUDA must produce the same fixed greedy stream for this
//! deterministic prompt. The full expected sequence makes this a regression
//! lock rather than a merely self-consistent backend comparison.
//!
//! ```bash
//! ONNX_GENAI_PHI4_MINI_CUDA_DIR=/path/to/model CUDA_VISIBLE_DEVICES=0 \
//! cargo test -p onnx-genai-engine --features cuda,native-backend \
//!   --test phi4_mini_divergence -- --ignored --nocapture
//! ```
#![cfg(all(feature = "cuda", feature = "native-backend"))]

use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GenerateRequest, GenerateResult, NativeDecodeDevice,
};

const DEFAULT_MODEL_DIR: &str =
    "/home/justinchu/.foundry/cache/models/Microsoft/Phi-4-mini-instruct-cuda-gpu-5/v5";
const PROMPT: &str = "The capital of France is";
const MAX_NEW_TOKENS: usize = 64;
const EXPECTED_TOKENS: [u32; MAX_NEW_TOKENS] = [
    12650, 13, 4614, 382, 290, 9029, 328, 10128, 30, 12650, 13, 199999, 198, 27, 956, 2518, 1904,
    29, 15, 198, 3575, 553, 261, 10297, 326, 44363, 20837, 29186, 13, 1608, 738, 6052, 5359, 4122,
    402, 290, 3992, 21179, 11, 1118, 382, 261, 77177, 22311, 328, 261, 53556, 885, 8866, 326, 3100,
    364, 56949, 290, 53556, 8866, 326, 3100, 316, 6052, 290, 3992, 4928, 25,
];

fn model_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("ONNX_GENAI_PHI4_MINI_CUDA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_DIR));
    let required = [
        "model.onnx",
        "model.onnx.data",
        "genai_config.json",
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
            "skipping Phi-4-mini native CUDA regression: model directory {} is missing {}",
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
    request.options.max_new_tokens = MAX_NEW_TOKENS;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;
    engine.generate(request)
}

#[test]
#[ignore = "requires the real Phi-4-mini int4 export and a CUDA device"]
fn phi4_mini_native_matches_ort_greedy() -> anyhow::Result<()> {
    let Some(dir) = model_dir() else {
        return Ok(());
    };
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping Phi-4-mini native CUDA regression: CUDA unavailable: {error}");
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
        "Phi-4-mini ORT parity lock requires CUDAExecutionProvider; available providers: {providers:?}"
    );

    let native = generate(&dir, EngineDecodeBackend::Native)?;
    assert_eq!(
        native.token_ids, EXPECTED_TOKENS,
        "native CUDA Phi-4-mini greedy stream drifted from the validated anchor"
    );

    let ort = generate(&dir, EngineDecodeBackend::Ort)?;
    assert_eq!(
        ort.token_ids, EXPECTED_TOKENS,
        "ORT CUDA Phi-4-mini greedy stream drifted from the validated anchor"
    );
    assert_eq!(
        native.token_ids, ort.token_ids,
        "native CUDA diverged from ORT CUDA within the Phi-4-mini parity horizon"
    );
    eprintln!(
        "Phi-4-mini native CUDA lock OK: text={:?}, tokens={:?}",
        native.text, native.token_ids
    );
    Ok(())
}

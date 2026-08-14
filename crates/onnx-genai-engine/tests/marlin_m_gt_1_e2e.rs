//! End-to-end parity for the opt-in Marlin int4 M>1 tensor-core GEMM.
//!
//! Prefill runs the `com.microsoft::MatMulNBits` op at M = prompt length (M>1),
//! so a prompt of several tokens exercises the Marlin path while decode stays on
//! the M=1 GEMV. This locks that enabling the Marlin M>1 GEMM
//! (`ONNX_GENAI_MARLIN_M_GT_1=1`) does not change the greedy token stream versus
//! the portable tiled GEMM on real int4 models with asymmetric zero points
//! (glm-4-9b, qwen2.5-14b).
//!
//! ```bash
//! CUDA_VISIBLE_DEVICES=7 cargo test -p onnx-genai-engine \
//!   --features cuda,native-backend --test marlin_m_gt_1_e2e \
//!   -- --ignored --nocapture
//! ```
#![cfg(all(feature = "cuda", feature = "native-backend"))]

use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GenerateRequest, GenerateResult, NativeDecodeDevice,
};

const GLM_DEFAULT_DIR: &str = "/home/justinchu/glm-e2e-artifacts/glm-4-9b-int4-cuda";
const QWEN_DEFAULT_DIR: &str = "/home/justinchu/shared-models/qwen2.5-14b-instruct-int4-zp-onnx";

// A multi-token prompt so prefill runs MatMulNBits at M>1 (the Marlin path).
const PROMPT: &str = "List three European capital cities and the countries they belong to.";
const MAX_NEW_TOKENS: usize = 24;

fn model_dir(env_key: &str, default_dir: &str) -> Option<PathBuf> {
    let dir = std::env::var_os(env_key)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default_dir));
    let required = ["model.onnx", "model.onnx.data", "tokenizer.json"];
    let missing: Vec<_> = required
        .iter()
        .filter(|name| !dir.join(name).is_file())
        .collect();
    if missing.is_empty() {
        Some(dir)
    } else {
        eprintln!(
            "skipping Marlin M>1 e2e: model directory {} is missing {}",
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

fn generate(dir: &Path) -> anyhow::Result<GenerateResult> {
    let mut engine = Engine::from_dir(
        dir,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
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

/// Runs the model twice — tiled (flag off) then Marlin (flag on) — and asserts
/// the greedy token streams match. The Marlin M>1 GEMM reorders partial sums so
/// it is not byte-exact, but on greedy decode the argmax token must be stable.
fn assert_marlin_matches_tiled(dir: &Path, label: &str) -> anyhow::Result<()> {
    // SAFETY: ignored e2e test runs serially; no concurrent readers of the flag.
    unsafe {
        std::env::set_var("ONNX_GENAI_EP", "cuda");
        std::env::remove_var("ONNX_GENAI_MARLIN_M_GT_1");
    }
    let tiled = generate(dir)?;

    // SAFETY: see above.
    unsafe {
        std::env::set_var("ONNX_GENAI_MARLIN_M_GT_1", "1");
    }
    let marlin = generate(dir);

    // SAFETY: clear the flag regardless of the result so it cannot leak.
    unsafe {
        std::env::remove_var("ONNX_GENAI_MARLIN_M_GT_1");
    }
    let marlin = marlin?;

    eprintln!("[{label}] tiled : {:?}", tiled.token_ids);
    eprintln!("[{label}] marlin: {:?}", marlin.token_ids);
    assert_eq!(
        marlin.token_ids, tiled.token_ids,
        "[{label}] Marlin M>1 prefill changed the greedy token stream vs the tiled GEMM"
    );
    Ok(())
}

#[test]
#[ignore = "requires the real glm-4-9b-int4 export and a CUDA device"]
fn marlin_m_gt_1_matches_tiled_on_glm_4_9b_int4() -> anyhow::Result<()> {
    let Some(dir) = model_dir("GLM_4_9B_CUDA_E2E_DIR", GLM_DEFAULT_DIR) else {
        return Ok(());
    };
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping Marlin M>1 e2e (glm): CUDA unavailable: {error}");
        return Ok(());
    }
    assert_marlin_matches_tiled(&dir, "glm-4-9b-int4")
}

#[test]
#[ignore = "requires the real qwen2.5-14b-int4-zp export and a CUDA device"]
fn marlin_m_gt_1_matches_tiled_on_qwen2_5_14b_int4() -> anyhow::Result<()> {
    let Some(dir) = model_dir("QWEN2_5_14B_CUDA_E2E_DIR", QWEN_DEFAULT_DIR) else {
        return Ok(());
    };
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping Marlin M>1 e2e (qwen): CUDA unavailable: {error}");
        return Ok(());
    }
    assert_marlin_matches_tiled(&dir, "qwen2.5-14b-int4")
}

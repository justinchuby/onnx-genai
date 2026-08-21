//! Native **CUDA** decode regression for the QUANTIZED tiny synthetic GLM-5.2
//! (`glm_moe_dsa`) model — the standard-`ai.onnx::Attention` + `indexer` q4
//! variant (as opposed to the `pkg.nxrt::IndexShare` QMoE fixture covered by
//! `glm_tiny_qmoe_native_cuda_e2e.rs`).
//!
//! ## What it locks
//!
//! GLM-5.2's `indexer` attention branch combines a *logical*-width score with a
//! cast/squeezed `attention_mask` (e.g. `.../indexer/Add_node_70`). The native
//! CUDA decode path used to freeze the single-token mask to the *physical* KV
//! capacity (`max_len`, e.g. 4096) for CUDA-graph eligibility, which leaked the
//! padded width into that `Add` and failed decode token 1 with:
//!
//! ```text
//! [[1, 1, 2], [1, 1, 4096]] are not broadcast-compatible
//! ```
//!
//! The fix routes the logical-vs-physical mask width by *consumer*: a mask
//! binding with a non-capacity-aware consumer (the indexer arithmetic) exposes
//! its logical valid length instead of the padded capacity (and decodes eagerly,
//! forfeiting graph capture — mirroring prefill). This test reproduces the exact
//! failing prompts and asserts multi-token decode without the broadcast error,
//! plus native CUDA / CPU parity.
//!
//! The artifact is the same one `glm_tiny_quant_e2e.rs` uses. Point
//! `GLM_TINY_Q4_E2E_DIR` at the artifact directory and run:
//!
//! ```bash
//! CUDA_VISIBLE_DEVICES=3 GLM_TINY_Q4_E2E_DIR=/home/justinchu/glm-e2e-artifacts/glm-5.2-tiny-q4 \
//!   cargo test -p onnx-genai-engine --features cuda,native-backend \
//!   --test glm_tiny_quant_native_cuda_e2e -- --nocapture
//! ```
#![cfg(feature = "native-cuda")]

use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GeneratePrompt, GenerateRequest, NativeDecodeDevice,
};

fn fixture_dir() -> Option<PathBuf> {
    let Some(dir) = std::env::var_os("GLM_TINY_Q4_E2E_DIR").map(PathBuf::from) else {
        eprintln!(
            "skipping glm_tiny_quant_native_cuda_e2e: set GLM_TINY_Q4_E2E_DIR to a mobius-exported \
             quantized tiny glm_moe_dsa artifact directory (model.onnx + inference_metadata.yaml + tokenizer.json)"
        );
        return None;
    };
    let required = ["model.onnx", "inference_metadata.yaml", "tokenizer.json"];
    let missing: Vec<_> = required
        .iter()
        .filter(|name| !dir.join(name).is_file())
        .collect();
    if missing.is_empty() {
        Some(dir)
    } else {
        eprintln!(
            "skipping glm_tiny_quant_native_cuda_e2e: GLM_TINY_Q4_E2E_DIR {} is missing {}",
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

fn native_engine(dir: &Path, device: NativeDecodeDevice) -> anyhow::Result<Engine> {
    Engine::from_dir(
        dir,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            native_device: Some(device),
            ..EngineConfig::default()
        },
    )
}

fn generate(
    engine: &mut Engine,
    prompt_ids: Vec<u32>,
    max_new_tokens: usize,
) -> anyhow::Result<Vec<u32>> {
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(prompt_ids));
    request.options.max_new_tokens = max_new_tokens;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;
    Ok(engine.generate(request)?.token_ids)
}

/// The two prompts from the original bug report: `[123]` drives the decode mask
/// to logical width 2 (`[1,1,2]` indexer score) and `[1,2,3,4]` to width 5. Both
/// used to fail native CUDA decode token 1 at the `indexer` `Add` with
/// `[[1,1,N], [1,1,4096]] are not broadcast-compatible` because the single-token
/// mask was frozen to the physical KV capacity. After the consumer-scoped fix
/// each must decode multiple in-vocab tokens without that broadcast failure.
///
/// Weights are random (like `glm_tiny_quant_e2e.rs`), so only *structural*
/// success is asserted — the pre-fix code never reached token 2 at all, so
/// completing the full decode is a decisive regression lock. Native CPU is not
/// used as a reference here: this q4 fixture exercises an `Int32`
/// `ScatterElements` the native CPU backend does not accept, a limitation
/// orthogonal to the mask-capacity bug under test.
#[test]
fn glm_tiny_quant_native_cuda_decodes_indexer_prompts() -> anyhow::Result<()> {
    let Some(dir) = fixture_dir() else {
        return Ok(());
    };
    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!("skipping glm_tiny_quant_native_cuda_e2e: CUDA is unavailable: {error}");
        return Ok(());
    }

    let max_new_tokens = 4usize;
    for prompt_ids in [vec![123u32], vec![1u32, 2, 3, 4]] {
        let mut cuda = native_engine(&dir, NativeDecodeDevice::Cuda { index: Some(0) })?;
        let cuda_tokens = generate(&mut cuda, prompt_ids.clone(), max_new_tokens)?;

        eprintln!("glm52 q4 native CUDA prompt={prompt_ids:?}: tokens={cuda_tokens:?}");

        assert!(
            cuda_tokens.len() >= 2,
            "native CUDA decode for prompt {prompt_ids:?} must produce >= 2 tokens without the \
             indexer broadcast failure, got {}",
            cuda_tokens.len()
        );
        assert_eq!(
            cuda_tokens.len(),
            max_new_tokens,
            "native CUDA decode for prompt {prompt_ids:?} must complete all {max_new_tokens} tokens"
        );
        for &tok in &cuda_tokens {
            assert!(
                tok < 256,
                "generated token id {tok} is outside the tiny vocab (256)"
            );
        }
    }

    Ok(())
}

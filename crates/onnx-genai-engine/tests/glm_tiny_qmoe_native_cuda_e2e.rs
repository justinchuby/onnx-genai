//! Native-executor end-to-end smoke test for the tiny synthetic GLM-5.2
//! (glm_moe_dsa) model whose Deepseek-Sparse-Attention (DSA) mask path is
//! lowered to `pkg.nxrt::IndexShare` (device-resident, capture-capable) instead
//! of a dense `TopK -> ScatterElements -> Attention` island.
//!
//! Unlike `glm_tiny_qmoe_e2e.rs` (which runs the model through ONNX Runtime),
//! this test drives the model through the **native** runtime/executor on the
//! **CUDA** execution provider. It guards the exporter/runtime composition that
//! unblocks GLM-5.2 DSA-MoE native decode: before the IndexShare lowering, the
//! native single-token decode exposed the attention mask at fixed KV capacity
//! while the index/KV caches were exposed at logical length, producing a
//! `... are not broadcast-compatible` failure. IndexShare sidesteps that: the
//! selected indices reference cache positions, so there is no dense
//! full-length bias add.
//!
//! Weights are random, so only *structural* success is asserted: the native
//! executor loads the model, runs prefill + several decode steps without the
//! broadcast error (or any panic), produces the requested number of in-vocab
//! tokens, and matches the native-CPU tokens byte-for-byte (greedy).
//!
//! The tiny model is built with `mobius/export_glm_tiny_qmoe.py`. Point
//! `GLM_TINY_QMOE_E2E_DIR` at the artifact directory and run:
//!
//! ```bash
//! GLM_TINY_QMOE_E2E_DIR=/home/justinchu/glm-e2e-artifacts/glm-5.2-tiny-qmoe \
//!   cargo test -p onnx-genai-engine --features cuda \
//!   --test glm_tiny_qmoe_native_cuda_e2e -- --ignored --nocapture
//! ```
#![cfg(all(feature = "native-backend", feature = "cuda"))]

use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GeneratePrompt, GenerateRequest, NativeDecodeDevice,
};

/// Returns true if the raw `model.onnx` bytes reference an `IndexShare` op. The
/// op type is stored as a length-prefixed string in the serialized NodeProto,
/// so a substring scan is sufficient to prove the DSA mask path was lowered to
/// `pkg.nxrt::IndexShare` rather than a dense `Attention` island.
fn model_contains_index_share(model_path: &std::path::Path) -> bool {
    let Ok(bytes) = std::fs::read(model_path) else {
        return false;
    };
    bytes.windows(10).any(|w| w == b"IndexShare")
}

#[test]
#[ignore = "requires a mobius-exported IndexShare tiny glm_moe_dsa model via GLM_TINY_QMOE_E2E_DIR + a CUDA device"]
fn glm_tiny_qmoe_native_cuda_e2e() -> anyhow::Result<()> {
    let Some(dir) = std::env::var_os("GLM_TINY_QMOE_E2E_DIR") else {
        eprintln!(
            "skipping glm_tiny_qmoe_native_cuda_e2e: set GLM_TINY_QMOE_E2E_DIR to a \
             mobius-exported IndexShare tiny glm_moe_dsa artifact directory (model.onnx + \
             inference_metadata.yaml + tokenizer.json)"
        );
        return Ok(());
    };
    let dir = std::path::PathBuf::from(dir);
    if !dir.is_dir() {
        eprintln!(
            "skipping glm_tiny_qmoe_native_cuda_e2e: GLM_TINY_QMOE_E2E_DIR is absent: {}",
            dir.display()
        );
        return Ok(());
    }

    if let Err(error) = onnx_runtime_ep_cuda::CudaExecutionProvider::new(0) {
        eprintln!(
            "skipping glm_tiny_qmoe_native_cuda_e2e: CUDA is unavailable: {error}"
        );
        return Ok(());
    }

    // Prove the export actually lowered the DSA mask path to IndexShare before
    // running it, so a silent regression to the dense Attention island (which
    // triggers the broadcast failure under native fixed-capacity decode) can't
    // pass this test.
    let model_path = dir.join("model.onnx");
    assert!(
        model_contains_index_share(&model_path),
        "model {} does not contain a pkg.nxrt::IndexShare node — the DSA mask path \
         was not lowered to IndexShare",
        model_path.display()
    );

    let config = |native_device| EngineConfig {
        decode_backend: EngineDecodeBackend::Native,
        native_device: Some(native_device),
        ..EngineConfig::default()
    };
    let generate = |engine: &mut Engine| -> anyhow::Result<Vec<u32>> {
        // Tiny tokenizer vocab is digits; keep the prompt inside it.
        let prompt_ids = vec![1u32, 2, 3];
        let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(prompt_ids));
        request.options.max_new_tokens = 12;
        request.options.temperature = 0.0;
        request.options.greedy = true;
        request.options.stop_on_eos = false;
        Ok(engine.generate(request)?.token_ids)
    };

    // Native CPU baseline: if this decodes, the IndexShare composition is sound
    // independent of the CUDA kernels.
    let mut cpu = Engine::from_dir(&dir, config(NativeDecodeDevice::Cpu))?;
    let cpu_tokens = generate(&mut cpu)?;

    // Native CUDA: this is the path that previously died with the
    // `... are not broadcast-compatible` error. Reaching this assertion at all
    // means the broadcast failure is gone.
    let mut cuda = Engine::from_dir(&dir, config(NativeDecodeDevice::Cuda { index: Some(0) }))?;
    let cuda_tokens = generate(&mut cuda)?;

    eprintln!(
        "glm_tiny_qmoe_native_cuda_e2e: cpu={cpu_tokens:?} cuda={cuda_tokens:?}"
    );

    assert_eq!(
        cpu_tokens.len(),
        12,
        "expected 12 native-CPU tokens, got {}",
        cpu_tokens.len()
    );
    assert_eq!(
        cuda_tokens.len(),
        12,
        "expected 12 native-CUDA tokens, got {}",
        cuda_tokens.len()
    );

    // vocab_size = 256 in the tiny config; gross shape/logits corruption from
    // the IndexShare/QMoE composition would push tokens out of range.
    for &tok in &cuda_tokens {
        assert!(
            tok < 256,
            "native-CUDA generated token id {tok} is outside the tiny vocab (256)"
        );
    }

    // Greedy decode is deterministic, so native CUDA must match native CPU
    // byte-for-byte. This is the parity signal that the CUDA IndexShare + QMoE
    // kernels compose correctly with the rest of the graph.
    assert_eq!(
        cuda_tokens, cpu_tokens,
        "native-CUDA tokens diverged from native-CPU tokens"
    );

    Ok(())
}

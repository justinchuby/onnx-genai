//! Structural end-to-end smoke test for a tiny synthetic GLM-5.2 (glm_moe_dsa)
//! model whose routed MoE experts are emitted as a single fused
//! `com.microsoft::QMoE` node (int4, block-32, expert-major layout) instead of a
//! per-expert unroll of `com.microsoft::MatMulNBits`.
//!
//! Sibling of `glm_tiny_quant_e2e.rs`. This exercises the real ORT-contrib QMoE
//! CPU kernel through the full GLM graph (MLA + IndexShare DSA + fused MoE),
//! including GLM's sigmoid + noaux_tc routing plumbed through the kernel's
//! `router_probs` (selection) + optional `router_weights` (combine) inputs.
//! Weights are random, so only *structural* success is asserted: the engine
//! loads the model, runs prefill + several decode steps without panicking, and
//! produces the requested number of in-vocab tokens.
//!
//! Before asserting execution, the test greps the exported `model.onnx` to
//! confirm a fused `QMoE` node is present. The tiny model is built with
//! `mobius/export_glm_tiny_qmoe.py`. Point `GLM_TINY_QMOE_E2E_DIR` at the
//! artifact directory and run:
//!
//! ```bash
//! GLM_TINY_QMOE_E2E_DIR=/home/justinchu/glm-e2e-artifacts/glm-5.2-tiny-qmoe \
//!   cargo test -p onnx-genai-engine --test glm_tiny_qmoe_e2e -- --ignored --nocapture
//! ```

use onnx_genai_engine::{Engine, EngineConfig, GeneratePrompt, GenerateRequest};

/// Returns true if the raw model.onnx bytes reference a `QMoE` op. The op name
/// is stored as a length-prefixed string in the serialized NodeProto, so a
/// substring scan is sufficient to prove the fused node survived export.
fn model_contains_qmoe(model_path: &std::path::Path) -> bool {
    let Ok(bytes) = std::fs::read(model_path) else {
        return false;
    };
    bytes.windows(4).any(|w| w == b"QMoE")
}

#[test]
#[ignore = "requires a mobius-exported fused-QMoE tiny glm_moe_dsa model via GLM_TINY_QMOE_E2E_DIR"]
fn glm_tiny_qmoe_e2e() -> anyhow::Result<()> {
    let Some(dir) = std::env::var_os("GLM_TINY_QMOE_E2E_DIR") else {
        eprintln!(
            "skipping glm_tiny_qmoe_e2e: set GLM_TINY_QMOE_E2E_DIR to a mobius-exported \
             fused-QMoE tiny glm_moe_dsa artifact directory (model.onnx + \
             inference_metadata.yaml + tokenizer.json)"
        );
        return Ok(());
    };
    let dir = std::path::PathBuf::from(dir);
    if !dir.is_dir() {
        eprintln!(
            "skipping glm_tiny_qmoe_e2e: GLM_TINY_QMOE_E2E_DIR is absent: {}",
            dir.display()
        );
        return Ok(());
    }

    // Prove the export actually contains the fused QMoE node before running it,
    // so a silent regression to per-expert MatMulNBits can't pass this test.
    let model_path = dir.join("model.onnx");
    assert!(
        model_contains_qmoe(&model_path),
        "model {} does not contain a fused com.microsoft::QMoE node — the routed \
         experts were not fused",
        model_path.display()
    );

    let mut engine = Engine::from_dir(&dir, EngineConfig::default())?;

    let max_new_tokens = 6usize;
    let prompt_ids = vec![1u32, 2, 3, 4];
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(prompt_ids.clone()));
    request.options.max_new_tokens = max_new_tokens;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;

    let result = engine.generate(request)?;

    eprintln!(
        "glm_tiny_qmoe_e2e: prompt={:?} generated {} tokens (prefill + {} decode): {:?}",
        prompt_ids,
        result.token_ids.len(),
        max_new_tokens - 1,
        result.token_ids
    );

    assert_eq!(
        result.token_ids.len(),
        max_new_tokens,
        "expected exactly {max_new_tokens} generated tokens (prefill + decode), got {}",
        result.token_ids.len()
    );

    // Every generated token must be a valid vocab id (vocab_size = 256 in the
    // tiny config). This catches gross shape/logits corruption from the fused op.
    for &tok in &result.token_ids {
        assert!(
            tok < 256,
            "generated token id {tok} is outside the tiny vocab (256)"
        );
    }

    Ok(())
}

//! Structural end-to-end smoke test for a QUANTIZED tiny synthetic GLM-5.2
//! (glm_moe_dsa) model exported by mobius.
//!
//! Sibling of `glm_tiny_synthetic_e2e.rs`, but the artifact is exported with a
//! `QuantizationConfig` (int4, block-32, asymmetric), so the linear projections
//! and per-expert MoE MLPs are emitted as `com.microsoft::MatMulNBits`. This
//! exercises the real int4 GEMM kernel through the full GLM graph (MLA +
//! IndexShare DSA + MoE). Like the fp32 sibling, weights are random, so only
//! *structural* success is asserted: the engine loads the model, runs prefill +
//! several decode steps without panicking, and produces the requested number of
//! in-vocab tokens.
//!
//! The tiny model is built with `mobius/export_glm_tiny_quant.py`. Point
//! `GLM_TINY_Q4_E2E_DIR` at the artifact directory and run:
//!
//! ```bash
//! GLM_TINY_Q4_E2E_DIR=/home/justinchu/glm-e2e-artifacts/glm-5.2-tiny-q4 \
//!   cargo test -p onnx-genai-engine --test glm_tiny_quant_e2e -- --ignored --nocapture
//! ```

use onnx_genai_engine::{Engine, EngineConfig, GeneratePrompt, GenerateRequest};

#[test]
#[ignore = "requires a mobius-exported quantized tiny glm_moe_dsa model via GLM_TINY_Q4_E2E_DIR"]
fn glm_tiny_quant_e2e() -> anyhow::Result<()> {
    let Some(dir) = std::env::var_os("GLM_TINY_Q4_E2E_DIR") else {
        eprintln!(
            "skipping glm_tiny_quant_e2e: set GLM_TINY_Q4_E2E_DIR to a mobius-exported \
             quantized tiny glm_moe_dsa artifact directory (model.onnx + inference_metadata.yaml + tokenizer.json)"
        );
        return Ok(());
    };
    let dir = std::path::PathBuf::from(dir);
    if !dir.is_dir() {
        eprintln!(
            "skipping glm_tiny_quant_e2e: GLM_TINY_Q4_E2E_DIR is absent: {}",
            dir.display()
        );
        return Ok(());
    }

    let mut engine = Engine::from_dir(&dir, EngineConfig::default())?;

    let max_new_tokens = 8usize;
    let prompt_ids = vec![1u32, 2, 3, 4];
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(prompt_ids.clone()));
    request.options.max_new_tokens = max_new_tokens;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;

    let result = engine.generate(request)?;

    eprintln!(
        "glm_tiny_quant_e2e: prompt={:?} generated {} tokens: {:?}",
        prompt_ids,
        result.token_ids.len(),
        result.token_ids
    );

    assert_eq!(
        result.token_ids.len(),
        max_new_tokens,
        "expected exactly {max_new_tokens} generated tokens (prefill + decode), got {}",
        result.token_ids.len()
    );

    // Every generated token must be a valid vocab id (vocab_size = 256 in the
    // tiny config). This catches gross shape/logits corruption.
    for &tok in &result.token_ids {
        assert!(
            tok < 256,
            "generated token id {tok} is outside the tiny vocab (256)"
        );
    }

    Ok(())
}

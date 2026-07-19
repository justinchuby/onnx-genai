//! Structural E2E smoke test for a tiny synthetic DeepSeek-V2 MLA + MoE model.
//!
//! Export with `mobius/export_deepseek_v2_tiny.py`, then run:
//! `DEEPSEEK_V2_TINY_E2E_DIR=/home/justinchu/ds-e2e-artifacts/deepseek-v2-tiny
//! cargo test -p onnx-genai-engine --test deepseek_e2e -- --ignored --nocapture`.

use onnx_genai_engine::{Engine, EngineConfig, GeneratePrompt, GenerateRequest};

#[test]
#[ignore = "requires a mobius-exported tiny DeepSeek-V2 model"]
fn deepseek_v2_tiny_synthetic_e2e() -> anyhow::Result<()> {
    let Some(dir) = std::env::var_os("DEEPSEEK_V2_TINY_E2E_DIR") else {
        eprintln!(
            "skipping deepseek_v2_tiny_synthetic_e2e: set DEEPSEEK_V2_TINY_E2E_DIR \
             to a model.onnx + inference_metadata.yaml + tokenizer.json artifact directory"
        );
        return Ok(());
    };
    let dir = std::path::PathBuf::from(dir);
    if !dir.is_dir() {
        eprintln!(
            "skipping deepseek_v2_tiny_synthetic_e2e: artifact directory is absent: {}",
            dir.display()
        );
        return Ok(());
    }

    let mut engine = Engine::from_dir(&dir, EngineConfig::default())?;
    let prompt_ids = vec![1u32, 2, 3, 4];
    let max_new_tokens = 8;
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(prompt_ids.clone()));
    request.options.max_new_tokens = max_new_tokens;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;

    let result = engine.generate(request)?;
    eprintln!(
        "deepseek_v2_tiny_synthetic_e2e: prompt={prompt_ids:?}; generated {} tokens: {:?}",
        result.token_ids.len(),
        result.token_ids
    );

    assert_eq!(result.token_ids.len(), max_new_tokens);
    assert!(result.token_ids.iter().all(|&token| token < 256));
    Ok(())
}

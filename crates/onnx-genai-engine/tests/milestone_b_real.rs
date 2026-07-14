//! Real-model, env-gated Milestone B harness for shared-KV speculative decode.
//!
//! This is a manual/benchmark harness rather than a hermetic unit test: it is
//! `#[ignore]`d and additionally no-ops unless the two model directories are
//! provided via environment variables, so it never runs in CI and carries no
//! model-specific logic (all paths, the prompt, and the token budget are
//! config-driven).
//!
//! It proves the two Milestone B pass bars for any shared-KV speculative
//! package:
//!   1. token-identity — speculative decode is exact greedy, so the tokens must
//!      match a plain target-only greedy run of the same prompt, and
//!   2. speedup/acceptance — it reports speculative vs greedy tok/s and the
//!      draft acceptance rate.
//!
//! Run (from the full ORT CUDA env):
//! ```bash
//! ONNX_GENAI_EP=cuda \
//! ONNX_GENAI_MB_FULL=$HOME/gemma4-e2b-onnx \
//! ONNX_GENAI_MB_TARGET=$HOME/gemma4-e2b-onnx-target \
//! cargo test -p onnx-genai-engine --test milestone_b_real -- --ignored --nocapture
//! ```

use std::time::Instant;

use onnx_genai_engine::{Engine, EngineConfig, GeneratePrompt, GenerateRequest};
use onnx_genai_runtime_config::runtime_config;

fn request(prompt: &str, max_new_tokens: usize) -> GenerateRequest {
    let mut request = GenerateRequest::new(GeneratePrompt::Text(prompt.to_string()));
    request.options.max_new_tokens = max_new_tokens;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    // Fixed length on both runs makes the token-identity comparison exact.
    request.options.stop_on_eos = false;
    request
}

#[test]
#[ignore = "requires real shared-KV speculative package via ONNX_GENAI_MB_* env vars"]
fn shared_kv_speculative_matches_greedy_and_reports_speedup() -> anyhow::Result<()> {
    let config = runtime_config();
    let (Some(full_dir), Some(target_dir)) =
        (config.mb_full.as_deref(), config.mb_target.as_deref())
    else {
        eprintln!(
            "skipping: set ONNX_GENAI_MB_FULL (merged speculative package) and \
             ONNX_GENAI_MB_TARGET (target-only greedy view) to run this harness"
        );
        return Ok(());
    };
    let prompt = &config.mb_prompt;
    let max_new_tokens = config.mb_max;

    // Plain greedy baseline: the target-only view carries no `speculative:`
    // block, so the engine decodes greedily with no draft model.
    let mut baseline = Engine::from_dir(target_dir, EngineConfig::default())?;
    let started = Instant::now();
    let greedy = baseline.generate(request(prompt, max_new_tokens))?;
    let greedy_secs = started.elapsed().as_secs_f64();

    // Speculative: the merged package advertises the shared-KV proposer in its
    // metadata, so the default config auto-selects the shared-KV path.
    let mut speculative = Engine::from_dir(full_dir, EngineConfig::default())?;
    let started = Instant::now();
    let spec = speculative.generate(request(prompt, max_new_tokens))?;
    let spec_secs = started.elapsed().as_secs_f64();
    let stats = speculative.last_speculative_stats();

    let greedy_toks = greedy.token_ids.len();
    let spec_toks = spec.token_ids.len();
    let greedy_tps = greedy_toks as f64 / greedy_secs;
    let spec_tps = spec_toks as f64 / spec_secs;
    let acceptance = if stats.proposed_tokens > 0 {
        stats.accepted_tokens as f64 / stats.proposed_tokens as f64
    } else {
        0.0
    };

    println!("\n================ MILESTONE B REPORT ================");
    println!("prompt: {prompt:?}");
    println!("max_new_tokens: {max_new_tokens}");
    println!("--- greedy (target-only) ---");
    println!("  tokens: {greedy_toks} in {greedy_secs:.3}s = {greedy_tps:.2} tok/s");
    println!("  text: {:?}", greedy.text);
    println!("--- speculative (shared-KV) ---");
    println!("  tokens: {spec_toks} in {spec_secs:.3}s = {spec_tps:.2} tok/s");
    println!("  text: {:?}", spec.text);
    println!("--- speculative stats ---");
    println!("  verification_steps: {}", stats.verification_steps);
    println!("  proposed_tokens:    {}", stats.proposed_tokens);
    println!("  accepted_tokens:    {}", stats.accepted_tokens);
    println!("  multi_token_accepts:{}", stats.multi_token_accepts);
    println!("  acceptance_rate:    {:.1}%", acceptance * 100.0);
    println!("  speedup:            {:.2}x", spec_tps / greedy_tps);
    println!("  token_identical:    {}", greedy.token_ids == spec.token_ids);
    println!("====================================================\n");

    assert!(
        stats.proposed_tokens > 0,
        "speculative path did not propose any draft tokens: {stats:?}"
    );
    assert!(
        stats.verification_steps > 0,
        "speculative path ran no verification steps: {stats:?}"
    );
    assert_eq!(
        greedy.token_ids, spec.token_ids,
        "shared-KV speculative decode diverged from plain greedy (must be exact)\n\
         greedy: {:?}\nspec:   {:?}",
        greedy.token_ids, spec.token_ids
    );
    assert_eq!(greedy.finish_reason, spec.finish_reason);

    Ok(())
}

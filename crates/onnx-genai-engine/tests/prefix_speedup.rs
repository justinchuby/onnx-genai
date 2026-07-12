use onnx_genai_engine::{Engine, EngineConfig, FinishReason, GeneratePrompt, GenerateRequest};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn tiny_fixture() -> anyhow::Result<PathBuf> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-llm")
        .canonicalize()?)
}

fn token_request(tokens: Vec<u32>, max_new_tokens: usize) -> GenerateRequest {
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(tokens));
    request.options.max_new_tokens = max_new_tokens;
    request.options.temperature = 0.0;
    request.options.stop_on_eos = false;
    request
}

#[test]
fn second_turn_latency_is_reported_for_prefix_cache_validation() -> anyhow::Result<()> {
    let mut engine = Engine::from_dir(&tiny_fixture()?, EngineConfig::default())?;
    let session_id = engine.create_session()?;

    let cold_start = Instant::now();
    let cold = engine.generate_in_session(session_id, token_request(vec![2, 4, 5, 3], 2))?;
    let cold_duration = cold_start.elapsed();
    let after_cold = engine.session_token_count(session_id)?;

    let warm_start = Instant::now();
    let warm = engine.generate_in_session(session_id, token_request(vec![6, 7], 2))?;
    let warm_duration = warm_start.elapsed();
    let after_warm = engine.session_token_count(session_id)?;

    eprintln!(
        "prefix speed harness: cold_turn={:?} warm_turn={:?} cold_prefix_cache_hit_len={} warm_prefix_cache_hit_len={}",
        cold_duration, warm_duration, cold.prefix_cache_hit_len, warm.prefix_cache_hit_len
    );

    assert_eq!(cold.token_ids.len(), 2);
    assert_eq!(warm.token_ids.len(), 2);
    assert_eq!(cold.finish_reason, FinishReason::MaxTokens);
    assert_eq!(warm.finish_reason, FinishReason::MaxTokens);
    assert_eq!(after_cold, 6);
    assert_eq!(after_warm, 10);
    assert_eq!(cold.prefix_cache_hit_len, 0);
    assert!(
        warm.prefix_cache_hit_len > 0,
        "second turn should report a same-session prefix cache hit"
    );

    engine.close_session(session_id)?;
    Ok(())
}

#[test]
fn repeated_prompt_reports_cross_session_prefix_hit() -> anyhow::Result<()> {
    let mut engine = Engine::from_dir(&tiny_fixture()?, EngineConfig::default())?;
    let cached_prefix = vec![2, 4, 5, 3];
    let extended_prompt = vec![2, 4, 5, 3, 6];

    let cold = engine.generate(token_request(cached_prefix.clone(), 2))?;
    let warm = engine.generate(token_request(extended_prompt, 2))?;

    assert_eq!(cold.prefix_cache_hit_len, 0);
    assert_eq!(warm.prefix_cache_hit_len, cached_prefix.len());
    Ok(())
}

#[test]
fn greedy_output_matches_with_and_without_prefix_reuse() -> anyhow::Result<()> {
    let fixture = tiny_fixture()?;
    let cached_prefix = vec![2, 4, 5, 3];
    let extended_prompt = vec![2, 4, 5, 3, 6];
    let cold = Engine::from_dir(&fixture, EngineConfig::default())?
        .generate(token_request(extended_prompt.clone(), 3))?;

    let mut cached = Engine::from_dir(&fixture, EngineConfig::default())?;
    let _ = cached.generate(token_request(cached_prefix, 3))?;
    let warm = cached.generate(token_request(extended_prompt, 3))?;

    assert!(warm.prefix_cache_hit_len > 0);
    assert_eq!(warm.token_ids, cold.token_ids);
    assert_eq!(warm.finish_reason, cold.finish_reason);
    Ok(())
}

use onnx_genai_engine::{Engine, EngineConfig, FinishReason, GeneratePrompt, GenerateRequest};
use std::path::{Path, PathBuf};

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
fn interleaved_sessions_keep_independent_contexts() -> anyhow::Result<()> {
    let mut engine = Engine::from_dir(&tiny_fixture()?, EngineConfig::default())?;
    let session_a = engine.create_session()?;
    let session_b = engine.create_session()?;
    let session_c = engine.create_session()?;

    assert_eq!(engine.session_token_count(session_a)?, 0);
    assert_eq!(engine.session_token_count(session_b)?, 0);
    assert_eq!(engine.session_token_count(session_c)?, 0);

    let a1 = engine.generate_in_session(session_a, token_request(vec![2, 4, 3], 2))?;
    assert_eq!(a1.token_ids.len(), 2);
    assert_eq!(a1.finish_reason, FinishReason::MaxTokens);
    assert_eq!(engine.session_token_count(session_a)?, 5);
    assert_eq!(engine.session_token_count(session_b)?, 0);
    assert_eq!(engine.session_token_count(session_c)?, 0);

    let b1 = engine.generate_in_session(session_b, token_request(vec![2, 5, 3], 1))?;
    assert_eq!(b1.token_ids.len(), 1);
    assert_eq!(b1.finish_reason, FinishReason::MaxTokens);
    assert_eq!(engine.session_token_count(session_a)?, 5);
    assert_eq!(engine.session_token_count(session_b)?, 4);
    assert_eq!(engine.session_token_count(session_c)?, 0);

    let a2 = engine.generate_in_session(session_a, token_request(vec![6], 1))?;
    assert_eq!(a2.token_ids.len(), 1);
    assert_eq!(engine.session_token_count(session_a)?, 7);
    assert_eq!(engine.session_token_count(session_b)?, 4);
    assert_eq!(engine.session_token_count(session_c)?, 0);

    let c1 = engine.generate_in_session(session_c, token_request(vec![2, 7, 8, 3], 2))?;
    assert_eq!(c1.token_ids.len(), 2);
    assert_eq!(engine.session_token_count(session_a)?, 7);
    assert_eq!(engine.session_token_count(session_b)?, 4);
    assert_eq!(engine.session_token_count(session_c)?, 6);

    let b2 = engine.generate_in_session(session_b, token_request(vec![9, 3], 3))?;
    assert_eq!(b2.token_ids.len(), 3);
    assert_eq!(engine.session_token_count(session_a)?, 7);
    assert_eq!(engine.session_token_count(session_b)?, 9);
    assert_eq!(engine.session_token_count(session_c)?, 6);

    engine.close_session(session_a)?;
    engine.close_session(session_b)?;
    engine.close_session(session_c)?;
    Ok(())
}

#[test]
fn reset_clears_only_the_target_session() -> anyhow::Result<()> {
    let mut engine = Engine::from_dir(&tiny_fixture()?, EngineConfig::default())?;
    let session_a = engine.create_session()?;
    let session_b = engine.create_session()?;

    engine.generate_in_session(session_a, token_request(vec![2, 4, 3], 1))?;
    engine.generate_in_session(session_b, token_request(vec![2, 5, 3], 2))?;
    assert_eq!(engine.session_token_count(session_a)?, 4);
    assert_eq!(engine.session_token_count(session_b)?, 5);

    engine.reset_session(session_a)?;
    assert_eq!(engine.session_token_count(session_a)?, 0);
    assert_eq!(engine.session_token_count(session_b)?, 5);

    engine.generate_in_session(session_a, token_request(vec![2, 6, 3], 2))?;
    assert_eq!(engine.session_token_count(session_a)?, 5);
    assert_eq!(engine.session_token_count(session_b)?, 5);

    engine.close_session(session_a)?;
    engine.close_session(session_b)?;
    Ok(())
}

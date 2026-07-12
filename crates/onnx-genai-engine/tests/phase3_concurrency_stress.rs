use onnx_genai_engine::{Engine, EngineConfig, FinishReason, GeneratePrompt, GenerateRequest};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const SESSION_COUNT: usize = 12;
const FIRST_TURN_TOKENS: usize = 2;
const SECOND_TURN_TOKENS: usize = 2;

fn tiny_fixture() -> anyhow::Result<PathBuf> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-llm")
        .canonicalize()?)
}

fn request(tokens: Vec<u32>, max_new_tokens: usize) -> GenerateRequest {
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(tokens));
    request.options.max_new_tokens = max_new_tokens;
    request.options.temperature = 0.0;
    request.options.stop_on_eos = false;
    request.options.max_context = Some(32);
    request
}

#[test]
fn ten_plus_interleaved_sessions_complete_under_kv_page_pressure() -> anyhow::Result<()> {
    let config = EngineConfig {
        num_gpu_pages: 2,
        page_size: 2,
        ..EngineConfig::default()
    };
    let mut engine = Engine::from_dir(&tiny_fixture()?, config)?;
    let sessions = (0..SESSION_COUNT)
        .map(|_| engine.create_session())
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut expected_lengths = HashMap::new();

    for (idx, session_id) in sessions.iter().copied().enumerate() {
        let prompt = vec![2, 4 + (idx as u32 % 6), 3];
        let result =
            engine.generate_in_session(session_id, request(prompt.clone(), FIRST_TURN_TOKENS))?;
        assert_eq!(result.token_ids.len(), FIRST_TURN_TOKENS);
        assert_eq!(result.finish_reason, FinishReason::MaxTokens);
        let expected = prompt.len() + FIRST_TURN_TOKENS;
        expected_lengths.insert(session_id, expected);

        for prior in sessions.iter().copied().take(idx) {
            assert_eq!(engine.session_token_count(prior)?, expected_lengths[&prior]);
        }
        for later in sessions.iter().copied().skip(idx + 1) {
            assert_eq!(engine.session_token_count(later)?, 0);
        }
    }

    for (round_idx, session_id) in sessions.iter().copied().rev().enumerate() {
        let prompt = vec![8 + (round_idx as u32 % 4), 3];
        let result =
            engine.generate_in_session(session_id, request(prompt.clone(), SECOND_TURN_TOKENS))?;
        assert_eq!(result.token_ids.len(), SECOND_TURN_TOKENS);
        assert_eq!(result.finish_reason, FinishReason::MaxTokens);
        *expected_lengths.get_mut(&session_id).unwrap() += prompt.len() + SECOND_TURN_TOKENS;
        assert_eq!(
            engine.session_token_count(session_id)?,
            expected_lengths[&session_id]
        );
    }

    for session_id in sessions.iter().copied() {
        assert_eq!(
            engine.session_token_count(session_id)?,
            expected_lengths[&session_id]
        );
        engine.close_session(session_id)?;
    }
    Ok(())
}

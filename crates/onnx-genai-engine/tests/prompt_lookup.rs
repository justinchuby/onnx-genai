use onnx_genai_engine::{Engine, EngineConfig, GeneratePrompt, GenerateRequest, SpeculativeMode};
use onnx_genai_ort::SessionOptions;
use std::path::{Path, PathBuf};

fn tiny_llm() -> anyhow::Result<PathBuf> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-llm")
        .canonicalize()?)
}

fn deterministic_engine(
    fixture: &Path,
    speculative_mode: SpeculativeMode,
) -> anyhow::Result<Engine> {
    Engine::from_dir_with_session_options(
        fixture,
        EngineConfig {
            speculative_mode,
            ..Default::default()
        },
        SessionOptions::default().with_intra_op_threads(1),
    )
}

fn greedy_request(prompt: GeneratePrompt, max_new_tokens: usize) -> GenerateRequest {
    let mut request = GenerateRequest::new(prompt);
    request.options.max_new_tokens = max_new_tokens;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;
    request
}

#[test]
fn prompt_lookup_greedy_matches_plain_greedy_exactly() -> anyhow::Result<()> {
    let fixture = tiny_llm()?;
    let mut baseline = deterministic_engine(&fixture, SpeculativeMode::None)?;
    let mut prompt_lookup = deterministic_engine(
        &fixture,
        SpeculativeMode::PromptLookup {
            ngram: 1,
            max_tokens: 4,
        },
    )?;
    let request = greedy_request(GeneratePrompt::Text("hello".to_string()), 12);

    let expected = baseline.generate(request.clone())?;
    let actual = prompt_lookup.generate(request)?;

    assert_eq!(actual.token_ids, expected.token_ids);
    assert_eq!(actual.finish_reason, expected.finish_reason);
    assert!(prompt_lookup.last_speculative_stats().proposed_tokens > 0);
    Ok(())
}

#[test]
fn prompt_lookup_accepts_multiple_tokens_on_repetitive_context() -> anyhow::Result<()> {
    let fixture = tiny_llm()?;
    let prompt = vec![3, 26, 11, 9, 29, 3, 26, 11, 9, 29];
    let request = greedy_request(GeneratePrompt::TokenIds(prompt), 4);
    let mut baseline = deterministic_engine(&fixture, SpeculativeMode::None)?;
    let mut prompt_lookup = deterministic_engine(
        &fixture,
        SpeculativeMode::PromptLookup {
            ngram: 1,
            max_tokens: 4,
        },
    )?;

    let expected = baseline.generate(request.clone())?;
    let actual = prompt_lookup.generate(request)?;
    let stats = prompt_lookup.last_speculative_stats();

    assert_eq!(actual.token_ids, expected.token_ids);
    assert_eq!(actual.finish_reason, expected.finish_reason);
    assert!(stats.accepted_tokens >= 2, "{stats:?}");
    assert!(stats.multi_token_accepts >= 1, "{stats:?}");
    Ok(())
}

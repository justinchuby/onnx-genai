use onnx_genai_engine::{
    Engine, EngineConfig, GeneratePrompt, GenerateRequest, SpeculativeMode,
    SpeculativeTraceFamily,
};
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
    let mut request = greedy_request(GeneratePrompt::Text("hello".to_string()), 12);
    request.options.top_logprobs = Some(3);

    let expected = baseline.generate(request.clone())?;
    let actual = prompt_lookup.generate(request)?;
    let stats = prompt_lookup.last_speculative_stats();

    assert_eq!(actual.token_ids, expected.token_ids);
    assert_eq!(actual.logprobs, expected.logprobs);
    assert_eq!(actual.finish_reason, expected.finish_reason);
    assert!(stats.proposed_tokens > 0);
    assert!(
        stats.accepted_tokens < stats.proposed_tokens,
        "fixture should exercise correction after a rejected proposal: {stats:?}"
    );
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

#[test]
fn speculative_trace_records_only_executed_token_decisions() -> anyhow::Result<()> {
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

    let (expected, target_trace) = baseline.generate_with_speculative_trace(request.clone())?;
    let (actual, speculative_trace) = prompt_lookup.generate_with_speculative_trace(request)?;

    assert_eq!(actual.token_ids, expected.token_ids);
    assert_eq!(target_trace.output_token_ids, expected.token_ids);
    assert_eq!(target_trace.family, None);
    assert_eq!(target_trace.max_additional_tokens, None);
    assert!(target_trace.iterations.is_empty());
    assert_eq!(speculative_trace.output_token_ids, actual.token_ids);
    assert_eq!(
        speculative_trace.family,
        Some(SpeculativeTraceFamily::PromptLookup)
    );
    assert_eq!(speculative_trace.max_additional_tokens, Some(4));
    assert!(!speculative_trace.iterations.is_empty());

    let mut output_offset = 0;
    let mut reconstructed = Vec::new();
    let mut observed_correction = false;
    for iteration in &speculative_trace.iterations {
        assert_eq!(iteration.output_offset, output_offset);
        output_offset += iteration.committed_token_ids.len();
        reconstructed.extend_from_slice(&iteration.committed_token_ids);

        let accepted = iteration
            .proposal_token_ids
            .iter()
            .zip(&iteration.target_token_ids)
            .take_while(|(proposal, target)| proposal == target)
            .count();
        if accepted < iteration.proposal_token_ids.len() {
            observed_correction = true;
            assert_eq!(
                iteration.target_token_ids.len(),
                accepted + 1,
                "capture must stop target selections at the first mismatch"
            );
        } else {
            assert!(
                iteration.target_token_ids.len() == iteration.proposal_token_ids.len()
                    || iteration.target_token_ids.len() == iteration.proposal_token_ids.len() + 1,
                "full acceptance may record only an executed bonus row"
            );
        }
    }
    assert!(observed_correction, "fixture must exercise correction");
    assert_eq!(reconstructed, actual.token_ids);
    Ok(())
}

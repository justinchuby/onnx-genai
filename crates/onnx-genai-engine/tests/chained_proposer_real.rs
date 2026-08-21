use onnx_genai_engine::{
    Eagle3Config, Engine, EngineConfig, GeneratePrompt, GenerateRequest, SpeculativeMode,
};
use onnx_genai_ort::Eagle3DraftKvMode;
use std::path::{Path, PathBuf};

fn request(prompt: &str) -> GenerateRequest {
    let mut request = GenerateRequest::new(GeneratePrompt::Text(prompt.to_string()));
    request.options.max_new_tokens = 12;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;
    request
}

fn engine(target: &Path, package: &Path, speculative: bool) -> anyhow::Result<Engine> {
    let speculative_mode = speculative
        .then(|| {
            SpeculativeMode::Eagle3(Eagle3Config {
                head_model: package.join("proposer/model.onnx"),
                target_hidden_outputs: vec![
                    "hidden_states.2".into(),
                    "hidden_states.14".into(),
                    "hidden_states.25".into(),
                ],
                embedding_weights: package.join("target_embedding.f32"),
                token_map: Some(package.join("draft_to_target.i64")),
                vocab_size: 151_936,
                hidden_size: 1024,
                kv_mode: Eagle3DraftKvMode::GrowCache,
                num_speculative_tokens: 6,
            })
        })
        .unwrap_or(SpeculativeMode::None);
    Engine::from_dir(
        target,
        EngineConfig {
            speculative_mode,
            num_speculative_tokens: 6,
            ..EngineConfig::default()
        },
    )
}

#[test]
fn real_chained_proposer_matches_target_and_accepts_and_rejects() -> anyhow::Result<()> {
    let Some(package) = std::env::var_os("ONNX_GENAI_CHAINED_SPEC_PACKAGE").map(PathBuf::from)
    else {
        eprintln!("skipping real chained-proposer test; set ONNX_GENAI_CHAINED_SPEC_PACKAGE");
        return Ok(());
    };
    let target = package.join("runtime-target");
    if !target.join("model.onnx").is_file() {
        anyhow::bail!("runtime target is missing at {}", target.display());
    }

    let prompts = [
        "The capital of France is",
        "Write a Python function that adds two integers.",
        "Complete the sequence: 1, 1, 2, 3, 5,",
        "A quick brown fox",
        "Explain gravity in one sentence.",
    ];
    let mut accepted = 0usize;
    let mut rejected = 0usize;
    let mut generations = Vec::new();
    for prompt in prompts {
        let mut baseline = engine(&target, &package, false)?;
        let expected = baseline.generate(request(prompt))?;
        let mut speculative = engine(&target, &package, true)?;
        let actual = speculative.generate(request(prompt))?;
        let stats = speculative.last_speculative_stats();
        assert_eq!(
            actual.token_ids, expected.token_ids,
            "L4 token parity failed"
        );
        assert_eq!(actual.text, expected.text, "L4 text parity failed");
        assert!(
            actual.token_ids.len() > 1,
            "L5 generation must emit multiple tokens"
        );
        accepted += stats.accepted_tokens;
        rejected += stats.proposed_tokens.saturating_sub(stats.accepted_tokens);
        generations.push((prompt, actual.text, actual.token_ids, stats));
        if accepted > 0 && rejected > 0 {
            break;
        }
    }
    assert!(accepted > 0, "real chained proposals accepted no tokens");
    assert!(rejected > 0, "real chained proposals rejected no tokens");
    eprintln!(
        "REAL_CHAINED_SPEC_EVIDENCE accepted={accepted} rejected={rejected} generations={generations:?}"
    );
    Ok(())
}

use onnx_genai_engine::{
    Engine, EngineConfig, GeneratePrompt, GenerateRequest, SharedKvBinding, SharedKvProposerConfig,
    SpeculativeMode,
};
use onnx_genai_ort::SessionOptions;
use std::path::{Path, PathBuf};

fn fixture() -> anyhow::Result<PathBuf> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-gemma4-assistant")
        .canonicalize()?)
}

fn engine(fixture: &Path, speculative_mode: SpeculativeMode) -> anyhow::Result<Engine> {
    Engine::from_dir_with_session_options(
        fixture,
        EngineConfig {
            speculative_mode,
            ..Default::default()
        },
        SessionOptions::default().with_intra_op_threads(1),
    )
}

fn assistant_config(fixture: &Path) -> SharedKvProposerConfig {
    SharedKvProposerConfig {
        assistant_model: fixture.join("assistant/model.onnx"),
        target_hidden_output: "hidden_states.0".into(),
        input_embedding_weights: fixture.join("input_embedding.f32"),
        backbone_hidden_size: 16,
        vocab_size: 32,
        num_speculative_tokens: 4,
        shared_kv: vec![
            SharedKvBinding {
                name: "sliding_attention".into(),
                target_layers: vec![0],
            },
            SharedKvBinding {
                name: "full_attention".into(),
                target_layers: vec![1],
            },
        ],
    }
}

fn request() -> GenerateRequest {
    let mut request = GenerateRequest::new(GeneratePrompt::Text("hello world".to_string()));
    request.options.max_new_tokens = 8;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;
    request
}

/// The shared-KV proposer shares slices of the target's paged KV cache and
/// threads `projected_state` forward. Because the target verifies every drafted
/// token, speculative decoding must reproduce plain greedy decoding
/// token-for-token.
#[test]
fn gemma4_assistant_speculative_generation_matches_plain_greedy() -> anyhow::Result<()> {
    let fixture = fixture()?;
    let mut baseline = engine(&fixture, SpeculativeMode::None)?;
    let mut assistant = engine(
        &fixture,
        SpeculativeMode::SharedKv(assistant_config(&fixture)),
    )?;

    let expected = baseline.generate(request())?;
    let actual = assistant.generate(request())?;
    let stats = assistant.last_speculative_stats();

    assert_eq!(actual.token_ids, expected.token_ids);
    assert_eq!(actual.finish_reason, expected.finish_reason);
    assert!(stats.proposed_tokens > 0, "{stats:?}");
    assert!(stats.verification_steps > 0, "{stats:?}");
    Ok(())
}

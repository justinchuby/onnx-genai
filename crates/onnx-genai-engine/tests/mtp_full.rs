use onnx_genai_engine::{
    Engine, EngineConfig, GeneratePrompt, GenerateRequest, MtpConfig, SpeculativeMode,
};
use onnx_genai_ort::{MtpDraftKvMode, SessionOptions};
use std::path::{Path, PathBuf};

fn fixture() -> anyhow::Result<PathBuf> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-mtp-full")
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

fn request() -> GenerateRequest {
    let mut request = GenerateRequest::new(GeneratePrompt::Text("hello".to_string()));
    request.options.max_new_tokens = 8;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.stop_on_eos = false;
    request
}

fn pre_phase1_mtp_mode(fixture: &Path) -> SpeculativeMode {
    SpeculativeMode::Mtp(MtpConfig {
        head_model: fixture.join("mtp/model.onnx"),
        target_hidden_output: "hidden_states.0".into(),
        embedding_weights: fixture.join("embedding.f32"),
        lm_head_weights: fixture.join("lm_head.f32"),
        vocab_size: 32,
        hidden_size: 16,
        kv_mode: MtpDraftKvMode::HiddenThreaded,
        num_speculative_tokens: 4,
    })
}

#[test]
fn pre_phase1_mtp_config_literal_remains_source_compatible() {
    let mode = pre_phase1_mtp_mode(Path::new("tiny-mtp-full"));
    assert!(matches!(mode, SpeculativeMode::Mtp(_)));
}

#[test]
#[ignore = "random Mobius fixture; run explicitly to exercise target -> MTP head -> verify"]
fn mtp_speculative_generation_matches_plain_greedy() -> anyhow::Result<()> {
    let fixture = fixture()?;
    let mut baseline = engine(&fixture, SpeculativeMode::None)?;
    let mut mtp = engine(&fixture, pre_phase1_mtp_mode(&fixture))?;

    let expected = baseline.generate(request())?;
    let actual = mtp.generate(request())?;
    let stats = mtp.last_speculative_stats();

    assert_eq!(actual.token_ids, expected.token_ids);
    assert_eq!(actual.finish_reason, expected.finish_reason);
    assert!(stats.proposed_tokens > 0, "{stats:?}");
    assert!(stats.verification_steps > 0, "{stats:?}");
    Ok(())
}

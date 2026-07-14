//! Regression test: validate the full mobius-emitted metadata -> runtime
//! auto-config chain for the shared-KV (Gemma4-Assistant) proposer.
//!
//! Baseline engine loads the fixture WITHOUT metadata (plain greedy). The
//! speculative engine loads a copy WITH a mobius-style `inference_metadata.yaml`
//! `speculative:` block and `EngineConfig::default()` (speculative auto-enabled
//! from metadata via `shared_kv_mode_from_metadata`). Asserts token-identity and
//! reports tokens/sec for both.

use onnx_genai_engine::{
    Engine, EngineConfig, GeneratePrompt, GenerateRequest, SpeculativeMode,
};
use onnx_genai_ort::SessionOptions;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-gemma4-assistant")
        .canonicalize()
        .unwrap()
}

fn copy_dir(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).unwrap();
    for entry in std::fs::read_dir(src).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        let target = dst.join(entry.file_name());
        if path.is_dir() {
            copy_dir(&path, &target);
        } else {
            std::fs::copy(&path, &target).unwrap();
        }
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

const METADATA: &str = r#"required_capabilities:
  - grouped_query_attention
model:
  attention:
    type: group_query_attention
    num_kv_heads: 2
    num_attention_heads: 2
    head_dim: 8
  max_sequence_length: 4096
  runtime_configurable:
    kv_cache:
      dtype:
        - float32
kv_cache:
  native_dtype: float32
speculative:
  proposal_type: shared_kv
  num_speculative_tokens: 4
  model: assistant/model.onnx
  backbone_hidden_size: 16
  vocab_size: 32
  projected_state_output: projected_state
  logits_output: logits
  shared_kv:
    - name: sliding_attention
      target_layers: [0]
    - name: full_attention
      target_layers: [1]
"#;

#[test]
fn gemma4_assistant_metadata_driven_matches_plain_greedy() -> anyhow::Result<()> {
    let fixture = fixture();
    let opts = SessionOptions::default().with_intra_op_threads(1);

    // Speculative package: fixture + mobius-style metadata (auto-config).
    let spec_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/gemma4-meta-smoke");
    let _ = std::fs::remove_dir_all(&spec_dir);
    copy_dir(&fixture, &spec_dir);
    std::fs::write(spec_dir.join("inference_metadata.yaml"), METADATA)?;

    // Baseline: original fixture (no metadata) with speculative explicitly off.
    let mut baseline = Engine::from_dir_with_session_options(
        &fixture,
        EngineConfig {
            speculative_mode: SpeculativeMode::None,
            ..Default::default()
        },
        opts.clone(),
    )?;

    // Speculative: default config -> metadata auto-enables the shared-KV proposer.
    let mut speculative = Engine::from_dir_with_session_options(
        &spec_dir,
        EngineConfig::default(),
        opts,
    )?;

    let t0 = Instant::now();
    let expected = baseline.generate(request())?;
    let greedy_secs = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let actual = speculative.generate(request())?;
    let spec_secs = t1.elapsed().as_secs_f64();
    let stats = speculative.last_speculative_stats();

    println!("greedy tokens={} time={:.4}s tok/s={:.2}",
        expected.token_ids.len(), greedy_secs,
        expected.token_ids.len() as f64 / greedy_secs);
    println!("spec   tokens={} time={:.4}s tok/s={:.2}",
        actual.token_ids.len(), spec_secs,
        actual.token_ids.len() as f64 / spec_secs);
    println!("spec stats: {stats:?}");
    println!("greedy ids: {:?}", expected.token_ids);
    println!("spec   ids: {:?}", actual.token_ids);

    assert_eq!(actual.token_ids, expected.token_ids, "token-identity");
    assert_eq!(actual.finish_reason, expected.finish_reason);
    assert!(stats.proposed_tokens > 0, "metadata did not enable proposer: {stats:?}");
    assert!(stats.verification_steps > 0, "{stats:?}");

    let _ = std::fs::remove_dir_all(&spec_dir);
    Ok(())
}

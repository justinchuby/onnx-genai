//! `Engine::batching_capability()` must match the functional past/present path
//! selected for bare decoder metadata. Workflow KV services cover composite
//! shared/paged serving; the native path has a unit test in `engine::runtime`.

use onnx_genai_engine::{Engine, EngineConfig};
use onnx_genai_ort::SessionOptions;
use std::path::{Path, PathBuf};

fn fixture(name: &str) -> anyhow::Result<PathBuf> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .join(name)
        .canonicalize()?)
}

fn cpu_engine(model_dir: &Path) -> anyhow::Result<Engine> {
    Engine::from_dir_with_session_options(
        model_dir,
        EngineConfig::default(),
        SessionOptions::default().with_intra_op_threads(1),
    )
}

/// The plain past/present fixture has no shared KV buffer on the CPU EP, so it
/// cannot batch. The capability must report `Some(1)`/unsupported, and the real
/// `continuous_batch_manager` must refuse a width > 1 — the two agree.
#[test]
fn past_present_capability_matches_single_sequence_decode() -> anyhow::Result<()> {
    let fixture = fixture("tiny-llm")?;
    let engine = cpu_engine(&fixture)?;

    let capability = engine.batching_capability();
    assert!(
        !capability.supports_batching(),
        "non-shared-buffer past/present decode cannot batch: {}",
        capability.reason()
    );
    assert_eq!(capability.max_concurrent_sequences(), Some(1));
    assert!(!capability.allows(2));
    assert_eq!(capability.effective_max_batch(4), 1);

    // Reality check: the capability's negative claim is backed by the manager
    // actually refusing a width > 1.
    assert!(
        engine.continuous_batch_manager(2).is_err(),
        "non-batching engine must refuse a width-2 continuous batch manager"
    );
    Ok(())
}

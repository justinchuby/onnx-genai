//! `Engine::batching_capability()` must match what the decode path can actually
//! do, for the ORT decode paths that a CPU CI machine can load. The native path
//! is covered by a unit test in `engine::runtime` (it needs no model files).

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

/// The static-cache fixture decodes a shared batch, so the reported capability
/// must say batching is supported AND a real `continuous_batch_manager` must be
/// constructible at that width. If the two ever disagreed, the capability report
/// would be lying to operators — which is the whole failure this issue is about.
#[test]
fn static_cache_capability_matches_batched_manager() -> anyhow::Result<()> {
    let fixture = fixture("tiny-llm-scatter")?;
    let mut engine = cpu_engine(&fixture)?;

    let capability = engine.batching_capability();
    assert!(
        capability.supports_batching(),
        "static-cache decode supports batching: {}",
        capability.reason()
    );
    assert_eq!(
        capability.max_concurrent_sequences(),
        None,
        "static-cache batching is bounded by memory, not a fixed cap"
    );
    assert!(capability.allows(4));
    assert_eq!(capability.effective_max_batch(4), 4);

    // Reality check: the capability's positive claim is backed by a real
    // batched manager at the same width.
    assert!(
        engine.continuous_batch_manager(4).is_ok(),
        "static-cache engine must build a width-4 continuous batch manager"
    );
    Ok(())
}

/// The plain past/present fixture has no shared KV buffer on the CPU EP, so it
/// cannot batch. The capability must report `Some(1)`/unsupported, and the real
/// `continuous_batch_manager` must refuse a width > 1 — the two agree.
#[test]
fn past_present_capability_matches_single_sequence_decode() -> anyhow::Result<()> {
    let fixture = fixture("tiny-llm")?;
    let mut engine = cpu_engine(&fixture)?;

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

//! `Engine::batching_capability()` must match the functional past/present path
//! selected for bare decoder metadata. Workflow KV services cover composite
//! shared/paged serving; the native path has a unit test in `engine::runtime`.

use onnx_genai_engine::{Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest};
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

/// A missing shared-KV backend is an optimization refusal, not a package-load
/// or request-execution failure.  The non-shared past/present fixture gives the
/// public batch API a real unsupported backend while preserving the ordinary
/// isolated decoder path as an oracle.
#[test]
fn unsupported_shared_forward_falls_back_to_isolated_requests() -> anyhow::Result<()> {
    let fixture = fixture("tiny-llm")?;
    let requests = vec![
        GenerateRequest {
            prompt: GeneratePrompt::TokenIds(vec![1]),
            options: GenerateOptions {
                greedy: true,
                stop_on_eos: false,
                max_new_tokens: 2,
                ..GenerateOptions::default()
            },
        },
        GenerateRequest {
            prompt: GeneratePrompt::TokenIds(vec![1, 2, 3]),
            options: GenerateOptions {
                greedy: true,
                stop_on_eos: false,
                max_new_tokens: 5,
                ..GenerateOptions::default()
            },
        },
    ];

    let mut isolated = cpu_engine(&fixture)?;
    let expected = requests
        .iter()
        .cloned()
        .map(|request| isolated.generate(request))
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut batched = cpu_engine(&fixture)?;
    let actual = batched.generate_batched_static(requests.clone())?;

    assert_eq!(actual, expected);
    assert!(
        !batched.batching_capability().supports_batching(),
        "this fixture must exercise the isolated fallback"
    );

    let mut scheduled = cpu_engine(&fixture)?;
    assert_eq!(
        scheduled.run_continuous_batch_scheduled(requests, 2)?,
        expected,
        "scheduler admission must also decline unsupported shared forwarding before \
         request-visible work and preserve isolated results"
    );
    Ok(())
}

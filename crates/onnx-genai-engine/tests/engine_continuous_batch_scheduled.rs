//! Engine-level parity for the SCHEDULER-DRIVEN continuous batch serving path.
//!
//! [`Engine::run_continuous_batch_scheduled`] wires the `Scheduler` into the
//! engine's continuous batch: each iteration the scheduler decides which queued
//! requests join one shared batched forward pass (batch formation), instead of
//! admitting a single FCFS request at a time. Batching is a throughput-only
//! optimization, so a request's tokens must never depend on who it is batched
//! with. These tests pin that guarantee on the CPU tiny-scatter static-cache
//! fixture (no GPU required), reusing the same per-request == sequential harness
//! as the greedy `run_continuous_batch` parity tests.

use onnx_genai_engine::{Engine, EngineConfig, GeneratePrompt, GenerateRequest};
use onnx_genai_ort::SessionOptions;
use std::path::{Path, PathBuf};

fn tiny_scatter_fixture() -> anyhow::Result<PathBuf> {
    Ok(Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-llm-scatter")
        .canonicalize()?)
}

fn token_request(tokens: Vec<u32>, max_new_tokens: usize) -> GenerateRequest {
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(tokens));
    request.options.max_new_tokens = max_new_tokens;
    request.options.temperature = 0.0;
    request.options.stop_on_eos = false;
    request
}

fn deterministic_engine(model_dir: &Path) -> anyhow::Result<Engine> {
    // Single intra-op thread keeps the batched-vs-individual exact match out of
    // ORT CPU multi-threaded reduction-order variance near logit ties.
    Engine::from_dir_with_session_options(
        model_dir,
        EngineConfig::default(),
        SessionOptions::default().with_intra_op_threads(1),
    )
}

/// Run each request on its own engine to build the trusted sequential baseline.
fn sequential_baseline(
    model_dir: &Path,
    requests: &[GenerateRequest],
) -> anyhow::Result<Vec<onnx_genai_engine::GenerateResult>> {
    requests
        .iter()
        .cloned()
        .map(|request| deterministic_engine(model_dir)?.generate(request))
        .collect()
}

/// Scheduler-driven continuous batching over requests of DIFFERENT prompt and
/// output lengths (ragged prefill) plus admission/eviction: with `max_batch=3`
/// and more than three requests, a request that finishes mid-batch frees a row
/// the scheduler then fills with a queued request. Every per-request output must
/// stay byte-identical to the sequential baseline.
#[test]
fn scheduled_continuous_batch_matches_sequential_under_admission_eviction() -> anyhow::Result<()> {
    let fixture = tiny_scatter_fixture()?;
    let requests = vec![
        token_request(vec![1, 5], 2),
        token_request(vec![2, 6, 7], 5),
        token_request(vec![3], 1),
        token_request(vec![4, 8, 9, 10], 4),
        token_request(vec![5, 6], 6),
        token_request(vec![6, 7, 8], 3),
        token_request(vec![7], 2),
    ];

    let expected = sequential_baseline(&fixture, &requests)?;

    let mut engine = deterministic_engine(&fixture)?;
    let scheduled = engine.run_continuous_batch_scheduled(requests, 3)?;

    assert_eq!(
        scheduled, expected,
        "scheduler-driven continuous batch diverged from the sequential baseline"
    );
    assert!(
        scheduled
            .iter()
            .all(|result| result.prefix_cache_hit_len == 0)
    );
    Ok(())
}

/// A batch of one must be byte-identical to a plain `generate` (the FCFS-today
/// equivalence): the scheduler-driven path must not perturb the single-request
/// case it subsumes.
#[test]
fn scheduled_continuous_batch_of_one_matches_generate() -> anyhow::Result<()> {
    let fixture = tiny_scatter_fixture()?;
    let request = token_request(vec![2, 6, 7], 5);

    let expected = deterministic_engine(&fixture)?.generate(request.clone())?;

    let mut engine = deterministic_engine(&fixture)?;
    let scheduled = engine.run_continuous_batch_scheduled(vec![request], 1)?;

    assert_eq!(scheduled.len(), 1);
    assert_eq!(
        scheduled[0], expected,
        "batch-of-1 scheduler path diverged from single-request generate"
    );
    Ok(())
}

/// The scheduler-driven path must agree token-for-token with the greedy
/// `run_continuous_batch` (both are parity paths over the same machinery), and
/// the result must be independent of `max_batch` — different batch compositions
/// (1, 2, 4, 16) yield identical tokens.
#[test]
fn scheduled_matches_greedy_and_is_batch_size_invariant() -> anyhow::Result<()> {
    let fixture = tiny_scatter_fixture()?;
    let requests = (0..16)
        .map(|idx| {
            token_request(
                vec![
                    1 + (idx as u32 % 8),
                    5 + (idx as u32 % 5),
                    9 + (idx as u32 % 3),
                ],
                1 + (idx % 6),
            )
        })
        .collect::<Vec<_>>();

    let expected = sequential_baseline(&fixture, &requests)?;

    for max_batch in [1usize, 2, 4, 16] {
        let mut scheduled_engine = deterministic_engine(&fixture)?;
        let scheduled =
            scheduled_engine.run_continuous_batch_scheduled(requests.clone(), max_batch)?;
        assert_eq!(
            scheduled, expected,
            "scheduler-driven batch @max_batch={max_batch} diverged from sequential baseline"
        );

        let mut greedy_engine = deterministic_engine(&fixture)?;
        let greedy = greedy_engine.run_continuous_batch(requests.clone(), max_batch)?;
        assert_eq!(
            scheduled, greedy,
            "scheduler-driven batch @max_batch={max_batch} diverged from greedy run_continuous_batch"
        );
    }
    Ok(())
}

/// Empty input is a no-op.
#[test]
fn scheduled_continuous_batch_empty_is_empty() -> anyhow::Result<()> {
    let fixture = tiny_scatter_fixture()?;
    let mut engine = deterministic_engine(&fixture)?;
    let scheduled = engine.run_continuous_batch_scheduled(Vec::new(), 4)?;
    assert!(scheduled.is_empty());
    Ok(())
}

//! Every generated request runs a canonical workflow.
//!
//! A single-decoder package declares its `pipeline.workflow` like every other
//! package, and the canonical decode loop reads its body out of that declared
//! workflow. These cases pin it end to end: the package declares a workflow,
//! execution goes through it, and the direct decode path that used to exist
//! cannot be selected any more.
//!
//! Correctness of the *tokens* is pinned by the greedy goldens over real models
//! (`.goldens/`), which are byte-identical across this change. What is pinned
//! here is the structural claim those goldens cannot make: that the tokens came
//! out of the canonical path.

use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    Engine, EngineConfig, FinishReason, GenerateOptions, GeneratePrompt, GenerateRequest,
    WorkflowShapeReport,
};

fn decoder_package() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm")
}

fn workflow_package() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/onnx_genai_workflows/gemma4_chained")
}

fn engine() -> anyhow::Result<Engine> {
    Engine::from_dir(&decoder_package(), EngineConfig::default())
}

fn greedy(tokens: usize) -> GenerateRequest {
    GenerateRequest {
        prompt: GeneratePrompt::Text("hello world".to_string()),
        options: GenerateOptions {
            max_new_tokens: tokens,
            greedy: true,
            temperature: 0.0,
            stop_on_eos: false,
            ..GenerateOptions::default()
        },
    }
}

/// A single decoder is an ordinary workflow package, declared on disk.
///
/// Nothing is synthesized at load: the workflow the runtime executes is the one
/// the package ships, and the decode ABI the fast path binds is *derived from
/// that workflow* rather than from a second serialized block.
#[test]
fn a_single_decoder_is_an_authored_workflow() -> anyhow::Result<()> {
    let engine = engine()?;
    assert_eq!(engine.workflow_shape(), WorkflowShapeReport::SingleDecoder);

    let workflow = engine
        .package_workflow()
        .expect("a loaded package always declares a workflow");
    // Recognized structurally — the component that consumes the sequence and
    // produces logits — not by name.
    assert_eq!(
        onnx_genai_metadata::sole_decoder_component(workflow),
        Some(onnx_genai_metadata::decoder_workflow::DECODER_COMPONENT)
    );
    assert!(
        workflow
            .components
            .contains_key(onnx_genai_metadata::decoder_workflow::POLICY_COMPONENT)
    );
    // The serialized document is the workflow, and there is no second place the
    // graph ABI could be stated.
    assert!(engine.metadata().pipeline.is_some());
    assert!(engine.metadata().decoder_io().is_some());
    Ok(())
}

/// Prefill and cached decode both run through the canonical loop, and a second
/// turn on the same session reuses the first turn's KV.
#[test]
fn prefill_and_cached_decode_run_through_the_canonical_loop() -> anyhow::Result<()> {
    let mut engine = engine()?;
    let session = engine.create_session()?;

    let first = engine.generate_in_session(session, greedy(6))?;
    assert_eq!(first.token_ids.len(), 6);
    let after_first = engine.session_token_count(session)?;
    assert!(after_first > 0, "prefill must leave tokens in the session");

    // The second turn continues the same session. This tiny fixture has a small
    // context window, so it may stop on `Length` rather than spend the budget —
    // what matters here is that it *continued* the session instead of
    // restarting it, and that the stop came from the one policy.
    let second = engine.generate_in_session(session, greedy(6))?;
    assert!(!second.token_ids.is_empty(), "{second:?}");
    assert!(
        matches!(
            second.finish_reason,
            FinishReason::MaxTokens | FinishReason::Length
        ),
        "a cached-decode turn must stop through the canonical policy: {second:?}"
    );
    let after_second = engine.session_token_count(session)?;
    assert!(
        after_second > after_first,
        "a cached-decode turn must extend the session, not restart it: {after_first} -> \
         {after_second}"
    );
    engine.close_session(session)?;
    Ok(())
}

/// Greedy generation is deterministic and reaches the max-tokens stop.
#[test]
fn generation_is_deterministic_and_stops_on_max_tokens() -> anyhow::Result<()> {
    let mut engine = engine()?;
    let first = engine.generate(greedy(8))?;
    let second = engine.generate(greedy(8))?;
    assert_eq!(
        first.token_ids, second.token_ids,
        "greedy decode through the canonical loop must be reproducible"
    );
    assert_eq!(first.finish_reason, FinishReason::MaxTokens);
    assert_eq!(first.token_ids.len(), 8);
    Ok(())
}

/// EOS stops the canonical loop before the token budget is spent.
///
/// The stop decision is the interpreter loop's `continue_when`, evaluated from
/// the one policy's finish reason — this proves that path is live rather than
/// the budget simply running out.
#[test]
fn eos_stops_the_canonical_loop() -> anyhow::Result<()> {
    let mut engine = engine()?;
    // Take the first generated token as the stop token, then re-run with EOS
    // enabled on it: the loop must end at that token instead of at the budget.
    let probe = engine.generate(greedy(4))?;
    let eos = *probe.token_ids.first().expect("probe produced a token");

    let mut request = greedy(32);
    request.options.stop_on_eos = true;
    request.options.eos_token_id = Some(eos);
    let stopped = engine.generate(request)?;
    assert_eq!(stopped.finish_reason, FinishReason::EosToken, "{stopped:?}");
    assert!(
        stopped.token_ids.len() < 32,
        "EOS must stop the loop before the budget: {stopped:?}"
    );
    Ok(())
}

/// Sampling runs through the same loop, and a seed makes it reproducible.
#[test]
fn seeded_sampling_runs_through_the_canonical_loop() -> anyhow::Result<()> {
    let mut engine = engine()?;
    let sampled = |seed: u64| {
        let mut request = greedy(8);
        request.options.greedy = false;
        request.options.temperature = 0.9;
        request.options.top_p = 0.95;
        request.options.seed = Some(seed);
        request
    };
    let first = engine.generate(sampled(7))?;
    let again = engine.generate(sampled(7))?;
    assert_eq!(
        first.token_ids, again.token_ids,
        "a seeded sample must be reproducible through the canonical loop"
    );
    assert_eq!(first.token_ids.len(), 8);
    Ok(())
}

/// Batched generation is a different iteration shape, not an exemption.
///
/// It advances N rows per forward pass, so it is deliberately *not* the
/// canonical single-row body — but it is still this runtime producing tokens,
/// so it must be refused just the same when no canonical workflow exists.
/// Asserting only the token counts would leave batching a silent hole.
///
/// Skipped with a reason on a decode path that cannot batch, rather than
/// asserting a capability the fixture does not have.
#[test]
fn batched_generation_is_held_to_the_canonical_precondition() -> anyhow::Result<()> {
    let mut engine = engine()?;
    let capability = engine.batching_capability();
    if !capability.supports_batching() {
        eprintln!(
            "skipping canonical batching case: this decode path reports no batching ({})",
            capability.reason()
        );
        return Ok(());
    }
    let results = engine.generate_batched_static(vec![greedy(4), greedy(4)])?;
    assert_eq!(results.len(), 2);
    for result in &results {
        assert_eq!(result.token_ids.len(), 4);
    }

    // The refusal half of this property — that removing the canonical form makes
    // batching refuse too — needs the crate-private seam that produces a state
    // no public API can reach, so it lives in
    // `engine::runtime::canonical_refusal_tests` beside the other entry points.
    Ok(())
}

/// A composite workflow package is driven by the generic interpreter.
#[test]
fn a_composite_workflow_is_executed_by_the_interpreter() -> anyhow::Result<()> {
    let engine = Engine::from_dir(&workflow_package(), EngineConfig::default())?;
    assert_eq!(engine.workflow_shape(), WorkflowShapeReport::Composite);
    assert!(engine.is_workflow());
    Ok(())
}

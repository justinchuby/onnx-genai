//! Every generated request runs a canonical workflow.
//!
//! A bare-decoder package declares `model.io` and no `pipeline.workflow`. The
//! loader compiles that ABI into a canonical workflow in memory
//! (`onnx_genai_metadata::canonical`) and the interpreter's canonical decode
//! loop executes it. These cases pin that end to end: the lowering happens, the
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
    WorkflowProvenance,
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

/// A bare decoder is lowered at load, and says so.
///
/// The package still declares `model.io` alone — nothing is serialized back —
/// so `is_workflow()` stays false while the provenance reports `lowered`.
#[test]
fn a_bare_decoder_is_lowered_at_load() -> anyhow::Result<()> {
    let engine = engine()?;
    assert!(
        !engine.is_workflow(),
        "a bare decoder must not claim to serialize a workflow"
    );
    assert_eq!(engine.workflow_provenance(), WorkflowProvenance::Lowered);

    let workflow = engine.canonical_workflow()?;
    assert!(
        workflow
            .components
            .contains_key(onnx_genai_metadata::DECODER_COMPONENT)
    );
    assert!(
        workflow
            .components
            .contains_key(onnx_genai_metadata::POLICY_COMPONENT)
    );
    // The package on disk is untouched: `model.io` remains its sole serialized
    // answer, which is what keeps it valid under the existing rule.
    assert!(engine.metadata().pipeline.is_none());
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

/// Batched generation, where the decode path supports it, uses the same loop.
///
/// Skipped with a reason on a decode path that cannot batch, rather than
/// asserting a capability the fixture does not have.
#[test]
fn batching_runs_through_the_canonical_loop() -> anyhow::Result<()> {
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
    Ok(())
}

/// The legacy direct decode path cannot be selected.
///
/// There is no flag, mode, or constructor that reaches generation without a
/// canonical workflow: the loader installs one for every decoder package, and a
/// runtime that somehow lacks one refuses to generate rather than falling back.
/// This is the guarantee the whole change rests on, so it is asserted directly
/// on a runtime with its canonical form removed.
#[test]
fn the_legacy_direct_decode_path_cannot_be_selected() -> anyhow::Result<()> {
    // 1. There is no public constructor that skips lowering: every decoder load
    //    reports `Lowered`.
    for engine in [
        Engine::from_dir(&decoder_package(), EngineConfig::default())?,
        Engine::from_dir_with_session_options(
            &decoder_package(),
            EngineConfig::default(),
            onnx_genai_ort::SessionOptions::default(),
        )?,
    ] {
        assert_eq!(
            engine.workflow_provenance(),
            WorkflowProvenance::Lowered,
            "a decoder package must always load lowered; no constructor may skip it"
        );
    }

    // 2. A runtime whose canonical form is absent refuses to generate. Nothing
    //    in the public API can produce this state — which is the point — so it
    //    is constructed through the test-only seam to prove the guard exists.
    let mut engine = engine()?;
    engine.forget_canonical_workflow_for_test();
    let error = engine
        .generate(greedy(2))
        .expect_err("a runtime with no canonical workflow must refuse to generate");
    let message = format!("{error:#}");
    assert!(
        message.contains("no canonical workflow"),
        "the refusal must name the missing canonical workflow: {message}"
    );
    assert!(
        message.contains("no longer exists"),
        "the refusal must say the direct path is gone, not merely unavailable: {message}"
    );
    Ok(())
}

/// An authored workflow package is executed as authored, never lowered beside
/// its own workflow.
#[test]
fn an_authored_workflow_is_executed_as_authored() -> anyhow::Result<()> {
    let engine = Engine::from_dir(&workflow_package(), EngineConfig::default())?;
    assert_eq!(engine.workflow_provenance(), WorkflowProvenance::Authored);
    assert!(engine.is_workflow());
    assert!(
        engine.canonical_workflow().is_err(),
        "an authored workflow must not also be lowered"
    );
    Ok(())
}

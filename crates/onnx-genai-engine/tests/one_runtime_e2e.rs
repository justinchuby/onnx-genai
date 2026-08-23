//! One runtime, both package shapes.
//!
//! These cases pin the property that removed the caller-side split: a single
//! [`Engine`] type, loaded by a single constructor, serves a bare-decoder
//! package and a `pipeline.workflow` package alike. A server, CLI, C ABI, or
//! benchmark holds one handle and never asks which kind it got.
//!
//! What each case guards against is a *regression to two runtimes*: a second
//! public type, a second constructor that a caller has to choose, or an
//! operation that silently works on only one shape.

use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    Engine, EngineConfig, FinishReason, GenerateOptions, GeneratePrompt, GenerateRequest,
};

fn decoder_package() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm")
}

fn workflow_package() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/onnx_genai_workflows/gemma4_chained")
}

fn text_request(prompt: &str, tokens: usize) -> GenerateRequest {
    GenerateRequest {
        prompt: GeneratePrompt::Text(prompt.to_string()),
        options: GenerateOptions {
            max_new_tokens: tokens,
            greedy: true,
            temperature: 0.0,
            stop_on_eos: false,
            ..GenerateOptions::default()
        },
    }
}

/// The same constructor loads both package shapes and reports which one it got
/// from the package itself.
///
/// This is the load-time half of the unification: `from_dir` no longer needs a
/// caller to have run `PipelineModelDirectory::load_if_declared` first and pick
/// a constructor accordingly, which is exactly what the server, CLI, and
/// benchmarks each used to do separately (and could each get wrong).
#[test]
fn one_constructor_loads_both_package_shapes() -> anyhow::Result<()> {
    let decoder = Engine::from_dir(&decoder_package(), EngineConfig::default())?;
    assert!(
        !decoder.is_workflow(),
        "a bare-decoder package must not report itself as a workflow"
    );

    let workflow = Engine::from_dir(&workflow_package(), EngineConfig::default())?;
    assert!(
        workflow.is_workflow(),
        "a package declaring pipeline.workflow must report itself as one"
    );
    Ok(())
}

/// Autoregressive text generation is one entry point on one type.
///
/// A decoder package decodes through the decode core; the caller does not name
/// it, and the result is the same `GenerateResult` a workflow package produces.
#[test]
fn decoder_text_generation_runs_through_the_one_runtime() -> anyhow::Result<()> {
    let mut engine = Engine::from_dir(&decoder_package(), EngineConfig::default())?;
    let first = engine.generate(text_request("hello world", 6))?;
    assert_eq!(first.token_ids.len(), 6, "{first:?}");
    assert!(!engine.is_workflow());

    // Deterministic: the same request through the same entry point twice.
    let second = engine.generate(text_request("hello world", 6))?;
    assert_eq!(
        first.token_ids, second.token_ids,
        "greedy decode through the one runtime must be reproducible"
    );
    Ok(())
}

/// Session lifecycle is one API on one type.
///
/// Sessions belong to the decode core, so a workflow package answers with an
/// actionable error rather than a panic or a silent no-op — the property the
/// server's session route depends on now that it asks the runtime instead of a
/// `handle.pipeline` flag.
#[test]
fn session_lifecycle_is_one_api() -> anyhow::Result<()> {
    let mut engine = Engine::from_dir(&decoder_package(), EngineConfig::default())?;
    let session = engine.create_session()?;
    let first = engine.generate_in_session(session, text_request("hello world", 4))?;
    assert_eq!(first.token_ids.len(), 4);
    let count = engine.session_token_count(session)?;
    assert!(count > 0, "a session must accumulate tokens");
    engine.close_session(session)?;
    assert!(
        engine.session_token_count(session).is_err(),
        "a closed session must no longer resolve"
    );

    // A workflow package owns no engine sessions. It must say so, not silently
    // hand back a session that nothing backs.
    let mut workflow = Engine::from_dir(&workflow_package(), EngineConfig::default())?;
    let error = workflow
        .create_session()
        .expect_err("a workflow package must refuse to open an engine session");
    let message = format!("{error:#}");
    assert!(
        !message.is_empty(),
        "a workflow package must explain why it has no engine sessions"
    );
    Ok(())
}

/// Diagnostics that only one shape can answer degrade to an empty answer rather
/// than an error or a panic, so a caller can report uniformly.
#[test]
fn shape_specific_diagnostics_are_uniform() -> anyhow::Result<()> {
    let decoder = Engine::from_dir(&decoder_package(), EngineConfig::default())?;
    // A decoder package has no islands and no adapters; asking is legal.
    assert!(decoder.execution_island_diagnostics().is_empty());
    assert_eq!(
        decoder.adapter_lifecycle_diagnostic(),
        Default::default(),
        "a decoder package reports an empty adapter lifecycle, not an error"
    );
    // Both shapes answer the resource and provider questions.
    assert!(!decoder.execution_provider_status().is_empty());
    let _ = decoder.resource_snapshot();

    let workflow = Engine::from_dir(&workflow_package(), EngineConfig::default())?;
    assert!(!workflow.execution_provider_status().is_empty());
    let _ = workflow.resource_snapshot();
    // A workflow package can be asked for its component models; a decoder one
    // says so rather than pretending to have none.
    assert!(workflow.models().is_ok());
    assert!(decoder.models().is_err());
    Ok(())
}

/// Tokenization is served from whichever half owns the tokenizer.
#[test]
fn tokenization_is_one_api_across_shapes() -> anyhow::Result<()> {
    let decoder = Engine::from_dir(&decoder_package(), EngineConfig::default())?;
    let decoder_tokens = decoder.tokenize("hello world")?;
    assert!(!decoder_tokens.is_empty());

    let workflow = Engine::from_dir(&workflow_package(), EngineConfig::default())?;
    let workflow_tokens = workflow.tokenize("hello world")?;
    assert!(!workflow_tokens.is_empty());
    Ok(())
}

/// A composite workflow reports why it actually stopped.
///
/// Before this, every composite generation returned `MaxTokens` — a constant,
/// not an observation. A workflow that ended at its own EOS told the caller it
/// had run out of budget, which is the single distinction a finish reason
/// exists to make: a client deciding whether to continue a truncated response
/// would have continued one the model considered complete.
///
/// The reason now comes from the loop, which is the only thing that knows: the
/// liveness predicate going false is the model's own stop, exhausting the bound
/// is the caller's.
///
/// # What this asserts, and what it does not
///
/// The bound case is asserted here. The predicate case is **not** asserted
/// end-to-end, because no committed composite fixture both executes through
/// `generate` and stops on its own liveness predicate: the authored `decoder`
/// fixture always runs to its bound, `gemma4_chained` is driven through the
/// proposal API, and `tiny-llm-scatter-workflow` declares graph ports its ONNX
/// model does not expose. Asserting it against a fixture that cannot reach the
/// state would be a test that passes for the wrong reason, so the gap is
/// recorded rather than papered over. The mechanism is covered at unit level by
/// `loop_ended_by_predicate`.
#[test]
fn a_composite_workflow_reports_the_reason_it_stopped() -> anyhow::Result<()> {
    let package = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/onnx_genai_workflows/decoder");
    let mut engine = Engine::from_dir(&package, EngineConfig::default())?;

    // This package ships no tokenizer adapter, so the prompt is token ids.
    let tokens = |limit: usize| GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![1]),
        options: GenerateOptions {
            max_new_tokens: limit,
            greedy: true,
            temperature: 0.0,
            stop_on_eos: false,
            ..GenerateOptions::default()
        },
    };

    let bounded = engine.generate(tokens(2))?;
    assert_eq!(
        bounded.finish_reason,
        FinishReason::MaxTokens,
        "a run that exhausts its bound must say so: {bounded:?}"
    );
    assert_eq!(bounded.token_ids.len(), 2, "{bounded:?}");
    Ok(())
}

//! Which loop algorithm runs is chosen by what the workflow authors.
//!
//! This is the property the convergence exists to establish, stated as a
//! measurement rather than a claim. Three packages declare three different loop
//! bodies. All three are walked by the same interpreter, and each one's
//! *executor* is selected by the contract its components declare — not by Rust
//! inspecting the package, the file layout, or which constructor the caller
//! reached for.
//!
//! # Why counters and not "the tokens came out"
//!
//! Token counts prove a loop ran; they do not prove *which* loop, and they
//! would keep passing if a `match` on package shape quietly came back. The
//! counters here are recorded inside the interpreter's node dispatch, so a body
//! that never named a contract cannot show an execution of it, and a body that
//! named one cannot have been served by anything else.
//!
//! The negative half matters as much as the positive: a package whose sampler
//! and termination predicate are ONNX components must show *zero* contract
//! executions. If those ever became non-zero, the runtime would be supplying a
//! step the package declared a graph for — the exact substitution that would
//! make the declared document decorative.

use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest, PipelineGenerateRequest,
};
use onnx_genai_ort::Value;

const DECODE_CONTRACT: &str = "onnx-genai.autoregressive-decode";
const POLICY_CONTRACT: &str = "onnx-genai.token-policy";

fn fixture(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
}

fn options(tokens: usize) -> GenerateOptions {
    GenerateOptions {
        max_new_tokens: tokens,
        greedy: true,
        temperature: 0.0,
        stop_on_eos: false,
        ..GenerateOptions::default()
    }
}

fn greedy(tokens: usize) -> GenerateRequest {
    GenerateRequest {
        prompt: GeneratePrompt::Text("hello world".to_string()),
        options: options(tokens),
    }
}

/// The one entry point, for a request that binds whatever a package declares.
///
/// Every case below goes through this: a prompt-only request for a package that
/// takes one, and the declared tensors for a package whose inputs are tensors.
/// What varies is the *request*, never the method.
fn drive(engine: &mut Engine, request: PipelineGenerateRequest) -> anyhow::Result<usize> {
    Ok(engine
        .generate_with_pipeline_request(request)?
        .token_ids
        .len())
}

/// A body naming the runtime's decode and policy contracts routes every node
/// of each to the executor registered for it.
///
/// One decode node and one policy node per generated token, because that is
/// what the body declares — not because a Rust loop was written to call them in
/// that order.
#[test]
fn a_single_token_body_selects_the_decode_core() -> anyhow::Result<()> {
    let mut engine =
        Engine::from_dir(&fixture("tests/fixtures/tiny-llm"), EngineConfig::default())?;
    let tokens = drive(&mut engine, PipelineGenerateRequest::new(greedy(5)))?;
    assert_eq!(tokens, 5);

    let executions = engine.contract_executions();
    assert_eq!(
        executions.get(DECODE_CONTRACT).copied(),
        Some(5),
        "one declared decode node per token: {executions:?}"
    );
    assert_eq!(
        executions.get(POLICY_CONTRACT).copied(),
        Some(5),
        "one declared token-policy node per token: {executions:?}"
    );
    assert_eq!(
        executions.len(),
        2,
        "this body names two contracts and nothing else ran: {executions:?}"
    );
    Ok(())
}

/// A body whose sampler and termination predicate are ONNX components runs
/// them, and asks the runtime for nothing.
///
/// The same interpreter, the same loop machinery, the same emits. What differs
/// is the body — so the executors differ, with no Rust branch involved.
#[test]
fn an_in_graph_body_selects_no_runtime_executor() -> anyhow::Result<()> {
    let mut engine = Engine::from_dir(
        &fixture("tests/fixtures/onnx_genai_workflows/decoder"),
        EngineConfig::default(),
    )?;
    let tokens = drive(
        &mut engine,
        PipelineGenerateRequest::new(GenerateRequest {
            prompt: GeneratePrompt::TokenIds(vec![4, 5]),
            options: options(4),
        }),
    )?;
    assert!(tokens > 0);

    assert!(
        engine.contract_executions().is_empty(),
        "a package that declares a graph for every step must not have the runtime supply one: \
         {:?}",
        engine.contract_executions()
    );
    // An empty contract count on its own is satisfied by a *second* drive that
    // counts nothing, so it is not evidence by itself. The component
    // invocations are: they are recorded by the interpreter's own `Invoke`
    // dispatch, so a non-empty set proves this package's tokens came out of
    // `run_workflow_node` and not from somewhere else.
    // Every dynamic leading-axis component remains visible to admission instead
    // of being hidden in an execution island.
    let components = engine.component_invocations();
    for component in ["model", "token_sampler", "termination"] {
        assert!(
            components.get(component).copied().unwrap_or(0) > 0,
            "the declared component '{component}' must have been invoked: {components:?}"
        );
    }
    assert!(
        engine.execution_island_diagnostics().is_empty(),
        "request-batched policy components must retain their admission boundaries"
    );
    Ok(())
}

/// A body that authors a proposal block runs the components it names.
///
/// The speculative algorithm is a different *shape of iteration* — a block
/// proposed and verified rather than a token selected — and it is reached the
/// same way as the other two: the workflow names the components, and the
/// interpreter invokes them. Nothing asked whether this package "is
/// speculative".
#[test]
fn a_speculative_body_selects_its_declared_components() -> anyhow::Result<()> {
    let package = fixture("tests/fixtures/onnx_genai_workflows/speculative");
    let mut engine = Engine::from_dir(&package, EngineConfig::default())?;
    let workflow = engine
        .package_workflow()
        .expect("a loaded package declares a workflow");
    assert!(
        workflow.components.contains_key("proposer")
            && workflow.components.contains_key("verifier"),
        "the speculative fixture authors a proposer and a verifier"
    );

    let tokens = drive(&mut engine, speculative_request(1)?)?;
    assert!(tokens > 0);
    assert!(
        engine.contract_executions().is_empty(),
        "this body names no runtime contract, so nothing may have been supplied: {:?}",
        engine.contract_executions()
    );
    // As above: the positive half is what rules out a second drive. An empty
    // contract count would also be produced by a loop that never reached the
    // interpreter at all.
    // The proposer and verifier both have dynamic request axes, so they remain
    // individual component stages where admission can validate them.
    let components = engine.component_invocations();
    assert!(
        components.keys().any(|name| name.starts_with("grammar_")),
        "the declared grammar adapters must have been invoked: {components:?}"
    );
    for component in ["proposer", "verifier"] {
        assert!(
            components.get(component).copied().unwrap_or(0) > 0,
            "the declared component '{component}' must have been invoked: {components:?}"
        );
    }
    assert!(
        engine.execution_island_diagnostics().is_empty(),
        "request-batched proposer and verifier must retain their admission boundaries"
    );
    Ok(())
}

/// The three bodies select three different executor sets, through one entry
/// point.
///
/// Stated together because the property is comparative: it is not that each
/// package works, but that the *same* call on the *same* type produced
/// different executors purely from what each package declared.
#[test]
fn three_bodies_select_three_executor_sets_through_one_entry_point() -> anyhow::Result<()> {
    let mut observed = Vec::new();
    for (relative, request) in [
        (
            "tests/fixtures/tiny-llm",
            PipelineGenerateRequest::new(greedy(3)),
        ),
        (
            "tests/fixtures/onnx_genai_workflows/decoder",
            PipelineGenerateRequest::new(GenerateRequest {
                prompt: GeneratePrompt::TokenIds(vec![4, 5]),
                options: options(3),
            }),
        ),
        (
            "tests/fixtures/onnx_genai_workflows/speculative",
            speculative_request(1)?,
        ),
    ] {
        let mut engine = Engine::from_dir(&fixture(relative), EngineConfig::default())?;
        // One method and one constructor, with no argument naming a package
        // kind and no caller-side choice of entry point.
        let _ = drive(&mut engine, request)?;
        let contracts = engine
            .contract_executions()
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        let components = engine
            .component_invocations()
            .into_iter()
            .filter(|(_, runs)| *runs > 0)
            .map(|(name, _)| name)
            .collect::<Vec<_>>();
        observed.push((relative, contracts, components));
    }

    let [single, in_graph, chained] = observed.as_slice() else {
        panic!("three packages were driven");
    };
    assert_eq!(
        single.1,
        vec![DECODE_CONTRACT.to_string(), POLICY_CONTRACT.to_string()],
        "the single-token body selects the decode core"
    );
    assert!(
        !single.2.contains(&"decoder".to_string()),
        "the decode core executes the declared decoder node; the interpreter must not have \
         invoked its graph as an ordinary component too: {:?}",
        single.2
    );
    // Each negative case pairs its empty contract set with a *non-empty* set of
    // interpreter-recorded component invocations. That pairing is what makes it
    // evidence: a second generation drive would leave both empty.
    assert!(
        in_graph.1.is_empty() && !in_graph.2.is_empty(),
        "the in-graph body ran no runtime contract and did run its own components: {in_graph:?}"
    );
    assert!(
        chained.1.is_empty() && !chained.2.is_empty(),
        "the speculative body ran no runtime contract and did run its own components: {chained:?}"
    );
    assert_ne!(
        in_graph.2, chained.2,
        "two authored bodies naming different components must run different components"
    );
    Ok(())
}

/// The speculative fixture's declared request inputs.
///
/// It states its borrowed verifier cache, grammar tables and proposal-budget
/// telemetry as request inputs rather than package literals, so a caller
/// supplies them. That is a property of the package's declaration, not of a
/// "pipeline mode": the same entry point binds whatever a package declares.
fn speculative_request(max_new_tokens: usize) -> anyhow::Result<PipelineGenerateRequest> {
    Ok(PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![1, 2, 3, 4]),
        options: options(max_new_tokens),
    })
    .with_input(
        "verifier.past_key_values.0.key",
        Value::from_slice_f32(&[], &[1, 2, 0, 8])?,
    )
    .with_input(
        "verifier.past_key_values.0.value",
        Value::from_slice_f32(&[], &[1, 4, 0, 4])?,
    )
    .with_input("grammar.initial_state", Value::from_slice_i64(&[0], &[1])?)
    .with_input(
        "grammar.transition_table",
        Value::from_slice_i64(&[0; 32], &[1, 32])?,
    )
    .with_input("adaptive.current_k", Value::from_slice_i64(&[4], &[1])?)
    .with_input(
        "adaptive.estimates",
        Value::from_slice_f32(&[0.0; 24], &[1, 24])?,
    )
    .with_input("telemetry.draft_ms", Value::from_slice_f32(&[1.0], &[1])?)
    .with_input("telemetry.target_ms", Value::from_slice_f32(&[1.0], &[1])?))
}

//! Which *iteration algorithm* runs is chosen by what the workflow authors.
//!
//! `authored_body_selects_executor.rs` established the property for one
//! algorithm: a body naming `autoregressive-decode` + `token-policy` selects the
//! decode core, and a body naming nothing selects nothing. This file states the
//! comparative half — three bodies, three *different* registered executors,
//! reached through the runtime's own node dispatch.
//!
//! # Why counters, again
//!
//! A batched generation that produced the right tokens would keep passing if
//! Rust quietly went back to picking the row-major loop from
//! `ModelDecodePath::StaticCache`. The per-contract counters are recorded inside
//! the interpreter's `Invoke` dispatch, so a count against
//! `onnx-genai.continuous-batch` can only exist because a declared node named
//! that contract — which is the thing a shape `match` cannot produce.
//!
//! The bodies are authored by the metadata crate from the package's *own*
//! declared loop, so the three cases below differ in exactly one respect: which
//! contract the loop body's node declares.

use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest, PipelineGenerateRequest,
};

const DECODE_CONTRACT: &str = "onnx-genai.autoregressive-decode";
const POLICY_CONTRACT: &str = "onnx-genai.token-policy";
const CONTINUOUS_BATCH_CONTRACT: &str = "onnx-genai.continuous-batch";
const SPECULATIVE_BLOCK_CONTRACT: &str = "onnx-genai.speculative-block";

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

fn batching_engine() -> anyhow::Result<Option<Engine>> {
    let engine = Engine::from_dir(
        &fixture("tests/fixtures/tiny-llm-batched"),
        EngineConfig::default(),
    )?;
    if !engine.batching_capability().supports_batching() {
        eprintln!(
            "skipping: this decode path reports no batching ({})",
            engine.batching_capability().reason()
        );
        return Ok(None);
    }
    Ok(Some(engine))
}

/// A single-token body routes each node to the decode core, as before.
#[test]
fn a_single_token_body_selects_the_decode_core() -> anyhow::Result<()> {
    let mut engine =
        Engine::from_dir(&fixture("tests/fixtures/tiny-llm"), EngineConfig::default())?;
    engine.generate_with_pipeline_request(PipelineGenerateRequest::new(greedy(4)))?;
    let executed = engine.contract_executions();
    assert_eq!(executed.get(DECODE_CONTRACT).copied(), Some(4));
    assert_eq!(executed.get(POLICY_CONTRACT).copied(), Some(4));
    assert_eq!(
        executed.get(CONTINUOUS_BATCH_CONTRACT),
        None,
        "a single-token body must not have run a row-major step: {executed:?}"
    );
    Ok(())
}

/// A row-scoped batch body routes its iteration to the continuous-batch
/// executor, and to nothing else.
///
/// The zero on the single-token contracts is the load-bearing half: it is what
/// says the batch really did run a *different* declared step, rather than the
/// same one N times behind a Rust loop.
#[test]
fn a_row_scoped_batch_body_selects_the_batch_executor() -> anyhow::Result<()> {
    let Some(mut engine) = batching_engine()? else {
        return Ok(());
    };
    let results = engine.run_continuous_batch(vec![greedy(3), greedy(3)], 2)?;
    assert_eq!(results.len(), 2);
    for result in &results {
        assert_eq!(result.token_ids.len(), 3);
    }

    let executed = engine.contract_executions();
    assert!(
        executed
            .get(CONTINUOUS_BATCH_CONTRACT)
            .copied()
            .unwrap_or(0)
            > 0,
        "the authored batch body's node must have reached the batch executor: {executed:?}"
    );
    assert_eq!(
        executed.get(DECODE_CONTRACT),
        None,
        "a row-major body declares no single-row decode step: {executed:?}"
    );
    assert_eq!(
        executed.get(POLICY_CONTRACT),
        None,
        "a row-major body declares no separate token policy: {executed:?}"
    );
    Ok(())
}

/// Static batching reaches the same declared contract as continuous batching.
///
/// Two entry points, one authored iteration: a fixed batch and a continuously
/// backfilled one are the same algorithm with different admission, so a body
/// that named a third contract for one of them would be describing a difference
/// that does not exist.
#[test]
fn static_batching_selects_the_same_authored_contract() -> anyhow::Result<()> {
    let Some(mut engine) = batching_engine()? else {
        return Ok(());
    };
    let results = engine.generate_batched_static(vec![greedy(3), greedy(3)])?;
    assert_eq!(results.len(), 2);
    let executed = engine.contract_executions();
    assert!(
        executed
            .get(CONTINUOUS_BATCH_CONTRACT)
            .copied()
            .unwrap_or(0)
            > 0,
        "static batching runs the authored row-major body too: {executed:?}"
    );
    assert_eq!(executed.get(DECODE_CONTRACT), None);
    Ok(())
}

/// A speculative request selects the block executor, not the single-token one.
#[test]
fn a_speculative_body_selects_the_block_executor() -> anyhow::Result<()> {
    let mut engine =
        Engine::from_dir(&fixture("tests/fixtures/tiny-llm"), EngineConfig::default())?;
    let mut request = greedy(6);
    request.options.speculative_mode = Some(onnx_genai_engine::SpeculativeMode::PromptLookup {
        ngram: 2,
        max_tokens: 3,
    });
    engine.generate(request)?;

    let executed = engine.contract_executions();
    assert!(
        executed
            .get(SPECULATIVE_BLOCK_CONTRACT)
            .copied()
            .unwrap_or(0)
            > 0,
        "the authored speculative body's node must have reached the block executor: {executed:?}"
    );
    assert_eq!(
        executed.get(DECODE_CONTRACT),
        None,
        "a block body declares no single-row decode step: {executed:?}"
    );
    Ok(())
}

/// The three bodies select three different executors through one runtime.
///
/// Stated together because the property is comparative: the same package, the
/// same interpreter and the same node dispatch produced three disjoint executor
/// sets purely from which contract each authored body named.
#[test]
fn three_iteration_bodies_select_three_executors() -> anyhow::Result<()> {
    let Some(mut batch) = batching_engine()? else {
        return Ok(());
    };
    batch.run_continuous_batch(vec![greedy(3), greedy(3)], 2)?;
    let batch_contracts = contracts(&batch);

    let mut single =
        Engine::from_dir(&fixture("tests/fixtures/tiny-llm"), EngineConfig::default())?;
    single.generate_with_pipeline_request(PipelineGenerateRequest::new(greedy(3)))?;
    let single_contracts = contracts(&single);

    let mut speculative =
        Engine::from_dir(&fixture("tests/fixtures/tiny-llm"), EngineConfig::default())?;
    let mut request = greedy(6);
    request.options.speculative_mode = Some(onnx_genai_engine::SpeculativeMode::PromptLookup {
        ngram: 2,
        max_tokens: 3,
    });
    speculative.generate(request)?;
    let block_contracts = contracts(&speculative);

    assert_eq!(
        single_contracts,
        vec![DECODE_CONTRACT.to_string(), POLICY_CONTRACT.to_string()]
    );
    assert_eq!(
        batch_contracts,
        vec![CONTINUOUS_BATCH_CONTRACT.to_string()],
        "the row-major body names one contract and nothing else ran"
    );
    assert_eq!(
        block_contracts,
        vec![SPECULATIVE_BLOCK_CONTRACT.to_string()],
        "the block body names one contract and nothing else ran"
    );
    for (left, right) in [
        (&single_contracts, &batch_contracts),
        (&single_contracts, &block_contracts),
        (&batch_contracts, &block_contracts),
    ] {
        assert!(
            left.iter().all(|contract| !right.contains(contract)),
            "three authored bodies must select three disjoint executor sets: {left:?} vs {right:?}"
        );
    }
    Ok(())
}

fn contracts(engine: &Engine) -> Vec<String> {
    engine
        .contract_executions()
        .into_iter()
        .filter(|(_, runs)| *runs > 0)
        .map(|(contract, _)| contract)
        .collect()
}

//! End-to-end behaviour of the two authored iteration algorithms.
//!
//! `authored_iteration_executors.rs` proves *which executor* each authored body
//! selects. This file proves the executors still do the job: a continuous batch
//! stops on a declared end token that is not the first one listed, and a
//! speculative block really proposes several tokens at once and rolls the
//! rejected suffix back rather than emitting it.
//!
//! Both are stated against a package's *observable* output, so they would fail
//! for a re-authoring that changed what a generation produces — which is the
//! risk of moving an iteration into a declared document.

use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    Engine, EngineConfig, FinishReason, GenerateOptions, GeneratePrompt, GenerateRequest,
    SpeculativeMode,
};

fn fixture(relative: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(relative)
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

/// A batched row stops at *any* declared end token, not only the first.
///
/// The stop id is declared **second**, behind an id this model never reaches.
/// A row-major executor that kept only the head of the list would run every row
/// to its budget, and the token counts below are what tell the two apart.
///
/// Reading the stop id out of an unstopped generation first is what makes this
/// independent of the fixture's weights: whatever the model emits second is the
/// token the batch is then told to end on.
#[test]
fn a_continuous_batch_row_stops_on_a_non_first_declared_end_token() -> anyhow::Result<()> {
    let mut engine = Engine::from_dir(
        &fixture("tests/fixtures/tiny-llm-batched"),
        EngineConfig::default(),
    )?;
    if !engine.batching_capability().supports_batching() {
        eprintln!(
            "skipping: this decode path reports no batching ({})",
            engine.batching_capability().reason()
        );
        return Ok(());
    }

    let baseline = engine.generate(greedy(4))?;
    assert_eq!(baseline.token_ids.len(), 4);
    let stop = baseline.token_ids[1];
    let unreachable = baseline
        .token_ids
        .iter()
        .copied()
        .max()
        .map_or(31, |token| token.max(31) + 1);

    let stopping = || {
        let mut request = greedy(8);
        request.options.stop_on_eos = true;
        request.options.eos_token_ids = vec![unreachable, stop];
        request
    };
    let results = engine.run_continuous_batch(vec![stopping(), stopping()], 2)?;

    assert_eq!(results.len(), 2);
    for result in &results {
        assert_eq!(
            result.token_ids,
            baseline.token_ids[..2].to_vec(),
            "the row must end at the second declared end token, not at its budget"
        );
        assert_eq!(result.finish_reason, FinishReason::EosToken);
    }
    // The stop came out of the authored row-major body, not a second loop.
    assert!(
        engine
            .contract_executions()
            .get("onnx-genai.continuous-batch")
            .copied()
            .unwrap_or(0)
            > 0
    );
    Ok(())
}

/// Rows that stop at different steps still finish independently.
///
/// The interpreter's loop carries per-row liveness, so a row that stops early
/// must keep its own tokens while its neighbour keeps decoding. Getting this
/// wrong in a row-major body shows up as the short row's stream leaking into the
/// long row's, which the exact comparison below catches.
#[test]
fn rows_with_different_budgets_finish_independently() -> anyhow::Result<()> {
    let mut engine = Engine::from_dir(
        &fixture("tests/fixtures/tiny-llm-batched"),
        EngineConfig::default(),
    )?;
    if !engine.batching_capability().supports_batching() {
        eprintln!(
            "skipping: this decode path reports no batching ({})",
            engine.batching_capability().reason()
        );
        return Ok(());
    }
    let baseline = engine.generate(greedy(5))?;
    let results = engine.run_continuous_batch(vec![greedy(2), greedy(5)], 2)?;
    assert_eq!(results[0].token_ids, baseline.token_ids[..2].to_vec());
    assert_eq!(results[1].token_ids, baseline.token_ids[..5].to_vec());
    Ok(())
}

/// A prompt whose repeated n-grams give prompt lookup something to propose.
///
/// Model-free proposal only fires when the context repeats, so a prompt with no
/// repetition would leave the block width at one and make the assertions below
/// vacuous rather than false.
fn repetitive(tokens: usize) -> GenerateRequest {
    let mut request = greedy(tokens);
    request.prompt = GeneratePrompt::TokenIds(vec![1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2]);
    request
}

/// A speculative block proposes a width and rolls back what the target rejects.
///
/// Three things must hold together, and only together do they describe a block:
///
/// * a **wider declared draft width proposes more tokens** on the same prompt,
///   so a block really carries a variable number of candidates rather than one;
/// * fewer tokens were *accepted* than proposed, so a rejected suffix was rolled
///   back out of the target's state rather than emitted; and
/// * the emitted stream is byte-identical to plain greedy decoding, so the
///   rollback removed exactly the rejected tokens and nothing else.
#[test]
fn a_speculative_block_proposes_a_width_and_rolls_back_rejections() -> anyhow::Result<()> {
    let plain = {
        let mut engine =
            Engine::from_dir(&fixture("tests/fixtures/tiny-llm"), EngineConfig::default())?;
        engine.generate(repetitive(12))?
    };

    let mut proposed = Vec::new();
    for width in [2usize, 4] {
        // A fresh runtime per width, so the block counter below describes this
        // generation rather than every generation the engine has ever run.
        let mut engine =
            Engine::from_dir(&fixture("tests/fixtures/tiny-llm"), EngineConfig::default())?;
        let mut request = repetitive(12);
        request.options.speculative_mode = Some(SpeculativeMode::PromptLookup {
            ngram: 2,
            max_tokens: width,
        });
        let speculative = engine.generate(request)?;
        let stats = engine.last_speculative_stats();

        assert_eq!(
            speculative.token_ids, plain.token_ids,
            "draft width {width} must reproduce greedy decoding exactly"
        );
        assert!(
            stats.accepted_tokens < stats.proposed_tokens,
            "this prompt produces rejections, which is what exercises rollback: {stats:?}"
        );
        let blocks = engine
            .contract_executions()
            .get("onnx-genai.speculative-block")
            .copied()
            .unwrap_or(0);
        assert!(
            blocks > 0,
            "the authored block body must have run for width {width}"
        );
        // One block per iteration, and a block commits at least one token, so a
        // body that had silently gone back to one token per iteration would run
        // no fewer blocks than it produced tokens. The `+ 1` is the iteration
        // that reads the stop and commits nothing.
        assert!(
            (blocks as usize) <= speculative.token_ids.len() + 1,
            "{blocks} block iterations for {} tokens",
            speculative.token_ids.len()
        );
        proposed.push(stats.proposed_tokens);
    }

    assert!(
        proposed[1] > proposed[0],
        "a wider declared draft width must propose more candidates per block: {proposed:?}"
    );
    Ok(())
}

/// Speculation off and speculation on are the same generation.
///
/// Stated separately from the block-shape assertions because it is the property
/// a caller has: turning the optimization on is not allowed to change what the
/// package produces, whichever iteration body the runtime authored for it.
#[test]
fn speculation_does_not_change_what_the_package_produces() -> anyhow::Result<()> {
    let mut engine =
        Engine::from_dir(&fixture("tests/fixtures/tiny-llm"), EngineConfig::default())?;
    for width in [1usize, 2, 3, 5] {
        let plain = engine.generate(greedy(9))?;
        let mut request = greedy(9);
        request.options.speculative_mode = Some(SpeculativeMode::PromptLookup {
            ngram: 2,
            max_tokens: width,
        });
        let speculative = engine.generate(request)?;
        assert_eq!(
            speculative.token_ids, plain.token_ids,
            "draft width {width} changed the generated stream"
        );
    }
    Ok(())
}

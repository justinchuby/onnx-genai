//! Reuse across multimodal generations: encoder memoization and decoder KV
//! prefix reuse, over the `tiny-gemma4-vlm` fixture whose generated tokens
//! genuinely depend on the image (`vision_encoder -> embedding fusion ->
//! decoder`).
//!
//! Every test here asserts two things together: that work was skipped, **and**
//! that skipping it produced exactly what recomputing produces. A cache that is
//! fast and wrong is worse than no cache, and for multimodal prompts the wrong
//! answer is unusually plausible — placeholder expansion makes two different
//! photographs produce byte-identical token sequences, so a token-keyed cache
//! would answer fluently about a picture the model was never shown.

use std::path::{Path, PathBuf};

use onnx_genai_engine::pipeline::{PipelineEngine, PipelineGenerateRequest};
use onnx_genai_engine::{Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest};
use onnx_genai_ort::Value;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-gemma4-vlm")
}

fn load() -> anyhow::Result<PipelineEngine> {
    Engine::from_pipeline_dir(&fixture_dir(), EngineConfig::default())
}

/// `pixel_values[1,3,2,2]`, scaled so each `bias` yields a distinct image.
fn pixels(bias: f32) -> anyhow::Result<Value> {
    Value::from_vec_f32(
        (0..12).map(|i| i as f32 / 12.0 + bias).collect(),
        &[1, 3, 2, 2],
    )
    .map_err(Into::into)
}

/// Token 7 is the fixture's image placeholder, where fusion scatters features.
fn turn(
    engine: &mut PipelineEngine,
    prompt: Vec<u32>,
    bias: f32,
    max_new_tokens: usize,
) -> anyhow::Result<Vec<u32>> {
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(prompt));
    request.options = GenerateOptions {
        max_new_tokens,
        temperature: 0.0,
        stop_on_eos: false,
        ..GenerateOptions::default()
    };
    let request = PipelineGenerateRequest::new(request)
        .with_input("vision_encoder.pixel_values", pixels(bias)?);
    Ok(engine.generate_with_pipeline_request(request)?.token_ids)
}

#[test]
fn re_asking_about_the_same_image_does_not_re_run_the_vision_encoder() -> anyhow::Result<()> {
    let mut engine = load()?;

    let first = turn(&mut engine, vec![3, 7], 0.0, 4)?;
    let after_first = engine.cache_stats();
    assert_eq!(
        after_first.encoder_hits, 0,
        "nothing is memoized before the first turn runs"
    );
    assert!(after_first.encoder_misses > 0, "the encoder had to run");

    let second = turn(&mut engine, vec![3, 7], 0.0, 4)?;
    let after_second = engine.cache_stats();

    assert!(
        after_second.encoder_hits > after_first.encoder_hits,
        "the same pixels must be served from the memoized encoder output"
    );
    assert_eq!(
        first, second,
        "reuse must not change what the model generates"
    );
    Ok(())
}

#[test]
fn a_different_image_is_recomputed_and_changes_the_answer() -> anyhow::Result<()> {
    let mut engine = load()?;

    let first = turn(&mut engine, vec![3, 7], 0.0, 4)?;
    let hits_before = engine.cache_stats().encoder_hits;
    let second = turn(&mut engine, vec![3, 7], -4.0, 4)?;

    assert_eq!(
        engine.cache_stats().encoder_hits,
        hits_before,
        "different pixels must not hit the memoized encoder output"
    );
    assert_ne!(
        first, second,
        "the fixture's output depends on the image, so a new image must change it"
    );
    Ok(())
}

#[test]
fn a_new_image_invalidates_retained_kv_even_when_the_tokens_still_extend() -> anyhow::Result<()> {
    // The correctness trap this whole design exists for. The follow-up prompt
    // is a genuine extension of the retained context — the case that normally
    // reuses everything — but the attachment changed. Keyed on tokens alone, a
    // prefix cache would hand the first image's KV to the second image's turn
    // and the model would answer fluently about a picture it was never shown.
    let mut engine = load()?;

    let generated = turn(&mut engine, vec![3, 7], 0.0, 4)?;
    let mut follow_up = vec![3, 7];
    follow_up.extend_from_slice(&generated);
    follow_up.push(2);

    engine.reset_cache_stats();
    let warm = turn(&mut engine, follow_up.clone(), -4.0, 3)?;
    assert_eq!(
        engine.cache_stats().prefix_reused_tokens,
        0,
        "a new image must invalidate the retained KV even though the tokens extend it"
    );

    // The two images genuinely disagree about the first token (bias -4.0 vs
    // 0.0 flips it), so contamination would be visible here.
    let mut fresh = load()?;
    assert_eq!(
        warm,
        turn(&mut fresh, follow_up, -4.0, 3)?,
        "the second image's answer must not be contaminated by the first"
    );
    Ok(())
}

#[test]
fn a_follow_up_turn_prefills_only_the_tokens_it_added() -> anyhow::Result<()> {
    let mut engine = load()?;

    // Turn one leaves KV for prompt [3, 7] plus everything it generated.
    let generated = turn(&mut engine, vec![3, 7], 0.0, 4)?;
    let mut follow_up = vec![3, 7];
    follow_up.extend_from_slice(&generated);
    follow_up.push(2);

    engine.reset_cache_stats();
    let warm = turn(&mut engine, follow_up.clone(), 0.0, 3)?;
    let stats = engine.cache_stats();

    // The previous turn ran the decoder over its prompt plus all but the last
    // generated token: the final sampled token was committed to the context but
    // never fed back in, so no KV exists for it. Reuse must stop exactly there —
    // claiming one token more silently corrupts the next turn's attention.
    assert_eq!(
        stats.prefix_reused_tokens as usize,
        follow_up.len() - 2,
        "reuse must cover exactly the tokens the decoder actually ran"
    );
    assert_eq!(
        stats.prefill_tokens, 2,
        "the last generated token has no KV yet, so it is prefilled with the new one"
    );

    // Same continuation on an engine with nothing retained.
    let mut fresh = load()?;
    let cold = turn(&mut fresh, follow_up, 0.0, 3)?;
    assert_eq!(
        warm, cold,
        "carrying KV over must produce exactly what recomputing produces"
    );
    Ok(())
}

#[test]
fn a_prompt_sharing_no_leading_token_is_recomputed() -> anyhow::Result<()> {
    let mut engine = load()?;

    turn(&mut engine, vec![3, 7], 0.0, 4)?;
    engine.reset_cache_stats();
    // Differs from the very first token.
    let warm = turn(&mut engine, vec![5, 7, 1], 0.0, 3)?;
    assert_eq!(
        engine.cache_stats().prefix_reused_tokens,
        0,
        "there is no shared head to keep"
    );

    let mut fresh = load()?;
    assert_eq!(warm, turn(&mut fresh, vec![5, 7, 1], 0.0, 3)?);
    Ok(())
}

#[test]
fn a_forked_conversation_reuses_the_head_it_still_shares() -> anyhow::Result<()> {
    // Branching from an earlier turn — or replaying a reasoning model's history
    // with the thinking stripped out — produces a prompt that shares a head
    // with the retained context and then diverges. The expensive shared part,
    // which for a VLM includes the whole expanded image, is still reusable.
    let mut engine = load()?;

    let generated = turn(&mut engine, vec![3, 7], 0.0, 4)?;
    let mut branch = vec![3, 7];
    branch.extend_from_slice(&generated[..2]);
    branch.push(1);
    branch.push(2);

    engine.reset_cache_stats();
    let warm = turn(&mut engine, branch.clone(), 0.0, 3)?;
    // The retained context is [3, 7, g0, g1, g2]; the branch replaces g2, so
    // exactly the four shared tokens survive. Anything less than the full four
    // means the KV was thrown away instead of truncated.
    assert_eq!(
        engine.cache_stats().prefix_reused_tokens,
        4,
        "the shared head must be truncated to, not discarded"
    );

    // The only thing that matters: reusing must not change the answer.
    let mut fresh = load()?;
    assert_eq!(
        warm,
        turn(&mut fresh, branch, 0.0, 3)?,
        "a truncated-and-extended KV must produce exactly what recomputing produces"
    );
    Ok(())
}

#[test]
fn a_zero_budget_disables_reuse_without_changing_results() -> anyhow::Result<()> {
    let config = EngineConfig {
        pipeline_cache_bytes: 0,
        ..EngineConfig::default()
    };
    let mut disabled = Engine::from_pipeline_dir(&fixture_dir(), config)?;
    let mut enabled = load()?;

    let first = turn(&mut disabled, vec![3, 7], 0.0, 4)?;
    let second = turn(&mut disabled, vec![3, 7], 0.0, 4)?;
    assert_eq!(
        disabled.cache_stats().encoder_hits,
        0,
        "a zero budget must retain nothing"
    );
    assert_eq!(first, second);
    assert_eq!(
        first,
        turn(&mut enabled, vec![3, 7], 0.0, 4)?,
        "the cache must be invisible in the output"
    );
    Ok(())
}

#[test]
fn subagents_sharing_a_system_prompt_each_reuse_it() -> anyhow::Result<()> {
    // The server's real workload: many independent conversations that share a
    // long system prompt and then diverge, arriving interleaved. Each one must
    // reuse the shared head rather than re-prefilling it, even though the
    // request before it belonged to someone else.
    let mut engine = load()?;
    let system = vec![3, 7, 0, 5];

    let mut reuse_per_request = Vec::new();
    for question in [6u32, 1, 6, 2] {
        let mut prompt = system.clone();
        prompt.push(question);
        engine.reset_cache_stats();
        turn(&mut engine, prompt, 0.0, 2)?;
        reuse_per_request.push(engine.cache_stats().prefix_reused_tokens as usize);
    }

    // The first request has nothing retained; every later one, each following a
    // *different* conversation, still keeps the shared system prompt.
    assert_eq!(reuse_per_request[0], 0, "nothing is retained yet");
    for (index, reused) in reuse_per_request.iter().enumerate().skip(1) {
        assert_eq!(
            *reused,
            system.len(),
            "request {index} must reuse the whole shared system prompt"
        );
    }
    Ok(())
}

#[test]
fn interleaved_conversations_do_not_corrupt_each_other() -> anyhow::Result<()> {
    // Reuse across conversations is only worth having if it is invisible.
    let mut engine = load()?;
    let system = vec![3, 7, 0, 5];

    let mut shared = Vec::new();
    for question in [6u32, 1, 2] {
        let mut prompt = system.clone();
        prompt.push(question);
        shared.push(turn(&mut engine, prompt, 0.0, 3)?);
    }

    for (index, question) in [6u32, 1, 2].into_iter().enumerate() {
        let mut prompt = system.clone();
        prompt.push(question);
        let mut isolated = load()?;
        assert_eq!(
            shared[index],
            turn(&mut isolated, prompt, 0.0, 3)?,
            "conversation {index} must not be affected by the ones interleaved with it"
        );
    }
    Ok(())
}

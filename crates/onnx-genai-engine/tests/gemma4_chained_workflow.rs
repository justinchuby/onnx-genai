//! The chained speculative proposal loop, driven by the universal interpreter.
//!
//! These cases exercise the interpreter-owned chained proposal driver against
//! the hermetic `gemma4_chained` package: a tiny Gemma4-shaped target plus a
//! borrowed-KV assistant declaring `proposal_execution: {kind: chained}` with an
//! explicit folded carry. Nothing here names a model: the driver reads
//! `folded_carry_seed`, `folded_carry_output`, and `token_embedding` from the
//! package contract, so the same code drives the real Gemma4-E2B packages that
//! carry the identical field shape.
//!
//! `native_workflow_parity.rs` runs the same package under the native backend
//! and asserts the two agree token for token.

use std::path::Path;

use onnx_genai_engine::pipeline::PipelineEngine;
use onnx_genai_engine::{Engine, EngineConfig};

#[path = "common/chained.rs"]
mod chained;

use chained::{
    ChainedFixture, HIDDEN, PROMPT_TOKENS, fixture_root, greedy_reference, target_greedy_map,
};

fn engine(root: &Path) -> anyhow::Result<PipelineEngine> {
    Engine::from_pipeline_dir(root, EngineConfig::default())
}

#[test]
fn chained_contract_is_read_from_the_package() -> anyhow::Result<()> {
    let engine = engine(&fixture_root())?;
    let contract = engine
        .speculative_contract()
        .expect("gemma4_chained declares a speculative contract");
    assert_eq!(contract.proposer, "assistant");
    assert_eq!(contract.target, "target");
    assert_eq!(contract.max_proposal_width, 4);
    assert!(contract.distribution_preserving);
    // The folded carry is recomputed from committed tokens, never restored, so
    // it must not appear among the cells a rejection rolls back.
    assert!(
        !contract
            .rollback_state
            .iter()
            .any(|cell| cell.contains("projected_state")),
        "a folded carry must not be rollback state: {:?}",
        contract.rollback_state
    );
    assert_eq!(contract.rollback_state.len(), 4);
    Ok(())
}

/// The embedding half of the fused input is gathered from the table the contract
/// names, not from a heuristic scan of the proposer graph.
///
/// The tiny drafter deliberately ignores the embedding half (it slices only the
/// carry), so the parity case cannot prove the gather is right. This does: the
/// package ships the same table twice — as the target's `hidden_table`
/// initializer and as the raw `input_embedding.f32` the legacy proposer read —
/// and the two must agree row for row.
#[test]
fn token_embedding_gather_matches_the_declared_table() -> anyhow::Result<()> {
    let root = fixture_root();
    let engine = engine(&root)?;
    let contract = engine.speculative_contract().expect("speculative contract");
    let source = chained::token_embedding_source(contract);
    assert_eq!(source.component, "target");

    let table = engine.embedding_table(&source.component, &source.table)?;
    assert_eq!(table.hidden_size(), HIDDEN);

    let raw = std::fs::read(root.join("input_embedding.f32"))?;
    let reference = raw
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    assert_eq!(
        reference.len(),
        table.vocab_size() * table.hidden_size(),
        "the raw embedding file and the declared table disagree on size"
    );
    for token in 0..table.vocab_size() {
        let row = table.row(token as i64)?;
        let expected = &reference[token * HIDDEN..(token + 1) * HIDDEN];
        assert_eq!(row, expected, "embedding row {token} diverged");
    }
    // An id outside the table is an error, not a clamp onto row 0.
    assert!(table.row(table.vocab_size() as i64).is_err());
    assert!(table.row(-1).is_err());
    Ok(())
}

/// A chained proposal costs one proposer invocation per block position and
/// threads the folded carry forward without ever touching a state cell.
#[test]
fn chained_proposal_drives_the_declared_chain() -> anyhow::Result<()> {
    let mut fixture = ChainedFixture::new(engine(&fixture_root())?)?;
    let proposal = fixture.propose(PROMPT_TOKENS, 4)?;
    assert_eq!(proposal.tokens.len(), 4);
    assert_eq!(
        proposal.proposer_invocations, 4,
        "one invocation per block position, including the bootstrap step"
    );
    assert_eq!(
        proposal.tokens[0],
        fixture.target_next_token(PROMPT_TOKENS)?,
        "position 0 of a proposal block is the target's own guaranteed token"
    );
    Ok(())
}

/// The proposal block is verified by the target and its matching prefix is
/// accepted; the first divergence is rejected and the target's own token is
/// committed in its place.
#[test]
fn proposal_acceptance_follows_the_target() -> anyhow::Result<()> {
    let mut fixture = ChainedFixture::new(engine(&fixture_root())?)?;
    let proposal = fixture.propose(PROMPT_TOKENS, 4)?;
    let greedy = target_greedy_map(&fixture_root())?;

    // The target's tokens for the block positions, i.e. what verification of
    // `[guaranteed, drafts...]` produces.
    let target_tokens = fixture.verify(&proposal.tokens)?;
    let acceptance = fixture
        .engine()
        .accept_chained_proposal(&proposal, &target_tokens)?;

    assert!(
        acceptance.accepted >= 1,
        "position 0 is always the target's token"
    );
    assert_eq!(
        acceptance.committed.len(),
        acceptance.accepted + 1,
        "a verification pass always commits one token beyond its accepted prefix"
    );
    // Every committed token must be what plain greedy decoding would produce.
    let mut expected = Vec::new();
    let mut token = *PROMPT_TOKENS.last().expect("prompt is non-empty");
    for _ in 0..acceptance.committed.len() {
        token = greedy[token as usize];
        expected.push(token);
    }
    assert_eq!(
        acceptance.committed, expected,
        "speculative commits must equal plain greedy decoding"
    );
    Ok(())
}

/// A rejected proposal rolls every declared `rollback_state` cell back to the
/// accepted length, and leaves the folded carry alone.
#[test]
fn rejection_rolls_declared_state_back() -> anyhow::Result<()> {
    let mut fixture = ChainedFixture::new(engine(&fixture_root())?)?;
    let proposal = fixture.propose(PROMPT_TOKENS, 4)?;
    let target_tokens = fixture.verify(&proposal.tokens)?;
    let acceptance = fixture
        .engine()
        .accept_chained_proposal(&proposal, &target_tokens)?;

    let prefix = PROMPT_TOKENS.len();
    let mut state = fixture.verification_state(&proposal.tokens)?;
    let before = state
        .get("past_key_values.1.key")
        .expect("target KV cell")
        .shape()
        .to_vec();
    assert_eq!(
        before[2] as usize,
        prefix + proposal.tokens.len(),
        "verification extends the KV by the whole proposal block"
    );

    let committed = prefix + acceptance.committed.len();
    fixture
        .engine()
        .rollback_speculative_state(&mut state, committed)?;
    for cell in &fixture
        .engine()
        .speculative_contract()
        .unwrap()
        .rollback_state
    {
        let rolled = state.get(cell).expect("rolled back cell");
        assert_eq!(
            rolled.shape()[2] as usize,
            committed,
            "state cell '{cell}' was not rolled back to the committed length"
        );
    }
    assert!(
        !state.contains_key("projected_state"),
        "the folded carry owns no state cell"
    );
    Ok(())
}

/// A width beyond the package's declared rollback bound is refused, because the
/// package cannot undo that many positions.
#[test]
fn proposal_width_is_bounded_by_the_contract() -> anyhow::Result<()> {
    let mut fixture = ChainedFixture::new(engine(&fixture_root())?)?;
    let error = fixture
        .propose(PROMPT_TOKENS, 5)
        .expect_err("width 5 exceeds max_proposal_width 4");
    let message = format!("{error:#}");
    assert!(
        message.contains("max_proposal_width"),
        "unhelpful diagnostic: {message}"
    );
    Ok(())
}

/// The whole point of a distribution-preserving speculative package: driving it
/// through propose → verify → accept/reject → rollback must reproduce plain
/// greedy decoding token for token, and must actually exercise both branches.
#[test]
fn speculative_decode_equals_greedy_decode() -> anyhow::Result<()> {
    let root = fixture_root();
    let mut fixture = ChainedFixture::new(engine(&root)?)?;
    let (tokens, tally) = fixture.speculative_decode(8, 4)?;
    assert_eq!(
        tokens,
        greedy_reference(&root, 8)?,
        "speculative decoding diverged from plain greedy decoding"
    );
    assert!(
        tally.proposer_invocations >= tally.proposed,
        "every block position costs a proposer invocation: {tally:?}"
    );
    assert!(
        tally.rejections > 0,
        "this fixture's constant drafter must be rejected at least once: {tally:?}"
    );
    assert!(
        tally.rolled_back_cells > 0,
        "a rejection must roll the declared state cells back: {tally:?}"
    );
    Ok(())
}

/// Heterogeneous per-layer KV geometry (layer 0 sliding: head_dim 8, layer 1
/// full: head_dim 16), the workflow-runtime successor to the direct-`Engine`
/// `tiny-gemma4-assistant-mixed` regression.
///
/// Under the deleted `SharedKvProposerConfig` path a Rust KV *slicer* had to
/// pick each shared-KV group's head width out of a materialized paged cache,
/// which is exactly what got a uniform global geometry wrong. Here each group's
/// ports declare their own head_dim and the interpreter binds them straight
/// through, so the property is proven where it now lives: speculative decoding
/// on a mixed-geometry package is still token-identical to plain greedy.
#[test]
fn mixed_head_dim_speculative_decode_equals_greedy_decode() -> anyhow::Result<()> {
    let root = chained::mixed_fixture_root();
    let mut fixture = ChainedFixture::with_geometry(engine(&root)?, chained::MIXED)?;
    let (tokens, tally) = fixture.speculative_decode(8, 4)?;
    assert_eq!(
        tokens,
        greedy_reference(&root, 8)?,
        "mixed head_dim: speculative output must be token-identical to plain greedy"
    );
    assert!(tally.proposed > 0, "the proposer was not active: {tally:?}");
    Ok(())
}

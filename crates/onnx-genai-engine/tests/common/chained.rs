//! Shared driver for the hermetic `gemma4_chained` speculative fixture.
//!
//! The package's target is a lookup model — `logits = Gather(lm_table,
//! input_ids)` and `hidden_states.0 = Gather(hidden_table, input_ids)` — so
//! plain greedy decoding is a total token → token map that a test can compute
//! independently and compare speculative results against. Its KV grows by one
//! zero-filled position per consumed token, which is exactly what a rollback has
//! to undo.
//!
//! Nothing here is specific to the interpreter's *backend*: `ChainedFixture`
//! takes whichever `PipelineEngine` the caller built, so the ORT-only cases and
//! the ORT ⇄ native parity case drive the identical code.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use onnx_genai_engine::pipeline::PipelineEngine;
use onnx_genai_engine::pipeline::speculative::{ChainedProposal, ChainedProposalOptions};
use onnx_genai_engine::{
    GenerateOptions, GeneratePrompt, GenerateRequest, PipelineGenerateRequest, PipelineTensors,
};
use onnx_genai_ort::{DataType, Value};

/// Backbone hidden width `H`; the fused proposer input is `2H`.
pub const HIDDEN: usize = 16;
/// Vocabulary of both graphs.
pub const VOCAB: usize = 32;
/// Number of KV heads in each of the two target layers.
pub const KV_HEADS: usize = 2;
/// Per-head KV width.
pub const HEAD_DIM: usize = 8;
/// A prompt inside the fixture's vocabulary and context bound.
pub const PROMPT_TOKENS: &[i64] = &[3, 11, 7];

pub fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/onnx_genai_workflows/gemma4_chained")
}

/// The package's declared `token_embedding` source.
pub fn token_embedding_source(
    contract: &onnx_genai_metadata::SpeculativeContract,
) -> onnx_genai_metadata::TokenEmbeddingSource {
    match &contract.proposal_execution {
        onnx_genai_metadata::SpeculativeProposalExecution::Chained {
            token_embedding, ..
        } => token_embedding
            .clone()
            .expect("a folded-carry proposer declares token_embedding"),
        other => panic!("expected a chained proposal execution, found {other:?}"),
    }
}

/// Plain greedy decoding for the fixture's target: `next = argmax(lm_table[t])`.
///
/// Read straight out of the target graph's `lm_table` initializer so the oracle
/// is the model's own weights, not a hand-copied table.
pub fn target_greedy_map(root: &Path) -> anyhow::Result<Vec<i64>> {
    let (graph, weights) =
        onnx_runtime_loader::load_model_with_weights(root.join("target/model.onnx.textproto"))?;
    let (id, _) = graph
        .values
        .iter()
        .find(|(_, value)| value.name.as_deref() == Some("lm_table"))
        .expect("the target declares an lm_table initializer");
    let weight = graph
        .initializers
        .get(&id)
        .expect("lm_table is an initializer");
    let bytes = weights.bytes(weight).expect("lm_table has bytes");
    let table = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    assert_eq!(table.len(), VOCAB * VOCAB);
    Ok((0..VOCAB)
        .map(|token| {
            let row = &table[token * VOCAB..(token + 1) * VOCAB];
            let mut best = 0usize;
            for (index, value) in row.iter().enumerate() {
                if *value > row[best] {
                    best = index;
                }
            }
            best as i64
        })
        .collect())
}

/// Build the fixture's single-pass request for `tokens` consumed against a KV
/// cache holding `past` positions.
pub fn verification_request(
    tokens: &[i64],
    past: usize,
) -> anyhow::Result<PipelineGenerateRequest> {
    let sequence = tokens.len();
    let total = past + sequence;
    let mut request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![]),
        options: GenerateOptions {
            max_new_tokens: 1,
            greedy: true,
            temperature: 0.0,
            stop_on_eos: false,
            ..Default::default()
        },
    })
    .with_input("request.active", boolean(&[true])?)
    .with_input("request.done", boolean(&[false])?)
    .with_input("request.accepted_len", Value::from_slice_i64(&[0], &[1])?)
    .with_input(
        "request.input_ids",
        Value::from_slice_i64(tokens, &[1, sequence as i64])?,
    )
    .with_input(
        "request.attention_mask",
        Value::from_slice_i64(&vec![1; total], &[1, total as i64])?,
    )
    .with_input(
        "request.position_ids",
        Value::from_slice_i64(
            &(past..total).map(|p| p as i64).collect::<Vec<_>>(),
            &[1, sequence as i64],
        )?,
    )
    // The single-pass binding of the fused proposer input. The chained driver
    // overrides it every step; it exists so the package has a complete
    // single-pass contract.
    .with_input(
        "request.inputs_embeds",
        Value::from_vec_f32(
            vec![0.0; sequence * 2 * HIDDEN],
            &[1, sequence as i64, 2 * HIDDEN as i64],
        )?,
    );
    for layer in 0..2 {
        for role in ["key", "value"] {
            request = request.with_input(
                format!("request.past_key_values.{layer}.{role}"),
                kv_zeros(past)?,
            );
        }
    }
    // The assistant borrows the target's KV read-only; before the target has run
    // it sees the same prefix the target does.
    for group in ["full_attention", "sliding_attention"] {
        for role in ["key", "value"] {
            request = request.with_input(
                format!("request.shared_kv.{group}.{role}"),
                kv_zeros(total.max(1))?,
            );
        }
    }
    Ok(request)
}

fn kv_zeros(positions: usize) -> anyhow::Result<Value> {
    Ok(Value::from_vec_f32(
        vec![0.0; KV_HEADS * positions * HEAD_DIM],
        &[1, KV_HEADS as i64, positions as i64, HEAD_DIM as i64],
    )?)
}

fn boolean(values: &[bool]) -> anyhow::Result<Value> {
    Ok(Value::from_raw_bytes(
        values.iter().map(|v| u8::from(*v)).collect(),
        &[values.len() as i64],
        DataType::Bool,
    )?)
}

/// Drives the fixture through one `PipelineEngine`, whichever backend built it.
pub struct ChainedFixture {
    engine: PipelineEngine,
}

impl ChainedFixture {
    pub fn new(engine: PipelineEngine) -> anyhow::Result<Self> {
        Ok(Self { engine })
    }

    pub fn engine(&self) -> &PipelineEngine {
        &self.engine
    }

    /// Run the package's single pass over `tokens` and return every SSA value.
    pub fn run(&mut self, tokens: &[i64]) -> anyhow::Result<PipelineTensors> {
        self.engine
            .run_pipeline_retained(verification_request(tokens, 0)?)
    }

    /// The target's greedy next token after consuming `tokens`.
    pub fn target_next_token(&mut self, tokens: &[i64]) -> anyhow::Result<i64> {
        let values = self.run(tokens)?;
        let logits = values.get("logits").expect("emitted target logits");
        Ok(i64::from(logits.argmax_last_row()?))
    }

    /// Materialize a chained proposal of `width` positions after `tokens`.
    pub fn propose(&mut self, tokens: &[i64], width: usize) -> anyhow::Result<ChainedProposal> {
        let values = self.run(tokens)?;
        let guaranteed = i64::from(
            values
                .get("logits")
                .expect("emitted target logits")
                .argmax_last_row()?,
        );
        self.engine.propose_chained(
            &values,
            ChainedProposalOptions {
                seed_token: *tokens.last().expect("prompt is non-empty"),
                guaranteed_token: guaranteed,
                width,
            },
        )
    }

    /// The target's own tokens for a proposal block, block-aligned.
    ///
    /// Verification consumes `prompt + block` in one pass, so the target's token
    /// for block position `i` is its prediction at sequence index
    /// `prompt_len - 1 + i` — the row that had consumed everything up to, but not
    /// including, that position. The window runs one past the block because the
    /// same pass already produced the token that follows a fully-accepted block.
    pub fn verify(&mut self, block: &[i64]) -> anyhow::Result<Vec<i64>> {
        let values = self.run_block(block)?;
        let logits = values.get("logits").expect("emitted target logits");
        let vocab = *logits.shape().last().expect("logits have a vocab axis") as usize;
        let data = logits.to_vec_f32()?;
        let predictions = data
            .chunks_exact(vocab)
            .map(|row| {
                let mut best = 0usize;
                for (index, value) in row.iter().enumerate() {
                    if *value > row[best] {
                        best = index;
                    }
                }
                best as i64
            })
            .collect::<Vec<_>>();
        let start = PROMPT_TOKENS.len() - 1;
        Ok(predictions[start..start + block.len() + 1].to_vec())
    }

    /// State cells after verifying a proposal block on top of the prompt.
    pub fn verification_state(&mut self, block: &[i64]) -> anyhow::Result<PipelineTensors> {
        let values = self.run_block(block)?;
        let mut state = HashMap::new();
        for layer in 0..2 {
            for role in ["key", "value"] {
                let cell = format!("past_key_values.{layer}.{role}");
                let produced = values
                    .get(&format!("target.{cell}"))
                    .unwrap_or_else(|| panic!("target produced no '{cell}'"));
                state.insert(cell, produced.clone_owned()?);
            }
        }
        Ok(state)
    }

    /// Consume prompt + proposal block in one verification pass.
    fn run_block(&mut self, block: &[i64]) -> anyhow::Result<PipelineTensors> {
        let mut tokens = PROMPT_TOKENS.to_vec();
        tokens.extend_from_slice(block);
        self.engine
            .run_pipeline_retained(verification_request(&tokens, 0)?)
    }

    /// Speculatively decode `budget` tokens: propose, verify, accept the
    /// confirmed prefix, and roll the declared state back on every rejection.
    ///
    /// Returns the committed tokens and a tally of how the rounds went, so a
    /// test can prove both branches were exercised rather than trusting that a
    /// rejection ever happened.
    pub fn speculative_decode(
        &mut self,
        budget: usize,
        width: usize,
    ) -> anyhow::Result<(Vec<i64>, SpeculativeTally)> {
        let mut committed: Vec<i64> = Vec::new();
        let mut tally = SpeculativeTally::default();
        while committed.len() < budget {
            let mut context = PROMPT_TOKENS.to_vec();
            context.extend_from_slice(&committed);

            let values = self
                .engine
                .run_pipeline_retained(verification_request(&context, 0)?)?;
            let guaranteed = i64::from(
                values
                    .get("logits")
                    .expect("emitted target logits")
                    .argmax_last_row()?,
            );
            let proposal = self.engine.propose_chained(
                &values,
                ChainedProposalOptions {
                    seed_token: *context.last().expect("context is non-empty"),
                    guaranteed_token: guaranteed,
                    width,
                },
            )?;
            tally.proposed += proposal.tokens.len();
            tally.proposer_invocations += proposal.proposer_invocations;

            // Verify the whole block in one pass on top of the committed prefix.
            let mut block_context = context.clone();
            block_context.extend_from_slice(&proposal.tokens);
            let verified = self
                .engine
                .run_pipeline_retained(verification_request(&block_context, 0)?)?;
            let target_tokens =
                block_aligned_predictions(&verified, context.len(), proposal.tokens.len())?;
            let acceptance = self
                .engine
                .accept_chained_proposal(&proposal, &target_tokens)?;
            tally.accepted += acceptance.accepted;
            if acceptance.requires_rollback() {
                tally.rejections += 1;
                let mut state = state_cells(&verified)?;
                let length = context.len() + acceptance.committed.len();
                self.engine.rollback_speculative_state(&mut state, length)?;
                for (cell, value) in &state {
                    assert_eq!(
                        value.shape()[2] as usize,
                        length,
                        "state cell '{cell}' was not rolled back"
                    );
                }
                tally.rolled_back_cells += state.len();
            } else {
                tally.full_accepts += 1;
            }
            committed.extend_from_slice(&acceptance.committed);
        }
        committed.truncate(budget);
        Ok((committed, tally))
    }
}

/// How a speculative decode run split between acceptance and rejection.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SpeculativeTally {
    pub proposed: usize,
    pub accepted: usize,
    pub rejections: usize,
    pub full_accepts: usize,
    pub proposer_invocations: usize,
    pub rolled_back_cells: usize,
}

/// The target's block-aligned tokens, one entry past the block.
pub fn block_aligned_predictions(
    values: &PipelineTensors,
    context_len: usize,
    block_len: usize,
) -> anyhow::Result<Vec<i64>> {
    let logits = values.get("logits").expect("emitted target logits");
    let vocab = *logits.shape().last().expect("logits have a vocab axis") as usize;
    let data = logits.to_vec_f32()?;
    let predictions = data
        .chunks_exact(vocab)
        .map(|row| {
            let mut best = 0usize;
            for (index, value) in row.iter().enumerate() {
                if *value > row[best] {
                    best = index;
                }
            }
            best as i64
        })
        .collect::<Vec<_>>();
    let start = context_len - 1;
    Ok(predictions[start..start + block_len + 1].to_vec())
}

/// The target's KV state cells from a completed pass.
pub fn state_cells(values: &PipelineTensors) -> anyhow::Result<PipelineTensors> {
    let mut state = HashMap::new();
    for layer in 0..2 {
        for role in ["key", "value"] {
            let cell = format!("past_key_values.{layer}.{role}");
            let produced = values
                .get(&format!("target.{cell}"))
                .unwrap_or_else(|| panic!("target produced no '{cell}'"));
            state.insert(cell, produced.clone_owned()?);
        }
    }
    Ok(state)
}

/// Plain greedy decoding of `budget` tokens from the fixture's own target map.
pub fn greedy_reference(root: &Path, budget: usize) -> anyhow::Result<Vec<i64>> {
    let map = target_greedy_map(root)?;
    let mut token = *PROMPT_TOKENS.last().expect("prompt is non-empty");
    Ok((0..budget)
        .map(|_| {
            token = map[token as usize];
            token
        })
        .collect())
}

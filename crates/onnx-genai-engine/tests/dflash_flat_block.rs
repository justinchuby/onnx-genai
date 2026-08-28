//! Executable conformance for the generalized DFlash flat-block proposer.
//!
//! The Qwen-derived geometry keeps the published Qwen3.8-27B DFlash 2 block
//! width (8), five target feature taps (layers 5/19/33/47/61), five drafter
//! layers, bidirectional within-block mixing, target-feature projection, shared
//! input/output weights, and flat parallel prediction. Hidden width,
//! vocabulary, and learned tensor sizes are reduced deterministic tensors; this
//! is structural/equation evidence, not official-weight parity. The alternate
//! fixture changes block width, tap count, hidden width, and drafter depth while
//! using the same metadata dispatch.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use onnx_genai_engine::pipeline::{
    TurnAbortReason, TurnTransactionOutcome,
    speculative::{DFlashProposalMode, DFlashProposalOptions, DFlashVerificationMode},
};
use onnx_genai_engine::{
    Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest,
    PipelineGenerateRequest, PipelineTensors,
};
use onnx_genai_ort::{DataType, Value};

#[derive(Debug, Clone, Copy)]
struct Geometry {
    name: &'static str,
    block_size: usize,
    hidden: usize,
    target_taps: &'static [usize],
    draft_layers: usize,
    vocabulary: usize,
    recurrent_width: usize,
    mask_token: i64,
}

const QWEN_TARGET_TAPS: &[usize] = &[5, 19, 33, 47, 61];
const ALT_TARGET_TAPS: &[usize] = &[2, 7, 12];

const QWEN_REDUCED: Geometry = Geometry {
    name: "qwen38-reduced",
    block_size: 8,
    hidden: 4,
    target_taps: QWEN_TARGET_TAPS,
    draft_layers: 5,
    vocabulary: 17,
    recurrent_width: 3,
    mask_token: 16,
};

const ALTERNATE: Geometry = Geometry {
    name: "alternate",
    block_size: 4,
    hidden: 3,
    target_taps: ALT_TARGET_TAPS,
    draft_layers: 2,
    vocabulary: 11,
    recurrent_width: 2,
    mask_token: 10,
};

fn package(geometry: Geometry) -> anyhow::Result<PathBuf> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target-s13/test-fixtures/dflash")
        .join(format!(
            "{}-{}",
            geometry.name,
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
    fs::create_dir_all(&root)?;
    fs::write(root.join("inference_metadata.yaml"), metadata(geometry))?;
    fs::write(root.join("target.onnx.textproto"), target_model(geometry))?;
    fs::write(
        root.join("proposer.onnx.textproto"),
        proposer_model(geometry),
    )?;
    Ok(root)
}

fn request(
    geometry: Geometry,
    target_tokens: &[i64],
    target_cache: Value,
    draft_cache: Value,
    token_history: Value,
    recurrent: Value,
) -> anyhow::Result<PipelineGenerateRequest> {
    let verify = target_tokens.len();
    let context = verify.max(1);
    let target_hidden_width = geometry.target_taps.len() * geometry.hidden;
    Ok(PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(Vec::new()),
        options: GenerateOptions::default(),
    })
    .with_input("request.active", bool_vector(&[true])?)
    .with_input("request.done", bool_vector(&[false])?)
    .with_input("request.accepted_len", Value::from_slice_i64(&[0], &[1])?)
    .with_input(
        "request.target_tokens",
        Value::from_slice_i64(target_tokens, &[1, verify as i64])?,
    )
    .with_input("request.target_cache", target_cache)
    .with_input("request.draft_cache", draft_cache)
    .with_input("request.token_history", token_history)
    .with_input("request.recurrent", recurrent)
    .with_input(
        "request.target_hidden",
        Value::from_vec_f32(
            vec![0.0; context * target_hidden_width],
            &[1, context as i64, target_hidden_width as i64],
        )?,
    )
    .with_input(
        "request.noise",
        Value::from_vec_f32(
            vec![0.0; geometry.block_size * geometry.hidden],
            &[1, geometry.block_size as i64, geometry.hidden as i64],
        )?,
    )
    .with_input(
        "request.masked",
        bool_matrix(
            &(0..geometry.block_size)
                .map(|position| position > 0)
                .collect::<Vec<_>>(),
        )?,
    )
    .with_input(
        "request.positions",
        Value::from_slice_i64(
            &(0..context + geometry.block_size)
                .map(|position| position as i64)
                .collect::<Vec<_>>(),
            &[1, (context + geometry.block_size) as i64],
        )?,
    )
    .with_input(
        "request.attention",
        Value::from_slice_i64(
            &vec![1; context + geometry.block_size],
            &[1, (context + geometry.block_size) as i64],
        )?,
    )
    .with_input(
        "request.output_projection",
        Value::from_vec_f32(
            vec![0.0; geometry.hidden * geometry.vocabulary],
            &[geometry.hidden as i64, geometry.vocabulary as i64],
        )?,
    ))
}

fn empty_state(geometry: Geometry) -> anyhow::Result<PipelineTensors> {
    Ok(PipelineTensors::from([
        (
            "target_cache".to_string(),
            Value::from_vec_f32(Vec::new(), &[1, 0, geometry.hidden as i64])?,
        ),
        (
            "draft_cache".to_string(),
            Value::from_vec_f32(Vec::new(), &[1, 0, geometry.hidden as i64])?,
        ),
        (
            "token_history".to_string(),
            Value::from_slice_i64(&[], &[1, 0])?,
        ),
        (
            "recurrent".to_string(),
            Value::from_vec_f32(
                vec![0.0; geometry.recurrent_width],
                &[1, geometry.recurrent_width as i64],
            )?,
        ),
    ]))
}

fn request_from_state(
    geometry: Geometry,
    target_tokens: &[i64],
    state: &PipelineTensors,
) -> anyhow::Result<PipelineGenerateRequest> {
    request(
        geometry,
        target_tokens,
        state["target_cache"].clone_owned()?,
        state["draft_cache"].clone_owned()?,
        state["token_history"].clone_owned()?,
        state["recurrent"].clone_owned()?,
    )
}

fn bool_vector(values: &[bool]) -> anyhow::Result<Value> {
    Ok(Value::from_raw_bytes(
        values.iter().map(|value| u8::from(*value)).collect(),
        &[values.len() as i64],
        DataType::Bool,
    )?)
}

fn bool_matrix(values: &[bool]) -> anyhow::Result<Value> {
    Ok(Value::from_raw_bytes(
        values.iter().map(|value| u8::from(*value)).collect(),
        &[1, values.len() as i64],
        DataType::Bool,
    )?)
}

fn target_pass(
    engine: &mut Engine,
    geometry: Geometry,
    tokens: &[i64],
    state: &PipelineTensors,
) -> anyhow::Result<PipelineTensors> {
    engine.run_pipeline_retained(request_from_state(geometry, tokens, state)?)
}

fn replace_target_logits(
    verified: &mut PipelineTensors,
    tokens: &[i64],
    vocabulary: usize,
) -> anyhow::Result<()> {
    let mut logits = vec![-20.0f32; tokens.len() * vocabulary];
    for (row, token) in tokens.iter().copied().enumerate() {
        logits[row * vocabulary + token as usize] = 20.0;
    }
    verified.insert(
        "target.scores".to_string(),
        Value::from_vec_f32(logits, &[1, tokens.len() as i64, vocabulary as i64])?,
    );
    Ok(())
}

fn replace_target_distribution(
    verified: &mut PipelineTensors,
    probabilities: &[f32],
    rows: usize,
) -> anyhow::Result<()> {
    let vocabulary = probabilities.len();
    let logits = (0..rows)
        .flat_map(|_| probabilities.iter().map(|probability| probability.ln()))
        .collect::<Vec<_>>();
    verified.insert(
        "target.scores".to_string(),
        Value::from_vec_f32(logits, &[1, rows as i64, vocabulary as i64])?,
    );
    Ok(())
}

#[test]
fn two_distinct_executable_geometries_use_one_generic_dispatch() -> anyhow::Result<()> {
    for geometry in [QWEN_REDUCED, ALTERNATE] {
        let root = package(geometry)?;
        let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
        let state = empty_state(geometry)?;
        let context = target_pass(&mut engine, geometry, &[3], &state)?;
        let proposal = engine.propose_dflash(
            &context,
            DFlashProposalOptions {
                anchor_token: 3,
                width: geometry.block_size - 1,
                context_start_position: 0,
                mode: DFlashProposalMode::Greedy,
                eos_token_ids: vec![],
            },
        )?;
        assert_eq!(proposal.tokens.len(), geometry.block_size - 1);
        assert!(
            proposal.probabilities.is_some(),
            "sampling evidence requires full proposal probabilities"
        );
        assert!(
            proposal
                .tokens
                .iter()
                .all(|token| (0..geometry.vocabulary as i64).contains(token))
        );
        assert_eq!(
            engine
                .speculative_contract()
                .expect("DFlash contract")
                .proposer,
            "parallel_adapter"
        );
        let diagnostic = engine.dflash_diagnostic().expect("DFlash diagnostic");
        assert_eq!(diagnostic.version, "1");
        assert_eq!(
            diagnostic.target_hidden_sources.len(),
            geometry.target_taps.len()
        );
        assert_eq!(diagnostic.structure, "base");
        assert!(!diagnostic.shared_batching_supported);
    }
    Ok(())
}

#[test]
fn greedy_zero_partial_full_acceptance_commits_only_the_prefix() -> anyhow::Result<()> {
    let geometry = QWEN_REDUCED;
    let root = package(geometry)?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let mut state = empty_state(geometry)?;
    let context = target_pass(&mut engine, geometry, &[3], &state)?;
    state.insert(
        "target_cache".to_string(),
        context["target.hidden_5"].clone_owned()?,
    );
    state.insert(
        "token_history".to_string(),
        Value::from_slice_i64(&[3], &[1, 1])?,
    );
    let proposal = engine.propose_dflash(
        &context,
        DFlashProposalOptions {
            anchor_token: 3,
            width: 5,
            context_start_position: 0,
            mode: DFlashProposalMode::Greedy,
            eos_token_ids: vec![],
        },
    )?;

    for accepted in [0usize, 2, proposal.tokens.len()] {
        let mut verified = target_pass(
            &mut engine,
            geometry,
            &std::iter::once(3)
                .chain(proposal.tokens.iter().copied())
                .collect::<Vec<_>>(),
            &state,
        )?;
        let correction = (geometry.vocabulary - 2) as i64;
        let mut target = proposal.tokens.clone();
        if accepted < target.len() {
            target[accepted] = correction;
        }
        let bonus = (geometry.vocabulary - 3) as i64;
        target.push(bonus);
        replace_target_logits(&mut verified, &target, geometry.vocabulary)?;

        let transaction = engine.begin_dflash_state_transaction(&state)?;
        let acceptance =
            engine.verify_dflash(&verified, &proposal, DFlashVerificationMode::Greedy)?;
        assert_eq!(acceptance.accepted, accepted);
        assert_eq!(
            acceptance.rejected_at,
            (accepted < proposal.tokens.len()).then_some(accepted)
        );
        assert_eq!(acceptance.committed.len(), accepted + 1);
        assert_eq!(
            acceptance.committed.last().copied(),
            Some(if accepted < proposal.tokens.len() {
                correction
            } else {
                bonus
            })
        );
        let mut replay_tokens = vec![3];
        replay_tokens.extend_from_slice(&proposal.tokens[..accepted]);
        let replay = target_pass(&mut engine, geometry, &replay_tokens, &state)?;
        let outcome = engine.commit_dflash_state_transaction(
            transaction,
            &mut state,
            &proposal,
            &verified,
            &acceptance,
        )?;
        assert!(matches!(outcome, TurnTransactionOutcome::Committed { .. }));
        let expected_sequence = 1 + accepted;
        assert_eq!(state["target_cache"].shape()[1] as usize, expected_sequence);
        assert_eq!(state["draft_cache"].shape()[1] as usize, accepted);
        assert_eq!(
            state["token_history"].shape()[1] as usize,
            expected_sequence
        );
        assert_eq!(
            state["recurrent"].shape(),
            &[1, geometry.recurrent_width as i64]
        );
        assert_eq!(
            state["target_cache"].to_vec_f32()?,
            replay["target.present"].to_vec_f32()?,
            "target cache must equal replay of exactly the accepted proposal prefix"
        );
        assert_eq!(
            state["token_history"].to_vec_i64()?,
            replay["target.tokens"].to_vec_i64()?,
            "token-context history must retract every rejected suffix"
        );
        assert_eq!(
            state["recurrent"].to_vec_f32()?,
            replay["target.recurrent"].to_vec_f32()?,
            "fixed recurrent state must select the same prefix snapshot as replay"
        );

        state = empty_state(geometry)?;
        state.insert(
            "target_cache".to_string(),
            context["target.hidden_5"].clone_owned()?,
        );
        state.insert(
            "token_history".to_string(),
            Value::from_slice_i64(&[3], &[1, 1])?,
        );
    }
    Ok(())
}

#[test]
fn seeded_rejection_sampling_preserves_the_target_distribution() -> anyhow::Result<()> {
    let geometry = ALTERNATE;
    let root = package(geometry)?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let state = empty_state(geometry)?;
    let context = target_pass(&mut engine, geometry, &[2], &state)?;
    let target = [
        0.05f32, 0.10, 0.35, 0.25, 0.15, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let draws = 2048usize;
    let mut counts = vec![0usize; geometry.vocabulary];
    for seed in 0..draws as u64 {
        let proposal = engine.propose_dflash(
            &context,
            DFlashProposalOptions {
                anchor_token: 2,
                width: 1,
                context_start_position: 0,
                mode: DFlashProposalMode::Sampling { seed },
                eos_token_ids: vec![],
            },
        )?;
        let mut verified = target_pass(&mut engine, geometry, &[2, proposal.tokens[0]], &state)?;
        replace_target_distribution(&mut verified, &target, 2)?;
        let acceptance = engine.verify_dflash(
            &verified,
            &proposal,
            DFlashVerificationMode::Sampling { temperature: 1.0 },
        )?;
        counts[acceptance.committed[0] as usize] += 1;
    }
    for (token, expected) in target.iter().copied().enumerate() {
        let observed = counts[token] as f32 / draws as f32;
        assert!(
            (observed - expected).abs() < 0.035,
            "token {token}: observed {observed}, expected {expected}, counts={counts:?}"
        );
    }
    Ok(())
}

#[test]
fn sampling_without_probabilities_and_shared_batching_decline_before_mutation() -> anyhow::Result<()>
{
    let geometry = ALTERNATE;
    let root = package(geometry)?;
    let mut metadata = fs::read_to_string(root.join("inference_metadata.yaml"))?;
    metadata = metadata.replace("      proposal_probabilities: proposal_probabilities\n", "");
    fs::write(root.join("inference_metadata.yaml"), metadata)?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let state = empty_state(geometry)?;
    let context = target_pass(&mut engine, geometry, &[2], &state)?;
    let error = engine
        .propose_dflash(
            &context,
            DFlashProposalOptions {
                anchor_token: 2,
                width: 1,
                context_start_position: 0,
                mode: DFlashProposalMode::Sampling { seed: 1 },
                eos_token_ids: vec![],
            },
        )
        .expect_err("sampling must fail before proposal execution");
    assert!(format!("{error:#}").contains("requires declared proposal probabilities"));

    let mut batched = PipelineTensors::new();
    for tap in geometry.target_taps {
        batched.insert(
            format!("target.hidden_{tap}"),
            Value::from_vec_f32(
                vec![0.0; 2 * geometry.hidden],
                &[2, 1, geometry.hidden as i64],
            )?,
        );
    }
    let error = engine
        .propose_dflash(
            &batched,
            DFlashProposalOptions {
                anchor_token: 2,
                width: 1,
                context_start_position: 0,
                mode: DFlashProposalMode::Greedy,
                eos_token_ids: vec![],
            },
        )
        .expect_err("shared proposal execution is optional");
    assert!(
        format!("{error:#}").contains("execute each request row in isolation"),
        "{error:#}"
    );
    Ok(())
}

#[test]
fn missing_shared_weights_fail_package_admission_before_execution() -> anyhow::Result<()> {
    let geometry = ALTERNATE;
    let root = package(geometry)?;
    let metadata = fs::read_to_string(root.join("inference_metadata.yaml"))?
        .replace("initializer: lm_head", "initializer: missing_lm_head")
        .replace(
            "shared_weights: [token_embedding, lm_head]",
            "shared_weights: [token_embedding, missing_lm_head]",
        );
    fs::write(root.join("inference_metadata.yaml"), metadata)?;
    let error = match Engine::from_dir(&root, EngineConfig::default()) {
        Ok(_) => anyhow::bail!("a dangling immutable relationship passed package admission"),
        Err(error) => error,
    };
    let message = format!("{error:#}");
    assert!(
        message.contains("missing_lm_head")
            && message.contains("not an initializer")
            && message.contains("immutable target"),
        "{message}"
    );
    Ok(())
}

#[test]
fn eos_limits_failures_abort_and_retry_restore_the_s3_baseline() -> anyhow::Result<()> {
    let geometry = ALTERNATE;
    let root = package(geometry)?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let mut state = empty_state(geometry)?;
    let context = target_pass(&mut engine, geometry, &[2], &state)?;
    let first = engine.propose_dflash(
        &context,
        DFlashProposalOptions {
            anchor_token: 2,
            width: geometry.block_size - 1,
            context_start_position: 0,
            mode: DFlashProposalMode::Greedy,
            eos_token_ids: vec![],
        },
    )?;
    let eos = first.tokens[0];
    let cropped = engine.propose_dflash(
        &context,
        DFlashProposalOptions {
            anchor_token: 2,
            width: geometry.block_size - 1,
            context_start_position: 0,
            mode: DFlashProposalMode::Greedy,
            eos_token_ids: vec![eos],
        },
    )?;
    assert_eq!(cropped.tokens, vec![eos]);

    let error = engine
        .propose_dflash(
            &context,
            DFlashProposalOptions {
                anchor_token: 2,
                width: geometry.block_size,
                context_start_position: 0,
                mode: DFlashProposalMode::Greedy,
                eos_token_ids: vec![],
            },
        )
        .expect_err("context exhaustion caps width before mutation");
    assert!(format!("{error:#}").contains("exceeds max_proposal_width"));

    let transaction = engine.begin_dflash_state_transaction(&state)?;
    state.insert(
        "draft_cache".to_string(),
        Value::from_vec_f32(vec![9.0; geometry.hidden], &[1, 1, geometry.hidden as i64])?,
    );
    let outcome = engine.abort_dflash_state_transaction(
        transaction,
        &mut state,
        TurnAbortReason::Cancellation,
    )?;
    assert!(matches!(
        outcome,
        TurnTransactionOutcome::AbortToBaseline {
            reason: TurnAbortReason::Cancellation,
            ..
        }
    ));
    assert_eq!(state["draft_cache"].shape()[1], 0);

    let transaction = engine.begin_dflash_state_transaction(&state)?;
    let missing = PipelineTensors::new();
    let error = engine
        .propose_dflash(
            &missing,
            DFlashProposalOptions {
                anchor_token: 2,
                width: 1,
                context_start_position: 0,
                mode: DFlashProposalMode::Greedy,
                eos_token_ids: vec![],
            },
        )
        .expect_err("proposer failure");
    assert!(format!("{error:#}").contains("did not produce"));
    engine.abort_dflash_state_transaction(
        transaction,
        &mut state,
        TurnAbortReason::ExecutionFailure,
    )?;

    let retry = engine.propose_dflash(
        &context,
        DFlashProposalOptions {
            anchor_token: 2,
            width: 1,
            context_start_position: 0,
            mode: DFlashProposalMode::Sampling { seed: 77 },
            eos_token_ids: vec![],
        },
    )?;
    let retry_again = engine.propose_dflash(
        &context,
        DFlashProposalOptions {
            anchor_token: 2,
            width: 1,
            context_start_position: 0,
            mode: DFlashProposalMode::Sampling { seed: 77 },
            eos_token_ids: vec![],
        },
    )?;
    assert_eq!(retry.tokens, retry_again.tokens);

    let transaction = engine.begin_dflash_state_transaction(&state)?;
    let error = engine
        .verify_dflash(
            &PipelineTensors::new(),
            &retry,
            DFlashVerificationMode::Greedy,
        )
        .expect_err("verifier failure");
    assert!(format!("{error:#}").contains("did not produce target logits"));
    engine.abort_dflash_state_transaction(
        transaction,
        &mut state,
        TurnAbortReason::ExecutionFailure,
    )?;
    assert_eq!(state["draft_cache"].shape()[1], 0);
    Ok(())
}

fn metadata(geometry: Geometry) -> String {
    let mut target_hidden_outputs = String::new();
    let mut target_roles = String::new();
    let mut target_step_outputs = String::new();
    let mut conditioning_sources = String::new();
    for tap in geometry.target_taps {
        writeln!(
            target_hidden_outputs,
            r#"            hidden_{tap}:
              dtype: float32
              shape: [batch, verify, {hidden}]
              batch_layout: {{ kind: request_aligned, axis: 0 }}"#,
            hidden = geometry.hidden
        )
        .unwrap();
        writeln!(target_roles, "            hidden_{tap}: hidden_states").unwrap();
        writeln!(
            target_step_outputs,
            "          hidden_{tap}: target.hidden_{tap}"
        )
        .unwrap();
        writeln!(
            conditioning_sources,
            "        - {{ component: verifier, output: hidden_{tap} }}"
        )
        .unwrap();
    }
    let fused = geometry.target_taps.len() * geometry.hidden;
    let width = geometry.block_size - 1;
    format!(
        r#"
schema_version: v1.5
package:
  tokenizer:
    special_tokens:
      eos_token_id: [{mask_token}]
pipeline:
  workflow:
    manifest:
      capabilities:
        [workflow_ssa, typed_emit, serving_service_contract, dflash_flat_block]
    inputs:
      request.active:
        contract: {{ dtype: bool, shape: [batch], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: active }}
        required: true
      request.done:
        contract: {{ dtype: bool, shape: [batch], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: done }}
        required: true
      request.accepted_len:
        contract: {{ dtype: int64, shape: [batch], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: accepted_len }}
        required: true
      request.target_tokens:
        contract: {{ dtype: int64, shape: [batch, verify], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: runtime, version: v1, role: prompt_tokens }}
        source: {{ kind: request }}
        required: true
      request.target_cache:
        contract: {{ dtype: float32, shape: [batch, target_sequence, {hidden}], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: target_cache }}
        required: true
      request.draft_cache:
        contract: {{ dtype: float32, shape: [batch, draft_sequence, {hidden}], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: draft_cache }}
        required: true
      request.token_history:
        contract: {{ dtype: int64, shape: [batch, token_sequence], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: token_history }}
        required: true
      request.recurrent:
        contract: {{ dtype: float32, shape: [batch, {recurrent}], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: recurrent }}
        required: true
      request.target_hidden:
        contract: {{ dtype: float32, shape: [batch, verify, {fused}], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: target_hidden }}
        required: true
      request.noise:
        contract: {{ dtype: float32, shape: [batch, {block}, {hidden}], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: noise }}
        required: true
      request.masked:
        contract: {{ dtype: bool, shape: [batch, {block}], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: masked }}
        required: true
      request.positions:
        contract: {{ dtype: int64, shape: [batch, total], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: positions }}
        required: true
      request.attention:
        contract: {{ dtype: int64, shape: [batch, total], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: attention }}
        required: true
      request.output_projection:
        contract: {{ dtype: float32, shape: [{hidden}, {vocab}], batch_layout: {{ kind: shared }} }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: output_projection }}
        required: true
    outputs:
      target_logits:
        contract: {{ dtype: float32, shape: [batch, verify, {vocab}], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: tensor
        stage: pre_adapter
    components:
      termination_policy:
        implementation: {{ kind: binding }}
        ports: {{}}
        contract: {{ id: onnx-genai.token-policy, version: "1.0" }}
      parallel_adapter:
        implementation: {{ kind: onnx, artifact: proposer.onnx.textproto }}
        ports:
          inputs:
            target_features: {{ dtype: float32, shape: [batch, verify, {fused}], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            noise_embeddings: {{ dtype: float32, shape: [batch, {block}, {hidden}], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            masked_positions: {{ dtype: bool, shape: [batch, {block}], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            position_ids: {{ dtype: int64, shape: [batch, total], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            attention_mask: {{ dtype: int64, shape: [batch, total], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            output_projection: {{ dtype: float32, shape: [{hidden}, {vocab}], batch_layout: {{ kind: shared }} }}
            past_draft: {{ dtype: float32, shape: [batch, draft_sequence, {hidden}], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
          outputs:
            candidate_tokens: {{ dtype: int64, shape: [batch, {width}], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            proposal_probabilities: {{ dtype: float32, shape: [batch, {width}, {vocab}], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            present_draft: {{ dtype: float32, shape: [batch, updated_draft_sequence, {hidden}], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
      verifier:
        implementation: {{ kind: onnx, artifact: target.onnx.textproto }}
        ports:
          inputs:
            tokens: {{ dtype: int64, shape: [batch, verify], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            past_target: {{ dtype: float32, shape: [batch, target_sequence, {hidden}], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            token_history: {{ dtype: int64, shape: [batch, token_sequence], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            recurrent: {{ dtype: float32, shape: [batch, {recurrent}], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
          outputs:
{target_hidden_outputs}            logits: {{ dtype: float32, shape: [batch, verify, {vocab}], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            present_target: {{ dtype: float32, shape: [batch, updated_target_sequence, {hidden}], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            present_tokens: {{ dtype: int64, shape: [batch, updated_target_sequence], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            next_recurrent: {{ dtype: float32, shape: [batch, {recurrent}], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            recurrent_prefixes: {{ dtype: float32, shape: [batch, prefix, {recurrent}], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
          roles:
{target_roles}            logits: logits
    state:
      target_cache:
        contract: {{ dtype: float32, shape: [batch, target_sequence, {hidden}], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        scope: invocation
        initializer: request.target_cache
        recurrence: {{ kind: invariant }}
        management: runtime
        release_boundary: invocation
        service_group: target_cache
      draft_cache:
        contract: {{ dtype: float32, shape: [batch, draft_sequence, {hidden}], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        scope: invocation
        initializer: request.draft_cache
        recurrence: {{ kind: invariant }}
        management: runtime
        release_boundary: invocation
        service_group: draft_cache
      token_history:
        contract: {{ dtype: int64, shape: [batch, token_sequence], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        scope: invocation
        initializer: request.token_history
        recurrence: {{ kind: invariant }}
        management: runtime
        release_boundary: invocation
        service_group: token_history
      recurrent:
        contract: {{ dtype: float32, shape: [batch, {recurrent}], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        scope: invocation
        initializer: request.recurrent
        recurrence: {{ kind: invariant }}
        management: runtime
        release_boundary: invocation
        service_group: recurrent
    steps:
      - kind: invoke
        component: verifier
        inputs:
          tokens: request.target_tokens
          past_target: request.target_cache
          token_history: request.token_history
          recurrent: request.recurrent
        outputs:
{target_step_outputs}          logits: target.scores
          present_target: target.present
          present_tokens: target.tokens
          next_recurrent: target.recurrent
          recurrent_prefixes: target.recurrent_prefixes
      - kind: invoke
        component: parallel_adapter
        inputs:
          target_features: request.target_hidden
          noise_embeddings: request.noise
          masked_positions: request.masked
          position_ids: request.positions
          attention_mask: request.attention
          output_projection: request.output_projection
          past_draft: request.draft_cache
        outputs:
          candidate_tokens: proposal.tokens
          proposal_probabilities: proposal.probabilities
          present_draft: proposal.present
      - {{ kind: emit, value: target.scores, output: target_logits, mode: replace }}
    serving:
      active: request.active
      done: request.done
      accepted_len: request.accepted_len
      state_service:
        groups:
          target_cache:
            kind: full_attention
            sequence_axis: 1
            layout: bsh
            update: {{ kind: append }}
            capabilities: {{ rollback_positions: {width}, snapshot: true, fork: true, cascade: [draft_cache, token_history, recurrent] }}
            ports:
              verifier:
                target_cache: {{ input: past_target, output: present_target }}
          draft_cache:
            kind: full_attention
            sequence_axis: 1
            layout: bsh
            update: {{ kind: append }}
            capabilities: {{ rollback_positions: {width}, snapshot: true, fork: true, cascade: [target_cache, token_history, recurrent] }}
            ports:
              parallel_adapter:
                draft_cache: {{ input: past_draft, output: present_draft }}
          token_history:
            kind: recurrent
            sequence_axis: 1
            layout: bs
            update: {{ kind: append }}
            capabilities: {{ rollback_positions: {width}, snapshot: true, fork: true, cascade: [target_cache, draft_cache, recurrent] }}
            ports:
              verifier:
                token_history: {{ input: token_history, output: present_tokens }}
          recurrent:
            kind: recurrent
            layout: bf
            update: {{ kind: replace }}
            capabilities: {{ rollback_positions: {width}, snapshot: true, fork: true, cascade: [target_cache, draft_cache, token_history] }}
            ports:
              verifier:
                recurrent: {{ input: recurrent, output: next_recurrent }}
speculative:
  proposer: parallel_adapter
  target: verifier
  proposal_execution:
    kind: dflash_flat_block
    version: "1"
    conditioning:
      sources:
{conditioning_sources}      proposer_input: target_features
      combination: {{ kind: concatenate, axis: 2 }}
    block:
      noise_embeddings_input: noise_embeddings
      masked_positions_input: masked_positions
      position_ids_input: position_ids
      attention_mask_input: attention_mask
      anchor_position: 0
      first_candidate_position: 1
      mask_token_id: {mask_token}
    outputs:
      candidate_tokens: candidate_tokens
      proposal_probabilities: proposal_probabilities
      verifier_logits: {{ component: verifier, output: logits }}
    shared_weights:
      input_embedding: {{ component: verifier, table: token_embedding }}
      output_projection:
        component: verifier
        initializer: lm_head
        proposer_input: output_projection
        layout: hidden_vocabulary
    draft_private_state: [draft_cache]
    accepted_prefix_state:
      target_cache: {{ kind: sequence, source: {{ component: verifier, output: present_target }} }}
      draft_cache: {{ kind: sequence, source: {{ component: parallel_adapter, output: present_draft }} }}
      token_history: {{ kind: sequence, source: {{ component: verifier, output: present_tokens }} }}
      recurrent: {{ kind: prefix_snapshots, source: {{ component: verifier, output: recurrent_prefixes }}, axis: 1 }}
    structure: {{ kind: base }}
  shared_weights: [token_embedding, lm_head]
  vocabulary: {{ kind: identical }}
  max_proposal_width: {width}
  distribution_preserving: true
  rollback_state: [target_cache, draft_cache, token_history, recurrent]
"#,
        block = geometry.block_size,
        width = width,
        hidden = geometry.hidden,
        fused = fused,
        vocab = geometry.vocabulary,
        recurrent = geometry.recurrent_width,
        mask_token = geometry.mask_token,
    )
}

fn node(op: &str, inputs: &[&str], output: &str, attributes: &str) -> String {
    let inputs = inputs
        .iter()
        .map(|input| format!("  input: \"{input}\"\n"))
        .collect::<String>();
    format!("node {{\n{inputs}  output: \"{output}\"\n  op_type: \"{op}\"\n{attributes}}}\n")
}

fn initializer_f32(name: &str, shape: &[usize], values: &[f32]) -> String {
    let dims = shape
        .iter()
        .map(|dimension| format!("  dims: {dimension}\n"))
        .collect::<String>();
    let values = values
        .iter()
        .map(|value| format!("  float_data: {value}\n"))
        .collect::<String>();
    format!("initializer {{\n{dims}  data_type: 1\n{values}  name: \"{name}\"\n}}\n")
}

fn initializer_i64(name: &str, shape: &[usize], values: &[i64]) -> String {
    let dims = shape
        .iter()
        .map(|dimension| format!("  dims: {dimension}\n"))
        .collect::<String>();
    let values = values
        .iter()
        .map(|value| format!("  int64_data: {value}\n"))
        .collect::<String>();
    format!("initializer {{\n{dims}  data_type: 7\n{values}  name: \"{name}\"\n}}\n")
}

fn value_info(name: &str, elem_type: i32, dims: &[(&str, i64)]) -> String {
    let dimensions = dims
        .iter()
        .map(|(symbol, fixed)| {
            if *fixed >= 0 {
                format!("        dim {{ dim_value: {fixed} }}\n")
            } else {
                format!("        dim {{ dim_param: \"{symbol}\" }}\n")
            }
        })
        .collect::<String>();
    format!(
        r#"
{{
  name: "{name}"
  type {{
    tensor_type {{
      elem_type: {elem_type}
      shape {{
{dimensions}      }}
    }}
  }}
}}"#
    )
}

fn target_model(geometry: Geometry) -> String {
    let hidden = geometry.hidden;
    let vocab = geometry.vocabulary;
    let recurrent = geometry.recurrent_width;
    let mut graph = String::from("ir_version: 8\ngraph {\n  name: \"dflash_target\"\n");

    let embedding = (0..vocab * hidden)
        .map(|index| {
            let token = index / hidden;
            let feature = index % hidden;
            (((token * 7 + feature * 3 + 1) % 19) as f32 - 9.0) / 9.0
        })
        .collect::<Vec<_>>();
    let lm_head = (0..hidden * vocab)
        .map(|index| {
            let feature = index / vocab;
            let token = index % vocab;
            (((feature * 11 + token * 5 + 2) % 23) as f32 - 11.0) / 7.0
        })
        .collect::<Vec<_>>();
    graph.push_str(&initializer_f32(
        "token_embedding",
        &[vocab, hidden],
        &embedding,
    ));
    graph.push_str(&initializer_f32("lm_head", &[hidden, vocab], &lm_head));
    graph.push_str(&initializer_i64("slice_start_1", &[1], &[1]));
    graph.push_str(&initializer_i64("slice_end_max", &[1], &[i64::MAX]));
    graph.push_str(&initializer_i64("axis_1", &[1], &[1]));
    graph.push_str(&initializer_i64("axis_2", &[1], &[2]));
    graph.push_str(&initializer_i64("step_1", &[1], &[1]));
    graph.push_str(&initializer_i64("last_index", &[], &[-1]));
    graph.push_str(&initializer_i64("axis_1_scalar", &[], &[1]));

    graph.push_str(&node(
        "Gather",
        &["token_embedding", "tokens"],
        "token_hidden",
        "  attribute { name: \"axis\" i: 0 type: INT }\n",
    ));
    graph.push_str(&node("MatMul", &["token_hidden", "lm_head"], "logits", ""));
    for (tap_index, tap) in geometry.target_taps.iter().copied().enumerate() {
        let weights = (0..hidden * hidden)
            .map(|index| {
                let row = index / hidden;
                let column = index % hidden;
                if row == column {
                    1.0 + tap_index as f32 * 0.05
                } else {
                    (((tap + row + column) % 7) as f32 - 3.0) * 0.03
                }
            })
            .collect::<Vec<_>>();
        graph.push_str(&initializer_f32(
            &format!("hidden_weight_{tap}"),
            &[hidden, hidden],
            &weights,
        ));
        graph.push_str(&node(
            "MatMul",
            &["token_hidden", &format!("hidden_weight_{tap}")],
            &format!("hidden_{tap}"),
            "",
        ));
    }
    graph.push_str(&node(
        "Slice",
        &[
            "token_hidden",
            "slice_start_1",
            "slice_end_max",
            "axis_1",
            "step_1",
        ],
        "candidate_hidden",
        "",
    ));
    graph.push_str(&node(
        "Slice",
        &[
            "tokens",
            "slice_start_1",
            "slice_end_max",
            "axis_1",
            "step_1",
        ],
        "candidate_tokens",
        "",
    ));
    graph.push_str(&node(
        "Concat",
        &["past_target", "candidate_hidden"],
        "present_target",
        "  attribute { name: \"axis\" i: 1 type: INT }\n",
    ));
    graph.push_str(&node(
        "Concat",
        &["token_history", "candidate_tokens"],
        "present_tokens",
        "  attribute { name: \"axis\" i: 1 type: INT }\n",
    ));
    graph.push_str(&node(
        "Cast",
        &["candidate_tokens"],
        "candidate_float",
        "  attribute { name: \"to\" i: 1 type: INT }\n",
    ));
    graph.push_str(&node(
        "Unsqueeze",
        &["candidate_float", "axis_2"],
        "candidate_scalar",
        "",
    ));
    graph.push_str(&node(
        "CumSum",
        &["candidate_scalar", "axis_1_scalar"],
        "recurrent_delta",
        "",
    ));
    graph.push_str(&node(
        "Unsqueeze",
        &["recurrent", "axis_1"],
        "recurrent_seed",
        "",
    ));
    graph.push_str(&node(
        "Add",
        &["recurrent_delta", "recurrent_seed"],
        "recurrent_updates",
        "",
    ));
    graph.push_str(&node(
        "Concat",
        &["recurrent_seed", "recurrent_updates"],
        "recurrent_prefixes",
        "  attribute { name: \"axis\" i: 1 type: INT }\n",
    ));
    graph.push_str(&node(
        "Gather",
        &["recurrent_prefixes", "last_index"],
        "next_recurrent",
        "  attribute { name: \"axis\" i: 1 type: INT }\n",
    ));

    graph.push_str(&format!(
        "  input {}\n",
        value_info("tokens", 7, &[("batch", -1), ("verify", -1)])
    ));
    graph.push_str(&format!(
        "  input {}\n",
        value_info(
            "past_target",
            1,
            &[
                ("batch", -1),
                ("state_sequence", -1),
                ("hidden", hidden as i64)
            ]
        )
    ));
    graph.push_str(&format!(
        "  input {}\n",
        value_info("token_history", 7, &[("batch", -1), ("state_sequence", -1)])
    ));
    graph.push_str(&format!(
        "  input {}\n",
        value_info(
            "recurrent",
            1,
            &[("batch", -1), ("recurrent", recurrent as i64)]
        )
    ));
    for tap in geometry.target_taps {
        graph.push_str(&format!(
            "  output {}\n",
            value_info(
                &format!("hidden_{tap}"),
                1,
                &[("batch", -1), ("verify", -1), ("hidden", hidden as i64)]
            )
        ));
    }
    graph.push_str(&format!(
        "  output {}\n",
        value_info(
            "logits",
            1,
            &[("batch", -1), ("verify", -1), ("vocab", vocab as i64)]
        )
    ));
    graph.push_str(&format!(
        "  output {}\n",
        value_info(
            "present_target",
            1,
            &[("batch", -1), ("updated", -1), ("hidden", hidden as i64)]
        )
    ));
    graph.push_str(&format!(
        "  output {}\n",
        value_info("present_tokens", 7, &[("batch", -1), ("updated", -1)])
    ));
    graph.push_str(&format!(
        "  output {}\n",
        value_info(
            "next_recurrent",
            1,
            &[("batch", -1), ("recurrent", recurrent as i64)]
        )
    ));
    graph.push_str(&format!(
        "  output {}\n",
        value_info(
            "recurrent_prefixes",
            1,
            &[
                ("batch", -1),
                ("prefix", -1),
                ("recurrent", recurrent as i64)
            ]
        )
    ));
    graph.push_str("}\nopset_import { version: 18 }\n");
    graph
}

fn proposer_model(geometry: Geometry) -> String {
    let block = geometry.block_size;
    let width = block - 1;
    let hidden = geometry.hidden;
    let fused = geometry.target_taps.len() * hidden;
    let vocab = geometry.vocabulary;
    let mut graph = String::from("ir_version: 8\ngraph {\n  name: \"dflash_proposer\"\n");
    graph.push_str(&initializer_i64("reduce_context_axis", &[1], &[1]));
    graph.push_str(&initializer_i64("unsqueeze_last", &[1], &[2]));
    graph.push_str(&initializer_i64("slice_start_1", &[1], &[1]));
    graph.push_str(&initializer_i64("slice_end_block", &[1], &[block as i64]));
    graph.push_str(&initializer_i64(
        "slice_start_tail",
        &[1],
        &[-(block as i64)],
    ));
    graph.push_str(&initializer_i64("slice_end_max", &[1], &[i64::MAX]));
    graph.push_str(&initializer_i64("slice_axis_1", &[1], &[1]));
    graph.push_str(&initializer_i64("slice_step_1", &[1], &[1]));
    graph.push_str(&initializer_f32("position_scale", &[], &[0.001]));
    graph.push_str(&initializer_f32("attention_scale", &[], &[0.01]));
    graph.push_str(&initializer_f32("mask_scale", &[], &[0.02]));
    graph.push_str(&initializer_f32("mix_scale", &[], &[0.15]));
    let context_weights = (0..fused * hidden)
        .map(|index| {
            let row = index / hidden;
            let column = index % hidden;
            (((row * 3 + column * 5 + 1) % 17) as f32 - 8.0) / 13.0
        })
        .collect::<Vec<_>>();
    graph.push_str(&initializer_f32(
        "context_projection",
        &[fused, hidden],
        &context_weights,
    ));
    let block_mix = (0..block * block)
        .map(|index| {
            let row = index / block;
            let column = index % block;
            if row == column {
                0.7
            } else if row.abs_diff(column) == 1 {
                0.2
            } else {
                0.1 / block as f32
            }
        })
        .collect::<Vec<_>>();
    graph.push_str(&initializer_f32("block_mix", &[block, block], &block_mix));

    graph.push_str(&node(
        "ReduceMean",
        &["target_features", "reduce_context_axis"],
        "context_mean",
        "  attribute { name: \"keepdims\" i: 1 type: INT }\n",
    ));
    graph.push_str(&node(
        "MatMul",
        &["context_mean", "context_projection"],
        "context_projected",
        "",
    ));
    graph.push_str(&node(
        "Add",
        &["noise_embeddings", "context_projected"],
        "hidden_0_context",
        "",
    ));
    graph.push_str(&node(
        "Slice",
        &[
            "position_ids",
            "slice_start_tail",
            "slice_end_max",
            "slice_axis_1",
            "slice_step_1",
        ],
        "block_positions",
        "",
    ));
    graph.push_str(&node(
        "Slice",
        &[
            "attention_mask",
            "slice_start_tail",
            "slice_end_max",
            "slice_axis_1",
            "slice_step_1",
        ],
        "block_attention",
        "",
    ));
    for (input, output) in [
        ("block_positions", "positions_float"),
        ("block_attention", "attention_float"),
        ("masked_positions", "masked_float"),
    ] {
        graph.push_str(&node(
            "Cast",
            &[input],
            output,
            "  attribute { name: \"to\" i: 1 type: INT }\n",
        ));
    }
    for (input, output) in [
        ("positions_float", "positions_column"),
        ("attention_float", "attention_column"),
        ("masked_float", "masked_column"),
    ] {
        graph.push_str(&node("Unsqueeze", &[input, "unsqueeze_last"], output, ""));
    }
    graph.push_str(&node(
        "Mul",
        &["positions_column", "position_scale"],
        "position_control",
        "",
    ));
    graph.push_str(&node(
        "Mul",
        &["attention_column", "attention_scale"],
        "attention_control",
        "",
    ));
    graph.push_str(&node(
        "Mul",
        &["masked_column", "mask_scale"],
        "mask_control",
        "",
    ));
    graph.push_str(&node(
        "Add",
        &["position_control", "attention_control"],
        "controls_0",
        "",
    ));
    graph.push_str(&node(
        "Add",
        &["controls_0", "mask_control"],
        "controls",
        "",
    ));
    graph.push_str(&node(
        "Add",
        &["hidden_0_context", "controls"],
        "hidden_0",
        "",
    ));

    let mut previous = "hidden_0".to_string();
    for layer in 0..geometry.draft_layers {
        let weights = (0..hidden * hidden)
            .map(|index| {
                let row = index / hidden;
                let column = index % hidden;
                if row == column {
                    0.8 + layer as f32 * 0.02
                } else {
                    (((layer + row * 2 + column * 3) % 9) as f32 - 4.0) * 0.04
                }
            })
            .collect::<Vec<_>>();
        graph.push_str(&initializer_f32(
            &format!("layer_weight_{layer}"),
            &[hidden, hidden],
            &weights,
        ));
        graph.push_str(&node(
            "MatMul",
            &[&previous, &format!("layer_weight_{layer}")],
            &format!("layer_{layer}_projected"),
            "",
        ));
        graph.push_str(&node(
            "Tanh",
            &[&format!("layer_{layer}_projected")],
            &format!("layer_{layer}_activated"),
            "",
        ));
        graph.push_str(&node(
            "Transpose",
            &[&previous],
            &format!("layer_{layer}_transposed"),
            "  attribute { name: \"perm\" ints: 0 ints: 2 ints: 1 type: INTS }\n",
        ));
        graph.push_str(&node(
            "MatMul",
            &[&format!("layer_{layer}_transposed"), "block_mix"],
            &format!("layer_{layer}_mixed_t"),
            "",
        ));
        graph.push_str(&node(
            "Transpose",
            &[&format!("layer_{layer}_mixed_t")],
            &format!("layer_{layer}_mixed"),
            "  attribute { name: \"perm\" ints: 0 ints: 2 ints: 1 type: INTS }\n",
        ));
        graph.push_str(&node(
            "Mul",
            &[&format!("layer_{layer}_mixed"), "mix_scale"],
            &format!("layer_{layer}_mixed_scaled"),
            "",
        ));
        graph.push_str(&node(
            "Add",
            &[&format!("layer_{layer}_activated"), "context_projected"],
            &format!("layer_{layer}_conditioned"),
            "",
        ));
        graph.push_str(&node(
            "Add",
            &[&format!("layer_{layer}_conditioned"), &previous],
            &format!("layer_{layer}_residual"),
            "",
        ));
        graph.push_str(&node(
            "Add",
            &[
                &format!("layer_{layer}_residual"),
                &format!("layer_{layer}_mixed_scaled"),
            ],
            &format!("hidden_{}", layer + 1),
            "",
        ));
        previous = format!("hidden_{}", layer + 1);
    }
    graph.push_str(&node(
        "Slice",
        &[
            &previous,
            "slice_start_1",
            "slice_end_block",
            "slice_axis_1",
            "slice_step_1",
        ],
        "candidate_hidden",
        "",
    ));
    graph.push_str(&node(
        "MatMul",
        &["candidate_hidden", "output_projection"],
        "candidate_logits",
        "",
    ));
    graph.push_str(&node(
        "Softmax",
        &["candidate_logits"],
        "proposal_probabilities",
        "  attribute { name: \"axis\" i: 2 type: INT }\n",
    ));
    graph.push_str(&node(
        "ArgMax",
        &["candidate_logits"],
        "candidate_tokens",
        "  attribute { name: \"axis\" i: 2 type: INT }\n  attribute { name: \"keepdims\" i: 0 type: INT }\n",
    ));
    graph.push_str(&node(
        "Concat",
        &["past_draft", "candidate_hidden"],
        "present_draft",
        "  attribute { name: \"axis\" i: 1 type: INT }\n",
    ));

    for (name, elem_type, dims) in [
        (
            "target_features",
            1,
            vec![("batch", -1), ("context", -1), ("fused", fused as i64)],
        ),
        (
            "noise_embeddings",
            1,
            vec![
                ("batch", -1),
                ("block", block as i64),
                ("hidden", hidden as i64),
            ],
        ),
        (
            "masked_positions",
            9,
            vec![("batch", -1), ("block", block as i64)],
        ),
        ("position_ids", 7, vec![("batch", -1), ("total", -1)]),
        ("attention_mask", 7, vec![("batch", -1), ("total", -1)]),
        (
            "output_projection",
            1,
            vec![("hidden", hidden as i64), ("vocab", vocab as i64)],
        ),
        (
            "past_draft",
            1,
            vec![
                ("batch", -1),
                ("state_sequence", -1),
                ("hidden", hidden as i64),
            ],
        ),
    ] {
        graph.push_str(&format!("  input {}\n", value_info(name, elem_type, &dims)));
    }
    for (name, elem_type, dims) in [
        (
            "candidate_tokens",
            7,
            vec![("batch", -1), ("proposal", width as i64)],
        ),
        (
            "proposal_probabilities",
            1,
            vec![
                ("batch", -1),
                ("proposal", width as i64),
                ("vocab", vocab as i64),
            ],
        ),
        (
            "present_draft",
            1,
            vec![("batch", -1), ("updated", -1), ("hidden", hidden as i64)],
        ),
    ] {
        graph.push_str(&format!(
            "  output {}\n",
            value_info(name, elem_type, &dims)
        ));
    }
    graph.push_str("}\nopset_import { version: 18 }\n");
    graph
}

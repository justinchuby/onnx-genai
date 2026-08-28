//! Production DFlash v1 execution through the public engine surface.
//!
//! The two component artifacts are deliberately reduced, but are genuine ONNX
//! sessions.  The target produces the hidden conditioning and verifier logits;
//! the proposer receives the runtime-built anchor/mask block and produces every
//! candidate in one invocation.  No test replaces either component output.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use onnx_genai_engine::{
    Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest, PackageCapabilityError,
    package_capability_error,
};
use rand::rngs::StdRng;
use rand::{Rng as _, SeedableRng as _};

const TARGET: &str = r#"
ir_version: 8
graph {
  node { input: "tokens" output: "shape" op_type: "Shape" }
  node {
    output: "hidden_width"
    op_type: "Constant"
    attribute { name: "value_ints" ints: 2 type: INTS }
  }
  node { input: "shape" input: "hidden_width" output: "hidden_shape" op_type: "Concat" attribute { name: "axis" i: 0 type: INT } }
  node {
    input: "hidden_shape"
    output: "hidden"
    op_type: "ConstantOfShape"
    attribute { name: "value" t { dims: 1 data_type: 1 raw_data: "\000\000\000\000" } type: TENSOR }
  }
  node {
    output: "vocab"
    op_type: "Constant"
    attribute { name: "value_ints" ints: 4 type: INTS }
  }
  node { input: "shape" input: "vocab" output: "logits_shape" op_type: "Concat" attribute { name: "axis" i: 0 type: INT } }
  node {
    input: "logits_shape"
    output: "logits"
    op_type: "ConstantOfShape"
    attribute { name: "value" t { dims: 1 data_type: 1 raw_data: "\000\000\000\000" } type: TENSOR }
  }
  name: "dflash_target"
  input { name: "tokens" type { tensor_type { elem_type: 7 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } } } } }
  output { name: "hidden" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } dim { dim_value: 2 } } } } }
  output { name: "logits" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } dim { dim_value: 4 } } } } }
  initializer { name: "token_embedding" data_type: 1 dims: 4 dims: 2 raw_data: "\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000" }
  initializer { name: "lm_head" data_type: 1 dims: 2 dims: 4 raw_data: "\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000" }
}
opset_import { domain: "" version: 18 }
"#;

const PROPOSER: &str = r#"
ir_version: 8
graph {
  node {
    output: "candidate_tokens"
    op_type: "Constant"
    attribute {
      name: "value"
      t {
        dims: 1
        dims: 3
        data_type: 7
{candidate_values}
      }
      type: TENSOR
    }
  }
  node {
    output: "proposal_probabilities"
    op_type: "Constant"
    attribute {
      name: "value"
      t {
        dims: 1
        dims: 3
        dims: 4
        data_type: 1
        float_data: 1.0
        float_data: 0.0
        float_data: 0.0
        float_data: 0.0
        float_data: 1.0
        float_data: 0.0
        float_data: 0.0
        float_data: 0.0
        float_data: 1.0
        float_data: 0.0
        float_data: 0.0
        float_data: 0.0
      }
      type: TENSOR
    }
  }
  name: "dflash_proposer"
  input { name: "target_features" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } dim { dim_value: 2 } } } } }
  input { name: "noise_embeddings" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_value: 3 } dim { dim_value: 2 } } } } }
  input { name: "masked_positions" type { tensor_type { elem_type: 9 shape { dim { dim_param: "batch" } dim { dim_value: 3 } } } } }
  input { name: "position_ids" type { tensor_type { elem_type: 7 shape { dim { dim_param: "batch" } dim { dim_param: "total" } } } } }
  input { name: "attention_mask" type { tensor_type { elem_type: 7 shape { dim { dim_param: "batch" } dim { dim_param: "total" } } } } }
  input { name: "output_projection" type { tensor_type { elem_type: 1 shape { dim { dim_value: 2 } dim { dim_value: 4 } } } } }
  output { name: "candidate_tokens" type { tensor_type { elem_type: 7 shape { dim { dim_param: "batch" } dim { dim_value: 3 } } } } }
  output { name: "proposal_probabilities" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_value: 3 } dim { dim_value: 4 } } } } }
}
opset_import { domain: "" version: 18 }
"#;

fn metadata(version: &str, probabilities: bool) -> String {
    let probability_port = if probabilities {
        "      proposal_probabilities: proposal_probabilities\n"
    } else {
        ""
    };
    let probability_binding = if probabilities {
        "          proposal_probabilities: proposal.probabilities\n"
    } else {
        ""
    };
    let probability_output = if probabilities {
        "            proposal_probabilities: { dtype: float32, shape: [batch, 3, 4], batch_layout: { kind: request_aligned, axis: 0 } }\n"
    } else {
        ""
    };
    let (v2_outputs, v2_bindings, structure) = if version == "2" {
        (
            "            selector_candidates: { dtype: int64, shape: [batch, 3, 1], batch_layout: { kind: request_aligned, axis: 0 } }\n            selector_probabilities: { dtype: float32, shape: [batch, 3, 1], batch_layout: { kind: request_aligned, axis: 0 } }\n",
            "          selector_candidates: proposal.selector_candidates\n          selector_probabilities: proposal.selector_probabilities\n",
            "    structure:\n      kind: selector_convolution_v1\n      selector:\n        selected_tokens_output: candidate_tokens\n        candidate_ids_output: selector_candidates\n        conditional_probabilities_output: selector_probabilities\n        top_k: 1\n        rank: 1\n      convolution:\n        kernel_size: 2\n        group_size: 1\n        first_position_reads_anchor: true",
        )
    } else {
        ("", "", "    structure: { kind: base }")
    };
    format!(
        r#"
schema_version: v1.5
package:
  tokenizer:
    special_tokens:
      eos_token_id: [3]
pipeline:
  workflow:
    manifest:
      capabilities: [workflow_ssa, typed_emit, serving_service_contract, dflash_flat_block]
    inputs:
      request.tokens:
        contract: {{ dtype: int64, shape: [batch, sequence], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: runtime, version: v1, role: prompt_tokens }}
        source: {{ kind: request }}
        required: true
      package.noise:
        contract: {{ dtype: float32, shape: [batch, 3, 2], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: literal }}
        required: false
        default: 0.0
      package.masked:
        contract: {{ dtype: bool, shape: [batch, 3], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: literal }}
        required: false
        default: false
      package.positions:
        contract: {{ dtype: int64, shape: [batch, total], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: literal }}
        required: false
        default: 0
      package.attention:
        contract: {{ dtype: int64, shape: [batch, total], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: literal }}
        required: false
        default: 0
      package.output_projection:
        contract: {{ dtype: float32, shape: [2, 4], batch_layout: {{ kind: shared }} }}
        role: {{ kind: opaque }}
        source: {{ kind: literal }}
        required: false
        default: 0.0
    outputs: {{}}
    components:
      termination_policy:
        implementation: {{ kind: binding }}
        ports: {{}}
        contract: {{ id: onnx-genai.token-policy, version: "1.0" }}
      target_component:
        implementation: {{ kind: onnx, artifact: target.onnx.textproto }}
        ports:
          inputs:
            tokens: {{ dtype: int64, shape: [batch, sequence], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
          outputs:
            hidden: {{ dtype: float32, shape: [batch, sequence, 2], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            logits: {{ dtype: float32, shape: [batch, sequence, 4], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
          roles: {{ hidden: hidden_states, logits: logits }}
      proposer_component:
        implementation: {{ kind: onnx, artifact: proposer.onnx.textproto }}
        ports:
          inputs:
            target_features: {{ dtype: float32, shape: [batch, sequence, 2], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            noise_embeddings: {{ dtype: float32, shape: [batch, 3, 2], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            masked_positions: {{ dtype: bool, shape: [batch, 3], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            position_ids: {{ dtype: int64, shape: [batch, total], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            attention_mask: {{ dtype: int64, shape: [batch, total], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            output_projection: {{ dtype: float32, shape: [2, 4], batch_layout: {{ kind: shared }} }}
          outputs:
            candidate_tokens: {{ dtype: int64, shape: [batch, 3], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
{v2_outputs}{probability_output}    steps:
      - kind: invoke
        component: target_component
        inputs: {{ tokens: request.tokens }}
        outputs: {{ hidden: target.hidden, logits: target.logits }}
      - kind: invoke
        component: proposer_component
        inputs:
          target_features: target.hidden
          noise_embeddings: package.noise
          masked_positions: package.masked
          position_ids: package.positions
          attention_mask: package.attention
          output_projection: package.output_projection
        outputs:
          candidate_tokens: proposal.tokens
{v2_bindings}{probability_binding}speculative:
  proposer: proposer_component
  target: target_component
  proposal_execution:
    kind: dflash_flat_block
    version: "{version}"
    conditioning:
      sources: [{{ component: target_component, output: hidden }}]
      proposer_input: target_features
      combination: {{ kind: concatenate, axis: 2 }}
    block:
      noise_embeddings_input: noise_embeddings
      masked_positions_input: masked_positions
      position_ids_input: position_ids
      attention_mask_input: attention_mask
      anchor_position: 0
      first_candidate_position: 1
      mask_token_id: 0
    outputs:
      candidate_tokens: candidate_tokens
{probability_port}      verifier_logits: {{ component: target_component, output: logits }}
    shared_weights:
      input_embedding: {{ component: target_component, table: token_embedding }}
      output_projection:
        component: target_component
        initializer: lm_head
        proposer_input: output_projection
        layout: hidden_vocabulary
    accepted_prefix_state: {{}}
{structure}
  shared_weights: [token_embedding, lm_head]
  vocabulary: {{ kind: identical }}
  max_proposal_width: 2
  distribution_preserving: true
  rollback_state: []
"#
    )
}

fn package(version: &str, probabilities: bool, candidates: &[i64]) -> anyhow::Result<PathBuf> {
    assert_eq!(candidates.len(), 3);
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target-s13/test-fixtures/dflash-flat-block")
        .join(NEXT.fetch_add(1, Ordering::Relaxed).to_string());
    fs::create_dir_all(&root)?;
    fs::write(
        root.join("inference_metadata.yaml"),
        metadata(version, probabilities),
    )?;
    fs::write(root.join("target.onnx.textproto"), TARGET)?;
    let candidate_values = candidates
        .iter()
        .map(|token| format!("        int64_data: {token}"))
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(
        root.join("proposer.onnx.textproto"),
        PROPOSER.replace("{candidate_values}", &candidate_values),
    )?;
    Ok(root)
}

#[test]
fn engine_dispatches_dflash_v1_to_real_target_and_proposer_sessions() -> anyhow::Result<()> {
    for (candidates, expected_blocks) in [
        (&[0, 0, 0][..], 2_u64), // full acceptance + bonus
        (&[0, 1, 1][..], 2),     // partial acceptance + correction
        (&[1, 1, 1][..], 4),     // zero acceptance + correction
    ] {
        let root = package("1", false, candidates)?;
        let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
        let result = engine.generate(GenerateRequest {
            prompt: GeneratePrompt::TokenIds(vec![1, 2]),
            options: GenerateOptions {
                max_new_tokens: 4,
                greedy: true,
                stop_on_eos: false,
                ..GenerateOptions::default()
            },
        })?;
        assert_eq!(result.token_ids, vec![0, 0, 0, 0]);
        assert_eq!(
            engine
                .contract_executions()
                .get("onnx-genai.speculative-block")
                .copied(),
            Some(expected_blocks),
            "candidate path {candidates:?} must reach the real target verifier, not a \
             test-provided acceptance decision"
        );
        assert_eq!(
            engine
                .dflash_diagnostic()
                .expect("DFlash diagnostic")
                .version,
            "1"
        );
    }
    Ok(())
}

#[test]
fn sampling_without_declared_proposal_probabilities_refuses_before_a_block_runs()
-> anyhow::Result<()> {
    let root = package("1", false, &[0, 0, 0])?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let error = engine
        .generate(GenerateRequest {
            prompt: GeneratePrompt::TokenIds(vec![1]),
            options: GenerateOptions {
                greedy: false,
                temperature: 1.0,
                seed: Some(7),
                max_new_tokens: 1,
                ..GenerateOptions::default()
            },
        })
        .expect_err("sampling cannot use an undeclared proposal distribution");
    assert!(
        format!("{error:#}").contains("proposal probability distribution"),
        "{error:#}"
    );
    assert!(engine.contract_executions().is_empty());
    Ok(())
}

/// The reference follows the declared acceptance equation, not a helper in the
/// DFlash implementation.  The proposer emits q(0)=1 while the target is
/// uniform, so accepted candidate zero and residual corrections 1..3 are each
/// exactly one quarter likely.
fn sampling_reference(seed: u64) -> u32 {
    let mut driver_rng = StdRng::seed_from_u64(seed);
    let _anchor_draw = driver_rng.random::<f32>();
    let proposal_seed = driver_rng.random::<u64>();
    let mut proposal_rng = StdRng::seed_from_u64(proposal_seed);
    let _candidate_draw = proposal_rng.random::<f32>();
    let verification_seed = proposal_rng.random::<u64>();
    let mut verification_rng = StdRng::seed_from_u64(verification_seed);
    if verification_rng.random::<f32>() >= 0.25 {
        1 + (verification_rng.random::<f32>() * 3.0) as u32
    } else {
        0
    }
}

#[test]
fn dispatched_dflash_sampling_matches_the_declared_rejection_reference() -> anyhow::Result<()> {
    let root = package("1", true, &[0, 0, 0])?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let mut observed = [0_usize; 4];
    for seed in 0..128_u64 {
        let result = engine.generate(GenerateRequest {
            prompt: GeneratePrompt::TokenIds(vec![1]),
            options: GenerateOptions {
                greedy: false,
                temperature: 1.0,
                seed: Some(seed),
                max_new_tokens: 1,
                stop_on_eos: false,
                ..GenerateOptions::default()
            },
        })?;
        assert_eq!(result.token_ids.len(), 1);
        assert_eq!(
            result.token_ids[0],
            sampling_reference(seed),
            "seed {seed} did not use the target/proposal rejection equation"
        );
        observed[result.token_ids[0] as usize] += 1;
    }
    assert!(
        observed.iter().all(|count| *count > 0),
        "the dispatched sampling path must retain all target outcomes: {observed:?}"
    );
    Ok(())
}

#[test]
fn context_exhaustion_finishes_before_the_first_dflash_component_run() -> anyhow::Result<()> {
    let root = package("1", false, &[0, 0, 0])?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let result = engine.generate(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![1, 2]),
        options: GenerateOptions {
            max_new_tokens: 4,
            max_context: Some(2),
            greedy: true,
            ..GenerateOptions::default()
        },
    })?;
    assert!(result.token_ids.is_empty());
    assert_eq!(
        result.finish_reason,
        onnx_genai_engine::FinishReason::Length
    );
    assert!(
        engine.contract_executions().is_empty(),
        "context exhaustion must not begin a DFlash block"
    );
    Ok(())
}

#[test]
fn selector_convolution_v2_stays_a_typed_pre_mutation_refusal() -> anyhow::Result<()> {
    let root = package("2", false, &[0, 0, 0])?;
    let error = match Engine::from_dir(&root, EngineConfig::default()) {
        Ok(_) => panic!("v2 selector/convolution semantics are not implemented"),
        Err(error) => error,
    };
    assert!(matches!(
        package_capability_error(&error),
        Some(PackageCapabilityError::DFlashExecutionUnavailable { version, .. }) if version == "2"
    ));
    Ok(())
}

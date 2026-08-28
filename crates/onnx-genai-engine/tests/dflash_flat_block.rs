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
    PipelineGenerateRequest, SessionForkError, SessionPosition, package_capability_error,
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
  node { input: "tokens" output: "token_f32" op_type: "Cast" attribute { name: "to" i: 1 type: INT } }
  node { output: "hidden_axes" op_type: "Constant" attribute { name: "value_ints" ints: 2 type: INTS } }
  node { input: "token_f32" input: "hidden_axes" output: "token_hidden" op_type: "Unsqueeze" }
  node { input: "token_hidden" input: "token_hidden" output: "hidden" op_type: "Concat" attribute { name: "axis" i: 2 type: INT } }
  node { input: "hidden" output: "recurrent_prefixes" op_type: "Identity" }
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
  input { name: "recurrent" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_value: 2 } } } } }
  output { name: "hidden" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } dim { dim_value: 2 } } } } }
  output { name: "logits" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } dim { dim_value: 4 } } } } }
  output { name: "recurrent_prefixes" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } dim { dim_value: 2 } } } } }
  initializer { name: "token_embedding" data_type: 1 dims: 4 dims: 2 raw_data: "\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000" }
  initializer { name: "lm_head" data_type: 1 dims: 2 dims: 4 raw_data: "\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000" }
}
opset_import { domain: "" version: 18 }
"#;

const CONTEXTUAL_TARGET: &str = r#"
ir_version: 8
graph {
  node { input: "tokens" output: "shape" op_type: "Shape" }
  node {
    output: "hidden_width"
    op_type: "Constant"
    attribute { name: "value_ints" ints: 2 type: INTS }
  }
  node { input: "shape" input: "hidden_width" output: "hidden_shape" op_type: "Concat" attribute { name: "axis" i: 0 type: INT } }
  node { input: "tokens" output: "token_f32" op_type: "Cast" attribute { name: "to" i: 1 type: INT } }
  node { output: "hidden_axes" op_type: "Constant" attribute { name: "value_ints" ints: 2 type: INTS } }
  node { input: "token_f32" input: "hidden_axes" output: "token_hidden" op_type: "Unsqueeze" }
  node { input: "token_hidden" input: "token_hidden" output: "hidden" op_type: "Concat" attribute { name: "axis" i: 2 type: INT } }
  node { input: "hidden" output: "recurrent_prefixes" op_type: "Identity" }
  node {
    output: "depth"
    op_type: "Constant"
    attribute { name: "value" t { data_type: 7 int64_data: 4 } type: TENSOR }
  }
  node {
    output: "one_hot_values"
    op_type: "Constant"
    attribute { name: "value" t { dims: 2 data_type: 1 float_data: 0.0 float_data: 1.0 } type: TENSOR }
  }
  node {
    input: "tokens"
    input: "depth"
    input: "one_hot_values"
    output: "logits"
    op_type: "OneHot"
    attribute { name: "axis" i: -1 type: INT }
  }
  name: "dflash_contextual_target"
  input { name: "tokens" type { tensor_type { elem_type: 7 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } } } } }
  input { name: "recurrent" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_value: 2 } } } } }
  output { name: "hidden" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } dim { dim_value: 2 } } } } }
  output { name: "logits" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } dim { dim_value: 4 } } } } }
  output { name: "recurrent_prefixes" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } dim { dim_value: 2 } } } } }
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
        dims: 8
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
        dims: 8
        dims: 4
        data_type: 1
{probability_values}
      }
      type: TENSOR
    }
  }
  name: "dflash_proposer"
  input { name: "target_features" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_param: "sequence" } dim { dim_value: 2 } } } } }
  input { name: "noise_embeddings" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_value: 8 } dim { dim_value: 2 } } } } }
  input { name: "masked_positions" type { tensor_type { elem_type: 9 shape { dim { dim_param: "batch" } dim { dim_value: 8 } } } } }
  input { name: "position_ids" type { tensor_type { elem_type: 7 shape { dim { dim_param: "batch" } dim { dim_param: "total" } } } } }
  input { name: "attention_mask" type { tensor_type { elem_type: 7 shape { dim { dim_param: "batch" } dim { dim_param: "total" } } } } }
  input { name: "output_projection" type { tensor_type { elem_type: 1 shape { dim { dim_value: 2 } dim { dim_value: 4 } } } } }
  output { name: "candidate_tokens" type { tensor_type { elem_type: 7 shape { dim { dim_param: "batch" } dim { dim_value: 8 } } } } }
  output { name: "proposal_probabilities" type { tensor_type { elem_type: 1 shape { dim { dim_param: "batch" } dim { dim_value: 8 } dim { dim_value: 4 } } } } }
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
        "            proposal_probabilities: { dtype: float32, shape: [batch, 8, 4], batch_layout: { kind: request_aligned, axis: 0 } }\n"
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
      request.active:
        contract: {{ dtype: bool, shape: [batch], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: literal }}
        required: false
        default: true
      request.done:
        contract: {{ dtype: bool, shape: [batch], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: literal }}
        required: false
        default: false
      request.accepted_len:
        contract: {{ dtype: int64, shape: [batch], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: literal }}
        required: false
        default: 0
      request.recurrent:
        contract: {{ dtype: float32, shape: [batch, 2], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: literal }}
        required: false
        default: 0.0
      package.noise:
        contract: {{ dtype: float32, shape: [batch, 8, 2], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        role: {{ kind: opaque }}
        source: {{ kind: literal }}
        required: false
        default: 0.0
      package.masked:
        contract: {{ dtype: bool, shape: [batch, 8], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
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
        contract: {{ id: onnx-genai.dflash-target-state, version: "1.0" }}
        ports:
          inputs:
            tokens: {{ dtype: int64, shape: [batch, sequence], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            recurrent: {{ dtype: float32, shape: [batch, 2], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
          outputs:
            hidden: {{ dtype: float32, shape: [batch, sequence, 2], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            logits: {{ dtype: float32, shape: [batch, sequence, 4], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            recurrent_prefixes: {{ dtype: float32, shape: [batch, sequence, 2], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
          roles: {{ hidden: hidden_states, logits: logits }}
      proposer_component:
        implementation: {{ kind: onnx, artifact: proposer.onnx.textproto }}
        ports:
          inputs:
            target_features: {{ dtype: float32, shape: [batch, sequence, 2], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            noise_embeddings: {{ dtype: float32, shape: [batch, 8, 2], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            masked_positions: {{ dtype: bool, shape: [batch, 8], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            position_ids: {{ dtype: int64, shape: [batch, total], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            attention_mask: {{ dtype: int64, shape: [batch, total], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
            output_projection: {{ dtype: float32, shape: [2, 4], batch_layout: {{ kind: shared }} }}
          outputs:
            candidate_tokens: {{ dtype: int64, shape: [batch, 8], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
{v2_outputs}{probability_output}    steps:
      - kind: invoke
        component: target_component
        inputs: {{ tokens: request.tokens, recurrent: request.recurrent }}
        outputs: {{ hidden: target.hidden, logits: target.logits, recurrent_prefixes: target.recurrent_prefixes }}
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
{v2_bindings}{probability_binding}    state:
      recurrent:
        contract: {{ dtype: float32, shape: [batch, 2], batch_layout: {{ kind: request_aligned, axis: 0 }} }}
        scope: invocation
        initializer: request.recurrent
        recurrence: {{ kind: invariant }}
        management: runtime
        release_boundary: invocation
        service_group: recurrent
    serving:
      active: request.active
      done: request.done
      accepted_len: request.accepted_len
      state_service:
        groups:
          recurrent:
            kind: recurrent
            layout: bf
            update: {{ kind: replace }}
            capabilities: {{ rollback_positions: 7, snapshot: true, fork: true }}
            ports:
              target_component:
                recurrent: {{ input: recurrent, output: recurrent_prefixes }}
speculative:
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
    accepted_prefix_state:
      recurrent: {{ kind: prefix_snapshots, source: {{ component: target_component, output: recurrent_prefixes }}, axis: 1 }}
{structure}
  shared_weights: [token_embedding, lm_head]
  vocabulary: {{ kind: identical }}
  max_proposal_width: 7
  distribution_preserving: true
  rollback_state: [recurrent]
"#
    )
}

fn package(version: &str, probabilities: bool, candidates: &[i64]) -> anyhow::Result<PathBuf> {
    package_with_target(version, probabilities, candidates, TARGET)
}

fn package_with_target(
    version: &str,
    probabilities: bool,
    candidates: &[i64],
    target: &str,
) -> anyhow::Result<PathBuf> {
    assert_eq!(candidates.len(), 8);
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target-s13/test-fixtures/dflash-flat-block")
        .join(NEXT.fetch_add(1, Ordering::Relaxed).to_string());
    fs::create_dir_all(&root)?;
    fs::write(
        root.join("inference_metadata.yaml"),
        metadata(version, probabilities),
    )?;
    fs::write(root.join("target.onnx.textproto"), target)?;
    let candidate_values = candidates
        .iter()
        .map(|token| format!("        int64_data: {token}"))
        .collect::<Vec<_>>()
        .join("\n");
    let probability_values = (0..8)
        .flat_map(|_| [1.0_f32, 0.0, 0.0, 0.0])
        .map(|probability| format!("        float_data: {probability:.1}"))
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(
        root.join("proposer.onnx.textproto"),
        PROPOSER
            .replace("{candidate_values}", &candidate_values)
            .replace("{probability_values}", &probability_values),
    )?;
    Ok(root)
}

fn alternate_geometry_package() -> anyhow::Result<PathBuf> {
    let root = package_with_target("1", false, &[1, 1, 1, 1, 1, 1, 1, 1], CONTEXTUAL_TARGET)?;
    let alternate_metadata = metadata("1", false)
        .replace("[batch, 8, 2]", "[batch, 5, 2]")
        .replace("[batch, 8, 4]", "[batch, 5, 4]")
        .replace("[batch, 8]", "[batch, 5]")
        .replace("rollback_positions: 7", "rollback_positions: 4")
        .replace("max_proposal_width: 7", "max_proposal_width: 4");
    fs::write(root.join("inference_metadata.yaml"), alternate_metadata)?;
    let candidate_values = (0..5)
        .map(|_| "        int64_data: 1")
        .collect::<Vec<_>>()
        .join("\n");
    let probability_values = (0..5)
        .flat_map(|_| [1.0_f32, 0.0, 0.0, 0.0])
        .map(|probability| format!("        float_data: {probability:.1}"))
        .collect::<Vec<_>>()
        .join("\n");
    let proposer = PROPOSER
        .replace("dims: 8", "dims: 5")
        .replace("dim_value: 8", "dim_value: 5")
        .replace("{candidate_values}", &candidate_values)
        .replace("{probability_values}", &probability_values);
    fs::write(root.join("proposer.onnx.textproto"), proposer)?;
    Ok(root)
}

fn pipeline_request() -> PipelineGenerateRequest {
    PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![1, 2]),
        options: GenerateOptions {
            max_new_tokens: 4,
            greedy: true,
            stop_on_eos: false,
            ..GenerateOptions::default()
        },
    })
}

fn expect_error<T>(result: anyhow::Result<T>) -> anyhow::Error {
    match result {
        Ok(_) => panic!("DFlash raw workflow API unexpectedly executed"),
        Err(error) => error,
    }
}

#[test]
fn every_raw_workflow_api_typed_refuses_before_dflash_component_execution() -> anyhow::Result<()> {
    let root = package("1", false, &[0, 0, 0, 0, 0, 0, 0, 0])?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    for (operation, error) in [
        (
            "Engine::run_pipeline",
            expect_error(engine.run_pipeline(pipeline_request())),
        ),
        (
            "Engine::run_pipeline_outputs",
            expect_error(engine.run_pipeline_outputs(pipeline_request())),
        ),
        (
            "Engine::run_pipeline_retained",
            expect_error(engine.run_pipeline_retained(pipeline_request())),
        ),
    ] {
        assert!(matches!(
            package_capability_error(&error),
            Some(PackageCapabilityError::DFlashRawWorkflowApi { operation: actual })
                if actual == operation
        ));
    }
    {
        let error = expect_error(engine.prepare_pipeline(pipeline_request()));
        assert!(matches!(
            package_capability_error(&error),
            Some(PackageCapabilityError::DFlashRawWorkflowApi { operation })
                if operation == "Engine::prepare_pipeline"
        ));
    }
    {
        let error = expect_error(engine.models());
        assert!(matches!(
            package_capability_error(&error),
            Some(PackageCapabilityError::DFlashRawWorkflowApi { operation })
                if operation == "Engine::models"
        ));
    }
    {
        let error = expect_error(engine.prepare_workflow_execution(pipeline_request()));
        assert!(matches!(
            package_capability_error(&error),
            Some(PackageCapabilityError::DFlashRawWorkflowApi { operation })
                if operation == "Engine::prepare_workflow_execution"
        ));
    }
    assert!(
        engine.contract_executions().is_empty(),
        "raw API refusal must happen before target or proposer mutation"
    );
    let generated = engine.generate_with_pipeline_request(pipeline_request())?;
    assert_eq!(generated.token_ids.len(), 4);
    Ok(())
}

#[test]
fn consecutive_blocks_commit_anchor_and_never_condition_on_rejected_suffix() -> anyhow::Result<()> {
    let root = package_with_target("1", false, &[1, 1, 1, 1, 1, 1, 1, 1], CONTEXTUAL_TARGET)?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let result = engine.generate(pipeline_request().request)?;
    assert_eq!(
        result.token_ids,
        vec![2, 2, 2, 2],
        "each block must commit its target anchor and correction; the rejected token 1 must \
         never become the next block's verifier anchor"
    );
    assert_eq!(
        engine
            .contract_executions()
            .get("onnx-genai.speculative-block")
            .copied(),
        Some(2),
        "zero-acceptance blocks commit anchor+correction, so four tokens require two blocks"
    );
    Ok(())
}

#[test]
fn dflash_batching_uses_isolated_generation_and_fork_declines_before_child_allocation()
-> anyhow::Result<()> {
    let root = package_with_target("1", false, &[1, 1, 1, 1, 1, 1, 1, 1], CONTEXTUAL_TARGET)?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let expected = engine.generate(pipeline_request().request)?;
    let batched = engine
        .generate_batched_static(vec![pipeline_request().request, pipeline_request().request])?;
    assert_eq!(batched.len(), 2);
    assert!(
        batched
            .iter()
            .all(|result| result.token_ids == expected.token_ids),
        "unsupported shared DFlash batching must preserve the isolated generation path"
    );

    let source = engine.create_session()?;
    let error = match engine.prepare_session_fork(source, SessionPosition::new(0)) {
        Ok(_) => panic!("DFlash fork must decline before publishing a child"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        SessionForkError::UnsupportedParticipant { participant, reason, .. }
            if participant == "dflash.accepted_prefix"
                && reason.contains("before child allocation")
    ));
    Ok(())
}

#[test]
fn output_callback_failure_is_post_commit_and_retry_is_deterministic() -> anyhow::Result<()> {
    let root = package_with_target("1", false, &[1, 1, 1, 1, 1, 1, 1, 1], CONTEXTUAL_TARGET)?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let mut delivered = Vec::new();
    let mut callback = |token: onnx_genai_engine::GenerateToken| -> anyhow::Result<()> {
        delivered.push(token.token_id);
        anyhow::bail!("injected delivery failure")
    };
    let error = engine
        .generate_with_callback(pipeline_request().request, Some(&mut callback))
        .expect_err("callback failure must be reported");
    let message = format!("{error:#}");
    assert!(message.contains("after semantic commit"), "{message}");
    assert!(message.contains("will not be replayed"), "{message}");
    assert_eq!(delivered, vec![2]);
    assert_eq!(
        engine
            .contract_executions()
            .get("onnx-genai.speculative-block")
            .copied(),
        Some(2),
        "all model execution must finish before the first callback"
    );

    let retry = engine.generate(pipeline_request().request)?;
    assert_eq!(retry.token_ids, vec![2, 2, 2, 2]);
    Ok(())
}

#[test]
fn alternate_block_geometry_uses_the_same_structural_dispatch() -> anyhow::Result<()> {
    let root = alternate_geometry_package()?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let diagnostic = engine.dflash_diagnostic().expect("DFlash diagnostic");
    assert_eq!(diagnostic.max_proposal_width, 4);
    let result = engine.generate(pipeline_request().request)?;
    assert_eq!(result.token_ids, vec![2, 2, 2, 2]);
    assert_eq!(
        engine
            .contract_executions()
            .get("onnx-genai.speculative-block")
            .copied(),
        Some(2)
    );
    Ok(())
}

#[test]
fn proposer_failure_publishes_nothing_and_a_valid_retry_is_deterministic() -> anyhow::Result<()> {
    let invalid = package("1", false, &[9, 9, 9, 9, 9, 9, 9, 9])?;
    let mut failed = Engine::from_dir(&invalid, EngineConfig::default())?;
    let error = failed
        .generate(pipeline_request().request)
        .expect_err("out-of-vocabulary proposal must fail before commit");
    assert!(
        format!("{error:#}").contains("outside shared target vocabulary"),
        "{error:#}"
    );
    assert!(
        failed.contract_executions().is_empty(),
        "a failed proposer block must not record a committed execution"
    );
    assert!(
        failed.take_committed_workflow_publications().is_empty(),
        "a failed proposer block must not expose a pre-commit publication"
    );

    let valid = package_with_target("1", false, &[1, 1, 1, 1, 1, 1, 1, 1], CONTEXTUAL_TARGET)?;
    let mut retry = Engine::from_dir(&valid, EngineConfig::default())?;
    assert_eq!(
        retry.generate(pipeline_request().request)?.token_ids,
        vec![2, 2, 2, 2]
    );
    Ok(())
}

#[test]
fn eos_anchor_commits_once_and_stops_before_any_suffix_is_exposed() -> anyhow::Result<()> {
    let root = package_with_target("1", false, &[1, 1, 1, 1, 1, 1, 1, 1], CONTEXTUAL_TARGET)?;
    let mut engine = Engine::from_dir(&root, EngineConfig::default())?;
    let mut request = pipeline_request().request;
    request.options.stop_on_eos = true;
    request.options.eos_token_ids = vec![2];
    let result = engine.generate(request)?;
    assert_eq!(result.token_ids, vec![2]);
    assert_eq!(
        result.finish_reason,
        onnx_genai_engine::FinishReason::EosToken
    );
    Ok(())
}

#[test]
fn engine_dispatches_dflash_v1_to_real_target_and_proposer_sessions() -> anyhow::Result<()> {
    for (candidates, expected_blocks) in [
        (&[0, 0, 0, 0, 0, 0, 0, 0][..], 1_u64), // anchor + full acceptance
        (&[0, 1, 1, 1, 1, 1, 1, 1][..], 2),     // partial acceptance
        (&[1, 1, 1, 1, 1, 1, 1, 1][..], 2),     // zero acceptance
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
    let root = package("1", false, &[0, 0, 0, 0, 0, 0, 0, 0])?;
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

/// With a one-token output budget the verifier-owned anchor is the complete
/// committed path. The reference samples that target distribution directly,
/// independently of the DFlash implementation.
fn sampling_reference(seed: u64) -> u32 {
    let mut driver_rng = StdRng::seed_from_u64(seed);
    (driver_rng.random::<f32>() * 4.0) as u32
}

#[test]
fn dispatched_dflash_sampling_matches_the_declared_rejection_reference() -> anyhow::Result<()> {
    let root = package("1", true, &[0, 0, 0, 0, 0, 0, 0, 0])?;
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
    let root = package("1", false, &[0, 0, 0, 0, 0, 0, 0, 0])?;
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
    let root = package("2", false, &[0, 0, 0, 0, 0, 0, 0, 0])?;
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

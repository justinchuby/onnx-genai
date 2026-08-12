use std::fs;
use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    Engine, EngineConfig, GeneratePrompt, GenerateRequest, PipelineGenerateRequest,
};
use onnx_genai_ort::Value;

fn package(name: &str, metadata: &str, models: &[(&str, &str)]) -> anyhow::Result<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/test-fixtures/workflow-policy")
        .join(name);
    fs::create_dir_all(&root)?;
    fs::write(root.join("inference_metadata.yaml"), metadata)?;
    for (name, model) in models {
        fs::write(root.join(name), model)?;
    }
    Ok(root)
}

const GREEDY: &str = r#"
ir_version: 8
graph {
  node {
    input: "logits"
    output: "token_ids"
    op_type: "ArgMax"
    attribute { name: "axis" i: -1 type: 2 }
    attribute { name: "keepdims" i: 0 type: 2 }
  }
  name: "greedy_sampler"
  input {
    name: "logits"
    type { tensor_type { elem_type: 1 shape {
      dim { dim_param: "batch" } dim { dim_param: "vocabulary" }
    }}}
  }
  output {
    name: "token_ids"
    type { tensor_type { elem_type: 7 shape { dim { dim_param: "batch" } }}}
  }
}
opset_import { domain: "" version: 12 }
"#;

const EULER: &str = r#"
ir_version: 8
graph {
  node { input: "sample" input: "derivative" output: "next_state" op_type: "Sub" }
  name: "euler"
  input { name: "sample" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_param: "width" }
  }}}}
  input { name: "derivative" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_param: "width" }
  }}}}
  input { name: "step" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
  input { name: "schedule" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "schedule_length" }
  }}}}
  output { name: "next_state" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_param: "width" }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

const MASKED_UPDATE: &str = r#"
ir_version: 8
graph {
  node { input: "masked" input: "proposed_tokens" input: "current_tokens"
         output: "next_state" op_type: "Where" }
  node { input: "masked" output: "next_mask" op_type: "Identity" }
  name: "masked_update"
  input { name: "current_tokens" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" } dim { dim_param: "sequence" }
  }}}}
  input { name: "proposed_tokens" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" } dim { dim_param: "sequence" }
  }}}}
  input { name: "masked" type { tensor_type { elem_type: 9 shape {
    dim { dim_param: "batch" } dim { dim_param: "sequence" }
  }}}}
  input { name: "step" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
  output { name: "next_state" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" } dim { dim_param: "sequence" }
  }}}}
  output { name: "next_mask" type { tensor_type { elem_type: 9 shape {
    dim { dim_param: "batch" } dim { dim_param: "sequence" }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

const SPECULATIVE_ACCEPTANCE: &str = r#"
ir_version: 8
graph {
  node { input: "target_scores" output: "target_tokens" op_type: "ArgMax"
         attribute { name: "axis" i: -1 type: 2 }
         attribute { name: "keepdims" i: 0 type: 2 } }
  node { input: "target_tokens" input: "proposed_tokens" output: "accepted" op_type: "Equal" }
  node { input: "accepted" output: "accepted_i64" op_type: "Cast"
         attribute { name: "to" i: 7 type: 2 } }
  node { input: "accepted_i64" output: "accepted_count" op_type: "ReduceSum"
         attribute { name: "axes" ints: -1 type: 7 }
         attribute { name: "keepdims" i: 0 type: 2 } }
  node { input: "proposed_tokens" output: "accepted_tokens" op_type: "Identity" }
  node { input: "accepted_count" input: "zero" output: "done" op_type: "Greater" }
  name: "speculative_acceptance"
  initializer { dims: 1 data_type: 7 name: "zero"
                raw_data: "\000\000\000\000\000\000\000\000" }
  input { name: "target_scores" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_param: "draft_sequence" }
    dim { dim_param: "vocabulary" }
  }}}}
  input { name: "proposed_tokens" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" } dim { dim_param: "draft_sequence" }
  }}}}
  output { name: "accepted_tokens" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" } dim { dim_param: "draft_sequence" }
  }}}}
  output { name: "accepted_count" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
  output { name: "done" type { tensor_type { elem_type: 9 shape {
    dim { dim_param: "batch" }
  }}}}
}
opset_import { domain: "" version: 12 }
"#;

const ADD_STATE: &str = r#"
ir_version: 8
graph {
  node { input: "current" input: "update" output: "next" op_type: "Add" }
  name: "add_state"
  input { name: "current" type { tensor_type { elem_type: 7 shape {} }}}
  input { name: "update" type { tensor_type { elem_type: 7 shape {} }}}
  output { name: "next" type { tensor_type { elem_type: 7 shape {} }}}
}
opset_import { domain: "" version: 13 }
"#;

const LESS: &str = r#"
ir_version: 8
graph {
  node { input: "value" input: "limit" output: "continue" op_type: "Less" }
  name: "less"
  input { name: "value" type { tensor_type { elem_type: 7 shape {} }}}
  input { name: "limit" type { tensor_type { elem_type: 7 shape {} }}}
  output { name: "continue" type { tensor_type { elem_type: 9 shape {} }}}
}
opset_import { domain: "" version: 13 }
"#;

const SEEDED_SAMPLER: &str = r#"
ir_version: 8
graph {
  node {
    input: "logits" output: "token_ids" op_type: "ArgMax"
    attribute { name: "axis" i: -1 type: 2 }
    attribute { name: "keepdims" i: 0 type: 2 }
  }
  node { input: "offset" input: "one" output: "next_offset" op_type: "Add" }
  name: "seeded_sampler"
  initializer { dims: 1 data_type: 7 name: "one"
                raw_data: "\001\000\000\000\000\000\000\000" }
  input { name: "logits" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_param: "vocabulary" }
  }}}}
  input { name: "seed" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
  input { name: "offset" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
  output { name: "token_ids" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
  output { name: "next_offset" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

const EOS: &str = r#"
ir_version: 8
graph {
  node { input: "token_ids" input: "eos_token_ids" output: "terminated" op_type: "Equal" }
  name: "eos"
  input { name: "token_ids" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
  input { name: "eos_token_ids" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
  input { name: "iteration" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
  input { name: "max_iterations" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
  output { name: "terminated" type { tensor_type { elem_type: 9 shape {
    dim { dim_param: "batch" }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

const NOT: &str = r#"
ir_version: 8
graph {
  node { input: "done" output: "continue" op_type: "Not" }
  name: "not"
  input { name: "done" type { tensor_type { elem_type: 9 shape {
    dim { dim_param: "batch" }
  }}}}
  output { name: "continue" type { tensor_type { elem_type: 9 shape {
    dim { dim_param: "batch" }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

#[test]
fn workflow_executes_real_greedy_policy_artifact() -> anyhow::Result<()> {
    let metadata = r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 13 }
      adapter_abis: {}
      custom_op_versions: {}
      capabilities: [workflow_ssa, linear_effects, typed_emit]
    inputs:
      logits:
        contract: { dtype: float32, rank: 2, shape: [batch, vocabulary] }
        role: { kind: opaque }
        source: { kind: application, name: logits }
        required: true
    outputs:
      token:
        contract: { dtype: int64, rank: 1, shape: [batch] }
        role: tokens
        stage: pre_adapter
    components:
      sampler:
        implementation: { kind: onnx, artifact: sampler.onnx.textproto }
        ports:
          inputs:
            logits: { dtype: float32, rank: 2, shape: [batch, vocabulary] }
          outputs:
            token_ids: { dtype: int64, rank: 1, shape: [batch] }
        policy:
          role: token_sampler
          mode: greedy
          logits: logits
          token: token_ids
          effect: sample
        effects: [sample]
    initial_effects: { sample: sample.0, stream: stream.0 }
    graph:
      kind: sequence
      nodes:
        - kind: invoke
          component: sampler
          inputs: { logits: logits }
          outputs: { token_ids: sampled }
          effects: { sample: { consumes: sample.0, produces: sample.1 } }
        - kind: emit
          value: sampled
          output: token
          mode: replace
          effect_name: stream
          effect: { consumes: stream.0, produces: stream.1 }
"#;
    let root = package("greedy", metadata, &[("sampler.onnx.textproto", GREEDY)])?;
    let mut engine = Engine::from_pipeline_dir(&root, EngineConfig::default())?;
    let request =
        PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![])))
            .with_input(
                "logits",
                Value::from_slice_f32(&[0.1, 0.7, 0.2, 2.0, 1.0, 3.0], &[2, 3])?,
            );
    let output = engine.run_pipeline(request)?;
    assert_eq!(output["token"].to_vec_i64()?, [1, 2]);
    Ok(())
}

#[test]
fn workflow_executes_autoregressive_rng_and_eos_loop() -> anyhow::Result<()> {
    let metadata = r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 13 }
      adapter_abis: {}
      custom_op_versions: {}
      capabilities: [workflow_ssa, linear_effects, typed_emit, nested_control_flow]
    inputs:
      logits: { contract: { dtype: float32, rank: 2, shape: [batch, vocabulary] },
                role: { kind: opaque }, source: { kind: application, name: logits }, required: true }
      seed:
        contract: { dtype: int64, rank: 1, shape: [batch] }
        role: { kind: runtime, version: v1, role: seed }
        source: { kind: request, field: seed }
        required: true
      offset: { contract: { dtype: int64, rank: 1, shape: [batch] },
                role: { kind: opaque }, source: { kind: application, name: offset }, required: true }
      eos: { contract: { dtype: int64, rank: 1, shape: [batch] },
             role: { kind: opaque }, source: { kind: application, name: eos }, required: true }
      iteration: { contract: { dtype: int64, rank: 1, shape: [batch] },
                   role: { kind: opaque }, source: { kind: application, name: iteration }, required: true }
      max_iterations: { contract: { dtype: int64, rank: 1, shape: [batch] },
                        role: { kind: opaque }, source: { kind: application, name: max_iterations }, required: true }
      iterations:
        contract: { dtype: int64, rank: 0, shape: [] }
        role: { kind: runtime, version: v1, role: max_output_tokens }
        source: { kind: request, field: max_output_tokens }
        required: true
    outputs:
      tokens: { contract: { dtype: int64, rank: 1, shape: [generated] },
                role: tokens, stage: pre_adapter }
    components:
      binding:
        implementation: { kind: binding }
        ports:
          inputs: { value: { dtype: int64, rank: 1, shape: [batch] } }
          outputs: { value: { dtype: int64, rank: 1, shape: [batch] } }
      sampler:
        implementation: { kind: onnx, artifact: sampler.onnx.textproto }
        ports:
          inputs:
            logits: { dtype: float32, rank: 2, shape: [batch, vocabulary] }
            seed: { dtype: int64, rank: 1, shape: [batch] }
            offset: { dtype: int64, rank: 1, shape: [batch] }
          outputs:
            token_ids: { dtype: int64, rank: 1, shape: [batch] }
            next_offset: { dtype: int64, rank: 1, shape: [batch] }
        policy:
          role: token_sampler
          mode: seeded_stochastic
          logits: logits
          token: token_ids
          rng: { seed: seed, offset: offset, next_offset: next_offset }
          effect: rng
        effects: [rng]
      termination:
        implementation: { kind: onnx, artifact: eos.onnx.textproto }
        ports:
          inputs:
            token_ids: { dtype: int64, rank: 1, shape: [batch] }
            eos_token_ids: { dtype: int64, rank: 1, shape: [batch] }
            iteration: { dtype: int64, rank: 1, shape: [batch] }
            max_iterations: { dtype: int64, rank: 1, shape: [batch] }
          outputs:
            terminated: { dtype: bool, rank: 1, shape: [batch] }
        policy:
          role: termination_predicate
          tokens: token_ids
          eos_ids: eos_token_ids
          iteration: iteration
          max_iterations: max_iterations
          done: terminated
          effect: termination
        effects: [termination]
      invert:
        implementation: { kind: onnx, artifact: not.onnx.textproto }
        ports:
          inputs: { done: { dtype: bool, rank: 1, shape: [batch] } }
          outputs: { continue: { dtype: bool, rank: 1, shape: [batch] } }
    state:
      rng:
        contract: { dtype: int64, rank: 1, shape: [batch] }
        scope: invocation
        initializer: offset
        recurrence: { kind: invariant }
    initial_effects:
      rng: rng.0
      termination: termination.0
      stream: stream.0
      state:rng: state.0
    graph:
      kind: loop
      setup:
        kind: invoke
        component: binding
        inputs: { value: offset }
        outputs: { value: rng.current }
        effects: {}
      body:
        kind: sequence
        nodes:
          - kind: invoke
            component: sampler
            inputs: { logits: logits, seed: seed, offset: rng.body }
            outputs: { token_ids: sampled, next_offset: rng.body_next }
            effects: { rng: { consumes: rng.0, produces: rng.1 } }
          - kind: invoke
            component: termination
            inputs: { token_ids: sampled, eos_token_ids: eos,
                      iteration: iteration, max_iterations: max_iterations }
            outputs: { terminated: done }
            effects:
              termination: { consumes: termination.0, produces: termination.1 }
          - kind: invoke
            component: invert
            inputs: { done: done }
            outputs: { continue: loop.continue }
            effects: {}
          - kind: emit
            value: sampled
            output: tokens
            mode: append
            effect_name: stream
            effect: { consumes: stream.0, produces: stream.1 }
      condition: loop.continue
      max_iterations: iterations
      carried:
        - cell: rng
          current: rng.current
          body_input: rng.body
          body_output: rng.body_next
          next: rng.final
          read_effect: { consumes: state.0, produces: state.1 }
          write_effect: { consumes: state.1, produces: state.2 }
"#;
    let root = package(
        "autoregressive",
        metadata,
        &[
            ("sampler.onnx.textproto", SEEDED_SAMPLER),
            ("eos.onnx.textproto", EOS),
            ("not.onnx.textproto", NOT),
        ],
    )?;
    let mut engine = Engine::from_pipeline_dir(&root, EngineConfig::default())?;
    let mut options = onnx_genai_engine::GenerateOptions::default();
    options.max_new_tokens = 3;
    options.seed = Some(7);
    let request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![]),
        options,
    })
    .with_input("logits", Value::from_slice_f32(&[0.1, 0.9, 0.0], &[1, 3])?)
    .with_input("offset", Value::from_slice_i64(&[0], &[1])?)
    .with_input("eos", Value::from_slice_i64(&[2], &[1])?)
    .with_input("iteration", Value::from_slice_i64(&[0], &[1])?)
    .with_input("max_iterations", Value::from_slice_i64(&[10], &[1])?);
    let output = engine.run_pipeline(request)?;
    assert_eq!(output["tokens"].to_vec_i64()?, [1, 1, 1]);
    assert_eq!(output["rng.final"].to_vec_i64()?, [3]);
    Ok(())
}

#[test]
fn workflow_executes_diffusion_solver_policy_artifact() -> anyhow::Result<()> {
    let metadata = r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 13 }
      adapter_abis: {}
      custom_op_versions: {}
      capabilities: [workflow_ssa, linear_effects, typed_emit]
    inputs:
      sample: { contract: { dtype: float32, rank: 2, shape: [batch, width] },
                role: { kind: opaque }, source: { kind: application, name: sample }, required: true }
      derivative: { contract: { dtype: float32, rank: 2, shape: [batch, width] },
                    role: { kind: opaque }, source: { kind: application, name: derivative }, required: true }
      step: { contract: { dtype: int64, rank: 1, shape: [batch] },
              role: { kind: opaque }, source: { kind: application, name: step }, required: true }
      schedule: { contract: { dtype: float32, rank: 1, shape: [schedule_length] },
                  role: { kind: opaque }, source: { kind: application, name: schedule }, required: true }
    outputs:
      latent: { contract: { dtype: float32, rank: 2, shape: [batch, width] },
                role: tensor, stage: pre_adapter }
    components:
      solver:
        implementation: { kind: onnx, artifact: solver.onnx.textproto }
        ports:
          inputs:
            sample: { dtype: float32, rank: 2, shape: [batch, width] }
            derivative: { dtype: float32, rank: 2, shape: [batch, width] }
            step: { dtype: int64, rank: 1, shape: [batch] }
            schedule: { dtype: float32, rank: 1, shape: [schedule_length] }
          outputs:
            next_state: { dtype: float32, rank: 2, shape: [batch, width] }
        policy:
          role: solver_step
          state: sample
          estimate: derivative
          step: step
          schedule: schedule
          next_state: next_state
          effect: solver
        effects: [solver]
    initial_effects: { solver: solver.0, stream: stream.0 }
    graph:
      kind: sequence
      nodes:
        - kind: invoke
          component: solver
          inputs: { sample: sample, derivative: derivative, step: step, schedule: schedule }
          outputs: { next_state: latent.next }
          effects: { solver: { consumes: solver.0, produces: solver.1 } }
        - kind: emit
          value: latent.next
          output: latent
          mode: replace
          effect_name: stream
          effect: { consumes: stream.0, produces: stream.1 }
"#;
    let root = package("solver", metadata, &[("solver.onnx.textproto", EULER)])?;
    let mut engine = Engine::from_pipeline_dir(&root, EngineConfig::default())?;
    let request =
        PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![])))
            .with_input("sample", Value::from_slice_f32(&[1.0, 2.0], &[1, 2])?)
            .with_input("derivative", Value::from_slice_f32(&[0.5, 0.25], &[1, 2])?)
            .with_input("step", Value::from_slice_i64(&[0], &[1])?)
            .with_input("schedule", Value::from_slice_f32(&[1.0, 0.0], &[2])?);
    let output = engine.run_pipeline(request)?;
    assert_eq!(output["latent"].to_vec_f32()?, [0.5, 1.75]);
    Ok(())
}

#[test]
fn workflow_executes_masked_and_speculative_policy_artifacts() -> anyhow::Result<()> {
    let masked_metadata = r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 13 }
      adapter_abis: {}
      custom_op_versions: {}
      capabilities: [workflow_ssa, linear_effects, typed_emit]
    inputs:
      current: { contract: { dtype: int64, rank: 2, shape: [batch, sequence] }, role: { kind: opaque },
                 source: { kind: application, name: current }, required: true }
      proposed: { contract: { dtype: int64, rank: 2, shape: [batch, sequence] }, role: { kind: opaque },
                  source: { kind: application, name: proposed }, required: true }
      mask: { contract: { dtype: bool, rank: 2, shape: [batch, sequence] }, role: { kind: opaque },
              source: { kind: application, name: mask }, required: true }
      step: { contract: { dtype: int64, rank: 1, shape: [batch] }, role: { kind: opaque },
              source: { kind: application, name: step }, required: true }
    outputs:
      tokens: { contract: { dtype: int64, rank: 2, shape: [batch, sequence] }, role: tokens, stage: pre_adapter }
    components:
      update:
        implementation: { kind: onnx, artifact: update.onnx.textproto }
        ports:
          inputs:
            current_tokens: { dtype: int64, rank: 2, shape: [batch, sequence] }
            proposed_tokens: { dtype: int64, rank: 2, shape: [batch, sequence] }
            masked: { dtype: bool, rank: 2, shape: [batch, sequence] }
            step: { dtype: int64, rank: 1, shape: [batch] }
          outputs:
            next_state: { dtype: int64, rank: 2, shape: [batch, sequence] }
            next_mask: { dtype: bool, rank: 2, shape: [batch, sequence] }
        policy:
          role: masked_update
          state: current_tokens
          proposal: proposed_tokens
          mask: masked
          step: step
          next_state: next_state
          next_mask: next_mask
          effect: update
        effects: [update]
    initial_effects: { update: update.0, stream: stream.0 }
    graph:
      kind: sequence
      nodes:
        - kind: invoke
          component: update
          inputs: { current_tokens: current, proposed_tokens: proposed, masked: mask, step: step }
          outputs: { next_state: updated, next_mask: remaining }
          effects: { update: { consumes: update.0, produces: update.1 } }
        - kind: emit
          value: updated
          output: tokens
          mode: replace
          effect_name: stream
          effect: { consumes: stream.0, produces: stream.1 }
"#;
    let root = package(
        "masked",
        masked_metadata,
        &[("update.onnx.textproto", MASKED_UPDATE)],
    )?;
    let mut engine = Engine::from_pipeline_dir(&root, EngineConfig::default())?;
    let request =
        PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![])))
            .with_input("current", Value::from_slice_i64(&[1, 99, 99], &[1, 3])?)
            .with_input("proposed", Value::from_slice_i64(&[4, 5, 6], &[1, 3])?)
            .with_input(
                "mask",
                Value::from_raw_bytes(vec![0, 1, 1], &[1, 3], onnx_genai_ort::DataType::Bool)?,
            )
            .with_input("step", Value::from_slice_i64(&[0], &[1])?);
    assert_eq!(
        engine.run_pipeline(request)?["tokens"].to_vec_i64()?,
        [1, 5, 6]
    );

    let speculative_metadata = r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 13 }
      adapter_abis: {}
      custom_op_versions: {}
      capabilities: [workflow_ssa, linear_effects, typed_emit]
    inputs:
      target: { contract: { dtype: float32, rank: 3, shape: [batch, draft, vocabulary] }, role: { kind: opaque },
                source: { kind: application, name: target }, required: true }
      proposed: { contract: { dtype: int64, rank: 2, shape: [batch, draft] }, role: { kind: opaque },
                  source: { kind: application, name: proposed }, required: true }
    outputs:
      accepted_len: { contract: { dtype: int64, rank: 1, shape: [batch] }, role: tensor, stage: pre_adapter }
    components:
      verifier:
        implementation: { kind: onnx, artifact: verifier.onnx.textproto }
        ports:
          inputs:
            target_scores: { dtype: float32, rank: 3, shape: [batch, draft, vocabulary] }
            proposed_tokens: { dtype: int64, rank: 2, shape: [batch, draft] }
          outputs:
            accepted_tokens: { dtype: int64, rank: 2, shape: [batch, draft] }
            accepted_count: { dtype: int64, rank: 1, shape: [batch] }
            done: { dtype: bool, rank: 1, shape: [batch] }
        policy:
          role: speculative_verifier
          target_scores: target_scores
          proposed_tokens: proposed_tokens
          accepted_tokens: accepted_tokens
          accepted_len: accepted_count
          done: done
          effect: verify
        effects: [verify]
    initial_effects: { verify: verify.0, stream: stream.0 }
    graph:
      kind: sequence
      nodes:
        - kind: invoke
          component: verifier
          inputs: { target_scores: target, proposed_tokens: proposed }
          outputs: { accepted_tokens: accepted, accepted_count: count, done: done }
          effects: { verify: { consumes: verify.0, produces: verify.1 } }
        - kind: emit
          value: count
          output: accepted_len
          mode: replace
          effect_name: stream
          effect: { consumes: stream.0, produces: stream.1 }
"#;
    let root = package(
        "speculative",
        speculative_metadata,
        &[("verifier.onnx.textproto", SPECULATIVE_ACCEPTANCE)],
    )?;
    let mut engine = Engine::from_pipeline_dir(&root, EngineConfig::default())?;
    let request =
        PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![])))
            .with_input(
                "target",
                Value::from_slice_f32(&[0.1, 0.9, 0.8, 0.2, 0.7, 0.3], &[1, 3, 2])?,
            )
            .with_input("proposed", Value::from_slice_i64(&[1, 1, 0], &[1, 3])?);
    assert_eq!(
        engine.run_pipeline(request)?["accepted_len"].to_vec_i64()?,
        [2]
    );
    Ok(())
}

#[test]
fn workflow_threads_loop_branch_effects_and_session_state() -> anyhow::Result<()> {
    let metadata = r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: { ai.onnx: 13 }
      adapter_abis: {}
      custom_op_versions: {}
      capabilities:
        [workflow_ssa, linear_effects, typed_emit, streaming_emit,
         nested_control_flow, session_state_lease]
    inputs:
      initial: { contract: { dtype: int64, rank: 0, shape: [] }, role: { kind: opaque },
                 source: { kind: application, name: initial }, required: true }
      increment: { contract: { dtype: int64, rank: 0, shape: [] }, role: { kind: opaque },
                   source: { kind: application, name: increment }, required: true }
      limit: { contract: { dtype: int64, rank: 0, shape: [] }, role: { kind: opaque },
               source: { kind: application, name: limit }, required: true }
      iterations:
        contract: { dtype: int64, rank: 0, shape: [] }
        role: { kind: runtime, version: v1, role: max_iterations }
        source: { kind: request, field: max_iterations }
        required: true
    outputs:
      state: { contract: { dtype: int64, rank: 0, shape: [] }, role: tensor, stage: pre_adapter }
      events: { contract: { dtype: int64, rank: 0, shape: [] }, role: event, stage: pre_adapter }
    components:
      binding:
        implementation: { kind: binding }
        ports:
          inputs: { value: { dtype: int64, rank: 0, shape: [] } }
          outputs: { value: { dtype: int64, rank: 0, shape: [] } }
      update:
        implementation: { kind: onnx, artifact: update.onnx.textproto }
        ports:
          inputs:
            current: { dtype: int64, rank: 0, shape: [] }
            update: { dtype: int64, rank: 0, shape: [] }
          outputs:
            next: { dtype: int64, rank: 0, shape: [] }
        policy:
          role: state_update
          current: current
          update: update
          next: next
          effect: update
        effects: [update]
      predicate:
        implementation: { kind: onnx, artifact: less.onnx.textproto }
        ports:
          inputs:
            value: { dtype: int64, rank: 0, shape: [] }
            limit: { dtype: int64, rank: 0, shape: [] }
          outputs:
            continue: { dtype: bool, rank: 0, shape: [] }
        effects: [predicate]
    state:
      world:
        contract: { dtype: int64, rank: 0, shape: [] }
        scope: session
        initializer: initial
        recurrence: { kind: invariant }
        session: { policy: exclusive }
    initial_effects:
      update: update.0
      predicate: predicate.0
      stream: stream.0
      state:world: state.0
    graph:
      kind: sequence
      nodes:
        - kind: loop
          setup:
            kind: invoke
            component: binding
            inputs: { value: initial }
            outputs: { value: world.current }
            effects: {}
          body:
            kind: sequence
            nodes:
              - kind: invoke
                component: update
                inputs: { current: world.body, update: increment }
                outputs: { next: world.body_next }
                effects: { update: { consumes: update.0, produces: update.1 } }
              - kind: invoke
                component: predicate
                inputs: { value: world.body_next, limit: limit }
                outputs: { continue: loop.continue }
                effects: { predicate: { consumes: predicate.0, produces: predicate.1 } }
              - kind: emit
                value: world.body_next
                output: events
                mode: event
                effect_name: stream
                effect: { consumes: stream.0, produces: stream.1 }
          condition: loop.continue
          max_iterations: iterations
          carried:
            - cell: world
              current: world.current
              body_input: world.body
              body_output: world.body_next
              next: world.final
              read_effect: { consumes: state.0, produces: state.1 }
              write_effect: { consumes: state.1, produces: state.2 }
        - kind: branch
          predicate: world.final
          cases:
            "3":
              kind: emit
              value: world.current
              output: state
              mode: replace
              effect_name: stream
              effect: { consumes: stream.1, produces: stream.2 }
            "5":
              kind: emit
              value: world.current
              output: state
              mode: replace
              effect_name: stream
              effect: { consumes: stream.1, produces: stream.2 }
          default:
            kind: emit
            value: world.current
            output: state
            mode: replace
            effect_name: stream
            effect: { consumes: stream.1, produces: stream.2 }
"#;
    let root = package(
        "world",
        metadata,
        &[
            ("update.onnx.textproto", ADD_STATE),
            ("less.onnx.textproto", LESS),
        ],
    )?;
    let mut engine = Engine::from_pipeline_dir(&root, EngineConfig::default())?;
    let mut first_options = onnx_genai_engine::GenerateOptions::default();
    first_options.max_new_tokens = 4;
    let first = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![]),
        options: first_options,
    })
    .with_session_id("world-a")
    .with_input("initial", Value::from_slice_i64(&[0], &[])?)
    .with_input("increment", Value::from_slice_i64(&[1], &[])?)
    .with_input("limit", Value::from_slice_i64(&[3], &[])?);
    assert_eq!(engine.run_pipeline(first)?["state"].to_vec_i64()?, [3]);

    let mut second_options = onnx_genai_engine::GenerateOptions::default();
    second_options.max_new_tokens = 4;
    let second = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![]),
        options: second_options,
    })
    .with_session_id("world-a")
    .with_input("initial", Value::from_slice_i64(&[0], &[])?)
    .with_input("increment", Value::from_slice_i64(&[1], &[])?)
    .with_input("limit", Value::from_slice_i64(&[5], &[])?);
    assert_eq!(engine.run_pipeline(second)?["state"].to_vec_i64()?, [5]);
    Ok(())
}

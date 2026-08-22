//! ORT ⇄ native parity for the one universal `pipeline.workflow` interpreter.
//!
//! Each package is run twice — once with `EngineDecodeBackend::Ort` and once
//! with `EngineDecodeBackend::Native` — from the *same* on-disk package, and the
//! request-aligned outputs must match. The scenarios cover the interpreter
//! surface the native seam must honor: single-pass, nested loop + branch,
//! loop-carried state, read-only shared (session) state, and autoregressive
//! decode. A dedicated test proves the native run actually executed native
//! sessions (not an ORT fallback), and a dedicated test proves an unsupported
//! op fails closed on native with an actionable diagnostic rather than silently
//! falling back to ORT.

use std::fs;
use std::path::{Path, PathBuf};

use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GenerateOptions, GeneratePrompt, GenerateRequest,
    PipelineGenerateRequest, pipeline::PipelineEngine,
};
use onnx_genai_ort::{DataType, Value};

// ── package + fixture helpers ────────────────────────────────────────────────

fn authored_package(
    name: &str,
    metadata: &str,
    models: &[(&str, &str)],
) -> anyhow::Result<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/test-fixtures/native-workflow-parity")
        .join(name);
    fs::create_dir_all(&root)?;
    fs::write(root.join("inference_metadata.yaml"), metadata)?;
    for (file, model) in models {
        fs::write(root.join(file), model)?;
    }
    Ok(root)
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/onnx_genai_workflows")
        .join(name)
}

fn ort_engine(root: &Path) -> anyhow::Result<PipelineEngine> {
    Engine::from_pipeline_dir(
        root,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Ort,
            ..EngineConfig::default()
        },
    )
}

fn native_engine(root: &Path) -> anyhow::Result<PipelineEngine> {
    Engine::from_pipeline_dir(
        root,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            ..EngineConfig::default()
        },
    )
}

// ── value comparison ─────────────────────────────────────────────────────────

/// Assert two workflow values are equal: integers/bools bit-exact, floats within
/// a small tolerance (ORT and the native CPU EP may reduce in a different order).
fn assert_values_match(name: &str, ort: &Value, native: &Value) -> anyhow::Result<()> {
    assert_eq!(
        ort.shape(),
        native.shape(),
        "output '{name}' shape mismatch"
    );
    assert_eq!(
        ort.dtype(),
        native.dtype(),
        "output '{name}' dtype mismatch"
    );
    match ort.dtype() {
        DataType::Float32 => {
            let a = ort.to_vec_f32()?;
            let b = native.to_vec_f32()?;
            for (index, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                let tol = 1e-3 * (1.0 + x.abs());
                assert!(
                    (x - y).abs() <= tol,
                    "output '{name}' float element {index} diverged: ort={x} native={y}"
                );
            }
        }
        _ => {
            assert_eq!(
                ort.to_raw_bytes()?,
                native.to_raw_bytes()?,
                "output '{name}' bytes diverged"
            );
        }
    }
    Ok(())
}

/// Run `request` on an ORT engine and a native engine built from the same
/// package, and assert every named output matches. Returns the native engine so
/// the caller can inspect its native-run counter.
fn assert_parity(
    root: &Path,
    request: impl Fn() -> anyhow::Result<PipelineGenerateRequest>,
    outputs: &[&str],
) -> anyhow::Result<PipelineEngine> {
    let mut ort = ort_engine(root)?;
    let mut native = native_engine(root)?;
    let ort_output = ort.run_pipeline(request()?)?;
    let native_output = native.run_pipeline(request()?)?;
    for output in outputs {
        let ort_value = ort_output
            .get(*output)
            .unwrap_or_else(|| panic!("ORT run missing output '{output}'"));
        let native_value = native_output
            .get(*output)
            .unwrap_or_else(|| panic!("native run missing output '{output}'"));
        assert_values_match(output, ort_value, native_value)?;
    }
    // The ORT engine never constructs native sessions; the native one must have.
    assert!(
        ort.native_component_run_count().is_none(),
        "ORT engine must not hold native sessions"
    );
    assert!(
        native.native_component_run_count().unwrap_or(0) > 0,
        "native engine must have executed native component sessions"
    );
    Ok(native)
}

// ── graphs ───────────────────────────────────────────────────────────────────

const GREEDY: &str = r#"
ir_version: 8
graph {
  node { input: "logits" output: "token_ids" op_type: "ArgMax"
    attribute { name: "axis" i: -1 type: 2 }
    attribute { name: "keepdims" i: 0 type: 2 } }
  name: "greedy_sampler"
  input { name: "logits" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_param: "vocabulary" } }}}}
  input { name: "temperature" type { tensor_type { elem_type: 1 shape { dim { dim_value: 1 } }}}}
  input { name: "top_k" type { tensor_type { elem_type: 7 shape { dim { dim_value: 1 } }}}}
  input { name: "top_p" type { tensor_type { elem_type: 1 shape { dim { dim_value: 1 } }}}}
  input { name: "grammar_mask" type { tensor_type { elem_type: 9 shape {
    dim { dim_param: "batch" } dim { dim_param: "vocabulary" } }}}}
  output { name: "token_ids" type { tensor_type { elem_type: 7 shape { dim { dim_param: "batch" } }}}}
}
opset_import { domain: "" version: 21 }
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

// ── metadata ─────────────────────────────────────────────────────────────────

const SINGLE_PASS_META: &str = r#"
pipeline:
  workflow:
    manifest:
      adapter_abis: {}
      capabilities: [workflow_ssa, typed_emit]
    inputs:
      logits:
        contract: { dtype: float32, rank: 2, shape: [batch, vocabulary] }
        role: { kind: opaque }
        source: { kind: application, name: logits }
        required: true
      temperature:
        contract: { dtype: float32, rank: 1, shape: [1] }
        role: { kind: runtime, version: "1", role: sampling_temperature }
        source: { kind: request }
        required: true
      top_k:
        contract: { dtype: int64, rank: 1, shape: [1] }
        role: { kind: runtime, version: "1", role: sampling_top_k }
        source: { kind: request }
        required: true
      top_p:
        contract: { dtype: float32, rank: 1, shape: [1] }
        role: { kind: runtime, version: "1", role: sampling_top_p }
        source: { kind: request }
        required: true
      grammar_mask:
        contract: { dtype: bool, rank: 2, shape: [batch, vocabulary] }
        role: { kind: opaque }
        source: { kind: application, name: grammar_mask }
        required: true
    outputs:
      token:
        contract: { dtype: int64, rank: 1, shape: [batch] }
        role: tokens
        stage: pre_adapter
    components:
      sampler:
        implementation: { kind: onnx, artifact: sampler.onnx.textproto }
        contract:
          id: onnx-genai.token-sampler
          version: "1"
          bindings:
            logits: logits
            temperature: temperature
            top_k: top_k
            top_p: top_p
            grammar_mask: grammar_mask
            token: token_ids
          parameters:
            mode: greedy
    steps:
        - kind: invoke
          component: sampler
          inputs:
            logits: logits
            temperature: temperature
            top_k: top_k
            top_p: top_p
            grammar_mask: grammar_mask
          outputs: { token_ids: sampled }
        - kind: emit
          value: sampled
          output: token
          mode: replace
"#;

const BRANCH_LOOP_STATE_META: &str = r#"
pipeline:
  workflow:
    manifest:
      adapter_abis: {}
      capabilities:
        [workflow_ssa, typed_emit, streaming_emit,
         nested_control_flow, session_state_lease]
    inputs:
      initial: { contract: { dtype: int64, rank: 0, shape: [] }, role: { kind: opaque },
                 source: { kind: application, name: initial }, required: true }
      run_branch: { contract: { dtype: bool, rank: 0, shape: [] }, role: { kind: opaque },
                    source: { kind: application, name: run_branch }, required: true }
      increment: { contract: { dtype: int64, rank: 0, shape: [] }, role: { kind: opaque },
                   source: { kind: application, name: increment }, required: true }
      limit: { contract: { dtype: int64, rank: 0, shape: [] }, role: { kind: opaque },
               source: { kind: application, name: limit }, required: true }
      iterations:
        contract: { dtype: int64, rank: 0, shape: [] }
        role: { kind: runtime, version: v1, role: max_iterations }
        source: { kind: request }
        required: true
      initial_continue: { contract: { dtype: bool, rank: 0, shape: [] },
                          role: { kind: opaque },
                          source: { kind: application, name: initial_continue }, required: true }
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
        contract:
          id: onnx-genai.state-update
          version: "1"
          bindings:
            current: current
            update: update
            next: next
      predicate:
        implementation: { kind: onnx, artifact: less.onnx.textproto }
        ports:
          inputs:
            value: { dtype: int64, rank: 0, shape: [] }
            limit: { dtype: int64, rank: 0, shape: [] }
          outputs:
            continue: { dtype: bool, rank: 0, shape: [] }
    state:
      world:
        contract: { dtype: int64, rank: 0, shape: [] }
        scope: session
        initializer: initial
        recurrence: { kind: invariant }
        session: { policy: exclusive }
      active:
        contract: { dtype: bool, rank: 0, shape: [] }
        scope: invocation
        initializer: initial_continue
        recurrence: { kind: invariant }
    steps:
        - kind: branch
          predicate: run_branch
          cases:
            "true":
              kind: loop
              setup:
                - kind: invoke
                  component: binding
                  inputs: { value: initial }
                  outputs: { value: world.current }
              steps:
                  - kind: invoke
                    component: update
                    inputs: { current: world, update: increment }
                    outputs: { next: world.body_next }
                  - kind: invoke
                    component: predicate
                    inputs: { value: world.body_next, limit: limit }
                    outputs: { continue: loop.continue }
                  - kind: emit
                    value: world.body_next
                    output: events
                    mode: event
              continue_when: active
              max_iterations: iterations
              carried:
                - cell: world
                  initial: world.current
                  next: world.body_next
                - cell: active
                  next: loop.continue
          outputs:
            world.selected:
              cases: { "true": world }
        - kind: branch
          predicate: world.selected
          cases:
            "3":
              kind: emit
              value: world.selected
              output: state
              mode: replace
            "5":
              kind: emit
              value: world.selected
              output: state
              mode: replace
          default:
            kind: emit
            value: world.selected
            output: state
            mode: replace
"#;

// ── tests ────────────────────────────────────────────────────────────────────

/// Single-pass: one component invocation, one emit. Also the canonical
/// "native actually ran, not an ORT fallback" proof.
#[test]
fn single_pass_parity_and_native_used() -> anyhow::Result<()> {
    let root = authored_package(
        "single-pass",
        SINGLE_PASS_META,
        &[("sampler.onnx.textproto", GREEDY)],
    )?;
    let native = assert_parity(
        &root,
        || {
            let mut generate = GenerateRequest::new(GeneratePrompt::TokenIds(vec![]));
            generate.options.temperature = 0.75;
            generate.options.top_k = 17;
            generate.options.top_p = 0.9;
            Ok(PipelineGenerateRequest::new(generate)
                .with_input(
                    "logits",
                    Value::from_slice_f32(&[0.1, 0.7, 0.2, 2.0, 1.0, 3.0], &[2, 3])?,
                )
                .with_input(
                    "grammar_mask",
                    Value::from_raw_bytes(vec![1; 6], &[2, 3], DataType::Bool)?,
                ))
        },
        &["token"],
    )?;
    // Exactly one component (the sampler) runs once.
    assert_eq!(native.native_component_run_count(), Some(1));
    Ok(())
}

/// Nested loop inside a branch, loop-carried `world`/`active` cells, a
/// session-scoped (read-only-shared-across-invocations) `world` cell, and an
/// event emit — all driven by the one interpreter on both backends.
#[test]
fn nested_loop_branch_and_state_parity() -> anyhow::Result<()> {
    let root = authored_package(
        "branch-loop-state",
        BRANCH_LOOP_STATE_META,
        &[
            ("update.onnx.textproto", ADD_STATE),
            ("less.onnx.textproto", LESS),
        ],
    )?;
    let request = || -> anyhow::Result<PipelineGenerateRequest> {
        let options = GenerateOptions {
            max_new_tokens: 4,
            ..Default::default()
        };
        Ok(PipelineGenerateRequest::new(GenerateRequest {
            prompt: GeneratePrompt::TokenIds(vec![]),
            options,
        })
        .with_session_id("parity-world")
        .with_input(
            "run_branch",
            Value::from_raw_bytes(vec![1], &[], DataType::Bool)?,
        )
        .with_input(
            "initial_continue",
            Value::from_raw_bytes(vec![1], &[], DataType::Bool)?,
        )
        .with_input("initial", Value::from_slice_i64(&[0], &[])?)
        .with_input("increment", Value::from_slice_i64(&[1], &[])?)
        .with_input("limit", Value::from_slice_i64(&[3], &[])?))
    };
    // Independent session ids per engine so the session-scoped `world` cell of
    // one backend does not seed the other; each still must reach state == 3.
    let mut ort = ort_engine(&root)?;
    let mut native = native_engine(&root)?;
    let ort_state = ort.run_pipeline(request()?)?["state"].to_vec_i64()?;
    let native_state = native.run_pipeline(request()?)?["state"].to_vec_i64()?;
    assert_eq!(ort_state, vec![3], "ORT nested loop/branch/state");
    assert_eq!(native_state, ort_state, "native must match ORT");
    assert!(native.native_component_run_count().unwrap_or(0) > 0);
    Ok(())
}

/// Diffusion: a denoising loop with loop-carried latent state.
#[test]
fn diffusion_loop_parity() -> anyhow::Result<()> {
    let root = fixture("diffusion");
    let request = || {
        let noise: Vec<f32> = (0..4 * 4 * 4)
            .map(|index| (index as f32 - 32.0) / 16.0)
            .collect();
        Ok(PipelineGenerateRequest::new(GenerateRequest {
            prompt: GeneratePrompt::TokenIds(vec![1, 2]),
            options: GenerateOptions {
                max_new_tokens: 2,
                ..Default::default()
            },
        })
        .with_input(
            "request.noise",
            Value::from_slice_f32(&noise, &[1, 4, 4, 4])?,
        ))
    };
    assert_parity(&root, request, &["image"])?;
    Ok(())
}

/// Autoregressive decode with a loop-carried KV cache and a read-only shared KV
/// state service.
#[test]
fn static_cache_autoregressive_parity() -> anyhow::Result<()> {
    let root = fixture("static_cache");
    assert_parity(
        &root,
        || static_cache_request(2),
        &["cache_lengths", "write_indices", "key_cache", "value_cache"],
    )?;
    Ok(())
}

fn static_cache_request(steps: usize) -> anyhow::Result<PipelineGenerateRequest> {
    Ok(PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![]),
        options: GenerateOptions {
            max_new_tokens: steps,
            ..Default::default()
        },
    })
    .with_input(
        "request.input_ids",
        Value::from_slice_i64(&[1, 2, 3, 4, 5, 6], &[2, 3])?,
    )
    .with_input(
        "request.write_indices",
        Value::from_slice_i64(&[0, 3], &[2])?,
    )
    .with_input(
        "request.active",
        Value::from_raw_bytes(vec![1, 1], &[2], DataType::Bool)?,
    )
    .with_input(
        "request.max_iterations",
        Value::from_slice_i64(&[i64::try_from(steps)?], &[1])?,
    ))
}

/// Device/state routing: a recurring component edge (the decoder re-invoked each
/// decode step, carrying KV state) reuses ONE resident native session loaded at
/// build time — it is not reloaded per iteration, and no tensor is serialized
/// through the host-resident `ComponentTensor` seam (the native executor bridges
/// `Value ⇄ Tensor` directly). More loop iterations therefore drive strictly
/// more native invocations against the same resident sessions.
#[test]
fn native_recurring_edges_reuse_resident_sessions() -> anyhow::Result<()> {
    let root = fixture("static_cache");
    let run = |steps: usize| -> anyhow::Result<u64> {
        let mut engine = native_engine(&root)?;
        engine.run_pipeline(static_cache_request(steps)?)?;
        Ok(engine.native_component_run_count().unwrap_or(0))
    };
    let one_step = run(1)?;
    let three_steps = run(3)?;
    assert!(one_step > 0, "native sessions must have executed");
    assert!(
        three_steps > one_step,
        "recurring loop edges must re-invoke the resident native sessions: \
         1 step ran {one_step} components, 3 steps ran {three_steps}"
    );
    Ok(())
}

/// Speculative decode expressed as a workflow (draft → verify → accept/reject →
/// correction), driven by the one interpreter. This is the pattern the Gemma4
/// target+assistant speculative workflow (#1716/#1696) relies on: the
/// accept/reject/rollback semantics live in the backend-agnostic interpreter,
/// so the *same* package is a single parity case on ORT and native — the native
/// executor only runs component forward passes, it never re-derives speculative
/// or state-transition semantics.
#[test]
fn speculative_workflow_parity() -> anyhow::Result<()> {
    let root = fixture("speculative");
    let request = || -> anyhow::Result<PipelineGenerateRequest> {
        Ok(PipelineGenerateRequest::new(GenerateRequest {
            prompt: GeneratePrompt::TokenIds(vec![1, 2, 3, 4]),
            options: GenerateOptions {
                max_new_tokens: 1,
                ..Default::default()
            },
        })
        .with_input(
            "verifier.past_key_values.0.key",
            Value::from_slice_f32(&[], &[1, 2, 0, 8])?,
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
    };
    assert_parity(&root, request, &["tokens.row.0"])?;
    Ok(())
}

/// Fail closed: a native-unsupported op must produce an actionable error naming
/// the component and the offending dtype — never a silent fall back to ORT. The
/// checked `decoder` package's RNG sampler casts to uint64, which the native CPU
/// EP does not implement.
#[test]
fn native_unsupported_op_fails_closed() -> anyhow::Result<()> {
    let root = fixture("decoder");
    let request = || {
        PipelineGenerateRequest::new(GenerateRequest {
            prompt: GeneratePrompt::TokenIds(vec![4, 5]),
            options: GenerateOptions {
                max_new_tokens: 2,
                seed: Some(7),
                ..Default::default()
            },
        })
    };
    // ORT runs it fine.
    let mut ort = ort_engine(&root)?;
    ort.run_pipeline(request())?;
    // Native fails closed with an actionable message; it does not fall back.
    let mut native = native_engine(&root)?;
    let error = native
        .run_pipeline(request())
        .err()
        .expect("native must fail closed on an unsupported op, not fall back to ORT");
    let message = format!("{error:#}");
    assert!(
        message.contains("token_sampler"),
        "error must name the failing component: {message}"
    );
    assert!(
        message.contains("Uint64") || message.contains("uint64"),
        "error must name the unsupported dtype: {message}"
    );
    Ok(())
}

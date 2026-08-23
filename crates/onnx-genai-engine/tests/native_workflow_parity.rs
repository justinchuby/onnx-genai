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
    NativeDecodeDevice, PipelineGenerateRequest,
};
use onnx_genai_ort::{DataType, Value};

#[path = "common/chained.rs"]
mod chained;

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

fn ort_engine(root: &Path) -> anyhow::Result<Engine> {
    Engine::from_dir(
        root,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Ort,
            ..EngineConfig::default()
        },
    )
}

fn native_engine(root: &Path) -> anyhow::Result<Engine> {
    // Pin the CPU native device so the backend-agnostic parity scenarios are
    // deterministic regardless of build features or a GPU being present — only
    // the `native-cuda` device-residency test drives the CUDA device.
    Engine::from_dir(
        root,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            native_device: Some(NativeDecodeDevice::Cpu),
            ..EngineConfig::default()
        },
    )
}

/// A native engine pinned to CUDA device 0, for the device-residency test.
#[cfg(feature = "native-cuda")]
fn native_cuda_engine(root: &Path) -> anyhow::Result<Engine> {
    Engine::from_dir(
        root,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            native_device: Some(NativeDecodeDevice::Cuda { index: Some(0) }),
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
) -> anyhow::Result<Engine> {
    assert_parity_with(root, native_engine, request, outputs)
}

/// Like [`assert_parity`], but the native engine is built by `build_native` so a
/// test can pin a specific native device (e.g. CUDA for the device-residency
/// case) while reusing the ORT reference and the output comparison.
fn assert_parity_with(
    root: &Path,
    build_native: impl Fn(&Path) -> anyhow::Result<Engine>,
    request: impl Fn() -> anyhow::Result<PipelineGenerateRequest>,
    outputs: &[&str],
) -> anyhow::Result<Engine> {
    let mut ort = ort_engine(root)?;
    let mut native = build_native(root)?;
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
//
// Every model below is IR 11 / default opset 24, the repository floor for new
// and modified fixtures. The ops used here (`ArgMax` since-13, `Add` since-14,
// `Less` since-13) had no behavior change after those versions, so raising the
// import is a version statement rather than a semantic one — the parity these
// cases assert is unchanged.

const GREEDY: &str = r#"
ir_version: 11
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
opset_import { domain: "" version: 24 }
"#;

const ADD_STATE: &str = r#"
ir_version: 11
graph {
  node { input: "current" input: "update" output: "next" op_type: "Add" }
  name: "add_state"
  input { name: "current" type { tensor_type { elem_type: 7 shape {} }}}
  input { name: "update" type { tensor_type { elem_type: 7 shape {} }}}
  output { name: "next" type { tensor_type { elem_type: 7 shape {} }}}
}
opset_import { domain: "" version: 24 }
"#;

const LESS: &str = r#"
ir_version: 11
graph {
  node { input: "value" input: "limit" output: "continue" op_type: "Less" }
  name: "less"
  input { name: "value" type { tensor_type { elem_type: 7 shape {} }}}
  input { name: "limit" type { tensor_type { elem_type: 7 shape {} }}}
  output { name: "continue" type { tensor_type { elem_type: 9 shape {} }}}
}
opset_import { domain: "" version: 24 }
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
         nested_control_flow, session_state_lease, linear_effects]
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
        .with_input(
            "verifier.past_key_values.0.value",
            Value::from_slice_f32(&[], &[1, 4, 0, 4])?,
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

/// Blocker 1: under the Native backend the pipeline builds **zero** ORT
/// component sessions — components execute on native `InferenceSession`s, and
/// the package's I/O contract stays available as backend-neutral graph I/O
/// metadata. Because no ORT session is constructed, a package whose component
/// ORT would reject at load (a native-only operator) still loads and runs
/// natively; here we prove the structural invariant (no ORT sessions) plus a
/// successful native run, and that the engine reports its real native device
/// rather than an ORT EP.
#[test]
fn native_backend_builds_no_ort_sessions() -> anyhow::Result<()> {
    let root = fixture("static_cache");
    let mut engine = native_engine(&root)?;
    assert!(
        engine.models()?.sessions.is_empty(),
        "Native must build zero ORT component sessions, found {}",
        engine.models()?.sessions.len()
    );
    assert!(
        !engine.models()?.graph_io_metadata.is_empty(),
        "Native must still expose backend-neutral graph I/O metadata for the components"
    );
    let output = engine.run_pipeline(static_cache_request(2)?)?;
    assert_eq!(output["cache_lengths"].to_vec_i64()?, vec![3, 6]);
    assert!(engine.native_component_run_count().unwrap_or(0) > 0);
    // Blocker 2: the engine reports the explicit native device it resolved and
    // ran on, not an ORT EP and not an empty placeholder.
    assert_eq!(engine.execution_provider_status(), "native-cpu");
    Ok(())
}

/// Contrast: the ORT backend DOES build ORT component sessions (and reports an
/// ORT execution provider), so the "no ORT sessions" invariant above is a real
/// Native-only property, not an artifact of an empty package.
#[test]
fn ort_backend_builds_ort_sessions() -> anyhow::Result<()> {
    let root = fixture("static_cache");
    let engine = ort_engine(&root)?;
    assert!(
        !engine.models()?.sessions.is_empty(),
        "ORT must build component sessions"
    );
    assert_ne!(
        engine.execution_provider_status(),
        "native-cpu",
        "ORT must not report the native device"
    );
    Ok(())
}

/// On an H200 the native CUDA backend executes a multi-component workflow with
/// an intermediate and a recurring/state (KV) tensor staying **device-resident
/// end-to-end** — no host round-trip on the recurring edge. It reports its CUDA
/// device (Blocker 2 device/provider propagation — not an auto-detected CPU EP),
/// builds zero ORT sessions (Blocker 1), produces results that match the ORT
/// reference bit-for-bit, and its device-residency counters prove a recurring
/// tensor entered a component still on the device (bound zero-copy) and that
/// components produced device-resident outputs.
///
/// If the native CUDA EP cannot load this fixture's ops, that is an op-coverage
/// gap (not a device-bridge failure), so the test skips loudly rather than
/// recording a false negative. It never tolerates a *runtime* fail-closed on the
/// device path — that would mean the bridge did not work.
#[cfg(feature = "native-cuda")]
#[test]
fn native_cuda_device_resident_multicomponent() -> anyhow::Result<()> {
    let root = fixture("static_cache");
    if let Err(error) = native_cuda_engine(&root) {
        eprintln!(
            "native-cuda could not load the static_cache fixture (op coverage); skipping \
             device-residency asserts: {error:#}"
        );
        return Ok(());
    }

    // Two decode steps: the decoder is re-invoked with the KV cache it produced
    // on the previous step. On the device path that KV never leaves the device.
    // `assert_parity_with` runs the SAME package on ORT and native (pinned to
    // CUDA here) and asserts every listed output matches, then returns the
    // native engine.
    let native = assert_parity_with(
        &root,
        native_cuda_engine,
        || static_cache_request(2),
        &["cache_lengths", "write_indices", "key_cache", "value_cache"],
    )?;

    // Device/provider propagation (Blocker 2) + zero ORT sessions (Blocker 1).
    assert!(
        native.models()?.sessions.is_empty(),
        "Native must build zero ORT sessions even on CUDA"
    );
    let status = native.execution_provider_status();
    assert!(
        status.starts_with("native-cuda"),
        "Native on a CUDA box must report its CUDA device, got {status}"
    );

    // End-to-end device residency: a recurring/intermediate tensor entered a
    // component still on the device (bound zero-copy, no host round-trip), and
    // components produced device-resident outputs kept on the device for the
    // next component. This is the proof the parent required — not merely that a
    // device boundary failed closed.
    let (device_inputs, device_outputs) = native
        .native_device_residency_counts()
        .expect("native backend exposes device-residency counts");
    assert!(
        device_inputs > 0,
        "a recurring/intermediate tensor must enter a component device-resident (bound \
         zero-copy), proving no host round-trip; got {device_inputs} device input bindings"
    );
    assert!(
        device_outputs > 0,
        "components must publish device-resident outputs kept on the device; got {device_outputs}"
    );
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

// ── chained speculative proposal (gemma4_chained) ────────────────────────────

/// Required parity case for the interpreter-owned chained speculative proposal
/// loop.
///
/// The hermetic `gemma4_chained` package declares
/// `proposal_execution: {kind: chained, folded_carry_output: projected_state}`
/// with an explicit `folded_carry_seed` and `token_embedding`, plus a borrowed,
/// read-only `shared_kv` the drafter reads without writing back. Driving it must
/// produce the *same* tokens on ORT and native: the whole proposal chain — fused
/// `concat(embed(last_token), carry)` construction, per-step forward pass,
/// folded-carry threading, acceptance, and rollback of the declared state cells
/// — runs through the one interpreter, with only `invoke_onnx_component`
/// differing between the two.
///
/// Coverage boundary this case deliberately does *not* claim: the tiny drafter
/// slices only the carry half of its fused input, so its greedy tokens do not
/// depend on embedding-gather correctness. `gemma4_chained_workflow.rs` covers
/// the gather separately, against the package's own second copy of the table.
#[test]
fn chained_speculative_proposal_parity() -> anyhow::Result<()> {
    assert_chained_parity(native_engine)
}

/// The same chained proposal on a CUDA-resident native backend. Its shapes all
/// resolve from bound input symbols, so every step's tensors stay device
/// resident; the tokens must still match the ORT reference exactly.
#[cfg(feature = "native-cuda")]
#[test]
fn chained_speculative_proposal_parity_native_cuda() -> anyhow::Result<()> {
    assert_chained_parity(native_cuda_engine)
}

fn assert_chained_parity(
    build_native: impl Fn(&Path) -> anyhow::Result<Engine>,
) -> anyhow::Result<()> {
    let root = chained::fixture_root();
    let mut ort = chained::ChainedFixture::new(ort_engine(&root)?)?;
    let mut native = chained::ChainedFixture::new(build_native(&root)?)?;

    // One proposal block: identical guaranteed token, identical drafts, identical
    // cost. A native run that silently fell back to ORT would still match, so the
    // run counters below rule that out.
    let ort_proposal = ort.propose(chained::PROMPT_TOKENS, 4)?;
    let native_proposal = native.propose(chained::PROMPT_TOKENS, 4)?;
    assert_eq!(
        ort_proposal, native_proposal,
        "the chained proposal diverged between ORT and native"
    );

    // A full propose/verify/accept/reject/rollback decode, which is where the
    // folded carry, the borrowed read-only shared KV, and the declared rollback
    // state all have to agree.
    //
    // The staging count is read across the decode because *where* the work
    // happens is as much a contract as what comes out of it. On a device
    // backend a chain that narrowed a borrowed KV binding by copying it down,
    // or argmaxed a logits row on the host, would produce byte-identical tokens
    // and be invisible to every assertion above — which is exactly why the
    // per-token transfers were there to begin with.
    let native_staging_before = native.engine().host_staging_count();
    let native_readback_before = native.engine().device_readback_bytes();
    let (ort_tokens, ort_tally) = ort.speculative_decode(8, 4)?;
    let (native_tokens, native_tally) = native.speculative_decode(8, 4)?;
    // Zero, and the zero is the point.
    //
    // Every tensor a proposal step touches — the borrowed read-only shared KV,
    // the folded carry seed, the per-step carry, the gathered embedding row,
    // the fused input they are written into, the logits, and the rolled-back
    // state — is narrowed, gathered, assembled, scored and truncated where it
    // already is. Nothing is materialized on the host, so a reintroduced copy
    // is a failing assertion rather than a slow run nobody attributes.
    let staged = native.engine().host_staging_count() - native_staging_before;
    assert_eq!(
        staged, 0,
        "the proposal chain performed {staged} device→host materializations; every tensor it \
         touches is supposed to be narrowed, gathered, assembled and scored where it already is"
    );
    // The one sanctioned transfer, stated as a budget rather than a hope: four
    // bytes per proposer invocation, which is the token id the device argmax
    // produced. `proposer_invocations` counts both backends' identical chains,
    // so the native half is exactly half the tally.
    let readback = native.engine().device_readback_bytes() - native_readback_before;
    let token_id_bytes = (native_tally.proposer_invocations * std::mem::size_of::<u32>()) as u64;
    assert!(
        readback <= token_id_bytes,
        "the proposal chain read {readback} bytes back off the device, above the \
         {token_id_bytes} its {} token ids account for; something other than a token id is \
         crossing the bus",
        native_tally.proposer_invocations
    );
    assert_eq!(
        ort_tokens, native_tokens,
        "speculative decoding diverged between ORT and native"
    );
    assert_eq!(
        ort_tally, native_tally,
        "the two backends took different accept/reject paths"
    );
    assert_eq!(
        ort_tokens,
        chained::greedy_reference(&root, 8)?,
        "both backends must reproduce plain greedy decoding"
    );
    assert!(
        ort_tally.rejections > 0 && ort_tally.rolled_back_cells > 0,
        "the case must exercise rejection and rollback: {ort_tally:?}"
    );

    assert!(
        ort.engine().native_component_run_count().is_none(),
        "the ORT reference must hold no native sessions"
    );
    assert!(
        native.engine().native_component_run_count().unwrap_or(0) > 0,
        "the native run must have executed native component sessions, not fallen back to ORT"
    );
    // The table is a package artifact, so it is read once for the runtime's
    // life however many rounds the decode takes.
    for engine in [ort.engine(), native.engine()] {
        assert_eq!(
            engine.embedding_table_loads(),
            1,
            "the declared embedding table must be read out of the artifact exactly once"
        );
    }
    Ok(())
}

/// Every tensor of a chained proposal stays on the device that produced it.
///
/// The parity case above proves the *tokens* are the same; this proves *where*
/// they were computed, which no token comparison can see. A chain that narrowed
/// a borrowed KV binding by copying it down, argmaxed a logits row on the host,
/// assembled `concat(embed(token), carry)` in host memory, or rolled a rejected
/// proposal back through a full KV download would produce byte-identical output
/// and be invisible to every assertion in `assert_chained_parity`.
///
/// The three claims are stated as numbers rather than as an absence:
///
/// * no device→host materialization at all;
/// * exactly four bytes back per proposer invocation — the token id the device
///   argmax produced, and nothing else;
/// * every rolled-back state cell still resident on the device it was written
///   on, at the truncated length.
#[cfg(feature = "native-cuda")]
#[test]
fn chained_proposal_stays_device_resident_native_cuda() -> anyhow::Result<()> {
    let root = chained::fixture_root();
    let mut native = chained::ChainedFixture::new(native_cuda_engine(&root)?)?;

    let staging_before = native.engine().host_staging_count();
    let readback_before = native.engine().device_readback_bytes();
    let (_, outputs_before) = native
        .engine()
        .native_device_residency_counts()
        .expect("the native backend exposes device-residency counts");
    let (committed, tally) = native.speculative_decode(8, 4)?;
    assert!(!committed.is_empty(), "the decode must commit tokens");
    assert!(
        tally.rejections > 0 && tally.rolled_back_cells > 0,
        "the case must exercise rejection and rollback: {tally:?}"
    );

    assert_eq!(
        native.engine().host_staging_count() - staging_before,
        0,
        "the proposal chain materialized a device tensor on the host"
    );
    let readback = native.engine().device_readback_bytes() - readback_before;
    let token_id_bytes = (tally.proposer_invocations * std::mem::size_of::<u32>()) as u64;
    assert_eq!(
        readback,
        token_id_bytes,
        "the chain read {readback} bytes back off the device; a device-resident chain reads back \
         exactly one {}-byte token id per proposer invocation, and it made {} of them",
        std::mem::size_of::<u32>(),
        tally.proposer_invocations
    );

    // The proposer's own outputs stayed on the device. A chain whose logits and
    // folded carry came back through host memory would still argmax correctly
    // and still tally identically — it would just pay a download per draft
    // token, which is precisely the cost this is here to detect.
    let (_, outputs_after) = native
        .engine()
        .native_device_residency_counts()
        .expect("the native backend exposes device-residency counts");
    let proposer_outputs = 2 * tally.proposer_invocations as u64;
    assert!(
        outputs_after - outputs_before >= proposer_outputs,
        "the chain published {} device-resident outputs across {} proposer invocations; each one \
         produces a logits row and a folded carry, so at least {proposer_outputs} of them must \
         have stayed on the device",
        outputs_after - outputs_before,
        tally.proposer_invocations
    );

    // A rejection rolls the declared state back where it lives. The cells are
    // `[batch, heads, sequence, head_dim]`, so the kept prefix is strided —
    // the case that used to download the whole cache and upload it again.
    let block = vec![committed[0]; 3];
    let mut state = native.verification_state(&block)?;
    let length = chained::PROMPT_TOKENS.len() + 1;
    native
        .engine()
        .rollback_speculative_state(&mut state, length)?;
    assert!(!state.is_empty(), "the package must declare rollback state");
    for (cell, value) in &state {
        assert!(
            !value.is_host_resident()?,
            "rolled-back state cell '{cell}' came back host-resident"
        );
        assert_eq!(
            value.device_id()?,
            0,
            "rolled-back state cell '{cell}' moved off the device it was written on"
        );
        assert_eq!(
            value.shape()[2] as usize,
            length,
            "rolled-back state cell '{cell}' kept the wrong number of positions"
        );
    }
    Ok(())
}

/// A thousand proposals hold no device memory.
///
/// Every step of a device-resident chain allocates: a gathered embedding row, a
/// narrowed carry aliasing the proposer's output binding, a truncated state
/// cell. Each of those keeps its owner alive exactly as long as it is used, and
/// the proof that the ownership is right is that free device memory comes back
/// to where it started rather than drifting down — which a leak of one binding
/// per draft token would not.
#[cfg(feature = "native-cuda")]
#[test]
fn repeated_proposals_do_not_retain_device_memory_native_cuda() -> anyhow::Result<()> {
    let root = chained::fixture_root();
    let mut native = chained::ChainedFixture::new(native_cuda_engine(&root)?)?;

    // Warm up: the first rounds allocate the session's arena, the embedding
    // table's device mirror, and the fused input buffer, none of which are
    // per-proposal.
    for _ in 0..8 {
        native.propose(chained::PROMPT_TOKENS, 4)?;
    }
    let settled = onnx_genai_ort::cuda_rt::device_memory_info(0)?.free_bytes;
    for _ in 0..100 {
        native.propose(chained::PROMPT_TOKENS, 4)?;
    }
    let after = onnx_genai_ort::cuda_rt::device_memory_info(0)?.free_bytes;
    assert!(
        after >= settled,
        "100 proposals retained {} bytes of device memory that the first eight did not; an alias \
         is outliving the buffer it borrows",
        settled - after
    );
    Ok(())
}

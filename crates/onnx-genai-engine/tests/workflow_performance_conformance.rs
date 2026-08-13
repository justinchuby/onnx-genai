//! Performance conformance for metadata-linked policy execution islands.
//!
//! Run on an otherwise idle machine so both paths see the same EP and clocks:
//!
//! ```bash
//! ONNX_GENAI_WORKFLOW_PERF_EP=cuda \
//! ONNX_GENAI_WORKFLOW_PERF_ITERS=200 \
//! cargo test -p onnx-genai-engine --features cuda,cuda-13000 \
//!   --test workflow_performance_conformance -- --ignored --nocapture
//! ```

use std::fs;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::Instant;

use onnx_genai_engine::{
    Engine, EngineConfig, GeneratePrompt, GenerateRequest, PipelineGenerateRequest,
};
use onnx_genai_ort::{Allocator, DataType, Environment, IoBinding, Session, SessionOptions, Value};

const BATCH: usize = 32;
const VOCAB: usize = 32_768;

const DECODER: &str = r#"
ir_version: 8
graph {
  node {
    input: "scores" output: "logits" op_type: "Softmax"
    attribute { name: "axis" i: -1 type: INT }
  }
  name: "decoder"
  input { name: "scores" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_param: "vocabulary" }
  }}}}
  output { name: "logits" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_param: "vocabulary" }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

const SAMPLER: &str = r#"
ir_version: 8
graph {
  node {
    input: "logits" output: "token" op_type: "ArgMax"
    attribute { name: "axis" i: -1 type: INT }
    attribute { name: "keepdims" i: 0 type: INT }
  }
  name: "sampler"
  input { name: "logits" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_param: "vocabulary" }
  }}}}
  output { name: "token" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

const TERMINATION: &str = r#"
ir_version: 8
graph {
  node { input: "token" input: "eos" output: "done" op_type: "Equal" }
  name: "termination"
  input { name: "token" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
  input { name: "eos" type { tensor_type { elem_type: 7 shape {
    dim { dim_value: 1 }
  }}}}
  output { name: "done" type { tensor_type { elem_type: 9 shape {
    dim { dim_param: "batch" }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

const MIN_P: &str = r#"
ir_version: 8
graph {
  node {
    input: "logits" output: "max_logit" op_type: "ReduceMax"
    attribute { name: "axes" ints: 1 type: INTS }
    attribute { name: "keepdims" i: 1 type: INT }
  }
  node { input: "min_p" output: "log_min_p" op_type: "Log" }
  node { input: "max_logit" input: "log_min_p" output: "threshold" op_type: "Add" }
  node { input: "logits" input: "threshold" output: "keep" op_type: "GreaterOrEqual" }
  node {
    input: "keep" input: "logits" input: "negative"
    output: "filtered_logits" op_type: "Where"
  }
  name: "min_p"
  initializer { data_type: 1 float_data: -1000000000 name: "negative" }
  input { name: "logits" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_param: "vocabulary" }
  }}}}
  input { name: "min_p" type { tensor_type { elem_type: 1 shape {
    dim { dim_value: 1 }
  }}}}
  output { name: "filtered_logits" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_param: "vocabulary" }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

const DECODER_NATIVE: &str = r#"
ir_version: 8
graph {
  node {
    input: "scores" output: "logits" op_type: "Softmax"
    attribute { name: "axis" i: -1 type: INT }
  }
  node {
    input: "logits" output: "token" op_type: "ArgMax"
    attribute { name: "axis" i: -1 type: INT }
    attribute { name: "keepdims" i: 0 type: INT }
  }
  node { input: "token" input: "eos" output: "done" op_type: "Equal" }
  name: "decoder_native"
  input { name: "scores" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_param: "vocabulary" }
  }}}}
  input { name: "eos" type { tensor_type { elem_type: 7 shape {
    dim { dim_value: 1 }
  }}}}
  output { name: "token" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
  output { name: "done" type { tensor_type { elem_type: 9 shape {
    dim { dim_param: "batch" }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

const MIN_P_NATIVE: &str = r#"
ir_version: 8
graph {
  node {
    input: "logits" output: "max_logit" op_type: "ReduceMax"
    attribute { name: "axes" ints: 1 type: INTS }
    attribute { name: "keepdims" i: 1 type: INT }
  }
  node { input: "min_p" output: "log_min_p" op_type: "Log" }
  node { input: "max_logit" input: "log_min_p" output: "threshold" op_type: "Add" }
  node { input: "logits" input: "threshold" output: "keep" op_type: "GreaterOrEqual" }
  node {
    input: "keep" input: "logits" input: "negative"
    output: "filtered_logits" op_type: "Where"
  }
  node {
    input: "filtered_logits" output: "token" op_type: "ArgMax"
    attribute { name: "axis" i: -1 type: INT }
    attribute { name: "keepdims" i: 0 type: INT }
  }
  node { input: "token" input: "eos" output: "done" op_type: "Equal" }
  name: "min_p_native"
  initializer { data_type: 1 float_data: -1000000000 name: "negative" }
  input { name: "logits" type { tensor_type { elem_type: 1 shape {
    dim { dim_param: "batch" } dim { dim_param: "vocabulary" }
  }}}}
  input { name: "min_p" type { tensor_type { elem_type: 1 shape {
    dim { dim_value: 1 }
  }}}}
  input { name: "eos" type { tensor_type { elem_type: 7 shape {
    dim { dim_value: 1 }
  }}}}
  output { name: "token" type { tensor_type { elem_type: 7 shape {
    dim { dim_param: "batch" }
  }}}}
  output { name: "done" type { tensor_type { elem_type: 9 shape {
    dim { dim_param: "batch" }
  }}}}
}
opset_import { domain: "" version: 13 }
"#;

struct StableRunner {
    outputs: Vec<(String, Value)>,
    inputs: Vec<(String, Value)>,
    binding: IoBinding,
    _allocator: Option<Allocator>,
    session: Session,
    capture: bool,
    captured: bool,
}

impl StableRunner {
    fn new(
        env: &Environment,
        model: &Path,
        options: SessionOptions,
        inputs: &[(&str, &Value)],
        outputs: &[(&str, &[i64], DataType)],
    ) -> anyhow::Result<Self> {
        let session = Session::new(env, model, options)?;
        let capture = session.graph_capture() && session.cuda_device_id().is_some();
        let allocator = capture
            .then(|| session.device_allocator())
            .transpose()?
            .flatten();
        let mut binding = IoBinding::new(&session)?;
        let mut stable_inputs = Vec::new();
        for (name, source) in inputs {
            let stable = if let Some(allocator) = allocator.as_ref() {
                Value::empty_in(source.shape(), source.dtype(), allocator)?
            } else {
                Value::empty(source.shape(), source.dtype())?
            };
            copy_value(source, &stable, session.cuda_device_id())?;
            binding.bind_input(name, &stable)?;
            stable_inputs.push(((*name).to_string(), stable));
        }
        let mut stable_outputs = Vec::new();
        for (name, shape, dtype) in outputs {
            let stable = if let Some(allocator) = allocator.as_ref() {
                Value::empty_in(shape, *dtype, allocator)?
            } else {
                Value::empty(shape, *dtype)?
            };
            binding.bind_output(name, &stable)?;
            stable_outputs.push(((*name).to_string(), stable));
        }
        Ok(Self {
            outputs: stable_outputs,
            inputs: stable_inputs,
            binding,
            _allocator: allocator,
            session,
            capture,
            captured: false,
        })
    }

    fn run(&mut self, inputs: &[(&str, &Value)]) -> anyhow::Result<Vec<Value>> {
        for ((expected, destination), (actual, source)) in self.inputs.iter().zip(inputs) {
            anyhow::ensure!(expected == actual, "native runner input order changed");
            copy_value(source, destination, self.session.cuda_device_id())?;
        }
        if self.capture {
            self.session.synchronize_device()?;
            self.session.run_with_binding_graph(&self.binding, 10_000)?;
            self.captured = true;
        } else {
            self.session.run_with_binding(&self.binding)?;
        }
        self.outputs
            .iter()
            .map(|(_, output)| {
                if let Some(device) = self.session.cuda_device_id() {
                    output.to_host_from_cuda(device).map_err(Into::into)
                } else {
                    Value::from_raw_bytes(output.to_raw_bytes()?, output.shape(), output.dtype())
                        .map_err(Into::into)
                }
            })
            .collect()
    }
}

fn copy_value(source: &Value, destination: &Value, device: Option<i32>) -> anyhow::Result<()> {
    if let Some(device) = device {
        destination.copy_from_cuda(source, device)?;
    } else {
        destination.copy_from_host(source)?;
    }
    Ok(())
}

fn package(
    name: &str,
    metadata: &str,
    models: &[(&str, &str)],
    native: &str,
) -> anyhow::Result<PathBuf> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/test-fixtures/workflow-performance")
        .join(name);
    fs::create_dir_all(&root)?;
    fs::write(root.join("inference_metadata.yaml"), metadata)?;
    for (name, model) in models {
        fs::write(root.join(name), model)?;
    }
    fs::write(root.join("native.onnx.textproto"), native)?;
    Ok(root)
}

fn workflow_metadata(first_component: &str, first_artifact: &str) -> String {
    let first_inputs = if first_component == "min_p_filter" {
        "{ logits: logits, min_p: min_p }"
    } else {
        "{ scores: logits }"
    };
    let first_output = if first_component == "min_p_filter" {
        "{ filtered_logits: policy_logits }"
    } else {
        "{ logits: policy_logits }"
    };
    format!(
        r#"
pipeline:
  workflow:
    manifest:
      ir_version: "1.0"
      onnx_opsets: {{ ai.onnx: 13 }}
      adapter_abis: {{}}
      custom_op_versions: {{}}
      capabilities: [workflow_ssa, typed_emit]
    inputs:
      logits:
        contract: {{ dtype: float32, rank: 2, shape: [{BATCH}, {VOCAB}] }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: logits }}
        required: true
      min_p:
        contract: {{ dtype: float32, rank: 1, shape: [1] }}
        role: {{ kind: runtime, version: "1", role: sampling_min_p }}
        source: {{ kind: request }}
        required: true
      eos:
        contract: {{ dtype: int64, rank: 1, shape: [1] }}
        role: {{ kind: opaque }}
        source: {{ kind: application, name: eos }}
        required: true
    outputs:
      token:
        contract: {{ dtype: int64, rank: 1, shape: [{BATCH}] }}
        role: tokens
        stage: pre_adapter
      done:
        contract: {{ dtype: bool, rank: 1, shape: [{BATCH}] }}
        role: tensor
        stage: pre_adapter
    components:
      {first_component}:
        implementation: {{ kind: onnx, artifact: {first_artifact} }}
      sampler:
        implementation: {{ kind: onnx, artifact: sampler.onnx.textproto }}
      termination:
        implementation: {{ kind: onnx, artifact: termination.onnx.textproto }}
    steps:
      - kind: invoke
        component: {first_component}
        inputs: {first_inputs}
        outputs: {first_output}
      - kind: invoke
        component: sampler
        inputs: {{ logits: policy_logits }}
        outputs: {{ token: sampled }}
      - kind: invoke
        component: termination
        inputs: {{ token: sampled, eos: eos }}
        outputs: {{ done: is_done }}
      - kind: emit
        value: sampled
        output: token
        mode: replace
      - kind: emit
        value: is_done
        output: done
        mode: replace
"#
    )
}

fn logits_bytes() -> Vec<u8> {
    (0..BATCH * VOCAB)
        .flat_map(|index| {
            let value = ((index % VOCAB) as f32 * 0.0001).sin();
            value.to_le_bytes()
        })
        .collect()
}

fn workflow_request(logits: Vec<u8>) -> anyhow::Result<PipelineGenerateRequest> {
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(Vec::new()));
    request.options.min_p = 0.1;
    Ok(PipelineGenerateRequest::new(request)
        .with_input(
            "logits",
            Value::from_raw_bytes(logits, &[BATCH as i64, VOCAB as i64], DataType::Float32)?,
        )
        .with_input("eos", Value::from_slice_i64(&[0], &[1])?))
}

fn native_inputs(logits: Vec<u8>, min_p: bool) -> anyhow::Result<(Vec<Value>, Vec<&'static str>)> {
    let mut values = vec![
        Value::from_raw_bytes(logits, &[BATCH as i64, VOCAB as i64], DataType::Float32)?,
        Value::from_slice_i64(&[0], &[1])?,
    ];
    let names = if min_p {
        values.insert(1, Value::from_slice_f32(&[0.1], &[1])?);
        vec!["logits", "min_p", "eos"]
    } else {
        vec!["scores", "eos"]
    };
    Ok((values, names))
}

fn benchmark_case(
    label: &str,
    root: &Path,
    min_p: bool,
    iterations: usize,
    samples: usize,
    minimum_ratio: f64,
    maximum_ttft_overhead_ms: f64,
) -> anyhow::Result<()> {
    let logits = logits_bytes();
    let mut engine = Engine::from_pipeline_dir(root, EngineConfig::default())?;
    let environment_name = format!("workflow-performance-{label}");
    let env = Environment::new(&environment_name)?;
    let options = SessionOptions::default();
    let (initial_values, input_names) = native_inputs(logits.clone(), min_p)?;
    let initial = input_names
        .iter()
        .zip(&initial_values)
        .map(|(name, value)| (*name, value))
        .collect::<Vec<_>>();
    let mut native = StableRunner::new(
        &env,
        &root.join("native.onnx.textproto"),
        options,
        &initial,
        &[
            ("token", &[BATCH as i64], DataType::Int64),
            ("done", &[BATCH as i64], DataType::Bool),
        ],
    )?;

    let workflow_cold_start = {
        let started = Instant::now();
        black_box(engine.run_pipeline(workflow_request(logits.clone())?)?);
        started.elapsed()
    };
    let native_cold_start = {
        let (values, names) = native_inputs(logits.clone(), min_p)?;
        let inputs = names
            .iter()
            .zip(&values)
            .map(|(name, value)| (*name, value))
            .collect::<Vec<_>>();
        let started = Instant::now();
        black_box(native.run(&inputs)?);
        started.elapsed()
    };

    for _ in 0..3 {
        black_box(engine.run_pipeline(workflow_request(logits.clone())?)?);
        let (values, names) = native_inputs(logits.clone(), min_p)?;
        let inputs = names
            .iter()
            .zip(&values)
            .map(|(name, value)| (*name, value))
            .collect::<Vec<_>>();
        black_box(native.run(&inputs)?);
    }
    let workflow_ttft = {
        let request = workflow_request(logits.clone())?;
        let started = Instant::now();
        black_box(engine.run_pipeline(request)?);
        started.elapsed()
    };
    let native_ttft = {
        let (values, names) = native_inputs(logits.clone(), min_p)?;
        let inputs = names
            .iter()
            .zip(&values)
            .map(|(name, value)| (*name, value))
            .collect::<Vec<_>>();
        let started = Instant::now();
        black_box(native.run(&inputs)?);
        started.elapsed()
    };

    let mut ratios = Vec::with_capacity(samples);
    let mut workflow_rates = Vec::with_capacity(samples);
    let mut native_rates = Vec::with_capacity(samples);
    for sample in 0..samples {
        let mut workflow_elapsed = std::time::Duration::ZERO;
        let mut native_elapsed = std::time::Duration::ZERO;
        for iteration in 0..iterations {
            let mut run_workflow = || -> anyhow::Result<()> {
                let request = workflow_request(logits.clone())?;
                let started = Instant::now();
                black_box(engine.run_pipeline(request)?);
                workflow_elapsed += started.elapsed();
                Ok(())
            };
            let mut run_native = || -> anyhow::Result<()> {
                let (values, names) = native_inputs(logits.clone(), min_p)?;
                let inputs = names
                    .iter()
                    .zip(&values)
                    .map(|(name, value)| (*name, value))
                    .collect::<Vec<_>>();
                let started = Instant::now();
                black_box(native.run(&inputs)?);
                native_elapsed += started.elapsed();
                Ok(())
            };
            if (sample + iteration) % 2 == 0 {
                run_workflow()?;
                run_native()?;
            } else {
                run_native()?;
                run_workflow()?;
            }
        }
        let workflow_steps = iterations as f64 / workflow_elapsed.as_secs_f64();
        let native_steps = iterations as f64 / native_elapsed.as_secs_f64();
        workflow_rates.push(workflow_steps);
        native_rates.push(native_steps);
        ratios.push(workflow_steps / native_steps);
    }
    workflow_rates.sort_by(f64::total_cmp);
    native_rates.sort_by(f64::total_cmp);
    ratios.sort_by(f64::total_cmp);
    let workflow_steps = workflow_rates[workflow_rates.len() / 2];
    let native_steps = native_rates[native_rates.len() / 2];
    let ratio = ratios[ratios.len() / 2];
    let diagnostic = engine.workflow_performance_diagnostic();
    let island = diagnostic
        .islands
        .first()
        .expect("policy chain must lower into an execution island");
    eprintln!(
        "{label}: median workflow={workflow_steps:.2} step/s native={native_steps:.2} step/s \
         paired_ratio={ratio:.3} samples={samples} workflow_ttft={workflow_ttft:?} \
         native_ttft={native_ttft:?} workflow_cold_start={workflow_cold_start:?} \
         native_cold_start={native_cold_start:?}\n\
         island={:?}",
        island
    );
    assert_eq!(island.components.len(), 3);
    assert_eq!(island.component_boundaries_elided, 2);
    if island.capture_eligible {
        assert!(island.captures >= 1, "{label} never captured");
        assert!(
            island.replays >= iterations as u64,
            "{label} did not replay"
        );
    }
    assert!(
        ratio >= minimum_ratio,
        "{label} workflow is {:.1}% slower than the equivalent native composite \
         (ratio {ratio:.3}, required {minimum_ratio:.3}); inspect island copy/sync/session counters",
        (1.0 - ratio) * 100.0
    );
    let ttft_overhead_ms = workflow_ttft.saturating_sub(native_ttft).as_secs_f64() * 1000.0;
    assert!(
        ttft_overhead_ms <= maximum_ttft_overhead_ms,
        "{label} workflow TTFT overhead is {ttft_overhead_ms:.1} ms, exceeding the \
         {maximum_ttft_overhead_ms:.1} ms acceptance limit"
    );
    Ok(())
}

fn configure_ep() -> (&'static str, bool) {
    let ep = std::env::var("ONNX_GENAI_WORKFLOW_PERF_EP").unwrap_or_else(|_| "cpu".into());
    let cuda = ep == "cuda";
    unsafe {
        std::env::set_var("ONNX_GENAI_EP", &ep);
        std::env::set_var("ONNX_GENAI_CUDA_GRAPH", if cuda { "1" } else { "0" });
    }
    (if cuda { "cuda" } else { "cpu" }, cuda)
}

#[test]
#[ignore = "performance conformance is hardware-sensitive and must run on an idle benchmark host"]
fn workflow_islands_are_competitive_with_native_composites() -> anyhow::Result<()> {
    let (ep, _) = configure_ep();
    let iterations = std::env::var("ONNX_GENAI_WORKFLOW_PERF_ITERS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(100);
    let minimum_ratio = std::env::var("ONNX_GENAI_WORKFLOW_PERF_MIN_RATIO")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0.90);
    let samples = std::env::var("ONNX_GENAI_WORKFLOW_PERF_SAMPLES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(5);
    let maximum_ttft_overhead_ms = std::env::var("ONNX_GENAI_WORKFLOW_PERF_MAX_TTFT_OVERHEAD_MS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(250.0);
    anyhow::ensure!(iterations > 0, "performance iterations must be non-zero");
    anyhow::ensure!(samples > 0, "performance samples must be non-zero");

    let decoder = package(
        &format!("decoder-{ep}"),
        &workflow_metadata("decoder", "decoder.onnx.textproto"),
        &[
            ("decoder.onnx.textproto", DECODER),
            ("sampler.onnx.textproto", SAMPLER),
            ("termination.onnx.textproto", TERMINATION),
        ],
        DECODER_NATIVE,
    )?;
    benchmark_case(
        "decoder+sampler+termination",
        &decoder,
        false,
        iterations,
        samples,
        minimum_ratio,
        maximum_ttft_overhead_ms,
    )?;

    let min_p = package(
        &format!("min-p-{ep}"),
        &workflow_metadata("min_p_filter", "min-p.onnx.textproto"),
        &[
            ("min-p.onnx.textproto", MIN_P),
            ("sampler.onnx.textproto", SAMPLER),
            ("termination.onnx.textproto", TERMINATION),
        ],
        MIN_P_NATIVE,
    )?;
    benchmark_case(
        "min-p+sampler+termination",
        &min_p,
        true,
        iterations,
        samples,
        minimum_ratio,
        maximum_ttft_overhead_ms,
    )?;
    Ok(())
}

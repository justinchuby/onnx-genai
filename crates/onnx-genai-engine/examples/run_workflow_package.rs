//! Run a workflow package with caller-supplied tensors and dump its outputs.
//!
//! Usage: run_workflow_package <package_dir> <spec.json> [more specs...]
//!
//! The spec names the request inputs, the raw little-endian tensor file backing
//! each one, and where the outputs should be written. It exists so an external
//! reference implementation can be compared against this runtime numerically
//! rather than by inspecting the graph. Several specs run back to back on one
//! engine, which is how per-invocation state release is observed: a later
//! request must not see anything the earlier ones left behind.
//!
//! ```json
//! {
//!   "max_new_tokens": 3,
//!   "outputs_dir": "out",
//!   "inputs": [
//!     {"name": "request.noise", "file": "noise.f32", "shape": [1, 3, 4, 2, 2], "dtype": "f32"}
//!   ]
//! }
//! ```
//!
//! Each output tensor is written to `<outputs_dir>/<name>.bin` and its shape and
//! dtype recorded in `<outputs_dir>/outputs.json`.

use onnx_genai_engine::{
    Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest,
    PipelineGenerateRequest, pipeline::PipelineEngine,
};
use onnx_genai_ort::{DataType, Value};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct InputSpec {
    name: String,
    file: String,
    shape: Vec<i64>,
    #[serde(default = "default_dtype")]
    dtype: String,
}

fn default_dtype() -> String {
    "f32".to_string()
}

#[derive(Deserialize)]
struct Spec {
    #[serde(default = "default_steps")]
    max_new_tokens: usize,
    outputs_dir: String,
    #[serde(default)]
    prompt_tokens: Vec<u32>,
    inputs: Vec<InputSpec>,
}

fn default_steps() -> usize {
    1
}

fn load_value(base: &Path, spec: &InputSpec) -> anyhow::Result<Value> {
    let path = base.join(&spec.file);
    let bytes = std::fs::read(&path)?;
    let value = match spec.dtype.as_str() {
        "f32" => {
            let floats: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            Value::from_slice_f32(&floats, &spec.shape)?
        }
        "i64" => {
            let ints: Vec<i64> = bytes
                .chunks_exact(8)
                .map(|chunk| {
                    i64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ])
                })
                .collect();
            Value::from_slice_i64(&ints, &spec.shape)?
        }
        "bool" => Value::from_raw_bytes(bytes, &spec.shape, DataType::Bool)?,
        other => anyhow::bail!("unsupported dtype {other}"),
    };
    Ok(value)
}

fn run_spec(engine: &mut PipelineEngine, spec_path: &Path, load_ms: f64) -> anyhow::Result<()> {
    let spec: Spec = serde_json::from_slice(&std::fs::read(spec_path)?)?;
    let spec_base = spec_path.parent().unwrap_or(Path::new(".")).to_path_buf();

    let options = GenerateOptions {
        max_new_tokens: spec.max_new_tokens,
        ..Default::default()
    };
    let mut request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(spec.prompt_tokens.clone()),
        options,
    });
    for input in &spec.inputs {
        request = request.with_input(&input.name, load_value(&spec_base, input)?);
    }

    let ran = std::time::Instant::now();
    let outputs = engine.run_pipeline_outputs(request)?;
    let elapsed = ran.elapsed();

    let out_dir = spec_base.join(&spec.outputs_dir);
    std::fs::create_dir_all(&out_dir)?;
    let mut manifest = serde_json::Map::new();
    for (name, value) in outputs.iter() {
        let floats = value.to_vec_f32()?;
        let mut bytes = Vec::with_capacity(floats.len() * 4);
        for float in &floats {
            bytes.extend_from_slice(&float.to_le_bytes());
        }
        std::fs::write(out_dir.join(format!("{name}.bin")), bytes)?;
        manifest.insert(
            name.clone(),
            serde_json::json!({"shape": value.shape(), "dtype": "f32"}),
        );
    }
    manifest.insert(
        "__timing_ms".to_string(),
        serde_json::json!({"load": load_ms, "run": elapsed.as_secs_f64() * 1000.0}),
    );
    std::fs::write(
        out_dir.join("outputs.json"),
        serde_json::to_vec_pretty(&serde_json::Value::Object(manifest))?,
    )?;
    println!("wrote {} outputs to {}", outputs.len(), out_dir.display());
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let package_dir = PathBuf::from(args.next().expect("package_dir"));
    let spec_paths: Vec<PathBuf> = args.map(PathBuf::from).collect();
    anyhow::ensure!(!spec_paths.is_empty(), "at least one spec is required");

    let started = std::time::Instant::now();
    let mut engine = Engine::from_pipeline_dir(&package_dir, EngineConfig::default())?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    for spec_path in &spec_paths {
        run_spec(&mut engine, spec_path, load_ms)?;
    }
    Ok(())
}

//! Executes a real metadata-defined text-to-video workflow package end to end.
//!
//! Set `VIDEO_WORKFLOW_PACKAGE_DIR`, `VIDEO_WORKFLOW_INPUT_DIR`, and
//! `VIDEO_WORKFLOW_OUTPUT_DIR`, then run this ignored test with the execution
//! provider appropriate for the package.

use onnx_genai_engine::{
    Engine, EngineConfig, GenerateOptions, GeneratePrompt, GenerateRequest, PipelineGenerateRequest,
};
use onnx_genai_ort::{DataType, Value};
use std::path::PathBuf;

#[derive(serde::Deserialize)]
struct InputSpec {
    name: String,
    dtype: String,
    shape: Vec<i64>,
}

fn dtype_of(name: &str) -> anyhow::Result<DataType> {
    Ok(match name {
        "float32" => DataType::Float32,
        "float16" => DataType::Float16,
        "int64" => DataType::Int64,
        other => anyhow::bail!("unsupported input dtype {other}"),
    })
}

fn env_dir(key: &str) -> Option<PathBuf> {
    std::env::var_os(key).map(PathBuf::from)
}

#[test]
#[ignore = "requires a locally exported real-weight video diffusion package"]
fn video_workflow_executes_complete_path() -> anyhow::Result<()> {
    let Some(package_dir) = env_dir("VIDEO_WORKFLOW_PACKAGE_DIR") else {
        eprintln!("VIDEO_WORKFLOW_PACKAGE_DIR unset; skipping");
        return Ok(());
    };
    let input_dir =
        env_dir("VIDEO_WORKFLOW_INPUT_DIR").unwrap_or_else(|| package_dir.join("inputs"));
    let output_dir =
        env_dir("VIDEO_WORKFLOW_OUTPUT_DIR").unwrap_or_else(|| package_dir.join("outputs"));
    std::fs::create_dir_all(&output_dir)?;
    let specs: Vec<InputSpec> =
        serde_json::from_slice(&std::fs::read(input_dir.join("manifest.json"))?)?;

    let load_start = std::time::Instant::now();
    let mut engine = Engine::from_dir(&package_dir, EngineConfig::default())?;
    eprintln!(
        "engine loaded in {:.1}s",
        load_start.elapsed().as_secs_f64()
    );
    let steps = std::env::var("VIDEO_WORKFLOW_STEPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(20);
    let prompt_tokens: Vec<u32> =
        serde_json::from_slice(&std::fs::read(input_dir.join("prompt_tokens.json"))?)?;
    let mut request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(prompt_tokens),
        options: GenerateOptions {
            max_new_tokens: steps,
            seed: Some(42),
            ..Default::default()
        },
    });
    for spec in &specs {
        let bytes = std::fs::read(input_dir.join(format!("{}.bin", spec.name)))?;
        request = request.with_input(
            &spec.name,
            Value::from_raw_bytes(bytes, &spec.shape, dtype_of(&spec.dtype)?)?,
        );
    }

    let run_start = std::time::Instant::now();
    let outputs = engine.run_pipeline_outputs(request)?;
    eprintln!("pipeline ran in {:.1}s", run_start.elapsed().as_secs_f64());
    let video = &outputs["video"];
    let frames = video.to_vec_f32_lossy()?;
    assert!(frames.iter().all(|value| value.is_finite()));
    let mean = frames.iter().sum::<f32>() / frames.len() as f32;
    let variance = frames
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f32>()
        / frames.len() as f32;
    assert!(variance > 1e-3, "video is degenerate (variance {variance})");
    std::fs::write(output_dir.join("video.bin"), f32_bytes(&frames))?;
    std::fs::write(
        output_dir.join("video.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "dtype": "float32",
            "shape": video.shape(),
            "mean": mean,
            "variance": variance,
        }))?,
    )?;
    Ok(())
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

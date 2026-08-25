//! Executes a real Qwen-Image-Edit workflow package end to end.
//!
//! This test is `#[ignore]`d because it needs a ~53 GiB bf16 package that no CI
//! job carries. Point `QWEN_IMAGE_EDIT_PACKAGE_DIR` at a Mobius-exported
//! package containing `inference_metadata.yaml`, `QWEN_IMAGE_EDIT_INPUT_DIR` at
//! a directory holding `manifest.json` plus one `<name>.bin` per application
//! input, and `QWEN_IMAGE_EDIT_OUTPUT_DIR` at a writable directory, then run:
//!
//! ```text
//! ONNX_GENAI_EP=cuda cargo test -p onnx-genai-engine \
//!     --test qwen_image_edit_workflow_e2e -- --ignored --nocapture
//! ```
//!
//! The emitted image is written back as raw little-endian bytes plus a shape
//! sidecar so the producer-side harness can score it against the upstream
//! diffusers reference.

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
        "bfloat16" => DataType::BFloat16,
        "int64" => DataType::Int64,
        "int32" => DataType::Int32,
        "bool" => DataType::Bool,
        "uint8" => DataType::Uint8,
        other => anyhow::bail!("unsupported input dtype {other}"),
    })
}

fn env_dir(key: &str) -> Option<PathBuf> {
    std::env::var_os(key).map(PathBuf::from)
}

#[test]
#[ignore = "requires a locally exported Qwen-Image-Edit package (~53 GiB)"]
fn qwen_image_edit_workflow_executes_complete_edit_path() -> anyhow::Result<()> {
    let Some(package_dir) = env_dir("QWEN_IMAGE_EDIT_PACKAGE_DIR") else {
        eprintln!("QWEN_IMAGE_EDIT_PACKAGE_DIR unset; skipping");
        return Ok(());
    };
    let input_dir =
        env_dir("QWEN_IMAGE_EDIT_INPUT_DIR").unwrap_or_else(|| package_dir.join("inputs"));
    let output_dir =
        env_dir("QWEN_IMAGE_EDIT_OUTPUT_DIR").unwrap_or_else(|| package_dir.join("outputs"));
    std::fs::create_dir_all(&output_dir)?;

    let specs: Vec<InputSpec> =
        serde_json::from_slice(&std::fs::read(input_dir.join("manifest.json"))?)?;

    let load_start = std::time::Instant::now();
    let mut engine = Engine::from_dir(&package_dir, EngineConfig::default())?;
    eprintln!(
        "engine loaded in {:.1}s",
        load_start.elapsed().as_secs_f64()
    );

    // `max_new_tokens` is what the runtime binds to the `max_iterations` role,
    // so it is the denoising step count for a diffusion workflow.
    let steps: usize = std::env::var("QWEN_IMAGE_EDIT_STEPS")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(8);
    // `max_new_tokens` is what the runtime binds to the `max_iterations` role,
    // which for a diffusion workflow is the denoising step count.
    let options = GenerateOptions {
        max_new_tokens: steps,
        seed: Some(42),
        ..Default::default()
    };
    let mut request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![]),
        options,
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

    let image = &outputs["image"];
    eprintln!("image {:?} {:?}", image.dtype(), image.shape());
    let pixels = image.to_vec_f32_lossy()?;
    assert!(
        pixels.iter().all(|value| value.is_finite()),
        "edited image must be finite"
    );
    // A degenerate all-constant image would still be finite; require real signal.
    let mean = pixels.iter().sum::<f32>() / pixels.len() as f32;
    let variance = pixels.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / pixels.len() as f32;
    assert!(
        variance > 1e-3,
        "edited image is degenerate (variance {variance})"
    );

    std::fs::write(output_dir.join("image.bin"), bytemuck_cast(&pixels))?;
    std::fs::write(
        output_dir.join("image.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "dtype": "float32",
            "shape": image.shape(),
        }))?,
    )?;
    eprintln!("wrote {}", output_dir.join("image.bin").display());
    Ok(())
}

fn bytemuck_cast(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

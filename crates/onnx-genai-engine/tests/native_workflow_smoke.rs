//! Smoke: does the native backend run checked workflow packages end-to-end
//! through the one universal interpreter? Full parity/routing assertions live
//! in `native_workflow_parity.rs`.

use std::path::PathBuf;

use onnx_genai_engine::{
    Engine, EngineConfig, EngineDecodeBackend, GenerateOptions, GeneratePrompt, GenerateRequest,
    NativeDecodeDevice, PipelineGenerateRequest, pipeline::PipelineEngine,
};
use onnx_genai_ort::{DataType, Value};

fn root(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/onnx_genai_workflows")
        .join(name)
}

fn native() -> EngineConfig {
    EngineConfig {
        decode_backend: EngineDecodeBackend::Native,
        // Pin CPU so the smoke run is deterministic regardless of build features
        // or a GPU being present; device-residency is covered in the parity file.
        native_device: Some(NativeDecodeDevice::Cpu),
        ..EngineConfig::default()
    }
}

fn options(max_new_tokens: usize) -> GenerateOptions {
    GenerateOptions {
        max_new_tokens,
        seed: Some(7),
        ..Default::default()
    }
}

#[test]
fn native_runs_diffusion_loop_package() -> anyhow::Result<()> {
    let mut engine: PipelineEngine = Engine::from_pipeline_dir(&root("diffusion"), native())?;
    let noise: Vec<f32> = (0..4 * 4 * 4)
        .map(|index| (index as f32 - 32.0) / 16.0)
        .collect();
    let request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![1, 2]),
        options: options(2),
    })
    .with_input(
        "request.noise",
        Value::from_slice_f32(&noise, &[1, 4, 4, 4])?,
    );
    let output = engine.run_pipeline_outputs(request)?;
    assert_eq!(output["image"].shape(), [1, 3, 4, 4]);
    assert!(engine.native_component_run_count().unwrap_or(0) > 0);
    Ok(())
}

#[test]
fn native_runs_static_cache_ar_package() -> anyhow::Result<()> {
    let mut engine: PipelineEngine = Engine::from_pipeline_dir(&root("static_cache"), native())?;
    let request = PipelineGenerateRequest::new(GenerateRequest {
        prompt: GeneratePrompt::TokenIds(vec![]),
        options: options(2),
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
    .with_input("request.max_iterations", Value::from_slice_i64(&[2], &[1])?);
    let output = engine.run_pipeline_outputs(request)?;
    assert_eq!(output["cache_lengths"].to_vec_i64()?, vec![3, 6]);
    assert!(engine.native_component_run_count().unwrap_or(0) > 0);
    Ok(())
}

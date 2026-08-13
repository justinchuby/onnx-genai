use std::path::Path;

use onnx_genai::metadata::load_metadata;
use onnx_genai::ort::{ModelDirectory, PipelineModelDirectory};
use onnx_genai_server::from_models_dir;

use super::resolve_model_dir;

pub(super) fn show(model: &Path) -> anyhow::Result<()> {
    let model_dir = resolve_model_dir(model);
    if let Some(directory) = PipelineModelDirectory::load_if_declared(&model_dir)? {
        println!("model directory: {}", directory.root.display());
        println!(
            "pipeline:        {} component(s)",
            directory.model_paths.len()
        );
        for (name, path) in &directory.model_paths {
            println!("  {name}: {}", path.display());
        }
        match &directory.metadata_path {
            Some(path) => println!("metadata:        {}", path.display()),
            None => println!("metadata:        (compatibility config)"),
        }
        let genai_config = model_dir.join("genai_config.json");
        if genai_config.is_file() {
            println!("genai config:    {}", genai_config.display());
        }
        if let Some(metadata_path) = &directory.metadata_path {
            show_metadata(metadata_path)?;
        }
        return Ok(());
    }
    let directory = ModelDirectory::load(&model_dir)?;

    println!("model directory: {}", directory.root.display());
    println!("model file:      {}", directory.model_path.display());
    println!("tokenizer:       {}", directory.tokenizer_path.display());
    match &directory.metadata_path {
        Some(path) => println!("metadata:        {}", path.display()),
        None => println!("metadata:        (none)"),
    }
    let genai_config = model_dir.join("genai_config.json");
    if genai_config.is_file() {
        println!("genai config:    {}", genai_config.display());
    }
    if let Some(metadata_path) = &directory.metadata_path {
        show_metadata(metadata_path)?;
    }
    Ok(())
}

fn show_metadata(metadata_path: &Path) -> anyhow::Result<()> {
    let metadata = load_metadata(metadata_path)?;
    if !metadata.required_capabilities.is_empty() {
        println!(
            "capabilities:    {}",
            metadata.required_capabilities.join(", ")
        );
    }
    if let Some(model_caps) = &metadata.model {
        if let Some(max_len) = model_caps.max_sequence_length {
            println!("max sequence:    {max_len}");
        }
        if let Some(attention) = &model_caps.attention {
            println!("attention:       {attention:?}");
        }
    }
    if let Some(quantization) = &metadata.quantization {
        println!("quantization:    {quantization:?}");
    }
    Ok(())
}

pub(super) fn list(models_dir: &Path) -> anyhow::Result<()> {
    let specs = from_models_dir(models_dir)?;
    if specs.is_empty() {
        println!("no models found under {}", models_dir.display());
        return Ok(());
    }
    for spec in specs {
        println!("{}\t{}", spec.id, spec.path.display());
    }
    Ok(())
}

pub(super) fn version() {
    println!("onnx-genai {}", env!("CARGO_PKG_VERSION"));
    let mut providers = vec!["cpu"];
    if cfg!(feature = "cuda") {
        providers.push("cuda");
    }
    println!("execution providers: {}", providers.join(", "));
    let ort_report = onnx_genai::ort::onnxruntime_library_report();
    println!("onnx runtime: {ort_report}");
    println!("select an execution provider at runtime with ONNX_GENAI_EP (e.g. cpu, cuda).");
}

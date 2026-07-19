// Copyright (c) Microsoft Corporation.
//
//! Minimal driver to run a non-autoregressive (diffusion / single-pass)
//! pipeline directory with raw-f32 tensor inputs and dump a raw-f32 output.
//!
//! Intended for validation/experiments (e.g. comparing onnx-genai's DDIM loop
//! against diffusers). Not a shipping CLI.
//!
//! Usage:
//!   run_diffusion <pipeline_dir> <output_endpoint> <out.f32> \
//!       <endpoint>:<d,d,..>:<in.f32> [<endpoint>:<d,d,..>:<in.f32> ...]
//!
//! Each input file is little-endian f32 in row-major order for the given shape.

use anyhow::{Context, Result, bail};
use onnx_genai::engine::{
    Engine, EngineConfig, GeneratePrompt, GenerateRequest, PipelineGenerateRequest,
};
use onnx_genai::ort::Value;
use std::fs;

fn read_f32(path: &str) -> Result<Vec<f32>> {
    let bytes = fs::read(path).with_context(|| format!("reading {path}"))?;
    if bytes.len() % 4 != 0 {
        bail!("{path}: length {} is not a multiple of 4", bytes.len());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn read_i64(path: &str) -> Result<Vec<i64>> {
    let bytes = fs::read(path).with_context(|| format!("reading {path}"))?;
    if bytes.len() % 8 != 0 {
        bail!("{path}: length {} is not a multiple of 8", bytes.len());
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn write_f32(path: &str, data: &[f32]) -> Result<()> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, bytes).with_context(|| format!("writing {path}"))
}

fn write_i64(path: &str, data: &[i64]) -> Result<()> {
    let mut bytes = Vec::with_capacity(data.len() * 8);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, bytes).with_context(|| format!("writing {path}"))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        bail!(
            "usage: {} <pipeline_dir> <output_endpoint> <out.f32> \
             <endpoint>:<d,d,..>:<in.f32> ...",
            args[0]
        );
    }
    let pipeline_dir = &args[1];
    let output_endpoint = &args[2];
    let out_path = &args[3];

    let mut request =
        PipelineGenerateRequest::new(GenerateRequest::new(GeneratePrompt::TokenIds(vec![])));
    for spec in &args[4..] {
        // endpoint:[dtype:]d,d,..:path   (dtype in {f32,i64}, default f32)
        // The path is parsed as the tail after the shape so that Windows paths
        // (e.g. `C:\dir\seed.i64`) containing a drive-letter colon still work.
        let parts: Vec<&str> = spec.split(':').collect();
        if parts.len() < 3 {
            bail!("bad input spec '{spec}' (expected endpoint:[dtype:]d,d,..:path)");
        }
        let endpoint = parts[0];
        let (dtype, shape_index) = if parts[1] == "f32" || parts[1] == "i64" {
            (parts[1], 2)
        } else {
            ("f32", 1)
        };
        if parts.len() < shape_index + 2 {
            bail!("bad input spec '{spec}' (expected endpoint:[dtype:]d,d,..:path)");
        }
        let shape_str = parts[shape_index];
        let path = parts[shape_index + 1..].join(":");
        let path = path.as_str();
        let shape: Vec<i64> = shape_str
            .split(',')
            .map(|d| d.trim().parse::<i64>())
            .collect::<Result<_, _>>()
            .with_context(|| format!("bad shape in '{spec}'"))?;
        let expected: i64 = shape.iter().product();
        let value = match dtype {
            "f32" => {
                let data = read_f32(path)?;
                if expected as usize != data.len() {
                    bail!(
                        "input '{endpoint}': shape {:?} implies {expected} f32 but file has {}",
                        shape,
                        data.len()
                    );
                }
                Value::from_slice_f32(&data, &shape)?
            }
            "i64" => {
                let data = read_i64(path)?;
                if expected as usize != data.len() {
                    bail!(
                        "input '{endpoint}': shape {:?} implies {expected} i64 but file has {}",
                        shape,
                        data.len()
                    );
                }
                Value::from_slice_i64(&data, &shape)?
            }
            other => bail!("unsupported dtype '{other}' (expected f32 or i64)"),
        };
        request = request.with_input(endpoint, value);
    }

    let load_start = std::time::Instant::now();
    let mut engine = Engine::from_pipeline_dir(std::path::Path::new(pipeline_dir), EngineConfig::default())?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1e3;
    let run_start = std::time::Instant::now();
    let outputs = engine.run_pipeline(request)?;
    let run_ms = run_start.elapsed().as_secs_f64() * 1e3;
    eprintln!("[timing] load={load_ms:.1}ms run={run_ms:.1}ms");
    let value = outputs
        .get(output_endpoint)
        .with_context(|| format!("output endpoint '{output_endpoint}' not produced"))?;
    let shape = value.shape().to_vec();
    // Integer outputs (e.g. the final token sequence of a masked-diffusion loop)
    // are written as little-endian i64; everything else as little-endian f32.
    use onnx_genai::ort::DataType;
    match value.dtype() {
        DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8 => {
            let data = value.to_vec_i64()?;
            write_i64(out_path, &data)?;
            eprintln!(
                "wrote {output_endpoint} shape {shape:?} ({} i64 elems) -> {out_path}",
                data.len()
            );
        }
        _ => {
            let data = value.to_vec_f32()?;
            write_f32(out_path, &data)?;
            eprintln!(
                "wrote {output_endpoint} shape {shape:?} ({} f32 elems) -> {out_path}",
                data.len()
            );
        }
    }
    Ok(())
}

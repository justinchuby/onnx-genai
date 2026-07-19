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
        let parts: Vec<&str> = spec.split(':').collect();
        let (endpoint, dtype, shape_str, path) = match parts.as_slice() {
            [ep, shape, path] => (*ep, "f32", *shape, *path),
            [ep, dt, shape, path] => (*ep, *dt, *shape, *path),
            _ => bail!("bad input spec '{spec}' (expected endpoint:[dtype:]d,d,..:path)"),
        };
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

    let mut engine = Engine::from_pipeline_dir(std::path::Path::new(pipeline_dir), EngineConfig::default())?;
    let outputs = engine.run_pipeline(request)?;
    let value = outputs
        .get(output_endpoint)
        .with_context(|| format!("output endpoint '{output_endpoint}' not produced"))?;
    let shape = value.shape().to_vec();
    let data = value.to_vec_f32()?;
    write_f32(out_path, &data)?;
    eprintln!("wrote {output_endpoint} shape {shape:?} ({} elems) -> {out_path}", data.len());
    Ok(())
}

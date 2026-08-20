//! One-way `genai_config.json` -> inference-metadata import check.
//!
//! Import is a load-time operation: the runtime converts a legacy config into an
//! in-memory [`InferenceMetadata`] when a package ships no native
//! `inference_metadata.yaml`. This tool runs that same conversion ahead of time
//! so a package author can see whether it succeeds and what it costs.
//!
//! Import is fail-closed: any fact the new contract cannot represent aborts the
//! run unless `--allow-lossy` is passed, in which case every dropped key path is
//! printed and the import continues.
//!
//! There is no export direction, and metadata is deliberately deserialize-only:
//! nothing here turns metadata back into a legacy config or re-serializes it,
//! because a reverse synthesizer would have to approximate facts the new
//! contract states precisely. Package authors write `inference_metadata.yaml`
//! by hand or from their exporter — never from this tool.

use std::path::PathBuf;

use onnx_genai_genai_config::{ImportOptions, drop_reason, import_from_path};

fn main() {
    let mut allow_lossy = false;
    let mut inputs = Vec::new();
    for argument in std::env::args().skip(1) {
        match argument.as_str() {
            "--allow-lossy" => allow_lossy = true,
            "-h" | "--help" => {
                println!(
                    "usage: import_genai_config [--allow-lossy] <genai_config.json> [...]\n\n\
                     Runs the load-time conversion of a legacy onnxruntime-genai config into\n\
                     inference metadata and reports the result. Import is one-way and\n\
                     fail-closed; there is no export direction.\n\n\
                     --allow-lossy  accept facts the contract cannot represent and list them"
                );
                return;
            }
            other if other.starts_with('-') => {
                eprintln!("unknown option: {other}");
                std::process::exit(2);
            }
            other => inputs.push(PathBuf::from(other)),
        }
    }
    if inputs.is_empty() {
        eprintln!("usage: import_genai_config [--allow-lossy] <genai_config.json> [...]");
        std::process::exit(2);
    }

    let options = ImportOptions { allow_lossy };
    let mut failed = false;
    for input in inputs {
        match import_from_path(&input, None, None, options) {
            Ok((metadata, report)) => {
                for key in &report.dropped_keys {
                    match drop_reason(key) {
                        Some(reason) => eprintln!("dropped: {key}: {reason}"),
                        None => eprintln!("dropped: {key}: not represented by this contract"),
                    }
                }
                let loss = if report.is_lossy() {
                    format!("lossy ({} dropped)", report.dropped_keys.len())
                } else {
                    "faithful".to_owned()
                };
                println!(
                    "imported: {} ({loss}): {}",
                    input.display(),
                    summarize(&metadata)
                );
            }
            Err(error) => {
                failed = true;
                eprintln!("failed to import {}: {error}", input.display());
            }
        }
    }
    if failed {
        std::process::exit(1);
    }
}

/// A one-line structural summary of what the conversion produced.
fn summarize(metadata: &onnx_genai_metadata::InferenceMetadata) -> String {
    let mut parts = Vec::new();
    if let Some(model) = &metadata.model {
        if let Some(attention) = &model.attention {
            parts.push(format!("attention={}", attention.attention_type));
        }
        if let Some(io) = &model.io {
            parts.push(format!(
                "state_ports={}",
                io.kv_inputs.as_ref().map_or(0, Vec::len)
            ));
        }
    }
    match &metadata.pipeline {
        Some(pipeline) => parts.push(format!("components={}", pipeline.workflow.components.len())),
        None => parts.push("components=0".to_owned()),
    }
    parts.join(" ")
}

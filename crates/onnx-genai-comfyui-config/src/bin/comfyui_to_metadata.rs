// Copyright (c) Microsoft Corporation.
//
//! Translate a ComfyUI API-format workflow JSON into onnx-genai's native
//! `inference_metadata` (pipeline) JSON plus the recovered run parameters.
//!
//! Usage: `comfyui_to_metadata <workflow.json>` (or read the workflow from
//! stdin). Prints `{"metadata": {...}, "run": {...}}` to stdout — used by the
//! `examples/diffusion-demo` backend to show the translated native config.

use std::io::Read;

use onnx_genai_comfyui_config::parse_workflow_str;
use serde_json::json;

fn main() {
    let mut input = String::new();
    let args: Vec<String> = std::env::args().collect();
    if let Some(path) = args.get(1) {
        input = std::fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("failed to read {path}: {e}");
            std::process::exit(1);
        });
    } else if let Err(e) = std::io::stdin().read_to_string(&mut input) {
        eprintln!("failed to read stdin: {e}");
        std::process::exit(1);
    }

    let workflow = match parse_workflow_str(&input) {
        Ok(workflow) => workflow,
        Err(e) => {
            eprintln!("failed to translate ComfyUI workflow: {e}");
            std::process::exit(2);
        }
    };

    let out = json!({
        "metadata": workflow.metadata_json,
        "run": {
            "prompt": workflow.prompt,
            "negative_prompt": workflow.negative_prompt,
            "width": workflow.width,
            "height": workflow.height,
            "batch_size": workflow.batch_size,
            "seed": workflow.seed,
            "steps": workflow.steps,
            "cfg": workflow.cfg,
            "sampler_name": workflow.sampler_name,
            "scheduler_kind": workflow.scheduler_kind,
            "scheduler_spacing": workflow.scheduler_spacing,
            "checkpoint": workflow.checkpoint,
            "denoise": workflow.denoise,
            "start_step": workflow.start_step,
            "loras": workflow.loras,
            "controlnet": workflow.controlnet,
        },
    });
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
}

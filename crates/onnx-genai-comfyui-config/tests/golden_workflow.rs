//! The checked-in converted package must stay exactly what the importer emits.
//!
//! Conversion is deterministic by construction: the same workflow and the same
//! options always produce byte-identical YAML. That is what makes the golden
//! document under `tests/fixtures/comfyui_workflows/` a regression test on the
//! converter rather than a snapshot that drifts on its own.
//!
//! This test needs no ONNX Runtime. The executable half of the same fixture is
//! `crates/onnx-genai-engine/tests/comfyui_workflow_e2e.rs`.

use std::path::PathBuf;

use onnx_genai_comfyui_config::{ComponentLayout, ConvertOptions, convert_file, to_yaml};

const REGENERATE: &str = "regenerate with: cargo run -p onnx-genai-comfyui-config --bin \
                          comfyui_to_metadata -- --textproto --out \
                          tests/fixtures/comfyui_workflows/txt2img_sd15/inference_metadata.yaml \
                          tests/fixtures/comfyui_workflows/txt2img_sd15/workflow.json";

fn package() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/comfyui_workflows/txt2img_sd15")
}

fn options() -> ConvertOptions {
    ConvertOptions {
        layout: ComponentLayout::textproto(),
        ..ConvertOptions::default()
    }
}

#[test]
fn golden_metadata_matches_a_fresh_conversion() {
    let (_, document, report) =
        convert_file(package().join("workflow.json"), &options()).expect("conversion");
    let regenerated = to_yaml(&document).expect("yaml");
    let checked_in = std::fs::read_to_string(package().join("inference_metadata.yaml"))
        .expect("inference_metadata.yaml");
    assert_eq!(regenerated, checked_in, "{REGENERATE}");
    assert!(
        report.ignored_nodes.is_empty(),
        "the fixture workflow should have no unreachable nodes: {:?}",
        report.ignored_nodes
    );
}

#[test]
fn regeneration_is_byte_stable_across_runs() {
    let first = to_yaml(
        &convert_file(package().join("workflow.json"), &options())
            .expect("conversion")
            .1,
    )
    .expect("yaml");
    let second = to_yaml(
        &convert_file(package().join("workflow.json"), &options())
            .expect("conversion")
            .1,
    )
    .expect("yaml");
    assert_eq!(first, second);
}

#[test]
fn the_golden_document_carries_the_workflows_run_parameters() {
    let checked_in = std::fs::read_to_string(package().join("inference_metadata.yaml"))
        .expect("inference_metadata.yaml");
    let document: serde_json::Value = serde_yaml::from_str(&checked_in).expect("yaml");
    let workflow = &document["pipeline"]["workflow"];

    // Everything the ComfyUI graph declared that decides the image is visible
    // in the canonical document, and nothing about the document is Comfy-shaped.
    assert_eq!(workflow["inputs"]["request.seed"]["default"], 20260821);
    assert_eq!(workflow["inputs"]["request.max_iterations"]["default"], 4);
    assert_eq!(workflow["inputs"]["request.guidance_scale"]["default"], 7.5);
    let solver = &workflow["components"]["solver_step"]["contract"]["parameters"];
    assert_eq!(solver["solver"], "euler");
    assert_eq!(solver["spacing"], "karras");
    assert_eq!(solver["prediction"], "epsilon");
    assert!(
        !checked_in.contains("KSampler") && !checked_in.contains("class_type"),
        "the emitted metadata must not carry ComfyUI vocabulary"
    );
}

#[test]
fn the_golden_document_is_valid_inference_metadata() {
    let checked_in = std::fs::read_to_string(package().join("inference_metadata.yaml"))
        .expect("inference_metadata.yaml");
    let metadata: onnx_genai_metadata::InferenceMetadata =
        serde_yaml::from_str(&checked_in).expect("typed metadata");
    onnx_genai_metadata::validate_metadata(&metadata).expect("valid metadata");
}

#[test]
fn every_referenced_artifact_exists_in_the_package() {
    let metadata = onnx_genai_metadata::load_metadata_package(&package())
        .expect("the converted package must load with its artifacts resolved");
    let workflow = &metadata.pipeline.as_ref().expect("pipeline").workflow;
    assert!(
        workflow.components.values().all(|component| matches!(
            component.implementation,
            onnx_genai_metadata::ComponentImplementation::Onnx { .. }
        )),
        "an imported diffusion workflow invokes ONNX components only"
    );
}

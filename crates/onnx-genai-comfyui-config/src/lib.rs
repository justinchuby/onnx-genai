//! One-way importer that lowers a **ComfyUI API-format workflow JSON** into the
//! canonical onnx-genai `pipeline.workflow` inference metadata.
//!
//! # What this is
//!
//! ComfyUI is a node-graph UI for diffusion. Its *"Save (API Format)"* export is
//! a flat map `{node_id: {"class_type": str, "inputs": {port: value | link}}}`
//! where a value of the form `[src_id, slot]` links to another node's output.
//!
//! This crate is the ComfyUI analogue of [`onnx-genai-genai-config`]: it reads an
//! external config and produces native [`InferenceMetadata`]. Import is
//! **one-way**. There is no export direction and no runtime that consults the
//! ComfyUI document again: once conversion succeeds, the emitted
//! `pipeline.workflow` is the sole source of execution truth, exactly as it is
//! for a package Mobius exported natively. A ComfyUI class name never reaches
//! the runtime, and no execution path branches on one.
//!
//! # What it emits
//!
//! The same canonical IR the checked Mobius diffusion packages carry
//! (`tests/fixtures/onnx_genai_workflows/diffusion`,
//! `.../diffusion_guided`): typed SSA inputs, components with explicit ports and
//! versioned contracts, invocation state cells, and a `loop`/`invoke`/`emit`
//! step tree. Guidance is two text-encoder invocations plus an
//! `onnx-genai.guidance-combine` component; the solver is
//! `onnx-genai.solver-step`; the seeded latent draw is `onnx-genai.counter-rng`.
//! Nothing about the emitted document is ComfyUI-shaped.
//!
//! # Fail-closed conversion
//!
//! Conversion is structural. It walks backwards from the workflow's single image
//! sink and must understand every node that can reach it. A class the importer
//! does not model is [`ComfyUiConfigError::UnknownNode`], naming the node id, the
//! class, and the remedy. Nodes that provably cannot reach the sink are reported
//! in [`ConversionReport::ignored_nodes`] and skipped, which is the only case
//! where skipping is sound.
//!
//! Nothing that changes the produced image is dropped quietly: a scheduler with
//! no canonical contract, a chained ControlNet, a step-windowed ControlNet, a
//! merged conditioning, a truncated CLIP, a patched denoiser, and a LoRA with no
//! declared adapter contract are all errors with a specific remedy.
//!
//! # Weights
//!
//! Like the ComfyUI document itself, this crate carries topology and run
//! parameters only. The ONNX components the emitted workflow invokes come from
//! the package (Mobius's diffusion exporter writes exactly the layout in
//! [`ComponentLayout`]); the importer names them and never creates them.

mod graph;
mod layout;
mod lower;
mod plan;
mod recognize;

#[cfg(test)]
mod tests;

use std::path::Path;

use onnx_genai_metadata::InferenceMetadata;
use serde_json::Value;

pub use graph::{ComfyGraph, Link, Node};
pub use layout::ComponentLayout;
pub use plan::{
    Conditioning, ControlNet, Guidance, LatentSource, Lora, Prediction, Solver, Spacing,
    WorkflowPlan, strength_to_start_step,
};
pub use recognize::recognize;

/// Errors produced while reading or lowering a ComfyUI workflow.
///
/// Every variant that refuses a workflow names the node and how to fix it,
/// because a converter that says only "unsupported" leaves the author guessing
/// which of forty nodes was the problem.
#[derive(Debug, thiserror::Error)]
pub enum ComfyUiConfigError {
    /// The file could not be read.
    #[error("failed to read ComfyUI workflow: {0}")]
    Io(#[from] std::io::Error),

    /// The file was not valid JSON.
    #[error("failed to parse ComfyUI workflow JSON: {0}")]
    Parse(#[from] serde_json::Error),

    /// The document is not a ComfyUI API-format workflow at all.
    #[error(
        "not a ComfyUI API-format workflow: {detail}. \
         How to fix: export the graph with ComfyUI's 'Save (API Format)' command; the UI's \
         plain 'Save' writes a different document that carries layout instead of a node map"
    )]
    NotAWorkflow {
        /// What was structurally wrong.
        detail: String,
    },

    /// A link referenced a node that does not exist.
    #[error(
        "ComfyUI workflow links to node '{node}', which the document does not define. \
         Why: a dangling link means the export is truncated or hand-edited, and the importer \
         will not guess what the missing node computed. \
         How to fix: re-export the workflow from ComfyUI"
    )]
    DanglingLink {
        /// The referenced node id.
        node: String,
    },

    /// No node consumes an image, so there is no output path to convert.
    #[error(
        "ComfyUI workflow has no image output node ({expected}). \
         Why: conversion walks backwards from the saved image, because that is the only part \
         of the graph that provably decides the result. \
         How to fix: connect the VAE decode to a SaveImage node and re-export"
    )]
    NoOutputPath {
        /// Sink classes that were looked for.
        expected: String,
    },

    /// The topology admits more than one reading.
    #[error("ambiguous ComfyUI topology: {detail}. How to fix: {remedy}")]
    AmbiguousTopology {
        /// What was ambiguous.
        detail: String,
        /// How to make the workflow unambiguous.
        remedy: String,
    },

    /// A node on the output path carries a fact the canonical IR cannot state.
    #[error("ComfyUI node {node} ({class}) cannot be represented: {detail}. How to fix: {remedy}")]
    Unrepresentable {
        /// Node id in the workflow document.
        node: String,
        /// ComfyUI `class_type`.
        class: String,
        /// What could not be represented.
        detail: String,
        /// How to change the workflow so it can be.
        remedy: String,
    },

    /// A node class on the output path is not modeled by this importer.
    #[error(
        "ComfyUI node {node} has unsupported class '{class}'. \
         Why: {remedy}. \
         How to fix: conversion is fail-closed, so a node whose semantics the importer cannot \
         state is refused rather than dropped. Replace it with a modeled class, remove it from \
         the path that produces the saved image, or add support for it before importing"
    )]
    UnknownNode {
        /// Node id in the workflow document.
        node: String,
        /// ComfyUI `class_type`.
        class: String,
        /// Why the node matters to the produced image.
        remedy: String,
    },

    /// A recognized feature has no canonical representation yet.
    #[error(
        "ComfyUI workflow uses {feature}, which the canonical workflow contract does not carry: \
         {detail}. How to fix: {remedy}"
    )]
    UnsupportedFeature {
        /// Human-readable feature name.
        feature: String,
        /// Why it cannot be represented.
        detail: String,
        /// How to change the workflow or the package.
        remedy: String,
    },

    /// The emitted document did not satisfy the metadata contract.
    ///
    /// This is an importer bug, not a user error, and it is surfaced rather than
    /// swallowed so a malformed conversion can never reach a package directory.
    #[error(
        "the converted workflow is not valid inference metadata: {0}. \
         Why: the importer validates everything it emits, so an invalid document is a defect \
         in the converter rather than something a package should carry"
    )]
    InvalidMetadata(String),
}

/// How strict a conversion should be, and what package facts it may rely on.
#[derive(Debug, Clone, Default)]
pub struct ConvertOptions {
    /// Artifact layout of the package the emitted workflow will run against.
    pub layout: ComponentLayout,
    /// The package's own `adapters` block, required when the workflow selects
    /// LoRAs.
    ///
    /// Adapter identity, target bindings, and base-model fingerprints live in
    /// the ONNX package, not in a ComfyUI graph. The importer routes the
    /// workflow's LoRA *selection* through the canonical adapter inputs and
    /// checks it against this contract; it never fabricates a target manifest.
    pub adapters: Option<Value>,
}

/// What a conversion recovered, for a caller that wants to report it.
#[derive(Debug, Clone, PartialEq)]
pub struct ConversionReport {
    /// The normalized, Comfy-free plan the metadata was lowered from.
    pub plan: WorkflowPlan,
    /// Nodes that cannot reach the image sink and were therefore not converted.
    pub ignored_nodes: Vec<String>,
    /// LoRA identities routed through the package's adapter contract.
    pub adapters: Vec<String>,
}

/// Convert a parsed ComfyUI workflow document into canonical metadata.
///
/// Returns the typed [`InferenceMetadata`], the JSON document it was parsed from
/// (metadata itself is deserialize-only), and a [`ConversionReport`].
pub fn convert(
    workflow: &Value,
    options: &ConvertOptions,
) -> Result<(InferenceMetadata, Value, ConversionReport), ComfyUiConfigError> {
    let graph = ComfyGraph::from_value(workflow)?;
    let plan = recognize(&graph)?;
    lower::supported_prediction(plan.prediction, plan.solver)?;

    let adapters = resolve_adapters(&plan, options)?;
    let mut lowering = lower::Lowering::new(&plan, &options.layout);
    if let Some((_, max_adapters)) = adapters.as_ref() {
        lowering.declare_adapter_selection(*max_adapters);
    }
    let document = lowering.build(adapters.as_ref().map(|(value, _)| value))?;

    let metadata: InferenceMetadata = serde_json::from_value(document.clone())?;
    onnx_genai_metadata::validate_metadata(&metadata)
        .map_err(|errors| ComfyUiConfigError::InvalidMetadata(errors.join("; ")))?;

    let report = ConversionReport {
        ignored_nodes: plan.ignored_nodes.clone(),
        adapters: plan.loras.iter().map(|lora| lora.name.clone()).collect(),
        plan,
    };
    Ok((metadata, document, report))
}

/// Convert a ComfyUI workflow from a JSON string.
pub fn convert_str(
    json: &str,
    options: &ConvertOptions,
) -> Result<(InferenceMetadata, Value, ConversionReport), ComfyUiConfigError> {
    let workflow: Value = serde_json::from_str(json)?;
    convert(&workflow, options)
}

/// Convert a ComfyUI workflow JSON file.
pub fn convert_file(
    path: impl AsRef<Path>,
    options: &ConvertOptions,
) -> Result<(InferenceMetadata, Value, ConversionReport), ComfyUiConfigError> {
    let text = std::fs::read_to_string(path)?;
    convert_str(&text, options)
}

/// Serialize a converted metadata document as the YAML a package carries.
///
/// Conversion is deterministic, so the same workflow and options always produce
/// byte-identical YAML. That is what makes a checked-in golden document a real
/// regression test rather than a snapshot that drifts.
pub fn to_yaml(document: &Value) -> Result<String, ComfyUiConfigError> {
    serde_yaml::to_string(document)
        .map_err(|error| ComfyUiConfigError::InvalidMetadata(error.to_string()))
}

/// Resolve the workflow's LoRA selection against the package adapter contract.
///
/// Returns the adapter block to embed and the fixed selection width, or `None`
/// when the workflow selects no LoRA at all.
fn resolve_adapters(
    plan: &WorkflowPlan,
    options: &ConvertOptions,
) -> Result<Option<(Value, usize)>, ComfyUiConfigError> {
    if plan.loras.is_empty() {
        return Ok(None);
    }
    let Some(adapters) = options.adapters.as_ref() else {
        return Err(ComfyUiConfigError::UnsupportedFeature {
            feature: format!(
                "{} LoRA selection(s) ({})",
                plan.loras.len(),
                plan.loras
                    .iter()
                    .map(|lora| lora.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            detail: "a ComfyUI graph names a LoRA file, but canonical adapter metadata needs the \
                     artifact identity, its base-model fingerprint, and the exact ONNX \
                     initializer each factor binds to. None of that exists in the workflow \
                     document, and inventing it would produce a package that claims bindings it \
                     cannot honor"
                .to_owned(),
            remedy: "pass the package's own `adapters` contract (the block its exporter wrote) \
                     so the importer can route this selection through it, or remove the LoRA \
                     loaders from the workflow"
                .to_owned(),
        });
    };

    let declared: Vec<String> = adapters
        .get("artifacts")
        .and_then(Value::as_object)
        .map(|artifacts| artifacts.keys().cloned().collect())
        .unwrap_or_default();
    let mut missing = Vec::new();
    for lora in &plan.loras {
        if !declared
            .iter()
            .any(|name| adapter_matches(name, &lora.name))
        {
            missing.push(lora.name.clone());
        }
    }
    if !missing.is_empty() {
        return Err(ComfyUiConfigError::UnsupportedFeature {
            feature: format!("LoRA selection of {}", missing.join(", ")),
            detail: format!(
                "the supplied adapter contract declares {}, so the workflow selects an adapter \
                 the package cannot apply",
                if declared.is_empty() {
                    "no artifacts".to_owned()
                } else {
                    declared.join(", ")
                }
            ),
            remedy: "declare the missing adapter in the package's `adapters.artifacts` block, or \
                     remove the LoRA loader from the workflow"
                .to_owned(),
        });
    }

    let max_adapters = adapters
        .get("selection")
        .and_then(|selection| selection.get("max_adapters"))
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .unwrap_or(plan.loras.len())
        .max(plan.loras.len());

    // The selection inputs are the canonical request-scoped adapter ABI, so the
    // emitted contract must name the SSA values this importer declares.
    let mut adapters = adapters.clone();
    if let Some(object) = adapters.as_object_mut() {
        object.insert(
            "selection".to_owned(),
            serde_json::json!({
                "segments": "request.adapter_segments",
                "adapter_counts": "request.adapter_counts",
                "scales": "request.adapter_scales",
                "max_adapters": max_adapters,
            }),
        );
        object
            .entry("application_capability")
            .or_insert_with(|| Value::String("onnx-genai.adapters@1".to_owned()));
    }
    Ok(Some((adapters, max_adapters)))
}

/// Whether a declared artifact key names the LoRA the workflow selected.
///
/// ComfyUI names a file (`detail.safetensors`); a package names an identity
/// (`detail`). Matching the stem keeps both spellings working without letting an
/// unrelated adapter match.
fn adapter_matches(declared: &str, selected: &str) -> bool {
    let stem = |value: &str| {
        value
            .rsplit('/')
            .next()
            .unwrap_or(value)
            .trim_end_matches(".safetensors")
            .trim_end_matches(".ckpt")
            .trim_end_matches(".pt")
            .to_owned()
    };
    declared == selected || stem(declared) == stem(selected)
}

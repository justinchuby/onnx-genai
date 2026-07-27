//! Integration test: run the runtime default optimizer pipeline over a real ONNX model
//! (`bert_toy`, loaded via `onnx-runtime-loader`) and assert the passes fire
//! and preserve graph validity.
//!
//! This proves the device-independent passes work on a 384-node real model, not
//! just hand-built fixtures.
//!
//! Operator fusion is now provider-scoped, so this test only asserts the generic
//! runtime passes. [`OpFusion`] has its own unit coverage and is scheduled by EPs
//! that own the fused kernels.

use std::path::Path;

use onnx_runtime_ir::Graph;
use onnx_runtime_optimizer::{PassContext, default_passes, run_passes};

fn model_path() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../onnx-runtime-session/tests/fixtures/bert_toy/model.onnx.textproto")
}

fn count(g: &Graph, op: &str) -> usize {
    g.nodes.values().filter(|n| n.op_type == op).count()
}

#[test]
fn pipeline_folds_constants_without_provider_fusion_on_bert_toy() {
    let path = model_path();
    if !path.exists() {
        eprintln!("skipping: {} not present", path.display());
        return;
    }

    let mut g = onnx_runtime_loader::load_model(&path).expect("load bert_toy");
    let nodes_before = g.num_nodes();
    let const_before = count(&g, "Constant");
    let matmul_before = count(&g, "MatMul");
    assert!(
        const_before > 0,
        "fixture should have Constant nodes to fold"
    );
    assert!(
        matmul_before > 0,
        "fixture should have MatMul nodes to fuse"
    );
    assert!(g.validate().is_ok(), "loaded graph must be valid");

    run_passes(&mut g, &default_passes(), &PassContext::new()).expect("pipeline runs");

    // Constant folding materialized every Constant node into an initializer.
    assert_eq!(count(&g, "Constant"), 0, "all Constants should be folded");
    assert_eq!(
        count(&g, "FusedMatMulBias"),
        0,
        "provider-scoped MatMul+Add fusion must not run in default_passes"
    );
    assert_eq!(
        count(&g, "MatMul"),
        matmul_before,
        "default_passes must preserve the provider-claimable op surface"
    );
    // The pipeline is a net simplification.
    assert!(g.num_nodes() < nodes_before, "node count should decrease");
    // And the result is still a structurally valid graph.
    assert!(g.validate().is_ok(), "optimized graph must remain valid");
}

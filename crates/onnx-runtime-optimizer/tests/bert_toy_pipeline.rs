//! Integration test: run the optimizer pipeline over a real ONNX model
//! (`bert_toy`, loaded via `onnx-runtime-loader`) and assert the passes fire
//! and preserve graph validity.
//!
//! This proves the device-independent passes work on a 384-node real model, not
//! just hand-built fixtures.
//!
//! ## Note on LayerNorm fusion
//!
//! `bert_toy`'s LayerNorm decomposition shares the `mean` (first `ReduceMean`)
//! across **two** `Sub` nodes (the variance branch and the numerator branch),
//! so it is a DAG, not the idealized linear 9-op chain in `docs/ORT2.md` §18.2.
//! The [`OpFusion`] safety rule correctly *declines* to fuse it, because the
//! shared `mean` value escapes any linear match. A DAG-aware LayerNorm matcher
//! is deferred (Phase 2b). The `MatMul+Add → FusedMatMulBias` fusion, whose
//! intermediates are single-consumer, does fire on this model.

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
fn pipeline_folds_constants_and_fuses_matmul_bias_on_bert_toy() {
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
    // Op fusion collapsed MatMul+Add spines into FusedMatMulBias.
    let fused = count(&g, "FusedMatMulBias");
    assert!(fused > 0, "MatMul+Add fusion should fire on a real model");
    assert!(
        count(&g, "MatMul") < matmul_before,
        "MatMul count should drop"
    );
    // The pipeline is a net simplification.
    assert!(g.num_nodes() < nodes_before, "node count should decrease");
    // And the result is still a structurally valid graph.
    assert!(g.validate().is_ok(), "optimized graph must remain valid");
}

//! End-to-end native-LoRA test for Phase-1 item **P4** (design
//! `docs/NATIVE_LORA_DESIGN.md` §D/§F). Exercises the *whole* engine path:
//!
//!   PEFT adapter directory
//!     → [`LoraManager::load`] (P2a loader + LRU)
//!     → [`LoraManager::spec`] (the PEFT → session-spec **bridge**)
//!     → [`SessionBuilder::lora_adapter`] (build-time injection, P2b/P3)
//!     → [`InferenceSession::set_lora_active`] (activation feed)
//!     → [`InferenceSession::run`] (decode step).
//!
//! # What is real and what is a deliberate simplification (honesty)
//!
//! The base projection is a real `MatMulNBits` int4/block-32 node executed by
//! the CPU execution provider — the same op a real export uses — but its packed
//! nibbles all equal the affine zero point, so it dequantizes to the zero
//! weight and the base output is the zero vector. That makes the observable
//! decode output isolate the adapter delta exactly, so this test can assert
//! the *activation semantics* (active applies the delta, deactivate restores
//! base, re-activation leaves no residual) without depending on a full int4
//! GEMM golden. The numeric `Y == Y_base + scale·((x·A_t)·B_t)` golden with a
//! *non-zero* base already exists in the `lora_inject` unit tests; a true
//! HF-parity end-to-end (real Qwen int4 base + a matching real PEFT adapter +
//! expected logits) needs assets not present in this repository and is tracked
//! as a follow-up, not fabricated here.

#![cfg(feature = "native-backend")]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use onnx_genai_engine::lora::manager::LoraManager;
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, TensorData, WeightRef, static_shape,
};
use onnx_runtime_loader::encoder::{Model, write_model};
use onnx_runtime_session::{InferenceSession, Tensor};
use safetensors::Dtype;
use safetensors::tensor::{TensorView, serialize_to_file};

const F32: DataType = DataType::Float32;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn f32_from_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Write a minimal single-`q_proj` PEFT adapter into a named subdirectory, so
/// the loaded adapter's `name` is the deterministic directory leaf. `A` is
/// `[r, k]` and `B` is `[n, r]` in PEFT (fan-out) orientation; the loader
/// transposes them into the `A_t = [k, r]` / `B_t = [r, n]` MatMul orientation.
fn write_peft_adapter(
    root: &Path,
    name: &str,
    r: usize,
    k: usize,
    n: usize,
    a_values: &[f32],
    b_values: &[f32],
    alpha: usize,
) -> PathBuf {
    let directory = root.join(name);
    fs::create_dir(&directory).unwrap();
    fs::write(
        directory.join("adapter_config.json"),
        format!(
            r#"{{"r": {r}, "lora_alpha": {alpha}, "target_modules": ["q_proj"], "fan_in_fan_out": false}}"#
        ),
    )
    .unwrap();
    let a = f32_bytes(a_values);
    let b = f32_bytes(b_values);
    let a_view = TensorView::new(Dtype::F32, vec![r, k], &a).unwrap();
    let b_view = TensorView::new(Dtype::F32, vec![n, r], &b).unwrap();
    let mut views = HashMap::new();
    views.insert(
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight".to_owned(),
        a_view,
    );
    views.insert(
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight".to_owned(),
        b_view,
    );
    serialize_to_file(&views, None, &directory.join("adapter_model.safetensors")).unwrap();
    directory
}

/// Build a synthetic decoder graph with a single layer-0 `q_proj` int4 base
/// projection whose weight dequantizes to zero (every packed nibble equals the
/// default affine zero point of `1 << (bits - 1) = 8`). The graph input is the
/// activation `x[1, k]`; the graph output is the projection `y[1, n]`.
fn write_zero_base_model(path: &Path, k: usize, n: usize) {
    const BITS: usize = 4;
    const BLOCK_SIZE: usize = 16;
    assert!(k % BLOCK_SIZE == 0, "test uses whole int4 blocks");
    let k_blocks = k / BLOCK_SIZE;
    let blob_size = BLOCK_SIZE * BITS / 8;

    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    graph.opset_imports.insert("com.microsoft".to_string(), 1);

    let x = graph.create_named_value("x", F32, static_shape([1, k]));
    graph.add_input(x);

    // Packed weight: every nibble = 8 (byte 0x88) ⇒ dequant (8 - 8)·scale = 0.
    let weight = graph.create_named_value(
        "model.layers.0.attn.q_proj.MatMulNBits.qweight",
        DataType::Uint8,
        static_shape([n, k_blocks, blob_size]),
    );
    graph.set_initializer(
        weight,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Uint8,
            vec![n, k_blocks, blob_size],
            vec![0x88u8; n * k_blocks * blob_size],
        )),
    );
    let scales = graph.create_named_value(
        "model.layers.0.attn.q_proj.MatMulNBits.scales",
        F32,
        static_shape([n, k_blocks]),
    );
    graph.set_initializer(
        scales,
        WeightRef::Inline(TensorData::from_raw(
            F32,
            vec![n, k_blocks],
            f32_bytes(&vec![1.0f32; n * k_blocks]),
        )),
    );

    let y = graph.create_named_value("y", F32, static_shape([1, n]));
    let mut node = Node::new(
        NodeId(0),
        "MatMulNBits",
        vec![Some(x), Some(weight), Some(scales)],
        vec![y],
    );
    node.name = "/model/layers.0/attn/q_proj/MatMulNBits".to_string();
    node.domain = "com.microsoft".to_string();
    node.attributes.insert("K".to_string(), Attribute::Int(k as i64));
    node.attributes.insert("N".to_string(), Attribute::Int(n as i64));
    node.attributes.insert("bits".to_string(), Attribute::Int(BITS as i64));
    node.attributes
        .insert("block_size".to_string(), Attribute::Int(BLOCK_SIZE as i64));
    graph.insert_node(node);
    graph.add_output(y);

    write_model(&Model::new(&graph), path).unwrap();
}

/// Reference delta `scale · ((x · A_t) · B_t)` from the loaded spec factors.
fn reference_delta(
    x: &[f32],
    a_t: &[f32],
    b_t: &[f32],
    k: usize,
    r: usize,
    n: usize,
    scale: f32,
) -> Vec<f32> {
    let mut mid = vec![0.0f64; r];
    for j in 0..r {
        let mut acc = 0.0f64;
        for p in 0..k {
            acc += x[p] as f64 * a_t[p * r + j] as f64;
        }
        mid[j] = acc;
    }
    let mut delta = vec![0.0f32; n];
    for j in 0..n {
        let mut acc = 0.0f64;
        for p in 0..r {
            acc += mid[p] * b_t[p * n + j] as f64;
        }
        delta[j] = (acc * scale as f64) as f32;
    }
    delta
}

#[test]
fn engine_lora_path_applies_and_reverts_adapter_delta() {
    let (k, n, r) = (16usize, 4usize, 2usize);
    let alpha = 2 * r; // scale = alpha / r = 2.0

    // Distinct, non-degenerate factors so the delta is unmistakably non-zero.
    let a_values: Vec<f32> = (0..r * k).map(|i| (i as f32) * 0.03 - 0.2).collect();
    let b_values: Vec<f32> = (0..n * r).map(|i| (i as f32) * -0.05 + 0.15).collect();

    let root = tempfile::tempdir().unwrap();
    let adapter_directory =
        write_peft_adapter(root.path(), "demo_adapter", r, k, n, &a_values, &b_values, alpha);
    let model_path = root.path().join("model.onnx");
    write_zero_base_model(&model_path, k, n);

    // Engine side: load through the manager and derive the injection spec.
    let mut manager = LoraManager::with_budget(0);
    let adapter_id = manager.load(&adapter_directory).expect("load PEFT adapter");
    let spec = manager.spec(&adapter_id).expect("build injection spec");
    manager.activate(&adapter_id).expect("activate adapter");
    assert_eq!(manager.active(), Some(&adapter_id));

    // Authoritative delta from the loader-produced (transposed) factors.
    let module = &spec.modules[0];
    let a_t = f32_from_bytes(&module.a_t.data);
    let b_t = f32_from_bytes(&module.b_t.data);
    let x: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1 - 0.5).collect();
    let delta = reference_delta(&x, &a_t, &b_t, k, r, n, module.scale);

    // Real session build: build-time injection wires the overridable A_t/B_t
    // inputs and installs their feeds; activation is a separate toggle.
    let mut session = InferenceSession::builder()
        .model(&model_path)
        .lora_adapter(spec)
        .build()
        .expect("build native session with injected adapter");

    let x_tensor = Tensor::from_f32(&[1, k], &x).unwrap();

    // Active ⇒ output is the adapter delta (base is the zero projection).
    session.set_lora_active(true);
    assert!(session.lora_active());
    let active = session.run(&[("x", &x_tensor)]).unwrap()[0].to_vec_f32();
    assert_eq!(active.len(), n);
    for (got, expected) in active.iter().zip(&delta) {
        assert!(
            (got - expected).abs() < 1e-3,
            "active output {active:?} should equal the delta {delta:?}"
        );
    }
    assert!(
        active.iter().any(|value| value.abs() > 1e-3),
        "the delta must be non-trivial, got {active:?}"
    );

    // Deactivate ⇒ base-only, and the base projection is exactly zero.
    session.set_lora_active(false);
    assert!(!session.lora_active());
    let base = session.run(&[("x", &x_tensor)]).unwrap()[0].to_vec_f32();
    assert_eq!(base, vec![0.0f32; n], "deactivated => base-only, exactly");

    // Re-activate ⇒ the same delta, proving no residual state leaked across the
    // deactivate/reactivate cycle or the repeated run.
    session.set_lora_active(true);
    let active_again = session.run(&[("x", &x_tensor)]).unwrap()[0].to_vec_f32();
    assert_eq!(
        active, active_again,
        "re-activation must reproduce the delta bit-for-bit"
    );
}

//! Finding #1 regression guard: grouped-adapter admission is governed by the
//! shared cross-session [`ByteBudget`] (design `docs/NATIVE_LORA_DESIGN.md`
//! §J.2 control plane).
//!
//! These tests drive the *whole* production wiring — the engine's
//! [`BudgetedLoraPool`] used as a [`LoraPoolSink`] through the grouped injection
//! pass — over a real `ByteBudget`, proving:
//!
//!   1. an over-budget adapter set fails loud (a typed
//!      [`LoraInjectError::PoolBudgetExceeded`]) and leaks no bytes, and
//!   2. a successful load reserves the adapter bytes from the shared budget and
//!      releases them back to the exact prior level when the session (here the
//!      injection that owns the pool) drops.

#![cfg(feature = "native-backend")]

use onnx_genai_engine::lora::pool::BudgetedLoraPool;
use onnx_genai_scheduler::ByteBudget;
use onnx_runtime_ep_api::{AdapterId, LoraPoolSink};
use onnx_runtime_ir::{DataType, Graph, Node, NodeId, TensorData, ValueId, WeightRef, static_shape};
use onnx_runtime_session::lora_inject::{
    inject_grouped_multi, LoraAdapterSpec, LoraInjectError, LoraManifest, LoraModuleSpec, Placement,
    TargetEntry,
};

const F32: DataType = DataType::Float32;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

/// A minimal fp32 base `MatMul` graph, mirroring the session-crate grouped-LoRA
/// test fixtures: `x[m,k] @ W[k,n] -> base[m,n]`.
fn base_matmul_graph(m: usize, k: usize, n: usize) -> (Graph, NodeId, ValueId, ValueId) {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let x = graph.create_named_value("x", F32, static_shape([m, k]));
    graph.add_input(x);
    let weight = graph.create_named_value("W", F32, static_shape([k, n]));
    graph.set_initializer(
        weight,
        WeightRef::Inline(TensorData::from_raw(F32, vec![k, n], f32_bytes(&vec![0.0; k * n]))),
    );
    let base = graph.create_named_value("base", F32, static_shape([m, n]));
    let node_id = graph.insert_node(Node::new(
        NodeId(0),
        "MatMul",
        vec![Some(x), Some(weight)],
        vec![base],
    ));
    graph.add_output(base);
    (graph, node_id, x, base)
}

fn direct_manifest(node_id: NodeId, base: ValueId, x: ValueId, k: usize, n: usize) -> LoraManifest {
    LoraManifest {
        entries: vec![TargetEntry {
            semantic: "layers.0.q_proj".to_string(),
            node_id,
            base_output: base,
            activation: x,
            k,
            n,
            dtype: F32,
            placement: Placement::Direct,
        }],
    }
}

fn q_proj_adapter(name: &str, k: usize, n: usize, r: usize) -> LoraAdapterSpec {
    let a: Vec<f32> = (0..k * r).map(|i| (i as f32) * 0.1 - 0.3).collect();
    let b: Vec<f32> = (0..r * n).map(|i| (i as f32) * -0.05 + 0.2).collect();
    LoraAdapterSpec {
        name: name.to_string(),
        modules: vec![LoraModuleSpec {
            module_name: "self_attn.q_proj".to_string(),
            layer_index: 0,
            rank: r,
            scale: 1.0,
            a_t: TensorData::from_raw(F32, vec![k, r], f32_bytes(&a)),
            b_t: TensorData::from_raw(F32, vec![r, n], f32_bytes(&b)),
        }],
    }
}

#[test]
fn grouped_admission_over_budget_fails_loud_and_leaks_nothing() {
    let (k, n, r) = (4usize, 3usize, 2usize);
    let (mut graph, node_id, x_v, base_v) = base_matmul_graph(2, k, n);
    let manifest = direct_manifest(node_id, base_v, x_v, k, n);
    let adapter_a = q_proj_adapter("alpha", k, n, r);
    let adapter_b = q_proj_adapter("beta", k, n, r);

    // A ceiling far smaller than one aligned page pair (128 B): the very first
    // admission must be rejected by the shared budget.
    let budget = ByteBudget::new(64);
    let sink = Box::new(BudgetedLoraPool::new(budget.clone())) as Box<dyn LoraPoolSink>;
    let result = inject_grouped_multi(
        &mut graph,
        &manifest,
        &[(AdapterId(0), &adapter_a), (AdapterId(1), &adapter_b)],
        Some(sink),
    );

    match result {
        Err(LoraInjectError::PoolBudgetExceeded {
            requested,
            available,
            ..
        }) => {
            assert!(requested > available, "rejection reports the true shortfall");
        }
        Err(other) => panic!("expected a fail-loud PoolBudgetExceeded, got {other:?}"),
        Ok(_) => panic!("grouped admission must exceed the shared byte budget"),
    }
    assert_eq!(
        budget.used(),
        0,
        "a rejected grouped admission must leak no reserved bytes"
    );
}

#[test]
fn grouped_admission_reserves_and_releases_shared_budget_on_drop() {
    let (k, n, r) = (4usize, 3usize, 2usize);
    let (mut graph, node_id, x_v, base_v) = base_matmul_graph(2, k, n);
    let manifest = direct_manifest(node_id, base_v, x_v, k, n);
    let adapter_a = q_proj_adapter("alpha", k, n, r);
    let adapter_b = q_proj_adapter("beta", k, n, r);

    // A generous ceiling with a pre-existing baseline reservation, so we can
    // assert the budget returns to *exactly* the prior level (not merely zero).
    let budget = ByteBudget::new(1 << 20);
    budget.try_reserve(4096).expect("baseline device reservation");
    let used_before = budget.used();

    let sink = Box::new(BudgetedLoraPool::new(budget.clone())) as Box<dyn LoraPoolSink>;
    let injection = inject_grouped_multi(
        &mut graph,
        &manifest,
        &[(AdapterId(0), &adapter_a), (AdapterId(1), &adapter_b)],
        Some(sink),
    )
    .expect("grouped admission fits the shared budget");

    let used_with_adapters = budget.used();
    assert!(
        used_with_adapters > used_before,
        "grouped adapter residency must be charged against the shared budget \
         (before={used_before}, with_adapters={used_with_adapters})"
    );

    // Dropping the injection drops the last pool `Arc`, whose residency owner is
    // the budget reservation — releasing exactly the admitted bytes, once.
    drop(injection);
    assert_eq!(
        budget.used(),
        used_before,
        "session drop must release the adapter bytes back to the exact prior level"
    );
}

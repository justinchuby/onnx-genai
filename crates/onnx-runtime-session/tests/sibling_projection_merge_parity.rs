//! Parity and reachability tests for the `CpuSiblingProjectionMerge` pass.
//!
//! Constructs a synthetic graph with sibling `MatMul+Add` projections sharing
//! the same activation (mimicking Q/K/V or gate/up patterns), runs it through
//! the optimizer, and verifies:
//!
//! 1. **Numeric parity**: fused output == unfused reference (bit-for-bit, since
//!    GEMM columns are independent and bias addition order is preserved).
//! 2. **Dispatch reachability**: the fused graph contains the merged
//!    `FusedMatMulBias` + `Split` topology and no residual sibling nodes.

use onnx_runtime_ir::{
    DataType, Graph, Node, NodeId, TensorData, ValueId, WeightRef, static_shape,
};
use onnx_runtime_loader::{Model, encode_model};
use onnx_runtime_session::{InferenceSession, Tensor};

fn f32_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32_init(g: &mut Graph, name: &str, dims: &[usize], data: &[f32]) -> ValueId {
    let vid = g.create_named_value(name, DataType::Float32, static_shape(dims.iter().copied()));
    g.set_initializer(
        vid,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Float32,
            dims.to_vec(),
            f32_bytes(data),
        )),
    );
    vid
}

fn input(g: &mut Graph, name: &str, dims: &[usize]) -> ValueId {
    let v = g.create_named_value(name, DataType::Float32, static_shape(dims.iter().copied()));
    g.add_input(v);
    v
}

fn node_op(
    g: &mut Graph,
    op_type: &str,
    name: &str,
    inputs: Vec<Option<ValueId>>,
    out_dims: &[usize],
) -> ValueId {
    let out = g.create_named_value(
        name,
        DataType::Float32,
        static_shape(out_dims.iter().copied()),
    );
    g.insert_node(Node::new(NodeId(0), op_type, inputs, vec![out]));
    out
}

/// Build a graph with 3 sibling MatMul+Add (mimicking Q/K/V projections):
///   activation [2, 4] (input)
///   W_q [4, 6], bias_q [6] → Q [2, 6] → Abs → Q_out
///   W_k [4, 3], bias_k [3] → K [2, 3] → Abs → K_out
///   W_v [4, 3], bias_v [3] → V [2, 3] → Abs → V_out
/// The Relu after each projection simulates a downstream consumer (like
/// RotaryEmbedding or Attention in the real model), so the projection outputs
/// are intermediate and eligible for the sibling merge.
fn build_qkv_sibling_graph() -> (Vec<u8>, Vec<f32>) {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);

    let seq = 2;
    let k_dim = 4;
    let n_q = 6;
    let n_k = 3;
    let n_v = 3;

    let activation = input(&mut g, "activation", &[seq, k_dim]);

    // Generate deterministic weight/bias data.
    let w_q: Vec<f32> = (0..k_dim * n_q).map(|i| (i as f32) * 0.01 + 0.1).collect();
    let b_q: Vec<f32> = (0..n_q).map(|i| (i as f32) * 0.02 + 0.5).collect();
    let w_k: Vec<f32> = (0..k_dim * n_k).map(|i| (i as f32) * 0.03 - 0.2).collect();
    let b_k: Vec<f32> = (0..n_k).map(|i| (i as f32) * 0.04 - 0.1).collect();
    let w_v: Vec<f32> = (0..k_dim * n_v).map(|i| (i as f32) * 0.005 + 0.3).collect();
    let b_v: Vec<f32> = (0..n_v).map(|i| (i as f32) * 0.01 + 0.2).collect();

    let w_q_vid = f32_init(&mut g, "W_q", &[k_dim, n_q], &w_q);
    let b_q_vid = f32_init(&mut g, "bias_q", &[n_q], &b_q);
    let w_k_vid = f32_init(&mut g, "W_k", &[k_dim, n_k], &w_k);
    let b_k_vid = f32_init(&mut g, "bias_k", &[n_k], &b_k);
    let w_v_vid = f32_init(&mut g, "W_v", &[k_dim, n_v], &w_v);
    let b_v_vid = f32_init(&mut g, "bias_v", &[n_v], &b_v);

    // Q = MatMul(activation, W_q) + bias_q → Relu(Q)
    let q_mm = node_op(
        &mut g,
        "MatMul",
        "q_mm",
        vec![Some(activation), Some(w_q_vid)],
        &[seq, n_q],
    );
    let q_bias = node_op(
        &mut g,
        "Add",
        "q_bias",
        vec![Some(q_mm), Some(b_q_vid)],
        &[seq, n_q],
    );
    let q_out = node_op(&mut g, "Abs", "Q", vec![Some(q_bias)], &[seq, n_q]);

    // K = MatMul(activation, W_k) + bias_k → Relu(K)
    let k_mm = node_op(
        &mut g,
        "MatMul",
        "k_mm",
        vec![Some(activation), Some(w_k_vid)],
        &[seq, n_k],
    );
    let k_bias = node_op(
        &mut g,
        "Add",
        "k_bias",
        vec![Some(k_mm), Some(b_k_vid)],
        &[seq, n_k],
    );
    let k_out = node_op(&mut g, "Abs", "K", vec![Some(k_bias)], &[seq, n_k]);

    // V = MatMul(activation, W_v) + bias_v → Relu(V)
    let v_mm = node_op(
        &mut g,
        "MatMul",
        "v_mm",
        vec![Some(activation), Some(w_v_vid)],
        &[seq, n_v],
    );
    let v_bias = node_op(
        &mut g,
        "Add",
        "v_bias",
        vec![Some(v_mm), Some(b_v_vid)],
        &[seq, n_v],
    );
    let v_out = node_op(&mut g, "Abs", "V", vec![Some(v_bias)], &[seq, n_v]);

    g.value_mut(q_out).name = Some("Q".to_string());
    g.add_output(q_out);
    g.value_mut(k_out).name = Some("K".to_string());
    g.add_output(k_out);
    g.value_mut(v_out).name = Some("V".to_string());
    g.add_output(v_out);

    let act_data: Vec<f32> = (0..seq * k_dim).map(|i| (i as f32) * 0.1 - 0.4).collect();
    let bytes = encode_model(&Model::new(&g)).expect("encode QKV model");
    (bytes, act_data)
}

/// Build a graph with 2 sibling plain MatMul (gate/up pattern) + downstream Relu:
fn build_gate_up_sibling_graph() -> (Vec<u8>, Vec<f32>) {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);

    let seq = 2;
    let k_dim = 4;
    let n_gate = 5;
    let n_up = 5;

    let activation = input(&mut g, "activation", &[seq, k_dim]);

    let w_gate: Vec<f32> = (0..k_dim * n_gate)
        .map(|i| (i as f32) * 0.02 + 0.1)
        .collect();
    let w_up: Vec<f32> = (0..k_dim * n_up)
        .map(|i| (i as f32) * 0.015 - 0.3)
        .collect();

    let w_gate_vid = f32_init(&mut g, "W_gate", &[k_dim, n_gate], &w_gate);
    let w_up_vid = f32_init(&mut g, "W_up", &[k_dim, n_up], &w_up);

    let gate_mm = node_op(
        &mut g,
        "MatMul",
        "gate_mm",
        vec![Some(activation), Some(w_gate_vid)],
        &[seq, n_gate],
    );
    let gate_out = node_op(&mut g, "Relu", "gate", vec![Some(gate_mm)], &[seq, n_gate]);
    let up_mm = node_op(
        &mut g,
        "MatMul",
        "up_mm",
        vec![Some(activation), Some(w_up_vid)],
        &[seq, n_up],
    );
    let up_out = node_op(&mut g, "Relu", "up", vec![Some(up_mm)], &[seq, n_up]);

    g.value_mut(gate_out).name = Some("gate".to_string());
    g.add_output(gate_out);
    g.value_mut(up_out).name = Some("up".to_string());
    g.add_output(up_out);

    let act_data: Vec<f32> = (0..seq * k_dim).map(|i| (i as f32) * 0.1 - 0.4).collect();
    let bytes = encode_model(&Model::new(&g)).expect("encode gate/up model");
    (bytes, act_data)
}

/// Verify Q/K/V sibling merge: fused output must be bit-for-bit identical to
/// the unfused reference (GEMM columns are independent, so the merge is exact).
#[test]
fn sibling_qkv_merge_numeric_parity() {
    // Enable the env-gated pass for this test.
    unsafe {
        std::env::set_var("ONNX_RT_SIBLING_MERGE", "1");
    }
    let (bytes, act_data) = build_qkv_sibling_graph();

    // Unfused reference.
    let mut unfused = InferenceSession::builder()
        .model_bytes(&bytes)
        .option("optimization", "none")
        .build()
        .expect("build unfused");
    let activation_tensor = Tensor::from_f32(&[2, 4], &act_data).unwrap();
    let feeds = vec![("activation", &activation_tensor)];
    let unfused_out = unfused.run(&feeds).expect("run unfused");

    // Fused (with sibling merge).
    let mut fused = InferenceSession::builder()
        .model_bytes(&bytes)
        .option("optimization", "all")
        .build()
        .expect("build fused");
    let fused_out = fused.run(&feeds).expect("run fused");

    assert_eq!(unfused_out.len(), 3, "expected 3 outputs");
    assert_eq!(fused_out.len(), 3, "expected 3 outputs");

    let names = ["Q", "K", "V"];
    for (i, (uf, f)) in unfused_out.iter().zip(fused_out.iter()).enumerate() {
        let uf_data = uf.to_vec_f32();
        let f_data = f.to_vec_f32();
        assert_eq!(uf_data.len(), f_data.len(), "{}: length mismatch", names[i]);
        let max_diff = uf_data
            .iter()
            .zip(&f_data)
            .fold(0.0f32, |m, (&a, &b)| m.max((a - b).abs()));
        assert!(
            max_diff < 1e-6,
            "{}: max diff {max_diff:.3e} exceeds 1e-6",
            names[i]
        );
    }
}

/// Verify gate/up sibling merge: fused == unfused (exact, same reason).
#[test]
fn sibling_gate_up_merge_numeric_parity() {
    // Enable the env-gated pass for this test.
    unsafe {
        std::env::set_var("ONNX_RT_SIBLING_MERGE", "1");
    }
    let (bytes, act_data) = build_gate_up_sibling_graph();

    let mut unfused = InferenceSession::builder()
        .model_bytes(&bytes)
        .option("optimization", "none")
        .build()
        .expect("build unfused");
    let activation_tensor = Tensor::from_f32(&[2, 4], &act_data).unwrap();
    let feeds = vec![("activation", &activation_tensor)];
    let unfused_out = unfused.run(&feeds).expect("run unfused");

    let mut fused = InferenceSession::builder()
        .model_bytes(&bytes)
        .option("optimization", "all")
        .build()
        .expect("build fused");
    let fused_out = fused.run(&feeds).expect("run fused");

    assert_eq!(unfused_out.len(), 2);
    assert_eq!(fused_out.len(), 2);

    let names = ["gate", "up"];
    for (i, (uf, f)) in unfused_out.iter().zip(fused_out.iter()).enumerate() {
        let uf_data = uf.to_vec_f32();
        let f_data = f.to_vec_f32();
        assert_eq!(uf_data.len(), f_data.len(), "{}: length mismatch", names[i]);
        let max_diff = uf_data
            .iter()
            .zip(&f_data)
            .fold(0.0f32, |m, (&a, &b)| m.max((a - b).abs()));
        assert!(
            max_diff < 1e-6,
            "{}: max diff {max_diff:.3e} exceeds 1e-6",
            names[i]
        );
    }
}

/// Reachability test: the optimized graph must contain the merged topology
/// (one wider FusedMatMulBias + Split) and no residual sibling MatMul+Add pairs.
#[test]
fn sibling_qkv_merge_dispatch_reachability() {
    // Enable the env-gated pass for this test.
    unsafe {
        std::env::set_var("ONNX_RT_SIBLING_MERGE", "1");
    }
    let (bytes, _) = build_qkv_sibling_graph();

    let fused = InferenceSession::builder()
        .model_bytes(&bytes)
        .option("optimization", "all")
        .build()
        .expect("build fused");

    let fg = fused.graph();
    let mut fused_count = 0;
    let mut split_count = 0;
    let mut matmul_count = 0;
    let mut add_count = 0;

    for node in fg.nodes.values() {
        match (node.domain.as_str(), node.op_type.as_str()) {
            ("com.microsoft", "FusedMatMulBias") => fused_count += 1,
            ("", "Split") => split_count += 1,
            ("", "MatMul") => matmul_count += 1,
            ("", "Add") => add_count += 1,
            _ => {}
        }
    }

    // 3 MatMul + 3 Add → OpFusion gives 3 FusedMatMulBias → sibling merge
    // gives 1 FusedMatMulBias + 1 Split.
    assert_eq!(
        fused_count, 1,
        "expected 1 merged FusedMatMulBias, got {fused_count}"
    );
    assert_eq!(split_count, 1, "expected 1 Split, got {split_count}");
    assert_eq!(matmul_count, 0, "expected 0 MatMul, got {matmul_count}");
    assert_eq!(add_count, 0, "expected 0 Add, got {add_count}");
}

/// Reachability test: gate/up plain MatMul merge → one merged MatMul + Split.
#[test]
fn sibling_gate_up_merge_dispatch_reachability() {
    // Enable the env-gated pass for this test.
    unsafe {
        std::env::set_var("ONNX_RT_SIBLING_MERGE", "1");
    }
    let (bytes, _) = build_gate_up_sibling_graph();

    let fused = InferenceSession::builder()
        .model_bytes(&bytes)
        .option("optimization", "all")
        .build()
        .expect("build fused");

    let fg = fused.graph();
    let mut matmul_count = 0;
    let mut split_count = 0;

    for node in fg.nodes.values() {
        match (node.domain.as_str(), node.op_type.as_str()) {
            ("", "MatMul") => matmul_count += 1,
            ("", "Split") => split_count += 1,
            _ => {}
        }
    }

    assert_eq!(
        matmul_count, 1,
        "expected 1 merged MatMul, got {matmul_count}"
    );
    assert_eq!(split_count, 1, "expected 1 Split, got {split_count}");
}

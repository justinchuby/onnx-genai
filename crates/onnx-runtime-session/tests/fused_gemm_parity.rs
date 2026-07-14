//! Synthetic end-to-end parity for the `MatMul + Add + Relu → FusedGemm`
//! fusion (`docs/ORT2.md` §18.2).
//!
//! Unlike LayerNorm and `FusedMatMulBias`, the `FusedGemm` pattern is **not**
//! exercised by the `bert_toy` conformance model — that model's feed-forward
//! blocks use GELU/`Erf`, never `Relu`, so the optimizer never emits a
//! `FusedGemm` node when loading it. This test therefore constructs a small
//! `MatMul → Add → Relu` graph by hand (via the ONNX IR API), encodes it to
//! ONNX bytes, and runs it through the production `onnx-runtime-session`
//! pipeline TWO ways:
//!
//! * `optimization="none"` — the unfused reference (`MatMul` + `Add` + `Relu`
//!   standalone kernels);
//! * `optimization="all"` — the full device-independent pipeline, which fuses
//!   the three ops into a single `com.microsoft::FusedGemm` node backed by the
//!   CPU kernel.
//!
//! We assert the two outputs are numerically identical (both paths use the same
//! `matmul_dense` GEMM + broadcast-add + `max(0, x)`, so this is byte-identical,
//! bounded by a tight `1e-6` atol) AND that the `"all"` graph really fused: it
//! contains exactly one `FusedGemm` node and zero surviving standalone
//! `MatMul`/`Add`/`Relu` for the region — mirroring
//! `full_optimization_actually_fuses_layernorm_and_matmul_bias`.
//!
//! Nothing here special-cases any model: the graph is a generic fixture and
//! `"optimization"` is a generic, model-agnostic option.

use onnx_runtime_ir::{
    static_shape, DataType, Graph, Node, NodeId, TensorData, ValueId, WeightRef,
};
use onnx_runtime_loader::{encode_model, Model};
use onnx_runtime_session::{InferenceSession, Tensor};

fn f32_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Add an inline f32 initializer, returning its value id.
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

fn op(
    g: &mut Graph,
    op_type: &str,
    name: &str,
    inputs: &[ValueId],
    out_dims: &[usize],
) -> ValueId {
    let out = g.create_named_value(name, DataType::Float32, static_shape(out_dims.iter().copied()));
    let node = Node::new(
        NodeId(0),
        op_type,
        inputs.iter().map(|&v| Some(v)).collect(),
        vec![out],
    );
    g.insert_node(node);
    out
}

/// Build `Relu(MatMul(X, W) + bias)` as ONNX bytes.
///
/// `X[2,3]` is the runtime input; `W[3,2]` and `bias[2]` are initializers
/// chosen so the pre-`Relu` result has BOTH positive and negative entries, so
/// the `Relu` actually clamps (proving the fused activation runs).
fn build_model_bytes() -> Vec<u8> {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);

    // X @ W = [[1,2,3],[4,5,6]] @ [[1,-1],[1,-1],[1,-1]] = [[6,-6],[15,-15]].
    // + bias [0.5, 1.0] -> [[6.5, -5.0], [15.5, -14.0]].
    // Relu             -> [[6.5,  0.0], [15.5,   0.0]].  (negatives clamped)
    let w_data = [1.0f32, -1.0, 1.0, -1.0, 1.0, -1.0];
    let bias = [0.5f32, 1.0];

    let x = g.create_named_value("X", DataType::Float32, static_shape([2, 3]));
    g.add_input(x);
    let w = f32_init(&mut g, "W", &[3, 2], &w_data);
    let m = op(&mut g, "MatMul", "mm", &[x, w], &[2, 2]);
    let b = f32_init(&mut g, "B", &[2], &bias);
    let a = op(&mut g, "Add", "biased", &[m, b], &[2, 2]);
    let y = op(&mut g, "Relu", "Y", &[a], &[2, 2]);
    g.add_output(y);

    encode_model(&Model::new(&g)).expect("encode synthetic MatMul+Add+Relu model")
}

fn count(g: &onnx_runtime_ir::Graph, op: &str) -> usize {
    g.nodes.values().filter(|n| n.op_type == op).count()
}

/// End-to-end: the fused `"all"` output equals the unfused `"none"` output, and
/// the `"all"` graph really collapsed the trio into one `FusedGemm` node.
#[test]
fn fused_gemm_parity_and_fusion_fires() {
    let bytes = build_model_bytes();
    let x = Tensor::from_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    // (a) Unfused reference: MatMul + Add + Relu standalone kernels.
    let mut unfused = InferenceSession::builder()
        .model_bytes(&bytes)
        .option("optimization", "none")
        .build()
        .expect("build unfused session");
    let unfused_out = unfused.run(&[("X", &x)]).expect("run unfused");

    // (b) Fused: the full pipeline emits a single FusedGemm.
    let mut fused = InferenceSession::builder()
        .model_bytes(&bytes)
        .option("optimization", "all")
        .build()
        .expect("build fused session");

    // Proof the fusion fired: exactly one FusedGemm (contrib domain), and no
    // standalone MatMul/Add/Relu survived for this region.
    let fg = fused.graph();
    assert_eq!(
        count(fg, "FusedGemm"),
        1,
        "expected exactly one FusedGemm node in the optimized graph"
    );
    assert_eq!(count(fg, "MatMul"), 0, "no standalone MatMul should survive");
    assert_eq!(count(fg, "Add"), 0, "no standalone Add should survive");
    assert_eq!(count(fg, "Relu"), 0, "no standalone Relu should survive");
    let fused_node = fg
        .nodes
        .values()
        .find(|n| n.op_type == "FusedGemm")
        .unwrap();
    assert_eq!(
        fused_node.domain, "com.microsoft",
        "FusedGemm must be emitted in the contrib domain"
    );

    let fused_out = fused.run(&[("X", &x)]).expect("run fused");

    // Both paths reuse the same matmul_dense + broadcast-add + max(0,x), so the
    // results must match to a tight tolerance (in fact byte-identical here).
    assert_eq!(unfused_out.len(), 1);
    assert_eq!(fused_out.len(), 1);
    let uf = unfused_out[0].to_vec_f32();
    let ff = fused_out[0].to_vec_f32();
    assert_eq!(uf.len(), ff.len(), "output element count mismatch");

    let max_abs = uf
        .iter()
        .zip(&ff)
        .fold(0.0f32, |m, (&a, &b)| m.max((a - b).abs()));
    eprintln!("FusedGemm parity: fused vs unfused max_abs = {max_abs:.3e}");
    const ATOL: f32 = 1e-6;
    assert!(
        max_abs < ATOL,
        "fused FusedGemm output must match unfused within atol={ATOL:.0e} (got {max_abs:.3e})"
    );

    // Sanity: the expected clamped reference, computed by hand.
    let expect = [6.5f32, 0.0, 15.5, 0.0];
    for (i, (&got, &want)) in ff.iter().zip(&expect).enumerate() {
        assert!(
            (got - want).abs() < ATOL,
            "element {i}: got {got}, want {want}"
        );
    }
    // Relu really clamped: at least one pre-Relu negative became exactly 0.
    assert!(
        ff.contains(&0.0),
        "Relu should have clamped at least one negative pre-activation to 0"
    );
    assert_eq!(fused_out[0].shape, vec![2, 2]);
}

//! Synthetic end-to-end parity for the SDPA-core fusion
//! (`MatMul → (Mul|Div) → [Add] → Softmax → MatMul → com.microsoft::FusedAttention`,
//! `docs/ORT2.md` §18.x AttentionFusion).
//!
//! This constructs a small scaled-dot-product-attention core by hand (via the
//! ONNX IR API), encodes it to ONNX bytes, and runs it through the production
//! `onnx-runtime-session` pipeline TWO ways:
//!
//! * `optimization="none"` — the unfused reference (`MatMul` + `Div` +
//!   optional `Add` + `Softmax` + `MatMul` standalone kernels);
//! * `optimization="all"` — the full device-independent pipeline, which fuses
//!   the chain into a single `com.microsoft::FusedAttention` node backed by the
//!   CPU kernel.
//!
//! Both variants (unmasked and additively-masked) assert the two outputs match
//! within a TIGHT `1e-6` atol — the fused single-pass reuses the same
//! `matmul_dense` GEMM, `softmax_slices` reduction and broadcast-add as the
//! standalone kernels, so any drift is only a few ULPs — AND that the `"all"`
//! graph really fused: exactly one `FusedAttention` node and zero surviving
//! standalone `Softmax` / score-`MatMul` / `Div` / mask-`Add` for the region.
//!
//! Nothing here special-cases any model: the graph is a generic fixture and
//! `"optimization"` is a generic, model-agnostic option.

use onnx_runtime_ir::{
    DataType, Graph, Node, NodeId, TensorData, ValueId, WeightRef, static_shape,
};
use onnx_runtime_loader::{Model, encode_model};
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

fn input(g: &mut Graph, name: &str, dims: &[usize]) -> ValueId {
    let v = g.create_named_value(name, DataType::Float32, static_shape(dims.iter().copied()));
    g.add_input(v);
    v
}

fn node(
    g: &mut Graph,
    op_type: &str,
    name: &str,
    inputs: &[ValueId],
    out_dims: &[usize],
) -> ValueId {
    let out = g.create_named_value(
        name,
        DataType::Float32,
        static_shape(out_dims.iter().copied()),
    );
    g.insert_node(Node::new(
        NodeId(0),
        op_type,
        inputs.iter().map(|&v| Some(v)).collect(),
        vec![out],
    ));
    out
}

/// Build `Softmax((Q·K)/scale [+ mask], axis=-1) · V` as ONNX bytes.
///
/// Rank-3 `[batch=1, seq, dim]` tensors. `K` is supplied **pre-transposed**
/// (`[1, head_dim, seq_k]`), matching the `bert_toy` layout where the score
/// `MatMul` consumes an already-transposed K, so the fused node carries
/// `k_transposed = 1`. When `masked`, an additive mask `[1, 1, seq_k]`
/// broadcasts across queries.
fn build_model_bytes(masked: bool) -> Vec<u8> {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 12);

    // seq_q = 2, head_dim = 3, seq_k = 2.
    let q = input(&mut g, "Q", &[1, 2, 3]);
    let k = input(&mut g, "K", &[1, 3, 2]); // pre-transposed [d=3, sk=2]
    let v = input(&mut g, "V", &[1, 2, 3]); // [sk=2, dv=3]

    // scale divisor = 2.0 (scalar initializer) → fused scale attr = 0.5.
    let scale_c = f32_init(&mut g, "scale_c", &[], &[2.0]);

    let scores = node(&mut g, "MatMul", "scores", &[q, k], &[1, 2, 2]);
    let scaled = node(&mut g, "Div", "scaled", &[scores, scale_c], &[1, 2, 2]);

    let sm_in = if masked {
        let mask = input(&mut g, "mask", &[1, 1, 2]);
        node(&mut g, "Add", "masked", &[scaled, mask], &[1, 2, 2])
    } else {
        scaled
    };

    let probs = g.create_named_value("probs", DataType::Float32, static_shape([1, 2, 2]));
    let mut sm = Node::new(NodeId(0), "Softmax", vec![Some(sm_in)], vec![probs]);
    sm.attributes
        .insert("axis".into(), onnx_runtime_ir::Attribute::Int(-1));
    g.insert_node(sm);

    let out = node(&mut g, "MatMul", "out", &[probs, v], &[1, 2, 3]);
    g.add_output(out);

    encode_model(&Model::new(&g)).expect("encode synthetic SDPA model")
}

fn count(g: &Graph, op: &str) -> usize {
    g.nodes.values().filter(|n| n.op_type == op).count()
}

fn run_variant(masked: bool) {
    let bytes = build_model_bytes(masked);
    let q = Tensor::from_f32(&[1, 2, 3], &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).unwrap();
    let k = Tensor::from_f32(&[1, 3, 2], &[1.0, 0.0, 0.0, 1.0, 0.5, -0.5]).unwrap();
    let v = Tensor::from_f32(&[1, 2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let mask = Tensor::from_f32(&[1, 1, 2], &[0.0, -1.0]).unwrap();

    let feeds: Vec<(&str, &Tensor)> = if masked {
        vec![("Q", &q), ("K", &k), ("V", &v), ("mask", &mask)]
    } else {
        vec![("Q", &q), ("K", &k), ("V", &v)]
    };

    // (a) Unfused reference: standalone MatMul/Div/[Add]/Softmax/MatMul.
    let mut unfused = InferenceSession::builder()
        .model_bytes(&bytes)
        .option("optimization", "none")
        .build()
        .expect("build unfused session");
    let unfused_out = unfused.run(&feeds).expect("run unfused");

    // (b) Fused: the full pipeline emits a single FusedAttention.
    let mut fused = InferenceSession::builder()
        .model_bytes(&bytes)
        .option("optimization", "all")
        .build()
        .expect("build fused session");

    let fg = fused.graph();
    assert_eq!(
        count(fg, "FusedAttention"),
        1,
        "[masked={masked}] expected exactly one FusedAttention node"
    );
    assert_eq!(
        count(fg, "Softmax"),
        0,
        "[masked={masked}] no standalone Softmax should survive"
    );
    assert_eq!(
        count(fg, "MatMul"),
        0,
        "[masked={masked}] no standalone score/out MatMul should survive"
    );
    assert_eq!(
        count(fg, "Div"),
        0,
        "[masked={masked}] the scale Div should be folded into the fused node"
    );
    if masked {
        assert_eq!(
            count(fg, "Add"),
            0,
            "[masked] the mask Add should be folded into the fused node"
        );
    }
    let fa = fg
        .nodes
        .values()
        .find(|n| n.op_type == "FusedAttention")
        .unwrap();
    assert_eq!(
        fa.domain, "com.microsoft",
        "FusedAttention must be emitted in the contrib domain"
    );
    assert_eq!(
        fa.attr("scale").and_then(|a| a.as_float()),
        Some(0.5),
        "[masked={masked}] scale = 1/2 = 0.5"
    );
    assert_eq!(
        fa.attr("k_transposed").and_then(|a| a.as_int()),
        Some(1),
        "[masked={masked}] pre-transposed K → k_transposed=1"
    );
    // masked → 4 inputs [Q,K,V,mask]; unmasked → 3 inputs [Q,K,V].
    assert_eq!(fa.inputs.len(), if masked { 4 } else { 3 });

    let fused_out = fused.run(&feeds).expect("run fused");

    assert_eq!(unfused_out.len(), 1);
    assert_eq!(fused_out.len(), 1);
    let uf = unfused_out[0].to_vec_f32();
    let ff = fused_out[0].to_vec_f32();
    assert_eq!(uf.len(), ff.len(), "output element count mismatch");

    let max_abs = uf
        .iter()
        .zip(&ff)
        .fold(0.0f32, |m, (&a, &b)| m.max((a - b).abs()));
    eprintln!("FusedAttention parity [masked={masked}]: fused vs unfused max_abs = {max_abs:.3e}");
    const ATOL: f32 = 1e-6;
    assert!(
        max_abs < ATOL,
        "[masked={masked}] fused output must match unfused within atol={ATOL:.0e} (got {max_abs:.3e})"
    );

    // Softmax rows sum to 1, so each output row is a convex combination of the V
    // rows → every entry lies within the min/max of V (a cheap sanity bound).
    for &x in &ff {
        assert!(x.is_finite(), "[masked={masked}] output must be finite");
        assert!(
            (1.0 - 1e-4..=6.0 + 1e-4).contains(&x),
            "[masked={masked}] output {x} outside the V range [1,6]"
        );
    }
    assert_eq!(fused_out[0].shape, vec![1, 2, 3]);
}

#[test]
fn fused_attention_parity_unmasked() {
    run_variant(false);
}

#[test]
fn fused_attention_parity_masked() {
    run_variant(true);
}

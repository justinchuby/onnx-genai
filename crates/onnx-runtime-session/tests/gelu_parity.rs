//! Synthetic end-to-end parity for the exact-GELU fusion
//! (`0.5·x·(1 + Erf(x / √2)) → com.microsoft::Gelu`, `docs/ORT2.md` §18.2).
//!
//! The `bert_toy` conformance model DOES exercise this fusion (its 6
//! feed-forward blocks each use the `Erf` GELU decomposition — see
//! `bert_toy_optimized_parity.rs`), but this test additionally pins the numerics
//! on a tiny hand-built graph that we can reason about exactly. It constructs
//! the `Mul/Div → Erf → Add → Mul` decomposition by hand (via the ONNX IR API),
//! encodes it to ONNX bytes, and runs it through the production
//! `onnx-runtime-session` pipeline TWO ways:
//!
//! * `optimization="none"` — the unfused reference (standalone `Mul`, `Div`,
//!   `Erf`, `Add` kernels);
//! * `optimization="all"` — the full device-independent pipeline, which fuses
//!   the five ops into a single `com.microsoft::Gelu` node backed by the CPU
//!   kernel.
//!
//! We assert the two outputs are numerically identical (both paths route
//! through the SAME `erf` helper, so this is byte-identical, bounded by a tight
//! `1e-6` atol), that the fused output matches an independent hand computation,
//! AND that the `"all"` graph really fused: exactly one `Gelu` node and zero
//! surviving standalone `Erf`/`Div`/`Add`/`Mul` for the region.
//!
//! Nothing here special-cases any model: the graph is a generic fixture and
//! `"optimization"` is a generic, model-agnostic option.

use std::f64::consts::SQRT_2;

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

fn op(g: &mut Graph, op_type: &str, name: &str, inputs: &[ValueId], out_dims: &[usize]) -> ValueId {
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

/// Build `0.5·X · (1 + Erf(X / √2))` over runtime input `X[2,3]`, with the
/// three coefficients (`0.5`, `√2`, `1.0`) as scalar initializers — exactly the
/// shape the `bert_toy` exporter emits (after `ConstantFolding` materializes the
/// `Constant` nodes into inline weights).
fn build_model_bytes() -> Vec<u8> {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);

    let x = g.create_named_value("X", DataType::Float32, static_shape([2, 3]));
    g.add_input(x);

    let half_c = f32_init(&mut g, "half_c", &[], &[0.5]);
    let sqrt2_c = f32_init(&mut g, "sqrt2_c", &[], &[std::f32::consts::SQRT_2]);
    let one_c = f32_init(&mut g, "one_c", &[], &[1.0]);

    let half = op(&mut g, "Mul", "half", &[x, half_c], &[2, 3]); // 0.5 * X
    let scaled = op(&mut g, "Div", "scaled", &[x, sqrt2_c], &[2, 3]); // X / √2
    let e = op(&mut g, "Erf", "e", &[scaled], &[2, 3]); // erf(X / √2)
    let a = op(&mut g, "Add", "a", &[e, one_c], &[2, 3]); // 1 + erf
    let y = op(&mut g, "Mul", "Y", &[half, a], &[2, 3]); // 0.5·X · (1 + erf)
    g.add_output(y);

    encode_model(&Model::new(&g)).expect("encode synthetic GELU decomposition model")
}

fn count(g: &onnx_runtime_ir::Graph, op: &str) -> usize {
    g.nodes.values().filter(|n| n.op_type == op).count()
}

/// Gauss error function via Abramowitz & Stegun 7.1.26 — the same reference the
/// CPU `Erf`/`Gelu` kernels use, replicated here so the parity check is
/// independent of the crate under test.
fn erf(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;
    const P: f64 = 0.327_591_1;
    let t = 1.0 / (1.0 + P * x);
    let poly = ((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t;
    let y = 1.0 - poly * (-x * x).exp();
    sign * y
}

fn reference_gelu(x: f32) -> f32 {
    let xf = x as f64;
    (0.5 * xf * (1.0 + erf(xf / SQRT_2))) as f32
}

/// End-to-end: the fused `"all"` output equals the unfused `"none"` output and a
/// hand-computed GELU, and the `"all"` graph collapsed the decomposition into a
/// single `Gelu` node.
#[test]
fn gelu_parity_and_fusion_fires() {
    let bytes = build_model_bytes();
    // Both signs of x so the erf branch is exercised across zero.
    let x_vals = [-2.0f32, -1.0, -0.25, 0.25, 1.0, 2.0];
    let x = Tensor::from_f32(&[2, 3], &x_vals).unwrap();

    // (a) Unfused reference: Mul + Div + Erf + Add + Mul standalone kernels.
    let mut unfused = InferenceSession::builder()
        .model_bytes(&bytes)
        .option("optimization", "none")
        .build()
        .expect("build unfused session");
    let unfused_out = unfused.run(&[("X", &x)]).expect("run unfused");

    // (b) Fused: the full pipeline emits a single Gelu.
    let mut fused = InferenceSession::builder()
        .model_bytes(&bytes)
        .option("optimization", "all")
        .build()
        .expect("build fused session");

    let fg = fused.graph();
    assert_eq!(
        count(fg, "Gelu"),
        1,
        "expected exactly one Gelu node in the optimized graph"
    );
    assert_eq!(count(fg, "Erf"), 0, "no standalone Erf should survive");
    assert_eq!(count(fg, "Div"), 0, "no standalone Div should survive");
    assert_eq!(count(fg, "Add"), 0, "no standalone Add should survive");
    assert_eq!(count(fg, "Mul"), 0, "no standalone Mul should survive");
    let gelu_node = fg.nodes.values().find(|n| n.op_type == "Gelu").unwrap();
    assert_eq!(
        gelu_node.domain, "com.microsoft",
        "Gelu must be emitted in the contrib domain"
    );
    assert_eq!(gelu_node.inputs.len(), 1, "exact Gelu takes the single input X");
    assert!(gelu_node.attributes.is_empty(), "exact Gelu has no attributes");

    let fused_out = fused.run(&[("X", &x)]).expect("run fused");

    assert_eq!(unfused_out.len(), 1);
    assert_eq!(fused_out.len(), 1);
    let uf = unfused_out[0].to_vec_f32();
    let ff = fused_out[0].to_vec_f32();
    assert_eq!(uf.len(), ff.len(), "output element count mismatch");
    assert_eq!(fused_out[0].shape, vec![2, 3]);

    // Fused vs unfused: same `erf` helper both ways → byte-identical, tight atol.
    let max_abs_fu = uf
        .iter()
        .zip(&ff)
        .fold(0.0f32, |m, (&a, &b)| m.max((a - b).abs()));
    // Fused vs an independent hand computation.
    let reference: Vec<f32> = x_vals.iter().map(|&v| reference_gelu(v)).collect();
    let max_abs_ref = ff
        .iter()
        .zip(&reference)
        .fold(0.0f32, |m, (&a, &b)| m.max((a - b).abs()));
    eprintln!(
        "Gelu parity: fused-vs-unfused max_abs = {max_abs_fu:.3e}, fused-vs-reference max_abs = {max_abs_ref:.3e}"
    );

    const ATOL: f32 = 1e-6;
    assert!(
        max_abs_fu < ATOL,
        "fused Gelu output must match unfused decomposition within atol={ATOL:.0e} (got {max_abs_fu:.3e})"
    );
    assert!(
        max_abs_ref < ATOL,
        "fused Gelu output must match the hand-computed reference within atol={ATOL:.0e} (got {max_abs_ref:.3e})"
    );
}

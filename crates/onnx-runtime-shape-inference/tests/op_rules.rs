//! Per-operator unit tests driving each rule through the single-node public API
//! ([`InferenceRegistry::infer_node`]). Covers concrete dims, symbolic dims,
//! broadcasting edge cases, and shape-data propagation.

use std::collections::HashMap;

use onnx_runtime_ir::{Attribute, DataType, Node, NodeId, SymbolId, ValueId};
use onnx_runtime_shape_inference::{
    DimExpr, InferenceRegistry, MergePolicy, NodeIo, ShapeData, SymbolInterner, TypeInfo,
};

// --- construction helpers -------------------------------------------------

fn c(n: i64) -> DimExpr {
    DimExpr::constant(n)
}

fn sym(n: u32) -> DimExpr {
    DimExpr::symbol(SymbolId(n))
}

/// A typed input with the given dtype and dims.
fn tin(dt: DataType, dims: Vec<DimExpr>) -> NodeIo {
    NodeIo::typed(TypeInfo::new(dt, dims))
}

/// A float32 input.
fn f32in(dims: Vec<DimExpr>) -> NodeIo {
    tin(DataType::Float32, dims)
}

/// An input carrying a resolved int64 shape-data vector.
fn sd_vec(elems: Vec<DimExpr>) -> NodeIo {
    NodeIo {
        type_info: Some(TypeInfo::new(DataType::Int64, vec![c(elems.len() as i64)])),
        shape_data: Some(ShapeData::vector(DataType::Int64, elems)),
    }
}

fn node(op: &str, n_in: usize, n_out: usize) -> Node {
    Node::new(
        NodeId(0),
        op,
        vec![Some(ValueId(0)); n_in],
        (0..n_out).map(|i| ValueId(i as u32)).collect(),
    )
}

fn with_attr(mut n: Node, name: &str, attr: Attribute) -> Node {
    n.attributes.insert(name.to_string(), attr);
    n
}

fn with_domain(mut n: Node, domain: &str) -> Node {
    n.domain = domain.to_string();
    n
}

fn run(n: &Node, inputs: Vec<NodeIo>, opset: u64) -> Vec<NodeIo> {
    run_policy(n, inputs, opset, MergePolicy::Permissive)
}

fn run_policy(n: &Node, inputs: Vec<NodeIo>, opset: u64, policy: MergePolicy) -> Vec<NodeIo> {
    let reg = InferenceRegistry::default_registry();
    let mut imports = HashMap::new();
    imports.insert(String::new(), opset);
    let mut interner = SymbolInterner::new(0x8000_0000);
    reg.infer_node(n, &imports, inputs, policy, &mut interner)
        .unwrap()
}

/// The resolved output shape of slot 0.
fn out_shape(outs: &[NodeIo]) -> Vec<DimExpr> {
    outs[0]
        .type_info
        .as_ref()
        .expect("output type resolved")
        .shape
        .clone()
}

fn out_dtype(outs: &[NodeIo]) -> DataType {
    outs[0].type_info.as_ref().unwrap().dtype
}

// --- MatMul ---------------------------------------------------------------

#[test]
fn matmul_2d() {
    let n = node("MatMul", 2, 1);
    let outs = run(
        &n,
        vec![f32in(vec![c(2), c(3)]), f32in(vec![c(3), c(4)])],
        13,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(4)]);
}

#[test]
fn matmul_batched_symbolic() {
    // [N, 8, 64] @ [N, 64, 32] -> [N, 8, 32]
    let n = node("MatMul", 2, 1);
    let outs = run(
        &n,
        vec![
            f32in(vec![sym(0), c(8), c(64)]),
            f32in(vec![sym(0), c(64), c(32)]),
        ],
        13,
    );
    assert_eq!(out_shape(&outs), vec![sym(0), c(8), c(32)]);
}

#[test]
fn matmul_broadcast_batch() {
    // [2,1,8,64] @ [64,32] -> [2,1,8,32]
    let n = node("MatMul", 2, 1);
    let outs = run(
        &n,
        vec![
            f32in(vec![c(2), c(1), c(8), c(64)]),
            f32in(vec![c(64), c(32)]),
        ],
        13,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(1), c(8), c(32)]);
}

#[test]
fn matmul_1d_1d_scalar() {
    let n = node("MatMul", 2, 1);
    let outs = run(&n, vec![f32in(vec![c(5)]), f32in(vec![c(5)])], 13);
    assert_eq!(out_shape(&outs), Vec::<DimExpr>::new());
}

#[test]
fn matmul_contraction_mismatch_errors() {
    let n = node("MatMul", 2, 1);
    let reg = InferenceRegistry::default_registry();
    let mut imports = HashMap::new();
    imports.insert(String::new(), 13u64);
    let mut interner = SymbolInterner::new(0x8000_0000);
    let res = reg.infer_node(
        &n,
        &imports,
        vec![f32in(vec![c(2), c(3)]), f32in(vec![c(4), c(5)])],
        MergePolicy::Permissive,
        &mut interner,
    );
    assert!(res.is_err());
}

// --- Gemm -----------------------------------------------------------------

#[test]
fn gemm_transb() {
    // A [8, 64], B [32, 64] with transB=1 -> [8, 32]
    let n = with_attr(node("Gemm", 3, 1), "transB", Attribute::Int(1));
    let outs = run(
        &n,
        vec![
            f32in(vec![c(8), c(64)]),
            f32in(vec![c(32), c(64)]),
            f32in(vec![c(32)]),
        ],
        13,
    );
    assert_eq!(out_shape(&outs), vec![c(8), c(32)]);
}

// --- FusedMatMul (com.microsoft) ------------------------------------------

/// A `com.microsoft::FusedMatMul` node with the given int attributes.
fn fused_matmul_node(attrs: &[(&str, i64)]) -> Node {
    let mut n = with_domain(node("FusedMatMul", 2, 1), "com.microsoft");
    for &(name, v) in attrs {
        n = with_attr(n, name, Attribute::Int(v));
    }
    n
}

#[test]
fn fused_matmul_transb() {
    // The exact case Chew cited: A [8,64] · B [32,64]^T -> [8,32]. The plain
    // matmul reuse produced the wrong [8,64]; the dedicated handler is correct.
    let n = fused_matmul_node(&[("transB", 1)]);
    let outs = run(&n, vec![f32in(vec![c(8), c(64)]), f32in(vec![c(32), c(64)])], 1);
    assert_eq!(out_shape(&outs), vec![c(8), c(32)]);
}

#[test]
fn fused_matmul_transa() {
    // A supplied as [K, M] = [64, 8], transA=1 -> M=8; B [64, 32] -> [8, 32].
    let n = fused_matmul_node(&[("transA", 1)]);
    let outs = run(&n, vec![f32in(vec![c(64), c(8)]), f32in(vec![c(64), c(32)])], 1);
    assert_eq!(out_shape(&outs), vec![c(8), c(32)]);
}

#[test]
fn fused_matmul_transa_and_transb() {
    // A [K,M]=[64,8] transA, B [N,K]=[32,64] transB -> [8, 32].
    let n = fused_matmul_node(&[("transA", 1), ("transB", 1)]);
    let outs = run(&n, vec![f32in(vec![c(64), c(8)]), f32in(vec![c(32), c(64)])], 1);
    assert_eq!(out_shape(&outs), vec![c(8), c(32)]);
}

#[test]
fn fused_matmul_batched_transb() {
    // Batched: A [N,8,64] · B [N,32,64]^T -> [N,8,32] (symbolic batch preserved).
    let n = fused_matmul_node(&[("transB", 1)]);
    let outs = run(
        &n,
        vec![
            f32in(vec![sym(0), c(8), c(64)]),
            f32in(vec![sym(0), c(32), c(64)]),
        ],
        1,
    );
    assert_eq!(out_shape(&outs), vec![sym(0), c(8), c(32)]);
}

#[test]
fn fused_matmul_plain_matches_matmul() {
    // With no transpose flags, FusedMatMul must equal plain MatMul.
    let n = fused_matmul_node(&[]);
    let outs = run(&n, vec![f32in(vec![c(2), c(3)]), f32in(vec![c(3), c(4)])], 1);
    assert_eq!(out_shape(&outs), vec![c(2), c(4)]);
}

#[test]
fn fused_matmul_alpha_is_shape_neutral() {
    // `alpha` scales values only; it must not affect the output shape.
    let mut n = fused_matmul_node(&[("transB", 1)]);
    n = with_attr(n, "alpha", Attribute::Float(2.0));
    let outs = run(&n, vec![f32in(vec![c(8), c(64)]), f32in(vec![c(32), c(64)])], 1);
    assert_eq!(out_shape(&outs), vec![c(8), c(32)]);
}

#[test]
fn fused_matmul_trans_batch_a_moves_leading_axis() {
    // transBatchA relocates the leading axis into the row (M) slot:
    // A [4, 2, 8] -> effective [2, 4, 8] (batch=2, M=4, K=8);
    // B [2, 8, 16] -> [2, 4, 16].
    let n = fused_matmul_node(&[("transBatchA", 1)]);
    let outs = run(
        &n,
        vec![
            f32in(vec![c(4), c(2), c(8)]),
            f32in(vec![c(2), c(8), c(16)]),
        ],
        1,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(4), c(16)]);
}

#[test]
fn fused_gemm_output_equals_matmul_shape() {
    // com.microsoft::FusedGemm = Relu(MatMul(A, B) + bias); output shape is the
    // plain MatMul shape (bias broadcasts, Relu is elementwise).
    let n = with_domain(node("FusedGemm", 3, 1), "com.microsoft");
    let outs = run(
        &n,
        vec![
            f32in(vec![c(2), c(3)]),
            f32in(vec![c(3), c(4)]),
            f32in(vec![c(4)]),
        ],
        1,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(4)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn fused_gemm_batched_symbolic_shape() {
    // Batched, symbolic leading dim carries through unchanged.
    let n = with_domain(node("FusedGemm", 3, 1), "com.microsoft");
    let outs = run(
        &n,
        vec![
            f32in(vec![sym(1), c(8), c(64)]),
            f32in(vec![c(64), c(32)]),
            f32in(vec![c(32)]),
        ],
        1,
    );
    assert_eq!(out_shape(&outs), vec![sym(1), c(8), c(32)]);
}

#[test]
fn add_broadcast_concrete() {
    let n = node("Add", 2, 1);
    let outs = run(
        &n,
        vec![f32in(vec![c(3), c(1)]), f32in(vec![c(1), c(4)])],
        13,
    );
    assert_eq!(out_shape(&outs), vec![c(3), c(4)]);
}

#[test]
fn add_broadcast_symbolic_batch() {
    // [N, 8, 768] + [768] -> [N, 8, 768]
    let n = node("Add", 2, 1);
    let outs = run(
        &n,
        vec![f32in(vec![sym(0), c(8), c(768)]), f32in(vec![c(768)])],
        13,
    );
    assert_eq!(out_shape(&outs), vec![sym(0), c(8), c(768)]);
}

#[test]
fn add_symbolic_vs_concrete_prefers_concrete() {
    // broadcast(N, 8) -> 8 (the concrete non-1 extent wins)
    let n = node("Add", 2, 1);
    let outs = run(&n, vec![f32in(vec![sym(0)]), f32in(vec![c(8)])], 13);
    assert_eq!(out_shape(&outs), vec![c(8)]);
}

#[test]
fn add_two_distinct_symbols_keeps_named_representative() {
    // Broadcasting a data-dependent/anonymous symbol (high-range id, as minted
    // by inference for an unresolved extent) against a named graph symbol
    // (low-range id) must re-unify onto the *named* one — never a fresh symbol
    // — so the session can bind it. This is the invariant that keeps a
    // `Shape`-driven `Expand`/`Add` chain resolvable end-to-end.
    let named = sym(1);
    let anon = sym(0x8000_0000);
    let n = node("Add", 2, 1);
    // Order-independent: named wins whether it is the left or the right operand.
    let outs = run(&n, vec![f32in(vec![anon.clone()]), f32in(vec![named.clone()])], 13);
    assert_eq!(out_shape(&outs), vec![named.clone()]);
    let outs = run(&n, vec![f32in(vec![named.clone()]), f32in(vec![anon])], 13);
    assert_eq!(out_shape(&outs), vec![named]);
}

#[test]
fn div_strict_incompatible_broadcast_errors() {
    let n = node("Div", 2, 1);
    let reg = InferenceRegistry::default_registry();
    let mut imports = HashMap::new();
    imports.insert(String::new(), 13u64);
    let mut interner = SymbolInterner::new(0x8000_0000);
    let res = reg.infer_node(
        &n,
        &imports,
        vec![f32in(vec![c(3)]), f32in(vec![c(4)])],
        MergePolicy::Strict,
        &mut interner,
    );
    assert!(res.is_err());
}

// --- unary ----------------------------------------------------------------

#[test]
fn relu_passthrough() {
    let n = node("Relu", 1, 1);
    let outs = run(&n, vec![f32in(vec![sym(0), c(8), c(768)])], 13);
    assert_eq!(out_shape(&outs), vec![sym(0), c(8), c(768)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

// --- Reshape (shape-data) -------------------------------------------------

#[test]
fn reshape_from_shape_data_with_minus_one() {
    // input [B, S, 768], target [0, 0, 12, -1] -> [B, S, 12, 64]
    let n = node("Reshape", 2, 1);
    let target = sd_vec(vec![c(0), c(0), c(12), c(-1)]);
    let outs = run(&n, vec![f32in(vec![sym(0), sym(1), c(768)]), target], 13);
    assert_eq!(out_shape(&outs), vec![sym(0), sym(1), c(12), c(64)]);
}

#[test]
fn reshape_zero_copies_input_dim() {
    // input [4, 8, 16], target [0, -1] -> [4, 128]
    let n = node("Reshape", 2, 1);
    let target = sd_vec(vec![c(0), c(-1)]);
    let outs = run(&n, vec![f32in(vec![c(4), c(8), c(16)]), target], 13);
    assert_eq!(out_shape(&outs), vec![c(4), c(128)]);
}

#[test]
fn reshape_symbolic_target_dim() {
    // target carries a symbolic dim (batch read from a Shape op)
    let n = node("Reshape", 2, 1);
    let target = sd_vec(vec![sym(0), c(-1)]);
    let outs = run(&n, vec![f32in(vec![sym(0), c(8), c(16)]), target], 13);
    // -1 = (N*8*16)/N = 128
    assert_eq!(out_shape(&outs), vec![sym(0), c(128)]);
}

#[test]
fn reshape_overflowing_total_degrades_to_symbol() {
    // Regression (Holden): an input whose concrete element count is 2^80
    // overflows i64. The inferred `-1` dim must degrade to a fresh symbol, not
    // panic (debug) and not wrap to a bogus static 0 (release).
    let n = node("Reshape", 2, 1);
    let big = c(1 << 20);
    let target = sd_vec(vec![c(-1)]);
    let outs = run(
        &n,
        vec![
            f32in(vec![big.clone(), big.clone(), big.clone(), big]),
            target,
        ],
        13,
    );
    let out = out_shape(&outs);
    assert_eq!(out.len(), 1);
    // Fresh symbol (anon range), never a concrete 0 or negative dim.
    assert_eq!(out[0].as_const(), None);
    assert!(out[0].as_symbol().is_some());
}

#[test]
fn size_overflowing_total_is_not_bogus() {
    // `Size` over a 2^80-element tensor overflows i64; the shape-data scalar it
    // emits must be an unknown (overflow) dim, never a wrapped concrete value.
    let n = node("Size", 1, 1);
    let big = c(1 << 20);
    let outs = run(
        &n,
        vec![f32in(vec![big.clone(), big.clone(), big.clone(), big])],
        13,
    );
    let sd = outs[0]
        .shape_data
        .as_ref()
        .expect("Size emits shape-data");
    assert_eq!(sd.elems.len(), 1);
    assert!(sd.elems[0].is_overflow());
    assert_eq!(sd.elems[0].as_const(), None);
}

// --- Transpose ------------------------------------------------------------

#[test]
fn transpose_perm() {
    let n = with_attr(
        node("Transpose", 1, 1),
        "perm",
        Attribute::Ints(vec![0, 2, 1, 3]),
    );
    let outs = run(&n, vec![f32in(vec![sym(0), c(8), c(12), c(64)])], 13);
    assert_eq!(out_shape(&outs), vec![sym(0), c(12), c(8), c(64)]);
}

#[test]
fn transpose_default_reverses() {
    let n = node("Transpose", 1, 1);
    let outs = run(&n, vec![f32in(vec![c(2), c(3), c(4)])], 13);
    assert_eq!(out_shape(&outs), vec![c(4), c(3), c(2)]);
}

// --- Gather ---------------------------------------------------------------

#[test]
fn gather_axis0_scalar_index() {
    // data [10, 768], scalar index -> [768]
    let n = node("Gather", 2, 1);
    let outs = run(
        &n,
        vec![f32in(vec![c(10), c(768)]), tin(DataType::Int64, vec![])],
        13,
    );
    assert_eq!(out_shape(&outs), vec![c(768)]);
}

#[test]
fn gather_shape_data_selects_dim() {
    // Shape of [N, 8, 768] gathered at index [0] -> shape-data [N]
    let shape_out = sd_vec(vec![sym(0), c(8), c(768)]);
    let idx = sd_vec(vec![c(0)]);
    let n = with_attr(node("Gather", 2, 1), "axis", Attribute::Int(0));
    let outs = run(&n, vec![shape_out, idx], 13);
    let sd = outs[0].shape_data.as_ref().expect("gather shape-data");
    assert_eq!(sd.elems, vec![sym(0)]);
}

// --- Concat ---------------------------------------------------------------

#[test]
fn concat_sums_axis() {
    let n = with_attr(node("Concat", 2, 1), "axis", Attribute::Int(1));
    let outs = run(
        &n,
        vec![f32in(vec![c(2), c(3)]), f32in(vec![c(2), c(5)])],
        13,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(8)]);
}

#[test]
fn concat_shape_data_builds_vector() {
    // Concat of scalars/vectors of dims -> a shape vector.
    let a = sd_vec(vec![sym(0)]);
    let b = sd_vec(vec![c(12), c(64)]);
    let n = with_attr(node("Concat", 2, 1), "axis", Attribute::Int(0));
    let outs = run(&n, vec![a, b], 13);
    let sd = outs[0].shape_data.as_ref().expect("concat shape-data");
    assert_eq!(sd.elems, vec![sym(0), c(12), c(64)]);
}

// --- Shape / Size ---------------------------------------------------------

#[test]
fn shape_emits_dims_as_shape_data() {
    let n = node("Shape", 1, 1);
    let outs = run(&n, vec![f32in(vec![sym(0), c(8), c(768)])], 13);
    assert_eq!(out_shape(&outs), vec![c(3)]);
    assert_eq!(out_dtype(&outs), DataType::Int64);
    let sd = outs[0].shape_data.as_ref().unwrap();
    assert_eq!(sd.elems, vec![sym(0), c(8), c(768)]);
}

#[test]
fn shape_with_start_end() {
    let n = with_attr(
        with_attr(node("Shape", 1, 1), "start", Attribute::Int(1)),
        "end",
        Attribute::Int(3),
    );
    let outs = run(&n, vec![f32in(vec![sym(0), c(8), c(768), c(2)])], 15);
    let sd = outs[0].shape_data.as_ref().unwrap();
    assert_eq!(sd.elems, vec![c(8), c(768)]);
}

// --- Unsqueeze / Squeeze (opset-range dispatch) ---------------------------

#[test]
fn unsqueeze_v1_axes_attr() {
    // opset 11: axes is an attribute.
    let n = with_attr(node("Unsqueeze", 1, 1), "axes", Attribute::Ints(vec![0]));
    let outs = run(&n, vec![f32in(vec![c(8), c(768)])], 11);
    assert_eq!(out_shape(&outs), vec![c(1), c(8), c(768)]);
}

#[test]
fn unsqueeze_v13_axes_input() {
    // opset 13: axes is input 1 (shape-data).
    let n = node("Unsqueeze", 2, 1);
    let outs = run(&n, vec![f32in(vec![c(8), c(768)]), sd_vec(vec![c(0)])], 13);
    assert_eq!(out_shape(&outs), vec![c(1), c(8), c(768)]);
}

#[test]
fn unsqueeze_scalar_shape_data_to_vector() {
    // A scalar dim unsqueezed to a 1-vector keeps its value (shape-chain).
    let scalar = NodeIo {
        type_info: Some(TypeInfo::new(DataType::Int64, vec![])),
        shape_data: Some(ShapeData::scalar(DataType::Int64, sym(0))),
    };
    let n = with_attr(node("Unsqueeze", 1, 1), "axes", Attribute::Ints(vec![0]));
    let outs = run(&n, vec![scalar], 11);
    let sd = outs[0].shape_data.as_ref().expect("unsqueeze shape-data");
    assert_eq!(sd.elems, vec![sym(0)]);
    assert!(!sd.is_scalar());
}

#[test]
fn squeeze_v13_axes_input() {
    let n = node("Squeeze", 2, 1);
    let outs = run(
        &n,
        vec![f32in(vec![c(1), c(8), c(1)]), sd_vec(vec![c(0), c(2)])],
        13,
    );
    assert_eq!(out_shape(&outs), vec![c(8)]);
}

// --- Slice ----------------------------------------------------------------

#[test]
fn slice_concrete_bounds() {
    // data [10, 768], slice axis 0 [2:8] -> [6, 768]
    let n = node("Slice", 5, 1);
    let outs = run(
        &n,
        vec![
            f32in(vec![c(10), c(768)]),
            sd_vec(vec![c(2)]),
            sd_vec(vec![c(8)]),
            sd_vec(vec![c(0)]),
            sd_vec(vec![c(1)]),
        ],
        13,
    );
    assert_eq!(out_shape(&outs), vec![c(6), c(768)]);
}

#[test]
fn slice_data_dependent_keeps_rank_symbolic() {
    // Bounds unknown (no shape-data on starts/ends): axis stays symbolic.
    let n = node("Slice", 3, 1);
    let starts = f32in(vec![c(1)]); // present but no shape-data
    let ends = f32in(vec![c(1)]);
    let outs = run(&n, vec![f32in(vec![c(10), c(768)]), starts, ends], 13);
    let shape = out_shape(&outs);
    assert_eq!(shape.len(), 2);
    // rank preserved; both dims still present
}

// --- ReduceMean -----------------------------------------------------------

#[test]
fn reduce_mean_keepdims() {
    let n = with_attr(
        with_attr(node("ReduceMean", 1, 1), "axes", Attribute::Ints(vec![-1])),
        "keepdims",
        Attribute::Int(1),
    );
    let outs = run(&n, vec![f32in(vec![sym(0), c(8), c(768)])], 12);
    assert_eq!(out_shape(&outs), vec![sym(0), c(8), c(1)]);
}

#[test]
fn reduce_mean_no_keepdims() {
    let n = with_attr(
        with_attr(node("ReduceMean", 1, 1), "axes", Attribute::Ints(vec![1])),
        "keepdims",
        Attribute::Int(0),
    );
    let outs = run(&n, vec![f32in(vec![c(2), c(8), c(768)])], 12);
    assert_eq!(out_shape(&outs), vec![c(2), c(768)]);
}

// --- Softmax / LayerNorm --------------------------------------------------

#[test]
fn softmax_passthrough() {
    let n = with_attr(node("Softmax", 1, 1), "axis", Attribute::Int(-1));
    let outs = run(&n, vec![f32in(vec![sym(0), c(12), c(8), c(8)])], 13);
    assert_eq!(out_shape(&outs), vec![sym(0), c(12), c(8), c(8)]);
}

#[test]
fn layer_norm_main_and_reduced_outputs() {
    let n = node("LayerNormalization", 3, 3);
    let outs = run(
        &n,
        vec![
            f32in(vec![sym(0), c(8), c(768)]),
            f32in(vec![c(768)]),
            f32in(vec![c(768)]),
        ],
        17,
    );
    assert_eq!(out_shape(&outs), vec![sym(0), c(8), c(768)]);
    // Mean / InvStdDev: last axis collapses to 1.
    let mean = outs[1].type_info.as_ref().unwrap().shape.clone();
    assert_eq!(mean, vec![sym(0), c(8), c(1)]);
}

// --- Cast -----------------------------------------------------------------

#[test]
fn cast_changes_dtype_keeps_shape_and_shape_data() {
    let input = sd_vec(vec![sym(0), c(8)]);
    // Cast int64 -> int32 (to=6)
    let n = with_attr(node("Cast", 1, 1), "to", Attribute::Int(6));
    let outs = run(&n, vec![input], 13);
    assert_eq!(out_dtype(&outs), DataType::Int32);
    assert_eq!(out_shape(&outs), vec![c(2)]);
    let sd = outs[0].shape_data.as_ref().unwrap();
    assert_eq!(sd.dtype, DataType::Int32);
    assert_eq!(sd.elems, vec![sym(0), c(8)]);
}

// --- ConstantOfShape / Expand --------------------------------------------

#[test]
fn constant_of_shape_uses_shape_data() {
    let n = node("ConstantOfShape", 1, 1);
    let outs = run(&n, vec![sd_vec(vec![sym(0), c(8)])], 13);
    assert_eq!(out_shape(&outs), vec![sym(0), c(8)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn expand_broadcasts_against_target() {
    // input [1, 8, 1], target [N, 8, 768] -> [N, 8, 768]
    let n = node("Expand", 2, 1);
    let outs = run(
        &n,
        vec![
            f32in(vec![c(1), c(8), c(1)]),
            sd_vec(vec![sym(0), c(8), c(768)]),
        ],
        13,
    );
    assert_eq!(out_shape(&outs), vec![sym(0), c(8), c(768)]);
}

// --- Where ----------------------------------------------------------------

#[test]
fn where_broadcasts_all_three() {
    let n = node("Where", 3, 1);
    let outs = run(
        &n,
        vec![
            tin(DataType::Bool, vec![c(1), c(8)]),
            f32in(vec![c(3), c(1)]),
            f32in(vec![c(3), c(8)]),
        ],
        13,
    );
    assert_eq!(out_shape(&outs), vec![c(3), c(8)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

// --- Flatten / Split ------------------------------------------------------

#[test]
fn flatten_axis() {
    let n = with_attr(node("Flatten", 1, 1), "axis", Attribute::Int(2));
    let outs = run(&n, vec![f32in(vec![c(2), c(3), c(4), c(5)])], 13);
    assert_eq!(out_shape(&outs), vec![c(6), c(20)]);
}

#[test]
fn split_equal() {
    let n = with_attr(node("Split", 1, 2), "axis", Attribute::Int(1));
    let outs = run(&n, vec![f32in(vec![c(2), c(8)])], 13);
    assert_eq!(out_shape(&outs), vec![c(2), c(4)]);
    assert_eq!(outs[1].type_info.as_ref().unwrap().shape, vec![c(2), c(4)]);
}

// --- Conv / Pool / Pad ----------------------------------------------------

#[test]
fn conv_spatial_formula() {
    // X [N, 3, 224, 224], W [64, 3, 7, 7], stride 2, pad 3 -> [N, 64, 112, 112]
    let n = {
        let n = with_attr(node("Conv", 2, 1), "strides", Attribute::Ints(vec![2, 2]));
        with_attr(n, "pads", Attribute::Ints(vec![3, 3, 3, 3]))
    };
    let outs = run(
        &n,
        vec![
            f32in(vec![sym(0), c(3), c(224), c(224)]),
            f32in(vec![c(64), c(3), c(7), c(7)]),
        ],
        13,
    );
    assert_eq!(out_shape(&outs), vec![sym(0), c(64), c(112), c(112)]);
}

#[test]
fn maxpool_spatial_formula() {
    // X [N, 64, 112, 112], kernel 3, stride 2, pad 1 -> [N, 64, 56, 56]
    let n = {
        let n = with_attr(
            node("MaxPool", 1, 1),
            "kernel_shape",
            Attribute::Ints(vec![3, 3]),
        );
        let n = with_attr(n, "strides", Attribute::Ints(vec![2, 2]));
        with_attr(n, "pads", Attribute::Ints(vec![1, 1, 1, 1]))
    };
    let outs = run(&n, vec![f32in(vec![sym(0), c(64), c(112), c(112)])], 13);
    assert_eq!(out_shape(&outs), vec![sym(0), c(64), c(56), c(56)]);
}

#[test]
fn pad_grows_dims() {
    let n = node("Pad", 2, 1);
    // pads = [0,0,1,1, 0,0,1,1] over rank 4 -> H,W grow by 2
    let pads = sd_vec(vec![c(0), c(0), c(1), c(1), c(0), c(0), c(1), c(1)]);
    let outs = run(&n, vec![f32in(vec![sym(0), c(3), c(32), c(32)]), pads], 13);
    assert_eq!(out_shape(&outs), vec![sym(0), c(3), c(34), c(34)]);
}

// --- unregistered op is permissive ---------------------------------------

#[test]
fn unregistered_op_leaves_output_unresolved() {
    let n = node("SomeExoticOp", 1, 1);
    let outs = run(&n, vec![f32in(vec![c(2), c(3)])], 13);
    assert!(outs[0].type_info.is_none());
}

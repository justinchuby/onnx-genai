//! Per-operator unit tests driving each rule through the single-node public API
//! ([`InferenceRegistry::infer_node`]). Covers concrete dims, symbolic dims,
//! broadcasting edge cases, and shape-data propagation.

use std::collections::HashMap;

use onnx_runtime_ir::{Attribute, DataType, Node, NodeId, SymbolId, ValueId};
use onnx_runtime_shape_inference::{
    DimExpr, InferenceRegistry, MergePolicy, NodeIo, ShapeData, ShapeInferError, SymbolInterner,
    TypeInfo,
};

// --- construction helpers -------------------------------------------------

fn c(n: i64) -> DimExpr {
    DimExpr::constant(n)
}

fn assert_invalid(error: ShapeInferError, op: &str, expected_detail: &str) {
    assert!(
        matches!(
            &error,
            ShapeInferError::Invalid { op: actual_op, detail }
                if actual_op == op && detail.contains(expected_detail)
        ),
        "expected Invalid {op:?} error containing {expected_detail:?}, got {error:?}"
    );
    assert!(error.to_string().contains(expected_detail));
}

#[test]
fn constant_variants_and_constant_of_shape_static_validation() {
    let int = run(
        &with_attr(node("Constant", 0, 1), "value_int", Attribute::Int(9)),
        vec![],
        13,
    );
    assert_eq!(out_shape(&int), Vec::<DimExpr>::new());
    assert_eq!(int[0].shape_data.as_ref().unwrap().elems, vec![c(9)]);

    let floats = run(
        &with_attr(
            node("Constant", 0, 1),
            "value_floats",
            Attribute::Floats(vec![1.0, 2.0]),
        ),
        vec![],
        13,
    );
    assert_eq!(out_dtype(&floats), DataType::Float32);
    assert_eq!(out_shape(&floats), vec![c(2)]);

    let constant_of_shape = node("ConstantOfShape", 1, 1);
    assert!(
        try_run(
            &constant_of_shape,
            vec![tin(DataType::Int64, vec![c(1), c(1)])],
            13,
        )
        .is_err()
    );
    assert!(try_run(&constant_of_shape, vec![sd_vec(vec![c(-1)])], 13,).is_err());
}

#[test]
fn quantization_static_metadata_rejects_invalid_scalar_block_and_dtype_cases() {
    assert!(
        try_run(
            &node("QuantizeLinear", 2, 1),
            vec![f32in(vec![]), f32in(vec![c(1)])],
            13,
        )
        .is_err()
    );

    let invalid_dtype = with_attr(
        node("QuantizeLinear", 2, 1),
        "output_dtype",
        Attribute::Int(-1),
    );
    assert!(try_run(&invalid_dtype, vec![f32in(vec![c(2)]), f32in(vec![])], 21,).is_err());

    let blocked = with_attr(
        node("QuantizeLinear", 3, 1),
        "block_size",
        Attribute::Int(2),
    );
    assert!(
        try_run(
            &blocked,
            vec![
                f32in(vec![c(2), c(4)]),
                f32in(vec![c(2)]),
                tin(DataType::Uint8, vec![c(2), c(2)]),
            ],
            21,
        )
        .is_err()
    );
    assert!(
        try_run(
            &blocked,
            vec![
                f32in(vec![c(2), c(4)]),
                f32in(vec![c(2), c(2)]),
                tin(DataType::Uint8, vec![c(2)]),
            ],
            21,
        )
        .is_err()
    );
    assert!(
        try_run(
            &blocked,
            vec![
                f32in(vec![c(2), c(4)]),
                f32in(vec![c(2), c(3)]),
                tin(DataType::Uint8, vec![c(2), c(3)]),
            ],
            21,
        )
        .is_err()
    );
}

fn sym(n: u32) -> DimExpr {
    DimExpr::symbol(SymbolId(n))
}

#[test]
fn gather_elementwise_and_nd_validate_rank_and_batch_depth() {
    let elements = with_attr(node("GatherElements", 2, 1), "axis", Attribute::Int(-1));
    let out = run(
        &elements,
        vec![
            f32in(vec![c(2), c(3)]),
            tin(DataType::Int64, vec![c(2), c(4)]),
        ],
        13,
    );
    assert_eq!(out_shape(&out), vec![c(2), c(4)]);
    assert!(
        try_run(
            &elements,
            vec![f32in(vec![c(2), c(3)]), tin(DataType::Int64, vec![c(2)])],
            13,
        )
        .is_err()
    );

    let gather_nd = with_attr(node("GatherND", 2, 1), "batch_dims", Attribute::Int(1));
    let out = run(
        &gather_nd,
        vec![
            f32in(vec![c(2), c(3), c(4)]),
            tin(DataType::Int64, vec![c(2), c(5), c(1)]),
        ],
        13,
    );
    assert_eq!(out_shape(&out), vec![c(2), c(5), c(4)]);
    assert!(
        try_run(
            &gather_nd,
            vec![
                f32in(vec![c(2), c(3)]),
                tin(DataType::Int64, vec![c(2), c(5), c(2)]),
            ],
            13,
        )
        .is_err()
    );
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

/// An input carrying a resolved integer scalar.
fn sd_int_scalar(dtype: DataType, value: DimExpr) -> NodeIo {
    NodeIo {
        type_info: Some(TypeInfo::new(dtype, vec![])),
        shape_data: Some(ShapeData::scalar(dtype, value)),
    }
}

/// A scalar input carrying a resolved floating-point constant.
fn sd_float_scalar(dt: DataType, value: f64) -> NodeIo {
    NodeIo {
        type_info: Some(TypeInfo::new(dt, vec![])),
        shape_data: Some(ShapeData::float_scalar(dt, value)),
    }
}

fn sd_float_vec(values: Vec<f64>) -> NodeIo {
    NodeIo {
        type_info: Some(TypeInfo::new(
            DataType::Float32,
            vec![c(values.len() as i64)],
        )),
        shape_data: Some(ShapeData::float_vector(DataType::Float32, values)),
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
    try_run(n, inputs, opset).unwrap()
}

fn try_run(n: &Node, inputs: Vec<NodeIo>, opset: u64) -> Result<Vec<NodeIo>, ShapeInferError> {
    let reg = InferenceRegistry::default_registry();
    let mut imports = HashMap::new();
    imports.insert(String::new(), opset);
    let mut interner = SymbolInterner::new(0x8000_0000);
    reg.infer_node(n, &imports, inputs, MergePolicy::Permissive, &mut interner)
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

#[test]
fn mod_broadcasts_and_preserves_dtype() {
    let n = node("Mod", 2, 1);
    let outs = run(
        &n,
        vec![
            tin(DataType::Int64, vec![c(3), c(1)]),
            tin(DataType::Int64, vec![c(1), c(4)]),
        ],
        10,
    );
    assert_eq!(out_shape(&outs), vec![c(3), c(4)]);
    assert_eq!(out_dtype(&outs), DataType::Int64);
}

// --- Quantized matmul ------------------------------------------------------

fn quantized_matmul_node(op: &str, domain: &str, n_in: usize, n: i64) -> Node {
    with_attr(
        with_domain(node(op, n_in, 1), domain),
        "N",
        Attribute::Int(n),
    )
}

fn assert_quantized_matmul_shapes(n: &Node, n_in: usize) {
    let packed_inputs = || (1..n_in).map(|_| NodeIo::default());

    let outs = run(
        n,
        std::iter::once(tin(DataType::Float16, vec![c(1), sym(0), c(896)]))
            .chain(packed_inputs())
            .collect(),
        1,
    );
    assert_eq!(out_shape(&outs), vec![c(1), sym(0), c(4864)]);
    assert_eq!(out_dtype(&outs), DataType::Float16);

    let outs = run(
        n,
        std::iter::once(f32in(vec![sym(1), c(896)]))
            .chain(packed_inputs())
            .collect(),
        1,
    );
    assert_eq!(out_shape(&outs), vec![sym(1), c(4864)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn block_quantized_matmul_uses_n_and_preserves_leading_dims() {
    let n = quantized_matmul_node("BlockQuantizedMatMul", "pkg.nxrt", 2, 4864);
    assert_quantized_matmul_shapes(&n, 2);
}

#[test]
fn matmul_nbits_uses_n_and_preserves_leading_dims() {
    let n = quantized_matmul_node("MatMulNBits", "com.microsoft", 3, 4864);
    assert_quantized_matmul_shapes(&n, 3);
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
    let outs = run(
        &n,
        vec![f32in(vec![c(8), c(64)]), f32in(vec![c(32), c(64)])],
        1,
    );
    assert_eq!(out_shape(&outs), vec![c(8), c(32)]);
}

#[test]
fn fused_matmul_transa() {
    // A supplied as [K, M] = [64, 8], transA=1 -> M=8; B [64, 32] -> [8, 32].
    let n = fused_matmul_node(&[("transA", 1)]);
    let outs = run(
        &n,
        vec![f32in(vec![c(64), c(8)]), f32in(vec![c(64), c(32)])],
        1,
    );
    assert_eq!(out_shape(&outs), vec![c(8), c(32)]);
}

#[test]
fn fused_matmul_transa_and_transb() {
    // A [K,M]=[64,8] transA, B [N,K]=[32,64] transB -> [8, 32].
    let n = fused_matmul_node(&[("transA", 1), ("transB", 1)]);
    let outs = run(
        &n,
        vec![f32in(vec![c(64), c(8)]), f32in(vec![c(32), c(64)])],
        1,
    );
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
    let outs = run(
        &n,
        vec![f32in(vec![c(2), c(3)]), f32in(vec![c(3), c(4)])],
        1,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(4)]);
}

#[test]
fn fused_matmul_alpha_is_shape_neutral() {
    // `alpha` scales values only; it must not affect the output shape.
    let mut n = fused_matmul_node(&[("transB", 1)]);
    n = with_attr(n, "alpha", Attribute::Float(2.0));
    let outs = run(
        &n,
        vec![f32in(vec![c(8), c(64)]), f32in(vec![c(32), c(64)])],
        1,
    );
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
fn fused_attention_pretransposed_k_concrete() {
    // com.microsoft::FusedAttention with k_transposed=1: K is already
    // [batch, heads, head_dim, seq_k]. Output == MatMul(probs, V) shape =
    // Q's leading dims + [seq_q, head_dim_v].
    // Q [2,4,3,8], K^T [2,4,8,5], V [2,4,5,16] -> out [2,4,3,16].
    let n = with_attr(
        with_domain(node("FusedAttention", 3, 1), "com.microsoft"),
        "k_transposed",
        Attribute::Int(1),
    );
    let outs = run(
        &n,
        vec![
            f32in(vec![c(2), c(4), c(3), c(8)]),
            f32in(vec![c(2), c(4), c(8), c(5)]),
            f32in(vec![c(2), c(4), c(5), c(16)]),
        ],
        1,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(4), c(3), c(16)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn fused_attention_internal_transpose_k_concrete() {
    // k_transposed unset/0: K is [batch, heads, seq_k, head_dim] and the rule
    // transposes its last two dims to form Kᵀ before the score MatMul.
    // Q [2,4,3,8], K [2,4,5,8], V [2,4,5,16] -> out [2,4,3,16].
    let n = with_domain(node("FusedAttention", 3, 1), "com.microsoft");
    let outs = run(
        &n,
        vec![
            f32in(vec![c(2), c(4), c(3), c(8)]),
            f32in(vec![c(2), c(4), c(5), c(8)]),
            f32in(vec![c(2), c(4), c(5), c(16)]),
        ],
        1,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(4), c(3), c(16)]);
}

#[test]
fn fused_attention_symbolic_batch_and_mask() {
    // Symbolic batch carries through; the optional 4th (mask) input is
    // shape-neutral. Q [B,4,S,8], K^T [B,4,8,S], V [B,4,S,8] -> out [B,4,S,8].
    let n = with_attr(
        with_domain(node("FusedAttention", 4, 1), "com.microsoft"),
        "k_transposed",
        Attribute::Int(1),
    );
    let outs = run(
        &n,
        vec![
            f32in(vec![sym(1), c(4), sym(2), c(8)]),
            f32in(vec![sym(1), c(4), c(8), sym(2)]),
            f32in(vec![sym(1), c(4), sym(2), c(8)]),
            f32in(vec![sym(1), c(1), c(1), sym(2)]),
        ],
        1,
    );
    assert_eq!(out_shape(&outs), vec![sym(1), c(4), sym(2), c(8)]);
}

#[test]
fn attention_4d_all_outputs_with_cache() {
    // Standard ai.onnx::Attention, 4D, with a past KV cache and all 4 outputs.
    // Q [2,4,3,8], K [2,4,5,8], V [2,4,5,16], past_key [2,4,7,8].
    // total_seq = 7 + 5 = 12.
    //   Y            = [2,4,3,16]
    //   present_key  = [2,4,12,8]
    //   present_value= [2,4,12,16]
    //   qk_matmul    = [2,4,3,12]
    let n = node("Attention", 5, 4);
    let outs = run(
        &n,
        vec![
            f32in(vec![c(2), c(4), c(3), c(8)]),
            f32in(vec![c(2), c(4), c(5), c(8)]),
            f32in(vec![c(2), c(4), c(5), c(16)]),
            NodeIo::default(),                   // attn_mask (skipped)
            f32in(vec![c(2), c(4), c(7), c(8)]), // past_key
        ],
        23,
    );
    let shape_i = |i: usize| outs[i].type_info.as_ref().unwrap().shape.clone();
    assert_eq!(shape_i(0), vec![c(2), c(4), c(3), c(16)]);
    assert_eq!(shape_i(1), vec![c(2), c(4), c(12), c(8)]);
    assert_eq!(shape_i(2), vec![c(2), c(4), c(12), c(16)]);
    assert_eq!(shape_i(3), vec![c(2), c(4), c(3), c(12)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn attention_3d_reshapes_hidden_by_num_heads() {
    // 3D inputs: Q [2,S,32] with q_num_heads=4 -> head_size=8; V [2,S,32]
    // with kv_num_heads=4 -> v_head_size=8. Y hidden = q_heads*v_head_size = 32.
    let n = with_attr(
        with_attr(node("Attention", 3, 1), "q_num_heads", Attribute::Int(4)),
        "kv_num_heads",
        Attribute::Int(4),
    );
    let outs = run(
        &n,
        vec![
            f32in(vec![c(2), sym(2), c(32)]),
            f32in(vec![c(2), sym(3), c(32)]),
            f32in(vec![c(2), sym(3), c(32)]),
        ],
        23,
    );
    assert_eq!(out_shape(&outs), vec![c(2), sym(2), c(32)]);
}

#[test]
fn attention_gqa_present_uses_kv_heads() {
    // GQA: q_heads=4, kv_heads=2. present_key/value carry kv_heads, not q_heads.
    // Q [1,4,S,8], K [1,2,S,8], V [1,2,S,8].
    let n = node("Attention", 3, 3);
    let outs = run(
        &n,
        vec![
            f32in(vec![c(1), c(4), sym(2), c(8)]),
            f32in(vec![c(1), c(2), sym(2), c(8)]),
            f32in(vec![c(1), c(2), sym(2), c(8)]),
        ],
        23,
    );
    let shape_i = |i: usize| outs[i].type_info.as_ref().unwrap().shape.clone();
    assert_eq!(shape_i(0), vec![c(1), c(4), sym(2), c(8)]);
    assert_eq!(shape_i(1), vec![c(1), c(2), sym(2), c(8)]);
    assert_eq!(shape_i(2), vec![c(1), c(2), sym(2), c(8)]);
}

#[test]
fn attention_resolves_for_opsets_23_through_26() {
    // The opset-23 rule serves model opsets 24, 25 and 26 too (the registry
    // resolves the highest `min_opset <= version`). Y is sized at every opset.
    let n = node("Attention", 3, 1);
    for opset in [23, 24, 25, 26] {
        let outs = run(
            &n,
            vec![
                f32in(vec![c(1), c(2), c(3), c(8)]),
                f32in(vec![c(1), c(2), c(5), c(8)]),
                f32in(vec![c(1), c(2), c(5), c(16)]),
            ],
            opset,
        );
        assert_eq!(
            out_shape(&outs),
            vec![c(1), c(2), c(3), c(16)],
            "Y shape wrong at opset {opset}"
        );
    }
}

#[test]
fn attention_opset24_nonpad_external_cache_no_past_concat() {
    // opset-24 external-cache path: `nonpad_kv_seqlen` (7th input) with no
    // past_key, so total_seq == kv_seq of K (no concat). All four outputs sized.
    // Q [1,2,3,8], K [1,2,5,8], V [1,2,5,16] -> total_seq = 5.
    let n = node("Attention", 7, 4);
    let outs = run(
        &n,
        vec![
            f32in(vec![c(1), c(2), c(3), c(8)]),
            f32in(vec![c(1), c(2), c(5), c(8)]),
            f32in(vec![c(1), c(2), c(5), c(16)]),
            NodeIo::default(),                // attn_mask (skipped)
            NodeIo::default(),                // past_key (absent)
            NodeIo::default(),                // past_value (absent)
            tin(DataType::Int64, vec![c(1)]), // nonpad_kv_seqlen
        ],
        24,
    );
    let shape_i = |i: usize| outs[i].type_info.as_ref().unwrap().shape.clone();
    assert_eq!(shape_i(0), vec![c(1), c(2), c(3), c(16)]);
    assert_eq!(shape_i(1), vec![c(1), c(2), c(5), c(8)]);
    assert_eq!(shape_i(2), vec![c(1), c(2), c(5), c(16)]);
    assert_eq!(shape_i(3), vec![c(1), c(2), c(3), c(5)]);
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
    let outs = run(
        &n,
        vec![f32in(vec![anon.clone()]), f32in(vec![named.clone()])],
        13,
    );
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

#[test]
fn round3_math_schemas_have_shape_rules() {
    let binary_inputs = vec![f32in(vec![sym(0), c(1), c(64)]), f32in(vec![c(8), c(64)])];
    for op in ["Sub", "Div", "Mod"] {
        let outs = run(&node(op, 2, 1), binary_inputs.clone(), 14);
        assert_eq!(out_shape(&outs), vec![sym(0), c(8), c(64)], "{op}");
        assert_eq!(out_dtype(&outs), DataType::Float32, "{op}");
    }
    for op in ["Neg", "Abs"] {
        let outs = run(
            &node(op, 1, 1),
            vec![tin(DataType::Int32, vec![sym(0), c(64)])],
            13,
        );
        assert_eq!(out_shape(&outs), vec![sym(0), c(64)], "{op}");
        assert_eq!(out_dtype(&outs), DataType::Int32, "{op}");
    }
}

#[test]
fn acos_passthrough() {
    let n = node("Acos", 1, 1);
    let outs = run(&n, vec![f32in(vec![sym(0), c(8), c(768)])], 7);
    assert_eq!(out_shape(&outs), vec![sym(0), c(8), c(768)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

// --- Selection ------------------------------------------------------------

#[test]
fn argmax_keepdims_variants_return_int64() {
    let input = f32in(vec![c(2), c(3), c(4)]);
    let keep = with_attr(node("ArgMax", 1, 1), "axis", Attribute::Int(1));
    let outs = run(&keep, vec![input.clone()], 13);
    assert_eq!(out_shape(&outs), vec![c(2), c(1), c(4)]);
    assert_eq!(out_dtype(&outs), DataType::Int64);

    let drop = with_attr(
        with_attr(node("ArgMax", 1, 1), "axis", Attribute::Int(1)),
        "keepdims",
        Attribute::Int(0),
    );
    let outs = run(&drop, vec![input], 13);
    assert_eq!(out_shape(&outs), vec![c(2), c(4)]);
    assert_eq!(out_dtype(&outs), DataType::Int64);
}

#[test]
fn argmin_returns_int64() {
    let n = with_attr(node("ArgMin", 1, 1), "keepdims", Attribute::Int(0));
    let outs = run(&n, vec![f32in(vec![c(2), c(3)])], 12);
    assert_eq!(out_shape(&outs), vec![c(3)]);
    assert_eq!(out_dtype(&outs), DataType::Int64);
}

#[test]
fn topk_outputs_and_dynamic_k() {
    let n = with_attr(node("TopK", 2, 2), "axis", Attribute::Int(-1));
    let outs = run(&n, vec![f32in(vec![c(2), c(8)]), sd_vec(vec![c(3)])], 11);
    assert_eq!(out_shape(&outs), vec![c(2), c(3)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
    assert_eq!(outs[1].type_info.as_ref().unwrap().shape, vec![c(2), c(3)]);
    assert_eq!(outs[1].type_info.as_ref().unwrap().dtype, DataType::Int64);

    let outs = run(
        &n,
        vec![f32in(vec![c(2), c(8)]), tin(DataType::Int64, vec![])],
        11,
    );
    let shape = out_shape(&outs);
    assert_eq!(shape.len(), 2);
    assert_eq!(shape[0], c(2));
    assert!(shape[1].as_symbol().is_some());
    assert_eq!(outs[1].type_info.as_ref().unwrap().dtype, DataType::Int64);
}

#[test]
fn topk_v1_reads_k_attribute() {
    let n = with_attr(node("TopK", 1, 2), "k", Attribute::Int(2));
    let outs = run(&n, vec![f32in(vec![c(3), c(8)])], 1);
    assert_eq!(out_shape(&outs), vec![c(3), c(2)]);
    assert_eq!(outs[1].type_info.as_ref().unwrap().dtype, DataType::Int64);
}

#[test]
fn topk_replaces_the_selected_middle_axis_and_rejects_invalid_axes() {
    let middle = with_attr(node("TopK", 2, 2), "axis", Attribute::Int(1));
    let outs = run(
        &middle,
        vec![f32in(vec![c(2), c(8), c(4)]), sd_vec(vec![c(3)])],
        11,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(3), c(4)]);
    assert_eq!(
        outs[1].type_info.as_ref().unwrap().shape,
        vec![c(2), c(3), c(4)]
    );

    let invalid = with_attr(node("TopK", 2, 2), "axis", Attribute::Int(3));
    assert!(
        try_run(
            &invalid,
            vec![f32in(vec![c(2), c(8), c(4)]), sd_vec(vec![c(3)])],
            11,
        )
        .is_err()
    );
}

#[test]
fn axis_operators_reject_out_of_range_and_duplicate_axes() {
    let input = f32in(vec![c(2), c(3), c(4)]);
    assert!(
        try_run(
            &with_attr(node("ArgMax", 1, 1), "axis", Attribute::Int(-4)),
            vec![input.clone()],
            13,
        )
        .is_err()
    );
    assert!(
        try_run(
            &with_attr(
                node("Transpose", 1, 1),
                "perm",
                Attribute::Ints(vec![0, 1, 3]),
            ),
            vec![input.clone()],
            13,
        )
        .is_err()
    );
    assert!(
        try_run(
            &with_attr(
                node("Transpose", 1, 1),
                "perm",
                Attribute::Ints(vec![0, 1, 1]),
            ),
            vec![input.clone()],
            13,
        )
        .is_err()
    );
    assert!(
        try_run(
            &node("Unsqueeze", 2, 1),
            vec![input.clone(), sd_vec(vec![c(4)])],
            13,
        )
        .is_err()
    );
    assert!(
        try_run(
            &node("Unsqueeze", 2, 1),
            vec![input.clone(), sd_vec(vec![c(0), c(0)])],
            13,
        )
        .is_err()
    );
    assert!(
        try_run(
            &with_attr(node("Gather", 2, 1), "axis", Attribute::Int(3)),
            vec![input, tin(DataType::Int64, vec![])],
            13,
        )
        .is_err()
    );
}

#[test]
fn tile_static_repeats() {
    let n = node("Tile", 2, 1);
    let outs = run(
        &n,
        vec![
            f32in(vec![c(2), c(3), c(4)]),
            sd_vec(vec![c(1), c(2), c(3)]),
        ],
        13,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(6), c(12)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn tile_unknown_repeats_keeps_rank() {
    // `repeats` has no shape-data (runtime-computed): every extent degrades to a
    // fresh symbol, but the rank stays == rank(input).
    let n = node("Tile", 2, 1);
    let outs = run(
        &n,
        vec![f32in(vec![c(2), c(3)]), tin(DataType::Int64, vec![c(2)])],
        13,
    );
    let shape = out_shape(&outs);
    assert_eq!(shape.len(), 2);
    assert!(shape[0].as_symbol().is_some());
    assert!(shape[1].as_symbol().is_some());
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn tile_rejects_non_vector_repeats_and_extent_overflow() {
    let n = node("Tile", 2, 1);
    assert!(
        try_run(
            &n,
            vec![
                f32in(vec![c(2), c(3)]),
                tin(DataType::Int64, vec![c(1), c(2)]),
            ],
            13,
        )
        .is_err()
    );
    assert!(
        try_run(
            &n,
            vec![f32in(vec![c(isize::MAX as i64)]), sd_vec(vec![c(2)]),],
            13,
        )
        .is_err()
    );
}

#[test]
fn range_static_and_dynamic() {
    let n = node("Range", 3, 1);
    let scalar = |value| NodeIo {
        type_info: Some(TypeInfo::new(DataType::Int64, vec![])),
        shape_data: Some(ShapeData::scalar(DataType::Int64, c(value))),
    };
    let outs = run(&n, vec![scalar(1), scalar(10), scalar(2)], 11);
    assert_eq!(out_shape(&outs), vec![c(5)]);
    assert_eq!(out_dtype(&outs), DataType::Int64);

    let outs = run(
        &n,
        vec![
            tin(DataType::Int64, vec![]),
            tin(DataType::Int64, vec![]),
            tin(DataType::Int64, vec![]),
        ],
        11,
    );
    let shape = out_shape(&outs);
    assert_eq!(shape.len(), 1);
    assert!(shape[0].as_symbol().is_some());
}

#[test]
fn range_float_positive_delta() {
    // start=0.0, limit=1.0, delta=0.3 -> ceil(1.0 / 0.3) = ceil(3.33) = 4
    let n = node("Range", 3, 1);
    let outs = run(
        &n,
        vec![
            sd_float_scalar(DataType::Float32, 0.0),
            sd_float_scalar(DataType::Float32, 1.0),
            sd_float_scalar(DataType::Float32, 0.3),
        ],
        11,
    );
    assert_eq!(out_shape(&outs), vec![c(4)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn range_float32_uses_cpu_kernel_arithmetic() {
    // Keep in sync with `float_range_count`: f32 arithmetic makes this 25,
    // whereas f64 arithmetic on this f32 round-trip yields 26.
    let n = node("Range", 3, 1);
    let outs = run(
        &n,
        vec![
            sd_float_scalar(DataType::Float32, 0.0),
            sd_float_scalar(DataType::Float32, 1.0),
            sd_float_scalar(DataType::Float32, f64::from(0.04_f32)),
        ],
        11,
    );
    assert_eq!(out_shape(&outs), vec![c(25)]);
}

#[test]
fn range_float_negative_delta() {
    // start=10.0, limit=2.0, delta=-2.5 -> ceil(-8.0 / -2.5) = ceil(3.2) = 4
    let n = node("Range", 3, 1);
    let outs = run(
        &n,
        vec![
            sd_float_scalar(DataType::Float64, 10.0),
            sd_float_scalar(DataType::Float64, 2.0),
            sd_float_scalar(DataType::Float64, -2.5),
        ],
        11,
    );
    assert_eq!(out_shape(&outs), vec![c(4)]);
    assert_eq!(out_dtype(&outs), DataType::Float64);
}

#[test]
fn range_float64_rejects_two_to_the_63_length() {
    let n = node("Range", 3, 1);
    let error = try_run(
        &n,
        vec![
            sd_float_scalar(DataType::Float64, 0.0),
            sd_float_scalar(DataType::Float64, 2_f64.powi(63)),
            sd_float_scalar(DataType::Float64, 1.0),
        ],
        11,
    )
    .unwrap_err();
    assert!(error.to_string().contains("exceeds isize::MAX"));
}

#[test]
fn range_float_dynamic() {
    // Non-constant float operands (typed but no shape-data) -> unknown length.
    let n = node("Range", 3, 1);
    let outs = run(
        &n,
        vec![
            tin(DataType::Float32, vec![]),
            tin(DataType::Float32, vec![]),
            tin(DataType::Float32, vec![]),
        ],
        11,
    );
    let shape = out_shape(&outs);
    assert_eq!(shape.len(), 1);
    assert!(shape[0].as_symbol().is_some());
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn cumsum_passthrough() {
    let n = node("CumSum", 2, 1);
    let outs = run(
        &n,
        vec![f32in(vec![sym(0), c(8)]), tin(DataType::Int64, vec![])],
        14,
    );
    assert_eq!(out_shape(&outs), vec![sym(0), c(8)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn squeeze_and_unsqueeze_static_axes_and_dynamic_axes() {
    let squeeze = node("Squeeze", 2, 1);
    let outs = run(
        &squeeze,
        vec![f32in(vec![c(2), c(1), c(4)]), sd_vec(vec![c(1)])],
        24,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(4)]);

    let dynamic_axes = run(
        &squeeze,
        vec![
            f32in(vec![c(2), c(1), c(4)]),
            tin(DataType::Int64, vec![sym(0)]),
        ],
        24,
    );
    assert!(dynamic_axes[0].type_info.is_none());

    let unsqueeze = node("Unsqueeze", 2, 1);
    let outs = run(
        &unsqueeze,
        vec![f32in(vec![c(2), c(4)]), sd_vec(vec![c(1)])],
        24,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(1), c(4)]);
    let dynamic_axes = run(
        &unsqueeze,
        vec![f32in(vec![c(2), c(4)]), tin(DataType::Int64, vec![sym(0)])],
        24,
    );
    assert!(dynamic_axes[0].type_info.is_none());
}

#[test]
fn nonzero_rank_and_dynamic_nnz() {
    let n = node("NonZero", 1, 1);
    let outs = run(&n, vec![f32in(vec![c(2), c(3), c(4)])], 13);
    let shape = out_shape(&outs);
    assert_eq!(shape[0], c(3));
    assert!(shape[1].as_symbol().is_some());
    assert_eq!(out_dtype(&outs), DataType::Int64);
}

#[test]
fn one_hot_inserts_known_depth_at_axis_for_opsets_9_and_11() {
    for opset in [9, 11] {
        for (axis, expected) in [
            (0, vec![c(5), c(2), c(3)]),
            (1, vec![c(2), c(5), c(3)]),
            (-1, vec![c(2), c(3), c(5)]),
            (-2, vec![c(2), c(5), c(3)]),
        ] {
            let n = with_attr(node("OneHot", 3, 1), "axis", Attribute::Int(axis));
            let outs = run(
                &n,
                vec![
                    tin(DataType::Int64, vec![c(2), c(3)]),
                    sd_int_scalar(DataType::Int64, c(5)),
                    tin(DataType::Float16, vec![c(2)]),
                ],
                opset,
            );
            assert_eq!(out_shape(&outs), expected, "opset {opset}, axis {axis}");
            assert_eq!(out_dtype(&outs), DataType::Float16);
        }
    }
}

#[test]
fn one_hot_preserves_symbolic_indices_and_handles_dynamic_depth() {
    let n = node("OneHot", 3, 1);
    let outs = run(
        &n,
        vec![
            tin(DataType::Int32, vec![sym(0), c(3)]),
            tin(DataType::Int64, vec![]),
            tin(DataType::Int32, vec![c(2)]),
        ],
        11,
    );
    let shape = out_shape(&outs);
    assert_eq!(shape.len(), 3);
    assert_eq!(shape[0], sym(0));
    assert_eq!(shape[1], c(3));
    assert!(shape[2].as_symbol().is_some());
    assert_eq!(out_dtype(&outs), DataType::Int32);

    let outs = run(
        &n,
        vec![
            tin(DataType::Int32, vec![sym(0)]),
            sd_int_scalar(DataType::Int64, sym(7)),
            tin(DataType::Uint8, vec![c(2)]),
        ],
        11,
    );
    assert_eq!(out_shape(&outs), vec![sym(0), sym(7)]);
    assert_eq!(out_dtype(&outs), DataType::Uint8);
}

#[test]
fn one_hot_rejects_invalid_axis_and_values_length() {
    let inputs = || {
        vec![
            tin(DataType::Int64, vec![c(2)]),
            sd_int_scalar(DataType::Int64, c(4)),
            tin(DataType::Float32, vec![c(2)]),
        ]
    };
    let n = with_attr(node("OneHot", 3, 1), "axis", Attribute::Int(2));
    assert!(try_run(&n, inputs(), 11).is_err());

    let n = node("OneHot", 3, 1);
    let mut bad_values = inputs();
    bad_values[2] = tin(DataType::Float32, vec![c(3)]);
    assert!(try_run(&n, bad_values, 11).is_err());
}

#[test]
fn compress_axis_and_flatten_variants_for_opsets_9_and_11() {
    for opset in [9, 11] {
        let n = with_attr(node("Compress", 2, 1), "axis", Attribute::Int(-2));
        let outs = run(
            &n,
            vec![
                tin(DataType::Float16, vec![sym(0), c(3), c(4)]),
                tin(DataType::Bool, vec![c(3)]),
            ],
            opset,
        );
        let shape = out_shape(&outs);
        assert_eq!(shape.len(), 3);
        assert_eq!(shape[0], sym(0));
        assert!(shape[1].as_symbol().is_some());
        assert_eq!(shape[2], c(4));
        assert_eq!(out_dtype(&outs), DataType::Float16);

        let n = node("Compress", 2, 1);
        let outs = run(
            &n,
            vec![
                tin(DataType::Int32, vec![c(2), c(3), c(4)]),
                tin(DataType::Bool, vec![sym(1)]),
            ],
            opset,
        );
        let shape = out_shape(&outs);
        assert_eq!(shape.len(), 1);
        assert!(shape[0].as_symbol().is_some());
        assert_eq!(out_dtype(&outs), DataType::Int32);
    }
}

#[test]
fn compress_rejects_invalid_axis_and_condition_rank() {
    let n = with_attr(node("Compress", 2, 1), "axis", Attribute::Int(2));
    assert!(
        try_run(
            &n,
            vec![f32in(vec![c(2), c(3)]), tin(DataType::Bool, vec![c(3)])],
            11
        )
        .is_err()
    );

    let n = node("Compress", 2, 1);
    assert!(
        try_run(
            &n,
            vec![
                f32in(vec![c(2), c(3)]),
                tin(DataType::Bool, vec![c(1), c(3)])
            ],
            11
        )
        .is_err()
    );
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
fn reshape_rejects_multiple_inferred_dimensions_and_product_mismatches() {
    let n = node("Reshape", 2, 1);

    // ONNX permits at most one -1 in the target shape.
    assert_invalid(
        try_run(
            &n,
            vec![f32in(vec![c(2), c(3)]), sd_vec(vec![c(-1), c(-1)])],
            13,
        )
        .unwrap_err(),
        "Reshape",
        "at most one dimension may be -1",
    );

    // A fully concrete target must preserve the element count.
    assert_invalid(
        try_run(&n, vec![f32in(vec![c(2), c(3)]), sd_vec(vec![c(4)])], 13).unwrap_err(),
        "Reshape",
        "input element count 6 does not match target element count 4",
    );
}

#[test]
fn reshape_validates_static_target_values_without_guessing_dynamic_targets() {
    let n = node("Reshape", 2, 1);
    assert_invalid(
        try_run(&n, vec![f32in(vec![c(2), c(3)]), sd_vec(vec![c(-2)])], 13).unwrap_err(),
        "Reshape",
        "target dimension -2 is invalid",
    );
    assert_invalid(
        try_run(
            &n,
            vec![f32in(vec![c(2), c(3)]), sd_vec(vec![c(-1), c(4)])],
            13,
        )
        .unwrap_err(),
        "Reshape",
        "input element count is not divisible",
    );
    assert_invalid(
        try_run(&n, vec![f32in(vec![c(2)]), sd_vec(vec![c(0), c(0)])], 13).unwrap_err(),
        "Reshape",
        "0 at target index 1 has no corresponding input dimension",
    );

    let allowzero = with_attr(node("Reshape", 2, 1), "allowzero", Attribute::Int(1));
    assert_invalid(
        try_run(
            &allowzero,
            vec![f32in(vec![c(0)]), sd_vec(vec![c(0), c(-1)])],
            14,
        )
        .unwrap_err(),
        "Reshape",
        "allowzero=1 does not permit 0 and -1",
    );
    let zero = run(&allowzero, vec![f32in(vec![c(0)]), sd_vec(vec![c(0)])], 14);
    assert_eq!(out_shape(&zero), vec![c(0)]);

    let dynamic = run(
        &n,
        vec![f32in(vec![c(2), c(3)]), tin(DataType::Int64, vec![c(2)])],
        13,
    );
    assert!(dynamic[0].type_info.is_none());
}

#[test]
fn reshape_rejects_indeterminate_minus_one_with_zero_product() {
    let error = try_run(
        &node("Reshape", 2, 1),
        vec![f32in(vec![c(0), c(3)]), sd_vec(vec![c(0), c(-1)])],
        13,
    )
    .unwrap_err();
    assert_invalid(
        error,
        "Reshape",
        "cannot infer -1 dimension when the remaining target product is zero",
    );
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
fn size_rejects_total_above_isize_max() {
    // A concrete tensor extent that cannot be represented by Rust indexing must
    // be rejected rather than wrapped or lowered to a bogus static dimension.
    let n = node("Size", 1, 1);
    let big = c(1 << 20);
    let error = try_run(
        &n,
        vec![f32in(vec![big.clone(), big.clone(), big.clone(), big])],
        13,
    )
    .unwrap_err();
    assert!(error.to_string().contains("exceeds isize::MAX"));
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

#[test]
fn trilu_preserves_known_and_symbolic_shape_and_dtype() {
    for upper in [0, 1] {
        let n = with_attr(node("Trilu", 2, 1), "upper", Attribute::Int(upper));
        let outs = run(
            &n,
            vec![
                tin(DataType::Float16, vec![sym(0), c(3), c(4)]),
                sd_int_scalar(DataType::Int64, c(-1)),
            ],
            14,
        );
        assert_eq!(out_shape(&outs), vec![sym(0), c(3), c(4)]);
        assert_eq!(out_dtype(&outs), DataType::Float16);
    }
}

#[test]
fn depth_to_space_known_dims_and_modes_across_schema_versions() {
    let n = with_attr(node("DepthToSpace", 1, 1), "blocksize", Attribute::Int(2));
    let outs = run(
        &n,
        vec![tin(DataType::Uint8, vec![c(2), c(12), c(5), c(7)])],
        1,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(3), c(10), c(14)]);
    assert_eq!(out_dtype(&outs), DataType::Uint8);

    for opset in [11, 13] {
        for mode in ["DCR", "CRD"] {
            let n = with_attr(
                with_attr(node("DepthToSpace", 1, 1), "blocksize", Attribute::Int(2)),
                "mode",
                Attribute::String(mode.as_bytes().to_vec()),
            );
            let outs = run(
                &n,
                vec![tin(DataType::Uint8, vec![c(2), c(12), c(5), c(7)])],
                opset,
            );
            assert_eq!(out_shape(&outs), vec![c(2), c(3), c(10), c(14)]);
            assert_eq!(out_dtype(&outs), DataType::Uint8);
        }
    }
}

#[test]
fn depth_to_space_handles_symbolic_dims_without_panicking() {
    let n = with_attr(node("DepthToSpace", 1, 1), "blocksize", Attribute::Int(2));
    let outs = run(&n, vec![f32in(vec![sym(0), sym(1), sym(2), c(7)])], 13);
    let shape = out_shape(&outs);
    assert_eq!(shape.len(), 4);
    assert_eq!(shape[0], sym(0));
    assert!(shape[1].as_symbol().is_some());
    assert_eq!(shape[2].as_const(), None);
    assert_eq!(shape[3], c(14));
}

#[test]
fn depth_to_space_rejects_non_divisible_channels() {
    let n = with_attr(node("DepthToSpace", 1, 1), "blocksize", Attribute::Int(2));
    assert!(try_run(&n, vec![f32in(vec![c(1), c(10), c(4), c(4)])], 13).is_err());
}

#[test]
fn space_to_depth_known_dims_across_schema_versions() {
    for opset in [1, 13] {
        let n = with_attr(node("SpaceToDepth", 1, 1), "blocksize", Attribute::Int(2));
        let outs = run(
            &n,
            vec![tin(DataType::Int32, vec![c(2), c(3), c(10), c(14)])],
            opset,
        );
        assert_eq!(out_shape(&outs), vec![c(2), c(12), c(5), c(7)]);
        assert_eq!(out_dtype(&outs), DataType::Int32);
    }
}

#[test]
fn space_to_depth_handles_symbolic_dims_without_panicking() {
    let n = with_attr(node("SpaceToDepth", 1, 1), "blocksize", Attribute::Int(2));
    let outs = run(&n, vec![f32in(vec![sym(0), sym(1), sym(2), c(14)])], 13);
    let shape = out_shape(&outs);
    assert_eq!(shape.len(), 4);
    assert_eq!(shape[0], sym(0));
    assert_eq!(shape[1].as_const(), None);
    assert!(shape[2].as_symbol().is_some());
    assert_eq!(shape[3], c(7));
}

#[test]
fn space_to_depth_rejects_non_divisible_spatial_dims() {
    let n = with_attr(node("SpaceToDepth", 1, 1), "blocksize", Attribute::Int(2));
    assert!(try_run(&n, vec![f32in(vec![c(1), c(3), c(5), c(8)])], 13).is_err());
    assert!(try_run(&n, vec![f32in(vec![c(1), c(3), c(8), c(5)])], 13).is_err());
}

#[test]
fn spatial_rearrangements_reject_blocksize_square_overflow() {
    for op in ["DepthToSpace", "SpaceToDepth"] {
        let n = with_attr(node(op, 1, 1), "blocksize", Attribute::Int(i64::MAX));
        assert!(
            try_run(&n, vec![f32in(vec![c(1), c(4), c(4), c(4)])], 13).is_err(),
            "{op}"
        );
    }
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

#[test]
fn gather_nd_canonical_shape() {
    // data [2, 3, 4], indices [5, 2] -> [5, 4].
    let n = node("GatherND", 2, 1);
    let outs = run(
        &n,
        vec![
            f32in(vec![c(2), c(3), c(4)]),
            tin(DataType::Int64, vec![c(5), c(2)]),
        ],
        13,
    );
    assert_eq!(out_shape(&outs), vec![c(5), c(4)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

// --- Scatter --------------------------------------------------------------

#[test]
fn scatter_nd_preserves_data_shape_and_dtype() {
    let n = node("ScatterND", 3, 1);
    let outs = run(
        &n,
        vec![
            tin(DataType::Float16, vec![c(2), c(3), c(4)]),
            tin(DataType::Int64, vec![c(5), c(2)]),
            tin(DataType::Float16, vec![c(5), c(4)]),
        ],
        18,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(3), c(4)]);
    assert_eq!(out_dtype(&outs), DataType::Float16);
}

#[test]
fn scatter_elements_non_default_axis_preserves_data_shape() {
    let n = with_attr(node("ScatterElements", 3, 1), "axis", Attribute::Int(-2));
    let outs = run(
        &n,
        vec![
            tin(DataType::Int32, vec![c(2), c(3), c(4)]),
            tin(DataType::Int64, vec![c(2), c(1), c(4)]),
            tin(DataType::Int32, vec![c(2), c(1), c(4)]),
        ],
        16,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(3), c(4)]);
    assert_eq!(out_dtype(&outs), DataType::Int32);
}

#[test]
fn scatter_deprecated_alias_preserves_data_shape_and_dtype() {
    let n = with_attr(node("Scatter", 3, 1), "axis", Attribute::Int(1));
    let outs = run(
        &n,
        vec![
            tin(DataType::Float64, vec![c(2), c(3), c(4)]),
            tin(DataType::Int64, vec![c(2), c(1), c(4)]),
            tin(DataType::Float64, vec![c(2), c(1), c(4)]),
        ],
        9,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(3), c(4)]);
    assert_eq!(out_dtype(&outs), DataType::Float64);
}

#[test]
fn scatter_unknown_data_shape_leaves_output_unresolved() {
    for (op, opset) in [("Scatter", 9), ("ScatterElements", 16), ("ScatterND", 18)] {
        let n = node(op, 3, 1);
        let outs = run(
            &n,
            vec![
                NodeIo::default(),
                tin(DataType::Int64, vec![c(2), c(1)]),
                f32in(vec![c(2)]),
            ],
            opset,
        );
        assert!(outs[0].type_info.is_none(), "{op}");
    }
}

#[test]
fn scatter_rank_relations_are_validated() {
    let elements = node("ScatterElements", 3, 1);
    assert!(
        try_run(
            &elements,
            vec![
                f32in(vec![c(2), c(3)]),
                tin(DataType::Int64, vec![c(2)]),
                f32in(vec![c(2)]),
            ],
            18,
        )
        .is_err()
    );

    let nd = node("ScatterND", 3, 1);
    assert!(
        try_run(
            &nd,
            vec![
                f32in(vec![c(2), c(3), c(4)]),
                tin(DataType::Int64, vec![c(5), c(2)]),
                f32in(vec![c(5)]),
            ],
            18,
        )
        .is_err()
    );
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

#[test]
fn concat_dynamic_axis_is_unresolved_and_other_dims_must_match() {
    let n = with_attr(node("Concat", 2, 1), "axis", Attribute::Int(-1));
    let outs = run(
        &n,
        vec![f32in(vec![c(2), sym(0)]), f32in(vec![c(2), c(5)])],
        13,
    );
    let shape = out_shape(&outs);
    assert_eq!(shape[0], c(2));
    assert!(shape[1].as_symbol().is_some());

    assert!(
        try_run(
            &n,
            vec![f32in(vec![c(2), c(3)]), f32in(vec![c(4), c(5)])],
            13,
        )
        .is_err()
    );
}

#[test]
fn concat_rejects_axis_sum_beyond_isize_max() {
    let n = with_attr(node("Concat", 2, 1), "axis", Attribute::Int(0));
    assert!(
        try_run(
            &n,
            vec![f32in(vec![c(isize::MAX as i64)]), f32in(vec![c(1)]),],
            13,
        )
        .is_err()
    );
}

#[test]
fn concat_symbolic_axis_rejects_overflowing_known_partial_and_stays_unresolved_normally() {
    let n = with_attr(node("Concat", 3, 1), "axis", Attribute::Int(0));
    let error = try_run(
        &n,
        vec![
            f32in(vec![c(isize::MAX as i64)]),
            f32in(vec![sym(0)]),
            f32in(vec![c(1)]),
        ],
        13,
    )
    .unwrap_err();
    assert!(error.to_string().contains("exceeds isize::MAX"));

    let outs = run(
        &n,
        vec![f32in(vec![c(2)]), f32in(vec![sym(0)]), f32in(vec![c(3)])],
        13,
    );
    let shape = out_shape(&outs);
    let axis = &shape[0];
    assert!(axis.as_const().is_none());
    assert!(axis.as_symbol().is_some());
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

#[test]
fn squeeze_static_axes_reject_invalid_dims_and_leave_dynamic_dims_unresolved() {
    let axes_input = node("Squeeze", 2, 1);
    let err = try_run(
        &axes_input,
        vec![f32in(vec![c(1), c(8)]), sd_vec(vec![c(2)])],
        13,
    )
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("axis 2 is out of range for rank 2")
    );

    let err = try_run(
        &axes_input,
        vec![f32in(vec![c(1), c(8)]), sd_vec(vec![c(1)])],
        13,
    )
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("cannot squeeze axis 1 with non-singleton extent 8")
    );

    let outs = run(
        &axes_input,
        vec![f32in(vec![c(1), c(8)]), sd_vec(vec![c(0)])],
        13,
    );
    assert_eq!(out_shape(&outs), vec![c(8)]);

    let dynamic_extent = run(
        &axes_input,
        vec![f32in(vec![sym(0), c(8)]), sd_vec(vec![c(0)])],
        13,
    );
    assert!(dynamic_extent[0].type_info.is_none());

    let dynamic_axes = run(
        &axes_input,
        vec![f32in(vec![c(1), c(8)]), tin(DataType::Int64, vec![sym(0)])],
        13,
    );
    assert!(dynamic_axes[0].type_info.is_none());
}

#[test]
fn squeeze_static_axes_validate_structure_before_dynamic_extents() {
    let axes_input = node("Squeeze", 2, 1);
    let input = f32in(vec![sym(0), c(1)]);

    let err = try_run(
        &axes_input,
        vec![input.clone(), sd_vec(vec![c(0), c(0)])],
        13,
    )
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("axis 0 is specified more than once")
    );

    let err = try_run(&axes_input, vec![input, sd_vec(vec![c(0), c(2)])], 13).unwrap_err();
    assert!(
        err.to_string()
            .contains("axis 2 is out of range for rank 2")
    );
}

#[test]
fn squeeze_v11_rejects_duplicate_static_axes() {
    let n = with_attr(node("Squeeze", 1, 1), "axes", Attribute::Ints(vec![0, 0]));
    let err = try_run(&n, vec![f32in(vec![c(1), c(8)])], 11).unwrap_err();
    assert!(
        err.to_string()
            .contains("axis 0 is specified more than once")
    );
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
    assert!(shape[0].as_symbol().is_some());
    assert_eq!(shape[1], c(768));
}

#[test]
fn slice_dynamic_bounds_only_clear_selected_axes() {
    let n = node("Slice", 4, 1);
    let outs = run(
        &n,
        vec![
            f32in(vec![c(10), c(20)]),
            tin(DataType::Int64, vec![c(1)]),
            tin(DataType::Int64, vec![c(1)]),
            sd_vec(vec![c(1)]),
        ],
        13,
    );
    let shape = out_shape(&outs);
    assert_eq!(shape[0], c(10));
    assert!(shape[1].as_symbol().is_some());
}

#[test]
fn slice_dynamic_axes_clear_every_extent() {
    let n = node("Slice", 4, 1);
    let outs = run(
        &n,
        vec![
            f32in(vec![c(10), c(20)]),
            sd_vec(vec![c(0)]),
            sd_vec(vec![c(5)]),
            tin(DataType::Int64, vec![c(1)]),
        ],
        13,
    );
    assert!(
        out_shape(&outs)
            .iter()
            .all(|extent| extent.as_symbol().is_some())
    );
}

#[test]
fn slice_negative_step_clamps_extreme_bounds() {
    let n = node("Slice", 5, 1);
    let outs = run(
        &n,
        vec![
            f32in(vec![c(5)]),
            sd_vec(vec![c(i64::MAX)]),
            sd_vec(vec![c(i64::MIN)]),
            sd_vec(vec![c(0)]),
            sd_vec(vec![c(-1)]),
        ],
        13,
    );
    assert_eq!(out_shape(&outs), vec![c(5)]);
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

#[test]
fn skip_layer_norm_emits_x_shaped_skip_bias_sum() {
    // com.microsoft SkipLayerNormalization with all four outputs: output 0 and
    // output 3 (input_skip_bias_sum) are X-shaped; mean/inv_std collapse last.
    let n = with_domain(node("SkipLayerNormalization", 3, 4), "com.microsoft");
    let outs = run(
        &n,
        vec![
            f32in(vec![sym(0), c(8), c(768)]),
            f32in(vec![sym(0), c(8), c(768)]),
            f32in(vec![c(768)]),
        ],
        1,
    );
    assert_eq!(out_shape(&outs), vec![sym(0), c(8), c(768)]);
    let mean = outs[1].type_info.as_ref().unwrap().shape.clone();
    assert_eq!(mean, vec![sym(0), c(8), c(1)]);
    let inv = outs[2].type_info.as_ref().unwrap().shape.clone();
    assert_eq!(inv, vec![sym(0), c(8), c(1)]);
    let skip_sum = outs[3].type_info.as_ref().unwrap().shape.clone();
    assert_eq!(skip_sum, vec![sym(0), c(8), c(768)]);
}

fn gqa_inputs(past_capacity: i64, total: Option<i64>) -> Vec<NodeIo> {
    let mut inputs = vec![
        f32in(vec![c(1), c(1), c(8)]),
        f32in(vec![c(1), c(1), c(4)]),
        f32in(vec![c(1), c(1), c(4)]),
        f32in(vec![c(1), c(2), c(past_capacity), c(2)]),
        f32in(vec![c(1), c(2), c(past_capacity), c(2)]),
        tin(DataType::Int32, vec![c(1)]),
        tin(DataType::Int32, vec![]),
    ];
    if let Some(total) = total {
        inputs[6] = sd_int_scalar(DataType::Int32, c(total));
    }
    inputs
}

fn gqa_node() -> Node {
    with_attr(
        with_attr(
            with_domain(node("GroupQueryAttention", 7, 3), "com.microsoft"),
            "num_heads",
            Attribute::Int(4),
        ),
        "kv_num_heads",
        Attribute::Int(2),
    )
}

#[test]
fn group_query_attention_missing_past_shape_still_emits_present_shapes() {
    let mut inputs = gqa_inputs(8, Some(3));
    inputs[3] = NodeIo::default();
    let outs = run(&gqa_node(), inputs, 1);
    let present_key = &outs[1]
        .type_info
        .as_ref()
        .expect("present key shape resolved")
        .shape;
    let present_value = &outs[2]
        .type_info
        .as_ref()
        .expect("present value shape resolved")
        .shape;
    assert_eq!(present_key.len(), 4);
    assert_eq!(present_key[0], c(1));
    assert_eq!(present_key[1], c(2));
    assert!(present_key[2].as_symbol().is_some());
    assert_eq!(present_key[3], c(2));
    assert_eq!(present_key, present_value);
}

#[test]
fn group_query_attention_fixed_capacity_present_uses_max_capacity_total() {
    let outs = run(&gqa_node(), gqa_inputs(8, Some(3)), 1);
    assert_eq!(
        outs[1].type_info.as_ref().unwrap().shape,
        vec![c(1), c(2), c(8), c(2)]
    );
    assert_eq!(
        outs[2].type_info.as_ref().unwrap().shape,
        vec![c(1), c(2), c(8), c(2)]
    );
}

#[test]
fn group_query_attention_non_rank_four_past_still_emits_present_shapes() {
    let mut inputs = gqa_inputs(8, Some(3));
    inputs[3] = f32in(vec![c(1), c(8), c(4)]);
    let outs = run(&gqa_node(), inputs, 1);
    let present_key = &outs[1]
        .type_info
        .as_ref()
        .expect("present key shape resolved")
        .shape;
    let present_value = &outs[2]
        .type_info
        .as_ref()
        .expect("present value shape resolved")
        .shape;
    assert_eq!(present_key.len(), 4);
    assert_eq!(present_key[0], c(1));
    assert_eq!(present_key[1], c(2));
    assert!(present_key[2].as_symbol().is_some());
    assert_eq!(present_key[3], c(2));
    assert_eq!(present_key, present_value);
}

#[test]
fn group_query_attention_growing_present_uses_logical_total() {
    let outs = run(&gqa_node(), gqa_inputs(2, Some(3)), 1);
    assert_eq!(
        outs[1].type_info.as_ref().unwrap().shape,
        vec![c(1), c(2), c(3), c(2)]
    );
    assert_eq!(
        outs[2].type_info.as_ref().unwrap().shape,
        vec![c(1), c(2), c(3), c(2)]
    );
}

#[test]
fn group_query_attention_dynamic_total_leaves_present_sequence_symbolic() {
    let outs = run(&gqa_node(), gqa_inputs(8, None), 1);
    let present_key = &outs[1].type_info.as_ref().unwrap().shape;
    let present_value = &outs[2].type_info.as_ref().unwrap().shape;
    assert!(
        present_key[2].as_symbol().is_some(),
        "dynamic max(capacity, total) must remain data-dependent"
    );
    assert_eq!(present_key[2], present_value[2]);
}

#[test]
fn group_query_attention_packed_qkv_splits_output_and_cache_shapes() {
    let mut inputs = gqa_inputs(8, Some(3));
    inputs[0] = f32in(vec![c(1), c(1), c(16)]);
    inputs[1] = NodeIo::default();
    inputs[2] = NodeIo::default();
    let mut packed_node = gqa_node();
    packed_node.inputs[1] = None;
    packed_node.inputs[2] = None;
    let outs = run(&packed_node, inputs, 1);
    assert_eq!(out_shape(&outs), vec![c(1), c(1), c(8)]);
    assert_eq!(
        outs[1].type_info.as_ref().unwrap().shape,
        vec![c(1), c(2), c(8), c(2)]
    );
    assert_eq!(
        outs[2].type_info.as_ref().unwrap().shape,
        vec![c(1), c(2), c(8), c(2)]
    );
}

#[test]
fn moe_and_qmoe_preserve_activation_shape() {
    for op in ["MoE", "QMoE"] {
        let n = with_domain(node(op, 7, 1), "com.microsoft");
        let inputs = vec![
            f32in(vec![sym(0), c(4), c(512)]),
            f32in(vec![c(4), c(8)]),
            tin(DataType::Uint8, vec![c(8), c(1024), c(256)]),
            f32in(vec![c(8), c(1024), c(16)]),
            NodeIo::default(),
            tin(DataType::Uint8, vec![c(8), c(512), c(512)]),
            f32in(vec![c(8), c(512), c(32)]),
        ];
        let outs = run(&n, inputs, 1);
        assert_eq!(out_shape(&outs), vec![sym(0), c(4), c(512)]);
        assert_eq!(out_dtype(&outs), DataType::Float32);
    }
}

#[test]
fn sparse_kv_gather_emits_selected_kv_shape() {
    let n = with_domain(node("SparseKvGather", 2, 1), "pkg.nxrt");
    let outs = run(
        &n,
        vec![
            f32in(vec![sym(0), c(2), c(64), c(128)]),
            tin(DataType::Int32, vec![sym(0), c(2), c(3), c(16)]),
        ],
        1,
    );
    assert_eq!(out_shape(&outs), vec![sym(0), c(2), c(3), c(16), c(128)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn custom_ops_validate_arity_rank_and_compression_contracts() {
    assert!(
        try_run(
            &with_domain(node("MoE", 1, 2), "com.microsoft"),
            vec![f32in(vec![c(2), c(4)])],
            1,
        )
        .is_err()
    );
    assert!(
        try_run(
            &with_domain(node("SparseKvGather", 2, 1), "pkg.nxrt"),
            vec![
                f32in(vec![c(2), c(3), c(4)]),
                tin(DataType::Int32, vec![c(2), c(3), c(4), c(5)]),
            ],
            1,
        )
        .is_err()
    );

    let invalid_ratio = with_attr(
        with_domain(node("CompressedSparseAttention", 10, 3), "pkg.nxrt"),
        "compression_ratio",
        Attribute::Int(8),
    );
    assert!(
        try_run(
            &invalid_ratio,
            vec![f32in(vec![c(1), c(1), c(1), c(64)])],
            1,
        )
        .is_err()
    );

    let mut wrong_ratio128_arity =
        with_domain(node("CompressedSparseAttention", 10, 2), "pkg.nxrt");
    for (name, value) in [
        ("num_heads", 1),
        ("head_dim", 64),
        ("compression_ratio", 128),
    ] {
        wrong_ratio128_arity
            .attributes
            .insert(name.into(), Attribute::Int(value));
    }
    assert!(
        try_run(
            &wrong_ratio128_arity,
            vec![f32in(vec![c(1), c(1), c(1), c(64)])],
            1,
        )
        .is_err()
    );
}

fn csa_ratio4_node(domain: &str) -> Node {
    let mut n = with_domain(node("CompressedSparseAttention", 19, 6), domain);
    for (name, value) in [
        ("num_heads", 8),
        ("head_dim", 512),
        ("qk_rope_head_dim", 64),
        ("compression_ratio", 4),
        ("index_num_heads", 2),
        ("index_head_dim", 128),
        ("index_topk", 512),
    ] {
        n.attributes.insert(name.into(), Attribute::Int(value));
    }
    n.attributes.insert(
        "cache_format".into(),
        Attribute::String("fp8_e4m3_block64".into()),
    );
    n
}

#[test]
fn compressed_sparse_attention_emits_all_ratio4_state_shapes() {
    for domain in ["pkg.nxrt", "com.microsoft"] {
        let mut inputs = vec![NodeIo::default(); 19];
        inputs[0] = f32in(vec![sym(0), c(5), c(8), c(512)]);
        inputs[9] = sd_int_scalar(DataType::Int64, c(12));
        let outs = run(&csa_ratio4_node(domain), inputs, 1);
        let expected = [
            (DataType::Float32, vec![sym(0), c(5), c(8), c(512)]),
            (DataType::Uint8, vec![sym(0), c(3), c(583)]),
            (DataType::Float32, vec![sym(0), c(8), c(2), c(1024)]),
            (DataType::Uint8, vec![sym(0), c(3), c(68)]),
            (DataType::Float32, vec![sym(0), c(8), c(2), c(256)]),
            (DataType::Int32, vec![sym(0), c(2), c(5), c(3)]),
        ];
        for (output, (dtype, shape)) in outs.iter().zip(expected.iter()) {
            let info = output.type_info.as_ref().expect("CSA output resolved");
            assert_eq!(info.dtype, *dtype);
            assert_eq!(info.shape, *shape);
        }
    }
}

#[test]
fn compressed_sparse_attention_dynamic_total_resolves_every_output() {
    let mut inputs = vec![NodeIo::default(); 19];
    inputs[0] = f32in(vec![c(2), sym(0), c(8), c(512)]);
    inputs[9] = tin(DataType::Int64, vec![]);
    let outs = run(&csa_ratio4_node("pkg.nxrt"), inputs, 1);
    assert!(outs.iter().all(|output| output.type_info.is_some()));
    let cache_records = outs[1].type_info.as_ref().unwrap().shape[1].clone();
    let index_records = outs[3].type_info.as_ref().unwrap().shape[1].clone();
    assert!(cache_records.as_symbol().is_some());
    assert_eq!(cache_records, index_records);
    assert!(
        outs[5].type_info.as_ref().unwrap().shape[3]
            .as_symbol()
            .is_some()
    );
}

#[test]
fn compressed_sparse_attention_ratio128_emits_three_outputs() {
    let mut n = with_domain(node("CompressedSparseAttention", 11, 3), "pkg.nxrt");
    for (name, value) in [
        ("num_heads", 8),
        ("head_dim", 512),
        ("qk_rope_head_dim", 64),
        ("compression_ratio", 128),
    ] {
        n.attributes.insert(name.into(), Attribute::Int(value));
    }
    n.attributes.insert(
        "cache_format".into(),
        Attribute::String("fp8_e4m3_block64".into()),
    );
    let mut inputs = vec![NodeIo::default(); 11];
    inputs[0] = f32in(vec![c(2), c(1), c(8), c(512)]);
    inputs[9] = sd_int_scalar(DataType::Int64, c(256));
    let outs = run(&n, inputs, 1);
    assert_eq!(
        outs[0].type_info.as_ref().unwrap().shape,
        vec![c(2), c(1), c(8), c(512)]
    );
    assert_eq!(
        outs[1].type_info.as_ref().unwrap().shape,
        vec![c(2), c(2), c(583)]
    );
    assert_eq!(
        outs[2].type_info.as_ref().unwrap().shape,
        vec![c(2), c(128), c(2), c(512)]
    );
}

#[test]
fn standard_simplified_layer_norm_passthrough() {
    let n = node("SimplifiedLayerNormalization", 2, 1);
    let outs = run(
        &n,
        vec![f32in(vec![sym(0), c(8), c(768)]), f32in(vec![c(768)])],
        21,
    );
    assert_eq!(out_shape(&outs), vec![sym(0), c(8), c(768)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn rms_norm_passthrough() {
    // Single output equal to X (opset 23).
    let n = node("RMSNormalization", 2, 1);
    let outs = run(
        &n,
        vec![f32in(vec![sym(0), c(8), c(768)]), f32in(vec![c(768)])],
        23,
    );
    assert_eq!(out_shape(&outs), vec![sym(0), c(8), c(768)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn batch_norm_inference_passthrough_opsets_9_14_15() {
    let n = node("BatchNormalization", 5, 1);
    for opset in [9, 14, 15] {
        let outs = run(
            &n,
            vec![
                tin(DataType::Float16, vec![c(2), c(3), c(4), c(5)]),
                tin(DataType::Float16, vec![c(3)]),
                tin(DataType::Float16, vec![c(3)]),
                tin(DataType::Float16, vec![c(3)]),
                tin(DataType::Float16, vec![c(3)]),
            ],
            opset,
        );
        assert_eq!(out_shape(&outs), vec![c(2), c(3), c(4), c(5)]);
        assert_eq!(out_dtype(&outs), DataType::Float16);
    }
}

#[test]
fn instance_norm_passthrough_opset_6() {
    let n = node("InstanceNormalization", 3, 1);
    let outs = run(
        &n,
        vec![
            tin(DataType::Float16, vec![c(1), c(8), c(16), c(16)]),
            tin(DataType::Float16, vec![c(8)]),
            tin(DataType::Float16, vec![c(8)]),
        ],
        6,
    );
    assert_eq!(out_shape(&outs), vec![c(1), c(8), c(16), c(16)]);
    assert_eq!(out_dtype(&outs), DataType::Float16);
}

#[test]
fn normalization_unknown_x_leaves_output_unresolved() {
    for (op, n_in, opset) in [
        ("BatchNormalization", 5, 15),
        ("InstanceNormalization", 3, 6),
    ] {
        let n = node(op, n_in, 1);
        let mut inputs = vec![NodeIo::default()];
        inputs.extend((1..n_in).map(|_| f32in(vec![c(3)])));
        let outs = run(&n, inputs, opset);
        assert!(outs[0].type_info.is_none());
    }
}

#[test]
fn rotary_embedding_passthrough_4d() {
    // Output equals input X (opset 23), 4D [batch, heads, seq, head_size].
    let n = node("RotaryEmbedding", 3, 1);
    let outs = run(
        &n,
        vec![
            f32in(vec![sym(0), c(12), c(16), c(64)]),
            f32in(vec![c(16), c(32)]),
            f32in(vec![c(16), c(32)]),
        ],
        23,
    );
    assert_eq!(out_shape(&outs), vec![sym(0), c(12), c(16), c(64)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn swish_passthrough() {
    // Elementwise, same shape/dtype (opset 24).
    let n = with_attr(node("Swish", 1, 1), "alpha", Attribute::Float(1.0));
    let outs = run(&n, vec![f32in(vec![sym(0), c(8), c(768)])], 24);
    assert_eq!(out_shape(&outs), vec![sym(0), c(8), c(768)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn std_gelu_passthrough() {
    // Standard ai.onnx::Gelu (opset 20), same shape/dtype.
    let n = with_attr(
        node("Gelu", 1, 1),
        "approximate",
        Attribute::String(b"tanh".to_vec()),
    );
    let outs = run(&n, vec![f32in(vec![sym(0), c(8), c(768)])], 20);
    assert_eq!(out_shape(&outs), vec![sym(0), c(8), c(768)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn std_gelu_is_unregistered_before_opset_20() {
    let registry = InferenceRegistry::default_registry();
    assert!(registry.get("", "Gelu", 19).is_none());
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

#[test]
fn data_ops_propagate_shape_data_and_reject_invalid_metadata() {
    let shape = with_attr(
        with_attr(node("Shape", 1, 1), "start", Attribute::Int(-2)),
        "end",
        Attribute::Int(99),
    );
    let shape_out = run(&shape, vec![f32in(vec![sym(0), c(3), c(4)])], 15);
    assert_eq!(out_shape(&shape_out), vec![c(2)]);
    assert_eq!(
        shape_out[0].shape_data.as_ref().unwrap().elems,
        vec![c(3), c(4)]
    );

    let size_out = run(&node("Size", 1, 1), vec![f32in(vec![c(2), c(3)])], 13);
    assert_eq!(size_out[0].shape_data.as_ref().unwrap().elems, vec![c(6)]);

    let constant = with_attr(
        node("Constant", 0, 1),
        "value_ints",
        Attribute::Ints(vec![2, 5]),
    );
    let constant_out = run(&constant, vec![], 13);
    assert_eq!(out_shape(&constant_out), vec![c(2)]);
    assert_eq!(
        constant_out[0].shape_data.as_ref().unwrap().elems,
        vec![c(2), c(5)]
    );

    let identity = run(&node("Identity", 1, 1), vec![sd_vec(vec![c(7)])], 13);
    assert_eq!(identity[0].shape_data.as_ref().unwrap().elems, vec![c(7)]);

    let cast_like = run(
        &node("CastLike", 2, 1),
        vec![f32in(vec![c(2), c(3)]), tin(DataType::Uint8, vec![c(1)])],
        19,
    );
    assert_eq!(out_dtype(&cast_like), DataType::Uint8);
    assert_eq!(out_shape(&cast_like), vec![c(2), c(3)]);

    assert!(try_run(&node("Cast", 1, 1), vec![f32in(vec![c(1)])], 13).is_err());
    assert!(
        try_run(
            &with_attr(node("Cast", 1, 1), "to", Attribute::Int(-1)),
            vec![f32in(vec![c(1)])],
            13,
        )
        .is_err()
    );
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
fn constant_of_shape_dynamic_input_is_unresolved_but_empty_shape_is_scalar() {
    let n = node("ConstantOfShape", 1, 1);
    let dynamic = run(&n, vec![tin(DataType::Int64, vec![c(3)])], 25);
    assert!(dynamic[0].type_info.is_none());

    let scalar = run(&n, vec![sd_vec(vec![])], 25);
    assert_eq!(out_shape(&scalar), Vec::<DimExpr>::new());
    assert_eq!(out_dtype(&scalar), DataType::Float32);
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

#[test]
fn expand_adds_leading_target_dimensions() {
    let n = node("Expand", 2, 1);
    let outs = run(
        &n,
        vec![f32in(vec![c(3), c(1)]), sd_vec(vec![c(2), c(1), c(6)])],
        8,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(3), c(6)]);
}

#[test]
fn expand_target_one_keeps_input_dimension() {
    let n = node("Expand", 2, 1);
    let outs = run(
        &n,
        vec![f32in(vec![c(3), c(4)]), sd_vec(vec![c(3), c(1)])],
        13,
    );
    assert_eq!(out_shape(&outs), vec![c(3), c(4)]);
}

#[test]
fn expand_preserves_input_dtype() {
    let n = node("Expand", 2, 1);
    let outs = run(
        &n,
        vec![
            tin(DataType::Float16, vec![c(1), c(4)]),
            sd_vec(vec![c(3), c(4)]),
        ],
        13,
    );
    assert_eq!(out_dtype(&outs), DataType::Float16);
}

#[test]
fn expand_unknown_shape_tensor_leaves_output_unresolved() {
    let n = node("Expand", 2, 1);
    let outs = run(
        &n,
        vec![f32in(vec![c(3), c(1)]), tin(DataType::Int64, vec![c(3)])],
        13,
    );
    assert!(outs[0].type_info.is_none());
}

#[test]
fn expand_rejects_incompatible_concrete_dimensions() {
    let n = node("Expand", 2, 1);
    assert!(
        try_run(
            &n,
            vec![f32in(vec![c(2), c(3)]), sd_vec(vec![c(2), c(4)])],
            13,
        )
        .is_err()
    );
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

#[test]
fn split_dynamic_sizes_leave_split_axis_unknown() {
    let n = with_attr(node("Split", 2, 2), "axis", Attribute::Int(1));
    let outs = run(
        &n,
        vec![f32in(vec![c(2), c(6)]), tin(DataType::Int64, vec![c(2)])],
        13,
    );
    for output in outs {
        let shape = output.type_info.unwrap().shape;
        assert_eq!(shape[0], c(2));
        assert!(shape[1].as_symbol().is_some());
    }
}

#[test]
fn split_num_outputs_uses_ceil_chunks_and_final_remainder() {
    let n = with_attr(
        with_attr(node("Split", 1, 3), "axis", Attribute::Int(1)),
        "num_outputs",
        Attribute::Int(3),
    );
    let outs = run(&n, vec![f32in(vec![c(2), c(7)])], 18);
    assert_eq!(out_shape(&outs), vec![c(2), c(3)]);
    assert_eq!(outs[1].type_info.as_ref().unwrap().shape, vec![c(2), c(3)]);
    assert_eq!(outs[2].type_info.as_ref().unwrap().shape, vec![c(2), c(1)]);
}

#[test]
fn split_num_outputs_zero_size_final_chunk() {
    let n = with_attr(
        with_attr(node("Split", 1, 3), "axis", Attribute::Int(1)),
        "num_outputs",
        Attribute::Int(3),
    );
    let outs = run(&n, vec![f32in(vec![c(2), c(2)])], 18);
    assert_eq!(out_shape(&outs), vec![c(2), c(1)]);
    assert_eq!(outs[1].type_info.as_ref().unwrap().shape, vec![c(2), c(1)]);
    assert_eq!(outs[2].type_info.as_ref().unwrap().shape, vec![c(2), c(0)]);
}

#[test]
fn split_input_sizes_are_uneven_and_must_match_outputs_and_axis_extent() {
    let n = with_attr(node("Split", 2, 2), "axis", Attribute::Int(1));
    let outs = run(
        &n,
        vec![f32in(vec![c(3), c(7)]), sd_vec(vec![c(2), c(5)])],
        18,
    );
    assert_eq!(out_shape(&outs), vec![c(3), c(2)]);
    assert_eq!(outs[1].type_info.as_ref().unwrap().shape, vec![c(3), c(5)]);

    assert_invalid(
        try_run(
            &n,
            vec![f32in(vec![c(3), c(7)]), sd_vec(vec![c(2), c(4)])],
            18,
        )
        .unwrap_err(),
        "Split",
        "split sizes sum to 6, but axis extent is 7",
    );

    let both = with_attr(n.clone(), "num_outputs", Attribute::Int(2));
    assert_invalid(
        try_run(
            &both,
            vec![f32in(vec![c(3), c(7)]), sd_vec(vec![c(2), c(5)])],
            18,
        )
        .unwrap_err(),
        "Split",
        "split input and num_outputs cannot both be specified",
    );
}

#[test]
fn split_rejects_non_positive_num_outputs() {
    for num_outputs in [0, -1] {
        let n = with_attr(
            node("Split", 1, 2),
            "num_outputs",
            Attribute::Int(num_outputs),
        );
        let error = try_run(&n, vec![f32in(vec![c(2), c(8)])], 18).unwrap_err();
        assert_invalid(
            error,
            "Split",
            &format!("num_outputs must be positive, got {num_outputs}"),
        );
    }
}

#[test]
fn comparison_ops_broadcast_to_bool() {
    for op in ["Less", "LessOrEqual", "Greater", "GreaterOrEqual", "Equal"] {
        let outs = run(
            &node(op, 2, 1),
            vec![
                tin(DataType::Float32, vec![c(2), c(1), c(4)]),
                tin(DataType::Float32, vec![c(1), c(3), c(1)]),
            ],
            19,
        );
        assert_eq!(out_shape(&outs), vec![c(2), c(3), c(4)], "{op}");
        assert_eq!(out_dtype(&outs), DataType::Bool, "{op}");
    }
}

#[test]
fn logical_ops_broadcast_to_bool() {
    for op in ["And", "Or", "Xor"] {
        let outs = run(
            &node(op, 2, 1),
            vec![
                tin(DataType::Bool, vec![c(2), c(1)]),
                tin(DataType::Bool, vec![c(1), c(3)]),
            ],
            19,
        );
        assert_eq!(out_shape(&outs), vec![c(2), c(3)], "{op}");
        assert_eq!(out_dtype(&outs), DataType::Bool, "{op}");
    }
}

#[test]
fn elementwise_shape_data_handles_vector_scalar_and_exact_division() {
    let add = run(
        &node("Add", 2, 1),
        vec![
            sd_vec(vec![c(2), c(5)]),
            sd_int_scalar(DataType::Int64, c(3)),
        ],
        13,
    );
    assert_eq!(add[0].shape_data.as_ref().unwrap().elems, vec![c(5), c(8)]);

    let div = run(
        &node("Div", 2, 1),
        vec![
            sd_int_scalar(DataType::Int64, c(12)),
            sd_vec(vec![c(2), c(3)]),
        ],
        13,
    );
    assert_eq!(div[0].shape_data.as_ref().unwrap().elems, vec![c(6), c(4)]);

    let maximum = run(
        &node("Max", 2, 1),
        vec![sd_vec(vec![c(2), c(9)]), sd_vec(vec![c(7), c(3)])],
        13,
    );
    assert_eq!(
        maximum[0].shape_data.as_ref().unwrap().elems,
        vec![c(7), c(9)]
    );
}

#[test]
fn not_preserves_shape_and_outputs_bool() {
    let outs = run(
        &node("Not", 1, 1),
        vec![tin(DataType::Bool, vec![c(2), c(3)])],
        19,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(3)]);
    assert_eq!(out_dtype(&outs), DataType::Bool);
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
fn pooling_dilation_ceil_mode_and_indices_shape() {
    let mut n = with_attr(
        node("MaxPool", 1, 2),
        "kernel_shape",
        Attribute::Ints(vec![3]),
    );
    n = with_attr(n, "strides", Attribute::Ints(vec![2]));
    n = with_attr(n, "pads", Attribute::Ints(vec![1, 1]));
    n = with_attr(n, "dilations", Attribute::Ints(vec![2]));
    n = with_attr(n, "ceil_mode", Attribute::Int(1));
    let outs = run(&n, vec![f32in(vec![c(1), c(4), c(10)])], 22);
    assert_eq!(out_shape(&outs), vec![c(1), c(4), c(5)]);
    assert_eq!(outs[1].type_info.as_ref().unwrap().shape, out_shape(&outs));
    assert_eq!(outs[1].type_info.as_ref().unwrap().dtype, DataType::Int64);
}

#[test]
fn pooling_auto_pad_same_and_valid() {
    let same = with_attr(
        with_attr(
            with_attr(
                node("AveragePool", 1, 1),
                "kernel_shape",
                Attribute::Ints(vec![3, 3]),
            ),
            "strides",
            Attribute::Ints(vec![2, 2]),
        ),
        "auto_pad",
        Attribute::String("SAME_LOWER".into()),
    );
    assert_eq!(
        out_shape(&run(&same, vec![f32in(vec![c(1), c(3), c(5), c(6)])], 22,)),
        vec![c(1), c(3), c(3), c(3)]
    );

    let valid = with_attr(
        with_attr(
            with_attr(
                node("MaxPool", 1, 1),
                "kernel_shape",
                Attribute::Ints(vec![3]),
            ),
            "strides",
            Attribute::Ints(vec![2]),
        ),
        "auto_pad",
        Attribute::String("VALID".into()),
    );
    assert_eq!(
        out_shape(&run(&valid, vec![f32in(vec![c(1), c(3), c(8)])], 22)),
        vec![c(1), c(3), c(3)]
    );
}

#[test]
fn pooling_validates_kernel_and_rank_and_preserves_dynamic_spatial_rank() {
    assert!(
        try_run(
            &node("AveragePool", 1, 1),
            vec![f32in(vec![c(1), c(2), c(8)])],
            22,
        )
        .is_err()
    );
    assert!(
        try_run(
            &with_attr(
                node("MaxPool", 1, 1),
                "kernel_shape",
                Attribute::Ints(vec![2, 2]),
            ),
            vec![f32in(vec![c(1), c(2), c(8)])],
            22,
        )
        .is_err()
    );
    assert!(
        try_run(
            &with_attr(
                node("MaxPool", 1, 1),
                "kernel_shape",
                Attribute::Ints(vec![2]),
            ),
            vec![f32in(vec![c(1)])],
            22,
        )
        .is_err()
    );

    let n = with_attr(
        node("AveragePool", 1, 1),
        "kernel_shape",
        Attribute::Ints(vec![3]),
    );
    let outs = run(&n, vec![f32in(vec![c(1), c(2), sym(0)])], 22);
    assert_eq!(out_shape(&outs).len(), 3);
    assert!(out_shape(&outs)[2].as_symbol().is_some());
}

#[test]
fn global_pool_sets_every_spatial_dimension_to_one() {
    for op in ["GlobalAveragePool", "GlobalMaxPool"] {
        let outs = run(
            &node(op, 1, 1),
            vec![f32in(vec![sym(0), c(8), c(7), sym(1)])],
            22,
        );
        assert_eq!(out_shape(&outs), vec![sym(0), c(8), c(1), c(1)]);
    }
}

#[test]
fn pooling_rejects_guaranteed_symbolic_overflow() {
    let n = with_attr(
        with_attr(
            node("MaxPool", 1, 1),
            "kernel_shape",
            Attribute::Ints(vec![1]),
        ),
        "pads",
        Attribute::Ints(vec![isize::MAX as i64, 1]),
    );
    let error = try_run(&n, vec![f32in(vec![c(1), c(1), sym(0)])], 22).unwrap_err();
    assert!(error.to_string().contains("exceeds isize::MAX"));
}

#[test]
fn pooling_rejects_cancellation_masked_symbolic_overflow() {
    let n = with_attr(
        with_attr(
            with_attr(
                node("MaxPool", 1, 1),
                "kernel_shape",
                Attribute::Ints(vec![isize::MAX as i64]),
            ),
            "dilations",
            Attribute::Ints(vec![2]),
        ),
        "pads",
        Attribute::Ints(vec![isize::MAX as i64, isize::MAX as i64]),
    );
    let error = try_run(&n, vec![f32in(vec![c(1), c(1), sym(0)])], 22).unwrap_err();
    assert!(error.to_string().contains("exceeds isize::MAX"));
}

// --- Resize ---------------------------------------------------------------

#[test]
fn resize_infers_constant_scales_and_sizes() {
    let mut scales_node = node("Resize", 4, 1);
    scales_node.inputs[1] = None;
    scales_node.inputs[3] = None;
    let scales = run(
        &scales_node,
        vec![
            f32in(vec![c(2), c(3), c(4)]),
            NodeIo::default(),
            sd_float_vec(vec![1.0, 1.0, 1.5]),
            NodeIo::default(),
        ],
        19,
    );
    assert_eq!(out_shape(&scales), vec![c(2), c(3), c(6)]);

    let mut sizes_node = with_attr(node("Resize", 4, 1), "axes", Attribute::Ints(vec![-2, -1]));
    sizes_node.inputs[1] = None;
    sizes_node.inputs[2] = None;
    let sizes = run(
        &sizes_node,
        vec![
            f32in(vec![c(2), c(3), c(4), c(5)]),
            NodeIo::default(),
            NodeIo::default(),
            sd_vec(vec![c(8), c(9)]),
        ],
        19,
    );
    assert_eq!(out_shape(&sizes), vec![c(2), c(3), c(8), c(9)]);
}

#[test]
fn resize_dynamic_scales_and_sizes_leave_extents_unresolved() {
    for use_scales in [true, false] {
        let mut n = node("Resize", 4, 1);
        n.inputs[1] = None;
        if use_scales {
            n.inputs[3] = None;
        } else {
            n.inputs[2] = None;
        }
        let vector = if use_scales {
            tin(DataType::Float32, vec![c(3)])
        } else {
            tin(DataType::Int64, vec![c(3)])
        };
        let inputs = if use_scales {
            vec![
                f32in(vec![c(2), c(3), c(4)]),
                NodeIo::default(),
                vector,
                NodeIo::default(),
            ]
        } else {
            vec![
                f32in(vec![c(2), c(3), c(4)]),
                NodeIo::default(),
                NodeIo::default(),
                vector,
            ]
        };
        let shape = out_shape(&run(&n, inputs, 19));
        assert_eq!(shape.len(), 3);
        assert!(
            shape
                .iter()
                .all(|dimension| dimension.as_symbol().is_some())
        );
    }
}

#[test]
fn resize_accepts_ignored_roi_and_absent_extent_inputs() {
    for (roi, has_roi) in [
        (tin(DataType::Float32, vec![c(1), c(4)]), true),
        (NodeIo::default(), false),
    ] {
        let mut resize = with_attr(
            node("Resize", 4, 1),
            "coordinate_transformation_mode",
            Attribute::String("asymmetric".into()),
        );
        if !has_roi {
            resize.inputs[1] = None;
        }
        resize.inputs[3] = None;
        let outputs = run(
            &resize,
            vec![
                f32in(vec![c(2), c(3)]),
                roi,
                sd_float_vec(vec![1.0, 2.0]),
                NodeIo::default(),
            ],
            19,
        );
        assert_eq!(out_shape(&outputs), vec![c(2), c(6)]);
    }

    let mut neither = node("Resize", 4, 1);
    neither.inputs[1] = None;
    neither.inputs[2] = None;
    neither.inputs[3] = None;
    let shape = out_shape(&run(
        &neither,
        vec![
            f32in(vec![c(2), c(3)]),
            NodeIo::default(),
            NodeIo::default(),
            NodeIo::default(),
        ],
        19,
    ));
    assert_eq!(shape.len(), 2);
    assert!(
        shape
            .iter()
            .all(|dimension| dimension.as_symbol().is_some())
    );
}

#[test]
fn resize_rejects_both_scales_and_sizes() {
    let mut both = node("Resize", 4, 1);
    both.inputs[1] = None;
    assert!(
        try_run(
            &both,
            vec![
                f32in(vec![c(2), c(3)]),
                NodeIo::default(),
                sd_float_vec(vec![1.0, 2.0]),
                sd_vec(vec![c(2), c(6)]),
            ],
            19,
        )
        .is_err()
    );
}

#[test]
fn resize_accepts_maximum_extent_with_unit_scale() {
    let mut resize = node("Resize", 4, 1);
    resize.inputs[1] = None;
    resize.inputs[3] = None;
    let outputs = run(
        &resize,
        vec![
            f32in(vec![c(isize::MAX as i64)]),
            NodeIo::default(),
            sd_float_vec(vec![1.0]),
            NodeIo::default(),
        ],
        19,
    );
    assert_eq!(out_shape(&outputs), vec![c(isize::MAX as i64)]);

    let mut aspect_resize = with_attr(
        node("Resize", 4, 1),
        "keep_aspect_ratio_policy",
        Attribute::String("not_larger".into()),
    );
    aspect_resize.inputs[1] = None;
    aspect_resize.inputs[2] = None;
    let outputs = run(
        &aspect_resize,
        vec![
            f32in(vec![c(isize::MAX as i64)]),
            NodeIo::default(),
            NodeIo::default(),
            sd_vec(vec![c(isize::MAX as i64)]),
        ],
        19,
    );
    assert_eq!(out_shape(&outputs), vec![c(isize::MAX as i64)]);

    let error = try_run(
        &resize,
        vec![
            f32in(vec![c(isize::MAX as i64)]),
            NodeIo::default(),
            sd_float_vec(vec![2.0]),
            NodeIo::default(),
        ],
        19,
    )
    .unwrap_err();
    assert!(error.to_string().contains("exceeds isize::MAX"));
}

// --- Linear quantization --------------------------------------------------

#[test]
fn quantize_and_dequantize_preserve_shape_and_infer_dtype() {
    let mut quantize = with_attr(node("QuantizeLinear", 3, 1), "axis", Attribute::Int(-1));
    quantize.inputs[2] = None;
    let quantized = run(
        &quantize,
        vec![
            f32in(vec![c(2), c(3)]),
            f32in(vec![c(3)]),
            NodeIo::default(),
        ],
        21,
    );
    assert_eq!(out_shape(&quantized), vec![c(2), c(3)]);
    assert_eq!(out_dtype(&quantized), DataType::Uint8);

    let dequantized = run(
        &node("DequantizeLinear", 3, 1),
        vec![
            tin(DataType::Int4, vec![c(2), c(8)]),
            tin(DataType::Float16, Vec::new()),
            tin(DataType::Int4, Vec::new()),
        ],
        21,
    );
    assert_eq!(out_shape(&dequantized), vec![c(2), c(8)]);
    assert_eq!(out_dtype(&dequantized), DataType::Float16);
}

#[test]
fn quantize_uses_zero_point_dtype_and_validates_blocking() {
    let quantized = run(
        &node("QuantizeLinear", 3, 1),
        vec![
            f32in(vec![c(2), c(3)]),
            f32in(Vec::new()),
            tin(DataType::Int4, Vec::new()),
        ],
        21,
    );
    assert_eq!(out_dtype(&quantized), DataType::Int4);

    let blocked = with_attr(
        with_attr(node("DequantizeLinear", 3, 1), "axis", Attribute::Int(-1)),
        "block_size",
        Attribute::Int(4),
    );
    let outs = run(
        &blocked,
        vec![
            tin(DataType::Uint4, vec![c(2), c(8)]),
            f32in(vec![c(2), c(2)]),
            tin(DataType::Uint4, vec![c(2), c(2)]),
        ],
        21,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(8)]);

    let rank_one_blocked = with_attr(
        with_attr(node("DequantizeLinear", 3, 1), "axis", Attribute::Int(0)),
        "block_size",
        Attribute::Int(4),
    );
    let outs = run(
        &rank_one_blocked,
        vec![
            tin(DataType::Uint4, vec![c(8)]),
            f32in(vec![c(2)]),
            tin(DataType::Uint4, vec![c(2)]),
        ],
        21,
    );
    assert_eq!(out_shape(&outs), vec![c(8)]);

    let rank_one_quantize = with_attr(
        with_attr(node("QuantizeLinear", 3, 1), "axis", Attribute::Int(0)),
        "block_size",
        Attribute::Int(4),
    );
    let outs = run(
        &rank_one_quantize,
        vec![
            f32in(vec![c(8)]),
            f32in(vec![c(2)]),
            tin(DataType::Uint4, vec![c(2)]),
        ],
        21,
    );
    assert_eq!(out_shape(&outs), vec![c(8)]);
}

#[test]
fn rank_one_blocked_quantization_validates_axis_range() {
    let inputs = |op: &str| {
        let data = if op == "QuantizeLinear" {
            f32in(vec![c(8)])
        } else {
            tin(DataType::Uint4, vec![c(8)])
        };
        vec![data, f32in(vec![c(2)]), tin(DataType::Uint4, vec![c(2)])]
    };

    for op in ["QuantizeLinear", "DequantizeLinear"] {
        for axis in [0, -1] {
            let blocked = with_attr(
                with_attr(node(op, 3, 1), "axis", Attribute::Int(axis)),
                "block_size",
                Attribute::Int(4),
            );
            let outputs = run(&blocked, inputs(op), 21);
            assert_eq!(out_shape(&outputs), vec![c(8)]);
        }
    }

    for (op, axis) in [
        ("DequantizeLinear", 1),
        ("DequantizeLinear", -2),
        ("QuantizeLinear", 1),
    ] {
        let blocked = with_attr(
            with_attr(node(op, 3, 1), "axis", Attribute::Int(axis)),
            "block_size",
            Attribute::Int(4),
        );
        let error = try_run(&blocked, inputs(op), 21).unwrap_err();
        assert!(
            error
                .to_string()
                .contains(&format!("axis {axis} is out of range for rank 1")),
            "{error}"
        );
    }
}

#[test]
fn quantization_rejects_invalid_axis_and_block_shape() {
    let bad_axis = with_attr(node("QuantizeLinear", 3, 1), "axis", Attribute::Int(-3));
    assert!(
        try_run(
            &bad_axis,
            vec![
                f32in(vec![c(2), c(3)]),
                f32in(vec![c(3)]),
                tin(DataType::Uint8, vec![c(3)]),
            ],
            21,
        )
        .is_err()
    );

    let bad_block = with_attr(
        with_attr(node("DequantizeLinear", 3, 1), "axis", Attribute::Int(1)),
        "block_size",
        Attribute::Int(4),
    );
    assert!(
        try_run(
            &bad_block,
            vec![
                tin(DataType::Uint4, vec![c(2), c(8)]),
                f32in(vec![c(2), c(3)]),
                tin(DataType::Uint4, vec![c(2), c(3)]),
            ],
            21,
        )
        .is_err()
    );
}

#[test]
fn dynamic_quantize_linear_outputs_tensor_and_scalars() {
    let outs = run(
        &node("DynamicQuantizeLinear", 1, 3),
        vec![f32in(vec![sym(0), c(7)])],
        11,
    );
    assert_eq!(
        outs[0].type_info.as_ref().unwrap().shape,
        vec![sym(0), c(7)]
    );
    assert_eq!(outs[0].type_info.as_ref().unwrap().dtype, DataType::Uint8);
    assert!(outs[1].type_info.as_ref().unwrap().shape.is_empty());
    assert_eq!(outs[1].type_info.as_ref().unwrap().dtype, DataType::Float32);
    assert!(outs[2].type_info.as_ref().unwrap().shape.is_empty());
    assert_eq!(outs[2].type_info.as_ref().unwrap().dtype, DataType::Uint8);
}

#[test]
fn pad_grows_dims() {
    let n = node("Pad", 2, 1);
    // pads = [0,0,1,1, 0,0,1,1] over rank 4 -> H,W grow by 2
    let pads = sd_vec(vec![c(0), c(0), c(1), c(1), c(0), c(0), c(1), c(1)]);
    let outs = run(&n, vec![f32in(vec![sym(0), c(3), c(32), c(32)]), pads], 13);
    assert_eq!(out_shape(&outs), vec![sym(0), c(3), c(34), c(34)]);
}

#[test]
fn pad_expanded_attention_axes_shape_and_bytes() {
    let n = node("Pad", 4, 1);
    // Expanded Attention pads a [2,3,4,4] bias on axis -1 by [0,2].
    let outs = run(
        &n,
        vec![
            f32in(vec![c(2), c(3), c(4), c(4)]),
            sd_vec(vec![c(0), c(2)]),
            sd_float_scalar(DataType::Float32, f64::NEG_INFINITY),
            sd_vec(vec![c(-1)]),
        ],
        23,
    );
    let shape = out_shape(&outs);
    assert_eq!(shape, vec![c(2), c(3), c(4), c(6)]);
    let elements: i64 = shape.iter().map(|dim| dim.as_const().unwrap()).product();
    assert_eq!(elements, 144);
    assert_eq!(elements * DataType::Float32.byte_size() as i64, 576);
}

#[test]
fn pad_dynamic_pads_only_clear_selected_axes() {
    let mut n = node("Pad", 4, 1);
    n.inputs[2] = None;
    let outs = run(
        &n,
        vec![
            f32in(vec![c(2), c(3), c(4)]),
            tin(DataType::Int64, vec![c(2)]),
            NodeIo::default(),
            sd_vec(vec![c(-1)]),
        ],
        25,
    );
    let shape = out_shape(&outs);
    assert_eq!(&shape[..2], &[c(2), c(3)]);
    assert!(shape[2].as_symbol().is_some());
}

#[test]
fn pad_rejects_extent_beyond_isize_max() {
    let n = node("Pad", 2, 1);
    assert!(
        try_run(
            &n,
            vec![f32in(vec![c(1)]), sd_vec(vec![c(isize::MAX as i64), c(0)]),],
            25,
        )
        .is_err()
    );
}

#[test]
fn pad_symbolic_extent_rejects_guaranteed_overflow_and_stays_symbolic_normally() {
    let n = node("Pad", 2, 1);
    let error = try_run(
        &n,
        vec![
            f32in(vec![sym(0)]),
            sd_vec(vec![c(isize::MAX as i64), c(1)]),
        ],
        25,
    )
    .unwrap_err();
    assert!(error.to_string().contains("exceeds isize::MAX"));

    let outs = run(&n, vec![f32in(vec![sym(0)]), sd_vec(vec![c(1), c(1)])], 25);
    let shape = out_shape(&outs);
    let extent = &shape[0];
    assert!(extent.as_const().is_none());
    assert!(extent.as_symbol().is_some());
}

// --- unregistered op is permissive ---------------------------------------

#[test]
fn unregistered_op_leaves_output_unresolved() {
    let n = node("SomeExoticOp", 1, 1);
    let outs = run(&n, vec![f32in(vec![c(2), c(3)])], 13);
    assert!(outs[0].type_info.is_none());
}

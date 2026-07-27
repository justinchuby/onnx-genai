//! Data-driven CUDA EP **conformance profile**.
//!
//! This suite treats the CPU execution provider as the reference oracle and
//! validates that every op the CUDA EP *claims* to support (the
//! [`CUDA_COVERED_OPS`] table) is actually covered by a parity test, in a
//! declarative, maintainable way rather than ad-hoc per-op tests.
//!
//! A **conformance profile** is a table of [`ProfileEntry`] — one per covered op
//! — where each op is either:
//!
//! * [`Coverage::Sweep`] — inline `(op, dtype, shapes, attrs)` parity cases that
//!   the generic sweep here runs against the CPU EP; or
//! * [`Coverage::Dedicated`] — covered by a named dedicated GPU parity suite
//!   (another `tests/*.rs` file that runs the op), recorded so the claim is
//!   auditable.
//!
//! The three highest-value tests need **no GPU** and run everywhere (incl. CI):
//!
//! * [`every_covered_op_has_a_conformance_entry`] — the *coverage-of-coverage*
//!   check: it fails if any [`CUDA_COVERED_OPS`] op is missing a profile entry
//!   (i.e. "claimed but untested" — the exact class of miss that let
//!   `ReduceLogSumExp` and bf16 coverage gaps slip through), and if any profile
//!   entry references an op no longer covered.
//! * [`dedicated_suites_exist_and_name_their_op`] — verifies each `Dedicated`
//!   suite file exists and actually names its op, so deleting/renaming a suite
//!   cannot silently leave an op unverified.
//! * [`profile_has_no_duplicate_entries`].
//!
//! The GPU sweep ([`conformance_sweep_matches_cpu`]) graceful-skips without a
//! CUDA device. Run it on a GPU box with:
//!
//! ```bash
//! CUDA_VISIBLE_DEVICES=0 cargo test -p onnx-runtime-ep-cuda --features cuda \
//!     --test cuda_conformance_gpu
//! ```
//!
//! See `docs/CUDA_COVERAGE.md` ("Conformance profile & GPU parity sweep").

mod common;

use std::collections::HashSet;
use std::path::Path;

use common::{Tensor, assert_close, cuda_ep, decode_floats, float_input, input, run_cpu, run_cuda};
use onnx_runtime_ep_cuda::{CUDA_COVERED_OPS, CudaExecutionProvider};
use onnx_runtime_ir::{Attribute, DataType};

// ─────────────────────────────────────────────────────────────────────────────
// Profile model
// ─────────────────────────────────────────────────────────────────────────────

/// How the outputs of an inline parity case are compared to the CPU oracle.
#[derive(Clone, Copy)]
enum Compare {
    /// Exact byte equality — integer / bool / metadata / data-movement ops.
    ExactBytes,
    /// Element-wise float parity within `tol`, decoding each output's dtype.
    Float { tol: f32 },
}

/// A single inline parity case: one node run on CUDA and on the CPU oracle.
struct Case {
    label: String,
    op: &'static str,
    domain: &'static str,
    opset: u64,
    inputs: Vec<Tensor>,
    outputs: Vec<(DataType, Vec<usize>)>,
    attrs: Vec<(&'static str, Attribute)>,
    compare: Compare,
}

/// How a covered op's conformance is validated.
enum Coverage {
    /// Inline parity cases executed by [`conformance_sweep_matches_cpu`].
    Sweep(Vec<Case>),
    /// Covered by a dedicated GPU parity suite (`tests/<suite>`). The
    /// coverage-of-coverage audit verifies the file exists and names the op.
    Dedicated {
        suite: &'static str,
        note: &'static str,
    },
}

/// One covered op and how it is validated.
struct ProfileEntry {
    op: &'static str,
    coverage: Coverage,
}

// ─────────────────────────────────────────────────────────────────────────────
// Case builders (DRY: one generator per op family, driven by data)
// ─────────────────────────────────────────────────────────────────────────────

const FLOAT_DTYPES: [DataType; 3] = [DataType::Float32, DataType::Float16, DataType::BFloat16];

/// Element-wise absolute tolerance per float dtype. f32 device intrinsics agree
/// with host libm to a few ulp; the half formats are dominated by their own
/// rounding step (values chosen small so the ulp stays tight).
fn float_tol(dtype: DataType) -> f32 {
    match dtype {
        DataType::Float32 => 1e-4,
        DataType::Float16 => 3e-3,
        DataType::BFloat16 => 3e-2,
        _ => unreachable!("non-float dtype {dtype:?}"),
    }
}

/// A unary float op fed domain-safe `values`, one case per f32/f16/bf16.
fn unary_float(op: &'static str, opset: u64, values: &[f32]) -> Vec<Case> {
    FLOAT_DTYPES
        .into_iter()
        .map(|dtype| {
            let shape = vec![values.len()];
            Case {
                label: format!("{op}[{dtype:?}]"),
                op,
                domain: "",
                opset,
                inputs: vec![float_input(dtype, &shape, values)],
                outputs: vec![(dtype, shape.clone())],
                attrs: vec![],
                compare: Compare::Float {
                    tol: float_tol(dtype),
                },
            }
        })
        .collect()
}

/// A binary float op with NumPy broadcasting, one case per f32/f16/bf16.
#[allow(clippy::too_many_arguments)]
fn binary_float(
    op: &'static str,
    opset: u64,
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
    out_shape: &[usize],
) -> Vec<Case> {
    FLOAT_DTYPES
        .into_iter()
        .map(|dtype| Case {
            label: format!("{op}[{dtype:?}]"),
            op,
            domain: "",
            opset,
            inputs: vec![
                float_input(dtype, a_shape, a),
                float_input(dtype, b_shape, b),
            ],
            outputs: vec![(dtype, out_shape.to_vec())],
            attrs: vec![],
            compare: Compare::Float {
                tol: float_tol(dtype),
            },
        })
        .collect()
}

/// Output shape of a reduction over `axes` (negative allowed) honouring
/// `keepdims`.
fn reduce_out_shape(in_shape: &[usize], axes: &[i64], keepdims: bool) -> Vec<usize> {
    let rank = in_shape.len() as i64;
    let reduced: Vec<usize> = axes
        .iter()
        .map(|&a| {
            if a < 0 {
                (a + rank) as usize
            } else {
                a as usize
            }
        })
        .collect();
    let mut out = Vec::new();
    for (axis, &dim) in in_shape.iter().enumerate() {
        if reduced.contains(&axis) {
            if keepdims {
                out.push(1);
            }
        } else {
            out.push(dim);
        }
    }
    out
}

/// An f32 reduction op across a few axes/keepdims combinations. Reductions
/// accumulate in f32; the surviving difference is a few ulp scaled by the
/// reduced magnitude.
fn reduction_f32(op: &'static str) -> Vec<Case> {
    let in_shape = vec![2usize, 3, 4];
    let count: usize = in_shape.iter().product();
    // Mixed positive/negative magnitudes keep max/min/mean well conditioned.
    let values: Vec<f32> = (0..count)
        .map(|v| (v as f32 * 0.37) - 4.0 + (v % 5) as f32 * 0.11)
        .collect();
    let axes_cases: &[(&[i64], bool)] = &[
        (&[1], true),
        (&[1], false),
        (&[0, 2], false),
        (&[-1], false),
    ];
    axes_cases
        .iter()
        .map(|&(axes, keepdims)| {
            let out_shape = reduce_out_shape(&in_shape, axes, keepdims);
            Case {
                label: format!("{op}(axes={axes:?},keepdims={keepdims})"),
                op,
                domain: "",
                opset: 13,
                inputs: vec![input(DataType::Float32, &in_shape, &values)],
                outputs: vec![(DataType::Float32, out_shape)],
                attrs: vec![
                    ("axes", Attribute::Ints(axes.to_vec())),
                    ("keepdims", Attribute::Int(keepdims as i64)),
                ],
                compare: Compare::Float { tol: 2e-2 },
            }
        })
        .collect()
}

/// `Cast` from f32 to `to`, comparing bytes exactly (CUDA mirrors the CPU
/// truncate-and-saturate / round-to-nearest numerics).
fn cast_case(to: DataType) -> Case {
    let values: Vec<f32> = vec![-3.6, -0.4, 0.0, 1.5, 2.9, 130.0, -130.0, 5.5];
    Case {
        label: format!("Cast[f32->{to:?}]"),
        op: "Cast",
        domain: "",
        opset: 13,
        inputs: vec![float_input(DataType::Float32, &[values.len()], &values)],
        outputs: vec![(to, vec![values.len()])],
        attrs: vec![("to", Attribute::Int(to as i64))],
        compare: Compare::ExactBytes,
    }
}

/// `CastLike` casting f32 input to the second input's dtype (`target`).
fn cast_like_case(target: DataType) -> Case {
    let values: Vec<f32> = vec![-3.6, -0.4, 0.0, 1.5, 2.9, 42.0];
    // The target-type tensor only conveys its dtype; a single element is enough.
    let target_tensor = Tensor {
        dtype: target,
        shape: vec![1],
        bytes: vec![0u8; target.storage_bytes(1)],
    };
    Case {
        label: format!("CastLike[f32->{target:?}]"),
        op: "CastLike",
        domain: "",
        opset: 15,
        inputs: vec![
            float_input(DataType::Float32, &[values.len()], &values),
            target_tensor,
        ],
        outputs: vec![(target, vec![values.len()])],
        attrs: vec![],
        compare: Compare::ExactBytes,
    }
}

/// Standard-domain `Gemm` cases: `Y = alpha·A'·B' + beta·C` (f32).
fn gemm_cases() -> Vec<Case> {
    let a: Vec<f32> = (0..6).map(|v| v as f32 * 0.5 - 1.0).collect(); // [2,3]
    let b: Vec<f32> = (0..12).map(|v| v as f32 * 0.25 - 0.5).collect(); // [3,4]
    let bias: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4]; // per-N, broadcasts to [2,4]
    let b_t: Vec<f32> = (0..12).map(|v| v as f32 * 0.2 - 0.7).collect(); // [4,3] for transB
    vec![
        // Plain A·B, no bias.
        Case {
            label: "Gemm(no-C)".into(),
            op: "Gemm",
            domain: "",
            opset: 13,
            inputs: vec![
                input(DataType::Float32, &[2, 3], &a),
                input(DataType::Float32, &[3, 4], &b),
            ],
            outputs: vec![(DataType::Float32, vec![2, 4])],
            attrs: vec![],
            compare: Compare::Float { tol: 1e-3 },
        },
        // alpha·A·B + beta·C with a broadcast per-N bias.
        Case {
            label: "Gemm(alpha,beta,C)".into(),
            op: "Gemm",
            domain: "",
            opset: 13,
            inputs: vec![
                input(DataType::Float32, &[2, 3], &a),
                input(DataType::Float32, &[3, 4], &b),
                input(DataType::Float32, &[4], &bias),
            ],
            outputs: vec![(DataType::Float32, vec![2, 4])],
            attrs: vec![
                ("alpha", Attribute::Float(0.75)),
                ("beta", Attribute::Float(1.0)),
            ],
            compare: Compare::Float { tol: 1e-3 },
        },
        // transB: B stored [N,K] = [4,3].
        Case {
            label: "Gemm(transB)".into(),
            op: "Gemm",
            domain: "",
            opset: 13,
            inputs: vec![
                input(DataType::Float32, &[2, 3], &a),
                input(DataType::Float32, &[4, 3], &b_t),
            ],
            outputs: vec![(DataType::Float32, vec![2, 4])],
            attrs: vec![("transB", Attribute::Int(1))],
            compare: Compare::Float { tol: 1e-3 },
        },
    ]
}

/// `com.microsoft::SkipLayerNormalization` (fused residual add + layernorm),
/// f32, exercising the optional `beta` and `bias` slots. Only the required `Y`
/// output is requested.
fn skip_layer_norm_case() -> Case {
    let rows = 2usize;
    let hidden = 4usize;
    let count = rows * hidden;
    let x: Vec<f32> = (0..count).map(|v| (v as f32 * 0.3) - 1.0).collect();
    let skip: Vec<f32> = (0..count).map(|v| (v as f32 * 0.1) - 0.5).collect();
    let gamma: Vec<f32> = vec![1.2, 0.8, 1.0, 0.5];
    let beta: Vec<f32> = vec![0.05, -0.1, 0.2, 0.0];
    let bias: Vec<f32> = vec![0.01, 0.02, -0.03, 0.04];
    Case {
        label: "SkipLayerNormalization[f32]".into(),
        op: "SkipLayerNormalization",
        domain: "com.microsoft",
        opset: 1,
        inputs: vec![
            input(DataType::Float32, &[rows, hidden], &x),
            input(DataType::Float32, &[rows, hidden], &skip),
            input(DataType::Float32, &[hidden], &gamma),
            input(DataType::Float32, &[hidden], &beta),
            input(DataType::Float32, &[hidden], &bias),
        ],
        outputs: vec![(DataType::Float32, vec![rows, hidden])],
        attrs: vec![("epsilon", Attribute::Float(1e-5))],
        compare: Compare::Float { tol: 2e-3 },
    }
}

/// `Not`: element-wise boolean negation.
fn not_case() -> Case {
    let values: Vec<u8> = vec![1, 0, 1, 0, 0, 1];
    Case {
        label: "Not[bool]".into(),
        op: "Not",
        domain: "",
        opset: 1,
        inputs: vec![input(DataType::Bool, &[values.len()], &values)],
        outputs: vec![(DataType::Bool, vec![values.len()])],
        attrs: vec![],
        compare: Compare::ExactBytes,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The conformance profile: one entry per CUDA_COVERED_OPS op.
// ─────────────────────────────────────────────────────────────────────────────

/// Convenience for a `Dedicated` entry.
fn dedicated(op: &'static str, suite: &'static str, note: &'static str) -> ProfileEntry {
    ProfileEntry {
        op,
        coverage: Coverage::Dedicated { suite, note },
    }
}

/// Convenience for a `Sweep` entry.
fn sweep(op: &'static str, cases: Vec<Case>) -> ProfileEntry {
    ProfileEntry {
        op,
        coverage: Coverage::Sweep(cases),
    }
}

/// The full conformance profile. Every op in [`CUDA_COVERED_OPS`] appears
/// exactly once. Inline `Sweep` entries carry new CUDA-vs-CPU parity cases
/// (concentrated on ops that previously had *no* parity test at all); the rest
/// are attributed to their dedicated GPU suite.
fn conformance_profile() -> Vec<ProfileEntry> {
    let mut p = Vec::with_capacity(CUDA_COVERED_OPS.len());

    // ── Inline parity sweep (previously untested ops get real coverage) ──────
    // Unary math.
    p.push(sweep(
        "Sqrt",
        unary_float("Sqrt", 13, &[0.0, 0.25, 1.0, 2.25, 9.0]),
    ));
    p.push(sweep(
        "Erf",
        unary_float("Erf", 13, &[-1.5, -0.5, 0.0, 0.7, 1.8]),
    ));
    p.push(sweep(
        "Tanh",
        unary_float("Tanh", 13, &[-2.0, -0.5, 0.0, 0.9, 2.5]),
    ));
    p.push(sweep(
        "Sigmoid",
        unary_float("Sigmoid", 13, &[-3.0, -0.7, 0.0, 0.8, 3.0]),
    ));
    p.push(sweep(
        "Abs",
        unary_float("Abs", 13, &[-2.5, -0.5, 0.0, 1.4, 2.5]),
    ));
    p.push(sweep(
        "Neg",
        unary_float("Neg", 13, &[-2.5, -0.5, 0.0, 1.4, 2.5]),
    ));
    p.push(sweep(
        "Reciprocal",
        unary_float("Reciprocal", 13, &[-2.0, -0.5, 0.5, 1.0, 4.0]),
    ));
    p.push(sweep(
        "Log",
        unary_float("Log", 13, &[0.1, 0.5, 1.0, 2.0, 5.0]),
    ));
    p.push(sweep(
        "Sign",
        unary_float("Sign", 13, &[-2.5, -0.5, 0.0, 1.4, 2.5]),
    ));
    p.push(sweep(
        "Floor",
        unary_float("Floor", 13, &[-2.5, -0.5, 0.0, 1.4, 2.6]),
    ));
    p.push(sweep(
        "Ceil",
        unary_float("Ceil", 13, &[-2.5, -0.5, 0.0, 1.4, 2.6]),
    ));
    p.push(sweep(
        "Round",
        unary_float("Round", 13, &[-2.5, -1.5, 0.0, 1.5, 2.6]),
    ));
    p.push(sweep(
        "Sin",
        unary_float("Sin", 13, &[-3.0, -1.0, 0.0, 1.0, 3.0]),
    ));
    p.push(sweep(
        "Cos",
        unary_float("Cos", 13, &[-3.0, -1.0, 0.0, 1.0, 3.0]),
    ));
    p.push(sweep(
        "Softplus",
        unary_float("Softplus", 13, &[-2.0, -0.5, 0.0, 1.0, 3.0]),
    ));

    // Binary math.
    p.push(sweep(
        "Pow",
        binary_float(
            "Pow",
            13,
            &[0.5, 1.0, 2.0, 3.0, 4.0],
            &[5],
            &[2.0, 0.5, 3.0, 1.0, 2.0],
            &[5],
            &[5],
        ),
    ));
    p.push(sweep(
        "Min",
        binary_float(
            "Min",
            13,
            &[-1.0, 2.0, 0.5, -3.0, 4.0],
            &[5],
            &[0.5, -2.0, 1.5, -1.0, 4.0],
            &[5],
            &[5],
        ),
    ));
    p.push(sweep(
        "Max",
        binary_float(
            "Max",
            13,
            &[-1.0, 2.0, 0.5, -3.0, 4.0],
            &[5],
            &[0.5, -2.0, 1.5, -1.0, 4.0],
            &[5],
            &[5],
        ),
    ));

    // Reductions.
    p.push(sweep("ReduceMean", reduction_f32("ReduceMean")));
    p.push(sweep("ReduceMax", reduction_f32("ReduceMax")));
    p.push(sweep("ReduceMin", reduction_f32("ReduceMin")));

    // Cast / CastLike.
    p.push(sweep(
        "Cast",
        vec![
            cast_case(DataType::Int32),
            cast_case(DataType::Int64),
            cast_case(DataType::Float16),
            cast_case(DataType::BFloat16),
        ],
    ));
    p.push(sweep(
        "CastLike",
        vec![
            cast_like_case(DataType::Int32),
            cast_like_case(DataType::Float16),
        ],
    ));

    // Logical, GEMM, fused norm.
    p.push(sweep("Not", vec![not_case()]));
    p.push(sweep("Gemm", gemm_cases()));
    p.push(sweep(
        "SkipLayerNormalization",
        vec![skip_layer_norm_case()],
    ));

    // ── Dedicated GPU parity suites (verified to name their op) ──────────────
    // GEMM / quantized-matmul family.
    p.push(dedicated(
        "MatMul",
        "matmul_gpu.rs",
        "cuBLASLt dense/batched GEMM",
    ));
    p.push(dedicated(
        "MatMulNBits",
        "matmul_nbits_gpu.rs",
        "packed INT4 block dequant + GEMM",
    ));
    p.push(dedicated(
        "QMoE",
        "qmoe_gpu.rs",
        "grouped block-dequant expert GEMM",
    ));
    p.push(dedicated(
        "BlockQuantizedMatMul",
        "block_quantized_matmul_gpu.rs",
        "block-quantized weights",
    ));
    p.push(dedicated(
        "FusedMatMulBias",
        "fused_epilogue_gpu.rs",
        "cuBLASLt BIAS epilogue",
    ));
    p.push(dedicated(
        "FusedGemm",
        "fused_epilogue_gpu.rs",
        "cuBLASLt BIAS/RELU/GELU epilogue",
    ));

    // Convolution / pooling.
    p.push(dedicated("Conv", "conv_gpu.rs", "cuDNN 2-D conv"));
    p.push(dedicated("MaxPool", "pooling_gpu.rs", "cuDNN pooling"));
    p.push(dedicated("AveragePool", "pooling_gpu.rs", "cuDNN pooling"));

    // Attention / KV / sparse family.
    p.push(dedicated(
        "Attention",
        "standard_attention_gpu.rs",
        "flash-style attention",
    ));
    p.push(dedicated(
        "GroupQueryAttention",
        "group_query_attention_gpu.rs",
        "GQA with KV cache",
    ));
    p.push(dedicated(
        "VarlenAttention",
        "varlen_attention_gpu.rs",
        "variable-length attention",
    ));
    p.push(dedicated(
        "PackedVarlenAttention",
        "packed_varlen_attention_gpu.rs",
        "packed varlen attention",
    ));
    p.push(dedicated(
        "CompressedSparseAttention",
        "compressed_sparse_attention_gpu.rs",
        "CSA",
    ));
    p.push(dedicated(
        "SparseKvGather",
        "sparse_kv_gather_gpu.rs",
        "sparse KV gather",
    ));
    p.push(dedicated(
        "IndexShare",
        "index_share_gpu.rs",
        "shared index buffer",
    ));
    p.push(dedicated(
        "RotaryEmbedding",
        "rope_capture_gpu.rs",
        "RoPE, graph-capture safe",
    ));

    // Normalization / softmax.
    p.push(dedicated(
        "LayerNormalization",
        "normalization_fp16_gpu.rs",
        "fused layernorm",
    ));
    p.push(dedicated(
        "SimplifiedLayerNormalization",
        "simplified_layer_norm_gpu.rs",
        "RMS-style norm",
    ));
    p.push(dedicated(
        "SkipSimplifiedLayerNormalization",
        "skip_simplified_layer_norm_gpu.rs",
        "fused residual + RMS norm",
    ));
    p.push(dedicated(
        "RMSNormalization",
        "normalization_fp16_gpu.rs",
        "RMS norm",
    ));
    p.push(dedicated("Softmax", "indexing_gpu.rs", "row softmax"));

    // Activations (attribute-driven).
    p.push(dedicated(
        "Relu",
        "pointwise_gpu.rs",
        "elementwise activation",
    ));
    p.push(dedicated(
        "Gelu",
        "fused_epilogue_gpu.rs",
        "GELU activation",
    ));
    p.push(dedicated("Silu", "activations_gpu.rs", "SiLU activation"));
    p.push(dedicated("LeakyRelu", "activations_gpu.rs", "alpha-driven"));
    p.push(dedicated("Elu", "activations_gpu.rs", "alpha-driven"));
    p.push(dedicated(
        "HardSigmoid",
        "activations_gpu.rs",
        "alpha/beta-driven",
    ));
    p.push(dedicated("Clip", "activations_gpu.rs", "min/max clamp"));
    p.push(dedicated("Softsign", "activations_gpu.rs", "x/(1+|x|)"));
    p.push(dedicated(
        "Selu",
        "activations_gpu.rs",
        "alpha/gamma-driven",
    ));

    // Elementwise binary / comparison / logical.
    p.push(dedicated("Add", "pointwise_gpu.rs", "broadcasting binary"));
    p.push(dedicated("Sub", "pointwise_gpu.rs", "broadcasting binary"));
    p.push(dedicated("Mul", "pointwise_gpu.rs", "broadcasting binary"));
    p.push(dedicated("Div", "pointwise_gpu.rs", "broadcasting binary"));
    p.push(dedicated("Exp", "pointwise_gpu.rs", "unary math"));
    p.push(dedicated("And", "pointwise_gpu.rs", "boolean binary"));
    p.push(dedicated("Or", "pointwise_gpu.rs", "boolean binary"));
    p.push(dedicated("Xor", "pointwise_gpu.rs", "boolean binary"));
    p.push(dedicated("Equal", "pointwise_gpu.rs", "comparison -> bool"));
    p.push(dedicated(
        "Greater",
        "pointwise_gpu.rs",
        "comparison -> bool",
    ));
    p.push(dedicated("Less", "pointwise_gpu.rs", "comparison -> bool"));
    p.push(dedicated(
        "GreaterOrEqual",
        "pointwise_gpu.rs",
        "comparison -> bool",
    ));
    p.push(dedicated(
        "LessOrEqual",
        "pointwise_gpu.rs",
        "comparison -> bool",
    ));

    // Trigonometric / hyperbolic family (batch 1).
    for op in [
        "Tan", "Sinh", "Cosh", "Asin", "Acos", "Atan", "Asinh", "Acosh", "Atanh",
    ] {
        p.push(dedicated(
            op,
            "op_coverage_batch_gpu.rs",
            "trig/hyperbolic unary parity",
        ));
    }

    // Extended reductions / activations / variadic / misc (batches 2 & 3).
    for op in [
        "ReduceProd",
        "ReduceSumSquare",
        "ReduceL1",
        "ReduceL2",
        "ReduceLogSum",
        "ReduceLogSumExp",
    ] {
        p.push(dedicated(
            op,
            "op_coverage_batch_gpu.rs",
            "extended reduction",
        ));
    }
    p.push(dedicated(
        "Swish",
        "op_coverage_batch_gpu.rs",
        "extended activation",
    ));
    p.push(dedicated(
        "ThresholdedRelu",
        "op_coverage_batch_gpu.rs",
        "extended activation",
    ));
    p.push(dedicated(
        "Sum",
        "op_coverage_batch_gpu.rs",
        "variadic elementwise",
    ));
    p.push(dedicated(
        "Mean",
        "op_coverage_batch_gpu.rs",
        "variadic elementwise",
    ));
    p.push(dedicated("Mod", "op_coverage_batch_gpu.rs", "modulo"));
    p.push(dedicated(
        "IsInf",
        "op_coverage_batch_gpu.rs",
        "unary predicate",
    ));
    p.push(dedicated(
        "IsNaN",
        "op_coverage_batch_gpu.rs",
        "unary predicate",
    ));
    p.push(dedicated(
        "PRelu",
        "op_coverage_batch_gpu.rs",
        "parametric ReLU",
    ));
    p.push(dedicated(
        "Identity",
        "op_coverage_batch_gpu.rs",
        "byte copy",
    ));
    p.push(dedicated("Flatten", "op_coverage_batch_gpu.rs", "reshape"));
    p.push(dedicated(
        "Size",
        "op_coverage_batch_gpu.rs",
        "element count scalar",
    ));
    p.push(dedicated(
        "Trilu",
        "op_coverage_batch_gpu.rs",
        "triangular mask",
    ));

    // Reduction / metadata / movement covered elsewhere.
    p.push(dedicated("ReduceSum", "movement_gpu.rs", "reduction"));
    p.push(dedicated("Gather", "movement_gpu.rs", "indexed gather"));
    p.push(dedicated("Shape", "movement_gpu.rs", "shape metadata"));
    p.push(dedicated(
        "Constant",
        "movement_gpu.rs",
        "constant materialize",
    ));
    p.push(dedicated(
        "ConstantOfShape",
        "opset24_ops_gpu.rs",
        "fill from shape",
    ));
    p.push(dedicated("OneHot", "opset24_ops_gpu.rs", "one-hot encode"));

    // Structural data-movement.
    for op in [
        "Concat",
        "Expand",
        "Reshape",
        "Slice",
        "Split",
        "Squeeze",
        "Tile",
        "Transpose",
        "Unsqueeze",
        "Where",
    ] {
        p.push(dedicated(op, "construction_gpu.rs", "shape/data movement"));
    }

    // Indexed / scan.
    p.push(dedicated("TopK", "indexing_gpu.rs", "top-k"));
    p.push(dedicated("CumSum", "indexing_gpu.rs", "prefix scan"));
    p.push(dedicated(
        "GatherElements",
        "indexing_gpu.rs",
        "gather along axis",
    ));
    p.push(dedicated(
        "ScatterElements",
        "indexing_gpu.rs",
        "scatter along axis",
    ));

    p
}

// ─────────────────────────────────────────────────────────────────────────────
// Coverage-of-coverage audits (no GPU required — run everywhere, incl. CI)
// ─────────────────────────────────────────────────────────────────────────────

/// Every op the CUDA EP claims to cover must have a conformance profile entry,
/// and no profile entry may reference an op that is no longer covered.
///
/// This is the highest-value guard: it fails the moment an op is added to
/// `CUDA_COVERED_OPS` without a corresponding parity test — the "claimed but
/// untested" defect class (e.g. the `ReduceLogSumExp` and bf16 misses).
#[test]
fn every_covered_op_has_a_conformance_entry() {
    let profile = conformance_profile();
    let profile_ops: HashSet<&str> = profile.iter().map(|e| e.op).collect();
    let covered: HashSet<&str> = CUDA_COVERED_OPS.iter().copied().collect();

    let missing: Vec<&str> = CUDA_COVERED_OPS
        .iter()
        .copied()
        .filter(|op| !profile_ops.contains(op))
        .collect();
    assert!(
        missing.is_empty(),
        "these CUDA_COVERED_OPS have no conformance profile entry (claimed but \
         untested — add them to conformance_profile()): {missing:?}"
    );

    let stale: Vec<&str> = profile_ops
        .iter()
        .copied()
        .filter(|op| !covered.contains(op))
        .collect();
    assert!(
        stale.is_empty(),
        "these profile entries reference ops not in CUDA_COVERED_OPS \
         (stale — remove or fix the entry): {stale:?}"
    );

    assert_eq!(
        profile.len(),
        CUDA_COVERED_OPS.len(),
        "profile must have exactly one entry per covered op"
    );
}

/// No op may appear twice in the profile, and every `Sweep` entry must carry at
/// least one case.
#[test]
fn profile_has_no_duplicate_entries() {
    let profile = conformance_profile();
    let mut seen = HashSet::new();
    for entry in &profile {
        assert!(
            seen.insert(entry.op),
            "duplicate conformance profile entry for {:?}",
            entry.op
        );
        if let Coverage::Sweep(cases) = &entry.coverage {
            assert!(
                !cases.is_empty(),
                "Sweep entry for {:?} has no cases",
                entry.op
            );
        }
    }
}

/// Each `Dedicated` suite file must exist and actually name its op, so a
/// deleted, renamed, or gutted suite cannot silently leave an op unverified
/// while it stays in `CUDA_COVERED_OPS`.
#[test]
fn dedicated_suites_exist_and_name_their_op() {
    let tests_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    for entry in conformance_profile() {
        if let Coverage::Dedicated { suite, note } = entry.coverage {
            let path = tests_dir.join(suite);
            let src = std::fs::read_to_string(&path).unwrap_or_else(|error| {
                panic!(
                    "dedicated suite {suite} for op {:?} could not be read at {}: {error}",
                    entry.op,
                    path.display()
                )
            });
            let needle = format!("\"{}\"", entry.op);
            assert!(
                src.contains(&needle),
                "dedicated suite {suite} does not name op {:?} (looked for {needle}); \
                 the coverage claim is stale",
                entry.op
            );
            eprintln!("{:>32}  ✓ {suite}  ({note})", entry.op);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU parity sweep (graceful-skips without a CUDA device)
// ─────────────────────────────────────────────────────────────────────────────

/// Run one inline case on CUDA and compare to the CPU oracle.
fn run_case(ep: &CudaExecutionProvider, case: &Case) {
    let cuda = run_cuda(
        ep,
        case.op,
        case.domain,
        case.opset,
        &case.inputs,
        &case.outputs,
        &case.attrs,
    );
    let cpu = run_cpu(
        case.op,
        case.domain,
        case.opset,
        &case.inputs,
        &case.outputs,
        &case.attrs,
    );
    assert_eq!(
        cuda.len(),
        cpu.len(),
        "{}: output count mismatch (CUDA {} vs CPU {})",
        case.label,
        cuda.len(),
        cpu.len()
    );
    match case.compare {
        Compare::ExactBytes => {
            for (index, (got, want)) in cuda.iter().zip(&cpu).enumerate() {
                assert_eq!(
                    got, want,
                    "{}: output {index} bytes differ (CUDA vs CPU oracle)",
                    case.label
                );
            }
        }
        Compare::Float { tol } => {
            for (index, ((got, want), (dtype, _))) in
                cuda.iter().zip(&cpu).zip(&case.outputs).enumerate()
            {
                let got_f = decode_floats(got, *dtype);
                let want_f = decode_floats(want, *dtype);
                assert_close(
                    &format!("{} out{index}", case.label),
                    *dtype,
                    &got_f,
                    &want_f,
                    tol,
                );
            }
        }
    }
}

/// Execute every inline `Sweep` case against the CPU oracle on the real GPU.
/// Skips cleanly on a host without a CUDA device.
#[test]
fn conformance_sweep_matches_cpu() {
    let Some(ep) = cuda_ep() else { return };
    let mut ran = 0usize;
    for entry in conformance_profile() {
        if let Coverage::Sweep(cases) = &entry.coverage {
            for case in cases {
                run_case(&ep, case);
                ran += 1;
            }
        }
    }
    assert!(ran > 0, "expected at least one inline conformance case");
    eprintln!("conformance sweep: {ran} inline CUDA-vs-CPU parity cases passed");
}

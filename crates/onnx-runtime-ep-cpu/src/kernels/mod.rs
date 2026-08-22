//! CPU kernels for the Phase-1 BERT-on-CPU correctness milestone (`docs/architecture/ORT2.md`
//! §4.4). One [`Kernel`] per ONNX op, keyed purely by op type — there are **no**
//! model-specific shapes or names anywhere in this crate; BERT is only the
//! validation target.
//!
//! ## Pure-Rust reference kernels (architecture decision)
//!
//! These are straightforward, **correct** pure-Rust kernels — the ops other than
//! the GEMM hot spot use naive loops with no FFI or `cc` build dependency. The
//! MatMul GEMM went through the Phase-1.5 perf pass (`docs/architecture/ORT2.md` §25.2): its
//! default backend is a blocked, register-tiled, rayon-parallelized pure-Rust
//! kernel. Every kernel sits behind the
//! [`Kernel`] trait, so backends swap in **without touching the EP contract or
//! the session**. The seam is [`Kernel`] itself; see [`matmul`] for the hot spot
//! and [`crate::backend`] for backend selection.
//!
//! ## Strided inputs
//!
//! Kernels accept non-contiguous inputs by reading through
//! [`to_dense_f32`]/[`to_dense_i64`], which materialize a view (applying its
//! strides and byte offset) into a dense row-major buffer. This keeps the
//! per-kernel `unsafe` surface to the two element accessors in this module.

use onnx_runtime_ep_api::{
    EpError, KernelFactory, OpKey, OpRegistry, Result, TensorMut, TensorView,
};
use onnx_runtime_ir::DataType;

use crate::strided::{elem_offset, next_index, numel};

// ─── Kernel registry entry derivation ────────────────────────────────────────

/// Descriptor of one registered op for the plugin kernel-registry advertisement.
/// Derived from the real `OpRegistry` registration calls — not hand-maintained.
#[derive(Clone, Debug)]
pub struct CpuOpDescriptor {
    pub op_type: String,
    pub domain: String,
    pub since_version: u64,
    /// Supported element types for the "T" type-constraint. Derived from the
    /// kernel's actual dtype dispatch — fail closed (f32-only) when unknown.
    pub supported_dtypes: &'static [DataType],
}

/// Wrapper around `OpRegistry` that records every registered key.
struct RecordingOpRegistry {
    inner: OpRegistry,
    keys: Vec<(String, String, u64)>,
}

impl RecordingOpRegistry {
    fn new() -> Self {
        Self {
            inner: OpRegistry::new(),
            keys: Vec::new(),
        }
    }

    fn register(&mut self, key: OpKey, factory: Box<dyn KernelFactory>) {
        self.keys
            .push((key.op_type.clone(), key.domain.clone(), key.since_version));
        self.inner.register(key, factory);
    }

    fn into_parts(self) -> (OpRegistry, Vec<(String, String, u64)>) {
        (self.inner, self.keys)
    }
}

// ── Dtype categories ─────────────────────────────────────────────────────────
//
// Derived from actual kernel dispatch macros/implementations:
// - `dispatch_arith!` → ARITH_DTYPES (f32,f16,bf16,f64,i8..u64)
// - `dispatch_float!` → FLOAT_DTYPES (f32,f16,bf16,f64)
// - byte-movers (Identity,Reshape,etc.) → ALL_DTYPES
// - f32-only kernels → F32_ONLY
// - comparison/logical → handled per-op

/// All dtypes the CPU EP can physically handle (byte-mover ops).
static ALL_DTYPES: &[DataType] = &[
    DataType::Float32,
    DataType::Float16,
    DataType::BFloat16,
    DataType::Float64,
    DataType::Int8,
    DataType::Int16,
    DataType::Int32,
    DataType::Int64,
    DataType::Uint8,
    DataType::Uint16,
    DataType::Uint32,
    DataType::Uint64,
    DataType::Bool,
];

/// Dtypes supported by `dispatch_arith!` (arithmetic ops).
static ARITH_DTYPES: &[DataType] = &[
    DataType::Float32,
    DataType::Float16,
    DataType::BFloat16,
    DataType::Float64,
    DataType::Int8,
    DataType::Int16,
    DataType::Int32,
    DataType::Int64,
    DataType::Uint8,
    DataType::Uint16,
    DataType::Uint32,
    DataType::Uint64,
];

/// Dtypes supported by `dispatch_float!` (transcendental/accumulate ops).
static FLOAT_DTYPES: &[DataType] = &[
    DataType::Float32,
    DataType::Float16,
    DataType::BFloat16,
    DataType::Float64,
];

/// Fail-closed default: only f32.
static F32_ONLY: &[DataType] = &[DataType::Float32];

/// Every dtype that appears on a `com.microsoft::MatMulNBits` edge.
///
/// This op is block-quantized, so it is *inherently* mixed-dtype: `A`, `scales`,
/// the optional `bias` and `Y` are float, `B` and the optional `zero_points` are
/// packed `uint8`, and the optional `g_idx` is `int32`. The plugin's node filter
/// (`node_passes_dtype_filter`) requires **every** input and output dtype of a
/// node to appear in this list, so declaring the float set alone silently
/// excluded the op from every claim: an int4 decode graph ran on ORT's CPU
/// kernels even when this EP was the selected one.
///
/// The float members are exactly what `require_float_compute_dtype` accepts for
/// `A`/`scales`/`Y` in `matmul_nbits::MatMulNBitsKernel::execute`, and the
/// integer members are exactly the `require_dtype` calls beside them, so this
/// list is derived from the kernel rather than widened to make a test pass.
static MATMUL_NBITS_DTYPES: &[DataType] = &[
    DataType::Float32,
    DataType::Float16,
    DataType::BFloat16,
    DataType::Uint8,
    DataType::Int32,
];

/// The float dtypes `require_float_compute_dtype` accepts.
///
/// Also what the MLAS-backed `Conv` accepts. Those kernels compute in f32 and
/// widen or narrow around it, so they cover f16 and bf16 but genuinely cannot
/// do f64 -- `ConvKernel::execute` rejects it outright. Distinct from
/// [`FLOAT_DTYPES`], which includes `Float64`.
///
/// Advertising `FLOAT_DTYPES` for such an op is an over-claim rather than a
/// harmless one: the plugin turns this list into the `KernelRegistryEntry`
/// dtype filter, so an f64 `Conv` would clear capability, get compiled, and
/// then fail at `Run` instead of being reported unsupported up front.
static FLOAT_COMPUTE_DTYPES: &[DataType] =
    &[DataType::Float32, DataType::Float16, DataType::BFloat16];

/// The two quantized storage dtypes `QLinearMatMul` operands may use.
static QUANTIZED_STORAGE_DTYPES: &[DataType] = &[DataType::Uint8, DataType::Int8];

static U8_ONLY: &[DataType] = &[DataType::Uint8];

static I32_ONLY: &[DataType] = &[DataType::Int32];

/// Integer index/length inputs, matching `to_dense_i64`'s acceptance set.
static INDEX_DTYPES: &[DataType] = &[DataType::Int64, DataType::Int32];

/// Per-input-slot dtype constraints for a mixed-dtype op.
///
/// `supported_dtypes_for_op` returns the *union* of the dtypes on an op's
/// edges, and the node filter tests membership in that union. For a uniform op
/// that is exact, but for a block-quantized op it is strictly weaker than the
/// kernel's own rule: `MatMulNBits`'s union contains both `float16` and
/// `uint8`, so a node with `float16` `zero_points` — which the ONNX contrib
/// spec permits, and which ORT's own kernel accepts — passes the union test
/// and is claimed, and then fails inside `execute` where the only outcome is a
/// run-time error.
///
/// These lists restore the missing precision. A slot listed here is checked
/// against its own set instead of the union; absent slots and slots not listed
/// keep the union rule. They must stay in step with the `require_dtype` calls
/// in the corresponding kernel's `execute`.
///
/// The same mechanism also fixes the *opposite* failure, which is the more
/// damaging one. `supported_dtypes_for_op` returns `FLOAT_DTYPES` for the
/// attention family, because that is what their tensor edges carry -- but
/// `RotaryEmbedding`'s `position_ids` is int64, and
/// `GroupQueryAttention`'s `seqlens_k` / `total_sequence_length` are int32. The
/// union test then rejects those slots and the node is silently handed to ORT's
/// CPU EP, even though our kernel runs it. That contradicts this EP's contract
/// (see `provider.rs::supports_op`): a node we can execute is never
/// delegated. Listing the integer slots explicitly is what makes the two most
/// important decode ops in the engine reachable at all;
/// `plugin_ort_e2e.rs::no_supported_node_is_ever_left_to_the_ort_cpu_ep` is the
/// falsifier, and it caught both of them.
pub fn input_dtype_constraints_for_op(
    op_type: &str,
    domain: &str,
) -> &'static [(usize, &'static [DataType])] {
    // A, B, scales, zero_points, g_idx, bias.
    static MATMUL_NBITS_SLOTS: &[(usize, &[DataType])] = &[
        (0, FLOAT_COMPUTE_DTYPES),
        (1, U8_ONLY),
        (2, FLOAT_COMPUTE_DTYPES),
        (3, U8_ONLY),
        (4, I32_ONLY),
        (5, FLOAT_COMPUTE_DTYPES),
    ];
    // a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zp.
    static QLINEAR_MATMUL_SLOTS: &[(usize, &[DataType])] = &[
        (0, QUANTIZED_STORAGE_DTYPES),
        (1, F32_ONLY),
        (2, QUANTIZED_STORAGE_DTYPES),
        (3, QUANTIZED_STORAGE_DTYPES),
        (4, F32_ONLY),
        (5, QUANTIZED_STORAGE_DTYPES),
        (6, F32_ONLY),
        (7, QUANTIZED_STORAGE_DTYPES),
    ];
    // X, position_ids, cos_cache, sin_cache.
    static MSFT_ROTARY_SLOTS: &[(usize, &[DataType])] = &[(1, INDEX_DTYPES)];
    // X, cos_cache, sin_cache, position_ids.
    static ONNX_ROTARY_SLOTS: &[(usize, &[DataType])] = &[(3, INDEX_DTYPES)];
    // query, key, value, past_key, past_value, seqlens_k, total_sequence_length,
    // cos_cache, sin_cache, position_ids. `seqlens_k` is strictly int32 (the
    // kernel rejects anything else); the other two go through `to_dense_i64`.
    // Slot 9 is optional and only present with `do_rotary`, but leaving it
    // unlisted made a `do_rotary` GQA node with explicit int64 positions fail
    // the float union and go to ORT -- the exact bug this table exists to stop.
    static GQA_SLOTS: &[(usize, &[DataType])] =
        &[(5, I32_ONLY), (6, INDEX_DTYPES), (9, INDEX_DTYPES)];
    // query, key, value, bias, key_padding_mask, attention_bias, past_key, past_value.
    static MHA_SLOTS: &[(usize, &[DataType])] = &[(4, INDEX_DTYPES)];
    // input, weights, bias, mask_index, past, attention_bias.
    static MSFT_ATTENTION_SLOTS: &[(usize, &[DataType])] = &[(3, INDEX_DTYPES)];
    // query, key, value, bias, token_offset, cumulative_sequence_length.
    // The last two are strictly int32 (`packed_multi_head_attention.rs`
    // rejects anything else). ORT has no CPU kernel for this op, so failing
    // the union here does not fall back -- it fails session creation.
    static PACKED_MHA_SLOTS: &[(usize, &[DataType])] = &[(4, I32_ONLY), (5, I32_ONLY)];
    // QMoE stores its experts int4/int8-packed in uint8, so the float union
    // rejects the weight and zero-point slots outright. Slot order:
    // input, router_probs, fc1_w, fc1_scales, fc1_bias, fc2_w, fc2_scales,
    // fc2_bias, fc3_w, fc3_scales, fc3_bias, fc1_zp, fc2_zp, fc3_zp,
    // router_weights. Mirrors the `require_dtype` calls in `qmoe.rs`.
    static QMOE_SLOTS: &[(usize, &[DataType])] = &[
        (2, U8_ONLY),
        (5, U8_ONLY),
        (8, U8_ONLY),
        (11, U8_ONLY),
        (12, U8_ONLY),
        (13, U8_ONLY),
    ];
    match (op_type, domain) {
        ("MatMulNBits", "com.microsoft") => MATMUL_NBITS_SLOTS,
        ("QLinearMatMul", "") => QLINEAR_MATMUL_SLOTS,
        ("RotaryEmbedding", "com.microsoft") => MSFT_ROTARY_SLOTS,
        ("RotaryEmbedding", "") => ONNX_ROTARY_SLOTS,
        ("GroupQueryAttention", "com.microsoft") => GQA_SLOTS,
        ("MultiHeadAttention", "com.microsoft") => MHA_SLOTS,
        ("Attention", "com.microsoft") => MSFT_ATTENTION_SLOTS,
        ("PackedMultiHeadAttention", "com.microsoft") => PACKED_MHA_SLOTS,
        ("QMoE", "com.microsoft") => QMOE_SLOTS,
        _ => &[],
    }
}

/// Determine the supported dtypes for a given (op_type, domain) based on the
/// actual kernel dispatch implementation. Fail closed: unknown ops get f32 only.
pub fn supported_dtypes_for_op(op_type: &str, domain: &str) -> &'static [DataType] {
    // Byte-mover ops: dtype-agnostic (just copy bytes, no arithmetic).
    match (op_type, domain) {
        ("Identity", "")
        | ("Reshape", "")
        | ("Flatten", "")
        | ("Squeeze", "")
        | ("Unsqueeze", "")
        | ("Expand", "")
        | ("Concat", "")
        | ("Slice", "")
        | ("Split", "")
        | ("Transpose", "")
        | ("Gather", "")
        | ("GatherElements", "")
        | ("GatherND", "")
        | ("ScatterElements", "")
        | ("ScatterND", "")
        | ("Shape", "")
        | ("Size", "")
        | ("Pad", "")
        | ("ConstantOfShape", "")
        | ("Constant", "")
        | ("Tile", "")
        | ("Compress", "")
        | ("Trilu", "")
        | ("OneHot", "")
        | ("Dropout", "")
        | ("Unique", "") => ALL_DTYPES,

        // Arithmetic ops (dispatch_arith!): f32, f16, bf16, f64, int types.
        ("Add", "")
        | ("Sub", "")
        | ("Mul", "")
        | ("Div", "")
        | ("Mod", "")
        | ("Pow", "")
        | ("Min", "")
        | ("Max", "")
        | ("Sum", "")
        | ("Mean", "") => ARITH_DTYPES,

        // MatMul supports f32 natively + f16/bf16 via half_gemm.
        ("MatMul", "") | ("Gemm", "") => FLOAT_DTYPES,

        // Float-only ops (dispatch_float! or explicit float handling).
        ("Sqrt", "")
        | ("Erf", "")
        | ("Tanh", "")
        | ("Exp", "")
        | ("Log", "")
        | ("Sigmoid", "")
        | ("Softplus", "")
        | ("Softsign", "")
        | ("Reciprocal", "")
        | ("Sin", "")
        | ("Cos", "")
        | ("Tan", "")
        | ("Acos", "")
        | ("Acosh", "")
        | ("Asin", "")
        | ("Asinh", "")
        | ("Atan", "")
        | ("Atanh", "")
        | ("Cosh", "")
        | ("Sinh", "")
        | ("Abs", "")
        | ("Neg", "")
        | ("Sign", "")
        | ("Floor", "")
        | ("Ceil", "")
        | ("Round", "")
        | ("Relu", "")
        | ("Elu", "")
        | ("LeakyRelu", "")
        | ("HardSigmoid", "")
        | ("Selu", "")
        | ("ThresholdedRelu", "")
        | ("Celu", "")
        | ("Mish", "") => FLOAT_DTYPES,

        // Softmax, LogSoftmax, ReduceMean, LayerNorm, etc.: float-only.
        ("Softmax", "")
        | ("LogSoftmax", "")
        | ("ReduceMean", "")
        | ("ReduceSum", "")
        | ("ReduceMax", "")
        | ("ReduceMin", "")
        | ("ReduceProd", "")
        | ("ReduceSumSquare", "")
        | ("ReduceL1", "")
        | ("ReduceL2", "")
        | ("ReduceLogSum", "")
        | ("ReduceLogSumExp", "")
        | ("LayerNormalization", "")
        | ("Gelu", "")
        | ("RMSNormalization", "")
        | ("RotaryEmbedding", "")
        | ("Swish", "")
        | ("Attention", "")
        | ("LpNormalization", "")
        | ("Hardmax", "") => FLOAT_DTYPES,

        // Cast/CastLike handle all types.
        ("Cast", "") | ("CastLike", "") => ALL_DTYPES,

        // Clip, ArgMax, ArgMin, TopK: arithmetic types.
        ("Clip", "") | ("ArgMax", "") | ("ArgMin", "") | ("TopK", "") => ARITH_DTYPES,

        // Logical: bool/int inputs.
        ("And", "") | ("Or", "") | ("Xor", "") | ("Not", "") => &[DataType::Bool],
        ("Equal", "")
        | ("Greater", "")
        | ("GreaterOrEqual", "")
        | ("Less", "")
        | ("LessOrEqual", "") => ARITH_DTYPES,

        // Where: all types (condition is bool, data can be anything).
        ("Where", "") => ALL_DTYPES,

        // NonZero: all types.
        ("NonZero", "") | ("NonMaxSuppression", "") => ALL_DTYPES,

        // Bitwise: integer types.
        ("BitShift", "")
        | ("BitwiseAnd", "")
        | ("BitwiseOr", "")
        | ("BitwiseXor", "")
        | ("BitwiseNot", "") => &[
            DataType::Uint8,
            DataType::Uint16,
            DataType::Uint32,
            DataType::Uint64,
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
        ],

        // IsInf, IsNaN: float inputs.
        ("IsInf", "") | ("IsNaN", "") | ("EyeLike", "") => FLOAT_DTYPES,

        // Quantization: specific types but advertise broad.
        ("QuantizeLinear", "")
        | ("DequantizeLinear", "")
        | ("DynamicQuantizeLinear", "")
        | ("QLinearMatMul", "") => ARITH_DTYPES,

        // Sequence/range ops.
        ("Range", "") | ("CumSum", "") | ("CumProd", "") => ARITH_DTYPES,

        // Window functions.
        ("HannWindow", "") | ("HammingWindow", "") | ("BlackmanWindow", "") | ("DFT", "") => {
            FLOAT_DTYPES
        }

        // com.microsoft contrib ops.
        ("LayerNormalization", "com.microsoft")
        | ("FusedMatMulBias", "com.microsoft")
        | ("FusedGemm", "com.microsoft")
        | ("FusedAttention", "com.microsoft")
        | ("GroupQueryAttention", "com.microsoft")
        | ("MultiHeadAttention", "com.microsoft")
        | ("Attention", "com.microsoft")
        | ("Gelu", "com.microsoft")
        | ("BiasGelu", "com.microsoft")
        | ("FastGelu", "com.microsoft")
        | ("QuickGelu", "com.microsoft")
        | ("Silu", "com.microsoft")
        | ("SkipLayerNormalization", "com.microsoft")
        | ("SimplifiedLayerNormalization", "com.microsoft")
        | ("SkipSimplifiedLayerNormalization", "com.microsoft")
        | ("RotaryEmbedding", "com.microsoft")
        | ("CausalConvWithState", "com.microsoft")
        | ("LinearAttention", "com.microsoft")
        | ("GatherBlockQuantized", "com.microsoft") => FLOAT_DTYPES,

        ("SimplifiedLayerNormalization", "") | ("LinearAttention", "") => FLOAT_DTYPES,

        ("MatMulNBits", "com.microsoft") => MATMUL_NBITS_DTYPES,
        // `moe.rs` widens f16/bf16 to f32, computes, and narrows on the way
        // out, so advertising f32 alone declined every realistic MoE node --
        // production mixtures are exported in half precision. `QMoE` is f32 in
        // and out with the experts carried as packed uint8, whose slots are
        // listed in `input_dtype_constraints_for_op`.
        ("MoE", "com.microsoft") => FLOAT_COMPUTE_DTYPES,
        ("QMoE", "com.microsoft") => F32_ONLY,

        // pkg.nxrt custom ops: f32-only (fail closed).
        (_, "pkg.nxrt") => F32_ONLY,

        // CNN ops (feature-gated). `Conv` computes through MLAS in f32 and
        // rejects f64 in `ConvKernel::execute`, so it must not advertise it.
        ("Conv", "") => FLOAT_COMPUTE_DTYPES,

        // The rest of the CNN family dispatches through `dispatch_float!` and
        // does handle f64.
        ("ConvTranspose", "")
        | ("AveragePool", "")
        | ("MaxPool", "")
        | ("GlobalAveragePool", "")
        | ("GlobalMaxPool", "")
        | ("LpPool", "")
        | ("GlobalLpPool", "")
        | ("Resize", "")
        | ("GridSample", "")
        | ("AffineGrid", "")
        | ("Col2Im", "")
        | ("CenterCropPad", "")
        | ("SpaceToDepth", "")
        | ("BatchNormalization", "")
        | ("InstanceNormalization", "")
        | ("GroupNormalization", "")
        | ("PRelu", "") => FLOAT_DTYPES,

        // NCHWC domain ops: f32-only.
        (_, "com.microsoft.nchwc") => F32_ONLY,

        // Fail closed: unknown op → f32 only.
        _ => F32_ONLY,
    }
}

pub mod activations;
pub mod add;
pub mod attention;
pub mod bitshift;
pub mod bitwise;
pub mod block_dequant;
pub mod block_quantized_matmul;
pub mod block_quantized_moe;
pub mod cast;
pub mod causal_conv;
pub mod compress;
pub mod compressed_sparse_attention;
pub mod concat;
pub mod constant;
pub mod constant_of_shape;
pub mod contrib_fused;
pub mod dense_elementwise;
pub mod dft;
pub mod dropout;
pub mod elementwise;
pub mod expand;
pub mod eye_like;
pub(crate) mod flops;
pub mod fused_attention;
pub mod fused_gemm;
pub mod fused_matmul_bias;
pub mod gather;
pub mod gather_block_quantized;
pub mod gelu;
pub mod gemm;
pub mod group_query_attention;
mod half_gemm;
// The GEMV is an F16C/AVX2 kernel with no portable body: off x86 the decode
// path is unchanged, so the module is not compiled at all rather than left as
// dead code.
pub mod governed_accumulator_budget;
pub mod governed_weight_cache;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod half_gemv;
pub mod hardmax;
pub mod identity;
pub mod index_share;
pub mod indexing;
pub(crate) mod int4_nibble;
pub mod is_inf;
pub mod is_nan;
pub mod layernorm;
pub mod linear_attention;
pub mod log_softmax;
pub mod logical;
pub mod lp_normalization;
pub mod matmul;
pub mod matmul_nbits;
pub mod moe;
pub mod movement_ops;
pub mod msft_attention;
pub mod multi_head_attention;
pub mod onehot;
pub mod packed_multi_head_attention;
pub mod packed_varlen_attention;
pub mod pad;
pub(crate) mod qgemm_native;
pub mod qlinear_matmul;
pub mod qmoe;
pub mod quantization;
pub mod reduce;
pub mod reduce_ops;
pub mod relu;
pub mod reshape;
pub mod rmsnorm;
pub mod rotary_embedding;
pub mod sdpa;
pub mod selection;
pub mod sequence;
pub mod shape;
pub mod simd_activations;
pub mod simd_normalize;
pub mod simd_quant;
pub mod simd_sumsq;
pub mod skip_simplified_layernorm;
pub mod slice;
pub mod softmax;
pub mod sparse_kv_gather;
pub mod split;
pub mod transpose;
pub mod unary_math;
pub mod unique;
pub mod unsqueeze;
pub mod varlen_attention;
pub mod weight_transpose;
pub mod where_op;
pub mod window;

macro_rules! operator_group_modules {
    ($feature:literal; $($module:ident),+ $(,)?) => {
        $(
            #[cfg(feature = $feature)]
            pub mod $module;
        )+
    };
}

operator_group_modules!(
    "ops-cnn";
    affine_grid,
    center_crop_pad,
    col2im,
    conv_transpose,
    grid_sample,
    norm_ops,
    pooling,
    resize,
    space_to_depth,
);

#[cfg(all(feature = "ops-cnn", feature = "mlas"))]
pub mod conv;
#[cfg(all(feature = "ops-cnn", not(feature = "mlas")))]
#[path = "conv_ref.rs"]
pub mod conv;
#[cfg(all(feature = "ops-cnn", feature = "mlas"))]
pub mod nchwc;

/// The set of ops the CPU EP implements for the Phase-1 BERT-on-CPU milestone.
pub const PHASE1_OPS: &[&str] = &[
    "MatMul",
    "Add",
    "Relu",
    "Reshape",
    "Transpose",
    "Gather",
    "LayerNormalization",
    // Elementwise binary (numpy broadcasting).
    "Sub",
    "Mul",
    "Div",
    "Mod",
    "Pow",
    "Min",
    "Max",
    "Sum",
    "Mean",
    // Elementwise unary.
    "Sqrt",
    "Erf",
    "Tanh",
    "Cast",
    "CastLike",
    // Additional elementwise unary math (unary_math.rs).
    "Abs",
    "Neg",
    "Reciprocal",
    "Exp",
    "Log",
    "Sign",
    "Floor",
    "Ceil",
    "Round",
    "Sin",
    "Cos",
    "Sigmoid",
    "Softplus",
    "Softsign",
    "Acos",
    "Acosh",
    "Asin",
    "Asinh",
    "Atan",
    "Atanh",
    "Cosh",
    "Sinh",
    "Tan",
    "Elu",
    "LeakyRelu",
    "HardSigmoid",
    "Celu",
    "Mish",
    // Logical / selection.
    "And",
    "Or",
    "Xor",
    "Not",
    "BitShift",
    "Equal",
    "Greater",
    "GreaterOrEqual",
    "Less",
    "LessOrEqual",
    "Where",
    // Reduction / normalization.
    "ReduceMean",
    "ReduceSum",
    "ReduceMax",
    "ReduceMin",
    "ReduceProd",
    "ReduceSumSquare",
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "Softmax",
    "LogSoftmax",
    // Shape / data movement.
    "Shape",
    "Unsqueeze",
    "Expand",
    "Slice",
    "Constant",
    "Identity",
    "Concat",
    "Flatten",
    "Squeeze",
    "Split",
    "Unique",
    "Pad",
    "ConstantOfShape",
    "Size",
    "Trilu",
    "GatherElements",
    "GatherND",
    "ScatterElements",
    "ScatterND",
    "OneHot",
    "Compress",
    "Tile",
    "Range",
    "CumSum",
    "Clip",
    "ArgMax",
    "ArgMin",
    "TopK",
    "NonZero",
    "NonMaxSuppression",
    // GEMM.
    "Gemm",
    "QuantizeLinear",
    "DequantizeLinear",
    "DynamicQuantizeLinear",
    "QLinearMatMul",
    "Dropout",
];

/// Whether `op_type` is one of the Phase-1 ops the CPU EP can run.
pub fn is_phase1_op(op_type: &str) -> bool {
    PHASE1_OPS.contains(&op_type)
}

macro_rules! register_operator_group {
    ($function:ident, $feature:literal, |$registry:ident| $body:block) => {
        #[cfg(feature = $feature)]
        fn $function($registry: &mut OpRegistry) $body

        #[cfg(not(feature = $feature))]
        fn $function(_: &mut OpRegistry) {}
    };
}

register_operator_group!(register_cnn_ops, "ops-cnn", |registry| {
    #[cfg(feature = "mlas")]
    {
        registry.register(
            OpKey::new(nchwc::REORDER_TO_BLOCKED_OP, nchwc::NCHWC_DOMAIN, 1),
            Box::new(nchwc::NchwcReorderToBlockedFactory),
        );
        registry.register(
            OpKey::new(nchwc::REORDER_TO_NCHW_OP, nchwc::NCHWC_DOMAIN, 1),
            Box::new(nchwc::NchwcReorderToNchwFactory),
        );
        registry.register(
            OpKey::new(nchwc::NCHWC_CONV_OP, nchwc::NCHWC_DOMAIN, 1),
            Box::new(nchwc::NchwcConvFactory),
        );
        registry.register(
            OpKey::new(nchwc::NCHWC_MAX_POOL_OP, nchwc::NCHWC_DOMAIN, 1),
            Box::new(nchwc::NchwcPoolFactory::max()),
        );
        registry.register(
            OpKey::new(nchwc::NCHWC_AVERAGE_POOL_OP, nchwc::NCHWC_DOMAIN, 1),
            Box::new(nchwc::NchwcPoolFactory::average()),
        );
        registry.register(
            OpKey::new(nchwc::NCHWC_GLOBAL_AVERAGE_POOL_OP, nchwc::NCHWC_DOMAIN, 1),
            Box::new(nchwc::NchwcPoolFactory::global_average()),
        );
    }
    registry.register(
        OpKey::new("GridSample", "", 16),
        Box::new(grid_sample::GridSampleFactory { since_version: 16 }),
    );
    registry.register(
        OpKey::new("GridSample", "", 20),
        Box::new(grid_sample::GridSampleFactory { since_version: 20 }),
    );
    registry.register(
        OpKey::new("Resize", "", 10),
        Box::new(resize::ResizeFactory { since_version: 10 }),
    );
    registry.register(
        OpKey::new("Resize", "", 11),
        Box::new(resize::ResizeFactory { since_version: 11 }),
    );
    registry.register(
        OpKey::new("AffineGrid", "", 20),
        Box::new(affine_grid::AffineGridFactory),
    );
    registry.register(
        OpKey::new("Col2Im", "", 18),
        Box::new(col2im::Col2ImFactory),
    );
    registry.register(
        OpKey::new("ConvTranspose", "", 1),
        Box::new(conv_transpose::ConvTransposeFactory),
    );
    registry.register(OpKey::new("Conv", "", 1), Box::new(conv::ConvFactory));
    registry.register(
        OpKey::new("CenterCropPad", "", 18),
        Box::new(center_crop_pad::CenterCropPadFactory),
    );
    for version in [1, 7, 10, 11, 19] {
        registry.register(
            OpKey::new("AveragePool", "", version),
            Box::new(pooling::AveragePoolFactory),
        );
    }
    for version in [1, 8, 10, 11, 12] {
        registry.register(
            OpKey::new("MaxPool", "", version),
            Box::new(pooling::MaxPoolFactory),
        );
    }
    registry.register(
        OpKey::new("GlobalAveragePool", "", 1),
        Box::new(pooling::GlobalAveragePoolFactory),
    );
    registry.register(
        OpKey::new("GlobalMaxPool", "", 1),
        Box::new(pooling::GlobalMaxPoolFactory),
    );
    registry.register(
        OpKey::new("LpPool", "", 18),
        Box::new(pooling::LpPoolFactory),
    );
    registry.register(
        OpKey::new("GlobalLpPool", "", 2),
        Box::new(pooling::GlobalLpPoolFactory),
    );
    registry.register(
        OpKey::new("SpaceToDepth", "", 13),
        Box::new(space_to_depth::SpaceToDepthFactory),
    );
    // BatchNormalization inference semantics are stable since opset 7, when
    // the legacy `is_test` attribute was removed.
    registry.register(
        OpKey::new("BatchNormalization", "", 7),
        Box::new(norm_ops::BatchNormFactory),
    );
    registry.register(
        OpKey::new("InstanceNormalization", "", 6),
        Box::new(norm_ops::InstanceNormFactory),
    );
    registry.register(
        OpKey::new("GroupNormalization", "", 18),
        Box::new(norm_ops::GroupNormFactory { since_version: 18 }),
    );
    registry.register(
        OpKey::new("GroupNormalization", "", 21),
        Box::new(norm_ops::GroupNormFactory { since_version: 21 }),
    );
    registry.register(
        OpKey::new("PRelu", "", 16),
        Box::new(norm_ops::PReluFactory),
    );
});

/// Build an [`OpRegistry`] populated with every Phase-1 CPU kernel factory.
///
/// The provider consults this to instantiate kernels, and Track D (session) can
/// reuse the same registry for its own placement/lookup. All ops are registered
/// under the default domain (`""`) at `since_version` 1; the registry's
/// `lookup` picks the highest applicable version, so future opset-specialized
/// kernels can be added alongside these.
pub fn build_cpu_registry() -> OpRegistry {
    let (reg, _keys) =
        build_cpu_registry_recorded_inner(qmoe::default_weight_offload_host_cache().clone());
    reg
}

/// Build the CPU registry AND return all registered op keys (for kernel-registry
/// type-constraint advertisement). The keys are derived from the exact same
/// registration calls — not hand-maintained.
pub fn build_cpu_registry_with_descriptors() -> (OpRegistry, Vec<CpuOpDescriptor>) {
    let (reg, _) =
        build_cpu_registry_recorded_inner(qmoe::default_weight_offload_host_cache().clone());
    let descriptors = descriptors_from_registry(&reg);
    (reg, descriptors)
}

/// Derive one [`CpuOpDescriptor`] per entry in `reg`.
///
/// Reads the registry itself rather than a list accumulated during
/// registration. Those two had drifted: `register_cnn_ops` takes `&mut
/// OpRegistry` and so writes past the recording wrapper, which left 18 ops --
/// `PRelu`, `BatchNormalization`, `InstanceNormalization`,
/// `GroupNormalization`, `Conv`, and the pooling and resize family -- present
/// in the registry but absent from the descriptors.
///
/// That gap was not cosmetic. The plugin turns these descriptors into the
/// `KernelRegistryEntry` list, and `node_passes_dtype_filter` fails closed on
/// an op with no entry, so every one of those 18 was claimed by `supports_op`
/// and then dropped at capability time -- handing it to ORT's CPU EP, which is
/// exactly what this EP must never do. Deriving from the registry makes the
/// two sets identical by construction instead of by convention.
fn descriptors_from_registry(reg: &OpRegistry) -> Vec<CpuOpDescriptor> {
    let mut descriptors: Vec<CpuOpDescriptor> = reg
        .keys()
        .map(|key| CpuOpDescriptor {
            op_type: key.op_type.clone(),
            domain: key.domain.clone(),
            since_version: key.since_version,
            supported_dtypes: supported_dtypes_for_op(&key.op_type, &key.domain),
        })
        .collect();
    // `OpRegistry` iterates a hash map, so fix an order: callers leak these
    // into a `'static` slice ORT reads, and an unstable order makes any
    // downstream diff or snapshot test flap.
    descriptors.sort_by(|a, b| {
        (&a.domain, &a.op_type, a.since_version).cmp(&(&b.domain, &b.op_type, b.since_version))
    });
    descriptors
}

/// Build the CPU registry AND return keys, with a custom weight-offload cache.
pub fn build_cpu_registry_with_descriptors_and_cache(
    host_cache: qmoe::WeightOffloadHostCache,
) -> (OpRegistry, Vec<CpuOpDescriptor>) {
    let (reg, _) = build_cpu_registry_recorded_inner(host_cache);
    let descriptors = descriptors_from_registry(&reg);
    (reg, descriptors)
}

pub(crate) fn build_cpu_registry_with_weight_offload_cache(
    host_cache: qmoe::WeightOffloadHostCache,
) -> OpRegistry {
    let (reg, _keys) = build_cpu_registry_recorded_inner(host_cache);
    reg
}

fn build_cpu_registry_recorded_inner(
    host_cache: qmoe::WeightOffloadHostCache,
) -> (OpRegistry, Vec<(String, String, u64)>) {
    let mut rec = RecordingOpRegistry::new();
    // CNN ops go directly into the inner registry (they use &mut OpRegistry).
    register_cnn_ops(&mut rec.inner);
    // CNN ops are feature-gated f32/FLOAT_DTYPES ops. They are registered into
    // the real OpRegistry but not individually recorded here — this is fine for
    // f16/bf16 routing since CNN ops don't support f16/bf16.
    //
    // All subsequent registrations go through the recording wrapper.
    rec.register(OpKey::new("MatMul", "", 1), Box::new(matmul::MatMulFactory));
    rec.register(
        OpKey::new("MatMulNBits", "com.microsoft", 1),
        Box::new(matmul_nbits::MatMulNBitsFactory),
    );
    rec.register(
        OpKey::new("BlockQuantizedMatMul", "pkg.nxrt", 1),
        Box::new(block_quantized_matmul::BlockQuantizedMatMulFactory),
    );
    rec.register(
        OpKey::new("BlockQuantizedMoE", "pkg.nxrt", 1),
        Box::new(block_quantized_moe::BlockQuantizedMoEFactory),
    );
    rec.register(
        OpKey::new("IndexShare", "pkg.nxrt", 1),
        Box::new(index_share::IndexShareFactory),
    );
    rec.register(
        OpKey::new("VarlenAttention", "pkg.nxrt", 1),
        Box::new(varlen_attention::VarlenAttentionFactory),
    );
    rec.register(
        OpKey::new("PackedVarlenAttention", "pkg.nxrt", 1),
        Box::new(packed_varlen_attention::PackedVarlenAttentionFactory),
    );
    rec.register(
        OpKey::new("PackedMultiHeadAttention", "com.microsoft", 1),
        Box::new(packed_multi_head_attention::PackedMultiHeadAttentionFactory),
    );
    rec.register(
        OpKey::new("SparseKvGather", "pkg.nxrt", 1),
        Box::new(sparse_kv_gather::SparseKvGatherFactory),
    );
    rec.register(
        OpKey::new("CompressedSparseAttention", "pkg.nxrt", 1),
        Box::new(compressed_sparse_attention::CompressedSparseAttentionFactory),
    );
    rec.register(OpKey::new("Add", "", 1), Box::new(add::AddFactory));
    rec.register(OpKey::new("Relu", "", 1), Box::new(relu::ReluFactory));
    rec.register(
        OpKey::new("Reshape", "", 1),
        Box::new(reshape::ReshapeFactory),
    );
    rec.register(
        OpKey::new("Transpose", "", 1),
        Box::new(transpose::TransposeFactory),
    );
    rec.register(OpKey::new("Gather", "", 1), Box::new(gather::GatherFactory));
    rec.register(
        OpKey::new("LayerNormalization", "", 1),
        Box::new(layernorm::LayerNormFactory),
    );
    // The optimizer emits fused `LayerNormalization` in the private contrib
    // domain (`com.microsoft`); bind the same kernel there so dispatch resolves
    // the fused op by (domain, op_type). The default-domain registration above
    // still serves standard ONNX `LayerNormalization`.
    rec.register(
        OpKey::new("LayerNormalization", "com.microsoft", 1),
        Box::new(layernorm::LayerNormFactory),
    );
    // The optimizer's `MatMul + Add(bias)` fusion emits `FusedMatMulBias` in the
    // contrib domain; bind its kernel there so dispatch resolves the fused op by
    // (domain, op_type). It reuses the shared MatMul GEMM + broadcast-Add.
    rec.register(
        OpKey::new("FusedMatMulBias", "com.microsoft", 1),
        Box::new(fused_matmul_bias::FusedMatMulBiasFactory),
    );
    // The optimizer's `MatMul + Add(bias) + Relu` fusion emits `FusedGemm` in
    // the contrib domain; bind its kernel there so dispatch resolves the fused
    // op by (domain, op_type). It reuses the shared MatMul GEMM + broadcast-Add
    // + elementwise Relu.
    rec.register(
        OpKey::new("FusedGemm", "com.microsoft", 1),
        Box::new(fused_gemm::FusedGemmFactory),
    );
    // The optimizer's SDPA-core fusion (MatMul(QKᵀ) → scale → [+mask] → Softmax
    // → MatMul(·V)) emits `FusedAttention` in the contrib domain; bind its
    // kernel there so dispatch resolves the fused op by (domain, op_type). It
    // reuses the shared MatMul GEMM (twice), broadcast-Add (mask) and the
    // extracted last-axis softmax helper.
    rec.register(
        OpKey::new("FusedAttention", "com.microsoft", 1),
        Box::new(fused_attention::FusedAttentionFactory),
    );
    rec.register(
        OpKey::new("GroupQueryAttention", "com.microsoft", 1),
        Box::new(group_query_attention::GroupQueryAttentionFactory),
    );
    // `com.microsoft::MultiHeadAttention` (opset 1): scaled dot-product
    // attention with separate Q/K/V inputs (value head size may differ from the
    // query/key head size), an optional Q/K/V bias, key_padding_mask, additive
    // attention_bias, causal (`unidirectional`) masking, and an in-op KV cache
    // (`past_*` → `present_*`). f32 reference kernel matching ORT 1.26.0.
    rec.register(
        OpKey::new("MultiHeadAttention", "com.microsoft", 1),
        Box::new(multi_head_attention::MultiHeadAttentionFactory),
    );
    // `com.microsoft::Attention` (opset 1): the packed-QKV BERT/GPT attention
    // op. Takes the raw hidden state plus a merged Q/K/V projection weight and
    // bias, projects `input @ weights + bias`, splits into Q/K/V, then runs the
    // same SDPA math as MHA. Supports `mask_index` (raw/key-length forms),
    // `attention_bias`, `unidirectional` causal masking, `qkv_hidden_sizes`,
    // an explicit `scale`, and the `past`→`present` KV cache. f32 reference
    // kernel matching ORT 1.26.0; unblocks Whisper's packed-QKV encoder.
    rec.register(
        OpKey::new("Attention", "com.microsoft", 1),
        Box::new(msft_attention::MsftAttentionFactory),
    );
    // `com.microsoft::CausalConvWithState` and `com.microsoft::LinearAttention`:
    // the hybrid linear-attention (Gated DeltaNet) primitives used by Qwen3.5 /
    // Qwen3-Next. Shape-driven and gate-configurable (no model-specific dims).
    rec.register(
        OpKey::new("CausalConvWithState", "com.microsoft", 1),
        Box::new(causal_conv::CausalConvWithStateFactory),
    );
    rec.register(
        OpKey::new("LinearAttention", "com.microsoft", 1),
        Box::new(linear_attention::LinearAttentionFactory),
    );
    // Standard ONNX-domain spelling (onnx/onnx#7689), semantically identical to
    // the com.microsoft op — served by the same fused kernel.
    rec.register(
        OpKey::new("LinearAttention", "", 1),
        Box::new(linear_attention::LinearAttentionFactory),
    );
    // `com.microsoft::GatherBlockQuantized`: block-quantized embedding gather
    // (the Qwen3.5 `embed_tokens` table is uint8 with `bits = 8`). Shape-driven,
    // dequantizes on the fly to the graph's activation dtype.
    rec.register(
        OpKey::new("GatherBlockQuantized", "com.microsoft", 1),
        Box::new(gather_block_quantized::GatherBlockQuantizedFactory),
    );
    // Standard `ai.onnx::Attention`: the richer SDPA op with 3D/4D inputs,
    // GQA/MQA head sharing, a KV cache (`past_*`/`present_*`), causal masking,
    // softcap, and up to four outputs. Distinct from the contrib
    // `FusedAttention` above. Added at opset 23 and revised at opset 24; since
    // no newer version exists, the opset-24 kernel serves model opsets 24, 25
    // and 26 (the registry resolves the highest `since_version <= opset`). Both
    // versions are registered so opset-23 models keep the original
    // `qk_matmul_output_mode` 1↔2 ordering while opset-24+ models get the
    // swapped ordering and `nonpad_kv_seqlen` support.
    rec.register(
        OpKey::new("Attention", "", 23),
        Box::new(attention::AttentionFactory { since_version: 23 }),
    );
    rec.register(
        OpKey::new("Attention", "", 24),
        Box::new(attention::AttentionFactory { since_version: 24 }),
    );
    // The optimizer's exact-GELU fusion emits `com.microsoft::Gelu`; bind its
    // CPU kernel in the same contrib domain (there is no standard-domain `Gelu`
    // op, so it is registered only under `com.microsoft`).
    rec.register(
        OpKey::new("Gelu", "com.microsoft", 1),
        Box::new(gelu::GeluFactory),
    );
    rec.register(
        OpKey::new("BiasGelu", "com.microsoft", 1),
        Box::new(contrib_fused::BiasGeluFactory),
    );
    rec.register(
        OpKey::new("FastGelu", "com.microsoft", 1),
        Box::new(contrib_fused::FastGeluFactory),
    );
    rec.register(
        OpKey::new("QuickGelu", "com.microsoft", 1),
        Box::new(contrib_fused::QuickGeluFactory),
    );
    rec.register(
        OpKey::new("Silu", "com.microsoft", 1),
        Box::new(activations::SiluFactory),
    );
    rec.register(
        OpKey::new("SkipLayerNormalization", "com.microsoft", 1),
        Box::new(contrib_fused::SkipLayerNormFactory),
    );
    rec.register(
        OpKey::new("SimplifiedLayerNormalization", "com.microsoft", 1),
        Box::new(contrib_fused::SimplifiedLayerNormFactory),
    );
    rec.register(
        OpKey::new("SimplifiedLayerNormalization", "", 1),
        Box::new(contrib_fused::SimplifiedLayerNormFactory),
    );
    rec.register(
        OpKey::new("SkipSimplifiedLayerNormalization", "com.microsoft", 1),
        Box::new(skip_simplified_layernorm::SkipSimplifiedLayerNormFactory),
    );
    rec.register(
        OpKey::new("MoE", "com.microsoft", 1),
        Box::new(moe::MoEFactory),
    );
    rec.register(
        OpKey::new("QMoE", "com.microsoft", 1),
        Box::new(qmoe::QMoEFactory::new(host_cache)),
    );
    // Standard-domain LLM/transformer primitives (ai.onnx). Registered at their
    // ONNX since_version; the registry resolves the highest since_version <=
    // model opset.
    //
    // `ai.onnx::Gelu` was added at opset 20 with the `approximate` attribute
    // ("none" = exact erf, "tanh" = tanh approximation). Distinct from the
    // com.microsoft::Gelu contrib op above.
    rec.register(OpKey::new("Gelu", "", 20), Box::new(gelu::StdGeluFactory));
    // `ai.onnx::RMSNormalization` added at opset 23.
    rec.register(
        OpKey::new("RMSNormalization", "", 23),
        Box::new(rmsnorm::RmsNormFactory),
    );
    rec.register(
        OpKey::new("LpNormalization", "", 1),
        Box::new(lp_normalization::LpNormalizationFactory),
    );
    // `ai.onnx::RotaryEmbedding` added at opset 23.
    rec.register(
        OpKey::new("RotaryEmbedding", "", 23),
        Box::new(rotary_embedding::RotaryEmbeddingFactory),
    );
    // `com.microsoft::RotaryEmbedding` contrib op: same rotation math, but the
    // inputs are ordered `(X, position_ids, cos_cache, sin_cache)`.
    rec.register(
        OpKey::new("RotaryEmbedding", "com.microsoft", 1),
        Box::new(rotary_embedding::RotaryEmbeddingContribFactory),
    );
    // `ai.onnx::Swish` added at opset 24: y = x·sigmoid(alpha·x).
    rec.register(
        OpKey::new("Swish", "", 24),
        Box::new(activations::SwishFactory),
    );
    // Elementwise binary broadcasting ops.
    rec.register(OpKey::new("Sub", "", 1), Box::new(elementwise::SubFactory));
    rec.register(OpKey::new("Mul", "", 1), Box::new(elementwise::MulFactory));
    rec.register(OpKey::new("Div", "", 1), Box::new(elementwise::DivFactory));
    rec.register(OpKey::new("Mod", "", 10), Box::new(elementwise::ModFactory));
    rec.register(OpKey::new("Pow", "", 1), Box::new(elementwise::PowFactory));
    rec.register(OpKey::new("IsInf", "", 10), Box::new(is_inf::IsInfFactory));
    rec.register(OpKey::new("IsNaN", "", 9), Box::new(is_nan::IsNaNFactory));
    rec.register(
        OpKey::new("EyeLike", "", 9),
        Box::new(eye_like::EyeLikeFactory),
    );
    rec.register(OpKey::new("Min", "", 1), Box::new(elementwise::MinFactory));
    rec.register(OpKey::new("Max", "", 1), Box::new(elementwise::MaxFactory));
    rec.register(OpKey::new("Sum", "", 1), Box::new(elementwise::SumFactory));
    rec.register(
        OpKey::new("Mean", "", 1),
        Box::new(elementwise::MeanFactory),
    );
    // Elementwise unary ops.
    rec.register(
        OpKey::new("Sqrt", "", 1),
        Box::new(elementwise::SqrtFactory),
    );
    rec.register(OpKey::new("Erf", "", 1), Box::new(elementwise::ErfFactory));
    rec.register(
        OpKey::new("Tanh", "", 1),
        Box::new(elementwise::TanhFactory),
    );
    rec.register(OpKey::new("Cast", "", 1), Box::new(cast::CastFactory));
    rec.register(
        OpKey::new("CastLike", "", 15),
        Box::new(cast::CastLikeFactory),
    );
    // Identity: dtype-agnostic passthrough (raw byte copy).
    rec.register(
        OpKey::new("Identity", "", 1),
        Box::new(identity::IdentityFactory),
    );
    rec.register(
        OpKey::new("ReduceMean", "", 1),
        Box::new(reduce::ReduceMeanFactory),
    );
    // Softmax: legacy coerce-to-2D at opset ≤ 12, per-axis at opset ≥ 13. The
    // provider's opset-aware lookup selects the version-correct kernel.
    rec.register(
        OpKey::new("Softmax", "", 1),
        Box::new(softmax::SoftmaxLegacyFactory),
    );
    rec.register(
        OpKey::new("Softmax", "", 13),
        Box::new(softmax::SoftmaxFactory),
    );
    // LogSoftmax shares Softmax's opset split: legacy flattened trailing axes
    // through opset 12, then one-axis normalization from opset 13.
    rec.register(
        OpKey::new("LogSoftmax", "", 1),
        Box::new(log_softmax::LogSoftmaxLegacyFactory),
    );
    rec.register(
        OpKey::new("LogSoftmax", "", 13),
        Box::new(log_softmax::LogSoftmaxFactory),
    );
    // Shape / data movement.
    rec.register(OpKey::new("Shape", "", 1), Box::new(shape::ShapeFactory));
    rec.register(
        OpKey::new("Unsqueeze", "", 1),
        Box::new(unsqueeze::UnsqueezeFactory),
    );
    rec.register(OpKey::new("Expand", "", 1), Box::new(expand::ExpandFactory));
    rec.register(OpKey::new("Slice", "", 1), Box::new(slice::SliceFactory));
    rec.register(OpKey::new("Split", "", 1), Box::new(split::SplitFactory));
    rec.register(OpKey::new("Split", "", 18), Box::new(split::SplitFactory));
    rec.register(
        OpKey::new("Unique", "", 11),
        Box::new(unique::UniqueFactory),
    );
    rec.register(
        OpKey::new("Dropout", "", 13),
        Box::new(dropout::DropoutFactory),
    );
    rec.register(
        OpKey::new("Dropout", "", 22),
        Box::new(dropout::DropoutFactory),
    );
    rec.register(OpKey::new("Pad", "", 1), Box::new(pad::PadFactory));
    rec.register(
        OpKey::new("ConstantOfShape", "", 1),
        Box::new(constant_of_shape::ConstantOfShapeFactory),
    );
    rec.register(
        OpKey::new("Constant", "", 1),
        Box::new(constant::ConstantFactory),
    );
    // GEMM.
    rec.register(OpKey::new("Gemm", "", 1), Box::new(gemm::GemmFactory));
    // Linear quantization evolved at opsets 10, 13, 19, 21, 23, and 25. The
    // implementation accepts the newest parameter set for all these revisions.
    for version in [10, 13, 19, 21, 23, 25] {
        rec.register(
            OpKey::new("QuantizeLinear", "", version),
            Box::new(quantization::QuantizeLinearFactory),
        );
        rec.register(
            OpKey::new("DequantizeLinear", "", version),
            Box::new(quantization::DequantizeLinearFactory),
        );
    }
    rec.register(
        OpKey::new("DynamicQuantizeLinear", "", 11),
        Box::new(quantization::DynamicQuantizeLinearFactory),
    );
    rec.register(
        OpKey::new("QLinearMatMul", "", 10),
        Box::new(qlinear_matmul::QLinearMatMulFactory),
    );
    // --- Additional ep-cpu op coverage (op-coverage wave) ---------------------
    // Elementwise unary math (f32). Additive, default-domain-only registrations.
    rec.register(OpKey::new("Abs", "", 1), Box::new(unary_math::AbsFactory));
    rec.register(OpKey::new("Neg", "", 1), Box::new(unary_math::NegFactory));
    rec.register(
        OpKey::new("Reciprocal", "", 1),
        Box::new(unary_math::ReciprocalFactory),
    );
    rec.register(OpKey::new("Exp", "", 1), Box::new(unary_math::ExpFactory));
    rec.register(OpKey::new("Log", "", 1), Box::new(unary_math::LogFactory));
    rec.register(OpKey::new("Sign", "", 1), Box::new(unary_math::SignFactory));
    rec.register(
        OpKey::new("Floor", "", 1),
        Box::new(unary_math::FloorFactory),
    );
    rec.register(OpKey::new("Ceil", "", 1), Box::new(unary_math::CeilFactory));
    rec.register(
        OpKey::new("Round", "", 1),
        Box::new(unary_math::RoundFactory),
    );
    rec.register(OpKey::new("Sin", "", 1), Box::new(unary_math::SinFactory));
    rec.register(OpKey::new("Cos", "", 1), Box::new(unary_math::CosFactory));
    rec.register(
        OpKey::new("Sigmoid", "", 1),
        Box::new(unary_math::SigmoidFactory),
    );
    rec.register(
        OpKey::new("Softplus", "", 1),
        Box::new(unary_math::SoftplusFactory),
    );
    rec.register(
        OpKey::new("Softsign", "", 1),
        Box::new(unary_math::SoftsignFactory),
    );
    rec.register(OpKey::new("Acos", "", 1), Box::new(unary_math::AcosFactory));
    rec.register(
        OpKey::new("Acosh", "", 1),
        Box::new(unary_math::AcoshFactory),
    );
    rec.register(OpKey::new("Asin", "", 1), Box::new(unary_math::AsinFactory));
    rec.register(
        OpKey::new("Asinh", "", 1),
        Box::new(unary_math::AsinhFactory),
    );
    rec.register(OpKey::new("Atan", "", 1), Box::new(unary_math::AtanFactory));
    rec.register(
        OpKey::new("Atanh", "", 1),
        Box::new(unary_math::AtanhFactory),
    );
    rec.register(OpKey::new("Cosh", "", 1), Box::new(unary_math::CoshFactory));
    rec.register(OpKey::new("Sinh", "", 1), Box::new(unary_math::SinhFactory));
    rec.register(OpKey::new("Tan", "", 1), Box::new(unary_math::TanFactory));
    rec.register(OpKey::new("Elu", "", 1), Box::new(activations::EluFactory));
    // Celu is opset 12, Mish opset 18 -- registering at their introducing
    // opset keeps older models routing to whatever handled them before.
    rec.register(
        OpKey::new("Celu", "", 12),
        Box::new(activations::CeluFactory),
    );
    rec.register(
        OpKey::new("Mish", "", 18),
        Box::new(activations::MishFactory),
    );
    rec.register(
        OpKey::new("LeakyRelu", "", 1),
        Box::new(activations::LeakyReluFactory),
    );
    rec.register(
        OpKey::new("HardSigmoid", "", 1),
        Box::new(activations::HardSigmoidFactory),
    );
    rec.register(
        OpKey::new("Selu", "", 6),
        Box::new(activations::SeluFactory),
    );
    rec.register(
        OpKey::new("ThresholdedRelu", "", 10),
        Box::new(activations::ThresholdedReluFactory),
    );
    // Logical / selection.
    rec.register(OpKey::new("And", "", 7), Box::new(logical::AndFactory));
    rec.register(OpKey::new("Or", "", 7), Box::new(logical::OrFactory));
    rec.register(OpKey::new("Xor", "", 7), Box::new(logical::XorFactory));
    rec.register(OpKey::new("Not", "", 1), Box::new(logical::NotFactory));
    rec.register(OpKey::new("Equal", "", 1), Box::new(logical::EqualFactory));
    rec.register(
        OpKey::new("Greater", "", 1),
        Box::new(logical::GreaterFactory),
    );
    rec.register(
        OpKey::new("GreaterOrEqual", "", 1),
        Box::new(logical::GreaterOrEqualFactory),
    );
    rec.register(OpKey::new("Less", "", 1), Box::new(logical::LessFactory));
    rec.register(
        OpKey::new("LessOrEqual", "", 1),
        Box::new(logical::LessOrEqualFactory),
    );
    rec.register(OpKey::new("Where", "", 1), Box::new(where_op::WhereFactory));
    // Reductions (axes attribute or opset-13/18 axes input).
    rec.register(
        OpKey::new("ReduceSum", "", 1),
        Box::new(reduce_ops::ReduceSumFactory),
    );
    rec.register(
        OpKey::new("ReduceMax", "", 1),
        Box::new(reduce_ops::ReduceMaxFactory),
    );
    rec.register(
        OpKey::new("ReduceMin", "", 1),
        Box::new(reduce_ops::ReduceMinFactory),
    );
    rec.register(
        OpKey::new("ReduceProd", "", 1),
        Box::new(reduce_ops::ReduceProdFactory),
    );
    rec.register(
        OpKey::new("ReduceSumSquare", "", 1),
        Box::new(reduce_ops::ReduceSumSquareFactory),
    );
    rec.register(
        OpKey::new("ReduceL1", "", 1),
        Box::new(reduce_ops::ReduceL1Factory),
    );
    rec.register(
        OpKey::new("ReduceL2", "", 1),
        Box::new(reduce_ops::ReduceL2Factory),
    );
    rec.register(
        OpKey::new("ReduceLogSum", "", 1),
        Box::new(reduce_ops::ReduceLogSumFactory),
    );
    rec.register(
        OpKey::new("ReduceLogSumExp", "", 1),
        Box::new(reduce_ops::ReduceLogSumExpFactory),
    );
    rec.register(
        OpKey::new("ReduceLogSumExp", "", 18),
        Box::new(reduce_ops::ReduceLogSumExpFactory),
    );
    // Shape / data movement (dtype-agnostic byte movers).
    rec.register(OpKey::new("Concat", "", 1), Box::new(concat::ConcatFactory));
    rec.register(
        OpKey::new("Flatten", "", 1),
        Box::new(movement_ops::FlattenFactory),
    );
    rec.register(
        OpKey::new("Squeeze", "", 1),
        Box::new(movement_ops::SqueezeFactory),
    );
    rec.register(
        OpKey::new("Size", "", 1),
        Box::new(movement_ops::SizeFactory),
    );
    rec.register(
        OpKey::new("Trilu", "", 14),
        Box::new(movement_ops::TriluFactory),
    );
    // Indexed data movement and sequence construction.
    rec.register(
        OpKey::new("GatherElements", "", 11),
        Box::new(indexing::GatherElementsFactory),
    );
    rec.register(
        OpKey::new("GatherND", "", 11),
        Box::new(indexing::GatherNDFactory),
    );
    // ScatterElements gained its reduction attribute at opset 16.
    rec.register(
        OpKey::new("ScatterElements", "", 11),
        Box::new(indexing::ScatterElementsFactory),
    );
    rec.register(
        OpKey::new("ScatterElements", "", 16),
        Box::new(indexing::ScatterElementsFactory),
    );
    // ScatterND gained reduction at opset 16 and max/min reductions at opset 18.
    for version in [11, 16, 18] {
        rec.register(
            OpKey::new("ScatterND", "", version),
            Box::new(indexing::ScatterNDFactory),
        );
    }
    rec.register(
        OpKey::new("OneHot", "", 9),
        Box::new(indexing::OneHotFactory),
    );
    rec.register(
        OpKey::new("OneHot", "", 11),
        Box::new(onehot::OneHotFactory),
    );
    rec.register(
        OpKey::new("BitShift", "", 11),
        Box::new(bitshift::BitShiftFactory),
    );
    rec.register(
        OpKey::new("Compress", "", 11),
        Box::new(compress::CompressFactory),
    );
    rec.register(OpKey::new("Tile", "", 6), Box::new(sequence::TileFactory));
    rec.register(
        OpKey::new("Range", "", 11),
        Box::new(sequence::RangeFactory),
    );
    rec.register(
        OpKey::new("CumSum", "", 14),
        Box::new(sequence::CumSumFactory),
    );
    rec.register(
        OpKey::new("CumProd", "", 26),
        Box::new(sequence::CumProdFactory),
    );
    rec.register(
        OpKey::new("HannWindow", "", 17),
        Box::new(window::HannWindowFactory),
    );
    rec.register(
        OpKey::new("HammingWindow", "", 17),
        Box::new(window::HammingWindowFactory),
    );
    rec.register(
        OpKey::new("BlackmanWindow", "", 17),
        Box::new(window::BlackmanWindowFactory),
    );
    rec.register(OpKey::new("DFT", "", 17), Box::new(dft::DftFactory));
    rec.register(
        OpKey::new("BitwiseAnd", "", 18),
        Box::new(bitwise::BitwiseAndFactory),
    );
    rec.register(
        OpKey::new("BitwiseOr", "", 18),
        Box::new(bitwise::BitwiseOrFactory),
    );
    rec.register(
        OpKey::new("BitwiseXor", "", 18),
        Box::new(bitwise::BitwiseXorFactory),
    );
    rec.register(
        OpKey::new("BitwiseNot", "", 18),
        Box::new(bitwise::BitwiseNotFactory),
    );
    // Value selection.
    rec.register(OpKey::new("Clip", "", 1), Box::new(selection::ClipFactory));
    rec.register(
        OpKey::new("ArgMax", "", 1),
        Box::new(selection::ArgMaxFactory),
    );
    rec.register(
        OpKey::new("ArgMin", "", 1),
        Box::new(selection::ArgMinFactory),
    );
    rec.register(OpKey::new("TopK", "", 10), Box::new(selection::TopKFactory));
    rec.register(
        OpKey::new("NonMaxSuppression", "", 10),
        Box::new(selection::NonMaxSuppressionFactory),
    );
    rec.register(
        OpKey::new("NonZero", "", 9),
        Box::new(selection::NonZeroFactory),
    );
    rec.register(
        OpKey::new("Hardmax", "", 13),
        Box::new(hardmax::HardmaxFactory),
    );
    rec.into_parts()
}

// ---------------------------------------------------------------------------
// Shared view accessors — the only `unsafe` in the kernel layer.
// ---------------------------------------------------------------------------

/// Materialize an `f32` view into a dense, row-major `Vec<f32>`, applying the
/// view's strides and byte offset. Rejects non-`Float32` views.
pub fn to_dense_f32(view: &TensorView) -> Result<Vec<f32>> {
    view.validate()?;
    require_dtype(view.dtype, DataType::Float32, "f32 kernel input")?;
    let n = numel(view.shape);
    let origin = view.data_ptr::<f32>();
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return Ok(out);
    }
    if view.is_contiguous() {
        // Contiguous row-major: the element origin addresses `n` consecutive
        // f32, so a single bulk copy replaces the per-element strided walk. This
        // is the common decode/prefill case (dense activations) and the strided
        // loop below is a serial, non-vectorized per-element hot path otherwise.
        //
        // SAFETY: identical read assumptions to the strided loop below -- the
        // validated host-accessible view describes `n` readable, contiguous f32
        // starting at `origin`, bounds-checked against the backing allocation by
        // the owning EP (ep-api safety invariant #1). `f32` has no invalid bit
        // patterns.
        let slice = unsafe { std::slice::from_raw_parts(origin, n) };
        out.extend_from_slice(slice);
        return Ok(out);
    }
    let mut idx = vec![0usize; view.shape.len()];
    loop {
        let off = elem_offset(view.strides, &idx);
        // SAFETY: `origin` is the element origin of a validated view; `off` is
        // an in-shape element offset (each index component is `< shape[d]`), so
        // the address lies within the range the view describes. The owning EP
        // has already checked that range against the backing allocation via
        // `strided::view_in_bounds` (ep-api safety invariant #1). We never read
        // past the addressed extent, and `f32` has no invalid bit patterns.
        out.push(unsafe { *origin.offset(off) });
        if !next_index(view.shape, &mut idx) {
            break;
        }
    }
    Ok(out)
}

/// Borrow a contiguous host `Uint8` tensor without materializing it.
pub(crate) fn contiguous_u8_slice<'a>(view: &'a TensorView<'_>) -> Result<&'a [u8]> {
    view.validate()?;
    require_dtype(view.dtype, DataType::Uint8, "u8 kernel input")?;
    if !view.device.is_host_accessible() || !view.is_contiguous() {
        return Err(EpError::InvalidTensorView {
            reason: "direct u8 slice requires a contiguous host-accessible tensor".into(),
        });
    }
    let elements = view
        .shape
        .iter()
        .try_fold(1usize, |count, &dim| count.checked_mul(dim));
    let len = elements.ok_or_else(|| EpError::InvalidTensorView {
        reason: "direct u8 slice element count overflow".into(),
    })?;
    if len > isize::MAX as usize {
        return Err(EpError::InvalidTensorView {
            reason: "direct u8 slice exceeds isize::MAX".into(),
        });
    }
    // SAFETY: the validated host-accessible contiguous view describes `len`
    // readable bytes from its element origin; the owning EP bounds-checks the
    // view against its allocation before kernel dispatch.
    Ok(unsafe { std::slice::from_raw_parts(view.data_ptr::<u8>(), len) })
}

/// Borrow a contiguous host `Float32` tensor without materializing it.
pub(crate) fn contiguous_f32_slice<'a>(view: &'a TensorView<'_>) -> Result<&'a [f32]> {
    view.validate()?;
    require_dtype(view.dtype, DataType::Float32, "f32 kernel input")?;
    if !view.device.is_host_accessible() || !view.is_contiguous() {
        return Err(EpError::InvalidTensorView {
            reason: "direct f32 slice requires a contiguous host-accessible tensor".into(),
        });
    }
    let elements = view
        .shape
        .iter()
        .try_fold(1usize, |count, &dim| count.checked_mul(dim));
    let len = elements.ok_or_else(|| EpError::InvalidTensorView {
        reason: "direct f32 slice element count overflow".into(),
    })?;
    len.checked_mul(std::mem::size_of::<f32>())
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| EpError::InvalidTensorView {
            reason: "direct f32 slice byte count overflow or exceeds isize::MAX".into(),
        })?;
    // SAFETY: as above, with `Float32` alignment additionally checked by
    // `TensorView::validate`.
    Ok(unsafe { std::slice::from_raw_parts(view.data_ptr::<f32>(), len) })
}

/// Materialize an integer index view (`Int64` or `Int32`) into a dense
/// `Vec<i64>`. Used for `Gather` indices.
pub fn to_dense_i64(view: &TensorView) -> Result<Vec<i64>> {
    view.validate()?;
    let n = numel(view.shape);
    let mut out = Vec::with_capacity(n);
    if n == 0 {
        return Ok(out);
    }
    let mut idx = vec![0usize; view.shape.len()];
    match view.dtype {
        DataType::Int64 => {
            let origin = view.data_ptr::<i64>();
            loop {
                let off = elem_offset(view.strides, &idx);
                // SAFETY: see `to_dense_f32` — in-shape offset over a validated,
                // bounds-checked view; `i64` has no invalid bit patterns.
                out.push(unsafe { *origin.offset(off) });
                if !next_index(view.shape, &mut idx) {
                    break;
                }
            }
        }
        DataType::Int32 => {
            let origin = view.data_ptr::<i32>();
            loop {
                let off = elem_offset(view.strides, &idx);
                // SAFETY: as above, for a 4-byte element type.
                out.push(unsafe { *origin.offset(off) } as i64);
                if !next_index(view.shape, &mut idx) {
                    break;
                }
            }
        }
        other => {
            return Err(EpError::InvalidTensorView {
                reason: format!("index tensor must be Int64 or Int32, got {other:?}"),
            });
        }
    }
    Ok(out)
}

/// Write a dense, row-major `f32` slice into `out`, applying the output view's
/// strides and byte offset. `data.len()` must equal the output element count.
pub fn write_dense_f32(out: &mut TensorMut, data: &[f32]) -> Result<()> {
    out.validate()?;
    require_dtype(out.dtype, DataType::Float32, "f32 kernel output")?;
    let n = numel(out.shape);
    if data.len() != n {
        return Err(EpError::KernelFailed(format!(
            "output element count {n} does not match produced {}",
            data.len()
        )));
    }
    if n == 0 {
        return Ok(());
    }
    let origin = out.data_ptr_mut::<f32>();
    if out.is_contiguous() {
        // Contiguous row-major output: write `n` consecutive f32 in one bulk
        // copy instead of the per-element strided walk. This is the common
        // decode/prefill case and keeps the write off the serial hot path.
        //
        // SAFETY: identical write assumptions to the strided loop below -- the
        // validated host-accessible output describes `n` writable, contiguous
        // f32 starting at `origin`, bounds-checked against the backing
        // allocation by the owning EP (ep-api safety invariant #1). `data.len()`
        // was checked to equal `n` above, so every element is written once.
        let slice = unsafe { std::slice::from_raw_parts_mut(origin, n) };
        slice.copy_from_slice(data);
        return Ok(());
    }
    let strides = out.strides;
    let shape = out.shape;
    let mut idx = vec![0usize; shape.len()];
    let mut i = 0usize;
    loop {
        let off = elem_offset(strides, &idx);
        // SAFETY: `origin` is the element origin of a validated output view;
        // `off` is an in-shape offset, so it lies within the extent the view
        // describes (bounds-checked against the backing allocation by the EP
        // per invariant #1). Each address is written exactly once because the
        // row-major walk visits every logical index once.
        unsafe {
            *origin.offset(off) = data[i];
        }
        i += 1;
        if !next_index(shape, &mut idx) {
            break;
        }
    }
    Ok(())
}

/// The fixed element byte-width of `dtype`. Errors for variable-width
/// ([`DataType::String`]) and sub-byte-packed (`Int4`/`Uint4`) types, which the
/// dtype-generic byte movers below cannot address one-element-at-a-time.
pub fn elem_size(dtype: DataType) -> Result<usize> {
    let size = dtype.byte_size();
    if size == 0 {
        return Err(EpError::InvalidTensorView {
            reason: format!("dtype {dtype:?} has no fixed-width byte layout"),
        });
    }
    Ok(size)
}

/// Materialize any fixed-width view into a dense, row-major byte buffer,
/// applying the view's strides and byte offset. This is the dtype-agnostic
/// counterpart to [`to_dense_f32`]: it copies raw element bytes without
/// interpreting them, so it serves the pure data-movement ops (Unsqueeze,
/// Expand, Slice, Cast source read) uniformly across dtypes.
pub fn to_dense_bytes(view: &TensorView) -> Result<Vec<u8>> {
    view.validate()?;
    let esize = elem_size(view.dtype)?;
    let n = numel(view.shape);
    let mut out = vec![0u8; n * esize];
    if n == 0 {
        return Ok(out);
    }
    // Byte origin of the element at logical index 0 (applies `byte_offset`).
    let origin = view.data_ptr::<u8>();
    if view.is_contiguous() {
        // Contiguous row-major: the byte origin addresses `n * esize`
        // consecutive bytes, so one bulk copy replaces the per-element strided
        // walk. The walk below costs an `elem_offset` dot product, a
        // one-element `copy_nonoverlapping`, and a `next_index` carry chain per
        // element, which dominates whole-tensor reads of large weights -- a
        // 3584x3584 `Uint8` operand is 12.8M iterations of that loop. This
        // mirrors the fast paths `to_dense_f32` and `write_dense_bytes` already
        // have; `to_dense_bytes` was the one whole-tensor mover missing it.
        //
        // SAFETY: identical read assumptions to the strided loop below -- the
        // validated view describes `n * esize` readable, contiguous bytes
        // starting at `origin`, bounds-checked against the backing allocation by
        // the owning EP (ep-api safety invariant #1). `u8` has no invalid bit
        // patterns and `out` is a fresh, uniquely-owned buffer of the same
        // length, so the regions cannot overlap.
        let src = unsafe { std::slice::from_raw_parts(origin, n * esize) };
        out.copy_from_slice(src);
        return Ok(out);
    }
    let mut idx = vec![0usize; view.shape.len()];
    let mut w = 0usize;
    loop {
        let elem_off = elem_offset(view.strides, &idx);
        let byte_off = elem_off * esize as isize;
        // SAFETY: `origin` is the byte origin of a validated view; `elem_off` is
        // an in-shape element offset, so `byte_off .. byte_off + esize` lies
        // within the extent the view describes (bounds-checked against the
        // backing allocation by the EP per invariant #1). `out[w..w + esize]` is
        // a fresh, uniquely-owned buffer. The regions do not overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(origin.offset(byte_off), out.as_mut_ptr().add(w), esize);
        }
        w += esize;
        if !next_index(view.shape, &mut idx) {
            break;
        }
    }
    Ok(out)
}

/// Write a dense, row-major byte buffer into `out`, applying the output view's
/// strides and byte offset. `data.len()` must equal `numel(out) * elem_size`.
/// The dtype-agnostic counterpart to [`write_dense_f32`].
pub fn write_dense_bytes(out: &mut TensorMut, data: &[u8]) -> Result<()> {
    out.validate()?;
    let esize = elem_size(out.dtype)?;
    let n = numel(out.shape);
    if data.len() != n * esize {
        return Err(EpError::KernelFailed(format!(
            "output byte count {} does not match produced {}",
            n * esize,
            data.len()
        )));
    }
    if n == 0 {
        return Ok(());
    }
    let origin = out.data_ptr_mut::<u8>();
    if out.is_contiguous() {
        // Contiguous row-major output: write `n * esize` consecutive bytes in
        // one bulk copy instead of the per-element strided walk below, which
        // costs an `elem_offset` dot product, a one-element
        // `copy_nonoverlapping`, and a `next_index` carry chain per element.
        // Counterpart to the fast path in `to_dense_bytes`.
        //
        // SAFETY: identical write assumptions to the strided loop below -- the
        // validated output view describes `n * esize` writable, contiguous bytes
        // starting at `origin`, bounds-checked against the backing allocation by
        // the owning EP (ep-api safety invariant #1). `data.len()` was checked to
        // equal `n * esize` above, so every byte is written exactly once, and
        // `data` is a caller-owned buffer distinct from the output allocation.
        let dst = unsafe { std::slice::from_raw_parts_mut(origin, n * esize) };
        dst.copy_from_slice(data);
        return Ok(());
    }
    let strides = out.strides;
    let shape = out.shape;
    let mut idx = vec![0usize; shape.len()];
    let mut r = 0usize;
    loop {
        let elem_off = elem_offset(strides, &idx);
        let byte_off = elem_off * esize as isize;
        // SAFETY: `origin` is the byte origin of a validated output view;
        // `byte_off .. byte_off + esize` is an in-shape offset lying within the
        // extent the view describes (bounds-checked by the EP per invariant #1).
        // Each destination range is written exactly once because the row-major
        // walk visits every logical index once; source and destination buffers
        // are distinct.
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr().add(r), origin.offset(byte_off), esize);
        }
        r += esize;
        if !next_index(shape, &mut idx) {
            break;
        }
    }
    Ok(())
}

/// Error out unless `got == want`.
fn require_dtype(got: DataType, want: DataType, ctx: &str) -> Result<()> {
    if got != want {
        return Err(EpError::InvalidTensorView {
            reason: format!("{ctx} requires {want:?}, got {got:?}"),
        });
    }
    Ok(())
}

/// Validate the arity of a kernel's input/output slices.
fn check_arity(
    op: &str,
    inputs: &[TensorView],
    outputs: &[TensorMut],
    min_inputs: usize,
    max_inputs: usize,
    outputs_wanted: usize,
) -> Result<()> {
    if inputs.len() < min_inputs || inputs.len() > max_inputs {
        return Err(EpError::KernelFailed(format!(
            "{op}: expected {min_inputs}..={max_inputs} inputs, got {}",
            inputs.len()
        )));
    }
    if outputs.len() < outputs_wanted {
        return Err(EpError::KernelFailed(format!(
            "{op}: expected at least {outputs_wanted} output(s), got {}",
            outputs.len()
        )));
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod testutil {
    //! Helpers to build owning-buffer-backed views for kernel unit tests.

    use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, TensorMut, TensorView};
    use onnx_runtime_ir::{DataType, DeviceId, compute_contiguous_strides};

    /// A dense f32 buffer plus the shape/stride metadata a view needs.
    pub struct Owned {
        pub bytes: Vec<u8>,
        pub shape: Vec<usize>,
        pub strides: Vec<i64>,
        pub dtype: DataType,
    }

    /// Dropping this buffer's weight-transpose entries is the test-side
    /// counterpart of the Executor-drop eviction in `onnx-runtime-session`
    /// (#1731).
    ///
    /// The transpose caches are keyed on `(weight address, K, N, tag)`, which
    /// carries no link to the buffer's lifetime. Production closes that window
    /// by evicting when an Executor drops; a test binary has no Executor, frees
    /// these buffers constantly, and readily hands the same block to the next
    /// same-shaped weight. The key then *matches*, the lookup hits, and the
    /// kernel silently multiplies by a previous test's matrix -- a
    /// deterministic wrong answer that surfaced as a ~50% flake in
    /// `gemm_and_matmul_take_the_same_decode_route` and
    /// `f16_decode_at_the_retired_weight_gate_keeps_the_gemv`.
    ///
    /// Eviction is scoped to *this* buffer's address so a live weight's entry
    /// is never discarded -- the cache-accounting tests assert that an executed
    /// kernel's transpose is still resident.
    ///
    /// `bytes.as_ptr()` is the right address because every weight the caches
    /// admit is a contiguous view at `byte_offset` 0, so `view().data_ptr()` --
    /// which is what the key is built from -- is this same pointer. A weight
    /// cached through an offset or sub-slice view would key on `base + offset`
    /// and slip past this; nothing does so today, and the f16 path admits only
    /// contiguous views.
    impl Drop for Owned {
        fn drop(&mut self) {
            crate::kernels::weight_transpose::evict_address(self.bytes.as_ptr() as usize);
        }
    }

    impl Owned {
        pub fn f32(shape: &[usize], data: &[f32]) -> Self {
            let strides = compute_contiguous_strides(shape);
            let mut bytes = Vec::with_capacity(data.len() * 4);
            for v in data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            Self {
                bytes,
                shape: shape.to_vec(),
                strides,
                dtype: DataType::Float32,
            }
        }

        pub fn f64(shape: &[usize], data: &[f64]) -> Self {
            let strides = compute_contiguous_strides(shape);
            let mut bytes = Vec::with_capacity(data.len() * 8);
            for v in data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            Self {
                bytes,
                shape: shape.to_vec(),
                strides,
                dtype: DataType::Float64,
            }
        }

        pub fn i64(shape: &[usize], data: &[i64]) -> Self {
            let strides = compute_contiguous_strides(shape);
            let mut bytes = Vec::with_capacity(data.len() * 8);
            for v in data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            Self {
                bytes,
                shape: shape.to_vec(),
                strides,
                dtype: DataType::Int64,
            }
        }

        pub fn i32(shape: &[usize], data: &[i32]) -> Self {
            let strides = compute_contiguous_strides(shape);
            let mut bytes = Vec::with_capacity(data.len() * 4);
            for v in data {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            Self {
                bytes,
                shape: shape.to_vec(),
                strides,
                dtype: DataType::Int32,
            }
        }

        /// An f16 buffer built by rounding `data` (given in f32) to half.
        pub fn f16(shape: &[usize], data: &[f32]) -> Self {
            let strides = compute_contiguous_strides(shape);
            let mut bytes = Vec::with_capacity(data.len() * 2);
            for &v in data {
                bytes.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
            Self {
                bytes,
                shape: shape.to_vec(),
                strides,
                dtype: DataType::Float16,
            }
        }

        /// An f16 buffer built from raw 16-bit patterns (for adversarial
        /// NaN/inf/denormal cases that must survive without f32-reinterpret).
        pub fn f16_bits(shape: &[usize], bits: &[u16]) -> Self {
            let strides = compute_contiguous_strides(shape);
            let mut bytes = Vec::with_capacity(bits.len() * 2);
            for &b in bits {
                bytes.extend_from_slice(&b.to_le_bytes());
            }
            Self {
                bytes,
                shape: shape.to_vec(),
                strides,
                dtype: DataType::Float16,
            }
        }

        /// A bf16 buffer built by rounding `data` (given in f32) to bfloat16.
        pub fn bf16(shape: &[usize], data: &[f32]) -> Self {
            let strides = compute_contiguous_strides(shape);
            let mut bytes = Vec::with_capacity(data.len() * 2);
            for &v in data {
                bytes.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes());
            }
            Self {
                bytes,
                shape: shape.to_vec(),
                strides,
                dtype: DataType::BFloat16,
            }
        }

        /// A bf16 buffer built from raw 16-bit patterns.
        pub fn bf16_bits(shape: &[usize], bits: &[u16]) -> Self {
            let strides = compute_contiguous_strides(shape);
            let mut bytes = Vec::with_capacity(bits.len() * 2);
            for &b in bits {
                bytes.extend_from_slice(&b.to_le_bytes());
            }
            Self {
                bytes,
                shape: shape.to_vec(),
                strides,
                dtype: DataType::BFloat16,
            }
        }

        /// A u8 buffer.
        pub fn u8(shape: &[usize], data: &[u8]) -> Self {
            let strides = compute_contiguous_strides(shape);
            Self {
                bytes: data.to_vec(),
                shape: shape.to_vec(),
                strides,
                dtype: DataType::Uint8,
            }
        }

        pub fn bool_(shape: &[usize], data: &[bool]) -> Self {
            let strides = compute_contiguous_strides(shape);
            let bytes = data.iter().map(|&b| b as u8).collect();
            Self {
                bytes,
                shape: shape.to_vec(),
                strides,
                dtype: DataType::Bool,
            }
        }

        /// A zero-filled f32 output buffer of `shape`.
        pub fn zeros_f32(shape: &[usize]) -> Self {
            let n: usize = shape.iter().product();
            Self::f32(shape, &vec![0.0; n])
        }

        /// A zero-filled output buffer of `shape` with element type `dtype`.
        pub fn zeros(dtype: DataType, shape: &[usize]) -> Self {
            let n: usize = shape.iter().product();
            let strides = compute_contiguous_strides(shape);
            let esize = dtype.byte_size();
            Self {
                bytes: vec![0u8; n * esize],
                shape: shape.to_vec(),
                strides,
                dtype,
            }
        }

        /// Override strides/shape to expose the same bytes as a strided view
        /// (e.g. a transpose without copying).
        pub fn with_view(mut self, shape: &[usize], strides: &[i64]) -> Self {
            self.shape = shape.to_vec();
            self.strides = strides.to_vec();
            self
        }

        pub fn view(&self) -> TensorView<'_> {
            TensorView::new(
                DevicePtr(self.bytes.as_ptr() as *const std::ffi::c_void),
                self.dtype,
                &self.shape,
                &self.strides,
                DeviceId::cpu(),
            )
        }

        pub fn view_mut(&mut self) -> TensorMut<'_> {
            TensorMut::new(
                DevicePtrMut(self.bytes.as_mut_ptr() as *mut std::ffi::c_void),
                self.dtype,
                &self.shape,
                &self.strides,
                DeviceId::cpu(),
            )
        }

        pub fn to_f32(&self) -> Vec<f32> {
            self.bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }

        pub fn to_f64(&self) -> Vec<f64> {
            self.bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| f64::from_le_bytes(*c))
                .collect()
        }

        pub fn to_i64(&self) -> Vec<i64> {
            self.bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| i64::from_le_bytes(*c))
                .collect()
        }

        pub fn to_i32(&self) -> Vec<i32> {
            self.bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }

        pub fn to_bool(&self) -> Vec<bool> {
            self.bytes.iter().map(|&b| b != 0).collect()
        }

        /// Widen an f16 buffer to f32 for comparison.
        pub fn to_f16_as_f32(&self) -> Vec<f32> {
            self.bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()
        }

        /// The raw 16-bit patterns of an f16/bf16 buffer (to assert no
        /// f32-reinterpret corruption of NaN/inf/denormal inputs).
        pub fn to_u16_bits(&self) -> Vec<u16> {
            self.bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect()
        }

        /// Widen a bf16 buffer to f32 for comparison.
        pub fn to_bf16_as_f32(&self) -> Vec<f32> {
            self.bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()
        }

        pub fn to_u8(&self) -> Vec<u8> {
            self.bytes.clone()
        }
    }
}

#[cfg(test)]
mod tests {

    /// The plugin EP's node filter refuses a node unless *every* input and
    /// output dtype appears in this list, so a list written from the compute
    /// dtype alone silently excludes the whole op. `MatMulNBits` carries a
    /// `uint8` packed weight, an optional `uint8` zero-point and an optional
    /// `int32` `g_idx` alongside its float activation.
    #[test]
    fn matmul_nbits_advertises_its_quantized_edge_dtypes() {
        let dtypes = supported_dtypes_for_op("MatMulNBits", "com.microsoft");
        for required in [
            DataType::Float32,
            DataType::Float16,
            DataType::BFloat16,
            DataType::Uint8,
            DataType::Int32,
        ] {
            assert!(
                dtypes.contains(&required),
                "MatMulNBits must advertise {required:?}; without it the plugin EP's \
                 dtype filter drops every node and ORT runs the op instead"
            );
        }
    }

    /// Same trap, other quantized matmul: `QLinearMatMul` mixes `uint8`/`int8`
    /// operands with `float` scales.
    #[test]
    fn qlinear_matmul_advertises_its_quantized_edge_dtypes() {
        let dtypes = supported_dtypes_for_op("QLinearMatMul", "");
        for required in [DataType::Float32, DataType::Uint8, DataType::Int8] {
            assert!(
                dtypes.contains(&required),
                "QLinearMatMul must advertise {required:?}"
            );
        }
    }
    use super::*;
    use crate::strided::view_in_bounds;
    use testutil::Owned;

    #[cfg(not(feature = "ops-cnn"))]
    #[test]
    fn minimal_registry_excludes_deselected_cnn_group() {
        use onnx_runtime_operator_selection::CPU_OPERATOR_CATALOG;

        let registry = build_cpu_registry();
        assert!(registry.lookup("MatMul", "ai.onnx", 21).is_some());
        for entry in CPU_OPERATOR_CATALOG
            .iter()
            .filter(|entry| entry.group.feature == "ops-cnn")
        {
            assert!(
                registry
                    .lookup(entry.op_type, entry.domain, entry.since_version)
                    .is_none(),
                "{}::{} should be excluded without ops-cnn",
                entry.domain,
                entry.op_type
            );
        }
    }

    #[test]
    fn dense_roundtrip_contiguous() {
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]);
        let v = a.view();
        assert_eq!(to_dense_f32(&v).unwrap(), vec![1., 2., 3., 4., 5., 6.]);
    }

    #[test]
    fn dense_reads_transposed_view() {
        // Backing [2,3] row-major; expose as transposed [3,2] with strides [1,3].
        let a = Owned::f32(&[2, 3], &[1., 2., 3., 4., 5., 6.]).with_view(&[3, 2], &[1, 3]);
        let v = a.view();
        // Transpose of [[1,2,3],[4,5,6]] is [[1,4],[2,5],[3,6]].
        assert_eq!(to_dense_f32(&v).unwrap(), vec![1., 4., 2., 5., 3., 6.]);
    }

    #[test]
    fn write_dense_contiguous_bulk_copies() {
        // Contiguous output takes the bulk-copy fast path.
        let mut backing = Owned::f32(&[2, 3], &[0.0; 6]);
        let mut out = backing.view_mut();
        write_dense_f32(&mut out, &[1., 2., 3., 4., 5., 6.]).unwrap();
        assert_eq!(backing.to_f32(), vec![1., 2., 3., 4., 5., 6.]);
    }

    /// Transcription of the pre-bulk-copy `to_dense_bytes` body: the
    /// per-element strided walk with no contiguity fast path. This is the
    /// oracle the bulk-copy path must reproduce byte for byte.
    fn to_dense_bytes_via_strided_walk(view: &TensorView) -> Vec<u8> {
        let esize = elem_size(view.dtype).unwrap();
        let n = numel(view.shape);
        let mut out = vec![0u8; n * esize];
        if n == 0 {
            return out;
        }
        let origin = view.data_ptr::<u8>();
        let mut idx = vec![0usize; view.shape.len()];
        let mut w = 0usize;
        loop {
            let byte_off = elem_offset(view.strides, &idx) * esize as isize;
            // SAFETY: same in-shape, in-bounds read as the production walk.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    origin.offset(byte_off),
                    out.as_mut_ptr().add(w),
                    esize,
                );
            }
            w += esize;
            if !next_index(view.shape, &mut idx) {
                break;
            }
        }
        out
    }

    #[test]
    fn to_dense_bytes_bulk_copy_matches_the_strided_walk() {
        // The contiguous fast path must be byte-identical to the walk it
        // replaces. Odd dimensions and a rank-3 shape exercise the carry chain;
        // the payload spans the full `u8` range so a truncation or sign bug
        // could not hide.
        for shape in [
            vec![7usize],
            vec![3, 5],
            vec![2, 3, 7],
            vec![1, 41],
            vec![41, 1],
        ] {
            let n: usize = shape.iter().product();
            let data: Vec<u8> = (0..n).map(|i| (i.wrapping_mul(37) % 256) as u8).collect();
            let owned = Owned::u8(&shape, &data);
            let view = owned.view();
            assert!(view.is_contiguous(), "{shape:?} should be contiguous");
            assert_eq!(
                to_dense_bytes(&view).unwrap(),
                to_dense_bytes_via_strided_walk(&view),
                "bulk copy diverged from the strided walk for {shape:?}"
            );
            assert_eq!(to_dense_bytes(&view).unwrap(), data, "payload changed");
        }
    }

    #[test]
    fn to_dense_bytes_still_gathers_a_strided_view() {
        // Backing [2,3] row-major exposed as transposed [3,2] with strides
        // [1,3]: not contiguous, so the fast path must be skipped and the
        // gather must still apply the strides. Without this the fast path could
        // be reached unconditionally and every test above would still pass.
        let owned = Owned::u8(&[2, 3], &[1, 2, 3, 4, 5, 6]).with_view(&[3, 2], &[1, 3]);
        let view = owned.view();
        assert!(!view.is_contiguous());
        assert_eq!(to_dense_bytes(&view).unwrap(), vec![1, 4, 2, 5, 3, 6]);
        assert_eq!(
            to_dense_bytes(&view).unwrap(),
            to_dense_bytes_via_strided_walk(&view)
        );
    }

    #[test]
    fn to_dense_bytes_reads_multi_byte_elements_whole() {
        // `esize > 1` is where a byte/element unit mix-up in the bulk copy
        // shows up: it must copy `n * esize` bytes, not `n`.
        let data: Vec<i64> = vec![-1, 0, i64::MAX, i64::MIN, 7, -9];
        let owned = Owned::i64(&[2, 3], &data);
        let view = owned.view();
        let dense = to_dense_bytes(&view).unwrap();
        assert_eq!(dense.len(), 6 * 8);
        assert_eq!(dense, to_dense_bytes_via_strided_walk(&view));
        assert_eq!(dense, owned.bytes);
    }

    #[test]
    fn write_dense_bytes_bulk_copy_and_scatter_agree_with_their_layouts() {
        // The same payload written through a contiguous output (fast path) and
        // through a transposed output (strided scatter) must land in storage
        // the way each layout dictates, so the fast path cannot be taken
        // unconditionally.
        let payload: Vec<u8> = (1..=6u8).collect();

        let mut contiguous = Owned::u8(&[2, 3], &[0; 6]);
        write_dense_bytes(&mut contiguous.view_mut(), &payload).unwrap();
        assert_eq!(contiguous.bytes, payload);

        let mut strided = Owned::u8(&[2, 3], &[0; 6]).with_view(&[3, 2], &[1, 3]);
        write_dense_bytes(&mut strided.view_mut(), &payload).unwrap();
        assert_eq!(strided.bytes, vec![1, 3, 5, 2, 4, 6]);
    }

    #[test]
    fn write_dense_bytes_round_trips_multi_byte_elements() {
        let data: Vec<i64> = vec![-1, 0, i64::MAX, i64::MIN, 7, -9];
        let source = Owned::i64(&[2, 3], &data);
        let dense = to_dense_bytes(&source.view()).unwrap();

        let mut sink = Owned::i64(&[2, 3], &[0; 6]);
        write_dense_bytes(&mut sink.view_mut(), &dense).unwrap();
        assert_eq!(sink.to_i64(), data);
        assert_eq!(sink.bytes, source.bytes);
    }

    #[test]
    fn to_dense_bytes_honors_a_nonzero_byte_offset() {
        // The fast path must read from the *element origin*
        // (`data + byte_offset`), not from the allocation base. `Owned::view`
        // builds a zero-offset view, so construct the offset view directly.
        // Both paths route through `data_ptr`, but nothing else here pins that
        // down for the bulk copy.
        let backing = Owned::u8(&[6], &[10, 20, 30, 40, 50, 60]);
        let shape = [3usize];
        let strides = [1i64];
        let mut view = TensorView::new(
            onnx_runtime_ep_api::DevicePtr(backing.bytes.as_ptr() as *const std::ffi::c_void),
            DataType::Uint8,
            &shape,
            &strides,
            onnx_runtime_ir::DeviceId::cpu(),
        );
        view.byte_offset = 2;
        assert!(view.is_contiguous());
        assert_eq!(to_dense_bytes(&view).unwrap(), vec![30, 40, 50]);
        assert_eq!(
            to_dense_bytes(&view).unwrap(),
            to_dense_bytes_via_strided_walk(&view)
        );
    }

    #[test]
    fn to_dense_bytes_handles_a_rank_zero_scalar() {
        // `numel([]) == 1` and `is_contiguous([], [])` is true, so a scalar
        // takes the fast path with `n * esize == esize`.
        let owned = Owned::i64(&[], &[-7]);
        let view = owned.view();
        assert!(view.is_contiguous());
        let dense = to_dense_bytes(&view).unwrap();
        assert_eq!(dense.len(), 8);
        assert_eq!(dense, to_dense_bytes_via_strided_walk(&view));

        let mut sink = Owned::i64(&[], &[0]);
        write_dense_bytes(&mut sink.view_mut(), &dense).unwrap();
        assert_eq!(sink.to_i64(), vec![-7]);
    }

    #[test]
    fn write_dense_strided_matches_logical_order() {
        // Backing [2,3] row-major exposed as transposed [3,2] with strides
        // [1,3] so the write must scatter through the non-contiguous stride
        // walk. `to_f32` reads raw storage order to confirm the scatter.
        let mut backing = Owned::f32(&[2, 3], &[0.0; 6]).with_view(&[3, 2], &[1, 3]);
        let mut out = backing.view_mut();
        // Logical [[1,2],[3,4],[5,6]] over strides [1,3] lands in storage as
        // [1,3,5, 2,4,6].
        write_dense_f32(&mut out, &[1., 2., 3., 4., 5., 6.]).unwrap();
        assert_eq!(backing.to_f32(), vec![1., 3., 5., 2., 4., 6.]);
    }

    #[cfg(feature = "full")]
    #[test]
    fn registry_has_all_phase1_ops() {
        let reg = build_cpu_registry();
        // Every Phase-1 op has at least one factory, and each resolves at a
        // modern opset. `Softmax` is registered twice (legacy v1 + per-axis
        // v13), and `LayerNormalization`, `FusedMatMulBias`, `FusedGemm`,
        // `FusedAttention` and the fused exact-GELU `Gelu` add contrib
        // (`com.microsoft`) entries. Standard `ai.onnx::Attention` is registered
        // at both opset 23 and 24 (two default-domain entries not in
        // `PHASE1_OPS`). The standard LLM primitives `Gelu` (opset 20),
        // `RMSNormalization` (23), `RotaryEmbedding` (23) and `Swish` (24) add
        // four more default-domain entries not in `PHASE1_OPS`; `Softmax` and
        // `LogSoftmax` each have a legacy and an opset-13 entry. Five contrib
        // (`com.microsoft`) fused transformer entries (BiasGelu, FastGelu,
        // QuickGelu, Silu, SkipLayerNormalization, SimplifiedLayerNormalization,
        // SkipSimplifiedLayerNormalization) add seven more; `MoE`, `QMoE`, and
        // `GroupQueryAttention` add one contrib entry each.
        // QuantizeLinear and DequantizeLinear each add six versioned entries,
        // while DynamicQuantizeLinear and QLinearMatMul add one each
        // (twenty-nine over the
        // op-name count). Pooling adds twelve more versioned entries: five each
        // for AveragePool and MaxPool, plus the two global pool operators, for
        // forty over the op-name count. ScatterElements also has distinct
        // opset-11 and opset-16 registrations. BatchNormalization,
        // InstanceNormalization and PRelu add one registration each, while
        // GroupNormalization adds opset-18 and opset-21 entries, for forty-seven
        // registrations over the Phase-1 op-name count in total. ScatterND has
        // opset-11, -16, and -18 entries, adding two more.
        // ReduceLogSumExp adds a separate opset-18 axes-input registration.
        // BitwiseAnd,
        // BitwiseOr, BitwiseXor, BitwiseNot, and Hardmax add five more.
        // MatMulNBits, BlockQuantizedMatMul, BlockQuantizedMoE, IndexShare,
        // VarlenAttention, PackedVarlenAttention, PackedMultiHeadAttention,
        // SparseKvGather, CompressedSparseAttention, and GroupQueryAttention add
        // private/contrib registrations.
        // CumProd and the three standard window generators add four more
        // default-domain entries beyond the original Phase-1 set.
        // GridSample has separate opset-16 and opset-20 registrations.
        // `CausalConvWithState` and `LinearAttention` (Qwen3.5 hybrid
        // linear-attention primitives) add two more contrib entries, and
        // `LinearAttention` is additionally registered under the standard ONNX
        // domain (onnx/onnx#7689), reusing the same kernel, for one more entry.
        // `GatherBlockQuantized` (block-quantized embedding gather) adds one,
        // the `com.microsoft::RotaryEmbedding` contrib alias adds one,
        // `com.microsoft::MultiHeadAttention` (separate-QKV SDPA) adds one, and
        // `com.microsoft::Attention` (packed-QKV SDPA) adds one.
        // The six `pkg.nxrt` NCHWc blocked-layout ops (reorder to/from blocked,
        // blocked Conv, blocked Max/Average/GlobalAverage pool) emitted by the
        // NCHWc layout-propagation pass add six more entries, but only when the
        // `mlas` feature is enabled (the NCHWc kernels are MLAS-backed). Standard
        // Conv is always registered, using the pure-Rust reference kernel without
        // `mlas` and the optimized implementation with it.
        // `IsNaN` (opset-9 float NaN predicate) adds one default-domain entry.
        let mlas_registrations = if cfg!(feature = "mlas") { 6 } else { 0 };
        assert_eq!(reg.len(), PHASE1_OPS.len() + 102 + mlas_registrations);
        for op in PHASE1_OPS {
            assert!(reg.lookup(op, "", 21).is_some(), "missing factory for {op}");
        }
        // Softmax selects legacy at opset ≤ 12 and per-axis at opset ≥ 13.
        assert!(reg.lookup("Softmax", "", 12).is_some());
        assert!(reg.lookup("Softmax", "", 13).is_some());
        assert!(reg.lookup("LogSoftmax", "", 12).is_some());
        assert!(reg.lookup("LogSoftmax", "", 13).is_some());
        assert!(reg.lookup("ReduceLogSumExp", "", 17).is_some());
        assert!(reg.lookup("ReduceLogSumExp", "", 18).is_some());
        assert!(reg.lookup("CumSum", "", 14).is_some());
        assert!(reg.lookup("CumProd", "", 26).is_some());
        assert!(reg.lookup("HannWindow", "", 17).is_some());
        assert!(reg.lookup("HammingWindow", "", 17).is_some());
        assert!(reg.lookup("BlackmanWindow", "", 17).is_some());
        assert!(reg.lookup("DFT", "", 17).is_some());
        assert!(reg.lookup("Conv", "", 22).is_some());
        assert!(reg.lookup("LpPool", "", 18).is_some());
        assert!(reg.lookup("GlobalLpPool", "", 2).is_some());
        assert!(reg.lookup("SpaceToDepth", "", 13).is_some());
        assert!(reg.lookup("Split", "", 18).is_some());
        assert!(reg.lookup("Unique", "", 11).is_some());
        assert!(reg.lookup("Dropout", "", 13).is_some());
        assert!(reg.lookup("Dropout", "", 22).is_some());
        assert!(reg.lookup("GridSample", "", 16).is_some());
        assert!(reg.lookup("GridSample", "", 20).is_some());
        assert!(reg.lookup("Resize", "", 10).is_some());
        assert!(reg.lookup("Resize", "", 25).is_some());
        assert!(reg.lookup("ConvTranspose", "", 22).is_some());
        assert!(reg.lookup("ScatterND", "", 18).is_some());
        assert!(reg.lookup("QLinearMatMul", "", 10).is_some());
        assert!(reg.lookup("MatMulNBits", "com.microsoft", 1).is_some());
        assert!(reg.lookup("QMoE", "com.microsoft", 1).is_some());
        assert!(reg.lookup("BlockQuantizedMatMul", "pkg.nxrt", 1).is_some());
        assert!(reg.lookup("BlockQuantizedMoE", "pkg.nxrt", 1).is_some());
        assert!(reg.lookup("IndexShare", "pkg.nxrt", 1).is_some());
        assert!(reg.lookup("VarlenAttention", "pkg.nxrt", 1).is_some());
        assert!(reg.lookup("PackedVarlenAttention", "pkg.nxrt", 1).is_some());
        assert!(
            reg.lookup("PackedMultiHeadAttention", "com.microsoft", 1)
                .is_some()
        );
        assert!(reg.lookup("SparseKvGather", "pkg.nxrt", 1).is_some());
        assert!(
            reg.lookup("CompressedSparseAttention", "pkg.nxrt", 1)
                .is_some()
        );
        assert!(reg.lookup("Conv", "", 21).is_some());
        assert!(
            reg.lookup("GroupQueryAttention", "com.microsoft", 1)
                .is_some()
        );
        assert!(
            reg.lookup("MultiHeadAttention", "com.microsoft", 1)
                .is_some()
        );
        assert!(
            reg.lookup("CausalConvWithState", "com.microsoft", 1)
                .is_some()
        );
        assert!(reg.lookup("LinearAttention", "com.microsoft", 1).is_some());
        assert!(
            reg.lookup("GatherBlockQuantized", "com.microsoft", 1)
                .is_some()
        );
        assert!(reg.lookup("SimplifiedLayerNormalization", "", 21).is_some());
        // The fused contrib-domain LayerNormalization resolves to the same
        // kernel as the standard default-domain op.
        assert!(
            reg.lookup("LayerNormalization", "com.microsoft", 1)
                .is_some()
        );
        assert!(reg.supports("LayerNormalization", "com.microsoft", 1));
        assert!(reg.supports("MatMul", "ai.onnx", 1));
        // The `MatMul + Add` fusion's contrib op now has a CPU kernel.
        assert!(reg.supports("FusedMatMulBias", "com.microsoft", 1));
        // The `MatMul + Add + Relu` fusion's contrib op now has a CPU kernel.
        assert!(reg.supports("FusedGemm", "com.microsoft", 1));
        assert!(reg.lookup("FusedGemm", "com.microsoft", 1).is_some());
        // The exact-GELU fusion's contrib op has a CPU kernel (contrib-only).
        assert!(reg.supports("Gelu", "com.microsoft", 1));
        assert!(reg.supports("MoE", "com.microsoft", 1));
        assert!(reg.lookup("Gelu", "com.microsoft", 1).is_some());
        for op in [
            "BiasGelu",
            "FastGelu",
            "QuickGelu",
            "Silu",
            "SkipLayerNormalization",
            "SimplifiedLayerNormalization",
            "SkipSimplifiedLayerNormalization",
        ] {
            assert!(
                reg.lookup(op, "com.microsoft", 1).is_some(),
                "missing contrib factory for {op}"
            );
        }
        // Standard `ai.onnx::Gelu` (opset 20) is now registered in the default
        // domain; it resolves at opset ≥ 20 but not below its since-version.
        assert!(reg.lookup("Gelu", "", 21).is_some());
        assert!(reg.lookup("Gelu", "", 20).is_some());
        assert!(reg.lookup("Gelu", "", 19).is_none());
        // Standard LLM primitives resolve at/after their since-versions.
        assert!(reg.lookup("RMSNormalization", "", 23).is_some());
        assert!(reg.lookup("RMSNormalization", "", 22).is_none());
        assert!(reg.lookup("BatchNormalization", "", 15).is_some());
        assert!(reg.lookup("BatchNormalization", "", 7).is_some());
        assert!(reg.lookup("BatchNormalization", "", 6).is_none());
        assert!(reg.lookup("InstanceNormalization", "", 6).is_some());
        assert!(reg.lookup("GroupNormalization", "", 18).is_some());
        assert!(reg.lookup("GroupNormalization", "", 21).is_some());
        assert!(reg.lookup("GroupNormalization", "", 17).is_none());
        assert!(reg.lookup("PRelu", "", 16).is_some());
        assert!(reg.lookup("PRelu", "", 15).is_none());
        assert!(reg.lookup("LpNormalization", "", 1).is_some());
        assert!(reg.lookup("Selu", "", 6).is_some());
        assert!(reg.lookup("Selu", "", 5).is_none());
        assert!(reg.lookup("ThresholdedRelu", "", 10).is_some());
        assert!(reg.lookup("ThresholdedRelu", "", 9).is_none());
        assert!(reg.lookup("RotaryEmbedding", "", 23).is_some());
        assert!(reg.lookup("RotaryEmbedding", "com.microsoft", 1).is_some());
        assert!(reg.lookup("RotaryEmbedding", "", 22).is_none());
        assert!(reg.lookup("Swish", "", 24).is_some());
        assert!(reg.lookup("Swish", "", 23).is_none());
        // Standard ai.onnx::Attention resolves at opsets 23–26 (default domain
        // and the `ai.onnx` alias), but not below its since-version. Opset 23
        // resolves to the v23 kernel; 24/25/26 resolve to the v24 kernel.
        assert!(reg.lookup("Attention", "", 23).is_some());
        assert!(reg.lookup("Attention", "", 24).is_some());
        assert!(reg.lookup("Attention", "", 25).is_some());
        assert!(reg.lookup("Attention", "", 26).is_some());
        assert!(reg.lookup("Attention", "ai.onnx", 23).is_some());
        assert!(reg.lookup("Attention", "ai.onnx", 26).is_some());
        assert!(reg.lookup("Attention", "", 22).is_none());
        assert!(reg.supports("Attention", "", 23));
    }

    #[test]
    fn dense_read_stays_in_bounds() {
        let a = Owned::f32(&[3, 2], &[1., 4., 2., 5., 3., 6.]);
        let v = a.view();
        view_in_bounds(v.shape, v.strides, v.byte_offset, 4, a.bytes.len()).unwrap();
    }

    // ─── Registry-entry derivation tests ─────────────────────────────────

    /// Shared fixture: build the registry + descriptors once for all
    /// descriptor-related tests, avoiding ~8 redundant full-registry
    /// constructions that inflate peak memory and wall-clock time.
    fn shared_registry_with_descriptors() -> &'static (OpRegistry, Vec<CpuOpDescriptor>) {
        use std::sync::OnceLock;
        static FIXTURE: OnceLock<(OpRegistry, Vec<CpuOpDescriptor>)> = OnceLock::new();
        FIXTURE.get_or_init(build_cpu_registry_with_descriptors)
    }

    #[test]
    fn build_cpu_registry_with_descriptors_returns_nonempty() {
        let (_reg, descriptors) = shared_registry_with_descriptors();
        assert!(
            descriptors.len() > 100,
            "expected >100 descriptors, got {}",
            descriptors.len()
        );
    }

    #[test]
    fn descriptors_include_add_with_f16_bf16() {
        let (_reg, descriptors) = shared_registry_with_descriptors();
        let add_entries: Vec<_> = descriptors
            .iter()
            .filter(|d| d.op_type == "Add" && d.domain.is_empty())
            .collect();
        assert!(!add_entries.is_empty(), "Add must appear in descriptors");
        for entry in &add_entries {
            assert!(
                entry.supported_dtypes.contains(&DataType::Float16),
                "Add must advertise Float16"
            );
            assert!(
                entry.supported_dtypes.contains(&DataType::BFloat16),
                "Add must advertise BFloat16"
            );
        }
    }

    #[test]
    fn descriptors_include_matmul_with_f16_bf16() {
        let (_reg, descriptors) = shared_registry_with_descriptors();
        let matmul_entries: Vec<_> = descriptors
            .iter()
            .filter(|d| d.op_type == "MatMul" && d.domain.is_empty())
            .collect();
        assert!(
            !matmul_entries.is_empty(),
            "MatMul must appear in descriptors"
        );
        for entry in &matmul_entries {
            assert!(
                entry.supported_dtypes.contains(&DataType::Float16),
                "MatMul must advertise Float16"
            );
            assert!(
                entry.supported_dtypes.contains(&DataType::BFloat16),
                "MatMul must advertise BFloat16"
            );
        }
    }

    #[test]
    fn fail_closed_unknown_op_gets_f32_only() {
        // An op not in our dtype mapping should get only f32.
        let dtypes = supported_dtypes_for_op("TotallyFakeOp", "");
        assert_eq!(dtypes, &[DataType::Float32]);
        assert!(
            !dtypes.contains(&DataType::Float16),
            "unknown op must not advertise Float16"
        );
    }

    #[test]
    fn fail_closed_pkg_nxrt_ops_get_f32_only() {
        let dtypes = supported_dtypes_for_op("BlockQuantizedMatMul", "pkg.nxrt");
        assert_eq!(dtypes, &[DataType::Float32]);
    }

    #[test]
    fn descriptors_derived_from_real_registry_not_hand_maintained() {
        // Every registered op must have a descriptor -- not "close to", exactly.
        //
        // This used to allow a delta of up to 50 and documented CNN ops as the
        // expected difference. That tolerance was hiding a real bug: the plugin
        // builds its `KernelRegistryEntry` list from these descriptors and
        // `node_passes_dtype_filter` fails closed on an op with no entry, so
        // each of the 18 missing ops (`PRelu`, `BatchNormalization`,
        // `InstanceNormalization`, `GroupNormalization`, `Conv`, pooling,
        // `Resize`, ...) was claimed by `supports_op` and then silently handed
        // back to ORT's CPU EP at capability time.
        let reg = build_cpu_registry();
        let (_reg2, descriptors) = shared_registry_with_descriptors();

        let described: std::collections::BTreeSet<(&str, &str, u64)> = descriptors
            .iter()
            .map(|d| (d.op_type.as_str(), d.domain.as_str(), d.since_version))
            .collect();
        let missing: Vec<String> = reg
            .keys()
            .filter(|k| {
                !described.contains(&(k.op_type.as_str(), k.domain.as_str(), k.since_version))
            })
            .map(|k| format!("{}::{}@{}", k.domain, k.op_type, k.since_version))
            .collect();
        assert!(
            missing.is_empty(),
            "{} registered ops have no descriptor, so the plugin would decline \
             them to ORT's CPU EP: {missing:?}",
            missing.len()
        );
        assert_eq!(
            descriptors.len(),
            reg.len(),
            "descriptor count must equal registry size exactly"
        );
    }

    /// Advertised dtypes must be what the kernel accepts, not a superset.
    ///
    /// The plugin turns these into the `KernelRegistryEntry` dtype filter, so
    /// an over-claim is not harmless: ORT routes the node to us, capability
    /// passes, and it fails at `Run` instead of being declined up front.
    /// `Conv` is the live case -- it computes through MLAS in f32 and
    /// `ConvKernel::execute` rejects f64 outright, so it must advertise
    /// `FLOAT_COMPUTE_DTYPES` and not `FLOAT_DTYPES`.
    #[test]
    fn conv_does_not_advertise_a_dtype_its_kernel_rejects() {
        let dtypes = supported_dtypes_for_op("Conv", "");
        assert!(
            !dtypes.contains(&DataType::Float64),
            "Conv advertises f64 but ConvKernel::execute rejects it: {dtypes:?}"
        );
        for want in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
            assert!(dtypes.contains(&want), "Conv must still advertise {want:?}");
        }
    }

    /// The descriptor order is leaked into a `'static` slice handed to ORT, so
    /// it must not depend on hash-map iteration order.
    #[test]
    fn descriptors_are_deterministically_ordered() {
        let (_r, a) = build_cpu_registry_with_descriptors();
        let (_r2, b) = build_cpu_registry_with_descriptors();
        let key = |d: &CpuOpDescriptor| (d.domain.clone(), d.op_type.clone(), d.since_version);
        assert_eq!(
            a.iter().map(key).collect::<Vec<_>>(),
            b.iter().map(key).collect::<Vec<_>>()
        );
    }
}

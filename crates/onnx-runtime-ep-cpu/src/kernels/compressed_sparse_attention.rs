//! Correctness-first reference paths for `pkg.nxrt::CompressedSparseAttention`
//! v1.
//!
//! The registered operator exposes the complete frozen stateful v1 boundary.
//! Ratio-128 and ratio-4 own their persistent compressed records and incremental
//! carries. Ratio-4 additionally builds the shared FP4 index-key stream and
//! learned top-k selection. Unfrozen top-k ties and the MTP sidecar remain
//! explicit Unsupported paths.

use std::borrow::Cow;

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Dim, Node, Shape, as_static_shape};

use super::block_dequant::{
    FP4_E2M1_BLOCK_SIZE, FP4_E2M1_PACKED_BYTES, FP8_E4M3_BLOCK_SIZE, FP8_E4M3_PACKED_BYTES,
    dequantize_fp4_e2m1_block, dequantize_fp8_e4m3_block, quantize_fp4_e2m1_block,
    quantize_fp8_e4m3_block,
};
use super::sparse_kv_gather::{
    checked_layout, checked_product, fallible_filled, read_dense_f32, read_dense_indices,
    sparse_kv_gather_masked_f32,
};
use super::{check_arity, to_dense_bytes, to_dense_i64, write_dense_bytes, write_dense_f32};

const OP: &str = "CompressedSparseAttention";
const LAYOUT_VERSION: i64 = 1;
const FROZEN_V1_REQUIRED_INPUTS: usize = 11;
const FROZEN_V1_MAX_INPUTS: usize = 20;
const FROZEN_V1_REQUIRED_OUTPUTS: usize = 3;
const FROZEN_V1_MAX_OUTPUTS: usize = 6;
const FROZEN_V1_REQUIRED_INPUT_NAMES: [&str; FROZEN_V1_REQUIRED_INPUTS] = [
    "query",
    "current_kv",
    "compressor_kv",
    "compressor_gate",
    "compressor_ape",
    "compressor_norm",
    "past_compressed_kv",
    "past_compression_carry",
    "seqlens_k",
    "total_sequence_length",
    "head_sink",
];

pub struct CompressedSparseAttentionFactory;

struct StatefulCompressedSparseAttentionKernel {
    num_heads: usize,
    head_dim: usize,
    qk_rope_head_dim: usize,
    compression_ratio: usize,
    index_num_heads: usize,
    index_head_dim: usize,
    index_topk: usize,
    scale: f32,
    cache_format: CacheFormat,
}

struct CompressedSparseAttentionKernel {
    num_heads: usize,
    head_dim: usize,
    qk_rope_head_dim: usize,
    compression_ratio: usize,
    index_num_heads: usize,
    scale: f32,
    cache_format: CacheFormat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CacheFormat {
    F32,
    // Uint8 records concatenate `[E8M0 scale, 64 E4M3FN values]` blocks for
    // non-RoPE dimensions, followed by little-endian BF16 RoPE values.
    Fp8E4m3Block64,
    // Uint8 records concatenate `[E8M0 scale, 16 adjacent-nibble bytes]` blocks.
    Fp4E2m1Block32,
}

impl CacheFormat {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "f32" => Ok(Self::F32),
            "fp8_e4m3_block64" => Ok(Self::Fp8E4m3Block64),
            "fp4_e2m1_block32" => Ok(Self::Fp4E2m1Block32),
            _ => Err(unsupported(format!(
                "cache_format='{value}' is unsupported; expected f32, fp8_e4m3_block64, or fp4_e2m1_block32"
            ))),
        }
    }

    fn block_layout(self) -> Option<(usize, usize)> {
        match self {
            Self::F32 => None,
            Self::Fp8E4m3Block64 => Some((FP8_E4M3_BLOCK_SIZE, FP8_E4M3_PACKED_BYTES + 1)),
            Self::Fp4E2m1Block32 => Some((FP4_E2M1_BLOCK_SIZE, FP4_E2M1_PACKED_BYTES + 1)),
        }
    }

    fn stored_width(self, logical_width: usize, qk_rope_head_dim: usize) -> Result<usize> {
        match self {
            Self::F32 => Ok(logical_width),
            Self::Fp8E4m3Block64 => {
                let non_rope = logical_width.checked_sub(qk_rope_head_dim).ok_or_else(|| {
                    error(format!(
                        "qk_rope_head_dim {qk_rope_head_dim} exceeds head_dim {logical_width}"
                    ))
                })?;
                if !non_rope.is_multiple_of(FP8_E4M3_BLOCK_SIZE) {
                    return Err(error(format!(
                        "non-RoPE head dimension {non_rope} must be divisible by FP8 block size {FP8_E4M3_BLOCK_SIZE}"
                    )));
                }
                let fp8_bytes = non_rope
                    .checked_div(FP8_E4M3_BLOCK_SIZE)
                    .and_then(|blocks| blocks.checked_mul(FP8_E4M3_PACKED_BYTES + 1))
                    .ok_or_else(|| error("FP8 cache record width overflow"))?;
                let rope_bytes = qk_rope_head_dim
                    .checked_mul(std::mem::size_of::<u16>())
                    .ok_or_else(|| error("BF16 RoPE tail width overflow"))?;
                fp8_bytes
                    .checked_add(rope_bytes)
                    .filter(|&bytes| bytes <= isize::MAX as usize)
                    .ok_or_else(|| {
                        error("hybrid cache record width overflow or exceeds isize::MAX")
                    })
            }
            Self::Fp4E2m1Block32 => {
                if !logical_width.is_multiple_of(FP4_E2M1_BLOCK_SIZE) {
                    return Err(error(format!(
                        "head_dim {logical_width} must be divisible by FP4 block size {FP4_E2M1_BLOCK_SIZE}"
                    )));
                }
                logical_width
                    .checked_div(FP4_E2M1_BLOCK_SIZE)
                    .and_then(|blocks| blocks.checked_mul(FP4_E2M1_PACKED_BYTES + 1))
                    .filter(|&bytes| bytes <= isize::MAX as usize)
                    .ok_or_else(|| error("FP4 cache record width overflow or exceeds isize::MAX"))
            }
        }
    }

    fn dtype(self) -> DataType {
        match self {
            Self::F32 => DataType::Float32,
            Self::Fp8E4m3Block64 | Self::Fp4E2m1Block32 => DataType::Uint8,
        }
    }
}

impl KernelFactory for CompressedSparseAttentionFactory {
    fn create(&self, node: &Node, input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        validate_frozen_v1_schema(node)?;
        self.create_impl(node, input_shapes, false)
    }
}

impl CompressedSparseAttentionFactory {
    fn create_impl(
        &self,
        node: &Node,
        input_shapes: &[Vec<usize>],
        assembled_cache_reference: bool,
    ) -> Result<Box<dyn Kernel>> {
        let num_heads = required_positive_int(node, "num_heads")?;
        let head_dim = required_positive_int(node, "head_dim")?;
        let qk_rope_head_dim = optional_nonnegative_int(node, "qk_rope_head_dim", 0)?;
        if qk_rope_head_dim > head_dim {
            return Err(error(format!(
                "qk_rope_head_dim {qk_rope_head_dim} exceeds head_dim {head_dim}"
            )));
        }
        let compression_ratio = required_positive_int(node, "compression_ratio")?;
        if !matches!(compression_ratio, 4 | 128) {
            return Err(error(format!(
                "compression_ratio must be exactly 4 or 128, got {compression_ratio}"
            )));
        }
        if !assembled_cache_reference {
            validate_ratio_specific_v1_schema(node, compression_ratio)?;
        }
        let index_num_heads = optional_nonnegative_int(node, "index_num_heads", 0)?;
        let index_head_dim = optional_nonnegative_int(node, "index_head_dim", 0)?;
        let index_topk = optional_nonnegative_int(node, "index_topk", 0)?;
        if compression_ratio == 4
            && (index_num_heads == 0 || index_head_dim == 0 || index_topk == 0)
        {
            return Err(error(
                "ratio-4 requires positive index_num_heads, index_head_dim, and index_topk",
            ));
        }
        if compression_ratio == 128
            && (index_num_heads != 0 || index_head_dim != 0 || index_topk != 0)
        {
            return Err(error(
                "ratio-128 requires index_num_heads=index_head_dim=index_topk=0",
            ));
        }
        require_int_attr(node, "causal", 1)?;
        require_int_attr(node, "cache_layout_version", LAYOUT_VERSION)?;
        require_int_attr(node, "index_layout_version", LAYOUT_VERSION)?;
        let sink_mode = node
            .attr("sink_mode")
            .map(|attribute| {
                attribute
                    .as_str()
                    .ok_or_else(|| error("attribute sink_mode must be a UTF-8 string"))
            })
            .transpose()?
            .unwrap_or("logit_only");
        if sink_mode != "logit_only" {
            return Err(unsupported(format!(
                "sink_mode='{sink_mode}' is unsupported for the learned per-head logit input \
                 `head_sink`; v1 requires 'logit_only'. Metadata `sink_tokens` configures retained \
                 prefix tokens and is unrelated"
            )));
        }
        let cache_format = node
            .attr("cache_format")
            .map(|attribute| {
                attribute
                    .as_str()
                    .ok_or_else(|| error("attribute cache_format must be a UTF-8 string"))
            })
            .transpose()?
            .unwrap_or("f32");
        let cache_format = CacheFormat::parse(cache_format)?;
        let scale = node
            .attr("scale")
            .and_then(|attribute| attribute.as_float())
            .unwrap_or(0.0);
        if !scale.is_finite() || scale < 0.0 {
            return Err(error("scale must be finite and non-negative"));
        }

        if assembled_cache_reference && input_shapes.len() >= 4 {
            infer_output_shape_for_format(
                &input_shapes[0],
                &input_shapes[1],
                &input_shapes[2],
                &input_shapes[3],
                num_heads,
                head_dim,
                qk_rope_head_dim,
                cache_format,
            )?;
        }
        if assembled_cache_reference {
            Ok(Box::new(CompressedSparseAttentionKernel {
                num_heads,
                head_dim,
                qk_rope_head_dim,
                compression_ratio,
                index_num_heads,
                scale,
                cache_format,
            }))
        } else {
            Ok(Box::new(StatefulCompressedSparseAttentionKernel {
                num_heads,
                head_dim,
                qk_rope_head_dim,
                compression_ratio,
                index_num_heads,
                index_head_dim,
                index_topk,
                scale,
                cache_format,
            }))
        }
    }

    #[cfg(test)]
    fn create_assembled_cache_reference(
        &self,
        node: &Node,
        input_shapes: &[Vec<usize>],
    ) -> Result<Box<dyn Kernel>> {
        self.create_impl(node, input_shapes, true)
    }
}

/// Claim-time denial for the frozen stateful CSA contract.
///
/// The factory dry-run is the source of truth for attribute and positional
/// arity validation. The remaining checks mirror the dtype and shape
/// requirements enforced by the stateful kernels before execution.
pub(crate) fn unsupported_reason(
    node: &Node,
    shapes: &[Shape],
    input_dtypes: &[DataType],
) -> Option<Cow<'static, str>> {
    let concrete_shapes = shapes
        .iter()
        .map(|shape| as_static_shape(shape))
        .collect::<Option<Vec<_>>>()
        .unwrap_or_default();
    if let Err(error) = CompressedSparseAttentionFactory.create(node, &concrete_shapes) {
        return Some(Cow::Owned(error.to_string()));
    }

    if shapes.len() != node.inputs.len() || input_dtypes.len() != node.inputs.len() {
        return Some(Cow::Owned(format!(
            "{OP}: claim metadata must cover all {} positional inputs (got {} shapes and {} dtypes)",
            node.inputs.len(),
            shapes.len(),
            input_dtypes.len()
        )));
    }

    let ratio = usize::try_from(
        node.attr("compression_ratio")
            .and_then(|attribute| attribute.as_int())
            .expect("CPU factory accepted compression_ratio"),
    )
    .expect("CPU factory accepted positive compression_ratio");
    let cache_format = node
        .attr("cache_format")
        .and_then(|attribute| attribute.as_str())
        .unwrap_or("f32");

    let result = match ratio {
        4 => validate_ratio4_claim(node, shapes, input_dtypes, cache_format),
        128 => validate_ratio128_claim(node, shapes, input_dtypes, cache_format),
        _ => unreachable!("CPU factory rejected unsupported compression ratio"),
    }
    .and_then(|()| validate_attention_bias_claim(node, shapes, input_dtypes));

    result
        .err()
        .map(|reason| Cow::Owned(format!("{OP}: {reason}")))
}

fn validate_ratio4_claim(
    node: &Node,
    shapes: &[Shape],
    input_dtypes: &[DataType],
    cache_format: &str,
) -> std::result::Result<(), String> {
    if node
        .attr("index_head_dim")
        .and_then(|attribute| attribute.as_int())
        != Some(128)
    {
        return Err("ratio-4 requires index_head_dim=128".into());
    }
    if cache_format != "fp8_e4m3_block64" {
        return Err(format!(
            "ratio-4 requires cache_format='fp8_e4m3_block64', got '{cache_format}'"
        ));
    }
    require_fixed_claim_contract(node, 4)?;

    for &(index, expected, name) in &[
        (0, DataType::Float32, "query"),
        (1, DataType::Float32, "current_kv"),
        (2, DataType::Float32, "compressor_kv"),
        (3, DataType::Float32, "compressor_gate"),
        (4, DataType::Float32, "compressor_ape"),
        (5, DataType::Float32, "compressor_norm"),
        (6, DataType::Uint8, "past_compressed_kv"),
        (7, DataType::Float32, "past_compression_carry"),
        (8, DataType::Int32, "seqlens_k"),
        (9, DataType::Int64, "total_sequence_length"),
        (10, DataType::Float32, "head_sink"),
        (11, DataType::Float32, "index_query"),
        (12, DataType::Float32, "index_weight"),
        (13, DataType::Float32, "index_compressor_kv"),
        (14, DataType::Float32, "index_compressor_gate"),
        (15, DataType::Float32, "index_compressor_ape"),
        (16, DataType::Float32, "index_compressor_norm"),
        (17, DataType::Uint8, "past_index_key"),
        (18, DataType::Float32, "past_index_carry"),
    ] {
        require_claim_dtype(input_dtypes, index, expected, name)?;
    }
    let heads = required_claim_attr(node, "num_heads")?;
    let index_heads = required_claim_attr(node, "index_num_heads")?;
    for (index, name, contract) in [
        (0, "query", vec![Any, NonZero, Fixed(heads), Fixed(512)]),
        (1, "current_kv", vec![Same(0, 0), Any, Fixed(512)]),
        (
            2,
            "compressor_kv",
            vec![Same(0, 0), Same(0, 1), Fixed(1024)],
        ),
        (
            3,
            "compressor_gate",
            vec![Same(0, 0), Same(0, 1), Fixed(1024)],
        ),
        (4, "compressor_ape", vec![Fixed(4), Fixed(1024)]),
        (5, "compressor_norm", vec![Fixed(512)]),
        (6, "past_compressed_kv", vec![Same(0, 0), Any, Fixed(583)]),
        (
            7,
            "past_compression_carry",
            vec![Same(0, 0), Fixed(8), Fixed(2), Fixed(1024)],
        ),
        (8, "seqlens_k", vec![Same(0, 0)]),
        (9, "total_sequence_length", vec![]),
        (10, "head_sink", vec![Fixed(heads)]),
        (
            11,
            "index_query",
            vec![Same(0, 0), Same(0, 1), Fixed(index_heads), Fixed(128)],
        ),
        (
            12,
            "index_weight",
            vec![Same(0, 0), Same(0, 1), Fixed(index_heads)],
        ),
        (
            13,
            "index_compressor_kv",
            vec![Same(0, 0), Same(0, 1), Fixed(256)],
        ),
        (
            14,
            "index_compressor_gate",
            vec![Same(0, 0), Same(0, 1), Fixed(256)],
        ),
        (15, "index_compressor_ape", vec![Fixed(4), Fixed(256)]),
        (16, "index_compressor_norm", vec![Fixed(128)]),
        (17, "past_index_key", vec![Same(0, 0), Any, Fixed(68)]),
        (
            18,
            "past_index_carry",
            vec![Same(0, 0), Fixed(8), Fixed(2), Fixed(256)],
        ),
    ] {
        require_claim_shape(shapes, index, name, &contract)?;
    }
    Ok(())
}

fn validate_ratio128_claim(
    node: &Node,
    shapes: &[Shape],
    input_dtypes: &[DataType],
    cache_format: &str,
) -> std::result::Result<(), String> {
    if cache_format == "fp4_e2m1_block32" {
        return Err(
            "ratio-128 attention-compressor state uses f32 or hybrid FP8/BF16 records, not FP4"
                .into(),
        );
    }
    require_fixed_claim_contract(node, 128)?;

    let cache_dtype = if cache_format == "f32" {
        DataType::Float32
    } else {
        DataType::Uint8
    };
    for &(index, expected, name) in &[
        (0, DataType::Float32, "query"),
        (1, DataType::Float32, "current_kv"),
        (2, DataType::Float32, "compressor_kv"),
        (3, DataType::Float32, "compressor_gate"),
        (4, DataType::Float32, "compressor_ape"),
        (5, DataType::Float32, "compressor_norm"),
        (6, cache_dtype, "past_compressed_kv"),
        (7, DataType::Float32, "past_compression_carry"),
        (8, DataType::Int32, "seqlens_k"),
        (9, DataType::Int64, "total_sequence_length"),
        (10, DataType::Float32, "head_sink"),
    ] {
        require_claim_dtype(input_dtypes, index, expected, name)?;
    }

    let heads = required_claim_attr(node, "num_heads")?;
    let stored_width = if cache_format == "f32" { 512 } else { 583 };
    for (index, name, contract) in [
        (0, "query", vec![Any, NonZero, Fixed(heads), Fixed(512)]),
        (1, "current_kv", vec![Same(0, 0), Any, Fixed(512)]),
        (2, "compressor_kv", vec![Same(0, 0), Same(0, 1), Fixed(512)]),
        (
            3,
            "compressor_gate",
            vec![Same(0, 0), Same(0, 1), Fixed(512)],
        ),
        (4, "compressor_ape", vec![Fixed(128), Fixed(512)]),
        (5, "compressor_norm", vec![Fixed(512)]),
        (
            6,
            "past_compressed_kv",
            vec![Same(0, 0), Any, Fixed(stored_width)],
        ),
        (
            7,
            "past_compression_carry",
            vec![Same(0, 0), Fixed(128), Fixed(2), Fixed(512)],
        ),
        (8, "seqlens_k", vec![Same(0, 0)]),
        (9, "total_sequence_length", vec![]),
        (10, "head_sink", vec![Fixed(heads)]),
    ] {
        require_claim_shape(shapes, index, name, &contract)?;
    }
    Ok(())
}

fn validate_attention_bias_claim(
    node: &Node,
    shapes: &[Shape],
    input_dtypes: &[DataType],
) -> std::result::Result<(), String> {
    if !node.inputs.get(19).is_some_and(Option::is_some) {
        return Ok(());
    }

    require_claim_dtype(input_dtypes, 19, DataType::Float32, "attention_bias")?;
    let bias_shape = &shapes[19];
    if bias_shape.len() > 4 {
        return Err(format!(
            "input 19 ('attention_bias') rank {} unsupported; expected rank <= 4",
            bias_shape.len()
        ));
    }

    if let Some(static_shape) = as_static_shape(bias_shape) {
        let elements = static_shape
            .iter()
            .try_fold(1usize, |count, &dimension| count.checked_mul(dimension));
        if elements
            .and_then(|count| count.checked_mul(std::mem::size_of::<f32>()))
            .is_none_or(|bytes| bytes > isize::MAX as usize)
        {
            return Err(format!(
                "input 19 ('attention_bias') byte count overflow or exceeds isize::MAX for shape {static_shape:?}"
            ));
        }
    }

    let heads = required_claim_attr(node, "num_heads")?;
    let target = [
        shapes[0][0].as_static(),
        Some(heads),
        shapes[0][1].as_static(),
        None,
    ];
    let offset = 4 - bias_shape.len();
    for (axis, dimension) in bias_shape.iter().enumerate() {
        let Some(got) = dimension.as_static() else {
            continue;
        };
        let target_axis = offset + axis;
        if got != 1 && target[target_axis].is_some_and(|expected| got != expected) {
            return Err(format!(
                "input 19 ('attention_bias') shape {bias_shape:?} is not broadcastable to attention scores [{:?}, {heads}, {:?}, ?]",
                shapes[0][0], shapes[0][1]
            ));
        }
    }
    Ok(())
}

fn require_fixed_claim_contract(node: &Node, ratio: usize) -> std::result::Result<(), String> {
    if required_claim_attr(node, "head_dim")? != 512 {
        return Err(format!("ratio-{ratio} requires head_dim=512"));
    }
    let rope_dim = match node.attr("qk_rope_head_dim") {
        Some(attribute) => attribute
            .as_int()
            .ok_or_else(|| "qk_rope_head_dim must be an integer".to_string())?,
        None => 0,
    };
    if rope_dim != 64 {
        return Err(format!("ratio-{ratio} requires qk_rope_head_dim=64"));
    }
    Ok(())
}

fn required_claim_attr(node: &Node, name: &str) -> std::result::Result<usize, String> {
    node.attr(name)
        .and_then(|attribute| attribute.as_int())
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| format!("missing or invalid integer attribute '{name}'"))
}

fn require_claim_dtype(
    input_dtypes: &[DataType],
    index: usize,
    expected: DataType,
    name: &str,
) -> std::result::Result<(), String> {
    let got = input_dtypes[index];
    if got != expected {
        return Err(format!(
            "input {index} ('{name}') dtype {got:?} unsupported; expected {expected:?}"
        ));
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum ClaimShapeAxis {
    Any,
    NonZero,
    Fixed(usize),
    Same(usize, usize),
}
use ClaimShapeAxis::{Any, Fixed, NonZero, Same};

fn require_claim_shape(
    shapes: &[Shape],
    index: usize,
    name: &str,
    contract: &[ClaimShapeAxis],
) -> std::result::Result<(), String> {
    let shape = &shapes[index];
    if shape.len() != contract.len() {
        return Err(format!(
            "input {index} ('{name}') rank {} unsupported; expected {}",
            shape.len(),
            contract.len()
        ));
    }
    for (axis, requirement) in contract.iter().enumerate() {
        let mismatch = match requirement {
            Any => None,
            NonZero if shape[axis] == Dim::Static(0) => Some("must be nonzero".into()),
            NonZero => None,
            Fixed(expected) => shape[axis]
                .as_static()
                .filter(|got| got != expected)
                .map(|got| format!("is {got}; expected {expected}")),
            Same(other_input, other_axis) => {
                match (
                    shape[axis].as_static(),
                    shapes[*other_input][*other_axis].as_static(),
                ) {
                    (Some(got), Some(expected)) if got != expected => {
                        Some(format!("is {got}; expected {expected}"))
                    }
                    _ => None,
                }
            }
        };
        if let Some(mismatch) = mismatch {
            return Err(format!("input {index} ('{name}') axis {axis} {mismatch}"));
        }
    }
    Ok(())
}

impl Kernel for StatefulCompressedSparseAttentionKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        validate_frozen_v1_runtime_arity(inputs, outputs)?;
        for (index, name) in FROZEN_V1_REQUIRED_INPUT_NAMES.iter().enumerate() {
            if inputs[index].is_absent() {
                return Err(error(format!(
                    "required frozen-v1 input {index} ('{name}') is absent"
                )));
            }
        }
        if self.compression_ratio == 4 {
            return self.execute_ratio4(inputs, outputs);
        }
        self.execute_ratio128(inputs, outputs)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

impl StatefulCompressedSparseAttentionKernel {
    fn execute_ratio4(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        const RATIO: usize = 4;
        const CARRY_SLOTS: usize = 8;

        if self.cache_format != CacheFormat::Fp8E4m3Block64 {
            return Err(unsupported(
                "ratio-4 attention records require hybrid FP8/BF16 cache_format='fp8_e4m3_block64'; the index-key stream is independently fixed to FP4 E2M1 block-32",
            ));
        }
        if inputs.len() < 19 || inputs[11..19].iter().any(TensorView::is_absent) {
            return Err(error(
                "ratio-4 requires all eight optional index inputs (11..=18)",
            ));
        }
        if outputs.len() < 5 {
            return Err(error(
                "ratio-4 requires present_index_key and present_index_carry outputs",
            ));
        }
        for (name, input) in [
            ("query", &inputs[0]),
            ("current_kv", &inputs[1]),
            ("compressor_kv", &inputs[2]),
            ("compressor_gate", &inputs[3]),
            ("compressor_ape", &inputs[4]),
            ("compressor_norm", &inputs[5]),
            ("past_compression_carry", &inputs[7]),
            ("head_sink", &inputs[10]),
            ("index_query", &inputs[11]),
            ("index_weight", &inputs[12]),
            ("index_compressor_kv", &inputs[13]),
            ("index_compressor_gate", &inputs[14]),
            ("index_compressor_ape", &inputs[15]),
            ("index_compressor_norm", &inputs[16]),
            ("past_index_carry", &inputs[18]),
        ] {
            require_dtype(name, input.dtype, DataType::Float32)?;
        }
        require_dtype("past_compressed_kv", inputs[6].dtype, DataType::Uint8)?;
        require_dtype("past_index_key", inputs[17].dtype, DataType::Uint8)?;
        require_dtype("seqlens_k", inputs[8].dtype, DataType::Int32)?;
        require_dtype("total_sequence_length", inputs[9].dtype, DataType::Int64)?;
        require_dtype("Y", outputs[0].dtype, DataType::Float32)?;
        require_dtype("present_compressed_kv", outputs[1].dtype, DataType::Uint8)?;
        require_dtype(
            "present_compression_carry",
            outputs[2].dtype,
            DataType::Float32,
        )?;
        require_dtype("present_index_key", outputs[3].dtype, DataType::Uint8)?;
        require_dtype("present_index_carry", outputs[4].dtype, DataType::Float32)?;
        if outputs.len() == 6 {
            require_dtype("selected_indices", outputs[5].dtype, DataType::Int32)?;
        }

        let query_shape = shape4("query", inputs[0].shape)?;
        let [batch, sequence, heads, dim] = query_shape;
        if dim != 512 || self.qk_rope_head_dim != 64 {
            return Err(unsupported(format!(
                "ratio-4 stateful compression is frozen to D=512 and RD=64, got D={dim}, RD={}",
                self.qk_rope_head_dim
            )));
        }
        if self.index_head_dim != 128 {
            return Err(unsupported(format!(
                "ratio-4 index compression is frozen to ID=128, got {}",
                self.index_head_dim
            )));
        }
        if heads != self.num_heads || dim != self.head_dim || sequence == 0 {
            return Err(error(format!(
                "query must be non-empty and end in [num_heads={}, head_dim={}], got {:?}",
                self.num_heads, self.head_dim, inputs[0].shape
            )));
        }
        let current_kv_shape = shape3("current_kv", inputs[1].shape)?;
        let compressor_width = dim
            .checked_mul(2)
            .ok_or_else(|| error("ratio-4 compressor width overflow"))?;
        let index_compressor_width = self
            .index_head_dim
            .checked_mul(2)
            .ok_or_else(|| error("ratio-4 index compressor width overflow"))?;
        if current_kv_shape[0] != batch || current_kv_shape[2] != dim {
            return Err(error(format!(
                "current_kv must have shape [B,K,D] with B={batch}, D={dim}, got {:?}",
                inputs[1].shape
            )));
        }
        for (name, input, expected) in [
            (
                "compressor_kv",
                &inputs[2],
                [batch, sequence, compressor_width],
            ),
            (
                "compressor_gate",
                &inputs[3],
                [batch, sequence, compressor_width],
            ),
            (
                "index_compressor_kv",
                &inputs[13],
                [batch, sequence, index_compressor_width],
            ),
            (
                "index_compressor_gate",
                &inputs[14],
                [batch, sequence, index_compressor_width],
            ),
        ] {
            if shape3(name, input.shape)? != expected {
                return Err(error(format!(
                    "{name} must have shape {expected:?}, got {:?}",
                    input.shape
                )));
            }
        }
        if inputs[4].shape != [RATIO, compressor_width]
            || inputs[5].shape != [dim]
            || inputs[7].shape != [batch, CARRY_SLOTS, 2, compressor_width]
            || inputs[11].shape != [batch, sequence, self.index_num_heads, self.index_head_dim]
            || inputs[12].shape != [batch, sequence, self.index_num_heads]
            || inputs[15].shape != [RATIO, index_compressor_width]
            || inputs[16].shape != [self.index_head_dim]
            || inputs[18].shape != [batch, CARRY_SLOTS, 2, index_compressor_width]
        {
            return Err(error(format!(
                "ratio-4 compressor/index tensor shape mismatch: ape={:?}, norm={:?}, carry={:?}, index_query={:?}, index_weight={:?}, index_ape={:?}, index_norm={:?}, index_carry={:?}",
                inputs[4].shape,
                inputs[5].shape,
                inputs[7].shape,
                inputs[11].shape,
                inputs[12].shape,
                inputs[15].shape,
                inputs[16].shape,
                inputs[18].shape
            )));
        }
        if inputs[8].shape != [batch] || !inputs[9].shape.is_empty() {
            return Err(error(format!(
                "seqlens_k must be [{batch}] and total_sequence_length scalar, got {:?} and {:?}",
                inputs[8].shape, inputs[9].shape
            )));
        }
        if inputs[10].shape != [heads] {
            return Err(error(format!(
                "head_sink must have shape [{heads}], got {:?}",
                inputs[10].shape
            )));
        }

        let seqlens = to_dense_i64(&inputs[8])?;
        let total_values = to_dense_i64(&inputs[9])?;
        let total_i64 = *total_values
            .first()
            .filter(|_| total_values.len() == 1)
            .ok_or_else(|| error("total_sequence_length must contain exactly one value"))?;
        let total = checked_position(total_i64, "total_sequence_length")?;
        for (b, &seqlen) in seqlens.iter().enumerate() {
            let row_total = seqlen
                .checked_add(1)
                .ok_or_else(|| error(format!("seqlens_k[{b}] + 1 overflows")))?;
            if checked_position(row_total, &format!("seqlens_k[{b}] + 1"))? != total {
                return Err(error(format!(
                    "v1 requires total_sequence_length == seqlens_k[b] + 1; row {b} gives {row_total}, total is {total}"
                )));
            }
        }
        let start = total.checked_sub(sequence).ok_or_else(|| {
            error("total_sequence_length is shorter than the current query sequence")
        })?;
        let current_kv_len = current_kv_shape[1];
        let current_kv_base = total
            .checked_sub(current_kv_len)
            .ok_or_else(|| error("current_kv length exceeds total_sequence_length"))?;
        let earliest_needed = start
            .checked_add(1)
            .ok_or_else(|| error("earliest query position overflow"))?
            .saturating_sub(128);
        if current_kv_base > earliest_needed {
            return Err(error(format!(
                "current_kv starts at {current_kv_base}, but the earliest query needs dense-window position {earliest_needed}"
            )));
        }

        let past_records = start / RATIO;
        let next_records = total / RATIO;
        let emitted_per_batch = next_records
            .checked_sub(past_records)
            .ok_or_else(|| error("ratio-4 compressed record count underflow"))?;
        let stored_width = self.cache_format.stored_width(dim, self.qk_rope_head_dim)?;
        let index_stored_width =
            CacheFormat::Fp4E2m1Block32.stored_width(self.index_head_dim, 0)?;
        if shape3("past_compressed_kv", inputs[6].shape)? != [batch, past_records, stored_width]
            || shape3("past_index_key", inputs[17].shape)?
                != [batch, past_records, index_stored_width]
        {
            return Err(error(format!(
                "ratio-4 past cache shapes must be [{batch},{past_records},{stored_width}] and [{batch},{past_records},{index_stored_width}], got {:?} and {:?}",
                inputs[6].shape, inputs[17].shape
            )));
        }
        let topk_width = self.index_topk.min(next_records);
        let expected_y = [batch, sequence, heads, dim];
        let expected_cache = [batch, next_records, stored_width];
        let expected_carry = [batch, CARRY_SLOTS, 2, compressor_width];
        let expected_index = [batch, next_records, index_stored_width];
        let expected_index_carry = [batch, CARRY_SLOTS, 2, index_compressor_width];
        if outputs[0].shape != expected_y
            || outputs[1].shape != expected_cache
            || outputs[2].shape != expected_carry
            || outputs[3].shape != expected_index
            || outputs[4].shape != expected_index_carry
        {
            return Err(error(format!(
                "ratio-4 output shape mismatch: expected Y={expected_y:?}, cache={expected_cache:?}, carry={expected_carry:?}, index={expected_index:?}, index_carry={expected_index_carry:?}; got {:?}, {:?}, {:?}, {:?}, {:?}",
                outputs[0].shape,
                outputs[1].shape,
                outputs[2].shape,
                outputs[3].shape,
                outputs[4].shape
            )));
        }
        if outputs.len() == 6
            && outputs[5].shape != [batch, self.index_num_heads, sequence, topk_width]
        {
            return Err(error(format!(
                "selected_indices must have shape [{batch},{},{sequence},{topk_width}]; each index-head row repeats the shared selection, got {:?}",
                self.index_num_heads, outputs[5].shape
            )));
        }

        let query = read_dense_f32(&inputs[0], "query")?;
        let current_kv = read_dense_f32(&inputs[1], "current_kv")?;
        let compressor_kv = read_dense_f32(&inputs[2], "compressor_kv")?;
        let compressor_gate = read_dense_f32(&inputs[3], "compressor_gate")?;
        let compressor_ape = read_dense_f32(&inputs[4], "compressor_ape")?;
        let compressor_norm = read_dense_f32(&inputs[5], "compressor_norm")?;
        let mut compression_carry = read_dense_f32(&inputs[7], "past_compression_carry")?;
        let sink = read_dense_f32(&inputs[10], "head_sink")?;
        let index_query = read_dense_f32(&inputs[11], "index_query")?;
        let index_weight = read_dense_f32(&inputs[12], "index_weight")?;
        let index_compressor_kv = read_dense_f32(&inputs[13], "index_compressor_kv")?;
        let index_compressor_gate = read_dense_f32(&inputs[14], "index_compressor_gate")?;
        let index_compressor_ape = read_dense_f32(&inputs[15], "index_compressor_ape")?;
        let index_compressor_norm = read_dense_f32(&inputs[16], "index_compressor_norm")?;
        let mut index_carry = read_dense_f32(&inputs[18], "past_index_carry")?;
        for (name, values) in [
            ("query", query.as_slice()),
            ("current_kv", current_kv.as_slice()),
            ("compressor_kv", compressor_kv.as_slice()),
            ("compressor_gate", compressor_gate.as_slice()),
            ("compressor_ape", compressor_ape.as_slice()),
            ("compressor_norm", compressor_norm.as_slice()),
            ("head_sink", sink.as_slice()),
            ("index_query", index_query.as_slice()),
            ("index_weight", index_weight.as_slice()),
            ("index_compressor_kv", index_compressor_kv.as_slice()),
            ("index_compressor_gate", index_compressor_gate.as_slice()),
            ("index_compressor_ape", index_compressor_ape.as_slice()),
            ("index_compressor_norm", index_compressor_norm.as_slice()),
        ] {
            require_finite(name, values)?;
        }
        validate_carry(&compression_carry, batch, CARRY_SLOTS, compressor_width)?;
        validate_carry(&index_carry, batch, CARRY_SLOTS, index_compressor_width)?;
        if start == 0 {
            reset_ratio4_carry(&mut compression_carry, batch, CARRY_SLOTS, compressor_width)?;
            reset_ratio4_carry(&mut index_carry, batch, CARRY_SLOTS, index_compressor_width)?;
        }

        let past_logical = dequantize_cache(
            &inputs[6],
            [batch, 1, past_records, stored_width],
            dim,
            self.qk_rope_head_dim,
            self.cache_format,
        )?;
        let past_index_logical = dequantize_cache(
            &inputs[17],
            [batch, 1, past_records, index_stored_width],
            self.index_head_dim,
            0,
            CacheFormat::Fp4E2m1Block32,
        )?;
        let past_packed = to_dense_bytes(&inputs[6])?;
        let past_index_packed = to_dense_bytes(&inputs[17])?;

        let attention_stream = compress_ratio4_stream(
            &compressor_kv,
            &compressor_gate,
            &compressor_ape,
            &compressor_norm,
            &mut compression_carry,
            batch,
            sequence,
            start,
            dim,
            stored_width,
            emitted_per_batch,
            |pooled, norm, block_start| {
                let (logical, packed) = finalize_attention_record(
                    pooled,
                    norm,
                    dim,
                    self.qk_rope_head_dim,
                    block_start,
                    self.cache_format,
                )?;
                Ok((
                    logical,
                    packed.ok_or_else(|| error("missing packed ratio-4 attention record"))?,
                ))
            },
        )?;
        let index_stream = compress_ratio4_stream(
            &index_compressor_kv,
            &index_compressor_gate,
            &index_compressor_ape,
            &index_compressor_norm,
            &mut index_carry,
            batch,
            sequence,
            start,
            self.index_head_dim,
            index_stored_width,
            emitted_per_batch,
            |pooled, norm, block_start| {
                finalize_index_record(
                    pooled,
                    norm,
                    self.index_head_dim,
                    self.qk_rope_head_dim,
                    block_start,
                )
            },
        )?;
        let all_logical = combine_logical_records(
            &past_logical,
            &attention_stream.logical,
            batch,
            past_records,
            emitted_per_batch,
            dim,
        )?;
        let all_index_logical = combine_logical_records(
            &past_index_logical,
            &index_stream.logical,
            batch,
            past_records,
            emitted_per_batch,
            self.index_head_dim,
        )?;
        let present_cache = combine_packed_records(
            &past_packed,
            &attention_stream.packed,
            batch,
            past_records,
            emitted_per_batch,
            stored_width,
        )?;
        let present_index = combine_packed_records(
            &past_index_packed,
            &index_stream.packed,
            batch,
            past_records,
            emitted_per_batch,
            index_stored_width,
        )?;

        let selected = select_ratio4_topk(
            &index_query,
            &index_weight,
            &all_index_logical,
            [batch, sequence, self.index_num_heads, self.index_head_dim],
            next_records,
            start,
            topk_width,
            self.qk_rope_head_dim,
        )?;
        let dense_candidates = if start == 0 {
            current_kv_len.min(128)
        } else {
            128
        };
        let attention_candidates = dense_candidates
            .checked_add(topk_width)
            .ok_or_else(|| error("ratio-4 attention candidate count overflow"))?;
        let attention_bias = inputs
            .get(19)
            .filter(|input| !input.is_absent())
            .map(|input| AttentionBias::new(input, [batch, heads, sequence, attention_candidates]))
            .transpose()?;
        let output = ratio4_attention(
            &query,
            query_shape,
            &current_kv,
            current_kv_shape,
            current_kv_base,
            &all_logical,
            next_records,
            &selected,
            topk_width,
            start,
            dense_candidates,
            &sink,
            self.scale,
            attention_bias.as_ref(),
        )?;

        write_dense_f32(&mut outputs[0], &output)?;
        write_dense_bytes(&mut outputs[1], &present_cache)?;
        write_dense_f32(&mut outputs[2], &compression_carry)?;
        write_dense_bytes(&mut outputs[3], &present_index)?;
        write_dense_f32(&mut outputs[4], &index_carry)?;
        if outputs.len() == 6 {
            write_shared_selected_i32(
                &mut outputs[5],
                &selected,
                batch,
                self.index_num_heads,
                sequence,
                topk_width,
            )?;
        }
        Ok(())
    }

    fn execute_ratio128(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if self.cache_format == CacheFormat::Fp4E2m1Block32 {
            return Err(unsupported(
                "ratio-128 attention-compressor state uses hybrid FP8/BF16 records, not FP4",
            ));
        }
        for (index, input) in inputs.iter().enumerate().take(19).skip(11) {
            if !input.is_absent() {
                return Err(unsupported(format!(
                    "ratio-4-only input {index} is not supported by the ratio-128 stateful path"
                )));
            }
        }
        if outputs.len() > FROZEN_V1_REQUIRED_OUTPUTS {
            return Err(unsupported(
                "ratio-4 state and diagnostic outputs are not implemented",
            ));
        }
        for (name, input) in [
            ("query", &inputs[0]),
            ("current_kv", &inputs[1]),
            ("compressor_kv", &inputs[2]),
            ("compressor_gate", &inputs[3]),
            ("compressor_ape", &inputs[4]),
            ("compressor_norm", &inputs[5]),
            ("past_compression_carry", &inputs[7]),
            ("head_sink", &inputs[10]),
        ] {
            require_dtype(name, input.dtype, DataType::Float32)?;
        }
        require_dtype(
            "past_compressed_kv",
            inputs[6].dtype,
            self.cache_format.dtype(),
        )?;
        require_dtype("seqlens_k", inputs[8].dtype, DataType::Int32)?;
        require_dtype("total_sequence_length", inputs[9].dtype, DataType::Int64)?;
        require_dtype("Y", outputs[0].dtype, DataType::Float32)?;
        require_dtype(
            "present_compressed_kv",
            outputs[1].dtype,
            self.cache_format.dtype(),
        )?;
        require_dtype(
            "present_compression_carry",
            outputs[2].dtype,
            DataType::Float32,
        )?;

        let query_shape = shape4("query", inputs[0].shape)?;
        let [batch, sequence, heads, dim] = query_shape;
        if dim != 512 || self.qk_rope_head_dim != 64 {
            return Err(unsupported(format!(
                "ratio-128 stateful compression is frozen to D=512 and RD=64, got D={dim}, RD={}",
                self.qk_rope_head_dim
            )));
        }
        if heads != self.num_heads || dim != self.head_dim {
            return Err(error(format!(
                "query must end in [num_heads={}, head_dim={}], got [{heads}, {dim}]",
                self.num_heads, self.head_dim
            )));
        }
        if sequence == 0 {
            return Err(error("query sequence length must be positive"));
        }
        let current_kv_shape = shape3("current_kv", inputs[1].shape)?;
        let compressor_kv_shape = shape3("compressor_kv", inputs[2].shape)?;
        let compressor_gate_shape = shape3("compressor_gate", inputs[3].shape)?;
        if current_kv_shape[0] != batch
            || current_kv_shape[2] != dim
            || compressor_kv_shape != [batch, sequence, dim]
            || compressor_gate_shape != [batch, sequence, dim]
        {
            return Err(error(format!(
                "current/compressor tensors must have current_kv=[B,K,D] and compressor_kv/gate=[B,S,D]; got {:?}, {:?}, {:?}",
                inputs[1].shape, inputs[2].shape, inputs[3].shape
            )));
        }
        if inputs[4].shape != [self.compression_ratio, dim] {
            return Err(error(format!(
                "compressor_ape must have shape [{}, {dim}], got {:?}",
                self.compression_ratio, inputs[4].shape
            )));
        }
        if inputs[5].shape != [dim] {
            return Err(error(format!(
                "compressor_norm must have shape [{dim}], got {:?}",
                inputs[5].shape
            )));
        }
        if inputs[7].shape != [batch, self.compression_ratio, 2, dim] {
            return Err(error(format!(
                "past_compression_carry must have shape [B,128,2,D]=[{batch},{},2,{dim}], got {:?}",
                self.compression_ratio, inputs[7].shape
            )));
        }
        if inputs[8].shape != [batch] {
            return Err(error(format!(
                "seqlens_k must have shape [{batch}], got {:?}",
                inputs[8].shape
            )));
        }
        if !inputs[9].shape.is_empty() {
            return Err(error(format!(
                "total_sequence_length must be a scalar, got {:?}",
                inputs[9].shape
            )));
        }
        if inputs[10].shape != [heads] {
            return Err(error(format!(
                "head_sink must have shape [{heads}], got {:?}",
                inputs[10].shape
            )));
        }
        for (shape, element_size, name) in [
            (inputs[0].shape, std::mem::size_of::<f32>(), "query"),
            (inputs[1].shape, std::mem::size_of::<f32>(), "current_kv"),
            (inputs[2].shape, std::mem::size_of::<f32>(), "compressor_kv"),
            (
                inputs[3].shape,
                std::mem::size_of::<f32>(),
                "compressor_gate",
            ),
            (
                inputs[7].shape,
                std::mem::size_of::<f32>(),
                "past_compression_carry",
            ),
        ] {
            checked_layout(shape, element_size, name)?;
        }

        let seqlens = to_dense_i64(&inputs[8])?;
        let total_values = to_dense_i64(&inputs[9])?;
        let total_i64 = *total_values
            .first()
            .filter(|_| total_values.len() == 1)
            .ok_or_else(|| error("total_sequence_length must contain exactly one value"))?;
        let total = checked_position(total_i64, "total_sequence_length")?;
        for (b, &seqlen) in seqlens.iter().enumerate() {
            let row_total = seqlen
                .checked_add(1)
                .ok_or_else(|| error(format!("seqlens_k[{b}] + 1 overflows")))?;
            if checked_position(row_total, &format!("seqlens_k[{b}] + 1"))? != total {
                return Err(error(format!(
                    "v1 requires equal-length rows and total_sequence_length == seqlens_k[b] + 1; row {b} gives {row_total}, total is {total}"
                )));
            }
        }
        let start = total.checked_sub(sequence).ok_or_else(|| {
            error("total_sequence_length is shorter than the current query sequence")
        })?;
        let current_kv_len = current_kv_shape[1];
        let current_kv_base = total
            .checked_sub(current_kv_len)
            .ok_or_else(|| error("current_kv length exceeds total_sequence_length"))?;
        let earliest_needed = start
            .checked_add(1)
            .ok_or_else(|| error("earliest query position overflow"))?
            .saturating_sub(128);
        if current_kv_base > earliest_needed {
            return Err(error(format!(
                "current_kv starts at absolute position {current_kv_base}, but the earliest query needs dense-window position {earliest_needed}"
            )));
        }

        let stored_width = self.cache_format.stored_width(dim, self.qk_rope_head_dim)?;
        let past_cache_shape = shape3("past_compressed_kv", inputs[6].shape)?;
        let past_records = start / self.compression_ratio;
        if past_cache_shape != [batch, past_records, stored_width] {
            return Err(error(format!(
                "past_compressed_kv must have shape [{batch},{past_records},{stored_width}] at start position {start}, got {:?}",
                inputs[6].shape
            )));
        }
        let next_records = total / self.compression_ratio;
        let emitted_per_batch = next_records
            .checked_sub(past_records)
            .ok_or_else(|| error("compressed record count underflow"))?;
        let expected_y = [batch, sequence, heads, dim];
        let expected_cache = [batch, next_records, stored_width];
        let expected_carry = [batch, self.compression_ratio, 2, dim];
        if outputs[0].shape != expected_y {
            return Err(error(format!(
                "Y must have shape {expected_y:?}, got {:?}",
                outputs[0].shape
            )));
        }
        if outputs[1].shape != expected_cache {
            return Err(error(format!(
                "present_compressed_kv must have shape {expected_cache:?}, got {:?}",
                outputs[1].shape
            )));
        }
        if outputs[2].shape != expected_carry {
            return Err(error(format!(
                "present_compression_carry must have shape {expected_carry:?}, got {:?}",
                outputs[2].shape
            )));
        }
        checked_layout(&expected_y, std::mem::size_of::<f32>(), "Y")?;
        checked_layout(
            &expected_cache,
            if self.cache_format == CacheFormat::F32 {
                std::mem::size_of::<f32>()
            } else {
                std::mem::size_of::<u8>()
            },
            "present_compressed_kv",
        )?;
        checked_layout(
            &expected_carry,
            std::mem::size_of::<f32>(),
            "present_compression_carry",
        )?;
        checked_layout(
            &[batch, next_records, dim],
            std::mem::size_of::<f32>(),
            "logical compressed cache workspace",
        )?;

        let query = read_dense_f32(&inputs[0], "query")?;
        let current_kv = read_dense_f32(&inputs[1], "current_kv")?;
        let compressor_kv = read_dense_f32(&inputs[2], "compressor_kv")?;
        let compressor_gate = read_dense_f32(&inputs[3], "compressor_gate")?;
        let compressor_ape = read_dense_f32(&inputs[4], "compressor_ape")?;
        let compressor_norm = read_dense_f32(&inputs[5], "compressor_norm")?;
        let mut carry = read_dense_f32(&inputs[7], "past_compression_carry")?;
        let sink = read_dense_f32(&inputs[10], "head_sink")?;
        require_finite("query", &query)?;
        require_finite("current_kv", &current_kv)?;
        require_finite("compressor_kv", &compressor_kv)?;
        require_finite("compressor_gate", &compressor_gate)?;
        require_finite("compressor_ape", &compressor_ape)?;
        require_finite("compressor_norm", &compressor_norm)?;
        require_finite("head_sink", &sink)?;
        validate_carry(&carry, batch, self.compression_ratio, dim)?;
        if start == 0 {
            reset_ratio128_carry(&mut carry, batch, self.compression_ratio, dim)?;
        }

        let past_logical = dequantize_cache(
            &inputs[6],
            [batch, 1, past_records, stored_width],
            dim,
            self.qk_rope_head_dim,
            self.cache_format,
        )?;
        let past_packed = if self.cache_format == CacheFormat::Fp8E4m3Block64 {
            Some(to_dense_bytes(&inputs[6])?)
        } else {
            None
        };
        let mut emitted_logical = fallible_filled(
            checked_product(
                &[batch, emitted_per_batch, dim],
                "emitted compressed records",
            )?,
            0.0f32,
            "emitted compressed records",
        )?;
        let mut emitted_packed = if self.cache_format == CacheFormat::Fp8E4m3Block64 {
            Some(fallible_filled(
                checked_product(
                    &[batch, emitted_per_batch, stored_width],
                    "emitted packed records",
                )?,
                0u8,
                "emitted packed records",
            )?)
        } else {
            None
        };
        let mut emitted_counts = fallible_filled(batch, 0usize, "per-batch emitted counts")?;

        for (b, emitted_count) in emitted_counts.iter_mut().enumerate() {
            for s in 0..sequence {
                let position = start
                    .checked_add(s)
                    .filter(|&value| value <= isize::MAX as usize)
                    .ok_or_else(|| error("absolute compression position overflow"))?;
                let slot = position % self.compression_ratio;
                let source_row = flat3([b, s, 0], [batch, sequence, dim], "compressor source row")?;
                let ape_row = slot
                    .checked_mul(dim)
                    .ok_or_else(|| error("compressor APE row offset overflow"))?;
                let source_end = source_row
                    .checked_add(dim)
                    .ok_or_else(|| error("compressor source row end overflow"))?;
                let ape_end = ape_row
                    .checked_add(dim)
                    .ok_or_else(|| error("compressor APE row end overflow"))?;
                let kv_source = compressor_kv
                    .get(source_row..source_end)
                    .ok_or_else(|| error("compressor_kv row is out of bounds"))?;
                let gate_source = compressor_gate
                    .get(source_row..source_end)
                    .ok_or_else(|| error("compressor_gate row is out of bounds"))?;
                let ape_source = compressor_ape
                    .get(ape_row..ape_end)
                    .ok_or_else(|| error("compressor_ape row is out of bounds"))?;
                for d in 0..dim {
                    let kv_offset = carry_offset(b, slot, 0, d, self.compression_ratio, dim)?;
                    let score_offset = carry_offset(b, slot, 1, d, self.compression_ratio, dim)?;
                    carry[kv_offset] = kv_source[d];
                    carry[score_offset] = gate_source[d] + ape_source[d];
                }
                let boundary = position
                    .checked_add(1)
                    .ok_or_else(|| error("compression boundary position overflow"))?
                    .is_multiple_of(self.compression_ratio);
                if boundary {
                    let block_start = position
                        .checked_add(1)
                        .and_then(|value| value.checked_sub(self.compression_ratio))
                        .ok_or_else(|| error("compressed block start underflow"))?;
                    let pooled = pool_ratio128_record(&carry, b, self.compression_ratio, dim)?;
                    let (logical, packed) = finalize_attention_record(
                        &pooled,
                        &compressor_norm,
                        dim,
                        self.qk_rope_head_dim,
                        block_start,
                        self.cache_format,
                    )?;
                    let emitted_index = *emitted_count;
                    if emitted_index >= emitted_per_batch {
                        return Err(error("emitted more compressed records than expected"));
                    }
                    let logical_offset = b
                        .checked_mul(emitted_per_batch)
                        .and_then(|value| value.checked_add(emitted_index))
                        .and_then(|value| value.checked_mul(dim))
                        .ok_or_else(|| error("emitted logical record offset overflow"))?;
                    let logical_end = logical_offset
                        .checked_add(dim)
                        .ok_or_else(|| error("emitted logical record end overflow"))?;
                    emitted_logical
                        .get_mut(logical_offset..logical_end)
                        .ok_or_else(|| error("emitted logical record is out of bounds"))?
                        .copy_from_slice(&logical);
                    if let (Some(destination), Some(source)) =
                        (emitted_packed.as_mut(), packed.as_deref())
                    {
                        let packed_offset = b
                            .checked_mul(emitted_per_batch)
                            .and_then(|value| value.checked_add(emitted_index))
                            .and_then(|value| value.checked_mul(stored_width))
                            .ok_or_else(|| error("emitted packed record offset overflow"))?;
                        let packed_end = packed_offset
                            .checked_add(stored_width)
                            .ok_or_else(|| error("emitted packed record end overflow"))?;
                        destination
                            .get_mut(packed_offset..packed_end)
                            .ok_or_else(|| error("emitted packed record is out of bounds"))?
                            .copy_from_slice(source);
                    }
                    *emitted_count = emitted_index
                        .checked_add(1)
                        .ok_or_else(|| error("emitted record count overflow"))?;
                    reset_ratio128_row(&mut carry, b, self.compression_ratio, dim)?;
                }
            }
        }
        if emitted_counts
            .iter()
            .any(|&count| count != emitted_per_batch)
        {
            return Err(error(format!(
                "compressed record emission mismatch: expected {emitted_per_batch} per batch, got {emitted_counts:?}"
            )));
        }

        let all_logical = combine_logical_records(
            &past_logical,
            &emitted_logical,
            batch,
            past_records,
            emitted_per_batch,
            dim,
        )?;
        let present_cache = match self.cache_format {
            CacheFormat::F32 => CacheOutput::F32(all_logical.clone()),
            CacheFormat::Fp8E4m3Block64 => CacheOutput::U8(combine_packed_records(
                past_packed
                    .as_deref()
                    .ok_or_else(|| error("missing packed past cache"))?,
                emitted_packed
                    .as_deref()
                    .ok_or_else(|| error("missing packed emitted cache"))?,
                batch,
                past_records,
                emitted_per_batch,
                stored_width,
            )?),
            CacheFormat::Fp4E2m1Block32 => {
                return Err(error(
                    "ratio-128 state unexpectedly reached FP4 cache output",
                ));
            }
        };
        let dense_candidates = if start == 0 {
            current_kv_len.min(128)
        } else {
            128
        };
        let attention_candidates = dense_candidates
            .checked_add(next_records)
            .ok_or_else(|| error("attention bias candidate count overflow"))?;
        let attention_bias = inputs
            .get(19)
            .filter(|input| !input.is_absent())
            .map(|input| AttentionBias::new(input, [batch, heads, sequence, attention_candidates]))
            .transpose()?;
        let output = ratio128_attention(
            &query,
            query_shape,
            &current_kv,
            current_kv_shape,
            current_kv_base,
            &all_logical,
            next_records,
            start,
            dense_candidates,
            &sink,
            self.scale,
            attention_bias.as_ref(),
        )?;

        write_dense_f32(&mut outputs[0], &output)?;
        match present_cache {
            CacheOutput::F32(values) => write_dense_f32(&mut outputs[1], &values)?,
            CacheOutput::U8(values) => write_dense_bytes(&mut outputs[1], &values)?,
        }
        write_dense_f32(&mut outputs[2], &carry)
    }
}

enum CacheOutput {
    F32(Vec<f32>),
    U8(Vec<u8>),
}

struct Ratio4Stream {
    logical: Vec<f32>,
    packed: Vec<u8>,
}

#[allow(clippy::too_many_arguments)]
fn compress_ratio4_stream<F>(
    source_kv: &[f32],
    source_gate: &[f32],
    ape: &[f32],
    norm: &[f32],
    carry: &mut [f32],
    batch: usize,
    sequence: usize,
    start: usize,
    output_width: usize,
    packed_width: usize,
    emitted_per_batch: usize,
    finalize: F,
) -> Result<Ratio4Stream>
where
    F: Fn(&[f32], &[f32], usize) -> Result<(Vec<f32>, Vec<u8>)>,
{
    const RATIO: usize = 4;
    const CARRY_SLOTS: usize = 8;
    let source_width = output_width
        .checked_mul(2)
        .ok_or_else(|| error("ratio-4 source width overflow"))?;
    let mut logical = fallible_filled(
        checked_product(
            &[batch, emitted_per_batch, output_width],
            "ratio-4 emitted logical records",
        )?,
        0.0f32,
        "ratio-4 emitted logical records",
    )?;
    let mut packed = fallible_filled(
        checked_product(
            &[batch, emitted_per_batch, packed_width],
            "ratio-4 emitted packed records",
        )?,
        0u8,
        "ratio-4 emitted packed records",
    )?;
    let mut emitted_counts = fallible_filled(batch, 0usize, "ratio-4 emitted counts")?;

    for b in 0..batch {
        for s in 0..sequence {
            let position = start
                .checked_add(s)
                .filter(|&value| value <= isize::MAX as usize)
                .ok_or_else(|| error("ratio-4 absolute position overflow"))?;
            let phase = position % RATIO;
            let slot = RATIO
                .checked_add(phase)
                .ok_or_else(|| error("ratio-4 current carry slot overflow"))?;
            let source_row = flat3(
                [b, s, 0],
                [batch, sequence, source_width],
                "ratio-4 compressor source",
            )?;
            let ape_row = phase
                .checked_mul(source_width)
                .ok_or_else(|| error("ratio-4 APE row overflow"))?;
            let source_end = source_row
                .checked_add(source_width)
                .ok_or_else(|| error("ratio-4 compressor source end overflow"))?;
            let ape_end = ape_row
                .checked_add(source_width)
                .ok_or_else(|| error("ratio-4 APE row end overflow"))?;
            let kv_row = source_kv
                .get(source_row..source_end)
                .ok_or_else(|| error("ratio-4 compressor KV row is out of bounds"))?;
            let gate_row = source_gate
                .get(source_row..source_end)
                .ok_or_else(|| error("ratio-4 compressor gate row is out of bounds"))?;
            let ape_row = ape
                .get(ape_row..ape_end)
                .ok_or_else(|| error("ratio-4 compressor APE row is out of bounds"))?;
            for d in 0..source_width {
                carry[carry_offset(b, slot, 0, d, CARRY_SLOTS, source_width)?] = kv_row[d];
                carry[carry_offset(b, slot, 1, d, CARRY_SLOTS, source_width)?] =
                    gate_row[d] + ape_row[d];
            }
            // "Overlap factor 2" is the two-channel 8-slot carry; the source
            // still emits one record at each four-token boundary.
            if !position
                .checked_add(1)
                .ok_or_else(|| error("ratio-4 boundary position overflow"))?
                .is_multiple_of(RATIO)
            {
                continue;
            }

            let block_start = position
                .checked_add(1)
                .and_then(|value| value.checked_sub(RATIO))
                .ok_or_else(|| error("ratio-4 block start underflow"))?;
            let pooled = pool_ratio4_record(carry, b, output_width, source_width, CARRY_SLOTS)?;
            let (record, bytes) = finalize(&pooled, norm, block_start)?;
            if record.len() != output_width || bytes.len() != packed_width {
                return Err(error("ratio-4 finalized record width mismatch"));
            }
            let emitted = emitted_counts[b];
            if emitted >= emitted_per_batch {
                return Err(error("ratio-4 emitted more records than expected"));
            }
            let logical_offset = b
                .checked_mul(emitted_per_batch)
                .and_then(|value| value.checked_add(emitted))
                .and_then(|value| value.checked_mul(output_width))
                .ok_or_else(|| error("ratio-4 logical record offset overflow"))?;
            let packed_offset = b
                .checked_mul(emitted_per_batch)
                .and_then(|value| value.checked_add(emitted))
                .and_then(|value| value.checked_mul(packed_width))
                .ok_or_else(|| error("ratio-4 packed record offset overflow"))?;
            let logical_end = logical_offset
                .checked_add(output_width)
                .ok_or_else(|| error("ratio-4 logical record end overflow"))?;
            let packed_end = packed_offset
                .checked_add(packed_width)
                .ok_or_else(|| error("ratio-4 packed record end overflow"))?;
            logical
                .get_mut(logical_offset..logical_end)
                .ok_or_else(|| error("ratio-4 logical record is out of bounds"))?
                .copy_from_slice(&record);
            packed
                .get_mut(packed_offset..packed_end)
                .ok_or_else(|| error("ratio-4 packed record is out of bounds"))?
                .copy_from_slice(&bytes);
            emitted_counts[b] = emitted
                .checked_add(1)
                .ok_or_else(|| error("ratio-4 emitted count overflow"))?;

            for previous_slot in 0..RATIO {
                let current_slot = RATIO
                    .checked_add(previous_slot)
                    .ok_or_else(|| error("ratio-4 shift slot overflow"))?;
                for state in 0..2 {
                    for d in 0..source_width {
                        let source =
                            carry_offset(b, current_slot, state, d, CARRY_SLOTS, source_width)?;
                        let destination =
                            carry_offset(b, previous_slot, state, d, CARRY_SLOTS, source_width)?;
                        carry[destination] = carry[source];
                        carry[source] = if state == 0 { 0.0 } else { f32::NEG_INFINITY };
                    }
                }
            }
        }
    }
    if emitted_counts
        .iter()
        .any(|&count| count != emitted_per_batch)
    {
        return Err(error(format!(
            "ratio-4 record emission mismatch: expected {emitted_per_batch} per batch, got {emitted_counts:?}"
        )));
    }
    Ok(Ratio4Stream { logical, packed })
}

fn pool_ratio4_record(
    carry: &[f32],
    batch: usize,
    output_width: usize,
    source_width: usize,
    carry_slots: usize,
) -> Result<Vec<f32>> {
    const RATIO: usize = 4;
    let mut pooled = fallible_filled(output_width, 0.0f32, "ratio-4 pooled record")?;
    for (d, destination) in pooled.iter_mut().enumerate() {
        let mut maximum = f32::NEG_INFINITY;
        for candidate in 0..RATIO * 2 {
            let (slot, channel) = if candidate < RATIO {
                (candidate, 0)
            } else {
                (candidate, output_width)
            };
            let source_dim = channel
                .checked_add(d)
                .ok_or_else(|| error("ratio-4 pooled source dimension overflow"))?;
            maximum = maximum
                .max(carry[carry_offset(batch, slot, 1, source_dim, carry_slots, source_width)?]);
        }
        if !maximum.is_finite() {
            return Err(error(format!(
                "ratio-4 compression block has no finite score for dimension {d}"
            )));
        }
        let mut numerator = 0.0f32;
        let mut denominator = 0.0f32;
        for candidate in 0..RATIO * 2 {
            let (slot, channel) = if candidate < RATIO {
                (candidate, 0)
            } else {
                (candidate, output_width)
            };
            let source_dim = channel
                .checked_add(d)
                .ok_or_else(|| error("ratio-4 pooled source dimension overflow"))?;
            let score = carry[carry_offset(batch, slot, 1, source_dim, carry_slots, source_width)?];
            if score == f32::NEG_INFINITY {
                continue;
            }
            let weight = (score - maximum).exp();
            numerator += weight
                * carry[carry_offset(batch, slot, 0, source_dim, carry_slots, source_width)?];
            denominator += weight;
        }
        if denominator == 0.0 || !denominator.is_finite() || !numerator.is_finite() {
            return Err(error(format!(
                "ratio-4 compression softmax is invalid for dimension {d}"
            )));
        }
        *destination = numerator / denominator;
    }
    Ok(pooled)
}

fn reset_ratio4_carry(carry: &mut [f32], batch: usize, slots: usize, width: usize) -> Result<()> {
    for b in 0..batch {
        reset_ratio128_row(carry, b, slots, width)?;
    }
    Ok(())
}

fn hadamard_bf16(values: &mut [f32]) -> Result<()> {
    if !values.len().is_power_of_two() {
        return Err(unsupported(format!(
            "Hadamard index width must be a power of two, got {}",
            values.len()
        )));
    }
    let mut span = 1usize;
    while span < values.len() {
        let step = span
            .checked_mul(2)
            .ok_or_else(|| error("Hadamard step overflow"))?;
        for start in (0..values.len()).step_by(step) {
            for offset in 0..span {
                let left = start
                    .checked_add(offset)
                    .ok_or_else(|| error("Hadamard left offset overflow"))?;
                let right = left
                    .checked_add(span)
                    .ok_or_else(|| error("Hadamard right offset overflow"))?;
                let a = values[left];
                let b = values[right];
                values[left] = a + b;
                values[right] = a - b;
            }
        }
        span = step;
    }
    let scale = 1.0 / (values.len() as f32).sqrt();
    for value in values {
        *value = half::bf16::from_f32(*value * scale).to_f32();
    }
    Ok(())
}

fn fp4_roundtrip(values: &mut [f32]) -> Result<Vec<u8>> {
    if !values.len().is_multiple_of(FP4_E2M1_BLOCK_SIZE) {
        return Err(error(format!(
            "FP4 index width {} must be divisible by {FP4_E2M1_BLOCK_SIZE}",
            values.len()
        )));
    }
    let blocks = values.len() / FP4_E2M1_BLOCK_SIZE;
    let packed_width = blocks
        .checked_mul(FP4_E2M1_PACKED_BYTES + 1)
        .ok_or_else(|| error("FP4 index packed width overflow"))?;
    let mut packed = Vec::with_capacity(packed_width);
    for block in values.chunks_exact_mut(FP4_E2M1_BLOCK_SIZE) {
        let input = block.to_vec();
        let mut exponent = 0u8;
        let mut codes = [0u8; FP4_E2M1_PACKED_BYTES];
        quantize_fp4_e2m1_block(&input, &mut exponent, &mut codes, block)?;
        packed.push(exponent);
        packed.extend_from_slice(&codes);
    }
    Ok(packed)
}

fn finalize_index_record(
    pooled: &[f32],
    norm: &[f32],
    dim: usize,
    rope_dim: usize,
    block_start: usize,
) -> Result<(Vec<f32>, Vec<u8>)> {
    if pooled.len() != dim || norm.len() != dim || rope_dim > dim {
        return Err(error("index record finalization width mismatch"));
    }
    let mut record = pooled
        .iter()
        .map(|&value| half::bf16::from_f32(value).to_f32())
        .collect::<Vec<_>>();
    let square_sum = record.iter().try_fold(0.0f32, |sum, &value| {
        let next = sum + value * value;
        next.is_finite()
            .then_some(next)
            .ok_or_else(|| error("index RMSNorm square sum is non-finite"))
    })?;
    let inverse_rms = (square_sum / dim as f32 + 1.0e-6).sqrt().recip();
    for (value, &weight) in record.iter_mut().zip(norm) {
        *value = half::bf16::from_f32(*value * inverse_rms * weight).to_f32();
    }
    apply_compressed_rope(&mut record[dim - rope_dim..], block_start)?;
    hadamard_bf16(&mut record)?;
    let packed = fp4_roundtrip(&mut record)?;
    Ok((record, packed))
}

fn finalize_index_query(source: &[f32], rope_dim: usize, position: usize) -> Result<Vec<f32>> {
    if rope_dim > source.len() {
        return Err(error("index query RoPE width exceeds index head dimension"));
    }
    let mut query = source
        .iter()
        .map(|&value| half::bf16::from_f32(value).to_f32())
        .collect::<Vec<_>>();
    apply_compressed_rope(&mut query[source.len() - rope_dim..], position)?;
    hadamard_bf16(&mut query)?;
    let _ = fp4_roundtrip(&mut query)?;
    Ok(query)
}

#[allow(clippy::too_many_arguments)]
fn select_ratio4_topk(
    query: &[f32],
    weights: &[f32],
    keys: &[f32],
    query_shape: [usize; 4],
    records: usize,
    query_start: usize,
    topk_width: usize,
    rope_dim: usize,
) -> Result<Vec<i32>> {
    let [batch, sequence, index_heads, index_dim] = query_shape;
    let mut selected = fallible_filled(
        checked_product(&[batch, sequence, topk_width], "ratio-4 selected indices")?,
        -1i32,
        "ratio-4 selected indices",
    )?;
    let weight_scale = 1.0 / (index_dim as f32).sqrt() / (index_heads as f32).sqrt();
    for b in 0..batch {
        for s in 0..sequence {
            let position = query_start
                .checked_add(s)
                .filter(|&value| value <= isize::MAX as usize)
                .ok_or_else(|| error("ratio-4 selection query position overflow"))?;
            let valid_records = position
                .checked_add(1)
                .ok_or_else(|| error("ratio-4 causal record count overflow"))?
                / 4;
            let mut transformed = Vec::with_capacity(
                index_heads
                    .checked_mul(index_dim)
                    .ok_or_else(|| error("index query workspace overflow"))?,
            );
            for head in 0..index_heads {
                let offset = flat4([b, s, head, 0], query_shape, "ratio-4 index query row")?;
                let end = offset
                    .checked_add(index_dim)
                    .ok_or_else(|| error("ratio-4 index query row end overflow"))?;
                transformed.extend(finalize_index_query(
                    query
                        .get(offset..end)
                        .ok_or_else(|| error("ratio-4 index query row is out of bounds"))?,
                    rope_dim,
                    position,
                )?);
            }
            let weight_row = b
                .checked_mul(sequence)
                .and_then(|value| value.checked_add(s))
                .and_then(|value| value.checked_mul(index_heads))
                .ok_or_else(|| error("ratio-4 index weight row overflow"))?;
            let mut scores = Vec::with_capacity(valid_records);
            for record in 0..valid_records.min(records) {
                let key_row = b
                    .checked_mul(records)
                    .and_then(|value| value.checked_add(record))
                    .and_then(|value| value.checked_mul(index_dim))
                    .ok_or_else(|| error("ratio-4 index key row overflow"))?;
                let mut score = 0.0f32;
                for head in 0..index_heads {
                    let query_row = head
                        .checked_mul(index_dim)
                        .ok_or_else(|| error("ratio-4 transformed query row overflow"))?;
                    let dot = dot(&transformed, query_row, keys, key_row, index_dim)?;
                    let weight_offset = weight_row
                        .checked_add(head)
                        .ok_or_else(|| error("ratio-4 index weight offset overflow"))?;
                    let weight = *weights
                        .get(weight_offset)
                        .ok_or_else(|| error("ratio-4 index weight is out of bounds"))?;
                    score += dot.max(0.0) * weight * weight_scale;
                }
                if !score.is_finite() {
                    return Err(error(format!(
                        "ratio-4 index score is non-finite at [batch={b}, query={s}, record={record}]"
                    )));
                }
                scores.push((record, score));
            }
            for left in 0..scores.len() {
                for right in left + 1..scores.len() {
                    if scores[left].1 == scores[right].1 {
                        return Err(unsupported(format!(
                            "portable top-k tie ordering is unfrozen: equal score {} at [batch={b}, query={s}] for compressed records {} and {}",
                            scores[left].1, scores[left].0, scores[right].0
                        )));
                    }
                }
            }
            scores.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
            let row = b
                .checked_mul(sequence)
                .and_then(|value| value.checked_add(s))
                .and_then(|value| value.checked_mul(topk_width))
                .ok_or_else(|| error("ratio-4 selected row overflow"))?;
            for (slot, &(record, _)) in scores.iter().take(topk_width).enumerate() {
                let destination = row
                    .checked_add(slot)
                    .ok_or_else(|| error("ratio-4 selected index offset overflow"))?;
                *selected
                    .get_mut(destination)
                    .ok_or_else(|| error("ratio-4 selected index is out of bounds"))? =
                    i32::try_from(record)
                        .map_err(|_| error("selected compressed index exceeds i32::MAX"))?;
            }
        }
    }
    Ok(selected)
}

#[allow(clippy::too_many_arguments)]
fn ratio4_attention(
    query: &[f32],
    query_shape: [usize; 4],
    current_kv: &[f32],
    current_kv_shape: [usize; 3],
    current_kv_base: usize,
    compressed: &[f32],
    compressed_records: usize,
    selected: &[i32],
    topk_width: usize,
    query_start: usize,
    dense_candidates: usize,
    sink: &[f32],
    configured_scale: f32,
    attention_bias: Option<&AttentionBias>,
) -> Result<Vec<f32>> {
    let [batch, sequence, heads, dim] = query_shape;
    let candidate_count = dense_candidates
        .checked_add(topk_width)
        .ok_or_else(|| error("ratio-4 attention candidate count overflow"))?;
    let mut output = fallible_filled(
        checked_product(&query_shape, "ratio-4 attention output")?,
        0.0f32,
        "ratio-4 attention output",
    )?;
    let mut scores = fallible_filled(
        candidate_count,
        f32::NEG_INFINITY,
        "ratio-4 attention scores",
    )?;
    let scale = if configured_scale == 0.0 {
        1.0 / (dim as f32).sqrt()
    } else {
        configured_scale
    };
    for b in 0..batch {
        for s in 0..sequence {
            let position = query_start
                .checked_add(s)
                .filter(|&value| value <= isize::MAX as usize)
                .ok_or_else(|| error("ratio-4 attention query position overflow"))?;
            let dense_start = current_kv_base.max(
                position
                    .checked_add(1)
                    .ok_or_else(|| error("ratio-4 dense-window position overflow"))?
                    .saturating_sub(128),
            );
            let selected_row = b
                .checked_mul(sequence)
                .and_then(|value| value.checked_add(s))
                .and_then(|value| value.checked_mul(topk_width))
                .ok_or_else(|| error("ratio-4 selected attention row overflow"))?;
            for (h, sink_value) in sink.iter().copied().enumerate().take(heads) {
                scores.fill(f32::NEG_INFINITY);
                let query_row = flat4([b, s, h, 0], query_shape, "ratio-4 query row")?;
                let mut maximum = f32::NEG_INFINITY;
                for (candidate, score_slot) in scores.iter_mut().enumerate().take(dense_candidates)
                {
                    let absolute = dense_start
                        .checked_add(candidate)
                        .ok_or_else(|| error("ratio-4 dense candidate overflow"))?;
                    if absolute > position {
                        continue;
                    }
                    let relative = absolute
                        .checked_sub(current_kv_base)
                        .ok_or_else(|| error("ratio-4 dense candidate precedes storage"))?;
                    if relative >= current_kv_shape[1] {
                        continue;
                    }
                    let kv_row = flat3([b, relative, 0], current_kv_shape, "ratio-4 dense KV row")?;
                    let mut score = dot(query, query_row, current_kv, kv_row, dim)? * scale;
                    if let Some(bias) = attention_bias {
                        score += bias.at(b, h, s, candidate)?;
                    }
                    *score_slot = score;
                    maximum = maximum.max(score);
                }
                for slot in 0..topk_width {
                    let selected_offset = selected_row
                        .checked_add(slot)
                        .ok_or_else(|| error("ratio-4 selected attention offset overflow"))?;
                    let raw = *selected.get(selected_offset).ok_or_else(|| {
                        error("ratio-4 selected attention index is out of bounds")
                    })?;
                    let Ok(record) = usize::try_from(raw) else {
                        continue;
                    };
                    if record >= compressed_records {
                        return Err(error("ratio-4 selected compressed record is out of bounds"));
                    }
                    let candidate = dense_candidates
                        .checked_add(slot)
                        .ok_or_else(|| error("ratio-4 compressed candidate overflow"))?;
                    let kv_row = flat3(
                        [b, record, 0],
                        [batch, compressed_records, dim],
                        "ratio-4 compressed KV row",
                    )?;
                    let mut score = dot(query, query_row, compressed, kv_row, dim)? * scale;
                    if let Some(bias) = attention_bias {
                        score += bias.at(b, h, s, candidate)?;
                    }
                    scores[candidate] = score;
                    maximum = maximum.max(score);
                }
                if maximum == f32::NEG_INFINITY {
                    continue;
                }
                let denominator = scores
                    .iter()
                    .filter(|&&score| score != f32::NEG_INFINITY)
                    .map(|&score| (score - maximum).exp())
                    .sum::<f32>()
                    + (sink_value - maximum).exp();
                if denominator == 0.0 || !denominator.is_finite() {
                    return Err(error(format!(
                        "ratio-4 softmax denominator is invalid at [batch={b}, head={h}, query={s}]"
                    )));
                }
                let output_row = flat4([b, s, h, 0], query_shape, "ratio-4 output row")?;
                for (candidate, &score) in scores.iter().enumerate() {
                    if score == f32::NEG_INFINITY {
                        continue;
                    }
                    let probability = (score - maximum).exp() / denominator;
                    if candidate < dense_candidates {
                        let absolute = dense_start
                            .checked_add(candidate)
                            .ok_or_else(|| error("ratio-4 dense value position overflow"))?;
                        let relative = absolute
                            .checked_sub(current_kv_base)
                            .ok_or_else(|| error("ratio-4 dense value precedes storage"))?;
                        let kv_row = flat3(
                            [b, relative, 0],
                            current_kv_shape,
                            "ratio-4 dense value row",
                        )?;
                        accumulate_value(
                            &mut output,
                            output_row,
                            current_kv,
                            kv_row,
                            dim,
                            probability,
                        )?;
                    } else {
                        let slot = candidate - dense_candidates;
                        let selected_offset = selected_row
                            .checked_add(slot)
                            .ok_or_else(|| error("ratio-4 selected value offset overflow"))?;
                        let selected_record = *selected.get(selected_offset).ok_or_else(|| {
                            error("ratio-4 selected value index is out of bounds")
                        })?;
                        let record = usize::try_from(selected_record)
                            .map_err(|_| error("invalid selected compressed value index"))?;
                        let kv_row = flat3(
                            [b, record, 0],
                            [batch, compressed_records, dim],
                            "ratio-4 compressed value row",
                        )?;
                        accumulate_value(
                            &mut output,
                            output_row,
                            compressed,
                            kv_row,
                            dim,
                            probability,
                        )?;
                    }
                }
            }
        }
    }
    Ok(output)
}

fn write_dense_i32(out: &mut TensorMut, values: &[i32]) -> Result<()> {
    let mut bytes = Vec::with_capacity(
        values
            .len()
            .checked_mul(std::mem::size_of::<i32>())
            .ok_or_else(|| error("selected_indices byte count overflow"))?,
    );
    for &value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    write_dense_bytes(out, &bytes)
}

fn write_shared_selected_i32(
    out: &mut TensorMut,
    shared: &[i32],
    batch: usize,
    index_heads: usize,
    sequence: usize,
    topk: usize,
) -> Result<()> {
    let mut replicated = fallible_filled(
        checked_product(
            &[batch, index_heads, sequence, topk],
            "replicated selected indices",
        )?,
        -1i32,
        "replicated selected indices",
    )?;
    let row_width = sequence
        .checked_mul(topk)
        .ok_or_else(|| error("shared selected row width overflow"))?;
    for b in 0..batch {
        let source = b
            .checked_mul(row_width)
            .ok_or_else(|| error("shared selected source offset overflow"))?;
        let source_end = source
            .checked_add(row_width)
            .ok_or_else(|| error("shared selected source end overflow"))?;
        let source = shared
            .get(source..source_end)
            .ok_or_else(|| error("shared selected source is out of bounds"))?;
        for head in 0..index_heads {
            let destination = b
                .checked_mul(index_heads)
                .and_then(|value| value.checked_add(head))
                .and_then(|value| value.checked_mul(row_width))
                .ok_or_else(|| error("replicated selected destination overflow"))?;
            let destination_end = destination
                .checked_add(row_width)
                .ok_or_else(|| error("replicated selected destination end overflow"))?;
            replicated
                .get_mut(destination..destination_end)
                .ok_or_else(|| error("replicated selected destination is out of bounds"))?
                .copy_from_slice(source);
        }
    }
    write_dense_i32(out, &replicated)
}

fn shape3(name: &str, shape: &[usize]) -> Result<[usize; 3]> {
    shape
        .try_into()
        .map_err(|_| error(format!("{name} must be rank 3, got shape {shape:?}")))
}

fn flat3(index: [usize; 3], shape: [usize; 3], what: &str) -> Result<usize> {
    index[0]
        .checked_mul(shape[1])
        .and_then(|value| value.checked_add(index[1]))
        .and_then(|value| value.checked_mul(shape[2]))
        .and_then(|value| value.checked_add(index[2]))
        .ok_or_else(|| error(format!("{what} offset overflow")))
}

fn checked_position(value: i64, what: &str) -> Result<usize> {
    usize::try_from(value)
        .ok()
        .filter(|&position| position <= isize::MAX as usize)
        .ok_or_else(|| {
            error(format!(
                "{what} must be non-negative and <= isize::MAX, got {value}"
            ))
        })
}

fn require_finite(name: &str, values: &[f32]) -> Result<()> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(error(format!(
            "{name}[{index}] must be finite, got {value}"
        )));
    }
    Ok(())
}

fn carry_offset(
    batch: usize,
    slot: usize,
    state: usize,
    dim: usize,
    ratio: usize,
    width: usize,
) -> Result<usize> {
    batch
        .checked_mul(ratio)
        .and_then(|value| value.checked_add(slot))
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_add(state))
        .and_then(|value| value.checked_mul(width))
        .and_then(|value| value.checked_add(dim))
        .ok_or_else(|| error("compression carry offset overflow"))
}

fn validate_carry(carry: &[f32], batch: usize, ratio: usize, dim: usize) -> Result<()> {
    for b in 0..batch {
        for slot in 0..ratio {
            for d in 0..dim {
                let kv = carry[carry_offset(b, slot, 0, d, ratio, dim)?];
                let score = carry[carry_offset(b, slot, 1, d, ratio, dim)?];
                if !kv.is_finite() {
                    return Err(error(format!(
                        "past_compression_carry kv_state[{b},{slot},{d}] must be finite"
                    )));
                }
                if !(score.is_finite() || score == f32::NEG_INFINITY) {
                    return Err(error(format!(
                        "past_compression_carry score_state[{b},{slot},{d}] must be finite or -inf"
                    )));
                }
            }
        }
    }
    Ok(())
}

fn reset_ratio128_carry(carry: &mut [f32], batch: usize, ratio: usize, dim: usize) -> Result<()> {
    for b in 0..batch {
        reset_ratio128_row(carry, b, ratio, dim)?;
    }
    Ok(())
}

fn reset_ratio128_row(carry: &mut [f32], batch: usize, ratio: usize, dim: usize) -> Result<()> {
    for slot in 0..ratio {
        for d in 0..dim {
            let kv = carry_offset(batch, slot, 0, d, ratio, dim)?;
            let score = carry_offset(batch, slot, 1, d, ratio, dim)?;
            carry[kv] = 0.0;
            carry[score] = f32::NEG_INFINITY;
        }
    }
    Ok(())
}

fn pool_ratio128_record(carry: &[f32], batch: usize, ratio: usize, dim: usize) -> Result<Vec<f32>> {
    let mut pooled = fallible_filled(dim, 0.0f32, "pooled compressed record")?;
    for (d, destination) in pooled.iter_mut().enumerate() {
        let mut maximum = f32::NEG_INFINITY;
        for slot in 0..ratio {
            maximum = maximum.max(carry[carry_offset(batch, slot, 1, d, ratio, dim)?]);
        }
        if !maximum.is_finite() {
            return Err(error(format!(
                "compression block has no finite score for dimension {d}"
            )));
        }
        let mut denominator = 0.0f32;
        let mut numerator = 0.0f32;
        for slot in 0..ratio {
            let score = carry[carry_offset(batch, slot, 1, d, ratio, dim)?];
            if score == f32::NEG_INFINITY {
                continue;
            }
            let weight = (score - maximum).exp();
            denominator += weight;
            numerator += weight * carry[carry_offset(batch, slot, 0, d, ratio, dim)?];
        }
        if denominator == 0.0 || !denominator.is_finite() || !numerator.is_finite() {
            return Err(error(format!(
                "compression softmax is invalid for dimension {d}"
            )));
        }
        *destination = numerator / denominator;
    }
    Ok(pooled)
}

fn finalize_attention_record(
    pooled: &[f32],
    norm: &[f32],
    dim: usize,
    rope_dim: usize,
    block_start: usize,
    cache_format: CacheFormat,
) -> Result<(Vec<f32>, Option<Vec<u8>>)> {
    if pooled.len() != dim || norm.len() != dim {
        return Err(error("compressed record finalization width mismatch"));
    }
    if rope_dim > dim || !rope_dim.is_multiple_of(2) {
        return Err(error(format!(
            "qk_rope_head_dim must be even and <= head_dim, got {rope_dim} for {dim}"
        )));
    }
    let mut record = pooled
        .iter()
        .map(|&value| half::bf16::from_f32(value).to_f32())
        .collect::<Vec<_>>();
    let square_sum = record.iter().try_fold(0.0f32, |sum, &value| {
        let next = sum + value * value;
        next.is_finite()
            .then_some(next)
            .ok_or_else(|| error("RMSNorm square sum is non-finite"))
    })?;
    let inverse_rms = (square_sum / dim as f32 + 1.0e-6).sqrt().recip();
    for (value, &weight) in record.iter_mut().zip(norm) {
        *value = half::bf16::from_f32(*value * inverse_rms * weight).to_f32();
    }
    apply_compressed_rope(&mut record[dim - rope_dim..], block_start)?;

    let non_rope = dim - rope_dim;
    if !non_rope.is_multiple_of(FP8_E4M3_BLOCK_SIZE) {
        return Err(error(format!(
            "non-RoPE head dimension {non_rope} must be divisible by {FP8_E4M3_BLOCK_SIZE}"
        )));
    }
    let stored_width = cache_format.stored_width(dim, rope_dim)?;
    let mut packed =
        (cache_format == CacheFormat::Fp8E4m3Block64).then(|| Vec::with_capacity(stored_width));
    for block_start in (0..non_rope).step_by(FP8_E4M3_BLOCK_SIZE) {
        let block_end = block_start
            .checked_add(FP8_E4M3_BLOCK_SIZE)
            .ok_or_else(|| error("FP8 finalization block end overflow"))?;
        let mut scale = 0u8;
        let mut codes = [0u8; FP8_E4M3_PACKED_BYTES];
        let mut dequantized = [0.0f32; FP8_E4M3_BLOCK_SIZE];
        quantize_fp8_e4m3_block(
            &record[block_start..block_end],
            &mut scale,
            &mut codes,
            &mut dequantized,
        )?;
        record[block_start..block_end].copy_from_slice(&dequantized);
        if let Some(bytes) = packed.as_mut() {
            bytes.push(scale);
            bytes.extend_from_slice(&codes);
        }
    }
    if let Some(bytes) = packed.as_mut() {
        for &value in &record[non_rope..] {
            bytes.extend_from_slice(&half::bf16::from_f32(value).to_bits().to_le_bytes());
        }
        if bytes.len() != stored_width {
            return Err(error(format!(
                "hybrid FP8/BF16 record has {} bytes, expected {stored_width}",
                bytes.len()
            )));
        }
    }
    Ok((record, packed))
}

fn apply_compressed_rope(tail: &mut [f32], position: usize) -> Result<()> {
    if tail.is_empty() {
        return Ok(());
    }
    if !tail.len().is_multiple_of(2) {
        return Err(error("compressed RoPE tail width must be even"));
    }
    const BASE: f32 = 160_000.0;
    const FACTOR: f32 = 16.0;
    const LOW: f32 = 15.0;
    const HIGH: f32 = 25.0;
    let rope_dim = tail.len();
    let position = position as f32;
    for pair in 0..rope_dim / 2 {
        let ramp = ((pair as f32 - LOW) / (HIGH - LOW)).clamp(0.0, 1.0);
        let base_frequency = BASE.powf(-((2 * pair) as f32) / rope_dim as f32);
        let frequency = base_frequency * (1.0 - ramp) + base_frequency / FACTOR * ramp;
        let (sin, cos) = (position * frequency).sin_cos();
        let offset = pair
            .checked_mul(2)
            .ok_or_else(|| error("compressed RoPE pair offset overflow"))?;
        let imaginary_offset = offset
            .checked_add(1)
            .ok_or_else(|| error("compressed RoPE imaginary offset overflow"))?;
        let real = tail[offset];
        let imaginary = tail[imaginary_offset];
        tail[offset] = half::bf16::from_f32(real * cos - imaginary * sin).to_f32();
        tail[imaginary_offset] = half::bf16::from_f32(real * sin + imaginary * cos).to_f32();
    }
    Ok(())
}

fn combine_logical_records(
    past: &[f32],
    emitted: &[f32],
    batch: usize,
    past_records: usize,
    emitted_records: usize,
    dim: usize,
) -> Result<Vec<f32>> {
    let total_records = past_records
        .checked_add(emitted_records)
        .ok_or_else(|| error("combined logical record count overflow"))?;
    let mut combined = fallible_filled(
        checked_product(&[batch, total_records, dim], "combined logical cache")?,
        0.0f32,
        "combined logical cache",
    )?;
    for b in 0..batch {
        let destination = b
            .checked_mul(total_records)
            .and_then(|value| value.checked_mul(dim))
            .ok_or_else(|| error("combined logical destination overflow"))?;
        let past_offset = b
            .checked_mul(past_records)
            .and_then(|value| value.checked_mul(dim))
            .ok_or_else(|| error("past logical source overflow"))?;
        let past_len = past_records
            .checked_mul(dim)
            .ok_or_else(|| error("past logical row width overflow"))?;
        let past_destination_end = destination
            .checked_add(past_len)
            .ok_or_else(|| error("past logical destination end overflow"))?;
        let past_source_end = past_offset
            .checked_add(past_len)
            .ok_or_else(|| error("past logical source end overflow"))?;
        combined
            .get_mut(destination..past_destination_end)
            .ok_or_else(|| error("past logical destination is out of bounds"))?
            .copy_from_slice(
                past.get(past_offset..past_source_end)
                    .ok_or_else(|| error("past logical source is out of bounds"))?,
            );
        let emitted_offset = b
            .checked_mul(emitted_records)
            .and_then(|value| value.checked_mul(dim))
            .ok_or_else(|| error("emitted logical source overflow"))?;
        let emitted_len = emitted_records
            .checked_mul(dim)
            .ok_or_else(|| error("emitted logical row width overflow"))?;
        let emitted_destination = past_destination_end;
        let emitted_destination_end = emitted_destination
            .checked_add(emitted_len)
            .ok_or_else(|| error("emitted logical destination end overflow"))?;
        let emitted_source_end = emitted_offset
            .checked_add(emitted_len)
            .ok_or_else(|| error("emitted logical source end overflow"))?;
        combined
            .get_mut(emitted_destination..emitted_destination_end)
            .ok_or_else(|| error("emitted logical destination is out of bounds"))?
            .copy_from_slice(
                emitted
                    .get(emitted_offset..emitted_source_end)
                    .ok_or_else(|| error("emitted logical source is out of bounds"))?,
            );
    }
    Ok(combined)
}

fn combine_packed_records(
    past: &[u8],
    emitted: &[u8],
    batch: usize,
    past_records: usize,
    emitted_records: usize,
    width: usize,
) -> Result<Vec<u8>> {
    let total_records = past_records
        .checked_add(emitted_records)
        .ok_or_else(|| error("combined packed record count overflow"))?;
    let mut combined = fallible_filled(
        checked_product(&[batch, total_records, width], "combined packed cache")?,
        0u8,
        "combined packed cache",
    )?;
    for b in 0..batch {
        let destination = b
            .checked_mul(total_records)
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| error("combined packed destination overflow"))?;
        let past_offset = b
            .checked_mul(past_records)
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| error("past packed source overflow"))?;
        let past_len = past_records
            .checked_mul(width)
            .ok_or_else(|| error("past packed row width overflow"))?;
        let past_destination_end = destination
            .checked_add(past_len)
            .ok_or_else(|| error("past packed destination end overflow"))?;
        let past_source_end = past_offset
            .checked_add(past_len)
            .ok_or_else(|| error("past packed source end overflow"))?;
        combined
            .get_mut(destination..past_destination_end)
            .ok_or_else(|| error("past packed destination is out of bounds"))?
            .copy_from_slice(
                past.get(past_offset..past_source_end)
                    .ok_or_else(|| error("past packed source is out of bounds"))?,
            );
        let emitted_offset = b
            .checked_mul(emitted_records)
            .and_then(|value| value.checked_mul(width))
            .ok_or_else(|| error("emitted packed source overflow"))?;
        let emitted_len = emitted_records
            .checked_mul(width)
            .ok_or_else(|| error("emitted packed row width overflow"))?;
        let emitted_destination = past_destination_end;
        let emitted_destination_end = emitted_destination
            .checked_add(emitted_len)
            .ok_or_else(|| error("emitted packed destination end overflow"))?;
        let emitted_source_end = emitted_offset
            .checked_add(emitted_len)
            .ok_or_else(|| error("emitted packed source end overflow"))?;
        combined
            .get_mut(emitted_destination..emitted_destination_end)
            .ok_or_else(|| error("emitted packed destination is out of bounds"))?
            .copy_from_slice(
                emitted
                    .get(emitted_offset..emitted_source_end)
                    .ok_or_else(|| error("emitted packed source is out of bounds"))?,
            );
    }
    Ok(combined)
}

#[allow(clippy::too_many_arguments)]
fn ratio128_attention(
    query: &[f32],
    query_shape: [usize; 4],
    current_kv: &[f32],
    current_kv_shape: [usize; 3],
    current_kv_base: usize,
    compressed: &[f32],
    compressed_records: usize,
    query_start: usize,
    dense_candidates: usize,
    sink: &[f32],
    configured_scale: f32,
    attention_bias: Option<&AttentionBias>,
) -> Result<Vec<f32>> {
    let [batch, sequence, heads, dim] = query_shape;
    let candidate_count = dense_candidates
        .checked_add(compressed_records)
        .ok_or_else(|| error("ratio-128 candidate count overflow"))?;
    checked_layout(
        &[candidate_count],
        std::mem::size_of::<f32>(),
        "ratio-128 attention score row",
    )?;
    let mut output = fallible_filled(
        checked_product(&query_shape, "ratio-128 attention output")?,
        0.0f32,
        "ratio-128 attention output",
    )?;
    let mut scores = fallible_filled(candidate_count, f32::NEG_INFINITY, "attention scores")?;
    let scale = if configured_scale == 0.0 {
        1.0 / (dim as f32).sqrt()
    } else {
        configured_scale
    };
    for b in 0..batch {
        for s in 0..sequence {
            let position = query_start
                .checked_add(s)
                .filter(|&value| value <= isize::MAX as usize)
                .ok_or_else(|| error("attention query position overflow"))?;
            let dense_start = current_kv_base.max(
                position
                    .checked_add(1)
                    .ok_or_else(|| error("attention dense-window position overflow"))?
                    .saturating_sub(128),
            );
            let valid_compressed = position
                .checked_add(1)
                .ok_or_else(|| error("compressed attention position overflow"))?
                / 128;
            for (h, sink_value) in sink.iter().copied().enumerate().take(heads) {
                scores.fill(f32::NEG_INFINITY);
                let query_row = flat4([b, s, h, 0], query_shape, "stateful query row")?;
                let mut maximum = f32::NEG_INFINITY;
                for (candidate, score_slot) in scores.iter_mut().enumerate().take(dense_candidates)
                {
                    let absolute = dense_start
                        .checked_add(candidate)
                        .ok_or_else(|| error("dense candidate position overflow"))?;
                    if absolute > position {
                        continue;
                    }
                    let relative = absolute
                        .checked_sub(current_kv_base)
                        .ok_or_else(|| error("dense candidate precedes current_kv storage"))?;
                    if relative >= current_kv_shape[1] {
                        continue;
                    }
                    let kv_row = flat3(
                        [b, relative, 0],
                        current_kv_shape,
                        "current_kv attention row",
                    )?;
                    let mut score = dot(query, query_row, current_kv, kv_row, dim)?;
                    score *= scale;
                    if let Some(bias) = attention_bias {
                        score += bias.at(b, h, s, candidate)?;
                    }
                    *score_slot = score;
                    maximum = maximum.max(score);
                }
                for record in 0..compressed_records.min(valid_compressed) {
                    let candidate = dense_candidates
                        .checked_add(record)
                        .ok_or_else(|| error("compressed candidate offset overflow"))?;
                    let kv_row = flat3(
                        [b, record, 0],
                        [batch, compressed_records, dim],
                        "compressed attention row",
                    )?;
                    let mut score = dot(query, query_row, compressed, kv_row, dim)?;
                    score *= scale;
                    if let Some(bias) = attention_bias {
                        score += bias.at(b, h, s, candidate)?;
                    }
                    scores[candidate] = score;
                    maximum = maximum.max(score);
                }
                if maximum == f32::NEG_INFINITY {
                    continue;
                }
                let mut denominator = 0.0f32;
                for &score in &scores {
                    if score != f32::NEG_INFINITY {
                        denominator += (score - maximum).exp();
                    }
                }
                denominator += (sink_value - maximum).exp();
                if denominator == 0.0 || !denominator.is_finite() {
                    return Err(error(format!(
                        "softmax denominator is invalid at [batch={b}, head={h}, query={s}]"
                    )));
                }
                let output_row = flat4([b, s, h, 0], query_shape, "stateful output row")?;
                for (candidate, &score) in scores.iter().enumerate().take(dense_candidates) {
                    if score == f32::NEG_INFINITY {
                        continue;
                    }
                    let absolute = dense_start
                        .checked_add(candidate)
                        .ok_or_else(|| error("dense value position overflow"))?;
                    let relative = absolute
                        .checked_sub(current_kv_base)
                        .ok_or_else(|| error("dense value precedes current_kv storage"))?;
                    let kv_row = flat3([b, relative, 0], current_kv_shape, "current_kv value row")?;
                    accumulate_value(
                        &mut output,
                        output_row,
                        current_kv,
                        kv_row,
                        dim,
                        (score - maximum).exp() / denominator,
                    )?;
                }
                for record in 0..compressed_records.min(valid_compressed) {
                    let candidate = dense_candidates
                        .checked_add(record)
                        .ok_or_else(|| error("compressed value candidate overflow"))?;
                    let score = scores[candidate];
                    if score == f32::NEG_INFINITY {
                        continue;
                    }
                    let kv_row = flat3(
                        [b, record, 0],
                        [batch, compressed_records, dim],
                        "compressed value row",
                    )?;
                    accumulate_value(
                        &mut output,
                        output_row,
                        compressed,
                        kv_row,
                        dim,
                        (score - maximum).exp() / denominator,
                    )?;
                }
            }
        }
    }
    Ok(output)
}

fn dot(
    left: &[f32],
    left_offset: usize,
    right: &[f32],
    right_offset: usize,
    dim: usize,
) -> Result<f32> {
    let left_end = left_offset
        .checked_add(dim)
        .ok_or_else(|| error("dot left end overflow"))?;
    let right_end = right_offset
        .checked_add(dim)
        .ok_or_else(|| error("dot right end overflow"))?;
    let left = left
        .get(left_offset..left_end)
        .ok_or_else(|| error("dot left row is out of bounds"))?;
    let right = right
        .get(right_offset..right_end)
        .ok_or_else(|| error("dot right row is out of bounds"))?;
    Ok(left
        .iter()
        .zip(right)
        .fold(0.0f32, |sum, (&a, &b)| sum + a * b))
}

fn accumulate_value(
    output: &mut [f32],
    output_offset: usize,
    value: &[f32],
    value_offset: usize,
    dim: usize,
    probability: f32,
) -> Result<()> {
    let output_end = output_offset
        .checked_add(dim)
        .ok_or_else(|| error("attention output end overflow"))?;
    let value_end = value_offset
        .checked_add(dim)
        .ok_or_else(|| error("attention value end overflow"))?;
    let output = output
        .get_mut(output_offset..output_end)
        .ok_or_else(|| error("attention output row is out of bounds"))?;
    let value = value
        .get(value_offset..value_end)
        .ok_or_else(|| error("attention value row is out of bounds"))?;
    for (destination, &source) in output.iter_mut().zip(value) {
        *destination += probability * source;
    }
    Ok(())
}

fn validate_frozen_v1_schema(node: &Node) -> Result<()> {
    if !(FROZEN_V1_REQUIRED_INPUTS..=FROZEN_V1_MAX_INPUTS).contains(&node.inputs.len()) {
        return Err(error(format!(
            "frozen v1 requires {FROZEN_V1_REQUIRED_INPUTS}..={FROZEN_V1_MAX_INPUTS} positional inputs, got {}",
            node.inputs.len()
        )));
    }
    if !(FROZEN_V1_REQUIRED_OUTPUTS..=FROZEN_V1_MAX_OUTPUTS).contains(&node.outputs.len()) {
        return Err(error(format!(
            "frozen v1 requires {FROZEN_V1_REQUIRED_OUTPUTS}..={FROZEN_V1_MAX_OUTPUTS} outputs, got {}",
            node.outputs.len()
        )));
    }
    for (index, name) in FROZEN_V1_REQUIRED_INPUT_NAMES.iter().enumerate() {
        if node.inputs[index].is_none() {
            return Err(error(format!(
                "required frozen-v1 input {index} ('{name}') is omitted"
            )));
        }
    }
    Ok(())
}

fn validate_ratio_specific_v1_schema(node: &Node, compression_ratio: usize) -> Result<()> {
    match compression_ratio {
        4 => {
            if node.inputs.len() < 19 || node.inputs[11..19].iter().any(Option::is_none) {
                return Err(error(
                    "ratio-4 requires all eight positional index inputs (11..=18)",
                ));
            }
            if !(5..=6).contains(&node.outputs.len()) {
                return Err(error(format!(
                    "ratio-4 requires 5 or 6 outputs, got {}",
                    node.outputs.len()
                )));
            }
        }
        128 => {
            if node.inputs.iter().skip(11).take(8).any(Option::is_some) {
                return Err(unsupported(
                    "ratio-4-only inputs (11..=18) are not supported by the ratio-128 stateful path",
                ));
            }
            if node.outputs.len() != FROZEN_V1_REQUIRED_OUTPUTS {
                return Err(unsupported(format!(
                    "ratio-128 supports exactly {FROZEN_V1_REQUIRED_OUTPUTS} outputs, got {}",
                    node.outputs.len()
                )));
            }
        }
        _ => unreachable!("compression ratio was validated before schema validation"),
    }
    Ok(())
}

fn validate_frozen_v1_runtime_arity(inputs: &[TensorView], outputs: &[TensorMut]) -> Result<()> {
    if !(FROZEN_V1_REQUIRED_INPUTS..=FROZEN_V1_MAX_INPUTS).contains(&inputs.len()) {
        return Err(error(format!(
            "frozen v1 requires {FROZEN_V1_REQUIRED_INPUTS}..={FROZEN_V1_MAX_INPUTS} positional inputs, got {}",
            inputs.len()
        )));
    }
    if !(FROZEN_V1_REQUIRED_OUTPUTS..=FROZEN_V1_MAX_OUTPUTS).contains(&outputs.len()) {
        return Err(error(format!(
            "frozen v1 requires {FROZEN_V1_REQUIRED_OUTPUTS}..={FROZEN_V1_MAX_OUTPUTS} outputs, got {}",
            outputs.len()
        )));
    }
    Ok(())
}

impl Kernel for CompressedSparseAttentionKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity(OP, inputs, outputs, 4, 6, 1)?;
        for (name, input) in [("query", &inputs[0]), ("head_sink", &inputs[3])] {
            require_dtype(name, input.dtype, DataType::Float32)?;
        }
        require_dtype("cache", inputs[1].dtype, self.cache_format.dtype())?;
        if !matches!(inputs[2].dtype, DataType::Int32 | DataType::Int64) {
            return Err(error(format!(
                "indices must have dtype Int32 or Int64, got {:?}",
                inputs[2].dtype
            )));
        }
        require_dtype("Y", outputs[0].dtype, DataType::Float32)?;

        let expected_output = infer_output_shape_for_format(
            inputs[0].shape,
            inputs[1].shape,
            inputs[2].shape,
            inputs[3].shape,
            self.num_heads,
            self.head_dim,
            self.qk_rope_head_dim,
            self.cache_format,
        )?;
        if outputs[0].shape != expected_output {
            return Err(error(format!(
                "Y must have shape {expected_output:?}, got {:?}",
                outputs[0].shape
            )));
        }
        let query_shape = shape4("query", inputs[0].shape)?;
        let cache_shape = shape4("cache", inputs[1].shape)?;
        let indices_shape = shape4("indices", inputs[2].shape)?;
        let [batch, sequence, heads, dim] = query_shape;
        let groups = cache_shape[1];
        if groups != 1 && groups != heads {
            return Err(unsupported(format!(
                "cache/index groups must be 1 or num_heads ({heads}) in the Phase-1 skeleton, got {groups}"
            )));
        }
        if self.compression_ratio == 4 && groups != 1 && groups != self.index_num_heads {
            return Err(error(format!(
                "ratio-4 grouped indices must use 1 or index_num_heads={} groups, got {groups}",
                self.index_num_heads
            )));
        }
        debug_assert_eq!(dim, self.head_dim);
        debug_assert_eq!(heads, self.num_heads);
        let valid_lengths = inputs
            .get(4)
            .filter(|input| !input.is_absent())
            .map(|input| read_valid_lengths(input, batch, cache_shape[2]))
            .transpose()?;
        let attention_bias = inputs
            .get(5)
            .filter(|input| !input.is_absent())
            .map(|input| AttentionBias::new(input, [batch, heads, sequence, indices_shape[3]]))
            .transpose()?;

        let query = read_dense_f32(&inputs[0], "query")?;
        let cache = dequantize_cache(
            &inputs[1],
            cache_shape,
            dim,
            self.qk_rope_head_dim,
            self.cache_format,
        )?;
        let indices = read_dense_indices(&inputs[2], "indices")?;
        let sink = read_dense_f32(&inputs[3], "head_sink")?;
        let gathered = sparse_kv_gather_masked_f32(
            &cache,
            [batch, groups, cache_shape[2], dim],
            &indices,
            indices_shape,
            valid_lengths.as_deref(),
        )?;

        let selections = indices_shape[3];
        let output_elements = checked_layout(
            &[batch, sequence, heads, dim],
            std::mem::size_of::<f32>(),
            "Y",
        )?;
        let score_elements =
            checked_product(&[batch, heads, sequence, selections], "score element count")?;
        checked_layout(
            &[batch, heads, sequence, selections],
            std::mem::size_of::<f32>(),
            "scores",
        )?;
        let mut scores = fallible_filled(score_elements, f32::NEG_INFINITY, "attention scores")?;
        let mut output = fallible_filled(output_elements, 0.0f32, "attention output")?;
        let scale = if self.scale == 0.0 {
            1.0 / (dim as f32).sqrt()
        } else {
            self.scale
        };

        for b in 0..batch {
            for (h, sink_value) in sink.iter().copied().enumerate().take(heads) {
                let group = if groups == 1 { 0 } else { h };
                for s in 0..sequence {
                    let score_row = flat4(
                        [b, h, s, 0],
                        [batch, heads, sequence, selections],
                        "score row",
                    )?;
                    let gathered_row = flat4(
                        [b, group, s, 0],
                        [batch, groups, sequence, selections],
                        "gathered row",
                    )?;
                    let query_row =
                        flat4([b, s, h, 0], [batch, sequence, heads, dim], "query row")?;
                    let mut maximum = f32::NEG_INFINITY;
                    for k in 0..selections {
                        let record = gathered_row
                            .checked_add(k)
                            .ok_or_else(|| error("gathered validity offset overflow"))?;
                        if !gathered.valid[record] {
                            continue;
                        }
                        let kv_row = record
                            .checked_mul(dim)
                            .ok_or_else(|| error("gathered KV offset overflow"))?;
                        let mut score = 0.0f32;
                        for d in 0..dim {
                            score += query[query_row + d] * gathered.values[kv_row + d];
                        }
                        score *= scale;
                        if let Some(bias) = &attention_bias {
                            score += bias.at(b, h, s, k)?;
                        }
                        scores[score_row + k] = score;
                        maximum = maximum.max(score);
                    }
                    if maximum == f32::NEG_INFINITY {
                        continue;
                    }

                    let mut denominator = 0.0f32;
                    for k in 0..selections {
                        let score = scores[score_row + k];
                        if score != f32::NEG_INFINITY {
                            denominator += (score - maximum).exp();
                        }
                    }
                    denominator += (sink_value - maximum).exp();
                    if denominator == 0.0 || denominator.is_nan() {
                        return Err(error(format!(
                            "softmax denominator is invalid at [batch={b}, head={h}, query={s}]"
                        )));
                    }
                    let output_row =
                        flat4([b, s, h, 0], [batch, sequence, heads, dim], "output row")?;
                    for k in 0..selections {
                        let score = scores[score_row + k];
                        if score == f32::NEG_INFINITY {
                            continue;
                        }
                        let record = gathered_row
                            .checked_add(k)
                            .ok_or_else(|| error("gathered record offset overflow"))?;
                        let kv_row = record
                            .checked_mul(dim)
                            .ok_or_else(|| error("gathered KV offset overflow"))?;
                        let probability = (score - maximum).exp() / denominator;
                        for d in 0..dim {
                            output[output_row + d] += probability * gathered.values[kv_row + d];
                        }
                    }
                }
            }
        }
        write_dense_f32(&mut outputs[0], &output)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

/// Infer `Y=[B,S,N,D]` for the dense assembled-cache boundary.
pub fn infer_output_shape(
    query: &[usize],
    cache: &[usize],
    indices: &[usize],
    head_sink: &[usize],
    num_heads: usize,
    head_dim: usize,
) -> Result<Vec<usize>> {
    infer_output_shape_for_format(
        query,
        cache,
        indices,
        head_sink,
        num_heads,
        head_dim,
        0,
        CacheFormat::F32,
    )
}

fn infer_output_shape_for_format(
    query: &[usize],
    cache: &[usize],
    indices: &[usize],
    head_sink: &[usize],
    num_heads: usize,
    head_dim: usize,
    qk_rope_head_dim: usize,
    cache_format: CacheFormat,
) -> Result<Vec<usize>> {
    let query = shape4("query", query)?;
    let cache = shape4("cache", cache)?;
    let indices = shape4("indices", indices)?;
    if query[2] != num_heads || query[3] != head_dim {
        return Err(error(format!(
            "query must end in [num_heads={num_heads}, head_dim={head_dim}], got {:?}",
            &query[2..]
        )));
    }
    let stored_width = cache_format.stored_width(head_dim, qk_rope_head_dim)?;
    if cache[0] != query[0] || cache[3] != stored_width {
        return Err(error(format!(
            "cache must have batch {} and stored record width {stored_width} for {cache_format:?}, got {cache:?}",
            query[0],
        )));
    }
    if indices[0] != query[0] || indices[1] != cache[1] || indices[2] != query[1] {
        return Err(error(format!(
            "indices must have [B,G,S]=[{},{},{}], got {:?}",
            query[0],
            cache[1],
            query[1],
            &indices[..3]
        )));
    }
    if head_sink != [num_heads] {
        return Err(error(format!(
            "head_sink must have shape [{num_heads}], got {head_sink:?}"
        )));
    }
    let output = query.to_vec();
    checked_layout(&output, std::mem::size_of::<f32>(), "Y")?;
    Ok(output)
}

fn dequantize_cache(
    view: &TensorView,
    stored_shape: [usize; 4],
    logical_width: usize,
    qk_rope_head_dim: usize,
    format: CacheFormat,
) -> Result<Vec<f32>> {
    if format == CacheFormat::F32 {
        return read_dense_f32(view, "cache");
    }
    require_dtype("cache", view.dtype, DataType::Uint8)?;
    let stored_width = format.stored_width(logical_width, qk_rope_head_dim)?;
    if stored_shape[3] != stored_width {
        return Err(error(format!(
            "cache stored record width must be {stored_width} for {format:?}, got {}",
            stored_shape[3]
        )));
    }
    checked_layout(&stored_shape, std::mem::size_of::<u8>(), "packed cache")?;
    let packed = to_dense_bytes(view)?;
    let record_count = checked_product(&stored_shape[..3], "packed cache record count")?;
    let output_elements = record_count
        .checked_mul(logical_width)
        .ok_or_else(|| error("dequantized cache element count overflow"))?;
    checked_layout(
        &[record_count, logical_width],
        std::mem::size_of::<f32>(),
        "dequantized cache",
    )?;
    let mut output = fallible_filled(output_elements, 0.0f32, "dequantized cache")?;
    let (block_size, block_bytes) = format
        .block_layout()
        .ok_or_else(|| error("packed cache format is missing its block layout"))?;
    let quantized_width = match format {
        CacheFormat::Fp8E4m3Block64 => logical_width
            .checked_sub(qk_rope_head_dim)
            .ok_or_else(|| error("qk_rope_head_dim exceeds logical cache width"))?,
        CacheFormat::Fp4E2m1Block32 => logical_width,
        CacheFormat::F32 => return Err(error("f32 cache reached packed dequantization path")),
    };
    let blocks_per_record = quantized_width
        .checked_div(block_size)
        .ok_or_else(|| error("cache blocks-per-record division failed"))?;

    for record in 0..record_count {
        let packed_record = record
            .checked_mul(stored_width)
            .ok_or_else(|| error("packed cache record offset overflow"))?;
        let output_record = record
            .checked_mul(logical_width)
            .ok_or_else(|| error("dequantized cache record offset overflow"))?;
        for block in 0..blocks_per_record {
            let packed_block = packed_record
                .checked_add(
                    block
                        .checked_mul(block_bytes)
                        .ok_or_else(|| error("packed cache block offset overflow"))?,
                )
                .ok_or_else(|| error("packed cache block start overflow"))?;
            let packed_values = packed_block
                .checked_add(1)
                .ok_or_else(|| error("packed cache value start overflow"))?;
            let packed_end = packed_block
                .checked_add(block_bytes)
                .ok_or_else(|| error("packed cache block end overflow"))?;
            let output_block = output_record
                .checked_add(
                    block
                        .checked_mul(block_size)
                        .ok_or_else(|| error("dequantized cache block offset overflow"))?,
                )
                .ok_or_else(|| error("dequantized cache block start overflow"))?;
            let output_end = output_block
                .checked_add(block_size)
                .ok_or_else(|| error("dequantized cache block end overflow"))?;
            let scale = *packed
                .get(packed_block)
                .ok_or_else(|| error("packed cache scale offset is out of bounds"))?;
            let values = packed
                .get(packed_values..packed_end)
                .ok_or_else(|| error("packed cache block is out of bounds"))?;
            let destination = output
                .get_mut(output_block..output_end)
                .ok_or_else(|| error("dequantized cache block is out of bounds"))?;
            match format {
                CacheFormat::F32 => {
                    return Err(error("f32 cache reached packed block decoding path"));
                }
                CacheFormat::Fp8E4m3Block64 => {
                    dequantize_fp8_e4m3_block(scale, values, destination)?
                }
                CacheFormat::Fp4E2m1Block32 => {
                    dequantize_fp4_e2m1_block(scale, values, destination)?
                }
            }
        }
        if format == CacheFormat::Fp8E4m3Block64 {
            let fp8_bytes = blocks_per_record
                .checked_mul(block_bytes)
                .ok_or_else(|| error("packed FP8 region width overflow"))?;
            let packed_tail = packed_record
                .checked_add(fp8_bytes)
                .ok_or_else(|| error("packed BF16 RoPE tail offset overflow"))?;
            let rope_bytes = qk_rope_head_dim
                .checked_mul(std::mem::size_of::<u16>())
                .ok_or_else(|| error("packed BF16 RoPE tail width overflow"))?;
            let packed_tail_end = packed_tail
                .checked_add(rope_bytes)
                .ok_or_else(|| error("packed BF16 RoPE tail end overflow"))?;
            let source = packed
                .get(packed_tail..packed_tail_end)
                .ok_or_else(|| error("packed BF16 RoPE tail is out of bounds"))?;
            let output_tail = output_record
                .checked_add(quantized_width)
                .ok_or_else(|| error("dequantized BF16 RoPE tail offset overflow"))?;
            let output_tail_end = output_tail
                .checked_add(qk_rope_head_dim)
                .ok_or_else(|| error("dequantized BF16 RoPE tail end overflow"))?;
            let destination = output
                .get_mut(output_tail..output_tail_end)
                .ok_or_else(|| error("dequantized BF16 RoPE tail is out of bounds"))?;
            for (bytes, value) in source.chunks_exact(2).zip(destination) {
                *value = half::bf16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32();
            }
        }
    }
    Ok(output)
}

struct AttentionBias {
    data: Vec<f32>,
    shape: Vec<usize>,
    padded_shape: [usize; 4],
    target: [usize; 4],
}

impl AttentionBias {
    fn new(view: &TensorView, target: [usize; 4]) -> Result<Self> {
        if view.dtype == DataType::Bool {
            return Err(unsupported(
                "boolean attention_bias semantics are deferred; use additive f32 bias",
            ));
        }
        require_dtype("attention_bias", view.dtype, DataType::Float32)?;
        if view.shape.len() > 4 {
            return Err(error(format!(
                "attention_bias rank must be <= 4, got {:?}",
                view.shape
            )));
        }
        checked_layout(view.shape, std::mem::size_of::<f32>(), "attention_bias")?;
        let mut padded_shape = [1usize; 4];
        padded_shape[4 - view.shape.len()..].copy_from_slice(view.shape);
        for axis in 0..4 {
            if padded_shape[axis] != 1 && padded_shape[axis] != target[axis] {
                return Err(error(format!(
                    "attention_bias shape {:?} is not broadcastable to {target:?}",
                    view.shape
                )));
            }
        }
        Ok(Self {
            data: read_dense_f32(view, "attention_bias")?,
            shape: view.shape.to_vec(),
            padded_shape,
            target,
        })
    }

    fn at(&self, b: usize, h: usize, s: usize, k: usize) -> Result<f32> {
        let target_index = [b, h, s, k];
        let mut offset = 0usize;
        for (&target_coordinate, &extent) in target_index.iter().zip(&self.padded_shape) {
            let coordinate = if extent == 1 { 0 } else { target_coordinate };
            offset = offset
                .checked_mul(extent)
                .and_then(|value| value.checked_add(coordinate))
                .ok_or_else(|| error("attention_bias offset overflow"))?;
        }
        self.data.get(offset).copied().ok_or_else(|| {
            error(format!(
                "attention_bias offset {offset} exceeds shape {:?} for target {:?}",
                self.shape, self.target
            ))
        })
    }
}

fn read_valid_lengths(view: &TensorView, batch: usize, cache_len: usize) -> Result<Vec<usize>> {
    if view.shape != [batch] {
        return Err(error(format!(
            "valid_lengths must have shape [{batch}], got {:?}",
            view.shape
        )));
    }
    let values = read_dense_indices(view, "valid_lengths")?;
    values
        .into_iter()
        .enumerate()
        .map(|(b, value)| {
            let value = usize::try_from(value)
                .map_err(|_| error(format!("valid_lengths[{b}] must be non-negative")))?;
            if value > cache_len {
                return Err(error(format!(
                    "valid_lengths[{b}]={value} exceeds cache length {cache_len}"
                )));
            }
            Ok(value)
        })
        .collect()
}

fn flat4(index: [usize; 4], shape: [usize; 4], what: &str) -> Result<usize> {
    index[0]
        .checked_mul(shape[1])
        .and_then(|value| value.checked_add(index[1]))
        .and_then(|value| value.checked_mul(shape[2]))
        .and_then(|value| value.checked_add(index[2]))
        .and_then(|value| value.checked_mul(shape[3]))
        .and_then(|value| value.checked_add(index[3]))
        .ok_or_else(|| error(format!("{what} offset overflow")))
}

fn shape4(name: &str, shape: &[usize]) -> Result<[usize; 4]> {
    shape
        .try_into()
        .map_err(|_| error(format!("{name} must be rank 4, got shape {shape:?}")))
}

fn required_positive_int(node: &Node, name: &str) -> Result<usize> {
    let value = node
        .attr(name)
        .and_then(|attribute| attribute.as_int())
        .ok_or_else(|| error(format!("missing required integer attribute {name}")))?;
    usize::try_from(value)
        .ok()
        .filter(|&value| value > 0)
        .ok_or_else(|| error(format!("{name} must be positive, got {value}")))
}

fn optional_nonnegative_int(node: &Node, name: &str, default: i64) -> Result<usize> {
    let value = node
        .attr(name)
        .map(|attribute| {
            attribute
                .as_int()
                .ok_or_else(|| error(format!("attribute {name} must be an integer")))
        })
        .transpose()?
        .unwrap_or(default);
    usize::try_from(value).map_err(|_| error(format!("{name} must be non-negative, got {value}")))
}

fn require_int_attr(node: &Node, name: &str, expected: i64) -> Result<()> {
    let value = node
        .attr(name)
        .map(|attribute| {
            attribute
                .as_int()
                .ok_or_else(|| error(format!("attribute {name} must be an integer")))
        })
        .transpose()?
        .unwrap_or(expected);
    if value != expected {
        return Err(error(format!("{name} must be {expected}, got {value}")));
    }
    Ok(())
}

fn require_dtype(name: &str, got: DataType, expected: DataType) -> Result<()> {
    if got != expected {
        return Err(error(format!(
            "{name} must have dtype {expected:?}, got {got:?}"
        )));
    }
    Ok(())
}

fn unsupported(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("{OP}: Unsupported: {}", message.into()))
}

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("{OP}: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ir::{Attribute, Graph, NodeId, static_shape};

    fn kernel(
        ratio: i64,
        cache_format: Option<&str>,
        shapes: &[Vec<usize>],
    ) -> Result<Box<dyn Kernel>> {
        kernel_with_rope_dim(ratio, cache_format, 0, shapes)
    }

    fn kernel_with_rope_dim(
        ratio: i64,
        cache_format: Option<&str>,
        qk_rope_head_dim: usize,
        shapes: &[Vec<usize>],
    ) -> Result<Box<dyn Kernel>> {
        let mut graph = Graph::new();
        graph.opset_imports.insert("pkg.nxrt".into(), 1);
        let cache_dtype = if cache_format.unwrap_or("f32") == "f32" {
            DataType::Float32
        } else {
            DataType::Uint8
        };
        let input_specs = [
            ("query", DataType::Float32),
            ("cache", cache_dtype),
            ("indices", DataType::Int32),
            ("head_sink", DataType::Float32),
        ];
        let inputs = input_specs
            .iter()
            .zip(shapes)
            .map(|((name, dtype), shape)| {
                Some(graph.create_named_value(*name, *dtype, static_shape(shape.iter().copied())))
            })
            .collect();
        let output = graph.create_named_value(
            "Y",
            DataType::Float32,
            static_shape(shapes[0].iter().copied()),
        );
        let mut node = onnx_runtime_ir::Node::new(NodeId(0), OP, inputs, vec![output]);
        node.domain = "pkg.nxrt".into();
        node.attributes
            .insert("num_heads".into(), Attribute::Int(shapes[0][2] as i64));
        node.attributes
            .insert("head_dim".into(), Attribute::Int(shapes[0][3] as i64));
        node.attributes.insert(
            "qk_rope_head_dim".into(),
            Attribute::Int(qk_rope_head_dim as i64),
        );
        node.attributes
            .insert("compression_ratio".into(), Attribute::Int(ratio));
        if ratio == 4 {
            node.attributes
                .insert("index_num_heads".into(), Attribute::Int(1));
            node.attributes
                .insert("index_head_dim".into(), Attribute::Int(128));
            node.attributes
                .insert("index_topk".into(), Attribute::Int(512));
        }
        if let Some(format) = cache_format {
            node.attributes
                .insert("cache_format".into(), Attribute::String(format.into()));
        }
        CompressedSparseAttentionFactory.create_assembled_cache_reference(&node, shapes)
    }

    fn stateful_node(
        graph: &mut Graph,
        ratio: i64,
        input_count: usize,
        output_count: usize,
    ) -> Node {
        let inputs = (0..input_count)
            .map(|index| {
                Some(graph.create_named_value(
                    format!("input_{index}"),
                    DataType::Float32,
                    static_shape([]),
                ))
            })
            .collect();
        let outputs = (0..output_count)
            .map(|index| {
                graph.create_named_value(
                    format!("output_{index}"),
                    DataType::Float32,
                    static_shape([]),
                )
            })
            .collect();
        let mut node = Node::new(NodeId(0), OP, inputs, outputs);
        node.domain = "pkg.nxrt".into();
        node.attributes
            .insert("num_heads".into(), Attribute::Int(1));
        node.attributes
            .insert("head_dim".into(), Attribute::Int(512));
        node.attributes
            .insert("qk_rope_head_dim".into(), Attribute::Int(64));
        node.attributes
            .insert("compression_ratio".into(), Attribute::Int(ratio));
        if ratio == 4 {
            node.attributes
                .insert("index_num_heads".into(), Attribute::Int(1));
            node.attributes
                .insert("index_head_dim".into(), Attribute::Int(128));
            node.attributes
                .insert("index_topk".into(), Attribute::Int(1));
        }
        node
    }

    fn run_reference(
        ratio: i64,
        cache_format: Option<&str>,
        query: &Owned,
        cache: &Owned,
        indices: &Owned,
        sink: &Owned,
    ) -> Vec<f32> {
        let shapes = vec![
            query.shape.clone(),
            cache.shape.clone(),
            indices.shape.clone(),
            sink.shape.clone(),
        ];
        let kernel = kernel(ratio, cache_format, &shapes).unwrap();
        let mut output = Owned::zeros_f32(&query.shape);
        kernel
            .execute(
                &[query.view(), cache.view(), indices.view(), sink.view()],
                &mut [output.view_mut()],
            )
            .unwrap();
        output.to_f32()
    }

    struct StatefulOutputs {
        y: Owned,
        cache: Owned,
        carry: Owned,
    }

    struct Ratio4Outputs {
        y: Owned,
        cache: Owned,
        carry: Owned,
        index: Owned,
        index_carry: Owned,
        selected: Owned,
    }

    #[allow(clippy::too_many_arguments)]
    fn run_ratio128_stateful(
        query: &Owned,
        current_kv: &Owned,
        compressor_kv: &Owned,
        compressor_gate: &Owned,
        compressor_ape: &Owned,
        compressor_norm: &Owned,
        past_cache: &Owned,
        past_carry: &Owned,
        seqlens: &Owned,
        total: &Owned,
        sink: &Owned,
    ) -> StatefulOutputs {
        let inputs = [
            query,
            current_kv,
            compressor_kv,
            compressor_gate,
            compressor_ape,
            compressor_norm,
            past_cache,
            past_carry,
            seqlens,
            total,
            sink,
        ];
        let mut graph = Graph::new();
        graph.opset_imports.insert("pkg.nxrt".into(), 1);
        let node_inputs = FROZEN_V1_REQUIRED_INPUT_NAMES
            .iter()
            .zip(inputs)
            .map(|(name, input)| {
                Some(graph.create_named_value(
                    *name,
                    input.dtype,
                    static_shape(input.shape.iter().copied()),
                ))
            })
            .collect();
        let next_records = usize::try_from(total.to_i64()[0]).unwrap() / 128;
        let outputs = [
            graph.create_named_value(
                "Y",
                DataType::Float32,
                static_shape(query.shape.iter().copied()),
            ),
            graph.create_named_value(
                "present_compressed_kv",
                DataType::Uint8,
                static_shape([query.shape[0], next_records, past_cache.shape[2]]),
            ),
            graph.create_named_value(
                "present_compression_carry",
                DataType::Float32,
                static_shape(past_carry.shape.iter().copied()),
            ),
        ];
        let mut node = Node::new(NodeId(0), OP, node_inputs, outputs.to_vec());
        node.domain = "pkg.nxrt".into();
        node.attributes
            .insert("num_heads".into(), Attribute::Int(query.shape[2] as i64));
        node.attributes
            .insert("head_dim".into(), Attribute::Int(query.shape[3] as i64));
        node.attributes
            .insert("qk_rope_head_dim".into(), Attribute::Int(64));
        node.attributes
            .insert("compression_ratio".into(), Attribute::Int(128));
        node.attributes.insert(
            "cache_format".into(),
            Attribute::String("fp8_e4m3_block64".into()),
        );
        let input_shapes = inputs
            .iter()
            .map(|input| input.shape.clone())
            .collect::<Vec<_>>();
        let kernel = CompressedSparseAttentionFactory
            .create(&node, &input_shapes)
            .unwrap();
        let mut result = StatefulOutputs {
            y: Owned::zeros_f32(&query.shape),
            cache: Owned::zeros(
                DataType::Uint8,
                &[query.shape[0], next_records, past_cache.shape[2]],
            ),
            carry: Owned::zeros(DataType::Float32, &past_carry.shape),
        };
        let input_views = inputs.iter().map(|input| input.view()).collect::<Vec<_>>();
        kernel
            .execute(
                &input_views,
                &mut [
                    result.y.view_mut(),
                    result.cache.view_mut(),
                    result.carry.view_mut(),
                ],
            )
            .unwrap();
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn run_ratio4_stateful(
        query: &Owned,
        current_kv: &Owned,
        compressor_kv: &Owned,
        compressor_gate: &Owned,
        compressor_ape: &Owned,
        compressor_norm: &Owned,
        past_cache: &Owned,
        past_carry: &Owned,
        seqlens: &Owned,
        total: &Owned,
        sink: &Owned,
        index_query: &Owned,
        index_weight: &Owned,
        index_compressor_kv: &Owned,
        index_compressor_gate: &Owned,
        index_compressor_ape: &Owned,
        index_compressor_norm: &Owned,
        past_index: &Owned,
        past_index_carry: &Owned,
        topk: usize,
    ) -> Ratio4Outputs {
        let inputs = [
            query,
            current_kv,
            compressor_kv,
            compressor_gate,
            compressor_ape,
            compressor_norm,
            past_cache,
            past_carry,
            seqlens,
            total,
            sink,
            index_query,
            index_weight,
            index_compressor_kv,
            index_compressor_gate,
            index_compressor_ape,
            index_compressor_norm,
            past_index,
            past_index_carry,
        ];
        let mut graph = Graph::new();
        graph.opset_imports.insert("pkg.nxrt".into(), 1);
        let names = [
            "query",
            "current_kv",
            "compressor_kv",
            "compressor_gate",
            "compressor_ape",
            "compressor_norm",
            "past_compressed_kv",
            "past_compression_carry",
            "seqlens_k",
            "total_sequence_length",
            "head_sink",
            "index_query",
            "index_weight",
            "index_compressor_kv",
            "index_compressor_gate",
            "index_compressor_ape",
            "index_compressor_norm",
            "past_index_key",
            "past_index_carry",
        ];
        let node_inputs = names
            .iter()
            .zip(inputs)
            .map(|(name, input)| {
                Some(graph.create_named_value(
                    *name,
                    input.dtype,
                    static_shape(input.shape.iter().copied()),
                ))
            })
            .collect();
        let next_records = usize::try_from(total.to_i64()[0]).unwrap() / 4;
        let selected_width = topk.min(next_records);
        let outputs = [
            graph.create_named_value(
                "Y",
                DataType::Float32,
                static_shape(query.shape.iter().copied()),
            ),
            graph.create_named_value(
                "present_compressed_kv",
                DataType::Uint8,
                static_shape([query.shape[0], next_records, past_cache.shape[2]]),
            ),
            graph.create_named_value(
                "present_compression_carry",
                DataType::Float32,
                static_shape(past_carry.shape.iter().copied()),
            ),
            graph.create_named_value(
                "present_index_key",
                DataType::Uint8,
                static_shape([query.shape[0], next_records, past_index.shape[2]]),
            ),
            graph.create_named_value(
                "present_index_carry",
                DataType::Float32,
                static_shape(past_index_carry.shape.iter().copied()),
            ),
            graph.create_named_value(
                "selected_indices",
                DataType::Int32,
                static_shape([
                    query.shape[0],
                    index_query.shape[2],
                    query.shape[1],
                    selected_width,
                ]),
            ),
        ];
        let mut node = Node::new(NodeId(0), OP, node_inputs, outputs.to_vec());
        node.domain = "pkg.nxrt".into();
        node.attributes
            .insert("num_heads".into(), Attribute::Int(query.shape[2] as i64));
        node.attributes
            .insert("head_dim".into(), Attribute::Int(query.shape[3] as i64));
        node.attributes
            .insert("qk_rope_head_dim".into(), Attribute::Int(64));
        node.attributes
            .insert("compression_ratio".into(), Attribute::Int(4));
        node.attributes.insert(
            "index_num_heads".into(),
            Attribute::Int(index_query.shape[2] as i64),
        );
        node.attributes.insert(
            "index_head_dim".into(),
            Attribute::Int(index_query.shape[3] as i64),
        );
        node.attributes
            .insert("index_topk".into(), Attribute::Int(topk as i64));
        node.attributes.insert(
            "cache_format".into(),
            Attribute::String("fp8_e4m3_block64".into()),
        );
        let input_shapes = inputs
            .iter()
            .map(|input| input.shape.clone())
            .collect::<Vec<_>>();
        let kernel = CompressedSparseAttentionFactory
            .create(&node, &input_shapes)
            .unwrap();
        let mut result = Ratio4Outputs {
            y: Owned::zeros_f32(&query.shape),
            cache: Owned::zeros(
                DataType::Uint8,
                &[query.shape[0], next_records, past_cache.shape[2]],
            ),
            carry: Owned::zeros(DataType::Float32, &past_carry.shape),
            index: Owned::zeros(
                DataType::Uint8,
                &[query.shape[0], next_records, past_index.shape[2]],
            ),
            index_carry: Owned::zeros(DataType::Float32, &past_index_carry.shape),
            selected: Owned::zeros(
                DataType::Int32,
                &[
                    query.shape[0],
                    index_query.shape[2],
                    query.shape[1],
                    selected_width,
                ],
            ),
        };
        let input_views = inputs.iter().map(|input| input.view()).collect::<Vec<_>>();
        kernel
            .execute(
                &input_views,
                &mut [
                    result.y.view_mut(),
                    result.cache.view_mut(),
                    result.carry.view_mut(),
                    result.index.view_mut(),
                    result.index_carry.view_mut(),
                    result.selected.view_mut(),
                ],
            )
            .unwrap();
        result
    }

    #[test]
    fn gathered_dense_fallback_matches_scalar_sink_oracle() {
        let shapes = vec![
            vec![1, 2, 2, 2],
            vec![1, 1, 3, 2],
            vec![1, 1, 2, 3],
            vec![2],
        ];
        let kernel = kernel(128, None, &shapes).unwrap();
        let query = Owned::f32(&shapes[0], &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0]);
        let cache = Owned::f32(&shapes[1], &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        let indices = Owned::i32(&shapes[2], &[0, 1, -1, 2, 0, 1]);
        let sink = Owned::f32(&shapes[3], &[0.25, -0.5]);
        let mut output = Owned::zeros_f32(&shapes[0]);
        kernel
            .execute(
                &[query.view(), cache.view(), indices.view(), sink.view()],
                &mut [output.view_mut()],
            )
            .unwrap();

        let q = query.to_f32();
        let kv = cache.to_f32();
        let idx = indices.to_i32();
        let sinks = sink.to_f32();
        let scale = 1.0 / 2.0f32.sqrt();
        let mut expected = vec![0.0f32; 8];
        for s in 0..2 {
            for h in 0..2 {
                let mut scores = Vec::new();
                for k in 0..3 {
                    let selected = idx[s * 3 + k];
                    if selected < 0 {
                        continue;
                    }
                    let selected = selected as usize;
                    let dot = q[(s * 2 + h) * 2] * kv[selected * 2]
                        + q[(s * 2 + h) * 2 + 1] * kv[selected * 2 + 1];
                    scores.push((selected, dot * scale));
                }
                let maximum = scores
                    .iter()
                    .map(|(_, score)| *score)
                    .fold(f32::NEG_INFINITY, f32::max);
                let denominator = scores
                    .iter()
                    .map(|(_, score)| (*score - maximum).exp())
                    .sum::<f32>()
                    + (sinks[h] - maximum).exp();
                for (selected, score) in scores {
                    let probability = (score - maximum).exp() / denominator;
                    expected[(s * 2 + h) * 2] += probability * kv[selected * 2];
                    expected[(s * 2 + h) * 2 + 1] += probability * kv[selected * 2 + 1];
                }
            }
        }
        for (actual, expected) in output.to_f32().iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn sink_mode_error_distinguishes_learned_logit_sink_from_sink_tokens() {
        let mut node = Node::new(NodeId(0), OP, vec![], vec![]);
        node.attributes
            .insert("num_heads".into(), Attribute::Int(1));
        node.attributes.insert("head_dim".into(), Attribute::Int(2));
        node.attributes
            .insert("compression_ratio".into(), Attribute::Int(128));
        node.attributes
            .insert("sink_mode".into(), Attribute::String("sink_tokens".into()));

        let message =
            match CompressedSparseAttentionFactory.create_assembled_cache_reference(&node, &[]) {
                Ok(_) => panic!("sink_tokens must not be accepted as a learned-logit sink mode"),
                Err(error) => error.to_string(),
            };
        assert!(message.contains("learned per-head logit input `head_sink`"));
        assert!(message.contains("Metadata `sink_tokens`"));
        assert!(message.contains("retained prefix tokens"));
        assert!(message.contains("unrelated"));
    }

    #[test]
    fn stateful_claim_gate_rejects_ratio_specific_arity() {
        let mut graph = Graph::new();
        let ratio4 = stateful_node(&mut graph, 4, 11, 5);
        let message = match CompressedSparseAttentionFactory.create(&ratio4, &vec![vec![]; 11]) {
            Ok(_) => panic!("ratio-4 must require its index inputs at claim time"),
            Err(error) => error.to_string(),
        };
        assert!(message.contains("ratio-4 requires all eight positional index inputs (11..=18)"));

        let mut graph = Graph::new();
        let ratio128_outputs = stateful_node(&mut graph, 128, 11, 4);
        let message =
            match CompressedSparseAttentionFactory.create(&ratio128_outputs, &vec![vec![]; 11]) {
                Ok(_) => panic!("ratio-128 must reject ratio-4 output arity at claim time"),
                Err(error) => error.to_string(),
            };
        assert!(message.contains("ratio-128 supports exactly 3 outputs, got 4"));

        let mut graph = Graph::new();
        let ratio128_index_input = stateful_node(&mut graph, 128, 12, 3);
        let message = match CompressedSparseAttentionFactory
            .create(&ratio128_index_input, &vec![vec![]; 12])
        {
            Ok(_) => panic!("ratio-128 must reject ratio-4 inputs at claim time"),
            Err(error) => error.to_string(),
        };
        assert!(message.contains("ratio-4-only inputs (11..=18)"));
    }

    #[test]
    fn attention_bias_rejects_unsupported_dtype_rank_and_broadcast() {
        let boolean = Owned::bool_(&[1], &[true]);
        let message = match AttentionBias::new(&boolean.view(), [1, 1, 1, 1]) {
            Ok(_) => panic!("boolean attention bias must remain unsupported"),
            Err(error) => error.to_string(),
        };
        assert!(message.contains("boolean attention_bias semantics are deferred"));

        let rank_five = Owned::zeros_f32(&[1, 1, 1, 1, 1]);
        let message = match AttentionBias::new(&rank_five.view(), [1, 1, 1, 1]) {
            Ok(_) => panic!("rank-five attention bias must be rejected"),
            Err(error) => error.to_string(),
        };
        assert!(message.contains("attention_bias rank must be <= 4"));

        let non_broadcastable = Owned::zeros_f32(&[2, 2]);
        let message = match AttentionBias::new(&non_broadcastable.view(), [1, 1, 1, 2]) {
            Ok(_) => panic!("non-broadcastable attention bias must be rejected"),
            Err(error) => error.to_string(),
        };
        assert!(message.contains("is not broadcastable"));
    }

    #[test]
    fn packed_cache_formats_reject_invalid_block_widths() {
        let fp8 = CacheFormat::Fp8E4m3Block64
            .stored_width(FP8_E4M3_BLOCK_SIZE + 1, 0)
            .unwrap_err()
            .to_string();
        assert!(fp8.contains("must be divisible by FP8 block size"));

        let fp4 = CacheFormat::Fp4E2m1Block32
            .stored_width(FP4_E2M1_BLOCK_SIZE + 1, 0)
            .unwrap_err()
            .to_string();
        assert!(fp4.contains("must be divisible by FP4 block size"));
    }

    #[test]
    fn fp8_compressed_cache_matches_same_logical_f32_cache() {
        let dim = FP8_E4M3_BLOCK_SIZE;
        let query = Owned::f32(&[1, 1, 1, dim], &vec![0.125; dim]);
        let indices = Owned::i32(&[1, 1, 1, 2], &[0, 1]);
        let sink = Owned::f32(&[1], &[0.25]);
        let record_codes = [[0x38, 0x3c, 0x40, 0xb8], [0x30, 0x34, 0xb0, 0xb4]];
        let mut packed = Vec::new();
        let mut dense = Vec::new();
        for codes in record_codes {
            packed.push(127);
            for code in codes.into_iter().cycle().take(dim) {
                packed.push(code);
                dense.push(super::super::block_dequant::decode_e4m3fn(code));
            }
        }
        let dense_cache = Owned::f32(&[1, 1, 2, dim], &dense);
        let packed_cache = Owned::u8(&[1, 1, 2, FP8_E4M3_PACKED_BYTES + 1], &packed);

        let expected = run_reference(128, None, &query, &dense_cache, &indices, &sink);
        let actual = run_reference(
            128,
            Some("fp8_e4m3_block64"),
            &query,
            &packed_cache,
            &indices,
            &sink,
        );
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn fp8_hybrid_d512_preserves_bf16_rope_tail() {
        const DIM: usize = 512;
        const ROPE_DIM: usize = 64;
        const NON_ROPE_DIM: usize = DIM - ROPE_DIM;
        const FP8_HALF_ULP_AT_TWO: f32 = 0.125;

        let block_codes = [0x38, 0x3c, 0x40, 0xb8, 0x30, 0x34, 0x28];
        let block_values = [1.0, 1.5, 2.0, -1.0, 0.5, 0.75, 0.25];
        let dense_block_values = [1.03, 1.47, 1.94, -1.03, 0.48, 0.77, 0.26];
        let rope_pattern = [8.0, -4.0, 0.5, -0.25];

        let mut packed = Vec::new();
        let mut expected = Vec::new();
        let mut dense = Vec::new();
        for ((code, expected_value), dense_value) in block_codes
            .into_iter()
            .zip(block_values)
            .zip(dense_block_values)
        {
            packed.push(127);
            packed.extend(std::iter::repeat_n(code, FP8_E4M3_BLOCK_SIZE));
            expected.extend(std::iter::repeat_n(expected_value, FP8_E4M3_BLOCK_SIZE));
            dense.extend(std::iter::repeat_n(dense_value, FP8_E4M3_BLOCK_SIZE));
        }
        for value in rope_pattern.into_iter().cycle().take(ROPE_DIM) {
            packed.extend_from_slice(&half::bf16::from_f32(value).to_bits().to_le_bytes());
            expected.push(value);
            dense.push(value);
        }
        assert_eq!(expected.len(), DIM);
        assert_eq!(dense.len(), DIM);
        assert_eq!(
            packed.len(),
            (NON_ROPE_DIM / FP8_E4M3_BLOCK_SIZE) * (FP8_E4M3_PACKED_BYTES + 1)
                + ROPE_DIM * std::mem::size_of::<u16>()
        );

        let query = Owned::f32(&[1, 1, 1, DIM], &vec![0.0; DIM]);
        let packed_cache = Owned::u8(&[1, 1, 1, packed.len()], &packed);
        let dense_cache = Owned::f32(&[1, 1, 1, DIM], &dense);
        let indices = Owned::i32(&[1, 1, 1, 1], &[0]);
        let sink = Owned::f32(&[1], &[f32::NEG_INFINITY]);
        let shapes = vec![
            query.shape.clone(),
            packed_cache.shape.clone(),
            indices.shape.clone(),
            sink.shape.clone(),
        ];
        let kernel =
            kernel_with_rope_dim(128, Some("fp8_e4m3_block64"), ROPE_DIM, &shapes).unwrap();
        let mut output = Owned::zeros_f32(&query.shape);
        kernel
            .execute(
                &[
                    query.view(),
                    packed_cache.view(),
                    indices.view(),
                    sink.view(),
                ],
                &mut [output.view_mut()],
            )
            .unwrap();
        let actual = output.to_f32();
        assert_eq!(actual, expected);

        let dense_output = run_reference(128, None, &query, &dense_cache, &indices, &sink);
        for d in 0..NON_ROPE_DIM {
            assert!(
                (actual[d] - dense_output[d]).abs() <= FP8_HALF_ULP_AT_TWO,
                "non-RoPE dim {d}: compressed={} dense={}",
                actual[d],
                dense_output[d]
            );
        }
        for d in NON_ROPE_DIM..DIM {
            assert_eq!(
                actual[d], dense_output[d],
                "RoPE dim {d} must bypass FP8 quantization"
            );
        }
    }

    #[test]
    fn fp4_compressed_cache_quantizes_nontrivial_values_with_e2m1_bound() {
        let dim = FP4_E2M1_BLOCK_SIZE;
        let query = Owned::f32(
            &[1, 1, 1, dim],
            &(0..dim)
                .map(|d| 0.05 + (d % 11) as f32 * 0.0125)
                .collect::<Vec<_>>(),
        );
        let indices = Owned::i32(&[1, 1, 1, 2], &[0, 1]);
        let sink = Owned::f32(&[1], &[-0.5]);
        let mut packed = Vec::new();
        let mut dense = Vec::new();
        let mut observed_quantization_error = false;
        for record in 0..2 {
            let source = (0..dim)
                .map(|d| {
                    let magnitude = 0.3 + ((d * 7 + record * 5) % 24) as f32 * 0.23;
                    if (d + record).is_multiple_of(3) {
                        -magnitude
                    } else {
                        magnitude
                    }
                })
                .collect::<Vec<_>>();
            let amax = source
                .iter()
                .map(|value| value.abs())
                .fold(6.0 * 2.0f32.powi(-126), f32::max);
            let scale_power = (amax / 6.0).log2().ceil() as i32;
            let scale = 2.0f32.powi(scale_power);
            packed.push((scale_power + 127) as u8);
            for pair in source.chunks_exact(2) {
                let mut byte = 0u8;
                for (nibble, &value) in pair.iter().enumerate() {
                    let normalized = (value / scale).clamp(-6.0, 6.0);
                    let (code, quantized) = (0u8..16)
                        .map(|code| (code, super::super::block_dequant::decode_e2m1(code)))
                        .min_by(|(_, left), (_, right)| {
                            (normalized - *left)
                                .abs()
                                .total_cmp(&(normalized - *right).abs())
                        })
                        .unwrap();
                    byte |= code << (nibble * 4);
                    let dequantized = quantized * scale;
                    let error = (dequantized - value).abs();
                    observed_quantization_error |= error > 0.0;
                    assert!(
                        error <= scale,
                        "FP4 error {error} exceeds half of the maximum E2M1 gap {}",
                        scale
                    );
                    dense.push(dequantized);
                }
                packed.push(byte);
            }
        }
        assert!(observed_quantization_error);
        let dense_cache = Owned::f32(&[1, 1, 2, dim], &dense);
        let packed_cache = Owned::u8(&[1, 1, 2, FP4_E2M1_PACKED_BYTES + 1], &packed);

        let expected = run_reference(128, None, &query, &dense_cache, &indices, &sink);
        let actual = run_reference(
            128,
            Some("fp4_e2m1_block32"),
            &query,
            &packed_cache,
            &indices,
            &sink,
        );
        assert_eq!(actual, expected);
    }

    #[test]
    fn ratio128_stateful_carry_matches_full_recompute_across_decode_boundary() {
        const DIM: usize = 512;
        const ROPE_DIM: usize = 64;
        const RATIO: usize = 128;
        const BLOCK_START: usize = 128;
        const STORED_WIDTH: usize =
            ((DIM - ROPE_DIM) / FP8_E4M3_BLOCK_SIZE) * (FP8_E4M3_PACKED_BYTES + 1) + ROPE_DIM * 2;

        fn compressor_value(position: usize, d: usize) -> f32 {
            0.4 + (position % RATIO) as f32 * 0.00625
                + (position / RATIO) as f32 * 0.03125
                + (d % 23) as f32 * 0.009
                + ((position * 11 + d * 3) % 7) as f32 * 0.001
        }

        fn compressor_score(position: usize, d: usize) -> f32 {
            ((position * 3 + d * 5) % 19) as f32 * 0.0625 - 0.5625
                + ((position + d) % 3) as f32 * 0.015625
        }

        fn ape_value(slot: usize, d: usize) -> f32 {
            0.03125 + ((slot * 5 + d * 7) % 17) as f32 * 0.0078125 - 0.0625
        }

        fn query_value(position: usize, d: usize) -> f32 {
            0.01 + ((position * 17 + d * 13) % 37) as f32 * 0.00025
        }

        fn kv_value(position: usize, d: usize) -> f32 {
            0.2 + ((position * 7 + d * 11) % 41) as f32 * 0.0125 + (position % 5) as f32 * 0.003
        }

        fn rows(start: usize, count: usize, value: impl Fn(usize, usize) -> f32) -> Vec<f32> {
            let mut values = Vec::with_capacity(count * DIM);
            for position in start..start + count {
                for d in 0..DIM {
                    values.push(value(position, d));
                }
            }
            values
        }

        fn initial_carry() -> Owned {
            let mut values = vec![0.0f32; RATIO * 2 * DIM];
            for slot in 0..RATIO {
                for d in 0..DIM {
                    values[(slot * 2 + 1) * DIM + d] = f32::NEG_INFINITY;
                }
            }
            Owned::f32(&[1, RATIO, 2, DIM], &values)
        }

        fn expected_carry(position: usize) -> Vec<f32> {
            let mut values = initial_carry().to_f32();
            if (position + 1).is_multiple_of(RATIO) {
                return values;
            }
            let block_start = position - position % RATIO;
            for absolute in block_start..=position {
                let slot = absolute % RATIO;
                for d in 0..DIM {
                    values[(slot * 2) * DIM + d] = compressor_value(absolute, d);
                    values[(slot * 2 + 1) * DIM + d] = compressor_score(absolute, d);
                }
            }
            values
        }

        fn oracle_decode_e4m3(code: u8) -> f32 {
            let sign = if code & 0x80 == 0 { 1.0 } else { -1.0 };
            let exponent = (code >> 3) & 0x0f;
            let mantissa = code & 0x07;
            let magnitude = if exponent == 0 {
                f32::from(mantissa) * 2.0f32.powi(-9)
            } else {
                (1.0 + f32::from(mantissa) / 8.0) * 2.0f32.powi(i32::from(exponent) - 7)
            };
            sign * magnitude
        }

        fn oracle_encode_e4m3(value: f32) -> u8 {
            let mut best_code = 0u8;
            let mut best_distance = f32::INFINITY;
            for code in 0u8..=0xfe {
                if code == 0x7f {
                    continue;
                }
                let distance = (value - oracle_decode_e4m3(code)).abs();
                if distance < best_distance
                    || (distance == best_distance && code & 1 == 0 && best_code & 1 != 0)
                {
                    best_code = code;
                    best_distance = distance;
                }
            }
            best_code
        }

        fn oracle_record(
            block_start: usize,
            norm: &[f32],
        ) -> (Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>) {
            let mut pooled = vec![0.0f32; DIM];
            for (d, destination) in pooled.iter_mut().enumerate() {
                let maximum = (block_start..block_start + RATIO)
                    .map(|position| compressor_score(position, d))
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut numerator = 0.0f32;
                let mut denominator = 0.0f32;
                for position in block_start..block_start + RATIO {
                    let weight = (compressor_score(position, d) - maximum).exp();
                    numerator += weight * compressor_value(position, d);
                    denominator += weight;
                }
                *destination = numerator / denominator;
            }

            let mut finalized = pooled
                .iter()
                .map(|&value| half::bf16::from_f32(value).to_f32())
                .collect::<Vec<_>>();
            let square_sum = finalized.iter().map(|value| value * value).sum::<f32>();
            let inverse_rms = (square_sum / DIM as f32 + 1.0e-6).sqrt().recip();
            for (value, &weight) in finalized.iter_mut().zip(norm) {
                *value = half::bf16::from_f32(*value * inverse_rms * weight).to_f32();
            }
            let pre_rope = finalized.clone();

            const BASE: f32 = 160_000.0;
            const FACTOR: f32 = 16.0;
            const LOW: f32 = 15.0;
            const HIGH: f32 = 25.0;
            let tail = &mut finalized[DIM - ROPE_DIM..];
            for pair in 0..ROPE_DIM / 2 {
                let ramp = ((pair as f32 - LOW) / (HIGH - LOW)).clamp(0.0, 1.0);
                let base_frequency = BASE.powf(-((2 * pair) as f32) / ROPE_DIM as f32);
                let frequency = base_frequency * (1.0 - ramp) + base_frequency / FACTOR * ramp;
                let (sin, cos) = (block_start as f32 * frequency).sin_cos();
                let real = tail[pair * 2];
                let imaginary = tail[pair * 2 + 1];
                tail[pair * 2] = half::bf16::from_f32(real * cos - imaginary * sin).to_f32();
                tail[pair * 2 + 1] = half::bf16::from_f32(real * sin + imaginary * cos).to_f32();
            }

            let pre_fp8 = finalized.clone();
            let mut packed = Vec::with_capacity(STORED_WIDTH);
            for block in finalized[..DIM - ROPE_DIM].chunks_exact_mut(FP8_E4M3_BLOCK_SIZE) {
                let amax = block
                    .iter()
                    .map(|value| value.abs())
                    .fold(1.0e-4f32, f32::max);
                let scale_power = (amax / 448.0).log2().ceil() as i32;
                let scale = 2.0f32.powi(scale_power);
                packed.push((scale_power + 127) as u8);
                for value in block {
                    let code = oracle_encode_e4m3((*value / scale).clamp(-448.0, 448.0));
                    packed.push(code);
                    *value = oracle_decode_e4m3(code) * scale;
                }
            }
            for value in &finalized[DIM - ROPE_DIM..] {
                packed.extend_from_slice(&half::bf16::from_f32(*value).to_bits().to_le_bytes());
            }
            (packed, finalized, pre_fp8, pre_rope)
        }

        let ape_values = rows(0, RATIO, ape_value);
        let norm_values = (0..DIM)
            .map(|d| 0.75 + (d % 17) as f32 * 0.03125)
            .collect::<Vec<_>>();
        let ape = Owned::f32(&[RATIO, DIM], &ape_values);
        let norm = Owned::f32(&[DIM], &norm_values);
        let sink = Owned::f32(&[1], &[-0.375]);
        let (past_bytes, _, _, _) = oracle_record(0, &norm_values);
        let past_cache = Owned::u8(&[1, 1, STORED_WIDTH], &past_bytes);
        let (expected_record, expected_logical, pre_fp8, pre_rope) =
            oracle_record(BLOCK_START, &norm_values);
        assert_eq!(
            &expected_record[..8],
            &[119, 115, 115, 116, 116, 117, 118, 118],
            "first block must use E8M0 scale 2^-8 and the contract-derived E4M3 codes"
        );
        assert_eq!(
            &expected_record[STORED_WIDTH - ROPE_DIM * 2..][..8],
            &[173, 191, 51, 186, 117, 63, 143, 63],
            "block-start 128 must produce the contract-derived BF16 RoPE bytes"
        );

        let prefill_query = Owned::f32(&[1, 126, 1, DIM], &rows(BLOCK_START, 126, query_value));
        let prefill_kv = Owned::f32(
            &[1, BLOCK_START + 126, DIM],
            &rows(0, BLOCK_START + 126, kv_value),
        );
        let prefill_compressor =
            Owned::f32(&[1, 126, DIM], &rows(BLOCK_START, 126, compressor_value));
        let prefill_gate = Owned::f32(
            &[1, 126, DIM],
            &rows(BLOCK_START, 126, |position, d| {
                compressor_score(position, d) - ape_value(position % RATIO, d)
            }),
        );
        let prefill = run_ratio128_stateful(
            &prefill_query,
            &prefill_kv,
            &prefill_compressor,
            &prefill_gate,
            &ape,
            &norm,
            &past_cache,
            &initial_carry(),
            &Owned::i32(&[1], &[253]),
            &Owned::i64(&[], &[254]),
            &sink,
        );
        assert_eq!(prefill.cache.to_u8(), past_bytes);
        assert_eq!(prefill.carry.to_f32(), expected_carry(253));

        let mut incremental = prefill;
        let mut decode_outputs = Vec::new();
        for position in 254usize..=256 {
            let window_start = position.saturating_sub(127);
            let window_len = position - window_start + 1;
            let query = Owned::f32(&[1, 1, 1, DIM], &rows(position, 1, query_value));
            let current_kv = Owned::f32(
                &[1, window_len, DIM],
                &rows(window_start, window_len, kv_value),
            );
            let compressor = Owned::f32(&[1, 1, DIM], &rows(position, 1, compressor_value));
            let gate = Owned::f32(
                &[1, 1, DIM],
                &rows(position, 1, |absolute, d| {
                    compressor_score(absolute, d) - ape_value(absolute % RATIO, d)
                }),
            );
            incremental = run_ratio128_stateful(
                &query,
                &current_kv,
                &compressor,
                &gate,
                &ape,
                &norm,
                &incremental.cache,
                &incremental.carry,
                &Owned::i32(&[1], &[position as i32]),
                &Owned::i64(&[], &[(position + 1) as i64]),
                &sink,
            );
            decode_outputs.push(incremental.y.to_f32());
            assert_eq!(
                incremental.carry.to_f32(),
                expected_carry(position),
                "complete carry mismatch after absolute position {position}"
            );
        }

        let full_query = Owned::f32(&[1, 129, 1, DIM], &rows(BLOCK_START, 129, query_value));
        let full_kv = Owned::f32(
            &[1, BLOCK_START + 129, DIM],
            &rows(0, BLOCK_START + 129, kv_value),
        );
        let full_compressor = Owned::f32(&[1, 129, DIM], &rows(BLOCK_START, 129, compressor_value));
        let full_gate = Owned::f32(
            &[1, 129, DIM],
            &rows(BLOCK_START, 129, |position, d| {
                compressor_score(position, d) - ape_value(position % RATIO, d)
            }),
        );
        let full = run_ratio128_stateful(
            &full_query,
            &full_kv,
            &full_compressor,
            &full_gate,
            &ape,
            &norm,
            &past_cache,
            &initial_carry(),
            &Owned::i32(&[1], &[256]),
            &Owned::i64(&[], &[257]),
            &sink,
        );

        let full_y = full.y.to_f32();
        for (step, actual) in decode_outputs.iter().enumerate() {
            let full_offset = (126 + step) * DIM;
            assert_eq!(
                actual.as_slice(),
                &full_y[full_offset..full_offset + DIM],
                "incremental/full attention mismatch at decode step {step}"
            );
        }
        assert_eq!(incremental.carry.to_f32(), full.carry.to_f32());
        assert_eq!(incremental.cache.to_u8(), full.cache.to_u8());
        let cache_bytes = incremental.cache.to_u8();
        assert_eq!(&cache_bytes[..STORED_WIDTH], past_bytes.as_slice());
        assert_eq!(&cache_bytes[STORED_WIDTH..], expected_record.as_slice());

        let decoded = dequantize_cache(
            &incremental.cache.view(),
            [1, 1, 2, STORED_WIDTH],
            DIM,
            ROPE_DIM,
            CacheFormat::Fp8E4m3Block64,
        )
        .unwrap();
        assert_eq!(&decoded[DIM..], expected_logical.as_slice());

        let mut observed_quantization_error = false;
        for block_start in (0..DIM - ROPE_DIM).step_by(FP8_E4M3_BLOCK_SIZE) {
            let scale = super::super::block_dequant::decode_e8m0_scale(
                expected_record[block_start / FP8_E4M3_BLOCK_SIZE * (FP8_E4M3_PACKED_BYTES + 1)],
            );
            let absolute_bound = 16.0 * scale;
            for d in block_start..block_start + FP8_E4M3_BLOCK_SIZE {
                let error = (expected_logical[d] - pre_fp8[d]).abs();
                observed_quantization_error |= error > 0.0;
                assert!(
                    error <= absolute_bound,
                    "FP8 dim {d} error {error} exceeds half of the maximum E4M3 ULP {absolute_bound}"
                );
            }
        }
        assert!(observed_quantization_error);
        assert_eq!(
            &expected_logical[DIM - ROPE_DIM..],
            &pre_fp8[DIM - ROPE_DIM..],
            "BF16 RoPE tail must bypass FP8"
        );
        assert_ne!(
            expected_logical[DIM - ROPE_DIM],
            pre_rope[DIM - ROPE_DIM],
            "nonzero block-start RoPE must rotate the first tail pair"
        );
    }

    #[test]
    fn ratio4_index_compression_topk_and_streaming_match_independent_oracle() {
        const D: usize = 512;
        const RD: usize = 64;
        const ID: usize = 128;
        const IH: usize = 2;
        const R: usize = 4;
        const S: usize = 16;
        const TOPK: usize = 2;
        const MAIN_WIDTH: usize =
            ((D - RD) / FP8_E4M3_BLOCK_SIZE) * (FP8_E4M3_PACKED_BYTES + 1) + RD * 2;
        const INDEX_WIDTH: usize = (ID / FP4_E2M1_BLOCK_SIZE) * (FP4_E2M1_PACKED_BYTES + 1);

        fn main_value(position: usize, channel: usize, d: usize) -> f32 {
            0.35 + position as f32 * 0.017 + channel as f32 * 0.11 + (d % 29) as f32 * 0.0065
        }
        fn main_score(position: usize, channel: usize, d: usize) -> f32 {
            ((position * 7 + channel * 11 + d * 3) % 23) as f32 * 0.041 - 0.37
        }
        fn index_value(position: usize, channel: usize, d: usize) -> f32 {
            0.22 + position as f32 * 0.023 + channel as f32 * 0.085 + (d % 19) as f32 * 0.008
        }
        fn index_score(position: usize, channel: usize, d: usize) -> f32 {
            ((position * 13 + channel * 5 + d * 7) % 31) as f32 * 0.027 - 0.29
        }
        fn main_ape(slot: usize, source_dim: usize) -> f32 {
            ((slot * 17 + source_dim * 3) % 37) as f32 * 0.002 - 0.031
        }
        fn index_ape(slot: usize, source_dim: usize) -> f32 {
            ((slot * 11 + source_dim * 5) % 29) as f32 * 0.0025 - 0.027
        }
        fn main_carried_score(position: usize, channel: usize, d: usize) -> f32 {
            let source_dim = channel * D + d;
            let ape = main_ape(position % R, source_dim);
            (main_score(position, channel, d) - ape) + ape
        }
        fn index_carried_score(position: usize, channel: usize, d: usize) -> f32 {
            let source_dim = channel * ID + d;
            let ape = index_ape(position % R, source_dim);
            (index_score(position, channel, d) - ape) + ape
        }
        fn query_value(position: usize, d: usize) -> f32 {
            0.012 + ((position * 19 + d * 7) % 43) as f32 * 0.0007
        }
        fn dense_value(position: usize, d: usize) -> f32 {
            0.18 + ((position * 5 + d * 11) % 47) as f32 * 0.009
        }
        fn index_query_value(position: usize, head: usize, d: usize) -> f32 {
            0.08 + position as f32 * 0.0017
                + head as f32 * 0.013
                + ((d * 9 + position * 3 + head) % 41) as f32 * 0.003
        }
        fn index_weight_value(position: usize, head: usize) -> f32 {
            0.65 + position as f32 * 0.009 + head as f32 * 0.17
        }
        fn rows(
            start: usize,
            count: usize,
            width: usize,
            value: impl Fn(usize, usize) -> f32,
        ) -> Vec<f32> {
            let mut values = Vec::with_capacity(count * width);
            for position in start..start + count {
                for d in 0..width {
                    values.push(value(position, d));
                }
            }
            values
        }
        fn compressor_rows(
            start: usize,
            count: usize,
            width: usize,
            value: impl Fn(usize, usize, usize) -> f32,
        ) -> Vec<f32> {
            let mut values = Vec::with_capacity(count * width * 2);
            for position in start..start + count {
                for source_dim in 0..width * 2 {
                    values.push(value(position, source_dim / width, source_dim % width));
                }
            }
            values
        }
        fn gate_rows(
            start: usize,
            count: usize,
            width: usize,
            score: impl Fn(usize, usize, usize) -> f32,
            ape: impl Fn(usize, usize) -> f32,
        ) -> Vec<f32> {
            compressor_rows(start, count, width, |position, channel, d| {
                let source_dim = channel * width + d;
                score(position, channel, d) - ape(position % R, source_dim)
            })
        }
        fn ape_rows(width: usize, value: impl Fn(usize, usize) -> f32) -> Vec<f32> {
            let mut values = Vec::with_capacity(R * width * 2);
            for slot in 0..R {
                for d in 0..width * 2 {
                    values.push(value(slot, d));
                }
            }
            values
        }
        fn initial_carry(width: usize) -> Owned {
            let mut values = vec![0.0f32; 8 * 2 * width * 2];
            for slot in 0..8 {
                for d in 0..width * 2 {
                    values[(slot * 2 + 1) * width * 2 + d] = f32::NEG_INFINITY;
                }
            }
            Owned::f32(&[1, 8, 2, width * 2], &values)
        }
        fn expected_carry(
            position: usize,
            width: usize,
            value: impl Fn(usize, usize, usize) -> f32,
            score: impl Fn(usize, usize, usize) -> f32,
        ) -> Vec<f32> {
            let mut result = initial_carry(width).to_f32();
            let completed = (position + 1) / R * R;
            if completed >= R {
                for slot in 0..R {
                    let absolute = completed - R + slot;
                    for source_dim in 0..width * 2 {
                        let channel = source_dim / width;
                        let d = source_dim % width;
                        result[(slot * 2) * width * 2 + source_dim] = value(absolute, channel, d);
                        result[(slot * 2 + 1) * width * 2 + source_dim] =
                            score(absolute, channel, d);
                    }
                }
            }
            for absolute in completed..=position {
                let slot = R + absolute % R;
                for source_dim in 0..width * 2 {
                    let channel = source_dim / width;
                    let d = source_dim % width;
                    result[(slot * 2) * width * 2 + source_dim] = value(absolute, channel, d);
                    result[(slot * 2 + 1) * width * 2 + source_dim] = score(absolute, channel, d);
                }
            }
            result
        }
        fn oracle_pool(
            block_start: usize,
            width: usize,
            value: impl Fn(usize, usize, usize) -> f32,
            score: impl Fn(usize, usize, usize) -> f32,
        ) -> Vec<f32> {
            let mut pooled = vec![0.0f32; width];
            for (d, pooled_value) in pooled.iter_mut().enumerate() {
                let mut candidates = Vec::with_capacity(8);
                if block_start >= R {
                    for position in block_start - R..block_start {
                        candidates.push((value(position, 0, d), score(position, 0, d)));
                    }
                }
                for position in block_start..block_start + R {
                    candidates.push((value(position, 1, d), score(position, 1, d)));
                }
                let maximum = candidates
                    .iter()
                    .map(|(_, score)| *score)
                    .fold(f32::NEG_INFINITY, f32::max);
                let denominator = candidates
                    .iter()
                    .map(|(_, score)| (*score - maximum).exp())
                    .sum::<f32>();
                *pooled_value = candidates
                    .iter()
                    .map(|(value, score)| value * (*score - maximum).exp())
                    .sum::<f32>()
                    / denominator;
            }
            pooled
        }
        fn oracle_rope(values: &mut [f32], position: usize) {
            for pair in 0..values.len() / 2 {
                let ramp = ((pair as f32 - 15.0) / 10.0).clamp(0.0, 1.0);
                let base = 160_000.0f32.powf(-((2 * pair) as f32) / values.len() as f32);
                let frequency = base * (1.0 - ramp) + base / 16.0 * ramp;
                let (sin, cos) = (position as f32 * frequency).sin_cos();
                let real = values[pair * 2];
                let imaginary = values[pair * 2 + 1];
                values[pair * 2] = half::bf16::from_f32(real * cos - imaginary * sin).to_f32();
                values[pair * 2 + 1] = half::bf16::from_f32(real * sin + imaginary * cos).to_f32();
            }
        }
        fn oracle_norm(mut values: Vec<f32>, norm: &[f32]) -> Vec<f32> {
            for value in &mut values {
                *value = half::bf16::from_f32(*value).to_f32();
            }
            let inverse = (values.iter().map(|value| value * value).sum::<f32>()
                / values.len() as f32
                + 1.0e-6)
                .sqrt()
                .recip();
            for (value, weight) in values.iter_mut().zip(norm) {
                *value = half::bf16::from_f32(*value * inverse * weight).to_f32();
            }
            values
        }
        fn oracle_hadamard(values: &mut [f32]) {
            let mut span = 1;
            while span < values.len() {
                for start in (0..values.len()).step_by(span * 2) {
                    for offset in 0..span {
                        let a = values[start + offset];
                        let b = values[start + offset + span];
                        values[start + offset] = a + b;
                        values[start + offset + span] = a - b;
                    }
                }
                span *= 2;
            }
            let scale = 1.0 / (values.len() as f32).sqrt();
            for value in values {
                *value = half::bf16::from_f32(*value * scale).to_f32();
            }
        }
        fn decode_fp4(code: u8) -> f32 {
            [
                0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0,
                -6.0,
            ][usize::from(code & 15)]
        }
        fn encode_fp4(value: f32) -> u8 {
            (0u8..16)
                .min_by(|left, right| {
                    let left_distance = (value - decode_fp4(*left)).abs();
                    let right_distance = (value - decode_fp4(*right)).abs();
                    left_distance
                        .total_cmp(&right_distance)
                        .then_with(|| (left & 1).cmp(&(right & 1)))
                })
                .unwrap()
        }
        fn oracle_fp4(values: &mut [f32]) -> Vec<u8> {
            let mut packed = Vec::new();
            for block in values.chunks_exact_mut(32) {
                let amax = block
                    .iter()
                    .map(|value| value.abs())
                    .fold(6.0 * 2.0f32.powi(-126), f32::max);
                let power = (amax / 6.0).log2().ceil() as i32;
                let scale = 2.0f32.powi(power);
                packed.push((power + 127) as u8);
                for pair in block.chunks_exact_mut(2) {
                    let low = encode_fp4((pair[0] / scale).clamp(-6.0, 6.0));
                    let high = encode_fp4((pair[1] / scale).clamp(-6.0, 6.0));
                    packed.push(low | high << 4);
                    pair[0] = decode_fp4(low) * scale;
                    pair[1] = decode_fp4(high) * scale;
                }
            }
            packed
        }
        fn decode_fp8(code: u8) -> f32 {
            let sign = if code & 0x80 == 0 { 1.0 } else { -1.0 };
            let exponent = (code >> 3) & 15;
            let mantissa = code & 7;
            let magnitude = if exponent == 0 {
                f32::from(mantissa) * 2.0f32.powi(-9)
            } else {
                (1.0 + f32::from(mantissa) / 8.0) * 2.0f32.powi(i32::from(exponent) - 7)
            };
            sign * magnitude
        }
        fn encode_fp8(value: f32) -> u8 {
            (0u8..=0xfe)
                .filter(|&code| code != 0x7f)
                .min_by(|left, right| {
                    let left_distance = (value - decode_fp8(*left)).abs();
                    let right_distance = (value - decode_fp8(*right)).abs();
                    left_distance
                        .total_cmp(&right_distance)
                        .then_with(|| (left & 1).cmp(&(right & 1)))
                })
                .unwrap()
        }
        fn oracle_main_record(block_start: usize, norm: &[f32]) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
            let pooled = oracle_pool(block_start, D, main_value, main_score);
            let mut record = oracle_norm(pooled, norm);
            oracle_rope(&mut record[D - RD..], block_start);
            let pre_fp8 = record.clone();
            let mut packed = Vec::new();
            for block in record[..D - RD].chunks_exact_mut(64) {
                let amax = block
                    .iter()
                    .map(|value| value.abs())
                    .fold(1.0e-4f32, f32::max);
                let power = (amax / 448.0).log2().ceil() as i32;
                let scale = 2.0f32.powi(power);
                packed.push((power + 127) as u8);
                for value in block {
                    let code = encode_fp8((*value / scale).clamp(-448.0, 448.0));
                    packed.push(code);
                    *value = decode_fp8(code) * scale;
                }
            }
            for value in &record[D - RD..] {
                packed.extend_from_slice(&half::bf16::from_f32(*value).to_bits().to_le_bytes());
            }
            (packed, record, pre_fp8)
        }
        fn oracle_index_record(
            block_start: usize,
            norm: &[f32],
        ) -> (Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>) {
            let pooled = oracle_pool(block_start, ID, index_value, index_score);
            let mut record = oracle_norm(pooled, norm);
            let pre_rope = record.clone();
            oracle_rope(&mut record[ID - RD..], block_start);
            oracle_hadamard(&mut record);
            let pre_fp4 = record.clone();
            let packed = oracle_fp4(&mut record);
            (packed, record, pre_fp4, pre_rope)
        }
        fn oracle_index_query(position: usize, head: usize) -> Vec<f32> {
            let mut values = (0..ID)
                .map(|d| half::bf16::from_f32(index_query_value(position, head, d)).to_f32())
                .collect::<Vec<_>>();
            oracle_rope(&mut values[ID - RD..], position);
            oracle_hadamard(&mut values);
            let _ = oracle_fp4(&mut values);
            values
        }

        let main_norm = (0..D)
            .map(|d| 0.72 + (d % 21) as f32 * 0.018)
            .collect::<Vec<_>>();
        let index_norm = (0..ID)
            .map(|d| 0.81 + (d % 13) as f32 * 0.021)
            .collect::<Vec<_>>();
        let query = Owned::f32(&[1, S, 1, D], &rows(0, S, D, query_value));
        let current_kv = Owned::f32(&[1, S, D], &rows(0, S, D, dense_value));
        let main_kv = Owned::f32(&[1, S, D * 2], &compressor_rows(0, S, D, main_value));
        let main_gate = Owned::f32(&[1, S, D * 2], &gate_rows(0, S, D, main_score, main_ape));
        let main_ape_tensor = Owned::f32(&[R, D * 2], &ape_rows(D, main_ape));
        let main_norm_tensor = Owned::f32(&[D], &main_norm);
        let mut index_query_values = Vec::with_capacity(S * IH * ID);
        let mut index_weight_values = Vec::with_capacity(S * IH);
        for position in 0..S {
            for head in 0..IH {
                for d in 0..ID {
                    index_query_values.push(index_query_value(position, head, d));
                }
                index_weight_values.push(index_weight_value(position, head));
            }
        }
        let index_query = Owned::f32(&[1, S, IH, ID], &index_query_values);
        let index_weight = Owned::f32(&[1, S, IH], &index_weight_values);
        let index_kv = Owned::f32(&[1, S, ID * 2], &compressor_rows(0, S, ID, index_value));
        let index_gate = Owned::f32(
            &[1, S, ID * 2],
            &gate_rows(0, S, ID, index_score, index_ape),
        );
        let index_ape_tensor = Owned::f32(&[R, ID * 2], &ape_rows(ID, index_ape));
        let index_norm_tensor = Owned::f32(&[ID], &index_norm);
        let empty_main = Owned::u8(&[1, 0, MAIN_WIDTH], &[]);
        let empty_index = Owned::u8(&[1, 0, INDEX_WIDTH], &[]);
        let full = run_ratio4_stateful(
            &query,
            &current_kv,
            &main_kv,
            &main_gate,
            &main_ape_tensor,
            &main_norm_tensor,
            &empty_main,
            &initial_carry(D),
            &Owned::i32(&[1], &[S as i32 - 1]),
            &Owned::i64(&[], &[S as i64]),
            &Owned::f32(&[1], &[-0.41]),
            &index_query,
            &index_weight,
            &index_kv,
            &index_gate,
            &index_ape_tensor,
            &index_norm_tensor,
            &empty_index,
            &initial_carry(ID),
            TOPK,
        );

        let mut expected_main = Vec::new();
        let mut expected_index = Vec::new();
        let mut index_records = Vec::new();
        for block_start in (0..S).step_by(R) {
            let (packed, logical, pre_fp8) = oracle_main_record(block_start, &main_norm);
            for block in 0..(D - RD) / 64 {
                let scale = 2.0f32.powi(i32::from(packed[block * 65]) - 127);
                for d in block * 64..block * 64 + 64 {
                    assert!(
                        (logical[d] - pre_fp8[d]).abs() <= 16.0 * scale,
                        "ratio-4 FP8 error exceeds calibrated 16x-scale bound at block {block_start}, dim {d}"
                    );
                }
            }
            expected_main.extend(packed);

            let (packed, logical, pre_fp4, pre_rope) =
                oracle_index_record(block_start, &index_norm);
            for block in 0..ID / 32 {
                let scale = 2.0f32.powi(i32::from(packed[block * 17]) - 127);
                for d in block * 32..block * 32 + 32 {
                    assert!(
                        (logical[d] - pre_fp4[d]).abs() <= scale,
                        "ratio-4 FP4 error exceeds calibrated 1x-scale bound at block {block_start}, dim {d}"
                    );
                }
            }
            if block_start != 0 {
                assert_ne!(
                    &pre_fp4[ID - RD..ID - RD + 2],
                    &pre_rope[ID - RD..ID - RD + 2],
                    "nonzero compressed block start must affect the RoPE tail before Hadamard"
                );
            }
            expected_index.extend(packed);
            index_records.push(logical);
        }
        assert_eq!(full.cache.to_u8(), expected_main);
        assert_eq!(full.index.to_u8(), expected_index);
        let expected_main_carry = expected_carry(S - 1, D, main_value, main_carried_score);
        let actual_main_carry = full.carry.to_f32();
        assert_eq!(actual_main_carry.len(), expected_main_carry.len());
        for (offset, (actual, expected)) in actual_main_carry
            .iter()
            .zip(&expected_main_carry)
            .enumerate()
        {
            assert_eq!(actual, expected, "main carry mismatch at {offset}");
        }
        let expected_index_carry = expected_carry(S - 1, ID, index_value, index_carried_score);
        let actual_index_carry = full.index_carry.to_f32();
        assert_eq!(actual_index_carry.len(), expected_index_carry.len());
        for (offset, (actual, expected)) in actual_index_carry
            .iter()
            .zip(&expected_index_carry)
            .enumerate()
        {
            assert_eq!(actual, expected, "index carry mismatch at {offset}");
        }

        let replicated_selected = full.selected.to_i32();
        assert_eq!(
            &replicated_selected[..S * TOPK],
            &replicated_selected[S * TOPK..],
            "official selection is shared after reducing index heads"
        );
        let actual_selected = replicated_selected[..S * TOPK].to_vec();
        let mut expected_selected = vec![-1i32; S * TOPK];
        for position in 0..S {
            let valid = (position + 1) / R;
            let queries = (0..IH)
                .map(|head| oracle_index_query(position, head))
                .collect::<Vec<_>>();
            let mut scores = (0..valid)
                .map(|record| {
                    let score = (0..IH)
                        .map(|head| {
                            let dot = queries[head]
                                .iter()
                                .zip(&index_records[record])
                                .map(|(q, k)| q * k)
                                .sum::<f32>();
                            dot.max(0.0) * index_weight_value(position, head)
                                / (ID as f32).sqrt()
                                / (IH as f32).sqrt()
                        })
                        .sum::<f32>();
                    (record, score)
                })
                .collect::<Vec<_>>();
            assert!(
                scores
                    .iter()
                    .enumerate()
                    .all(|(left, (_, a))| scores[left + 1..].iter().all(|(_, b)| a != b)),
                "oracle fixture must not contain unresolved top-k ties"
            );
            scores.sort_unstable_by(|left, right| right.1.total_cmp(&left.1));
            for (slot, &(record, _)) in scores.iter().take(TOPK).enumerate() {
                expected_selected[position * TOPK + slot] = record as i32;
            }
        }
        assert_eq!(actual_selected, expected_selected);
        let last = &actual_selected[(S - 1) * TOPK..S * TOPK];
        assert_eq!(last.len(), TOPK);
        assert!(
            (0..S / R)
                .filter(|record| !last.contains(&(*record as i32)))
                .count()
                >= 2,
            "top-k fixture must observably exclude coarse blocks"
        );

        let prefix_len = 6;
        let prefix = run_ratio4_stateful(
            &Owned::f32(&[1, prefix_len, 1, D], &rows(0, prefix_len, D, query_value)),
            &Owned::f32(&[1, prefix_len, D], &rows(0, prefix_len, D, dense_value)),
            &Owned::f32(
                &[1, prefix_len, D * 2],
                &compressor_rows(0, prefix_len, D, main_value),
            ),
            &Owned::f32(
                &[1, prefix_len, D * 2],
                &gate_rows(0, prefix_len, D, main_score, main_ape),
            ),
            &main_ape_tensor,
            &main_norm_tensor,
            &empty_main,
            &initial_carry(D),
            &Owned::i32(&[1], &[prefix_len as i32 - 1]),
            &Owned::i64(&[], &[prefix_len as i64]),
            &Owned::f32(&[1], &[-0.41]),
            &Owned::f32(
                &[1, prefix_len, IH, ID],
                &index_query_values[..prefix_len * IH * ID],
            ),
            &Owned::f32(
                &[1, prefix_len, IH],
                &index_weight_values[..prefix_len * IH],
            ),
            &Owned::f32(
                &[1, prefix_len, ID * 2],
                &compressor_rows(0, prefix_len, ID, index_value),
            ),
            &Owned::f32(
                &[1, prefix_len, ID * 2],
                &gate_rows(0, prefix_len, ID, index_score, index_ape),
            ),
            &index_ape_tensor,
            &index_norm_tensor,
            &empty_index,
            &initial_carry(ID),
            TOPK,
        );
        assert_eq!(
            prefix.index_carry.to_f32(),
            expected_carry(prefix_len - 1, ID, index_value, index_carried_score)
        );
        let mut incremental = prefix;
        for position in prefix_len..=7 {
            let mut one_index_query = Vec::with_capacity(IH * ID);
            let mut one_index_weight = Vec::with_capacity(IH);
            for head in 0..IH {
                for d in 0..ID {
                    one_index_query.push(index_query_value(position, head, d));
                }
                one_index_weight.push(index_weight_value(position, head));
            }
            incremental = run_ratio4_stateful(
                &Owned::f32(&[1, 1, 1, D], &rows(position, 1, D, query_value)),
                &Owned::f32(
                    &[1, position + 1, D],
                    &rows(0, position + 1, D, dense_value),
                ),
                &Owned::f32(&[1, 1, D * 2], &compressor_rows(position, 1, D, main_value)),
                &Owned::f32(
                    &[1, 1, D * 2],
                    &gate_rows(position, 1, D, main_score, main_ape),
                ),
                &main_ape_tensor,
                &main_norm_tensor,
                &incremental.cache,
                &incremental.carry,
                &Owned::i32(&[1], &[position as i32]),
                &Owned::i64(&[], &[(position + 1) as i64]),
                &Owned::f32(&[1], &[-0.41]),
                &Owned::f32(&[1, 1, IH, ID], &one_index_query),
                &Owned::f32(&[1, 1, IH], &one_index_weight),
                &Owned::f32(
                    &[1, 1, ID * 2],
                    &compressor_rows(position, 1, ID, index_value),
                ),
                &Owned::f32(
                    &[1, 1, ID * 2],
                    &gate_rows(position, 1, ID, index_score, index_ape),
                ),
                &index_ape_tensor,
                &index_norm_tensor,
                &incremental.index,
                &incremental.index_carry,
                TOPK,
            );
            assert_eq!(
                incremental.index_carry.to_f32(),
                expected_carry(position, ID, index_value, index_carried_score),
                "index carry mismatch at streaming position {position}"
            );
        }
        let full_eight = run_ratio4_stateful(
            &Owned::f32(&[1, 8, 1, D], &rows(0, 8, D, query_value)),
            &Owned::f32(&[1, 8, D], &rows(0, 8, D, dense_value)),
            &Owned::f32(&[1, 8, D * 2], &compressor_rows(0, 8, D, main_value)),
            &Owned::f32(&[1, 8, D * 2], &gate_rows(0, 8, D, main_score, main_ape)),
            &main_ape_tensor,
            &main_norm_tensor,
            &empty_main,
            &initial_carry(D),
            &Owned::i32(&[1], &[7]),
            &Owned::i64(&[], &[8]),
            &Owned::f32(&[1], &[-0.41]),
            &Owned::f32(&[1, 8, IH, ID], &index_query_values[..8 * IH * ID]),
            &Owned::f32(&[1, 8, IH], &index_weight_values[..8 * IH]),
            &Owned::f32(&[1, 8, ID * 2], &compressor_rows(0, 8, ID, index_value)),
            &Owned::f32(
                &[1, 8, ID * 2],
                &gate_rows(0, 8, ID, index_score, index_ape),
            ),
            &index_ape_tensor,
            &index_norm_tensor,
            &empty_index,
            &initial_carry(ID),
            TOPK,
        );
        assert_eq!(incremental.cache.to_u8(), full_eight.cache.to_u8());
        assert_eq!(incremental.index.to_u8(), full_eight.index.to_u8());
        assert_eq!(incremental.carry.to_f32(), full_eight.carry.to_f32());
        assert_eq!(
            incremental.index_carry.to_f32(),
            full_eight.index_carry.to_f32()
        );
        assert_eq!(incremental.y.to_f32(), full_eight.y.to_f32()[7 * D..8 * D]);
        assert_eq!(
            &incremental.selected.to_i32()[..TOPK],
            &full_eight.selected.to_i32()[7 * TOPK..8 * TOPK]
        );
    }

    #[test]
    fn ratio4_topk_ties_remain_explicitly_unsupported() {
        let message = select_ratio4_topk(
            &[0.0; 128],
            &[1.0],
            &[0.25; 256],
            [1, 1, 1, 128],
            2,
            7,
            2,
            64,
        )
        .unwrap_err()
        .to_string();
        assert!(message.contains("Unsupported"));
        assert!(message.contains("top-k tie ordering is unfrozen"));
    }

    #[test]
    fn public_v1_rejects_phase1_reference_arity() {
        let mut graph = Graph::new();
        let output = graph.create_named_value("Y", DataType::Float32, static_shape([1, 1, 2, 2]));
        let mut node = Node::new(NodeId(0), OP, vec![None; 4], vec![output]);
        node.attributes
            .insert("num_heads".into(), Attribute::Int(2));
        node.attributes.insert("head_dim".into(), Attribute::Int(2));
        node.attributes
            .insert("compression_ratio".into(), Attribute::Int(128));

        let shapes = vec![vec![]; 4];
        let message = CompressedSparseAttentionFactory
            .create(&node, &shapes)
            .err()
            .unwrap()
            .to_string();
        assert!(message.contains("frozen v1 requires 11..=20 positional inputs"));
    }
}

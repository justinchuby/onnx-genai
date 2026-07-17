//! Correctness-first reference paths for `pkg.nxrt::CompressedSparseAttention`
//! v1.
//!
//! The registered operator exposes the complete frozen stateful v1 boundary.
//! Ratio-128 owns its persistent compressed records and incremental
//! compression carry. Ratio-4 index state/top-k and the MTP sidecar remain
//! explicit Unsupported paths. The assembled-cache reference remains the
//! independently tested gather/attention seam.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::block_dequant::{
    FP4_E2M1_BLOCK_SIZE, FP4_E2M1_PACKED_BYTES, FP8_E4M3_BLOCK_SIZE, FP8_E4M3_PACKED_BYTES,
    dequantize_fp4_e2m1_block, dequantize_fp8_e4m3_block, quantize_fp8_e4m3_block,
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
                "sink_mode='{sink_mode}' is unsupported; v1 requires 'logit_only'"
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
            return Err(unsupported(
                "ratio-4 compressed-KV construction/carry updates, index-key construction/carry updates, and top-k index selection are deferred",
            ));
        }
        self.execute_ratio128(inputs, outputs)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

impl StatefulCompressedSparseAttentionKernel {
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

        for b in 0..batch {
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
                    let emitted_index = emitted_counts[b];
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
                    emitted_counts[b] = emitted_index
                        .checked_add(1)
                        .ok_or_else(|| error("emitted record count overflow"))?;
                    if start == 0 {
                        reset_ratio128_row(&mut carry, b, self.compression_ratio, dim)?;
                    }
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
            for h in 0..heads {
                scores.fill(f32::NEG_INFINITY);
                let query_row = flat4([b, s, h, 0], query_shape, "stateful query row")?;
                let mut maximum = f32::NEG_INFINITY;
                for candidate in 0..dense_candidates {
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
                    scores[candidate] = score;
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
                denominator += (sink[h] - maximum).exp();
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
            for h in 0..heads {
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
                    denominator += (sink[h] - maximum).exp();
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
        for axis in 0..4 {
            let coordinate = if self.padded_shape[axis] == 1 {
                0
            } else {
                target_index[axis]
            };
            offset = offset
                .checked_mul(self.padded_shape[axis])
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
    fn fp4_compressed_cache_matches_same_logical_f32_cache() {
        let dim = FP4_E2M1_BLOCK_SIZE;
        let query = Owned::f32(&[1, 1, 1, dim], &vec![0.125; dim]);
        let indices = Owned::i32(&[1, 1, 1, 2], &[0, 1]);
        let sink = Owned::f32(&[1], &[-0.5]);
        let record_bytes = [[0x21, 0x43, 0x65, 0xa9], [0x10, 0x32, 0x54, 0xba]];
        let mut packed = Vec::new();
        let mut dense = Vec::new();
        for bytes in record_bytes {
            packed.push(127);
            for byte in bytes.into_iter().cycle().take(FP4_E2M1_PACKED_BYTES) {
                packed.push(byte);
                dense.push(super::super::block_dequant::decode_e2m1(byte));
                dense.push(super::super::block_dequant::decode_e2m1(byte >> 4));
            }
        }
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
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-6, "{actual} != {expected}");
        }
    }

    #[test]
    fn ratio128_stateful_carry_matches_full_recompute_across_decode_boundary() {
        const DIM: usize = 512;
        const ROPE_DIM: usize = 64;
        const STORED_WIDTH: usize =
            ((DIM - ROPE_DIM) / FP8_E4M3_BLOCK_SIZE) * (FP8_E4M3_PACKED_BYTES + 1) + ROPE_DIM * 2;

        fn token_rows(start: usize, count: usize, dim: usize, compressor: bool) -> Vec<f32> {
            let mut values = Vec::with_capacity(count * dim);
            for position in start..start + count {
                let value = if compressor {
                    (position + 1) as f32
                } else {
                    (position % 7 + 1) as f32
                };
                values.extend(std::iter::repeat_n(value, dim));
            }
            values
        }

        fn initial_carry() -> Owned {
            let mut values = vec![0.0f32; 128 * 2 * DIM];
            for slot in 0..128 {
                for d in 0..DIM {
                    values[(slot * 2 + 1) * DIM + d] = f32::NEG_INFINITY;
                }
            }
            Owned::f32(&[1, 128, 2, DIM], &values)
        }

        fn expected_dense_sum(start: usize, end_inclusive: usize) -> f32 {
            (start..=end_inclusive)
                .map(|position| (position % 7 + 1) as f32)
                .sum()
        }

        let ape = Owned::f32(&[128, DIM], &vec![0.0; 128 * DIM]);
        let norm = Owned::f32(&[DIM], &vec![1.0; DIM]);
        let sink = Owned::f32(&[1], &[0.0]);
        let empty_cache = Owned::u8(&[1, 0, STORED_WIDTH], &[]);

        let prefill_query = Owned::f32(&[1, 126, 1, DIM], &vec![0.0; 126 * DIM]);
        let prefill_kv = Owned::f32(&[1, 126, DIM], &token_rows(0, 126, DIM, false));
        let prefill_compressor = Owned::f32(&[1, 126, DIM], &token_rows(0, 126, DIM, true));
        let prefill_gate = Owned::f32(&[1, 126, DIM], &vec![0.0; 126 * DIM]);
        let prefill = run_ratio128_stateful(
            &prefill_query,
            &prefill_kv,
            &prefill_compressor,
            &prefill_gate,
            &ape,
            &norm,
            &empty_cache,
            &initial_carry(),
            &Owned::i32(&[1], &[125]),
            &Owned::i64(&[], &[126]),
            &sink,
        );
        assert!(prefill.cache.to_u8().is_empty());
        let prefill_carry = prefill.carry.to_f32();
        assert_eq!(prefill_carry[(125 * 2) * DIM], 126.0);
        assert_eq!(prefill_carry[(125 * 2 + 1) * DIM], 0.0);
        assert_eq!(prefill_carry[(126 * 2 + 1) * DIM], f32::NEG_INFINITY);

        let mut incremental = prefill;
        let mut decode_outputs = Vec::new();
        for position in 126usize..=128 {
            let window_start = position.saturating_sub(127);
            let window_len = position - window_start + 1;
            let query = Owned::f32(&[1, 1, 1, DIM], &vec![0.0; DIM]);
            let current_kv = Owned::f32(
                &[1, window_len, DIM],
                &token_rows(window_start, window_len, DIM, false),
            );
            let compressor = Owned::f32(&[1, 1, DIM], &token_rows(position, 1, DIM, true));
            let gate = Owned::f32(&[1, 1, DIM], &vec![0.0; DIM]);
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
        }

        let full_query = Owned::f32(&[1, 129, 1, DIM], &vec![0.0; 129 * DIM]);
        let full_kv = Owned::f32(&[1, 129, DIM], &token_rows(0, 129, DIM, false));
        let full_compressor = Owned::f32(&[1, 129, DIM], &token_rows(0, 129, DIM, true));
        let full_gate = Owned::f32(&[1, 129, DIM], &vec![0.0; 129 * DIM]);
        let full = run_ratio128_stateful(
            &full_query,
            &full_kv,
            &full_compressor,
            &full_gate,
            &ape,
            &norm,
            &empty_cache,
            &initial_carry(),
            &Owned::i32(&[1], &[128]),
            &Owned::i64(&[], &[129]),
            &sink,
        );

        let expected = [
            expected_dense_sum(0, 126) / 128.0,
            (expected_dense_sum(0, 127) + 1.0) / 130.0,
            (expected_dense_sum(1, 128) + 1.0) / 130.0,
        ];
        let full_y = full.y.to_f32();
        for (step, (actual, expected_scalar)) in decode_outputs.iter().zip(expected).enumerate() {
            let full_offset = (126 + step) * DIM;
            for d in 0..DIM {
                assert!(
                    (actual[d] - expected_scalar).abs() <= 1e-5,
                    "decode step {step}, dim {d}: {} != {expected_scalar}",
                    actual[d]
                );
                assert!(
                    (actual[d] - full_y[full_offset + d]).abs() <= 1e-5,
                    "decode/full mismatch at step {step}, dim {d}: {} != {}",
                    actual[d],
                    full_y[full_offset + d]
                );
            }
        }
        assert_eq!(incremental.cache.to_u8(), full.cache.to_u8());
        let decoded = dequantize_cache(
            &incremental.cache.view(),
            [1, 1, 1, STORED_WIDTH],
            DIM,
            ROPE_DIM,
            CacheFormat::Fp8E4m3Block64,
        )
        .unwrap();
        assert!(decoded.iter().all(|&value| value == 1.0));
        let final_carry = incremental.carry.to_f32();
        assert_eq!(final_carry[0], 129.0);
        assert_eq!(final_carry[DIM], 0.0);
    }

    #[test]
    fn ratio4_stateful_path_remains_explicitly_unsupported() {
        let mut graph = Graph::new();
        graph.opset_imports.insert("pkg.nxrt".into(), 1);
        let inputs = FROZEN_V1_REQUIRED_INPUT_NAMES
            .iter()
            .map(|name| Some(graph.create_named_value(*name, DataType::Float32, static_shape([1]))))
            .collect();
        let outputs = ["Y", "present_compressed_kv", "present_compression_carry"]
            .into_iter()
            .map(|name| graph.create_named_value(name, DataType::Float32, static_shape([1])))
            .collect();
        let mut node = Node::new(NodeId(0), OP, inputs, outputs);
        node.domain = "pkg.nxrt".into();
        node.attributes
            .insert("num_heads".into(), Attribute::Int(2));
        node.attributes.insert("head_dim".into(), Attribute::Int(2));
        node.attributes
            .insert("compression_ratio".into(), Attribute::Int(4));
        node.attributes
            .insert("index_num_heads".into(), Attribute::Int(1));
        node.attributes
            .insert("index_head_dim".into(), Attribute::Int(128));
        node.attributes
            .insert("index_topk".into(), Attribute::Int(512));

        let shapes = vec![vec![1]; FROZEN_V1_REQUIRED_INPUTS];
        let kernel = CompressedSparseAttentionFactory
            .create(&node, &shapes)
            .unwrap();
        let owned_inputs = (0..FROZEN_V1_REQUIRED_INPUTS)
            .map(|_| Owned::f32(&[1], &[0.0]))
            .collect::<Vec<_>>();
        let input_views = owned_inputs.iter().map(Owned::view).collect::<Vec<_>>();
        let mut owned_outputs = (0..FROZEN_V1_REQUIRED_OUTPUTS)
            .map(|_| Owned::zeros_f32(&[1]))
            .collect::<Vec<_>>();
        let mut output_views = owned_outputs
            .iter_mut()
            .map(Owned::view_mut)
            .collect::<Vec<_>>();

        let message = kernel
            .execute(&input_views, &mut output_views)
            .unwrap_err()
            .to_string();
        assert!(message.contains("Unsupported"));
        assert!(message.contains("ratio-4 compressed-KV construction"));
        assert!(message.contains("index-key construction"));
        assert!(message.contains("top-k index selection"));
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

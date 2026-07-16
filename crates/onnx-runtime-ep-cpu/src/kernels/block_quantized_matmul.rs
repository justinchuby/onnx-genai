//! Correctness-first `BlockQuantizedMatMul` for native GGUF block formats.
//!
//! The packed weight tensor keeps llama.cpp's serialized block layout. MXFP4
//! decoding follows OCP MX E2M1/E8M0 and llama.cpp's `block_mxfp4`; IQ
//! decoding follows llama.cpp's native super-block layouts and audited grids.

use std::sync::OnceLock;

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};
use onnx_runtime_quantization::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XS_SIGNS, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID,
};
use rayon::prelude::*;

use super::matmul::gemm;
use super::{check_arity, to_dense_bytes, to_dense_f32, write_dense_f32};
use crate::strided::numel;

const OP: &str = "BlockQuantizedMatMul";
const DOMAIN: &str = "com.github.onnxruntime.genai";
const LAYOUT_VERSION: i64 = 1;

const MXFP4_QK: usize = 32;
const MXFP4_BLOCK_BYTES: usize = 17;
const IQ4_NL_QK: usize = 32;
const IQ4_NL_BLOCK_BYTES: usize = 18;
const IQ_SUPER_QK: usize = 256;
const IQ4_XS_BLOCK_BYTES: usize = 136;
const IQ3_S_BLOCK_BYTES: usize = 110;
const IQ3_XXS_BLOCK_BYTES: usize = 98;
const IQ2_S_BLOCK_BYTES: usize = 82;
const IQ2_XS_BLOCK_BYTES: usize = 74;
const IQ2_XXS_BLOCK_BYTES: usize = 66;
const IQ1_S_BLOCK_BYTES: usize = 50;
const IQ1_M_BLOCK_BYTES: usize = 56;
const IQ1_S_DELTA: f32 = 0.125;
const IQ1_M_DELTA: f32 = 0.125;

// OCP E2M1 values, doubled to pair with llama.cpp's half-scale E8M0 decode.
const E2M1_DOUBLED: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

// llama.cpp commit b15ca938, ggml-common.h::kvalues_iq4nl.
const IQ4_NL_CODEBOOK: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

// Vendored byte-for-byte from llama.cpp commit b15ca938, ggml-common.h.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BlockFormat {
    Mxfp4,
    Iq4Nl,
    Iq4Xs,
    Iq3S,
    Iq3Xxs,
    Iq2S,
    Iq2Xs,
    Iq2Xxs,
    Iq1S,
    Iq1M,
}

impl BlockFormat {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "mxfp4" => Ok(Self::Mxfp4),
            "iq4_nl" => Ok(Self::Iq4Nl),
            "iq4_xs" => Ok(Self::Iq4Xs),
            "iq3_s" => Ok(Self::Iq3S),
            "iq3_xxs" => Ok(Self::Iq3Xxs),
            "iq2_s" => Ok(Self::Iq2S),
            "iq2_xs" => Ok(Self::Iq2Xs),
            "iq2_xxs" => Ok(Self::Iq2Xxs),
            "iq1_s" => Ok(Self::Iq1S),
            "iq1_m" => Ok(Self::Iq1M),
            _ => Err(error(format!(
                "unsupported format '{value}'; supported formats are mxfp4, iq4_nl, iq4_xs, iq3_s, iq3_xxs, iq2_s, iq2_xs, iq2_xxs, iq1_s, and iq1_m"
            ))),
        }
    }

    fn qk(self) -> usize {
        match self {
            Self::Mxfp4 => MXFP4_QK,
            Self::Iq4Nl => IQ4_NL_QK,
            Self::Iq4Xs
            | Self::Iq3S
            | Self::Iq3Xxs
            | Self::Iq2S
            | Self::Iq2Xs
            | Self::Iq2Xxs
            | Self::Iq1S
            | Self::Iq1M => IQ_SUPER_QK,
        }
    }

    fn block_bytes(self) -> usize {
        match self {
            Self::Mxfp4 => MXFP4_BLOCK_BYTES,
            Self::Iq4Nl => IQ4_NL_BLOCK_BYTES,
            Self::Iq4Xs => IQ4_XS_BLOCK_BYTES,
            Self::Iq3S => IQ3_S_BLOCK_BYTES,
            Self::Iq3Xxs => IQ3_XXS_BLOCK_BYTES,
            Self::Iq2S => IQ2_S_BLOCK_BYTES,
            Self::Iq2Xs => IQ2_XS_BLOCK_BYTES,
            Self::Iq2Xxs => IQ2_XXS_BLOCK_BYTES,
            Self::Iq1S => IQ1_S_BLOCK_BYTES,
            Self::Iq1M => IQ1_M_BLOCK_BYTES,
        }
    }

    fn scalar_decoder(self) -> fn(&[u8], &mut [f32]) {
        match self {
            Self::Mxfp4 => decode_mxfp4_block,
            Self::Iq4Nl => decode_iq4_nl_block,
            Self::Iq4Xs => decode_iq4_xs_block,
            Self::Iq3S => decode_iq3_s_block,
            Self::Iq3Xxs => decode_iq3_xxs_block,
            Self::Iq2S => decode_iq2_s_block,
            Self::Iq2Xs => decode_iq2_xs_block,
            Self::Iq2Xxs => decode_iq2_xxs_block,
            Self::Iq1S => decode_iq1_s_block,
            Self::Iq1M => decode_iq1_m_block,
        }
    }

    fn decoder(self) -> fn(&[u8], &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            return match self {
                Self::Mxfp4 => decode_mxfp4_block_avx2_dispatch,
                Self::Iq4Nl => decode_iq4_nl_block_avx2_dispatch,
                Self::Iq4Xs => decode_iq4_xs_block_avx2_dispatch,
                _ => self.scalar_decoder(),
            };
        }
        self.scalar_decoder()
    }
}

pub struct BlockQuantizedMatMulKernel {
    k: usize,
    n: usize,
    format: BlockFormat,
    packed_b_constant: bool,
    weight_kn: OnceLock<Vec<f32>>,
}

pub struct BlockQuantizedMatMulFactory;

impl KernelFactory for BlockQuantizedMatMulFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let k = required_positive_attr(node, "K")?;
        let n = required_positive_attr(node, "N")?;
        let layout_version = optional_int_attr(node, "block_layout_version")?.unwrap_or(1);
        if layout_version != LAYOUT_VERSION {
            return Err(error(format!(
                "block_layout_version must be {LAYOUT_VERSION}, got {layout_version}"
            )));
        }
        let format = match node.attr("format") {
            Some(attribute) => attribute
                .as_str()
                .ok_or_else(|| error("attribute 'format' must be a UTF-8 string"))
                .and_then(BlockFormat::parse)?,
            None => return Err(error("missing required string attribute 'format'")),
        };

        Ok(Box::new(BlockQuantizedMatMulKernel {
            k,
            n,
            format,
            packed_b_constant: false,
            weight_kn: OnceLock::new(),
        }))
    }
}

impl Kernel for BlockQuantizedMatMulKernel {
    fn set_constant_inputs(&mut self, constant_inputs: &[bool]) {
        self.packed_b_constant = constant_inputs.get(1).copied().unwrap_or(false);
    }

    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity(OP, inputs, outputs, 2, 3, 1)?;
        require_dtype("A", inputs[0].dtype, DataType::Float32)?;
        require_dtype("packed_B", inputs[1].dtype, DataType::Uint8)?;
        require_dtype("Y", outputs[0].dtype, DataType::Float32)?;

        let a_shape = inputs[0].shape;
        if a_shape.is_empty() || a_shape[a_shape.len() - 1] != self.k {
            return Err(error(format!(
                "A must have rank >= 1 and last dimension K={}, got {:?}",
                self.k, a_shape
            )));
        }
        let expected_output_shape = [&a_shape[..a_shape.len() - 1], &[self.n]].concat();
        require_shape("Y", outputs[0].shape, &expected_output_shape)?;

        let blocks = self.k.div_ceil(self.format.qk());
        require_shape(
            "packed_B",
            inputs[1].shape,
            &[self.n, blocks, self.format.block_bytes()],
        )?;

        let bias = if let Some(bias) = inputs.get(2).filter(|input| !input.is_absent()) {
            require_dtype("bias", bias.dtype, DataType::Float32)?;
            require_shape("bias", bias.shape, &[self.n])?;
            Some(to_dense_f32(bias)?)
        } else {
            None
        };

        let activations = to_dense_f32(&inputs[0])?;
        let owned_weight;
        let weight_kn = if self.packed_b_constant {
            if let Some(weight) = self.weight_kn.get() {
                weight
            } else {
                let weight = self.dequantize_weight_kn(&inputs[1])?;
                let _ = self.weight_kn.set(weight);
                self.weight_kn
                    .get()
                    .expect("constant block-quantized weight was just initialized")
            }
        } else {
            owned_weight = self.dequantize_weight_kn(&inputs[1])?;
            &owned_weight
        };

        let m = numel(&a_shape[..a_shape.len() - 1]);
        let result_elements = m
            .checked_mul(self.n)
            .ok_or_else(|| error("Y element count overflow"))?;
        let mut result = vec![0.0f32; result_elements];
        gemm(&activations, weight_kn, &mut result, m, self.k, self.n)?;
        if let Some(bias) = bias {
            for row in result.chunks_exact_mut(self.n) {
                for (value, bias) in row.iter_mut().zip(&bias) {
                    *value += bias;
                }
            }
        }
        write_dense_f32(&mut outputs[0], &result)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

impl BlockQuantizedMatMulKernel {
    fn dequantize_weight_kn(&self, packed: &TensorView) -> Result<Vec<f32>> {
        let packed = to_dense_bytes(packed)?;
        let qk = self.format.qk();
        let block_bytes = self.format.block_bytes();
        let blocks = self.k.div_ceil(qk);
        let expected_bytes = self
            .n
            .checked_mul(blocks)
            .and_then(|value| value.checked_mul(block_bytes))
            .ok_or_else(|| error("packed_B byte count overflow"))?;
        if packed.len() != expected_bytes {
            return Err(error(format!(
                "packed_B must contain exactly {expected_bytes} bytes, got {}",
                packed.len()
            )));
        }

        let weight_elements = self
            .k
            .checked_mul(self.n)
            .ok_or_else(|| error("dequantized weight element count overflow"))?;
        let mut weight_kn = vec![0.0f32; weight_elements];
        let block_row_elements = qk
            .min(self.k)
            .checked_mul(self.n)
            .ok_or_else(|| error("dequantized block-row element count overflow"))?;
        let decoder = self.format.decoder();
        weight_kn
            .par_chunks_mut(block_row_elements)
            .enumerate()
            .for_each(|(block_index, weight_rows)| {
                let mut decoded = [0.0f32; IQ_SUPER_QK];
                let valid = weight_rows.len() / self.n;
                for output in 0..self.n {
                    let packed_start = (output * blocks + block_index) * block_bytes;
                    decoder(
                        &packed[packed_start..packed_start + block_bytes],
                        &mut decoded[..qk],
                    );
                    for (offset, value) in decoded[..valid].iter().copied().enumerate() {
                        weight_rows[offset * self.n + output] = value;
                    }
                }
            });
        Ok(weight_kn)
    }
}

fn decode_mxfp4_block(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), MXFP4_BLOCK_BYTES);
    debug_assert_eq!(output.len(), MXFP4_QK);
    let half_scale = e8m0_half_scale(block[0]);
    for j in 0..16 {
        let packed = block[1 + j];
        output[j] = E2M1_DOUBLED[(packed & 0x0f) as usize] as f32 * half_scale;
        output[j + 16] = E2M1_DOUBLED[(packed >> 4) as usize] as f32 * half_scale;
    }
}

fn e8m0_half_scale(exponent: u8) -> f32 {
    match exponent {
        // OCP E8M0 reserves 0xff for NaN. llama.cpp does not emit it.
        0xff => f32::NAN,
        // Exact subnormal representations of 2^-128 and 2^-127.
        0 => f32::from_bits(0x0020_0000),
        1 => f32::from_bits(0x0040_0000),
        // Half of 2^(e-127) is 2^(e-128), encoded with f32 exponent e-1.
        _ => f32::from_bits((u32::from(exponent) - 1) << 23),
    }
}

fn decode_iq4_nl_block(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), IQ4_NL_BLOCK_BYTES);
    debug_assert_eq!(output.len(), IQ4_NL_QK);
    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    for j in 0..16 {
        let packed = block[2 + j];
        output[j] = scale * IQ4_NL_CODEBOOK[(packed & 0x0f) as usize] as f32;
        output[j + 16] = scale * IQ4_NL_CODEBOOK[(packed >> 4) as usize] as f32;
    }
}

fn decode_iq4_xs_block(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), IQ4_XS_BLOCK_BYTES);
    debug_assert_eq!(output.len(), IQ_SUPER_QK);
    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let scales_h = u16::from_le_bytes([block[2], block[3]]);
    let scales_l = &block[4..8];
    let quants = &block[8..];
    for subblock in 0..8 {
        let low = (scales_l[subblock / 2] >> (4 * (subblock % 2))) & 0x0f;
        let high = ((scales_h >> (2 * subblock)) & 0x03) as u8;
        let subscale = scale * f32::from((low | (high << 4)) as i8 - 32);
        let output = &mut output[subblock * 32..][..32];
        let quants = &quants[subblock * 16..][..16];
        for j in 0..16 {
            output[j] = subscale * IQ4_NL_CODEBOOK[(quants[j] & 0x0f) as usize] as f32;
            output[j + 16] = subscale * IQ4_NL_CODEBOOK[(quants[j] >> 4) as usize] as f32;
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn decode_mxfp4_block_avx2_dispatch(block: &[u8], output: &mut [f32]) {
    // SAFETY: BlockFormat::decoder selects this wrapper only after AVX2 detection.
    unsafe { decode_mxfp4_block_avx2(block, output) }
}

#[cfg(target_arch = "x86_64")]
fn decode_iq4_nl_block_avx2_dispatch(block: &[u8], output: &mut [f32]) {
    // SAFETY: BlockFormat::decoder selects this wrapper only after AVX2 detection.
    unsafe { decode_iq4_nl_block_avx2(block, output) }
}

#[cfg(target_arch = "x86_64")]
fn decode_iq4_xs_block_avx2_dispatch(block: &[u8], output: &mut [f32]) {
    // SAFETY: BlockFormat::decoder selects this wrapper only after AVX2 detection.
    unsafe { decode_iq4_xs_block_avx2(block, output) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn decode_mxfp4_block_avx2(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), MXFP4_BLOCK_BYTES);
    debug_assert_eq!(output.len(), MXFP4_QK);
    let half_scale = e8m0_half_scale(block[0]);
    // SAFETY: the block and output lengths above cover the 16-byte load and 32 outputs.
    unsafe {
        decode_nibbles_scaled_avx2(&block[1..], &E2M1_DOUBLED, half_scale, output);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn decode_iq4_nl_block_avx2(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), IQ4_NL_BLOCK_BYTES);
    debug_assert_eq!(output.len(), IQ4_NL_QK);
    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    // SAFETY: the block and output lengths above cover the 16-byte load and 32 outputs.
    unsafe {
        decode_nibbles_scaled_avx2(&block[2..], &IQ4_NL_CODEBOOK, scale, output);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn decode_iq4_xs_block_avx2(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), IQ4_XS_BLOCK_BYTES);
    debug_assert_eq!(output.len(), IQ_SUPER_QK);
    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let scales_h = u16::from_le_bytes([block[2], block[3]]);
    let scales_l = &block[4..8];
    let quants = &block[8..];
    for subblock in 0..8 {
        let low = (scales_l[subblock / 2] >> (4 * (subblock % 2))) & 0x0f;
        let high = ((scales_h >> (2 * subblock)) & 0x03) as u8;
        let subscale = scale * f32::from((low | (high << 4)) as i8 - 32);
        // SAFETY: each subblock owns 16 packed bytes and 32 decoded outputs.
        unsafe {
            decode_nibbles_scaled_avx2(
                &quants[subblock * 16..][..16],
                &IQ4_NL_CODEBOOK,
                subscale,
                &mut output[subblock * 32..][..32],
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn decode_nibbles_scaled_avx2(
    packed: &[u8],
    codebook: &[i8; 16],
    scale: f32,
    output: &mut [f32],
) {
    use std::arch::x86_64::*;

    debug_assert!(packed.len() >= 16);
    debug_assert!(output.len() >= 32);
    // SAFETY: callers provide at least 16 packed bytes and 32 output elements.
    let (low_values, high_values) = unsafe {
        let bytes = _mm_loadu_si128(packed.as_ptr().cast());
        let mask = _mm_set1_epi8(0x0f);
        let low_indices = _mm_and_si128(bytes, mask);
        let high_indices = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
        let table = _mm_loadu_si128(codebook.as_ptr().cast());
        (
            _mm_shuffle_epi8(table, low_indices),
            _mm_shuffle_epi8(table, high_indices),
        )
    };
    let scale = _mm256_set1_ps(scale);
    // SAFETY: each store writes eight elements inside the validated output slice.
    unsafe {
        let low0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(low_values));
        let low1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(low_values)));
        let high0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(high_values));
        let high1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(high_values)));
        _mm256_storeu_ps(output.as_mut_ptr(), _mm256_mul_ps(low0, scale));
        _mm256_storeu_ps(output.as_mut_ptr().add(8), _mm256_mul_ps(low1, scale));
        _mm256_storeu_ps(output.as_mut_ptr().add(16), _mm256_mul_ps(high0, scale));
        _mm256_storeu_ps(output.as_mut_ptr().add(24), _mm256_mul_ps(high1, scale));
    }
}

fn decode_iq3_s_block(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), IQ3_S_BLOCK_BYTES);
    debug_assert_eq!(output.len(), IQ_SUPER_QK);
    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let quants = &block[2..66];
    let high_bits = &block[66..74];
    let signs = &block[74..106];
    let scales = &block[106..110];

    for group64 in 0..4 {
        let packed_scale = scales[group64];
        for half in 0..2 {
            let subscale = scale * f32::from(1 + 2 * ((packed_scale >> (4 * half)) & 0x0f));
            let qh = high_bits[group64 * 2 + half];
            let quant_base = group64 * 16 + half * 8;
            let sign_base = group64 * 8 + half * 4;
            let output_base = group64 * 64 + half * 32;
            for vector in 0..4 {
                let index0 = usize::from(quants[quant_base + 2 * vector])
                    | (usize::from((qh >> (2 * vector)) & 1) << 8);
                let index1 = usize::from(quants[quant_base + 2 * vector + 1])
                    | (usize::from((qh >> (2 * vector + 1)) & 1) << 8);
                let grid0 = IQ3S_GRID[index0];
                let grid1 = IQ3S_GRID[index1];
                let sign_mask = signs[sign_base + vector];
                let vector_base = output_base + vector * 8;
                for j in 0..4 {
                    let magnitude0 = ((grid0 >> (8 * j)) & 0xff) as f32;
                    let magnitude1 = ((grid1 >> (8 * j)) & 0xff) as f32;
                    output[vector_base + j] = if sign_mask & (1 << j) != 0 {
                        -subscale * magnitude0
                    } else {
                        subscale * magnitude0
                    };
                    output[vector_base + j + 4] = if sign_mask & (1 << (j + 4)) != 0 {
                        -subscale * magnitude1
                    } else {
                        subscale * magnitude1
                    };
                }
            }
        }
    }
}

fn decode_iq3_xxs_block(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), IQ3_XXS_BLOCK_BYTES);
    debug_assert_eq!(output.len(), IQ_SUPER_QK);
    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let quants = &block[2..66];
    let scales_and_signs = &block[66..98];

    for group32 in 0..8 {
        let metadata_base = group32 * 4;
        let metadata = u32::from_le_bytes([
            scales_and_signs[metadata_base],
            scales_and_signs[metadata_base + 1],
            scales_and_signs[metadata_base + 2],
            scales_and_signs[metadata_base + 3],
        ]);
        let subscale = scale * (0.5 + (metadata >> 28) as f32) * 0.5;
        let quant_base = group32 * 8;
        for vector in 0..4 {
            let sign_mask = IQ2XS_SIGNS[((metadata >> (7 * vector)) & 127) as usize];
            let grid0 = IQ3XXS_GRID[quants[quant_base + 2 * vector] as usize];
            let grid1 = IQ3XXS_GRID[quants[quant_base + 2 * vector + 1] as usize];
            let output_base = group32 * 32 + vector * 8;
            for j in 0..4 {
                let magnitude0 = ((grid0 >> (8 * j)) & 0xff) as f32;
                let magnitude1 = ((grid1 >> (8 * j)) & 0xff) as f32;
                output[output_base + j] = if sign_mask & (1 << j) != 0 {
                    -subscale * magnitude0
                } else {
                    subscale * magnitude0
                };
                output[output_base + j + 4] = if sign_mask & (1 << (j + 4)) != 0 {
                    -subscale * magnitude1
                } else {
                    subscale * magnitude1
                };
            }
        }
    }
}

fn decode_iq2_s_block(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), IQ2_S_BLOCK_BYTES);
    debug_assert_eq!(output.len(), IQ_SUPER_QK);
    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let quants = &block[2..34];
    let signs = &block[34..66];
    let high_bits = &block[66..74];
    let scales = &block[74..82];

    for group32 in 0..8 {
        let packed_scale = scales[group32];
        let qh = high_bits[group32];
        for vector in 0..4 {
            let subscale =
                scale * (0.5 + ((packed_scale >> (4 * (vector / 2))) & 0x0f) as f32) * 0.25;
            let index = usize::from(quants[group32 * 4 + vector])
                | (usize::from((qh >> (2 * vector)) & 0x03) << 8);
            let grid = IQ2S_GRID[index];
            let sign_mask = signs[group32 * 4 + vector];
            let output_base = group32 * 32 + vector * 8;
            for j in 0..8 {
                let magnitude = ((grid >> (8 * j)) & 0xff) as f32;
                output[output_base + j] = if sign_mask & (1 << j) != 0 {
                    -subscale * magnitude
                } else {
                    subscale * magnitude
                };
            }
        }
    }
}

fn decode_iq2_xs_block(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), IQ2_XS_BLOCK_BYTES);
    debug_assert_eq!(output.len(), IQ_SUPER_QK);
    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let quants = &block[2..66];
    let scales = &block[66..74];

    for group32 in 0..8 {
        let packed_scale = scales[group32];
        for vector in 0..4 {
            let quant_base = group32 * 8 + vector * 2;
            let quant = u16::from_le_bytes([quants[quant_base], quants[quant_base + 1]]);
            let subscale =
                scale * (0.5 + ((packed_scale >> (4 * (vector / 2))) & 0x0f) as f32) * 0.25;
            let grid = IQ2XS_GRID[usize::from(quant & 511)];
            let sign_mask = IQ2XS_SIGNS[usize::from(quant >> 9)];
            let output_base = group32 * 32 + vector * 8;
            for j in 0..8 {
                let magnitude = ((grid >> (8 * j)) & 0xff) as f32;
                output[output_base + j] = if sign_mask & (1 << j) != 0 {
                    -subscale * magnitude
                } else {
                    subscale * magnitude
                };
            }
        }
    }
}

fn decode_iq2_xxs_block(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), IQ2_XXS_BLOCK_BYTES);
    debug_assert_eq!(output.len(), IQ_SUPER_QK);
    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    for group32 in 0..8 {
        let base = 2 + group32 * 8;
        let metadata = u32::from_le_bytes([
            block[base + 4],
            block[base + 5],
            block[base + 6],
            block[base + 7],
        ]);
        let subscale = scale * (0.5 + (metadata >> 28) as f32) * 0.25;
        for vector in 0..4 {
            let grid = IQ2XXS_GRID[block[base + vector] as usize];
            let sign_mask = IQ2XS_SIGNS[((metadata >> (7 * vector)) & 127) as usize];
            let output_base = group32 * 32 + vector * 8;
            for j in 0..8 {
                let magnitude = ((grid >> (8 * j)) & 0xff) as f32;
                output[output_base + j] = if sign_mask & (1 << j) != 0 {
                    -subscale * magnitude
                } else {
                    subscale * magnitude
                };
            }
        }
    }
}

fn decode_iq1_s_block(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), IQ1_S_BLOCK_BYTES);
    debug_assert_eq!(output.len(), IQ_SUPER_QK);
    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let quants = &block[2..34];
    let high_bits = &block[34..50];

    for group32 in 0..8 {
        let high_base = group32 * 2;
        let qh = u16::from_le_bytes([high_bits[high_base], high_bits[high_base + 1]]);
        let subscale = scale * f32::from(2 * ((qh >> 12) & 7) + 1);
        let delta = if qh & 0x8000 != 0 {
            -IQ1_S_DELTA
        } else {
            IQ1_S_DELTA
        };
        for vector in 0..4 {
            let index = usize::from(quants[group32 * 4 + vector])
                | (usize::from((qh >> (3 * vector)) & 7) << 8);
            let grid = IQ1S_GRID[index];
            let output_base = group32 * 32 + vector * 8;
            for j in 0..8 {
                let value = ((grid >> (8 * j)) & 0xff) as u8 as i8;
                output[output_base + j] = subscale * (f32::from(value) + delta);
            }
        }
    }
}

fn decode_iq1_m_block(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), IQ1_M_BLOCK_BYTES);
    debug_assert_eq!(output.len(), IQ_SUPER_QK);
    let quants = &block[..32];
    let high_bits = &block[32..48];
    let scales = &block[48..56];
    let packed_scales = [
        u16::from_le_bytes([scales[0], scales[1]]),
        u16::from_le_bytes([scales[2], scales[3]]),
        u16::from_le_bytes([scales[4], scales[5]]),
        u16::from_le_bytes([scales[6], scales[7]]),
    ];
    let scale_bits = (packed_scales[0] >> 12)
        | ((packed_scales[1] >> 8) & 0x00f0)
        | ((packed_scales[2] >> 4) & 0x0f00)
        | (packed_scales[3] & 0xf000);
    let scale = half::f16::from_bits(scale_bits).to_f32();

    for group32 in 0..8 {
        let packed_scale = packed_scales[group32 / 2];
        let scale_shift = 6 * (group32 % 2);
        let subscale1 = scale * f32::from(2 * ((packed_scale >> scale_shift) & 7) + 1);
        let subscale2 = scale * f32::from(2 * ((packed_scale >> (scale_shift + 3)) & 7) + 1);
        for vector in 0..4 {
            let qh = high_bits[group32 * 2 + vector / 2];
            let high_shift = 4 * (vector % 2);
            let index = usize::from(quants[group32 * 4 + vector])
                | (usize::from((qh >> high_shift) & 7) << 8);
            let delta = if qh & (0x08 << high_shift) != 0 {
                -IQ1_M_DELTA
            } else {
                IQ1_M_DELTA
            };
            let subscale = if vector < 2 { subscale1 } else { subscale2 };
            let grid = IQ1S_GRID[index];
            let output_base = group32 * 32 + vector * 8;
            for j in 0..8 {
                let value = ((grid >> (8 * j)) & 0xff) as u8 as i8;
                output[output_base + j] = subscale * (f32::from(value) + delta);
            }
        }
    }
}

fn required_positive_attr(node: &Node, name: &str) -> Result<usize> {
    let value = optional_int_attr(node, name)?
        .ok_or_else(|| error(format!("missing required integer attribute '{name}'")))?;
    if value <= 0 {
        return Err(error(format!(
            "attribute '{name}' must be positive, got {value}"
        )));
    }
    Ok(value as usize)
}

fn optional_int_attr(node: &Node, name: &str) -> Result<Option<i64>> {
    match node.attr(name) {
        Some(attribute) => attribute
            .as_int()
            .map(Some)
            .ok_or_else(|| error(format!("attribute '{name}' must be an integer"))),
        None => Ok(None),
    }
}

fn require_dtype(name: &str, got: DataType, expected: DataType) -> Result<()> {
    if got != expected {
        return Err(error(format!(
            "{name} must have dtype {expected:?}, got {got:?}"
        )));
    }
    Ok(())
}

fn require_shape(name: &str, got: &[usize], expected: &[usize]) -> Result<()> {
    if got != expected {
        return Err(error(format!(
            "{name} must have shape {expected:?}, got {got:?}"
        )));
    }
    Ok(())
}

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("{DOMAIN}::{OP}: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuExecutionProvider;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ep_api::ExecutionProvider;
    use onnx_runtime_ir::{Attribute, Graph, NodeId, static_shape};
    use onnx_runtime_loader::Model;

    fn model_node(
        format: &str,
        a_shape: &[usize],
        b_shape: &[usize],
        output_shape: &[usize],
        k: usize,
        n: usize,
        with_bias: bool,
    ) -> (Graph, NodeId) {
        let mut graph = Graph::new();
        graph.opset_imports.insert(DOMAIN.into(), 1);
        let a = graph.create_named_value(
            "A",
            DataType::Float32,
            static_shape(a_shape.iter().copied()),
        );
        graph.add_input(a);
        let packed_b = graph.create_named_value(
            "packed_B",
            DataType::Uint8,
            static_shape(b_shape.iter().copied()),
        );
        graph.add_input(packed_b);
        let mut inputs = vec![Some(a), Some(packed_b)];
        if with_bias {
            let bias = graph.create_named_value("bias", DataType::Float32, static_shape([n]));
            graph.add_input(bias);
            inputs.push(Some(bias));
        }
        let output = graph.create_named_value(
            "Y",
            DataType::Float32,
            static_shape(output_shape.iter().copied()),
        );
        let mut node = Node::new(NodeId(0), OP, inputs, vec![output]);
        node.domain = DOMAIN.into();
        node.attributes.insert("K".into(), Attribute::Int(k as i64));
        node.attributes.insert("N".into(), Attribute::Int(n as i64));
        node.attributes.insert(
            "format".into(),
            Attribute::String(format.as_bytes().to_vec()),
        );
        node.attributes
            .insert("block_layout_version".into(), Attribute::Int(1));
        let node = graph.insert_node(node);
        graph.add_output(output);
        (graph, node)
    }

    fn kernel(graph: &Graph, node: NodeId) -> Box<dyn Kernel> {
        let model = Model::new(graph);
        CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .expect("CPU EP must register BlockQuantizedMatMul")
    }

    #[test]
    fn mxfp4_known_block_matches_ocp_e2m1_and_llama_layout() {
        let mut packed = vec![127u8];
        packed.extend((0u8..16).map(|code| code | (code << 4)));
        let view = Owned::u8(&[1, 1, 17], &packed);
        let kernel = BlockQuantizedMatMulKernel {
            k: 32,
            n: 1,
            format: BlockFormat::Mxfp4,
            packed_b_constant: false,
            weight_kn: OnceLock::new(),
        };
        let actual = kernel.dequantize_weight_kn(&view.view()).unwrap();
        let values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let mut expected = Vec::with_capacity(32);
        expected.extend(values);
        expected.extend(values.map(|value| -value));
        expected.extend(values);
        expected.extend(values.map(|value| -value));
        assert_eq!(actual, expected);
    }

    #[test]
    fn e8m0_decode_covers_subnormal_extremes_and_nan() {
        assert_eq!((e8m0_half_scale(0) * 2.0).to_bits(), 0x0040_0000);
        assert_eq!((e8m0_half_scale(1) * 2.0).to_bits(), 0x0080_0000);
        assert_eq!(e8m0_half_scale(127), 0.5);
        assert_eq!(e8m0_half_scale(128), 1.0);
        assert_eq!((e8m0_half_scale(254) * 2.0).to_bits(), 0x7f00_0000);
        assert!(e8m0_half_scale(255).is_nan());
    }

    #[test]
    fn mxfp4_batched_matmul_with_partial_block_and_bias_matches_reference() {
        let (m, k, n): (usize, usize, usize) = (2, 45, 2);
        let blocks = k.div_ceil(32);
        let mut packed = vec![0u8; n * blocks * MXFP4_BLOCK_BYTES];
        let mut weight_nk = vec![0.0f32; n * k];
        for output in 0..n {
            for block in 0..blocks {
                let start = (output * blocks + block) * MXFP4_BLOCK_BYTES;
                packed[start] = 127 + output as u8;
                for j in 0..16 {
                    let low = ((j + block + output) % 16) as u8;
                    let high = ((15 + output - (j % 2)) % 16) as u8;
                    packed[start + 1 + j] = low | (high << 4);
                }
                let mut decoded = [0.0; 32];
                decode_mxfp4_block(&packed[start..start + MXFP4_BLOCK_BYTES], &mut decoded);
                for offset in 0..(k - block * 32).min(32) {
                    weight_nk[output * k + block * 32 + offset] = decoded[offset];
                }
            }
        }
        let activations: Vec<f32> = (0..m * k)
            .map(|index| ((index * 7 % 19) as f32 - 9.0) / 8.0)
            .collect();
        let bias = [0.25, -0.5];
        let mut expected = vec![0.0; m * n];
        for row in 0..m {
            for output in 0..n {
                expected[row * n + output] = bias[output]
                    + (0..k)
                        .map(|inner| activations[row * k + inner] * weight_nk[output * k + inner])
                        .sum::<f32>();
            }
        }

        let (graph, node) = model_node("mxfp4", &[m, k], &[n, blocks, 17], &[m, n], k, n, true);
        let kernel = kernel(&graph, node);
        let a = Owned::f32(&[m, k], &activations);
        let b = Owned::u8(&[n, blocks, 17], &packed);
        let bias = Owned::f32(&[n], &bias);
        let mut y = Owned::zeros_f32(&[m, n]);
        kernel
            .execute(&[a.view(), b.view(), bias.view()], &mut [y.view_mut()])
            .unwrap();
        for (actual, expected) in y.to_f32().iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn iq4_nl_uses_llama_codebook_and_fp16_scale() {
        let mut packed = half::f16::from_f32(0.5).to_le_bytes().to_vec();
        packed.extend((0u8..16).map(|code| code | ((15 - code) << 4)));
        let view = Owned::u8(&[1, 1, IQ4_NL_BLOCK_BYTES], &packed);
        let decoder = BlockQuantizedMatMulKernel {
            k: 32,
            n: 1,
            format: BlockFormat::Iq4Nl,
            packed_b_constant: false,
            weight_kn: OnceLock::new(),
        };
        let actual = decoder.dequantize_weight_kn(&view.view()).unwrap();
        let expected: Vec<f32> = IQ4_NL_CODEBOOK
            .iter()
            .chain(IQ4_NL_CODEBOOK.iter().rev())
            .map(|value| *value as f32 * 0.5)
            .collect();
        assert_eq!(actual, expected);

        let activation: Vec<f32> = (1..=32).map(|value| value as f32 / 16.0).collect();
        let reference = activation
            .iter()
            .zip(&expected)
            .map(|(a, b)| a * b)
            .sum::<f32>();
        let (graph, node) = model_node("iq4_nl", &[1, 32], &[1, 1, 18], &[1, 1], 32, 1, false);
        let kernel = kernel(&graph, node);
        let a = Owned::f32(&[1, 32], &activation);
        let b = Owned::u8(&[1, 1, 18], &packed);
        let mut y = Owned::zeros_f32(&[1, 1]);
        kernel
            .execute(&[a.view(), b.view()], &mut [y.view_mut()])
            .unwrap();
        assert!((y.to_f32()[0] - reference).abs() <= 1e-5);
    }

    #[test]
    fn iq4_xs_decodes_six_bit_subscales_and_iq4_nl_values() {
        let mut packed = half::f16::from_f32(0.5).to_le_bytes().to_vec();
        packed.extend(0xaaaau16.to_le_bytes());
        packed.extend([0x55; 4]);
        packed.extend([0x98; 128]);
        let view = Owned::u8(&[1, 1, IQ4_XS_BLOCK_BYTES], &packed);
        let decoder = BlockQuantizedMatMulKernel {
            k: IQ_SUPER_QK,
            n: 1,
            format: BlockFormat::Iq4Xs,
            packed_b_constant: false,
            weight_kn: OnceLock::new(),
        };
        let actual = decoder.dequantize_weight_kn(&view.view()).unwrap();
        let mut expected = Vec::with_capacity(IQ_SUPER_QK);
        for _ in 0..8 {
            expected.extend([2.5; 16]);
            expected.extend([32.5; 16]);
        }
        // First sub-block: ls = 0b10_0101 = 37, dl = 0.5*(37-32) = 2.5.
        // Byte 0x98 therefore gives 2.5*codebook[8] = 2.5 at y[0] and
        // 2.5*codebook[9] = 32.5 at y[16].
        assert_eq!(actual, expected);
    }

    #[test]
    fn iq3_s_decodes_grid_high_bits_signs_and_odd_subscales() {
        let mut packed = vec![0u8; IQ3_S_BLOCK_BYTES];
        packed[..2].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
        packed[66] = 0x01;
        packed[74] = 0x81;
        packed[106..110].fill(0x10);
        let view = Owned::u8(&[1, 1, IQ3_S_BLOCK_BYTES], &packed);
        let decoder = BlockQuantizedMatMulKernel {
            k: IQ_SUPER_QK,
            n: 1,
            format: BlockFormat::Iq3S,
            packed_b_constant: false,
            weight_kn: OnceLock::new(),
        };
        let actual = decoder.dequantize_weight_kn(&view.view()).unwrap();
        let mut expected = vec![0.0; IQ_SUPER_QK];
        for group64 in 0..4 {
            expected[group64 * 64..group64 * 64 + 32].fill(0.5);
            expected[group64 * 64 + 32..group64 * 64 + 64].fill(1.5);
        }
        expected[0..4].copy_from_slice(&[-3.5, 2.5, 4.5, 2.5]);
        expected[7] = -0.5;
        // qh bit zero raises the first index to 256, whose grid is {7,5,9,5};
        // the paired zero index is {1,1,1,1}. Scale byte 0x10 gives db1=0.5
        // and db2=1.5. signs[0]=0x81 negates weights zero and seven.
        assert_eq!(actual, expected);
    }

    #[test]
    fn iq3_xxs_decodes_two_grids_packed_signs_and_scale() {
        let mut packed = vec![0u8; IQ3_XXS_BLOCK_BYTES];
        packed[..2].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
        let metadata = 2u32 << 28 | 3u32 << 21 | 2u32 << 14 | 1u32 << 7;
        for group32 in 0..8 {
            packed[2 + group32 * 8..2 + group32 * 8 + 8]
                .copy_from_slice(&[0, 255, 1, 254, 2, 253, 3, 252]);
            packed[66 + group32 * 4..66 + group32 * 4 + 4].copy_from_slice(&metadata.to_le_bytes());
        }
        let view = Owned::u8(&[1, 1, IQ3_XXS_BLOCK_BYTES], &packed);
        let decoder = BlockQuantizedMatMulKernel {
            k: IQ_SUPER_QK,
            n: 1,
            format: BlockFormat::Iq3Xxs,
            packed_b_constant: false,
            weight_kn: OnceLock::new(),
        };
        let actual = decoder.dequantize_weight_kn(&view.view()).unwrap();
        let group = [
            10.0, 10.0, 10.0, 10.0, 10.0, 70.0, 130.0, 155.0, -50.0, 10.0, 10.0, 10.0, 90.0, 50.0,
            110.0, -155.0, 90.0, -10.0, 10.0, 10.0, 50.0, 10.0, 110.0, -155.0, -30.0, -30.0, 10.0,
            10.0, 10.0, 10.0, 110.0, 155.0,
        ];
        let expected: Vec<f32> = group.into_iter().cycle().take(IQ_SUPER_QK).collect();
        // scale4=2 gives db=2*(0.5+2)*0.5=2.5. The first pair uses
        // grids 0={4,4,4,4} and 255={4,28,52,62}; sign index zero is positive.
        // Sign indices 1,2,3 then apply masks 0x81, 0x82, and 0x03.
        assert_eq!(actual, expected);
    }

    #[test]
    fn iq2_s_decodes_ten_bit_grids_explicit_signs_and_nibble_scales() {
        let mut packed = vec![0u8; IQ2_S_BLOCK_BYTES];
        packed[..2].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
        for group32 in 0..8 {
            packed[2 + group32 * 4..2 + group32 * 4 + 4].copy_from_slice(&[0, 0, 0, 255]);
            packed[34 + group32 * 4..34 + group32 * 4 + 4]
                .copy_from_slice(&[0x00, 0x81, 0x82, 0x03]);
            packed[66 + group32] = 0xe4;
            packed[74 + group32] = 0x21;
        }
        let view = Owned::u8(&[1, 1, IQ2_S_BLOCK_BYTES], &packed);
        let decoder = BlockQuantizedMatMulKernel {
            k: IQ_SUPER_QK,
            n: 1,
            format: BlockFormat::Iq2S,
            packed_b_constant: false,
            weight_kn: OnceLock::new(),
        };
        let actual = decoder.dequantize_weight_kn(&view.view()).unwrap();
        let group = [
            6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, -18.75, 18.75, 18.75, 18.75, 18.75, 6.0, 18.75,
            -6.0, 31.25, -31.25, 53.75, 10.0, 31.25, 10.0, 10.0, -31.25, -53.75, -53.75, 53.75,
            53.75, 53.75, 53.75, 53.75, 53.75,
        ];
        let expected: Vec<f32> = group.into_iter().cycle().take(IQ_SUPER_QK).collect();
        // qh=0xe4 combines low indices {0,0,0,255} into {0,256,512,1023}.
        // Scale byte 0x21 gives db={0.75,1.25}; signs are explicit per vector.
        assert_eq!(actual, expected);
    }

    #[test]
    fn iq2_xs_decodes_nine_bit_grids_sign_table_and_nibble_scales() {
        let mut packed = vec![0u8; IQ2_XS_BLOCK_BYTES];
        packed[..2].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
        let grids = [0u16, 511, 1, 510];
        for group32 in 0..8 {
            for (vector, grid) in grids.into_iter().enumerate() {
                let quant = grid | ((vector as u16) << 9);
                let base = 2 + group32 * 8 + vector * 2;
                packed[base..base + 2].copy_from_slice(&quant.to_le_bytes());
            }
            packed[66 + group32] = 0x21;
        }
        let view = Owned::u8(&[1, 1, IQ2_XS_BLOCK_BYTES], &packed);
        let decoder = BlockQuantizedMatMulKernel {
            k: IQ_SUPER_QK,
            n: 1,
            format: BlockFormat::Iq2Xs,
            packed_b_constant: false,
            weight_kn: OnceLock::new(),
        };
        let actual = decoder.dequantize_weight_kn(&view.view()).unwrap();
        let group = [
            6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, -32.25, 32.25, 32.25, 32.25, 32.25, 32.25,
            32.25, -32.25, 53.75, -10.0, 10.0, 10.0, 10.0, 10.0, 10.0, -10.0, -31.25, -10.0, 31.25,
            53.75, 53.75, 53.75, 53.75, 53.75,
        ];
        let expected: Vec<f32> = group.into_iter().cycle().take(IQ_SUPER_QK).collect();
        // Scale byte 0x21 gives db={0.75,1.25}. Grid 511 is all 43s,
        // while grid 510 is {25,8,25,43,43,43,43,43}; sign indices 0..3
        // select masks 0x00, 0x81, 0x82, and 0x03.
        assert_eq!(actual, expected);
    }

    #[test]
    fn iq2_xxs_decodes_packed_grid_sign_and_scale_metadata() {
        let mut packed = vec![0u8; IQ2_XXS_BLOCK_BYTES];
        packed[..2].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
        let metadata = 2u32 << 28 | 3u32 << 21 | 2u32 << 14 | 1u32 << 7;
        for group32 in 0..8 {
            let base = 2 + group32 * 8;
            packed[base..base + 4].copy_from_slice(&[0, 1, 254, 255]);
            packed[base + 4..base + 8].copy_from_slice(&metadata.to_le_bytes());
        }
        let view = Owned::u8(&[1, 1, IQ2_XXS_BLOCK_BYTES], &packed);
        let decoder = BlockQuantizedMatMulKernel {
            k: IQ_SUPER_QK,
            n: 1,
            format: BlockFormat::Iq2Xxs,
            packed_b_constant: false,
            weight_kn: OnceLock::new(),
        };
        let actual = decoder.dequantize_weight_kn(&view.view()).unwrap();
        let sign_masks = [0u8, 129, 130, 3];
        let grids = [
            [8, 8, 8, 8, 8, 8, 8, 8],
            [43, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 25, 25, 8, 43, 43, 43],
            [8, 25, 8, 8, 25, 43, 43, 43],
        ];
        let mut expected = Vec::with_capacity(IQ_SUPER_QK);
        for _ in 0..8 {
            for (sign_mask, grid) in sign_masks.into_iter().zip(grids) {
                for j in 0..8 {
                    let value = grid[j] as f32 * 1.25;
                    expected.push(if sign_mask & (1 << j) != 0 {
                        -value
                    } else {
                        value
                    });
                }
            }
        }
        // scale4=2 gives db=2*(0.5+2)*0.25=1.25. Grid indices 0,1,254,255
        // begin {8,...}, {43,8,...}, {8,8,25,...}, and {8,25,8,...}.
        // Sign indices 0,1,2,3 map to 0x00,0x81,0x82,0x03.
        assert_eq!(actual, expected);
    }

    #[test]
    fn iq1_s_decodes_eleven_bit_grids_odd_scale_and_delta() {
        let mut packed = vec![0u8; IQ1_S_BLOCK_BYTES];
        packed[..2].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
        packed[4] = 0xff;
        packed[34..36].copy_from_slice(&0xa1c0u16.to_le_bytes());
        let view = Owned::u8(&[1, 1, IQ1_S_BLOCK_BYTES], &packed);
        let decoder = BlockQuantizedMatMulKernel {
            k: IQ_SUPER_QK,
            n: 1,
            format: BlockFormat::Iq1S,
            packed_b_constant: false,
            weight_kn: OnceLock::new(),
        };
        let actual = decoder.dequantize_weight_kn(&view.view()).unwrap();
        let mut expected = vec![-1.75; IQ_SUPER_QK];
        expected[..32].fill(-11.25);
        expected[16..24].fill(8.75);
        // qh=0xa1c0 gives odd scale 5, negative delta, and index 2047
        // for vector two. With d=2, grid 0=-1 and grid 2047=+1:
        // 10*(-1-0.125)=-11.25 and 10*(1-0.125)=8.75.
        assert_eq!(actual, expected);
    }

    #[test]
    fn iq1_m_decodes_bitsliced_fp16_two_odd_scales_and_vector_deltas() {
        let mut packed = vec![0u8; IQ1_M_BLOCK_BYTES];
        packed[1] = 0xff;
        packed[2] = 0xff;
        packed[32] = 0xf0;
        packed[33] = 0x8f;
        packed[48..56].copy_from_slice(&[0x1a, 0, 0, 0, 0, 0, 0, 0x40]);
        let view = Owned::u8(&[1, 1, IQ1_M_BLOCK_BYTES], &packed);
        let decoder = BlockQuantizedMatMulKernel {
            k: IQ_SUPER_QK,
            n: 1,
            format: BlockFormat::Iq1M,
            packed_b_constant: false,
            weight_kn: OnceLock::new(),
        };
        let actual = decoder.dequantize_weight_kn(&view.view()).unwrap();
        let mut expected = vec![-1.75; IQ_SUPER_QK];
        expected[..8].fill(-8.75);
        expected[8..16].fill(8.75);
        expected[16..24].fill(12.25);
        expected[24..32].fill(-15.75);
        // Scale high nibbles reconstruct fp16 0x4000 (2.0). sc[0]=0x001a
        // gives odd multipliers 5 and 7. qh selects grids 0,2047,2047,0
        // with deltas +,-,-,-, producing -8.75,8.75,12.25,-15.75.
        assert_eq!(actual, expected);
    }

    #[test]
    fn vendored_iq_grid_fingerprints_match_llama_cpp() {
        fn fnv1a(bytes: impl IntoIterator<Item = u8>) -> u64 {
            bytes.into_iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
            })
        }

        assert_eq!(fnv1a(IQ2XS_SIGNS), 0xf19b_a8f8_c329_2ba5);
        assert_eq!(
            fnv1a(IQ2XXS_GRID.into_iter().flat_map(u64::to_le_bytes)),
            0xbb4e_e025_b5ac_6e8e
        );
        assert_eq!(
            fnv1a(IQ3S_GRID.into_iter().flat_map(u32::to_le_bytes)),
            0xfa37_020c_25b4_4829
        );
        assert_eq!(
            fnv1a(IQ2XS_GRID.into_iter().flat_map(u64::to_le_bytes)),
            0xc9b1_ee61_e799_09bd
        );
        assert_eq!(
            fnv1a(IQ2S_GRID.into_iter().flat_map(u64::to_le_bytes)),
            0x123e_dd38_a3b6_2b90
        );
        assert_eq!(
            fnv1a(IQ3XXS_GRID.into_iter().flat_map(u32::to_le_bytes)),
            0xdfa5_dc83_d6a1_55d5
        );
        assert_eq!(
            fnv1a(IQ1S_GRID.into_iter().flat_map(u64::to_le_bytes)),
            0x6703_ed86_3501_ae2e
        );
    }

    #[test]
    fn selected_decoders_are_bit_exact_with_scalar_reference() {
        for format in [
            BlockFormat::Mxfp4,
            BlockFormat::Iq4Nl,
            BlockFormat::Iq4Xs,
            BlockFormat::Iq3S,
            BlockFormat::Iq3Xxs,
            BlockFormat::Iq2S,
            BlockFormat::Iq2Xs,
            BlockFormat::Iq2Xxs,
            BlockFormat::Iq1S,
            BlockFormat::Iq1M,
        ] {
            let mut block = vec![0u8; format.block_bytes()];
            for (index, byte) in block.iter_mut().enumerate() {
                *byte = index.wrapping_mul(73).wrapping_add(19) as u8;
            }
            match format {
                BlockFormat::Mxfp4 => block[0] = 128,
                BlockFormat::Iq1M => block[48..56].fill(0),
                _ => block[..2].copy_from_slice(&half::f16::from_f32(0.125).to_le_bytes()),
            }

            let mut scalar = [0.0f32; IQ_SUPER_QK];
            let mut selected = [0.0f32; IQ_SUPER_QK];
            format.scalar_decoder()(&block, &mut scalar[..format.qk()]);
            format.decoder()(&block, &mut selected[..format.qk()]);
            assert_eq!(
                scalar[..format.qk()]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                selected[..format.qk()]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "{format:?} selected decoder changed f32 bits"
            );
        }
    }

    #[test]
    fn new_iq_formats_register_with_upstream_block_sizes() {
        for (format, block_bytes) in [
            ("iq2_xs", IQ2_XS_BLOCK_BYTES),
            ("iq2_s", IQ2_S_BLOCK_BYTES),
            ("iq3_xxs", IQ3_XXS_BLOCK_BYTES),
            ("iq1_s", IQ1_S_BLOCK_BYTES),
            ("iq1_m", IQ1_M_BLOCK_BYTES),
        ] {
            let (graph, node) = model_node(
                format,
                &[1, IQ_SUPER_QK],
                &[1, 1, block_bytes],
                &[1, 1],
                IQ_SUPER_QK,
                1,
                false,
            );
            let model = Model::new(&graph);
            CpuExecutionProvider::new()
                .get_kernel(model.graph.node(node), &[], 1)
                .expect("implemented IQ format must create a CPU kernel");
        }
    }

    #[test]
    #[ignore = "representative CPU throughput benchmark; run with --release -- --ignored"]
    fn benchmark_prefill_4096x4096_m64() {
        use std::hint::black_box;
        use std::time::Instant;

        const M: usize = 64;
        const K: usize = 4096;
        const N: usize = 4096;

        for format in [BlockFormat::Mxfp4, BlockFormat::Iq4Nl] {
            let block_bytes = format.block_bytes();
            let blocks = K.div_ceil(format.qk());
            let mut packed = vec![0u8; N * blocks * block_bytes];
            for (index, block) in packed.chunks_exact_mut(block_bytes).enumerate() {
                match format {
                    BlockFormat::Mxfp4 => block[0] = 128,
                    BlockFormat::Iq4Nl => {
                        block[..2].copy_from_slice(&half::f16::from_f32(0.01).to_le_bytes());
                    }
                    _ => unreachable!(),
                }
                for (offset, byte) in block[block_bytes - format.qk() / 2..]
                    .iter_mut()
                    .enumerate()
                {
                    *byte = index.wrapping_mul(17).wrapping_add(offset * 29) as u8;
                }
            }
            let packed = Owned::u8(&[N, blocks, block_bytes], &packed);
            let kernel = BlockQuantizedMatMulKernel {
                k: K,
                n: N,
                format,
                packed_b_constant: false,
                weight_kn: OnceLock::new(),
            };

            let decode_start = Instant::now();
            let weight = black_box(kernel.dequantize_weight_kn(&packed.view()).unwrap());
            let decode = decode_start.elapsed();
            let activations: Vec<f32> = (0..M * K)
                .map(|index| (index % 31) as f32 * (1.0 / 31.0))
                .collect();
            let mut result = vec![0.0f32; M * N];
            let gemm_start = Instant::now();
            gemm(
                black_box(&activations),
                black_box(&weight),
                black_box(&mut result),
                M,
                K,
                N,
            )
            .unwrap();
            let gemm_time = gemm_start.elapsed();
            let gflops = 2.0 * M as f64 * K as f64 * N as f64 / gemm_time.as_secs_f64() / 1.0e9;
            eprintln!(
                "{format:?}: decode={:.1} ms prefill={:.1} ms ({gflops:.1} GFLOP/s)",
                decode.as_secs_f64() * 1.0e3,
                gemm_time.as_secs_f64() * 1.0e3,
            );
            black_box(result);
        }
    }
}

//! Correctness-first `com.microsoft::MatMulNBits` for f32 activations and
//! block-quantized 2-bit or 4-bit weights.
//!
//! ORT stores `B` as
//! `[N, ceil(K / block_size), block_size * bits / 8]`, least-significant bits
//! first within each byte. For M=1 decode, constant quantized weights are
//! prepacked once and reused by a N-parallel GEMV. For symmetric block-32
//! int4 M=1, `accuracy_level=4` streams the packed weights directly into a VNNI
//! dot product. Other int4 accuracy-level-4 shapes keep the weights in int8 and
//! quantize each activation row to int8. The 2-bit correctness path and default
//! int4 path dequantize to f32; batched shapes then use the shared CPU GEMM,
//! including its SIMD backend.

use std::cell::Cell;
use std::sync::OnceLock;

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};
use rayon::prelude::*;

use super::matmul::gemm;
use super::{check_arity, to_dense_bytes, to_dense_f32, to_dense_i64, write_dense_f32};
use crate::strided::numel;

/// Overrides the bounded M=1 decode pool size; set to `0` to use the global
/// Rayon pool as an escape hatch.
const DECODE_THREADS_ENV: &str = "ONNX_GENAI_CPU_DECODE_THREADS";
/// Default M=1 decode pool size. Profiling found 4--8 workers optimal for the
/// tiny projections in decode, while 16 or more workers regressed throughput.
const DEFAULT_DECODE_THREADS: usize = 8;
static DECODE_POOL: OnceLock<std::result::Result<Option<rayon::ThreadPool>, String>> =
    OnceLock::new();

/// Env knob for the MLAS SQNBit int4 prefill/decode crossover (`m` row count).
/// MatMulNBits with `m < NXRT_SQNBIT_PREFILL_MIN` falls back to the specialized
/// hand-written int4 GEMV path (`int4_matmul_m1`/`int8_matmul`), which wins the
/// bandwidth-bound M=1 decode; MLAS `MlasQNBitGemmBatch` is only used once `m`
/// reaches the threshold, where its cache-tiled kernels win prefill.
#[cfg(feature = "mlas")]
const SQNBIT_PREFILL_MIN_ENV: &str = "NXRT_SQNBIT_PREFILL_MIN";

/// Default MLAS SQNBit int4 prefill threshold. Measured on Sapphire Rapids
/// (Xeon 8480C): the hand int4 path leads for small `m` and MLAS overtakes it
/// around `m` in the mid-teens, so decode (`m == 1`) keeps the hand path while
/// prefill batches route to MLAS. Override with `NXRT_SQNBIT_PREFILL_MIN`.
#[cfg(feature = "mlas")]
const DEFAULT_SQNBIT_PREFILL_MIN: usize = 16;

#[cfg(feature = "mlas")]
static SQNBIT_PREFILL_MIN: OnceLock<usize> = OnceLock::new();

/// Smallest `m` (batch·seq row count) that routes MatMulNBits int4 to MLAS
/// SQNBit; smaller `m` falls back to the hand int4 path. Parsed once from
/// `NXRT_SQNBIT_PREFILL_MIN`, defaulting to [`DEFAULT_SQNBIT_PREFILL_MIN`].
#[cfg(feature = "mlas")]
fn sqnbit_prefill_min() -> usize {
    *SQNBIT_PREFILL_MIN
        .get_or_init(|| resolve_prefill_min(std::env::var(SQNBIT_PREFILL_MIN_ENV).ok().as_deref()))
}

/// Parse the SQNBit prefill threshold, falling back to
/// [`DEFAULT_SQNBIT_PREFILL_MIN`] for absent, empty, or malformed values.
#[cfg(feature = "mlas")]
fn resolve_prefill_min(raw: Option<&str>) -> usize {
    raw.and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_SQNBIT_PREFILL_MIN)
}

pub struct MatMulNBitsKernel {
    k: usize,
    n: usize,
    bits: usize,
    block_size: usize,
    accuracy_level: i64,
    constant_inputs: [bool; 5],
    weight_nk: OnceLock<Vec<f32>>,
    int8_weight: OnceLock<Int8Weight>,
    packed_int4_weight: OnceLock<PackedInt4Weight>,
    #[cfg(feature = "mlas")]
    mlas_packed: OnceLock<Option<mlas_sys::SQNBitPackedB>>,
}

struct Int8Weight {
    values: Vec<i8>,
    scales: Vec<f32>,
    block_sums: Vec<i32>,
}

struct PackedInt4Weight {
    values: Vec<u8>,
    scales: Vec<f32>,
}

pub struct MatMulNBitsFactory;

impl KernelFactory for MatMulNBitsFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let k = required_positive_attr(node, "K")?;
        let n = required_positive_attr(node, "N")?;
        let bits = optional_int_attr(node, "bits")?.unwrap_or(4);
        if !matches!(bits, 2 | 4) {
            return Err(error(format!(
                "only bits=2 and bits=4 are supported in the CPU kernel, got {bits}"
            )));
        }
        let weight_prepacked = optional_int_attr(node, "weight_prepacked")?.unwrap_or(0);
        if weight_prepacked != 0 {
            return Err(error(format!(
                "weight_prepacked={weight_prepacked} is unsupported: CPU only supports the standard (non-prepacked) layout"
            )));
        }
        let block_size = required_positive_attr(node, "block_size")?;
        if block_size < 16 || !block_size.is_power_of_two() {
            return Err(error(format!(
                "block_size must be a power of two and at least 16, got {block_size}"
            )));
        }

        let accuracy_level = node
            .attr("accuracy_level")
            .and_then(|value| value.as_int())
            .unwrap_or(0);

        Ok(Box::new(MatMulNBitsKernel {
            k,
            n,
            bits: bits as usize,
            block_size,
            accuracy_level,
            constant_inputs: [false; 5],
            weight_nk: OnceLock::new(),
            int8_weight: OnceLock::new(),
            packed_int4_weight: OnceLock::new(),
            #[cfg(feature = "mlas")]
            mlas_packed: OnceLock::new(),
        }))
    }
}

impl Kernel for MatMulNBitsKernel {
    fn set_constant_inputs(&mut self, constant_inputs: &[bool]) {
        for (index, is_constant) in self.constant_inputs.iter_mut().enumerate() {
            *is_constant = constant_inputs.get(index).copied().unwrap_or(false);
        }
    }

    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("MatMulNBits", inputs, outputs, 3, 6, 1)?;
        require_dtype("A", inputs[0].dtype, DataType::Float32)?;
        require_dtype("B", inputs[1].dtype, DataType::Uint8)?;
        require_dtype("scales", inputs[2].dtype, DataType::Float32)?;
        require_dtype("Y", outputs[0].dtype, DataType::Float32)?;

        let a_shape = inputs[0].shape;
        if a_shape.is_empty() || a_shape[a_shape.len() - 1] != self.k {
            return Err(error(format!(
                "A must have rank >= 1 and last dimension K={}, got {:?}",
                self.k, a_shape
            )));
        }
        let expected_output_shape = [&a_shape[..a_shape.len() - 1], &[self.n]].concat();
        if outputs[0].shape != expected_output_shape {
            return Err(error(format!(
                "Y must have shape {expected_output_shape:?}, got {:?}",
                outputs[0].shape
            )));
        }

        let k_blocks = self.k.div_ceil(self.block_size);
        let blob_size = self.block_size * self.bits / 8;
        require_shape("B", inputs[1].shape, &[self.n, k_blocks, blob_size])?;
        require_flat_or_matrix_shape("scales", inputs[2].shape, self.n, k_blocks)?;

        let zero_points = optional_input(inputs, 3);
        if let Some(zp) = zero_points {
            require_dtype("zero_points", zp.dtype, DataType::Uint8)?;
            let zp_blob_size = (k_blocks * self.bits).div_ceil(8);
            require_flat_or_matrix_shape("zero_points", zp.shape, self.n, zp_blob_size)?;
        }

        let group_indices = optional_input(inputs, 4);
        if let Some(g_idx) = group_indices {
            require_dtype("g_idx", g_idx.dtype, DataType::Int32)?;
            let padded_k = k_blocks * self.block_size;
            if g_idx.shape != [self.k] && g_idx.shape != [padded_k] {
                return Err(error(format!(
                    "g_idx must have shape [{}] or [{padded_k}], got {:?}",
                    self.k, g_idx.shape
                )));
            }
        }

        let bias = if let Some(bias) = optional_input(inputs, 5) {
            require_dtype("bias", bias.dtype, DataType::Float32)?;
            require_shape("bias", bias.shape, &[self.n])?;
            Some(to_dense_f32(bias)?)
        } else {
            None
        };

        let can_prepack = self.constant_inputs[1]
            && self.constant_inputs[2]
            && zero_points.is_none_or(|_| self.constant_inputs[3])
            && group_indices.is_none_or(|_| self.constant_inputs[4]);
        let activations = to_dense_f32(&inputs[0])?;
        let m = numel(&a_shape[..a_shape.len() - 1]);
        crate::trace::record_kernel_metrics(inputs, outputs, || {
            let mut flops = (m as u64)
                .saturating_mul(self.n as u64)
                .saturating_mul(self.k as u64)
                .saturating_mul(2);
            if bias.is_some() {
                flops = flops.saturating_add((m as u64).saturating_mul(self.n as u64));
            }
            flops
        });
        let mut result = vec![0.0f32; m * self.n];
        let dot_kernel = selected_dot_kernel();
        #[cfg(feature = "mlas")]
        {
            if let Some(()) = self.try_mlas_sqnbit(
                &inputs[1],
                &inputs[2],
                zero_points,
                group_indices,
                can_prepack,
                &activations,
                m,
                bias.as_deref(),
                &mut result,
            )? {
                return write_dense_f32(&mut outputs[0], &result);
            }
        }
        if self.bits == 4
            && self.accuracy_level == 4
            && m == 1
            && self.block_size == 32
            && zero_points.is_none()
            && group_indices.is_none()
            && dot_kernel != DotKernel::Scalar
        {
            let owned_weight;
            let packed_weight = if can_prepack {
                if let Some(weight) = self.packed_int4_weight.get() {
                    weight
                } else {
                    let weight = PackedInt4Weight {
                        values: to_dense_bytes(&inputs[1])?,
                        scales: to_dense_f32(&inputs[2])?,
                    };
                    let _ = self.packed_int4_weight.set(weight);
                    self.packed_int4_weight
                        .get()
                        .expect("constant MatMulNBits packed int4 weight was just initialized")
                }
            } else {
                owned_weight = PackedInt4Weight {
                    values: to_dense_bytes(&inputs[1])?,
                    scales: to_dense_f32(&inputs[2])?,
                };
                &owned_weight
            };
            with_decode_pool(|| {
                int4_matmul_m1(
                    &activations,
                    packed_weight,
                    &mut result,
                    self.k,
                    self.n,
                    dot_kernel,
                );
            })?;
        } else if self.bits == 4 && self.accuracy_level == 4 && group_indices.is_none() {
            let owned_weight;
            let int8_weight = if can_prepack {
                if let Some(weight) = self.int8_weight.get() {
                    weight
                } else {
                    let weight = self.prepack_int8_weight(&inputs[1], &inputs[2], zero_points)?;
                    let _ = self.int8_weight.set(weight);
                    self.int8_weight
                        .get()
                        .expect("constant MatMulNBits int8 prepack was just initialized")
                }
            } else {
                owned_weight = self.prepack_int8_weight(&inputs[1], &inputs[2], zero_points)?;
                &owned_weight
            };
            let mut matmul = || {
                int8_matmul(
                    &activations,
                    int8_weight,
                    &mut result,
                    m,
                    self.k,
                    self.n,
                    self.block_size,
                    dot_kernel,
                );
            };
            if m == 1 {
                with_decode_pool(matmul)?;
            } else {
                matmul();
            }
        } else if m == 1 {
            let owned_weight;
            let weight_nk = if can_prepack {
                if let Some(weight) = self.weight_nk.get() {
                    weight
                } else {
                    let weight = self.dequantize_weight(
                        &inputs[1],
                        &inputs[2],
                        zero_points,
                        group_indices,
                        WeightLayout::Nk,
                    )?;
                    let _ = self.weight_nk.set(weight);
                    self.weight_nk
                        .get()
                        .expect("constant MatMulNBits prepack was just initialized")
                }
            } else {
                owned_weight = self.dequantize_weight(
                    &inputs[1],
                    &inputs[2],
                    zero_points,
                    group_indices,
                    WeightLayout::Nk,
                )?;
                &owned_weight
            };
            with_decode_pool(|| {
                gemv_nk(&activations, weight_nk, &mut result, self.k, self.n);
            })?;
        } else {
            let weight_kn = self.dequantize_weight(
                &inputs[1],
                &inputs[2],
                zero_points,
                group_indices,
                WeightLayout::Kn,
            )?;
            gemm(&activations, &weight_kn, &mut result, m, self.k, self.n)?;
        }
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

impl MatMulNBitsKernel {
    /// Route the blockwise-quantized MatMul through MLAS's `MlasQNBitGemmBatch`
    /// when the `mlas` feature is on, the backend resolves to
    /// [`CpuBackend::Mlas`], and the case is one MLAS supports. Returns
    /// `Ok(Some(()))` when it filled `result` (the caller writes output and
    /// returns), or `Ok(None)` to signal a fall back to the hand-written paths.
    ///
    /// Fallback cases (return `Ok(None)`): `m` is below the SQNBit prefill
    /// threshold ([`sqnbit_prefill_min`]) so decode keeps the fast hand int4
    /// path, backend is not MLAS, `bits != 4` (2-bit is left to the existing
    /// correctness path), `g_idx` is present (MLAS SQNBit has no per-row group
    /// indices), or MLAS reports no kernel is available for this shape on the
    /// host. Bias, when present, is added by MLAS itself, so the caller's
    /// post-loop bias add is skipped on this path.
    #[cfg(feature = "mlas")]
    #[allow(clippy::too_many_arguments)]
    fn try_mlas_sqnbit(
        &self,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
        group_indices: Option<&TensorView>,
        can_prepack: bool,
        activations: &[f32],
        m: usize,
        bias: Option<&[f32]>,
        result: &mut [f32],
    ) -> Result<Option<()>> {
        use crate::backend::CpuBackend;

        // Cheapest gate first: small `m` (decode/GEMV) is bandwidth-bound and the
        // hand int4 path beats MLAS there, so fall back before any weight packing.
        if m < sqnbit_prefill_min() {
            return Ok(None);
        }

        if self.bits != 4
            || group_indices.is_some()
            || CpuBackend::auto_detect() != CpuBackend::Mlas
        {
            return Ok(None);
        }

        let comp = if self.accuracy_level == 4 {
            mlas_sys::SQNBitComputeType::Int8
        } else {
            mlas_sys::SQNBitComputeType::Fp32
        };

        let owned;
        let packed_ref: Option<&mlas_sys::SQNBitPackedB> = if can_prepack {
            if let Some(cached) = self.mlas_packed.get() {
                cached.as_ref()
            } else {
                let built = self.build_mlas_packed(packed, scales, zero_points, comp)?;
                let _ = self.mlas_packed.set(built);
                self.mlas_packed
                    .get()
                    .expect("constant MatMulNBits MLAS weight was just initialized")
                    .as_ref()
            }
        } else {
            owned = self.build_mlas_packed(packed, scales, zero_points, comp)?;
            owned.as_ref()
        };

        let Some(packed_weight) = packed_ref else {
            return Ok(None);
        };

        mlas_sys::sqnbit_gemm(packed_weight, m, activations, bias, result, true);
        Ok(Some(()))
    }

    /// Pack the constant int4 weight into MLAS's SQNBit layout, or `None` when
    /// MLAS has no kernel for this `(bits, block_size, compute_type)` on the
    /// host. The ONNX `B`/scales/zero-point bytes map directly onto MLAS's pack
    /// inputs; an absent zero point defaults to the shared int4 midpoint (8).
    #[cfg(feature = "mlas")]
    fn build_mlas_packed(
        &self,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
        comp: mlas_sys::SQNBitComputeType,
    ) -> Result<Option<mlas_sys::SQNBitPackedB>> {
        let packed = to_dense_bytes(packed)?;
        let scales = to_dense_f32(scales)?;
        let zero_points = zero_points.map(to_dense_bytes).transpose()?;
        Ok(mlas_sys::SQNBitPackedB::new(
            self.n,
            self.k,
            self.bits,
            self.block_size,
            comp,
            &packed,
            &scales,
            zero_points.as_deref(),
        ))
    }

    fn prepack_int8_weight(
        &self,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
    ) -> Result<Int8Weight> {
        let packed = to_dense_bytes(packed)?;
        let scales = to_dense_f32(scales)?;
        let packed_zero_points = zero_points.map(to_dense_bytes).transpose()?;
        let k_blocks = self.k.div_ceil(self.block_size);
        debug_assert_eq!(self.bits, 4);
        let blob_size = self.block_size / 2;
        let zp_row_bytes = k_blocks.div_ceil(2);
        let padded_k = k_blocks * self.block_size;
        let mut values = vec![0i8; self.n * padded_k];
        let mut block_sums = vec![0i32; self.n * k_blocks];

        for output in 0..self.n {
            for block in 0..k_blocks {
                let zero_point = packed_zero_points.as_ref().map_or(8, |points| {
                    let byte = points[output * zp_row_bytes + block / 2];
                    if block.is_multiple_of(2) {
                        byte & 0x0f
                    } else {
                        byte >> 4
                    }
                });
                let block_start = block * self.block_size;
                let valid = self.k.saturating_sub(block_start).min(self.block_size);
                let packed_start = (output * k_blocks + block) * blob_size;
                let values_start = output * padded_k + block_start;
                let mut sum = 0i32;
                for offset in 0..valid {
                    let byte = packed[packed_start + offset / 2];
                    let quantized = if offset.is_multiple_of(2) {
                        byte & 0x0f
                    } else {
                        byte >> 4
                    };
                    let value = quantized as i8 - zero_point as i8;
                    values[values_start + offset] = value;
                    sum += value as i32;
                }
                block_sums[output * k_blocks + block] = sum;
            }
        }

        Ok(Int8Weight {
            values,
            scales,
            block_sums,
        })
    }

    fn dequantize_weight(
        &self,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
        group_indices: Option<&TensorView>,
        layout: WeightLayout,
    ) -> Result<Vec<f32>> {
        let packed = to_dense_bytes(packed)?;
        let scales = to_dense_f32(scales)?;
        let packed_zero_points = zero_points.map(to_dense_bytes).transpose()?;
        let group_indices = group_indices.map(to_dense_i64).transpose()?;
        let k_blocks = self.k.div_ceil(self.block_size);
        if let Some(indices) = &group_indices {
            for (index, &group) in indices.iter().enumerate() {
                if group < 0 || group as usize >= k_blocks {
                    return Err(error(format!(
                        "g_idx[{index}]={group} is outside 0..{k_blocks}"
                    )));
                }
            }
        }

        let blob_size = self.block_size * self.bits / 8;
        let zp_row_bytes = (k_blocks * self.bits).div_ceil(8);
        let quantized_mask = (1u8 << self.bits) - 1;
        let default_zero_point = 1u8 << (self.bits - 1);
        let mut weight_kn = vec![0.0f32; self.k * self.n];
        for output in 0..self.n {
            if group_indices.is_none() {
                let packed_start = output * k_blocks * blob_size;
                let scale_start = output * k_blocks;
                let zero_point_start = output * zp_row_bytes;
                let packed_row = &packed[packed_start..packed_start + k_blocks * blob_size];
                let scale_row = &scales[scale_start..scale_start + k_blocks];
                let zero_point_row = packed_zero_points
                    .as_ref()
                    .map(|points| &points[zero_point_start..zero_point_start + zp_row_bytes]);
                for depth in 0..self.k {
                    let index = match layout {
                        WeightLayout::Kn => depth * self.n + output,
                        WeightLayout::Nk => output * self.k + depth,
                    };
                    weight_kn[index] = dequantize_nbits_value(
                        packed_row,
                        scale_row,
                        zero_point_row,
                        depth,
                        self.bits,
                        self.block_size,
                    );
                }
                continue;
            }
            for depth in 0..self.k {
                let block = depth / self.block_size;
                let within_block = depth % self.block_size;
                let bit_offset = within_block * self.bits;
                let byte = packed[(output * k_blocks + block) * blob_size + bit_offset / 8];
                let quantized = (byte >> (bit_offset % 8)) & quantized_mask;
                let group = group_indices
                    .as_ref()
                    .map_or(block, |indices| indices[depth] as usize);
                let zero_point = packed_zero_points
                    .as_ref()
                    .map_or(default_zero_point, |points| {
                        let bit_offset = group * self.bits;
                        let byte = points[output * zp_row_bytes + bit_offset / 8];
                        (byte >> (bit_offset % 8)) & quantized_mask
                    });
                let index = match layout {
                    WeightLayout::Kn => depth * self.n + output,
                    WeightLayout::Nk => output * self.k + depth,
                };
                weight_kn[index] =
                    (quantized as f32 - zero_point as f32) * scales[output * k_blocks + group];
            }
        }
        Ok(weight_kn)
    }
}

/// Dequantize one packed output row using ORT's LSB-first affine NBits layout.
///
/// `scales` contains one value per K block. `zero_points`, when present,
/// contains those block zero points packed with the same bit width.
pub(super) fn dequantize_nbits_row(
    packed: &[u8],
    scales: &[f32],
    zero_points: Option<&[u8]>,
    output: &mut [f32],
    bits: usize,
    block_size: usize,
) {
    for (depth, value) in output.iter_mut().enumerate() {
        *value = dequantize_nbits_value(packed, scales, zero_points, depth, bits, block_size);
    }
}

#[inline]
fn dequantize_nbits_value(
    packed: &[u8],
    scales: &[f32],
    zero_points: Option<&[u8]>,
    depth: usize,
    bits: usize,
    block_size: usize,
) -> f32 {
    let mask = if bits == 8 {
        u8::MAX
    } else {
        (1u8 << bits) - 1
    };
    let default_zero_point = 1u8 << (bits - 1);
    let block = depth / block_size;
    let within_block = depth % block_size;
    let bit_offset = within_block * bits;
    let quantized =
        (packed[block * block_size * bits / 8 + bit_offset / 8] >> (bit_offset % 8)) & mask;
    let zero_point = zero_points.map_or(default_zero_point, |points| {
        let bit_offset = block * bits;
        (points[bit_offset / 8] >> (bit_offset % 8)) & mask
    });
    (quantized as f32 - zero_point as f32) * scales[block]
}

fn configured_decode_threads() -> Option<usize> {
    let value = std::env::var(DECODE_THREADS_ENV).ok();
    let available = std::thread::available_parallelism().ok()?.get();
    resolve_decode_threads(value.as_deref(), available)
}

fn resolve_decode_threads(raw: Option<&str>, available: usize) -> Option<usize> {
    let available = std::num::NonZeroUsize::new(available)?.get();
    let threads = match raw {
        Some("0") => return None,
        Some(raw) => raw.parse::<usize>().unwrap_or(DEFAULT_DECODE_THREADS),
        None => DEFAULT_DECODE_THREADS,
    };
    (threads > 0).then(|| threads.min(available))
}

fn build_decode_pool(
    threads: Option<usize>,
) -> std::result::Result<Option<rayon::ThreadPool>, String> {
    threads
        .map(|threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|index| format!("onnx-genai-decode-{index}"))
                .build()
                .map_err(|err| format!("failed to build {DECODE_THREADS_ENV} pool: {err}"))
        })
        .transpose()
}

fn with_decode_pool<T: Send>(operation: impl FnOnce() -> T + Send) -> Result<T> {
    // If we are already resident inside a `with_decode_pool_scope` installation
    // on this worker thread, run inline: the enclosing `pool.install(...)` already
    // put us on a decode-pool worker, so a fresh `install` here would only add a
    // redundant external-thread-to-pool crossing (task publication + wakeup +
    // join) per projection -- exactly the per-op fork-join fragmentation the
    // whole-forward residency scope eliminates. Inline `operation()` still
    // fans out via rayon's work-stealing on the current (decode) pool.
    if IN_DECODE_POOL.with(Cell::get) {
        return Ok(operation());
    }
    match DECODE_POOL.get_or_init(|| build_decode_pool(configured_decode_threads())) {
        Ok(Some(pool)) => Ok(pool.install(operation)),
        Ok(None) => Ok(operation()),
        Err(message) => Err(error(message.clone())),
    }
}

thread_local! {
    /// Per-worker-thread flag marking that the current thread is executing
    /// inside a [`with_decode_pool_scope`] installation. Set on the decode-pool
    /// worker that runs the wrapped forward pass so the inner [`with_decode_pool`]
    /// calls run inline instead of re-installing.
    static IN_DECODE_POOL: Cell<bool> = const { Cell::new(false) };
}

/// RAII guard that marks the current thread as resident inside the decode pool
/// and restores the previous state on drop -- including during panic unwinding,
/// so a panicking forward pass cannot leak a stale `true` onto a pooled worker.
struct DecodeResidencyGuard {
    previous: bool,
}

impl DecodeResidencyGuard {
    fn enter() -> Self {
        let previous = IN_DECODE_POOL.with(|flag| flag.replace(true));
        Self { previous }
    }
}

impl Drop for DecodeResidencyGuard {
    fn drop(&mut self) {
        let previous = self.previous;
        IN_DECODE_POOL.with(|flag| flag.set(previous));
    }
}

/// Run `f` with the whole call tree resident inside the bounded M=1 decode pool.
///
/// Wrapping an entire single-token CPU decode forward in one installation lets
/// the many inner `MatMulNBits` projections execute inline on already-woken
/// decode-pool workers (see [`with_decode_pool`]), eliminating the per-op
/// external-thread-to-pool crossing that fragments end-to-end decode throughput.
///
/// Behaviour by pool state:
/// * `Ok(Some(pool))` -- install `f` on the decode pool; the residency flag is
///   set *inside* the installed closure (on the worker thread that actually runs
///   `f`, not the caller) and cleared by the RAII guard on exit or panic.
/// * `Ok(None)` -- decode pool opted out (`ONNX_GENAI_CPU_DECODE_THREADS=0`); run
///   `f` inline on the global rayon pool with the flag left `false`, so inner
///   `with_decode_pool` calls keep their existing global-pool behaviour.
/// * `Err(_)` -- pool construction failed; run `f` inline with the flag `false`.
///   The inner `with_decode_pool` calls surface the same error and the forward
///   fails identically to the un-scoped path.
///
/// Callers should enter this scope only for the M=1 CPU decode case; prefill
/// (M>1) and non-CPU paths must keep using the global pool.
pub fn with_decode_pool_scope<R: Send>(f: impl FnOnce() -> R + Send) -> R {
    match DECODE_POOL.get_or_init(|| build_decode_pool(configured_decode_threads())) {
        Ok(Some(pool)) => pool.install(move || {
            let _guard = DecodeResidencyGuard::enter();
            f()
        }),
        _ => f(),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DotKernel {
    Scalar,
    #[cfg(target_arch = "x86_64")]
    AvxVnni,
    #[cfg(target_arch = "x86_64")]
    Avx512Vnni,
}

fn selected_dot_kernel() -> DotKernel {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("avx512vnni")
            && std::arch::is_x86_feature_detected!("avx512vl")
        {
            return DotKernel::Avx512Vnni;
        }
        if std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("avxvnni")
        {
            return DotKernel::AvxVnni;
        }
    }
    DotKernel::Scalar
}

fn quantize_activation_signed(activation: &[f32], padded_k: usize) -> (Vec<i8>, f32) {
    let max_abs = activation
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f32::max);
    if max_abs == 0.0 {
        return (vec![0; padded_k], 0.0);
    }
    let scale = max_abs / 127.0;
    let inverse_scale = scale.recip();
    let mut quantized = vec![0i8; padded_k];
    for (output, &value) in quantized.iter_mut().zip(activation) {
        *output = (value * inverse_scale).round().clamp(-127.0, 127.0) as i8;
    }
    (quantized, scale)
}

fn int4_matmul_m1(
    activation: &[f32],
    weight: &PackedInt4Weight,
    result: &mut [f32],
    k: usize,
    n: usize,
    dot_kernel: DotKernel,
) {
    const BLOCK_SIZE: usize = 32;
    const PACKED_BLOCK_SIZE: usize = BLOCK_SIZE / 2;

    let k_blocks = k.div_ceil(BLOCK_SIZE);
    let padded_k = k_blocks * BLOCK_SIZE;
    debug_assert_eq!(activation.len(), k);
    debug_assert_eq!(weight.values.len(), n * k_blocks * PACKED_BLOCK_SIZE);
    debug_assert_eq!(weight.scales.len(), n * k_blocks);
    debug_assert_eq!(result.len(), n);

    let (activation, activation_scale) = quantize_activation_signed(activation, padded_k);
    let compute = |output_start: usize, outputs: &mut [f32]| {
        for (offset, output) in outputs.iter_mut().enumerate() {
            let output_index = output_start + offset;
            let packed_start = output_index * k_blocks * PACKED_BLOCK_SIZE;
            let packed_end = packed_start + k_blocks * PACKED_BLOCK_SIZE;
            let scale_start = output_index * k_blocks;
            let scale_end = scale_start + k_blocks;
            *output = int4_dot_row(
                &activation,
                &weight.values[packed_start..packed_end],
                &weight.scales[scale_start..scale_end],
                activation_scale,
                dot_kernel,
            );
        }
    };

    let chunk = output_chunk_len(n, padded_k);
    if chunk < n {
        result
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(chunk_index, outputs)| compute(chunk_index * chunk, outputs));
    } else {
        compute(0, result);
    }
}

fn int4_dot_row(
    activation: &[i8],
    packed_weight: &[u8],
    scales: &[f32],
    activation_scale: f32,
    kernel: DotKernel,
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        match kernel {
            DotKernel::AvxVnni => {
                // SAFETY: selected_dot_kernel checked AVX2 and AVX-VNNI.
                return unsafe {
                    int4_dot_row_avxvnni(activation, packed_weight, scales, activation_scale)
                };
            }
            DotKernel::Avx512Vnni => {
                // SAFETY: selected_dot_kernel checked AVX2, AVX512-VNNI, and AVX512VL.
                return unsafe {
                    int4_dot_row_avx512vnni(activation, packed_weight, scales, activation_scale)
                };
            }
            DotKernel::Scalar => {}
        }
    }
    int4_dot_row_scalar(activation, packed_weight, scales, activation_scale)
}

fn int4_dot_row_scalar(
    activation: &[i8],
    packed_weight: &[u8],
    scales: &[f32],
    activation_scale: f32,
) -> f32 {
    debug_assert_eq!(activation.len(), scales.len() * 32);
    debug_assert_eq!(packed_weight.len(), scales.len() * 16);
    let mut value = 0.0f32;
    for (block, &scale) in scales.iter().enumerate() {
        let activation = &activation[block * 32..(block + 1) * 32];
        let packed = &packed_weight[block * 16..(block + 1) * 16];
        let mut dot = 0i32;
        for (pair, &byte) in packed.iter().enumerate() {
            dot += activation[pair * 2] as i32 * (i32::from(byte & 0x0f) - 8);
            dot += activation[pair * 2 + 1] as i32 * (i32::from(byte >> 4) - 8);
        }
        value += dot as f32 * scale;
    }
    value * activation_scale
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avxvnni")]
unsafe fn int4_dot_row_avxvnni(
    activation: &[i8],
    packed_weight: &[u8],
    scales: &[f32],
    activation_scale: f32,
) -> f32 {
    use std::arch::x86_64::*;

    let mut accumulator = _mm256_setzero_ps();
    let low_mask = _mm_set1_epi8(0x0f);
    let zero_point = _mm256_set1_epi8(8);
    for (block, &scale) in scales.iter().enumerate() {
        // SAFETY: each scale corresponds to 32 activation bytes and 16 packed bytes.
        let packed = unsafe { _mm_loadu_si128(packed_weight.as_ptr().add(block * 16).cast()) };
        let low = _mm_and_si128(packed, low_mask);
        let high = _mm_and_si128(_mm_srli_epi16(packed, 4), low_mask);
        let weight = _mm256_sub_epi8(
            _mm256_set_m128i(_mm_unpackhi_epi8(low, high), _mm_unpacklo_epi8(low, high)),
            zero_point,
        );
        // SAFETY: each block has 32 activation bytes, including zero padding.
        let activation = unsafe { _mm256_loadu_si256(activation.as_ptr().add(block * 32).cast()) };
        let absolute_weight = _mm256_sign_epi8(weight, weight);
        let signed_activation = _mm256_sign_epi8(activation, weight);
        let dot =
            _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), absolute_weight, signed_activation);
        let scaled = _mm256_mul_ps(_mm256_cvtepi32_ps(dot), _mm256_set1_ps(scale));
        accumulator = _mm256_add_ps(accumulator, scaled);
    }
    horizontal_sum_f32_256(accumulator) * activation_scale
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,avx512vnni,avx512vl")]
unsafe fn int4_dot_row_avx512vnni(
    activation: &[i8],
    packed_weight: &[u8],
    scales: &[f32],
    activation_scale: f32,
) -> f32 {
    use std::arch::x86_64::*;

    let mut accumulator = _mm256_setzero_ps();
    let low_mask = _mm_set1_epi8(0x0f);
    let zero_point = _mm256_set1_epi8(8);
    for (block, &scale) in scales.iter().enumerate() {
        // SAFETY: each scale corresponds to 32 activation bytes and 16 packed bytes.
        let packed = unsafe { _mm_loadu_si128(packed_weight.as_ptr().add(block * 16).cast()) };
        let low = _mm_and_si128(packed, low_mask);
        let high = _mm_and_si128(_mm_srli_epi16(packed, 4), low_mask);
        let weight = _mm256_sub_epi8(
            _mm256_set_m128i(_mm_unpackhi_epi8(low, high), _mm_unpacklo_epi8(low, high)),
            zero_point,
        );
        // SAFETY: each block has 32 activation bytes, including zero padding.
        let activation = unsafe { _mm256_loadu_si256(activation.as_ptr().add(block * 32).cast()) };
        let absolute_weight = _mm256_sign_epi8(weight, weight);
        let signed_activation = _mm256_sign_epi8(activation, weight);
        let dot = _mm256_dpbusd_epi32(_mm256_setzero_si256(), absolute_weight, signed_activation);
        let scaled = _mm256_mul_ps(_mm256_cvtepi32_ps(dot), _mm256_set1_ps(scale));
        accumulator = _mm256_add_ps(accumulator, scaled);
    }
    horizontal_sum_f32_256(accumulator) * activation_scale
}

#[cfg(target_arch = "x86_64")]
fn horizontal_sum_f32_256(value: std::arch::x86_64::__m256) -> f32 {
    // SAFETY: __m256 and [f32; 8] are both 32-byte plain-data values.
    let lanes: [f32; 8] = unsafe { std::mem::transmute(value) };
    lanes.into_iter().sum()
}

fn quantize_activation(activation: &[f32], padded_k: usize) -> (Vec<u8>, f32) {
    let max_abs = activation
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f32::max);
    if max_abs == 0.0 {
        return (vec![128; padded_k], 0.0);
    }
    let scale = max_abs / 127.0;
    let inverse_scale = scale.recip();
    let mut quantized = vec![128u8; padded_k];
    for (output, &value) in quantized.iter_mut().zip(activation) {
        let signed = (value * inverse_scale).round().clamp(-127.0, 127.0) as i8;
        *output = (signed as i16 + 128) as u8;
    }
    (quantized, scale)
}

fn int8_matmul(
    activations: &[f32],
    weight: &Int8Weight,
    result: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    block_size: usize,
    dot_kernel: DotKernel,
) {
    let k_blocks = k.div_ceil(block_size);
    let padded_k = k_blocks * block_size;
    debug_assert_eq!(weight.values.len(), n * padded_k);
    debug_assert_eq!(weight.scales.len(), n * k_blocks);
    debug_assert_eq!(weight.block_sums.len(), n * k_blocks);

    if m == 1 {
        let (activation, activation_scale) = quantize_activation(activations, padded_k);
        int8_row(
            &activation,
            activation_scale,
            weight,
            result,
            k_blocks,
            padded_k,
            block_size,
            dot_kernel,
            true,
        );
    } else {
        let parallel_columns =
            m < rayon::current_num_threads() && output_chunk_len(n, padded_k) < n;
        result
            .par_chunks_mut(n)
            .zip(activations.par_chunks_exact(k))
            .for_each(|(output, activation)| {
                let (activation, activation_scale) = quantize_activation(activation, padded_k);
                int8_row(
                    &activation,
                    activation_scale,
                    weight,
                    output,
                    k_blocks,
                    padded_k,
                    block_size,
                    dot_kernel,
                    parallel_columns,
                );
            });
    }
}

#[allow(clippy::too_many_arguments)]
fn int8_row(
    activation: &[u8],
    activation_scale: f32,
    weight: &Int8Weight,
    result: &mut [f32],
    k_blocks: usize,
    padded_k: usize,
    block_size: usize,
    dot_kernel: DotKernel,
    parallel: bool,
) {
    let compute = |output_start: usize, outputs: &mut [f32]| {
        for (offset, output) in outputs.iter_mut().enumerate() {
            let output_index = output_start + offset;
            let mut value = 0.0f32;
            let weight_row = &weight.values[output_index * padded_k..(output_index + 1) * padded_k];
            for block in 0..k_blocks {
                let start = block * block_size;
                let end = start + block_size;
                let unsigned_dot =
                    dot_u8_i8(&activation[start..end], &weight_row[start..end], dot_kernel);
                let signed_dot =
                    unsigned_dot - 128 * weight.block_sums[output_index * k_blocks + block];
                value += signed_dot as f32
                    * (activation_scale * weight.scales[output_index * k_blocks + block]);
            }
            *output = value;
        }
    };

    let chunk = output_chunk_len(result.len(), padded_k);
    if parallel && chunk < result.len() {
        result
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(chunk_index, outputs)| compute(chunk_index * chunk, outputs));
    } else {
        compute(0, result);
    }
}

fn dot_u8_i8(activation: &[u8], weight: &[i8], kernel: DotKernel) -> i32 {
    debug_assert_eq!(activation.len(), weight.len());
    #[cfg(target_arch = "x86_64")]
    {
        match kernel {
            DotKernel::AvxVnni => {
                // SAFETY: selected_dot_kernel checked AVX-VNNI at runtime.
                return unsafe { dot_u8_i8_avxvnni(activation, weight) };
            }
            DotKernel::Avx512Vnni => {
                // SAFETY: selected_dot_kernel checked AVX512-VNNI and AVX512VL.
                return unsafe { dot_u8_i8_avx512vnni(activation, weight) };
            }
            DotKernel::Scalar => {}
        }
    }
    dot_u8_i8_scalar(activation, weight)
}

fn dot_u8_i8_scalar(activation: &[u8], weight: &[i8]) -> i32 {
    activation
        .iter()
        .zip(weight)
        .map(|(&activation, &weight)| activation as i32 * weight as i32)
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avxvnni")]
unsafe fn dot_u8_i8_avxvnni(activation: &[u8], weight: &[i8]) -> i32 {
    use std::arch::x86_64::*;

    let vector_len = activation.len() / 32 * 32;
    let mut accumulator = _mm256_setzero_si256();
    for index in (0..vector_len).step_by(32) {
        // SAFETY: index is within equal-length slices and loadu permits unaligned pointers.
        let a = unsafe { _mm256_loadu_si256(activation.as_ptr().add(index).cast()) };
        // SAFETY: index is within equal-length slices and loadu permits unaligned pointers.
        let b = unsafe { _mm256_loadu_si256(weight.as_ptr().add(index).cast()) };
        accumulator = _mm256_dpbusd_avx_epi32(accumulator, a, b);
    }
    horizontal_sum_256(accumulator)
        + dot_u8_i8_scalar(&activation[vector_len..], &weight[vector_len..])
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512vnni,avx512vl")]
unsafe fn dot_u8_i8_avx512vnni(activation: &[u8], weight: &[i8]) -> i32 {
    use std::arch::x86_64::*;

    let vector_len = activation.len() / 32 * 32;
    let mut accumulator = _mm256_setzero_si256();
    for index in (0..vector_len).step_by(32) {
        // SAFETY: index is within equal-length slices and loadu permits unaligned pointers.
        let a = unsafe { _mm256_loadu_si256(activation.as_ptr().add(index).cast()) };
        // SAFETY: index is within equal-length slices and loadu permits unaligned pointers.
        let b = unsafe { _mm256_loadu_si256(weight.as_ptr().add(index).cast()) };
        accumulator = _mm256_dpbusd_epi32(accumulator, a, b);
    }
    horizontal_sum_256(accumulator)
        + dot_u8_i8_scalar(&activation[vector_len..], &weight[vector_len..])
}

#[cfg(target_arch = "x86_64")]
fn horizontal_sum_256(value: std::arch::x86_64::__m256i) -> i32 {
    // SAFETY: __m256i and [i32; 8] are both 32-byte plain-data values.
    let lanes: [i32; 8] = unsafe { std::mem::transmute(value) };
    lanes.into_iter().sum()
}

#[derive(Clone, Copy)]
enum WeightLayout {
    Kn,
    Nk,
}

fn gemv_nk(activation: &[f32], weight_nk: &[f32], result: &mut [f32], k: usize, n: usize) {
    debug_assert_eq!(activation.len(), k);
    debug_assert_eq!(weight_nk.len(), n * k);
    debug_assert_eq!(result.len(), n);
    let compute = |output_start: usize, outputs: &mut [f32]| {
        let weights = &weight_nk[output_start * k..(output_start + outputs.len()) * k];
        for (output, weight) in outputs.iter_mut().zip(weights.chunks_exact(k)) {
            *output = activation.iter().zip(weight).map(|(&a, &b)| a * b).sum();
        }
    };
    let chunk = output_chunk_len(n, k);
    if chunk < n {
        result
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(chunk_index, outputs)| compute(chunk_index * chunk, outputs));
    } else {
        compute(0, result);
    }
}

const MIN_PARALLEL_DOT_PRODUCTS_PER_TASK: usize = 32 * 1024;
const MIN_PARALLEL_DOT_PRODUCTS_PER_THREAD: usize = 8 * 1024;
const MANY_THREAD_DOT_PRODUCTS_PER_THREAD: usize = 64 * 1024;
const MIN_OUTPUTS_PER_TASK: usize = 16;
const MANY_THREAD_CUTOFF: usize = 48;

fn output_chunk_len(n: usize, k: usize) -> usize {
    let threads = rayon::current_num_threads();
    let total_work = n.saturating_mul(k);
    // Small projections amortize Rayon well on one socket, but dispatching each
    // one across a larger pool costs more than its GEMV on the dual-socket host.
    let work_per_thread = if threads <= MANY_THREAD_CUTOFF {
        MIN_PARALLEL_DOT_PRODUCTS_PER_THREAD
    } else {
        MANY_THREAD_DOT_PRODUCTS_PER_THREAD
    };
    if threads <= 1 || total_work < threads.saturating_mul(work_per_thread) {
        return n.max(1);
    }
    let max_tasks = if threads <= MANY_THREAD_CUTOFF {
        threads.saturating_mul(2)
    } else {
        threads
    };
    let tasks = total_work
        .div_ceil(MIN_PARALLEL_DOT_PRODUCTS_PER_TASK)
        .min(max_tasks)
        .min(n);
    if tasks < 2 {
        return n.max(1);
    }
    n.div_ceil(tasks).max(MIN_OUTPUTS_PER_TASK).min(n)
}

fn optional_input<'a>(inputs: &'a [TensorView<'a>], index: usize) -> Option<&'a TensorView<'a>> {
    inputs.get(index).filter(|input| !input.is_absent())
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

fn require_flat_or_matrix_shape(
    name: &str,
    got: &[usize],
    rows: usize,
    columns: usize,
) -> Result<()> {
    if got != [rows * columns] && got != [rows, columns] {
        return Err(error(format!(
            "{name} must have shape [{}] or [{rows}, {columns}], got {got:?}",
            rows * columns
        )));
    }
    Ok(())
}

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("MatMulNBits: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuExecutionProvider;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ep_api::ExecutionProvider;
    use onnx_runtime_ir::{Attribute, Graph, NodeId, static_shape};
    use onnx_runtime_loader::{Model, encode_model_proto};

    fn model_node(
        a_shape: &[usize],
        b_shape: &[usize],
        scales_shape: &[usize],
        zero_points_shape: Option<&[usize]>,
        output_shape: &[usize],
        k: usize,
        n: usize,
        block_size: usize,
    ) -> (Graph, NodeId) {
        let mut graph = Graph::new();
        graph.opset_imports.insert("com.microsoft".into(), 1);
        let mut inputs = Vec::new();
        for (name, dtype, shape) in [
            ("A", DataType::Float32, a_shape),
            ("B", DataType::Uint8, b_shape),
            ("scales", DataType::Float32, scales_shape),
        ] {
            let value = graph.create_named_value(name, dtype, static_shape(shape.iter().copied()));
            graph.add_input(value);
            inputs.push(Some(value));
        }
        if let Some(shape) = zero_points_shape {
            let value = graph.create_named_value(
                "zero_points",
                DataType::Uint8,
                static_shape(shape.iter().copied()),
            );
            graph.add_input(value);
            inputs.push(Some(value));
        }
        let output = graph.create_named_value(
            "Y",
            DataType::Float32,
            static_shape(output_shape.iter().copied()),
        );
        let mut node = Node::new(NodeId(0), "MatMulNBits", inputs, vec![output]);
        node.domain = "com.microsoft".into();
        node.attributes.insert("K".into(), Attribute::Int(k as i64));
        node.attributes.insert("N".into(), Attribute::Int(n as i64));
        node.attributes.insert("bits".into(), Attribute::Int(4));
        node.attributes
            .insert("block_size".into(), Attribute::Int(block_size as i64));
        let node = graph.insert_node(node);
        graph.add_output(output);
        (graph, node)
    }

    fn test_kernel(k: usize, n: usize, block_size: usize) -> MatMulNBitsKernel {
        MatMulNBitsKernel {
            k,
            n,
            bits: 4,
            block_size,
            accuracy_level: 0,
            constant_inputs: [false; 5],
            weight_nk: OnceLock::new(),
            int8_weight: OnceLock::new(),
            packed_int4_weight: OnceLock::new(),
            #[cfg(feature = "mlas")]
            mlas_packed: OnceLock::new(),
        }
    }

    fn accuracy4_kernel(k: usize, n: usize, block_size: usize) -> MatMulNBitsKernel {
        MatMulNBitsKernel {
            accuracy_level: 4,
            ..test_kernel(k, n, block_size)
        }
    }

    fn quantize(
        weights_nk: &[f32],
        n: usize,
        k: usize,
        block_size: usize,
        asymmetric: bool,
    ) -> (Vec<u8>, Vec<f32>, Option<Vec<u8>>, Vec<f32>) {
        let blocks = k.div_ceil(block_size);
        let blob = block_size / 2;
        let mut packed = vec![0u8; n * blocks * blob];
        let mut scales = vec![0.0f32; n * blocks];
        let mut zps = vec![0u8; n * blocks.div_ceil(2)];
        let mut dequantized = vec![0.0f32; n * k];
        for row in 0..n {
            for block in 0..blocks {
                let start = block * block_size;
                let end = (start + block_size).min(k);
                let values = &weights_nk[row * k + start..row * k + end];
                let (scale, zp) = if asymmetric {
                    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
                    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let scale = ((max - min) / 15.0).max(1e-6);
                    (scale, (-min / scale).round().clamp(0.0, 15.0) as u8)
                } else {
                    let max_abs = values.iter().map(|value| value.abs()).fold(0.0, f32::max);
                    ((max_abs / 7.0).max(1e-6), 8)
                };
                scales[row * blocks + block] = scale;
                if asymmetric {
                    let byte = &mut zps[row * blocks.div_ceil(2) + block / 2];
                    *byte |= zp << (4 * (block % 2));
                }
                for (offset, &value) in values.iter().enumerate() {
                    let q = (value / scale + zp as f32).round().clamp(0.0, 15.0) as u8;
                    packed[(row * blocks + block) * blob + offset / 2] |= q << (4 * (offset % 2));
                    dequantized[row * k + start + offset] = (q as f32 - zp as f32) * scale;
                }
            }
        }
        (packed, scales, asymmetric.then_some(zps), dequantized)
    }

    fn reference(a: &[f32], weights_nk: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut output = vec![0.0f32; m * n];
        for row in 0..m {
            for column in 0..n {
                for depth in 0..k {
                    output[row * n + column] += a[row * k + depth] * weights_nk[column * k + depth];
                }
            }
        }
        output
    }

    fn dequantize_reference(
        packed: &[u8],
        scales: &[f32],
        zero_points: Option<&[u8]>,
        n: usize,
        k: usize,
        block_size: usize,
    ) -> Vec<f32> {
        let blocks = k.div_ceil(block_size);
        let blob_size = block_size / 2;
        let zp_row_bytes = blocks.div_ceil(2);
        let mut weights = vec![0.0; n * k];
        for output in 0..n {
            for depth in 0..k {
                let block = depth / block_size;
                let within_block = depth % block_size;
                let byte = packed[(output * blocks + block) * blob_size + within_block / 2];
                let q = if within_block.is_multiple_of(2) {
                    byte & 0x0f
                } else {
                    byte >> 4
                };
                let zero_point = zero_points.map_or(8, |points| {
                    let byte = points[output * zp_row_bytes + block / 2];
                    if block.is_multiple_of(2) {
                        byte & 0x0f
                    } else {
                        byte >> 4
                    }
                });
                weights[output * k + depth] =
                    (q as f32 - zero_point as f32) * scales[output * blocks + block];
            }
        }
        weights
    }

    fn quantize_symmetric_2bit(
        weights_nk: &[f32],
        n: usize,
        k: usize,
        block_size: usize,
    ) -> (Vec<u8>, Vec<f32>) {
        let blocks = k.div_ceil(block_size);
        let blob_size = block_size / 4;
        let mut packed = vec![0u8; n * blocks * blob_size];
        let mut scales = vec![0.0f32; n * blocks];
        for output in 0..n {
            for block in 0..blocks {
                let start = block * block_size;
                let end = (start + block_size).min(k);
                let values = &weights_nk[output * k + start..output * k + end];
                let max_abs = values.iter().map(|value| value.abs()).fold(0.0, f32::max);
                let scale = max_abs.max(1e-6);
                scales[output * blocks + block] = scale;
                for (offset, &value) in values.iter().enumerate() {
                    let q = (value / scale + 2.0).round().clamp(0.0, 3.0) as u8;
                    packed[(output * blocks + block) * blob_size + offset / 4] |=
                        q << (2 * (offset % 4));
                }
            }
        }
        (packed, scales)
    }

    fn dequantize_2bit_reference(
        packed: &[u8],
        scales: &[f32],
        n: usize,
        k: usize,
        block_size: usize,
    ) -> Vec<f32> {
        let blocks = k.div_ceil(block_size);
        let blob_size = block_size / 4;
        let mut dequantized = vec![0.0f32; n * k];
        for output in 0..n {
            for depth in 0..k {
                let block = depth / block_size;
                let within_block = depth % block_size;
                let byte = packed[(output * blocks + block) * blob_size + within_block / 4];
                let q = (byte >> (2 * (within_block % 4))) & 0x03;
                dequantized[output * k + depth] =
                    (q as f32 - 2.0) * scales[output * blocks + block];
            }
        }
        dequantized
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-5,
                "index {index}: actual={actual}, expected={expected}"
            );
        }
    }

    fn assert_int8_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = 0.05 + 0.05 * expected.abs();
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
            );
        }
    }

    fn accuracy4_model(m: usize, k: usize, n: usize, block_size: usize) -> (Graph, NodeId) {
        let blocks = k.div_ceil(block_size);
        let (mut graph, node) = model_node(
            &[m, k],
            &[n, blocks, block_size / 2],
            &[n, blocks],
            None,
            &[m, n],
            k,
            n,
            block_size,
        );
        graph
            .node_mut(node)
            .attributes
            .insert("accuracy_level".into(), Attribute::Int(4));
        let proto = encode_model_proto(&Model::new(&graph)).expect("IR model must encode to ONNX");
        let attribute = &proto.graph.as_ref().unwrap().node[0].attribute;
        assert!(
            attribute
                .iter()
                .any(|attr| attr.name == "accuracy_level" && attr.i == 4)
        );
        (graph, node)
    }

    fn run_accuracy4_case(m: usize, k: usize, n: usize, block_size: usize) {
        let a_values: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 % 43) as f32 - 21.0) / 13.0)
            .collect();
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 19 % 47) as f32 - 23.0) / 12.0)
            .collect();
        let (packed, scales, _, _) = quantize(&weights, n, k, block_size, false);
        let dequantized = dequantize_reference(&packed, &scales, None, n, k, block_size);
        let expected = reference(&a_values, &dequantized, m, k, n);
        let (graph, node) = accuracy4_model(m, k, n, block_size);
        let model = Model::new(&graph);
        let kernel = CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .unwrap();
        let a = Owned::f32(&[m, k], &a_values);
        let b = Owned::u8(&[n, k.div_ceil(block_size), block_size / 2], &packed);
        let scales = Owned::f32(&[n, k.div_ceil(block_size)], &scales);
        let mut y = Owned::zeros_f32(&[m, n]);
        kernel
            .execute(&[a.view(), b.view(), scales.view()], &mut [y.view_mut()])
            .unwrap();
        assert_int8_close(&y.to_f32(), &expected);
    }

    #[test]
    fn matmulnbits_accuracy4_block32_partial_k_m1_matches_fp32_reference() {
        run_accuracy4_case(1, 45, 9, 32);
    }

    #[test]
    fn matmulnbits_accuracy4_block128_partial_k_batched_matches_fp32_reference() {
        run_accuracy4_case(3, 141, 7, 128);
    }

    #[test]
    fn matmulnbits_accuracy4_prepack_reuses_selected_weight_format() {
        let (k, n, block_size) = (45, 5, 32);
        let activations: Vec<f32> = (0..k)
            .map(|i| ((i * 11 % 37) as f32 - 18.0) / 9.0)
            .collect();
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 13 % 41) as f32 - 20.0) / 11.0)
            .collect();
        let (packed, scales, _, _) = quantize(&weights, n, k, block_size, false);
        let mut kernel = accuracy4_kernel(k, n, block_size);
        kernel.set_constant_inputs(&[false, true, true]);
        let a = Owned::f32(&[1, k], &activations);
        let b = Owned::u8(&[n, 2, 16], &packed);
        let scales = Owned::f32(&[n, 2], &scales);
        let mut y = Owned::zeros_f32(&[1, n]);
        kernel
            .execute(&[a.view(), b.view(), scales.view()], &mut [y.view_mut()])
            .unwrap();
        let direct_int4 = selected_dot_kernel() != DotKernel::Scalar;
        let cached = if direct_int4 {
            kernel
                .packed_int4_weight
                .get()
                .expect("packed int4 weight must be cached")
                .values
                .as_ptr()
        } else {
            kernel
                .int8_weight
                .get()
                .expect("int8 weight must be cached")
                .values
                .as_ptr()
                .cast()
        };
        kernel
            .execute(&[a.view(), b.view(), scales.view()], &mut [y.view_mut()])
            .unwrap();
        let reused = if direct_int4 {
            kernel.packed_int4_weight.get().unwrap().values.as_ptr()
        } else {
            kernel.int8_weight.get().unwrap().values.as_ptr().cast()
        };
        assert_eq!(reused, cached);
        assert!(kernel.weight_nk.get().is_none());
        assert_eq!(kernel.packed_int4_weight.get().is_some(), direct_int4);
        assert_eq!(kernel.int8_weight.get().is_some(), !direct_int4);
    }

    #[test]
    fn matmulnbits_accuracy4_vnni_matches_scalar_when_available() {
        let activation: Vec<u8> = (0..128).map(|i| ((i * 29 + 7) % 255) as u8).collect();
        let weight: Vec<i8> = (0..128).map(|i| ((i * 17 % 31) as i8) - 15).collect();
        let scalar = dot_u8_i8(&activation, &weight, DotKernel::Scalar);
        let selected = selected_dot_kernel();
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avxvnni")
            || (std::arch::is_x86_feature_detected!("avx512vnni")
                && std::arch::is_x86_feature_detected!("avx512vl"))
        {
            assert_ne!(
                selected,
                DotKernel::Scalar,
                "a VNNI CPU must select the VNNI path"
            );
        }
        assert_eq!(dot_u8_i8(&activation, &weight, selected), scalar);

        let activations: Vec<f32> = (0..256)
            .map(|i| ((i * 23 % 53) as f32 - 26.0) / 17.0)
            .collect();
        let values: Vec<i8> = (0..384).map(|i| ((i * 11 % 16) as i8) - 8).collect();
        let block_sums = values
            .chunks_exact(128)
            .map(|block| block.iter().map(|&value| value as i32).sum())
            .collect();
        let prepacked = Int8Weight {
            values,
            scales: vec![0.01, 0.02, 0.03],
            block_sums,
        };
        let mut scalar_output = vec![0.0; 6];
        let mut selected_output = vec![0.0; 6];
        int8_matmul(
            &activations,
            &prepacked,
            &mut scalar_output,
            2,
            128,
            3,
            128,
            DotKernel::Scalar,
        );
        int8_matmul(
            &activations,
            &prepacked,
            &mut selected_output,
            2,
            128,
            3,
            128,
            selected,
        );
        assert_close(&selected_output, &scalar_output);
    }

    #[test]
    fn matmulnbits_direct_int4_gemv_matches_int8_reference() {
        let (k, n, block_size) = (77usize, 9usize, 32usize);
        let blocks = k.div_ceil(block_size);
        let padded_k = blocks * block_size;
        let activations: Vec<f32> = (0..k)
            .map(|i| ((i * 23 % 53) as f32 - 26.0) / 17.0)
            .collect();
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 19 % 47) as f32 - 23.0) / 12.0)
            .collect();
        let (packed, scales, _, _) = quantize(&weights, n, k, block_size, false);
        let packed_weight = PackedInt4Weight {
            values: packed.clone(),
            scales: scales.clone(),
        };
        let kernel = accuracy4_kernel(k, n, block_size);
        let b = Owned::u8(&[n, blocks, block_size / 2], &packed);
        let scales_tensor = Owned::f32(&[n, blocks], &scales);
        let int8_weight = kernel
            .prepack_int8_weight(&b.view(), &scales_tensor.view(), None)
            .unwrap();
        let mut expected = vec![0.0; n];
        let mut scalar = vec![0.0; n];
        let mut actual = vec![0.0; n];
        int8_matmul(
            &activations,
            &int8_weight,
            &mut expected,
            1,
            k,
            n,
            block_size,
            DotKernel::Scalar,
        );
        int4_matmul_m1(
            &activations,
            &packed_weight,
            &mut scalar,
            k,
            n,
            DotKernel::Scalar,
        );
        int4_matmul_m1(
            &activations,
            &packed_weight,
            &mut actual,
            k,
            n,
            selected_dot_kernel(),
        );
        assert_eq!(
            padded_k,
            activations.len().div_ceil(block_size) * block_size
        );
        for (index, ((&actual, &scalar), &expected)) in
            actual.iter().zip(&scalar).zip(&expected).enumerate()
        {
            let tolerance = 1e-4 + 1e-5 * expected.abs();
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {index}: direct int4={actual}, int8 reference={expected}, tolerance={tolerance}"
            );
            assert!(
                (scalar - expected).abs() <= tolerance,
                "index {index}: scalar int4={scalar}, int8 reference={expected}, tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn matmulnbits_direct_int4_parallel_partial_k_matches_serial() {
        let (k, n, block_size) = (77usize, 1025usize, 32usize);
        let blocks = k.div_ceil(block_size);
        let activations: Vec<f32> = (0..k)
            .map(|i| ((i * 23 % 53) as f32 - 26.0) / 17.0)
            .collect();
        let packed_weight = PackedInt4Weight {
            values: (0..n * blocks * block_size / 2)
                .map(|i| ((i * 29 + 7) % 256) as u8)
                .collect(),
            scales: (0..n * blocks)
                .map(|i| ((i * 13 % 17) + 1) as f32 / 100.0)
                .collect(),
        };
        let mut serial = vec![0.0; n];
        let mut parallel = vec![0.0; n];
        let dot_kernel = selected_dot_kernel();
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| {
                int4_matmul_m1(&activations, &packed_weight, &mut serial, k, n, dot_kernel);
            });
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| {
                int4_matmul_m1(
                    &activations,
                    &packed_weight,
                    &mut parallel,
                    k,
                    n,
                    dot_kernel,
                );
            });
        assert_eq!(parallel, serial);
    }

    #[test]
    fn matmulnbits_parallel_n_partition_matches_serial() {
        let (k, n, block_size) = (1025usize, 1025usize, 32usize);
        let padded_k = k.div_ceil(block_size) * block_size;
        let activations: Vec<f32> = (0..k)
            .map(|i| ((i * 23 % 53) as f32 - 26.0) / 17.0)
            .collect();
        let values: Vec<i8> = (0..n * padded_k)
            .map(|i| ((i * 11 % 16) as i8) - 8)
            .collect();
        let block_sums = values
            .chunks_exact(block_size)
            .map(|block| block.iter().map(|&value| value as i32).sum())
            .collect();
        let weight = Int8Weight {
            values,
            scales: vec![0.01; n * k.div_ceil(block_size)],
            block_sums,
        };
        let mut serial = vec![0.0; n];
        let mut parallel = vec![0.0; n];
        rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| {
                int8_matmul(
                    &activations,
                    &weight,
                    &mut serial,
                    1,
                    k,
                    n,
                    block_size,
                    DotKernel::Scalar,
                );
            });
        rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| {
                int8_matmul(
                    &activations,
                    &weight,
                    &mut parallel,
                    1,
                    k,
                    n,
                    block_size,
                    DotKernel::Scalar,
                );
            });
        assert_eq!(parallel, serial);
    }

    #[test]
    fn matmulnbits_partition_scales_with_pool_size_and_work() {
        let chunk = |threads, n, k| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| output_chunk_len(n, k))
        };

        assert_eq!(chunk(1, 4864, 896), 4864);
        assert_eq!(chunk(24, 16, 32), 16);
        assert_eq!(chunk(24, 896, 896), 36);
        assert_eq!(chunk(48, 896, 896), 36);
        assert_eq!(chunk(96, 896, 896), 896);
        assert_eq!(chunk(96, 4864, 896), 4864);
        assert_eq!(chunk(96, 151_936, 896), 1583);
    }

    #[test]
    fn decode_thread_count_defaults_invalid_values_and_clamps() {
        assert_eq!(resolve_decode_threads(None, 96), Some(8));
        assert_eq!(resolve_decode_threads(None, 4), Some(4));
        assert_eq!(resolve_decode_threads(Some(""), 96), Some(8));
        assert_eq!(resolve_decode_threads(Some("0"), 8), None);
        assert_eq!(resolve_decode_threads(Some("4"), 96), Some(4));
        assert_eq!(resolve_decode_threads(Some("1000"), 96), Some(96));
        assert_eq!(resolve_decode_threads(Some("abc"), 96), Some(8));
        assert_eq!(resolve_decode_threads(Some("-4"), 4), Some(4));
        assert_eq!(resolve_decode_threads(Some("4"), 0), None);
    }

    #[test]
    fn decode_thread_pool_supports_global_pool_opt_out() {
        assert!(build_decode_pool(None).unwrap().is_none());
        let pool = build_decode_pool(Some(3)).unwrap().unwrap();
        assert_eq!(pool.install(rayon::current_num_threads), 3);
    }

    #[test]
    fn decode_residency_guard_sets_and_restores_flag() {
        assert!(!IN_DECODE_POOL.with(Cell::get));
        {
            let _outer = DecodeResidencyGuard::enter();
            assert!(IN_DECODE_POOL.with(Cell::get));
            {
                let _inner = DecodeResidencyGuard::enter();
                assert!(IN_DECODE_POOL.with(Cell::get));
            }
            // Nested drop restores the previous (still-resident) state.
            assert!(IN_DECODE_POOL.with(Cell::get));
        }
        assert!(!IN_DECODE_POOL.with(Cell::get));
    }

    #[test]
    fn decode_residency_guard_clears_on_panic() {
        assert!(!IN_DECODE_POOL.with(Cell::get));
        let result = std::panic::catch_unwind(|| {
            let _guard = DecodeResidencyGuard::enter();
            assert!(IN_DECODE_POOL.with(Cell::get));
            panic!("decode forward panicked");
        });
        assert!(result.is_err());
        assert!(
            !IN_DECODE_POOL.with(Cell::get),
            "residency flag must be cleared after a panicking forward unwinds"
        );
    }

    #[test]
    fn with_decode_pool_runs_inline_when_resident() {
        // With the residency flag set, `with_decode_pool` must NOT re-install the
        // pool: it runs `operation` inline on the current thread. Observing the
        // running thread id proves no external-thread-to-pool crossing happened.
        let _guard = DecodeResidencyGuard::enter();
        let caller = std::thread::current().id();
        let ran_on = with_decode_pool(|| std::thread::current().id()).unwrap();
        assert_eq!(
            ran_on, caller,
            "resident with_decode_pool must run inline on the caller thread"
        );
    }

    #[test]
    fn with_decode_pool_scope_marks_residency_when_pool_active() {
        // When a bounded decode pool exists, the scope must set the residency
        // flag on the worker thread that runs the closure, and inner
        // `with_decode_pool` calls must then run inline on that same worker.
        let pool_active = DECODE_POOL
            .get_or_init(|| build_decode_pool(configured_decode_threads()))
            .as_ref()
            .ok()
            .and_then(Option::as_ref)
            .is_some();
        let (flag_inside, inline_same_thread) = with_decode_pool_scope(|| {
            let worker = std::thread::current().id();
            let inner = with_decode_pool(|| std::thread::current().id()).unwrap();
            (IN_DECODE_POOL.with(Cell::get), inner == worker)
        });
        if pool_active {
            assert!(flag_inside, "scope must set residency flag inside the pool");
            assert!(
                inline_same_thread,
                "inner with_decode_pool must run inline on the scope worker"
            );
        }
        // The calling thread never observes the flag set (it is set on the worker).
        assert!(!IN_DECODE_POOL.with(Cell::get));
    }

    #[test]
    fn matmulnbits_symmetric_block32_matches_independent_dequantization() {
        let (m, k, n, block_size) = (3, 64, 8, 32);
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 % 29) as f32 - 14.0) / 11.0)
            .collect();
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 13 % 31) as f32 - 15.0) / 9.0)
            .collect();
        let (packed, scales, _, dequantized) = quantize(&weights, n, k, block_size, false);
        let (graph, node) = model_node(
            &[m, k],
            &[n, 2, 16],
            &[n, 2],
            None,
            &[m, n],
            k,
            n,
            block_size,
        );
        let model = Model::new(&graph);
        let kernel = CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .unwrap();
        let a = Owned::f32(&[m, k], &a);
        let b = Owned::u8(&[n, 2, 16], &packed);
        let scales = Owned::f32(&[n, 2], &scales);
        let mut y = Owned::zeros_f32(&[m, n]);
        kernel
            .execute(&[a.view(), b.view(), scales.view()], &mut [y.view_mut()])
            .unwrap();
        assert_close(&y.to_f32(), &reference(&a.to_f32(), &dequantized, m, k, n));
    }

    #[test]
    fn matmulnbits_2bit_symmetric_block32_matches_dequantized_f32_reference() {
        let (m, k, n, block_size) = (3, 45, 7, 32);
        let a_values: Vec<f32> = (0..m * k)
            .map(|i| ((i * 17 % 43) as f32 - 21.0) / 13.0)
            .collect();
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 19 % 47) as f32 - 23.0) / 12.0)
            .collect();
        let (packed, scales) = quantize_symmetric_2bit(&weights, n, k, block_size);
        let dequantized = dequantize_2bit_reference(&packed, &scales, n, k, block_size);
        let blocks = k.div_ceil(block_size);
        let (graph, node) = model_node(
            &[m, k],
            &[n, blocks, block_size / 4],
            &[n, blocks],
            None,
            &[m, n],
            k,
            n,
            block_size,
        );
        let mut graph = graph;
        graph
            .node_mut(node)
            .attributes
            .insert("bits".into(), Attribute::Int(2));
        let model = Model::new(&graph);
        let kernel = CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .expect("CPU EP must register bits=2 MatMulNBits");
        let a = Owned::f32(&[m, k], &a_values);
        let b = Owned::u8(&[n, blocks, block_size / 4], &packed);
        let scales = Owned::f32(&[n, blocks], &scales);
        let mut y = Owned::zeros_f32(&[m, n]);
        kernel
            .execute(&[a.view(), b.view(), scales.view()], &mut [y.view_mut()])
            .unwrap();
        assert_close(&y.to_f32(), &reference(&a_values, &dequantized, m, k, n));
    }

    #[test]
    fn matmulnbits_2bit_unpacks_low_bits_first() {
        let k = 32;
        let (graph, node) = model_node(&[1, k], &[1, 1, 8], &[1], None, &[1, 1], k, 1, 32);
        let mut graph = graph;
        graph
            .node_mut(node)
            .attributes
            .insert("bits".into(), Attribute::Int(2));
        let model = Model::new(&graph);
        let kernel = CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .unwrap();
        let mut activation = vec![0.0; k];
        activation[..4].copy_from_slice(&[1.0, 10.0, 100.0, 1000.0]);
        let mut packed = vec![0xaa; 8];
        packed[0] = 0b11_10_01_00;
        let a = Owned::f32(&[1, k], &activation);
        let b = Owned::u8(&[1, 1, 8], &packed);
        let scales = Owned::f32(&[1], &[1.0]);
        let mut y = Owned::zeros_f32(&[1, 1]);
        kernel
            .execute(&[a.view(), b.view(), scales.view()], &mut [y.view_mut()])
            .unwrap();
        assert_eq!(y.to_f32(), vec![988.0]); // -2*1 + -1*10 + 0*100 + 1*1000
    }

    #[test]
    fn matmulnbits_asymmetric_block16_batched_non_square() {
        let (m, k, n, block_size) = (6, 48, 5, 16);
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7 % 23) as f32 - 5.0) / 8.0)
            .collect();
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 19 % 37) as f32 - 9.0) / 10.0)
            .collect();
        let (packed, scales, zero_points, dequantized) = quantize(&weights, n, k, block_size, true);
        let zero_points = zero_points.unwrap();
        let (graph, node) = model_node(
            &[2, 3, k],
            &[n, 3, 8],
            &[n * 3],
            Some(&[n, 2]),
            &[2, 3, n],
            k,
            n,
            block_size,
        );
        let model = Model::new(&graph);
        let kernel = CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .unwrap();
        let a = Owned::f32(&[2, 3, k], &a);
        let b = Owned::u8(&[n, 3, 8], &packed);
        let scales = Owned::f32(&[n * 3], &scales);
        let zero_points = Owned::u8(&[n, 2], &zero_points);
        let mut y = Owned::zeros_f32(&[2, 3, n]);
        kernel
            .execute(
                &[a.view(), b.view(), scales.view(), zero_points.view()],
                &mut [y.view_mut()],
            )
            .unwrap();
        assert_close(&y.to_f32(), &reference(&a.to_f32(), &dequantized, m, k, n));
    }

    #[test]
    fn matmulnbits_prepacked_m1_block32_symmetric_reuses_weight_for_new_activations() {
        let (k, n, block_size) = (35, 7, 32);
        let a1_values: Vec<f32> = (0..k)
            .map(|i| ((i * 11 % 41) as f32 - 20.0) / 13.0)
            .collect();
        let a2_values: Vec<f32> = a1_values
            .iter()
            .enumerate()
            .map(|(i, &value)| value * -0.5 + i as f32 / 17.0)
            .collect();
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 11 % 41) as f32 - 20.0) / 13.0)
            .collect();
        let (packed, scales, _, _) = quantize(&weights, n, k, block_size, false);
        let dequantized = dequantize_reference(&packed, &scales, None, n, k, block_size);
        let mut kernel = test_kernel(k, n, block_size);
        kernel.set_constant_inputs(&[false, true, true]);

        let b = Owned::u8(&[n, 2, 16], &packed);
        let scales = Owned::f32(&[n, 2], &scales);
        let a1 = Owned::f32(&[1, k], &a1_values);
        let mut y1 = Owned::zeros_f32(&[1, n]);
        kernel
            .execute(&[a1.view(), b.view(), scales.view()], &mut [y1.view_mut()])
            .unwrap();
        assert_close(&y1.to_f32(), &reference(&a1_values, &dequantized, 1, k, n));

        let cached_weight = kernel
            .weight_nk
            .get()
            .expect("M=1 constant B must populate the prepacked weight cache")
            .as_ptr();
        let a2 = Owned::f32(&[1, k], &a2_values);
        let mut y2 = Owned::zeros_f32(&[1, n]);
        kernel
            .execute(&[a2.view(), b.view(), scales.view()], &mut [y2.view_mut()])
            .unwrap();
        assert_eq!(kernel.weight_nk.get().unwrap().as_ptr(), cached_weight);
        assert_close(&y2.to_f32(), &reference(&a2_values, &dequantized, 1, k, n));
        assert_ne!(y1.to_f32(), y2.to_f32());
    }

    #[test]
    fn matmulnbits_prepacked_m1_block128_explicit_zp_partial_block_matches_reference() {
        let (k, n, block_size) = (141, 7, 128);
        let a_values: Vec<f32> = (0..k)
            .map(|i| ((i * 11 % 41) as f32 - 20.0) / 13.0)
            .collect();
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 23 % 47) as f32 - 19.0) / 12.0)
            .collect();
        let (packed, scales, zero_points, _) = quantize(&weights, n, k, block_size, true);
        let zero_points = zero_points.unwrap();
        let dequantized =
            dequantize_reference(&packed, &scales, Some(&zero_points), n, k, block_size);
        let mut kernel = test_kernel(k, n, block_size);
        kernel.set_constant_inputs(&[false, true, true, true]);

        let a = Owned::f32(&[1, k], &a_values);
        let b = Owned::u8(&[n, 2, 64], &packed);
        let scales = Owned::f32(&[n, 2], &scales);
        let zero_points = Owned::u8(&[n, 1], &zero_points);
        let mut y = Owned::zeros_f32(&[1, n]);
        kernel
            .execute(
                &[a.view(), b.view(), scales.view(), zero_points.view()],
                &mut [y.view_mut()],
            )
            .unwrap();

        assert_close(&y.to_f32(), &reference(&a_values, &dequantized, 1, k, n));
        assert!(
            kernel.weight_nk.get().is_some(),
            "M=1 constant B/scales/zero-points must take the prepacked GEMV path"
        );
    }

    #[test]
    fn matmulnbits_m1_dynamic_b_falls_back_without_populating_prepack_cache() {
        let (k, n, block_size) = (35, 5, 32);
        let a_values: Vec<f32> = (0..k).map(|i| ((i * 5 % 29) as f32 - 14.0) / 9.0).collect();
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 7 % 31) as f32 - 15.0) / 10.0)
            .collect();
        let (packed, scales, _, _) = quantize(&weights, n, k, block_size, false);
        let dequantized = dequantize_reference(&packed, &scales, None, n, k, block_size);
        let mut kernel = test_kernel(k, n, block_size);
        kernel.set_constant_inputs(&[false, false, true]);

        let a = Owned::f32(&[1, k], &a_values);
        let b = Owned::u8(&[n, 2, 16], &packed);
        let scales = Owned::f32(&[n, 2], &scales);
        let mut y = Owned::zeros_f32(&[1, n]);
        kernel
            .execute(&[a.view(), b.view(), scales.view()], &mut [y.view_mut()])
            .unwrap();

        assert_close(&y.to_f32(), &reference(&a_values, &dequantized, 1, k, n));
        assert!(
            kernel.weight_nk.get().is_none(),
            "dynamic B must use the fallback rather than populate the prepack cache"
        );
    }

    #[test]
    fn matmulnbits_unpacks_low_nibble_before_high_nibble() {
        let k = 16;
        let (graph, node) = model_node(&[1, k], &[1, 1, 8], &[1], None, &[1, 1], k, 1, 16);
        let model = Model::new(&graph);
        let kernel = CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .unwrap();
        let mut activation = vec![0.0; k];
        activation[0] = 1.0;
        activation[1] = 10.0;
        let mut packed = vec![0x88; 8];
        packed[0] = 0xe1;
        let a = Owned::f32(&[1, k], &activation);
        let b = Owned::u8(&[1, 1, 8], &packed);
        let scales = Owned::f32(&[1], &[1.0]);
        let mut y = Owned::zeros_f32(&[1, 1]);
        kernel
            .execute(&[a.view(), b.view(), scales.view()], &mut [y.view_mut()])
            .unwrap();
        assert_eq!(y.to_f32(), vec![53.0]); // (1-8)*1 + (14-8)*10
    }

    #[test]
    fn matmulnbits_honors_non_contiguous_group_indices() {
        let k = 32;
        let mut graph = Graph::new();
        graph.opset_imports.insert("com.microsoft".into(), 1);
        let a_value = graph.create_named_value("A", DataType::Float32, static_shape([1, k]));
        let b_value = graph.create_named_value("B", DataType::Uint8, static_shape([1, 2, 8]));
        let scales_value =
            graph.create_named_value("scales", DataType::Float32, static_shape([1, 2]));
        let g_idx_value = graph.create_named_value("g_idx", DataType::Int32, static_shape([k]));
        for value in [a_value, b_value, scales_value, g_idx_value] {
            graph.add_input(value);
        }
        let output = graph.create_named_value("Y", DataType::Float32, static_shape([1, 1]));
        let mut node = Node::new(
            NodeId(0),
            "MatMulNBits",
            vec![
                Some(a_value),
                Some(b_value),
                Some(scales_value),
                None,
                Some(g_idx_value),
            ],
            vec![output],
        );
        node.domain = "com.microsoft".into();
        node.attributes.insert("K".into(), Attribute::Int(k as i64));
        node.attributes.insert("N".into(), Attribute::Int(1));
        node.attributes.insert("bits".into(), Attribute::Int(4));
        node.attributes
            .insert("block_size".into(), Attribute::Int(16));
        let node = graph.insert_node(node);
        graph.add_output(output);

        let model = Model::new(&graph);
        let kernel = CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .unwrap();
        let mut activation = vec![1.0; k];
        activation[16..].fill(2.0);
        let a = Owned::f32(&[1, k], &activation);
        let b = Owned::u8(&[1, 2, 8], &vec![0x99; 16]);
        let scales = Owned::f32(&[1, 2], &[1.0, 2.0]);
        let groups: Vec<i32> = (0..k).map(|i| if i < 16 { 1 } else { 0 }).collect();
        let groups = Owned::i32(&[k], &groups);
        let absent_zp = TensorView::absent(DataType::Uint8);
        let mut y = Owned::zeros_f32(&[1, 1]);
        kernel
            .execute(
                &[a.view(), b.view(), scales.view(), absent_zp, groups.view()],
                &mut [y.view_mut()],
            )
            .unwrap();
        assert_eq!(y.to_f32(), vec![64.0]);
    }

    #[test]
    fn matmulnbits_rejects_unsupported_bit_width() {
        let (graph, node) = model_node(&[1, 16], &[1, 1, 8], &[1], None, &[1, 1], 16, 1, 16);
        let mut graph = graph;
        graph
            .node_mut(node)
            .attributes
            .insert("bits".into(), Attribute::Int(8));
        let model = Model::new(&graph);
        let error = CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .err()
            .expect("bits=8 must be rejected");
        assert!(format!("{error}").contains("only bits=2 and bits=4"));
    }

    #[test]
    fn matmulnbits_defaults_missing_bits_to_int4() {
        let k = 16;
        let (graph, node) = model_node(&[1, k], &[1, 1, 8], &[1], None, &[1, 1], k, 1, 16);
        let mut graph = graph;
        graph.node_mut(node).attributes.remove("bits");
        let model = Model::new(&graph);
        let kernel = CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .expect("missing bits must default to 4");
        let mut activation = vec![0.0; k];
        activation[0] = 1.0;
        activation[1] = 10.0;
        let mut packed = vec![0x88; 8];
        packed[0] = 0xe1;
        let a = Owned::f32(&[1, k], &activation);
        let b = Owned::u8(&[1, 1, 8], &packed);
        let scales = Owned::f32(&[1], &[1.0]);
        let mut y = Owned::zeros_f32(&[1, 1]);
        kernel
            .execute(&[a.view(), b.view(), scales.view()], &mut [y.view_mut()])
            .unwrap();
        assert_eq!(y.to_f32(), vec![53.0]);
    }

    #[test]
    fn matmulnbits_rejects_prepacked_weight_layout() {
        let (graph, node) = model_node(&[1, 16], &[1, 1, 8], &[1], None, &[1, 1], 16, 1, 16);
        let mut graph = graph;
        graph
            .node_mut(node)
            .attributes
            .insert("weight_prepacked".into(), Attribute::Int(1));
        let model = Model::new(&graph);
        let error = CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .err()
            .expect("prepacked weights must be rejected");
        let message = format!("{error}");
        assert!(message.contains("weight_prepacked=1"));
        assert!(message.contains("standard (non-prepacked) layout"));
    }

    #[cfg(feature = "mlas")]
    fn mlas_close(actual: &[f32], expected: &[f32], tol: f32, ctx: &str) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            let diff = (a - e).abs();
            let rel = diff / e.abs().max(1.0);
            assert!(
                diff <= tol || rel <= tol,
                "{ctx}: index {i} mlas={a} ref={e} diff={diff}"
            );
        }
    }

    #[cfg(feature = "mlas")]
    fn pseudo(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 * 0.017 + seed).sin()) * 1.5)
            .collect()
    }

    /// The MLAS SQNBit path (`build_mlas_packed` + `mlas_sys::sqnbit_gemm`, the
    /// exact code `execute` runs when the backend is MLAS) must match the
    /// existing dequantize-then-GEMM oracle across block sizes, symmetric and
    /// asymmetric zero points, decode (M=1) and prefill (M>1), both compute
    /// types (`accuracy_level` 0 → CompFp32, 4 → CompInt8), and bias.
    #[cfg(feature = "mlas")]
    #[test]
    fn matmulnbits_mlas_matches_dequant_reference() {
        let (n, k) = (96usize, 256usize);
        for &block_size in &[32usize, 64, 128] {
            let k_blocks = k.div_ceil(block_size);
            let blob = block_size / 2;
            let weights_nk = pseudo(n * k, 0.3);
            for &asymmetric in &[false, true] {
                let (packed, scales, zps, _dq) =
                    quantize(&weights_nk, n, k, block_size, asymmetric);
                let ref_weights =
                    dequantize_reference(&packed, &scales, zps.as_deref(), n, k, block_size);
                let b = Owned::u8(&[n, k_blocks, blob], &packed);
                let scales_t = Owned::f32(&[n, k_blocks], &scales);
                let zp_owned = zps
                    .as_ref()
                    .map(|z| Owned::u8(&[n, k_blocks.div_ceil(2)], z));

                for &accuracy_level in &[0i64, 4] {
                    let comp = if accuracy_level == 4 {
                        mlas_sys::SQNBitComputeType::Int8
                    } else {
                        mlas_sys::SQNBitComputeType::Fp32
                    };
                    let kernel = MatMulNBitsKernel {
                        accuracy_level,
                        ..test_kernel(k, n, block_size)
                    };
                    let zp_view = zp_owned.as_ref().map(|z| z.view());
                    let Some(packed_weight) = kernel
                        .build_mlas_packed(&b.view(), &scales_t.view(), zp_view.as_ref(), comp)
                        .unwrap()
                    else {
                        eprintln!(
                            "MLAS SQNBit int4 blk={block_size} {comp:?} unavailable; skipping"
                        );
                        continue;
                    };
                    for &m in &[1usize, 5] {
                        let a = pseudo(m * k, 0.8);
                        for bias in [None, Some(pseudo(n, 0.1))] {
                            let mut out = vec![0.0f32; m * n];
                            mlas_sys::sqnbit_gemm(
                                &packed_weight,
                                m,
                                &a,
                                bias.as_deref(),
                                &mut out,
                                true,
                            );
                            let mut expected = reference(&a, &ref_weights, m, k, n);
                            if let Some(bias) = &bias {
                                for row in expected.chunks_exact_mut(n) {
                                    for (v, b) in row.iter_mut().zip(bias) {
                                        *v += b;
                                    }
                                }
                            }
                            // CompInt8 quantizes A to int8, so it needs a looser
                            // tolerance than the near-exact CompFp32 dequant.
                            let tol = if accuracy_level == 4 { 6e-2 } else { 2e-3 };
                            mlas_close(
                                &out,
                                &expected,
                                tol,
                                &format!(
                                    "blk{block_size} asym{asymmetric} acc{accuracy_level} m{m} bias{}",
                                    bias.is_some()
                                ),
                            );
                        }
                    }
                }
            }
        }
    }

    /// `try_mlas_sqnbit` must fall back (return `Ok(None)`) for cases MLAS
    /// SQNBit cannot serve: `g_idx` present (no per-row group indices) and
    /// `bits == 2` (left to the correctness path). These guards short-circuit
    /// ahead of backend detection, so the decision is deterministic.
    #[cfg(feature = "mlas")]
    #[test]
    fn matmulnbits_try_mlas_falls_back_for_gidx_and_bits2() {
        let (n, k, block_size) = (2usize, 32usize, 32usize);
        let k_blocks = k.div_ceil(block_size);
        let a = vec![0.5f32; k];
        let mut result = vec![0.0f32; n];

        // int4 with g_idx present → fall back.
        let kernel = test_kernel(k, n, block_size);
        let b = Owned::u8(
            &[n, k_blocks, block_size / 2],
            &vec![0x88; n * k_blocks * block_size / 2],
        );
        let scales = Owned::f32(&[n, k_blocks], &vec![1.0; n * k_blocks]);
        let g_idx: Vec<i32> = (0..k).map(|i| (i / block_size) as i32).collect();
        let g_idx = Owned::i32(&[k], &g_idx);
        assert_eq!(
            kernel
                .try_mlas_sqnbit(
                    &b.view(),
                    &scales.view(),
                    None,
                    Some(&g_idx.view()),
                    false,
                    &a,
                    1,
                    None,
                    &mut result,
                )
                .unwrap(),
            None,
            "g_idx present must fall back",
        );

        // bits == 2 → fall back.
        let blob2 = block_size / 4;
        let kernel2 = MatMulNBitsKernel {
            bits: 2,
            ..test_kernel(k, n, block_size)
        };
        let b2 = Owned::u8(&[n, k_blocks, blob2], &vec![0x55; n * k_blocks * blob2]);
        assert_eq!(
            kernel2
                .try_mlas_sqnbit(
                    &b2.view(),
                    &scales.view(),
                    None,
                    None,
                    false,
                    &a,
                    1,
                    None,
                    &mut result,
                )
                .unwrap(),
            None,
            "bits==2 must fall back",
        );
    }

    /// The SQNBit prefill threshold parses `NXRT_SQNBIT_PREFILL_MIN`, falling
    /// back to the measured default for absent, empty, or malformed values.
    #[cfg(feature = "mlas")]
    #[test]
    fn matmulnbits_resolve_prefill_min_parses_or_defaults() {
        assert_eq!(resolve_prefill_min(None), DEFAULT_SQNBIT_PREFILL_MIN);
        assert_eq!(resolve_prefill_min(Some("")), DEFAULT_SQNBIT_PREFILL_MIN);
        assert_eq!(resolve_prefill_min(Some("abc")), DEFAULT_SQNBIT_PREFILL_MIN);
        assert_eq!(resolve_prefill_min(Some("32")), 32);
        assert_eq!(resolve_prefill_min(Some("  8 ")), 8);
        assert_eq!(resolve_prefill_min(Some("1")), 1);
    }

    /// Serialize the few tests that mutate `NXRT_CPU_GEMM_BACKEND` so the global
    /// backend override does not race concurrent test threads.
    #[cfg(feature = "mlas")]
    fn backend_env_lock() -> &'static std::sync::Mutex<()> {
        static LOCK: OnceLock<std::sync::Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
    }

    /// M-based hybrid routing gate: with an otherwise-eligible int4 case and the
    /// MLAS backend selected, `try_mlas_sqnbit` must still fall back
    /// (`Ok(None)`) for `m` below the prefill threshold (decode keeps the hand
    /// path) and serve MLAS (`Ok(Some(()))`) for `m` at/above it (prefill). This
    /// regression-locks the decode/prefill split. Uses the default threshold
    /// ([`DEFAULT_SQNBIT_PREFILL_MIN`]); the host must have an MLAS SQNBit int4
    /// kernel or the assertions are skipped.
    #[cfg(feature = "mlas")]
    #[test]
    fn matmulnbits_try_mlas_gates_decode_by_m_threshold() {
        let (n, k, block_size) = (32usize, 64usize, 32usize);
        let k_blocks = k.div_ceil(block_size);
        let blob = block_size / 2;
        let weights_nk = pseudo(n * k, 0.3);
        let (packed_bytes, scales, _zps, _dq) = quantize(&weights_nk, n, k, block_size, false);

        // Skip when the host has no MLAS SQNBit int4 kernel for this shape.
        if mlas_sys::SQNBitPackedB::new(
            n,
            k,
            4,
            block_size,
            mlas_sys::SQNBitComputeType::Int8,
            &packed_bytes,
            &scales,
            None,
        )
        .is_none()
        {
            eprintln!("MLAS SQNBit int4 kernel unavailable; skipping M-gate test");
            return;
        }

        let kernel = accuracy4_kernel(k, n, block_size);
        let b = Owned::u8(&[n, k_blocks, blob], &packed_bytes);
        let scales_t = Owned::f32(&[n, k_blocks], &scales);

        let below = DEFAULT_SQNBIT_PREFILL_MIN - 1;
        let at = DEFAULT_SQNBIT_PREFILL_MIN;

        let _guard = backend_env_lock().lock().unwrap();
        let previous = std::env::var("NXRT_CPU_GEMM_BACKEND").ok();
        // SAFETY: the backend env lock serializes readers/writers of this var.
        unsafe { std::env::set_var("NXRT_CPU_GEMM_BACKEND", "mlas") };

        let call = |m: usize| {
            let a = pseudo(m * k, 0.8);
            let mut result = vec![0.0f32; m * n];
            kernel
                .try_mlas_sqnbit(
                    &b.view(),
                    &scales_t.view(),
                    None,
                    None,
                    false,
                    &a,
                    m,
                    None,
                    &mut result,
                )
                .unwrap()
        };

        let decode = call(below);
        let prefill = call(at);

        // SAFETY: still holding the backend env lock; restore prior value.
        unsafe {
            match &previous {
                Some(value) => std::env::set_var("NXRT_CPU_GEMM_BACKEND", value),
                None => std::env::remove_var("NXRT_CPU_GEMM_BACKEND"),
            }
        }

        assert_eq!(
            decode, None,
            "m={below} (< {DEFAULT_SQNBIT_PREFILL_MIN}) must fall back to the hand int4 path",
        );
        assert_eq!(
            prefill,
            Some(()),
            "m={at} (>= {DEFAULT_SQNBIT_PREFILL_MIN}) must route to MLAS SQNBit",
        );
    }

    /// Before/after perf for int4 MatMulNBits: the existing hand-written VNNI
    /// path (`int4_matmul_m1` for M=1 decode, `int8_matmul` for M>1 prefill,
    /// both `accuracy_level=4`) vs the MLAS SQNBit CompInt8 path, at 1 and 8
    /// threads, for representative LLM shapes. Ignored by default; run with:
    ///   cargo test -p onnx-runtime-ep-cpu --features mlas --release \
    ///     matmulnbits_mlas_perf -- --ignored --nocapture
    #[cfg(feature = "mlas")]
    #[test]
    #[ignore = "perf probe; run explicitly with --ignored --nocapture"]
    fn matmulnbits_mlas_perf() {
        use std::time::Instant;

        fn time<F: FnMut() + Send>(threads: usize, mut run: F) -> f64 {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                for _ in 0..20 {
                    run();
                }
                let iters = 200u32;
                let start = Instant::now();
                for _ in 0..iters {
                    run();
                }
                start.elapsed().as_secs_f64() * 1e6 / iters as f64
            })
        }

        let block_size = 32usize;
        let dot_kernel = selected_dot_kernel();
        for &(k, n) in &[(2048usize, 2048usize), (4096, 11008)] {
            let k_blocks = k.div_ceil(block_size);
            let blob = block_size / 2;
            let weights_nk = pseudo(n * k, 0.3);
            let (packed_bytes, scales, _zps, _dq) = quantize(&weights_nk, n, k, block_size, false);

            let kernel = accuracy4_kernel(k, n, block_size);
            let b = Owned::u8(&[n, k_blocks, blob], &packed_bytes);
            let scales_t = Owned::f32(&[n, k_blocks], &scales);
            let int8_weight = kernel
                .prepack_int8_weight(&b.view(), &scales_t.view(), None)
                .unwrap();
            let int4_weight = PackedInt4Weight {
                values: packed_bytes.clone(),
                scales: scales.clone(),
            };
            let mlas_packed = mlas_sys::SQNBitPackedB::new(
                n,
                k,
                4,
                block_size,
                mlas_sys::SQNBitComputeType::Int8,
                &packed_bytes,
                &scales,
                None,
            )
            .expect("MLAS SQNBit int4 must be available for the perf probe");

            for &m in &[1usize, 32] {
                let a = pseudo(m * k, 0.8);
                for threads in [1usize, 8] {
                    let hand_us = if m == 1 {
                        time(threads, || {
                            let mut out = vec![0.0f32; n];
                            int4_matmul_m1(&a, &int4_weight, &mut out, k, n, dot_kernel);
                        })
                    } else {
                        time(threads, || {
                            let mut out = vec![0.0f32; m * n];
                            int8_matmul(
                                &a,
                                &int8_weight,
                                &mut out,
                                m,
                                k,
                                n,
                                block_size,
                                dot_kernel,
                            );
                        })
                    };
                    let mlas_us = time(threads, || {
                        let mut out = vec![0.0f32; m * n];
                        mlas_sys::sqnbit_gemm(&mlas_packed, m, &a, None, &mut out, true);
                    });
                    eprintln!(
                        "int4 K={k} N={n} M={m} {threads}t: hand={hand_us:.1}us mlas={mlas_us:.1}us \
                         speedup={:.2}x",
                        hand_us / mlas_us
                    );
                }
            }
        }
    }
}

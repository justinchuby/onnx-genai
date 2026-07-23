//! Correctness-first `com.microsoft::MatMulNBits` for f32/f16/bf16 activations
//! and block-quantized 2-bit, 4-bit, or 8-bit weights.
//!
//! ORT stores `B` as
//! `[N, ceil(K / block_size), block_size * bits / 8]`, least-significant bits
//! first within each byte. For M=1 decode, constant quantized weights are
//! prepacked once and reused by a N-parallel GEMV. For symmetric block-32
//! int4 M=1, `accuracy_level=4` streams the packed weights directly into a VNNI
//! dot product. Other int4 accuracy-level-4 shapes keep the weights in int8 and
//! quantize each activation row to int8. The 2-bit correctness path and default
//! int4 path dequantize to f32; batched shapes then use the shared CPU GEMM,
//! including its SIMD backend. The 8-bit correctness path uses the same affine
//! dequantization with one uint8 weight and optional uint8 zero point per block.

use std::cell::Cell;
use std::sync::OnceLock;

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};
use rayon::prelude::*;

use super::matmul::gemm;
use super::{check_arity, to_dense_bytes, to_dense_f32, to_dense_i64, write_dense_f32};
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};
use crate::strided::numel;

/// Overrides the bounded M=1 decode pool size; set to `0` to use the global
/// Rayon pool as an escape hatch.
const DECODE_THREADS_ENV: &str = "ONNX_GENAI_CPU_DECODE_THREADS";
/// Decode is bandwidth-bound and pays one fork/join per projection. Profiling
/// across the existing 4--96 worker sweep found no gain above eight workers and
/// clear regressions at 16+, so topology scaling is capped here; the environment
/// override remains available for processors whose measurements differ.
const MAX_TOPOLOGY_DECODE_THREADS: usize = 8;
static DECODE_POOL: OnceLock<std::result::Result<Option<rayon::ThreadPool>, String>> =
    OnceLock::new();

/// Env knob for the int4 MatMulNBits hand-decode ↔ MLAS SQNBit crossover (`m`
/// row count). MatMulNBits int4 with `m < NXRT_SQNBIT_DECODE_MIN` uses the
/// specialized hand-written int4/int8 decode path (`int4_matmul_m1` for block-32
/// symmetric M=1, `int8_matmul` otherwise), which ties MLAS on the
/// bandwidth-bound M=1 decode while avoiding int8 activation rounding; `m` at or
/// above the threshold routes to MLAS `MlasQNBitGemmBatch`, whose cache-tiled
/// kernels win prefill by 6--9x.
#[cfg(feature = "mlas")]
const SQNBIT_DECODE_MIN_ENV: &str = "NXRT_SQNBIT_DECODE_MIN";

/// Basis for the topology-derived int4 MatMulNBits hand-decode ↔ MLAS crossover.
/// Measurements on Sapphire Rapids (Xeon 8480C) found:
///
/// * Isolated GEMV microbench (`matmulnbits_mlas_perf`, weights L3-resident)
///   reports MLAS int4 M=1 ~1.7--1.9x faster, but that is a cache artifact.
/// * Cold, DRAM-streamed full-decode-step microbench
///   (`matmulnbits_mlas_decode_step`, one distinct 3.5 GB weight set per token,
///   32 threads) has the hand path and MLAS CompInt8 **tie** at ~90 GB/s
///   (~25 tok/s) for M=1 -- decode is memory-bandwidth bound, so the int4 path
///   choice is a wash and the hand path is preferred (no int8 rounding).
/// * End-to-end Qwen2.5-Coder-7B decode is the same (~8 tok/s) with either M=1
///   route; the 2.3x gap vs ORT/foundry is per-op Rayon fork-join and NUMA
///   locality, not the MatMulNBits kernel (see docs/BENCH_MLAS_INT4_E2E.md).
///
/// The default crossover is twice the topology-derived decode worker count.
/// That preserves `m=16` on the profiled 96-way host while scaling down on
/// smaller machines where MLAS needs fewer rows to occupy the available cores.
/// Override with `NXRT_SQNBIT_DECODE_MIN`.
#[cfg(feature = "mlas")]
static SQNBIT_DECODE_MIN: OnceLock<usize> = OnceLock::new();

/// Smallest `m` (batch·seq row count) that routes int4 MatMulNBits to MLAS
/// SQNBit; smaller `m` uses the hand int4/int8 decode path. Parsed once from
/// `NXRT_SQNBIT_DECODE_MIN`, defaulting to [`default_sqnbit_decode_min`].
#[cfg(feature = "mlas")]
fn sqnbit_decode_min() -> usize {
    *SQNBIT_DECODE_MIN.get_or_init(|| {
        let available = available_parallelism();
        resolve_decode_min(
            std::env::var(SQNBIT_DECODE_MIN_ENV).ok().as_deref(),
            available,
        )
    })
}

/// Parse the SQNBit decode crossover, falling back to
/// [`default_sqnbit_decode_min`] for absent, empty, or malformed values.
#[cfg(feature = "mlas")]
fn resolve_decode_min(raw: Option<&str>, available: usize) -> usize {
    raw.and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or_else(|| default_sqnbit_decode_min(available))
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
        if !matches!(bits, 2 | 4 | 8) {
            return Err(error(format!(
                "MatMulNBits CPU supports bits in {{2, 4, 8}}, got bits={bits}. Why: other packed \
                 widths do not have a validated dequantization path. How to fix: export bits=2, \
                 bits=4, or bits=8, or select another execution provider"
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
        require_float_compute_dtype("A", inputs[0].dtype)?;
        require_dtype("B", inputs[1].dtype, DataType::Uint8)?;
        require_float_compute_dtype("scales", inputs[2].dtype)?;
        require_float_compute_dtype("Y", outputs[0].dtype)?;

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
            require_float_compute_dtype("bias", bias.dtype)?;
            require_shape("bias", bias.shape, &[self.n])?;
            Some(to_dense_compute_f32(bias)?)
        } else {
            None
        };

        let can_prepack = self.constant_inputs[1]
            && self.constant_inputs[2]
            && zero_points.is_none_or(|_| self.constant_inputs[3])
            && group_indices.is_none_or(|_| self.constant_inputs[4]);
        let activations = to_dense_compute_f32(&inputs[0])?;
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
                return write_compute_f32(&mut outputs[0], &result);
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
                        scales: to_dense_compute_f32(&inputs[2])?,
                    };
                    let weight = numa_place_int4(weight, self.n);
                    let _ = self.packed_int4_weight.set(weight);
                    self.packed_int4_weight
                        .get()
                        .expect("constant MatMulNBits packed int4 weight was just initialized")
                }
            } else {
                let built = PackedInt4Weight {
                    values: to_dense_bytes(&inputs[1])?,
                    scales: to_dense_compute_f32(&inputs[2])?,
                };
                owned_weight = numa_place_int4(built, self.n);
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
                    let weight = numa_place_int8(weight, self.n);
                    let _ = self.int8_weight.set(weight);
                    self.int8_weight
                        .get()
                        .expect("constant MatMulNBits int8 prepack was just initialized")
                }
            } else {
                let built = self.prepack_int8_weight(&inputs[1], &inputs[2], zero_points)?;
                owned_weight = numa_place_int8(built, self.n);
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
                    let weight = numa_place_nk(weight, self.n);
                    let _ = self.weight_nk.set(weight);
                    self.weight_nk
                        .get()
                        .expect("constant MatMulNBits prepack was just initialized")
                }
            } else {
                let built = self.dequantize_weight(
                    &inputs[1],
                    &inputs[2],
                    zero_points,
                    group_indices,
                    WeightLayout::Nk,
                )?;
                owned_weight = numa_place_nk(built, self.n);
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
        write_compute_f32(&mut outputs[0], &result)
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
    /// Fallback cases (return `Ok(None)`): the decode regime (`m` below the
    /// crossover [`sqnbit_decode_min`]) when the hand path is a *fast*
    /// specialized int4/int8 route (`bits == 4 && accuracy_level == 4`), which
    /// ties MLAS on bandwidth-bound M=1 while avoiding int8 activation rounding;
    /// `accuracy_level == 4` when the resolved [`CpuBackend`] is not MLAS (its
    /// hand int8 path owns MatMulNBits and matches ORT's CompInt8 numerics);
    /// `bits != 4` (2-bit is left to the existing correctness path); `g_idx` is
    /// present (MLAS SQNBit has no per-row group indices); or MLAS reports no
    /// kernel is available for this shape on the host. A case whose hand path
    /// would instead fall to the slow full-f32-dequant GEMV (any
    /// `accuracy_level != 4`, e.g. the `accuracy_level = 0` "implementation's
    /// choice" that Foundry `cuda-gpu` int4 exports emit) is **not** dropped
    /// here: MLAS SQNBit (CompFp32) beats a dequantize-then-GEMM there, matching
    /// how ORT/onnxruntime-genai run those models. Bias, when present, is added
    /// by MLAS itself, so the caller's post-loop bias add is skipped on this
    /// path.
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

        // Cheapest gate first: in the decode regime (small `m`), keep the fast
        // hand int4/int8 path -- it ties MLAS on the bandwidth-bound M=1 GEMV
        // and avoids int8 activation rounding -- so fall back before any weight
        // packing. A slow-hand-path case (`accuracy_level != 4`, which would
        // dequantize to f32 and run a dense GEMV) is left for MLAS below.
        let hand_decode_is_fast = self.bits == 4 && self.accuracy_level == 4;
        if m < sqnbit_decode_min() && hand_decode_is_fast {
            return Ok(None);
        }

        // MLAS SQNBit is a specialized blockwise-quantized kernel, distinct from
        // the dense-f32 GEMM microkernel that `CpuBackend` selects. For
        // `accuracy_level == 4` the fast hand int8/int4 paths own MatMulNBits and
        // match ORT's CompInt8 numerics, so only defer to MLAS (CompInt8) when the
        // whole GEMM backend was explicitly forced to MLAS. For every other
        // accuracy level the hand fallback is a slow full-f32-dequant GEMV, so
        // prefer MLAS SQNBit (CompFp32) whenever MLAS actually has a kernel --
        // this matches ORT/onnxruntime-genai, which treat `accuracy_level` 0/1 as
        // CompFp32 rather than dequantizing the whole weight.
        let backend_is_mlas = CpuBackend::auto_detect() == CpuBackend::Mlas;
        let use_mlas = backend_is_mlas || self.accuracy_level != 4;
        if self.bits != 4 || group_indices.is_some() || !use_mlas {
            return Ok(None);
        }

        let comp = if self.accuracy_level == 4 {
            mlas_sys::SQNBitComputeType::Int8
        } else {
            // accuracy_level is a *minimum* compute-precision hint: 0 = kernel's
            // choice, 1 = fp32, 2 = fp16, 3 = bf16, 4 = int8. We route every
            // non-int8 level to MLAS SQNBit CompFp32, i.e. the fp16/bf16 levels
            // (2/3) are deliberately upgraded to fp32 compute -- more accuracy
            // than requested, never less, and a conservative, bandwidth-bound
            // choice that MLAS actually implements (it has no fp16/bf16 SQNBit
            // path here). This matches ORT/onnxruntime-genai, which treat 0/1 as
            // CompFp32.
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
        let scales = to_dense_compute_f32(scales)?;
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
        let scales = to_dense_compute_f32(scales)?;
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
        let scales = to_dense_compute_f32(scales)?;
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
        let quantized_mask = if self.bits == 8 {
            u8::MAX
        } else {
            (1u8 << self.bits) - 1
        };
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
    let available = available_parallelism();
    resolve_decode_threads(value.as_deref(), available)
}

/// The worker count for the persistent SPMD decode pool ([`crate::decode_spmd`]).
///
/// It honors `ONNX_GENAI_CPU_DECODE_THREADS` when set (`0` opts out), but when
/// the variable is unset it uses a *different, higher* default than the flat
/// pool: [`default_persistent_threads`] (about half the logical CPUs) instead of
/// the flat pool's eight-worker ceiling. The flat Rayon pool caps at eight
/// because its per-op fork/join regresses beyond that; the persistent pool
/// replaces that fork/join with one hot broadcast barrier, so it keeps scaling
/// with cores until it hits the memory-bandwidth knee (measured plateau ~half
/// the logical CPUs on a 2-socket Xeon 8480C). Sizing it from the flat default
/// would leave the out-of-box path at the flat pool's throughput and defeat the
/// point of making it the default.
pub fn configured_persistent_decode_threads() -> Option<usize> {
    let value = std::env::var(DECODE_THREADS_ENV).ok();
    let available = available_parallelism();
    resolve_persistent_decode_threads(value.as_deref(), available)
}

/// Default persistent-pool worker count for `available` logical CPUs: half of
/// them (at least one), derived purely from topology (Rule 2).
///
/// Half leaves a full set of hardware threads free for the dispatcher (which
/// runs the forward inline and spins on the completion counters), the prefill
/// global Rayon pool, and co-tenants on a shared box. Because the SPMD workers
/// *spin* before parking, a fully-subscribed pool starves the dispatcher and
/// collapses throughput (measured 1.4 tok/s at 96 workers vs 28.7 at 48 on a
/// 96-logical-CPU host); half sits at the measured plateau while avoiding that
/// cliff, and on SMT hosts it maps to roughly the physical-core count.
fn default_persistent_threads(available: usize) -> Option<usize> {
    let available = std::num::NonZeroUsize::new(available)?.get();
    Some((available / 2).max(1))
}

/// Resolve the persistent-pool worker count from the raw `ONNX_GENAI_CPU_DECODE_THREADS`
/// value and the host's logical CPU count. `Some("0")` opts out (`None`); an
/// explicit positive count is honored (clamped to `available`); an unset or
/// unparseable value falls back to [`default_persistent_threads`].
fn resolve_persistent_decode_threads(raw: Option<&str>, available: usize) -> Option<usize> {
    let available = std::num::NonZeroUsize::new(available)?.get();
    let default = default_persistent_threads(available)?;
    let threads = match raw {
        Some("0") => return None,
        Some(raw) => raw
            .parse::<usize>()
            .ok()
            .filter(|threads| *threads > 0)
            .unwrap_or(default),
        None => default,
    };
    Some(threads.min(available))
}

fn available_parallelism() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
}

/// The host's logical CPU count, exposed so the persistent SPMD pool can apply
/// its safe auto-enable gate ([`crate::decode_spmd`]) without duplicating the
/// `available_parallelism` fallback logic.
pub fn available_parallelism_public() -> usize {
    available_parallelism()
}

/// Choose a bounded decode pool from the host's logical CPU count.
///
/// Decode projections are small and bandwidth-bound, so worker demand grows
/// much more slowly than core count: `1 + ceil(log2(logical_cpus))` gives 3
/// workers on 4-way hosts, 4 on 8-way hosts, and the profiled 8 workers on the
/// 96-way Xeon. The measured eight-worker ceiling limits fork/join overhead, and
/// the result never exceeds the CPUs available to a cgroup.
fn default_decode_threads(available: usize) -> Option<usize> {
    let available = std::num::NonZeroUsize::new(available)?.get();
    let ceil_log2 = usize::BITS as usize - (available - 1).leading_zeros() as usize;
    Some((ceil_log2 + 1).min(MAX_TOPOLOGY_DECODE_THREADS).min(available))
}

fn resolve_decode_threads(raw: Option<&str>, available: usize) -> Option<usize> {
    let available = std::num::NonZeroUsize::new(available)?.get();
    let default = default_decode_threads(available)?;
    let threads = match raw {
        Some("0") => return None,
        Some(raw) => raw.parse::<usize>().unwrap_or(default),
        None => default,
    };
    (threads > 0).then(|| threads.min(available))
}

#[cfg(feature = "mlas")]
fn default_sqnbit_decode_min(available: usize) -> usize {
    default_decode_threads(available)
        .unwrap_or(1)
        .saturating_mul(2)
}

fn build_decode_pool(
    threads: Option<usize>,
) -> std::result::Result<Option<rayon::ThreadPool>, String> {
    threads
        .map(|threads| {
            let affinity_cpus = decode_affinity_cpus(threads)?;
            let mut builder = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .thread_name(|index| format!("onnx-genai-decode-{index}"));
            if let Some(cpus) = affinity_cpus {
                builder = builder.start_handler(move |worker_index| {
                    // Pin each worker to a distinct CPU of the selected NUMA node
                    // so the per-op fork-join barrier and the streamed weights
                    // stay node-local. Best-effort: a restricted cgroup that
                    // rejects the request is logged once, not fatal, so decode
                    // still runs (unpinned) rather than failing outright.
                    let cpu = cpus[worker_index % cpus.len()];
                    if let Err(message) =
                        crate::decode_affinity::pin_current_thread_to_cpu(cpu)
                    {
                        report_decode_affinity_failure(&message);
                    }
                });
            }
            builder
                .build()
                .map_err(|err| format!("failed to build {DECODE_THREADS_ENV} pool: {err}"))
        })
        .transpose()
}

/// Resolve the CPU set the decode pool should pin `threads` workers to, honoring
/// the explicit [`crate::decode_affinity::DECODE_AFFINITY_ENV`] switch, the
/// auto-enable policy, and the process's allowed CPU set (cpuset/taskset). The
/// chosen auto-policy is logged once at info. Returns `Ok(None)` when pinning is
/// off, unsupported, or declined; propagates malformed configuration as a clear
/// error.
fn decode_affinity_cpus(threads: usize) -> std::result::Result<Option<Vec<usize>>, String> {
    let plan = crate::decode_affinity::plan_decode_affinity(threads)?;
    if let Some(message) = plan.log {
        report_decode_affinity_policy(&message);
    }
    Ok(plan.cpus)
}

/// Log the decode-affinity auto-policy decision once (info): whether pinning was
/// auto-enabled, declined (cpuset/single-node/unsupported OS), and why.
fn report_decode_affinity_policy(message: &str) {
    static REPORTED: OnceLock<()> = OnceLock::new();
    if REPORTED.set(()).is_ok() {
        eprintln!("onnx-genai: decode affinity policy: {message}");
    }
}

/// Log the first decode-affinity pinning failure once so a restricted
/// environment surfaces the reason without spamming every worker.
fn report_decode_affinity_failure(message: &str) {
    static REPORTED: OnceLock<()> = OnceLock::new();
    if REPORTED.set(()).is_ok() {
        eprintln!(
            "onnx-genai: decode-pool CPU affinity unavailable; \
             continuing without pinning ({message})"
        );
    }
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

    /// Per-thread flag marking that the current thread is running the forward
    /// pass inside a `numa-split` [`with_decode_pool_scope`] installation. Set on
    /// the dispatcher worker that runs the forward so each M=1 projection fans
    /// its output rows out across the per-node sub-pools (see
    /// [`parallel_output_rows`] and [`crate::decode_numa`]).
    static IN_NUMA_SCOPE: Cell<bool> = const { Cell::new(false) };

    /// Per-thread flag marking that the current thread is running the forward
    /// pass inside a persistent SPMD-pool ([`crate::decode_spmd`])
    /// [`with_decode_pool_scope`] installation, so each M=1 projection fans its
    /// output rows out across the persistent worker set instead of a per-op
    /// Rayon region.
    static IN_SPMD_SCOPE: Cell<bool> = const { Cell::new(false) };
}

/// The lazily built `numa-split` decode layout, or `None` when the mode is not
/// requested or the host cannot be split (fallback, logged once).
fn numa_pools() -> Option<&'static crate::decode_numa::NumaDecodePools> {
    static NUMA_POOLS: OnceLock<Option<crate::decode_numa::NumaDecodePools>> = OnceLock::new();
    NUMA_POOLS
        .get_or_init(|| crate::decode_numa::build_from_env(configured_decode_threads()))
        .as_ref()
}

/// The active `numa-split` layout when the current thread is running a
/// `numa-split` decode forward; `None` otherwise (so prefill, non-decode work,
/// and the flat single-node modes keep their existing behaviour).
fn numa_decode_active() -> Option<&'static crate::decode_numa::NumaDecodePools> {
    if IN_NUMA_SCOPE.with(Cell::get) {
        numa_pools()
    } else {
        None
    }
}

/// The active persistent SPMD layout when the current thread is running a
/// persistent-pool decode forward; `None` otherwise.
fn spmd_decode_active() -> Option<&'static crate::decode_spmd::SpmdDecodePools> {
    if IN_SPMD_SCOPE.with(Cell::get) {
        crate::decode_spmd::pools()
    } else {
        None
    }
}

#[cfg(test)]
static SPMD_TEST_DISPATCHES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Fan a projection's output rows out across the decode workers.
///
/// With `numa-split` active, the rows are sharded across the per-node sub-pools
/// (node-local weights, single cross-node join). Otherwise the flat single-node
/// pool chunks them as before. `compute(output_start, outputs)` fills the rows
/// `output_start .. output_start + outputs.len()`, so the math is identical
/// regardless of how the rows are partitioned (row-sharding a GEMV is exactly
/// associative -- no cross-row reduction -- so results are bit-identical).
fn parallel_output_rows<F>(result: &mut [f32], k: usize, compute: F)
where
    F: Fn(usize, &mut [f32]) + Sync,
{
    if let Some(numa) = numa_decode_active() {
        numa.dispatch_output_rows(result, k, &compute);
        return;
    }
    if let Some(spmd) = spmd_decode_active() {
        #[cfg(test)]
        SPMD_TEST_DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        spmd.dispatch_output_rows(result, k, &compute);
        return;
    }
    let chunk = output_chunk_len(result.len(), k);
    if chunk < result.len() {
        result
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(chunk_index, outputs)| compute(chunk_index * chunk, outputs));
    } else {
        compute(0, result);
    }
}

/// Fan `num_rows` fixed-width output rows (each `row_len` elements of `result`)
/// out across the active decode workers, running `compute(row_index, row_slice)`
/// on each whole row.
///
/// This is the row-block analogue of [`parallel_output_rows`] for decode kernels
/// (e.g. `GroupQueryAttention`) whose parallel unit is a full contiguous row
/// rather than a GEMV scalar output. It exists so those kernels use the *same*
/// decode pool as the `MatMulNBits` projections instead of a second thread pool:
///
/// * When a persistent SPMD decode scope is active the forward runs on the
///   engine thread (not a Rayon worker), so a bare `par_chunks_mut` here would
///   fall to the *global* Rayon pool and contend with the SPMD pool's resident,
///   pinned, spinning workers. Routing through the SPMD pool removes that
///   contention (measured to dominate 7B CPU decode).
/// * The `numa-split` and flat decode scopes install the forward onto a bounded
///   Rayon pool, so `par_chunks_mut` already runs on that pool (no global-pool
///   contention); they keep the existing behaviour.
///
/// Each row is independent, so sharding it across workers reproduces the
/// single-threaded result bit-for-bit. Generality: the routing keys off which
/// decode scope is active, never off op or model identity (RULES.md §2).
pub fn decode_parallel_output_row_blocks<F>(
    result: &mut [f32],
    row_len: usize,
    num_rows: usize,
    compute: F,
) where
    F: Fn(usize, &mut [f32]) + Sync,
{
    if let Some(spmd) = spmd_decode_active() {
        spmd.dispatch_output_row_blocks(result, row_len, num_rows, &compute);
        return;
    }
    // numa-split and flat decode scopes run the forward on a bounded Rayon pool,
    // so this `par_chunks_mut` uses that pool rather than the global one.
    result
        .par_chunks_mut(row_len)
        .enumerate()
        .for_each(|(row_index, row)| compute(row_index, row));
}

/// First-touch each row-major weight component on the NUMA node that will read
/// it under `numa-split` or the persistent SPMD pool, so each node's workers
/// stream node-local memory. A no-op (returns the input) when neither node-aware
/// decode mode is active.
fn numa_place_int4(weight: PackedInt4Weight, n: usize) -> PackedInt4Weight {
    if let Some(numa) = numa_decode_active() {
        return PackedInt4Weight {
            values: numa.place_rows(&weight.values, n),
            scales: numa.place_rows(&weight.scales, n),
        };
    }
    if let Some(spmd) = spmd_decode_active() {
        return PackedInt4Weight {
            values: spmd.place_rows(&weight.values, n),
            scales: spmd.place_rows(&weight.scales, n),
        };
    }
    weight
}

/// Node-local first-touch for the prepacked int8 weight (see [`numa_place_int4`]).
fn numa_place_int8(weight: Int8Weight, n: usize) -> Int8Weight {
    if let Some(numa) = numa_decode_active() {
        return Int8Weight {
            values: numa.place_rows(&weight.values, n),
            scales: numa.place_rows(&weight.scales, n),
            block_sums: numa.place_rows(&weight.block_sums, n),
        };
    }
    if let Some(spmd) = spmd_decode_active() {
        return Int8Weight {
            values: spmd.place_rows(&weight.values, n),
            scales: spmd.place_rows(&weight.scales, n),
            block_sums: spmd.place_rows(&weight.block_sums, n),
        };
    }
    weight
}

/// Node-local first-touch for the dequantized `[N, K]` weight (see
/// [`numa_place_int4`]).
fn numa_place_nk(weight: Vec<f32>, n: usize) -> Vec<f32> {
    if let Some(numa) = numa_decode_active() {
        return numa.place_rows(&weight, n);
    }
    if let Some(spmd) = spmd_decode_active() {
        return spmd.place_rows(&weight, n);
    }
    weight
}

/// RAII guard that marks the current thread as running a `numa-split` decode
/// forward and restores the previous state on drop (including on panic).
struct NumaScopeGuard {
    previous: bool,
}

impl NumaScopeGuard {
    fn enter() -> Self {
        let previous = IN_NUMA_SCOPE.with(|flag| flag.replace(true));
        Self { previous }
    }
}

impl Drop for NumaScopeGuard {
    fn drop(&mut self) {
        let previous = self.previous;
        IN_NUMA_SCOPE.with(|flag| flag.set(previous));
    }
}

/// RAII guard that marks the current thread as running a persistent SPMD-pool
/// decode forward and restores the previous state on drop (including on panic).
struct SpmdScopeGuard {
    previous: bool,
}

impl SpmdScopeGuard {
    fn enter() -> Self {
        let previous = IN_SPMD_SCOPE.with(|flag| flag.replace(true));
        Self { previous }
    }
}

impl Drop for SpmdScopeGuard {
    fn drop(&mut self) {
        let previous = self.previous;
        IN_SPMD_SCOPE.with(|flag| flag.set(previous));
    }
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
    // The persistent SPMD pool is default-on (opt out with
    // `ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL=0`). Precedence: explicit numa-split
    // env > (Auto default, unless an explicit non-numa-split affinity defers it to
    // the flat path -- see `decode_spmd::auto_defers_to_flat`) persistent SPMD >
    // flat + auto-compact. The "mutually exclusive" diagnostic below is scoped to
    // users who *explicitly* forced the persistent pool (`PERSISTENT_POOL=1`);
    // under the Auto default the user never asked for it, and an explicit
    // non-numa-split affinity already defers the pool anyway, so logging a
    // conflict there would be noise.
    let both_requested = crate::decode_spmd::is_forced()
        && std::env::var(crate::decode_affinity::DECODE_AFFINITY_ENV)
            .is_ok_and(|value| value.trim() == "numa-split");
    // `numa-split`: run the forward on the dispatcher pool and let each M=1
    // projection fan its output rows out across the per-node sub-pools. The
    // decode-residency flag is set too, so the inner `with_decode_pool` calls
    // run inline on the dispatcher worker (they must not re-install the flat
    // single-node pool); the numa-scope flag makes `parallel_output_rows`
    // choose the two-level per-node dispatch.
    if let Some(numa) = numa_pools() {
        if both_requested {
            report_decode_strategy_precedence(
                "ONNX_GENAI_CPU_DECODE_AFFINITY=numa-split and the forced persistent \
                 SPMD decode pool (ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL=1) are mutually \
                 exclusive; numa-split is active because it has precedence and its two-level \
                 NUMA layout was built successfully. Unset \
                 ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL to silence this if intentional",
            );
        }
        return numa.install_scope(move || {
            let _numa_guard = NumaScopeGuard::enter();
            let _decode_guard = DecodeResidencyGuard::enter();
            f()
        });
    }
    // Persistent SPMD pool: run the forward inline on this (dispatcher) thread
    // and let each M=1 projection broadcast its output-row shards to the hot
    // persistent workers under one lightweight barrier. The decode-residency
    // flag makes inner `with_decode_pool` calls run inline (they must not
    // re-install the flat pool); the SPMD-scope flag routes `parallel_output_rows`
    // through the persistent pool.
    if crate::decode_spmd::pools().is_some() {
        if both_requested {
            report_decode_strategy_precedence(
                "ONNX_GENAI_CPU_DECODE_AFFINITY=numa-split and the forced persistent \
                 SPMD decode pool (ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL=1) are mutually \
                 exclusive; persistent SPMD is active because the higher-precedence \
                 numa-split layout was unavailable",
            );
        }
        let _spmd_guard = SpmdScopeGuard::enter();
        let _decode_guard = DecodeResidencyGuard::enter();
        return f();
    }
    if both_requested {
        report_decode_strategy_precedence(
            "ONNX_GENAI_CPU_DECODE_AFFINITY=numa-split and the forced persistent SPMD \
             decode pool (ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL=1) are mutually exclusive; \
             neither strategy is active because no bounded decode worker count or usable \
             numa-split layout is available",
        );
    }
    match DECODE_POOL.get_or_init(|| build_decode_pool(configured_decode_threads())) {
        Ok(Some(pool)) => pool.install(move || {
            let _guard = DecodeResidencyGuard::enter();
            f()
        }),
        _ => f(),
    }
}

fn report_decode_strategy_precedence(message: &str) {
    static REPORTED: OnceLock<()> = OnceLock::new();
    if REPORTED.set(()).is_ok() {
        eprintln!("onnx-genai: decode strategy selection: {message}");
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

    parallel_output_rows(result, padded_k, compute);
}

fn int4_dot_row(
    activation: &[i8],
    packed_weight: &[u8],
    scales: &[f32],
    activation_scale: f32,
    _kernel: DotKernel,
) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        match _kernel {
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
        parallel_output_rows(result, padded_k, compute);
    } else {
        compute(0, result);
    }
}

fn dot_u8_i8(activation: &[u8], weight: &[i8], _kernel: DotKernel) -> i32 {
    debug_assert_eq!(activation.len(), weight.len());
    #[cfg(target_arch = "x86_64")]
    {
        match _kernel {
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
        parallel_output_rows(result, k, compute);
    } else {
        compute(0, result);
    }
}

const MIN_PARALLEL_DOT_PRODUCTS_PER_TASK: usize = 32 * 1024;
const MIN_PARALLEL_DOT_PRODUCTS_PER_THREAD: usize = 8 * 1024;
const MANY_THREAD_DOT_PRODUCTS_PER_THREAD: usize = 64 * 1024;
const MIN_OUTPUTS_PER_TASK: usize = 16;
const MANY_THREAD_CUTOFF: usize = 48;

pub(crate) fn output_chunk_len(n: usize, k: usize) -> usize {
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

fn require_float_compute_dtype(name: &str, got: DataType) -> Result<()> {
    if !matches!(
        got,
        DataType::Float32 | DataType::Float16 | DataType::BFloat16
    ) {
        return Err(error(format!(
            "{name} must have dtype Float32, Float16, or BFloat16, got {got:?}"
        )));
    }
    Ok(())
}

/// Preserve the original f32 materialization path exactly; lower-precision
/// tensors reuse the shared scalar, cross-architecture widening machinery.
fn to_dense_compute_f32(view: &TensorView) -> Result<Vec<f32>> {
    match view.dtype {
        DataType::Float32 => to_dense_f32(view),
        DataType::Float16 | DataType::BFloat16 => {
            Ok(to_dense_f32_widen("MatMulNBits", view)?.into_owned())
        }
        other => Err(error(format!(
            "compute input must have dtype Float32, Float16, or BFloat16, got {other:?}"
        ))),
    }
}

/// Preserve the original f32 writer exactly; f16/bf16 outputs reuse the shared
/// narrowing path, which has portable scalar conversion on every processor.
fn write_compute_f32(out: &mut TensorMut, data: &[f32]) -> Result<()> {
    match out.dtype {
        DataType::Float32 => write_dense_f32(out, data),
        DataType::Float16 | DataType::BFloat16 => write_dense_f32_narrow("MatMulNBits", out, data),
        other => Err(error(format!(
            "Y must have dtype Float32, Float16, or BFloat16, got {other:?}"
        ))),
    }
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

    /// Address of whichever prepack reuse cache the routed path populated, or
    /// `None` if none is populated yet. Which cache is filled depends on the
    /// route: MLAS SQNBit (`mlas_packed`) for `accuracy_level != 4` when the MLAS
    /// kernel is available, otherwise the hand GEMV/int8 caches. Returning a raw
    /// address lets tests assert the cache is *reused* (stable) across calls, not
    /// merely populated. The address is stable because every cache is a
    /// `OnceLock` that stores its value in place.
    fn prepack_cache_ptr(kernel: &MatMulNBitsKernel) -> Option<*const ()> {
        if let Some(w) = kernel.weight_nk.get() {
            return Some(w as *const _ as *const ());
        }
        if let Some(w) = kernel.int8_weight.get() {
            return Some(w as *const _ as *const ());
        }
        if let Some(w) = kernel.packed_int4_weight.get() {
            return Some(w as *const _ as *const ());
        }
        #[cfg(feature = "mlas")]
        if let Some(w) = kernel.mlas_packed.get() {
            return Some(w as *const _ as *const ());
        }
        None
    }

    /// True when a constant `MatMulNBits` weight has been prepacked into any of
    /// the reuse caches (see [`prepack_cache_ptr`]).
    fn prepack_cache_populated(kernel: &MatMulNBitsKernel) -> bool {
        prepack_cache_ptr(kernel).is_some()
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
        assert_eq!(resolve_decode_threads(None, 4), Some(3));
        assert_eq!(resolve_decode_threads(None, 8), Some(4));
        assert_eq!(resolve_decode_threads(None, 1), Some(1));
        assert_eq!(resolve_decode_threads(Some(""), 96), Some(8));
        assert_eq!(resolve_decode_threads(Some("0"), 8), None);
        assert_eq!(resolve_decode_threads(Some("4"), 96), Some(4));
        assert_eq!(resolve_decode_threads(Some("1000"), 96), Some(96));
        assert_eq!(resolve_decode_threads(Some("abc"), 96), Some(8));
        assert_eq!(resolve_decode_threads(Some("-4"), 4), Some(3));
        assert_eq!(resolve_decode_threads(Some("4"), 0), None);
    }

    #[test]
    fn persistent_decode_thread_default_is_half_the_logical_cpus() {
        // The persistent pool scales past the flat pool's 8-worker ceiling: unset
        // -> half the logical CPUs (topology-derived, rule 2), not the flat cap.
        assert_eq!(default_persistent_threads(96), Some(48));
        assert_eq!(default_persistent_threads(8), Some(4));
        assert_eq!(default_persistent_threads(4), Some(2));
        assert_eq!(default_persistent_threads(2), Some(1));
        assert_eq!(default_persistent_threads(1), Some(1));
        assert_eq!(default_persistent_threads(0), None);
        // Distinct from the flat default on a big host (48 vs 8) -- proving the
        // persistent path does not inherit the fork/join-bound cap.
        assert_ne!(default_persistent_threads(96), default_decode_threads(96));
    }

    #[test]
    fn persistent_decode_threads_honor_env_and_opt_out() {
        // Unset -> the persistent default (half cores), not the flat cap.
        assert_eq!(resolve_persistent_decode_threads(None, 96), Some(48));
        assert_eq!(resolve_persistent_decode_threads(Some(""), 96), Some(48));
        // Explicit `0` opts out of the bounded pool (flat legacy path).
        assert_eq!(resolve_persistent_decode_threads(Some("0"), 96), None);
        // An explicit positive count is honored and clamped to the host.
        assert_eq!(resolve_persistent_decode_threads(Some("32"), 96), Some(32));
        assert_eq!(resolve_persistent_decode_threads(Some("1"), 96), Some(1));
        assert_eq!(resolve_persistent_decode_threads(Some("1000"), 96), Some(96));
        // Unparseable/negative values fall back to the persistent default.
        assert_eq!(resolve_persistent_decode_threads(Some("abc"), 96), Some(48));
        assert_eq!(resolve_persistent_decode_threads(Some("-4"), 8), Some(4));
        assert_eq!(resolve_persistent_decode_threads(Some("8"), 0), None);
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
    fn matmulnbits_f16_bf16_inputs_match_widened_f32_for_decode_and_prefill() {
        let (k, n, block_size) = (64usize, 9usize, 32usize);
        let weights: Vec<f32> = (0..n * k)
            .map(|i| ((i * 13 % 31) as f32 - 15.0) / 9.0)
            .collect();
        let (packed, scales, _, _) = quantize(&weights, n, k, block_size, false);
        let bias_values: Vec<f32> = (0..n).map(|i| (i as f32 - 4.0) / 17.0).collect();

        for dtype in [DataType::Float16, DataType::BFloat16] {
            for m in [1usize, 3usize] {
                let a_values: Vec<f32> = (0..m * k)
                    .map(|i| ((i * 17 % 43) as f32 - 21.0) / 13.0)
                    .collect();
                let low_a = match dtype {
                    DataType::Float16 => Owned::f16(&[m, k], &a_values),
                    DataType::BFloat16 => Owned::bf16(&[m, k], &a_values),
                    _ => unreachable!(),
                };
                let low_scales = match dtype {
                    DataType::Float16 => Owned::f16(&[n, 2], &scales),
                    DataType::BFloat16 => Owned::bf16(&[n, 2], &scales),
                    _ => unreachable!(),
                };
                let low_bias = match dtype {
                    DataType::Float16 => Owned::f16(&[n], &bias_values),
                    DataType::BFloat16 => Owned::bf16(&[n], &bias_values),
                    _ => unreachable!(),
                };
                let widened = |owned: &Owned| match dtype {
                    DataType::Float16 => owned.to_f16_as_f32(),
                    DataType::BFloat16 => owned.to_bf16_as_f32(),
                    _ => unreachable!(),
                };
                let f32_a = Owned::f32(&[m, k], &widened(&low_a));
                let f32_scales = Owned::f32(&[n, 2], &widened(&low_scales));
                let f32_bias = Owned::f32(&[n], &widened(&low_bias));
                let b = Owned::u8(&[n, 2, 16], &packed);
                let absent_zp = TensorView::absent(DataType::Uint8);
                let absent_gidx = TensorView::absent(DataType::Int32);

                let mut low_y = Owned::zeros(dtype, &[m, n]);
                accuracy4_kernel(k, n, block_size)
                    .execute(
                        &[
                            low_a.view(),
                            b.view(),
                            low_scales.view(),
                            absent_zp,
                            absent_gidx,
                            low_bias.view(),
                        ],
                        &mut [low_y.view_mut()],
                    )
                    .unwrap();

                let mut f32_y = Owned::zeros_f32(&[m, n]);
                accuracy4_kernel(k, n, block_size)
                    .execute(
                        &[
                            f32_a.view(),
                            b.view(),
                            f32_scales.view(),
                            absent_zp,
                            absent_gidx,
                            f32_bias.view(),
                        ],
                        &mut [f32_y.view_mut()],
                    )
                    .unwrap();

                let actual = widened(&low_y);
                let reference = f32_y.to_f32();
                let narrowed_reference: Vec<f32> = reference
                    .iter()
                    .map(|&value| match dtype {
                        DataType::Float16 => half::f16::from_f32(value).to_f32(),
                        DataType::BFloat16 => half::bf16::from_f32(value).to_f32(),
                        _ => unreachable!(),
                    })
                    .collect();
                assert_eq!(
                    actual, narrowed_reference,
                    "{dtype:?} M={m} must compute in f32 and narrow only at output"
                );

                let tolerance: f32 = match dtype {
                    DataType::Float16 => 2e-2,
                    DataType::BFloat16 => 1.5e-1,
                    _ => unreachable!(),
                };
                for (index, (&actual, &reference)) in actual.iter().zip(&reference).enumerate() {
                    assert!(
                        (actual - reference).abs() <= tolerance.max(tolerance * reference.abs()),
                        "{dtype:?} M={m} index {index}: actual={actual}, widened f32={reference}"
                    );
                }
            }
        }
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

        let cached_ptr = prepack_cache_ptr(&kernel);
        assert!(
            cached_ptr.is_some(),
            "M=1 constant B must populate a prepacked weight cache"
        );
        let a2 = Owned::f32(&[1, k], &a2_values);
        let mut y2 = Owned::zeros_f32(&[1, n]);
        kernel
            .execute(&[a2.view(), b.view(), scales.view()], &mut [y2.view_mut()])
            .unwrap();
        assert_eq!(
            prepack_cache_ptr(&kernel),
            cached_ptr,
            "prepacked weight cache must be reused (stable) across activations"
        );
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
            prepack_cache_populated(&kernel),
            "M=1 constant B/scales/zero-points must take a prepacked path"
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
        let b = Owned::u8(&[1, 2, 8], &[0x99; 16]);
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
            .insert("bits".into(), Attribute::Int(3));
        let model = Model::new(&graph);
        let error = CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .err()
            .expect("bits=3 must be rejected");
        assert!(format!("{error}").contains("supports bits in {2, 4, 8}"));
    }

    #[test]
    fn matmulnbits_factory_accepts_bits8() {
        let (graph, node) = model_node(&[1, 16], &[1, 1, 16], &[1], None, &[1, 1], 16, 1, 16);
        let mut graph = graph;
        graph
            .node_mut(node)
            .attributes
            .insert("bits".into(), Attribute::Int(8));
        let model = Model::new(&graph);
        CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .expect("bits=8 must be accepted");
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

    /// The SQNBit decode crossover parses `NXRT_SQNBIT_DECODE_MIN`, falling
    /// back to the topology-derived default for absent, empty, or malformed values.
    #[cfg(feature = "mlas")]
    #[test]
    fn matmulnbits_resolve_decode_min_parses_or_defaults() {
        assert_eq!(default_sqnbit_decode_min(96), 16);
        assert_eq!(default_sqnbit_decode_min(4), 6);
        assert_eq!(default_sqnbit_decode_min(8), 8);
        assert_eq!(resolve_decode_min(None, 96), 16);
        assert_eq!(resolve_decode_min(Some(""), 96), 16);
        assert_eq!(resolve_decode_min(Some("abc"), 96), 16);
        assert_eq!(resolve_decode_min(Some("32"), 96), 32);
        assert_eq!(resolve_decode_min(Some("  8 "), 96), 8);
        assert_eq!(resolve_decode_min(Some("1"), 96), 1);
    }

    /// Serialize the few tests that mutate `NXRT_CPU_GEMM_BACKEND` so the global
    /// backend override does not race concurrent test threads.
    #[cfg(feature = "mlas")]
    fn backend_env_lock() -> &'static std::sync::Mutex<()> {
        static LOCK: OnceLock<std::sync::Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| std::sync::Mutex::new(()))
    }

    const SPMD_PARITY_CHILD_ENV: &str = "NXRT_SPMD_PARITY_CHILD";
    const SPMD_PARITY_MARKER: &str = "NXRT_SPMD_PARITY_BYTES=";

    fn real_int4_decode_fixture_bytes() -> Vec<u8> {
        let (n, k, block_size) = (1024usize, 1024usize, 32usize);
        let blocks = k / block_size;
        let packed = PackedInt4Weight {
            values: (0..n * blocks * (block_size / 2))
                .map(|index| {
                    let low = ((index * 13 + 3) & 0xf) as u8;
                    let high = ((index * 7 + 11) & 0xf) as u8;
                    low | (high << 4)
                })
                .collect(),
            scales: (0..n * blocks)
                .map(|index| 0.000_5 + (index % 29) as f32 * 0.000_031_25)
                .collect(),
        };
        let mut activation: Vec<f32> = (0..k)
            .map(|index| ((index * 37 % 257) as f32 - 128.0) * 0.007_812_5)
            .collect();
        let dot_kernel = selected_dot_kernel();
        let mut bytes = Vec::with_capacity(6 * n * std::mem::size_of::<f32>());

        with_decode_pool_scope(|| {
            for op in 0..6usize {
                let mut output = vec![0.0f32; n];
                int4_matmul_m1(&activation, &packed, &mut output, k, n, dot_kernel);
                for value in &output {
                    bytes.extend_from_slice(&value.to_bits().to_le_bytes());
                }
                for (index, value) in activation.iter_mut().enumerate() {
                    *value = output[index] * 0.125
                        + ((op * 17 + index * 5) % 31) as f32 * 0.000_976_562_5;
                }
            }
        });
        bytes
    }

    fn parity_child_output(persistent: bool) -> Vec<u8> {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .arg("--exact")
            .arg("kernels::matmul_nbits::tests::spmd_real_int4_parity_subprocess")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .env(SPMD_PARITY_CHILD_ENV, if persistent { "on" } else { "off" })
            .env(DECODE_THREADS_ENV, "31")
            .env("RAYON_NUM_THREADS", "31")
            .env_remove(crate::decode_affinity::DECODE_AFFINITY_ENV);
        if persistent {
            command.env(crate::decode_spmd::PERSISTENT_POOL_ENV, "1");
        } else {
            // Default-on: the OFF child must explicitly opt out (`=0`) to exercise
            // the flat legacy path; simply unsetting the var now auto-enables.
            command.env(crate::decode_spmd::PERSISTENT_POOL_ENV, "0");
        }
        let output = command.output().expect("run SPMD parity child process");
        assert!(
            output.status.success(),
            "SPMD parity child failed (persistent={persistent}):\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).expect("child output is UTF-8");
        let encoded = stdout
            .lines()
            .find_map(|line| {
                line.find(SPMD_PARITY_MARKER)
                    .map(|index| &line[index + SPMD_PARITY_MARKER.len()..])
            })
            .expect("child emitted parity bytes");
        assert_eq!(encoded.len() % 2, 0);
        encoded
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                let pair = std::str::from_utf8(pair).unwrap();
                u8::from_str_radix(pair, 16).unwrap()
            })
            .collect()
    }

    #[test]
    fn spmd_real_int4_parity_subprocess() {
        let Ok(mode) = std::env::var(SPMD_PARITY_CHILD_ENV) else {
            return;
        };
        let persistent = mode == "on";
        assert_eq!(
            crate::decode_spmd::pools().is_some(),
            persistent,
            "the ON child must build the persistent pool and the OFF child must not"
        );
        SPMD_TEST_DISPATCHES.store(0, std::sync::atomic::Ordering::Relaxed);
        let bytes = real_int4_decode_fixture_bytes();
        if persistent {
            assert!(
                SPMD_TEST_DISPATCHES.load(std::sync::atomic::Ordering::Relaxed) >= 6,
                "persistent parity child did not route every real int4 op through SPMD"
            );
        }
        let encoded: String = bytes.iter().map(|byte| format!("{byte:02x}")).collect();
        println!("{SPMD_PARITY_MARKER}{encoded}");
    }

    #[test]
    fn spmd_real_multi_op_int4_is_bit_identical_at_odd_worker_count() {
        let baseline = parity_child_output(false);
        let persistent = parity_child_output(true);
        assert_eq!(
            persistent, baseline,
            "31-worker persistent SPMD output must be byte-identical to flag-OFF \
             across every sequential packed-int4 MatMulNBits op"
        );
    }

    const AFFINITY_DEFER_CHILD_ENV: &str = "NXRT_AFFINITY_DEFER_CHILD";
    const AFFINITY_DEFER_MARKER: &str = "NXRT_AFFINITY_DEFER=";

    /// Child process for the affinity-defer routing tests (Chew #1). Dispatches on
    /// the scenario in `AFFINITY_DEFER_CHILD_ENV`; a plain run (var unset) is a
    /// no-op so the test is inert in the normal suite. Env is set by the parent
    /// *before* the process starts, so `pools()`/`plan_decode_affinity` observe it
    /// on their first (and only) evaluation.
    #[test]
    fn affinity_defer_routing_child() {
        let Ok(scenario) = std::env::var(AFFINITY_DEFER_CHILD_ENV) else {
            return;
        };
        match scenario.as_str() {
            // (a) Auto default + explicit non-numa-split affinity -> defer to the
            // flat path: the persistent SPMD pool must NOT be built.
            "auto_off" | "auto_node" | "auto_compact" => {
                assert!(
                    crate::decode_spmd::pools().is_none(),
                    "Auto default + explicit affinity ({scenario}) must defer to the flat \
                     path and build no persistent SPMD pool"
                );
            }
            // (b) Forced (`=1`) + affinity set -> SPMD still wins.
            "forced_off" => {
                assert!(
                    crate::decode_spmd::pools().is_some(),
                    "Forced persistent pool must ignore the affinity defer and build SPMD"
                );
            }
            // (c) Auto + malformed affinity -> deferred to flat AND the flat path
            // still surfaces the malformed-value error.
            "auto_malformed" => {
                assert!(
                    crate::decode_spmd::pools().is_none(),
                    "Auto default + malformed affinity must defer to the flat path"
                );
                assert!(
                    crate::decode_affinity::plan_decode_affinity(4).is_err(),
                    "malformed affinity must still surface an error on the deferred flat path"
                );
            }
            other => panic!("unknown affinity-defer scenario `{other}`"),
        }
        println!("{AFFINITY_DEFER_MARKER}ok");
    }

    fn run_affinity_defer_child(scenario: &str, affinity: &str, forced: bool) {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .arg("--exact")
            .arg("kernels::matmul_nbits::tests::affinity_defer_routing_child")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .env(AFFINITY_DEFER_CHILD_ENV, scenario)
            .env(crate::decode_affinity::DECODE_AFFINITY_ENV, affinity)
            .env(DECODE_THREADS_ENV, "8")
            .env("RAYON_NUM_THREADS", "8");
        if forced {
            command.env(crate::decode_spmd::PERSISTENT_POOL_ENV, "1");
        } else {
            // Auto: the persistence env must be unset so the default-on Auto mode
            // is what routes (or defers), not an explicit `=0`/`=1`.
            command.env_remove(crate::decode_spmd::PERSISTENT_POOL_ENV);
        }
        let output = command.output().expect("run affinity-defer child process");
        assert!(
            output.status.success(),
            "affinity-defer child failed (scenario={scenario}):\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).expect("child output is UTF-8");
        assert!(
            stdout.contains(&format!("{AFFINITY_DEFER_MARKER}ok")),
            "affinity-defer child did not confirm scenario {scenario}:\n{stdout}"
        );
    }

    /// (a) Auto default (`PERSISTENT_POOL` unset) with an explicit non-numa-split
    /// affinity defers to the flat path and builds no persistent SPMD pool.
    #[test]
    fn auto_default_with_explicit_affinity_defers_to_flat() {
        run_affinity_defer_child("auto_off", "off", false);
        run_affinity_defer_child("auto_node", "node:0", false);
        run_affinity_defer_child("auto_compact", "compact", false);
    }

    /// (b) Forced (`=1`) keeps the persistent SPMD pool even when an explicit
    /// affinity is set -- the affinity defer must not apply.
    #[test]
    fn forced_persistent_pool_ignores_explicit_affinity() {
        run_affinity_defer_child("forced_off", "off", true);
    }

    /// (c) A malformed affinity value in the Auto-defer path still errors (the flat
    /// path's `plan_decode_affinity` validates it exactly as before).
    #[test]
    fn auto_default_malformed_affinity_still_errors_on_flat_path() {
        run_affinity_defer_child("auto_malformed", "not-a-real-mode", false);
    }

    /// M-based hybrid routing gate: with an otherwise-eligible int4 case
    /// (`accuracy_level == 4`, so the hand decode path is fast) and the MLAS
    /// backend selected, `try_mlas_sqnbit` must still fall back (`Ok(None)`) for
    /// `m` below the decode crossover (decode keeps the hand path) and serve
    /// MLAS (`Ok(Some(()))`) for `m` at/above it (prefill). This regression-locks
    /// the decode/prefill split. Uses the topology-derived default threshold; the
    /// host must have an MLAS SQNBit int4 kernel or the assertions are skipped.
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

        let at = default_sqnbit_decode_min(available_parallelism());
        let below = at - 1;

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
            "m={below} (< {at}) must fall back to the hand int4 path",
        );
        assert_eq!(
            prefill,
            Some(()),
            "m={at} must route to MLAS SQNBit",
        );
    }

    /// Slow-hand-path decode routing: for `m == 1` with `bits == 4` but
    /// `accuracy_level != 4`, the hand path would dequantize the whole weight to
    /// f32 and run a dense GEMV. MLAS SQNBit (CompFp32) beats that, so
    /// `try_mlas_sqnbit` must route this small-`m` case to MLAS
    /// (`Ok(Some(()))`), unlike the fast `accuracy_level == 4` decode case which
    /// stays on the hand path. Skipped when the host lacks the MLAS kernel.
    #[cfg(feature = "mlas")]
    #[test]
    fn matmulnbits_try_mlas_serves_slow_dequant_decode() {
        let (n, k, block_size) = (32usize, 64usize, 32usize);
        let k_blocks = k.div_ceil(block_size);
        let blob = block_size / 2;
        let weights_nk = pseudo(n * k, 0.3);
        let (packed_bytes, scales, _zps, dq) = quantize(&weights_nk, n, k, block_size, false);

        if mlas_sys::SQNBitPackedB::new(
            n,
            k,
            4,
            block_size,
            mlas_sys::SQNBitComputeType::Fp32,
            &packed_bytes,
            &scales,
            None,
        )
        .is_none()
        {
            eprintln!("MLAS SQNBit int4 CompFp32 kernel unavailable; skipping slow-decode test");
            return;
        }

        // accuracy_level 0 => hand path would use the slow f32 dequant GEMV.
        let kernel = test_kernel(k, n, block_size);
        let b = Owned::u8(&[n, k_blocks, blob], &packed_bytes);
        let scales_t = Owned::f32(&[n, k_blocks], &scales);

        let _guard = backend_env_lock().lock().unwrap();
        let previous = std::env::var("NXRT_CPU_GEMM_BACKEND").ok();
        // SAFETY: the backend env lock serializes readers/writers of this var.
        unsafe { std::env::set_var("NXRT_CPU_GEMM_BACKEND", "mlas") };

        let a = pseudo(k, 0.8);
        let mut result = vec![0.0f32; n];
        let served = kernel
            .try_mlas_sqnbit(
                &b.view(),
                &scales_t.view(),
                None,
                None,
                false,
                &a,
                1,
                None,
                &mut result,
            )
            .unwrap();

        // SAFETY: still holding the backend env lock; restore prior value.
        unsafe {
            match &previous {
                Some(value) => std::env::set_var("NXRT_CPU_GEMM_BACKEND", value),
                None => std::env::remove_var("NXRT_CPU_GEMM_BACKEND"),
            }
        }

        assert_eq!(
            served,
            Some(()),
            "m=1 bits=4 accuracy_level=0 (slow hand dequant GEMV) must route to MLAS SQNBit",
        );
        // CompFp32 dequant is near-exact, so it must match the f32 reference.
        let expected = reference(&a, &dq, 1, k, n);
        mlas_close(&result, &expected, 2e-3, "slow-dequant m1 CompFp32");
    }

    /// Regression for the `accuracy_level = 0` slow-path bug: MLAS SQNBit is a
    /// specialized quantized kernel independent of the dense-f32 [`CpuBackend`]
    /// microkernel, so an `accuracy_level != 4` MatMulNBits must route to MLAS
    /// (CompFp32) even when the resolved backend is *not* MLAS -- the real
    /// default on an AVX2 host, where [`CpuBackend::auto_detect`] returns
    /// `SimdX86`. Before the fix the `auto_detect() != Mlas` gate dropped this
    /// case to the slow full-f32-dequant GEMV. `accuracy_level = 4` must be
    /// unaffected: its fast hand int8/int4 path stays selected (returns `None`)
    /// unless the whole backend is explicitly forced to MLAS. Skipped when the
    /// host lacks the MLAS kernel.
    #[cfg(feature = "mlas")]
    #[test]
    fn matmulnbits_try_mlas_routes_acclevel0_without_mlas_backend() {
        let (n, k, block_size) = (32usize, 64usize, 32usize);
        let k_blocks = k.div_ceil(block_size);
        let blob = block_size / 2;
        let weights_nk = pseudo(n * k, 0.3);
        let (packed_bytes, scales, _zps, dq) = quantize(&weights_nk, n, k, block_size, false);

        if mlas_sys::SQNBitPackedB::new(
            n,
            k,
            4,
            block_size,
            mlas_sys::SQNBitComputeType::Fp32,
            &packed_bytes,
            &scales,
            None,
        )
        .is_none()
        {
            eprintln!("MLAS SQNBit int4 CompFp32 kernel unavailable; skipping acc0-default-backend test");
            return;
        }

        let b = Owned::u8(&[n, k_blocks, blob], &packed_bytes);
        let scales_t = Owned::f32(&[n, k_blocks], &scales);
        let a = pseudo(k, 0.8);

        let _guard = backend_env_lock().lock().unwrap();
        let previous = std::env::var("NXRT_CPU_GEMM_BACKEND").ok();
        // SAFETY: the backend env lock serializes readers/writers of this var.
        // Force a non-MLAS backend to model the real-world default: MLAS SQNBit
        // routing for accuracy_level != 4 must not depend on the dense-GEMM
        // backend being MLAS.
        unsafe { std::env::set_var("NXRT_CPU_GEMM_BACKEND", "generic") };
        assert_ne!(
            crate::backend::CpuBackend::auto_detect(),
            crate::backend::CpuBackend::Mlas,
            "test precondition: backend must not resolve to MLAS",
        );

        let call = |kernel: &MatMulNBitsKernel| {
            let mut result = vec![0.0f32; n];
            let served = kernel
                .try_mlas_sqnbit(
                    &b.view(),
                    &scales_t.view(),
                    None,
                    None,
                    false,
                    &a,
                    1,
                    None,
                    &mut result,
                )
                .unwrap();
            (served, result)
        };

        let (acc0_served, acc0_result) = call(&test_kernel(k, n, block_size));
        let (acc4_served, _) = call(&accuracy4_kernel(k, n, block_size));

        // SAFETY: still holding the backend env lock; restore prior value.
        unsafe {
            match &previous {
                Some(value) => std::env::set_var("NXRT_CPU_GEMM_BACKEND", value),
                None => std::env::remove_var("NXRT_CPU_GEMM_BACKEND"),
            }
        }

        assert_eq!(
            acc0_served,
            Some(()),
            "accuracy_level=0 must route to MLAS SQNBit even when the backend is not MLAS",
        );
        // CompFp32 dequant is near-exact, so it must match the f32 reference.
        let expected = reference(&a, &dq, 1, k, n);
        mlas_close(&acc0_result, &expected, 2e-3, "acc0 default-backend CompFp32");

        assert_eq!(
            acc4_served, None,
            "accuracy_level=4 decode must stay on the fast hand path when the backend is not MLAS",
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

    /// Full M=1 decode-step probe at real 7B (Qwen2.5-Coder-7B) projection
    /// shapes: replays the exact per-token MatMulNBits op sequence (qkv, o,
    /// gate, up, down per layer, plus the lm_head) back-to-back inside one
    /// decode-pool residency, so it captures the *sequential per-op dispatch*
    /// overhead the isolated `matmulnbits_mlas_perf` probe misses. Compares the
    /// hand int4 GEMV path against MLAS SQNBit CompInt8 at the real decode
    /// thread count. Shapes come from the model (read once, listed here only as
    /// a probe fixture); production routing never hardcodes them.
    ///
    ///   cargo test -p onnx-runtime-ep-cpu --features mlas --release \
    ///     matmulnbits_mlas_decode_step -- --ignored --nocapture
    #[cfg(feature = "mlas")]
    #[test]
    #[ignore = "perf probe; run explicitly with --ignored --nocapture"]
    fn matmulnbits_mlas_decode_step() {
        use std::time::Instant;

        // (K, N, count-per-token) for one Qwen2.5-Coder-7B decode step.
        let layers = 28usize;
        let ops: &[(usize, usize, usize)] = &[
            (3584, 4608, layers),    // qkv_proj
            (3584, 3584, layers),    // o_proj
            (3584, 18944, layers),   // gate_proj
            (3584, 18944, layers),   // up_proj
            (18944, 3584, layers),   // down_proj
            (3584, 152064, 1),       // lm_head
        ];
        let block_size = 32usize;
        let dot_kernel = selected_dot_kernel();

        struct Weights {
            k: usize,
            n: usize,
            int4: PackedInt4Weight,
            mlas_int8: mlas_sys::SQNBitPackedB,
            mlas_fp32: mlas_sys::SQNBitPackedB,
        }

        // Build one *distinct* weight per op instance so the step streams the
        // full ~3.5 GB of cold int4 weights from DRAM, exactly like the model
        // (reusing a handful of buffers would keep them L3-resident and report
        // fantasy bandwidth). Distinct scale seeds also defeat page dedup.
        let mut built: Vec<Weights> = Vec::new();
        let mut weight_bytes = 0u64;
        for (shape_index, &(k, n, count)) in ops.iter().enumerate() {
            for instance in 0..count {
                let seed = 0.3 + shape_index as f32 * 0.11 + instance as f32 * 0.001;
                let weights_nk = pseudo(n * k, seed);
                let (packed_bytes, scales, _zps, _dq) =
                    quantize(&weights_nk, n, k, block_size, false);
                let make = |comp| {
                    mlas_sys::SQNBitPackedB::new(n, k, 4, block_size, comp, &packed_bytes, &scales, None)
                };
                let (Some(mlas_int8), Some(mlas_fp32)) = (
                    make(mlas_sys::SQNBitComputeType::Int8),
                    make(mlas_sys::SQNBitComputeType::Fp32),
                ) else {
                    eprintln!("MLAS SQNBit int4 kernel unavailable; skipping decode-step probe");
                    return;
                };
                weight_bytes += (n as u64) * (k as u64) / 2;
                built.push(Weights {
                    k,
                    n,
                    int4: PackedInt4Weight {
                        values: packed_bytes,
                        scales,
                    },
                    mlas_int8,
                    mlas_fp32,
                });
            }
        }

        let threads = configured_decode_threads()
            .or_else(|| default_decode_threads(available_parallelism()))
            .unwrap_or(1);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();

        let run_hand = || {
            for w in &built {
                let a = vec![0.03f32; w.k];
                let mut out = vec![0.0f32; w.n];
                int4_matmul_m1(&a, &w.int4, &mut out, w.k, w.n, dot_kernel);
            }
        };
        let run_mlas_int8 = || {
            for w in &built {
                let a = vec![0.03f32; w.k];
                let mut out = vec![0.0f32; w.n];
                mlas_sys::sqnbit_gemm(&w.mlas_int8, 1, &a, None, &mut out, true);
            }
        };
        let run_mlas_fp32 = || {
            for w in &built {
                let a = vec![0.03f32; w.k];
                let mut out = vec![0.0f32; w.n];
                mlas_sys::sqnbit_gemm(&w.mlas_fp32, 1, &a, None, &mut out, true);
            }
        };

        let step = |label: &str, run: &(dyn Fn() + Sync)| {
            pool.install(|| {
                for _ in 0..3 {
                    run();
                }
                let iters = 20u32;
                let start = Instant::now();
                for _ in 0..iters {
                    run();
                }
                let per_step = start.elapsed().as_secs_f64() / iters as f64;
                let gbs = weight_bytes as f64 / per_step / 1e9;
                eprintln!(
                    "decode-step {label}: {:.2} ms/step  {:.2} tok/s  {:.1} GB/s ({threads}t)",
                    per_step * 1e3,
                    1.0 / per_step,
                    gbs,
                );
            });
        };

        eprintln!(
            "decode-step probe: {} weight bytes/token, {threads} decode threads",
            weight_bytes
        );
        step("hand", &run_hand);
        step("mlas-int8", &run_mlas_int8);
        step("mlas-fp32", &run_mlas_fp32);
    }
}

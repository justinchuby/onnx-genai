//! `com.microsoft::MatMulNBits`: decode-specialized packed INT4 GEMV plus the
//! block-wise dequantization and f32 cuBLASLt GEMM fallback used for prefill.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use cudarc::driver::{LaunchConfig, PushKernelArg, sys::CUdeviceptr};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::blas::{self, GemmDtype, GemmEpilogue, GemmEpilogueKind, GemmParams, WORKSPACE_BYTES};
use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};

const DEQUANT_MODULE: &str = "matmul_nbits_dequant_f32";
const DEQUANT_ENTRY: &str = "matmul_nbits_dequant_f32";
const GEMV_MODULE: &str = "matmul_nbits_gemv";
const GEMV_F32_ENTRY: &str = "matmul_nbits_gemv_f32";
const QUANTIZE_ACCURACY4_ENTRY: &str = "matmul_nbits_quantize_accuracy4_block32";
const GEMV_ACCURACY4_ENTRY: &str = "matmul_nbits_gemv_accuracy4_block32";
const ACCURACY4_MODULE: &str = "matmul_nbits_accuracy4";
const ACCURACY4_ENTRY: &str = "matmul_nbits_accuracy4";
const BLOCK_THREADS: u32 = 256;
const GEMV_ACCURACY4_THREADS: u32 = 256;
const GEMV_ACCURACY4_COLUMNS_PER_BLOCK: usize = 8;
const GEMV_ACCURACY4_SHARED_BYTES: u32 = 32 * 32;

const DEQUANT_SRC: &str = r#"
extern "C" __global__ void matmul_nbits_dequant_f32(
    const unsigned char* packed,
    const float* scales,
    const unsigned char* zero_points,
    const int* group_indices,
    float* weight_kn,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes)
{
    const long total = (long)k * n;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (long)gridDim.x * blockDim.x) {
        const int depth = (int)(idx / n);
        const int output = (int)(idx % n);
        const int block = depth / block_size;
        const int within = depth - block * block_size;
        const unsigned char byte =
            packed[((long)output * k_blocks + block) * blob_size + within / 2];
        const int quantized = (within & 1) ? (byte >> 4) : (byte & 15);
        const int group = group_indices ? group_indices[depth] : block;
        if (group < 0 || group >= k_blocks) {
            weight_kn[idx] = 0.0f;
            continue;
        }
        int zero_point = 8;
        if (zero_points) {
            const unsigned char zp = zero_points[(long)output * zp_row_bytes + group / 2];
            zero_point = (group & 1) ? (zp >> 4) : (zp & 15);
        }
        weight_kn[idx] =
            ((float)quantized - (float)zero_point) * scales[(long)output * k_blocks + group];
    }
}
"#;

const GEMV_SRC: &str = r#"
__device__ __forceinline__ float warp_sum(float value)
{
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float block_sum(float value)
{
    __shared__ float warp_sums[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();
    value = threadIdx.x < ((blockDim.x + 31) >> 5) ? warp_sums[lane] : 0.0f;
    return warp == 0 ? warp_sum(value) : 0.0f;
}

extern "C" __global__ void matmul_nbits_gemv_f32(
    const float* activation,
    const unsigned char* packed,
    const float* scales,
    const unsigned char* zero_points,
    const float* bias,
    float* output,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes)
{
    const int column = (int)blockIdx.x;
    if (column >= n) {
        return;
    }

    float value = 0.0f;
    for (int depth = (int)threadIdx.x; depth < k; depth += (int)blockDim.x) {
        const int block = depth / block_size;
        const int within = depth - block * block_size;
        const unsigned char byte =
            packed[((long)column * k_blocks + block) * blob_size + within / 2];
        const int quantized = (within & 1) ? (byte >> 4) : (byte & 15);
        int zero_point = 8;
        if (zero_points) {
            const unsigned char zp =
                zero_points[(long)column * zp_row_bytes + block / 2];
            zero_point = (block & 1) ? (zp >> 4) : (zp & 15);
        }
        value += activation[depth] * ((float)quantized - (float)zero_point)
            * scales[(long)column * k_blocks + block];
    }

    value = block_sum(value);
    if (threadIdx.x == 0) {
        output[column] = value + (bias ? bias[column] : 0.0f);
    }
}

extern "C" __global__ void matmul_nbits_quantize_accuracy4_block32(
    const float* activation,
    signed char* quantized_activation,
    float* activation_scale_out,
    const int k,
    const int padded_k)
{
    const int lane = (int)threadIdx.x;
    float max_abs = 0.0f;
    for (int depth = lane; depth < k; depth += 32) {
        max_abs = fmaxf(max_abs, fabsf(activation[depth]));
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_abs = fmaxf(max_abs,
            __shfl_down_sync(0xffffffffu, max_abs, offset));
    }
    max_abs = __shfl_sync(0xffffffffu, max_abs, 0);

    const float activation_scale = max_abs == 0.0f ? 0.0f : max_abs / 127.0f;
    const float inverse_scale =
        activation_scale == 0.0f ? 0.0f : 1.0f / activation_scale;
    if (lane == 0) {
        *activation_scale_out = activation_scale;
    }
    for (int depth = lane; depth < padded_k; depth += 32) {
        int quantized = 0;
        if (depth < k && activation_scale != 0.0f) {
            quantized = (int)roundf(fminf(127.0f, fmaxf(-127.0f,
                activation[depth] * inverse_scale)));
        }
        quantized_activation[depth] = (signed char)quantized;
    }
}

__device__ __forceinline__ int unpack_int4x4(unsigned int packed, int offset)
{
    const int w0 = (int)((packed >> (offset + 0)) & 15u) - 8;
    const int w1 = (int)((packed >> (offset + 4)) & 15u) - 8;
    const int w2 = (int)((packed >> (offset + 8)) & 15u) - 8;
    const int w3 = (int)((packed >> (offset + 12)) & 15u) - 8;
    return (w0 & 255) | ((w1 & 255) << 8) | ((w2 & 255) << 16)
        | ((w3 & 255) << 24);
}

extern "C" __global__ void matmul_nbits_gemv_accuracy4_block32(
    const signed char* quantized_activation,
    const float* activation_scale_ptr,
    const unsigned char* packed,
    const float* scales,
    const float* bias,
    float* output,
    const int k,
    const int n,
    const int k_blocks)
{
    extern __shared__ signed char activation_tile[];
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int column = (int)blockIdx.x * 8 + warp;
    const float activation_scale = *activation_scale_ptr;
    if (activation_scale == 0.0f) {
        if (lane == 0 && column < n) {
            output[column] = bias ? bias[column] : 0.0f;
        }
        return;
    }

    float value = 0.0f;
    for (int tile_block = 0; tile_block < k_blocks; tile_block += 32) {
        const int tile_blocks = min(32, k_blocks - tile_block);
        const int tile_depths = tile_blocks * 32;
        for (int depth = tid; depth < tile_depths; depth += (int)blockDim.x) {
            activation_tile[depth] =
                quantized_activation[tile_block * 32 + depth];
        }
        __syncthreads();

        const int block = tile_block + lane;
        if (column < n && block < k_blocks) {
            const long packed_start = ((long)column * k_blocks + block) * 16;
            const uint4 packed_weights =
                *reinterpret_cast<const uint4*>(packed + packed_start);
            const unsigned int words[4] = {
                packed_weights.x, packed_weights.y, packed_weights.z, packed_weights.w
            };
            const signed char* activation_block = activation_tile + lane * 32;
            int dot = 0;
#pragma unroll
            for (int word = 0; word < 4; ++word) {
                const int activation0 =
                    *reinterpret_cast<const int*>(activation_block + word * 8);
                const int activation1 =
                    *reinterpret_cast<const int*>(activation_block + word * 8 + 4);
                dot = __dp4a(activation0, unpack_int4x4(words[word], 0), dot);
                dot = __dp4a(activation1, unpack_int4x4(words[word], 16), dot);
            }
            const float scaled =
                __fmul_rn((float)dot, scales[(long)column * k_blocks + block]);
            value = __fadd_rn(value, scaled);
        }
        __syncthreads();
    }

    value = __fmul_rn(warp_sum(value), activation_scale);
    if (lane == 0 && column < n) {
        output[column] = bias ? __fadd_rn(value, bias[column]) : value;
    }
}
"#;

const ACCURACY4_SRC: &str = r#"
extern "C" __global__ void matmul_nbits_accuracy4(
    const float* a,
    const unsigned char* packed,
    const float* scales,
    const unsigned char* zero_points,
    const float* bias,
    float* y,
    const int m,
    const int k,
    const int n,
    const int block_size,
    const int k_blocks,
    const int blob_size,
    const int zp_row_bytes)
{
    const long total = (long)m * n;
    for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (long)gridDim.x * blockDim.x) {
        const int row = (int)(idx / n);
        const int output = (int)(idx % n);
        const float* activation = a + (long)row * k;

        float max_abs = 0.0f;
        for (int depth = 0; depth < k; ++depth) {
            max_abs = fmaxf(max_abs, fabsf(activation[depth]));
        }
        if (max_abs == 0.0f) {
            y[idx] = bias ? bias[output] : 0.0f;
            continue;
        }

        const float activation_scale = max_abs / 127.0f;
        const float inverse_scale = 1.0f / activation_scale;
        float value = 0.0f;
        for (int block = 0; block < k_blocks; ++block) {
            int dot = 0;
            const int begin = block * block_size;
            const int end = min(begin + block_size, k);
            int zero_point = 8;
            if (zero_points) {
                const unsigned char zp =
                    zero_points[(long)output * zp_row_bytes + block / 2];
                zero_point = (block & 1) ? (zp >> 4) : (zp & 15);
            }
            for (int depth = begin; depth < end; ++depth) {
                int quantized_activation =
                    (int)roundf(fminf(127.0f, fmaxf(-127.0f,
                        activation[depth] * inverse_scale)));
                const int within = depth - begin;
                const unsigned char byte =
                    packed[((long)output * k_blocks + block) * blob_size + within / 2];
                const int quantized_weight =
                    (within & 1) ? (byte >> 4) : (byte & 15);
                dot += quantized_activation * (quantized_weight - zero_point);
            }
            if (m == 1 && block_size == 32 && !zero_points) {
                const float scaled =
                    __fmul_rn((float)dot, scales[(long)output * k_blocks + block]);
                value = __fadd_rn(value, scaled);
            } else {
                const float combined_scale = __fmul_rn(
                    activation_scale,
                    scales[(long)output * k_blocks + block]);
                value = __fadd_rn(value, __fmul_rn((float)dot, combined_scale));
            }
        }
        if (m == 1 && block_size == 32 && !zero_points) {
            value = __fmul_rn(value, activation_scale);
        }
        y[idx] = bias ? __fadd_rn(value, bias[output]) : value;
    }
}
"#;

pub struct MatMulNBitsFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for MatMulNBitsFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let k = required_positive_attr(node, "K")?;
        let n = required_positive_attr(node, "N")?;
        let bits = optional_int_attr(node, "bits")?.unwrap_or(4);
        if bits != 4 {
            return Err(error(format!(
                "only bits=4 is supported in the CUDA kernel, got {bits}"
            )));
        }
        let weight_prepacked = optional_int_attr(node, "weight_prepacked")?.unwrap_or(0);
        if weight_prepacked != 0 {
            return Err(error(format!(
                "weight_prepacked={weight_prepacked} is unsupported: CUDA only supports the standard (non-prepacked) layout"
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

        let accuracy4_workspace = if accuracy_level == 4 && block_size == 32 {
            Some(Mutex::new(Accuracy4Workspace::new(
                self.runtime.clone(),
                k,
            )?))
        } else {
            None
        };
        Ok(Box::new(MatMulNBitsKernel {
            runtime: self.runtime.clone(),
            k,
            n,
            block_size,
            accuracy_level,
            accuracy4_workspace,
            last_call_capture_safe: AtomicBool::new(false),
        }))
    }
}

#[derive(Debug)]
struct Accuracy4Workspace {
    runtime: Arc<CudaRuntime>,
    quantized_activation: CUdeviceptr,
    activation_scale: CUdeviceptr,
    padded_k: usize,
}

impl Accuracy4Workspace {
    fn new(runtime: Arc<CudaRuntime>, k: usize) -> Result<Self> {
        let padded_k = k.div_ceil(32) * 32;
        let quantized_activation = runtime.alloc_raw(padded_k + std::mem::size_of::<f32>())?;
        Ok(Self {
            runtime,
            quantized_activation,
            activation_scale: quantized_activation + padded_k as CUdeviceptr,
            padded_k,
        })
    }
}

impl Drop for Accuracy4Workspace {
    fn drop(&mut self) {
        if self.quantized_activation != 0 {
            // SAFETY: this persistent buffer is exclusively owned by the kernel.
            let _ = unsafe { self.runtime.free_raw(self.quantized_activation) };
            self.quantized_activation = 0;
            self.activation_scale = 0;
        }
    }
}

#[derive(Debug)]
pub struct MatMulNBitsKernel {
    runtime: Arc<CudaRuntime>,
    k: usize,
    n: usize,
    block_size: usize,
    accuracy_level: i64,
    accuracy4_workspace: Option<Mutex<Accuracy4Workspace>>,
    last_call_capture_safe: AtomicBool,
}

impl MatMulNBitsKernel {
    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.last_call_capture_safe.store(false, Ordering::Relaxed);
        if !(3..=6).contains(&inputs.len()) || outputs.len() != 1 {
            return Err(error(format!(
                "expected 3 to 6 inputs and 1 output, got {} inputs and {} outputs",
                inputs.len(),
                outputs.len()
            )));
        }
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
        let blob_size = self.block_size / 2;
        require_shape("B", inputs[1].shape, &[self.n, k_blocks, blob_size])?;
        require_flat_or_matrix_shape("scales", inputs[2].shape, self.n, k_blocks)?;

        let zero_points = optional_input(inputs, 3);
        let zp_row_bytes = k_blocks.div_ceil(2);
        if let Some(zp) = zero_points {
            require_dtype("zero_points", zp.dtype, DataType::Uint8)?;
            require_flat_or_matrix_shape("zero_points", zp.shape, self.n, zp_row_bytes)?;
        }

        let group_indices = optional_input(inputs, 4);
        if let Some(g_idx) = group_indices {
            require_dtype("g_idx", g_idx.dtype, DataType::Int32)?;
            if !g_idx.is_contiguous() {
                return Err(error(
                    "g_idx must be contiguous on the CUDA execution provider",
                ));
            }
            let padded_k = k_blocks * self.block_size;
            if g_idx.shape != [self.k] && g_idx.shape != [padded_k] {
                return Err(error(format!(
                    "g_idx must have shape [{}] or [{padded_k}], got {:?}",
                    self.k, g_idx.shape
                )));
            }
            let mut bytes = vec![0u8; g_idx.numel() * 4];
            // SAFETY: `g_idx` is a live contiguous device tensor and `bytes`
            // exactly covers all of its i32 elements.
            unsafe {
                self.runtime
                    .dtoh(&mut bytes, cuptr(g_idx.data_ptr::<u8>() as *const c_void))?
            };
            for (index, value) in bytes.chunks_exact(4).enumerate() {
                let group = i32::from_ne_bytes([value[0], value[1], value[2], value[3]]);
                if group < 0 || group as usize >= k_blocks {
                    return Err(error(format!(
                        "g_idx[{index}]={group} is outside 0..{k_blocks}"
                    )));
                }
            }
        }

        let bias = optional_input(inputs, 5);
        if let Some(bias) = bias {
            require_dtype("bias", bias.dtype, DataType::Float32)?;
            require_shape("bias", bias.shape, &[self.n])?;
        }

        for (name, contiguous) in [
            ("A", inputs[0].is_contiguous()),
            ("B", inputs[1].is_contiguous()),
            ("scales", inputs[2].is_contiguous()),
            (
                "zero_points",
                zero_points.is_none_or(TensorView::is_contiguous),
            ),
            ("g_idx", group_indices.is_none_or(TensorView::is_contiguous)),
            ("bias", bias.is_none_or(TensorView::is_contiguous)),
            ("Y", outputs[0].is_contiguous()),
        ] {
            if !contiguous {
                return Err(error(format!(
                    "{name} must be contiguous on the CUDA execution provider"
                )));
            }
        }

        let m = a_shape[..a_shape.len() - 1].iter().product::<usize>();
        self.last_call_capture_safe
            .store(m == 1 && group_indices.is_none(), Ordering::Relaxed);
        if m == 1 && group_indices.is_none() {
            if self.accuracy_level == 4 && self.block_size == 32 && zero_points.is_none() {
                return self.launch_accuracy4_gemv(
                    &inputs[0],
                    &inputs[1],
                    &inputs[2],
                    bias,
                    &mut outputs[0],
                    k_blocks,
                );
            }
            if self.accuracy_level != 4 {
                return self.launch_f32_gemv(
                    &inputs[0],
                    &inputs[1],
                    &inputs[2],
                    zero_points,
                    bias,
                    &mut outputs[0],
                    k_blocks,
                    blob_size,
                    zp_row_bytes,
                );
            }
        }
        if self.accuracy_level == 4 && group_indices.is_none() {
            return self.launch_accuracy4(
                &inputs[0],
                &inputs[1],
                &inputs[2],
                zero_points,
                bias,
                &mut outputs[0],
                m,
                k_blocks,
                blob_size,
                zp_row_bytes,
            );
        }

        let weight = self.runtime.alloc_raw(self.k * self.n * 4)?;
        let workspace = match self.runtime.alloc_raw(WORKSPACE_BYTES) {
            Ok(workspace) => workspace,
            Err(err) => {
                // SAFETY: `weight` was allocated above and has not been freed.
                let _ = unsafe { self.runtime.free_raw(weight) };
                return Err(err);
            }
        };

        let result = self
            .launch_dequant(
                &inputs[1],
                &inputs[2],
                zero_points,
                group_indices,
                weight,
                k_blocks,
                blob_size,
                zp_row_bytes,
            )
            .and_then(|()| {
                let params = GemmParams {
                    dtype: GemmDtype::F32,
                    a: cuptr(inputs[0].data_ptr::<u8>() as *const c_void),
                    b: weight,
                    c: cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void),
                    m,
                    k: self.k,
                    n: self.n,
                    batch: 1,
                    a_batch_stride: m * self.k,
                    b_batch_stride: 0,
                    epilogue: bias.map(|bias| GemmEpilogue {
                        kind: GemmEpilogueKind::Bias,
                        bias: cuptr(bias.data_ptr::<u8>() as *const c_void),
                    }),
                };
                // SAFETY: validated dense f32 A/Y and the dequantized [K,N]
                // allocation cover the complete GEMM; workspace and stream live
                // through the call and Y aliases neither input.
                unsafe {
                    blas::gemm(
                        self.runtime.blas(),
                        self.runtime.stream_ptr(),
                        &params,
                        workspace,
                        WORKSPACE_BYTES,
                    )
                }
            })
            .and_then(|()| self.runtime.synchronize());

        // SAFETY: both pointers came from `alloc_raw` and are released once,
        // after all submitted work has synchronized (or the submission failed).
        let free_workspace = unsafe { self.runtime.free_raw(workspace) };
        let free_weight = unsafe { self.runtime.free_raw(weight) };
        result.and(free_workspace).and(free_weight)
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_f32_gemv(
        &self,
        activation: &TensorView,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
        bias: Option<&TensorView>,
        output: &mut TensorMut,
        k_blocks: usize,
        blob_size: usize,
        zp_row_bytes: usize,
    ) -> Result<()> {
        let function = self
            .runtime
            .nvrtc_function(GEMV_MODULE, GEMV_SRC, GEMV_F32_ENTRY)?;
        let activation_ptr = cuptr(activation.data_ptr::<u8>() as *const c_void);
        let packed_ptr = cuptr(packed.data_ptr::<u8>() as *const c_void);
        let scales_ptr = cuptr(scales.data_ptr::<u8>() as *const c_void);
        let zero_points_ptr = zero_points
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let bias_ptr = bias
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let k = as_i32("K", self.k)?;
        let n = as_i32("N", self.n)?;
        let block_size = as_i32("block_size", self.block_size)?;
        let k_blocks = as_i32("K block count", k_blocks)?;
        let blob_size = as_i32("block blob size", blob_size)?;
        let zp_row_bytes = as_i32("zero-point row size", zp_row_bytes)?;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&activation_ptr)
            .arg(&packed_ptr)
            .arg(&scales_ptr)
            .arg(&zero_points_ptr)
            .arg(&bias_ptr)
            .arg(&output_ptr)
            .arg(&k)
            .arg(&n)
            .arg(&block_size)
            .arg(&k_blocks)
            .arg(&blob_size)
            .arg(&zp_row_bytes);
        // SAFETY: validated dense tensors cover the complete M=1 operation and
        // the scalar ABI matches `matmul_nbits_gemv_f32`.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (self.n as u32, 1, 1),
                block_dim: (BLOCK_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|err| driver_err("launch MatMulNBits f32 GEMV", err))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_accuracy4_gemv(
        &self,
        activation: &TensorView,
        packed: &TensorView,
        scales: &TensorView,
        bias: Option<&TensorView>,
        output: &mut TensorMut,
        k_blocks: usize,
    ) -> Result<()> {
        let workspace = self
            .accuracy4_workspace
            .as_ref()
            .ok_or_else(|| error("accuracy_level=4 GEMV workspace is unavailable"))?
            .lock()
            .map_err(|_| error("accuracy_level=4 GEMV workspace lock poisoned"))?;
        let quantize_function =
            self.runtime
                .nvrtc_function(GEMV_MODULE, GEMV_SRC, QUANTIZE_ACCURACY4_ENTRY)?;
        let gemv_function =
            self.runtime
                .nvrtc_function(GEMV_MODULE, GEMV_SRC, GEMV_ACCURACY4_ENTRY)?;
        let activation_ptr = cuptr(activation.data_ptr::<u8>() as *const c_void);
        let packed_ptr = cuptr(packed.data_ptr::<u8>() as *const c_void);
        let scales_ptr = cuptr(scales.data_ptr::<u8>() as *const c_void);
        let bias_ptr = bias
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let k = as_i32("K", self.k)?;
        let n = as_i32("N", self.n)?;
        let k_blocks = as_i32("K block count", k_blocks)?;
        let padded_k = as_i32("padded K", workspace.padded_k)?;

        let mut quantize_builder = self.runtime.stream().launch_builder(&quantize_function);
        quantize_builder
            .arg(&activation_ptr)
            .arg(&workspace.quantized_activation)
            .arg(&workspace.activation_scale)
            .arg(&k)
            .arg(&padded_k);
        // SAFETY: the persistent workspace covers padded_k int8 values plus the
        // f32 scale, and the scalar ABI matches the quantization entry point.
        unsafe {
            quantize_builder.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|err| driver_err("launch MatMulNBits accuracy_level=4 quantization", err))?;

        let mut gemv_builder = self.runtime.stream().launch_builder(&gemv_function);
        gemv_builder
            .arg(&workspace.quantized_activation)
            .arg(&workspace.activation_scale)
            .arg(&packed_ptr)
            .arg(&scales_ptr)
            .arg(&bias_ptr)
            .arg(&output_ptr)
            .arg(&k)
            .arg(&n)
            .arg(&k_blocks);
        // SAFETY: this path is restricted to symmetric block-32 M=1 inputs; the
        // persistent quantized activation is initialized by the preceding stream
        // launch, and the scalar ABI matches the tiled GEMV entry point.
        unsafe {
            gemv_builder.launch(LaunchConfig {
                grid_dim: (
                    self.n.div_ceil(GEMV_ACCURACY4_COLUMNS_PER_BLOCK) as u32,
                    1,
                    1,
                ),
                block_dim: (GEMV_ACCURACY4_THREADS, 1, 1),
                shared_mem_bytes: GEMV_ACCURACY4_SHARED_BYTES,
            })
        }
        .map(|_| ())
        .map_err(|err| driver_err("launch MatMulNBits accuracy_level=4 GEMV", err))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_accuracy4(
        &self,
        activation: &TensorView,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
        bias: Option<&TensorView>,
        output: &mut TensorMut,
        m: usize,
        k_blocks: usize,
        blob_size: usize,
        zp_row_bytes: usize,
    ) -> Result<()> {
        let total = m.checked_mul(self.n).ok_or_else(|| {
            error(format!(
                "accuracy_level=4 output size {m} * {} overflows usize",
                self.n
            ))
        })?;
        let blocks = total.div_ceil(BLOCK_THREADS as usize).clamp(1, 65_535) as u32;
        let function =
            self.runtime
                .nvrtc_function(ACCURACY4_MODULE, ACCURACY4_SRC, ACCURACY4_ENTRY)?;
        let activation_ptr = cuptr(activation.data_ptr::<u8>() as *const c_void);
        let packed_ptr = cuptr(packed.data_ptr::<u8>() as *const c_void);
        let scales_ptr = cuptr(scales.data_ptr::<u8>() as *const c_void);
        let zero_points_ptr = zero_points
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let bias_ptr = bias
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let output_ptr = cuptr(output.data_ptr_mut::<u8>() as *const c_void);
        let m = as_i32("M", m)?;
        let k = as_i32("K", self.k)?;
        let n = as_i32("N", self.n)?;
        let block_size = as_i32("block_size", self.block_size)?;
        let k_blocks = as_i32("K block count", k_blocks)?;
        let blob_size = as_i32("block blob size", blob_size)?;
        let zp_row_bytes = as_i32("zero-point row size", zp_row_bytes)?;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&activation_ptr)
            .arg(&packed_ptr)
            .arg(&scales_ptr)
            .arg(&zero_points_ptr)
            .arg(&bias_ptr)
            .arg(&output_ptr)
            .arg(&m)
            .arg(&k)
            .arg(&n)
            .arg(&block_size)
            .arg(&k_blocks)
            .arg(&blob_size)
            .arg(&zp_row_bytes);
        // SAFETY: all tensors were dtype/shape/contiguity validated above and
        // the scalar ABI matches `matmul_nbits_accuracy4`.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|err| driver_err("launch MatMulNBits accuracy_level=4", err))
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_dequant(
        &self,
        packed: &TensorView,
        scales: &TensorView,
        zero_points: Option<&TensorView>,
        group_indices: Option<&TensorView>,
        weight: cudarc::driver::sys::CUdeviceptr,
        k_blocks: usize,
        blob_size: usize,
        zp_row_bytes: usize,
    ) -> Result<()> {
        let packed_ptr = cuptr(packed.data_ptr::<u8>() as *const c_void);
        let scales_ptr = cuptr(scales.data_ptr::<u8>() as *const c_void);
        let zero_points_ptr = zero_points
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let group_indices_ptr = group_indices
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let k = as_i32("K", self.k)?;
        let n = as_i32("N", self.n)?;
        let block_size = as_i32("block_size", self.block_size)?;
        let k_blocks = as_i32("K block count", k_blocks)?;
        let blob_size = as_i32("block blob size", blob_size)?;
        let zp_row_bytes = as_i32("zero-point row size", zp_row_bytes)?;
        let total = self.k * self.n;
        let blocks = total.div_ceil(BLOCK_THREADS as usize).clamp(1, 65_535) as u32;
        let function = self
            .runtime
            .nvrtc_function(DEQUANT_MODULE, DEQUANT_SRC, DEQUANT_ENTRY)?;
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&packed_ptr)
            .arg(&scales_ptr)
            .arg(&zero_points_ptr)
            .arg(&group_indices_ptr)
            .arg(&weight)
            .arg(&k)
            .arg(&n)
            .arg(&block_size)
            .arg(&k_blocks)
            .arg(&blob_size)
            .arg(&zp_row_bytes);
        // SAFETY: argument order/types match the CUDA entry point; all device
        // buffers were shape-validated and `weight` has K*N f32 elements.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (BLOCK_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map(|_| ())
        .map_err(|err| driver_err("launch MatMulNBits dequant", err))
    }
}

impl Kernel for MatMulNBitsKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        false
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        // The m=1 no-g_idx GEMV uses only launch-time shared memory and a
        // shape-fixed persistent accuracy-4 activation workspace; it performs no
        // per-call allocation, D2H, or synchronization. Prefill GEMM and g_idx
        // validation retain their non-capturable behavior.
        if self.last_call_capture_safe.load(Ordering::Relaxed) {
            onnx_runtime_ep_api::CaptureSupport::Supported
        } else {
            onnx_runtime_ep_api::CaptureSupport::unsupported(
                "requires M==1 decode GEMV without group_indices; prefill allocates scratch and group_indices validation reads D2H",
            )
        }
    }
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

fn as_i32(name: &str, value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| error(format!("{name}={value} exceeds i32")))
}

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("cuda_ep MatMulNBits: {}", message.into()))
}

//! `com.microsoft::MatMulNBits`: block-wise INT4 weight dequantization followed
//! by a full-precision f32 cuBLASLt GEMM.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::blas::{self, GemmDtype, GemmEpilogue, GemmEpilogueKind, GemmParams, WORKSPACE_BYTES};
use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};

const DEQUANT_MODULE: &str = "matmul_nbits_dequant_f32";
const DEQUANT_ENTRY: &str = "matmul_nbits_dequant_f32";
const ACCURACY4_MODULE: &str = "matmul_nbits_accuracy4";
const ACCURACY4_ENTRY: &str = "matmul_nbits_accuracy4";
const BLOCK_THREADS: u32 = 256;

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
                value += (float)dot * scales[(long)output * k_blocks + block];
            } else {
                value += (float)dot *
                    (activation_scale * scales[(long)output * k_blocks + block]);
            }
        }
        if (m == 1 && block_size == 32 && !zero_points) {
            value *= activation_scale;
        }
        y[idx] = value + (bias ? bias[output] : 0.0f);
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

        Ok(Box::new(MatMulNBitsKernel {
            runtime: self.runtime.clone(),
            k,
            n,
            block_size,
            accuracy_level,
        }))
    }
}

#[derive(Debug)]
pub struct MatMulNBitsKernel {
    runtime: Arc<CudaRuntime>,
    k: usize,
    n: usize,
    block_size: usize,
    accuracy_level: i64,
}

impl MatMulNBitsKernel {
    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
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
        .map_err(|err| driver_err("launch MatMulNBits accuracy_level=4", err))?;
        self.runtime.synchronize()
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

    fn cuda_graph_compatible(&self) -> bool {
        false
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

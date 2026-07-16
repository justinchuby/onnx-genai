//! Decode-specialized CUDA GEMV for native GGUF block formats.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node, Shape};

use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};

const OP: &str = "BlockQuantizedMatMul";
const DOMAIN: &str = "com.github.onnxruntime.genai";
const LAYOUT_VERSION: i64 = 1;
const QK: usize = 32;
const MXFP4_BLOCK_BYTES: usize = 17;
const IQ4_NL_BLOCK_BYTES: usize = 18;
const BLOCK_THREADS: u32 = 256;
const GEMV_MODULE: &str = "block_quantized_matmul_gemv";
const GEMV_ENTRY: &str = "block_quantized_matmul_gemv_f32";

const GEMV_SRC: &str = r#"
__device__ __constant__ signed char e2m1_doubled[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

__device__ __constant__ signed char iq4_nl_codebook[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113
};

__device__ __forceinline__ float fp16_to_fp32(unsigned short value)
{
    const unsigned int sign = ((unsigned int)value & 0x8000u) << 16;
    unsigned int exponent = ((unsigned int)value >> 10) & 0x1fu;
    unsigned int mantissa = (unsigned int)value & 0x03ffu;
    unsigned int bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int unbiased = -14;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --unbiased;
            }
            mantissa &= 0x03ffu;
            bits = sign | ((unsigned int)(unbiased + 127) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }
    return __uint_as_float(bits);
}

__device__ __forceinline__ float e8m0_half_scale(unsigned char exponent)
{
    if (exponent == 0xffu) {
        return __uint_as_float(0x7fc00000u);
    }
    if (exponent == 0u) {
        return __uint_as_float(0x00200000u);
    }
    if (exponent == 1u) {
        return __uint_as_float(0x00400000u);
    }
    return __uint_as_float(((unsigned int)exponent - 1u) << 23);
}

__device__ __forceinline__ float decode_weight(
    const unsigned char* packed,
    int format,
    int blocks,
    int block_bytes,
    int column,
    int depth)
{
    const int block = depth >> 5;
    const int within = depth & 31;
    const unsigned char* data =
        packed + ((long long)column * blocks + block) * block_bytes;
    const int quant_index = within & 15;
    const unsigned char quant = data[block_bytes - 16 + quant_index];
    const int code = within < 16 ? (quant & 15) : (quant >> 4);
    if (format == 0) {
        return (float)e2m1_doubled[code] * e8m0_half_scale(data[0]);
    }
    const unsigned short scale_bits =
        (unsigned short)data[0] | ((unsigned short)data[1] << 8);
    return fp16_to_fp32(scale_bits) * (float)iq4_nl_codebook[code];
}

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

extern "C" __global__ void block_quantized_matmul_gemv_f32(
    const float* activation,
    const unsigned char* packed,
    const float* bias,
    float* output,
    const int k,
    const int n,
    const int blocks,
    const int block_bytes,
    const int format)
{
    const int column = (int)blockIdx.x;
    if (column >= n) {
        return;
    }

    float value = 0.0f;
    for (int depth = (int)threadIdx.x; depth < k; depth += (int)blockDim.x) {
        value += activation[depth]
            * decode_weight(packed, format, blocks, block_bytes, column, depth);
    }
    value = block_sum(value);
    if (threadIdx.x == 0) {
        output[column] = value + (bias ? bias[column] : 0.0f);
    }
}
"#;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BlockFormat {
    Mxfp4,
    Iq4Nl,
}

impl BlockFormat {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "mxfp4" => Ok(Self::Mxfp4),
            "iq4_nl" => Ok(Self::Iq4Nl),
            other => Err(error(format!(
                "format '{other}' is unsupported by CUDA; supported formats are mxfp4 and iq4_nl"
            ))),
        }
    }

    fn block_bytes(self) -> usize {
        match self {
            Self::Mxfp4 => MXFP4_BLOCK_BYTES,
            Self::Iq4Nl => IQ4_NL_BLOCK_BYTES,
        }
    }

    fn kernel_id(self) -> i32 {
        match self {
            Self::Mxfp4 => 0,
            Self::Iq4Nl => 1,
        }
    }
}

pub struct BlockQuantizedMatMulFactory {
    pub runtime: Arc<CudaRuntime>,
}

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
            runtime: self.runtime.clone(),
            k,
            n,
            format,
        }))
    }
}

#[derive(Debug)]
struct BlockQuantizedMatMulKernel {
    runtime: Arc<CudaRuntime>,
    k: usize,
    n: usize,
    format: BlockFormat,
}

impl Kernel for BlockQuantizedMatMulKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if !(2..=3).contains(&inputs.len()) || outputs.len() != 1 {
            return Err(error(format!(
                "expected 2 to 3 inputs and 1 output, got {} inputs and {} outputs",
                inputs.len(),
                outputs.len()
            )));
        }
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
        let m = a_shape[..a_shape.len() - 1].iter().product::<usize>();
        if m != 1 {
            return Err(error(format!(
                "CUDA currently supports only the decode GEMV path M=1, got M={m}"
            )));
        }
        let expected_output_shape = [&a_shape[..a_shape.len() - 1], &[self.n]].concat();
        require_shape("Y", outputs[0].shape, &expected_output_shape)?;

        let blocks = self.k.div_ceil(QK);
        require_shape(
            "packed_B",
            inputs[1].shape,
            &[self.n, blocks, self.format.block_bytes()],
        )?;
        let bias = inputs.get(2).filter(|input| !input.is_absent());
        if let Some(bias) = bias {
            require_dtype("bias", bias.dtype, DataType::Float32)?;
            require_shape("bias", bias.shape, &[self.n])?;
        }
        for (name, contiguous) in [
            ("A", inputs[0].is_contiguous()),
            ("packed_B", inputs[1].is_contiguous()),
            ("bias", bias.is_none_or(TensorView::is_contiguous)),
            ("Y", outputs[0].is_contiguous()),
        ] {
            if !contiguous {
                return Err(error(format!(
                    "{name} must be contiguous on the CUDA execution provider"
                )));
            }
        }

        let function = self
            .runtime
            .nvrtc_function(GEMV_MODULE, GEMV_SRC, GEMV_ENTRY)?;
        let activation_ptr = cuptr(inputs[0].data_ptr::<u8>() as *const c_void);
        let packed_ptr = cuptr(inputs[1].data_ptr::<u8>() as *const c_void);
        let bias_ptr = bias
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let output_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let k = as_i32("K", self.k)?;
        let n = as_i32("N", self.n)?;
        let blocks = as_i32("block count", blocks)?;
        let block_bytes = as_i32("block byte count", self.format.block_bytes())?;
        let format = self.format.kernel_id();
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&activation_ptr)
            .arg(&packed_ptr)
            .arg(&bias_ptr)
            .arg(&output_ptr)
            .arg(&k)
            .arg(&n)
            .arg(&blocks)
            .arg(&block_bytes)
            .arg(&format);
        // SAFETY: all tensors are dense and shape-checked, and the scalar ABI
        // matches `block_quantized_matmul_gemv_f32`.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (self.n as u32, 1, 1),
                block_dim: (BLOCK_THREADS, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|err| driver_err("launch BlockQuantizedMatMul GEMV", err))?;
        self.runtime.synchronize()
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        false
    }

    fn cuda_graph_compatible(&self) -> bool {
        false
    }
}

pub(crate) fn supports_node(node: &Node, shapes: &[Shape]) -> bool {
    let format_supported = node
        .attr("format")
        .and_then(|attribute| attribute.as_str())
        .is_some_and(|format| matches!(format, "mxfp4" | "iq4_nl"));
    let layout_supported = match node.attr("block_layout_version") {
        Some(attribute) => attribute.as_int() == Some(LAYOUT_VERSION),
        None => true,
    };
    let attributes_valid = ["K", "N"].into_iter().all(|name| {
        node.attr(name)
            .and_then(|attribute| attribute.as_int())
            .is_some_and(|value| value > 0)
    });
    let Some(a_shape) = shapes.first() else {
        return false;
    };
    let decode_shape = !a_shape.is_empty()
        && a_shape[..a_shape.len() - 1]
            .iter()
            .try_fold(1usize, |product, dim| {
                dim.as_static().and_then(|value| product.checked_mul(value))
            })
            == Some(1);
    format_supported && layout_supported && attributes_valid && decode_shape
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

fn as_i32(name: &str, value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| error(format!("{name}={value} exceeds CUDA i32 limits")))
}

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("cuda_ep {DOMAIN}::{OP}: {}", message.into()))
}

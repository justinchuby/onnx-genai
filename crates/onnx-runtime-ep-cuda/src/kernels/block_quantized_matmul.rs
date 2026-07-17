//! Decode-specialized CUDA GEMV for native GGUF block formats.

use std::ffi::c_void;
use std::fmt::Write;
use std::sync::{Arc, OnceLock};

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node, Shape};
use onnx_runtime_quantization::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XS_SIGNS, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID,
};

use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};

const OP: &str = "BlockQuantizedMatMul";
const DOMAIN: &str = "pkg.nxrt";
const LAYOUT_VERSION: i64 = 1;
const SMALL_QK: usize = 32;
const IQ_SUPER_QK: usize = 256;
const MXFP4_BLOCK_BYTES: usize = 17;
const IQ4_NL_BLOCK_BYTES: usize = 18;
const IQ4_XS_BLOCK_BYTES: usize = 136;
const IQ2_XXS_BLOCK_BYTES: usize = 66;
const IQ3_XXS_BLOCK_BYTES: usize = 98;
const IQ2_XS_BLOCK_BYTES: usize = 74;
const IQ2_S_BLOCK_BYTES: usize = 82;
const IQ3_S_BLOCK_BYTES: usize = 110;
const IQ1_S_BLOCK_BYTES: usize = 50;
const IQ1_M_BLOCK_BYTES: usize = 56;
const BLOCK_THREADS: u32 = 256;
const GEMV_MODULE: &str = "block_quantized_matmul_gemv";
const GEMV_ENTRY: &str = "block_quantized_matmul_gemv_f32";

const GEMV_PREFIX: &str = r#"
__device__ __constant__ signed char e2m1_doubled[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

__device__ __constant__ signed char iq4_nl_codebook[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113
};
"#;

const GEMV_SUFFIX: &str = r#"
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

__device__ __forceinline__ unsigned short load_u16_le(const unsigned char* data)
{
    return (unsigned short)data[0] | ((unsigned short)data[1] << 8);
}

__device__ __forceinline__ unsigned int load_u32_le(const unsigned char* data)
{
    return (unsigned int)data[0]
        | ((unsigned int)data[1] << 8)
        | ((unsigned int)data[2] << 16)
        | ((unsigned int)data[3] << 24);
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

__device__ __forceinline__ float signed_grid_value_u64(
    unsigned long long grid,
    int element,
    unsigned char sign_mask,
    float scale)
{
    const float magnitude = (float)((grid >> (8 * element)) & 0xffull);
    return sign_mask & (1u << element) ? -scale * magnitude : scale * magnitude;
}

__device__ __forceinline__ float signed_grid_value_u32(
    unsigned int grid,
    int element,
    int sign_element,
    unsigned char sign_mask,
    float scale)
{
    const float magnitude = (float)((grid >> (8 * element)) & 0xffu);
    return sign_mask & (1u << sign_element) ? -scale * magnitude : scale * magnitude;
}

__device__ __forceinline__ float iq1_grid_value(unsigned long long grid, int element)
{
    const int byte = (int)((grid >> (8 * element)) & 0xffull);
    return (float)(byte < 128 ? byte : byte - 256);
}

__device__ __forceinline__ float decode_weight(
    const unsigned char* packed,
    int format,
    int blocks,
    int block_bytes,
    int column,
    int depth)
{
    const int superblock = format >= 2;
    const int qk = superblock ? 256 : 32;
    const int block = depth / qk;
    const int within = depth - block * qk;
    const unsigned char* data =
        packed + ((long long)column * blocks + block) * block_bytes;

    if (format == 0) {
        const int quant_index = within & 15;
        const unsigned char quant = data[1 + quant_index];
        const int code = within < 16 ? (quant & 15) : (quant >> 4);
        return (float)e2m1_doubled[code] * e8m0_half_scale(data[0]);
    }
    const float scale = fp16_to_fp32(load_u16_le(data));
    if (format == 1) {
        const int quant_index = within & 15;
        const unsigned char quant = data[2 + quant_index];
        const int code = within < 16 ? (quant & 15) : (quant >> 4);
        return scale * (float)iq4_nl_codebook[code];
    }
    if (format == 2) {
        const int subblock = within >> 5;
        const int subwithin = within & 31;
        const unsigned short scales_h = load_u16_le(data + 2);
        const unsigned char low =
            (data[4 + subblock / 2] >> (4 * (subblock & 1))) & 0x0fu;
        const unsigned char high = (scales_h >> (2 * subblock)) & 0x03u;
        const int factor = (int)(low | (high << 4)) - 32;
        const float subscale = scale * (float)factor;
        const unsigned char quant = data[8 + subblock * 16 + (subwithin & 15)];
        const int code = subwithin < 16 ? (quant & 15) : (quant >> 4);
        return subscale * (float)iq4_nl_codebook[code];
    }

    const int group32 = within >> 5;
    const int subwithin = within & 31;
    const int vector = subwithin >> 3;
    const int element = subwithin & 7;
    if (format == 3) {
        const int base = 2 + group32 * 8;
        const unsigned int metadata = load_u32_le(data + base + 4);
        const float subscale = scale * (0.5f + (float)(metadata >> 28)) * 0.25f;
        const unsigned long long grid = iq2xxs_grid[data[base + vector]];
        const unsigned char signs =
            iq2xs_signs[(metadata >> (7 * vector)) & 127u];
        return signed_grid_value_u64(grid, element, signs, subscale);
    }
    if (format == 4) {
        const unsigned int metadata = load_u32_le(data + 66 + group32 * 4);
        const float subscale = scale * (0.5f + (float)(metadata >> 28)) * 0.5f;
        const int quant_base = 2 + group32 * 8 + vector * 2;
        const unsigned int grid = iq3xxs_grid[data[quant_base + element / 4]];
        const unsigned char signs =
            iq2xs_signs[(metadata >> (7 * vector)) & 127u];
        return signed_grid_value_u32(
            grid, element & 3, element, signs, subscale);
    }
    if (format == 5) {
        const int quant_base = 2 + group32 * 8 + vector * 2;
        const unsigned short quant = load_u16_le(data + quant_base);
        const unsigned char packed_scale = data[66 + group32];
        const float subscale =
            scale * (0.5f + (float)((packed_scale >> (4 * (vector / 2))) & 15u))
            * 0.25f;
        const unsigned long long grid = iq2xs_grid[quant & 511u];
        const unsigned char signs = iq2xs_signs[quant >> 9];
        return signed_grid_value_u64(grid, element, signs, subscale);
    }
    if (format == 6) {
        const unsigned char packed_scale = data[74 + group32];
        const float subscale =
            scale * (0.5f + (float)((packed_scale >> (4 * (vector / 2))) & 15u))
            * 0.25f;
        const unsigned char qh = data[66 + group32];
        const unsigned int index =
            (unsigned int)data[2 + group32 * 4 + vector]
            | ((unsigned int)((qh >> (2 * vector)) & 3u) << 8);
        const unsigned long long grid = iq2s_grid[index];
        const unsigned char signs = data[34 + group32 * 4 + vector];
        return signed_grid_value_u64(grid, element, signs, subscale);
    }

    if (format == 7) {
        const int group64 = within >> 6;
        const int half = (within >> 5) & 1;
        const int vector4 = (within >> 3) & 3;
        const int element4 = within & 7;
        const unsigned char packed_scale = data[106 + group64];
        const float subscale =
            scale * (float)(1 + 2 * ((packed_scale >> (4 * half)) & 15u));
        const unsigned char qh = data[66 + group64 * 2 + half];
        const int quant_base = 2 + group64 * 16 + half * 8 + vector4 * 2;
        const unsigned int index =
            (unsigned int)data[quant_base + element4 / 4]
            | ((unsigned int)((qh >> (2 * vector4 + element4 / 4)) & 1u) << 8);
        const unsigned int grid = iq3s_grid[index];
        const unsigned char signs = data[74 + group64 * 8 + half * 4 + vector4];
        return signed_grid_value_u32(
            grid, element4 & 3, element4, signs, subscale);
    }
    if (format == 8) {
        const unsigned short qh = load_u16_le(data + 34 + group32 * 2);
        const float subscale = scale * (float)(2 * ((qh >> 12) & 7u) + 1);
        const float delta = qh & 0x8000u ? -0.125f : 0.125f;
        const unsigned int index =
            (unsigned int)data[2 + group32 * 4 + vector]
            | ((unsigned int)((qh >> (3 * vector)) & 7u) << 8);
        return subscale * (iq1_grid_value(iq1s_grid[index], element) + delta);
    }

    const unsigned short packed_scale0 = load_u16_le(data + 48);
    const unsigned short packed_scale1 = load_u16_le(data + 50);
    const unsigned short packed_scale2 = load_u16_le(data + 52);
    const unsigned short packed_scale3 = load_u16_le(data + 54);
    const unsigned short scale_bits =
        (packed_scale0 >> 12)
        | ((packed_scale1 >> 8) & 0x00f0u)
        | ((packed_scale2 >> 4) & 0x0f00u)
        | (packed_scale3 & 0xf000u);
    const float iq1m_scale = fp16_to_fp32(scale_bits);
    const unsigned short packed_scale =
        load_u16_le(data + 48 + 2 * (group32 / 2));
    const int scale_shift = 6 * (group32 & 1);
    const float subscale = iq1m_scale
        * (float)(2 * ((packed_scale >> (scale_shift + (vector >= 2 ? 3 : 0))) & 7u) + 1);
    const unsigned char qh = data[32 + group32 * 2 + vector / 2];
    const int high_shift = 4 * (vector & 1);
    const unsigned int index =
        (unsigned int)data[group32 * 4 + vector]
        | ((unsigned int)((qh >> high_shift) & 7u) << 8);
    const float delta = qh & (0x08u << high_shift) ? -0.125f : 0.125f;
    return subscale * (iq1_grid_value(iq1s_grid[index], element) + delta);
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

fn gemv_src() -> &'static str {
    static SOURCE: OnceLock<String> = OnceLock::new();
    SOURCE.get_or_init(|| {
        let mut source = String::from(GEMV_PREFIX);
        append_u8_table(&mut source, "iq2xs_signs", &IQ2XS_SIGNS);
        append_u64_table(&mut source, "iq2xxs_grid", &IQ2XXS_GRID);
        append_u32_table(&mut source, "iq3xxs_grid", &IQ3XXS_GRID);
        append_u64_table(&mut source, "iq2xs_grid", &IQ2XS_GRID);
        append_u64_table(&mut source, "iq2s_grid", &IQ2S_GRID);
        append_u32_table(&mut source, "iq3s_grid", &IQ3S_GRID);
        append_u64_table(&mut source, "iq1s_grid", &IQ1S_GRID);
        source.push_str(GEMV_SUFFIX);
        source
    })
}

fn append_u8_table(source: &mut String, name: &str, values: &[u8]) {
    writeln!(
        source,
        "__device__ __constant__ unsigned char {name}[{}] = {{",
        values.len()
    )
    .expect("writing CUDA source to String cannot fail");
    for values in values.chunks(16) {
        for value in values {
            write!(source, "{value},").expect("writing CUDA source to String cannot fail");
        }
        source.push('\n');
    }
    source.push_str("};\n");
}

fn append_u32_table(source: &mut String, name: &str, values: &[u32]) {
    writeln!(
        source,
        "__device__ __constant__ unsigned int {name}[{}] = {{",
        values.len()
    )
    .expect("writing CUDA source to String cannot fail");
    for values in values.chunks(8) {
        for value in values {
            write!(source, "0x{value:08x}u,").expect("writing CUDA source to String cannot fail");
        }
        source.push('\n');
    }
    source.push_str("};\n");
}

fn append_u64_table(source: &mut String, name: &str, values: &[u64]) {
    writeln!(
        source,
        "__device__ __constant__ unsigned long long {name}[{}] = {{",
        values.len()
    )
    .expect("writing CUDA source to String cannot fail");
    for values in values.chunks(4) {
        for value in values {
            write!(source, "0x{value:016x}ull,")
                .expect("writing CUDA source to String cannot fail");
        }
        source.push('\n');
    }
    source.push_str("};\n");
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BlockFormat {
    Mxfp4,
    Iq4Nl,
    Iq4Xs,
    Iq2Xxs,
    Iq3Xxs,
    Iq2Xs,
    Iq2S,
    Iq3S,
    Iq1S,
    Iq1M,
}

impl BlockFormat {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "mxfp4" => Ok(Self::Mxfp4),
            "iq4_nl" => Ok(Self::Iq4Nl),
            "iq4_xs" => Ok(Self::Iq4Xs),
            "iq2_xxs" => Ok(Self::Iq2Xxs),
            "iq3_xxs" => Ok(Self::Iq3Xxs),
            "iq2_xs" => Ok(Self::Iq2Xs),
            "iq2_s" => Ok(Self::Iq2S),
            "iq3_s" => Ok(Self::Iq3S),
            "iq1_s" => Ok(Self::Iq1S),
            "iq1_m" => Ok(Self::Iq1M),
            other => Err(error(format!(
                "format '{other}' is unsupported by CUDA; supported formats are mxfp4, iq4_nl, iq4_xs, iq2_xxs, iq3_xxs, iq2_xs, iq2_s, iq3_s, iq1_s, and iq1_m"
            ))),
        }
    }

    fn qk(self) -> usize {
        match self {
            Self::Mxfp4 | Self::Iq4Nl => SMALL_QK,
            Self::Iq4Xs
            | Self::Iq2Xxs
            | Self::Iq3Xxs
            | Self::Iq2Xs
            | Self::Iq2S
            | Self::Iq3S
            | Self::Iq1S
            | Self::Iq1M => IQ_SUPER_QK,
        }
    }

    fn block_bytes(self) -> usize {
        match self {
            Self::Mxfp4 => MXFP4_BLOCK_BYTES,
            Self::Iq4Nl => IQ4_NL_BLOCK_BYTES,
            Self::Iq4Xs => IQ4_XS_BLOCK_BYTES,
            Self::Iq2Xxs => IQ2_XXS_BLOCK_BYTES,
            Self::Iq3Xxs => IQ3_XXS_BLOCK_BYTES,
            Self::Iq2Xs => IQ2_XS_BLOCK_BYTES,
            Self::Iq2S => IQ2_S_BLOCK_BYTES,
            Self::Iq3S => IQ3_S_BLOCK_BYTES,
            Self::Iq1S => IQ1_S_BLOCK_BYTES,
            Self::Iq1M => IQ1_M_BLOCK_BYTES,
        }
    }

    fn kernel_id(self) -> i32 {
        match self {
            Self::Mxfp4 => 0,
            Self::Iq4Nl => 1,
            Self::Iq4Xs => 2,
            Self::Iq2Xxs => 3,
            Self::Iq3Xxs => 4,
            Self::Iq2Xs => 5,
            Self::Iq2S => 6,
            Self::Iq3S => 7,
            Self::Iq1S => 8,
            Self::Iq1M => 9,
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

        let blocks = self.k.div_ceil(self.format.qk());
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
            .nvrtc_function(GEMV_MODULE, gemv_src(), GEMV_ENTRY)?;
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
        .is_some_and(|format| {
            matches!(
                format,
                "mxfp4"
                    | "iq4_nl"
                    | "iq4_xs"
                    | "iq2_xxs"
                    | "iq3_xxs"
                    | "iq2_xs"
                    | "iq2_s"
                    | "iq3_s"
                    | "iq1_s"
                    | "iq1_m"
            )
        });
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

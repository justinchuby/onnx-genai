//! CUDA GEMV/GEMM kernels for native GGUF block formats.

use std::borrow::Cow;
use std::ffi::c_void;
use std::fmt::Write;
use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::block_quant_schema::{
    BLOCK_QUANTIZED_MATMUL_INPUT_COUNT as INPUT_COUNT,
    BLOCK_QUANTIZED_MATMUL_INPUT_NAMES as INPUT_NAMES, BQMM_ACTIVATION, BQMM_BIAS, BQMM_SCALE,
    BQMM_WEIGHT, PlanarBlockGeometry, planar_geometry_from_node, require_layout_v1,
};
use onnx_runtime_ir::{DataType, Node, Shape};
use onnx_runtime_quantization::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XS_SIGNS, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID,
};

use crate::error::driver_err;
use crate::kernels::planar_block_decode::{
    PlanarActivationDtype, PlanarLinearDims, PlanarLinearPointers, launch_planar_linear_borrowed,
    validate_planar_bank_device, warm_planar_linear,
};
use crate::runtime::{CudaRuntime, cuptr};

const OP: &str = "BlockQuantizedMatMul";
const DOMAIN: &str = onnx_runtime_ir::RUNTIME_DOMAIN;
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
const GEMM_TILE_M: u32 = 8;
const CUDA_MAX_GRID_DIM_Y: u32 = 65_535;
// 4K row tiles saturate the device while keeping the grid-stride path testable.
const GEMM_GRID_DIM_Y_CAP: u32 = 4_096;
const _: () = assert!(GEMM_GRID_DIM_Y_CAP <= CUDA_MAX_GRID_DIM_Y);
const GEMV_MODULE: &str = "block_quantized_matmul_gemv";
const GEMV_ENTRY: &str = "block_quantized_matmul_gemv_f32";
const GEMM_MODULE: &str = "block_quantized_matmul_gemm";
const GEMM_ENTRY: &str = "block_quantized_matmul_gemm_f32";

const PREFIX: &str = r#"
__device__ __constant__ signed char e2m1_doubled[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};

__device__ __constant__ signed char iq4_nl_codebook[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113
};
"#;

const SUFFIX: &str = r#"
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
"#;

const GEMV_KERNEL: &str = r#"
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

const GEMM_KERNEL: &str = r#"
extern "C" __global__ void block_quantized_matmul_gemm_f32(
    const float* activation,
    const unsigned char* packed,
    const float* bias,
    float* output,
    const unsigned long long m,
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

    const unsigned long long row_stride =
        (unsigned long long)gridDim.y * GEMM_TILE_M;
    for (unsigned long long row_base =
             (unsigned long long)blockIdx.y * GEMM_TILE_M;
         row_base < m;
         row_base += row_stride) {
        float values[GEMM_TILE_M] = {0.0f};
        for (int depth = (int)threadIdx.x; depth < k; depth += (int)blockDim.x) {
            const float weight =
                decode_weight(packed, format, blocks, block_bytes, column, depth);
#pragma unroll
            for (int row = 0; row < GEMM_TILE_M; ++row) {
                const unsigned long long row_index = row_base + (unsigned long long)row;
                if (row_index < m) {
                    values[row] +=
                        activation[row_index * (unsigned long long)k + (unsigned long long)depth]
                        * weight;
                }
            }
        }

#pragma unroll
        for (int row = 0; row < GEMM_TILE_M; ++row) {
            const float value = block_sum(values[row]);
            __syncthreads();
            const unsigned long long row_index = row_base + (unsigned long long)row;
            if (threadIdx.x == 0 && row_index < m) {
                output[row_index * (unsigned long long)n + (unsigned long long)column] =
                    value + (bias ? bias[column] : 0.0f);
            }
        }
    }
}
"#;

fn gemv_src() -> &'static str {
    static SOURCE: OnceLock<String> = OnceLock::new();
    SOURCE.get_or_init(|| module_src(GEMV_KERNEL, None))
}

fn gemm_src() -> &'static str {
    static SOURCE: OnceLock<String> = OnceLock::new();
    SOURCE.get_or_init(|| module_src(GEMM_KERNEL, Some(GEMM_TILE_M)))
}

/// Build the shared NVRTC device prelude: the E2M1/IQ4 codebooks, every IQ grid
/// table, the fp16/E8M0 helpers, the `decode_weight` GGUF block decoder, and the
/// `warp_sum`/`block_sum` reductions. Other kernels (e.g. `BlockQuantizedMoE`)
/// concatenate their own `__global__` entry points after this to reuse the exact
/// same per-weight decode numerics as the parity oracle.
pub(crate) fn decoder_prelude() -> String {
    let mut source = String::from(PREFIX);
    append_u8_table(&mut source, "iq2xs_signs", &IQ2XS_SIGNS);
    append_u64_table(&mut source, "iq2xxs_grid", &IQ2XXS_GRID);
    append_u32_table(&mut source, "iq3xxs_grid", &IQ3XXS_GRID);
    append_u64_table(&mut source, "iq2xs_grid", &IQ2XS_GRID);
    append_u64_table(&mut source, "iq2s_grid", &IQ2S_GRID);
    append_u32_table(&mut source, "iq3s_grid", &IQ3S_GRID);
    append_u64_table(&mut source, "iq1s_grid", &IQ1S_GRID);
    source.push_str(SUFFIX);
    source
}

fn module_src(kernel: &str, gemm_tile_m: Option<u32>) -> String {
    let mut source = decoder_prelude();
    if let Some(tile_m) = gemm_tile_m {
        writeln!(source, "#define GEMM_TILE_M {tile_m}")
            .expect("writing CUDA source to String cannot fail");
    }
    source.push_str(kernel);
    source
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
pub(crate) enum BlockFormat {
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
    pub(crate) fn parse(value: &str) -> Result<Self> {
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

    pub(crate) fn qk(self) -> usize {
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

    pub(crate) fn block_bytes(self) -> usize {
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

    pub(crate) fn kernel_id(self) -> i32 {
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
        if node.inputs.len() != INPUT_COUNT {
            return Err(error(format!(
                "expected exactly {INPUT_COUNT} positional inputs, got {}",
                node.inputs.len()
            )));
        }
        let k = required_positive_attr(node, "K")?;
        let n = required_positive_attr(node, "N")?;
        require_layout_v1(node, OP).map_err(error)?;
        let format = if let Some(geometry) =
            planar_geometry_from_node(node, OP, "format", "block_size_out", "block_size_in")
                .map_err(error)?
        {
            warm_planar_linear(&self.runtime)?;
            MatMulFormat::Planar(geometry)
        } else {
            MatMulFormat::Interleaved(match node.attr("format") {
                Some(attribute) => attribute
                    .as_str()
                    .ok_or_else(|| error("attribute 'format' must be a UTF-8 string"))
                    .and_then(BlockFormat::parse)?,
                None => return Err(error("missing required string attribute 'format'")),
            })
        };
        let validation_scratch = if matches!(format, MatMulFormat::Planar(_)) {
            Some(self.runtime.alloc_raw(std::mem::size_of::<u32>())?)
        } else {
            None
        };
        Ok(Box::new(BlockQuantizedMatMulKernel {
            runtime: self.runtime.clone(),
            k,
            n,
            format,
            constant_inputs: [false; INPUT_COUNT],
            validation_scratch,
            validated_bank: Mutex::new(None),
        }))
    }
}

#[derive(Clone, Copy, Debug)]
enum MatMulFormat {
    Interleaved(BlockFormat),
    Planar(PlanarBlockGeometry),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PlanarBankIdentity {
    packed: CUdeviceptr,
    scale: CUdeviceptr,
}

#[derive(Debug)]
struct BlockQuantizedMatMulKernel {
    runtime: Arc<CudaRuntime>,
    k: usize,
    n: usize,
    format: MatMulFormat,
    constant_inputs: [bool; INPUT_COUNT],
    validation_scratch: Option<CUdeviceptr>,
    validated_bank: Mutex<Option<PlanarBankIdentity>>,
}

impl Kernel for BlockQuantizedMatMulKernel {
    fn set_constant_inputs(&mut self, constant_inputs: &[bool]) {
        for (index, value) in constant_inputs
            .iter()
            .copied()
            .enumerate()
            .take(INPUT_COUNT)
        {
            self.constant_inputs[index] = value;
        }
    }

    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if inputs.len() != INPUT_COUNT || outputs.len() != 1 {
            return Err(error(format!(
                "expected exactly {INPUT_COUNT} inputs and 1 output, got {} inputs and {} outputs",
                inputs.len(),
                outputs.len()
            )));
        }
        if let MatMulFormat::Planar(geometry) = self.format {
            return self.execute_planar(inputs, outputs, geometry);
        }
        let MatMulFormat::Interleaved(format) = self.format else {
            unreachable!()
        };
        if !inputs[BQMM_SCALE].is_absent() {
            return Err(error("aux_scale_B must be omitted for interleaved format"));
        }
        require_dtype("A", inputs[0].dtype, DataType::Float32)?;
        require_dtype("packed_B", inputs[1].dtype, DataType::Uint8)?;
        require_dtype("Y", outputs[0].dtype, DataType::Float32)?;

        let (m, blocks) = validate_tensor_layouts(
            inputs[0].shape,
            inputs[1].shape,
            outputs[0].shape,
            self.k,
            self.n,
            format,
        )?;
        let bias = inputs.get(BQMM_BIAS).filter(|input| !input.is_absent());
        if let Some(bias) = bias {
            require_dtype("bias", bias.dtype, DataType::Float32)?;
            require_shape("bias", bias.shape, &[self.n])?;
            checked_tensor_layout("bias", bias.shape, DataType::Float32)?;
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

        let k = as_i32("K", self.k)?;
        let n = as_i32("N", self.n)?;
        let grid_x = as_grid_x("N", n)?;
        let blocks = as_i32("block count", blocks)?;
        let block_bytes = as_i32("block byte count", format.block_bytes())?;
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
        if m == 0 {
            return Ok(());
        }

        let activation_ptr = cuptr(inputs[0].data_ptr::<u8>() as *const c_void);
        let packed_ptr = cuptr(inputs[1].data_ptr::<u8>() as *const c_void);
        let bias_ptr = bias
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let output_ptr = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let format = format.kernel_id();

        if m == 1 {
            let function = self
                .runtime
                .nvrtc_function(GEMV_MODULE, gemv_src(), GEMV_ENTRY)?;
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
                    grid_dim: (grid_x, 1, 1),
                    block_dim: (BLOCK_THREADS, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|err| driver_err("launch BlockQuantizedMatMul GEMV", err))?;
        } else {
            let function = self
                .runtime
                .nvrtc_function(GEMM_MODULE, gemm_src(), GEMM_ENTRY)?;
            let m = as_u64("M", m)?;
            let launch_config = gemm_launch_config(m, grid_x)?;
            let mut builder = self.runtime.stream().launch_builder(&function);
            builder
                .arg(&activation_ptr)
                .arg(&packed_ptr)
                .arg(&bias_ptr)
                .arg(&output_ptr)
                .arg(&m)
                .arg(&k)
                .arg(&n)
                .arg(&blocks)
                .arg(&block_bytes)
                .arg(&format);
            // SAFETY: all tensors are dense and shape-checked, and the scalar ABI
            // matches `block_quantized_matmul_gemm_f32`.
            unsafe { builder.launch(launch_config) }
                .map_err(|err| driver_err("launch BlockQuantizedMatMul GEMM", err))?;
        }
        self.runtime.synchronize()
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        false
    }

    fn capture_support(&self) -> onnx_runtime_ep_api::CaptureSupport {
        match self.format {
            MatMulFormat::Planar(_) => onnx_runtime_ep_api::CaptureSupport::Supported,
            MatMulFormat::Interleaved(_) => onnx_runtime_ep_api::CaptureSupport::unsupported(
                "interleaved block-quantized MatMul performs a trailing host stream synchronization",
            ),
        }
    }
}

impl BlockQuantizedMatMulKernel {
    fn execute_planar(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        geometry: PlanarBlockGeometry,
    ) -> Result<()> {
        for &index in &[BQMM_ACTIVATION, BQMM_WEIGHT, BQMM_SCALE] {
            if inputs[index].is_absent() {
                return Err(error(format!(
                    "required input {index} ('{}') is absent",
                    INPUT_NAMES[index]
                )));
            }
        }
        if !self.constant_inputs[BQMM_WEIGHT] || !self.constant_inputs[BQMM_SCALE] {
            return Err(error(
                "planar packed_B and aux_scale_B must be immutable session constants",
            ));
        }
        let dtype = PlanarActivationDtype::from_data_type(inputs[BQMM_ACTIVATION].dtype)?;
        if outputs[0].dtype != inputs[BQMM_ACTIVATION].dtype {
            return Err(error(format!(
                "Y dtype {:?} must match A dtype {:?}",
                outputs[0].dtype, inputs[BQMM_ACTIVATION].dtype
            )));
        }
        require_dtype(
            "packed_B",
            inputs[BQMM_WEIGHT].dtype,
            geometry.format.weight_dtype(),
        )?;
        require_dtype(
            "aux_scale_B",
            inputs[BQMM_SCALE].dtype,
            geometry.format.scale_dtype(),
        )?;
        let a_shape = inputs[BQMM_ACTIVATION].shape;
        if a_shape.is_empty() || a_shape[a_shape.len() - 1] != self.k {
            return Err(error(format!(
                "A must have rank >= 1 and last dimension K={}, got {a_shape:?}",
                self.k
            )));
        }
        let m = checked_product(&a_shape[..a_shape.len() - 1], "A leading dimension product")?;
        let expected_output = [&a_shape[..a_shape.len() - 1], &[self.n]].concat();
        require_shape("Y", outputs[0].shape, &expected_output)?;
        let dims = PlanarLinearDims {
            format: geometry.format.kernel_id(),
            m_rows: m,
            in_features: self.k,
            out_features: self.n,
            bs0: geometry.block_out,
            bs1: geometry.block_in,
        };
        let lengths = dims.expected_lengths()?;
        let expected_weight = [self.n, self.k / geometry.format.pack_factor()];
        require_shape("packed_B", inputs[BQMM_WEIGHT].shape, &expected_weight)?;
        let expected_scale = [
            self.n.div_ceil(geometry.block_out),
            self.k.div_ceil(geometry.block_in),
        ];
        require_shape("aux_scale_B", inputs[BQMM_SCALE].shape, &expected_scale)?;
        if inputs[BQMM_WEIGHT].byte_size() != lengths.packed_bytes
            || inputs[BQMM_SCALE].byte_size() != lengths.scale_bytes
        {
            return Err(error("planar weight or scale byte extent mismatch"));
        }
        let bias = inputs.get(BQMM_BIAS).filter(|input| !input.is_absent());
        if let Some(bias) = bias {
            require_dtype("bias", bias.dtype, inputs[BQMM_ACTIVATION].dtype)?;
            require_shape("bias", bias.shape, &[self.n])?;
        }
        for (name, contiguous) in [
            ("A", inputs[BQMM_ACTIVATION].is_contiguous()),
            ("packed_B", inputs[BQMM_WEIGHT].is_contiguous()),
            ("aux_scale_B", inputs[BQMM_SCALE].is_contiguous()),
            ("bias", bias.is_none_or(TensorView::is_contiguous)),
            ("Y", outputs[0].is_contiguous()),
        ] {
            if !contiguous {
                return Err(error(format!(
                    "{name} must be contiguous on the CUDA execution provider"
                )));
            }
        }

        let packed = cuptr(inputs[BQMM_WEIGHT].data_ptr::<u8>() as *const c_void);
        let scale = cuptr(inputs[BQMM_SCALE].data_ptr::<u8>() as *const c_void);
        let identity = PlanarBankIdentity { packed, scale };
        {
            let mut validated = self
                .validated_bank
                .lock()
                .map_err(|_| error("planar bank validation state is poisoned"))?;
            if *validated != Some(identity) {
                if self.runtime.is_capturing()? {
                    return Err(error(
                        "planar bank must be admitted before CUDA graph capture",
                    ));
                }
                validate_planar_bank_device(
                    &self.runtime,
                    &dims,
                    1,
                    packed,
                    scale,
                    self.validation_scratch
                        .ok_or_else(|| error("planar validation scratch is missing"))?,
                )?;
                *validated = Some(identity);
            }
        }
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
        if m == 0 {
            return Ok(());
        }
        let activation = cuptr(inputs[BQMM_ACTIVATION].data_ptr::<u8>() as *const c_void);
        let bias = bias
            .map(|tensor| cuptr(tensor.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let output = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let pointers = PlanarLinearPointers {
            activation,
            packed,
            scale,
            bias,
            output,
        };
        // SAFETY: shape, dtype, byte extent, value admission, and immutable
        // pointer identity were all established above.
        unsafe { launch_planar_linear_borrowed(&self.runtime, dtype, &dims, &pointers) }
    }
}

impl Drop for BlockQuantizedMatMulKernel {
    fn drop(&mut self) {
        if let Some(scratch) = self.validation_scratch.take() {
            // SAFETY: this kernel uniquely owns the raw validation word.
            let _ = unsafe { self.runtime.free_raw(scratch) };
        }
    }
}

pub(crate) fn unsupported_reason(
    node: &Node,
    shapes: &[Shape],
    dtypes: &[DataType],
) -> Option<Cow<'static, str>> {
    let reject = |message: String| Some(Cow::Owned(format!("{OP}: {message}")));
    if node.inputs.len() != INPUT_COUNT
        || shapes.len() != INPUT_COUNT
        || dtypes.len() != INPUT_COUNT
    {
        return reject(format!(
            "expected exactly {INPUT_COUNT} positional inputs and matching metadata"
        ));
    }
    if let Err(message) = require_layout_v1(node, OP) {
        return Some(Cow::Owned(message));
    }
    let k = match required_positive_attr(node, "K") {
        Ok(value) => value,
        Err(error) => return Some(Cow::Owned(error.to_string())),
    };
    let n = match required_positive_attr(node, "N") {
        Ok(value) => value,
        Err(error) => return Some(Cow::Owned(error.to_string())),
    };
    let geometry =
        match planar_geometry_from_node(node, OP, "format", "block_size_out", "block_size_in") {
            Ok(value) => value,
            Err(message) => return Some(Cow::Owned(message)),
        };
    for &index in &[BQMM_ACTIVATION, BQMM_WEIGHT] {
        if node.inputs[index].is_none() {
            return reject(format!(
                "required input {index} ('{}') is omitted",
                INPUT_NAMES[index]
            ));
        }
    }
    let planar = geometry.is_some();
    if node.inputs[BQMM_SCALE].is_some() != planar {
        return reject(if planar {
            "aux_scale_B is required for planar format".into()
        } else {
            "aux_scale_B must be omitted for interleaved format".into()
        });
    }
    let activation_dtypes: &[DataType] = if planar {
        &[DataType::Float32, DataType::Float16, DataType::BFloat16]
    } else {
        &[DataType::Float32]
    };
    if !activation_dtypes.contains(&dtypes[BQMM_ACTIVATION]) {
        return reject(format!(
            "A dtype {:?} is unsupported",
            dtypes[BQMM_ACTIVATION]
        ));
    }
    if let Some(geometry) = geometry {
        if dtypes[BQMM_WEIGHT] != geometry.format.weight_dtype()
            || dtypes[BQMM_SCALE] != geometry.format.scale_dtype()
        {
            return reject(format!(
                "planar weight/scale dtypes must be {:?}/{:?}",
                geometry.format.weight_dtype(),
                geometry.format.scale_dtype()
            ));
        }
        if shapes[BQMM_WEIGHT].len() != 2 || shapes[BQMM_SCALE].len() != 2 {
            return reject("planar weight and scale must both have rank 2".into());
        }
        let dims = PlanarLinearDims {
            format: geometry.format.kernel_id(),
            m_rows: 1,
            in_features: k,
            out_features: n,
            bs0: geometry.block_out,
            bs1: geometry.block_in,
        };
        if let Err(error) = dims.expected_lengths() {
            return Some(Cow::Owned(error.to_string()));
        }
        for (index, axis, expected) in [
            (BQMM_WEIGHT, 0, n),
            (BQMM_WEIGHT, 1, k / geometry.format.pack_factor()),
            (BQMM_SCALE, 0, n.div_ceil(geometry.block_out)),
            (BQMM_SCALE, 1, k.div_ceil(geometry.block_in)),
        ] {
            if let Some(actual) = shapes[index][axis].as_static()
                && actual != expected
            {
                return reject(format!(
                    "{} shape axis {axis} must be {expected}, got {actual}",
                    INPUT_NAMES[index]
                ));
            }
        }
    } else {
        if dtypes[BQMM_WEIGHT] != DataType::Uint8 {
            return reject("interleaved packed_B must be Uint8".into());
        }
        let text = node
            .attr("format")
            .and_then(|attribute| attribute.as_str())
            .unwrap_or("");
        if BlockFormat::parse(text).is_err() {
            return reject(format!("CUDA does not support format '{text}'"));
        }
    }
    if node.inputs[BQMM_BIAS].is_some() && dtypes[BQMM_BIAS] != dtypes[BQMM_ACTIVATION] {
        return reject("bias dtype must match A".into());
    }
    None
}

fn required_positive_attr(node: &Node, name: &str) -> Result<usize> {
    let value = optional_int_attr(node, name)?
        .ok_or_else(|| error(format!("missing required integer attribute '{name}'")))?;
    if value <= 0 {
        return Err(error(format!(
            "attribute '{name}' must be positive, got {value}"
        )));
    }
    usize::try_from(value)
        .map_err(|_| error(format!("attribute '{name}'={value} exceeds usize limits")))
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

fn validate_tensor_layouts(
    a_shape: &[usize],
    packed_shape: &[usize],
    output_shape: &[usize],
    k: usize,
    n: usize,
    format: BlockFormat,
) -> Result<(usize, usize)> {
    if a_shape.is_empty() || a_shape[a_shape.len() - 1] != k {
        return Err(error(format!(
            "A must have rank >= 1 and last dimension K={k}, got {a_shape:?}"
        )));
    }
    let m = checked_product(&a_shape[..a_shape.len() - 1], "A leading dimension product")?;
    let expected_output_shape = [&a_shape[..a_shape.len() - 1], &[n]].concat();
    require_shape("Y", output_shape, &expected_output_shape)?;

    let blocks = checked_div_ceil(k, format.qk(), "block count")?;
    require_shape("packed_B", packed_shape, &[n, blocks, format.block_bytes()])?;

    checked_tensor_layout("A", a_shape, DataType::Float32)?;
    checked_tensor_layout("packed_B", packed_shape, DataType::Uint8)?;
    checked_tensor_layout("Y", output_shape, DataType::Float32)?;
    Ok((m, blocks))
}

fn checked_product(factors: &[usize], context: &str) -> Result<usize> {
    let mut product = 1usize;
    let mut has_zero = false;
    for &factor in factors {
        if factor == 0 {
            has_zero = true;
        } else {
            product = product
                .checked_mul(factor)
                .ok_or_else(|| error(format!("{context} exceeds usize limits")))?;
        }
    }
    Ok(if has_zero { 0 } else { product })
}

fn checked_tensor_layout(name: &str, shape: &[usize], dtype: DataType) -> Result<usize> {
    let elements = checked_product(shape, &format!("{name} element count"))?;
    let bytes = elements
        .checked_mul(dtype.byte_size())
        .ok_or_else(|| error(format!("{name} byte count exceeds usize limits")))?;
    if bytes > isize::MAX as usize {
        return Err(error(format!(
            "{name} byte count {bytes} exceeds isize::MAX"
        )));
    }
    Ok(elements)
}

fn checked_div_ceil(value: usize, divisor: usize, context: &str) -> Result<usize> {
    value
        .checked_add(divisor - 1)
        .map(|adjusted| adjusted / divisor)
        .ok_or_else(|| error(format!("{context} exceeds usize limits")))
}

fn gemm_launch_config(m: u64, grid_x: u32) -> Result<LaunchConfig> {
    let row_tiles = m.div_ceil(u64::from(GEMM_TILE_M));
    let grid_y = u32::try_from(row_tiles.min(u64::from(GEMM_GRID_DIM_Y_CAP))).map_err(|_| {
        error(format!(
            "GEMM row-tile count {row_tiles} exceeds u32 limits"
        ))
    })?;
    Ok(LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (BLOCK_THREADS, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn as_i32(name: &str, value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| error(format!("{name}={value} exceeds CUDA i32 limits")))
}

fn as_grid_x(name: &str, value: i32) -> Result<u32> {
    u32::try_from(value).map_err(|_| error(format!("{name}={value} exceeds CUDA grid-X limits")))
}

fn as_u64(name: &str, value: usize) -> Result<u64> {
    u64::try_from(value).map_err(|_| error(format!("{name}={value} exceeds CUDA u64 limits")))
}

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("cuda_ep {DOMAIN}::{OP}: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{Attribute, NodeId, ValueId, static_shape};

    #[test]
    fn placement_decline_names_unsupported_format_and_fix() {
        let mut node = Node::new(
            NodeId(0),
            "BlockQuantizedMatMul",
            vec![Some(ValueId(0)), Some(ValueId(1)), None, None],
            vec![],
        );
        node.domain = "pkg.nxrt".into();
        node.attributes
            .insert("format".into(), Attribute::String(b"q4_0".to_vec()));
        node.attributes.insert("K".into(), Attribute::Int(32));
        node.attributes.insert("N".into(), Attribute::Int(1));
        node.attributes
            .insert("block_layout_version".into(), Attribute::Int(1));

        let reason = unsupported_reason(
            &node,
            &[
                static_shape([1, 32]),
                static_shape([1, 1, 18]),
                vec![],
                vec![],
            ],
            &[
                DataType::Float32,
                DataType::Uint8,
                DataType::Undefined,
                DataType::Undefined,
            ],
        )
        .expect("q4_0 must be declined");
        assert!(reason.contains("q4_0"), "{reason}");
        assert!(reason.contains("does not support"), "{reason}");
    }

    #[test]
    fn gemm_launch_config_caps_grid_y_and_keeps_all_row_tiles_reachable() {
        let row_tiles = u64::from(GEMM_GRID_DIM_Y_CAP) + 1;
        let m = (row_tiles - 1) * u64::from(GEMM_TILE_M) + 1;
        let config = gemm_launch_config(m, 7).unwrap();

        assert_eq!(config.grid_dim, (7, GEMM_GRID_DIM_Y_CAP, 1));
        assert!(config.grid_dim.1 <= CUDA_MAX_GRID_DIM_Y);
        assert!(row_tiles > u64::from(config.grid_dim.1));
        let final_tile = row_tiles - 1;
        let starting_block = final_tile % u64::from(config.grid_dim.1);
        let stride_iteration = final_tile / u64::from(config.grid_dim.1);
        assert_eq!(starting_block, 0);
        assert_eq!(stride_iteration, 1);
    }

    #[test]
    fn zero_leading_dimension_does_not_hide_nonzero_product_overflow() {
        let result = validate_tensor_layouts(
            &[0, usize::MAX, 2, 32],
            &[1, 1, MXFP4_BLOCK_BYTES],
            &[0, usize::MAX, 2, 1],
            32,
            1,
            BlockFormat::Mxfp4,
        );

        let error = result.unwrap_err().to_string();
        assert!(error.contains("A leading dimension product exceeds usize limits"));
    }

    #[test]
    fn legitimate_empty_tensor_layout_is_valid() {
        let result = validate_tensor_layouts(
            &[0, 32],
            &[3, 1, MXFP4_BLOCK_BYTES],
            &[0, 3],
            32,
            3,
            BlockFormat::Mxfp4,
        );

        assert_eq!(result.unwrap(), (0, 1));
    }

    #[test]
    fn tensor_layouts_reject_byte_counts_above_isize_max() {
        let oversized_m = isize::MAX as usize / std::mem::size_of::<f32>() + 1;
        let a_error = validate_tensor_layouts(
            &[oversized_m, 1],
            &[1, 1, MXFP4_BLOCK_BYTES],
            &[oversized_m, 1],
            1,
            1,
            BlockFormat::Mxfp4,
        )
        .unwrap_err()
        .to_string();
        assert!(a_error.contains("A byte count"));
        assert!(a_error.contains("exceeds isize::MAX"));

        let oversized_n = isize::MAX as usize / MXFP4_BLOCK_BYTES + 1;
        let packed_error = validate_tensor_layouts(
            &[0, 32],
            &[oversized_n, 1, MXFP4_BLOCK_BYTES],
            &[0, oversized_n],
            32,
            oversized_n,
            BlockFormat::Mxfp4,
        )
        .unwrap_err()
        .to_string();
        assert!(packed_error.contains("packed_B byte count"));
        assert!(packed_error.contains("exceeds isize::MAX"));

        let output_n = isize::MAX as usize / 20 + 1;
        let output_error = validate_tensor_layouts(
            &[5, 1],
            &[output_n, 1, MXFP4_BLOCK_BYTES],
            &[5, output_n],
            1,
            output_n,
            BlockFormat::Mxfp4,
        )
        .unwrap_err()
        .to_string();
        assert!(output_error.contains("Y byte count"));
        assert!(output_error.contains("exceeds isize::MAX"));
    }
}

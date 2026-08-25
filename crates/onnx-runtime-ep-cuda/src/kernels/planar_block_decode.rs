//! Exact-format CUDA **device execution** for the DeepSeek-V4 planar B2 weight
//! formats (`block_fp8`, `fp4_planar`) — the runtime slice stacked on the CPU
//! planar ABI (`onnx_runtime_ep_cpu::kernels::planar_block_quant`) and the
//! host-mirror decode (`#2068`).
//!
//! ## What this slice delivers
//!
//! A **launched** NVRTC kernel that decodes the two planar formats directly from
//! their packed weights + UE8M0 aux scales on the device and contracts them
//! against `f32` / `f16` / `bf16` activations (`C[M,N] = A[M,K]·W[K,N]`). No host
//! mirror, dequantize-to-full-copy, or transcode runs in the launch path: the
//! kernel consumes the exact on-disk byte layout. A companion [`PlanarLinear`]
//! launcher warm-compiles the module outside any CUDA-graph capture and launches
//! it **without a trailing synchronize**, so a warmed fixed-shape planar linear
//! records cleanly into a captured segment.
//!
//! The device arithmetic is kept adjacent to a **bit-identical host Rust mirror**
//! (`mirror_*`, below). The mirror is proven equal to the CPU oracle
//! (`onnx_runtime_ep_cpu::kernels::planar_block_quant`) by the unit tests here
//! *without a GPU*, and the on-device kernel is proven equal to the CPU oracle by
//! the GPU integration test (`tests/planar_block_decode_gpu.rs`). The mirror
//! therefore stays as the inspection anchor between the two.
//!
//! ## Claim boundary (still honest)
//!
//! This slice proves and advertises the **matmul primitive** only:
//! [`planar_matmul_capable_formats`] reports `block_fp8` / `fp4_planar` once the
//! GPU parity test passes on hardware. The mixed-projection **routed-MoE** CUDA
//! path in [`super::block_quantized_moe`] still *typed-rejects* these formats:
//! there is no proven routed top-k kernel yet, so it may not claim. Wiring an
//! op-level `BlockQuantizedMatMul` planar node (a new aux-scale input) is the
//! explicit next slice — the exporter (Mobius #602/#593) is co-designing the node
//! ABI and cannot emit planar nodes until this capability is advertised.
//!
//! ## Reserved-code contract
//!
//! The CPU oracle's `decode_element` *fail-closes* on reserved E4M3 NaN codes and
//! reserved UE8M0 `0xff` scale exponents. The device decode here mirrors only the
//! *arithmetic*, so a reserved code decodes to a propagating `NaN` rather than a
//! typed error. That is safe while the routed-MoE claim gate typed-rejects these
//! formats and the parity fixtures contain no reserved codes.
//! [`validate_planar_linear`] checks tensor *lengths/shapes* against the logical
//! dims (typed-rejecting ragged/overflowing aux banks) but, like the CPU
//! `PlanarLayout::validate_tensors`, not individual codes; a routed-MoE claim
//! must add reserved-code validation before it flips.
//!
//! ## Formats
//!
//! * `block_fp8` (`format = 0`): `F8_E4M3` weight, logical `[out, in]`, one byte
//!   per element, paired with a 2D `F8_E8M0` block scale
//!   `[ceil(out/bs0), ceil(in/bs1)]` (DeepSeek uses `bs0 = bs1 = 128`). Scale
//!   index `(out_row / bs0) * ceil(in / bs1) + (in_col / bs1)`.
//! * `fp4_planar` (`format = 1`): `I8` weight storing two `E2M1` nibbles per byte
//!   (packed `[out, in/2]`, low nibble = even `in_col`, high = odd), paired with
//!   a 1D `F8_E8M0` block-32 micro-scale `[out, in/32]`. Scale index
//!   `out_row * (in / 32) + (in_col / 32)`.
//!
//! Both decode to the runtime's `[K = in, N = out]` (`in`-major) orientation so
//! `C[M, N] = A[M, K] * W[K, N]`.

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Result};
use onnx_runtime_ir::DataType;

use crate::error::driver_err;
use crate::runtime::CudaRuntime;

/// Logical input elements per UE8M0 micro-scale in `fp4_planar` (MXFP4 block-32).
pub(crate) const FP4_MICROSCALE_BLOCK: usize = 32;

/// Logical `fp4_planar` weight elements packed per stored `I8` byte (two nibbles).
pub(crate) const FP4_PACK_FACTOR: usize = 2;

/// `format` selector shared by the device kernel and the host mirror.
pub const PLANAR_FORMAT_BLOCK_FP8: i32 = 0;
pub const PLANAR_FORMAT_FP4_PLANAR: i32 = 1;

/// NVRTC module cache key for the planar linear kernels.
pub(crate) const PLANAR_LINEAR_MODULE: &str = "planar_block_decode_linear_v1";

/// NVRTC device source: exact planar decode plus a straight `C[M,N]=A·W` linear
/// kernel templated over the activation dtype (`f32`/`f16`/`bf16`). Decode and
/// accumulation are always in `f32`; only the activation load / result store use
/// the requested precision.
pub(crate) const PLANAR_BLOCK_DECODE_CUH: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
// E2M1 value LUT, sign bit included (index 8 is -0.0). Matches the CPU
// onnx_runtime_ep_cpu::kernels::block_dequant E2M1 table bit-for-bit.
__device__ __constant__ float planar_e2m1_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};
// UE8M0 power-of-two scale: 0xff reserved (NaN), 0 -> 2^-127, else 2^(e-127).
// Matches onnx_runtime_ep_cpu::kernels::block_dequant::decode_e8m0_scale.
__device__ __forceinline__ float planar_e8m0_scale(unsigned char e) {
    if (e == 0xffu) return __uint_as_float(0x7fc00000u);
    if (e == 0u)    return __uint_as_float(0x00400000u);
    return __uint_as_float((unsigned int)e << 23);
}
__device__ __forceinline__ float planar_e2m1(unsigned char code) {
    return planar_e2m1_lut[code & 15u];
}
// E4M3FN: exp==15 && mant==7 reserved (NaN); subnormal m*2^-9; normal
// (1+m/8)*2^(e-7). Matches decode_e4m3fn.
__device__ __forceinline__ float planar_e4m3(unsigned char code) {
    const float sign = (code & 0x80u) ? -1.0f : 1.0f;
    const unsigned int e = (code >> 3) & 15u;
    const unsigned int m = code & 7u;
    if (e == 15u && m == 7u) return __uint_as_float(0x7fc00000u);
    return sign * (e == 0u ? (float)m * 0x1p-9f
                           : (1.0f + (float)m / 8.0f) * exp2f((int)e - 7));
}
__device__ __forceinline__ float planar_bf8_element(
    const unsigned char* packed, const unsigned char* scale,
    int out_features, int in_features, int bs0, int bs1,
    int out_row, int in_col) {
    const int scale_cols = 1 + (in_features - 1) / bs1;
    const unsigned char se = scale[(long long)(out_row / bs0) * scale_cols + (in_col / bs1)];
    const unsigned char code = packed[(long long)out_row * in_features + in_col];
    return planar_e4m3(code) * planar_e8m0_scale(se);
}
__device__ __forceinline__ float planar_fp4_element(
    const unsigned char* packed, const unsigned char* scale,
    int out_features, int in_features,
    int out_row, int in_col) {
    const int scale_cols = in_features / 32;
    const unsigned char se = scale[(long long)out_row * scale_cols + (in_col / 32)];
    const unsigned char byte = packed[(long long)out_row * (in_features / 2) + (in_col / 2)];
    const unsigned char nib = (in_col & 1) ? (byte >> 4) : (byte & 0x0fu);
    return planar_e2m1(nib) * planar_e8m0_scale(se);
}
// Activation load / result store helpers so one templated body serves every
// activation precision. Decode + accumulation stay in f32 regardless.
__device__ __forceinline__ float planar_to_f32(float v) { return v; }
__device__ __forceinline__ float planar_to_f32(__half v) { return __half2float(v); }
__device__ __forceinline__ float planar_to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ void planar_store(float* out, float v) { *out = v; }
__device__ __forceinline__ void planar_store(__half* out, float v) { *out = __float2half_rn(v); }
__device__ __forceinline__ void planar_store(__nv_bfloat16* out, float v) { *out = __float2bfloat16(v); }
// C[M,N] = A[M,K] * W[K,N]; W decoded per (out_row = n, in_col = k). One thread
// per (row, col). format: 0 = block_fp8, 1 = fp4_planar.
template<typename T>
__device__ __forceinline__ void planar_linear_impl(
    const T* a, const unsigned char* packed, const unsigned char* scale,
    T* c, int m_rows, int in_features, int out_features,
    int format, int bs0, int bs1) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)m_rows * out_features) return;
    const int row = (int)(idx / out_features);
    const int col = (int)(idx % out_features);
    const T* a_row = a + (long long)row * in_features;
    float acc = 0.0f;
    for (int k = 0; k < in_features; ++k) {
        const float w = (format == 0)
            ? planar_bf8_element(packed, scale, out_features, in_features, bs0, bs1, col, k)
            : planar_fp4_element(packed, scale, out_features, in_features, col, k);
        acc += planar_to_f32(a_row[k]) * w;
    }
    planar_store(&c[(long long)row * out_features + col], acc);
}
extern "C" __global__ void planar_linear_f32(
    const float* a, const unsigned char* packed, const unsigned char* scale,
    float* c, int m_rows, int in_features, int out_features,
    int format, int bs0, int bs1) {
    planar_linear_impl<float>(a, packed, scale, c, m_rows, in_features,
                              out_features, format, bs0, bs1);
}
extern "C" __global__ void planar_linear_f16(
    const __half* a, const unsigned char* packed, const unsigned char* scale,
    __half* c, int m_rows, int in_features, int out_features,
    int format, int bs0, int bs1) {
    planar_linear_impl<__half>(a, packed, scale, c, m_rows, in_features,
                               out_features, format, bs0, bs1);
}
extern "C" __global__ void planar_linear_bf16(
    const __nv_bfloat16* a, const unsigned char* packed, const unsigned char* scale,
    __nv_bfloat16* c, int m_rows, int in_features, int out_features,
    int format, int bs0, int bs1) {
    planar_linear_impl<__nv_bfloat16>(a, packed, scale, c, m_rows, in_features,
                                      out_features, format, bs0, bs1);
}
"#;

/// Entry-point name of the `f32` planar linear kernel in [`PLANAR_BLOCK_DECODE_CUH`].
#[allow(dead_code)]
pub(crate) const PLANAR_LINEAR_ENTRY: &str = "planar_linear_f32";

/// Activation precision the planar linear kernel is launched for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanarActivationDtype {
    F32,
    F16,
    Bf16,
}

impl PlanarActivationDtype {
    /// The `extern "C"` entry point implementing this precision.
    pub fn entry(self) -> &'static str {
        match self {
            PlanarActivationDtype::F32 => "planar_linear_f32",
            PlanarActivationDtype::F16 => "planar_linear_f16",
            PlanarActivationDtype::Bf16 => "planar_linear_bf16",
        }
    }

    /// Every entry point (used to warm-compile the whole module before capture).
    pub fn all() -> [PlanarActivationDtype; 3] {
        [
            PlanarActivationDtype::F32,
            PlanarActivationDtype::F16,
            PlanarActivationDtype::Bf16,
        ]
    }

    /// Map an IR activation dtype onto a planar kernel precision, or typed-reject.
    pub fn from_data_type(dtype: DataType) -> Result<PlanarActivationDtype> {
        match dtype {
            DataType::Float32 => Ok(PlanarActivationDtype::F32),
            DataType::Float16 => Ok(PlanarActivationDtype::F16),
            DataType::BFloat16 => Ok(PlanarActivationDtype::Bf16),
            other => Err(EpError::KernelFailed(format!(
                "cuda_ep planar linear: unsupported activation dtype {other:?}; \
                 only f32/f16/bf16 have a proven planar kernel"
            ))),
        }
    }
}

/// Logical geometry of a single planar linear `C[M,N] = A[M,K]·W[K,N]`.
#[derive(Clone, Copy, Debug)]
pub struct PlanarLinearDims {
    /// `PLANAR_FORMAT_BLOCK_FP8` or `PLANAR_FORMAT_FP4_PLANAR`.
    pub format: i32,
    /// Rows of the activation (`M`).
    pub m_rows: usize,
    /// Contraction dimension (`K = in_features`).
    pub in_features: usize,
    /// Output dimension (`N = out_features`).
    pub out_features: usize,
    /// Row block size of the 2D `block_fp8` scale (`bs0`); ignored by `fp4_planar`.
    pub bs0: usize,
    /// Column block size of the 2D `block_fp8` scale (`bs1`); ignored by `fp4_planar`.
    pub bs1: usize,
}

fn kernel_err(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("cuda_ep planar linear: {}", message.into()))
}

impl PlanarLinearDims {
    /// Exact packed-weight and aux-scale byte lengths this geometry requires,
    /// typed-rejecting any geometry the kernel cannot decode (non-positive dims,
    /// odd/unaligned `fp4_planar` contraction, zero `block_fp8` block sizes, or a
    /// size that overflows the kernel's `i32` scalar ABI).
    pub fn expected_lengths(&self) -> Result<PlanarTensorLengths> {
        if self.m_rows == 0 || self.in_features == 0 || self.out_features == 0 {
            return Err(kernel_err(format!(
                "non-positive dims M={} K={} N={}",
                self.m_rows, self.in_features, self.out_features
            )));
        }
        // The device kernel takes i32 dims and computes `m_rows * out_features`
        // thread indices as i64; reject anything that would not fit its ABI.
        for (label, value) in [
            ("M", self.m_rows),
            ("K", self.in_features),
            ("N", self.out_features),
        ] {
            if i32::try_from(value).is_err() {
                return Err(kernel_err(format!(
                    "{label}={value} exceeds the i32 kernel ABI"
                )));
            }
        }

        match self.format {
            PLANAR_FORMAT_BLOCK_FP8 => {
                if self.bs0 == 0 || self.bs1 == 0 {
                    return Err(kernel_err(format!(
                        "block_fp8 requires bs0>0 and bs1>0, got bs0={} bs1={}",
                        self.bs0, self.bs1
                    )));
                }
                for (label, value) in [("bs0", self.bs0), ("bs1", self.bs1)] {
                    if i32::try_from(value).is_err() {
                        return Err(kernel_err(format!(
                            "{label}={value} exceeds the i32 kernel ABI"
                        )));
                    }
                }
                let scale_rows = self.out_features.div_ceil(self.bs0);
                let scale_cols = self.in_features.div_ceil(self.bs1);
                Ok(PlanarTensorLengths {
                    packed_bytes: self.out_features * self.in_features,
                    scale_bytes: scale_rows * scale_cols,
                    output_elems: self.m_rows * self.out_features,
                })
            }
            PLANAR_FORMAT_FP4_PLANAR => {
                if !self.in_features.is_multiple_of(FP4_PACK_FACTOR) {
                    return Err(kernel_err(format!(
                        "fp4_planar requires an even contraction, got K={}",
                        self.in_features
                    )));
                }
                if !self.in_features.is_multiple_of(FP4_MICROSCALE_BLOCK) {
                    return Err(kernel_err(format!(
                        "fp4_planar requires K divisible by the block-{} micro-scale, got K={}",
                        FP4_MICROSCALE_BLOCK, self.in_features
                    )));
                }
                Ok(PlanarTensorLengths {
                    packed_bytes: self.out_features * (self.in_features / FP4_PACK_FACTOR),
                    scale_bytes: self.out_features * (self.in_features / FP4_MICROSCALE_BLOCK),
                    output_elems: self.m_rows * self.out_features,
                })
            }
            other => Err(kernel_err(format!("unknown planar format id {other}"))),
        }
    }
}

/// Exact tensor extents a planar linear's geometry requires: packed-weight and
/// aux-scale byte lengths, plus the destination element count the kernel writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanarTensorLengths {
    pub packed_bytes: usize,
    pub scale_bytes: usize,
    pub output_elems: usize,
}

/// Typed-reject a planar linear whose supplied tensor extents do not match the
/// exact extents its geometry requires. This is the host-side aux/OOB guard
/// (ragged banks, truncated scales, an under-sized output buffer, overflow) that
/// must pass before any launch.
pub fn validate_planar_linear(
    dims: &PlanarLinearDims,
    activation_elems: usize,
    packed_bytes: usize,
    scale_bytes: usize,
    output_elems: usize,
) -> Result<()> {
    let expected = dims.expected_lengths()?;
    let expected_activation = dims.m_rows * dims.in_features;
    if activation_elems != expected_activation {
        return Err(kernel_err(format!(
            "activation has {activation_elems} elements, expected M*K = {expected_activation}"
        )));
    }
    if packed_bytes != expected.packed_bytes {
        return Err(kernel_err(format!(
            "packed weight has {packed_bytes} bytes, expected {}",
            expected.packed_bytes
        )));
    }
    if scale_bytes != expected.scale_bytes {
        return Err(kernel_err(format!(
            "aux scale has {scale_bytes} bytes, expected {}",
            expected.scale_bytes
        )));
    }
    if output_elems != expected.output_elems {
        return Err(kernel_err(format!(
            "output has {output_elems} elements, expected M*N = {}",
            expected.output_elems
        )));
    }
    Ok(())
}

/// Warm-compile every planar linear entry point on `runtime`.
///
/// Must run **outside** a CUDA-graph capture: NVRTC compile/module-load
/// synchronizes the device. After this returns, [`PlanarLinear::launch`] loads a
/// cached function with no compile, so a warmed fixed-shape launch is
/// capture-safe.
pub fn warm_planar_linear(runtime: &CudaRuntime) -> Result<()> {
    runtime.require_nvrtc_half_headers("planar linear")?;
    for dtype in PlanarActivationDtype::all() {
        runtime.nvrtc_function(PLANAR_LINEAR_MODULE, PLANAR_BLOCK_DECODE_CUH, dtype.entry())?;
    }
    Ok(())
}

/// Device pointers for one planar linear launch. Every pointer is a live device
/// allocation the caller owns for the launch's duration.
#[derive(Clone, Copy, Debug)]
pub struct PlanarLinearPtrs {
    pub activation: CUdeviceptr,
    pub packed: CUdeviceptr,
    pub scale: CUdeviceptr,
    pub output: CUdeviceptr,
}

/// Threads per block for the one-thread-per-output planar linear kernel.
const PLANAR_LINEAR_BLOCK: u32 = 256;

/// Launch the planar linear kernel on `runtime`'s EP stream for `dtype`.
///
/// Geometry is re-validated here (defence in depth); tensor byte lengths must be
/// validated by the caller with [`validate_planar_linear`] before this call. The
/// launch issues **no** host synchronization and allocates nothing, so a warmed
/// signature records cleanly into a CUDA-graph capture. Ordering with a later
/// device→host read is guaranteed by the single in-order EP stream.
pub fn launch_planar_linear(
    runtime: &CudaRuntime,
    dtype: PlanarActivationDtype,
    dims: &PlanarLinearDims,
    ptrs: &PlanarLinearPtrs,
) -> Result<()> {
    dims.expected_lengths()?;
    let function =
        runtime.nvrtc_function(PLANAR_LINEAR_MODULE, PLANAR_BLOCK_DECODE_CUH, dtype.entry())?;

    let total = (dims.m_rows as u64) * (dims.out_features as u64);
    let grid_x = total.div_ceil(u64::from(PLANAR_LINEAR_BLOCK));
    let grid_x = u32::try_from(grid_x)
        .map_err(|_| kernel_err(format!("grid dimension {grid_x} exceeds u32")))?;

    let m_rows = dims.m_rows as i32;
    let in_features = dims.in_features as i32;
    let out_features = dims.out_features as i32;
    let format = dims.format;
    let (bs0, bs1) = if format == PLANAR_FORMAT_BLOCK_FP8 {
        (
            i32::try_from(dims.bs0)
                .map_err(|_| kernel_err(format!("bs0={} exceeds the i32 kernel ABI", dims.bs0)))?,
            i32::try_from(dims.bs1)
                .map_err(|_| kernel_err(format!("bs1={} exceeds the i32 kernel ABI", dims.bs1)))?,
        )
    } else {
        (0, 0)
    };

    let stream = runtime.stream();
    let mut builder = stream.launch_builder(&function);
    builder
        .arg(&ptrs.activation)
        .arg(&ptrs.packed)
        .arg(&ptrs.scale)
        .arg(&ptrs.output)
        .arg(&m_rows)
        .arg(&in_features)
        .arg(&out_features)
        .arg(&format)
        .arg(&bs0)
        .arg(&bs1);
    // SAFETY: the scalar ABI matches `planar_linear_<dtype>`; every pointer is a
    // live device allocation the caller sized to `expected_lengths` (validated
    // above and by `validate_planar_linear`).
    unsafe {
        builder.launch(LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (PLANAR_LINEAR_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map_err(|err| driver_err(&format!("launch {}", dtype.entry()), err))?;
    Ok(())
}

/// Planar matmul weight formats with a proven, launched CUDA kernel on this
/// build. These are the stable runtime capability strings the Mobius #602 /
/// Deckard #593 planar emitters probe: emitting a planar node is only correct
/// once the runtime advertises the matching format here.
///
/// Scope: the **matmul primitive** ([`launch_planar_linear`]), proven on device
/// against the CPU oracle. The mixed-projection routed-MoE path
/// ([`super::block_quantized_moe`]) still typed-rejects these formats — a routed
/// claim is a separate, unproven kernel and is deliberately not advertised here.
pub fn planar_matmul_capable_formats() -> &'static [&'static str] {
    &["block_fp8", "fp4_planar"]
}

// ---------------------------------------------------------------------------
// Host Rust mirror of the device arithmetic above (bit-identical formulae).
// Hand-written to match `PLANAR_BLOCK_DECODE_CUH` so the device algorithm can be
// verified against the CPU oracle without a GPU. NOT a call into the CPU crate:
// these are independent transcriptions of the CUDA-C, proven equal to the CPU
// oracle by the tests.
// ---------------------------------------------------------------------------

#[allow(dead_code)]
const PLANAR_E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Mirror of device `planar_e8m0_scale`.
#[allow(dead_code)]
pub(crate) fn mirror_e8m0_scale(exponent: u8) -> f32 {
    match exponent {
        0xff => f32::NAN,
        0 => f32::from_bits(0x0040_0000),
        _ => f32::from_bits((exponent as u32) << 23),
    }
}

/// Mirror of device `planar_e2m1`.
#[allow(dead_code)]
pub(crate) fn mirror_e2m1(code: u8) -> f32 {
    PLANAR_E2M1_LUT[usize::from(code & 15)]
}

/// Mirror of device `planar_e4m3`.
#[allow(dead_code)]
pub(crate) fn mirror_e4m3(code: u8) -> f32 {
    let sign = if code & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = u32::from((code >> 3) & 15);
    let m = u32::from(code & 7);
    if e == 15 && m == 7 {
        return f32::NAN;
    }
    let magnitude = if e == 0 {
        m as f32 * 2.0f32.powi(-9)
    } else {
        (1.0 + m as f32 / 8.0) * 2.0f32.powi(e as i32 - 7)
    };
    sign * magnitude
}

/// Mirror of device `planar_bf8_element`: decode one logical `(out_row, in_col)`
/// of a `block_fp8` weight. Indices transcribed from the CUDA-C.
#[allow(dead_code)]
pub(crate) fn mirror_bf8_element(
    packed: &[u8],
    scale: &[u8],
    in_features: usize,
    bs0: usize,
    bs1: usize,
    out_row: usize,
    in_col: usize,
) -> f32 {
    let scale_cols = in_features.div_ceil(bs1);
    let se = scale[(out_row / bs0) * scale_cols + (in_col / bs1)];
    let code = packed[out_row * in_features + in_col];
    mirror_e4m3(code) * mirror_e8m0_scale(se)
}

/// Mirror of device `planar_fp4_element`.
#[allow(dead_code)]
pub(crate) fn mirror_fp4_element(
    packed: &[u8],
    scale: &[u8],
    in_features: usize,
    out_row: usize,
    in_col: usize,
) -> f32 {
    let scale_cols = in_features / FP4_MICROSCALE_BLOCK;
    let se = scale[out_row * scale_cols + (in_col / FP4_MICROSCALE_BLOCK)];
    let byte = packed[out_row * (in_features / 2) + (in_col / 2)];
    let nibble = if in_col & 1 == 1 {
        byte >> 4
    } else {
        byte & 0x0f
    };
    mirror_e2m1(nibble) * mirror_e8m0_scale(se)
}

/// Mirror of device `planar_linear_f32`: `C[M,N] = A[M,K] * W[K,N]`, accumulating
/// over `k` per output element in the same order the GPU thread does (so the GPU
/// is expected to match this to the bit, and this to match the CPU oracle within
/// float tolerance).
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn mirror_planar_linear_f32(
    a: &[f32],
    packed: &[u8],
    scale: &[u8],
    m_rows: usize,
    in_features: usize,
    out_features: usize,
    format: i32,
    bs0: usize,
    bs1: usize,
) -> Vec<f32> {
    let mut c = vec![0.0f32; m_rows * out_features];
    for row in 0..m_rows {
        let a_row = &a[row * in_features..][..in_features];
        for col in 0..out_features {
            let mut acc = 0.0f32;
            for (k, &a_val) in a_row.iter().enumerate() {
                let w = if format == PLANAR_FORMAT_BLOCK_FP8 {
                    mirror_bf8_element(packed, scale, in_features, bs0, bs1, col, k)
                } else {
                    mirror_fp4_element(packed, scale, in_features, col, k)
                };
                acc += a_val * w;
            }
            c[row * out_features + col] = acc;
        }
    }
    c
}

#[cfg(test)]
mod tests;

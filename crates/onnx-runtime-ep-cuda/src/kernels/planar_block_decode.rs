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
//! [`admit_planar_linear`] validates immutable host weight/scale bytes through
//! the CPU planar oracle and uploads those exact bytes into private owned
//! allocations. It rejects E4M3FN `0x7f`/`0xff`, UE8M0 `0xff`, and every finite
//! code×scale pair whose decoded `f32` product overflows. The returned
//! [`AdmittedPlanarLinear`] is the only safe source of bank pointers for
//! [`launch_planar_linear`], so eager launch and graph replay never rescan, copy,
//! allocate, or synchronize for validation and malformed/substituted banks
//! cannot reach the device arithmetic.
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

use std::sync::Arc;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{DeviceBuffer, EpError, ExecutionProvider, Result};
use onnx_runtime_ep_cpu::kernels::planar_block_quant::{
    FP4_MICROSCALE_BLOCK as CPU_FP4_MICROSCALE_BLOCK, PlanarBankIdentity, PlanarBlockFormat,
    PlanarLayout, validate_planar_values,
};
use onnx_runtime_ir::DataType;
use onnx_runtime_memory_governor::ProviderContextIdentity;

use crate::error::driver_err;
use crate::provider::{CudaExecutionProvider, CudaSealedAllocation};
use crate::runtime::{CudaRuntime, cuptr};

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

    fn byte_size(self) -> usize {
        match self {
            PlanarActivationDtype::F32 => 4,
            PlanarActivationDtype::F16 | PlanarActivationDtype::Bf16 => 2,
        }
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
    fn cpu_layout(&self) -> Result<PlanarLayout> {
        let (format, block_out, block_in) = match self.format {
            PLANAR_FORMAT_BLOCK_FP8 => (PlanarBlockFormat::BlockFp8, self.bs0, self.bs1),
            PLANAR_FORMAT_FP4_PLANAR => (PlanarBlockFormat::Fp4Planar, 1, CPU_FP4_MICROSCALE_BLOCK),
            other => return Err(kernel_err(format!("unknown planar format id {other}"))),
        };
        PlanarLayout::new(
            format,
            self.out_features,
            self.in_features,
            block_out,
            block_in,
        )
        .map_err(|err| kernel_err(err.to_string()))
    }

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
/// plus the one-time CPU-authoritative value admission (reserved encodings and
/// decoded-product overflow) that must pass before upload or launch.
fn validate_planar_linear_host(
    dims: &PlanarLinearDims,
    activation_elems: usize,
    packed: &[u8],
    scale: &[u8],
    output_elems: usize,
) -> Result<ValidatedPlanarLinear> {
    let expected = dims.expected_lengths()?;
    let expected_activation = dims.m_rows * dims.in_features;
    if activation_elems != expected_activation {
        return Err(kernel_err(format!(
            "activation has {activation_elems} elements, expected M*K = {expected_activation}"
        )));
    }
    if packed.len() != expected.packed_bytes {
        return Err(kernel_err(format!(
            "packed weight has {} bytes, expected {}",
            packed.len(),
            expected.packed_bytes
        )));
    }
    if scale.len() != expected.scale_bytes {
        return Err(kernel_err(format!(
            "aux scale has {} bytes, expected {}",
            scale.len(),
            expected.scale_bytes
        )));
    }
    if output_elems != expected.output_elems {
        return Err(kernel_err(format!(
            "output has {output_elems} elements, expected M*N = {}",
            expected.output_elems
        )));
    }
    let bank_identity = validate_planar_values(&dims.cpu_layout()?, packed, scale)
        .map_err(|err| kernel_err(err.to_string()))?;
    Ok(ValidatedPlanarLinear {
        dims: *dims,
        bank_identity,
    })
}

/// Cached proof that one immutable planar linear bank passed the CPU oracle's
/// exact geometry and value contract.
#[derive(Clone, Copy, Debug)]
struct ValidatedPlanarLinear {
    dims: PlanarLinearDims,
    bank_identity: PlanarBankIdentity,
}

impl ValidatedPlanarLinear {
    fn dims(&self) -> &PlanarLinearDims {
        &self.dims
    }
}

pub(crate) struct ImmutablePlanarDeviceBuffer {
    allocation: CudaSealedAllocation,
}

impl ImmutablePlanarDeviceBuffer {
    pub(crate) fn upload(
        provider: &Arc<CudaExecutionProvider>,
        bytes: &[u8],
        label: &str,
    ) -> Result<Self> {
        let allocation = provider
            .upload_sealed(bytes, 256)
            .map_err(|err| kernel_err(format!("allocate/upload immutable {label}: {err}")))?;
        Ok(Self { allocation })
    }

    pub(crate) fn ptr(&self, access: &super::SealedLaunchAccess) -> CUdeviceptr {
        self.allocation.launch_ptr(access)
    }
}

struct PlanarLinearBanks {
    packed: ImmutablePlanarDeviceBuffer,
    scale: ImmutablePlanarDeviceBuffer,
}

/// Sealed admission for one planar linear bank.
///
/// The handle owns the exact packed-weight and aux-scale device allocations
/// populated from the bytes that passed the CPU oracle. Its fields and device
/// buffers are private, it is neither `Clone` nor `Copy`, and the safe launch
/// surface has no independent weight pointers to substitute or overwrite.
///
/// The 64-bit [`PlanarBankIdentity`] is exposed only for diagnostics; launch
/// trust comes from allocation ownership and immutability, never hash equality.
/// There is deliberately no content-addressed admission cache. Provider-internal
/// stable-VA remapping may retain an admission only when allocation ownership
/// and immutable content identity are preserved; replacing or mutating backing
/// storage invalidates the handle and requires a fresh atomic admission. During
/// CUDA graph capture the graph registry strongly owns the sealed banks, so no
/// release, remap, or content replacement can become reachable until every
/// graph pin is reset or destroyed.
///
/// ```
/// fn accepts(_: &onnx_runtime_ep_cuda::AdmittedPlanarLinear) {}
/// ```
///
/// ```compile_fail,E0451
/// # use onnx_runtime_ep_cuda::AdmittedPlanarLinear;
/// let forged = AdmittedPlanarLinear {
///     runtime: todo!(),
/// };
/// ```
///
/// ```compile_fail
/// # use onnx_runtime_ep_cuda::AdmittedPlanarLinear;
/// fn requires_clone<T: Clone>() {}
/// requires_clone::<AdmittedPlanarLinear>();
/// ```
///
/// Safe upload and VMM APIs require a `DeviceBuffer`; an admission cannot be
/// converted to one, so admitted bytes cannot be overwritten or remapped.
///
/// ```compile_fail
/// # use onnx_runtime_ep_api::ExecutionProvider;
/// # use onnx_runtime_ep_cuda::{AdmittedPlanarLinear, CudaExecutionProvider};
/// # fn overwrite(ep: &CudaExecutionProvider, bank: &mut AdmittedPlanarLinear) {
/// ep.copy_from_host(&[0xff], bank);
/// # }
/// ```
///
/// ```compile_fail
/// # use onnx_runtime_ep_api::ExecutionProvider;
/// # use onnx_runtime_ep_cuda::{AdmittedPlanarLinear, CudaExecutionProvider};
/// # fn remap(ep: &CudaExecutionProvider, bank: &AdmittedPlanarLinear) {
/// ep.decommit_allocation_range(bank, 0, 1);
/// # }
/// ```
///
/// ```compile_fail
/// # use onnx_runtime_ep_cuda::AdmittedPlanarLinear;
/// # fn escape(bank: &AdmittedPlanarLinear) {
/// let _ = bank.as_ptr();
/// # }
/// ```
pub struct AdmittedPlanarLinear {
    // Declared before `provider`: an eager-only admission releases its banks
    // while the provider queue is still open. Captured graphs retain `banks`
    // directly without retaining the provider/runtime cycle.
    banks: Arc<PlanarLinearBanks>,
    provider: Arc<CudaExecutionProvider>,
    device: onnx_runtime_ir::DeviceId,
    provider_context: ProviderContextIdentity,
    validation: ValidatedPlanarLinear,
}

impl AdmittedPlanarLinear {
    pub fn dims(&self) -> &PlanarLinearDims {
        self.validation.dims()
    }

    pub fn diagnostic_bank_identity(&self) -> PlanarBankIdentity {
        self.validation.bank_identity
    }
}

/// Validate exact host bytes, then atomically upload them into sealed,
/// provider-owned device allocations. Admission is illegal during capture
/// because it allocates and performs synchronous H2D exactly once.
pub fn admit_planar_linear(
    provider: &Arc<CudaExecutionProvider>,
    dims: &PlanarLinearDims,
    activation_elems: usize,
    packed: &[u8],
    scale: &[u8],
    output_elems: usize,
) -> Result<AdmittedPlanarLinear> {
    if provider.runtime().is_capturing()? {
        return Err(kernel_err(
            "cannot admit a planar bank during CUDA graph capture",
        ));
    }
    let validation =
        validate_planar_linear_host(dims, activation_elems, packed, scale, output_elems)?;
    let packed = ImmutablePlanarDeviceBuffer::upload(provider, packed, "packed weights")?;
    let scale = ImmutablePlanarDeviceBuffer::upload(provider, scale, "aux scales")?;
    Ok(AdmittedPlanarLinear {
        banks: Arc::new(PlanarLinearBanks { packed, scale }),
        provider: Arc::clone(provider),
        device: provider.device_id(),
        provider_context: provider.provider_context_identity(),
        validation,
    })
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

#[derive(Clone, Copy)]
struct PlanarLinearRawPtrs {
    activation: CUdeviceptr,
    packed: CUdeviceptr,
    scale: CUdeviceptr,
    output: CUdeviceptr,
}

/// Threads per block for the one-thread-per-output planar linear kernel.
const PLANAR_LINEAR_BLOCK: u32 = 256;

/// Launch the planar linear kernel on `runtime`'s EP stream for `dtype`.
///
/// The admission owns the exact bank allocations populated by
/// [`admit_planar_linear`]. The launch issues **no** host synchronization and
/// allocates nothing, so a warmed signature records cleanly into a CUDA-graph
/// capture. Ordering with a later device→host read is guaranteed by the single
/// in-order EP stream.
pub fn launch_planar_linear(
    admission: &AdmittedPlanarLinear,
    dtype: PlanarActivationDtype,
    activation: &DeviceBuffer,
    output: &mut DeviceBuffer,
) -> Result<()> {
    let dims = admission.dims();
    let activation_bytes = dims
        .m_rows
        .checked_mul(dims.in_features)
        .and_then(|elems| elems.checked_mul(dtype.byte_size()))
        .ok_or_else(|| kernel_err("activation byte count overflow"))?;
    let output_bytes = dims
        .m_rows
        .checked_mul(dims.out_features)
        .and_then(|elems| elems.checked_mul(dtype.byte_size()))
        .ok_or_else(|| kernel_err("output byte count overflow"))?;
    for (label, buffer, expected) in [
        ("activation", activation, activation_bytes),
        ("output", &*output, output_bytes),
    ] {
        if buffer.device() != admission.device {
            return Err(kernel_err(format!(
                "{label} device {:?} does not match admitted bank device {:?}",
                buffer.device(),
                admission.device
            )));
        }
        if buffer.len() != expected {
            return Err(kernel_err(format!(
                "{label} has {} bytes, expected {expected}",
                buffer.len()
            )));
        }
        let context = buffer
            .bound_owner()
            .ok_or_else(|| {
                kernel_err(format!(
                    "{label} has no binding-issued provider-context identity"
                ))
            })?
            .identity()
            .binding()
            .provider_context();
        if context != admission.provider_context {
            return Err(kernel_err(format!(
                "{label} provider context {context:?} does not match admitted bank context {:?}",
                admission.provider_context
            )));
        }
    }
    let runtime = admission.provider.runtime();
    runtime.retain_active_graph_resource(
        Arc::as_ptr(&admission.banks) as usize,
        &admission.banks,
        "planar linear bank",
    )?;
    let access = super::SealedLaunchAccess::new();
    let ptrs = PlanarLinearRawPtrs {
        activation: cuptr(activation.as_ptr()),
        packed: admission.banks.packed.ptr(&access),
        scale: admission.banks.scale.ptr(&access),
        output: cuptr(output.as_mut_ptr()),
    };
    // SAFETY: the sealed admission owns the exact immutable bank allocations;
    // activation/output device and extents were checked above.
    unsafe { launch_planar_linear_raw(runtime, dtype, dims, &ptrs) }
}

unsafe fn launch_planar_linear_raw(
    runtime: &CudaRuntime,
    dtype: PlanarActivationDtype,
    dims: &PlanarLinearDims,
    ptrs: &PlanarLinearRawPtrs,
) -> Result<()> {
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

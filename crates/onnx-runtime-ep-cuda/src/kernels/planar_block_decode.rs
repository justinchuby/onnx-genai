//! Exact-format CUDA decode for the DeepSeek-V4 planar B2 weight formats
//! (`block_fp8`, `fp4_planar`) — the follow-up stacked on the CPU planar ABI
//! (`onnx_runtime_ep_cpu::kernels::planar_block_quant`).
//!
//! ## What this slice delivers (and what it deliberately does not)
//!
//! This is the **exact device decode arithmetic** for the two planar formats,
//! staged as an NVRTC source string plus a **bit-identical host Rust mirror**.
//! The mirror lets us verify the device algorithm against the already-vetted CPU
//! oracle *without a GPU*, exactly as [`super::block_quant`] verifies its device
//! quantizers against the CPU `block_dequant` reference. The device source and
//! the mirror are hand-written to the same formulae and kept adjacent so their
//! correspondence is checkable by inspection; the mirror is then proven equal to
//! the CPU oracle by the tests below.
//!
//! **No success claim.** The `BlockQuantizedMoE` CUDA claim gate still
//! *typed-rejects* `block_fp8` / `fp4_planar` (see
//! [`super::block_quantized_moe`]): a real GPU has neither compiled nor run this
//! NVRTC module here (no nvcc / no device in this environment), so the exact
//! numeric parity on-device is unproven. The claim gate may only flip to *claim*
//! once [`planar_decode_gpu_parity`](self)-style on-device parity (a real launch
//! compared against the CPU oracle) passes on hardware. Until then this module
//! is the verified-by-mirror foundation, not an executable GPU path.
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

/// Logical input elements per UE8M0 micro-scale in `fp4_planar` (MXFP4 block-32).
pub(crate) const FP4_MICROSCALE_BLOCK: usize = 32;

/// `format` selector shared by the device kernel and the host mirror.
pub(crate) const PLANAR_FORMAT_BLOCK_FP8: i32 = 0;
#[allow(dead_code)]
pub(crate) const PLANAR_FORMAT_FP4_PLANAR: i32 = 1;

/// NVRTC device source for the exact planar decode + a straight `[M,N]` linear
/// kernel. Staged for the GPU claim; not yet compiled/launched here (no device).
#[allow(dead_code)]
pub(crate) const PLANAR_BLOCK_DECODE_CUH: &str = r#"
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
    const int scale_cols = (in_features + bs1 - 1) / bs1;
    const unsigned char se = scale[(out_row / bs0) * scale_cols + (in_col / bs1)];
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
// C[M,N] = A[M,K] * W[K,N]; W decoded per (out_row = n, in_col = k). One thread
// per (row, col). format: 0 = block_fp8, 1 = fp4_planar.
extern "C" __global__ void planar_linear_f32(
    const float* a, const unsigned char* packed, const unsigned char* scale,
    float* c, int m_rows, int in_features, int out_features,
    int format, int bs0, int bs1) {
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)m_rows * out_features) return;
    const int row = (int)(idx / out_features);
    const int col = (int)(idx % out_features);
    const float* a_row = a + (long long)row * in_features;
    float acc = 0.0f;
    for (int k = 0; k < in_features; ++k) {
        const float w = (format == 0)
            ? planar_bf8_element(packed, scale, out_features, in_features, bs0, bs1, col, k)
            : planar_fp4_element(packed, scale, out_features, in_features, col, k);
        acc += a_row[k] * w;
    }
    c[(long long)row * out_features + col] = acc;
}
"#;

/// Entry-point name of the planar linear kernel in [`PLANAR_BLOCK_DECODE_CUH`].
#[allow(dead_code)]
pub(crate) const PLANAR_LINEAR_ENTRY: &str = "planar_linear_f32";

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

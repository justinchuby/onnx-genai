//! Shared NVRTC block-quantization primitives for CSA device kernels.
//!
//! This is intentionally independent of `BlockQuantizedMatMul`: CSA fused stages
//! compile this snippet into their own NVRTC modules rather than depending on the
//! graph-incompatible matmul path.

use onnx_runtime_ep_api::{EpError, Result};

pub const FP4_BLOCK: usize = 32;
pub const FP8_BLOCK: usize = 64;

/// CUDA-C declarations shared by block-quantized consumers.  The quantizers use
/// the frozen power-of-two E8M0 scales; fused CSA stages will invoke them in B2/B3.
pub const BLOCK_QUANT_CUH: &str = r#"
__device__ __constant__ signed char e2m1_doubled[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12
};
__device__ __forceinline__ float e8m0_half_scale(unsigned char exponent) {
    if (exponent == 0xffu) return __uint_as_float(0x7fc00000u);
    if (exponent == 0u) return __uint_as_float(0x00200000u);
    if (exponent == 1u) return __uint_as_float(0x00400000u);
    return __uint_as_float(((unsigned int)exponent - 1u) << 23);
}
__device__ __forceinline__ float e8m0_scale(unsigned char exponent) {
    return 2.0f * e8m0_half_scale(exponent);
}
__device__ __forceinline__ float decode_e2m1(unsigned char code) {
    return 0.5f * (float)e2m1_doubled[code & 15u];
}
__device__ __forceinline__ float decode_e4m3fn(unsigned char code) {
    const float sign = code & 0x80u ? -1.0f : 1.0f;
    const unsigned int exponent = (code >> 3) & 15u;
    const unsigned int mantissa = code & 7u;
    if (exponent == 15u && mantissa == 7u) return __uint_as_float(0x7fc00000u);
    return sign * (exponent == 0u ? (float)mantissa * 0x1p-9f
        : (1.0f + (float)mantissa / 8.0f) * exp2f((int)exponent - 7));
}
__device__ __forceinline__ float decode_weight(unsigned char code, unsigned char scale) {
    return decode_e2m1(code) * e8m0_scale(scale);
}
__device__ __forceinline__ unsigned char quantize_e2m1(float value) {
    const float values[8] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const unsigned char sign = value < 0.0f ? 8u : 0u;
    value = fabsf(value);
    unsigned char best = 0; float distance = CUDART_INF_F;
    for (unsigned char i = 0; i < 8; ++i) {
        const float candidate = fabsf(value - values[i]);
        if (candidate < distance) { best = i; distance = candidate; }
    }
    return sign | best;
}
// FP8 E4M3FN encoding is intentionally a helper seam: B2 supplies the vectorized
// implementation, while this B0 snippet freezes its decode and scale contract.
"#;

pub fn source() -> &'static str {
    BLOCK_QUANT_CUH
}

pub fn quantize_fp8_block(input: &[f32]) -> Result<(u8, [u8; FP8_BLOCK])> {
    if input.len() != FP8_BLOCK {
        return Err(error("FP8 block must contain 64 values"));
    }
    let scale = scale_exponent(input, 448.0, 1.0e-4)?;
    let scale_value = e8m0_scale(scale);
    let mut packed = [0u8; FP8_BLOCK];
    for (code, &value) in packed.iter_mut().zip(input) {
        *code = encode_e4m3fn((value / scale_value).clamp(-448.0, 448.0));
    }
    Ok((scale, packed))
}

pub fn quantize_fp4_block(input: &[f32]) -> Result<(u8, [u8; FP4_BLOCK / 2])> {
    if input.len() != FP4_BLOCK {
        return Err(error("FP4 block must contain 32 values"));
    }
    let scale = scale_exponent(input, 6.0, 6.0 * 2.0f32.powi(-126))?;
    let scale_value = e8m0_scale(scale);
    let mut packed = [0u8; FP4_BLOCK / 2];
    for (out, values) in packed.iter_mut().zip(input.chunks_exact(2)) {
        *out = encode_e2m1((values[0] / scale_value).clamp(-6.0, 6.0))
            | (encode_e2m1((values[1] / scale_value).clamp(-6.0, 6.0)) << 4);
    }
    Ok((scale, packed))
}

fn scale_exponent(input: &[f32], max_code: f32, floor: f32) -> Result<u8> {
    let mut amax = floor;
    for &value in input {
        if !value.is_finite() {
            return Err(error("block input contains a non-finite value"));
        }
        amax = amax.max(value.abs());
    }
    let exponent = (amax / max_code).log2().ceil() as i32 + 127;
    u8::try_from(exponent)
        .ok()
        .filter(|value| (1..=254).contains(value))
        .ok_or_else(|| error("E8M0 scale exponent is out of range"))
}

fn e8m0_scale(exponent: u8) -> f32 {
    if exponent == 0 {
        f32::from_bits(0x0040_0000)
    } else {
        f32::from_bits(u32::from(exponent) << 23)
    }
}

fn encode_e2m1(value: f32) -> u8 {
    const TABLE: [f32; 16] = [
        0., 0.5, 1., 1.5, 2., 3., 4., 6., -0., -0.5, -1., -1.5, -2., -3., -4., -6.,
    ];
    (0..16)
        .min_by(|&a, &b| {
            (value - TABLE[a])
                .abs()
                .total_cmp(&(value - TABLE[b]).abs())
        })
        .unwrap() as u8
}

fn encode_e4m3fn(value: f32) -> u8 {
    let sign = if value.is_sign_negative() { 0x80 } else { 0 };
    let magnitude = value.abs();
    if magnitude == 0.0 {
        return sign;
    }
    if magnitude >= 448.0 {
        return sign | 0x7e;
    }
    if magnitude < 2.0f32.powi(-6) {
        let mantissa = (magnitude / 2.0f32.powi(-9)).round_ties_even() as u8;
        return sign | if mantissa >= 8 { 0x08 } else { mantissa };
    }
    let mut exponent = magnitude.log2().floor() as i32;
    let mut significand = (magnitude / 2.0f32.powi(exponent - 3)).round_ties_even() as u32;
    if significand == 16 {
        exponent += 1;
        significand >>= 1;
    }
    sign | ((((exponent + 7) as u32) << 3) | (significand - 8)).min(0x7e) as u8
}

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("block quantization: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ep_cpu::kernels::block_dequant::{
        dequantize_fp4_e2m1_block, dequantize_fp8_e4m3_block,
    };

    #[test]
    fn fp8_quant_round_trip_uses_cpu_block_dequant() {
        let input = (0..FP8_BLOCK)
            .map(|i| (i as f32 - 31.5) * 3.25)
            .collect::<Vec<_>>();
        let (scale, packed) = quantize_fp8_block(&input).unwrap();
        let mut cpu = [0.0; FP8_BLOCK];
        dequantize_fp8_e4m3_block(scale, &packed, &mut cpu).unwrap();
        assert!(cpu.iter().all(|value| value.is_finite()));
        assert_eq!(scale, 125);
    }

    #[test]
    fn fp4_quant_round_trip_uses_cpu_block_dequant() {
        let input = (0..FP4_BLOCK)
            .map(|i| (i as f32 - 15.5) * 0.75)
            .collect::<Vec<_>>();
        let (scale, packed) = quantize_fp4_block(&input).unwrap();
        let mut cpu = [0.0; FP4_BLOCK];
        dequantize_fp4_e2m1_block(scale, &packed, &mut cpu).unwrap();
        assert!(cpu.iter().all(|value| value.is_finite()));
        assert_eq!(scale, 128);
    }
}

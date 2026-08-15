//! Shared OCP block-float decoding used by quantized weights and CSA caches.

use onnx_runtime_ep_api::{EpError, Result};

pub(crate) const FP4_E2M1_BLOCK_SIZE: usize = 32;
pub(crate) const FP8_E4M3_BLOCK_SIZE: usize = 64;
pub(crate) const FP4_E2M1_PACKED_BYTES: usize = FP4_E2M1_BLOCK_SIZE / 2;
pub(crate) const FP8_E4M3_PACKED_BYTES: usize = FP8_E4M3_BLOCK_SIZE;

const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

pub fn decode_e8m0_scale(exponent: u8) -> f32 {
    match exponent {
        0xff => f32::NAN,
        0 => f32::from_bits(0x0040_0000),
        _ => f32::from_bits(u32::from(exponent) << 23),
    }
}

pub fn decode_e2m1(code: u8) -> f32 {
    E2M1[usize::from(code & 0x0f)]
}

pub fn decode_e4m3fn(code: u8) -> f32 {
    let sign = if code & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (code >> 3) & 0x0f;
    let mantissa = code & 0x07;
    if exponent == 0x0f && mantissa == 0x07 {
        return f32::NAN;
    }
    let magnitude = if exponent == 0 {
        f32::from(mantissa) * 2.0f32.powi(-9)
    } else {
        (1.0 + f32::from(mantissa) / 8.0) * 2.0f32.powi(i32::from(exponent) - 7)
    };
    sign * magnitude
}

pub(crate) fn quantize_fp8_e4m3_block(
    input: &[f32],
    scale_exponent: &mut u8,
    packed: &mut [u8],
    output: &mut [f32],
) -> Result<()> {
    require_block_lengths(
        "FP8 E4M3FN",
        packed,
        FP8_E4M3_PACKED_BYTES,
        output,
        FP8_E4M3_BLOCK_SIZE,
    )?;
    if input.len() != FP8_E4M3_BLOCK_SIZE {
        return Err(error(format!(
            "FP8 E4M3FN block must consume {FP8_E4M3_BLOCK_SIZE} values, got {}",
            input.len()
        )));
    }
    let mut amax = 1.0e-4f32;
    for &value in input {
        if !value.is_finite() {
            return Err(error("FP8 E4M3FN input contains a non-finite value"));
        }
        amax = amax.max(value.abs());
    }
    let scale_power = (amax / 448.0).log2().ceil() as i32;
    let exponent = scale_power
        .checked_add(127)
        .filter(|&value| (1..=254).contains(&value))
        .ok_or_else(|| error("FP8 E4M3FN E8M0 scale exponent is out of range"))?;
    *scale_exponent = exponent as u8;
    let scale = 2.0f32.powi(scale_power);
    for ((code, &value), destination) in packed.iter_mut().zip(input).zip(output.iter_mut()) {
        *code = encode_e4m3fn((value / scale).clamp(-448.0, 448.0));
        *destination = 0.0;
    }
    dequantize_fp8_e4m3_block(*scale_exponent, packed, output)
}

pub fn dequantize_fp4_e2m1_block(
    scale_exponent: u8,
    packed: &[u8],
    output: &mut [f32],
) -> Result<()> {
    require_block_lengths(
        "FP4 E2M1",
        packed,
        FP4_E2M1_PACKED_BYTES,
        output,
        FP4_E2M1_BLOCK_SIZE,
    )?;
    let scale = require_finite_scale("FP4 E2M1", scale_exponent)?;
    for (pair, &byte) in packed.iter().enumerate() {
        let output_offset = pair
            .checked_mul(2)
            .ok_or_else(|| error("FP4 output offset overflow"))?;
        let second = output_offset
            .checked_add(1)
            .ok_or_else(|| error("FP4 second output offset overflow"))?;
        output[output_offset] = checked_scaled_value("FP4 E2M1", decode_e2m1(byte), scale)?;
        output[second] = checked_scaled_value("FP4 E2M1", decode_e2m1(byte >> 4), scale)?;
    }
    Ok(())
}

pub(crate) fn quantize_fp4_e2m1_block(
    input: &[f32],
    scale_exponent: &mut u8,
    packed: &mut [u8],
    output: &mut [f32],
) -> Result<()> {
    require_block_lengths(
        "FP4 E2M1",
        packed,
        FP4_E2M1_PACKED_BYTES,
        output,
        FP4_E2M1_BLOCK_SIZE,
    )?;
    if input.len() != FP4_E2M1_BLOCK_SIZE {
        return Err(error(format!(
            "FP4 E2M1 block must consume {FP4_E2M1_BLOCK_SIZE} values, got {}",
            input.len()
        )));
    }
    let mut amax = 6.0 * 2.0f32.powi(-126);
    for &value in input {
        if !value.is_finite() {
            return Err(error("FP4 E2M1 input contains a non-finite value"));
        }
        amax = amax.max(value.abs());
    }
    let scale_power = (amax / 6.0).log2().ceil() as i32;
    let exponent = scale_power
        .checked_add(127)
        .filter(|&value| (1..=254).contains(&value))
        .ok_or_else(|| error("FP4 E2M1 E8M0 scale exponent is out of range"))?;
    *scale_exponent = exponent as u8;
    let scale = 2.0f32.powi(scale_power);
    for (pair, destination) in input.chunks_exact(2).zip(packed.iter_mut()) {
        let low = encode_e2m1((pair[0] / scale).clamp(-6.0, 6.0));
        let high = encode_e2m1((pair[1] / scale).clamp(-6.0, 6.0));
        *destination = low | (high << 4);
    }
    dequantize_fp4_e2m1_block(*scale_exponent, packed, output)
}

fn encode_e2m1(value: f32) -> u8 {
    let mut best_code = 0u8;
    let mut best_distance = f32::INFINITY;
    for code in 0u8..16 {
        let distance = (value - decode_e2m1(code)).abs();
        if distance < best_distance
            || (distance == best_distance && code & 1 == 0 && best_code & 1 != 0)
        {
            best_code = code;
            best_distance = distance;
        }
    }
    best_code
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
    let min_normal = 2.0f32.powi(-6);
    if magnitude < min_normal {
        let mantissa = (magnitude / 2.0f32.powi(-9)).round_ties_even() as u8;
        if mantissa == 0 {
            return sign;
        }
        if mantissa >= 8 {
            return sign | 0x08;
        }
        return sign | mantissa;
    }
    let mut exponent = magnitude.log2().floor() as i32;
    let step = 2.0f32.powi(exponent - 3);
    let mut significand = (magnitude / step).round_ties_even() as u32;
    if significand == 16 {
        exponent += 1;
        significand >>= 1;
    }
    let encoded = (((exponent + 7) as u32) << 3) | (significand - 8);
    sign | encoded.min(0x7e) as u8
}

pub fn dequantize_fp8_e4m3_block(
    scale_exponent: u8,
    packed: &[u8],
    output: &mut [f32],
) -> Result<()> {
    require_block_lengths(
        "FP8 E4M3FN",
        packed,
        FP8_E4M3_PACKED_BYTES,
        output,
        FP8_E4M3_BLOCK_SIZE,
    )?;
    let scale = require_finite_scale("FP8 E4M3FN", scale_exponent)?;
    for (destination, &code) in output.iter_mut().zip(packed) {
        let value = decode_e4m3fn(code);
        if !value.is_finite() {
            return Err(error(format!(
                "FP8 E4M3FN block contains reserved NaN code 0x{code:02x}"
            )));
        }
        *destination = checked_scaled_value("FP8 E4M3FN", value, scale)?;
    }
    Ok(())
}

fn require_finite_scale(format: &str, exponent: u8) -> Result<f32> {
    let scale = decode_e8m0_scale(exponent);
    if !scale.is_finite() {
        return Err(error(format!(
            "{format} block uses reserved E8M0 scale exponent 0xff"
        )));
    }
    Ok(scale)
}

fn checked_scaled_value(format: &str, value: f32, scale: f32) -> Result<f32> {
    let result = value * scale;
    if !result.is_finite() {
        return Err(error(format!(
            "{format} value {value} overflows with block scale {scale}"
        )));
    }
    Ok(result)
}

fn require_block_lengths(
    format: &str,
    packed: &[u8],
    expected_packed: usize,
    output: &[f32],
    expected_output: usize,
) -> Result<()> {
    if packed.len() != expected_packed {
        return Err(error(format!(
            "{format} block must contain {expected_packed} packed bytes, got {}",
            packed.len()
        )));
    }
    if output.len() != expected_output {
        return Err(error(format!(
            "{format} block must produce {expected_output} values, got {}",
            output.len()
        )));
    }
    Ok(())
}

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("block dequantization: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fp4_known_block_applies_e8m0_scale_and_adjacent_nibbles() {
        let codes = [0x21, 0x43, 0x65, 0x87, 0xa9, 0xcb, 0xed, 0x0f];
        let mut packed = [0u8; FP4_E2M1_PACKED_BYTES];
        for (destination, source) in packed.iter_mut().zip(codes.into_iter().cycle()) {
            *destination = source;
        }
        let mut actual = [0.0f32; FP4_E2M1_BLOCK_SIZE];
        dequantize_fp4_e2m1_block(128, &packed, &mut actual).unwrap();

        let expected_pair = [
            1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, -0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0,
            0.0,
        ];
        let expected = expected_pair
            .into_iter()
            .cycle()
            .take(FP4_E2M1_BLOCK_SIZE)
            .collect::<Vec<_>>();
        assert_eq!(actual.as_slice(), expected);
    }

    #[test]
    fn fp8_known_block_applies_e8m0_scale_exactly() {
        let codes = [0x00, 0x01, 0x08, 0x38, 0x3c, 0x7e, 0x81, 0xb8];
        let mut packed = [0u8; FP8_E4M3_PACKED_BYTES];
        for (destination, source) in packed.iter_mut().zip(codes.into_iter().cycle()) {
            *destination = source;
        }
        let mut actual = [0.0f32; FP8_E4M3_BLOCK_SIZE];
        dequantize_fp8_e4m3_block(128, &packed, &mut actual).unwrap();

        let expected_group = [
            0.0,
            1.0 / 256.0,
            1.0 / 32.0,
            2.0,
            3.0,
            896.0,
            -1.0 / 256.0,
            -2.0,
        ];
        let expected = expected_group
            .into_iter()
            .cycle()
            .take(FP8_E4M3_BLOCK_SIZE)
            .collect::<Vec<_>>();
        assert_eq!(actual.as_slice(), expected);
    }
}

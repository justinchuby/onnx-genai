//! Portable software conversion for the FP8 formats used by KV caches.

/// Supported 8-bit floating-point formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8Format {
    /// OCP E4M3FN: four exponent bits, three mantissa bits, finite-only.
    E4M3Fn,
    /// OCP E5M2: five exponent bits, two mantissa bits, IEEE-style infinities.
    E5M2,
}

impl Fp8Format {
    const fn mantissa_bits(self) -> u32 {
        match self {
            Self::E4M3Fn => 3,
            Self::E5M2 => 2,
        }
    }

    const fn exponent_bias(self) -> i32 {
        match self {
            Self::E4M3Fn => 7,
            Self::E5M2 => 15,
        }
    }

    const fn max_finite_bits(self) -> u8 {
        match self {
            Self::E4M3Fn => 0x7e,
            Self::E5M2 => 0x7b,
        }
    }

    const fn nan_bits(self) -> u8 {
        match self {
            Self::E4M3Fn => 0x7f,
            Self::E5M2 => 0x7e,
        }
    }

    /// Largest finite positive value representable by the format.
    pub const fn max_finite(self) -> f32 {
        match self {
            Self::E4M3Fn => 448.0,
            Self::E5M2 => 57_344.0,
        }
    }
}

/// Convert an `f32` to FP8 with round-to-nearest, ties-to-even.
///
/// Finite overflow and infinities saturate to the largest finite value. NaNs
/// are converted to a canonical quiet NaN while preserving no payload.
pub fn encode_f32(value: f32, format: Fp8Format) -> u8 {
    let sign = if value.is_sign_negative() { 0x80 } else { 0 };
    if value.is_nan() {
        return sign | format.nan_bits();
    }

    let magnitude = value.abs();
    if magnitude == 0.0 {
        return sign;
    }
    if !magnitude.is_finite() || magnitude >= format.max_finite() {
        return sign | format.max_finite_bits();
    }

    let mantissa_bits = format.mantissa_bits();
    let bias = format.exponent_bias();
    let min_normal_exponent = 1 - bias;
    let min_normal = 2.0_f32.powi(min_normal_exponent);

    if magnitude < min_normal {
        let step = 2.0_f32.powi(min_normal_exponent - mantissa_bits as i32);
        let mantissa = (magnitude / step).round_ties_even() as u8;
        if mantissa == 0 {
            return sign;
        }
        if mantissa >= 1 << mantissa_bits {
            return sign | (1 << mantissa_bits);
        }
        return sign | mantissa;
    }

    let mut exponent = magnitude.log2().floor() as i32;
    let step = 2.0_f32.powi(exponent - mantissa_bits as i32);
    let mut significand = (magnitude / step).round_ties_even() as u32;
    if significand == 1 << (mantissa_bits + 1) {
        exponent += 1;
        significand >>= 1;
    }

    let exponent_field = exponent + bias;
    let mantissa = significand - (1 << mantissa_bits);
    let encoded = ((exponent_field as u32) << mantissa_bits) | mantissa;
    if encoded >= format.max_finite_bits() as u32 {
        sign | format.max_finite_bits()
    } else {
        sign | encoded as u8
    }
}

/// Convert an FP8 bit pattern to `f32`.
pub fn decode_f32(bits: u8, format: Fp8Format) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let magnitude_bits = bits & 0x7f;
    let mantissa_bits = format.mantissa_bits();
    let mantissa_mask = (1_u8 << mantissa_bits) - 1;
    let exponent = magnitude_bits >> mantissa_bits;
    let mantissa = magnitude_bits & mantissa_mask;

    match format {
        Fp8Format::E4M3Fn if exponent == 0x0f && mantissa == 0x07 => f32::NAN,
        Fp8Format::E5M2 if exponent == 0x1f && mantissa == 0 => sign * f32::INFINITY,
        Fp8Format::E5M2 if exponent == 0x1f => f32::NAN,
        _ if exponent == 0 => {
            let value = f32::from(mantissa)
                * 2.0_f32.powi(1 - format.exponent_bias() - mantissa_bits as i32);
            sign * value
        }
        _ => {
            let fraction = 1.0 + f32::from(mantissa) / (1_u32 << mantissa_bits) as f32;
            sign * fraction * 2.0_f32.powi(i32::from(exponent) - format.exponent_bias())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3fn_known_encodings() {
        let cases = [
            (0.0, 0x00),
            (-0.0, 0x80),
            (2.0_f32.powi(-9), 0x01),
            (2.0_f32.powi(-6), 0x08),
            (0.5, 0x30),
            (1.0, 0x38),
            (-1.0, 0xb8),
            (2.0, 0x40),
            (448.0, 0x7e),
        ];
        for (value, expected) in cases {
            assert_eq!(encode_f32(value, Fp8Format::E4M3Fn), expected);
            assert_eq!(decode_f32(expected, Fp8Format::E4M3Fn), value);
        }
        assert!(decode_f32(0x7f, Fp8Format::E4M3Fn).is_nan());
        assert_eq!(encode_f32(f32::INFINITY, Fp8Format::E4M3Fn), 0x7e);
    }

    #[test]
    fn e5m2_known_encodings() {
        let cases = [
            (0.0, 0x00),
            (-0.0, 0x80),
            (2.0_f32.powi(-16), 0x01),
            (2.0_f32.powi(-14), 0x04),
            (0.5, 0x38),
            (1.0, 0x3c),
            (-1.0, 0xbc),
            (2.0, 0x40),
            (57_344.0, 0x7b),
        ];
        for (value, expected) in cases {
            assert_eq!(encode_f32(value, Fp8Format::E5M2), expected);
            assert_eq!(decode_f32(expected, Fp8Format::E5M2), value);
        }
        assert_eq!(decode_f32(0x7c, Fp8Format::E5M2), f32::INFINITY);
        assert!(decode_f32(0x7e, Fp8Format::E5M2).is_nan());
        assert_eq!(encode_f32(f32::INFINITY, Fp8Format::E5M2), 0x7b);
    }

    #[test]
    fn conversion_rounds_ties_to_even() {
        assert_eq!(encode_f32(1.0625, Fp8Format::E4M3Fn), 0x38);
        assert_eq!(encode_f32(1.1875, Fp8Format::E4M3Fn), 0x3a);
        assert_eq!(encode_f32(1.125, Fp8Format::E5M2), 0x3c);
        assert_eq!(encode_f32(1.375, Fp8Format::E5M2), 0x3e);
    }

    #[test]
    fn every_finite_encoding_round_trips_canonically() {
        for format in [Fp8Format::E4M3Fn, Fp8Format::E5M2] {
            for bits in 0_u8..=u8::MAX {
                let decoded = decode_f32(bits, format);
                if decoded.is_finite() {
                    assert_eq!(
                        encode_f32(decoded, format),
                        bits,
                        "{format:?} failed for {bits:#04x}"
                    );
                }
            }
        }
    }
}

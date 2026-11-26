//! Shared FP8 Cast policy and exact binary64 rounding boundaries.

use std::fmt::Write as _;
use std::sync::LazyLock;

use onnx_runtime_ir::{DataType, Node};

use crate::{EpError, Result};

#[derive(Clone, Copy, Debug, Default)]
pub enum RoundMode {
    #[default]
    Up,
    Down,
    Nearest,
}

#[derive(Clone, Copy, Debug)]
pub struct CastOptions {
    pub saturate: bool,
    pub round: RoundMode,
    opset: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct CastAttributes {
    opset: u64,
    saturate: Option<std::result::Result<bool, ()>>,
    round: Option<std::result::Result<RoundMode, ()>>,
}

impl CastAttributes {
    pub const fn defaults(opset: u64) -> Self {
        Self {
            opset,
            saturate: None,
            round: None,
        }
    }

    pub fn from_node(node: &Node, opset: u64) -> Self {
        let saturate = node.attr("saturate").map(|attr| match attr.as_int() {
            Some(0) if opset >= 19 => Ok(false),
            Some(1) if opset >= 19 => Ok(true),
            _ => Err(()),
        });
        let round = node.attr("round_mode").map(|attr| match attr.as_str() {
            Some("up") if opset >= 24 => Ok(RoundMode::Up),
            Some("down") if opset >= 24 => Ok(RoundMode::Down),
            Some("nearest") if opset >= 24 => Ok(RoundMode::Nearest),
            _ => Err(()),
        });
        Self {
            opset,
            saturate,
            round,
        }
    }

    pub fn resolve(self, op: &str, src: DataType, dst: DataType) -> Result<CastOptions> {
        let mut options = CastOptions::defaults(self.opset);
        options.validate_pair(op, src, dst)?;
        let invalid =
            |help: &str| EpError::KernelFailed(format!("{op} at opset {}: {help}", self.opset));
        if Float8::of(dst).is_some()
            && let Some(saturate) = self.saturate
        {
            options.saturate = saturate
                .map_err(|()| invalid("`saturate` requires opset 19+ and must be 0 or 1"))?;
        }
        if Float8::of(dst) == Some(Float8::E8M0)
            && let Some(round) = self.round
        {
            options.round = round.map_err(|()| {
                invalid("`round_mode` requires opset 24+ and must be `up`, `down`, or `nearest`")
            })?;
        }
        Ok(options)
    }
}

impl CastOptions {
    pub const fn defaults(opset: u64) -> Self {
        Self {
            saturate: true,
            round: RoundMode::Up,
            opset,
        }
    }

    pub fn validate_pair(self, op: &str, src: DataType, dst: DataType) -> Result<()> {
        if let Some(reason) = unsupported_reason(op, self.opset, Some(src), Some(dst)) {
            return Err(EpError::KernelFailed(reason));
        }
        Ok(())
    }
}

pub fn unsupported_reason(
    op: &str,
    opset: u64,
    src: Option<DataType>,
    dst: Option<DataType>,
) -> Option<String> {
    for (role, dtype) in [("source", src), ("target", dst)] {
        if let Some(dtype) = dtype
            && let Some(format) = Float8::of(dtype)
        {
            let min_opset = if format == Float8::E8M0 { 24 } else { 19 };
            if opset < min_opset {
                return Some(format!(
                    "{op}: {role} dtype {dtype:?} requires opset {min_opset} or newer (got {opset}); export the node at a supported revision"
                ));
            }
        }
    }
    None
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Float8 {
    E4M3FN,
    E4M3FNUZ,
    E5M2,
    E5M2FNUZ,
    E8M0,
}

const FORMATS: [Float8; 5] = [
    Float8::E4M3FN,
    Float8::E4M3FNUZ,
    Float8::E5M2,
    Float8::E5M2FNUZ,
    Float8::E8M0,
];

impl Float8 {
    pub const fn of(dtype: DataType) -> Option<Self> {
        Some(match dtype {
            DataType::Float8E4M3FN => Self::E4M3FN,
            DataType::Float8E4M3FNUZ => Self::E4M3FNUZ,
            DataType::Float8E5M2 => Self::E5M2,
            DataType::Float8E5M2FNUZ => Self::E5M2FNUZ,
            DataType::Float8E8M0 => Self::E8M0,
            _ => return None,
        })
    }

    pub fn decode(self, code: u8) -> f64 {
        f64::from_bits(DECODE[self as usize][usize::from(code)])
    }

    pub fn encode(self, value: f64, options: CastOptions) -> u8 {
        let row = self.row(options.round) as usize;
        let spec = ROWS[row];
        let bits = value.to_bits();
        let magnitude = bits & (u64::MAX >> 1);
        let code = if value.is_nan()
            || (!options.saturate && outside_e8m0_nearest_range(row, magnitude))
        {
            spec.nan
        } else {
            let count = usize::from(spec.max) + 1;
            let rounded =
                THRESHOLDS[row][..count].partition_point(|&boundary| boundary <= magnitude);
            if rounded < count {
                rounded as u8
            } else if options.saturate {
                spec.max
            } else {
                spec.overflow
            }
        };
        spec.with_sign(code, (bits >> 56) as u8 & 0x80)
    }

    /// Index into the shared CUDA decode table.
    pub const fn decode_row(self) -> i32 {
        self as i32
    }

    /// Index into the shared CUDA encode table (E8M0 has one row per mode).
    pub const fn row(self, mode: RoundMode) -> i32 {
        if matches!(self, Self::E8M0) {
            self as i32 + mode as i32
        } else {
            self as i32
        }
    }

    fn signed_magnitude(self, code: u8) -> f64 {
        let (mantissa, bias) = match self {
            Self::E4M3FN => (3, 7),
            Self::E4M3FNUZ => (3, 8),
            Self::E5M2 => (2, 15),
            Self::E5M2FNUZ => (2, 16),
            Self::E8M0 => return pow2(i32::from(code) - 127),
        };
        let exponent = i32::from(code >> mantissa);
        let fraction = code & ((1 << mantissa) - 1);
        let significand = if exponent == 0 {
            fraction
        } else {
            (1 << mantissa) | fraction
        };
        f64::from(significand) * pow2(exponent.max(1) - bias - mantissa)
    }
}

fn pow2(exponent: i32) -> f64 {
    f64::from_bits(((exponent + 1023) as u64) << 52)
}

const E8_NEAREST: usize = Float8::E8M0.row(RoundMode::Nearest) as usize;
const E8_MIN_BITS: u64 = ((1023 - 127) as u64) << 52;
const E8_MAX_BITS: u64 = ((1023 + 127) as u64) << 52;

fn outside_e8m0_nearest_range(row: usize, magnitude: u64) -> bool {
    // The operator's non-saturating nearest table rejects values outside the
    // finite E8M0 range before rounding, unlike ONNX's Python reference.
    row == E8_NEAREST && !(E8_MIN_BITS..=E8_MAX_BITS).contains(&magnitude)
}

#[derive(Clone, Copy)]
struct Format {
    max: u8,
    nan: u8,
    overflow: u8,
    unsigned_zero: bool,
    signed: bool,
}

impl Format {
    const fn new(max: u8, nan: u8, overflow: u8, unsigned_zero: bool, signed: bool) -> Self {
        Self {
            max,
            nan,
            overflow,
            unsigned_zero,
            signed,
        }
    }

    fn with_sign(self, code: u8, sign: u8) -> u8 {
        if self.signed && !(self.unsigned_zero && code == 0) {
            code | sign
        } else {
            code
        }
    }
}

const E8: Format = Format::new(254, 255, 255, false, false);
const ROWS: [Format; 7] = [
    Format::new(126, 127, 127, false, true),
    Format::new(127, 128, 128, true, true),
    Format::new(123, 126, 124, false, true),
    Format::new(127, 128, 128, true, true),
    E8,
    E8,
    E8,
];

static DECODE: LazyLock<[[u64; 256]; 5]> = LazyLock::new(|| {
    FORMATS.map(|format| {
        let spec = ROWS[format as usize];
        std::array::from_fn(|bits| {
            let code = bits as u8;
            let magnitude = if spec.signed { code & 0x7f } else { code };
            let value = if spec.unsigned_zero && code == 128 {
                f64::NAN
            } else if magnitude <= spec.max {
                format.signed_magnitude(magnitude)
            } else if format == Float8::E5M2 && magnitude == 124 {
                f64::INFINITY
            } else {
                f64::NAN
            };
            (if spec.signed && code & 0x80 != 0 {
                -value
            } else {
                value
            })
            .to_bits()
        })
    })
});

static THRESHOLDS: LazyLock<[[u64; 255]; 7]> = LazyLock::new(|| {
    std::array::from_fn(|row| {
        let format = FORMATS[row.min(Float8::E8M0 as usize)];
        let spec = ROWS[row];
        std::array::from_fn(|code| {
            let code = code as u8;
            if code > spec.max {
                return u64::MAX;
            }
            let lo = format.signed_magnitude(code);
            let hi = if code == spec.max && spec.signed {
                lo + (lo - format.signed_magnitude(code - 1))
            } else {
                format.signed_magnitude(code + 1)
            };
            // The table stores the first binary64 magnitude rounding to the next
            // code, including at the overflow edge. Advance by one bit for an
            // exclusive boundary, so ties don't depend on provider float settings.
            if row == Float8::E8M0.row(RoundMode::Up) as usize {
                lo.to_bits() + 1
            } else if row == Float8::E8M0.row(RoundMode::Down) as usize {
                hi.to_bits()
            } else {
                let ties_to_lower = spec.signed && code.is_multiple_of(2);
                ((lo + hi) * 0.5).to_bits() + u64::from(ties_to_lower)
            }
        })
    })
});

/// NVRTC constants and lookup routines, generated from the same tables CPU Cast uses.
pub fn cuda_source() -> &'static str {
    static SOURCE: LazyLock<String> = LazyLock::new(|| {
        let mut source = String::new();
        fn write_rows<const N: usize>(source: &mut String, name: &str, rows: &[[u64; N]]) {
            writeln!(
                source,
                "__device__ __constant__ unsigned long long {name}[{}][{N}] = {{",
                rows.len()
            )
            .unwrap();
            for row in rows {
                source.push('{');
                for bits in row {
                    write!(source, "0x{bits:x}ULL,").unwrap();
                }
                source.push_str("},\n");
            }
            source.push_str("};\n");
        }
        write_rows(&mut source, "fp8_decode", DECODE.as_slice());
        write_rows(&mut source, "fp8_threshold", THRESHOLDS.as_slice());
        source.push_str("__device__ __constant__ int fp8_format[7][5] = {\n");
        for spec in ROWS {
            writeln!(
                source,
                "{{{},{},{},{},{}}},",
                spec.max,
                spec.nan,
                spec.overflow,
                i32::from(spec.unsigned_zero),
                i32::from(spec.signed)
            )
            .unwrap();
        }
        writeln!(
            source,
            "}};\n#define FP8_E8_NEAREST {E8_NEAREST}\n#define FP8_E8_MIN_BITS 0x{E8_MIN_BITS:x}ULL\n#define FP8_E8_MAX_BITS 0x{E8_MAX_BITS:x}ULL\n#define FP8_E8_FIRST_ROW {}",
            Float8::E8M0.row(RoundMode::Up)
        )
        .unwrap();
        source.push_str(
            r#"
__device__ double decode_fp8(int row, unsigned char code) {
    return __longlong_as_double((long long)fp8_decode[row][code]);
}
__device__ unsigned char encode_fp8(int row, double value, int saturate) {
    const int* format = fp8_format[row];
    const unsigned long long bits = (unsigned long long)__double_as_longlong(value);
    const unsigned long long magnitude = bits & 0x7fffffffffffffffULL;
    int code;
    if (magnitude > 0x7ff0000000000000ULL ||
        (!saturate && row == FP8_E8_NEAREST &&
         (magnitude < FP8_E8_MIN_BITS || magnitude > FP8_E8_MAX_BITS))) {
        code = format[1];
    } else {
        int lo = 0, hi = format[0] + 1;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (fp8_threshold[row][mid] <= magnitude) lo = mid + 1;
            else hi = mid;
        }
        code = lo <= format[0] ? lo : (saturate ? format[0] : format[2]);
    }
    if (format[4] && !(format[3] && code == 0)) code |= (bits >> 56) & 0x80;
    return (unsigned char)code;
}
"#,
        );
        source
    });
    SOURCE.as_str()
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{Attribute, NodeId};

    #[test]
    fn float8_cast_rejects_invalid_versions_and_attributes() {
        let legacy = CastOptions::defaults(18);
        assert!(
            legacy
                .validate_pair("Cast", DataType::Float32, DataType::Float8E4M3FN)
                .unwrap_err()
                .to_string()
                .contains("opset 19 or newer")
        );
        let signed_only = CastOptions::defaults(23);
        assert!(
            signed_only
                .validate_pair("CastLike", DataType::Float8E8M0, DataType::Float32)
                .unwrap_err()
                .to_string()
                .contains("opset 24 or newer")
        );
        let mut node = Node::new(NodeId(0), "CastLike", vec![], vec![]);
        node.attributes
            .insert("round_mode".into(), Attribute::String(b"closest".to_vec()));
        assert!(
            CastAttributes::from_node(&node, 24)
                .resolve("CastLike", DataType::Float32, DataType::Float8E8M0)
                .unwrap_err()
                .to_string()
                .contains("`up`, `down`, or `nearest`")
        );
        assert!(
            CastAttributes::from_node(&node, 24)
                .resolve("CastLike", DataType::Float32, DataType::Float8E4M3FN)
                .is_ok()
        );
        node.attributes.clear();
        node.attributes.insert("saturate".into(), Attribute::Int(2));
        assert!(
            CastAttributes::from_node(&node, 24)
                .resolve("CastLike", DataType::Float32, DataType::Float8E4M3FN)
                .unwrap_err()
                .to_string()
                .contains("0 or 1")
        );
        assert!(
            CastAttributes::from_node(&node, 24)
                .resolve("CastLike", DataType::Float8E4M3FN, DataType::Float32)
                .is_ok()
        );
    }

    #[test]
    fn float8_cast_round_trips_every_finite_value() {
        for format in FORMATS {
            for code in 0..=u8::MAX {
                let value = format.decode(code);
                if value.is_finite() {
                    assert_eq!(
                        format.encode(value, CastOptions::defaults(28)),
                        code,
                        "{format:?} [{code:#04x}]"
                    );
                }
            }
        }
    }
}

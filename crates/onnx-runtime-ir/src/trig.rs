//! ONNX trigonometric and hyperbolic unary schema revisions.

use crate::DataType;

// ONNX math/defs.cc adds BF16 via UnaryFloatMathOpGenerator's default types;
// math/old.cc passes the original f16/f32/f64 list explicitly.
pub const BF16_SINCE: u64 = 22;
pub const OPS: &[(&str, u64)] = &[
    ("Tan", 7),
    ("Asin", 7),
    ("Acos", 7),
    ("Atan", 7),
    ("Sinh", 9),
    ("Cosh", 9),
    ("Asinh", 9),
    ("Acosh", 9),
    ("Atanh", 9),
];

/// Check the type added by the v22 revision, independent of kernel capabilities.
pub fn bf16_unsupported_reason(
    op: &str,
    domain: &str,
    opset: u64,
    dtype: DataType,
) -> Option<String> {
    (dtype == DataType::BFloat16
        && opset < BF16_SINCE
        && crate::is_default_domain(domain)
        && OPS.iter().any(|&(name, _)| op == name))
    .then(|| {
        format!(
            "{op} does not accept BF16 (BFloat16) at opset {opset}; export at opset {BF16_SINCE} or later, or supply Float16, Float32, or Float64"
        )
    })
}

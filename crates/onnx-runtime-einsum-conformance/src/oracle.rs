use std::collections::BTreeMap;

use half::{bf16, f16};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    CaseRecord, CaseValidationError, ConformanceDType, EquationError, ValueProfile,
    analyze_equation,
};

/// Compact row-major tensor storing exact dtype bits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalTensor {
    dtype: ConformanceDType,
    shape: Vec<usize>,
    bits: Vec<u64>,
}

impl CanonicalTensor {
    /// Construct a checked tensor from exact element bits.
    pub fn new(
        dtype: ConformanceDType,
        shape: Vec<usize>,
        bits: Vec<u64>,
    ) -> Result<Self, OracleError> {
        let expected = checked_numel(&shape).ok_or(OracleError::SizeOverflow)?;
        if bits.len() != expected {
            return Err(OracleError::ElementCount {
                expected,
                found: bits.len(),
            });
        }
        let mask = integer_mask(dtype);
        if bits.iter().any(|&value| value & !mask != 0) {
            return Err(OracleError::OutOfRangeBits { dtype });
        }
        Ok(Self { dtype, shape, bits })
    }

    /// Tensor dtype.
    pub const fn dtype(&self) -> ConformanceDType {
        self.dtype
    }

    /// Tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Exact element bit patterns.
    pub fn raw_bits(&self) -> &[u64] {
        &self.bits
    }

    /// Decode numeric values for diagnostics and Python adapters.
    pub fn to_f64_values(&self) -> Vec<f64> {
        self.bits
            .iter()
            .copied()
            .map(|bits| bits_to_f64(self.dtype, bits))
            .collect()
    }
}

/// Direct evaluator result plus per-output cancellation scale.
#[derive(Clone, Debug)]
pub struct Evaluation {
    output: CanonicalTensor,
    condition_scale: Vec<f64>,
    reduction_elements: usize,
    factor_count: usize,
}

impl Evaluation {
    /// Canonical fixed-order output.
    pub const fn output(&self) -> &CanonicalTensor {
        &self.output
    }

    /// Sum of absolute fixed-order products for each output element.
    pub fn condition_scale(&self) -> &[f64] {
        &self.condition_scale
    }

    /// Number of reduction tuples per output element.
    pub const fn reduction_elements(&self) -> usize {
        self.reduction_elements
    }

    /// Number of input factors in every product.
    pub const fn factor_count(&self) -> usize {
        self.factor_count
    }
}

/// Numeric comparison policy selected by a forced route.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparisonMode {
    /// Exact output bits, including NaN payload and signed zero.
    CanonicalBits,
    /// Condition-aware floating tolerance; integers remain exact.
    ConditionAware,
}

/// Successful comparison summary.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ComparisonReport {
    /// Compared elements.
    pub elements: usize,
    /// Largest finite absolute error.
    pub max_abs_error: f64,
    /// Largest allowed absolute error at the corresponding condition scale.
    pub max_allowed_error: f64,
}

/// Materialize deterministic row-major inputs from a compact case record.
pub fn materialize_inputs(case: &CaseRecord) -> Result<Vec<CanonicalTensor>, OracleError> {
    case.validate()?;
    case.input_shapes
        .iter()
        .enumerate()
        .map(|(input, shape)| {
            let count = checked_numel(shape).ok_or(OracleError::SizeOverflow)?;
            let mut rng = SplitMix64::new(
                case.values.seed ^ (input as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93),
            );
            let bits = (0..count)
                .map(|index| {
                    generated_bits(case.dtype, case.values.profile, &mut rng, input, index)
                })
                .collect();
            CanonicalTensor::new(case.dtype, shape.clone(), bits)
        })
        .collect()
}

/// Evaluate with no production planner/tree/cost/index-program dependency.
///
/// Input factors are multiplied in input order. Reduction coordinates advance
/// lexicographically in canonical axis order. F16/BF16 inputs are promoted to
/// F32, all arithmetic stays F32, and the result narrows once at final output.
pub fn evaluate(case: &CaseRecord, inputs: &[CanonicalTensor]) -> Result<Evaluation, OracleError> {
    case.validate()?;
    if inputs.len() != case.input_shapes.len() {
        return Err(OracleError::InputCount {
            expected: case.input_shapes.len(),
            found: inputs.len(),
        });
    }
    for (index, ((tensor, expected_shape), expected_dtype)) in inputs
        .iter()
        .zip(&case.input_shapes)
        .zip(std::iter::repeat(case.dtype))
        .enumerate()
    {
        if tensor.dtype != expected_dtype {
            return Err(OracleError::InputDtype {
                input: index,
                expected: expected_dtype,
                found: tensor.dtype,
            });
        }
        if tensor.shape != *expected_shape {
            return Err(OracleError::InputShape {
                input: index,
                expected: expected_shape.clone(),
                found: tensor.shape.clone(),
            });
        }
    }
    let analysis = analyze_equation(&case.equation, &case.input_shapes)?;
    let output_count = checked_numel(analysis.output_shape()).ok_or(OracleError::SizeOverflow)?;
    let iteration_positions = analysis
        .iteration_axes()
        .iter()
        .copied()
        .enumerate()
        .map(|(index, axis)| (axis, index))
        .collect::<BTreeMap<_, _>>();
    let output_rank = analysis.output_rank();
    let reduction_count = analysis.reduction_elements();
    let mut output_bits = Vec::with_capacity(output_count);
    let mut condition_scale = Vec::with_capacity(output_count);
    let mut coordinates = vec![0usize; analysis.iteration_axes().len()];
    for output_linear in 0..output_count {
        decode_linear(
            output_linear,
            analysis.output_shape(),
            &mut coordinates[..output_rank],
        );
        let mut accumulator: Option<Arithmetic> = None;
        let mut condition = 0.0f64;
        for reduction_linear in 0..reduction_count {
            decode_linear(
                reduction_linear,
                &analysis.iteration_extents()[output_rank..],
                &mut coordinates[output_rank..],
            );
            let mut product: Option<Arithmetic> = None;
            for (tensor, operand) in inputs.iter().zip(analysis.operands()) {
                let mut offset = 0usize;
                for (physical_axis, (&axis, &stride)) in operand
                    .physical_axes()
                    .iter()
                    .zip(operand.strides())
                    .enumerate()
                {
                    let logical = coordinates[iteration_positions[&axis]];
                    let source = if operand.shape()[physical_axis] == 1 {
                        0
                    } else {
                        logical
                    };
                    offset += source * stride;
                }
                let factor = Arithmetic::from_bits(case.dtype, tensor.bits[offset]);
                product = Some(match product {
                    None => factor,
                    Some(value) => value.mul(factor, case.dtype),
                });
            }
            let product = product.expect("validated Einsum has at least one input");
            condition += product.abs_f64(case.dtype);
            accumulator = Some(match accumulator {
                None => product,
                Some(value) => value.add(product, case.dtype),
            });
        }
        let value = accumulator.unwrap_or_else(|| Arithmetic::zero(case.dtype));
        output_bits.push(value.output_bits(case.dtype));
        condition_scale.push(condition);
    }
    Ok(Evaluation {
        output: CanonicalTensor::new(case.dtype, analysis.output_shape().to_vec(), output_bits)?,
        condition_scale,
        reduction_elements: reduction_count,
        factor_count: inputs.len(),
    })
}

/// Compare a backend result with exact integer/special handling and a
/// condition-aware floating bound.
pub fn compare(
    case: &CaseRecord,
    expected: &Evaluation,
    actual: &CanonicalTensor,
    mode: ComparisonMode,
) -> Result<ComparisonReport, ComparisonFailure> {
    if actual.dtype != expected.output.dtype {
        return Err(ComparisonFailure::Dtype {
            expected: expected.output.dtype,
            found: actual.dtype,
        });
    }
    if actual.shape != expected.output.shape {
        return Err(ComparisonFailure::Shape {
            expected: expected.output.shape.clone(),
            found: actual.shape.clone(),
        });
    }
    let mut report = ComparisonReport {
        elements: actual.bits.len(),
        ..ComparisonReport::default()
    };
    for (index, (&expected_bits, &actual_bits)) in
        expected.output.bits.iter().zip(&actual.bits).enumerate()
    {
        if case.dtype.integer_bits().is_some() || mode == ComparisonMode::CanonicalBits {
            if actual_bits != expected_bits {
                return Err(ComparisonFailure::Exact {
                    index,
                    expected: expected_bits,
                    found: actual_bits,
                });
            }
            continue;
        }
        let expected_value = bits_to_f64(case.dtype, expected_bits);
        let actual_value = bits_to_f64(case.dtype, actual_bits);
        if expected_value.is_nan() {
            if !actual_value.is_nan() {
                return Err(ComparisonFailure::Special {
                    index,
                    expected: "NaN",
                    found: actual_value,
                });
            }
            continue;
        }
        if expected_value.is_infinite() {
            if actual_value != expected_value {
                return Err(ComparisonFailure::Special {
                    index,
                    expected: if expected_value.is_sign_positive() {
                        "+infinity"
                    } else {
                        "-infinity"
                    },
                    found: actual_value,
                });
            }
            continue;
        }
        if expected_value == 0.0
            && actual_value == 0.0
            && case.values.profile == ValueProfile::BinarySpecials
            && expected_value.is_sign_negative() != actual_value.is_sign_negative()
        {
            return Err(ComparisonFailure::SignedZero { index });
        }
        let absolute = (actual_value - expected_value).abs();
        let allowed = allowed_error(case.dtype, expected, index, expected_value);
        report.max_abs_error = report.max_abs_error.max(absolute);
        report.max_allowed_error = report.max_allowed_error.max(allowed);
        if absolute > allowed {
            return Err(ComparisonFailure::Tolerance {
                index,
                expected: expected_value,
                found: actual_value,
                absolute,
                allowed,
                condition_scale: expected.condition_scale[index],
                reduction_elements: expected.reduction_elements,
            });
        }
    }
    Ok(report)
}

fn allowed_error(
    dtype: ConformanceDType,
    evaluation: &Evaluation,
    index: usize,
    expected: f64,
) -> f64 {
    let operations = evaluation
        .reduction_elements
        .saturating_mul(evaluation.factor_count.saturating_sub(1))
        .saturating_add(evaluation.reduction_elements.saturating_sub(1))
        .max(1);
    let accumulator_unit = match dtype {
        ConformanceDType::Float64 => f64::EPSILON / 2.0,
        _ => f64::from(f32::EPSILON) / 2.0,
    };
    let nu = operations as f64 * accumulator_unit;
    let gamma = if nu < 0.5 { nu / (1.0 - nu) } else { 1.0 };
    let accumulation = 8.0 * gamma * evaluation.condition_scale[index];
    let (output_unit, subnormal_floor) = match dtype {
        ConformanceDType::Float16 => (2f64.powi(-11), 2f64.powi(-24)),
        ConformanceDType::BFloat16 => (2f64.powi(-8), 2f64.powi(-133)),
        ConformanceDType::Float32 => (2f64.powi(-24), 2f64.powi(-149)),
        ConformanceDType::Float64 => (2f64.powi(-53), 2f64.powi(-1074)),
        _ => (0.0, 0.0),
    };
    accumulation + 2.0 * output_unit * expected.abs() + subnormal_floor
}

#[derive(Clone, Copy)]
enum Arithmetic {
    F32(f32),
    F64(f64),
    Int(u64),
}

impl Arithmetic {
    fn from_bits(dtype: ConformanceDType, bits: u64) -> Self {
        match dtype {
            ConformanceDType::Float16 => Self::F32(f16::from_bits(bits as u16).to_f32()),
            ConformanceDType::BFloat16 => Self::F32(bf16::from_bits(bits as u16).to_f32()),
            ConformanceDType::Float32 => Self::F32(f32::from_bits(bits as u32)),
            ConformanceDType::Float64 => Self::F64(f64::from_bits(bits)),
            _ => Self::Int(bits),
        }
    }

    fn zero(dtype: ConformanceDType) -> Self {
        match dtype {
            ConformanceDType::Float64 => Self::F64(0.0),
            dtype if dtype.is_float() => Self::F32(0.0),
            _ => Self::Int(0),
        }
    }

    fn mul(self, right: Self, dtype: ConformanceDType) -> Self {
        match (self, right) {
            (Self::F32(left), Self::F32(right)) => Self::F32(left * right),
            (Self::F64(left), Self::F64(right)) => Self::F64(left * right),
            (Self::Int(left), Self::Int(right)) => {
                Self::Int(left.wrapping_mul(right) & integer_mask(dtype))
            }
            _ => unreachable!("homogeneous arithmetic"),
        }
    }

    fn add(self, right: Self, dtype: ConformanceDType) -> Self {
        match (self, right) {
            (Self::F32(left), Self::F32(right)) => Self::F32(left + right),
            (Self::F64(left), Self::F64(right)) => Self::F64(left + right),
            (Self::Int(left), Self::Int(right)) => {
                Self::Int(left.wrapping_add(right) & integer_mask(dtype))
            }
            _ => unreachable!("homogeneous arithmetic"),
        }
    }

    fn abs_f64(self, dtype: ConformanceDType) -> f64 {
        match self {
            Self::F32(value) => f64::from(value).abs(),
            Self::F64(value) => value.abs(),
            Self::Int(bits) => bits_to_f64(dtype, bits).abs(),
        }
    }

    fn output_bits(self, dtype: ConformanceDType) -> u64 {
        match (self, dtype) {
            (Self::F32(value), ConformanceDType::Float16) => {
                u64::from(f16::from_f32(value).to_bits())
            }
            (Self::F32(value), ConformanceDType::BFloat16) => {
                u64::from(bf16::from_f32(value).to_bits())
            }
            (Self::F32(value), ConformanceDType::Float32) => u64::from(value.to_bits()),
            (Self::F64(value), ConformanceDType::Float64) => value.to_bits(),
            (Self::Int(value), _) => value & integer_mask(dtype),
            _ => unreachable!("dtype arithmetic result"),
        }
    }
}

fn generated_bits(
    dtype: ConformanceDType,
    profile: ValueProfile,
    rng: &mut SplitMix64,
    input: usize,
    index: usize,
) -> u64 {
    match profile {
        ValueProfile::Finite => {
            let integer = (rng.next() % 257) as i32 - 128;
            let value = integer as f32 / 17.0;
            float_or_integer_bits(dtype, value, rng.next())
        }
        ValueProfile::IntegerEdges => {
            let mask = integer_mask(dtype);
            let edges = [
                0,
                1,
                mask,
                mask >> 1,
                (mask >> 1).wrapping_add(1) & mask,
                2,
                mask.wrapping_sub(1),
            ];
            edges[(input + index) % edges.len()]
        }
        ValueProfile::BinarySpecials => special_bits(dtype, input + index),
    }
}

fn float_or_integer_bits(dtype: ConformanceDType, value: f32, random: u64) -> u64 {
    match dtype {
        ConformanceDType::Float16 => u64::from(f16::from_f32(value).to_bits()),
        ConformanceDType::BFloat16 => u64::from(bf16::from_f32(value).to_bits()),
        ConformanceDType::Float32 => u64::from(value.to_bits()),
        ConformanceDType::Float64 => (value as f64).to_bits(),
        _ => random & integer_mask(dtype),
    }
}

fn special_bits(dtype: ConformanceDType, index: usize) -> u64 {
    let values: &[u64] = match dtype {
        ConformanceDType::Float16 => &[
            0x0000, 0x8000, 0x3c00, 0xbc00, 0x7c00, 0xfc00, 0x7e01, 0x0001, 0x7bff,
        ],
        ConformanceDType::BFloat16 => &[
            0x0000, 0x8000, 0x3f80, 0xbf80, 0x7f80, 0xff80, 0x7fc1, 0x0001, 0x7f7f,
        ],
        ConformanceDType::Float32 => &[
            0x0000_0000,
            0x8000_0000,
            0x3f80_0000,
            0xbf80_0000,
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_0001,
            0x0000_0001,
            0x7f7f_ffff,
        ],
        ConformanceDType::Float64 => &[
            0x0000_0000_0000_0000,
            0x8000_0000_0000_0000,
            0x3ff0_0000_0000_0000,
            0xbff0_0000_0000_0000,
            0x7ff0_0000_0000_0000,
            0xfff0_0000_0000_0000,
            0x7ff8_0000_0000_0001,
            0x0000_0000_0000_0001,
            0x7fef_ffff_ffff_ffff,
        ],
        _ => return index as u64 & integer_mask(dtype),
    };
    values[index % values.len()]
}

fn bits_to_f64(dtype: ConformanceDType, bits: u64) -> f64 {
    match dtype {
        ConformanceDType::Float16 => f64::from(f16::from_bits(bits as u16).to_f32()),
        ConformanceDType::BFloat16 => f64::from(bf16::from_bits(bits as u16).to_f32()),
        ConformanceDType::Float32 => f64::from(f32::from_bits(bits as u32)),
        ConformanceDType::Float64 => f64::from_bits(bits),
        ConformanceDType::Uint8
        | ConformanceDType::Uint16
        | ConformanceDType::Uint32
        | ConformanceDType::Uint64 => (bits & integer_mask(dtype)) as f64,
        ConformanceDType::Int8
        | ConformanceDType::Int16
        | ConformanceDType::Int32
        | ConformanceDType::Int64 => signed_value(bits, dtype) as f64,
    }
}

fn signed_value(bits: u64, dtype: ConformanceDType) -> i64 {
    let width = dtype.integer_bits().expect("integer dtype");
    if width == 64 {
        bits as i64
    } else {
        let shift = 64 - width;
        ((bits << shift) as i64) >> shift
    }
}

fn integer_mask(dtype: ConformanceDType) -> u64 {
    match dtype.byte_size() {
        1 => 0xff,
        2 => 0xffff,
        4 => 0xffff_ffff,
        8 => u64::MAX,
        _ => unreachable!("supported fixed-width dtype"),
    }
}

fn checked_numel(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |count, &dim| count.checked_mul(dim))
}

fn decode_linear(mut linear: usize, shape: &[usize], output: &mut [usize]) {
    debug_assert_eq!(shape.len(), output.len());
    for axis in (0..shape.len()).rev() {
        if shape[axis] == 0 {
            output[axis] = 0;
        } else {
            output[axis] = linear % shape[axis];
            linear /= shape[axis];
        }
    }
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }
}

/// Direct oracle construction/evaluation failure.
#[derive(Debug, Error)]
pub enum OracleError {
    /// Case signature or limits are invalid.
    #[error(transparent)]
    Case(#[from] CaseValidationError),
    /// Equation analysis failed.
    #[error(transparent)]
    Equation(#[from] EquationError),
    /// Shape arithmetic overflow.
    #[error("Einsum oracle tensor size overflow")]
    SizeOverflow,
    /// Tensor bits do not match its shape.
    #[error("Einsum oracle tensor expected {expected} elements, found {found}")]
    ElementCount {
        /// Shape element count.
        expected: usize,
        /// Payload count.
        found: usize,
    },
    /// Tensor bits exceed the dtype width.
    #[error("Einsum oracle tensor contains bits outside the {dtype:?} storage width")]
    OutOfRangeBits {
        /// Tensor dtype.
        dtype: ConformanceDType,
    },
    /// Runtime input count mismatch.
    #[error("Einsum oracle expected {expected} inputs, found {found}")]
    InputCount {
        /// Expected count.
        expected: usize,
        /// Found count.
        found: usize,
    },
    /// Runtime input dtype mismatch.
    #[error("Einsum oracle input #{input} expected {expected:?}, found {found:?}")]
    InputDtype {
        /// Input index.
        input: usize,
        /// Expected dtype.
        expected: ConformanceDType,
        /// Found dtype.
        found: ConformanceDType,
    },
    /// Runtime input shape mismatch.
    #[error("Einsum oracle input #{input} expected shape {expected:?}, found {found:?}")]
    InputShape {
        /// Input index.
        input: usize,
        /// Expected shape.
        expected: Vec<usize>,
        /// Found shape.
        found: Vec<usize>,
    },
}

/// Backend output mismatch.
#[derive(Debug, Error)]
pub enum ComparisonFailure {
    /// Dtype mismatch.
    #[error("Einsum output dtype mismatch: expected {expected:?}, found {found:?}")]
    Dtype {
        /// Expected dtype.
        expected: ConformanceDType,
        /// Found dtype.
        found: ConformanceDType,
    },
    /// Shape mismatch.
    #[error("Einsum output shape mismatch: expected {expected:?}, found {found:?}")]
    Shape {
        /// Expected shape.
        expected: Vec<usize>,
        /// Found shape.
        found: Vec<usize>,
    },
    /// Exact bit mismatch.
    #[error(
        "Einsum output element {index} bit mismatch: expected 0x{expected:016x}, found 0x{found:016x}"
    )]
    Exact {
        /// Element index.
        index: usize,
        /// Expected bits.
        expected: u64,
        /// Found bits.
        found: u64,
    },
    /// Nonfinite-class mismatch.
    #[error("Einsum output element {index} expected {expected}, found {found}")]
    Special {
        /// Element index.
        index: usize,
        /// Expected class.
        expected: &'static str,
        /// Found value.
        found: f64,
    },
    /// Signed zero mismatch.
    #[error("Einsum output element {index} did not preserve the expected signed zero")]
    SignedZero {
        /// Element index.
        index: usize,
    },
    /// Finite tolerance mismatch.
    #[error(
        "Einsum output element {index} found {found}, expected {expected}: absolute error {absolute} exceeds condition-aware allowance {allowed} (condition scale {condition_scale}, reduction elements {reduction_elements})"
    )]
    Tolerance {
        /// Element index.
        index: usize,
        /// Expected value.
        expected: f64,
        /// Found value.
        found: f64,
        /// Absolute error.
        absolute: f64,
        /// Allowed error.
        allowed: f64,
        /// Sum of absolute products.
        condition_scale: f64,
        /// Reduction terms.
        reduction_elements: usize,
    },
}

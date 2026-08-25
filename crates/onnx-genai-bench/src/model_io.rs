//! Synthetic model inputs and native-versus-ORT output comparison.
//!
//! Extracted from the `bench_generic` binary so every model-level benchmark in
//! this crate builds its inputs and judges parity the same way. Two harnesses
//! that synthesize inputs differently are not comparable, and two that apply
//! different tolerances can disagree about whether the same kernel is correct.

use anyhow::{Context, Result, bail};
use onnx_genai_ort::{DataType as OrtDataType, Session, Value};
use onnx_runtime_ir::{DataType as NativeDataType, Dim};
use onnx_runtime_session::{InferenceSession, Tensor};

/// Machine epsilon of IEEE binary16 (`2^-10`). f16 carries a 10-bit mantissa,
/// so two f16 values that differ by one ULP near 1.0 differ by this much.
pub const F16_EPSILON: f32 = 9.765_625e-4;

pub struct InputPair {
    pub name: String,
    pub shape: Vec<usize>,
    pub native: Tensor,
    pub ort: Value,
}

#[derive(Debug)]
pub struct OutputDiff {
    pub index: usize,
    pub max_abs: f32,
    pub max_rel: f32,
    pub pass: bool,
}

pub fn parse_shape(value: &str) -> std::result::Result<Vec<usize>, String> {
    let shape = value
        .split([',', 'x', 'X'])
        .map(|dim| {
            dim.trim()
                .parse::<usize>()
                .map_err(|error| format!("invalid dimension '{dim}': {error}"))
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if shape.is_empty() || shape.contains(&0) {
        return Err("input shape must contain only positive dimensions".to_string());
    }
    Ok(shape)
}

pub fn validate_tolerance(name: &str, value: f32) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        bail!("--{name} must be finite and non-negative");
    }
    Ok(())
}

pub fn resolved_shape(declared: &[Dim], override_shape: Option<&[usize]>) -> Result<Vec<usize>> {
    if let Some(shape) = override_shape {
        if shape.len() != declared.len() {
            bail!(
                "--input-shape rank {} does not match declared input rank {}",
                shape.len(),
                declared.len()
            );
        }
        return Ok(shape.to_vec());
    }

    let rank = declared.len();
    Ok(declared
        .iter()
        .enumerate()
        .map(|(axis, dim)| {
            dim.as_static().unwrap_or_else(|| {
                if rank >= 4 && axis >= rank - 2 {
                    224
                } else {
                    1
                }
            })
        })
        .collect())
}

pub fn resolved_ort_shape(
    declared: &[i64],
    override_shape: Option<&[usize]>,
) -> Result<Vec<usize>> {
    if let Some(shape) = override_shape {
        if shape.len() != declared.len() {
            bail!(
                "--input-shape rank {} does not match declared input rank {}",
                shape.len(),
                declared.len()
            );
        }
        return Ok(shape.to_vec());
    }
    let rank = declared.len();
    declared
        .iter()
        .enumerate()
        .map(|(axis, &dim)| {
            if dim > 0 {
                usize::try_from(dim).context("declared ORT input dimension exceeds usize")
            } else if rank >= 4 && axis >= rank - 2 {
                Ok(224)
            } else {
                Ok(1)
            }
        })
        .collect()
}

pub fn element_count(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |count, &dim| {
        count
            .checked_mul(dim)
            .context("input shape element count overflow")
    })
}

pub fn synthetic_f32(count: usize) -> Vec<f32> {
    (0..count)
        .map(|index| ((index.wrapping_mul(37) % 257) as f32 - 128.0) / 128.0)
        .collect()
}

pub fn synthetic_i64(count: usize) -> Vec<i64> {
    (0..count).map(|index| (index % 17) as i64).collect()
}

/// Float16 bit patterns for the same values [`synthetic_f32`] produces, so a
/// Float16 graph is fed the numerically closest version of the f32 input.
pub fn synthetic_f16_bits(count: usize) -> Vec<u16> {
    synthetic_f32(count)
        .into_iter()
        .map(|value| half::f16::from_f32(value).to_bits())
        .collect()
}

/// Unsigned 8-bit inputs spread over the whole quantized range (QLinearMatMul
/// and friends interpret these through a scale/zero-point, so the raw spread
/// matters more than the float value).
pub fn synthetic_u8(count: usize) -> Vec<u8> {
    (0..count)
        .map(|index| (index.wrapping_mul(37) % 251) as u8)
        .collect()
}

pub fn synthetic_i8_bytes(count: usize) -> Vec<u8> {
    (0..count)
        .map(|index| (((index.wrapping_mul(37) % 251) as i32 - 125) as i8) as u8)
        .collect()
}

pub fn synthetic_i32(count: usize) -> Vec<i32> {
    (0..count).map(|index| (index % 17) as i32).collect()
}

pub fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

pub fn build_inputs(
    native_session: &InferenceSession,
    ort_session: &Session,
    override_shape: Option<&[usize]>,
) -> Result<Vec<InputPair>> {
    if native_session.inputs().len() != ort_session.inputs().len() {
        bail!(
            "runtime input-count mismatch: native={} ORT={}",
            native_session.inputs().len(),
            ort_session.inputs().len()
        );
    }

    native_session
        .inputs()
        .iter()
        .enumerate()
        .map(|(index, input)| {
            let ort_input = &ort_session.inputs()[index];
            if input.name != ort_input.name {
                bail!(
                    "runtime input-name mismatch at index {index}: native='{}' ORT='{}'",
                    input.name,
                    ort_input.name
                );
            }
            let shape = resolved_shape(
                &input.shape,
                (index == 0).then_some(override_shape).flatten(),
            )?;
            let ort_shape = shape
                .iter()
                .map(|&dim| i64::try_from(dim).context("input dimension exceeds i64"))
                .collect::<Result<Vec<_>>>()?;
            let count = element_count(&shape)?;
            let (native, ort) = match (input.dtype, ort_input.dtype) {
                (NativeDataType::Float32, OrtDataType::Float32) => {
                    let data = synthetic_f32(count);
                    (
                        Tensor::from_f32(&shape, &data)?,
                        Value::from_slice_f32(&data, &ort_shape)?,
                    )
                }
                (NativeDataType::Int64, OrtDataType::Int64) => {
                    let data = synthetic_i64(count);
                    (
                        Tensor::from_i64(&shape, &data)?,
                        Value::from_slice_i64(&data, &ort_shape)?,
                    )
                }
                (NativeDataType::Int32, OrtDataType::Int32) => {
                    let bytes = i32_bytes(&synthetic_i32(count));
                    (
                        Tensor::from_raw(NativeDataType::Int32, shape.clone(), &bytes)?,
                        Value::from_raw_bytes(bytes, &ort_shape, OrtDataType::Int32)?,
                    )
                }
                (NativeDataType::Float16, OrtDataType::Float16) => {
                    let bits = synthetic_f16_bits(count);
                    let bytes = bits.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
                    (
                        Tensor::from_raw(NativeDataType::Float16, shape.clone(), &bytes)?,
                        Value::from_slice_f16_bits(&bits, &ort_shape)?,
                    )
                }
                (NativeDataType::Uint8, OrtDataType::Uint8) => {
                    let bytes = synthetic_u8(count);
                    (
                        Tensor::from_raw(NativeDataType::Uint8, shape.clone(), &bytes)?,
                        Value::from_raw_bytes(bytes, &ort_shape, OrtDataType::Uint8)?,
                    )
                }
                (NativeDataType::Int8, OrtDataType::Int8) => {
                    let bytes = synthetic_i8_bytes(count);
                    (
                        Tensor::from_raw(NativeDataType::Int8, shape.clone(), &bytes)?,
                        Value::from_raw_bytes(bytes, &ort_shape, OrtDataType::Int8)?,
                    )
                }
                (native, ort) => bail!(
                    "input '{}' has unsupported or mismatched dtype: native={native:?} ORT={ort:?}; \
                     bench_generic currently synthesizes Float32, Float16, Int32, Int64, Uint8, \
                     and Int8 inputs",
                    input.name
                ),
            };
            Ok(InputPair {
                name: input.name.clone(),
                shape,
                native,
                ort,
            })
        })
        .collect()
}

/// Largest absolute and relative gap between two f32 sequences, plus whether
/// every element is inside `abs_tolerance + rel_tolerance * max(|a|, |b|)`.
pub fn compare_f32(
    native: &[f32],
    ort: &[f32],
    abs_tolerance: f32,
    rel_tolerance: f32,
) -> (f32, f32, bool) {
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    let mut pass = true;
    for (&native, &ort) in native.iter().zip(ort) {
        if native == ort {
            continue;
        }
        if !native.is_finite() || !ort.is_finite() {
            max_abs = f32::INFINITY;
            max_rel = f32::INFINITY;
            pass = false;
            continue;
        }
        let abs = (native - ort).abs();
        let rel = abs / native.abs().max(ort.abs()).max(f32::MIN_POSITIVE);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        pass &= abs <= abs_tolerance + rel_tolerance * native.abs().max(ort.abs());
    }
    (max_abs, max_rel, pass)
}

pub fn compare_outputs(
    native: &[Tensor],
    ort: &[Value],
    abs_tolerance: f32,
    rel_tolerance: f32,
    f16_abs_tolerance: f32,
    f16_rel_tolerance: f32,
) -> Result<Vec<OutputDiff>> {
    if native.len() != ort.len() {
        bail!(
            "runtime output-count mismatch: native={} ORT={}",
            native.len(),
            ort.len()
        );
    }
    native
        .iter()
        .zip(ort)
        .enumerate()
        .map(|(index, (native, ort))| {
            if native
                .shape
                .iter()
                .copied()
                .map(|dim| dim as i64)
                .ne(ort.shape().iter().copied())
            {
                bail!(
                    "output {index} shape mismatch: native={:?} ORT={:?}",
                    native.shape,
                    ort.shape()
                );
            }
            match (native.dtype, ort.dtype()) {
                (NativeDataType::Float32, OrtDataType::Float32) => {
                    let (max_abs, max_rel, pass) = compare_f32(
                        &native.to_vec_f32(),
                        &ort.to_vec_f32()?,
                        abs_tolerance,
                        rel_tolerance,
                    );
                    Ok(OutputDiff {
                        index,
                        max_abs,
                        max_rel,
                        pass,
                    })
                }
                (NativeDataType::Float16, OrtDataType::Float16) => {
                    let widen = |bits: &[u16]| -> Vec<f32> {
                        bits.iter()
                            .map(|&bits| half::f16::from_bits(bits).to_f32())
                            .collect()
                    };
                    let native_bits: Vec<u16> = native
                        .as_bytes()
                        .chunks_exact(2)
                        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
                        .collect();
                    // f16-scaled tolerances: the f32 defaults are ~1 f16 ULP,
                    // which would pass almost any pair of f16 values and make
                    // the parity check meaningless.
                    let (max_abs, max_rel, pass) = compare_f32(
                        &widen(&native_bits),
                        &widen(&ort.to_vec_f16_bits()?),
                        f16_abs_tolerance,
                        f16_rel_tolerance,
                    );
                    Ok(OutputDiff {
                        index,
                        max_abs,
                        max_rel,
                        pass,
                    })
                }
                (NativeDataType::Uint8, OrtDataType::Uint8)
                | (NativeDataType::Int8, OrtDataType::Int8) => {
                    // Quantized outputs are exact integers: any mismatch is a
                    // real disagreement, so report the largest code-unit gap and
                    // require zero of them.
                    let ort_bytes = ort.to_raw_bytes()?;
                    let signed = native.dtype == NativeDataType::Int8;
                    let max_abs = native
                        .as_bytes()
                        .iter()
                        .zip(&ort_bytes)
                        .map(|(&native, &ort)| {
                            if signed {
                                ((native as i8) as i32 - (ort as i8) as i32).unsigned_abs()
                            } else {
                                (native as i32 - ort as i32).unsigned_abs()
                            }
                        })
                        .max()
                        .unwrap_or(0);
                    Ok(OutputDiff {
                        index,
                        max_abs: max_abs as f32,
                        max_rel: 0.0,
                        pass: max_abs == 0,
                    })
                }
                (NativeDataType::Int64, OrtDataType::Int64) => {
                    let pass = native.to_vec_i64() == ort.to_vec_i64()?;
                    Ok(OutputDiff {
                        index,
                        max_abs: if pass { 0.0 } else { f32::INFINITY },
                        max_rel: if pass { 0.0 } else { f32::INFINITY },
                        pass,
                    })
                }
                (native_dtype, ort_dtype) => bail!(
                    "output {index} has unsupported or mismatched dtype: \
                     native={native_dtype:?} ORT={ort_dtype:?}; parity supports Float32 and Int64"
                ),
            }
        })
        .collect()
}

pub fn classifier_top1_native(output: &Tensor) -> Option<usize> {
    (output.dtype == NativeDataType::Float32
        && output.shape.len() == 2
        && output.shape[0] == 1
        && output.shape[1] > 1)
        .then(|| argmax(&output.to_vec_f32()))
}

pub fn classifier_top1_ort(output: &Value) -> Result<Option<usize>> {
    Ok((output.dtype() == OrtDataType::Float32
        && output.shape().len() == 2
        && output.shape()[0] == 1
        && output.shape()[1] > 1)
        .then(|| output.to_vec_f32().map(|values| argmax(&values)))
        .transpose()?)
}

pub fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map_or(0, |(index, _)| index)
}

pub fn median_ms(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

/// p50/p90/min of one runtime's samples. p90 uses the nearest-rank definition
/// (`ceil(0.9 * n)`-th smallest), so a 10-run comparison reports the 9th
/// sample rather than interpolating; dispersion is reported as p90/p50 so a
/// noisy shared host is visible in the record instead of hidden by the median.
#[derive(Clone, Copy)]
pub struct Stats {
    pub p50: f64,
    pub p90: f64,
    pub min: f64,
}

impl Stats {
    /// Nearest-rank percentiles over `samples`, which must be non-empty (the
    /// caller bails on `--runs 0`; this asserts rather than panicking on an
    /// out-of-bounds index if a future caller filters samples down to nothing).
    pub fn from(mut samples: Vec<f64>) -> Self {
        assert!(
            !samples.is_empty(),
            "Stats::from requires at least one timing sample"
        );
        samples.sort_by(f64::total_cmp);
        let rank = ((samples.len() as f64) * 0.9).ceil().max(1.0) as usize;
        Self {
            p50: samples[samples.len() / 2],
            p90: samples[rank.min(samples.len()) - 1],
            min: samples[0],
        }
    }

    pub fn spread(&self) -> f64 {
        self.p90 / self.p50
    }
}

/// Which CPU-kernel arm this binary was built with.
///
/// Printed on every result line because the distinction is not cosmetic: `mlas`
/// is not a default feature of `onnx-runtime-ep-cpu`, so an MLAS-linked build
/// does not measure what ships. This binary used to *require* the `mlas`
/// feature, which meant every ratio ever published from it came from the
/// research arm while being read as a production number. Labelling the arm in
/// the output makes that impossible to do again by accident.
pub fn build_arm() -> &'static str {
    if cfg!(feature = "mlas") {
        "mlas-reference"
    } else {
        "native"
    }
}

/// Synthetic inputs for an ORT session alone, for arms that never build a
/// native session (an ORT baseline, or the ORT half of a solo-arm comparison).
///
/// Shares [`resolved_ort_shape`] and the `synthetic_*` generators with
/// [`build_inputs`], so an ORT-only arm is fed byte-identical tensors to the
/// ones the paired arm would have fed it.
pub fn build_ort_inputs(
    session: &Session,
    override_shape: Option<&[usize]>,
) -> Result<Vec<(String, Value)>> {
    session
        .inputs()
        .iter()
        .enumerate()
        .map(|(index, input)| {
            let shape = resolved_ort_shape(
                &input.shape,
                (index == 0).then_some(override_shape).flatten(),
            )?;
            let ort_shape = shape
                .iter()
                .map(|&dim| i64::try_from(dim).context("input dimension exceeds i64"))
                .collect::<Result<Vec<_>>>()?;
            let count = element_count(&shape)?;
            let value = match input.dtype {
                OrtDataType::Float32 => Value::from_slice_f32(&synthetic_f32(count), &ort_shape)?,
                OrtDataType::Int64 => Value::from_slice_i64(&synthetic_i64(count), &ort_shape)?,
                OrtDataType::Int32 => Value::from_raw_bytes(
                    i32_bytes(&synthetic_i32(count)),
                    &ort_shape,
                    OrtDataType::Int32,
                )?,
                dtype => bail!(
                    "input '{}' has unsupported dtype {dtype:?}; only Float32, Int32, and \
                     Int64 inputs are synthesized",
                    input.name
                ),
            };
            Ok((input.name.clone(), value))
        })
        .collect()
}

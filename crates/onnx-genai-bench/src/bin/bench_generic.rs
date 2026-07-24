//! Generic native nxrt versus ONNX Runtime CPU single-inference benchmark.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Parser;
use onnx_genai_ort::{
    DataType as OrtDataType, Environment, Session, SessionOptions, Value, ep_selection,
};
use onnx_runtime_ir::{DataType as NativeDataType, Dim};
use onnx_runtime_session::{InferenceSession, Tensor};

#[derive(Debug, Parser)]
#[command(about = "Compare native nxrt and ONNX Runtime CPU single inference on any ONNX model")]
struct Args {
    /// ONNX model file.
    #[arg(long)]
    model: PathBuf,
    /// Number of measured runs per runtime.
    #[arg(long, default_value_t = 10)]
    runs: usize,
    /// Number of untimed warmup runs per runtime.
    #[arg(long, default_value_t = 3)]
    warmups: usize,
    /// Override the first model input shape, for example 1,3,416,416.
    #[arg(long)]
    input_shape: Option<String>,
    /// Measure ORT only. Useful for recording a baseline when native loading or execution fails.
    #[arg(long)]
    ort_only: bool,
    /// Time the native runtime only. Parity is still checked once against ORT, but the timed loop
    /// runs native alone so ORT's intra-op threadpool is not spinning and polluting native samples.
    #[arg(long)]
    native_only: bool,
    /// Relative tolerance used for Float32 output parity.
    #[arg(long, default_value_t = 1e-3)]
    rel_tolerance: f32,
    /// Absolute tolerance used for Float32 output parity.
    #[arg(long, default_value_t = 1e-4)]
    abs_tolerance: f32,
}

struct InputPair {
    name: String,
    shape: Vec<usize>,
    native: Tensor,
    ort: Value,
}

#[derive(Debug)]
struct OutputDiff {
    index: usize,
    max_abs: f32,
    max_rel: f32,
    pass: bool,
}

fn parse_shape(value: &str) -> std::result::Result<Vec<usize>, String> {
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

fn validate_tolerance(name: &str, value: f32) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        bail!("--{name} must be finite and non-negative");
    }
    Ok(())
}

fn resolved_shape(declared: &[Dim], override_shape: Option<&[usize]>) -> Result<Vec<usize>> {
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

fn resolved_ort_shape(declared: &[i64], override_shape: Option<&[usize]>) -> Result<Vec<usize>> {
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

fn element_count(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |count, &dim| {
        count
            .checked_mul(dim)
            .context("input shape element count overflow")
    })
}

fn synthetic_f32(count: usize) -> Vec<f32> {
    (0..count)
        .map(|index| ((index.wrapping_mul(37) % 257) as f32 - 128.0) / 128.0)
        .collect()
}

fn synthetic_i64(count: usize) -> Vec<i64> {
    (0..count).map(|index| (index % 17) as i64).collect()
}

fn synthetic_i32(count: usize) -> Vec<i32> {
    (0..count).map(|index| (index % 17) as i32).collect()
}

fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn build_inputs(
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
                (native, ort) => bail!(
                    "input '{}' has unsupported or mismatched dtype: native={native:?} ORT={ort:?}; \
                     bench_generic currently synthesizes Float32, Int32, and Int64 inputs",
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

fn run_ort_only(
    session: &Session,
    override_shape: Option<&[usize]>,
    warmups: usize,
    runs: usize,
) -> Result<()> {
    let inputs = session
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
                    "input '{}' has unsupported dtype {dtype:?}; bench_generic currently \
                     synthesizes Float32, Int32, and Int64 inputs",
                    input.name
                ),
            };
            println!("input: {} {:?} shape={shape:?}", input.name, input.dtype);
            Ok((input.name.clone(), value))
        })
        .collect::<Result<Vec<_>>>()?;
    let input_refs = inputs
        .iter()
        .map(|(name, value)| (name.as_str(), value))
        .collect::<Vec<_>>();
    for _ in 0..warmups {
        std::hint::black_box(session.run(&input_refs).context("ORT warmup")?);
    }
    let mut samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        std::hint::black_box(session.run(&input_refs).context("ORT measured run")?);
        samples.push(start.elapsed().as_secs_f64() * 1_000.0);
    }
    let ort_ms = median_ms(samples);
    println!(
        "result: native=FAIL ort={ort_ms:.3} ms ({:.2} infer/s) native/ort=N/A parity=N/A",
        1_000.0 / ort_ms
    );
    Ok(())
}

fn compare_outputs(
    native: &[Tensor],
    ort: &[Value],
    abs_tolerance: f32,
    rel_tolerance: f32,
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
                    let native = native.to_vec_f32();
                    let ort = ort.to_vec_f32()?;
                    let mut max_abs = 0.0_f32;
                    let mut max_rel = 0.0_f32;
                    let mut pass = true;
                    for (&native, &ort) in native.iter().zip(&ort) {
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
                    Ok(OutputDiff {
                        index,
                        max_abs,
                        max_rel,
                        pass,
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

fn classifier_top1_native(output: &Tensor) -> Option<usize> {
    (output.dtype == NativeDataType::Float32
        && output.shape.len() == 2
        && output.shape[0] == 1
        && output.shape[1] > 1)
        .then(|| argmax(&output.to_vec_f32()))
}

fn classifier_top1_ort(output: &Value) -> Result<Option<usize>> {
    Ok((output.dtype() == OrtDataType::Float32
        && output.shape().len() == 2
        && output.shape()[0] == 1
        && output.shape()[1] > 1)
        .then(|| output.to_vec_f32().map(|values| argmax(&values)))
        .transpose()?)
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map_or(0, |(index, _)| index)
}

fn median_ms(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.runs == 0 {
        bail!("--runs must be greater than zero");
    }
    validate_tolerance("rel-tolerance", args.rel_tolerance)?;
    validate_tolerance("abs-tolerance", args.abs_tolerance)?;
    let input_shape = args
        .input_shape
        .as_deref()
        .map(parse_shape)
        .transpose()
        .map_err(anyhow::Error::msg)?;

    let environment = Environment::new("bench-generic")?;
    let ort_intra_threads = if args.native_only { 1 } else { 0 };
    let ort_options = SessionOptions::with_execution_provider(ep_selection("cpu"))
        .with_intra_op_threads(ort_intra_threads);
    let ort_session = Session::new(&environment, &args.model, ort_options)
        .with_context(|| format!("load ORT CPU session from {}", args.model.display()))?;
    println!("model: {}", args.model.display());
    if args.ort_only {
        return run_ort_only(
            &ort_session,
            input_shape.as_deref(),
            args.warmups,
            args.runs,
        );
    }

    let mut native_session = InferenceSession::load(&args.model)
        .with_context(|| format!("load native session from {}", args.model.display()))?;
    let inputs = build_inputs(&native_session, &ort_session, input_shape.as_deref())?;

    for input in &inputs {
        println!(
            "input: {} {:?} shape={:?}",
            input.name, input.native.dtype, input.shape
        );
    }
    for (index, output) in native_session.outputs().iter().enumerate() {
        println!(
            "output[{index}]: {} {:?} declared_shape={:?}",
            output.name, output.dtype, output.shape
        );
    }

    let native_inputs = inputs
        .iter()
        .map(|input| (input.name.as_str(), &input.native))
        .collect::<Vec<_>>();
    let ort_inputs = inputs
        .iter()
        .map(|input| (input.name.as_str(), &input.ort))
        .collect::<Vec<_>>();

    let native_reference = native_session
        .run(&native_inputs)
        .context("native parity run")?;
    let ort_reference = ort_session.run(&ort_inputs).context("ORT parity run")?;
    let diffs = compare_outputs(
        &native_reference,
        &ort_reference,
        args.abs_tolerance,
        args.rel_tolerance,
    )?;
    for diff in &diffs {
        println!(
            "parity_output[{}]: max_abs={:.6e} max_rel={:.6e} {}",
            diff.index,
            diff.max_abs,
            diff.max_rel,
            if diff.pass { "PASS" } else { "FAIL" }
        );
    }
    let parity_pass = diffs.iter().all(|diff| diff.pass);

    let native_top1 = native_reference.first().and_then(classifier_top1_native);
    let ort_top1 = ort_reference
        .first()
        .map(classifier_top1_ort)
        .transpose()?
        .flatten();
    match (native_top1, ort_top1) {
        (Some(native), Some(ort)) => println!(
            "top1: native={native} ort={ort} {}",
            if native == ort { "AGREE" } else { "DISAGREE" }
        ),
        _ => println!("top1: N/A (first output is not a [1, classes] Float32 tensor)"),
    }

    for _ in 0..args.warmups {
        std::hint::black_box(
            native_session
                .run(&native_inputs)
                .context("native warmup")?,
        );
    }
    if !args.native_only {
        for _ in 0..args.warmups {
            std::hint::black_box(ort_session.run(&ort_inputs).context("ORT warmup")?);
        }
    }
    let mut native_samples = Vec::with_capacity(args.runs);
    let mut ort_samples = Vec::with_capacity(args.runs);
    for run in 0..args.runs {
        let mut measure_native = || -> Result<f64> {
            let start = Instant::now();
            std::hint::black_box(
                native_session
                    .run(&native_inputs)
                    .context("native measured run")?,
            );
            Ok(start.elapsed().as_secs_f64() * 1_000.0)
        };
        let measure_ort = || -> Result<f64> {
            let start = Instant::now();
            std::hint::black_box(ort_session.run(&ort_inputs).context("ORT measured run")?);
            Ok(start.elapsed().as_secs_f64() * 1_000.0)
        };
        if args.native_only {
            native_samples.push(measure_native()?);
        } else if run % 2 == 0 {
            native_samples.push(measure_native()?);
            ort_samples.push(measure_ort()?);
        } else {
            ort_samples.push(measure_ort()?);
            native_samples.push(measure_native()?);
        }
    }

    let native_ms = median_ms(native_samples);
    if args.native_only {
        println!(
            "result: native={native_ms:.3} ms ({:.2} infer/s) ort=skipped native-only=true \
             parity={}",
            1_000.0 / native_ms,
            if parity_pass { "PASS" } else { "FAIL" }
        );
        return Ok(());
    }
    let ort_ms = median_ms(ort_samples);
    println!(
        "result: native={native_ms:.3} ms ({:.2} infer/s) ort={ort_ms:.3} ms \
         ({:.2} infer/s) native/ort={:.3} parity={}",
        1_000.0 / native_ms,
        1_000.0 / ort_ms,
        native_ms / ort_ms,
        if parity_pass { "PASS" } else { "FAIL" }
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_common_shape_spellings() {
        assert_eq!(parse_shape("1,3,224,224").unwrap(), [1, 3, 224, 224]);
        assert_eq!(parse_shape("1x3x416x416").unwrap(), [1, 3, 416, 416]);
        assert!(parse_shape("1,0,224").is_err());
    }

    #[test]
    fn serializes_synthetic_i32_inputs_little_endian() {
        assert_eq!(synthetic_i32(3), [0, 1, 2]);
        assert_eq!(
            i32_bytes(&[-1, 0x0102_0304]),
            [0xff, 0xff, 0xff, 0xff, 0x04, 0x03, 0x02, 0x01]
        );
    }

    #[test]
    fn resolves_dynamic_batch_and_spatial_dimensions() {
        let shape = [
            Dim::Symbolic(onnx_runtime_ir::SymbolId(0)),
            Dim::Static(3),
            Dim::Symbolic(onnx_runtime_ir::SymbolId(1)),
            Dim::Symbolic(onnx_runtime_ir::SymbolId(2)),
        ];
        assert_eq!(resolved_shape(&shape, None).unwrap(), [1, 3, 224, 224]);
    }

    #[test]
    fn override_must_match_declared_rank() {
        let shape = [Dim::Static(1), Dim::Static(3)];
        assert!(resolved_shape(&shape, Some(&[1, 3, 224, 224])).is_err());
    }

    #[test]
    fn non_finite_output_mismatches_fail_parity() {
        let native = Tensor::from_f32(&[3], &[f32::INFINITY, f32::NEG_INFINITY, f32::NAN]).unwrap();
        let ort = Value::from_slice_f32(&[1.0, f32::INFINITY, f32::NAN], &[3]).unwrap();
        let diffs = compare_outputs(&[native], &[ort], 1e-4, 1e-3).unwrap();
        assert!(!diffs[0].pass);
        assert_eq!(diffs[0].max_abs, f32::INFINITY);
    }

    #[test]
    fn parity_tolerances_must_be_finite_and_non_negative() {
        assert!(validate_tolerance("rel-tolerance", 1e-3).is_ok());
        assert!(validate_tolerance("rel-tolerance", f32::INFINITY).is_err());
        assert!(validate_tolerance("abs-tolerance", f32::NAN).is_err());
        assert!(validate_tolerance("abs-tolerance", -1.0).is_err());
    }
}

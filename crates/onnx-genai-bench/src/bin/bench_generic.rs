//! Generic native nxrt versus ONNX Runtime CPU single-inference benchmark.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Parser;
use onnx_genai_ort::{
    DataType as OrtDataType, Environment, Session, SessionOptions, Value, ep_selection,
};
use onnx_runtime_ep_cpu::decode_spmd::DecodeWidth;
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
    /// Report the native executor's per-run phase breakdown (setup, shape
    /// resolution, buffer sizing, node execution, graph-output collection)
    /// after the measured runs. Warmups are excluded: the accumulator is reset
    /// once warmup finishes, so every printed total covers exactly `--runs`
    /// native runs. This is the only way to attribute the part of a run that
    /// the per-op profiler (`ONNX_GENAI_PROFILE_OPS=1`) leaves undifferentiated,
    /// because that one times node execution and nothing around it.
    ///
    /// Does not enable the activation-memory planner. That planner re-plans
    /// every activation on every run - work the shipped runtime never does -
    /// so while this flag switched it on, the profiler perturbed what it was
    /// measuring and then reported its own cost back as a phase of the run.
    /// Set `NXRT_ACTIVATION_MEMORY_PLAN=1` to opt into it deliberately.
    #[arg(long)]
    phase_profile: bool,
    /// Measure ORT only. Useful for recording a baseline when native loading or execution fails.
    #[arg(long)]
    ort_only: bool,
    /// Time the native runtime only. Parity is still checked once against ORT, but the timed loop
    /// runs native alone so ORT's intra-op threadpool is not spinning and polluting native samples.
    #[arg(long)]
    native_only: bool,
    /// ORT `intra_op_num_threads` for the timed session. Unset matches the ORT
    /// pool to the native budget whenever ORT would otherwise build a wider pool
    /// than the CPUs the native arm gets -- chiefly `--native-threads N`, which
    /// pins the process only after the ORT session already sized itself to the
    /// whole mask, leaving ORT's threads spinning on the native arm's cores and
    /// inflating the native number only (measured 10x at width 4). Pass `0` for
    /// ORT's own default of one thread per logical CPU it is allowed to run on
    /// -- what a user gets out of the box, but not a matched A/B on a narrowed
    /// budget.
    #[arg(long)]
    ort_intra_threads: Option<i32>,
    /// ORT `inter_op_num_threads` for the timed session. `0` keeps ORT's
    /// default. Single-node benchmarks have no parallel subgraphs, so this only
    /// matters for multi-node graphs where ORT's inter-op pool is an advantage
    /// the native EP does not have.
    #[arg(long, default_value_t = 0)]
    ort_inter_threads: i32,
    /// Native CPU decode-pool width. `0` leaves `ONNX_GENAI_CPU_DECODE_THREADS`
    /// exactly as inherited. Any other value sets it before the session is
    /// built, so a thread-matched A/B is enforced by this tool rather than by
    /// the operator remembering to export the variable.
    #[arg(long, default_value_t = 0)]
    native_threads: usize,
    /// Relative tolerance used for Float32 output parity.
    #[arg(long, default_value_t = 1e-3)]
    rel_tolerance: f32,
    /// Absolute tolerance used for Float32 output parity.
    #[arg(long, default_value_t = 1e-4)]
    abs_tolerance: f32,
    /// Relative tolerance used for Float16 output parity. Defaults to 4 f16 ULP
    /// (f16 epsilon is 2^-10 ~= 9.8e-4, so the f32 default of 1e-3 is barely
    /// one ULP and would pass almost any pair of f16 values, making the check
    /// vacuous). Absolute tolerance is scaled the same way.
    #[arg(long, default_value_t = 4.0 * F16_EPSILON)]
    f16_rel_tolerance: f32,
    /// Absolute tolerance used for Float16 output parity. See
    /// `--f16-rel-tolerance`.
    #[arg(long, default_value_t = F16_EPSILON)]
    f16_abs_tolerance: f32,
}

/// Machine epsilon of IEEE binary16 (`2^-10`). f16 carries a 10-bit mantissa,
/// so two f16 values that differ by one ULP near 1.0 differ by this much.
const F16_EPSILON: f32 = 9.765_625e-4;

/// The native CPU EP's decode-pool width knob. Read once into a `OnceLock` by
/// the EP, so it must be set before the first session is built.
const NATIVE_DECODE_THREADS_ENV: &str = "ONNX_GENAI_CPU_DECODE_THREADS";

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

/// Float16 bit patterns for the same values [`synthetic_f32`] produces, so a
/// Float16 graph is fed the numerically closest version of the f32 input.
fn synthetic_f16_bits(count: usize) -> Vec<u16> {
    synthetic_f32(count)
        .into_iter()
        .map(|value| half::f16::from_f32(value).to_bits())
        .collect()
}

/// Unsigned 8-bit inputs spread over the whole quantized range (QLinearMatMul
/// and friends interpret these through a scale/zero-point, so the raw spread
/// matters more than the float value).
fn synthetic_u8(count: usize) -> Vec<u8> {
    (0..count)
        .map(|index| (index.wrapping_mul(37) % 251) as u8)
        .collect()
}

fn synthetic_i8_bytes(count: usize) -> Vec<u8> {
    (0..count)
        .map(|index| (((index.wrapping_mul(37) % 251) as i32 - 125) as i8) as u8)
        .collect()
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
                // The same synthesizers the paired path uses, so an `--ort-only`
                // arm is fed byte-identical inputs to the ORT half of a paired
                // run and the two are comparable. Without these, `--ort-only`
                // could not measure an `f16` graph at all, which made the
                // separate-arm method unavailable for exactly the half-precision
                // kernels that most need it.
                OrtDataType::Float16 => {
                    Value::from_slice_f16_bits(&synthetic_f16_bits(count), &ort_shape)?
                }
                OrtDataType::Uint8 => {
                    Value::from_raw_bytes(synthetic_u8(count), &ort_shape, OrtDataType::Uint8)?
                }
                OrtDataType::Int8 => {
                    Value::from_raw_bytes(synthetic_i8_bytes(count), &ort_shape, OrtDataType::Int8)?
                }
                dtype => bail!(
                    "input '{}' has unsupported dtype {dtype:?}; bench_generic currently \
                     synthesizes Float32, Float16, Int32, Int64, Uint8, and Int8 inputs",
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

/// Largest absolute and relative gap between two f32 sequences, plus whether
/// every element is inside `abs_tolerance + rel_tolerance * max(|a|, |b|)`.
fn compare_f32(
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

fn compare_outputs(
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

/// p50/p90/min of one runtime's samples. p90 uses the nearest-rank definition
/// (`ceil(0.9 * n)`-th smallest), so a 10-run comparison reports the 9th
/// sample rather than interpolating; dispersion is reported as p90/p50 so a
/// noisy shared host is visible in the record instead of hidden by the median.
#[derive(Clone, Copy)]
struct Stats {
    p50: f64,
    p90: f64,
    min: f64,
}

impl Stats {
    /// Nearest-rank percentiles over `samples`, which must be non-empty (the
    /// caller bails on `--runs 0`; this asserts rather than panicking on an
    /// out-of-bounds index if a future caller filters samples down to nothing).
    fn from(mut samples: Vec<f64>) -> Self {
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

    fn spread(&self) -> f64 {
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
/// What a row needs in order to carry a width label honestly.
///
/// Two independent mechanisms answer to `ONNX_GENAI_CPU_DECODE_THREADS`, and a
/// row is only mislabelled if *neither* delivered what was asked for:
///
/// - the **persistent SPMD decode pool**, entered only through
///   `with_decode_pool_scope` -- from the engine's generation loop
///   (`native_decode/cpu.rs`), speculative decode (`native_decode/proposer.rs`)
///   and the `roofline_gemv` bench, none of which a single-inference run
///   through `InferenceSession::run` reaches, so in *this* binary it never
///   resolves and its two fields read `none` / `unresolved`; and
/// - the **task runtime**, whose width falls back to the same budget
///   (`task_runtime::resolve_width`) and which parallelises kernels everywhere
///   else -- including this binary's path.
///
/// Reporting only the first is what made the earlier draft of this change claim
/// "no decode pool was resolved, this row does not measure decode width at all"
/// on a run whose width demonstrably *was* in effect: 4096x4096 int4 M=1
/// measured 3.098 / 1.415 / 0.598 ms at requested widths 1 / 2 / 16, with the
/// pool unresolved throughout. The task runtime was doing the work.
#[derive(Clone, Copy, Debug)]
struct WidthReport {
    /// The width the harness asked for, as it appears in the row. `None` when
    /// nothing was asked for, so no row is claiming a width.
    requested: Option<usize>,
    /// The persistent decode pool's view.
    spmd: DecodeWidth,
    /// The task runtime's realized width. Note this is post-`smt_cap`, so it is
    /// itself a place a request can be silently reduced.
    task_width: usize,
    /// CPUs the process is actually allowed to run on, as the denominator for
    /// the two widths above.
    ///
    /// Without it `native_width_as_requested=yes` is ambiguous in the one case
    /// that most deserves a second look: confined to 2 CPUs, a request for 8
    /// lanes *is* honoured -- the runtime really does build 8 -- but they share
    /// two cores. That is oversubscription, not a reduction, and the two must not
    /// be conflated; printing the denominator lets a reader tell them apart
    /// without this field having to guess which one they care about.
    ///
    /// Sampled at report time, so it reflects any narrowing the EP itself
    /// applied while building -- observed reading 16 at `--native-threads 16`
    /// and 32 at `--native-threads 32` on the same 32-vCPU host. That is the
    /// denominator the run actually had rather than the machine's, which is the
    /// useful one, but it means this field is a property of the run and not of
    /// the box, and two rows from the same host may legitimately differ.
    cpus: usize,
}

impl WidthReport {
    /// Whether every mechanism that ran actually delivered the requested lane
    /// count.
    ///
    /// A conjunction, not "either one matched". Under the looser rule a
    /// satisfied decode pool would launder a capped task runtime into a clean
    /// `yes` -- `requested=16, pool=16, task=8` would report honoured while half
    /// the lanes were missing from every kernel the pool does not serve. A
    /// mechanism that never ran cannot contradict the label, so it does not
    /// count against it; one that ran and fell short always does.
    ///
    /// Compared against the *harness's* request rather than
    /// `DecodeWidth::is_as_requested`, because the number on trial is the one
    /// the row is labelled with. The EP records its own copy only when a pool
    /// resolves, so relying on it would silently skip the comparison in exactly
    /// the case where it never did.
    fn is_satisfied(&self) -> bool {
        let Some(requested) = self.requested else {
            return true;
        };
        let pool_delivered = match self.spmd.realized {
            None => true,
            Some(realized) => realized == requested,
        };
        pool_delivered && self.task_width == requested
    }
}

/// The realized widths, as fields for the result row.
///
/// `--native-threads` reports only what was *asked for*: it is read back out of
/// the environment variable this binary just wrote. So a `native_threads=8` row
/// has been a *label* asserting what the harness wanted, not a measurement of
/// what it got.
///
/// Four mechanisms can hand back fewer lanes than requested and none logs at
/// default verbosity, but they are **not equally observable from here**. The
/// pre-clamp to `available_parallelism`, the NUMA split reserve and the
/// single-CPU-cpuset fallback all live on the persistent-pool path, which a
/// single-inference run never reaches; only the task runtime's `smt_cap` is
/// reachable in this binary today. The pool fields are still reported rather
/// than dropped, because they are what makes that statement checkable in the
/// row instead of a claim in a comment -- and because a harness that later
/// drives the engine needs them.
///
/// Pure in its argument so the rendering can be tested without building a pool.
fn width_fields(report: WidthReport) -> String {
    let spmd = report
        .spmd
        .realized
        .map_or_else(|| "none".to_string(), |value| value.to_string());
    format!(
        "native_pool_width={spmd} native_path={} native_task_width={} native_cpus={} \
         native_width_as_requested={}",
        report.spmd.path,
        report.task_width,
        report.cpus,
        match (report.requested, report.is_satisfied()) {
            // Distinguished from `yes` so an aggregator cannot count rows that
            // never claimed a width as rows whose claim was honoured.
            (None, _) => "n/a",
            (Some(_), true) => "yes",
            (Some(_), false) => "no",
        }
    )
}

/// The warning to print when a row cannot honestly carry the width it is
/// labelled with, or `None` when it can.
///
/// Only fires when a width was *explicitly* requested. Without that guard it
/// would fire on every model that never decodes -- `bench_generic` runs
/// arbitrary ONNX graphs, and most never touch the decode pool -- and a warning
/// that is noise on the common case gets filtered out, which leaves it exactly
/// as useful as the debug-gated reports it exists to replace.
///
/// Not debug-gated, for the same reason.
fn width_reduction_warning(report: WidthReport) -> Option<String> {
    let requested = report.requested?;
    if report.is_satisfied() {
        return None;
    }
    let spmd = report
        .spmd
        .realized
        .map_or_else(|| "none".to_string(), |value| value.to_string());
    // Name a single lane count only when there is a single answer. If the pool
    // also ran and realized something different, claiming the row "measures
    // {task_width} lanes" would contradict the `native_pool_width` printed
    // beside it, and a warning that argues with its own row teaches people to
    // ignore both.
    let measured = match report.spmd.realized {
        None => format!(
            "this row measures {} compute lanes, not {requested}",
            report.task_width
        ),
        Some(_) => format!("this row does not measure {requested} compute lanes on every path"),
    };
    Some(format!(
        "WARNING: decode width requested={requested}, but the persistent decode pool \
         realized {spmd} (path={}) and the task runtime realized {}; {measured}",
        report.spmd.path, report.task_width
    ))
}

/// Read the realized widths after the measured runs, never before.
///
/// `decode_width` peeks at already-initialized statics rather than forcing the
/// pool, so asking early would report `unresolved` and asking cannot change
/// which path the process takes. `task_runtime::width` *does* build its pool if
/// nothing has yet, which is why it is also read here rather than up front.
fn report_native_width(requested: Option<usize>) -> String {
    let report = WidthReport {
        requested,
        spmd: onnx_runtime_ep_cpu::decode_spmd::decode_width(),
        task_width: onnx_runtime_ep_cpu::task_runtime::width(),
        // Respects the affinity mask, so under `taskset` or a cpuset this is the
        // confined count rather than the machine's.
        cpus: std::thread::available_parallelism().map_or(0, std::num::NonZeroUsize::get),
    };
    if let Some(warning) = width_reduction_warning(report) {
        eprintln!("{warning}");
    }
    width_fields(report)
}

/// Renders the host-contention cell for a measured window.
///
/// Separate from the width report because they answer different questions about
/// the same row: width says *how many lanes ran*, contention says *whether the
/// cores those lanes had were actually theirs*. A row can be correctly labelled
/// t=16 and still be worthless because a co-tenant owned two of the sixteen.
///
/// Why this belongs in the shared binary rather than only in the decode bench:
/// `ONNX_GENAI_CPU_DECODE_THREADS=N` confines the process to N CPUs, and a
/// dispatch is a barrier, so one foreign thread on one of those N CPUs costs the
/// whole dispatch rather than `1/N` of it. Measured reversibly at width 2: a
/// single pinned spinner inside the set was a clean 2x wall regression at
/// unchanged CPU per token, and the same spinner pinned *outside* the set
/// changed nothing. Neither `/proc/loadavg`'s EMA nor its instantaneous runnable
/// count moves for one runnable thread out of 32, so every host-quiet gate we
/// have passes that contaminated run.
///
/// Four verdicts rather than a bare number, because "not measured" and
/// "measured, and quiet" must not render the same way -- and because an
/// incomplete own-time subtraction makes the figure a lower bound, which can
/// prove the row dirty but never prove it clean. This binary is the case that
/// motivates carrying that distinction: it runs an ORT session whose intra-op
/// threads need not share the native EP's confinement, so `BOUNDED` is an
/// expected outcome here rather than a pathology.
fn host_fields(host: &onnx_runtime_hostmon::Contention) -> String {
    let verdict = if !host.measured {
        return "host_foreign=n/a host=unmeasured".to_string();
    } else if host.is_contended() {
        "CONTENDED"
    } else if host.is_clean() {
        "CLEAN"
    } else {
        "BOUNDED"
    };
    format!(
        "host_foreign={} host_sib={} host_busy={:.1} host={verdict}",
        onnx_runtime_hostmon::foreign_column(std::slice::from_ref(host)),
        onnx_runtime_hostmon::sibling_column(std::slice::from_ref(host)),
        host.total_pct,
    )
}

/// Warns when the native arm is confined to fewer CPUs than the ORT arm's pool
/// spans.
///
/// Since #1839 this is a **backstop**, not the primary remedy:
/// [`resolve_ort_intra_threads`] now matches the ORT pool to the native budget
/// by default, so on a narrowed budget this is normally silent. It still earns
/// its place because it runs *after* the native pool has been built and keys on
/// the **realized** affinity mask, so it fires in three cases the startup
/// resolution cannot see or deliberately does not act on:
///
/// * an explicit `--ort-intra-threads` wider than the mask -- the operator asked
///   for the unmatched comparison, and the row should still say so;
/// * a native pool that ends up narrower than the width that was requested;
/// * the SMT residue. Under `taskset -c 0-3` on a host with adjacent siblings,
///   ORT's default builds 4 threads (it respects the mask) while the native pool
///   builds one worker per *physical* core and pins to 2. Matching those means
///   imposing the native pool's policy on ORT, so the resolution leaves it
///   alone -- and this reports it from the realized mask instead of it going
///   unsaid.
///
/// `--native-threads N` confines the *whole process* to N CPUs, but ORT's
/// intra-op pool is sized from the machine and spins between runs. In an
/// interleaved A/B those spinning threads land on the same N CPUs the native arm
/// is confined to, so the native arm is timed against an oversubscribed core set
/// while the ORT arm is not. The bias is one-directional: it inflates native and
/// leaves ORT alone, which is indistinguishable in the row from "native does not
/// scale".
///
/// Measured on this model (4096x4096 int4, M=1) at `--native-threads 4`, 20 runs:
///
/// | arm | native p50 |
/// |---|---|
/// | `--native-only` | 0.760 ms |
/// | A/B, ORT pool unconstrained | 7.642 ms |
/// | A/B, `--ort-intra-threads 1` | 1.111 ms |
/// | A/B, `--ort-intra-threads 4` | 1.242 ms |
///
/// A 10x inflation of the native number, removed by constraining the ORT pool --
/// so it is the pool and not the model. `--native-only` already documents this
/// on its own help text, but nothing says anything when it is *omitted*, which
/// is the case that publishes the wrong number.
///
/// Note this is invisible to `host_foreign`: ORT's threads belong to this
/// process, so their CPU is subtracted as own time. `host_busy` is where it
/// shows -- 403% of a 4-CPU set above -- which is why that field is printed
/// beside the verdict rather than folded into it.
///
/// Keyed on the realized mask rather than on `--native-threads` being present,
/// for two reasons. `taskset -c 0-3 bench_generic ...` with no flag is confined
/// just as hard and would otherwise go unwarned -- the external case is if
/// anything more likely to catch someone out, since nothing in the command line
/// mentions threads. And `--native-threads 4 --ort-intra-threads 4` has already
/// applied the fix, so warning there would be advice to do what the user just
/// did, which is how a warning gets tuned out before it reaches the run that
/// needed it.
fn ort_pool_bias_warning(
    native_only: bool,
    native_cpus: Option<usize>,
    ort_intra_threads: usize,
    ort_default_width: Option<usize>,
) -> Option<String> {
    if native_only {
        return None;
    }
    let native_cpus = native_cpus?;
    // An unset `--ort-intra-threads` means ORT sized the pool itself, from the
    // narrower of the startup affinity mask and the machine -- *not* from the
    // machine alone, which is what this originally assumed. Measured on this
    // host: `taskset -c 0-3` with ORT's default builds 4 threads, not 32, so
    // passing the machine count here made the warning overstate the pool by 8x
    // in exactly the externally-confined case it was written to catch.
    let ort_width = if ort_intra_threads == 0 {
        ort_default_width?
    } else {
        ort_intra_threads
    };
    if ort_width <= native_cpus {
        return None;
    }
    Some(format!(
        "WARNING: the native arm is confined to {native_cpus} CPUs but the interleaved ORT arm's \
         intra-op pool spans {ort_width}, and it spins between runs, so it oversubscribes exactly \
         the cores the native arm is confined to. This inflates the native number only (measured \
         10x at width 4) and reads as a native scaling loss. Use --native-only for the native \
         timing, or --ort-intra-threads {native_cpus} to compare at equal width."
    ))
}

/// Chooses the ORT intra-op width for the timed session, matching it to the
/// native budget when the native arm is confined and the user has not chosen a
/// width.
///
/// The bias this removes is measured in #1839: `--native-threads N` confines the
/// native arm to N lanes, but ORT sizes its intra-op pool from the *machine* and
/// spins between runs, so in an interleaved A/B those spinning threads land on
/// exactly the cores the native arm is confined to. On this model (4096x4096
/// int4, M=1) at width 4 that inflated the native p50 from 0.760 ms to 7.642 ms
/// -- **10x, one-directional, on the native number only** -- and constraining
/// the ORT pool removed it. In a result row that is indistinguishable from
/// "native does not scale", and it grows as the budget narrows.
///
/// #1835 warned about it. A warning was the wrong remedy: the run that needs it
/// is the one where nobody passed a thread flag at all, and a warning on stderr
/// does not stop the biased number from being printed, copied into a table and
/// believed. Interleaving exists to cancel drift, so defaulting to
/// `--native-only` would lose something real; matching the widths keeps the
/// interleave and removes the bias.
///
/// Precedence, and why each rule is where it is:
///
/// 1. **An explicit positive `--ort-intra-threads` always wins.** Changing the
///    default must not take away the ability to ask for a specific width.
/// 2. `--native-only` keeps its existing meaning (pool of 1, parked, out of the
///    way of the timed native runs). It outranks an explicit *non-positive*
///    value, because ORT's "size it yourself" sentinel would otherwise un-park
///    the pool that flag exists to park.
/// 3. An explicit non-positive value on its own is a deliberate opt-out of
///    matching: "out-of-the-box ORT versus a confined native arm" is a
///    legitimate question -- it is just not a *matched* one, and the row should
///    be the result of asking for it rather than of forgetting a flag.
/// 4. `--ort-only` opts out: there is no native arm to match, so narrowing ORT
///    there would silently alter the very baseline that mode exists to record.
/// 5. Otherwise the budget is the width the operator asked the native pool for
///    -- via `--native-threads` or an inherited `ONNX_GENAI_CPU_DECODE_THREADS`
///    -- and it is applied only when ORT would otherwise build a *wider* pool
///    than that. A run that asked for no particular width is never matched:
///    there is no budget to match it to.
///
/// # What ORT's default actually is, measured rather than assumed
///
/// The gate needs ORT's default pool width, and getting it wrong in either
/// direction is a silent failure -- too high and the tool "fixes" a run that was
/// never biased while printing a false explanation; too low and the biased run
/// goes unmatched. So it was measured on this host (16 physical cores, 32
/// logical, SMT siblings adjacent) by sampling `/proc/<pid>/task` during an
/// `--ort-only` run:
///
/// | configuration | ORT threads |
/// |---|---:|
/// | `--ort-intra-threads N` | exactly `N`, for N in 1,2,4,8,16,32 |
/// | default, unconfined | **32** |
/// | default, `taskset -c 0-3` | **4** |
/// | default, `taskset -c 0,2,4,6` | **4** |
/// | `--ort-intra-threads 32`, `taskset -c 0-3` | 32 |
///
/// Two facts follow, and only the first was already believed:
///
/// * ORT's default is one thread per **logical** CPU, not per physical core --
///   32, not 16, on a host with 16 cores and SMT.
/// * **ORT's default respects the process affinity mask.** Under an external
///   `taskset` it has already sized itself to the mask, so there is nothing to
///   match and any note claiming otherwise would be false. Keying the gate on
///   the machine's CPU count alone would have fired here and printed "instead of
///   ORT's default of 32" about a pool that was going to be 4 threads.
///
/// The bias is therefore *not* "confined process versus machine-sized ORT". It
/// is specifically **`--native-threads N` (or an inherited
/// `ONNX_GENAI_CPU_DECODE_THREADS`)**, where the native pool pins the process
/// *after* the ORT session has already been built at the startup mask's width.
/// That ordering is what leaves 32 ORT threads spinning on N cores, and it is
/// why the gate compares against `min(startup mask, online CPUs)`.
///
/// # The residue this deliberately does not match
///
/// The native pool's own default is one worker per *physical* core, so under
/// `taskset -c 0-3` on an SMT host native runs 2 workers while ORT's mask-sized
/// default runs 4 threads on the same 2 cores. That is a real remaining
/// inequality, and it is left alone on purpose: closing it means imposing the
/// native pool's one-per-core policy on ORT, which changes the comparison from
/// "ORT as configured versus native as configured" into something else. It is
/// not silent -- [`ort_pool_bias_warning`] runs after the pool has pinned itself
/// and reports exactly this case from the realized mask.
///
/// Resolution happens at startup, before either session is built, because
/// `intra_op_num_threads` is a session-construction option. That is *earlier*
/// than the point where the native pool has pinned itself, which is the other
/// reason the backstop earns its place.
fn resolve_ort_intra_threads(
    explicit: Option<i32>,
    native_only: bool,
    ort_only: bool,
    requested_native_width: Option<usize>,
    startup_cpus: Option<usize>,
    online_cpus: Option<usize>,
) -> (i32, Option<String>) {
    // A non-positive explicit value is ORT's "size it yourself" sentinel, so it
    // is an opt-out of matching rather than a width -- but it must not also opt
    // out of `--native-only`'s parked pool, which exists precisely so ORT is not
    // spinning during the native timing. Ordering these three rules the other
    // way round silently un-parks the pool for anyone who wrote both flags.
    if let Some(explicit) = explicit
        && explicit > 0
    {
        return (explicit, None);
    }
    if native_only {
        return (1, None);
    }
    if explicit.is_some() || ort_only {
        return (0, None);
    }
    // Deliberately *not* `min(requested, startup_cpus)`. That reads as
    // defensive, but the mask leg is provably dead: `ort_default` is itself
    // capped by the mask, so whenever the mask would win the min the result is
    // `budget >= ort_default` and nothing is matched anyway. A redundant `min`
    // that looks load-bearing is a small lie about what the rule is, and a
    // mutation that deletes it is uncatchable by construction.
    let budget = requested_native_width;
    // ORT sizes its default pool from the affinity mask when it has one, so the
    // width it would have chosen is the narrower of the mask and the machine --
    // not the machine alone. Measured, see above.
    let ort_default = effective_ort_default(startup_cpus, online_cpus);
    let (Some(budget), Some(ort_default)) = (budget, ort_default) else {
        return (0, None);
    };
    if budget == 0 || budget >= ort_default {
        return (0, None);
    }
    let budget_i32 = i32::try_from(budget).unwrap_or(i32::MAX);
    (
        budget_i32,
        Some(format!(
            "NOTE: the native arm is confined to {budget} CPUs, so the interleaved ORT arm's \
             intra-op pool was matched to {budget} threads instead of the {ort_default} it would \
             have built here. An unmatched ORT pool spins on exactly the cores the native arm is \
             confined to and inflates the native number only (measured 10x at width 4, #1839). \
             Pass --ort-intra-threads 0 for ORT's out-of-the-box pool, which is a real question \
             but not a matched comparison."
        )),
    )
}

/// The width ORT will give an unconfigured intra-op pool: the narrower of the
/// affinity mask it starts under and the machine it is on.
///
/// Measured, not assumed -- `taskset -c 0-3` with ORT's default builds 4
/// threads, not 32 (see [`resolve_ort_intra_threads`]). Using the machine count
/// alone overstates the pool by 8x in exactly the externally-confined case, and
/// an overstated default is what makes a tool "fix" a run that was never biased
/// while printing a false explanation for it.
///
/// `None` means "not measurable here", not "zero", so an unknown value must
/// never win the comparison and narrow something on a guess.
fn effective_ort_default(startup_cpus: Option<usize>, online_cpus: Option<usize>) -> Option<usize> {
    match (startup_cpus, online_cpus) {
        (Some(mask), Some(machine)) => Some(mask.min(machine)),
        (Some(only), None) | (None, Some(only)) => Some(only),
        (None, None) => None,
    }
}

fn build_arm() -> &'static str {
    if cfg!(feature = "mlas") {
        "mlas-reference"
    } else {
        "native"
    }
}

/// Labels *why* the row's `ort_intra_threads` has the value it does.
///
/// The width alone cannot distinguish "4 because the operator asked for 4" from
/// "4 because the tool matched a confined native budget", and those two rows
/// answer different questions. Printing the provenance beside the value is what
/// lets a row be re-read months later without re-deriving it from the command
/// line -- the same reason `native_path` is printed beside `native_pool_width`.
fn ort_width_source(explicit: bool, matched: bool) -> &'static str {
    match (explicit, matched) {
        (true, _) => "explicit",
        (false, true) => "matched-native-budget",
        (false, false) => "ort-default",
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.runs == 0 {
        bail!("--runs must be greater than zero");
    }
    validate_tolerance("rel-tolerance", args.rel_tolerance)?;
    validate_tolerance("abs-tolerance", args.abs_tolerance)?;
    if args.phase_profile {
        if args.ort_only {
            // The profiler only instruments the native executor, so this
            // combination would enable it and then print nothing. Say so rather
            // than emitting a silently empty report.
            bail!(
                "--phase-profile has no effect with --ort-only: the phase profiler instruments the native executor, not the ORT session"
            );
        }
        // Turn the executor's phase accounting on programmatically rather than
        // through `NXRT_EXEC_PHASE_PROFILE`, so the flag works on its own and
        // cannot be half-set by an inherited environment.
        onnx_runtime_session::enable_exec_phase_profile_for_process();
    }
    let input_shape = args
        .input_shape
        .as_deref()
        .map(parse_shape)
        .transpose()
        .map_err(anyhow::Error::msg)?;

    // Set the native decode-pool width *before* anything builds a session: the
    // EP reads `ONNX_GENAI_CPU_DECODE_THREADS` once into a `OnceLock`, so a
    // later write would be ignored and silently produce an unmatched A/B.
    if args.native_threads > 0 {
        // SAFETY: single-threaded startup, before any session, thread pool or
        // other reader of the process environment exists.
        unsafe { std::env::set_var(NATIVE_DECODE_THREADS_ENV, args.native_threads.to_string()) };
    }
    let native_threads = std::env::var(NATIVE_DECODE_THREADS_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "default".to_string());
    // Covers `--native-threads` and a pre-set environment variable alike: either
    // way the row is about to be labelled with a width somebody chose. `None`
    // means nothing was asked for -- including an explicit `=0`, which opts out
    // of the bounded pool rather than requesting zero lanes -- so no row is
    // claiming a width.
    let requested_width = native_threads
        .parse::<usize>()
        .ok()
        .filter(|width| *width > 0);

    let environment = Environment::new("bench-generic")?;
    // Read once, before the native pool can re-pin the process: this is the mask
    // ORT's own default pool is about to be sized from. The realized mask read
    // later, after pinning, is a different number and answers a different
    // question -- see `ort_pool_bias_warning`.
    let startup_cpus = onnx_runtime_hostmon::AllowedCpus::current().map(|allowed| allowed.len());
    let (ort_intra_threads, ort_width_note) = resolve_ort_intra_threads(
        args.ort_intra_threads,
        args.native_only,
        args.ort_only,
        requested_width,
        startup_cpus,
        onnx_runtime_hostmon::online_cpus(),
    );
    let mut ort_options = SessionOptions::with_execution_provider(ep_selection("cpu"))
        .with_intra_op_threads(ort_intra_threads);
    ort_options.inter_op_num_threads = args.ort_inter_threads;
    let ort_session = Session::new(&environment, &args.model, ort_options)
        .with_context(|| format!("load ORT CPU session from {}", args.model.display()))?;
    // After the session exists, so the past tense is true. A note claiming a
    // pool "was matched" for a session that then failed to load would be a
    // small lie in the one output a failed run leaves behind.
    if let Some(note) = &ort_width_note {
        eprintln!("{note}");
    }
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
        args.f16_abs_tolerance,
        args.f16_rel_tolerance,
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
    if args.phase_profile {
        // Warmups pay first-touch page faults and lazy plan construction that
        // no measured run repeats. Counting them would attribute one-time cost
        // to the steady state.
        onnx_runtime_session::reset_exec_phase_profile();
    }
    let mut native_samples = Vec::with_capacity(args.runs);
    let mut ort_samples = Vec::with_capacity(args.runs);
    // Spans the measured runs only. Warmups are excluded deliberately: they are
    // not part of any published number, and including them would let first-touch
    // faults and plan construction dilute the foreign fraction of the window
    // that is.
    if let Some(warning) = ort_pool_bias_warning(
        args.native_only,
        onnx_runtime_hostmon::AllowedCpus::current().map(|a| a.len()),
        ort_intra_threads.max(0) as usize,
        effective_ort_default(startup_cpus, onnx_runtime_hostmon::online_cpus()),
    ) {
        eprintln!("{warning}");
    }
    let host_before = onnx_runtime_hostmon::snapshot();
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
    let host_after = onnx_runtime_hostmon::snapshot();

    let host = onnx_runtime_hostmon::contention(host_before.as_ref(), host_after.as_ref());

    let native = Stats::from(native_samples);
    if args.native_only {
        println!(
            "result: native={:.3} ms ({:.2} infer/s) native_p90={:.3} ms native_min={:.3} ms \
             native_spread={:.2} native_threads={native_threads} {} {} ort=skipped \
             native-only=true arm={} parity={}",
            native.p50,
            1_000.0 / native.p50,
            native.p90,
            native.min,
            native.spread(),
            report_native_width(requested_width),
            host_fields(&host),
            build_arm(),
            if parity_pass { "PASS" } else { "FAIL" }
        );
        if args.phase_profile {
            onnx_runtime_session::print_exec_phase_profile();
        }
        return Ok(());
    }
    let ort = Stats::from(ort_samples);
    println!(
        "result: native={:.3} ms ({:.2} infer/s) ort={:.3} ms ({:.2} infer/s) \
         native/ort={:.3} native_p90={:.3} ort_p90={:.3} native_min={:.3} ort_min={:.3} \
         native_spread={:.2} ort_spread={:.2} native_threads={native_threads} {} {} \
         ort_intra_threads={ort_intra_threads} ort_width_src={} arm={} parity={}",
        native.p50,
        1_000.0 / native.p50,
        ort.p50,
        1_000.0 / ort.p50,
        native.p50 / ort.p50,
        native.p90,
        ort.p90,
        native.min,
        ort.min,
        native.spread(),
        ort.spread(),
        report_native_width(requested_width),
        host_fields(&host),
        ort_width_source(args.ort_intra_threads.is_some(), ort_width_note.is_some()),
        build_arm(),
        if parity_pass { "PASS" } else { "FAIL" }
    );
    if args.phase_profile {
        onnx_runtime_session::print_exec_phase_profile();
    }
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
        let diffs = compare_outputs(
            &[native],
            &[ort],
            1e-4,
            1e-3,
            F16_EPSILON,
            4.0 * F16_EPSILON,
        )
        .unwrap();
        assert!(!diffs[0].pass);
        assert_eq!(diffs[0].max_abs, f32::INFINITY);
    }

    /// The f32 defaults are ~1 f16 ULP, so reusing them for Float16 outputs
    /// makes the parity check vacuous: two f16 values a single representable
    /// step apart would "match". The f16 defaults must be loose enough to
    /// tolerate f16 rounding but tight enough to still reject a real
    /// disagreement.
    #[test]
    fn f16_parity_tolerances_reject_real_disagreement_and_accept_rounding() {
        assert!(
            4.0 * F16_EPSILON > F16_EPSILON,
            "the f16 relative tolerance must exceed one f16 ULP"
        );
        // One f16 ULP apart near 1.0: rounding noise, must pass.
        let near = half::f16::from_f32(1.0);
        let next = half::f16::from_bits(near.to_bits() + 1);
        let (_, _, pass) = compare_f32(
            &[near.to_f32()],
            &[next.to_f32()],
            F16_EPSILON,
            4.0 * F16_EPSILON,
        );
        assert!(pass, "one f16 ULP of rounding must not fail parity");
        // 1% apart: ~10 f16 ULP, a real disagreement, must fail.
        let (_, _, pass) = compare_f32(&[1.0], &[1.01], F16_EPSILON, 4.0 * F16_EPSILON);
        assert!(
            !pass,
            "a 1% disagreement is ~10 f16 ULP and must fail parity"
        );
        // ...and would have passed under a tolerance an order of magnitude
        // looser, which is what makes the 4-ULP choice meaningful.
        let (_, _, pass) = compare_f32(&[1.0], &[1.01], F16_EPSILON, 5e-2);
        assert!(pass, "control: 5e-2 is too loose to catch a 1% gap");
    }

    #[test]
    fn parity_tolerances_must_be_finite_and_non_negative() {
        assert!(validate_tolerance("rel-tolerance", 1e-3).is_ok());
        assert!(validate_tolerance("rel-tolerance", f32::INFINITY).is_err());
        assert!(validate_tolerance("abs-tolerance", f32::NAN).is_err());
        assert!(validate_tolerance("abs-tolerance", -1.0).is_err());
    }
}

#[cfg(test)]
mod width_report_tests {
    use super::{DecodeWidth, WidthReport, width_fields, width_reduction_warning};

    fn report(
        requested: Option<usize>,
        pool: Option<usize>,
        path: &'static str,
        task_width: usize,
    ) -> WidthReport {
        WidthReport {
            requested,
            spmd: DecodeWidth {
                requested,
                realized: pool,
                path,
            },
            task_width,
            cpus: 32,
        }
    }

    /// Every shape below with `pool = None` is what this binary actually
    /// produces: `InferenceSession::run` never enters `with_decode_pool_scope`,
    /// so the pool never resolves here and `native_pool_width` always reads
    /// `none`. The `Some(..)` shapes are reachable only if this report is reused
    /// from a harness that drives the engine, and are marked where they appear.
    #[test]
    fn a_row_reports_the_width_that_was_built_not_the_one_asked_for() {
        let fields = width_fields(report(Some(8), None, "unresolved", 2));
        assert!(
            fields.contains("native_task_width=2"),
            "must carry the realized width: {fields}"
        );
        assert!(
            fields.contains("native_width_as_requested=no"),
            "must mark the row as not matching its label: {fields}"
        );
    }

    #[test]
    fn a_satisfied_request_is_marked_as_requested_and_warns_about_nothing() {
        let satisfied = report(Some(8), None, "unresolved", 8);
        assert!(width_fields(satisfied).contains("native_width_as_requested=yes"));
        assert!(width_reduction_warning(satisfied).is_none());
    }

    #[test]
    fn a_silently_reduced_width_warns_and_names_every_number() {
        let warning = width_reduction_warning(report(Some(16), None, "unresolved", 15))
            .expect("a reduced width must warn");
        for expected in ["requested=16", "realized 15", "15 compute lanes, not 16"] {
            assert!(
                warning.contains(expected),
                "must name {expected} so the row can be corrected: {warning}"
            );
        }
    }

    /// Not reachable from this binary today, and guarded anyway: under an
    /// "either mechanism matched" rule a satisfied pool would launder a capped
    /// task runtime into a clean `yes`, hiding eight missing lanes on every
    /// kernel the pool does not serve.
    #[test]
    fn a_satisfied_pool_does_not_launder_a_capped_task_runtime() {
        let laundered = report(Some(16), Some(16), "spmd", 8);
        assert!(
            width_fields(laundered).contains("native_width_as_requested=no"),
            "one mechanism delivering is not the row's claim being honoured"
        );
        assert!(width_reduction_warning(laundered).is_some());
    }

    /// The warning must never argue with the fields printed beside it: when both
    /// mechanisms ran and realized different counts there is no single lane
    /// count to name, and naming one would contradict the other field.
    #[test]
    fn a_warning_never_claims_a_lane_count_its_own_row_contradicts() {
        let disagreeing = report(Some(16), Some(15), "spmd", 8);
        let fields = width_fields(disagreeing);
        let warning = width_reduction_warning(disagreeing).expect("nothing delivered 16");
        assert!(
            fields.contains("native_pool_width=15") && fields.contains("native_task_width=8"),
            "{fields}"
        );
        assert!(
            !warning.contains("measures 8 compute lanes")
                && !warning.contains("measures 15 compute lanes"),
            "must not pick one of two disagreeing counts: {warning}"
        );
        assert!(
            warning.contains("does not measure 16 compute lanes"),
            "must still say the label is wrong: {warning}"
        );
    }

    /// The bug this whole report exists to avoid, and one an earlier draft of it
    /// committed: the persistent pool is only entered from the engine's
    /// generation loop, so a single-inference run never resolves it -- while the
    /// task runtime, seeded from the same budget, really does deliver the width.
    /// Measured: 3.098 / 1.415 / 0.598 ms at widths 1 / 2 / 16 with the pool
    /// unresolved throughout. Warning there would call a correct row a lie.
    #[test]
    fn a_width_delivered_by_the_task_runtime_alone_is_not_reported_as_missing() {
        let via_task_runtime = report(Some(16), None, "unresolved", 16);
        assert!(
            width_reduction_warning(via_task_runtime).is_none(),
            "the task runtime delivered the requested width; the row is honest"
        );
        assert!(
            width_fields(via_task_runtime).contains("native_width_as_requested=yes"),
            "must credit the mechanism that actually did the work"
        );
        assert!(
            width_fields(via_task_runtime).contains("native_pool_width=none"),
            "and must still show the pool never ran, which is what makes that checkable"
        );
    }

    /// The complement: no pool *and* a capped task runtime is a genuinely
    /// mislabelled row, and must still be caught.
    #[test]
    fn a_width_no_mechanism_delivered_is_reported_as_missing() {
        let capped = report(Some(16), None, "unresolved", 8);
        let warning =
            width_reduction_warning(capped).expect("nothing delivered 16; this row must warn");
        assert!(warning.contains("requested=16"), "{warning}");
        assert!(
            warning.contains("8 compute lanes, not 16"),
            "must say what the row actually measures: {warning}"
        );
        assert!(
            warning.contains("realized none"),
            "must distinguish an absent pool from a narrow one: {warning}"
        );
    }

    /// Honoured but oversubscribed is not the same as reduced, and the row has
    /// to let a reader tell them apart: confined to 2 CPUs, a request for 8
    /// lanes really does build 8 -- verified by running it -- so warning would be
    /// wrong, but printing no denominator would make `as_requested=yes` read as
    /// "all is well".
    #[test]
    fn an_oversubscribed_but_honoured_width_is_reported_with_its_cpu_denominator() {
        let oversubscribed = WidthReport {
            cpus: 2,
            ..report(Some(8), None, "unresolved", 8)
        };
        assert!(
            width_reduction_warning(oversubscribed).is_none(),
            "8 lanes were built; the request was honoured"
        );
        let fields = width_fields(oversubscribed);
        assert!(
            fields.contains("native_task_width=8") && fields.contains("native_cpus=2"),
            "must show both the lanes and the cores they share: {fields}"
        );
    }

    /// Most models `bench_generic` runs never decode. Warning on those would be
    /// noise on the common case, and a noisy warning is filtered out -- leaving
    /// it exactly as useful as the debug-gated reports it replaces.
    #[test]
    fn a_run_that_requested_no_particular_width_never_warns() {
        assert!(width_reduction_warning(report(None, None, "unresolved", 32)).is_none());
        assert!(width_reduction_warning(report(None, Some(32), "flat", 32)).is_none());
        // `n/a`, not `yes`: an aggregator must not count a row that claimed no
        // width as a row whose claim was honoured.
        assert!(
            width_fields(report(None, None, "unresolved", 32))
                .contains("native_width_as_requested=n/a")
        );
    }
}

/// Tests for the host-contention cell.
///
/// The rendering is pure so that the four verdicts can be asserted directly. The
/// measurement itself is tested in `onnx-runtime-hostmon`; what is at stake here
/// is that a row never *reads* clean when it was not measured clean, which is a
/// property of this formatting and not of the measurement.
#[cfg(test)]
mod host_fields_tests {
    use super::*;
    use onnx_runtime_hostmon::Contention;

    /// A reading on the `foreign_pct` axis, with the sibling axis pinned quiet
    /// and known so that each test moves one variable.
    fn reading(foreign_pct: f64, own_time_complete: bool) -> Contention {
        Contention {
            foreign_pct,
            total_pct: 100.0,
            measured: true,
            own_time_complete,
            sibling_peak_pct: 0.0,
            siblings_known: true,
        }
    }

    #[test]
    fn an_unmeasured_window_says_so_instead_of_printing_a_zero() {
        let cell = host_fields(&Contention::default());
        assert!(cell.contains("host=unmeasured"), "{cell}");
        assert!(
            !cell.contains("0.0"),
            "an unmeasured window must not emit a number a reader could take as quiet: {cell}"
        );
    }

    #[test]
    fn a_quiet_window_with_a_complete_subtraction_is_the_only_clean_verdict() {
        let cell = host_fields(&reading(0.4, true));
        assert!(cell.contains("host=CLEAN"), "{cell}");
        assert!(cell.contains("host_foreign=0.4"), "{cell}");
        assert!(
            !cell.contains("0.4!"),
            "a complete subtraction is an estimate, not a bound: {cell}"
        );
    }

    #[test]
    fn a_quiet_looking_lower_bound_is_never_reported_as_clean() {
        let cell = host_fields(&reading(0.4, false));
        assert!(
            cell.contains("host=BOUNDED"),
            "an incomplete own-time subtraction under-reports, so a low figure \
             proves nothing and must not read as CLEAN: {cell}"
        );
        assert!(cell.contains("host_foreign=0.4!"), "{cell}");
    }

    /// The asymmetry that makes the bound worth printing at all: it cannot
    /// certify quiet, but it can still condemn a row.
    #[test]
    fn a_lower_bound_above_the_threshold_still_condemns_the_row() {
        let cell = host_fields(&reading(60.0, false));
        assert!(cell.contains("host=CONTENDED"), "{cell}");
        assert!(cell.contains("host_foreign=60.0!"), "{cell}");
    }

    #[test]
    fn a_contended_window_is_flagged_regardless_of_completeness() {
        for complete in [true, false] {
            let cell = host_fields(&reading(60.0, complete));
            assert!(
                cell.contains("host=CONTENDED"),
                "complete={complete}: {cell}"
            );
        }
    }

    /// The case `host_foreign` alone renders as a clean row.
    ///
    /// A co-runner on the SMT sibling of a core we own outright consumes none of
    /// our confined set, so `foreign_pct` is genuinely zero while a decode
    /// worker runs at half speed -- and a dispatch is a barrier, so the whole
    /// dispatch pays it. Without the sibling term this row reads `CLEAN`.
    #[test]
    fn a_busy_sibling_condemns_a_row_whose_own_cores_are_quiet() {
        let cell = host_fields(&Contention {
            sibling_peak_pct: 97.0,
            ..reading(0.0, true)
        });
        assert!(
            cell.contains("host=CONTENDED"),
            "a saturated sibling is contention even at foreign_pct 0: {cell}"
        );
        assert!(cell.contains("host_foreign=0.0"), "{cell}");
        assert!(cell.contains("host_sib=97.0"), "{cell}");
    }

    /// Unreadable topology is `BOUNDED`, and says which term is missing.
    ///
    /// Restricted `/sys` in a container leaves the sibling set unknown while
    /// `/proc/stat` still reads, so the foreign term is measured and the sibling
    /// term is not. That cannot certify quiet -- but the reader has to be able to
    /// see *why* a row with a low foreign figure refuses to be CLEAN, or the
    /// verdict looks like a bug.
    #[test]
    fn unknown_topology_shows_the_missing_term_rather_than_an_unexplained_verdict() {
        let cell = host_fields(&Contention {
            siblings_known: false,
            ..reading(0.4, true)
        });
        assert!(cell.contains("host=BOUNDED"), "{cell}");
        assert!(
            cell.contains("host_sib=n/a"),
            "the unmeasured term has to be visible next to the verdict it caused: {cell}"
        );
        assert!(
            !cell.contains("host_sib=0.0"),
            "an unread sibling set must never print as a quiet zero: {cell}"
        );
    }
}

/// Tests for the ORT-pool bias warning.
#[cfg(test)]
mod ort_pool_bias_tests {
    use super::*;

    /// The mechanism is "ORT's pool spans more CPUs than the native arm may
    /// use", so the trigger is the realized mask -- which is what catches an
    /// external `taskset` that never passed `--native-threads` at all.
    #[test]
    fn a_confined_native_arm_against_a_machine_sized_ort_pool_is_warned_about() {
        let warning = ort_pool_bias_warning(false, Some(4), 0, Some(32))
            .expect("a 4-CPU native arm against a 32-CPU ORT pool is the biased case");
        assert!(warning.contains("--native-only"), "{warning}");
        assert!(
            warning.contains("--ort-intra-threads 4"),
            "the suggested equal-width setting must match the realized mask: {warning}"
        );
        assert!(
            warning.contains("spans 32"),
            "the warning must name the width it is comparing against: {warning}"
        );
    }

    /// The configurations that are *not* biased must stay silent, or the warning
    /// becomes noise and gets tuned out before the run that needed it.
    #[test]
    fn the_unbiased_configurations_are_silent() {
        assert!(
            ort_pool_bias_warning(true, Some(4), 0, Some(32)).is_none(),
            "--native-only does not interleave, which is the whole point of the flag"
        );
        assert!(
            ort_pool_bias_warning(false, Some(4), 4, Some(32)).is_none(),
            "an equal-width ORT pool is the fix the warning recommends; advising \
             someone to do what they already did is how a warning gets ignored"
        );
        assert!(
            ort_pool_bias_warning(false, Some(4), 2, Some(32)).is_none(),
            "a narrower ORT pool cannot oversubscribe the native arm's cores"
        );
        assert!(
            ort_pool_bias_warning(false, Some(32), 0, Some(32)).is_none(),
            "an unconfined native arm competes with ORT symmetrically"
        );
    }

    /// Unknown widths must not synthesise a warning naming a width nobody
    /// requested.
    #[test]
    fn an_unreadable_mask_or_cpu_count_is_silent_rather_than_guessed() {
        assert!(ort_pool_bias_warning(false, None, 0, Some(32)).is_none());
        assert!(ort_pool_bias_warning(false, Some(4), 0, None).is_none());
        // ...but an explicit ORT width needs no host count to be comparable.
        assert!(ort_pool_bias_warning(false, Some(4), 16, None).is_some());
    }

    /// The default that #1839 asks for. `--native-threads 4` on an unconfined
    /// host is the shape that actually loses: the ORT session is built at the
    /// full mask *before* the native pool pins the process, so ORT ends up with
    /// 32 threads spinning on the 4 cores the native arm was given.
    #[test]
    fn a_narrowed_native_budget_matches_the_ort_pool_by_default() {
        let (width, note) =
            resolve_ort_intra_threads(None, false, false, Some(4), Some(32), Some(32));
        assert_eq!(
            width, 4,
            "the ORT pool must be matched to the native budget"
        );
        let note = note.expect("silently changing what the row measures is the failure mode");
        assert!(
            note.contains("--ort-intra-threads 0"),
            "the note must name the escape hatch back to ORT's default: {note}"
        );
        assert!(
            note.contains("#1839"),
            "the note must be traceable to the measurement that justifies it: {note}"
        );
        assert!(
            note.contains("32 it would"),
            "the note must name the width ORT would actually have built: {note}"
        );
    }

    /// Measured, not assumed: ORT's default pool respects the affinity mask
    /// (`taskset -c 0-3` builds 4 ORT threads, not 32). So external confinement
    /// alone is *already* matched, and firing here would narrow nothing while
    /// printing "instead of the 32 it would have built" about a 4-thread pool.
    #[test]
    fn external_confinement_alone_is_already_matched_and_says_nothing() {
        assert_eq!(
            resolve_ort_intra_threads(None, false, false, None, Some(4), Some(32)),
            (0, None),
            "ORT sizes its default pool from the mask, so there is nothing to match"
        );
    }

    /// The two confinements compose: 4 lanes inside an 8-CPU mask is a 4-wide
    /// native arm against a pool ORT would have built 8 threads for.
    #[test]
    fn a_narrowed_budget_inside_an_external_mask_is_matched_to_the_narrower() {
        let (width, note) =
            resolve_ort_intra_threads(None, false, false, Some(4), Some(8), Some(32));
        assert_eq!(width, 4);
        let note = note.expect("this case is a real narrowing and must be reported");
        assert!(
            note.contains("8 it would"),
            "the note must name 8, the width ORT would build inside this mask, not 32: {note}"
        );
        // ...and a request wider than the mask cannot be honoured by either arm.
        assert_eq!(
            resolve_ort_intra_threads(None, false, false, Some(16), Some(8), Some(32)),
            (0, None)
        );
    }

    /// The whole point of keeping the flag: asking for out-of-the-box ORT is a
    /// real question, and an explicit answer must survive the new default.
    #[test]
    fn an_explicit_width_is_honoured_including_an_explicit_ort_default() {
        assert_eq!(
            resolve_ort_intra_threads(Some(0), false, false, Some(4), Some(32), Some(32)),
            (0, None),
            "an explicit 0 must still mean ORT's own pool, not a matched one"
        );
        assert_eq!(
            resolve_ort_intra_threads(Some(16), false, false, Some(4), Some(32), Some(32)).0,
            16
        );
        assert_eq!(
            resolve_ort_intra_threads(Some(8), true, false, Some(4), Some(4), Some(32)).0,
            8,
            "--native-only must not override a width the operator asked for"
        );
        assert_eq!(
            resolve_ort_intra_threads(Some(0), true, false, Some(4), Some(4), Some(32)),
            (1, None),
            "an explicit ORT default must not un-park the pool --native-only parked"
        );
    }

    /// `--ort-only` has no native arm to match. Narrowing ORT there would alter
    /// the baseline that mode exists to record, and the note -- every clause of
    /// which is about "the native arm" and "the interleaved ORT arm" -- would be
    /// false in all of them.
    #[test]
    fn an_ort_only_run_is_never_matched_and_never_explained() {
        assert_eq!(
            resolve_ort_intra_threads(None, false, true, Some(4), Some(32), Some(32)),
            (0, None)
        );
        assert_eq!(
            resolve_ort_intra_threads(Some(8), false, true, Some(4), Some(32), Some(32)).0,
            8,
            "an explicit width still applies to the baseline being recorded"
        );
    }

    /// A default change is only safe if it is inert everywhere it is not needed.
    #[test]
    fn nothing_changes_on_an_unconfined_host_or_under_native_only() {
        assert_eq!(
            resolve_ort_intra_threads(None, false, false, Some(32), Some(32), Some(32)),
            (0, None),
            "matching a full-width budget would be ORT's default under another name"
        );
        assert_eq!(
            resolve_ort_intra_threads(None, false, false, None, None, Some(32)),
            (0, None)
        );
        assert_eq!(
            resolve_ort_intra_threads(None, true, false, Some(4), Some(4), Some(32)),
            (1, None),
            "--native-only keeps its parked pool of one"
        );
    }

    /// An unreadable host count is not evidence of confinement. Guessing here
    /// would silently narrow ORT on a host we simply failed to measure.
    #[test]
    fn an_unknown_host_width_leaves_the_ort_pool_alone() {
        assert_eq!(
            resolve_ort_intra_threads(None, false, false, None, None, None),
            (0, None)
        );
        assert_eq!(
            resolve_ort_intra_threads(None, false, false, Some(0), Some(0), Some(32)),
            (0, None),
            "a zero budget is an opt-out, not a request for a zero-wide pool"
        );
        // A platform with no readable mask still has a machine count to compare
        // an explicitly narrowed budget against.
        assert_eq!(
            resolve_ort_intra_threads(None, false, false, Some(4), None, Some(32)).0,
            4
        );
    }

    /// ORT's default is capped by the affinity mask, and `None` means
    /// unmeasurable rather than zero -- an unknown value must never win a
    /// comparison whose loser gets narrowed.
    #[test]
    fn the_ort_default_is_capped_by_the_mask_and_never_guessed() {
        assert_eq!(effective_ort_default(Some(4), Some(32)), Some(4));
        assert_eq!(effective_ort_default(Some(64), Some(32)), Some(32));
        assert_eq!(effective_ort_default(Some(4), None), Some(4));
        assert_eq!(effective_ort_default(None, Some(32)), Some(32));
        assert_eq!(effective_ort_default(None, None), None);
    }

    /// The backstop takes ORT's *effective* default, not the machine count. It
    /// used to be handed `online_cpus()`, which on an externally confined run
    /// claimed a 32-thread pool for a pool ORT sized to 4 -- a warning that
    /// names a number the run never had is worse than no warning, because the
    /// next real one gets discounted too.
    #[test]
    fn the_backstop_is_silent_when_ort_already_sized_itself_to_the_mask() {
        assert!(
            ort_pool_bias_warning(false, Some(4), 0, effective_ort_default(Some(4), Some(32)))
                .is_none(),
            "ORT builds 4 threads under a 4-CPU mask; there is nothing to warn about"
        );
        assert!(
            ort_pool_bias_warning(false, Some(4), 0, Some(32)).is_some(),
            "the machine count is what the old call site passed, and it warns falsely"
        );
        // The SMT residue the resolution deliberately leaves alone: a 4-CPU mask
        // whose pool pinned to one worker per physical core still faces 4 ORT
        // threads on those 2 cores, and this is where that gets said.
        let warning =
            ort_pool_bias_warning(false, Some(2), 0, effective_ort_default(Some(4), Some(32)))
                .expect("the physical-core residue must not go unreported");
        assert!(warning.contains("spans 4"), "{warning}");
    }

    /// The row has to say which question it answered; the width alone cannot.
    #[test]
    fn the_row_reports_where_the_ort_width_came_from() {
        assert_eq!(ort_width_source(true, false), "explicit");
        assert_eq!(ort_width_source(false, true), "matched-native-budget");
        assert_eq!(ort_width_source(false, false), "ort-default");
    }
}

//! Generic native nxrt versus ONNX Runtime CPU single-inference benchmark.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use clap::Parser;
use onnx_genai_bench::model_io::{
    F16_EPSILON, Stats, build_arm, build_inputs, build_ort_inputs, classifier_top1_native,
    classifier_top1_ort, compare_outputs, median_ms, parse_shape, validate_tolerance,
};
use onnx_genai_ort::{Environment, Session, SessionOptions, ep_selection};
use onnx_runtime_session::InferenceSession;

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
    /// ORT `intra_op_num_threads` for the timed session. `0` keeps ORT's default
    /// (one thread per logical core), which is what a user gets out of the box;
    /// set it to the native decode-pool width for a thread-matched comparison.
    #[arg(long, default_value_t = 0)]
    ort_intra_threads: i32,
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

/// The native CPU EP's decode-pool width knob. Read once into a `OnceLock` by
/// the EP, so it must be set before the first session is built.
const NATIVE_DECODE_THREADS_ENV: &str = "ONNX_GENAI_CPU_DECODE_THREADS";

fn run_ort_only(
    session: &Session,
    override_shape: Option<&[usize]>,
    warmups: usize,
    runs: usize,
) -> Result<()> {
    let inputs = build_ort_inputs(session, override_shape)?;
    for (name, value) in &inputs {
        println!("input: {name} shape={:?}", value.shape());
    }
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

    let environment = Environment::new("bench-generic")?;
    let ort_intra_threads = if args.ort_intra_threads > 0 {
        args.ort_intra_threads
    } else if args.native_only {
        1
    } else {
        0
    };
    let mut ort_options = SessionOptions::with_execution_provider(ep_selection("cpu"))
        .with_intra_op_threads(ort_intra_threads);
    ort_options.inter_op_num_threads = args.ort_inter_threads;
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

    let native = Stats::from(native_samples);
    if args.native_only {
        println!(
            "result: native={:.3} ms ({:.2} infer/s) native_p90={:.3} ms native_min={:.3} ms \
             native_spread={:.2} native_threads={native_threads} ort=skipped native-only=true \
             arm={} parity={}",
            native.p50,
            1_000.0 / native.p50,
            native.p90,
            native.min,
            native.spread(),
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
         native_spread={:.2} ort_spread={:.2} native_threads={native_threads} \
         ort_intra_threads={ort_intra_threads} arm={} parity={}",
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

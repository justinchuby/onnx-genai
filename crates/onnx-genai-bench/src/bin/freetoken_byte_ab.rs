use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::{Parser, ValueEnum};
use onnx_genai_bench::freetoken_byte_ab::{
    ByteClass, Phase, SyntheticFixture, read_workload, require_passing, run_estimate_comparison,
    synthetic_workload, write_estimate_report,
};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Fixture {
    DeepseekLike,
    Glm52Like,
}

impl From<Fixture> for SyntheticFixture {
    fn from(value: Fixture) -> Self {
        match value {
            Fixture::DeepseekLike => Self::DeepseekLike,
            Fixture::Glm52Like => Self::Glm52Like,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    about = "Run a deterministic synthetic FreeToken byte-estimate A/B",
    long_about = "Runs equivalent synthetic MoE routes and typed state progression through \
                  baseline-absent, optimized, and failure-control arms. Every byte value is a \
                  declared synthetic estimate under estimated_* output fields; no production \
                  loader, CUDA, VMM, checkpoint, or wall-clock claim is made."
)]
struct Args {
    /// Built-in structural fixture. Ignored when --workload is supplied.
    #[arg(long, value_enum, default_value_t = Fixture::DeepseekLike)]
    fixture: Fixture,
    /// Versioned workload JSON. This permits future model-independent typed
    /// dimensions and multimodal state groups without changing behavior gates.
    #[arg(long)]
    workload: Option<PathBuf>,
    /// Stable machine-readable comparison report.
    #[arg(long, default_value = "target/freetoken-byte-ab/report-v3.json")]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let workload = match args.workload.as_deref() {
        Some(path) => read_workload(path)?,
        None => synthetic_workload(args.fixture.into()),
    };
    if workload.routes.is_empty() {
        bail!("FreeToken workload is empty; refusing a vacuous PASS");
    }
    let report = run_estimate_comparison(workload)?;
    write_estimate_report(&args.output, &report)?;
    require_passing(&report)?;

    let baseline_steady = report
        .baseline
        .estimated_phases
        .iter()
        .find(|phase| phase.phase == Phase::DecodeSteady)
        .expect("passing report contains baseline decode steady state");
    let optimized_steady = report
        .optimized
        .estimated_phases
        .iter()
        .find(|phase| phase.phase == Phase::DecodeSteady)
        .expect("passing report contains optimized decode steady state");
    println!(
        "freetoken_byte_ab_estimate: schema={} report={} workload={} tokens={} \
         estimated_decode_h2d_baseline={} estimated_decode_h2d_optimized={} \
         estimated_decode_source_read_baseline={} estimated_decode_source_read_optimized={} \
         observed_production_events=not_observed contract=PASS",
        report.schema,
        args.output.display(),
        report.workload.label,
        report
            .baseline
            .estimated_totals
            .counter(onnx_genai_bench::freetoken_byte_ab::CounterClass::Tokens),
        baseline_steady.accounting.bytes.value(ByteClass::H2d),
        optimized_steady.accounting.bytes.value(ByteClass::H2d),
        baseline_steady
            .accounting
            .bytes
            .value(ByteClass::SourceRead),
        optimized_steady
            .accounting
            .bytes
            .value(ByteClass::SourceRead),
    );
    Ok(())
}

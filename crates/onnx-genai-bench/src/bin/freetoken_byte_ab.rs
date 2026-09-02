use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::{Parser, ValueEnum};
use onnx_genai_bench::freetoken_byte_ab::{
    ByteClass, Phase, SyntheticFixture, read_workload, require_passing, run_comparison,
    synthetic_workload, write_report,
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
    about = "Run a deterministic synthetic FreeToken baseline/optimized byte A/B",
    long_about = "Runs equivalent synthetic MoE routes and typed state progression through \
                  baseline-absent, optimized, and failure-control arms. Byte counters are exact \
                  for the declared synthetic source extents. No checkpoint or wall-clock claim \
                  is made."
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
    #[arg(long, default_value = "target/freetoken-byte-ab/report-v2.json")]
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
    let report = run_comparison(workload)?;
    write_report(&args.output, &report)?;
    require_passing(&report)?;

    let baseline_steady = report
        .baseline
        .phases
        .iter()
        .find(|phase| phase.phase == Phase::DecodeSteady)
        .expect("passing report contains baseline decode steady state");
    let optimized_steady = report
        .optimized
        .phases
        .iter()
        .find(|phase| phase.phase == Phase::DecodeSteady)
        .expect("passing report contains optimized decode steady state");
    println!(
        "freetoken_byte_ab: schema={} report={} workload={} tokens={} \
         decode_h2d_baseline={} decode_h2d_optimized={} \
         decode_source_read_baseline={} decode_source_read_optimized={} contract=PASS",
        report.schema,
        args.output.display(),
        report.workload.label,
        report
            .baseline
            .totals
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

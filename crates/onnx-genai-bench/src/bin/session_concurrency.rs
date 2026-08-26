use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use onnx_genai_server::benchmark::{BenchmarkMode, BenchmarkOptions, BenchmarkReport, run};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Mode {
    Synthetic,
    Ort,
}

#[derive(Debug, Parser)]
#[command(about = "Measure session safety and ORT worker-sharding performance")]
struct Args {
    /// Fixed-work fixture or real ONNX Runtime execution.
    #[arg(long, value_enum, default_value_t = Mode::Synthetic)]
    mode: Mode,
    /// Comma-separated ORT worker counts.
    #[arg(long, default_value = "1,2,4")]
    workers: String,
    /// Comma-separated request concurrency levels.
    #[arg(long, default_value = "1,2,4,8")]
    concurrency: String,
    /// Untimed requests before each worker pool's measured matrix.
    #[arg(long, default_value_t = 3)]
    warmups: usize,
    /// Completed owner requests per matrix cell.
    #[arg(long, default_value_t = 30)]
    iterations: usize,
    /// Generated tokens per real-ORT request.
    #[arg(long, default_value_t = 4)]
    max_new_tokens: usize,
    /// Deterministic fixed-work iterations per synthetic request.
    #[arg(long, default_value_t = 1_000_000)]
    work_units: usize,
    /// Model directory for real ORT mode.
    #[arg(long)]
    model: Option<PathBuf>,
    /// Explicit ORT execution provider for real mode.
    #[arg(long, default_value = "cpu")]
    provider: String,
    /// ORT intra-op threads per shared Session.
    #[arg(long, default_value_t = 1)]
    intra_op_threads: i32,
    /// Include every timing sample in JSON instead of summary statistics only.
    #[arg(long)]
    raw_samples: bool,
    /// Write the full machine-readable report here instead of stdout.
    #[arg(long)]
    output: Option<PathBuf>,
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();
    let worker_counts = parse_matrix(&args.workers, "workers")?;
    let concurrency_levels = parse_matrix(&args.concurrency, "concurrency")?;
    let mode = match args.mode {
        Mode::Synthetic => BenchmarkMode::Synthetic {
            work_units: args.work_units,
        },
        Mode::Ort => BenchmarkMode::Ort {
            model_dir: args.model.unwrap_or_else(default_fixture),
            provider: args.provider,
            intra_op_threads: args.intra_op_threads,
        },
    };
    let host_window = onnx_runtime_hostmon::window::Window::open();
    let mut report = run(BenchmarkOptions {
        mode,
        worker_counts,
        concurrency_levels,
        warmups: args.warmups,
        iterations: args.iterations,
        max_new_tokens: args.max_new_tokens,
        include_raw_samples: args.raw_samples,
    })
    .await?;
    let host_lock = host_window.close();
    report.environment.host_lock = Some(host_lock.field().to_string());
    report.environment.host_lock_reason = host_lock.reason().map(str::to_string);
    report.environment.host_lock_protected = Some(host_lock.is_protected());
    report.environment.host_lock_warning = host_lock.warning();
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(output) = args.output {
        std::fs::write(&output, format!("{json}\n"))
            .with_context(|| format!("write {}", output.display()))?;
        print_summary(&report);
        println!("json={}", output.display());
    } else {
        println!("{json}");
    }
    Ok(())
}

fn parse_matrix(value: &str, name: &str) -> Result<Vec<usize>> {
    let values = value
        .split(',')
        .map(str::trim)
        .filter(|entry| !entry.is_empty())
        .map(|entry| {
            entry
                .parse::<usize>()
                .with_context(|| format!("invalid --{name} value '{entry}'"))
        })
        .collect::<Result<Vec<_>>>()?;
    if values.is_empty() || values.contains(&0) {
        bail!("--{name} must contain nonzero comma-separated integers");
    }
    Ok(values)
}

fn default_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm")
}

fn print_summary(report: &BenchmarkReport) {
    println!(
        "| scenario | W | C | wall ms | speedup | TTFT p50/p95/p99 ms | total p50/p95/p99 ms | req/s | prefix hits req/tok | conflicts | overlap |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for row in &report.rows {
        println!(
            "| {} | {} | {} | {:.3} | {} | {} | {} | {:.2} | {}/{} | {} | {} |",
            row.scenario,
            row.worker_count,
            row.concurrency,
            row.wall_ms,
            optional(row.wall_speedup_vs_w1),
            latency(&row.ttft_ms),
            latency(&row.total_latency_ms),
            row.request_throughput_per_second,
            row.prefix_cache_hit_requests,
            row.prefix_cache_hit_tokens,
            row.conflicts,
            row.max_steady_state_overlap,
        );
    }
}

fn optional(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_string(), |value| format!("{value:.3}x"))
}

fn latency(summary: &onnx_genai_server::benchmark::LatencySummary) -> String {
    format!(
        "{}/{}/{}",
        decimal(summary.p50),
        decimal(summary.p95),
        decimal(summary.p99)
    )
}

fn decimal(value: Option<f64>) -> String {
    value.map_or_else(|| "-".to_string(), |value| format!("{value:.3}"))
}

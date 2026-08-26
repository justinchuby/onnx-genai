//! Reproducible session-concurrency and ORT worker-sharding measurements.
//!
//! The synthetic mode drives the production lease map, least-loaded worker
//! selection, counters, command channels, and owner threads around deterministic
//! fixed CPU work. The ORT mode drives the production [`EngineDriver`] with a
//! committed or caller-provided model and also measures the closest direct
//! decode and direct engine paths.

use std::{
    collections::BTreeMap,
    convert::Infallible,
    future::Future,
    hint::black_box,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc as std_mpsc,
    },
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail, ensure};
use onnx_genai::{Engine, GeneratePrompt, GenerateRequest};
use onnx_genai_engine::{EngineConfig, EngineDecodeBackend, PackageCapabilityError};
use onnx_genai_ort::{
    DecodeSession, DecodeSessionOptions, Environment, ModelDirectory, Session, SessionOptions,
    ep_selection,
};
use serde::Serialize;
use tokio::sync::Mutex as AsyncMutex;

use crate::{
    driver::{DriverCommand, DriverEvent, DriverGeneration, EngineDriver, GenerateSubmitError},
    lease::{ModelKey, ModelSessionPlacement, SessionLeases},
    worker::{SessionPlacement, WorkerHandle, WorkerId, WorkerPool},
};

const FIXED_SEED: u64 = 20_260_825;
const SERIAL_TURNS_PER_SESSION: usize = 8;

#[derive(Debug, Clone)]
pub enum BenchmarkMode {
    Synthetic {
        work_units: usize,
    },
    Ort {
        model_dir: PathBuf,
        provider: String,
        intra_op_threads: i32,
    },
}

#[derive(Debug, Clone)]
pub struct BenchmarkOptions {
    pub mode: BenchmarkMode,
    pub worker_counts: Vec<usize>,
    pub concurrency_levels: Vec<usize>,
    pub warmups: usize,
    pub iterations: usize,
    pub max_new_tokens: usize,
    pub include_raw_samples: bool,
}

#[derive(Debug, Serialize)]
pub struct BenchmarkReport {
    pub schema_version: u32,
    pub environment: EnvironmentReport,
    pub configuration: ConfigurationReport,
    pub invariants: InvariantReport,
    pub overhead: OverheadReport,
    pub rows: Vec<BenchmarkRow>,
}

impl BenchmarkReport {
    pub fn ensure_correctness_gates_passed(&self) -> Result<()> {
        let failed = failed_gate_names(&self.invariants);
        ensure!(
            failed.is_empty(),
            "benchmark correctness gates failed: {}",
            failed.join(", ")
        );
        Ok(())
    }
}

fn failed_gate_names(invariants: &InvariantReport) -> Vec<&'static str> {
    [
        ("w1_output_parity", invariants.w1_output_parity),
        (
            "typed_same_session_conflict",
            invariants.typed_same_session_conflict,
        ),
        (
            "distinct_session_stream_overlap",
            invariants.distinct_session_stream_overlap,
        ),
        (
            "exact_completion_counts",
            invariants.exact_completion_counts,
        ),
        ("no_counter_drift", invariants.no_counter_drift),
    ]
    .into_iter()
    .filter_map(|(name, value)| (value == Some(false)).then_some(name))
    .collect()
}

#[derive(Debug, Serialize)]
pub struct EnvironmentReport {
    pub timestamp_utc: String,
    pub commit: String,
    pub dirty: bool,
    pub command: Vec<String>,
    pub package_version: String,
    pub build_profile: String,
    pub rustc: String,
    pub os: String,
    pub kernel: String,
    pub architecture: String,
    pub cpu_model: Option<String>,
    pub logical_cpus: usize,
    pub cpu_affinity: Option<String>,
    pub host_memory_bytes: Option<u64>,
    pub initial_rss_bytes: Option<u64>,
    pub initial_peak_rss_bytes: Option<u64>,
    pub gpu: Option<String>,
    pub host_lock: Option<String>,
    pub host_lock_reason: Option<String>,
    pub host_lock_protected: Option<bool>,
    pub host_lock_warning: Option<String>,
    pub ort_runtime: Option<String>,
    pub available_execution_providers: Option<Vec<String>>,
    pub model_artifact: Option<ModelArtifactReport>,
}

#[derive(Debug, Serialize)]
pub struct ModelArtifactReport {
    pub directory: String,
    pub model_path: String,
    pub sha256: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ConfigurationReport {
    pub mode: String,
    pub worker_counts: Vec<usize>,
    pub concurrency_levels: Vec<usize>,
    pub warmups: usize,
    pub iterations: usize,
    pub max_new_tokens: usize,
    pub seed: u64,
    pub prompt_formula: String,
    pub synthetic_work_units: Option<usize>,
    pub provider: Option<String>,
    pub intra_op_threads: Option<i32>,
    pub prefix_cache_policy: &'static str,
    pub raw_samples: bool,
    pub percentile_method: &'static str,
}

#[derive(Debug, Default, Serialize)]
pub struct InvariantReport {
    pub w1_output_parity: Option<bool>,
    pub typed_same_session_conflict: Option<bool>,
    pub distinct_session_stream_overlap: Option<bool>,
    pub exact_completion_counts: Option<bool>,
    pub no_counter_drift: Option<bool>,
}

#[derive(Debug, Default, Serialize)]
pub struct OverheadReport {
    pub direct_model_p50_ms: Option<f64>,
    pub direct_engine_p50_ms: Option<f64>,
    pub w1_driver_p50_ms: Option<f64>,
    pub workflow_overhead_p50_ms: Option<f64>,
    pub dispatch_queue_overhead_p50_ms: Option<f64>,
    pub note: String,
}

#[derive(Debug, Serialize)]
pub struct BenchmarkRow {
    pub mode: String,
    pub scenario: String,
    pub worker_count: usize,
    pub concurrency: usize,
    pub warmups: usize,
    pub requested_iterations: usize,
    pub unit: String,
    pub prompt_tokens_per_request: Option<usize>,
    pub prompt_tokens_submitted: usize,
    pub prompt_tokens_computed: usize,
    pub prefix_cache_hit_requests: usize,
    pub prefix_cache_hit_tokens: usize,
    pub max_prefix_cache_hit_len: usize,
    pub target_units_per_request: usize,
    pub completed: usize,
    pub errors: usize,
    pub conflicts: usize,
    pub typed_conflicts: usize,
    pub wall_ms: f64,
    pub wall_speedup_vs_w1: Option<f64>,
    pub cpu_time_ms: Option<f64>,
    pub cpu_utilization_percent: Option<f64>,
    pub request_throughput_per_second: f64,
    pub unit_throughput_per_second: f64,
    pub steady_state_units_per_second: Option<f64>,
    pub ttft_ms: LatencySummary,
    pub steady_state_latency_ms: LatencySummary,
    pub total_latency_ms: LatencySummary,
    pub conflict_latency_ms: LatencySummary,
    pub max_client_stream_overlap: usize,
    pub worker_completions: BTreeMap<usize, usize>,
    pub rss_bytes_after: Option<u64>,
    pub peak_rss_bytes_after: Option<u64>,
    pub governed_resources: Option<GovernedResourcesReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<RawSamples>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GovernedResourcesReport {
    pub host: ResourceTierReport,
    pub disk: Option<ResourceTierReport>,
    pub device: ResourceTierReport,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResourceTierReport {
    pub used_bytes: u64,
    pub limit_bytes: u64,
    pub headroom_bytes: u64,
}

#[derive(Debug, Default, Serialize)]
pub struct RawSamples {
    pub ttft_us: Vec<u64>,
    pub steady_state_us: Vec<u64>,
    pub total_us: Vec<u64>,
    pub conflict_ns: Vec<u64>,
    pub prefix_cache_hit_len: Vec<usize>,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct LatencySummary {
    pub count: usize,
    pub min: Option<f64>,
    pub mean: Option<f64>,
    pub p50: Option<f64>,
    pub p95: Option<f64>,
    pub p99: Option<f64>,
    pub max: Option<f64>,
}

#[derive(Debug)]
struct TimedSample {
    started: Instant,
    first: Instant,
    finished: Instant,
    prompt_tokens: usize,
    prefix_cache_hit_len: usize,
    units: usize,
    worker: usize,
    checksum: u64,
    token_ids: Vec<u32>,
}

#[derive(Debug, Default)]
struct Measurement {
    samples: Vec<TimedSample>,
    conflict_ns: Vec<u64>,
    errors: usize,
    typed_conflicts: usize,
    wall: Duration,
    cpu_time: Option<Duration>,
    governed_resources: Option<GovernedResourcesReport>,
}

#[derive(Debug, Default)]
struct JobResults {
    samples: Vec<TimedSample>,
    errors: usize,
}

#[derive(Clone, Copy)]
struct SyntheticSession {
    placement: SessionPlacement,
}

pub async fn run(options: BenchmarkOptions) -> Result<BenchmarkReport> {
    validate_options(&options)?;
    let environment = environment_report(&options.mode)?;
    let configuration = configuration_report(&options);
    let (mut rows, invariants, overhead) = match &options.mode {
        BenchmarkMode::Synthetic { work_units } => {
            run_synthetic_matrix(&options, *work_units).await?
        }
        BenchmarkMode::Ort {
            model_dir,
            provider,
            intra_op_threads,
        } => run_ort_matrix(&options, model_dir, provider, *intra_op_threads).await?,
    };
    apply_w1_speedups(&mut rows);
    if !options.include_raw_samples {
        for row in &mut rows {
            row.raw = None;
        }
    }
    Ok(BenchmarkReport {
        schema_version: 4,
        environment,
        configuration,
        invariants,
        overhead,
        rows,
    })
}

fn validate_options(options: &BenchmarkOptions) -> Result<()> {
    ensure!(
        !options.worker_counts.is_empty(),
        "worker count matrix is empty"
    );
    ensure!(
        options.worker_counts.iter().all(|count| *count > 0),
        "worker counts must be nonzero"
    );
    ensure!(
        options.worker_counts.contains(&1),
        "worker matrix must contain W=1 for parity and speedup baselines"
    );
    ensure!(
        !options.concurrency_levels.is_empty(),
        "concurrency matrix is empty"
    );
    ensure!(
        options.concurrency_levels.iter().all(|count| *count > 0),
        "concurrency levels must be nonzero"
    );
    ensure!(options.iterations > 0, "iterations must be nonzero");
    ensure!(options.max_new_tokens > 0, "max_new_tokens must be nonzero");
    Ok(())
}

fn configuration_report(options: &BenchmarkOptions) -> ConfigurationReport {
    let (mode, work_units, provider, intra_op_threads) = match &options.mode {
        BenchmarkMode::Synthetic { work_units } => {
            ("synthetic".to_string(), Some(*work_units), None, None)
        }
        BenchmarkMode::Ort {
            provider,
            intra_op_threads,
            ..
        } => (
            "ort".to_string(),
            None,
            Some(provider.clone()),
            Some(*intra_op_threads),
        ),
    };
    ConfigurationReport {
        mode,
        worker_counts: options.worker_counts.clone(),
        concurrency_levels: options.concurrency_levels.clone(),
        warmups: options.warmups,
        iterations: options.iterations,
        max_new_tokens: options.max_new_tokens,
        seed: FIXED_SEED,
        prompt_formula: "[1 + i%29, 1 + (7i+3)%29, 1 + (13i+5)%29]".to_string(),
        synthetic_work_units: work_units,
        provider,
        intra_op_threads,
        prefix_cache_policy: if matches!(&options.mode, BenchmarkMode::Ort { .. }) {
            "cold_start=true; every measured prefix_cache_hit_len must be zero"
        } else {
            "not applicable"
        },
        raw_samples: options.include_raw_samples,
        percentile_method: "nearest-rank (ceil(q*n)-1), warmups excluded",
    }
}

fn environment_report(mode: &BenchmarkMode) -> Result<EnvironmentReport> {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let commit = command_text("git", &["-C", path_text(&repository), "rev-parse", "HEAD"])
        .unwrap_or_else(|| "unknown".to_string());
    let dirty = command_text(
        "git",
        &["-C", path_text(&repository), "status", "--porcelain"],
    )
    .is_some_and(|output| !output.is_empty());
    let memory = process_memory();
    let model_artifact = match mode {
        BenchmarkMode::Synthetic { .. } => None,
        BenchmarkMode::Ort { model_dir, .. } => {
            let directory = ModelDirectory::load(model_dir)
                .with_context(|| format!("resolve model directory {}", model_dir.display()))?;
            Some(ModelArtifactReport {
                directory: model_dir.display().to_string(),
                model_path: directory.model_path.display().to_string(),
                sha256: command_text("sha256sum", &[path_text(directory.model_path.as_path())])
                    .and_then(|line| line.split_whitespace().next().map(str::to_string)),
            })
        }
    };
    let ort_runtime =
        matches!(mode, BenchmarkMode::Ort { .. }).then(onnx_genai_ort::onnxruntime_library_report);
    let available_execution_providers = matches!(mode, BenchmarkMode::Ort { .. }).then(|| {
        onnx_genai_ort::available_execution_providers()
            .unwrap_or_else(|error| vec![format!("provider query failed: {error}")])
    });
    Ok(EnvironmentReport {
        timestamp_utc: command_text("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"])
            .unwrap_or_else(|| "unknown".to_string()),
        commit,
        dirty,
        command: std::env::args().collect(),
        package_version: env!("CARGO_PKG_VERSION").to_string(),
        build_profile: if cfg!(debug_assertions) {
            "debug".to_string()
        } else {
            "release".to_string()
        },
        rustc: command_text("rustc", &["--version"]).unwrap_or_else(|| "unknown".to_string()),
        os: std::env::consts::OS.to_string(),
        kernel: command_text("uname", &["-srvo"]).unwrap_or_else(|| "unknown".to_string()),
        architecture: std::env::consts::ARCH.to_string(),
        cpu_model: cpu_model(),
        logical_cpus: std::thread::available_parallelism().map_or(1, usize::from),
        cpu_affinity: process_status_value("Cpus_allowed_list:"),
        host_memory_bytes: host_memory_bytes(),
        initial_rss_bytes: memory.rss,
        initial_peak_rss_bytes: memory.peak_rss,
        gpu: gpu_report(),
        host_lock: None,
        host_lock_reason: None,
        host_lock_protected: None,
        host_lock_warning: None,
        ort_runtime,
        available_execution_providers,
        model_artifact,
    })
}

async fn run_synthetic_matrix(
    options: &BenchmarkOptions,
    work_units: usize,
) -> Result<(Vec<BenchmarkRow>, InvariantReport, OverheadReport)> {
    ensure!(work_units > 0, "synthetic work units must be nonzero");
    let direct = measure_synthetic_direct(options.warmups, options.iterations, work_units).await?;
    let direct_checksum = direct
        .samples
        .first()
        .context("synthetic direct path produced no samples")?
        .checksum;
    let direct_row = row_from_measurement(
        "synthetic",
        "direct_fixed_work",
        0,
        1,
        options.warmups,
        options.iterations,
        direct,
    );
    let direct_p50 = direct_row.total_latency_ms.p50;
    let mut rows = vec![direct_row];
    let mut invariants = InvariantReport {
        w1_output_parity: Some(true),
        typed_same_session_conflict: None,
        distinct_session_stream_overlap: None,
        exact_completion_counts: Some(true),
        no_counter_drift: Some(true),
    };

    for &workers in &options.worker_counts {
        let pool = Arc::new(synthetic_pool(workers)?);
        let leases = SessionLeases::with_shards(8);
        for _ in 0..options.warmups {
            let sample = run_synthetic_request(&pool, None, work_units, Instant::now()).await?;
            ensure!(
                sample.checksum == direct_checksum,
                "synthetic warmup checksum drift"
            );
        }
        for &concurrency in &options.concurrency_levels {
            let serialized = measure_synthetic_serialized(
                &pool,
                &leases,
                options.iterations,
                concurrency,
                work_units,
            )
            .await?;
            record_result_gate(
                &mut invariants.exact_completion_counts,
                verify_measurement(&serialized, options.iterations, 0),
            );
            record_gate(
                &mut invariants.w1_output_parity,
                serialized
                    .samples
                    .iter()
                    .all(|sample| sample.checksum == direct_checksum),
            );
            rows.push(row_from_measurement(
                "synthetic",
                "one_session_serialized",
                workers,
                concurrency,
                options.warmups,
                options.iterations,
                serialized,
            ));

            let conflict = measure_synthetic_conflicts(
                &pool,
                &leases,
                options.iterations,
                concurrency,
                work_units,
            )
            .await?;
            let expected_conflicts = options.iterations * concurrency.saturating_sub(1);
            record_result_gate(
                &mut invariants.exact_completion_counts,
                verify_measurement(&conflict, options.iterations, expected_conflicts),
            );
            if workers > 1 && concurrency > 1 {
                record_gate(
                    &mut invariants.typed_same_session_conflict,
                    conflict.typed_conflicts == expected_conflicts,
                );
            }
            rows.push(row_from_measurement(
                "synthetic",
                "same_session_conflict",
                workers,
                concurrency,
                options.warmups,
                options.iterations,
                conflict,
            ));

            let distinct = measure_synthetic_distinct(
                &pool,
                &leases,
                options.iterations,
                concurrency,
                work_units,
            )
            .await?;
            record_result_gate(
                &mut invariants.exact_completion_counts,
                verify_measurement(&distinct, options.iterations, 0),
            );
            let overlap = max_client_stream_overlap(&distinct.samples);
            if workers > 1 && concurrency > 1 && options.iterations > 1 {
                record_gate(&mut invariants.distinct_session_stream_overlap, overlap > 1);
            }
            rows.push(row_from_measurement(
                "synthetic",
                "distinct_sessions",
                workers,
                concurrency,
                options.warmups,
                options.iterations,
                distinct,
            ));

            let stateless =
                measure_synthetic_stateless(&pool, options.iterations, concurrency, work_units)
                    .await?;
            record_result_gate(
                &mut invariants.exact_completion_counts,
                verify_measurement(&stateless, options.iterations, 0),
            );
            if workers == 1 {
                record_gate(
                    &mut invariants.w1_output_parity,
                    stateless
                        .samples
                        .iter()
                        .all(|sample| sample.checksum == direct_checksum),
                );
            }
            rows.push(row_from_measurement(
                "synthetic",
                "stateless_least_loaded",
                workers,
                concurrency,
                options.warmups,
                options.iterations,
                stateless,
            ));

            record_result_gate(
                &mut invariants.no_counter_drift,
                assert_synthetic_counters(&pool, &leases),
            );
        }
        pool.shutdown();
    }

    let w1_driver_p50 = rows
        .iter()
        .find(|row| {
            row.scenario == "stateless_least_loaded"
                && row.worker_count == 1
                && row.concurrency == 1
        })
        .and_then(|row| row.total_latency_ms.p50);
    let dispatch_overhead = difference(w1_driver_p50, direct_p50);
    Ok((
        rows,
        invariants,
        OverheadReport {
            direct_model_p50_ms: direct_p50,
            direct_engine_p50_ms: None,
            w1_driver_p50_ms: w1_driver_p50,
            workflow_overhead_p50_ms: None,
            dispatch_queue_overhead_p50_ms: dispatch_overhead,
            note: "Synthetic overhead is routing + channel + owner-thread handshake around the \
                   exact same deterministic CPU function; it is not model performance."
                .to_string(),
        },
    ))
}

async fn run_ort_matrix(
    options: &BenchmarkOptions,
    model_dir: &Path,
    provider: &str,
    intra_op_threads: i32,
) -> Result<(Vec<BenchmarkRow>, InvariantReport, OverheadReport)> {
    ensure!(
        model_dir.is_dir(),
        "model directory does not exist: {}",
        model_dir.display()
    );
    let direct_model = measure_direct_decode(
        model_dir,
        provider,
        intra_op_threads,
        options.warmups,
        options.iterations,
        options.max_new_tokens,
    )?;
    verify_zero_prefix_cache_hits(&direct_model, "direct ORT decode")?;
    let direct_model_tokens = direct_model
        .samples
        .first()
        .context("direct decode produced no samples")?
        .token_ids
        .clone();
    let direct_engine = measure_direct_engine(
        model_dir,
        provider,
        intra_op_threads,
        options.warmups,
        options.iterations,
        options.max_new_tokens,
    )?;
    verify_zero_prefix_cache_hits(&direct_engine, "direct Engine workflow")?;
    verify_equivalent_model_work(&direct_model, &direct_engine)?;
    let direct_engine_tokens = direct_engine
        .samples
        .first()
        .context("direct engine produced no samples")?
        .token_ids
        .clone();
    ensure!(
        direct_engine_tokens == direct_model_tokens,
        "direct DecodeSession and direct Engine token parity failed: decode={direct_model_tokens:?}, \
         engine={direct_engine_tokens:?}"
    );
    let direct_model_row = row_from_measurement(
        "ort",
        "direct_ort_decode",
        0,
        1,
        options.warmups,
        options.iterations,
        direct_model,
    );
    let direct_model_p50 = direct_model_row.total_latency_ms.p50;
    let direct_engine_row = row_from_measurement(
        "ort",
        "direct_engine_workflow",
        0,
        1,
        options.warmups,
        options.iterations,
        direct_engine,
    );
    let direct_engine_p50 = direct_engine_row.total_latency_ms.p50;
    let mut rows = vec![direct_model_row, direct_engine_row];
    let mut invariants = InvariantReport {
        w1_output_parity: Some(true),
        typed_same_session_conflict: None,
        distinct_session_stream_overlap: None,
        exact_completion_counts: Some(true),
        no_counter_drift: Some(true),
    };

    for &workers in &options.worker_counts {
        let driver = Arc::new(build_ort_driver(
            model_dir,
            provider,
            intra_op_threads,
            workers,
            options
                .concurrency_levels
                .iter()
                .copied()
                .max()
                .unwrap_or(1)
                * 2,
        )?);
        for warmup in 0..options.warmups {
            let sample = run_driver_request(
                Arc::clone(&driver),
                None,
                request(warmup, options.max_new_tokens),
                Instant::now(),
            )
            .await?;
            ensure!(sample.units == options.max_new_tokens, "short ORT warmup");
            ensure!(
                sample.prefix_cache_hit_len == 0,
                "ORT warmup reused {} prompt tokens despite cold_start",
                sample.prefix_cache_hit_len
            );
        }
        if workers == 1 {
            let parity = run_driver_request(
                Arc::clone(&driver),
                None,
                request(options.warmups, options.max_new_tokens),
                Instant::now(),
            )
            .await?;
            ensure!(
                parity.prefix_cache_hit_len == 0,
                "W=1 parity request reused {} prompt tokens despite cold_start",
                parity.prefix_cache_hit_len
            );
            record_gate(
                &mut invariants.w1_output_parity,
                parity.token_ids == direct_engine_tokens,
            );
        }

        let leases = SessionLeases::with_shards(8);
        for &concurrency in &options.concurrency_levels {
            let serialized = measure_driver_serialized(
                Arc::clone(&driver),
                &leases,
                options.iterations,
                concurrency,
            )
            .await?;
            record_result_gate(
                &mut invariants.exact_completion_counts,
                verify_measurement(&serialized, options.iterations, 0),
            );
            verify_zero_prefix_cache_hits(&serialized, "serialized ORT session")?;
            rows.push(row_from_measurement(
                "ort",
                "one_session_serialized",
                workers,
                concurrency,
                options.warmups,
                options.iterations,
                serialized,
            ));

            let conflict = measure_driver_conflicts(
                Arc::clone(&driver),
                &leases,
                options.iterations,
                concurrency,
                options.max_new_tokens,
                options.warmups,
            )
            .await?;
            let expected_conflicts = options.iterations * concurrency.saturating_sub(1);
            record_result_gate(
                &mut invariants.exact_completion_counts,
                verify_measurement(&conflict, options.iterations, expected_conflicts),
            );
            verify_zero_prefix_cache_hits(&conflict, "same-session conflict owner")?;
            if workers > 1 && concurrency > 1 {
                record_gate(
                    &mut invariants.typed_same_session_conflict,
                    conflict.typed_conflicts == expected_conflicts,
                );
            }
            rows.push(row_from_measurement(
                "ort",
                "same_session_conflict",
                workers,
                concurrency,
                options.warmups,
                options.iterations,
                conflict,
            ));

            let distinct = measure_driver_distinct(
                Arc::clone(&driver),
                &leases,
                options.iterations,
                concurrency,
                options.max_new_tokens,
                options.warmups,
            )
            .await?;
            record_result_gate(
                &mut invariants.exact_completion_counts,
                verify_measurement(&distinct, options.iterations, 0),
            );
            verify_zero_prefix_cache_hits(&distinct, "distinct ORT sessions")?;
            let overlap = max_client_stream_overlap(&distinct.samples);
            if workers > 1 && concurrency > 1 && options.iterations > 1 {
                record_gate(&mut invariants.distinct_session_stream_overlap, overlap > 1);
            }
            rows.push(row_from_measurement(
                "ort",
                "distinct_sessions",
                workers,
                concurrency,
                options.warmups,
                options.iterations,
                distinct,
            ));

            let stateless = measure_driver_stateless(
                Arc::clone(&driver),
                options.iterations,
                concurrency,
                options.max_new_tokens,
                options.warmups,
            )
            .await?;
            record_result_gate(
                &mut invariants.exact_completion_counts,
                verify_measurement(&stateless, options.iterations, 0),
            );
            verify_zero_prefix_cache_hits(&stateless, "stateless ORT requests")?;
            rows.push(row_from_measurement(
                "ort",
                "stateless_least_loaded",
                workers,
                concurrency,
                options.warmups,
                options.iterations,
                stateless,
            ));

            record_result_gate(
                &mut invariants.no_counter_drift,
                assert_driver_counters(&driver, &leases),
            );
        }
        let shutdown = Arc::try_unwrap(driver)
            .map_err(|_| anyhow::anyhow!("benchmark retained an EngineDriver clone"))?;
        shutdown.shutdown();
    }

    let w1_driver_p50 = rows
        .iter()
        .find(|row| {
            row.scenario == "stateless_least_loaded"
                && row.worker_count == 1
                && row.concurrency == 1
        })
        .and_then(|row| row.total_latency_ms.p50);
    Ok((
        rows,
        invariants,
        OverheadReport {
            direct_model_p50_ms: direct_model_p50,
            direct_engine_p50_ms: direct_engine_p50,
            w1_driver_p50_ms: w1_driver_p50,
            workflow_overhead_p50_ms: difference(direct_engine_p50, direct_model_p50),
            dispatch_queue_overhead_p50_ms: difference(w1_driver_p50, direct_engine_p50),
            note: "Direct ORT DecodeSession isolates incremental model execution; direct Engine \
                   adds workflow interpretation, sampling, scheduler, and session lifecycle; W=1 \
                   driver adds routing, admission, channel, and streaming. Negative deltas are \
                   possible because caches and allocator state differ; treat them as attribution \
                   bounds, not additive constants."
                .to_string(),
        },
    ))
}

fn build_ort_driver(
    model_dir: &Path,
    provider: &str,
    intra_op_threads: i32,
    worker_count: usize,
    queue_depth: usize,
) -> Result<EngineDriver> {
    let directory = ModelDirectory::load(model_dir)
        .with_context(|| format!("resolve model directory {}", model_dir.display()))?;
    let options = session_options(provider, intra_op_threads);
    let config = EngineConfig {
        decode_backend: EngineDecodeBackend::Ort,
        ..EngineConfig::default()
    };
    let build_dir = model_dir.to_path_buf();
    let build_config = config.clone();
    let (mut driver, ()) = EngineDriver::start_ort_workers(
        move || {
            let engine =
                Engine::from_dir_with_session_options(&build_dir, build_config.clone(), options)?;
            let factory = Arc::new(
                engine
                    .ort_worker_factory(directory, build_config)
                    .context("requested ORT worker sharding is unsupported")?,
            );
            Ok((engine, (), factory))
        },
        worker_count,
        1,
        queue_depth.max(1),
    )?;
    driver.bind_model("session-concurrency-benchmark");
    Ok(driver)
}

fn session_options(provider: &str, intra_op_threads: i32) -> SessionOptions {
    SessionOptions::with_execution_provider(ep_selection(provider))
        .with_intra_op_threads(intra_op_threads)
}

fn request(index: usize, max_new_tokens: usize) -> GenerateRequest {
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(prompt(index)));
    request.options.max_new_tokens = max_new_tokens;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.seed = Some(FIXED_SEED);
    request.options.stop_on_eos = false;
    request.options.cold_start = true;
    request
}

fn serialized_request(index: usize) -> GenerateRequest {
    let mut request = GenerateRequest::new(GeneratePrompt::TokenIds(vec![1 + (index % 29) as u32]));
    request.options.max_new_tokens = 1;
    request.options.temperature = 0.0;
    request.options.greedy = true;
    request.options.seed = Some(FIXED_SEED);
    request.options.stop_on_eos = false;
    request.options.cold_start = true;
    request
}

fn benchmark_prompt_tokens(request: &GenerateRequest) -> Result<usize> {
    match &request.prompt {
        GeneratePrompt::TokenIds(tokens) => Ok(tokens.len()),
        GeneratePrompt::Text(_) | GeneratePrompt::TokenRows(_) => {
            bail!("session-concurrency benchmark requires token-id prompts")
        }
    }
}

fn prompt(index: usize) -> Vec<u32> {
    vec![
        1 + (index % 29) as u32,
        1 + ((index * 7 + 3) % 29) as u32,
        1 + ((index * 13 + 5) % 29) as u32,
    ]
}

fn measure_direct_decode(
    model_dir: &Path,
    provider: &str,
    intra_op_threads: i32,
    warmups: usize,
    iterations: usize,
    max_new_tokens: usize,
) -> Result<Measurement> {
    let directory = ModelDirectory::load(model_dir)?;
    let environment = Environment::new("session-concurrency-direct-decode")?;
    let session = Session::new(
        &environment,
        &directory.model_path,
        session_options(provider, intra_op_threads),
    )?;
    let metadata_path = directory
        .metadata_path
        .as_deref()
        .context("direct decode requires inference metadata")?;
    let metadata = onnx_genai_metadata::load_metadata(metadata_path)?;
    for index in 0..warmups {
        let _ = direct_decode_sample(&session, metadata.decoder_io(), index, max_new_tokens)?;
    }
    let before_cpu = process_cpu_ticks();
    let wall_started = Instant::now();
    let mut samples = Vec::with_capacity(iterations);
    for index in 0..iterations {
        samples.push(direct_decode_sample(
            &session,
            metadata.decoder_io(),
            warmups + index,
            max_new_tokens,
        )?);
    }
    let wall = wall_started.elapsed();
    Ok(Measurement {
        samples,
        wall,
        cpu_time: cpu_duration(before_cpu, process_cpu_ticks()),
        ..Measurement::default()
    })
}

fn direct_decode_sample(
    session: &Session,
    io: Option<&onnx_genai_metadata::DecoderAbi>,
    index: usize,
    max_new_tokens: usize,
) -> Result<TimedSample> {
    let prompt = prompt(index);
    let prompt_i64 = prompt
        .iter()
        .map(|token| i64::from(*token))
        .collect::<Vec<_>>();
    let mut decode = DecodeSession::new_with_io(session, DecodeSessionOptions::default(), io)?;
    let started = Instant::now();
    let first_token = decode.step_argmax(
        &prompt_i64,
        &vec![1; prompt.len()],
        &(0..prompt.len())
            .map(|position| position as i64)
            .collect::<Vec<_>>(),
    )?;
    let first = Instant::now();
    let mut token_ids = vec![first_token];
    for generated_index in 1..max_new_tokens {
        let position = prompt.len() + generated_index - 1;
        let next = decode.step_argmax(
            &[i64::from(*token_ids.last().expect("first token exists"))],
            &vec![1; position + 1],
            &[position as i64],
        )?;
        token_ids.push(next);
    }
    let finished = Instant::now();
    Ok(TimedSample {
        started,
        first,
        finished,
        prompt_tokens: prompt.len(),
        prefix_cache_hit_len: 0,
        units: token_ids.len(),
        worker: 0,
        checksum: token_checksum(&token_ids),
        token_ids,
    })
}

fn measure_direct_engine(
    model_dir: &Path,
    provider: &str,
    intra_op_threads: i32,
    warmups: usize,
    iterations: usize,
    max_new_tokens: usize,
) -> Result<Measurement> {
    let config = EngineConfig {
        decode_backend: EngineDecodeBackend::Ort,
        ..EngineConfig::default()
    };
    let mut engine = Engine::from_dir_with_session_options(
        model_dir,
        config,
        session_options(provider, intra_op_threads),
    )?;
    for index in 0..warmups {
        let result = engine.generate(request(index, max_new_tokens))?;
        ensure!(
            result.prefix_cache_hit_len == 0,
            "direct Engine warmup reused {} prompt tokens despite cold_start",
            result.prefix_cache_hit_len
        );
    }
    let before_cpu = process_cpu_ticks();
    let wall_started = Instant::now();
    let mut samples = Vec::with_capacity(iterations);
    for index in 0..iterations {
        let started = Instant::now();
        let mut first = None;
        let mut callback = |_token| {
            first.get_or_insert_with(Instant::now);
            Ok(())
        };
        let result = engine.generate_with_callback(
            request(warmups + index, max_new_tokens),
            Some(&mut callback),
        )?;
        let finished = Instant::now();
        let prefix_cache_hit_len = result.prefix_cache_hit_len;
        let token_ids = result.token_ids;
        samples.push(TimedSample {
            started,
            first: first.context("direct engine emitted no token")?,
            finished,
            prompt_tokens: prompt(warmups + index).len(),
            prefix_cache_hit_len,
            units: token_ids.len(),
            worker: 0,
            checksum: token_checksum(&token_ids),
            token_ids,
        });
    }
    let wall = wall_started.elapsed();
    Ok(Measurement {
        samples,
        wall,
        cpu_time: cpu_duration(before_cpu, process_cpu_ticks()),
        ..Measurement::default()
    })
}

async fn measure_driver_stateless(
    driver: Arc<EngineDriver>,
    iterations: usize,
    concurrency: usize,
    max_new_tokens: usize,
    prompt_start: usize,
) -> Result<Measurement> {
    let driver_for_jobs = Arc::clone(&driver);
    let before_cpu = process_cpu_ticks();
    let started = Instant::now();
    let jobs = run_jobs(iterations, concurrency, move |index| {
        let driver = Arc::clone(&driver_for_jobs);
        async move {
            run_driver_request(
                driver,
                None,
                request(prompt_start + index, max_new_tokens),
                Instant::now(),
            )
            .await
        }
    })
    .await?;
    let wall = started.elapsed();
    let governed_resources = governed_resources(&driver).await;
    Ok(Measurement {
        samples: jobs.samples,
        errors: jobs.errors,
        wall,
        cpu_time: cpu_duration(before_cpu, process_cpu_ticks()),
        governed_resources,
        ..Measurement::default()
    })
}

async fn measure_driver_distinct(
    driver: Arc<EngineDriver>,
    leases: &Arc<SessionLeases>,
    iterations: usize,
    concurrency: usize,
    max_new_tokens: usize,
    prompt_start: usize,
) -> Result<Measurement> {
    let mut sessions = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        sessions.push(driver.create_session().await?);
    }
    let sessions = Arc::new(sessions);
    let sessions_for_jobs = Arc::clone(&sessions);
    let leases_for_jobs = Arc::clone(leases);
    let driver_for_jobs = Arc::clone(&driver);
    let before_cpu = process_cpu_ticks();
    let started = Instant::now();
    let jobs = run_jobs(iterations, concurrency, move |index| {
        let driver = Arc::clone(&driver_for_jobs);
        let leases = Arc::clone(&leases_for_jobs);
        let placement = sessions_for_jobs[index];
        async move {
            let lease = leases
                .acquire(driver.binding(placement), &format!("distinct-{index}"))
                .context("acquire distinct-session lease")?;
            run_driver_request(
                driver,
                Some(lease),
                request(prompt_start + index, max_new_tokens),
                Instant::now(),
            )
            .await
        }
    })
    .await?;
    let wall = started.elapsed();
    let cpu_time = cpu_duration(before_cpu, process_cpu_ticks());
    for (index, placement) in Arc::try_unwrap(sessions)
        .map_err(|_| anyhow::anyhow!("distinct session list retained"))?
        .into_iter()
        .enumerate()
    {
        let lease = leases
            .acquire(driver.binding(placement), &format!("distinct-{index}"))
            .context("acquire close lease")?;
        driver.close_session(lease).await?;
    }
    let governed_resources = governed_resources(&driver).await;
    Ok(Measurement {
        samples: jobs.samples,
        errors: jobs.errors,
        wall,
        cpu_time,
        governed_resources,
        ..Measurement::default()
    })
}

async fn measure_driver_serialized(
    driver: Arc<EngineDriver>,
    leases: &Arc<SessionLeases>,
    iterations: usize,
    concurrency: usize,
) -> Result<Measurement> {
    let mut samples = Vec::with_capacity(iterations);
    let mut wall = Duration::ZERO;
    let mut cpu_time = None;
    let mut errors = 0;
    let mut completed = 0;
    while completed < iterations {
        let turns = SERIAL_TURNS_PER_SESSION.min(iterations - completed);
        let placement = driver.create_session().await?;
        let serialization = Arc::new(AsyncMutex::new(()));
        let driver_for_jobs = Arc::clone(&driver);
        let leases_for_jobs = Arc::clone(leases);
        let before_cpu = process_cpu_ticks();
        let trial_started = Instant::now();
        let base = completed;
        let mut trial = run_jobs(turns, concurrency, move |offset| {
            let driver = Arc::clone(&driver_for_jobs);
            let leases = Arc::clone(&leases_for_jobs);
            let serialization = Arc::clone(&serialization);
            let queued_at = Instant::now();
            async move {
                let _serial = serialization.lock().await;
                let lease = leases
                    .acquire(driver.binding(placement), "serialized")
                    .context("serialized lease")?;
                run_driver_request(
                    driver,
                    Some(lease),
                    serialized_request(30_000 + base + offset),
                    queued_at,
                )
                .await
            }
        })
        .await?;
        wall += trial_started.elapsed();
        if let Some(duration) = cpu_duration(before_cpu, process_cpu_ticks()) {
            cpu_time = Some(cpu_time.unwrap_or_default() + duration);
        }
        samples.append(&mut trial.samples);
        errors += trial.errors;
        let lease = leases
            .acquire(driver.binding(placement), "serialized")
            .context("serialized close lease")?;
        driver.close_session(lease).await?;
        completed += turns;
    }
    let governed_resources = governed_resources(&driver).await;
    Ok(Measurement {
        samples,
        errors,
        wall,
        cpu_time,
        governed_resources,
        ..Measurement::default()
    })
}

async fn measure_driver_conflicts(
    driver: Arc<EngineDriver>,
    leases: &Arc<SessionLeases>,
    iterations: usize,
    concurrency: usize,
    max_new_tokens: usize,
    prompt_start: usize,
) -> Result<Measurement> {
    let before_cpu = process_cpu_ticks();
    let wall_started = Instant::now();
    let mut samples = Vec::with_capacity(iterations);
    let mut conflict_ns = Vec::with_capacity(iterations * concurrency.saturating_sub(1));
    let mut typed_conflicts = 0;
    let mut errors = 0;
    for index in 0..iterations {
        let placement = driver.create_session().await?;
        let binding = driver.binding(placement);
        let owner_lease = leases
            .acquire(binding.clone(), "conflict")
            .context("owner conflict lease")?;
        let pending = match submit_driver_request(
            Arc::clone(&driver),
            Some(owner_lease),
            request(prompt_start + index, max_new_tokens),
            Instant::now(),
        )
        .await
        {
            Ok(pending) => pending,
            Err(_) => {
                errors += 1;
                let close = leases
                    .acquire(binding, "conflict")
                    .context("conflict close lease after submit failure")?;
                if driver.close_session(close).await.is_err() {
                    errors += 1;
                }
                continue;
            }
        };
        for _ in 1..concurrency {
            let started = Instant::now();
            match leases.acquire(binding.clone(), "conflict") {
                Err(PackageCapabilityError::ExclusiveLeaseConflict { .. }) => {
                    conflict_ns.push(duration_ns(started.elapsed()));
                    typed_conflicts += 1;
                }
                Err(_) => errors += 1,
                Ok(lease) => {
                    drop(lease);
                    errors += 1;
                }
            }
        }
        match finish_driver_request(pending).await {
            Ok(sample) => samples.push(sample),
            Err(_) => errors += 1,
        }
        let close = leases
            .acquire(binding, "conflict")
            .context("conflict close lease")?;
        if driver.close_session(close).await.is_err() {
            errors += 1;
        }
    }
    let wall = wall_started.elapsed();
    let governed_resources = governed_resources(&driver).await;
    Ok(Measurement {
        samples,
        conflict_ns,
        errors,
        typed_conflicts,
        wall,
        cpu_time: cpu_duration(before_cpu, process_cpu_ticks()),
        governed_resources,
    })
}

struct PendingDriverRequest {
    worker: WorkerId,
    generation: DriverGeneration,
    started: Instant,
    prompt_tokens: usize,
}

async fn submit_driver_request(
    driver: Arc<EngineDriver>,
    lease: Option<crate::lease::SessionLeaseGuard>,
    request: GenerateRequest,
    started: Instant,
) -> Result<PendingDriverRequest> {
    let prompt_tokens = benchmark_prompt_tokens(&request)?;
    let (worker, generation) = driver
        .benchmark_generate(lease, request)
        .await
        .map_err(submit_error)?;
    Ok(PendingDriverRequest {
        worker,
        generation,
        started,
        prompt_tokens,
    })
}

async fn finish_driver_request(mut pending: PendingDriverRequest) -> Result<TimedSample> {
    pending
        .generation
        .admission
        .await
        .context("driver admission channel closed")?
        .map_err(|error| anyhow::anyhow!(error.message))?;
    let mut first = None;
    let mut token_ids = Vec::new();
    while let Some(event) = pending.generation.events.recv().await {
        match event {
            DriverEvent::Token(token) => {
                first.get_or_insert_with(Instant::now);
                token_ids.push(token.token_id);
            }
            DriverEvent::Finished(result) => {
                let finished = Instant::now();
                ensure!(
                    token_ids == result.token_ids,
                    "streamed and finished token ids differ"
                );
                let first = first.context("driver emitted no token")?;
                return Ok(TimedSample {
                    started: pending.started,
                    first,
                    finished,
                    prompt_tokens: pending.prompt_tokens,
                    prefix_cache_hit_len: result.prefix_cache_hit_len,
                    units: token_ids.len(),
                    worker: pending.worker.index(),
                    checksum: token_checksum(&token_ids),
                    token_ids,
                });
            }
            DriverEvent::Error(error) => bail!(error.message),
        }
    }
    bail!("driver stream ended before Finished")
}

async fn run_driver_request(
    driver: Arc<EngineDriver>,
    lease: Option<crate::lease::SessionLeaseGuard>,
    request: GenerateRequest,
    started: Instant,
) -> Result<TimedSample> {
    let pending = submit_driver_request(driver, lease, request, started).await?;
    finish_driver_request(pending).await
}

fn submit_error(error: GenerateSubmitError) -> anyhow::Error {
    match error {
        GenerateSubmitError::Overloaded => anyhow::anyhow!("generation capacity exceeded"),
        GenerateSubmitError::DriverStopped => anyhow::anyhow!("engine driver stopped"),
        GenerateSubmitError::Failed(error) => anyhow::anyhow!(error.message),
    }
}

fn synthetic_pool(worker_count: usize) -> Result<WorkerPool> {
    let mut workers = Vec::with_capacity(worker_count);
    for index in 0..worker_count {
        let id = WorkerId::new(index);
        let (commands, rx) = tokio::sync::mpsc::channel(64);
        let (worker, ()) = WorkerHandle::spawn(
            id,
            format!("onnx-genai-synthetic-benchmark-{id}"),
            commands,
            move || Ok::<_, Infallible>((rx, ())),
            |mut rx| {
                while let Some(command) = rx.blocking_recv() {
                    match command {
                        DriverCommand::Block {
                            entered,
                            release,
                            completed,
                        } => {
                            let _ = entered.send(());
                            let _ = release.recv();
                            if let Some(completed) = completed {
                                let _ = completed.send(());
                            }
                        }
                        _ => panic!("synthetic worker received a non-benchmark command"),
                    }
                }
            },
        )
        .map_err(|error| anyhow::anyhow!("start synthetic worker {id}: {error}"))?;
        workers.push(worker);
    }
    Ok(WorkerPool::new(workers))
}

fn create_synthetic_session(pool: &WorkerPool, sequence: u64) -> Result<SyntheticSession> {
    let reservation = pool.reserve_session_placement()?;
    let worker = reservation.worker();
    reservation.commit()?.persist();
    Ok(SyntheticSession {
        placement: SessionPlacement::new(worker, sequence),
    })
}

fn close_synthetic_session(pool: &WorkerPool, session: SyntheticSession) -> Result<()> {
    pool.worker(session.placement.worker)?
        .session_close_accounting()
        .session_closed();
    Ok(())
}

fn synthetic_binding(session: SyntheticSession) -> ModelSessionPlacement {
    ModelSessionPlacement::new(ModelKey::new("synthetic-benchmark"), session.placement)
}

async fn measure_synthetic_direct(
    warmups: usize,
    iterations: usize,
    work_units: usize,
) -> Result<Measurement> {
    tokio::task::spawn_blocking(move || {
        for _ in 0..warmups {
            black_box(direct_synthetic_sample(work_units));
        }
    })
    .await
    .context("direct synthetic warmup task panicked")?;
    let before_cpu = process_cpu_ticks();
    let started = Instant::now();
    let samples = tokio::task::spawn_blocking(move || {
        (0..iterations)
            .map(|_| direct_synthetic_sample(work_units))
            .collect::<Vec<_>>()
    })
    .await
    .context("direct synthetic task panicked")?;
    let wall = started.elapsed();
    Ok(Measurement {
        samples,
        wall,
        cpu_time: cpu_duration(before_cpu, process_cpu_ticks()),
        ..Measurement::default()
    })
}

fn direct_synthetic_sample(work_units: usize) -> TimedSample {
    let started = Instant::now();
    let split = work_units.div_ceil(2);
    let first_checksum = synthetic_work_range(0, split, FIXED_SEED);
    let first = Instant::now();
    let final_checksum = synthetic_work_range(split, work_units, first_checksum);
    let finished = Instant::now();
    TimedSample {
        started,
        first,
        finished,
        prompt_tokens: 0,
        prefix_cache_hit_len: 0,
        units: work_units,
        worker: 0,
        checksum: final_checksum,
        token_ids: Vec::new(),
    }
}

fn synthetic_work_range(start: usize, end: usize, mut state: u64) -> u64 {
    for index in start..end {
        state ^= (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        state = state.rotate_left(17).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        state ^= state >> 29;
    }
    black_box(state)
}

async fn run_synthetic_request(
    pool: &Arc<WorkerPool>,
    session: Option<(SyntheticSession, crate::lease::SessionLeaseGuard)>,
    work_units: usize,
    started: Instant,
) -> Result<TimedSample> {
    let (worker, turn) = match session.as_ref() {
        Some((session, _)) => (
            session.placement.worker,
            pool.reserve_turn(session.placement.worker)?,
        ),
        None => pool.reserve_stateless_turn()?,
    };
    let sender = pool.sender_for(worker)?;
    let lease = session.map(|(_, lease)| lease);
    tokio::task::spawn_blocking(move || {
        let (entered_tx, entered_rx) = std_mpsc::sync_channel(1);
        let (release_tx, release_rx) = std_mpsc::sync_channel(1);
        let (completed_tx, completed_rx) = std_mpsc::sync_channel(1);
        sender
            .blocking_send(DriverCommand::Block {
                entered: entered_tx,
                release: release_rx,
                completed: Some(completed_tx),
            })
            .map_err(|_| anyhow::anyhow!("synthetic worker stopped"))?;
        entered_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("synthetic worker stopped before execution"))?;
        let split = work_units.div_ceil(2);
        let first_checksum = synthetic_work_range(0, split, FIXED_SEED);
        let first = Instant::now();
        let final_checksum = synthetic_work_range(split, work_units, first_checksum);
        release_tx
            .send(())
            .map_err(|_| anyhow::anyhow!("synthetic worker release failed"))?;
        completed_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("synthetic worker did not complete"))?;
        let finished = Instant::now();
        drop(lease);
        drop(turn);
        Ok(TimedSample {
            started,
            first,
            finished,
            prompt_tokens: 0,
            prefix_cache_hit_len: 0,
            units: work_units,
            worker: worker.index(),
            checksum: final_checksum,
            token_ids: Vec::new(),
        })
    })
    .await
    .context("synthetic request task panicked")?
}

async fn measure_synthetic_stateless(
    pool: &Arc<WorkerPool>,
    iterations: usize,
    concurrency: usize,
    work_units: usize,
) -> Result<Measurement> {
    let pool_for_jobs = Arc::clone(pool);
    let before_cpu = process_cpu_ticks();
    let started = Instant::now();
    let jobs = run_jobs(iterations, concurrency, move |_| {
        let pool = Arc::clone(&pool_for_jobs);
        async move { run_synthetic_request(&pool, None, work_units, Instant::now()).await }
    })
    .await?;
    let wall = started.elapsed();
    Ok(Measurement {
        samples: jobs.samples,
        errors: jobs.errors,
        wall,
        cpu_time: cpu_duration(before_cpu, process_cpu_ticks()),
        ..Measurement::default()
    })
}

async fn measure_synthetic_distinct(
    pool: &Arc<WorkerPool>,
    leases: &Arc<SessionLeases>,
    iterations: usize,
    concurrency: usize,
    work_units: usize,
) -> Result<Measurement> {
    let sessions = (0..iterations)
        .map(|index| create_synthetic_session(pool, index as u64))
        .collect::<Result<Vec<_>>>()?;
    let sessions = Arc::new(sessions);
    let sessions_for_jobs = Arc::clone(&sessions);
    let pool_for_jobs = Arc::clone(pool);
    let leases_for_jobs = Arc::clone(leases);
    let before_cpu = process_cpu_ticks();
    let started = Instant::now();
    let jobs = run_jobs(iterations, concurrency, move |index| {
        let pool = Arc::clone(&pool_for_jobs);
        let leases = Arc::clone(&leases_for_jobs);
        let session = sessions_for_jobs[index];
        async move {
            let lease = leases
                .acquire(synthetic_binding(session), &format!("synthetic-{index}"))
                .context("synthetic distinct lease")?;
            run_synthetic_request(&pool, Some((session, lease)), work_units, Instant::now()).await
        }
    })
    .await?;
    let wall = started.elapsed();
    let cpu_time = cpu_duration(before_cpu, process_cpu_ticks());
    let sessions = Arc::try_unwrap(sessions)
        .map_err(|_| anyhow::anyhow!("synthetic distinct session list retained"))?;
    for session in sessions {
        close_synthetic_session(pool, session)?;
    }
    Ok(Measurement {
        samples: jobs.samples,
        errors: jobs.errors,
        wall,
        cpu_time,
        ..Measurement::default()
    })
}

async fn measure_synthetic_serialized(
    pool: &Arc<WorkerPool>,
    leases: &Arc<SessionLeases>,
    iterations: usize,
    concurrency: usize,
    work_units: usize,
) -> Result<Measurement> {
    let session = create_synthetic_session(pool, 0)?;
    let serialization = Arc::new(AsyncMutex::new(()));
    let pool_for_jobs = Arc::clone(pool);
    let leases_for_jobs = Arc::clone(leases);
    let before_cpu = process_cpu_ticks();
    let started = Instant::now();
    let jobs = run_jobs(iterations, concurrency, move |_| {
        let pool = Arc::clone(&pool_for_jobs);
        let leases = Arc::clone(&leases_for_jobs);
        let serialization = Arc::clone(&serialization);
        let queued_at = Instant::now();
        async move {
            let _serial = serialization.lock().await;
            let lease = leases
                .acquire(synthetic_binding(session), "synthetic-serialized")
                .context("synthetic serialized lease")?;
            run_synthetic_request(&pool, Some((session, lease)), work_units, queued_at).await
        }
    })
    .await?;
    let wall = started.elapsed();
    let cpu_time = cpu_duration(before_cpu, process_cpu_ticks());
    close_synthetic_session(pool, session)?;
    Ok(Measurement {
        samples: jobs.samples,
        errors: jobs.errors,
        wall,
        cpu_time,
        ..Measurement::default()
    })
}

async fn measure_synthetic_conflicts(
    pool: &Arc<WorkerPool>,
    leases: &Arc<SessionLeases>,
    iterations: usize,
    concurrency: usize,
    work_units: usize,
) -> Result<Measurement> {
    let before_cpu = process_cpu_ticks();
    let wall_started = Instant::now();
    let mut samples = Vec::with_capacity(iterations);
    let mut conflict_ns = Vec::with_capacity(iterations * concurrency.saturating_sub(1));
    let mut typed_conflicts = 0;
    let mut errors = 0;
    for index in 0..iterations {
        let session = create_synthetic_session(pool, index as u64)?;
        let binding = synthetic_binding(session);
        let owner = leases
            .acquire(binding.clone(), "synthetic-conflict")
            .context("synthetic owner lease")?;
        let pool_for_owner = Arc::clone(pool);
        let pending = tokio::spawn(async move {
            run_synthetic_request(
                &pool_for_owner,
                Some((session, owner)),
                work_units,
                Instant::now(),
            )
            .await
        });
        for _ in 1..concurrency {
            let started = Instant::now();
            match leases.acquire(binding.clone(), "synthetic-conflict") {
                Err(PackageCapabilityError::ExclusiveLeaseConflict { .. }) => {
                    conflict_ns.push(duration_ns(started.elapsed()));
                    typed_conflicts += 1;
                }
                Err(_) => errors += 1,
                Ok(lease) => {
                    drop(lease);
                    errors += 1;
                }
            }
        }
        match pending.await {
            Ok(Ok(sample)) => samples.push(sample),
            Ok(Err(_)) | Err(_) => errors += 1,
        }
        if close_synthetic_session(pool, session).is_err() {
            errors += 1;
        }
    }
    let wall = wall_started.elapsed();
    Ok(Measurement {
        samples,
        conflict_ns,
        errors,
        typed_conflicts,
        wall,
        cpu_time: cpu_duration(before_cpu, process_cpu_ticks()),
        ..Measurement::default()
    })
}

async fn run_jobs<F, Fut>(jobs: usize, concurrency: usize, operation: F) -> Result<JobResults>
where
    F: Fn(usize) -> Fut + Send + Sync + 'static,
    Fut: Future<Output = Result<TimedSample>> + Send + 'static,
{
    let next = Arc::new(AtomicUsize::new(0));
    let operation = Arc::new(operation);
    let mut tasks = Vec::with_capacity(concurrency.min(jobs));
    for _ in 0..concurrency.min(jobs) {
        let next = Arc::clone(&next);
        let operation = Arc::clone(&operation);
        tasks.push(tokio::spawn(async move {
            let mut samples = Vec::new();
            let mut errors = 0;
            loop {
                let index = next.fetch_add(1, Ordering::Relaxed);
                if index >= jobs {
                    break;
                }
                match operation(index).await {
                    Ok(sample) => samples.push(sample),
                    Err(_) => errors += 1,
                }
            }
            JobResults { samples, errors }
        }));
    }
    let mut samples = Vec::with_capacity(jobs);
    let mut errors = 0;
    for task in tasks {
        match task.await {
            Ok(mut results) => {
                samples.append(&mut results.samples);
                errors += results.errors;
            }
            Err(_) => errors += 1,
        }
    }
    samples.sort_by_key(|sample| sample.started);
    Ok(JobResults { samples, errors })
}

fn verify_measurement(
    measurement: &Measurement,
    expected_completed: usize,
    expected_conflicts: usize,
) -> Result<()> {
    ensure!(
        measurement.samples.len() == expected_completed,
        "completion count drift: expected {expected_completed}, got {}",
        measurement.samples.len()
    );
    ensure!(measurement.errors == 0, "benchmark recorded errors");
    ensure!(
        measurement.conflict_ns.len() == expected_conflicts,
        "conflict count drift: expected {expected_conflicts}, got {}",
        measurement.conflict_ns.len()
    );
    ensure!(
        measurement.typed_conflicts == expected_conflicts,
        "typed conflict count drift: expected {expected_conflicts}, got {}",
        measurement.typed_conflicts
    );
    Ok(())
}

fn verify_zero_prefix_cache_hits(measurement: &Measurement, label: &str) -> Result<()> {
    let hit_requests = measurement
        .samples
        .iter()
        .filter(|sample| sample.prefix_cache_hit_len > 0)
        .count();
    let hit_tokens = measurement
        .samples
        .iter()
        .map(|sample| sample.prefix_cache_hit_len)
        .sum::<usize>();
    ensure!(
        hit_requests == 0 && hit_tokens == 0,
        "{label} reused cached prompt state in {hit_requests} requests ({hit_tokens} tokens); \
         cold benchmark requests must recompute the full prompt"
    );
    Ok(())
}

fn verify_equivalent_model_work(direct: &Measurement, workflow: &Measurement) -> Result<()> {
    ensure!(
        direct.samples.len() == workflow.samples.len(),
        "direct decode/workflow sample count differs: {} versus {}",
        direct.samples.len(),
        workflow.samples.len()
    );
    for (index, (direct, workflow)) in direct.samples.iter().zip(&workflow.samples).enumerate() {
        ensure!(
            direct.prompt_tokens == workflow.prompt_tokens
                && direct.prefix_cache_hit_len == 0
                && workflow.prefix_cache_hit_len == 0
                && direct.units == workflow.units
                && direct.token_ids == workflow.token_ids,
            "direct decode/workflow work differs at sample {index}: prompt={}/{}, cache_hit={}/{}, \
             units={}/{}, tokens={:?}/{:?}",
            direct.prompt_tokens,
            workflow.prompt_tokens,
            direct.prefix_cache_hit_len,
            workflow.prefix_cache_hit_len,
            direct.units,
            workflow.units,
            direct.token_ids,
            workflow.token_ids
        );
    }
    Ok(())
}

fn record_gate(gate: &mut Option<bool>, passed: bool) {
    *gate = Some(gate.unwrap_or(true) && passed);
}

fn record_result_gate(gate: &mut Option<bool>, result: Result<()>) {
    record_gate(gate, result.is_ok());
}

fn assert_synthetic_counters(pool: &WorkerPool, leases: &SessionLeases) -> Result<()> {
    ensure!(leases.held() == 0, "synthetic lease counter drift");
    for status in pool.statuses() {
        ensure!(
            status.active_turns == 0 && status.live_sessions == 0,
            "synthetic worker {} counter drift: active={}, sessions={}",
            status.id,
            status.active_turns,
            status.live_sessions
        );
    }
    Ok(())
}

fn assert_driver_counters(driver: &EngineDriver, leases: &SessionLeases) -> Result<()> {
    ensure!(leases.held() == 0, "driver lease counter drift");
    for status in driver.worker_statuses() {
        ensure!(
            status.worker.active_turns == 0 && status.worker.live_sessions == 0,
            "worker {} counter drift: active={}, sessions={}",
            status.worker.id,
            status.worker.active_turns,
            status.worker.live_sessions
        );
    }
    Ok(())
}

async fn governed_resources(driver: &EngineDriver) -> Option<GovernedResourcesReport> {
    driver
        .resource_snapshot()
        .await
        .ok()
        .map(|snapshot| GovernedResourcesReport {
            host: resource_tier_report(&snapshot.host_ram),
            disk: snapshot.disk_spill.as_ref().map(resource_tier_report),
            device: resource_tier_report(&snapshot.vram),
        })
}

fn resource_tier_report(snapshot: &onnx_genai::scheduler::TierSnapshot) -> ResourceTierReport {
    ResourceTierReport {
        used_bytes: snapshot.used,
        limit_bytes: snapshot.limit,
        headroom_bytes: snapshot.headroom,
    }
}

fn row_from_measurement(
    mode: &str,
    scenario: &str,
    worker_count: usize,
    concurrency: usize,
    warmups: usize,
    requested_iterations: usize,
    measurement: Measurement,
) -> BenchmarkRow {
    let ttft_us = measurement
        .samples
        .iter()
        .map(|sample| duration_us(sample.first.duration_since(sample.started)))
        .collect::<Vec<_>>();
    let steady_us = measurement
        .samples
        .iter()
        .map(|sample| duration_us(sample.finished.duration_since(sample.first)))
        .collect::<Vec<_>>();
    let total_us = measurement
        .samples
        .iter()
        .map(|sample| duration_us(sample.finished.duration_since(sample.started)))
        .collect::<Vec<_>>();
    let units = measurement
        .samples
        .iter()
        .map(|sample| sample.units)
        .sum::<usize>();
    let prompt_tokens_submitted = measurement
        .samples
        .iter()
        .map(|sample| sample.prompt_tokens)
        .sum::<usize>();
    let prefix_cache_hit_tokens = measurement
        .samples
        .iter()
        .map(|sample| sample.prefix_cache_hit_len)
        .sum::<usize>();
    let prefix_cache_hit_requests = measurement
        .samples
        .iter()
        .filter(|sample| sample.prefix_cache_hit_len > 0)
        .count();
    let prompt_tokens_computed = measurement
        .samples
        .iter()
        .map(|sample| {
            sample
                .prompt_tokens
                .saturating_sub(sample.prefix_cache_hit_len.min(sample.prompt_tokens))
        })
        .sum::<usize>();
    let max_prefix_cache_hit_len = measurement
        .samples
        .iter()
        .map(|sample| sample.prefix_cache_hit_len)
        .max()
        .unwrap_or(0);
    let steady_units = measurement
        .samples
        .iter()
        .map(|sample| sample.units.saturating_sub(1))
        .sum::<usize>();
    let steady_seconds = steady_us.iter().sum::<u64>() as f64 / 1_000_000.0;
    let wall_seconds = measurement.wall.as_secs_f64();
    let cpu_time_ms = measurement
        .cpu_time
        .map(|duration| duration.as_secs_f64() * 1_000.0);
    let cpu_utilization_percent = measurement.cpu_time.and_then(|duration| {
        (wall_seconds > 0.0).then_some(duration.as_secs_f64() / wall_seconds * 100.0)
    });
    let mut worker_completions = BTreeMap::new();
    for sample in &measurement.samples {
        *worker_completions.entry(sample.worker).or_insert(0) += 1;
    }
    let memory = process_memory();
    BenchmarkRow {
        mode: mode.to_string(),
        scenario: scenario.to_string(),
        worker_count,
        concurrency,
        warmups,
        requested_iterations,
        unit: if mode == "synthetic" {
            "fixed_work_iterations".to_string()
        } else {
            "generated_tokens".to_string()
        },
        prompt_tokens_per_request: measurement
            .samples
            .first()
            .and_then(|sample| (sample.prompt_tokens > 0).then_some(sample.prompt_tokens)),
        prompt_tokens_submitted,
        prompt_tokens_computed,
        prefix_cache_hit_requests,
        prefix_cache_hit_tokens,
        max_prefix_cache_hit_len,
        target_units_per_request: measurement.samples.first().map_or(0, |sample| sample.units),
        completed: measurement.samples.len(),
        errors: measurement.errors,
        conflicts: measurement.conflict_ns.len(),
        typed_conflicts: measurement.typed_conflicts,
        wall_ms: measurement.wall.as_secs_f64() * 1_000.0,
        wall_speedup_vs_w1: None,
        cpu_time_ms,
        cpu_utilization_percent,
        request_throughput_per_second: safe_rate(measurement.samples.len(), wall_seconds),
        unit_throughput_per_second: safe_rate(units, wall_seconds),
        steady_state_units_per_second: (steady_seconds > 0.0)
            .then_some(steady_units as f64 / steady_seconds),
        ttft_ms: summarize_us(&ttft_us),
        steady_state_latency_ms: summarize_us(&steady_us),
        total_latency_ms: summarize_us(&total_us),
        conflict_latency_ms: summarize_ns(&measurement.conflict_ns),
        max_client_stream_overlap: max_client_stream_overlap(&measurement.samples),
        worker_completions,
        rss_bytes_after: memory.rss,
        peak_rss_bytes_after: memory.peak_rss,
        governed_resources: measurement.governed_resources,
        raw: Some(RawSamples {
            ttft_us,
            steady_state_us: steady_us,
            total_us,
            conflict_ns: measurement.conflict_ns,
            prefix_cache_hit_len: measurement
                .samples
                .iter()
                .map(|sample| sample.prefix_cache_hit_len)
                .collect(),
        }),
    }
}

fn apply_w1_speedups(rows: &mut [BenchmarkRow]) {
    let baselines = rows
        .iter()
        .filter(|row| row.worker_count == 1)
        .map(|row| {
            (
                (row.mode.clone(), row.scenario.clone(), row.concurrency),
                row.wall_ms,
            )
        })
        .collect::<BTreeMap<_, _>>();
    for row in rows {
        if row.worker_count == 0 {
            continue;
        }
        row.wall_speedup_vs_w1 = baselines
            .get(&(row.mode.clone(), row.scenario.clone(), row.concurrency))
            .and_then(|baseline| (row.wall_ms > 0.0).then_some(*baseline / row.wall_ms));
    }
}

fn summarize_us(values: &[u64]) -> LatencySummary {
    summarize_scaled(values, 1_000.0)
}

fn summarize_ns(values: &[u64]) -> LatencySummary {
    summarize_scaled(values, 1_000_000.0)
}

fn summarize_scaled(values: &[u64], units_per_millisecond: f64) -> LatencySummary {
    if values.is_empty() {
        return LatencySummary::default();
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let milliseconds = |value: u64| value as f64 / units_per_millisecond;
    LatencySummary {
        count: sorted.len(),
        min: sorted.first().copied().map(milliseconds),
        mean: Some(
            sorted.iter().map(|value| milliseconds(*value)).sum::<f64>() / sorted.len() as f64,
        ),
        p50: Some(milliseconds(nearest_rank(&sorted, 0.50))),
        p95: Some(milliseconds(nearest_rank(&sorted, 0.95))),
        p99: Some(milliseconds(nearest_rank(&sorted, 0.99))),
        max: sorted.last().copied().map(milliseconds),
    }
}

fn nearest_rank(sorted: &[u64], quantile: f64) -> u64 {
    let rank = (quantile * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

fn max_client_stream_overlap(samples: &[TimedSample]) -> usize {
    let mut events = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        events.push((sample.first, true));
        events.push((sample.finished, false));
    }
    events.sort_by_key(|(instant, start)| (*instant, *start));
    let mut active = 0usize;
    let mut maximum = 0usize;
    for (_, start) in events {
        if start {
            active += 1;
            maximum = maximum.max(active);
        } else {
            active = active.saturating_sub(1);
        }
    }
    maximum
}

fn token_checksum(tokens: &[u32]) -> u64 {
    tokens.iter().fold(FIXED_SEED, |checksum, token| {
        checksum.rotate_left(7) ^ u64::from(*token)
    })
}

fn safe_rate(units: usize, seconds: f64) -> f64 {
    if seconds > 0.0 {
        units as f64 / seconds
    } else {
        0.0
    }
}

fn difference(lhs: Option<f64>, rhs: Option<f64>) -> Option<f64> {
    Some(lhs? - rhs?)
}

fn duration_us(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

#[derive(Clone, Copy)]
struct ProcessMemory {
    rss: Option<u64>,
    peak_rss: Option<u64>,
}

fn process_memory() -> ProcessMemory {
    let status = std::fs::read_to_string("/proc/self/status").ok();
    ProcessMemory {
        rss: status
            .as_deref()
            .and_then(|contents| parse_status_bytes(contents, "VmRSS:")),
        peak_rss: status
            .as_deref()
            .and_then(|contents| parse_status_bytes(contents, "VmHWM:")),
    }
}

fn process_status_value(field: &str) -> Option<String> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        line.strip_prefix(field)
            .map(|value| value.trim().to_string())
    })
}

fn parse_status_bytes(contents: &str, field: &str) -> Option<u64> {
    let line = contents.lines().find(|line| line.starts_with(field))?;
    let mut values = line[field.len()..].split_whitespace();
    let value = values.next()?.parse::<u64>().ok()?;
    match values.next() {
        Some("kB") | None => value.checked_mul(1024),
        Some("mB") => value.checked_mul(1024 * 1024),
        Some(_) => None,
    }
}

fn process_cpu_ticks() -> Option<u64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    let after_name = stat.rsplit_once(") ")?.1;
    let fields = after_name.split_whitespace().collect::<Vec<_>>();
    let user = fields.get(11)?.parse::<u64>().ok()?;
    let system = fields.get(12)?.parse::<u64>().ok()?;
    user.checked_add(system)
}

fn cpu_duration(before: Option<u64>, after: Option<u64>) -> Option<Duration> {
    let ticks = after?.checked_sub(before?)?;
    let ticks_per_second = command_text("getconf", &["CLK_TCK"])?.parse::<f64>().ok()?;
    (ticks_per_second > 0.0).then(|| Duration::from_secs_f64(ticks as f64 / ticks_per_second))
}

fn cpu_model() -> Option<String> {
    let contents = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    contents.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        matches!(key.trim(), "model name" | "Hardware").then(|| value.trim().to_string())
    })
}

fn host_memory_bytes() -> Option<u64> {
    let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
    parse_status_bytes(&contents, "MemTotal:")
}

fn gpu_report() -> Option<String> {
    command_text(
        "nvidia-smi",
        &[
            "--query-gpu=name,driver_version,memory.total,memory.used,memory.free",
            "--format=csv,noheader",
        ],
    )
}

fn command_text(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn path_text(path: &Path) -> &str {
    path.to_str().unwrap_or("<non-utf8-path>")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_correctness_gates_remain_serializable() {
        let invariants = InvariantReport {
            w1_output_parity: Some(false),
            typed_same_session_conflict: None,
            distinct_session_stream_overlap: Some(true),
            exact_completion_counts: Some(false),
            no_counter_drift: Some(true),
        };

        assert_eq!(
            failed_gate_names(&invariants),
            vec!["w1_output_parity", "exact_completion_counts"]
        );
        let json = serde_json::to_value(invariants).expect("serialize invariants");
        assert_eq!(json["w1_output_parity"], false);
        assert_eq!(json["typed_same_session_conflict"], serde_json::Value::Null);
        assert_eq!(json["exact_completion_counts"], false);
    }

    #[tokio::test]
    async fn job_failures_are_counted_instead_of_aborting_the_report() {
        let jobs = run_jobs(4, 2, |index| async move {
            if index % 2 == 1 {
                anyhow::bail!("injected job failure");
            }
            let now = Instant::now();
            Ok(TimedSample {
                started: now,
                first: now,
                finished: now,
                prompt_tokens: 1,
                prefix_cache_hit_len: 0,
                units: 1,
                worker: 0,
                checksum: index as u64,
                token_ids: vec![index as u32],
            })
        })
        .await
        .expect("job runner");

        let measurement = Measurement {
            samples: jobs.samples,
            errors: jobs.errors,
            ..Measurement::default()
        };
        assert_eq!(measurement.samples.len(), 2);
        assert_eq!(measurement.errors, 2);
        assert!(verify_measurement(&measurement, 4, 0).is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn synthetic_fixture_proves_concurrency_invariants_without_timing_thresholds() {
        let report = run(BenchmarkOptions {
            mode: BenchmarkMode::Synthetic { work_units: 20_000 },
            worker_counts: vec![1, 2],
            concurrency_levels: vec![1, 2],
            warmups: 1,
            iterations: 4,
            max_new_tokens: 2,
            include_raw_samples: true,
        })
        .await
        .expect("synthetic benchmark");

        assert_eq!(report.schema_version, 4);
        assert_eq!(report.invariants.w1_output_parity, Some(true));
        assert_eq!(report.invariants.typed_same_session_conflict, Some(true));
        assert_eq!(
            report.invariants.distinct_session_stream_overlap,
            Some(true)
        );
        assert_eq!(report.invariants.exact_completion_counts, Some(true));
        assert_eq!(report.invariants.no_counter_drift, Some(true));
        let conflict = report
            .rows
            .iter()
            .find(|row| {
                row.scenario == "same_session_conflict"
                    && row.worker_count == 2
                    && row.concurrency == 2
            })
            .expect("conflict row");
        assert_eq!(conflict.completed, 4);
        assert_eq!(conflict.typed_conflicts, 4);
        assert_eq!(conflict.errors, 0);
        assert_eq!(
            conflict
                .raw
                .as_ref()
                .expect("raw samples")
                .conflict_ns
                .len(),
            4
        );
        assert!(
            conflict
                .conflict_latency_ms
                .p50
                .is_some_and(|value| value > 0.0)
        );
        let distinct = report
            .rows
            .iter()
            .find(|row| {
                row.scenario == "distinct_sessions" && row.worker_count == 2 && row.concurrency == 2
            })
            .expect("distinct row");
        assert!(distinct.max_client_stream_overlap > 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn single_worker_single_request_matrix_serializes_untested_gates_as_null() {
        let report = run(BenchmarkOptions {
            mode: BenchmarkMode::Synthetic { work_units: 2_000 },
            worker_counts: vec![1],
            concurrency_levels: vec![1],
            warmups: 0,
            iterations: 2,
            max_new_tokens: 1,
            include_raw_samples: false,
        })
        .await
        .expect("synthetic benchmark");

        assert_eq!(report.invariants.typed_same_session_conflict, None);
        assert_eq!(report.invariants.distinct_session_stream_overlap, None);
        let json = serde_json::to_value(&report.invariants).expect("serialize invariants");
        assert!(json["typed_same_session_conflict"].is_null());
        assert!(json["distinct_session_stream_overlap"].is_null());
        assert_eq!(json["w1_output_parity"], true);
    }

    #[test]
    fn resource_report_preserves_used_limit_and_headroom_semantics() {
        let host = onnx_genai::scheduler::TierSnapshot {
            used: 8,
            limit: 32,
            headroom: 24,
        };
        let report = GovernedResourcesReport {
            host: resource_tier_report(&host),
            disk: Some(ResourceTierReport {
                used_bytes: 3,
                limit_bytes: 10,
                headroom_bytes: 7,
            }),
            device: ResourceTierReport {
                used_bytes: 5,
                limit_bytes: 20,
                headroom_bytes: 15,
            },
        };

        let json = serde_json::to_value(report).expect("serialize resources");
        assert_eq!(json["host"]["used_bytes"], 8);
        assert_eq!(json["host"]["limit_bytes"], 32);
        assert_eq!(json["host"]["headroom_bytes"], 24);
        assert_eq!(json["disk"]["used_bytes"], 3);
        assert_eq!(json["device"]["limit_bytes"], 20);
        assert!(json["host"].get("budget_bytes").is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ort_cold_benchmark_recomputes_equivalent_work_without_prefix_hits() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm");
        let report = run(BenchmarkOptions {
            mode: BenchmarkMode::Ort {
                model_dir,
                provider: "cpu".to_string(),
                intra_op_threads: 1,
            },
            worker_counts: vec![1],
            concurrency_levels: vec![1],
            warmups: 2,
            iterations: 35,
            max_new_tokens: 4,
            include_raw_samples: true,
        })
        .await
        .expect("ORT benchmark");

        for row in &report.rows {
            assert_eq!(
                row.prefix_cache_hit_requests, 0,
                "{} unexpectedly hit prefix cache",
                row.scenario
            );
            assert_eq!(row.prefix_cache_hit_tokens, 0);
            assert_eq!(row.max_prefix_cache_hit_len, 0);
            assert_eq!(row.prompt_tokens_computed, row.prompt_tokens_submitted);
            assert!(
                row.raw
                    .as_ref()
                    .expect("raw samples")
                    .prefix_cache_hit_len
                    .iter()
                    .all(|hit| *hit == 0)
            );
        }
        assert_eq!(report.invariants.w1_output_parity, Some(true));
        assert_eq!(report.invariants.typed_same_session_conflict, None);
        assert_eq!(report.invariants.distinct_session_stream_overlap, None);
    }
}

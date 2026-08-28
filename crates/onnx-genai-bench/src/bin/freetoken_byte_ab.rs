use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use anyhow::{Context, Result, bail};
use clap::Parser;
use onnx_genai_bench::freetoken_byte_ab::{
    AB_SCHEMA, ContractStatus, FreeTokenRunReport, NATIVE_CUDA_BINARY_MARKER, ResidencyArm,
    validate_pair,
};
use serde::Serialize;

#[derive(Debug, Parser)]
#[command(about = "Run paired native-CUDA FreeToken residency OFF/ON byte-accounting trials")]
struct Args {
    /// Native model directory accepted by profile_native.
    #[arg(long)]
    model: PathBuf,
    /// Native-cuda profile_native binary. Defaults to the sibling of this binary.
    #[arg(long)]
    profile_native: Option<PathBuf>,
    /// Stable combined JSON result.
    #[arg(long, default_value = "target/freetoken-byte-ab/report.json")]
    output: PathBuf,
    /// Persistent per-run JSON and stdout/stderr directory.
    #[arg(long, default_value = "target/freetoken-byte-ab/runs")]
    scratch_dir: PathBuf,
    #[arg(long, default_value = "Hello")]
    prompt: String,
    #[arg(long)]
    prompt_ids: Option<PathBuf>,
    #[arg(long, default_value_t = 128)]
    tokens: usize,
    #[arg(long, default_value_t = 8)]
    decode_skip: usize,
    /// Number of back-to-back OFF/ON pairs. At least 3 are required before
    /// wall clock is eligible as corroboration; deterministic counters remain
    /// valid with fewer trials.
    #[arg(long, default_value_t = 3)]
    trials: usize,
    /// Minimum in-process generation warm-up before each measured arm.
    #[arg(long, default_value_t = 8.0)]
    warmup_seconds: f64,
    /// Physical GPU index exported as CUDA_VISIBLE_DEVICES. The child uses
    /// ONNX_GENAI_CUDA_DEVICE=0 within that visibility mask.
    #[arg(long, default_value_t = 0)]
    device: u32,
    /// Runtime policy gate compared as OFF=off-value and ON=on-value.
    #[arg(
        long,
        default_value = "ONNX_GENAI_WEIGHT_OFFLOAD_COARSE_RESIDENCY_ENABLE"
    )]
    policy_env: String,
    #[arg(long, default_value = "0")]
    off_value: String,
    #[arg(long, default_value = "1")]
    on_value: String,
    /// Optional explicit weight-residency cache budget.
    #[arg(long)]
    device_budget_bytes: Option<u64>,
    /// Engine managed VRAM limit. `auto` resolves from the selected device.
    #[arg(long, default_value = "auto")]
    vram_limit: String,
}

#[derive(Debug, Serialize)]
struct IdleProbe {
    command_succeeded: bool,
    gpu_uuid: Option<String>,
    gpu_name: Option<String>,
    utilization_percent: Option<u32>,
    memory_used_mib: Option<u64>,
    sm_clock_mhz: Option<u32>,
    power_watts: Option<f64>,
    compute_processes: Vec<String>,
    exclusive_idle: bool,
    is_a100: bool,
    detail: Option<String>,
}

#[derive(Debug, Serialize)]
struct Trial {
    pair_index: usize,
    arm: ResidencyArm,
    idle_before: IdleProbe,
    run_report_path: String,
    stdout_log_path: String,
    stderr_log_path: String,
    report: FreeTokenRunReport,
}

#[derive(Debug, Serialize)]
struct Distribution {
    samples: Vec<f64>,
    median: f64,
    min: f64,
    max: f64,
}

impl Distribution {
    fn from_samples(mut samples: Vec<f64>) -> Option<Self> {
        if samples.is_empty() || samples.iter().any(|sample| !sample.is_finite()) {
            return None;
        }
        samples.sort_by(f64::total_cmp);
        let middle = samples.len() / 2;
        let median = if samples.len().is_multiple_of(2) {
            (samples[middle - 1] + samples[middle]) / 2.0
        } else {
            samples[middle]
        };
        Some(Self {
            median,
            min: samples[0],
            max: samples[samples.len() - 1],
            samples,
        })
    }
}

#[derive(Debug, Serialize)]
struct ArmSummary {
    weight_h2d_bytes_per_emitted_token: Option<Distribution>,
    weight_host_link_bytes_per_emitted_token: Option<Distribution>,
    weight_page_ins_per_emitted_token: Option<Distribution>,
    weight_vram_byte_hit_rate: Option<Distribution>,
    corroborative_decode_tokens_per_second: Option<Distribution>,
}

#[derive(Debug, Serialize)]
struct Comparison {
    weight_h2d_bytes_per_token_on_minus_off: Option<f64>,
    weight_h2d_bytes_per_token_on_over_off: Option<f64>,
    expert_weight_movement_claim_eligible: bool,
    expert_weight_movement_claim_blocker: String,
}

#[derive(Debug, Serialize)]
struct Conditions {
    model_path: String,
    prompt: String,
    prompt_ids_path: Option<String>,
    requested_output_tokens: usize,
    decode_skip_tokens: usize,
    paired_trials: usize,
    warmup_seconds_per_arm: f64,
    physical_cuda_device: u32,
    policy_environment_variable: String,
    off_value: String,
    on_value: String,
    weight_offload_enabled_for_both_arms: bool,
    device_budget_bytes: Option<u64>,
    vram_limit: String,
    wall_clock_is_corroborative_only: bool,
}

#[derive(Debug, Serialize)]
struct AbReport {
    schema: String,
    native_cuda_binary_marker: String,
    profile_native_binary: String,
    conditions: Conditions,
    wall_clock_eligible: bool,
    wall_clock_ineligibility_reason: Option<String>,
    off: ArmSummary,
    on: ArmSummary,
    comparison: Comparison,
    trials: Vec<Trial>,
    contract: ContractStatus,
}

fn sibling_profile_native() -> Result<PathBuf> {
    let exe = std::env::current_exe().context("resolve freetoken_byte_ab executable")?;
    Ok(exe
        .parent()
        .context("freetoken_byte_ab executable has no parent directory")?
        .join(format!("profile_native{}", std::env::consts::EXE_SUFFIX)))
}

fn binary_contains_marker(path: &Path) -> Result<bool> {
    let file = File::open(path)
        .with_context(|| format!("open profile_native binary {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let needle = NATIVE_CUDA_BINARY_MARKER.as_bytes();
    let mut carry = Vec::new();
    let mut chunk = vec![0u8; 64 * 1024];
    loop {
        let read = reader
            .read(&mut chunk)
            .with_context(|| format!("read profile_native binary {}", path.display()))?;
        if read == 0 {
            return Ok(false);
        }
        carry.extend_from_slice(&chunk[..read]);
        if carry.windows(needle.len()).any(|window| window == needle) {
            return Ok(true);
        }
        if carry.len() >= needle.len() {
            carry.drain(..carry.len() - (needle.len() - 1));
        }
    }
}

fn parse_csv_field<T: std::str::FromStr>(fields: &[&str], index: usize) -> Option<T> {
    fields.get(index)?.trim().parse().ok()
}

fn idle_probe(device: u32) -> IdleProbe {
    let gpu = Command::new("nvidia-smi")
        .args([
            "-i",
            &device.to_string(),
            "--query-gpu=uuid,name,utilization.gpu,memory.used,clocks.sm,power.draw",
            "--format=csv,noheader,nounits",
        ])
        .output();
    let processes = Command::new("nvidia-smi")
        .args([
            "-i",
            &device.to_string(),
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output();
    let (Ok(gpu), Ok(processes)) = (gpu, processes) else {
        return IdleProbe {
            command_succeeded: false,
            gpu_uuid: None,
            gpu_name: None,
            utilization_percent: None,
            memory_used_mib: None,
            sm_clock_mhz: None,
            power_watts: None,
            compute_processes: Vec::new(),
            exclusive_idle: false,
            is_a100: false,
            detail: Some("nvidia-smi invocation failed; throughput is ineligible".to_string()),
        };
    };
    if !gpu.status.success() || !processes.status.success() {
        return IdleProbe {
            command_succeeded: false,
            gpu_uuid: None,
            gpu_name: None,
            utilization_percent: None,
            memory_used_mib: None,
            sm_clock_mhz: None,
            power_watts: None,
            compute_processes: Vec::new(),
            exclusive_idle: false,
            is_a100: false,
            detail: Some(format!(
                "nvidia-smi failed: gpu_status={} process_status={}",
                gpu.status, processes.status
            )),
        };
    }
    let gpu_line = String::from_utf8_lossy(&gpu.stdout).trim().to_string();
    let fields = gpu_line.split(',').collect::<Vec<_>>();
    let compute_processes = String::from_utf8_lossy(&processes.stdout)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.contains("No running processes found"))
        .map(str::to_string)
        .collect::<Vec<_>>();
    let gpu_name = fields.get(1).map(|field| field.trim().to_string());
    let is_a100 = gpu_name
        .as_deref()
        .is_some_and(|name| name.to_ascii_uppercase().contains("A100"));
    let utilization_percent = parse_csv_field::<u32>(&fields, 2);
    let command_succeeded = fields.len() == 6;
    let exclusive_idle =
        command_succeeded && compute_processes.is_empty() && utilization_percent == Some(0);
    IdleProbe {
        command_succeeded,
        gpu_uuid: fields.first().map(|field| field.trim().to_string()),
        gpu_name,
        utilization_percent,
        memory_used_mib: parse_csv_field(&fields, 3),
        sm_clock_mhz: parse_csv_field(&fields, 4),
        power_watts: parse_csv_field(&fields, 5),
        compute_processes,
        exclusive_idle,
        is_a100,
        detail: (!exclusive_idle).then_some(
            "GPU had a compute process, nonzero sampled utilization, or incomplete telemetry; \
             counters remain usable but wall clock is excluded"
                .to_string(),
        ),
    }
}

fn write_output(
    path: &Path,
    output: &Output,
    stdout_path: &Path,
    stderr_path: &Path,
) -> Result<()> {
    std::fs::write(stdout_path, &output.stdout)
        .with_context(|| format!("write child stdout {}", stdout_path.display()))?;
    std::fs::write(stderr_path, &output.stderr)
        .with_context(|| format!("write child stderr {}", stderr_path.display()))?;
    if !output.status.success() {
        bail!(
            "profile_native failed for {} (status {}); see {} and {}",
            path.display(),
            output.status,
            stdout_path.display(),
            stderr_path.display()
        );
    }
    Ok(())
}

fn run_arm(
    args: &Args,
    profile_native: &Path,
    pair_index: usize,
    arm: ResidencyArm,
) -> Result<Trial> {
    let idle_before = idle_probe(args.device);
    eprintln!(
        "freetoken_byte_ab: pair={} arm={} idle={} utilization={:?} processes={}",
        pair_index,
        arm.as_str(),
        idle_before.exclusive_idle,
        idle_before.utilization_percent,
        idle_before.compute_processes.len()
    );
    let stem = format!("pair-{pair_index:02}-{}", arm.as_str());
    let report_path = args.scratch_dir.join(format!("{stem}.json"));
    let stdout_path = args.scratch_dir.join(format!("{stem}.stdout.log"));
    let stderr_path = args.scratch_dir.join(format!("{stem}.stderr.log"));
    let policy_value = match arm {
        ResidencyArm::Off => &args.off_value,
        ResidencyArm::On => &args.on_value,
    };

    let mut command = Command::new(profile_native);
    command
        .env("CUDA_VISIBLE_DEVICES", args.device.to_string())
        .env("ONNX_GENAI_CUDA_DEVICE", "0")
        .env("ONNX_GENAI_EP", "cuda")
        .env("ONNX_GENAI_CUDA_GRAPH", "1")
        .env("ONNX_GENAI_WEIGHT_OFFLOAD", "1")
        .env("ONNX_GENAI_VRAM_LIMIT", &args.vram_limit)
        .env(&args.policy_env, policy_value)
        .args([
            "--model",
            args.model
                .to_str()
                .context("--model path is not valid UTF-8")?,
            "--ep",
            "cuda",
            "--backend",
            "native",
            "--steady",
            "--no-prefix-cache",
            "--tokens",
            &args.tokens.to_string(),
            "--decode-skip",
            &args.decode_skip.to_string(),
            "--warmups",
            "0",
            "--warmup-seconds",
            &args.warmup_seconds.to_string(),
            "--runs",
            "1",
            "--freetoken-report-json",
            report_path
                .to_str()
                .context("report path is not valid UTF-8")?,
            "--freetoken-arm",
            arm.as_str(),
            "--freetoken-policy-env",
            &args.policy_env,
        ]);
    if let Some(prompt_ids) = args.prompt_ids.as_ref() {
        command.args([
            "--prompt-ids",
            prompt_ids
                .to_str()
                .context("--prompt-ids path is not valid UTF-8")?,
        ]);
    } else {
        command.args(["--prompt", &args.prompt]);
    }
    if let Some(device_budget_bytes) = args.device_budget_bytes {
        command.env(
            "ONNX_GENAI_WEIGHT_OFFLOAD_DEVICE_BYTES",
            device_budget_bytes.to_string(),
        );
    }

    let output = command
        .output()
        .with_context(|| format!("run profile_native {}", profile_native.display()))?;
    write_output(&report_path, &output, &stdout_path, &stderr_path)?;
    let report: FreeTokenRunReport = serde_json::from_slice(
        &std::fs::read(&report_path)
            .with_context(|| format!("read child report {}", report_path.display()))?,
    )
    .with_context(|| format!("parse child report {}", report_path.display()))?;
    Ok(Trial {
        pair_index,
        arm,
        idle_before,
        run_report_path: report_path.display().to_string(),
        stdout_log_path: stdout_path.display().to_string(),
        stderr_log_path: stderr_path.display().to_string(),
        report,
    })
}

fn metric_values(
    trials: &[Trial],
    arm: ResidencyArm,
    get: impl Fn(&FreeTokenRunReport) -> Option<f64>,
) -> Vec<f64> {
    trials
        .iter()
        .filter(|trial| trial.arm == arm)
        .filter_map(|trial| get(&trial.report))
        .collect()
}

fn summary(trials: &[Trial], arm: ResidencyArm, include_wall_clock: bool) -> ArmSummary {
    let h2d = metric_values(trials, arm, |report| {
        report.metrics.weight_h2d_bytes_per_emitted_token.value
    });
    let host_link = metric_values(trials, arm, |report| {
        report
            .metrics
            .weight_host_link_bytes_per_emitted_token
            .value
    });
    let page_ins = metric_values(trials, arm, |report| {
        report
            .metrics
            .weight_page_ins
            .value
            .map(|count| count as f64 / report.emitted_tokens.max(1) as f64)
    });
    let hit_rate = metric_values(trials, arm, |report| {
        report.metrics.weight_vram_byte_hit_rate.value
    });
    let wall_clock = include_wall_clock.then(|| {
        metric_values(trials, arm, |report| {
            report
                .metrics
                .wall_clock_decode_tokens_per_second
                .value
                .as_ref()
                .and_then(|values| values.first().copied())
        })
    });
    ArmSummary {
        weight_h2d_bytes_per_emitted_token: Distribution::from_samples(h2d),
        weight_host_link_bytes_per_emitted_token: Distribution::from_samples(host_link),
        weight_page_ins_per_emitted_token: Distribution::from_samples(page_ins),
        weight_vram_byte_hit_rate: Distribution::from_samples(hit_rate),
        corroborative_decode_tokens_per_second: wall_clock.and_then(Distribution::from_samples),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.trials == 0 {
        bail!("--trials must be greater than zero");
    }
    if args.tokens <= args.decode_skip {
        bail!("--tokens must be greater than --decode-skip");
    }
    if !args.warmup_seconds.is_finite() || args.warmup_seconds < 0.0 {
        bail!("--warmup-seconds must be finite and non-negative");
    }
    if args.policy_env.is_empty()
        || args.policy_env.contains('=')
        || args.off_value == args.on_value
    {
        bail!("policy environment variable must be valid and OFF/ON values must differ");
    }
    let profile_native = args
        .profile_native
        .clone()
        .map_or_else(sibling_profile_native, Ok)?;
    if !profile_native.is_file() {
        bail!(
            "profile_native binary is absent at {}; build both binaries with \
             `cargo build --release -p onnx-genai-bench --features native-cuda \
             --bin profile_native --bin freetoken_byte_ab`",
            profile_native.display()
        );
    }
    if !binary_contains_marker(&profile_native)? {
        bail!(
            "{} does not contain the required native-cuda marker {}; \
             ort-cuda and bench-native-only binaries are rejected",
            profile_native.display(),
            NATIVE_CUDA_BINARY_MARKER
        );
    }
    std::fs::create_dir_all(&args.scratch_dir)
        .with_context(|| format!("create scratch directory {}", args.scratch_dir.display()))?;

    let mut trials = Vec::with_capacity(args.trials * 2);
    let mut errors = Vec::new();
    let mut reference_tokens: Option<Vec<u32>> = None;
    for pair_index in 1..=args.trials {
        let off = run_arm(&args, &profile_native, pair_index, ResidencyArm::Off)?;
        let on = run_arm(&args, &profile_native, pair_index, ResidencyArm::On)?;
        if off.report.policy_control.value != args.off_value {
            errors.push(format!(
                "pair {pair_index}: OFF process observed policy value {:?}, expected {:?}",
                off.report.policy_control.value, args.off_value
            ));
        }
        if on.report.policy_control.value != args.on_value {
            errors.push(format!(
                "pair {pair_index}: ON process observed policy value {:?}, expected {:?}",
                on.report.policy_control.value, args.on_value
            ));
        }
        for error in validate_pair(&off.report, &on.report) {
            errors.push(format!("pair {pair_index}: {error}"));
        }
        for report in [&off.report, &on.report] {
            if let Some(reference) = &reference_tokens {
                if reference != &report.generated_token_ids {
                    errors.push(format!(
                        "pair {pair_index} arm {} differs from the first measured token stream",
                        report.arm.as_str()
                    ));
                }
            } else {
                reference_tokens = Some(report.generated_token_ids.clone());
            }
        }
        trials.push(off);
        trials.push(on);
    }

    let all_idle = trials.iter().all(|trial| trial.idle_before.exclusive_idle);
    let all_a100 = trials.iter().all(|trial| trial.idle_before.is_a100);
    let all_warm = trials
        .iter()
        .all(|trial| trial.report.warmup.actual_seconds >= 8.0);
    let wall_clock_eligible = args.trials >= 3 && all_warm && all_idle && all_a100;
    let wall_clock_ineligibility_reason = (!wall_clock_eligible).then(|| {
        let mut reasons = Vec::new();
        if args.trials < 3 {
            reasons.push(format!("paired trials {} < 3", args.trials));
        }
        if !all_warm {
            reasons.push("at least one arm completed less than 8 s of warm-up".to_string());
        }
        if !all_idle {
            reasons.push("at least one pre-run idle probe was not exclusively idle".to_string());
        }
        if !all_a100 {
            reasons.push("at least one arm did not identify an NVIDIA A100".to_string());
        }
        reasons.join("; ")
    });
    let off = summary(&trials, ResidencyArm::Off, wall_clock_eligible);
    let on = summary(&trials, ResidencyArm::On, wall_clock_eligible);
    let off_h2d = off
        .weight_h2d_bytes_per_emitted_token
        .as_ref()
        .map(|distribution| distribution.median);
    let on_h2d = on
        .weight_h2d_bytes_per_emitted_token
        .as_ref()
        .map(|distribution| distribution.median);
    let expert_metrics_available = trials.iter().all(|trial| {
        trial
            .report
            .metrics
            .selected_expert_logical_bytes
            .value
            .is_some()
            && trial
                .report
                .metrics
                .gpu_resident_expert_hit_bytes
                .value
                .is_some()
            && trial
                .report
                .metrics
                .host_to_device_expert_page_in_bytes
                .value
                .is_some()
            && trial.report.metrics.cpu_served_expert_bytes.value.is_some()
            && trial.report.metrics.expert_page_ins.value.is_some()
            && trial.report.metrics.expert_byte_hit_rate.value.is_some()
            && trial
                .report
                .metrics
                .expert_bytes_per_emitted_token
                .value
                .is_some()
    });
    let comparison = Comparison {
        weight_h2d_bytes_per_token_on_minus_off: off_h2d.zip(on_h2d).map(|(off, on)| on - off),
        weight_h2d_bytes_per_token_on_over_off: off_h2d
            .zip(on_h2d)
            .and_then(|(off, on)| (off > 0.0).then_some(on / off)),
        expert_weight_movement_claim_eligible: expert_metrics_available,
        expert_weight_movement_claim_blocker: if expert_metrics_available {
            "none; reports contain route-attributed expert logical, GPU-hit, H2D, CPU-served, \
             page-in, hit-rate, and bytes/token metrics"
                .to_string()
        } else {
            "complete route-attributed expert byte/page-in/bytes-token counters are unavailable; \
             total lazy-weight H2D deltas must not be labeled measured expert or physical HBM \
             traffic"
                .to_string()
        },
    };
    let contract = ContractStatus {
        passed: errors.is_empty(),
        errors,
    };
    let report = AbReport {
        schema: AB_SCHEMA.to_string(),
        native_cuda_binary_marker: NATIVE_CUDA_BINARY_MARKER.to_string(),
        profile_native_binary: profile_native.display().to_string(),
        conditions: Conditions {
            model_path: args.model.display().to_string(),
            prompt: args.prompt,
            prompt_ids_path: args.prompt_ids.map(|path| path.display().to_string()),
            requested_output_tokens: args.tokens,
            decode_skip_tokens: args.decode_skip,
            paired_trials: args.trials,
            warmup_seconds_per_arm: args.warmup_seconds,
            physical_cuda_device: args.device,
            policy_environment_variable: args.policy_env,
            off_value: args.off_value,
            on_value: args.on_value,
            weight_offload_enabled_for_both_arms: true,
            device_budget_bytes: args.device_budget_bytes,
            vram_limit: args.vram_limit,
            wall_clock_is_corroborative_only: true,
        },
        wall_clock_eligible,
        wall_clock_ineligibility_reason,
        off,
        on,
        comparison,
        trials,
        contract,
    };
    if let Some(parent) = args
        .output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create output directory {}", parent.display()))?;
    }
    std::fs::write(
        &args.output,
        serde_json::to_vec_pretty(&report).context("serialize combined A/B report")?,
    )
    .with_context(|| format!("write combined A/B report {}", args.output.display()))?;
    println!(
        "freetoken_byte_ab: report={} contract_passed={} wall_clock_eligible={}",
        args.output.display(),
        report.contract.passed,
        report.wall_clock_eligible
    );
    if !report.contract.passed {
        bail!(
            "FreeToken OFF/ON contract failed: {}",
            report.contract.errors.join("; ")
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distribution_reports_median_and_full_range() {
        let distribution =
            Distribution::from_samples(vec![4.0, 1.0, 3.0]).expect("three finite samples");
        assert_eq!(distribution.samples, vec![1.0, 3.0, 4.0]);
        assert_eq!(distribution.median, 3.0);
        assert_eq!(distribution.min, 1.0);
        assert_eq!(distribution.max, 4.0);
        assert_eq!(
            Distribution::from_samples(vec![1.0, 3.0])
                .expect("two finite samples")
                .median,
            2.0
        );
        assert!(Distribution::from_samples(vec![f64::NAN]).is_none());
    }

    #[test]
    fn binary_marker_probe_is_non_vacuous() {
        let dir =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/freetoken-byte-ab-tests");
        std::fs::create_dir_all(&dir).expect("create marker test directory");
        let absent = dir.join("marker-absent.bin");
        let present = dir.join("marker-present.bin");
        std::fs::write(&absent, b"not the marker").expect("write absent fixture");
        std::fs::write(
            &present,
            format!("prefix-{NATIVE_CUDA_BINARY_MARKER}-suffix"),
        )
        .expect("write present fixture");
        assert!(!binary_contains_marker(&absent).expect("scan absent fixture"));
        assert!(binary_contains_marker(&present).expect("scan present fixture"));
    }
}

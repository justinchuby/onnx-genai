use clap::ValueEnum;
use serde::{Deserialize, Serialize};

pub const RUN_SCHEMA: &str = "onnx-genai.freetoken-byte-ab.run.v1";
pub const AB_SCHEMA: &str = "onnx-genai.freetoken-byte-ab.comparison.v1";
pub const NATIVE_CUDA_BINARY_MARKER: &str = "ONNX_GENAI_FREETOKEN_BYTE_AB_NATIVE_CUDA_V1_7F31A9D2";

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum ResidencyArm {
    Off,
    On,
}

impl ResidencyArm {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::On => "on",
        }
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Metric<T> {
    pub value: Option<T>,
    pub unit: String,
    pub accounting_boundary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unavailable_reason: Option<String>,
}

impl<T> Metric<T> {
    pub fn available(
        value: T,
        unit: impl Into<String>,
        accounting_boundary: impl Into<String>,
    ) -> Self {
        Self {
            value: Some(value),
            unit: unit.into(),
            accounting_boundary: accounting_boundary.into(),
            unavailable_reason: None,
        }
    }

    pub fn unavailable(
        unit: impl Into<String>,
        accounting_boundary: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            value: None,
            unit: unit.into(),
            accounting_boundary: accounting_boundary.into(),
            unavailable_reason: Some(reason.into()),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PolicyControl {
    pub environment_variable: String,
    pub value: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct WarmupRecord {
    pub requested_seconds: f64,
    pub actual_seconds: f64,
    pub generations: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FreeTokenMetrics {
    pub selected_expert_logical_bytes: Metric<u64>,
    pub gpu_resident_expert_hit_bytes: Metric<u64>,
    pub host_to_device_expert_page_in_bytes: Metric<u64>,
    pub cpu_served_expert_bytes: Metric<u64>,
    pub expert_page_ins: Metric<u64>,
    pub expert_byte_hit_rate: Metric<f64>,
    pub expert_bytes_per_emitted_token: Metric<f64>,
    pub prefill_expert_bytes_by_layer: Metric<Vec<u64>>,
    pub decode_expert_bytes_by_layer: Metric<Vec<u64>>,

    pub model_weight_layout_bytes: Metric<u64>,
    pub weight_gpu_resident_hit_bytes: Metric<u64>,
    pub weight_h2d_accounted_bytes: Metric<u64>,
    pub weight_zero_copy_host_read_bytes: Metric<u64>,
    pub weight_page_ins: Metric<u64>,
    pub weight_cache_hits: Metric<u64>,
    pub weight_vram_byte_hit_rate: Metric<f64>,
    pub weight_h2d_bytes_per_emitted_token: Metric<f64>,
    pub weight_host_link_bytes_per_emitted_token: Metric<f64>,

    pub cuda_graph_captures: Metric<u64>,
    pub cuda_graph_replays: Metric<u64>,
    pub cuda_graph_fallbacks: Metric<u64>,
    pub measured_cuda_graph_captures: Metric<u64>,
    pub measured_cuda_graph_replays: Metric<u64>,
    pub measured_cuda_graph_fallbacks: Metric<u64>,

    pub peak_committed_physical_bytes: Metric<u64>,
    pub managed_limit_bytes: Metric<u64>,
    pub oversubscribed_bytes: Metric<u64>,
    pub ref_underflows: Metric<u64>,
    pub byte_underflows: Metric<u64>,
    pub unaccounted_committed_bytes: Metric<u64>,

    pub wall_clock_prefill_milliseconds: Metric<Vec<f64>>,
    pub wall_clock_decode_milliseconds_per_token: Metric<Vec<f64>>,
    pub wall_clock_decode_tokens_per_second: Metric<Vec<f64>>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ContractStatus {
    pub passed: bool,
    pub errors: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FreeTokenRunReport {
    pub schema: String,
    pub binary_marker: String,
    pub backend: String,
    pub arm: ResidencyArm,
    pub policy_control: PolicyControl,
    pub model_path: String,
    pub prompt_token_ids: Vec<u32>,
    pub requested_output_tokens: u64,
    pub emitted_tokens: u64,
    pub measured_generations: u64,
    pub decode_skip_tokens: u64,
    pub generated_token_ids: Vec<u32>,
    pub measurement_scope: String,
    pub warmup: WarmupRecord,
    pub metrics: FreeTokenMetrics,
    pub contract: ContractStatus,
}

impl FreeTokenRunReport {
    pub fn refresh_contract(&mut self) {
        let errors = validate_run(self);
        self.contract = ContractStatus {
            passed: errors.is_empty(),
            errors,
        };
    }
}

fn required_u64(errors: &mut Vec<String>, name: &str, metric: &Metric<u64>) -> Option<u64> {
    match metric.value {
        Some(value) => Some(value),
        None => {
            errors.push(format!(
                "{name} is unavailable: {}",
                metric.unavailable_reason.as_deref().unwrap_or("no reason")
            ));
            None
        }
    }
}

pub fn validate_run(report: &FreeTokenRunReport) -> Vec<String> {
    let mut errors = Vec::new();
    if report.schema != RUN_SCHEMA {
        errors.push(format!(
            "schema must be {RUN_SCHEMA}, got {}",
            report.schema
        ));
    }
    if report.binary_marker != NATIVE_CUDA_BINARY_MARKER {
        errors.push("native-cuda binary marker is absent or wrong".to_string());
    }
    if report.backend != "native-cuda" {
        errors.push(format!(
            "backend must be native-cuda; ort-cuda and other backends are rejected (got {})",
            report.backend
        ));
    }
    if report.policy_control.environment_variable.is_empty()
        || report.policy_control.value.is_empty()
    {
        errors
            .push("OFF/ON policy control must name an environment variable and value".to_string());
    }
    if report.prompt_token_ids.is_empty() {
        errors.push("prompt token ids must be recorded".to_string());
    }
    if report.generated_token_ids.is_empty() {
        errors.push("generated token ids must be recorded".to_string());
    }
    if report.measured_generations == 0 {
        errors.push("measured_generations must be greater than zero".to_string());
    }
    if report.requested_output_tokens <= report.decode_skip_tokens {
        errors.push(format!(
            "requested_output_tokens {} must exceed decode_skip_tokens {}",
            report.requested_output_tokens, report.decode_skip_tokens
        ));
    }
    if !report.warmup.requested_seconds.is_finite()
        || !report.warmup.actual_seconds.is_finite()
        || report.warmup.requested_seconds < 0.0
        || report.warmup.actual_seconds < report.warmup.requested_seconds
    {
        errors.push(format!(
            "warm-up is invalid: requested={} actual={}",
            report.warmup.requested_seconds, report.warmup.actual_seconds
        ));
    }
    let expected_emitted = report
        .requested_output_tokens
        .saturating_mul(report.measured_generations);
    if report.emitted_tokens != expected_emitted {
        errors.push(format!(
            "emitted token count {} does not equal requested_output_tokens {} * measured_generations {}",
            report.emitted_tokens, report.requested_output_tokens, report.measured_generations
        ));
    }
    if report.generated_token_ids.len() as u64 != report.requested_output_tokens {
        errors.push(format!(
            "recorded token stream has {} ids, expected {}",
            report.generated_token_ids.len(),
            report.requested_output_tokens
        ));
    }

    if let Some(captures) = required_u64(
        &mut errors,
        "cuda_graph_captures",
        &report.metrics.cuda_graph_captures,
    ) && captures == 0
    {
        errors.push("cuda_graph_captures must be greater than zero".to_string());
    }
    if let Some(fallbacks) = required_u64(
        &mut errors,
        "cuda_graph_fallbacks",
        &report.metrics.cuda_graph_fallbacks,
    ) && fallbacks != 0
    {
        errors.push(format!(
            "cuda_graph_fallbacks must be zero, got {fallbacks}"
        ));
    }
    for (name, metric) in [
        ("oversubscribed_bytes", &report.metrics.oversubscribed_bytes),
        ("ref_underflows", &report.metrics.ref_underflows),
        ("byte_underflows", &report.metrics.byte_underflows),
        (
            "unaccounted_committed_bytes",
            &report.metrics.unaccounted_committed_bytes,
        ),
    ] {
        if let Some(value) = required_u64(&mut errors, name, metric)
            && value != 0
        {
            errors.push(format!("{name} must be zero, got {value}"));
        }
    }
    let peak = required_u64(
        &mut errors,
        "peak_committed_physical_bytes",
        &report.metrics.peak_committed_physical_bytes,
    );
    let limit = required_u64(
        &mut errors,
        "managed_limit_bytes",
        &report.metrics.managed_limit_bytes,
    );
    if let (Some(peak), Some(limit)) = (peak, limit)
        && peak > limit
    {
        errors.push(format!(
            "peak_committed_physical_bytes must not exceed managed_limit_bytes, got {peak} > {limit}"
        ));
    }
    errors
}

pub fn validate_pair(off: &FreeTokenRunReport, on: &FreeTokenRunReport) -> Vec<String> {
    let mut errors = Vec::new();
    if off.arm != ResidencyArm::Off {
        errors.push(format!("first report must be OFF, got {:?}", off.arm));
    }
    if on.arm != ResidencyArm::On {
        errors.push(format!("second report must be ON, got {:?}", on.arm));
    }
    for (label, report) in [("off", off), ("on", on)] {
        for error in validate_run(report) {
            errors.push(format!("{label}: {error}"));
        }
    }
    if off.model_path != on.model_path {
        errors.push("OFF/ON model paths differ".to_string());
    }
    if off.prompt_token_ids != on.prompt_token_ids {
        errors.push("OFF/ON prompt token ids differ".to_string());
    }
    if off.requested_output_tokens != on.requested_output_tokens {
        errors.push("OFF/ON requested token counts differ".to_string());
    }
    if off.measured_generations != on.measured_generations {
        errors.push("OFF/ON measured generation counts differ".to_string());
    }
    if off.decode_skip_tokens != on.decode_skip_tokens {
        errors.push("OFF/ON decode-skip counts differ".to_string());
    }
    if off.measurement_scope != on.measurement_scope {
        errors.push("OFF/ON measurement scopes differ".to_string());
    }
    if off.warmup.requested_seconds.to_bits() != on.warmup.requested_seconds.to_bits() {
        errors.push("OFF/ON requested warm-up durations differ".to_string());
    }
    if off.generated_token_ids != on.generated_token_ids {
        errors.push("OFF/ON generated token ids are not byte-identical".to_string());
    }
    if off.policy_control.environment_variable != on.policy_control.environment_variable {
        errors.push("OFF/ON policy environment variables differ".to_string());
    }
    if off.policy_control.value == on.policy_control.value {
        errors.push("OFF/ON policy values are identical".to_string());
    }
    errors
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metric(value: u64) -> Metric<u64> {
        Metric::available(value, "count", "test boundary")
    }

    fn unavailable_bytes() -> Metric<u64> {
        Metric::unavailable("bytes", "not observed", "missing production attribution")
    }

    fn run(arm: ResidencyArm) -> FreeTokenRunReport {
        let mut report = FreeTokenRunReport {
            schema: RUN_SCHEMA.to_string(),
            binary_marker: NATIVE_CUDA_BINARY_MARKER.to_string(),
            backend: "native-cuda".to_string(),
            arm,
            policy_control: PolicyControl {
                environment_variable: "POLICY".to_string(),
                value: arm.as_str().to_string(),
            },
            model_path: "model".to_string(),
            prompt_token_ids: vec![1, 2],
            requested_output_tokens: 2,
            emitted_tokens: 2,
            measured_generations: 1,
            decode_skip_tokens: 1,
            generated_token_ids: vec![3, 4],
            measurement_scope: "test".to_string(),
            warmup: WarmupRecord {
                requested_seconds: 8.0,
                actual_seconds: 8.1,
                generations: 1,
            },
            metrics: FreeTokenMetrics {
                selected_expert_logical_bytes: unavailable_bytes(),
                gpu_resident_expert_hit_bytes: unavailable_bytes(),
                host_to_device_expert_page_in_bytes: unavailable_bytes(),
                cpu_served_expert_bytes: unavailable_bytes(),
                expert_page_ins: Metric::unavailable(
                    "count",
                    "not observed",
                    "missing production attribution",
                ),
                expert_byte_hit_rate: Metric::unavailable(
                    "ratio",
                    "not observed",
                    "missing production attribution",
                ),
                expert_bytes_per_emitted_token: Metric::unavailable(
                    "bytes/token",
                    "not observed",
                    "missing production attribution",
                ),
                prefill_expert_bytes_by_layer: Metric::unavailable(
                    "bytes",
                    "not observed",
                    "missing production attribution",
                ),
                decode_expert_bytes_by_layer: Metric::unavailable(
                    "bytes",
                    "not observed",
                    "missing production attribution",
                ),
                model_weight_layout_bytes: Metric::available(100, "bytes", "analytical layout"),
                weight_gpu_resident_hit_bytes: Metric::available(40, "bytes", "process window"),
                weight_h2d_accounted_bytes: Metric::available(60, "bytes", "process window"),
                weight_zero_copy_host_read_bytes: Metric::available(0, "bytes", "process window"),
                weight_page_ins: metric(2),
                weight_cache_hits: metric(2),
                weight_vram_byte_hit_rate: Metric::available(0.4, "ratio", "process window"),
                weight_h2d_bytes_per_emitted_token: Metric::available(
                    30.0,
                    "bytes/token",
                    "process window",
                ),
                weight_host_link_bytes_per_emitted_token: Metric::available(
                    30.0,
                    "bytes/token",
                    "process window",
                ),
                cuda_graph_captures: metric(1),
                cuda_graph_replays: metric(1),
                cuda_graph_fallbacks: metric(0),
                measured_cuda_graph_captures: metric(0),
                measured_cuda_graph_replays: metric(1),
                measured_cuda_graph_fallbacks: metric(0),
                peak_committed_physical_bytes: Metric::available(90, "bytes", "arena lifetime"),
                managed_limit_bytes: Metric::available(100, "bytes", "engine policy"),
                oversubscribed_bytes: Metric::available(0, "bytes", "engine snapshot"),
                ref_underflows: metric(0),
                byte_underflows: metric(0),
                unaccounted_committed_bytes: Metric::available(0, "bytes", "arena snapshot"),
                wall_clock_prefill_milliseconds: Metric::available(
                    vec![1.0],
                    "milliseconds",
                    "host callback",
                ),
                wall_clock_decode_milliseconds_per_token: Metric::available(
                    vec![2.0],
                    "milliseconds/token",
                    "host callback",
                ),
                wall_clock_decode_tokens_per_second: Metric::available(
                    vec![500.0],
                    "tokens/second",
                    "host callback",
                ),
            },
            contract: ContractStatus::default(),
        };
        report.refresh_contract();
        report
    }

    #[test]
    fn pair_contract_rejects_token_drift_and_unsafe_memory() {
        let off = run(ResidencyArm::Off);
        let mut on = run(ResidencyArm::On);
        assert!(validate_pair(&off, &on).is_empty());

        on.generated_token_ids[1] = 9;
        on.metrics.unaccounted_committed_bytes.value = Some(1);
        on.decode_skip_tokens = 0;
        let errors = validate_pair(&off, &on).join("\n");
        assert!(errors.contains("not byte-identical"), "{errors}");
        assert!(errors.contains("unaccounted_committed_bytes"), "{errors}");
        assert!(errors.contains("decode-skip"), "{errors}");
    }

    #[test]
    fn run_contract_rejects_ort_and_missing_capture() {
        let mut report = run(ResidencyArm::Off);
        report.backend = "ort-cuda".to_string();
        report.metrics.cuda_graph_captures.value = Some(0);
        let errors = validate_run(&report).join("\n");
        assert!(errors.contains("ort-cuda"), "{errors}");
        assert!(errors.contains("greater than zero"), "{errors}");
    }

    #[test]
    fn unavailable_metrics_are_explicit_and_limit_is_inclusive() {
        let mut report = run(ResidencyArm::Off);
        report.metrics.peak_committed_physical_bytes.value = Some(100);
        assert!(validate_run(&report).is_empty());

        let json = serde_json::to_value(&report).expect("serialize run report");
        let selected = &json["metrics"]["selected_expert_logical_bytes"];
        assert!(selected["value"].is_null());
        assert_eq!(selected["unit"], "bytes");
        assert!(selected["unavailable_reason"].is_string());
    }
}

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExplicitOffloadEvidence {
    pub enabled: bool,
    pub device_budget_bytes: Option<u64>,
    pub scan_resistant_dense: bool,
    pub managed_no_spill: bool,
}

pub(crate) fn infer_memory_strategy_plan(
    config: &EngineConfig,
    resolved_vram_bytes: u64,
    model_weight_bytes: u64,
    kv_config: ModelKvConfig,
    weight_access_pattern: WeightAccessPattern,
    per_layer_weight_bytes: Vec<LayerWeightBytes>,
    offload: Option<ExplicitOffloadEvidence>,
) -> MemoryStrategyPlan {
    let explicit_vram_limit = config.limits.vram_limit != ResourceLimits::default().vram_limit;
    let kv_bytes_per_token = kv_config.bytes_per_token();
    let fits = model_weight_bytes <= resolved_vram_bytes;
    let mut decisions = Vec::new();

    decisions.push(MemoryStrategyDecision::new(
        "vram_limit",
        format_resource_limit_for_plan(config.limits.vram_limit),
        if explicit_vram_limit {
            DecisionSource::ExplicitOverride
        } else {
            DecisionSource::CompatibilityDefault
        },
        if explicit_vram_limit {
            "operator configured a device budget"
        } else {
            "no explicit device budget; preserving compatibility behavior"
        },
        format!("resolved_vram_bytes={resolved_vram_bytes}"),
    ));
    decisions.push(MemoryStrategyDecision::new(
        "weight_access_pattern",
        format!("{weight_access_pattern:?}"),
        match weight_access_pattern {
            WeightAccessPattern::Unknown => DecisionSource::Unknown,
            _ => DecisionSource::Inference,
        },
        "inferred from model metadata or graph helpers",
        format!(
            "total_weight_bytes={model_weight_bytes} qmoe_layer_count={}",
            per_layer_weight_bytes.len()
        ),
    ));
    decisions.push(MemoryStrategyDecision::new(
        "kv_bytes_per_token",
        kv_bytes_per_token
            .map(|bytes| bytes.to_string())
            .unwrap_or_else(|| "Unknown".to_string()),
        if kv_bytes_per_token.is_some() {
            DecisionSource::Inference
        } else {
            DecisionSource::Unknown
        },
        "derived from the resolved KV tensor geometry",
        format!(
            "page_size_bytes={:?} tokens_per_page={}",
            kv_config.page_size_bytes, kv_config.tokens_per_page
        ),
    ));

    if config.device_policy != DevicePolicy::Auto {
        decisions.push(MemoryStrategyDecision::new(
            "device_policy",
            format!("{:?}", config.device_policy),
            DecisionSource::ExplicitOverride,
            "static placement override is configured",
            "serving.memory.weights.device_policy is not auto",
        ));
    }

    if let Some(offload) = offload
        && (offload.enabled
            || offload.device_budget_bytes.is_some()
            || !offload.scan_resistant_dense)
    {
        decisions.push(MemoryStrategyDecision::new(
            "weight_offload",
            format!(
                "enabled={} device_budget_bytes={:?} scan_resistant_dense={}",
                offload.enabled, offload.device_budget_bytes, offload.scan_resistant_dense
            ),
            DecisionSource::ExplicitOverride,
            "CUDA weight-offload environment policy is configured",
            format!("managed_no_spill={}", offload.managed_no_spill),
        ));
    }

    let kv_unknown = kv_config.page_geometry_required && kv_bytes_per_token.is_none();
    let (strategy, source, reason) = match weight_access_pattern {
        _ if kv_unknown => (
            MemoryStrategy::Unknown,
            DecisionSource::Unknown,
            "KV geometry is unknown for a graph that requires paged KV sizing; falling back to current behavior",
        ),
        WeightAccessPattern::Unknown => (
            MemoryStrategy::Unknown,
            DecisionSource::Unknown,
            "graph pattern is ambiguous or unsupported; falling back to current behavior",
        ),
        WeightAccessPattern::MoeRouted => (
            MemoryStrategy::MoeRoutingAware,
            DecisionSource::Inference,
            "MoE routed expert access uses the existing routing/popularity-aware expert policy",
        ),
        WeightAccessPattern::SequentialDense if explicit_vram_limit && fits => (
            MemoryStrategy::FullResident,
            DecisionSource::Inference,
            "all package weights fit the resolved managed budget",
        ),
        WeightAccessPattern::SequentialDense if explicit_vram_limit => (
            MemoryStrategy::DynamicWeightResidency,
            DecisionSource::Inference,
            "package weights exceed the explicit managed budget; use dense scan-resistant eviction",
        ),
        WeightAccessPattern::Iterative if explicit_vram_limit && fits => (
            MemoryStrategy::FullResident,
            DecisionSource::Inference,
            "iterative package weights fit the resolved managed budget",
        ),
        _ => (
            MemoryStrategy::Compatibility,
            DecisionSource::CompatibilityDefault,
            "no explicit managed budget; preserving existing compatibility behavior",
        ),
    };

    decisions.push(MemoryStrategyDecision::new(
        "strategy",
        format!("{strategy:?}"),
        source,
        reason,
        format!(
            "total_weight_bytes={model_weight_bytes} resolved_vram_bytes={resolved_vram_bytes} fits={fits}"
        ),
    ));

    MemoryStrategyPlan {
        strategy,
        weight_access_pattern,
        total_weight_bytes: model_weight_bytes,
        kv_bytes_per_token,
        per_layer_weight_bytes,
        fits_resolved_device_budget: Some(fits),
        decisions,
    }
}

pub(crate) fn log_memory_strategy_plan(plan: &MemoryStrategyPlan) {
    tracing::info!(
        strategy = ?plan.strategy,
        weight_access_pattern = ?plan.weight_access_pattern,
        total_weight_bytes = plan.total_weight_bytes,
        kv_bytes_per_token = ?plan.kv_bytes_per_token,
        fits_resolved_device_budget = ?plan.fits_resolved_device_budget,
        per_layer_weight_bytes = ?plan.per_layer_weight_bytes,
        decisions = ?plan.decisions,
        "inferred memory strategy plan before applying memory policy"
    );
}

fn format_resource_limit_for_plan(limit: ResourceLimit) -> String {
    match limit {
        ResourceLimit::Bytes(bytes) => bytes.to_string(),
        ResourceLimit::Fraction(fraction) => fraction.to_string(),
        ResourceLimit::Auto => "auto".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_with_vram(bytes: u64) -> EngineConfig {
        EngineConfig {
            limits: ResourceLimits {
                vram_limit: ResourceLimit::Bytes(bytes),
                ..ResourceLimits::default()
            },
            ..EngineConfig::default()
        }
    }

    #[test]
    fn fits_budget_model_reports_full_resident() {
        let plan = infer_memory_strategy_plan(
            &config_with_vram(1_000),
            1_000,
            900,
            ModelKvConfig::known(160, 16),
            WeightAccessPattern::SequentialDense,
            Vec::new(),
            None,
        );
        assert_eq!(plan.strategy, MemoryStrategy::FullResident);
        assert_eq!(plan.kv_bytes_per_token, Some(10));
        assert_eq!(plan.fits_resolved_device_budget, Some(true));
    }

    #[test]
    fn dense_over_budget_selects_scan_resistant_dynamic_residency() {
        let plan = infer_memory_strategy_plan(
            &config_with_vram(1_000),
            1_000,
            1_001,
            ModelKvConfig::known(160, 16),
            WeightAccessPattern::SequentialDense,
            Vec::new(),
            Some(ExplicitOffloadEvidence {
                enabled: false,
                device_budget_bytes: None,
                scan_resistant_dense: true,
                managed_no_spill: true,
            }),
        );
        assert_eq!(plan.strategy, MemoryStrategy::DynamicWeightResidency);
        assert!(plan.decisions.iter().any(|decision| {
            decision.field == "strategy" && decision.evidence.contains("fits=false")
        }));
    }

    #[test]
    fn moe_pattern_selects_routing_aware_policy() {
        let plan = infer_memory_strategy_plan(
            &config_with_vram(1_000),
            1_000,
            2_000,
            ModelKvConfig::known(160, 16),
            WeightAccessPattern::MoeRouted,
            vec![LayerWeightBytes {
                layer_index: 0,
                bytes: 512,
            }],
            None,
        );
        assert_eq!(plan.strategy, MemoryStrategy::MoeRoutingAware);
        assert_eq!(plan.per_layer_weight_bytes[0].bytes, 512);
    }

    #[test]
    fn unknown_pattern_reports_unknown_without_guessing() {
        let plan = infer_memory_strategy_plan(
            &config_with_vram(1_000),
            1_000,
            2_000,
            ModelKvConfig::unknown(16),
            WeightAccessPattern::Unknown,
            Vec::new(),
            None,
        );
        assert_eq!(plan.strategy, MemoryStrategy::Unknown);
        assert_eq!(plan.kv_bytes_per_token, None);
    }

    #[test]
    fn compatibility_default_preserves_current_behavior_without_explicit_limit() {
        let plan = infer_memory_strategy_plan(
            &EngineConfig::default(),
            8_000,
            1_000,
            ModelKvConfig::known(160, 16),
            WeightAccessPattern::SequentialDense,
            Vec::new(),
            None,
        );
        assert_eq!(plan.strategy, MemoryStrategy::Compatibility);
    }

    #[test]
    fn explicit_override_is_observable_in_same_plan() {
        let plan = infer_memory_strategy_plan(
            &config_with_vram(1_000),
            1_000,
            2_000,
            ModelKvConfig::known(160, 16),
            WeightAccessPattern::SequentialDense,
            Vec::new(),
            Some(ExplicitOffloadEvidence {
                enabled: true,
                device_budget_bytes: Some(256),
                scan_resistant_dense: false,
                managed_no_spill: true,
            }),
        );
        assert!(plan.decisions.iter().any(|decision| {
            decision.source == DecisionSource::ExplicitOverride
                && decision.field == "weight_offload"
                && decision.value.contains("device_budget_bytes=Some(256)")
        }));
    }
}

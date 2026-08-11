use super::*;
use onnx_runtime_ep_api::{LazyWeightBoundary, lazy_weight_candidates};
use onnx_runtime_ir::{Graph, WeightRef};

#[derive(Clone, Debug)]
pub(crate) struct GraphMemoryEvidence {
    pub(crate) access_pattern: WeightAccessPattern,
    pub(crate) per_layer_weight_bytes: Vec<LayerWeightBytes>,
    pub(crate) reason: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct MemoryStrategyOverrides {
    pub(crate) weight_offload: Option<bool>,
    pub(crate) device_budget_bytes: Option<u64>,
    pub(crate) scan_resistant_dense: Option<bool>,
    pub(crate) async_pagein: Option<bool>,
}

pub(crate) struct MemoryStrategyPlanInput<'a> {
    pub(crate) config: &'a EngineConfig,
    pub(crate) resolved_vram_bytes: u64,
    pub(crate) model_weight_bytes: u64,
    pub(crate) kv_config: ModelKvConfig,
    pub(crate) graph: GraphMemoryEvidence,
    pub(crate) required_device_non_weight_bytes: u64,
    pub(crate) minimum_useful_weight_budget_bytes: u64,
    pub(crate) default_dynamic_device_budget_bytes: Option<u64>,
    /// The current runtime activation gate. This remains explicit-byte-only
    /// until #755 flips the managed VMM default after #716.
    pub(crate) inferred_policy_enabled: bool,
    pub(crate) overrides: MemoryStrategyOverrides,
    pub(crate) advisory_only: bool,
}

pub(crate) fn analyze_model_memory(model_path: &Path) -> GraphMemoryEvidence {
    match onnx_runtime_loader::load_model(model_path) {
        Ok(graph) => analyze_graph_memory(&graph),
        Err(error) => GraphMemoryEvidence {
            access_pattern: WeightAccessPattern::Unknown,
            per_layer_weight_bytes: Vec::new(),
            reason: format!("ONNX graph analysis was unavailable: {error}"),
        },
    }
}

pub(crate) fn combine_graph_memory(
    components: impl IntoIterator<Item = GraphMemoryEvidence>,
    pipeline_is_iterative: bool,
) -> GraphMemoryEvidence {
    let mut per_layer_weight_bytes = Vec::new();
    let mut saw_dense = false;
    let mut saw_moe = false;
    let mut saw_unknown = false;
    for component in components {
        per_layer_weight_bytes.extend(component.per_layer_weight_bytes);
        match component.access_pattern {
            WeightAccessPattern::SequentialDense => saw_dense = true,
            WeightAccessPattern::MoeRouted => saw_moe = true,
            WeightAccessPattern::Unknown => saw_unknown = true,
            WeightAccessPattern::Iterative => {}
        }
    }
    let (access_pattern, reason) = if pipeline_is_iterative {
        (
            WeightAccessPattern::Iterative,
            "pipeline metadata declares iterative execution".to_string(),
        )
    } else if saw_moe {
        (
            WeightAccessPattern::MoeRouted,
            "at least one pipeline component contains a routed MoE boundary".to_string(),
        )
    } else if saw_dense && !saw_unknown {
        (
            WeightAccessPattern::SequentialDense,
            "all weighted pipeline components expose sequential dense boundaries".to_string(),
        )
    } else {
        (
            WeightAccessPattern::Unknown,
            "pipeline component access patterns are mixed, empty, or unsupported".to_string(),
        )
    };
    GraphMemoryEvidence {
        access_pattern,
        per_layer_weight_bytes,
        reason,
    }
}

pub(crate) fn build_memory_strategy_plan(input: MemoryStrategyPlanInput<'_>) -> MemoryStrategyPlan {
    let kv_bytes_per_token = input.kv_config.bytes_per_token();
    let kv_unknown = input.kv_config.page_geometry_required && kv_bytes_per_token.is_none();
    let fits = input.model_weight_bytes <= input.resolved_vram_bytes;
    let available_weight_budget_bytes = input
        .resolved_vram_bytes
        .saturating_sub(input.required_device_non_weight_bytes);
    let inferred_strategy = if kv_unknown {
        MemoryStrategy::Unknown
    } else {
        match (fits, input.graph.access_pattern) {
            (_, WeightAccessPattern::Unknown) => MemoryStrategy::Unknown,
            (true, _) => MemoryStrategy::FullResident,
            (false, WeightAccessPattern::SequentialDense) => MemoryStrategy::DynamicWeightResidency,
            (false, WeightAccessPattern::MoeRouted) => MemoryStrategy::MoeRoutingAware,
            (false, WeightAccessPattern::Iterative) => MemoryStrategy::Unknown,
        }
    };

    let forced_offload = input.overrides.weight_offload == Some(true);
    let (strategy, strategy_source, strategy_reason) = if forced_offload {
        match input.graph.access_pattern {
            WeightAccessPattern::SequentialDense => (
                MemoryStrategy::DynamicWeightResidency,
                DecisionSource::ExplicitOverride,
                "ONNX_GENAI_WEIGHT_OFFLOAD explicitly enabled dense weight paging",
            ),
            WeightAccessPattern::MoeRouted => (
                MemoryStrategy::MoeRoutingAware,
                DecisionSource::ExplicitOverride,
                "ONNX_GENAI_WEIGHT_OFFLOAD explicitly enabled routed expert paging",
            ),
            WeightAccessPattern::Iterative | WeightAccessPattern::Unknown => (
                MemoryStrategy::Unknown,
                DecisionSource::Unknown,
                "the offload override is preserved, but the graph access pattern is unsupported",
            ),
        }
    } else if matches!(
        inferred_strategy,
        MemoryStrategy::DynamicWeightResidency | MemoryStrategy::MoeRoutingAware
    ) && !input.inferred_policy_enabled
    {
        (
            MemoryStrategy::Compatibility,
            DecisionSource::CompatibilityDefault,
            "inference runs without a flag, but automatic activation remains gated on #716/#755",
        )
    } else {
        (
            inferred_strategy,
            if inferred_strategy == MemoryStrategy::Unknown {
                DecisionSource::Unknown
            } else {
                DecisionSource::Inference
            },
            match inferred_strategy {
                MemoryStrategy::FullResident => "package weights fit the resolved device budget",
                MemoryStrategy::DynamicWeightResidency => {
                    "sequential dense weights exceed the resolved device budget"
                }
                MemoryStrategy::MoeRoutingAware => {
                    "routed MoE weights exceed the resolved device budget"
                }
                MemoryStrategy::Unknown => {
                    "graph access or required KV geometry is ambiguous or unsupported"
                }
                MemoryStrategy::Compatibility => unreachable!("not an inferred strategy"),
            },
        )
    };

    let compatibility_application = compatibility_application(&input, fits);
    let application = if input.advisory_only {
        compatibility_application
    } else {
        match strategy {
            MemoryStrategy::FullResident => MemoryPolicyApplication {
                weight_offload_enabled: false,
                device_budget_bytes: None,
                scan_resistant_dense: input.overrides.scan_resistant_dense.unwrap_or(true),
                managed_no_spill: matches!(input.config.limits.vram_limit, ResourceLimit::Bytes(_)),
                managed_limit_bytes: match input.config.limits.vram_limit {
                    ResourceLimit::Bytes(_) => Some(input.resolved_vram_bytes),
                    _ => None,
                },
                device_budget_is_override: false,
                auto_enabled_from_vram_limit: false,
            },
            MemoryStrategy::DynamicWeightResidency | MemoryStrategy::MoeRoutingAware => {
                MemoryPolicyApplication {
                    weight_offload_enabled: true,
                    device_budget_bytes: input.overrides.device_budget_bytes.or({
                        if forced_offload
                            && !matches!(input.config.limits.vram_limit, ResourceLimit::Bytes(_))
                        {
                            input.default_dynamic_device_budget_bytes
                        } else {
                            Some(available_weight_budget_bytes)
                        }
                    }),
                    scan_resistant_dense: input.overrides.scan_resistant_dense.unwrap_or(true),
                    managed_no_spill: matches!(
                        input.config.limits.vram_limit,
                        ResourceLimit::Bytes(_)
                    ),
                    managed_limit_bytes: match input.config.limits.vram_limit {
                        ResourceLimit::Bytes(_) => Some(input.resolved_vram_bytes),
                        _ => None,
                    },
                    device_budget_is_override: input.overrides.device_budget_bytes.is_some(),
                    auto_enabled_from_vram_limit: !forced_offload
                        && matches!(input.config.limits.vram_limit, ResourceLimit::Bytes(_)),
                }
            }
            MemoryStrategy::Compatibility | MemoryStrategy::Unknown => compatibility_application,
        }
    };

    let mut decisions = vec![
        MemoryStrategyDecision::new(
            "resolved_device_budget_bytes",
            input.resolved_vram_bytes.to_string(),
            match input.config.limits.vram_limit {
                ResourceLimit::Bytes(_) => DecisionSource::ExplicitOverride,
                _ => DecisionSource::Inference,
            },
            "resolved once through the same capacity helper used by the governor",
            format!(
                "configured_vram_limit={}",
                format_resource_limit_for_plan(input.config.limits.vram_limit)
            ),
        ),
        MemoryStrategyDecision::new(
            "weight_access_pattern",
            format!("{:?}", input.graph.access_pattern),
            if input.graph.access_pattern == WeightAccessPattern::Unknown {
                DecisionSource::Unknown
            } else {
                DecisionSource::Inference
            },
            input.graph.reason.clone(),
            format!(
                "pageable_boundary_count={}",
                input.graph.per_layer_weight_bytes.len()
            ),
        ),
        MemoryStrategyDecision::new(
            "total_weight_bytes",
            input.model_weight_bytes.to_string(),
            DecisionSource::Inference,
            "computed by the shared model-package weight helper",
            format!("fits_resolved_device_budget={fits}"),
        ),
        MemoryStrategyDecision::new(
            "available_weight_budget_bytes",
            available_weight_budget_bytes.to_string(),
            DecisionSource::Inference,
            "derived before provider construction from runtime-owned non-weight geometry",
            format!(
                "resolved_device_budget_bytes={} required_device_non_weight_bytes={} minimum_useful_weight_budget_bytes={}",
                input.resolved_vram_bytes,
                input.required_device_non_weight_bytes,
                input.minimum_useful_weight_budget_bytes
            ),
        ),
        MemoryStrategyDecision::new(
            "kv_bytes_per_token",
            kv_bytes_per_token
                .map(|bytes| bytes.to_string())
                .unwrap_or_else(|| "Unknown".to_string()),
            if kv_bytes_per_token.is_some() {
                DecisionSource::Inference
            } else {
                DecisionSource::Unknown
            },
            "derived from the exact ModelKvConfig passed to runtime admission",
            format!(
                "page_size_bytes={:?} tokens_per_page={}",
                input.kv_config.page_size_bytes, input.kv_config.tokens_per_page
            ),
        ),
        MemoryStrategyDecision::new(
            "inferred_strategy",
            format!("{inferred_strategy:?}"),
            if inferred_strategy == MemoryStrategy::Unknown {
                DecisionSource::Unknown
            } else {
                DecisionSource::Inference
            },
            "inferred unconditionally from graph evidence and the resolved budget",
            format!(
                "total_weight_bytes={} resolved_vram_bytes={} fits={fits}",
                input.model_weight_bytes, input.resolved_vram_bytes
            ),
        ),
    ];

    let mut strategy_decision = MemoryStrategyDecision::new(
        "strategy",
        format!("{strategy:?}"),
        strategy_source,
        strategy_reason,
        format!(
            "policy_enabled={} advisory_only={}",
            input.inferred_policy_enabled, input.advisory_only
        ),
    );
    if strategy != inferred_strategy {
        strategy_decision = strategy_decision.with_inferred_value(format!("{inferred_strategy:?}"));
    }
    decisions.push(strategy_decision);

    if input.config.device_policy != DevicePolicy::Auto {
        decisions.push(MemoryStrategyDecision::new(
            "device_policy",
            format!("{:?}", input.config.device_policy),
            DecisionSource::ExplicitOverride,
            "static placement override is configured",
            "serving.memory.weights.device_policy is not auto",
        ));
    }
    if let Some(enabled) = input.overrides.weight_offload {
        decisions.push(MemoryStrategyDecision::new(
            "weight_offload",
            enabled.to_string(),
            DecisionSource::ExplicitOverride,
            "ONNX_GENAI_WEIGHT_OFFLOAD is explicitly configured",
            format!(
                "effective_weight_offload={}",
                application.weight_offload_enabled
            ),
        ));
    }
    if let Some(bytes) = input.overrides.device_budget_bytes {
        decisions.push(MemoryStrategyDecision::new(
            "weight_device_budget_bytes",
            bytes.to_string(),
            DecisionSource::ExplicitOverride,
            "ONNX_GENAI_WEIGHT_OFFLOAD_DEVICE_BYTES is explicitly configured",
            format!(
                "effective_device_budget_bytes={:?}",
                application.device_budget_bytes
            ),
        ));
    }
    if let Some(scan_resistant) = input.overrides.scan_resistant_dense {
        decisions.push(MemoryStrategyDecision::new(
            "scan_resistant_dense",
            scan_resistant.to_string(),
            DecisionSource::ExplicitOverride,
            "ONNX_GENAI_WEIGHT_OFFLOAD_SCAN_RESISTANT is explicitly configured",
            format!(
                "effective_scan_resistant_dense={}",
                application.scan_resistant_dense
            ),
        ));
    }
    if let Some(requested) = input.overrides.async_pagein {
        decisions.push(
            MemoryStrategyDecision::new(
                "async_pagein",
                "false",
                DecisionSource::CompatibilityDefault,
                "sequential prefetch remains disabled because #715 measured it ineffective",
                "the runtime keeps measured host-blocking demand page-in",
            )
            .with_inferred_value(requested.to_string()),
        );
    }
    decisions.push(MemoryStrategyDecision::new(
        "application",
        format!(
            "offload={} scan_resistant_dense={} managed_no_spill={} device_budget_bytes={:?}",
            application.weight_offload_enabled,
            application.scan_resistant_dense,
            application.managed_no_spill,
            application.device_budget_bytes
        ),
        if forced_offload
            || input.overrides.device_budget_bytes.is_some()
            || input.overrides.scan_resistant_dense.is_some()
        {
            DecisionSource::ExplicitOverride
        } else if strategy == MemoryStrategy::Compatibility || strategy == MemoryStrategy::Unknown {
            DecisionSource::CompatibilityDefault
        } else {
            DecisionSource::Inference
        },
        "native provider construction consumes these exact fields from the plan",
        format!("strategy={strategy:?}"),
    ));

    MemoryStrategyPlan {
        strategy,
        inferred_strategy,
        weight_access_pattern: input.graph.access_pattern,
        total_weight_bytes: input.model_weight_bytes,
        kv_bytes_per_token,
        per_layer_weight_bytes: input.graph.per_layer_weight_bytes,
        resolved_device_budget_bytes: Some(input.resolved_vram_bytes),
        fits_resolved_device_budget: Some(fits),
        application,
        advisory_only: input.advisory_only,
        decisions,
    }
}

fn compatibility_application(
    input: &MemoryStrategyPlanInput<'_>,
    fits: bool,
) -> MemoryPolicyApplication {
    let explicit_bytes = matches!(input.config.limits.vram_limit, ResourceLimit::Bytes(_));
    let forced = input.overrides.weight_offload == Some(true);
    let auto_enabled = explicit_bytes && !fits && !forced;
    let enabled = forced || auto_enabled;
    MemoryPolicyApplication {
        weight_offload_enabled: enabled,
        device_budget_bytes: if enabled {
            input
                .overrides
                .device_budget_bytes
                .or_else(|| {
                    auto_enabled.then_some(
                        input
                            .resolved_vram_bytes
                            .saturating_sub(input.required_device_non_weight_bytes),
                    )
                })
                .or(input.default_dynamic_device_budget_bytes)
        } else {
            None
        },
        scan_resistant_dense: input.overrides.scan_resistant_dense.unwrap_or(true),
        managed_no_spill: explicit_bytes,
        managed_limit_bytes: explicit_bytes.then_some(input.resolved_vram_bytes),
        device_budget_is_override: input.overrides.device_budget_bytes.is_some(),
        auto_enabled_from_vram_limit: auto_enabled,
    }
}

pub(crate) fn log_memory_strategy_plan(plan: &MemoryStrategyPlan, scope: &'static str) {
    let application = plan.runtime_application();
    tracing::info!(
        scope,
        strategy = ?plan.strategy,
        inferred_strategy = ?plan.inferred_strategy,
        weight_access_pattern = ?plan.weight_access_pattern,
        total_weight_bytes = plan.total_weight_bytes,
        kv_bytes_per_token = ?plan.kv_bytes_per_token,
        resolved_device_budget_bytes = ?plan.resolved_device_budget_bytes,
        fits_resolved_device_budget = ?plan.fits_resolved_device_budget,
        weight_offload_enabled = application.weight_offload_enabled,
        scan_resistant_dense = application.scan_resistant_dense,
        managed_no_spill = application.managed_no_spill,
        device_budget_bytes = ?application.device_budget_bytes,
        advisory_only = plan.advisory_only,
        "memory strategy plan before applying memory policy"
    );
    tracing::debug!(
        scope,
        plan = %serde_json::to_string(plan).unwrap_or_else(|_| format!("{plan:?}")),
        "memory strategy plan details"
    );
}

#[cfg(feature = "cuda")]
pub(crate) fn memory_strategy_overrides_from_cuda_env(
    policy: onnx_runtime_ep_cuda::DeviceOffloadPolicy,
) -> MemoryStrategyOverrides {
    MemoryStrategyOverrides {
        weight_offload: std::env::var_os(onnx_runtime_ep_cuda::WEIGHT_OFFLOAD_ENV)
            .map(|_| policy.enabled),
        device_budget_bytes: std::env::var_os(
            onnx_runtime_ep_cuda::WEIGHT_OFFLOAD_DEVICE_BYTES_ENV,
        )
        .and(policy.device_budget_bytes),
        scan_resistant_dense: std::env::var_os(
            onnx_runtime_ep_cuda::WEIGHT_OFFLOAD_SCAN_RESISTANT_ENV,
        )
        .map(|_| policy.scan_resistant_dense),
        async_pagein: std::env::var_os(onnx_runtime_ep_cuda::WEIGHT_OFFLOAD_ASYNC_PAGEIN_ENV)
            .map(|_| policy.async_pagein),
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_policy_from_memory_strategy_plan(
    plan: &MemoryStrategyPlan,
) -> onnx_runtime_ep_cuda::DeviceOffloadPolicy {
    let application = plan.runtime_application();
    onnx_runtime_ep_cuda::DeviceOffloadPolicy {
        enabled: application.weight_offload_enabled,
        managed_no_spill: application.managed_no_spill,
        managed_limit_bytes: application.managed_limit_bytes,
        device_budget_bytes: application.device_budget_bytes,
        // #715 removed prefetch. Keep the policy on measured demand page-in.
        async_pagein: false,
        scan_resistant_dense: application.scan_resistant_dense,
    }
}

fn analyze_graph_memory(graph: &Graph) -> GraphMemoryEvidence {
    let order = match graph.topological_order() {
        Ok(order) => order,
        Err(error) => {
            return GraphMemoryEvidence {
                access_pattern: WeightAccessPattern::Unknown,
                per_layer_weight_bytes: Vec::new(),
                reason: format!("graph topology is unsupported: {error}"),
            };
        }
    };
    let mut per_layer_weight_bytes = Vec::new();
    let mut saw_dense = false;
    let mut saw_moe = false;
    if order.iter().any(|&node_id| {
        let node = graph.node(node_id);
        matches!(node.op_type.as_str(), "If" | "Loop" | "Scan")
    }) {
        return GraphMemoryEvidence {
            access_pattern: WeightAccessPattern::Unknown,
            per_layer_weight_bytes,
            reason: "control-flow graph requires runtime-dependent access analysis".to_string(),
        };
    }
    let candidates = lazy_weight_candidates(graph);
    for node_id in order {
        let mut bytes = 0_u64;
        let mut boundary = None;
        for candidate in candidates
            .iter()
            .filter(|candidate| candidate.first_consumer == node_id)
        {
            boundary.get_or_insert(candidate.boundary);
            let weight = &graph.initializers[&candidate.value];
            let Some(weight_bytes) = weight_bytes(weight) else {
                return GraphMemoryEvidence {
                    access_pattern: WeightAccessPattern::Unknown,
                    per_layer_weight_bytes,
                    reason: format!("initializer geometry overflows at node {}", node_id.0),
                };
            };
            let Some(total) = bytes.checked_add(weight_bytes) else {
                return GraphMemoryEvidence {
                    access_pattern: WeightAccessPattern::Unknown,
                    per_layer_weight_bytes,
                    reason: format!("layer weight bytes overflow at node {}", node_id.0),
                };
            };
            bytes = total;
        }
        let Some(boundary) = boundary else {
            continue;
        };
        match boundary {
            LazyWeightBoundary::MatMul | LazyWeightBoundary::MatMulNBits => saw_dense = true,
            LazyWeightBoundary::BlockQuantizedMoe | LazyWeightBoundary::QMoe => saw_moe = true,
        }
        per_layer_weight_bytes.push(LayerWeightBytes {
            layer_index: per_layer_weight_bytes.len(),
            bytes,
        });
    }
    let (access_pattern, reason) = if saw_moe {
        (
            WeightAccessPattern::MoeRouted,
            "at least one lazy-weight boundary uses routed MoE expert access".to_string(),
        )
    } else if saw_dense {
        (
            WeightAccessPattern::SequentialDense,
            "dense lazy-weight boundaries execute in deterministic topological order".to_string(),
        )
    } else {
        (
            WeightAccessPattern::Unknown,
            "no supported lazy-weight boundary was found".to_string(),
        )
    };
    GraphMemoryEvidence {
        access_pattern,
        per_layer_weight_bytes,
        reason,
    }
}

fn weight_bytes(weight: &WeightRef) -> Option<u64> {
    let bytes = match weight {
        WeightRef::Inline(tensor) => tensor.checked_expected_bytes()?,
        WeightRef::External { length, .. } => *length,
    };
    u64::try_from(bytes).ok()
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
    use onnx_runtime_ir::{DataType, Node, NodeId, TensorData};

    fn graph_with_boundary(domain: &str, op_type: &str) -> GraphMemoryEvidence {
        let mut graph = Graph::new();
        let input = graph.create_named_value("input", DataType::Float32, vec![1.into(), 4.into()]);
        graph.add_input(input);
        let weight =
            graph.create_named_value("weight", DataType::Float32, vec![4.into(), 4.into()]);
        graph.set_initializer(
            weight,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Float32,
                vec![4, 4],
                vec![0; 64],
            )),
        );
        let output =
            graph.create_named_value("output", DataType::Float32, vec![1.into(), 4.into()]);
        graph.add_output(output);
        let mut node = Node::new(
            NodeId(0),
            op_type,
            vec![Some(input), Some(weight)],
            vec![output],
        );
        node.domain = domain.to_string();
        node.name = "layer.0".to_string();
        graph.insert_node(node);
        analyze_graph_memory(&graph)
    }

    fn input(
        config: &EngineConfig,
        graph: GraphMemoryEvidence,
        limit: u64,
        weights: u64,
        overrides: MemoryStrategyOverrides,
    ) -> MemoryStrategyPlan {
        build_memory_strategy_plan(MemoryStrategyPlanInput {
            config,
            resolved_vram_bytes: limit,
            model_weight_bytes: weights,
            kv_config: ModelKvConfig::known(160, 16),
            graph,
            required_device_non_weight_bytes: 0,
            minimum_useful_weight_budget_bytes: 0,
            default_dynamic_device_budget_bytes: None,
            inferred_policy_enabled: matches!(config.limits.vram_limit, ResourceLimit::Bytes(_)),
            overrides,
            advisory_only: false,
        })
    }

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
    fn no_flag_inference_runs_but_activation_remains_compatibility_gated() {
        let config = EngineConfig::default();
        let plan = input(
            &config,
            graph_with_boundary("", "MatMul"),
            64,
            128,
            MemoryStrategyOverrides::default(),
        );
        assert_eq!(
            plan.inferred_strategy,
            MemoryStrategy::DynamicWeightResidency
        );
        assert_eq!(plan.strategy, MemoryStrategy::Compatibility);
        assert!(!plan.application.weight_offload_enabled);
    }

    #[test]
    fn dense_over_budget_plan_drives_scan_resistant_application() {
        let config = config_with_vram(64);
        let plan = input(
            &config,
            graph_with_boundary("", "MatMul"),
            64,
            128,
            MemoryStrategyOverrides::default(),
        );
        assert_eq!(plan.strategy, MemoryStrategy::DynamicWeightResidency);
        assert!(plan.application.weight_offload_enabled);
        assert!(plan.application.scan_resistant_dense);
        assert_eq!(plan.application.device_budget_bytes, Some(64));
    }

    #[test]
    fn fitting_model_is_full_resident_without_offload() {
        let config = config_with_vram(256);
        let plan = input(
            &config,
            graph_with_boundary("", "MatMul"),
            256,
            128,
            MemoryStrategyOverrides::default(),
        );
        assert_eq!(plan.inferred_strategy, MemoryStrategy::FullResident);
        assert_eq!(plan.strategy, MemoryStrategy::FullResident);
        assert!(!plan.application.weight_offload_enabled);
    }

    #[test]
    fn fitting_moe_is_full_resident_and_over_budget_moe_is_routing_aware() {
        let fitting_config = config_with_vram(256);
        let fitting = input(
            &fitting_config,
            graph_with_boundary("com.microsoft", "QMoE"),
            256,
            128,
            MemoryStrategyOverrides::default(),
        );
        assert_eq!(fitting.strategy, MemoryStrategy::FullResident);

        let paged_config = config_with_vram(64);
        let paged = input(
            &paged_config,
            graph_with_boundary("com.microsoft", "QMoE"),
            64,
            128,
            MemoryStrategyOverrides::default(),
        );
        assert_eq!(paged.strategy, MemoryStrategy::MoeRoutingAware);
        assert!(paged.application.weight_offload_enabled);
    }

    #[test]
    fn explicit_override_changes_effective_strategy_and_application() {
        let config = config_with_vram(256);
        let inferred = input(
            &config,
            graph_with_boundary("", "MatMul"),
            256,
            128,
            MemoryStrategyOverrides::default(),
        );
        let overridden = input(
            &config,
            graph_with_boundary("", "MatMul"),
            256,
            128,
            MemoryStrategyOverrides {
                weight_offload: Some(true),
                device_budget_bytes: Some(32),
                scan_resistant_dense: Some(false),
                async_pagein: None,
            },
        );
        assert_eq!(inferred.strategy, MemoryStrategy::FullResident);
        assert!(
            !inferred.runtime_application().weight_offload_enabled,
            "the unmodified run must stay fully resident"
        );
        assert_eq!(overridden.inferred_strategy, MemoryStrategy::FullResident);
        assert_eq!(overridden.strategy, MemoryStrategy::DynamicWeightResidency);
        assert!(overridden.application.weight_offload_enabled);
        assert!(
            overridden.runtime_application().weight_offload_enabled,
            "the override must change the policy consumed by the runtime"
        );
        assert_eq!(overridden.application.device_budget_bytes, Some(32));
        assert!(!overridden.application.scan_resistant_dense);
        assert!(overridden.decisions.iter().any(|decision| {
            decision.field == "strategy"
                && decision.source == DecisionSource::ExplicitOverride
                && decision.inferred_value.as_deref() == Some("FullResident")
        }));
    }

    #[test]
    fn dynamic_unsupported_graph_reports_unknown_and_preserves_compatibility() {
        let mut graph = Graph::new();
        let symbol = graph.create_symbol(Some("dynamic".to_string()));
        let dynamic_input =
            graph.create_named_value("input", DataType::Float32, vec![symbol.into()]);
        graph.add_input(dynamic_input);
        let output = graph.create_named_value("output", DataType::Float32, vec![symbol.into()]);
        graph.add_output(output);
        let mut loop_node = Node::new(NodeId(0), "Loop", vec![Some(dynamic_input)], vec![output]);
        loop_node.name = "dynamic_loop".to_string();
        graph.insert_node(loop_node);
        let config = config_with_vram(64);
        let evidence = analyze_graph_memory(&graph);
        assert!(
            evidence.reason.contains("control-flow"),
            "unexpected graph evidence: {}",
            evidence.reason
        );
        let plan = input(
            &config,
            evidence,
            64,
            128,
            MemoryStrategyOverrides::default(),
        );
        assert_eq!(plan.inferred_strategy, MemoryStrategy::Unknown);
        assert_eq!(plan.strategy, MemoryStrategy::Unknown);
        assert!(plan.application.weight_offload_enabled);
        assert!(plan.application.auto_enabled_from_vram_limit);
    }

    #[test]
    fn model_path_inference_reads_live_graph_fixture() {
        let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tiny-llm/model.onnx.textproto");
        let evidence = analyze_model_memory(&model_path);

        assert_eq!(
            evidence.access_pattern,
            WeightAccessPattern::SequentialDense,
            "live graph inference failed: {}",
            evidence.reason
        );
        assert!(
            evidence
                .per_layer_weight_bytes
                .iter()
                .any(|layer| layer.bytes > 0),
            "live graph inference must measure pageable layer weights"
        );
    }

    #[test]
    fn deleting_graph_inference_changes_the_plan() {
        let config = config_with_vram(64);
        let dense = input(
            &config,
            graph_with_boundary("", "MatMul"),
            64,
            128,
            MemoryStrategyOverrides::default(),
        );
        let unknown = input(
            &config,
            GraphMemoryEvidence {
                access_pattern: WeightAccessPattern::Unknown,
                per_layer_weight_bytes: Vec::new(),
                reason: "inference removed".to_string(),
            },
            64,
            128,
            MemoryStrategyOverrides::default(),
        );
        assert_eq!(dense.strategy, MemoryStrategy::DynamicWeightResidency);
        assert_eq!(unknown.strategy, MemoryStrategy::Unknown);
        assert_ne!(dense.strategy, unknown.strategy);
    }

    #[test]
    fn perturbing_the_plan_changes_the_runtime_application() {
        let config = config_with_vram(64);
        let mut plan = input(
            &config,
            graph_with_boundary("", "MatMul"),
            64,
            128,
            MemoryStrategyOverrides::default(),
        );
        assert!(plan.runtime_application().weight_offload_enabled);

        plan.strategy = MemoryStrategy::FullResident;
        assert!(
            !plan.runtime_application().weight_offload_enabled,
            "runtime policy must consume the effective strategy from the plan"
        );
    }
}

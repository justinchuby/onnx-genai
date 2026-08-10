use super::{DevicePolicy, ResourceLimit};
use onnx_runtime_ep_api::LazyWeightBoundary;
use onnx_runtime_ir::{Graph, WeightRef};
use serde::Serialize;
use std::path::Path;

/// Load-time memory residency choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryStrategy {
    FullResident,
    DynamicWeightResidency,
    MoeExpertPaged,
    Unknown,
}

/// Weight reuse pattern established from the loaded ONNX IR or pipeline metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WeightAccessPattern {
    SequentialDense,
    MoeRouted,
    Iterative,
    Unknown,
}

/// Eviction behavior selected for pageable weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryEvictionPolicy {
    None,
    ScanResistant,
    MoeRoutingAware,
    CompatibilityLru,
    Unknown,
}

/// Provenance of one plan decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryDecisionSource {
    Inference,
    ExplicitOverride,
    CompatibilityFallback,
}

/// One effective value plus its provenance and the inferred value it replaced.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct MemoryPlanDecision<T> {
    pub value: T,
    pub source: MemoryDecisionSource,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inferred_value: Option<T>,
    pub reason: String,
}

/// Byte size attributed to one pageable execution boundary in topological order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct LayerWeightSize {
    pub node: String,
    pub boundary: String,
    pub bytes: u64,
}

/// Explicit compatibility controls observed while resolving the plan.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct MemoryStrategyOverrides {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_limit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_policy: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight_offload: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight_device_budget_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scan_resistant_dense: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub async_pagein: Option<bool>,
}

impl MemoryStrategyOverrides {
    pub(crate) fn from_config(config: &super::EngineConfig) -> Self {
        let vram_limit = match config.limits.vram_limit {
            ResourceLimit::Auto => None,
            ResourceLimit::Bytes(bytes) => Some(bytes.to_string()),
            ResourceLimit::Fraction(fraction) => Some(fraction.to_string()),
        };
        let device_policy = match config.device_policy {
            DevicePolicy::Auto => None,
            DevicePolicy::Cpu => Some("cpu".to_string()),
            DevicePolicy::GpuLayers(layers) => Some(format!("gpu_layers:{layers}")),
            DevicePolicy::DeviceBytes(bytes) => Some(format!("device_bytes:{bytes}")),
        };
        Self {
            vram_limit,
            device_policy,
            ..Self::default()
        }
    }
}

/// Observable result of graph/metadata memory-strategy resolution.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct MemoryStrategyPlan {
    pub strategy: MemoryPlanDecision<MemoryStrategy>,
    pub weight_access_pattern: MemoryPlanDecision<WeightAccessPattern>,
    pub total_weight_bytes: MemoryPlanDecision<u64>,
    pub exact_kv_bytes_per_token: MemoryPlanDecision<Option<u64>>,
    pub per_layer_weight_bytes: MemoryPlanDecision<Vec<LayerWeightSize>>,
    pub resolved_device_limit_bytes: MemoryPlanDecision<Option<u64>>,
    pub weights_fit_resolved_budget: MemoryPlanDecision<Option<bool>>,
    pub weight_offload_enabled: MemoryPlanDecision<bool>,
    pub eviction_policy: MemoryPlanDecision<MemoryEvictionPolicy>,
    pub overrides: MemoryStrategyOverrides,
    /// True when the runtime can report the decision but lacks the activation,
    /// scratch, or lifecycle knowledge needed to enforce it safely.
    pub advisory_only: bool,
}

impl Default for MemoryStrategyPlan {
    fn default() -> Self {
        build_memory_strategy_plan(MemoryStrategyPlanInput {
            graph: GraphMemoryEvidence {
                access_pattern: WeightAccessPattern::Unknown,
                per_layer_weight_bytes: Vec::new(),
                reason: "model graph has not been analyzed".to_string(),
            },
            total_weight_bytes: 0,
            exact_kv_bytes_per_token: None,
            resolved_device_limit_bytes: None,
            limit_is_override: false,
            automatic_offload_allowed: false,
            compatibility_offload_enabled: false,
            overrides: MemoryStrategyOverrides::default(),
            advisory_only: true,
        })
    }
}

#[derive(Clone, Debug)]
pub(crate) struct GraphMemoryEvidence {
    pub(crate) access_pattern: WeightAccessPattern,
    pub(crate) per_layer_weight_bytes: Vec<LayerWeightSize>,
    pub(crate) reason: String,
}

pub(crate) fn analyze_model_memory(model_path: &Path) -> GraphMemoryEvidence {
    match onnx_runtime_loader::load_model(model_path) {
        Ok(graph) => analyze_graph_memory(&graph),
        Err(error) => GraphMemoryEvidence {
            access_pattern: WeightAccessPattern::Unknown,
            per_layer_weight_bytes: Vec::new(),
            reason: format!("ONNX IR analysis was unavailable: {error}"),
        },
    }
}

pub(crate) fn combine_graph_memory(
    components: impl IntoIterator<Item = GraphMemoryEvidence>,
    pipeline_is_iterative: bool,
) -> GraphMemoryEvidence {
    let mut layers = Vec::new();
    let mut saw_moe = false;
    let mut saw_unknown = false;
    let mut saw_dense = false;
    for component in components {
        layers.extend(component.per_layer_weight_bytes);
        match component.access_pattern {
            WeightAccessPattern::MoeRouted => saw_moe = true,
            WeightAccessPattern::SequentialDense => saw_dense = true,
            WeightAccessPattern::Unknown => saw_unknown = true,
            WeightAccessPattern::Iterative => {}
        }
    }
    let (access_pattern, reason) = if pipeline_is_iterative {
        (
            WeightAccessPattern::Iterative,
            "pipeline metadata declares an iterative execution strategy".to_string(),
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
            "pipeline component access patterns are mixed or unsupported".to_string(),
        )
    };
    GraphMemoryEvidence {
        access_pattern,
        per_layer_weight_bytes: layers,
        reason,
    }
}

pub(crate) struct MemoryStrategyPlanInput {
    pub(crate) graph: GraphMemoryEvidence,
    pub(crate) total_weight_bytes: u64,
    pub(crate) exact_kv_bytes_per_token: Option<u64>,
    pub(crate) resolved_device_limit_bytes: Option<u64>,
    pub(crate) limit_is_override: bool,
    pub(crate) automatic_offload_allowed: bool,
    pub(crate) compatibility_offload_enabled: bool,
    pub(crate) overrides: MemoryStrategyOverrides,
    pub(crate) advisory_only: bool,
}

pub(crate) fn build_memory_strategy_plan(input: MemoryStrategyPlanInput) -> MemoryStrategyPlan {
    let fits = input
        .resolved_device_limit_bytes
        .map(|limit| input.total_weight_bytes <= limit);
    let inferred_strategy = match (fits, input.graph.access_pattern) {
        (Some(true), _) => MemoryStrategy::FullResident,
        (Some(false), WeightAccessPattern::SequentialDense) if input.automatic_offload_allowed => {
            MemoryStrategy::DynamicWeightResidency
        }
        (Some(false), WeightAccessPattern::MoeRouted) if input.automatic_offload_allowed => {
            MemoryStrategy::MoeExpertPaged
        }
        _ => MemoryStrategy::Unknown,
    };
    let inferred_reason = match inferred_strategy {
        MemoryStrategy::FullResident => format!(
            "{} weight bytes fit the resolved {} byte device budget",
            input.total_weight_bytes,
            input.resolved_device_limit_bytes.unwrap_or(0)
        ),
        MemoryStrategy::DynamicWeightResidency => format!(
            "{} sequential-dense weight bytes exceed the resolved {} byte device budget",
            input.total_weight_bytes,
            input.resolved_device_limit_bytes.unwrap_or(0)
        ),
        MemoryStrategy::MoeExpertPaged => format!(
            "{} routed-MoE weight bytes exceed the resolved {} byte device budget",
            input.total_weight_bytes,
            input.resolved_device_limit_bytes.unwrap_or(0)
        ),
        MemoryStrategy::Unknown => format!(
            "no proven automatic strategy: {}; resolved_limit={:?}",
            input.graph.reason, input.resolved_device_limit_bytes
        ),
    };

    let forced_offload = input.overrides.weight_offload;
    let (effective_strategy, strategy_source, strategy_reason, inferred_value) =
        match forced_offload {
            Some(true) => {
                let forced = match input.graph.access_pattern {
                    WeightAccessPattern::SequentialDense => MemoryStrategy::DynamicWeightResidency,
                    WeightAccessPattern::MoeRouted => MemoryStrategy::MoeExpertPaged,
                    WeightAccessPattern::Iterative | WeightAccessPattern::Unknown => {
                        MemoryStrategy::Unknown
                    }
                };
                (
                    forced,
                    MemoryDecisionSource::ExplicitOverride,
                    "ONNX_GENAI_WEIGHT_OFFLOAD explicitly enabled weight paging".to_string(),
                    Some(inferred_strategy),
                )
            }
            Some(false) => (
                if fits == Some(true) {
                    MemoryStrategy::FullResident
                } else {
                    MemoryStrategy::Unknown
                },
                MemoryDecisionSource::ExplicitOverride,
                "ONNX_GENAI_WEIGHT_OFFLOAD explicitly disabled weight paging".to_string(),
                Some(inferred_strategy),
            ),
            None => (
                inferred_strategy,
                MemoryDecisionSource::Inference,
                inferred_reason,
                None,
            ),
        };

    let inferred_eviction = match effective_strategy {
        MemoryStrategy::DynamicWeightResidency => MemoryEvictionPolicy::ScanResistant,
        MemoryStrategy::MoeExpertPaged => MemoryEvictionPolicy::MoeRoutingAware,
        MemoryStrategy::FullResident => MemoryEvictionPolicy::None,
        MemoryStrategy::Unknown => MemoryEvictionPolicy::Unknown,
    };
    let (eviction, eviction_source, eviction_reason, inferred_eviction_value) = match input
        .overrides
        .scan_resistant_dense
    {
        Some(false) if effective_strategy == MemoryStrategy::DynamicWeightResidency => (
            MemoryEvictionPolicy::CompatibilityLru,
            MemoryDecisionSource::ExplicitOverride,
            "ONNX_GENAI_WEIGHT_OFFLOAD_SCAN_RESISTANT explicitly selected compatibility LRU"
                .to_string(),
            Some(inferred_eviction),
        ),
        Some(true) if effective_strategy == MemoryStrategy::DynamicWeightResidency => (
            MemoryEvictionPolicy::ScanResistant,
            MemoryDecisionSource::ExplicitOverride,
            "ONNX_GENAI_WEIGHT_OFFLOAD_SCAN_RESISTANT explicitly selected scan-resistant eviction"
                .to_string(),
            Some(inferred_eviction),
        ),
        _ => (
            inferred_eviction,
            MemoryDecisionSource::Inference,
            match inferred_eviction {
                MemoryEvictionPolicy::MoeRoutingAware => {
                    "routed experts retain the existing routing/popularity-aware policy"
                }
                MemoryEvictionPolicy::ScanResistant => {
                    "sequential dense scans use the measured scan-resistant policy"
                }
                MemoryEvictionPolicy::None => "resident weights require no eviction",
                _ => "unsupported access pattern preserves compatibility behavior",
            }
            .to_string(),
            None,
        ),
    };

    let inferred_offload = matches!(
        effective_strategy,
        MemoryStrategy::DynamicWeightResidency | MemoryStrategy::MoeExpertPaged
    );
    let (offload_enabled, offload_source, offload_reason) =
        if effective_strategy == MemoryStrategy::Unknown {
            (
                input.compatibility_offload_enabled,
                MemoryDecisionSource::CompatibilityFallback,
                "unknown strategy preserves the pre-existing offload resolution".to_string(),
            )
        } else {
            (
                inferred_offload,
                strategy_source,
                "derived from the effective residency strategy".to_string(),
            )
        };

    MemoryStrategyPlan {
        strategy: MemoryPlanDecision {
            value: effective_strategy,
            source: strategy_source,
            inferred_value,
            reason: strategy_reason,
        },
        weight_access_pattern: MemoryPlanDecision {
            value: input.graph.access_pattern,
            source: MemoryDecisionSource::Inference,
            inferred_value: None,
            reason: input.graph.reason,
        },
        total_weight_bytes: MemoryPlanDecision {
            value: input.total_weight_bytes,
            source: MemoryDecisionSource::Inference,
            inferred_value: None,
            reason: "computed by the shared model package weight-size helper".to_string(),
        },
        exact_kv_bytes_per_token: MemoryPlanDecision {
            value: input.exact_kv_bytes_per_token,
            source: MemoryDecisionSource::Inference,
            inferred_value: None,
            reason: if input.exact_kv_bytes_per_token.is_some() {
                "computed from exact per-tensor KV geometry".to_string()
            } else {
                "KV geometry is absent or unsupported".to_string()
            },
        },
        per_layer_weight_bytes: MemoryPlanDecision {
            value: input.graph.per_layer_weight_bytes,
            source: MemoryDecisionSource::Inference,
            inferred_value: None,
            reason:
                "summed initializer regions at existing lazy-weight boundaries in topological order"
                    .to_string(),
        },
        resolved_device_limit_bytes: MemoryPlanDecision {
            value: input.resolved_device_limit_bytes,
            source: if input.limit_is_override {
                MemoryDecisionSource::ExplicitOverride
            } else {
                MemoryDecisionSource::Inference
            },
            inferred_value: None,
            reason: if input.limit_is_override {
                "resolved from an explicit CLI/configured VRAM limit".to_string()
            } else {
                "resolved by the shared device memory authority".to_string()
            },
        },
        weights_fit_resolved_budget: MemoryPlanDecision {
            value: fits,
            source: MemoryDecisionSource::Inference,
            inferred_value: None,
            reason: "compared total package weight bytes with the resolved device limit"
                .to_string(),
        },
        weight_offload_enabled: MemoryPlanDecision {
            value: offload_enabled,
            source: offload_source,
            inferred_value: None,
            reason: offload_reason,
        },
        eviction_policy: MemoryPlanDecision {
            value: eviction,
            source: eviction_source,
            inferred_value: inferred_eviction_value,
            reason: eviction_reason,
        },
        overrides: input.overrides,
        advisory_only: input.advisory_only,
    }
}

pub(crate) fn log_memory_strategy_plan(plan: &MemoryStrategyPlan, scope: &'static str) {
    tracing::info!(
        scope,
        strategy = ?plan.strategy.value,
        strategy_source = ?plan.strategy.source,
        access_pattern = ?plan.weight_access_pattern.value,
        total_weight_bytes = plan.total_weight_bytes.value,
        exact_kv_bytes_per_token = ?plan.exact_kv_bytes_per_token.value,
        per_layer_entries = plan.per_layer_weight_bytes.value.len(),
        resolved_device_limit_bytes = ?plan.resolved_device_limit_bytes.value,
        weights_fit = ?plan.weights_fit_resolved_budget.value,
        weight_offload_enabled = plan.weight_offload_enabled.value,
        eviction_policy = ?plan.eviction_policy.value,
        advisory_only = plan.advisory_only,
        reason = %plan.strategy.reason,
        "memory strategy plan"
    );
    tracing::debug!(
        scope,
        plan = %serde_json::to_string(plan).unwrap_or_else(|_| format!("{plan:?}")),
        "memory strategy plan details"
    );
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
    let mut layers = Vec::new();
    let mut saw_dense = false;
    let mut saw_moe = false;
    let mut saw_control_flow = false;
    for node_id in order {
        let node = graph.node(node_id);
        if matches!(node.op_type.as_str(), "If" | "Loop" | "Scan") {
            saw_control_flow = true;
        }
        let Some(boundary) = LazyWeightBoundary::for_op(&node.domain, &node.op_type) else {
            continue;
        };
        let mut bytes = 0_u64;
        for value in node.input_values() {
            if let Some(weight) = graph.initializers.get(&value) {
                let Some(weight_bytes) = weight_bytes(weight) else {
                    return GraphMemoryEvidence {
                        access_pattern: WeightAccessPattern::Unknown,
                        per_layer_weight_bytes: layers,
                        reason: format!(
                            "initializer geometry overflows at node {}",
                            display_node(node)
                        ),
                    };
                };
                bytes = match bytes.checked_add(weight_bytes) {
                    Some(bytes) => bytes,
                    None => {
                        return GraphMemoryEvidence {
                            access_pattern: WeightAccessPattern::Unknown,
                            per_layer_weight_bytes: layers,
                            reason: format!(
                                "layer weight bytes overflow at node {}",
                                display_node(node)
                            ),
                        };
                    }
                };
            }
        }
        if bytes == 0 {
            continue;
        }
        match boundary {
            LazyWeightBoundary::MatMul | LazyWeightBoundary::MatMulNBits => saw_dense = true,
            LazyWeightBoundary::BlockQuantizedMoe | LazyWeightBoundary::QMoe => saw_moe = true,
        }
        layers.push(LayerWeightSize {
            node: display_node(node),
            boundary: boundary_name(boundary).to_string(),
            bytes,
        });
    }
    let (access_pattern, reason) = if saw_control_flow {
        (
            WeightAccessPattern::Unknown,
            "control-flow graph requires runtime-dependent access analysis".to_string(),
        )
    } else if saw_moe {
        (
            WeightAccessPattern::MoeRouted,
            "at least one pageable boundary uses routed MoE expert access".to_string(),
        )
    } else if saw_dense && !saw_moe {
        (
            WeightAccessPattern::SequentialDense,
            "pageable dense boundaries execute in deterministic topological order".to_string(),
        )
    } else {
        (
            WeightAccessPattern::Unknown,
            "no unambiguous dense or routed-MoE pageable boundary set was found".to_string(),
        )
    };
    GraphMemoryEvidence {
        access_pattern,
        per_layer_weight_bytes: layers,
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

fn display_node(node: &onnx_runtime_ir::Node) -> String {
    if node.name.is_empty() {
        format!("node#{}", node.id.0)
    } else {
        node.name.clone()
    }
}

fn boundary_name(boundary: LazyWeightBoundary) -> &'static str {
    match boundary {
        LazyWeightBoundary::MatMul => "mat_mul",
        LazyWeightBoundary::MatMulNBits => "mat_mul_nbits",
        LazyWeightBoundary::BlockQuantizedMoe => "block_quantized_moe",
        LazyWeightBoundary::QMoe => "qmoe",
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
        graph: GraphMemoryEvidence,
        limit: u64,
        overrides: MemoryStrategyOverrides,
    ) -> MemoryStrategyPlanInput {
        MemoryStrategyPlanInput {
            graph,
            total_weight_bytes: 128,
            exact_kv_bytes_per_token: Some(32),
            resolved_device_limit_bytes: Some(limit),
            limit_is_override: true,
            automatic_offload_allowed: true,
            compatibility_offload_enabled: false,
            overrides,
            advisory_only: false,
        }
    }

    #[test]
    fn dense_graph_selects_scan_resistant_dynamic_residency() {
        let plan = build_memory_strategy_plan(input(
            graph_with_boundary("", "MatMul"),
            64,
            MemoryStrategyOverrides::default(),
        ));
        assert_eq!(plan.strategy.value, MemoryStrategy::DynamicWeightResidency);
        assert_eq!(
            plan.weight_access_pattern.value,
            WeightAccessPattern::SequentialDense
        );
        assert_eq!(
            plan.eviction_policy.value,
            MemoryEvictionPolicy::ScanResistant
        );
        assert_eq!(plan.per_layer_weight_bytes.value[0].bytes, 64);
    }

    #[test]
    fn moe_graph_selects_routing_aware_expert_policy() {
        let plan = build_memory_strategy_plan(input(
            graph_with_boundary("com.microsoft", "QMoE"),
            64,
            MemoryStrategyOverrides::default(),
        ));
        assert_eq!(plan.strategy.value, MemoryStrategy::MoeExpertPaged);
        assert_eq!(
            plan.weight_access_pattern.value,
            WeightAccessPattern::MoeRouted
        );
        assert_eq!(
            plan.eviction_policy.value,
            MemoryEvictionPolicy::MoeRoutingAware
        );
    }

    #[test]
    fn fitting_model_is_full_resident_without_offload() {
        let plan = build_memory_strategy_plan(input(
            graph_with_boundary("", "MatMul"),
            256,
            MemoryStrategyOverrides::default(),
        ));
        assert_eq!(plan.strategy.value, MemoryStrategy::FullResident);
        assert!(!plan.weight_offload_enabled.value);
        assert_eq!(plan.weights_fit_resolved_budget.value, Some(true));
    }

    #[test]
    fn explicit_override_changes_effective_strategy_and_records_inference() {
        let inferred = build_memory_strategy_plan(input(
            graph_with_boundary("", "MatMul"),
            256,
            MemoryStrategyOverrides::default(),
        ));
        let overridden = build_memory_strategy_plan(input(
            graph_with_boundary("", "MatMul"),
            256,
            MemoryStrategyOverrides {
                weight_offload: Some(true),
                ..MemoryStrategyOverrides::default()
            },
        ));
        assert_eq!(inferred.strategy.value, MemoryStrategy::FullResident);
        assert_eq!(
            overridden.strategy.value,
            MemoryStrategy::DynamicWeightResidency
        );
        assert_eq!(
            overridden.strategy.source,
            MemoryDecisionSource::ExplicitOverride
        );
        assert_eq!(
            overridden.strategy.inferred_value,
            Some(MemoryStrategy::FullResident)
        );
        assert!(overridden.weight_offload_enabled.value);
    }

    #[test]
    fn unsupported_graph_reports_unknown_and_preserves_compatibility() {
        let mut graph = Graph::new();
        let symbol = graph.create_symbol(Some("dynamic".to_string()));
        let input = graph.create_named_value("input", DataType::Float32, vec![symbol.into()]);
        graph.add_input(input);
        graph.add_output(input);
        let plan = build_memory_strategy_plan(MemoryStrategyPlanInput {
            graph: analyze_graph_memory(&graph),
            total_weight_bytes: 128,
            exact_kv_bytes_per_token: None,
            resolved_device_limit_bytes: Some(64),
            limit_is_override: true,
            automatic_offload_allowed: true,
            compatibility_offload_enabled: true,
            overrides: MemoryStrategyOverrides {
                weight_offload: Some(true),
                ..MemoryStrategyOverrides::default()
            },
            advisory_only: false,
        });
        assert_eq!(plan.strategy.value, MemoryStrategy::Unknown);
        assert_eq!(plan.strategy.source, MemoryDecisionSource::ExplicitOverride);
        assert_eq!(plan.overrides.weight_offload, Some(true));
        assert_eq!(
            plan.weight_offload_enabled.source,
            MemoryDecisionSource::CompatibilityFallback
        );
        assert!(plan.weight_offload_enabled.value);
    }
}

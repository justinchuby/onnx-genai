use super::*;
use onnx_runtime_ep_api::{LazyWeightBoundary, lazy_weight_candidates};
use onnx_runtime_ir::{Graph, WeightRef};

/// Render an optional byte capacity/budget for a plan decision. `None` means the
/// value could not be measured and is reported as `Unknown` — never as a
/// concrete-looking number the reader would have to reverse-engineer (#947).
fn opt_bytes_or_unknown(value: Option<u64>) -> String {
    match value {
        Some(bytes) => bytes.to_string(),
        None => "Unknown".to_string(),
    }
}

/// Render an optional fit result. `None` (unknown device capacity) is reported
/// as `Unknown`, distinct from a measured `false`.
fn opt_bool_or_unknown(value: Option<bool>) -> String {
    match value {
        Some(flag) => flag.to_string(),
        None => "Unknown".to_string(),
    }
}

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
    /// The resolved *device* (VRAM) capacity limit, or `None` when the device
    /// capacity could not be measured and the limit was a fraction of it (#947:
    /// a fraction of unknown capacity is unknown, not a fabricated number). This
    /// is reported verbatim as `resolved_device_budget_bytes` and drives the
    /// `fits_resolved_device_budget` verdict; it is never borrowed from the host
    /// tier.
    pub(crate) resolved_vram_bytes: Option<u64>,
    /// The physical hot tier the weights will actually occupy, used *only* for
    /// the residency verdict (`strategy`): the measured VRAM budget when a
    /// device is queryable, else the measured host-RAM ceiling the weights live
    /// in on a device-less / ORT-CPU load. This is a different fact from device
    /// capacity -- "I do not know how big the device is" and "this model fits
    /// the memory it will really live in" can both be true -- so a fitting model
    /// reads `FullResident` instead of `Unknown` without fabricating a device
    /// number. `None` only when even the host tier could not be measured.
    pub(crate) residency_ceiling_bytes: Option<u64>,
    pub(crate) model_weight_bytes: u64,
    /// Predicted resident dequantized-f32 decode-cache expansion for this model
    /// on the native CPU EP (#971). The `MatMulNBits` generic decode path holds
    /// a full f32 dequant of its packed weight (~8x) resident for the session on
    /// the shapes no packed / on-the-fly path covers. This is the sum of that
    /// expansion, queried from the EP so the accounting cannot drift from the
    /// kernel's own dispatch (#947). Zero on backends/models that never take the
    /// f32 cache (CUDA, ORT, and — on the native CPU path — any int4 node whose
    /// dispatch avoids it, e.g. an asymmetric zero-point int4 node that takes
    /// the on-the-fly `borrowed_affine` path). The discriminator is kernel
    /// dispatch (`bits`/`accuracy_level`/`group_indices`/`m`), *not* symmetry;
    /// the EP is the sole authority so this comment stays illustrative, not a
    /// rule. The plan uses it to (a) report a truthful weight footprint and
    /// (b) govern whether the cache is affordable.
    pub(crate) resident_f32_cache_bytes: u64,
    pub(crate) kv_config: ModelKvConfig,
    pub(crate) graph: GraphMemoryEvidence,
    pub(crate) required_device_non_weight_bytes: u64,
    pub(crate) minimum_useful_weight_budget_bytes: u64,
    pub(crate) default_dynamic_device_budget_bytes: Option<u64>,
    /// The runtime activation gate. Since #755 this is set by the caller to
    /// `managed_vmm || <explicit --vram-limit>`, so inference drives policy on
    /// the no-flag path whenever the managed VMM default is active.
    pub(crate) inferred_policy_enabled: bool,
    /// True when the managed no-spill VMM path is the selected allocator for
    /// this load. Since #755 this is the default on the native CUDA path and is
    /// only cleared by the explicit legacy opt-out. It governs `managed_no_spill`
    /// (physical-handle pool + committed-granule admission, no WDDM spill) and
    /// the resolved-budget cap reported as `managed_limit_bytes`.
    pub(crate) managed_vmm: bool,
    pub(crate) overrides: MemoryStrategyOverrides,
    pub(crate) advisory_only: bool,
    /// True when the platform provides an OS shared-memory weight fallback
    /// (Windows/WDDM: "shared GPU memory" in host RAM, read in place over PCIe).
    /// #864 measured this ~30x faster than managed weight streaming for the
    /// single-touch decode access pattern, because copying a weight into VRAM
    /// only to evict it before any re-read is pure overhead. Set by the loader
    /// to `cfg!(windows)`. On Linux there is no such fallback, so an over-budget
    /// model must stream (or it does not run at all) and this stays `false`,
    /// leaving the managed path untouched (#783: do not inherit a WDDM-specific
    /// conclusion on other platforms).
    pub(crate) shared_memory_weight_fallback: bool,
    /// When `true`, force the managed weight-streaming path even where the
    /// shared-memory fallback would otherwise be auto-preferred (#864). Opt-in
    /// via `ONNX_GENAI_MANAGED_WEIGHT_STREAMING`; unrecognized values keep the
    /// faster fallback (see [`force_managed_weight_streaming_from_env_value`]).
    pub(crate) force_managed_weight_streaming: bool,
}

/// Environment knob forcing the managed weight-streaming path even where the
/// WDDM shared-memory fallback would otherwise be auto-preferred (#864).
pub(crate) const MANAGED_WEIGHT_STREAMING_ENV: &str = "ONNX_GENAI_MANAGED_WEIGHT_STREAMING";

/// Parse [`MANAGED_WEIGHT_STREAMING_ENV`]. Forcing managed streaming is an
/// opt-**in**: unset (`None`) or any unrecognized value keeps the faster WDDM
/// shared-memory fallback, and only `1`/`true`/`yes`/`on`
/// (case/whitespace-insensitive) force the managed path.
///
/// This deliberately follows the "unrecognized falls to the safe default" shape
/// of [`onnx_runtime_ep_cuda::scan_resistant_from_env_value`] rather than the
/// `ONNX_GENAI_WEIGHT_OFFLOAD_ASYNC_PAGEIN` trap, where an unrecognized value
/// silently selects the *slow* path. Here the safe default is the fast
/// fallback, so an unrecognized value must NOT force the slow managed path.
pub(crate) fn force_managed_weight_streaming_from_env_value(value: Option<&str>) -> bool {
    match value {
        Some(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        None => false,
    }
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
    // The residency verdict (`strategy`) is sized against the tier the weights
    // physically occupy -- `residency_ceiling_bytes`: measured VRAM on a
    // queryable device, else the measured host-RAM hot tier. `None` only when
    // even that tier is unmeasurable; then residency is genuinely unknown. This
    // is a *different fact* from device capacity: a box with no queryable device
    // still has its weights demonstrably resident in host RAM, and whether they
    // fit there is knowable. #947 must stop us fabricating a device size; it
    // must not stop us stating a residency we can measure.
    // #971: the native CPU `MatMulNBits` generic decode path holds a resident
    // f32 dequant cache ~8x the packed weight for the whole session. That trade
    // buys ~2.4x decode throughput when it fits and collapses to paging (~50x
    // slower) when it does not. `resident_f32_cache_bytes` is the predicted
    // expansion (0 on backends/models that never take the path). The plan
    // governs the trade here: admit the cache only when the *expanded* footprint
    // fits the residency budget, else decline it -- the kernels then dequantize
    // on the fly, so the runtime holds only the on-disk weights. The reported
    // weight figure follows the decision so it states what the runtime will
    // actually hold, not an on-disk number ~8x too small.
    let on_disk_weight_bytes = input.model_weight_bytes;
    let expanded_weight_bytes = on_disk_weight_bytes.saturating_add(input.resident_f32_cache_bytes);
    let f32_weight_cache_admitted = if input.resident_f32_cache_bytes == 0 {
        true
    } else {
        input
            .residency_ceiling_bytes
            .is_none_or(|budget| expanded_weight_bytes <= budget)
    };
    let effective_weight_bytes = if f32_weight_cache_admitted {
        expanded_weight_bytes
    } else {
        on_disk_weight_bytes
    };
    let fits = input
        .residency_ceiling_bytes
        .map(|budget| effective_weight_bytes <= budget);
    // The *reported* device-budget fit is a strictly device-capacity fact and
    // stays `None` when the device capacity could not be measured -- it never
    // borrows the host-tier ceiling. Rendering unknown device capacity as
    // `Some(false)` (or as a fabricated number) is exactly the #947 bug.
    let device_fits = input
        .resolved_vram_bytes
        .map(|budget| effective_weight_bytes <= budget);
    let available_weight_budget_bytes = input
        .resolved_vram_bytes
        .map(|budget| budget.saturating_sub(input.required_device_non_weight_bytes));
    // Since #755 the managed no-spill VMM path is the default: it owns the
    // authority-scoped physical-handle pool and caps at the resolved budget
    // instead of relying on WDDM shared-memory spill. Reported in the plan so
    // the new default is observable rather than implicit.
    let managed_no_spill = input.managed_vmm;
    let managed_limit_bytes = if managed_no_spill {
        input.resolved_vram_bytes
    } else {
        None
    };
    let inferred_strategy = if kv_unknown {
        MemoryStrategy::Unknown
    } else {
        match (fits, input.graph.access_pattern) {
            (_, WeightAccessPattern::Unknown) => MemoryStrategy::Unknown,
            // Even the residency tier could not be measured: residency cannot be
            // chosen honestly, so it is genuinely unknown (not a false "fit").
            (None, _) => MemoryStrategy::Unknown,
            (Some(true), _) => MemoryStrategy::FullResident,
            (Some(false), WeightAccessPattern::SequentialDense) => {
                MemoryStrategy::DynamicWeightResidency
            }
            (Some(false), WeightAccessPattern::MoeRouted) => MemoryStrategy::MoeRoutingAware,
            (Some(false), WeightAccessPattern::Iterative) => MemoryStrategy::Unknown,
        }
    };

    let forced_offload = input.overrides.weight_offload == Some(true);
    let explicit_vram_limit = matches!(input.config.limits.vram_limit, ResourceLimit::Bytes(_));
    let device_budget_override = input.overrides.device_budget_bytes.is_some();
    // #864: over-budget weights would auto-enable managed weight streaming here.
    // But on WDDM the OS pages non-resident weights from "shared GPU memory"
    // (host RAM) in place over PCIe, which for the single-touch decode access
    // pattern is ~30x faster than copying each weight into VRAM only to evict it
    // before any re-read. Only the *inferred* default is affected: an explicit
    // ONNX_GENAI_WEIGHT_OFFLOAD, --vram-limit, or device-budget override still
    // selects managed streaming (they are honored, not overridden), as does the
    // ONNX_GENAI_MANAGED_WEIGHT_STREAMING force knob.
    let inferred_over_budget_streaming = matches!(
        inferred_strategy,
        MemoryStrategy::DynamicWeightResidency | MemoryStrategy::MoeRoutingAware
    ) && input.inferred_policy_enabled
        && !forced_offload
        && !device_budget_override
        && !explicit_vram_limit;
    // Gated on the platform fallback (Windows/WDDM). On Linux this stays false,
    // so nothing below changes managed streaming there (#783).
    let prefer_shared_memory_fallback = inferred_over_budget_streaming
        && input.shared_memory_weight_fallback
        && !input.force_managed_weight_streaming;
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
    } else if prefer_shared_memory_fallback {
        (
            MemoryStrategy::Compatibility,
            DecisionSource::CompatibilityDefault,
            "weights exceed the device budget, but on WDDM the OS shared-memory fallback pages \
             them from host RAM over PCIe faster than managed streaming for the single-touch \
             decode pattern (#864 measured ~30x on medians), so managed weight streaming stays \
             off by default; set ONNX_GENAI_MANAGED_WEIGHT_STREAMING=1 to force it",
        )
    } else if matches!(
        inferred_strategy,
        MemoryStrategy::DynamicWeightResidency | MemoryStrategy::MoeRoutingAware
    ) && !input.inferred_policy_enabled
    {
        (
            MemoryStrategy::Compatibility,
            DecisionSource::CompatibilityDefault,
            "inference runs, but the managed VMM default is disabled (legacy allocator opt-out) \
             and no explicit --vram-limit is set, so paging stays off",
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
    } else if prefer_shared_memory_fallback {
        // #864: hand residency to the WDDM shared-memory fallback. This is the
        // exact application the measured WDDM arm reports — offload off, managed
        // no-spill off (so the physical-handle pool does not cap the load below
        // the weights and refuse them), no governed device budget. Residency
        // becomes the OS's job; the trade is stated in the plan decision below.
        wddm_shared_memory_application(&input)
    } else {
        match strategy {
            MemoryStrategy::FullResident => MemoryPolicyApplication {
                weight_offload_enabled: false,
                device_budget_bytes: None,
                scan_resistant_dense: input.overrides.scan_resistant_dense.unwrap_or(true),
                managed_no_spill,
                managed_limit_bytes,
                device_budget_is_override: false,
                auto_enabled_from_vram_limit: false,
            },
            MemoryStrategy::DynamicWeightResidency | MemoryStrategy::MoeRoutingAware => {
                MemoryPolicyApplication {
                    weight_offload_enabled: true,
                    device_budget_bytes: input.overrides.device_budget_bytes.or({
                        if forced_offload
                            && !matches!(input.config.limits.vram_limit, ResourceLimit::Bytes(_))
                            && !input.managed_vmm
                        {
                            input.default_dynamic_device_budget_bytes
                        } else {
                            available_weight_budget_bytes
                        }
                    }),
                    scan_resistant_dense: input.overrides.scan_resistant_dense.unwrap_or(true),
                    managed_no_spill,
                    managed_limit_bytes,
                    device_budget_is_override: input.overrides.device_budget_bytes.is_some(),
                    auto_enabled_from_vram_limit: !forced_offload
                        && (input.managed_vmm
                            || matches!(input.config.limits.vram_limit, ResourceLimit::Bytes(_))),
                }
            }
            MemoryStrategy::Compatibility | MemoryStrategy::Unknown => compatibility_application,
        }
    };

    let mut decisions = vec![
        MemoryStrategyDecision::new(
            "resolved_device_budget_bytes",
            opt_bytes_or_unknown(input.resolved_vram_bytes),
            match (input.resolved_vram_bytes, input.config.limits.vram_limit) {
                // No capacity could be measured and no explicit byte limit was
                // set: the budget is genuinely unavailable, not inferred.
                (None, _) => DecisionSource::Unavailable,
                (Some(_), ResourceLimit::Bytes(_)) => DecisionSource::ExplicitOverride,
                (Some(_), _) => DecisionSource::Inference,
            },
            match input.resolved_vram_bytes {
                Some(_) => "resolved once through the same capacity helper used by the governor",
                None => {
                    "device capacity could not be measured; a fraction of unknown capacity is \
                     unknown, so no device budget is resolved"
                }
            },
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
            effective_weight_bytes.to_string(),
            DecisionSource::Inference,
            "on-disk package weight plus the resident f32 decode-cache expansion the runtime will \
             actually hold (#971); reverts to on-disk when the cache is declined",
            format!(
                "on_disk_weight_bytes={} resident_f32_cache_bytes={} f32_weight_cache_admitted={} fits_resolved_device_budget={}",
                on_disk_weight_bytes,
                input.resident_f32_cache_bytes,
                f32_weight_cache_admitted,
                opt_bool_or_unknown(device_fits)
            ),
        ),
        MemoryStrategyDecision::new(
            "available_weight_budget_bytes",
            opt_bytes_or_unknown(available_weight_budget_bytes),
            match available_weight_budget_bytes {
                Some(_) => DecisionSource::Inference,
                None => DecisionSource::Unavailable,
            },
            "derived before provider construction from runtime-owned non-weight geometry",
            format!(
                "resolved_device_budget_bytes={} required_device_non_weight_bytes={} minimum_useful_weight_budget_bytes={}",
                opt_bytes_or_unknown(input.resolved_vram_bytes),
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
            "inferred unconditionally from graph evidence and the residency tier",
            format!(
                "total_weight_bytes={} residency_ceiling_bytes={} fits={}",
                effective_weight_bytes,
                opt_bytes_or_unknown(input.residency_ceiling_bytes),
                opt_bool_or_unknown(fits)
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

    // #971: surface the governed f32 decode-cache decision wherever the plan is
    // read. A user who is ~2.4x slower than expected must be able to see that the
    // runtime declined the resident dequant cache because its expanded footprint
    // would not fit the budget -- the same visibility rationale as
    // `advisory_only`/#955. Only emitted when a f32 expansion is actually in play.
    if input.resident_f32_cache_bytes > 0 {
        let (value, reason): (&str, String) = if f32_weight_cache_admitted {
            (
                "admitted",
                format!(
                    "the resident f32 decode cache fits the budget, so it is taken for ~2.4x \
                     faster decode; it adds {} resident bytes on top of the {} on-disk weights",
                    input.resident_f32_cache_bytes, on_disk_weight_bytes
                ),
            )
        } else {
            (
                "declined",
                format!(
                    "the expanded footprint ({expanded_weight_bytes} bytes) exceeds the residency \
                     budget ({}), so the runtime skips the resident f32 decode cache and \
                     dequantizes on the fly -- slower per token, but avoids the ~8x resident \
                     blow-up that would page the box (#971)",
                    opt_bytes_or_unknown(input.residency_ceiling_bytes)
                ),
            )
        };
        decisions.push(MemoryStrategyDecision::new(
            "resident_f32_weight_cache",
            value,
            DecisionSource::Inference,
            reason,
            format!(
                "on_disk_weight_bytes={on_disk_weight_bytes} resident_f32_cache_bytes={} \
                 expanded_weight_bytes={expanded_weight_bytes} residency_ceiling_bytes={}",
                input.resident_f32_cache_bytes,
                opt_bytes_or_unknown(input.residency_ceiling_bytes)
            ),
        ));
    }

    if inferred_over_budget_streaming && input.shared_memory_weight_fallback {
        let (value, source, reason): (&str, DecisionSource, &str) = if prefer_shared_memory_fallback
        {
            (
                "shared_memory_fallback_preferred",
                DecisionSource::CompatibilityDefault,
                "#864: WDDM demand-pages over-budget weights from host RAM over PCIe ~30x faster \
                 than managed streaming for the single-touch decode pattern, so managed weight \
                 streaming was auto-disabled. Residency is now managed by the OS; on a host with \
                 little free RAM an over-budget model may thrash. Set \
                 ONNX_GENAI_MANAGED_WEIGHT_STREAMING=1 to force managed streaming.",
            )
        } else {
            (
                "managed_streaming_forced",
                DecisionSource::ExplicitOverride,
                "ONNX_GENAI_MANAGED_WEIGHT_STREAMING forced managed weight streaming despite the \
                 WDDM shared-memory fallback that #864 measured ~30x faster.",
            )
        };
        decisions.push(MemoryStrategyDecision::new(
            "weight_streaming_platform_policy",
            value,
            source,
            reason,
            format!(
                "shared_memory_weight_fallback={} force_managed_weight_streaming={} total_weight_bytes={} resolved_device_budget_bytes={}",
                input.shared_memory_weight_fallback,
                input.force_managed_weight_streaming,
                effective_weight_bytes,
                opt_bytes_or_unknown(input.resolved_vram_bytes)
            ),
        ));
    }

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
        total_weight_bytes: effective_weight_bytes,
        resident_f32_cache_bytes: input.resident_f32_cache_bytes,
        f32_weight_cache_admitted,
        kv_bytes_per_token,
        per_layer_weight_bytes: input.graph.per_layer_weight_bytes,
        resolved_device_budget_bytes: input.resolved_vram_bytes,
        fits_resolved_device_budget: device_fits,
        application,
        advisory_only: input.advisory_only,
        decisions,
    }
}

/// #864: the application selected when the WDDM shared-memory fallback is
/// preferred over managed weight streaming. Offload is off (the OS pages the
/// weights), managed no-spill is off (so the physical-handle pool does not cap
/// the load below the weight bytes and refuse them — the whole point is to let
/// WDDM hold the over-budget remainder in host RAM), and no governed device
/// budget is enforced. This mirrors the arm #864 measured as ~30x faster.
fn wddm_shared_memory_application(input: &MemoryStrategyPlanInput<'_>) -> MemoryPolicyApplication {
    MemoryPolicyApplication {
        weight_offload_enabled: false,
        device_budget_bytes: None,
        scan_resistant_dense: input.overrides.scan_resistant_dense.unwrap_or(true),
        managed_no_spill: false,
        managed_limit_bytes: None,
        device_budget_is_override: false,
        auto_enabled_from_vram_limit: false,
    }
}

fn compatibility_application(
    input: &MemoryStrategyPlanInput<'_>,
    fits: Option<bool>,
) -> MemoryPolicyApplication {
    let explicit_bytes = matches!(input.config.limits.vram_limit, ResourceLimit::Bytes(_));
    let forced = input.overrides.weight_offload == Some(true);
    // The compatibility/Unknown application cannot page safely (no known weight
    // access boundary), so automatic offload stays keyed on an explicit byte
    // limit under which the weights demonstrably do not fit. When the device
    // capacity is unknown, `fits` is `None` and auto-offload stays off — the
    // whole point of #947 is that unknown must not be read as "does not fit".
    let auto_enabled = explicit_bytes && fits == Some(false) && !forced;
    let enabled = forced || auto_enabled;
    MemoryPolicyApplication {
        weight_offload_enabled: enabled,
        device_budget_bytes: if enabled {
            input
                .overrides
                .device_budget_bytes
                .or_else(|| {
                    auto_enabled
                        .then(|| {
                            input.resolved_vram_bytes.map(|budget| {
                                budget.saturating_sub(input.required_device_non_weight_bytes)
                            })
                        })
                        .flatten()
                })
                .or(input.default_dynamic_device_budget_bytes)
        } else {
            None
        },
        scan_resistant_dense: input.overrides.scan_resistant_dense.unwrap_or(true),
        managed_no_spill: input.managed_vmm,
        managed_limit_bytes: if input.managed_vmm {
            input.resolved_vram_bytes
        } else {
            None
        },
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
        resident_f32_cache_bytes = plan.resident_f32_cache_bytes,
        f32_weight_cache_admitted = plan.f32_weight_cache_admitted,
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
    if let Some(decision) = plan
        .decisions
        .iter()
        .find(|decision| decision.field == "weight_streaming_platform_policy")
    {
        tracing::warn!(
            scope,
            policy = %decision.value,
            weight_offload_enabled = plan.runtime_application().weight_offload_enabled,
            managed_no_spill = plan.runtime_application().managed_no_spill,
            total_weight_bytes = plan.total_weight_bytes,
            resolved_device_budget_bytes = ?plan.resolved_device_budget_bytes,
            evidence = %decision.evidence,
            "{}",
            decision.reason
        );
    }
    warn_if_budget_is_not_enforced(scope, plan);
}

/// Say, before the run rather than after it, that the memory budget this backend
/// just reported is one it will not apply.
///
/// `advisory_only` is literally accurate — the plan is computed, logged, and
/// consumed by nothing — but it has been a silent property of the code rather
/// than a declared property of the backend. A user who passed a limit was shown
/// the limit, shown their weights as a percentage of it, and never told that the
/// percentage could exceed 100 without anything happening (#955). Reporting a
/// bound we have handed away is the whole of the user-visible harm, and saying so
/// costs a line.
fn warn_if_budget_is_not_enforced(scope: &str, plan: &MemoryStrategyPlan) {
    let Some((budget, percent)) = unenforced_budget_overrun(plan) else {
        return;
    };
    tracing::warn!(
        scope,
        total_weight_bytes = plan.total_weight_bytes,
        resolved_device_budget_bytes = budget,
        weight_percent_of_budget = format!("{percent:.1}%"),
        "this backend does not enforce the memory budget: model weights are \
         {percent:.1}% of the resolved budget and nothing will constrain them. \
         The plan below is advisory. Weight streaming is implemented on the \
         native CUDA path; on this backend the limit is reported but not applied, \
         so the KV cache is sized against a weight reservation that is never \
         honoured (#955)."
    );
}

/// `Some((budget, percent))` when the plan reports a budget the backend will not
/// apply *and* the weights exceed it.
///
/// Deliberately silent when the budget is merely unenforced without being
/// exceeded: that costs the user nothing, and warning on every run would train
/// people to ignore the message that matters.
fn unenforced_budget_overrun(plan: &MemoryStrategyPlan) -> Option<(u64, f64)> {
    if !plan.advisory_only {
        return None;
    }
    let budget = plan.resolved_device_budget_bytes?;
    if plan.fits_resolved_device_budget != Some(false) {
        return None;
    }
    let percent = if budget > 0 {
        (plan.total_weight_bytes as f64 / budget as f64) * 100.0
    } else {
        f64::INFINITY
    };
    Some((budget, percent))
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
        // Byte-aware residency is an opt-in A/B knob (#837 item 3), not a
        // memory-strategy decision, so it is read straight from the environment
        // rather than threaded through the runtime-application config.
        byte_aware_residency: onnx_runtime_ep_cuda::byte_aware_residency_from_env(),
        // Eviction-order probe (#888): default LRU (byte-identical to shipped),
        // read straight from the environment like the byte-aware knob above so
        // the eviction-order investigation needs no runtime-application field.
        evict_order_probe: onnx_runtime_ep_cuda::evict_order_probe_from_env(),
        // Zero-copy hybrid (#864): opt-in A/B knob read straight from the
        // environment, like the byte-aware and eviction-order knobs above, so it
        // needs no runtime-application field. Inert unless managed streaming is
        // active (offload on + VMM stable-VA).
        zero_copy_hybrid: onnx_runtime_ep_cuda::zero_copy_hybrid_from_env(),
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

    /// A plan shaped like the one a user gets on the ORT path: the budget is
    /// resolved and reported, and `advisory_only` means nothing will apply it.
    fn advisory_plan(weights: u64, budget: Option<u64>, fits: Option<bool>) -> MemoryStrategyPlan {
        let mut plan = MemoryStrategyPlan::unknown(weights, Some(196_608), "test");
        assert!(
            plan.advisory_only,
            "unknown() is expected to be advisory; this test depends on it"
        );
        plan.resolved_device_budget_bytes = budget;
        plan.fits_resolved_device_budget = fits;
        plan
    }

    /// The exact case reported in #955: a 3 GiB limit, 8,330,595,870 bytes of
    /// weights, and the runtime cheerfully printing 258.6% while doing nothing.
    #[test]
    fn unenforced_budget_overrun_reports_the_reported_percentage() {
        let plan = advisory_plan(8_330_595_870, Some(3 << 30), Some(false));
        let (budget, percent) = unenforced_budget_overrun(&plan)
            .expect("an advisory budget that the weights exceed must be reported");
        assert_eq!(budget, 3 << 30);
        assert!(
            (percent - 258.6).abs() < 0.1,
            "expected the 258.6% from #955, got {percent}"
        );
    }

    #[test]
    fn an_unenforced_budget_that_is_not_exceeded_stays_quiet() {
        // Warning on every advisory run would train users to ignore the one
        // message that matters.
        let plan = advisory_plan(1 << 30, Some(3 << 30), Some(true));
        assert!(unenforced_budget_overrun(&plan).is_none());
    }

    #[test]
    fn an_enforcing_backend_does_not_warn_even_when_over_budget() {
        // Native CUDA streams the overflow; being over budget is the normal,
        // handled case there, not a defect to report.
        let mut plan = advisory_plan(8_330_595_870, Some(3 << 30), Some(false));
        plan.advisory_only = false;
        assert!(unenforced_budget_overrun(&plan).is_none());
    }

    #[test]
    fn an_unmeasured_device_budget_does_not_warn() {
        // After #947 an unknown capacity resolves to None. There is no budget to
        // have violated, so there is nothing to say.
        let plan = advisory_plan(8_330_595_870, None, None);
        assert!(unenforced_budget_overrun(&plan).is_none());
    }

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
        plan_with_managed(
            config, graph, limit, weights, overrides, false, false, false,
        )
    }

    /// Build a plan as the native CUDA loader does under the #755 managed VMM
    /// default: inference drives policy and the managed no-spill path is on.
    /// Models a platform WITHOUT the WDDM shared-memory fallback (e.g. Linux),
    /// so #864's auto-disable does not fire and over-budget models still stream.
    fn input_managed(
        config: &EngineConfig,
        graph: GraphMemoryEvidence,
        limit: u64,
        weights: u64,
        overrides: MemoryStrategyOverrides,
    ) -> MemoryStrategyPlan {
        plan_with_managed(config, graph, limit, weights, overrides, true, false, false)
    }

    /// As [`input_managed`], but models a platform WITH the WDDM shared-memory
    /// weight fallback available (Windows). #864's auto-disable applies here.
    fn input_managed_wddm(
        config: &EngineConfig,
        graph: GraphMemoryEvidence,
        limit: u64,
        weights: u64,
        overrides: MemoryStrategyOverrides,
    ) -> MemoryStrategyPlan {
        plan_with_managed(config, graph, limit, weights, overrides, true, true, false)
    }

    /// As [`input_managed_wddm`], but with `ONNX_GENAI_MANAGED_WEIGHT_STREAMING`
    /// forcing the managed path despite the fallback.
    fn input_managed_wddm_forced(
        config: &EngineConfig,
        graph: GraphMemoryEvidence,
        limit: u64,
        weights: u64,
        overrides: MemoryStrategyOverrides,
    ) -> MemoryStrategyPlan {
        plan_with_managed(config, graph, limit, weights, overrides, true, true, true)
    }

    #[allow(clippy::too_many_arguments)]
    fn plan_with_managed(
        config: &EngineConfig,
        graph: GraphMemoryEvidence,
        limit: u64,
        weights: u64,
        overrides: MemoryStrategyOverrides,
        managed_default: bool,
        shared_memory_weight_fallback: bool,
        force_managed_weight_streaming: bool,
    ) -> MemoryStrategyPlan {
        let explicit_bytes = matches!(config.limits.vram_limit, ResourceLimit::Bytes(_));
        let managed_vmm = managed_default || explicit_bytes;
        build_memory_strategy_plan(MemoryStrategyPlanInput {
            config,
            resolved_vram_bytes: Some(limit),
            residency_ceiling_bytes: Some(limit),
            model_weight_bytes: weights,
            resident_f32_cache_bytes: 0,
            kv_config: ModelKvConfig::known(160, 16),
            graph,
            required_device_non_weight_bytes: 0,
            minimum_useful_weight_budget_bytes: 0,
            default_dynamic_device_budget_bytes: None,
            inferred_policy_enabled: managed_vmm || explicit_bytes,
            managed_vmm,
            overrides,
            advisory_only: false,
            shared_memory_weight_fallback,
            force_managed_weight_streaming,
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

    /// Build a plan with an *unmeasured* device capacity, as the reported
    /// Windows-ARM / no-NVIDIA machine does: the default fractional VRAM limit
    /// resolves to `None` because a fraction of unknown capacity is unknown.
    fn plan_unknown_device(
        config: &EngineConfig,
        graph: GraphMemoryEvidence,
        weights: u64,
        overrides: MemoryStrategyOverrides,
    ) -> MemoryStrategyPlan {
        let explicit_bytes = matches!(config.limits.vram_limit, ResourceLimit::Bytes(_));
        build_memory_strategy_plan(MemoryStrategyPlanInput {
            config,
            resolved_vram_bytes: None,
            residency_ceiling_bytes: None,
            model_weight_bytes: weights,
            resident_f32_cache_bytes: 0,
            kv_config: ModelKvConfig::known(160, 16),
            graph,
            required_device_non_weight_bytes: 0,
            minimum_useful_weight_budget_bytes: 0,
            default_dynamic_device_budget_bytes: None,
            inferred_policy_enabled: explicit_bytes,
            managed_vmm: explicit_bytes,
            overrides,
            advisory_only: true,
            shared_memory_weight_fallback: true,
            force_managed_weight_streaming: false,
        })
    }

    /// Build a plan as the real Windows-ARM / no-NVIDIA single-model load does:
    /// the device (VRAM) capacity is unmeasured (`None`), but host RAM -- the
    /// tier the weights physically occupy -- IS measured, so the residency
    /// ceiling is `Some`.
    fn plan_unknown_device_host_resident(
        config: &EngineConfig,
        graph: GraphMemoryEvidence,
        host_ceiling: u64,
        weights: u64,
    ) -> MemoryStrategyPlan {
        build_memory_strategy_plan(MemoryStrategyPlanInput {
            config,
            resolved_vram_bytes: None,
            residency_ceiling_bytes: Some(host_ceiling),
            model_weight_bytes: weights,
            resident_f32_cache_bytes: 0,
            kv_config: ModelKvConfig::known(160, 16),
            graph,
            required_device_non_weight_bytes: 0,
            minimum_useful_weight_budget_bytes: 0,
            default_dynamic_device_budget_bytes: None,
            inferred_policy_enabled: false,
            managed_vmm: false,
            overrides: MemoryStrategyOverrides::default(),
            advisory_only: true,
            shared_memory_weight_fallback: true,
            force_managed_weight_streaming: false,
        })
    }

    #[test]
    fn resident_f32_cache_admitted_when_expanded_footprint_fits() {
        // #971: when a native CPU model takes the resident f32 decode cache and
        // the *expanded* footprint fits the residency budget, the plan admits
        // the cache and reports the expanded weight figure -- not the on-disk
        // size that is ~8x too small. The user sees the truthful resident bytes.
        let config = EngineConfig::default();
        let on_disk = 353_000_000_u64;
        let cache = 1_975_844_864_u64; // measured f32 expansion for qwen05b-q4
        let budget = 8_000_000_000_u64; // comfortably above on_disk + cache
        let plan = build_memory_strategy_plan(MemoryStrategyPlanInput {
            config: &config,
            resolved_vram_bytes: Some(budget),
            residency_ceiling_bytes: Some(budget),
            model_weight_bytes: on_disk,
            resident_f32_cache_bytes: cache,
            kv_config: ModelKvConfig::known(160, 16),
            graph: graph_with_boundary("com.microsoft", "MatMulNBits"),
            required_device_non_weight_bytes: 0,
            minimum_useful_weight_budget_bytes: 0,
            default_dynamic_device_budget_bytes: None,
            inferred_policy_enabled: true,
            managed_vmm: true,
            overrides: MemoryStrategyOverrides::default(),
            advisory_only: false,
            shared_memory_weight_fallback: false,
            force_managed_weight_streaming: false,
        });
        assert!(
            plan.f32_weight_cache_admitted,
            "the cache must be admitted when the expanded footprint fits"
        );
        assert_eq!(plan.resident_f32_cache_bytes, cache);
        assert_eq!(
            plan.total_weight_bytes,
            on_disk + cache,
            "the reported weight figure must include the admitted f32 cache"
        );
        assert!(
            plan.decisions
                .iter()
                .any(|d| d.field == "resident_f32_weight_cache" && d.value == "admitted"),
            "an admitted decision must be reported for visibility"
        );
    }

    #[test]
    fn resident_f32_cache_declined_when_expanded_footprint_exceeds_budget() {
        // #971: when the expanded footprint would not fit the residency budget
        // the plan declines the cache (kernels dequantize on the fly), and the
        // reported weight figure falls back to the on-disk size the runtime
        // will actually hold -- not the expansion it refused to build.
        let config = EngineConfig::default();
        let on_disk = 353_000_000_u64;
        let cache = 1_975_844_864_u64;
        let budget = 1_073_741_824_u64; // 1 GiB -- below on_disk + cache
        let plan = build_memory_strategy_plan(MemoryStrategyPlanInput {
            config: &config,
            resolved_vram_bytes: Some(budget),
            residency_ceiling_bytes: Some(budget),
            model_weight_bytes: on_disk,
            resident_f32_cache_bytes: cache,
            kv_config: ModelKvConfig::known(160, 16),
            graph: graph_with_boundary("com.microsoft", "MatMulNBits"),
            required_device_non_weight_bytes: 0,
            minimum_useful_weight_budget_bytes: 0,
            default_dynamic_device_budget_bytes: None,
            inferred_policy_enabled: true,
            managed_vmm: true,
            overrides: MemoryStrategyOverrides::default(),
            advisory_only: false,
            shared_memory_weight_fallback: false,
            force_managed_weight_streaming: false,
        });
        assert!(
            !plan.f32_weight_cache_admitted,
            "the cache must be declined when the expanded footprint exceeds the budget"
        );
        assert_eq!(
            plan.total_weight_bytes, on_disk,
            "a declined cache must report the on-disk weight the runtime actually holds"
        );
        assert!(
            plan.decisions
                .iter()
                .any(|d| d.field == "resident_f32_weight_cache" && d.value == "declined"),
            "a declined decision must be reported so the user sees why decode is slower"
        );
    }

    #[test]
    fn zero_f32_cache_always_admits_and_reports_on_disk_weight() {
        // Backends/models that never take the f32 path (ORT, CUDA, or a native
        // CPU int4 node whose dispatch avoids the resident cache) pass zero cache
        // bytes: the decision defaults to admitted, the reported weight is the
        // on-disk size, and no cache decision is emitted.
        let config = EngineConfig::default();
        let on_disk = 362_000_000_u64;
        let plan = build_memory_strategy_plan(MemoryStrategyPlanInput {
            config: &config,
            resolved_vram_bytes: Some(500_000_000),
            residency_ceiling_bytes: Some(500_000_000),
            model_weight_bytes: on_disk,
            resident_f32_cache_bytes: 0,
            kv_config: ModelKvConfig::known(160, 16),
            graph: graph_with_boundary("com.microsoft", "MatMulNBits"),
            required_device_non_weight_bytes: 0,
            minimum_useful_weight_budget_bytes: 0,
            default_dynamic_device_budget_bytes: None,
            inferred_policy_enabled: true,
            managed_vmm: true,
            overrides: MemoryStrategyOverrides::default(),
            advisory_only: false,
            shared_memory_weight_fallback: false,
            force_managed_weight_streaming: false,
        });
        assert!(plan.f32_weight_cache_admitted);
        assert_eq!(plan.total_weight_bytes, on_disk);
        assert!(
            !plan
                .decisions
                .iter()
                .any(|d| d.field == "resident_f32_weight_cache"),
            "no cache decision should be emitted when there is no f32 expansion in play"
        );
    }

    #[test]
    fn unmeasured_device_still_reports_host_resident_fit() {
        // #947 follow-up: making an unmeasured device *capacity* honest (`None`)
        // must not swing us to refusing a residency we can measure. On the real
        // Windows-ARM box the weights (359 MB) plainly fit the measured host RAM
        // (68 GB) they will live in, so the residency verdict is `FullResident`.
        // "I do not know the device size" and "this model fits the memory it
        // will really live in" are different claims: the first keeps
        // `resolved_device_budget_bytes` at `None` (no fabricated device
        // number), the second is stated as the strategy.
        let config = EngineConfig::default();
        let plan = plan_unknown_device_host_resident(
            &config,
            graph_with_boundary("", "MatMul"),
            68_535_443_456,
            359_107_027,
        );
        assert_eq!(
            plan.strategy,
            MemoryStrategy::FullResident,
            "a model that fits the measured host tier must read as resident, not Unknown"
        );
        assert_eq!(
            plan.resolved_device_budget_bytes, None,
            "the device capacity is still unmeasured; no device number may be fabricated"
        );
        assert_eq!(
            plan.fits_resolved_device_budget, None,
            "the device-budget fit stays unknown -- it is a device-capacity fact, not a host one"
        );
        assert!(
            !plan.application.weight_offload_enabled,
            "a resident model on the advisory-only host path must not page"
        );
    }

    #[test]
    fn unknown_device_capacity_is_reported_as_unknown_not_a_false_fit() {
        // #947 regression guard. On `main` an unmeasured device capacity is
        // fabricated to 8 GiB and a model larger than 7.73 GiB is reported as
        // `fits=Some(false)` against a device that does not exist. The honest
        // answer is "unknown": no resolved device budget, no fit verdict.
        let config = EngineConfig::default();
        let plan = plan_unknown_device(
            &config,
            graph_with_boundary("", "MatMul"),
            8_330_595_020,
            MemoryStrategyOverrides::default(),
        );
        assert_eq!(
            plan.resolved_device_budget_bytes, None,
            "an unmeasured device capacity must not resolve to a fabricated budget"
        );
        assert_eq!(
            plan.fits_resolved_device_budget, None,
            "unknown capacity must render as unknown fit, never Some(false)"
        );
        // The unactionable "raise the VRAM limit" pressure must not appear: with
        // no device ceiling there is nothing to raise. Auto-offload stays off.
        assert!(!plan.application.weight_offload_enabled);
        let budget_decision = plan
            .decisions
            .iter()
            .find(|decision| decision.field == "resolved_device_budget_bytes")
            .expect("plan reports the resolved device budget decision");
        assert_eq!(budget_decision.value, "Unknown");
        assert_eq!(budget_decision.source, DecisionSource::Unavailable);
    }

    #[test]
    fn legacy_opt_out_no_flag_keeps_compatibility_gate() {
        // With the managed VMM default disabled (legacy allocator opt-out) and no
        // explicit --vram-limit, inference still runs but paging stays off. `input`
        // models managed_vmm=false because the default config carries no byte limit.
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
        assert!(!plan.application.managed_no_spill);
    }

    #[test]
    fn managed_default_no_flag_fitting_model_is_full_resident_without_paging() {
        // #755: a model that fits the resolved default budget must stay fully
        // resident with offload OFF even though managed VMM is now the default.
        let config = EngineConfig::default();
        let plan = input_managed(
            &config,
            graph_with_boundary("", "MatMul"),
            256,
            128,
            MemoryStrategyOverrides::default(),
        );
        assert_eq!(plan.inferred_strategy, MemoryStrategy::FullResident);
        assert_eq!(plan.strategy, MemoryStrategy::FullResident);
        assert!(!plan.application.weight_offload_enabled);
        assert!(
            plan.application.managed_no_spill,
            "managed VMM must be the default allocator without a flag"
        );
        assert_eq!(plan.application.managed_limit_bytes, Some(256));
    }

    #[test]
    fn managed_default_no_flag_over_budget_model_auto_streams() {
        // #755: over-budget under the managed default automatically enables weight
        // streaming instead of failing, with no explicit --vram-limit set. This
        // is the behaviour on a platform WITHOUT the WDDM shared-memory fallback
        // (e.g. Linux): #864's auto-disable does not apply, so managed streaming
        // stays the only way to run an over-budget model. `input_managed` models
        // `shared_memory_weight_fallback = false`.
        let config = EngineConfig::default();
        let plan = input_managed(
            &config,
            graph_with_boundary("", "MatMul"),
            64,
            128,
            MemoryStrategyOverrides::default(),
        );
        assert_eq!(plan.strategy, MemoryStrategy::DynamicWeightResidency);
        assert!(plan.application.weight_offload_enabled);
        assert!(plan.application.managed_no_spill);
        assert!(plan.application.auto_enabled_from_vram_limit);
        assert_eq!(plan.application.device_budget_bytes, Some(64));
        assert_eq!(plan.application.managed_limit_bytes, Some(64));
        assert!(
            !plan
                .decisions
                .iter()
                .any(|d| d.field == "weight_streaming_platform_policy"),
            "no WDDM policy decision without the shared-memory fallback: {plan:?}"
        );
    }

    #[test]
    fn managed_default_over_budget_moe_auto_streams() {
        // Non-WDDM (Linux) path, as `managed_default_no_flag_over_budget_model_auto_streams`.
        let config = EngineConfig::default();
        let plan = input_managed(
            &config,
            graph_with_boundary("com.microsoft", "QMoE"),
            64,
            128,
            MemoryStrategyOverrides::default(),
        );
        assert_eq!(plan.strategy, MemoryStrategy::MoeRoutingAware);
        assert!(plan.application.weight_offload_enabled);
        assert!(plan.application.managed_no_spill);
    }

    #[test]
    fn wddm_over_budget_prefers_shared_memory_and_disables_streaming() {
        // #864: on WDDM the auto-enabled managed streaming default is ~30x slower
        // than letting the OS page over-budget weights from host RAM. The inferred
        // strategy is still DynamicWeightResidency, but the effective strategy
        // becomes Compatibility with offload OFF, managed no-spill OFF (so the
        // physical pool does not cap and refuse the over-budget weights), and no
        // governed device budget — the exact arm #864 measured as faster.
        let config = EngineConfig::default();
        let plan = input_managed_wddm(
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
        assert!(!plan.application.managed_no_spill);
        assert_eq!(plan.application.device_budget_bytes, None);
        assert_eq!(plan.application.managed_limit_bytes, None);
        assert!(!plan.application.auto_enabled_from_vram_limit);
        let decision = plan
            .decisions
            .iter()
            .find(|d| d.field == "weight_streaming_platform_policy")
            .expect("WDDM policy decision must be recorded loudly");
        assert_eq!(decision.value, "shared_memory_fallback_preferred");
        assert_eq!(decision.source, DecisionSource::CompatibilityDefault);
    }

    #[test]
    fn wddm_over_budget_moe_prefers_shared_memory() {
        let config = EngineConfig::default();
        let plan = input_managed_wddm(
            &config,
            graph_with_boundary("com.microsoft", "QMoE"),
            64,
            128,
            MemoryStrategyOverrides::default(),
        );
        assert_eq!(plan.inferred_strategy, MemoryStrategy::MoeRoutingAware);
        assert_eq!(plan.strategy, MemoryStrategy::Compatibility);
        assert!(!plan.application.weight_offload_enabled);
        assert!(!plan.application.managed_no_spill);
    }

    #[test]
    fn wddm_fitting_model_is_unaffected_and_stays_full_resident() {
        // The #864 auto-disable is scoped to over-budget models. A fitting model
        // on WDDM must still be FullResident with managed no-spill ON.
        let config = EngineConfig::default();
        let plan = input_managed_wddm(
            &config,
            graph_with_boundary("", "MatMul"),
            256,
            128,
            MemoryStrategyOverrides::default(),
        );
        assert_eq!(plan.strategy, MemoryStrategy::FullResident);
        assert!(!plan.application.weight_offload_enabled);
        assert!(plan.application.managed_no_spill);
        assert_eq!(plan.application.managed_limit_bytes, Some(256));
        assert!(
            !plan
                .decisions
                .iter()
                .any(|d| d.field == "weight_streaming_platform_policy"),
            "fitting models must not trigger the WDDM streaming policy: {plan:?}"
        );
    }

    #[test]
    fn wddm_explicit_offload_request_is_still_honored() {
        // Requirement 1: an explicit ONNX_GENAI_WEIGHT_OFFLOAD=1 must still enable
        // managed streaming even on WDDM. Only the inferred default changes.
        let config = EngineConfig::default();
        let plan = input_managed_wddm(
            &config,
            graph_with_boundary("", "MatMul"),
            64,
            128,
            MemoryStrategyOverrides {
                weight_offload: Some(true),
                ..MemoryStrategyOverrides::default()
            },
        );
        assert_eq!(plan.strategy, MemoryStrategy::DynamicWeightResidency);
        assert!(plan.application.weight_offload_enabled);
    }

    #[test]
    fn wddm_explicit_device_budget_override_is_still_honored() {
        // Requirement 1: an explicit device-budget override selects managed
        // streaming even on WDDM.
        let config = EngineConfig::default();
        let plan = input_managed_wddm(
            &config,
            graph_with_boundary("", "MatMul"),
            64,
            128,
            MemoryStrategyOverrides {
                device_budget_bytes: Some(32),
                ..MemoryStrategyOverrides::default()
            },
        );
        assert_eq!(plan.strategy, MemoryStrategy::DynamicWeightResidency);
        assert!(plan.application.weight_offload_enabled);
        assert_eq!(plan.application.device_budget_bytes, Some(32));
    }

    #[test]
    fn wddm_explicit_vram_limit_is_still_honored() {
        // Requirement 1: an explicit --vram-limit is an override, so managed
        // streaming is honored even on WDDM. `config_with_vram` sets a byte limit;
        // `input_managed_wddm` still models the shared-memory fallback present.
        let config = config_with_vram(64);
        let plan = input_managed_wddm(
            &config,
            graph_with_boundary("", "MatMul"),
            64,
            128,
            MemoryStrategyOverrides::default(),
        );
        assert_eq!(plan.strategy, MemoryStrategy::DynamicWeightResidency);
        assert!(plan.application.weight_offload_enabled);
        assert!(plan.application.managed_no_spill);
    }

    #[test]
    fn managed_streaming_force_knob_overrides_wddm_default() {
        // Requirement 4: ONNX_GENAI_MANAGED_WEIGHT_STREAMING forces the managed
        // path even where the WDDM fallback would otherwise be preferred.
        let config = EngineConfig::default();
        let plan = input_managed_wddm_forced(
            &config,
            graph_with_boundary("", "MatMul"),
            64,
            128,
            MemoryStrategyOverrides::default(),
        );
        assert_eq!(plan.strategy, MemoryStrategy::DynamicWeightResidency);
        assert!(plan.application.weight_offload_enabled);
        assert!(plan.application.managed_no_spill);
        let decision = plan
            .decisions
            .iter()
            .find(|d| d.field == "weight_streaming_platform_policy")
            .expect("forced managed streaming must still be recorded");
        assert_eq!(decision.value, "managed_streaming_forced");
        assert_eq!(decision.source, DecisionSource::ExplicitOverride);
    }

    #[test]
    fn force_managed_weight_streaming_env_parsing() {
        // Opt-in truthy; unrecognized values keep the fast fallback (unlike the
        // ASYNC_PAGEIN trap, an unrecognized value here does NOT select managed).
        assert!(!force_managed_weight_streaming_from_env_value(None));
        assert!(force_managed_weight_streaming_from_env_value(Some("1")));
        assert!(force_managed_weight_streaming_from_env_value(Some("true")));
        assert!(force_managed_weight_streaming_from_env_value(Some("YES")));
        assert!(force_managed_weight_streaming_from_env_value(Some("  On ")));
        assert!(!force_managed_weight_streaming_from_env_value(Some("0")));
        assert!(!force_managed_weight_streaming_from_env_value(Some(
            "false"
        )));
        assert!(!force_managed_weight_streaming_from_env_value(Some("")));
        assert!(!force_managed_weight_streaming_from_env_value(Some(
            "maybe"
        )));
        assert!(!force_managed_weight_streaming_from_env_value(Some("2")));
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

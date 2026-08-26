use super::*;

/// Count default-domain `Scan` nodes in a graph — the population
/// [`Executor::build_decode_inline_sibling`] compares before/after inlining to
/// decide whether a decode-inline plan is warranted.
fn default_domain_scan_count(graph: &Graph) -> usize {
    graph
        .nodes
        .iter()
        .filter(|(_, node)| node.is_default_domain() && node.op_type == "Scan")
        .count()
}

pub(super) struct WeightStoreInitializerResolver(Arc<WeightStore>);

impl InitializerResolver for WeightStoreInitializerResolver {
    fn bytes<'a>(&'a self, weight: &'a onnx_runtime_ir::WeightRef) -> Option<&'a [u8]> {
        self.0.bytes(weight)
    }
}

pub(super) fn run_ep_scoped_passes(
    graph: &mut Graph,
    weights: &Arc<WeightStore>,
    ep: &dyn ExecutionProvider,
) -> Result<()> {
    let passes = ep.custom_passes();
    if passes.is_empty() {
        return Ok(());
    }

    let resolver = Arc::new(WeightStoreInitializerResolver(Arc::clone(weights)));
    let context = onnx_runtime_optimizer::PassContext::new().with_initializer_resolver(resolver);
    onnx_runtime_optimizer::run_passes(graph, &passes, &context)?;

    // Best-effort shape refresh: the passes may have rewritten nodes whose
    // output shapes downstream reads. A *data-dependent* invalidity (e.g. a
    // `Slice` with step 0) is the runtime kernel's contract to reject, not a
    // load-time error — before EP passes existed this re-inference did not run,
    // so the graph built and the actionable diagnostic surfaced at `run`.
    // Re-infer on a clone and adopt the refreshed shapes only on success so such
    // a failure neither aborts the build nor leaves the graph partially updated;
    // the executor's own resolution still validates shapes at run time.
    let registry = InferenceRegistry::default_registry();
    let opset_imports = graph.opset_imports.clone();
    let mut refreshed = graph.clone();
    if registry
        .infer_graph(&mut refreshed, &opset_imports, MergePolicy::Permissive)
        .is_ok()
    {
        *graph = refreshed;
    }
    Ok(())
}

/// The S3 capacity-emission gate: rewrite every KV-cache-growth `Concat` the
/// mask-cone classifier proves safe
/// ([`geometry::kv_capacity_write_eligible_concats`]) into the
/// CUDA-graph-capture-safe `pkg.nxrt::KvCacheCapacityAppend` op, so a
/// decomposed-attention decode step becomes capture/replay-eligible instead of
/// being permanently pinned to `captures == 0` — see that function's doc
/// comment, and [`geometry::KV_CAPACITY_APPEND_OP`], for the full derivation
/// of why a plain `Concat`'s own launch geometry cannot survive a captured
/// replay while this op's can (it reads its one legitimately varying quantity,
/// the destination row, from `position_ids`'s *device memory contents* at
/// execute time instead of baking it into host launch parameters).
///
/// This is a **capability-gated structural rewrite, not a model allowlist**:
/// every candidate is re-checked against `ep.supports_op` individually before
/// it fires, so an EP that has not registered the op (every EP but CUDA
/// today — the CPU EP, or a future EP that never adds the kernel) leaves the
/// original growing `Concat` completely untouched, exactly as correct as it
/// was before this pass existed. Nothing here inspects a model name, op
/// count, or shape beyond what the classifier itself already requires.
///
/// Returns whether any rewrite fired, purely for the caller's bookkeeping (no
/// caller currently branches on it, but it keeps this pass's contract
/// explicit and testable in isolation).
pub(super) fn rewrite_kv_capacity_appends(graph: &mut Graph, ep: &dyn ExecutionProvider) -> bool {
    // `position_ids` is the load-bearing per-step signal the new op's kernel
    // reads at execute time; a graph with no such input (e.g. an encoder, or
    // any export that never threads position ids to begin with) has nothing
    // for the rewrite to bind, so bail before even running the classifier.
    let Some(position_ids) = graph
        .inputs
        .iter()
        .copied()
        .find(|&value| graph.value(value).name.as_deref() == Some("position_ids"))
    else {
        return false;
    };

    let mut eligible: Vec<NodeId> = kv_capacity_write_eligible_concats(graph)
        .into_iter()
        .collect();
    if eligible.is_empty() {
        return false;
    }
    // Deterministic order: `HashSet` iteration order is not stable, and this
    // pass's effect (which nodes end up rewritten) must not depend on hash
    // seed noise.
    eligible.sort_by_key(|id| id.0);

    // The op's registered version (see `onnx-runtime-shape-inference`'s
    // `custom_ops::kv_cache_capacity_append` registration and the CUDA kernel
    // factory's `OpKey`). Queried as a literal rather than via
    // `effective_opset`, since `pkg.nxrt` may not yet be a graph opset-import
    // for a decomposed-attention export that uses no other `pkg.nxrt` op —
    // exactly the case this rewrite exists to upgrade.
    const KV_CAPACITY_APPEND_OPSET: u64 = 1;

    let mut rewritten = false;
    for node_id in eligible {
        let node = graph.node(node_id);
        // The classifier's `is_kv_cache_growth_concat` also matches an
        // *already*-rewritten node (so the mask-cone walk keeps recognizing
        // its own output across re-placement); skip those here; there is
        // nothing left for this pass to do to them.
        if !(node.is_default_domain() && node.op_type == "Concat") {
            continue;
        }
        // A KV-growth append is always binary: `Concat(past, current)`. A
        // node that structurally matched but is not exactly this shape is not
        // one this rewrite understands; leave it as an ordinary Concat.
        if node.inputs.len() != 2 {
            continue;
        }
        let (Some(past), Some(current)) = (node.inputs[0], node.inputs[1]) else {
            continue;
        };
        let present = node.outputs[0];

        // The eligibility classifier proves the *value provenance* of this
        // `Concat`'s output is safe (it structurally reaches a proven-safe
        // sink) -- it says nothing about which axis the concatenation grows
        // on. The new op's kernel and `unsupported_reason` both hardcode that
        // the growth axis is index 2 of a `[batch, heads, seq, head_dim]`
        // physical layout; a `Concat` that structurally matched but grows on
        // a different axis is not something this rewrite understands, and
        // rewriting it anyway would silently write into the wrong physical
        // dimension rather than fail loudly (the batch/heads/head_dim
        // cross-shape checks in `unsupported_reason` only catch this
        // incidentally, and only when every other axis happens to be
        // statically known and distinct). Read the source `Concat`'s
        // mandatory `axis` attribute, normalize a negative index against
        // `past`'s rank, and decline unless it resolves to exactly axis 2 of
        // a rank-4 tensor -- leaving the original `Concat`, which is
        // axis-generic, untouched.
        let past_rank = graph.value(past).shape.len();
        let normalized_axis = node
            .attr("axis")
            .and_then(Attribute::as_int)
            .and_then(|axis| {
                let axis = if axis < 0 {
                    axis + past_rank as i64
                } else {
                    axis
                };
                usize::try_from(axis).ok()
            });
        if past_rank != 4 || normalized_axis != Some(2) {
            continue;
        }

        let mut candidate = Node::new(
            node_id,
            KV_CAPACITY_APPEND_OP,
            vec![Some(past), Some(current), Some(position_ids)],
            vec![present],
        );
        candidate.domain = KV_CAPACITY_APPEND_DOMAIN.to_string();

        let shapes = [past, current, position_ids].map(|value| graph.value(value).shape.clone());
        let dtypes = [past, current, position_ids].map(|value| graph.value(value).dtype);
        let layouts = vec![TensorLayout::contiguous(); shapes.len()];
        if !ep
            .supports_op(
                &candidate,
                KV_CAPACITY_APPEND_OPSET,
                &shapes,
                &dtypes,
                &layouts,
            )
            .is_supported()
        {
            // No kernel for this op on the current EP (e.g. CPU): leave the
            // original Concat exactly as it was.
            continue;
        }

        graph.replace_node(node_id, candidate);
        graph
            .opset_imports
            .entry(KV_CAPACITY_APPEND_DOMAIN.to_string())
            .or_insert(KV_CAPACITY_APPEND_OPSET);
        rewritten = true;
    }

    if rewritten {
        debug_assert!(
            graph.validate().is_ok(),
            "rewrite_kv_capacity_appends produced an invalid graph: {:?}",
            graph.validate().err()
        );
    }
    rewritten
}

pub(super) fn validate_if_branch_outputs(graph: &Graph, node: &Node) -> Result<()> {
    let Some(then_branch) = graph.subgraphs.get(&(node.id, "then_branch".to_string())) else {
        return Ok(());
    };
    let Some(else_branch) = graph.subgraphs.get(&(node.id, "else_branch".to_string())) else {
        return Ok(());
    };

    if then_branch.outputs.len() != else_branch.outputs.len() {
        return Err(SessionError::ControlFlow {
            op: "If".to_string(),
            reason: format!(
                "branches declare different output counts: then_branch has {}, \
                 else_branch has {}",
                then_branch.outputs.len(),
                else_branch.outputs.len()
            ),
        });
    }
    if then_branch.outputs.len() != node.outputs.len() {
        return Err(SessionError::ControlFlow {
            op: "If".to_string(),
            reason: format!(
                "node declares {} output(s), but each branch declares {}",
                node.outputs.len(),
                then_branch.outputs.len()
            ),
        });
    }
    for (index, (&then_output, &else_output)) in then_branch
        .outputs
        .iter()
        .zip(&else_branch.outputs)
        .enumerate()
    {
        if then_branch.value_type_is_known(then_output)
            && else_branch.value_type_is_known(else_output)
        {
            let then_dtype = then_branch.value(then_output).dtype;
            let else_dtype = else_branch.value(else_output).dtype;
            if then_dtype != else_dtype {
                return Err(SessionError::ControlFlow {
                    op: "If".to_string(),
                    reason: format!(
                        "branches declare different dtypes for output {index}: \
                         then_branch is {then_dtype:?}, else_branch is {else_dtype:?}"
                    ),
                });
            }
        }
    }
    Ok(())
}

pub(super) fn validate_control_flow_signatures(graph: &Graph) -> Result<()> {
    for (_, node) in graph.nodes.iter() {
        if node.op_type == "If" && matches!(node.domain.as_str(), "" | "ai.onnx") {
            validate_if_branch_outputs(graph, node)?;
        }
    }
    for subgraph in graph.subgraphs.values() {
        validate_control_flow_signatures(subgraph)?;
    }
    Ok(())
}

/// Whether opt-in per-op heterogeneous placement is enabled on the default
/// session build path (`ONNX_GENAI_HETERO`). Default OFF: unset/empty and the
/// explicit `0`/`false`/`off` values (case-insensitive) all disable it, so the
/// whole-session fallback stays byte-identical unless a caller opts in.
pub(super) fn hetero_placement_env_enabled() -> bool {
    match std::env::var("ONNX_GENAI_HETERO") {
        Ok(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on"
        ),
        Err(_) => false,
    }
}

/// Reject operators no execution provider can run, before EP optimizer passes
/// run. An optimizer pass's postcondition validation walks the whole graph and
/// would otherwise surface a less actionable structural error (e.g. an
/// opset-import invariant) instead of the actionable unsupported-operator
/// diagnostic callers rely on.
///
/// A CUDA graph may legitimately delegate unsupported nodes to a CPU fallback
/// (see [`cuda_fallback_report`]), so an unsupported op is not fatal there; the
/// check is limited to the terminal (non-CUDA) EP. Only nodes with fully static
/// declared input shapes are pre-validated: a symbolic/data-dependent shape is
/// resolved and validated at run time, so pre-checking a contrib op whose
/// support is shape-conditional would change behavior for valid graphs.
pub(super) fn reject_unsupported_operators(
    graph: &Graph,
    ep: &dyn ExecutionProvider,
) -> Result<()> {
    if ep.device_type() == DeviceType::Cuda {
        return Ok(());
    }
    for (node_id, node) in graph.nodes.iter() {
        if onnx_runtime_loader::is_ep_context_op(&node.op_type, &node.domain)
            || is_control_flow_op(&node.op_type, &node.domain)
            || is_sequence_op(&node.op_type, &node.domain)
        {
            continue;
        }

        let shapes = node
            .inputs
            .iter()
            .map(|input| {
                input
                    .map(|value| graph.value(value).shape.clone())
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>();
        // Defer nodes with any non-static declared input shape to the run-time
        // kernel gate, which sees concrete shapes.
        if !shapes.iter().all(|shape| as_static_shape(shape).is_some()) {
            continue;
        }
        let input_dtypes = node
            .inputs
            .iter()
            .map(|input| {
                input
                    .map(|value| graph.value(value).dtype)
                    .unwrap_or(DataType::Undefined)
            })
            .collect::<Vec<_>>();
        let layouts = vec![TensorLayout::contiguous(); shapes.len()];
        let opset = effective_opset(graph, node);
        if let KernelMatch::Unsupported { reason } =
            ep.supports_op(node, opset, &shapes, &input_dtypes, &layouts)
        {
            return Err(SessionError::unsupported_op(
                node,
                node_id,
                opset,
                ep.name(),
                reason,
            ));
        }
    }
    Ok(())
}

pub(super) fn cuda_fallback_report(
    graph: &Graph,
    ep: &dyn ExecutionProvider,
) -> Option<ExecutionProviderFallbackReport> {
    if ep.device_type() != DeviceType::Cuda {
        return None;
    }

    let mut issues = Vec::new();
    collect_cuda_coverage_issues(graph, graph, ep, "graph", &mut issues);
    if issues.is_empty() {
        return None;
    }

    let mut assigned_ops = BTreeSet::new();
    let assigned_node_count = collect_executable_ops(graph, &mut assigned_ops);
    Some(ExecutionProviderFallbackReport {
        requested_provider: ep.name().to_string(),
        fallback_provider: "cpu_ep".to_string(),
        assigned_node_count,
        assigned_ops: assigned_ops.into_iter().collect(),
        declines: issues,
    })
}

pub(super) fn collect_executable_ops(graph: &Graph, ops: &mut BTreeSet<String>) -> usize {
    let mut count = 0;
    for (_, node) in graph.nodes.iter() {
        if !onnx_runtime_loader::is_ep_context_op(&node.op_type, &node.domain) {
            count += 1;
            ops.insert(format!("{}::{}", canonical_domain(node), node.op_type));
        }
    }
    for subgraph in graph.subgraphs.values() {
        count += collect_executable_ops(subgraph, ops);
    }
    count
}

pub(super) fn format_cuda_coverage_issues(issues: &[ExecutionProviderDecline]) -> String {
    const MAX_EXAMPLES_PER_CLASS: usize = 3;

    let mut groups: BTreeMap<(String, String, String), Vec<String>> = BTreeMap::new();
    for issue in issues {
        groups
            .entry((
                issue.domain.clone(),
                issue.op_type.clone(),
                issue.reason.clone(),
            ))
            .or_default()
            .push(issue.node.clone());
    }

    groups
        .into_iter()
        .map(|((domain, op_type, reason), mut nodes)| {
            nodes.sort();
            let count = nodes.len();
            nodes.truncate(MAX_EXAMPLES_PER_CLASS);
            format!(
                "{domain}::{op_type}: {reason} [count={count}; examples: {}]",
                nodes.join(", ")
            )
        })
        .collect::<Vec<_>>()
        .join("; ")
}

pub(super) fn collect_cuda_coverage_issues(
    graph: &Graph,
    opset_graph: &Graph,
    ep: &dyn ExecutionProvider,
    scope: &str,
    issues: &mut Vec<ExecutionProviderDecline>,
) {
    for (node_id, node) in graph.nodes.iter() {
        if onnx_runtime_loader::is_ep_context_op(&node.op_type, &node.domain)
            || is_control_flow_op(&node.op_type, &node.domain)
            || is_sequence_op(&node.op_type, &node.domain)
        {
            continue;
        }

        let shapes = node
            .inputs
            .iter()
            .map(|input| {
                input
                    .map(|value| graph.value(value).shape.clone())
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>();
        let layouts = node
            .inputs
            .iter()
            .map(|input| {
                input
                    .map(|value| graph.value(value).layout.clone())
                    .unwrap_or_else(TensorLayout::contiguous)
            })
            .collect::<Vec<_>>();
        let input_dtypes = node
            .inputs
            .iter()
            .map(|input| {
                input
                    .map(|value| graph.value(value).dtype)
                    .unwrap_or(DataType::Undefined)
            })
            .collect::<Vec<_>>();

        let opset = effective_opset(opset_graph, node);
        if let KernelMatch::Unsupported { reason } =
            ep.supports_op(node, opset, &shapes, &input_dtypes, &layouts)
        {
            issues.push(ExecutionProviderDecline {
                node: format_node_identity(scope, node_id, node),
                domain: canonical_domain(node),
                op_type: node.op_type.clone(),
                reason: reason.into_owned(),
            });
            continue;
        }

        let Some(concrete_shapes) = shapes
            .iter()
            .map(|shape| as_static_shape(shape))
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        if let Err(error) = ep.get_kernel(node, &concrete_shapes, opset) {
            issues.push(ExecutionProviderDecline {
                node: format_node_identity(scope, node_id, node),
                domain: canonical_domain(node),
                op_type: node.op_type.clone(),
                reason: format!("kernel creation failed: {error}"),
            });
        }
    }

    for ((node_id, attribute), subgraph) in &graph.subgraphs {
        let sub_scope = format!("{scope}/node#{}/{}", node_id.0, attribute);
        collect_cuda_coverage_issues(subgraph, opset_graph, ep, &sub_scope, issues);
    }
}

pub(super) fn canonical_domain(node: &Node) -> String {
    if node.domain.is_empty() {
        "ai.onnx".to_string()
    } else {
        node.domain.clone()
    }
}

pub(super) fn format_node_identity(scope: &str, node_id: NodeId, node: &Node) -> String {
    if node.name.is_empty() {
        format!("{scope}/node#{}", node_id.0)
    } else {
        format!("{scope}/node#{} {:?}", node_id.0, node.name)
    }
}

/// Build lazy-weight handles exactly as before, plus (additively) per-expert
/// region candidates for `com.microsoft::QMoE` expert-bank initializers.
///
/// The candidate map is measurement-neutral bookkeeping only (issue #82's
/// first slice): nothing here changes which initializers become lazy, how
/// many [`ExternalMmapRegion`]s a [`LazyWeight`] carries, or how many
/// allocations/H2D copies the CUDA paging path performs. A QMoE initializer
/// whose layout cannot be proven pageable is recorded with its
/// [`onnx_runtime_loader::NonPageableReason`] rather than silently omitted.
pub(super) fn build_lazy_weight_handles_and_candidates(
    graph: &Graph,
    weights: &Arc<WeightStore>,
    ep: &dyn ExecutionProvider,
) -> Result<(
    HashMap<ValueId, WeightHandle>,
    HashMap<ValueId, onnx_runtime_loader::WeightRegionCatalog>,
)> {
    let capabilities = ep.capabilities();
    if !capabilities.advertises(onnx_runtime_ep_api::NXRT_WEIGHT_PAGING_CAPABILITY) {
        return Ok((HashMap::new(), HashMap::new()));
    }

    let qmoe_layouts = qmoe_expert_layouts_by_value(graph);

    let mut handles = HashMap::new();
    let mut candidates = HashMap::new();
    for candidate in lazy_weight_candidates(graph) {
        let value = candidate.value;
        let boundary = candidate.boundary;
        let weight = &graph.initializers[&value];
        if boundary == LazyWeightBoundary::QMoe
            && let Some(layout) = qmoe_layouts.get(&value)
        {
            candidates.insert(
                value,
                onnx_runtime_loader::WeightRegionCatalog::classify(weight, layout.clone()),
            );
        }
        let Some((mapping_id, offset, len)) = weights.external_mmap_provenance(weight) else {
            continue;
        };
        let region = ExternalMmapRegion {
            mapping_id,
            offset,
            len,
        };
        let dtype = weight.dtype();
        let shape = weight.dims().to_vec();
        let weight = weight.clone();
        let store = Arc::clone(weights);
        let lazy = LazyWeight::new(boundary, dtype, shape.clone(), vec![region], move || {
            let bytes = store.bytes(&weight).ok_or_else(|| {
                onnx_runtime_ep_api::WeightHandleError::InvalidResident(
                    "external weight bytes are no longer available".into(),
                )
            })?;
            ResidentWeight::new(dtype, shape.clone(), Arc::<[u8]>::from(bytes))
        })
        .map_err(|error| {
            SessionError::Internal(format!(
                "cannot create lazy weight handle for value#{}: {error}",
                value.0
            ))
        })?;
        handles.insert(value, WeightHandle::Lazy(lazy));
    }
    Ok((handles, candidates))
}

/// Build the default [`onnx_runtime_ep_api::ResidencyPlan`] for this build's
/// per-expert region candidates.
///
/// This is the one call site where a plan is *created*. It always uses
/// [`onnx_runtime_ep_api::WholeBankResidentPolicy`] today — the only shipped
/// policy — so the produced plan is handle-for-handle/byte-for-byte
/// equivalent to pre-#82 behavior: every candidate value resolves to
/// `WholeBankResident`. A future slice may take the policy as a parameter;
/// this slice intentionally does not expose that knob outside tests.
pub(super) fn plan_default_residency(
    graph: &Graph,
    candidates: &HashMap<ValueId, onnx_runtime_loader::WeightRegionCatalog>,
) -> onnx_runtime_ep_api::ResidencyPlan {
    plan_residency_with(
        graph,
        candidates,
        &onnx_runtime_ep_api::WholeBankResidentPolicy,
    )
}

/// Test/measurement seam: build a plan with an arbitrary
/// [`onnx_runtime_ep_api::ResidencyPolicy`], to prove the boundary is
/// substitutable without wiring a second production call site.
pub(super) fn plan_residency_with(
    graph: &Graph,
    candidates: &HashMap<ValueId, onnx_runtime_loader::WeightRegionCatalog>,
    policy: &dyn onnx_runtime_ep_api::ResidencyPolicy,
) -> onnx_runtime_ep_api::ResidencyPlan {
    let boundary_by_value: HashMap<ValueId, LazyWeightBoundary> = lazy_weight_candidates(graph)
        .into_iter()
        .map(|candidate| (candidate.value, candidate.boundary))
        .collect();
    let inputs = candidates.iter().filter_map(|(value, catalog)| {
        boundary_by_value
            .get(value)
            .map(|&boundary| (*value, boundary, catalog))
    });
    onnx_runtime_ep_api::plan_residency(inputs, policy, None)
}

/// Derive an [`onnx_runtime_loader::ExpertTensorLayout`] for every QMoE
/// expert-bank initializer value in `graph`, keyed by the initializer's
/// [`ValueId`].
///
/// Mirrors `onnx-genai-engine::engine::placement::qmoe_layers`'s input-index
/// convention (fc1 packed/scales/zero-points at 2/3/11, fc2 at 5/6/12, and an
/// optional fc3 triple at 8/9/13) so both call sites derive layouts from the
/// same node attributes/initializer shapes and cannot silently drift onto
/// different byte-range formulas. A value absent from the returned map either
/// isn't a QMoE expert-bank tensor, or its node/attributes/shape could not
/// support deriving a layout at all (as opposed to deriving one that then
/// fails [`onnx_runtime_loader::WeightRegionCatalog::classify`]'s validation,
/// which is instead recorded as a `NonPageable` catalog by the caller).
fn qmoe_expert_layouts_by_value(
    graph: &Graph,
) -> HashMap<ValueId, onnx_runtime_loader::ExpertTensorLayout> {
    let mut layouts = HashMap::new();
    for (_, node) in graph.nodes.iter() {
        if !(node.op_type == "QMoE" && (node.domain.is_empty() || node.domain == "com.microsoft")) {
            continue;
        }
        let Some(bits) = qmoe_int_attr(node, "expert_weight_bits", 4)
            .ok()
            .and_then(|value| usize::try_from(value).ok())
            .filter(|bits| matches!(bits, 1 | 2 | 4 | 8))
        else {
            continue;
        };
        let Some(block_size) = qmoe_int_attr(node, "block_size", 0)
            .ok()
            .and_then(|value| usize::try_from(value).ok())
            .filter(|block_size| *block_size > 0)
        else {
            continue;
        };
        for &(packed_idx, scales_idx, zero_idx) in &[(2, 3, 11), (5, 6, 12), (8, 9, 13)] {
            if packed_idx == 8 && node.inputs.get(8).and_then(|slot| *slot).is_none() {
                continue;
            }
            qmoe_push_expert_layout(
                graph,
                node,
                scales_idx,
                packed_idx,
                bits,
                block_size,
                &mut layouts,
            );
            qmoe_push_expert_layout(
                graph,
                node,
                scales_idx,
                scales_idx,
                bits,
                block_size,
                &mut layouts,
            );
            if node.inputs.get(zero_idx).and_then(|slot| *slot).is_some() {
                qmoe_push_expert_layout(
                    graph,
                    node,
                    scales_idx,
                    zero_idx,
                    bits,
                    block_size,
                    &mut layouts,
                );
            }
        }
    }
    layouts
}

/// Derive and record the layout for one QMoE input tensor at `input_index`,
/// using the sibling scales tensor at `scales_index` to read `blocks_per_row`
/// (the scales tensor's own storage-elements-per-row axis).
///
/// Whenever `value` resolves to a graph initializer, an entry is always
/// recorded (never silently omitted): a derivable expert-major layout is
/// recorded as-is, and a value whose shape/sibling scales cannot support
/// deriving one (e.g. non-rank-3 dims) is instead recorded with
/// [`onnx_runtime_loader::ExpertStorageOrder::Interleaved`], which
/// [`onnx_runtime_loader::WeightRegionCatalog::classify`] turns into an
/// explicit `NonPageableReason::NotExpertMajor` rather than the value
/// vanishing from the candidate map entirely.
fn qmoe_push_expert_layout(
    graph: &Graph,
    node: &Node,
    scales_index: usize,
    input_index: usize,
    bits: usize,
    block_size: usize,
    layouts: &mut HashMap<ValueId, onnx_runtime_loader::ExpertTensorLayout>,
) -> Option<()> {
    let value = node.inputs.get(input_index).and_then(|slot| *slot)?;
    let weight = qmoe_initializer(graph, value)?;
    let derived = (|| {
        let scales_value = node.inputs.get(scales_index).and_then(|slot| *slot)?;
        let scales_weight = qmoe_initializer(graph, scales_value)?;
        let scales_dims = scales_weight.dims();
        if scales_dims.len() != 3 {
            return None;
        }
        let blocks_per_row = scales_dims[2];
        onnx_runtime_loader::qmoe_expert_tensor_layout(
            bits,
            block_size,
            blocks_per_row,
            weight.dims(),
        )
    })();
    let layout = derived.unwrap_or(onnx_runtime_loader::ExpertTensorLayout {
        version: 1,
        experts: 0,
        rows_per_expert: 0,
        storage_elements_per_row: 0,
        order: onnx_runtime_loader::ExpertStorageOrder::Interleaved,
        quantization: None,
    });
    layouts.insert(value, layout);
    Some(())
}

/// Resolve `value` to the [`onnx_runtime_ir::WeightRef`] it initializes,
/// following one `Cast` producer (the same convention
/// `onnx-genai-engine::engine::placement` uses for QMoE inputs that a graph
/// rewrite has cast to a wider compute dtype).
fn qmoe_initializer(graph: &Graph, value: ValueId) -> Option<&onnx_runtime_ir::WeightRef> {
    if let Some(weight) = graph.initializers.get(&value) {
        return Some(weight);
    }
    let producer = graph.values.get(value)?.producer?;
    let node = graph.nodes.get(producer)?;
    if node.domain.is_empty() && node.op_type == "Cast" {
        let source = node.inputs.first().and_then(|slot| *slot)?;
        return graph.initializers.get(&source);
    }
    None
}

fn qmoe_int_attr(node: &Node, name: &'static str, default: i64) -> std::result::Result<i64, ()> {
    match node.attr(name) {
        Some(attr) => attr.as_int().ok_or(()),
        None => Ok(default),
    }
}

impl Executor {
    /// Per-expert region candidates recorded for `com.microsoft::QMoE` lazy
    /// weights, keyed by initializer [`ValueId`]. See
    /// [`Executor::expert_region_candidates`](state) for the additive,
    /// measurement-neutral contract this exposes.
    #[cfg(test)]
    pub(super) fn expert_region_candidates(
        &self,
    ) -> &HashMap<ValueId, onnx_runtime_loader::WeightRegionCatalog> {
        &self.expert_region_candidates
    }

    /// The default [`onnx_runtime_ep_api::ResidencyPlan`] built for this
    /// executor's expert region candidates. Read-only outside tests today —
    /// no dispatch-time consumer exists yet; this exposes the plan purely
    /// for telemetry/measurement and future policy wiring.
    #[cfg(test)]
    pub(super) fn residency_plan(&self) -> &onnx_runtime_ep_api::ResidencyPlan {
        &self.residency_plan
    }

    /// Compile a graph + weights into a runnable executor on the CPU EP.
    pub(crate) fn build(
        graph: Graph,
        weights: Arc<WeightStore>,
        ep: Arc<dyn ExecutionProvider>,
    ) -> Result<Self> {
        Self::build_with_cuda_requirement(
            graph,
            weights,
            ep,
            onnx_genai_runtime_config::runtime_config().require_cuda,
        )
    }

    pub(super) fn build_with_cuda_requirement(
        mut graph: Graph,
        weights: Arc<WeightStore>,
        mut ep: Arc<dyn ExecutionProvider>,
        require_cuda: bool,
    ) -> Result<Self> {
        let execution_provider_fallback_report =
            Self::place_graph(&mut graph, &weights, &mut ep, require_cuda)?;
        // Topological order up front: also validates the selected graph is a DAG.
        let order = graph.topological_order()?;
        let (weight_handles, expert_region_candidates) = {
            let mut span = trace_span("session.lazy_weight_handles", "session");
            let (handles, candidates) =
                build_lazy_weight_handles_and_candidates(&graph, &weights, ep.as_ref())?;
            if let Some(span) = span.as_mut() {
                span.set_args(
                    Args::new()
                        .with("handles", handles.len() as u64)
                        .with("initializers", graph.initializers.len() as u64)
                        .with("expert_region_candidates", candidates.len() as u64),
                );
            }
            (handles, candidates)
        };

        let residency_plan = {
            let mut span = trace_span("session.residency_plan", "session");
            let plan = plan_default_residency(&graph, &expert_region_candidates);
            if let Some(span) = span.as_mut() {
                span.set_args(
                    Args::new()
                        .with("policy", plan.policy_name().to_string())
                        .with("resident", plan.resident_count() as u64)
                        .with("degraded", plan.degraded_count() as u64)
                        .with("total", plan.len() as u64),
                );
            }
            plan
        };

        let (mut value_shapes, mut value_dtypes, buffers, buffer_shapes) =
            Self::materialize_initializers(&graph, &weights, ep.as_ref(), &weight_handles)?;

        // 2) Record the loader shape + dtype of every remaining value (graph
        //    inputs and node outputs). No allocation yet — shapes may be
        //    symbolic and are only sized once resolved.
        let has_symbols =
            Self::collect_value_metadata(&graph, &order, &mut value_shapes, &mut value_dtypes);

        let (sequence_values, control_flow_output_values) =
            Self::classify_special_values(&graph, &order);

        // 3) Build the structural per-node plan.
        let capabilities = ep.capabilities();
        let plan = Self::build_node_plan(
            &graph,
            &order,
            &value_dtypes,
            has_symbols,
            &weight_handles,
            &capabilities,
        );

        // 4) name → value id and the set of caller-required inputs.
        let (input_index, required_inputs, name_index) = Self::build_name_indexes(&graph);

        let plan_len = plan.len();
        let capture_growing_symbols = compute_capture_disqualifying_symbols(&graph);
        let mut exec = Self {
            graph,
            weights,
            ep,
            graph_slot: DeviceGraphSlot::Primary,
            weight_handles,
            expert_region_candidates,
            residency_plan,
            prefetch_issue_nodes: std::sync::Mutex::new(HashMap::new()),
            prefetch_lookahead_nodes: dense_weight_prefetch_lookahead_nodes(),
            buffers,
            buffer_shapes,
            value_shapes,
            value_dtypes,
            plan,
            input_index,
            required_inputs,
            has_symbols,
            cache: KernelCache::default(),
            name_index,
            subgraph_execs: HashMap::new(),
            control_flow_stats: ControlFlowStats::default(),
            if_last_predicate: HashMap::new(),
            slot_capture: std::array::from_fn(|_| SlotCaptureState::default()),
            control_flow_output_values,
            capture_growing_symbols,
            capacity_pinned_kv_symbols: HashSet::new(),
            last_capture_failed_node: None,
            views: HashMap::new(),
            pinned: HashSet::new(),
            sequence_values,
            activation_memory_plan: None,
            shared_buffers: HashMap::new(),
            parked_input_buffers: Vec::new(),
            capture_deferred_frees: Vec::new(),
            sequences: HashMap::new(),
            seq_elem_values: HashMap::new(),
            execution_provider_fallback_report,
            trace: TraceContext::noop(),
            scratch_input_shapes: Vec::new(),
            decode_memo_enabled: decode_memo_env_enabled(),
            decode_memo_verify: cfg!(debug_assertions) || decode_memo_verify_env_enabled(),
            decode_memo: None,
            decode_memo_prev_bindings: None,
            decode_memo_last_action: DecodeMemoAction::Disabled,
            decode_memo_resolved: HashMap::new(),
            decode_memo_primed_count: 0,
            decode_memo_rebuilt_count: 0,
            decode_memo_replayed_count: 0,
            decode_memo_ineligible_count: 0,
            decode_view_plan: None,
            decode_views_reused_count: 0,
            decode_dispatch_elided_count: 0,
            decode_view_plan_sig_mismatch_streak: 0,
            decode_view_plan_disabled: false,
            compute_in_place_enabled: compute_in_place_env_enabled(),
            release_dead_values_enabled: false,
            compute_in_place_alias_count: 0,
            scan_inline_single_trip_enabled: scan_inline_single_trip_env_enabled(),
            scan_inline_single_trip_count: 0,
            kernel_bindings: vec![None; plan_len],
            persistent_workspace: None,
            step_workspace: None,
            pin_step_workspace: false,
            inherited_workspace: None,
            workspace_preparation_required: false,
        };

        // 5) Fully-static graphs are materialized eagerly (buffers + the whole
        //    "compiled plan" of kernels), so the first `run` sees only cache
        //    hits. Symbolic graphs cannot be sized until a `run` fixes their
        //    shapes, so their buffers/kernels are created on first use.
        if !exec.has_symbols {
            let mut span = trace_span("session.static_materialize", "session");
            let empty = HashMap::new();
            let resolved = exec.resolve_all(&empty)?;
            exec.compile_all(&resolved)?;
            exec.size_buffers(&resolved)?;
            if let Some(span) = span.as_mut() {
                span.set_args(
                    Args::new()
                        .with("resolved_values", resolved.len() as u64)
                        .with("buffers", exec.buffers.len() as u64)
                        .with("plan_len", exec.plan.len() as u64)
                        .with("cache_entries", exec.cache.stats().entries as u64),
                );
            }
        }

        // Pre-compute weight transposes for the GEMV decode path on Apple
        // Silicon. Model load is 15× faster than ORT (105 ms vs 1596 ms), so
        // spending ~1 s here still wins the model-load metric while eliminating
        // a ~1 s spike on the first decode step.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            use onnx_runtime_ir::DataType;
            let mut _span = trace_span("session.precompute_weight_transposes", "session");
            let mut transposed_count = 0_u64;
            for node_plan in &exec.plan {
                let node = exec.graph.node(node_plan.node_id);
                if node.op_type != "MatMul" && node.op_type != "FusedMatMulBias" {
                    continue;
                }
                let Some(b_vid) = node_plan.inputs.get(1).copied().flatten() else {
                    continue;
                };
                if !exec.graph.initializers.contains_key(&b_vid) {
                    continue;
                }
                if node_plan.input_dtypes.get(1) != Some(&DataType::Float16) {
                    continue;
                }
                let Some(shape) = exec.value_shapes.get(&b_vid) else {
                    continue;
                };
                if shape.len() != 2 {
                    continue;
                }
                let (k, n) = match (shape[0].as_static(), shape[1].as_static()) {
                    (Some(k), Some(n)) => (k, n),
                    _ => continue,
                };
                let Some(buf) = exec.buffers.get(&b_vid) else {
                    continue;
                };
                // Verify the buffer is dense (no padding/gaps). ONNX format
                // guarantees initializer tensors are contiguous, but assert so
                // a future change cannot silently produce wrong transposes.
                debug_assert_eq!(
                    buf.len(),
                    k * n * 2,
                    "precompute_f16_weight_transpose: buffer size {} != expected {} for shape [{k}, {n}]",
                    buf.len(),
                    k * n * 2,
                );
                let ptr = buf.as_ptr() as *const u16;
                // SAFETY: `ptr` comes from a materialized DeviceBuffer backed by
                // the model's mmap. The buffer holds k*n contiguous f16 values
                // (verified by the initializer dtype + shape checks above) and
                // outlives this call (model lifetime).
                unsafe {
                    onnx_runtime_ep_cpu::kernels::matmul::precompute_f16_weight_transpose(
                        ptr, k, n,
                    );
                }
                transposed_count += 1;
            }
            // Also pre-compute f32 transposes for weights eligible for the
            // thin-M GEMM path (large K*N). Same rationale: move the transpose
            // cost into model load to avoid a TTFT spike on first inference.
            for node_plan in &exec.plan {
                let node = exec.graph.node(node_plan.node_id);
                if node.op_type != "MatMul" && node.op_type != "FusedMatMulBias" {
                    continue;
                }
                let Some(b_vid) = node_plan.inputs.get(1).copied().flatten() else {
                    continue;
                };
                if !exec.graph.initializers.contains_key(&b_vid) {
                    continue;
                }
                if node_plan.input_dtypes.get(1) != Some(&DataType::Float32) {
                    continue;
                }
                let Some(shape) = exec.value_shapes.get(&b_vid) else {
                    continue;
                };
                if shape.len() != 2 {
                    continue;
                }
                let (k, n) = match (shape[0].as_static(), shape[1].as_static()) {
                    (Some(k), Some(n)) => (k, n),
                    _ => continue,
                };
                let Some(buf) = exec.buffers.get(&b_vid) else {
                    continue;
                };
                debug_assert_eq!(
                    buf.len(),
                    k * n * 4,
                    "precompute_f32_weight_transpose: buffer size {} != expected {} for shape [{k}, {n}]",
                    buf.len(),
                    k * n * 4,
                );
                let ptr = buf.as_ptr() as *const f32;
                // SAFETY: `ptr` comes from a materialized DeviceBuffer backed by
                // the model's mmap. The buffer holds k*n contiguous f32 values
                // (verified by the initializer dtype + shape checks above) and
                // outlives this call (model lifetime).
                unsafe {
                    onnx_runtime_ep_cpu::kernels::matmul::precompute_f32_weight_transpose(
                        ptr, k, n,
                    );
                }
                transposed_count += 1;
            }
            if let Some(span) = _span.as_mut() {
                span.set_args(Args::new().with("transposed_weights", transposed_count));
            }
        }

        Ok(exec)
    }

    /// Build the decode-specialized *inlined-body* sibling of this executor
    /// (Inc-1b PR-2, `cohaagen-27b-inc1b-design.md` §1). Runs
    /// [`inline_single_trip_scan_bodies`](onnx_runtime_ir::inline_single_trip_scan_bodies)
    /// over this executor's graph, re-resolves interior shapes with Permissive
    /// inference (exactly as `ChildExecutor::compile` does), and builds a second
    /// [`Executor`] that **shares this executor's `Arc<WeightStore>` and
    /// `Arc<dyn ExecutionProvider>`** — the multi-plan/shared-weights pattern.
    ///
    /// Returns `Ok(None)` when the graph has no single-trip-eligible recurrent
    /// `Scan` (a dense / non-hybrid decoder), so the caller keeps today's Scan
    /// child-session path unchanged. The prefill/main executor (`self`) is left
    /// **byte-identical** — this is a separate, additive plan.
    ///
    /// The returned sibling runs **eager** (capture is out of scope for PR-2);
    /// letting the inlined body fold into the captured graph is PR-3.
    pub(crate) fn build_decode_inline_sibling(&self) -> Result<Option<Self>> {
        let scans_before = default_domain_scan_count(&self.graph);
        if scans_before == 0 {
            return Ok(None);
        }
        let mut graph = onnx_runtime_ir::inline_single_trip_scan_bodies(&self.graph);
        // The transform is a structural no-op unless it actually lowered an
        // eligible Scan; if nothing changed there is no decode-inline plan to
        // build (the model is not on the hybrid single-trip Scan path).
        if default_domain_scan_count(&graph) >= scans_before {
            return Ok(None);
        }

        // Re-resolve the merged interior shapes (Squeeze/Unsqueeze boundaries and
        // remapped body values) before build, mirroring `ChildExecutor::compile`.
        let registry = InferenceRegistry::default_registry();
        let opset_imports = graph.opset_imports.clone();
        registry.infer_graph(&mut graph, &opset_imports, MergePolicy::Permissive)?;

        let sibling = Self::build(graph, Arc::clone(&self.weights), Arc::clone(&self.ep))?;
        Ok(Some(sibling))
    }

    /// Build a verify-dedicated sibling executor for native MTP self-speculative
    /// decode. Structurally identical to the main executor (same graph), but with
    /// its **own** interior device-buffer arena (`buffers`/`buffer_shapes`) so the
    /// fixed M=k+1 verify forward's JIT-sized interior scratch is never resized by
    /// the interleaved M=1 base decode running on the main executor — the exact
    /// clobber that made the shared-arena verify capture decline (interior
    /// `Slice` sized `[1,1]` by the M=1 decode vs the `[1,2]` the M=2 verify
    /// needs). Shares ONLY the immutable `Arc<WeightStore>` and
    /// `Arc<dyn ExecutionProvider>` with the main executor, so a verify step
    /// routed here binds the identical persistent **external** state buffers (KV
    /// cache, recurrent/conv state, embeds) supplied through the caller's
    /// `bindings`, while its **interior** scratch stays private and fixed at the
    /// M=k+1 shape across every replay. Drives the [`DeviceGraphSlot::Verify`]
    /// slot so its captured M=k+1 graph is independent of the main executor's
    /// `Primary` M=1 decode graph (both coexist on the shared EP and each replays
    /// by shape key). No graph transform is applied: unlike the decode-inline
    /// sibling this is a plain structural clone, so it is available for **any**
    /// model with recurrent state (including artifacts whose recurrent `Scan` is
    /// not single-trip-inlineable).
    pub(crate) fn build_verify_sibling(&self) -> Result<Self> {
        let mut graph = self.graph.clone();
        // Re-resolve interior shapes on the cloned (post-placement) graph before
        // build, mirroring `build_decode_inline_sibling`'s permissive re-inference.
        let registry = InferenceRegistry::default_registry();
        let opset_imports = graph.opset_imports.clone();
        registry.infer_graph(&mut graph, &opset_imports, MergePolicy::Permissive)?;
        let mut sibling = Self::build(graph, Arc::clone(&self.weights), Arc::clone(&self.ep))?;
        sibling.graph_slot = DeviceGraphSlot::Verify;
        // Pin the sibling's StepScoped workspace. The sibling ONLY ever runs the
        // fixed M=k+1 verify shape, so its workspace is reserved once at that peak
        // and never needs to grow or shrink. Its captured verify graph bakes the
        // workspace device pointer, so it must NOT be freed between replays:
        // without the pin, `release_step_workspace` returns the workspace to the
        // shared EP arena after each step, the interleaved M=1 decode on the main
        // executor reserves that same freed slot, and the next verify replay reads
        // reallocated memory (a nondeterministic illegal access, #1647's stale-ptr
        // hazard). Pinning keeps the sibling's scratch pointer stable for every
        // replay.
        sibling.pin_step_workspace = true;
        Ok(sibling)
    }

    /// Place the graph on execution providers: reject incompatible graphs, run
    /// the EP-scoped optimizer passes, and — when the requested CUDA EP cannot
    /// cover the graph and CUDA is not required — fall back to the CPU EP.
    /// Reassigns `graph` and `ep` in place and returns the fallback report (if
    /// any) for the executor to retain. Preserves the `session.node_placement`
    /// tracing span and every span argument.
    ///
    /// Also runs [`rewrite_kv_capacity_appends`] (the S3 capacity-emission
    /// gate) between the EP-scoped optimizer passes and the CUDA-coverage
    /// fallback check: it must see the *stabilized* post-EP-pass graph (any
    /// CUDA-only fusion that could otherwise touch a KV-growth `Concat` has
    /// already run), and its own output must be accounted for by the coverage
    /// check that follows.
    fn place_graph(
        graph: &mut Graph,
        weights: &Arc<WeightStore>,
        ep: &mut Arc<dyn ExecutionProvider>,
        require_cuda: bool,
    ) -> Result<Option<ExecutionProviderFallbackReport>> {
        let mut placement_span = trace_span("session.node_placement", "session");
        let requested_provider = placement_span.as_ref().map(|_| ep.name().to_string());
        let requested_device = placement_span
            .as_ref()
            .map(|_| ep.device_type().trace_name().into_owned());
        let nodes_before_placement = graph.num_nodes();
        // Reject incompatible control-flow signatures before EP optimizers run:
        // optimizer postconditions recursively validate subgraphs and can
        // otherwise obscure the actionable If diagnostic with a structural
        // error from a malformed branch.
        validate_control_flow_signatures(graph)?;
        // Reject structurally invalid graphs (a non-DAG) and operators no EP can
        // run *before* EP optimizers run. An optimizer pass's postcondition
        // validation would otherwise obscure the actionable load-time diagnostic
        // (a wrapped `CycleDetected`, or an opset-import invariant instead of the
        // unsupported-operator error) with a structural error. Mirrors the
        // control-flow signature check above.
        graph.topological_order()?;
        reject_unsupported_operators(graph, ep.as_ref())?;
        let graph_before_ep_passes = graph.clone();
        let ep_pass_nodes_before = graph.num_nodes();
        run_ep_scoped_passes(graph, weights, ep.as_ref())?;
        let ep_pass_nodes_after = graph.num_nodes();
        rewrite_kv_capacity_appends(graph, ep.as_ref());
        let mut execution_provider_fallback_report = cuda_fallback_report(graph, ep.as_ref());
        let fallback_declines = execution_provider_fallback_report
            .as_ref()
            .map_or(0, |report| report.declines.len());
        if let Some(report) = &mut execution_provider_fallback_report {
            if require_cuda {
                return Err(SessionError::HeterogeneousPlacementRequired {
                    unsupported_nodes: report.to_string(),
                });
            }
            // Thread-3 Phase 3: before silently dropping the whole session onto
            // CPU (a catastrophic perf cliff), consult the per-op heterogeneous
            // planner when opted in (`ONNX_GENAI_HETERO`). A genuinely mixed
            // graph fails closed with an actionable per-op summary; a homogeneous
            // graph proceeds to the byte-identical whole-session fallback below.
            // Default OFF ⇒ this is a no-op and the fallback is unchanged.
            let hetero_providers = [
                crate::hetero::ProviderPlacement {
                    ep: onnx_runtime_ep_api::EpId(0),
                    provider: Arc::clone(ep),
                },
                crate::hetero::ProviderPlacement {
                    ep: onnx_runtime_ep_api::EpId(1),
                    provider: auto_detect_cpu_ep()?,
                },
            ];
            crate::hetero::guard_heterogeneous_fallback(
                &graph_before_ep_passes,
                &hetero_providers,
                hetero_placement_env_enabled(),
            )?;
            *graph = graph_before_ep_passes;
            *ep = auto_detect_cpu_ep()?;
            run_ep_scoped_passes(graph, weights, ep.as_ref())?;
            let mut assigned_ops = BTreeSet::new();
            report.assigned_node_count = collect_executable_ops(graph, &mut assigned_ops);
            report.assigned_ops = assigned_ops.into_iter().collect();
            eprintln!(
                "[onnx-genai-warning] {report}. Set ONNX_GENAI_REQUIRE_CUDA=1 to reject this fallback"
            );
        }
        if let Some(span) = placement_span.as_mut() {
            let mut assigned_ops = BTreeSet::new();
            let assigned_nodes = collect_executable_ops(graph, &mut assigned_ops);
            span.set_args(
                Args::new()
                    .with("requested_provider", requested_provider.unwrap_or_default())
                    .with("requested_device", requested_device.unwrap_or_default())
                    .with("selected_provider", ep.name().to_string())
                    .with(
                        "selected_device",
                        ep.device_type().trace_name().into_owned(),
                    )
                    .with("nodes_before", nodes_before_placement as u64)
                    .with("nodes_after", graph.num_nodes() as u64)
                    .with("ep_pass_nodes_before", ep_pass_nodes_before as u64)
                    .with("ep_pass_nodes_after", ep_pass_nodes_after as u64)
                    .with("assigned_nodes", assigned_nodes as u64)
                    .with("assigned_op_classes", assigned_ops.len() as u64)
                    .with("fallback_declines", fallback_declines as u64),
            );
        }
        drop(placement_span);
        Ok(execution_provider_fallback_report)
    }

    /// Record initializer metadata and back resident consumers with a device
    /// buffer. A non-host nxrt initializer used exclusively at the lazy
    /// fused-MoE boundary deliberately has no eager buffer; the EP materializes
    /// it through its WeightHandle on demand. If any resident consumer (or graph
    /// output) coexists, no handle is built and the one eager buffer is shared by
    /// every consumer. Host mmap bytes retain the existing zero-copy borrow path.
    /// Preserves the `session.initializer_buffers` tracing span and its
    /// arguments.
    #[allow(clippy::type_complexity)]
    fn materialize_initializers(
        graph: &Graph,
        weights: &Arc<WeightStore>,
        ep: &dyn ExecutionProvider,
        weight_handles: &HashMap<ValueId, WeightHandle>,
    ) -> Result<(
        HashMap<ValueId, Shape>,
        HashMap<ValueId, DataType>,
        HashMap<ValueId, DeviceBuffer>,
        HashMap<ValueId, Vec<usize>>,
    )> {
        let mut value_shapes: HashMap<ValueId, Shape> = HashMap::new();
        let mut value_dtypes: HashMap<ValueId, DataType> = HashMap::new();
        let mut buffers: HashMap<ValueId, DeviceBuffer> = HashMap::new();
        let mut buffer_shapes: HashMap<ValueId, Vec<usize>> = HashMap::new();

        let init_align = TensorLayout::contiguous().alignment;
        let mut initializer_span = trace_span("session.initializer_buffers", "session");
        let mut initializer_count = 0_u64;
        let mut initializer_bytes = 0_u64;
        let mut borrowed_initializers = 0_u64;
        let mut copied_initializers = 0_u64;
        let mut lazy_initializers = 0_u64;
        for (&vid, weight) in &graph.initializers {
            let dtype = weight.dtype();
            let dims = weight.dims().to_vec();
            value_dtypes.insert(vid, dtype);
            value_shapes.insert(vid, dims.iter().map(|&d| Dim::Static(d)).collect());
            if !ep.device_id().is_host_accessible() && weight_handles.contains_key(&vid) {
                if initializer_span.is_some() {
                    lazy_initializers += 1;
                }
                continue;
            }
            let bytes = weights.bytes(weight).ok_or_else(|| {
                SessionError::Internal(format!("weight bytes unavailable for value#{}", vid.0))
            })?;
            if initializer_span.is_some() {
                initializer_count += 1;
                initializer_bytes += bytes.len() as u64;
            }
            // Only borrow when the value has NO producer. The borrowed
            // `DeviceBuffer` aliases read-only mmap/inline storage, so it must
            // never be written. A legitimate initializer always has
            // `producer == None`; a malformed graph can reuse an initializer's
            // `ValueId` as a node output (see loader `validate_no_initializer_producer`),
            // giving it a producer — a kernel would then write through
            // `as_mut_ptr()` into read-only mmap (SIGSEGV / aliasing UB). In
            // that case fall back to the owned writable copy below.
            let producer_less = graph.value(vid).producer.is_none();
            let borrow_align = if matches!(weight, WeightRef::External { .. }) {
                host_dtype_alignment(dtype)
            } else {
                init_align
            };
            let buf = if ep.device_id().is_host_accessible()
                && producer_less
                && !bytes.is_empty()
                && (bytes.as_ptr() as usize).is_multiple_of(borrow_align)
            {
                if initializer_span.is_some() {
                    borrowed_initializers += 1;
                }
                // Zero-copy: alias the suitably aligned initializer bytes. For
                // external data this is only the dtype alignment; inline data
                // retains the EP allocation alignment requirement.
                // SAFETY: `bytes` borrows live mmap storage in `weights` or
                // inline storage in `graph`; both executor fields outlive every
                // buffer use. The range is `bytes.len()` long,
                // `borrow_align`-aligned, and treated as read-only.
                unsafe {
                    DeviceBuffer::from_borrowed_parts(
                        bytes.as_ptr() as *mut std::ffi::c_void,
                        ep.device_id(),
                        bytes.len(),
                        borrow_align,
                    )
                }
            } else {
                if initializer_span.is_some() {
                    copied_initializers += 1;
                }
                let mut owned = ep.allocate(bytes.len().max(1), init_align)?;
                ep.copy_from_host(bytes, &mut owned)?;
                owned
            };
            buffer_shapes.insert(vid, dims);
            buffers.insert(vid, buf);
        }
        if let Some(span) = initializer_span.as_mut() {
            span.set_args(
                Args::new()
                    .with("initializers", initializer_count)
                    .with("bytes", initializer_bytes)
                    .with("borrowed_initializers", borrowed_initializers)
                    .with("copied_initializers", copied_initializers)
                    .with("lazy_initializers", lazy_initializers)
                    .with("buffers", buffers.len() as u64),
            );
        }
        Ok((value_shapes, value_dtypes, buffers, buffer_shapes))
    }

    /// Record the loader shape + dtype of every remaining value (graph inputs
    /// and node outputs). No allocation yet — shapes may be symbolic and are
    /// only sized once resolved. Returns whether any recorded shape is symbolic.
    fn collect_value_metadata(
        graph: &Graph,
        order: &[NodeId],
        value_shapes: &mut HashMap<ValueId, Shape>,
        value_dtypes: &mut HashMap<ValueId, DataType>,
    ) -> bool {
        for &vid in &graph.inputs {
            value_shapes
                .entry(vid)
                .or_insert_with(|| graph.value(vid).shape.clone());
            value_dtypes.entry(vid).or_insert(graph.value(vid).dtype);
        }
        for &nid in order {
            for &out in &graph.node(nid).outputs {
                value_shapes
                    .entry(out)
                    .or_insert_with(|| graph.value(out).shape.clone());
                value_dtypes.entry(out).or_insert(graph.value(out).dtype);
            }
        }

        value_shapes.values().any(|s| as_static_shape(s).is_none())
    }

    /// Classify values that need special buffer-sizing treatment: those produced
    /// by a sequence-producing op (own no tensor buffer — a Sequence op stores
    /// its list in `sequences` at run time) and the outputs of every
    /// control-flow node (seeded into the capture plan so downstream capturable
    /// consumers do not each form an eager seam). Returns
    /// `(sequence_values, control_flow_output_values)`.
    fn classify_special_values(
        graph: &Graph,
        order: &[NodeId],
    ) -> (HashSet<ValueId>, HashSet<ValueId>) {
        let mut sequence_values: HashSet<ValueId> = HashSet::new();
        for &nid in order {
            let node = graph.node(nid);
            if produces_sequence_output(&node.op_type, &node.domain) {
                for &out in &node.outputs {
                    sequence_values.insert(out);
                }
            }
        }

        let mut control_flow_output_values: HashSet<ValueId> = HashSet::new();
        for &nid in order {
            let node = graph.node(nid);
            if is_control_flow_op(&node.op_type, &node.domain) {
                for &out in &node.outputs {
                    control_flow_output_values.insert(out);
                }
            }
        }

        (sequence_values, control_flow_output_values)
    }

    /// Build the structural per-node plan in topological order, skipping
    /// pre-compiled EPContext nodes. Preserves the `session.execution_plan`
    /// tracing span and its arguments.
    fn build_node_plan(
        graph: &Graph,
        order: &[NodeId],
        value_dtypes: &HashMap<ValueId, DataType>,
        has_symbols: bool,
        weight_handles: &HashMap<ValueId, WeightHandle>,
        capabilities: &onnx_runtime_ep_api::ExecutionProviderCapabilities,
    ) -> Vec<NodePlan> {
        let mut plan_span = trace_span("session.execution_plan", "session");
        let mut plan = Vec::with_capacity(order.len());
        let mut skipped_epcontext = 0_u64;
        for &nid in order {
            let node = graph.node(nid);
            // EPContext nodes are pre-compiled: they bypass placement and were
            // already restored through their owning EP by the session's
            // consume path (§55.3). They must never be resolved as ordinary
            // kernels — the CPU EP has no `EPContext` kernel — so skip them
            // here.
            if onnx_runtime_loader::is_ep_context_op(&node.op_type, &node.domain) {
                if plan_span.is_some() {
                    skipped_epcontext += 1;
                }
                continue;
            }
            // Preserve positional input arity: keep interior `None` (omitted
            // optional) slots so a later present input is not misread as the
            // omitted one, but trim trailing `None`s (a trailing omitted
            // optional just lowers the arity, matching ONNX semantics).
            let mut slots: Vec<Option<ValueId>> = node.inputs.clone();
            while matches!(slots.last(), Some(None)) {
                slots.pop();
            }
            let inputs = slots;
            let outputs: Vec<ValueId> = node.outputs.clone();
            let input_dtypes: Vec<DataType> = inputs
                .iter()
                .map(|v| {
                    v.map(|vid| value_dtypes[&vid])
                        .unwrap_or(DataType::Undefined)
                })
                .collect();
            let output_dtypes: Vec<DataType> = outputs.iter().map(|v| value_dtypes[v]).collect();
            let mut lazy_weight_inputs = Vec::new();
            // Dense (`MatMul`/`MatMulNBits`) and `BlockQuantizedMoE` weights may
            // all be look-ahead prefetched: each boundary's EP-side
            // `prefetch_lazy_weight` decides independently whether it can act on
            // a given weight, so listing a boundary here only makes it eligible
            // to be *offered* early, never that it will be. `QMoE`'s routed
            // residency is intentionally excluded — it is a per-dispatch
            // whole-bank guard (`RoutedResidencyProof`/`acquire_routed_residency`,
            // #1797), not a look-ahead page-in, and mixing the two seams would
            // require a second residency authority for the same weight.
            if LazyWeightBoundary::MatMul.matches(&node.domain, &node.op_type)
                || LazyWeightBoundary::MatMulNBits.matches(&node.domain, &node.op_type)
                || LazyWeightBoundary::BlockQuantizedMoe.matches(&node.domain, &node.op_type)
            {
                for vid in inputs.iter().flatten() {
                    if weight_handles
                        .get(vid)
                        .is_some_and(|handle| handle.is_lazy_for(capabilities))
                        && !lazy_weight_inputs.contains(vid)
                    {
                        lazy_weight_inputs.push(*vid);
                    }
                }
            }
            plan.push(NodePlan {
                node_id: nid,
                inputs,
                outputs,
                input_dtypes,
                output_dtypes,
                inplace_dead_inputs: Vec::new(),
                dead_after: Vec::new(),
                lazy_weight_inputs,
            });
        }
        let graph_outputs: HashSet<ValueId> = graph.outputs.iter().copied().collect();
        // Outer values each control-flow node implicitly captures by name from
        // the enclosing scope (see `control_flow_captures_by_node`). These never
        // appear in any plan node's formal `inputs`, so liveness must treat the
        // capturing node as a use site — otherwise an earlier in-place alias
        // could free a buffer an If/Loop/Scan body still reads at runtime.
        let captures_by_node = Self::control_flow_captures_by_node(graph);
        let mut last_use = HashMap::new();
        for (pi, node) in plan.iter().enumerate() {
            for vid in node.inputs.iter().flatten() {
                last_use.insert(*vid, pi);
            }
            if let Some(captured) = captures_by_node.get(&node.node_id) {
                for vid in captured {
                    // `pi` increases monotonically, so this keeps the maximum
                    // (latest) use position across formal inputs and captures.
                    last_use.insert(*vid, pi);
                }
            }
        }
        for (pi, node) in plan.iter_mut().enumerate() {
            node.inplace_dead_inputs = node
                .inputs
                .iter()
                .map(|input| {
                    input.is_some_and(|vid| {
                        last_use.get(&vid) == Some(&pi) && !graph_outputs.contains(&vid)
                    })
                })
                .collect();
            // Values this node consumes for the last time. Weights are excluded
            // here rather than at runtime because an initializer's buffer is
            // built once and reused by every later run -- releasing it after its
            // final *use in this run* would leave the next run with no weights.
            // Graph inputs are excluded for the same reason at a different
            // lifetime: their storage belongs to the caller's binding.
            let mut seen = std::collections::HashSet::new();
            let dead_after: Vec<ValueId> = node
                .inputs
                .iter()
                .flatten()
                .copied()
                .filter(|vid| {
                    last_use.get(vid) == Some(&pi)
                        && !graph_outputs.contains(vid)
                        && !graph.initializers.contains_key(vid)
                        && !graph.inputs.contains(vid)
                        && seen.insert(*vid)
                })
                .collect();
            node.dead_after = dead_after;
        }
        if let Some(span) = plan_span.as_mut() {
            span.set_args(
                Args::new()
                    .with("topological_nodes", order.len() as u64)
                    .with("plan_len", plan.len() as u64)
                    .with("skipped_epcontext_nodes", skipped_epcontext)
                    .with("values", graph.values.len() as u64)
                    .with("inputs", graph.inputs.len() as u64)
                    .with("outputs", graph.outputs.len() as u64)
                    .with("has_symbols", has_symbols),
            );
        }
        plan
    }

    /// Map each control-flow node to the outer [`ValueId`]s its subgraph bodies
    /// implicitly capture by name. If/Loop/Scan bodies reference free variables
    /// from the enclosing scope (resolved at runtime by `prepare_subgraph` via
    /// [`required_outer_names`] and the name index); those captures never appear
    /// in the node's formal `inputs`, so callers that reason about value
    /// liveness must consult this map to avoid freeing a still-captured buffer.
    fn control_flow_captures_by_node(graph: &Graph) -> HashMap<NodeId, HashSet<ValueId>> {
        let mut captures_by_node: HashMap<NodeId, HashSet<ValueId>> = HashMap::new();
        if graph.subgraphs.is_empty() {
            return captures_by_node;
        }
        let name_index: HashMap<&str, ValueId> = graph
            .values
            .iter()
            .filter_map(|(vid, value)| value.name.as_deref().map(|name| (name, vid)))
            .collect();
        for ((owner, _attr_key), body) in graph.subgraphs.iter() {
            let entry = captures_by_node.entry(*owner).or_default();
            for name in required_outer_names(body) {
                if let Some(&vid) = name_index.get(name.as_str()) {
                    entry.insert(vid);
                }
            }
        }
        captures_by_node.retain(|_, values| !values.is_empty());
        captures_by_node
    }

    /// required_inputs, name_index)`: the caller-input name map and the set of
    /// caller-required inputs (graph inputs that are not pre-filled
    /// initializers), plus the full name → value id map over every named value
    /// (used to resolve a nested subgraph's outer-scope captures by name).
    fn build_name_indexes(
        graph: &Graph,
    ) -> (
        HashMap<String, ValueId>,
        Vec<ValueId>,
        HashMap<String, ValueId>,
    ) {
        let mut input_index = HashMap::new();
        let mut required_inputs = Vec::new();
        for &vid in &graph.inputs {
            if graph.initializers.contains_key(&vid) {
                continue; // pre-filled; not a caller input
            }
            required_inputs.push(vid);
            if let Some(name) = &graph.value(vid).name {
                input_index.insert(name.clone(), vid);
            }
        }

        let mut name_index = HashMap::new();
        for (vid, value) in graph.values.iter() {
            if let Some(name) = &value.name {
                name_index.insert(name.clone(), vid);
            }
        }

        (input_index, required_inputs, name_index)
    }

    /// Allocate `vid`'s buffer for `dims`, or reuse the existing allocation when
    /// it is already sized for `dims` (the run-scoped reuse path).
    pub(super) fn ensure_buffer(
        &mut self,
        vid: ValueId,
        dtype: DataType,
        dims: &[usize],
    ) -> Result<()> {
        if self.buffer_shapes.get(&vid).map(|s| s.as_slice()) == Some(dims) {
            return Ok(()); // identical shape → reuse allocation
        }
        if let Some(old) = self.buffers.remove(&vid) {
            self.ep.deallocate(old)?;
        }
        self.shared_buffers.remove(&vid);
        let numel = checked_numel(dims, || format!("value#{}", vid.0))?;
        let size = checked_storage_bytes(dtype, numel, || format!("value#{}", vid.0), dims)?;
        let buf = self
            .ep
            .allocate(size.max(1), TensorLayout::contiguous().alignment)?;
        self.buffers.insert(vid, buf);
        self.buffer_shapes.insert(vid, dims.to_vec());
        Ok(())
    }

    /// Resolve every value's concrete shape by substituting `bindings` into its
    /// loader shape. A value whose shape stays symbolic (unbound) cannot be
    /// sized: report it as an uninferred shape, naming its producing op.
    pub(super) fn resolve_all(
        &self,
        bindings: &HashMap<SymbolId, usize>,
    ) -> Result<HashMap<ValueId, Vec<usize>>> {
        let mut resolved = HashMap::with_capacity(self.value_shapes.len());
        for (&vid, shape) in &self.value_shapes {
            // Sequence-typed values have no meaningful tensor shape and are
            // never buffer-sized; skip them so a static graph does not trip the
            // unresolved-shape check on a sequence value.
            if self.sequence_values.contains(&vid) {
                continue;
            }
            match substitute(shape, bindings) {
                Some(dims) => {
                    resolved.insert(vid, dims);
                }
                None => {
                    let value = self.graph.value(vid);
                    let name = value
                        .name
                        .clone()
                        .unwrap_or_else(|| format!("value#{}", vid.0));
                    let op = value
                        .producer
                        .map(|nid| self.graph.node(nid).op_type.clone())
                        .unwrap_or_else(|| "<graph input>".to_string());
                    return Err(SessionError::UnresolvedShape { value: name, op });
                }
            }
        }
        Ok(resolved)
    }

    /// Like [`Self::resolve_all`] but never errors: values whose shape stays
    /// symbolic (a data-dependent extent the loader could not pin down) are
    /// simply omitted, to be resolved just-in-time during execution once their
    /// producing node's inputs are concrete.
    pub(super) fn resolve_soft(
        &self,
        bindings: &HashMap<SymbolId, usize>,
    ) -> HashMap<ValueId, Vec<usize>> {
        let mut resolved = HashMap::with_capacity(self.value_shapes.len());
        for (&vid, shape) in &self.value_shapes {
            if let Some(dims) = substitute(shape, bindings) {
                resolved.insert(vid, dims);
            }
        }
        resolved
    }

    /// F5 Stage 1: resolve every value's concrete shape for a memo-eligible
    /// eager step, replaying the length-invariant partition through the
    /// [`DecodePlanMemo`] when the step is plan-identical to the memoized one,
    /// and re-substituting only the length-varying tail. On any signature change
    /// (prefill→decode, batch change, non-length dim change, …) it falls back to
    /// a full [`Self::resolve_soft`] and (re)builds the memo by diffing this
    /// step's bindings with the previous eligible step's (R1 two-real-step
    /// derivation). The output is provably byte-identical to `resolve_soft`
    /// (asserted every replay when [`Self::decode_memo_verify`] is set).
    pub(super) fn resolve_soft_decode_memo(
        &mut self,
        bindings: &HashMap<SymbolId, usize>,
        external: &ExternalBindings,
    ) -> HashMap<ValueId, Vec<usize>> {
        // L-abstracted fingerprint of the persistent binding set (KV cache). Pure
        // length growth leaves it unchanged; a structural change forces a rebuild.
        let external_sig = self.decode_external_signature(external);
        // --- Fast path: an active memo whose non-varying bindings and binding
        //     signature are unchanged. Replays the invariant partition with ZERO
        //     allocation: the persistent working map is taken in place, the
        //     previous step's just-in-time entries are stripped, invariant entries
        //     are left untouched (byte-identical by construction), and only the
        //     variant tail is re-substituted into its existing `Vec`s.
        if self
            .decode_memo
            .as_ref()
            .is_some_and(|memo| memo.matches(bindings, &external_sig))
        {
            // Own the memo for the duration so `self.value_shapes` /
            // `decode_memo_resolved` can be borrowed disjointly; restored below.
            let memo = self.decode_memo.take().unwrap();
            let mut resolved = std::mem::take(&mut self.decode_memo_resolved);
            // Drop the previous step's data-dependent (JIT) entries so the run
            // loop recomputes them; the canonical partition is retained in place.
            resolved.retain(|vid, _| memo.canonical.contains(vid));
            // Restore any length-invariant entry missing from the persistent map.
            // By construction (the run loop only adds/overwrites, never drops
            // canonical keys, and the rebuild step persisted the full map) this
            // never fires in steady state, so replay stays allocation-free; it is
            // a defensive re-seed from the memo's authoritative invariant plan.
            for (&vid, dims) in &memo.invariant_shapes {
                resolved.entry(vid).or_insert_with(|| dims.clone());
            }
            // Re-substitute only the variant tail, reusing each `Vec`'s capacity.
            for &vid in &memo.variant_values {
                let shape = &self.value_shapes[&vid];
                match resolved.get_mut(&vid) {
                    Some(slot) => {
                        if !substitute_into(shape, bindings, slot) {
                            resolved.remove(&vid);
                        }
                    }
                    None => {
                        if let Some(dims) = substitute(shape, bindings) {
                            resolved.insert(vid, dims);
                        }
                    }
                }
            }
            if self.decode_memo_verify {
                // R1 verifiable safety net: the replay must equal a fresh resolve.
                let fresh = self.resolve_soft(bindings);
                assert_eq!(
                    resolved, fresh,
                    "decode-plan memo replay diverged from resolve_soft (unsound invariant \
                     classification)"
                );
            }
            self.decode_memo = Some(memo);
            self.decode_memo_last_action = DecodeMemoAction::Replayed;
            self.decode_memo_replayed_count += 1;
            self.decode_memo_prev_bindings = Some(bindings.clone());
            return resolved;
        }

        // --- Slow path: full resolve, then try to (re)build the memo by diffing
        //     this step with the previous eligible step (two real steps, R1) —
        //     but only for a steady single-token-decode growth transition (M==1
        //     gate), so the memo never activates on prefill.
        //
        // Defense-in-depth (Chew): drop the persistent working map on every
        // non-replay step so a stale invariant `Vec` from a retired plan can
        // never leak into a future replay (e.g. if a run errored before the
        // end-of-step persist-back). It is repopulated by this step's persist-back
        // (or, if that step errors, left empty and defensively re-seeded next
        // replay), so the clear costs nothing on the steady path.
        self.decode_memo_resolved.clear();
        // F5 Stage 2 defense-in-depth: retire the cached view plan on every
        // non-replay (rebuild/prime) step. A Rebuilt step rebuilds it fresh only
        // at its successful end (below, in `run_scoped_mode`); a step that errors
        // before that leaves it `None`, so a stale invariant view alias from a
        // retired plan can never be reinstated into a later replay.
        self.decode_view_plan = None;
        let resolved = self.resolve_soft(bindings);
        match self.decode_memo_prev_bindings.take() {
            Some(prev) if is_decode_growth_transition(&prev, bindings) => {
                let decode_varying: HashSet<SymbolId> = bindings
                    .iter()
                    .filter(|(sym, val)| prev.get(*sym) != Some(*val))
                    .map(|(&sym, _)| sym)
                    .collect();
                let mut invariant_shapes = HashMap::with_capacity(resolved.len());
                let mut variant_values = Vec::new();
                let mut canonical = HashSet::with_capacity(resolved.len());
                for (&vid, dims) in &resolved {
                    canonical.insert(vid);
                    if shape_references_any(&self.value_shapes[&vid], &decode_varying) {
                        variant_values.push(vid);
                    } else {
                        invariant_shapes.insert(vid, dims.clone());
                    }
                }
                self.decode_memo = Some(DecodePlanMemo {
                    reference_bindings: bindings.clone(),
                    decode_varying,
                    invariant_shapes,
                    variant_values,
                    canonical,
                    reference_external_sig: external_sig,
                });
                self.decode_memo_last_action = DecodeMemoAction::Rebuilt;
                self.decode_memo_rebuilt_count += 1;
            }
            _ => {
                // First observation of a regime, a bound-symbol-set change, or a
                // non-decode transition (e.g. prefill): drop any stale memo and
                // wait for the next steady-decode step to diff against.
                self.decode_memo = None;
                self.decode_memo_last_action = DecodeMemoAction::Primed;
                self.decode_memo_primed_count += 1;
            }
        }
        self.decode_memo_prev_bindings = Some(bindings.clone());
        resolved
    }

    /// L-abstracted structural fingerprint of the persistent device-I/O binding
    /// set (see [`DecodeBindingSig`]). Order-independent; the declared symbolic
    /// shape (graph-static) stands in for the concrete one, so pure-L KV growth
    /// yields an unchanged signature while a binding added/removed, a role flip,
    /// or a dtype change yields a different one.
    pub(super) fn decode_external_signature(
        &self,
        external: &ExternalBindings,
    ) -> Vec<DecodeBindingSig> {
        let mut sig: Vec<DecodeBindingSig> = external
            .inputs
            .keys()
            .map(|&vid| (vid, true))
            .chain(external.outputs.keys().map(|&vid| (vid, false)))
            .map(|(vid, is_input)| DecodeBindingSig {
                vid,
                is_input,
                dtype: self.value_dtypes[&vid],
                decl_shape: self.value_shapes[&vid].clone(),
            })
            .collect();
        sig.sort_by_key(|s| (s.vid.0, s.is_input));
        sig
    }

    #[cfg(test)]
    pub(super) fn set_decode_memo_enabled(&mut self, enabled: bool) {
        self.decode_memo_enabled = enabled;
        self.decode_memo_verify = true;
        self.decode_memo = None;
        self.decode_memo_prev_bindings = None;
        self.decode_memo_resolved.clear();
        self.decode_memo_last_action = DecodeMemoAction::Disabled;
        self.decode_memo_primed_count = 0;
        self.decode_memo_rebuilt_count = 0;
        self.decode_memo_replayed_count = 0;
        self.decode_memo_ineligible_count = 0;
        self.decode_view_plan = None;
        self.decode_views_reused_count = 0;
        self.decode_dispatch_elided_count = 0;
        self.decode_view_plan_sig_mismatch_streak = 0;
        self.decode_view_plan_disabled = false;
    }

    #[cfg(test)]
    pub(super) fn decode_memo_action(&self) -> DecodeMemoAction {
        self.decode_memo_last_action
    }

    /// F5 Stage 1 memo activity counters `(primed, rebuilt, replayed, ineligible)`
    /// accumulated over this executor's lifetime. `replayed > 0` on a real decode
    /// run is the proof the memo actually fires (not silently gated out); the
    /// coordinator's on-model A/B reads these to reject a vacuous pass.
    pub(crate) fn decode_memo_counts(&self) -> (u64, u64, u64, u64) {
        (
            self.decode_memo_primed_count,
            self.decode_memo_rebuilt_count,
            self.decode_memo_replayed_count,
            self.decode_memo_ineligible_count,
        )
    }

    /// F5 Stage 2 activity counters `(views_reused, dispatch_elided)` accumulated
    /// over this executor's lifetime. Both `> 0` on a real decode run prove the
    /// invariant view-reuse / dispatch-elision path actually fired (not a vacuous
    /// pass); an on-model A/B reads these alongside the Stage-1 counters.
    pub(crate) fn decode_view_plan_counts(&self) -> (u64, u64) {
        (
            self.decode_views_reused_count,
            self.decode_dispatch_elided_count,
        )
    }

    /// How many times the single-trip `Scan` inline path engaged over this
    /// executor's lifetime. `> 0` after a decode run proves the dual-path is
    /// non-vacuously firing (an on-model A/B reads this to reject a silently
    /// gated-out pass); stays 0 whenever the flag is OFF or every `Scan` runs at
    /// `trip_count != 1`.
    pub(crate) fn scan_inline_single_trip_count(&self) -> u64 {
        self.scan_inline_single_trip_count
    }

    /// F5 Stage 2 replay guard: every retained view's source buffer must still be
    /// the identical allocation (same base pointer *and* capacity) it was under
    /// when the plan was built. A realloc or move — even one that preserves the
    /// logical shape — invalidates the cached byte offsets/strides, so this must
    /// return `false` and force a full rebuild. This is the pointer/capacity
    /// obligation Stage 1 deferred (it cached shapes only); Stage 2 pays it here.
    pub(super) fn stage2_buffer_sig_matches(&self, plan: &DecodeViewPlan) -> bool {
        plan.source_buffer_sig.iter().all(|(vid, ptr, cap)| {
            self.buffers
                .get(vid)
                .is_some_and(|buf| buf.as_ptr() as usize == *ptr && buf.len() == *cap)
        })
    }

    /// F5 Stage 2: build the *candidate* view plan from the state left by a
    /// successful memo Rebuilt step. A node is a candidate iff every one of its
    /// outputs is a zero-copy view (`self.views`) whose **shape is in the memo's
    /// proven-invariant partition** — so Stage 1 guarantees the replayed `resolved`
    /// map carries that exact shape every step. The candidate's source buffers can
    /// still be classified variant (e.g. a fixed-capacity KV buffer whose logical
    /// length grows): its concrete stability is confirmed separately by
    /// [`Self::validate_decode_view_plan`] (byte-identical view across a second real
    /// step) and guarded each replay by the buffer-identity signature. Returns
    /// `None` if nothing is a candidate.
    pub(super) fn build_decode_view_plan(&self) -> Option<DecodeViewPlan> {
        let memo = self.decode_memo.as_ref()?;
        let invariant = |vid: &ValueId| memo.invariant_shapes.contains_key(vid);
        let mut elided_nodes = HashSet::new();
        let mut retained_views = Vec::new();
        let mut sources: HashSet<ValueId> = HashSet::new();
        for pi in 0..self.plan.len() {
            let outputs = &self.plan[pi].outputs;
            if outputs.is_empty() {
                continue;
            }
            // Every output must be a zero-copy view whose shape Stage 1 already
            // proves invariant (so `resolved[output]` is stable and correct when
            // the node is elided).
            let all_view_invariant = outputs
                .iter()
                .all(|ovid| invariant(ovid) && self.views.contains_key(ovid));
            if !all_view_invariant {
                continue;
            }
            elided_nodes.insert(pi);
            for ovid in outputs {
                let view = self.views[ovid].clone();
                sources.insert(view.source);
                retained_views.push((*ovid, view));
            }
        }
        if elided_nodes.is_empty() {
            return None;
        }
        // Record the buffer identity of every aliased source (the Stage-2 guard).
        let mut source_buffer_sig = Vec::with_capacity(sources.len());
        for &src in &sources {
            let buf = self.buffers.get(&src)?;
            source_buffer_sig.push((src, buf.as_ptr() as usize, buf.len()));
        }
        Some(DecodeViewPlan {
            elided_nodes,
            retained_views,
            pinned_sources: sources.into_iter().collect(),
            source_buffer_sig,
            validated: false,
        })
    }

    /// F5 Stage 2: confirm a candidate plan on a second real decode step. The step
    /// ran every node normally (no elision), so `self.views` now holds freshly
    /// built aliases; keep only the candidate nodes whose every output view is
    /// **byte-identical** (source, shape, strides, byte offset) to the one captured
    /// when the plan was built. This two-real-step confirmation (mirroring Stage 1's
    /// varying-set derivation) rejects any view whose geometry actually drifts — e.g.
    /// a position-indexed slice into a fixed-capacity buffer — before it is ever
    /// elided. Sources and the buffer-identity signature are recomputed from the
    /// surviving views. The plan is marked validated iff anything survives.
    pub(super) fn validate_decode_view_plan(
        &self,
        mut plan: DecodeViewPlan,
    ) -> Option<DecodeViewPlan> {
        let view_matches = |a: &ValueView, b: &ValueView| {
            a.source == b.source
                && a.shape == b.shape
                && a.strides == b.strides
                && a.byte_offset == b.byte_offset
        };
        // A node survives iff every one of its retained outputs still matches the
        // freshly rebuilt view this step.
        let mut surviving_nodes: HashSet<usize> = HashSet::new();
        let node_outputs = |pi: usize| self.plan[pi].outputs.clone();
        for &pi in &plan.elided_nodes {
            let ok = node_outputs(pi).iter().all(|ovid| {
                match (
                    plan.retained_views.iter().find(|(v, _)| v == ovid),
                    self.views.get(ovid),
                ) {
                    (Some((_, cached)), Some(fresh)) => view_matches(cached, fresh),
                    _ => false,
                }
            });
            if ok {
                surviving_nodes.insert(pi);
            }
        }
        if surviving_nodes.is_empty() {
            return None;
        }
        // Rebuild retained views / sources / signature from the survivors only,
        // using the freshly built (identical) views.
        let surviving_outputs: HashSet<ValueId> = surviving_nodes
            .iter()
            .flat_map(|&pi| self.plan[pi].outputs.clone())
            .collect();
        let mut retained_views = Vec::new();
        let mut sources: HashSet<ValueId> = HashSet::new();
        for ovid in surviving_outputs {
            let view = self.views.get(&ovid)?.clone();
            sources.insert(view.source);
            retained_views.push((ovid, view));
        }
        let mut source_buffer_sig = Vec::with_capacity(sources.len());
        for &src in &sources {
            let buf = self.buffers.get(&src)?;
            source_buffer_sig.push((src, buf.as_ptr() as usize, buf.len()));
        }
        plan.elided_nodes = surviving_nodes;
        plan.retained_views = retained_views;
        plan.pinned_sources = sources.into_iter().collect();
        plan.source_buffer_sig = source_buffer_sig;
        plan.validated = true;
        Some(plan)
    }

    /// Size (allocate or reuse) a backing buffer for every value from its
    /// resolved concrete shape. Initializers already hold their weights and are
    /// left untouched. Values whose shape is not (yet) in `resolved` — the
    /// data-dependent ones filled in during execution — are skipped here and
    /// sized just-in-time in the run loop.
    pub(super) fn size_buffers(&mut self, resolved: &HashMap<ValueId, Vec<usize>>) -> Result<()> {
        self.size_buffers_excluding(resolved, &HashSet::new())
    }

    pub(super) fn size_buffers_excluding(
        &mut self,
        resolved: &HashMap<ValueId, Vec<usize>>,
        excluded: &HashSet<ValueId>,
    ) -> Result<()> {
        let vids: Vec<ValueId> = self.value_shapes.keys().copied().collect();
        for vid in vids {
            if self.graph.initializers.contains_key(&vid) || excluded.contains(&vid) {
                continue;
            }
            // Sequence-typed values own no tensor buffer (their list lives in
            // `sequences` at run time), so never size one for them.
            if self.sequence_values.contains(&vid) {
                continue;
            }
            let dtype = self.value_dtypes[&vid];
            let Some(dims) = resolved.get(&vid).cloned() else {
                continue;
            };
            self.ensure_buffer(vid, dtype, &dims)?;
        }
        Ok(())
    }

    /// Resolved input shapes of a plan node, in positional order. An omitted
    /// optional input (`None` slot) has no shape; it takes an empty shape,
    /// which the run loop only ever pairs with an absent placeholder view.
    pub(super) fn node_input_shapes(
        plan: &NodePlan,
        resolved: &HashMap<ValueId, Vec<usize>>,
    ) -> Vec<Vec<usize>> {
        plan.inputs
            .iter()
            .map(|v| v.map(|vid| resolved[&vid].clone()).unwrap_or_default())
            .collect()
    }

    /// Populate the kernel cache for the compiled plan against `resolved` shapes.
    pub(super) fn compile_all(&mut self, resolved: &HashMap<ValueId, Vec<usize>>) -> Result<()> {
        let mut span = trace_span("session.kernel_compile_plan", "session");
        let cache_entries_before = self.cache.stats().entries;
        let mut compiled_nodes = 0_u64;
        let mut skipped_control_flow = 0_u64;
        let mut skipped_sequence = 0_u64;
        for i in 0..self.plan.len() {
            let node_id = self.plan[i].node_id;
            let node = self.graph.node(node_id);
            // Control-flow ops (If/Loop/Scan) are not leaf kernels — they execute
            // nested subgraphs through the executor's own path, so they have no
            // entry in the EP kernel registry and must not be compiled here.
            if is_control_flow_op(&node.op_type, &node.domain) {
                if span.is_some() {
                    skipped_control_flow += 1;
                }
                continue;
            }
            // Sequence ops are executor-handled (they operate on sequence-of-
            // tensor values, not tensor views) — they have no EP kernel and must
            // not be compiled here, exactly like control-flow ops.
            if is_sequence_op(&node.op_type, &node.domain) {
                if span.is_some() {
                    skipped_sequence += 1;
                }
                continue;
            }
            if span.is_some() {
                compiled_nodes += 1;
            }
            let input_shapes = Self::node_input_shapes(&self.plan[i], resolved);
            let input_dtypes = self.plan[i].input_dtypes.clone();
            let constant_inputs: Vec<bool> = self.plan[i]
                .inputs
                .iter()
                .map(|input| input.is_some_and(|vid| self.graph.initializers.contains_key(&vid)))
                .collect();
            let node = self.graph.node(node_id);
            let opset = effective_opset(&self.graph, node);
            let seq_independent =
                node_capture_seq_independent(&self.graph, node, &self.capture_growing_symbols);
            let (_, key) = self.cache.get_or_create(
                node_id,
                node,
                &input_shapes,
                &input_dtypes,
                &constant_inputs,
                opset,
                seq_independent,
                self.ep.as_ref(),
            )?;
            // Pre-populate the kernel binding so the first decode step already
            // hits the zero-alloc fast path for static-shape graphs.
            self.kernel_bindings[i] = Some(key);
        }
        if let Some(span) = span.as_mut() {
            span.set_args(
                Args::new()
                    .with("plan_len", self.plan.len() as u64)
                    .with("compiled_nodes", compiled_nodes)
                    .with("skipped_control_flow", skipped_control_flow)
                    .with("skipped_sequence", skipped_sequence)
                    .with("cache_entries_before", cache_entries_before as u64)
                    .with("cache_entries_after", self.cache.stats().entries as u64),
            );
        }
        Ok(())
    }

    pub(crate) fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    pub(crate) fn control_flow_stats(&self) -> ControlFlowStats {
        self.control_flow_stats
    }

    pub(crate) fn device_id(&self) -> onnx_runtime_ir::DeviceId {
        self.ep.device_id()
    }

    pub(crate) fn allocate_device_binding(
        &self,
        input_name: String,
        output_name: Option<String>,
        dtype: DataType,
        physical_shape: Vec<usize>,
        logical_shape: Vec<usize>,
    ) -> Result<DeviceIoBinding> {
        let expose_logical_input_shape = self.input_index.get(&input_name).is_some_and(|&vid| {
            if output_name.is_some() {
                !self.binding_consumers_use_physical_capacity(vid)
            } else {
                !self.binding_consumers_use_padded_capacity(vid)
            }
        });
        let decode_freeze_safe_mask = self
            .input_index
            .get(&input_name)
            .is_some_and(|&vid| self.binding_mask_is_decode_freeze_safe(vid));
        DeviceIoBinding::allocate(
            self.ep.clone(),
            DeviceBindingSpec {
                input_name,
                bind_input: true,
                output_name,
                dtype,
                physical_shape,
                logical_shape,
                expose_logical_input_shape,
                decode_freeze_safe_mask,
                fixed_physical_strides: false,
                allocation_bytes: None,
                committed_ranges: None,
            },
        )
    }

    pub(crate) fn allocate_device_binding_fixed_strides(
        &self,
        input_name: String,
        output_name: String,
        dtype: DataType,
        physical_shape: Vec<usize>,
        logical_shape: Vec<usize>,
    ) -> Result<DeviceIoBinding> {
        DeviceIoBinding::allocate(
            self.ep.clone(),
            DeviceBindingSpec {
                input_name,
                bind_input: true,
                output_name: Some(output_name),
                dtype,
                physical_shape,
                logical_shape,
                expose_logical_input_shape: true,
                decode_freeze_safe_mask: false,
                fixed_physical_strides: true,
                allocation_bytes: None,
                committed_ranges: None,
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn allocate_device_binding_committed(
        &self,
        input_name: String,
        output_name: Option<String>,
        dtype: DataType,
        physical_shape: Vec<usize>,
        logical_shape: Vec<usize>,
        allocation_bytes: usize,
        committed_ranges: Vec<std::ops::Range<usize>>,
    ) -> Result<DeviceIoBinding> {
        let expose_logical_input_shape = self.input_index.get(&input_name).is_some_and(|&vid| {
            if output_name.is_some() {
                !self.binding_consumers_use_physical_capacity(vid)
            } else {
                !self.binding_consumers_use_padded_capacity(vid)
            }
        });
        let decode_freeze_safe_mask = self
            .input_index
            .get(&input_name)
            .is_some_and(|&vid| self.binding_mask_is_decode_freeze_safe(vid));
        DeviceIoBinding::allocate(
            self.ep.clone(),
            DeviceBindingSpec {
                input_name,
                bind_input: true,
                output_name,
                dtype,
                physical_shape,
                logical_shape,
                expose_logical_input_shape,
                decode_freeze_safe_mask,
                fixed_physical_strides: false,
                allocation_bytes: Some(allocation_bytes),
                committed_ranges: Some(committed_ranges),
            },
        )
    }

    /// Bind a buffer the **caller** allocated instead of allocating one here.
    ///
    /// # Safety
    ///
    /// See [`DeviceIoBinding::from_external_memory`]: the buffer must be large
    /// enough (checked), and must outlive the binding and every run that
    /// touches it (not checkable).
    pub(crate) unsafe fn device_binding_from_external_memory(
        &self,
        spec: crate::tensor::ExternalMemorySpec,
    ) -> Result<DeviceIoBinding> {
        let crate::tensor::ExternalMemorySpec {
            input_name,
            bind_input,
            output_name,
            dtype,
            physical_shape,
            logical_shape,
            ptr,
            len_bytes,
        } = spec;
        if !bind_input && output_name.is_none() {
            return Err(SessionError::ExternalBuffer {
                binding: input_name,
                reason: "it binds neither an input nor an output, so nothing would ever \
                         read or write it"
                    .to_string(),
            });
        }
        let expose_logical_input_shape = self.input_index.get(&input_name).is_some_and(|&vid| {
            if output_name.is_some() {
                !self.binding_consumers_use_physical_capacity(vid)
            } else {
                !self.binding_consumers_use_padded_capacity(vid)
            }
        });
        let decode_freeze_safe_mask = self
            .input_index
            .get(&input_name)
            .is_some_and(|&vid| self.binding_mask_is_decode_freeze_safe(vid));
        // SAFETY: delegated to this function's contract.
        unsafe {
            DeviceIoBinding::from_external_memory(
                self.ep.clone(),
                DeviceBindingSpec {
                    input_name,
                    bind_input,
                    output_name,
                    dtype,
                    physical_shape,
                    logical_shape,
                    expose_logical_input_shape,
                    decode_freeze_safe_mask,
                    fixed_physical_strides: false,
                    allocation_bytes: None,
                    committed_ranges: None,
                },
                ptr,
                len_bytes,
            )
        }
    }

    pub(crate) fn allocate_device_output_binding(
        &self,
        output_name: String,
        dtype: DataType,
        physical_shape: Vec<usize>,
        logical_shape: Vec<usize>,
    ) -> Result<DeviceIoBinding> {
        DeviceIoBinding::allocate(
            self.ep.clone(),
            DeviceBindingSpec {
                input_name: String::new(),
                bind_input: false,
                output_name: Some(output_name),
                dtype,
                physical_shape,
                logical_shape,
                expose_logical_input_shape: false,
                decode_freeze_safe_mask: false,
                fixed_physical_strides: false,
                allocation_bytes: None,
                committed_ranges: None,
            },
        )
    }

    pub(super) fn binding_consumers_use_physical_capacity(&self, input: ValueId) -> bool {
        let mut found = false;
        for plan in &self.plan {
            for (slot, value) in plan.inputs.iter().enumerate() {
                if *value != Some(input) {
                    continue;
                }
                found = true;
                if !kernel_input_uses_physical_capacity(self.graph.node(plan.node_id), slot) {
                    return false;
                }
            }
        }
        found
    }

    /// Name the direct consumers of `input` that are **not** padded-capacity-safe
    /// — the ones that force the binding to expose its logical valid length and
    /// therefore forfeit CUDA-graph capture. The predicate itself only answers
    /// yes/no, so bring-up on a new architecture otherwise has to re-derive the
    /// offender from the graph by hand; report it instead of inferring it.
    pub(super) fn padded_capacity_offenders(&self, input: ValueId) -> Vec<String> {
        let mut offenders = Vec::new();
        for plan in &self.plan {
            for (slot, value) in plan.inputs.iter().enumerate() {
                if *value != Some(input) {
                    continue;
                }
                let node = self.graph.node(plan.node_id);
                if let Some(described) = describe_non_padded_consumer(node, slot) {
                    offenders.push(described);
                }
            }
        }
        offenders
    }

    pub(super) fn binding_consumers_use_padded_capacity(&self, input: ValueId) -> bool {
        let padded = self.binding_consumers_use_padded_capacity_inner(input);
        if !padded {
            self.report_padded_capacity_decline(input);
        }
        padded
    }

    /// Emit the attribution for a padded-capacity decline. Gated behind
    /// `ONNX_GENAI_CUDA_DEBUG_MASK_CAPACITY` so the default path stays silent,
    /// matching the RMSNorm-fold attribution added in #1671.
    fn report_padded_capacity_decline(&self, input: ValueId) {
        let offenders = self.padded_capacity_offenders(input);
        if offenders.is_empty() {
            return;
        }
        if std::env::var("ONNX_GENAI_CUDA_DEBUG_MASK_CAPACITY").is_ok_and(|v| v != "0") {
            eprintln!(
                "mask_capacity_decline: non-capacity-aware consumer(s) {}",
                offenders.join(", ")
            );
            // The direct-consumer list only explains the fast path. When the
            // topology-gated fallbacks also declined, name the reason each gave.
            // `Disqualify` drives static freezing; `Allow` is the weaker
            // decode-freeze-safe classification that actually decides whether a
            // single-token decode step may still capture, so report both.
            if let Some(reason) =
                mask_cone_rejection(&self.graph, input, ShapeConsumptionPolicy::Disqualify)
            {
                eprintln!("mask_capacity_decline: static freeze rejected: {reason}");
            }
            match mask_cone_rejection(&self.graph, input, ShapeConsumptionPolicy::Allow) {
                Some(reason) => {
                    eprintln!("mask_capacity_decline: decode-freeze-safe rejected: {reason}")
                }
                None => eprintln!(
                    "mask_capacity_decline: decode-freeze-safe HOLDS (capture may still proceed)"
                ),
            }
        }
    }

    fn binding_consumers_use_padded_capacity_inner(&self, input: ValueId) -> bool {
        let mut found = false;
        let mut all_direct_padded = true;
        for plan in &self.plan {
            for (slot, value) in plan.inputs.iter().enumerate() {
                if *value != Some(input) {
                    continue;
                }
                found = true;
                if !kernel_input_uses_padded_capacity(self.graph.node(plan.node_id), slot) {
                    all_direct_padded = false;
                }
            }
        }
        // Fast path: every direct consumer already reads the physical extent
        // (`Shape`/`ReduceSum`), as in dense GQA masks (mask → ReduceSum→seqlens_k
        // and Shape only).
        if found && all_direct_padded {
            return true;
        }
        // Topology-gated path: the mask binding feeds *only* the standard additive
        // causal-mask builder, terminating at capacity-form `Attention` mask
        // inputs (the DeepSeek-V2-Lite / MLA shape). There the frozen (physical
        // width) mask yields a byte-identical additive bias, so the binding is
        // padded-capacity-safe even though its cone contains prefix-sensitive
        // `CumSum`/`Unsqueeze` (which stay non-padded-safe for any other topology,
        // e.g. GLM-5.2's indexer `Add`).
        mask_binding_feeds_capacity_form_attention(&self.graph, input)
    }

    /// Whether the mask binding `input` is *decode-freeze-safe*: it feeds only the
    /// additive causal-mask builder cone terminating at capacity-form `Attention`,
    /// so a single-token decode step (`q_seq == 1`) can freeze the mask to
    /// physical capacity even when [`Self::binding_consumers_use_padded_capacity`]
    /// declines to freeze it *statically* (because `Shape(mask)` leaks the padded
    /// width into the multi-token prefill query-position window). The decode
    /// window saturates at `q_seq == 1`, so a frozen decode mask is byte-identical
    /// and keeps `logical == physical`, restoring CUDA-graph capture eligibility
    /// for these models (e.g. DeepSeek-V2-Lite). GLM-5.2's indexer `Add` cone is
    /// still excluded, so it is not decode-freeze-safe.
    pub(super) fn binding_mask_is_decode_freeze_safe(&self, input: ValueId) -> bool {
        mask_binding_feeds_additive_causal_builder(&self.graph, input)
    }

    /// The compiled graph, retained for the §55.4 EPContext dump path: the
    /// exporter needs the (post-optimize) graph to serialise a `*_ctx.onnx`
    /// context-cache model with compiled partitions spliced out.
    pub(crate) fn graph(&self) -> &Graph {
        &self.graph
    }

    pub(crate) fn execution_provider_fallback_report(
        &self,
    ) -> Option<&ExecutionProviderFallbackReport> {
        self.execution_provider_fallback_report.as_ref()
    }

    /// Attach the shared runtime trace context. When enabled, the executor opens
    /// one span per executed op so kernels can attach kernel-variant and
    /// capture-rejection reasons. Propagated to any already-built child
    /// (control-flow subgraph) executors so nested ops are traced too.
    pub(crate) fn set_trace_context(&mut self, trace: TraceContext) {
        for child in self.subgraph_execs.values_mut() {
            child.set_trace_context(trace.clone());
        }
        self.trace = trace;
    }

    /// Enable/disable dead-value buffer release, propagating to control-flow
    /// child executors so a Scan/Loop body frees its intermediates too -- the
    /// vision encoder's per-image Scan bodies are a large part of its live set.
    pub(crate) fn set_release_dead_values(&mut self, enabled: bool) {
        for child in self.subgraph_execs.values_mut() {
            child.set_release_dead_values(enabled);
        }
        self.release_dead_values_enabled = enabled;
    }

    /// Live weight bytes backing the graph, needed alongside [`Self::graph`] so
    /// the EPContext dump can encode initializers into the context model.
    pub(crate) fn weights(&self) -> &Arc<WeightStore> {
        &self.weights
    }

    /// Warmup: re-touch the shape-keyed cache for the compiled plan so the first
    /// real `run` sees only cache hits (§11.3). Only meaningful for fully-static
    /// graphs, whose plan shapes are known at build; symbolic graphs cannot be
    /// pre-compiled without a concrete shape and warm up on their first `run`.
    pub(crate) fn warmup(&mut self) -> Result<()> {
        if self.has_symbols {
            return Ok(());
        }
        let empty = HashMap::new();
        let resolved = self.resolve_all(&empty)?;
        self.compile_all(&resolved)
    }
}

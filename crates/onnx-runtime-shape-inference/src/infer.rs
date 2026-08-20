//! Whole-graph and single-node inference driving logic.

use std::collections::{HashMap, HashSet};

use onnx_runtime_ir::{
    Attribute, DataType, Dim, Graph, Node, SymbolConstraints, SymbolId, ValueId, WeightRef,
};

use crate::context::{
    MergePolicy, NodeIo, SymbolInterner, TensorType, TypeInfo, TypedShape, ValueType, merge_shapes,
    unify_value_type,
};
use crate::dim_expr::DimExpr;
use crate::error::ShapeInferError;
use crate::registry::InferenceRegistry;
use crate::report::InferenceReport;
use crate::shape_data::ShapeData;

type ScopeBindings = HashMap<String, Option<NodeIo>>;

struct ScopedInference {
    report: InferenceReport,
    parent_symbols: HashMap<SymbolId, SymbolId>,
}

/// Fresh symbolic dimensions minted by this crate live at or above this id, in
/// the "anonymous" range the loader also reserves (`u32_symbol` starts at
/// `0x8000_0000`). Graph-interned dim-params (`batch`, `seq_len`, …) stay in the
/// low range, so a fresh symbol here can never be confused with a named one nor
/// with a future [`Graph::create_symbol`](onnx_runtime_ir::Graph) allocation
/// (which advances the graph's private low-range counter).
pub(crate) const ANON_SYMBOL_FLOOR: u32 = 0x8000_0000;

impl InferenceRegistry {
    /// Infer shapes for every value in `graph`, in topological order.
    ///
    /// Seeds every explicitly known value type, runs each node's rule to fill or
    /// refine its outputs' types and shape-data, then writes the resolved shapes
    /// back into the graph (lowering symbolic dimension expressions to IR
    /// [`Dim`]s). Graph outputs are reconciled with their declared shapes under
    /// `policy`. Returns an [`InferenceReport`] of what resolved.
    ///
    /// `opset_imports` selects the effective operator versions; pass
    /// `graph.opset_imports.clone()` for the model's own imports.
    pub fn infer_graph(
        &self,
        graph: &mut Graph,
        opset_imports: &HashMap<String, u64>,
        policy: MergePolicy,
    ) -> Result<InferenceReport, ShapeInferError> {
        self.infer_graph_scoped(
            graph,
            opset_imports,
            policy,
            &HashMap::new(),
            &HashMap::new(),
        )
        .map(|result| result.report)
    }

    fn infer_graph_scoped(
        &self,
        graph: &mut Graph,
        opset_imports: &HashMap<String, u64>,
        policy: MergePolicy,
        outer_scope: &ScopeBindings,
        seed_containers: &HashMap<ValueId, ValueType>,
    ) -> Result<ScopedInference, ShapeInferError> {
        // The interner floor must clear every symbol already present in `graph`
        // *and* every symbol carried by a seeded container element shape (which
        // never lands on a body `Value`, so `seed_next_symbol` cannot see it) —
        // otherwise a freshly-minted body symbol could alias a parent one.
        let floor = seed_next_symbol(graph).max(next_container_symbol(seed_containers));
        let mut interner = SymbolInterner::new(floor);
        let (imported_scope, parent_symbols) = import_scope(graph, outer_scope, &mut interner);

        let order = graph
            .topological_order()
            .map_err(|_| ShapeInferError::CycleDetected)?;

        let mut types: HashMap<ValueId, TypeInfo> = HashMap::new();
        let mut shape_data: HashMap<ValueId, ShapeData> = HashMap::new();
        // Parallel container-type layer: only populated for values that are
        // actually `Sequence`/`Optional`/`Map`. Empty for pure-tensor graphs, so
        // the tensor-only path is byte-identical. Seeded from the owning
        // control-flow node's container operands (empty at the top level).
        let mut containers: HashMap<ValueId, ValueType> = seed_containers.clone();

        seed_sources(graph, &mut types, &mut shape_data);
        bind_captures(
            graph,
            &imported_scope,
            &mut types,
            &mut shape_data,
            &mut containers,
        );

        // Snapshot graph outputs' declared shapes for the final merge.
        let declared_outputs: HashMap<ValueId, Vec<Dim>> = graph
            .outputs
            .iter()
            .filter_map(|&vid| graph.try_value(vid).map(|v| (vid, v.shape.clone())))
            .collect();

        let mut child_scope = None;
        let mut scope_sources_added = false;
        let mut pending_scope_values = Vec::new();
        let mut remaining_subgraph_nodes = graph
            .subgraphs
            .keys()
            .map(|(owner, _)| *owner)
            .collect::<HashSet<_>>()
            .len();

        // Propagate in topological order.
        for nid in order {
            let node = graph.node(nid).clone();
            let subgraph_results = self.infer_child_subgraphs(
                graph,
                &node,
                opset_imports,
                policy,
                &imported_scope,
                &types,
                &shape_data,
                &containers,
                &mut child_scope,
                &mut scope_sources_added,
                &mut pending_scope_values,
                &mut remaining_subgraph_nodes,
                &mut interner,
            )?;

            if infer_control_flow(
                graph,
                &node,
                &subgraph_results,
                &mut types,
                &mut shape_data,
                &mut containers,
                &mut interner,
                remaining_subgraph_nodes,
                &mut pending_scope_values,
                effective_version(&node, opset_imports),
            )? {
                continue;
            }

            self.infer_node_outputs(
                &node,
                opset_imports,
                policy,
                &mut types,
                &mut shape_data,
                &mut containers,
                &mut interner,
            )?;
            if remaining_subgraph_nodes > 0 {
                pending_scope_values.extend(node.outputs.iter().copied());
            }
        }

        merge_declared_outputs(&mut types, &declared_outputs, policy)?;

        let resolved = write_back_types(graph, &types, &mut interner);

        // Register any freshly-minted symbols on the graph.
        for &sym in interner.fresh_symbols() {
            graph
                .symbol_constraints
                .entry(sym)
                .or_insert_with(|| SymbolConstraints::new(sym, None));
        }

        // Persist the authoritative symbol-lineage records for this graph. A
        // fresh (overwriting) assignment, not an append: each inference run is a
        // complete pass, so the last run's records are the ones that match the
        // shapes just written back. Consumers (e.g. the CUDA-graph
        // capture-eligibility classifier) read these instead of re-deriving a
        // partial copy of `broadcast_dim`'s/`lower`'s lineage per op.
        graph.symbol_unifications = interner.unifications().to_vec();
        // Derivation edges can be recorded on every cache hit for a hot derived
        // dim, so dedup before persisting (order-independent consumers).
        let mut derivations = interner.derivations().to_vec();
        derivations.sort_unstable_by_key(|&(SymbolId(d), SymbolId(s))| (d, s));
        derivations.dedup();
        graph.symbol_derivations = derivations;
        graph.symbol_opaque = interner.opaque().to_vec();
        graph.inference_symbol_floor = Some(interner.initial_floor());

        let unresolved: Vec<ValueId> = graph
            .values
            .keys()
            .filter(|vid| !types.contains_key(vid))
            .collect();

        Ok(ScopedInference {
            report: InferenceReport {
                total_values: graph.num_values(),
                fresh_symbols: interner.fresh_symbols().len(),
                resolved,
                unresolved,
                containers,
            },
            parent_symbols,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn infer_child_subgraphs(
        &self,
        graph: &mut Graph,
        node: &Node,
        opset_imports: &HashMap<String, u64>,
        policy: MergePolicy,
        imported_scope: &ScopeBindings,
        types: &HashMap<ValueId, TypeInfo>,
        shape_data: &HashMap<ValueId, ShapeData>,
        containers: &HashMap<ValueId, ValueType>,
        child_scope: &mut Option<ScopeBindings>,
        scope_sources_added: &mut bool,
        pending_scope_values: &mut Vec<ValueId>,
        remaining_subgraph_nodes: &mut usize,
        interner: &mut SymbolInterner,
    ) -> Result<HashMap<String, ScopedInference>, ShapeInferError> {
        let mut child_keys: Vec<_> = graph
            .subgraphs
            .keys()
            .filter(|(owner, _)| *owner == node.id)
            .cloned()
            .collect();
        child_keys.sort_by(|left, right| left.1.cmp(&right.1));
        let mut subgraph_results = HashMap::new();
        if child_keys.is_empty() {
            return Ok(subgraph_results);
        }

        let scope = child_scope.get_or_insert_with(|| imported_scope.clone());
        if !*scope_sources_added {
            let formal_inputs: HashSet<_> = graph.inputs.iter().copied().collect();
            let source_values: Vec<_> = graph
                .values
                .iter()
                .filter(|(vid, _)| {
                    formal_inputs.contains(vid) || graph.initializers.contains_key(vid)
                })
                .map(|(vid, _)| vid)
                .collect();
            extend_visible_scope(
                graph,
                types,
                shape_data,
                containers,
                scope,
                source_values,
                interner,
            );
            *scope_sources_added = true;
        }
        extend_visible_scope(
            graph,
            types,
            shape_data,
            containers,
            scope,
            pending_scope_values.drain(..),
            interner,
        );

        for key in child_keys {
            let subgraph =
                graph
                    .subgraphs
                    .get_mut(&key)
                    .ok_or_else(|| ShapeInferError::Invalid {
                        op: node.op_type.clone(),
                        detail: format!("subgraph attribute `{}` disappeared", key.1),
                    })?;
            seed_control_flow_body(
                node,
                &key.1,
                subgraph,
                types,
                containers,
                effective_version(node, opset_imports),
                interner,
            );
            // A control-flow body's formal inputs may be seeded with *container*
            // types (a Loop carrying a sequence), which the tensor seeding above
            // cannot express on a `Value`; build those directly into the child's
            // container map.
            let seed_containers = body_container_seeds(
                node,
                &key.1,
                subgraph,
                containers,
                effective_version(node, opset_imports),
            );
            let result =
                self.infer_graph_scoped(subgraph, opset_imports, policy, scope, &seed_containers)?;
            subgraph_results.insert(key.1, result);
        }
        *remaining_subgraph_nodes -= 1;

        Ok(subgraph_results)
    }

    #[allow(clippy::too_many_arguments)]
    fn infer_node_outputs(
        &self,
        node: &Node,
        opset_imports: &HashMap<String, u64>,
        policy: MergePolicy,
        types: &mut HashMap<ValueId, TypeInfo>,
        shape_data: &mut HashMap<ValueId, ShapeData>,
        containers: &mut HashMap<ValueId, ValueType>,
        interner: &mut SymbolInterner,
    ) -> Result<(), ShapeInferError> {
        let inputs = gather_inputs(node, types, shape_data, containers);
        let outputs = self.infer_node(node, opset_imports, inputs, policy, interner)?;
        for (slot, io) in node.outputs.iter().zip(outputs) {
            if let Some(type_info) = io.type_info {
                types.insert(*slot, type_info);
            }
            if let Some(data) = io.shape_data
                && data.within_bounds()
            {
                shape_data.insert(*slot, data);
            }
            if let Some(value_type) = io.value_type {
                containers.insert(*slot, value_type);
            }
        }
        Ok(())
    }
}

/// The opset version effective for `node`, mirroring the registry's resolution:
/// a usable node-local version wins, otherwise the graph import for the node's
/// domain (defaulting to `1`).
fn effective_version(node: &Node, opset_imports: &HashMap<String, u64>) -> u64 {
    if let Some(version) = node.local_opset() {
        return version;
    }
    if node.is_default_domain() {
        opset_imports.get("").copied().unwrap_or(1)
    } else {
        opset_imports.get(&node.domain).copied().unwrap_or(1)
    }
}

/// Dispatch the control-flow ops (`If`/`Loop`/`Scan`) that carry subgraph
/// bodies. Returns `true` when the node was handled here (so the caller skips
/// the ordinary per-op rule), `false` to fall through to the registry.
///
/// All three share the same machinery: [`infer_child_subgraphs`] has already
/// run each body under lexical outer-scope visibility (and, for `Loop`/`Scan`,
/// with the body's formal inputs seeded from this node's operands via
/// [`seed_control_flow_body`]). The per-op functions below only map the inferred
/// *body outputs* back onto this node's outputs, applying each op's axis
/// bookkeeping (branch reconciliation, the scan/trip-count axis).
#[allow(clippy::too_many_arguments)]
fn infer_control_flow(
    graph: &mut Graph,
    node: &Node,
    subgraph_results: &HashMap<String, ScopedInference>,
    types: &mut HashMap<ValueId, TypeInfo>,
    shape_data: &mut HashMap<ValueId, ShapeData>,
    containers: &mut HashMap<ValueId, ValueType>,
    interner: &mut SymbolInterner,
    remaining_subgraph_nodes: usize,
    pending_scope_values: &mut Vec<ValueId>,
    version: u64,
) -> Result<bool, ShapeInferError> {
    if !node.is_default_domain() {
        return Ok(false);
    }
    let outputs = match node.op_type.as_str() {
        "If" => infer_if_outputs(graph, node, subgraph_results, interner)?,
        "Loop" => infer_loop_outputs(graph, node, subgraph_results, interner)?,
        // `Scan`'s opset-8 form carried an extra `sequence_lens` input and a
        // different body signature; only the stable opset-9+ form is modelled.
        "Scan" if version >= 9 => {
            infer_scan_outputs(graph, node, subgraph_results, types, interner)?
        }
        "SequenceMap" => infer_sequence_map_outputs(graph, node, subgraph_results, interner)?,
        _ => return Ok(false),
    };

    if let Some(outputs) = outputs {
        apply_cf_outputs(graph, node, outputs, types, shape_data, containers);
    }
    if remaining_subgraph_nodes > 0 {
        pending_scope_values.extend(node.outputs.iter().copied());
    }

    Ok(true)
}

/// Write the per-slot [`CfOutput`]s onto the node's outputs, shared by all
/// control-flow ops.
fn apply_cf_outputs(
    graph: &mut Graph,
    node: &Node,
    outputs: Vec<CfOutput>,
    types: &mut HashMap<ValueId, TypeInfo>,
    shape_data: &mut HashMap<ValueId, ShapeData>,
    containers: &mut HashMap<ValueId, ValueType>,
) {
    for (slot, output) in node.outputs.iter().zip(outputs) {
        match output {
            CfOutput::Container(value_type) => {
                // A container CF output lives in the parallel container map, not
                // in `types` (the IR `Value` has no container representation) —
                // consistent with how a top-level `Sequence` producer is handled.
                types.remove(slot);
                shape_data.remove(slot);
                containers.insert(*slot, value_type);
            }
            CfOutput::Typed(type_info) => {
                types.insert(*slot, type_info);
            }
            CfOutput::UnknownRank(element_type) => {
                types.remove(slot);
                shape_data.remove(slot);
                let value = graph.value_mut(*slot);
                value.dtype = element_type;
                graph.mark_value_type_known(*slot);
                graph.mark_value_shape_unknown(*slot);
            }
            CfOutput::Unresolved => {}
        }
    }
}

fn merge_declared_outputs(
    types: &mut HashMap<ValueId, TypeInfo>,
    declared_outputs: &HashMap<ValueId, Vec<Dim>>,
    policy: MergePolicy,
) -> Result<(), ShapeInferError> {
    for (&vid, declared) in declared_outputs {
        if let Some(type_info) = types.get(&vid) {
            let merged = merge_shapes(vid, &type_info.shape, declared, policy)?;
            let dtype = type_info.dtype;
            types.insert(vid, TypeInfo::new(dtype, merged));
        }
    }
    Ok(())
}

fn write_back_types(
    graph: &mut Graph,
    types: &HashMap<ValueId, TypeInfo>,
    interner: &mut SymbolInterner,
) -> Vec<ValueId> {
    let mut resolved = Vec::new();
    for (&vid, type_info) in types {
        if graph.try_value(vid).is_none() {
            continue;
        }
        let dims: Vec<Dim> = type_info
            .shape
            .iter()
            .map(|dimension| interner.lower(dimension))
            .collect();
        let value = graph.value_mut(vid);
        value.shape = dims;
        value.dtype = type_info.dtype;
        resolved.push(vid);
    }
    resolved
}

fn is_standard_if(node: &Node) -> bool {
    node.op_type == "If" && node.is_default_domain()
}

/// One control-flow node output slot: a container [`ValueType`], a fully typed
/// tensor shape, a known dtype whose shape could not be reconciled (unknown
/// rank), or nothing resolved.
enum CfOutput {
    Container(ValueType),
    Typed(TypeInfo),
    UnknownRank(DataType),
    Unresolved,
}

fn infer_if_outputs(
    graph: &Graph,
    node: &Node,
    subgraph_results: &HashMap<String, ScopedInference>,
    interner: &mut SymbolInterner,
) -> Result<Option<Vec<CfOutput>>, ShapeInferError> {
    if !is_standard_if(node) {
        return Ok(None);
    }
    let then_key = (node.id, "then_branch".to_string());
    let else_key = (node.id, "else_branch".to_string());
    let Some(then_branch) = graph.subgraphs.get(&then_key) else {
        return Ok(None);
    };
    let Some(else_branch) = graph.subgraphs.get(&else_key) else {
        return Ok(None);
    };
    let Some(then_result) = subgraph_results.get("then_branch") else {
        return Ok(None);
    };
    let Some(else_result) = subgraph_results.get("else_branch") else {
        return Ok(None);
    };
    let then_resolved: HashSet<_> = then_result.report.resolved.iter().copied().collect();
    let else_resolved: HashSet<_> = else_result.report.resolved.iter().copied().collect();

    let paired_outputs = node
        .outputs
        .len()
        .min(then_branch.outputs.len())
        .min(else_branch.outputs.len());
    let mut outputs = Vec::with_capacity(node.outputs.len());
    for (&then_id, &else_id) in then_branch
        .outputs
        .iter()
        .zip(&else_branch.outputs)
        .take(paired_outputs)
    {
        // Container outputs (sequence/optional/map) live in the parallel
        // container map, not the IR `Value`, so they are handled before the
        // tensor read-back path.
        let then_container = then_result.report.containers.get(&then_id);
        let else_container = else_result.report.containers.get(&else_id);
        match (then_container, else_container) {
            (Some(then_vt), Some(else_vt)) => {
                let unified = unify_value_type(interner, "If", then_vt, else_vt)?;
                let mapped =
                    map_container_to_parent(&unified, &then_result.parent_symbols, interner);
                outputs.push(CfOutput::Container(mapped));
                continue;
            }
            (Some(_), None) | (None, Some(_)) => {
                return Err(ShapeInferError::Invalid {
                    op: "If".to_string(),
                    detail:
                        "one branch produces a container output while the other produces a tensor"
                            .to_string(),
                });
            }
            (None, None) => {}
        }

        if !branch_output_is_resolved(then_branch, then_id, &then_resolved)
            || !branch_output_is_resolved(else_branch, else_id, &else_resolved)
        {
            outputs.push(CfOutput::Unresolved);
            continue;
        }
        let then_value =
            then_branch
                .try_value(then_id)
                .ok_or_else(|| ShapeInferError::Invalid {
                    op: "If".to_string(),
                    detail: format!("then_branch output {then_id:?} is not live"),
                })?;
        let else_value =
            else_branch
                .try_value(else_id)
                .ok_or_else(|| ShapeInferError::Invalid {
                    op: "If".to_string(),
                    detail: format!("else_branch output {else_id:?} is not live"),
                })?;

        if then_value.dtype != else_value.dtype {
            return Err(ShapeInferError::Invalid {
                op: "If".to_string(),
                detail: format!(
                    "branch output element types differ: {:?} != {:?}",
                    then_value.dtype, else_value.dtype
                ),
            });
        }

        if then_value.shape.len() != else_value.shape.len() {
            outputs.push(CfOutput::UnknownRank(then_value.dtype));
            continue;
        }

        let shape = then_value
            .shape
            .iter()
            .zip(&else_value.shape)
            .map(|(&then_dim, &else_dim)| match (then_dim, else_dim) {
                (Dim::Static(then_size), Dim::Static(else_size)) if then_size == else_size => {
                    i64::try_from(then_size)
                        .map(DimExpr::constant)
                        .unwrap_or_else(|_| interner.fresh_dim())
                }
                (Dim::Symbolic(then_symbol), Dim::Symbolic(else_symbol))
                    if then_result.parent_symbols.get(&then_symbol)
                        == else_result.parent_symbols.get(&else_symbol)
                        && then_result.parent_symbols.contains_key(&then_symbol) =>
                {
                    DimExpr::symbol(then_result.parent_symbols[&then_symbol])
                }
                _ => interner.fresh_dim(),
            })
            .collect();
        outputs.push(CfOutput::Typed(TypeInfo::new(then_value.dtype, shape)));
    }
    outputs.resize_with(node.outputs.len(), || CfOutput::Unresolved);

    Ok(Some(outputs))
}

fn branch_output_is_resolved(branch: &Graph, output: ValueId, resolved: &HashSet<ValueId>) -> bool {
    resolved.contains(&output) && branch.try_value(output).is_some()
}

// ===========================================================================
// Loop / Scan: shared subgraph-body propagation.
//
// Unlike `If` — whose branches read the outer scope purely by name — a `Loop`
// or `Scan` body has *formal inputs* bound positionally to the owning node's
// operands (loop-carried dependencies, scan-input slices, the iteration
// counter). `seed_control_flow_body` writes those operand types onto the body's
// formal inputs *before* the body is inferred, so the ordinary node rules
// inside the body see concrete shapes; the `infer_*_outputs` functions then map
// the inferred body outputs back onto this node's outputs.
// ===========================================================================

/// Seed a control-flow body's formal inputs from the owning node's operands,
/// immediately before the body is inferred. A no-op for `If` (its branches take
/// no formal inputs) and for bodies whose operand types are not yet known.
fn seed_control_flow_body(
    node: &Node,
    attr_name: &str,
    body: &mut Graph,
    types: &HashMap<ValueId, TypeInfo>,
    containers: &HashMap<ValueId, ValueType>,
    version: u64,
    interner: &mut SymbolInterner,
) {
    if !node.is_default_domain() || attr_name != "body" {
        return;
    }
    match node.op_type.as_str() {
        "Loop" => seed_loop_body(node, body, types, interner),
        "Scan" if version >= 9 => seed_scan_body(node, body, types, interner),
        "SequenceMap" => seed_sequence_map_body(node, body, types, containers, interner),
        _ => {}
    }
}

/// Seed a `SequenceMap` body: body formal input `i` takes the *per-element* type
/// of `SequenceMap` operand `i`. A sequence operand contributes its element
/// tensor type; a plain tensor operand is passed whole to every iteration. The
/// container-element case (a sequence of sequences) is seeded separately via
/// [`body_container_seeds`], so this only writes tensor leaves onto body inputs.
fn seed_sequence_map_body(
    node: &Node,
    body: &mut Graph,
    types: &HashMap<ValueId, TypeInfo>,
    containers: &HashMap<ValueId, ValueType>,
    interner: &mut SymbolInterner,
) {
    for (i, &dst) in body.inputs.clone().iter().enumerate() {
        let Some(src) = node_input(node, i) else {
            continue;
        };
        // A sequence operand's element tensor; otherwise the operand's own tensor
        // type (a broadcast whole-tensor additional input).
        let element = containers
            .get(&src)
            .and_then(ValueType::as_sequence_element)
            .and_then(ValueType::as_tensor)
            .and_then(TensorType::to_type_info)
            .or_else(|| types.get(&src).cloned());
        if let Some(type_info) = element {
            let dims = lower_shape(&type_info.shape, interner);
            set_body_input(body, dst, type_info.dtype, dims);
        }
    }
}

/// Build container-typed seeds for a control-flow body's formal inputs from the
/// owning node's *container* operands, keyed by body-input [`ValueId`]. Covers
/// `Loop` carried dependencies, `Scan` container **state** variables, and
/// `SequenceMap` operands whose per-element type is itself a container (a
/// sequence of sequences). An `If` branch takes no formal inputs (it reads the
/// outer scope by name). Empty when the node has no container operands, which
/// keeps the tensor path byte-identical.
fn body_container_seeds(
    node: &Node,
    attr_name: &str,
    body: &Graph,
    containers: &HashMap<ValueId, ValueType>,
    version: u64,
) -> HashMap<ValueId, ValueType> {
    let mut seeds = HashMap::new();
    if !node.is_default_domain() || attr_name != "body" {
        return seeds;
    }
    match node.op_type.as_str() {
        // Body inputs are (iter_num, cond_in, v_1..v_N); carried operand `2 + i`
        // seeds carried body input `2 + i`.
        "Loop" => {
            let carried = body.inputs.len().saturating_sub(2);
            for i in 0..carried {
                seed_container_input(node, body, containers, 2 + i, 2 + i, &mut seeds);
            }
        }
        // The first `num_state` body inputs are loop-state variables, bound to
        // node operands `0..num_state`; a container state var seeds its input.
        "Scan" if version >= 9 => {
            if let Some(num_scan) = scan_num_scan_inputs(node, body.inputs.len()) {
                let num_state = body.inputs.len() - num_scan;
                for i in 0..num_state {
                    seed_container_input(node, body, containers, i, i, &mut seeds);
                }
            }
        }
        // Body input `i` takes the per-element type of operand `i`; a container
        // element (a sequence of sequences) seeds the body input as a container.
        "SequenceMap" => {
            for i in 0..body.inputs.len() {
                let Some(src) = node_input(node, i) else {
                    continue;
                };
                let Some(element) = containers
                    .get(&src)
                    .and_then(ValueType::as_sequence_element)
                else {
                    continue;
                };
                if !matches!(element, ValueType::Tensor(_))
                    && let Some(&dst) = body.inputs.get(i)
                {
                    seeds.insert(dst, element.clone());
                }
            }
        }
        _ => {}
    }
    seeds
}

/// Seed body input `body_idx` from node operand `operand_idx` when that operand
/// carries a container type (shared by `Loop` carried deps and `Scan` state).
fn seed_container_input(
    node: &Node,
    body: &Graph,
    containers: &HashMap<ValueId, ValueType>,
    operand_idx: usize,
    body_idx: usize,
    seeds: &mut HashMap<ValueId, ValueType>,
) {
    let Some(src) = node_input(node, operand_idx) else {
        return;
    };
    let Some(value_type) = containers.get(&src) else {
        return;
    };
    if let Some(&dst) = body.inputs.get(body_idx) {
        seeds.insert(dst, value_type.clone());
    }
}

/// The first symbol id strictly above every symbol appearing in a seeded
/// container's element shapes (`0` when there are none). Container element
/// shapes never land on a body `Value`, so [`seed_next_symbol`] cannot see them;
/// raising the child interner floor to this value stops a body-local fresh
/// symbol from aliasing a parent symbol carried in only via a container seed.
fn next_container_symbol(seed_containers: &HashMap<ValueId, ValueType>) -> u32 {
    let mut next = 0u32;
    for value_type in seed_containers.values() {
        for_each_container_symbol(value_type, &mut |SymbolId(id)| {
            next = next.max(id.saturating_add(1));
        });
    }
    next
}

/// Visit every symbol id in a container [`ValueType`]'s tensor-leaf shapes.
fn for_each_container_symbol(value_type: &ValueType, visit: &mut impl FnMut(SymbolId)) {
    match value_type {
        ValueType::Tensor(tensor) => {
            if let Some(shape) = &tensor.shape {
                for dim in shape {
                    for symbol in dim.symbol_ids() {
                        visit(symbol);
                    }
                }
            }
        }
        ValueType::Sequence(inner) | ValueType::Optional(inner) | ValueType::Map(_, inner) => {
            for_each_container_symbol(inner, visit)
        }
    }
}

/// Remap a container [`ValueType`]'s tensor-leaf shape symbols from a child body
/// scope into the parent scope, mirroring the per-dim rule the tensor
/// control-flow path uses: constants pass through, a parent-origin symbol
/// (present in `child_to_parent`) passes through, and any body-local symbol
/// degrades to a fresh parent symbol. Sound: a body-local symbol has no meaning
/// outside the body.
fn map_container_to_parent(
    value_type: &ValueType,
    child_to_parent: &HashMap<SymbolId, SymbolId>,
    interner: &mut SymbolInterner,
) -> ValueType {
    match value_type {
        ValueType::Tensor(tensor) => {
            let shape = tensor.shape.as_ref().map(|shape| {
                shape
                    .iter()
                    .map(|dim| map_container_dim(dim, child_to_parent, interner))
                    .collect()
            });
            ValueType::Tensor(TensorType {
                dtype: tensor.dtype,
                shape,
            })
        }
        ValueType::Sequence(inner) => {
            ValueType::sequence(map_container_to_parent(inner, child_to_parent, interner))
        }
        ValueType::Optional(inner) => ValueType::Optional(Box::new(map_container_to_parent(
            inner,
            child_to_parent,
            interner,
        ))),
        ValueType::Map(key, inner) => ValueType::Map(
            *key,
            Box::new(map_container_to_parent(inner, child_to_parent, interner)),
        ),
    }
}

/// Map a single container element dim into the parent scope (see
/// [`map_container_to_parent`]).
fn map_container_dim(
    dim: &DimExpr,
    child_to_parent: &HashMap<SymbolId, SymbolId>,
    interner: &mut SymbolInterner,
) -> DimExpr {
    if dim.as_const().is_some() {
        return dim.clone();
    }
    match dim.as_symbol() {
        Some(symbol) => match child_to_parent.get(&symbol) {
            Some(&parent) => DimExpr::symbol(parent),
            None => interner.fresh_dim(),
        },
        None => interner.fresh_dim(),
    }
}

/// The value id of the node's input at slot `i`, if that slot is connected.
fn node_input(node: &Node, i: usize) -> Option<ValueId> {
    node.inputs.get(i).copied().flatten()
}

/// Lower a symbolic shape to IR dims in the owning graph's symbol space.
fn lower_shape(shape: &[DimExpr], interner: &mut SymbolInterner) -> Vec<Dim> {
    shape.iter().map(|dim| interner.lower(dim)).collect()
}

/// Write `dtype`/`shape` onto a body formal input and mark it fully known.
fn set_body_input(body: &mut Graph, vid: ValueId, dtype: DataType, shape: Vec<Dim>) {
    if body.try_value(vid).is_none() {
        return;
    }
    let value = body.value_mut(vid);
    value.dtype = dtype;
    value.shape = shape;
    body.mark_value_type_known(vid);
    body.mark_value_shape_known(vid);
}

/// Seed a `Loop` body: formal inputs are `(iter_num, cond_in, v_1..v_N)`, bound
/// to the node's loop-carried operands `(M, cond, v_1..v_N)`.
fn seed_loop_body(
    node: &Node,
    body: &mut Graph,
    types: &HashMap<ValueId, TypeInfo>,
    interner: &mut SymbolInterner,
) {
    let body_inputs = body.inputs.clone();
    if let Some(&iter) = body_inputs.first() {
        set_body_input(body, iter, DataType::Int64, Vec::new());
    }
    if let Some(&cond) = body_inputs.get(1) {
        set_body_input(body, cond, DataType::Bool, Vec::new());
    }
    // Loop-carried dependencies: node operand `2 + i` seeds body input `2 + i`.
    let carried = body_inputs.len().saturating_sub(2);
    for i in 0..carried {
        let Some(src) = node_input(node, 2 + i) else {
            continue;
        };
        let Some(type_info) = types.get(&src) else {
            continue;
        };
        let dims = lower_shape(&type_info.shape, interner);
        set_body_input(body, body_inputs[2 + i], type_info.dtype, dims);
    }
}

/// Seed a `Scan` body (opset 9+): the first `N` formal inputs are the loop-state
/// variables (shape unchanged); the remaining `M = num_scan_inputs` are
/// per-iteration *slices* of the scan inputs with their scan axis stripped.
fn seed_scan_body(
    node: &Node,
    body: &mut Graph,
    types: &HashMap<ValueId, TypeInfo>,
    interner: &mut SymbolInterner,
) {
    let body_inputs = body.inputs.clone();
    let Some(num_scan) = scan_num_scan_inputs(node, body_inputs.len()) else {
        return;
    };
    let num_state = body_inputs.len() - num_scan;
    let axes = node.attr("scan_input_axes").and_then(Attribute::as_ints);

    for (i, &dst) in body_inputs.iter().enumerate().take(num_state) {
        let Some(src) = node_input(node, i) else {
            continue;
        };
        let Some(type_info) = types.get(&src) else {
            continue;
        };
        let dims = lower_shape(&type_info.shape, interner);
        set_body_input(body, dst, type_info.dtype, dims);
    }

    for j in 0..num_scan {
        let Some(src) = node_input(node, num_state + j) else {
            continue;
        };
        let Some(type_info) = types.get(&src) else {
            continue;
        };
        let raw_axis = axes.and_then(|axes| axes.get(j).copied()).unwrap_or(0);
        let Some(axis) = normalize_axis(raw_axis, type_info.rank()) else {
            continue;
        };
        let mut sliced = lower_shape(&type_info.shape, interner);
        sliced.remove(axis);
        set_body_input(body, body_inputs[num_state + j], type_info.dtype, sliced);
    }
}

/// The `num_scan_inputs` attribute clamped to a plausible range (`0..=total`).
fn scan_num_scan_inputs(node: &Node, total_inputs: usize) -> Option<usize> {
    let raw = node.attr("num_scan_inputs")?.as_int()?;
    let num = usize::try_from(raw).ok()?;
    (num <= total_inputs).then_some(num)
}

/// Normalize a possibly-negative axis against `rank`, returning `None` when it
/// is out of range.
fn normalize_axis(axis: i64, rank: usize) -> Option<usize> {
    let rank = i64::try_from(rank).ok()?;
    let axis = if axis < 0 { axis + rank } else { axis };
    (0..rank).contains(&axis).then_some(axis as usize)
}

/// Symbols reaching a body from the owning node, collected from the body's
/// (already-seeded) formal inputs. A body output that is one of these symbols
/// is passed straight through to the parent; any other symbol is body-local and
/// is remapped to a fresh parent symbol (its identity is meaningless outside).
fn body_parent_symbols(body: &Graph) -> HashSet<SymbolId> {
    let mut symbols = HashSet::new();
    for &vid in &body.inputs {
        if let Some(value) = body.try_value(vid) {
            for dim in &value.shape {
                if let Dim::Symbolic(symbol) = dim {
                    symbols.insert(*symbol);
                }
            }
        }
    }
    symbols
}

/// Map a body output's IR shape into the parent symbol space: static extents are
/// preserved, parent-origin symbols pass through, and everything else becomes a
/// fresh parent symbol.
fn map_body_shape(
    shape: &[Dim],
    parent_symbols: &HashSet<SymbolId>,
    interner: &mut SymbolInterner,
) -> TypedShape {
    shape
        .iter()
        .map(|dim| match *dim {
            Dim::Static(extent) => i64::try_from(extent)
                .map(DimExpr::constant)
                .unwrap_or_else(|_| interner.fresh_dim()),
            Dim::Symbolic(symbol) if parent_symbols.contains(&symbol) => DimExpr::symbol(symbol),
            Dim::Symbolic(_) => interner.fresh_dim(),
        })
        .collect()
}

/// Read a resolved body output as `(dtype, mapped-shape)`, or a dtype-only
/// [`CfOutput`] when the output's shape could not be resolved.
fn read_body_output(
    body: &Graph,
    output: ValueId,
    resolved: &HashSet<ValueId>,
    parent_symbols: &HashSet<SymbolId>,
    interner: &mut SymbolInterner,
) -> Result<(DataType, TypedShape), CfOutput> {
    if !branch_output_is_resolved(body, output, resolved) {
        return match body.try_value(output) {
            Some(value) if body.value_type_is_known(output) => {
                Err(CfOutput::UnknownRank(value.dtype))
            }
            _ => Err(CfOutput::Unresolved),
        };
    }
    let value = body.value(output);
    let shape = map_body_shape(&value.shape, parent_symbols, interner);
    Ok((value.dtype, shape))
}

/// Map a `Loop` body's outputs `(cond_out, v_1..v_N, scan_1..scan_K)` onto the
/// node's outputs `(v_1..v_N, scan_1..scan_K)`.
///
/// A loop-carried output takes its body carried-output shape directly; a scan
/// output gains a prepended trip-count axis.
///
/// The trip-count axis is **always symbolic**, even when the `M` operand is a
/// static constant: the loop may exit early on `cond`, so a static `M` is only
/// an unsound *upper bound*, not the true iteration count. Emitting it as a
/// concrete extent would (for a huge `M`) provoke eager buffer over-reservation
/// downstream; execution computes the real count.
fn infer_loop_outputs(
    graph: &Graph,
    node: &Node,
    subgraph_results: &HashMap<String, ScopedInference>,
    interner: &mut SymbolInterner,
) -> Result<Option<Vec<CfOutput>>, ShapeInferError> {
    let key = (node.id, "body".to_string());
    let Some(body) = graph.subgraphs.get(&key) else {
        return Ok(None);
    };
    let Some(result) = subgraph_results.get("body") else {
        return Ok(None);
    };
    let resolved: HashSet<_> = result.report.resolved.iter().copied().collect();
    let parent_symbols = body_parent_symbols(body);

    // Body inputs: iter_num, cond_in, then N carried. Body outputs: cond_out,
    // then N carried, then K scan outputs. All scan outputs share one iteration
    // count, so they share one trip-count symbol.
    let carried = body.inputs.len().saturating_sub(2);
    let trip_count = interner.fresh_dim();

    let mut outputs = Vec::with_capacity(node.outputs.len());
    for slot in 0..node.outputs.len() {
        // Body output 0 is `cond_out`; carried/scan outputs start at index 1.
        let body_index = slot + 1;
        let Some(&body_output) = body.outputs.get(body_index) else {
            outputs.push(CfOutput::Unresolved);
            continue;
        };
        // A loop-carried dependency (slot < carried) can be a container: its
        // body carried-output container type flows to the Loop output. Scan
        // outputs stack tensors, so they are never containers.
        if slot < carried
            && let Some(value_type) = result.report.containers.get(&body_output)
        {
            let mapped = map_container_to_parent(value_type, &result.parent_symbols, interner);
            outputs.push(CfOutput::Container(mapped));
            continue;
        }
        match read_body_output(body, body_output, &resolved, &parent_symbols, interner) {
            Ok((dtype, mut shape)) => {
                if slot >= carried {
                    // A per-iteration scan output stacks along a new leading axis.
                    shape.insert(0, trip_count.clone());
                }
                outputs.push(cf_typed(dtype, shape));
            }
            Err(fallback) => outputs.push(fallback),
        }
    }
    Ok(Some(outputs))
}

/// Build a typed control-flow output, degrading to a dtype-only (unknown-shape)
/// output when a *fully static* shape would overflow eager buffer sizing.
///
/// A control-flow op can stack a per-iteration extent onto a large operand dim;
/// if every extent is concrete and their byte size overflows, writing it as a
/// static shape would make eager buffer planning reject the graph at build time
/// instead of letting execution compute (and, where required, gracefully
/// reject) the true shape. A shape with any symbolic dim is never eagerly sized,
/// so it is passed through unchanged.
fn cf_typed(dtype: DataType, shape: TypedShape) -> CfOutput {
    let static_dims: Option<Vec<usize>> = shape
        .iter()
        .map(|dim| {
            dim.as_const()
                .and_then(|extent| usize::try_from(extent).ok())
        })
        .collect();
    if let Some(dims) = static_dims
        && onnx_runtime_ir::checked_expected_bytes(dtype, &dims).is_none()
    {
        return CfOutput::UnknownRank(dtype);
    }
    CfOutput::Typed(TypeInfo::new(dtype, shape))
}

/// Map a `Scan` body's outputs `(state_1..state_N, scan_1..scan_K)` onto the
/// node's outputs of the same arity.
///
/// A final-state output keeps its body shape; a scan output re-inserts the scan
/// axis (`scan_output_axes[k]`, default 0) sized to the sequence length taken
/// from the scan inputs.
fn infer_scan_outputs(
    graph: &Graph,
    node: &Node,
    subgraph_results: &HashMap<String, ScopedInference>,
    types: &HashMap<ValueId, TypeInfo>,
    interner: &mut SymbolInterner,
) -> Result<Option<Vec<CfOutput>>, ShapeInferError> {
    let key = (node.id, "body".to_string());
    let Some(body) = graph.subgraphs.get(&key) else {
        return Ok(None);
    };
    let Some(result) = subgraph_results.get("body") else {
        return Ok(None);
    };
    let Some(num_scan) = scan_num_scan_inputs(node, body.inputs.len()) else {
        return Ok(None);
    };
    let num_state = body.inputs.len() - num_scan;
    let resolved: HashSet<_> = result.report.resolved.iter().copied().collect();
    let parent_symbols = body_parent_symbols(body);
    let sequence_length = scan_sequence_length(node, types, num_state, interner);
    let output_axes = node.attr("scan_output_axes").and_then(Attribute::as_ints);

    let mut outputs = Vec::with_capacity(node.outputs.len());
    for slot in 0..node.outputs.len() {
        let Some(&body_output) = body.outputs.get(slot) else {
            outputs.push(CfOutput::Unresolved);
            continue;
        };
        // A loop-state slot (slot < num_state) can be a container; scan-output
        // slots stack tensors and are never containers.
        if slot < num_state
            && let Some(value_type) = result.report.containers.get(&body_output)
        {
            let mapped = map_container_to_parent(value_type, &result.parent_symbols, interner);
            outputs.push(CfOutput::Container(mapped));
            continue;
        }
        match read_body_output(body, body_output, &resolved, &parent_symbols, interner) {
            Ok((dtype, mut shape)) => {
                if slot >= num_state {
                    // Re-insert the scan axis at its (rank+1) position.
                    let raw_axis = output_axes
                        .and_then(|axes| axes.get(slot - num_state).copied())
                        .unwrap_or(0);
                    let axis = normalize_axis(raw_axis, shape.len() + 1).unwrap_or(0);
                    shape.insert(axis, sequence_length.clone());
                }
                outputs.push(cf_typed(dtype, shape));
            }
            Err(fallback) => outputs.push(fallback),
        }
    }
    Ok(Some(outputs))
}

/// The `Scan` sequence length: the extent of the first scan input along its scan
/// axis (`scan_input_axes[0]`, default 0). Falls back to a fresh symbol when the
/// scan input's type or scan-axis extent is unknown.
fn scan_sequence_length(
    node: &Node,
    types: &HashMap<ValueId, TypeInfo>,
    num_state: usize,
    interner: &mut SymbolInterner,
) -> DimExpr {
    let length = node_input(node, num_state)
        .and_then(|vid| types.get(&vid))
        .and_then(|type_info| {
            let raw_axis = node
                .attr("scan_input_axes")
                .and_then(Attribute::as_ints)
                .and_then(|axes| axes.first().copied())
                .unwrap_or(0);
            let axis = normalize_axis(raw_axis, type_info.rank())?;
            type_info.shape.get(axis).cloned()
        });
    length.unwrap_or_else(|| interner.fresh_dim())
}

/// Map a `SequenceMap` body's outputs onto the node's outputs: each body output
/// `j` (a per-element tensor, or a container for a seq-of-seq body) is wrapped as
/// `Sequence<body_output_j>` — the sequence the body produces one element of per
/// iteration. Body inputs were already seeded with each operand's per-element
/// type by [`seed_sequence_map_body`]/[`body_container_seeds`], so this only
/// wraps the inferred body outputs, reusing the same read-back + parent-remap
/// helpers as `Loop`/`Scan`.
fn infer_sequence_map_outputs(
    graph: &Graph,
    node: &Node,
    subgraph_results: &HashMap<String, ScopedInference>,
    interner: &mut SymbolInterner,
) -> Result<Option<Vec<CfOutput>>, ShapeInferError> {
    let key = (node.id, "body".to_string());
    let Some(body) = graph.subgraphs.get(&key) else {
        return Ok(None);
    };
    let Some(result) = subgraph_results.get("body") else {
        return Ok(None);
    };
    let resolved: HashSet<_> = result.report.resolved.iter().copied().collect();
    let parent_symbols = body_parent_symbols(body);

    let mut outputs = Vec::with_capacity(node.outputs.len());
    for slot in 0..node.outputs.len() {
        let Some(&body_output) = body.outputs.get(slot) else {
            outputs.push(CfOutput::Unresolved);
            continue;
        };
        // A seq-of-seq body output is itself a container element; otherwise the
        // element is the body output's tensor type. Either way the SequenceMap
        // output is that element wrapped one level in a `Sequence`.
        if let Some(value_type) = result.report.containers.get(&body_output) {
            let element = map_container_to_parent(value_type, &result.parent_symbols, interner);
            outputs.push(CfOutput::Container(ValueType::sequence(element)));
            continue;
        }
        match read_body_output(body, body_output, &resolved, &parent_symbols, interner) {
            Ok((dtype, shape)) => {
                let element = ValueType::Tensor(TensorType::new(dtype, shape));
                outputs.push(CfOutput::Container(ValueType::sequence(element)));
            }
            Err(CfOutput::UnknownRank(dtype)) => {
                let element = ValueType::Tensor(TensorType::dtype_only(dtype));
                outputs.push(CfOutput::Container(ValueType::sequence(element)));
            }
            Err(other) => outputs.push(other),
        }
    }
    Ok(Some(outputs))
}

/// Seed every explicitly known value type, including intermediate `value_info`.
///
/// A producer rule can overwrite this seed with a freshly inferred type. If the
/// rule cannot resolve its output, the declared metadata remains available to
/// downstream consumers instead of being silently discarded.
fn seed_sources(
    graph: &Graph,
    types: &mut HashMap<ValueId, TypeInfo>,
    shape_data: &mut HashMap<ValueId, ShapeData>,
) {
    for (vid, value) in graph.values.iter() {
        if !graph.value_type_is_known(vid) || !graph.value_shape_is_known(vid) {
            continue;
        }
        let shape: TypedShape = value.shape.iter().map(|&d| DimExpr::from(d)).collect();
        types.insert(vid, TypeInfo::new(value.dtype, shape));
    }
    // Initializers carry concrete data; capture their shape-data too.
    for (&vid, weight) in &graph.initializers {
        if let WeightRef::Inline(t) = weight
            && let Some(sd) = ShapeData::from_tensor(t.dtype, &t.dims, &t.data)
        {
            shape_data.insert(vid, sd);
        }
    }
}

fn bind_captures(
    graph: &Graph,
    scope: &ScopeBindings,
    types: &mut HashMap<ValueId, TypeInfo>,
    shape_data: &mut HashMap<ValueId, ShapeData>,
    containers: &mut HashMap<ValueId, ValueType>,
) {
    let formal_inputs: HashSet<_> = graph.inputs.iter().copied().collect();
    for (vid, value) in graph.values.iter() {
        if value.producer.is_some()
            || formal_inputs.contains(&vid)
            || graph.initializers.contains_key(&vid)
        {
            continue;
        }
        let Some(name) = value.name.as_deref() else {
            continue;
        };
        let Some(Some(binding)) = scope.get(name) else {
            continue;
        };
        if let Some(type_info) = &binding.type_info {
            types.insert(vid, type_info.clone());
        }
        if let Some(data) = &binding.shape_data {
            shape_data.insert(vid, data.clone());
        }
        // A captured container (e.g. a sequence referenced from an outer scope)
        // resolves to its element type inside this body.
        if let Some(value_type) = &binding.value_type {
            containers.insert(vid, value_type.clone());
        }
    }
}

fn import_scope(
    graph: &Graph,
    outer_scope: &ScopeBindings,
    interner: &mut SymbolInterner,
) -> (ScopeBindings, HashMap<SymbolId, SymbolId>) {
    let local_names = local_value_names(graph);
    let mut parent_to_child = HashMap::new();
    let mut child_to_parent = HashMap::new();
    let mut names: Vec<_> = outer_scope
        .keys()
        .filter(|name| !local_names.contains(name.as_str()))
        .collect();
    names.sort_unstable();
    let imported = names
        .into_iter()
        .map(|name| {
            let binding = outer_scope[name]
                .as_ref()
                .map(|io| remap_node_io(io, interner, &mut parent_to_child, &mut child_to_parent));
            (name.clone(), binding)
        })
        .collect();
    (imported, child_to_parent)
}

fn local_value_names(graph: &Graph) -> HashSet<&str> {
    let formal_inputs: HashSet<_> = graph.inputs.iter().copied().collect();
    graph
        .values
        .iter()
        .filter(|(vid, value)| {
            value.producer.is_some()
                || formal_inputs.contains(vid)
                || graph.initializers.contains_key(vid)
        })
        .filter_map(|(_, value)| value.name.as_deref())
        .collect()
}

fn remap_node_io(
    io: &NodeIo,
    interner: &mut SymbolInterner,
    parent_to_child: &mut HashMap<SymbolId, SymbolId>,
    child_to_parent: &mut HashMap<SymbolId, SymbolId>,
) -> NodeIo {
    NodeIo {
        type_info: io.type_info.as_ref().map(|type_info| {
            TypeInfo::new(
                type_info.dtype,
                type_info
                    .shape
                    .iter()
                    .map(|dim| remap_dim_expr(dim, interner, parent_to_child, child_to_parent))
                    .collect(),
            )
        }),
        shape_data: io.shape_data.as_ref().map(|data| {
            let mut data = data.clone();
            data.elems = data
                .elems
                .iter()
                .map(|dim| remap_dim_expr(dim, interner, parent_to_child, child_to_parent))
                .collect();
            data
        }),
        // A captured container edge (a sequence referenced by name from an outer
        // scope) threads its `ValueType` through the same parent->child symbol
        // remap as the tensor path, so a body sees the element type.
        value_type: io
            .value_type
            .as_ref()
            .map(|vt| remap_container_type(vt, interner, parent_to_child, child_to_parent)),
    }
}

/// Parent->child remap of a captured container [`ValueType`], recursing into the
/// element type and remapping every tensor-leaf element-shape symbol through the
/// same [`remap_dim_expr`] the tensor capture path uses. The mirror image of
/// [`map_container_to_parent`] (child->parent), used when a subgraph captures an
/// outer-scope container by name.
fn remap_container_type(
    value_type: &ValueType,
    interner: &mut SymbolInterner,
    parent_to_child: &mut HashMap<SymbolId, SymbolId>,
    child_to_parent: &mut HashMap<SymbolId, SymbolId>,
) -> ValueType {
    match value_type {
        ValueType::Tensor(tensor) => ValueType::Tensor(TensorType {
            dtype: tensor.dtype,
            shape: tensor.shape.as_ref().map(|shape| {
                shape
                    .iter()
                    .map(|dim| remap_dim_expr(dim, interner, parent_to_child, child_to_parent))
                    .collect()
            }),
        }),
        ValueType::Sequence(inner) => ValueType::sequence(remap_container_type(
            inner,
            interner,
            parent_to_child,
            child_to_parent,
        )),
        ValueType::Optional(inner) => ValueType::Optional(Box::new(remap_container_type(
            inner,
            interner,
            parent_to_child,
            child_to_parent,
        ))),
        ValueType::Map(key, inner) => ValueType::Map(
            *key,
            Box::new(remap_container_type(
                inner,
                interner,
                parent_to_child,
                child_to_parent,
            )),
        ),
    }
}

fn remap_dim_expr(
    dim: &DimExpr,
    interner: &mut SymbolInterner,
    parent_to_child: &mut HashMap<SymbolId, SymbolId>,
    child_to_parent: &mut HashMap<SymbolId, SymbolId>,
) -> DimExpr {
    if let Some(value) = dim.as_const() {
        return DimExpr::constant(value);
    }
    let Some(parent) = dim.as_symbol() else {
        return interner.fresh_dim();
    };
    let child = *parent_to_child.entry(parent).or_insert_with(|| {
        let child = interner.fresh_symbol();
        child_to_parent.insert(child, parent);
        child
    });
    DimExpr::symbol(child)
}

fn extend_visible_scope(
    graph: &Graph,
    types: &HashMap<ValueId, TypeInfo>,
    shape_data: &HashMap<ValueId, ShapeData>,
    containers: &HashMap<ValueId, ValueType>,
    scope: &mut ScopeBindings,
    values: impl IntoIterator<Item = ValueId>,
    interner: &mut SymbolInterner,
) {
    for vid in values {
        let Some(value) = graph.try_value(vid) else {
            continue;
        };
        let Some(name) = value.name.as_ref() else {
            continue;
        };
        let type_info = types.get(&vid).map(|type_info| {
            TypeInfo::new(
                type_info.dtype,
                type_info
                    .shape
                    .iter()
                    .map(|dim| DimExpr::from(interner.lower(dim)))
                    .collect(),
            )
        });
        let shape_data = shape_data.get(&vid).map(|data| {
            let mut data = data.clone();
            data.elems = data
                .elems
                .iter()
                .map(|dim| DimExpr::from(interner.lower(dim)))
                .collect();
            data
        });
        // Publish an outer-produced container (e.g. a `SequenceConstruct` result
        // later referenced inside a sibling subgraph) so capture threads it.
        let value_type = containers.get(&vid).cloned();
        let binding = if type_info.is_some() || shape_data.is_some() || value_type.is_some() {
            Some(NodeIo {
                type_info,
                shape_data,
                value_type,
            })
        } else {
            None
        };
        scope.insert(name.clone(), binding);
    }
}

/// Assemble the per-input [`NodeIo`]s for a node, aligned with `node.inputs`.
fn gather_inputs(
    node: &Node,
    types: &HashMap<ValueId, TypeInfo>,
    shape_data: &HashMap<ValueId, ShapeData>,
    containers: &HashMap<ValueId, ValueType>,
) -> Vec<NodeIo> {
    node.inputs
        .iter()
        .map(|slot| match slot {
            Some(vid) => NodeIo {
                type_info: types.get(vid).cloned(),
                shape_data: shape_data.get(vid).cloned(),
                value_type: containers.get(vid).cloned(),
            },
            None => NodeIo::default(),
        })
        .collect()
}

/// The first fresh-symbol id to allocate: strictly above every symbol id already
/// present in the graph, and at least [`ANON_SYMBOL_FLOOR`].
fn seed_next_symbol(graph: &Graph) -> u32 {
    let mut max = ANON_SYMBOL_FLOOR.saturating_sub(1);
    for &SymbolId(id) in graph.symbol_constraints.keys() {
        max = max.max(id);
    }
    for value in graph.values.values() {
        for dim in &value.shape {
            if let Dim::Symbolic(SymbolId(id)) = dim {
                max = max.max(*id);
            }
        }
    }
    max.saturating_add(1)
}

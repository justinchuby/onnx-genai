//! Whole-graph and single-node inference driving logic.

use std::collections::{HashMap, HashSet};

use onnx_runtime_ir::{Dim, Graph, Node, SymbolConstraints, SymbolId, ValueId, WeightRef};

use crate::context::{MergePolicy, NodeIo, SymbolInterner, TypeInfo, TypedShape, merge_shapes};
use crate::dim_expr::DimExpr;
use crate::error::ShapeInferError;
use crate::registry::InferenceRegistry;
use crate::report::InferenceReport;
use crate::shape_data::ShapeData;

/// Fresh symbolic dimensions minted by this crate live at or above this id, in
/// the "anonymous" range the loader also reserves (`u32_symbol` starts at
/// `0x8000_0000`). Graph-interned dim-params (`batch`, `seq_len`, …) stay in the
/// low range, so a fresh symbol here can never be confused with a named one nor
/// with a future [`Graph::create_symbol`](onnx_runtime_ir::Graph) allocation
/// (which advances the graph's private low-range counter).
const ANON_SYMBOL_FLOOR: u32 = 0x8000_0000;

impl InferenceRegistry {
    /// Infer shapes for every value in `graph`, in topological order.
    ///
    /// Seeds the known types (graph inputs and initializers), runs each node's
    /// rule to fill its outputs' types and shape-data, then writes the resolved
    /// shapes back into the graph (lowering symbolic dimension expressions to IR
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
        let mut subgraph_resolved = HashMap::new();
        for (key, subgraph) in &mut graph.subgraphs {
            let report = self.infer_graph(subgraph, opset_imports, policy)?;
            subgraph_resolved.insert(
                key.clone(),
                report.resolved.into_iter().collect::<HashSet<_>>(),
            );
        }

        let order = graph
            .topological_order()
            .map_err(|_| ShapeInferError::CycleDetected)?;

        let mut interner = SymbolInterner::new(seed_next_symbol(graph));
        let mut types: HashMap<ValueId, TypeInfo> = HashMap::new();
        let mut shape_data: HashMap<ValueId, ShapeData> = HashMap::new();

        seed_sources(graph, &mut types, &mut shape_data);

        // Snapshot graph outputs' declared shapes for the final merge.
        let declared_out: HashMap<ValueId, Vec<Dim>> = graph
            .outputs
            .iter()
            .filter_map(|&vid| graph.try_value(vid).map(|v| (vid, v.shape.clone())))
            .collect();

        // Propagate in topological order.
        for nid in order {
            let node = graph.node(nid).clone();
            let inputs = gather_inputs(&node, &types, &shape_data);
            let outputs = if is_standard_if(&node) {
                infer_if_outputs(graph, &node, &subgraph_resolved, &mut interner)?
                    .unwrap_or_else(|| vec![NodeIo::default(); node.outputs.len()])
            } else {
                self.infer_node(&node, opset_imports, inputs, policy, &mut interner)?
            };
            for (slot, io) in node.outputs.iter().zip(outputs) {
                if let Some(ti) = io.type_info {
                    types.insert(*slot, ti);
                }
                if let Some(sd) = io.shape_data
                    && sd.within_bounds()
                {
                    shape_data.insert(*slot, sd);
                }
            }
        }

        // Reconcile graph outputs with their declared shapes.
        for (&vid, declared) in &declared_out {
            if let Some(ti) = types.get(&vid) {
                let merged = merge_shapes(vid, &ti.shape, declared, policy)?;
                let dtype = ti.dtype;
                types.insert(vid, TypeInfo::new(dtype, merged));
            }
        }

        // Write resolved types back into the graph (lowering DimExprs to Dims).
        let mut resolved = Vec::new();
        for (&vid, ti) in &types {
            if graph.try_value(vid).is_none() {
                continue;
            }
            let dims: Vec<Dim> = ti.shape.iter().map(|d| interner.lower(d)).collect();
            let value = graph.value_mut(vid);
            value.shape = dims;
            value.dtype = ti.dtype;
            resolved.push(vid);
        }

        // Register any freshly-minted symbols on the graph.
        for &sym in interner.fresh_symbols() {
            graph
                .symbol_constraints
                .entry(sym)
                .or_insert_with(|| SymbolConstraints::new(sym, None));
        }

        let unresolved: Vec<ValueId> = graph
            .values
            .keys()
            .filter(|vid| !types.contains_key(vid))
            .collect();

        Ok(InferenceReport {
            total_values: graph.num_values(),
            fresh_symbols: interner.fresh_symbols().len(),
            resolved,
            unresolved,
        })
    }
}

fn is_standard_if(node: &Node) -> bool {
    node.op_type == "If" && (node.domain.is_empty() || node.domain == "ai.onnx")
}

fn infer_if_outputs(
    graph: &Graph,
    node: &Node,
    subgraph_resolved: &HashMap<(onnx_runtime_ir::NodeId, String), HashSet<ValueId>>,
    interner: &mut SymbolInterner,
) -> Result<Option<Vec<NodeIo>>, ShapeInferError> {
    let then_key = (node.id, "then_branch".to_string());
    let else_key = (node.id, "else_branch".to_string());
    let Some(then_branch) = graph.subgraphs.get(&then_key) else {
        return Ok(None);
    };
    let Some(else_branch) = graph.subgraphs.get(&else_key) else {
        return Ok(None);
    };
    let Some(then_resolved) = subgraph_resolved.get(&then_key) else {
        return Ok(None);
    };
    let Some(else_resolved) = subgraph_resolved.get(&else_key) else {
        return Ok(None);
    };

    if then_branch.outputs.len() != else_branch.outputs.len() {
        return Err(ShapeInferError::Invalid {
            op: "If".to_string(),
            detail: format!(
                "then_branch and else_branch produce different numbers of outputs: {} != {}",
                then_branch.outputs.len(),
                else_branch.outputs.len()
            ),
        });
    }
    if then_branch.outputs.len() != node.outputs.len() {
        return Err(ShapeInferError::Invalid {
            op: "If".to_string(),
            detail: format!(
                "node has {} outputs but its branches produce {}",
                node.outputs.len(),
                then_branch.outputs.len()
            ),
        });
    }

    let mut outputs = Vec::with_capacity(node.outputs.len());
    for (&then_id, &else_id) in then_branch.outputs.iter().zip(&else_branch.outputs) {
        if !branch_output_is_resolved(then_branch, then_id, then_resolved)
            || !branch_output_is_resolved(else_branch, else_id, else_resolved)
        {
            outputs.push(NodeIo::default());
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
            outputs.push(NodeIo::default());
            continue;
        }

        let shape = then_value
            .shape
            .iter()
            .zip(&else_value.shape)
            .map(|(&then_dim, &else_dim)| {
                if then_dim == else_dim {
                    DimExpr::from(then_dim)
                } else {
                    interner.fresh_dim()
                }
            })
            .collect();
        outputs.push(NodeIo::typed(TypeInfo::new(then_value.dtype, shape)));
    }

    Ok(Some(outputs))
}

fn branch_output_is_resolved(branch: &Graph, output: ValueId, resolved: &HashSet<ValueId>) -> bool {
    resolved.contains(&output)
        && branch.try_value(output).is_some_and(|value| {
            value.producer.is_some()
                || (branch.value_type_is_known(output) && branch.value_shape_is_known(output))
        })
}

/// Seed the type (and shape-data) of every source value — graph inputs,
/// initializers, and any other producer-less value.
fn seed_sources(
    graph: &Graph,
    types: &mut HashMap<ValueId, TypeInfo>,
    shape_data: &mut HashMap<ValueId, ShapeData>,
) {
    for (vid, value) in graph.values.iter() {
        if value.producer.is_some() {
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

/// Assemble the per-input [`NodeIo`]s for a node, aligned with `node.inputs`.
fn gather_inputs(
    node: &Node,
    types: &HashMap<ValueId, TypeInfo>,
    shape_data: &HashMap<ValueId, ShapeData>,
) -> Vec<NodeIo> {
    node.inputs
        .iter()
        .map(|slot| match slot {
            Some(vid) => NodeIo {
                type_info: types.get(vid).cloned(),
                shape_data: shape_data.get(vid).cloned(),
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

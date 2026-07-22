//! Liveness analysis: the live interval of every activation buffer in
//! execution (topological) order.
//!
//! The unit of allocation is a *buffer owner* — a value that needs a real
//! activation slot. Graph inputs (supplied by the caller) and initializers /
//! streamed weights (caller-owned, memory-mapped) are **excluded** from the
//! activation arena. Zero-copy views own no buffer either; their liveness is
//! folded into the root owner they alias (see [`crate::ViewMap`]).

use std::collections::HashMap;

use onnx_runtime_ir::{Graph, NodeId, ValueId};

use crate::error::PlanError;
use crate::options::PlanOptions;
use crate::view_map::ViewMap;

/// The half-open-agnostic live interval of a buffer owner, in units of
/// topological node index.
///
/// * `def` — the index of the node that produces the value (or `0` for an
///   included graph input, which is live from the start of execution).
/// * `use_end` — the index of the value's last use. Two owners whose intervals
///   touch even at a single index (`a.def <= b.use_end && b.def <= a.use_end`)
///   are considered overlapping and must not share a slot, because at that
///   node the producer's output and the still-needed input coexist.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Interval {
    pub def: usize,
    pub use_end: usize,
}

impl Interval {
    /// Whether two intervals overlap (share at least one node index).
    pub fn overlaps(&self, other: &Interval) -> bool {
        self.def <= other.use_end && other.def <= self.use_end
    }
}

/// Result of liveness analysis over a graph.
#[derive(Clone, Debug)]
pub struct Liveness {
    /// Topological position of each live node.
    pub order_index: HashMap<NodeId, usize>,
    /// Node ids in topological order.
    pub order: Vec<NodeId>,
    /// Index of the last node (0 for an empty/single-node schedule).
    pub last_index: usize,
    /// Live interval of each buffer *owner* (never a view, input, or weight).
    pub intervals: HashMap<ValueId, Interval>,
    /// Owners in deterministic allocation order: ascending `def`, then id.
    pub owners: Vec<ValueId>,
}

/// Whether `value` needs its own activation slot.
fn is_buffer_owner(
    graph: &Graph,
    view_map: &ViewMap,
    options: &PlanOptions,
    value: ValueId,
) -> bool {
    if graph.initializers.contains_key(&value) {
        return false; // streamed / caller-owned weight
    }
    if view_map.is_view(value) {
        return false; // zero-copy alias, no buffer of its own
    }
    let Some(val) = graph.try_value(value) else {
        return false;
    };
    if val.producer.is_some() {
        return true; // an activation produced by a node
    }
    // No producer: only a graph input can qualify, and only if opted in.
    options.include_graph_inputs && graph.inputs.contains(&value)
}

/// Compute the live interval of every activation buffer owner.
///
/// Views are folded to their root owner: a view's consumers (and graph-output
/// status) extend the root's `use_end`, guaranteeing the source outlives every
/// alias. Returns [`PlanError::Cycle`] if the graph is not schedulable.
pub fn compute_liveness(
    graph: &Graph,
    view_map: &ViewMap,
    options: &PlanOptions,
) -> Result<Liveness, PlanError> {
    let order = graph.topological_order().map_err(|_| PlanError::Cycle)?;
    let order_index: HashMap<NodeId, usize> =
        order.iter().enumerate().map(|(i, &n)| (n, i)).collect();
    let last_index = order.len().saturating_sub(1);

    let outputs: std::collections::HashSet<ValueId> = graph.outputs.iter().copied().collect();

    // 1. Seed intervals for every buffer owner at its definition point.
    let mut intervals: HashMap<ValueId, Interval> = HashMap::new();
    for vid in graph.values.keys() {
        if !is_buffer_owner(graph, view_map, options, vid) {
            continue;
        }
        let def = graph
            .value(vid)
            .producer
            .and_then(|p| order_index.get(&p).copied())
            .unwrap_or(0); // included graph input: live from the start
        intervals.insert(vid, Interval { def, use_end: def });
    }

    // 2. Extend each root owner's interval by every activation value (the owner
    //    itself or any view folding to it): its consumers and output status.
    for vid in graph.values.keys() {
        // Skip values that are neither owners nor views (e.g. excluded graph
        // inputs and initializers); their root is not in `intervals`.
        let root = view_map.root(vid);
        let Some(interval) = intervals.get_mut(&root) else {
            continue;
        };
        for consumer in graph.consumers(vid) {
            if let Some(&idx) = order_index.get(&consumer) {
                interval.use_end = interval.use_end.max(idx);
            }
        }
        // A graph output (or a view that is a graph output) pins its root to the
        // end of execution: it must never be overwritten.
        if outputs.contains(&vid) {
            interval.use_end = interval.use_end.max(last_index);
        }
    }

    // 3. Deterministic owner ordering: ascending def, then value id.
    let mut owners: Vec<ValueId> = intervals.keys().copied().collect();
    owners.sort_by_key(|v| (intervals[v].def, v.0));

    Ok(Liveness {
        order_index,
        order,
        last_index,
        intervals,
        owners,
    })
}

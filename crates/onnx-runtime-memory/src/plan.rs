//! The activation allocation plan and the greedy, slot-reusing planner.

use std::collections::HashMap;

use onnx_runtime_ir::{Graph, ValueId};

use crate::error::PlanError;
use crate::liveness::compute_liveness;
use crate::options::PlanOptions;
use crate::oracle::static_size_oracle;
use crate::view_map::ViewMap;

/// Identifier of a reusable activation slot in an [`ActivationPlan`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct SlotId(pub u32);

/// A single reusable activation slot: an arena region the executor allocates
/// once and hands, in turn, to every value assigned to it (their lifetimes are
/// guaranteed disjoint).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotInfo {
    pub id: SlotId,
    /// Bytes the executor must allocate for this slot: the max size of any value
    /// assigned to it.
    pub capacity_bytes: usize,
}

/// A computed activation memory plan: which slot backs each value, and how big
/// the whole activation arena must be.
#[derive(Clone, Debug)]
pub struct ActivationPlan {
    /// Buffer owner value → the slot that backs it. Views are absent (they own
    /// no slot; they alias their source's slot).
    pub assignments: HashMap<ValueId, SlotId>,
    /// Every slot and its capacity, in ascending id order.
    pub slots: Vec<SlotInfo>,
    /// Total bytes the executor must allocate = sum of slot capacities. This is
    /// the *concurrent peak* activation footprint after buffer sharing.
    pub peak_bytes: usize,
    /// Number of distinct slots.
    pub num_slots: usize,
    /// Bytes the naive "one buffer per value forever" strategy would use = sum
    /// of every owner's size. Exposed to quantify the win.
    pub naive_bytes: usize,
    /// Fraction of activation memory saved vs. naive: `1 - peak/naive` (in
    /// `[0, 1]`; `0.0` when `naive_bytes == 0`).
    pub savings_ratio: f64,
}

/// The outcome of a planning attempt.
#[derive(Clone, Debug)]
pub enum PlanStatus {
    /// Every buffer owner had a known size; a full plan was produced.
    Complete(ActivationPlan),
    /// At least one owner's size is unknown (symbolic shape) at planning time.
    /// The executor should re-plan once shapes resolve for the current run.
    Deferred {
        /// The owners whose sizes the oracle could not resolve.
        unknown_sizes: Vec<ValueId>,
    },
}

impl PlanStatus {
    /// The plan, if this is [`PlanStatus::Complete`].
    pub fn as_complete(&self) -> Option<&ActivationPlan> {
        match self {
            PlanStatus::Complete(p) => Some(p),
            PlanStatus::Deferred { .. } => None,
        }
    }

    /// Whether planning was deferred (unknown sizes).
    pub fn is_deferred(&self) -> bool {
        matches!(self, PlanStatus::Deferred { .. })
    }

    /// Unwrap the complete plan, panicking if deferred (test convenience).
    pub fn unwrap_complete(self) -> ActivationPlan {
        match self {
            PlanStatus::Complete(p) => p,
            PlanStatus::Deferred { unknown_sizes } => {
                panic!("plan was deferred; unknown sizes for {unknown_sizes:?}")
            }
        }
    }
}

/// Compute a liveness-based activation plan with slot reuse.
///
/// # Algorithm
///
/// 1. **Liveness** — order nodes topologically; each buffer owner's interval is
///    `[def, use_end]`, with view consumers and graph-output status folded into
///    the root owner (see [`compute_liveness`]).
/// 2. **Size** — query the `size_oracle` for every owner. If any is unknown
///    (symbolic shape), return [`PlanStatus::Deferred`] instead of guessing.
/// 3. **Greedy allocation** — walk nodes in topological order. At each node,
///    allocate slots for the node's owner outputs (best-fit reuse of a retired
///    slot with `capacity >= size`, else open a new slot), then retire the
///    slots of owners whose `use_end` is this node. Retiring *after* allocation
///    guarantees a node's own inputs are never clobbered by its outputs, and a
///    graph output (whose `use_end` is the last node) is never recycled.
///
/// The output is deterministic for identical input: node order, output order,
/// slot-id assignment, and best-fit tie-breaking (smallest capacity, then
/// smallest id) are all stable.
pub fn plan_activations<F>(
    graph: &Graph,
    view_map: &ViewMap,
    size_oracle: F,
    options: &PlanOptions,
) -> Result<PlanStatus, PlanError>
where
    F: Fn(ValueId) -> Option<usize>,
{
    let live = compute_liveness(graph, view_map, options)?;

    // Resolve sizes; defer if any owner size is unknown.
    let mut sizes: HashMap<ValueId, usize> = HashMap::new();
    let mut unknown: Vec<ValueId> = Vec::new();
    for &owner in &live.owners {
        match size_oracle(owner) {
            Some(bytes) => {
                sizes.insert(owner, bytes);
            }
            None => unknown.push(owner),
        }
    }
    if !unknown.is_empty() {
        unknown.sort_by_key(|v| v.0);
        return Ok(PlanStatus::Deferred {
            unknown_sizes: unknown,
        });
    }

    let naive_bytes: usize = live.owners.iter().map(|o| sizes[o]).sum();

    // Which owners retire at each node index (their slot returns to the free
    // list after that node's outputs are allocated).
    let mut retire_at: HashMap<usize, Vec<ValueId>> = HashMap::new();
    for (&owner, interval) in &live.intervals {
        retire_at.entry(interval.use_end).or_default().push(owner);
    }

    let mut slots: Vec<SlotInfo> = Vec::new();
    let mut free: Vec<SlotId> = Vec::new();
    let mut assignments: HashMap<ValueId, SlotId> = HashMap::new();

    // Best-fit allocation over the free list: smallest capacity that fits,
    // tie-broken by lowest slot id for determinism. Opens a new slot if nothing
    // free fits (a strict `>=` policy — a reused slot is never grown, keeping
    // each slot's capacity fixed at its first occupant's size unless a larger
    // owner is later assigned to a fresh slot).
    let allocate = |need: usize, slots: &mut Vec<SlotInfo>, free: &mut Vec<SlotId>| -> SlotId {
        // Track (free-list index, capacity, slot id) of the best fit so far.
        let mut best: Option<(usize, usize, u32)> = None;
        for (i, &sid) in free.iter().enumerate() {
            let cap = slots[sid.0 as usize].capacity_bytes;
            if cap < need {
                continue;
            }
            let better = match best {
                None => true,
                Some((_, best_cap, best_id)) => {
                    cap < best_cap || (cap == best_cap && sid.0 < best_id)
                }
            };
            if better {
                best = Some((i, cap, sid.0));
            }
        }
        if let Some((idx, _, _)) = best {
            free.remove(idx)
        } else {
            let sid = SlotId(slots.len() as u32);
            slots.push(SlotInfo {
                id: sid,
                capacity_bytes: need,
            });
            sid
        }
    };

    // Pre-allocate included graph inputs (owners with no producing node) before
    // the node walk; they are "live from the start".
    if options.include_graph_inputs {
        let mut input_owners: Vec<ValueId> = live
            .owners
            .iter()
            .copied()
            .filter(|v| graph.value(*v).producer.is_none())
            .collect();
        input_owners.sort_by_key(|v| v.0);
        for owner in input_owners {
            let sid = allocate(sizes[&owner], &mut slots, &mut free);
            assignments.insert(owner, sid);
        }
    }

    // Walk nodes in topological order.
    for (i, &node_id) in live.order.iter().enumerate() {
        for &out in &graph.node(node_id).outputs {
            if !live.intervals.contains_key(&out) {
                continue; // view / weight / excluded output — no slot
            }
            let sid = allocate(sizes[&out], &mut slots, &mut free);
            assignments.insert(out, sid);
        }
        // Retire owners whose last use is this node (after allocating outputs).
        if let Some(retiring) = retire_at.get(&i) {
            for owner in retiring {
                if let Some(&sid) = assignments.get(owner) {
                    free.push(sid);
                }
            }
        }
    }

    let peak_bytes: usize = slots.iter().map(|s| s.capacity_bytes).sum();
    let num_slots = slots.len();
    let savings_ratio = if naive_bytes == 0 {
        0.0
    } else {
        1.0 - (peak_bytes as f64 / naive_bytes as f64)
    };

    Ok(PlanStatus::Complete(ActivationPlan {
        assignments,
        slots,
        peak_bytes,
        num_slots,
        naive_bytes,
        savings_ratio,
    }))
}

/// Convenience: plan directly from fully-static shapes.
///
/// Uses [`static_size_oracle`] internally, so a graph with any symbolic-shaped
/// activation yields [`PlanStatus::Deferred`].
pub fn plan_activations_static(
    graph: &Graph,
    view_map: &ViewMap,
    options: &PlanOptions,
) -> Result<PlanStatus, PlanError> {
    let oracle = static_size_oracle(graph);
    plan_activations(graph, view_map, oracle, options)
}

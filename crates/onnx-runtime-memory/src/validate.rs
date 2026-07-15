//! Post-hoc validation of an [`ActivationPlan`] against the graph's liveness.
//!
//! Recomputes liveness independently of the plan's slot assignments, then
//! enforces the correctness invariants of buffer sharing. Tests call this on
//! every produced plan, and it must *catch* a deliberately corrupted plan.

use std::collections::HashMap;

use onnx_runtime_ir::{Graph, ValueId};

use crate::error::ValidateError;
use crate::liveness::compute_liveness;
use crate::options::PlanOptions;
use crate::oracle::static_size_oracle;
use crate::plan::{ActivationPlan, SlotId};
use crate::view_map::ViewMap;

/// Validate a plan against the graph, using `size_oracle` to check slot
/// capacities. Enforces:
///
/// 1. **No overlap sharing** — two owners with overlapping live intervals never
///    share a slot (the core safety invariant).
/// 2. **Coverage & capacity** — every owner has a slot that exists and is large
///    enough; a graph output's slot is never reused after its def (implied by
///    (1), since an output's interval overlaps everything after its def).
/// 3. **View folding** — every zero-copy view's last use falls within its
///    source owner's live interval (the source outlives the alias).
pub fn validate<F>(
    plan: &ActivationPlan,
    graph: &Graph,
    view_map: &ViewMap,
    size_oracle: F,
    options: &PlanOptions,
) -> Result<(), ValidateError>
where
    F: Fn(ValueId) -> Option<usize>,
{
    let live = compute_liveness(graph, view_map, options).map_err(|_| ValidateError::Cycle)?;

    let slot_cap: HashMap<SlotId, usize> =
        plan.slots.iter().map(|s| (s.id, s.capacity_bytes)).collect();

    // (1) + (2): coverage, capacity, and overlap-free slot sharing.
    let mut by_slot: HashMap<SlotId, Vec<ValueId>> = HashMap::new();
    for &owner in live.intervals.keys() {
        let Some(&sid) = plan.assignments.get(&owner) else {
            return Err(ValidateError::MissingAssignment { value: owner });
        };
        let Some(&cap) = slot_cap.get(&sid) else {
            return Err(ValidateError::UnknownSlot {
                value: owner,
                slot: sid,
            });
        };
        if let Some(need) = size_oracle(owner)
            && need > cap
        {
            return Err(ValidateError::UndersizedSlot {
                value: owner,
                slot: sid,
                needed: need,
                capacity: cap,
            });
        }
        by_slot.entry(sid).or_default().push(owner);
    }

    for (&sid, members) in &by_slot {
        for i in 0..members.len() {
            for j in (i + 1)..members.len() {
                let a = members[i];
                let b = members[j];
                if live.intervals[&a].overlaps(&live.intervals[&b]) {
                    return Err(ValidateError::SlotConflict { a, b, slot: sid });
                }
            }
        }
    }

    // (3): every view's last use is covered by its source owner's interval.
    for vid in graph.values.keys() {
        if !view_map.is_view(vid) {
            continue;
        }
        let root = view_map.root(vid);
        let Some(root_interval) = live.intervals.get(&root) else {
            continue; // source not part of the arena (e.g. excluded input)
        };
        let view_use = graph
            .value(vid)
            .consumers
            .iter()
            .filter_map(|c| live.order_index.get(c).copied())
            .max();
        let view_use = match view_use {
            Some(u) => u.max(view_output_end(graph, &live, vid)),
            None => view_output_end(graph, &live, vid),
        };
        if view_use > root_interval.use_end {
            return Err(ValidateError::ViewOutlivesSource {
                view: vid,
                source_owner: root,
                view_use,
                source_end: root_interval.use_end,
            });
        }
    }

    Ok(())
}

/// If `vid` is a graph output, its liveness extends to the last node.
fn view_output_end(graph: &Graph, live: &crate::liveness::Liveness, vid: ValueId) -> usize {
    if graph.outputs.contains(&vid) {
        live.last_index
    } else {
        0
    }
}

/// Convenience: validate a plan built from fully-static shapes.
pub fn validate_static(
    plan: &ActivationPlan,
    graph: &Graph,
    view_map: &ViewMap,
    options: &PlanOptions,
) -> Result<(), ValidateError> {
    let oracle = static_size_oracle(graph);
    validate(plan, graph, view_map, oracle, options)
}

//! Pure worker-width authority for the default persistent CPU decode pool.
//!
//! Width is resolved in two ordered stages:
//!
//! 1. The affinity/logical-CPU capacity, `available_parallelism` quota, physical
//!    core topology, placement policy, and architecture cap produce one
//!    **global** worker budget. Compact counts the inline dispatcher as an
//!    active decode thread and keeps at least half the logical capacity free;
//!    explicit spread may use one worker per physical core.
//! 2. That budget is distributed across the usable NUMA nodes, then each fully
//!    subscribed node gives up its configured service-core reserve.
//!
//! The ordering is intentional. Quotas and architecture caps are process-wide
//! ceilings, while the reserve is placement-local headroom. Reserving against
//! the unconstrained topology first would remove workers from nodes that a
//! quota-limited or architecture-capped pool never saturates.

use crate::core_topology::CoreTopology;
use crate::decode_affinity::{CorePlacement, NodeShard, NumaTopology};

pub(crate) const DEFAULT_SERVICE_CPUS_PER_NUMA_NODE: usize = 1;

#[derive(Clone, Copy, Debug)]
pub(crate) struct DefaultPoolWidthInputs<'a> {
    pub(crate) allowed_cpus: Option<&'a [usize]>,
    pub(crate) core_topology: Option<&'a CoreTopology>,
    pub(crate) numa_topology: Option<&'a NumaTopology>,
    pub(crate) available_parallelism: usize,
    pub(crate) architecture_cap: Option<usize>,
    pub(crate) service_cpus_per_numa_node: usize,
    pub(crate) placement: CorePlacement,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum DefaultPoolDisposition {
    FlatSingleCpu,
    Pool(ResolvedPoolLayout),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DefaultPoolWidthPlan {
    pub(crate) physical_cores: Option<usize>,
    pub(crate) global_workers: usize,
    pub(crate) disposition: DefaultPoolDisposition,
}

impl DefaultPoolWidthPlan {
    #[cfg(test)]
    pub(crate) fn realized_workers(&self) -> usize {
        match &self.disposition {
            DefaultPoolDisposition::FlatSingleCpu => 0,
            DefaultPoolDisposition::Pool(layout) => layout.realized_workers(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PoolLayoutInputs<'a> {
    pub(crate) requested_workers: usize,
    pub(crate) allowed_cpus: Option<&'a [usize]>,
    pub(crate) core_topology: Option<&'a CoreTopology>,
    pub(crate) numa_topology: Option<&'a NumaTopology>,
    pub(crate) available_parallelism: usize,
    pub(crate) service_cpus_per_numa_node: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ResolvedPoolLayout {
    pub(crate) shards: Vec<NodeShard>,
    pub(crate) dispatcher_owns_shard: bool,
}

impl ResolvedPoolLayout {
    #[cfg(test)]
    pub(crate) fn realized_workers(&self) -> usize {
        self.shards.iter().map(|shard| shard.workers).sum::<usize>()
            + usize::from(self.dispatcher_owns_shard)
    }
}

pub(crate) fn resolve_default_pool_width(
    inputs: DefaultPoolWidthInputs<'_>,
) -> Option<DefaultPoolWidthPlan> {
    let logical_capacity = match inputs.allowed_cpus {
        Some(cpus) => inputs.available_parallelism.min(cpus.len()),
        None => inputs.available_parallelism,
    };
    let logical_capacity = std::num::NonZeroUsize::new(logical_capacity)?.get();
    let physical_cores = inputs
        .core_topology
        .map(|topology| match inputs.allowed_cpus {
            Some(cpus) => topology.physical_cores_within(cpus),
            None => topology.core_count(),
        })
        .filter(|cores| *cores > 0);
    // The dispatcher publishes every op and spin-waits for completion, so it
    // consumes scheduling capacity even when it owns no compute shard. Budget
    // spawned workers from the capacity left after that thread, keeping at
    // least floor(logical_capacity / 2) CPUs available to co-tenants. The
    // single-worker floor covers unknown/tiny capacity; the exact two-CPU mask
    // is converted to a dispatcher-only lane below.
    let shared_host_width = (logical_capacity.saturating_sub(1) / 2).max(1);
    let topology_width = match (physical_cores, inputs.placement) {
        (Some(cores), CorePlacement::Compact) => cores.min(shared_host_width),
        (Some(cores), CorePlacement::Spread) => cores,
        (None, _) => shared_host_width,
    };
    let global_workers = inputs
        .architecture_cap
        .filter(|cap| *cap > 0)
        .map_or(topology_width, |cap| topology_width.min(cap))
        .clamp(1, logical_capacity);

    if inputs.allowed_cpus.is_some_and(|cpus| cpus.len() == 1) {
        return Some(DefaultPoolWidthPlan {
            physical_cores,
            global_workers,
            disposition: DefaultPoolDisposition::FlatSingleCpu,
        });
    }

    let mut layout = resolve_pool_layout(PoolLayoutInputs {
        requested_workers: global_workers,
        allowed_cpus: inputs.allowed_cpus,
        core_topology: inputs.core_topology,
        numa_topology: inputs.numa_topology,
        available_parallelism: inputs.available_parallelism,
        service_cpus_per_numa_node: inputs.service_cpus_per_numa_node,
    });
    // With exactly two allowed CPUs, even one spawned worker plus the inline
    // dispatcher consumes the whole cpuset. The compact automatic policy keeps
    // one compute lane but lets the dispatcher own it, leaving the other CPU
    // available to a co-tenant. A one-CPU mask still uses the explicit flat
    // fallback above.
    if inputs.placement == CorePlacement::Compact
        && inputs.allowed_cpus.is_some_and(|cpus| cpus.len() == 2)
        && global_workers == 1
        && layout.shards.len() == 1
        && layout.shards[0].workers == 1
    {
        layout.shards[0].workers = 0;
        layout.dispatcher_owns_shard = true;
    }
    Some(DefaultPoolWidthPlan {
        physical_cores,
        global_workers,
        disposition: DefaultPoolDisposition::Pool(layout),
    })
}

pub(crate) fn resolve_pool_layout(inputs: PoolLayoutInputs<'_>) -> ResolvedPoolLayout {
    if let Some(topology) = inputs.numa_topology {
        let topology = topology.restrict_to_allowed(inputs.allowed_cpus);
        if let Some(mut shards) = topology.split_workers(inputs.requested_workers) {
            reserve_split_headroom(&mut shards, inputs.service_cpus_per_numa_node);
            return ResolvedPoolLayout {
                shards,
                dispatcher_owns_shard: false,
            };
        }
    }

    let cpus = inputs.allowed_cpus.unwrap_or_default().to_vec();
    let budget = effective_cpu_budget(cpus.len(), inputs.available_parallelism);
    let core_count = inputs
        .core_topology
        .map_or(0, |cores| cores.leaders_within(&cpus).len())
        .min(budget);
    let workers = reserve_single_group_headroom(
        inputs.requested_workers,
        budget,
        core_count,
        inputs.service_cpus_per_numa_node,
    );
    ResolvedPoolLayout {
        shards: vec![NodeShard {
            index: 0,
            cpus,
            workers,
        }],
        dispatcher_owns_shard: workers < inputs.requested_workers,
    }
}

pub(crate) fn effective_cpu_budget(mask_len: usize, parallelism: usize) -> usize {
    match (mask_len, parallelism) {
        (0, _) => 0,
        (m, 0) => m,
        (m, p) => m.min(p),
    }
}

pub(crate) fn reserve_single_group_headroom(
    total: usize,
    allowed_count: usize,
    core_count: usize,
    service_cpus: usize,
) -> usize {
    if allowed_count == 0 || service_cpus == 0 {
        return total;
    }
    if core_count > 0 && total <= core_count {
        return total.min(core_count.saturating_sub(service_cpus)).max(1);
    }
    if total < allowed_count {
        return total;
    }
    allowed_count.saturating_sub(service_cpus).max(1)
}

pub(crate) fn reserve_split_headroom(shards: &mut [NodeShard], service_cpus: usize) {
    if service_cpus == 0 {
        return;
    }
    for shard in shards {
        let cap = shard.cpus.len().saturating_sub(service_cpus).max(1);
        shard.workers = shard.workers.min(cap);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_smt(cpus: usize) -> CoreTopology {
        CoreTopology::from_sibling_groups((0..cpus).map(|cpu| vec![cpu]))
    }

    fn numa(nodes: impl IntoIterator<Item = (usize, Vec<usize>)>) -> NumaTopology {
        NumaTopology::from_node_cpus(nodes)
    }

    fn plan<'a>(
        allowed: Option<&'a [usize]>,
        cores: Option<&'a CoreTopology>,
        numa: Option<&'a NumaTopology>,
        available: usize,
        architecture_cap: Option<usize>,
        placement: CorePlacement,
    ) -> DefaultPoolWidthPlan {
        resolve_default_pool_width(DefaultPoolWidthInputs {
            allowed_cpus: allowed,
            core_topology: cores,
            numa_topology: numa,
            available_parallelism: available,
            architecture_cap,
            service_cpus_per_numa_node: DEFAULT_SERVICE_CPUS_PER_NUMA_NODE,
            placement,
        })
        .expect("the table only contains non-zero hosts")
    }

    #[test]
    fn placement_controls_whether_a_no_smt_cpuset_keeps_shared_host_headroom() {
        let allowed: Vec<usize> = (0..96).collect();
        let cores = no_smt(96);
        let four_nodes = numa((0..4).map(|node| (node, (node * 24..(node + 1) * 24).collect())));
        let compact = plan(
            Some(&allowed),
            Some(&cores),
            Some(&four_nodes),
            96,
            None,
            CorePlacement::Compact,
        );
        assert_eq!(compact.global_workers, 47);
        assert_eq!(
            compact.realized_workers(),
            47,
            "47 workers plus the dispatcher must leave half of a 96-CPU cpuset free"
        );

        let spread = plan(
            Some(&allowed),
            Some(&cores),
            Some(&four_nodes),
            96,
            None,
            CorePlacement::Spread,
        );
        assert_eq!(spread.global_workers, 96);
        assert_eq!(
            spread.realized_workers(),
            92,
            "explicit spread preserves the dedicated-host width, less one service CPU per node"
        );
    }

    #[test]
    fn default_width_properties_cover_global_clamps_before_numa_reserve() {
        let allowed: Vec<usize> = (0..96).collect();
        let cores = no_smt(96);
        let four_nodes = numa((0..4).map(|node| (node, (node * 24..(node + 1) * 24).collect())));
        let cases = [
            ("quota-below-topology", 4, None, 1, 1),
            ("quota-one", 1, None, 1, 1),
            ("linux-aarch64-over-eight", 96, Some(8), 8, 8),
        ];
        for (name, available, cap, global, realized) in cases {
            let plan = plan(
                Some(&allowed),
                Some(&cores),
                Some(&four_nodes),
                available,
                cap,
                CorePlacement::Compact,
            );
            assert_eq!(plan.global_workers, global, "{name}: global clamp");
            assert_eq!(plan.realized_workers(), realized, "{name}: NUMA reserve");
        }
    }

    #[test]
    fn architecture_cap_does_not_inflate_an_eight_or_smaller_host() {
        let allowed: Vec<usize> = (0..6).collect();
        let cores = no_smt(6);
        let two_nodes = numa([(0, (0..3).collect()), (1, (3..6).collect())]);
        let plan = plan(
            Some(&allowed),
            Some(&cores),
            Some(&two_nodes),
            6,
            Some(8),
            CorePlacement::Compact,
        );
        assert_eq!(plan.global_workers, 2);
        assert_eq!(
            plan.realized_workers(),
            2,
            "the architecture ceiling must not inflate the shared-host half-capacity budget"
        );
    }

    #[test]
    fn architecture_cap_is_a_ceiling_not_an_eight_worker_override() {
        let allowed: Vec<usize> = (0..8).collect();
        let cores =
            CoreTopology::from_sibling_groups((0..4).map(|core| vec![core * 2, core * 2 + 1]));
        let plan = plan(
            Some(&allowed),
            Some(&cores),
            None,
            8,
            Some(8),
            CorePlacement::Compact,
        );
        assert_eq!(
            plan.global_workers, 3,
            "three workers plus the dispatcher leave half of eight logical CPUs free"
        );
        assert_eq!(plan.realized_workers(), 3);
    }

    #[test]
    fn sparse_affinity_is_restricted_before_per_node_reserve() {
        let cores = CoreTopology::from_sibling_groups([
            vec![0, 1],
            vec![2, 3],
            vec![100, 101],
            vec![102, 103],
        ]);
        let nodes = numa([(0, vec![0, 1, 2, 3]), (1, vec![100, 101, 102, 103])]);
        let allowed = [1, 2, 101, 103];
        let plan = plan(
            Some(&allowed),
            Some(&cores),
            Some(&nodes),
            4,
            None,
            CorePlacement::Compact,
        );
        assert_eq!(plan.physical_cores, Some(4));
        assert_eq!(plan.global_workers, 1);
        assert_eq!(plan.realized_workers(), 1);
        let DefaultPoolDisposition::Pool(layout) = plan.disposition else {
            panic!("a four-CPU sparse mask must build a pool");
        };
        assert_eq!(layout.shards.len(), 1);
        assert_eq!(layout.shards[0].cpus, allowed);
    }

    #[test]
    fn heterogeneous_numa_preserves_the_existing_even_split_and_local_caps() {
        let allowed: Vec<usize> = (0..10).collect();
        let cores = no_smt(10);
        let nodes = numa([(0, vec![0, 1]), (1, (2..10).collect())]);
        let plan = plan(
            Some(&allowed),
            Some(&cores),
            Some(&nodes),
            10,
            None,
            CorePlacement::Compact,
        );
        assert_eq!(plan.global_workers, 4);
        assert_eq!(plan.realized_workers(), 3);
        let DefaultPoolDisposition::Pool(layout) = plan.disposition else {
            panic!("a two-node host must build a split pool");
        };
        assert_eq!(
            layout
                .shards
                .iter()
                .map(|shard| shard.workers)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[test]
    fn unknown_topology_still_accounts_for_the_dispatcher() {
        let allowed: Vec<usize> = (0..16).collect();
        let plan = plan(Some(&allowed), None, None, 16, None, CorePlacement::Compact);
        assert_eq!(plan.physical_cores, None);
        assert_eq!(plan.global_workers, 7);
        assert_eq!(plan.realized_workers(), 7);
    }

    #[test]
    fn two_cpu_compact_default_is_dispatcher_only() {
        let allowed = [0, 2];
        let cores = no_smt(2);
        let plan = plan(
            Some(&allowed),
            Some(&cores),
            None,
            2,
            None,
            CorePlacement::Compact,
        );
        assert_eq!(plan.global_workers, 1);
        assert_eq!(plan.realized_workers(), 1);
        let DefaultPoolDisposition::Pool(layout) = plan.disposition else {
            panic!("two allowed CPUs must retain one inline compute lane");
        };
        assert!(layout.dispatcher_owns_shard);
        assert_eq!(layout.shards[0].workers, 0);
    }

    #[test]
    fn single_cpu_affinity_is_an_explicit_flat_fallback() {
        let allowed = [7];
        let cores = CoreTopology::from_sibling_groups([vec![7]]);
        let plan = plan(
            Some(&allowed),
            Some(&cores),
            None,
            1,
            None,
            CorePlacement::Compact,
        );
        assert_eq!(plan.global_workers, 1);
        assert_eq!(plan.realized_workers(), 0);
        assert_eq!(plan.disposition, DefaultPoolDisposition::FlatSingleCpu);
    }
}

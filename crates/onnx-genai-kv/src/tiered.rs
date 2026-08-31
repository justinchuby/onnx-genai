//! Tiered storage: hot GPU-resident pages with cold CPU offload.
//!
//! The current backend stores both tiers in host RAM, but the page table treats
//! `Device::Gpu(0)` as the hot tier and `Device::Cpu` as the cold tier. These
//! are declared/emulated residency locations: both stores are host-addressable.
//! Moving a page allocates a target store, copies every storage component, and
//! atomically replaces the source.
//!
//! That is the end state for this crate, not a waypoint. The module used to say
//! "until Stage 3 supplies a CUDA store"; #721 stage 3 is superseded, because on
//! native CUDA device KV paging is owned by the VMM layer (`CudaVmmAllocator`,
//! #740/#745/#748) and a second page allocator here would duplicate that
//! ownership rather than complete it. The factory-and-copy contract stays
//! because it is what lets an out-of-tree backend implement a device store
//! without changing cache-facing callers -- an optional view, not a prerequisite
//! anybody in this repository is waiting on.
//!
//! Quantized K/V storage supports symmetric int8 and scaled FP8 E4M3FN/E5M2.
//! Each layer, K/V component, and head has an independent scale. On write, f32
//! values are quantized into compact page storage; reads reconstruct f32 values.

/// Independent inputs consumed by runtime tiering.
///
/// Moving an authoritative copy to another lossless store is distinct from
/// discarding a reusable copy and recomputing it later. `spillable` is a
/// physical backend/type capability; `recomputable` is semantic permission.
/// Callers must derive them independently.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StateStorageDisposition {
    pub spillable: bool,
    pub recomputable: bool,
}

impl StateStorageDisposition {
    pub const fn new(spillable: bool, recomputable: bool) -> Self {
        Self {
            spillable,
            recomputable,
        }
    }
}

/// Runtime budgets used to choose placement without changing state metadata.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeTieringPolicy {
    pub hot_page_budget: usize,
    pub cached_page_budget: usize,
}

impl RuntimeTieringPolicy {
    pub const fn new(hot_page_budget: usize, cached_page_budget: usize) -> Self {
        Self {
            hot_page_budget,
            cached_page_budget,
        }
    }

    pub fn needs_cold_migration(self, hot_pages: usize, incoming_pages: usize) -> bool {
        hot_pages.saturating_add(incoming_pages) > self.hot_page_budget
    }

    pub fn exceeds_cache_budget(self, cached_pages: usize) -> bool {
        cached_pages > self.cached_page_budget
    }

    pub fn may_spill_payload(
        self,
        disposition: StateStorageDisposition,
        backing_store_available: bool,
    ) -> bool {
        backing_store_available && disposition.spillable
    }

    pub fn may_evict_for_recompute(self, disposition: StateStorageDisposition) -> bool {
        disposition.recomputable
    }

    pub fn prefer_restore(
        estimated_load_ms: f64,
        num_tokens: usize,
        recompute_ms_per_token: f64,
    ) -> bool {
        estimated_load_ms <= num_tokens as f64 * recompute_ms_per_token.max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_budgets_drive_placement_without_semantic_fields() {
        let policy = RuntimeTieringPolicy::new(4, 8);
        assert!(!policy.needs_cold_migration(3, 1));
        assert!(policy.needs_cold_migration(4, 1));
        assert!(!policy.exceeds_cache_budget(8));
        assert!(policy.exceeds_cache_budget(9));
        assert!(RuntimeTieringPolicy::prefer_restore(3.9, 4, 1.0));
        assert!(!RuntimeTieringPolicy::prefer_restore(4.1, 4, 1.0));
    }

    #[test]
    fn spill_and_recompute_permissions_are_independent() {
        let policy = RuntimeTieringPolicy::new(1, 1);
        for (spillable, recomputable) in
            [(false, false), (false, true), (true, false), (true, true)]
        {
            let disposition = StateStorageDisposition::new(spillable, recomputable);
            assert_eq!(
                policy.may_spill_payload(disposition, true),
                spillable,
                "spill decision must use only spillability"
            );
            assert_eq!(
                policy.may_evict_for_recompute(disposition),
                recomputable,
                "eviction decision must use only recomputability"
            );
        }
        assert!(!policy.may_spill_payload(StateStorageDisposition::new(true, false), false));
    }
}

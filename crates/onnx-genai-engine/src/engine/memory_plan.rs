//! Every claim a loaded model makes on memory, in one place.
//!
//! # Why this exists
//!
//! The design is that one governor owns the tiers and every component leases
//! from it. What was actually in the tree was closer to several components each
//! deciding for itself:
//!
//! * the KV pool leased, but only on the ONNX Runtime path where the KV
//!   geometry is known — the native path and the no-geometry fallback did not;
//! * the CUDA weight-residency cache carried its **own** budget, defaulting to
//!   4 GiB or read from `ONNX_GENAI_WEIGHT_OFFLOAD_DEVICE_BYTES`, reconciled
//!   with nothing. Grant KV most of an 8 GiB card and let residency default to
//!   4 GiB and both are individually satisfied while the card is
//!   oversubscribed;
//! * recurrent state — the fixed-size `conv_state`/`recurrent_state` a hybrid
//!   decoder keeps per sequence — was allocated and never charged at all;
//! * activations were reported as zero, which is not the same as free.
//!
//! Each of those is defensible alone. Together they mean no single place can
//! answer "what does this model hold", which is the question the whole
//! architecture is built to answer.
//!
//! # What it does
//!
//! Holds the leases. A component that needs memory is given a budget from here
//! instead of choosing one, so its claim is visible to every other claim on the
//! same tier, and a model that does not fit is refused at load rather than
//! discovered at a `cuMemAlloc` in the middle of generation.
//!
//! It also owns the holder identities. `HolderId` documents that "uniqueness is
//! the caller's responsibility", and the callers were three `HolderId::new(N)`
//! constants with hand-picked numbers in two different files.

use onnx_runtime_memory_governor::{HolderId, MemoryError, MemoryGovernor, MemoryRole, Tier};

/// Who holds a lease, so the governor can attribute and ask for a release.
///
/// An enum rather than scattered constants: the previous arrangement was three
/// `HolderId::new(N)` definitions in two files, unique by inspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Holder {
    /// The target model's KV page pool.
    KvPool,
    /// A composite pipeline's KV page pool.
    PipelineKvPool,
    /// A speculative draft model's own KV page pool.
    DraftKvPool,
    /// The device weight-residency cache, when weight offload is enabled.
    WeightResidency,
    /// Fixed-size recurrent state for hybrid decoders.
    RecurrentState,
    /// Intermediate activations for one graph execution.
    Activations,
}

impl Holder {
    /// The identifier the governor accounts against.
    ///
    /// The first three values are pinned to what the previous constants used, so
    /// a trace recorded before this module is still readable against one after.
    pub(crate) const fn id(self) -> HolderId {
        HolderId::new(match self {
            Holder::KvPool => 1,
            Holder::PipelineKvPool => 2,
            Holder::DraftKvPool => 3,
            Holder::WeightResidency => 4,
            Holder::RecurrentState => 5,
            Holder::Activations => 6,
        })
    }

    /// What this holder's bytes are, which decides how they are treated under
    /// pressure.
    pub(crate) const fn role(self) -> MemoryRole {
        match self {
            Holder::KvPool | Holder::PipelineKvPool | Holder::DraftKvPool => MemoryRole::KvCache,
            Holder::WeightResidency => MemoryRole::Weights,
            // Rolling state that is destroyed as it is updated: it cannot be
            // rewound, recomputed or shared, so it is not a `Weights`-style
            // demotion candidate and not step-scoped `Workspace` either. It
            // lives and dies with its sequence, like KV.
            Holder::RecurrentState => MemoryRole::KvCache,
            Holder::Activations => MemoryRole::Activation,
        }
    }

    /// A name for diagnostics.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Holder::KvPool => "KV page pool",
            Holder::PipelineKvPool => "pipeline KV page pool",
            Holder::DraftKvPool => "draft model KV page pool",
            Holder::WeightResidency => "device weight residency cache",
            Holder::RecurrentState => "recurrent state",
            Holder::Activations => "activations",
        }
    }
}

/// The leases a loaded model holds, beyond the KV pool.
///
/// The KV pool's lease lives inside `PagedKvCache`, which takes it at
/// construction so the pool cannot exist without it. Everything else is held
/// here.
#[derive(Debug, Default)]
pub(crate) struct ModelMemoryPlan {
    entries: Vec<PlanEntry>,
}

#[derive(Debug)]
struct PlanEntry {
    holder: Holder,
    tier: Tier,
    lease: onnx_runtime_memory_governor::MemoryLease,
}

impl ModelMemoryPlan {
    /// Nothing reserved yet.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Reserve `bytes` for `holder`, or fail saying what could not be granted.
    ///
    /// Zero is not an error and takes no lease: a model with no recurrent
    /// layers asks for no recurrent state, and that is a fact about the model
    /// rather than a failure.
    pub(crate) fn reserve(
        &mut self,
        governor: &dyn MemoryGovernor,
        holder: Holder,
        tier: Tier,
        bytes: u64,
    ) -> Result<u64, MemoryError> {
        if bytes == 0 {
            return Ok(0);
        }
        let lease = governor.reserve(tier, bytes, holder.role(), holder.id())?;
        let granted = lease.bytes();
        self.entries.push(PlanEntry {
            holder,
            tier,
            lease,
        });
        Ok(granted)
    }

    /// Bytes held on `tier`, across every holder.
    pub(crate) fn bytes_on(&self, tier: Tier) -> u64 {
        self.entries
            .iter()
            .filter(|entry| entry.tier == tier)
            .map(|entry| entry.lease.bytes())
            .sum()
    }

    /// What is held, for diagnostics and for the profile report.
    pub(crate) fn breakdown(&self) -> Vec<(&'static str, Tier, u64)> {
        self.entries
            .iter()
            .map(|entry| (entry.holder.name(), entry.tier, entry.lease.bytes()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_memory_governor::{LeaseLedger, LedgerGovernor};

    fn governor(device_bytes: u64) -> LedgerGovernor {
        LedgerGovernor::new(LeaseLedger::new(device_bytes, 0, 0))
    }

    /// Holder identities do not collide.
    ///
    /// They were three hand-picked constants in two files, unique by
    /// inspection. Two holders sharing an id would let the governor attribute
    /// one's bytes to the other and ask the wrong component to release.
    #[test]
    fn every_holder_has_its_own_identity() {
        let holders = [
            Holder::KvPool,
            Holder::PipelineKvPool,
            Holder::DraftKvPool,
            Holder::WeightResidency,
            Holder::RecurrentState,
            Holder::Activations,
        ];
        for (index, holder) in holders.iter().enumerate() {
            for other in &holders[index + 1..] {
                assert_ne!(
                    holder.id().get(),
                    other.id().get(),
                    "{} and {} share a holder id",
                    holder.name(),
                    other.name()
                );
            }
        }
    }

    /// Claims made through the plan are counted against one another.
    ///
    /// This is the property the arrangement it replaces did not have: the
    /// weight-residency cache chose its own budget, so it and the KV pool could
    /// each be satisfied while together exceeding the device.
    #[test]
    fn two_holders_cannot_both_be_granted_more_than_the_tier_has() {
        let governor = governor(1000);
        let mut plan = ModelMemoryPlan::new();

        plan.reserve(&governor, Holder::WeightResidency, Tier::Device, 700)
            .expect("the first claim fits");
        let refused = plan
            .reserve(&governor, Holder::RecurrentState, Tier::Device, 700)
            .expect_err("700 more does not fit in what is left of 1000");

        assert!(
            matches!(refused, MemoryError::TierExhausted { .. }),
            "a refusal must say the tier is exhausted, not something else: {refused}"
        );
        assert_eq!(plan.bytes_on(Tier::Device), 700, "the failed claim leaked");
    }

    /// A model with no recurrent layers asks for nothing and that is not an
    /// error.
    #[test]
    fn a_zero_byte_claim_takes_no_lease() {
        let governor = governor(1000);
        let mut plan = ModelMemoryPlan::new();
        assert_eq!(
            plan.reserve(&governor, Holder::RecurrentState, Tier::Device, 0)
                .expect("zero is not a failure"),
            0
        );
        assert!(plan.breakdown().is_empty());
        assert_eq!(plan.bytes_on(Tier::Device), 0);
    }

    /// Dropping the plan returns everything it held.
    #[test]
    fn releasing_the_plan_returns_the_bytes_to_the_tier() {
        let governor = governor(1000);
        {
            let mut plan = ModelMemoryPlan::new();
            plan.reserve(&governor, Holder::WeightResidency, Tier::Device, 800)
                .expect("fits");
            assert_eq!(governor.available(Tier::Device), 200);
        }
        assert_eq!(
            governor.available(Tier::Device),
            1000,
            "unloading a model must return its memory"
        );
    }
}

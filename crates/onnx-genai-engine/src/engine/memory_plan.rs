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
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "RecurrentState and Activations name claims that are not charged yet -- \
                  see #639 and #514. They are defined here rather than added later so the \
                  identity space is decided in one place, which is the whole reason this \
                  enum replaced three hand-picked constants."
    )
)]
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
    /// The fixed device reservation the engine's ceiling already accounts for --
    /// model weights and runtime overhead.
    ///
    /// Charged as a lease so the ledger's device tier can be the device itself.
    /// Seeding that tier with the KV sub-budget instead was what stopped every
    /// other holder from joining it.
    FixedDeviceReservation,
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
            Holder::FixedDeviceReservation => 7,
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
            Holder::FixedDeviceReservation => MemoryRole::Weights,
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
            Holder::FixedDeviceReservation => "fixed device reservation",
        }
    }

    /// Which tier this holder's bytes come from.
    ///
    /// A property of the holder, not a choice at the call site. Every caller
    /// used to pass it, which meant every caller could pass the wrong one --
    /// and the KV pools genuinely are `Host` while the rest are `Device`, so it
    /// is the kind of thing that gets copied from the wrong neighbour.
    pub(crate) const fn tier(self) -> Tier {
        match self {
            // The pool is host-allocated despite its `num_gpu_pages` lineage.
            // Charging it to `Device` would let it exhaust host RAM while the
            // device ledger still reported headroom.
            Holder::KvPool | Holder::PipelineKvPool | Holder::DraftKvPool => Tier::Host,
            Holder::WeightResidency
            | Holder::RecurrentState
            | Holder::Activations
            | Holder::FixedDeviceReservation => Tier::Device,
        }
    }
}

/// The leases a loaded model holds, beyond the KV pool.
///
/// The KV pool's lease lives inside `PagedKvCache`, which takes it at
/// construction so the pool cannot exist without it. Everything else is held
/// here.
///
/// Owns the governor handle so a call site says what it wants and how much,
/// and nothing else. The alternative -- every caller passing
/// `(governor, tier, holder)` -- is three chances to pass the wrong thing at
/// every site, and the tier in particular is not a choice: it belongs to the
/// holder.
#[derive(Debug)]
pub(crate) struct ModelMemoryPlan {
    governor: onnx_runtime_memory_governor::LedgerGovernor,
    entries: Vec<PlanEntry>,
}

#[derive(Debug)]
struct PlanEntry {
    holder: Holder,
    /// Set when the tier was a fact about the running system rather than about
    /// the holder -- see [`ModelMemoryPlan::reserve_on`].
    tier: Option<Tier>,
    lease: onnx_runtime_memory_governor::MemoryLease,
}

impl ModelMemoryPlan {
    /// A plan that leases from `governor`.
    pub(crate) fn new(governor: onnx_runtime_memory_governor::LedgerGovernor) -> Self {
        Self {
            governor,
            entries: Vec::new(),
        }
    }

    /// Reserve `bytes` for `holder`, or fail saying what could not be granted.
    ///
    /// Zero is not an error and takes no lease: a model with no recurrent
    /// layers asks for no recurrent state, and that is a fact about the model
    /// rather than a failure.
    pub(crate) fn reserve(&mut self, holder: Holder, bytes: u64) -> Result<u64, MemoryError> {
        if bytes == 0 {
            return Ok(0);
        }
        let lease = self
            .governor
            .reserve(holder.tier(), bytes, holder.role(), holder.id())?;
        let granted = lease.bytes();
        self.entries.push(PlanEntry {
            holder,
            tier: None,
            lease,
        });
        Ok(granted)
    }

    #[cfg(feature = "native-backend")]
    /// Put a provider's standing pool on this plan's governor.
    ///
    /// A provider built before the governor existed sized its pool for itself.
    /// This is where that becomes a claim the rest of the plan can see. The
    /// provider keeps the lease, because it is the thing that must outlive it.
    pub(crate) fn adopt_provider_pool(
        &self,
        session: &crate::native_decode::NativeDecodeSession,
        holder: Holder,
    ) -> anyhow::Result<u64> {
        session.adopt_memory_governor(&self.governor, holder.tier(), holder.id())
    }

    /// Reserve `bytes` for `holder` on a tier the caller determined.
    ///
    /// Only for holders whose tier is a fact about the running system rather
    /// than about the holder: recurrent state lives on the host for a CPU
    /// session and on the device for a CUDA one, so `Holder::tier()` cannot
    /// answer for it. Everything else uses [`Self::reserve`], which does not let
    /// a call site pick.
    pub(crate) fn reserve_on(
        &mut self,
        holder: Holder,
        tier: Tier,
        bytes: u64,
    ) -> Result<u64, MemoryError> {
        if bytes == 0 {
            return Ok(0);
        }
        let lease = self
            .governor
            .reserve(tier, bytes, holder.role(), holder.id())?;
        let granted = lease.bytes();
        self.entries.push(PlanEntry {
            holder,
            tier: Some(tier),
            lease,
        });
        Ok(granted)
    }

    /// Bytes held on `tier`, across every holder.
    pub(crate) fn bytes_on(&self, tier: Tier) -> u64 {
        self.entries
            .iter()
            .filter(|entry| entry.tier.unwrap_or_else(|| entry.holder.tier()) == tier)
            .map(|entry| entry.lease.bytes())
            .sum()
    }

    /// Build a KV page pool that leases from this plan's governor.
    ///
    /// The pool holds its own lease -- `PagedKvCache` takes one at construction
    /// so it cannot exist without one -- which is why this hands the governor
    /// over rather than reserving here. What it does own is the
    /// `(governor, tier, holder)` triple, so a call site says what pool it wants
    /// and how big, and cannot pass the device tier for a pool that is
    /// host-allocated.
    pub(crate) fn kv_pool(
        &self,
        holder: Holder,
        page_size: usize,
        dtype: onnx_genai_kv::KvDType,
        layer_configs: Vec<onnx_genai_kv::LayerTensorConfig>,
        pages: usize,
    ) -> Result<onnx_genai_kv::PagedKvCache, onnx_genai_kv::KvError> {
        onnx_genai_kv::PagedKvCache::new_leased(
            page_size,
            dtype,
            layer_configs,
            pages,
            &self.governor,
            holder.tier(),
            holder.id(),
        )
    }

    /// What is held, for diagnostics and for the profile report.
    pub(crate) fn breakdown(&self) -> Vec<(&'static str, Tier, u64)> {
        self.entries
            .iter()
            .map(|entry| {
                (
                    entry.holder.name(),
                    entry.tier.unwrap_or_else(|| entry.holder.tier()),
                    entry.lease.bytes(),
                )
            })
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
            Holder::FixedDeviceReservation,
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
        let mut plan = ModelMemoryPlan::new(governor.clone());

        plan.reserve(Holder::WeightResidency, 700)
            .expect("the first claim fits");
        let refused = plan
            .reserve(Holder::RecurrentState, 700)
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
        let mut plan = ModelMemoryPlan::new(governor.clone());
        assert_eq!(
            plan.reserve(Holder::RecurrentState, 0)
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
            let mut plan = ModelMemoryPlan::new(governor.clone());
            plan.reserve(Holder::WeightResidency, 800).expect("fits");
            assert_eq!(governor.available(Tier::Device), 200);
        }
        assert_eq!(
            governor.available(Tier::Device),
            1000,
            "unloading a model must return its memory"
        );
    }

    /// The engine's device tier is the device, and the fixed reservation is a
    /// claim on it rather than arithmetic done before it.
    ///
    /// The ledger used to be seeded with `derived_budget.kv_bytes`, so its
    /// device tier meant "bytes KV may have". Safe for the one holder there was,
    /// and precisely why nothing else could join: a weight-residency pool leased
    /// from it would have been charged twice and taken the room out of KV.
    ///
    /// This asserts the shape that unblocked it -- ceiling in, reservation
    /// charged, remainder available to everyone else -- without needing a real
    /// engine.
    #[test]
    fn charging_the_fixed_reservation_leaves_the_rest_for_every_other_holder() {
        let ceiling = 1000;
        let reservation = 300;
        let governor = governor(ceiling);

        let mut plan = ModelMemoryPlan::new(governor.clone());
        plan.reserve(Holder::FixedDeviceReservation, reservation)
            .expect("the reservation fits its own ceiling");

        assert_eq!(
            governor.available(Tier::Device),
            ceiling - reservation,
            "what is left must be the ceiling less the reservation"
        );

        // A weight-residency pool and a recurrent-state reservation now compete
        // for the same remainder, which is the entire point. Both are device
        // holders; `Holder::KvPool` is deliberately not used here because its
        // tier is Host -- the pool is host-allocated despite its `num_gpu_pages`
        // lineage, and the holder now carries that rather than each call site.
        plan.reserve(Holder::WeightResidency, 500)
            .expect("500 of the remaining 700 fits");
        let refused = plan
            .reserve(Holder::RecurrentState, 500)
            .expect_err("only 200 is left");
        assert!(matches!(refused, MemoryError::TierExhausted { .. }));

        assert_eq!(plan.bytes_on(Tier::Device), reservation + 500);
    }

    /// A holder's tier is a property of the holder, not a call-site argument.
    ///
    /// The KV pools are `Host` -- the pool is host-allocated despite its
    /// `num_gpu_pages` lineage, and charging it to `Device` would let it exhaust
    /// host RAM while the device ledger still reported headroom. Every caller
    /// used to pass the tier, which is one chance per site to copy the wrong one
    /// from a neighbour.
    #[test]
    fn a_holders_tier_comes_from_the_holder() {
        assert_eq!(Holder::KvPool.tier(), Tier::Host);
        assert_eq!(Holder::PipelineKvPool.tier(), Tier::Host);
        assert_eq!(Holder::DraftKvPool.tier(), Tier::Host);
        assert_eq!(Holder::WeightResidency.tier(), Tier::Device);
        assert_eq!(Holder::RecurrentState.tier(), Tier::Device);
        assert_eq!(Holder::Activations.tier(), Tier::Device);
        assert_eq!(Holder::FixedDeviceReservation.tier(), Tier::Device);
    }

    /// A reservation that was not applied is not charged.
    ///
    /// The scheduler drops it when honouring it would leave no room for even one
    /// KV page, because it is an estimate over a possibly provisional ceiling
    /// and must never be the reason a model refuses to start. Charging it anyway
    /// would reintroduce that refusal through the ledger.
    #[test]
    fn a_reservation_that_was_not_applied_is_not_charged() {
        let governor = governor(1000);
        let mut plan = ModelMemoryPlan::new(governor.clone());
        plan.reserve(Holder::FixedDeviceReservation, 0)
            .expect("nothing to charge is not a failure");
        assert_eq!(governor.available(Tier::Device), 1000);
    }

    /// The tier total sees leases the plan does not hold.
    ///
    /// The KV pool's lease lives inside `PagedKvCache` and the weight-residency
    /// pool's inside the execution provider, because each is held by the thing
    /// that must outlive it. Summing the plan's own entries therefore misses
    /// them -- and they are the two largest holders, so a total that omitted
    /// them would be read as the whole picture and be wrong by most of it.
    #[test]
    fn the_tier_total_counts_leases_the_plan_does_not_hold() {
        let governor = governor(1000);
        let mut plan = ModelMemoryPlan::new(governor.clone());

        plan.reserve(Holder::FixedDeviceReservation, 100)
            .expect("fits");

        // Something else takes a lease from the same governor and keeps it --
        // exactly what `PagedKvCache` and the CUDA residency cache do.
        let held_elsewhere = governor
            .reserve(Tier::Device, 400, MemoryRole::KvCache, Holder::KvPool.id())
            .expect("fits");

        assert_eq!(
            plan.bytes_on(Tier::Device),
            100,
            "the plan only knows what it holds itself"
        );
        assert_eq!(
            governor.used(Tier::Device),
            500,
            "the ledger knows both, which is why the total is read from it"
        );

        drop(held_elsewhere);
        assert_eq!(governor.used(Tier::Device), 100);
    }
}

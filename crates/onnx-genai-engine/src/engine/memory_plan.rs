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
    /// The CPU EP's warm host cache for dequantized MoE experts.
    WeightOffloadHostCache,
    /// Fixed-size recurrent state for hybrid decoders.
    RecurrentState,
    /// Semantic-prefix snapshots of recurrent/native loop-carried state.
    RecurrentPrefixSnapshot,
    /// The native decode path's past/present KV tensors.
    ///
    /// Distinct from the `*KvPool` holders: those are page pools that lease
    /// their capacity when constructed. The native path's page table carries no
    /// storage, so its real KV is the session's own tensors, which the
    /// execution provider allocates without telling the ledger.
    NativeKvCache,
    /// Intermediate activations for one graph execution.
    Activations,
    /// The fixed device reservation the engine's ceiling already accounts for --
    /// model weights and runtime overhead.
    ///
    /// Charged as a lease so the ledger's device tier can be the device itself.
    /// Seeding that tier with the KV sub-budget instead was what stopped every
    /// other holder from joining it.
    FixedDeviceReservation,
    /// The standing device pool held by a pipeline component session (a vision
    /// encoder, say) that is loaded lazily alongside the decoder.
    ///
    /// These sessions build their own execution provider, so without a lease
    /// their device memory is spent behind the ledger's back: the decoder's
    /// governor keeps sizing its ceiling from a `measured_free` that silently
    /// includes bytes another session on the same device already holds. Two
    /// co-resident tenants each admitting against the whole card is how a
    /// 3264-patch vision prefill drove a 30B decoder into `CUDA_ERROR_OUT_OF_MEMORY`
    /// with the card genuinely full.
    PipelineComponentPool,
}

impl Holder {
    /// Every holder, so a list of them is maintained in one place.
    ///
    /// Two hand-copied lists in the tests below drifted apart the moment a
    /// variant was added, and the drift is silent: a uniqueness test that does
    /// not mention a holder still passes. `all_covers_every_holder_id` ties
    /// this to the id space so a new variant cannot be left out quietly.
    ///
    /// Production code matches on holders exhaustively and the compiler
    /// enforces that, so nothing outside the tests iterates this today. It is
    /// still the registry rather than test scaffolding -- the id space is
    /// recorded in traces and must not shift with build features, which is why
    /// the variants are unconditional even when only one build configuration
    /// constructs them.
    #[allow(dead_code)]
    pub(crate) const ALL: [Holder; 11] = [
        Holder::PipelineComponentPool,
        Holder::KvPool,
        Holder::PipelineKvPool,
        Holder::DraftKvPool,
        Holder::WeightResidency,
        Holder::WeightOffloadHostCache,
        Holder::RecurrentState,
        Holder::RecurrentPrefixSnapshot,
        Holder::NativeKvCache,
        Holder::Activations,
        Holder::FixedDeviceReservation,
    ];

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
            Holder::WeightOffloadHostCache => 10,
            Holder::RecurrentState => 5,
            Holder::RecurrentPrefixSnapshot => 9,
            Holder::NativeKvCache => 8,
            Holder::Activations => 6,
            Holder::FixedDeviceReservation => 7,
            Holder::PipelineComponentPool => 11,
        })
    }

    /// What this holder's bytes are, which decides how they are treated under
    /// pressure.
    pub(crate) const fn role(self) -> MemoryRole {
        match self {
            Holder::KvPool | Holder::PipelineKvPool | Holder::DraftKvPool => MemoryRole::KvCache,
            Holder::WeightResidency | Holder::WeightOffloadHostCache => MemoryRole::Weights,
            // Rolling state that is destroyed as it is updated: it cannot be
            // rewound, recomputed or shared, so it is not a `Weights`-style
            // demotion candidate and not step-scoped `Workspace` either. It
            // lives and dies with its sequence, like KV.
            Holder::RecurrentState | Holder::RecurrentPrefixSnapshot => MemoryRole::KvCache,
            Holder::NativeKvCache => MemoryRole::KvCache,
            Holder::Activations => MemoryRole::Activation,
            Holder::FixedDeviceReservation => MemoryRole::Weights,
            // A standing pool the component keeps for as long as the session
            // lives, not a per-step scratch buffer.
            Holder::PipelineComponentPool => MemoryRole::Workspace { step_scoped: false },
        }
    }

    /// A name for diagnostics.
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Holder::KvPool => "KV page pool",
            Holder::PipelineKvPool => "pipeline KV page pool",
            Holder::DraftKvPool => "draft model KV page pool",
            Holder::WeightResidency => "device weight residency cache",
            Holder::WeightOffloadHostCache => "host weight-offload expert cache",
            Holder::RecurrentState => "recurrent state",
            Holder::RecurrentPrefixSnapshot => "recurrent prefix snapshots",
            Holder::NativeKvCache => "native decode KV tensors",
            Holder::Activations => "activations",
            Holder::FixedDeviceReservation => "fixed device reservation",
            Holder::PipelineComponentPool => "pipeline component device pool",
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
            Holder::WeightOffloadHostCache => Tier::Host,
            Holder::WeightResidency
            | Holder::RecurrentState
            | Holder::RecurrentPrefixSnapshot
            // The native decode session allocates these through its execution
            // provider, so a CPU-EP session holds them in host memory. The call
            // site supplies the real tier via `reserve_on`; this is the default
            // for a device session.
            | Holder::NativeKvCache
            | Holder::Activations
            | Holder::FixedDeviceReservation
            | Holder::PipelineComponentPool => Tier::Device,
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
    governor: crate::memory_authority::EngineMemoryGovernor,
    entries: Vec<PlanEntry>,
    /// Device KV pool grants taken through [`ModelMemoryPlan::kv_pool`].
    ///
    /// Shared with the admission ceiling, which runs on the scheduler's path
    /// and cannot take this plan's lock.
    kv_pool_bytes: std::sync::Arc<std::sync::atomic::AtomicU64>,
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
    pub(crate) fn new(governor: crate::memory_authority::EngineMemoryGovernor) -> Self {
        Self {
            governor,
            entries: Vec::new(),
            kv_pool_bytes: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
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
    #[cfg(feature = "native-backend")]
    pub(crate) fn adopt_provider_pool(
        &self,
        session: &crate::native_decode::NativeDecodeSession,
        holder: Holder,
    ) -> anyhow::Result<u64> {
        session.adopt_memory_governor(&self.governor, holder.tier(), holder.id())
    }

    /// Give back `bytes` of `holder`'s claim, returning what was actually
    /// released.
    ///
    /// # When a reservation stops being needed
    ///
    /// A reservation exists because nothing else accounts for the bytes. When
    /// something starts to -- a device allocator that commits physically on
    /// demand and leases each granule -- the reservation and the allocator
    /// charge the same memory twice, and the tier reads high by that amount.
    ///
    /// Releasing rather than never taking it is deliberate. At the moment the
    /// reservation is made no session exists, so nobody can yet say whether the
    /// allocator commits on demand. Taking it and giving it back keeps the
    /// conservative answer for the window where the question is unanswerable.
    ///
    /// Returns the bytes released, which may be less than asked for if the
    /// holder never held that many -- a caller correcting a double count needs
    /// to know what it actually corrected.
    ///
    /// Gated with its only caller: the native decode path is where an
    /// on-demand allocator is asked about, and an ungated helper is dead code
    /// in every other build.
    #[cfg(feature = "native-backend")]
    pub(crate) fn release(&mut self, holder: Holder, bytes: u64) -> u64 {
        let Some(entry) = self.entries.iter_mut().find(|entry| entry.holder == holder) else {
            return 0;
        };
        entry.lease.shrink(bytes)
    }

    /// Give the startup weight reservation back before the provider adopts it.
    ///
    /// CUDA weight offload is loaded in two phases: the engine first reserves
    /// the device residency budget as part of the fixed model claim because no
    /// execution provider exists yet, and the provider later adopts the same
    /// budget as the cache lease it will hold. If the startup reservation stays
    /// live while adoption calls `reserve`, the ledger sees two independent
    /// weight claims for one residency cache and refuses a model that actually
    /// fits (#704).
    #[cfg(feature = "native-backend")]
    #[cfg_attr(
        all(not(feature = "cuda"), not(test)),
        expect(
            dead_code,
            reason = "the production caller is the CUDA weight-offload adoption path; the \
                      native-backend-only CI build keeps the regression test but cannot call the \
                      CUDA branch"
        )
    )]
    pub(crate) fn release_fixed_device_reservation_for_provider_pool(&mut self, bytes: u64) -> u64 {
        self.release(Holder::FixedDeviceReservation, bytes)
    }

    /// Reserve `bytes` for `holder` on a tier the caller determined.
    ///
    /// Only for holders whose tier is a fact about the running system rather
    /// than about the holder: recurrent state lives on the host for a CPU
    /// session and on the device for a CUDA one, so `Holder::tier()` cannot
    /// answer for it. Everything else uses [`Self::reserve`], which does not let
    /// a call site pick -- a tier that can be passed is a tier that can be
    /// passed wrong, which is how the pipeline came to charge a host-allocated
    /// KV pool to the device.
    #[cfg(feature = "native-backend")]
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
        if holder == Holder::NativeKvCache && tier == Tier::Device {
            self.kv_pool_bytes
                .fetch_add(granted, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(granted)
    }

    /// Bytes held on `tier` **by the plan itself**.
    ///
    /// Deliberately not the tier total: the KV pool's lease lives inside
    /// `PagedKvCache` and the weight-residency pool's inside the execution
    /// provider, so this misses both. `EngineResourceGovernor::leased_bytes_on`
    /// reads the ledger, which sees everything.
    ///
    /// Kept because the difference between the two is the property worth
    /// asserting -- see `the_tier_total_counts_leases_the_plan_does_not_hold`.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the tier total is read from the ledger; this exists to show that the \
                      plan's own view is the partial one, and is asserted against it"
        )
    )]
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
        let cache = onnx_genai_kv::PagedKvCache::new_leased(
            page_size,
            dtype,
            layer_configs,
            pages,
            &self.governor,
            holder.tier(),
            holder.id(),
        )?;
        // Recorded here rather than at the call site because this is the only
        // place a governed pool is built, so admission cannot be shown a stale
        // figure by someone forgetting to report one.
        if holder.tier() == Tier::Device {
            self.kv_pool_bytes.fetch_add(
                cache.pool_lease_bytes().unwrap_or(0),
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        Ok(cache)
    }

    /// Bytes free on `tier` right now.
    ///
    /// Read from the ledger rather than from this plan's own entries, so it
    /// accounts for every holder rather than the ones this plan happens to
    /// carry.
    ///
    /// Gated with its only caller: the native decode path is what needs to
    /// compare a worst case against real headroom, and an ungated helper is
    /// dead code in every other build.
    #[cfg(feature = "native-backend")]
    pub(crate) fn available_on(&self, tier: Tier) -> u64 {
        self.governor.available(tier)
    }

    /// A live handle on device KV bytes reserved for admitted sequences.
    ///
    /// Admission needs it to turn "bytes free on the device" into "bytes
    /// available for KV": a device-tier pool's grant is already spent from the
    /// ledger's view, but it is exactly the memory admitted sequences run in,
    /// so it is not competition for them.
    ///
    pub(crate) fn kv_pool_bytes_handle(&self) -> std::sync::Arc<std::sync::atomic::AtomicU64> {
        std::sync::Arc::clone(&self.kv_pool_bytes)
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

    fn governor(device_bytes: u64) -> crate::memory_authority::EngineMemoryGovernor {
        crate::memory_authority::EngineMemoryGovernor::new(
            crate::DeviceMemoryAuthority::new(
                crate::DeviceCompatibilityDomain::Cuda(0),
                device_bytes,
            ),
            0,
            0,
        )
    }

    /// The admission ceiling adds device-tier KV pool grants back to
    /// `available(Device)`, because a pool's own grant is the memory admitted
    /// sequences run in rather than competition for them.
    ///
    /// Paged KV pool holders are host-tier today, so their add-back is zero.
    #[test]
    fn a_host_tier_pool_is_not_added_back() {
        for holder in [Holder::KvPool, Holder::PipelineKvPool, Holder::DraftKvPool] {
            assert_eq!(
                holder.tier(),
                Tier::Host,
                "{} moved to the device tier -- the admission add-back in \
                 LedgerAdmissionCeiling now applies to it, which is correct, but \
                 confirm the ceiling still means what it says before deleting this",
                holder.name()
            );
        }

        let plan = ModelMemoryPlan::new(governor(1024));
        assert_eq!(
            plan.kv_pool_bytes_handle()
                .load(std::sync::atomic::Ordering::Relaxed),
            0,
            "a plan with no device-tier pool has nothing to add back"
        );
    }

    /// The add-back is driven by the holder's tier rather than by the call
    /// site, so a device-tier pool is counted without anyone remembering to
    /// report it.
    #[test]
    fn only_device_tier_holders_reach_the_add_back() {
        // `kv_pool` guards on `holder.tier()`, so the set of holders that can
        // ever contribute is exactly the device-tier ones. Stated as a test
        // because the guard is one line inside a constructor and reads as an
        // optimisation rather than as the correctness condition it is.
        let device_holders: Vec<_> = Holder::ALL
            .into_iter()
            .filter(|holder| holder.tier() == Tier::Device)
            .collect();

        assert!(
            !device_holders.iter().any(|holder| matches!(
                holder,
                Holder::KvPool | Holder::PipelineKvPool | Holder::DraftKvPool
            )),
            "a KV pool is on the device tier, so the add-back is live: \
             {device_holders:?}"
        );
    }

    #[test]
    #[cfg(feature = "native-backend")]
    fn native_device_kv_reservation_is_available_to_admission() {
        let mut plan = ModelMemoryPlan::new(governor(1024));
        plan.reserve_on(Holder::NativeKvCache, Tier::Device, 768)
            .unwrap();

        assert_eq!(plan.available_on(Tier::Device), 256);
        assert_eq!(
            plan.kv_pool_bytes_handle()
                .load(std::sync::atomic::Ordering::Relaxed),
            768
        );
    }

    /// `ALL` covers every holder.
    ///
    /// `id()` is an exhaustive match, so the compiler makes sure a new variant
    /// gets an id. Nothing makes sure it gets added to `ALL`, and a holder
    /// missing from `ALL` silently drops out of every test that iterates it --
    /// including the uniqueness check below, which would then pass while not
    /// checking the new holder at all.
    ///
    /// Ids are consecutive from 1, so a gap in that range means a variant is
    /// missing here.
    #[test]
    fn all_covers_every_holder_id() {
        for id in 1..=Holder::ALL.len() as u64 {
            assert!(
                Holder::ALL.iter().any(|holder| holder.id().get() == id),
                "no holder in Holder::ALL has id {id}; a variant was given an id but not \
                 added to ALL, so nothing that iterates holders covers it"
            );
        }
    }

    /// Holder identities do not collide.
    ///
    /// They were three hand-picked constants in two files, unique by
    /// inspection. Two holders sharing an id would let the governor attribute
    /// one's bytes to the other and ask the wrong component to release.
    #[test]
    fn every_holder_has_its_own_identity() {
        let holders = Holder::ALL;
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

    /// Provider adoption must transfer the startup fixed reservation, not add a
    /// second equal claim.
    ///
    /// This is the #704 offload-only failure in miniature. The engine has
    /// already charged the residency budget as a fixed device reservation
    /// because the CUDA EP did not exist when the governor was built. When the
    /// EP later adopts that same budget, the true device total is still one
    /// residency budget, not the fixed reservation plus another copy.
    #[test]
    #[cfg(feature = "native-backend")]
    fn provider_pool_adoption_replaces_the_fixed_weight_reservation() {
        // The ceiling is deliberately tight: 1_000 leaves room for one 600-byte
        // claim but not for two at once. That is what pins the *ordering* rather
        // than only the final total -- with the release moved after the
        // adoption, the transient peak of 1_200 exceeds the tier and the
        // `reserve` below fails. A roomier ceiling would let the buggy
        // interleaving succeed and settle on the same total, which is exactly
        // the failure this test exists to catch (#704), and the ordering is
        // what the equivalent VMM fix in #667 turned on.
        let governor = governor(1_000);
        let mut plan = ModelMemoryPlan::new(governor.clone());

        plan.reserve(Holder::FixedDeviceReservation, 600)
            .expect("startup reservation fits");
        assert_eq!(governor.used(Tier::Device), 600);

        let _released = plan.release_fixed_device_reservation_for_provider_pool(600);
        let provider_lease = governor
            .reserve(
                Tier::Device,
                600,
                MemoryRole::Weights,
                Holder::WeightResidency.id(),
            )
            .expect("the provider can adopt the transferred budget");

        assert_eq!(
            governor.used(Tier::Device),
            600,
            "adoption must leave the ledger at the true total, not double the residency budget"
        );

        drop(provider_lease);
        assert_eq!(governor.used(Tier::Device), 0);
    }
}

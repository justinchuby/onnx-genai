//! User-facing resource ceilings for the scheduler (DESIGN.md §26.11).
//!
//! The governor resolves vendor-neutral capacity limits and derives the hot-tier
//! KV budget consumed by [`ByteBudget`]. Engine-owned eviction/offload is not
//! performed here; lowering reports the exact overage and required eviction order.

use std::sync::{Arc, Mutex, MutexGuard};

use crate::{ByteBudget, ByteBudgetReconfigureOutcome};

const DEFAULT_VRAM_FRACTION: f32 = 0.90;
const DEFAULT_HOST_RAM_FRACTION: f32 = 0.25;
const DEFAULT_DISK_FRACTION: f32 = 1.0;

/// A resource ceiling resolved against detected tier capacity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResourceLimit {
    /// Absolute ceiling in bytes.
    Bytes(u64),
    /// Fraction of the tier's total detected capacity.
    Fraction(f32),
    /// Use the tier's default fraction.
    Auto,
}

/// User-facing resource ceilings for one engine on one device.
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceLimits {
    pub vram_limit: ResourceLimit,
    pub host_ram_limit: ResourceLimit,
    pub disk_spill_limit: Option<ResourceLimit>,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            vram_limit: ResourceLimit::Fraction(DEFAULT_VRAM_FRACTION),
            host_ram_limit: ResourceLimit::Fraction(DEFAULT_HOST_RAM_FRACTION),
            disk_spill_limit: None,
        }
    }
}

/// Vendor-neutral capacity query supplied by the active execution environment.
///
/// Both queries return `Option`: `None` means the capacity **could not be
/// measured** on this platform, which is a different fact from a small number
/// and must never render as one. A fraction/auto limit resolved against an
/// unmeasured capacity is itself unknown (see [`resolve_limit`]), not a guess.
pub trait CapacityProvider: Send + Sync {
    /// Total capacity in bytes, or `None` when the platform cannot report it.
    fn total_bytes(&self) -> Option<u64>;
    /// Free capacity in bytes, or `None` when the platform cannot report it.
    fn free_bytes(&self) -> Option<u64>;
}

/// Fixed capacity provider useful for tests and statically known tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedCapacity {
    pub total: u64,
    pub free: u64,
}

impl FixedCapacity {
    pub fn new(total: u64, free: u64) -> Self {
        Self { total, free }
    }
}

impl CapacityProvider for FixedCapacity {
    fn total_bytes(&self) -> Option<u64> {
        Some(self.total)
    }

    fn free_bytes(&self) -> Option<u64> {
        Some(self.free)
    }
}

/// A capacity the platform cannot report. Every query returns `None`, so a
/// fraction/auto limit resolved against it is *unknown* rather than a fabricated
/// number. This is what a tier reports when no execution provider, OS call, or
/// filesystem query can supply a real figure — the honest opposite of a
/// specific-looking constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct UnknownCapacity;

impl CapacityProvider for UnknownCapacity {
    fn total_bytes(&self) -> Option<u64> {
        None
    }

    fn free_bytes(&self) -> Option<u64> {
        None
    }
}

/// Capacity providers for the hot, warm, and optional cold tiers.
#[derive(Clone)]
pub struct CapacityProviders {
    pub vram: Arc<dyn CapacityProvider>,
    pub host_ram: Arc<dyn CapacityProvider>,
    pub disk_spill: Option<Arc<dyn CapacityProvider>>,
}

/// Fixed non-KV consumers of the hot-tier ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VramBreakdown {
    pub model_weights_bytes: u64,
    /// Intermediate activations, when they have been measured.
    ///
    /// `None` means *not measured*, which is not the same as zero and must not
    /// be reported as it. Nothing sizes activations yet: the liveness planner in
    /// `onnx-runtime-memory` has no consumer (#514), so the engine cannot answer
    /// and says so rather than answering `0`.
    ///
    /// A `u64` here is how `activations_bytes: 0` came to be published as a
    /// measured fact in the profile JSON until #629. The type now refuses to let
    /// the two be confused.
    pub activations_bytes: Option<u64>,
    /// Runtime overhead, when it has been measured. `None` as above.
    pub ort_overhead_bytes: Option<u64>,
}

impl VramBreakdown {
    /// The fixed reservation, counting only what was actually measured.
    ///
    /// An unmeasured component contributes nothing, which makes the reservation
    /// an under-estimate and admission optimistic. That is the house rule for a
    /// quantity whose absence does not stop anything being built (#649): run,
    /// and be loud about it -- the alternative is refusing to load over a number
    /// nobody has ever computed.
    fn reserved_bytes(self) -> Option<u64> {
        self.model_weights_bytes
            .checked_add(self.activations_bytes.unwrap_or(0))?
            .checked_add(self.ort_overhead_bytes.unwrap_or(0))
    }

    /// Which components of this breakdown were never measured.
    pub fn unmeasured(self) -> Vec<&'static str> {
        let mut names = Vec::new();
        if self.activations_bytes.is_none() {
            names.push("activations");
        }
        if self.ort_overhead_bytes.is_none() {
            names.push("runtime overhead");
        }
        names
    }
}

/// Model-specific mapping between KV bytes, pages, and tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelKvConfig {
    pub page_size_bytes: Option<u64>,
    pub tokens_per_page: u64,
    pub page_geometry_required: bool,
}

impl ModelKvConfig {
    pub fn known(page_size_bytes: u64, tokens_per_page: u64) -> Self {
        Self {
            page_size_bytes: Some(page_size_bytes),
            tokens_per_page,
            page_geometry_required: true,
        }
    }

    pub fn unknown(tokens_per_page: u64) -> Self {
        Self {
            page_size_bytes: None,
            tokens_per_page,
            page_geometry_required: true,
        }
    }

    pub fn no_paged_cache(tokens_per_page: u64) -> Self {
        Self {
            page_size_bytes: None,
            tokens_per_page,
            page_geometry_required: false,
        }
    }

    pub fn bytes_per_token(&self) -> Option<u64> {
        if self.tokens_per_page == 0 {
            return None;
        }
        self.page_size_bytes
            .map(|page_size_bytes| page_size_bytes.div_ceil(self.tokens_per_page))
    }

    pub fn pages_for_bytes(&self, bytes: u64) -> Option<u64> {
        let page_size_bytes = self.page_size_bytes?;
        if page_size_bytes == 0 {
            return Some(0);
        }
        Some(bytes / page_size_bytes)
    }

    pub fn tokens_for_pages(&self, pages: u64) -> Option<u64> {
        pages.checked_mul(self.tokens_per_page)
    }
}

/// Page/token budget derived from the authoritative hot-tier byte ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DerivedBudget {
    pub kv_bytes: u64,
    pub total_pages: u64,
    pub max_total_tokens: u64,
    pub reserved_bytes: u64,
    /// Whether the fixed reservation was actually subtracted from the ceiling.
    ///
    /// False when honoring it would have left no room for even one KV page. The
    /// reservation is an estimate carved out of a ceiling that may itself be
    /// provisional, so it must never be the reason a model refuses to start.
    pub reservation_applied: bool,
}

/// Concrete per-tier ceilings after capacity resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedLimits {
    /// Resolved device (VRAM) ceiling, or `None` when the device capacity could
    /// not be measured and the limit was a fraction/auto. `None` is *unknown*,
    /// not zero and not a provisional constant: nothing downstream may resolve a
    /// device budget from a fraction of an unmeasured capacity.
    pub vram_bytes: Option<u64>,
    pub host_ram_bytes: u64,
    pub disk_spill_bytes: Option<u64>,
}

/// Engine eviction tiers, in the order required by DESIGN.md §26.11.2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionTier {
    BackgroundKv,
    PausedStandardToWarmOrCold,
    RunningStandard,
    InteractiveLast,
}

const EVICTION_ORDER: [EvictionTier; 4] = [
    EvictionTier::BackgroundKv,
    EvictionTier::PausedStandardToWarmOrCold,
    EvictionTier::RunningStandard,
    EvictionTier::InteractiveLast,
];

/// Result of atomically replacing the governor's limits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GovernorReconfigureOutcome {
    pub old_limits: ResolvedLimits,
    pub new_limits: ResolvedLimits,
    pub derived_budget: DerivedBudget,
    pub byte_budget: ByteBudgetReconfigureOutcome,
    /// Hot-tier bytes the engine must reclaim after a lowering.
    pub overage_bytes: u64,
    /// Ordered engine actions to try when `overage_bytes` is non-zero.
    pub eviction_order: Vec<EvictionTier>,
}

/// Point-in-time usage and headroom for one configured resource tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TierSnapshot {
    pub used: u64,
    pub limit: u64,
    pub headroom: u64,
}

impl TierSnapshot {
    fn new(used: u64, limit: u64) -> Self {
        Self {
            used,
            limit,
            headroom: limit.saturating_sub(used),
        }
    }
}

/// Point-in-time governor state, including every configured resource tier.
#[derive(Debug, Clone, PartialEq)]
pub struct GovernorSnapshot {
    pub configured_limits: ResourceLimits,
    pub resolved_limits: ResolvedLimits,
    pub derived_budget: DerivedBudget,
    /// What the fixed (non-KV) reservation is made of.
    ///
    /// `derived_budget.reserved_bytes` is the sum; this is the composition, so a
    /// caller can report *where* device memory went rather than only how much.
    pub breakdown: VramBreakdown,
    pub vram: TierSnapshot,
    pub host_ram: TierSnapshot,
    pub disk_spill: Option<TierSnapshot>,
}

/// Resource-governor configuration failures.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ResourceError {
    #[error(
        "cannot satisfy lowered resource limit: requested {requested_bytes} B, but at least \
         {minimum_bytes} B is required; {reason}; raise the limit to at least \
         {minimum_bytes} B or reduce the model's fixed memory/KV page requirements"
    )]
    CannotSatisfyLoweredLimit {
        requested_bytes: u64,
        minimum_bytes: u64,
        reason: String,
    },

    #[error(
        "invalid {tier} resource fraction {fraction}: expected a finite value in [0, 1]; \
         use ResourceLimit::Bytes, a valid fraction, or ResourceLimit::Auto"
    )]
    InvalidFraction {
        tier: &'static str,
        fraction: String,
    },

    #[error(
        "disk spill was enabled but no disk capacity provider was supplied; provide a \
         filesystem capacity provider or set disk_spill_limit to None"
    )]
    MissingDiskCapacityProvider,

    #[error(
        "cannot resolve the {tier} budget: it was requested as a fraction of capacity but the \
         {tier} capacity could not be measured on this platform; set an explicit byte limit so \
         the runtime is sized against a real number instead of a fabricated one"
    )]
    UnmeasuredCapacity { tier: &'static str },

    #[error(
        "cannot derive the KV memory budget because per-layer KV page geometry is unknown \
         ({tokens_per_page} token(s) per page but no byte size); fix by declaring model.io.kv_inputs \
         and model.io.kv_outputs so the runtime can inspect the model's real KV head geometry"
    )]
    UnknownKvGeometry { tokens_per_page: u64 },

    #[error(
        "cannot derive a valid resource budget because {operation} overflowed u64; {reason}; \
         reduce the configured ceiling, fixed reservations, KV page size, or tokens per page"
    )]
    BudgetArithmeticOverflow {
        operation: &'static str,
        reason: String,
    },

    // ── Ticketed non-blocking pressure protocol (HostGovernor, §5.3.1) ──
    #[error(
        "host page request is invalid: {reason}; request a non-zero extent no larger than the \
         reclaimable host budget"
    )]
    InvalidHostRequest { reason: String },

    #[error(
        "host quota denied: requested {requested_bytes} B but the machine-wide reclaimable host \
         budget is only {reclaimable_budget_bytes} B; this request can never be satisfied even \
         after full reclaim"
    )]
    HostQuotaDenied {
        requested_bytes: u64,
        reclaimable_budget_bytes: u64,
    },

    #[error(
        "host pressure mailbox is full: all {capacity} cancellation slots are reserved; \
         retry after an outstanding ticket resolves"
    )]
    HostMailboxBackpressure { capacity: usize },

    #[error("host pressure ticket {request_id} timed out before it could be granted or claimed")]
    HostPressureTimeout { request_id: u64 },

    #[error(
        "host pressure ticket {request_id} was invalidated by HostGovernor reconfiguration \
         (admitted under generation {stale_generation}, current generation {current_generation})"
    )]
    HostReconfigurationInvalidated {
        request_id: u64,
        stale_generation: u64,
        current_generation: u64,
    },

    #[error(
        "host ledger arithmetic error during {operation}: {reason}; overflow, negative headroom, \
         duplicate physical identity, or a snapshot mismatch is a hard conformance failure"
    )]
    HostLedgerInvariant {
        operation: &'static str,
        reason: String,
    },
}

/// Resolve a limit against a tier's detected total capacity.
///
/// Returns `Ok(None)` when the limit is a fraction (or `auto`) and the tier's
/// capacity could not be measured: a fraction of an unknown is unknown, not a
/// number. An explicit [`ResourceLimit::Bytes`] is always `Ok(Some(bytes))` —
/// it is the caller's authoritative statement about the device and is honoured
/// with or without a measured capacity.
pub fn resolve_limit(
    limit: ResourceLimit,
    capacity: &dyn CapacityProvider,
    tier: &'static str,
) -> Result<Option<u64>, ResourceError> {
    let default_fraction = match tier {
        "vram" => DEFAULT_VRAM_FRACTION,
        "host RAM" => DEFAULT_HOST_RAM_FRACTION,
        "disk spill" => DEFAULT_DISK_FRACTION,
        _ => DEFAULT_DISK_FRACTION,
    };

    match limit {
        // An explicit byte limit is the caller's assertion about the device and
        // is taken at face value. Clamping it to the reported capacity would be
        // right only if that capacity were measured; while it may be unknown,
        // clamping silently discards the one knob a user has for telling the
        // runtime how much memory it may actually use.
        ResourceLimit::Bytes(bytes) => Ok(Some(bytes)),
        // A fraction/auto of an unmeasured capacity is unknown, not a guess.
        ResourceLimit::Auto => Ok(capacity
            .total_bytes()
            .map(|total| resolve_fraction(default_fraction, total))),
        ResourceLimit::Fraction(fraction)
            if fraction.is_finite() && (0.0..=1.0).contains(&fraction) =>
        {
            Ok(capacity
                .total_bytes()
                .map(|total| resolve_fraction(fraction, total)))
        }
        ResourceLimit::Fraction(fraction) => Err(ResourceError::InvalidFraction {
            tier,
            fraction: fraction.to_string(),
        }),
    }
}

fn resolve_fraction(fraction: f32, total_bytes: u64) -> u64 {
    ((total_bytes as f64) * f64::from(fraction)).round() as u64
}

/// Derive the page/token budget after reserving fixed hot-tier consumers.
///
/// `resolved_vram_bytes` is `None` when the device (VRAM) capacity could not be
/// measured and the limit was a fraction. In that case there is **no device
/// ceiling** to size a device KV budget against, so the device KV budget is
/// reported as zero with `reservation_applied = false` — and, crucially, the
/// "raise the VRAM limit" advice is suppressed: telling a user to raise a
/// ceiling that does not exist is worse than saying nothing. Host-tier KV is
/// sized separately and is unaffected.
pub fn derive_kv_budget(
    resolved_vram_bytes: Option<u64>,
    breakdown: &VramBreakdown,
    kv_config: &ModelKvConfig,
) -> Result<DerivedBudget, ResourceError> {
    let reserved_bytes =
        breakdown
            .reserved_bytes()
            .ok_or_else(|| ResourceError::BudgetArithmeticOverflow {
                operation: "summing model weights, activations, and runtime overhead",
                reason: "the fixed VRAM reservations exceed the representable byte range".into(),
            })?;
    let Some(resolved_vram_bytes) = resolved_vram_bytes else {
        // Device capacity is unknown: no ceiling exists to fit the reservation
        // under, so there is no honest device KV budget and no actionable "raise
        // the limit" advice. Report unknown as zero device KV, not a fabricated
        // number derived from a provisional constant.
        if kv_config.page_geometry_required && kv_config.page_size_bytes.is_none() {
            return Err(ResourceError::UnknownKvGeometry {
                tokens_per_page: kv_config.tokens_per_page,
            });
        }
        return Ok(DerivedBudget {
            kv_bytes: 0,
            total_pages: 0,
            max_total_tokens: 0,
            reserved_bytes,
            reservation_applied: false,
        });
    };
    let Some(page_size_bytes) = kv_config.page_size_bytes else {
        if kv_config.page_geometry_required {
            return Err(ResourceError::UnknownKvGeometry {
                tokens_per_page: kv_config.tokens_per_page,
            });
        }
        let (reserved_bytes, reservation_applied) =
            if resolved_vram_bytes < reserved_bytes && reserved_bytes > 0 {
                tracing::warn!(
                    reserved_bytes,
                    resolved_vram_bytes,
                    "fixed memory reservation does not fit under the resolved ceiling; \
                     deriving the non-paged KV budget without it. Raise the VRAM limit \
                     (--vram-limit / serving.memory.limits.vram_limit) so the runtime \
                     is sized against real capacity."
                );
                (0, false)
            } else {
                (reserved_bytes, reserved_bytes > 0)
            };
        let kv_bytes = resolved_vram_bytes
            .checked_sub(reserved_bytes)
            .ok_or_else(|| ResourceError::BudgetArithmeticOverflow {
                operation: "subtracting fixed VRAM reservations from the ceiling",
                reason: "the resolved ceiling is smaller than the fixed reservations".into(),
            })?;
        return Ok(DerivedBudget {
            kv_bytes,
            total_pages: 0,
            max_total_tokens: 0,
            reserved_bytes,
            reservation_applied,
        });
    };
    let minimum_bytes = reserved_bytes.checked_add(page_size_bytes).ok_or_else(|| {
        ResourceError::BudgetArithmeticOverflow {
            operation: "adding one KV page to the fixed VRAM reservations",
            reason: "even the one-page minimum exceeds the representable byte range".into(),
        }
    })?;
    // The reservation is an estimate (measured weights) carved out of a ceiling
    // that may itself be provisional. When the two cannot both hold, drop the
    // reservation rather than refuse to start: the previous behaviour reserved
    // nothing at all, so failing here would be a pure regression.
    let (reserved_bytes, reservation_applied) =
        if page_size_bytes > 0 && resolved_vram_bytes < minimum_bytes && reserved_bytes > 0 {
            tracing::warn!(
                reserved_bytes,
                resolved_vram_bytes,
                page_size_bytes,
                "fixed memory reservation does not fit under the resolved ceiling; \
                 deriving the KV budget without it. Raise the VRAM limit \
                 (--vram-limit / serving.memory.limits.vram_limit) so the KV cache \
                 is sized against real capacity."
            );
            (0, false)
        } else {
            (reserved_bytes, reserved_bytes > 0)
        };
    let minimum_bytes = reserved_bytes.checked_add(page_size_bytes).ok_or_else(|| {
        ResourceError::BudgetArithmeticOverflow {
            operation: "adding one KV page to the fixed VRAM reservations",
            reason: "even the one-page minimum exceeds the representable byte range".into(),
        }
    })?;
    if page_size_bytes == 0 || resolved_vram_bytes < minimum_bytes {
        let reason = if resolved_vram_bytes < reserved_bytes {
            format!(
                "fixed model weights, activations, and runtime overhead reserve {reserved_bytes} B"
            )
        } else if page_size_bytes == 0 {
            "the model reports a zero-byte KV page, so no valid page budget can be derived".into()
        } else {
            let remaining_bytes =
                resolved_vram_bytes
                    .checked_sub(reserved_bytes)
                    .ok_or_else(|| ResourceError::BudgetArithmeticOverflow {
                        operation: "subtracting fixed VRAM reservations from the ceiling",
                        reason: "the resolved ceiling is smaller than the fixed reservations"
                            .into(),
                    })?;
            format!("the remaining {remaining_bytes} B cannot hold one {page_size_bytes} B KV page")
        };
        return Err(ResourceError::CannotSatisfyLoweredLimit {
            requested_bytes: resolved_vram_bytes,
            minimum_bytes,
            reason,
        });
    }

    let kv_bytes = resolved_vram_bytes
        .checked_sub(reserved_bytes)
        .ok_or_else(|| ResourceError::BudgetArithmeticOverflow {
            operation: "subtracting fixed VRAM reservations from the ceiling",
            reason: "the resolved ceiling is smaller than the fixed reservations".into(),
        })?;
    let total_pages =
        kv_config
            .pages_for_bytes(kv_bytes)
            .ok_or(ResourceError::UnknownKvGeometry {
                tokens_per_page: kv_config.tokens_per_page,
            })?;
    if total_pages == 0 {
        return Err(ResourceError::CannotSatisfyLoweredLimit {
            requested_bytes: resolved_vram_bytes,
            minimum_bytes,
            reason: format!(
                "the derived KV budget of {kv_bytes} B cannot hold one {page_size_bytes} B KV page"
            ),
        });
    }
    let max_total_tokens = kv_config.tokens_for_pages(total_pages).ok_or_else(|| {
        ResourceError::BudgetArithmeticOverflow {
            operation: "multiplying KV pages by tokens per page",
            reason: format!(
                "{total_pages} pages at {} tokens per page exceed the representable token range",
                kv_config.tokens_per_page
            ),
        }
    })?;
    Ok(DerivedBudget {
        kv_bytes,
        total_pages,
        max_total_tokens,
        reserved_bytes,
        reservation_applied,
    })
}

#[derive(Debug)]
struct GovernorState {
    configured_limits: ResourceLimits,
    resolved_limits: ResolvedLimits,
    derived_budget: DerivedBudget,
}

/// The hot tier the KV byte budget is sized against.
///
/// When a device capacity was measured (`vram_bytes == Some`), the device
/// (VRAM) ceiling bounds the hot-tier KV budget, exactly as before. When no
/// device could be measured (`vram_bytes == None`, e.g. an ORT/CPU load or a
/// machine with no NVIDIA GPU), the KV cache physically lives in host RAM, so
/// the *measured* host-RAM ceiling bounds it instead of the fabricated 8 GiB
/// device constant that #947 removed.
///
/// This is deliberately **not** device/host authority aliasing (the deferred
/// UMA work): the device authority stays separately unknown and inert, and no
/// device lease is charged against host memory. This only picks the physically
/// correct ceiling for the KV byte budget so a device-less machine can still
/// size a working KV cache from a real number rather than a manufactured one.
fn kv_hot_tier_ceiling(resolved: &ResolvedLimits) -> Option<u64> {
    resolved.vram_bytes.or(Some(resolved.host_ram_bytes))
}

/// Per-device resource governor driving the shared hot-tier [`ByteBudget`].
pub struct ResourceGovernor {
    capacities: CapacityProviders,
    breakdown: VramBreakdown,
    kv_config: ModelKvConfig,
    byte_budget: ByteBudget,
    state: Mutex<GovernorState>,
}

impl ResourceGovernor {
    pub fn new(
        limits: ResourceLimits,
        capacities: CapacityProviders,
        breakdown: VramBreakdown,
        kv_config: ModelKvConfig,
    ) -> Result<Self, ResourceError> {
        let resolved_limits = resolve_limits(&limits, &capacities)?;
        let derived_budget = derive_kv_budget(
            kv_hot_tier_ceiling(&resolved_limits),
            &breakdown,
            &kv_config,
        )?;
        Ok(Self {
            capacities,
            breakdown,
            kv_config,
            byte_budget: ByteBudget::new(derived_budget.kv_bytes),
            state: Mutex::new(GovernorState {
                configured_limits: limits,
                resolved_limits,
                derived_budget,
            }),
        })
    }

    /// Shared hot-tier byte budget to pass to schedulers.
    pub fn byte_budget(&self) -> ByteBudget {
        self.byte_budget.clone()
    }

    /// Atomically replace all configured limits and report required engine eviction.
    pub fn reconfigure(
        &self,
        limits: ResourceLimits,
    ) -> Result<GovernorReconfigureOutcome, ResourceError> {
        let mut state = self.lock_state();
        self.reconfigure_locked(&mut state, limits)
    }

    fn reconfigure_locked(
        &self,
        state: &mut GovernorState,
        limits: ResourceLimits,
    ) -> Result<GovernorReconfigureOutcome, ResourceError> {
        let new_limits = resolve_limits(&limits, &self.capacities)?;
        let derived_budget = derive_kv_budget(
            kv_hot_tier_ceiling(&new_limits),
            &self.breakdown,
            &self.kv_config,
        )?;

        // All fallible validation precedes mutation, so an impossible target leaves
        // both governor state and ByteBudget unchanged.
        let old_limits = state.resolved_limits;
        let byte_budget = self.byte_budget.reconfigure(derived_budget.kv_bytes);
        state.configured_limits = limits;
        state.resolved_limits = new_limits;
        state.derived_budget = derived_budget;

        Ok(GovernorReconfigureOutcome {
            old_limits,
            new_limits,
            derived_budget,
            byte_budget,
            overage_bytes: byte_budget.overage,
            eviction_order: if byte_budget.overage == 0 {
                Vec::new()
            } else {
                EVICTION_ORDER.to_vec()
            },
        })
    }

    pub fn set_vram_limit(
        &self,
        limit: ResourceLimit,
    ) -> Result<GovernorReconfigureOutcome, ResourceError> {
        let mut state = self.lock_state();
        let mut limits = state.configured_limits.clone();
        limits.vram_limit = limit;
        self.reconfigure_locked(&mut state, limits)
    }

    pub fn set_host_ram_limit(
        &self,
        limit: ResourceLimit,
    ) -> Result<GovernorReconfigureOutcome, ResourceError> {
        let mut state = self.lock_state();
        let mut limits = state.configured_limits.clone();
        limits.host_ram_limit = limit;
        self.reconfigure_locked(&mut state, limits)
    }

    pub fn set_disk_spill_limit(
        &self,
        limit: Option<ResourceLimit>,
    ) -> Result<GovernorReconfigureOutcome, ResourceError> {
        let mut state = self.lock_state();
        let mut limits = state.configured_limits.clone();
        limits.disk_spill_limit = limit;
        self.reconfigure_locked(&mut state, limits)
    }

    pub fn snapshot(&self) -> GovernorSnapshot {
        let state = self.lock_state();
        let vram_budget = self.byte_budget.snapshot();
        GovernorSnapshot {
            configured_limits: state.configured_limits.clone(),
            resolved_limits: state.resolved_limits,
            derived_budget: state.derived_budget,
            breakdown: self.breakdown,
            vram: TierSnapshot::new(vram_budget.used, vram_budget.limit),
            host_ram: capacity_snapshot(
                self.capacities.host_ram.as_ref(),
                state.resolved_limits.host_ram_bytes,
            ),
            disk_spill: self.capacities.disk_spill.as_deref().and_then(|capacity| {
                state
                    .resolved_limits
                    .disk_spill_bytes
                    .map(|limit| capacity_snapshot(capacity, limit))
            }),
        }
    }

    fn lock_state(&self) -> MutexGuard<'_, GovernorState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

fn capacity_snapshot(capacity: &dyn CapacityProvider, limit: u64) -> TierSnapshot {
    let used = match (capacity.total_bytes(), capacity.free_bytes()) {
        (Some(total), Some(free)) => total.saturating_sub(free.min(total)),
        // Usage is unknown when the tier capacity cannot be measured; report
        // zero used rather than inventing a figure from a missing total.
        _ => 0,
    };
    TierSnapshot::new(used, limit)
}

fn resolve_limits(
    limits: &ResourceLimits,
    capacities: &CapacityProviders,
) -> Result<ResolvedLimits, ResourceError> {
    let disk_spill_bytes = match limits.disk_spill_limit {
        None => None,
        Some(limit) => {
            let capacity = capacities
                .disk_spill
                .as_deref()
                .ok_or(ResourceError::MissingDiskCapacityProvider)?;
            // Disk spill is opt-in; a fraction of an unmeasured filesystem is
            // still unmeasurable, so refuse rather than fabricate.
            Some(
                resolve_limit(limit, capacity, "disk spill")?
                    .ok_or(ResourceError::UnmeasuredCapacity { tier: "disk spill" })?,
            )
        }
    };

    Ok(ResolvedLimits {
        // VRAM may legitimately be unknown (no device to query); the device KV
        // budget derivation treats `None` as "no device ceiling".
        vram_bytes: resolve_limit(limits.vram_limit, capacities.vram.as_ref(), "vram")?,
        host_ram_bytes: resolve_limit(
            limits.host_ram_limit,
            capacities.host_ram.as_ref(),
            "host RAM",
        )?
        .ok_or(ResourceError::UnmeasuredCapacity { tier: "host RAM" })?,
        disk_spill_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capacities() -> CapacityProviders {
        CapacityProviders {
            vram: Arc::new(FixedCapacity::new(1_000, 800)),
            host_ram: Arc::new(FixedCapacity::new(4_000, 3_000)),
            disk_spill: Some(Arc::new(FixedCapacity::new(10_000, 9_000))),
        }
    }

    fn breakdown() -> VramBreakdown {
        VramBreakdown {
            model_weights_bytes: 100,
            activations_bytes: Some(50),
            ort_overhead_bytes: Some(50),
        }
    }

    fn kv_config() -> ModelKvConfig {
        ModelKvConfig::known(10, 16)
    }

    fn governor(vram_bytes: u64) -> ResourceGovernor {
        ResourceGovernor::new(
            ResourceLimits {
                vram_limit: ResourceLimit::Bytes(vram_bytes),
                host_ram_limit: ResourceLimit::Bytes(1_000),
                disk_spill_limit: None,
            },
            capacities(),
            breakdown(),
            kv_config(),
        )
        .unwrap()
    }

    #[test]
    fn default_limits_match_design() {
        assert_eq!(
            ResourceLimits::default(),
            ResourceLimits {
                vram_limit: ResourceLimit::Fraction(0.90),
                host_ram_limit: ResourceLimit::Fraction(0.25),
                disk_spill_limit: None,
            }
        );
    }

    #[test]
    fn resolves_bytes_fraction_and_auto_against_total_capacity() {
        let capacity = FixedCapacity::new(1_000, 100);
        // An explicit byte limit is authoritative, not clamped to the reported
        // capacity: clamping would silently discard the caller's only way to
        // state the real budget.
        assert_eq!(
            resolve_limit(ResourceLimit::Bytes(2_000), &capacity, "vram").unwrap(),
            Some(2_000)
        );
        assert_eq!(
            resolve_limit(ResourceLimit::Fraction(0.5), &capacity, "vram").unwrap(),
            Some(500)
        );
        assert_eq!(
            resolve_limit(ResourceLimit::Auto, &capacity, "vram").unwrap(),
            Some(900)
        );
        assert_eq!(
            resolve_limit(ResourceLimit::Auto, &capacity, "host RAM").unwrap(),
            Some(250)
        );
    }

    #[test]
    fn a_fraction_of_unmeasured_capacity_is_unknown_not_a_number() {
        // The whole point of #947: a fraction/auto of a capacity that could not
        // be measured must resolve to `None` (unknown), never to a specific
        // number derived from a fabricated constant. An explicit byte limit is
        // still honoured because it is the caller's own assertion.
        let capacity = UnknownCapacity;
        assert_eq!(
            resolve_limit(ResourceLimit::Fraction(0.90), &capacity, "vram").unwrap(),
            None
        );
        assert_eq!(
            resolve_limit(ResourceLimit::Auto, &capacity, "vram").unwrap(),
            None
        );
        assert_eq!(
            resolve_limit(ResourceLimit::Bytes(1_234), &capacity, "vram").unwrap(),
            Some(1_234)
        );
    }

    #[test]
    fn unknown_device_capacity_yields_no_device_kv_budget_without_advice() {
        // `None` VRAM means there is no device ceiling to fit the reservation
        // under, so the device KV budget is zero and nothing is "applied" —
        // there is no fabricated ceiling and no unactionable raise-the-limit path.
        let derived = derive_kv_budget(None, &breakdown(), &kv_config()).unwrap();
        assert_eq!(derived.kv_bytes, 0);
        assert_eq!(derived.total_pages, 0);
        assert_eq!(derived.max_total_tokens, 0);
        assert!(!derived.reservation_applied);
        assert_eq!(derived.reserved_bytes, 200);
    }

    #[test]
    fn rejects_invalid_fraction() {
        let capacity = FixedCapacity::new(1_000, 1_000);
        assert!(matches!(
            resolve_limit(ResourceLimit::Fraction(1.1), &capacity, "vram"),
            Err(ResourceError::InvalidFraction { .. })
        ));
        assert!(matches!(
            resolve_limit(ResourceLimit::Fraction(f32::NAN), &capacity, "vram"),
            Err(ResourceError::InvalidFraction { .. })
        ));
    }

    #[test]
    fn derives_kv_pages_and_tokens_after_fixed_reservations() {
        let derived = derive_kv_budget(Some(1_000), &breakdown(), &kv_config()).unwrap();
        assert_eq!(
            derived,
            DerivedBudget {
                kv_bytes: 800,
                total_pages: 80,
                max_total_tokens: 1_280,
                reserved_bytes: 200,
                reservation_applied: true,
            }
        );
    }

    #[test]
    fn unknown_kv_geometry_does_not_synthesize_one_byte_tokens() {
        let kv_config = ModelKvConfig::unknown(16);
        assert_eq!(kv_config.page_size_bytes, None);
        assert_eq!(kv_config.bytes_per_token(), None);
        assert_eq!(kv_config.pages_for_bytes(256), None);

        let error = derive_kv_budget(Some(1_000), &breakdown(), &kv_config).unwrap_err();
        assert!(matches!(
            error,
            ResourceError::UnknownKvGeometry {
                tokens_per_page: 16
            }
        ));
        let message = error.to_string();
        assert!(message.contains("per-layer KV page geometry is unknown"));
        assert!(message.contains("model.io.kv_inputs"));
        assert!(message.contains("model.io.kv_outputs"));
    }

    #[test]
    fn a_ceiling_below_the_fixed_reservation_drops_it_rather_than_failing() {
        // The reservation is an estimate carved from a ceiling that may itself
        // be provisional, so it must never be the reason a model cannot start:
        // the previous behaviour reserved nothing at all.
        let derived = derive_kv_budget(Some(150), &breakdown(), &kv_config()).unwrap();

        assert!(!derived.reservation_applied);
        assert_eq!(derived.reserved_bytes, 0);
        assert_eq!(derived.kv_bytes, 150);
        assert!(derived.total_pages >= 1);
    }

    #[test]
    fn derive_rejects_budget_too_small_for_one_page() {
        // With no fixed reservation to drop, a ceiling under one page is a
        // genuine configuration error and still fails.
        let no_reservation = VramBreakdown {
            model_weights_bytes: 0,
            activations_bytes: Some(0),
            ort_overhead_bytes: Some(0),
        };

        let error = derive_kv_budget(Some(5), &no_reservation, &kv_config()).unwrap_err();

        assert!(matches!(
            error,
            ResourceError::CannotSatisfyLoweredLimit {
                requested_bytes: 5,
                ..
            }
        ));
    }

    #[test]
    fn derive_accepts_ceiling_exactly_large_enough_for_one_page() {
        let derived = derive_kv_budget(Some(210), &breakdown(), &kv_config()).unwrap();
        assert_eq!(
            derived,
            DerivedBudget {
                kv_bytes: 10,
                total_pages: 1,
                max_total_tokens: 16,
                reserved_bytes: 200,
                reservation_applied: true,
            }
        );
    }

    #[test]
    fn derive_rejects_overflowing_fixed_reservations() {
        let breakdown = VramBreakdown {
            model_weights_bytes: u64::MAX,
            activations_bytes: Some(1),
            ort_overhead_bytes: Some(0),
        };

        let error = derive_kv_budget(Some(u64::MAX), &breakdown, &kv_config()).unwrap_err();
        assert!(matches!(
            error,
            ResourceError::BudgetArithmeticOverflow { .. }
        ));
        assert!(error.to_string().contains("fixed VRAM reservations"));
    }

    #[test]
    fn lower_below_usage_reports_overage_and_engine_eviction_order() {
        let governor = governor(1_000);
        governor.byte_budget().try_reserve(700).unwrap();

        let outcome = governor.set_vram_limit(ResourceLimit::Bytes(600)).unwrap();
        assert_eq!(outcome.derived_budget.kv_bytes, 400);
        assert_eq!(outcome.overage_bytes, 300);
        assert_eq!(outcome.byte_budget.overage, 300);
        assert_eq!(outcome.eviction_order, EVICTION_ORDER);
        assert_eq!(governor.snapshot().vram.limit, 400);
        assert_eq!(governor.snapshot().vram.used, 700);
    }

    #[test]
    fn impossible_lowering_is_atomic_and_preserves_previous_ceiling() {
        let governor = governor(1_000);
        governor.byte_budget().try_reserve(300).unwrap();
        let before = governor.snapshot();

        // Below one KV page: impossible even after the fixed reservation is
        // dropped, so this still fails atomically.
        let error = governor
            .set_vram_limit(ResourceLimit::Bytes(5))
            .unwrap_err();
        assert!(matches!(
            error,
            ResourceError::CannotSatisfyLoweredLimit { .. }
        ));
        assert_eq!(governor.snapshot(), before);
    }

    #[test]
    fn overflowing_max_ceiling_is_atomic_and_preserves_previous_budget() {
        let capacities = CapacityProviders {
            vram: Arc::new(FixedCapacity::new(u64::MAX, u64::MAX)),
            host_ram: Arc::new(FixedCapacity::new(4_000, 3_000)),
            disk_spill: None,
        };
        let governor = ResourceGovernor::new(
            ResourceLimits {
                vram_limit: ResourceLimit::Bytes(1_000),
                host_ram_limit: ResourceLimit::Bytes(1_000),
                disk_spill_limit: None,
            },
            capacities,
            breakdown(),
            kv_config(),
        )
        .unwrap();
        governor.byte_budget().try_reserve(300).unwrap();
        let before = governor.snapshot();

        let error = governor
            .set_vram_limit(ResourceLimit::Bytes(u64::MAX))
            .unwrap_err();
        assert!(matches!(
            error,
            ResourceError::BudgetArithmeticOverflow { .. }
        ));
        assert_eq!(governor.snapshot(), before);
    }

    #[test]
    fn raising_limit_increases_hot_tier_budget() {
        let governor = governor(600);
        governor.byte_budget().try_reserve(350).unwrap();
        assert!(governor.byte_budget().try_reserve(100).is_err());

        let outcome = governor
            .set_vram_limit(ResourceLimit::Bytes(1_000))
            .unwrap();
        assert_eq!(outcome.byte_budget.old_limit, 400);
        assert_eq!(outcome.byte_budget.new_limit, 800);
        assert_eq!(outcome.overage_bytes, 0);
        assert!(outcome.eviction_order.is_empty());
        governor.byte_budget().try_reserve(100).unwrap();
    }

    #[test]
    fn disk_limit_requires_an_injected_capacity_provider() {
        let mut capacities = capacities();
        capacities.disk_spill = None;
        let error = ResourceGovernor::new(
            ResourceLimits {
                disk_spill_limit: Some(ResourceLimit::Auto),
                ..ResourceLimits::default()
            },
            capacities,
            breakdown(),
            kv_config(),
        )
        .err()
        .unwrap();
        assert_eq!(error, ResourceError::MissingDiskCapacityProvider);
    }
}

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
pub trait CapacityProvider: Send + Sync {
    fn total_bytes(&self) -> u64;
    fn free_bytes(&self) -> u64;
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
    fn total_bytes(&self) -> u64 {
        self.total
    }

    fn free_bytes(&self) -> u64 {
        self.free
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
    pub activations_bytes: u64,
    pub ort_overhead_bytes: u64,
}

impl VramBreakdown {
    fn reserved_bytes(self) -> Option<u64> {
        self.model_weights_bytes
            .checked_add(self.activations_bytes)?
            .checked_add(self.ort_overhead_bytes)
    }
}

/// Model-specific mapping between KV bytes, pages, and tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelKvConfig {
    pub page_size_bytes: u64,
    pub tokens_per_page: u64,
}

impl ModelKvConfig {
    pub fn pages_for_bytes(&self, bytes: u64) -> u64 {
        if self.page_size_bytes == 0 {
            return 0;
        }
        bytes / self.page_size_bytes
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
}

/// Concrete per-tier ceilings after capacity resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedLimits {
    pub vram_bytes: u64,
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
pub fn resolve_limit(
    limit: ResourceLimit,
    capacity: &dyn CapacityProvider,
    tier: &'static str,
) -> Result<u64, ResourceError> {
    let default_fraction = match tier {
        "vram" => DEFAULT_VRAM_FRACTION,
        "host RAM" => DEFAULT_HOST_RAM_FRACTION,
        "disk spill" => DEFAULT_DISK_FRACTION,
        _ => DEFAULT_DISK_FRACTION,
    };

    match limit {
        ResourceLimit::Bytes(bytes) => Ok(bytes.min(capacity.total_bytes())),
        ResourceLimit::Auto => Ok(resolve_fraction(default_fraction, capacity.total_bytes())),
        ResourceLimit::Fraction(fraction)
            if fraction.is_finite() && (0.0..=1.0).contains(&fraction) =>
        {
            Ok(resolve_fraction(fraction, capacity.total_bytes()))
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
pub fn derive_kv_budget(
    resolved_vram_bytes: u64,
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
    let minimum_bytes = reserved_bytes
        .checked_add(kv_config.page_size_bytes)
        .ok_or_else(|| ResourceError::BudgetArithmeticOverflow {
            operation: "adding one KV page to the fixed VRAM reservations",
            reason: "even the one-page minimum exceeds the representable byte range".into(),
        })?;
    if kv_config.page_size_bytes == 0 || resolved_vram_bytes < minimum_bytes {
        let reason = if resolved_vram_bytes < reserved_bytes {
            format!(
                "fixed model weights, activations, and runtime overhead reserve {reserved_bytes} B"
            )
        } else if kv_config.page_size_bytes == 0 {
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
            format!(
                "the remaining {} B cannot hold one {} B KV page",
                remaining_bytes, kv_config.page_size_bytes
            )
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
    let total_pages = kv_config.pages_for_bytes(kv_bytes);
    if total_pages == 0 {
        return Err(ResourceError::CannotSatisfyLoweredLimit {
            requested_bytes: resolved_vram_bytes,
            minimum_bytes,
            reason: format!(
                "the derived KV budget of {kv_bytes} B cannot hold one {} B KV page",
                kv_config.page_size_bytes
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
    })
}

#[derive(Debug)]
struct GovernorState {
    configured_limits: ResourceLimits,
    resolved_limits: ResolvedLimits,
    derived_budget: DerivedBudget,
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
        let derived_budget = derive_kv_budget(resolved_limits.vram_bytes, &breakdown, &kv_config)?;
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
        let derived_budget =
            derive_kv_budget(new_limits.vram_bytes, &self.breakdown, &self.kv_config)?;

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
    let total = capacity.total_bytes();
    let used = total.saturating_sub(capacity.free_bytes().min(total));
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
            Some(resolve_limit(limit, capacity, "disk spill")?)
        }
    };

    Ok(ResolvedLimits {
        vram_bytes: resolve_limit(limits.vram_limit, capacities.vram.as_ref(), "vram")?,
        host_ram_bytes: resolve_limit(
            limits.host_ram_limit,
            capacities.host_ram.as_ref(),
            "host RAM",
        )?,
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
            activations_bytes: 50,
            ort_overhead_bytes: 50,
        }
    }

    fn kv_config() -> ModelKvConfig {
        ModelKvConfig {
            page_size_bytes: 10,
            tokens_per_page: 16,
        }
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
        assert_eq!(
            resolve_limit(ResourceLimit::Bytes(2_000), &capacity, "vram").unwrap(),
            1_000
        );
        assert_eq!(
            resolve_limit(ResourceLimit::Fraction(0.5), &capacity, "vram").unwrap(),
            500
        );
        assert_eq!(
            resolve_limit(ResourceLimit::Auto, &capacity, "vram").unwrap(),
            900
        );
        assert_eq!(
            resolve_limit(ResourceLimit::Auto, &capacity, "host RAM").unwrap(),
            250
        );
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
        let derived = derive_kv_budget(1_000, &breakdown(), &kv_config()).unwrap();
        assert_eq!(
            derived,
            DerivedBudget {
                kv_bytes: 800,
                total_pages: 80,
                max_total_tokens: 1_280,
                reserved_bytes: 200,
            }
        );
    }

    #[test]
    fn derive_rejects_ceiling_below_weights_and_overhead() {
        let error = derive_kv_budget(150, &breakdown(), &kv_config()).unwrap_err();
        assert!(matches!(
            error,
            ResourceError::CannotSatisfyLoweredLimit {
                requested_bytes: 150,
                minimum_bytes: 210,
                ..
            }
        ));
        assert!(error.to_string().contains("raise the limit"));
    }

    #[test]
    fn derive_rejects_budget_too_small_for_one_page() {
        let error = derive_kv_budget(205, &breakdown(), &kv_config()).unwrap_err();
        assert!(matches!(
            error,
            ResourceError::CannotSatisfyLoweredLimit {
                requested_bytes: 205,
                minimum_bytes: 210,
                ..
            }
        ));
    }

    #[test]
    fn derive_accepts_ceiling_exactly_large_enough_for_one_page() {
        let derived = derive_kv_budget(210, &breakdown(), &kv_config()).unwrap();
        assert_eq!(
            derived,
            DerivedBudget {
                kv_bytes: 10,
                total_pages: 1,
                max_total_tokens: 16,
                reserved_bytes: 200,
            }
        );
    }

    #[test]
    fn derive_rejects_overflowing_fixed_reservations() {
        let breakdown = VramBreakdown {
            model_weights_bytes: u64::MAX,
            activations_bytes: 1,
            ort_overhead_bytes: 0,
        };

        let error = derive_kv_budget(u64::MAX, &breakdown, &kv_config()).unwrap_err();
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

        let error = governor
            .set_vram_limit(ResourceLimit::Bytes(205))
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

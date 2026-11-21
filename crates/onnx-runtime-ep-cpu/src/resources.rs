//! Immutable memory resources supplied before the CPU registry is constructed.

use std::sync::{Arc, Mutex, Weak};

use onnx_runtime_ep_api::{EpError, Result};
use onnx_runtime_memory_api::{MemoryError, MemoryRole, Tier};
use onnx_runtime_memory_governor::{
    HolderId, MemoryGovernor, MemoryLease, PressureRegistration, PressureResponder,
};

use crate::CpuCacheRetention;
use crate::kernels::qmoe::WeightOffloadHostCache;

/// Retention ceilings, separate from the Host grants that fund actual payloads.
#[derive(Clone, Copy, Debug)]
pub struct CpuWeightCacheLimits {
    /// Per-kernel f32 dense expansion retention ceiling (block formats and NBits).
    pub dense_bytes: usize,
    /// Shared QMoE expert retention ceiling.
    pub expert_bytes: u64,
}

impl Default for CpuWeightCacheLimits {
    fn default() -> Self {
        Self {
            dense_bytes: crate::kernels::dense_weight_cache::configured_dense_weight_cache_bytes(),
            expert_bytes: 0,
        }
    }
}

/// Immutable CPU weight resources. Full f32 block-format/NBits expansions and retained
/// or transient QMoE experts use one Host grant source. Private fields
/// prevent a shared expert cache from being paired with a different authority.
/// Cache ceilings and the legacy planned-family verdict only admit retention;
/// they reserve no idle physical capacity.
///
/// ```
/// use onnx_runtime_ep_cpu::{CpuExecutionProvider, CpuMemoryResources, CpuWeightCacheLimits, DerivedWeightMemory};
/// use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let governor = LedgerGovernor::new(LeaseLedger::new(0, 1024 * 1024, 0));
/// let resources = CpuMemoryResources::new(
///     DerivedWeightMemory::governed(&governor, HolderId::new(13))?,
///     CpuWeightCacheLimits { dense_bytes: 64 * 1024, expert_bytes: 512 * 1024 },
/// )?;
/// let provider = CpuExecutionProvider::with_resources(resources);
/// # let _ = provider;
/// # Ok(())
/// # }
/// ```
///
/// A caller cannot replace only the source of an already constructed cache:
/// ```compile_fail
/// use onnx_runtime_ep_cpu::CpuMemoryResources;
/// let mut resources = CpuMemoryResources::default();
/// resources.host_cache = CpuMemoryResources::default().host_cache;
/// ```
#[derive(Clone)]
pub struct CpuMemoryResources {
    host_cache: WeightOffloadHostCache,
    dense_bytes: usize,
    cache_retention: CpuCacheRetention,
}

impl Default for CpuMemoryResources {
    fn default() -> Self {
        Self::new(
            DerivedWeightMemory::default(),
            CpuWeightCacheLimits::default(),
        )
        .expect("the default zero expert-cache retention ceiling is addressable")
    }
}

impl CpuMemoryResources {
    pub fn new(
        memory: DerivedWeightMemory,
        limits: CpuWeightCacheLimits,
    ) -> std::result::Result<Self, MemoryError> {
        let host_cache =
            WeightOffloadHostCache::new(limits.expert_bytes, memory).map_err(|reason| {
                MemoryError::InvalidRequest {
                    tier: Tier::Host.name(),
                    requested: limits.expert_bytes,
                    reason,
                }
            })?;
        Ok(Self {
            host_cache,
            dense_bytes: limits.dense_bytes,
            cache_retention: CpuCacheRetention::default(),
        })
    }

    /// Capture the legacy planned-family retention verdict before registry construction.
    pub fn with_cache_retention(mut self, retention: CpuCacheRetention) -> Self {
        self.cache_retention = retention;
        self
    }

    pub fn cache_retention(&self) -> &CpuCacheRetention {
        &self.cache_retention
    }
    /// Configured limits; an optional expert environment override resolves at first use.
    pub fn weight_cache_limits(&self) -> CpuWeightCacheLimits {
        CpuWeightCacheLimits {
            dense_bytes: self.dense_bytes,
            expert_bytes: self.host_cache.retention_budget_bytes(),
        }
    }
    pub(crate) fn expert_cache(&self) -> &WeightOffloadHostCache {
        &self.host_cache
    }
    /// Whether block/NBits/QMoE full payloads obtain actual Host grants.
    pub fn is_governed(&self) -> bool {
        self.host_cache.memory().is_governed()
    }
    pub(crate) fn weight_memory(&self) -> &DerivedWeightMemory {
        self.host_cache.memory()
    }
}

/// Grant source for owned f32 copies derived from packed weights.
///
/// A governed source retains a zero-byte anchor, rather than a standing cache
/// allowance. Each owned expansion claims its payload bytes against the same
/// Host books as other holders, and keeps them until its last owner drops.
/// Clones share weak idle-cache registrations, enrolled with the supplying
/// authority's optional pressure capability. Any Host claimant can then reclaim
/// inactive entries; actual grants, growth and retries remain authority-owned.
/// Allocator/configuration errors and warm hits do not trigger CPU eviction.
#[derive(Clone, Default)]
pub struct DerivedWeightMemory {
    owner: Option<Arc<DerivedWeightSource>>,
}

struct DerivedWeightSource {
    domain: Arc<DerivedWeightDomain>,
    _registration: Option<PressureRegistration>,
}

struct DerivedWeightDomain {
    anchor: MemoryLease,
    idle_caches: Mutex<IdleCacheRegistry>,
}

/// A cache may remove only entries whose last owner is the cache itself.
/// Implementations drop payloads after releasing their data lock, so returning
/// their Host grants can safely call an external accountant.
pub(crate) trait IdleWeightCache: Send + Sync {
    fn reclaim_idle(&self, want: usize) -> Result<usize>;
}

#[derive(Default)]
struct IdleCacheRegistry {
    caches: Vec<Weak<dyn IdleWeightCache>>,
    next: usize,
}

impl IdleCacheRegistry {
    fn snapshot(&mut self) -> Vec<Arc<dyn IdleWeightCache>> {
        self.caches.retain(|cache| cache.strong_count() != 0);
        let mut caches: Vec<_> = self.caches.iter().filter_map(Weak::upgrade).collect();
        if !caches.is_empty() {
            let first = self.next % caches.len();
            caches.rotate_left(first);
            self.next = (first + 1) % caches.len();
        }
        caches
    }
}

impl DerivedWeightMemory {
    pub fn governed(
        governor: &dyn MemoryGovernor,
        holder: HolderId,
    ) -> std::result::Result<Self, MemoryError> {
        let anchor = governor.reserve(Tier::Host, 0, MemoryRole::Weights, holder)?;
        if anchor.tier() != Tier::Host
            || anchor.role() != MemoryRole::Weights
            || anchor.bytes() != 0
            || anchor.holder() != holder
        {
            return Err(MemoryError::InvalidRequest {
                tier: Tier::Host.name(),
                requested: 0,
                reason: "derived-weight governor returned a different tier, role, holder or nonzero anchor; implement reserve with the requested Host/Weights zero-byte ownership",
            });
        }
        if anchor
            .authority_id()
            .is_some_and(|actual| actual != governor.authority_id(Tier::Host))
        {
            return Err(MemoryError::InvalidRequest {
                tier: Tier::Host.name(),
                requested: 0,
                reason: "derived-weight anchor uses different Host books than its governor; return reserve leases from the requested tier's authority",
            });
        }
        let domain = Arc::new(DerivedWeightDomain {
            anchor,
            idle_caches: Mutex::new(IdleCacheRegistry::default()),
        });
        let registration = if let Some(registry) = governor.pressure_registry(Tier::Host) {
            registry.validate(governor.authority_id(Tier::Host), Tier::Host)?;
            let responder: Arc<dyn PressureResponder> = domain.clone();
            Some(registry.register(&responder)?)
        } else {
            None
        };
        Ok(Self {
            owner: Some(Arc::new(DerivedWeightSource {
                domain,
                _registration: registration,
            })),
        })
    }

    pub fn is_governed(&self) -> bool {
        self.owner.is_some()
    }

    /// Whether the supplying authority can reclaim these idle weights for any
    /// Host claimant, including KV lease growth. Third-party authorities without
    /// a pressure capability still fund payloads, but cannot reclaim them here.
    pub fn supports_pressure_reclaim(&self) -> bool {
        self.owner
            .as_ref()
            .is_some_and(|owner| owner._registration.is_some())
    }

    pub(crate) fn register_idle_cache<T: IdleWeightCache + 'static>(&self, cache: &Arc<T>) {
        let Some(owner) = &self.owner else {
            return;
        };
        let cache: Arc<dyn IdleWeightCache> = cache.clone();
        let mut registry = owner
            .domain
            .idle_caches
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        registry.caches.retain(|cache| cache.strong_count() != 0);
        registry.caches.push(Arc::downgrade(&cache));
    }

    pub(crate) fn try_grant(&self, bytes: usize) -> Result<WeightAdmission> {
        let Some(owner) = &self.owner else {
            return Ok(WeightAdmission::Granted(DenseWeightGrant::Local));
        };
        let requested = u64::try_from(bytes).map_err(|_| EpError::KernelFailed(
            "derived dense weight byte count exceeds u64; use dimensions addressable by the Host memory governor".into()
        ))?;
        // Admission/retry belongs to the actual accounting authority, including
        // ordinary KV reserve/grow/sibling requests. Do not run a second CPU-only
        // pressure pass after that authority declines.
        match owner.domain.anchor.reserve_sibling(requested) {
            Ok(lease) => Ok(WeightAdmission::Granted(DenseWeightGrant::Governed(lease))),
            Err(
                failure @ (MemoryError::TierExhausted { .. }
                | MemoryError::CapacityUnavailable { .. }),
            ) => Ok(WeightAdmission::Declined(failure)),
            Err(failure) => Err(EpError::Memory(failure)),
        }
    }
}

impl PressureResponder for DerivedWeightDomain {
    fn on_pressure(&self, tier: Tier, want: u64) -> std::result::Result<u64, MemoryError> {
        if tier != Tier::Host || want == 0 {
            return Ok(0);
        }
        let caches = self
            .idle_caches
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .snapshot();
        let mut released = 0u64;
        for cache in caches {
            let remaining = usize::try_from(want.saturating_sub(released)).unwrap_or(usize::MAX);
            let bytes =
                cache
                    .reclaim_idle(remaining)
                    .map_err(|source| MemoryError::ReclaimFailed {
                        tier: tier.name(),
                        requested: want,
                        source: Box::new(source),
                    })?;
            released = released.saturating_add(bytes as u64);
            if released >= want {
                break;
            }
        }
        Ok(released)
    }
}

pub(crate) enum WeightAdmission {
    Granted(DenseWeightGrant),
    Declined(MemoryError),
}

pub(crate) enum DenseWeightGrant {
    Local,
    Governed(MemoryLease),
}

impl DenseWeightGrant {
    pub(crate) fn split_off(&mut self, bytes: usize) -> Result<Self> {
        match self {
            Self::Local => Ok(Self::Local),
            Self::Governed(lease) => {
                let bytes = u64::try_from(bytes).map_err(|_| EpError::KernelFailed(
                    "derived dense weight slice exceeds u64; use addressable projection dimensions".into()
                ))?;
                Ok(Self::Governed(lease.split_off(bytes)?))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_memory_governor::{
        LeaseAccounting, LeaseLedger, LedgerGovernor, MemoryAuthorityId,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Clone, Copy)]
    enum Response {
        Valid,
        Nonzero,
        WrongTier,
        WrongRole,
        WrongHolder,
        WrongBooks,
    }
    struct SubstitutingGovernor {
        expected: LedgerGovernor,
        foreign: LedgerGovernor,
        response: Response,
    }
    impl MemoryGovernor for SubstitutingGovernor {
        fn authority_id(&self, tier: Tier) -> MemoryAuthorityId {
            self.expected.authority_id(tier)
        }
        fn reserve(
            &self,
            tier: Tier,
            bytes: u64,
            role: MemoryRole,
            holder: HolderId,
        ) -> std::result::Result<MemoryLease, MemoryError> {
            match self.response {
                Response::Valid => self.expected.reserve(tier, bytes, role, holder),
                Response::Nonzero => self.expected.reserve(tier, 4, role, holder),
                Response::WrongTier => self.expected.reserve(Tier::Device, bytes, role, holder),
                Response::WrongRole => {
                    self.expected
                        .reserve(tier, bytes, MemoryRole::KvCache, holder)
                }
                Response::WrongHolder => {
                    self.expected.reserve(tier, bytes, role, HolderId::new(99))
                }
                Response::WrongBooks => self.foreign.reserve(tier, bytes, role, holder),
            }
        }
        fn available(&self, tier: Tier) -> u64 {
            self.expected.available(tier)
        }
        fn used(&self, tier: Tier) -> u64 {
            self.expected.used(tier)
        }
    }

    #[test]
    fn derived_source_rejects_and_refunds_miswired_anchor_ownership() {
        assert!(!DerivedWeightMemory::default().is_governed());
        for response in [
            Response::Valid,
            Response::Nonzero,
            Response::WrongTier,
            Response::WrongRole,
            Response::WrongHolder,
            Response::WrongBooks,
        ] {
            let governor = SubstitutingGovernor {
                expected: LedgerGovernor::new(LeaseLedger::new(16, 16, 0)),
                foreign: LedgerGovernor::new(LeaseLedger::new(16, 16, 0)),
                response,
            };
            let result = DerivedWeightMemory::governed(&governor, HolderId::new(13));
            match response {
                Response::Valid => assert!(result.unwrap().is_governed()),
                _ => {
                    let failure = result.err().expect("malformed anchor must be refused");
                    assert!(matches!(failure, MemoryError::InvalidRequest { .. }));
                    assert!(failure.to_string().contains("derived-weight"));
                }
            }
            assert_eq!(governor.expected.used(Tier::Host), 0);
            assert_eq!(governor.foreign.used(Tier::Host), 0);
            assert_eq!(governor.expected.used(Tier::Device), 0);
        }
    }

    struct IdleFixture {
        source: DerivedWeightMemory,
        grant: Mutex<Option<DenseWeightGrant>>,
        wants: Mutex<Vec<usize>>,
    }
    impl IdleWeightCache for IdleFixture {
        fn reclaim_idle(&self, want: usize) -> Result<usize> {
            // A callback may register another cache or claim already available
            // bytes: the shared registry must not be locked across callbacks.
            assert!(
                self.source
                    .owner
                    .as_ref()
                    .unwrap()
                    .domain
                    .idle_caches
                    .try_lock()
                    .is_ok()
            );
            assert!(matches!(
                self.source.try_grant(0)?,
                WeightAdmission::Granted(_)
            ));
            self.wants.lock().unwrap().push(want);
            let retired = self.grant.lock().unwrap().take();
            let bytes = match &retired {
                Some(DenseWeightGrant::Governed(lease)) => lease.bytes() as usize,
                _ => 0,
            };
            drop(retired);
            Ok(bytes)
        }
    }

    #[test]
    fn shared_pressure_requests_only_the_shortfall_and_does_not_retain_registered_caches() {
        let governor = LedgerGovernor::new(LeaseLedger::new(0, 32, 0));
        let source = DerivedWeightMemory::governed(&governor, HolderId::new(13)).unwrap();
        let idle = || {
            let WeightAdmission::Granted(grant) = source.try_grant(8).unwrap() else {
                panic!("fixture cache capacity");
            };
            let cache = Arc::new(IdleFixture {
                source: source.clone(),
                grant: Mutex::new(Some(grant)),
                wants: Mutex::new(Vec::new()),
            });
            source.register_idle_cache(&cache);
            cache
        };
        let first = idle();
        let second = idle();
        assert!(matches!(
            source.try_grant(33).unwrap(),
            WeightAdmission::Declined(_)
        ));
        assert!(first.wants.lock().unwrap().is_empty());
        assert!(second.wants.lock().unwrap().is_empty());
        let kv = governor
            .reserve(Tier::Host, 12, MemoryRole::KvCache, HolderId::new(1))
            .unwrap();
        let WeightAdmission::Granted(held) = source.try_grant(8).unwrap() else {
            panic!("one idle entry must fund this request");
        };
        assert_eq!(*first.wants.lock().unwrap(), vec![4]);
        assert!(second.wants.lock().unwrap().is_empty());
        assert_eq!(governor.used(Tier::Host), 28);
        let first_weak = Arc::downgrade(&first);
        let second_weak = Arc::downgrade(&second);
        drop(first);
        drop(second);
        assert!(first_weak.upgrade().is_none());
        assert!(second_weak.upgrade().is_none());
        assert_eq!(governor.used(Tier::Host), 20);
        // The next refused request prunes dead registrations and terminates.
        assert!(matches!(
            source.try_grant(16).unwrap(),
            WeightAdmission::Declined(_)
        ));
        assert!(
            source
                .owner
                .as_ref()
                .unwrap()
                .domain
                .idle_caches
                .lock()
                .unwrap()
                .caches
                .is_empty()
        );
        drop(held);
        drop(kv);
        assert_eq!(governor.used(Tier::Host), 0);
    }

    #[test]
    fn derived_source_validates_optional_pressure_capabilities_without_losing_grant_ownership() {
        struct View {
            books: LedgerGovernor,
            registry: Option<onnx_runtime_memory_governor::PressureRegistry>,
        }
        impl MemoryGovernor for View {
            fn authority_id(&self, tier: Tier) -> MemoryAuthorityId {
                self.books.authority_id(tier)
            }
            fn pressure_registry(
                &self,
                _tier: Tier,
            ) -> Option<onnx_runtime_memory_governor::PressureRegistry> {
                self.registry.clone()
            }
            fn reserve(
                &self,
                tier: Tier,
                bytes: u64,
                role: MemoryRole,
                holder: HolderId,
            ) -> std::result::Result<MemoryLease, MemoryError> {
                self.books.reserve(tier, bytes, role, holder)
            }
            fn available(&self, tier: Tier) -> u64 {
                self.books.available(tier)
            }
            fn used(&self, tier: Tier) -> u64 {
                self.books.used(tier)
            }
        }
        let books = LedgerGovernor::new(LeaseLedger::new(0, 16, 0));
        let foreign = LedgerGovernor::new(LeaseLedger::new(0, 16, 0));
        for registry in [
            books.pressure_registry(Tier::Device).unwrap(),
            foreign.pressure_registry(Tier::Host).unwrap(),
        ] {
            let view = View {
                books: books.clone(),
                registry: Some(registry),
            };
            let failure = DerivedWeightMemory::governed(&view, HolderId::new(13))
                .err()
                .expect("miswired pressure capability must be rejected");
            assert!(matches!(failure, MemoryError::InvalidRequest { .. }));
            assert!(failure.to_string().contains("pressure registry"));
            assert_eq!(books.used(Tier::Host), 0);
            assert_eq!(foreign.used(Tier::Host), 0);
        }
        let view = View {
            books: books.clone(),
            registry: None,
        };
        let source = DerivedWeightMemory::governed(&view, HolderId::new(13)).unwrap();
        assert!(source.is_governed());
        assert!(!source.supports_pressure_reclaim());
        let WeightAdmission::Granted(grant) = source.try_grant(8).unwrap() else {
            panic!(
                "unsupported pressure is not permission to reject an otherwise funded allocation"
            );
        };
        assert_eq!(books.used(Tier::Host), 8);
        drop(grant);
        assert_eq!(books.used(Tier::Host), 0);
    }

    #[test]
    fn idle_reclaim_hard_errors_preserve_the_cpu_cause_and_never_become_capacity_refusals() {
        use std::error::Error;
        struct Broken;
        impl IdleWeightCache for Broken {
            fn reclaim_idle(&self, _want: usize) -> Result<usize> {
                Err(EpError::KernelFailed(
                    "fixture idle cache failed; restore its accounting".into(),
                ))
            }
        }
        let governor = LedgerGovernor::new(LeaseLedger::new(0, 16, 0));
        let source = DerivedWeightMemory::governed(&governor, HolderId::new(13)).unwrap();
        let broken = Arc::new(Broken);
        source.register_idle_cache(&broken);
        let kv = governor
            .reserve(Tier::Host, 16, MemoryRole::KvCache, HolderId::new(2))
            .unwrap();
        match source.try_grant(4) {
            Err(EpError::Memory(failure)) => {
                assert!(matches!(
                    failure,
                    MemoryError::ReclaimFailed { requested: 4, .. }
                ));
                assert!(
                    failure
                        .source()
                        .unwrap()
                        .downcast_ref::<EpError>()
                        .is_some()
                );
                assert!(failure.to_string().contains("restore its accounting"));
            }
            _ => panic!("reclaimer failure was discarded or misclassified as capacity"),
        }
        assert_eq!(governor.used(Tier::Host), 16);
        drop(kv);
        assert_eq!(governor.used(Tier::Host), 0);
    }

    #[derive(Default)]
    struct EmptyIdleCache(AtomicUsize);
    impl IdleWeightCache for EmptyIdleCache {
        fn reclaim_idle(&self, _want: usize) -> Result<usize> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(0)
        }
    }

    #[derive(Debug, Clone, Copy)]
    enum FailureKind {
        Configuration,
        Allocator,
        Capacity,
    }

    #[derive(Debug)]
    struct FaultingBooks {
        identity: MemoryAuthorityId,
        kind: FailureKind,
        attempts: AtomicUsize,
    }
    impl LeaseAccounting for FaultingBooks {
        fn authority_id(&self) -> Option<MemoryAuthorityId> {
            Some(self.identity)
        }
        fn release(&self, _tier: Tier, _bytes: u64) {}
        fn try_claim(
            &self,
            tier: Tier,
            bytes: u64,
            _role: MemoryRole,
        ) -> std::result::Result<(), MemoryError> {
            self.attempts.fetch_add(1, Ordering::Relaxed);
            match self.kind {
                FailureKind::Allocator => Err(MemoryError::AllocationFailed { tier: tier.name(), requested: bytes, reason: "test Host allocator rejected storage; use the configured allocator".into() }),
                FailureKind::Configuration => Err(MemoryError::InvalidRequest { tier: tier.name(), requested: bytes, reason: "test Host authority rejected owner; use the owning authority" }),
                FailureKind::Capacity => Err(MemoryError::CapacityUnavailable { tier: tier.name(), requested: bytes, available: 0, role: MemoryRole::Weights, detail: "test capacity could not be reclaimed; release inactive weights or raise the Host limit".into(), source: None }),
            }
        }
    }
    struct FaultingGovernor(Arc<FaultingBooks>);
    impl MemoryGovernor for FaultingGovernor {
        fn authority_id(&self, _tier: Tier) -> MemoryAuthorityId {
            self.0.identity
        }
        fn reserve(
            &self,
            tier: Tier,
            bytes: u64,
            role: MemoryRole,
            holder: HolderId,
        ) -> std::result::Result<MemoryLease, MemoryError> {
            Ok(MemoryLease::new(tier, bytes, role, holder, self.0.clone()))
        }
        fn available(&self, _tier: Tier) -> u64 {
            16
        }
        fn used(&self, _tier: Tier) -> u64 {
            0
        }
    }
    #[test]
    fn grant_refusals_are_classified_by_capacity_while_allocator_and_configuration_errors_propagate()
     {
        for kind in [
            FailureKind::Configuration,
            FailureKind::Allocator,
            FailureKind::Capacity,
        ] {
            let books = Arc::new(FaultingBooks {
                identity: LedgerGovernor::new(LeaseLedger::new(0, 16, 0)).authority_id(Tier::Host),
                kind,
                attempts: AtomicUsize::new(0),
            });
            let source =
                DerivedWeightMemory::governed(&FaultingGovernor(books.clone()), HolderId::new(13))
                    .unwrap();
            let idle = Arc::new(EmptyIdleCache::default());
            source.register_idle_cache(&idle);
            assert!(!source.supports_pressure_reclaim());
            match (kind, source.try_grant(16)) {
                (
                    FailureKind::Allocator,
                    Err(EpError::Memory(MemoryError::AllocationFailed { requested: 16, .. })),
                ) => {}
                (
                    FailureKind::Configuration,
                    Err(EpError::Memory(MemoryError::InvalidRequest { requested: 16, .. })),
                ) => {}
                (
                    FailureKind::Capacity,
                    Ok(WeightAdmission::Declined(MemoryError::CapacityUnavailable {
                        requested: 16,
                        ..
                    })),
                ) => {}
                _ => panic!("memory failure was misclassified"),
            }
            assert_eq!(books.attempts.load(Ordering::Relaxed), 1);
            assert_eq!(
                idle.0.load(Ordering::Relaxed),
                0,
                "an authority without pressure support must not be followed by a CPU-only eviction pass"
            );
        }
    }
}

//! Immutable memory resources supplied before the CPU registry is constructed.

use std::sync::Arc;

use onnx_runtime_ep_api::{EpError, Result};
use onnx_runtime_memory_api::{MemoryError, MemoryRole, Tier};
use onnx_runtime_memory_governor::{HolderId, MemoryGovernor, MemoryLease};

use crate::{EinsumScratchRetention, WeightOffloadHostCache};

/// Session memory resources supplied before the CPU registry is constructed.
/// Block-quantized factories share `derived_weights`; QMoE uses `host_cache`
/// and Einsum uses its scratch-retention verdict. Other allocations are not
/// covered by the derived-weight grant source. The default is the standalone,
/// ungoverned policy; engines supply their Host authority explicitly.
///
/// ```
/// use onnx_runtime_ep_cpu::{CpuExecutionProvider, CpuMemoryResources, DerivedWeightMemory};
/// use onnx_runtime_memory_governor::{HolderId, LeaseLedger, LedgerGovernor};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let governor = LedgerGovernor::new(LeaseLedger::new(0, 1024 * 1024, 0));
/// let resources = CpuMemoryResources {
///     derived_weights: DerivedWeightMemory::governed(&governor, HolderId::new(13))?,
///     dense_weight_cache_bytes: 64 * 1024,
///     ..Default::default()
/// };
/// let provider = CpuExecutionProvider::with_resources(resources);
/// # let _ = provider;
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct CpuMemoryResources {
    pub host_cache: WeightOffloadHostCache,
    pub derived_weights: DerivedWeightMemory,
    /// Per-kernel retention/eligibility policy, separate from shared physical grants.
    pub dense_weight_cache_bytes: usize,
    pub einsum_scratch_retention: EinsumScratchRetention,
}

impl Default for CpuMemoryResources {
    fn default() -> Self {
        Self {
            host_cache: crate::kernels::qmoe::default_weight_offload_host_cache().clone(),
            derived_weights: DerivedWeightMemory::default(),
            dense_weight_cache_bytes:
                crate::kernels::dense_weight_cache::configured_dense_weight_cache_bytes(),
            einsum_scratch_retention: EinsumScratchRetention::default(),
        }
    }
}

/// Grant source for owned f32 copies derived from packed weights.
///
/// A governed source retains a zero-byte anchor, rather than a standing cache
/// allowance. Each live buffer claims its exact bytes against the same Host
/// books as other holders, and keeps those bytes until its last owner drops.
#[derive(Clone, Default)]
pub struct DerivedWeightMemory {
    anchor: Option<Arc<MemoryLease>>,
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
        Ok(Self {
            anchor: Some(Arc::new(anchor)),
        })
    }

    pub fn is_governed(&self) -> bool {
        self.anchor.is_some()
    }

    pub(crate) fn try_grant(&self, bytes: usize) -> Result<WeightAdmission> {
        let Some(anchor) = &self.anchor else {
            return Ok(WeightAdmission::Granted(DenseWeightGrant::Local));
        };
        let bytes = u64::try_from(bytes).map_err(|_| EpError::KernelFailed(
            "derived dense weight byte count exceeds u64; use dimensions addressable by the Host memory governor".into()
        ))?;
        match anchor.reserve_sibling(bytes) {
            Ok(lease) => Ok(WeightAdmission::Granted(DenseWeightGrant::Governed(lease))),
            Err(
                failure @ (MemoryError::TierExhausted { .. }
                | MemoryError::CapacityUnavailable { .. }),
            ) => Ok(WeightAdmission::Declined(failure)),
            Err(failure) => Err(EpError::Memory(failure)),
        }
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
        }
    }
}

use std::{fmt, sync::Arc};

use onnx_genai_scheduler::ResourceLimit;
use onnx_runtime_memory_governor::{
    DeviceKey, HolderId, LeaseLedger, LedgerGovernor, MappedAllowance, MappedGrowthGrant,
    MappedGrowthMetrics, MappedHolderRegistration, MemoryAuthorityId, MemoryError, MemoryGovernor,
    MemoryLease, MemoryRole, ProcessMemoryManager, ReclaimableMappedHolder, Tier,
};

/// Physical-device compatibility domain for a shared device memory authority.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DeviceCompatibilityDomain {
    Host,
    Cuda(u32),
    Accelerator { backend: String, index: u32 },
}

impl DeviceCompatibilityDomain {
    pub(crate) fn device_key(&self) -> DeviceKey {
        match self {
            Self::Host => DeviceKey::HOST,
            Self::Cuda(index) | Self::Accelerator { index, .. } => DeviceKey::device(*index),
        }
    }
}

impl fmt::Display for DeviceCompatibilityDomain {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Host => formatter.write_str("host"),
            Self::Cuda(index) => write!(formatter, "cuda:{index}"),
            Self::Accelerator { backend, index } => write!(formatter, "{backend}:{index}"),
        }
    }
}

/// One device-tier ledger and its stable authority identity.
#[derive(Debug, Clone)]
pub struct DeviceMemoryAuthority {
    domain: DeviceCompatibilityDomain,
    governor: LedgerGovernor,
}

impl DeviceMemoryAuthority {
    pub fn new(domain: DeviceCompatibilityDomain, limit_bytes: u64) -> Self {
        let ledger = LeaseLedger::new_for_device(domain.device_key(), limit_bytes, 0, 0);
        Self {
            domain,
            governor: LedgerGovernor::new(ledger),
        }
    }

    pub fn domain(&self) -> &DeviceCompatibilityDomain {
        &self.domain
    }

    pub fn authority_id(&self) -> MemoryAuthorityId {
        self.governor.authority_id()
    }

    pub fn limit_bytes(&self) -> u64 {
        self.governor.ledger().limit(Tier::Device)
    }

    pub fn used_bytes(&self) -> u64 {
        self.governor.used(Tier::Device)
    }

    pub fn headroom_bytes(&self) -> u64 {
        self.governor.available(Tier::Device)
    }

    pub fn growth_metrics(&self) -> MappedGrowthMetrics {
        self.governor.mapped_growth_metrics().unwrap_or_default()
    }

    pub fn pause_reconfiguration(&self) -> onnx_runtime_memory_governor::LeaseLimitGuard<'_> {
        self.governor.ledger().pause_claims(Tier::Device)
    }

    pub fn pause_mapped_growth(
        &self,
    ) -> Result<onnx_runtime_memory_governor::MappedGrowthOperationGuard, MemoryError> {
        self.governor.pause_mapped_growth()
    }

    pub fn trim_unmapped_bytes(&self, bytes: u64) -> anyhow::Result<u64> {
        #[cfg(feature = "native-cuda")]
        {
            onnx_runtime_ep_cuda::virtual_memory::trim_physical_handle_pools(
                self.authority_id(),
                bytes,
            )
            .map_err(anyhow::Error::new)
        }

        #[cfg(not(feature = "native-cuda"))]
        {
            let _ = bytes;
            Ok(0)
        }
    }

    pub fn releasable_unmapped_bytes(&self) -> u64 {
        #[cfg(feature = "native-cuda")]
        {
            onnx_runtime_ep_cuda::virtual_memory::pooled_unmapped_bytes_for_authority(
                self.authority_id(),
            )
        }

        #[cfg(not(feature = "native-cuda"))]
        {
            0
        }
    }

    #[cfg(feature = "native-cuda")]
    pub fn physical_pool_operation_gate(&self) -> std::sync::Arc<std::sync::RwLock<()>> {
        onnx_runtime_ep_cuda::virtual_memory::physical_pool_authority_gate(self.authority_id())
    }

    /// Lower the device limit after releasing any retained, unmapped CUDA
    /// handles. Mapped or otherwise leased bytes make the shrink fail without
    /// changing the old limit.
    pub fn try_set_limit_bytes(&self, bytes: u64) -> anyhow::Result<()> {
        let _mapped_growth = self.governor.pause_mapped_growth()?;
        let guard = self.pause_reconfiguration();
        #[cfg(feature = "native-cuda")]
        let pool_gate = self.physical_pool_operation_gate();
        #[cfg(feature = "native-cuda")]
        let _pool_operations = pool_gate
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let used = guard.used();
        if used > bytes {
            self.trim_unmapped_bytes(used - bytes)?;
        }
        if let Err(remaining) = guard.try_set_limit(bytes) {
            anyhow::bail!(
                "cannot satisfy lowered resource limit of {bytes} bytes: {} currently has \
                 {remaining} mapped or otherwise leased bytes",
                self.domain
            );
        }
        Ok(())
    }
}

impl MemoryGovernor for DeviceMemoryAuthority {
    fn authority_id(&self) -> MemoryAuthorityId {
        self.governor.authority_id()
    }

    fn reserve(
        &self,
        tier: Tier,
        bytes: u64,
        role: MemoryRole,
        holder: HolderId,
    ) -> Result<MemoryLease, MemoryError> {
        self.governor.reserve(tier, bytes, role, holder)
    }

    fn record_committed(
        &self,
        tier: Tier,
        bytes: u64,
        role: MemoryRole,
        holder: HolderId,
    ) -> Result<MemoryLease, MemoryError> {
        self.governor.record_committed(tier, bytes, role, holder)
    }

    fn reserve_mapped_allowance(
        &self,
        tier: Tier,
        bytes: u64,
        role: MemoryRole,
        holder: HolderId,
    ) -> Result<MappedAllowance, MemoryError> {
        self.governor
            .reserve_mapped_allowance(tier, bytes, role, holder)
    }

    fn register_reclaimable_mapped_holder(
        &self,
        holder: &Arc<dyn ReclaimableMappedHolder>,
    ) -> Result<MappedHolderRegistration, MemoryError> {
        self.governor.register_reclaimable_mapped_holder(holder)
    }

    fn prepare_mapped_growth(
        &self,
        requester: &MappedAllowance,
        bytes: u64,
    ) -> Result<MappedGrowthGrant, MemoryError> {
        self.governor.prepare_mapped_growth(requester, bytes)
    }

    fn mapped_growth_metrics(&self) -> Option<MappedGrowthMetrics> {
        self.governor.mapped_growth_metrics()
    }

    fn available(&self, tier: Tier) -> u64 {
        self.governor.available(tier)
    }

    fn oversubscribed_bytes(&self, tier: Tier) -> u64 {
        self.governor.oversubscribed_bytes(tier)
    }

    fn used(&self, tier: Tier) -> u64 {
        self.governor.used(tier)
    }
}

/// Server-owned factory for device authorities.
///
/// Standalone callers omit this provider and retain the historical behavior:
/// every engine constructs a unique device ledger.
pub trait MemoryAuthorityProvider: Send + Sync {
    /// One process manager shared by every authority/context this provider
    /// creates. The manager coordinates identity and transactions; authorities
    /// remain the budget-policy owners.
    fn process_memory_manager(&self) -> ProcessMemoryManager;

    fn validate_limit(
        &self,
        domain: &DeviceCompatibilityDomain,
        requested: ResourceLimit,
    ) -> anyhow::Result<()>;

    fn authority(
        &self,
        domain: &DeviceCompatibilityDomain,
        resolved_limit_bytes: u64,
    ) -> anyhow::Result<DeviceMemoryAuthority>;
}

pub(crate) type SharedMemoryAuthorityProvider = Arc<dyn MemoryAuthorityProvider>;

/// Engine view that shares only the device tier while retaining private host
/// and disk ledgers.
#[derive(Debug, Clone)]
pub(crate) struct EngineMemoryGovernor {
    device: DeviceMemoryAuthority,
    local: LedgerGovernor,
}

impl EngineMemoryGovernor {
    pub(crate) fn new(device: DeviceMemoryAuthority, host_bytes: u64, disk_bytes: u64) -> Self {
        Self {
            device,
            local: LedgerGovernor::new(LeaseLedger::new(0, host_bytes, disk_bytes)),
        }
    }

    pub(crate) fn device_authority(&self) -> DeviceMemoryAuthority {
        self.device.clone()
    }

    fn governor(&self, tier: Tier) -> &LedgerGovernor {
        match tier {
            Tier::Device => &self.device.governor,
            Tier::Host | Tier::Disk => &self.local,
        }
    }
}

impl MemoryGovernor for EngineMemoryGovernor {
    fn authority_id(&self) -> MemoryAuthorityId {
        self.device.authority_id()
    }

    fn reserve(
        &self,
        tier: Tier,
        bytes: u64,
        role: MemoryRole,
        holder: HolderId,
    ) -> Result<MemoryLease, MemoryError> {
        self.governor(tier).reserve(tier, bytes, role, holder)
    }

    fn record_committed(
        &self,
        tier: Tier,
        bytes: u64,
        role: MemoryRole,
        holder: HolderId,
    ) -> Result<MemoryLease, MemoryError> {
        self.governor(tier)
            .record_committed(tier, bytes, role, holder)
    }

    fn reserve_mapped_allowance(
        &self,
        tier: Tier,
        bytes: u64,
        role: MemoryRole,
        holder: HolderId,
    ) -> Result<MappedAllowance, MemoryError> {
        self.governor(tier)
            .reserve_mapped_allowance(tier, bytes, role, holder)
    }

    fn register_reclaimable_mapped_holder(
        &self,
        holder: &Arc<dyn ReclaimableMappedHolder>,
    ) -> Result<MappedHolderRegistration, MemoryError> {
        self.device.register_reclaimable_mapped_holder(holder)
    }

    fn prepare_mapped_growth(
        &self,
        requester: &MappedAllowance,
        bytes: u64,
    ) -> Result<MappedGrowthGrant, MemoryError> {
        self.device.prepare_mapped_growth(requester, bytes)
    }

    fn mapped_growth_metrics(&self) -> Option<MappedGrowthMetrics> {
        self.device.mapped_growth_metrics()
    }

    fn available(&self, tier: Tier) -> u64 {
        self.governor(tier).available(tier)
    }

    fn oversubscribed_bytes(&self, tier: Tier) -> u64 {
        self.governor(tier).oversubscribed_bytes(tier)
    }

    fn used(&self, tier: Tier) -> u64 {
        self.governor(tier).used(tier)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_authority_combines_headroom_and_releases_per_lease() {
        let authority = DeviceMemoryAuthority::new(DeviceCompatibilityDomain::Cuda(0), 100);
        let first = authority
            .reserve(Tier::Device, 40, MemoryRole::Weights, HolderId::new(1))
            .unwrap();
        let second = authority
            .reserve(Tier::Device, 50, MemoryRole::KvCache, HolderId::new(2))
            .unwrap();

        assert_eq!(authority.used_bytes(), 90);
        assert_eq!(authority.headroom_bytes(), 10);
        assert!(
            authority
                .reserve(Tier::Device, 11, MemoryRole::Activation, HolderId::new(3))
                .is_err()
        );

        drop(first);
        assert_eq!(authority.used_bytes(), 50);
        assert_eq!(authority.headroom_bytes(), 50);
        drop(second);
        assert_eq!(authority.used_bytes(), 0);
    }

    #[test]
    fn cloned_engine_governors_share_host_and_disk_ceiling() {
        let first = EngineMemoryGovernor::new(
            DeviceMemoryAuthority::new(DeviceCompatibilityDomain::Host, 100),
            100,
            80,
        );
        let second = first.clone();

        let host = first
            .reserve(Tier::Host, 60, MemoryRole::KvCache, HolderId::new(1))
            .unwrap();
        assert!(
            second
                .reserve(Tier::Host, 41, MemoryRole::KvCache, HolderId::new(2))
                .is_err(),
            "workers must not each receive the full host-RAM budget"
        );
        let disk = second
            .reserve(Tier::Disk, 50, MemoryRole::KvCache, HolderId::new(3))
            .unwrap();
        assert!(
            first
                .reserve(Tier::Disk, 31, MemoryRole::KvCache, HolderId::new(4))
                .is_err(),
            "workers must not each receive the full disk-spill budget"
        );

        drop(host);
        drop(disk);
        assert_eq!(first.used(Tier::Host), 0);
        assert_eq!(second.used(Tier::Disk), 0);
    }

    #[test]
    fn compatibility_domains_receive_distinct_authorities() {
        let cuda0 = DeviceMemoryAuthority::new(DeviceCompatibilityDomain::Cuda(0), 100);
        let cuda1 = DeviceMemoryAuthority::new(DeviceCompatibilityDomain::Cuda(1), 100);
        let other = DeviceMemoryAuthority::new(
            DeviceCompatibilityDomain::Accelerator {
                backend: "webgpu".to_string(),
                index: 0,
            },
            100,
        );

        assert_ne!(cuda0.authority_id(), cuda1.authority_id());
        assert_ne!(cuda0.authority_id(), other.authority_id());
    }
}

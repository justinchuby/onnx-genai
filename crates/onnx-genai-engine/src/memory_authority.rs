use std::{fmt, sync::Arc};

use onnx_genai_scheduler::ResourceLimit;
use onnx_runtime_memory_governor::{
    DeviceKey, HolderId, LeaseLedger, LedgerGovernor, MemoryAuthorityId, MemoryError,
    MemoryGovernor, MemoryLease, MemoryRole, Tier,
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

    pub fn set_limit_bytes(&self, bytes: u64) {
        self.governor.ledger().set_limit(Tier::Device, bytes);
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

    pub(crate) fn set_device_limit(&self, bytes: u64) {
        self.device.set_limit_bytes(bytes);
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

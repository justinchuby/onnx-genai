//! Uncached ordinary Host execution payloads with real manager-owned backing.
use std::sync::Arc;

use onnx_runtime_ep_api::{DeviceBuffer, EpError, Result};
use onnx_runtime_ir::DeviceId;
use onnx_runtime_memory_api::{DeviceKey, HostAllocator, MemoryRole, Tier};
use onnx_runtime_memory_governor::{
    AllocationPublication, AllocationRequest, MemoryAuthorityId, MemoryGovernor,
    ProcessMemoryManager, RegisteredMemoryAuthority, RegisteredMemoryContext,
    RegisteredMemoryHolder, RegisteredMemoryMechanism, ScopedMemoryBinding,
};

/// One ordinary execution-buffer domain, shared by clones, not a copied budget.
///
/// Every allocation is uncached and charged for its actual nonzero capacity.
/// This covers provider-owned tensor/state backing, not mmap weight residency,
/// arbitrary IR heap, or caller-owned tensors. Workspace and packed weights keep
/// their separate resource domains and must use the same Host authority.
#[derive(Clone)]
pub struct CpuBufferMemory(Arc<BufferDomain>);

struct BufferDomain {
    manager: ProcessMemoryManager,
    binding: ScopedMemoryBinding,
    holder: RegisteredMemoryHolder,
    authority: RegisteredMemoryAuthority,
    // Last so the scoped binding drops before backing registration retires.
    _registration: BufferRegistration,
    authority_id: MemoryAuthorityId,
}
struct BufferRegistration {
    manager: ProcessMemoryManager,
    context: RegisteredMemoryContext,
    mechanism: RegisteredMemoryMechanism,
}
impl Drop for BufferRegistration {
    fn drop(&mut self) {
        if let Err(error) = self.manager.retire(&self.mechanism) {
            eprintln!("CPU ordinary Host buffer mechanism could not retire: {error}");
            return;
        }
        if let Err(error) = self.manager.remove_mechanism(&self.mechanism) {
            eprintln!("CPU ordinary Host backing remains pinned by live ownership: {error}");
            return;
        }
        if let Err(error) = self.manager.remove_provider_context(&self.context) {
            eprintln!("CPU ordinary Host context could not retire: {error}");
        }
    }
}
impl Drop for BufferDomain {
    fn drop(&mut self) {
        // Outstanding/quarantined owners stay authoritative in the manager.
        let _ = self.manager.unregister_holder(&self.holder);
    }
}
fn backend(
    operation: &'static str,
    source: impl std::error::Error + Send + Sync + 'static,
) -> EpError {
    EpError::Backend {
        operation,
        source: Box::new(source),
    }
}
impl CpuBufferMemory {
    /// Register an uncached allocator over the caller's actual Host books.
    pub fn governed(
        manager: &ProcessMemoryManager,
        governor: Arc<dyn MemoryGovernor + Send + Sync>,
    ) -> Result<Self> {
        let authority = manager
            .register_authority(
                DeviceKey::HOST,
                "CPU ordinary Host payload authority",
                Arc::new(()),
                governor,
            )
            .map_err(|e| backend("register CPU ordinary Host authority", e))?;
        let authority_id = authority.memory_authority_id().ok_or_else(|| EpError::KernelFailed("CPU ordinary Host payloads require a governed authority; supply the actual model Host governor".into()))?;
        let context = manager
            .register_provider_context(DeviceKey::HOST, "CPU ordinary Host payloads", Arc::new(()))
            .map_err(|e| backend("register CPU ordinary Host context", e))?;
        let mechanism = match manager.register_allocator(
            &context,
            &authority,
            "uncached CPU ordinary Host backing",
            Arc::new(HostAllocator),
        ) {
            Ok(mechanism) => mechanism,
            Err(error) => {
                let _ = manager.remove_provider_context(&context);
                return Err(backend("register CPU ordinary Host allocator", error));
            }
        };
        let registration = BufferRegistration {
            manager: manager.clone(),
            context,
            mechanism,
        };
        let binding = manager
            .bind_registered(&registration.mechanism)
            .map_err(|e| backend("bind CPU ordinary Host backing", e))?;
        let holder = manager
            .register_holder(
                &authority,
                "CPU ordinary execution and retained state backing",
                None,
            )
            .map_err(|e| backend("register CPU ordinary Host holder", e))?;
        Ok(Self(Arc::new(BufferDomain {
            manager: manager.clone(),
            binding,
            holder,
            authority,
            _registration: registration,
            authority_id,
        })))
    }
    pub(crate) fn host_payload_funding(&self) -> onnx_runtime_ep_api::HostPayloadFunding {
        onnx_runtime_ep_api::HostPayloadFunding::new(self.0.authority_id, self.0.manager.clone())
    }
    pub fn authority_id(&self) -> MemoryAuthorityId {
        self.0.authority_id
    }
    pub fn uses_process_manager(&self, manager: &ProcessMemoryManager) -> bool {
        self.0.manager.is_same_instance(manager)
    }
    pub(crate) fn matches_workspace(
        &self,
        pool: &onnx_runtime_memory_governor::HostWorkspacePool,
    ) -> bool {
        pool.authority_id() == Some(self.authority_id())
            && pool.uses_process_manager(&self.0.manager)
    }
    pub(crate) fn allocate(
        &self,
        size: usize,
        alignment: usize,
        device: DeviceId,
    ) -> Result<DeviceBuffer> {
        let bytes = size.max(1);
        if bytes > isize::MAX as usize
            || std::alloc::Layout::from_size_align(bytes, alignment).is_err()
        {
            return Err(EpError::KernelFailed(format!(
                "CPU ordinary Host allocation rejects {size} bytes aligned to {alignment}; use addressable capacity and power-of-two alignment"
            )));
        }
        let bytes64 =
            u64::try_from(bytes).map_err(|e| backend("size CPU ordinary Host allocation", e))?;
        let process_bytes = if self.0.authority.has_process_delegation(Tier::Host) {
            0
        } else {
            bytes64
        };
        let request = AllocationRequest::managed(
            bytes,
            alignment,
            Tier::Host,
            MemoryRole::Activation,
            self.0.holder.clone(),
            bytes64,
        )
        .with_process_reservation(process_bytes);
        let mut publication = AllocationPublication::exclusive(bytes64, bytes64, bytes64);
        publication.process_reserved_bytes = process_bytes;
        let owner = self
            .0
            .binding
            .allocate(request, publication)
            .map_err(|e| backend("allocate CPU ordinary Host execution/state payload", e))?;
        Ok(DeviceBuffer::from_managed_allocation(
            Box::new(owner),
            device,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CpuExecutionProvider, CpuMemoryResources, CpuWeightCacheLimits, DerivedWeightMemory,
    };
    use onnx_runtime_ep_api::ExecutionProvider;
    use onnx_runtime_memory_governor::{HolderId, HostWorkspacePool, LeaseLedger, LedgerGovernor};

    fn books(limit: u64) -> Arc<LedgerGovernor> {
        Arc::new(LedgerGovernor::new(LeaseLedger::new_for_device(
            DeviceKey::HOST,
            0,
            limit,
            0,
        )))
    }
    #[test]
    fn ordinary_buffers_contend_across_real_providers_and_refund_after_release() {
        let manager = ProcessMemoryManager::new().unwrap();
        let governor = books(16);
        let domain = CpuBufferMemory::governed(&manager, governor.clone()).unwrap();
        let resources = CpuMemoryResources::default()
            .with_buffer_memory(domain)
            .unwrap();
        // Ordinary capability is independent of the derived-weight predicate.
        assert!(!resources.is_governed());
        assert!(resources.buffer_memory().is_some());
        let first = CpuExecutionProvider::initialized_with_resources(resources.clone()).unwrap();
        let second = CpuExecutionProvider::initialized_with_resources(resources).unwrap();
        let mut live = first.allocate(12, 64).unwrap();
        first.copy_from_host(&[7; 12], &mut live).unwrap();
        let error = second.allocate(8, 64).unwrap_err();
        assert!(error.to_string().contains("ordinary Host"), "{error}");
        assert_eq!(governor.used(Tier::Host), 12);
        assert_eq!(manager.process_used(Tier::Host), 12);
        let mut bytes = [0; 12];
        first.copy_to_host(&live, &mut bytes).unwrap();
        assert_eq!(bytes, [7; 12]);
        first.deallocate(live).unwrap();
        let retried = second.allocate(16, 64).unwrap();
        second.deallocate(retried).unwrap();
        assert_eq!(governor.used(Tier::Host), 0);
        assert_eq!(manager.process_used(Tier::Host), 0);
        assert!(manager.snapshot().unwrap().allocations.is_empty());
    }
    #[test]
    fn empty_buffer_charges_one_physical_byte_and_invalid_alignment_never_publishes() {
        let manager = ProcessMemoryManager::new().unwrap();
        let governor = books(1);
        let domain = CpuBufferMemory::governed(&manager, governor.clone()).unwrap();
        let provider = CpuExecutionProvider::initialized_with_resources(
            CpuMemoryResources::default()
                .with_buffer_memory(domain)
                .unwrap(),
        )
        .unwrap();
        assert!(provider.allocate(0, 3).is_err());
        assert_eq!(governor.used(Tier::Host), 0);
        let zero = provider.allocate(0, 64).unwrap();
        assert_eq!(zero.len(), 1);
        assert_eq!(governor.used(Tier::Host), 1);
        assert_eq!(manager.process_used(Tier::Host), 1);
        assert!(provider.allocate(0, 64).is_err());
        provider.deallocate(zero).unwrap();
        assert_eq!(governor.used(Tier::Host), 0);
    }
    #[test]
    fn attachment_orders_require_exact_authority_and_process_manager() {
        let manager = ProcessMemoryManager::new().unwrap();
        let other_manager = ProcessMemoryManager::new().unwrap();
        let governor = books(1024);
        let other_governor = books(1024); // Same limit is not the same authority.
        let domain = CpuBufferMemory::governed(&manager, governor.clone()).unwrap();
        let shared_pool = HostWorkspacePool::governed(&manager, governor.clone()).unwrap();
        let foreign_pool = HostWorkspacePool::governed(&other_manager, governor.clone()).unwrap();
        let wrong_pool = HostWorkspacePool::governed(&manager, other_governor.clone()).unwrap();
        for pool in [foreign_pool, wrong_pool] {
            assert!(
                CpuMemoryResources::default()
                    .with_buffer_memory(domain.clone())
                    .unwrap()
                    .with_workspace_pool(pool.clone())
                    .is_err()
            );
            assert!(
                CpuMemoryResources::default()
                    .with_workspace_pool(pool)
                    .unwrap()
                    .with_buffer_memory(domain.clone())
                    .is_err()
            );
        }
        CpuMemoryResources::default()
            .with_buffer_memory(domain.clone())
            .unwrap()
            .with_workspace_pool(shared_pool.clone())
            .unwrap();
        CpuMemoryResources::default()
            .with_workspace_pool(shared_pool)
            .unwrap()
            .with_buffer_memory(domain.clone())
            .unwrap();
        let weights =
            DerivedWeightMemory::governed(other_governor.as_ref(), HolderId::new(71)).unwrap();
        assert!(
            CpuMemoryResources::new(weights, CpuWeightCacheLimits::default())
                .unwrap()
                .with_buffer_memory(domain)
                .is_err()
        );
    }
}

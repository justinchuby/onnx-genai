//! Workspace ownership for providers whose kernel calls complete synchronously.
//! The registration witness is constructed from a concrete synchronous provider,
//! never inferred from a device name, host accessibility, or a default fence.
use std::alloc::Layout;
use std::sync::Arc;

use onnx_runtime_ep_api::{
    DevicePtrMut, EpError, ExecutionProvider, Kernel, TensorMetadata, TensorMut, TensorView,
    WorkspaceAllocation, WorkspaceLayout, WorkspaceLifetime, WorkspaceRequirement, WorkspaceView,
};
use onnx_runtime_ep_cpu::CpuExecutionProvider;

use crate::{EagerError, Result};

#[derive(Clone)]
pub(crate) struct SynchronousProvider(Arc<dyn ExecutionProvider>);

impl SynchronousProvider {
    pub(crate) fn cpu(provider: CpuExecutionProvider) -> Self {
        Self(Arc::new(provider))
    }

    pub(crate) fn ep(&self) -> &Arc<dyn ExecutionProvider> {
        &self.0
    }

    #[cfg(test)]
    pub(crate) fn for_test(provider: Arc<dyn ExecutionProvider>) -> Self {
        Self(provider)
    }
}

/// A physical owner always remembers the exact provider that must retire it.
struct WorkspaceOwner {
    provider: SynchronousProvider,
    allocation: Option<WorkspaceAllocation>,
}

impl WorkspaceOwner {
    fn allocate(provider: &SynchronousProvider, requirement: WorkspaceRequirement) -> Result<Self> {
        let bytes = usize::try_from(requirement.bytes).map_err(|_| {
            EpError::KernelFailed(format!(
                "eager workspace requires {} bytes, exceeding addressable capacity; reduce the input shape",
                requirement.bytes
            ))
        })?;
        Layout::from_size_align(bytes, requirement.alignment).map_err(|_| {
            EpError::KernelFailed(format!(
                "eager workspace requires invalid layout ({bytes} bytes, alignment {}); the kernel must report an addressable size and power-of-two alignment",
                requirement.alignment
            ))
        })?;
        let allocation =
            provider
                .0
                .allocate_workspace(bytes, requirement.alignment, requirement.role())?;
        Ok(Self {
            provider: provider.clone(),
            allocation: Some(allocation),
        })
    }

    fn fits(&self, requirement: WorkspaceRequirement) -> bool {
        self.allocation.as_ref().is_some_and(|allocation| {
            allocation.len() as u64 >= requirement.bytes
                && allocation.alignment() >= requirement.alignment
        })
    }

    fn view(&mut self, requirement: WorkspaceRequirement) -> Result<WorkspaceView> {
        let layout = Layout::from_size_align(
            usize::try_from(requirement.bytes).map_err(|_| EpError::KernelFailed("eager workspace byte count exceeds usize; reduce the input shape".into()))?,
            requirement.alignment,
        ).map_err(|_| EpError::KernelFailed(format!(
            "eager workspace layout is invalid: {} bytes aligned to {}; correct the kernel workspace requirement",
            requirement.bytes, requirement.alignment
        )))?;
        let allocation = self
            .allocation
            .as_mut()
            .expect("live workspace owns its allocation");
        let backing = WorkspaceView::new(DevicePtrMut(allocation.as_mut_ptr()), allocation.len());
        let [view] = WorkspaceLayout::new([layout])?.bind(Some(backing))?;
        Ok(view)
    }

    fn release(mut self) -> Result<()> {
        let allocation = self
            .allocation
            .take()
            .expect("live workspace owns its allocation");
        self.provider.0.deallocate_workspace(allocation)?;
        Ok(())
    }
}

impl Drop for WorkspaceOwner {
    fn drop(&mut self) {
        if let Some(allocation) = self.allocation.take()
            && let Err(error) = self.provider.0.deallocate_workspace(allocation)
        {
            // Consuming release must not be retried. Managed providers retain
            // unsettled backing and its charge through their quarantine contract.
            eprintln!(
                "eager workspace retirement failed for provider '{}': {error}; inspect the provider's retained/quarantined allocations",
                self.provider.0.name()
            );
        }
    }
}

pub(crate) struct PreparedKernel {
    kernel: Box<dyn Kernel>,
    persistent: Option<WorkspaceOwner>,
}

impl PreparedKernel {
    pub(crate) fn new(kernel: Box<dyn Kernel>) -> Self {
        Self {
            kernel,
            persistent: None,
        }
    }

    pub(crate) fn execute(
        &mut self,
        provider: &SynchronousProvider,
        op: &str,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
    ) -> Result<()> {
        let metadata: smallvec::SmallVec<[_; 8]> = inputs
            .iter()
            .map(|input| TensorMetadata::new(input.dtype, input.shape, !input.is_absent()))
            .collect();
        let output_metadata: smallvec::SmallVec<[_; 4]> = outputs
            .iter()
            .map(onnx_runtime_ep_api::OutputMetadata::from_view)
            .collect();
        let requirement = self.kernel.workspace_requirement_for_execution(
            inputs,
            outputs,
            &metadata,
            &output_metadata,
            None,
        )?;
        if requirement.bytes == 0 {
            self.kernel.execute_with_workspace(inputs, outputs, None)?;
            return Ok(());
        }
        match requirement.lifetime {
            WorkspaceLifetime::StepScoped => {
                let mut owner = WorkspaceOwner::allocate(provider, requirement)?;
                // The registration witness proves completion on both success
                // and error, so returning this owner to the pool is now safe.
                let execution = (|| -> Result<()> {
                    let view = owner.view(requirement)?;
                    self.kernel
                        .execute_with_workspace(inputs, outputs, Some(view))?;
                    Ok(())
                })();
                let release = owner.release();
                match (execution, release) {
                    (Ok(()), Ok(())) => Ok(()),
                    (Err(error), Ok(())) => Err(error),
                    (Ok(()), Err(error)) => Err(error),
                    (Err(execution), Err(release)) => Err(EagerError::WorkspaceRelease {
                        op: op.into(),
                        execution: Box::new(execution),
                        release: Box::new(release),
                    }),
                }
            }
            WorkspaceLifetime::SessionPersistent => {
                if self
                    .persistent
                    .as_ref()
                    .is_none_or(|owner| !owner.fits(requirement))
                {
                    if let Some(old) = self.persistent.take() {
                        old.release()?;
                    }
                    self.persistent = Some(WorkspaceOwner::allocate(provider, requirement)?);
                }
                let view = self
                    .persistent
                    .as_mut()
                    .expect("persistent workspace was prepared")
                    .view(requirement)?;
                self.kernel
                    .execute_with_workspace(inputs, outputs, Some(view))?;
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ep_api::{DeviceBuffer, EpConfig, Fence, KernelMatch};
    use onnx_runtime_ep_cpu::{CpuMemoryResources, DerivedWeightMemory};
    use onnx_runtime_ir::{DataType, DeviceId, DeviceType, Node, Shape, TensorLayout};
    use onnx_runtime_memory_api::{DeviceKey, MemoryError, MemoryRole, Tier};
    use onnx_runtime_memory_governor::{
        HolderId, HostWorkspacePool, LeaseAccounting, LeaseLedger, LedgerGovernor,
        MemoryAuthorityId, MemoryGovernor, MemoryLease, ProcessMemoryManager,
    };
    use std::collections::HashMap;
    use std::sync::{
        Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    };

    #[derive(Debug)]
    struct Books {
        inner: LedgerGovernor,
        claims: AtomicUsize,
        refunds: AtomicUsize,
    }
    impl Books {
        fn callbacks(&self) -> (usize, usize) {
            (
                self.claims.load(Ordering::Relaxed),
                self.refunds.load(Ordering::Relaxed),
            )
        }
    }
    impl LeaseAccounting for Books {
        fn authority_id(&self) -> Option<MemoryAuthorityId> {
            Some(self.inner.authority_id(Tier::Host))
        }
        fn try_claim(
            &self,
            tier: Tier,
            bytes: u64,
            role: MemoryRole,
        ) -> std::result::Result<(), MemoryError> {
            self.claims.fetch_add(1, Ordering::Relaxed);
            self.inner.ledger().try_claim(tier, bytes, role)
        }
        fn release(&self, tier: Tier, bytes: u64) {
            self.refunds.fetch_add(1, Ordering::Relaxed);
            self.inner.ledger().release(tier, bytes);
        }
    }
    struct Governor(Arc<Books>);
    impl MemoryGovernor for Governor {
        fn authority_id(&self, tier: Tier) -> MemoryAuthorityId {
            self.0.inner.authority_id(tier)
        }
        fn reserve(
            &self,
            tier: Tier,
            bytes: u64,
            role: MemoryRole,
            holder: HolderId,
        ) -> std::result::Result<MemoryLease, MemoryError> {
            self.0.try_claim(tier, bytes, role)?;
            Ok(MemoryLease::new(tier, bytes, role, holder, self.0.clone()))
        }
        fn available(&self, tier: Tier) -> u64 {
            self.0.inner.available(tier)
        }
        fn used(&self, tier: Tier) -> u64 {
            self.0.inner.used(tier)
        }
    }

    struct Probe {
        bytes: AtomicUsize,
        alignment: AtomicUsize,
        lifetime: WorkspaceLifetime,
        fail: AtomicBool,
        panic: AtomicBool,
        release_error: AtomicBool,
        short_backing: AtomicBool,
        allocations: AtomicUsize,
        releases: AtomicUsize,
        addresses: Mutex<Vec<usize>>,
        release_hook: Mutex<Option<Box<dyn Fn() + Send + Sync>>>,
    }
    impl Probe {
        fn new(lifetime: WorkspaceLifetime) -> Arc<Self> {
            Arc::new(Self {
                bytes: AtomicUsize::new(64),
                alignment: AtomicUsize::new(16),
                lifetime,
                fail: AtomicBool::new(false),
                panic: AtomicBool::new(false),
                release_error: AtomicBool::new(false),
                short_backing: AtomicBool::new(false),
                allocations: AtomicUsize::new(0),
                releases: AtomicUsize::new(0),
                addresses: Mutex::new(Vec::new()),
                release_hook: Mutex::new(None),
            })
        }
    }
    struct ProbeKernel(Arc<Probe>);
    impl Kernel for ProbeKernel {
        fn execute(
            &self,
            _: &[TensorView],
            _: &mut [TensorMut],
        ) -> onnx_runtime_ep_api::Result<()> {
            panic!("eager must invoke the prepared execution entry point")
        }
        fn workspace_requirement(
            &self,
            _: &[TensorMetadata<'_>],
            _outputs: &[onnx_runtime_ep_api::OutputMetadata<'_>],
        ) -> onnx_runtime_ep_api::Result<onnx_runtime_ep_api::WorkspacePreparation> {
            Ok(onnx_runtime_ep_api::WorkspacePreparation::Ready(
                WorkspaceRequirement {
                    bytes: self.0.bytes.load(Ordering::Relaxed) as u64,
                    alignment: self.0.alignment.load(Ordering::Relaxed),
                    lifetime: self.0.lifetime,
                },
            ))
        }
        fn execute_with_workspace(
            &self,
            inputs: &[TensorView],
            outputs: &mut [TensorMut],
            workspace: Option<WorkspaceView>,
        ) -> onnx_runtime_ep_api::Result<()> {
            assert_eq!(
                workspace.is_some(),
                self.0.bytes.load(Ordering::Relaxed) != 0
            );
            if let Some(workspace) = workspace {
                assert!(workspace.bytes() >= self.0.bytes.load(Ordering::Relaxed));
                self.0
                    .addresses
                    .lock()
                    .unwrap()
                    .push(workspace.ptr().0 as usize);
                // SAFETY: the test provider supplied exclusive live Host scratch.
                unsafe { workspace.ptr().0.cast::<u8>().write(73) };
            }
            assert!(
                !self.0.panic.load(Ordering::Relaxed),
                "injected synchronous kernel panic"
            );
            if self.0.fail.load(Ordering::Relaxed) {
                return Err(EpError::KernelFailed(
                    "injected synchronous kernel failure".into(),
                ));
            }
            if let (Some(input), Some(output)) = (inputs.first(), outputs.first_mut()) {
                for i in 0..output.numel() {
                    // SAFETY: the dispatch fixture passes contiguous f32 input/output of equal length.
                    unsafe {
                        output
                            .data_ptr_mut::<f32>()
                            .add(i)
                            .write(input.data_ptr::<f32>().add(i).read() + 1.0)
                    };
                }
            }
            Ok(())
        }
    }

    struct TestProvider {
        cpu: CpuExecutionProvider,
        probe: Arc<Probe>,
    }
    impl ExecutionProvider for TestProvider {
        fn name(&self) -> &str {
            "synchronous_test_provider"
        }
        fn device_type(&self) -> DeviceType {
            DeviceType::Cpu
        }
        fn device_id(&self) -> DeviceId {
            DeviceId::cpu()
        }
        fn initialize(&mut self, config: &EpConfig) -> onnx_runtime_ep_api::Result<()> {
            self.cpu.initialize(config)
        }
        fn shutdown(&mut self) -> onnx_runtime_ep_api::Result<()> {
            self.cpu.shutdown()
        }
        fn supports_op(
            &self,
            _: &Node,
            _: u64,
            _: &[Shape],
            _: &[DataType],
            _: &[TensorLayout],
        ) -> KernelMatch {
            KernelMatch::unsupported("test invokes get_kernel directly")
        }
        fn get_kernel(
            &self,
            _: &Node,
            _: &[Vec<usize>],
            _: u64,
        ) -> onnx_runtime_ep_api::Result<Box<dyn Kernel>> {
            Ok(Box::new(ProbeKernel(self.probe.clone())))
        }
        fn allocate(
            &self,
            bytes: usize,
            alignment: usize,
        ) -> onnx_runtime_ep_api::Result<DeviceBuffer> {
            self.cpu.allocate(bytes, alignment)
        }
        fn deallocate(&self, buffer: DeviceBuffer) -> onnx_runtime_ep_api::Result<()> {
            self.cpu.deallocate(buffer)
        }
        fn allocate_workspace(
            &self,
            bytes: usize,
            alignment: usize,
            role: MemoryRole,
        ) -> onnx_runtime_ep_api::Result<WorkspaceAllocation> {
            self.probe.allocations.fetch_add(1, Ordering::Relaxed);
            let bytes = if self.probe.short_backing.load(Ordering::Relaxed) {
                bytes / 2
            } else {
                bytes
            };
            self.cpu.allocate_workspace(bytes, alignment, role)
        }
        fn deallocate_workspace(
            &self,
            owner: WorkspaceAllocation,
        ) -> onnx_runtime_ep_api::Result<()> {
            self.probe.releases.fetch_add(1, Ordering::Relaxed);
            self.cpu.deallocate_workspace(owner)?;
            if let Some(hook) = &*self.probe.release_hook.lock().unwrap() {
                hook();
            }
            if self.probe.release_error.load(Ordering::Relaxed) {
                return Err(EpError::KernelFailed(
                    "injected consuming workspace release failure".into(),
                ));
            }
            Ok(())
        }
        fn copy(
            &self,
            src: &DeviceBuffer,
            dst: &mut DeviceBuffer,
            bytes: usize,
        ) -> onnx_runtime_ep_api::Result<()> {
            self.cpu.copy(src, dst, bytes)
        }
        fn copy_async(
            &self,
            src: &DeviceBuffer,
            dst: &mut DeviceBuffer,
            bytes: usize,
        ) -> onnx_runtime_ep_api::Result<Fence> {
            self.cpu.copy_async(src, dst, bytes)
        }
        fn sync(&self) -> onnx_runtime_ep_api::Result<()> {
            panic!("eager must not introduce a per-dispatch synchronization")
        }
    }
    fn fixture(
        lifetime: WorkspaceLifetime,
    ) -> (
        SynchronousProvider,
        Arc<Probe>,
        Arc<Books>,
        HostWorkspacePool,
        ProcessMemoryManager,
    ) {
        let books = Arc::new(Books {
            inner: LedgerGovernor::new(LeaseLedger::new_for_device(DeviceKey::HOST, 0, 4096, 0)),
            claims: AtomicUsize::new(0),
            refunds: AtomicUsize::new(0),
        });
        let governor = Arc::new(Governor(books.clone()));
        let manager = ProcessMemoryManager::new().unwrap();
        let pool = HostWorkspacePool::governed(&manager, governor.clone()).unwrap();
        let resources = CpuMemoryResources::new(
            DerivedWeightMemory::governed(governor.as_ref(), HolderId::new(17)).unwrap(),
            Default::default(),
        )
        .unwrap()
        .with_workspace_pool(pool.clone())
        .unwrap();
        let probe = Probe::new(lifetime);
        let cpu = CpuExecutionProvider::initialized_with_resources(resources).unwrap();
        let provider = SynchronousProvider::for_test(Arc::new(TestProvider {
            cpu,
            probe: probe.clone(),
        }));
        (provider, probe, books, pool, manager)
    }

    struct OutputBoundProbe(Arc<Probe>);
    impl Kernel for OutputBoundProbe {
        fn execute(
            &self,
            _: &[TensorView],
            _: &mut [TensorMut],
        ) -> onnx_runtime_ep_api::Result<()> {
            panic!("output-bound probe requires prepared workspace")
        }
        fn workspace_requirement(
            &self,
            _: &[TensorMetadata<'_>],
            outputs: &[onnx_runtime_ep_api::OutputMetadata<'_>],
        ) -> onnx_runtime_ep_api::Result<onnx_runtime_ep_api::WorkspacePreparation> {
            let [onnx_runtime_ep_api::OutputMetadata::Tensor { shape, .. }] = outputs else {
                return Err(EpError::KernelFailed(
                    "resolve the probe output before planning".into(),
                ));
            };
            let bytes = shape.iter().product::<usize>() * 16;
            self.0.bytes.store(bytes, Ordering::Relaxed);
            Ok(onnx_runtime_ep_api::WorkspacePreparation::Ready(
                WorkspaceRequirement {
                    bytes: bytes as u64,
                    alignment: 16,
                    lifetime: self.0.lifetime,
                },
            ))
        }
        fn execute_with_workspace(
            &self,
            inputs: &[TensorView],
            outputs: &mut [TensorMut],
            workspace: Option<WorkspaceView>,
        ) -> onnx_runtime_ep_api::Result<()> {
            ProbeKernel(self.0.clone()).execute_with_workspace(inputs, outputs, workspace)
        }
    }

    struct DeferredOutputProbe {
        probe: Arc<Probe>,
        origin: std::sync::atomic::AtomicUsize,
    }
    impl Kernel for DeferredOutputProbe {
        fn execute(
            &self,
            _: &[TensorView],
            _: &mut [TensorMut],
        ) -> onnx_runtime_ep_api::Result<()> {
            panic!("deferred output probe requires admitted scratch")
        }
        fn workspace_requirement(
            &self,
            _: &[TensorMetadata<'_>],
            _: &[onnx_runtime_ep_api::OutputMetadata<'_>],
        ) -> onnx_runtime_ep_api::Result<onnx_runtime_ep_api::WorkspacePreparation> {
            Ok(onnx_runtime_ep_api::WorkspacePreparation::Deferred)
        }
        fn workspace_requirement_for_execution(
            &self,
            _: &[TensorView],
            outputs: &[TensorMut],
            _: &[TensorMetadata<'_>],
            metadata: &[onnx_runtime_ep_api::OutputMetadata<'_>],
            cached: Option<onnx_runtime_ep_api::WorkspacePreparation>,
        ) -> onnx_runtime_ep_api::Result<WorkspaceRequirement> {
            assert!(cached.is_none());
            let [output] = outputs else {
                return Err(EpError::KernelFailed(
                    "deferred eager plan requires one actual output buffer".into(),
                ));
            };
            assert_eq!(
                metadata,
                [onnx_runtime_ep_api::OutputMetadata::from_view(output)]
            );
            let origin = output.data.0 as usize + output.byte_offset;
            self.origin.store(origin, Ordering::Relaxed);
            let bytes = if output.byte_offset == 0 { 64 } else { 80 };
            self.probe.bytes.store(bytes, Ordering::Relaxed);
            Ok(WorkspaceRequirement {
                bytes: bytes as u64,
                alignment: 16,
                lifetime: WorkspaceLifetime::SessionPersistent,
            })
        }
        fn execute_with_workspace(
            &self,
            inputs: &[TensorView],
            outputs: &mut [TensorMut],
            workspace: Option<WorkspaceView>,
        ) -> onnx_runtime_ep_api::Result<()> {
            assert_eq!(
                outputs[0].data.0 as usize + outputs[0].byte_offset,
                self.origin.load(Ordering::Relaxed)
            );
            ProbeKernel(self.probe.clone()).execute_with_workspace(inputs, outputs, workspace)
        }
    }

    #[test]
    fn deferred_workspace_output_bindings_reach_eager_admission_without_warm_issuer_calls() {
        let (provider, probe, books, _pool, _manager) =
            fixture(WorkspaceLifetime::SessionPersistent);
        let mut kernel = PreparedKernel::new(Box::new(DeferredOutputProbe {
            probe: probe.clone(),
            origin: AtomicUsize::new(0),
        }));
        let mut backing = [0u64; 8];
        for offset in [0, 4, 4, 0] {
            let mut output = TensorMut::new(
                onnx_runtime_ep_api::DevicePtrMut(backing.as_mut_ptr().cast()),
                onnx_runtime_ir::DataType::Float32,
                &[1],
                &[1],
                onnx_runtime_ir::DeviceId::cpu(),
            );
            output.byte_offset = offset;
            let before = books.callbacks();
            let warmed = probe.allocations.load(Ordering::Relaxed) >= 2;
            kernel
                .execute(&provider, "DeferredOutputProbe", &[], &mut [output])
                .unwrap();
            if warmed {
                assert_eq!(
                    books.callbacks(),
                    before,
                    "fitting deferred refinement must not request or return quota"
                );
            }
        }
        assert_eq!(probe.allocations.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn output_workspace_metadata_reaches_eager_admission_and_growth() {
        let (provider, probe, books, _pool, _manager) =
            fixture(WorkspaceLifetime::SessionPersistent);
        let mut kernel = PreparedKernel::new(Box::new(OutputBoundProbe(probe.clone())));
        for elements in [2, 5, 1] {
            let shape = [elements];
            let strides = [1];
            let mut backing = vec![0f32; elements];
            let mut outputs = [TensorMut::new(
                onnx_runtime_ep_api::DevicePtrMut(backing.as_mut_ptr().cast()),
                onnx_runtime_ir::DataType::Float32,
                &shape,
                &strides,
                onnx_runtime_ir::DeviceId::cpu(),
            )];
            kernel
                .execute(&provider, "OutputBound", &[], &mut outputs)
                .unwrap();
            assert_eq!(probe.bytes.load(Ordering::Relaxed), elements * 16);
            assert!(books.inner.used(Tier::Host) >= (elements * 16) as u64);
        }
        assert_eq!(probe.allocations.load(Ordering::Relaxed), 2);
        drop(kernel);
    }

    #[test]
    fn warm_step_workspace_and_zero_requirement_never_contact_issuer() {
        let (provider, probe, books, pool, _manager) = fixture(WorkspaceLifetime::StepScoped);
        let mut kernel = PreparedKernel::new(Box::new(ProbeKernel(probe.clone())));
        kernel.execute(&provider, "Probe", &[], &mut []).unwrap();
        let warm = books.callbacks();
        assert_eq!(books.inner.used(Tier::Host), 64);
        for bytes in [64, 16, 0, 32, 64, 0] {
            probe.bytes.store(bytes, Ordering::Relaxed);
            kernel.execute(&provider, "Probe", &[], &mut []).unwrap();
            assert_eq!(
                books.callbacks(),
                warm,
                "warm/zero workspace touched issuer for {bytes} bytes"
            );
            assert_eq!(
                books.inner.used(Tier::Host),
                64,
                "logical shrink must not refund retained physical capacity"
            );
        }
        let addresses = probe.addresses.lock().unwrap();
        assert!(addresses.iter().all(|address| *address == addresses[0]));
        assert_eq!(probe.allocations.load(Ordering::Relaxed), 5);
        assert_eq!(probe.releases.load(Ordering::Relaxed), 5);
        assert_eq!(pool.trim_idle(64).unwrap(), 64);
        assert_eq!(books.inner.used(Tier::Host), 0);
    }

    #[test]
    fn persistent_workspace_survives_dispatch_and_grows_then_retires() {
        let (provider, probe, books, pool, _manager) =
            fixture(WorkspaceLifetime::SessionPersistent);
        let mut kernel = PreparedKernel::new(Box::new(ProbeKernel(probe.clone())));
        kernel.execute(&provider, "Probe", &[], &mut []).unwrap();
        let warm = books.callbacks();
        for bytes in [16, 0, 64] {
            probe.bytes.store(bytes, Ordering::Relaxed);
            kernel.execute(&provider, "Probe", &[], &mut []).unwrap();
            assert_eq!(books.callbacks(), warm);
        }
        assert_eq!(probe.allocations.load(Ordering::Relaxed), 1);
        assert_eq!(probe.releases.load(Ordering::Relaxed), 0);
        probe.bytes.store(128, Ordering::Relaxed);
        kernel.execute(&provider, "Probe", &[], &mut []).unwrap();
        assert_eq!(books.inner.used(Tier::Host), 128);
        assert_eq!(
            pool.trim_idle(4096).unwrap(),
            0,
            "persistent owner is not idle"
        );
        drop(kernel);
        assert_eq!(probe.releases.load(Ordering::Relaxed), 2);
        assert_eq!(books.inner.used(Tier::Host), 128);
        assert_eq!(pool.trim_idle(4096).unwrap(), 128);
    }

    #[test]
    fn execution_and_consuming_release_errors_both_survive_without_retry() {
        let (provider, probe, _, _, _manager) = fixture(WorkspaceLifetime::StepScoped);
        probe.fail.store(true, Ordering::Relaxed);
        probe.release_error.store(true, Ordering::Relaxed);
        let mut kernel = PreparedKernel::new(Box::new(ProbeKernel(probe.clone())));
        let error = kernel
            .execute(&provider, "Probe", &[], &mut [])
            .unwrap_err();
        let message = error.to_string();
        assert!(
            message.contains("Probe")
                && message.contains("kernel failure")
                && message.contains("release failure"),
            "{message}"
        );
        assert_eq!(probe.releases.load(Ordering::Relaxed), 1);
        drop(kernel);
        assert_eq!(probe.releases.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn binding_failure_and_release_failure_are_both_reported() {
        let (provider, probe, _, _, _manager) = fixture(WorkspaceLifetime::StepScoped);
        probe.short_backing.store(true, Ordering::Relaxed);
        probe.release_error.store(true, Ordering::Relaxed);
        let mut kernel = PreparedKernel::new(Box::new(ProbeKernel(probe.clone())));
        let message = kernel
            .execute(&provider, "Probe", &[], &mut [])
            .unwrap_err()
            .to_string();
        assert!(
            message.contains("64 bytes") && message.contains("release failure"),
            "{message}"
        );
        assert_eq!(probe.releases.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn step_owner_is_returned_on_panic_and_invalid_layout_never_allocates() {
        let (provider, probe, _, pool, _manager) = fixture(WorkspaceLifetime::StepScoped);
        let mut kernel = PreparedKernel::new(Box::new(ProbeKernel(probe.clone())));
        probe.alignment.store(3, Ordering::Relaxed);
        assert!(
            kernel
                .execute(&provider, "Probe", &[], &mut [])
                .unwrap_err()
                .to_string()
                .contains("alignment 3")
        );
        assert_eq!(probe.allocations.load(Ordering::Relaxed), 0);
        probe.alignment.store(16, Ordering::Relaxed);
        probe.panic.store(true, Ordering::Relaxed);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| kernel.execute(
                &provider,
                "Probe",
                &[],
                &mut []
            )))
            .is_err()
        );
        assert_eq!(probe.releases.load(Ordering::Relaxed), 1);
        assert_eq!(pool.trim_idle(4096).unwrap(), 64);
    }

    #[test]
    fn public_dispatch_prepares_workspace_and_evicts_outside_cache_lock() {
        let (provider, probe, _, _pool, _manager) = fixture(WorkspaceLifetime::SessionPersistent);
        let mut context = crate::EagerContext::new().unwrap();
        context.eps = vec![provider];
        context.cache = Mutex::new(crate::cache::KernelCache::new(1));
        let context = Arc::new(context);
        let weak = Arc::downgrade(&context);
        let unlocked = Arc::new(AtomicBool::new(false));
        let observed = unlocked.clone();
        *probe.release_hook.lock().unwrap() = Some(Box::new(move || {
            if let Some(context) = weak.upgrade() {
                observed.store(context.cache.try_lock().is_ok(), Ordering::Relaxed);
            }
        }));
        for data in [&[1f32, 2.0][..], &[3f32, 4.0, 5.0][..]] {
            let input = crate::Tensor::from_f32(&[data.len()], data).unwrap();
            let output = context
                .dispatch("Add", "", &[&input, &input], &HashMap::new(), None)
                .unwrap();
            assert_eq!(
                output[0].to_vec_f32(),
                data.iter().map(|value| value + 1.0).collect::<Vec<_>>()
            );
        }
        assert!(
            unlocked.load(Ordering::Relaxed),
            "provider retirement ran while holding global kernel-cache lock"
        );
        assert_eq!(probe.releases.load(Ordering::Relaxed), 1);
        assert_eq!(context.cache_stats().entries, 1);
        drop(context);
        assert_eq!(probe.releases.load(Ordering::Relaxed), 2);
    }
}

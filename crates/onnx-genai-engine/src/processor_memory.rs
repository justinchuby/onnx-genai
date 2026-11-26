//! Explicit request allocation domains for processor payloads. This accounts
//! owned numeric payloads, not arbitrary allocator metadata, graph IR, or RSS.
use anyhow::Context as _;
use onnx_runtime_memory_api::{DeviceKey, Tier};
use onnx_runtime_memory_governor::{LeaseLedger, LedgerGovernor, MemoryGovernor, MemoryLease};
use std::sync::Arc;

mod sealed {
    pub trait Sealed {}
}
#[cfg(feature = "native-backend")]
pub trait ProcessorElement: sealed::Sealed + onnx_runtime_session::HostElement {
    const ZERO: Self;
}
#[cfg(not(feature = "native-backend"))]
pub trait ProcessorElement: sealed::Sealed + Copy + Send + Sync + 'static {
    const ZERO: Self;
}
macro_rules! element {
    ($($ty:ty => $zero:expr),* $(,)?) => {$(
        impl sealed::Sealed for $ty {}
        impl ProcessorElement for $ty { const ZERO: Self = $zero; }
    )*};
}
element!(u8=>0, i8=>0, u16=>0, i16=>0, u32=>0, i32=>0, u64=>0, i64=>0, usize=>0, isize=>0, f32=>0., f64=>0., bool=>false);

/// Inspectable attribution, independent of numeric element layout. Tuples and
/// arbitrary Copy types are deliberately excluded: Copy does not prove that
/// padding is initialized or that an all-zero representation is valid.
#[derive(Clone, Copy, Debug)]
pub enum ProcessorAllocationRole {
    Scratch,
    History,
    Probabilities,
}
impl ProcessorAllocationRole {
    fn holder(self) -> crate::engine::memory_plan::Holder {
        use crate::engine::memory_plan::Holder;
        match self {
            Self::Scratch => Holder::ProcessorScratch,
            Self::History => Holder::ProcessorHistory,
            Self::Probabilities => Holder::ProcessorProbabilities,
        }
    }
}

#[derive(Clone)]
pub struct ProcessorAllocationScope {
    inner: Arc<ScopeInner>,
}
struct ScopeInner {
    memory: Arc<dyn MemoryGovernor + Send + Sync>,
    storage: ScopeStorage,
}
enum ScopeStorage {
    /// ORT/reference and explicitly bounded standalone callers retain their
    /// own typed allocations with a real pre-allocation payload grant.
    Reference,
    #[cfg(feature = "native-backend")]
    Host {
        allocator: onnx_runtime_session::HostTensorAllocator,
        already_funded: bool,
    },
}
impl ProcessorAllocationScope {
    /// A low-level caller must select a finite explicit payload budget. Engine
    /// paths instead construct a scope from their existing actual authority.
    pub fn standalone(max_bytes: u64) -> Self {
        Self::reference(Arc::new(LedgerGovernor::new(LeaseLedger::new_for_device(
            DeviceKey::HOST,
            0,
            max_bytes,
            0,
        ))))
    }
    pub(crate) fn reference(memory: Arc<dyn MemoryGovernor + Send + Sync>) -> Self {
        Self {
            inner: Arc::new(ScopeInner {
                memory,
                storage: ScopeStorage::Reference,
            }),
        }
    }
    #[cfg(feature = "native-backend")]
    pub(crate) fn host(
        memory: Arc<dyn MemoryGovernor + Send + Sync>,
        manager: &onnx_runtime_memory_governor::ProcessMemoryManager,
        mut allocator: onnx_runtime_session::HostTensorAllocator,
    ) -> Self {
        let funding = allocator.host_payload_funding();
        let already_funded = funding
            .as_ref()
            .is_some_and(|domain| domain.matches(memory.authority_id(Tier::Host), manager));
        if funding.is_some() && !already_funded {
            // Do not debit or allocate through an unrelated domain. This
            // explicit staging owner is funded by this scope's Engine grant.
            allocator = onnx_runtime_session::HostTensorAllocator::standalone();
        }
        Self {
            inner: Arc::new(ScopeInner {
                memory,
                storage: ScopeStorage::Host {
                    allocator,
                    already_funded,
                },
            }),
        }
    }
    pub fn allocate<T: ProcessorElement>(
        &self,
        length: usize,
        role: ProcessorAllocationRole,
    ) -> anyhow::Result<ProcessorArray<T>> {
        let bytes = length.checked_mul(std::mem::size_of::<T>()).context(
            "processor payload length overflows; reduce vocabulary/history or request limits",
        )?;
        if length == 0 {
            return Ok(ProcessorArray {
                data: ArrayStorage::Empty,
                _grant: None,
                scope: self.clone(),
                role,
            });
        }
        let needs_grant = match &self.inner.storage {
            ScopeStorage::Reference => true,
            #[cfg(feature = "native-backend")]
            ScopeStorage::Host { already_funded, .. } => !already_funded,
        };
        let grant = if needs_grant {
            let holder = role.holder();
            Some(self.inner.memory.reserve(Tier::Host, u64::try_from(bytes)?, holder.role(), holder.id()).with_context(|| format!("admit {role:?} processor payload of {length} elements ({bytes} bytes) before allocation; release retained state or increase Host headroom"))?)
        } else {
            None
        };
        let data = match &self.inner.storage {
            ScopeStorage::Reference=> {
                // Allocate the exact typed layout once: Vec shrink-to-box
                // must not introduce an implicit second live allocation.
                let layout = std::alloc::Layout::array::<T>(length).context("processor reference payload exceeds addressable layout; shorten the request")?;
                // SAFETY: nonempty sealed zero-valid primitive elements have
                // nonzero size and valid all-zero representations. The exact
                // typed allocation is immediately installed in its Box owner.
                let pointer = unsafe { std::alloc::alloc_zeroed(layout) }.cast::<T>();
                let pointer = std::ptr::NonNull::new(pointer).with_context(|| format!("allocate {role:?} processor reference payload of {length} elements ({bytes} bytes); release memory or reduce request limits"))?;
                // SAFETY: pointer belongs to the global allocator with exactly
                // Layout::array::<T>(length); Box releases that same layout.
                let values = unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(pointer.as_ptr(), length)) };
                ArrayStorage::Reference(values)
            }
            #[cfg(feature="native-backend")]
            ScopeStorage::Host { allocator, .. }=>ArrayStorage::Host(allocator.array::<T>(length).with_context(||format!("allocate {role:?} processor payload of {length} elements ({bytes} bytes) in its Host domain; release retained state or increase Host headroom"))?),
        };
        Ok(ProcessorArray {
            data,
            _grant: grant,
            scope: self.clone(),
            role,
        })
    }
}

/// Unique mutable storage; no mutable Arc alias, ungoverned resize, byte view,
/// or Vec escape. Deep clone/resize admits the true old+new live peak first.
pub struct ProcessorArray<T: ProcessorElement> {
    data: ArrayStorage<T>,
    _grant: Option<MemoryLease>,
    scope: ProcessorAllocationScope,
    role: ProcessorAllocationRole,
}
enum ArrayStorage<T: ProcessorElement> {
    Empty,
    Reference(Box<[T]>),
    #[cfg(feature = "native-backend")]
    Host(onnx_runtime_session::OwnedHostArray<T>),
}
impl<T: ProcessorElement> ProcessorArray<T> {
    pub fn as_slice(&self) -> &[T] {
        match &self.data {
            ArrayStorage::Empty => &[],
            ArrayStorage::Reference(values) => values,
            #[cfg(feature = "native-backend")]
            ArrayStorage::Host(values) => values.as_slice(),
        }
    }
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match &mut self.data {
            ArrayStorage::Empty => &mut [],
            ArrayStorage::Reference(values) => values,
            #[cfg(feature = "native-backend")]
            ArrayStorage::Host(values) => values.as_mut_slice(),
        }
    }
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn try_clone(&self) -> anyhow::Result<Self> {
        let mut replacement = self.scope.allocate(self.len(), self.role)?;
        replacement.as_mut_slice().copy_from_slice(self.as_slice());
        Ok(replacement)
    }
    pub fn try_resize(&mut self, length: usize) -> anyhow::Result<()> {
        if length == self.len() {
            return Ok(());
        }
        let mut replacement = self.scope.allocate(length, self.role)?;
        let retained = length.min(self.len());
        replacement.as_mut_slice()[..retained].copy_from_slice(&self.as_slice()[..retained]);
        *self = replacement;
        Ok(())
    }
}
impl<T: ProcessorElement + std::fmt::Debug> std::fmt::Debug for ProcessorArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_slice().fmt(f)
    }
}

impl crate::Engine {
    /// Construct a processor scope from this Engine's actual immutable Host allocation domain.
    pub fn processor_allocation_scope(&self) -> anyhow::Result<ProcessorAllocationScope> {
        let memory: Arc<dyn MemoryGovernor + Send + Sync> =
            Arc::new(self.governor.memory().clone());
        #[cfg(feature = "native-backend")]
        if self.decode_backend == crate::EngineDecodeBackend::Native {
            let session = self.native_session.as_ref().context("Native processor allocation scope requires the loaded decoder; reload the model")?.inference_session();
            return Ok(ProcessorAllocationScope::host(
                memory,
                &self.governor.process_memory_manager(),
                crate::native_decode::host_tensors_for_session(session)?,
            ));
        }
        Ok(ProcessorAllocationScope::reference(memory))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn explicit_reference_scope_refuses_real_old_new_peak_and_preserves_retry() {
        let scope = ProcessorAllocationScope::standalone(36);
        let mut values = scope
            .allocate::<u32>(3, ProcessorAllocationRole::History)
            .unwrap();
        values.as_mut_slice().copy_from_slice(&[2, 3, 5]);
        let blocker = scope
            .allocate::<u8>(8, ProcessorAllocationRole::Scratch)
            .unwrap();
        assert!(values.try_resize(6).is_err());
        assert_eq!(values.as_slice(), &[2, 3, 5]);
        assert_eq!(scope.inner.memory.used(Tier::Host), 20);
        drop(blocker);
        values.try_resize(6).unwrap();
        assert_eq!(values.as_slice(), &[2, 3, 5, 0, 0, 0]);
        assert_eq!(scope.inner.memory.used(Tier::Host), 24);
        assert!(values.try_clone().is_err());
        drop(values);
        assert_eq!(scope.inner.memory.used(Tier::Host), 0);
    }
    #[test]
    fn explicit_zero_budget_allows_empty_owner_but_rejects_initialized_nonempty_payload() {
        let scope = ProcessorAllocationScope::standalone(0);
        let empty = scope
            .allocate::<f32>(0, ProcessorAllocationRole::Scratch)
            .unwrap();
        assert!(empty.is_empty());
        assert!(
            scope
                .allocate::<bool>(1, ProcessorAllocationRole::Scratch)
                .is_err()
        );
        let scope = ProcessorAllocationScope::standalone(2);
        let mut flags = scope
            .allocate::<bool>(2, ProcessorAllocationRole::Scratch)
            .unwrap();
        assert_eq!(flags.as_slice(), &[false, false]);
        flags.as_mut_slice()[1] = true;
        assert_eq!(flags.as_slice(), &[false, true]);
    }

    #[cfg(feature = "native-backend")]
    fn allocator(
        manager: &onnx_runtime_memory_governor::ProcessMemoryManager,
        books: Arc<LedgerGovernor>,
    ) -> onnx_runtime_session::HostTensorAllocator {
        use onnx_runtime_ep_cpu::{CpuBufferMemory, CpuExecutionProvider, CpuMemoryResources};
        let memory = CpuBufferMemory::governed(manager, books).unwrap();
        let cpu = Arc::new(
            CpuExecutionProvider::initialized_with_resources(
                CpuMemoryResources::default()
                    .with_buffer_memory(memory)
                    .unwrap(),
            )
            .unwrap(),
        );
        let mut graph = onnx_runtime_ir::Graph::new();
        graph.opset_imports.insert(String::new(), 17);
        let value = graph.create_named_value(
            "input",
            onnx_runtime_ir::DataType::Uint8,
            onnx_runtime_ir::static_shape([0]),
        );
        graph.add_input(value);
        graph.add_output(value);
        let session = onnx_runtime_session::InferenceSession::from_graph_with_provider(
            graph,
            Arc::new(onnx_runtime_loader::WeightStore::new()),
            std::path::Path::new("."),
            cpu,
            None,
        )
        .unwrap();
        session.host_tensor_allocator().unwrap()
    }
    #[cfg(feature = "native-backend")]
    #[test]
    fn matching_domain_charges_once_and_retains_actual_funding_until_last_array_owner() {
        let manager = onnx_runtime_memory_governor::ProcessMemoryManager::new().unwrap();
        let books = Arc::new(LedgerGovernor::new(LeaseLedger::new_for_device(
            DeviceKey::HOST,
            0,
            36,
            0,
        )));
        let context = allocator(&manager, books.clone());
        let scope = ProcessorAllocationScope::host(books.clone(), &manager, context);
        let mut values = scope
            .allocate::<f32>(3, ProcessorAllocationRole::Probabilities)
            .unwrap();
        values.as_mut_slice().copy_from_slice(&[0.25, 0.5, 0.25]);
        assert_eq!(books.used(Tier::Host), 12);
        assert_eq!(manager.process_used(Tier::Host), 12);
        let retained = Arc::new(values);
        let alias = retained.clone();
        drop(scope);
        drop(retained);
        assert_eq!(alias.as_slice(), &[0.25, 0.5, 0.25]);
        assert_eq!(books.used(Tier::Host), 12);
        drop(alias);
        assert_eq!(books.used(Tier::Host), 0);
        assert!(manager.snapshot().unwrap().allocations.is_empty());
    }
    #[cfg(feature = "native-backend")]
    #[test]
    fn foreign_manager_or_authority_uses_scope_grant_without_foreign_allocations() {
        for foreign_authority in [false, true] {
            let local = onnx_runtime_memory_governor::ProcessMemoryManager::new().unwrap();
            let foreign = onnx_runtime_memory_governor::ProcessMemoryManager::new().unwrap();
            let books = Arc::new(LedgerGovernor::new(LeaseLedger::new_for_device(
                DeviceKey::HOST,
                0,
                36,
                0,
            )));
            let other = Arc::new(LedgerGovernor::new(LeaseLedger::new_for_device(
                DeviceKey::HOST,
                0,
                36,
                0,
            )));
            let (context_manager, context_books) = if foreign_authority {
                (&local, other.clone())
            } else {
                (&foreign, books.clone())
            };
            let context = allocator(context_manager, context_books);
            let scope = ProcessorAllocationScope::host(books.clone(), &local, context);
            let values = scope
                .allocate::<u32>(3, ProcessorAllocationRole::History)
                .unwrap();
            assert_eq!(books.used(Tier::Host), 12);
            assert_eq!(other.used(Tier::Host), 0);
            assert_eq!(local.process_used(Tier::Host), 0);
            assert_eq!(foreign.process_used(Tier::Host), 0);
            drop(values);
            assert_eq!(books.used(Tier::Host), 0);
        }
    }
}

//! One immutable metadata artifact, staged independently of kernel publication.

use std::sync::{Arc, Mutex};

use cudarc::driver::sys::CUdeviceptr;
use onnx_runtime_ep_api::{DeviceGraphResource, EpError, Result};

use crate::runtime::{CudaRuntime, NativeDeviceAllocation};

#[derive(Debug, thiserror::Error)]
#[error(
    "cuda_ep {label}: {stage} failed: {source}; correct the metadata inputs or available device capacity, then eagerly prepare again before capture"
)]
struct MetadataPreparationError {
    label: String,
    stage: &'static str,
    #[source]
    source: EpError,
}

fn contextual_error(label: &str, stage: &'static str, source: EpError) -> EpError {
    EpError::Backend {
        operation: "prepare immutable CUDA metadata",
        source: Box::new(MetadataPreparationError {
            label: label.into(),
            stage,
            source,
        }),
    }
}

#[derive(Debug)]
struct PreparedDeviceMetadata<K> {
    key: K,
    allocation: Arc<NativeDeviceAllocation>,
}

/// A single-entry cache, not another allocator or quota authority. Kernels
/// stage a clone and publish it only after their launch succeeds. Sharing the
/// whole immutable entry avoids copying its key during candidate creation.
#[derive(Debug)]
pub(crate) struct DeviceMetadataCache<K> {
    runtime: Arc<CudaRuntime>,
    current: Option<Arc<PreparedDeviceMetadata<K>>>,
}

impl<K> Clone for DeviceMetadataCache<K> {
    fn clone(&self) -> Self {
        Self {
            runtime: Arc::clone(&self.runtime),
            current: self.current.clone(),
        }
    }
}

impl<K> DeviceMetadataCache<K> {
    pub(crate) fn new(runtime: Arc<CudaRuntime>) -> Self {
        Self {
            runtime,
            current: None,
        }
    }

    /// `matches` borrows invocation facts; encoding and ownership of a new key
    /// are needed only on a miss. The builder preserves the kernel's ABI,
    /// including any initialized scalar/empty-table sentinel it requires.
    pub(crate) fn get_or_prepare(
        &mut self,
        label: &str,
        matches: impl FnOnce(&K) -> bool,
        build: impl FnOnce() -> Result<(K, Vec<u8>)>,
    ) -> Result<CUdeviceptr> {
        if let Some(current) = &self.current
            && matches(&current.key)
        {
            if self.runtime.is_capturing()? {
                self.runtime.require_captured_address_owner(
                    NativeDeviceAllocation::identity(&current.allocation),
                    label,
                )?;
            }
            return Ok(current.allocation.ptr());
        }
        if self.runtime.is_capturing()? {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {label}: metadata signature changed during CUDA graph capture; abort recording and eagerly warm the exact replacement signature before capture"
            )));
        }
        let contextual = |stage, error| contextual_error(label, stage, error);
        // Replacements own distinct immutable bytes. Prior readers and graphs
        // retain their owners; final retirement belongs to the native queue.
        let (key, bytes) =
            build().map_err(|error| contextual("encoding replacement metadata", error))?;
        let allocation = NativeDeviceAllocation::allocate_shared(
            &self.runtime,
            bytes.len(),
            onnx_runtime_memory_api::MemoryRole::Workspace { step_scoped: false },
        )
        .map_err(|error| contextual("allocating replacement metadata", error))?;
        // SAFETY: the new owner covers the complete synchronous upload. The
        // builder's Host bytes can be released after htod returns.
        unsafe { self.runtime.htod(&bytes, allocation.ptr()) }
            .map_err(|error| contextual("uploading replacement metadata", error))?;
        self.runtime.staged_warm_cache_mutation(label)?;
        let ptr = allocation.ptr();
        self.current = Some(Arc::new(PreparedDeviceMetadata { key, allocation }));
        Ok(ptr)
    }

    pub(crate) fn device_graph_resource(&self) -> Option<DeviceGraphResource> {
        self.current
            .as_ref()
            .map(|current| NativeDeviceAllocation::device_graph_resource(&current.allocation))
    }

    /// Borrow the prepared binding without introducing another owner or
    /// independently cached pointer/extent. Graph retention uses the resource
    /// method above rather than cloning this allocation into kernel state.
    pub(crate) fn allocation(&self) -> Option<&NativeDeviceAllocation> {
        self.current
            .as_ref()
            .map(|current| current.allocation.as_ref())
    }
}

/// Metadata and its most recent successful launch proof are one publication.
/// Cloning stages a private candidate; failed preparation/consumption does not
/// change the published pair. This is not an execution-concurrency authority.
#[derive(Clone, Debug)]
pub(crate) struct WarmMetadata<M, S> {
    pub(crate) metadata: M,
    pub(crate) signature: Option<S>,
}

impl<M, S> WarmMetadata<M, S> {
    pub(crate) fn new(metadata: M) -> Self {
        Self {
            metadata,
            signature: None,
        }
    }
}

/// Commit only a successfully consumed candidate. Return/drop the old state
/// after unlocking: native retirement may wait for capture or run queue actions.
pub(crate) fn publish_metadata_candidate<T>(
    state: &Mutex<T>,
    candidate: T,
    label: &str,
) -> Result<()> {
    let previous = {
        let mut current = state.lock().map_err(|_| EpError::KernelFailed(format!(
            "cuda_ep {label}: metadata state is poisoned before publication; stop inference and rebuild the provider before preparing again"
        )))?;
        std::mem::replace(&mut *current, candidate)
    };
    drop(previous);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_publication_retires_previous_and_refused_owners_after_unlocking() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct Owner {
            state: std::sync::Weak<Mutex<WarmMetadata<Option<Owner>, u64>>>,
            drops: Arc<AtomicUsize>,
            generation: u64,
        }
        impl Drop for Owner {
            fn drop(&mut self) {
                if let Some(state) = self.state.upgrade() {
                    assert!(
                        !matches!(state.try_lock(), Err(std::sync::TryLockError::WouldBlock)),
                        "native retirement must never run inside the metadata publication lock"
                    );
                }
                self.drops.fetch_add(1, Ordering::Relaxed);
            }
        }
        let state = Arc::new(Mutex::new(WarmMetadata::new(None)));
        let drops = Arc::new(AtomicUsize::new(0));
        let owner = |generation| WarmMetadata {
            metadata: Some(Owner {
                state: Arc::downgrade(&state),
                drops: Arc::clone(&drops),
                generation,
            }),
            signature: Some(generation),
        };
        publish_metadata_candidate(&state, owner(1), "test broadcast metadata").unwrap();
        publish_metadata_candidate(&state, owner(2), "test broadcast metadata").unwrap();
        assert_eq!(drops.load(Ordering::Relaxed), 1);
        {
            let published = state.lock().unwrap();
            assert_eq!(published.signature, Some(2));
            assert_eq!(published.metadata.as_ref().unwrap().generation, 2);
        }
        let _ = std::panic::catch_unwind(|| {
            let _guard = state.lock().unwrap();
            panic!("injected metadata publication panic");
        });
        let failure = publish_metadata_candidate(&state, owner(3), "test broadcast metadata")
            .unwrap_err()
            .to_string();
        assert!(
            failure.contains("test broadcast metadata") && failure.contains("rebuild the provider"),
            "{failure}"
        );
        assert_eq!(
            drops.load(Ordering::Relaxed),
            2,
            "refused candidate must retire after the poisoned lock is released"
        );
        {
            let published = state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            assert_eq!(
                published.signature,
                Some(2),
                "refusal must preserve the old success proof"
            );
            assert_eq!(
                published.metadata.as_ref().unwrap().generation,
                2,
                "refusal must preserve the same metadata/proof pair"
            );
        }
        drop(state);
        assert_eq!(drops.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn staging_metadata_does_not_require_a_cloneable_key() {
        struct Key;
        fn require_clone<T: Clone>() {}
        require_clone::<DeviceMetadataCache<Key>>();
    }

    #[test]
    fn metadata_failure_context_retains_typed_backend_cause() {
        let wrapped = contextual_error(
            "Where broadcast metadata",
            "encoding replacement metadata",
            EpError::Backend {
                operation: "read metadata inputs",
                source: Box::new(std::io::Error::other("metadata input unavailable")),
            },
        );
        let text = wrapped.to_string();
        assert!(
            text.contains("Where broadcast metadata")
                && text.contains("encoding replacement metadata")
                && text.contains("eagerly prepare again")
                && text.contains("metadata input unavailable")
        );
        let mut error: &dyn std::error::Error = &wrapped;
        loop {
            if let Some(cause) = error.downcast_ref::<std::io::Error>() {
                assert_eq!(cause.kind(), std::io::ErrorKind::Other);
                break;
            }
            error = error
                .source()
                .expect("typed original cause must remain in the error chain");
        }
    }

    #[test]
    #[cfg_attr(
        not(feature = "gpu-tests"),
        ignore = "requires CUDA device; enable gpu-tests"
    )]
    fn metadata_hits_are_lazy_and_replacements_preserve_published_entries() {
        let (_provider, runtime) = crate::test_support::maybe_native_runtime()
            .expect("metadata cache requires native CUDA provider backing");
        runtime.set_defer_eager_sync(true);
        let mut cache = DeviceMetadataCache::new(Arc::clone(&runtime));
        let pointer = cache
            .get_or_prepare(
                "test metadata",
                |_| false,
                || Ok((1u64, 7u64.to_ne_bytes().to_vec())),
            )
            .unwrap();
        let identity = cache.device_graph_resource().unwrap().identity();
        let counts = runtime.allocation_counts();
        let transfers = runtime.transfer_counts();
        let syncs = runtime.forced_synchronization_count();
        let mut candidate = cache.clone();
        assert!(Arc::ptr_eq(
            cache.current.as_ref().unwrap(),
            candidate.current.as_ref().unwrap()
        ));
        for _ in 0..8 {
            assert_eq!(
                candidate
                    .get_or_prepare(
                        "test metadata",
                        |key| *key == 1,
                        || panic!("warm metadata must not build a key or payload")
                    )
                    .unwrap(),
                pointer
            );
        }
        assert_eq!(runtime.allocation_counts(), counts);
        assert_eq!(runtime.transfer_counts(), transfers);
        assert_eq!(runtime.forced_synchronization_count(), syncs);

        let failure = candidate
            .get_or_prepare(
                "test metadata",
                |_| false,
                || Err(EpError::KernelFailed("invalid test geometry".into())),
            )
            .unwrap_err()
            .to_string();
        assert!(failure.contains("test metadata") && failure.contains("invalid test geometry"));
        assert_eq!(
            candidate.device_graph_resource().unwrap().identity(),
            identity
        );
        assert_eq!(
            runtime.allocation_counts(),
            counts,
            "encoding refusal must precede native admission"
        );
        assert_eq!(
            runtime.transfer_counts(),
            transfers,
            "encoding refusal must not upload"
        );
        assert_eq!(
            runtime.forced_synchronization_count(),
            syncs,
            "encoding failure must preserve old readers without any explicit drain"
        );

        #[cfg(feature = "gpu-tests")]
        {
            runtime.fail_warm_transaction_at_for_test(1);
            assert!(
                candidate
                    .get_or_prepare(
                        "test metadata",
                        |_| false,
                        || Ok((2, 9u64.to_ne_bytes().to_vec()))
                    )
                    .unwrap_err()
                    .to_string()
                    .contains("injected staged warm-cache failure")
            );
            assert_eq!(
                candidate.device_graph_resource().unwrap().identity(),
                identity
            );
        }
        candidate
            .get_or_prepare(
                "test metadata",
                |_| false,
                || Ok((2, 11u64.to_ne_bytes().to_vec())),
            )
            .unwrap();
        assert_ne!(
            candidate.device_graph_resource().unwrap().identity(),
            identity
        );
        assert_eq!(
            cache.device_graph_resource().unwrap().identity(),
            identity,
            "an uncommitted candidate must not replace the published owner"
        );
        let mut original = [0u8; 8];
        // SAFETY: cache retains the original complete eight-byte owner.
        unsafe { runtime.dtoh(&mut original, pointer).unwrap() };
        assert_eq!(u64::from_ne_bytes(original), 7);

        runtime
            .begin_graph_capture_with_resources(&[], vec![cache.device_graph_resource().unwrap()])
            .unwrap();
        let syncs = runtime.forced_synchronization_count();
        let counts = runtime.allocation_counts();
        let transfers = runtime.transfer_counts();
        assert_eq!(
            cache
                .get_or_prepare(
                    "test metadata",
                    |key| *key == 1,
                    || panic!("capture hit must not encode")
                )
                .unwrap(),
            pointer
        );
        assert!(
            cache
                .get_or_prepare(
                    "test metadata",
                    |_| false,
                    || panic!("capture miss must reject before encoding")
                )
                .unwrap_err()
                .to_string()
                .contains("eagerly warm")
        );
        assert_eq!(runtime.forced_synchronization_count(), syncs);
        assert_eq!(runtime.allocation_counts(), counts);
        assert_eq!(runtime.transfer_counts(), transfers);
        assert_eq!(cache.device_graph_resource().unwrap().identity(), identity);
        runtime.abort_graph_capture().unwrap();
    }
}

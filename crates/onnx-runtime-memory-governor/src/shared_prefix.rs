//! Request admission transactions for physically shared KV prefixes.

use std::ptr::NonNull;

use crate::{DeviceAllocator, MemoryError, SharedDevicePrefix, SharedPrefixCommitInfo};

/// One side of an all-or-private K/V shared-prefix admission transaction.
#[derive(Clone, Copy, Debug)]
pub struct SharedPrefixCommitTarget<'a> {
    pub prefix: &'a dyn SharedDevicePrefix,
    pub ptr: NonNull<u8>,
    pub allocation_bytes: usize,
    pub align: usize,
    pub byte_offset: usize,
}

/// Failure of an atomic K/V shared-prefix admission attempt.
#[derive(Debug, thiserror::Error)]
pub enum SharedPrefixPairCommitError {
    /// No shared mapping remains in either target, so private fallback is safe.
    #[error("shared-prefix pair was refused with no mapping retained: {0}")]
    Refused(MemoryError),
    /// The second map failed and the first could not be removed. The allocation
    /// must not be exposed to kernels or reused as a private fallback.
    #[error(
        "shared-prefix pair commit failed ({commit}) and rollback of the first mapping also failed \
         ({rollback}); the partially converted allocation is not usable"
    )]
    RollbackFailed {
        commit: Box<MemoryError>,
        rollback: Box<MemoryError>,
    },
}

impl SharedPrefixPairCommitError {
    /// Whether neither target retains a mapping from this transaction.
    pub const fn private_fallback_is_safe(&self) -> bool {
        matches!(self, Self::Refused(_))
    }
}

/// Commit K and V shared prefixes as one host-side admission transaction.
///
/// A request must finish this call before enqueueing any kernel that can observe
/// either target. If the first map succeeds and the second fails, the first is
/// decommitted before [`SharedPrefixPairCommitError::Refused`] is returned.
/// Only that error permits the caller to commit private backing and continue.
pub fn commit_shared_prefix_pair(
    allocator: &dyn DeviceAllocator,
    key: SharedPrefixCommitTarget<'_>,
    value: SharedPrefixCommitTarget<'_>,
) -> Result<[SharedPrefixCommitInfo; 2], SharedPrefixPairCommitError> {
    let Some(mapping) = allocator.as_shared_mapping() else {
        return Err(SharedPrefixPairCommitError::Refused(
            MemoryError::InvalidRequest {
                tier: allocator.device().tier.name(),
                requested: 0,
                reason: "allocator does not expose shared mapping",
            },
        ));
    };
    let Some(backing) = allocator.as_virtual_backing() else {
        return Err(SharedPrefixPairCommitError::Refused(
            MemoryError::InvalidRequest {
                tier: allocator.device().tier.name(),
                requested: 0,
                reason: "atomic shared-prefix pairs require virtual backing for rollback",
            },
        ));
    };

    let key_commit = mapping
        .commit_shared_prefix(key.prefix, key.ptr, key.allocation_bytes, key.byte_offset)
        .map_err(SharedPrefixPairCommitError::Refused)?;
    match mapping.commit_shared_prefix(
        value.prefix,
        value.ptr,
        value.allocation_bytes,
        value.byte_offset,
    ) {
        Ok(value_commit) => Ok([key_commit, value_commit]),
        Err(commit) => match backing.decommit_allocation_range(
            key.ptr,
            key.allocation_bytes,
            key.align,
            key.byte_offset,
            key.prefix.mapped_bytes(),
        ) {
            Ok(_) => Err(SharedPrefixPairCommitError::Refused(commit)),
            Err(rollback) => Err(SharedPrefixPairCommitError::RollbackFailed {
                commit: Box::new(commit),
                rollback: Box::new(rollback),
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use crate::{
        AllocationCommitRange, DeviceKey, HostAllocator, SharedMapping, Tier, VirtualBacking,
    };

    use super::*;

    #[derive(Debug)]
    struct PairPrefix;

    impl SharedDevicePrefix for PairPrefix {
        fn device_ptr(&self) -> u64 {
            0
        }

        fn committed_physical_bytes(&self) -> u64 {
            64
        }

        fn mapped_bytes(&self) -> usize {
            64
        }

        fn requested_bytes(&self) -> usize {
            64
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[derive(Debug, Default)]
    struct PairAllocator {
        commit_calls: AtomicUsize,
        decommit_calls: AtomicUsize,
        expose_backing: AtomicBool,
        fail_first: AtomicBool,
        fail_second: AtomicBool,
        fail_rollback: AtomicBool,
    }

    impl PairAllocator {
        fn failure(reason: &str) -> MemoryError {
            MemoryError::AllocationFailed {
                tier: Tier::Device.name(),
                requested: 64,
                reason: reason.to_owned(),
            }
        }
    }

    impl DeviceAllocator for PairAllocator {
        fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
            HostAllocator.allocate(bytes, align)
        }

        unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) {
            unsafe { HostAllocator.deallocate(ptr, bytes, align) };
        }

        fn device(&self) -> DeviceKey {
            DeviceKey::device(0)
        }

        fn as_virtual_backing(&self) -> Option<&dyn VirtualBacking> {
            self.expose_backing.load(Ordering::Relaxed).then_some(self)
        }

        fn as_shared_mapping(&self) -> Option<&dyn SharedMapping> {
            Some(self)
        }
    }

    impl VirtualBacking for PairAllocator {
        fn allocate_committed(
            &self,
            bytes: usize,
            align: usize,
            _committed_ranges: &[std::ops::Range<usize>],
        ) -> Result<NonNull<u8>, MemoryError> {
            self.allocate(bytes, align)
        }

        fn commit_allocation_range(
            &self,
            _ptr: NonNull<u8>,
            _allocation_bytes: usize,
            _align: usize,
            _offset: usize,
            _bytes: usize,
        ) -> Result<(), MemoryError> {
            Ok(())
        }

        fn mapped_bytes_for_allocation_ranges(
            &self,
            ranges: &[AllocationCommitRange],
        ) -> Result<u64, MemoryError> {
            Ok(ranges.iter().map(|range| range.bytes as u64).sum())
        }

        fn mapped_bytes_for_allocation(
            &self,
            bytes: usize,
            _align: usize,
        ) -> Result<u64, MemoryError> {
            Ok(bytes as u64)
        }

        fn decommit_allocation_range(
            &self,
            _ptr: NonNull<u8>,
            _allocation_bytes: usize,
            _align: usize,
            _offset: usize,
            bytes: usize,
        ) -> Result<u64, MemoryError> {
            self.decommit_calls.fetch_add(1, Ordering::Relaxed);
            if self.fail_rollback.load(Ordering::Relaxed) {
                Err(Self::failure("rollback failed"))
            } else {
                Ok(bytes as u64)
            }
        }

        fn allocation_committed_bytes(
            &self,
            _ptr: NonNull<u8>,
            _allocation_bytes: usize,
            _align: usize,
        ) -> usize {
            0
        }
    }

    impl SharedMapping for PairAllocator {
        fn create_shared_prefix(
            &self,
            _bytes: usize,
        ) -> Result<Box<dyn SharedDevicePrefix>, MemoryError> {
            Ok(Box::new(PairPrefix))
        }

        fn incremental_owned_bytes_for_shared_prefix(
            &self,
            _prefix: &dyn SharedDevicePrefix,
        ) -> Result<u64, MemoryError> {
            Ok(0)
        }

        fn commit_shared_prefix(
            &self,
            _prefix: &dyn SharedDevicePrefix,
            _ptr: NonNull<u8>,
            _allocation_bytes: usize,
            _byte_offset: usize,
        ) -> Result<SharedPrefixCommitInfo, MemoryError> {
            let call = self.commit_calls.fetch_add(1, Ordering::Relaxed);
            if (call == 0 && self.fail_first.load(Ordering::Relaxed))
                || (call == 1 && self.fail_second.load(Ordering::Relaxed))
            {
                Err(Self::failure("value map failed"))
            } else {
                Ok(SharedPrefixCommitInfo {
                    additional_owned_bytes: 0,
                    newly_mapped_bytes: 64,
                    granules: 1,
                })
            }
        }
    }

    fn target(prefix: &PairPrefix, ptr: NonNull<u8>) -> SharedPrefixCommitTarget<'_> {
        SharedPrefixCommitTarget {
            prefix,
            ptr,
            allocation_bytes: 128,
            align: 64,
            byte_offset: 0,
        }
    }

    #[test]
    fn rolls_back_first_map_before_private_fallback() {
        let allocator = PairAllocator::default();
        allocator.expose_backing.store(true, Ordering::Relaxed);
        allocator.fail_second.store(true, Ordering::Relaxed);
        let prefix = PairPrefix;
        let (mut key, mut value) = (0u8, 0u8);

        let error = commit_shared_prefix_pair(
            &allocator,
            target(&prefix, NonNull::from(&mut key)),
            target(&prefix, NonNull::from(&mut value)),
        )
        .expect_err("second map must fail");

        assert!(error.private_fallback_is_safe());
        assert_eq!(allocator.commit_calls.load(Ordering::Relaxed), 2);
        assert_eq!(allocator.decommit_calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn blocks_fallback_when_rollback_fails() {
        let allocator = PairAllocator::default();
        allocator.expose_backing.store(true, Ordering::Relaxed);
        allocator.fail_second.store(true, Ordering::Relaxed);
        allocator.fail_rollback.store(true, Ordering::Relaxed);
        let prefix = PairPrefix;
        let (mut key, mut value) = (0u8, 0u8);

        let error = commit_shared_prefix_pair(
            &allocator,
            target(&prefix, NonNull::from(&mut key)),
            target(&prefix, NonNull::from(&mut value)),
        )
        .expect_err("rollback must fail");

        assert!(matches!(
            &error,
            SharedPrefixPairCommitError::RollbackFailed { .. }
        ));
        assert!(!error.private_fallback_is_safe());
    }

    #[test]
    fn commits_both_targets_without_rollback() {
        let allocator = PairAllocator::default();
        allocator.expose_backing.store(true, Ordering::Relaxed);
        let prefix = PairPrefix;
        let (mut key, mut value) = (0u8, 0u8);

        let commits = commit_shared_prefix_pair(
            &allocator,
            target(&prefix, NonNull::from(&mut key)),
            target(&prefix, NonNull::from(&mut value)),
        )
        .expect("both maps succeed");

        assert_eq!(commits[0].newly_mapped_bytes, 64);
        assert_eq!(commits[1].newly_mapped_bytes, 64);
        assert_eq!(allocator.commit_calls.load(Ordering::Relaxed), 2);
        assert_eq!(allocator.decommit_calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn first_map_refusal_needs_no_rollback() {
        let allocator = PairAllocator::default();
        allocator.expose_backing.store(true, Ordering::Relaxed);
        allocator.fail_first.store(true, Ordering::Relaxed);
        let prefix = PairPrefix;
        let (mut key, mut value) = (0u8, 0u8);

        let error = commit_shared_prefix_pair(
            &allocator,
            target(&prefix, NonNull::from(&mut key)),
            target(&prefix, NonNull::from(&mut value)),
        )
        .expect_err("first map must fail");

        assert!(error.private_fallback_is_safe());
        assert_eq!(allocator.commit_calls.load(Ordering::Relaxed), 1);
        assert_eq!(allocator.decommit_calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn refuses_before_mapping_when_rollback_capability_is_absent() {
        let allocator = PairAllocator::default();
        let prefix = PairPrefix;
        let (mut key, mut value) = (0u8, 0u8);

        let error = commit_shared_prefix_pair(
            &allocator,
            target(&prefix, NonNull::from(&mut key)),
            target(&prefix, NonNull::from(&mut value)),
        )
        .expect_err("pair admission requires rollback support");

        assert!(error.private_fallback_is_safe());
        assert!(error.to_string().contains("require virtual backing"));
        assert_eq!(allocator.commit_calls.load(Ordering::Relaxed), 0);
    }
}

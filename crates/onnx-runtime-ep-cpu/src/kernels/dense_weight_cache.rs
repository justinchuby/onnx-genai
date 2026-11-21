//! Dense derived weights: cache membership is a policy, ownership holds the grant.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use onnx_runtime_ep_api::{EpError, Result};
use onnx_runtime_memory_api::MemoryError;

use super::block_quantized_matmul::DenseWeightCacheKey;
use crate::DerivedWeightMemory;
use crate::resources::{DenseWeightGrant, WeightAdmission};

pub(crate) const DEFAULT_DENSE_WEIGHT_CACHE_BYTES: usize = 256 * 1024 * 1024;
const DENSE_WEIGHT_CACHE_BYTES_ENV: &str = "ONNX_GENAI_CPU_BLOCK_QUANT_CACHE_BYTES";

pub(crate) fn configured_dense_weight_cache_bytes() -> usize {
    static BYTES: OnceLock<usize> = OnceLock::new();
    *BYTES.get_or_init(|| {
        parse_dense_weight_cache_bytes(std::env::var(DENSE_WEIGHT_CACHE_BYTES_ENV).ok().as_deref())
    })
}

pub(super) fn parse_dense_weight_cache_bytes(value: Option<&str>) -> usize {
    value
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_DENSE_WEIGHT_CACHE_BYTES)
}

pub(super) struct DenseWeight {
    values: Vec<f32>,
    // Declaration order frees the payload before returning its grant.
    _grant: DenseWeightGrant,
}

impl DenseWeight {
    pub(super) fn as_slice(&self) -> &[f32] {
        &self.values
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DenseWeightCacheStatus {
    Hit,
    MissStored,
    MissNotStored,
}

pub(super) struct DenseWeightSpec<'a> {
    key: Option<&'a DenseWeightCacheKey>,
    bytes: usize,
}

impl<'a> DenseWeightSpec<'a> {
    pub(super) fn cached(key: &'a DenseWeightCacheKey) -> Result<Self> {
        Ok(Self {
            key: Some(key),
            bytes: key.dense_bytes()?,
        })
    }

    pub(super) fn transient(bytes: usize) -> Self {
        Self { key: None, bytes }
    }
}

pub(super) struct LoadedDenseWeights {
    pub(super) weights: Vec<Arc<DenseWeight>>,
    pub(super) statuses: Vec<DenseWeightCacheStatus>,
}

pub(super) enum DenseWeightLoad {
    Ready(LoadedDenseWeights),
    Declined(MemoryError),
}

pub(super) struct DenseWeightCache {
    max_bytes: usize,
    memory: DerivedWeightMemory,
    inner: Mutex<CacheState>,
    // Cold fills serialize separately: accountant callbacks and decoding never
    // run under the payload lock. Warm hits do not acquire this gate.
    fill: Mutex<()>,
}

#[derive(Default)]
struct CacheState {
    used_bytes: usize,
    tick: u64,
    entries: HashMap<DenseWeightCacheKey, CacheEntry>,
    #[cfg(test)]
    hits: usize,
    #[cfg(test)]
    builds: usize,
}

struct CacheEntry {
    value: Arc<DenseWeight>,
    bytes: usize,
    last_used: u64,
}

impl Default for DenseWeightCache {
    fn default() -> Self {
        Self::new(
            DerivedWeightMemory::default(),
            configured_dense_weight_cache_bytes(),
        )
    }
}

impl DenseWeightCache {
    pub(super) fn new(memory: DerivedWeightMemory, max_bytes: usize) -> Self {
        Self {
            max_bytes,
            memory,
            inner: Mutex::new(CacheState::default()),
            fill: Mutex::new(()),
        }
    }

    #[cfg(test)]
    pub(super) fn with_limit(limit: usize) -> Self {
        Self::new(DerivedWeightMemory::default(), limit)
    }

    #[cfg(test)]
    pub(super) fn with_limit_and_memory(limit: usize, memory: DerivedWeightMemory) -> Self {
        Self::new(memory, limit)
    }

    pub(super) fn byte_limit(&self) -> usize {
        self.max_bytes
    }

    fn lookup(&self, specs: &[DenseWeightSpec<'_>]) -> Vec<Option<Arc<DenseWeight>>> {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let values: Vec<_> = specs
            .iter()
            .map(|spec| {
                spec.key
                    .and_then(|key| inner.entries.get(key))
                    .map(|entry| Arc::clone(&entry.value))
            })
            .collect();
        if values.iter().all(Option::is_some) {
            for spec in specs {
                inner.tick = inner.tick.wrapping_add(1);
                let tick = inner.tick;
                if let Some(entry) = spec.key.and_then(|key| inner.entries.get_mut(key)) {
                    entry.last_used = tick;
                }
            }
            #[cfg(test)]
            {
                inner.hits = inner.hits.saturating_add(values.len());
            }
        }
        values
    }

    fn ready_hits(values: Vec<Option<Arc<DenseWeight>>>) -> DenseWeightLoad {
        let statuses = vec![DenseWeightCacheStatus::Hit; values.len()];
        DenseWeightLoad::Ready(LoadedDenseWeights {
            weights: values
                .into_iter()
                .map(|value| value.expect("complete dense-cache hit set"))
                .collect(),
            statuses,
        })
    }

    /// Preflight every missing projection together, including nonconstant
    /// weights (key=None). A refusal does not invoke any builder.
    pub(super) fn load_group(
        &self,
        specs: &[DenseWeightSpec<'_>],
        mut build: impl FnMut(usize, &mut [f32]) -> Result<()>,
    ) -> Result<DenseWeightLoad> {
        for (index, spec) in specs.iter().enumerate() {
            if spec.bytes > isize::MAX as usize
                || !spec.bytes.is_multiple_of(std::mem::size_of::<f32>())
            {
                return Err(EpError::KernelFailed(format!(
                    "dense projection {index} requests {} bytes, which is not an addressable f32 buffer; use an f32-aligned byte count no larger than isize::MAX",
                    spec.bytes
                )));
            }
            if let Some(key) = spec.key
                && specs[..index]
                    .iter()
                    .any(|earlier| earlier.key == Some(key))
            {
                return Err(EpError::KernelFailed(format!(
                    "dense weight projection group repeats cache key at index {index}; give every projection one unique identity"
                )));
            }
        }
        let keyed = specs.iter().any(|spec| spec.key.is_some());
        let mut values = if keyed {
            self.lookup(specs)
        } else {
            vec![None; specs.len()]
        };
        if values.iter().all(Option::is_some) {
            return Ok(Self::ready_hits(values));
        }
        let _fill = keyed.then(|| {
            self.fill
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
        });
        if keyed {
            values = self.lookup(specs);
        }
        if values.iter().all(Option::is_some) {
            return Ok(Self::ready_hits(values));
        }
        let required = specs.iter().zip(&values).try_fold(0usize, |total, (spec, value)| {
            if value.is_some() { Ok(total) } else {
                total.checked_add(spec.bytes).ok_or_else(|| EpError::KernelFailed("dense weight projection-group byte count overflow; use addressable projection dimensions".into()))
            }
        })?;
        let max_bytes = self.byte_limit();
        let retained_group_bytes = specs
            .iter()
            .filter(|spec| spec.key.is_some())
            .try_fold(0usize, |total, spec| total.checked_add(spec.bytes));
        let retain_new =
            max_bytes != 0 && retained_group_bytes.is_some_and(|bytes| bytes <= max_bytes);
        let cache_growth = specs
            .iter()
            .zip(&values)
            .filter(|(spec, value)| value.is_none() && spec.key.is_some() && retain_new)
            .try_fold(0usize, |total, (spec, _)| {
                total.checked_add(spec.bytes).ok_or_else(|| {
                    EpError::KernelFailed("dense weight cache growth byte count overflow".into())
                })
            })?;
        // Release evicted payloads outside the data lock: the last Arc may
        // return capacity through a third-party accountant callback.
        let evicted = if keyed {
            let mut inner = self
                .inner
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let mut evicted = Vec::new();
            while inner.used_bytes > max_bytes.saturating_sub(cache_growth) {
                let Some(key) = oldest_unused(&inner, specs, false) else {
                    break;
                };
                evicted.push(remove_entry(&mut inner, &key));
            }
            evicted
        } else {
            Vec::new()
        };
        drop(evicted);
        let mut grant = loop {
            match self.memory.try_grant(required)? {
                WeightAdmission::Granted(grant) => break grant,
                WeightAdmission::Declined(reason) => {
                    let victim = {
                        let mut inner = self
                            .inner
                            .lock()
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        oldest_unused(&inner, specs, true).map(|key| remove_entry(&mut inner, &key))
                    };
                    if let Some(victim) = victim {
                        drop(victim);
                    } else {
                        return Ok(DenseWeightLoad::Declined(reason));
                    }
                }
            }
        };
        let mut statuses = vec![DenseWeightCacheStatus::Hit; specs.len()];
        for (index, spec) in specs.iter().enumerate() {
            if values[index].is_some() {
                continue;
            }
            let child = grant.split_off(spec.bytes)?;
            let elements = spec.bytes / std::mem::size_of::<f32>();
            let mut payload = Vec::new();
            payload.try_reserve_exact(elements).map_err(|source| EpError::Memory(MemoryError::AllocationFailed {
                tier: "host", requested: spec.bytes as u64,
                reason: format!("dense projection {index} could not allocate its admitted f32 buffer: {source}; lower concurrent weight demand or free Host memory"),
            }))?;
            if payload.capacity().checked_mul(std::mem::size_of::<f32>()) != Some(spec.bytes) {
                return Err(EpError::KernelFailed(format!(
                    "dense projection {index} allocator returned capacity {} for {} bytes; use an allocator honoring exact f32 capacity",
                    payload.capacity(),
                    spec.bytes
                )));
            }
            payload.resize(elements, 0.0);
            build(index, &mut payload)?;
            values[index] = Some(Arc::new(DenseWeight {
                values: payload,
                _grant: child,
            }));
            statuses[index] = if spec.key.is_some() && retain_new {
                DenseWeightCacheStatus::MissStored
            } else {
                DenseWeightCacheStatus::MissNotStored
            };
            #[cfg(test)]
            {
                let mut inner = self
                    .inner
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                inner.builds = inner.builds.saturating_add(1);
            }
        }
        let weights: Vec<_> = values
            .into_iter()
            .map(|value| value.expect("every dense projection hit or built successfully"))
            .collect();
        if keyed {
            let mut inner = self
                .inner
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            for ((spec, value), status) in specs.iter().zip(&weights).zip(&statuses) {
                inner.tick = inner.tick.wrapping_add(1);
                let tick = inner.tick;
                match status {
                    DenseWeightCacheStatus::Hit => {
                        if let Some(entry) = spec.key.and_then(|key| inner.entries.get_mut(key)) {
                            entry.last_used = tick;
                        }
                        #[cfg(test)]
                        {
                            inner.hits = inner.hits.saturating_add(1);
                        }
                    }
                    DenseWeightCacheStatus::MissStored => {
                        let key = spec
                            .key
                            .expect("only keyed projections can be retained")
                            .clone();
                        inner.used_bytes =
                            inner.used_bytes.checked_add(spec.bytes).ok_or_else(|| {
                                EpError::KernelFailed(
                                    "dense cache resident byte count overflow".into(),
                                )
                            })?;
                        inner.entries.insert(
                            key,
                            CacheEntry {
                                value: Arc::clone(value),
                                bytes: spec.bytes,
                                last_used: tick,
                            },
                        );
                    }
                    DenseWeightCacheStatus::MissNotStored => {}
                }
            }
        }
        Ok(DenseWeightLoad::Ready(LoadedDenseWeights {
            weights,
            statuses,
        }))
    }

    #[cfg(test)]
    pub(super) fn stats(&self) -> (usize, usize) {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        (inner.entries.len(), inner.used_bytes)
    }

    #[cfg(test)]
    pub(super) fn activity(&self) -> (usize, usize) {
        let inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        (inner.hits, inner.builds)
    }

    #[cfg(test)]
    pub(super) fn test_get_or_insert_with(
        &self,
        key: &DenseWeightCacheKey,
        build: impl FnOnce() -> Result<Vec<f32>>,
    ) -> Result<(Arc<DenseWeight>, DenseWeightCacheStatus)> {
        let mut build = Some(build);
        match self.load_group(&[DenseWeightSpec::cached(key)?], |_, output| {
            let values = build.take().expect("single fixture builder")()?;
            if values.len() != output.len() {
                return Err(EpError::KernelFailed(
                    "dense cache fixture output length mismatch".into(),
                ));
            }
            output.copy_from_slice(&values);
            Ok(())
        })? {
            DenseWeightLoad::Ready(mut loaded) => {
                Ok((loaded.weights.remove(0), loaded.statuses.remove(0)))
            }
            DenseWeightLoad::Declined(reason) => Err(EpError::Memory(reason)),
        }
    }
}

fn oldest_unused(
    inner: &CacheState,
    specs: &[DenseWeightSpec<'_>],
    inactive_only: bool,
) -> Option<DenseWeightCacheKey> {
    inner
        .entries
        .iter()
        .filter(|(key, entry)| {
            !specs.iter().any(|spec| spec.key == Some(*key))
                && (!inactive_only || Arc::strong_count(&entry.value) == 1)
        })
        .min_by_key(|(_, entry)| entry.last_used)
        .map(|(key, _)| key.clone())
}

fn remove_entry(inner: &mut CacheState, key: &DenseWeightCacheKey) -> CacheEntry {
    let entry = inner
        .entries
        .remove(key)
        .expect("selected dense-cache victim exists");
    inner.used_bytes = inner
        .used_bytes
        .checked_sub(entry.bytes)
        .expect("dense-cache residency accounts every retained entry");
    entry
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::block_quantized_matmul::{
        BlockFormat, DenseWeightIdentity, packed_tensor_bytes,
    };
    use crate::kernels::testutil::Owned;
    use onnx_runtime_memory_api::{MemoryRole, Tier};
    use onnx_runtime_memory_governor::{
        HolderId, LeaseAccounting, LeaseLedger, LedgerGovernor, MemoryAuthorityId, MemoryGovernor,
        MemoryLease,
    };
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Barrier, Weak};

    fn key(packed: &Owned) -> Arc<DenseWeightCacheKey> {
        let view = packed.view();
        DenseWeightIdentity::default()
            .resolve(&view, BlockFormat::Mxfp4, 4, 1, 0, None, || {
                packed_tensor_bytes(&view)
            })
            .unwrap()
            .key
    }
    fn cached(key: &DenseWeightCacheKey) -> DenseWeightSpec<'_> {
        DenseWeightSpec {
            key: Some(key),
            bytes: 16,
        }
    }
    fn memory(limit: u64) -> (LedgerGovernor, DerivedWeightMemory) {
        let governor = LedgerGovernor::new(LeaseLedger::new(0, limit, 0));
        let source = DerivedWeightMemory::governed(&governor, HolderId::new(13)).unwrap();
        (governor, source)
    }
    fn ready(loaded: DenseWeightLoad) -> LoadedDenseWeights {
        match loaded {
            DenseWeightLoad::Ready(ready) => ready,
            DenseWeightLoad::Declined(reason) => panic!("expected grant: {reason}"),
        }
    }

    #[test]
    fn invalid_buffer_sizes_are_rejected_before_any_builder_or_grant() {
        let cache = DenseWeightCache::default();
        for bytes in [3, isize::MAX as usize + 1] {
            let failure = cache
                .load_group(&[DenseWeightSpec { key: None, bytes }], |_, _| {
                    panic!("invalid buffer must not be allocated")
                })
                .err()
                .expect("invalid f32 buffer size");
            assert!(failure.to_string().contains("addressable f32 buffer"));
        }
    }

    #[test]
    fn separate_kernel_caches_share_capacity_and_evicted_active_arcs_keep_the_charge() {
        let p1 = Owned::u8(&[4], &[1; 4]);
        let p2 = Owned::u8(&[4], &[2; 4]);
        let k1 = key(&p1);
        let k2 = key(&p2);
        let (governor, source) = memory(16);
        let first = DenseWeightCache::new(source.clone(), 16);
        let second = DenseWeightCache::new(source, 1024);
        let live = ready(
            first
                .load_group(&[cached(&k1)], |_, output| {
                    output.fill(1.0);
                    Ok(())
                })
                .unwrap(),
        );
        assert_eq!(governor.used(Tier::Host), 16);
        assert!(matches!(
            first
                .load_group(&[cached(&k2)], |_, _| panic!(
                    "active eviction must not create grant headroom"
                ))
                .unwrap(),
            DenseWeightLoad::Declined(_)
        ));
        assert_eq!(
            first.stats(),
            (0, 0),
            "LRU membership is gone while caller ownership remains"
        );
        assert_eq!(live.weights[0].as_slice(), &[1.0; 4]);
        assert_eq!(governor.used(Tier::Host), 16);
        drop(first);
        let mut builds = 0;
        let denied = second
            .load_group(&[cached(&k2)], |_, output| {
                builds += 1;
                output.fill(2.0);
                Ok(())
            })
            .unwrap();
        assert!(matches!(
            denied,
            DenseWeightLoad::Declined(MemoryError::TierExhausted { requested: 16, .. })
        ));
        assert_eq!(builds, 0, "refusal must precede dense allocation");
        assert_eq!(
            governor.used(Tier::Host),
            16,
            "eviction must not return a caller's live payload"
        );
        drop(live);
        assert_eq!(governor.used(Tier::Host), 0);
        let loaded = ready(
            second
                .load_group(&[cached(&k2)], |_, output| {
                    output.fill(2.0);
                    Ok(())
                })
                .unwrap(),
        );
        drop(loaded);
        assert_eq!(
            governor.used(Tier::Host),
            16,
            "retained data still owns its exact grant"
        );
        drop(second);
        assert_eq!(governor.used(Tier::Host), 0);
    }

    #[test]
    fn complete_projection_group_is_preflighted_before_any_builder() {
        let (governor, source) = memory(31);
        let cache = DenseWeightCache::new(source, 1024);
        let specs = [
            DenseWeightSpec {
                key: None,
                bytes: 16,
            },
            DenseWeightSpec {
                key: None,
                bytes: 16,
            },
        ];
        let mut builds = 0;
        assert!(matches!(
            cache
                .load_group(&specs, |_, output| {
                    builds += 1;
                    output.fill(1.0);
                    Ok(())
                })
                .unwrap(),
            DenseWeightLoad::Declined(MemoryError::TierExhausted { requested: 32, .. })
        ));
        assert_eq!(builds, 0);
        assert_eq!(governor.used(Tier::Host), 0);
        assert_eq!(cache.stats(), (0, 0));
        let (governor, source) = memory(32);
        let cache = DenseWeightCache::new(source, 1024);
        let mut loaded = ready(
            cache
                .load_group(&specs, |_, output| {
                    output.fill(1.0);
                    Ok(())
                })
                .unwrap(),
        );
        assert_eq!(governor.used(Tier::Host), 32);
        drop(loaded.weights.pop());
        assert_eq!(
            governor.used(Tier::Host),
            16,
            "partitioned grant returns exactly the released projection"
        );
        drop(loaded);
        assert_eq!(governor.used(Tier::Host), 0);
    }

    #[test]
    fn partial_rekey_replaces_only_missing_projection_under_an_exact_group_budget() {
        let p1 = Owned::u8(&[4], &[1; 4]);
        let p2 = Owned::u8(&[4], &[2; 4]);
        let p3 = Owned::u8(&[4], &[3; 4]);
        let k1 = key(&p1);
        let k2 = key(&p2);
        let k3 = key(&p3);
        let (governor, source) = memory(32);
        let cache = DenseWeightCache {
            max_bytes: 32,
            ..DenseWeightCache::new(source, 1024)
        };
        let first = ready(
            cache
                .load_group(&[cached(&k1), cached(&k2)], |index, output| {
                    output.fill(index as f32 + 1.0);
                    Ok(())
                })
                .unwrap(),
        );
        let retained = Arc::clone(&first.weights[0]);
        drop(first);
        let mut builds = 0;
        let second = ready(
            cache
                .load_group(&[cached(&k1), cached(&k3)], |_, output| {
                    builds += 1;
                    output.fill(3.0);
                    Ok(())
                })
                .unwrap(),
        );
        assert_eq!(builds, 1);
        assert!(Arc::ptr_eq(&retained, &second.weights[0]));
        assert_eq!(
            second.statuses,
            [
                DenseWeightCacheStatus::Hit,
                DenseWeightCacheStatus::MissStored
            ]
        );
        assert_eq!(governor.used(Tier::Host), 32);
        assert_eq!(cache.stats(), (2, 32));
        let warm = ready(
            cache
                .load_group(&[cached(&k1), cached(&k3)], |_, _| {
                    panic!("warm group must not rebuild")
                })
                .unwrap(),
        );
        assert!(
            warm.statuses
                .iter()
                .all(|status| *status == DenseWeightCacheStatus::Hit)
        );
    }

    #[test]
    fn failed_group_builder_refunds_all_slices_and_publishes_no_partial_group() {
        let p1 = Owned::u8(&[4], &[1; 4]);
        let p2 = Owned::u8(&[4], &[2; 4]);
        let k1 = key(&p1);
        let k2 = key(&p2);
        let (governor, source) = memory(32);
        let cache = DenseWeightCache::new(source, 1024);
        let failure = cache
            .load_group(&[cached(&k1), cached(&k2)], |index, output| {
                if index == 0 {
                    output.fill(1.0);
                    Ok(())
                } else {
                    Err(EpError::KernelFailed(
                        "fc2 projection decoder rejected the test block".into(),
                    ))
                }
            })
            .err()
            .expect("second projection must fail");
        assert!(failure.to_string().contains("fc2 projection"));
        assert_eq!(governor.used(Tier::Host), 0);
        assert_eq!(cache.stats(), (0, 0));
    }

    #[test]
    fn simultaneous_group_misses_are_single_flight_and_do_not_overclaim() {
        let p1 = Owned::u8(&[4], &[1; 4]);
        let p2 = Owned::u8(&[4], &[2; 4]);
        let k1 = key(&p1);
        let k2 = key(&p2);
        let (governor, source) = memory(32);
        let cache = Arc::new(DenseWeightCache::new(source, 1024));
        let builds = AtomicUsize::new(0);
        let barrier = Barrier::new(4);
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    barrier.wait();
                    let loaded = ready(
                        cache
                            .load_group(&[cached(&k1), cached(&k2)], |index, output| {
                                builds.fetch_add(1, Ordering::Relaxed);
                                output.fill(index as f32 + 1.0);
                                Ok(())
                            })
                            .unwrap(),
                    );
                    assert_eq!(loaded.weights[1].as_slice(), &[2.0; 4]);
                });
            }
        });
        assert_eq!(builds.load(Ordering::Relaxed), 2);
        assert_eq!(governor.used(Tier::Host), 32);
    }

    #[derive(Debug)]
    struct ObservingBooks {
        governor: LedgerGovernor,
        cache: Mutex<Weak<DenseWeightCache>>,
        blocked: AtomicBool,
        observations: AtomicUsize,
    }
    impl ObservingBooks {
        fn observe(&self) {
            if let Some(cache) = self.cache.lock().unwrap().upgrade() {
                self.observations.fetch_add(1, Ordering::Relaxed);
                if cache.inner.try_lock().is_err() {
                    self.blocked.store(true, Ordering::Relaxed);
                }
            }
        }
    }
    impl LeaseAccounting for ObservingBooks {
        fn authority_id(&self) -> Option<MemoryAuthorityId> {
            Some(self.governor.authority_id(Tier::Host))
        }
        fn release(&self, tier: Tier, bytes: u64) {
            self.observe();
            LeaseAccounting::release(self.governor.ledger().as_ref(), tier, bytes);
        }
        fn try_claim(
            &self,
            tier: Tier,
            bytes: u64,
            role: MemoryRole,
        ) -> std::result::Result<(), MemoryError> {
            self.observe();
            self.governor.ledger().try_claim(tier, bytes, role)
        }
    }
    struct ObservingGovernor(Arc<ObservingBooks>);
    impl MemoryGovernor for ObservingGovernor {
        fn authority_id(&self, tier: Tier) -> MemoryAuthorityId {
            self.0.governor.authority_id(tier)
        }
        fn reserve(
            &self,
            tier: Tier,
            bytes: u64,
            role: MemoryRole,
            holder: HolderId,
        ) -> std::result::Result<MemoryLease, MemoryError> {
            self.0.governor.ledger().try_claim(tier, bytes, role)?;
            Ok(MemoryLease::new(tier, bytes, role, holder, self.0.clone()))
        }
        fn available(&self, tier: Tier) -> u64 {
            self.0.governor.available(tier)
        }
        fn used(&self, tier: Tier) -> u64 {
            self.0.governor.used(tier)
        }
    }
    #[test]
    fn accountant_can_observe_cache_during_claim_and_inactive_reclamation() {
        let books = Arc::new(ObservingBooks {
            governor: LedgerGovernor::new(LeaseLedger::new(0, 16, 0)),
            cache: Mutex::new(Weak::new()),
            blocked: AtomicBool::new(false),
            observations: AtomicUsize::new(0),
        });
        let source =
            DerivedWeightMemory::governed(&ObservingGovernor(books.clone()), HolderId::new(13))
                .unwrap();
        let cache = Arc::new(DenseWeightCache::new(source, 1024));
        *books.cache.lock().unwrap() = Arc::downgrade(&cache);
        let p1 = Owned::u8(&[4], &[1; 4]);
        let p2 = Owned::u8(&[4], &[2; 4]);
        let k1 = key(&p1);
        let k2 = key(&p2);
        drop(ready(
            cache
                .load_group(&[cached(&k1)], |_, output| {
                    output.fill(1.0);
                    Ok(())
                })
                .unwrap(),
        ));
        drop(ready(
            cache
                .load_group(&[cached(&k2)], |_, output| {
                    output.fill(2.0);
                    Ok(())
                })
                .unwrap(),
        ));
        assert_eq!(cache.stats(), (1, 16));
        assert!(books.observations.load(Ordering::Relaxed) >= 4);
        assert!(
            !books.blocked.load(Ordering::Relaxed),
            "accounting callbacks must run outside the payload lock"
        );
        let before = books.observations.load(Ordering::Relaxed);
        drop(ready(
            cache
                .load_group(&[cached(&k2)], |_, _| panic!("warm hit must not rebuild"))
                .unwrap(),
        ));
        assert_eq!(
            books.observations.load(Ordering::Relaxed),
            before,
            "warm hits must not consult the grant source"
        );
    }
}

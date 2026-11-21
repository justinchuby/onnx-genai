//! Dense derived weights: cache membership is a policy, ownership holds the grant.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Mutex, OnceLock};

use onnx_runtime_ep_api::{EpError, Result};
use onnx_runtime_memory_api::MemoryError;

use crate::DerivedWeightMemory;
use crate::resources::{DerivedWeightGrant, IdleWeightCache, WeightAdmission};

pub(super) trait DenseWeightKey: Clone + Eq + Hash + Send + Sync + 'static {
    fn dense_bytes(&self) -> Result<usize>;
}

pub(crate) const DEFAULT_DENSE_WEIGHT_CACHE_BYTES: usize = 256 * 1024 * 1024;
const DENSE_WEIGHT_CACHE_BYTES_ENV: &str = "ONNX_GENAI_CPU_DENSE_WEIGHT_CACHE_BYTES";

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
    _grant: DerivedWeightGrant,
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

pub(super) struct DenseWeightSpec<'a, K> {
    key: Option<&'a K>,
    bytes: usize,
    build_peak_bytes: usize,
}

impl<'a, K: DenseWeightKey> DenseWeightSpec<'a, K> {
    pub(super) fn cached(key: &'a K) -> Result<Self> {
        let bytes = key.dense_bytes()?;
        Ok(Self {
            key: Some(key),
            bytes,
            build_peak_bytes: bytes,
        })
    }

    pub(super) fn transient(bytes: usize) -> Self {
        Self {
            key: None,
            bytes,
            build_peak_bytes: bytes,
        }
    }
    /// Upper bound on simultaneously live f32 weight buffers during building.
    /// Only payload bytes remain charged after the operation completes.
    pub(super) fn with_build_peak(mut self, bytes: usize) -> Self {
        self.build_peak_bytes = bytes;
        self
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

pub(super) struct DenseWeightCache<K> {
    max_bytes: usize,
    memory: DerivedWeightMemory,
    inner: Arc<Mutex<CacheState<K>>>,
    // Cold fills serialize separately: accountant callbacks and decoding never
    // run under the payload lock. Warm hits do not acquire this gate.
    fill: Mutex<()>,
}

struct CacheState<K> {
    used_bytes: usize,
    tick: u64,
    entries: HashMap<K, CacheEntry>,
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

impl<K> Default for CacheState<K> {
    fn default() -> Self {
        Self {
            used_bytes: 0,
            tick: 0,
            entries: HashMap::new(),
            #[cfg(test)]
            hits: 0,
            #[cfg(test)]
            builds: 0,
        }
    }
}

impl<K: DenseWeightKey> Default for DenseWeightCache<K> {
    fn default() -> Self {
        Self::new(
            DerivedWeightMemory::default(),
            configured_dense_weight_cache_bytes(),
        )
    }
}

impl<K: DenseWeightKey> DenseWeightCache<K> {
    pub(super) fn new(memory: DerivedWeightMemory, max_bytes: usize) -> Self {
        let inner = Arc::new(Mutex::new(CacheState::default()));
        if max_bytes != 0 {
            memory.register_idle_cache(&inner);
        }
        Self {
            max_bytes,
            memory,
            inner,
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

    fn lookup(&self, specs: &[DenseWeightSpec<'_, K>]) -> Vec<Option<Arc<DenseWeight>>> {
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
        specs: &[DenseWeightSpec<'_, K>],
        mut build: impl FnMut(usize, &mut [f32]) -> Result<()>,
    ) -> Result<DenseWeightLoad> {
        self.load_group_owned(specs, |index| {
            let elements = specs[index].bytes / std::mem::size_of::<f32>();
            let mut payload = super::decoded_projection::zeroed_f32(
                "dense projection",
                "admitted weight payload",
                elements,
            )?;
            build(index, &mut payload)?;
            Ok(payload)
        })
    }

    /// Preflight construction peaks before any builder runs. Builders return
    /// exactly sized final payloads; unused scratch allowance returns at the
    /// end of this operation while each live payload keeps only its own grant.
    pub(super) fn load_group_owned(
        &self,
        specs: &[DenseWeightSpec<'_, K>],
        mut build: impl FnMut(usize) -> Result<Vec<f32>>,
    ) -> Result<DenseWeightLoad> {
        for (index, spec) in specs.iter().enumerate() {
            if spec.build_peak_bytes < spec.bytes {
                return Err(EpError::KernelFailed(format!(
                    "dense projection {index} build peak {} is below its {}-byte payload; account for every simultaneously live f32 weight buffer",
                    spec.build_peak_bytes, spec.bytes
                )));
            }
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
                total.checked_add(spec.build_peak_bytes).ok_or_else(|| EpError::KernelFailed("dense weight projection-group byte count overflow; use addressable projection dimensions".into()))
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
        let mut grant = match self.memory.try_grant(required)? {
            WeightAdmission::Granted(grant) => grant,
            WeightAdmission::Declined(reason) => return Ok(DenseWeightLoad::Declined(reason)),
        };
        let mut statuses = vec![DenseWeightCacheStatus::Hit; specs.len()];
        for (index, spec) in specs.iter().enumerate() {
            if values[index].is_some() {
                continue;
            }
            let child = grant.split_off(spec.bytes)?;
            let payload = build(index)?;
            if payload.len().checked_mul(std::mem::size_of::<f32>()) != Some(spec.bytes)
                || payload.capacity().checked_mul(std::mem::size_of::<f32>()) != Some(spec.bytes)
            {
                return Err(EpError::KernelFailed(format!(
                    "dense projection {index} builder returned length {} and capacity {} for {} admitted bytes; return an exactly sized f32 payload",
                    payload.len(),
                    payload.capacity(),
                    spec.bytes
                )));
            }
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
    pub(super) fn test_get(&self, key: &K) -> Option<Arc<DenseWeight>> {
        self.inner
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .entries
            .get(key)
            .map(|entry| Arc::clone(&entry.value))
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
        key: &K,
        build: impl FnOnce() -> Result<Vec<f32>>,
    ) -> Result<(Arc<DenseWeight>, DenseWeightCacheStatus)> {
        let mut build = Some(build);
        match self.load_group_owned(&[DenseWeightSpec::cached(key)?], |_| {
            build.take().expect("single fixture builder")()
        })? {
            DenseWeightLoad::Ready(mut loaded) => {
                Ok((loaded.weights.remove(0), loaded.statuses.remove(0)))
            }
            DenseWeightLoad::Declined(reason) => Err(EpError::Memory(reason)),
        }
    }
}

impl<K: DenseWeightKey> IdleWeightCache for Mutex<CacheState<K>> {
    fn reclaim_idle(&self, want: usize) -> Result<usize> {
        let (retired, released) = {
            let mut inner = self
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let mut retired = Vec::new();
            let mut released = 0usize;
            while released < want {
                let Some(key) = oldest_unused(&inner, &[], true) else {
                    break;
                };
                let entry = remove_entry(&mut inner, &key);
                released += entry.bytes;
                retired.push(entry);
            }
            (retired, released)
        };
        drop(retired);
        Ok(released)
    }
}

fn oldest_unused<K: DenseWeightKey>(
    inner: &CacheState<K>,
    specs: &[DenseWeightSpec<'_, K>],
    inactive_only: bool,
) -> Option<K> {
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

fn remove_entry<K: DenseWeightKey>(inner: &mut CacheState<K>, key: &K) -> CacheEntry {
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
        BlockFormat, DenseWeightCacheKey, DenseWeightIdentity, packed_tensor_bytes,
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
    fn cached(key: &DenseWeightCacheKey) -> DenseWeightSpec<'_, DenseWeightCacheKey> {
        DenseWeightSpec {
            key: Some(key),
            bytes: 16,
            build_peak_bytes: 16,
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
    fn shared_pressure_reclaims_another_dense_cache_without_touching_active_readers() {
        let (governor, source) = memory(32);
        let a = DenseWeightCache::<DenseWeightCacheKey>::new(source.clone(), 32);
        let b = DenseWeightCache::<DenseWeightCacheKey>::new(source.clone(), 32);
        let c = DenseWeightCache::<DenseWeightCacheKey>::new(source, 0);
        let packed_a = Owned::u8(&[4], &[1; 4]);
        let packed_b = Owned::u8(&[4], &[2; 4]);
        let ka = key(&packed_a);
        let kb = key(&packed_b);
        drop(ready(
            a.load_group(&[cached(&ka)], |_, row| {
                row.fill(1.0);
                Ok(())
            })
            .unwrap(),
        ));
        let held = ready(
            b.load_group(&[cached(&kb)], |_, row| {
                row.fill(2.0);
                Ok(())
            })
            .unwrap(),
        );
        assert_eq!(governor.used(Tier::Host), 32);
        let transient = ready(
            c.load_group(&[DenseWeightSpec::transient(16)], |_, row| {
                row.fill(3.0);
                Ok(())
            })
            .unwrap(),
        );
        assert_eq!(a.stats(), (0, 0));
        assert_eq!(b.stats(), (1, 16));
        assert_eq!(held.weights[0].as_slice(), &[2.0; 4]);
        assert_eq!(transient.weights[0].as_slice(), &[3.0; 4]);
        assert_eq!(governor.used(Tier::Host), 32);
        let mut built = false;
        assert!(matches!(
            c.load_group(&[DenseWeightSpec::transient(16)], |_, _| {
                built = true;
                Ok(())
            })
            .unwrap(),
            DenseWeightLoad::Declined(_)
        ));
        assert!(!built);
        drop(transient);
        drop(b);
        assert_eq!(
            governor.used(Tier::Host),
            16,
            "active reader retains the exact grant after its cache drops"
        );
        drop(held);
        assert_eq!(governor.used(Tier::Host), 0);
    }

    #[test]
    fn kv_lease_pressure_reclaims_idle_cpu_weights_across_sources_on_the_same_authority() {
        let (governor, first_source) = memory(36);
        let second_source = DerivedWeightMemory::governed(&governor, HolderId::new(14)).unwrap();
        assert!(first_source.supports_pressure_reclaim());
        assert!(second_source.supports_pressure_reclaim());
        let mut kv = governor
            .reserve(Tier::Host, 4, MemoryRole::KvCache, HolderId::new(2))
            .unwrap();
        let a = DenseWeightCache::<DenseWeightCacheKey>::new(first_source, 16);
        let b = DenseWeightCache::<DenseWeightCacheKey>::new(second_source, 16);
        let pa = Owned::u8(&[4], &[1; 4]);
        let pb = Owned::u8(&[4], &[2; 4]);
        let ka = key(&pa);
        let kb = key(&pb);
        drop(ready(
            a.load_group(&[cached(&ka)], |_, row| {
                row.fill(1.0);
                Ok(())
            })
            .unwrap(),
        ));
        drop(ready(
            b.load_group(&[cached(&kb)], |_, row| {
                row.fill(2.0);
                Ok(())
            })
            .unwrap(),
        ));
        let active = b.test_get(&kb).unwrap();
        assert_eq!(governor.used(Tier::Host), 36);
        kv.grow(8).unwrap();
        assert_eq!(a.stats(), (0, 0));
        assert_eq!(b.stats(), (1, 16));
        assert_eq!(kv.bytes(), 12);
        assert_eq!(governor.used(Tier::Host), 28);
        assert!(matches!(
            kv.grow(16),
            Err(MemoryError::TierExhausted { .. })
        ));
        assert_eq!(
            kv.bytes(),
            12,
            "failed growth cannot alter the live KV lease"
        );
        assert_eq!(active.as_slice(), &[2.0; 4]);
        drop(active);
        kv.grow(16).unwrap();
        assert_eq!(b.stats(), (0, 0));
        assert_eq!(kv.bytes(), 28);
        assert_eq!(governor.used(Tier::Host), 28);
        let sibling = kv.reserve_sibling(8).unwrap();
        assert_eq!(governor.used(Tier::Host), 36);
        drop(sibling);
        drop(kv);
        assert_eq!(governor.used(Tier::Host), 0);
    }

    #[test]
    fn shared_pressure_concurrent_cold_fills_keep_active_payloads_and_do_not_take_foreign_fill_gates()
     {
        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        struct Key(usize);
        impl DenseWeightKey for Key {
            fn dense_bytes(&self) -> Result<usize> {
                Ok(16)
            }
        }
        let (governor, source) = memory(16);
        let caches = [
            DenseWeightCache::<Key>::new(source.clone(), 16),
            DenseWeightCache::<Key>::new(source, 16),
        ];
        let barrier = Barrier::new(2);
        let admitted = AtomicUsize::new(0);
        let declined = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for (worker, cache) in caches.iter().enumerate() {
                let barrier = &barrier;
                let admitted = &admitted;
                let declined = &declined;
                let governor = &governor;
                scope.spawn(move || {
                    for round in 0..32 {
                        let key = Key(2 * round + worker);
                        barrier.wait();
                        let loaded = cache
                            .load_group(&[DenseWeightSpec::cached(&key).unwrap()], |_, row| {
                                row.fill(key.0 as f32);
                                Ok(())
                            })
                            .unwrap();
                        match &loaded {
                            DenseWeightLoad::Ready(ready) => {
                                admitted.fetch_add(1, Ordering::Relaxed);
                                assert_eq!(ready.weights[0].as_slice(), &[key.0 as f32; 4]);
                            }
                            DenseWeightLoad::Declined(_) => {
                                declined.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        assert!(governor.used(Tier::Host) <= 16);
                        // Both results stay alive until both allocations finish:
                        // a successful payload must not fund the competing miss.
                        barrier.wait();
                        drop(loaded);
                        barrier.wait();
                    }
                });
            }
        });
        assert_eq!(admitted.load(Ordering::Relaxed), 32);
        assert_eq!(declined.load(Ordering::Relaxed), 32);
        assert_eq!(governor.used(Tier::Host), 16);
        drop(caches);
        assert_eq!(governor.used(Tier::Host), 0);
    }

    #[test]
    fn shared_pressure_keeps_partial_hit_groups_pinned_and_respects_authorities() {
        let (governor, source) = memory(32);
        let a = DenseWeightCache::<DenseWeightCacheKey>::new(source.clone(), 64);
        let other = DenseWeightCache::<DenseWeightCacheKey>::new(source, 64);
        let p = [
            Owned::u8(&[4], &[1; 4]),
            Owned::u8(&[4], &[2; 4]),
            Owned::u8(&[4], &[3; 4]),
        ];
        let keys: Vec<_> = p.iter().map(key).collect();
        drop(ready(
            a.load_group(&[cached(&keys[0])], |_, row| {
                row.fill(1.0);
                Ok(())
            })
            .unwrap(),
        ));
        drop(ready(
            other
                .load_group(&[cached(&keys[1])], |_, row| {
                    row.fill(2.0);
                    Ok(())
                })
                .unwrap(),
        ));
        let (foreign_governor, foreign_source) = memory(16);
        let foreign = DenseWeightCache::<DenseWeightCacheKey>::new(foreign_source.clone(), 16);
        drop(ready(
            foreign
                .load_group(&[cached(&keys[2])], |_, row| {
                    row.fill(9.0);
                    Ok(())
                })
                .unwrap(),
        ));
        let hit = a.test_get(&keys[0]).unwrap();
        let loaded = ready(
            a.load_group(&[cached(&keys[0]), cached(&keys[2])], |index, row| {
                assert_eq!(index, 1);
                row.fill(3.0);
                Ok(())
            })
            .unwrap(),
        );
        assert!(Arc::ptr_eq(&hit, &loaded.weights[0]));
        assert_eq!(other.stats(), (0, 0));
        assert_eq!(foreign.stats(), (1, 16));
        assert_eq!(governor.used(Tier::Host), 32);
        assert_eq!(foreign_governor.used(Tier::Host), 16);
        drop(loaded);
        drop(hit);
        let pins = foreign.test_get(&keys[2]).unwrap();
        assert!(
            matches!(
                foreign_source.try_grant(16).unwrap(),
                WeightAdmission::Declined(_)
            ),
            "another authority cannot evict idle entries from these books"
        );
        drop(pins);
        assert_eq!(a.stats(), (2, 32));
        drop(a);
        drop(foreign);
        assert_eq!(governor.used(Tier::Host), 0);
        assert_eq!(foreign_governor.used(Tier::Host), 0);
    }

    #[test]
    fn owned_builder_peak_refunds_scratch_and_keeps_only_the_live_payload_charge() {
        let packed = Owned::u8(&[4], &[1; 4]);
        let key = key(&packed);
        let (governor, source) = memory(32);
        let cache = DenseWeightCache::<DenseWeightCacheKey>::new(source, 16);
        let spec = cached(&key).with_build_peak(32);
        let mut loaded = ready(
            cache
                .load_group_owned(&[spec], |_| {
                    assert_eq!(governor.used(Tier::Host), 32);
                    let source = vec![2.0f32; 4];
                    let placed = source.clone();
                    assert_eq!(governor.used(Tier::Host), 32);
                    Ok(placed)
                })
                .unwrap(),
        );
        assert_eq!(governor.used(Tier::Host), 16);
        let live = loaded.weights.remove(0);
        drop(loaded);
        let other = governor
            .reserve(Tier::Host, 16, MemoryRole::KvCache, HolderId::new(7))
            .unwrap();
        let hit = ready(
            cache
                .load_group_owned(&[cached(&key).with_build_peak(32)], |_| {
                    panic!("warm payload must not rebuild or request a second peak allowance")
                })
                .unwrap(),
        );
        assert!(Arc::ptr_eq(&live, &hit.weights[0]));
        assert_eq!(governor.used(Tier::Host), 32);
        drop(hit);
        drop(other);
        drop(cache);
        assert_eq!(governor.used(Tier::Host), 16);
        drop(live);
        assert_eq!(governor.used(Tier::Host), 0);
    }

    #[test]
    fn owned_builder_refusal_and_invalid_payloads_leave_no_reservation_or_cache_entry() {
        let packed = Owned::u8(&[4], &[1; 4]);
        let key = key(&packed);
        let (governor, source) = memory(16);
        let cache = DenseWeightCache::<DenseWeightCacheKey>::new(source, 16);
        assert!(matches!(
            cache
                .load_group_owned(&[cached(&key).with_build_peak(32)], |_| {
                    panic!("placement peak must be admitted before invoking the builder")
                })
                .unwrap(),
            DenseWeightLoad::Declined(_)
        ));
        assert_eq!(governor.used(Tier::Host), 0);
        for mode in 0..3 {
            let failure = cache
                .load_group_owned(&[cached(&key)], |_| {
                    if mode == 0 {
                        return Ok(vec![1.0; 3]);
                    }
                    if mode == 1 {
                        let mut v = Vec::with_capacity(8);
                        v.resize(4, 1.0);
                        return Ok(v);
                    }
                    Err(EpError::KernelFailed("fixture decode failure".into()))
                })
                .err()
                .expect("invalid builder result must fail");
            assert!(failure.to_string().contains(if mode == 2 {
                "fixture decode failure"
            } else {
                "exactly sized f32 payload"
            }));
            assert_eq!(governor.used(Tier::Host), 0);
            assert_eq!(cache.stats(), (0, 0));
        }
        let failure = cache
            .load_group_owned(&[cached(&key).with_build_peak(15)], |_| {
                panic!("an underspecified construction peak must not run")
            })
            .err()
            .unwrap();
        assert!(failure.to_string().contains("below its 16-byte payload"));
        assert_eq!(governor.used(Tier::Host), 0);
    }

    #[test]
    fn invalid_buffer_sizes_are_rejected_before_any_builder_or_grant() {
        let cache = DenseWeightCache::<DenseWeightCacheKey>::default();
        for bytes in [3, isize::MAX as usize + 1] {
            let failure = cache
                .load_group(
                    &[DenseWeightSpec {
                        key: None,
                        bytes,
                        build_peak_bytes: bytes,
                    }],
                    |_, _| panic!("invalid buffer must not be allocated"),
                )
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
        let first = DenseWeightCache::<DenseWeightCacheKey>::new(source.clone(), 16);
        let second = DenseWeightCache::<DenseWeightCacheKey>::new(source, 1024);
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
        let cache = DenseWeightCache::<DenseWeightCacheKey>::new(source, 1024);
        let specs = [
            DenseWeightSpec {
                key: None,
                bytes: 16,
                build_peak_bytes: 16,
            },
            DenseWeightSpec {
                key: None,
                bytes: 16,
                build_peak_bytes: 16,
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
        let cache = DenseWeightCache::<DenseWeightCacheKey>::new(source, 1024);
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
            ..DenseWeightCache::<DenseWeightCacheKey>::new(source, 1024)
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
        let cache = DenseWeightCache::<DenseWeightCacheKey>::new(source, 1024);
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
        let cache = Arc::new(DenseWeightCache::<DenseWeightCacheKey>::new(source, 1024));
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
        cache: Mutex<Weak<DenseWeightCache<DenseWeightCacheKey>>>,
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
        fn pressure_registry(
            &self,
            tier: Tier,
        ) -> Option<onnx_runtime_memory_governor::PressureRegistry> {
            self.0.governor.pressure_registry(tier)
        }
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
        let cache = Arc::new(DenseWeightCache::<DenseWeightCacheKey>::new(source, 1024));
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
        assert_eq!(
            books.observations.load(Ordering::Relaxed),
            3,
            "two issuer requests and one retired payload refund; the authority retries its own books internally"
        );
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

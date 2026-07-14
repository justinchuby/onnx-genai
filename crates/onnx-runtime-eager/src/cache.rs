//! The compiled-kernel cache (`docs/EAGER.md` §8.2).
//!
//! A bounded LRU keyed by the op identity plus the concrete input shapes,
//! dtypes, and device — so that repeated eager calls with the same op and shapes
//! reuse an already-compiled kernel instead of recreating one per call.
//!
//! ## Design-vs-real-API reconciliation
//!
//! The design stores `Arc<dyn Kernel>` (`docs/EAGER.md` §8.2). The real
//! [`Kernel`](onnx_runtime_ep_api::Kernel) trait is `Send` **but not `Sync`**
//! (`onnx-runtime-ep-api/src/kernel.rs`), so a bare `Arc<dyn Kernel>` is not
//! `Send` and could not live inside the process-global, `Sync`
//! [`EagerContext`](crate::EagerContext). We therefore store
//! `Arc<Mutex<Box<dyn Kernel>>>`: the `Mutex` restores `Send + Sync` while
//! preserving the design's "share one compiled kernel across calls" intent.
//! `Kernel::execute` takes `&self`, so the lock only serialises reuse of the
//! *same* cached kernel; distinct ops/shapes dispatch concurrently.
//!
//! The eviction policy is a straightforward bounded LRU (a `HashMap` +
//! recency `VecDeque`) rather than an external `lru` crate, keeping the
//! dependency set minimal as requested. Correctness of the *key* matters more
//! than eviction sophistication (§8.2).

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use onnx_runtime_ep_api::Kernel;
use onnx_runtime_ir::{DataType, DeviceId};

/// A cached, compiled kernel, shareable across dispatches. See the module docs
/// for why this is `Arc<Mutex<Box<dyn Kernel>>>` rather than `Arc<dyn Kernel>`.
pub type CachedKernel = Arc<Mutex<Box<dyn Kernel>>>;

/// The cache key (`docs/EAGER.md` §8.2 `KernelCacheKey`).
///
/// Two dispatches share a compiled kernel iff they agree on the op identity
/// (`op_type`, `domain`, `opset`), the input shapes, the input dtypes, and the
/// device. Shapes and dtypes are part of the key because kernels are compiled
/// specialised to concrete shapes (§4.2 / §8.2).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct KernelCacheKey {
    pub op_type: String,
    pub domain: String,
    pub opset: u64,
    pub input_shapes: Vec<Vec<usize>>,
    pub input_dtypes: Vec<DataType>,
    pub device: DeviceId,
}

/// Cache instrumentation, exposed for tests and diagnostics.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheStats {
    /// Distinct compiled entries currently held.
    pub entries: usize,
    /// Lookups served from an existing entry.
    pub hits: u64,
    /// Lookups that compiled a new kernel.
    pub misses: u64,
}

/// A bounded LRU of compiled kernels (`docs/EAGER.md` §8.2).
pub struct KernelCache {
    capacity: usize,
    map: HashMap<KernelCacheKey, CachedKernel>,
    /// Recency order, least-recently-used at the front.
    order: VecDeque<KernelCacheKey>,
    hits: u64,
    misses: u64,
}

impl KernelCache {
    /// A cache holding at most `capacity` compiled kernels (`capacity` is
    /// clamped to at least 1).
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            map: HashMap::new(),
            order: VecDeque::new(),
            hits: 0,
            misses: 0,
        }
    }

    /// Return the cached kernel for `key`, compiling and inserting it via
    /// `create` on a miss (`docs/EAGER.md` §8.2 `get_or_create`).
    pub fn get_or_create<E>(
        &mut self,
        key: KernelCacheKey,
        create: impl FnOnce() -> Result<Box<dyn Kernel>, E>,
    ) -> Result<CachedKernel, E> {
        if let Some(kernel) = self.map.get(&key) {
            let kernel = kernel.clone();
            self.hits += 1;
            self.touch(&key);
            return Ok(kernel);
        }
        self.misses += 1;
        let kernel: CachedKernel = Arc::new(Mutex::new(create()?));
        self.map.insert(key.clone(), kernel.clone());
        self.order.push_back(key);
        self.evict_if_needed();
        Ok(kernel)
    }

    /// Current cache statistics.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.map.len(),
            hits: self.hits,
            misses: self.misses,
        }
    }

    /// Move `key` to the most-recently-used end of the recency order.
    fn touch(&mut self, key: &KernelCacheKey) {
        if let Some(pos) = self.order.iter().position(|k| k == key)
            && let Some(k) = self.order.remove(pos)
        {
            self.order.push_back(k);
        }
    }

    /// Evict least-recently-used entries until within capacity.
    fn evict_if_needed(&mut self) {
        while self.map.len() > self.capacity {
            if let Some(lru) = self.order.pop_front() {
                self.map.remove(&lru);
            } else {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ep_api::{Result as EpResult, TensorMut, TensorView};

    /// A trivial no-op kernel for exercising the cache in isolation.
    struct NopKernel;
    impl Kernel for NopKernel {
        fn execute(&self, _inputs: &[TensorView], _outputs: &mut [TensorMut]) -> EpResult<()> {
            Ok(())
        }
    }

    fn key(op: &str, shape: Vec<usize>) -> KernelCacheKey {
        KernelCacheKey {
            op_type: op.to_string(),
            domain: String::new(),
            opset: 26,
            input_shapes: vec![shape],
            input_dtypes: vec![DataType::Float32],
            device: DeviceId::cpu(),
        }
    }

    #[test]
    fn miss_then_hit_same_key() {
        let mut cache = KernelCache::new(8);
        let k = key("Add", vec![2, 3]);
        let mut created = 0;
        let _ = cache
            .get_or_create::<()>(k.clone(), || {
                created += 1;
                Ok(Box::new(NopKernel))
            })
            .unwrap();
        let _ = cache
            .get_or_create::<()>(k.clone(), || {
                created += 1;
                Ok(Box::new(NopKernel))
            })
            .unwrap();
        assert_eq!(created, 1, "second dispatch must reuse the cached kernel");
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().entries, 1);
    }

    #[test]
    fn distinct_shapes_are_distinct_entries() {
        let mut cache = KernelCache::new(8);
        let _ = cache
            .get_or_create::<()>(key("Add", vec![2, 3]), || Ok(Box::new(NopKernel)))
            .unwrap();
        let _ = cache
            .get_or_create::<()>(key("Add", vec![4, 5]), || Ok(Box::new(NopKernel)))
            .unwrap();
        assert_eq!(cache.stats().entries, 2);
        assert_eq!(cache.stats().misses, 2);
    }

    #[test]
    fn lru_evicts_oldest_over_capacity() {
        let mut cache = KernelCache::new(2);
        let a = key("Add", vec![1]);
        let b = key("Mul", vec![1]);
        let c = key("Sub", vec![1]);
        let _ = cache.get_or_create::<()>(a.clone(), || Ok(Box::new(NopKernel))).unwrap();
        let _ = cache.get_or_create::<()>(b.clone(), || Ok(Box::new(NopKernel))).unwrap();
        // Touch `a` so `b` becomes the LRU victim.
        let _ = cache.get_or_create::<()>(a.clone(), || Ok(Box::new(NopKernel))).unwrap();
        let _ = cache.get_or_create::<()>(c.clone(), || Ok(Box::new(NopKernel))).unwrap();
        assert_eq!(cache.stats().entries, 2);
        // `b` evicted → recompiled on next access (a fresh miss).
        let mut recreated = 0;
        let _ = cache
            .get_or_create::<()>(b, || {
                recreated += 1;
                Ok(Box::new(NopKernel))
            })
            .unwrap();
        assert_eq!(recreated, 1);
    }
}

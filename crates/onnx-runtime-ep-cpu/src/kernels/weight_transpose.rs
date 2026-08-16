//! Process-global weight-transpose caches for the MatMul hot path.
//!
//! The Accelerate/NEON GEMV and thin-M GEMM kernels consume `B_T[N,K]`
//! row-major, so a constant (initializer) weight `B[K,N]` is transposed once and
//! reused for the lifetime of the model. The transpose is O(N×K) — ~1 s for a
//! large `lm_head` — so it must survive kernel-cache shape evictions (prefill
//! M=40 → decode M=1), which is why the cache is process-global rather than
//! per-kernel.
//!
//! ## Cache identity must be total (#845)
//!
//! The original key was the source buffer address alone. An address is **not**
//! an identity: allocators recycle addresses, so the same `usize` can name a
//! different matrix with a different shape and a different length. That produced
//! two distinct defects:
//!
//! 1. **Wrong results** — a stale transpose *longer* than `N×K` is silently
//!    consumed as if it were this weight's, so logits are wrong with no crash.
//! 2. **Out-of-bounds reads** — a stale transpose *shorter* than `N×K` is
//!    indexed past its end by `neon_thin_m_tile`. `debug_assert_eq!` catches it
//!    in debug builds; release builds (how the runtime ships) do not.
//!
//! [`WeightTransposeKey`] closes both by keying on the full identity of the
//! result: source address, `K`, and `N`. Steady-state cache size is unchanged —
//! a given weight has exactly one shape — so this costs nothing at runtime.
//!
//! Address recycling *within* one key (a second model whose mmap lands the same
//! weight shape at the same address) is a lifetime problem, not a keying one,
//! and is still handled by [`clear_all`] on `Executor` drop.
//!
//! ## Portability
//!
//! Only Apple targets consume these transposes today, but the module itself is
//! target-independent so the identity rules above are exercised by unit tests on
//! every CI platform. The bug in #845 reached CI precisely because the affected
//! code could not be tested off macOS.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

use rayon::prelude::*;

/// Admission verdict for the process-global weight-transpose caches (#1056).
///
/// The transpose cache is a resident, session-lifetime, weight-scaled buffer:
/// one full `K x N` copy per transposed constant weight. Like the resident
/// dequant f32 cache (#987) and the MLAS SQNBit packed buffer (#1051), it must
/// be *declinable* — when the memory-strategy plan cannot fit its predicted
/// bytes, the kernels must compute the transpose per call and cache nothing,
/// trading decode speed for footprint instead of overrunning the budget.
///
/// Defaults to enabled so the out-of-box path is unchanged; the engine flips it
/// once, at load, from the plan's verdict (see `onnx-genai-engine`'s
/// `set_weight_transpose_cache_enabled` beside the other two buffers' gates).
static WEIGHT_TRANSPOSE_CACHE_ENABLED: AtomicBool = AtomicBool::new(true);

/// Admit (or decline) the process-global weight-transpose caches. See
/// [`WEIGHT_TRANSPOSE_CACHE_ENABLED`].
///
/// This is the **production** entry point: the engine calls it exactly once, at
/// load, from the plan's verdict. It writes a process-global that every worker
/// thread reads, which is correct for production but toxic for a parallel test
/// harness — a test that flipped it would race every other test in the process
/// (this is the #983 / #1033 / #1056 "passes alone, fails in company" trap).
/// Tests must therefore never call this; they use [`CacheEnabledScope`], a
/// thread-local override that leaves this global untouched.
pub fn set_cache_enabled(enabled: bool) {
    WEIGHT_TRANSPOSE_CACHE_ENABLED.store(enabled, Ordering::Relaxed);
}

thread_local! {
    /// Test-only, per-thread override of the admission verdict. `None` means
    /// "defer to the process-global"; `Some(v)` forces `v` on **this thread
    /// only**, so one test's decline cannot leak into another test running
    /// concurrently on a different worker thread. Set exclusively through
    /// [`CacheEnabledScope`], which restores the previous value on drop (even on
    /// panic). Production never touches this.
    static CACHE_ENABLED_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

/// Whether the process-global weight-transpose caches are currently admitted.
///
/// A thread-local override (set only by tests via [`CacheEnabledScope`]) wins
/// over the process-global when present, so the decline path can be exercised
/// on one thread without disturbing the rest of the parallel test harness.
pub fn cache_enabled() -> bool {
    if let Some(forced) = CACHE_ENABLED_OVERRIDE.with(|c| c.get()) {
        return forced;
    }
    WEIGHT_TRANSPOSE_CACHE_ENABLED.load(Ordering::Relaxed)
}

/// RAII, thread-local scoping of the admission verdict for tests (#1056).
///
/// Constructing one forces [`cache_enabled`] to `enabled` on the current thread;
/// dropping it restores whatever was in effect before — including on panic, so a
/// failing test can never leave the flag flipped for the tests that follow.
/// Because the override is thread-local, a test scoping a decline here does not
/// race the `transposed_b` tests running concurrently on other threads, which
/// was the failure this type exists to prevent.
#[cfg(test)]
pub(crate) struct CacheEnabledScope {
    prev: Option<bool>,
}

#[cfg(test)]
impl CacheEnabledScope {
    /// Force the admission verdict to `enabled` on this thread until dropped.
    pub(crate) fn new(enabled: bool) -> Self {
        let prev = CACHE_ENABLED_OVERRIDE.with(|c| c.replace(Some(enabled)));
        Self { prev }
    }
}

#[cfg(test)]
impl Drop for CacheEnabledScope {
    fn drop(&mut self) {
        CACHE_ENABLED_OVERRIDE.with(|c| c.set(self.prev));
    }
}

/// Total identity of one cached weight transpose.
///
/// A cached `B_T[N,K]` is a pure function of the source address, `K`, `N`, and
/// the element type. Those fields — and no others — are in the key:
///
/// * **address** distinguishes different weights;
/// * **`k` and `n`** distinguish different tensors that reuse one address. They
///   are kept as two fields rather than one length because `(k=64, n=128)` and
///   `(k=128, n=64)` have the same length and different transposes.
///
/// The element type is not a field because each dtype has its own cache
/// instance. The permutation is not a field because this cache stores exactly
/// one operation: row-major `K×N → N×K`. Device/allocator identity is not a
/// field because the CPU EP has a single host address space.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct WeightTransposeKey {
    addr: usize,
    k: usize,
    n: usize,
}

impl WeightTransposeKey {
    /// Build the key for the transpose of a `[k, n]` matrix based at `ptr`.
    pub fn new<T>(ptr: *const T, k: usize, n: usize) -> Self {
        Self {
            addr: ptr as usize,
            k,
            n,
        }
    }

    /// Element count of both the source and the transpose (`k * n`), or `None`
    /// when the product overflows `usize`.
    pub fn numel(&self) -> Option<usize> {
        self.k.checked_mul(self.n)
    }
}

/// A process-global map from [`WeightTransposeKey`] to the transposed data.
///
/// Entries are `Arc` so a kernel-local memo and the global cache share one
/// allocation. The map is guarded by a plain `Mutex` held only for the lookup
/// and the insert — never across the transpose itself, so concurrent decode
/// threads do not serialize on it.
pub struct TransposeCache<T> {
    entries: Mutex<HashMap<WeightTransposeKey, Arc<Vec<T>>>>,
}

impl<T> Default for TransposeCache<T> {
    fn default() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }
}

impl<T: Copy + Default + Send + Sync> TransposeCache<T> {
    /// Look up an already-computed transpose. Returns `None` on a miss.
    pub fn get(&self, key: &WeightTransposeKey) -> Option<Arc<Vec<T>>> {
        self.entries
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(key)
            .cloned()
    }

    /// Drop a single entry by key. Test-only; used to clear a stale entry left
    /// at a since-recycled address so a peek has a deterministic starting point.
    #[cfg(test)]
    pub fn remove(&self, key: &WeightTransposeKey) {
        self.entries
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(key);
    }

    /// Number of live entries.
    pub fn len(&self) -> usize {
        self.entries.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Bytes of transposed weight this cache currently holds.
    ///
    /// Entry counts cannot answer "will this fit" (#1056): each entry is a full
    /// `K x N` copy of a weight, so two caches with the same length can differ by
    /// gigabytes. `Arc` clones are not double counted -- the map holds one `Arc`
    /// per key and the byte total is over the map, not over live handles.
    pub fn bytes(&self) -> usize {
        self.entries
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|entry| entry.len() * std::mem::size_of::<T>())
            .sum()
    }

    /// Whether the cache holds no entries.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drop every entry. Called when an `Executor` drops so a later model's
    /// mmap cannot inherit this model's transposes at a recycled address.
    pub fn clear(&self) {
        self.entries
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    /// Return the `[n, k]` row-major transpose of the `[k, n]` row-major matrix
    /// `src`, computing and caching it on a miss.
    ///
    /// Returns `None` — never a wrong-length buffer — when `src.len()` is not
    /// exactly `k * n`. That is the fail-closed leg of the #845 contract: a
    /// caller whose geometry disagrees with its buffer gets no transpose and
    /// falls back to the untransposed kernel, instead of reading past the end of
    /// a cached vector.
    ///
    /// Zero-element matrices are answered with an empty vector and never
    /// inserted: they cost nothing to produce and would otherwise let a caller
    /// with a degenerate shape grow the map.
    pub fn get_or_insert_transpose(&self, src: &[T], k: usize, n: usize) -> Option<Arc<Vec<T>>> {
        let key = WeightTransposeKey::new(src.as_ptr(), k, n);
        if key.numel() != Some(src.len()) {
            return None;
        }
        if src.is_empty() {
            return Some(Arc::new(Vec::new()));
        }
        if let Some(hit) = self.get(&key) {
            return Some(hit);
        }
        let transposed = Arc::new(transpose_row_major(src, k, n));
        // A concurrent racer may have inserted an identical entry meanwhile.
        // Keep whichever landed first so every reader shares one allocation.
        Some(
            self.entries
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .entry(key)
                .or_insert(transposed)
                .clone(),
        )
    }
}

/// Transpose `src[k, n]` row-major into a fresh `[n, k]` row-major vector.
///
/// Tiled over `K` (64-element strips keep the source rows L1-hot) and
/// parallelized over disjoint output row ranges, so no synchronization is
/// needed. This is the single implementation behind every weight transpose in
/// the CPU EP.
pub fn transpose_row_major<T: Copy + Default + Send + Sync>(
    src: &[T],
    k: usize,
    n: usize,
) -> Vec<T> {
    assert!(
        k.checked_mul(n) == Some(src.len()),
        "transpose_row_major: source length {} does not match [{k}, {n}]",
        src.len()
    );
    let mut out = vec![T::default(); src.len()];
    if out.is_empty() {
        return out;
    }
    let threads = rayon::current_num_threads();
    let rows_per_thread = n.div_ceil(threads).max(1);
    out.par_chunks_mut(rows_per_thread * k)
        .enumerate()
        .for_each(|(t, chunk)| {
            let j0 = t * rows_per_thread;
            let j_end = (j0 + rows_per_thread).min(n);
            let chunk_n = j_end - j0;
            const TILE: usize = 64;
            for i0 in (0..k).step_by(TILE) {
                let ie = (i0 + TILE).min(k);
                for jj in 0..chunk_n {
                    let j = j0 + jj;
                    for i in i0..ie {
                        chunk[jj * k + i] = src[i * n + j];
                    }
                }
            }
        });
    out
}

/// Process-global cache of transposed f16 weights, stored as the raw `u16` bit
/// patterns read straight from the model's mmap (no widening to f32).
static WEIGHT_TRANSPOSE_F16: LazyLock<TransposeCache<u16>> = LazyLock::new(TransposeCache::default);

/// Process-global cache of transposed f32 weights (Accelerate GEMV / thin-M).
static WEIGHT_TRANSPOSE_F32: LazyLock<TransposeCache<f32>> = LazyLock::new(TransposeCache::default);

/// Compute the `[n, k]` transpose of `src[k, n]` **without** touching the
/// process-global cache — the declined path (#1056).
///
/// Honours the same fail-closed length contract as
/// [`TransposeCache::get_or_insert_transpose`] (returns `None` on a length
/// mismatch, an empty vector for a zero-element matrix) so a caller cannot tell
/// admitted from declined except by cost: the returned `Arc` is freshly
/// allocated on every call and freed when the caller drops it, so nothing
/// survives the kernel call and the cache's byte total stays put.
fn transpose_uncached<T: Copy + Default + Send + Sync>(
    src: &[T],
    k: usize,
    n: usize,
) -> Option<Arc<Vec<T>>> {
    let key = WeightTransposeKey::new(src.as_ptr(), k, n);
    if key.numel() != Some(src.len()) {
        return None;
    }
    if src.is_empty() {
        return Some(Arc::new(Vec::new()));
    }
    Some(Arc::new(transpose_row_major(src, k, n)))
}

/// Cached `B_T[N,K]` for an f32 weight `B[K,N]`, or `None` when `src.len()` is
/// not exactly `k * n`. See [`TransposeCache::get_or_insert_transpose`].
///
/// When the cache is declined ([`cache_enabled`] is `false`) the transpose is
/// computed per call and not retained (#1056).
pub fn cached_transpose_f32(src: &[f32], k: usize, n: usize) -> Option<Arc<Vec<f32>>> {
    if !cache_enabled() {
        return transpose_uncached(src, k, n);
    }
    WEIGHT_TRANSPOSE_F32.get_or_insert_transpose(src, k, n)
}

/// Test-only: is the transpose of the `[k, n]` f32 weight based at `ptr`
/// currently resident in the global cache? Unlike a global byte-total delta,
/// this isolates one specific weight, so a concurrent test populating an
/// unrelated entry cannot perturb the answer (#1056 isolation).
#[cfg(test)]
pub(crate) fn f32_cache_contains(ptr: *const f32, k: usize, n: usize) -> bool {
    WEIGHT_TRANSPOSE_F32
        .get(&WeightTransposeKey::new(ptr, k, n))
        .is_some()
}

/// Test-only: drop any f32 entry keyed on `(ptr, k, n)`.
///
/// The cache keys on `(address, K, N)`. An address only names a weight while
/// that weight is *live*; once freed, the allocator may recycle it for an
/// unrelated buffer of the same dimensions, at which point a stale entry left
/// by an earlier (now-freed) weight would answer [`f32_cache_contains`] `true`
/// for a buffer that never went through the cache. A test that wants a
/// deterministic "before" state calls this first: it is safe under the parallel
/// harness because a *live* concurrent allocation can never share `ptr`, so the
/// only entry this can remove is a stale one nobody is using.
#[cfg(test)]
pub(crate) fn f32_cache_evict(ptr: *const f32, k: usize, n: usize) {
    WEIGHT_TRANSPOSE_F32.remove(&WeightTransposeKey::new(ptr, k, n));
}

/// Cached `B_T[N,K]` for an f16 weight `B[K,N]` held as raw `u16` bit patterns,
/// or `None` when `src.len()` is not exactly `k * n`.
///
/// When the cache is declined ([`cache_enabled`] is `false`) the transpose is
/// computed per call and not retained (#1056).
pub fn cached_transpose_f16(src: &[u16], k: usize, n: usize) -> Option<Arc<Vec<u16>>> {
    if !cache_enabled() {
        return transpose_uncached(src, k, n);
    }
    WEIGHT_TRANSPOSE_F16.get_or_insert_transpose(src, k, n)
}

/// Entry counts of the global caches as `(f16, f32)`.
pub fn cache_sizes() -> (usize, usize) {
    (WEIGHT_TRANSPOSE_F16.len(), WEIGHT_TRANSPOSE_F32.len())
}

/// Bytes of transposed weight held by the global caches, summed across both.
///
/// This is a resident, session-lifetime, weight-scaled allocation: one full
/// `K x N` copy per transposed constant weight. #1056 requires such buffers to
/// be reportable in bytes rather than in entry counts, because only bytes can
/// answer whether they fit.
pub fn cache_bytes() -> usize {
    WEIGHT_TRANSPOSE_F16
        .bytes()
        .saturating_add(WEIGHT_TRANSPOSE_F32.bytes())
}

/// Evict every entry from both global caches.
pub fn clear_all() {
    WEIGHT_TRANSPOSE_F16.clear();
    WEIGHT_TRANSPOSE_F32.clear();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference transpose: independent of the production tiled/parallel code.
    fn reference_transpose(src: &[f32], k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; n * k];
        for j in 0..n {
            for i in 0..k {
                out[j * k + i] = src[i * n + j];
            }
        }
        out
    }

    fn fill(buf: &mut [f32], seed: f32) {
        for (i, v) in buf.iter_mut().enumerate() {
            *v = seed + i as f32 * 0.5;
        }
    }

    /// A private cache instance, so tests never race the process-global ones.
    fn cache() -> TransposeCache<f32> {
        TransposeCache::default()
    }

    /// #1056: the byte figure must equal the bytes actually held, so it can
    /// answer "will this fit". An entry count cannot: two caches of length 2 can
    /// differ by gigabytes.
    ///
    /// This asserts against the sum of the cached buffers' own lengths rather
    /// than against a formula, so a future change to what gets stored per entry
    /// cannot leave the reported number describing the old layout.
    #[test]
    fn cache_bytes_equals_the_bytes_actually_held() {
        let cache = cache();
        assert_eq!(cache.bytes(), 0, "an empty cache holds no bytes");

        let mut small = vec![0.0f32; 4 * 8];
        fill(&mut small, 1.0);
        let mut large = vec![0.0f32; 64 * 32];
        fill(&mut large, 2.0);

        let a = cache.get_or_insert_transpose(&small, 4, 8).unwrap();
        let b = cache.get_or_insert_transpose(&large, 64, 32).unwrap();

        let expected = (a.len() + b.len()) * std::mem::size_of::<f32>();
        assert_eq!(cache.bytes(), expected);
        assert_eq!(cache.len(), 2);

        // The point of bytes over entries: same length, wildly different cost.
        assert!(
            expected > 8 * cache.len(),
            "entry count cannot stand in for byte cost"
        );

        // A hit must not double count: the map holds one Arc per key.
        let again = cache.get_or_insert_transpose(&large, 64, 32).unwrap();
        assert!(Arc::ptr_eq(&b, &again));
        assert_eq!(cache.bytes(), expected, "a cache hit allocated nothing");

        cache.clear();
        assert_eq!(cache.bytes(), 0, "clearing releases the bytes it reported");
    }

    /// #845 falsifier — the grow case.
    ///
    /// One buffer address is used first for a small `[k1, n1]` matrix and then
    /// for a larger `[k2, n2]` one, exactly as an allocator that recycles an
    /// address does. Under the old address-only key the second call returned the
    /// first transpose, whose length is *shorter* than `n2 * k2` — the
    /// release-mode out-of-bounds read.
    #[test]
    fn same_address_grow_serves_the_current_shape() {
        let cache = cache();
        let mut storage = vec![0.0f32; 16 * 24];

        let (k1, n1) = (4, 6);
        fill(&mut storage[..k1 * n1], 1.0);
        let small = cache
            .get_or_insert_transpose(&storage[..k1 * n1], k1, n1)
            .expect("small transpose");
        assert_eq!(small.len(), n1 * k1);
        assert_eq!(
            small.as_slice(),
            reference_transpose(&storage[..k1 * n1], k1, n1)
        );

        // Same base address, larger tensor.
        let (k2, n2) = (16, 24);
        fill(&mut storage, 100.0);
        let large = cache
            .get_or_insert_transpose(&storage, k2, n2)
            .expect("large transpose");
        assert_eq!(
            large.len(),
            n2 * k2,
            "cache served a {}-element transpose for a {}-element tensor at the same address \
             — release builds index past the end of it",
            large.len(),
            n2 * k2
        );
        assert_eq!(large.as_slice(), reference_transpose(&storage, k2, n2));
        assert_eq!(cache.len(), 2, "both shapes must hold distinct entries");
    }

    /// #845 falsifier — the shrink case.
    ///
    /// The inverse ordering: a large tensor is cached first, then a smaller one
    /// lands on the same address. Under the old key the second call returned a
    /// *longer* stale transpose — the silently-wrong-logits variant.
    #[test]
    fn same_address_shrink_serves_the_current_shape() {
        let cache = cache();
        let mut storage = vec![0.0f32; 16 * 24];

        let (k1, n1) = (16, 24);
        fill(&mut storage, 7.0);
        let large = cache
            .get_or_insert_transpose(&storage, k1, n1)
            .expect("large transpose");
        assert_eq!(large.len(), n1 * k1);

        let (k2, n2) = (4, 6);
        fill(&mut storage[..k2 * n2], -3.0);
        let small = cache
            .get_or_insert_transpose(&storage[..k2 * n2], k2, n2)
            .expect("small transpose");
        assert_eq!(
            small.len(),
            n2 * k2,
            "cache served a {}-element transpose for a {}-element tensor at the same address \
             — the consumer would read stale weights",
            small.len(),
            n2 * k2
        );
        assert_eq!(
            small.as_slice(),
            reference_transpose(&storage[..k2 * n2], k2, n2)
        );
    }

    /// Length alone is not identity: `[64, 128]` and `[128, 64]` have the same
    /// element count and different transposes, so the key keeps `k` and `n`
    /// separately.
    #[test]
    fn same_address_same_length_transposed_dims_are_distinct() {
        let cache = cache();
        let mut storage = vec![0.0f32; 64 * 128];
        fill(&mut storage, 0.25);

        let tall = cache
            .get_or_insert_transpose(&storage, 64, 128)
            .expect("64x128");
        let wide = cache
            .get_or_insert_transpose(&storage, 128, 64)
            .expect("128x64");

        assert_eq!(tall.len(), wide.len());
        assert_eq!(tall.as_slice(), reference_transpose(&storage, 64, 128));
        assert_eq!(
            wide.as_slice(),
            reference_transpose(&storage, 128, 64),
            "equal-length shapes at one address must not share a cache entry"
        );
        assert_ne!(tall.as_slice(), wide.as_slice());
        assert_eq!(cache.len(), 2);
    }

    /// A deterministic stand-in for an allocator that recycles addresses.
    ///
    /// One backing allocation is filled once and then handed out as logical
    /// tensors of different shapes, every one of them based at the same
    /// address. That is precisely the situation a recycling allocator creates —
    /// one address naming successive tensors of different geometry — but it is
    /// produced by construction instead of by asking the platform allocator to
    /// please reuse a freed block.
    ///
    /// Contents are a pure function of the element index, so every prefix has
    /// stable contents for the whole run: revisiting a shape must observe the
    /// same bytes it was first cached with, otherwise the cache's documented
    /// same-key behaviour (see `same_address_same_shape_is_stale_until_cleared`)
    /// would be indistinguishable from the shape-blindness under test.
    struct RecyclingArena {
        storage: Vec<f32>,
    }

    impl RecyclingArena {
        fn new(capacity: usize) -> Self {
            let mut storage = vec![0.0f32; capacity];
            for (i, v) in storage.iter_mut().enumerate() {
                *v = 1.0 + i as f32 * 0.5;
            }
            Self { storage }
        }

        fn base_addr(&self) -> usize {
            self.storage.as_ptr() as usize
        }

        /// The `[k, n]` tensor at the arena's base address.
        fn tensor(&self, k: usize, n: usize) -> &[f32] {
            &self.storage[..k * n]
        }
    }

    /// The transpose a geometry was first served at the arena's address.
    struct FirstVisit {
        shape: (usize, usize),
        transpose: Arc<Vec<f32>>,
    }

    /// #845 falsifier — address reuse across incompatible shapes, deterministic.
    ///
    /// Eight rounds walk four geometries that all live at one address: two that
    /// share a length and differ only in orientation (`[8, 32]` / `[32, 8]`),
    /// one shorter (`[4, 6]`) and one longer (`[16, 24]`) than what the cache
    /// already holds for that address, and then a repeat of each so the second
    /// visit must *hit* the entry the first visit created.
    ///
    /// Under the old address-only key every round after the first is served the
    /// first round's transpose, which fails here in three distinct ways: wrong
    /// contents for the equal-length reorientation, a too-short buffer for the
    /// grow round (the release-mode out-of-bounds read, exercised below by
    /// walking every output row), and a too-long buffer for the shrink round.
    ///
    /// This test used to allocate and free a fresh `Vec` per round and rely on
    /// the platform allocator to recycle the address, with `reused > 0` as the
    /// non-vacuity guard. glibc obliges; the Windows heap does not have to, and
    /// on the Windows CI lane it recycled nothing (`0/8`), so the guard failed
    /// the run. The invariant under test is a property of the *cache key*, not
    /// of any allocator, so the reuse is now constructed rather than hoped for:
    /// the guard below asserts full reuse (`7/8` collisions, all at one
    /// address) and is therefore stronger than the probabilistic one it
    /// replaces, on every platform.
    #[test]
    fn allocator_address_reuse_across_shapes() {
        let cache = cache();
        // Capacity is the largest geometry; every other shape is a prefix of it.
        let arena = RecyclingArena::new(16 * 24);
        let base = arena.base_addr();

        // Equal-length reorientation, shrink, grow — then the same four again.
        const ROUNDS: [(usize, usize); 8] = [
            (8, 32),
            (32, 8),
            (4, 6),
            (16, 24),
            (8, 32),
            (32, 8),
            (4, 6),
            (16, 24),
        ];

        let mut reused = 0usize;
        let mut seen_addrs: Vec<usize> = Vec::new();
        let mut first_visits: Vec<FirstVisit> = Vec::new();

        for (round, &(k, n)) in ROUNDS.iter().enumerate() {
            let src = arena.tensor(k, n);
            // Observe the address rather than assume it: `reused` counts rounds
            // that landed on an address an earlier round had already cached
            // under a different geometry, exactly as the old test counted them.
            let addr = src.as_ptr() as usize;
            if seen_addrs.contains(&addr) {
                reused += 1;
            }
            seen_addrs.push(addr);
            assert_eq!(
                addr, base,
                "round {round}: arena handed out a different address, so this round \
                 would not collide with the entries the earlier rounds cached"
            );

            let bt = cache
                .get_or_insert_transpose(src, k, n)
                .expect("transpose of a well-formed [k, n] slice");

            assert_eq!(
                bt.len(),
                n * k,
                "round {round}: cache served a {}-element transpose for the \
                 {}-element [{k}, {n}] tensor at address {base:#x}",
                bt.len(),
                n * k
            );
            assert_eq!(
                bt.as_slice(),
                reference_transpose(src, k, n),
                "round {round}: transpose does not match this round's [{k}, {n}] data"
            );

            // Out-of-bounds coverage that survives `--release`: the consumer
            // (`neon_thin_m_tile`) walks `B_T` row by row, so touch the last
            // element of every output row. A stale, shorter entry indexes past
            // its end, which a slice bounds-checks in release too — the kernel's
            // raw-pointer walk over the same entry would not.
            for j in 0..n {
                assert_eq!(
                    bt[j * k + (k - 1)],
                    src[(k - 1) * n + j],
                    "round {round}: row {j} of the [{n}, {k}] transpose is not \
                     backed by this tensor's data"
                );
            }

            match first_visits.iter().find(|v| v.shape == (k, n)) {
                None => first_visits.push(FirstVisit {
                    shape: (k, n),
                    transpose: bt,
                }),
                Some(first) => assert!(
                    Arc::ptr_eq(&first.transpose, &bt),
                    "round {round}: [{k}, {n}] at address {base:#x} missed the entry \
                     its first visit created, so the key is not stable"
                ),
            }
        }

        assert_eq!(
            reused,
            ROUNDS.len() - 1,
            "every round after the first must land on the address an earlier \
             round already cached under a different geometry"
        );
        assert_eq!(
            cache.len(),
            4,
            "one entry per distinct geometry at this address: {} entries means the \
             key is either collapsing distinct shapes or failing to reuse them",
            cache.len()
        );
    }

    /// Characterizes the limitation the key does **not** remove: one address
    /// holding a *same-shaped* tensor with different contents (a second model
    /// whose mmap lands on a recycled address) still hits the cached entry.
    /// That is a lifetime problem, and this is why `clear_weight_transpose_caches`
    /// on `Executor` drop stays load-bearing.
    #[test]
    fn same_address_same_shape_is_stale_until_cleared() {
        let cache = cache();
        let mut buf = vec![0.0f32; 4 * 6];
        fill(&mut buf, 1.0);
        let first = cache.get_or_insert_transpose(&buf, 4, 6).unwrap().to_vec();

        fill(&mut buf, 900.0);
        let stale = cache.get_or_insert_transpose(&buf, 4, 6).unwrap();
        assert_eq!(
            stale.as_slice(),
            first.as_slice(),
            "documented behaviour: identical keys share one entry"
        );

        cache.clear();
        let fresh = cache.get_or_insert_transpose(&buf, 4, 6).unwrap();
        assert_eq!(fresh.as_slice(), reference_transpose(&buf, 4, 6));
    }

    /// Fail closed rather than transposing out of bounds when the caller's
    /// geometry disagrees with its buffer.
    #[test]
    fn geometry_mismatch_returns_none() {
        let cache = cache();
        let buf = vec![1.0f32; 12];
        assert!(cache.get_or_insert_transpose(&buf, 3, 4).is_some());
        assert!(cache.get_or_insert_transpose(&buf, 3, 5).is_none());
        assert!(cache.get_or_insert_transpose(&buf, 4, 4).is_none());
        assert!(
            cache.get_or_insert_transpose(&buf, usize::MAX, 2).is_none(),
            "k * n overflow must fail closed, not wrap"
        );
        assert_eq!(cache.len(), 1, "rejected geometries must not be cached");
    }

    /// Zero-element matrices are served without growing the cache.
    #[test]
    fn zero_sized_transposes_are_not_cached() {
        let cache = cache();
        let empty: Vec<f32> = Vec::new();
        for (k, n) in [(0usize, 0usize), (0, 8), (8, 0)] {
            let bt = cache
                .get_or_insert_transpose(&empty, k, n)
                .unwrap_or_else(|| panic!("zero-size [{k}, {n}] must be served"));
            assert!(bt.is_empty());
        }
        assert!(cache.is_empty(), "zero-size entries must not be cached");
    }

    /// `clear` drops entries so a later model cannot inherit them.
    #[test]
    fn clear_drops_entries() {
        let cache = cache();
        let buf = vec![2.0f32; 40];
        cache.get_or_insert_transpose(&buf, 5, 8).unwrap();
        assert_eq!(cache.len(), 1);
        cache.clear();
        assert_eq!(cache.len(), 0);
        let again = cache.get_or_insert_transpose(&buf, 5, 8).unwrap();
        assert_eq!(again.as_slice(), reference_transpose(&buf, 5, 8));
    }

    /// Concurrent first-touch of one key converges on a single shared entry and
    /// every thread observes correct data.
    #[test]
    fn concurrent_first_touch_shares_one_entry() {
        let cache = Arc::new(cache());
        let mut buf = vec![0.0f32; 64 * 32];
        fill(&mut buf, 3.0);
        let buf = Arc::new(buf);
        let expect = reference_transpose(&buf, 64, 32);

        let mut handles = Vec::new();
        for _ in 0..8 {
            let cache = Arc::clone(&cache);
            let buf = Arc::clone(&buf);
            handles.push(std::thread::spawn(move || {
                let bt = cache.get_or_insert_transpose(&buf, 64, 32).expect("hit");
                (Arc::as_ptr(&bt) as usize, bt)
            }));
        }
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        for (_, bt) in &results {
            assert_eq!(bt.as_slice(), expect.as_slice());
        }
        let first = results[0].0;
        assert!(
            results.iter().all(|(p, _)| *p == first),
            "racing threads must converge on one cached allocation"
        );
        assert_eq!(cache.len(), 1);
    }

    /// Concurrent insertion of *different* keys (the address-reuse race, played
    /// out in parallel) keeps every entry distinct and correct.
    #[test]
    fn concurrent_distinct_keys_stay_distinct() {
        let cache = Arc::new(cache());
        let shapes = [(4usize, 6usize), (6, 4), (8, 3), (3, 8)];
        let mut buf = vec![0.0f32; 24];
        fill(&mut buf, 5.0);
        let buf = Arc::new(buf);

        let handles: Vec<_> = shapes
            .iter()
            .map(|&(k, n)| {
                let cache = Arc::clone(&cache);
                let buf = Arc::clone(&buf);
                std::thread::spawn(move || {
                    let bt = cache.get_or_insert_transpose(&buf, k, n).expect("hit");
                    assert_eq!(bt.as_slice(), reference_transpose(&buf, k, n));
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(cache.len(), shapes.len());
    }

    /// The tiled/parallel transpose matches the naive reference across tile and
    /// thread boundaries.
    #[test]
    fn tiled_transpose_matches_reference() {
        for (k, n) in [(1usize, 1usize), (3, 5), (64, 64), (65, 63), (129, 7)] {
            let mut buf = vec![0.0f32; k * n];
            fill(&mut buf, 0.125);
            assert_eq!(
                transpose_row_major(&buf, k, n),
                reference_transpose(&buf, k, n),
                "[{k}, {n}] transpose mismatch"
            );
        }
    }

    /// The process-global entry points reach their own dtype's cache and honour
    /// the same total key. Assertions are monotone (`>=` against a baseline)
    /// because other tests in this binary share the global caches.
    #[test]
    fn global_caches_are_per_dtype_and_keyed_by_shape() {
        let (f16_before, f32_before) = cache_sizes();

        let mut f32_buf = vec![0.0f32; 12];
        fill(&mut f32_buf, 1.0);
        let f16_buf: Vec<u16> = (0..12u16).map(|i| 0x3C00 + i).collect();

        let a = cached_transpose_f32(&f32_buf, 3, 4).expect("f32 [3,4]");
        let b = cached_transpose_f32(&f32_buf, 4, 3).expect("f32 [4,3]");
        assert_eq!(a.as_slice(), reference_transpose(&f32_buf, 3, 4));
        assert_ne!(a.as_slice(), b.as_slice());

        let c = cached_transpose_f16(&f16_buf, 3, 4).expect("f16 [3,4]");
        assert_eq!(c.len(), 12);
        for j in 0..4 {
            for i in 0..3 {
                assert_eq!(c[j * 3 + i], f16_buf[i * 4 + j]);
            }
        }

        let (f16_after, f32_after) = cache_sizes();
        assert!(
            f32_after >= f32_before + 2,
            "both f32 shapes must be cached"
        );
        assert!(f16_after > f16_before, "the f16 entry must be cached");
    }
}

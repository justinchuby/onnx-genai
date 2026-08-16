//! A recycling front-end for *large* host allocations.
//!
//! # The specific thing this fixes
//!
//! [`HostAllocator`] is a thin wrapper over `std::alloc`, and its doc comment
//! records a real earlier result: an arena layered over the system allocator was
//! *slower* than the system allocator. That result stands for small
//! allocations. glibc serves those from per-thread caches, so a pool on top adds
//! a lock without removing one.
//!
//! Large allocations are a different mechanism. Above `M_MMAP_THRESHOLD`
//! (128 KiB by default, and glibc's dynamic adjustment does not apply to a
//! request that is freed at the same size it was taken) `malloc` calls `mmap`
//! and `free` calls `munmap`. A fresh mapping is *demand-zeroed by the kernel*:
//! the first write to each page traps, and the kernel zeroes a page — or, under
//! transparent huge pages, two megabytes — before the store can retire. That
//! cost is proportional to the buffer, it is paid again on every run, and no
//! amount of work inside a kernel removes it.
//!
//! The native CPU EP hits this on every inference. Graph outputs are handed to
//! the caller by *moving* the produced buffer out of the executor
//! (`try_move_host_output`), which is zero-copy but forfeits that value's
//! cross-run buffer reuse, so the next run allocates the output afresh. A
//! 180 MB softmax output is therefore mmap'd, faulted in page by page, and
//! munmap'd — every single run. ONNX Runtime does not pay this, because its CPU
//! allocator hands back memory it already owns.
//!
//! # What this does
//!
//! On `deallocate`, a block whose size is in the cached band is pushed onto a
//! free list keyed by its **exact** `(bytes, align)` pair instead of being
//! returned to the system. On `allocate`, an exact match is popped and handed
//! back. Anything outside the band is delegated to the inner allocator
//! untouched, so the small-allocation path this crate already measured as
//! optimal is not changed at all.
//!
//! # Why exact-match rather than size classes
//!
//! A size-class pool answers a request for `n` bytes with a block of `m >= n`,
//! which means the block's true `Layout` no longer matches the caller's view of
//! it. Every subsequent size-carrying operation — most importantly the final
//! `dealloc`, which in Rust *must* be given the originating layout — then has to
//! carry the physical size separately from the requested one. Exact match keeps
//! `(bytes, align)` an invariant of the block for its whole life, so a cached
//! block is indistinguishable from a fresh one and the eventual real free uses
//! the same layout the real allocation used.
//!
//! It also costs nothing here. Decode is a loop over a small set of stable
//! shapes; the second iteration asks for exactly what the first one released.
//!
//! # Safety argument
//!
//! [`DeviceAllocator`] requires that "a region becomes reusable only once its
//! matching `deallocate` has been called". A block enters this cache *from*
//! `deallocate`, i.e. at the moment its sole owner relinquished it, and leaves
//! the cache into exactly one `allocate`, under a lock. So a cached block is
//! never live twice, which is the property the trait asks for, and recycling is
//! compliant rather than an exception to the contract.
//!
//! Two consequences worth stating plainly:
//!
//! * **Recycled memory is not zeroed.** Neither is `std::alloc::alloc`'s, so the
//!   contract is unchanged — but a fresh `mmap` *happens* to be zero, and a
//!   kernel that only partially writes its output would have been silently
//!   masked by that accident. Under `debug_assertions` every recycled block is
//!   poisoned with `0xA5` on the way out, so such a kernel fails loudly in tests
//!   rather than passing on an accident. Recycling within a single value's
//!   buffer is already how the executor reuses intermediates, so this does not
//!   introduce a new contract, only a new way to notice a violation of it.
//! * **Cached bytes are retained, not leaked.** They are live process memory
//!   under a hard cap; the cap is enforced *before* insertion, and anything that
//!   does not fit is freed immediately through the inner allocator.

use std::collections::HashMap;
use std::ptr::NonNull;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::MemoryError;
use crate::allocator::{DeviceAllocator, DeviceKey, HostAllocator};

/// Smallest allocation this cache will retain.
///
/// glibc's `M_MMAP_THRESHOLD` starts at 128 KiB. Below it `free` returns the
/// block to an arena that already reuses it without a syscall or a page fault,
/// so caching adds a lock and removes nothing — which is the earlier arena
/// result this module deliberately does not relitigate. 256 KiB leaves a margin
/// above the threshold so a block that glibc would have kept is not stolen from
/// the path that handles it better.
pub const MIN_CACHED_BYTES: usize = 256 * 1024;

/// Largest allocation this cache will retain.
///
/// A single multi-hundred-megabyte block held on a free list is memory the host
/// cannot use for anything else, and the run-to-run saving on one such block is
/// already captured by the cap on total retained bytes. 1 GiB is above every
/// per-tensor size in the decode and prefill shapes this targets.
pub const MAX_CACHED_BYTES: usize = 1024 * 1024 * 1024;

/// Default ceiling on total retained bytes across all size classes.
///
/// Sized to hold the working set of one decode loop (a few large activations and
/// the present-KV outputs) without turning the cache into a second heap. Tune
/// with `ONNX_GENAI_HOST_ALLOC_CACHE_BYTES`; `0` disables retention entirely.
pub const DEFAULT_CACHE_BUDGET_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// Number of independent shards.
///
/// One allocator serves every session on the device, so concurrent sessions are
/// the normal case. Sharding by size class keeps two sessions working on
/// different shapes off the same lock. A power of two so the index is a mask.
const SHARDS: usize = 8;

fn shard_index(bytes: usize, align: usize) -> usize {
    // Mix the size class rather than the raw byte count: sizes in one decode
    // loop are often related by small factors, and the low bits of a byte count
    // are usually zero.
    let class = bytes.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(17) ^ align;
    class % SHARDS
}

/// Cached free lists for one shard.
#[derive(Debug, Default)]
struct Shard {
    /// `(bytes, align) -> pointers`, all of exactly that layout.
    blocks: HashMap<(usize, usize), Vec<NonNull<u8>>>,
}

// SAFETY: the pointers are owned host allocations that no other handle aliases
// while they sit here (a block enters only from `deallocate`, at which point its
// owner has relinquished it, and leaves into exactly one `allocate`). The
// `Mutex` serialises all access, so moving the shard between threads cannot
// produce two live handles to one block.
unsafe impl Send for Shard {}

/// Counters for tests and for the profiling documentation.
#[derive(Debug, Default)]
pub struct LargeAllocCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub retained: u64,
    pub rejected: u64,
    pub retained_bytes: u64,
}

/// A [`DeviceAllocator`] that recycles large host blocks and delegates
/// everything else to an inner allocator.
#[derive(Debug)]
pub struct LargeAllocCache<A: DeviceAllocator = HostAllocator> {
    inner: A,
    shards: [Mutex<Shard>; SHARDS],
    budget_bytes: usize,
    retained_bytes: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    retained: AtomicU64,
    rejected: AtomicU64,
}

impl Default for LargeAllocCache<HostAllocator> {
    fn default() -> Self {
        Self::new(HostAllocator, budget_from_env())
    }
}

/// Read the retention budget from the environment.
///
/// `ONNX_GENAI_HOST_ALLOC_CACHE_BYTES` sets the cap directly. `0` turns
/// retention off, which is the escape hatch for a host where holding the memory
/// matters more than the page faults, and the control arm for A/B measurement.
fn budget_from_env() -> usize {
    match std::env::var("ONNX_GENAI_HOST_ALLOC_CACHE_BYTES") {
        Ok(raw) => raw
            .trim()
            .parse::<usize>()
            .unwrap_or(DEFAULT_CACHE_BUDGET_BYTES),
        Err(_) => DEFAULT_CACHE_BUDGET_BYTES,
    }
}

impl<A: DeviceAllocator> LargeAllocCache<A> {
    pub fn new(inner: A, budget_bytes: usize) -> Self {
        Self {
            inner,
            shards: std::array::from_fn(|_| Mutex::new(Shard::default())),
            budget_bytes,
            retained_bytes: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            retained: AtomicU64::new(0),
            rejected: AtomicU64::new(0),
        }
    }

    /// Whether a block of this layout is eligible for retention.
    ///
    /// Zero-sized requests are excluded because `allocate` rounds them up to one
    /// byte, so the layout the caller sees and the layout the block was taken
    /// with would disagree.
    fn cacheable(&self, bytes: usize, align: usize) -> bool {
        self.budget_bytes > 0
            && (MIN_CACHED_BYTES..=MAX_CACHED_BYTES).contains(&bytes)
            && align.is_power_of_two()
    }

    pub fn stats(&self) -> LargeAllocCacheStats {
        LargeAllocCacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            retained: self.retained.load(Ordering::Relaxed),
            rejected: self.rejected.load(Ordering::Relaxed),
            retained_bytes: self.retained_bytes.load(Ordering::Relaxed),
        }
    }

    /// Free every retained block through the inner allocator.
    pub fn drain(&self) {
        for shard in &self.shards {
            let mut guard = shard.lock().unwrap_or_else(|poison| poison.into_inner());
            for ((bytes, align), pointers) in guard.blocks.drain() {
                for ptr in pointers {
                    self.retained_bytes
                        .fetch_sub(bytes as u64, Ordering::Relaxed);
                    // SAFETY: `ptr` came from `self.inner.allocate(bytes, align)`
                    // and is keyed here by that exact layout. It is being removed
                    // from the cache under the shard lock, so no other handle to
                    // it exists.
                    unsafe { self.inner.deallocate(ptr, bytes, align) };
                }
            }
        }
    }
}

impl<A: DeviceAllocator> Drop for LargeAllocCache<A> {
    fn drop(&mut self) {
        self.drain();
    }
}

impl<A: DeviceAllocator> DeviceAllocator for LargeAllocCache<A> {
    fn allocate(&self, bytes: usize, align: usize) -> Result<NonNull<u8>, MemoryError> {
        if self.cacheable(bytes, align) {
            let shard = &self.shards[shard_index(bytes, align)];
            let popped = {
                let mut guard = shard.lock().unwrap_or_else(|poison| poison.into_inner());
                guard
                    .blocks
                    .get_mut(&(bytes, align))
                    .and_then(|pointers| pointers.pop())
            };
            if let Some(ptr) = popped {
                self.retained_bytes
                    .fetch_sub(bytes as u64, Ordering::Relaxed);
                self.hits.fetch_add(1, Ordering::Relaxed);
                #[cfg(debug_assertions)]
                {
                    // A fresh `mmap` is zero; a recycled block is not. A kernel
                    // that only partially writes its output would be masked by
                    // that accident, so make the difference loud in tests.
                    // SAFETY: `ptr` is a unique, live allocation of `bytes`
                    // bytes just removed from the cache; no other handle exists.
                    unsafe { std::ptr::write_bytes(ptr.as_ptr(), 0xA5, bytes) };
                }
                return Ok(ptr);
            }
            self.misses.fetch_add(1, Ordering::Relaxed);
        }
        self.inner.allocate(bytes, align)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, bytes: usize, align: usize) {
        if self.cacheable(bytes, align) {
            // Enforce the cap *before* inserting, so retained bytes can never
            // exceed the budget even transiently.
            let admitted = self
                .retained_bytes
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |held| {
                    let next = held.saturating_add(bytes as u64);
                    (next <= self.budget_bytes as u64).then_some(next)
                })
                .is_ok();
            if admitted {
                let shard = &self.shards[shard_index(bytes, align)];
                let mut guard = shard.lock().unwrap_or_else(|poison| poison.into_inner());
                guard.blocks.entry((bytes, align)).or_default().push(ptr);
                self.retained.fetch_add(1, Ordering::Relaxed);
                return;
            }
            self.rejected.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: delegated to this method's contract — `ptr` came from
        // `allocate` with this exact layout, and this path did not retain it.
        unsafe { self.inner.deallocate(ptr, bytes, align) };
    }

    fn device(&self) -> DeviceKey {
        self.inner.device()
    }

    fn commits_on_demand(&self) -> bool {
        self.inner.commits_on_demand()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache() -> LargeAllocCache<HostAllocator> {
        LargeAllocCache::new(HostAllocator, DEFAULT_CACHE_BUDGET_BYTES)
    }

    /// The point of the module: the second request for a layout that was just
    /// released must be served from the cache rather than the system.
    #[test]
    fn a_released_large_block_is_handed_back_to_the_next_matching_request() {
        let cache = cache();
        let bytes = MIN_CACHED_BYTES * 2;
        let first = cache.allocate(bytes, 64).unwrap();
        // SAFETY: `first` came from `allocate` with this layout.
        unsafe { cache.deallocate(first, bytes, 64) };
        let second = cache.allocate(bytes, 64).unwrap();
        assert_eq!(
            first, second,
            "the released block must be recycled, not re-taken from the system"
        );
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1, "the first request necessarily missed");
        // SAFETY: as above.
        unsafe { cache.deallocate(second, bytes, 64) };
    }

    /// Exact match is the whole safety story: a block must never be served to a
    /// request whose layout differs, in either direction, because the final real
    /// free is performed with the requesting layout.
    #[test]
    fn a_cached_block_is_never_served_to_a_different_layout() {
        let cache = cache();
        let bytes = MIN_CACHED_BYTES * 4;
        let block = cache.allocate(bytes, 64).unwrap();
        // SAFETY: `block` came from `allocate` with this layout.
        unsafe { cache.deallocate(block, bytes, 64) };

        let smaller = cache.allocate(bytes - 4096, 64).unwrap();
        assert_ne!(smaller, block, "a smaller request must not take the block");
        let wider = cache.allocate(bytes, 128).unwrap();
        assert_ne!(
            wider, block,
            "a request with a different alignment must not take the block"
        );
        let exact = cache.allocate(bytes, 64).unwrap();
        assert_eq!(exact, block, "the exact layout must still find it");

        // SAFETY: each pointer is freed with the layout it was allocated with.
        unsafe {
            cache.deallocate(smaller, bytes - 4096, 64);
            cache.deallocate(wider, bytes, 128);
            cache.deallocate(exact, bytes, 64);
        }
    }

    /// Small allocations must reach the system allocator untouched. This is the
    /// path an earlier arena made slower, and the reason the band has a floor.
    #[test]
    fn allocations_below_the_floor_are_never_retained() {
        let cache = cache();
        let bytes = MIN_CACHED_BYTES - 1;
        let ptr = cache.allocate(bytes, 64).unwrap();
        // SAFETY: `ptr` came from `allocate` with this layout.
        unsafe { cache.deallocate(ptr, bytes, 64) };
        let stats = cache.stats();
        assert_eq!(stats.retained, 0, "a sub-floor block must not be retained");
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0, "a sub-floor request must not be counted");
        assert_eq!(stats.retained_bytes, 0);
    }

    /// Above the ceiling the run-to-run saving is not worth pinning the host's
    /// memory, so those blocks go straight back.
    #[test]
    fn allocations_above_the_ceiling_are_never_retained() {
        let cache = LargeAllocCache::new(HostAllocator, DEFAULT_CACHE_BUDGET_BYTES);
        assert!(
            !cache.cacheable(MAX_CACHED_BYTES + 1, 64),
            "a block larger than the ceiling must be ineligible"
        );
        assert!(cache.cacheable(MAX_CACHED_BYTES, 64));
    }

    /// The cap must hold *before* insertion, not be repaired afterwards, or a
    /// burst of frees would spike retained memory past the budget.
    #[test]
    fn the_budget_is_never_exceeded_even_transiently() {
        let bytes = MIN_CACHED_BYTES;
        let budget = bytes * 2;
        let cache = LargeAllocCache::new(HostAllocator, budget);
        let blocks: Vec<_> = (0..4).map(|_| cache.allocate(bytes, 64).unwrap()).collect();
        for &ptr in &blocks {
            // SAFETY: each came from `allocate` with this layout.
            unsafe { cache.deallocate(ptr, bytes, 64) };
        }
        let stats = cache.stats();
        assert_eq!(stats.retained, 2, "only two blocks fit in the budget");
        assert_eq!(stats.rejected, 2, "the rest must be freed, not held");
        assert!(
            stats.retained_bytes <= budget as u64,
            "retained {} exceeds budget {budget}",
            stats.retained_bytes
        );
    }

    /// A zero budget must make the wrapper transparent, which is the control arm
    /// every A/B measurement in the benchmark doc relies on.
    #[test]
    fn a_zero_budget_disables_retention_entirely() {
        let cache = LargeAllocCache::new(HostAllocator, 0);
        let bytes = MIN_CACHED_BYTES * 2;
        let first = cache.allocate(bytes, 64).unwrap();
        // SAFETY: `first` came from `allocate` with this layout.
        unsafe { cache.deallocate(first, bytes, 64) };
        let stats = cache.stats();
        assert_eq!(stats.retained, 0);
        assert_eq!(stats.retained_bytes, 0);
        let second = cache.allocate(bytes, 64).unwrap();
        // SAFETY: as above.
        unsafe {
            cache.deallocate(second, bytes, 64);
        }
    }

    /// Two live allocations must never share a region — the trait's central
    /// promise, and the one a pooling bug breaks silently.
    #[test]
    fn concurrently_live_blocks_never_overlap() {
        let bytes = MIN_CACHED_BYTES;
        let cache = cache();
        let mut live = Vec::new();
        for _ in 0..64 {
            live.push(cache.allocate(bytes, 64).unwrap());
        }
        let mut seen: Vec<usize> = live.iter().map(|p| p.as_ptr() as usize).collect();
        seen.sort_unstable();
        let unique = {
            let mut copy = seen.clone();
            copy.dedup();
            copy.len()
        };
        assert_eq!(unique, live.len(), "two live blocks aliased");
        for window in seen.windows(2) {
            assert!(
                window[1] - window[0] >= bytes,
                "live blocks at {:#x} and {:#x} overlap for {bytes} bytes",
                window[0],
                window[1]
            );
        }
        // Release, re-take, and check the same property of the recycled set.
        for &ptr in &live {
            // SAFETY: each came from `allocate` with this layout.
            unsafe { cache.deallocate(ptr, bytes, 64) };
        }
        let again: Vec<_> = (0..64)
            .map(|_| cache.allocate(bytes, 64).unwrap())
            .collect();
        let mut addrs: Vec<usize> = again.iter().map(|p| p.as_ptr() as usize).collect();
        addrs.sort_unstable();
        addrs.dedup();
        assert_eq!(addrs.len(), again.len(), "recycling handed out an alias");
        for &ptr in &again {
            // SAFETY: as above.
            unsafe { cache.deallocate(ptr, bytes, 64) };
        }
    }

    /// Concurrent sessions are the normal case for one allocator, so hammer it
    /// from several threads and require that every block a thread holds is its
    /// own.
    #[test]
    fn concurrent_allocate_and_release_never_hands_out_a_live_block_twice() {
        use std::sync::Arc;
        use std::sync::atomic::AtomicBool;

        let cache = Arc::new(cache());
        let failed = Arc::new(AtomicBool::new(false));
        let live = Arc::new(Mutex::new(std::collections::HashSet::<usize>::new()));
        let mut handles = Vec::new();
        for thread in 0..8u32 {
            let cache = Arc::clone(&cache);
            let failed = Arc::clone(&failed);
            let live = Arc::clone(&live);
            handles.push(std::thread::spawn(move || {
                // A few distinct layouts so shards and classes both get exercised.
                let bytes = MIN_CACHED_BYTES * (1 + (thread as usize % 3));
                for _ in 0..200 {
                    let ptr = cache.allocate(bytes, 64).unwrap();
                    let address = ptr.as_ptr() as usize;
                    if !live.lock().unwrap().insert(address) {
                        failed.store(true, Ordering::Relaxed);
                    }
                    // Touch the block so a genuine aliasing bug corrupts state
                    // rather than merely being recorded.
                    // SAFETY: `ptr` is a live, uniquely-owned allocation of
                    // `bytes` bytes.
                    unsafe { std::ptr::write_bytes(ptr.as_ptr(), thread as u8, bytes) };
                    live.lock().unwrap().remove(&address);
                    // SAFETY: `ptr` came from `allocate` with this layout and is
                    // no longer recorded as live.
                    unsafe { cache.deallocate(ptr, bytes, 64) };
                }
            }));
        }
        for handle in handles {
            handle.join().unwrap();
        }
        assert!(
            !failed.load(Ordering::Relaxed),
            "a block was live in two threads at once"
        );
    }

    /// `drain` must return every retained block, so a caller that wants the
    /// memory back can have it and the cache's `Drop` does not leak.
    #[test]
    fn drain_releases_every_retained_block() {
        let cache = cache();
        let bytes = MIN_CACHED_BYTES;
        for _ in 0..4 {
            let ptr = cache.allocate(bytes, 64).unwrap();
            // SAFETY: came from `allocate` with this layout.
            unsafe { cache.deallocate(ptr, bytes, 64) };
        }
        assert!(cache.stats().retained_bytes > 0);
        cache.drain();
        assert_eq!(cache.stats().retained_bytes, 0);
    }

    /// The wrapper must not change what the device is; a caller uses this to
    /// decide whether a pointer may be dereferenced on the host.
    #[test]
    fn the_wrapper_reports_the_inner_device() {
        assert_eq!(cache().device(), HostAllocator.device());
    }
}

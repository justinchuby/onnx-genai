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
//! `malloc` calls `mmap` and `free` calls `munmap`. A fresh mapping is
//! *demand-zeroed by the kernel*: the first write to each page traps, and the
//! kernel zeroes a page — or, under transparent huge pages, two megabytes —
//! before the store can retire. That cost is proportional to the buffer, it is
//! paid again on every run, and no amount of work inside a kernel removes it.
//!
//! # Why the floor is not simply the mmap threshold
//!
//! An earlier revision of this module set the floor just above glibc's *initial*
//! `M_MMAP_THRESHOLD` of 128 KiB, reasoning that everything above it is mmapped
//! and therefore worth recycling. That reasoning is incomplete, and measurement
//! showed it costs a lock for no gain across most of the band it opened.
//!
//! glibc's threshold is **dynamic, and it adapts in exactly the situation this
//! cache targets**. In `_int_free`, when an mmapped chunk is released, glibc
//! sets `mmap_threshold = chunksize(p)` provided that size is at or below
//! `DEFAULT_MMAP_THRESHOLD_MAX` (32 MiB on 64-bit). So a buffer that is taken
//! and released at a stable size — a decode loop's activations, precisely this
//! cache's motivating workload — is mmapped *once*; from the second cycle
//! onward glibc serves it from an arena it already faulted in, and this cache
//! can only add a mutex on top of that.
//!
//! Above `DEFAULT_MMAP_THRESHOLD_MAX` the adaptation stops: the threshold is
//! never raised past 32 MiB, so a 180 MB output really is `mmap`'d, faulted page
//! by page and `munmap`'d on every single run, forever. That is the band where
//! recycling is decisive, and measurement puts the transition exactly there —
//! see `benches/large_alloc_cache.rs`, which finds ~8 ns per 4 KiB page below
//! the cliff (a plain store into resident memory) against ~200 ns per page above
//! it (a trap into the kernel plus a page zeroed).
//!
//! That reasoning is glibc's, though, and this crate also builds for musl,
//! Windows and macOS, whose allocators draw the line elsewhere and do not
//! document a stable equivalent. So the floor is **calibrated at construction by
//! measuring this exact effect** rather than hardcoding any one allocator's
//! constant, with a documented per-platform fallback if the measurement is
//! inconclusive and an environment override that beats both.
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

/// Fallback floor, used when calibration is disabled or inconclusive.
///
/// This is the point above which a released block stops being recycled *by the
/// system allocator* and starts costing a fresh kernel mapping on every cycle.
/// It is a property of the platform's allocator, not of this crate, and the
/// three platforms this ships on draw it in three different places:
///
/// * **glibc** — `_int_free` raises `mmap_threshold` to the size of any freed
///   mmapped chunk up to `DEFAULT_MMAP_THRESHOLD_MAX`, which is
///   `4 * 1024 * 1024 * sizeof(long) / 4` = **32 MiB** on 64-bit. Stably-sized
///   blocks below that are arena-served from their second cycle on; blocks above
///   it are mmapped forever.
/// * **musl** — mallocng services anything larger than its largest size class
///   (`MMAP_THRESHOLD`, 128 KiB) directly with `mmap`, and has **no dynamic
///   adaptation at all**, so every large cycle is a real mapping. The floor is
///   correspondingly lower.
/// * **Windows / macOS** — the NT heap's `VirtualAlloc` path and macOS
///   magazine-malloc's "large" allocator both take over well below 32 MiB but
///   neither documents a stable, adapting threshold. 1 MiB is a conservative
///   choice: high enough to stay out of the small-allocation path this crate
///   already measured as best left alone, low enough not to miss a real cliff.
///
/// [`LargeAllocCache::default`] does not use this value directly — it
/// [calibrates](calibrate_floor_bytes) against the allocator actually linked in,
/// which is the only allocator-independent answer. This constant is what that
/// calibration falls back to.
pub const FALLBACK_FLOOR_BYTES: usize = if cfg!(target_env = "musl") {
    256 * 1024
} else if cfg!(target_os = "linux") {
    32 * 1024 * 1024
} else {
    1024 * 1024
};

/// Largest allocation this cache will retain.
///
/// A single multi-hundred-megabyte block held on a free list is memory the host
/// cannot use for anything else, and the run-to-run saving on one such block is
/// already captured by the cap on total retained bytes. 1 GiB is above every
/// per-tensor size in the decode and prefill shapes this targets.
pub const MAX_CACHED_BYTES: usize = 1024 * 1024 * 1024;

/// Ceiling on total retained bytes across all size classes, on a host with
/// enough memory for it to be irrelevant.
///
/// Sized to hold the working set of one decode loop (a few large activations and
/// the present-KV outputs) without turning the cache into a second heap. Tune
/// with `ONNX_GENAI_HOST_ALLOC_CACHE_BYTES`; `0` disables retention entirely.
///
/// This is a *cap*, not the budget: [`default_budget_bytes`] takes the smaller
/// of this and a fraction of the memory the process is actually allowed, because
/// on a 1 GiB container a flat 2 GiB budget is not a tuning parameter, it is an
/// OOM kill.
pub const DEFAULT_CACHE_BUDGET_BYTES: usize = 2 * 1024 * 1024 * 1024;

/// Fraction of the process's memory allowance this cache may retain.
///
/// Retained bytes are live process memory that no allocation can use, so the
/// cache must stay a minority tenant. An eighth leaves the model weights, the KV
/// cache and the activations the other seven.
const BUDGET_SHARE_DENOMINATOR: u64 = 8;

/// Number of independent shards.
///
/// One allocator serves every session on the device, so concurrent sessions are
/// the normal case. Sharding by size class keeps two sessions working on
/// different shapes off the same lock. A power of two so the index is a mask.
const SHARDS: usize = 8;

/// How much memory this process is actually allowed to use.
///
/// A container's limit is not the machine's RAM, and the machine's RAM is what
/// every naive reading of "how much can I hold" returns. Under cgroup v2 the
/// limit lives in `memory.max` (literally `"max"` when unlimited); cgroup v1 put
/// it in `memory.limit_in_bytes` with a sentinel near `u64::MAX`. Both are read
/// here, most-specific first, and the machine's total is the floor of last
/// resort.
///
/// Returns `None` when nothing could be established, which on a non-Linux target
/// is the normal answer — there the flat cap applies.
fn process_memory_allowance() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let cgroup = [
            "/sys/fs/cgroup/memory.max",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        ]
        .iter()
        .find_map(|path| {
            let raw = std::fs::read_to_string(path).ok()?;
            let trimmed = raw.trim();
            if trimmed == "max" {
                return None;
            }
            let value = trimmed.parse::<u64>().ok()?;
            // cgroup v1 encodes "unlimited" as a huge page-aligned sentinel
            // rather than a word, so treat implausible values as absent.
            (value > 0 && value < (1 << 50)).then_some(value)
        });
        if cgroup.is_some() {
            return cgroup;
        }
        // MemAvailable is the kernel's own estimate of what can be handed out
        // without swapping, which is a better basis than MemTotal for deciding
        // how much to sit on.
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        for key in ["MemAvailable:", "MemTotal:"] {
            if let Some(line) = meminfo.lines().find(|l| l.starts_with(key))
                && let Some(kb) = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|v| v.parse::<u64>().ok())
            {
                return Some(kb.saturating_mul(1024));
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// The retention budget to use when the caller did not pick one.
///
/// `ONNX_GENAI_HOST_ALLOC_CACHE_BYTES` wins outright when set — including `0`,
/// which disables retention and is the control arm for A/B measurement.
/// Otherwise the budget is the smaller of [`DEFAULT_CACHE_BUDGET_BYTES`] and a
/// [share](BUDGET_SHARE_DENOMINATOR) of what the process is allowed, so the
/// same binary is not a 2 GiB squatter inside a 1 GiB container.
pub fn default_budget_bytes() -> usize {
    if let Ok(raw) = std::env::var("ONNX_GENAI_HOST_ALLOC_CACHE_BYTES")
        && let Ok(explicit) = raw.trim().parse::<usize>()
    {
        return explicit;
    }
    budget_for_allowance(process_memory_allowance())
}

/// The budget policy, split out so it can be tested without a machine that has
/// the memory in question.
fn budget_for_allowance(allowance: Option<u64>) -> usize {
    let Some(allowance) = allowance else {
        return DEFAULT_CACHE_BUDGET_BYTES;
    };
    let share = allowance / BUDGET_SHARE_DENOMINATOR;
    let capped = share.min(DEFAULT_CACHE_BUDGET_BYTES as u64);
    // `usize::try_from` cannot fail on the 64-bit targets this runs on, but a
    // 32-bit build must saturate rather than wrap.
    usize::try_from(capped).unwrap_or(usize::MAX)
}

/// Measure the size at which the system allocator stops recycling and starts
/// costing a fresh kernel mapping.
///
/// # What is being measured
///
/// One `alloc` → first-touch every page → `free` cycle, repeated. The
/// first-touch is the whole point: a fresh `mmap` is cheap to *obtain* and
/// expensive to *use*, because each page traps on its first store so the kernel
/// can zero it. Timing an allocation without touching it measures nothing.
///
/// A size that the allocator recycles internally is already resident, so the
/// touch is a plain store — single-digit nanoseconds per page. A size it
/// re-maps every cycle pays the trap — a couple of hundred nanoseconds per page.
/// The gap is more than an order of magnitude, which is why this can be decided
/// from a handful of samples on a noisy machine.
///
/// # Why calibrate rather than hardcode
///
/// [`FALLBACK_FLOOR_BYTES`] documents where glibc, musl, macOS and Windows each
/// draw this line, and they disagree by two orders of magnitude. A user may also
/// have `LD_PRELOAD`ed jemalloc or tcmalloc, set `MALLOC_MMAP_THRESHOLD_`, or
/// linked mimalloc — none of which this crate can see at compile time. Measuring
/// answers for the allocator actually present.
///
/// # Cost and failure
///
/// The ladder is geometric and bounded, and the whole probe is budgeted at a few
/// milliseconds; it runs once per cache. If the ladder is exhausted without a
/// clear cliff — a plausible outcome on a heavily contended machine, or with an
/// allocator that recycles everything — the answer is [`FALLBACK_FLOOR_BYTES`],
/// which is never worse than the previous hardcoded behaviour.
pub fn calibrate_floor_bytes() -> usize {
    /// Below this there is no question: these are small allocations, which this
    /// crate has already measured as best left to the system allocator.
    const LADDER_START: usize = 256 * 1024;
    /// Past this the cache is unconditionally worthwhile on every allocator
    /// examined, so there is nothing left to learn by probing further.
    const LADDER_END: usize = 256 * 1024 * 1024;
    /// Enough to see through scheduler noise, few enough to stay cheap. The
    /// signal is >10x, so the median of five is ample.
    const SAMPLES: usize = 5;
    /// A cycle costing at least this much per page is trapping into the kernel.
    /// Resident stores are ~10x under it and faults ~2x over it, so the decision
    /// does not sit near the boundary on any machine measured.
    const FAULTING_NS_PER_PAGE: f64 = 60.0;
    /// Hard stop, so a pathological machine cannot make construction hang.
    const PROBE_BUDGET: std::time::Duration = std::time::Duration::from_millis(50);

    let started = std::time::Instant::now();
    let mut size = LADDER_START;
    while size <= LADDER_END {
        if started.elapsed() > PROBE_BUDGET {
            return FALLBACK_FLOOR_BYTES;
        }
        // One untimed cycle first: it is this free that lets an adaptive
        // allocator raise its threshold, and measuring before it would report
        // every size as faulting.
        touch_cycle(size);
        let mut samples = [0f64; SAMPLES];
        for sample in &mut samples {
            let start = std::time::Instant::now();
            touch_cycle(size);
            *sample = start.elapsed().as_nanos() as f64;
        }
        samples.sort_by(|a, b| a.partial_cmp(b).expect("durations are never NaN"));
        let pages = (size / 4096).max(1) as f64;
        if samples[SAMPLES / 2] / pages >= FAULTING_NS_PER_PAGE {
            return size;
        }
        size *= 2;
    }
    FALLBACK_FLOOR_BYTES
}

/// One allocate → touch every page → free cycle through the global allocator.
fn touch_cycle(bytes: usize) {
    let layout = std::alloc::Layout::from_size_align(bytes, 64).expect("probe layout is valid");
    // SAFETY: `layout` has a non-zero size, and the block is written only within
    // `bytes` and freed exactly once with the layout it was taken with.
    unsafe {
        let ptr = std::alloc::alloc(layout);
        if ptr.is_null() {
            return;
        }
        let mut off = 0;
        while off < bytes {
            // Volatile so the loop cannot be optimised away: the fault it
            // provokes *is* the measurement.
            std::ptr::write_volatile(ptr.add(off), 0x5Au8);
            off += 4096;
        }
        std::alloc::dealloc(ptr, layout);
    }
}

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
    /// Smallest retained size, measured against the allocator actually linked
    /// in rather than assumed. See [`calibrate_floor_bytes`].
    floor_bytes: usize,
    retained_bytes: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    retained: AtomicU64,
    rejected: AtomicU64,
}

impl Default for LargeAllocCache<HostAllocator> {
    fn default() -> Self {
        Self::with_floor(HostAllocator, default_budget_bytes(), floor_from_env())
    }
}

/// Read the retention floor from the environment, else measure it.
///
/// `ONNX_GENAI_HOST_ALLOC_CACHE_FLOOR_BYTES` overrides the probe outright, for a
/// deployment whose allocator this crate guesses wrong or which wants the old
/// behaviour back verbatim.
fn floor_from_env() -> usize {
    if let Ok(raw) = std::env::var("ONNX_GENAI_HOST_ALLOC_CACHE_FLOOR_BYTES")
        && let Ok(explicit) = raw.trim().parse::<usize>()
    {
        return explicit;
    }
    calibrate_floor_bytes()
}

impl<A: DeviceAllocator> LargeAllocCache<A> {
    /// Construct with an explicit budget and a calibrated floor.
    pub fn new(inner: A, budget_bytes: usize) -> Self {
        Self::with_floor(inner, budget_bytes, calibrate_floor_bytes())
    }

    /// Construct with both bounds given explicitly.
    ///
    /// Tests and benchmarks need a floor that does not depend on the machine
    /// they run on; a caller that has measured its own workload may know better
    /// than the probe.
    pub fn with_floor(inner: A, budget_bytes: usize, floor_bytes: usize) -> Self {
        Self {
            inner,
            shards: std::array::from_fn(|_| Mutex::new(Shard::default())),
            budget_bytes,
            floor_bytes,
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
            && (self.floor_bytes..=MAX_CACHED_BYTES).contains(&bytes)
            && align.is_power_of_two()
    }

    /// The smallest size this instance retains, after calibration and any
    /// environment override.
    pub fn floor_bytes(&self) -> usize {
        self.floor_bytes
    }

    /// The retention cap in force for this instance.
    pub fn budget_bytes(&self) -> usize {
        self.budget_bytes
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

    /// Tests pin the floor rather than calibrating: a test must assert the same
    /// thing on a developer laptop, in CI and under a preloaded allocator, and
    /// `calibrate_floor_bytes` deliberately answers differently on each.
    const TEST_FLOOR: usize = 256 * 1024;

    fn cache() -> LargeAllocCache<HostAllocator> {
        LargeAllocCache::with_floor(HostAllocator, DEFAULT_CACHE_BUDGET_BYTES, TEST_FLOOR)
    }

    /// The point of the module: the second request for a layout that was just
    /// released must be served from the cache rather than the system.
    #[test]
    fn a_released_large_block_is_handed_back_to_the_next_matching_request() {
        let cache = cache();
        let bytes = TEST_FLOOR * 2;
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
        let bytes = TEST_FLOOR * 4;
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
        let bytes = TEST_FLOOR - 1;
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
        let cache =
            LargeAllocCache::with_floor(HostAllocator, DEFAULT_CACHE_BUDGET_BYTES, TEST_FLOOR);
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
        let bytes = TEST_FLOOR;
        let budget = bytes * 2;
        let cache = LargeAllocCache::with_floor(HostAllocator, budget, TEST_FLOOR);
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
        let cache = LargeAllocCache::with_floor(HostAllocator, 0, TEST_FLOOR);
        let bytes = TEST_FLOOR * 2;
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
        let bytes = TEST_FLOOR;
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
                let bytes = TEST_FLOOR * (1 + (thread as usize % 3));
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
        let bytes = TEST_FLOOR;
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
    /// The budget is a promise about *process* memory, and retained bytes are
    /// the only part of it this module can count. This checks the promise
    /// against the kernel's own number instead: run a decode-shaped loop long
    /// enough that an unbounded cache would be obvious, and require that
    /// resident memory settles rather than climbing with the iteration count.
    ///
    /// # Why this re-executes itself
    ///
    /// RSS is a property of the *process*, and `cargo test` runs the whole
    /// module's tests concurrently in one process. Sampling `/proc/self/statm`
    /// around this loop therefore also samples every other test's allocations,
    /// which made a first version of this test pass alone and fail in the suite.
    /// The workload runs in a child process that does nothing else, so the
    /// number measured is attributable to this loop and the assertion means what
    /// it says.
    #[cfg(target_os = "linux")]
    #[test]
    fn resident_memory_stays_bounded_across_a_long_allocation_loop() {
        const WORKER_VAR: &str = "ONNX_GENAI_RSS_PROBE_WORKER";
        const BUDGET: usize = 8 * 1024 * 1024;

        if std::env::var(WORKER_VAR).is_ok() {
            rss_probe_worker(BUDGET);
            return;
        }

        let exe = std::env::current_exe().expect("the test binary knows its own path");
        let output = std::process::Command::new(exe)
            .args([
                "--exact",
                "large_alloc_cache::tests::resident_memory_stays_bounded_across_a_long_allocation_loop",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(WORKER_VAR, "1")
            .output()
            .expect("the test binary can be re-executed");
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(
            output.status.success(),
            "the isolated worker failed:\n{stdout}{}",
            String::from_utf8_lossy(&output.stderr)
        );

        // `--nocapture` writes the worker's line into the middle of libtest's own
        // progress output, so the marker is searched for rather than anchored.
        const MARKER: &str = "rss-growth-bytes=";
        let at = stdout
            .find(MARKER)
            .unwrap_or_else(|| panic!("the worker reports its resident growth:\n{stdout}"));
        let growth: u64 = stdout[at + MARKER.len()..]
            .split(|c: char| !c.is_ascii_digit())
            .find(|token| !token.is_empty())
            .expect("the marker is followed by a number")
            .parse()
            .expect("the reported growth is a number");

        // The loop released roughly forty times more bytes than the budget
        // allows to be retained. A cache that grew with the iteration count
        // would be orders of magnitude over this; a bounded one grows by nothing
        // beyond the blocks it is entitled to hold.
        assert!(
            growth <= BUDGET as u64,
            "resident memory grew by {growth} bytes across the loop, over the \
             {BUDGET}-byte retention budget: the cache is not bounding itself"
        );
    }

    /// The body of the RSS test, run in a process of its own.
    #[cfg(target_os = "linux")]
    fn rss_probe_worker(budget: usize) {
        fn resident_bytes() -> u64 {
            let statm = std::fs::read_to_string("/proc/self/statm").expect("statm is readable");
            let pages: u64 = statm
                .split_whitespace()
                .nth(1)
                .expect("statm has a resident field")
                .parse()
                .expect("resident field is a number");
            pages * 4096
        }

        let cache = LargeAllocCache::with_floor(HostAllocator, budget, TEST_FLOOR);
        // Four distinct shapes, as a decode step produces, each well under the
        // budget so all four can be retained at once.
        let shapes = [TEST_FLOOR, TEST_FLOOR * 2, TEST_FLOOR * 3, TEST_FLOOR * 5];

        let touch = |rounds: usize| {
            for _ in 0..rounds {
                for &bytes in &shapes {
                    let ptr = cache.allocate(bytes, 64).expect("allocation succeeds");
                    // SAFETY: `ptr` is live and uniquely owned for `bytes`.
                    unsafe {
                        std::ptr::write_bytes(ptr.as_ptr(), 0x11, bytes);
                        cache.deallocate(ptr, bytes, 64);
                    }
                }
            }
        };

        // Warm up so the steady-state working set is already resident when the
        // baseline is taken; otherwise ordinary first-touch growth reads as
        // leakage.
        touch(50);
        let settled = resident_bytes();
        touch(2_000);
        let after = resident_bytes();

        assert!(
            cache.stats().retained_bytes <= budget as u64,
            "retained bytes exceeded the budget"
        );
        println!("rss-growth-bytes={}", after.saturating_sub(settled));
    }

    /// A graph output handed to the caller outlives the run that produced it.
    /// While the caller holds it, that block must never be handed to anybody
    /// else — the cache may only ever recycle blocks it has been given back.
    #[test]
    fn a_block_retained_by_its_caller_is_never_recycled_underneath_it() {
        let cache = cache();
        let bytes = TEST_FLOOR * 2;

        // The caller keeps this one, exactly as a moved-out graph output is kept
        // past the end of `run`.
        let held = cache.allocate(bytes, 64).expect("allocation succeeds");
        // SAFETY: `held` is live and uniquely owned.
        unsafe { std::ptr::write_bytes(held.as_ptr(), 0x27, bytes) };

        // Many further runs churn the same shape through the cache.
        let mut seen = Vec::new();
        for _ in 0..64 {
            let ptr = cache.allocate(bytes, 64).expect("allocation succeeds");
            assert_ne!(
                ptr, held,
                "a block still owned by the caller was handed out again"
            );
            seen.push(ptr);
            // SAFETY: `ptr` is live and uniquely owned; it is released below.
            unsafe { cache.deallocate(ptr, bytes, 64) };
        }

        // The retained block's contents must be exactly as its owner left them.
        // SAFETY: `held` has been live and untouched by anyone else throughout.
        let contents = unsafe { std::slice::from_raw_parts(held.as_ptr(), bytes) };
        assert!(
            contents.iter().all(|&b| b == 0x27),
            "a caller-held block was written through while it was still owned"
        );

        // SAFETY: `held` came from `allocate` with this layout and is released
        // exactly once, here.
        unsafe { cache.deallocate(held, bytes, 64) };
    }

    /// Calibration must return something a cache can actually be built with, on
    /// whatever allocator the test host happens to link.
    #[test]
    fn calibration_returns_a_usable_floor() {
        let floor = calibrate_floor_bytes();
        assert!(
            floor >= 256 * 1024,
            "a floor of {floor} bytes would put the cache into the small-allocation \
             path this crate measured as best left to the system allocator"
        );
        assert!(
            floor <= 256 * 1024 * 1024,
            "a floor of {floor} bytes would exclude the multi-hundred-megabyte \
             outputs this cache exists for"
        );
        assert!(
            floor.is_power_of_two(),
            "the ladder is geometric from 256 KiB"
        );
    }

    /// The probe is on the construction path of every EP, so its cost is paid by
    /// every session. It is budgeted at 50 ms; anything near that is a bug.
    #[test]
    fn calibration_is_cheap_enough_to_run_at_construction() {
        let start = std::time::Instant::now();
        let _ = calibrate_floor_bytes();
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_millis(250),
            "calibration took {elapsed:?}, which is too much to pay per cache"
        );
    }

    /// The budget must scale with what the process is actually allowed, because
    /// the failure mode of getting this wrong on a small container is an OOM
    /// kill rather than a slow run.
    #[test]
    fn the_budget_tracks_the_process_memory_allowance() {
        // A container far smaller than the flat default.
        assert_eq!(
            budget_for_allowance(Some(512 * 1024 * 1024)),
            64 * 1024 * 1024,
            "a 512 MiB container must not be handed a multi-gigabyte cache"
        );
        // A host large enough that the flat cap binds instead.
        assert_eq!(
            budget_for_allowance(Some(128 * 1024 * 1024 * 1024)),
            DEFAULT_CACHE_BUDGET_BYTES,
            "the flat cap must still bind on a large host"
        );
        // Exactly at the crossover.
        assert_eq!(
            budget_for_allowance(Some(
                DEFAULT_CACHE_BUDGET_BYTES as u64 * BUDGET_SHARE_DENOMINATOR
            )),
            DEFAULT_CACHE_BUDGET_BYTES
        );
        // Nothing could be established: fall back rather than guess low.
        assert_eq!(budget_for_allowance(None), DEFAULT_CACHE_BUDGET_BYTES);
    }

    /// A cgroup so small that an eighth of it is below one cached block. The
    /// budget must still be a number the cache can enforce, and enforcing it
    /// must mean retaining nothing rather than retaining one block anyway.
    #[test]
    fn a_tiny_allowance_disables_retention_rather_than_overshooting() {
        let budget = budget_for_allowance(Some(1024 * 1024));
        assert_eq!(budget, 128 * 1024);
        let cache = LargeAllocCache::with_floor(HostAllocator, budget, TEST_FLOOR);
        let bytes = TEST_FLOOR;
        let ptr = cache.allocate(bytes, 64).expect("allocation succeeds");
        // SAFETY: `ptr` came from `allocate` with this layout.
        unsafe { cache.deallocate(ptr, bytes, 64) };
        assert_eq!(
            cache.stats().retained_bytes,
            0,
            "a block larger than the whole budget must not be retained"
        );
        assert_eq!(cache.stats().rejected, 1);
    }

    /// The floor is what decides whether a size is retained at all, so a caller
    /// that overrides it must actually get the band it asked for.
    #[test]
    fn the_configured_floor_decides_what_is_retained() {
        let high = LargeAllocCache::with_floor(HostAllocator, DEFAULT_CACHE_BUDGET_BYTES, 1 << 20);
        assert_eq!(high.floor_bytes(), 1 << 20);
        let below = (1 << 20) - 4096;
        let ptr = high.allocate(below, 64).expect("allocation succeeds");
        // SAFETY: `ptr` came from `allocate` with this layout.
        unsafe { high.deallocate(ptr, below, 64) };
        assert_eq!(
            high.stats().retained,
            0,
            "a size below the configured floor must go straight to the system allocator"
        );

        let low = LargeAllocCache::with_floor(HostAllocator, DEFAULT_CACHE_BUDGET_BYTES, 4096);
        let ptr = low.allocate(below, 64).expect("allocation succeeds");
        // SAFETY: `ptr` came from `allocate` with this layout.
        unsafe { low.deallocate(ptr, below, 64) };
        assert_eq!(
            low.stats().retained,
            1,
            "the same size must be retained once the floor is lowered under it"
        );
    }

    #[test]
    fn the_wrapper_reports_the_inner_device() {
        assert_eq!(cache().device(), HostAllocator.device());
    }
}

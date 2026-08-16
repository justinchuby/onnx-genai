//! Where is the recycling floor, measured rather than assumed?
//!
//! [`LargeAllocCache`] retains blocks from `MIN_CACHED_BYTES` upward on the
//! premise that above the system allocator's mmap threshold every cycle costs a
//! fresh mapping and a kernel-zeroed page fault. That premise is only true for
//! part of the band. glibc raises `M_MMAP_THRESHOLD` to the size of an mmapped
//! chunk *when that chunk is freed* (`_int_free`, capped at
//! `DEFAULT_MMAP_THRESHOLD_MAX` = 32 MiB on 64-bit), so a size that is allocated
//! and released repeatedly stops being mmapped after its first cycle and is
//! served from an already-faulted arena thereafter.
//!
//! Both arms below allocate, first-touch every page — which is what a kernel
//! writing its output does, and the only way to make demand-zeroing visible —
//! and free. The only difference is whether the block came from the cache. The
//! ratio between them is the cache's actual value at that size, and the size at
//! which it stops being ~1.0 is the floor the constant should encode.

use std::hint::black_box;
use std::sync::Arc;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use onnx_runtime_memory_governor::allocator::{DeviceAllocator, HostAllocator};
use onnx_runtime_memory_governor::large_alloc_cache::LargeAllocCache;

/// Straddles glibc's 32 MiB `DEFAULT_MMAP_THRESHOLD_MAX` and the current
/// 256 KiB floor, with enough resolution either side to locate the cliff.
const SIZES: &[usize] = &[
    256 << 10,
    1 << 20,
    4 << 20,
    16 << 20,
    32 << 20,
    64 << 20,
    192 << 20,
];

fn label(bytes: usize) -> String {
    if bytes >= 1 << 20 {
        format!("{}MiB", bytes >> 20)
    } else {
        format!("{}KiB", bytes >> 10)
    }
}

/// Allocate, touch every page, free. `alloc` is whichever allocator is on test.
fn cycle<A: DeviceAllocator>(alloc: &A, bytes: usize, align: usize) {
    let ptr = alloc.allocate(bytes, align).expect("allocation failed");
    // SAFETY: `ptr` is a live, uniquely owned block of `bytes` bytes.
    unsafe {
        let base = ptr.as_ptr();
        let mut off = 0;
        while off < bytes {
            std::ptr::write_volatile(base.add(off), 0x5Au8);
            off += 4096;
        }
        black_box(base);
        alloc.deallocate(ptr, bytes, align);
    }
}

/// One size at a time, the friendliest possible case for the system allocator:
/// its dynamic threshold has a single size to adapt to.
fn bench_single_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("alloc_cycle_single_size");
    for &bytes in SIZES {
        group.throughput(Throughput::Bytes(bytes as u64));
        let system = HostAllocator;
        group.bench_with_input(BenchmarkId::new("system", label(bytes)), &bytes, |b, &n| {
            b.iter(|| cycle(&system, n, 64));
        });
        // A budget large enough that nothing is ever rejected, and a floor of 1
        // so the cache retains every size under test rather than only those
        // above the shipped constant.
        let cached = LargeAllocCache::with_floor(HostAllocator, 4 << 30, 1);
        // Prime the free list so the measured iterations are hits.
        cycle(&cached, bytes, 64);
        group.bench_with_input(BenchmarkId::new("cached", label(bytes)), &bytes, |b, &n| {
            b.iter(|| cycle(&cached, n, 64));
        });
    }
    group.finish();
}

/// Several live sizes interleaved, which is what an executor actually does. A
/// single dynamic threshold cannot fit every size at once, so this is where the
/// system allocator is least able to adapt.
fn bench_interleaved(c: &mut Criterion) {
    let mut group = c.benchmark_group("alloc_cycle_interleaved");
    // Four related-but-distinct sizes, as a decode loop produces.
    for &base in &[1usize << 20, 16 << 20, 64 << 20] {
        let mix = [base, base + (base / 4), base / 2, base + (base / 2)];
        let total: usize = mix.iter().sum();
        group.throughput(Throughput::Bytes(total as u64));
        let system = HostAllocator;
        group.bench_with_input(BenchmarkId::new("system", label(base)), &mix, |b, m| {
            b.iter(|| {
                for &n in m {
                    cycle(&system, n, 64);
                }
            });
        });
        let cached = LargeAllocCache::with_floor(HostAllocator, 4 << 30, 1);
        for &n in &mix {
            cycle(&cached, n, 64);
        }
        group.bench_with_input(BenchmarkId::new("cached", label(base)), &mix, |b, m| {
            b.iter(|| {
                for &n in m {
                    cycle(&cached, n, 64);
                }
            });
        });
    }
    group.finish();
}

/// The cache takes a shard lock that the system allocator's per-thread caches do
/// not. Below the floor that lock is pure overhead, and this is where it shows.
fn bench_threaded(c: &mut Criterion) {
    let mut group = c.benchmark_group("alloc_cycle_8_threads");
    for &bytes in &[256usize << 10, 4 << 20, 64 << 20] {
        group.throughput(Throughput::Bytes((bytes * 8) as u64));
        group.bench_with_input(BenchmarkId::new("system", label(bytes)), &bytes, |b, &n| {
            b.iter(|| {
                std::thread::scope(|s| {
                    for _ in 0..8 {
                        s.spawn(move || cycle(&HostAllocator, n, 64));
                    }
                });
            });
        });
        let cached = Arc::new(LargeAllocCache::with_floor(HostAllocator, 4 << 30, 1));
        for _ in 0..8 {
            cycle(cached.as_ref(), bytes, 64);
        }
        group.bench_with_input(BenchmarkId::new("cached", label(bytes)), &bytes, |b, &n| {
            b.iter(|| {
                std::thread::scope(|s| {
                    for _ in 0..8 {
                        let c = Arc::clone(&cached);
                        s.spawn(move || cycle(c.as_ref(), n, 64));
                    }
                });
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_single_size,
    bench_interleaved,
    bench_threaded
);
criterion_main!(benches);

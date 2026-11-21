//! Isolated-process startup probe: run under `/usr/bin/time -l` on macOS.
//! Copy this same source into the baseline tree for an identical control.
use onnx_runtime_memory_api::{DeviceAllocator, HostAllocator};
use onnx_runtime_memory_governor::LargeAllocCache;
use std::hint::black_box;

fn main() {
    let mode = std::env::args().nth(1).expect(
        "usage: host_cache_startup <construct|disabled|small|explicit>; measure each in a fresh process",
    );
    let budget = 64 * 1024 * 1024;
    let cache = match mode.as_str() {
        "construct" | "small" => LargeAllocCache::new(HostAllocator, budget),
        "disabled" => LargeAllocCache::new(HostAllocator, 0),
        "explicit" => LargeAllocCache::with_floor(HostAllocator, budget, 4096),
        _ => panic!("unknown mode {mode:?}; use construct, disabled, small or explicit"),
    };
    if mode != "construct" {
        let bytes = if mode == "disabled" { 512 * 1024 } else { 4096 };
        for _ in 0..32 {
            let ptr = cache
                .allocate(bytes, 64)
                .expect("allocate startup probe block");
            // SAFETY: a live uniquely owned block of `bytes` with this layout.
            unsafe {
                std::ptr::write_volatile(ptr.as_ptr(), 0x5a);
                std::ptr::write_volatile(ptr.as_ptr().add(bytes - 1), 0xa5);
                cache.deallocate(ptr, bytes, 64);
            }
        }
    }
    let stats = black_box(cache.stats());
    assert_eq!(stats.hits, if mode == "explicit" { 31 } else { 0 });
    assert_eq!(
        stats.retained_bytes,
        if mode == "explicit" { 4096 } else { 0 }
    );
    println!(
        "mode={mode} hits={} misses={} retained_bytes={}",
        stats.hits, stats.misses, stats.retained_bytes
    );
}

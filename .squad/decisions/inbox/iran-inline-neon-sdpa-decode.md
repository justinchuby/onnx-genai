# Decision: Inline NEON SDPA decode for small per-head work on macOS

**Date:** 2026-07-28T10:48:20Z
**Author:** Iran (Mac CPU Optimization Engineer)
**Status:** Accepted (implemented)

## Context

On macOS without MLAS, `sdpa_f32` dispatches all shapes to `sdpa_f32_accelerate`, which uses `cblas_sgemm` per `(batch, head)` tile, parallelized via Rayon. For decode (`q_seq=1`), each `cblas_sgemm` computes a 1×N matrix multiply (effectively a dot product or axpy). The cblas framework call + AMX dispatch setup costs ~2-3µs per invocation, and the path makes 2 cblas calls per head tile.

When `head_size × kv_seq` is small (early decode steps, small models like TinyStories-1M or Qwen-0.5B), this fixed overhead dominates the actual arithmetic.

## Attribution measurement

Microbenchmark (release mode, M1 Max):

| Shape | NEON µs | Accel µs | NEON faster |
|-------|---------|----------|-------------|
| qwen-0.5b, tk=32, Dh=64 | 6.7 | 42.8 | 6.4× |
| qwen-0.5b, tk=64, Dh=64 | 12.7 | 44.6 | 3.5× |
| qwen-0.5b, tk=128, Dh=64 | 25.9 | 50.2 | 1.9× |
| qwen-0.5b, tk=256, Dh=64 | 52.1 | 49.5 | ~even |
| tiny-1m, tk=32, Dh=4 | 4.3 | 48.4 | 11.3× |
| llama-7b, tk=128, Dh=128 | 104.3 | 76.4 | Accel wins |

Crossover: `kv_seq × max(head_size, v_head_size) ≈ 8192`. Below that, NEON wins; above, Accelerate's AMX throughput dominates.

## Decision

When `q_seq == 1` (decode) and `kv_seq × max(head_size, v_head_size) ≤ 8192`, bypass `sdpa_f32_accelerate` and use `sdpa_f32_neon` instead. This avoids cblas call overhead for tiny GEMVs.

## Portability rationale

The 8192 threshold depends on the ratio of cblas call overhead (~2-3µs) to NEON throughput (~0.1µs/1024 elements). Both scale with the hardware's SIMD width (128-bit NEON, constant across all Apple Silicon) and memory subsystem (same cache line size). The threshold is independent of core count, frequency, or AMX generation.

## Result

TinyStories-1M decode:
- **Before:** 803 tok/s, 0.195× ORT
- **After:** 1526 tok/s, 0.368× ORT
- **Speedup:** 1.9×, ratio nearly doubled

Load during measurement: 16-22 (low for this host).

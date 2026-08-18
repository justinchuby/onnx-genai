---
title: CPU Execution Provider
aliases:
  - CPU EP
tags:
  - execution
  - cpu
  - ep
  - performance
status: maintained
created: 2026-08-17
updated: 2026-08-17
---

# CPU Execution Provider

> [!summary] Question answered
> How does the native CPU EP balance portability, correctness, SIMD performance, threading and memory?

The CPU EP is both a portable execution backend and the native runtime's most
accessible correctness baseline. It implements the shared
[[execution/Execution Provider Contract]] and registers kernels by ONNX
domain/op type.

## Backend strategy

```text
ONNX node
  ↓ registry + shape/dtype checks
CPU kernel
  ├── portable Generic/reference path
  ├── built-in x86 SIMD path when available
  └── MLAS-backed paths for selected operations/features
```

The portable path matters even when an optimized path exists:

- it runs without a vendor toolkit;
- it provides a correctness comparison;
- it prevents a missing ISA from becoming a runtime failure;
- it gives tests a simple mechanism to isolate optimized-path bugs.

## Runtime capability, not build-machine identity

Fast paths should be selected from runtime CPU capabilities and tensor
requirements. AVX-512/AVX2/NEON/SVE availability changes speed, not semantics.
Unsupported instructions must degrade to a correct path.

Kernels are shape- and dtype-driven. Model names and fixed hidden dimensions do
not belong in the EP.

## Hot-path architecture

The CPU EP includes:

- blocked/register-tiled GEMM and SIMD backends;
- quantized matmul and MoE kernels;
- attention, normalization, indexing and data-movement kernels;
- EP-specific fusion/optimization passes;
- host parallelism, decode affinity and NUMA-aware support;
- weight-offload placement and host-cache mechanisms.

The session should observe a `Kernel`, not which internal GEMM implementation ran.

## Threading lessons

Thread count is part of the algorithm:

- per-thread scratch multiplied by worker count can become a process-scale memory
  claim;
- nested parallelism can oversubscribe cores;
- decode often benefits from a bounded worker pool rather than all available
  hardware threads;
- NUMA placement can dominate arithmetic improvements for large weights;
- process affinity and runtime thread budgets are different controls.

Any resident per-thread or per-kernel buffer that scales with model weight or
thread count must be planned in actual bytes and be declinable.

## Persistent caches

CPU performance may use:

- transposed weight caches;
- dense/widened weight caches;
- quantized packed-B buffers;
- resident dequantized weights;
- reusable large host allocations;
- accumulator scratch pools.

> [!warning] A cache is a memory policy
> If it outlives one kernel call and scales with weights or threads, it must be
> declared before allocation, charged by actual footprint and have a correct
> fallback when declined.

## Correctness and performance gates

An optimization should preserve:

- output values within the justified tolerance;
- byte-identical token IDs where deterministic generation is expected;
- supported shapes/dtypes/opsets;
- explicit fallback behavior;
- bounded persistent memory;
- portability to machines without the fast ISA.

Kernel-vs-old-kernel speedup is insufficient evidence. Compare production shapes
against the relevant ORT CPU EP or another strong baseline.

## Formal sources

- [`onnx-runtime-ep-cpu`](../../crates/onnx-runtime-ep-cpu/src/lib.rs)
- [`docs/performance/KERNEL_PERF.md`](../../docs/performance/KERNEL_PERF.md)
- [`docs/performance/CPU_MATMUL_ASSIGNMENT.md`](../../docs/performance/CPU_MATMUL_ASSIGNMENT.md)
- [`CPU EP vs ORT benchmark`](../../docs/benchmarks/2026-08-15-cpu-ep-vs-ort-attention-moe.md)
- [`docs/architecture/CROSS_PLATFORM.md`](../../docs/architecture/CROSS_PLATFORM.md)

## Related notes

- [[execution/Execution Provider Contract]]
- [[performance/Performance Engineering Playbook]]
- [[memory/Memory Management for Beginners]]

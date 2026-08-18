---
title: Execution Backends
aliases:
  - ORT and Native Backends
  - Backend Selection
tags:
  - execution
  - onnx-runtime
  - nxrt
  - architecture
status: maintained
created: 2026-08-17
updated: 2026-08-17
---

# Execution Backends

> [!summary] Question answered
> What is the difference between the ORT and native backends, and where do execution providers fit?

## Three concepts that are easy to mix up

| Concept | Meaning |
|---|---|
| Generation engine | Owns prompt/decode/sampling/session semantics |
| Decode backend | Executes the model step through ORT or native nxrt |
| Execution Provider (EP) | Implements/claims graph operators for a device such as CPU or CUDA |

A backend is not the same as a device. Native CUDA means “nxrt backend using the
CUDA EP”; ORT CUDA means “ONNX Runtime backend configured with its CUDA provider.”

## ORT backend

The ORT path uses ONNX Runtime sessions through `onnx-genai-ort`.

Strengths include:

- mature operator and model coverage;
- existing ORT execution-provider ecosystem;
- a strong parity/reference path;
- established graph optimization and runtime behavior.

The generation engine still owns the outer token loop, scheduling, sampling and
session semantics; ORT performs graph execution.

## Native nxrt backend

The native path uses `onnx-runtime-session` and the `onnx-runtime-*` stack:

```text
ONNX/model package
    ↓
loader → IR → shape inference → optimization
    ↓
placement / EP claims / kernel compilation
    ↓
memory plan + executor
    ↓
CPU, CUDA or plugin EP kernels
```

It provides direct control over:

- graph IR and layouts;
- kernel selection and placement;
- activation planning;
- device transfers and fences;
- CUDA graph capture;
- VMM-backed allocations;
- tracing and per-phase profiling.

Native does not automatically mean faster. Coverage, kernel quality, graph shape,
hardware, capture state and memory strategy all matter.

## Execution providers

`onnx-runtime-ep-api` defines the native EP/kernel contract. Concrete providers
include CPU and CUDA. Plugin crates bridge external/dynamic providers.

An EP typically owns or supplies:

- device identity and capabilities;
- operator support and kernel factories;
- device context and streams;
- allocation/copy/commit mechanisms;
- graph-capture behavior;
- synchronization and release ordering.

It should not own generation policy, request priority or global eviction choices.

## Backend/device selection

The CLI supports separate backend and EP choices. Changing model, backend or EP
requires reloading because a session is constructed against a graph execution
strategy and cannot simply move its live state to another provider.

Environment and CLI configuration can select CPU, CUDA, WebGPU, CoreML or plugin
providers where compiled/available. Requested unavailable providers fail clearly
unless explicit fallback is enabled.

## Why keep both paths

1. **Parity:** ORT provides a reference for native correctness investigations.
2. **Coverage:** unsupported native operators can remain available through ORT or
   heterogeneous/plugin paths.
3. **Performance experiments:** nxrt exposes planner, kernel and memory choices.
4. **Ecosystem compatibility:** plugin EP work protects access to existing hardware
   backends.
5. **Incremental migration:** generation features can stay backend-agnostic.

## Comparing performance correctly

> [!warning] Backend labels are not enough
> “Native vs ORT” is meaningful only when model, precision, EP/device, batch,
> context, memory strategy, graph capture, threading and system contention are
> held fixed and reported.

Start with:

- [`docs/benchmarks/README.md`](../../docs/benchmarks/README.md)
- [`docs/execution/NATIVE_CUDA_DECODE.md`](../../docs/execution/NATIVE_CUDA_DECODE.md)
- [`docs/execution/CUDA_EP_STATUS.md`](../../docs/execution/CUDA_EP_STATUS.md)
- [`docs/execution/EP_CONFORMANCE.md`](../../docs/execution/EP_CONFORMANCE.md)

## Related notes

- [[architecture/Crate Architecture]]
- [[architecture/Inference Request Lifecycle]]
- [[memory/Memory Management for Beginners]]

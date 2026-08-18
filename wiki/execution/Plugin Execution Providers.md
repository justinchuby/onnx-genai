---
title: Plugin Execution Providers
aliases:
  - Plugin EPs
  - EP ABI
tags:
  - execution
  - ep
  - plugin
  - abi
  - ffi
status: maintained
created: 2026-08-17
updated: 2026-08-17
---

# Plugin Execution Providers

> [!summary] Question answered
> How can execution providers cross dynamic-library boundaries without confusing the Rust trait, ORT's plugin ABI and nxrt's native ABI?

The repository supports three related surfaces:

| Surface | Purpose |
|---|---|
| Rust `ExecutionProvider` trait | In-process native nxrt EP contract |
| ORT plugin-EP C ABI adapter | Export/load providers compatible with ORT's plugin protocol |
| Native nxrt dynamic ABI | Load an out-of-tree EP directly into nxrt |

They should express compatible capabilities, but they are not the same ABI.

## Why a C ABI is necessary

Rust trait-object layout and `Arc` ownership are not stable dynamic-library
contracts. A plugin boundary needs:

- `#[repr(C)]` value/vtable layouts;
- explicit version and size negotiation;
- clear ownership for every pointer;
- panic containment;
- allocator/module-safe status transport;
- explicit optional callbacks.

The host wraps this ABI in safe Rust types after validation.

## Single source of truth

> [!important] Consumers depend on the ABI crate
> Host and plugin must import the same contract definitions. Re-declaring a
> private copy can compile successfully while symbol names, layouts or ownership
> rules disagree.

The native host depends on `onnx-runtime-ep-nxrt-abi`; test plugins are real
workspace members so normal checks cannot omit them accidentally.

## ORT plugin export

ORT loads well-known exported symbols, including `CreateEpFactories` and
`ReleaseEpFactory`, then interacts through C vtables for factories, devices,
allocators, transfer, claims and compute.

Key rules:

- resolve names from actual headers/runtime behavior, not typedef guesses;
- negotiate the ORT API version;
- fence every `extern "C"` callback against panic;
- keep ORT-owned/callback-frame pointers within their documented lifetime;
- fail closed for unsupported transfer/capability directions;
- maintain Rust-trait/C-ABI claim parity.

## Ownership lessons

1. **One documented owner.** If ORT retains a pointer, releasing it immediately is
   a use-after-free even if the registration call returned.
2. **Callback lifetime is a hard boundary.** Kernel-context pointers cannot be
   stored beyond compute.
3. **No cross-module heap ownership.** Data allocated in one CRT/module should
   not be freed by another. Inline/fixed status values or same-module release
   callbacks avoid this.
4. **Unload is a lifetime event.** A module cannot unload while handles,
   callbacks, kernels, allocations or deferred frees still reference its code.

## Trait/ABI parity

The C ABI may have less information than the native trait, especially for shape
inference. A plugin must not claim more through C than it can safely compile:

```text
C claims = native trait claims ∩ nodes resolvable through the ABI
```

Parity tests should exercise both positive and deliberate decline cases.

## External providers

CPU and CUDA have native in-tree implementations. Other provider names exposed
by product configuration—such as WebGPU, CoreML, QNN or OpenVINO—must not be
described as native in-tree implementations unless corresponding code exists.
They are generally reached through ONNX Runtime or plugin paths and inherit those
host/version/platform constraints.

## Formal sources

- [`NXRT_ABI.md`](../../docs/architecture/NXRT_ABI.md)
- [`EP_PLUGIN_EXPORT_ABI_TRUTH.md`](../../docs/ep-plugin/EP_PLUGIN_EXPORT_ABI_TRUTH.md)
- [`onnx-runtime-ep-plugin`](../../crates/onnx-runtime-ep-plugin/src/lib.rs)
- [`onnx-runtime-ep-nxrt-abi`](../../crates/onnx-runtime-ep-nxrt-abi/src/lib.rs)
- [`onnx-runtime-ep-nxrt-host`](../../crates/onnx-runtime-ep-nxrt-host/src/lib.rs)

## Related notes

- [[execution/Execution Provider Contract]]
- [[contracts/Runtime Contracts]]
- [[api/API Design Principles]]

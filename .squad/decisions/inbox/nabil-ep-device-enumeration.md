# Decision: EP Device Enumeration Contract (ORT 1.27)

**Author:** Nabil  
**Date:** 2026-08-10  
**Status:** Accepted (verified with real ORT 1.27.0)

## Context

ORT 1.27's `RegisterExecutionProviderLibrary` calls `GetSupportedDevices` on our factory,
then dereferences ALL factory vtable function pointers without null-checking. A factory
that returns zero devices OR has any null vtable entry causes a SIGSEGV inside ORT.

## Device Enumeration Contract

### `GetSupportedDevices` semantics

- ORT passes an array of `OrtHardwareDevice*` it discovered plus the max output capacity.
- Our factory filters for `OrtHardwareDeviceType_CPU` using `OrtApi::HardwareDevice_Type()`.
- For each matching device, call `OrtEpApi::CreateEpDevice(factory, hw_device, metadata, options, &out)`.
- Register allocator info with `OrtEpApi::EpDevice_AddAllocatorInfo(ep_device, mem_info)`.
- Write ep_device pointers into `out_devices[]`, respecting `max_out` bounds.
- Set `*out_num` to the count actually written.

### Ownership

- **ORT owns** the returned `OrtEpDevice` pointers — the factory must NOT free them.
- **ORT owns** the input `OrtHardwareDevice*` array — read-only.
- The factory retains ownership of its internal state (`ExportedFactory`).

## Critical ORT 1.27 Bugs Worked Around

1. **All vtable entries must be non-null.** ORT calls every factory function pointer
   without checking for null. Every slot must be `Some(fn_ptr)` even if the
   implementation is a no-op stub.

2. **`CreateAllocator` must not write null.** The header says "Set to nullptr if the
   default CPU allocator is used" but ORT immediately dereferences the output pointer
   (calls `allocator->Info()`) without null-checking. We return ORT's default allocator
   via `GetAllocatorWithDefaultOptions`.

## Vtable Layout (verified via disassembly)

All 20 slots (0x00–0x98) must be populated. Non-applicable functions return
`ok_status()` with null/zero outputs. `IsStreamAware` returns `false`.

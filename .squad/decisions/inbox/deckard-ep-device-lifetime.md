# OrtEpDevice / OrtMemoryInfo / Allocator Ownership Contract

**Date:** 2026-08-10
**Author:** Deckard (Systems Dev)
**Context:** Reviewer-rejection fix for BUG 1 (device descriptor corruption) and BUG 2 (allocator not found)

## Root Cause

**BUG 1 and BUG 2 share a single root cause** with two contributing factors:

### Factor 1: Use-after-free on OrtMemoryInfo (BUG 1)

`EpDevice_AddAllocatorInfo(_In_ OrtEpDevice*, _In_ const OrtMemoryInfo*)` stores the
`OrtMemoryInfo` pointer **inside** the `OrtEpDevice`. The `EpDevice_MemoryInfo` API later
returns this pointer directly. ORT does NOT copy the OrtMemoryInfo — it stores the raw pointer.

The old code called `ReleaseMemoryInfo` immediately after `AddAllocatorInfo`, creating a
dangling pointer. On first use the freed memory still contained valid data (luck). After
repeated register/unregister cycles the freed memory was reused by other allocations,
producing garbage DeviceType/MemoryType values.

**Fix:** Do not release the OrtMemoryInfo after a successful `EpDevice_AddAllocatorInfo`.
ORT releases it when `ReleaseEpDevice` is called (ORT owns the EpDevice per the ABI:
"ORT will take ownership of the values returned").

### Factor 2: Wrong OrtMemoryInfo API (BUG 2)

The old code used `CreateCpuMemoryInfo(OrtAllocatorType, OrtMemType)` — a legacy API that
creates memory info using the **old** enum system (`OrtMemType` = {-2..-1, 0, 1}).

The EP device system (ORT 1.22+) reads `OrtMemoryInfoDeviceType` and `OrtDeviceMemoryType`
from the OrtMemoryInfo. These fields are not populated by the legacy API, producing
uninitialized/incorrect values (DeviceType:64, MemoryType:28 = garbage).

**Fix:** Use `CreateMemoryInfo_V2` with explicit:
- `OrtMemoryInfoDeviceType_CPU` (= 0)
- `OrtDeviceMemoryType_DEFAULT` (= 0)
- `OrtDeviceAllocator` (= 0)
- vendor_id = 0, device_id = 0, alignment = 0

This produces a properly-typed OrtMemoryInfo that the EP device system can read correctly.

## Ownership Contract Summary

| Object | Created by | Owned by | Released by |
|--------|-----------|----------|-------------|
| `OrtEpDevice` | Plugin via `OrtEpApi::CreateEpDevice` | ORT (after GetSupportedDevices returns) | ORT via `OrtEpApi::ReleaseEpDevice` |
| `OrtMemoryInfo` (for device) | Plugin via `CreateMemoryInfo_V2` | ORT (stored inside OrtEpDevice) | ORT (via ReleaseEpDevice internals) |
| `OrtKeyValuePairs` (metadata/options) | Plugin via `CreateKeyValuePairs` | Plugin | Plugin releases after CreateEpDevice (CreateEpDevice copies) |
| `OrtEpFactory` | Plugin via `Box::into_raw` | Plugin | ORT calls `ReleaseEpFactory` → plugin's `release_ep_factory` |
| `OrtAllocator` (from CreateAllocator) | Plugin (returns ORT's default) | ORT | ORT (it's ORT's own default allocator; our ReleaseAllocator is a no-op) |

## Key Invariant

After `EpDevice_AddAllocatorInfo(device, mem_info)` succeeds, the caller must NOT
call `ReleaseMemoryInfo(mem_info)`. The OrtEpDevice now owns that pointer. Only release
on failure (when the pointer was not consumed).

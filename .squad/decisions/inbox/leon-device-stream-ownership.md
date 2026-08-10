# Stream Lifetime Contract — `OrtSyncStreamImpl`

**Author:** Leon  
**Date:** 2026-08-10  
**Context:** M2-1 (MEDIUM) memory leak fix in `device.rs`

## Header-verified stream ownership contract

From `onnxruntime_ep_c_api.h` (lines 204–258):

> `OrtSyncStreamImpl.Release` is called by ORT exactly once when the stream
> instance is no longer needed. The implementation must release all resources
> in its `Release` callback.

There is **no** separate `ReleaseStream` factory callback (unlike allocators which have
`ReleaseAllocator`). The stream vtable `Release` is the sole cleanup point.

## `into_raw`/`from_raw` pairing rule for device paths

| Path | `Box::into_raw` site | `Box::from_raw` site |
|------|----------------------|----------------------|
| Allocator EP | `factory_create_allocator` (factory.rs ~574) | `factory_release_allocator` (factory.rs ~597) |
| Stream EP | `factory_create_sync_stream` (factory.rs ~668) | `stream_release` (device.rs ~232) — **FIXED** |
| Stream struct | `factory_create_sync_stream` (factory.rs ~669) | `stream_release` (device.rs ~232) |
| Allocator struct | `factory_create_allocator` (factory.rs ~576) | `factory_release_allocator` (factory.rs ~595) |

## Rule

Every `Box::into_raw` in device/stream/allocator paths must have a matching
`Box::from_raw` in the ORT-specified release callback. Stream tests must use
heap-allocated (`Box::into_raw`) EPs to mirror the real factory path.

## Double-free ruling

`Release` is called exactly once per created stream (header contract). No other
code path calls `Box::from_raw` on the stream EP pointer. The allocator path has
its own independent EP instance. No double-free risk.

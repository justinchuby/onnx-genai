# Decision: CUDA EP Use-After-Free Fix (B1/B3/S4)

**Author:** Nabil (FFI/Systems)
**Date:** 2026-08-11
**Context:** Gaff rejected PR #762 CUDA commits; Sapper under reviewer lockout.

## Problem

Three blocking defects made the CUDA EP path unsound:

1. **B1:** Raw pointer extracted from `MutexGuard` was dangling after guard
   dropped. Every allocator/stream/transfer callback dereferenced UB.
2. **B3:** `CopyTensors` wrapped both src and dst as device buffers, passing
   host pointers to `cudaMemcpyDeviceToDevice`.
3. **S4:** Factory creation called a panic-bomb constructor to read the EP name,
   making success unreachable even on real hardware.

## Decision

### B1: `Arc` clones replace raw pointers

Each component (`DeviceAllocator`, `DeviceSyncStream`, `DeviceDataTransferFull`)
now stores an `EpRef::Shared(Arc<Mutex<..>>)` — a strong reference that keeps
the EP alive independently of the factory's lifetime. The `with_ep` method
locks the mutex for each operation, replacing `.unwrap()` with
`.map_err()` (S2 fix).

**Why not hold the lock across the callback?** ORT calls back on its own
threads. A mutex held across a blocking CUDA call (e.g. `cudaStreamSynchronize`)
would block all other ORT threads calling allocate/copy. The lock is held only
for the duration of the Rust EP method dispatch.

### B3: Direction classification via `Value_GetMemoryDevice`

`transfer_full_copy_tensors` now:
1. Calls `ep_api.Value_GetMemoryDevice` on each OrtValue
2. Calls `ep_api.MemoryDevice_GetDeviceType` to get CPU vs GPU
3. Classifies direction via `CopyDirection::classify`
4. Dispatches to `copy_from_host` / `copy_to_host` / `copy`

### S4: `create_ep_factories_for_shared_ep` takes name directly

New factory creation path that accepts `ep_name: &str` instead of calling a
constructor. The CUDA plugin extracts the name from the pre-constructed EP
before wrapping it in the Arc.

## Deferred

- **B2 (pointer equality for same-device):** Conservative/fail-closed. May
  cause ORT to fall back for D2D copies on the same GPU. Proper fix needs
  `MemoryDevice_GetDeviceId` which may not exist. Not a soundness bug.

## Alternatives Considered

1. **`Arc::as_ptr` on the inner Box:** Would still require the Box to never be
   swapped inside the Mutex. Fragile — anyone adding a `mem::replace` inside
   the lock would reintroduce UB. Rejected.

2. **`RwLock` instead of `Mutex`:** Most EP operations are read-only (allocate,
   copy). A `RwLock` would allow parallel reads. Deferred: the current lock
   duration is microseconds; contention won't manifest until real hardware
   profiling.

## Verification

- `cargo check -p onnx-runtime-ep-cuda-plugin` ✓
- `cargo check -p onnx-runtime-ep-cuda-plugin --features cuda` ✓
- `cargo clippy --workspace --all-targets -- -D warnings` (targeted crates) ✓
- `cargo test -p onnx-runtime-ep-plugin -p onnx-runtime-ep-cuda-plugin` — all pass
- `cargo fmt --check` ✓
- New regression tests: B1 (allocator outlives Arc), S4 (no panic), B3 (direction matrix)

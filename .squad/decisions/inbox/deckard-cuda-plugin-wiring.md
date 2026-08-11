# CUDA Plugin Wiring — Real EP Behind Feature Gates

**By:** Deckard  
**Date:** 2026-08-11  
**Branch:** `squad/ep-plugin-parity-cuda`

## What Is Wired

1. **Real `CudaExecutionProvider` construction** behind the `cuda` feature gate.
   With `--features cuda`, the plugin constructs a genuine `CudaExecutionProvider::new_default()` (ordinal 0), validates it can be created on the host, and exports it through `create_ep_factories_with_device_support`.

2. **GPU DeviceSupport declared.** The factory advertises:
   - `OrtHardwareDeviceType_GPU`
   - NVIDIA vendor ID `0x10DE`
   - Stream-aware: `true`
   - Host-accessible: `false` (device-only memory)
   - Allocator name: `"Cuda"`

3. **Kernel registry entries** derived from `CUDA_COVERED_OPS` advertising f32/f16/bf16 for all covered ops.

4. **Fail-closed behavior preserved:**
   - Feature off → zero factories + actionable error status naming the missing feature.
   - Feature on but no GPU/driver → `new_default()` returns `Err`, plugin emits zero factories + error status naming the failure.
   - Panic guard: `catch_unwind` at the C ABI boundary; on panic → zero factories + error status.

5. **`fail_status` made `pub`** in `status.rs` so plugin crates can produce proper `OrtStatus` errors without going through `panic_to_fail_status` (which swallows the message when no ORT API is loaded).

## Feature Gate Behavior

| Configuration | Result |
|---|---|
| `cargo check -p onnx-runtime-ep-cuda-plugin` (no feature) | ✅ Compiles, zero factories at runtime |
| `cargo check -p onnx-runtime-ep-cuda-plugin --features cuda` | ✅ Compiles (cudarc dynamic-loading, no toolkit needed) |
| Runtime with `cuda` feature + no GPU | Zero factories, error status returned to ORT |
| Runtime with `cuda` feature + GPU present | Real CUDA EP registered with ORT |

## `prefetch_lazy_weight` Decision

**Left as stub (`Ok(false)`).** Rationale:

- The standing directive says "CUDA admits prefetch only when it fits without eviction or lease growth."
- `CudaWeightResidency` exposes `resident_mapped` which may evict. There is no `try_without_eviction` API.
- Implementing prefetch incorrectly (allowing eviction) would violate the standing directive and could degrade steady-state decode.
- Returning `Ok(false)` is semantically correct: "did not prefetch; weight will be demand-paged."
- Proper implementation requires either a new `try_prefetch` method on the residency or hardware validation of the eviction path.

## What Is Unvalidated (No GPU on This Host)

- **Runtime construction of `CudaExecutionProvider`** — `new_default()` will fail immediately (no driver). The plugin's fail-closed path is tested only in the "error returned cleanly" sense.
- **ORT session execution through the CUDA plugin** — no `Run` call has ever been made through this cdylib on a GPU host.
- **Allocator / data-transfer / sync-stream paths** — device memory operations are unexercised.
- **Kernel registry routing** — ORT's type-constraint matching against the advertised dtypes is untested end-to-end.
- **`page_lazy_weight`** — resident-mapped path exercises real cuMemcpy; never run.
- **`prefetch_lazy_weight`** — remains a no-op stub.

## What Pris's Hardware Runner Must Exercise

On a real GPU host with CUDA toolkit and driver:

1. `CreateEpFactories` with `--features cuda` → must return 1 factory, valid `OrtEpFactory*`.
2. `GetSupportedDevices` → must report GPU hardware type for device 0.
3. `CreateEp` → must succeed; EP must report `device_type() == Cuda`.
4. `CreateAllocator` → must return a working device allocator (`Alloc`/`Free` round-trip).
5. `CreateSyncStreamForDevice` → must return a sync stream (EP is stream-aware).
6. `CreateDataTransfer` → must succeed (if Leon's `transfer.rs` lands).
7. Full session: `CreateSession` + `Run` on a simple MatMul model → correct output.
8. Weight offload: load a model with `ONNX_GENAI_WEIGHT_OFFLOAD=1` → `page_lazy_weight` exercises real H2D copy.

**Env preconditions:** `nvidia-smi` reachable, `libcuda.so` / `libcublasLt.so.13` on `LD_LIBRARY_PATH` or in system paths, at least one CUDA-capable GPU.

## Leon Integration Note

`transfer.rs` does not exist yet in the plugin crate. When Leon's data-transfer adapter lands, the CUDA plugin's `CreateDataTransfer` callback will become functional. Currently `factory_create_data_transfer` returns a stub/null (handled by the factory vtable's existing implementation).

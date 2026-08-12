# Session: EP Plugin Export — 2026-08-10

**Branch:** squad/ep-plugin-export  
**Requested by:** @justinchuby  
**Coordinator session recovery:** yes (lost coordinator, Scribe writing from manifest)

## Key outcome

Upstream ONNX Runtime 1.27.0 now genuinely loads, registers, and executes our Rust CPU execution provider (`onnx-runtime-ep-cpu-plugin`) as a real plugin EP:

```
CreateEnv → RegisterExecutionProviderLibrary → GetEpDevices
→ SessionOptionsAppendExecutionProvider_V2 → CreateSession → Run (correct outputs)
```

**82 adapter unit tests pass, 21 real-ORT conformance tests pass, 0 ignored.**

## Agents (16 spawns across 8 roles)

| Role | Rounds | Key contribution |
|------|--------|-----------------|
| Nabil (ORT Plugin EP Eng) | 3 | Adapter crate impl; hardening; shape inference wiring |
| Deckard (Systems) | 3 | Compute path; 22 shape-inference rules; device lifetime UAF fix |
| Pris (Tester) | 3 | Conformance harness; stress test; 0 ignored |
| Roy (Lead) | 2 | Inventory; docs; PR body |
| Holden (Security) | 2 | FFI audit; final YELLOW ship verdict |
| Leon (Engine/Buffers) | 1 | N1+N2 remediation; validate_dims |
| Isidore (Bindings) | 1 | N3 macro panic guards |
| Leon/Isidore/Deckard | 1 | Final clippy lint gate |

## Push status

Branch has 7 commits. **COULD NOT BE PUSHED** — no GitHub write credentials on this host. Coordinator must push.

## Durable bugs found and fixed

1. **OrtMemoryInfo use-after-free** (`factory.rs`): `ReleaseMemoryInfo` called after `AddAllocatorInfo`; ORT stores raw pointer. Corrupted device descriptor after ≥6 register/unregister cycles.
2. **Legacy `CreateCpuMemoryInfo` API**: leaves `OrtMemoryInfoDeviceType`/`OrtDeviceMemoryType` uninitialized → garbage DeviceType:64 / MemoryType:28.
3. **Shape inference fail-open**: silent `SameAsInput(0)` fallback replaced with `Declined` → error status.
4. **`validate_dims` unwired**: implemented but not called in `read_inputs` until final lint gate.

## Post-merge advisories (LOW, non-blocking)

- `compute_release_state` missing `catch_unwind` (assign Leon)
- `ep_compile_inner` partial-output cleanup on mid-loop failure (assign Deckard)

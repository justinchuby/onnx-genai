# EP Inventory Complete — Roy

**Date:** 2026-08-10T23:30Z
**Branch:** `squad/ep-plugin-parity-cuda` (stacked on `squad/ep-plugin-export`)
**HEAD at validation:** `5fa8cb2a8`
**Author:** Roy (Lead)

---

## 1. Search methodology

```
grep -rn "impl ExecutionProvider" crates/ --include="*.rs"
```

Observed output (all files with matches):

```
crates/onnx-runtime-ep-api/src/abi/mod.rs
crates/onnx-runtime-ep-api/src/epcontext.rs
crates/onnx-runtime-ep-api/src/weight.rs
crates/onnx-runtime-ep-api/tests/mock_ep.rs
crates/onnx-runtime-ep-cpu/src/provider.rs
crates/onnx-runtime-ep-cuda/src/provider.rs
crates/onnx-runtime-session/src/executor/prefetch.rs
crates/onnx-runtime-session/src/executor/tests.rs
crates/onnx-runtime-session/src/hetero.rs
crates/onnx-runtime-session/src/hetero/tests.rs
crates/onnx-runtime-session/src/plugin_provider.rs
crates/onnx-runtime-session/tests/epcontext.rs
crates/onnx-runtime-session/tests/executor.rs
```

Of these, the **production EPs** are exactly two:

| Crate | Type | Location |
|---|---|---|
| `onnx-runtime-ep-cpu` | **Production EP** — `CpuExecutionProvider` | `src/provider.rs:118` |
| `onnx-runtime-ep-cuda` | **Production EP** — `CudaExecutionProvider` | `src/provider.rs:513` |

---

## 2. Non-EP exclusions — confirmed with evidence

| Item | Verdict | Evidence |
|---|---|---|
| `LegacyOrtEp` (`onnx-runtime-ep-api/src/abi/mod.rs:160`) | **Excluded — inbound adapter.** Wraps an incoming ORT plugin `.so` as a Rust `ExecutionProvider`. Wrong direction for outbound export. | Code: `impl ExecutionProvider for LegacyOrtEp`; its constructor takes `CreateEpFactories` from a dlopen'd library. |
| `PluginExecutionProvider` (`onnx-runtime-session/src/plugin_provider.rs:72`) | **Excluded — inbound bridge.** Claims subgraphs for a loaded plugin and delegates unclaimed ops to an embedded CPU EP. Not an in-repo EP. | Code: `impl ExecutionProvider for PluginExecutionProvider` wraps an `Arc<dyn ExecutionProvider>` received from the outside. |
| `onnx-runtime-eager` | **Excluded — orchestrator.** Holds `Vec<Arc<dyn ExecutionProvider>>` and dispatches to them; implements no EP itself. | `grep -n "impl ExecutionProvider" crates/onnx-runtime-eager/` → no matches. |
| `mlas-sys` | **Excluded — BLAS library.** Vendored C++/asm; optional build dependency of `onnx-runtime-ep-cpu` under the `mlas` feature. Not an EP. | `grep -n "impl ExecutionProvider" crates/mlas-sys/` → no matches. |
| Test/mock impls (`MockEp`, `RecordingEp`, `AcceleratorEp`, `MockCompiledEp`, `PlainEp`, `WeightDeliveryEp`, `HostDownloadCountingEp`, `AssignedOracle`) | **Excluded — test fixtures only.** All in `#[cfg(test)]` blocks or `tests/` files. | Confirmed by inspection of file paths. |

---

## 3. Scope verdicts for Metal EP and QNN EP

### Metal EP (`../onnxruntime-mlx`)

**Verdict: OUT OF SCOPE for this workspace. Requires a separate sibling repo.**

Evidence:
```
ls /workspace/dev/
# Output: onnx-genai
```
Only one directory. `../onnxruntime-mlx` does not exist on this host. No Metal/MPS EP code exists in this repo's `crates/` tree. `grep -r "mlx\|metal\|Metal\|mps\|MPS" crates/ --include="*.toml"` returns only pyproject references to Python CLI packages — nothing structural.

`.squad/team.md` describes a Metal/MPS pod (Nabil, Mariette, Coco, Freysa) operating in `../onnxruntime-mlx`. That repo must be cloned separately to bring Metal EP work into scope. To bring it in: `git clone <onnxruntime-mlx-url> /workspace/dev/onnxruntime-mlx`, then follow whatever Cargo workspace integration Nabil's charter describes. **No action required in this PR wave.**

### QNN NPU EP (Luba's domain)

**Verdict: ASPIRATIONAL. No crate exists; no stub; no planned structure found.**

Evidence: `grep -r "qnn\|QNN\|npu\|NPU" crates/ --include="*.rs" --include="*.toml" -l` returns only bench/CLI files where those strings appear inside comments or model-config strings — no EP implementation, no `Cargo.toml` for a QNN EP crate, no `impl ExecutionProvider` for any QNN type.

Luba's charter (`crates/…/luba/charter.md`) defines QNN NPU offload as a domain, but no crate scaffolding has been created. This is a planned/aspirational feature; it is **not in scope for the current wave**. To add it: create `crates/onnx-runtime-ep-qnn/` following the same pattern as `onnx-runtime-ep-cpu/`, implement `ExecutionProvider` against the Qualcomm QNN C SDK, then wire a `crates/onnx-runtime-ep-qnn-plugin/` cdylib following the mechanical checklist below.

---

## 4. Mechanical adaptation checklist — per EP

The shared adapter (`onnx-runtime-ep-plugin`) is now proven. Adding an EP export is a short procedure. The exact steps, and what gates each one, are:

---

### 4.1 `onnx-runtime-ep-cpu` → `onnx-runtime-ep-cpu-plugin` (cdylib)

**Classification: READY (already shipped in Milestone 1)**

The cdylib crate exists and all 21 ORT conformance tests pass. No remaining steps for the core export. Open items are post-merge improvements only:

| Step | Status | Notes |
|---|---|---|
| Create `crates/onnx-runtime-ep-cpu-plugin/` with `crate-type = ["cdylib","lib"]` | ✅ DONE | Crate exists |
| Depend on `onnx-runtime-ep-cpu` + `onnx-runtime-ep-plugin` | ✅ DONE | `Cargo.toml` wired |
| Call `export_ep!(CpuExecutionProvider, "cpu_ep")` macro (or equivalent) in `lib.rs` | ✅ DONE | `CreateEpFactories`/`ReleaseEpFactory` exported |
| `GetSupportedDevices` returns a real `OrtEpDevice` with CPU type | ✅ DONE | `factory.rs` |
| `GetCapability` + `Compile` + `CreateState`/`Compute`/`ReleaseState` wired | ✅ DONE | 82 unit tests + 21 ORT conformance tests pass |
| `catch_unwind` on all `extern "C"` callbacks including `compute_release_state` | ✅ DONE | `compute.rs:1563` — comment confirms "fixes NEW-1" |
| Wire `GetKernelRegistry` with f16/bf16 type constraints | ⚠️ OPEN | `GetKernelRegistry: None` at `ep.rs:48`; no type-constraint metadata registered with ORT. Blocks end-to-end f16/bf16. Owner: Deckard/Nabil |
| `ep_compile_inner` mid-loop partial-output cleanup on failure | ⚠️ OPEN | NEW-2: no cleanup of `out_infos[0..i]` on return-early. ORT contract unspecified; LOW risk, defer until clarified. Owner: Deckard |
| Push and open PR | 🔴 BLOCKED — credentials | No `GH_TOKEN`/SSH key on this host |

---

### 4.2 `onnx-runtime-ep-cuda` → `onnx-runtime-ep-cuda-plugin` (cdylib)

**Classification: BLOCKED — hardware-gated + design work remaining**

The underlying `CudaExecutionProvider` is a complete, production-quality `impl ExecutionProvider`. Exporting it via the plugin ABI requires the following steps, all currently blocked:

| Step | Status | Gate |
|---|---|---|
| Create `crates/onnx-runtime-ep-cuda-plugin/` with `crate-type = ["cdylib","lib"]` | 🔴 NOT STARTED | Crate does not exist |
| Depend on `onnx-runtime-ep-cuda` + `onnx-runtime-ep-plugin` | 🔴 NOT STARTED | Design work |
| Wire `export_ep!(CudaExecutionProvider, "cuda_ep")` macro in `lib.rs` | 🔴 NOT STARTED | |
| `GetSupportedDevices`: enumerate CUDA devices via `cuDeviceGetCount`; create `OrtEpDevice` with `OrtHardwareDeviceType_GPU` | 🔴 NOT STARTED | Requires CUDA toolkit to build/test |
| Allocator surface: wire `CreateAllocator` / `ReleaseAllocator` so ORT can request device memory through the plugin ABI | 🔴 NOT STARTED | Genuine design work — CUDA device pointers are opaque, must satisfy ORT's allocator contract (pointer must be free-able by ORT's internal mechanisms or through `ReleaseAllocator`) |
| Stream/sync surface: wire `CreateSyncStreamForDevice` / `ReleaseSyncStream` | 🔴 NOT STARTED | Genuine design work — ORT must be able to sequence operations on the CUDA stream |
| Device-pointer ABI crossing: ensure tensors allocated on device are handed to ORT in the correct form expected by the plugin-EP compute path | 🔴 NOT STARTED | Most technically uncertain step |
| `catch_unwind` on all `extern "C"` callbacks (including `compute_release_state`) | 🔴 NOT STARTED (will be inherited from shared adapter) | Will come for free once shared adapter is used |
| Wire `GetKernelRegistry` with f16/bf16 type constraints | 🔴 NOT STARTED | Same gap as CPU; CUDA EP supports these dtypes via its kernel registry |
| `prefetch_lazy_weight` stub: `let _ = (self, key, weight, source); Ok(false)` at `provider.rs:564–573` | ⚠️ OPEN STUB | "Phase 2a" per comment; not blocking export but limits functionality |
| CI validation on GPU host | 🔴 HARDWARE GATED | No CUDA toolkit or GPU on this host |

**Summary of what's needed before this can even compile:** CUDA toolkit ≥ 12.6, cuBLAS, cuDNN, and the three design decisions above (allocator ABI, stream ABI, device-pointer crossing). This is not purely hardware-blocked — the design work is independent of hardware availability.

---

### 4.3 Future EPs (Metal, QNN, etc.)

For any new EP following this pattern, the mechanical procedure is:

1. Create `crates/onnx-runtime-ep-{name}/` and implement `ExecutionProvider` trait fully (no stubs in required methods).
2. Create `crates/onnx-runtime-ep-{name}-plugin/` with `crate-type = ["cdylib","lib"]`.
3. Add `onnx-runtime-ep-{name}` + `onnx-runtime-ep-plugin` as dependencies.
4. Call `export_ep!(YourProvider, "name_ep")` in `lib.rs`.
5. Override `GetSupportedDevices` to enumerate actual hardware devices with the correct `OrtHardwareDeviceType`.
6. For device EPs (GPU, NPU): design allocator + stream ABI (see §4.2 above).
7. Wire `GetKernelRegistry` with the EP's supported dtype/op type constraints.
8. Add `catch_unwind` on all `extern "C"` callbacks (shared adapter provides the pattern).
9. Add ORT conformance integration tests under `crates/onnx-runtime-ep-{name}-plugin/tests/`.

Steps 1–5 + 8–9 are proven-and-boring (CPU EP demonstrated them). Steps 6–7 are the only non-mechanical parts for a new accelerator EP.

---

## 5. Milestone 2 in-flight status (as of HEAD `5fa8cb2a8`)

The `squad/ep-plugin-parity-cuda` branch is currently at the same commit as `squad/ep-plugin-export` — no Milestone 2 commits have landed yet. Status of four in-flight engineers:

| Engineer | Task | Status |
|---|---|---|
| Leon | NEW-1 `catch_unwind` on `compute_release_state` | ✅ **DONE** — already in code at `compute.rs:1563`; comment confirms "This fixes NEW-1 from the EP plugin security audit." Landed on `squad/ep-plugin-export` before hand-off. |
| Pris | Trait ↔ C-ABI capability/numeric/error parity tests | 🔴 **NOT YET LANDED** — no such tests found in `crates/onnx-runtime-ep-api/tests/` beyond existing mock_ep.rs graph-view tests. |
| Deckard | NEW-2 partial-cleanup + `GetKernelRegistry` + f16/bf16 type constraints | 🔴 **NOT YET LANDED** — `GetKernelRegistry: None` at `ep.rs:48`; no cleanup logic in `ep_compile_inner`. |
| Nabil | Device/allocator/stream adapter surfaces + `onnx-runtime-ep-cuda-plugin` shim + mock-device fail-closed tests | 🔴 **NOT YET LANDED** — crate `onnx-runtime-ep-cuda-plugin` does not exist. |

---

## 6. Classification summary

| EP | Classification | Bottleneck |
|---|---|---|
| `onnx-runtime-ep-cpu` → plugin | **READY** (Milestone 1 shipped) | Only open: GetKernelRegistry (f16/bf16), NEW-2 cleanup |
| `onnx-runtime-ep-cuda` → plugin | **BLOCKED** | Hardware (CUDA toolkit + GPU) AND design (allocator/stream ABI) |
| Metal EP (`onnxruntime-mlx`) | **OUT OF SCOPE** | Sibling repo not present; requires clone + separate wave |
| QNN NPU EP | **ASPIRATIONAL** | No crate; no design; requires Luba + Qualcomm QNN SDK |

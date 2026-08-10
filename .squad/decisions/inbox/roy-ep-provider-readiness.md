# EP Provider Readiness — Verified Verdict

**Author:** Roy (Lead)
**Date:** 2026-08-10T21:15:32Z
**Branch:** squad/ep-plugin-export
**Requestor:** @justinchuby

---

## 1. Provider Inventory (re-verified)

Search command: `grep -rn "impl ExecutionProvider" crates/`

### Production EPs (outbound export candidates)

| EP | Crate | File:line | Status | Evidence |
|----|-------|-----------|--------|----------|
| `CpuExecutionProvider` | `onnx-runtime-ep-cpu` | `src/provider.rs:118` | **NEAR** | No `todo!()` / `unimplemented!()`. All trait methods real. 166 op registrations. `as_ort_plugin()` returns `None` — blocked only on adapter wiring. |
| `CudaExecutionProvider` | `onnx-runtime-ep-cuda` | `src/provider.rs:513` | **BLOCKED** | `prefetch_lazy_weight` stub (`src/provider.rs:564–573`: `let _ = (self, key, weight, source); Ok(false)`). `as_ort_plugin()` returns `None`. Requires CUDA toolkit ≥ 12.6 to link — see §3 below. |

### Inbound adapters (not candidates)

| Name | Crate | Why excluded |
|------|-------|-------------|
| `LegacyOrtEp` | `onnx-runtime-ep-api/src/abi/mod.rs:160` | Inbound adapter — loads a foreign `.so`, wraps it as Rust EP. Wrong direction. |
| `PluginExecutionProvider` | `onnx-runtime-session/src/plugin_provider.rs:72` | Inbound bridge — claims subgraphs for a loaded plugin EP. Not an in-repo EP. |

### Non-EPs excluded

| Crate | Reason |
|-------|--------|
| `onnx-runtime-eager` | Orchestrator holding `Vec<Arc<dyn ExecutionProvider>>`. Not an EP. |
| `mlas-sys` | BLAS kernel library (C++/asm). Build dep of CPU EP's `mlas` feature. |
| Test / mock impls (7 found) | Appear only in `#[cfg(test)]` / `tests/` modules. Not production. |

---

## 2. The `../onnxruntime-mlx` Contradiction — Resolved

**Evidence:** `ls /workspace/dev/` returns only `onnx-genai`. No `onnxruntime-mlx` directory exists on disk.

**Resolution:** `.squad/team.md` lists `onnxruntime-mlx` as a *sibling repo* (a separate repository living at `../onnxruntime-mlx` relative to this repo). That repo does not exist in this workspace and is not checked out here. The Metal/MLX/MPS execution providers it would contain are **out of scope** for this repo's workspace. Deckard's inventory statement ("Does not exist in this workspace") is **correct**. The `.squad/team.md` listing describes an external dependency, not a local crate.

**Consequence:** There is no Metal/MLX EP to inventory in this codebase. No action needed in the inventory.

---

## 3. CUDA Blocker — Definitive Verdict

### Evidence: Compilation test

```
cargo check -p onnx-runtime-ep-cuda 2>&1 | tail -10
```
**Result:** `Finished 'dev' profile ... in 10.02s` — **clean compile** on this host.

### Why it compiles without a CUDA toolkit

`crates/onnx-runtime-ep-cuda/Cargo.toml` uses `cudarc` with `features = ["dynamic-loading"]`. This tells `cudarc` to use `libloading` (dlopen at runtime) rather than linking against `libcuda.so` at build time. Therefore:

- **Build-time:** No CUDA toolkit required. Clean `cargo check` on any host.
- **Runtime:** Will panic or return an error when `libcuda.so` / `libcudart.so` / `libcublasLt.so` / `libcudnn.so` are absent (i.e., on this host).

### CUDA plugin export status

`cargo check -p onnx-runtime-ep-cpu-plugin` **FAILS** with:
```
error[E0063]: missing fields `CreateProfiler`, `GetAvailableResource`,
`GetDefaultMemoryDevice` and 8 other fields in initializer of `OrtEp`
  --> crates/onnx-runtime-ep-plugin/src/ep.rs:34:21
```

This is in the **shared adapter crate** (`onnx-runtime-ep-plugin`), not in the CUDA EP itself. The `OrtEp` struct in ORT 1.27.0 bindings has 24 fields; the adapter's `ep.rs:34` struct initializer is missing 11 of them (the ORT 1.23–1.27 additions: `CreateProfiler`, `GetAvailableResource`, `GetDefaultMemoryDevice`, `ReleaseCapturedGraph`, `OnSessionInitializationEnd`, `Sync`, `IsConcurrentRunSupported`, `GetKernelRegistry`, `IsGraphCaptured`, `ReplayGraph`, `IsGraphCaptureEnabled`). These are all `Option<fn>` fields and can be set to `None`; this is a mechanical fix.

### Verdict: **Dual block — adapter ABI gap + runtime hardware requirement**

| Blocker | Scope | Nature |
|---------|-------|--------|
| `onnx-runtime-ep-plugin` fails to compile (`OrtEp` missing 11 fields) | Both CPU and CUDA plugin | **Design/adapter gap** — fixable without hardware. Nabil owns this. |
| `CudaExecutionProvider` requires `libcuda.so` at runtime | CUDA EP only | **Hardware-blocked** — runtime only, not build-time. This host has no NVIDIA GPU. `nvidia-smi`: absent. `nvcc`: absent. `/usr/local/cuda*`: absent. `/dev/nvidia*`: absent. |
| `prefetch_lazy_weight` stub (`provider.rs:564–573`) | CUDA EP only | **Design/implementation gap** — acknowledged stub. Disables double-buffer weight prefetch even when `ONNX_GENAI_WEIGHT_OFFLOAD=1`. Does not block basic CUDA inference but blocks full weight-paging capability. |
| `as_ort_plugin()` returns `None` in both EPs | Both CPU and CUDA | **Design gap** — superseded by the adapter crate design; irrelevant once adapter is wired. |
| Allocator/stream/context sharing for CUDA plugin | CUDA plugin only | **Genuine design gap** — CUDA EP owns its own CUDA context + streams + cuBLASLt/cuDNN handles. ORT's plugin model may pass its own stream via `OrtKernelContext`. These must be reconciled (adopt ORT's context, rebind handles, sync streams). This requires both design work AND CUDA hardware to test. |

**For Justin:** The CPU EP plugin path is blocked by a compile error in the shared adapter (11 missing `OrtEp` optional fields), which is a **mechanical adapter fix** with no hardware dependency. Once that fix lands, `onnx-runtime-ep-cpu-plugin` should compile and the full `CreateEpFactories → GetCapability → Compile → Compute` path can be exercised. The CUDA EP plugin path has a **dual block**: the same mechanical adapter fix, PLUS a design decision on CUDA context/stream sharing that requires CUDA hardware to validate.

---

## 4. ORT Compatibility Boundary

- **ORT version built against:** 1.27.0 (`ORT_API_VERSION = 27`)
- **ORT version available at test time:** 1.28.0 (wheel), backward-compatible (`GetApi(27)` non-null)
- **Required export symbols:** `CreateEpFactories`, `ReleaseEpFactory` (both confirmed in ORT 1.27 headers)
- **Version check policy:** Fail-closed. `CreateEpFactories` must call `OrtApiBase::GetApi(ORT_API_VERSION)`; if null, return error. See `EP_PLUGIN_EXPORT_ABI_TRUTH.md` §4.
- **`OrtEp` field count:** 24 fields (ORT 1.22–1.27). All fields except `GetName`, `GetCapability`, `Compile`, `ReleaseNodeComputeInfos` may be `None` for v1 CPU EP.
- **`OrtEpFactory` field count:** 19 fields. Required: `ort_version_supported`, `GetName`, `GetSupportedDevices`, `CreateEp`, `ReleaseEp`. Others `None` for v1.

---

## 5. Stale Doc Findings

Corrected in companion doc updates (see `docs/EP_PLUGIN_EXPORT.md` revision):
- Removed claim "v1 implemented (CPU EP) — Compute path live for elementwise ops." The adapter crate **does not compile** as of this session.
- `EP_PLUGIN_EXPORT_ABI_TRUTH.md` updated: added full accurate field lists for `OrtEp` (24 fields) and `OrtEpFactory` (19 fields); confirmed `ValidateCompiledModelCompatibilityInfo` is on `OrtEpFactory`, `GetCompiledModelCompatibilityInfo` is on `OrtEp` — distinction is correct in bindings.
- `EP_PLUGIN_EXPORT_ABI_GAPS.md`: noted the `OrtEp` missing-fields compile error as the immediate blocker.

---

## Disposition

This record is a truth-layer deliverable by Roy. Implementation fixes (the 11 missing `OrtEp` fields) belong to Nabil. The CUDA allocator/stream design is deferred until the CPU plugin is end-to-end verified.

# Outbound EP Plugin Export — Test Plan

**Author:** Pris (Tester)  
**Date:** 2026-08-10  
**Status:** Recon/Plan pass — environment results embedded. No source code modified.

---

## 1. Context

nxrt currently only implements the *inbound* direction: it loads third-party plugin
EPs (dylibs that export `CreateEpFactories`) and hosts them through its own ABI
bridge (`crates/onnx-runtime-ep-api/src/abi/`).

The *outbound* direction — one of our EPs compiled as a `cdylib` that upstream ORT
can discover and load — does not exist yet. This document defines the test strategy
for proving it actually works once Nabil's adapter is built.

---

## 2. Environment Feasibility — Empirical Results

### 2.1 ORT library availability

```
pip install onnxruntime  →  ORT 1.28.0 wheel installed successfully
```

Library path on this machine:
```
/workspace/dev/onnx-genai/.ort-probe/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.28.0
```

**Public dynamic-symbol table** (`nm -D`):

```
OrtGetApiBase@@VERS_1.28.0
OrtSessionOptionsAppendExecutionProvider_CPU@@VERS_1.28.0
```

That is the *entire* exported surface — two symbols. ORT 1.28 distributes a symbol-stripped
wheel with all internals hidden. There is no `RegisterCustomEpLibrary`, no
`RegisterExecutionProviderLibrary`, and no plugin-EP loader in the public ABI.

### 2.2 ORT API version compatibility

Our `ort-sys` crate is built against **ORT API 27** (`ORT_API_VERSION = 27`). ORT 1.28.0 is
backward-compatible: `GetApi(27)` returns a non-null pointer (verified empirically). The
`try_load_candidate` check in `ort-sys/src/lib.rs` would therefore **accept** this library.

### 2.3 Plugin EP V2 ABI (ORT 1.28, `onnxruntime_ep_c_api.h`)

ORT 1.28 defines the outbound EP contract in
`include/onnxruntime/core/session/onnxruntime_ep_c_api.h`. The required dylib exports are:

| Symbol | Signature | Required |
|--------|-----------|----------|
| `CreateEpFactories` | `OrtStatus* (registration_name, OrtApiBase*, OrtLogger*, OrtEpFactory**, max, *num)` | **Yes** |
| `ReleaseEpFactory` | `OrtStatus* (OrtEpFactory*)` | **Yes** |

ORT then calls methods on the returned `OrtEpFactory` vtable:
`GetName`, `GetVendor`, `GetVendorId`, `GetVersion`, `GetSupportedDevices`, `CreateEp`,
`ReleaseEp`, `CreateAllocator`, `ReleaseAllocator`, `CreateSyncStreamForDevice`,
`ValidateCompiledModelCompatibilityInfo`.

### 2.4 Can upstream ORT load our dylib? — VERDICT: YES

**CORRECTION (2026-08-10):** The earlier conclusion that ORT has no public API was wrong.
The entire ORT C API is a vtable (`OrtApi` struct, ~250 function pointers) reached through
`OrtGetApiBase()->GetApi(version)`. `nm -D` only shows 2 exported dynamic symbols because
the rest is the vtable. The registration path exists since ORT 1.22:

- `OrtApi::RegisterExecutionProviderLibrary(env, name, path)` — ORT dlopen's the plugin
- `OrtApi::GetEpDevices(env, &devices, &count)` — enumerate registered EP devices
- `OrtApi::SessionOptionsAppendExecutionProvider_V2(opts, env, devices, ...)` — attach
- `CreateSession` → `Run` — inference
- `OrtApi::UnregisterExecutionProviderLibrary(env, name)` — cleanup

**L3 (real ORT loading our dylib end-to-end) IS achievable with ORT 1.27.0.**

---

## 3. Existing Test Surface

| Location | What it tests | Reusable for outbound? |
|----------|---------------|------------------------|
| `crates/onnx-runtime-ep-api/tests/mock_ep.rs` | Full `ExecutionProvider` + `OpRegistry` + `KernelFactory` lifecycle (safe Rust, no FFI) | Yes — mock EP design mirrors what our outbound adapter will look like internally |
| `crates/onnx-runtime-ep-api/src/abi/runtime.rs` | Inbound `dlopen` + `CreateEpFactories` consumer path, `OrtEpFactory` vtable dispatch | Yes — the factory/vtable types are the *same* ABI our outbound side must implement |
| `crates/onnx-runtime-ep-api/src/abi/host.rs` | `OrtKernelContext`, `OrtValue`, `OrtStatus` host-side stubs | Yes — these types are what L2 will call back through |
| `crates/onnx-runtime-ep-cpu/tests/kernel_numeric_regression.rs` | CPU kernel correctness per operator | Yes — feeds correctness fixtures into any L2/L3 run |
| `conformance/run_onnx_tests.py` + `conformance_runner` example | Full conformance harness: load model → run → compare | L2/L3 can reuse the `.nxrt` tensor format and tiny model fixtures |

---

## 4. Proposed Test Ladder

### L1 — Symbol export check (always runnable, zero dependencies)

**What:** Build `onnx-runtime-ep-cpu` with a new `cdylib` target (gated behind a feature
flag, e.g. `plugin-export`) and assert that the produced `.so`/`.dylib` exports the
required symbols.

**How:**

```bash
cargo build -p onnx-runtime-ep-cpu --features plugin-export
nm -D target/debug/libonnx_runtime_ep_cpu.so \
  | grep -E "CreateEpApiFactories|ReleaseEpApiFactory"
# Must print both (at minimum CreateEpApiFactories) as type "T" (defined text symbols).
```

This can also be a CI shell step or a `#[test]` that `std::process::Command`s
`nm -D` and asserts presence. No ORT, no hardware required.

**Gate:** Merge-blocking in CI.

---

### L2 — In-process `dlopen` ABI driver (always runnable, no ORT)

**What:** A Rust integration test that:
1. Locates the `plugin-export` cdylib built in the same `cargo test` invocation.
2. `dlopen`s it with `libloading::Library::new(path)`.
3. Resolves `CreateEpApiFactories` and calls it with a live `OrtApiBase*` (obtained from
   our own `ort-sys::OrtGetApiBase()`).
4. Iterates the returned `OrtEpFactory*` array; for each factory:
   - Calls `GetName`, `GetVendor`, `GetVendorId`, `GetVersion` — asserts non-null/non-empty.
   - Calls `GetSupportedDevices` with a stub hardware device list — asserts it returns at
     least one `OrtEpDevice`.
   - Calls `CreateEp` — asserts it succeeds and returns a non-null `OrtEp*`.
   - Calls `ReleaseEp`, then `ReleaseEpApiFactory` if present.
5. Asserts no status errors throughout; all `OrtStatus*` returns are checked via
   `check_status` (the existing helper in `crates/onnx-runtime-ep-api/src/abi/host.rs`).

**Why this is the strongest always-runnable gate:** It exercises the full C ABI boundary —
vtable dispatch, pointer ownership, status propagation — without needing upstream ORT to act
as a loader. It proves the ABI is well-formed and won't panic/segfault when called from C.

**Crate:** New test file at `crates/onnx-runtime-ep-cpu/tests/plugin_export_abi.rs` (or a
dedicated `onnx-runtime-ep-cpu-plugin` crate if the build target separation demands it).

**Gate:** Merge-blocking in CI on Linux x86_64 (same runner as EP conformance).

---

### L3 — Real upstream ORT loads our dylib (environment-gated, NOW ACHIEVABLE)

**What:** Upstream ORT creates an `OrtEnv`, discovers our registered EP, appends it to
session options, and runs a tiny ONNX model to completion, comparing outputs to CPU
reference values.

**Previous blocker (RESOLVED):** Earlier recon incorrectly concluded ORT has no public API
to register plugin EPs at runtime. This was wrong — `nm -D` only shows exported symbols,
but the entire ORT C API is a vtable reached through `OrtGetApiBase()->GetApi(version)`.
The `OrtApi` struct includes `RegisterExecutionProviderLibrary` (since ORT 1.22).

**Registration path (confirmed in ORT 1.27.0 headers + bindings):**
1. `RegisterExecutionProviderLibrary(env, "cpu_ep", "/path/to/libonnx_runtime_ep_cpu_plugin.so")`
2. `GetEpDevices(env, &devices, &count)` — enumerate
3. `SessionOptionsAppendExecutionProvider_V2(opts, env, devices, ...)` — attach
4. `CreateSession` → `Run` — inference
5. `UnregisterExecutionProviderLibrary` — cleanup

**Required export symbols:** `CreateEpFactories` and `ReleaseEpFactory` (NOT
`CreateEpApiFactories` — that is the typedef name `CreateEpApiFactoriesFn`, not the dlsym
symbol).

**Test file:** `crates/onnx-runtime-ep-cpu-plugin/tests/plugin_ort_e2e.rs`

**Fixture:** `crates/onnx-runtime-ep-cpu-plugin/tests/fixtures/add_1x4/model.onnx` (float32
Add, opset 17, shape [1,4]).

**Environment gating:** Skips cleanly with explanatory message if ORT library not found.
Set `NXRT_ORT_LIB_DIR` to override; auto-discovers from ort-sys build output.

**Current status (2026-08-10):**
- ✅ Pre-flight: `CreateEpFactories` succeeds, factory pointer returned, vtable populated
  for `GetName`, `GetSupportedDevices`, `CreateEp`, `ReleaseEp`.
- ❌ **BLOCKED by missing vtable entries:** `GetVendor`, `GetVendorId`, `GetVersion` are
  `None` in the factory vtable. ORT calls `GetVendor` (offset 16) during
  `RegisterExecutionProviderLibrary` and segfaults on the null function pointer.
- Stages NOT YET REACHED: registration, device enumeration, session creation, Run.

**Fix required:** In `crates/onnx-runtime-ep-plugin/src/factory.rs`, add implementations
for `GetVendor`, `GetVendorId`, and `GetVersion` in the `OrtEpFactory` vtable init.

**Gate:** Not merge-blocking until the vtable is complete. Run in CI with `NXRT_ORT_LIB_DIR`.

**Gate:** Not merge-blocking. Run only in an explicitly configured environment.

---

## 5. Fixtures

### Existing fixtures that suffice

| Path | Ops | Suitable for |
|------|-----|--------------|
| `crates/onnx-model-package/tests/fixtures/valid-package/cpu-fp32/model.onnx` | Unknown (assumed basic) | L2 smoke |
| `tests/fixtures/tiny-llm-scatter/model.onnx` | Scatter-based | L2/L3 |
| `crates/onnx-runtime-ep-cpu/tests/fixtures/qmoe_weight_offload/model.onnx` | QMoE | L2/L3 |

For L2's ABI driver test, **no model is needed** — the test drives the factory vtable
directly without feeding a graph.

For L3 (if ever achievable), we need a minimal model with ops in `PHASE1_OPS`
(Add/Relu/Gemm, etc. — all passing in the existing conformance baseline). The
conformance harness already generates these; the same generators and fixtures can be reused.

### If a new fixture is needed

The conformance harness (`conformance/run_onnx_tests.py` + the `cbourjau/onnx-tests`
generators) already produces tiny deterministic ONNX models for every `PHASE1_OPS`
operator. A new fixture can be generated with:

```python
import onnx
from onnx import helper, TensorProto
# 1-Add-node model: float32[1,4] + float32[1,4] → float32[1,4]
X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 4])
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 4])
Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [1, 4])
add = helper.make_node("Add", ["X", "Y"], ["Z"])
graph = helper.make_graph([add], "add_graph", [X, Y], [Z])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
onnx.save(model, "tests/fixtures/add_1x4/model.onnx")
```

This is a one-time step; commit the generated file at
`crates/onnx-runtime-ep-cpu/tests/fixtures/add_1x4/model.onnx`.

---

## 6. CI Integration

| Level | File | Job | Blocking |
|-------|------|-----|----------|
| L1 | `scripts/check-ep-plugin-symbols.sh` (new) | `EP plugin export (Linux x86_64)` | Yes |
| L2 | `crates/onnx-runtime-ep-cpu/tests/plugin_export_abi.rs` (new) | Same job | Yes |
| L3 | `crates/onnx-runtime-ep-cpu/tests/plugin_export_ort_e2e.rs` (stub, `#[ignore]`) | Separate `EP plugin ORT e2e` job | No (env-gated) |

---

## 7. Summary — What Is and Is Not Achievable Here

| Level | Description | Achievable? | Blocker |
|-------|-------------|-------------|---------|
| L1 | Symbol export assertion | ✅ Yes, passing | None |
| L2 | In-process `dlopen` + vtable driver | ✅ Yes, passing | None |
| L3 | Real ORT loads our dylib | ✅ Yes, achievable | Factory vtable incomplete: `GetVendor`, `GetVendorId`, `GetVersion` are None → ORT segfaults |

**L2 is the strongest gate we can run in CI without external infrastructure.** It proves
ABI correctness completely without upstream ORT acting as a loader.

---

## 8. Open Questions for Nabil

1. Will the outbound adapter live in `onnx-runtime-ep-cpu` directly (new `cdylib` target
   gated by feature `plugin-export`) or in a new `onnx-runtime-ep-cpu-plugin` crate?
   The test paths above assume the former; a separate crate is cleaner if the export shim
   pulls in additional dependencies.
2. Should L2's vtable driver test be co-located with `plugin_export_abi.rs` or in a
   dedicated integration crate? (Affects `cargo test -p` targeting in CI.)
3. Is there a known ORT nightly or custom build in our infrastructure that supports plugin
   EP path loading? If so, L3 can be unblocked without waiting for a public API change.

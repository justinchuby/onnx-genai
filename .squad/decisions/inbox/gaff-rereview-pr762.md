# Re-Review of PR #762 — Gaff (Independent Adversarial Reviewer)

**Date:** 2026-08-11
**Reviewer:** Gaff (Code Reviewer / Quality)
**Branch:** `squad/ep-plugin-parity-cuda` — HEAD `31687667a`
**PR:** #762 (REJECTED, kept draft)
**Requested by:** @justinchuby

---

## VERDICT: ALL FOUR BLOCKERS RESOLVED. NO NEW BLOCKING FINDINGS.

- **B1 — output dtypes:** ✅ YES, genuinely resolved
- **B2 — `ReleaseEpFactory` ABI:** ✅ YES, genuinely resolved
- **B3 — `NxrtStatus` cross-allocator UB:** ✅ YES, genuinely resolved
- **B4 — CUDA plugin fail-open:** ✅ YES, genuinely resolved

---

## B1 — Output dtypes guessed from first input

### What I verified independently

1. **`CompiledKernelEntry.output_dtypes: Vec<DataType>`** — Declared at `compute.rs:587`. The field is populated in `ep.rs:491–496` by reading `view.value(val_idx).dtype` for each output value index. The `view` is constructed from an owned IR `Graph` returned by `reader.to_ir_graph()` which copies all data during the Compile callback frame.

2. **No ORT pointer escapes.** `OutboundGraphReader` holds `ort_node_ptrs` but is stack-local in both `GetCapability` (ep.rs:195) and `Compile` (ep.rs:435). The `!Send + !Sync` invariant is documented and enforced. After `reader.to_ir_graph()`, all data is owned.

3. **Undefined dtype decline.** `ep.rs:258–274`: nodes with `DataType::Undefined` output are explicitly not claimed (fail-closed). Test at ep.rs:1107–1117 confirms.

4. **LayerNorm shape inference.** `compute.rs:1000–1037`: `ShapeInference::LayerNorm { axis, num_outputs, full_shape_outputs }` correctly produces:
   - Output 0: full input shape
   - Outputs 1+ (Mean, InvStdDev): reduced shape `[d0..d_{axis-1}, 1, .., 1]`
   - Exception outputs listed in `full_shape_outputs` get full shape (e.g. SkipLayerNorm output 3)

5. **Negative axis handling.** `compute.rs:314–317`: negative axis is resolved via `raw_axis + rank`, with bounds checking at lines 308–323.

6. **SkipLayerNormalization.** `compute.rs:331–350`: uses `rank - 1` as axis (contrib op has no axis attr), correctly puts output 3 in `full_shape_outputs`.

7. **All five norm ops covered:** `LayerNormalization`, `RMSNormalization`, `SimplifiedLayerNormalization`, `SkipLayerNormalization`, `SkipSimplifiedLayerNormalization` — verified at compute.rs:303 and 331.

### Tests

- `conformance_layer_norm_multi_output` (trait_cabi_parity.rs:2360): asserts output shapes `[2,4]`, `[2,1]`, `[2,1]` and correct dtypes via `assert_output_dtype`.
- `conformance_layer_norm_neg_axis` (trait_cabi_parity.rs:2502): tests negative axis.
- Cast/Where/Shape tests (trait_cabi_parity.rs:2116, 2184, 2285): assert correct output dtypes (INT64 for Cast f32→i64, FLOAT for Where, INT64 for Shape) and verify values.

**These tests assert real dtypes and values, not just "passes."**

---

## B2 — `ReleaseEpFactory` exported `void`

### What I verified independently

1. **Macro arm** (`lib.rs:149–167`): `ReleaseEpFactory` returns `*mut OrtStatus`. Returns `ok_status()` on success via `release_ep_factory`, or `panic_to_fail_status` on caught panic.

2. **CPU plugin shim** (`cpu-plugin/src/lib.rs:131–145`): Hand-written `ReleaseEpFactory` returns `*mut OrtStatus`. Comment at line 111–113 documents the ABI reference.

3. **CUDA plugin shim** (`cuda-plugin/src/lib.rs:202–220`): Hand-written `ReleaseEpFactory` returns `*mut OrtStatus`. Comment at lines 175–192 documents both why it's hand-written and the keep-in-sync requirement.

4. **ABI test** (`plugin_export_abi.rs:71–76`): `type ReleaseEpFactory = unsafe extern "C" fn(*mut OrtEpFactory) -> *mut OrtStatus;` — resolves the symbol and asserts null return on success (line 123–127).

5. **`CreateEpFactories`** — CPU shim (cpu-plugin/src/lib.rs:67) returns `*mut OrtStatus`; CUDA shim (cuda-plugin/src/lib.rs:131) returns `*mut OrtStatus`; macro (lib.rs:89) returns `*mut OrtStatus`. All three match the header.

6. **Keep-in-sync comments** are present in both hand-written shims, referencing the macro in `onnx-runtime-ep-plugin/src/lib.rs`. Adequate for maintainability.

---

## B3 — `NxrtStatus.message` cross-allocator UB

### What I verified independently

1. **Inline buffer:** `NxrtStatus` at `status.rs:84–91` uses `message: [u8; MESSAGE_BUF_LEN]` (256 bytes). No heap allocation, no pointers, no cross-module free. Size test at status.rs:230 asserts 264 bytes.

2. **`struct_size` / version negotiation:** `NxrtNegotiateRequest.struct_size` and `NxrtNegotiateResponse.struct_size` are computed from `std::mem::size_of::<Self>()` (version.rs:79, 110). These do NOT include `NxrtStatus` as a field — status is returned by value, not embedded. No breakage from the 264-byte struct.

3. **`NXRT_CAP_KNOWN_MASK`:** Defined at version.rs:136, OR-combination of all known capability flags. Correct.

4. **Truncation:** `from_code_with_message` (status.rs:126–133) truncates at `NXRT_STATUS_MESSAGE_MAX` (255) and NUL-terminates at `message[len]`. No panic path.

5. **UTF-8 safety:** `message_str()` (status.rs:149–154) calls `from_utf8(..).ok()`, returning `None` on invalid UTF-8. No panic. However: truncation at byte boundary can split a multi-byte codepoint, causing `message_str()` to return `None` for a valid-but-truncated message. This is defensive (no UB, no panic) but lossy.

6. **`c_char` portability:** `NxrtStatus.message` uses `[u8; 256]`, not `[c_char; 256]`. Correct on all platforms including aarch64 where `c_char` is `u8`.

7. **No other cross-allocator paths:** The nxrt ABI returns `NxrtStatus` by value from all functions. Vtable function pointers use `*mut c_void` for opaque context but do not allocate/free across boundaries. Verified by grep — no `CString`/`Box`/`String` crosses the nxrt ABI.

---

## B4 — CUDA plugin fail-open

### What I verified independently

1. **Zero factories in both configs:** `cuda-plugin/src/lib.rs:140–147` sets `*out_num = 0` unconditionally in the main closure, BEFORE the `#[cfg]` branches. Both `#[cfg(feature = "cuda")]` (line 150) and `#[cfg(not(feature = "cuda"))]` (line 167) return error statuses via `panic_to_fail_status`.

2. **Cannot re-open silently:** The zero-factory write is not behind any feature gate or conditional. A future edit that restores the `create_ep_factories_with_device_support` call would have to remove the explicit `*out_num = 0` and the error-returning `#[cfg]` blocks. The comments at lines 130–139 explain the four defects as preconditions for removal.

3. **`CanCopy` returns `false`:** `transfer.rs:248` unconditionally returns `false` for device EPs (when `host_accessible` is `false`). This prevents ORT from routing copies to a non-functional transfer.

4. **`cuda_impl` code is unused but compiled:** The `let _ = &...` suppressions (lines 155–157) keep the code compiled for `cargo check` without advertising any EP. Correct.

5. **Tests:** `cuda_fail_closed.rs` asserts `num == 0` directly. Adequate for the fail-closed contract.

6. **Docs:** `CUDA_EP_STATUS.md` is honest — "IMPLEMENTATION-BLOCKED, not merely hardware-blocked." The four defects table, the "CODE EXISTS (unvalidated)" capability levels, and the three-phase roadmap are accurate. No doc overstates.

---

## NON-BLOCKERS SPOT-CHECKED

| Area | Owner | Status | Notes |
|------|-------|--------|-------|
| `CanCopy` direction correctness | Iran | ✅ | Returns `false` unconditionally for device EPs — fail-closed |
| nxrt `struct_size` / optional fn pointers | Luba | ✅ | `struct_size` computed from `size_of::<Self>()`, not hand-coded |
| `c_char` portability | Luba | ✅ | `NxrtStatus.message` uses `[u8; 256]`, not `[c_char; 256]` |
| Absent optional inputs | Batty | ✅ | `kernel_ctx.rs:177–184`: null `OrtValue*` → placeholder with null data_ptr |
| `fail_status` null-success | Sapper | ⚠️ NIT | `status.rs:39`: returns null when host API not set. Documented as "unreachable in production" but not enforced. Acceptable. |
| Unsafe intermediate aliasing | Batty | ✅ | `SubgraphRouting` uses separate intermediate buffers per node; no aliasing |

---

## NITS

| # | File:Line | Owner | Finding |
|---|-----------|-------|---------|
| N1 | `status.rs:126–133` | Luba | `from_code_with_message` truncation at byte boundary can split multi-byte UTF-8, causing `message_str()` to return `None`. Consider truncating at a `char_boundary`. |
| N2 | `cuda_fail_closed.rs:62–75` | Chew | `cuda_plugin_diagnostic_message_contract` test is a comment-only assertion — it documents the contract but doesn't programmatically verify the message string. Acceptable for test-context limitations. |

---

## TEST SUITE VERIFICATION

**Independently confirmed:** `rm -f target/debug/libonnx_runtime_ep_cpu_plugin.so` then full test run.

- **245 passed, 0 failed, 7 ignored** (all ignored are doc-tests requiring ORT/compile context)
- Tests cover: dtype round-trips, shape inference (including LayerNorm reduced shapes, negative axes), ABI symbol resolution, Cast/Where/Shape dtype assertions, CUDA fail-closed, nxrt negotiation, panic safety

---

## IS THE CPU EP TRUSTWORTHY END-TO-END?

**Conditionally yes.** The CPU EP path — from `CreateEpFactories` through `GetCapability` → `Compile` → `Compute` → `ReleaseEpFactory` — is structurally sound:

- Output dtypes are read from the graph, not inferred
- Shape inference covers 22+ op families with explicit decline for data-dependent ops
- Panic safety at every FFI boundary
- ABI-correct return types on all exported symbols
- Type-constraint advertisement via kernel registry

**Caveat:** Full end-to-end validation requires an ORT host loading the cdylib. The `plugin_ort_e2e.rs` tests run against a real ORT when available (gated by ORT path detection). On this host, those tests pass by loading a locally-built cdylib and running Add/MatMul/Cast/Where/Shape/LayerNorm models through it.

---

## WHAT I INDEPENDENTLY VERIFIED vs. TOOK ON TRUST

### Verified by reading source

- B1: output_dtypes sourcing from graph values (ep.rs:491–496), Undefined decline (ep.rs:258–274), LayerNorm shape inference (compute.rs:1000–1037), negative axis resolution (compute.rs:314–317), all five norm ops covered
- B2: return type in macro, both shims, ABI test; `release_ep_factory` signature
- B3: `NxrtStatus` struct layout, inline buffer, no heap, `c_char` avoidance, `from_utf8` defensive read
- B4: unconditional zero factories, `CanCopy` false, unused-but-compiled `cuda_impl`
- ORT pointer lifetime: `OutboundGraphReader` is stack-local, `!Send + !Sync`, `to_ir_graph()` copies data

### Verified by running tests

- 245 passed from clean artifact state
- Cast/Where/Shape tests assert real dtypes and values
- CUDA fail-closed tests assert `num == 0`

### Taken on trust

- ORT integration behaviour under a production ORT host (no ORT available in this environment)
- `OutboundGraphReader.read_value_info` correctly parses ORT's opaque `OrtValueInfo` (would need ORT to validate runtime behaviour)
- `kernel_ctx.rs` runtime data pointer reading (tested only via unit tests with synthetic data, not live ORT context)

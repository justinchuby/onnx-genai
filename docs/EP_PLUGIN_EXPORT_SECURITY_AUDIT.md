# EP Plugin Export — Security Audit

**Auditor:** Holden (Security Engineer)
**Initial audit date:** 2026-08-10T20:12:35.793+00:00
**Re-audit date (RED):** 2026-08-10T21:30:26Z
**Final re-audit date (SHIP VERDICT):** 2026-08-10T22:42:21Z
**Branch audited:** `squad/ep-plugin-export`
**Commits reviewed:** `526a883c4` (Nabil's partial remediation), `c92838dba` (Deckard's UAF fix), Leon's N1/N2 fix, Isidore's N3 fix
**Scope:** `crates/onnx-runtime-ep-plugin/src/{factory,ep,graph_reader,compute,kernel_ctx,status,lib}.rs` + `crates/onnx-runtime-ep-cpu-plugin/src/lib.rs`

---

## ═══ FINAL SHIP VERDICT — 2026-08-10T22:42:21Z ═══

### 🟡 YELLOW — May ship. Advisory items recorded below; no blockers remain.

All three original ship-blocking findings (N1 CRITICAL, N2 HIGH, N3 MEDIUM) have been independently verified as resolved in the current branch head. The Deckard-authored use-after-free fix in `factory.rs` (commit `c92838dba`) is structurally correct. Two new LOW advisory items are filed for post-merge follow-up.

| Finding | Severity | Fixer | Status |
|---------|----------|-------|--------|
| N1 — `compute_execute` no `catch_unwind` | CRITICAL | Leon | **RESOLVED** — `compute.rs:552` guarded |
| N2 — negative dims wrap to `usize::MAX` | HIGH | Leon | **RESOLVED** — `kernel_ctx.rs:193` calls `validate_dims()` |
| N3 — macro entry points unguarded | MEDIUM | Isidore | **RESOLVED** — `lib.rs` both symbols guarded |
| UAF `factory.rs` `OrtMemoryInfo` | CRITICAL (new) | Deckard | **RESOLVED** — ownership transfer correct |
| NEW-1 — `compute_release_state` no guard | LOW (advisory) | Leon (post-merge) | Advisory |
| NEW-2 — partial info leak in `ep_compile_inner` | LOW (advisory) | Deckard (post-merge) | Advisory |

---

### N1 Verification — RESOLVED

`compute.rs:547–762`:
```rust
unsafe extern "C" fn compute_execute(...) -> *mut ort::OrtStatus {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        ...
    }));
    result.unwrap_or_else(|_| fail_status("Compute: internal panic"))
}
```
`catch_unwind` wraps the entire execution body. `compute_create_state` (line 519) is also guarded. Regression test `compute_execute_catches_panic_returns_error_status` in `compute.rs` tests` at line 2115 confirmed present. All `OrtNodeComputeInfo` callbacks (`CreateState`, `Compute`, `ReleaseState`) checked — see NEW-1 for minor note on `ReleaseState`.

### N2 Verification — RESOLVED

`kernel_ctx.rs:23–58`: `validate_dims()` rejects negative dims by index and value, uses `checked_mul` for element-count and byte-length overflow, accepts zero dims as legal (ONNX-compatible). Called from `read_inputs()` at line 193:
```rust
let (shape, _, _) = validate_dims(&dims, dtype, &format!("input {i}"))?;
```
Eight test cases present: negative, large-negative, element-count overflow, byte-length overflow, zero-dim, scalar, normal. The old unchecked `dims.iter().map(|&d| d as usize)` is confirmed replaced.

### N3 Verification — RESOLVED

`lib.rs`: `CreateEpFactories` is wrapped in `catch_unwind(AssertUnwindSafe(...))` that zero-clears `*out_num` and returns `fail_status(...)` on panic. `ReleaseEpFactory` return type is `void` (correct per ORT ABI — prior version incorrectly returned `*mut OrtStatus`). `panic_to_fail_status` helper is `pub` and documented. Two regression tests: `panicking_constructor_caught_and_zero_factories_returned` and `panic_to_fail_status_never_panics`.

### factory.rs UAF Fix Verification — CORRECT

`factory.rs` (Deckard, commit `c92838dba`):

The fix is in `factory_get_supported_devices`, the `EpDevice_AddAllocatorInfo` call site:

**Before (buggy):** `mem_info` was released immediately after `add_alloc_info`, but ORT stores the raw pointer inside `OrtEpDevice` without copying it — the device then held a dangling pointer.

**After (fixed):**
```rust
let status = unsafe { add_alloc_info(ep_device, mem_info) };
if !status.is_null() {
    // Release mem_info only on failure since it was not consumed.
    if let Some(release) = unsafe { (*api).ReleaseMemoryInfo } {
        unsafe { release(mem_info) };
    }
    return status;
}
// Success: ORT owns mem_info via OrtEpDevice; do not release here.
```

Ownership analysis:
- **Success path:** `add_alloc_info` transfers ownership of `mem_info` to ORT's `OrtEpDevice`. ORT releases it when the device is released. We never call `ReleaseMemoryInfo`. No leak.
- **Failure path:** `add_alloc_info` returns a non-null status (failure); ownership was not transferred. We call `ReleaseMemoryInfo` exactly once. No leak.
- **Double-free impossibility:** The two release paths are mutually exclusive (branched on `status.is_null()`). There is no path that calls `ReleaseMemoryInfo` twice on the same pointer.
- **`CreateMemoryInfo_V2`:** Correct replacement for the deprecated `CreateCpuMemoryInfo`. Supplies `OrtMemoryInfoDeviceType_CPU` and `OrtDeviceMemoryType_DEFAULT`, which the new EP device ABI requires; the old API left those fields uninitialized, producing the garbage `DeviceType:-112 MemoryType:-85` reported in production.

This fix is structurally sound and matches the ownership-transfer contract described in the ORT plugin-EP header.

---

### NEW-1 (LOW Advisory) — `compute_release_state` missing `catch_unwind`

**File:** `compute.rs:1416`

```rust
unsafe extern "C" fn compute_release_state(
    _info: *mut ort::OrtNodeComputeInfo,
    state: *mut c_void,
) {
    if !state.is_null() {
        unsafe { drop(Box::from_raw(state.cast::<ComputeState>())) };
    }
}
```

`ComputeState` is `struct ComputeState { _placeholder: u8 }` — no heap fields, no custom `Drop`. Dropping a `Box<ComputeState>` is a trivial heap deallocation that cannot panic in Rust's standard allocator. The exploit surface is **zero in current code**.

However, every other `extern "C"` callback with a non-trivial body is wrapped in `catch_unwind`. If `ComputeState` is later extended with a field that has a custom `Drop` (e.g., a file handle, a lock guard, a connection pool), the missing guard will silently become a latent UB vector. This should be patched before `ComputeState` grows.

**Recommended fix (one-liner, assign to Leon post-merge):**
```rust
unsafe extern "C" fn compute_release_state(...) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if !state.is_null() {
            unsafe { drop(Box::from_raw(state.cast::<ComputeState>())) };
        }
    }));
}
```

**Severity: LOW. Non-blocking. File follow-up issue.**

---

### NEW-2 (LOW Advisory) — Partial `OrtNodeComputeInfo` leak on `Compile` error

**File:** `ep.rs`, `ep_compile_inner`

If `Compile` is called with `count = N` subgraphs and compilation fails at index `i` (e.g., `get_kernel` returns `Err`), the function returns `fail_status(...)` immediately. `out_infos[0..i]` contain valid `Box<ExportedComputeInfo>` raw pointers that were already written; `out_infos[i..N]` are uninitialised (or null per caller contract).

The ORT header is silent on whether `ReleaseNodeComputeInfos` is called after a non-null `Compile` return. If ORT skips it, `out_infos[0..i]` leak; if ORT calls it, the caller tries to free pointers it did not place and we double-free. Neither is correct.

This is a carry-forward of M2 from the previous audit. Not a memory-safety issue with a known ORT 1.27 behavior, but should be hardened: on the failure path, free the already-written infos before returning.

**Severity: LOW. Non-blocking. Assign to Deckard post-merge.**

---

### Broader Scope Audit (new code since last pass)

**`graph_reader.rs` — node attribute and initializer reading:**

| Concern | Verdict |
|---------|---------|
| `OrtGraph*`/`OrtNode*` cached beyond `Compile` | **SAFE** — `OutboundGraphReader` is `!Send + !Sync` (doc comment at line 28); `ExportedComputeInfo` stores only owned `Box<dyn Kernel>`, never raw ORT pointers. `to_ir_graph()` returns a reference to a stack-local-owned `Graph`; no ORT pointers escape the callback. |
| Bounds/overflow on attribute arrays | **SAFE** — `read_attr_value` uses a two-call pattern (zero-length probe for required size, then allocate exact). Buffer size comes from ORT, not from attacker-controlled model data directly. |
| Overflow on initializer tensor copy | **SAFE** — `read_initializers_int64` restricts copies to 1-D tensors with ≤ 64 elements (`if dims_count > 1 { continue; }` / `if elem_count > 64 { continue; }`). |
| `CStr` conversions on ORT strings | **SAFE** — All use `.to_string_lossy().into_owned()`. Non-UTF-8 bytes replaced with `U+FFFD`; no panic. ORT strings are null-checked before `CStr::from_ptr`. |
| String attribute type mismatches | **SAFE** — `read_attr_value` matches on `attr_type` from ORT and falls through to `None` for unrecognised types. No type confusion. |
| `OrtValue` from initializers released | **SAFE** — `ValueInfo_GetInitializerValue` follows the ORT `Get*` borrow pattern (pointer into graph-owned storage, not a caller allocation). Consistent with `KernelContext_GetInput` which also returns a borrowed `OrtValue*` without needing `ReleaseValue`. |

**`factory.rs` — allocator/memory-info registration:**

See UAF fix verification above. EP name, vendor, and version `CString` fields are owned by `ExportedFactory` for its lifetime. ORT reads them via `GetName`/`GetVendor`/`GetVersion` callbacks while the factory is alive. Lifetimes are correct.

**`ep.rs` — capability filtering:**

When `GetCapability` declines a node via the `ShapeInference::Declined` filter:
- `claims` is filtered to an empty `Vec`.
- The function returns `ok_status()` with zero claimed nodes.
- `ir_graph`, `cache`, and `view` are stack-local and dropped.
- No partially-built state escapes. The reader's `ort_node_ptrs` are also stack-local and freed with the reader.

Fail-closed behavior per `.squad/decisions.md`: declining a node never claims it; no shape is guessed; no dimension is clamped. **Verified.**

---

## Prior Re-audit Record (2026-08-10T21:30:26Z)

Nabil's remediation commit (`526a883c4`) resolved H1/H2/H3 but left `compute_execute` unguarded (reinstating CRITICAL as N1). Leon, Deckard, and Isidore were assigned N1/N2/N3 respectively under Reviewer Rejection Protocol lockout. Details below.

---

## Re-audit Summary (2026-08-10)

Nabil's remediation commit (`526a883c4`) resolves three of the four original findings. One finding (C1, panic safety) is **partially fixed** — most callbacks are now guarded, but `compute_execute` is not. That single gap reinstates the CRITICAL verdict.

| ID | Original finding | Status | Evidence |
|----|-----------------|--------|----------|
| C1 | No `catch_unwind` on extern "C" callbacks | **OPEN (partial)** — `compute_execute` unguarded | `compute.rs:119` |
| H1 | `static mut HOST_ORT_API` data race | **RESOLVED** | `status.rs:10–30`, AtomicPtr + Acquire/Release |
| H2 | `graphs` null-deref in `ep_compile` | **RESOLVED** | `ep.rs:209` null guard |
| H3 | Unsound `unsafe impl Send+Sync` on `OutboundGraphReader` | **RESOLVED** | `graph_reader.rs:28–30`, impl removed + comment |

**New findings introduced in this session:**

| ID | Severity | Description |
|----|----------|-------------|
| N1 | CRITICAL | `compute_execute` has no `catch_unwind` — reinstates C1 |
| N2 | HIGH | Negative/dynamic dims wrapped to `usize::MAX` in `kernel_ctx.rs:154` |
| N3 | MEDIUM | `CreateEpFactories` / `ReleaseEpFactory` (macro-generated `extern "C"`) call into `create_ep_factories` / `release_ep_factory` without a `catch_unwind` |

**Overall verdict: 🔴 RED — ship-blocking.**

---

## Original Findings — Detailed Re-audit

### C1. `catch_unwind` on `extern "C"` callbacks — **OPEN (partial fix)**

**Original finding:** No `catch_unwind` on any extern "C" callback.

**Remediation by Nabil:** Applied to 12 of 13 live callbacks. Verified present on:
`factory_get_name` (factory.rs:181), `factory_get_vendor` (factory.rs:194), `factory_get_version` (factory.rs:214), `factory_get_supported_devices` (factory.rs:232), `factory_create_ep` (factory.rs:254), `factory_release_ep` (factory.rs:281), `ep_get_name` (ep.rs:63), `ep_get_capability` (ep.rs:82), `ep_compile` (ep.rs:192), `ep_release_node_compute_infos` (ep.rs:321), `compute_create_state` (compute.rs:97).

`factory_get_vendor_id` (factory.rs:204) has no `catch_unwind` but the body is the literal `0` — no panic is possible. Sound.

**`compute_execute` (compute.rs:119) has NO `catch_unwind`.** This is the OrtNodeComputeInfo `Compute` callback — ORT calls it at every inference. The body calls:
- `read_inputs(api_ref, kernel_context)` — complex logic including Vec allocations, ORT API calls (which panic internally if malformed), `DataType::from_onnx().ok_or_else(...)` mapped to `?`-via-Err return, but also `vec![0i64; ndim]` which panics on OOM.
- `inputs[input_offset..input_offset + entry.num_inputs]` (compute.rs:154) — slice range panics if `input_offset + num_inputs > inputs.len()`. No bounds guard.
- `entry.kernel.execute(&kernel_inputs, &mut output_views)` (compute.rs:198) — trait dispatch; user-provided kernels can panic.
- `infer_shapes` calls `broadcast_shapes` from `onnx_runtime_ir`; an internal assertion there could panic.

**Any of these panics unwind across `extern "C"` into ORT's process — immediate UB.**

**Status: OPEN — ship-blocking. Assign to Deckard (owns compute.rs).**

---

### H1. `static mut HOST_ORT_API` data race — **RESOLVED**

`status.rs:10–30`: replaced with `static HOST_ORT_API: AtomicPtr<ort::OrtApi>`. Stored with `Ordering::Release`, loaded with `Ordering::Acquire`. Correct; no TOCTOU risk because the pointer is process-lifetime (set once at `CreateEpFactories`, never reset). Verified sound.

---

### H2. `graphs` null-deref in `ep_compile` — **RESOLVED**

`ep.rs:209`: `if graphs.is_null() { return invalid_arg_status("Compile: graphs pointer is null"); }` is present before any indexing. Verified.

---

### H3. Unsound `unsafe impl Send+Sync` on `OutboundGraphReader` — **RESOLVED**

`graph_reader.rs:28–30`: both impls removed. A comment explains the rationale (raw `OrtNode*` valid only within the callback frame, no cross-thread use). Verified. No `ort_node_ptrs` escape `ExportedComputeInfo` — `CompiledKernelEntry` stores only `Box<dyn Kernel>`, not OrtGraph/OrtNode pointers. The ORT header's prohibition on caching `OrtGraph` beyond `Compile` is not violated.

---

## New Findings

### N1. CRITICAL — `compute_execute` missing `catch_unwind` (see C1 above)

See C1 re-audit above. This is the ship blocker.

**Fix:** Wrap the entire body of `compute_execute` in `std::panic::catch_unwind(AssertUnwindSafe(|| { ... }))` returning `fail_status("Compute: internal panic")` on `Err`. `AssertUnwindSafe` is justified here for the same reason as other callbacks: on panic the state is poisoned, the callback frame unwinds, and ORT will release the EP; we are not hiding broken invariants we intend to reuse.

**Owner: Deckard** (owns `compute.rs`/`kernel_ctx.rs`; Nabil is locked out from re-fixing a finding he missed).

---

### N2. HIGH — Negative (dynamic) dims silently wrap to `usize::MAX` in `kernel_ctx.rs:154`

```rust
// kernel_ctx.rs:154
let shape: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
```

ORT's `KernelContext_GetInput` returns runtime tensor shapes. For models with dynamic batch size the dim value is `-1` (symbolic). Casting `-1i64` to `usize` produces `18_446_744_073_709_551_615` (`usize::MAX`) on 64-bit. This value is then:

1. Passed as a shape element to `infer_shapes` → `broadcast_shapes` from `onnx_runtime_ir`. If that function does `shape.iter().product::<usize>()` to compute element count, it overflows to 0 or wraps; if it panics on internal assertion that is instant UB inside unguarded `compute_execute`.
2. Passed to `allocate_output` → cast back to `i64` (`usize::MAX as i64 == -1`) → forwarded to `KernelContext_GetOutput`. ORT would reject this with an error status and we return `fail_status(...)`. Survivable if C1 is fixed, but incorrect.

**Scenario with attacker-controlled ONNX model:** Any model with a symbolic batch dim causes `read_inputs` to produce a shape containing `usize::MAX`. Subsequent arithmetic on that shape can panic inside the unguarded `compute_execute`, corrupting ORT's process.

**Fix (Deckard):** Replace the cast with a checked conversion:
```rust
let shape: Vec<usize> = dims.iter().map(|&d| {
    if d < 0 { return Err(format!("dynamic dim {d} not supported at compute time")); }
    Ok(d as usize)
}).collect::<Result<Vec<_>, _>>()
.map_err(|e| format!("input {i}: {e}"))?;
```
Return `Err(...)` to propagate as a `fail_status` result from `read_inputs`.

---

### N3. MEDIUM — Macro-generated `CreateEpFactories` / `ReleaseEpFactory` lack `catch_unwind`

`lib.rs:64–88` (macro expansion): The two top-level `extern "C"` symbols call directly into `factory::create_ep_factories` and `factory::release_ep_factory` without a `catch_unwind` guard. Panic sources in `create_ep_factories`:

- `constructor()` call (factory.rs:97) — user closure, can panic.
- `ep.name()` — trait method on user EP.
- `Box::new(ExportedFactory { ... })` — OOM panic.

`release_ep_factory` drops `ExportedFactory` which owns a `Box<dyn Fn()>` constructor closure; drop of that closure theoretically can panic.

These are load-time / session-close paths rather than per-inference. Lower risk than N1, but a panic at `CreateEpFactories` time will crash ORT before the session initializes. With `panic=abort` this is moot, but the plugin cannot mandate that for the host process.

**Fix (Nabil):** Add `catch_unwind` wrappers in the macro expansion around the `create_ep_factories` and `release_ep_factory` calls, matching the pattern already used for all other callbacks.

---

## Original Medium / Low Findings (carry-forward)

### M1. `check()` discards ORT error message — **RESOLVED**

`graph_reader.rs` now calls `GetErrorMessage` before `ReleaseStatus` and includes the real message in the `Err` string. Verified at `graph_reader.rs:459–490`.

### M2. `ep_compile` does not clean up partial `CompiledKernelEntry` allocations on mid-loop error — **STILL OPEN (advisory)**

`ep.rs:200–304`: If `get_kernel` or `ExportedComputeInfo::new` fails at subgraph index `i`, previously-written `out_infos[0..i]` entries are not freed before returning. ORT's contract for `Compile` error returns is unspecified in the header excerpts available. If ORT does not call `ReleaseNodeComputeInfos` on error, these leak; if it does, they are double-freed.

This is a hardening item, not currently exploitable without a specific ORT version that leaks on Compile error. Carry forward as MEDIUM.

### L1. `node_id_to_ort_index` returns 0 on miss — **STILL OPEN (advisory)**

`graph_reader.rs:186`: returns `0` for unknown `NodeId`. No change. Still a logic correctness landmine but not a memory safety issue. Carry forward as LOW.

---

## Cross-Check: Contract Compliance

Per `.squad/decisions.md` extension contract: **fail closed on unsupported capabilities.** Verified:
- Version mismatch on `CreateEpFactories` returns an error status, writes 0 factories (factory.rs:76–95). ✅
- `GetSupportedDevices` returns 0 devices for CPU EP (factory.rs:232–243). ✅
- `ep_get_capability_inner` returns `ok_status()` (not silence-on-unsupported-graph) when claiming zero nodes. ✅
- `ShapeInference::for_op` falls through to `SameAsInput(0)` for unknown ops, which fails loudly on mismatch rather than silently producing wrong output. ✅

---

## Verified Sound (unchanged from initial audit)

- `#[repr(C)]` vtable layout for `ExportedFactory`, `ExportedEp`, `ExportedComputeInfo` — first-field-at-offset-0 cast is sound.
- `Box::into_raw` / `Box::from_raw` pairing: factory ↔ release, EP ↔ release, compute info ↔ release, state ↔ release_state. All matched; no type mismatch or double-free.
- `host_api()` AtomicPtr ordering: Acquire load after Release store. Process-lifetime pointer; TOCTOU impossible.
- `OutboundGraphReader` does not cache `OrtGraph*` or `OrtNode*` past the callback frame; `ExportedComputeInfo.entries` contains only owned Rust objects.
- CStr / string safety in `graph_reader.rs`: all ORT string pointers are null-checked before `CStr::from_ptr`; `to_string_lossy` handles non-UTF-8 safely.
- `GetSupportedDevices` `max_out` bound: CPU EP writes 0 devices, so no bounds concern.

---

## Verdict

**🔴 RED — ship-blocking.**

**N1** (`compute_execute` missing `catch_unwind`, compute.rs:119) reinstates CRITICAL C1. `Kernel::execute` is a trait method callable by user-supplied EP implementations; any panic there unwinds across the C ABI into ORT's process. This is unconditional UB and must be fixed before merge.

**N2** (negative dim wrap to `usize::MAX`, kernel_ctx.rs:154) is HIGH and compounds N1 — a dynamic-dim model triggers the unchecked path. Must also be fixed.

**N3** (macro-generated entry points unguarded, lib.rs:64–88) is MEDIUM and should be fixed in the same PR.

**Required fixes:**
1. **Deckard** — wrap `compute_execute` body in `catch_unwind` (N1, CRITICAL).
2. **Deckard** — validate dims ≥ 0 in `read_inputs`, return error for dynamic dims (N2, HIGH).
3. **Nabil** — add `catch_unwind` to macro-generated `CreateEpFactories` / `ReleaseEpFactory` (N3, MEDIUM).

Per Reviewer Rejection Protocol: Nabil is locked out from revising N1/N2 (Deckard's files). Deckard is locked out from revising N3 (Nabil's macro). These are different files; no cross-lock conflict.

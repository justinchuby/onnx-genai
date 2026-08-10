# PR: ORT Plugin EP Export — Rust CPU EP now loads and runs via upstream ORT 1.27.0

**Branch:** `squad/ep-plugin-export`
**Commits:** `526a883`, `f81d98d`, `09635cd`, `c92838d`, `2fb7150`, `bad3682`

---

## Summary

Our Rust CPU execution provider can now be loaded, registered, and executed as a
real ORT plugin EP by upstream ONNX Runtime 1.27.0. The full call sequence —
`RegisterExecutionProviderLibrary` → `GetEpDevices` (finds `cpu_ep`) →
`SessionOptionsAppendExecutionProvider_V2` → `CreateSession` → `Run` —
succeeds with numerically correct outputs.

Before this branch the adapter crates did not compile. After it they compile, pass
82 unit tests, and pass 21 integration tests (6 lib + 15 e2e including a 25-cycle
use-after-free stress regression) — all in parallel, zero ignored, zero failed.
See Validation below for verbatim output.

---

## Why It Matters

nxrt EPs implement the `ExecutionProvider` Rust trait and have been exercisable
inside nxrt, but upstream ORT had no way to load them. Without the plugin ABI, our
EPs are invisible to every ORT-based deployment pipeline, ONNX model zoo tooling,
and partner integration. This PR closes that gap for the CPU provider and lays the
ABI foundation for the CUDA EP (pending hardware and design).

---

## What Changed

### FFI hardening (`crates/onnx-runtime-ep-plugin/src/`)

**Files:** `lib.rs`, `ep.rs`, `factory.rs`, `compute.rs`, `status.rs`, `kernel_ctx.rs`

- `catch_unwind` on **all** `extern "C"` callbacks: `lib.rs` (macro-generated
  `CreateEpFactories`/`ReleaseEpFactory` — N3, fixed by Isidore), `ep.rs`
  (`get_capability`, `ep_compile`, `release_node_compute_infos`), and
  `compute.rs:552` (`compute_execute` — N1, fixed by Leon).
- Replaced `static mut HOST_ORT_API` with `AtomicPtr<OrtApi>` (Acquire/Release
  ordering). Holden finding H1: **resolved**.
- Added null guard for `graphs` pointer in `ep_compile`. Holden finding H2:
  **resolved**.
- Removed unsound `unsafe impl Send + Sync` on `OutboundGraphReader`. Holden
  finding H3: **resolved**.

### Safety fix: `validate_dims` now actually called (`crates/onnx-runtime-ep-plugin/src/kernel_ctx.rs`)

**This is a genuine safety fix, not cleanup.** `validate_dims()` existed but was
never called — `read_inputs` still cast ORT dims with `d as usize`, so a `-1`
dynamic-dim sentinel would wrap to `usize::MAX`. Leon wired `validate_dims` into
`read_inputs` (commit `2fb7150`). Negative or overflowing dims now return an error
status instead of silently producing garbage shapes. Eight tests verify the boundary
conditions (negative dim, overflow, zero-dim scalar, ND batch).

### Real compute path (`crates/onnx-runtime-ep-plugin/src/compute.rs`)

Previously `CreateState`/`Compute`/`ReleaseState` returned `ORT_NOT_IMPLEMENTED`.
Now:
- `CreateState` allocates a `ComputeState` box via `Box::into_raw`.
- `Compute` runs a topological pass over `ExportedComputeInfo::entries`:
  reads inputs, infers output shapes, allocates ORT outputs, calls kernels.
- `ReleaseState` frees the box via `Box::from_raw`.
- Two shape-resolution strategies chosen at Compile time per op:
  `ElementwiseBroadcast` (numpy-style multi-input) and `SameAsInput(idx)`.

### Device enumeration (`crates/onnx-runtime-ep-plugin/src/factory.rs`, `ep.rs`)

`GetSupportedDevices` previously returned 0 devices, which caused ORT to segfault
during registration. Now it creates a real `OrtEpDevice` with:
- `CreateHardwareDevice` (CPU type, vendor_id=0, device_id=0)
- `CreateEpDevice` with the factory pointer
- `CreateMemoryInfo_V2` with `OrtMemoryInfoDeviceType_CPU` /
  `OrtDeviceMemoryType_DEFAULT` / `OrtDeviceAllocator` (not the legacy
  `CreateCpuMemoryInfo`, which does not populate the fields ORT reads)

### OrtMemoryInfo lifetime fix (`crates/onnx-runtime-ep-plugin/src/factory.rs`, commit `c92838d`)

**Root cause of the "DeviceType:-112 garbage" use-after-free bug:**
`EpDevice_AddAllocatorInfo` stores the `OrtMemoryInfo` pointer inside the
`OrtEpDevice`; ORT does not copy it. The old code called `ReleaseMemoryInfo`
immediately after `AddAllocatorInfo`, leaving a dangling pointer. Fixed by keeping
`mem_info` alive for the lifetime of the `OrtEpDevice` — ORT releases it via
`ReleaseEpDevice`. The fix is verified correct by Holden: the success and failure
release paths are mutually exclusive; no leak, no double-free
(see `docs/EP_PLUGIN_EXPORT_SECURITY_AUDIT.md`).

### Shape inference (`crates/onnx-runtime-ep-plugin/src/ep.rs`)

22 shape-inference rules wired via `for_node`, covering elementwise broadcast,
unary identity, Reshape/Flatten/Squeeze/Unsqueeze with attribute-based dims,
Cast, Expand, Transpose, LayerNormalization, Gemm, Concat, Gather, Slice,
MatMul, and others. Fail-closed `Declined` path: if we cannot infer a shape,
we decline the op rather than silently accepting it. This fixed the `NonZero`
over-claiming bug (was accepting ops whose shapes we could not resolve, then
panicking at runtime).

### Conformance suite (`crates/onnx-runtime-ep-cpu-plugin/tests/plugin_ort_e2e.rs`)

Integration tests covering:
- Broadcast: `[1,4]` + `[3,4]` → `[3,4]`, verified values
- Dynamic/symbolic dim: input with dim `-1`, resolved at runtime
- int32: non-float32 dtype through the full pipeline
- Multi-node fused chain: `Add → Mul` in one compiled subgraph
- MatMul 2-D: `[2,3]` × `[3,2]` = `[2,2]`, verified values
- Mixed partitioning: some ops claimed by our EP, others left to ORT's CPU EP
- Multiple `Run` calls on the same session
- Two concurrent sessions on different models (`conformance_two_sessions`) —
  previously `#[ignore]`d due to a test-assertion bug; **now passing** after Pris
  fixed the assertion (`EpDevice_EpName` returns the factory's declared name
  `"cpu_ep"`, not the registration key)
- `stress_register_run_unregister_cycles`: **25 complete** register→Run→unregister
  cycles without corruption — regression guard for the use-after-free fixed in `c92838d`

---

## Validation

All commands run by Roy on `squad/ep-plugin-export` at commit `bad3682`,
2026-08-10T22:56Z. Output is quoted verbatim.

### `cargo clippy -p onnx-runtime-ep-plugin --all-targets -- -D warnings`

```
Checking onnx-runtime-ep-plugin v0.1.0-dev.5 (...)
Finished `dev` profile [unoptimized + debuginfo] target(s) in 1.59s
```

Zero warnings, zero errors.

### `cargo test -p onnx-runtime-ep-plugin --lib`

```
test result: ok. 82 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

### `cargo test -p onnx-runtime-ep-cpu-plugin`

```
test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.02s
test result: ok. 15 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.73s
```

21 tests total (6 lib + 15 integration), zero failures, zero ignored. Full parallel
run, no `--test-threads=1` workaround needed.

### `cargo check --workspace`

```
Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.26s
```

Workspace compiles cleanly. (Note: `crates/onnx-runtime-cpuinfo/vendor/cpuinfo`
was a pre-existing uninitialized git submodule on this host — unrelated to this
work; initializing it was a one-time host setup step.)

The ORT warning `Skipping pci_bus_id for PCI path at ... 5620e0c7-...` is an ORT
device-discovery diagnostic for this host's virtual PCI bus; it does not affect
correctness.

---

## Security

Holden's **final verdict: 🟡 YELLOW — May ship**
(`docs/EP_PLUGIN_EXPORT_SECURITY_AUDIT.md`, `.squad/decisions/inbox/holden-ep-plugin-final-verdict.md`,
2026-08-10T22:42Z). All three original ship-blocking findings resolved; no new
CRITICAL or HIGH findings. Two LOW advisory items are recorded for post-merge
follow-up — they do not block merge.

### Resolved findings (all ship-blockers cleared)

| ID | Finding | Fixer | Status |
|----|---------|-------|--------|
| H1 | `static mut HOST_ORT_API` data race | Nabil | **RESOLVED** — `AtomicPtr` Acquire/Release |
| H2 | `graphs` null-deref in `ep_compile` | Nabil | **RESOLVED** — null guard at `ep.rs` |
| H3 | Unsound `Send+Sync` on `OutboundGraphReader` | Nabil | **RESOLVED** — impls removed |
| N1 | `compute_execute` missing `catch_unwind` (CRITICAL) | Leon | **RESOLVED** — `compute.rs:552`; regression test at line 2115 |
| N2 | Negative dims wrap to `usize::MAX` in `kernel_ctx.rs` (HIGH) | Leon | **RESOLVED** — `validate_dims()` wired into `read_inputs`; 8 boundary tests |
| N3 | Macro entry points `CreateEpFactories`/`ReleaseEpFactory` unguarded (MEDIUM) | Isidore | **RESOLVED** — both wrapped; `ReleaseEpFactory` return type corrected to `void` |
| UAF | `OrtMemoryInfo` released while ORT holds pointer (CRITICAL) | Deckard | **RESOLVED** — `c92838d`; Holden confirmed correct, no double-free possible |

### Post-merge advisory items (LOW, not blocking)

| ID | Item | Owner |
|----|------|-------|
| NEW-1 | `compute_release_state` lacks `catch_unwind` — pattern violation; `ComputeState` is trivially droppable now but guard should be added before extending it | Leon |
| NEW-2 | `ep_compile_inner` does not clean up `out_infos[0..i]` on mid-loop failure; ORT contract for Compile errors is unspecified | Deckard |

---

## Process

The Reviewer Rejection Protocol was enforced throughout this branch:

- **N1 (`compute_execute`)** — original author Nabil locked out; reassigned to
  Leon. Verified resolved at `compute.rs:552`.
- **N2 (negative dims / `validate_dims`)** — Nabil locked out from `kernel_ctx.rs`;
  Leon resolved and wired the call.
- **N3 (macro catch_unwind)** — Deckard locked out from `lib.rs`/macro entry;
  Isidore resolved. `ReleaseEpFactory` return type also corrected.
- **UAF (`factory.rs`)** — reassigned to Deckard (owns `factory.rs`).
- **NonZero over-claiming** — reassigned to Isidore after Nabil's initial
  capability-claiming code was rejected.
- **`conformance_two_sessions` assertion bug** — reassigned to Pris (tester);
  fixed and now passing.

---

## Known Limitations / Not In Scope

- **CUDA EP:** Blocked on two fronts: (1) no CUDA toolkit or GPU on this host;
  (2) genuine adapter design work remains — allocator via `CreateAllocator`, stream/sync
  via `CreateSyncStreamForDevice`, device pointer crossing the ABI. This is not purely
  hardware-blocked.
- **f16/bf16 end-to-end coverage:** Our CPU kernels implement f16/bf16, but the EP
  implements no `GetKernelRegistry`, so type-constraint metadata is never registered
  with ORT. End-to-end f16/bf16 cannot be proven without wiring `GetKernelRegistry`.
  No fake test was written. Owner if pursued: Nabil / `ep.rs`.
- **Ops still declined:** Any op not in the 22 shape-inference rules is declined
  (`Declined` / fail-closed). They fall through to ORT's own CPU EP.
- **No GitHub push credentials:** This host has no `GH_TOKEN`/`GITHUB_TOKEN`, no
  SSH private key, and GCM cache is empty. The branch is committed locally. The PR
  must be opened by the user or a runner with credentials.
- **nxrt-native Rust trait ABI:** The `ExecutionProvider` Rust trait surface is
  implemented in `onnx-runtime-ep-api` but is not independently tested as a
  first-class plugin surface in this PR. The extension contract §524 requires both
  C ABI and Rust trait; the Rust trait half is partially wired but not separately
  validated end-to-end.

---

## Architecture-Contract Compliance (Standing Directive §524)

The standing directive requires: every extension seam exposes a stable C ABI with
dynamic loading support **and** a first-class Rust trait; the two surfaces stay in
sync; the ORT ABI evolves toward nxrt; fail closed on unsupported capabilities.

| Requirement | This PR |
|-------------|---------|
| Stable C ABI with dynamic loading | ✅ `CreateEpFactories`/`ReleaseEpFactory` exports; ORT `dlopen`s the cdylib |
| First-class Rust trait | 🟡 `ExecutionProvider` trait exists and is implemented by `CpuExecutionProvider`; the plugin adapter bridges it, but the Rust trait surface is not independently tested as a plugin surface |
| ORT ABI evolves toward nxrt | ✅ Plugin adapter is thin shim; core logic lives in the Rust trait impl |
| Fail closed on unsupported capabilities | ✅ `Declined` path in shape inference; `GetCapability` refuses ops whose shapes cannot be resolved; 22 explicit rules, no wildcard accept |

**Honest status:** The C ABI half is complete and verified. The nxrt-native Rust
trait half is wired but not independently validated as a plugin surface. Full §524
compliance requires a follow-up that tests the Rust trait side without going through
the C ABI bridge.

---

## Follow-Ups

1. **Leon:** Add `catch_unwind` to `compute_release_state` (NEW-1 LOW advisory —
   pattern violation; safe now but must guard before `ComputeState` is extended).
2. **Deckard:** Handle `out_infos[0..i]` cleanup on mid-loop `ep_compile_inner`
   failure once ORT contract is clarified (NEW-2 LOW advisory).
3. **Nabil / `ep.rs`:** Wire `GetKernelRegistry` so f16/bf16 type constraints are
   registered with ORT; enables end-to-end f16/bf16 conformance testing.
4. **CUDA EP:** Design allocator (`CreateAllocator`), stream/sync
   (`CreateSyncStreamForDevice`), and device-pointer crossing ABI; validate on a
   CUDA-capable host.
5. **Shape inference:** Add per-op rules for Reshape (computed dims), Split,
   TopK, Slice (negative axis) to expand what the CPU EP can claim.
6. **§524 Rust trait surface:** Add an integration test that exercises
   `CpuExecutionProvider` directly via the `ExecutionProvider` Rust trait without
   the C ABI bridge, to verify the two surfaces stay in sync.

# PR: ORT Plugin EP Export — Rust CPU EP via upstream ORT 1.27.0 (Milestone 1) + Parity & CUDA Prep (Milestone 2 in progress)

## Branch structure

| Branch | Milestone | Status |
|---|---|---|
| `squad/ep-plugin-export` | M1 — CPU EP fully exported as ORT plugin | 🟡 YELLOW — may ship (all CRITICAL/HIGH cleared) |
| `squad/ep-plugin-parity-cuda` | M2 — trait↔C-ABI parity, CUDA shim, f16/bf16 type constraints | 🔴 IN PROGRESS — no M2 commits landed yet (stacked; not independently mergeable) |

**Recommendation: two stacked PRs, not one.**
M1 (`squad/ep-plugin-export`) is independently correct and mergeable; merging it now unblocks downstream tooling without waiting for M2. M2 (`squad/ep-plugin-parity-cuda`) adds parity tests and CUDA scaffolding that are genuinely additive; they depend on M1's adapter but don't affect M1 correctness. Squashing them into one PR would make the review surface larger with no benefit, and would stall M1 behind M2's hardware-gated CUDA work. Stack PR2 on PR1 with a base-branch dependency in the PR description.

**Push is blocked:** No `GH_TOKEN`/`GITHUB_TOKEN`, no SSH private key, and GCM cache is empty on this host. `git ls-remote origin refs/heads/squad/ep-plugin-export` returned empty — neither branch exists remotely. A user or CI runner with write credentials must push and open the PRs.

---

**M1 commits:** `526a883`, `f81d98d`, `09635cd`, `c92838d`, `2fb7150`, `bad3682`, `415289bc`, `5fa8cb2a`

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

All commands run by Roy on `squad/ep-plugin-parity-cuda` at commit `5fa8cb2a8`,
2026-08-10T23:30Z. Output is quoted verbatim. (The branch is currently at the same
commit as `squad/ep-plugin-export`; no M2 commits have landed yet.)

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
test result: ok. 15 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.74s
```

21 tests total (6 lib + 15 integration), zero failures, zero ignored.

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

### Post-merge advisory items status (re-verified at `5fa8cb2a8`)

| ID | Item | Actual status |
|----|------|---------------|
| NEW-1 | `compute_release_state` lacks `catch_unwind` | **RESOLVED** — `catch_unwind` is present at `compute.rs:1563`; comment explicitly says "This fixes NEW-1 from the EP plugin security audit." Leon's work landed before the M1 hand-off, not post-merge. The PR doc listed it as post-merge advisory in error. |
| NEW-2 | `ep_compile_inner` does not clean up `out_infos[0..i]` on mid-loop failure | **OPEN** — no cleanup logic found in `ep_compile_inner` (`ep.rs:229`). LOW risk; ORT contract on `Compile` errors is still unspecified. Owner: Deckard. |

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

## Milestone 2 — In-progress work (branch `squad/ep-plugin-parity-cuda`)

As of HEAD `5fa8cb2a8`, the M2 branch has zero new commits beyond M1. The four
engineers are working concurrently but nothing has been committed to this branch yet.
Status per engineer based on code inspection:

| Engineer | Task | Status |
|---|---|---|
| Leon | `catch_unwind` on `compute_release_state` (NEW-1) | ✅ **DONE** — landed before M1 hand-off; code at `compute.rs:1563` with explicit comment. Formerly listed as "post-merge advisory" in error. |
| Pris | Trait ↔ C-ABI capability/numeric/error parity tests | 🔴 **NOT YET COMMITTED** — no such tests in `crates/onnx-runtime-ep-api/tests/`. §524 trait-half proof depends on this work. |
| Deckard | NEW-2 `ep_compile_inner` partial cleanup + `GetKernelRegistry` f16/bf16 | 🔴 **NOT YET COMMITTED** — `GetKernelRegistry: None` at `ep.rs:48`; no cleanup in `ep_compile_inner`. |
| Nabil | Device/allocator/stream adapter surfaces + `onnx-runtime-ep-cuda-plugin` shim | 🔴 **NOT YET COMMITTED** — crate `onnx-runtime-ep-cuda-plugin` does not exist. |

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

## Architecture-Contract Compliance (Standing Directive §524) — Updated for M1+M2

The standing directive requires: every extension seam exposes a stable C ABI with
dynamic loading support **and** a first-class Rust trait; the two surfaces stay in
sync; the ORT ABI evolves toward nxrt; fail closed on unsupported capabilities.

| Requirement | M1 status | M2 status |
|-------------|-----------|-----------|
| Stable C ABI with dynamic loading | ✅ Complete — `CreateEpFactories`/`ReleaseEpFactory` exports; ORT `dlopen`s the cdylib | ✅ Unchanged |
| First-class Rust trait | 🟡 `ExecutionProvider` trait exists and is implemented; plugin adapter bridges it | 🔴 Trait↔C-ABI parity tests (Pris's work) have NOT landed yet on `squad/ep-plugin-parity-cuda`. §524 requires both seams proven; only the C ABI side is verified by the 21 ORT conformance tests. |
| ORT ABI evolves toward nxrt | ✅ Plugin adapter is a thin shim; core logic lives in the Rust trait impl | ✅ Unchanged |
| Fail closed on unsupported capabilities | ✅ `Declined` path in shape inference; 22 explicit rules, no wildcard accept | ✅ Unchanged |
| Native nxrt dynamic ABI | 🔴 Not implemented — no `extern "C"` nxrt-native ABI exists; the only dynamic-loading surface is the ORT plugin ABI | 🔴 Not started in M2 either |

**Honest M2 §524 status:** The C ABI half is complete and verified by 21 ORT conformance tests. The Rust trait half is structurally wired — `CpuExecutionProvider` implements the trait and the adapter bridges it — but the trait surface has NOT been independently tested as a first-class plugin seam. Pris's trait↔C-ABI parity tests are the intended evidence for the trait half; they are in scope for M2 but have not been committed yet. The native nxrt dynamic ABI remains entirely undesigned and unimplemented in both milestones.

---

## Follow-Ups

1. ~~**Leon:** Add `catch_unwind` to `compute_release_state` (NEW-1)~~ — **DONE** (already in code at `compute.rs:1563`).
2. **Deckard:** Handle `out_infos[0..i]` cleanup on mid-loop `ep_compile_inner` failure once ORT contract is clarified (NEW-2 LOW advisory).
3. **Deckard/Nabil:** Wire `GetKernelRegistry` so f16/bf16 type constraints are registered with ORT; enables end-to-end f16/bf16 conformance testing.
4. **Pris:** Add trait↔C-ABI parity tests (capability round-trip, numeric precision, error propagation) in `crates/onnx-runtime-ep-api/tests/`. These are the §524 evidence for the Rust trait half.
5. **Nabil:** Design and implement `onnx-runtime-ep-cuda-plugin` shim with allocator ABI (`CreateAllocator`), stream/sync ABI (`CreateSyncStreamForDevice`), and device-pointer crossing. Validate on a CUDA-capable host.
6. **Shape inference:** Add per-op rules for Reshape (computed dims), Split, TopK, Slice (negative axis) to expand what the CPU EP can claim.
7. **Native nxrt dynamic ABI:** Design and implement a first-class native nxrt `extern "C"` ABI (separate from the ORT plugin ABI) to complete §524 compliance.

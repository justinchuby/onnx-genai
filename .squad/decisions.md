# Decisions — live standing directives

Last consolidated: 2026-08-12T06:00:00Z (Scribe #762 memory-safety wave; 6 inbox drops merged; wave narratives archived to decisions-archive/2026-08.md; live file compacted)
Last consolidated: 2026-08-12T04:30:00Z (Scribe #31974 coverage wave; 2 inbox drops merged; narrative waves from 2026-08-10 through 2026-08-12 archived to decisions-archive/2026-08.md; live file compacted from 50,788 bytes to ~19 KB)
Last consolidated: 2026-08-11T17:55:00Z (Scribe issue-triage session + autonomous fixes; 7 inbox drops merged — 6 new: mobius io-metadata #477 + cosmos3-edge readiness assessment, GBQ zero-point #785, ORT recurrent guard/loader dedup #786, VLM fixture #788, DRY decoder-io #784, VMM contiguous-VA investigation; 1 deduped: qwen35-27b-native already recorded under #779. 30-day archive gate evaluated at 28KB: no dated entries older than 2026-07-12, so nothing archived. Prior: 2026-08-11T16:03:10Z Scribe TopK-perf + 27B-native batch; 37 inbox drops merged, full narrative archived to 2026-08.md)

Standing governance rules and active directives. Full narrative is archived in `.squad/decisions-archive/2026-07.md`, `.squad/decisions-archive/2026-08.md`, and older `.squad/decisions/archive/` files.

This compaction preserved the complete pre-compaction live file in `.squad/decisions-archive/2026-08.md` under "Live decisions snapshot before #695/#700 compaction". Processed inbox drops archived there: cohaagen-695-hybrid-cache-fix.md, cohaagen-qmoe-route-parallel.md, copilot-contract-decisions-q2-q12.md, copilot-plugin-c-abi-everywhere.md, deckard-645-cached-dense-identity.md, harry-700-hybrid-cache-review.md, quaid-676-oracle-testfix.md.
Narrative waves through 2026-08-06 (hybrid Mamba #695/#700, QMoE #676, CUDA-graph #708, C1 capture) archived to `.squad/decisions-archive/2026-08.md`.

## Ledger health rule

Archive by SIZE, not age. Age-only no-ops during high-volume campaigns because most entries are recent, so the live file can exceed spawn-budget while "older than N days" matches nothing. When over the gate, preserve full history in `.squad/decisions-archive/{YYYY-MM}.md`, dedupe rebase-reintroduced sections, and keep live `decisions.md` to standing directives plus pointers. Concurrent Scribe runs are a structural hazard; assemble from inbox drops and check `git log origin/main..HEAD` before committing.

## Current active wave — 2026-08-12 (#762 Opus memory-safety wave)

### PR #762 — Memory safety defects in absent optional output machinery (draft, ready to leave draft)

**By:** Nabil (B1+B2), Batty (corrective wave), Sebastian (S1/S2/S3), Challenger (fourth adversarial review)
**Commits:** af45043fd (Nabil), b906ab2bb (Batty), a5448fa36 (Sebastian)

**B1 (heap buffer overflow):** Scratch buffers for absent optional outputs sized from slot dtype byte size (2 bytes for f16/bf16) but TensorMut hardcoded to Float32 — 2× overflow on every f16/bf16 op with omitted optional output. Fix: dtype from `output_dtypes[slot]`, buffer sized `max(byte_size, 8)`, `TensorMut.absent` flag, fail-closed on Undefined.

**B2 (routed path positional compaction):** Fused multi-node path skipped allocation for absent slots then re-paired sinks through shortened iterators — panic or misroute. Fix: `RoutedSlotKind` enum (Ort/Buffer/Absent) keeps every slot index aligned end-to-end.

**Corrective wave (Batty):** EP assignment assertion (Add/SkipLayerNormalization/Mul pinned to cpu_ep); `end_version: since` → `i32::MAX`; `struct_size` loader validation; `NXRT_REQUIRE_ORT_TESTS=1` fail-loud gate (verified by renaming all 16 ort-prebuilt dirs); `matmul_initializer_weights` fixture; 5 `.gitignore` negations.

**Challenger (v4 review):** 0 blockers. S1: canary test size mismatch (byte_size vs max(byte_size,8)); S2: mark_absent advisory-only; S3: phantom intermediate buffer slots. Ready to leave draft.

**Sebastian (S1/S2/S3):** `production_scratch_alloc()` helper extracted; `TensorMut::validate_write_dtype()` added; `NodeOutputSink::Absent` variant — `num_intermediate_buffers` no longer inflated. Removed 4 no-op identity transmutes.

**Test counts:** 280 passed, 0 failed (269 baseline → 280). Clippy + fmt clean. Miri: 4/4 canary tests clean.

### CUDA test honesty: dummy_fill_and_crossover whitelisted (PR #789)

**By:** Cohaagen — Added `dummy_fill_and_crossover` to `ALWAYS_RUN` in `.github/scripts/verify_cuda_test_honesty.py`, mirroring `capture_sync_contract` carve-out. Four pure-CPU deterministic probes that legitimately pass on no-CUDA host were not whitelisted. Unblocks CUDA compile lane for all PRs on current main. One-line whitelist + justification comment; no test behavior changes.

### cuBLASLt GEMM workspace: session-persistent shared peak

**By:** Copilot — MatMul, Gemm, MatMulNBits, FusedMatMulBias, and FusedGemm report the selected cuBLASLt heuristic `workspaceSize` and share one session-persistent executor peak. Attention Phase-2a remains in step-scoped composite buffer. Planning and execution use the same plan helper; reject any shortfall deterministically.


## Durable lessons — #762 absent-slot machinery (2026-08-12)

- **The absent-slot machinery has now produced four distinct defects:** compacted output slots, absent inputs aliased to input 0, a forgeable name-based sentinel, and a 2× heap buffer overflow. Any change touching optional-slot handling deserves disproportionate scrutiny.
- **Allocate and interpret with the same dtype.** Sizing a buffer from one dtype while handing the consumer a different one is a memory-safety bug. Derive both from one source and fail closed when it is unknown.
- **A canary test must mirror production allocation exactly.** Canaries allocating at `byte_size` while production used `max(byte_size, 8)` could not detect wrong-dtype writes — the padding absorbed them. A test that passes for a reason unrelated to its claim is the most-repeated defect on this PR.
- **Verify a fail-loud gate by actually creating the failure condition.** Renaming one `ort-prebuilt` directory was a false negative; only renaming all 16 proved the gate fires.
- **Third false "API does not exist" deferral.** `MemoryDevice_GetDeviceId` and `Session_GetEpGraphAssignmentInfo` (twice) were all claimed unavailable and all existed — the latter already in use in our own tree. Check the generated bindings before deferring.
- **Merging upstream `main` into a long-lived branch:** resolve append-only archives and `.gitignore` as **unions**, never by taking one side, or user work is silently lost.


## Extension contract standing directive (#524)

**By:** Justin Chu / contract audit

Every extension seam must expose a stable C ABI with dynamic `.dll`/`.so` loading support **and** a first-class Rust trait; the two surfaces must stay in sync. Ship both upstream ORT plugin-EP ABI and native nxrt ABI, evolving the ORT ABI toward nxrt over time. Do not replace dynamic extension seams with compile-time-only workspace linkage.

## Performance claim discipline

- A per-layer or microbenchmark speedup is not a model-level claim; confirm with Amdahl and real model-level measurement. Always state exact model, dtype, metric, prompt/token regime, host load, and runner.
- Separate measured/estimated/projected. Do not compare measurements under different host load without labeling. Same-run PR-vs-merge-base deltas beat absolute PR numbers.
- A SIMD/accelerated path without a reachability test is equivalent to an unwired placeholder.
- Benchmarks for 35B-A3B must build from a fresh `origin/main` worktree; stale local main caused a false blocker report on 2026-08-03.

## Native-vs-ORT fairness rule

Native-vs-ORT claims must compare the same artifact, quantization, accuracy level, and steady-state methodology with oracle-correct output. If one engine crashes, rejects the graph, or falls back to CPU/different kernels, report a capability gap rather than a throughput multiplier. ORT-CUDA still hard-crashes on 27B/35B-A3B artifacts, so 35B QMoE native tok/s is a standalone native number.

## CUDA / QMoE / hybrid model standing directives

- Classic transformer decode is 100% covered on CUDA for the listed qwen/phi dense families; control-flow ops (`If`/`Loop`/`Scan`) are executor-handled recursively and must not be added to the CUDA EP as normal kernels.
- Qwen3.5 hybrid CUDA coverage includes `CausalConvWithState`, fused `LinearAttention`/Gated DeltaNet, RotaryEmbedding, Bool NonZero, GBQ, rank-3 native positions, and text-only decode pipeline synthesis. Numerics accumulate in f32 and claim gates must reject unsupported configs loudly.
- 27B fused LinearAttention is the active lesson: loader keeps a model-local function as an op iff the selected EP claims it; otherwise inline for byte-identical fallback. Do not revive the removed `ONNX_GENAI_DECODE_INLINE_SCAN` flag.
- 35B-A3B next perf levers after QMoE route parallelism: CUDA-graph capture repair and norm fusion; norm work is roughly 50x above roofline and must be validated at model level.

## Native multi-component pipeline decoder seam

The pipeline decode loop is backend-agnostic through a **stateful** `PipelineDecoderComponent`. Do not drive native pipeline decode through stateless host seams that drop device-resident KV. `NativePipelineDecoder` owns device KV continuity; `PipelineDecodeLoopBackend` holds one component. Rank-3 mRoPE positions derive from declared `position_ids` shape, not model-name gates.

## Metadata and shape-inference rules

- All inference/pipeline metadata except io-shape must be explicit and general. Name guessing is forbidden. Missing required metadata should produce a clear error naming the missing key.
- Shape-inference container support is complete for `ValueType{Tensor|Sequence|Optional|Map}` foundation and Sequence/If/Loop/Scan/SequenceMap threading; tensor path must remain byte-identical. Optional/Map handlers and IR persistence remain deferred until demanded.
- Minimal-build transforms gate on both infrastructure and operator groups; shape-inference registrations use actual ONNX domain/version; attribute-dependent output typing follows the active default/value attribute.

## ORT cached-value cloning

Cloning an ORT cached `Value` covers all POD dtypes via the dtype-agnostic raw-bytes fallback. Use `Value::from_raw_bytes(value.as_raw_bytes()?.to_vec(), shape, dtype)` in terminal arms. Use `as_raw_bytes()` (host-guarded precise error on device tensors), never `to_raw_bytes()`.

## CUDA live weight offload (#63/#87)

Live CUDA weight paging is wired into the decode hot path but gated behind `ONNX_GENAI_WEIGHT_OFFLOAD=1`; default-off is byte-identical. Async page-in is on by default after #544; double-buffer look-ahead remains plan-only until Justin green-lights. Do not retry o_proj 2-way split-K (`K_SPLIT=2`) because it repeatably regressed 7B o_proj GEMV by 0.59%.

## Heterogeneous execution / function inlining

Current public session path selects one EP; `hetero.rs` is not the default stateful executor. Bounded legalization in `hetero::plan` must fail closed when an assigned provider declines a kept function op or function identity is ambiguous. Attribute-parameterized functions require first-class FunctionLibrary/overload-safe IR support before open-ended binding; integrated stateful per-op hetero execution remains tracked separately (#603 family).

## CLI and CI standing directives

- The CLI is a maintainer/developer tool, not a consumer product. Prefer features that shorten debugging/iteration or expose engine behavior. Remote-client mode, model registry/pull lifecycle, and conversion/quantization/fine-tune loops are explicitly rejected as CLI features.
- The REPL is the primary CLI investment; preserve native scrollback via ratatui inline viewport rather than full-screen alternate screen.
- Run tests on every platform. Linux fast jobs are early signal only; they do not replace full platform gates. Instrument coverage only where informative.
- A step that warns instead of failing is not verification: check HTTP status explicitly and validate archive magic bytes before extracting.

## Testing discipline

Assert on what the code did, not summaries. Run new tests in isolation before trusting full-suite green. A fixture whose every assertion is “the turn was dropped” cannot distinguish correct behavior from total breakage. Resolve shared policy once via a shared helper instead of duplicating stale resolution at two sites.

## Model artifact hygiene

Fetch large external models only when needed, measure, and delete immediately. Do not leave benchmark models in `models/` or worktrees.

## Testing and CI standing directives (additions 2026-08-11)

- **`cargo test --workspace` silently truncates on failure.** Always use `--no-fail-fast`. A run reporting "1555/2" was really 4580 passed / 20 failed / 436 ignored across 304 binaries. Fail-fast mode exits at the first failing binary and reports wrong totals; this masked real failures across the session.
- **Never commit `.squad/` files to external repos.** Deleting the files in a follow-up commit does not remove the content — git history retains it and the delete commit's message re-exposes the path. If `.squad/` is accidentally committed, purge via `git filter-branch` or `git-filter-repo` and force-push. This was discovered on upstream ORT PRs #31973 and #31974; both branches required history purge.
- **An agent's self-report is not evidence.** Sapper reported all four CUDA defects fixed; independent review found a use-after-free, a panic bomb making success unreachable, and a direction classification gap. Nabil's B2 deferral cited an API that did not exist — the API was present in the generated bindings. Verify implementation claims via command output, code reading, and test results; never accept "implemented" on face value.
- **Reviewer lockout is enforced end-to-end.** Sapper authored CUDA fixes → rejected by Gaff → Nabil fixed B1/B3/S4 → Batty fixed B2. No author revised their own rejected artifact. The chain must close with an independent verifier confirming each fix.

## Active historical pointers

For detailed per-PR narrative, use archives rather than expanding this live file. Primary locations: `.squad/decisions-archive/2026-07.md` for pre-August ledger, CUDA parity waves, Mac CPU EP/perf methodology, and July CLI/runtime records; `.squad/decisions-archive/2026-08.md` for fused LinearAttention, hetero legalization, 35B-A3B QMoE, #695/#700 hybrid cache fix, and August Scribe batches; older material remains under `.squad/decisions/archive/`.


## ORT plugin-EP ABI standing directives

### OrtMemoryInfo lifetime (USE-AFTER-FREE — caused real bugs)

`EpDevice_AddAllocatorInfo(_In_ OrtEpDevice*, _In_ const OrtMemoryInfo*)` stores the raw pointer; ORT does NOT copy it. **Do NOT call `ReleaseMemoryInfo` after a successful `AddAllocatorInfo`.** ORT releases it when `ReleaseEpDevice` is called. Release only on failure. Use `CreateMemoryInfo_V2` with explicit `OrtMemoryInfoDeviceType_CPU` / `OrtDeviceMemoryType_DEFAULT`; the legacy `CreateCpuMemoryInfo` leaves those fields uninitialized, producing garbage DeviceType:64 / MemoryType:28 after repeated register/unregister cycles.

### OrtGraph*/OrtNode* scope (CACHING BUG — caused real bugs)

`OrtGraph*` and `OrtNode*` handles passed to `GetCapability` / `Compile` callbacks must NOT be stored or cached beyond the callback return. ORT may free them immediately after. Copy all needed attributes and initializers into owned Rust data structures during the callback.

### Shape-inference fail-closed policy

`ShapeInference::for_op` / `for_node` return `Declined { op_type, domain }` for any op with no modelled rule. `infer_shapes` turns `Declined` into an error status — ORT receives a proper failure, not silently-wrong output tensors. Do not reintroduce a silent `SameAsInput(0)` fallback.

### Evidence discipline for implementation claims

A previous session reported the adapter crate as "Implemented (v1)" when it did not compile. **Implementation claims require quoted command output as evidence** (`cargo check` / `cargo test` output). "Passes locally" is not evidence; command transcript is.

## CI and workflow standing directives

**CI is asynchronous.** Do not wait for CI before continuing, reporting, or merging. Required local targeted tests, Clippy, builds, and hardware probes remain blocking. Fix CI failures found later in follow-up commits.

**Design autonomy.** The coordinator may make architecture and design decisions when evidence supports them. Direction-changing decisions must update durable design documentation (measurement, falsifier, limitations, rollback path). When work is separable without shared mutable state, prefer parallel agents in separate worktrees.

## Memory governance standing directives (2026-08-06/08/09)

- `MemoryGovernor` exposes a stable `MemoryAuthorityId`; each backing authority is named at construction; `VirtualBuffer` rejects a different governor before reserving or committing.
- CUDA weight residency admission uses two constraints: mapped granules vs. weight allowance, newly created handles vs. global physical headroom. Failed transactions release newly created handles.
- Multi-model server owns one concurrency-safe device authority per backend/device domain; engine host/disk ledgers remain private.
- QMoE workspace: kernels declare typed workspace requirements; native CUDA prefill resolves QMoE shapes and reserves one reusable session-persistent workspace peak before the admission callback.
- Explicit byte `--vram-limit` enforced at engine load: native CUDA derives offload budget; non-offload backends fail at load if weights exceed limit. Derived budget = VRAM limit minus device KV/recurrent state, and must meet the largest lazy-weight node working set.
- CUDA weight offload defaults to async mmap-backed page-in with fence-ordered copy into reusable pinned staging. Synchronous demand-copy path available via `ONNX_GENAI_WEIGHT_OFFLOAD_ASYNC_PAGEIN=0`.

Full narrative in `.squad/decisions-archive/2026-08.md` (DROP sections: copilot-memory-authority-contract, copilot-committed-granule-admission, copilot-shared-device-authority, copilot-qmoe-workspace-stage0, copilot-vram-limit-load-enforcement, copilot-705-weight-offload-prefetch, copilot-mapped-growth-grant, copilot-ci-is-asynchronous, copilot-design-autonomy-and-parallel-work).


## 2026-08-12 — Apple scope: macOS arm64 / Apple Silicon ONLY

**By:** @justinchuby (scope correction), Mariette (PR #31993 lockout revision), Coco (PR #32001 lockout revision), Coordinator (verification)

### ⚠️ STANDING CONSTRAINT — applies to ALL future Apple work

**Apple scope is macOS arm64 / Apple Silicon ONLY.** Intel Mac and universal2 are out of scope. Gate with `APPLE + arm64/aarch64`. Do not add x86_64 Apple slices or Intel fallback tests. Do not let universal-binary concerns block enabling ARM kernels. Preserve the portable non-Apple fallback when the compile option is off. **iOS is not implied** — unless separately justified, Apple work stays scoped to macOS arm64.

This **narrows** the earlier Apple framework policy entry (Accelerate/BNNS/vDSP eligible when Apple-only, opt-in, portable fallback): that policy still stands, but its platform scope is now macOS arm64 only. **Read both entries together — neither stands alone.**

**Rescoping is not the same as removing a guard.** The `#if defined(__APPLE__) && defined(MLAS_TARGET_ARM64)` compile-time gate stays — it prevents the kernel reaching targets without FEAT_FP16. What was removed was the x86_64 *test slice*, not the *gate*.

**Use the tree's existing arch idiom.** `onnxruntime_target_platform STREQUAL "arm64"` is the canonical upstream variable, already used at `cmake/CMakeLists.txt:567/575/589` — prefer it over inventing a new condition from `CMAKE_OSX_ARCHITECTURES`.

### PR #31993 (Mariette, lockout revision) — rescoped to macOS arm64 only

- Removed the `#else` branch in `test_cast_fp16.cpp` that asserted null dispatch pointers on non-ARM64 Apple (x86_64 slice test, now out of scope).
- Rescoped commit messages and PR body from universal2/iOS/Intel to macOS arm64 only.
- Compile-time gate `#if defined(__APPLE__) && defined(MLAS_TARGET_ARM64)` unchanged.
- Positive `ASSERT_NE(...Kernel, nullptr)` dispatch assertions survive — test remains non-vacuous.
- Head: `68ee0de`.

### PR #32001 (Coco, lockout revision) — rescoped to macOS arm64 only

- Added `onnxruntime_target_platform STREQUAL "arm64"` condition to `cmake/CMakeLists.txt`.
- Implemented as `elseif` after the `if(NOT APPLE)` check, using warn-and-disable to match `onnxruntime_USE_SVE`/`onnxruntime_USE_KLEIDIAI`.
- Rescoped option description, MLAS comment (removed "iOS 4.0+"/"macOS 10.3+") and `build_args.py` help text.
- Verified no-behaviour-change-when-disabled on Linux x86-64.
- Head: `52db6351b5`. PR remains draft.

### Durable lessons

- **Non-arm64 Apple slices are out of scope.** Do not add `#else` branches for x86_64 Apple test slices; do not reference universal2 or Intel Mac in commit messages or PR bodies for ARM kernel work.
- **`onnxruntime_target_platform` is the canonical arch variable on Apple.** Already used at CMakeLists.txt:567/575/589; do not invent an alternative from `CMAKE_OSX_ARCHITECTURES`.
- **warn-and-disable, not FATAL_ERROR**, for platform-check failures in optional ISA options (per SVE/KleidiAI idiom).

---


## Archived narrative waves pointer (2026-08-12)

All wave narratives from 2026-08-10 (EP plugin export), 2026-08-11 (PR #762 parity, upstream CI correction, Apple MLAS f16 cast), and 2026-08-12 (CUDA MatMulNBits, Apple framework infra, rejection-response wave, PR #31973 threshold fixes) are in `.squad/decisions-archive/2026-08.md` under "ARCHIVED 2026-08-12T04:30:00Z".

**Status snapshot (2026-08-12T04:30:00Z):**

| PR | Status | Head |
|----|--------|------|
| #31985 | **MERGED** `f2dfa4e9eb` | — |
| #31973 | Draft — threshold + comment fixes landed | — |
| #31974 | Draft — coverage wave landed (`a12c7ddde3`), no blockers |  |
| #31988 | Draft — **parked pending GPU** | `dc1e173e4b` |
| #31993 | Draft | `02a9f34` |
| #32001 | Draft | `0d924a421b` |
| #32003 | Draft | `23dcfddaaf` |

## 2026-08-12 — BF16 PrePack and generic-broadcast coverage (PR #31974 Opus v2 wave)

**By:** Chew (coverage), Holden (delta review), Coordinator (verification)

### Durable lessons

- **Test the path real models take.** BF16 scale/bias are always constant initializers in practice — they route through `PrePack()` at session init (bf16→f32 conversion). The graph-input path was well tested; the PrePack path had zero coverage. `is_initializer=true` in OpTester triggers PrePack. Coverage gaps on paths models always take are worse than gaps on exotic paths.
- **Internal blocker labels (`B1`–`B6`, `N1`–`N4`) are a leak class.** They mean nothing upstream and reveal internal process. Sweep for them alongside `squad`, `nxrt`, persona names, and internal issue numbers before every push.
- **When correcting a tolerance comment, verify against the checker, not intuition.** The real behaviour is `absolute + relative × |expected|` (numpy.isclose semantics, `checkers.cc:117-120`). A stated absolute tolerance understates effective tolerance. Confirm the regression-detecting margin survives the relative component — here it did: 0.00011 effective vs 0.004 pre-fix error (36× margin).
- **Apple/macOS CI dependency downloads fail often and before compilation.** Observed causes: cpuinfo, XNNPACK, eigen3, FXdiv (`status_code: 60`) and vcpkg bootstrap (`curl failed to verify the legitimacy of the server`). `gh run rerun` refuses fork-PR jobs; only retrigger is a push. Do not mark ready while CI is red even when the failure is provably infra.

### Test counts after coverage wave (fresh build, coordinator-verified)

- 20/20 BF16 tests pass (was 17)
- 106/106 LayerNorm suite tests pass (was 103)
- 7/7 SkipLayerNorm PrePack validation tests pass


---

## 2026-08-12 — PR #762 EP plugin parity: ready-for-review lessons

**By:** Rachael, Coco, Isidore, Gaff, Freysa, Coordinator

### Durable lessons

- **#762 reached ready after five full Opus reviews plus a focused delta.** Each round found real defects: guessed output dtypes, a use-after-free, a panic bomb making the success path unreachable, compacted optional slots, a forgeable name-based sentinel, and a 2× heap buffer overflow. Rounds four and five found progressively smaller issues — the rate of discovery, not the absence of findings, is what signalled readiness.

- **Extract shared helpers rather than "keeping copies in sync".** Two `find_ort_lib_dir` copies had already diverged before anyone noticed; a `tests/common/` module plus `#[path]` includes made drift impossible. The same reasoning applied to `scratch_alloc_bytes`, where drift meant a heap overflow.

- **A validator nothing calls is a claim, not a mechanism.** `validate_write_dtype` had no production caller; the honest resolution was to document it as a test-exercised contract helper and name the real guard, not to leave it implying runtime enforcement.

- **Prefer leaking to calling through an unvalidated vtable pointer.** An undersized `struct_size` means `release` may not exist; jumping through whatever follows is arbitrary code execution from a malformed plugin.

- **Marking ready over red CI requires documented baseline comparison.** Identify the shared root cause, reproduce it on `main` in a clean worktree, and show the branch does not touch the implicated crate. Always measure with `--no-fail-fast` and rebuilt binaries — the plain command truncates totals and stale binaries report stale counts.

### Merged inbox drops

- `rachael-762-gate.md` — `NXRT_REQUIRE_ORT_TESTS` gate hardening: `find_ort_lib_dir` honours `CARGO_TARGET_DIR`; all skip paths through gate; CI lane enabled.
- `coco-762-scratch.md` — `scratch_alloc_bytes` single source of truth; `validate_write_dtype` wired into tests; unroutable graphs fail at Compile.
- `isidore-762-abi.md` — CUDA `end_version i32::MAX` per-family; `offset_of!` for vtable offsets; undersized-vtable guard (leak-not-UB policy).
- `gaff-762-delta.md` — delta review; no blockers; two substantive follow-ups (both addressed by Freysa).
- `freysa-762-final.md` — `tests/common/ort_discovery.rs` via `#[path]`; `validate_write_dtype` documented as test-only.
- `iran-762-clippy.md` — clippy identical-branch fix in `loader.rs`; `||` merge preserving `struct_size` short-circuit.

## 2026-08-12 — PR #32003 CUDA matmul_4bits_common.cuh narrative fix + PTX evidence (Batty)

**By:** Batty (corrections), Coordinator (independent PTX verification)
**PR:** microsoft/onnxruntime#32003

### Corrections made to PR body

1. **`(void)` unused-parameter guards — labelled "defensive; not reproduced locally".** `nvcc 12.0 --compiler-options="-Wall -Wextra -Wunused-parameter"` at sm_53 and sm_80 produced zero diagnostics. `cudafe` strips the dead `#else` host body before the host compiler runs, so the guard cannot be shown to suppress anything.

2. **Strict-aliasing claim narrowed to specific component reads.** The pun fixed is `uint32_t` member → `half2`/`__nv_bfloat162` lvalue. `reinterpret_cast<half2*>(sums)` (same-type vectorised access) is deliberately retained and explicitly noted; the PR body no longer implies the file is pun-free.

3. **"New template instantiations" wording removed.** That framing was a leftover from the parent PR this was split out of; it does not apply to this standalone one-file fix.

### PTX codegen-equivalence evidence (Coordinator-verified)

Compiled minimal TU (base and head copies of `matmul_4bits_common.cuh` with `GPU_WARP_SIZE` shim, instantiating `half` + `__nv_bfloat16` overloads) for **sm_53, 70, 75, 80, 86, 90 × {-O0, -O3}**. Result: **12/12 pairs raw byte-identical**, no normalisation required. Toolchain: nvcc CUDA 12.0 V12.0.140.

### Durable lessons

- **PTX equivalence is a cheap, strong argument for a "codegen-neutral" refactor.** Compiling base and head to PTX across target architectures at -O0 and -O3 and diffing turns "this should be equivalent" into evidence. It needs only `nvcc`, no GPU. 12/12 pairs were raw byte-identical here.
- **Do not claim a warning was fixed without the exact diagnostic.** The `(void)` guards could not be shown to suppress anything: `cudafe` strips the dead `#else` host body before the host compiler runs. "Defensive; not reproduced" is the honest wording.
- **Narrow a safety claim to what was actually fixed.** Equivalent-looking `reinterpret_cast<half2*>(sums)` array access was deliberately kept as the canonical CUDA vectorised idiom, so claiming the file is now free of type-punning would have been false.
- **Wording inherited from a parent PR goes stale on a split.** "New template instantiations" made sense in the PR this was split out of and was simply wrong in a standalone one-file fix — re-read the whole body after splitting.

**Status:** PR #32003 marked ready for review.

Last consolidated: 2026-08-12T08:30:00Z (Scribe #32003 PTX-evidence wave; 1 inbox drop merged: batty-32003-ptx.md)


## 2026-08-12 — PR #32001: Comprehensive --use_apple_accelerate validation (Resch + Holden + Zhora)

**By:** Resch (initial fix), Holden (focused review), Zhora (deduplication)
**PR:** microsoft/onnxruntime#32001 — `onnxruntime_USE_APPLE_ACCELERATE` CMake option
**Head:** `3a0bd75aa3`

### What happened

`build.py` only checked `is_macOS()` for `--use_apple_accelerate`, so Intel Macs,
x86_64 cross-compiles, iOS, tvOS, visionOS, and Mac Catalyst all passed Python
validation and then silently downgraded in CMake — contrary to the PR body's "fails
loudly" promise.

Resch added rejections in `build_args.py` (`parser.error()`) and `build.py` (`BuildError`).
Holden's review found S1: the `build.py` copy was dead code (unreachable). Zhora
consolidated to a single site, keeping only `build_args.py` as the canonical validation
home. 13/13 tests pass; ruff clean; PR ready.

### Rejection set

| Target | Flag(s) | Guard |
|--------|---------|-------|
| Non-macOS (Linux, Windows) | `is_macOS()` returns False | `parser.error()` |
| Intel Mac / x86_64 cross | `getattr(args, "osx_arch", None)` not in `("arm64", "arm64e")` | `parser.error()` |
| iOS | `getattr(args, "ios", False)` | `parser.error()` |
| tvOS | `getattr(args, "tvos", False)` | `parser.error()` |
| visionOS | `getattr(args, "visionos", False)` | `parser.error()` |
| Mac Catalyst | `getattr(args, "macos", None) == "Catalyst"` | `parser.error()` |
| universal2 | not a valid `--osx_arch` choice | argparse-level rejection |

### Durable lessons

- **Duplicated validation is the fourth instance of the copy-drift failure family.** Two `find_ort_lib_dir` copies diverged silently; a scratch-sizing formula duplicated between production and tests left a heap overflow canary ineffective; an MLAS dispatch threshold restated as a literal in a test disagreed with production on RISC-V. The rule is now explicit: **never maintain two copies of a rule — extract, or have one delegate to the other.**

- **Verify the copy that actually executes.** The `build.py` validation block was reviewed line-by-line and was dead code; `build_args.py` exits first. Trace control flow before trusting a review of a duplicated site.

- **Argument validation belongs at parse time.** `parser.error()` in `build_args.py` fails cleanly before any build work starts; raising later in `build.py` is both later and, here, unreachable.

- **Error messages should name their actual cause.** "Only supported on macOS arm64" is misleading on an Intel Mac, where the host *is* macOS and only the architecture is wrong.

- **Body-versus-code agreement deserves explicit checking every round.** #32001 had three separate rounds where the PR body promised behaviour the code did not implement — a `FATAL_ERROR` that had become warn-and-disable, claims of x86_64/universal2/iOS support, and a loud-failure promise that only checked `is_macOS()`.

### Merged inbox drops

- `resch-32001-validation.md` — initial comprehensive validation + 9 new tests
- `holden-32001-focused.md` — focused review; S1 dead-code finding; no blockers
- `zhora-32001-dedupe.md` — deduplication to single validation site

Last consolidated: 2026-08-12T09:45:00Z (Scribe #32001 ready wave; 3 inbox drops merged: resch-32001-validation.md, holden-32001-focused.md, zhora-32001-dedupe.md)

## Durable lessons — #31993 NaN semantics + AArch64 runtime evidence (2026-08-12)

- **QEMU + a cross-compiler turns "unverifiable on this host" into real runtime evidence.** `g++-aarch64-linux-gnu` plus `qemu-aarch64-static` executed the actual NEON kernel and settled a NaN-semantics question no amount of code reading could. Reach for this before declaring an architecture untestable — but label it emulation, not hardware.
- **Hardware and software NaN handling differ in more than the quiet bit.** `FCVTN` preserves the payload; the scalar reference canonicalizes to `0x7E00`. Assert NaN-ness (exponent all ones, mantissa non-zero) and sign only; keep raw-bit equality for non-NaN values.
- **"Compiles and links" is not "runs".** #31993's CI lane only built macOS arm64; `onnxruntime_mlas_test` was never executed, while the PR body implied otherwise. Check whether a lane actually runs the tests before citing it as validation.
- **A PR can change behaviour for types it does not mention.** #31974 is a BF16 PR that also alters pre-existing **MLFloat16** stat precision and registration. Diff every touched overload, not just the new one, and disclose it.
- **Verify an agent's "no behaviour change" claim against the diff.** Sapper's was wrong on both counts and would have shipped an undisclosed change to existing fp16 output.

### Merged inbox drops (2026-08-12 final review wave)

- `iran-31993-nan-runtime.md` — NaN assertion fix, QEMU runtime evidence, FEAT_FP16 correction
- `sapper-31974-ab.md` — PrePack A/B testing decision

Last consolidated: 2026-08-12T10:15:00Z (Scribe final-review-wave; 2 inbox drops merged: iran-31993-nan-runtime.md, sapper-31974-ab.md)

## 2026-08-12 — PR #31973 N1: arch-specific test guard + benchmarks (Batty, Challenger, Deckard)

**By:** Batty (N1 fix + benchmarks), Challenger (delta review), Deckard (wording fix), Coordinator (verification)
**PR:** microsoft/onnxruntime#31973 — MLAS AVX2 LayerNorm kernel
**Head:** `4a16925a88`

### What happened

Three separate instances of the same cross-arch test bug appeared on this PR family:
1. A production `NormSize < 8` gate suppressed the RVV kernel on RISC-V.
2. A test restated the x86 dispatch threshold as a literal.
3. Six precision suites asserted centered-two-pass properties but on RISC-V ran against the RVV uncentered kernel.

Batty closed N1 (instance 3) by adding `HasCenteredTwoPassKernel()` — a compile-time predicate guarded by the same `#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_IX86)` as the production gate. Six precision suites now `GTEST_SKIP` when this returns false; on x86 they run and assert as before. Batty also produced benchmark numbers on AMD EPYC 9V74 (AVX2/FMA, no AVX-512): LayerNorm 6.8–11.9× over scalar Welford fp32 at N=128–4096, RMSNorm 2.3–3.6×, 1000 iterations, p50 median.

Challenger (delta review) confirmed no blockers: guard correctly scoped, benchmark baseline (Welford fp32) fair, 41/2 → 43/43 build verified. One nit: `mlas.h` said "x86-64" while the `#if` also covers 32-bit `MLAS_TARGET_IX86`. Deckard fixed this to "x86 (32-bit and 64-bit)" across `mlas.h`, `layernorm_kernel_avx2.cpp`, and six `GTEST_SKIP` messages (`4a16925a88`). Both PRs marked ready for review.

### Durable lessons

- **Arch-specific assumptions leaked into tests three separate times on one PR.** When a kernel is architecture-specific, its assertions are too — gate them with a predicate mirroring the production `#if`, never a restated literal. Use `HasCenteredTwoPassKernel()` as the pattern.
- **Publish benchmarks only with the baseline named and the scope bounded.** The numbers here are defensible because the baseline provably mirrors `layer_norm_impl.cc` (Welford fp32 for LayerNorm, single-pass sum-of-squares for RMSNorm), and the PR body states it is a per-row in-process microbenchmark with no end-to-end claim.
- **Making a comment more readable can make it less accurate.** "AMD64/IX86" → "x86-64" read better but excluded 32-bit x86, which the `#if` covers. "x86 (32-bit and 64-bit)" achieves both.
- **This host can measure x86 honestly** (AVX2/FMA, no AVX-512) — x86 performance claims should be measured rather than omitted on this host.

### Merged inbox drops

- `batty-31973-n1.md` — `HasCenteredTwoPassKernel()` guard decision
- `challenger-31973-delta.md` — delta review verdict
- `deckard-31973-wording.md` — "x86" wording fix

Last consolidated: 2026-08-12T11:30:00Z (Scribe #31973/#31974 ready wave; 3 inbox drops merged)

## 2026-08-12 — PR #31973 evidence-accuracy wave (Mariette, Gaff, Coordinator)

**By:** Mariette (kernel/numerics), Gaff (focused review), Coordinator (reproduction + correction)
**PR:** microsoft/onnxruntime#31973 — MLAS AVX2 LayerNorm kernel
**Head:** `fbf322f76b`

### What happened

Two evidence blockers were found and fixed by Mariette.

**B1 — Accuracy headline was not reproducible.** The original body compared the production scalar path against a deleted implementation; no reader could re-run the comparison. Replaced with figures printed by the committed test:

| Path | Error vs fp64 oracle |
|------|---------------------|
| Scalar Welford fp32 (baseline) | 9.3573e-01 |
| AVX2 centered two-pass (kernel) | 3.2976e-02 |

Kernel is **28.4× more accurate** at base=1e5, spread=1e-2, N=1024, eps=1e-6. Sweep: 180 cases / 0 failures / worst 2.2318e-02.

**B2 — RMSNorm benchmark exercised dead work.** Benchmark passed non-null `MeanOut` for simplified mode; production always passes `nullptr`. Fixed. RMSNorm speedups rose ~15-30% at larger sizes.

**Additional fixes:** dispatch assertion on first warmup iteration; non-zero case-count assertion on fp64 sweep; stale label (`avx2_welford` → `avx2_centered`); SCENARIO 3 comment corrected; scalar-vs-kernel division disclosure added.

**Gaff review:** Reproduced all accuracy figures to 4 significant figures. Confirmed `nullptr` MeanOut matches production at `layer_norm_impl.cc:507`. One nit: RMSNorm ~3.3x at NormSize 256 is optimistic (Gaff measured ~2.84x, ~14% lower). NormSize 15 RMSNorm body says ~0.83x; Gaff measured 1.00x — body is conservative. No blockers.

**Coordinator correction:** Coordinator had published benchmark figures (6.8x LayerNorm 128, prior RMSNorm values) before reproducing them. Shared-runner variance was ~15%, exceeding published precision. PR body was corrected to lead with reproducible accuracy figures, widen variance disclosure to ~15%, and round table to 1-2 significant figures.

### Durable lessons

- **Never publish a measurement a reader cannot re-run.** Evidence must come from committed code, printed by a test, with the exact command given. Comparing against a deleted implementation is not evidence.
- **Benchmark the arguments production actually passes.** Check argument shapes against the real call site before trusting a ratio. Non-null `MeanOut` in simplified mode charged dead work and hid the fast path.
- **Do not publish more precision than a shared runner supports.** 6.3x vs 6.8x (LayerNorm 128) and 2.84x vs 3.30x (RMSNorm 256) represent ~15% spread on the same host. Round, state the spread, and point at the reproduce command as source of truth.
- **Assert that a benchmark dispatched what it claims to measure**, and that a parameter sweep generated a non-zero number of cases. Without these, a benchmark can silently time the fallback and a sweep can silently prove nothing.
- **The coordinator published these errors.** They were caught by review, not self-check. When writing a PR body from agent-reported figures, reproduce the figures first — the same standard applied to agents applies to the coordinator.

### Merged inbox drops

- `mariette-31973-evidence.md` — B1/B2 fix decisions
- `gaff-31973-evidence.md` — focused evidence review verdict

Last consolidated: 2026-08-12T13:30:00Z (Scribe #31973 evidence-accuracy wave; 2 inbox drops merged)

## Merged inbox drops (2026-08-12 prior waves, merged this session)

- `coco-32001-lint.md` — ruff lint fixes for test_build_args.py (PR #32001)
- `pris-31993-ctest.md` — macOS arm64 CI already runs tests; no add_test needed (PR #31993)
- `rachael-32001-crosstarget.md` — cross-target rejection for --use_apple_accelerate (PR #32001)
- `holden-final-two.md` — final review of PR #32001 and PR #31993; both ready to leave draft

Last consolidated: 2026-08-12T13:30:00Z (4 additional prior-wave inbox drops merged)

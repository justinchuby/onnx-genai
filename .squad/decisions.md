# Decisions — live standing directives

Last consolidated: 2026-08-12T00:00:00Z (Scribe CUDA-graph-capture-arc wave; 4 inbox drops merged — chew-bf16-gqa-numerics-review, sebastian-nxrt-ep-cuda-wheelfix, sebastian-nxrt-ep-pypi merged into live entries; leon-762-test-followups archived with #762. SIZE gate: archived the initial EP-PyPI narrative, #762 memory-safety wave, and merged upstream ORT PR waves (#31973/#31974/#32001/#32003 + BF16 PrePack + parity + regression) to decisions-archive/2026-08.md. 49,117 B → leaner; standing directives preserved verbatim.)
Standing governance rules and active directives. Full narrative is archived in `.squad/decisions-archive/2026-07.md`, `.squad/decisions-archive/2026-08.md`, and older `.squad/decisions/archive/` files.

This compaction preserved the complete pre-compaction live file in `.squad/decisions-archive/2026-08.md` under "Live decisions snapshot before #695/#700 compaction". Processed inbox drops archived there: cohaagen-695-hybrid-cache-fix.md, cohaagen-qmoe-route-parallel.md, copilot-contract-decisions-q2-q12.md, copilot-plugin-c-abi-everywhere.md, deckard-645-cached-dense-identity.md, harry-700-hybrid-cache-review.md, quaid-676-oracle-testfix.md.
Narrative waves through 2026-08-06 (hybrid Mamba #695/#700, QMoE #676, CUDA-graph #708, C1 capture) archived to `.squad/decisions-archive/2026-08.md`.


## Ledger health rule

Archive by SIZE, not age. Age-only no-ops during high-volume campaigns because most entries are recent, so the live file can exceed spawn-budget while "older than N days" matches nothing. When over the gate, preserve full history in `.squad/decisions-archive/{YYYY-MM}.md`, dedupe rebase-reintroduced sections, and keep live `decisions.md` to standing directives plus pointers. Concurrent Scribe runs are a structural hazard; assemble from inbox drops and check `git log origin/main..HEAD` before committing.

## VMM / offload / streaming / batching push — durable results (2026-08-12)

**By:** Copilot (coordinator). Every claim is backed by a merged, executable test;
refutations are recorded alongside confirmations.

**Governing rule (#772/#776/#787):** `cuMemMap` maps whole granule-aligned windows onto
whole physical granules, so `committed bytes = granule × (windows containing ≥1 live byte)`.
**Layout controls residency** — the allocator cannot compact what layout scattered.
`CU_MEM_ALLOC_GRANULARITY_MINIMUM == RECOMMENDED == 2 MiB` here, so the floor is fixable
only by layout, not by shrinking the granule. Minimum mapping granularity spans ~500× across
platforms (Level Zero/Vulkan ~64 KiB, CPU mmap 4 KiB) → layout must be a queried per-EP,
per-platform capability (#783), not a constant.

**Confirmed:**
- Floor is layout-determined: 768 granules (~1.5 GiB) head-major → 96 (~192 MiB) seq-major →
  1/seq (~2 MiB) token-major = **768× reduction** (#787).
- Strided reads are not the obstacle: seq/head bandwidth ratio 0.80–1.02; 192 KB token-major
  stride measured 1.000 at 6 GiB working set — reads are DRAM-bound independent of stride
  (device memory already 2 MiB-page backed) (#778/#787).
- Offload and capture no longer mutually exclusive (#796): weights page under a stable VA;
  page-in remaps physical granules instead of returning a new pointer. Unblocked #755.
- Managed no-spill VMM is default, auto weight-streaming when a model exceeds budget (#798);
  a fitting model does not page (`FullResident`, offload off, 0 page-ins).
- Prefix sharing is sound (#793/#803): one handle maps into N=8 sequences under captured
  replay; ledger charges once, alive until last sharer, additional sharer costs 0 bytes.

**Refuted (and why it mattered):**
- "seq-major landed ⇒ 8× floor realised" — false: #794 measured head-major and seq-major
  committing identical bytes (bindings didn't consume the layout descriptor); fixed #797.
- "decoder structurally declines capture" — false (#804): `captures=0` came from a cached
  `ONNX_GENAI_CUDA_GRAPH=0` in a long-lived test process. #794/#801 misattributed it.
- "fixed KV stride removes growth-triggered re-capture" — true in mechanism, irrelevant:
  engine invalidates the graph unconditionally on growth (#805).
- "tokens per granule" KV cost model — wrong for head-major (retracted), exactly right for
  token-major. Layout is the whole story.

**#736 audit recurring finding (six slices):** 4/5 completed slices found **over-reservation**
(bytes charged on a path that never uses them), not ungoverned allocation — #751 IndexShare,
#795 GQA WS_SCORES (~128 MiB f32-only), #799 cuBLASLt GEMM (32 MiB heuristic ceiling, measured
0–96 B), #802 default-domain Attention scores (genuinely needed), #806 GQA QKV staging. Guidance
in `MEMORY_ARCHITECTURE.md`: **start from use, not from allocation** — governing a bypass without
sizing it to use converts invisible waste into charged waste (tightens #745 admission, reduces
concurrency).

**Method notes:** order-dependent test state cost two wrong conclusions this week
(process-frozen `RuntimeConfig` #804; CUDA context warmed by alphabetically-earlier sibling
#797) — #807 added a debug-only freeze guard, single-stream helper, and an inventory. Negative
results delivered as first-class outcomes. Never extrapolate an unmeasured number (`qwen14b-zp`
lacks `inference_metadata.yaml`, not native-loadable #384 — reported as not measured).

## 2026-08-12 — CUDA-graph capture arc COMPLETE for Muse-Glimmer-30B native decode

**By:** Deckard (#848), Batty (#850), Leon (#852), Sebastian (#855, #854), Chew (numerics review), Coordinator (dependency-ordered review + admin-merge)
**Result:** native CUDA decode **11.4 → 23.13 tok/s** (+103%); CUDA-graph capture now fully engages — **1 captured segment, 0 eager seams**. All 5 PRs merged to main (HEAD f85a82f0).

Sebastian's measured escalation diagnosis found capture blocked by a 5-stage chain
(classify → load → pin → bf16-kernel → skip-norm). Each PR removed one blocker, merged in
dependency order (#848 → #850 → #852 → #855 → #854; #854 rebased onto main after #855
squash-merged):

- **#848 (Deckard, Systems)** — graph-truth SWA detection; the vestigial `sliding_window`
  signal now routes Muse-Glimmer to shared-buffer / fixed-capacity KV (capture-stable).
  463 tests.
- **#850 (Batty, Engine)** — `PipelineEngine` runs the Muse-Glimmer embedding component on
  the native CUDA EP; model loads + decodes end-to-end on
  `--pipeline --backend native --ep cuda`; parity identical.
- **#852 (Leon, Engine/KV)** — pin the GQA fixed-capacity KV seq symbol so the capture
  classifier admits 52 GQA nodes (53 → 0 disqualifying symbols); two-gate-AND kept safe;
  growth-invalidation intact.
- **#855 (Sebastian, Perf)** — bf16 capture-safe `gqa_decode` kernel; 54 → 2 segments;
  22.52 tok/s; f64-oracle max_abs 1.953e-3. fp32 accumulation throughout (Chew 🟢 APPROVE,
  verified on H200).
- **#854 (Sebastian, Perf)** — bf16 skip-norm capture-flag fix; 2 → 1 segment, 0 seams;
  23.13 tok/s (= +33% with capture ON).

**Next lever (dispatched separately):** the remaining gap to ORT's 40 tok/s is now
**kernel-bound**, not capture-bound — Cast 40.1% (626 casts/token), MatMulNBits 21.1%,
GQA 14.1%. Cast round-trip elimination is the next target.

### bf16 GQA decode kernel numerics — 🟢 APPROVE (Chew, PR #855, verified on H200)

- **fp32 accumulation is airtight.** bf16 touches data only at load/store boundaries
  (`__bfloat1622float2`, `__floats2bfloat162_rn`); every reduction (QK dot, online-softmax
  stats, warp merge, split-K scratch `float gqa_split_scratch[]`) is fp32. No bf16 arithmetic
  intrinsics anywhere in any reduction path.
- **Oracle is real f64 and actually gates.** `cpu_reference` accumulates in f64 over the
  bf16-rounded inputs; reproduced `max_abs=1.953e-3 max_rel=3.888e-3`. Bounds
  `abs<2e-2, rel<1e-1` (rel floored 1e-2). The 2e-2 ceiling is ~5× expected bf16 output
  rounding — a sane hard cap; the reported metric, not the ceiling, is the regression signal.
- **Split-K exercised + deterministic.** H200 132 SMs → 16-way split-K driven by parity
  totals 513/1023; determinism asserts under capture replay. Single-split fast path is
  bit-exact to the decode+merge two-step path (head_dim 64 and 128).
- **No capture-only precision shortcut.** Same kernel under capture as eager; distinct NVRTC
  module key (`decode_module_key_bf16`) avoids fp16 module-cache collision; unsupported bf16
  shapes (prefill Sq>1, odd/oversized head_dim) fall through to phase-2a.

## nxrt EP plugins on PyPI (nxrt-ep-cpu / nxrt-ep-cuda) — current wheel + publish facts

**By:** Sebastian (packaging + CUDA wheel CI fix), Coordinator. Consolidates the initial
2026-08-12 publish narrative (archived to `.squad/decisions-archive/2026-08.md`).

- **Two PyPI packages under `python/`:** `nxrt-ep-cpu` (import `nxrt_ep_cpu`) and
  `nxrt-ep-cuda` (import `nxrt_ep_cuda`), each bundling its plugin cdylib and exposing
  `get_library_path() -> str` + `register(session_options=None)` (thin wrapper over
  onnxruntime `register_execution_provider_library`, present in onnxruntime 1.28.0).
- **Build backend = setuptools + plain `cargo`, NOT maturin.** The plugin crates are C-ABI
  cdylibs (`CreateEpFactories`/`ReleaseEpFactory`), not PyO3 modules. `build_py` runs
  `cargo build --release -p <crate>`, copies the cdylib in, and `bdist_wheel` tags it
  `py3-none-<platform>` (no CPython ABI dep → one wheel per platform serves all Pythons).
  auditwheel (Linux) / delvewheel (Windows) repair. EP cdylibs must NOT link `libonnxruntime`.
- **`nxrt-ep-cuda` needs NO CUDA toolkit and NO GPU to build.** `onnx-runtime-ep-cuda` binds
  CUDA via cudarc `dynamic-loading`; `cargo build --features cuda` needs no nvcc/headers/GPU
  — libcuda/cuBLASLt/nvrtc/cupti are dlopen'd at runtime (`readelf -d` confirms zero CUDA
  libs linked). CUDA wheel CI therefore builds in the standard PyPA
  `quay.io/pypa/manylinux_2_28_x86_64` image (same as cpu), NOT
  `nvidia/cuda:13.0.0-devel-ubi9` (which lacks cibuildwheel's `/opt/python/*` toolchains and
  aborted the build). `auditwheel repair --exclude libcud*` kept as a defensive no-op.
- **CUDA 13 runtime wheels are REQUIRED deps**, pinned `>=13,<14`: `nvidia-cuda-runtime`,
  `nvidia-cublas`, `nvidia-cuda-nvrtc`, `nvidia-cuda-cupti`. Use the UNSUFFIXED names (the
  real CUDA 13 wheels: cuda-runtime 13.3.29, cublas 13.6.1.10, cuda-nvrtc 13.3.33,
  cuda-cupti 13.3.75); the `-cu13`-suffixed names are 0.0.1 stubs — avoid. `<14` locks the
  major so a future CUDA 14 can't be pulled against the CUDA-13-built cdylib. (Justin
  directive: "记得用cuda 13" / "cuda要包含需要的nvidia pypi包当dependency".)
- **Publish pipeline** `.github/workflows/publish-ep-plugins.yml` (environment `pypi`, OIDC
  Trusted Publishing, no tokens): cpu and cuda build/publish jobs are fully independent (no
  cross-`needs:`); `nxrt-ep-cpu` publishes first (PyPI allows one pending trusted publisher
  at a time); `publish-cuda` runs only on `publish_cuda=true` or an `nxrt-ep-v*` tag.
  `nxrt-ep-cpu` 0.1.0.dev5 is LIVE (manylinux_2_28 + macosx_arm64 + win_amd64).
- **EP hardware validation remains open (#768):** CI builds are compile + import only, no GPU
  session.

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


## Active narrative archive pointer (2026-08-12 capture-arc wave)

Merged/superseded narrative waves live in `.squad/decisions-archive/2026-08.md` under "ARCHIVED 2026-08-12T00:00:00Z (Scribe CUDA-graph-capture-arc wave)": initial nxrt EP PyPI publish narrative, #762 memory-safety wave + Leon's #762 test-quality followups, and the merged upstream ORT PR waves (#31973/#31974/#32001/#32003, BF16 PrePack coverage, #762 EP plugin parity lessons, #31993 NaN semantics, #31974 regression). Older material in `.squad/decisions-archive/2026-07.md` and `.squad/decisions/archive/`.

# Decisions — live standing directives

Last consolidated: 2026-08-12T00:00:00Z (Scribe CUDA-capture escalation batch; 1 inbox drop merged — coordinator-profiling-staging-gotchas (Muse-Glimmer native-CUDA profiling gotchas + vram-governor portability bug, no dedup collision). Size gate: 37,960 → archived EP-wheels/bf16/H200 narrative wave + verbatim inbox drops to decisions-archive/2026-08.md → ~28 KB, well under charter 50 KB. NOTE: charter mandates archive-by-SIZE; an "older than 30 days" age filter would have no-op'd here since all live entries are dated 2026-07-30..2026-08-12. Standing-directive floor is ~25 KB, so the prompt's 20,480-byte gate is below what can be shed without deleting standing directives.)
Last consolidated: 2026-08-12T20:40:00Z (Scribe EP-wheels/bf16/H200 merge session; 3 inbox drops merged — sebastian PyPI packaging + CUDA wheel-fix, leon #762 test-quality followups; new findings entry for #829/#831/#832/#838 merges + CDN incident + vlm KV regression. Size gate: 49,117 → archived #762 active-wave + all upstream-ORT PR narrative to decisions-archive/2026-08.md → 37,540 bytes, well under 50 KB.)
Last consolidated: 2026-08-12T15:52:00Z (Scribe nxrt-ep PyPI-publish session; 2 inbox drops merged — coordinator-nxrt-ep-cuda13 + copilot-vmm-push-summary; no dedup collisions. Size gate: 43,278 → ~50 KB, under 50 KB charter threshold, nothing archived.)
Last consolidated: 2026-08-12T06:00:00Z (Scribe #762 memory-safety wave; 6 inbox drops merged; wave narratives archived to decisions-archive/2026-08.md; live file compacted)
Last consolidated: 2026-08-12T04:30:00Z (Scribe #31974 coverage wave; 2 inbox drops merged; narrative waves from 2026-08-10 through 2026-08-12 archived to decisions-archive/2026-08.md; live file compacted from 50,788 bytes to ~19 KB)
Last consolidated: 2026-08-11T17:55:00Z (Scribe issue-triage session + autonomous fixes; 7 inbox drops merged — 6 new: mobius io-metadata #477 + cosmos3-edge readiness assessment, GBQ zero-point #785, ORT recurrent guard/loader dedup #786, VLM fixture #788, DRY decoder-io #784, VMM contiguous-VA investigation; 1 deduped: qwen35-27b-native already recorded under #779. 30-day archive gate evaluated at 28KB: no dated entries older than 2026-07-12, so nothing archived. Prior: 2026-08-11T16:03:10Z Scribe TopK-perf + 27B-native batch; 37 inbox drops merged, full narrative archived to 2026-08.md)

Standing governance rules and active directives. Full narrative is archived in `.squad/decisions-archive/2026-07.md`, `.squad/decisions-archive/2026-08.md`, and older `.squad/decisions/archive/` files.

This compaction preserved the complete pre-compaction live file in `.squad/decisions-archive/2026-08.md` under "Live decisions snapshot before #695/#700 compaction". Processed inbox drops archived there: cohaagen-695-hybrid-cache-fix.md, cohaagen-qmoe-route-parallel.md, copilot-contract-decisions-q2-q12.md, copilot-plugin-c-abi-everywhere.md, deckard-645-cached-dense-identity.md, harry-700-hybrid-cache-review.md, quaid-676-oracle-testfix.md.
Narrative waves through 2026-08-06 (hybrid Mamba #695/#700, QMoE #676, CUDA-graph #708, C1 capture) archived to `.squad/decisions-archive/2026-08.md`.

## Ledger health rule

Archive by SIZE, not age. Age-only no-ops during high-volume campaigns because most entries are recent, so the live file can exceed spawn-budget while "older than N days" matches nothing. When over the gate, preserve full history in `.squad/decisions-archive/{YYYY-MM}.md`, dedupe rebase-reintroduced sections, and keep live `decisions.md` to standing directives plus pointers. Concurrent Scribe runs are a structural hazard; assemble from inbox drops and check `git log origin/main..HEAD` before committing.

## nxrt EP plugins on PyPI + CUDA 13 target (2026-08-12)

### 2026-08-12: EP plugin cdylibs published to PyPI as `nxrt-ep-cpu` / `nxrt-ep-cuda`
**By:** Squad (Coordinator), req. by Justin (@justinchuby)
**What:** The two ORT plugin-EP cdylibs are packaged and published to PyPI via
`.github/workflows/publish-ep-plugins.yml` (PR #819) with `python/nxrt-ep-cpu/*` and
`python/nxrt-ep-cuda/*`. Packaging uses **setuptools + plain cargo, NOT maturin** — the
plugins are cdylibs exporting the ORT plugin-EP C ABI, not PyO3 modules. EP cdylibs must
**NOT** link `libonnxruntime`. `nxrt-ep-cpu` 0.1.0.dev5 is LIVE (manylinux_2_28 +
macosx_arm64 + win_amd64). CUDA wheel build (PR #824) switched the cuda job from
`nvidia/cuda:13.0.0-devel-ubi9` to standard `quay.io/pypa/manylinux_2_28_x86_64`.
**Why:** Ship the EP plugins as installable wheels consistent with the extension-contract
directive (#524: stable C ABI + dynamic loading).

### 2026-08-12: `nxrt-ep-cuda` needs no CUDA toolkit/GPU to build; NVIDIA runtime wheels are required deps
**By:** Squad (Coordinator)
**What:** `onnx-runtime-ep-cuda` uses cudarc `dynamic-loading`, so `cargo build --features
cuda` needs **NO CUDA toolkit and NO GPU** — CUDA libs are `dlopen`'d at runtime
(`readelf -d` confirmed the `.so` links zero CUDA libs). The four NVIDIA runtime wheels are
**REQUIRED** deps pinned `>=13,<14` (unsuffixed names are the real CUDA 13 wheels;
`-cu13`-suffixed are 0.0.1 stubs).
**Why:** Removes the toolchain/GPU requirement from the CUDA wheel CI job and pins the EP
wheel to CUDA 13 at runtime.

### 2026-07-30: nxrt-ep-cuda wheel targets CUDA 13
**By:** Squad (Coordinator), req. by Justin (@justinchuby)
**What:** The `nxrt-ep-cuda` PyPI package must build against / target CUDA 13. Runtime NVIDIA
deps use the CUDA 13 wheels (nvidia-cuda-runtime>=13, nvidia-cublas>=13,
nvidia-cuda-nvrtc>=13, nvidia-cuda-cupti>=13), matching the existing `nxrt[cuda]` extra in
crates/onnx-runtime-python/pyproject.toml.
**Why:** User directive "记得用cuda 13"; keeps EP wheel consistent with the main nxrt CUDA
wheel toolchain.

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


## 2026-08-12: Profiling Muse-Glimmer-30B on native CUDA — two gotchas + a portability bug

**By:** Squad (Coordinator), for Justin Chu (merged from inbox: coordinator-profiling-staging-gotchas.md)
**What:**
1. **Multimodal genai_config ≠ single-decoder profiling.** Muse-Glimmer's genai_config.json declares `vision`+`embedding`+`decoder`, so the compat translation resolves `shape()==Multimodal` and NEVER builds the single-decoder `model.io` block (kv_inputs/kv_outputs). Result: `profile_native --model <dir>` (single-decoder path) fails governor init with "cannot derive the KV memory budget ... declare model.io.kv_inputs". This is NOT a KV-geometry regression — the KV tensor head dims are concrete ([batch,2,seq,128]). Fix for profiling: stage a decoder-only dir with a genai_config.json that has vision/embedding stripped (shape()==SingleDecoder), model.onnx+data+tokenizer.json co-located. Staged copy lives at /tmp/muse-glimmer-staged. Alternative: `profile_native --pipeline` against the real multimodal dir.
2. **profile_native model resolver** needs a dir containing BOTH model.onnx AND tokenizer.json (not the decoder/ subdir alone, not the top-level models/ dir).
3. **PORTABILITY BUG (real):** `resolve_vram_limit_bytes` resolves the default `Fraction(0.90)` vram_limit against `fallback_capacity_providers` = PROVISIONAL 8 GiB (governor.rs:93), never the real device. So on any GPU (even 143GB H200) the governor caps device leases at ~7.2GB and models >7.2GB fail to load resident. profile_native has no vram flag; `ONNX_GENAI_VRAM_LIMIT` is wired only into the server CLI. Sebastian is fixing this via real cuMemGetInfo device-capacity detection (branch squad/native-cuda-decode-perf). [UPDATE: landed in batch #840 / 629fbf90 — real cudaMemGetInfo device-capacity detection + CudaFoldConstantCast pass; native decode 10.2→11.4 tok/s, +11.8%.]

**Why:** Two agents already lost time re-hitting the model-path/KV errors when profiling this multimodal 30B. The provisional-8GB cap silently forces weight offload/paging, which corrupts perf measurements. Recording so the team stops re-deriving it.

---

## Archived narrative waves pointer

Full wave narratives are archived in `.squad/decisions-archive/2026-08.md` and `.squad/decisions-archive/2026-07.md`.

- **2026-08-12T00-00-00Z (Scribe CUDA-capture escalation batch):** the EP-wheels / bf16 / H200 merge wave (2026-08-12T20:40:00Z), its status snapshot table (#31985 merged, #31973/#31974/#31988/#31993/#32001/#32003 drafts), and the three verbatim inbox drops (sebastian nxrt-ep-pypi packaging, sebastian nxrt-ep-cuda wheelfix, leon #762 test-followups) were moved to `.squad/decisions-archive/2026-08.md` to keep the live ledger lean.
- Earlier 2026-08-10/11/12 waves (EP plugin export, PR #762 parity, upstream CI correction, Apple MLAS f16 cast, CUDA MatMulNBits, rejection-response, PR #31973/#31974 threshold+regression fixes) are also in `.squad/decisions-archive/2026-08.md`.

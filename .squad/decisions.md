# Decisions — live standing directives

Last consolidated: 2026-08-13T03:03:13Z (Scribe CUDA-40tok/s milestone batch; merged sebastian-cuda-cast-elimination (PR #860) + recorded Chew's PR #860 numerics sign-off — Chew's inbox drop file was absent, decision reconstructed from spawn manifest. NO archive: decisions.md 44,755 B, below charter 50 KB gate. NOTE: the spawn prompt's "archive entries older than 30 days at ≥20,480 B" is an age-based gate the charter forbids — it no-ops since all live entries are 2026-07/08, and 20 KB is below the standing-directive floor. Histories: sebastian 3,948 B / chew 6,951 B, both below the chronicle + 15,360 B gates, none summarized.)
Last consolidated: 2026-08-12T00:00:00Z (Scribe inbox-consolidation run @ main f85a82f0; 18 inbox drops merged — 9 CUDA-graph-capture-arc drops folded into a single "CUDA-graph capture arc" section (classify #848 → load #850 → pin #852 → bf16 kernel #855 → skip-norm #854; native decode 11.4→23.13 tok/s; capture fully engaged 1 segment / 0 seams; next lever = Cast round-trip elimination) + 9 parallel-work drops preserved as their own entries (copilot fence-witness/#851 mobius-flake/VMM-release, isidore win-arm64/ort-retry, nabil ort-discovery, resch cpu-bf16/clippy, roy registry-shrink). NO archive this run — decisions.md 28,679 B, below the gate. Histories checked: max 8,105 B (holden), below the 15,360 B gate, none summarized.)
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

## Native CUDA decode 23→40 tok/s — RMSNorm cast-fold + parallel bf16 reduction (2026-08-12, PR #860)

**MILESTONE:** Native CUDA EP now matches ORT (~40 tok/s) on Muse-Glimmer-30B int4
decode. The multi-session goal is **MET: 11.4 → 40.21 tok/s** (coordinator-confirmed on
H200, 3-run median), 1 capture segment / 0 seams, first-16 greedy token ids match
reference. Full 4-gate capture chain + Cast/norm elimination complete.

### 2026-08-12: Native CUDA decode 23→40 tok/s — RMSNorm cast-fold + parallel bf16 reduction (#860, MERGED)
**By:** Sebastian
**What:** Closed the final 23→40 tok/s lever by attacking RMSNorm cast round-trips *and*
the RMSNorm reduction they were hiding.
- Generalized ep-cuda `CudaDropNormalizationCasts` (was fp16 + Skip/SimplifiedLayerNorm
  only) to fold **bf16** activation casts around **`RMSNormalization`**. Muse-Glimmer wraps
  all 312 decoder RMSNorm nodes in `Cast(bf16→f32)→RMSNorm(f32)→Cast(f32→bf16)` (624 of
  834 decoder casts); the fold removes both wrappers and retypes the norm to native bf16 I/O.
- **Op-swap `RMSNormalization`→`SimplifiedLayerNormalization` in the fold.** ONNX
  `RMSNormalization` (opset 23) types output Y as scale type `V`, not activation type `T`;
  Muse-Glimmer's scale is f32, so post-optimization shape re-inference (`registry.infer_graph`)
  kept clobbering the bf16 retype back to f32, breaking the kernel's `output==X` invariant
  and forcing whole-session CPU fallback. `SimplifiedLayerNormalization` inference follows X,
  and both ops map to the **same** fused `RmsNormFactory→RmsNormKernel` (no mean subtraction)
  on CUDA and CPU EPs — swap is mathematically identical and re-inference-stable.
- **Parallel f32 tree reduction in `rmsnorm_bf16`** (kernels/normalization.rs) — where the
  throughput comes from. Full f32 accumulation; only the summation *order* differs from the
  serial `rmsnorm_f32` reference.
**Numbers (H200, `--pipeline --ep cuda --backend native`, `ONNX_GENAI_CUDA_GRAPH=1`):**
baseline (fold OFF) 23.16 tok/s; fold ON + byte-exact serial 23.43 tok/s (cast removal alone
is ~free under capture — Cast invocations fell 96%, but tok/s barely moved); fold ON +
parallel norm (shipped default) **39.94 tok/s, +72% over baseline**. Coordinator 3-run median
confirmed 40.21 tok/s.
**Why parallel reduction is the lever:** at M=1 decode `num_groups=1`, so the RMSNorm reduction
runs on one block; the serial `rmsnorm_f32` sums the 6656-wide mean-square strictly
left-to-right on `tid==0` (to CPU-byte-match). Across 312 norms/token that `fadd` chain is
~40% of captured decode; a tree reduction removes it. **40 tok/s and strict byte-exact-vs-serial
parity are mutually exclusive** (serial chain floor ≈33 tok/s).
**Escape hatch:** `ONNX_GENAI_CUDA_DISABLE_NORM_CAST_FOLD=1` routes back to serial `rmsnorm_f32`
for strict CPU-order byte-exact parity (at 23 tok/s).

### 2026-08-12: PR #860 numerics gate — 🟢 APPROVE (parallel tree reduction is *more* accurate)
**By:** Chew (numerics reviewer)
**What:** Gated Sebastian's #860. Verified fp32 accumulation is airtight, the op-swap is
execution-identical (both `RMSNormalization` and `SimplifiedLayerNormalization` map to the same
`RmsNormFactory→RmsNormKernel`), and an independent f64 oracle passes (4/4 tests, ≤1 bf16 ulp).
**Key finding:** the parallel tree reduction is **~807× MORE accurate** than the old serial
order (tree_err 2.07e-8 vs serial 1.67e-5 vs f64 truth) — the mean-square is at least as close
to f64 as, and in fact far closer than, the serial path. The observed ~37-token byte-exact-then-drift
is downstream int4-quant (accuracy-level-4 MatMulNBits) greedy sensitivity flipping an int8
boundary on a sub-ulp delta, **not** a norm regression. Env-var escape hatch retained for strict
CPU-order byte-exact. **Standing rule reinforced:** bf16 kernels accumulate in fp32 and are
oracle-gated against f64; a parallel tree reduction may be adopted over a serial order when the
oracle shows it is at least as accurate.

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

## CUDA-graph capture arc — Muse-Glimmer-30B native decode 11.4 → 23.13 tok/s (2026-08-12)

**By:** Sebastian (Perf/CUDA-EP), Deckard (Systems), Batty (Engine), Leon (KV & Buffers), Chew (precision review). Consolidated from 9 inbox drops. Every tok/s figure is measured (H200, `--pipeline --backend native --ep cuda`, int4, steady 128 tok, greedy parity preserved throughout: ids `[24, 372, 1045, 10016, 328, 2885, 262, 5091, ...]`).

**The 5-blocker chain (each merged; all prerequisites landed in order):**

1. **CLASSIFY — #848 (Deckard).** `detect_model_decode_path` (`decode/metadata.rs`) no longer routes a model to the capture-unstable growing/paged KV path merely because `inference_metadata.yaml` declares a `sliding_window`. A window is active **only when the exported graph enforces it** — an attention op (`GroupQueryAttention`/`MHA`/`Attention`/`SparseAttention`) carrying `local_window_size > 0` (graph-truth; ORT default -1 = global). New `graph_enforces_sliding_window()` + `effective_sliding_window()` (conservative: keep window when no graph readable, so Gemma/Mistral SWA never regress). Muse-Glimmer's 52 GQA ops carry **no** `local_window_size` (global attention); its `sliding_window:2048` was vestigial (only in our generated metadata; `genai_config.json` declares none + `past_present_share_buffer:true`).

2. **LOAD — #850 (Batty).** `PipelineEngine` now loads+decodes Muse-Glimmer end-to-end on the native CUDA EP. Previously the multimodal (vision+embedding+decoder) pipeline routed its **embedding** component to ORT, which lacks bf16 `Where(16)`, so load failed. Fixes: embeds-producer reclassified `prompt_only → every_step` (an `inputs_embeds`-driven decoder needs a fresh embedding per step); every_step components load on the decoder's CUDA device; skip ORT sessions for all components on the native backend (they'd reject bf16 `Where`/int4 `MatMulNBits`); lazy native prologue with inactive-component skip; empty `[0,hidden]` image-features seed for text-only prompts; bf16 acceptance on the native decode target (`native_decode/{load,cuda}.rs`, gates relaxed `f32|f16` → `f32|f16|bf16`); KV context ceiling threaded from `model.max_sequence_length`. Byte-exact greedy parity vs the `muse_decode` raw-session harness.

3. **PIN — #852 (Leon).** Capture engaged after #848+#850 but delivered zero speedup: the captured step fragmented into 54 segments / 53 eager seams (52 GQA + 1 SkipSimplifiedLayerNorm). Root cause: the build-time capture classifier seeds the GQA present/past KV penultimate seq axis as a **growing symbol** and force-declines every GQA node — a false positive for the fixed-capacity device-valid-length KV the runtime actually binds. Fix: an **engine-gated symbol exclusion** — at the point fixed-capacity device KV is bound, pin the GQA KV seq-axis symbols CONSTANT (`collect_capacity_pinned_kv_symbols` + `Executor::pin_fixed_capacity_kv_capture_symbols`, called only when `graph_enabled`). Growing-concat / paged / mask-less attention paths do NOT qualify → genuinely growing KV stays vetoed. **Two-gate AND preserved**: node captures only if (classifier: seq-independent) AND (kernel `capture_support()`: Supported) — the pin removes only the classifier gate; the kernel gate remains an authoritative backstop. Result: disqualifying set 53 → 0, but segments stayed 54 — GQA nodes now declined at the **CUDA-EP kernel gate** because Muse-Glimmer is bf16 and only f32/f16 GQA decode kernels existed. The pin is a necessary prerequisite for GQA capture on ANY model.

4. **bf16 KERNEL — #855 (Sebastian).** Added `gqa_decode_bf16` — a bf16 device-length split-K GQA flash-decode kernel mirroring `gqa_decode_fp16` (`__nv_bfloat16`/`__nv_bfloat162`, **fp32 accumulation preserved** — matmul + softmax in fp32, bf16 only at load/store; distinct NVRTC module key to avoid fp16 cache collision). Wired into `group_query_attention.rs` (`capture_candidate` dtype gate, read-path, dispatch) + `KvCachePath::Bf16DecodeRead`/`decode_module_key_bf16()`. Accuracy (Chew is standing precision reviewer): parity vs an f64-accumulated softmax oracle fed bf16-rounded inputs → max_abs=1.953e-3, max_rel=3.888e-3, within justified bounds (abs<2e-2, rel<1e-1; bf16's 8-bit mantissa ~8× coarser than fp16). Segments 54 → 2 (1 residual eager seam: bf16 SkipSimplifiedLayerNorm); throughput ~14.5 → 22.52 tok/s.

5. **SKIP-NORM — #854 (Sebastian).** Removed the last eager seam. `SkipSimplifiedLayerNormKernel`'s bf16-via-f32 path now uses a persistent grow-only f32 staging arena (`NormBf16Scratch`, mirroring `matmul_nbits::Bf16Scratch`) instead of per-call `cudaMalloc`/`cudaFree` (a `cuMemFree` forces a per-token stream sync → capture-unsafe seam). The bug: the first warm call *grows* the arena and the pre-capture audit sampled `capture_support()` right after, so `grew` demoted the flag at exactly that moment. Fix: only demote on `grew` when `is_capturing()` (a grow racing an in-progress capture is unsafe; the first warm-time grow sizes the arena once and leaves the base fixed for steady replay). Segments 2 → **1 captured segment, 0 eager seams** (whole decode step captures as one graph). Measured: capture OFF 17.35 tok/s → capture ON **23.13 tok/s (+33%)**.

**Prior groundwork (#840 / 629fbf90, Sebastian):** real `cudaMemGetInfo` device-capacity detection (fixed a portability bug where the default `Fraction(0.90)` resolved against a provisional 8 GiB cap, failing 15.3 GB models even on a 143 GiB H200) + `CudaFoldConstantCast` EP pass (folds 208 constant norm-weight `Cast(bf16→f32)`/token). Native decode 10.2 → 11.4 tok/s. Diagnosis that redirected the arc: decode is **dispatch/launch-overhead bound** (~1600 launches/token, GPU 0–2% idle), not GEMV-bound — CUDA-graph capture over fixed KV is the only lever to close the gap to ORT.

**H200 hardware validation (#832, Sebastian):** CUDA EP validated on physical H200 (Muse-Glimmer-30B, zero CPU fallbacks). Fixed runtime bf16 dtype rejections despite 100% placement: `Clip(int64)`, `MatMulNBits` (bf16 activations → stage bf16→f16, reuse tuned f16 GEMV, cached grow-only `Bf16Scratch`; 5.87→11.08 tok/s), `GroupQueryAttention` (bf16 cos/sin cache), `SkipSimplifiedLayerNorm` (bf16). Also enabled the shared-EP plugin `CreateEp` path (was silently CPU-falling-back) and fixed device intermediates for multi-node fused subgraphs (`KernelContext_GetScratchBuffer`, was `vec![0u8]` host ptr → `CUDA_ERROR_ILLEGAL_ADDRESS`).

**Next lever (post-arc):** decode is now **kernel-bound**, not dispatch-bound (per-op: Cast 40%, MatMulNBits 21%, GQA 14%). The dominant remaining cost is **Cast bf16↔f32 round-trips** (~626 calls/token). Closing 23 → ORT's ~40 tok/s needs **Cast round-trip elimination** — native bf16 data path / fuse Cast into MatMulNBits + norm consumers, extending `CudaFoldConstantCast`. Substantial EP graph-rewrite / kernel-io-dtype effort (overlaps Batty's decode-graph domain), tracked separately.

**Durable lessons:** (a) a metadata-declared feature (sliding_window) must be validated against **graph-truth**, not trusted blind — a vestigial window silently forced the non-capturable path. (b) The capture classifier's growing-symbol veto is a **false positive for fixed-capacity device-KV**; pin the seq symbol engine-side (the engine knows the binding is fixed; `Executor::build` does not), keeping the kernel `capture_support()` gate as an independent backstop. (c) A capture-safety flag sampled right after a warm-time arena grow reads false at the worst moment — gate the demotion on `is_capturing()`. (d) bf16 kernels accumulate in fp32; bf16 only at load/store boundaries, oracle-gated against f64 softmax.

## Parallel-work decisions (2026-08-12)

### Enforce async-copy fence-safety with a type-level completion witness (#843, Copilot)
When a host buffer is the *source* of an async H2D copy and later reused/freed (pinned-staging pool), reuse must be ordered after the copy **completes**, not enqueues. Enforce with a type, not a comment: the host-syncing copy primitive (`CudaRuntime::htod_async_elapsed_ms`) returns a `CopyCompleted` witness (zero-sized, field private to the `runtime` module — unforgeable); the reuse path (`PinnedStagingPool::release`/`PooledStaging::retire`) **consumes** one, so reuse is unreachable without proof the copy finished. A future switch to non-blocking copy fails to compile until a witness is threaded post-fence. `Drop` cannot *require* an argument → make `retire(self, CopyCompleted)` explicit and demote `Drop` to a leak-safe free-only fallback (never returns to pool; catch misuse with a `pinned_alloc_calls` counter). **Never assert in `Drop`** on this codebase (STATUS_STACK_BUFFER_OVERRUN). Compile-time only — no runtime/perf change.

### mobius_seqmajor parity gate is intrinsically flaky ~10–20% solo — issue #851 (Copilot)
`mobius_seqmajor_growth_parity_native_cuda` is flaky **even solo on a clean base** (coordinator measured 4/5 on clean `dccb40e8`; combined ~1 failure per ~10 solo runs) — an earlier "5/5 solo reliable" claim was under-sampled and is **retracted**. Mechanism hypothesis (#851): seq-major + capture ON + KV-growth retains a captured graph whose baked-in **weight** pointer is invalidated when a growth commit remaps backing in the shared VMM arena → replay dereferences a stale ptr → intermittent `CUDA_ERROR_ILLEGAL_ADDRESS` on a weight `cuMemcpyHtoD` (node/layer varies run-to-run). **A single green run is ~80–90% reliable, NOT 100%** — do not treat one pass as proof; do not dismiss a red without classifying it (crash vs data-mismatch). Contention adds its OWN OOM-family reds on top of the ~15% intrinsic flake, so a contended red is still worth a solo re-run — but a **solo red is a real signal to preserve**. Operational triage (still valid): check `nvidia-smi --query-compute-apps`; prefer a verified-solo window; a real return-to-pool-before-fence corruption fails the bit-identical parity subtest *deterministically and solo*.

### VMM releases counted as driver unmap runs (Copilot)
`GlobalVmmStats::releases` now counts contiguous `cuMemUnmap` operations rather than individual granules (adjacent weight-page granules are unmapped in one driver call). Keeps the metric honest about release-side driver churn reduction; committed-byte gauges still track the full released quantity.

### Windows ARM64 wheel for nxrt-ep-cpu (#829, Isidore)
Per Justin ("我们的ep也要发 windows arm 64的wheel"), added a `win_arm64` matrix row (`os: windows-11-arm`, `archs: ARM64`) to the **cpu** job of `.github/workflows/publish-ep-plugins.yml`. CUDA EP out of scope (no CUDA on Windows ARM64). **Critical caveat:** official CPython has **no `win_arm64` build for 3.10** (ARM64 Windows CPython starts at 3.11), so the global `build="cp310-*"` selector matches nothing there. The ARM64 row overrides `CIBW_BUILD: cp311-*` and setup-python to 3.11; other rows keep 3.10. The wheel is ABI-less (`py3-none-win_arm64`, bundles a plain C-ABI cdylib), so any CPython ≥3.11 drives the build. `rustup` installs the native `aarch64-pc-windows-msvc` toolchain (no cross-compile). Build-only CI (run 31623032180) validated the runner provisioning.

### Harden ort-sys ORT download against transient network flakes (#829, Isidore)
`onnx-genai-ort/ort-sys/build.rs::download_prebuilt` used `--retry 3` but not `--retry-all-errors`, so curl exit 52 (`CURLE_GOT_NOTHING`, empty reply) — not in curl's default retryable set — failed the whole job. **`--retry-all-errors` is NOT portable**: it's a curl ≥7.71.0 flag and the manylinux_2_28 (AlmaLinux 8) build container ships curl 7.61.1, which rejects it (exit 2). Fix: a **Rust-level retry loop** (4 attempts, success = `status.success()` AND `http_code == "200"`, 3s/6s/12s backoff) plus portable curl flags 7.61 supports (`--retry 5 --retry-delay 2 --connect-timeout 30 --max-time 300`). Preserves the exact 404/missing-asset panic messages on the final attempt; checksum/magic verification runs only after success.

### Unify ORT library filename via `ort_discovery::ort_lib_name()` (#762, Nabil)
`plugin_ort_e2e.rs` hardcoded `"libonnxruntime.so"` in 7 sites, panicking on Windows (`onnxruntime.dll`)/macOS (`.dylib`). Standing rules: all tests use `ort_discovery::ort_lib_name()` (never a hardcoded string); use `PathBuf::join` (never string concat) for path construction; `skip_if_missing!` is the ONLY skip mechanism in EP e2e tests (respects `NXRT_REQUIRE_ORT_TESTS=1`) — a hand-rolled match in `diag_ort_ep_api_nullcheck` bypassed the fail-loud gate and was fixed. Windows/macOS correct-by-construction (`cfg!(target_os)`), CI-verified only.

### Native bfloat16 coverage for the CPU EP — all ops (#831, Resch)
Per Justin ("全面检查cpu ep对bfloat16的原生支持 一口气支持所有op"). Audit of all **194 registered op keys**: the CPU EP already had broad first-class bf16 via the shared compute-in-f32 machinery (`dtype.rs` `dispatch_arith`/`dispatch_float` + `to_dense_f32_widen`/`write_dense_f32_narrow`); no op special-cased f16 while forgetting bf16. Only 4 f32-locked kernels that already compute in f32 rejected non-f32 at their gate — widened `DFT`, `VarlenAttention` (`pkg.nxrt`), `MoE` (`com.microsoft`), `IndexShare` (`pkg.nxrt`) to accept f32/f16/bf16 via the same helpers. Deliberately excluded (bf16 not a valid native type): integer/bitwise ops, quantized-int ops (`QLinearMatMul`, `DynamicQuantizeLinear`), `CompressedSparseAttention` (FP8/FP4 compressed-cache contract), and int/bool-output value-agnostic ops. **Principle: compute in f32, widen on read, narrow on write — never a bespoke bf16 arithmetic path.** Regression lock: data-driven `tests/bf16_conformance.rs` (~68 nodes, f32-vs-bf16). Compliant with Justin's instruction-availability directive by construction: bf16↔f32 widen/narrow is an exact scalar op requiring no CPU feature; the one native bf16 arithmetic path (`_mm512_dpbf16_ps` in `x86_bf16.rs`, used by MatMul/Gemm) is runtime-gated with a proven portable fallback.

### Fix arch-gated `dot_kernel` unused-param clippy error blocking CI (Resch)
Main's quality lane (`RUSTFLAGS="-D warnings" cargo clippy --all-targets`) was RED: `borrowed_affine_int4_matmul` (`matmul_nbits.rs`) takes `dot_kernel: DotKernel` referenced only inside `#[cfg(target_arch="aarch64")]` blocks → `unused_variables` hard error on x86_64. Fix: `let _ = dot_kernel;` at the function top — mirrors the in-file convention (three sibling helpers at ~3779/4272/4448 already do this; chosen over `cfg_attr(...allow...)` for DRY consistency). No behavior change.

### Fix server registry shrink test assertion (#821, Roy)
`failed_runtime_shrink_preserves_policy_and_ledger_limit`: updated assertion from `error.contains("committed bytes")` to `error.contains("cannot satisfy lowered resource limit")`. Commit `c7633eec4` (#740, VMM handle pooling) changed the wording "committed bytes"→"leased bytes" without updating the test. The stable prefix is the only anchor covering both rejection paths (`state.rs` pooled-unmapped + `memory_authority.rs` mapped-or-leased, which have different suffixes); the test's purpose (reject+rollback shrink-below-usage) is also guarded behaviorally by a snapshot-rollback assertion, not just the string.

## Archived narrative waves pointer

Full wave narratives are archived in `.squad/decisions-archive/2026-08.md` and `.squad/decisions-archive/2026-07.md`.

- **2026-08-12T00-00-00Z (Scribe CUDA-capture escalation batch):** the EP-wheels / bf16 / H200 merge wave (2026-08-12T20:40:00Z), its status snapshot table (#31985 merged, #31973/#31974/#31988/#31993/#32001/#32003 drafts), and the three verbatim inbox drops (sebastian nxrt-ep-pypi packaging, sebastian nxrt-ep-cuda wheelfix, leon #762 test-followups) were moved to `.squad/decisions-archive/2026-08.md` to keep the live ledger lean.
- Earlier 2026-08-10/11/12 waves (EP plugin export, PR #762 parity, upstream CI correction, Apple MLAS f16 cast, CUDA MatMulNBits, rejection-response, PR #31973/#31974 threshold+regression fixes) are also in `.squad/decisions-archive/2026-08.md`.

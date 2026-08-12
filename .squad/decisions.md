# Decisions — live standing directives

Last consolidated: 2026-08-10T21:09:11Z (Scribe EP plugin export wave; 35 inbox drops merged; full narrative appended to decisions-archive/2026-08.md)

Standing governance rules and active directives. Full narrative is archived in `.squad/decisions-archive/2026-07.md`, `.squad/decisions-archive/2026-08.md`, and older `.squad/decisions/archive/` files.

This compaction preserved the complete pre-compaction live file in `.squad/decisions-archive/2026-08.md` under "Live decisions snapshot before #695/#700 compaction". Processed inbox drops archived there: cohaagen-695-hybrid-cache-fix.md, cohaagen-qmoe-route-parallel.md, copilot-contract-decisions-q2-q12.md, copilot-plugin-c-abi-everywhere.md, deckard-645-cached-dense-identity.md, harry-700-hybrid-cache-review.md, quaid-676-oracle-testfix.md.
Narrative waves through 2026-08-06 (hybrid Mamba #695/#700, QMoE #676, CUDA-graph #708, C1 capture) archived to `.squad/decisions-archive/2026-08.md`.

## Ledger health rule

Archive by SIZE, not age. Age-only no-ops during high-volume campaigns because most entries are recent, so the live file can exceed spawn-budget while "older than N days" matches nothing. When over the gate, preserve full history in `.squad/decisions-archive/{YYYY-MM}.md`, dedupe rebase-reintroduced sections, and keep live `decisions.md` to standing directives plus pointers. Concurrent Scribe runs are a structural hazard; assemble from inbox drops and check `git log origin/main..HEAD` before committing.


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

## Current active wave — 2026-08-10 (EP plugin export)

### ORT plugin EP export — CPU EP end-to-end milestone

**By:** Roy, Nabil, Deckard, Pris, Holden, Leon, Isidore (branch `squad/ep-plugin-export`)

**What:** Upstream ORT 1.27.0 now genuinely loads, registers, and executes our Rust CPU execution provider (`onnx-runtime-ep-cpu-plugin`) as a real plugin EP via `RegisterExecutionProviderLibrary` → `GetEpDevices` → `SessionOptionsAppendExecutionProvider_V2` → `CreateSession` → `Run`. 82 adapter unit tests + 21 real-ORT conformance tests pass, 0 ignored. Branch has 7 commits; push blocked (no GitHub credentials on host).

**Ship status:** 🟡 YELLOW — may ship. All CRITICAL/HIGH blockers cleared (Holden final audit). Two LOW advisories post-merge: `compute_release_state` missing `catch_unwind` (assign Leon), `ep_compile_inner` partial-output cleanup on mid-loop failure (assign Deckard).

**Correct ORT 1.27 export symbols:** `CreateEpFactories` and `ReleaseEpFactory` (both required). Source: `onnxruntime_c_api.h:5579`, `onnxruntime_ep_c_api.h:2637,2661`. The full call sequence is `dlopen` → `CreateEpFactories` → `GetSupportedDevices` → `SessionOptionsAppendExecutionProvider_V2` → `CreateSession` → `Run`.

**CUDA EP:** Hard-blocked (no CUDA toolkit/GPU on host, plus device-memory/stream/allocator ownership unresolved). Not in scope this wave.

Full narrative in `.squad/decisions-archive/2026-08.md` (DROP sections: challenger-ort-plugin-abi-truth, nabil-ep-plugin-adapter, nabil-ep-device-enumeration, deckard-ep-compute-path, deckard-ep-shape-inference, deckard-ep-device-lifetime, holden-ep-plugin-ffi-audit, holden-ep-plugin-reaudit, holden-ep-plugin-final-verdict, isidore-ep-export-guards, leon-ep-compute-hardening, nabil-ep-plugin-hardening, nabil-ep-capability-integration, pris-ep-plugin-conformance, pris-ep-conformance-suite, pris-ep-conformance-final, roy-ep-plugin-export, roy-ep-export-milestone, roy-ep-provider-readiness).

### PR #728 — growing-symbol classifier round 4–8

**By:** Batty, Leon, Deckard, Roy, Gaff, Sapper, Sebastian

**What:** Multi-round reviewer rejections for the growing-symbol classifier in the executor. Key fix: the growing-set closure must cover ALL broadcasting handlers (not just elementwise). Path-B authoritative record closes the set. Full narrative in `.squad/decisions-archive/2026-08.md` (DROP sections: batty/deckard/gaff/leon/roy/sapper/sebastian-728-revision).

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

## Current active wave — 2026-08-11 (EP plugin parity + CUDA + upstream ORT LayerNorm)

### PR #762 — CUDA EP parity wave (branch `squad/ep-plugin-parity-cuda`)

**By:** Sapper (initial CUDA), Gaff (rejection review ×2), Nabil (B1/B3/S4), Batty (B2), Iran (B4 fail-closed), Roy (docs + undraft), Holden (final sign-off), Chew (test repair), Luba (nxrt inline buffer), Deckard (CUDA wiring + clippy), Pris (conformance suite)

**Outcome:** PR #762 undrafted, marked ready for review. 15 CI checks green; coverage job long-running.

**Key resolved defects (B1–B4 in CUDA EP):**

- **B1 — Use-after-free:** Raw pointer escaped `MutexGuard`, dangling on every allocator/stream/transfer callback. Fixed: components now hold `Arc<Mutex<..>>` clones; `with_ep` locks per-operation. Test `shared_ep_allocator_outlives_original_arc` is non-vacuous.
- **B2 — Pointer equality for same-device D2D:** Fixed via `is_same_device()` using `MemoryDevice_GetDeviceId` (present in ORT 1.27 bindings). Fast path: pointer equality. Null guard: fail-closed. API absent at runtime: fail-closed. 6 unit tests.
- **B3 — `CopyTensors` no direction classification:** Both src/dst were wrapped as device buffers regardless of actual memory type. Fixed via `Value_GetMemoryDevice` + `MemoryDevice_GetDeviceType` classify path. `CopyDirection::classify` exhaustive; unsupported directions return fail-status.
- **B4 (S4) — Panic bomb in factory constructor:** `create_ep_factories_for_shared_ep` now takes `ep_name: &str` directly; CUDA plugin extracts name before wrapping in Arc. The old constructor closure is unreachable on the shared-EP path.

**CUDA fail-closed gate:** `CreateEpFactories` returns zero factories in both `cuda`-on and `cuda`-off configs until hardware validation (#768). This replaced the earlier fail-open state that advertised a broken GPU EP.

**Output dtype standing directive:** `CompiledKernelEntry.output_dtypes: Vec<DataType>` — per-output dtypes read from ORT graph IR at Compile time; never inferred from inputs. Nodes with Undefined output dtype are declined at `GetCapability` (fail-closed).

**LayerNorm shape inference:** `ShapeInference::LayerNorm { axis, num_outputs, full_shape_outputs }` — output 0 gets full input shape; outputs 1+ get reduced shape `[d[0]..d[axis-1], 1,..,1]`. Negative axis handled at `for_node` time. Fail-closed: axis resolves outside range → node declined.

**nxrt inline buffer rule:** `NxrtStatus.message` is `[u8; 256]` with `message_len: u32` — a pure value type, no heap allocation. Eliminates cross-CRT allocator-mismatch UB at the cdylib boundary. `c_char` portability: use `*const std::os::raw::c_char`, not `*const i8` (aarch64 has `c_char = u8`).

**Full narrative:** `.squad/decisions/inbox/` (all drops for this wave are merged and deleted by this Scribe session).

### Upstream ORT PRs #31973 and #31974

**By:** Resch (kernel), Iran (fixes + dedup), Chew (BF16 numerics), Deckard (U type constraint), Gaff (reviews), Holden (re-reviews), Batty (ARM64 CI diagnosis), Luba (ARM inventory)

**#31973 — AVX2 MLAS LayerNorm/RMSNorm (`nxrt/mlas-avx2-layernorm`):**
- Welford SIMD AVX2 was badly inaccurate: per-lane fp32 means at high base/small spread hit 28% relative error. Fixed with centered two-pass + double-precision first-pass sum. The `_mm256_cvtps_pd` + `_mm256_add_pd` loop costs ~10% throughput vs 8-wide fp32 but eliminates variance collapse.
- AVX2+FMA3 dispatch guard: `platform.cpp:478` CPUID checks. `NormSize < 8` threshold moved inside x86-only guard (was gating RVV/other platforms).
- ARM64 Debug CI failure at [1452/1458]: strong OOM circumstantial evidence (silent process death, large Debug link targets, 32GB runner). Confirmed flake — went green on re-run. No code bug.

**#31974 — MLAS BFloat16 LayerNorm/RMSNorm (`nxrt/mlas-bf16-layernorm`):**
- BF16 stat precision fix (B5): `ComputeJob<BFloat16>` overloads now call `WriteStat<U>` (U=float) — stats written at f32 precision, not BF16 (3-digit) precision.
- Contrib U type constraint: macro changed from `(T)` to `(T, U)` for narrow types with `U=float`. Consistent with CUDA contrib and schema.
- `NarrowToFloat`/`FloatToNarrow` deduplicated into `onnxruntime/core/util/narrow_float_utils.h` (commit `6dd19a6f56`).
- BF16 numerics: f32 two-pass accumulation on BF16 inputs produces ≤1 BF16 ULP above quantization floor at N=65536. Recommended kernel tolerance: ≤2 BF16 ULP (≈1.6e-2 at unit scale). Do NOT use f32 tolerance (1e-4) for BF16.
- CI failures: Gradle CDN timeout, pytorch_cpuinfo FetchContent failure, DNS error in quantization model download — all INFRA FLAKES.

**CUDA upstream candidates dead:** Both MatMulNBits int4 block-128 GEMV and QMoE parallel routing are already covered by upstream ORT `main`. No portable gap found. Our CUDA advantages (graph capture, VMM, tiered KV) are runtime-level and not portable.

**`.squad/` leakage purged from both upstream branches via `filter-branch` + force-push.** Trees verified byte-identical after purge.

## Current active wave — 2026-08-11 (PR #762 third corrective wave)

### PR #762 — optional slot fidelity, ABI safety, test integrity (third rejection correction)

**By:** Leon, Sebastian, Isidore, Freysa (fixes); Luv, Challenger, Fact Checker, Pris, Gaff (reviews); Mariette, Coco, Resch, Rachael (corrections/hardening); Zhora (docs)

**Outcome:** PR #762 marked **ready for review**. EP crates 269 passed / 0 failed; workspace 4598 passed / 20 failed / 436 ignored — the 20 pre-existing on base `675b697bc`, none in EP crates. Five EP crates clippy-clean; fmt clean.

**Key resolved defects:**

- **BL2 — Output slot compaction (Leon):** `filter_map` in `graph_reader.rs` compacted optional absent outputs, breaking positional indexing. Fixed with `ValueId` placeholders for empty-named slots and scratch buffers for `DataType::Undefined` slots. `NodeInputSource::Absent` + `TensorView::absent()` added for BL3.
- **BL1 — LayerNorm axis pre-resolved against truncated rank (Sebastian):** `filter_map(|d| d.as_static())` collapsed `[B, S, H]` to `[H]`, resolving `axis=-1` to index 0. Fixed: `raw_axis: i64` stored; resolved per-invocation against actual runtime rank.
- **ABI safety (Isidore):** `NxrtStatus.code` is now raw `u32`; checked conversion via `NxrtStatusCode::from_u32()`; unknown codes → `None` → fail closed. `struct_size` validated before vtable access. CUDA diagnostics initialised before running status queries.
- **`disable_cpu_ep_fallback` (Freysa):** Set in `conformance_setup()`. Forces ORT to error if any node falls back; proved non-vacuous by observing `mixed_partition` fail correctly before exemption.
- **Dead code / EP declined optional-slot nodes (Mariette):** With fallback=1, optional-slot tests failed — EP was declining nodes. Three roots: claim filter rejecting `DataType::Undefined` outputs; `Clip` missing from shape-inference list; single-kernel fast path skipping input slot mapping. Fixed all three; scratch buffer sized from primary output dtype, not hardcoded 4 bytes.
- **Forgeable sentinel (Coco):** `__absent_output_*` prefix replaced with `absent_outputs: HashSet<ValueId>` (out-of-band; arena indices are uninfluenceable from model content). Rank destruction eliminated: `filter_map` → `map` → `Vec<Option<usize>>`; `build_conv` fails closed on `None` dims.
- **`Session_GetEpGraphAssignmentInfo` (Resch):** False deferral claim corrected by Fact Checker — API present since ORT 1.24. Wired in; 8 tests now assert specific ops owned by `"cpu_ep"`.
- **Test hardening (Rachael):** 14 tests now assert EP assignment. f16/bf16 and LayerNorm/RMSNorm coverage added. Non-vacuity proved via forced `"Relu"` assertion failures.
- **Docs accuracy (Zhora):** PR body rewritten; 8 stale SHA refs updated to `c1d2556b5`; explicit "What Is NOT Proven" section for CUDA hardware gap.
- **Final delta (Gaff):** No blockers. Helper duplication in test files noted as follow-up tech debt.

**Reviewer lockout chain:** Deckard → Batty/Sapper/Luba/Iran/Chew → Nabil → Batty → Leon/Sebastian/Isidore/Freysa → Mariette → Coco → Resch → Rachael; Gaff, Holden, Luv, Challenger, Fact Checker, Pris reviewed without revising own findings.

### Durable lessons from PR #762 (third corrective wave)

- **A passing test is not evidence the code under test ran.** Leon's BL2 fix shipped green while our EP declined the nodes and ORT's built-in CPU EP produced correct answers. Require `disable_cpu_ep_fallback=1` **and** `Session_GetEpGraphAssignmentInfo` assignment assertions for every real-ORT test.
- **Never encode semantics in a name.** `__absent_output_*` matched by `starts_with` was forgeable from untrusted model content. Use out-of-band state (`HashSet<ValueId>`) — ValueIds are arena indices the reader assigns; model content cannot influence them.
- **`filter_map` is wrong wherever position or rank is load-bearing.** It caused four separate bugs here: compacted output slots, compacted input slots, truncated rank for axis resolution, and truncated rank at two further sites. Use `map` to `Option<T>` and preserve length.
- **Verify an API's absence before deferring on it.** Two deferrals were justified by ORT APIs that existed: `MemoryDevice_GetDeviceId` (1.27) and `Session_GetEpGraphAssignmentInfo` (since 1.24). Check the generated bindings before filing a deferral.
- **Reviewer lockout held across seven rounds.** Chain above closed cleanly; no author revised their own rejected artifact.

## Current active wave — 2026-08-11 (Upstream CI correction wave)

### PRs #31973, #31974, #31985 — upstream ORT rebase and CI unblock

**By:** Deckard (doc-fix PR), Iran (rebase #31973), Sapper (rebase #31974 + conflict), Luba (Apple/arm64 CI triage), Holden (re-review #31973), Chew (leak scrub under lockout), Challenger (re-review #31974), Luv (review #31985)

**#31985 (mrope doc fix):** One-line removal of `(or omitting it)` from `docs/ContribOperators.md`. `mrope_section` is a required attribute (no default in `bert_defs.cc`); the phrase was factually wrong, not merely stale. PR reached **86/86 CI green** and was marked **ready for review**.

**#31973 rebase (Iran):** Rebased 7 commits onto `upstream/main` (`86d38813a8`). Zero conflicts. 42 MLAS LayerNorm tests pass. All five preserved properties intact.

**#31974 rebase (Sapper):** First attempt clean. Second rebase hit semantic conflict with upstream #31676 ("Validate SkipLayerNorm prepacked lengths") in `skip_layer_norm.cc`. Resolved by keeping upstream's `tensor_size > 0` guard and extending it to our bf16 path. Upstream's validation still covers bf16 because `ConvertMLFloat16ToFloatIfNeeded` handles bf16. 17 bf16 + 103 LayerNorm + 6 upstream prepacked tests pass.

**Persona name leaks (Holden → Chew):** Holden's re-review of #31973 found two agent names in C++ test comments ("Iran", "Pris"). Iran and Pris were barred under reviewer lockout. Chew replaced comments with technical descriptions and rewrote history (interactive rebase, amend two commits). Force-pushed; strings unreachable in any reachable commit.

**Apple/arm64 CI (Luba):** All failures occur before compilation. `cpuinfo` and `XNNPACK` archive download failures on #31973/#31974; job timeout at step 1453/1459 on #31974. Same jobs passed on control PR #31985 at the same time. All confirmed infra flakes. `gh run rerun` refused for fork-PR jobs — only retrigger is a push.

**Re-reviews:** Challenger re-reviewed #31974 — 0 blocking, 0 substantive, 2 cosmetic nits. Stat tests genuinely fail against pre-B5 code (bf16 quantization step is ~390× coarser than 1e-5 tolerance). Both #31973 and #31974 converted **back to draft** per user instruction — correct posture is draft until CI board is green.

### Durable lessons from upstream CI correction wave

- **Leak scans must cover committed source content, not just `.squad/` paths and commit messages.** Two prior sweeps passed while persona names sat in C++ comments in a public upstream PR, forcing a third history rewrite. Grep the diff for agent names.
- **"Not caused by us" is not the same as "safe to mark ready."** Marking two upstream PRs ready while red — reasoning that failures were inherited or infra — was wrong. The correct posture is draft until the board is green.
- **A clean control PR is the cheapest way to separate infra from code.** PR #31985 — one line, docs-only, same `main` — reached 86/86 green while ours were red, both refuting and confirming flakiness faster than log-reading alone.
- **Apple/arm64 fork-PR jobs fail frequently at dependency download** (cpuinfo, XNNPACK, eigen3, protoc), always before compilation. `gh run rerun` refuses fork-PR jobs; only retrigger available is a push.
- **Reviewer lockout held:** Iran and Pris were barred from fixing the persona-name comments they authored; Chew did it.

## Current active wave — 2026-08-11 (Apple MLAS f16 cast — upstream PR #31993)

### Upstream ORT PR #31993 — MLAS f16↔f32 cast kernel on Apple ARM64

**By:** Luba (audit + implementation), Holden (review), Freysa (lockout revision — S1/S2)

**PR:** microsoft/onnxruntime#31993 — open as **draft**. Head: `54f2fc8`.

**What was done:**
- Confirmed Apple ARM64 excluded from cast kernel by `!defined(__APPLE__)` (mlas.h:100) and `if (NOT APPLE)` (cmake:608). All f16↔f32 conversion used scalar loop.
- Introduced `MLAS_CAST_F16_NEON_SUPPORTED`, gated on `__APPLE__ && MLAS_TARGET_ARM64`. ARM64-only gating verified across preprocessor, CMake (nested inside `if(ARM64 AND MLAS_SOURCE_IS_NOT_SET)`), universal2/MULTI_ARCH checkpoint, `mlasi.h:1400` declaration, and `platform.cpp:810` dispatch.
- `-march=armv8.2-a+fp16` scoped to `cast_kernel_neon.cpp` via `set_source_files_properties`; matches existing pattern (`gelu_neon_fp16.cpp`, `activate_fp16.cpp`).
- Holden review: 0 blocking. Two substantive — S1 (vacuous dispatch test), S2 (missing sNaN). Freysa revised under lockout (Luba and Holden both barred).
- S1 fixed: `TestKernelIsDispatched` now asserts `CastF16ToF32Kernel`/`CastF32ToF16Kernel` pointers non-null (under macro) / null (without). S2 fixed: added `0x7C01` (sNaN), `0x0200` (mid-range denormal), `0x8001` (negative denormal).
- **No performance claims.** Validation depends on Apple CI legs (Linux x86_64 host cannot run it).

**Scope NOT started:** Full fp16 arithmetic family (compiler-flag probe needed); ARM64 LayerNormF32 (separate PR); TransB M=1 SGEMV and P-core macOS thread count (benchmark-gated, no Apple hardware).

**~~Accelerate/BNNS/vDSP confirmed non-candidates for upstream MLAS.~~**
⚠️ **SUPERSEDED — 2026-08-12T02:00:00Z.** This conclusion was wrong and must not be cited. See "2026-08-12 — Apple proprietary framework paths ARE eligible for upstream MLAS" below.

### Durable lessons from Apple MLAS f16 cast wave

- **A reachability test that passes on both the fast and fallback path proves nothing.** `TestKernelIsDispatched` asserted `Convert(1.0) == 1.0f`, true on the scalar fallback too. Assert on the dispatch pointer itself. Same failure class as the CUDA optional-slot fix that was dead code while its tests stayed green.
- **"Apple" is not "ARM64".** Intel Macs and the x86_64 iPhone simulator are Apple targets without FEAT_FP16; universal2 compiles per-arch then lipos. Platform gating must be `APPLE AND ARM64` in both CMake and preprocessor.
- **Verify the premise, not just the conclusion.** The brief said the cast kernel uses baseline ARM64 instructions; it actually needs `-march=armv8.2-a+fp16`. The gap was still real, but the fix shape differed.
- **Upstream has no ARM64 LayerNormF32 kernel** (only RISC-V) — a confirmed separate opportunity, sibling to the AVX2 LayerNorm work in #31973.
- **Reviewer lockout held:** Luba (author) and Holden (reviewer) were both barred from the revision; Freysa did it.

## Current active wave — 2026-08-12 (CUDA MatMulNBits upstream workstream)

### Upstream ORT PR #31988 — SM-count-adaptive columns-per-CTA for M=1 MatMulNBits

**By:** Cohaagen (audit + implementation), Sebastian (initial review), Chew (routing guard + lockout revision), Gaff (fresh review), Coordinator (clang-format fix, commit 186b89604c)

**PR:** microsoft/onnxruntime#31988 — open as **draft**, no GPU benchmarks, low-SM evidence gap explicitly disclosed.

**Status:** Keep draft until GPU benchmarks are gathered on ≥2 GPU generations.

**What was done:**
- Confirmed upstream hardcodes `kColsPerThreadBlock = 8` in `matmul_4bits_m1_impl.cuh:135`; no SM-count adaptation exists in any GEMV path.
- `SelectColsPerBlock(n, sm_count)` → 8/4/2 templated on `cols_per_block`. Bit-identical: per-column warp reduction is invariant to CTA width.
- Guard: `n % kColsPerThreadBlock != 0 → return false` preserves upstream's exact accepted-shape set (n%8==0). Chew confirmed Sebastian's "SAFE" on the n%8≠0 path was wrong — `n=12` would have been newly accepted, changing shape routing.
- Tests: `SelectColsPerBlock_OnlyMod8Accepted` and `SelectColsPerBlock_RoutingInvariance_NMod8Required` pin the routing invariant exhaustively.
- Template cost: 24 → 72 instantiations (~38 KB), all reachable, documented in PR.

**Performance claims: NONE published.** The "+2.08% on 3 GPUs" claim from an earlier explore pass had no benchmark record, no 3-GPU record, and no low-SM data in this repo. It was kept out of the PR entirely.

**CPU AMX QNBit prefill: no PR.** This host is AMD EPYC 9V74 — no AMX, no VNNI — so the AMX-vs-VNNI comparison cannot be run. No PR without a measured win.

**Split-K: deliberately excluded.** Changes reduction association (not byte-identical); 2-way split-K regressed 7B `o_proj` GEMV by −0.59%.

**Key reviewer lockout:** Cohaagen (author) and Sebastian (reviewer) both barred from the routing-guard revision; Chew revised; Gaff reviewed fresh.

### Durable lessons from this workstream

- **Audit provenance before publishing any performance claim.** "+2.08% on 3 GPUs" had no benchmark record, no 3-GPU record and no low-SM data anywhere in this repo. It was kept out of PR #31988 entirely. If the raw data exists outside the repo it must be supplied and recorded before publication.
- **A reviewer's "SAFE" is not proof.** Sebastian cleared the `n % 8 != 0` path; the coordinator's own reading found `n = 12` would have been newly accepted, changing shape routing. Chew confirmed and guarded it. Verify reviewer conclusions on the load-bearing claim.
- **An occupancy change must not become a routing change.** Selecting 4 or 2 columns per CTA silently widened the accepted-shape set. Pin the accepted set with an exhaustive test so it cannot drift.
- **Hardware absence is a legitimate stop.** No GPU and no AMX on this host (AMD EPYC 9V74), so CUDA numbers cannot be measured and the AMX-vs-VNNI comparison cannot be run — hence no AMX PR.
- **Reviewer lockout held:** Cohaagen (author) and Sebastian (reviewer) were both barred from the revision; Chew did it; Gaff reviewed fresh.

## Current active wave — 2026-08-12 (Apple framework infrastructure + TensorRT fix)

### 2026-08-12 — Apple proprietary framework paths ARE eligible for upstream MLAS

**By:** @justinchuby (policy), Luba (PR #32001 initial), Luv (review), Isidore (lockout revision), Leon (TensorRT fix), Deckard (CI triage)

**⚠️ SUPERSEDES** the earlier conclusion (from the Apple MLAS f16 cast wave, ~2026-08-11) that "Accelerate/BNNS/vDSP are non-candidates for upstream MLAS." That exclusion was wrong and **must not be cited.** The stale line has been struck above.

**What:** Apple proprietary framework backends — Accelerate (cblas/vecLib), BNNS, vDSP — are **eligible** for upstream contribution to ONNX Runtime MLAS when:
1. Gated behind an **Apple-only, opt-in compile option** (narrowly named; default OFF).
2. Backed by a **portable MLAS fallback** as the default path.
3. **No behaviour change when the option is disabled** (provably — all side effects guarded).

**PR structure (separate PRs, never bundled):**
- **PR A (infrastructure) = #32001** (`nxrt/mlas-apple-framework-option`): `onnxruntime_USE_APPLE_ACCELERATE` CMake option, default OFF, Apple-only, `find_library(Accelerate)` linkage, `build.py --use_apple_accelerate` argument. 23 lines, 2 files. Draft.
- **PR B** (Accelerate cblas SGEMM/SDPA), **C** (BNNS), **D** (vDSP) — prepared but not started; require Apple-hardware benchmarks unavailable on this host.

**Hard constraints:**
- **Do not hand-declare private or unstable BNNS struct layouts upstream.** Public SDK headers/APIs only, or reject that candidate.
- Each kernel PR requires: portable fallback, reachability tests, numeric parity, Apple hardware benchmarks, arm64/universal2/iOS validation, fresh Opus review.

**Known blocker for B/C/D:** This host is Linux x86-64 (AMD EPYC) with no Apple hardware. Only PR A can be completed locally; it is behaviour-neutral when disabled and validates via upstream Apple CI.

### PR #32001 — review fixes (S1/S2/S3, Isidore under lockout)

**By:** Luba (author, locked out of revision), Luv (reviewer, locked out), Isidore (lockout revision)

**Fixes:**
- **S1:** Replaced `FATAL_ERROR` with `message(WARNING ...) + set(onnxruntime_USE_APPLE_ACCELERATE OFF)` on non-Apple — matches `onnxruntime_USE_SVE` / `onnxruntime_USE_KLEIDIAI` idiom.
- **S2:** Added `--use_apple_accelerate` argument to `build.py` / `build_args.py`, forwarding as `-Donnxruntime_USE_APPLE_ACCELERATE=ON`.
- **S3:** Removed `target_compile_definitions(onnxruntime_mlas PRIVATE MLAS_USE_APPLE_ACCELERATE=1)` — no consumer exists yet; avoids upstream static-analysis noise.
- Head: `d16a108252`. PR remains draft.

### PR #31988 — TensorRT build fix (OURS, not inherited)

**By:** Leon (fix), Deckard (initial diagnosis, disproved)

**Root cause:** `matmul_nbits_cols_per_block_test.cc` (host `.cc`) included `matmul_4bits_common.cuh`, which pulls `<cuda_bf16.h>` → CUB device headers. Host compiler receives ~40 `'blockIdx' was not declared` errors.

**Cross-PR evidence:** PR #31678 (unrelated) had TensorRT green; ours red — proves the break was introduced by our change. Deckard's earlier assumption that these were CUDA-13 base-codebase issues was disproved.

**Fix (Leon):** Extracted `SelectColsPerBlock`, `kColsPerThreadBlock`, and `kTargetCtasPerSm` into `matmul_4bits_cols_per_block.h` — a host-only header with no CUDA device includes. Test uses only this header; `.cuh` re-exports via `#include`. Head: `34fe91e8dd`.

### Durable lessons from this wave

- **Apple Accelerate/BNNS/vDSP ARE upstream-eligible** when gated behind an Apple-only opt-in compile option with a portable MLAS fallback. The earlier blanket exclusion does not hold once both portability objections (implicit behaviour change, non-portable) are addressed by opt-in gating.
- **Never hand-declare private or unstable BNNS struct layouts upstream.** Use public SDK headers/APIs only, or reject the candidate outright.
- **A build option nothing can set is half-finished.** `onnxruntime_USE_APPLE_ACCELERATE` existed in CMake but had no `build.py` argument; upstream contributors had no standard path to enable it.
- **Match upstream's failure idiom for mis-set platform options:** warn and disable (per `onnxruntime_USE_SVE`, `onnxruntime_USE_KLEIDIAI`), not `FATAL_ERROR`.
- **Host-only test code must not include CUDA `.cuh` headers.** `matmul_4bits_common.cuh` from a `.cc` pulled CUB via `<cuda_bf16.h>`, breaking TensorRT with ~40 device-intrinsic errors in host context.
- **Cross-PR comparison is the fastest way to settle CI failure ownership.** #31678 green vs ours red proved the TensorRT break was ours; a docs-only control PR at 86/86 green proved Apple download failures were infra.
- **Reviewer lockout held:** Luba (author) and Luv (reviewer) were both barred from #32001 revision; Isidore did it.

---

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

## 2026-08-12 — Rejection-response wave: upstream merge, B-fixes, aliasing split

**By:** Leon (#32001 B1+N1-N4), Mariette (#31988 B1-B3), Coco (#31993 NaN fix), Deckard (#32003 draft), Isidore (#32003 complete), Batty (#31988 build fix), Challenger (re-review #32001), Coordinator (harness + PR body rewrite)

### First upstream contribution merged — PR #31985

PR #31985 (`f2dfa4e9eb`, merged 2026-08-12T00:49:43Z) is the first landed contribution to `microsoft/onnxruntime`. It cleared an inherited doc-validation failure. PRs #31973 and #31974 were rebased onto it.

### PR #32001 — Apple Accelerate arm64 detection (B1 fix, Leon)

`CMAKE_OSX_ARCHITECTURES` defined-but-empty + `onnxruntime_target_platform` unset → feature silently disabled on Apple Silicon with a plain `cmake` configure. Fix: check `CMAKE_OSX_ARCHITECTURES` (single-value) for `arm64`/`arm64e` first; fall back to `CMAKE_SYSTEM_PROCESSOR` when unset or empty. Coordinator proved the regression with a 5-case standalone harness. Also: N1 Darwin-only gate, N2 `MLAS_USE_APPLE_ACCELERATE=1` reinstated as observable contract, N3 loud `BuildError` on explicit CLI opt-in (tolerant CMake), N4 CPU EP argument group. Head `0d924a421b`.

**Challenger re-review claimed `add_argument` was missing (B-NEW-1 BLOCKING).** `python3 tools/ci_build/build.py --help` exits 0 with the flag listed under "CPU Execution Provider" — false positive. Verify reviewer blockers with the same rigor as author claims. Coordinator rewrote the PR body (B2 original): it still claimed `FATAL_ERROR`, x86_64, universal2, iOS — all contradicted by the code.

### PR #31988 — MatMulNBits admission separated from launch (B1, Mariette)

Shared-memory gate scaled with `cols_per_block` (2/4), silently admitting large-K shapes that upstream declines at `cols=8` — moving them from cuBLAS fp32 to fp16 GEMV (accuracy regression). Fix: admission always uses `kColsPerThreadBlock`=8; launch uses the selected `cols_per_block`. Proven by 20,800-combination acceptance-set regression. B2: `cudaOccupancyMaxActiveBlocksPerMultiprocessor` per instantiation replaces fixed `kTargetCtasPerSm=12`. B3: forcing hook, acceptance-set regression, wide-N cols=8 coverage, GPU parity (GTEST_SKIP'd). **Parked pending GPU access.** Head `dc1e173e4b`.

### PR #31993 — Hardware quiets sNaN (NaN fix, Coco)

NEON `FCVTL`/`FCVTN` quiet signalling NaNs; MLAS software reference does not. Bit-exact comparison fails on Apple Silicon. Fix: assert `isnan` + sign + payload-modulo-quiet-bit for NaN; raw-bit equality for non-NaN. Also: RNE tie input corrected to 1 + 2⁻¹¹ (genuine tie). Removed `-march=armv8.2-a+fp16` — clang's `arm_neon.h` guards `vcvt_f32_f16`/`vcvt_f16_f32` under `#if (__ARM_FP & 2)` (AArch64 baseline), not `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC`. macOS-only gate via `TARGET_OS_OSX`. Head `02a9f34`.

### PR #32003 — Strict-aliasing split (Deckard draft, Isidore complete)

Deckard split strict-aliasing/`-Werror` fixes from #31988 into standalone draft PR #32003. Coordinator found the fix incomplete: `vec_permuted` overload fixed but 4 identical `vec_a` punning sites (lines 117–120) missed. Isidore fixed all 4 sites under lockout; justified leaving `reinterpret_cast<half2*>(sums)` (canonical CUDA vectorised-access idiom) alone. 0 member-punning `reinterpret_cast` sites remain. Head `23dcfddaaf`.

### PR #31988 build fix (Batty)

`TryMatMulNBits` gained `sm_count` parameter; `fpA_intB_gemm_kernel_test.cc` not updated (13 args vs 14). Fixed by passing `device_prop_.multiProcessorCount`. Commit `55e438ca6f`.

### Durable lessons

- **First upstream contribution merged:** microsoft/onnxruntime #31985.
- **The same bug class recurred on a fourth axis.** Compaction/scaling that silently widens an accepted-shape set has appeared as: compacted output slots, compacted input slots, `n % 8` acceptance, and shared-memory admission scaling with columns-per-CTA. **Separate admission from launch**, and pin the accepted set with an exhaustive regression test.
- **A reviewer's blocker can be a false positive.** Challenger claimed missing `add_argument` would crash the CLI; `build.py --help` exits 0 with the flag in the right group. Verify reviewer claims the same way we verify author claims.
- **Correcting a claim can itself be wrong.** An earlier round "corrected" the brief to say the f16 cast kernel needs `-march=armv8.2-a+fp16`. Clang's `arm_neon.h` guards the conversion intrinsics under `#if (__ARM_FP & 2)` (AArch64 baseline); `+fp16` is for fp16 *arithmetic*. The original brief was right and the flag was removed.
- **Hardware quiets signalling NaNs; software references may not.** `FCVTL`/`FCVTN` quiet sNaN, so bit-exact comparison against a software reference fails. Assert `isnan` + sign + payload-modulo-quiet-bit for NaN; raw-bit equality only for non-NaN.
- **Check a fix was applied everywhere.** A strict-aliasing fix landed on one overload and missed four identical sites in another; grep for the pattern, do not assume completeness.
- **Reviewer lockout held:** Luba/Isidore/Coco → Leon on #32001; Luba/Holden/Freysa/Mariette → Coco on #31993; Cohaagen/Sebastian/Chew/Deckard/Leon/Batty → Mariette on #31988; Deckard → Isidore on #32003.

### Status snapshot (2026-08-12T03:00:00Z)

| PR | Status | Head |
|----|--------|------|
| #31985 | **MERGED** `f2dfa4e9eb` | — |
| #31973 | Draft (rebased onto #31985) | — |
| #31974 | Draft (rebased onto #31985) | — |
| #31988 | Draft — **parked pending GPU** | `dc1e173e4b` |
| #31993 | Draft | `02a9f34` |
| #32001 | Draft | `0d924a421b` |
| #32003 | Draft | `23dcfddaaf` |

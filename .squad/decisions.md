# Decisions — live standing directives

Last consolidated: 2026-08-10T21:09:11Z (Scribe EP plugin export wave; 35 inbox drops merged; full narrative appended to decisions-archive/2026-08.md)

Standing governance rules and active directives. Full narrative is archived in `.squad/decisions-archive/2026-07.md`, `.squad/decisions-archive/2026-08.md`, and older `.squad/decisions/archive/` files.

This compaction preserved the complete pre-compaction live file in `.squad/decisions-archive/2026-08.md` under "Live decisions snapshot before #695/#700 compaction". Processed inbox drops archived there: cohaagen-695-hybrid-cache-fix.md, cohaagen-qmoe-route-parallel.md, copilot-contract-decisions-q2-q12.md, copilot-plugin-c-abi-everywhere.md, deckard-645-cached-dense-identity.md, harry-700-hybrid-cache-review.md, quaid-676-oracle-testfix.md.

## Ledger health rule

Archive by SIZE, not age. Age-only no-ops during high-volume campaigns because most entries are recent, so the live file can exceed spawn-budget while "older than N days" matches nothing. When over the gate, preserve full history in `.squad/decisions-archive/{YYYY-MM}.md`, dedupe rebase-reintroduced sections, and keep live `decisions.md` to standing directives plus pointers. Concurrent Scribe runs are a structural hazard; assemble from inbox drops and check `git log origin/main..HEAD` before committing.

## Current active wave — 2026-08-06

### Hybrid Mamba continuation correctness fixed (#695/#700); ORT follow-up #701

**By:** Cohaagen (fix), Harry (review), Coordinator (merge/follow-up)

**What:** PR #700 fixed issue #695 by making native host/device KV-mirror prefix reuse unsupported whenever the decoder has recurrent state. `supports_host_kv_mirror` and `supports_device_kv_mirror` now return false for `has_recurrent_state()`, forcing full recompute for hybrid Mamba/attention models instead of reusing attention KV without the recurrent state. Single-shot behavior stays byte-identical. Tests added: always-on gate unit coverage plus an env-gated GPU continuation regression where reused argmax matches the fresh oracle token `33803`.

**Review:** Harry approved #700. Minor residual: the ORT paged-reuse path has a similar hybrid guard concern (`kv_bridge.rs:407`); coordinator filed tracking issue #701.

### 35B-A3B QMoE performance and oracle follow-ups

**By:** Cohaagen, Quaid, Harry, Coordinator

- Native sparse QMoE for 35B-A3B shipped via #625/#676. The correct QMoE continuation token is `33803`; dense int4 token `5342` is the low-precision outlier.
- QMoE router parallelization (`qmoe_route`) changes top-k routing from row-serial single-thread to block-cooperative reduction with the same total-order tie rule, preserving byte-exact selected experts while raising 35B-A3B QMoE decode to about 62 tok/s (#684 context).
- Quaid's #676 follow-up fixes an unsound teacher-forced sub-assertion: reproduced serial and parallel kernels agreed, proving the failing teacher-forced argmax was a test oracle issue rather than a QMoE kernel bug.
- Deckard's cached-dense identity rule: bounded dense-weight memoization uses kernel-local immutable constant-slot identity, mmap metadata when present, one-time hashing per packed matrix/expert slice, and mutex single-flight expansion.

### 35B-A3B CUDA-graph capture fragmentation wave (#708 + follow-ups)

**By:** Cohaagen (investigation/fix), Harry (review)

**What:** CUDA-graph capture repair should target the GatedDeltaNet/runtime seams that actually fragment steady decode. PR #708 shipped the low-risk C3 fix: CUDA Split now derives split sizes from already-resolved output shapes instead of host-reading the split tensor, making GatedDeltaNet Split capture-safe with no host sync. Measured 35B-A3B QMoE hybrid decode improved from 13.415 to 12.132 ms/tok (74.5 to 82.4 tok/s), capture segments dropped 184 to 154, and token@119 stayed byte-exact at `33803`.

**Why:** Steady-state profiling corrected the initial matmul hypothesis: `matmul_nbits` was already 99.8% captured, so a workspace-pool PR would be a no-op and is deprioritized. The true residual fragmentation is in GatedDeltaNet eager islands. C2, eliding the LinearAttention trailing `synchronize()`, was rejected because it changed decode token 33803 to 46283; that barrier is load-bearing. Strict C1 (only `Dim::Static` capture eligibility) was also proven a no-op because GDN pointwise ops carry symbolic batch/sequence dims. The next real fix is executor-context symbol classification: treat batch and decode-step singleton sequence as pinned for capture, while leaving truly growing symbols capture-ineligible.

**Review:** Harry approved #708 after verifying Split sizes come only from static shape inference, cold-kernel `Unsupported`-until-warmed behavior is correct, and copy semantics are unchanged. Nits only: remove/dead-code the obsolete `None` fallback documentation and avoid redundant per-execute re-locking in a later cleanup.

### C1 pointwise capture collapse is mechanically valid but blocked by fp16 oracle lock (#722)

**By:** Cohaagen

**What:** The build-time growing-symbol classifier for C1 is preserved on `squad/elementwise-capture-seqindep` at `36cdd3aa`, but is a NO-GO for shipping until issue #722 is resolved. It mechanically collapses 35B-A3B QMoE hybrid CUDA-graph decode from 154 to 34 captured segments and improves steady decode from 12.263 to 11.763 ms/tok (+4.3% tok/s, 81.5 to 85.0). Capture==eager is byte-exact for both the 35B hybrid and pure-attention qwen2.5-0.5b.

**Why:** The 35B hybrid has a latent fp16 near-tie where baseline captured decode and eager decode diverge around token 20. Heavy accumulators are already fp32; the divergence is an inter-layer fp16 reassociation/rounding coin-flip, not a reduction bug. The current oracle lock passes only on the baseline captured path, while C1 moves the captured stream onto the eager stream and fails the lock. Issue #722 tracks the correction; recommended unblock is to re-anchor the oracle on fp32 teacher-forced adjudication, which both paths satisfy, rather than on one autoregressive captured-path landing.

### VMM KV and dense prefetch standing notes

**By:** Copilot

- Dense prefetch stays eviction-neutral: executor-driven lazy-weight prefetch is scoped to dense MatMulNBits weights, and CUDA admits prefetch only when it fits without eviction or lease growth so MoE behavior and cache-victim selection are unchanged.
- Native CUDA KV bindings may reserve the full-context VMM address range while exposing bucketed physical strides; growth commits the next bucket and repacks valid prefixes in place. Full-context strides committed one granule per head and worsened arena pressure, so floor claims must wait for #694 because ledger-refusal passes made the sweep non-monotonic.

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

**Accelerate/BNNS/vDSP confirmed non-candidates for upstream MLAS.**

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

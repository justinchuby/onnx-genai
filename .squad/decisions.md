# Decisions — live standing directives

Last consolidated: 2026-08-11T16:03:10Z (Scribe TopK-perf + 27B-native batch; 37 inbox drops merged, full narrative archived to 2026-08.md; live file ~28KB, under size gate — no compaction needed)

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

> **Update (2026-08-11):** This classifier shipped via PR #728 (APPROVED round 8) with an authoritative symbol-provenance record, central hard veto, host-seam precedence, and a restored fatal oracle tripwire. See "PR #728 C1 growing-symbol capture classifier — APPROVED" in the 2026-08-11 wave above. The note below is retained as the original diagnosis.

**By:** Cohaagen

**What:** The build-time growing-symbol classifier for C1 is preserved on `squad/elementwise-capture-seqindep` at `36cdd3aa`, but is a NO-GO for shipping until issue #722 is resolved. It mechanically collapses 35B-A3B QMoE hybrid CUDA-graph decode from 154 to 34 captured segments and improves steady decode from 12.263 to 11.763 ms/tok (+4.3% tok/s, 81.5 to 85.0). Capture==eager is byte-exact for both the 35B hybrid and pure-attention qwen2.5-0.5b.

**Why:** The 35B hybrid has a latent fp16 near-tie where baseline captured decode and eager decode diverge around token 20. Heavy accumulators are already fp32; the divergence is an inter-layer fp16 reassociation/rounding coin-flip, not a reduction bug. The current oracle lock passes only on the baseline captured path, while C1 moves the captured stream onto the eager stream and fails the lock. Issue #722 tracks the correction; recommended unblock is to re-anchor the oracle on fp32 teacher-forced adjudication, which both paths satisfy, rather than on one autoregressive captured-path landing.

### VMM KV and dense prefetch standing notes

**By:** Copilot

- Dense prefetch stays eviction-neutral: executor-driven lazy-weight prefetch is scoped to dense MatMulNBits weights, and CUDA admits prefetch only when it fits without eviction or lease growth so MoE behavior and cache-victim selection are unchanged.
- Native CUDA KV bindings may reserve the full-context VMM address range while exposing bucketed physical strides; growth commits the next bucket and repacks valid prefixes in place. Full-context strides committed one granule per head and worsened arena pressure, so floor claims must wait for #694 because ledger-refusal passes made the sweep non-monotonic.
- Async weight page-in is the default (`ONNX_GENAI_WEIGHT_OFFLOAD_ASYNC_PAGEIN=1`): fence-ordered copies stream directly from external mmap regions into reusable pinned staging instead of materializing an owned host tensor per miss (sync demand-copy retained at `=0` for A/B). Reports materialize/H2D/wait/sync counters.
- **Managed no-spill VMM is becoming the default (#755), owner-directed:** when a model exceeds the resolved device budget the runtime auto-enables weight streaming/offload rather than hard-OOMing; the explicit `serve --vram-limit` becomes an override. Hard prerequisite is **#716** (offload and CUDA-graph capture are mutually exclusive today — the pager's alloc/copy/free ops are capture-illegal, so flipping the default before #716 would silently disable capture for exactly the models this helps and give back the 154→34 collapse). Sequencing: (1) #735 strategy inference runs unconditionally and reports the plan; (2) #716 makes offload capture-compatible under stable VA slots; (3) flip default with a one-release opt-out; (4) publish interleaved same-session comparison. Must not regress: no silent spill to WDDM shared memory; capacity refusal stays a pre-header 429 (#743); public budgets stay committed physical bytes; a model that fits in VRAM must not start paging.
- **Multi-request KV batching uses VMM contiguous virtual addresses, not paged attention (#750), owner-directed:** each sequence gets its own contiguous device VA with physical granules mapped on demand, so the attention kernel keeps seeing a flat contiguous KV buffer and never walks a page table; `#721` stage-3 device-resident paged KV (`CudaPageStore`) is superseded unless this route fails. The route reduces to one question — can the decode kernel be bounded to the live sequence length instead of the padded shape? (#721 stage-4 committed 1.5 GB full-context VA vs 48 MB bucket growth, 32× worse, because the kernel read the padded shape and relied on masking.) Granules come from the #740 authority-scoped shared pool via `carve()`; commit floors are per-object, not per-token; `cuMemMap` during capture is not proven replayable; never unmap while a replay may be in flight.
- **Transactional mapped growth / shared device authority:** mapped-capacity lending is coordinated by the shared memory authority through weakly-registered `ReclaimableMappedHolder`s and RAII `MappedGrowthGrant`s (victim allowance transferred before callbacks; only `mapped_bytes - new_limit` physically reclaimed). VMM weight admission uses two authority-coordinated constraints — mapped granules consume the cache's weight allowance while newly-created handles consume global physical headroom — reserved transactionally, releasing new handles on failed transactions. `MemoryGovernor` exposes a stable `MemoryAuthorityId`; `VirtualBuffer` rejects a different governor before reserving/committing. Multi-model servers own one concurrency-safe device authority per backend/device domain and inject it into every construction path. QMoE workspace is resolved and reserved as one session-persistent peak before the admission callback (shared planning/execution layout helper), not inside execute. Managed VMM/pool construction failure is fatal before model allocation; insufficient reclaim is a typed capacity refusal → HTTP 429 with `Retry-After`, never 500/InvalidRequest.
- **LinearAttention trailing stream sync is not load-bearing for capture:** it launches one same-stream kernel that reads each state column before overwriting it, so stream ordering suffices; the sync only surfaced eager launch errors and caused the 33 `CaptureRecordingFailed` seams in #728. A capture/replay regression proved four in-place recurrent decode steps byte-identical to eager with no capture-time allocations (H200 same-model measurement still recommended before relying on the segment collapse).

### 2026-08-11: Whole-step megakernel is the next bounded batch-1 latency lever

**By:** Roper

**What:** Roper's read-only scoping concluded that a whole-step/persistent megakernel is the only remaining lever that attacks the current batch-1 decode regime's GPU-side bubbles after CUDA graph capture. vLLM full CUDA graph and llama.cpp are capture/per-op systems, not true megakernels; Mirage MPK is not directly adoptable for this Rust/ONNX/int4 QMoE stack.

**Why:** Prior FC2 fusion, ILP, and vectorized-int4 attempts show QMoE decode is occupancy/MLP/barrier bound rather than launch or bandwidth bound. Recommendation is Phase 0 only first: a persistent single-op QMoE decode kernel that removes FC1/FC3/activation/FC2/combine scratch round-trips while preserving fp32 accumulation order. Gate continuation on oracle margin `0.09375` and at least a 3% model wall-clock win.

**Detail:** Full feasibility memo archived in `.squad/decisions-archive/2026-08.md` under "Roper megakernel feasibility inbox drop".

## Current active wave — 2026-08-11 (TopK correctness/perf + 27B hybrid native)

### CPU-EP TopK: k-major output layout for non-final axes (#774, MERGED)

**By:** Coordinator (fixed inline after a sub-agent canary false-positive)

**What:** `TopKKernel::execute` (`crates/onnx-runtime-ep-cpu/src/kernels/selection.rs`) built outputs by sequential `push` in `outer→inner→rank` order, giving layout `[outer][inner][k]`. ONNX TopK keeps the input shape with `shape[axis]=k`, i.e. k-major `[outer][k][inner]`; winner rank `r` for slot `(outer,i)` must land at flat index `(outer*k + r)*inner + i`. The two layouts are equal only when `inner==1` (final axis), so the bug was latent for the common final-axis/router/argmax case but corrupted values+indices order for any non-final-axis TopK with `k>1` (dtype-independent). Fix pre-sizes outputs to `numel(shape)` and writes each element to its strided destination; removed the now-unused `sorted` field (kernel always emits sorted winners, satisfying both `sorted=1` and `sorted=0`). CUDA EP / ORT already emit k-major.

**Validation:** New discriminating tests `topk_non_final_axis_uses_k_major_layout` ([2,3,2] axis=1 k=2) and `topk_first_axis_is_k_major` ([3,2] axis=0 k=2) assert k-major values AND indices; final-axis tests unchanged. `cargo test -p onnx-runtime-ep-cpu --lib selection` 22/22; fmt+clippy clean.

### CPU-EP TopK: O(width) partial-select instead of full axis sort (#775, MERGED)

**By:** Coordinator

**What:** Partial-selection TopK now uses `select_nth_unstable_by` introselect (O(width)) instead of fully sorting each axis slice (O(w log w)), matching the sampler `top_k_threshold` technique. Selection semantics/order preserved. 24/24 selection tests pass.

### 27B hybrid GDN native CUDA enablement via io-derivation (#779, auto-merge awaiting CI)

**By:** Cohaagen

**What:** Qwen3.5/3.6-27B hybrid GDN native CUDA was blocked because the 27B artifact ships a thin `inference_metadata.yaml` with no `io` port contract → `resolve_kv_layers` returns `None` → "per-layer KV page geometry unknown". Fix: `maybe_fill_hybrid_io_from_graph` in `engine/load.rs` auto-derives the decoder `io` contract from the ONNX graph's port inventory, gated on non-empty `state_pairs` (recurrent-hybrid only). DRY, no model-name gate — unblocks the whole hybrid GDN family. Byte-exact: native argmax `11751` " Paris" == fp32 oracle, top-1 margin 2.549 nats. Locked by `qwen35_27b_hybrid_native_cuda_e2e.rs`. Cohaagen's decision memo merges via the #779 branch, not this worktree's inbox.

### GLM-4-9B and DeepSeek-V2-Lite native CUDA load unblock (#770/#771)

**By:** Cohaagen (fix), Harry (review — both APPROVE)

- **GLM-4-9B (#770):** blocker was KV admission, not an unsupported op — native reserved metadata `max_sequence_length=131072` (5 GiB) instead of the effective runtime cap. Native KV reservation now charges the actual runtime CUDA KV capacity (`cuda_kv_debug_stats().hard_max_len`, e.g. `ONNX_GENAI_CUDA_KV_MAX_LEN=4096`) before falling back to metadata. Partial-RoPE GQA already handled (rotary width derived from cos/sin cache). Native greedy loads at 93.6 tok/s; golden lock passes.
- **DeepSeek-V2-Lite QMoE (#771):** blocker was static QMoE placement rejecting `Cast(fp16 initializer→fp32)` scale inputs ("QMoE input 3 is not a graph initializer"). Placement now accepts a one-hop default-domain `Cast(initializer)` as initializer-backed for weight-region classification; runtime still receives the fp32 Cast value. Native greedy at 52.5 tok/s; golden lock passes.
- No model-name gates; both are DRY runtime/placement behavior. Harry's one hardening follow-up recommended; reviews are not a merge gate (auto-merge).

### QMoE decode fusion / megakernel experiment wave — one SHIP, four NO-SHIP

**By:** Cohaagen, Quaid

**Shipped:** Conservative small-route (`rows==1`, `routes<=16`) decode fusions on the 35B-A3B QMoE hybrid path:
- Fused FC1 gate/up + SwiGLU into one `qmoe_gate_up_activate_*` kernel (removes the `qmoe_activate` launch and `fc1_output` scratch round-trip). ~3.3% model win (11.511→11.126 ms/tok).
- One-CTA-per-output-task GEMV for small-route decode (skips the trailing `__syncthreads()` reuse barrier; fp32 K-reduction order unchanged). `qmoe_linear_f32` ~31.8→27.6 us. (#764 wave.)
- Oracle held byte-exact: teacher-forced token `33803`, margin `logprob(33803)-logprob(5342)=0.09375`.

**NO-SHIP (kept as memos, reverted code):** all failed the >3% model-level gate and/or the oracle tripwire —
- FC2/down+combine fusion: +0.08% (noise); serializes route loop, adds sync on a numerically sensitive path.
- ILP-2 K-unroll of `qmoe_linear_impl`: regressed (`qmoe_linear_f32` 27.6→32.2 us; active warps 13.2→9.7; occupancy loss > latency hidden).
- Megakernel Phase 0 (persistent single-op QMoE decode, counter-based producer/consumer): oracle byte-exact but decode **regressed +7.4%** — occupancy trap; Quaid recommends NOT greenlighting Phase 1 as specified. Note: current main (#766) already fuses FC1+FC3+activate, so the real decode chain is 4 launches / 2 scratch, not the stale "5-launch/4-scratch" spec.
- Int4 DP4A / 128-bit vectorized weight-stream: +4.90% ms/tok, QMoE op +25%; teacher-forced margin unchanged (0.09375) but autoregressive #722 tripwire drifted to token 13 (outside benign {33803,46283}).

**Standing lesson:** 35B QMoE decode is ~10–20× above the int4 bandwidth roofline (~0.1–0.24 ms/tok floor vs ~2.8 ms/tok measured); it is latency/occupancy/barrier-bound, not bandwidth- or launch-bound, so scalar ILP and simple scratch removal do not clear the noise floor. The next real lever is a designed fused/persistent decode kernel that keeps FC1 activations local while exposing enough FC2 output-feature parallelism, gated on the 35B teacher-forced oracle margin as the primary acceptance gate. Full memos in `.squad/decisions-archive/2026-08.md`.

### PR #728 C1 growing-symbol capture classifier — APPROVED (round 8)

**By:** Gaff (final revision), Harry (review), plus Roy/Sapper/Sebastian/Leon/Batty/Deckard revision rounds

**What:** Supersedes the "#722 blocked" note above for the classifier design. The build-time growing-symbol classifier that collapses 35B-A3B QMoE hybrid CUDA-graph decode 154→34 captured segments (~4% tok/s) is now correct and merged. Final shipped design after 8 review rounds:
- **Authoritative symbol-provenance/unification record.** Shape inference records every symbol unification/derived-symbol lineage at its single chokepoint and persists it on the `Graph`; the executor closes its growing set over that authoritative map (union-find equivalence closure), covering elementwise, MatMul/Einsum/Concat/Expand broadcast, and `Reshape([-1])`/`Flatten` derived expressions — with zero op-enumeration in the executor. Classifier defaults **fail-safe** (an op stays eager unless its shape lineage is provably pinned).
- **Central hard veto.** The classifier verdict is enforced as a hard capture veto in `Executor::node_capture_reason` before any kernel-specific `capture_support()`, closing bypasses from kernels that returned `Supported` unconditionally (`UnaryMathKernel`/`NotKernel`/`BitwiseNotKernel`, etc.). `capture_shape_eligible` reduced to `seq_independent` (removed the permissive `numel==1 || is_fixed_decode_shape` OR).
- **Host-seam precedence.** Veto runs AFTER the EP structural policy (`plan_capture_region`) so `If`/`Loop`/`Scan`/Sequence nodes report `HostControlFlowOrSequence`/`HostSeam`, not `EagerDeviceSeam` — preserving the public capture-segmentation contract.
- **CSA coverage:** `CompressedSparseAttention` growing `selections` on output-5 last axis is collected; generic declared `past…`/`present…` rank-4 KV I/O scan collects growing symbols without per-model gates. Rank-4 + symbolic-penultimate guard keeps fixed-capacity Mamba/GDN recurrent state capturable.
- **Oracle tripwire restored to FATAL:** autoregressive check asserts token ∈ {`33803`, `46283`} (the two benign #722 fp16-tie outcomes); any other token (5342/279/unrelated) fails CI. Final measurement 34 segments / ~83–87 tok/s, GDN pointwise stays capturable. Full round-by-round narrative archived in `.squad/decisions-archive/2026-08.md`.

### CI is asynchronous (governance)

**By:** Copilot, at owner's direction

Unless the user explicitly requests otherwise, agents and the coordinator do **not** wait for CI before continuing, reporting, or merging. Required local targeted tests, Clippy, builds, and hardware probes remain blocking; CI runs asynchronously and later failures are fixed forward. Do not poll or hold a turn open for CI, and do not keep completed worktrees solely for CI. An explicit user instruction may make a particular CI result blocking. Source — User: "现在规定所有人除非明确指令否则不要等ci。本地测试。ci看到有问题再修。可以fix forward".

### Design autonomy and parallel worktrees (governance)

**By:** Copilot, at owner's direction

The coordinator may independently make architecture/design optimizations when evidence supports them; direction-changing decisions must update durable design docs with measurement, falsifier, limitations, and rollback/override path. Prefer parallel agents in separate git worktrees when work has no overlapping files, shared mutable state, or unresolved common contract; keep changes to the same core contract or continuous call chain serial until the shared dependency lands. Source — User: "如果需要设计上的优化 你可以自行裁决 更新设计文档。" / "还有如果可以最好并行多worktree推进".

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

## Active historical pointers

For detailed per-PR narrative, use archives rather than expanding this live file. Primary locations: `.squad/decisions-archive/2026-07.md` for pre-August ledger, CUDA parity waves, Mac CPU EP/perf methodology, and July CLI/runtime records; `.squad/decisions-archive/2026-08.md` for fused LinearAttention, hetero legalization, 35B-A3B QMoE, #695/#700 hybrid cache fix, and August Scribe batches; older material remains under `.squad/decisions/archive/`.

# Decisions — live standing directives

Last consolidated: 2026-07-31T10:24:07Z (Scribe round 9 — 27B decode profile Scan=56.5%, Scan-capture scoping PENDING JUSTIN, #554 session-reuse merged; round-8 in archive)

Standing governance rules and constraints. Dated wave records and historical ledger updates
are archived to `.squad/decisions-archive/2026-07.md`.
Last compacted: 2026-08-04T00:40:00Z (Scribe #625/QMoE batch; July operational narrative compacted by size gate)

This file is the resolution of two concurrent Scribe compactions that rewrote it in the same
minutes: #427 ("consolidate CUDA parity 161 state", 04:10:00Z) and this round-2 tidy
(04:30:00Z). Both were merged; where the two sides disagreed on a single entry (one kept it
live, one archived it) the entry was kept live, because a wrongly-archived rule silently stops
governing while a wrongly-kept record only costs bytes.

Full historical ledger archived to `.squad/decisions-archive/2026-07.md`:
- "Full decisions snapshot archived by size gate — 2026-07-28T11:30:55Z"
- "Post-rebase decisions archived by size gate — 2026-07-28T11:35:49Z"
- "Narrative entries compacted by size gate — 2026-07-29T21:19:00Z" (first run)
- "Narrative entries compacted by size gate — 2026-07-29T23:30:00Z" (merge resolution)
- "Post-rebase narrative tail compacted by size gate — 2026-07-30T04:10:00Z" (#427)
- "Narrative entries compacted by size gate — 2026-07-30T04:30:00Z (Scribe tidy round 2)"
  — CUDA op-parity wave records (Kuato/Doug), native-KV benchmark record, PTY-harness
  technique notes, the reasoning-fixture review narrative, and the spent round-2 checklist.
- "Narrative tail compacted by size gate — 2026-08-02T10:05:00+0000 (Scribe fused LinearAttention batch)" — moved the July DeepSeek/QMoE/R1/Qwen/Foundry/native-state tail to `.squad/decisions-archive/2026-07.md`; processed August fused-LinearAttention drops are in `.squad/decisions-archive/2026-08.md`.
- "CUDA parity 161 live narrative compacted — 2026-08-02T19:00:00+0000" — full PR #423/#424 CUDA parity wave text remains in `.squad/decisions-archive/2026-07.md`; live file keeps the active rank/deferred-fact rule.
- "Thread-3 hetero inlining relocation — 2026-08-02T19:00:00+0000" — full Cohaagen design/scoping drop and Coordinator Phase 0+1 decision archived to `.squad/decisions-archive/2026-08.md`; compact live directive retained below.
- "Thread-3 Phase 3 fail-closed hetero scaffold — 2026-08-03T02:40:00+0000" — PR #606 merged; live directive records opt-in guard semantics, stale option-flip resolution, and pivot to 35B-A3B admission root cause.
- "35B-A3B PackedMHA admission fix — 2026-08-03T03:10:00+0000" — mobius PR #449 adds the missing PackedMultiHeadAttention bias formal; onnx-genai arity rejection was spec-correct.
- "35B-A3B validation and native-vs-ORT fairness vet — 2026-08-03T06:40:00+0000" — issue #610 uses the vetted apples-to-apples table; 35B-A3B is a capability gap on both engines; fp16 TopK enablement is in flight.
- "fp16 TopK conformance merge and GAP-3 kickoff — 2026-08-03T07:40:00+0000" — PR #612 merged test-only coverage for fp16/bf16 TopK already on main via #445; GAP-3 native pipeline decode scoping is in flight.
- "35B-A3B origin/main revalidation correction — 2026-08-03T09:00:00Z" — GAP-3/rank-3/TopK blockers were stale; native pipeline decode is correct on fresh origin/main, GPU throughput now waits on cuDNN f16/bf16 ReduceSum comp-type + device wiring; ORT still crashes.
- "35B-A3B native GPU decode unblocked — 2026-08-03T10:00:00Z" — PR #616 merged cuDNN reduce comp-type + native device wiring; 35B native GPU measured 2726 ms/tok (0.37 tok/s), correct, ORT still crashes.
- "35B-A3B Lever A reduce capture — 2026-08-03T12:30:00Z" — PR #618 merged cuDNN float ReduceSum/Mean capture eligibility; 35B native decode improved 2725→405 ms/tok (0.37→2.47 tok/s), byte-exact; Lever B next.

Older archives: `.squad/decisions/archive/`.

## Ledger health rule

Archive by SIZE, not age (age-only no-ops during high-volume campaigns — most entries are recent,
so the file exceeds 1 MB while "older than 7 days" matches nothing). When over the gate, preserve
history in `.squad/decisions-archive/{YYYY-MM}.md`, dedupe rebase-reintroduced sections, keep the
live file to standing directives + pointers. Size compaction of a shared append-only file is not
rebase-safe: concurrent appends can reinflate it without a conflict — re-run against tip if main
moved. **Concurrent Scribe runs are a structural hazard** (two runs diverged 2026-07-29); assemble
decisions.md from distinct inbox drops rather than hand-merging, and check
`git log origin/main..HEAD` before committing.

## Performance claim discipline

- A per-layer or microbenchmark speedup is not a model-level claim — confirm with Amdahl and a
  real model-level measurement. Always state exact model, dtype, metric, prompt/token regime,
  host load, and runner (TinyStories-1M and -33M ratios are not interchangeable).
- Separate measured/estimated/projected; don't compare measurements under different host load
  without labeling. PR benchmark absolute times are informational only; same-run
  PR-vs-merge-base deltas are the useful signal. Two agreeing measurements beat one confident
  outlier (retracted examples: 197 GB/s roofline, load-corrupted calibrator, 15× ORT estimate,
  1×1 Conv headline, SDPA 1.9× vs 1.37× — all in archive).
- A SIMD/accelerated path without a reachability test is equivalent to an unwired placeholder.

## Apple Silicon portability and Mac CPU EP rules

- Mac CPU EP optimizations must generalize across Apple Silicon (M1–M4, base/Pro/Max/Ultra); the
  M1 Max is a measurement rig, not the target. No compile-time constants from one machine — query
  topology/cache/features at runtime; feature-detect any path beyond the ARM baseline with a
  correct fallback.
- Reach the Apple matrix coprocessor through Accelerate (BLAS/BNNS), never hand-rolled AMX; those
  calls happen at dispatch level, not inside Rayon. The CPU EP stays one general impl shared with
  Intel/ARM; Apple specialization lives behind runtime dispatch, not a parallel kernel tree.
- BNNS `BNNSMatMul` deprecation (macOS 15) is a migration to BNNSGraph, not evidence the AMX/fp16
  path or measurements are invalid. (Full BNNS/Conv/GEMM narrative in archive.)

## Load-adaptive decode path

`ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL`: unset/`=1` → `On` (deterministic persistent SPMD pool);
`=0` → `Off` (flat); `=auto` → `Adaptive` (opt-in calibrator). Load-adaptive selection silently
changed paths under agent load and produced false verdicts, so the default is predictable and
adaptive is request-only. Expose the selected path via `decode_path_label()`/tracing.

## Dispatch-manifest inverse rule

Every claimed `(op, variant, platform) -> minimum tier` optimization needs a curated manifest
row, a `_TEST_HITS` counter, and a test proving the counter fires. Inverse is also binding: if a
fast path exists but a higher-priority guard intercepts it, the test must fail. Manifest is
CI-only, zero runtime cost; removing a row is a conscious un-claim; a claim without reachability
is a merge blocker. Historical dispatch-miss patterns catalogued in the archive. **Manifest lint
(#414):** a lint whose regex ignores rustfmt-wrapped increments is blind to them, and a lint
without a `--self-test` proves nothing when silent — wire a self-test exercising wrapped +
single-line increments plus genuine dead-counter cases.

## Minimal-build and shape-inference rules

- Graph/layout transforms gate on BOTH their infrastructure feature and the operator group
  supplying their kernels (Wave 9: NCHWc needs `mlas` + `ops-cnn`; MLAS-only must not advertise
  transforms whose CNN kernels are absent).
- Shape-inference registrations use the operator's actual ONNX domain/version, not a convenient
  family namespace (`StringNormalizer`/`TfIdfVectorizer` are `ai.onnx`, not `ai.onnx.ml`).
- Attribute-dependent output typing follows the active default/value attribute, not an unrelated
  class-list attribute (`LabelEncoder-1` mirrors `CategoryMapper`). Container propagation stays
  blocked until the tensor-only `TypeInfo` gains container representation; do not fake as tensors.

## BNNS / Conv / GEMM current guidance

- fp16 MatMul on macOS: BNNS f16→f32 reaches AMX and is the preferred compute-bound prefill/batch
  path at M≥2; M=1 decode remains a GEMV problem. Never call BNNS from inside Rayon (it uses
  system threading internally).
- 1×1 Conv: BNNS Conv can be dominated by filter creation/copy overhead; Deckard's #347 routes
  spatial-size-dependently through the real `im2col_gemm_execute` path, claims scoped by model
  measurement not microbenchmark ratios. A fitted threshold is acceptable only when labeled fitted
  and bracketed by measured data (a wrong rationale is worse than none).
- `BNNSFilterApplyBatch` is unreliable for `BNNSFilterCreateLayerConvolution` filters (SIGSEGV in
  libBNNS.dylib at batch>1); use per-image `BNNSFilterApply` until BNNSGraph migration.

## Model artifact hygiene

Fetch large external models only when needed, measure, and delete immediately — do not leave
benchmark models in `models/` or worktrees (the archived ResNet/Whisper run used
fetch-measure-delete and restored the disk baseline).

## 2026-08-02 — Thread-3 hetero inlining relocation: bounded Phase 0+1 now

**By:** Cohaagen (design), Coordinator (scope), Deckard (revision), Harry (review)

**What:** The multi-EP hetero correctness hole is latent, not on the default session path: `SessionBuilder` still selects one EP and `place_graph` does whole-session selected-EP-or-CPU fallback; `hetero.rs` exists but is not wired as the normal public session executor. PR #602 implements only the bounded safe slice: post-assignment legalization in `hetero::plan`, using a bounded fixpoint to inline kept model-local function calls when the **assigned** provider declines them with real graph metadata. Ambiguous `(domain, op_type)` function identity fails closed.

**Why:** Load-time #594 keep-as-op is correct for the single selected EP, but a future public multi-EP planner can assign a kept fused function op to a different provider. The planning invariant is now: after hetero legalization, every executable node is either non-function or supported by its assigned EP; otherwise the exact ONNX function body expansion is used. CPU-only fake-provider tests cover load-time claim vs assignment-time decline.

**Review outcome:** Harry rejected round 1 because attribute-parameterized functions could fail open when formal/call-site/ref_attr_name attributes were dropped. Deckard revised under author lockout: `function_has_attribute_parameters` now fails closed on formal attrs, body `ref_attr_name`, or call-site attrs with an actionable Phase-2 TODO; loader preserves `ModelFunction.attributes`/`has_attribute_refs` metadata before IR drops it; `ParamLeakyRelu` mutation regression proves the guard. Harry approved round 2; #602 auto-merge armed.

**Deferred:** Phase 2 first-class `FunctionLibrary` + overload-safe IR identity; Phase 3 public multi-EP session wiring and child hetero plans for If/Loop/Scan; Phase 4 capture/kernel-cache keying after legalization plus perf counters. Full design/scoping record is archived in `.squad/decisions-archive/2026-08.md`.

## 2026-08-03 — Thread-3 Phase 3 scaffold and option-flip resolution

**By:** Cohaagen (PR #606), Harry (review), Coordinator (scope/default decision)

**What:** PR #606 merged the first Phase-3 increment without pretending stateful per-op hetero execution exists. Cross-EP tensor movement is currently only in standalone `hetero::execute` (host-staged), not integrated with the stateful session `Executor` that owns KV, capture, and decode memoization. The shipped scaffold adds `classify_placement`, `placement_summary`, and `guard_heterogeneous_fallback` in `hetero.rs`, wired into `place_graph`'s CUDA-fallback branch behind opt-in `ONNX_GENAI_HETERO` (default OFF), plus `SessionError::HeterogeneousExecutionUnsupported` and C API mapping. With the flag off, the single-EP path remains byte-identical; with a genuine mixed split under the flag, the guard returns an actionable error naming fallback ops and never silently executes wrong bytes.

**Review outcome:** Harry approved after verifying flag-off byte identity, fail-closed mixed-plan behavior, that `hetero::execute` cannot be reached from the stateful executor, mutation proof (`Heterogeneous` arm returning `Ok` makes `guard_enabled_mixed_fails_closed` fail), and clean fmt/clippy/C API checks.

**Coordinator decisions:**
- The pending "default-flip `ONNX_GENAI_DECODE_INLINE_SCAN` -> ON" item is stale: #592 removed that engine flag and made decode-inline automatic/graph-property-gated via `DecodeInlineState`, `route_decode_inline_decision`, and `maybe_enable_decode_inline`. The perf default is already ON when an inlineable single-trip Scan exists.
- Justin authorized flipping "that option" on, but the only residual default-off knob is `ONNX_GENAI_SCAN_INLINE_SINGLE_TRIP` in session `control_flow.rs` `exec_scan` Slice-1a. It is host-only, no-capture, byte-exact for a one-iteration loop, gives approximately zero perf gain, and is superseded by the automatic engine path. Decision: do **not** flip it; leave the on/off regression undisturbed and consider dead-flag removal later.
- Strategic pivot: integrated per-op Phase-3 execution has no current consuming model because native CUDA runs target models fully and whole-session fallback does not trip. Defer integrated execution to issue #603 and pivot to concrete model-support value: Qwen3.6-35B-A3B `vision_encoder` `PackedMultiHeadAttention` 6-vs-5 admission bug (Cohaagen-21 root cause in flight).

**Deferred:** #603 tracks correct attr-binding/FunctionLibrary, public multi-EP session wiring, and capture re-keying.

## 2026-08-03 — 35B-A3B PackedMHA admission fix belongs in mobius

**By:** Cohaagen (root cause), Deckard (mobius PR #449), Harry (review)

**What:** The Qwen3.6/Qwen3.5 35B-A3B `vision_encoder` admission failure is a mobius export bug, not an onnx-genai loader bug. ORT `com.microsoft::PackedMultiHeadAttention` has 6–7 positional inputs: `query, key, value, bias, token_offset, cumulative_sequence_length, attention_bias`. Because `token_offset` and `cumulative_sequence_length` occupy slots 5/6, the optional `bias` slot 4 must still exist as a formal, even when absent at call sites. mobius call sites correctly emitted 6 inputs with `None` in slot 4, but its model-local fallback function declared only 5 formals by dropping `bias`. onnx-genai `function_inline.rs` correctly rejected `6 > 5`; do not loosen that arity check.

**Fix:** Deckard authored mobius PR #449 (`squad/packed-mha-bias-slot`), adding `bias` as the 4th formal in `functions/packed_multi_head_attention.py`, keeping it inert in the fallback body, updating docs, and adding `tests/packed_multi_head_attention_function_test.py` to lock positional order and 6-input admissibility. mobius validation: ruff clean, 3/3 new tests pass, 30/30 `ep_optimization_test` regression pass. Justin merges mobius PRs; do not self-merge.

**Review outcome:** Harry approved #449 after verifying positional wiring, inert bias behavior, all three call sites emit `None` in slot 4, and mutation proof (moving `bias` formal to the end makes the new test fail). No onnx-genai product changes were needed.

## 2026-08-03 — 35B-A3B validation and native-vs-ORT fairness rule

**By:** Cohaagen (validation/fairness vet), Coordinator (issue #610)

**What:** Qwen3.6-35B-A3B is a capability gap on both engines today, not a throughput comparison. ORT-CUDA hard-crashes during graph optimization with the same `std::vector::operator[]` assertion seen on 27B. Native-CUDA loads farther but cannot produce a GPU decode number with current tooling because the dense_fallback MoE router has 40 fp16 `TopK` nodes with no CUDA kernel, the hybrid needs rank-3 mRoPE positions, and native pipeline decode is still unimplemented.

**Fairness rule:** native-vs-ORT performance claims must compare the same artifact/quantization/accuracy level under steady-state methodology and oracle-correct output. If one engine crashes, rejects the graph, or falls back to CPU/different kernels, report a capability gap rather than a multiplier. Issue #610 records the vetted two-section table: legitimate apples-to-apples wins are Qwen2.5-0.5B 1.75×, Qwen2.5-1.5B 1.66× (native oracle-correct, PR #597), Phi-4-mini 1.35×, Qwen2.5-7B 1.14×, and DeepSeek-R1-1.5B ~1.70× with a quick-profile caveat. Capability gaps include 27B (ORT crashes; 3.5× was native-vs-native-baseline), DeepSeek-V2-Lite QMoE (ORT CUDA lacks QMoE and CPU-fallbacks), GLM-4-9B (ORT rejects graph), and 35B-A3B (both blocked).

**Next work:** Cohaagen-23 is authoring fp16 `TopK` CUDA support (`squad/cuda-fp16-topk`) to unblock dense_fallback MoE routers from whole-session CPU fallback; leave the in-flight inbox drop for the next batch.

## 2026-08-03 — fp16 TopK conformance coverage and GAP-3 kickoff

**By:** Cohaagen (PR #612), Harry (review), Coordinator (merge)

**What:** fp16/bf16 CUDA `TopK` support was already on main via PR #445 (`d2333664`), so Cohaagen did not duplicate the kernel. PR #612 merged test-only conformance coverage required by the parity convention: fp16 router-shape `[2,256]`, `k=8` byte-exact GPU==CPU coverage, a non-final-axis test, and an EP-claim test in `ep-cuda` indexing tests. Validation: 15/15 GPU tests passed, CUDA clippy `-D warnings` clean, fmt clean.

**Review outcome:** Harry approved after verifying the kernel writes back the original raw fp16 element while only upcasting for total-order compare, mutation-proofing tie-break behavior, and confirming the CUDA k-major non-final-axis order is ONNX-spec-correct. The review also identified a latent CPU EP non-final-axis TopK ordering bug (CPU push-order outer,inner,k vs CUDA/ONNX k-major outer,k,inner); treat it as a separate low-priority follow-up, not a #612 blocker.

**Next work:** GAP-3 native pipeline decode scoping is in flight (Cohaagen-24): replace ORT-specific `DecodeState` / `PipelineDecodeLoopBackend` ownership with backend-neutral tensors/component sessions, then invoke the native target step. Qwen3.6-35B-A3B is the consuming model.

## 2026-08-03 — 35B-A3B origin/main revalidation correction

**By:** Cohaagen-24/25; Coordinator updated issue #610

**What changed:** The earlier 35B-A3B native blockers were stale-build artifacts from a local main 74 commits behind. On fresh `origin/main` @ `0a5ac3c5`, GAP-3 native pipeline decode was already landed (#479/#565 family), rank-3 mRoPE support was landed (#543), and fp16 TopK/Softmax execute on the CUDA EP (#612/#445). Full Qwen3.6-35B-A3B native pipeline decode runs end-to-end and is output-correct against the native-CPU oracle.

**Current truth:** Native GPU throughput is blocked by one new CUDA EP op bug: cuDNN `reduce_t` sets the reduction compute type to the half/bf16 I/O dtype, causing `CUDNN_STATUS_NOT_SUPPORTED` on MoE-router fp16 `ReduceSum`; it must use `CUDNN_DATA_FLOAT` compute for half/bf16 I/O. This is also a claim/runtime mismatch because CUDA claims f16/bf16 `ReduceSum`. Secondary bug: native pipeline decode ignores `--ep cuda` unless `ONNX_GENAI_PIPELINE_NATIVE_DECODER_DEVICE` is set, so benchmarks can silently run the decoder on CPU. ORT-CUDA still hard-crashes on the artifact with the same `std::vector<NodeArg*>` optimizer assertion, so 35B remains an ORT capability gap and any native GPU tok/s will be a standalone native number, not a native-vs-ORT ratio.

**Actions:** PR #613 merged as docs truth-up/design note after Cohaagen-24 found GAP-3 already landed. Cohaagen-26 is in flight to fix the cuDNN comp-type and device-wiring bugs and then measure real 35B native GPU throughput; do not log its outcome yet. Issue #610 received interim and final correction comments.

## 2026-08-03 — 35B-A3B native GPU decode unblocked by PR #616

**By:** Cohaagen (PR #616), Harry (review), Coordinator (merge/#610 update)

**What:** PR #616 merged both fixes from the 35B revalidation: f16 CUDA reductions now use raw cuDNN FFI with `CUDNN_DATA_FLOAT` compute type while keeping f16 descriptors and f32 alpha/beta; bf16 reductions route to the f32-accumulating NVRTC block kernel because cuDNN rejects bf16 reductions even with f32 compute. The EP claim gate now matches runtime for f32/f16/bf16 Sum/Mean and a latent no-op stub was removed. `build_native_pipeline_decoder` now honors `config.native_device` when `ONNX_GENAI_PIPELINE_NATIVE_DECODER_DEVICE` is unset; the env override still wins.

**Review outcome:** Harry approved after mutation-verifying the cuDNN compute-type failure and device-preference fallback, checking FFI soundness (error-checked calls, RAII descriptors, sized workspace, null indices for no-indices, serialized handle), confirming f32 byte-identity and bf16 NVRTC f32 accumulation, and rerunning fmt/clippy plus GPU/engine/EP reduce suites.

**Measured result:** Qwen3.6-35B-A3B int4 native GPU decode on H200 now runs correctly: steady median 2726 ms/tok (0.37 tok/s), prefill about 18.5s, greedy output matches the native-CPU oracle prefix, and the run is GPU-confirmed (21.5GB, 19–31% utilization). This is a standalone native capability number, not a native-vs-ORT ratio: ORT-CUDA still hard-crashes on the artifact. The remaining performance issue is host/sync overhead / low GPU utilization, a follow-up optimization track rather than a correctness unblock.

## 2026-08-03 — 35B-A3B Lever A reduce capture lands

**By:** Cohaagen (profile + PR #618), Harry (review), Coordinator (merge/#610 update)

**Profile finding:** 35B-A3B native CUDA decode after #616 was host/sync-bound at ~2725 ms/tok with CUDA graph capture shredded into 10424 segments / 10423 eager seams. The root cause was 10240 fp16 `ReduceSum` seams (256 dense_fallback MoE experts × 40 layers): the cuDNN float reduce path allocated workspace per call, synchronized unconditionally, and never marked the call capture-safe.

**Lever A shipped:** PR #618 made cuDNN float `ReduceSum`/`ReduceMean` capture-eligible by caching descriptors/workspace, rejecting cache misses during capture, gating sync on `!capturing`, reusing warmed axes metadata, and setting capture-safe only after a shape-stable warm call. Numerics are unchanged: f32 remains byte-identical, f16 keeps the #616 f32-comp cuDNN path, and bf16 stays on NVRTC. Harry approved after mutation-verifying cache-key shape coverage and sync gating, checking warm/shape-change behavior, and rerunning fmt/clippy plus 6/6 capture and 3/3 parity GPU tests.

**Measured result:** Qwen3.6-35B-A3B native CUDA decode improved from 2725 ms/tok / 0.37 tok/s to 405 ms/tok / 2.47 tok/s (~6.7×), byte-exact vs CPU oracle. Captured graph fragmentation dropped from 10424 segments / 10423 seams to 184 / 183, and fp16 `ReduceSum` seams dropped 10240 → 0. Remaining follow-up is Lever B: RMSNorm `ReduceSumSquare`, Split host-sync, and LinearAttention capture-abort seams in the linear-attn hybrid path.


## 2026-08-04 — Loader/QMoE current state (#621/#625 and Mobius queue)

**By:** Scribe, from processed inbox drops and spawn manifest at 2026-08-04T00:40:00Z

Full processed inbox drops for this batch are archived in `.squad/decisions-archive/2026-08.md`; merged and deleted drop files: cohaagen-35b-configC-ortgenai.md, cohaagen-35b-qmoe-measure.md, cohaagen-leverB-nvrtc-reduce-capture.md, cohaagen-native-loader-fix.md, cohaagen-qmoe-sparse-decode-design.md, deckard-446-451-444-review.md, deckard-447-450-review.md, deckard-glm404-review.md, deckard-qwen35-qmoe-export.md, harry-625-loader-rereview.md, harry-625-loader-review.md.

- PR #621 (Lever B) merged: NVRTC `ReduceSumSquare`/RMSNorm capture work reduces 35B native CUDA graph seams beyond Lever A.
- PR #625 native loader fix: GraphIo / GraphIoMetadata lets native loading bypass ORT Session creation. Harry rejected rev1 for an initializer-input leak; Quaid revised under Cohaagen lockout with initializer exclusion mirroring `graph_builder.rs` and metadata/Session KV-geometry parity coverage, after which Harry approved. HEAD `3b615953`; auto-merge enabled/mergeable.
- ORT 1.28 still rejects the fp16-activation + fp32-scale QMoE artifact, so config-B remains an ORT capability gap rather than a loader bug.
- 35B QMoE sparse graph rewrite/export is done with 40 QMoE nodes. Config-C ORT-GenAI validation used dense_fallback (0 QMoE) and is not a sparse-QMoE proof.
- Seven Mobius PRs (#446, #447, #450, #451, #444, plus GLM/Qwen export reviews) are review-resolved and await Justin merge.
- Config-A GPU measurement is blocked by external vLLM GPU occupancy; Cohaagen left watcher PID 1060559 for the measurement.

## Active historical pointers

For per-PR narrative use `.squad/decisions-archive/2026-07.md`. Archived there: consolidation
checkpoints (2026-07-28 size-gate snapshots; 07-29 compactions; rounds 2–7 CUDA/native/MoE +
native-pipeline + CUDA-hybrid wave records; prior `.squad/decisions/archive/`); Mac CPU EP topics
(#227 roofline, load-adaptive opt-in, Apple Silicon portability, BNNS prefill/deprecation,
benchmark-CI rule, dispatch-manifest lint, 1×1 Conv + SDPA corrections, GEMV notes); Wave 8/9
(CUDA coverage batches 8/9, shape-inference catalog batches 3/4, NCHWc gating, reviewer-lockout).
For detailed per-PR narrative, use the archives rather than expanding this live file. Primary locations: `.squad/decisions-archive/2026-07.md` for the pre-August ledger, CUDA parity waves, Mac CPU EP/perf methodology, and July CLI/runtime records; `.squad/decisions-archive/2026-08.md` for fused LinearAttention, Thread-3 hetero legalization, and August Scribe batches; older material remains under `.squad/decisions/archive/`.

## CUDA standard-domain parity — current state (through 161 ops)

Full narrative for PRs #423/#424 and the 161-op parity wave is archived in `.squad/decisions-archive/2026-07.md` under the 2026-07-30 compaction entries. Live rule: CUDA shape/rank claim gates must distinguish unsupported static shapes from deferred rank facts; deferred facts preserve CPU fallback instead of falsely declining the graph. Current tractable parity is 161 ops; remaining heavy gaps include NonMaxSuppression and Resize-cubic.

## CLI charter — standing directives

**By:** Justin Chu (2026-07-27). Live policy (restated from archive).

- **The CLI is a developer/maintainer tool, not a consumer product.** Rank CLI work by *does this
  shorten a maintainer's debug/iterate loop or expose otherwise-unobservable engine behavior?* —
  not *does a competitor have it?* **Explicitly rejected, do not re-propose:** remote-client mode
  against an OpenAI-compatible server; model registry/pull/consumer lifecycle;
  conversion/quantization/fine-tune loops as CLI features. See `docs/research/cli/00-backlog.md`.
- **The REPL is the primary CLI investment.** Target bar: Copilot CLI's interactive shell with one
  deliberate divergence — **ratatui inline viewport, not full-screen alternate screen** (native
  scrollback + terminal copy; `docs/research/cli/05-repl-redesign.md` §2). Phase 1 landed (#289);
  `/fork`/`/rewind` depend on runtime APIs (`04-runtime-capability-inventory.md`,
  `06-fork-rewind-api.md`); fork is type-gated and not yet enabled on any backend.

## CI: run tests on every platform; instrument for coverage only where informative

**By:** Pris (2026-07-28). Full coverage required on PRs; a parallel uninstrumented Linux fast job
(5–9 min) gives early feedback but never substitutes for the full gate. Windows ARM64 keeps
tests/clippy but not llvm-cov. Platform execution is the signal; instrumentation is the cost.
Critical path: `CLI ORT (Windows x86_64)` ~18m50s.

## Standing durable rules — 2026-07-29 wave (distilled; full narrative in archive)

- **Native multi-turn perf uses the session-persistent KV API** (Pris #408), not the stateless
  path, unless explicitly `--native-stateless`.
- **A step that warns instead of failing is not verification** (Holden #401): check HTTP status
  explicitly (`curl -f`/`-w %{http_code}`); validate archive magic bytes before extracting.
- **Model-declared generation defaults are canonical** (Leon #385/#392): precedence explicit
  caller flag > model-declared > greedy; enforced in the engine (CLI/server/Python inherit).
- **Worktree lifecycle** (Justin): never delete a worktree before Scribe merges its decision inbox
  (inbox is git-tracked, so drops survive deletion).
- **Warmup uses a shared registry method** (Lull+Rachael #407): `ModelRegistry::warmup` for the
  per-model setting and `POST /v1/admin/models/{id}/warm`; typed errors 404/500/500.
- CLI/terminal rules (Rachael #372, Zhora #393, Leon #395) — full detail in archive: probe the
  stream you write to (stats→stderr ⇒ test `stderr().is_terminal()`); run the exact CI gate
  (`cargo fmt --all --check`); terminal behaviour needs PTY-driven tests (piped-stdio can't cover
  control sequences); ConPTY type-ahead loss during generation is not a backend bug; the CUDA
  driver API ships with the display driver (`nvcuda.dll`), not the toolkit.

### All inference/pipeline metadata must be explicit; name guessing is forbidden
**By:** Justin Chu directive #377; Cohaagen/Benny/Melina/Matthias (PRs #380/#382/#377/#412)

ALL inference/pipeline metadata except io-SHAPE must be EXPLICIT and GENERAL. Replace name
guessing/historical-name fallback with explicit metadata plus a clear ERROR naming the
missing key. Only io-SHAPE may disambiguate. Do not re-propose deferral.

**Active schema fields (emit these names verbatim):**
- `pipeline.strategy.inner_embedding_output: Option<String>` — nested-AR inner decoder embedding output port; absent ⇒ ERROR.
- `model.io.static_cache: Option<StaticCacheIoSpec>` — `write_indices_input`, `kv_sequence_length_input`, per-layer `key/value_cache_inputs/outputs` (equal-length, positional); inconsistent ⇒ ERROR. Must be declared; convention-based binding removed (#412) — a TensorScatter static-cache graph without the block fails closed. `StaticCacheAbi::classify` stays name-agnostic.
- Encoder prompt-input role from `model.encoder.inputs.audio_features` vs `.input_ids` (no port-name matching); paged-KV geometry from `model.io.kv_inputs`/`kv_outputs` only (no metadata ⇒ `Ok(None)`). Off-limits: `decode_contract.rs` `KvNamingConvention` is only for #99 speculative proposers.

## Testing discipline — standing rules (from reasoning-fixture review, #410/#411)

- **Assert on what the code did, not a summary of what it should do** — a test keying on a
  display/summary line stays green while the real path (`resolve_sampling_defaults`) is broken;
  surface the resolved policy into `--stats`/`--profile` and assert there.
- **Run a new test in isolation before believing it** — a single green in a full parallel suite
  can be a stderr-interleave artifact. **A fixture whose every assertion is "the turn was
  dropped" cannot distinguish correct behaviour from total breakage** — make the success path
  reachable. **A near-deterministic fixture cannot witness sampling** — assert on the resolved
  policy object, not the token stream.
- **One policy resolved at two sites is the defect** — resolve once via a shared helper both
  paths call, reading the live backend on demand (no staleness across `/reload`/`/ep`/`/backend`).

## CUDA EP op-coverage scope — standing directive

**By:** Cohaagen (issue #67; #480/#484/#525). Data-driven placement audit (production loader +
per-node `supports_op`, recursing subgraph bodies) over the real decode models.

- **Classic transformer decode is 100% covered on CUDA** (qwen2.5-0.5b/1.5b/7b, Phi-4-mini,
  Qwen3.6-27B, Qwen3.5-35B-A3B int4): every covered-type node places, zero fallbacks. **Control
  flow (`If`/`Loop`/`Scan`) is executor-handled recursively and MUST NOT be added to the CUDA EP**
  (subgraph bodies already place on CUDA; not EP ops). Do not re-propose.
- **Qwen3.5 hybrid (Mamba + linear-attention) family is fully CUDA-covered:**
  `CausalConvWithState` (#480), `LinearAttention`/Gated DeltaNet (#484: per-thread
  f32-register-column state, placement 0→18/18/24), com.microsoft RotaryEmbedding + Bool NonZero
  (#525). `GatherBlockQuantized` covered (#480); #525 added a LOUD fail-closed gate for GBQ
  `bits=4` odd-blocks-per-row and fixed a RoPE dtype-check bug (Int64 position_ids vs float).
- **Numerics rule for these hybrid kernels:** accumulate in f32 (matching the ORT/CPU EP oracle);
  widen f16/bf16 on read, narrow on write ⇒ dtype-invariant (RULES.md §2); the claim gate must
  reject configs the kernel cannot run (e.g. `d_k > 256`). Full design archived.
- **#529/#535/#543:** qwen3.5-0.8b hybrid places 100% on CUDA (1289 nodes, 0 declines;
  `qwen35_0_8b_placement_lock`) AND now decodes e2e (#535 loader synthesis; #543 rank-3 native
  positions + ep-cuda `Range` `[1]`-scalar relaxation — the mrope `k_mrope/range/Range` gap).
  Native-CUDA hybrid decode == ORT token-for-token on real weights. **Lesson: 100% placement is
  not execution** — a covered op can still reject a real graph's tensor shape.

## Native multi-component pipeline decoder seam — standing directive

**By:** Mary (issue #384). The pipeline decode loop is backend-agnostic via a **stateful** seam
(distinct from Inc1's stateless `ComponentSession`). Per-increment narrative in the archive.

- **`trait PipelineDecoderComponent`** drives the decoder: `step(input_tokens, past_len, extras)`
  advances internal KV and **retains outputs internally**; the loop never touches ORT `Value`/nxrt
  tensors (`PipelineDecodeLoopBackend` holds one `Box<dyn PipelineDecoderComponent>`).
- **Do NOT drive a stateful decoder through a stateless host seam** — it drops native device-KV
  continuity and re-stages the whole KV cache across the host boundary every step; KV must stay
  device-resident. Impls: `OrtPipelineDecoder` (host KV, #478); `NativePipelineDecoder`
  (device-resident KV, #479; CUDA `inputs_embeds` #485, generic routed CUDA ports #487).
- **MILESTONE:** the native pipeline CUDA decode path is fully on main; real qwen3-0.6b
  native-CUDA e2e matches ORT-CUDA for 32 tokens (mask/ReduceSum #487 is an ARTIFACT, not a
  blocker). **Inc3c (#533) native CUDA decode BEATS ORT:** default-off
  `ONNX_GENAI_NATIVE_DECODER_CAPTURE_STEP_INPUTS` writes a persistent `[1,1,width]` device binding
  per routed port each step and reuses captured `run_one_token` (mask frozen, KV device-resident)
  ⇒ 1.38–1.42× ORT-CUDA on real qwen3-0.6b (counter `NATIVE_DECODER_CAPTURED_STEP_INPUT_DECODES`
  OFF=0/ON=3, tokens byte-identical).
- **LANDMARK — rank-3 mrope native positions (#543):** native-CUDA hybrid decode == ORT
  token-for-token (16 tokens) on real qwen3.5-0.8b — first real-weights `inputs_embeds`
  split-package native == ORT proof. DRY: shared `decode::position_ids_from_starts(starts,
  input_len)` factored from ORT `build_position_step`, called by BOTH drivers (ORT byte-identical).
  Coordinate rank from the declared `position_ids` shape via `declared_position_rank` (rank 2 → 1
  legacy `[1,S]`; rank 3 → static leading dim; symbolic → loud error) — **no hardcode-to-3, no
  model-name gate**; stored once on `NativeDecodeSession`+`DecodeCudaState`.
- **Text-only decode pipeline synthesis (#535)** unblocks a split VLM package whose image
  preprocessing is unrepresentable (`smart_resize`): new `GenAiConfigError::
  UnrepresentablePreprocessing` (distinct from `IncompletePipeline`) → `to_strict_text_only_
  pipeline_metadata` synthesizes an embedding→decoder AR pipeline with NO vision component
  (positions rank-3 `linear_increment`, decoder `inputs_embeds`). Modality-driven, NOT a
  model-name case. Also resolves the symbolic leading (batch) axis in `decode/values.rs`.
- **Capture-step-inputs flag is a MULTI-COMPONENT `inputs_embeds`/routed property (#541).** It
  cannot engage on single-component `input_ids` models (qwen3-0.6b loads via `Engine::from_dir`,
  counter stays 0 — `qwen3_0_6b_capture_step_inputs_decline`; its 614/206/433 tok/s beats-ORT-1.42×
  is the token-id CUDA-graph lever, not this flag). Keep **default-off** until a real-weights
  `inputs_embeds` model (qwen3.5-hybrid, gemma-3n) runs it e2e; mechanically safe to default-on.

## Shape-inference sequence/container ops — standing directive

**By:** Harry (issue #449, CLOSED at #531). Container-type shape inference is COMPLETE: additive
`ValueType{Tensor|Sequence|Optional|Map}` (foundation #477; seq ops + seq↔tensor conversion #486;
If/Loop/Scan/SequenceMap threading + cross-subgraph capture #527/#531), byte-identical tensor path
guaranteed. Catalog 217 ops/262 entries. Deferred (no in-tree demand): Optional/Map handlers,
IR-persistence of `ValueType`.

## ORT cached-value cloning — standing directive

**By:** Harry (#540, requested by Justin). Cloning an ORT cached `Value` covers **all POD dtypes**
via one dtype-agnostic raw-bytes fallback; do not re-add per-dtype bail arms.
`decode/values.rs::clone_value` and `onnx-genai-ort::value.rs::clone_owned` terminal arms use
`Value::from_raw_bytes(value.as_raw_bytes()?.to_vec(), shape, dtype)` (typed f32/f16/bf16/i64 fast
paths kept). Use `as_raw_bytes()` (host-guarded — precise `InvalidArgument` on a device tensor),
NEVER `to_raw_bytes()`. Unblocked the gemma-3n Bool audio mask.

## CUDA live weight offload (#63) — standing directive

**By:** Cohaagen (#444 first increment; #87 plan; #82 routed-expert deferred).

- Live CUDA weight paging is wired into the decode hot-path but **gated behind
  `ONNX_GENAI_WEIGHT_OFFLOAD=1`**; default-off returns `stock()` capabilities → byte-identical.
  Lazy weights resolve to a device pointer in the **dispatch layer** via kernel-agnostic EP trait
  `page_lazy_weight` (default `Ok(None)`), so the large CUDA kernels stay untouched.
  `CudaWeightResidency` is a bounded-VRAM (`..._DEVICE_BYTES`) LRU; eviction is strong-count-safe
  and `admit()` syncs the compute stream first (no use-after-free; skipped under graph capture).
  `LazyWeightBoundary` matches `com.microsoft::QMoE` + `MatMulNBits`. Token-identical on qwen3-0.6b
  int4 (~1.21× slowdown at a 2 MiB budget).
- **#87 async prefetch is PLAN-ONLY (awaiting Justin green-light).** Mechanism (copy stream
  `htod_async`, fences, `plan/drive_double_buffer`) already shipped & GPU-tested; gap is
  synchronous inline page-in. Inc1 = async page-in + fence-ordered consume (no extra VRAM); Inc2 =
  double-buffer look-ahead. `cp.async` is NOT applicable. Only a win when transfer-bound. Full plan
  in archive.
- **Guardrail — o_proj 2-way split-K (K_SPLIT=2) REGRESSES the 7B o_proj GEMV (−0.59%,
  repeatable). Do NOT re-try that lever** (reduction tax > sub-wave grid-fill); a K_SPLIT>2 new
  kernel with its own A/B is the future candidate.

## 2026-07-31 — 27B decode profile + Scan-capture scoping (round 9)

**By:** Scribe. Round-8 (#544/#552/#554/27B-A/B/GQA) → archive.

- **27B native-CUDA decode: Scan is bottleneck** (Cohaagen; profile-only): 168 ms/tok (~35× off roofline). `Scan` (48 LinearAttention blocks/step) = **56.5%**, structurally un-capturable. MatMulNBits at roofline (4.4 ms, 2%). Ceiling: **~15–30× speedup** if Scan enters capture/fuse. NOT a kernel fix.
- **#554 MERGED** (Mary; Harry APPROVED): `DecodeCudaState.rewind(0)` re-zeroes `fixed_state_binding_range`; pure-KV models unaffected (empty range).

## ⚑ PENDING JUSTIN: 27B Scan→CUDA-capture (Mary; no code changed, awaiting go-ahead)

Structurally larger than an increment — blockers: (1) shared prefill+decode plan; seq=1 inline corrupts prefill. (2) Control-flow declined at `provider.rs:458`, no trip_count exemption. (3) Child bodies never fold into parent plan.

**Approach 1 — runtime dual-path** (only correct+feasible path):
- **1a** (flag-gated): inline body into parent plan alongside Scan; runtime trip_count==1 picks body. Correctness-only, no capture.
- **1b**: body enters capture; validate captures/replays rise; assert 27B tokens byte-identical to locked reference.
- Blast radius: #443/#543 core. Prefill MUST be validated (shared-plan is the correctness tripwire).

**Baseline + locked reference tokens ready.** Awaiting Justin go-ahead.
In flight: #87 inc2 double-buffer; native paged-KV; 35B-A3B MoE; gemma-3n text-only.

---

## 2026-07-31: GAP-3 increments (#565–#568) — native paged-decode pipeline (MERGED)

Last consolidated: 2026-07-31T23-33-51Z (Scribe round 10 — GAP-3 Inc-A/C/D/D.1 merged; decisions inbox consolidated).

### 2026-07-31: GAP-3 Inc-A — native multi-component pipeline decode construction (#565 MERGED)

**Author:** Cohaagen · **Status:** implemented, tests green, MERGED.

Unblocked pure-native pipeline selection: route `PipelineBackend::Native` into the already-working flat-autoregressive decode path. Shared builder converges native-selection sources (ORT env-flag hybrid path + new backend selection) on ONE decision point `NativeComponentSelection` + identical builders. Construction no longer early-bails on native; paged is gated behind `supports_paged_kv()` (Inc-C+D). Non-paged flat-AR path unchanged. Token-exact parity: pure-native == hybrid-env == ORT on `tiny-gemma4-vlm` (CPU) and GQA on `tiny-gqa-embeds-cuda` (CUDA).

Regressions all green: native-backend decoder parity + native CUDA captured-step-inputs + session-reuse + hybrid e2e + 343 lib unit tests.

### 2026-07-31: GAP-3 Inc-C — native present-KV mirroring + paged native decode (#566 MERGED)

**Author:** Cohaagen · **Status:** implemented, tests green, MERGED.

Closed the S2 bail (`mirror_last_present_kv`). Scope: **host-resident f32 rank-4 KV only**; device-resident and in-place-GQA KV stay non-paged (Inc-D/D.1).

Wired: `NativeDecodeSession` reads present-KV from growable host tensors via `host_present_kv()` + seeds materialized prefix via `seed_growable_kv()`; both driven through shared `extract_present_token`/`append_token_kv` geometry (byte-identical to ORT paged KV). Paged gate: `supports_paged_kv()` = `host_kv && rank4 && f32`. DRY: shared claim/materialize logic in `claim_paged_prefix`; native + ORT feed the same geometry.

Test: `native_paged_prefix_reuse_matches_fresh_and_ort` on `tiny-gemma4-vlm` (page_size=2): warm native == cold pure-native == ORT oracle `[7,0,5]`; reuse engaged (prefix reused > 0); byte-identical pages to ORT. Non-vacuity: (a) gate revert fails reuse, (b) geometry swap (KV transpose) fails byte-assert, (c) no-op mirror fails page-collect.

### 2026-07-31: GAP-3 Inc-C — test-rigor fix (Mary; authorized reviser) — NO production code change

Mary rejected Inc-C: parity test was vacuous for KV geometry (argmax invariant to reused-prefix KV on `tiny-gemma4-vlm`, so key/value swap still passed tokens). Geometry-corruption mutation (b) only caught by NEW direct byte/element-equality assertion on materialized paged KV. Added `materialize_published_prefix_kv(&mut self)` test-support accessor (read-only, cfg-gated `native-backend`) and byte-comparison assertion; zero production-code edits to `native_decode/mod.rs`, `pipeline/decoder_component.rs`, `pipeline/flat_autoregressive.rs`. All 3 mutations now FAIL; green run with byte-identical pages vs ORT.

### 2026-07-31: GAP-3 Inc-D — device-resident present-KV read-out + paged native CUDA decode (#567 MERGED)

**Author:** Cohaagen · **Status:** implemented, tests green, MERGED.

Lifted paged gate for **f32 device-resident rank-4 CUDA KV** (Inc-C only covered host). Wired: `DecodeCudaState::read_present_kv()` reads full capacity-padded device buffer (post-step, race-free) and pairs with **physical shape** `[1,H,max_len,head_dim]` (not logical); `seed_prefix()` mirrors capacity-offset writes. Critical: physical-vs-logical stride handling — the device buffer stride is `max_len*head_dim`, so feeding logical shape at `max_len > valid_len` + `H > 1` silently reads wrong head rows. Isolated in `device_present_kv_view(buffer, physical_shape, logical_shape)`.

Unification: `present_kv()` + `seed_kv()` dispatch host-growable (Inc-C) vs device-CUDA by session store; both feed same `extract_present_token`/`append_token_kv` into same host f32 paged store. Test: `native_paged_prefix_reuse_matches_ort_on_cuda_device` — paged-native-CUDA warm == cold pure-native == ORT `[0,5,6,7]`; **byte-equality** of device-mirrored pages vs ORT (catches value errors, not just tokens); H=2 unit test `device_kv_view_uses_physical_stride` proves stride bug. Non-vacuity: (a) gate-revert → reused=0, (b) wrong stride → mismatch in H>1, (c) no-op mirror → page-collect fails.

### 2026-07-31: GAP-3 Inc-D.1 — f16 device-resident present-KV + paged native CUDA decode (#568 MERGED)

**Author:** Cohaagen · **Status:** implemented, tests green, MERGED.

Real-model native paged decode. Inc-D only gated f32; every real target (Gemma4-e2b, Qwen3-30b-a3b) exports f16 device KV. Gate relaxed to `f32 || f16` (still gated → non-paged fallback on bf16, non-rank-4, in-place-CPU, sink-discontinuous). Shared `half` widen `f16→f32` via `half::slice::{reinterpret_cast, to_f32_vec}` (matching ORT `to_vec_f32_lossy`); shared narrow `f32→f16` via `half::f16::from_f32` (matching ORT `from_f32_slice_as`). Host paged store stays `Vec<f32>` (identical to ORT).

Fixture: `tiny-gemma4-vlm-cuda-f16` (Concat-KV, FLOAT16 KV) — KV must be arithmetic-exact; first build used `value = key + 0.5` (f16 round-to-even midpoint, CUDA/ORT divergence); fixed to `value = key * 2` (bit-exact multiply). Test: `native_paged_prefix_reuse_matches_ort_on_cuda_device_f16` — f16 device-mirrored pages **byte-identical** to ORT via unified `half` widen (crux resolved); parity + reuse + stride (H=2) all green. 3 mutation proofs: (a) gate-revert → reused=0, (b) wrong convert (raw u16 bits) → byte-equality fails, (c) no-op mirror → reused=0. Regression: parity suite 4/4 + multimodal-reuse 14/14 + session + #541 + #543 — all green.

**Remaining gaps (honestly gated):** bf16 (Inc-D.2: gate flip + fixture + widen test; verify ORT widens bf16→f32 same way); CPU in-place-GQA f32 (evaluated, not free); non-rank-4; sink-discontinuous; MoE routed-expert specifics (out of scope).

### 2026-07-31: 27B Scan-capture perf lever DEFERRED — redirect to GAP-3 increment lane

**By:** Coordinator (Justin away). Slice 1a (runtime dual-path single-trip inline, MERGED #564) is correctness-only, no capture. Slice 1b (inlined body enters CUDA-graph capture) requires handle-keyed multi-graph registry refactor (`CudaGraphLifecycle` singleton → registry; `graph.rs:114-122`); risk lands on working top-level GQA decode capture (2.14× eager / 1.75× ORT common-case win). Payoff is architecture-specific (recurrent/LinearAttention, e.g. 27B); higher-value lane exists: **GAP-3 Inc-A just merged → Inc-C unblocks Qwen3.6-35B-A3B MoE with real ORT oracle** (Justin target). State preserved: Mary's full 1b analysis in `feat/27b-scan-capture-1b` branch (scaffolding, counters, CUDA test, `_UNSAFE` spike flag) — parked UNMERGED. Resumption trigger: when device-graph-registry refactor prioritized; guard: GQA decode capture stays byte-exact + same tok/s.

### 2026-07-31: Scan 1a — single-trip inline dual-path (#564 MERGED)

**Author:** Mary · **Status:** implemented, tests green, MERGED.

Slice 1a (foundation for 1b): make single-trip Scan (`trip_count==1`, decode step) execute straight-line instead of generic loop (prefill stays looped). Runtime dual-path in `exec_scan`: flag-gated `ONNX_GENAI_SCAN_INLINE_SINGLE_TRIP` (default OFF), keyed at **execution time** on observed `trip_count`, not static rewrite (shared prefill+decode plan = correctness guarantee). Both loop and inline paths share `run_scan_body_step` + finishing code → byte-exact by construction. Counter `scan_inline_single_trip_count` (OFF=0, ON at trip_count==1). CPU byte-exact test + CUDA regression; on-model 27B (qwen3.6-27b int4, ~6.1 tok/s) token-identical flag-OFF vs flag-ON over prefill (~790 ms) and 48 decode steps. Regressions: session-reuse + async-fence + control-flow all green.
## 2026-08-04 — Scribe size compaction before #625/QMoE inbox merge

July 29-30 operational narrative entries were moved to `.squad/decisions-archive/2026-07.md` under `2026-08-04T00:40:00Z` to keep the live decision file focused on standing directives and current constraints. The coordinator prompt requested an age cutoff, but Scribe's charter treats size as the binding gate; the complete record was preserved in the archive.

## 2026-08-02 — Fused LinearAttention and Inc-1b current state

**By:** Scribe, from processed inbox drops and spawn manifest at 2026-08-02T10:05:00+0000

- PR #592 removed `ONNX_GENAI_DECODE_INLINE_SCAN`; Inc-1b decode-inline is now automatic and graph-property-gated. Dense/no-Scan models build no sibling, while genuine inlineable single-trip Scans keep the byte-exact ~2.03× decode win.
- PR #594 landed the EP-driven keep-as-op hook and dual-domain `LinearAttention` dispatch. On the 27B export the plan has 48 fused `LinearAttention` ops, 0 Scans, byte-exact greedy tokens, and about 41.3 ms/tok decode.
- The two levers compose: fused LinearAttention removes Scans where a kernel claims the function; Inc-1b remains for generic single-trip Scans with no fused kernel.
- Harry approved #592, rejected #594 only for rustfmt, and named Deckard as revision author. Deckard cleared the fmt-only blocker and later fixed pinned shape-inference registry counts after rebasing (#594 operator_count 217→218, entry_count 262→263 for the new standard-domain LinearAttention rule).
- Full processed inbox drops for this batch are archived in `.squad/decisions-archive/2026-08.md`; the July native-CUDA drops remain archived in `.squad/decisions-archive/2026-07.md`.

## 2026-08-02 — Final fused LinearAttention validation and #595 bench fix

**By:** Scribe, from processed inbox drops and spawn manifest at 2026-08-02T11:40:00+0000

- Cohaagen validated final merged main for the #592 + #594 composition gate: 27B greedy oracle PASS (CUDA == CPU == expected ids) and structural 48 fused `LinearAttention` / 0 Scans PASS.
- Follow-up steady decode measurement on merged main confirmed 40.8 ms/tok (24.5 tok/s). The fused-LinearAttention lane is complete; multi-EP hetero inlining relocation remains future work.
- Deckard authored and merged #595, restoring `reset_exec_phase_profile` so `profile_native --steady` bench binaries compile on main.
- Harry independently approved #595 after mutation-verifying the reset test, checking hot-path behavior, and confirming bench builds.

### 2026-08-02: #592+#594 composition oracle on main
**By:** Cohaagen
**What:** Ran 27b byte-exact greedy oracle + structural 0-Scan/48-fused lock on final main (0700c0fb) with BOTH merged PRs composed. Result: PASS, decode timing not emitted by the oracle harness; CUDA generate wall-clock was 14.39659273s for the 16-token greedy sequence (full test 5128.79s including CPU oracle).
**Why:** Neither PR's oracle ran with the other present; Justin is regression-sensitive. This is the composition integration gate.

### 2026-08-02: Review of PR #595 (profile_native compile fix)
**By:** Harry (independent reviewer)
**Verdict:** APPROVE
**Rationale:** Dangling `reset_exec_phase_profile` call on main confirmed; fix re-exports it, `profile_native` builds with `bench-native,cuda`, hot paths unchanged, reset test non-vacuous (mutation-verified), fmt/clippy clean, scope limited to reset plumbing + re-export + test. Minor non-blocking note: test does not assert PRINTED-guard reset.


## 2026-08-02 — Foundry native-vs-ORT CUDA sweep and #597 correctness lock

Full sweep tables and review notes are archived in `.squad/decisions-archive/2026-08.md` under the 2026-08-03T06:40 size-gate compaction. Live outcome: native CUDA beat ORT CUDA on every dense Foundry model measured (0.5B, 1.5B, Phi-4-mini, 7B); Qwen2.5-1.5B divergence was locked as native-correct in merged #597; hybrid 35B-A3B was blocked by the PackedMHA function-signature issue later fixed in mobius PR #449 and by subsequent capability gaps.

## 2026-08-02 — DeepSeek/GLM validation and post-#434 metadata regeneration

Full matrix is archived in `.squad/decisions-archive/2026-08.md` under August size-gate compactions. Live outcome: initial broad DeepSeek/GLM failures were stale or incomplete artifact metadata, not native numeric bugs; regenerated current Mobius #434 metadata admitted DeepSeek-V2 tiny, DeepSeek-V2-Lite real int4 native/ORT token match, and GLM-4-9B native golden lock. GLM-5.2 tiny/q4/qmoe remained blocked on unmerged Mobius #404 at the time.

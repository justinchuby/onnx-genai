# Decisions — live standing directives

Last compacted: 2026-07-28T11:35:49Z

Full historical ledger archived to `.squad/decisions-archive/2026-07.md`. The original
full snapshot is under "Full decisions snapshot archived by size gate — 2026-07-28T11:30:55Z";
post-rebase additions are under "Post-rebase decisions archived by size gate — 2026-07-28T11:35:49Z".
Older archives also live under `.squad/decisions/archive/`.

## Ledger health rule

The age-only archive criterion is structurally insufficient for this repository: during
high-volume campaigns, most entries are stamped within the last week, so a file can exceed
1 MB while "older than 7 days" matches nothing. When `.squad/decisions.md` exceeds the
hard gate, Scribe must archive by size as well as age: preserve the historical ledger in
`.squad/decisions-archive/{YYYY-MM}.md`, deduplicate rebase-reintroduced sections, then
keep this live file below 50 KB with only standing directives, active constraints, and
pointers to archived history.

Size compaction of a shared append-only file is not rebase-safe: concurrent appends can
silently reinflate the file without a conflict while leaving the compacted header intact.
A compaction PR must be re-run against tip immediately before merge if main advanced.

## Performance claim discipline

- A per-layer or microbenchmark speedup is not a model-level claim. Confirm with Amdahl
  and, when practical, a real model-level measurement before presenting campaign impact.
- Always state the exact model, dtype, metric, prompt/token regime, host load, and runner.
  TinyStories-1M and TinyStories-33M ratios are not interchangeable.
- Separate measured, estimated, and projected values. Do not compare measurements taken
  under different host load without labeling them. PR benchmark workflow absolute times
  are informational only; same-run PR-vs-merge-base deltas are the useful signal.
- Two independent measurements that agree beat one confident outlier. Retracted/corrected
  examples preserved in the archive include: the impossible 197 GB/s roofline target,
  the load-corrupted adaptive-calibrator verdict, the unmeasured 15x ORT batch-decode
  estimate, the 1x1 Conv microbenchmark headline later deflated through the real path,
  and the SDPA decode 1.9x vs 1.37x model-confusion correction.
- A SIMD or accelerated path without a reachability test is equivalent to an unwired
  placeholder.

## Apple Silicon portability and Mac CPU EP rules

- Mac CPU EP optimizations must generalize across Apple Silicon (M1/M2/M3/M4, base/Pro/
  Max/Ultra). The M1 Max is a measurement rig, not the target.
- No compile-time constants derived from one machine's measurements. Query topology,
  cache sizes, and features at runtime; derive tiling and thread counts from those facts.
- Feature-detect any path beyond the shared ARM baseline and keep a correct fallback.
- Reach the Apple matrix coprocessor through Accelerate (BLAS/BNNS), never hand-rolled
  AMX encodings. BNNS/Accelerate calls must happen at dispatch level, not inside Rayon
  parallel regions.
- The CPU EP stays one general implementation shared with Intel and ARM. Apple Silicon
  specialization lives behind runtime dispatch, not a parallel kernel tree.
- BNNS `BNNSMatMul` deprecation in macOS 15 is a maintenance migration to BNNSGraph, not
  evidence that the AMX/fp16 path or existing measurements are invalid.

## Load-adaptive decode path

`ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL` semantics are binding:

| Value | Meaning |
|---|---|
| unset | `On`: deterministic persistent SPMD pool, no probing |
| `=1` | `On`: explicit persistent pool |
| `=0` | `Off`: flat path |
| `=auto` | `Adaptive`: opt-in calibrator |

Reason: load-adaptive selection silently changed paths under agent load, damaged
reproducibility, and produced false verdicts. Libraries should be predictable by default
and adaptive only on request. Expose selected path via `decode_path_label()` and tracing or
`NXRT_CALIB_DEBUG` fallback.

## Dispatch-manifest inverse rule

Every claimed `(op, variant, platform) -> minimum tier` optimization needs a curated
manifest row, a `_TEST_HITS` counter, and a test proving the counter fires. The inverse is
also binding: if a fast path exists but a higher-priority guard intercepts it, the test
must fail. The manifest is CI-only and has zero runtime cost. Removing a manifest row is a
conscious un-claim; leaving a claim without reachability is a merge blocker.

Historical dispatch-miss pattern preserved in archive: Accelerate placeholder not wired,
M=1 GEMV intercepted by GEMM/half_gemm, SDPA NEON coverage gaps, BNNS unreachable for
non-contiguous weights, rescue block returning zeros, Conv macOS scalar/BNNS tier mistakes,
non-Conv CNN scalar paths, and 1x1 Conv's GEMM bypass existing but being intercepted by
BNNS Conv.

## Minimal-build and shape-inference rules

- Graph/layout transforms must be gated on both their infrastructure feature and the
  operator group that supplies their kernels. Wave 9 fixed NCHWc layout propagation to
  require both `mlas` and `ops-cnn`; an MLAS-only build must not advertise transforms whose
  CNN kernels are absent.
- Shape-inference registrations must use the operator's actual ONNX domain/version, not a
  convenient family namespace. Wave 9 corrected `StringNormalizer` and `TfIdfVectorizer`
  from `ai.onnx.ml` to the default `ai.onnx` domain.
- Attribute-dependent output typing must follow the active default/value attribute, not an
  unrelated class-list attribute. Wave 9 corrected `LabelEncoder-1` to mirror
  `CategoryMapper` default dtype selection.
- Sequence/Optional/Map/ZipMap-style container propagation remains blocked until the
  tensor-only `TypeInfo` model gains container-aware representation; do not fake these as
  tensors.

## BNNS / Conv / GEMM current guidance

- For fp16 MatMul on macOS, BNNS f16->f32 reaches AMX and is the preferred compute-bound
  prefill/batch path at M>=2 when available; M=1 decode remains a GEMV problem.
- Do not call BNNS from inside Rayon. BNNS uses system threading internally.
- For 1x1 Conv, the historical Sebastian diagnosis showed BNNS Conv can be dominated by
  filter creation/copy overhead; Deckard's final PR #347 used a spatial-size-dependent
  route through the real `im2col_gemm_execute` path, with practical claims scoped by model
  measurement rather than raw microbenchmark ratios.
- A fitted threshold is acceptable only when labeled as fitted and bracketed by measured
  data. A wrong rationale is worse than no rationale because future engineers will tune
  code to the false premise.

## Model artifact hygiene

Fetch large external models only when needed, measure them, and delete them immediately.
Do not leave downloaded benchmark models behind in `models/` or worktrees. The archived
ResNet/Whisper cross-model run explicitly used fetch-measure-delete and restored the disk
baseline afterward.

## Active historical pointers

For detailed per-PR narrative, use the archive rather than expanding this live file:

- Full pre-compaction ledger: `.squad/decisions-archive/2026-07.md` → "Full decisions
  snapshot archived by size gate — 2026-07-28T11:30:55Z".
- Post-rebase additions: `.squad/decisions-archive/2026-07.md` → "Post-rebase decisions
  archived by size gate — 2026-07-28T11:35:49Z".
- Prior active-ledger archives: `.squad/decisions/archive/`.
- Mac CPU EP load-bearing topics in the archive: PR #227 roofline lessons, load-adaptive
  opt-in, Apple Silicon portability directive, BNNS prefill/deprecation notes,
  benchmark-CI informational-only rule, dispatch-manifest lint, Sebastian/Deckard 1x1
  Conv correction, Iran SDPA model-ratio correction, and negative-result GEMV notes.
- Wave 8/9 topics in the archive: CUDA coverage batches 8/9, shape-inference catalog
  batches 3/4, NCHWc minimal-build gating, and strict reviewer-lockout correction cycle.

## Latest campaign summary — Mac CPU EP pointwise Conv, SDPA, and wave-9 rebase

- PR #342 merged: NEON depthwise, reviewed in the prior batch.
- PR #345 merged documentation-only: inline NEON GEMV small-shape investigation was a
  valid negative result; existing inline/cblas paths were already competitive.
- PR #347 merged (`00081cac`): 1x1 Conv routing corrected with model-scoped claim;
  documentation defects fixed before merge.
- PR #349 merged (`dc1ae0c5`): inline NEON SDPA decode path approved; headline corrected
  after TinyStories model-ratio confusion.
- Wave 8/9 consolidation archived: CUDA ScatterND/window functions, QuantizeLinear,
  DequantizeLinear, Dropout, NonZero, shape-inference batches 3/4, and NCHWc minimal-build
  gating.
- Chew approved PR #347 and PR #349 after numerics/reachability gates.

<!-- History before 2026-07-28T11:35:49Z was archived by size. Keep this file small. -->

## 2026-07-28T17:40:00+0000 — control flow, metadata, and QMoE offload wave

- PR #362 (`5a079029`) completed #355's tensor control-flow work: `If`, `Loop`, and
  `Scan` infer child graphs after formal body inputs are seeded from their owning
  node; Loop scan dimensions stay symbolic because a constant trip count is only
  an early-exit upper bound. Fully-static control-flow results whose byte size
  overflows degrade to dtype-only rather than triggering eager-planning overflow.
  Sequence, Optional, and Map propagation remains deliberately unresolved until
  the tensor-only SSA `TypeInfo` gains a container-aware representation.
- The cache roofline ceiling is informational when SLC covers at least 10% of a
  model's decode set (`cache_assisted`); retain the DRAM ceiling only for larger
  working sets and never report an unexplained roofline percentage above 100%.
- PR #364 (`3b08c025`) established route-first CPU QMoE's one-ahead prefetch
  strategy. It may engage only in eviction-neutral regimes: mmap streaming
  (`budget == 0`) or a fully resident layer. Partial-cache budgets execute the
  serial path. This preserves output, cache statistics, and `pipe_bytes <=
  serial` while retaining a measured 36–47% streaming throughput gain. Unifying
  this host path with GPU prefetch awaits Phase-3b live device binding; Granite's
  unfused MatMul MoE cannot currently engage it.
- PR #365 (`83f4c293`) made `onnx_runtime.*` metadata effective at engine load,
  with explicit programmatic settings retaining priority. Metadata collection
  recursively scans GRAPH/GRAPHS attributes. Every scanned node now uses a
  uniform structural `NodePath` identity (including anonymous and duplicate-name
  top-level nodes); names are diagnostic labels only. External name-addressed
  hints resolve to a unique structural node or remain in a separate external-name
  keyspace, so they cannot collide.
- Large-model smoke status is a blocker, not an offload regression: the 27B
  native path hits the Unsqueeze rank bug, ORT lacks
  `past_key_values.*.recurrent_state`, and the needed Mobius #432 mapping remains
  unmerged; all H200 capacity was occupied externally. Granite MoE runs on ORT
  CUDA but exports unfused MatMuls, while the fused fixture verifies route-first
  CPU paging independently.
## 2026-07-28T13:50:00Z — MobileNetV2 remaining gap: Clip dispatch-miss (defect #14)

Per-op profiling on MobileNetV2-12: Clip dominated at 76.8% of runtime. Fixed with
`clip_contiguous_f32_fast()` — NEON-accelerated zero-copy path. Per-op: 34ms → 0.86ms
(39.5x). Amdahl projected 3.93x; measured 3.75–3.91x model speedup. Residual gap (11.5ms
vs ORT 6.3ms) is Conv-dominated, no single lever remaining. **PR #359.**

## 2026-07-28 — NEON fast path for Relu (MLAS gate audit item #2)

Relu was second HIGH-priority violation: only fast path behind `cfg(feature="mlas")`.
ResNet-18: Relu 0.43 ms (5.17%) before → 0.094 ms (1.17%) after. Per-op: 4.6×. Model:
1.044× (within Amdahl projection). Added `clip_contiguous_f32_fast()` pattern. Manifest
row for `(Relu, contiguous_f32, all, tier2)` with `RELU_F32_FAST_TEST_HITS` counter.

## 2026-07-28 — Standing Directive: Hardware-Rationale Accuracy in Dispatch Comments

Source comments justifying dispatch thresholds with hardware figures must: (1) cite
verified figure from specs/teardown/on-device; (2) state if constant is derived or fitted;
(3) if fitted, state measured bracket and confirmed platforms. Wrong rationale is worse
than no rationale — prevents future engineers from silently breaking working dispatch.

## 2026-07-28 — Thin-M GEMM bypass for f32 prefill on Apple Silicon

For small M (2-16) with large B (K×N > 4M), cblas only achieves ~25 GB/s; bypass to
NEON column-parallel path for constant (pre-transposed) weights. M crossover at 16 measured
on M1 Max, bracket [16,24]. TinyStories-33M TTFT: 17.7ms → 13.3ms (-25%). f32 weight
transposes precomputed at model load. Counter: `THIN_M_GEMM_TEST_HITS`. **PR #351.**

## 2026-07-28 — report time-to-first-token from process start

`compare.rs` now reports process start → first token ms (model_load + TTFT) as derived
column. TTFT alone favors runtimes that front-load work into model load (ORT pre-packs).
Cold-start metric: native 55–57.5ms, ORT 150–165.5ms (2.6–2.7×). Updated
`examples/profiles/README.md` with mechanism explanation.

## 2026-07-28 — Republish profile figures with load context

Measurements at load 2.5–3.7 show: qwen2.5-0.5b-f16: 1.72× decode, 1.68× e2e, 5.01×
cold-start. TinyStories-33M: 0.91× decode (ORT wins), 0.82× e2e (ORT wins), 2.47×
cold-start. Verified against Justin's independent measurement (< 2% agreement). Every
number carries host load context. Unflattering numbers preserved in README.

## 2026-07-28T00:00:00Z — probe the stream you are writing to

`emit_stats_line` in PR #372: stats written to stderr but probed stdout TTY status, inverting
decision under redirection. **Durable rule:** probe the stream you are writing to. Stats on
stderr → test `stderr().is_terminal()`. Never cross-probe. Extracted `stats_text()` for pure
testable logic; added `stats_format_follows_stderr_not_stdout` unit test covering all 4 cases.

## 2026-07-28 — Feature Gate Coverage Lint

Added `scripts/check_feature_gate_coverage.py` — sixth CI layer targeting blind spot:
cfg-gated performance paths whose fallback is unmonitored. Audit findings: CRITICAL (Clip),
HIGH (Relu), MEDIUM (GlobalPool) — double-copy allocations on macOS. No other feature flags
guard kernel performance paths. Script catches missing *instrumentation* on fallback paths,
not missing optimizations — manifest then makes tier visible for human review.

## 2026-07-28 — Roofline ceiling: cache-assisted threshold for small models

TinyStories-33M (107M params, 267.8 MB decode set) exceeds 100% because SLC (48 MiB = 18%
coverage) provides inter-token reuse lift. Broadened check from "cache_resident" (entire model
fits) to "cache_assisted" (SLC ≥ 10% of decode set). Roofline ceiling marked informational
for cache-assisted models. No change to floor constants. **PR #354.**

## 2026-07-28T04:30:00-07:00 — CLI improvement track durable lessons

- Recurring verification defects are a durable review pattern: do not accept code that merely appears to verify, preserve, or clean up its claim. Empty-turn handling, flaky tests, speculative rewind placement, benchmark caps, cache keys, broad assertions, and stale fixture inventories were each caught by different review/automation layers; keep redundant review layers in place.
- Bugs live where automation cannot reach. The CLI track widened automation coverage as first-class product work: CLI CI, cross-platform contract tests, visible ORT-library reporting, and Miri in CI all closed places where defects had been invisible.
- PR #315 proved observability pays for itself: `cargo test -p onnx-genai-engine --lib` moved from 178 passed / 64 failed to 253 passed / 0 failed once agents could select a working ORT and see which ORT library was actually loaded.
- Silent behavior is a bug class to eliminate. Ignored `--temperature`, silent CUDA fallback, empty turns on context exhaustion, requested-vs-resolved backend reporting, and invisible budget caps were fixed by making behavior observable, not just internally correct.
- Coordinator merge and diagnosis discipline: distinguish allocator growth from scheduler worst-case reservation, and verify merges by building or equivalent validation; conflict markers and scratch files can survive if git output is trusted without inspection.

## CLI charter — standing directives

These two govern all ongoing CLI work and outlive the PRs that produced them. They were
compacted into `.squad/decisions-archive/2026-07.md` on 2026-07-28 and are restated here
because they are live policy, not history: an agent reading only this ledger must see them.

### The CLI is a developer/maintainer tool, not a consumer product
**By:** Justin Chu (2026-07-27)

The `onnx-genai` CLI is scoped as a development and maintainer instrument. It is not
competing with consumer local-inference tools. Rank CLI work by *does this shorten a
maintainer's debug/iterate loop, or expose engine behavior we cannot otherwise observe?* —
not by *does a competitor have it?*

**Explicitly rejected, do not re-propose:** remote-client mode against an OpenAI-compatible
server (third-party CLIs cover it); model registry / pull / consumer model lifecycle;
conversion, quantization and fine-tune loops as CLI product features. See
`docs/research/cli/00-backlog.md`.

### The REPL is the primary CLI investment
**By:** Justin Chu (2026-07-27)

Target quality bar is GitHub Copilot CLI's interactive shell, with one deliberate
divergence: **the terminal model is ratatui's inline viewport, not a full-screen alternate
screen.** Native scrollback and terminal-native copy are features for a tool whose output is
pasted into issues, benchmarks and traces, and the alternate screen costs both. Justin was
shown the tradeoff and chose inline; see `docs/research/cli/05-repl-redesign.md` §2.

Phase 1 landed in #289. Remaining phases cover session/runtime interaction — `/fork` and
`/rewind` — which depend on runtime APIs tracked in
`docs/research/cli/04-runtime-capability-inventory.md` and `06-fork-rewind-api.md`.
Fork is reserved behind a type gate and **not yet enabled on any backend**.

## 2026-07-28 — Declarative, name-agnostic model I/O and shared-KV contracts

**PR #373 / issue #231 (Melina; reviewed by Richter; merged `61d3bdac`).** Decoder and
proposer ports resolve first from exact `model.io` / `speculative.io` declarations, then
from unique dtype/shape signals; legacy terminal-name matching remains compatibility-only.
Declared KV lists pair positionally. Attention representation permissions are independent of
attention implementation, and `io.kv_update: shared_buffer` declares operator-agnostic
shared-buffer KV updates. Strict attention sequence-length validation rejects incompatible
contracts early.

## 2026-07-28 — Honest route-first QMoE residency tests under coverage

**PR #378 (Nandez; reviewed by Kuato; merged `ac75e146`).** Coverage-mode route-first QMoE
offload assertions now reflect the scheduler contract: serial execution peaks at one resident
expert; prefetch peaks in `[1, 2]` and remains below selected experts. The shared
`hold_metrics_test_lock` helper is poison-recovering across 20 call sites, preventing a
previous test panic from cascading into phantom residency regressions on unrelated PRs.

## 2026-07-29 — Core decode ports must not depend on exporter names

**PR #380 / issue #377 (Melina; reviewed by Cohaagen; merged `47c3331d`).** The core decode
path resolves roles only from explicit metadata or a unique tensor-shape match, and reports
the required metadata key for ambiguity. Encoder-decoder Whisper/TTS fixtures declare
component decoder I/O explicitly, rather than restoring decoder-name guessing.

**Review rule:** metadata or I/O-detection changes must run the CLI ORT E2E suite in addition
to engine/native unit tests; Cohaagen's fix-delta re-review ran the gate successfully
(23/23).

## 2026-07-29 — Shared-buffer decode must thread declared KV pairs

**PR #382 / issue #377 continuation (Benny; reviewed by Lori; regression test by Leon; merged `85b9ba15`).** ORT batched shared-buffer and static-cache decode adapters no longer guess exporter I/O names; they consume declared KV pairs. The repair also restores those pairs when constructing `BatchedSharedBufferDecodeSession`, fixing the latent #380 regression where passing `None` made construction always fail. A CPU engine-level continuous-batch test using `tiny-llm-sharedbuffer` now compares sequential generation and fails at construction if declared `model.io.kv_inputs` / `kv_outputs` stop reaching the session. This is deliberately CPU coverage because the previous CUDA-only E2E auto-skips without CUDA.

## 2026-07-29 — Recurrent shape geometry shares an opset-aware contract

**PR #386 / issue #355 slice (Hauser; reviewed by Helm; merged `39c28b44`).** RNN, GRU, and LSTM shape inference share `recurrent()`, which propagates symbolic sequence and batch dimensions, derives direction count and hidden size, and emits only declared Y/Y_h/Y_c outputs (including LSTM Y_c). Registrations at opsets 1 and 14 enforce the layout boundary: pre-14 ignores a stray `layout`, while 14+ honors it. Missing or insufficient shape information remains permissive rather than panicking. Helm independently checked ONNX axis order for both layouts and ran the recurrent suite (238 tests) plus clean clippy.

## 2026-07-29 — OpenAI JSON Schema response format reaches engine constraints

**PR #388 / issue #183 follow-up (Tony; reviewed by Harry; merged `804ba860`).** Chat completions accept OpenAI Structured Outputs `response_format: {type: "json_schema", json_schema: {name, schema, strict?}}` and map the schema object to `GenerateConstraint::JsonSchema`; malformed schema/name input produces a type-driven HTTP 400. `wants_json_object` became `wants_constrained_json`, so streaming buffering and incomplete-JSON retry apply to both constrained formats. Forced specific-tool choice retains its Lark-constraint precedence. Harry independently verified live llguidance consumption, targeted tests, and clean clippy; the two full HTTP-suite failures reproduce on origin/main from an unrelated tiny-fixture context limit.

## 2026-07-29 — Tool-call parser handles Qwen, Llama, and Mistral formats

**PR #390 / issue #183 follow-up (Stevens; reviewed by Edgemar; merged `0e62150e`).** `parse_tool_calls` recognizes Qwen/Hermes `<tool_call>` objects, Llama 3 `<|python_tag|>` objects, and Mistral `[TOOL_CALLS]` arrays. A serde `StreamDeserializer::byte_offset()` prefix scanner consumes consecutive complete top-level Llama JSON values (including semicolons inside strings) without naive splitting and terminates safely on malformed input or model terminators. One converter prefers `arguments`, falls back to `parameters`, and assigns sequential IDs. Edgemar independently ran eight parser tests and clean clippy; the roughly 49 local `tests.rs` failures are pre-existing missing-weights-fixture failures.


## 2026-07-29 — Qwen3 native CPU reached ORT peak parity

**Branch/PR:** `qwen3-perf-followups` / PR #398. **Agents:** Resch, Batty, Deckard support.

- Profile before the final wave: per-token native CPU was dominated by `MatMulNBits`
  (197 calls, ~8.6 ms, 79-88% of time). Secondary costs were `Reshape` (~0.86 ms from
  113 no-op dispatches ORT eliminates), GQA (~0.56 ms), norms (~0.57 ms), and
  `FusedSiluMul` (~0.17 ms).
- Banked path to parity: KAI packed-SDOT made native viable; MLAS QNBit SPMD sharding fixed
  the threadpool integration gap; residual GQA/norm/Silu fusion reduced non-GEMV overhead;
  kernel preselection (`348c39a6`) cached MLAS packed-B plus reusable SQNBit workspace and
  skipped per-call route/kernel selection, improving best MatMulNBits bucket ~8.6 -> 7.3 ms.
- Batty's no-op reshape elimination (`76f116cf`) removes provably no-op `Reshape`/`Identity`
  from the load-time plan; `Reshape` disappears from the profile instead of spending 113
  calls / ~0.86 ms per token.
- Resch's decode work-stealing prototype (`fe54dd9d`) regressed under this design (best/median
  ~95/82 tok/s vs fixed SPMD ~105/100), so fixed SPMD stays default. Do not reintroduce
  naive per-op atomic tile stealing as the median fix.
- Result: native CPU moved from ~66% of ORT (~69 tok/s) to peak/p90 parity. Best native
  measured ~110.2-110.6 tok/s vs ORT ~109.3-111.1; native p90 ~108.5-109.2 vs ORT p90
  ~110.5. ORT still leads median by ~5% (native ~102.5-103.9 vs ORT ~108.8) because fixed
  SPMD is contention-sensitive on the shared host; good native runs match ORT, slow runs drag
  the median. Remaining robust-median lever is a low-overhead Eigen-parity/whole-step
  work-stealing pool, not another local MatMulNBits kernel.

## 2026-07-29 — Work-stealing wave ended at ORT parity, not beyond it

**Branch/PR:** `qwen3-perf-followups` / PR #398. **Agents:** Deckard, Resch, Batty.

- Deckard's Eigen-style `WorkStealingThreadPool` prototype (`8fad4915`) cleared the isolated dispatch gate: p50/p90 dispatch 2.3/2.6 us vs fixed-SPMD 29.2/65.9 us and Rayon 49.8/87.7 us.
- Resch's decode integration (`542f2ebd`) regressed real Qwen3 decode: work-stealing best/median 97/90 tok/s vs fixed-SPMD 106/99, with ORT 109/100. This was the second work-stealing attempt to regress; isolated dispatch latency does not predict decode throughput. Fixed-SPMD's cache locality and lower coordination remain the default; `ONNX_GENAI_CPU_DECODE_SCHEDULE=steal` stays opt-in only.
- **Windows exe-naming rule:** Deckard's admin blocker (`ec062ebb`) was benign. Windows UAC installer detection flagged the unsigned benchmark name `threadpool_dispatch` because `dispatch` contains `patch`, causing `ERROR_ELEVATION_REQUIRED` 740 before `main()`. The same binary renamed ran non-admin; pool, MLAS, and CLI all run non-admin. Never name generated binaries/exes with installer words or substrings: `patch`, `dispatch`, `setup`, `install`, `update`.
- Final result: native CPU EP reached practical ORT parity from ~66% / ~69 tok/s at session start. Final window: fixed-SPMD best/p90/median 106.0/105.2/99.4 vs ORT 108.9/107.3/99.8, tying median and landing best/p90 within ~2-3%. Cleanly beating ORT now likely requires a kernel faster than KleidiAI hand-tuned asm, which is large and uncertain.

## 2026-07-29T22:00:00-07:00 — Deep overhead investigation final verdict

**Branch/PR:** `qwen3-perf-followups` / PR #398. **Agents:** Sebastian, Resch, Deckard, Batty.

- Sebastian's overhead decomposition (`dcd3dda3`) found native `MatMulNBits` at about 7.45 ms in the static-SPMD bucket, while calibrated ORT attribution put its `MatMulNBits` around 7.2 ms; ORT's Chrome profiler distorted raw ORT kernel time toward ~6.0 ms. Native executor dispatch costs about 1.1 ms/token (~2.3 us/node), while KV, sampling, logits fetch, and input prep are tiny (~0.18 ms/token). The remaining gap is kernel invocation/threading plus small executor overhead, not KV or sampling.
- ORT deep research explains the shape of the gap: ORT calls `MlasQNBitGemmBatch` once at full width, MLAS partitions N with `LoopCounter` dynamic load balance on the Eigen intra-op pool, `SequentialExecutor` keeps per-node overhead near zero with memory planning/no per-token allocation, and ORT benefits from `SkipSimplifiedLayerNorm`/`MatMulNBits` fusions plus GQA flash decode (`MlasFlashAttentionGQA`). The three follow-up priorities were: (1) try full-width MLAS, (2) trim native executor hot-path overhead, and (3) audit available fusions.
- Priority 1 was a negative result. Resch's full-width MLAS experiment (`9a48b46d`) regressed on our pool: `MatMulNBits` median 8.07 ms versus static 7.45 ms, and throughput around 93 tok/s versus 106 tok/s. Static-SPMD remains the default; full-width MLAS stays opt-in only through `ONNX_GENAI_CPU_MM_MLAS_NO_SHARD=1`.
- Deckard's MLAS pool backend (`aef9dfd7`) still matters as infrastructure: MLAS standalone threading hooks now route to the persistent `WorkStealingThreadPool` instead of Rayon. That made the experiment clean, but did not make full-width decode faster than static-SPMD.
- Priority 2/3 was useful but not decisive. Batty's executor cleanup (`aa5b67e6`) deferred tensor-view construction, reduced bookkeeping, and kept output buffers resident; subspans improved, `Reshape` remained eliminated, and no Qwen3 `Add -> SkipSimplifiedLayerNormalization` fusion existed to add. Final benchmark: native best/p90/median 106.5/106.2/101.0 tok/s versus ORT 109.5/109.3/107.3.
- Final standing lesson: native CPU EP is now about 97% of ORT, up from ~66%. The last gap is small and diffuse: ORT drives the same MLAS kernel slightly better through a mature Eigen pool and leaner `SequentialExecutor`. Every tested pool variant (Rayon, Eigen-style work stealing, full-width MLAS) regressed versus fixed static-SPMD. Runtime overhead is competitive rather than the main problem; closing the last ~3% likely means reproducing ORT's exact threadpool/executor tuning, a multi-week effort with diminishing returns.


### 2026-07-30: Replicate ORT QNBit dynamic partitioning in mlas-sys
**By:** Deckard
**What:** `WorkStealingThreadPool::parallel_for` now mirrors ORT's `ParallelForFixedBlockSizeScheduling`: fixed-size blocks, up to one claimant per pool lane, up to eight cache-line-separated `LoopCounter` shards, and atomic dynamic block claiming from a home shard before scanning other shards. The MLAS standalone hook runs each `MlasTrySimpleParallel` work item through that dynamic block claim path.

**Why:** ORT's QNBit path computes `TargetThreadCount = M*N*K*BatchN / 65536 + 1`, caps it at `MlasGetMaximumThreadCount(ThreadPool) * 8`, aligns the N split to `MLAS_QGEMM_STRIDEN_THREAD_ALIGN` (16 in our vendored MLAS), then calls `MlasTrySimpleParallel(ThreadPool, ThreadsPerGemm * BatchN, ...)`. ORT's `SimpleParallelFor` maps that to dynamic `LoopCounter::ClaimIterations`, avoiding the fixed-SPMD barrier tail. For Qwen3 decode-style `M=1,K=1024,BlkLen=128`, the resulting 8-thread partitions are: N=1024 => strideN=64/16 tiles; N=2048 => 64/32; N=3072 => 64/48; N=5120 => 80/64; N=8192 => 128/64. On 6 threads: N=1024 => 64/16; N=2048 => 64/32; N=3072 => 64/48; N=5120 => 112/46; N=8192 => 176/47. ORT claimants/shards are min(pool_threads, tiles) and min(8, pool_threads, tiles).

**Sources:** ORT `threadpool.cc` has `LoopCounter` and `ParallelForFixedBlockSizeScheduling` dynamic claiming (raw main lines ~271, 413-473) plus cost-model `CalculateParallelForBlock`/`ParallelFor` (lines ~565-632). ORT `threadpool.h` documents `TrySimpleParallelFor` and `TryBatchParallelFor` (lines ~296-347) and calls out `LoopCounter::ClaimIterations` dynamic balancing (lines ~84-87). ORT MLAS `threading.cpp` routes `MlasTrySimpleParallel` and `MlasTryBatchParallel` to those threadpool APIs (lines ~62-130). ORT `qnbitgemm.cpp` computes QNBit complexity/stride/tile count and calls `MlasTrySimpleParallel` (lines ~1718-1776).


## 2026-07-30 — MLAS SQNBit full-width dynamic partition is not the default

Context: Deckard wired ORT-style dynamic block claiming into `mlas-sys`. I briefly
added a CPU EP toggle `ONNX_GENAI_CPU_MM_MLAS_FULLWIDTH=1` so constant
`MatMulNBits` weights could bypass the static SPMD N-shard path and run a single
full-width `sqnbit_gemm_into_with_workspace(..., multithread=true)` call through
the cached full-width MLAS pack/workspace.

Measurement host was `aarch64-pc-windows-msvc` on a contended Windows machine;
`cargo check -p onnx-runtime-ep-cpu --features mlas --target
x86_64-pc-windows-msvc` passes, but these are not x86 throughput numbers.

Qwen3 0.6B CPU int4 steady decode (`--tokens 96 --runs 15`, decode skip 8):

| path | tok/s best | tok/s p90 | tok/s median | MatMulNBits ms/token |
|---|---:|---:|---:|---:|
| native static-SPMD | 110.32 | 109.76 | 97.61 | 8.93 |
| native full-width dynamic | 91.72 | 91.64 | 81.65 | 8.29 |
| ORT CPU | 106.16 | 106.03 | 104.89 | n/a here |

Correctness: full-width and static native emitted identical 96 generated token
ids in the decode check. ORT emitted a different sequence on this model/backend.

Verdict: do **not** make full-width dynamic the default. The MLAS-only portion
improved on one profiled run (8.93 -> 8.29 ms/token) but end-to-end steady
decode lost badly and 15-run full-width processes intermittently hung/stalled
after several measured runs.

Follow-up: the live `ONNX_GENAI_CPU_MM_MLAS_FULLWIDTH` product/test toggle was
reverted. Its process-global `OnceLock` env cache polluted in-process ARM64
tests that also mutate MatMulNBits routing env vars, making the suite
order-dependent. Keep the negative result as the artifact; keep the product on
the previously-green static-SPMD default and use isolated subprocess harnesses
for any future full-width A/B experiments.

Test hygiene follow-up: ARM64 QNBit/KAI route tests now avoid using shared hit
counters to infer the path taken after execution; they inspect the kernel's own
cache instead, so concurrent tests cannot make an MLAS run look like a KAI run.
The work-stealing parity child explicitly requests two steal tiles per worker
when it asserts extra dynamic segments; the production default remains one tile
per worker.


### 2026-07-28: CI tiering rejected; run full coverage on PRs
**By:** Pris
**What:** CI has two parallel PR signals: a Linux-only uninstrumented `Fast (Linux x86_64)` job in `ci.yml` for early feedback, and the full coverage gate. PRs, `main` pushes, nightly schedules, and manual dispatch still run the full signal: quality checks, security audit, coverage-instrumented offline crate tests across Linux x86_64 / Windows x86_64 / macOS arm64, uninstrumented Windows ARM64 offline crate tests, Linux-only MLAS coverage, Linux/Windows CLI ORT coverage, and CUDA compile lanes. Rule: run tests on every platform; instrument for coverage only where the coverage is informative. The fast lane is deliberately duplicated work, has no `needs`, and does not gate or serialize the full lane; passing it alone is not merge-ready. The `ci:full` label trigger was removed because it became a no-op. Codecov carryforward remains disabled because PRs upload the same four flags as `main`.
**Why:** Tiering was tried to reduce PR cost. The first split had the axis wrong: platforms are signal in this repo, while instrumentation is cost. Justin then decided the coverage runtime is acceptable too, and subsequently requested a Linux-only no-coverage fast lane for fail-fast ergonomics (`覆盖率都打开吧 我可以接受这个运行时间`), so the conscious final choice is to pay the full cost rather than risk reintroducing blind spots. Keep this rationale visible so a future slow-CI discussion starts from the fact that platform and coverage reductions were considered and rejected.

### 2026-07-28: CI timing measurements for fast lane and ARM64 coverage tradeoff
**By:** Pris
**What:** Warm-cache measurement shows the fast lane is earlier but not a ~3 minute lane. Run 30387825072 restored the fast `target` cache and completed `Fast (Linux x86_64)` in 5m43s. Step breakdown: target restore 43s, build 1m26s, tests 1m35s, clippy 1m30s, plus setup/cache overhead. The prior no-target-cache run 30386077420 took 8m12s; the first cold run 30382113387 took 9m09s.

Full-gate clean run 30386077420 passed in 18m27s with per-job timings: `Rust coverage (Windows ARM64)` 18m22s; `CLI ORT (Windows x86_64)` 18m06s; `Rust coverage (Windows x86_64)` 13m36s; `Rust coverage (macOS arm64)` 12m41s; `CLI ORT (Linux x86_64)` 6m52s; `Rust quality` 6m48s; `Rust coverage (Linux x86_64)` 5m36s; `CUDA compile (Windows x86_64)` 3m37s; `CUDA compile (Linux x86_64)` 1m35s. Earlier full run 30382113387 was 21m27s with `Rust coverage (Windows ARM64)` at 21m20s, and run 30383978159 showed ARM64 variance at 26m19s.

**Why:** The fast lane is not missing because of a stale cache key once the cache exists: the target key hit in 30387825072 (`...-fast-rust-1.97.1-...`), and the remaining time is the full Linux offline build/test/clippy workload plus restoring a ~3GB target cache. It still surfaces Linux failures 12-20 minutes before the slowest full-gate completion, but it does not meet the hoped-for ~3 minute target. Windows ARM64 coverage owns the full-gate critical path. Replacing ARM64 coverage instrumentation with uninstrumented ARM64 tests would keep the platform signal and likely save the difference between the observed 18-26m ARM64 instrumented job and the historical ~6-7m uninstrumented ARM64 job; wall-clock savings depend on the next-slowest job, ranging from ~0-3m on the clean success runs where Windows CLI was also slow, up to ~16m when the other jobs are warm/fast. This is an option for Justin, not a change made here.

### 2026-07-28: Drop Windows ARM64 coverage, keep Windows ARM64 tests
**By:** Pris
**What:** Removed the `Rust coverage (Windows ARM64)` job and replaced it with `Rust (Windows ARM64)`, which runs the same offline crate tests and clippy without coverage instrumentation or Codecov upload. The ARM64 job had uploaded under the shared `offline` flag, so no Codecov flag disappears: `offline` is still uploaded by Linux, Windows x86_64, and macOS; `mlas`, `cli-ort-linux`, and `cli-ort-windows` are unchanged.
**Why:** Durable rule: run tests on every platform; instrument for coverage only where the coverage is informative. Windows ARM64 platform execution catches real platform bugs, but coverage for these pure-Rust crates duplicates x64/macOS coverage while owning the full-gate critical path. Justin chose to drop ARM64 coverage after seeing the timings.



### 2026-07-28: Windows ARM64 coverage removal verified
**By:** Pris
**What:** Verified run 30390299025 after removing ARM64 coverage. Full CI passed in 18m54s wall-clock. New critical path is `CLI ORT (Windows x86_64)` at 18m50s, followed by uninstrumented `Rust (Windows ARM64)` at 15m12s, `Rust coverage (Windows x86_64)` at 14m38s, `Rust coverage (macOS arm64)` at 9m32s, `Fast (Linux x86_64)` at 9m21s, `Rust coverage (Linux x86_64)` at 8m34s, `CLI ORT (Linux x86_64)` at 7m16s, `Rust quality` at 6m29s, `CUDA compile (Windows x86_64)` at 1m37s, and `CUDA compile (Linux x86_64)` at 0m48s.

Codecov consequence: no flag disappeared. The removed ARM64 job did not have its own flag; it had only contributed to the shared `offline` flag. In run 30390299025, Codecov uploads succeeded for `offline` from Linux, Windows x86_64, and macOS; `mlas` from Linux; `cli-ort-linux`; and `cli-ort-windows`. Carryforward remains disabled. Open question for the first post-merge `main` run: whether removing one contributor to the shared `offline` flag causes Codecov to report a project-wide coverage change against historical commits. This branch dispatch proved the remaining uploads happen, but the post-merge Codecov comparison behavior is still unverified.
**Why:** Dropping ARM64 coverage alone did not materially reduce this measured wall-clock because Windows CLI ORT was nearly as slow as the old ARM64 critical path and became the new critical path. The durable rule still stands: run tests on every platform; instrument for coverage only where coverage is informative.


# Decision: Multi-turn and batch benchmarks reveal structural native deficit

**Date:** 2026-07-28
**Author:** Pris
**Status:** Active
**Affects:** Iran, Deckard (native backend architecture)

## Context

PR #351 established native's cold-start advantage (2.47–4.63× faster process
start → first token). Justin flagged that real usage is multi-turn: load once,
prefill every turn. We needed to know whether ORT's pre-packing cost amortises
across a session — and if so, what the fix is.

## Findings

All measurements taken under exclusive bench lock at load 1.5–3.6 on Apple M1 Max.
Corroborated with second runs at comparable load.

### Multi-turn LLM

| Model | Break-even turn | ORT overall advantage (10 turns) |
|---|---|---|
| TinyStories-33M (f32) | 2 | 2.1× |
| Qwen2.5-0.5B (f16) | 8 | 1.18× |

**Root cause: NOT pre-packing amortisation.** The native backend has no
session-persistent KV cache. Each turn re-prefills the entire conversation
(O(context_length)). ORT's session API preserves KV, so each turn prefills only
new tokens (O(new_tokens)). At turn 10, native TTFT is 6–8× its turn-1 value
while ORT TTFT stays flat.

### Steady-state per-prefill (turns 3–10)

| Model | Native TTFT ms | ORT TTFT ms | Ratio |
|---|---|---|---|
| TinyStories-33M | 93 | 28 | 3.4× ORT faster |
| Qwen2.5-0.5B-f16 | 463 | 150 | 3.1× ORT faster |

### Batch vision (MobileNetV2)

- Batch=1: native 0.50× ORT (11.6 ms vs 5.8 ms)
- Batch>1: **native crashes (segfault)** — correctness bug
- ORT scales 2.2× from batch=1→16

### Cache survival (PR #353)

Weight transpose caches ARE correctly reused across turns:
- Qwen f16: 168 entries at load, stable across all turns
- TinyStories f32: lazily fills to 25 entries, then stable

This is NOT the cause of the deficit.

## Should we pre-pack?

**No — pre-packing would not address the dominant issue.**

The multi-turn deficit is caused by the absence of persistent KV, not by
slower per-token computation. Pre-packing could narrow the per-prefill gap at
equal context length (estimated 1.5–2× improvement), but it cannot eliminate
the O(context_length) vs O(new_tokens) structural disadvantage.

If persistent KV sessions are added to the native backend, THEN pre-packing
should be revisited to close any remaining per-prefill gap. The load-time cost
of pre-packing (estimated +200–400 ms for Qwen-0.5B-f16, based on ORT's 1.8 s
vs native's 340 ms load) would be acceptable for long-lived servers but
unacceptable for cold-start use cases — an opt-in mode would be needed.

## Decisions for Iran/Deckard

1. **Session-persistent KV for native backend** is the #1 priority for
   multi-turn competitiveness. Without it, no kernel optimization can close
   the gap beyond 3 turns.
2. **Batch>1 vision segfault** is a correctness bug that should be filed and
   fixed before any batch benchmark claims.
3. **Pre-packing** should be deferred until after persistent KV lands, then
   re-evaluated.

## Published conclusion changes

The PR #351 cold-start advantage **remains valid for one-shot use**. For
multi-turn sessions (≥3 turns on small models, ≥5–8 on large), ORT is
cumulatively faster. This is now documented in `examples/profiles/README.md`
with the multi-turn framing section.

## 2026-07-30T08:20:00-07:00 — ORT cost-model partitioning is not the native Qwen3 default

**By:** Scribe
**What:** Native static-SPMD CPU EP now matches or slightly beats ORT on best-case and p90 Qwen3 decode throughput on the contended Windows ARM64 host: native 110.3/109.8 tok/s versus ORT 106.2/106.0 tok/s. Native still trails ORT on median because fixed-SPMD has higher variance on the shared host.
**Why:** Matching ORT's dynamic `ParallelForFixedBlockSizeScheduling`/`LoopCounter` partitioning helps the isolated full-width QNBit kernel (23.5 us mean / 18.9 us p50 versus static split 89.7 / 87.8 us), but loses end-to-end once pool park/wake variance is paid across many small ops: full-width dynamic reached only 91.72 tok/s best versus static-SPMD 110.32 and ORT 106.16. The live full-width toggle was abandoned. If this track is pushed further, the next plausible lever is vendoring Eigen `NonBlockingThreadPool` to reduce wakeup variance, with uncertain payoff.

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

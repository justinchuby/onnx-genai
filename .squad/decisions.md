# Decisions — live standing directives

Last compacted: 2026-07-30T04:10:00Z

Full historical ledger archived to `.squad/decisions-archive/2026-07.md`:
- "Full decisions snapshot archived by size gate — 2026-07-28T11:30:55Z"
- "Post-rebase decisions archived by size gate — 2026-07-28T11:35:49Z"
- "Narrative entries compacted by size gate — 2026-07-29T21:19:00Z" (first run)
- "Narrative entries compacted by size gate — 2026-07-29T23:30:00Z" (merge resolution)
- "Post-rebase narrative tail compacted by size gate — 2026-07-30T04:10:00Z"

Older archives: `.squad/decisions/archive/`.

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

**Concurrent Scribe runs are a structural hazard.** Two Scribe runs merged the same file
in the same window on 2026-07-29, producing divergent copies that git had to reconcile.
The root fix is to assemble decisions.md from the inbox rather than hand-merge it: if the
live file is only ever written by appending inbox drops, two runs produce no divergence
in overlapping content — each run's drops are distinct files. Until that is built,
the coordination rule is: Scribe checks `git log origin/main..HEAD` before committing;
if main has moved, rebase or merge first, then apply the compaction on top.

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

**Manifest lint reliability (PR #414):** A dispatch-manifest lint whose increment regex
does not handle rustfmt-wrapped counter increments is blind to those increments. A lint
without a `--self-test` mode that passes before the lint itself runs proves nothing when
it goes silent — a real dead branch looks the same as a rustfmt-wrapped live one. Wire a
self-test into CI before the lint check; the self-test must exercise both wrapped and
single-line increment patterns plus genuine dead-counter cases.

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
- `BNNSFilterApplyBatch` is unreliable for `BNNSFilterCreateLayerConvolution` filters
  (SIGSEGV inside libBNNS.dylib at batch>1, confirmed via AddressSanitizer on PR
  squad/fix-batch-segfault). Use per-image `BNNSFilterApply` until migration to `BNNSGraph`.

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
- Narrative entries compacted 2026-07-29 (two runs): `.squad/decisions-archive/2026-07.md`
  → "Narrative entries compacted by size gate — 2026-07-29T21:19:00Z" and
  "Narrative entries compacted by size gate — 2026-07-29T23:30:00Z".
- Prior active-ledger archives: `.squad/decisions/archive/`.
- Mac CPU EP load-bearing topics in the archive: PR #227 roofline lessons, load-adaptive
  opt-in, Apple Silicon portability directive, BNNS prefill/deprecation notes,
  benchmark-CI informational-only rule, dispatch-manifest lint, Sebastian/Deckard 1x1
  Conv correction, Iran SDPA model-ratio correction, and negative-result GEMV notes.
- Wave 8/9 topics in the archive: CUDA coverage batches 8/9, shape-inference catalog
  batches 3/4, NCHWc minimal-build gating, and strict reviewer-lockout correction cycle.
- 2026-07-29 narrative (archived): Mac CPU EP campaign summary, wheels.yml PR #401,
  DFT/Perch PR #357, KV Phase 1, BNNS batch>1 fix, declarative I/O PR #373, QMoE PR #378,
  decode ports PRs #380/#382, recurrent PR #386, JSON schema PR #388, tool-call parser
  PR #390, DeepSeek investigation, Fact-Checker KV verdict, Pris CI timing data, Luv
  round-3 review counts, tool/CUDA drops, Doug CUDA benchmark (pre-SiLU-fix), Cohaagen
  #377 verbose, Matthias static-cache implementation, Johnny PackedVarlenAttention,
  Pris multiturn corrected measurements.

## Compacted active directives — 2026-07-30T04:10:00Z

Details for the pre-161 CUDA wave, ORT 1.28 benchmark, CLI charter, CI, KV, REPL,
terminal-test narratives, explicit-metadata reconstruction, reasoning-fixture review, and
Scribe tidy-round notes were moved to `.squad/decisions-archive/2026-07.md` under the
2026-07-30 compaction entries. Live standing rules that remain binding:

- CLI work is a developer/maintainer tool investment; prioritize maintainer debug/iterate
  leverage, not consumer registry/pull/fine-tune features. The inline ratatui REPL remains
  the primary CLI surface; `/fork` and `/rewind` stay behind runtime capability gates.
- Full CI platform execution is the gate; uninstrumented fast jobs and package-scoped local
  commands do not substitute for the exact CI command.
- Native multi-turn performance claims must use the session-persistent KV API, not the old
  stateless path, unless explicitly labeled `--native-stateless`.
- Verification must fail on bad downloads or malformed archives; warnings are not checks.
- Model-declared generation defaults are canonical below explicit caller flags; engine
  constants are fallback, not override.
- All inference/pipeline metadata except io-shape must be explicit and general; name
  guessing is forbidden. Static-cache metadata must be declared, with `StaticCacheAbi::classify`
  authoritative and name-agnostic.
- Probe the stream being written, use PTY-driven tests for terminal behavior, and treat
  cfg-gated tests as unverified until they run on an admitted platform.
- `.squad/decisions/inbox/` is a tracked durable queue. Merge each in-scope drop, dedupe,
  delete processed files, and keep `inbox/README.md`.
- Do not delete a worktree before Scribe has merged its decision inbox.

### 2026-07-30: Add QLinearMatMul and common Resize CUDA parity
**By:** Kuato
**What:** CUDA now claims QLinearMatMul for Int8/Uint8 per-tensor and operand-axis quantization, plus Resize nearest/linear with half_pixel, align_corners, and asymmetric coordinates using scales or sizes. Cubic, pytorch_half_pixel, tf_crop_and_resize, half_pixel_symmetric, antialiasing, and non-stretch aspect policies remain fail-closed.
**Why:** These implementations match the CPU EP's integer accumulation/requantization and interpolation formulas while raising standard-domain CUDA parity from 157 to 159 without claiming unsupported Resize semantics.

### 2026-07-30: Add common ConvTranspose and GridSample CUDA geometry paths
**By:** Kuato
**What:** CUDA now covers 1-D/2-D ConvTranspose with explicit/VALID padding, strides, dilation, output padding, groups/depthwise, and optional bias, plus 4-D GridSample bilinear/nearest with zeros/border/reflection padding and both align_corners values. SAME auto-padding, output_shape-driven ConvTranspose geometry, cubic GridSample, and volumetric GridSample remain fail-closed.
**Why:** Output-owned NVRTC kernels match the CPU EP formulas without atomic accumulation nondeterminism, raising advertised CUDA coverage from 159 to 161 while refusing geometry modes not validated in this wave.

### 2026-07-30: Shape-aware CUDA claim gates for deferred ranks
**By:** Mary (revision to Kuato PR #424); reviewed by Lori
**What:** CUDA claim gates for rank-dependent operators must distinguish unsupported static shapes from deferred rank information. The #424 revision introduced a `require_input_rank` helper so ConvTranspose/GridSample claim only when rank is known and supported, while unknown/deferred rank preserves CPU fallback instead of falsely declining the graph.
**Why:** Fail-closed CUDA claims are correct only when the shape fact is known. Treating deferred rank as unsupported breaks heterogeneous fallback and can turn safe CPU execution into an over-eager CUDA refusal. Future CUDA claim gates with rank/shape predicates must be shape-aware and preserve CPU fallback for deferred facts.

### 2026-07-30: CUDA standard-domain parity reaches 161 covered ops
**By:** Squad CUDA parity wave
**What:** Merged #423 (`eed2fbf2`) for `QLinearMatMul` + common `Resize` and #424 (`1574e87a`, Mary revision `93d9e7b8`) for `ConvTranspose` + `GridSample`, raising advertised CUDA coverage from 157 to 161 ops. Lori approved #423 and, after requesting changes on #424, approved Mary's shape-aware correction with independent on-device evidence across 308 GPU parity cases.
**Why:** The tractable CUDA parity wave is now landed through 161 ops while retaining fail-closed behavior for unsupported Resize cubic/coordinate modes, ConvTranspose output-shape/SAME modes, and GridSample cubic/volumetric modes. Remaining heavy gaps are NonMaxSuppression and Resize-cubic.

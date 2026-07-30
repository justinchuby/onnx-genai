# Decisions — live standing directives

Last compacted: 2026-07-29T23:30:00Z

Full historical ledger archived to `.squad/decisions-archive/2026-07.md`:
- "Full decisions snapshot archived by size gate — 2026-07-28T11:30:55Z"
- "Post-rebase decisions archived by size gate — 2026-07-28T11:35:49Z"
- "Narrative entries compacted by size gate — 2026-07-29T21:19:00Z" (first run)
- "Narrative entries compacted by size gate — 2026-07-29T23:30:00Z" (merge resolution)

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

### 2026-07-30: CUDA op-parity + reduce-f16 wave merged
**By:** Squad (Coordinator) — requested by justinchuby
**What:** Merged #418 (SiLU-marker harden + Instance/GroupNorm), #419 (LpPool/CenterCropPad/Col2Im, CUDA 154→157), #420 (extended reductions widened to f16/bf16, f32-accum). Each independently opus-reviewed (Lori #419, Kuato #420) with on-GPU parity re-run. #67 progress comment posted (157 ops). #384 updated with Mary's large-model root-cause; Doug isolated the ORT-CUDA abort to an upstream graph-optimizer bug (extended/all crashes, basic loads) on both 1.27 and 1.28.
**Why:** Advances #67 CUDA parity and Justin's "beat ORT / support larger models" directive. #420 clears the native large-model reduce-fallback (96 FP16 ReduceSumSquare now CUDA-claimed); remaining native 27B blocker is explicit token metadata (#377 / mobius#434 pending Justin's merge).


### 2026-07-30: ORT 1.28 CUDA Qwen3.6-27B INT4 basic-opt reference
**By:** Doug

**What:** Tested the official ONNX Runtime 1.28.0 CUDA 13 Linux x64 release on
NVIDIA H200 GPU 7 against:
`/home/justinchu/mary-models/qwen3.6-27b-int4-cuda`.

ORT package:

- Asset: `onnxruntime-linux-x64-gpu_cuda13-1.28.0.tgz`
- Source: `https://github.com/microsoft/onnxruntime/releases/download/v1.28.0/onnxruntime-linux-x64-gpu_cuda13-1.28.0.tgz`
- Installed at: `/home/justinchu/onnx-genai/.ort-cuda-1.28/root`
- Release SHA-256: `84d28f27589090b280d4312743efd3d450cd4ac7d1e1d75e7d9076d9637bf9de` (verified)
- `VERSION_NUMBER`: `1.28.0`
- `GIT_COMMIT_ID`: `da9b5e364c465de65c49d91e696cd6485270757f`
- `libonnxruntime.so.1.28.0` SHA-256:
  `87097979b341c4df9c1bf71b14f7376f84a91206fbc64c0ccc4733dcbbab9e40`
- CUDA provider dependencies are CUDA 13 (`libcublas.so.13`,
  `libcudart.so.13`) and cuDNN 9, matching the current `.cudaenv.sh` library
  environment.

The ORT 1.27 session-init abort is **not cleared** by 1.28.0. The requested
benchmark aborts before engine construction, warmup, or generation:

```text
/opt/rh/gcc-toolset-14/root/usr/include/c++/14/bits/stl_vector.h:1130:
constexpr std::vector<_Tp, _Alloc>::reference
std::vector<_Tp, _Alloc>::operator[](size_type)
[with _Tp = onnxruntime::NodeArg*; ...]:
Assertion '__n < this->size()' failed.
```

Exit status is 134 (`SIGABRT`). The last optimizer warning is constant folding
`model/norm/Add_node_4200`. Peak GPU-7 memory observed before abort was only
527 MiB. Therefore there is no valid 27B ORT-CUDA tok/s number or generated text
to assess for coherence.

Requested repro:

```bash
cd /home/justinchu/onnx-genai
source .cudaenv.sh
export ONNX_GENAI_ORT_LIB="$PWD/.ort-cuda-1.28/root/lib/libonnxruntime.so.1.28.0"
CUDA_VISIBLE_DEVICES=7 taskset -c 7 ./target/release/profile_native \
  --model /home/justinchu/mary-models/qwen3.6-27b-int4-cuda \
  --ep cuda --backend ort --steady --tokens 128 --warmups 2 --runs 3 \
  --decode-skip 8 \
  --prompt 'Explain what a transformer is in two sentences.'
```

Minimal harness repro (same abort):

```bash
CUDA_VISIBLE_DEVICES=7 taskset -c 7 ./target/release/profile_native \
  --model /home/justinchu/mary-models/qwen3.6-27b-int4-cuda \
  --ep cuda --backend ort --steady --tokens 9 --warmups 0 --runs 1 \
  --decode-skip 8 --prompt x
```

A standalone ORT C++ session-creation repro establishes that this is in ORT's
optimizer, not generation or the Rust engine:

| ORT graph optimization level | CUDA session creation |
|---|---|
| disabled | succeeds |
| basic | succeeds |
| extended | aborts |
| all (project default) | aborts |

The same 1.28 library with the CPU EP also completes session construction; its
first generation then stops normally with the model package's ambiguous
`model.io.token_input` metadata error. This further isolates the abort to the
CUDA EP plus extended/all graph optimization.

#### Labeled workaround reference

**ORT-CUDA 1.28.0, graph-opt=basic (extended/all aborts upstream):**

| Metric | Result |
|---|---:|
| Steady median decode | **17.38 tok/s** |
| Median decode latency | 57.527 ms/token |
| Measured runs | 17.32, 17.57, 17.38 tok/s |
| Median prefill to first emitted token | 50.740 ms |
| Peak H200 VRAM | **18,127 MiB** |

Regime: H200 GPU 7, CPU core 7, 128 generated tokens, 2 warmups, 3 runs,
`decode_skip=8`, greedy decode, prompt `Explain what a transformer is in two
sentences.`.

Generated text was syntactically coherent and non-repetitive, but did not follow
the prompt; it began:

```text
---

### 1. Introduction

In this paper, we investigate the problem of finding the optimal control for a
system governed by a partial differential equation (PDE)...
```

This likely reflects the raw prompt/model-package chat-format or artifact
metadata, not token corruption. The output was deterministic across all three
measured runs.

Measurement required temporary, subsequently reverted scaffolding because the
artifact was not yet a complete runnable model package:

1. A temporary `ONNX_GENAI_ORT_GRAPH_OPT=basic` hook selected ORT optimization
   level 1.
2. `inference_metadata.yaml` was temporarily completed with the graph's explicit
   token, mask, position, logits, 32 growing KV, and 96 fixed recurrent-state
   port pairs.
3. The symbolic `batch` dimension on recurrent state ports was temporarily
   specialized to batch 1. Without this, the engine correctly refuses to
   zero-initialize fixed state with shape `[-1, 10240, 3]`.

Command:

```bash
source .cudaenv.sh
export ONNX_GENAI_ORT_LIB="$PWD/.ort-cuda-1.28/root/lib/libonnxruntime.so.1.28.0"
export ONNX_GENAI_ORT_GRAPH_OPT=basic  # temporary benchmark scaffold
CUDA_VISIBLE_DEVICES=7 taskset -c 7 ./target/release/profile_native \
  --model /home/justinchu/mary-models/qwen3.6-27b-int4-cuda \
  --ep cuda --backend ort --steady --tokens 128 --warmups 2 --runs 3 \
  --decode-skip 8 \
  --prompt 'Explain what a transformer is in two sentences.'
```

Current native CUDA status: it loads the artifact but does **not** execute on
CUDA. The CUDA EP declines 96 fp16 `ReduceSumSquare` nodes (`Float16 unsupported;
expected Float32`), heterogeneous CUDA+CPU placement is unavailable, and the
whole session falls back to `cpu_ep`. A one-token status run measured only
0.04 tok/s after a 274-second prefill, so it is not a native-CUDA reference.

The failing boundary is ORT's Level2/extended optimizer set: basic/Level1
succeeds, while extended/Level2 aborts. The exact individual Level2 transformer
was not isolated; candidates include the CUDA-specific Level2 attention,
normalization, gather/split, and QDQ selector/fusion transformations. This is
sufficiently narrow for an upstream issue without misattributing the assertion
to the last Level1 constant-folding warning.

**Why:** #384 now has a measured, explicitly qualified 27B-class ORT-CUDA
baseline while preserving the stronger finding that the normal ORT extended/all
configuration is unusable on this graph. The 17.38 tok/s figure is a workaround
reference, not equivalent to the normal project-default ORT `all` configuration.
No project ORT pin, Cargo file, environment example, or CUDA kernel was committed.


### 2026-07-30: Land the remaining tractable CUDA index and pooling operators
**By:** Kuato
**What:** Added CUDA `LpPool`, `CenterCropPad`, and `Col2Im`, raising `CUDA_COVERED_OPS` from 154 to 157. `LpPool` uses a general N-D NVRTC window reduction, while the two index transforms share one dtype-aware NVRTC module.
**Why:** All three operators have compact, model-agnostic GPU implementations and passed CPU-EP parity on GPU 3, including p=1/p=2 pooling geometry, odd mixed crop/pad, and overlapping/dilated Col2Im accumulation. This leaves the six heavier or data-dependent standard-domain gaps for focused waves.


### 2026-07-29: Harden decomposed SiLU and add CUDA normalization parity
**By:** Kuato
**What:** Standalone CUDA `Silu` now honors `_cuda_decomposed_silu`, retaining the explicit fp16 sigmoid and multiply rounding when the downstream SwiGLU fusion does not fire. Added CUDA `InstanceNormalization` and `GroupNormalization` (opsets 18 and 21) for contiguous f32/f16/bf16 NCHW-style tensors, with GPU-vs-CPU conformance across all three dtypes, both GroupNormalization affine contracts, and a large-offset variance case. `CUDA_COVERED_OPS` rises from 152 to 154.
**Why:** The SiLU marker previously affected only the fused path, leaving a correctness hole in the standalone fallback. Normalization was the cleanest next #67 batch because both operators share one model-agnostic two-pass NVRTC reduction/affine implementation. Deferred `LpPool` until its full arbitrary-rank window/auto-pad/ceil-mode contract can be implemented rather than shipping a rank-4 subset; deferred `CenterCropPad` and `Col2Im` to a focused index-transform batch; deferred `QLinearMatMul` because per-axis quantization, batched broadcasting, integer accumulation, and output requantization do not fit cleanly into the existing float dequant/GEMM path without a larger design.
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

## CI: run tests on every platform; instrument for coverage only where informative

**By:** Pris (2026-07-28)

Full coverage required on PRs. A parallel uninstrumented Linux fast job (5–9 min
warm/cold) offers early feedback but never substitutes for the full gate. Windows ARM64
retains tests/clippy but not llvm-cov coverage (duplicates x64/macOS while owning the
CI critical path — up to 26 min). Platform execution is the signal; instrumentation is
the cost. Current critical path: `CLI ORT (Windows x86_64)` at ~18m50s.

## Native backend multi-turn: session-persistent KV closes the structural gap for headline models

**By:** Pris (benchmarked 2026-07-28; corrected with session API 2026-07-29, PR #408)

Without persistent KV each turn re-prefills the entire conversation O(context). Phase 1
landed (Deckard, squad/session-kv-phase1); the multiturn benchmark was then wired to the
session API. **Corrected results with KV reuse (Pris, M1 Max):**

- **Qwen2.5-0.5B-f16 (session):** Native wins at every turn — 1.13x faster over 10
  turns; native TTFT 60 ms vs ORT 150 ms. Roy's Phase 1 hypothesis confirmed for the
  headline model.
- **TinyStories-33M (session):** Break-even at turn 1–4; ORT 1.5–1.7x faster overall.
  Native TTFT faster (21 ms vs 27 ms ORT), but decode throughput is 2x slower — that is
  now the bottleneck, not prefill.
- **Stateless (old path, --native-stateless):** Qwen break-even at turn 8, ORT 1.18x;
  TinyStories break-even at turn 3, ORT 2.2x. These are the pre-Phase-1 numbers.

Batch>1 native crash fixed separately (Resch, squad/fix-batch-segfault). Pre-packing
deferred — re-evaluate after decode throughput gap is addressed.

## 2026-07-29 — Verification steps that warn instead of fail are not verification

**By:** Holden; recorded by Scribe; PR #401

A verification step that warns instead of failing is not verification. The ORT
`osx-x86_64` download had no pinned checksum, so `verify_archive_checksum` warned and
returned; `curl` exited 0 on a 404; three "successful" steps produced a corrupt input
before `tar` objected.

**Rules:** Check HTTP status explicitly (`curl -f` or `-w %{http_code}`); validate
archive magic bytes (gzip `1f 8b` / zip `PK`) before extracting; every check that only
warns on failure is a silent failure waiting for the worst moment.

## 2026-07-29 — Model-declared generation defaults are canonical; our constants are fallback, not override

**By:** Leon; PRs #385, #392; closes issues #290, #296

Model `do_sample`, `temperature`, `top_p`, `top_k` were parsed then discarded, forcing
greedy — on models that ship explicit values precisely because greedy degenerates (e.g.,
DeepSeek, Qwen). **Precedence is now strict:** explicit caller flag > model-declared value >
greedy fallback. Enforcement is in the engine so CLI, server, and Python all inherit it.

## 2026-07-29 — The CUDA driver API ships with the display driver, not the toolkit

**By:** Leon; clears PR #395 misconception

`nvcuda.dll` is present whenever the GPU driver is installed. Both `cust` and `cudarc`
fall back `*_13` → `*_12` cleanly.

**Rule:** An inferred capability claim that turns out to be wrong costs more than the
investigation it prevented. Never suppress an inferred bad fact without verifying it
directly against a fresh environment.

## 2026-07-29 — Probe the stream you are writing to

**By:** Rachael; PR #372

`emit_stats_line` chose its format from `io::stdout().is_terminal()` while writing to
stderr — inverts under redirection. **Rule:** always probe the destination stream, never a
convenient nearby one. Stats to stderr → test `stderr().is_terminal()`, not stdout.

**Durable pattern:** Extract testable pure functions. `stats_text()` and
`needs_trailing_newline()` had to become named pure functions before their branches could
be unit-tested; inlining had hidden both defects in the production path.

## 2026-07-29 — Run the gate command CI runs, verbatim

**By:** Rachael; PR #372

A package-scoped `cargo fmt --check` passed locally while the entire workspace was
unformatted. CI's `cargo fmt --all --check` caught it in 37 seconds. **Rule:** use the
exact gate command that CI uses, not a convenient subset. Local validation with a narrower
scope is structurally untrustable.

## 2026-07-29 — Terminal behaviour requires PTY-driven tests; piped I/O cannot cover it

**By:** Rachael; PR #372

Two `#[cfg(unix)]` PTY tests written on Windows compiled to zero tests on Windows and
were reported alongside a green 168-test suite. **Rule:** A `cfg`-gated test is unverified
until it runs on a platform where the gate admits it. Do not assume compilation equals
verification. Piped-stdio tests structurally cannot cover PTY-specific behavior (control
sequences, window size events, terminal probe responses).

## 2026-07-29 — Type-ahead is not lost during generation on Unix or Windows

**By:** Zhora; closes issue #298; PR #393

ConPTY does not echo type-ahead during generation at all, so no REPL repaint can overwrite
it. This is a terminal characteristic, not a backend bug. **Do not re-open as a
native-backend issue.** Still unverified: native conhost (not IDE or ConPTY) mid-stream
type-ahead, which may be cosmetic echo overpaint or real loss — test in a native terminal.

## 2026-07-29 — PTY-harness hazards that present as swallowed input

**By:** Zhora; PR #393

A **0×0 window** makes `ratatui::insert_before_no_scrolling_regions()` infinite-loop.
Feeding `\n` where a terminal sends `\r` means the line never completes. Both present as a
hang or lost input and send the next investigator toward the wrong cause.

## 2026-07-29 — Prefer an idle timeout to a total one when the child is silent before first output

**By:** Zhora; PR #393

A 30s total drain budget lost ~48s to a cold model load; it was a coin flip, not a safety
net. Rule: 120s idle — justified at ~2.5× the measured worst case — with a failure message
stating it timed out waiting for bytes, not a trailing-newline defect. An empty read can
never masquerade as an assertion failure.

## 2026-07-29 — Standing Operational Rule: Worktree lifecycle and decision merging

**By:** Justin Chu (recorded by Scribe)

Do not delete a worktree before Scribe has merged its decision inbox. `.squad/decisions/inbox/` is gitignored and per-worktree, so removing a worktree destroys any unmerged decision drops in it. Coordination workflow:

1. Agent writes decision to `.squad/decisions/inbox/{agent}-{slug}.md` (in-worktree, gitignored).
2. PR lands; worktree remains temporarily.
3. Scribe runs (either same session or next): merges inbox → `.squad/decisions.md`, deletes merged files.
4. Scribe commits and pushes.
5. Safe to delete the worktree.

Merging and deleting the inbox files produces no git diff (expected, not a failure). The loss that occurred here: Rachael, Zhora, Leon wrote inbox files in separate worktrees, those worktrees were deleted before Scribe ran, and inbox files were lost. The substance survived in merged `history.md` files and PR descriptions, but the durable-rule fragments did not make it to this ledger — they had to be manually recovered from context.

## 2026-07-29 — All inference/pipeline metadata must be explicit; name guessing is forbidden

**By:** Justin Chu directive #377; Cohaagen, Benny, Melina, Matthias (PRs #380, #382, #377/`squad/377-explicit-metadata`, #412)

ALL inference/pipeline metadata except io-SHAPE must be EXPLICIT and GENERAL. Name
guessing/historical-name fallback must be replaced by explicit metadata plus a clear ERROR
naming the missing key. Only io-SHAPE may disambiguate. Do not re-propose deferral.

**Active schema fields (Benny/mobius: emit these names verbatim):**
- `pipeline.strategy.inner_embedding_output: Option<String>` — nested-AR inner decoder embedding output port. Absent ⇒ ERROR.
- `model.io.static_cache: Option<StaticCacheIoSpec>` — `write_indices_input`, `kv_sequence_length_input`, per-layer `key/value_cache_inputs/outputs` (equal-length, positional). Inconsistent ⇒ ERROR. **Must be declared; convention-based binding removed (Matthias, PR #412).**
- Encoder prompt-input role: from `model.encoder.inputs.audio_features` vs `.input_ids`; no port-name string matching.
- Paged-KV bridge geometry: from `model.io.kv_inputs`/`kv_outputs` only; no metadata ⇒ `Ok(None)`.

**Matthias (PR #412):** `detect_static_cache_by_convention` removed from
`crates/onnx-genai-ort/src/decode/io.rs`. A TensorScatter static-cache graph without a
declared `model.io.static_cache` block now fails closed with an error naming the missing
key — never silently binds by hardcoded `write_indices`/`nonpad_kv_seqlen`/`key_cache.{i}`
port names. `StaticCacheAbi::classify` stays authoritative and name-agnostic. In-repo
static-cache fixtures now declare the block explicitly.

**Remaining name path (off-limits scope):** `decode_contract.rs` `KvNamingConvention` — only for #99 speculative proposers; do not remove from this workstream.

## 2026-07-29 — Warmup: shared registry method; error categories preserved at admin boundary

**By:** Lull + Rachael; PR #407

Opt-in `warmup` per-model setting and `POST /v1/admin/models/{id}/warm` both use
`ModelRegistry::warmup` — one deterministic generated token, idempotent. **Rules:**
- Share the registry method so configured and on-demand warmups are identical; failures
  can be retried without corrupting registry state.
- Return typed errors (absent-model, registry, runtime-failure); the admin endpoint maps
  them 404 / 500 / 500. A loaded model's failed warmup must not be reported as a 404.

## 2026-07-29 — Reasoning fixture review: reconstructed durable rules (PRs #410, #411)

*Reconstructed from session context, 2026-07-29. Authored in a worktree whose inbox was deleted before Scribe ran — the same mistake recorded in the standing operational rule above. Content reflects authors' stated positions; provenance is session context, not merged inbox drops.*

**A fixture whose every assertion is "the turn was dropped" cannot distinguish correct behaviour from total breakage.** Iteration 1 had no reachable `</think>`, so no input produced a committed answer; a regression dropping every turn would have passed. Fixed by making the close reachable on three prompts. — *coordinator (PR #410)*

**Assert on what the code did, not on a summary of what it should have done.** Every sampling test keyed on the `/session` display, which resolved sampling independently of generation. Gaff commented out `resolve_sampling_defaults` — recreating the #385/#392 forced-greedy bug — and the suite stayed green. The tests pinned a summary line, not a policy. — *Gaff, REJECT (PR #410)*

**Run a new test in isolation before believing it.** The round-2 token-stream test had one green in the full parallel suite; Luv ran it alone: 15/15 failures with the fix intact. The "8/8 distinct outputs" evidence was a stderr-timestamp artifact — the test compared stdout only. — *Luv, REJECT (PR #411 round 2)*

**A near-deterministic fixture cannot witness sampling through its tokens.** At `temperature 0.6, top_k 20`, decode is effectively greedy: 80/80 no-flag runs byte-identical to the greedy stream. The token-stream assertion was ~95% false-fail, not false-pass. Raising run count or picking a seed does not rescue it. — *Luv*

**Instrument the boundary you care about.** Surface the sampling policy generation actually resolved into `--stats`/`--profile` and assert on that: deterministic, no subprocess cost, observes the real object. Under mutation the stats line and `/session` visibly disagreed — only possible if stats reads the generation path, not the display path. — *Leon, approved by Luv*

**Two independent resolution sites for one policy is the defect, not an inconvenience.** A summary that can disagree with what generation did is a bug waiting to be re-reported. Resolved by one helper called by both `/session` and every turn, reading the live backend on demand — no cache, no staleness across `/reload`/`/ep`/`/backend`. — *Leon*

**Close a gap by construction rather than by comment where you can.** Asked for a warning comment about a future refactor re-resolving options after the capture point, Leon instead moved the capture inside the consuming function, reading the exact struct passed to `backend.generate` — no window between capture and use. — *Leon, delta approved by Luv*

**A committed turn with an empty answer poisons context exactly as an unclosed one does.** `quick --greedy --max-new-tokens 3` stopped on `</think>` and committed an empty assistant turn; the commit path was unconditional on non-emptiness while `manifest.json` asserted the invariant. Fix: closed path now drops whitespace-only answers with a diagnostic distinct from "stopped inside reasoning". An overstated invariant is worse than an absent one. — *rubber-duck, confirmed by Gaff, fixed by Batty*

**A checked-in fixture must be reproducible from its generator.** `manifest.json` was corrected directly while the generator's embedded description string was not; regenerating would have silently reverted the correction. Found during merge-conflict resolution. — *Leon*

**Reviewer depth paid for itself.** Copilot found a stale comment; rubber-duck raised the doubt; Gaff proved it by mutation; Luv disproved the fix by running it in isolation; Leon's third attempt was approved only after Luv independently re-verified, twice. With Copilot review alone this would have merged a test that caught nothing — while existing solely to catch that bug. — *coordinator*

**Scribe repeat failure — worktree deletion before inbox merge (second occurrence 2026-07-29).** The standing operational rule was correct; it was not applied. Consider whether the safeguard should be procedural rather than a remembered rule — e.g. Scribe runs before any worktree removal, or drops are written to the main checkout from the start. Lost: `gaff-review-reasoning-fixture.md`, `batty-reasoning-fixture-revision.md`, `leon-reasoning-fixture-round3.md`, `pris-tiny-reasoning-fixture.md`. — *Justin Chu / Scribe*

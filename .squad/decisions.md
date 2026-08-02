# Decisions — live standing directives

Last compacted: 2026-07-30T04:30:00Z (Scribe tidy round 2, merged with #427 at 04:10:00Z)

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
- Round-2 compaction (2026-07-30): `.squad/decisions-archive/2026-07.md` → "Narrative
  entries compacted by size gate — 2026-07-30T04:30:00Z (Scribe tidy round 2)". Holds the
  archived CUDA op-parity wave changelog (Kuato/Doug: SiLU/Instance/GroupNorm 152→154,
  LpPool/CenterCropPad/Col2Im 154→157, QLinearMatMul/Resize 139→141, the merged inbox drop
  ConvTranspose/GridSample 159→161, and Doug's ORT-1.28 27B INT4 basic-opt 17.38 tok/s
  workaround reference / extended-opt SIGABRT), the native multi-turn KV benchmark record
  (Pris #408), the PTY-harness technique notes (Zhora #393), and the reasoning-fixture
  review narrative (#410/#411). Standing CUDA/CLI/test rules stayed live.
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

## CUDA standard-domain parity — current state (through 161 ops)

Kept live from PR #427's `## Compacted active directives — 2026-07-30T04:10:00Z` section
(merge round 2). PR #427 folded the CLI/CI/KV/verification/metadata/probe/PTY/inbox/worktree
rules into one-line bullets and moved their full text to the archive; this merge instead keeps
the fuller live sections below (a rule kept live governs at spawn; a wrongly-archived rule does
not), and carries #427's CUDA records — the part it kept live — here. These are the current CUDA
op-parity records; the earlier pre-161 wave narrative (SiLU/Instance/GroupNorm 152->154,
LpPool/CenterCropPad/Col2Im 154->157, and Doug's ORT-1.28 27B INT4 benchmark) is archived
under the 2026-07-30 compaction entries in `.squad/decisions-archive/2026-07.md`.

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

## 2026-07-29 — Native multi-turn performance claims must use the session-persistent KV API

**By:** Pris (PR #408); kept live from PR #427's consolidation (merge round 2)

Native multi-turn performance claims must use the session-persistent KV API, not the old
stateless path, unless explicitly labeled `--native-stateless`. The full benchmark record
(Qwen2.5-0.5B-f16 and TinyStories-33M, M1 Max) is archived under the 2026-07-30 compaction
entries.

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

## 2026-07-29 — Decision inbox is a tracked durable queue (not gitignored scratch)

**By:** Scribe (tidy round 1)

**What:** `.squad/decisions/inbox/` is now **tracked in git** rather than ignored.
`.gitignore` no longer lists it; `inbox/README.md` keeps the directory present and
documents the semantics. The charter's responsibility #2 is updated to match.

**Why:** The inbox was gitignored in the 2026-07-12 "Add squad" commit, grouped with
runtime scratch (logs, sessions) on the assumption that drops are consumed and cleared
on the same machine that wrote them. With worktrees deleted before Scribe merges, and
three teams writing decision logs concurrently on separate machines, that assumption is
false — drops that live only locally are lost on worktree deletion (four records lost on
2026-07-29, others earlier) and drops on other machines are invisible here until merged.
Tracking fixes all three: drops survive worktree deletion, are visible in-flight across
machines, and concurrent Scribes each add distinct files that git merges without
conflict (the only overlap, merge-and-delete, is a clean delete/delete). This is the
structural version of the fix already noted under "Concurrent Scribe runs are a
structural hazard": assemble `decisions.md` from inbox drops instead of hand-merging.

**Costs accepted (honest accounting):** more churn in git history; drops now appear in
PR diffs where they previously did not; and we deliberately override a line that had
been in place since the initial squad scaffolding. That line was not a considered
inbox-specific decision — it was a bulk "ignore runtime state" grouping — and its one
real premise (drops are transient and cleared) still holds: Scribe still deletes drops
after merge. Only the durability assumption was wrong, so tracking is compatible with
the original intent rather than a reversal of a deliberate choice.

## Testing discipline — standing rules (distilled from reasoning-fixture review, PRs #410/#411)

The full review narrative is archived (round-2 compaction). These are the binding rules:

- **Assert on what the code did, not a summary of what it should have done.** Tests that
  key on a display/summary line (e.g. `/session`) can stay green while the real path
  (`resolve_sampling_defaults`) is broken. Instrument the boundary you care about and
  assert on that — surface the resolved policy into `--stats`/`--profile` and check it.
- **Run a new test in isolation before believing it.** A single green in a full parallel
  suite can be a stderr-timestamp/interleave artifact; run it alone (a real fix survived
  15/15 isolated runs that the suite hid).
- **A fixture whose every assertion is "the turn was dropped" cannot distinguish correct
  behaviour from total breakage.** Make the success path reachable so a regression fails.
- **A near-deterministic fixture cannot witness sampling.** At low temperature/top_k decode
  is effectively greedy; a token-stream assertion is ~95% false-fail — raising run count or
  seeds does not rescue it. Assert on the resolved policy object, not sampled tokens.
- **One policy resolved at two sites is the defect.** A summary that can disagree with what
  generation did is a latent bug; resolve once via a shared helper both paths call, reading
  the live backend on demand (no cache, no staleness across `/reload`/`/ep`/`/backend`).

## 2026-07-30 — Scribe tidy round 3: what to check next

**By:** Scribe (tidy round 2, post-merge)

Round 2 compacted `decisions.md` from 40,477 → ~30 KB and then had to **merge a concurrent
compaction**: PR #427 ("consolidate CUDA parity 161 state") rewrote the same file down to
14,913 bytes in the same minutes, while I was pushing. Both were merged by property (union of
standing rules live; disagreements resolved live). **The concurrent-Scribe collision is now
observed, not predicted — the window was minutes, not days.** Two conflicting resolutions of a
single hand-merged file is exactly the failure the tracked inbox was meant to remove but does
not: the inbox stopped drop-loss, not the hand-merge rewrite. For round 3:

- **Adopt the round-2 Task-D recommendation.** The cheap fix is structural: keep the live file
  for **standing rules only** (rarely edited, low collision) and route **all dated wave/
  changelog records** through inbox drops that Scribe files **straight into the monthly
  archive, never into the live file**. #427 kept four CUDA wave records live; round 2 archived
  the equivalents — that divergence *is* the collision. If wave records never enter the live
  file, two Scribes cannot disagree about them.
- **`decisions.md` size / dedup.** Re-measure against tip first. When two compactions race,
  reconcile by property, keep every rule either side kept live, and dedupe the archive —
  build it as `base + one clean appendix`, never by concatenating two rewritten tails.
- **Inbox.** Still tracked. Merge every `*.md` except `README.md`; dedupe against **both** the
  live file and the archive — another team's Scribe may have merged the same drop on another
  machine (this round #427 merged `kuato-cuda-parity-4.md` live while I archived it; I had to
  match prose because drops carry no team/machine attribution). Add attribution to drops.
- **Histories.** Sweep all `.squad/agents/*/history.md` against the chronicle gate (>8 dated
  entries, or oldest live entry predating the previous wave measured against that file's newest
  entry — never against today). deckard/roy re-accumulate fastest.
- **Do not archive agent directories.** Fail-closed; with three teams active elsewhere,
  absence of local commits/drops/history proves nothing.


### 2026-07-30: DeepSeek and GLM native-CUDA correctness bring-up
**By:** Mary

**What:** Tested the current `main` (`1574e87a`) native CUDA decode path on GPU 0
against ORT-CUDA with greedy decoding. The original export directories do not
contain explicit `model.io` declarations, so direct loading fails before
placement with the ambiguous `input_ids`/`attention_mask` token-input error. I
created external diagnostic package overlays under
`/home/justinchu/mary-model-overlays/` that symlink the unchanged model data and
add only explicit token, mask, position, logits, and KV port names.

| Model | Native CUDA | Native tokens match ORT | Correctness verdict / blocker |
|---|---|---|---|
| DeepSeek-Coder-1.3B INT4 | Yes | Yes, exact 64/64 | Coherent: `Paris... Germany is Berlin... Italy is Rome...`; supported path is clean. |
| DeepSeek-R1-Distill-Qwen-1.5B INT4 | Yes | No; first divergence at generated token 8 (zero-based index 7), native `374` vs ORT `315` | Both start `" **C iter**. The capital..."`, then native and ORT repeat differently. Native CPU exactly matches native CUDA for the first 16 tokens, so this is not isolated to CUDA arithmetic. The strongest discriminator is the native `GroupQueryAttention` grouped-head/non-interleaved-rotary decode path (`num_heads=12`, `kv_num_heads=2`, `do_rotary=1`, `rotary_interleaved=0`) or shared native KV plumbing. The MHA-shaped Coder graph (`16/16`) is exact. Needs a focused GQA/KV parity issue. |
| GLM-4-9B INT4 | Yes | Unavailable | Native output is coherent and answers Paris, then explains the answer in Chinese. ORT 1.27 and 1.28 both reject the authored Microsoft `GroupQueryAttention` attribute `rotary_embedding_dim`; therefore ORT cannot provide a token oracle for this export. Native uses grouped/interleaved rotary (`32/2`, `rotary_interleaved=1`, `rotary_embedding_dim=64`). |
| DeepSeek-V2-Lite INT4 | Yes | Yes, exact 64/64 | Coherent answer begins `Paris. The currency of France is the Euro...`. All 26 `QMoE` nodes run on native CUDA; QMoE/expert execution is not a correctness blocker. |
| DeepSeek-V2-Lite block-32 INT4 | Yes | Yes, exact 16/16 | Coherent short run; confirms the alternate block-32 package too. |

The native V2-Lite graph contains 26 each of `QMoE`, `TopK`,
`GatherElements`, and `ScatterElements`; it completed without a CUDA claim
decline or whole-graph CPU fallback. This does not prove live expert paging, but
it clears `QMoE` itself as the E2E correctness blocker for this artifact.

**Why:** This establishes broad native-CUDA viability before performance work.
Three independently useful model/package variants match ORT exactly, GLM is
coherent but lacks an ORT-compatible oracle, and R1 isolates one remaining
native decode parity gap to grouped non-interleaved GQA/KV behavior rather than
CUDA-specific numerical execution.

Representative command (replace model and backend):

```bash
cd /home/justinchu/onnx-genai
source /home/justinchu/onnx-genai/.cudaenv.sh
export ONNX_GENAI_ORT_LIB="$ORT_ROOT/lib/libonnxruntime.so.1.27.0"
cargo build --release -p onnx-genai-bench --features bench-native,bench-ort,cuda
CUDA_VISIBLE_DEVICES=0 taskset -c 0 ./target/release/profile_native \
  --model /home/justinchu/mary-model-overlays/coder \
  --ep cuda --backend native --steady --tokens 64 --decode-skip 8 \
  --warmups 0 --runs 1 --prompt 'The capital of France is'
```

Use `--backend ort` for the oracle. GLM was also retried with
`/home/justinchu/onnx-genai/.ort-cuda-1.28/root/lib/libonnxruntime.so.1.28.0`
and failed on the same unrecognized GQA attribute.


### 2026-07-30: DeepSeek-V2-Lite MoE offload correctness and wiring status
**By:** Mary

**What:** Ran greedy native-CUDA A/B decoding on the 26-QMoE
DeepSeek-V2-Lite INT4 package using GPU 0 and the same 64-token prompt as the
resident bring-up. All tested environment variants returned exactly the same
64 token IDs:

| Variant | Settings | Baseline token equality | Did paging fire? |
|---|---|---:|---:|
| Resident baseline | `ONNX_GENAI_WEIGHT_OFFLOAD=0` | Reference | No |
| Offload enabled, no owned warm cache | `ONNX_GENAI_WEIGHT_OFFLOAD=1`, `ONNX_GENAI_WEIGHT_OFFLOAD_HOST_BYTES=0` | Yes, exact 64/64 | No |
| Aggressive small/serial variant | `ONNX_GENAI_WEIGHT_OFFLOAD=1`, `ONNX_GENAI_WEIGHT_OFFLOAD_HOST_BYTES=1048576`, `ONNX_GENAI_WEIGHT_OFFLOAD_PREFETCH=0` | Yes, exact 64/64 | No |

The matching variants are a configuration no-op on native CUDA, not evidence
that expert eviction/reload is correct. Current controls and defaults are:

- `ONNX_GENAI_WEIGHT_OFFLOAD`: exact value `1` enables the route-first
  mmap-backed expert path; absent/other values mean disabled. This is consumed
  by the **CPU QMoE kernel**.
- `ONNX_GENAI_WEIGHT_OFFLOAD_HOST_BYTES`: unsigned decimal byte override for
  the CPU warm-host expert cache. Without it, the engine governor's resolved
  host-RAM budget is used; its default resource limit is 25% of detected host
  RAM.
- `ONNX_GENAI_WEIGHT_OFFLOAD_PREFETCH`: expert prefetch defaults ON; exact `0`
  forces the serial CPU route-first loop.
- CLI `--vram-limit` defaults to 90% of detected VRAM and `--host-ram-limit`
  defaults to 25% of detected host RAM. These feed the resource governor/KV and
  CPU host cache, but do not activate CUDA expert paging. `profile_native` does
  not expose either flag.
- `LocalTieredConnector` is KV paging, not immutable expert-weight paging.
  Defaults: 1024 hot GPU pages, 16384 total cached pages, 64-KiB accounting per
  page, and no compression or disk tier.

Static wiring confirms why no paging fired:

1. Native CUDA constructs the stock `CudaExecutionProvider`; only the CPU
   device path receives the engine's `WeightOffloadHostCache`.
2. The CUDA EP does not advertise `NXRT_WEIGHT_PAGING_CAPABILITY`, so
   `build_lazy_weight_handles` returns no handles.
3. The executor's only lazy boundary is
   `pkg.nxrt::BlockQuantizedMoE`; this model uses 26
   `com.microsoft::QMoE` nodes.
4. `CudaWeightPager` exists and has isolated GPU byte-identity tests, but its
   own module documents live executor/BlockQuantizedMoE dispatch, multi-page
   LRU eviction, and prefetch overlap as deferred.

Therefore all QMoE expert initializers are still eagerly uploaded and remain
resident. There is no forced-eviction/small-device-budget knob capable of
making this DeepSeek-V2-Lite CUDA run page experts today, and no live paging
counter can advance on this path.

**Why:** The token-equality checks show that merely setting the advertised
offload variables does not perturb resident CUDA correctness, but #82/#63
cannot yet be validated end-to-end on this QMoE model. The actionable gap is
CUDA capability + executor dispatch integration, including either a lazy
`com.microsoft::QMoE` boundary or conversion to the intended
`pkg.nxrt::BlockQuantizedMoE` representation, followed by a bounded VRAM
residency manager with observable page-in/eviction counters.

Reproduction:

```bash
source /home/justinchu/onnx-genai/.cudaenv.sh
export ONNX_GENAI_ORT_LIB="$ORT_ROOT/lib/libonnxruntime.so.1.27.0"
ONNX_GENAI_WEIGHT_OFFLOAD=1 \
ONNX_GENAI_WEIGHT_OFFLOAD_HOST_BYTES=0 \
CUDA_VISIBLE_DEVICES=0 taskset -c 0 \
/home/justinchu/onnx-genai/target/release/profile_native \
  --model /home/justinchu/mary-model-overlays/v2 \
  --ep cuda --backend native --steady --tokens 64 --decode-skip 8 \
  --warmups 0 --runs 1 --prompt 'The capital of France is'
```


### 2026-07-30: DeepSeek-R1 GQA decode resolution

**By:** Mary

**What:** Native R1 decode is correct: native CPU, native CUDA, and ORT CPU select token 374 with the f32 oracle margin. ORT CUDA's fp16 MatMulNBits near-tie flips generated token 7 to 315 and leads to repetition. CI regression coverage now exercises 12:2 grouped-query attention with non-interleaved rotary and multi-step KV decode at head widths 64 and 128.

**Why:** The width-128 case mirrors the deployed R1 graph, while width 64 retains coverage for the originally reported geometry. Both guard rotary position advancement, 6:1 head grouping, causal attention, and chained past/present KV correctness.


### 2026-07-30: Qwen 27B Unsqueeze blocker is already resolved
**By:** Mary
**What:** The reported native Unsqueeze rank failure does not reproduce on main. The real Qwen3.6-27B INT4 graph's only Unsqueeze maps rank-2 `position_ids` through constant axes `[0]` to the declared rank-3 output, and native CUDA proceeds past it. Existing dynamic, symbolic, negative-axis, legacy-attribute, modern-input, CPU, and CUDA tests all pass. Commit `8d9d2fa2` previously added the generic dynamic Unsqueeze runtime-shape fix. With explicit model I/O and recurrent-state metadata supplied, the next native load blocker is CUDA decoder state allocation rejecting rank-3 FP16 `past_key_values.12.conv_state` at `crates/onnx-genai-engine/src/native_decode/cuda.rs:480`; it currently accepts only rank-4 KV/state tensors.
**Why:** A new Unsqueeze patch would duplicate existing coverage and risk regression. The next #384 work should generalize CUDA persistent decoder state bindings to fixed recurrent states of arbitrary declared rank, beginning with rank-3 convolution state, while preserving rank-4 KV behavior.


# Decision: Foundry Local native CUDA EP vs ORT decode baseline

**Author:** Cohaagen (perf)
**Date:** 2026-07-27
**Scope:** Baseline measurement — informational, no code change.

## Headline finding

On H200 (ORT 1.27.0, CUDA int4, greedy steady-state decode), the **native CUDA EP
is faster than ORT on every Foundry Local target measured** — there is currently
no native-slower case to optimize:

| Model | Native tok/s | ORT tok/s | Ratio |
| --- | --- | --- | --- |
| qwen2.5-0.5b | 995 | 580 | 1.72× |
| qwen2.5-1.5b | 720 | 438 | 1.64× |
| qwen2.5-7b   | 308 | 272 | 1.13× |
| Phi-4-mini   | 315 | 231 | 1.36× |

Native's lead shrinks as the model grows (1.72× → 1.13×), consistent with large
models becoming GEMM/bandwidth-bound where both backends share the same int4
matmul kernels; native wins on per-step launch/scheduling overhead, which matters
most on small models.

Correctness: native/ort token-identical on 0.5B, 7B, and Phi-4-mini; 1.5B diverges
after sentence 1 (expected greedy sensitivity) but both coherent. **No repetition /
spacing / garbage artifacts on any model.**

## Operational note (worth propagating)

`.cudaenv.sh` does **not** set `ONNX_GENAI_ORT_LIB`, despite tasking docs implying
it does. Without it, `profile_native --backend ort` silently loads the CPU-only
prebuilt ORT and errors that `CUDAExecutionProvider` is unavailable. ORT-EP CUDA
benchmarks must export
`ONNX_GENAI_ORT_LIB=$ORT_ROOT/lib/libonnxruntime.so.1.27.0`.

Full report: `docs/benchmarks/2026-07-27-foundry-native-vs-ort-cuda.md`
(PR branch `squad/perf-foundry-native-vs-ort`).


### 2026-07-30: Generalize native CUDA persistent recurrent-state bindings
**By:** Mary
**What:** Native CUDA now distinguishes metadata-declared fixed `state_pairs` from growable KV pairs. Growable rank-4 KV retains its existing capacity-bucket and logical sequence-axis behavior. Fixed recurrent state is allocated at its declared rank and static geometry, batch is bound to one, storage is zero-initialized, and the state is excluded from KV bytes-per-token accounting and capacity growth. The Qwen3.6-27B INT4 graph now clears the former rank-3 FP16 `conv_state` allocation failure and reaches its next blocker: unsupported rank-3/1-D CUDA Conv.
**Why:** Hybrid attention/recurrent decoders carry fixed convolution and recurrent tensors that use replace semantics rather than sequence growth. Treating every past/present pair as rank-4 KV incorrectly applied axis-2 capacity growth to arbitrary declared state.

### 2026-07-30: Add native CUDA rank-3 Conv1D
**By:** Mary
**What:** CUDA `Conv` now handles rank-3 NCL tensors with an output-owned NVRTC kernel while preserving the existing rank-4 cuDNN path. The rank-3 path supports f32/f16/bf16, optional bias, groups/depthwise convolution, stride, dilation, explicit asymmetric/causal padding, and ONNX `VALID`/`SAME_UPPER`/`SAME_LOWER` geometry. GPU parity against the CPU EP covers basic, depthwise causal FP16, and grouped strided/dilated cases. The Qwen3.6-27B INT4 native CUDA probe clears `__fn0_Conv_node_12` and proceeds to the next blocker: missing inferred shape for the layer-0 linear-attention `Silu` output.
**Why:** Hybrid/recurrent LLM blocks use depthwise causal Conv1D over their fixed convolution state. The CUDA kernel previously accepted only rank-4 NCHW tensors and failed the real 27B graph after recurrent-state allocation was fixed.

### 2026-07-30: Register Microsoft Silu with unary shape inference
**By:** Mary
**What:** Added `com.microsoft::Silu` version 1 to the shared shape- and dtype-preserving unary inference catalog. Static, symbolic, rank-agnostic, unknown-rank, dtype, and since-version behavior are covered by unit tests. The Qwen3.6-27B source graph contains `com.microsoft::CausalConvWithState`; native lowering exposes its activation as `com.microsoft::Silu`, whose output previously remained untyped. With the rule registered, the real native CUDA probe clears the Silu shape failure and reaches the next blocker: `internal executor error: value#1414 not produced`.
**Why:** SiLU is elementwise and must preserve its input tensor's complete symbolic geometry and dtype. Routing it through the existing generic unary rule fixes the whole contrib-op class without model-specific shape logic.

### 2026-07-27: 7B native CUDA decode bottleneck localization (Foundry, H200)
**By:** Cohaagen (perf)
**What:** A native CUDA-graph decode trace for qwen2.5-7b-instruct on H200 (device 1) attributes 66.9% of one steady decode step's kernel time to symmetric int4 `MatMulNBits` GEMVs and 33.1% to split-K GQA. The largest individual slices are GQA decode (33.1%), square o_proj `gemv_f16_general` (19.5%), down projection (16.0%), fused gate/up (15.6%), and qkv GEMV (15.3%).
**Why:** This is measurement-only localization. It scopes future work to a separately reviewed o_proj grid-widening/split-K experiment or GQA partial-plus-merge fusion, while preserving the guardrail that symmetric gate/up register prefetch regresses.

### 2026-07-27: o_proj split-K grid-widening is a negative result
**By:** Cohaagen (perf)
**What:** Widening the existing two-way split-K dispatch gate for the 7B square o_proj GEMV regressed steady native CUDA decode on H200 from 309.05 to 307.23 tok/s (−0.59%, repeatable across 5/5 trials); 1.5B and 0.5B remained within noise, and 7B greedy token IDs were byte-identical. The change was reverted.
**Why:** Two-way split-K raises o_proj only from roughly 0.42 to 0.85 wave while adding a shared-memory reduction, so its reduction cost exceeds the grid-fill benefit. Do not retry this lever. A larger (3–4 way) specialized split factor would be a new kernel requiring its own A/B; GQA profiling remains the other candidate.


## Active native CUDA 27B / Inc-1b guidance

### 2026-08-02: Inc-1b PR-3 capture-fold shipped and merged (#589)
**By:** Cohaagen (build) and Harry (independent review); merged by coordinator.
**What:** PR #589 completed Inc-1b PR-3 by driving the decode-inline sibling `Executor` through existing CUDA-graph capture, flag-gated default-OFF and confined to bucket-A decode-inline/native-decode surface. It changed 4 files, left flag-off behavior a no-op, captured the single graph slot/latch path, and produced 2.05x native Qwen3.6-27B decode speedup (143.8 -> 70.1 ms/tok) with byte-exact output vs CPU fp32 oracle while capture was engaged.
**Why:** This validates the bounded staged path for the orchestration-bound single-trip Scan/LinearAttention decode case without taking ownership of shared capture machinery or altering the #443/#543 capture surface. Harry independently approved #589, mutation-proved the critical invariants, re-ran GPU engagement, and confirmed single-slot/latch safety.

### 2026-08-02: Decision inbox batch moved to archive
**By:** Scribe
**What:** Processed 40 inbox drops into `.squad/decisions-archive/2026-08.md`, including Cohaagen 27B/offload/Inc-1b scope and build notes, Harry reviews through #589, Mary scan/GAP review notes, and coordinator ownership/deferral records. Live file keeps only spawn-relevant Inc-1b guidance plus archive pointers.
**Why:** The live ledger was already 45963 bytes before merging; preserving full inbox text live would exceed the Scribe size gate and slow every spawn. The archive keeps the complete record while this file carries the current directive: #589 is merged, Inc-1b capture-fold is bucket-A and default-OFF, and future claims must preserve byte-exact oracle plus capture-engagement proof.

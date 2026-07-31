# Decisions — live standing directives

Last consolidated: 2026-07-30T12:45:00Z (Scribe consolidation — CLI sampling/rendering inbox merged; dated narrative archived)
Last consolidated: 2026-07-30T15:20:00Z (Scribe consolidation round 4 — native-pipeline + CUDA-hybrid wave; dated design/assessment drops routed straight to archive per the round-3 Task-D policy, standing kernels distilled below)

Standing governance rules and constraints maintained by Squad. Dated wave records, narrative entry trails, and historical ledger updates archived to `.squad/decisions-archive/2026-07.md`.

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
  entries compacted by size gate — 2026-07-30T04:30:00Z (Scribe tidy round 2)".
- Round-3 consolidation (2026-07-30): `.squad/decisions-archive/2026-07.md` → "Scribe
  consolidation round 3: CUDA/native/MoE wave records, 35B-A3B blocker, #87 prefetch plan
  — 2026-07-30T13:36:00Z".
- Prior active-ledger archives: `.squad/decisions/archive/`.
- Mac CPU EP load-bearing topics in the archive: PR #227 roofline lessons, load-adaptive
  opt-in, Apple Silicon portability directive, BNNS prefill/deprecation notes,
  benchmark-CI informational-only rule, dispatch-manifest lint, Sebastian/Deckard 1x1
  Conv correction, Iran SDPA model-ratio correction, and negative-result GEMV notes.
- Wave 8/9 topics in the archive: CUDA coverage batches 8/9, shape-inference catalog
  batches 3/4, NCHWc minimal-build gating, and strict reviewer-lockout correction cycle.

- CLI sampling/rendering and active inbox consolidation: `.squad/decisions-archive/2026-07.md` → "Scribe consolidation: CLI sampling/rendering and active inbox — 2026-07-30T12:45:00Z".

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

## CLI Qwen3 sampling/rendering follow-up facts (2026-07-30)

**By:** Scribe from Batty, Sebastian, and coordinator verification on `qwen3-perf-followups`.

- Default CLI builds do **not** enable the `native-backend` feature. For `run`/`generate`, `--backend auto` resolves to the ORT backend and loads `onnxruntime.dll`; `--backend native` requires rebuilding with `--features native-backend`.
- Coordinator-verified Qwen3-0.6B HF end-to-end decode with global `--profile`, ORT backend, 150 tokens: sampling is 93.97 tok/s versus greedy 98.74 tok/s, now within about 5% and up from ~59.7 tok/s. The model and metadata were not the problem: `profile_native` on the user's model measured 100.6 native / 98.1 ORT.
- This model's `chat_template` opens `<think>` only for past messages or `enable_thinking=false`, not for the current live turn. `opened_by_template` must therefore be false for the live turn, and streaming/final reasoning classification must share one marker state machine.
- Sampling hot-path fixes should avoid per-token full-vocab sorts: top-k uses selection for the threshold and top-p ranks only candidate survivors. Report processor microbenchmarks separately from model-level greedy-vs-sampling throughput.
- Inline REPL rendering must reserve rows using ratatui's wrapped-line count and coalesce frame draws; buffering every token is correct, drawing a full frame per token is not.

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

## 2026-07-30 — Processed decision inbox drops
## CUDA EP op-coverage scope — standing directive

**By:** Cohaagen (issue #67; PR #480). Data-driven placement audit (production loader +
per-node `supports_op`, recursing subgraph bodies) over the real decode models.

- **Classic transformer decode is 100% covered on CUDA** (qwen2.5-0.5b/1.5b/7b, Phi-4-mini,
  Qwen3.6-27B, Qwen3.5-35B-A3B int4): every covered-type node places, zero dtype/shape
  claim-gate silent fallbacks.
- **Control-flow ops (`If`/`Loop`/`Scan`) are executor-handled recursively and MUST NOT be
  added to the CUDA EP.** Their subgraph bodies are already placed on CUDA; they are not EP
  ops. Do not re-propose adding them as coverage gaps.
- **Remaining genuine gaps = the Qwen3.5 hybrid (Mamba + linear-attention) family**:
  `CausalConvWithState` (landed #480) and `LinearAttention` (in flight); both landed ⇒ full
  CUDA coverage of the three Qwen3.5 text decoders. `GatherBlockQuantized` is registered +
  covered as of #480 (was a registered-but-undeclared coverage-of-coverage hole); GBQ
  `bits=4` odd-blocks-per-row is a documented safe-to-defer, fail-closed follow-up.
- **Numerics rule for these hybrid kernels:** accumulate in f32 (matching the ORT/CPU EP
  oracle); widen f16/bf16 on read, narrow on write ⇒ dtype-invariant (RULES.md §2); the claim
  gate must reject configs the kernel cannot run (e.g. `d_k > 256`). Full design archived under
  round-4 consolidation.

## Native multi-component pipeline decoder seam — standing directive

**By:** Mary (issue #384; PRs #478 Inc2a, #479 Inc2b). The pipeline decode loop is made
backend-agnostic via a **stateful** seam, distinct from Inc1's stateless `ComponentSession`.

- **`trait PipelineDecoderComponent`** drives the decoder: `step(input_tokens, past_len,
  extras)` advances internal KV and **retains the step's outputs internally**;
  `next_token_logits()` / `mirror_last_present_kv(...)` / KV-window queries follow. Because the
  impl owns its per-step outputs, the loop never touches ORT `Value`/nxrt tensors and stays
  backend-agnostic. `PipelineDecodeLoopBackend` holds one `Box<dyn PipelineDecoderComponent>`
  instead of `&Session` + `&mut DecodeState`.
- **Do NOT drive a stateful decoder through a stateless host seam** — it drops native
  device-KV continuity and re-stages the whole KV cache across the host boundary every step,
  destroying decode throughput. KV is the large per-layer per-token-growing tensor and must
  stay device-resident.
- **Impls:** `OrtPipelineDecoder` (behaviour-identical, host KV, #478);
  `NativePipelineDecoder` (device-resident KV, #479 — routed per-step inputs like
  `inputs_embeds`/positions are one-token uploads per step; static cross-KV uploaded once;
  token parity vs ORT proven). In flight: Inc3 = device-KV paged mirroring + cross-component/
  vision handoff, full 35B-A3B on native. Full design archived under round-4 consolidation.

## 2026-07-30 — Scribe consolidation round 4 (native-pipeline + CUDA-hybrid wave)

**By:** Scribe (round 4)

Executed the round-3 "what to check next" checklist (now archived under
`.squad/decisions-archive/2026-07.md` → round-4 consolidation). This round applied the
adopted **Task-D policy**: dated wave/design/assessment drops are filed **straight into the
monthly archive, never into the live file**; only their distilled standing kernels enter this
ledger (the two directives above). Four inbox drops processed
(cohaagen-67-coverage-assessment, cohaagen-87-prefetch-plan, cohaagen-linear-attention-design,
mary-pipeline-inc2-design) and deleted. Merged this wave: #477 (Harry, shape-inference IR
container-type + Sequence foundation), #478/#479 (Mary, native pipeline Inc2a/Inc2b), #480
(Cohaagen, CUDA CausalConvWithState + GBQ coverage). In flight: Mary Inc3, Cohaagen
LinearAttention.

Standing carry-forward for the next Scribe round:
- **Keep dated records out of the live file.** Route wave/design/assessment drops to the
  archive; distil only standing rules here. This is the structural fix for the concurrent-Scribe
  hand-merge collision.
- **Dedupe against both the live file and the archive** — another team's Scribe may have merged
  the same drop on another machine; drops carry no team/machine attribution, so match prose.
- **Histories:** sweep `.squad/agents/*/history.md` against the chronicle gate (>8 dated entries,
  or oldest live entry predating the previous wave measured against that file's newest entry —
  never against today). deckard/roy re-accumulate fastest.
- **Do not archive agent directories.** Fail-closed; with teams active elsewhere, absence of
  local commits/drops/history proves nothing.
- **Size note (honest):** this live file is ~28 KB, above the 20 KB soft gate, and grew ~3 KB
  this round from the two distilled directives above (offset only partly by dropping the round-3
  checklist). Its content is standing-directive-dense, not a chronicle — rounds 1–3 already
  archived the per-PR narrative and this round added no dated records. Per charter, standing
  directives stay live and a directive-dense file is not compacted merely for bytes; but the
  next round should evaluate whether the older 2026-07-29 standing entries can be distilled
  further to bring the ledger back under the gate.


**By:** Scribe
**What:** Merged and archived 11 decision inbox drops for the CLI sampling/rendering and concurrent follow-up wave: batty-repl-render-fix.md, cohaagen-63-offload-gaps.md, cohaagen-87-prefetch-plan.md, cohaagen-oproj-splitk.md, harry-355-scope.md, mary-35b-a3b-blocker.md, mary-384-conv1d.md, mary-384-silu-shape.md, mary-384-value1414.md, mary-native-pipeline-plan.md, sebastian-sampling-topk-perf.md. `README.md` remained in the inbox as the durable-queue template.
**Why:** The live ledger keeps standing directives and active facts; bulky campaign narratives are preserved in the monthly archive to keep spawn context below the Scribe size gate.

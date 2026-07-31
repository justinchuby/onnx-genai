# Decisions — live standing directives

Last consolidated: 2026-07-31T00:25:00Z (Scribe round 6 — native-CUDA decode beats-ORT landmark #533, qwen3.5-0.8b 100% CUDA #529, #449 closed #531; standing kernels folded, round-5 narrative in archive)

Standing governance rules and constraints. Dated wave records and historical ledger updates
are archived to `.squad/decisions-archive/2026-07.md`.

## Ledger health rule

Archive by SIZE, not age (age-only no-ops during high-volume campaigns — most entries are
recent, so a file exceeds 1 MB while "older than 7 days" matches nothing). When this file
exceeds the gate, preserve history in `.squad/decisions-archive/{YYYY-MM}.md`, dedupe
rebase-reintroduced sections, and keep the live file to standing directives + pointers.
Size compaction of a shared append-only file is not rebase-safe: concurrent appends can
reinflate it without a conflict — re-run a compaction PR against tip before merge if main moved.

**Concurrent Scribe runs are a structural hazard** (two runs diverged on 2026-07-29). Root
fix: assemble decisions.md from distinct inbox drops rather than hand-merging. Until then,
check `git log origin/main..HEAD` before committing; rebase/merge first if main moved.

## Performance claim discipline

- A per-layer or microbenchmark speedup is not a model-level claim. Confirm with Amdahl
  and, when practical, a real model-level measurement before presenting campaign impact.
- Always state the exact model, dtype, metric, prompt/token regime, host load, and runner.
  TinyStories-1M and TinyStories-33M ratios are not interchangeable.
- Separate measured, estimated, and projected values. Do not compare measurements taken
  under different host load without labeling them. PR benchmark workflow absolute times
  are informational only; same-run PR-vs-merge-base deltas are the useful signal.
- Two independent measurements that agree beat one confident outlier. Retracted/corrected
  examples in the archive: the impossible 197 GB/s roofline, the load-corrupted
  adaptive-calibrator verdict, the unmeasured 15x ORT batch-decode estimate, the 1x1 Conv
  microbenchmark headline deflated through the real path, and the SDPA decode 1.9x vs 1.37x
  correction.
- A SIMD or accelerated path without a reachability test is equivalent to an unwired
  placeholder.

## Apple Silicon portability and Mac CPU EP rules

- Mac CPU EP optimizations must generalize across Apple Silicon (M1/M2/M3/M4, base/Pro/
  Max/Ultra). The M1 Max is a measurement rig, not the target.
- No compile-time constants derived from one machine's measurements. Query topology,
  cache sizes, and features at runtime; derive tiling and thread counts from those facts.
- Feature-detect any path beyond the shared ARM baseline and keep a correct fallback.
- Reach the Apple matrix coprocessor through Accelerate (BLAS/BNNS), never hand-rolled AMX
  encodings; BNNS/Accelerate calls happen at dispatch level, not inside Rayon regions.
- The CPU EP stays one general implementation shared with Intel and ARM; Apple Silicon
  specialization lives behind runtime dispatch, not a parallel kernel tree.
- BNNS `BNNSMatMul` deprecation in macOS 15 is a migration to BNNSGraph, not evidence the
  AMX/fp16 path or existing measurements are invalid.

## Load-adaptive decode path

`ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL` semantics are binding:

| Value | Meaning |
|---|---|
| unset | `On`: deterministic persistent SPMD pool, no probing |
| `=1` | `On`: explicit persistent pool |
| `=0` | `Off`: flat path |
| `=auto` | `Adaptive`: opt-in calibrator |

Reason: load-adaptive selection silently changed paths under agent load, damaged
reproducibility, and produced false verdicts. Default predictable; adaptive only on request.
Expose selected path via `decode_path_label()` and tracing or `NXRT_CALIB_DEBUG` fallback.

## Dispatch-manifest inverse rule

Every claimed `(op, variant, platform) -> minimum tier` optimization needs a curated manifest
row, a `_TEST_HITS` counter, and a test proving the counter fires. Inverse is also binding: if
a fast path exists but a higher-priority guard intercepts it, the test must fail. Manifest is
CI-only, zero runtime cost. Removing a row is a conscious un-claim; a claim without
reachability is a merge blocker.

Historical dispatch-miss patterns are catalogued in the archive (Accelerate placeholder not
wired, M=1 GEMV intercepted by GEMM, SDPA NEON gaps, BNNS non-contiguous weights, etc.).

**Manifest lint reliability (#414):** A lint whose increment regex ignores rustfmt-wrapped
counter increments is blind to them; a lint without a `--self-test` that passes before it
runs proves nothing when silent (a dead branch looks like a rustfmt-wrapped live one). Wire
a self-test into CI exercising wrapped + single-line increments plus genuine dead-counter cases.

## Minimal-build and shape-inference rules

- Graph/layout transforms must be gated on both their infrastructure feature and the
  operator group that supplies their kernels. Wave 9 fixed NCHWc layout propagation to
  require both `mlas` and `ops-cnn`; an MLAS-only build must not advertise transforms whose
  CNN kernels are absent.
- Shape-inference registrations must use the operator's actual ONNX domain/version, not a
  convenient family namespace (Wave 9: `StringNormalizer`/`TfIdfVectorizer` `ai.onnx.ml`→`ai.onnx`).
- Attribute-dependent output typing must follow the active default/value attribute, not an
  unrelated class-list attribute (Wave 9: `LabelEncoder-1` mirrors `CategoryMapper`).
- Sequence/Optional/Map/ZipMap container propagation stays blocked until the tensor-only
  `TypeInfo` model gains container-aware representation; do not fake these as tensors.

## BNNS / Conv / GEMM current guidance

- For fp16 MatMul on macOS, BNNS f16->f32 reaches AMX and is the preferred compute-bound
  prefill/batch path at M>=2 when available; M=1 decode remains a GEMV problem.
- Do not call BNNS from inside Rayon. BNNS uses system threading internally.
- For 1x1 Conv, BNNS Conv can be dominated by filter creation/copy overhead (Sebastian
  diagnosis); Deckard's PR #347 used a spatial-size-dependent route through the real
  `im2col_gemm_execute` path, with claims scoped by model measurement, not microbenchmark
  ratios. A fitted threshold is acceptable only when labeled fitted and bracketed by measured
  data (a wrong rationale is worse than none — future engineers tune to the false premise).
- `BNNSFilterApplyBatch` is unreliable for `BNNSFilterCreateLayerConvolution` filters
  (SIGSEGV inside libBNNS.dylib at batch>1, confirmed via AddressSanitizer on PR
  squad/fix-batch-segfault). Use per-image `BNNSFilterApply` until migration to `BNNSGraph`.

## Model artifact hygiene

Fetch large external models only when needed, measure, and delete immediately — do not leave
benchmark models in `models/` or worktrees (the archived ResNet/Whisper run used
fetch-measure-delete and restored the disk baseline).

## Active historical pointers

For detailed per-PR narrative, use `.squad/decisions-archive/2026-07.md` rather than
expanding this live file. Archived there:
- Consolidation checkpoints: size-gate snapshots 2026-07-28; narrative compactions 2026-07-29;
  rounds 2–6 (2026-07-30…07-31: CUDA/native/MoE + native-pipeline + CUDA-hybrid wave records).
  Prior active-ledger archives: `.squad/decisions/archive/`.
- Mac CPU EP load-bearing topics: PR #227 roofline lessons, load-adaptive opt-in, Apple Silicon
  portability, BNNS prefill/deprecation, benchmark-CI informational-only rule, dispatch-manifest
  lint, Sebastian/Deckard 1x1 Conv correction, Iran SDPA model-ratio correction, GEMV notes.
- Wave 8/9 topics: CUDA coverage batches 8/9, shape-inference catalog batches 3/4, NCHWc
  minimal-build gating, strict reviewer-lockout correction cycle.

## CLI charter — standing directives

**By:** Justin Chu (2026-07-27). Live policy (restated from archive); an agent reading only
this ledger must see these.

- **The CLI is a developer/maintainer tool, not a consumer product.** Rank CLI work by
  *does this shorten a maintainer's debug/iterate loop, or expose engine behavior we cannot
  otherwise observe?* — not by *does a competitor have it?* **Explicitly rejected, do not
  re-propose:** remote-client mode against an OpenAI-compatible server; model registry / pull
  / consumer model lifecycle; conversion/quantization/fine-tune loops as CLI product features.
  See `docs/research/cli/00-backlog.md`.
- **The REPL is the primary CLI investment.** Target bar is Copilot CLI's interactive shell
  with one deliberate divergence: **ratatui's inline viewport, not a full-screen alternate
  screen** (native scrollback + terminal-native copy matter for output pasted into issues/
  benchmarks/traces; Justin chose inline — `docs/research/cli/05-repl-redesign.md` §2). Phase 1
  landed in #289. Remaining phases (`/fork`, `/rewind`) depend on runtime APIs tracked in
  `docs/research/cli/04-runtime-capability-inventory.md` and `06-fork-rewind-api.md`. Fork is
  reserved behind a type gate and **not yet enabled on any backend**.

## CI: run tests on every platform; instrument for coverage only where informative

**By:** Pris (2026-07-28). Full coverage required on PRs; a parallel uninstrumented Linux fast
job (5–9 min) offers early feedback but never substitutes for the full gate. Windows ARM64
retains tests/clippy but not llvm-cov (duplicates x64/macOS while owning the CI critical path).
Platform execution is the signal; instrumentation is the cost. Critical path:
`CLI ORT (Windows x86_64)` at ~18m50s.

## Standing durable rules — 2026-07-29 wave (distilled; full narrative in archive)

- **Native multi-turn perf uses the session-persistent KV API** (Pris #408), not the old
  stateless path, unless explicitly `--native-stateless`.
- **A step that warns instead of failing is not verification** (Holden #401). Check HTTP
  status explicitly (`curl -f`/`-w %{http_code}`); validate archive magic bytes
  (gzip `1f 8b` / zip `PK`) before extracting. A warn-on-failure check is a silent failure.
- **Model-declared generation defaults are canonical** (Leon #385/#392, closes #290/#296).
  Precedence: explicit caller flag > model-declared value > greedy fallback. Enforced in
  the engine so CLI/server/Python inherit it. Do not force greedy by discarding model config.
- **The CUDA driver API ships with the display driver, not the toolkit** (Leon #395).
  `nvcuda.dll` is present whenever the GPU driver is installed; `cust`/`cudarc` fall back
  `*_13`→`*_12`. Verify inferred bad facts against a fresh env before suppressing them.
- **Probe the stream you write to** (Rachael #372). Stats to stderr ⇒ test
  `stderr().is_terminal()`, not stdout (inverts under redirection). Extract testable pure
  functions (`stats_text()`, `needs_trailing_newline()`) so branches can be unit-tested.
- **Run the exact gate command CI runs** (Rachael #372). `cargo fmt --all --check`, not a
  package-scoped subset; a narrower-scope local validation is structurally untrustable.
- **Terminal behaviour requires PTY-driven tests** (Rachael #372). A `cfg`-gated test is
  unverified until it runs on a platform where the gate admits it; compilation ≠ verification.
  Piped-stdio cannot cover control sequences, window-size events, terminal probe responses.
- **Type-ahead is not lost during generation** (Zhora #393, closes #298). ConPTY does not
  echo type-ahead during generation; not a backend bug — do not re-open as native. Native
  conhost mid-stream type-ahead still unverified.
- **Worktree lifecycle** (Justin): never delete a worktree before Scribe merges its decision
  inbox. (Inbox is git-tracked, so drops survive worktree deletion.)
- **Warmup uses a shared registry method** (Lull+Rachael #407): `ModelRegistry::warmup` for
  both the per-model setting and `POST /v1/admin/models/{id}/warm` — one deterministic token,
  idempotent, retryable. Return typed errors mapped 404/500/500; a loaded model's failed
  warmup must not report 404.

### All inference/pipeline metadata must be explicit; name guessing is forbidden
**By:** Justin Chu directive #377; Cohaagen/Benny/Melina/Matthias (PRs #380/#382/#377/#412)

ALL inference/pipeline metadata except io-SHAPE must be EXPLICIT and GENERAL. Replace name
guessing/historical-name fallback with explicit metadata plus a clear ERROR naming the
missing key. Only io-SHAPE may disambiguate. Do not re-propose deferral.

**Active schema fields (emit these names verbatim):**
- `pipeline.strategy.inner_embedding_output: Option<String>` — nested-AR inner decoder embedding output port. Absent ⇒ ERROR.
- `model.io.static_cache: Option<StaticCacheIoSpec>` — `write_indices_input`, `kv_sequence_length_input`, per-layer `key/value_cache_inputs/outputs` (equal-length, positional). Inconsistent ⇒ ERROR. Must be declared; convention-based binding removed (Matthias #412) — a TensorScatter static-cache graph without the block fails closed naming the missing key. `StaticCacheAbi::classify` stays name-agnostic.
- Encoder prompt-input role: from `model.encoder.inputs.audio_features` vs `.input_ids`; no port-name string matching.
- Paged-KV bridge geometry: from `model.io.kv_inputs`/`kv_outputs` only; no metadata ⇒ `Ok(None)`.
- Off-limits: `decode_contract.rs` `KvNamingConvention` is only for #99 speculative proposers.

## Testing discipline — standing rules (from reasoning-fixture review, #410/#411)

- **Assert on what the code did, not a summary of what it should do.** Tests keying on a
  display/summary line (e.g. `/session`) stay green while the real path
  (`resolve_sampling_defaults`) is broken. Surface the resolved policy into `--stats`/
  `--profile` and assert on that boundary.
- **Run a new test in isolation before believing it.** A single green in a full parallel
  suite can be a stderr-interleave artifact (a real fix survived 15/15 isolated runs the
  suite hid).
- **A fixture whose every assertion is "the turn was dropped" cannot distinguish correct
  behaviour from total breakage.** Make the success path reachable so a regression fails.
- **A near-deterministic fixture cannot witness sampling** (low temp/top_k ⇒ effectively
  greedy; token-stream assertion ~95% false-fail). Assert on the resolved policy object.
- **One policy resolved at two sites is the defect.** Resolve once via a shared helper both
  paths call, reading the live backend on demand (no cache/staleness across
  `/reload`/`/ep`/`/backend`).

## CUDA EP op-coverage scope — standing directive

**By:** Cohaagen (issue #67; #480/#484/#525). Data-driven placement audit (production loader +
per-node `supports_op`, recursing subgraph bodies) over the real decode models.

- **Classic transformer decode is 100% covered on CUDA** (qwen2.5-0.5b/1.5b/7b, Phi-4-mini,
  Qwen3.6-27B, Qwen3.5-35B-A3B int4): every covered-type node places, zero claim-gate fallbacks.
- **Control-flow ops (`If`/`Loop`/`Scan`) are executor-handled recursively and MUST NOT be
  added to the CUDA EP** (subgraph bodies already place on CUDA; not EP ops). Do not re-propose.
- **Qwen3.5 hybrid (Mamba + linear-attention) family is now fully CUDA-covered:**
  `CausalConvWithState` (#480), `LinearAttention`/Gated DeltaNet (#484: per-thread
  f32-register-column state, 4/4 parity, hybrid node placement 0→18/18/24), RotaryEmbedding
  com.microsoft + Bool NonZero (#525). `GatherBlockQuantized` covered as of #480; #525 added a
  LOUD fail-closed gate for GBQ `bits=4` odd-blocks-per-row (safe-to-defer), fixed a
  RotaryEmbedding dtype-check bug (Int64 position_ids compared vs float), softened over-broad docs.
- **Numerics rule for these hybrid kernels:** accumulate in f32 (matching the ORT/CPU EP
  oracle); widen f16/bf16 on read, narrow on write ⇒ dtype-invariant (RULES.md §2); the claim
  gate must reject configs the kernel cannot run (e.g. `d_k > 256`). Full design archived.
- **#529:** qwen3.5-0.8b hybrid places 100% on CUDA (split package, 1289 nodes, 0 declines);
  regression-locked `qwen35_0_8b_placement_lock`. E2e decode is still BLOCKED on the loader
  (`Engine::from_dir` rejects the 3-onnx split; `from_pipeline_dir` refuses during vision
  `smart_resize` admission); parity harness `qwen35_0_8b_hybrid_native_cuda_e2e` graceful-skips
  until the loader is fixed.

## Native multi-component pipeline decoder seam — standing directive

**By:** Mary (issue #384; #478 Inc2a, #479 Inc2b). The pipeline decode loop is backend-agnostic
via a **stateful** seam, distinct from Inc1's stateless `ComponentSession`.

- **`trait PipelineDecoderComponent`** drives the decoder: `step(input_tokens, past_len,
  extras)` advances internal KV and **retains outputs internally**; `next_token_logits()` /
  `mirror_last_present_kv(...)` / KV-window queries follow. Because the impl owns its per-step
  outputs, the loop never touches ORT `Value`/nxrt tensors. `PipelineDecodeLoopBackend` holds
  one `Box<dyn PipelineDecoderComponent>` instead of `&Session` + `&mut DecodeState`.
- **Do NOT drive a stateful decoder through a stateless host seam** — it drops native device-KV
  continuity and re-stages the whole KV cache across the host boundary every step, destroying
  throughput. KV must stay device-resident.
- **Impls:** `OrtPipelineDecoder` (behaviour-identical, host KV, #478);
  `NativePipelineDecoder` (device-resident KV, #479 — routed per-step inputs like
  `inputs_embeds`/positions are one-token uploads per step; static cross-KV uploaded once;
  token parity vs ORT proven). **Inc3a (#485):** CUDA native decoder via `inputs_embeds`,
  on-GPU token parity at positions [0,5,6,7]. **Inc3b (#487):** generic routed CUDA ports —
  metadata-driven `decode_cuda_eager_step_inputs`/`prepare_cuda_owned_step_inputs`; removed
  `load.rs` CUDA Routed refusal; captured fast path byte-identical; KV device-resident.
- **MILESTONE:** the native multi-component pipeline CUDA decode path (Inc2a→Inc3b) is fully
  on main, and **real qwen3-0.6b native-CUDA e2e matches ORT-CUDA for 32 tokens** — landmark
  real-model validation of native CUDA decode. The mask/ReduceSum finding (#487, Lori
  APPROVED) is an ARTIFACT, not a blocker: proven by a real mask-consuming decoder locking 32
  tokens to ORT-CUDA. **Inc3c (#533, Lori APPROVED) LANDED — native CUDA decode now BEATS
  ORT:** default-off `ONNX_GENAI_NATIVE_DECODER_CAPTURE_STEP_INPUTS` writes a persistent
  `[1,1,width]` device binding per routed port each step and reuses the captured `run_one_token`
  (mask frozen, KV device-resident) ⇒ 1.38x ORT-CUDA on real qwen3-0.6b. Metadata-driven from
  `session.inputs()`; generalizes to 35B-A3B GQA. Engagement proven non-tautologically via
  counter `NATIVE_DECODER_CAPTURED_STEP_INPUT_DECODES` (OFF=0/ON=3, tokens byte-identical).

## Shape-inference sequence/container ops — standing directive

**By:** Harry (issue #449; PR #477 foundation, #486 inc2).

- `#477` laid the IR container-type + Sequence foundation. `#486` added
  `SequenceInsert`/`SequenceErase`/`SplitToSequence`/`ConcatFromSequence` plus seq↔tensor
  conversion. `#531` (inc4) added `SequenceMap` + `Scan` container support + cross-subgraph
  capture and **CLOSED #449**. Container-type shape inference is COMPLETE: additive
  `ValueType{Tensor|Sequence|Optional|Map}`, byte-identical tensor path guaranteed (gated on a
  non-empty container map). Catalog now 217 ops/262 entries. Deferred (non-load-bearing):
  Optional/Map handlers, IR-persistence of `ValueType`.

## 2026-07-31 — Scribe consolidation (round 6)

**By:** Scribe

Merges since round 5 (kernels folded into the directives above; per-PR narrative in
`.squad/decisions-archive/2026-07.md`):
- **#533 — Mary — native pipeline Inc3c** — LANDMARK: native CUDA decode beats ORT.
- **#529 — Cohaagen — qwen3.5-0.8b hybrid 100% CUDA** (e2e still loader-blocked).
- **#531 — Harry — #449 inc4** — SequenceMap + Scan; closed #449.
- **#532 — Scribe round 5** — decisions.md re-distilled under the 20 KB gate.

Held (not merged): **#534** (Harry, server contracts #481/#482, Melina APPROVED) targets Justin's
active branch `feat/genai-demo-dashboard` (PR #476); that code is not on main.

In flight: mary-2 real-model capture-engagement + default-on rec; cohaagen-4 loader-unblock;
harry-5 generalize ORT `clone_value` to all POD dtypes (Bool / gemma-3n).
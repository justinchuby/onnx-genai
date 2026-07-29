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

## 2026-07-29 — Verification steps that warn instead of fail are not verification

**By:** Scribe (recording Holden's diagnosis)  
**Blocks:** PR #401 (wheels.yml hardening)  
**Durable rule:** A verification step that warns instead of failing is not verification.

Evidence: the ORT `osx-x86_64` download was the only asset without a pinned checksum, so its integrity check emitted a warning and the build continued with a 9-byte HTML-ish error body. Combined with `curl` exiting 0 on a 404, three "successful" steps produced a corrupt input before `tar` finally objected. The archive format was never validated until the terminal step.

**Related:** `curl` needs `-f` (or an explicit status check via `-w %{http_code}`) or an HTTP error body will be treated as a downloaded file and exit code 0.

## 2026-07-29 — wheels.yml: drop unpublishable macOS x86_64 wheel; harden ORT download errors

**Author:** Holden (release/CI-hardening). Branch `fix/wheels-macos-x86`, merged as **PR #401**.

### Context

Issue #326 ("wheels.yml failing every run since 2026-07-21, including on a release tag") was closed by PR #337, which fixed the Windows DXCore API-set DLL exclusion and the manylinux image pin. But wheels.yml stayed red: three of four CPU jobs passed and **CPU wheel (macOS x86_64) still failed**, so a broken release pipeline was closed as fixed for over a week (through tag `v0.1.0-dev.3`).

Failure signature from `onnx-genai-ort-sys`'s build script:
```
tar: Error opening archive: Unrecognized archive format
Failed to extract ORT archive
```

### What was actually downloaded (verified, not assumed)

The build script downloads `onnxruntime-{os}-{ORT_VERSION}.{ext}` from ORT's GitHub releases. For Intel macOS that resolves to `https://github.com/microsoft/onnxruntime/releases/download/v1.27.0/onnxruntime-osx-x86_64-1.27.0.tgz`.

Reproduced with the same `curl -L -o` the build script uses:
- **HTTP status: 404**, `Content-Type: text/plain`, **9 bytes** on disk, body = `Not Found`.
- `curl` exits **0** on a 404 without `-f`, so the build script's success check passed.
- **No SHA-256 checksum is pinned** for `onnxruntime-osx-x86_64-1.27.0.tgz` (all four other platforms have one), so `verify_archive_checksum` only warned and returned.
- `tar` then received the 9-byte "Not Found" text → "Unrecognized archive format".

### Is the artifact still published upstream? No.

GitHub release assets, macOS only:
| ORT version | osx-arm64 | osx-x86_64 | osx-universal2 |
|---|---|---|---|
| 1.18.0 | ✅ | ✅ | ✅ |
| 1.20.0 | ✅ | ✅ | ✅ |
| 1.22.0 | ✅ | ✅ | ✅ |
| **1.27.0** | ✅ | ❌ 404 | ❌ 404 |

Upstream ONNX Runtime stopped shipping `osx-x86_64` (and `universal2`) prebuilts after the 1.22.x series. We pin **ORT 1.27.0** because it must match `ORT_API_VERSION` 27 used by the bindgen headers (`ort-sys/build.rs:16-19`), so we cannot downgrade to recover an Intel binary. The absent checksum was itself the early warning that whoever pinned checksums couldn't fetch that asset.

### Decision

**Drop the `macOS x86_64` entry from the `cpu-wheels` matrix in `.github/workflows/wheels.yml`.**

Rationale: the artifact genuinely no longer exists upstream, and building ORT from source for an EOL Intel-macOS target (GitHub's `macos-15-intel` is the final hosted Intel line) is not justified. A job that can never pass trains everyone to ignore a red pipeline — exactly how #326 stayed "fixed but broken" for a week. Documented the gap in `crates/onnx-runtime-python/README.md` (new "Supported wheel platforms" note) so Intel-Mac users know the wheel is intentionally absent and can build from source via `ORT_ROOT`.

### Error-message quality hardening

`"Failed to extract ORT archive"` named neither the URL, the HTTP status, the file size, nor the first bytes — it cost real triage time. `ort-sys/build.rs` now:
- Captures curl's HTTP status via `-w %{http_code}` and **fails on any non-200**, with a message naming the URL, status, byte size, first bytes, and the likely cause (upstream not publishing that asset for this platform).
- Validates archive magic bytes (gzip `1f 8b` / zip `PK`) before invoking tar/zip, so a 200-with-error-page is caught with an actionable message.
- The tar-extract failure now reports the URL, size, and first bytes and points at `ORT_ROOT` as the escape hatch.
- Added `--retry 3 --retry-delay 2` to the download for CI flakiness.

### Verification

`workflow_dispatch` on `fix/wheels-macos-x86`; per-job results recorded in the PR / run link. **PR #401 merged.**

## 2026-07-29 — DFT kernel for Perch bioacoustics model

**Author:** Deckard  
**Status:** Implemented  
**Verified:** PR #357

### Summary

Implemented the ONNX DFT operator (opset 17+) with a vDSP Accelerate fast path for power-of-two lengths on macOS/iOS, plus a Cooley–Tukey radix-2 fallback for all platforms. Verified end-to-end on the Perch v2 bioacoustics model from HuggingFace.

### Key findings

- Opset registration: DFT at `since_version: 17` correctly covers all models at opset ≥ 17 (including Perch at opset 18). Verified via `DFT_VDSP_TEST_HITS` counter increments (1000 during Perch inference).
- Attribution (M1 Max): DFT is 0.80% (9.3ms) of total model time (~1171ms). Amdahl projection: even reducing it to zero yields only 1.008× speedup — negligible.
- Numerics: vDSP f32 vs double-precision naive DFT max absolute error < 1e-2 (N=1024); radix-2 fallback within 1e-4 absolute tolerance.

### Decision

No further DFT optimization is warranted for Perch. The vDSP path is already hardware-optimal for the power-of-two case. The real Perch bottlenecks are elementwise ops (Add/Mul/Div/Neg/Exp = 66%) which benefit from the SIMD vectorization work in `onnx-genai-dense-elem`.

## 2026-07-29 — Session-persistent KV cache — Phase 1 implementation

**Author:** Deckard  
**Status:** Implemented  
**PR:** squad/session-kv-phase1

### Decision

Implemented Roy's Phase 1 design: remove the unconditional `reset()` from the native decode session's multi-turn path and add incremental prefill so a continued conversation only prefills new tokens.

### Key design choices

**Cache invalidation:** The API computes `common_prefix_len(session_tokens, new_prompt_tokens)` on every call. If the new prompt diverges from the cached history, the KV is rewound to the divergence point via the existing `rewind()` machinery. Default behavior is safe: the stateless `generate()` path still resets unconditionally.

**resume_from capping:** `resume_from = min(prefix_len, native.current_len())` — because the session token history includes the last generated token which was sampled but never fed through the model.

**Weight-transpose cache interaction:** Phase 1 does not change model/executor lifetime. Global weight-transpose caches (#353) are keyed by data pointer and cleared on `Executor::drop`. One `InferenceSession` per `Engine` lifetime is preserved, so the interaction is nil.

**Single-session limitation (Phase 1):** Only one native session is supported. Attempting to create a second fails explicitly. The stateless `generate()` path remains unchanged.

### Verification

1. `native_session_incremental_matches_stateless` — token-identical output.
2. `native_session_rewind_produces_correct_output` — divergent prefix correctness.
3. `native_session_creation_guards` — API safety rails.
4. `NATIVE_SESSION_INCREMENTAL_PREFILL_TEST_HITS` counter + dispatch_manifest.toml row.

## 2026-07-29 — Stream parsed tool calls as OpenAI deltas

**By:** McClane

**What:** Emit one metadata delta followed by an arguments delta for every parsed tool call, then finish with `tool_calls`.

**Why:** Clients can assemble tool invocations incrementally without receiving a monolithic completed tool-call object, while retaining full-output parsing for Qwen, Llama, and Mistral safety.

## 2026-07-28 — Fix BNNS batch>1 SIGSEGV in Conv/Pool kernels

**Author:** Resch  
**Status:** Implemented  
**PR:** squad/fix-batch-segfault

### Root cause

`BNNSFilterApplyBatch` with `batch_size > 1` causes a SIGSEGV inside `libBNNS.dylib` for convolution filters created via `BNNSFilterCreateLayerConvolution`. The crash is inside Apple's framework code (frame #0 in libBNNS.dylib, confirmed via AddressSanitizer). The single-image `BNNSFilterApply` works correctly.

The bug was introduced when BNNS Conv was added (PR #324 / #317) — it exercised only batch=1 shapes. The batch dimension was correctly threaded through buffer allocation and stride calculations, but the BNNS framework itself crashes when the deprecated `BNNSFilterCreateLayerConvolution` + `BNNSFilterApplyBatch` combination receives batch>1.

### Fix

Replace `BNNSFilterApplyBatch` with a per-image loop using `BNNSFilterApply`:
- Conv: `bnns_conv_execute` in `conv_ref.rs`
- Pool: `bnns_pool_execute` in `pooling.rs` (same class, prophylactic fix)

BNNS still uses its internal thread pool per `BNNSFilterApply` call, so the AMX compute advantage is preserved. The overhead is one extra function call per image — negligible relative to the convolution work.

### Batch>1 performance (MobileNetV2, native)

Measured at load 2.7–3.0:
| batch | median ms | throughput (samples/s) | scaling |
|------:|----------:|----------------------:|--------:|
| 1 | 11.5 | 86.7 | 1.00× |
| 2 | 22.7 | 88.1 | 1.02× |
| 4 | 45.1 | 88.7 | 1.02× |
| 8 | 89.2 | 89.7 | 1.04× |
| 16 | 178.1 | 89.8 | 1.04× |

Native batch scaling is ~1.0× (linear cost, no amortization). ORT gets 1.9× because its internal NCHWc path and thread pool amortize overhead. Our per-image BNNS dispatch preserves scaling but cannot achieve that overhead amortization without the newer `BNNSGraph` API (which supports batch natively). This is a future optimization axis, not a regression — before this fix, batch>1 simply crashed.

### Standing lesson

`BNNSFilterApplyBatch` is unreliable for `BNNSFilterCreateLayerConvolution` filters. Use per-image `BNNSFilterApply` until migration to `BNNSGraph` (which supersedes the deprecated per-layer API and supports batch natively).

## 2026-07-28 and earlier — archived by size gate

Detailed narrative entries from 2026-07-28T17:40:00+0000 through 2026-07-28T04:30:00-07:00 (Holden wheels summary: PR #347, PR #349, Wave 8/9 consolidation), plus extended Pris CI tiering/ARM64/multi-turn benchmark entries (lines 310–531 of previous compaction) have been moved to `.squad/decisions-archive/2026-07.md` → "Narrative entries compacted by size gate — 2026-07-29". 

These remain available for detailed reference. Summaries: control-flow/QMoE wave work, MobileNetV2 Clip/Relu dispatch, Cache roofline for small models, CLI improvement track lessons, CI tiering/full coverage decision, ARM64 coverage removal, multi-turn and batch benchmark deficits. All foundational context is preserved in the archive.

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

## 2026-07-28 — Fact-Checker Verdict: ORT shared-buffer KV allocation strategy

**Requested by:** Justin Chu (@justinchuby)  
**Blocks:** PR #367 (`fix/cuda-kv-capacity`)

**Verdict: ✅ PR #367's wording is correct. `main`'s wording is stale.**

ORT shared-buffer KV allocation: `kv_capacity_bucket(0, max_length)` at `dynamic.rs:386` allocates the **minimum bucket (256)** via and grows lazily by `ensure_kv_capacity`, not by pre-allocating the full context. `main`'s comment describing pre-allocation is stale — it predates the bucketing policy. PR #367's wording is verified by code tracing and passing tests `kv_capacity_bucket_rounds_to_power_of_two_min_256` and `ensure_kv_capacity_orders_fallible_work_before_invalidation`.

### 2026-07-28: CI tiering rejected; run full coverage on PRs

**By:** Pris

CI has two parallel PR signals: a Linux-only uninstrumented `Fast (Linux x86_64)` job in `ci.yml` for early feedback, and the full coverage gate. PRs, `main` pushes, nightly schedules, and manual dispatch still run the full signal. Rule: run tests on every platform; instrument for coverage only where the coverage is informative. Detailed entry archived.

### 2026-07-28: Windows ARM64 coverage removal

**By:** Pris

Removed the `Rust coverage (Windows ARM64)` job and replaced it with `Rust (Windows ARM64)` uninstrumented tests. Windows ARM64 platform execution catches real platform bugs, but coverage for pure-Rust crates duplicates x64/macOS while owning the critical path. Uninstrumented ARM64 tests retain signal without overhead. Detailed entry archived.

### 2026-07-28: Multi-turn and batch benchmarks reveal structural native deficit

**Author:** Pris  
**Root cause:** Native backend has no session-persistent KV cache; each turn re-prefills entire conversation. ORT preserves KV, prefilling only new tokens.

**Findings (Apple M1 Max):** TinyStories-33M ORT 2.1× faster over 10 turns; Qwen2.5-0.5B ORT 1.18× faster. Batch vision: native crashes at batch>1 (segfault bug). **Decision:** Session-persistent KV is priority #1; pre-packing deferred post-KV. Detailed measurements and projections archived.

## 2026-07-29 — Probe the stream you are writing to

**By:** Rachael  
**Blocks:** PR #372 (REPL stats display)

`emit_stats_line` chose its format from `io::stdout().is_terminal()` while writing to stderr. This inverts under redirection: `> out.txt` in a terminal gives the cramped one-line form intended for log files; `2> stats.log` puts terminal-oriented layout into a file. **Rule:** always probe the destination stream, never a convenient nearby one. Stats to stderr → test `stderr().is_terminal()`, not stdout.

**Durable pattern:** Extract testable pure functions. `stats_text()` and `needs_trailing_newline()` had to become named pure functions before their branches could be unit-tested; inlining had hidden both defects in the production path.

## 2026-07-29 — Run the gate command CI runs, verbatim

**By:** Rachael  
**Blocks:** PR #372 (formatting)

A package-scoped `cargo fmt --check` passed locally while the entire workspace was unformatted. CI's `cargo fmt --all --check` caught it in 37 seconds. **Rule:** use the exact gate command that CI uses, not a convenient subset. Local validation with a narrower scope is structurally untrustable.

## 2026-07-29 — Terminal behaviour requires PTY-driven tests; piped I/O cannot cover it

**By:** Rachael  
**Blocks:** PR #372 (PTY harness)

Two `#[cfg(unix)]` PTY tests written on Windows compiled to **zero tests** on Windows and were reported alongside a green 168-test suite. Ran under WSL: did not compile (`nix` missing `term`/`fs` features), could hang the runner forever, failed `clippy -D warnings`. **Rule:** A `cfg`-gated test is unverified until it runs on a platform where the gate admits it. Do not assume compilation equals verification. Piped-stdio tests structurally cannot cover PTY-specific behavior (control sequences, window size events, terminal probe responses).

## 2026-07-29 — Type-ahead is not lost during generation on Unix or Windows

**By:** Zhora  
**Closes:** Issue #298 (type-ahead swallowed)  
**Verified:** PR #393

**Investigation scope:** user keystrokes during decode on Unix (`ratatui` + `crossterm 0.29`) and Windows (`ratatui` + native conhost/ConPTY).

**Findings:**
- crossterm 0.29 routes `cursor::position()` keystrokes to `skipped_events` and drains them back; ratatui 0.30's inline viewport reads stdin only at init/resize.
- Windows `ReadConsoleInputW` is unaffected by `ENABLE_LINE_INPUT`; `SetConsoleMode` does not flush the queue; there is no `FlushConsoleInputBuffer` in the Windows API.
- Tested against ~19s real stream with type-ahead injected pasted, delayed, and character-by-character.

**Verdict:** ConPTY does not echo type-ahead during generation at all, so no REPL repaint can overwrite it. This is a terminal characteristic, not a backend bug. **Do not re-open as a native-backend issue.**

**Still unverified (native conhost):** In a native terminal (not IDE or ConPTY), start a long stream, type `/help` mid-stream, watch for appear-then-vanish, then press Enter at next prompt. If help renders, keystrokes were safe (cosmetic echo overpaint); if nothing, it is real loss.

## 2026-07-29 — PTY-harness hazards that present as swallowed input

**By:** Zhora  
**Blocks:** PR #393 (timeout and drain fixes)

A **0×0 window** makes `ratatui::insert_before_no_scrolling_regions()` infinite-loop. Feeding `\n` where a terminal sends `\r` means the line never completes. Both present as a hang or lost input and send the next investigator down the wrong path (looking for a keystroke loss instead of a terminal setup defect). Fixed in PR #393 and documented for future harness work.

## 2026-07-29 — Prefer an idle timeout to a total one when the child is silent before first output

**By:** Zhora  
**Blocks:** PR #393 (drain timeout)

A 30s total drain budget lost ~48s to a cold model load on a slow machine; it was not a safety net, it was a coin flip. Now 120s idle — justified at ~2.5× the measured worst case — with a failure message stating it timed out waiting for bytes, **not** a trailing-newline defect. An empty read can never masquerade as an assertion failure; this distinction prevents phantom failures during CI variance.

## 2026-07-29 — The DeepSeek "repeats thinking, won't stop" report is not a backend bug

**By:** Leon  
**Blocks:** PRs #367, #385, #392, #395  
**Verdict:** ✅ Verified not native-backend issue

Native CUDA and ORT fall into **identical verbatim repetition loop**, diverging only at one `,`/`.` tie-break at character position 325 (fp16 GPU vs fp32 CPU). It is **greedy-decoding degeneration**. The `CUDA KV capacity exceeded (4097 > 4096)` error was a **downstream symptom** — the loop grew KV into the native path's cap; ORT had no cap and looped silently.

**Root cause** was a model that ships `do_sample: true, temperature: 0.6` but the CLI was forcing greedy override (next durable rule below).

## 2026-07-29 — Model-declared generation defaults are canonical; our constants are fallback, not override

**By:** Leon  
**Blocks:** PRs #385, #392  
**Closed:** Issues #290, #296 (silent temperature/do_sample override)

We parsed model `do_sample`, `temperature`, `top_p`, `top_k` and then discarded them, forcing greedy — on models that ship explicit values precisely because greedy degenerates (e.g., DeepSeek, Qwen).

**Precedence is now strict:** explicit caller flag > model-declared value > greedy fallback. Enforcement is in the engine so CLI, server, and Python all inherit it without duplication.

## 2026-07-29 — The CUDA driver API ships with the display driver, not the toolkit

**By:** Leon  
**Clears:** PR #395 misconception

`nvcuda.dll` is present whenever the GPU driver is installed. Both `cust` (the EP loader) and `cudarc` fall back `*_13` → `*_12` cleanly. An earlier conclusion that native CUDA was unrunnable here was inferred from a `Cargo.toml` pin and was wrong.

**Durable rule:** An inferred capability claim that turns out to be wrong costs more than the investigation it prevented. Never suppress an inferred bad fact without verifying it directly against a fresh environment.

## 2026-07-29 — Standing Operational Rule: Worktree lifecycle and decision merging

**By:** Justin Chu (recorded by Scribe)

Do not delete a worktree before Scribe has merged its decision inbox. `.squad/decisions/inbox/` is gitignored and per-worktree, so removing a worktree destroys any unmerged decision drops in it. Coordination workflow:

1. Agent writes decision to `.squad/decisions/inbox/{agent}-{slug}.md` (in-worktree, gitignored).
2. PR lands; worktree remains temporarily.
3. Scribe runs (either same session or next): merges inbox → `.squad/decisions.md`, deletes merged files.
4. Scribe commits and pushes.
5. Safe to delete the worktree.

Merging and deleting the inbox files produces no git diff (expected, not a failure). The loss that occurred here: Rachael, Zhora, Leon wrote inbox files in separate worktrees, those worktrees were deleted before Scribe ran, and inbox files were lost. The substance survived in merged `history.md` files and PR descriptions, but the durable-rule fragments did not make it to this ledger — they had to be manually recovered from context.

### 2026-07-29: stream parsed tool calls as OpenAI deltas
**By:** McClane
**What:** Emit one metadata delta followed by an arguments delta for every parsed tool call, then finish with `tool_calls`.
**Why:** Clients can assemble tool invocations incrementally without receiving a monolithic completed tool-call object, while retaining full-output parsing for Qwen, Llama, and Mistral safety.

### 2026-07-29: CUDA operator parity batch 10
**By:** Ernie
**What:** Added CUDA kernels and CPU-parity coverage for AffineGrid, BatchNormalization, Compress, DynamicQuantizeLinear, GlobalAveragePool, GlobalLpPool, GlobalMaxPool, and LpNormalization. Deferred CenterCropPad, Col2Im, ConvTranspose, GridSample, GroupNormalization, InstanceNormalization, LpPool, NonMaxSuppression, QLinearMatMul, Resize, Unique, and com.microsoft FusedAttention.
**Why:** The selected operators form a reviewable low-risk batch around fixed-width transforms, channel-wise normalization, and block reductions. Heavy geometry, convolution, detection, and data-dependent operators need dedicated follow-up waves.


### 2026-07-29: Native CUDA versus ORT CUDA decode standing
**By:** Doug

**What:** On commit `37d87e27c6272dc1ab7a44138c21318f23794b9f`, NVIDIA H200
(GPU 0, ~143 GB, idle), CPU-pinned to core 1, steady greedy decode with the same
prompt (`Explain what a transformer is in two sentences.`), 128 generated tokens,
2 warmups, 3 measured runs, and 8 skipped decode tokens:

| Foundry Local model | Native CUDA | ORT CUDA | Native/ORT | Directive |
|---|---:|---:|---:|---|
| Qwen2.5-7B Instruct int4 | 252.09 tok/s | 273.08 tok/s | 0.923x | **FAIL** |
| Phi-4 Mini Instruct | 279.53 tok/s | 232.43 tok/s | 1.203x | **PASS** |
| Qwen2.5-1.5B Instruct int4 | 563.49 tok/s | 433.81 tok/s | 1.299x | **PASS** |

Qwen2.5-7B values are medians across two alternating benchmark groups. Native
group medians were 252.37 and 251.80 tok/s; ORT medians were 274.64 and 271.52
tok/s. Native is 20.995 tok/s (7.69%) slower than ORT and 16.53% below the
historical ~302 tok/s native baseline. Within-group run spread was <=0.35 tok/s
native and <=1.86 tok/s ORT. Generated text was coherent for both backends on all
three models; native Qwen2.5-7B showed no garbled or repeated-token regression.

Native beats ORT substantially on the smaller models, so the principal
opportunity is model/shape-specific in the 7B int4 decode path, especially
MatMulNBits memory traffic, kernel selection, and launch/capture overhead rather
than a universal engine-loop problem. Native prefill was also much slower than
ORT (Qwen2.5-7B ~66.6 ms versus ~11.5 ms), though prefill is excluded from the
standing steady-decode comparison. A requested native trace was not emitted by
the real-model flow, so detailed per-op attribution remains a benchmark-harness
follow-up.

Build:
```bash
cd /home/justinchu/onnx-genai
source .cudaenv.sh
CUDA_VISIBLE_DEVICES=0 taskset -c 1 cargo build --release \
  -p onnx-genai-bench --bin profile_native \
  --features bench-native,bench-ort,cuda
```

Run template (repeated with each model and `BACKEND=native`, then `ort`):
```bash
source /home/justinchu/onnx-genai/.cudaenv.sh
export ONNX_GENAI_ORT_LIB="$ORT_ROOT/lib/libonnxruntime.so.1.27.0"
CUDA_VISIBLE_DEVICES=0 taskset -c 1 \
  /home/justinchu/onnx-genai/target/release/profile_native \
  --model "$MODEL" --ep cuda --backend "$BACKEND" --steady \
  --tokens 128 --warmups 2 --runs 3 --decode-skip 8 \
  --prompt 'Explain what a transformer is in two sentences.'
```

Model paths:
```text
/home/justinchu/.foundry/cache/models/Microsoft/qwen2.5-7b-instruct-cuda-gpu-4/v4
/home/justinchu/.foundry/cache/models/Microsoft/Phi-4-mini-instruct-cuda-gpu-5/v5
/home/justinchu/.foundry/cache/models/Microsoft/qwen2.5-1.5b-instruct-cuda-gpu-4/v4
```

**Why:** The standing directive requires native CUDA EP decode to outperform ORT
on Foundry Local models. Current main passes on Phi-4 Mini and Qwen2.5-1.5B but
fails on the primary Qwen2.5-7B target, with a material regression from the
historical native baseline.


### 2026-07-29: Restore CUDA SiLU decomposition fusion
**By:** Kuato
**What:** Bisected the 302→252 tok/s regression to `598f8622` (#202), not KV capacity (`f0a4e7f1`), weight paging (`8bbf0332`), or compute-in-place (`fc262e6f`/`74d50f6e`). Restored CUDA-scoped `x * Sigmoid(x)` lowering and preserved the exported three-op fp16 rounding through the fused SwiGLU and MatMulNBits paths. Qwen2.5-7B improved 252.04→307.83 tok/s versus ORT 272.44; Phi-4-mini improved 279.54→314.91 versus ORT 231.85; Qwen2.5-1.5B improved 562.61→717.37 versus ORT 443.83. Fixed native token IDs are identical to pre-fix native IDs.
**Why:** `598f8622` removed session-global `fuse_silu_patterns` while moving private fusions to provider scope, but restored SiLU decomposition only for CPU. CUDA stopped forming its tagged SwiGLU and paired gate/up MatMulNBits fusion: Qwen2.5-7B grew from 113 to 141 MatMulNBits calls and added 28 Sigmoid plus 56 Mul calls per forward. CUDA-owned lowering restores the intended fusion without exposing private ops to other providers; a marker selects an epilogue reproducing the original fp16 Sigmoid and Mul rounding exactly.


### 2026-07-29: #377 explicit metadata fields

**By:** Cohaagen

**What:**

New explicit inference-metadata fields that replace the remaining name-guessing
sites tracked on #377 (after #380 / #382). Every field describes GRAPH
STRUCTURE, never a model family. Mobius/export (Benny's workstream) must emit
exactly these names/types so the runtime never falls back to a name guess.

1. `pipeline.strategy.inner_embedding_output` — `Option<String>` (non-empty).
   - Replaces: `nested_autoregressive.rs::resolve_inner_embedding_output`, which
     picked the inner decoder's per-code embedding output by guessing "the sole
     output whose name does not contain `logits` and is not a present-KV port"
     (`to_ascii_lowercase().contains("logits")` + `is_present_output`).
   - Semantics: the inner (code-predictor) decoder output port whose value is
     threaded back into the next inner step's `inputs_embeds` seed. Declared on
     the `nested_autoregressive` strategy (top-level or composite stage),
     alongside `outer`/`inner`/`num_code_groups`. Absent ⇒ actionable ERROR
     naming `pipeline.strategy.inner_embedding_output`.

2. `model.io.static_cache` — `Option<StaticCacheIoSpec>`.
   New struct `StaticCacheIoSpec`:
   - `write_indices_input: String` (non-empty) — scatter write-position input
     (was hardcoded `"write_indices"`).
   - `kv_sequence_length_input: String` (non-empty) — non-pad KV sequence-length
     input (was hardcoded `"nonpad_kv_seqlen"`).
   - `key_cache_inputs: Vec<String>` / `value_cache_inputs: Vec<String>` —
     per-layer static K/V cache buffer inputs, positional per layer (were
     hardcoded `"key_cache.{i}"` / `"value_cache.{i}"`).
   - `key_cache_outputs: Vec<String>` / `value_cache_outputs: Vec<String>` —
     per-layer updated K/V cache outputs, positionally paired with the inputs
     (were hardcoded `"updated_key_cache.{i}"` / `"updated_value_cache.{i}"`).
   - Replaces: `decode/io.rs::detect_static_cache`, which selected the
     TensorScatter static-cache ABI by hardcoded port names (int vectors are
     shape-indistinguishable, so shape alone cannot disambiguate). When
     `model.io.static_cache` is present it is authoritative and name-agnostic;
     the four cache lists must be equal length and pair positionally. Declared
     but inconsistent (unequal lengths, missing ports) ⇒ ERROR naming the key.

3. (compatibility emission, no schema field) encoder prompt-input role.
   - Replaces: `compatibility.rs:1152` `encoder_input_field.ends_with("audio_features")`.
   - The role (`audio_features_input` vs `token_input`) is now taken directly
     from WHICH explicit genai-config field the exporter declared
     (`model.encoder.inputs.audio_features` vs `.input_ids`), captured in the
     existing match, never re-derived by string-matching the port name.

**Still name-based after this change (documented, needs contract before removal
— unchanged from #382's deferral, NOT regressed here):**
- Paged-KV bridge geometry (`engine/kv_bridge.rs`): key/value substring +
  `kv_layer_index` + `matching_past_input`. Made metadata-authoritative when
  `model.io.kv_inputs`/`kv_outputs` are declared; name matching remains only for
  the no-metadata path. CUDA-only, correctness-critical (mis-resolved port
  corrupts generation), no CPU fixture — full removal deferred with the paged
  contract.
- `decode_contract.rs` KV name convention (`kv_suffix`, `KvNamingConvention`):
  still consumed by the off-limits #99 speculative proposers; cannot be deleted
  from this workstream.

**Why:** Justin's #377 directive — ALL inference/pipeline metadata except io
SHAPE must be EXPLICIT and GENERAL; only io-shape may disambiguate. Name
guessing / historical-name fallback must be replaced by explicit metadata plus a
clear ERROR (naming the exact missing key) when the required metadata is absent.
These fields let the exporter state graph structure directly so the runtime
never interprets a graph port name.

---

**SHIPPED (2026-07-29, branch `squad/377-explicit-metadata`):** All three fields
above landed exactly as specified — `PipelineStrategy.inner_embedding_output:
Option<String>`, `ModelIoSpec.static_cache: Option<StaticCacheIoSpec>` (with
`write_indices_input`, `kv_sequence_length_input`, `key_cache_inputs`,
`value_cache_inputs`, `key_cache_outputs`, `value_cache_outputs`), and the
encoder-role emission change (no new schema field). Committed regenerated
`schema/inference_metadata.schema.json`. Benny/mobius: emit these names verbatim.

---

### 2026-07-29: #377 follow-up — closed remaining name fallbacks (paged-KV, nested-AR logits)
**By:** Cohaagen
**What:** Per Justin's "don't defer", removed the last two non-off-limits
name-guessing fallbacks. NO new schema fields were needed — both reuse existing
`ModelIoSpec` ports.

1. **Paged-KV bridge geometry (`engine/kv_bridge.rs`) — now fully name-free.**
   - Removed the name-based `is_present_output` / `to_ascii_lowercase().contains("key"|"value")`
     / `kv_layer_index` / `matching_past_input` resolution.
   - `infer_kv_model_info(session, io, page_size, dtype)` now pairs layers purely
     from explicit `model.io.kv_inputs`/`kv_outputs` (equal-length, positional
     per-layer `[key, value]`) via the pure, session-free, unit-tested
     `pair_kv_ports` helper. Extracted `resolve_kv_layers`, `require_present_kv_output`,
     `require_kv_input`.
   - No metadata ⇒ **`Ok(None)`** (build no paged cache), never a name/shape guess:
     a growing paged present output is shape-indistinguishable from a static-cache
     buffer or a logits/hidden output. KV correctness for the decode loop is
     enforced independently by the decode-path resolver (`decode::resolved_io`),
     which already fails closed naming `model.io.kv_inputs`/`kv_outputs`.
   - Declaring only one of `kv_inputs`/`kv_outputs` ⇒ ERROR naming both keys.

2. **Nested-AR `logits` output — explicit, no `contains("logits")` guess.**
   - `named_output` reduced to exact-match; the `contains` substring fallback removed.
   - `NestedAutoregressivePlan` now carries `outer_logits_output` / `inner_logits_output`
     from each component's explicit `models.{component}.io.logits_output`
     (`require_component_logits_output` errors naming the key when absent/empty).

3. **Decode-state I/O threading (completes the mission end-to-end).** The nested-AR
   loop resolved the talker/code_predictor from tensor SHAPE (`DecodeState::new`,
   io=None). Now threads each component's explicit `io` via
   `DecodeState::new_with_io` (new `NestedAutoregressivePlan.outer_io`/`inner_io`).
   Fixtures updated to declare the required explicit ports (`token_input` /
   `sequence_source: inputs_embeds` + `inputs_embeds_input`, `position_ids_input`,
   `logits_output`, `kv_inputs`/`kv_outputs`): `tiny-tts-nested`,
   `tiny-tts-nested-preembed`, `tiny-tts-nested-prefill`, plus the inline
   `pipeline_executor` and `optional_modality_pipeline_e2e` test fixtures.

**Only remaining name path:** `decode_contract.rs` KV name convention
(`kv_suffix`, `name_contains_past_key_value`, `KvNamingConvention`) — consumed
ONLY by the off-limits #99 speculative-decoding proposers; left intact per scope.

**Why:** Justin's #377 directive — no deferral; explicit metadata + actionable
errors, only io-SHAPE may disambiguate. Paged-KV pairing cannot be disambiguated
by shape, so it is metadata-only with a clear error, and the paged resolver is
now name-free.


### 2026-07-29: Static-cache scatter ABI is explicit-or-error (no name-guessing)
**By:** Matthias
**What:** Removed `detect_static_cache_by_convention` from `crates/onnx-genai-ort/src/decode/io.rs`. A TensorScatter static-cache graph is now bound ONLY from `model.io.static_cache`; a scatter-shaped graph without that block fails closed with an error naming `model.io.static_cache` instead of binding by the hardcoded `write_indices`/`nonpad_kv_seqlen`/`key_cache.{i}`/`updated_key_cache.{i}` ports. In-repo static-cache fixtures (`tests/fixtures/tiny-llm-scatter`, engine `model-package-cpu`) now declare the block explicitly.
**Why:** Completes #377 for the static-cache path (Quaid's PR #412 REQUEST-CHANGES blocker): the scatter control ports are shape-indistinguishable integers, so per #377 they must be explicit or an error — never inferred from port names. `StaticCacheAbi::classify` (the explicit path) stays authoritative and name-agnostic; #99 specdec `KvNamingConvention` is untouched. Exporters/authors emitting static-cache graphs MUST now declare `model.io.static_cache` (write_indices_input, kv_sequence_length_input, and equal-length positionally-paired key/value cache input/output lists).


### 2026-07-29: Registry-backed model warmup
**By:** Lull
**What:** Added an opt-in `warmup` per-model setting and `POST /v1/admin/models/{id}/warm`. Both use `ModelRegistry::warmup`, which performs one deterministic generated token and records a successful warmup idempotently.
**Why:** The first generation initializes lazy runtime allocations; sharing the registry method keeps configured and on-demand warmups identical while allowing failures to be retried without corrupting registry state.


### 2026-07-29: Preserve warmup error categories at the admin boundary
**By:** Rachael
**What:** `ModelRegistry::warmup` now returns typed absent-model, registry, and runtime-failure errors; the admin warm endpoint maps them to 404, 500, and 500 respectively.
**Why:** A loaded model's failed warmup must not be reported as an unloaded-model 404.


### 2026-07-29: CPU PackedVarlenAttention shares the scalar SDPA core
**By:** Johnny
**What:** Register `pkg.nxrt::PackedVarlenAttention` on the CPU EP with the CUDA v1 schema, while factoring packed segment execution into a helper shared with CPU `VarlenAttention`.
**Why:** Sharing f32 accumulation, split-sqrt scaling, softcap, GQA grouping, and tail-aligned causal masking prevents CPU packed and padded-entry kernels from drifting numerically.

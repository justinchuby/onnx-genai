# Decisions — live standing directives

Last consolidated: 2026-07-31T10:24:07Z (Scribe round 9 — 27B decode profile Scan=56.5%, Scan-capture scoping PENDING JUSTIN, #554 session-reuse merged; round-8 in archive)

Standing governance rules and constraints. Dated wave records and historical ledger updates
are archived to `.squad/decisions-archive/2026-07.md`.

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

## Active historical pointers

For per-PR narrative use `.squad/decisions-archive/2026-07.md`. Archived there: consolidation
checkpoints (2026-07-28 size-gate snapshots; 07-29 compactions; rounds 2–7 CUDA/native/MoE +
native-pipeline + CUDA-hybrid wave records; prior `.squad/decisions/archive/`); Mac CPU EP topics
(#227 roofline, load-adaptive opt-in, Apple Silicon portability, BNNS prefill/deprecation,
benchmark-CI rule, dispatch-manifest lint, 1×1 Conv + SDPA corrections, GEMV notes); Wave 8/9
(CUDA coverage batches 8/9, shape-inference catalog batches 3/4, NCHWc gating, reviewer-lockout).

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
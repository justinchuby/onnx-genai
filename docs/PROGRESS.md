# onnx-genai — Implementation Progress

Tracks implementation status of `docs/DESIGN.md` (§1–§40) plus the ORT 2.0 from-scratch
Rust runtime track. This is a curated snapshot of current state with a short historical
spine — full narrative lives in `.squad/decisions-archive/` and `docs/benchmarks/`.

**Published:** `onnx-genai` v0.1.0 + 8 sub-crates on crates.io; the `onnx-runtime-*` layer
(including `onnx-runtime-tracer`) is released as v0.1.0-dev.1. The two ORT plugin-EP
cdylibs are both LIVE on PyPI at **0.1.0.dev5**: **`nxrt-ep-cpu`** (manylinux_2_28 +
macOS-arm64 + win_amd64) and **`nxrt-ep-cuda`** (CUDA 13, manylinux_2_28_x86_64).
CI = fmt/build/test + **blocking clippy** + Miri unsafe-crate soundness + scheduled
`cargo-audit`. Coverage ~77% line.

_Last updated: 2026-08-13T03:08:00Z_

**Current `origin/main` implementation HEAD:** `b871c869`.

---

## Current status (snapshot — where the engine is today)

- **Native CUDA beats onnxruntime-genai-cuda on every on-box int4 dense model** — Qwen2.5
  0.5B/1.5B/7B, Qwen3-0.6B, Phi-4-mini, DeepSeek-Coder-1.3B, DeepSeek-R1-1.5B — each
  bit-exact or native-more-accurate vs an fp32 oracle, zero fallbacks. The ORT 1.28
  three-config fairness benchmark measured native **1.23–2.74×** faster than
  ORT-GenAI-direct (Qwen2.5-0.5B 557 vs 203 tok/s = 2.74×; DeepSeek-R1-1.5B 1.23×).
- **Muse-Glimmer-30B (dense int4, bf16 decoder, heavy GQA) decodes at ORT parity on
  native CUDA** — **11.4 → 40.21 tok/s** (matches ORT's ~40) after a 4-gate CUDA-graph
  capture chain (#848/#850/#852/#855/#854 → 1 segment / 0 seams) plus a bf16 RMSNorm
  cast-fold + parallel f32 tree reduction (#860). Capture collapses ~1600 launches/token
  into one replay; first-16 greedy ids match reference.
- **Large / hybrid models run native-only** where ORT cannot load them: GLM-4-9B
  (partial-RoPE GQA, ORT rejects the schema), DeepSeek-V2-Lite (MLA + QMoE), and
  Qwen3.5/3.6-**27B** hybrid Gated-DeltaNet — all load and decode on native CUDA via a
  DRY, graph-derived io contract (no model-name gates).
- **35B-A3B QMoE is native-only** (ORT 1.28 crashes on it through both the backend and
  GenAI-direct paths). The fused QMoE decode kernel reaches **~90 tok/s (~11.13 ms/tok,
  ~33× vs the dense baseline)**, byte-exact against the fp32 teacher-forced oracle. QMoE
  decode is now occupancy/HBM-bandwidth-bound; surgical single-op fusion is exhausted
  (four experiments — FC2 fusion, ILP-2, DP4A, persistent — all NO-SHIP).
- **CPU EP** has broad ONNX op coverage (backend node conformance grew well past 921
  cases) plus a correctness fix (k-major TopK output layout for non-final axis) and a
  partial-select TopK perf path.
- **Memory / VMM:** managed no-spill VMM is the default with automatic weight streaming
  when a model exceeds budget; a fitting model does not page. KV residency is
  **layout-governed** — the committed floor drops from ~1.5 GiB (head-major) to ~192 MiB
  (seq-major) to ~2 MiB/seq (token-major), up to **768×**. Weight offload and CUDA-graph
  capture now coexist (stable-VA paging), and page-level prefix sharing is proven under
  captured replay (ledger charges once; extra sharers cost 0 bytes).
- **EP extensibility:** our Rust CPU/CUDA EPs run *inside* upstream ONNX Runtime via the
  plugin-EP C ABI, packaged and published to PyPI as `nxrt-ep-cpu` / `nxrt-ep-cuda`.
- **ORT 2.0 track:** the from-scratch pure-Rust runtime has all Phase-1 crates merged
  (`bert_toy` matches onnxruntime 1.27 CPUEP to fp32 rounding), Phase-2 symbolic shape
  inference wired into the loader, and the EPContext plugin-EP contract designed/landed at
  the ep-api layer.

### DESIGN §1–§40 status (condensed)

| § | Area | Status |
|---|------|--------|
| 1–8 | Vision, architecture, core components, crates, deps | ✅ Done |
| 9 | HTTP API surface | 🟡 chat/completions/models/sessions/status/metrics/audio/embeddings/logprobs/debug + Perfetto trace ✅; OTLP deferred |
| 11,12,15 | Testing, decisions | ✅ Done (~77% coverage) |
| 16 | Quantized models | ✅ EP-select + int8/fp8 selectable KV storage |
| 17,29 | Image + language diffusion | 🟡 executor seam + DDIM/CFG on real DiT ✅; full image/language e2e pending |
| 18–19 | ORT wrapper, dep graph | ✅ Done |
| 20 | Generalized pipeline | 🟡 AR/composite/vision/audio ✅; iterative/diffusion seam ✅ |
| 21–25 | Tool use, grammar, FIM, sampling, extensibility | ✅ Done |
| 26 | Multi-agent serving | ✅ batched continuous (~6× throughput) |
| 27,28 | Speculative decode | ✅ draft/prompt-lookup/MTP/EAGLE-3/Gemma4 shared-KV; vLLM speculator compat |
| 31,32 | Observability, metrics | 🟡 metrics/status/trace/debug ✅; OTLP deferred |
| 34 | Cluster/session router | ✅ `onnx-genai-router` crate |
| 35 | Native preprocessing | ✅ image + audio log-mel, tiling |
| 36,37 | Backpressure, model lifecycle | ✅ admission/429; multi-model registry + load/unload/LRU |
| 38 | Distributed KV connector | ✅ pluggable trait + local-tiered backend (real byte materialization) |
| 39 | Paged/radix attention | 🟡 Mobius block-table KV draft (mobius#395); runtime wiring pending |
| 40 | Sliding-window attention | 🟡 contiguous SWA + attention-sink ✅; per-layer hybrid deferred |

---

## Recent milestones (2026-07-28 → 2026-08-13) — newest first

### 2026-08-13 — Muse-Glimmer-30B native CUDA decode reaches ORT parity (40 tok/s)

- **11.4 → 40.21 tok/s (native now matches ORT's ~40) on Muse-Glimmer-30B** (dense int4,
  52 layers, **bf16** decoder, hidden 6656, heavy GQA num_kv_heads=2, vocab 202048). The
  decode was dispatch/launch-overhead bound (~1600 kernel launches/token, GPU ~99% idle);
  the fix was a 4-gate CUDA-graph-capture chain followed by a kernel/graph lever:
  - **Gate 1 — classify onto shared-buffer KV (#848):** vestigial `sliding_window`
    detected via graph truth (`local_window_size`) so Muse-Glimmer lands on the
    capture-stable fixed-capacity KV path, not the growing/paged path.
  - **Gate 2 — native pipeline embed load (#850):** `PipelineEngine` runs the embedding
    component on the native CUDA EP, so the model loads + greedy-decodes end-to-end on
    `--pipeline --backend native --ep cuda`.
  - **Gate 3 — GQA KV seq-symbol pin (#852):** pins the 52 GQA nodes' fixed-capacity KV
    seq symbols, dropping the classifier's disqualifying-symbol set 53 → 0.
  - **Gate 4 — bf16 capture-safe GQA decode kernel (#855/#854):** new `gqa_decode_bf16`
    device-length split-K decode (fp32 accumulation) admits bf16 q_seq==1 aliased device
    KV as capture-safe, plus a skip-norm capture-safety flag fix → **54 → 1 segment / 0
    seams**, lifting decode to ~23 tok/s.
- **Cast / RMSNorm elimination lever (#860):** generalized the ep-cuda
  `CudaDropNormalizationCasts` pass to fold **bf16** casts around **`RMSNormalization`**
  (Muse-Glimmer wraps all 312 RMSNorm nodes in `Cast(bf16→f32)→RMSNorm→Cast(f32→bf16)`,
  624 of 834 decoder casts). The fold **op-swaps `RMSNormalization` →
  `SimplifiedLayerNormalization`** so the session's post-optimization shape re-inference
  stays bf16 (ONNX RMSNormalization output Y follows the *scale* dtype `V`, not activation
  `T`; both ops map to the same fused `RmsNormKernel`). Honest attribution: the cast-fold
  alone is ~free under capture (**23.16 → 23.43 tok/s** — casts are cheap once launches are
  captured); the real lever is `rmsnorm_bf16`'s **parallel f32 tree reduction** replacing
  the serial single-thread mean-square (~40% of captured decode at M=1) → **23.16 → 40.21
  tok/s** (median of 40.13/40.29/40.21).
- **Correctness / capture:** capture stays **1 segment / 0 seams**, first-16 greedy ids
  match the reference exactly. The parallel tree reduction is full f32 precision and, per
  Chew's numerics review 🟢, **~807× more accurate than the old serial order** (within 1
  bf16 ulp of an f64 oracle at hidden 6656). Greedy decode is byte-exact for the first ~37
  tokens then shows expected sub-ulp greedy sensitivity (accuracy-level-4 int4 quant);
  `ONNX_GENAI_CUDA_DISABLE_NORM_CAST_FOLD=1` restores the strict CPU-order byte-exact
  serial path (at ~23 tok/s).

### 2026-08-12 — EP plugins run inside ONNX Runtime and ship on PyPI

- **EP plugin export ✅ (#762, `e9c0ab6a`):** our Rust CPU/CUDA execution providers now run
  *inside* upstream ONNX Runtime through the plugin-EP C ABI
  (`CreateEpFactories`/`ReleaseEpFactory`, loaded via `RegisterExecutionProviderLibrary`).
  Six new crates (shared plugin adapter + CPU/CUDA cdylibs + native `nxrt` ABI + dlopen
  host + test plugin). EP conformance suite (`NXRT_REQUIRE_ORT_TESTS=1`) enforced in CI.
- **PyPI publish pipeline ✅ (#819/#824):** `.github/workflows/publish-ep-plugins.yml`
  packages the two cdylibs with **setuptools + plain cargo (not maturin)** — they are
  cdylibs exporting the ORT plugin ABI, not PyO3 modules, and must **not** link
  `libonnxruntime`. **Both `nxrt-ep-cpu` and `nxrt-ep-cuda` 0.1.0.dev5 are LIVE.** The CUDA
  job builds on the standard
  `manylinux_2_28` image; because `onnx-runtime-ep-cuda` uses cudarc `dynamic-loading`, the
  wheel needs **no CUDA toolkit or GPU** to build — the four NVIDIA runtime wheels are
  required deps pinned `>=13,<14` (CUDA 13).
- **Test-quality follow-ups ✅ (#820):** closed 3 gaps from the #762 review (real
  fail-closed CUDA assertion, all 28 CPU fixtures regenerate byte-identical, f16/bf16
  optional-slot value oracles).

### 2026-08-12 — Memory-safety wave: absent optional-output machinery (#762 review)

- **Heap-overflow + misroute fixes ✅:** scratch buffers for absent optional outputs were
  sized from the slot dtype (2 bytes for f16/bf16) but `TensorMut` was hardcoded to
  Float32 — a 2× heap overflow on every f16/bf16 op with an omitted optional output. Now
  dtype-derived and fail-closed on Undefined. A separate routed-path bug (positional
  compaction of absent slots) was fixed with a `RoutedSlotKind` enum that keeps every slot
  index aligned end-to-end. 280 tests pass; Miri 4/4 canary tests clean.
- **Lesson recorded:** the absent-slot machinery has now produced four distinct defects —
  any change to optional-slot handling gets disproportionate scrutiny; canaries must mirror
  production allocation exactly.

### 2026-08-12 — VMM / KV-layout / offload / batching residency wave (#736 audit, #755–#814)

- **Managed no-spill VMM is the default (#755/#798)** with automatic weight streaming when a
  model exceeds budget; a fitting model stays `FullResident` with 0 page-ins.
- **KV residency is layout-determined (#787/#792/#783):** the `KvLayout` enum was replaced
  by a KV-cache stride descriptor (layout is a queried per-EP, per-platform capability, not
  a constant). Committed floor: 768 granules (~1.5 GiB) head-major → 96 (~192 MiB) seq-major
  → 1/seq (~2 MiB) token-major = **768× reduction**. Strided reads are DRAM-bound
  independent of stride, so seq/token-major layouts cost no measurable bandwidth.
- **Offload + capture coexist (#796/#716):** offloaded weights page under a stable VA
  (page-in remaps physical granules instead of returning a new pointer), so weight offload
  no longer disables CUDA-graph capture.
- **Prefix sharing is sound (#793/#803/#809/#822/#777):** one physical handle maps into
  N=8 sequences under captured replay; the ledger charges once and additional sharers cost
  0 bytes. Seq-major is refused on head-major-only KV consumers (#812).
- **Fewer graph invalidations (#811):** CUDA-graph invalidation on KV growth is now
  conditional — seq-major keep drops growth invalidations 4→0.
- **#736 over-reservation audit (six slices):** 4/5 completed slices found *over-reservation*
  (bytes charged on a path that never uses them), not ungoverned allocation — IndexShare
  (#751), GQA WS_SCORES (#795, ~128 MiB f32-only), cuBLASLt GEMM (#799), default-domain
  Attention scores (#802), GQA QKV staging (#806), GQA BNSH transpose (#810), GQA workspace
  (#814), default staged KV (#813). Guidance: *start from use, not from allocation.*
- **Method hardening (#807/#797/#801/#804):** order-dependent test state produced two wrong
  conclusions this week; a debug-only freeze guard + single-stream helper now make
  order-dependence loud. Native batching capability is observable/honest (#750/#758).

### 2026-08-11 — Issue triage + autonomous correctness fixes

- **~90 open issues triaged, 18 stale closed**, and five fixes shipped: CUDA
  `GatherBlockQuantized` now applies the symmetric default zero-point `1<<(bits-1)` when
  absent (#785/#702); ORT recurrent-state reuse guard + loader error dedup (#786/#701/#467);
  working VLM compat fixture + re-enabled server CI (#788/#686); DRY decoder-io derivation
  glue into a shared helper (#784); CI test-honesty whitelist (#789).
- **CPU-EP TopK ✅:** k-major output layout for non-final-axis TopK (#774, correctness — was
  emitting `[outer][inner][k]` instead of the required `[outer][k][inner]`) plus a
  partial-select perf path (`select_nth_unstable_by`, O(width) instead of a full sort, #775).
- **mobius io-metadata robustness** PR opened upstream (silent-skip of graph reload produced
  thin metadata); never self-merged.

### 2026-08-11 — Qwen3.5/3.6-27B hybrid Gated-DeltaNet on native CUDA (#779)

- **27B enabled end-to-end ✅:** the blocker was a thin `inference_metadata.yaml` (no `io`
  port contract), not a missing kernel — the required GDN/GQA/int4 kernels already existed.
  `maybe_fill_hybrid_io_from_graph` auto-derives the decoder io contract from the ONNX graph
  (gated on non-empty state_pairs), so all hybrid GDN models load. Byte-exact fp32 oracle
  (argmax 11751 " Paris", margin 2.549 nats). DRY, no model-name gates.

### 2026-08-11 — GLM-4-9B + DeepSeek-V2-Lite native + ORT 1.28 fairness

- **GLM-4-9B ✅ (#770):** the blocker was native KV reservation using metadata
  `max_sequence_length` (131072 → oversized reservation → load fail), *not* partial-rotary
  (native GQA already honors `rotary_embedding_dim`). Fix honors the runtime CUDA KV cap
  first. GLM-4-9B decodes coherently native-only (ORT cannot load its schema).
- **DeepSeek-V2-Lite ✅ (#771):** QMoE scale inputs arrived as `Cast(fp16 initializer→fp32)`
  rather than direct initializers; static placement now accepts a one-hop default-domain
  `Cast(initializer)`. Not an MLA-kernel gap.
- **ORT 1.28 three-config fairness benchmark ✅ (#766):** native CUDA vs ORT-as-backend vs
  ORT-GenAI-direct, greedy temp=0 with token-parity checks. Native is **1.23–2.74×** faster
  than ORT-GenAI-direct; ORT (both paths) crashes on 35B-A3B QMoE, so native is the only
  runtime that runs it. See `docs/benchmarks/2026-08-11-ort128-3config-fairness.md`.
  CI feature-gating fix (#773) kept the ORT-only build green.

### 2026-08-10 — Fused QMoE decode kernel; QMoE surgical-optimization arc concluded

- **Fused QMoE decode kernel ✅ (#765):** fused FC1 gate/up + SwiGLU (down/combine
  unchanged) eliminates the `qmoe_activate` launch + FC1 scratch round-trip. 35B-A3B decode
  11.511 → **11.126 ms/tok (~3.3%, ~90 tok/s, ~33× vs dense)**; argmax-stable within fp32
  parity tolerance. Preceded by a barrier/launch tune (#764).
- **Arc concluded:** QMoE decode at batch-1 (each expert count=1 → tiny GEMVs) is
  occupancy/HBM-bandwidth-bound. FC2/down+combine fusion (+0.08%), ILP-2 (regressed),
  int4 DP4A/128-bit vec-read (+4.9%), and a persistent single-op kernel (+7.4%) were all
  NO-SHIP — any warp-for-width/fusion trade loses. Correctness risk is ~zero (the oracle
  held byte-identical through every experiment).

### 2026-08-10 — CUDA-graph capture trilogy, version-selectable CUDA, megakernel study

- **Capture trilogy merged:** C1 growing-symbol classifier for capture-eligible pointwise
  ops + re-anchored 35B oracle on fp32 teacher-forcing (#728, closing #722); C2
  LinearAttention capture seams (#757, capture-aware kernel sync); C3 (#708).
- **User-selectable CUDA version ✅ (#760):** cudarc 0.19 compiles with exactly one
  `cuda-1xxxx` feature; a loud single-version compile-time guard (`onnx-genai-cuda-version-
  guard`) fires a friendly `compile_error!` before the ~379-error cudarc cascade. Default
  `cuda-13000`.
- **Megakernel feasibility (#769, docs/education only):** a whole-step megakernel is the
  real remaining batch-1 latency lever, but it is multi-week/high-risk and **deferred**;
  vLLM (`full_cuda_graph`) and llama.cpp do not have a true megakernel (same layer as us),
  while Mirage MPK / Hazy "Look Ma, No Bubbles!" are the frontier references. A per-op
  persistent QMoE kernel is Amdahl-capped (~23%) and regressed in practice.

---

## Foundational milestones (2026-07-15 → 2026-07-27) — compressed

The runtime was built from scaffold and published during this window; the following is a
short spine (full day-by-day is archived).

- **Full generation stack built + published (2026-07-14→19):** onnx-genai v0.1.0 + 8
  sub-crates on crates.io; `onnx-runtime-*` v0.1.0-dev.1. Shipped samplers (fixed a
  categorical-sampling RNG bug that always returned token 0), FIM, grammar-constrained
  decoding, tool use (Hermes-verified), speculative decode (draft / prompt-lookup / MTP /
  EAGLE-3 / Gemma4 shared-KV), multi-session + prefix/paged/tiered/int8-fp8 KV, batched
  continuous serving (~6× throughput), OpenAI HTTP surface, observability + Perfetto trace,
  the `onnx-genai-router` (§34) and distributed KV connector (§38) crates, sliding-window +
  attention-sink attention (§40), and native image/audio preprocessing (§35). CPU-EP ONNX
  backend node conformance grew from ~687 to 921+ passing cases.
- **Diffusion / any-to-any (2026-07-19):** Mobius builds every model from scratch (no
  `torch.onnx.export`). Stable Diffusion 1.x renders end-to-end from the from-scratch UNet
  (diffusers parity ~1e-4); runtime LoRA is numerically validated via per-adapter gate
  inputs (switch/blend, no re-export); live pipeline overrides (steps/cfg/scheduler); a
  from-scratch LLaDA masked-diffusion LM is parity-validated (max|Δ| 1.5e-7); the MLX EP
  runs diffusion ~4× faster; composite any-to-any pipelines (audio-to-audio, VLM) proven.
- **Native CUDA int4 decode perf campaign (2026-07-16→23):** fp16 decode climbed
  200→789 tok/s across waves; segmented CUDA-graph capture; generic lm_head fusion
  (Llama-3.2-1B 97→449 tok/s, 4.6×); SwiGLU-RMS / int8 GEMV / block-128 MatMulNBits fusions
  flipped native positive vs fresh ORT GenAI 0.14.1 on Qwen2.5 0.5B/1.5B/7B and DeepSeek.
  GLM/DeepSeek DSA `IndexShare` + MLA landed; VLM enablement + Gemma4 E2B; metadata-driven
  CUDA-graph auto-enable.
- **Correctness + fairness hardening (2026-07-24→27):** accuracy-level-4 int8-activation
  correctness (#123) and opt-in fp16 decode (#127, ~1.9× payoff); a per-model native-CUDA
  decode-correctness regression lock for every on-box model; Foundry Qwen3-0.6B whole-graph
  CUDA enablement; Phi-4-mini beats ORT (+36–43%); GLM-5.2 synthetic tiny-QMoE native e2e;
  trustworthy uncontended-H200 native-vs-ORT sweeps; and Miri unsafe-crate soundness
  enforced in a dedicated CI workflow.

### ORT 2.0 — from-scratch pure-Rust runtime (parallel track, `docs/ORT2.md`)

- **Phase-1 ✅ all six `onnx-runtime-*` crates merged** (ir / ep-api / loader / ep-cpu /
  session / capi), ~128 tests green. Exit milestone: `bert_toy_optimized.onnx` (384 nodes)
  runs end-to-end on the pure-Rust CPU EP and **matches onnxruntime 1.27.0 CPUEP to fp32
  rounding** (max_abs 1.19e-7), with zero cross-crate fixes needed on the first real run.
- **Phase-2 ✅ symbolic shape inference** (`onnx-runtime-shape-inference`, 40+ op handlers,
  `DimExpr` polynomial + shape-DATA propagation) is wired into the loader; the old
  const-fold-lite pass is retired and the session JIT is fallback-only.
- **Design:** `com.microsoft::EPContext` contrib-op fully specified (§55) with the ep-api
  registry + trait contract landed against a mock EP; a byte-exact ONNX encoder
  (IR→ModelProto) landed (STRING attrs are raw bytes, model-agnostic round-trip).

---

## Open items / known gaps

### DeepSeek native support
See [`deepseek-native-status-2026-07-25.md`](deepseek-native-status-2026-07-25.md).

- [ ] **Full-model QMoE coherence:** validate native CUDA QMoE with a complete DeepSeek-V2
  package and real tokenizer. The real-shape conformance artifact proves exact
  routing/token parity but decodes only decimal token IDs.
- [ ] **GPU-resident ORT QMoE baseline:** still unavailable — ORT 1.28 crashes on QMoE
  through both the backend and GenAI-direct paths (#766), and the earlier reference inserted
  four `Memcpy` nodes at 0% sustained GPU utilization. No native-vs-ORT QMoE perf claim is
  possible; native is a standalone number.
- [ ] **DeepSeek-R1 numerical-parity policy:** keep the fp32-oracle regression lock
  (`deepseek_r1_1_5b_divergence.rs` — native picks oracle-correct 374, ORT CUDA flips to
  315) and extend the oracle to the benchmark prompt where native/ORT diverge at a close
  MatMulNBits decision.
- [x] **DeepSeek-Coder dense int4:** native CUDA loads, emits coherent code, matches ORT
  CUDA for 128 greedy tokens.
- [x] **DeepSeek-V2 real-shape QMoE routing:** native CUDA loads and matches ORT for 32
  greedy tokens (token-0 top-40 log-prob max error 0.001409).
- [x] **DeepSeek-V2-Lite native load (#771):** Cast-backed QMoE scales now accepted; loads
  and decodes on native CUDA.

### GLM native support
See [`glm-native-status-2026-07-25.md`](glm-native-status-2026-07-25.md).

- [ ] **Restore GLM-5.2 dense q4 multi-token native decode:** the model emits token `110`
  then fails at `layers.0/self_attn/indexer/Add_node_70` (growing logical prefix cannot
  broadcast with `[1,1,4096]`). Restrict physical-mask exposure in
  `DecodeCudaState::extend_mask` to safe topologies (or keep a logical mask), then add
  `[123]` / `[1,2,3,4]` regressions. Regresses the historical 148.58 tok/s result.
- [ ] **ORT-compatible GLM-4 partial-RoPE reference:** the available ORT CUDA build rejects
  `rotary_embedding_dim` on `com.microsoft::GroupQueryAttention`, so GLM-4 token/log-prob
  parity and a legitimate native-vs-ORT throughput comparison remain unavailable. (Native
  GLM-4-9B itself now loads and decodes — #770.)
- [ ] **ORT-compatible GLM-5.2 QMoE reference:** ORT cannot load the conformance model
  (`pkg.nxrt::IndexShare` unregistered). Export a standard-op graph or provide an ORT custom
  op before making parity/speed claims.
- [ ] **Real-checkpoint GLM-5.2 QMoE validation:** the tiny random-weight model confirms
  native DSA/`IndexShare`/`QMoE` execution (~176 tok/s) but cannot establish
  natural-language coherence or real-model performance.
- [x] **GLM-4 native coherence:** native CUDA loads the real 9B int4 artifact, matches the
  golden prefix, and emits coherent text (~108 tok/s).
- [x] **GLM-5.2 tiny QMoE native execution:** native CUDA matches the committed 12-token
  CPU/CUDA anchor and completes deterministic 64-token decode.

### Packaging / infrastructure
- [x] **EP plugins on PyPI:** both `nxrt-ep-cpu` and `nxrt-ep-cuda` 0.1.0.dev5 are LIVE
  (CUDA wheel: manylinux_2_28_x86_64, CUDA 13 required deps; PR #824 merged).
- [ ] **mobius io-metadata robustness:** upstream PR open (reload the graph instead of the
  silent-skip that produced thin metadata); never self-merged.

### Performance research (deferred)
- [ ] **Whole-step megakernel (#769):** the remaining batch-1 latency lever (not
  Amdahl-capped), but multi-week/high-risk — deferred pending a go-ahead. Per-op persistent
  QMoE and int4 DP4A were empirically NO-SHIP (occupancy/bandwidth-bound).

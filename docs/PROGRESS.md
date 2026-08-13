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

_Last updated: 2026-08-13T06:34:00Z_

**Current `origin/main` implementation HEAD:** `887e3742`.

---

## Current status (snapshot — where the engine is today)

- **Native CUDA beats onnxruntime-genai-cuda on every on-box int4 dense model** — Qwen2.5
  0.5B/1.5B/7B, Qwen3-0.6B, Phi-4-mini, DeepSeek-Coder-1.3B, DeepSeek-R1-1.5B — each
  bit-exact or native-more-accurate vs an fp32 oracle, zero fallbacks. The ORT 1.28
  three-config fairness benchmark measured native **1.23–2.74×** faster than
  ORT-GenAI-direct (Qwen2.5-0.5B 557 vs 203 tok/s = 2.74×; DeepSeek-R1-1.5B 1.23×).
- **Muse-Glimmer-30B (dense int4, bf16 decoder, heavy GQA) decodes faster than ORT on
  native CUDA** — **11.4 → 47.25 tok/s (clearly beats ORT's ~40, +18%)** after a 4-gate
  CUDA-graph capture chain (#848/#850/#852/#855/#854 → 1 segment / 0 seams) plus a bf16
  RMSNorm cast-fold + parallel f32 tree reduction (#860 → 40.21) and a MatMulNBits
  constant-scale cache (#867 → 47.25). Capture collapses ~1600 launches/token into one
  replay; first-16 greedy ids match reference. **Three independent A/B experiments
  (#870/#872/#873) then proved this is the architectural ceiling:** native CUDA int4
  decode of this model is **weight-bandwidth/compute-floor bound at ~47.25 tok/s on H200**,
  NOT launch-dispatch bound — so `GroupQueryAttention` (~41% of eager decode) is *not* a
  lever, and node/launch fusion cannot help.
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

### 2026-08-13 — Over-budget models on Windows: ~100× from *not* streaming, and the memory line's accounting corrected

This line is about the **other** regime: models whose weights do **not** fit device
memory. It is separate from, and does not interact with, the dispatch-bound work below —
see "two regimes" at the end.

- **#874 — stop auto-enabling managed weight streaming on Windows/WDDM.** On `qwen14b-zp`
  (8.33 GB weights vs a 7.73 GB budget), byte-identical output, solo with `nvidia-smi`
  verified empty before every run: **8.09 tok/s with `htod_bytes_per_token = 0`** (true
  zero-copy — kernels read weights in place from host RAM over PCIe) against **0.11 tok/s**
  for managed streaming; `main` immediately prior measured 0.05–0.08, so **~100× end to
  end**. The cause is structural: each weight is read *exactly once per decode step*
  (922 initializers, ~867 lookups/step), so both paths move the same bytes over the same
  link, but ours added a CPU memcpy into pinned staging, a VRAM allocation, a `cuMemMap`,
  an eviction and a synchronize — **to buy VRAM residency discarded before it is ever
  re-read**. Explicit requests (`ONNX_GENAI_WEIGHT_OFFLOAD=1`, `--vram-limit`, a budget
  override) are still honoured; `ONNX_GENAI_MANAGED_WEIGHT_STREAMING=1` forces the managed
  path back, parsed so an *unrecognized* value keeps the fast default. **Linux is
  unchanged and must stay so** — with no shared-memory fallback an over-budget model simply
  fails there, so managed streaming competes with "does not run", not with something
  faster (#783's lesson about not inheriting a platform-specific conclusion).
- **#866 — elastic weight budget against KV occupancy.** The weight budget was
  `resolved_device_budget − kv_bytes_per_token × max_context`, computed once and never
  renegotiated, so a 16-token run idled **1.611 GB** holding KV it never committed.
  Lending it back with a guaranteed reclaim path (tested through the *production* reclaim
  code, so max-context reachability is preserved) cut `htod_bytes_per_token` **3.94 →
  2.35 GB (1.68×)** and byte-weighted hit rate **+20.3 points**, byte-identical tokens.
  Keeps a tunable 512 MiB headroom rather than lending to the last byte, and can never
  drop below what the static reservation granted.
- **#853/#856 — `total_weight_bytes` was 2.00× too large.** It measured *file size*, and
  `qwen14b-zp`'s external-data blob is **50.0% orphaned prefix** (920 initializers
  reference 8.33 GB of a 16.65 GB file, contiguously, first blob at offset 8,322,547,712 —
  a re-export that never truncated). The model is over budget by **0.599 GB, not ~8.9 GB**.
  Found by arithmetic that could not close: measured traffic sat *below* its own
  theoretical floor, which is impossible. The loader now sums what initializers reference
  and warns when file size exceeds it by >10%.
- **#863 — `no-spill` is an accounting guarantee, not physical residency, on WDDM.**
  Proven single-process: a `cuMemCreate`+`cuMemMap`+`cuMemSetAccess` allocator mirroring
  the engine arena committed **and touched every byte** of 9,984 MiB on an 8,188 MiB card;
  device-resident capped at ~7,942 MiB while the host working set reached 9.49 GB.
  Refined by a second measurement: **solo and under `managed_limit`, no-spill holds
  physically** (`nvidia-smi` tracks us 1:1); spill is specifically the **system-wide
  over-commit** case, invisible to our ledger because our own committed bytes do not change.
- **#869 — byte-weighted residency hit rate.** The count-based rate weights a 10 KiB norm
  like an 11 MiB projection: raising the budget once moved `hit_rate` 57.09% → 81.31%
  while the gap to the streaming floor **widened** 1.78× → 2.30×. `htod_bytes` is what
  decode pays for, so policy work is now judged on `byte_hit_rate`.
- **#877 — hybrid feasibility measured.** `cuMemHostRegister(READ_ONLY|DEVICEMAP)` on a
  read-only weight mapping succeeds at 3.1 ms/GiB; a device read of mapped host memory runs
  at **11.41 GB/s** vs **11.28** for an explicit HtoD copy and **110.46** already-resident.
  **This reversed the hybrid's assumed priority order:** zero-copy is *not* a bandwidth win
  (1.01×) — it removes the copy's second pass and its machinery — while **the resident hot
  set is worth ~9.7×**. So maximise residency first, de-copy the remainder second.

**Two regimes, and wins do not transfer between them.** When weights fit, decode is
**node-dispatch bound** (#870: 2,568 graph nodes/token at ~8.17 µs/node; cheapening any
single kernel's inner loop is flat). When they do not fit, decode is **PCIe-streaming
bound** and kernel work is irrelevant. Node-count reduction is invisible in the streaming
regime; everything above is invisible on a model that fits. Report bytes-moved-per-token
alongside throughput (#844's amortization line) — that counter distinguishes the regimes
where wall-clock cannot, and on this box wall-clock is not evidence (identical
configurations have ranged 3.9–28 tok/s, much of it WDDM paging our own memory).

### 2026-08-13 — Muse-Glimmer-30B native CUDA decode beats ORT (47 tok/s)

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
- **MatMulNBits scale-cache lever (#867): 40.21 → 47.25 tok/s — native now clearly beats
  ORT (~47 vs ~40, +18%).** With launches captured, decode is kernel-bound and MatMulNBits
  was the dominant op (~44% of eager decode). The bf16 activation path re-cast **every**
  input bf16→f16 each decode step, including the immutable int4 block **scales** — ~3.3
  GB/token of pure-copy traffic (≈25% of the int4 weight traffic) plus 417 redundant cast
  launches reproducing an identical f16 buffer. A persistent per-kernel `Bf16ConstCache`
  stages the constant scale slots **once** (general path input 2; gate/up SwiGLU inputs 2
  and 4) and reuses them across steps; the per-step activation and any per-token residual
  bound into the bias slot stay on the ephemeral arena (caching never keys on pointer
  identity for those dynamic slots). Byte-exact — bf16→f16 yields identical f16 bits
  whether cast once or per step, so the full 128-token greedy sequence is bit-identical;
  MatMulNBits eager share drops ~44% → **~31%**, capture stays **1 segment / 0 seams**. The
  cache fills on the pre-capture warmup call so replays hit only lookups (no alloc, no
  cast). `GroupQueryAttention` is now the largest eager share (~41%) — tested next as the
  candidate lever.

### 2026-08-13 — Native CUDA decode ceiling: bandwidth-bound at ~47 tok/s (#870/#872/#873)

- **Three independent A/B experiments conclusively prove native CUDA int4 decode of
  Muse-Glimmer-30B is weight-bandwidth/compute-floor bound at ~47.25 tok/s on H200, NOT
  launch-dispatch bound** — so the `GroupQueryAttention` (~41% eager) "next open lever" is
  disproven and node/launch fusion is a dead end on this model:
  - **#870 (GQA cheapening): flat.** Cheapening the GQA kernel's inner loop under capture
    did not move tok/s — decode is not GQA-compute bound.
  - **#872 (−208 cheap Add nodes): −2.8% regression.** A byte-exact constant-Add fold that
    removed 208 cheap elementwise nodes/token *hurt* throughput (47.17 → 45.85 captured),
    shipped as a doc-only negative finding.
  - **#873 (−104 expensive QKV GEMV launches): flat.** Fusing the 3 per-layer attention
    projections into one wider `MatMulNBits` (417 → 313 MatMulNBits/token, +52 Split,
    capture stays 1 segment / 0 seams) is byte-exact (all 64 greedy ids identical) but
    throughput is flat (47.33 → 47.26 median over 3 interleaved trials).
- **Why:** at M=1 decode reads each disjoint int4 weight blob exactly once from DRAM (the
  roofline is bytes-moved, not launches), so neither cheap-node nor expensive-launch
  fusion can cut bytes moved — a wider fused GEMV moves the same total bytes as the
  separate ones. 47.25 tok/s (already +18% vs ORT's ~40) is the architectural ceiling; we
  bank the win.
- **Disposition:** the correct, byte-exact `CudaQkvProjectionFusion` pass (#873) is
  retained **opt-in / disabled-by-default** (`ONNX_GENAI_CUDA_ENABLE_QKV_FUSION=1`) for
  future dispatch-bound architectures (e.g. fp16 activations, higher launch-latency-to-
  bandwidth shapes); the default binary keeps the three separate GEMVs.
- **Real levers to beat 47 are not node fusion:** reduce weight **bytes/token** (lower-bit
  quant, sparsity) or switch kernel family (a decode megakernel). Future work.

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

- [ ] **The #864 hybrid — resident hot set + zero-copy cold reads.** The only route to
  *beating* the OS on over-budget models rather than merely no longer losing to it (#874
  buys the ~100× by not choosing the slow path; it does not make us faster than WDDM). One
  risk is unresolved and is being measured: the #877 figure comes from a **sequential**
  `cuMemcpyDtoD` from mapped host memory, which is the right proxy for a kernel reading in
  place but is **not** a strided GEMV — PCIe punishes small scattered transactions where
  VRAM does not, so a real kernel could be far worse and that would kill the zero-copy half.
  Also untested: whether a host-mapped device pointer can be baked into a **captured** graph
  (`captures > 0`, `fallbacks == 0` are hard gates, #796).
- [ ] **#837 item 3 — residency policy gap.** Post-#866 the policy sits **1.97× above its
  own streaming floor** (2.349 vs 1.191 GB/step), i.e. **1.158 GB/step recoverable**, worth
  ~91 ms/step at measured bandwidths. `byte_hit_rate` 71.8% against an achievable 85.7%
  (`B/W`). `bypassed_page_ins` (704/run, ~12% of page-in events) has never been attributed
  and should be ruled in or out first. Governs Linux, forced-managed Windows, and — the
  reason it still matters after #874 — the hybrid, whose entire thesis is that we choose
  residency better than the driver does blind.
- [ ] **#750 native multi-request batching.** Structurally batch-1 today; `--max-batch` is
  honestly reported as ineffective (#758). Stage 1 (#844) confirmed the premise **and its
  condition**: batching amortizes the weight stream only if implemented as **one fused
  forward with `M = N`** — N independent forwards would miss every weight N times per step
  on an over-budget model and amortize nothing. Staged 2a (fused forward + batch-N binding),
  2b (batched KV), 2c (wire `continuous_batch_manager`).
- [ ] **#851 — intermittent `ILLEGAL_ADDRESS` in the mobius gate.** Parked, not solved.
  Three hypotheses eliminated (not gate flakiness — 8/8 strict solo; not kept-graph replay —
  25/25 isolated; not a relocated weight mapping — the fault targets a freshly allocated
  buffer, so it is a deferred fault whose reported node is the detection site). Never a data
  mismatch across ~49 runs, only whole-process crashes. Blocked on `compute-sanitizer`
  (no CUDA toolkit on this box) and on a card large enough to reproduce genuine
  multi-process oversubscription without contention confounds.
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

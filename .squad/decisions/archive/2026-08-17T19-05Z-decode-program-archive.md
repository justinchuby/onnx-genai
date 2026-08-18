# Archived decisions — 2026-08-14/15 decode-vs-ORT program

Archived by Scribe at 2026-08-17T19:05Z to keep live decisions.md below the spawn-budget gate.

## 2026-08-14/15 — glm-4-9b-int4 decode-vs-ORT program

### Same-weights ground truth vs stock ORT (Sebastian)

On identical int4 weights and fair ORT CUDA fastcfg (`past_present_share_buffer=true`, CUDA graph enabled), native default M=1 decode initially lost to stock ORT:

| model | native | ORT | gap |
|---|---:|---:|---:|
| glm-4-9b-int4 (block-128) | 97.5 tok/s | 250.3 tok/s | ORT 2.57× |
| qwen2.5-14b-int4 (block-32) | 127.0 tok/s | 184.5 tok/s | ORT 1.45× |

Fairness: glm ORT artifact strips only the redundant GQA `rotary_embedding_dim` while hardlinking the same `model.onnx.data`; native and ORT greedy tokens match byte-identically for the certified glm prompt. qwen shares the same weight file; its early divergence is the known near-tie cascade, not a weight mismatch. Eager op profiling overweights launch-heavy GQA; CUDA-graph kernel tracing is the authority for decode mix.

### PR #978 — int4 M=1 decode GEMV split-K (Deckard; Chew/Gaff approved)

**MERGED `532ef6bc`.** Deckard added default-on portable LOP3 dequant plus grid-fill split-K (`ONNX_GENAI_GENERAL_SPLITK=0|1` A/B). Result: glm **97.5→112.4 tok/s** (+15.3%); ORT gap **2.57×→2.24×**; qwen block-32 untouched by the `block_size != 32` gate. Greedy tokens byte-identical; f64 oracle passes. Chew and Gaff approved.

### PR #980 / cp.async GEMV spike — measured NO-GO (Deckard)

**CLOSED, not merged.** SM80-gated cp.async variants behind `ONNX_GENAI_GEMV_CPASYNC=1` were correct but regressed every measured glm config on H200: direct **112.2 tok/s**, cp.async stages 3/5 **95.9/95.4**, batched **89.7**, single-warp **102.6→88.3**. M=1 GEMV has only ~8 FMAs per loaded word, so cp.async overhead has no compute to overlap. Standing verdict: **do not default-enable or retry cp.async for M=1 int4 GEMV**.

### PR #981 — multi-warp block SkipSimplifiedLayerNorm for M=1 decode (Deckard; Chew approved)

**MERGED `b24e961e`.** Deckard replaced the one-warp decode SkipRMSNorm path with a portable block-parallel half4 kernel for `num_groups <= 8`; prefill keeps the warp path. Env A/B: `ONNX_GENAI_CUDA_DISABLE_SKIP_RMSNORM_BLOCK=1`. Result: glm **112.3→137.8 tok/s** (+22.7%), qwen **125.6→148.85 tok/s** (+18% range), greedy tokens byte-identical. Chew approved with notes: reduction order changes by ULPs inside tolerance; the capture assertion was pre-existing.

### PR #986 — 128-bit wide-load int4 M=1 decode GEMV (Deckard; Chew/Gaff approved)

**MERGED `e8f76c53`.** Default-on portable `uint4` wide-load GEMV for the glm-class block-128 general-BS path; env A/B `ONNX_GENAI_GEMV_WIDELOAD=0`. Mechanism: each lane issues one 128-bit load for 32 nibbles plus a depth-2 synchronous software pipeline, reusing LOP3 dequant and preserving per-lane ascending-K accumulation within the wide lane.

Key results: glm **140.7→192.4 tok/s** (+36.7%) in Deckard's idle run; Gaff reproduced **137.4→185.7 tok/s** (+35.1%) under shared-machine variance. Kernel gate_up improved **65.3µs→43.2µs**, DRAM **0.92→1.40 TB/s**. Cumulative glm base decode is now **97.5→112.4→137.8→192.4 tok/s**; ORT gap narrowed **2.57×→~1.30×** vs ORT ~250.3 tok/s, but native base still loses to ORT base.

Chew verdict: 🟢 APPROVE — f64 oracle **7/7**, glm and qwen greedy streams byte-identical for default wide vs narrow; qwen block-32 remains on the existing narrow/tuned path. Gaff verdict: 🟢 APPROVE — +35% perf reproduced, qwen flat/no-regression, CUDA graph capture clean, Rule 11 portable, fmt/clippy clean. qwen block-32 fused wide variants were reverted/not shipped because they were flat/slightly negative and compute/SM-bound, not DRAM-bound.

### Corrected ORT-gap mechanism and GEMV-v2 program

The earlier base-decode floor verdict is **OVERTURNED**. ORT's dominant glm `MatMulFloatInt4Kernel<__half,128,0>` streams the same gate_up geometry at **2.42 TB/s** vs native now **1.40 TB/s** after #986 (previously 0.92 TB/s). The gap is memory-level parallelism / weight streaming, not algorithm, occupancy, or irreducible dequant math.

**In flight:** Deckard owns GEMV-v2 on `squad/int4-gemv-wideload-v2`, targeting deeper MLP and ORT-like streaming (**1.40→2.42 TB/s**) to beat ORT base decode (~250 tok/s; target around ~280 tok/s if full streaming is captured). This is the base-vs-base path for any ORT-win claim.

### PR #984 — captured fused verify speculative decode (Sebastian; closed/superseded)

**CLOSED / SUPERSEDED.** Chew 🔴 and Gaff 🔴 rejected PR #984 at frozen head `82cde423`: opt-in captured verify was default-off and GLM token identity could pass on some prompts, but qwen repetition hit `GroupQueryAttention` prepared-workspace mismatch; Gaff also reproduced invalidated graph replay / illegal-address warmup failures and identified a shared graph-slot state hazard. The fix lineage moved to #988; #984 is not the active artifact.

### PR #988 — graph-slot captured spec-decode fix (Deckard; rejected pending Batty contract fix)

**DRAFT / NOT MERGEABLE.** PR #988 (`squad/spec-decode-graphslot`, frozen head `cfa96c5c`) fixed the capture crashes. Gaff 🟢 approved quality/perf/capture: GLM and qwen captured-spec runs completed, transition invalidation looked sound, CUDA regressions passed, and no stale replay/illegal address reproduced.

Chew 🔴 rejected correctness: the binding contract **speculative output == plain M=1 greedy** still fails on qwen W=9 (`spec_tokens=8`) at token index 2 (**plain 9370 vs spec 2810**). Forcing `ONNX_GENAI_SPEC_ROW0_TIE_EPS=100000` did not repair the divergence, so the row-0 M=1 fallback does not cover accepted DRAFT tokens. Per reviewer lockout, Deckard is locked out of revising #988; Batty owns the contract fix on `squad/spec-decode-w9-contract`, with Chew re-review expected.

### Standing directive for this arc

Binding metric: compare **BASE non-speculative native decode vs ORT base** for ORT-win claims. Speculative decode is additive-on-top and may not be the basis of an ORT-win claim. Current honest standing after #986: native base **192.4 tok/s** vs ORT base **~250 tok/s**, still **~1.30× behind**. Prioritize: (1) Deckard GEMV-v2 base MLP/streaming; (2) Batty W=9 speculative contract fix as an additive layer; (3) only then revisit smaller base kernels such as GQA or lm_head. Keep cp.async M=1 GEMV as a recorded NO-GO.


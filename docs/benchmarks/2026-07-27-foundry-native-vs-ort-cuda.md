# Foundry Local — Native CUDA EP vs ORT decode throughput (H200)

**Date:** 2026-07-27
**Engineer:** Cohaagen (perf)
**Mission:** Measure the native CUDA EP vs ORT decode-throughput gap on real Foundry
Local int4 deployment targets, and localize bottlenecks where native is slower.

## Hardware / software

| Item | Value |
| --- | --- |
| GPU | NVIDIA H200 (device 0, pinned `CUDA_VISIBLE_DEVICES=0 taskset -c 0`) |
| ONNX Runtime | 1.27.0 (API 27), CUDA EP — `.ort-cuda-1.27/root/lib/libonnxruntime.so.1.27.0` |
| Tool | `onnx-genai-bench` `profile_native` (`--features bench-native,bench-ort,cuda`) |
| Mode | `--steady` steady-state decode, `--tokens 128 --warmups 2 --runs 3` (median of 3 internal runs) |
| Prompt | "Explain the theory of relativity in detail." |
| Sampling | OFF (greedy, byte-identical fast path) |

> Note: the ORT backend requires `ONNX_GENAI_ORT_LIB` to point at the CUDA-enabled
> ORT shared library. `.cudaenv.sh` alone leaves it unset, so `profile_native`
> falls back to the CPU-only prebuilt ORT (`CUDAExecutionProvider` not exposed).
> All ORT runs below were taken with
> `ONNX_GENAI_ORT_LIB=$ORT_ROOT/lib/libonnxruntime.so.1.27.0`.

## Results

Each model was run at least twice per backend; medians below are steady-state
decode tok/s. Spread across repeats was <~2% after discarding first-touch
outliers (see 0.5B note).

| Model (Foundry int4, CUDA) | Native tok/s | ORT tok/s | Ratio (native/ort) | Verdict |
| --- | --- | --- | --- | --- |
| qwen2.5-0.5b-instruct | **995** | 580 | 1.72× | **native faster** |
| qwen2.5-1.5b-instruct | **720** | 438 | 1.64× | **native faster** |
| qwen2.5-7b-instruct   | **308** | 272 | 1.13× | **native faster** |
| Phi-4-mini-instruct   | **315** | 231 | 1.36× | **native faster** |

Per-repeat medians (steady tok/s):

| Model | Native reps | ORT reps |
| --- | --- | --- |
| qwen2.5-0.5b | 1001.9, 995.9, 993.6, 986.5 (rep discarded: 918.6 first-touch blip) | 581.7, 577.6 |
| qwen2.5-1.5b | 720.0, 721.1 | 436.2, 439.6 |
| qwen2.5-7b   | 307.5, 308.1 | 270.3, 274.2 |
| Phi-4-mini   | 314.2, 315.0 | 230.9, 230.7 |

**Headline:** Native CUDA EP is faster than ORT on every Foundry Local target
measured — from +13% on the 7B up to +72% on the 0.5B. The advantage shrinks as
the model grows (0.5B 1.72× → 7B 1.13×), consistent with the decode becoming
increasingly GEMM/memory-bandwidth-bound where both backends converge on the same
underlying int4 matmul kernels; native's per-step scheduling/launch overhead
savings dominate on the small models.

**No model showed native < ort**, so the bottleneck-localization / `--trace`
step (which applies only to native-slower cases) was not triggered.

## Correctness sanity (native vs ort, first ~30 tokens)

No garbage, repetition, or spacing artifacts observed on any model. ✅

### qwen2.5-0.5b — native
> " The theory of relativity is a set of principles that describe the physical
> laws of the universe at the speed of light. The theory of relativity was
> developed by Albert Einstein in the 20th century and has been the foundation…"

### qwen2.5-0.5b — ort
> " The theory of relativity is a set of principles that describe the physical
> laws of the universe at the speed of light. The theory of relativity was
> developed by Albert Einstein in the 20th century and has been the foundation…"

(0.5B native and ort are **token-identical**.)

### qwen2.5-7b — native
> " The theory of relativity is a fundamental concept in modern physics that
> describes the relationship between space and time. It was developed by Albert
> Einstein in the early 20th century and consists of two parts: the special
> theory of relativity and the general theory of relativity…"

### qwen2.5-7b — ort
> " The theory of relativity is a fundamental concept in modern physics that
> describes the relationship between space and time. It was developed by Albert
> Einstein in the early 20th century and consists of two parts: the special
> theory of relativity and the general theory of relativity…"

(7B native and ort are **token-identical**.)

Additional coherence checks: Phi-4-mini native/ort token-identical; qwen2.5-1.5b
native and ort diverge after the first sentence (expected under greedy decode
from tiny int4 logit differences) but both remain fully coherent English with no
repetition or spacing defects — **not** a regression flag.

## Bottleneck notes

None required — native is faster on all four targets. The narrowing margin on
the 7B (1.13×) is the natural place to look next if we want to widen native's
lead on large models, but optimization is explicitly out of scope for this
baseline (separate reviewed PR).

## Reproduction

```bash
source /home/justinchu/onnx-genai/.cudaenv.sh
export ONNX_GENAI_ORT_LIB="$ORT_ROOT/lib/libonnxruntime.so.1.27.0"
cd /home/justinchu/onnx-genai
cargo build --release -p onnx-genai-bench \
  --features bench-native,bench-ort,cuda --bin profile_native

for backend in native ort; do
  CUDA_VISIBLE_DEVICES=0 taskset -c 0 \
    target/release/profile_native \
      --model <MODEL_DIR> --ep cuda --backend $backend \
      --steady --tokens 128 --warmups 2 --runs 3 \
      --prompt "Explain the theory of relativity in detail."
done
```

---

# 7B bottleneck localization (follow-up, 2026-07-27)

**Why:** the 7B native lead over ORT was the thinnest of the four models
(1.13×). Justin wants us strong on larger models, so we localized where the 7B
native CUDA decode spends its on-device time to aim a future *reviewed* kernel
PR at the right place. **Scoping only — no kernels were modified.**

## Method

The `profile_native --steady --trace` path short-circuits before the tracer
block, and the tool's standalone `NativeDecodeSession::load` can't resolve this
Foundry model's token port (`input_ids` and `attention_mask` are both
`int64 [-1,-1]`, so port auto-detection is ambiguous — the ambiguity the
engine's `genai_config.json`-derived `model.io.token_input` resolves). So the
trace was captured through the real engine via the supported CLI timeline
(`--profile-trace`, which merges the engine's ORT-profiler spans with the native
runtime's per-operator `onnx-runtime-tracer` spans — the same `write_merged_trace`
the interactive CLI uses). No source was modified.

```bash
source /home/justinchu/onnx-genai/.cudaenv.sh
export ONNX_GENAI_ORT_LIB="$ORT_ROOT/lib/libonnxruntime.so.1.27.0"
cargo build --release -p onnx-genai-cli --features native-cuda --bin onnx-genai
CUDA_VISIBLE_DEVICES=1 taskset -c 1 \
  env ONNX_GENAI_EP=cuda ONNX_GENAI_TRACE_VERBOSITY=full \
  target/release/onnx-genai --profile --profile-trace qwen7b-native.json \
    generate --backend native --raw --greedy --max-new-tokens 64 \
    --prompt "Explain the theory of relativity in detail." \
    /home/justinchu/.foundry/cache/models/Microsoft/qwen2.5-7b-instruct-cuda-gpu-4/v4
```

Pinned to **GPU 1** (device 1, idle). The decode runs under **CUDA-graph
capture** (every op reports `capture_status: captured`): the graph is exercised
eagerly twice (a cold capture-warmup pass and one warm pass) and then replayed
silently, so per-op spans exist only for those two passes. The table below uses
the **warm** per-node occurrence (the cold first pass, e.g. a 68 ms cublas/JIT
first-touch on `layers.0/qkv_proj`, is discarded), summed across all 28 layers —
i.e. one steady decode step's on-device kernel spans.

## Top kernels by device time (one steady decode step, 28 layers)

Total attributed on-device kernel span ≈ **1.283 ms**. (Measured steady decode
is ~3.4 ms/token; the remainder is graph-replay launch overhead, sampling and
detokenize, none of which is kernel time.)

| # | op | kernel_variant | calls | total_ms | %-of-kernel | capture_status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | GroupQueryAttention | `attention_gqa_decode_fp16_splitk` | 28 | 0.425 | **33.1%** | captured |
| 2 | MatMulNBits | `gemv_f16_general` (o_proj) | 29 | 0.250 | **19.5%** | captured |
| 3 | MatMulNBits | `gemv_f16_down_projection` | 28 | 0.205 | 16.0% | captured |
| 4 | MatMulNBits | `gate_up_swiglu_rmsnorm_fused` | 28 | 0.200 | 15.6% | captured |
| 5 | MatMulNBits | `gemv_f16_scales_f16_rmsnorm` (qkv+RMSNorm) | 27 | 0.196 | 15.3% | captured |
| 6 | MatMulNBits | `gemv_f16_scales_f16_rmsnorm` (lm_head) | 1 | 0.007 | 0.5% | captured |

**Family rollup:** MatMulNBits int4 GEMV **66.9%**, GroupQueryAttention **33.1%**.

All GEMVs are **symmetric int4** (`zero_points=false`), `block_size=32`,
`scales=fp16`. At steady state the GQA decode prep is already fused
(`gqa_prep_fused_with_metadata`: `Sq==1, k_seq==1`, aliased device-KV) — the
unfused split/transpose/append/RoPE prep only appears on the first
post-prefill step (`k_seq=10`), so it is **not** a steady-state cost.

## #1 time sink — GroupQueryAttention decode (33.1%)

- **Kernel:** `crates/onnx-runtime-ep-cuda/src/kernels/group_query_attention.rs`,
  entry `attention_gqa_decode_fp16_splitk` (~L2233; module `gqa_decode_fp16`,
  `MAX_SPLITS = 16`).
- **Capture:** captured in the CUDA graph. Reason string:
  *"capture-safe fp16 split-K flash-decode: q_seq=1, even head_dim=128; active
  split count (up to 16) chosen on-device from the valid length and a host
  occupancy target that fills the multiprocessors."*
- **Assessment:** this path is **already the tuned split-K flash decode** — it
  fills the SMs with an on-device split count and its prep is already fused. This
  is genuine attention bandwidth/FLOP work, not a grid-starvation artifact, so it
  is the *lower*-leverage target. The only structural lever left is the
  **two-launch partial + softmax-merge reduction** (the partial and merge kernels
  record/replay as separate launches, ~L2212): folding the merge into the partial
  epilogue would remove one launch per layer (28/step). This is **not** a
  register-prefetch case.

## #2 time sink — MatMulNBits int4 GEMV family (66.9%; largest single variant o_proj `gemv_f16_general`, 19.5%)

- **Kernel:** `crates/onnx-runtime-ep-cuda/src/kernels/matmul_nbits.rs`. The
  o_proj lands on `gemv_f16_general` because it is **square** (K=N=3584) and so
  fails the tall-skinny gate: reason
  *"variant=general; class=not(tall_skinny K>N & block_size=32 & scales=fp16 &
  K%32==0)"*. The `general` path is the single-warp / few-columns-per-CTA kernel
  and is the prime **grid-starvation** suspect on H200 (132 SMs) — the same
  failure mode the code already fixes elsewhere with grid widening:
  `gemv_f16_down_projection` multiplies columns-per-CTA for tall-skinny down
  projections, and `matmul_nbits_gemv_f16_scales_f16_splitk`
  (`GEMV_F16_SCALES_F16_SPLITK_ENTRY`, ~L96) K-slices to fill idle SMs
  (~0.36 waves/SM noted in-source).
- **Hypothesis (to verify, then implement in a separate reviewed PR):** route the
  **grid-starved `gemv_f16_general` (o_proj)** through the existing **split-K /
  grid-widening** treatment (a `_general_splitk` sibling), gated on a measured
  `<1 wave/SM` occupancy check — *not* register prefetch.
- **Cross-reference — guardrails from prior perf memory (respect these):**
  - *Register-prefetch on the symmetric gate/up GEMV **regresses**.* The 15.6%
    `gate_up_swiglu_rmsnorm_fused` path is exactly that symmetric gate/up kernel —
    **do not** add register prefetch there.
  - *Grid-starved (<1 wave/SM) int GEMV wants **split-K, not prefetch**.* This is
    why the recommended direction for the o_proj `general` variant is split-K /
    grid widening, consistent with the down_projection and scales_f16 siblings.

## Next step (out of scope here)

Confirm the o_proj `gemv_f16_general` occupancy (`waves/SM`, achieved occupancy)
with Nsight Compute on device 1, then, in a separate reviewed PR, add a
grid-widened / split-K `general` variant gated on the grid-starvation check.
Attention (#1) is already split-K-tuned; the higher-leverage, guardrail-safe
target for the 7B is the o_proj GEMV.

---

# o_proj split-K attempt — NEGATIVE result (follow-up, 2026-07-27)

Acting on the #2 localization above, I implemented the guardrail-safe candidate:
route the grid-starved **square o_proj int4 GEMV** (`gemv_f16_general`,
K=N=3584, symmetric int4, block_size=32, fp16 scales) through the **existing**
`matmul_nbits_gemv_f16_scales_f16_splitk` machinery (PR #203), by widening the
dispatch gate from #203's `n < SM*16` (~2 CTA/SM) to a device-driven
**`< 1 wave/SM`** occupancy check (mirroring `use_accuracy4_stage64`: 64
warps/SM on sm_80/sm_90, 48 on consumer). No new kernel was written.

**Result: reverted.** The change is correct but does not beat baseline — it is a
small, *repeatable regression* on 7B and parity elsewhere. Reported here so the
next engineer does not re-try the same lever (cf. the register-prefetch negative
memories).

## What the gate did

- `use_f16_symmetric_splitk(k, n, mp_count, compute_capability, max_threads)`:
  split-K when the single-warp general grid `ceil(N/8)` CTAs is below one
  resident wave (`mp_count * resident_warps/8`).
- On H200 (132 SM, sm_90, one wave = 1056 CTAs) this newly routes 7B o_proj
  (N=3584 → 448 CTAs) and qkv (N=4608 → 576 CTAs) to split-K; lm_head-width
  stays single-warp; 0.5B/1.5B (N≤1152) already split under **both** gates, so
  they are unchanged by construction.
- Correctness held: GPU parity test at the exact K=N=3584 shape matched the f64
  reference within int4 tolerance, and 7B greedy token IDs were **byte-identical**
  before/after (no repetition/garbage).

## A/B (device 1, H200, `--steady --tokens 128 --warmups 2 --runs 3`, alternating)

BEFORE = `origin/main`; AFTER = split-K gate. Steady-decode tok/s (median of 3
per trial):

| model | BEFORE median | AFTER median | Δ | verdict |
|---|---|---|---|---|
| qwen2.5-7b  | **309.05** | 307.23 | **−0.59%** | repeatable regression |
| qwen2.5-1.5b | 725.05 | 723.65 | −0.19% | parity (noise) |
| qwen2.5-0.5b | ~993 | ~996 | +0.3% | parity (noise) |

7B trials (BEFORE / AFTER tok/s): 309.21/307.55, 308.61/307.14, 309.05/307.10,
309.21/307.75, 308.96/307.23 — intra-group spread < 0.3%, AFTER slower in 5/5.

## Why it lost

The **existing** split-K is `K_SPLIT = 2` — it only *doubles* the grid, lifting
o_proj from ~0.42 → ~0.85 wave (still sub-wave), while adding a shared-memory
cross-partition reduction to every column. At this occupancy the reduction
overhead outweighs the partial grid-fill gain, so decode is marginally slower.
This is consistent with PR #203 deliberately capping the symmetric gate at
~2 CTA/SM: moderately-starved (not severely-starved) shapes do not benefit from
a 2-way split. 0.5B/1.5B are unaffected because their narrower projections were
already splitting.

**Guardrail confirmed:** grid-widening the o_proj general GEMV with the current
2-way split-K machinery does **not** widen the 7B native-vs-ORT lead. A real win
would need a *larger* split factor (K_SPLIT>2) or a bespoke grid-widened
`general` variant that fills ≥1 wave without the 2-way reduction tax — i.e. a
new kernel, which is out of scope for the "reuse existing machinery" guardrail
and would need its own reviewed A/B. Recommend **not** pursuing the 2-way lever
for o_proj.

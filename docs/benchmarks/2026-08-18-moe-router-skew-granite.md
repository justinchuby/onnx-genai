# MoE router skew and always-on experts: how it was measured (granite-3.0-1b-a400m)

**Date:** 2026-08-18
**Author:** Copilot (streaming slice → MoE expert paging)
**Owner directive:** "继续推进 vmm、offload、streaming、multi-request batching 对大模型的支持,
提高速度，实现简洁高效" — the "streaming" slice, clarified to mean **MoE expert-weight
streaming**, not HTTP/SSE.

**Question this answers (`docs/memory/MEMORY_MANAGEMENT_MODEL_DESIGN.md` open item):**
dense is settled at `reads_per_step = 1.000` across all 867 weight keys — every weight is
read exactly once per step, there is no hot subset, so a residency policy has nothing to
prefer. The doc records MoE as the **first** case where a residency policy could have
something to be right or wrong about, *if* expert selection is genuinely skewed — and says
this **should be measured before any MoE residency policy is designed**. This is that
measurement. It establishes whether real, trained-router expert selection is skewed enough
that a residency policy could exploit it, and in particular whether some experts are
**always-on** (selected in 100% of decode steps for their layer).

## Result in one line

> [!summary] Measured
> On `granite-3.0-1b-a400m-instruct` (32 experts, top-8, 24 layers, **real trained IBM
> router**), decode expert selection is **skewed, not uniform**: the top-8 of 32 experts
> carry **45.4%** of per-layer read volume against a **25%** uniform baseline (Gini mean
> **0.334**, max 0.555), and every layer has a hottest expert selected in **46–100%** of
> decode steps (median **64%**) — layers 1 and 2 have experts selected in **100%** of steps
> (**always-on**). The skew is present in prefill on diverse tokens too (top-8 share
> **0.49–0.55**), so it is not an artefact of greedy decoding repeating itself.

## Hardware / method (house rule §32.2)

- **Box:** Intel i7-13800H (14C/20T), RTX 4060 Laptop 8 GB (driver 591.55, CUDA 13.1),
  Windows/WDDM.
- **Execution:** **CPU EP** (onnxruntime 1.27.0), **batch 1**, greedy (argmax) decoding.
  The CUDA box is irrelevant to *this* number and no GPU is used — see "Why CPU is valid".
- **Model:** `granite-3.0-1b-a400m-instruct`, built **f16 dense** through Mobius
  (`C:\Users\justinchu\dev\models\granite-1b-a400m-f16-mobius`). 32 experts, top-8 routing,
  24 layers, **no shared expert** (`inference_metadata.yaml`). "Dense" here means the ONNX
  graph is decomposed into a loop over all 32 experts with per-layer `TopK` selection, not
  a fused `QMoE` node — which is exactly what makes the per-layer selection observable.
- **Reference baseline:** **uniform routing**, `reads_per_step = top_k / num_experts =
  8 / 32 = 0.250`. This is the null hypothesis: if selection were uniform, every expert
  would average 0.250 reads/step and no residency policy could prefer one over another.
- **Instrument:** each per-layer `TopK` selected-experts output (the `..._1` indices tensor)
  is added as a graph output, so every decode step records the exact set of 8 experts each
  of the 24 layers selected. `reads_per_step` for an expert = (steps it was selected) /
  (total steps). Selection is `TopK(MatMul(hidden, gate))` — no dtype- or EP-dependent
  kernel is involved, so the picks are the model's, not the runtime's.
- **Sampling design:** **3 prompts** (English prose / Python code / math) **× 64 greedy
  decode tokens = 192 decode steps**, plus each prompt's prefill. Three content domains
  guard against a single prompt's topic manufacturing a private hot set; prefill is analysed
  separately on the diverse prompt tokens to rule out the objection that greedy decoding
  merely repeats itself into a hot set.
- **Repetitions / variance:** the measurement is deterministic (greedy decode, fixed
  prompts, CPU) — re-running `scripts/moe_router_skew.py` reproduces
  `scripts/moe_router_skew_counts.json` bit-for-bit. Reproduced 2026-08-18.

## Why CPU is valid (and the CUDA box does not enter this number)

Expert selection is `indices = TopK(MatMul(hidden, gate_weight), k=8)`. `MatMul` + `TopK`
are numerically stable integer-index operations; the **argmax-style top-k set does not
change between f16/f32 or between CPU and CUDA** for this router. So the *which experts*
question — the only thing this measurement asks — is dtype- and EP-independent, and CPU
picks equal CUDA picks. That is what licenses measuring skew on the CPU EP and applying the
conclusion to a CUDA decode. (Timing, bandwidth and paging cost are of course EP-dependent;
those are measured separately in the churn benchmark below and are **not** claimed here.)

## The load-bearing constraint: the router must be trained

> [!warning] The one trap that produces a confident false negative
> Skew is a property of **trained** router weights. A randomly-initialised router selects
> experts uniformly by construction, so it would measure `reads_per_step ≈ 0.250` flat —
> a **false negative** that says "MoE has no reuse either, stop the work". This is why the
> model is a real IBM Granite checkpoint exported through Mobius, **not** a synthesised
> MoE with random weights. Any repetition of this measurement must use a trained router,
> or its "no skew" result is meaningless.

## Numbers (measured, `scripts/moe_router_skew_counts.json`, 192 decode steps)

| Quantity | Uniform baseline | Measured (granite decode) |
|---|---|---|
| `reads_per_step` per (layer, expert) | 0.250 (flat) | min 0.000, **median 0.229**, mean 0.250, **max 1.000** |
| Top-8/32 experts' share of layer read volume | 0.250 | **mean 0.454** (min 0.389, max 0.632) |
| Gini of per-layer expert read counts | 0.000 | **mean 0.334, max 0.555** |
| Hottest expert per layer (fraction of decode steps) | 0.250 | **min 0.46, median 0.64, max 1.00** |
| Global share of read volume in top 25% of (layer,expert) cells | 0.250 | **0.456** |

Prefill, per prompt (diverse tokens, top-8 share of layer read volume; guards against
greedy self-repetition):

| Prompt | Prefill top-8 share | Decode top-8 share |
|---|---|---|
| English prose | 0.485 | 0.458 |
| Python code | 0.535 | 0.546 |
| Math | 0.487 | 0.483 |

All comfortably above the 0.250 uniform baseline in both prefill and decode.

## Why the distribution matters and the mean does not

The mean `reads_per_step` is **0.250 by construction** — each layer selects exactly 8 of
32 experts every step, so the average over all 32 is always `8/32`. Reporting the mean would
say nothing. The residency-policy question is entirely about the **shape**: a flat
distribution (all experts near 0.250) means MoE is no better than dense and the line of work
should stop; a **heavy tail** (a few experts near 1.000, many near 0.000) is precisely the
reuse a residency policy can exploit. The measured median 0.229 with a max of 1.000, a top-8
share of 0.454 (1.8× uniform), and per-layer hottest experts up to 100% is a heavy tail by
any reasonable standard.

## What this does and does not license

- **Does show:** a residency policy has something real to exploit — the `MEMORY_MANAGEMENT_
  MODEL_DESIGN.md` open MoE question is answered in the affirmative. Layers 1–2's always-on
  experts are free, zero-prediction pins.
- **Does not show a policy will win.** Two caveats, measured separately:
  1. The paging layer **currently cannot see this skew**: `bind_block_quantized_moe`
     (`crates/onnx-runtime-ep-cuda/src/weight_paging.rs`) pages the whole expert bank as a
     **single key**, so a real QMoE run reports whole-bank `reads_per_step ≈ 1.0`
     (dense-like). Per-expert paging is required just to make the skew *visible* to the
     runtime — see `2026-08-18-moe-per-expert-paging-churn.md`.
  2. Per-expert VMM paging is a **large-expert** technique: the 2 MiB device granule makes
     sub-granule experts (granite int4 ≈ 0.75 MiB) impossible to page individually — again
     see the churn benchmark.
- **Scope:** this is a property of **this model and these prompts**, not a universal law of
  MoE. It is strong evidence that trained routers skew, not a proof that every MoE skews by
  this amount.

## Reproduction

```powershell
# From repo root. Uses the CPU EP; no GPU required.
# Requires: onnxruntime>=1.27, onnx, numpy, tokenizers (system Python).
# MODEL_DIR is set inside the script to the granite f16 Mobius fixture:
#   C:\Users\justinchu\dev\models\granite-1b-a400m-f16-mobius
python scripts/moe_router_skew.py
# -> prints per-prompt prefill/decode + aggregate + persistence tables,
#    and writes scripts/moe_router_skew_counts.json (the committed result).
```

The script patches every layer's `TopK` indices as a graph output, greedily decodes the
three prompts, and tallies per-(layer, expert) selection counts. `moe_router_skew_counts.json`
is the committed raw tally (`decode_counts`, `total_decode_steps`, `num_experts`, `top_k`,
`num_layers`) so the tables above can be recomputed without re-running inference.

## Formal sources

- Open question answered: [`docs/memory/MEMORY_MANAGEMENT_MODEL_DESIGN.md`](../memory/MEMORY_MANAGEMENT_MODEL_DESIGN.md)
- Paging cost / granule floor / skew-invisibility: [`2026-08-18-moe-per-expert-paging-churn.md`](2026-08-18-moe-per-expert-paging-churn.md)
- Whole-bank keying today: [`crates/onnx-runtime-ep-cuda/src/weight_paging.rs`](../../crates/onnx-runtime-ep-cuda/src/weight_paging.rs) (`bind_block_quantized_moe`)
- Per-key `reads_per_step` trace for the native path: [`crates/onnx-genai-bench/src/bin/profile_native.rs`](../../crates/onnx-genai-bench/src/bin/profile_native.rs)
- Reproduction: [`scripts/moe_router_skew.py`](../../scripts/moe_router_skew.py), [`scripts/moe_router_skew_counts.json`](../../scripts/moe_router_skew_counts.json)
- Beginner-facing explanation: [`wiki/memory/MoE Router Skew and Always-On Experts.md`](../../wiki/memory/MoE%20Router%20Skew%20and%20Always-On%20Experts.md)

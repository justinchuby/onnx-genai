---
title: MoE Router Skew and Always-On Experts
aliases:
  - Always-On Experts
  - Expert Selection Skew
  - How MoE Router Skew Was Measured
tags:
  - memory
  - moe
  - measurement
  - offload
  - beginner
status: maintained
created: 2026-08-18
updated: 2026-08-18
---

# MoE Router Skew and Always-On Experts

> [!summary] Question answered
> In a Mixture-of-Experts model, does the router pick some experts far more often than
> others — enough that keeping the popular ones resident could pay off — and **how do you
> measure that honestly**? This note explains the method behind the measured result that
> `granite-3.0-1b-a400m` has **always-on experts** and a heavy selection tail. The numbers
> and their reproduction live in the benchmark record; this note explains what they mean and
> how not to fool yourself.

## Why this question exists

A dense decoder reads **every** weight exactly once per step. The memory design records this
as `reads_per_step = 1.000` across all 867 weight keys — there is no "hot" subset, so no
residency policy can prefer one weight over another: keeping weight A resident instead of B
saves nothing, because both are read every step. That is why dense weight streaming is
settled as a non-win.

A Mixture-of-Experts (MoE) layer is different. Each layer has many experts (small feed-forward
networks), but a small **router** picks only `k` of them per token. If the router's choices
were spread evenly, MoE would be no better than dense for residency: every expert would be
read about `k / E` of the time and none would be worth pinning. But if the router **skews** —
returning to the same few experts token after token — then those experts are genuinely re-read,
and keeping them resident while paging the rest could win. The memory design calls MoE "the
first case where a residency policy has something to be right or wrong about" and asks for this
to be **measured before any policy is designed**. See [[memory/Memory Management for Beginners]]
for the surrounding allocator/backing/governor vocabulary.

## What "always-on expert" means precisely

> [!important] Definition
> An **always-on expert** is one selected in **100% of decode steps** for its layer — not
> merely a frequent expert, but one the router never skips. It is the strongest possible form
> of reuse: such an expert can be pinned resident with **zero** prediction, because it is
> needed every single step.

In the measured model this was observed in **layers 1 and 2**. Every layer has *a* hottest
expert (selected 46–100% of the time), but only some layers have one selected 100% of the time.

> [!warning] This is a property of the model measured, not a law of MoE
> "Layers 1–2 are always-on" is true for `granite-3.0-1b-a400m` on the prompts tested. It is
> **not** a universal fact about MoE models. A different checkpoint may place its always-on
> experts elsewhere, or have none. The measurement establishes that skew *can* be strong enough
> to exploit, not that it always is.

## The model, and the one trap that would ruin the measurement

The measurement uses `granite-3.0-1b-a400m-instruct`: 32 experts, top-8 routing, 24 layers,
no shared expert, exported **f16 dense** through Mobius. "Dense" here means the ONNX graph is a
decomposed loop over all 32 experts with a per-layer `TopK`, which is precisely what makes each
layer's expert choice observable as a graph output.

> [!danger] The single most important methodological point
> **The router must be genuinely trained.** Skew is a property of *trained* router weights. A
> randomly-initialised router selects experts **uniformly by construction**, so it would measure
> a flat `reads_per_step ≈ k/E` and produce a **confident false negative** — "MoE has no reuse,
> stop the work" — that is an artefact of the random weights, not a fact about MoE. This is the
> trap the next person is most likely to fall into. The measurement therefore exports a real
> IBM Granite checkpoint; it never synthesises a toy MoE with random weights. If you cannot get
> a trained router, you cannot answer this question — a synthesised router's "no skew" tells you
> nothing.

## Why measuring on the CPU is valid

Expert selection is `indices = TopK(MatMul(hidden, gate_weight), k)`. The set of chosen experts
is an integer top-`k` over a matrix product; it does **not** change between f16 and f32, or
between the CPU and CUDA execution providers. So the *which experts* question is dtype- and
EP-independent, and the CPU's picks are identical to CUDA's. That is what licenses measuring the
skew on the [[execution/CPU Execution Provider]] and applying the conclusion to a CUDA decode.
Only *timing and bandwidth* are EP-dependent — and those are measured separately, not inferred
from this trace.

## The sampling design, and the objection it defends against

The trace decodes **3 prompts** — English prose, Python code, and math — for **64 greedy tokens
each** (192 decode steps total), plus each prompt's prefill.

- **Three content domains** guard against a single topic manufacturing a private hot set: if the
  skew only appeared for prose, it might be about that text, not the router.
- **Prefill is analysed separately.** Greedy decoding can fall into loops, so a hot set seen only
  during decode could be an artefact of the model repeating itself. Measuring the **prefill**
  tokens — which are the diverse, externally-supplied prompt, not model output — checks the skew
  on genuinely varied inputs. It is still present there (top-8 share ≈ 0.49–0.55), which is the
  answer to the obvious objection.

## Reading the statistics: shape, not mean

> [!important] The mean is fixed at `k/E`; only the distribution carries information
> Because each layer always selects exactly `k` of `E` experts, the mean `reads_per_step` over
> all experts is **`k/E = 8/32 = 0.250` by construction**. Reporting the mean would say nothing.
> The whole question is the **shape** of the distribution: flat means no reuse (stop), a heavy
> tail means exploitable reuse (continue).

The measured shape is a heavy tail: a median near the uniform `0.229`, a **max of 1.000**
(always-on experts), the top-8 of 32 experts carrying **45.4%** of read volume versus a 25%
uniform baseline, and a Gini of 0.334. Concretely: a residency policy that pinned the hot
experts would get real hits, whereas under uniform routing it could not.

## What this licenses — and what it does not

> [!note] This answers one question and opens the next
> - **It shows** a residency policy has something real to exploit — the memory design's open MoE
>   question, answered affirmatively. Always-on experts are free, zero-prediction pins.
> - **It does not show a policy will win.** Today the paging layer pages the whole expert bank as
>   **one key**, so the skew is *invisible* to the runtime (it reports whole-bank
>   `reads_per_step ≈ 1.0`, dense-like). And per-expert VMM paging is a **large-expert** technique
>   — the 2 MiB device granule makes granite's sub-granule int4 experts impossible to page
>   individually. Both caveats are measured in the per-expert paging churn record.

Plumbing (per-expert paging, to make the skew *visible*) therefore comes before policy (exploiting
it). See the linked churn benchmark for that ordering and its measured costs.

## Reproduction

The exact invocation, environment, and raw tally are in the benchmark record. In short, from the
repository root:

```powershell
python scripts/moe_router_skew.py
```

This patches every layer's `TopK` indices as a graph output, greedily decodes the three prompts on
the CPU EP, prints the per-prompt and aggregate tables, and writes `scripts/moe_router_skew_counts.json`
— the committed raw tally the tables are computed from. The run is deterministic, so it reproduces
the committed JSON exactly. For the mechanics of adding graph outputs and reading per-op signals,
see [[observability/Tracing and Profiling]].

## Formal sources

- Measured evidence (numbers, hardware, method, house rule §32.2):
  [Router skew benchmark](../../docs/benchmarks/2026-08-18-moe-router-skew-granite.md)
- Paging cost, granule floor, and why the skew is invisible today:
  [Per-expert MoE paging churn](../../docs/benchmarks/2026-08-18-moe-per-expert-paging-churn.md)
- The open question this answers:
  [Memory Management Model Design](../../docs/memory/MEMORY_MANAGEMENT_MODEL_DESIGN.md)
- Reproduction script and raw tally:
  [`scripts/moe_router_skew.py`](../../scripts/moe_router_skew.py),
  [`scripts/moe_router_skew_counts.json`](../../scripts/moe_router_skew_counts.json)
- Related notes: [[memory/Memory Management for Beginners]],
  [[execution/CPU Execution Provider]], [[observability/Tracing and Profiling]]

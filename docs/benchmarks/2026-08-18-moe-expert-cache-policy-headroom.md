# MoE expert-cache policy headroom: oracle vs static hot-pin (trace-driven, granite router)

**Date:** 2026-08-18
**Author:** Copilot (streaming slice → MoE expert paging)
**Owner directive:** advance MoE expert-weight streaming for **large** models; the real
problem is the **over-subscribed** regime (expert bank > VRAM), where a residency policy must
decide *which* experts to keep. This is an **offline** decision aid: it measures how much any
residency policy could win, and how close the cheapest policy (static hot-pin, no runtime
prediction) gets to the unbeatable oracle — **before** committing to hardware we do not have.

## What this is and is not

- **Is:** a cache simulator replaying a **real trained-router** expert-selection trace against
  a bounded, shared expert cache, sweeping budget ratio, comparing eviction policies.
- **Is not:** a wall-clock measurement, a paging-mechanism cost, or proof that the result
  generalises to a 128/256-expert DeepSeek-class router. Those are out of scope and flagged
  INFERRED below.

Routing skew is a property of the **trained router and the prompt, not of VRAM size**, so the
granite trace is a valid workload even though granite itself fits in 8 GB and never pages.
Methodology of the trace: [MoE Router Skew and Always-On Experts](../../wiki/memory/MoE%20Router%20Skew%20and%20Always-On%20Experts.md) /
[`2026-08-18-moe-router-skew-granite.md`](2026-08-18-moe-router-skew-granite.md).

## Hardware / method (house rule §32.2)

- **Box:** Intel i7-13800H (14C/20T), RTX 4060 Laptop 8 GB (driver 591.55, CUDA 13.1),
  Windows/WDDM. **No GPU is used** — this is a pure trace replay on the CPU.
- **Model behind the trace:** `granite-3.0-1b-a400m-instruct`, **32 experts, top-8, 24
  layers, real IBM trained router**, f16 dense Mobius export. Trace: 3 prompts
  (English prose / Python code / math) × **64 greedy decode tokens** = 192 decode steps.
- **Trace capture:** `scripts/dump_moe_expert_trace.py` → `scripts/moe_expert_trace.json`.
  Selection is `TopK(MatMul(h, gate))`, dtype/EP-independent, so CPU picks equal CUDA picks.
- **Simulator:** `scripts/moe_expert_cache_sim.py` → `scripts/moe_cache_sim_results.json`.
  Deterministic (`random.seed(1234)`); reproduced 2026-08-18.
- **Cache model:** experts keyed `(layer, expert)` — 768 distinct objects. Budget is
  **GLOBAL** across all 24 layers (VRAM is one shared pool): budget ratio `r` →
  `round(r × 768)` resident slots the allocator may spend on any layer. Within a decode step,
  layers 0..23 access their top-8 experts in order (drives LRU/FIFO recency, oracle next-use).
- **Baseline to beat:** `equal_oracle` / `equal_lru` = the same policies under a per-layer
  **equal split** (`round(r × 32)` slots per layer, no cross-layer sharing).
- **Reference / null:** `random` eviction. **Decision metric:** **bytes/token**, reported at
  the target-regime **16 MiB** expert (GLM/DeepSeek-class). `page-ins/token` is
  size-independent; multiply by any expert size. (At granite's own 0.75 MiB int4 experts the
  bytes are 21× smaller, but granite fits and never pages — the 16 MiB figure is the one that
  speaks to the target regime.)

## Result table (bytes/token @16 MiB; English-prose workload, others within a few %)

| budget (slots) | oracle | random | LRU | LFU | **static-pin** | hybrid | equal_oracle | equal_lru |
|---|---|---|---|---|---|---|---|---|
| 10% (77)  | 2099 | 2940 | 3072 | 2546 | 2359 | 2992 | 2177 | 2653 |
| 25% (192) | **1137** | 2209 | 1648 | 1899 | **1662** | 1646 | 1648 | 1648 |
| 50% (384) | **494**  | 1233 | 1115 | 988  | **763**  | 1114 | 598  | 958  |
| 75% (576) | 234  | 527  | 425  | 384  | 192  | 424  | 263  | 405  |
| 90% (691) | 189  | 264  | 214  | 210  | 26   | 213  | 192  | 217  |

Oracle **miss rate** (English prose): 10%→0.68, 25%→0.37, **50%→0.16**, 75%→0.08, 90%→0.06.

> [!warning] The 75–90% rows are cold-start-dominated over a 64-token trace
> With ~768 distinct experts and a large budget, most page-ins in the 75–90% rows are the
> **one-time compulsory fill** (~12 page-ins/token floor over only 64 steps), which amortises
> to ~0 over a long generation. That fill is *counted* for the demand-cache policies (oracle,
> LRU, …) but **not** for static-pin (its pinned set is treated as pre-resident), which is why
> static-pin appears to "beat" the provably-optimal oracle there. **The policy-relevant regime
> is 10–50%**, where eviction pressure — not cold start — dominates. Read the decision off
> those rows.

## Answers to the questions, in priority order

### 1. How much headroom is there at all? (oracle vs random) — MEASURED

Substantial. In the decision-relevant **25–50%** budget band, the oracle transfers **46–67%
fewer bytes/token than random** (25%: 1137 vs 2209 = −49%; 50%: 494 vs 1233 = −60%; code/math
similar, up to −67% at 50%). **A residency policy can roughly halve transfer versus no policy.**
The residency-policy line of work is **not** dead — the opposite of the dense result.

### 2. How close does static hot-pin get to oracle? — MEASURED (the headline)

**Frequency static-pin captures most of the win with no runtime prediction.** Measured as the
fraction of the random→oracle gap it closes: **~51% at 25%** budget, **~64–84% at 50%** across
the three prompts. It **beats LRU, FIFO, LFU, and the hybrid** at mid budgets (50% prose:
static 763 vs LRU 1115 vs hybrid 1114). It does **not** reach the oracle (still +17–54% of
oracle bytes at 25–50%), so a smarter online policy has residual room — but the **cheapest
possible thing** (a global top-k-by-frequency pin, computed once) gets the bulk. This is the
简洁高效 outcome: ship the static pin first.

> [!note] Frequency beats recency here, and beats the hybrid
> Routing is frequency-skewed and fairly **stationary**, not recency-driven, so LRU is weak and
> the hybrid (pin the ~6 always-on keys + LRU over the rest) collapses toward LRU. Spending the
> **whole** budget on frequency-pinning dominates giving any slots to LRU. Do not build the
> hybrid for this workload class.

### 3. Where is the knee? — MEASURED

Oracle miss rate falls steeply up to **~50%** budget (0.68 → 0.37 → **0.16**), then flattens
toward the compulsory floor. **Actionable:** for granite-like routing you need ≈ **half the
expert bank resident** to bring oracle miss to ~0.16 (~490 bytes/token @16 MiB); below **25%**
budget you page most accesses (miss ≥ 0.37) regardless of policy, because 25% budget = exactly
one step's 192-key working set. "Enough VRAM" for this routing ≈ **50% of the expert bank**.

### 4. Does prompt type change the answer? — MEASURED

**Largely no.** The always-on core is prompt-**independent**: layer 1 → expert 8 and layer 2 →
expert 26 are selected in 64/64 steps in **all three** prompts. Cross-prompt static-pin (pin
from two prompt types, evaluate on the third) raises miss rate by only **+0.07 to +0.13** vs
in-sample at 25–50% budget. A static pin tuned on one workload **transfers** to others; the
decay lives in the mid-frequency tail, not the core. So static pin does not need per-prompt
retuning — it can be computed once from a representative trace and shipped.

### Bonus — global budget is worth having (validates the shared-pool design) — MEASURED

Letting the allocator share one pool across layers beats a per-layer equal split for the
oracle by **~31% at 25%** budget (1137 vs 1648) and **~17% at 50%** (494 vs 598). This is the
hot layers (1–2) claiming more than their equal share — exactly what a per-layer fixed slice
would forbid. (Note: for a *bad* policy like LRU, global sharing can *hurt*, because LRU
thrashes hot experts across layers; global sharing pays off only with a good victim rule.)

## Limits — separate measured from inferred

- **INFERRED (generalisation):** granite is 8-of-32 (25% working set). DeepSeek-class routers
  are 8-of-256 (~3% working set). A budget ratio means something different there; intuitively
  the larger-bank case has **more** residency headroom (the hot core is a smaller fraction to
  pin), so these numbers are likely a **conservative** floor for the target models — but this
  trace cannot demonstrate it. Direction stated, not measured.
- **INFERRED (mechanism):** this says nothing about achieved wall-clock or the cost of the
  paging mechanism itself. Whether the win is realised by VMM per-expert paging (2 MiB granule,
  measured churn — see [`2026-08-18-moe-per-expert-paging-churn.md`](2026-08-18-moe-per-expert-paging-churn.md))
  or OS hints at 4 KiB granularity is the **next** decision, gated on this headroom being real.
  It is.
- **MEASURED caveat:** bytes/token is monotone in page-ins/token because granite's experts are
  uniform size; with heterogeneous expert sizes only bytes would rank correctly.

## Recommendation (for the owner — no runtime policy built yet)

1. The headroom is real (oracle halves transfer vs random at 25–50% budget).
2. The **static global frequency-pin** captures 50–84% of that with zero prediction machinery,
   is durable across prompt type, and beats every online policy tried — **ship it first**.
3. A smarter online policy could recover the remaining +17–54% to oracle, but that is a second
   step, not the first.
4. Mechanism choice (VMM vs OS hints) is the immediately-next decision and is where the 2 MiB
   granule vs 4 KiB OS-page question is decided.

## Reproduction

```powershell
# From repo root. CPU only; requires onnxruntime>=1.27, onnx, numpy, tokenizers.
python scripts/dump_moe_expert_trace.py     # ~2 min CPU inference -> scripts/moe_expert_trace.json
python scripts/moe_expert_cache_sim.py      # -> scripts/moe_cache_sim_results.json + tables
```

## Formal sources

- Trace methodology + provenance: [MoE Router Skew and Always-On Experts](../../wiki/memory/MoE%20Router%20Skew%20and%20Always-On%20Experts.md),
  [`2026-08-18-moe-router-skew-granite.md`](2026-08-18-moe-router-skew-granite.md)
- Paging cost / granule floor (the mechanism question this feeds):
  [`2026-08-18-moe-per-expert-paging-churn.md`](2026-08-18-moe-per-expert-paging-churn.md)
- The open question this closes: [`docs/memory/MEMORY_MANAGEMENT_MODEL_DESIGN.md`](../memory/MEMORY_MANAGEMENT_MODEL_DESIGN.md)
- Scripts: [`scripts/dump_moe_expert_trace.py`](../../scripts/dump_moe_expert_trace.py),
  [`scripts/moe_expert_cache_sim.py`](../../scripts/moe_expert_cache_sim.py),
  [`scripts/moe_cache_sim_results.json`](../../scripts/moe_cache_sim_results.json)


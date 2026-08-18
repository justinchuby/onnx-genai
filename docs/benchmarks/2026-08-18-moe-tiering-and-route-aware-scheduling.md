---
title: "MoE offload — tiering, concentration, and route-aware scheduling (trace-driven)"
date: 2026-08-18
hardware: "Intel i7-13800H (14C/20T), RTX 4060 Laptop 8 GB, WDDM, driver 591.55, CUDA 13.1"
model: "granite-3.0-1b-a400m-instruct (32 experts, top-8, 24 layers), onnxruntime 1.27.0 CPU EP, batch 1"
status: measured + inferred (clearly separated)
---

# MoE offload — tiering, concentration, and route-aware scheduling

This extends the single-tier policy-headroom study
([2026-08-18-moe-expert-cache-policy-headroom.md](2026-08-18-moe-expert-cache-policy-headroom.md))
to test three claims from an external MoE-offload analysis, using the same real
trained-router trace. It answers, from measurement rather than assumption:

1. **Is routing as Zipfian as the analysis assumes?** (concentration curve)
2. **Does a DRAM warm tier take the SSD out of the critical path?** (three-tier sim)
3. **How much does expert-overlap request scheduling save, and what does it cost?**
   (route-aware scheduling)

All numbers are from the granite trace (3 prompts × 64 greedy decode steps = 192
steps; 24 layers × top-8 of 32 experts), replayed on **CPU only** — no GPU is used;
routing skew is a property of the trained router and the prompt, not of VRAM.
Hardware is stamped for provenance per house rule §32.2:
**i7-13800H (14C/20T), RTX 4060 8 GB, WDDM.** Reproduce with:

```
python scripts/dump_moe_expert_trace.py        # (re)capture the trace, ~2 min CPU
python scripts/moe_concentration_curve.py      # arm 1
python scripts/moe_tier_sim.py                 # arm 2
python scripts/moe_route_aware_sched.py        # arm 3
```

Provenance of the trace and method: [MoE Router Skew and Always-On Experts](../../wiki/memory/MoE%20Router%20Skew%20and%20Always-On%20Experts.md)
and [2026-08-18-moe-router-skew-granite.md](2026-08-18-moe-router-skew-granite.md).

---

## Arm 1 — Concentration curve: our routing is materially milder than the assumed Zipf (MEASURED)

The external analysis illustrates its tiering argument with "top 32 of 256 experts
(**12.5 % of the bank**) carry **~80 %** of traffic". Our only real measurement
disagrees. Global concentration over all 768 `(layer, expert)` keys:

| top p% of keys | cumulative % of traffic |
|---|---|
| 10 % | 22.9 % |
| **12.5 %** | **27.1 %**  (assumed Zipf: 80 %) |
| 25 % | 45.6 % |
| 50 % | 74.4 % |
| 75 % | 92.7 % |
| 90 % | 98.6 % |

Global Gini = **0.343** (consistent with the independently-measured 0.334 in the
router-skew record — a capture cross-check). Per-layer, the top-8/32 (= 25 % of a
layer) carries **mean 45.4 %** of that layer's traffic (min 38.9 %, max 63.2 %),
reproducing the earlier 45.4 % figure exactly; the max is layers 1–2, which have
**always-on** experts (a single expert selected in 100 % of steps).

**Verdict (measured):** granite's top 12.5 % carries **27 %, not 80 %**. The skew
is real and exploitable, but **the tiering wins are smaller than the 80/20
illustration implies** on a 32-expert router. We should not plan against an 80/20
assumption when the only data we have says ~27/12.5 (global) or ~45/25 (per-layer).

**Inferred (not measured):** DeepSeek-class routers select top-8 of a *much* larger
bank (e.g. 8 of 256 = 3 % working set vs granite's 25 %). Intuition and the analysis
both expect larger banks to be *more* concentrated, so their curve may sit closer to
the assumed Zipf than granite's. Our trace cannot demonstrate this; direction only.

---

## Arm 2 — Three-tier VRAM/DRAM/SSD: a DRAM tier does remove the SSD from the hot path (MEASURED, with INFERRED bandwidths)

Exclusive two-level LRU over experts keyed `(layer, expert)`: VRAM (`bv` slots, free
hits), DRAM (`bd` warm slots, a hit costs a DRAM→VRAM PCIe copy), SSD (everything
else, a miss costs an NVMe read). We separate **compulsory** SSD traffic (first-ever
touch of a key — unavoidable, and an artefact of a 64-step trace that amortises over
a long generation) from **capacity** SSD traffic (the avoidable steady-state number).

Representative (english-prose; code/math within a few %), experts @ 16 MiB, budgets
as a fraction of the 768-key bank:

| VRAM | DRAM | SSD-capacity (exp/tok) | SSD-total (exp/tok) | DRAM PCIe (exp/tok) |
|---|---|---|---|---|
| 25 % | 0 %   | 111.9 | 123.5 | 0 |
| 25 % | 25 %  | 58.2  | 69.8  | 53.7 |
| 25 % | 50 %  | 15.1  | 26.7  | 96.8 |
| 25 % | **100 %** | **0.0** | 11.6 (compulsory only) | 111.9 |
| 5 %  | 100 % | **0.0** | 11.6 (compulsory only) | 180.4 |

**Verdict (measured):** once the **DRAM tier can hold the whole expert bank
(≥ 100 % budget), avoidable SSD traffic falls to zero** — only the compulsory
first-touch remains, and that amortises away over a real generation. The analysis's
central claim holds on our trace: **a DRAM warm tier keeps the SSD out of the
critical path, so DirectStorage/GDS is a cold-miss backstop, not the main event.**
For granite's bank (24 × 32 × 16 MiB ≈ 12 GiB) that DRAM budget is trivial on any
workstation.

**But the SSD is not the remaining bottleneck — PCIe is.** With SSD gone, every
non-VRAM-resident expert still crosses PCIe from DRAM: ~112–180 experts/token at
16 MiB ≈ **1.8–2.8 GB/token**. Using **assumed, not measured** bandwidths (PCIe 4.0
x16 ≈ 26 GB/s — this laptop 4060 may be x8, i.e. half; NVMe Gen4 ≈ 6 GB/s; DDR5 ≈ 80
GB/s), the estimated per-token transfer time drops from ~500 ms (SSD-bound, no DRAM)
to ~100–140 ms (PCIe-bound, DRAM holds the bank). **The dominant remaining cost is
DRAM→VRAM PCIe**, which is exactly what maximising VRAM residency (the single-tier
static hot-pin) reduces. The two studies compose: **tier to delete the SSD, then pin
to shrink the PCIe bill.**

*Caveat:* the bandwidth constants are order-of-magnitude assumptions, not measured on
this box; the ms/token figures are for relative comparison only, not a wall-clock
claim.

---

## Arm 3 — Route-aware scheduling: real, but batch size is the bigger lever and reordering has a fairness price (MEASURED)

Each decode step is a request needing its 192 `(layer, expert)` keys; 192 requests
arrive interleaved across the 3 prompts. A batch of W requests loads the **union** of
its experts once and serves all W, so bandwidth/token = (Σ batch unions)/requests.
Two schedulers see a lookahead window of the Q oldest pending requests: **fifo**
(first W) vs **route_aware** (greedily pick W with the smallest union). No persistent
cache, to isolate the scheduling effect. Experts @ 16 MiB.

**Batching alone (FIFO) is the biggest win and is free of fairness cost:**

| scheduler | bytes/token @16 MiB | vs no-batch |
|---|---|---|
| W=1 (no batching) | 3072 | — |
| FIFO W=2 | 2544 | −17 % |
| FIFO W=4 | 1885 | −39 % |
| FIFO W=8 | 1208 | −61 % |

Adjacent steps already share the always-on / hot core, so plain batching captures
most of the overlap. **Route-aware reordering adds a secondary win that grows with
lookahead Q — but so does the worst-case delay** (delay = positions a request slips
past its arrival order; FIFO delay is 0 by construction):

| W | Q | route-aware bytes/tok | vs FIFO | delay p50 / p99 / max |
|---|---|---|---|---|
| 4 | 8 (2×W)  | 1550 | −17.8 % | 0 / 7 / 8 |
| 4 | 16 (4×W) | 1366 | −27.5 % | 1 / 14 / 15 |
| 4 | 192 (all)| 1160 | −38.5 % | 13 / 74 / 76 |
| 8 | 16 (2×W) | 938  | −22.4 % | −2 / 21 / 25 |
| 8 | 64 (8×W) | 790  | −34.6 % | −1 / 62 / 69 |
| 8 | 192 (all)| 740  | −38.8 % | 10 / **102** / 106 |

**Verdict (measured):** route-aware scheduling is genuinely worth **17–39 % of
bandwidth on top of FIFO**, and the win grows with scheduler freedom, as the analysis
claims. **But two caveats the analysis omits are decisive for a serving system:**

1. **Batch size dominates.** Going FIFO W=2→8 saves 52 % of bandwidth at **zero**
   reordering cost. That is the first lever to pull; a busy server that already
   batches for compute gets it for free.
2. **Reordering trades latency for bandwidth, and the trade gets bad fast.** At full
   lookahead, W=8 route-aware saves 39 % but pushes p99 delay to **102 positions** —
   a request waits behind roughly half the entire queue. A **bounded** lookahead
   (Q ≈ 2–4×W) captures about half the reorder win (−18 % to −28 %) for a p99 delay
   of 7–25, which is the defensible operating point. Unbounded overlap maximisation is
   not a serving win.

The absolute floor (all 192 co-scheduled, union = 764 distinct keys) is **64
MiB/token** — i.e. perfect scheduling would load each distinct expert once per 192
tokens. No online scheduler approaches that without unbounded latency.

---

## What this establishes, and what it does not

**Measured (granite, this trace, CPU):**
- Routing concentration is **milder than the assumed Zipf** (top 12.5 % → 27 %, not 80 %).
- A **DRAM tier sized to the bank removes avoidable SSD traffic entirely**; the SSD
  becomes a cold-miss backstop and DirectStorage is a last-10-20 % optimisation.
- With the SSD gone, **PCIe DRAM→VRAM is the dominant transfer cost**, tying this back
  to VRAM residency/pinning.
- **Route-aware scheduling saves 17–39 % over FIFO**, but **batch size is the larger,
  cost-free lever**, and reordering's bandwidth win is bought with p99/max latency.

**Inferred (stated, not measured):**
- Generalisation from granite's 8-of-32 (25 % working set) to DeepSeek-class 8-of-256
  (~3 %): larger banks likely *more* concentrated → tiering and scheduling wins likely
  *larger* there, but this trace cannot show it.
- The three-tier ms/token figures use **assumed** PCIe/NVMe/DDR5 bandwidths, not
  measured on this box.

**Out of scope for any trace-driven simulation:** achieved wall-clock, the real cost
of the paging mechanism (CUDA VMM 2 MiB granule vs OS 4 KiB hints vs the existing
zero-copy host-mapped hybrid), and whether skew holds on a 256-expert router. Those
are the next, mechanism-level decisions — and this study says a residency/tiering
policy is worth having before we spend hardware on them.

## Priority ordering implied by the three arms (inferred synthesis)

1. **Keep experts resident** — VRAM static hot-pin (single-tier study) and batch for
   compute; these are the cost-free levers.
2. **DRAM warm tier** to delete SSD from the hot path — large, cheap, confirmed here.
3. **Route-aware scheduling** with a *bounded* lookahead — a real secondary bandwidth
   win, gated on an acceptable p99 latency budget.
4. **DirectStorage / GDS** — a cold-miss backstop for the compulsory first-touch and
   true capacity overflow, i.e. the last 10–20 %, consistent with the analysis's own
   ranking.

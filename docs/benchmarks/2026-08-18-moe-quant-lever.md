---
title: "MoE offload — the quantisation lever priced on the same axis as VRAM"
date: 2026-08-18
hardware: "Intel i7-13800H (14C/20T), RTX 4060 Laptop 8 GB, WDDM, driver 591.55, CUDA 13.1"
model: "granite-3.0-1b-a400m-instruct (32 experts, top-8, 24 layers), onnxruntime 1.27.0 CPU EP, batch 1"
status: measured (routing) + inferred (bandwidth, expert size) — strictly separated
---

# MoE offload — the quantisation lever

The [composed PCIe floor](2026-08-18-moe-offload-pcie-floor.md) established that
**PCIe DRAM→VRAM bytes/token is the wall** for a MoE whose expert bank overflows
VRAM. Because bytes/token scales **linearly with expert size**, quantisation
attacks that wall directly — no policy, no prediction, no scheduler. This prices it
on the *same axis* as the VRAM (static-hot-pin) lever, using the same measured
granite routing curve.

Routing is **measured** (granite trace, CPU-only, no GPU). Expert **size and every
bandwidth constant are inferred/assumed**. Provenance hw (house rule §32.2):
**i7-13800H (14C/20T), RTX 4060 8 GB, WDDM.** Reproduce:
`python scripts/moe_quant_lever.py`.

---

## The lever, on the same axis as VRAM (MEASURED routing, INFERRED sizes)

At **fixed 25 % VRAM**, static hot-pin, batch 1:

| expert size | bytes/token | equivalent VRAM lever |
|---|---|---|
| 16 MiB | 1670 MiB/tok | (baseline, 25 % VRAM) |
| 8 MiB (2×) | 835 MiB/tok | ≈ static-pin at **50 %** VRAM (786) |
| 4 MiB (4×) | 418 MiB/tok | ≈ static-pin at **65 %** VRAM (416) |

**A 4× expert-size reduction at 25 % VRAM equals what raising VRAM to 65 % buys** —
a 40-percentage-point VRAM saving, which for a bank that overflows VRAM is simply
unattainable. And **the two levers compose**: 4 MiB experts at 50 % VRAM = 197
MiB/token, half again what either buys alone. This confirms the prediction: **in the
25–50 % VRAM range quantisation beats anything static pinning buys**, measured
against our own curve, not asserted.

### tok/s ceiling from transfer alone, by expert size (W=1, PCIe4 x8 ~13 GB/s, INFERRED)

| VRAM % | 16 MiB | 8 MiB | 4 MiB |
|---|---|---|---|
| 25 % | 8.0 | 15.9 | 31.9 |
| 50 % | 16.9 | 33.9 | 67.7 |

Each halving of expert size doubles the ceiling, as expected. At batch 8 and/or
PCIe 5.0 the numbers scale up proportionally (see `moe_quant_lever_results.json`).

### Revised rule of thumb — min VRAM % for target tok/s (W=1, x8, INFERRED)

| expert | 5 | 10 | 20 | 40 tok/s |
|---|---|---|---|---|
| 16 MiB | 5 % | 50 % | 65 % | 75 % |
| 8 MiB  | 5 % | 5 %  | 50 % | 65 % |
| 4 MiB  | 5 % | 5 %  | 5 %  | 50 % |

**Quantising to a 4 MiB expert drops the VRAM needed for 20 tok/s from 65 % to 5 %.**
That is the most decisive single lever in the whole study — but it is the only one
that trades accuracy, and the factor depends entirely on the starting precision.

---

## The two honesty constraints (these decide whether the factor is real)

### 1. Quantisation trades accuracy — it is a product decision, not a free win
Unlike pinning and batching (which are lossless in output), smaller experts mean
lower numerical precision. The bytes/token and tok/s wins above are only realisable
if the resulting model quality is acceptable for the product. We flagged a 1-ULP
reassociation elsewhere this week; a genuine precision *change* deserves at least as
explicit a label. **Every quant row here is a quality trade, not a free speedup.**

### 2. The baseline precision changes the answer completely
The "16 MiB target expert" is a GLM/DeepSeek-class size, and what it *already is*
decides what quantisation can still buy:

- **If the target expert is f16/bf16:** int8 = 2×, **int4 = 4×** — a well-understood,
  routinely-shipped accuracy trade. This is the easy, large win and the recommended
  default for any offload deployment.
- **If the target 16 MiB expert is *already* int4** (plausible: a DeepSeek-V3-class
  expert of ~29M params is ~14 MiB at int4), then the 8 MiB / 4 MiB rows mean **int2
  / int1** — sharp accuracy loss and little kernel support. In that case **there is
  no further quantisation lever on the offload path**, and the 16 MiB row is already
  the quantised floor.

**Nail this down before quoting a factor.** "4× from quantisation" is true from f16
and false (unavailable) from int4.

### 3. Mechanism constraint: only canonical int4 is file-backable
The mechanism agent measured that **only the canonical MatMulNBits int4 layout can
be used file-backed**; Marlin and MLAS-prepacked tensors need a layout that does not
exist on disk and are **excluded** from any storage-to-device / mmap path. So on the
offload path the quantisation lever effectively means **"get the expert to canonical
int4"** — not arbitrary sub-int4 or exotic formats. This lines the two constraints
up neatly: **canonical int4 is both the file-backable format and the sweet spot of
the accuracy trade**, so from an f16 baseline the recommended and mechanically-viable
target is the same point — int4 — for a clean 4×.

---

## Where quantisation sits among the levers (synthesis)

On the same bytes/token axis, for a bank that overflows VRAM:

1. **Quantise f16→canonical int4 (4×)** — the largest single lever, file-backable,
   accuracy-traded. If the experts are f16 on disk, do this first.
2. **Keep experts resident (static hot-pin) + DRAM tier** — lossless; static pin
   captures 50–84 % of the oracle's byte win, DRAM removes the SSD.
3. **Batch** — halves bytes/token, but **not free in wall-clock** on this stack
   (sibling: M=1→2 ≈ 5.4×/step for 2× work), so a bandwidth lever only for now.
4. **Route-aware scheduling (bounded lookahead)** — a secondary bandwidth win with a
   p99 latency cost.

Quantisation and residency **compose multiplicatively** on bytes; the honest ranking
is quantise-if-you-can (accuracy permitting), then keep as much resident as VRAM
allows.

---

## What this establishes / does not
- **Measured:** the static-hot-pin experts/token curve (granite trace); the linear
  same-axis comparison follows exactly from it.
- **Inferred:** expert sizes (16/8/4 MiB are illustrative target-class sizes, not
  granite's own 0.75 MiB int4), every bandwidth, and thus every ms/token and tok/s.
- **Not established:** the *accuracy* cost of each quant level (that needs an eval,
  not a trace), achieved wall-clock, the paging mechanism's cost, and generalisation
  to a 256-expert router.

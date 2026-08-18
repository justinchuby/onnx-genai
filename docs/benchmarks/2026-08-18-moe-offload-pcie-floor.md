---
title: "MoE offload — the composed PCIe floor: how much VRAM for a target tok/s"
date: 2026-08-18
hardware: "Intel i7-13800H (14C/20T), RTX 4060 Laptop 8 GB, WDDM, driver 591.55, CUDA 13.1"
model: "granite-3.0-1b-a400m-instruct (32 experts, top-8, 24 layers), onnxruntime 1.27.0 CPU EP, batch 1"
status: measured (routing) + inferred (bandwidth constants) — strictly separated
---

# MoE offload — the composed PCIe floor

This composes the two prior trace-driven studies into the one number a deployer
actually needs. Inputs:

- [Single-tier policy headroom](2026-08-18-moe-expert-cache-policy-headroom.md):
  the best *practical* residency policy is **static hot-pin** (pin top-B experts
  by measured global frequency; it beat LRU/FIFO/LFU/hybrid at mid budget and sat
  17–54 % above the Belady oracle).
- [Tiering / concentration / scheduling](2026-08-18-moe-tiering-and-route-aware-scheduling.md):
  with a **DRAM tier holding the whole bank, avoidable SSD traffic is zero** and
  **PCIe DRAM→VRAM becomes the only per-step expert-transfer cost.**

So the composed cost is: *DRAM holds the bank, static hot-pin keeps the hottest
experts in VRAM, and everything else crosses PCIe every time it is used.* This
report expresses that as bytes/token, a time floor, and — inverted — the VRAM a
given MoE needs to hit a target tok/s.

All routing numbers are **measured** on the granite trace (3 prompts × 64 greedy
steps = 192 decode steps, 24 layers × top-8 of 32, real IBM trained router),
**CPU-only replay, no GPU**. Hardware stamped for provenance (house rule §32.2):
**i7-13800H (14C/20T), RTX 4060 8 GB, WDDM.** Every bandwidth constant is
**inferred / assumed**, not measured on this box. Reproduce:

```
python scripts/moe_pcie_floor.py    # reads scripts/moe_expert_trace.json
```

---

## The composed floor (MEASURED routing; INFERRED bandwidth)

PCIe experts/token under static hot-pin, as a function of VRAM budget (fraction of
the 768-key bank), with DRAM holding the bank. Experts @ **16 MiB** (GLM/DeepSeek
class); granite's own 0.75 MiB int4 experts are 21× smaller and never need this.
Time and tok/s are the **transfer floor only** — before any compute — at three
assumed PCIe bandwidths (PCIe 4.0 x8 ≈ 13, x16 ≈ 26, PCIe 5.0 x16 ≈ 55 GB/s).

### Batch W=1 (one token per step)

| VRAM % | PCIe exp/tok | GB/tok @16 MiB | tok/s @13 | tok/s @26 | tok/s @55 | oracle exp/tok |
|---|---|---|---|---|---|---|
| 5 %  | 166.2 | 2.60 | 5.0  | 10.0 | 21.2 | 162.6 |
| 10 % | 148.0 | 2.31 | 5.6  | 11.2 | 23.8 | 134.0 |
| 25 % | 104.4 | 1.63 | **8.0**  | **15.9** | 33.7 | 82.2 |
| 35 % | 79.4  | 1.24 | 10.5 | 20.9 | 44.3 | 59.3 |
| 50 % | 49.1  | 0.77 | 16.9 | 33.9 | 71.6 | 33.3 |
| 65 % | 26.0  | 0.41 | 32.0 | 63.9 | 135  | 17.2 |
| 75 % | 14.1  | 0.22 | 59.1 | 118  | 250  | 10.4 |
| 90 % | 2.7   | 0.04 | 310  | 620  | 1312 | 4.8 |

### Batch W=8 (bytes/token amortised over the batch's non-pinned expert union)

| VRAM % | PCIe exp/tok | GB/tok @16 MiB | tok/s @13 | tok/s @26 | tok/s @55 |
|---|---|---|---|---|---|
| 5 %  | 70.7 | 1.11 | 11.8 | 23.5 | 49.8 |
| 25 % | 51.7 | 0.81 | 16.1 | 32.2 | 68.1 |
| 50 % | 29.2 | 0.46 | 28.5 | 57.0 | 121  |
| 75 % | 9.9  | 0.15 | 84.3 | 169  | 357  |

**Static hot-pin sits ~17–27 % above the oracle floor** across this range
(e.g. 25 %: 104.4 vs 82.2 exp/tok), consistent with the single-tier study — the
practical policy leaves a modest, bounded amount on the table.

---

## The alarming headline, stated plainly (MEASURED routing, INFERRED bandwidth)

On a realistic **PCIe 4.0 x8** link (~13 GB/s — plausible for this 4060 laptop
class), a 16 MiB-expert MoE offloaded with **DRAM holding the bank and static
hot-pin keeping 25 % of experts in VRAM** transfers **1.63 GB per token** and is
capped at **≈ 8 tok/s from expert transfer alone, before any compute**, at batch 1.
Even at 50 % VRAM it is ~17 tok/s. **This is the single most important fact about
MoE offload on a bandwidth-limited link: for a bank that does not fit in VRAM, the
decode rate is PCIe-bound in the single-digit-to-low-double-digit tok/s range, and
compute only makes it slower.** DirectStorage/faster-SSD cannot help here — the SSD
is already out of the path; the wall is PCIe.

Two levers move it, and both have caveats:

1. **More VRAM.** Because routing is only *mildly* skewed (see below), you must keep
   a **large** fraction of the bank resident — ~50 % for ~17 tok/s, ~65 % for ~32
   tok/s at x8 — not just the hot 12.5 %. For a bank that overflows VRAM by a lot,
   that fraction is unreachable, so offload of a *much*-larger-than-VRAM MoE is
   transfer-bound regardless of policy.
2. **Batching.** W=8 roughly halves GB/token (25 %: 1.63 → 0.81 GB/tok) because the
   batch's non-pinned experts are loaded once and serve 8 tokens.

---

## The knee is *later* than a Zipf assumption predicts (MEASURED)

Marginal PCIe saved per +1 % VRAM (W=1, @16 MiB) declines **smoothly**, with no
cliff:

| VRAM step | GB/tok | MiB/tok saved per +1 % VRAM |
|---|---|---|
| 5→10 %  | 2.60→2.31 | 58 |
| 20→25 % | 1.84→1.63 | 43 |
| 35→50 % | 1.24→0.77 | 32 |
| 50→65 % | 0.77→0.41 | 25 |
| 75→90 % | 0.22→0.04 | 12 |

There is **no sharp knee**. The concentration curve is flatter than the assumed
80/20 Zipf (top 12.5 % of keys → 27 %, top 25 % → 46 %, top 50 % → 74 % of
traffic), so **you keep paying for VRAM well past the point people expect**: even
at 50 % VRAM you still stream ~49 experts/token. The actionable inversion is that
a *mildly-skewed* MoE needs VRAM covering a **large** share of its bank — roughly
half or more — to get transfer cheap. A model whose router is *more* Zipfian
(larger banks are expected to be — **inferred**) would have an earlier knee and
would need less VRAM; granite is the conservative, flatter end.

---

## Rule of thumb: minimum VRAM to hit a target tok/s (transfer floor only)

Read as "to reach *T* tok/s from expert transfer alone, keep at least this fraction
of the expert bank resident in VRAM". **All bandwidths inferred; @16 MiB experts.**

| batch | link | 5 tok/s | 10 tok/s | 20 tok/s | 40 tok/s |
|---|---|---|---|---|---|
| W=1 | x8 (13 GB/s)  | 5 %  | 35 % | 65 % | 75 % |
| W=1 | x16 (26 GB/s) | 5 %  | 5 %  | 35 % | 65 % |
| W=1 | PCIe5 (55)    | 5 %  | 5 %  | 5 %  | 35 % |
| W=8 | x8 (13 GB/s)  | 5 %  | 5 %  | 50 % | 65 % |
| W=8 | x16 (26 GB/s) | 5 %  | 5 %  | 5 %  | 50 % |
| W=8 | PCIe5 (55)    | 5 %  | 5 %  | 5 %  | 5 %  |

The practical reading: **batch size and PCIe generation matter as much as VRAM.**
Batching W=8 plus PCIe 5.0 removes the transfer wall for these targets almost
entirely; batch-1 on PCIe 4.0 x8 needs 65–75 % of the bank resident for 20–40
tok/s, which is unattainable for a bank far larger than VRAM.

---

## The batching caveat — the byte saving is NOT free in wall-clock (cross-reference)

Every W=8 figure above says "batching halves bytes/token", and in **bytes** that
is exactly true. But **batching is not free in wall-clock on this stack today.** A
sibling measurement finds that going **M=1 → M=2 costs ~5.4× per step for 2× the
work** (~2.55 → ~14 ms), a fixed batch-decode penalty, with the GEMV *excluded* as
the cause. So the −52 %/−61 % bytes-per-token wins from batching (here and in the
scheduling study) hold **in bandwidth only**; realising them in tokens/second
depends on that batch-decode penalty being fixed first. This figure must be read
as "the transfer floor batching *could* buy", not an achieved throughput.

## Mechanism note (not this slice)

Which layer does the paging — CUDA VMM (2 MiB granule), OS page cache / `mmap` +
`cudaHostRegister`, or DirectStorage — is a **separate slice** (another agent).
This study strengthens the cheapest-first instinct there: if DRAM holds the whole
bank, **the OS page cache is already doing precisely the job the DRAM tier here
models**, and our managed path historically lost to the OS by ~30×. The
zero-copy host-mapped hybrid (`weight_paging.rs` #864) is the mechanism that turns
"expert resident in DRAM" into "GPU reads it over PCIe" without a VRAM copy —
i.e. it is exactly the PCIe cost this report prices.

---

## What this establishes / does not

**Measured (granite trace, CPU):** the PCIe experts/token vs VRAM-budget curve, the
static-pin-vs-oracle gap, the smooth (late) knee, and the concentration that drives
it.

**Inferred (stated):** every GB/s, hence every ms/token and tok/s and every cell of
the rule-of-thumb table — the routing is measured, the *conversion to time* is not.
Also inferred: generalisation to a DeepSeek-class 8-of-256 router (expected to be
*more* concentrated → earlier knee, less VRAM needed; granite is the flatter,
conservative end).

**Out of scope for any trace-driven sim:** achieved wall-clock, the paging
mechanism's real cost, the batch-decode penalty (measured elsewhere, cross-referenced
above), and compute time (this is the *transfer* floor only — real tok/s is lower).

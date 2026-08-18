# Sub-granule MoE VMM packing: the trade curve says "one granule, two experts" — not "commit big and subdivide"

**Date:** 2026-08-18
**Author:** Copilot (VMM/offload slice — #1295 follow-up)
**Owner directive:** "继续推进 vmm、offload、streaming、multi-request batching 对大模型的支持，
提高速度，实现简洁高效" — the vmm/offload slice.

**Question this answers (owner):** the 2 MiB CUDA VMM granule is a hard floor
(`2026-08-18-moe-per-expert-paging-churn.md` §"The 2 MiB granule is a hard floor"),
so granite int4 experts (~0.75 MiB) waste **2.7× physical** when paged individually.
The owner proposed the obvious escape: *commit one large region (tens of MiB), pack
many experts into it, subdivide in our own allocator.* The arithmetic half is
trivially favourable (a 64 MiB commit holding ~85 experts drops granule waste to
~0). **The owner asked for the hard half: because committing is also the unit of
residency, does sub-division still buy selectivity, and under what packing?** This
is that analysis, computed from the already-measured granite router-skew
distribution — no new inference of the design was needed before the numbers.

## Result in one line

> [!summary] Measured
> On the measured granite router skew, the commit-unit granularity that minimises
> **resident mapped physical** (and the owner's own `waste × transferred-bytes`
> product) is **one 2 MiB granule holding two experts** (E=2), *not* a large
> region. Hotness-sorted E=2 packing gives **13.34 MiB/layer/step** resident vs
> **16.00** for per-expert (E=1) and **24.00** for whole-bank/large-region
> (E≥16). Resident physical is **U-shaped with a minimum at E=2 and rises back to
> whole-bank levels for every larger chunk.** The owner's "tens of MiB" region
> sits at the *worst* end of the curve, and any region larger than one layer's
> bank (>24 MiB on granite) is fatally always-resident because every layer is
> active every decode step.

## Hardware / method (house rule §32.2)

- **Box:** Intel i7-13800H (14C/20T), RTX 4060 Laptop 8 GB (driver 591.55, CUDA
  13.1), Windows/WDDM. **This is a distribution analysis: the skew is measured on
  the CPU EP** (the *which experts* question is dtype-/EP-independent — see the
  router-skew doc's "Why CPU is valid"); the granule arithmetic is the CUDA device
  granule (#776). No GPU decode is claimed here.
- **Model:** `granite-3.0-1b-a400m-instruct`, f16 dense via Mobius
  (`C:\Users\justinchu\dev\models\granite-1b-a400m-f16-mobius`). **32 experts,
  top-8, 24 layers, no shared expert**, real trained IBM router.
- **Expert byte size:** granite **int4** ≈ **0.75 MiB/expert** (measured,
  per-expert-paging-churn doc). **Device VMM granule = 2 MiB** (#776, hard floor).
- **Reference baseline:** uniform routing, `reads_per_step = 8/32 = 0.250`; and the
  two named reference points E=1 (per-expert, 2.67× waste) and large-region
  (near-zero granule waste). The **ideal** resident set is the top-8 actually used
  = 8 experts = 6.00 MiB/layer/step useful.
- **Instrument:** `scripts/moe_granularity_analysis.py` reuses the *exact* probe and
  decode loop of `scripts/moe_router_skew.py` (same 3 prompts — English prose /
  Python code / math — × 64 greedy decode tokens = **192 decode steps**), but
  captures the **per-step** selected-expert set for all 24 layers (the committed
  `moe_router_skew_counts.json` holds only aggregate per-(layer,expert) counts,
  which is insufficient to compute per-step chunk-hotness unions). For a static
  packing of experts into commit-unit chunks it computes, averaged over all
  (layer, step): hot-chunk count, resident experts (hot-chunk union), resident
  **physical** MiB (granule-rounded), **transferred** MiB (real content H2D-copied),
  granule waste factor, and always-on chunks (hot in 100% of steps).
- **Two packings compared:** *index-order* (arbitrary, experts 0..31 chunked by E)
  and *hotness-sorted* (experts sorted by aggregate decode selection count desc,
  then chunked — coherent-residency packing that needs no runtime prediction).
- **Repetitions / variance:** deterministic (greedy decode, fixed prompts, CPU) —
  re-running reproduces bit-for-bit. Reproduced 2026-08-18.

## The trade curve (measured — hotness-sorted packing, per layer per step)

`residentMiB` = granule-rounded physical of all hot chunks; `transferMiB` = real
expert content copied H2D; `waste` = committed/useful; `sel%` = resident experts
as a fraction of the ideal 8; ideal transfer = 6.00 MiB.

| E (experts/chunk) | chunk MiB (use/commit) | waste | resident MiB | transfer MiB | waste×transfer | sel% | always-on chunks |
|---|---|---|---|---|---|---|---|
| **1 (per-expert)** | 0.75 / 2.0 | **2.67×** | **16.00** | **6.00** | 16.00 | 100% | 5 |
| **2 (← optimum)** | 1.50 / 2.0 | 1.33× | **13.34** | 10.01 | **13.34** | 167% | 5 |
| 3 | 2.25 / 4.0 | 1.75× | 22.41 | 12.63 | 22.10 | 210% | 4 |
| 4 | 3.00 / 4.0 | 1.33× | 19.66 | 14.75 | 19.66 | 246% | 5 |
| 6 | 4.50 / 6.0 | 1.33× | 22.77 | 17.08 | 22.77 | 285% | 10 |
| 8 | 6.00 / 6.0 | 1.00× | 19.07 | 19.07 | 19.07 | 318% | 14 |
| 16 | 12.0 / 12.0 | 1.00× | 22.58 | 22.58 | 22.58 | 376% | 24 |
| 32 (whole bank) | 24.0 / 24.0 | 1.00× | 24.00 | 24.00 | 24.00 | 400% | 24 |

Index-order packing is the same shape but uniformly worse (E=2: 14.34 vs 13.34;
E=8: 22.28 vs 19.07) — hotness packing helps at every E≥2. Min resident physical
**and** min `waste×transferred` are both at **E=2** for both packings.

## Reading (measured)

1. **The optimum is one granule holding two experts, and it beats per-expert on
   physical.** Two 0.75 MiB experts fit in one 2 MiB granule (1.5 MiB used, 1.33×
   waste). Because you pay for the whole granule anyway, the second expert's
   physical is **free**: when only one of the pair is hot, its partner rides along
   at *zero extra committed bytes* (same granule). So resident physical drops
   16.00 → **13.34 MiB/layer/step (−17%)** despite holding ~67% more experts.
2. **Larger chunks are strictly worse, not better.** Past E=2 the curve climbs
   back to whole-bank (24 MiB): the granule-waste recovery is real but the
   selectivity loss — cold experts riding along in coarse chunks — dominates it.
   The owner's "tens of MiB" region (E=16/32) is at 22.6–24.0 MiB, i.e. **~70%
   worse than the E=2 optimum** and no better than pinning the whole bank.
3. **The two objectives split, and both still exclude large regions.** Minimum
   **resident physical** is E=2 (what the oversubscription cliff / admission cap
   cares about). Minimum **transferred bytes** is E=1 (6.0 MiB — per-expert streams
   *only* the 8 used, accepting 2.67× physical waste; what H2D bandwidth cares
   about). By the owner's chosen `waste × transferred` product, **E=2 wins** (the
   2.67× waste at E=1 multiplies its low transfer back up to 16.0). Whichever
   metric binds, the answer is **≤ one granule**, never a large region.
4. **Hotness-sorted packing rescues selectivity, modestly.** Coherent-residency
   packing (hot experts clustered) beats arbitrary packing at every chunk size
   (E=2: 13.34 vs 14.34, ~7%; E=8: 19.07 vs 22.28, ~14%). It does **not** rescue
   *large* chunks — even perfectly sorted, E=8/16/32 stay ≥19 MiB. So hotness
   packing is a bonus on top of E=2, not a way to make big regions viable.
5. **Static packing is stable across prompt domains.** Under a single *global*
   hotness packing decided once, per-prompt resident physical decays only
   **3.4–7.1%** vs an oracle that repacks per prompt (E=4: prose +4.1%, code +5.2%,
   math +3.4%; E=8: +6.4% / +7.1% / +3.5%). The layer-1/2 always-on experts remain
   free pins (~5 always-on chunks at E≤2). Static hot/cold packing is justified.

## Reading (inferred)

- **Cross-layer packing is fatal, which kills the literal "64 MiB / 85 experts"
  proposal.** Every decode step selects experts in **all 24 layers**, so a chunk
  spanning layers is hot in 100% of steps — selectivity collapses to
  `experts_in_chunk / 8`. A 64 MiB region holds ~85 granite experts, but a layer's
  entire 32-expert bank is only 24 MiB, so 64 MiB **must** span >2 layers and is
  therefore resident every single step = whole-bank pinning. The favourable
  "near-zero granule waste" arithmetic is real but irrelevant, because the binding
  objective is resident physical, not waste. *(Inferred from the all-layers-active
  structure of decode, not from a live 64 MiB run.)*
- **The honest recommendation is conditional on who pages.** See below.

## Interaction with #1295 (what I already established)

- **Span count is *not* the oversubscription cliff driver** (#1295 characterisation,
  refuted with a bare-driver-API repro). So E=2's halving of the granule/commit
  count (16 vs 32 granules/layer) will **not** help the cliff, and this doc does not
  claim it does — that is a *different* problem. E=2's map/unmap *call*-count
  reduction only touches the ~3–6 ms/step VMM overhead measured in the churn doc.
- **But E=2 lowers peak mapped physical, which helps the admission cap (PR #1325).**
  The cap refuses admission when summed mapped physical would exceed usable VRAM ×
  0.90. Lowering the resident expert working set 16.00 → 13.34 MiB/layer/step (and,
  for the all-resident bank, halving 2.67× → 1.33× waste) means the cap admits more
  concurrent sequences before refusing — it **raises the plateau height**, not the
  fundamental `N_max` cliff. This is the same cliff→plateau honesty as #1325:
  packing buys throughput headroom under the cap, it does not move the ceiling.
- **The cap must remain the single shared authority.** Whatever packing MoE adopts,
  its resident physical is governed by the same device-authority ledger the cap
  sources from usable VRAM — MoE does not get a private budget.

## The decisive framing: sub-division only matters if *we* keep paging

- **If we keep our own VMM map/unmap:** implement **E=2, hotness-sorted, within-layer**
  (two granite int4 experts per 2 MiB granule, packed by static aggregate hotness).
  It is provably the trade-curve optimum for this model and a *small* change — it
  still commits exactly one granule (the floor), it just stops wasting that granule
  on a single sub-granule expert. **Do not build large-region commit-and-subdivide**
  — the measurement says it is dominated on every metric and degenerates to
  whole-bank residency once it spans layers.
- **If OS hinting works, sub-division is unnecessary.** Windows unified-memory / mmap
  paging operates at **4 KiB pages**, which dissolves the granule problem entirely:
  a 0.75 MiB expert is 192 independently-faultable pages with **zero granule waste
  and full per-expert selectivity** — strictly better than any VMM packing
  (it achieves E=1's 6.0 MiB transfer *and* 1.0× waste simultaneously). No packing
  scheme can match that.
- **My #1295 finding leans toward OS paging.** The binding cost I measured is **WDDM
  fault-in beyond usable VRAM**, not our own bookkeeping; if the driver, not our
  map/unmap, owns the expensive part, our VMM machinery is largely fighting for
  control it does not benefit from. So the *condition under which we should not build
  even E=2* is: **the OS-hint agent confirms WDDM honours residency hints with a
  prediction window long enough to pre-fault the next step's experts.** I have **not**
  measured hinted-vs-unhinted fault-in — that is their measurement — but my data does
  say the cost worth hiding is exactly per-page fault-in, which is what a prefetch
  hint would target. Recorded for that slice.

**Recommendation:** do not implement large-region commit-and-subdivide under any
condition. If we stay on our own VMM paging, the minimal correct change is E=2
(two experts per granule, hotness-packed, within-layer) — a modest −17% resident
physical that feeds the admission cap. If OS hinting is honoured, prefer it and
skip packing entirely. **Reporting before implementing, as asked** — no packing
code was written.

## Reproduction

```powershell
# From repo root; CPU EP, no GPU. Requires onnxruntime>=1.27, onnx, numpy, tokenizers.
cd scripts
python moe_granularity_analysis.py
# Reuses scripts/moe_router_skew.py's probe on the granite f16 Mobius fixture,
# captures per-step selection sets, and prints the trade curve for both packings.
```

## Formal sources

- Granule floor / per-expert paging cost: [`2026-08-18-moe-per-expert-paging-churn.md`](2026-08-18-moe-per-expert-paging-churn.md)
- Router skew distribution (the input to this analysis): [`2026-08-18-moe-router-skew-granite.md`](2026-08-18-moe-router-skew-granite.md), [`scripts/moe_router_skew.py`](../../scripts/moe_router_skew.py)
- Admission cap sourced from usable VRAM (the cap this feeds): PR #1325, [`2026-08-19-vmm-admission-cap-usable-vram-and-free-attribution.md`](2026-08-19-vmm-admission-cap-usable-vram-and-free-attribution.md)
- Oversubscription cliff characterisation (span count not the driver): [`2026-08-18-batch-n-streaming-cliff-wddm-oversubscription.md`](2026-08-18-batch-n-streaming-cliff-wddm-oversubscription.md)
- This analysis: [`scripts/moe_granularity_analysis.py`](../../scripts/moe_granularity_analysis.py)

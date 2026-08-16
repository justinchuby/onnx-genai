### 2026-08-15: GQA group-head KV reuse prototype no-go / attention floor
**By:** Deckard
**What:** Prototyped the final structural attention lever: one CTA computes multiple query heads that share a KV head so K/V loads are reused across the GQA group. The prototype kept per-query online-softmax order unchanged and only changed where K/V was loaded/consumed. The measured result regressed, so no kernel change is kept.

#### Reuse factor confirmed

| Model | Query heads | KV heads | GQA group size |
|---|---:|---:|---:|
| glm-4-9b-int4-cuda | 32 | 2 | 16 |
| qwen2.5-14b-int4 | 40 | 8 | 5 |

#### Prototype design

Implemented a bounded variant with `group_heads_per_cta` = 2 and 4 (rather than all 16 glm heads) to cap register pressure. Each CTA still owned one KV head and one split, but computed 2 or 4 query heads from that KV group. K/V `half2` loads were shared across those query heads; each query head kept its own fp32 running max/sum and value accumulator so the per-head reduction order matched the current kernel.

#### Measurements, glm KV2048, graph ON, H200 GPU5

| Kernel/config | e2e median | Notes |
|---|---:|---|
| Current main / 8-warps baseline | ~185-186 tok/s in same-session baselines | #1007 kernel |
| Prototype, 2 query heads/CTA | 172-175 tok/s | regressed |
| Prototype, 4 query heads/CTA | 161.96 tok/s | regressed harder |

NCU for the 2-head prototype confirmed the intended load reduction but also the failure mode:

| Metric | Current #1007 | 2-head prototype |
|---|---:|---:|
| Grid size per core launch | 512 CTAs | 256 CTAs |
| Registers/thread | 43 | 118 |
| Theoretical occupancy | 62.5% | 25% |
| Achieved occupancy | ~45% | ~21.5% |
| Global sectors (sample) | ~1,096,960 | ~307,456 |
| Dominant stall | L1TEX scoreboard | L1TEX scoreboard, lower per-warp cycles but not enough |
| Core duration/call | ~19.4-20.5 us | ~18.7-19.1 us |

The prototype did cut the number of K/V memory sectors substantially, but the extra per-query accumulators inflated register usage to 118 regs/thread and collapsed occupancy. Per-call core time barely improved, and halving the grid did not overcome scheduler/occupancy loss plus added arithmetic/control overhead. The 4-head version worsened this trade-off.

#### Conclusion

The final structural reuse lever is a measured **NO-GO** for the current kernel shape. Simple bandwidth/coalescing, shared staging, more warps, and GQA group-head reuse have now all been tested. Attention appears at a practical floor without a much larger redesign (e.g., different algorithm/layout or specialized multi-head kernel with substantially lower accumulator/register footprint). The honest post-wave standing remains: native is ahead of reproducible ORT graph-off and still behind certified ORT graph-on; remaining gap is not a small attention-kernel tweak.

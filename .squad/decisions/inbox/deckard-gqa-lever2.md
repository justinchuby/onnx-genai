### 2026-08-15: GQA decode lever-2 diagnosis after #1007
**By:** Deckard
**What:** Profiled the post-wave fp16 split-KV GQA decode core on current main (`6e3dd30f`, includes #1007) at KV2048 and tested the next plausible small levers. No winning kernel change was kept.

#### NCU diagnosis: latency/issue bound, not DRAM-bandwidth bound

Command shape: glm-4-9b-int4-cuda, graph ON, `gqa_decode_attention_f16`, late KV2048 launches, H200 GPU4.

| Metric | Value |
|---|---:|
| Duration per core call | 19.4-20.5 us |
| Block/grid | 256 threads, 512 CTAs |
| Registers/thread | 43 |
| Theoretical / achieved occupancy | 62.5% / ~45% |
| Waves/SM | 0.78 |
| Issue slots busy | ~40-43% |
| Eligible warps/scheduler | ~0.85-0.92 |
| DRAM throughput | ~2.2-2.3% peak |
| Total memory throughput | ~104-110 GB/s (~19% Mem Busy) |
| L1/TEX hit rate | ~7.1% |
| Dominant stall | L1TEX scoreboard dependency, ~8.8-9.6 cycles / ~61% of issue interval |
| Global coalescing | only 768 excessive sectors out of 1,096,960 total (~0%) |

Conclusion: the core is **global-load latency / issue-starved**, not DRAM-bandwidth-bound. K/V cache reads are fp16 `half2` loads; q/k/v lanes are already coalesced. Occupancy is register-capped (`Block Limit Registers=5`, `Block Limit Warps=8`) but raising occupancy alone is not the clear lever.

#### Levers tested and rejected

1. **Shared-memory warp-acc padding (`head_size + 1`)** to attack NCU's shared-bank warning.
   - Registers fell 43→42 but core duration stayed effectively unchanged (19.3-20.6 us vs 19.4-20.5 us).
   - Shared excessive wavefronts only moved 35%→32%; no e2e win, so reverted.

2. **More warps/CTA (12/16) beyond #1007's 8.**
   - KV2048 median, graph ON, runs=3:
     - 8 warps: 186.30 tok/s
     - 12 warps: 178.11 tok/s
     - 16 warps: 181.64 tok/s
   - More intra-CTA warps reduce per-warp key iterations but hurt occupancy/scheduling enough to regress. Reverted.

3. **Shared K/V staging across the existing 8 warps.**
   - Not implemented because the profile/code shows no reuse to exploit inside one CTA: each warp owns disjoint key positions in the split. Staging K/V would load each token into shared memory and then consume it once, adding shared traffic without reducing HBM reads.

#### What remains

The requested candidate levers (coalescing, simple shared K/V staging, more occupancy) are exhausted for this kernel shape. The one remaining plausible attention lever is a larger redesign: compute multiple query heads from the same GQA group in one CTA so a K/V token loaded for a KV head is reused across 2-4 query heads. That could reduce redundant HBM/L1TEX latency, but it will increase registers/accumulator state and may lose the occupancy gained in #1007. It should be treated as a separate high-risk prototype, not a small lever-2 patch.

#### Capture / correctness

No code change is kept. All measurements used graph ON and maintained `fallbacks=0` in the benchmark runs.

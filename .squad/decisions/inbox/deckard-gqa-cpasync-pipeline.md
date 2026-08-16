### 2026-08-15: GQA cp.async latency-hiding pipeline no-go / definitive attention floor
**By:** Deckard
**What:** Prototyped the final attention lever: a 2-stage and 3-stage `cp.async` ring buffer for the fp16 GQA decode core. The prototype copied each warp's next K/V `half2` tile into dynamic shared memory while computing the current online-softmax step, preserving the per-query reduction order. The measured result did not beat the #1007 kernel, so no kernel change is kept.

#### Baseline diagnosis recap

Post-#1007 `gqa_decode_attention_f16` at glm KV2048 is latency/issue-bound, not bandwidth-bound: DRAM is ~2.2-2.3% peak, global coalescing is effectively ideal, and the top stall is L1TEX scoreboard (~61% of the issue interval). Occupancy is ~45% and register/warp-limited.

#### Prototype details

- Added a dynamic shared-memory K/V ring: `[stages][warps][K/V][head/2 half2]`.
- Used `cp.async.ca.shared.global` + commit/wait groups to prefetch future K/V positions.
- Consumed K/V from shared memory; online-softmax max/sum/value accumulation order stayed per-head/per-key identical.
- Tested 2-stage and 3-stage variants. A fully aggressive 2-stage `wait_group 1` variant was rejected because measured multi-run decode became nondeterministic at the tail; the safe variant waits all when fewer than two async groups remain.

#### Measurements, glm-4-9b-int4, graph ON, H200 GPU6, median of 5

| Config | Short ctx | KV2048 | Capture |
|---|---:|---:|---|
| Current #1007/main baseline | 211.15 tok/s | 186.62 tok/s | fallbacks=0 |
| 2-stage cp.async safe-tail | 210.28 tok/s | 186.36 tok/s | fallbacks=0 |
| 3-stage cp.async safe-tail | not better | 186.32 tok/s | fallbacks=0 |

#### NCU result for 2-stage cp.async

| Metric | #1007 baseline | 2-stage cp.async |
|---|---:|---:|
| Core duration/call | ~19.4-20.5 us | ~20.0-20.2 us |
| Registers/thread | 43 | 62 |
| Dynamic shared/block | ~4.16 KB | ~12.35 KB |
| Theoretical occupancy | 62.5% | 50% |
| Achieved occupancy | ~45% | ~41% |
| Issue slots busy | ~40-43% | ~64% |
| Eligible warps/scheduler | ~0.85-0.92 | ~2.5 |
| Warp cycles/issued inst | ~15 | ~8.8-9.1 |
| DRAM throughput | ~2.2-2.3% | ~2.2% |

The pipeline did what it was supposed to do locally: issue efficiency improved and scoreboard pressure was reduced/hidden. But it added many more instructions plus 19 extra registers/thread and 8 KB more shared memory per CTA. Those costs offset the hidden latency, leaving per-call core time flat/slightly worse and e2e slightly regressed.

#### Conclusion

This is the definitive attention floor for the current kernel family. We have now measured and rejected: more split/occupancy beyond #1007, shared staging for bank conflicts, GQA group-head K/V reuse, and cp.async latency hiding. Remaining gap to certified ORT graph-on is not a small GQA decode tweak; it likely requires a fundamentally different attention kernel/layout or accepting other product-level numeric changes (e.g. fp16 GEMV default-on).

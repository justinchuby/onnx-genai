# MRU versus LRU managed weight-residency sweep

Issue: #893. Date: 2026-08-13.

## Verdict

MRU's advantage survives across two models and two pressure points per model,
but its size is strongly budget-dependent. It is not evidence to change the
default yet.

- On Qwen2.5 14B, MRU reduced H2D bytes/token by 22.6% at 3 GB and 33.5% at
  6 GB.
- On Qwen2 0.5B Q4, MRU reduced H2D bytes/token by only 3.1% at 250 MB, but by
  34.1% at 275 MB.
- Every arm emitted byte-identical token IDs within its model.
- Every successful arm had CUDA graph captures greater than zero, zero
  fallbacks, zero oversubscription, peak committed physical bytes below the
  managed limit, and clean VMM ledger counters.

The 0.5B model was chosen because its geometry differs materially from the
original observation: 24 layers / hidden size 896 versus 48 layers / hidden
size 5120, and its authored weights are about 361 MB versus 8.33 GB. Forced
offload was used so both selected budgets exercised the managed residency path.

## Protocol

All runs used the release `profile_native` binary built at `8ef5507c` (the
#892 merge commit), native CUDA backend, greedy decode, 16 emitted tokens, one
warmup, one measured steady run:

```text
ONNX_GENAI_CUDA_GRAPH=1
ONNX_GENAI_MANAGED_WEIGHT_STREAMING=1
ONNX_GENAI_WEIGHT_OFFLOAD_DEVICE_BYTES=<budget>
ONNX_GENAI_WEIGHT_OFFLOAD_EVICT_ORDER=lru|mru
profile_native --model <model> --ep cuda --backend native --tokens 16 --steady
```

`ONNX_GENAI_WEIGHT_OFFLOAD=1` was additionally required for the fitting 0.5B
model. `nvidia-smi --query-compute-apps` was verified empty immediately before
every individual run. No run below was contended. This is the forced-managed
Windows arm; the Windows default since #874 is WDDM zero-copy and does not use
this cache.

## Results

The table leads with the byte-weighted metrics. Count hit rate is included only
as a secondary diagnostic.

| model | budget | order | `byte_hit_rate` | `htod_bytes_per_token` | page-ins | evictions | count hit rate | bypass share of H2D |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5 14B Q4 | 3.0 GB | LRU | 0.00% | 7,870,916,608 | 13,872 | 12,157 | 0.00% | 45.70% |
| Qwen2.5 14B Q4 | 3.0 GB | MRU | **22.60%** | **6,092,302,336** | **10,480** | **8,992** | 24.45% | 47.48% |
| Qwen2.5 14B Q4 | 6.0 GB | LRU | 49.45% | 3,979,122,688 | 4,672 | 3,888 | 66.32% | 49.69% |
| Qwen2.5 14B Q4 | 6.0 GB | MRU | **66.39%** | **2,645,704,704** | **4,096** | **3,568** | 70.47% | 47.49% |
| Qwen2 0.5B Q4 | 250 MB | LRU | 33.30% | 185,351,936 | 3,136 | 3,104 | 45.86% | 41.31% |
| Qwen2 0.5B Q4 | 250 MB | MRU | **35.34%** | **179,687,424** | **2,944** | **2,912** | 49.17% | 42.62% |
| Qwen2 0.5B Q4 | 275 MB | LRU | 94.02% | 16,624,384 | 240 | 224 | 95.86% | 51.18% |
| Qwen2 0.5B Q4 | 275 MB | MRU | **96.06%** | **10,959,872** | **48** | **32** | 99.17% | 77.63% |

MRU versus LRU H2D reductions were 22.6%, 33.5%, 3.1%, and 34.1% respectively.
The direction is consistent, but the 250 MB result is nearly neutral. At
300 MB the 0.5B model became fully resident after warmup (100% byte hit rate,
zero measured H2D), so eviction order had no reachable effect; 275 MB was used
as the second pressured budget. A 200 MB probe failed admission during warmup
and is excluded rather than treated as a measurement.

### Token identity

- Qwen2.5 14B, all four arms:
  `[96347, 3375, 724, 11, 358, 2776, 14589, 311, 6723, 429, 498, 3003, 2581, 6617, 315, 752]`
- Qwen2 0.5B, all four arms:
  `[271, 40, 1079, 264, 48948, 304, 13027, 323, 358, 1079, 4460, 311, 1855, 264, 2025, 429]`

## Interaction with scan-resistant admission

This is not MRU replacing scan-resistant residency. `StableResident` admission
was unchanged; only the victim among evictable residents changed.

The mechanisms are complementary but attack the same cyclic-scan defect:

1. scan-resistant admission pins a stable subset and bypasses weights it elects
   not to retain, preventing the entire cache from churning;
2. MRU preserves older entries within the remaining admitted/churning
   population, which are closer to reuse in the next layer cycle.

Therefore the MRU gain is incremental to the shipped admission policy, but it
must not be added to #723's stable-subset gain as though they were independent.
They share the same causal pathology. The 14B 3 GB LRU arm returning zero hits
despite scan-resistant admission, while MRU recovers 22.6% byte hit rate, shows
that the current admission rule does not eliminate churn at every pressure
point. Conversely, the almost-flat 0.5B 250 MB arm shows admission and bypass
can leave little useful victim-order leverage.

## Reachable ceiling when bypass cannot be changed

Eviction order cannot affect `bypassed_page_in_bytes`: those tensors failed the
admission test and never enter the victim population. For any arm, the strict
counterfactual upper bound is:

```text
maximum removable H2D = total H2D - bypassed H2D
```

| model | budget | LRU H2D/token | LRU bypass/token | maximum reachable share | observed MRU saving | share of reachable ceiling captured |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5 14B | 3.0 GB | 7.871 GB | 3.597 GB | 54.3% | 1.779 GB (22.6%) | 41.6% |
| Qwen2.5 14B | 6.0 GB | 3.979 GB | 1.977 GB | 50.3% | 1.333 GB (33.5%) | 66.6% |
| Qwen2 0.5B | 250 MB | 185.35 MB | 76.58 MB | 58.7% | 5.66 MB (3.1%) | 5.2% |
| Qwen2 0.5B | 275 MB | 16.62 MB | 8.51 MB | 48.8% | 5.66 MB (34.1%) | 69.8% |

These run-local bounds are about half of total H2D because warmup and resident
churn are included in the reported counters. They do not contradict the
#886/#837 steady-state gap attribution: if bypass is about 90% of the
*recoverable 1.158 GB/step gap*, victim order can directly reach at most the
other roughly 10%, or about **0.116 GB/step**. MRU can substantially improve the
cache ratio while still being a secondary lever against the end-to-end gap.
Promoting a tensor at the point where `!can_fit(bytes)` refuses it requires an
admission-policy change, not a victim-policy change.

The measured bypass totals can still differ between orders: preserving a tensor
that was admitted earlier can turn its later access into a hit, avoiding a new
admission attempt that LRU would reach and refuse. That indirect avoidance does
not make eviction order capable of admitting a currently refused tensor, and it
does not remove the admission-policy ceiling identified by #886/#837.

## Recommendation and falsification

Keep LRU as the default. MRU is directionally positive in all four pressured
comparisons, but the effect ranges from 3% to 34%, the second model required
forced offload because it normally fits, and the dominant bypass term is
unreachable.

Reconsider the default only after a naturally over-budget second large model
(preferably a different architecture, such as Gemma) and Linux reproduce a
meaningful byte/token reduction. Evidence that should falsify a default flip:
any representative budget/model with worse H2D bytes/token, token mismatch,
ledger/capture failure, or a median improvement small enough to disappear
under repeated counter measurements.

# Qwen 14B native CUDA offload and capture

- Date: 2026-08-12
- Host: `DESKTOP-POS52H4`
- Measured commit: `514181d63d872aca8701250ce5c9b1040ef942bb` (`origin/main`)
- Model: `C:\Users\justinchu\dev\models\qwen14b-zp`
- Runtime: ONNX Runtime 1.27.0, Rust 1.97.1

This closes the large-model measurement gap recorded in #796 and exercises the
default managed no-spill path from #798 on the model made native-loadable by
#384. It also records the current scan-resistant residency result from #723 and
physical-commit evidence relevant to #704.

## Method

No explicit `--vram-limit` or weight-offload override was supplied. Capture ON,
OFF, then ON again ran in separate child processes because `RuntimeConfig` is
process-frozen:

```powershell
$env:ONNX_GENAI_CUDA_GRAPH = "1" # then "0", then "1"
$env:ONNX_GENAI_REQUIRE_CUDA = "1"
.\target\release\profile_native.exe `
  --model C:\Users\justinchu\dev\models\qwen14b-zp `
  --ep cuda --backend native --steady `
  --tokens 8 --warmups 0 --runs 2 --decode-skip 2 `
  --prompt "Virtual memory is"
```

The runs were serialized with the concurrent KV-floor work when possible.
Wall-clock values were visibly affected by contention, so no throughput claim
is made.

## Deterministic results

| Counter | Capture ON | Capture OFF | Capture ON repeat |
|---|---:|---:|---:|
| captures | 2 | 0 | 2 |
| replays | 11 | 0 | 11 |
| fallbacks | 0 | 0 | 0 |
| invalidations | 2 | 2 | 2 |
| page-ins | 5,299 | 5,299 | 5,299 |
| cache hits | 8,573 | 8,573 | 8,573 |
| hit rate | 61.80% | 61.80% | 61.80% |
| evictions | 3,850 | 3,850 | 3,850 |
| bypassed page-ins | 691 | 691 | 691 |
| generated token IDs | `[264, 7286, 429, 374, 537, 1632, 15985, 553]` | same | same |

The two measured generations inside every process were also identical. Decoded
text was `" a concept that is not well understood by"` in all three processes.

Capture ON had no decline reason. Capture OFF reported the expected named
reason:

```text
predicate `ONNX_GENAI_CUDA_GRAPH` declined capture: the process-wide runtime
configuration captured an explicit value of 0 on first use
```

Therefore #796 generalizes to this genuinely over-budget model: graph capture
and replay occurred in the same sessions that performed thousands of weight
page-ins and evictions. Capture ON and OFF produced identical greedy output.

## Default strategy and no-spill evidence

The default path selected:

```text
strategy=DynamicWeightResidency
inferred=DynamicWeightResidency
access=SequentialDense
total_weight_bytes=16653143582
kv_bytes_per_token=196608
resolved_device_budget_bytes=7730940928
fits_resolved_device_budget=false
weight_offload_enabled=true
managed_no_spill=true
scan_resistant_dense=true
device_budget_bytes=6120328192
```

Thus #798's automatic streaming path engaged without an explicit byte limit.

Physical counters were identical in all three processes:

| Physical measure | Bytes | Share of resolved 7,730,940,928-byte budget |
|---|---:|---:|
| weight mapped physical bytes | 5,683,281,920 | 73.51% |
| authority physical handles owned | 6,826,229,760 | 88.30% |
| VMM physical bytes mapped at report | 6,557,794,304 | 84.83% |
| VMM peak physical bytes mapped | 6,996,099,072 | 90.49% |

The resource ledger reported zero oversubscription. VMM reported zero reference
underflows, byte underflows, and unaccounted committed bytes. Together with
`managed_no_spill=true`, this confirms the run stayed within the no-spill
`cuMemCreate` physical budget rather than silently using shared system memory.

## Residency result

The current scan-resistant policy was active, but this workload measured a
61.80% hit rate and 3,850 evictions, not the earlier 74.18% / zero-eviction
result reported for #723. The counters repeated exactly across capture ON/OFF/ON,
so this is not capture-dependent noise. It is an honest large-model result, but
not an old-policy A/B comparison; this measurement alone does not quantify the
improvement over the pre-#723 policy.

`vram_alloc` was 0 ms in all three runs. Measured `vram_free` was 7,080.780 ms,
9,185.736 ms, and 15,504.791 ms respectively. Those durations varied with
contention and are not treated as deterministic, but they make the eviction
churn visible as expected.

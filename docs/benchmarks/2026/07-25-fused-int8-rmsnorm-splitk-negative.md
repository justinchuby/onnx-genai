# Fused INT8 RMSNorm GEMV split-K experiment

Device: NVIDIA H200, GPU 2, `taskset -c 1`. Model:
`Phi-4-mini-instruct-cuda-gpu-5/v5`. Baseline revision: `540c3108`.

Nsight Compute confirmed the fused asymmetric INT8 decode kernel was
grid-starved: 640 blocks, 0.61 waves/SM, and 56.31% achieved occupancy. The
experiment adaptively selected two K partitions from the live SM count,
threads/SM, shared-memory/SM, blocks/SM, K, and N. This raised the launch to
1,280 blocks, 1.21 waves/SM, and 78.08% achieved occupancy. Consumer-sized
28-46 SM configurations selected one partition because the original
640-block grid already filled a resident wave.

Despite the occupancy increase, Nsight Systems showed the kernel median
increase from 18.016 us to 20.832 us (+15.6%). Repeating the full-model
benchmark in alternating baseline/split-K order with
`--steady --warmups 2 --runs 3 --tokens 128` gave:

| Variant | Median tok/s | Intra-group spread |
|---|---:|---:|
| Baseline | 322.33 | 0.06 tok/s |
| Split-K | 317.66 | 0.05 tok/s |

This is a 1.45% decode regression. A preceding alternating pair independently
gave 321.89 versus 318.23 tok/s (-1.14%); one baseline sample in that group
was host-contention affected, but its median remained consistent.

All 128 generated greedy token IDs matched. The 20 targeted CUDA
`MatMulNBits` tests passed; the existing INT8 fp16 oracle tolerance is
`max_abs < max(max_output * 2e-3, 1e-3)` and `max_rel < 5e-2`.

Verdict: do not ship fused split-K. Repeating the full RMSNorm reduction and
normalized-activation staging in twice as many CTAs costs more than the added
latency hiding saves.

## Attempt 2: normalize once, then split-K

A second implementation moved RMSNorm into the existing standalone
`matmul_nbits_rmsnorm_f16_warp_half4` kernel, writing one persistent K-element
fp16 scratch buffer. The following non-fused INT8 GEMV consumed that buffer
with fp32 partial accumulation. Its split factor was selected from live
SM/thread/block limits plus K and N; the modeled 28-46 SM consumer cases again
selected one partition and retained the fused production path.

Nsight Compute confirmed the target split launch was 1,280 blocks and 1.21
waves/SM, with 71.82% achieved occupancy. The extra pre-pass was about 5.76 us
in the Nsight Systems timeline, so removing repeated normalization did not make
the two-launch path cheaper than the original fused kernel.

The final uncontended alternating median-of-three pair was:

| Variant | Median tok/s | Intra-group spread |
|---|---:|---:|
| Baseline | 320.46 | 0.09 tok/s |
| RMSNorm pre-pass + split-K | 316.53 | 2.97 tok/s |

This is a 1.23% decode regression. The preceding alternating pair independently
gave 321.87 versus 314.25 tok/s (-2.37%), with spreads of 0.68 and 1.98 tok/s.
All 128 greedy token IDs matched, and all 20 targeted CUDA `MatMulNBits` tests
passed under the same tolerance quoted above.

Definitive verdict: drop split-K for this fused kernel. Attempt 1 loses to
duplicated RMSNorm work; attempt 2 removes that duplication but the standalone
normalization launch and K-element global round-trip still exceed the grid-fill
benefit. Both implementations were reverted; the production kernel remains
unchanged.

# Pointwise 1×1 Conv: Reuse-Factor-Based BNNS Bypass

**Author:** Deckard (Systems)
**Date:** 2026-07-28T10:27:45Z
**Status:** Implemented
**Relates to:** Sebastian's diagnosis (sebastian-pointwise-conv.md), instance #12 of "optimization exists but is unreachable"

## Decision

Skip BNNS for 1×1 pointwise convolutions when:
1. The weight tensor exceeds the L1 data cache (OC × IC > 16384 = 64KB/4, consistent across Apple Silicon P-cores M1–M4), AND
2. The spatial reuse factor is too low (N ≤ IC × 6, where N = output h×w)

This replaces the original magic-constant threshold with a guard derived from observable properties.

## Mechanism

BNNS copies the full weight tensor (OC × IC × 4 bytes) and creates/destroys a filter on every call. Each weight element is reused N times (once per spatial position). When this reuse factor N/IC < 6, the copy/pack overhead exceeds BNNS's compute advantage from its internal AMX parallelization.

The L1 cache filter uses the E-core L1 size (64KB = 16K f32 elements), deliberately sized to the smaller E-core cache so the guard is conservative and portable across all Apple Silicon core types. P-cores have 128KB L1 (M1–M4). Below this weight size, the copy is cache-local and essentially free on any core.

The reuse factor (BNNS_REUSE_MIN = 6) is empirically fitted on M1 Max over shapes from 24→144 @ 14×14 to 2048→512 @ 7×7. The mechanism (BNNS per-call setup amortized by compute) is general across Apple Silicon; the coefficient 6 is not — it is the observed minimum N/IC ratio at which BNNS recovers its overhead in our measurements.

## Per-Layer Measurements

Interleaved A/B, `--release`, 100 iterations, 10 warmup, corroborated 3× at loads 5–53:

| Shape (IC→OC @ h×w) | N | IC×6 | Weight>L1? | BNNS µs | GEMM µs | Ratio | Guard |
|---------------------|---|------|-----------|---------|---------|-------|-------|
| 24→144 @ 14×14 | 196 | 144 | no (3.5K) | 9 | 12 | 0.75 | keep BNNS ✅ |
| 24→144 @ 20×20 | 400 | 144 | no | 11 | 12 | 0.92 | keep BNNS ✅ |
| 24→144 @ 28×28 | 784 | 144 | no | 17 | 21 | 0.82 | keep BNNS ✅ |
| 96→576 @ 14×14 | 196 | 576 | yes (55K) | 48–56 | 43–55 | 1.02–1.12 | skip → GEMM ✅ |
| 96→576 @ 16×16 | 256 | 576 | yes | 57 | 52–56 | 1.02–1.10 | skip → GEMM ✅ |
| 96→576 @ 20×20 | 400 | 576 | yes | 82–91 | 77–81 | 1.01–1.12 | skip → GEMM ✅ |
| 96→576 @ 24×24 | 576 | 576 | yes | 106–224 | 97–136 | 1.05–1.65 | skip → GEMM ✅ |
| 96→576 @ 28×28 | 784 | 576 | yes | 126–306 | 131–506 | 0.60–0.96 | keep BNNS ✅ |
| 96→576 @ 32×32 | 1024 | 576 | yes | 141–625 | 203–676 | 0.67–0.92 | keep BNNS ✅ |
| 320→1280 @ 7×7 | 49 | 1920 | yes (410K) | 125–156 | 80–104 | 1.50–1.56 | skip → GEMM ✅ |
| 512→128 @ 20×20 | 400 | 3072 | yes (65K) | 83–165 | 51–232 | 1.63–1.65 | skip → GEMM ✅ |
| 512→128 @ 28×28 | 784 | 3072 | yes | 112–308 | 92–469 | 1.20–1.22 | skip → GEMM ✅ |
| 1024→256 @ 14×14 | 196 | 6144 | yes (262K) | 162–397 | 112–588 | 1.43–1.51 | skip → GEMM ✅ |
| 64→256 @ 56×56 | 3136 | 384 | no (16K=L1) | 147–263 | 242–725 | 0.36–0.61 | keep BNNS ✅ |
| 2048→512 @ 7×7 | 49 | 12288 | yes (1M) | 392–1135 | 264–1289 | 1.48–1.51 | skip → GEMM ✅ |

All 15 tested shapes are correctly classified by the guard. Range shows variation across load conditions (5–53).

## Model-Level Impact

Interleaved A/B with `profile_vision`, `--release`, 15 runs, 5 warmup, corroborated 2×:

| Model | Before (ms) | After (ms) | Change | vs ORT |
|-------|-------------|------------|--------|--------|
| EfficientNet-B0 | 76.0 avg | 69.2 avg | **-8.9%** | 5.9× behind (ORT: 11.7ms) |
| MobileNetV2 | 44.31 | 44.30 | **< 0.1%** | 10.6× behind (ORT: 4.17ms) |
| ResNet-18 | 8.55 | 8.17 | **-4.4% (noise)** | 1.6× ahead (ORT: 13.11ms) |

Load during model runs: 16–27 (uptime). All corroborated 2×.

**EfficientNet-B0: real, measured ~9% model-level speedup.** EfficientNet-B0 has heavy MBConv blocks with 1×1 expand/project convolutions at reduced spatial resolutions (14×14 and 7×7), which are exactly the shapes this bypass targets. Four interleaved runs consistently showed 5.6–10.6% improvement.

**MobileNetV2: negligible model-level impact.** Most pointwise layers are at 56×56 and 28×28 (BNNS-favorable, kept on BNNS). Only the late 14×14 and 7×7 layers bypass BNNS, and these represent a small fraction of total runtime.

**ResNet-18: at most a small improvement.** ResNet-18 uses BasicBlock (two 3×3 convs per block), not Bottleneck. The only 1×1 convolutions are skip connections at resolution transitions — very few layers are affected.

## Conclusions

1. The fix is correct and free (no layer regresses).
2. **Model-level gain is architecture-dependent:** models dominated by small-spatial pointwise convolutions (EfficientNet-B0: ~9% speedup) benefit; models whose pointwise layers are at large spatial sizes (MobileNetV2) or have few 1×1 convolutions (ResNet-18) see negligible change.
3. Sebastian's 1.7–2.1× model-level projection was based on raw cblas_sgemm vs full BNNS overhead (5.7–9.8×), which overstated the practical improvement through `im2col_gemm_execute` (1.0–1.6× per layer). The model-level win is real but scoped.
4. The fix prevents the "optimization exists but is unreachable" defect from causing future regressions if the dispatch landscape changes.

## Cross-Cutting Note

Instance #12 of the "optimization exists but is unreachable" pattern. The structural lesson holds. This case demonstrates that per-layer gains translate to model-level wins only when the affected layers dominate runtime — EfficientNet-B0 validates this; MobileNetV2 and ResNet-18 do not.

# LoRA Phase-2 measurements — group-by-adapter dense path vs Phase-1 subgraph

Date: 2026-07-28. Author: Edgemar (native-runtime kernel / memory subsystem).
Machine: 96 logical cores, CPU execution provider, fp32 accumulators throughout.
Harness: `crates/onnx-runtime-ep-cpu/src/bin/lora_grouped_bench.rs`
(`cargo run -p onnx-runtime-ep-cpu --release --bin lora_grouped_bench`).

All timings are mean wall-clock per kernel call. "Phase-1" is the existing
4-node injected subgraph (MatMul A_t -> MatMul B_t -> scale -> Add). "grouped"
is the `GroupedLoraDelta` custom op running the group-by-adapter dense path
(partition rows by adapter, one dense A_t then B_t fp32 matmul per group,
scatter back). The XL fused BGMV/SGMV grouped kernel is **not** implemented —
these numbers exist to decide whether it is worth building.

## Result summary — the single-adapter gate PASSES

The hard gate is: the single-adapter grouped path must NEVER be slower than the
Phase-1 dense path. It passes at both decode and prefill shapes after the
zero-copy f32 factor fix (see "Decode regression" below).

### Single-adapter, tokens = 1 (decode-shaped)

| shape | K | N | rank | Phase-1 (µs) | grouped (µs) | grouped/Phase-1 |
|---|---|---|---|---|---|---|
| attn q_proj 2048x2048 r8  | 2048 | 2048  | 8  | 362.44 | 244.77 | 0.68x |
| attn q_proj 2048x2048 r16 | 2048 | 2048  | 16 | 320.01 | 235.54 | 0.74x |
| attn q_proj 4096x4096 r16 | 4096 | 4096  | 16 | 345.94 | 247.35 | 0.72x |
| mlp gate 4096x11008 r16   | 4096 | 11008 | 16 | 801.73 | 372.25 | 0.46x |

Grouped is 0.46–0.74x of Phase-1 at decode — i.e. faster, gate satisfied.

### Single-adapter, tokens = 128 (prefill-shaped)

| shape | K | N | rank | Phase-1 (µs) | grouped (µs) | grouped/Phase-1 |
|---|---|---|---|---|---|---|
| attn q_proj 2048x2048 r8  | 2048 | 2048  | 8  |  4923.64 | 1820.40 | 0.37x |
| attn q_proj 2048x2048 r16 | 2048 | 2048  | 16 |  5026.76 | 1822.85 | 0.36x |
| attn q_proj 4096x4096 r16 | 4096 | 4096  | 16 |  7341.50 | 2617.45 | 0.36x |
| mlp gate 4096x11008 r16   | 4096 | 11008 | 16 | 19790.40 | 3677.28 | 0.19x |

Grouped is 3–5x faster at prefill: the custom op fuses what Phase-1 spends on
separate node dispatch, intermediate materialization and the extra scale pass.

### Multi-adapter batch (group-by-adapter path)

| shape | batch | distinct adapters | total (µs) | per-token (µs) | tokens/sec |
|---|---|---|---|---|---|
| attn q_proj 4096x4096 r16 |  2 | 2 |  594.51 | 297.256 |  3364 |
| attn q_proj 4096x4096 r16 |  4 | 2 |  574.63 | 143.656 |  6961 |
| attn q_proj 4096x4096 r16 |  4 | 4 | 1073.52 | 268.380 |  3726 |
| attn q_proj 4096x4096 r16 |  8 | 2 |  624.81 |  78.102 | 12804 |
| attn q_proj 4096x4096 r16 |  8 | 4 | 1172.45 | 146.557 |  6823 |
| attn q_proj 4096x4096 r16 |  8 | 8 | 2158.31 | 269.788 |  3707 |
| attn q_proj 4096x4096 r16 | 16 | 2 | 1088.88 |  68.055 | 14694 |
| attn q_proj 4096x4096 r16 | 16 | 4 | 1393.42 |  87.089 | 11483 |
| attn q_proj 4096x4096 r16 | 16 | 8 | 2493.99 | 155.874 |  6415 |
| mlp gate 4096x11008 r16   |  2 | 2 |  773.66 | 386.830 |  2585 |
| mlp gate 4096x11008 r16   |  4 | 2 |  858.66 | 214.664 |  4658 |
| mlp gate 4096x11008 r16   |  4 | 4 | 1583.72 | 395.929 |  2526 |
| mlp gate 4096x11008 r16   |  8 | 2 | 1103.60 | 137.951 |  7249 |
| mlp gate 4096x11008 r16   |  8 | 4 | 1881.59 | 235.199 |  4252 |
| mlp gate 4096x11008 r16   |  8 | 8 | 3165.20 | 395.650 |  2527 |
| mlp gate 4096x11008 r16   | 16 | 2 | 2073.66 | 129.604 |  7716 |
| mlp gate 4096x11008 r16   | 16 | 4 | 2665.35 | 166.584 |  6003 |
| mlp gate 4096x11008 r16   | 16 | 8 | 3766.04 | 235.378 |  4248 |

## Reading the multi-adapter data

Per-token cost is essentially a function of the **number of distinct adapters in
the batch**, not the batch size. For a fixed distinct-adapter count, throughput
scales cleanly with batch (e.g. 4096x4096 with 2 adapters: 297 -> 144 -> 78 ->
68 µs/token as batch goes 2 -> 4 -> 8 -> 16). But for a fixed batch, adding
distinct adapters raises per-token cost almost linearly (batch 8: 2 adapters =
78, 4 = 147, 8 = 270 µs/token).

The floating-point work is `O(total_tokens * (K + N) * rank)` and is
**independent of the distinct-adapter count** — so the growth with adapter count
is pure per-group fixed overhead: one rayon-parallel gemm launch pair, one
scratch resize and one gather/scatter **per adapter group**. When each group
shrinks to a handful of rows (distinct adapters ≈ batch), that fixed overhead
stops amortizing and dominates. This is precisely the regime a fused BGMV/SGMV
kernel targets: it would replace N tiny per-group gemm launches with a single
batched, segment-indexed kernel, amortizing the launch/gather overhead across
all adapters.

## Decode regression found and fixed (honest note)

The first cut of the kernel widened (copied) the full A_t `[K,rank]` and B_t
`[rank,N]` factors into scratch on *every* call via `decode_f32`. At prefill
(128 tokens) this is amortized and invisible; at decode (tokens=1) that copy —
up to ~240K elements for 4096x11008 — dominated and made grouped **1.5–2.6x
SLOWER** than Phase-1, failing the gate. Fix: when the pooled factor dtype is
`Float32`, reinterpret the 64-byte-aligned pool bytes directly as `&[f32]`
(zero copy); only f16/bf16 factors are widened. The unsafe reinterpret is
justified — pool pages are `LORA_PAGE_ALIGNMENT` (64-byte, hence f32-aligned),
byte length is a multiple of 4, the borrow is immutable and non-aliasing. After
the fix, decode grouped drops to 0.46–0.74x of Phase-1 (table above).

## Verdict — is the XL fused BGMV/SGMV grouped kernel worth building now? **NO.**

For our realistic workloads — single-adapter serving, and multi-adapter batches
with a *small* number of distinct adapters — the group-by-adapter dense path is
already sufficient and in fact beats the Phase-1 dense path at every measured
single-adapter shape (decode and prefill) and delivers good throughput scaling
with batch when distinct-adapter count is low (2 adapters: up to ~14.7K
tokens/sec at batch 16).

The XL fused kernel only pays off in the **high-distinct-adapter regime**
(distinct adapters approaching the batch size, each adapter owning only 1–2
rows), where per-group launch/gather overhead stops amortizing. We have no
current workload in that regime, and building a correct fp32-accumulator fused
segment kernel is a substantial, higher-risk effort. Recommendation: **ship the
group-by-adapter dense path; do not build the XL grouped kernel yet.** Revisit
only when a concrete multi-tenant workload with many distinct adapters per batch
(distinct ≳ 4–8 at large batch) is on the roadmap — the crossover in the table
(per-token cost climbing with distinct adapters) is the trigger to watch, and
this harness is the tool to re-confirm it.

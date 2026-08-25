# Benchmark and runtime: do not ask for more decode threads than physical cores

**Author:** Sebastian (Performance Engineer)
**Context:** Phase 19, `docs/benchmarks/2026-08-15-cpu-ep-vs-ort-attention-moe.md` §39

## Decision

CPU benchmarks and deployments should set the decode budget to the host's
*physical* core count, not its logical CPU count. On the 16-physical/32-logical
benchmark host that means `--native-threads 16`, not 32.

## Why

Measured native-only on latest main, a 32-thread budget is never faster than a
16-thread one and is sometimes much slower:

| cell | budget 16 | budget 32 |
| --- | --- | --- |
| `llama3_8b_mlp_t512` | 969 ms | 1537 ms |
| `qwen3_0p6b_qkv_t8` | 0.921 ms | 0.934 ms |

It also costs 2.4x the one-off pool construction (1.55 -> 3.67 CPU-s) and ~8%
more CPU per inference. The shipping default (flat pool capped at 8 workers,
SMT-capped task lanes) costs 0.54 CPU-s where an explicit `32` costs 4.72 CPU-s.

## Consequence

The apparent "t=16 -> t=32 scheduler drift" tracked since phase 16 does not
exist in steady state. It was a warm-up transient plus a fixed pool-construction
cost, both of which the 7-run A/B harness sits inside. Wide-thread rows in the
benchmark ledger should be read with that and with §38's co-residency finding.

The runtime now warns when an explicit budget exceeds the physical core count.
It does **not** override the request: explicit stays explicit.

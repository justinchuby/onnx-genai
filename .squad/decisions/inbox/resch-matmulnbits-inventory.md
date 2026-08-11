# Decision: MatMulNBits CPU has no viable upstream kernel gap

**Author:** Resch
**Date:** 2026-08-11
**Status:** Proposed

## Context
Investigated whether our MatMulNBits CPU kernels (activation quantizers, dot products, M=1 decode) contain optimizations missing from ORT's MLAS.

## Decision
No kernel-level PR should be opened. Upstream MLAS already covers AVX2/AVX-512/VNNI activation quantization and dot products with equivalent instruction selection. The only differences are rounding semantics (design choice, not gap) and NaN safety (too niche). Our value is in runtime orchestration (not portable).

## Consequences
- The MatMulNBits CPU upstream track is closed
- Five of five candidates across the programme have been honest negatives
- Future upstream efforts should focus on non-kernel contributions (tooling, documentation, graph-level)

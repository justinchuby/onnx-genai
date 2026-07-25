### 2026-07-25: DeepSeek native CUDA validation status
**By:** Hudson
**What:** Native CUDA loads all three exercised DeepSeek artifacts. DeepSeek-V2
QMoE matches ORT for 32 greedy tokens and token-0 top-40 log-probabilities
(max absolute delta 0.001409); DeepSeek-Coder matches for 128 tokens.
DeepSeek-R1 diverges at generated token 16 (native 374, ORT CUDA 594), consistent
with the existing fp32-oracle finding that native is more accurate. Stable
native/ORT rates were 629.31/442.76 tok/s for R1 and 798.44/623.51 tok/s for
Coder. The QMoE ORT run was CPU-heavy (four Memcpy nodes, 0% observed GPU,
2.45 tok/s), so it is not a valid GPU speed baseline.
**Why:** The durable status must distinguish numerical correctness from an
invalid ORT performance baseline. Top remaining gaps are full-model QMoE
language-coherence validation, a GPU-resident ORT QMoE reference, and continued
explicit handling of DeepSeek-R1 MatMulNBits accuracy-level divergence.

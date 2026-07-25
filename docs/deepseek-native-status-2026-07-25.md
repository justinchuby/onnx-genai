# DeepSeek native CUDA decode status — 2026-07-25

Validated at `80990b7254adf77620ee88bba2298c16a4bbc210` on an NVIDIA H200.
Commands used the release `profile_native` binary, `CUDA_VISIBLE_DEVICES=0`,
and `taskset -c 1`. Throughput is steady-state greedy decode after eight
skipped tokens.

## Summary

| Model | Native CUDA | ORT CUDA | Greedy parity | Coherence |
|---|---:|---:|---|---|
| DeepSeek-V2 real-shape QMoE int4 | 765.16 tok/s | 2.45 tok/s* | Exact for 32 tokens | Not semantically assessable: conformance tokenizer decodes token IDs as decimal strings |
| DeepSeek-R1-Distill-Qwen-1.5B int4 | 629.31 tok/s | 442.76 tok/s | Diverges at generated token 16 | Both produce fluent text; native answer is factually poor but grammatical |
| DeepSeek-Coder-1.3B int4 | 798.44 tok/s | 623.51 tok/s | Exact for 128 tokens | Coherent Rust function and continuation |

\* The QMoE ORT number is **not a valid GPU performance baseline**. ORT added
four `Memcpy` nodes, sustained sampling showed 0% GPU utilization, and decode
fell to 2.45 tok/s. Native/ORT numerical parity remains useful, but the ORT
execution was CPU-heavy.

## DeepSeek-V2 MoE (QMoE int4)

- Artifact: `~/ds-e2e-artifacts/deepseek-v2-realshape-qmoe-int4`
- Native CUDA loads successfully and executes the `QMoE` kernel.
- Greedy tokens match ORT exactly for all 32 generated tokens:
  `[169, 216, 197, 250, ..., 141, 14, 64, 224]`.
- Token-0 log-probabilities also agree closely. Both select token `169`;
  selected log-probability is `-4.175841` native versus `-4.175086` ORT.
  Maximum absolute difference across the common top-40 is `0.001409`.
- Native median: **765.16 tok/s** (3 runs, 2 warmups, 32 tokens).
- ORT: **2.45 tok/s** (1 run, 1 warmup, 32 tokens), with four inserted
  `Memcpy` nodes and no observed GPU utilization.
- The artifact uses a 256-entry WordLevel tokenizer whose decoded output is a
  sequence of decimal IDs. It proves routing/numerical conformance, not
  natural-language coherence.

**Result:** the current native QMoE path matches ORT for this real-shape
conformance model. No native unsupported op or kernel error was observed.

## DeepSeek-R1-Distill-Qwen-1.5B (dense int4)

- Artifact: `~/glm-e2e-artifacts/deepseek-r1-distill-qwen-1.5b-int4-cuda`
- Both backends load and generate fluent text.
- Native median: **629.31 tok/s**; ORT median: **442.76 tok/s**. Native is
  **1.421x** faster.
- The CUDA baseline is valid: no `Memcpy` warning appeared, and a sustained
  ORT run held approximately 85–87% GPU utilization with 2,979 MiB resident.
- The sequences share the first 15 generated tokens and diverge at generated
  token 16:
  - native: token `374` (`" is"`)
  - ORT CUDA: token `594` (`"'s"`)
- At the shared teacher-forced prefix, native favors `374` over `594` by
  **0.0625 log-probability/logit units** (`-0.882679` vs `-0.945179`);
  ORT CUDA favors `594` by **0.015625** (`-0.896112` vs `-0.911737`).
- This is an already characterized accuracy-level difference, not evidence of
  a native regression: `deepseek_r1_1_5b_divergence.rs` records an independent
  fp32 CPU oracle where native selects the oracle-correct token while ORT CUDA
  flips the argmax.

**Result:** native decode is coherent and faster, but byte-for-byte ORT CUDA
parity is intentionally absent at close MatMulNBits decisions.

## DeepSeek-Coder-1.3B (dense int4)

- Artifact: `~/glm-e2e-artifacts/deepseek-coder-1.3b-int4-cuda`
- Both backends load successfully.
- Both generate the same coherent Rust code, beginning with
  `fn add(a: i32, b: i32) -> i32 { a + b }`.
- Greedy token sequences match exactly for all 128 generated tokens.
- Native median: **798.44 tok/s**; ORT median: **623.51 tok/s**. Native is
  **1.281x** faster.
- ORT reported only its normal shape-op assignment warning; no `Memcpy`
  insertion warning or fallback-thrash signature appeared.

**Result:** coherent, exact greedy parity, and no native op/kernel gap found.

## Current gaps

1. Validate native QMoE on a full DeepSeek-V2 package with a real tokenizer and
   meaningful prompt; the current real-shape artifact cannot establish
   language coherence.
2. Produce or obtain an ORT CUDA QMoE reference that keeps the workload on GPU.
   The present artifact is suitable for numerical comparison but not speed.
3. Keep the DeepSeek-R1 MatMulNBits accuracy-level divergence documented and
   regression-tested; parity claims must distinguish oracle accuracy from
   byte-identical ORT CUDA output.

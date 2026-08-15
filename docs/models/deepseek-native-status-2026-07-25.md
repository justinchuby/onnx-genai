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
- This is consistent with an already characterized accuracy-level phenomenon,
  not evidence of a native regression. `deepseek_r1_1_5b_divergence.rs` locks a
  **different** prompt/divergence with an independent fp32 CPU oracle: with the
  chat-template prompt `"The capital of France is"`, native and ORT CUDA agree
  for seven tokens, then native selects the oracle-correct token `374` while ORT
  CUDA flips to `315`. That test proves native is *more accurate* than ORT CUDA
  at that decision.
- **Caveat (honest scope):** the benchmark divergence documented above (token 16,
  native `374` vs ORT CUDA `594`) is a *separate* run and is **not itself**
  fp32-oracle-adjudicated. The 0.0625 / 0.015625 numbers are each backend's own
  margin for its own choice — they show the two backends disagree, not that
  native is more accurate on this specific prompt. It is consistent with the
  locked accuracy-level phenomenon, but proving native-more-accurate here
  requires extending the fp32 oracle to this prompt (see gap 3).

**Result:** native decode is coherent and faster. Byte-for-byte ORT CUDA parity
is intentionally absent at close MatMulNBits decisions; native is proven more
accurate than ORT CUDA for the oracle-locked `"capital of France"` prompt, and
the benchmark-prompt divergence is consistent with — but not yet independently
adjudicated by — that same phenomenon.

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
   byte-identical ORT CUDA output. The committed `deepseek_r1_1_5b_divergence.rs`
   oracle covers only the `"capital of France"` prompt (token 8, native 374 vs
   ORT 315); **extend the fp32 oracle to the status-doc benchmark prompt** so the
   token-16 (374 vs 594) divergence is independently adjudicated rather than
   argued by analogy.

## Golden decode-lock follow-up — 2026-08-14

A committed tiny synthetic DeepSeek-V2-style fixture now locks the native path in
`deepseek_v2_tiny_qmoe_native_e2e.rs`. The fixture is deterministic and small
(`tests/fixtures/tiny-deepseek-v2-qmoe-attention`) and contains:

- two standard `ai.onnx::RotaryEmbedding` nodes for q/k RoPE,
- one standard `ai.onnx::Attention` node, and
- one sparse top-k integer `com.microsoft::QMoE` block (int4, `activation_type=swiglu`).

Attention-path finding: the real DeepSeek-V2-Lite int4 export does **not** use a
native custom MLA op. Mobius lowers the MLA-shaped block to standard ONNX
`RotaryEmbedding` + `Attention`; the native EP executes that standard Attention
path plus QMoE. The golden fixture intentionally mirrors that emitted path.

Golden stream: prompt token ids `[3]` greedily decode to
`[11, 11, 11, 11, 11, 11, 11, 11]`. The new lock asserts native CPU produces
that stream and native CUDA matches CPU. CUDA eager (`ONNX_GENAI_CUDA_GRAPH=0`)
passes. With graph capture requested (`ONNX_GENAI_CUDA_GRAPH=1`), this tiny
fixture currently declines capture for the documented capacity-awareness reason
(`attention_mask_consumers_are_capacity_aware`) because the int64 metadata mask
is cast to bool before Attention; the token stream still matches CPU exactly.

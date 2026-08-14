# GLM native CUDA decode status — 2026-07-25

Validated at `1160f321a3823d69cb0021a71421eb05d1055f86` on an NVIDIA H200.
Commands used the release `profile_native` binary, `CUDA_VISIBLE_DEVICES=0`,
and `taskset -c 1`. Throughput is steady-state greedy decode after eight
skipped tokens.

## Summary

| Model | Native CUDA | ORT CUDA | Greedy parity | Coherence |
|---|---:|---:|---|---|
| GLM-4-9B int4 CUDA | 108.78 tok/s | Cannot load | Not testable: ORT rejects partial-RoPE GQA | Coherent natural-language/math continuation |
| GLM-5.2 tiny QMoE | 176.66 tok/s | Cannot load | Not testable: ORT lacks `pkg.nxrt::IndexShare` | Structurally valid deterministic decimal-token stream |
| GLM-5.2 tiny q4 | Decodes (eager, no CUDA-graph capture) — see fix note below | 1,182.89 tok/s* | Token 0 matches (`110`); tiny random weights, full parity oracle N/A | Structurally valid deterministic decimal-token stream |

\* This tiny-model ORT number is GPU-resident but is not representative of a
real GLM checkpoint. A sustained 50-run probe reached 75% sampled GPU
utilization with 685 MiB resident and emitted no inserted-`Memcpy` warning.

## GLM-4-9B int4 CUDA

- Artifact: `~/glm-e2e-artifacts/glm-4-9b-int4-cuda`
- Native CUDA loads and generates coherent text. For prompt `"Hello"`, the
  continuation begins `", I am a 3rd year student at the University of
  Waterloo..."` and proceeds into a consistent linear-algebra question and
  characteristic-polynomial explanation.
- The first 64 generated IDs exactly match the committed
  `glm4_9b_decode_lock.rs` golden sequence.
- Native median: **108.78 tok/s** (3 runs, 2 warmups, 128 tokens).
- ORT CUDA cannot initialize the graph. Exact load error:

  ```text
  Error Unrecognized attribute: rotary_embedding_dim for operator GroupQueryAttention
  ```

  The rejected node is
  `model/layers.0/self_attn/GroupQueryAttention_node_19`
  (`com.microsoft::GroupQueryAttention`).
- Because ORT fails during model loading, there is no ORT token stream,
  log-probability comparison, GPU throughput baseline, or fp32-oracle
  adjudication to perform.

**Result:** native GLM-4-9B decode is coherent and operational, but it is
incorrect to claim that native beats ORT: the available ORT build cannot load
the model, so no valid native-vs-ORT performance comparison exists.

## GLM-5.2 tiny QMoE

- Artifact: `~/glm-e2e-artifacts/glm-5.2-tiny-qmoe`
- Native CUDA loads and executes the GLM DSA/`IndexShare` plus fused `QMoE`
  path without a NaN, crash, fallback, or unsupported-kernel error.
- With prompt token `123`, the stream begins
  `[62, 164, 59, 205, 48, 166, 27, 9, 221, 190, 123, 108]`, exactly matching
  the committed native CPU/CUDA anchor in
  `glm_tiny_qmoe_native_cuda_e2e.rs`.
- Native median: **176.66 tok/s** (3 runs, 2 warmups, 64 tokens).
- The decimal-token output is deterministic and structurally valid, but random
  tiny weights make semantic coherence inapplicable.
- ORT CUDA cannot load the model. Exact load error:

  ```text
  Fatal error: pkg.nxrt:IndexShare(-1) is not a registered function/op
  ```

- ORT therefore supplies neither a token/log-probability reference nor a valid
  throughput baseline for this artifact.

**Result:** the native GLM QMoE path runs successfully on the tiny conformance
model. Cross-runtime parity and performance remain unmeasured because the
artifact contains a native-only `pkg.nxrt::IndexShare` op.

## GLM-5.2 tiny dense q4

> **Update (fix `fix/glm52-decode-mask-capacity`):** this decode-token-1 broadcast
> failure is **resolved**. The single-token `attention_mask` exposure is now routed
> by *consumer class*: a mask binding whose value feeds a non-capacity-aware
> consumer (the indexer `Cast → Add` arithmetic, i.e. anything outside the
> `Shape`/`ReduceSum` padded-capacity allowlist) exposes its **logical** valid
> length instead of the frozen physical capacity (`max_len`). Because that mask
> then grows per step, such models decode **eagerly and forfeit CUDA-graph
> capture** on every decode step — the same trade-off the eager prefill path
> already makes. This is a correctness-over-throughput trade that applies **only**
> to indexer/logical-mask exports; capacity-safe masks (Qwen, DeepSeek, GLM-4)
> keep the frozen fast path and CUDA-graph capture, verified byte-identical to the
> pre-fix baseline. Native CUDA now decodes both `[123]` and `[1,2,3,4]` past
> token 1 (regression locked by `glm_tiny_quant_native_cuda_e2e.rs`). The failing
> behaviour below is the pre-fix state, retained for provenance.

- Artifact: `~/glm-e2e-artifacts/glm-5.2-tiny-q4`
- Native CUDA loads and emits the first greedy token `110`, but the next decode
  step fails. With prompt token `123`, the exact error is:

  ```text
  runtime broadcast shape resolution failed for node
  "model/layers.0/self_attn/indexer/Add_node_70" (ai.onnx::Add):
  concrete input shapes [[1, 1, 2], [1, 1, 4096]] are not
  broadcast-compatible
  ```

  A second probe with prompt tokens `[1, 2, 3, 4]` fails at the same node with
  shapes `[[1, 1, 5], [1, 1, 4096]]`, confirming a growing logical-prefix
  versus fixed-4096 mask/bias mismatch rather than prompt-token corruption.
- Native token-0 top-40 collection selects token `110` with log-probability
  `-5.536019`. ORT also selects token `110`, so token 0 agrees; no later token
  or log-probability parity is possible until the native shape failure is fixed.
- ORT CUDA completes 64-token decode. Sustained median: **1,182.89 tok/s**
  (50 runs, 2 warmups). Monitoring observed 75% GPU utilization and 685 MiB
  resident, with no `Memcpy` insertion warning, so this is a real CUDA
  execution rather than a CPU-fallback number.
- ORT's generated stream begins
  `[110, 104, 112, 161, 235, 189, 98, ...]`. Its decimal-token output is
  structurally valid but not semantically meaningful.
- This is a current regression relative to the historical 2026-07-23 report,
  which recorded native q4 end-to-end decode at 148.58 tok/s.

**Result:** dense GLM-5.2 q4 native decode is currently broken at the second
generated token. This artifact has two standard `ai.onnx::Attention` nodes and
no `pkg.nxrt::IndexShare` node: `indexer` here names the GLM DSA subgraph, not
the native IndexShare kernel. The native CUDA fixed-capacity decode state
deliberately exposes `attention_mask` at physical capacity (4096) on
single-token steps. That is valid for the ordinary attention bias, but this
export also feeds the mask through `Cast → Squeeze → Cast` into the indexer
score `Add`; its other operand retains the logical prefix. Thus this is a
native decode-time mask-binding/capacity-exposure bug, not an unsupported op,
kernel failure, numerical divergence, or export artifact. An accuracy-level
oracle is not applicable until the binding policy is fixed.

## Current gaps

1. ~~Restore native GLM-5.2 tiny-q4 multi-token decode by changing the CUDA
   single-token `attention_mask` exposure policy~~ **(DONE — fix
   `fix/glm52-decode-mask-capacity`).** The mask now retains its logical length
   whenever a mask-dependent, non-capacity-aware consumer (outside the
   `Shape`/`ReduceSum` allowlist) reaches the GLM indexer score path, rather than
   always exposing the fixed 4096 capacity; `Add_node_70` operands stay at the
   same logical prefix (`2`, `5`, ...). Implemented via
   `DeviceIoBinding::exposes_logical_input_shape` +
   `DecodeCudaState::decode_mask_expose_len` in `native_decode.rs`; such models
   decode eagerly (no CUDA-graph capture). The native-CUDA regression requiring
   more than one generated token, with both `[123]` and `[1,2,3,4]` prompts, is
   `glm_tiny_quant_native_cuda_e2e.rs`.
2. Obtain an ORT-compatible GLM-4 export/runtime supporting the
   `GroupQueryAttention.rotary_embedding_dim` partial-RoPE schema. Until then,
   GLM-4 parity, log-probability comparison, and native-vs-ORT speed claims are
   unavailable.
3. Produce an ORT-compatible GLM-5.2 QMoE reference without the native-only
   `pkg.nxrt::IndexShare` op, or register an equivalent ORT custom op, so greedy
   tokens/log-probabilities and GPU throughput can be compared.
4. Validate GLM-5.2 QMoE with real checkpoint weights and a real tokenizer; the
   current tiny random-weight artifact proves execution only, not language
   quality.

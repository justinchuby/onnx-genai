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
| GLM-5.2 tiny q4 | Decode fails after token 0 | 1,182.89 tok/s* | Token 0 matches (`110`); subsequent parity blocked | Not assessable on native because decode stops at token 1 |

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
generated token. The failure is an actionable shape-contract/runtime-binding
gap at `indexer/Add_node_70`, not a numerical divergence, so an accuracy-level
oracle is not applicable yet.

## Current gaps

1. Restore native GLM-5.2 tiny-q4 multi-token decode by resolving the growing
   logical-prefix (`2`, `5`, ...) versus fixed `4096` input mismatch at
   `model/layers.0/self_attn/indexer/Add_node_70`; add a native-CUDA regression
   that requires more than one generated token.
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

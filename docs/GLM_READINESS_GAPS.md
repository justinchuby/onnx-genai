# GLM-5.2 / DeepSeek-V4-Flash native-runtime readiness gaps

**Audit point:** `main` at `a7a1685`, 2026-07-17
**Scope:** native `onnx-runtime-session` with the in-tree CPU or CUDA execution
provider. This is not an ORT EP-fallback assessment.

## Executive conclusion

The native **CPU** runtime is close to a correctness-first GLM-5.2 decode:
the portable Mobius graph's required standard operators are registered, including
opset-23/24 `ai.onnx::Attention`, `RMSNormalization`, `RotaryEmbedding`,
`TopK`, `ScatterElements`, `Tile`, and quantized linear operators. The shortest
CPU path is therefore an **f32, Q4-requantized, portable-MoE export**, after the
Mobius export and cache-shape fixes land. That path is not smooth at GLM scale:
it evaluates all 256 experts and dense masked attention.

The native **CUDA** runtime cannot yet load either current portable export.
It has the expensive kernels (`MatMulNBits`, `BlockQuantizedMatMul`, `QMoE`,
contrib attention/GQA), but lacks the graph's standard `ai.onnx::Attention`,
`RotaryEmbedding`, `TopK`, `Where`, `CumSum`, `ScatterElements`, and most
movement/construction operators. A CUDA-only session has no CPU fallback
([session/lib.rs:111-115](../crates/onnx-runtime-session/src/lib.rs#L111-L115),
[557-585](../crates/onnx-runtime-session/src/lib.rs#L557-L585)).

For **smooth** GLM execution on either device, the main missing boundary is
`pkg.nxrt::BlockQuantizedMoE`: the runtime has lazy-weight/offload plumbing for
that name but no CPU or CUDA kernel. A selected-token IndexShare attention
boundary is also required to avoid dense million-token attention. DeepSeek's
native CSA path is CPU-only and still has an explicit equal-score top-k
unsupported case; CUDA CSA is absent. MTP graphs are representable, but package
discovery, HC-state threading, persistent proposer state, and composite rollback
are not implemented.

## What the exported graphs require

The operator set below is derived from the current Mobius PR heads:

- GLM-5.2 PR
  [#404 at `fb52e727`](https://github.com/onnxruntime/mobius/commit/fb52e7279c8a77bb1862f52a880d3743ca8e081e):
  standard opset 24, standard `Attention`, standard `RotaryEmbedding`,
  standard `RMSNormalization`, portable loop-over-256-experts MoE, IndexShare
  `TopK` converted to an additive mask, and an external MTP sidecar.
  The exporter design explicitly says the portable MoE is correct but
  impractical and selected-token attention is a runtime follow-up
  ([PROGRESS.md:383](PROGRESS.md#L383)).
- DeepSeek-V4-Flash PR
  [#405 at `7e26e6e`](https://github.com/onnxruntime/mobius/commit/7e26e6eb4e3a8839b311d59160ca947254afff4b):
  a dense sink-aware attention fallback, portable MoE, optional quantized
  embedding, HC tensors, and an external MTP sidecar. The learned CSA equations
  are not encoded in that graph; native-required packages need the private CSA
  boundary described in
  [DEEPSEEK_CSA_MTP_RUNTIME.md:18-35](DEEPSEEK_CSA_MTP_RUNTIME.md#L18-L35).
- Mixed native quantization PR
  [#406 at `797fff9`](https://github.com/onnxruntime/mobius/commit/797fff946c3db04a424a2667285a5e5becb929c5):
  `MatMulNBits`, optional `GatherBlockQuantized`, and native block formats.
  **Current-head warning:** its `_quantized_linear.py` still emits
  `com.github.onnxruntime.genai::BlockQuantizedMatMul`, while this repository
  registers only `pkg.nxrt` after the domain rename
  ([PROGRESS.md:27](PROGRESS.md#L27)). The PR must be refreshed or merged with
  the advertised `pkg.nxrt` fix; otherwise native dispatch fails by domain.

No special ONNX "MTP op" exists. The sidecar is another decoder graph. Its
remaining gaps are runtime orchestration/state gaps, not graph arithmetic ops.

### Citation keys used in the matrices

| Key | Authoritative registration/catalogue |
|---|---|
| C-L | `crates/onnx-runtime-ep-cpu/src/kernels/mod.rs:215-230` |
| C-A | `crates/onnx-runtime-ep-cpu/src/kernels/mod.rs:232-385` |
| C-E | `crates/onnx-runtime-ep-cpu/src/kernels/mod.rs:393-465` |
| C-U | `crates/onnx-runtime-ep-cpu/src/kernels/mod.rs:536-614` |
| C-R | `crates/onnx-runtime-ep-cpu/src/kernels/mod.rs:615-650` |
| C-M | `crates/onnx-runtime-ep-cpu/src/kernels/mod.rs:653-714` |
| G-L | `crates/onnx-runtime-ep-cuda/src/kernels/mod.rs:163-205` |
| G-E | `crates/onnx-runtime-ep-cuda/src/kernels/mod.rs:244-299` |
| G-A | `crates/onnx-runtime-ep-cuda/src/kernels/mod.rs:301-359` |
| G-P | `crates/onnx-runtime-ep-cuda/src/kernels/mod.rs:375-476` |
| S-E | `crates/onnx-runtime-shape-inference/src/handlers/elementwise.rs:171-216` |
| S-L | `crates/onnx-runtime-shape-inference/src/handlers/linalg.rs:423-447` |
| S-N | `crates/onnx-runtime-shape-inference/src/handlers/norm.rs:269-323` |
| S-M | `crates/onnx-runtime-shape-inference/src/handlers/movement.rs:1555-1589` |
| S-D | `crates/onnx-runtime-shape-inference/src/handlers/data_ops.rs:231-236` |
| S-Q | `crates/onnx-runtime-shape-inference/src/handlers/selection.rs:235-239` |
| S-S | `crates/onnx-runtime-shape-inference/src/handlers/sequence.rs:202-205` |
| R | `crates/onnx-rs/src/schema/mod.rs:345-424` |

`✅` means a matching registration exists, not that every dtype/attribute is
supported. `◐` means the op is registered but the required graph usage exceeds
the kernel contract. `❌` means no matching `(domain, op, opset)` registration.

## Required standard-op coverage

The union includes the GLM target/MTP graph, DeepSeek dense target/MTP graph,
their shared causal-mask builder, and the portable sparse-mixer routing
decomposition. Standard-domain graphs are opset 24 (registrations at an earlier
compatible since-version satisfy the lookup).

| Required op (`ai.onnx`) | Used by | CPU EP | CUDA EP | Shape inference | onnx-rs |
|---|---|---:|---:|---:|---:|
| `Abs` | sparse_mixer | ✅ C-U | ✅ G-P | ✅ S-E | ✅ R |
| `Add` | both/MTP/MoE | ✅ C-A | ✅ G-E | ✅ S-E | ✅ R |
| `And` | causal mask | ✅ C-U | ◐ G-P | ✅ S-E | ✅ R |
| `Attention` (23/24) | GLM DSA/dense | ✅ C-A | ❌ MISSING | ✅ S-L | ❌ MISSING |
| `Cast` | both | ✅ C-E | ✅ G-P | ✅ S-D | ✅ R |
| `CastLike` | portable MoE/HC | ✅ C-E | ✅ G-P | ✅ S-D | ❌ MISSING |
| `Clip` | DeepSeek SwiGLU | ✅ C-M | ✅ G-E | ✅ S-E | ✅ R |
| `Concat` | both/cache/HC | ✅ C-M | ❌ MISSING | ✅ S-M | ✅ R |
| `Constant` | both | ✅ C-E | ✅ G-L | ✅ S-D | ❌ MISSING |
| `CumSum` | causal mask | ✅ C-M | ❌ MISSING | ✅ S-S | ❌ MISSING |
| `Div` | routing/HC | ✅ C-E | ✅ G-E | ✅ S-E | ✅ R |
| `Equal` | portable expert mask | ✅ C-U | ◐ G-P | ✅ S-E | ✅ R |
| `Expand` | both/HC | ✅ C-E | ❌ MISSING | ✅ S-M | ✅ R |
| `Gather` | embedding/RoPE | ✅ C-A | ✅ G-L | ✅ S-M | ✅ R |
| `GatherElements` | DeepSeek routing/sparse_mixer | ✅ C-M | ❌ MISSING | ✅ S-M | ✅ R |
| `Greater` | sparse_mixer | ✅ C-U | ◐ G-P | ✅ S-E | ✅ R |
| `GreaterOrEqual` | causal mask | ✅ C-U | ◐ G-P | ✅ S-E | ❌ MISSING |
| `LayerNormalization` | GLM IndexShare key projection | ✅ C-A | ✅ G-A | ✅ S-N | ✅ R |
| `MatMul` | projections/router/attention | ✅ C-L | ✅ G-L | ✅ S-L | ✅ R |
| `Max` | sparse_mixer | ✅ C-E | ✅ G-E | ✅ S-E | ❌ MISSING |
| `Min` | GLM dynamic top-k | ✅ C-E | ✅ G-E | ✅ S-E | ❌ MISSING |
| `Mul` | both | ✅ C-E | ✅ G-E | ✅ S-E | ✅ R |
| `Neg` | DeepSeek compressed RoPE | ✅ C-U | ✅ G-P | ✅ S-E | ✅ R |
| `ReduceMax` | DeepSeek grouped routing | ✅ C-R | ✅ G-P | ✅ S-N | ✅ R |
| `ReduceMean` | DeepSeek HC/norm | ✅ C-E | ✅ G-P | ✅ S-N | ✅ R |
| `ReduceSum` | both/router/HC | ✅ C-R | ✅ G-P | ✅ S-N | ✅ R |
| `Relu` | GLM indexer | ✅ C-A | ✅ G-E | ✅ S-E | ✅ R |
| `Reshape` | both | ✅ C-A | ❌ MISSING | ✅ S-M | ✅ R |
| `RMSNormalization` (23/24) | both/MTP | ✅ C-A | ✅ G-A | ✅ S-N | ✅ R |
| `RotaryEmbedding` (23/24) | both | ✅ C-A | ❌ MISSING | ✅ S-N | ❌ MISSING |
| `ScatterElements` | GLM DSA mask/sparse_mixer | ✅ C-M | ❌ MISSING | ✅ S-M | ✅ R |
| `Shape` | both/dynamic top-k | ✅ C-E | ✅ G-L | ✅ S-D | ✅ R |
| `Sigmoid` | router/SwiGLU/HC | ✅ C-U | ✅ G-E | ✅ S-E | ✅ R |
| `Slice` | masks/cache/DeepSeek sink | ✅ C-E | ❌ MISSING | ✅ S-M | ✅ R |
| `Softmax` | attention/routing/HC | ✅ C-A | ✅ G-A | ✅ S-N | ✅ R |
| `Softplus` | DeepSeek router | ✅ C-U | ✅ G-P | ✅ S-E | ❌ MISSING |
| `Split` | Q/K/V, HC | ✅ C-E | ❌ MISSING | ✅ S-M | ✅ R |
| `Sqrt` | DeepSeek norm/router | ✅ C-E | ✅ G-E | ✅ S-E | ✅ R |
| `Squeeze` | GLM indexer/cache | ✅ C-M | ❌ MISSING | ✅ S-M | ❌ MISSING |
| `Sub` | causal mask | ✅ C-E | ✅ G-E | ✅ S-E | ✅ R |
| `Tile` | GLM repeated RoPE key | ✅ C-M | ❌ MISSING | ✅ S-S | ✅ R |
| `TopK` | both routers/GLM indexer | ✅ C-M | ❌ MISSING | ✅ S-Q | ❌ MISSING |
| `Transpose` | both | ✅ C-A | ❌ MISSING | ✅ S-M | ✅ R |
| `Unsqueeze` | both/masks/HC | ✅ C-E | ❌ MISSING | ✅ S-M | ❌ MISSING |
| `Where` | causal mask/sparse_mixer | ✅ C-U | ❌ MISSING | ✅ S-E | ✅ R |

CUDA's `And`/comparison kernels are equal-shape-only
([ep-cuda/kernels/mod.rs:446-468](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L446-L468)).
The exported causal mask relies on broadcasting
`[B,S,1]` with `[B,1,T]`, and portable expert masking compares selected-expert
tensors with scalars. Those rows are therefore not usable merely because a
factory is registered.

## Required contrib/private/native-boundary coverage

| Required op / facility | Domain, version | Why required | CPU EP | CUDA EP | Shape inference | onnx-rs |
|---|---|---|---:|---:|---:|---:|
| `MatMulNBits` | `com.microsoft`, 1 | affine quantized projections and Q4 fallback | ✅ C-L | ✅ G-L | ✅ S-L | ❌ R |
| `BlockQuantizedMatMul` | `pkg.nxrt`, 1 | preserved IQ/MXFP4 dense/shared projections | ✅ C-L | ✅ G-L | ✅ S-L | ❌ R |
| `GatherBlockQuantized` | `com.microsoft`, 1 | optional quantized DeepSeek embedding | ❌ MISSING | ❌ MISSING | ❌ MISSING | ❌ R |
| `GroupQueryAttention` | `com.microsoft`, 1 | alternative fused decoder export; known cache-shape work | ✅ C-A | ✅ G-A | ✅ S-N | ❌ R |
| `MoE` | `com.microsoft`, 1 | possible fused float expert path | ✅ C-A | ❌ MISSING | ❌ MISSING | ❌ R |
| `QMoE` | `com.microsoft`, 1 | possible fused affine-int expert path | ✅ C-A | ✅ G-L | ❌ MISSING | ❌ R |
| `SparseKvGather` | `pkg.nxrt`, 1 | CSA correctness/debug primitive | ✅ C-L | ❌ MISSING | ❌ MISSING | ❌ R |
| `CompressedSparseAttention` | `pkg.nxrt`, 1 | native DeepSeek ratio-4/128 CSA | ✅ C-L | ❌ MISSING | ❌ MISSING | ❌ R |
| `BlockQuantizedMoE` | `pkg.nxrt`, proposed v1 | selected IQ1/IQ2/IQ3/IQ4/MXFP4 experts + lazy leases | ❌ MISSING | ❌ MISSING | ❌ MISSING | ❌ R |
| selected-token IndexShare DSA | private contract not frozen | avoid dense attention over masked K/V positions | ❌ MISSING | ❌ MISSING | ❌ MISSING | ❌ R |
| MTP package/state orchestration | runtime facility, not an op | load sidecar, HC state, persistent draft, rollback | ❌ incomplete | ❌ incomplete | N/A | N/A |

### Important kernel-contract qualifications

1. **f32 is the shortest native profile.** CPU and CUDA `MatMulNBits` require
   f32 activation/scales/output
   ([CPU: matmul_nbits.rs:102-107](../crates/onnx-runtime-ep-cpu/src/kernels/matmul_nbits.rs#L102-L107),
   [CUDA: matmul_nbits.rs:323-328](../crates/onnx-runtime-ep-cuda/src/kernels/matmul_nbits.rs#L323-L328)).
   CPU standard `Attention` and `RotaryEmbedding` are f32 reference kernels
   ([attention.rs:51-54](../crates/onnx-runtime-ep-cpu/src/kernels/attention.rs#L51-L54),
   [rotary_embedding.rs:73-79](../crates/onnx-runtime-ep-cpu/src/kernels/rotary_embedding.rs#L73-L79)).
   A BF16-native export needs dtype extensions or explicit casts.
   CPU `MatMulNBits` supports bits 2/4, while CUDA currently supports only
   bits 4
   ([CPU: matmul_nbits.rs:56-60](../crates/onnx-runtime-ep-cpu/src/kernels/matmul_nbits.rs#L56-L60),
   [CUDA: matmul_nbits.rs:271-278](../crates/onnx-runtime-ep-cuda/src/kernels/matmul_nbits.rs#L271-L278)).
2. **QMoE is not native IQ/MXFP4 MoE.** Both providers accept only affine
   integer `quant_type="int"`. CPU is f32-only and both reject
   `use_sparse_mixer=1`
   ([CPU qmoe.rs:64-88](../crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs#L64-L88),
   [moe.rs:81-85](../crates/onnx-runtime-ep-cpu/src/kernels/moe.rs#L81-L85),
   [CUDA qmoe.rs:694-721](../crates/onnx-runtime-ep-cuda/src/kernels/qmoe.rs#L694-L721)).
3. **CUDA block matmul is format-complete but f32-only.** It recognizes the ten
   current MXFP4/IQ formats
   ([block_quantized_matmul.rs:474-519](../crates/onnx-runtime-ep-cuda/src/kernels/block_quantized_matmul.rs#L474-L519))
   but requires f32 `A`, bias, and `Y`
   ([585-600](../crates/onnx-runtime-ep-cuda/src/kernels/block_quantized_matmul.rs#L585-L600)).
4. **CPU CSA is not unconditional.** Its frozen boundary implements persistent
   ratio-4/128 state, but equal-score ratio-4 top-k remains explicitly
   unsupported because tie ordering is unfrozen
   ([compressed_sparse_attention.rs:1-8](../crates/onnx-runtime-ep-cpu/src/kernels/compressed_sparse_attention.rs#L1-L8),
   [1581-1586](../crates/onnx-runtime-ep-cpu/src/kernels/compressed_sparse_attention.rs#L1581-L1586)).
5. **Shape inference is on the runtime load path.** The loader and optimizer
   invoke the registry permissively
   ([loader/lib.rs:374-378](../crates/onnx-runtime-loader/src/lib.rs#L374-L378),
   [session/executor.rs:758-763](../crates/onnx-runtime-session/src/executor.rs#L758-L763)).
   Missing custom handlers are conditional blockers when the exporter has not
   stamped every output, especially multi-output/stateful ops.
6. **onnx-rs schemas do not gate the executor today.** onnx-rs is the separate
   standard-library/checker layer, but its missing schemas prevent authoritative
   validation, inference, and version tooling for these packages
   ([onnx-rs/src/lib.rs:7-18](../crates/onnx-rs/src/lib.rs#L7-L18)).

## Prioritized gap list

### P0 — blocks native correctness

1. **CUDA graph-op completeness for the actual exports — CUDA correctness, L.**
   Add opset-23/24 standard `Attention` (not the currently registered
   `com.microsoft::Attention`), `RotaryEmbedding`, `TopK`, `Where`, `CumSum`,
   `ScatterElements`, `GatherElements`, and the missing construction/movement
   family (`Concat`, `Expand`, `Reshape`, `Slice`, `Split`, `Squeeze`, `Tile`,
   `Transpose`, `Unsqueeze`). Add broadcasting to CUDA logical/comparison
   kernels. The registry ends without these handlers
   ([ep-cuda/kernels/mod.rs:301-478](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L301-L478));
   unsupported nodes are rejected by exact domain/opset lookup
   ([provider.rs:113-146](../crates/onnx-runtime-ep-cuda/src/provider.rs#L113-L146)).

2. **Refresh Mobius #406's private domain before consuming it — CPU+CUDA
   correctness for native IQ/MXFP4, S (Mobius), awaiting user merge.**
   Current PR-head source emits the removed
   `com.github.onnxruntime.genai` domain, while both native registries claim
   `pkg.nxrt` only (C-L/G-L). No in-repo compatibility alias exists.

3. **`GatherBlockQuantized` native coverage — conditional CPU+CUDA correctness,
   M per EP; shape/schema S.**
   DeepSeek's exporter emits this when `quantize_embeddings=true`; all four
   audited registries are missing it. The correctness workaround is an
   unquantized standard `Gather`, but at GLM/DeepSeek vocabulary sizes that
   materially increases resident memory.

4. **Custom-op shape handlers — conditional load/compile correctness, S-M.**
   Add handlers for `MoE`, `QMoE`, `GatherBlockQuantized`, `SparseKvGather`,
   and `CompressedSparseAttention`. Multi-output CSA needs the frozen state
   shapes, not merely output-0 passthrough. `BlockQuantizedMoE` needs a handler
   when its ABI is implemented.

5. **MTP metadata and HC/state adapter — GLM/DeepSeek MTP correctness, L.**
   Metadata discovery still labels MTP `NotYetSupported`
   ([metadata/parser.rs:76-85](../crates/onnx-genai-metadata/src/parser.rs#L76-L85));
   target hidden extraction accepts only ranks 1-3
   ([engine/decode.rs:1215-1249](../crates/onnx-genai-engine/src/decode.rs#L1215-L1249));
   the MTP session binds both inputs as `[B,S,H]`
   ([onnx-genai-ort/src/mtp.rs:211-240](../crates/onnx-genai-ort/src/mtp.rs#L211-L240));
   and the proposer is rebuilt per verification iteration
   ([engine/speculative.rs:965-980](../crates/onnx-genai-engine/src/speculative.rs#L965-L980)).
   Implement package references, rank-4 `[B,S,hc_mult,H]`, explicit recurrent
   state, persistent per-generation ownership, and atomic target/CSA/MTP rewind.

### P1 — correctness for a native feature, or required for smooth execution

6. **`pkg.nxrt::BlockQuantizedMoE` CPU and CUDA kernels — smooth GLM on both,
   L.**
   This is the largest practical blocker. GLM's routed tensors are dominated by
   IQ1_M/IQ2_XXS/IQ3_XXS; portable export evaluates every expert, and QMoE
   cannot represent codebook formats. The lazy initializer/offload seam already
   recognizes only this boundary
   ([ep-api/weight.rs:94-104](../crates/onnx-runtime-ep-api/src/weight.rs#L94-L104)),
   but neither EP registers a factory. Implement selected-expert dispatch,
   fused gate/up/activation/down, native block decoders, and lazy selected-slice
   binding.

7. **Selected-token IndexShare DSA operator/contract — GLM perf+memory, L.**
   The current additive-mask `Attention` path is correctness-preserving, so this
   is not a first-token blocker. It still computes over every cached token and
   is not viable at the intended 1,048,576-token context. Freeze a private ABI
   sharing cache/index primitives with CSA where semantics overlap, then add CPU
   and CUDA kernels. Do not conflate GLM IndexShare selection semantics with
   DeepSeek CSA.

8. **DeepSeek CSA CUDA plus full state integration — native-required DeepSeek
   CUDA correctness, L; CPU hardening M.**
   CUDA has neither `SparseKvGather` nor `CompressedSparseAttention`. CPU has
   the kernels, but package export/state cursor integration and the top-k tie
   rule remain. A `native_csa_required` package must fail rather than silently
   use dense fallback, per the approved design
   ([DEEPSEEK_CSA_MTP_RUNTIME.md:984-993](DEEPSEEK_CSA_MTP_RUNTIME.md#L984-L993)).

9. **Native sparse-mixer routing — sparse_mixer feature correctness, M.**
   Portable sparse-mixer arithmetic is CPU-representable, but native MoE/QMoE
   explicitly rejects `use_sparse_mixer=1`; CUDA additionally lacks several
   portable routing ops and broadcast semantics. Implement the frozen routing
   equation in MoE/QMoE/BlockQuantizedMoE or keep an explicit portable gate
   outside the fused expert op.

10. **Complete onnx-rs schemas — validation/tooling, M.**
    Add the missing standard schemas (`Attention`, `CastLike`, `Constant`,
    `CumSum`, `GreaterOrEqual`, `Min`, `Max`, `RotaryEmbedding`, `Softplus`,
    `Squeeze`, `TopK`, `Unsqueeze`) and contrib/private schemas. This does not
    currently block executor dispatch, but it blocks using onnx-rs as the
    authoritative checker/inference/version layer for the target packages.

### P2 — does not block first correct decode, but blocks the north-star quality

11. **Wire the activation liveness planner — CPU+CUDA peak memory, M.**
    The executor still owns one `DeviceBuffer` per value
    ([executor.rs:261-267](../crates/onnx-runtime-session/src/executor.rs#L261-L267)).
    `onnx-runtime-memory` already implements the plan but explicitly defers
    executor integration
    ([memory/lib.rs:12-23](../crates/onnx-runtime-memory/src/lib.rs#L12-L23),
    [65-89](../crates/onnx-runtime-memory/src/lib.rs#L65-L89)).
    GLM's enormous portable expert graph makes this especially important.

12. **CUDA graph-capture compatibility — CUDA decode performance, M-L.**
    Even after op coverage, `MatMul`, `Gemm`, reductions, and bounds-validating
    `Gather` report non-capturable behavior because of per-call allocation or
    host synchronization
    ([ep-cuda/capture.rs:5-10](../crates/onnx-runtime-ep-cuda/src/capture.rs#L5-L10),
    [matmul.rs:280-283](../crates/onnx-runtime-ep-cuda/src/kernels/matmul.rs#L280-L283),
    [gather.rs:226-229](../crates/onnx-runtime-ep-cuda/src/kernels/gather.rs#L226-L229)).
    Pool workspaces, remove capture-time host sync, and keep CSA/MTP state at
    stable addresses.

13. **Close the known numerical/cache review gates — validation, S-M.**
    The `present.0.key` GQA/cache-shape fix is in review. The token-16
    MatMulNBits CUDA drift is in revision; it is not an op-coverage blocker but
    must be resolved or accepted with an explicit tolerance before claiming
    GLM/DeepSeek CUDA parity.

## Shortest path to native end-to-end GLM decode

### CPU — correctness-first

1. **Await/merge Mobius #404 and the cache-shape fix.**
2. **Refresh/merge #406 with `pkg.nxrt`, not the removed domain.**
3. **Export the f32 portability profile:** standard unquantized embedding,
   affine Q4 `MatMulNBits` for routed experts (requantize IQ sources), standard
   `RMSNormalization`/`RotaryEmbedding`/`Attention`, and portable loop-over-
   experts MoE. Do not require `GatherBlockQuantized`, CSA, MTP, or native IQ
   expert execution for the first run.
4. **Run target-only greedy decode and freeze tokens/logits.** The in-tree CPU
   registry and shape handlers cover this graph. Any failure here is then a
   kernel-contract bug, not a known missing registration.
5. **Add MTP only after target-only decode is stable**, because MTP requires the
   separate metadata/HC/persistent-state work.

This reaches a correct native CPU token fastest, but it is not a smooth
GLM-5.2 implementation. The ordered smoothness work is:
`BlockQuantizedMoE` → selected-token IndexShare DSA → liveness planner →
MTP orchestration.

### CUDA — correctness-first

1. Complete the same Mobius merge/domain/cache-shape prerequisites.
2. Add the missing CUDA graph core, starting with:
   1. standard opset-24 `Attention`;
   2. `RotaryEmbedding`;
   3. `TopK`, `Where`, `ScatterElements`, `CumSum`;
   4. broadcast-capable logical/comparison;
   5. `Concat`, `Expand`, `Reshape`, `Slice`, `Split`, `Squeeze`, `Tile`,
      `Transpose`, `Unsqueeze`, and `GatherElements`.
3. Export/run an all-f32 CUDA graph with affine Q4 `MatMulNBits` and portable
   MoE. There is no heterogeneous fallback; every node must be claimed.
4. Resolve the token-16 MatMulNBits parity review and freeze end-to-end greedy
   tokens.
5. Replace portable experts with CUDA `BlockQuantizedMoE`, then add selected-
   token DSA. Only after graph correctness, stabilize allocations/state for CUDA
   graph capture.

For DeepSeek native CUDA, append
`CompressedSparseAttention`/`SparseKvGather` and state-journal support before
calling the path CSA-native. The dense fallback can validate graph arithmetic,
but it does not validate the native CSA contract.

## Awaiting user/upstream versus actionable here

### Awaiting user/upstream

- Merge/resolve Mobius PRs
  [#404](https://github.com/onnxruntime/mobius/pull/404),
  [#405](https://github.com/onnxruntime/mobius/pull/405), and
  [#406](https://github.com/onnxruntime/mobius/pull/406)
  ([PROGRESS.md:383](PROGRESS.md#L383)).
- Ensure #406's current-head old-domain emission is replaced by `pkg.nxrt`.
- Land the in-review `present.0.key` shape fix and resolve the in-revision
  token-16 CUDA MatMulNBits drift.
- Freeze/confirm any MTP recurrence and acceptance semantics that are not
  established by the official DeepSeek reference. The repository design
  correctly forbids inventing recurrent HC state from tensor names.

### Actionable in this repository now

1. CUDA standard-op and broadcast coverage for the actual graph.
2. CPU/CUDA `BlockQuantizedMoE`.
3. Native `GatherBlockQuantized`.
4. Shape handlers and onnx-rs schemas.
5. MTP package/HC/persistent-state/rollback plumbing.
6. CUDA CSA and GLM selected-token DSA.
7. Executor integration of the existing activation liveness planner.

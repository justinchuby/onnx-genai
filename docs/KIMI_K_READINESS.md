# Kimi K-series native-runtime architecture readiness

**Audit point:** `main` at `5dcd075`, 2026-07-17
**Scope:** native `onnx-runtime-session` with the in-tree CPU or CUDA execution
provider. This is an architecture-readiness analysis, not a reproduction of an
unpublished Moonshot specification.

> **Companion:** [MOE_EXPERT_PARALLELISM.md](./MOE_EXPERT_PARALLELISM.md) covers the
> multi-GPU/multi-node *deployment* side of K3-class MoE (session-per-GPU expert
> parallelism, control/data-plane split, `MoeDispatch`/`MoeGather` NCCL ops,
> distributed KV cache). This note covers the complementary *op/kernel-coverage*
> side: which operators, attention mechanisms, quant formats, and state seams the
> native runtime must implement to load and run Kimi K-series at all.

## Sourcing boundary: verified versus extrapolated

The released Kimi K2 architecture is public. Moonshot's official repository and
model card describe a 1T-parameter, 32B-activated MoE with 384 routed experts,
eight selected experts, one shared expert, MLA, and a 128K context window
([official repository](https://github.com/MoonshotAI/Kimi-K2),
[official model card](https://huggingface.co/moonshotai/Kimi-K2-Base)).
The published K2 config additionally identifies the implementation as
`DeepseekV3ForCausalLM` and records `kv_lora_rank=512`,
`q_lora_rank=1536`, a 128-dimensional non-RoPE query/key component, a
64-dimensional RoPE component, YaRN scaling, and no next-token-prediction
layers
([official K2 config](https://huggingface.co/moonshotai/Kimi-K2-Base/raw/main/config.json)).
The K2 technical report is also public
([Kimi K2 report](https://arxiv.org/abs/2507.20534)).

As of this audit, Kimi K3 has been **announced**, but its weights and full
technical report have not yet been released. Moonshot verifies 2.8T parameters,
a 1M-token context, native vision, Kimi Delta Attention (KDA), Attention
Residuals (AttnRes), Stable LatentMoE with 16 of 896 experts active, Gated MLA,
and MXFP4-weight/MXFP8-activation quantization-aware training. The announcement
says the weights and more detailed report are due by July 27, 2026
([official K3 announcement](https://www.kimi.com/blog/kimi-k3),
[official K3 API documentation](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart)).
Those are verified **announcement-level** facts; exact tensor layouts, operator
contracts, shared-expert count, cache ABI, routing equations, and checkpoint
format remain unverified until the artifacts arrive.

For the likely KDA execution shape, this analysis uses Moonshot's separately
released Kimi Linear reference: a 3:1 KDA-to-global-MLA hybrid with finite-state
recurrent KDA, short convolution, and a 1M context
([official Kimi Linear repository](https://github.com/MoonshotAI/Kimi-Linear),
[official Kimi Linear config](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base/raw/main/config.json),
[Kimi Linear paper](https://arxiv.org/abs/2510.26692)).
That is a reasoned readiness proxy, **not confirmation that K3 uses the same
ratio, dimensions, state layout, or kernel ABI**. The public KDA kernel exposes
an initial/final recurrent state shaped by value heads and key/value dimensions
([public KDA recurrent kernel](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/fused_recurrent.py)).

## Verified Kimi K2 architecture

| Property | Verified K2 value | Runtime consequence |
|---|---|---|
| Scale | 1T total parameters; 32B activated per token ([Moonshot](https://github.com/MoonshotAI/Kimi-K2)) | Whole-model residency is unrealistic on ordinary machines; sparse expert paging matters. |
| MoE | 384 routed experts, top-8, plus one shared expert; SwiGLU ([Moonshot](https://huggingface.co/moonshotai/Kimi-K2-Base)) | Preserve router semantics, selected-expert dispatch, shared dense path, and expert-major weights. |
| Attention | MLA with low-rank Q/KV projections and split non-RoPE/RoPE dimensions ([K2 config](https://huggingface.co/moonshotai/Kimi-K2-Base/raw/main/config.json)); MLA's compressed latent cache and decoupled RoPE design are documented by the DeepSeek-V3 lineage ([DeepSeek-V3 report](https://arxiv.org/abs/2412.19437)) | GQA alone is not sufficient: the runtime must retain a latent KV state and apply model projections around attention without expanding a full conventional KV cache. |
| Context | 128K ([Moonshot](https://huggingface.co/moonshotai/Kimi-K2-Base)) | Cache representation and offload dominate memory; dense reconstructed K/V is only a correctness fallback. |
| Released precision | The published checkpoint config declares FP8 quantization with E4M3 format, 128×128 weight blocks, and dynamic activation quantization ([K2 config](https://huggingface.co/moonshotai/Kimi-K2-Base/raw/main/config.json)) | Exact FP8 execution is not covered by current native linear/MoE kernels; local sub-4-bit packages require an explicitly converted format. |
| MTP | The published config has `num_nextn_predict_layers=0` ([K2 config](https://huggingface.co/moonshotai/Kimi-K2-Base/raw/main/config.json)) | K2 does not currently require an MTP sidecar. |

### Why MLA is not GQA

`GroupQueryAttention` reduces the number of **full K/V heads** and stores
ordinary K/V cache tensors. The in-tree CPU and CUDA kernels explicitly use
`num_heads` and `kv_num_heads`, build/preserve `[B, kv_heads, S, D]` caches, and
apply ordinary RoPE
([CPU GQA](../crates/onnx-runtime-ep-cpu/src/kernels/group_query_attention.rs),
[CUDA GQA](../crates/onnx-runtime-ep-cuda/src/kernels/group_query_attention.rs)).

MLA instead stores a low-rank KV latent, reconstructs or algebraically absorbs
per-head K/V projections, and carries a separate RoPE key component. K2's
published `kv_lora_rank`, `qk_nope_head_dim`, and `qk_rope_head_dim` make that
distinction concrete
([K2 config](https://huggingface.co/moonshotai/Kimi-K2-Base/raw/main/config.json)).
Therefore:

- GQA can serve only an **expanded-K/V fallback** after MLA projection.
- CPU `RotaryEmbedding` can rotate only the designated tail when the exporter
  supplies precomputed K2/YaRN cos/sin tables
  ([rotary_embedding.rs:18-31](../crates/onnx-runtime-ep-cpu/src/kernels/rotary_embedding.rs#L18-L31)).
- `pkg.nxrt::CompressedSparseAttention` is reusable evidence that the EP can own
  compressed persistent state, but it is a frozen temporal ratio-4/128 sparse
  attention contract, not MLA's learned low-rank KV factorization
  ([compressed_sparse_attention.rs:1-8](../crates/onnx-runtime-ep-cpu/src/kernels/compressed_sparse_attention.rs#L1-L8),
  [163-185](../crates/onnx-runtime-ep-cpu/src/kernels/compressed_sparse_attention.rs#L163-L185)).

**Conclusion:** efficient native MLA is a real gap. The CPU graph can express
the surrounding projection/RoPE/attention arithmetic as a slow expanded-cache
reference, but neither GQA nor CSA implements K2 MLA semantics.

## K3: verified announcement facts and unconfirmed implementation assumptions

### Verified announcement-level characteristics

- K3 is a 2.8T-parameter model with a 1M-token context and native vision
  ([official announcement](https://www.kimi.com/blog/kimi-k3)).
- It uses KDA, AttnRes, Gated MLA, and Stable LatentMoE, activating 16 of 896
  experts
  ([official announcement](https://www.kimi.com/blog/kimi-k3)).
- It applies quantization-aware training with MXFP4 weights and MXFP8
  activations
  ([official announcement](https://www.kimi.com/blog/kimi-k3)).

### Anticipated runtime characteristics — unconfirmed for K3

1. **Hybrid recurrent/global attention state.** Kimi Linear suggests KDA layers
   carry a fixed-size recurrent matrix state while periodic global layers use
   MLA
   ([official Kimi Linear repository](https://github.com/MoonshotAI/Kimi-Linear)).
   K3's exact layer ratio and state shape are not yet public.
2. **A KDA-specific prefix-cache transform.** Moonshot says KDA requires new
   prefix-caching work, but has not yet published K3's cache ABI
   ([official K3 announcement](https://www.kimi.com/blog/kimi-k3)).
3. **Inference-visible Stable LatentMoE routing may require a new router
   contract.** Quantile Balancing is announced, but the report needed to
   distinguish training-only balancing from inference-time selection is not
   available
   ([official K3 announcement](https://www.kimi.com/blog/kimi-k3)).
4. **No verified K3 MTP requirement.** The K3 announcement discusses always-on
   reasoning, not an MTP sidecar
   ([official K3 API documentation](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart)).
   Do not infer MTP merely from DeepSeek lineage.

## Feature-to-runtime coverage matrix

`✅` means the required semantics are implemented in the audited native path.
`◐` means pieces or a correctness fallback exist, but not the efficient/model-
exact feature. `❌` means no matching native operator/state contract exists.

| Kimi feature | CPU EP | CUDA EP | Shape/IR/runtime | Assessment |
|---|---:|---:|---:|---|
| **K2 MLA: low-rank latent KV + decoupled RoPE** | ◐ | ❌ | ◐ | CPU has `MatMul`, partial-dimension `RotaryEmbedding`, standard `Attention`, and GQA registrations ([CPU registry:215-230, 280-299, 382-385](../crates/onnx-runtime-ep-cpu/src/kernels/mod.rs#L215-L230)); a decomposed expanded-K/V fallback is plausible. CUDA has contrib GQA/Attention but no standard `RotaryEmbedding` and no MLA kernel ([CUDA registry:301-313](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L301-L313)). Metadata recognizes `multi_latent`, but there is no native MLA cache adapter ([schema.rs:225-247](../crates/onnx-genai-metadata/src/schema.rs#L225-L247)). |
| **GQA/MQA fallback** | ✅ | ✅ | ✅ | Both EPs register `com.microsoft::GroupQueryAttention`; shape inference models its ordinary fixed-capacity K/V cache ([CPU:279-282](../crates/onnx-runtime-ep-cpu/src/kernels/mod.rs#L279-L282), [CUDA:301-313](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L301-L313), [shape:105-188](../crates/onnx-runtime-shape-inference/src/handlers/norm.rs#L105-L188)). This does **not** upgrade the MLA row above. |
| **K3 KDA recurrent attention** | ❌ | ❌ | ❌ | No KDA/DeltaNet operator or kernel is registered. The current KV abstraction stores K/V pages and token-position checkpoints, not arbitrary per-layer recurrent matrices ([kv/lib.rs:1-22, 72-102](../crates/onnx-genai-kv/src/lib.rs#L1-L22)). |
| **K3 periodic Gated MLA** | ❌ | ❌ | ❌ | Plain MLA is already missing; no gate-bearing MLA contract exists. Exact K3 semantics remain unpublished. |
| **AttnRes / Block AttnRes** | ◐ | ◐ | ◐ | AttnRes is ordinary depth-wise norm/projection/softmax/weighted-sum math and can be represented as graph values; the IR supports arbitrary nodes, attributes, and SSA values ([node.rs:25-46](../crates/onnx-runtime-ir/src/node.rs#L25-L46), [graph.rs:14-34](../crates/onnx-runtime-ir/src/graph.rs#L14-L34)). There is no fused op, shape handler, or activation-liveness policy specialized for retaining block representations. AttnRes is an activation-lifetime issue, **not conventional token KV-cache sharing** ([official AttnRes repository](https://github.com/MoonshotAI/Attention-Residuals)). |
| **384/896-expert sparse MoE, top-8/top-16** | ◐ | ◐ | ◐ | CPU has float `MoE` and affine-int `QMoE`; CUDA has affine-int `QMoE` ([CPU:340-345](../crates/onnx-runtime-ep-cpu/src/kernels/mod.rs#L340-L345), [CUDA:190-200](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L190-L200)). Kernels validate dynamic expert counts/top-k, but `QMoE` accepts only affine integer formats and rejects native MXFP4/IQ ([GLM readiness:184-189](GLM_READINESS_GAPS.md#L184-L189)). `BlockQuantizedMoE` has no CPU/CUDA/shape registration ([GLM readiness:154-168](GLM_READINESS_GAPS.md#L154-L168)). |
| **Shared expert** | ✅ | ✅ | ◐ | A shared expert can remain an ordinary dense SwiGLU path using `MatMulNBits` or `BlockQuantizedMatMul`, beside routed `QMoE`. A single fused shared+routed K3 ABI is not frozen. |
| **K2 128K context** | ◐ | ◐ | ◐ | Paged/tiered KV, prefix sharing, and checkpoint/restore exist ([kv/lib.rs:1-18, 72-102](../crates/onnx-genai-kv/src/lib.rs#L1-L18)). CPU attention can be correct, but efficient MLA state is absent. CUDA's own catalogue still defers paged KV ([CUDA mod.rs:8-14](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L8-L14)). |
| **K3 1M hybrid context/prefix cache** | ❌ | ❌ | ❌ | Ordinary token KV pages do not encode KDA recurrent state or its prefix-composition rules. A typed attention-state interface is required before the K3 cache contract can be implemented. |
| **Sub-4-bit dense projections** | ✅ | ✅ | ✅ | `pkg.nxrt::BlockQuantizedMatMul` supports MXFP4 and IQ1/2/3/4 families on CPU and CUDA; CUDA is currently f32-activation/output only ([SUB4BIT_QUANT.md:218-262](SUB4BIT_QUANT.md#L218-L262), [GLM readiness:190-194](GLM_READINESS_GAPS.md#L190-L194)). |
| **K3 MXFP4-weight/MXFP8-activation MoE** | ❌ | ❌ | ❌ | Current `QMoE` rejects non-`int` quant types; `BlockQuantizedMoE` is only a designed boundary. K3's exact QAT packing is unknown until weights/report release. |
| **Huge-model weight offload** | ◐ | ◐ | ◐ | External mmap, lazy `WeightHandle`, capability negotiation, and placement/budget planning exist; live device paging/binding remains Phase 3b ([WEIGHT_OFFLOAD.md:116-161](WEIGHT_OFFLOAD.md#L116-L161), [650-678](WEIGHT_OFFLOAD.md#L650-L678), [weight.rs:94-105, 164-226](../crates/onnx-runtime-ep-api/src/weight.rs#L94-L105)). The lazy boundary is already named `BlockQuantizedMoE`, but no kernel consumes it. |
| **MTP/speculative decoding** | ◐ | ◐ | ◐ | The engine has generic speculative and MTP proposer code, but native package metadata still marks MTP unsupported and reconstructs the proposer per iteration ([parser.rs:70-90](../crates/onnx-genai-metadata/src/parser.rs#L70-L90), [speculative.rs:965-980](../crates/onnx-genai-engine/src/speculative.rs#L965-L980)). K2 and announced K3 do not currently verify an MTP requirement. |
| **Native vision front end (K3)** | ❌ | ❌ | ❌ | K3 verifies native visual understanding, but no released weights/export contract exists yet. This audit covers the decoder runtime; vision encoder/projector coverage must be audited from the released graph rather than guessed. |

## Ranked architectural gaps

### P0 — hybrid KDA/MLA attention and typed persistent state — **XL**

This is the headline K-series gap. Add private, versioned, model-agnostic
contracts such as `pkg.nxrt::MultiLatentAttention` and
`pkg.nxrt::KimiDeltaAttention`, with:

- explicit projected Q, latent KV, RoPE tail, gate/decay, and optional bias
  inputs rather than model-name checks;
- explicit past/present latent or recurrent state outputs;
- prefill, decode, continuous-batch slot, checkpoint, restore, and prefix-import
  semantics;
- CPU reference kernels, then fused CUDA kernels;
- matching shape inference and package metadata; and
- a correctness fallback that is clearly labeled expanded/dense rather than
  silently claimed as MLA/KDA.

CSA supplies a useful implementation pattern for versioned compressed state,
but must not be generalized by pretending ratio-4/128 temporal compression is
low-rank MLA or KDA.

### P0 — native MXFP4/MXFP8 `BlockQuantizedMoE` plus live leases — **L–XL**

Implement the already-designed `pkg.nxrt::BlockQuantizedMoE` boundary with
selected-expert dispatch, shared-expert composition, exact format/layout
negotiation, and lazy expert leases. K3 adds an exact-layout gate: do not call
GGUF MXFP4 compatible with Moonshot's QAT format until byte-level conversion and
numeric parity are proven after weight release. This is the dominant
weight-capacity path for 1T–2.8T models
([SUB4BIT_QUANT.md:264-322](SUB4BIT_QUANT.md#L264-L322),
[WEIGHT_OFFLOAD.md:694-720](WEIGHT_OFFLOAD.md#L694-L720)).

### P1 — 1M-context attention-state/prefix-cache integration — **L–XL**

Generalize the KV layer into an attention-state manager whose state kind is
declared by metadata: dense KV, MLA latent, KDA recurrent matrix, or future
private state. Each state kind needs capacity accounting, tiering,
prefix-compatibility rules, checkpoints, and atomic speculative rollback.
Existing token-position KV checkpoints are the right lifecycle model, but not
the complete payload contract.

### P1 — multi-device expert parallelism and heterogeneous placement — **XL**

At 2.8T parameters, fast deployment needs expert sharding/all-to-all rather
than only single-device paging. The current executor owns one EP for a whole
plan, and non-host initializers are otherwise uploaded eagerly
([WEIGHT_OFFLOAD.md:142-152](WEIGHT_OFFLOAD.md#L142-L152)).
Retain the existing model-agnostic EP registry, but add per-node/per-region
placement, topology-aware expert ownership, dispatch/combine collectives, and
shared-expert replication policy.

### P2 — AttnRes fusion and activation residency — **M**

First support a portable decomposition and let the existing SSA graph represent
block states. Then add an optional fused `AttentionResidual` op if profiling
shows launch or memory pressure. The activation liveness planner must keep only
the required block representatives; this should not be implemented as KV pages.

### Conditional — MTP sidecar orchestration — **M–L**

Do not make Kimi support depend on MTP without a released Kimi artifact that
uses it. If one appears, reuse the approved persistent proposer, explicit state,
and composite rollback design in
[DEEPSEEK_CSA_MTP_RUNTIME.md:700-813](DEEPSEEK_CSA_MTP_RUNTIME.md#L700-L813).

## What the IR/EP architecture needs so Kimi K “can be supported”

The current direction is viable if these boundaries remain first-class:

1. **Versioned semantic operators, not model switches.** The existing
   `(domain, op_type, opset)` registry is the correct dispatch foundation
   ([registry.rs:12-90](../crates/onnx-runtime-ep-api/src/registry.rs#L12-L90)).
   Add KDA/MLA/AttnRes contracts by semantics and layout version.
2. **Typed state as graph-visible I/O plus runtime-owned lifecycle.** The IR
   already supports arbitrary multi-output nodes. The engine needs a generalized
   state group with `append/checkpoint/restore/fork/prefix-import`, rather than
   assuming every attention state is `[K,V]`.
3. **Shape inference for every private stateful op.** Loader/executor allocation
   cannot rely on names or guessed dimensions. MLA and KDA handlers must infer
   all present-state outputs, including fixed-size recurrent states.
4. **Capability negotiation richer than op names.** Advertise attention state
   kinds, quant formats/layouts, dtypes, maximum state dimensions, prefix-cache
   support, and lazy-weight support. Reject incompatible K3 packages before
   allocating model-scale weights.
5. **Lazy immutable weights and mutable attention state must stay separate.**
   Continue using `WeightHandle`/leases for expert weights and KV/state
   checkpoint APIs for per-generation state. They have different ownership,
   mutability, and eviction rules.
6. **Per-node/per-region placement and collectives.** A K3-class model needs
   attention tensor parallelism, expert parallelism, shared-expert policy, and
   bounded offload under one Resource Governor.
7. **Portable oracle profiles.** Keep decomposed f32 attention/MoE exports for
   differential testing, but make package metadata explicit when the profile
   expands MLA, omits KDA prefix caching, requantizes MXFP4, or disables MTP.

## Verdict

For K2, the runtime has most ordinary graph arithmetic and strong MoE/quant/offload
seams, but **does not yet have efficient native MLA**. For announced K3, the gap
widens to KDA recurrent attention, Gated MLA, model-exact MXFP4/MXFP8 MoE, and
1M-context typed state. The architecture direction is on track—private
versioned ops, EP capability dispatch, lazy weights, paged state, and rollback
are the right primitives—but Kimi K support is not achieved until those seams
are generalized and backed by CPU/CUDA kernels.

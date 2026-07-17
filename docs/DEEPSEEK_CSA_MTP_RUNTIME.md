# DeepSeek-V4-Flash CSA and Iterative MTP Runtime Design

> **Status:** Approved (owner @justinchuby, 2026-07-14). Phase 0 (freeze contracts + goldens) cleared. §9 decisions confirmed (CSA-7 fallback default flipped — see §10 Q11). Open questions §10 resolved below.
>
> **Primary target:** DeepSeek-V4-Flash as exported by Mobius PR
> [#405](https://github.com/onnxruntime/mobius/pull/405), commit
> [`7e26e6e`](https://github.com/onnxruntime/mobius/commit/7e26e6eb4e3a8839b311d59160ca947254afff4b).
>
> **Sibling target:** GLM-5.2 IndexShare DSA and improved MTP, where the sparse
> index/cache primitives can be shared but model-specific selection semantics
> must remain separate.
>
> **Date:** 2026-07-16

## 1. Executive recommendation

Land the two missing features through separate, independently testable tracks:

1. **CSA:** add a private, versioned
   `pkg.nxrt::CompressedSparseAttention` operator for the
   production path, plus a small `SparseKvGather` correctness/debug primitive.
   Keep learned projections in ordinary `MatMul`, `MatMulNBits`, or
   `BlockQuantizedMatMul` nodes; the custom op owns only temporal compression,
   sparse index/cache state, selected attention, and learned sink semantics.
2. **MTP:** adapt the existing speculative-decoding loop rather than create a
   second generation loop. Add a persistent, per-generation MTP proposer state,
   a Mobius-native metadata descriptor, explicit Hyper-Connection state
   threading, and checkpoint/restore of target, CSA, and sidecar caches.

CPU correctness should precede CUDA. The earliest useful changes are:

- parse and load Mobius's MTP sidecar without manually constructing
  `MtpConfig`;
- teach the MTP adapter about target `[B,S,hc_mult,H]` state;
- add learned per-head logit sinks to the dense attention reference path; and
- add `SparseKvGather` with bounds, duplicate-index, and masking tests.

Those can land before the compressor equations and fused CSA kernel are ready.

The central blocker is a contract issue, not merely missing Rust code:
Mobius PR #405 preserves the official CSA tensors but deliberately references
them through zero-valued shape anchors. It does **not** encode the learned
compression or sparse-selection equations in ONNX. Likewise, the MTP graph
accepts Hyper-Connection state but exports only a collapsed `mtp_hidden`
result. Before kernel implementation, Mobius and the runtime need one exact,
golden-tested state-transition contract.

## 2. Mobius source-of-truth contract

This section describes the graph at Mobius commit `7e26e6e`, not a proposed
future graph.

### 2.1 Model schedule and configuration

Mobius derives `compress_ratios` from the model's per-layer `layer_types` and
`compress_rates`. The real-config test maps:

```text
sliding_attention             -> 0
compressed_sparse_attention   -> 4
heavily_compressed_attention  -> 128
```

and verifies the representative schedule `[0, 0, 4, 128]`
([test lines 52-121](https://github.com/onnxruntime/mobius/blob/7e26e6eb4e3a8839b311d59160ca947254afff4b/src/mobius/models/deepseek_v4_flash_test.py#L52-L121)).
Only ratios `0`, `4`, and `128` are accepted
([exporter lines 329-342](https://github.com/onnxruntime/mobius/blob/7e26e6eb4e3a8839b311d59160ca947254afff4b/src/mobius/models/deepseek_v4.py#L329-L342)).

Relevant architecture attributes are:

```text
num_attention_heads
head_dim
q_lora_rank
qk_rope_head_dim
index_n_heads
index_head_dim
index_topk
compress_ratios
compress_rope_theta
hc_mult
hc_sinkhorn_iters
hc_eps
num_nextn_predict_layers
```

Compressed layers use a separate rotary embedding initialized from
`compress_rope_theta`, while ratio-0 layers use the ordinary rotary embedding
([exporter lines 620-695](https://github.com/onnxruntime/mobius/blob/7e26e6eb4e3a8839b311d59160ca947254afff4b/src/mobius/models/deepseek_v4.py#L620-L695)).

### 2.2 CSA tensors retained by the current export

For a layer with compression ratio `R`, Mobius retains:

```text
self_attn.compressor.ape                 [R, overlap * D]
self_attn.compressor.wkv.weight          [overlap * D, H]
self_attn.compressor.wgate.weight        [overlap * D, H]
self_attn.compressor.norm.weight         [D]

overlap = 2 when R == 4
overlap = 1 when R == 128
```

where `H=hidden_size` and `D=head_dim`. Quantized exports preserve the
projection as packed weight/scales/optional zero-point tensors with the same
logical in/out dimensions
([exporter lines 68-123](https://github.com/onnxruntime/mobius/blob/7e26e6eb4e3a8839b311d59160ca947254afff4b/src/mobius/models/deepseek_v4.py#L68-L123),
[lines 237-261](https://github.com/onnxruntime/mobius/blob/7e26e6eb4e3a8839b311d59160ca947254afff4b/src/mobius/models/deepseek_v4.py#L237-L261)).

Ratio-4 layers additionally retain an indexer:

```text
self_attn.indexer.wq_b.weight                 [index_n_heads*index_head_dim, q_lora_rank]
self_attn.indexer.weights_proj.weight         [index_n_heads, H]
self_attn.indexer.compressor.ape              [4, 2*index_head_dim]
self_attn.indexer.compressor.wkv.weight       [2*index_head_dim, H]
self_attn.indexer.compressor.wgate.weight     [2*index_head_dim, H]
self_attn.indexer.compressor.norm.weight      [index_head_dim]
```

Ratio-128 layers have the compressor but no ratio-4 sparse indexer
([exporter lines 264-286](https://github.com/onnxruntime/mobius/blob/7e26e6eb4e3a8839b311d59160ca947254afff4b/src/mobius/models/deepseek_v4.py#L264-L286),
[test lines 139-160](https://github.com/onnxruntime/mobius/blob/7e26e6eb4e3a8839b311d59160ca947254afff4b/src/mobius/models/deepseek_v4_flash_test.py#L139-L160)).

Every layer also exports:

```text
self_attn.attn_sink  float [num_attention_heads]
```

This is a **learned logit sink per attention head**. It is not a count of
leading tokens retained by StreamingLLM-style sink-token caching.

### 2.3 What the current graph actually executes

The compressor and indexer tensors are not used in compression math. Each
deferred tensor is flattened, element zero is gathered, compared with itself,
cast, summed, multiplied by zero, and finally added to the attention output.
This keeps initializers reachable without changing numerics
([exporter lines 54-65](https://github.com/onnxruntime/mobius/blob/7e26e6eb4e3a8839b311d59160ca947254afff4b/src/mobius/models/deepseek_v4.py#L54-L65),
[lines 455-459](https://github.com/onnxruntime/mobius/blob/7e26e6eb4e3a8839b311d59160ca947254afff4b/src/mobius/models/deepseek_v4.py#L455-L459)).

The executable attention is a standard-ONNX dense fallback:

1. project and RMS-normalize Q and one-head MQA KV;
2. apply ordinary or compressed-theta RoPE;
3. concatenate dense past K/V;
4. expand the one KV head to all query heads;
5. compute `QK^T * scale + attention_bias`;
6. append one learned sink logit per head;
7. softmax over keys plus the sink;
8. remove the sink probability column, so it contributes denominator mass but
   no value vector;
9. multiply by dense V, apply inverse RoPE, and run grouped output projections;
10. return dense `present_key` and `present_value`.

The exact decomposition is visible in
[exporter lines 398-480](https://github.com/onnxruntime/mobius/blob/7e26e6eb4e3a8839b311d59160ca947254afff4b/src/mobius/models/deepseek_v4.py#L398-L480).
There is no CSA custom op, compressed cache tensor, selected-index tensor, or
sparse gather node in the current graph.

The fallback is portable but not the official sparse execution path. It also
retains a dense one-head KV cache at every compressed layer, forfeiting the
capacity and bandwidth benefit of ratios 4 and 128.

### 2.4 MTP target and sidecar graphs

When `num_nextn_predict_layers == 1`, Mobius exports two models:

```text
model/model.onnx   target decoder
mtp/model.onnx     one official MTP decoder block
```

The target adds this output:

```text
hidden_states  T [B, S, hc_mult, H]
```

It is the final Hyper-Connection state before the target's `_hc_head`
reduction, not a conventional `[B,S,H]` hidden tensor
([task lines 71-88](https://github.com/onnxruntime/mobius/blob/7e26e6eb4e3a8839b311d59160ca947254afff4b/src/mobius/tasks/_deepseek_v4.py#L71-L88),
[exporter lines 654-695](https://github.com/onnxruntime/mobius/blob/7e26e6eb4e3a8839b311d59160ca947254afff4b/src/mobius/models/deepseek_v4.py#L654-L695)).

The sidecar inputs are:

```text
inputs_embeds              T   [B, S, H]
hidden_states              T   [B, S, hc_mult, H]
attention_mask             i64 [B, P+S]
position_ids               i64 [B, S]
past_key_values.0.key      T   [B, 1, P, D]
past_key_values.0.value    T   [B, 1, P, D]
```

and outputs:

```text
mtp_hidden                 T [B, S, H]
present.0.key              T [B, 1, P+S, D]
present.0.value            T [B, 1, P+S, D]
```

([task lines 90-115](https://github.com/onnxruntime/mobius/blob/7e26e6eb4e3a8839b311d59160ca947254afff4b/src/mobius/tasks/_deepseek_v4.py#L90-L115)).

The MTP block:

1. normalizes and projects the previous token embedding with `e_proj`;
2. normalizes and projects the HC target/state input with `h_proj`;
3. broadcasts and adds those streams;
4. executes one full V4 decoder layer with its own KV;
5. collapses HC lanes through `_hc_head`;
6. normalizes the collapsed result and exports `mtp_hidden`.

The target embedding and LM head are shared externally rather than duplicated
inside the sidecar
([exporter lines 698-764](https://github.com/onnxruntime/mobius/blob/7e26e6eb4e3a8839b311d59160ca947254afff4b/src/mobius/models/deepseek_v4.py#L698-L764)).

Mobius's test confirms the two-model package, target `hidden_states`, sidecar
weights, and the three sidecar outputs. Its ORT GenAI sidecar JSON records the
MTP model filename and `num_nextn_predict_layers=1`
([test lines 163-211](https://github.com/onnxruntime/mobius/blob/7e26e6eb4e3a8839b311d59160ca947254afff4b/src/mobius/models/deepseek_v4_flash_test.py#L163-L211)).

## 3. Runtime gap analysis

### 3.1 Dense fallback operations

The CPU EP already registers the standard operations used by the fallback,
including `MatMul`, `MatMulNBits`, `Add`, `Mul`, `Div`, `Sqrt`, `Cast`,
`CastLike`, `Reshape`, `Transpose`, `Split`, `Concat`, `Expand`, `Slice`,
`Softmax`, `Gather`, `GatherElements`, `TopK`, and `RMSNormalization`
([registry lines 199-220](../crates/onnx-runtime-ep-cpu/src/kernels/mod.rs#L199-L220),
[lines 325-430](../crates/onnx-runtime-ep-cpu/src/kernels/mod.rs#L325-L430),
[lines 623-681](../crates/onnx-runtime-ep-cpu/src/kernels/mod.rs#L623-L681)).
Native IQ/MXFP4 projections are also available through the private
`BlockQuantizedMatMul` v1 contract
([kernel lines 20-22](../crates/onnx-runtime-ep-cpu/src/kernels/block_quantized_matmul.rs#L20-L22),
[lines 150-214](../crates/onnx-runtime-ep-cpu/src/kernels/block_quantized_matmul.rs#L150-L214)).

Therefore the zero-valued compressor/indexer anchors and dense attention
decomposition are representable on CPU. That is a correctness fallback, not
native CSA.

### 3.2 Attention kernel gaps

The CPU `com.microsoft::GroupQueryAttention` kernel supports packed or unpacked
QKV, dense BNSH past/present caches, causal/local-window masking, RoPE, and
softcap. It explicitly rejects:

- `head_sink`;
- attention bias;
- packed or quantized KV cache;
- smooth softmax; and
- QK capture.

([kernel lines 1-6](../crates/onnx-runtime-ep-cpu/src/kernels/group_query_attention.rs#L1-L6),
[lines 332-367](../crates/onnx-runtime-ep-cpu/src/kernels/group_query_attention.rs#L332-L367)).
Its core allocates dense present K/V and loops over the contiguous key range
([lines 527-608](../crates/onnx-runtime-ep-cpu/src/kernels/group_query_attention.rs#L527-L608)).

The ordinary `Gather`, `GatherND`, `GatherElements`, `ScatterElements`, and
`TopK` kernels do not close the gap. They have no persistent compressed-cache
contract, no index-cache cursor, no per-layer 0/4/128 state transition, and
would materialize selected tensors between nodes. The CUDA and CPU GQA paths
likewise have no compressed/index-cache input.

### 3.3 KV-cache gaps

The engine's paged KV system already provides page allocation, prefix sharing,
checkpoint/rewind, and dense sliding-window retention
([KV overview lines 1-18](../crates/onnx-genai-kv/src/lib.rs#L1-L18)).
The decode backend exposes `decode(tokens, past_len)` and `rewind(target_len)`
([decode lines 43-59](../crates/onnx-genai-engine/src/decode.rs#L43-L59)).
Native CUDA decode also distinguishes logical length from fixed physical
capacity and can rewind without copying accepted cache prefixes
([native decode lines 590-620](../crates/onnx-genai-engine/src/native_decode.rs#L590-L620)).

CSA needs more state than dense K/V:

- compressed KV records;
- a partial compression carry for tokens not yet completing a ratio block;
- ratio-4 index-key records;
- logical lengths for each state stream; and
- a token-position-to-state-cursor journal for rollback.

The existing cache assumes token-indexed K/V pairs. It cannot infer that, for
example, rewinding one token may remove no compressed record but alter a
partial ratio-128 carry. It also does not discover non-K/V graph state from the
current `past_key_values.*` naming convention.

The engine's existing `sink_tokens` feature is unrelated. It retains leading
token pages for sliding-window attention
([decode lines 26-40](../crates/onnx-genai-engine/src/decode.rs#L26-L40));
DeepSeek's `attn_sink` is an extra learned softmax logit with no value vector.

### 3.4 Existing MTP infrastructure: useful but incompatible

The repository already contains the right high-level speculative mechanism:

- `SpeculativeProposer`;
- target batched verification;
- longest-prefix greedy acceptance;
- correction-token generation;
- target/draft KV rewind; and
- acceptance statistics.

The loop is explicitly shared by draft-model, MTP, EAGLE-3, shared-KV, and
prompt-lookup proposers
([speculative lines 1-5](../crates/onnx-genai-engine/src/speculative.rs#L1-L5),
[lines 999-1122](../crates/onnx-genai-engine/src/speculative.rs#L999-L1122)).
Target rollback already truncates tokens, paged KV, and the active decode
runner
([KV bridge lines 392-409](../crates/onnx-genai-engine/src/kv_bridge.rs#L392-L409),
[lines 449-497](../crates/onnx-genai-engine/src/kv_bridge.rs#L449-L497)).

The current MTP adapter nevertheless cannot run the DeepSeek sidecar:

1. `MtpConfig` requires separate raw f32 embedding and LM-head files
   ([config lines 175-200](../crates/onnx-genai-engine/src/config.rs#L175-L200)).
   Mobius shares those weights from the target package.
2. Metadata discovery marks `proposal_type: mtp` as `NotYetSupported`
   ([parser lines 70-90](../crates/onnx-genai-metadata/src/parser.rs#L70-L90)).
   MTP currently works only when configured programmatically.
3. `extract_last_hidden` accepts only rank 1, 2, or 3 tensors, so the target's
   `[B,S,hc_mult,H]` output is rejected
   ([decode lines 1215-1250](../crates/onnx-genai-engine/src/decode.rs#L1215-L1250)).
4. `MtpDecodeSession::step` requires `hidden_states.len() ==
   inputs_embeds.len()` and binds both as `[B,S,H]`
   ([MTP session lines 211-240](../crates/onnx-genai-ort/src/mtp.rs#L211-L240)).
   DeepSeek requires `[B,S,hc_mult,H]` for `hidden_states`.
5. The current proposer threads collapsed `mtp_hidden` as the next
   `hidden_states` value
   ([speculative lines 344-382](../crates/onnx-genai-engine/src/speculative.rs#L344-L382)).
   The Mobius graph does not define how that rank-3 tensor becomes the next
   rank-4 HC state.
6. A new `MtpProposer` and `MtpDecodeSession` are constructed for each verify
   iteration
   ([speculative lines 957-972](../crates/onnx-genai-engine/src/speculative.rs#L957-L972)).
   Only draft-model acceptance is notified to persistent state
   ([lines 1179-1181](../crates/onnx-genai-engine/src/speculative.rs#L1179-L1181)).
7. Hidden-output speculation disables the optimized decode runner because that
   runner does not preserve arbitrary target outputs
   ([decode lines 534-546](../crates/onnx-genai-engine/src/decode.rs#L534-L546)).

The correct design is therefore an adapter and state-lifetime extension over
the existing loop, not a new speculative algorithm.

## 4. Proposed CSA native path

### 4.1 Design principles

1. **No model-name dispatch.** Placement is by domain/op/version and metadata.
2. **Keep learned linear algebra outside the custom op.** Mobius can emit the
   compressor/indexer projections with its existing quantization policy.
3. **Fuse temporal state and selected attention.** Do not materialize a
   `[B,heads,Q,topk,D]` gather in the production path.
4. **Version every private layout.** Cache records must not depend on an
   undocumented Rust struct.
5. **Preserve a dense fallback package.** Ratio-0 always uses dense attention;
   ratios 4/128 may use the current fallback when native CSA is not requested.
6. **Fail closed.** A package requiring CSA must be rejected before large
   allocations when no compatible op/layout is available.

### 4.2 Low-level correctness primitive

Add:

```text
domain: pkg.nxrt
name:   SparseKvGather
since:  1

inputs:
  cache          T          [B, G, C, D]
  indices        int32|int64 [B, G, Q, K]
  valid_lengths  int32|int64 [B]              optional

output:
  selected       T          [B, G, Q, K, D]

attributes:
  index_layout_version: int = 1
  out_of_range: string = "error"  # v1 supports only "error"
```

Semantics:

- preserve index order and duplicates;
- validate non-negative indices and `index < valid_lengths[b]` (or `< C`);
- permit `G=1` followed by explicit/built-in broadcast;
- support f32 first, then f16/bf16;
- produce a contiguous output; and
- report exact offending batch/group/query/index coordinates.

This op is intentionally not the final fast path. It provides a small unit of
portable semantics for CPU reference tests, Mobius golden fixtures, and reuse
by GLM's DSA work. The fused CSA kernel may call the same internal gather
helper without allocating `selected`.

### 4.3 Production fused operator

Add:

```text
domain: pkg.nxrt
name:   CompressedSparseAttention
since:  1
```

The operator consumes **projected activations**, not learned matrix weights:

```text
required inputs:
  query                 T [B, S, N, D]       # already normalized/rotary-applied
  current_kv            T [B, S, D]          # one-head V4 MQA source
  compressor_kv         T [B, S, CW]
  compressor_gate       T [B, S, CW]
  compressor_ape        T [R, CW]
  compressor_norm       T [D]
  past_compressed_kv    T [B, C, CW]
  past_compression_carry T [B, Tcarry, 2, CW]
  seqlens_k             int32 [B]
  total_sequence_length int64 scalar
  head_sink             T [N]

optional ratio-4 inputs:
  index_query           T [B, S, I, ID]
  index_weight          T [B, S, I]
  index_compressor_kv   T [B, S, ICW]
  index_compressor_gate T [B, S, ICW]
  index_compressor_ape  T [4, ICW]
  index_compressor_norm T [ID]
  past_index_key        T [B, I, CI, ID]
  past_index_carry      T [B, TIcarry, 2, ICW]

optional common input:
  attention_bias        T/bool broadcastable to [B,N,S,Candidate]

outputs:
  Y                     T [B, S, N, D]
  present_compressed_kv T [B, Cnext, CW]
  present_compression_carry T [B, Tnext, 2, CW]
  present_index_key     T [B, I, CInext, ID]          # absent unless R=4
  present_index_carry   T [B, TInext, 2, ICW]         # absent unless R=4
  selected_indices      int32 [B, I, S, K]            # optional diagnostic output
```

Symbol meanings:

```text
R   compression_ratio, 4 or 128
N   query heads
D   attention head dimension
I   index heads
ID  index head dimension
K   index_topk
CW  compressed record width fixed by cache_layout_version
ICW index-compressor record width fixed by cache_layout_version
```

Attributes:

```text
num_heads: int
head_dim: int
compression_ratio: int              # exactly 4 or 128 in v1
index_num_heads: int = 0
index_head_dim: int = 0
index_topk: int = 0
scale: float = 0                    # 0 means 1/sqrt(D)
causal: int = 1                     # v1 requires 1
sink_mode: string = "logit_only"    # v1 requires this value
cache_layout_version: int = 1
index_layout_version: int = 1
```

`CW`, carry contents, compression boundaries, index-key construction, and the
mapping from index heads to attention heads must be frozen from the official
implementation and golden vectors before registration. The shapes above define
the API boundary; they do not authorize guessing the missing equations.

#### Why projected activations are inputs

Mobius can lower:

```text
hidden -> compressor.wkv
hidden -> compressor.wgate
q_lora -> indexer.wq_b
hidden -> indexer.weights_proj
hidden -> indexer.compressor.{wkv,wgate}
```

through existing `MatMul`, `MatMulNBits`, or `BlockQuantizedMatMul` nodes.
That preserves quantization and weight-offload policy, avoids teaching the CSA
op every packed weight format, and lets CPU/GPU CSA share one state contract.
The private op begins where ordinary ONNX lacks semantics: cross-token
compression, index-cache update, selected attention, and sink normalization.

### 4.4 Ratio-specific behavior

#### Ratio 0

Do not emit `CompressedSparseAttention`. Use dense GQA or the current standard
decomposition. Ratio 0 has ordinary dense KV state.

#### Ratio 4

- update compressed KV and partial carry;
- update the independent index-key cache and its carry;
- compute top-k indices using the official index-query/index-weight rule;
- apply causal and valid-length filtering before top-k;
- preserve deterministic tie-breaking defined by the golden reference;
- attend only selected compressed records; and
- include the learned head sink as denominator mass with zero value
  contribution.

`index_topk` larger than available causal records selects only the available
records. Padding must never become a selectable zero vector.

#### Ratio 128

- update compressed KV and carry;
- do not require or emit ratio-4 index state;
- attend according to the official heavily-compressed rule.

Mobius currently exports no ratio-128 indexer. Whether ratio-128 attends every
compressed record or applies another implicit selection rule must be confirmed
from the official reference before implementation.

### 4.5 Dense fallback and sink handling

The learned sink must exactly match Mobius's current fallback:

```text
probabilities = softmax(concat(real_scores, head_sink))
output        = probabilities[..., :num_real_keys] @ V
```

The sink has no K/V record and is never written to cache. It is unrelated to
`sink_tokens`, page pinning, or local-window retention.

For debugging, the CPU CSA test should run both:

1. the fused kernel; and
2. a decomposed oracle using `SparseKvGather`, explicit score computation,
   sink concatenation, softmax, and value reduction.

The current Mobius dense fallback remains a third reference for short contexts,
but PR #405 itself notes that dense fallback is not generally numerically
equivalent to official learned compression.

### 4.6 Cache ownership and rollback

Introduce metadata-declared **state groups** rather than infer every persistent
tensor from `past_key_values.*` names:

```yaml
state_groups:
  - kind: compressed_sparse_attention
    layer: 2
    compression_ratio: 4
    tensors:
      compressed_kv: [past_csa.2.kv, present_csa.2.kv]
      compression_carry: [past_csa.2.carry, present_csa.2.carry]
      index_key: [past_csa.2.index, present_csa.2.index]
      index_carry: [past_csa.2.index_carry, present_csa.2.index_carry]
```

`DecodeState` should maintain, per forward:

```text
token_len
compressed_len
compression_carry_len
index_len
index_carry_len
```

Before a speculative verification, capture a cursor checkpoint. After target
verification, restore the cursor corresponding to `base_len + accepted`.
For CPU growing tensors, restoration may use prefix views. For fixed-capacity
native CPU/CUDA state, restoration changes logical lengths and clears only
invalid carry/index tails. It must not recompress the entire accepted prefix.

The existing token-only `rewind(target_len)` API can remain the public engine
call if each decode backend owns the token-to-auxiliary-cursor journal.
Internally, adding `checkpoint()`/`restore(checkpoint)` to the decode runner is
safer than trying to derive compressed lengths from token length after the fact.

### 4.7 CPU implementation shape

The first CPU kernel is a correctness implementation:

- f32 only;
- batch 1 and arbitrary `S` first, then batch >1;
- checked scalar compressor/index equations;
- stable deterministic top-k;
- no materialized selected tensor in the fused path;
- growing cache outputs initially;
- exact diagnostic `selected_indices` available in tests; and
- dense oracle parity at every compression boundary.

After correctness:

- parallelize query-head and query-token work with Rayon;
- vectorize score/value reductions;
- use fixed-capacity cache buffers for decode;
- specialize `S=1`;
- add f16/bf16 input/output with f32 accumulation; and
- expose honest FLOP/bytes estimates to placement.

### 4.8 CUDA follow-up

CUDA should consume the same op and layout versions. The production kernel
should:

- keep compressed KV, index key, carry, and logical lengths device-resident;
- update only new records;
- fuse selection, score, sink softmax, and value reduction;
- avoid a global selected-KV materialization;
- preserve stable buffer addresses for CUDA graph capture;
- expose no host index round trip; and
- use stream-ordered checkpoint/restore of logical cursors.

`supports_op` must reject unsupported ratio/layout/dtype/shape combinations
instead of claiming the node and falling back inside the kernel.

## 5. Mobius CSA capability and export changes

Extend Mobius's centralized EP capability model with semantic features:

```python
@dataclass(frozen=True)
class SparseAttentionCapabilities:
    compressed_sparse_attention_versions: frozenset[int] = frozenset()
    sparse_kv_gather_versions: frozenset[int] = frozenset()
    compression_ratios: frozenset[int] = frozenset()
    cache_layout_versions: frozenset[int] = frozenset()
    learned_logit_sink: bool = False

@dataclass(frozen=True)
class EpCapabilities:
    ...
    sparse_attention: SparseAttentionCapabilities = SparseAttentionCapabilities()
```

Emission policy:

1. ratio 0 always emits dense attention;
2. ratios 4/128 emit the private op only when the selected runtime advertises
   op v1, ratio support, cache layout v1, and learned-logit sinks;
3. learned projections remain ordinary quantized/unquantized linear nodes;
4. otherwise emit the current dense fallback and do not claim native CSA;
5. a `native_csa_required` export option rejects instead of silently
   producing a dense package; and
6. package metadata records:

```yaml
required_capabilities:
  - compressed_sparse_attention_v1
  - sparse_attention_cache_layout_v1
```

The runtime's current default capability list contains only KV cache, GQA,
MHA, prefix cache, and continuous batching
([validation lines 7-23](../crates/onnx-genai-metadata/src/validation.rs#L7-L23)).
CSA capability strings must be added only when the complete load/execute/rewind
path exists.

## 6. Proposed iterative MTP orchestration

### 6.1 Reuse the existing speculative state machine

One verification iteration should remain:

```text
target base step
  -> guaranteed target token + target HC seed
  -> iterative MTP draft of up to k additional tokens
  -> one target verification forward over proposed tokens
  -> longest accepted prefix
  -> target correction token on first mismatch, or bonus target token
  -> restore target/CSA/MTP state to committed prefix
  -> repeat
```

This is already the shape of `generate_speculative_loop`. The implementation
work is to make DeepSeek's proposer and caches conform to it.

### 6.2 Persistent per-generation MTP state

Add `MtpSessionState` beside `DraftSession` in `EngineSession`:

```text
MtpSessionState
  proposer/session binding
  sidecar KV state
  current HC state
  absolute position
  last accepted target length
  checkpoint journal for the active verify iteration
```

Do not construct a new `MtpProposer` inside every loop iteration. The engine
loads the MTP model once, creates one proposer state per generation/session,
and calls:

```text
begin_iteration(target_len, target_hc_state)
propose(max_additional_tokens)
accept(accepted_prefix_len, committed_tokens)
restore(target_len)  # session rewind/reset
```

For a stateless/hidden-threaded sidecar, these methods may reset cheaply. For a
growing sidecar cache, accepted state remains reusable.

### 6.3 Make HC threading explicit

The target seed is `[B,1,hc_mult,H]`; the token candidate state used by the
shared LM head is `[B,1,H]`. These are different types and must not be stored
in one `Vec<f32>` field.

The recommended Mobius contract change is to export a second sidecar output:

```text
mtp_hidden  T [B,S,H]          # feed shared target LM head
mtp_state   T [B,S,hc_mult,H]  # feed next MTP iteration
```

`mtp_state` is the post-layer HC state before `_hc_head` collapse. The first
sidecar call receives target `hidden_states`; subsequent calls receive the
previous `mtp_state`. This avoids inventing a broadcast/lift rule from
`mtp_hidden` back to HC lanes.

If the official algorithm instead requires a different recurrent state,
Mobius must export that state explicitly with matching input/output shapes.
The runtime should not infer it from tensor names.

### 6.4 Draft length and positions

Define:

```text
num_speculative_tokens = maximum MTP-produced tokens after the guaranteed
                         target token
proposal width         = 1 + num_speculative_tokens
```

This preserves the current engine interpretation
([speculative lines 765-777](../crates/onnx-genai-engine/src/speculative.rs#L765-L777)).

At draft iteration `j`:

```text
previous_token = guaranteed token when j=0, else prior MTP token
inputs_embeds  = target embedding(previous_token)
position_ids   = base_target_length + j
attention_mask = valid sidecar past plus this token
```

The position is absolute target sequence position, not merely the sidecar
cache length. This matters when sidecar state is reset per verify iteration.

### 6.5 Verification, acceptance, and correction

Target verification remains authoritative:

- run the target once over the complete proposal when the backend supports
  multi-token decode;
- apply the normal processor chain to each target logit row;
- accept the longest prefix equal to target selection;
- on mismatch, commit the target-selected correction token;
- if all drafts match and limits permit, commit the bonus token selected from
  the final target logit row; and
- report proposal/acceptance statistics exactly as today.

The MTP path initially remains greedy/temperature-zero only, matching
`should_use_speculative`
([speculative lines 724-747](../crates/onnx-genai-engine/src/speculative.rs#L724-L747)).
Sampling-speculation acceptance is a separate design.

### 6.6 KV and auxiliary-state rollback

Capture one composite checkpoint before target verification:

```text
target dense/CSA state cursor
target paged-KV cursor
MTP sidecar KV cursor
MTP recurrent HC cursor
logical token length
```

After accepting `a` proposed tokens:

```text
restore target state to base_len + a
restore MTP state to the recurrent state after a MTP-produced tokens
discard all rejected dense/CSA/index/carry records
commit correction/bonus token through the normal target path
```

If sidecar KV is purely proposal-local, reset it after every verify iteration.
If the official contract allows cross-iteration reuse, retain only state proven
to correspond to committed target tokens. The metadata must declare the mode;
do not guess from whether the graph happens to expose past/present tensors.

CSA makes target rollback stricter: restoring by dense token count must also
restore compressed record, index record, and partial-carry cursors.

### 6.7 Package and metadata contract

Extend native `inference_metadata` so `proposal_type: mtp` resolves to a usable
descriptor:

```yaml
speculative:
  proposal_type: mtp
  model: mtp/model.onnx
  num_speculative_tokens: 4
  target_hidden_output: hidden_states
  target_hidden_layout: BSHC
  target_hidden_size: 4096
  hc_mult: 4
  mtp_hidden_output: mtp_hidden
  mtp_state_output: mtp_state
  kv_mode: proposal_local          # or accepted_prefix
  embedding:
    source: target_initializer
    name: model.embed_tokens.weight
  lm_head:
    source: target_initializer
    name: lm_head.weight
```

The exact initializer names should be emitted, not discovered heuristically.
The loader should borrow the target `WeightStore` representation rather than
require duplicate raw `.f32` files. The embedding/LM-head adapters must support
the package's actual dtype/quantization:

- f32 can use the current linear helpers;
- quantized embeddings use the runtime embedding kernel/component;
- quantized LM heads use existing MatMulNBits/BlockQuantizedMatMul execution;
- tied weights share one backing range.

A simpler first milestone may require f32 target embedding and LM head, but the
metadata and ownership model should not make that limitation permanent.

### 6.8 Optimized decode integration

MTP currently forces the legacy output-preserving path because optimized decode
runners return logits only. Add named auxiliary outputs to the decode backend:

```text
decode_with_outputs(tokens, past_len, requested_outputs)
  -> logits + named tensors
```

Native CPU/CUDA decode should bind `hidden_states` to a persistent or reusable
buffer and return only the final token row needed by the proposer. Do not copy
the full `[B,S,hc_mult,H]` history to host on every step.

The first correct implementation may use the legacy path. Moving MTP to the
optimized native runner is a performance milestone, not a prerequisite for
validating orchestration.

## 7. Phased delivery

### Phase 0 — freeze contracts and goldens

**Can land first; no runtime behavior change.**

- obtain the official CSA compressor/indexer equations and deterministic top-k
  tie rule;
- add Mobius golden fixtures that cross ratio-4 and ratio-128 boundaries;
- export explicit projected CSA activations and custom-op fixtures;
- export explicit recurrent `mtp_state`;
- record exact cache/carry layouts as layout version 1; and
- include target-vs-MTP iterative golden tokens and intermediate states.

**Pass bar:** a Python/reference implementation can serialize every proposed
op input/output and replay one prefill plus multiple decode/rewind steps.

### Phase 1 — MTP metadata and HC adapter

**Independent of CSA kernels.**

- resolve native MTP metadata instead of returning `NotYetSupported`;
- load target embedding/LM head by package reference;
- support rank-4 target HC extraction;
- bind sidecar `hidden_states` as BSHC;
- thread explicit `mtp_state`;
- keep persistent per-generation proposer state; and
- reuse existing greedy draft/verify/correction logic.

**Pass bar:** the tiny DeepSeek MTP package is token-identical to target-only
greedy decode for zero-accept, partial-accept, and full-accept fixtures.

### Phase 2 — learned sink and sparse gather CPU primitives

**Independent, reusable primitives.**

- support `head_sink` in the CPU dense attention reference;
- add CPU `SparseKvGather` v1;
- add bounds, duplicates, masks, empty-prefix, and deterministic-layout tests;
- distinguish learned logit sinks from `sink_tokens` in metadata/errors.

**Pass bar:** decomposed selected attention matches an independent scalar
oracle, including sink denominator semantics.

### Phase 3 — CPU ratio-128 compressed cache

- implement compressor/carry update for ratio 128;
- add metadata-declared CSA state groups and cursor journal;
- implement ratio-128 fused attention;
- support prefill, decode, reset, and rewind across a 128-token boundary;
- retain dense fallback for unsupported packages.

**Pass bar:** official golden state/logits match; target-only generation remains
stable across rollback and context continuation.

### Phase 4 — CPU ratio-4 indexer and mixed schedule

- implement ratio-4 compressor, index-key cache, selection, and fused attention;
- support the complete `[0,0,4,128]` mixed-layer schedule;
- expose optional selected indices in diagnostics/tests;
- verify batch 1 first, then batch >1;
- benchmark fused versus materialized gather.

**Pass bar:** the Mobius tiny schedule and a real-model slice match official
goldens through prefill, decode, speculative reject, and resume.

### Phase 5 — native-runner and fixed-capacity state

- add named hidden-output binding to native decode;
- allocate fixed-capacity CSA/index/carry buffers;
- make rollback cursor-only where valid;
- eliminate full hidden-state host copies;
- add capacity planning and clear overflow errors.

**Pass bar:** pointer stability, zero full-cache copies, exact logical cursor
restoration, and no regression to dense-GQA decode behavior.

### Phase 6 — CUDA

- implement CUDA sparse selection and fused attention for ratios 4/128;
- keep all CSA and MTP state device-resident;
- add stream-safe restore and graph-capture compatibility;
- claim only supported dtype/shape/layout combinations;
- validate CPU/CUDA state, selected indices, logits, and greedy tokens.

**Pass bar:** no host index/cache round trips during steady decode and measured
speed/memory improvement over the dense fallback.

### Phase 7 — GLM-5.2 reuse

- reuse `SparseKvGather`, state-group metadata, checkpointing, and speculative
  orchestration;
- define a separate IndexShare selection contract/op if its equations do not
  match DeepSeek CSA; and
- do not overload `CompressedSparseAttention` with model-specific branches.

## 8. Observability and failure policy

Required metrics:

- active attention mode per layer: dense, ratio-4 CSA, ratio-128 HCA;
- dense KV bytes avoided;
- compressed/index/carry bytes and logical lengths;
- selected K and available candidate count;
- compression/index update time;
- gather/score/softmax/value time;
- sink probability mass summary;
- MTP proposed, accepted, correction, and bonus tokens;
- MTP target verification time and sidecar time;
- target, CSA, and MTP rollback counts;
- host/device bytes moved for hidden and index state; and
- fallback reason when native CSA or MTP is not selected.

Failure rules:

1. unknown op/layout versions fail before execution;
2. malformed cache or index shapes fail with layer/tensor names;
3. out-of-range sparse indices fail, never clamp silently;
4. missing official recurrent MTP state fails, never broadcasts implicitly;
5. a `native_csa_required` package never falls back to dense;
6. a fallback-capable package logs the memory/performance consequence once;
7. speculative rejection must leave all target and sidecar cursors at the
   committed prefix; and
8. capability metadata is advertised only after end-to-end load, run, and
   rewind support exists.

## 9. Decisions proposed for approval

| ID | Proposed decision | Rationale |
|---|---|---|
| CSA-1 | Use private domain `pkg.nxrt`, version 1. | Matches `BlockQuantizedMatMul` incubation and permits explicit layout versioning. |
| CSA-2 | Keep learned projections outside the CSA op. | Reuses existing quantized linear kernels and avoids coupling sparse attention to every weight format. |
| CSA-3 | Add `SparseKvGather` as a correctness/reuse primitive, but fuse gather into production CSA. | Gives testable semantics without imposing a large intermediate on the fast path. |
| CSA-4 | Do not emit the custom op for ratio 0. | Dense attention is the correct and simpler contract. |
| CSA-5 | Treat learned `attn_sink` as logit-only denominator mass. | Exactly matches the current Mobius fallback. |
| CSA-6 | Use metadata-declared state groups and per-forward cursor journals. | CSA carries multiple state streams whose lengths cannot be inferred from dense token count alone. |
| CSA-7 | Preserve an explicit dense fallback package, with an opt-in native-required export mode. | Enables portability without disguising fallback cost or semantics. |
| MTP-1 | Reuse `generate_speculative_loop`. | Draft, verify, accept, correction, and target rewind already exist. |
| MTP-2 | Make the MTP proposer persistent per generation/session. | Required for sidecar cache/state ownership and correct acceptance rollback. |
| MTP-3 | Export explicit recurrent `mtp_state` matching the next sidecar input. | Avoids guessing how collapsed `[B,S,H]` becomes `[B,S,hc_mult,H]`. |
| MTP-4 | Resolve embedding and LM head from package components/initializers, not duplicate raw f32 files. | Matches Mobius weight sharing and supports quantized packages. |
| MTP-5 | Define `num_speculative_tokens` as MTP tokens after one guaranteed target token. | Preserves the current engine width convention. |
| MTP-6 | Use one composite checkpoint across target dense/CSA state and MTP state. | A rejected draft must atomically restore every cache stream. |
| GPU-1 | CPU correctness and goldens precede CUDA. | Sparse state bugs are easier to isolate before device residency and graph capture. |

**Owner verdict (2026-07-14):** All decisions CONFIRMED except CSA-7 — the package now DEFAULTS to native_csa_required and REJECTS dense fallback (see §10 Q11 resolution). Numerical source of truth = official HF deepseek-ai/DeepSeek-V4-Flash reference (inference/model.py + inference/kernel.py); goldens dumped from it in Phase 0. llama.cpp is NOT faithful (Flash-Attn disabled, DSV4 graph WIP) and must not be used for goldens; mobius is our exporter, not ground truth.

## 10. Open questions and risks

1. **Official CSA equations:** PR #405 preserves tensor names and shapes but not
   compressor/indexer computation. Which implementation and commit is the
   numerical source of truth?
   **Resolution:** The numerical source of truth is the official HF `deepseek-ai/DeepSeek-V4-Flash` reference (`inference/model.py` + `inference/kernel.py`). Run it to dump and freeze layout-v1 golden intermediate tensors in Phase 0. NeMo (Megatron-Bridge/AutoModel) and arXiv 2606.19348 are cross-checks only; llama.cpp is not faithful (Flash-Attn disabled, DSV4 graph WIP) and must not be used for goldens; mobius is the exporter, not ground truth.
2. **Ratio-128 semantics:** does HCA attend every compressed record, or is there
   an additional selection rule not represented by the exported indexer
   tensors?
   **Resolution:** Pin the behavior from the official reference goldens; do not presuppose “attend all” versus an additional selection rule.
3. **Cache record layout:** what exactly do `CW`, `ICW`, and the partial carry
   contain? The op cannot be registered until layout v1 has golden vectors.
   **Resolution:** Pin `CW`, `ICW`, and carry layout from official reference goldens; never infer it from mobius tensor names.
4. **Top-k ties and causality:** what stable tie order, duplicate policy, and
   boundary masking does the official ratio-4 indexer require?
   **Resolution:** Pin ratio-4 top-k tie order, duplicate policy, and causal boundary masking from official reference goldens; never infer them from mobius tensor names.
5. **Compressed RoPE:** which values are rotated before compression, and what
   state must be retained to reproduce `compress_rope_theta` exactly?
   **Resolution:** Pin compressed-RoPE rotated values and retained state from official reference goldens; never infer them from mobius tensor names.
6. **MTP recurrent state:** is post-layer pre-`_hc_head` HC state the official
   next-iteration state? If not, Mobius must export the correct state explicitly.
   **Resolution:** Pin the official recurrent MTP state from official reference goldens; never infer it from mobius tensor names or by broadcasting `mtp_hidden`.
7. **MTP cache lifetime:** is the sidecar KV proposal-local, accepted-prefix
   persistent, or shared with target/CSA state?
   **Resolution:** Pin sidecar KV lifetime from official reference goldens; never infer it from mobius tensor names or by broadcasting `mtp_hidden`.
8. **Weight sharing format:** should metadata reference target initializer names,
   named model-package components, or both? How are quantized/tied embeddings
   represented without copying?
   **Resolution:** Support both: metadata prefers named model-package components and falls back to target initializer names. Reference tied and quantized embeddings zero-copy; do not duplicate raw f32 weights.
9. **Verification width:** can every target decode path verify `1+k` tokens in
   one forward, or must static/native runners use a sequence of single-token
   steps while preserving identical acceptance?
   **Resolution:** The contract requires `1+k` tokens to be verified in one forward. Static/native runners may degrade to a sequence of single-token steps, but must produce bit-identical greedy-token acceptance; enforce this with an equivalence-test gate.
10. **Batching:** may different batch rows have different compression/index
    cursor lengths? The v1 cache layout and CUDA kernel need a clear answer.
    **Resolution:** v1 requires equal-length compression/index cursors within a batch for a regular cache layout and simple CUDA kernel. Ragged per-row cursor lengths are an immediate fast-follow, modeled on vLLM PagedAttention per-row block-table indirection plus DeepSeek-V4 per-row `(offset,length)` ragged gather. `SparseKvGather`, per-forward cursor journals, and metadata state groups already accommodate per-row cursor lengths, so ragged support is an extension rather than a rewrite.
11. **Fallback policy:** is the dense fallback acceptable for user-selected
    compatibility, given that it can erase the intended long-context memory
    advantage and is not officially numerically equivalent?
    **Resolution:** CSA-7 is flipped: the package defaults to `native_csa_required` and rejects dense fallback. The portability cost is accepted to prevent silent degradation of the long-context memory advantage.
12. **Shared GLM primitive boundary:** is `SparseKvGather` sufficient common
    ground, or should DeepSeek CSA and GLM IndexShare expose separate fused ops?
    **Resolution:** `SparseKvGather` is the shared correctness/reuse primitive. DeepSeek CompressedSparseAttention and GLM IndexShare DSA each get their own fused production op because their selection semantics differ and must not be coupled.
13. **Upstream path:** should the private op be designed immediately as an ORT
    contrib proposal, or incubated until DeepSeek and GLM contracts both
    stabilize?
    **Resolution:** Incubate in the private `pkg.nxrt` domain, as with `BlockQuantizedMatMul`. Propose ORT contrib only after both DeepSeek and GLM contracts stabilize and v1 layout and goldens are frozen; tracking this upstream is optional and non-urgent.
14. **Acceptance tolerance:** what tensor/logit tolerance is acceptable for
    f16/bf16 CPU/CUDA while still requiring exact greedy token identity?
    **Resolution:** Greedy final tokens must be bit-identical: exact integer argmax is a hard correctness gate. Intermediate compressor, index, and attention f16/bf16 tolerances must be calibrated from official goldens by measuring the real error distribution, not guessed, and set per layer for localizability.

## 11. Greenlight requested

**Greenlight granted (owner @justinchuby, 2026-07-14).** Phase 0 may begin: run the official DeepSeek reference, dump and freeze layout-v1 goldens, and freeze the CSA/MTP state-transition contract before any kernel work.

## Frozen Numerical Contract (from official deepseek-ai/DeepSeek-V4-Flash, 2026-07-14)

### Source snapshot and notation

This section is pinned to the official Hugging Face revision
[`60d8d70770c6776ff598c94bb586a859a38244f1`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/60d8d70770c6776ff598c94bb586a859a38244f1).
The Hugging Face API identifies that revision as the model repository head. The
GitHub contents endpoint for `deepseek-ai/DeepSeek-V4-Flash` returned HTTP 404,
so the files below were fetched from the pinned Hugging Face revision. No
`configuration_deepseek.py` is present in the official file list.

| Source | Governing ranges | SHA-256 |
|---|---|---|
| [`inference/model.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py) | [183-251](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L183-L251), [255-377](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L255-L377), [380-543](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L380-L543), [647-809](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L647-L809) | `ce962f1face79d4f633d36436576214057a7e11443c9789935e1deb5c6cd1d71` |
| [`inference/kernel.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/kernel.py) | [41-125](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/kernel.py#L41-L125), [128-200](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/kernel.py#L128-L200), [277-368](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/kernel.py#L277-L368), [372-438](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/kernel.py#L372-L438) | `59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2` |
| [`inference/generate.py`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/generate.py) | [19-69](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/generate.py#L19-L69), [90-103](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/generate.py#L90-L103) | `d4d443c0be8499b20ae5eaff0a623df02f47a8309be6feeba4eb4e0eeb5342c3` |
| [`inference/config.json`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/config.json) | [1-34](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/config.json#L1-L34) | `6cc6f816ca73a8d38750194e330398e4f6955b4b45f674f7d29c96da14ccb733` |
| [`config.json`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/config.json) | [1-67](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/config.json#L1-L67) | `b628e63398a645abc711d92207f8737dd8140f7a4ef1e0a5b3616019e0ddd818` |
| [`inference/requirements.txt`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/requirements.txt) | [1-5](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/requirements.txt#L1-L5) | `857e0b8b58e41cabe16e55bf4ab7ff791677c53b25f0f3e104ef85227cd11eab` |

Notation:

```text
B   batch
S   query/input sequence length for this forward
H   hidden size = 4096
N   query heads = 64; Nl = N / tensor_parallel_world_size
D   attention head dimension = 512
RD  rotary dimension = 64; ND = D - RD = 448
W   sliding-window capacity = 128
R   compression ratio, 4 or 128
I   index heads = 64; Il = I / tensor_parallel_world_size
ID  index head dimension = 128
K   index_topk = 512
C   hc_mult = 4
T   model activation/cache dtype, BF16 in the reference runner
```

For every `RMSNorm`, the source performs

```text
RMSNorm_w(z) =
  cast_T(w_fp32 * z_fp32 / sqrt(mean(z_fp32^2, last_dim) + 1e-6)).
```

This FP32 normalization and cast back to the input dtype are explicit in
`model.py` [183-196](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L183-L196).

### CSA: query, current KV, and RoPE

For hidden input `x : T[B,S,H]`, absolute positions
`p = start_pos .. start_pos+S-1`, and each attention layer:

```text
qr = RMSNorm(Wq_a x)                                  # T[B,S,1024]
q  = reshape(Wq_b qr, [B,S,Nl,D])                    # T[B,S,Nl,D]
q  = q * rsqrt(mean(q^2, -1, keepdim=True) + 1e-6)  # no explicit FP32 cast
q[..., ND:D] = RoPE(q[..., ND:D], p)                 # FP32 complex multiply -> T

kv = RMSNorm(Wkv x)                                   # T[B,S,D], one shared KV head
kv[..., ND:D] = RoPE(kv[..., ND:D], p)               # FP32 complex multiply -> T
kv[..., 0:ND] = fake_fp8_dequant(kv[..., 0:ND])       # in place, still T
```

These operations and their order are fixed by `model.py`
[484-506](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L484-L506).
`q` receives the learned `q_norm` and then the second unweighted RMS
renormalization shown above.

RoPE views adjacent pairs in the last `RD=64` dimensions as complex numbers,
multiplies by `exp(i*p*freq_j)`, and copies the FP32 result back to the original
tensor dtype. Inverse RoPE uses the complex conjugate
(`model.py` [232-244](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L232-L244)).
For compressed layers, `base=compress_rope_theta=160000`,
`original_seq_len=65536`, `factor=16`, `beta_fast=32`, and `beta_slow=1`.
For frequency pair `j`:

```text
base_freq_j = base^(-2j/RD)
low  = max(floor(RD * log(original_seq_len/(beta_fast*2*pi))
                 / (2*log(base))), 0)
high = min(ceil(RD * log(original_seq_len/(beta_slow*2*pi))
                 / (2*log(base))), RD-1)
r_j  = clamp((j-low)/(high-low), 0, 1)
freq_j = base_freq_j * (1-r_j) + (base_freq_j/factor) * r_j
```

For the pinned constants this evaluates to `low=15`, `high=25`. The exact
clamping is in `model.py`
[199-229](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L199-L229).
Ratio-0 layers disable YaRN and use `rope_theta=10000`
(`model.py` [475-482](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L475-L482)).

The in-place FP8 simulation is per 64 non-RoPE dimensions. With the official
inference config, its scale is power-of-two:

```text
amax  = max(max(abs(block)), 1e-4)
scale = 2^ceil(log2(amax/448))
y     = cast_BF16(cast_FP8_E4M3(clamp(x/scale, -448, 448)) * scale)
```

The kernel computes `amax` and scaling in FP32
(`kernel.py` [41-100](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/kernel.py#L41-L100)).

### CSA: learned compression and exact cache/carry layout

The compressor source is `model.py`
[279-377](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L279-L377).
It first casts `x` to FP32 and evaluates FP32 projections:

```text
coff = 2 if R == 4 else 1
u_t = Wcompress_kv x_t       # FP32[coff*D]
g_t = Wcompress_gate x_t     # FP32[coff*D]
a_p = APE[p]                 # FP32[coff*D], p = absolute_position mod R
```

The softmax gate is **per output dimension**, not one scalar per token.

For `R=128`, compressed record `z_c : FP32[D]` for complete block
`t=cR..cR+R-1` is:

```text
alpha_{t,d} = exp(g_{t,d}+a_{t mod R,d})
              / sum_{u=cR}^{cR+R-1} exp(g_{u,d}+a_{u mod R,d})
z_{c,d} = sum_{t=cR}^{cR+R-1} alpha_{t,d} * u_{t,d}.
```

For `R=4`, split `u_t` and `g_t+a_p` into left/right `D`-wide channels.
Record `c` pools `2R` candidates:

```text
J_c =
  {(t=(c-1)R+p, left)  | p=0..R-1}, if c > 0
  union
  {(t=cR+p, right)     | p=0..R-1}.

alpha_{j,d} = softmax over j in J_c of (g_{j,d} + a_{p(j),channel(j),d})
z_{c,d} = sum_{j in J_c} alpha_{j,d} * u_{j,d}.
```

For `c=0`, the missing previous-block candidates are represented exactly as
zero values with `-inf` scores. The governing transform is:

```python
new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
```

(`model.py` [307-314](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L307-L314)).

After FP32 pooling, the record is finalized in this exact order:

```text
z = cast_BF16(z)
z = RMSNorm(z)                                     # FP32 internal, BF16 output
z[..., D-RD:D] = RoPE(z[..., D-RD:D], block_start)
if attention compressor:
    z[..., 0:D-RD] = fake_fp8_dequant(z[..., 0:D-RD])  # groups of 64
if index compressor:
    z = Hadamard(z, scale=1/sqrt(D))
    z = fake_fp4_dequant(z)                            # groups of 32
```

The compressed RoPE position is the **uncompressed block start** `cR`, not
compressed index `c`: prefill uses `freqs_cis[0:cutoff:R]`; decode uses
`start_pos+1-R` (`model.py` [362-376](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L362-L376)).

Actual reference state, replacing the placeholder `CW`/`ICW` descriptions in
§4.3, is:

| Stream | Persistent records | Incremental `kv_state` | Incremental `score_state` |
|---|---|---|---|
| attention, `R=4` | `T[B,max_seq_len/4,D]` | `FP32[B,8,2D]` | `FP32[B,8,2D]`, initialized `-inf` |
| attention, `R=128` | `T[B,max_seq_len/128,D]` | `FP32[B,128,D]` | `FP32[B,128,D]`, initialized `-inf` |
| index key, `R=4` | `T[B,max_seq_len/4,ID]` | `FP32[B,8,2ID]` | `FP32[B,8,2ID]`, initialized `-inf` |

For an overlapping `R=4` decode carry, slots `0:4` contain the last completed
block's full `2D` projections and slots `4:8` contain the current block. At a
boundary, the source concatenates `state[:,0:4,:D]` with
`state[:,4:8,D:]`, performs the dimension-wise FP32 softmax pool, then shifts
the current full block into slots `0:4`
(`model.py` [344-359](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L344-L359)).
For `R=128`, slot `absolute_position mod 128` is updated and one record is
emitted exactly when `(absolute_position+1) mod 128 == 0`.

Prefill emits `floor(S/R)` records. An incomplete suffix emits no record and is
copied into the carry. For `R=4`, the final complete block is also retained in
the previous-block half so the next record can overlap it
(`model.py` [325-342](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L325-L342)).

### CSA: ratio-4 index scoring and selection

The ratio-4 indexer is fixed by `model.py`
[380-433](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L380-L433).
It has an independent `R=4`, `ID=128` compressor with the overlap rule above.
For `qr : T[B,S,1024]` from the main query path:

```text
qi = reshape(Windex_q qr, [B,S,Il,ID])               # T
qi[..., ID-RD:ID] = RoPE(qi[..., ID-RD:ID], p)
qi = Hadamard(qi, scale=1/sqrt(ID))
qi = fake_fp4_dequant(qi)                             # T, group size 32

ki = index_compressor(x)                              # T[B,Ctotal,ID]
w  = Windex_weight x * (ID^(-1/2) * I^(-1/2))       # T[B,S,Il]

dot_{b,s,i,c} = sum_d qi[b,s,i,d] * ki[b,c,d]
score_{b,s,c} = sum_i relu(dot_{b,s,i,c}) * w[b,s,i]
```

Tensor-parallel ranks sum `score` with `all_reduce`. There is no additional
scale on `dot`; the only explicit scale is on `w`. Neither the `einsum` nor the
head reduction has an explicit FP32 cast in Python.

For prefill (`start_pos=0`), candidate compressed record `c` is causal iff

```text
c < floor((s+1)/R).
```

All other scores become `-inf`. The source then executes exactly:

```text
topk_idxs = score.topk(min(512, floor((start_pos+S)/R)), dim=-1)[1]
```

with PyTorch defaults `largest=True, sorted=True`. Invalid prefill selections
are rewritten to `-1`; valid compressed indices receive an offset equal to `S`
for the prefill-local concatenated KV tensor or `W=128` for the persistent
decode cache. The fake-FP4 operation is:

```text
amax  = max(max(abs(block32)), 6*2^-126)
scale = 2^ceil(log2(amax/6))
y     = cast_BF16(cast_FP4_E2M1(clamp(x/scale, -6, 6)) * scale)
```

with FP32 `amax` and scale computation
(`kernel.py` [128-200](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/kernel.py#L128-L200)).

### CSA: candidate list, sparse softmax, sink, and output

The candidate index list is assembled in this order
(`model.py` [255-276](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L255-L276),
[501-515](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L501-L515)):

```text
ratio 0:   [causal sliding-window indices]
ratio 4:   [causal sliding-window indices, learned top-k compressed indices]
ratio 128: [causal sliding-window indices, every completed compressed index]
```

For prefill query `s`, the window indices are ascending
`max(0,s-W+1)..s`, padded by `-1`. For decode after the ring is full, the
physical ring order is
`[(start_pos mod W)+1 .. W-1, 0 .. (start_pos mod W)]`, oldest to newest.
For ratio 128, compressed indices are ascending
`0..floor((absolute_query_position+1)/128)-1`. Thus ratio 128 has no second
selection rule: it attends every completed compressed record.

Prefill sparse-attention KV is:

```text
ratio 0:   kv_all : T[B,S,D]
ratio > 0: kv_all = concat(current_dense_kv, compressed_kv, dim=1)
                     : T[B,S+floor(S/R),D].
```

Decode uses the persistent
`T[B,W + max_seq_len/R,D]` buffer: the first `W` records are the dense ring and
the remaining records are compressed. The selected indices are converted to
`int32` immediately before the kernel
(`model.py` [507-533](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L507-L533)).

The sparse kernel contract is:

```text
q           BF16[B,S,Nl,D]
kv_all      BF16[B,Nkv,D]
attn_sink   FP32[Nl]
indices     INT32[B,S,L], where -1 is invalid
output      BF16[B,S,Nl,D]
softmax_scale = D^(-1/2) = 512^(-1/2)
```

For each `(b,s,h)`, indices are processed in their supplied order in blocks of
64. The exact online algorithm is:

```text
m = -inf_fp32
Z = 0_fp32
O = zeros_fp32[D]

for each 64-index block:
    ell_j = dot_BF16(q, kv[index_j]) * D^(-1/2)   # FP32 score accumulator
    ell_j = -inf if index_j == -1
    m_new = max(m, max_j ell_j)
    r = exp(m - m_new)
    p_j = exp(ell_j - m_new)                      # FP32
    Z = Z*r + sum_j p_j                           # FP32
    O = O*r + sum_j cast_BF16(p_j) * kv[index_j] # BF16 GEMM, FP32 O
    m = m_new

Z = Z + exp(attn_sink[h] - m)                    # FP32; sink is not in max()
O = O / Z
output = cast_BF16(O)
```

Invalid records load a zero value and receive `-inf` score. The learned sink
contributes denominator mass only; it has no value vector. Note that the
implementation adds the sink only after the online maximum has been computed,
rather than including it in `m`. These details, including the BF16 cast of
`p_j` before the value GEMM, are fixed by `kernel.py`
[293-350](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/kernel.py#L293-L350).

Attention output assembly is:

```text
o[..., D-RD:D] = inverse_RoPE(o[..., D-RD:D], query_position)
o = reshape(o, [B,S,o_groups=8,-1])
u[b,s,g,r] = sum_d o[b,s,g,d] * Wo_a[g,r,d]      # BF16 einsum
y = Wo_b(flatten(u, groups_and_rank))             # T[B,S,H]
```

(`model.py` [534-543](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L534-L543)).

### MTP block structure and the contract that is actually present

The official configuration has exactly one next-token-prediction layer:
`num_nextn_predict_layers=1`. The inference model constructs one `MTPBlock`
after the 43 target blocks and assigns it the **same module objects** as the
target embedding and LM head:

```python
self.mtp[-1].embed = self.embed
self.mtp[-1].head = self.head
```

(`model.py` [769-799](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L769-L799)).
There are no per-step embedding or unembedding copies. The embedding and LM
head are nevertheless distinct from each other (`tie_word_embeddings=false`);
MTP shares each with its corresponding target component.

One MTP invocation is defined for:

```text
x_in      T[B,S,C,H]   target/HC state supplied by the caller
input_ids int64[B,S]
start_pos scalar absolute position
```

and executes:

```text
e = target_embedding(input_ids)                  # T[B,S,H]
e = RMSNorm_mtp_embedding(e)
h = RMSNorm_mtp_hidden(x_in)                     # T[B,S,C,H], per lane
u = W_e e[:, :, None, :] + W_h h                 # T[B,S,C,H], embedding broadcasts
x_block = MTP_Block_43(u, start_pos, input_ids)  # T[B,S,C,H]
logits = shared_target_head(x_block)              # FP32[B,vocab], final position only
```

The ordering is exact from `model.py`
[738-766](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L738-L766).
Compression schedule entry 43 is `0`, so this MTP block uses its own
sliding-window attention cache and no compressed-attention/index cache.

The inherited block updates HC state twice, once around attention and once
around MoE. For `r : T[B,S,C,H]`, it flattens the last two dimensions to FP32,
computes

```text
rho = rsqrt(mean(flatten(r)^2) + 1e-6)
mixes = W_hc flatten(r) * rho                     # FP32[B,S,(2+C)C]
pre_j  = sigmoid(mixes_j*s0 + base_j) + 1e-6
post_j = 2*sigmoid(mixes_{C+j}*s1 + base_{C+j})
comb   = Sinkhorn(mixes_tail*s2 + base_tail)      # FP32[C,C], 20 iterations
reduced = sum_j pre_j * r_j                       # T[B,S,H]
updated_j = post_j * branch_output
            + sum_i comb_{i,j} * r_i              # T[B,S,C,H]
```

The Sinkhorn initialization is row-softmax plus epsilon, followed by column
normalization, then 19 row/column normalization iterations
(`kernel.py` [372-438](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/kernel.py#L372-L438)).
The block application is fixed by `model.py`
[647-700](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L647-L700).

The shared head collapses HC lanes with:

```text
flat = flatten(x_block).float()
rho = rsqrt(mean(flat^2) + 1e-6)
pre_j = sigmoid((W_head flat * rho)_j * scale + base_j) + 1e-6
collapsed = cast_T(sum_j pre_j * x_block_j)
normalized = RMSNorm_target(collapsed)
logits = W_lm_fp32 * normalized[:, -1, :].float()
```

and therefore returns FP32 logits for only the final sequence position
(`model.py` [703-735](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L703-L735)).

### MTP generation, verification, and acceptance

The pinned official reference contains **no MTP inference loop**. This is a
source fact, not an inferred omission:

- `Transformer.forward` runs only `self.layers` and the target head; it never
  calls `self.mtp` (`model.py`
  [801-809](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L801-L809)).
- `generate.py` performs ordinary autoregressive generation and never accesses
  `model.mtp` (`generate.py`
  [27-69](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/generate.py#L27-L69)).
- `MTPBlock.forward` returns only logits, not `x_block`; consequently the
  official API exposes no recurrent MTP hidden state to feed a second MTP step.

The only official greedy selection equation available is:

```text
logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
next_token = argmax(logits, dim=-1)                # when temperature <= 0
next_token = prompt_token if cur_pos is still prompt, else next_token
stop when every generated row emits eos
```

For temperature sampling, the reference instead computes FP32
`softmax(logits/max(temperature,1e-5))` and uses exponential-noise
Gumbel-max (`generate.py`
[19-24](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/generate.py#L19-L24)).
There is no official draft width, target verification forward, longest-prefix
acceptance equation, correction-token rule, bonus-token rule, or MTP cache
lifetime in these files. Those items are therefore **not frozen** by this
snapshot and must not be invented from the existing runtime loop.

Per §10 Q14, the runtime correctness gates are:

```text
greedy selected token IDs: bit-identical integer equality, mandatory
FP16/BF16 compressor/index/attention intermediates:
    tolerance allowed only after per-layer calibration against official goldens
    from this pinned source; no guessed global tolerance
```

### Configuration constants that pin this contract

Values are from official `config.json`
[10-66](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/config.json#L10-L66)
and executable `inference/config.json`
[1-34](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/config.json#L1-L34):

```text
hidden_size / dim                  4096
num_hidden_layers / n_layers      43
num_nextn_predict_layers          1 (ModelArgs default n_mtp_layers=1)
num_attention_heads / n_heads     64
num_key_value_heads               1
head_dim                          512
qk_rope_head_dim / rope_head_dim  64
non-RoPE head dimension           448
q_lora_rank                       1024
o_groups                          8
o_lora_rank                       1024
sliding_window / window_size      128
index_n_heads                     64
index_head_dim                    128
index_topk                        512
hc_mult                           4
hc_sinkhorn_iters                 20
hc_eps / rms_norm_eps             1e-6
rope_theta                        10000
compress_rope_theta               160000
YaRN original length              65536
YaRN factor                       16
YaRN beta_fast / beta_slow        32 / 1
torch activation dtype            bfloat16
default projection-weight dtype   FP8 E4M3, 128x128 weight blocks
default linear activation block   128
compressor Wkv/Wgate dtype         FP32
APE, RMS weights, sink, LM head   FP32
index weights_proj dtype           BF16
compression ratios                4 (overlap factor 2), 128 (factor 1)
index Q/K simulation              FP4 E2M1, block size 32, power-of-two scale
attention non-RoPE simulation     FP8 E4M3, block size 64, power-of-two scale
sparse-attention tile             64 selected indices
```

The 44-entry compression schedule covers 43 target layers plus the one MTP
layer:

```text
layer 0=0, layer 1=0,
target layers 2..42 alternate 4,128 and end at layer 42=4,
MTP layer 43=0.
```

Thus the target has 21 ratio-4 layers, 20 ratio-128 layers, and two ratio-0
layers. The MTP layer is the third ratio-0 schedule entry. The HF config's
maximum position is `1,048,576`; the standalone reference runner's
`ModelArgs.max_seq_len` default is `4096` unless the caller overrides it
(`model.py`
[34-80](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L34-L80)).

### Open deltas

1. **MTP recurrence is absent.** The source computes `x_block` but returns only
   logits. It does not establish that post-block, pre-head `x_block` is the
   state for the next MTP step. Mobius must not label that tensor `mtp_state`
   as an official recurrence until DeepSeek publishes or confirms the loop, or
   an official golden demonstrates the transition.
2. **MTP verification/acceptance is absent.** `generate.py` is target-only
   autoregressive decoding. The proposed reuse of
   `generate_speculative_loop`, longest-prefix acceptance, correction/bonus
   behavior, and proposal-local versus accepted-prefix MTP KV are runtime
   design choices, not yet official DeepSeek numerical contract.
3. **Top-k ties are not portable.** The only rule in source is CUDA
   `torch.topk(..., largest=True, sorted=True)`. PyTorch does not promise stable
   indices for equal values, and the official
   [`inference/requirements.txt`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/requirements.txt#L1-L5)
   specifies `torch>=2.10.0` rather than an exact build. Layout-v1 goldens must
   record tie cases, or the contract needs an explicit DeepSeek-approved tie
   rule before a portable CPU implementation can claim bit identity.
4. **Hadamard dependency is unpinned.** `rotate_activation` calls external
   `fast_hadamard_transform.hadamard_transform(x, scale=D^-0.5)`
   (`model.py`
   [247-251](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/60d8d70770c6776ff598c94bb586a859a38244f1/inference/model.py#L247-L251)),
   while `inference/requirements.txt` gives no version. The exact transform
   ordering/build must be captured in the golden environment.
5. **Some backend arithmetic is implicit.** Index `einsum`, weighted head
   reduction, BF16 output projection `einsum`, and `torch.topk` have no explicit
   accumulator/backend contract in Python. Official GPU goldens, CUDA/PyTorch
   versions, and per-layer error distributions must accompany the equations.
6. **Mobius CSA layouts must be corrected to the source.** Stored compressed
   attention records are width `D`, stored index records are width `ID`, and
   overlap carries are raw FP32 `[B,2R,2D]` / `[B,2R,2ID]`, not a single opaque
   `CW` record width. Mobius must emit the projected `u`, `g`, and APE streams
   in the exact channel order above and preserve FP32 compression before the
   BF16 norm/RoPE/quantization sequence.
7. **Mobius MTP export needs two separately justified outputs.** A shared LM
   head input may use the official normalized collapsed hidden value, but an
   iterative sidecar needs an official recurrent HC state. The current exporter
   must not reconstruct `[B,S,C,H]` by broadcasting `[B,S,H]`, and it must not
   claim official acceptance semantics merely because the target embedding and
   head are shared.

# Inc3a real-model mask / cuDNN-ReduceSum check

**Question (from review):** In Inc3a the CUDA parity fixture declares
`attention_mask`/`position_ids` as **unconsumed** graph inputs, because CUDA
cuDNN rejected the fixture's degenerate all-axes `ReduceSum`-to-scalar. A *real*
model (Qwen3.6-27B / 35B-A3B) **consumes** the mask/position through real ops.
Is Inc3a's green parity partly an artifact of the fixture avoiding the
consumed-mask path? Would the cuDNN `ReduceSum` limitation (or any mask/position
handling) actually block a real model's native CUDA decode through the pipeline?

## Verdict: **fixture artifact — NOT a real-model blocker.**

The cuDNN failure is specific to reducing to a **rank-0 scalar** (all axes, no
dims left). Real decoders reduce the mask over a **specific axis**, leaving a
non-degenerate result, which cuDNN handles — proven bit-exact on a real model.

## Evidence

### 1. Real model, consumed mask, native CUDA == ORT CUDA (32 tokens) — GREEN
`qwen3_0_6b_native_cuda_e2e` on `qwen3-0.6b-int4-cuda-postfix`, GPU device 4:

```
Qwen3-0.6B native CUDA lock OK: tokens=[12095, 11, 323, 279, 6722, 315, 15344,
374, 21718, 13, 576, 6722, 315, 9625, 374, 1083, 279, 6722, 315, 279, 3146, 429,
702, 279, 1429, 1251, 13, 576, 6722, 315, 15344, 374]
test ... ok. 1 passed
```

This real int4 decoder **consumes `attention_mask` through a real op on CUDA**
and its 32 greedy tokens match ORT-CUDA exactly. Consumed-mask native CUDA decode
demonstrably works.

### 2. Graph inspection — how a real decoder consumes the mask
`qwen3-0.6b-int4-cuda-postfix/model.onnx` (`onnx.load`, 28 `GroupQueryAttention`
layers). Direct consumers:

- `attention_mask` -> **`ReduceSum`** (`model/ReduceSum_node_6`), then `Sub` ->
  `Cast` -> `GroupQueryAttention` seqlens inputs (slots 5/6 = total/​k seqlens).
- `position_ids` -> **0 direct consumers** (RoPE uses `cos_cache`/`sin_cache`;
  GQA applies rotary internally from the total-sequence length).

The mask `ReduceSum` node:

```
op:      ReduceSum
inputs:  ['attention_mask', 'const_1d_0']   # const_1d_0 = [1]  (axes input)
attr:    keepdims = 0
```

It reduces `attention_mask [batch, total_seq]` over **axis 1** (keepdims=0) ->
`[batch]` seqlens. That is a reduction over a real (>1) axis leaving a **rank-1,
non-empty** tensor — exactly the case cuDNN supports (and #1 above proves runs).

### 3. Why the Inc3a fixture tripped cuDNN (the artifact)
The Inc3a fixture consumed the mask with a **zero-term trick**: `Cast` ->
`ReduceSum` **over all axes** -> rank-0 scalar `[]` (and, when reducing the
1-token `position_ids [1,1]` during decode, a degenerate `[1,1]->[1,1]`). cuDNN's
`cudnnReduceTensor` returns `CUDNN_STATUS_BAD_PARAM` for those degenerate
all-axes / no-op reductions. No real decoder does an all-axes reduce of the mask;
it reduces over the sequence axis only (evidence #2). So the failure never
occurs on a real model. The Inc3a fixture therefore declares the mask/position
as **unconsumed** inputs (still bound + populated on-device each step; the binding
path is exercised) purely to sidestep an artificial op that is not part of any
real decode graph.

### 4. Inc3a did not touch the mask path
The CUDA mask binding is `bindings[0]`, allocated + populated via `extend_mask`
every step (`native_decode/cuda.rs`), **identical** for the token-id and
`inputs_embeds` sequence sources — Inc3a only changed the *sequence* binding
(Int64 token `[1,1]` -> float embeds `[1,1,hidden]`). Consumed-mask handling is
shared, unchanged, and covered by the real-model CUDA e2e (evidence #1).

## Consequence for sequencing
No dedicated cuDNN-ReduceSum / mask-handling fix PR is warranted: real decoders
route the mask through an axis-specific `ReduceSum` -> `GroupQueryAttention`
seqlens path that already works on the native CUDA EP. **Proceeding to Inc3b
(generic routed ports) as planned; the mask path is not a blocker.**

### 5. Direct intersection proof — `inputs_embeds` + CONSUMED mask on CUDA
To close the one gap not covered by an existing test (a single graph that both
uses `inputs_embeds` and consumes the mask on CUDA), I rebuilt the Inc3a fixture
decoder to **consume `attention_mask` via the real pattern** — `Cast` ->
`ReduceSum(axes=[1], keepdims=1)` -> `[1,1]` -> `*0` -> broadcast-add into logits
(mirrors qwen3's `attention_mask -> ReduceSum(axes=[1]) -> GQA seqlens`, zero
contribution so tokens stay `[0,5,6,7]`). Native-CPU vs native-CUDA (device 4)
through the pipeline:

```
test native_cuda_pipeline_decoder_matches_cpu_token_ids ... ok   # both [0,5,6,7]
```

GREEN. The axis-specific `ReduceSum` that real decoders use runs fine on the
native CUDA EP together with the `inputs_embeds` sequence binding — confirming
Inc3a's green parity is **not** an artifact of avoiding the consumed-mask path.
(This variant was a throwaway probe for evidence and was not committed, to keep
the Inc3a fixture minimal; it can be promoted to a permanent regression fixture
if desired.)

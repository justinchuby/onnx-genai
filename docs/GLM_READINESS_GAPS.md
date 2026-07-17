# GLM-5.2 / DeepSeek-V4-Flash native-runtime readiness gaps

**Audit point:** `main` at `4ff24cb`, 2026-07-18
**Scope:** native `onnx-runtime-session` using one in-tree CPU or CUDA execution
provider. This is not an ORT EP-fallback assessment.

## Executive conclusion

The previous audit (`a7a1685`, 2026-07-17) correctly identified a large CUDA
standard-op loading gap. **That finding is superseded.** CUDA now registers the
standard ops required by the portable GLM-5.2 graph: opset-23/24
`Attention`, `RotaryEmbedding`, `RMSNormalization`, `TopK`, `CumSum`,
`GatherElements`, `ScatterElements`, `Where`, and the complete movement /
construction family ([CUDA registry: `kernels/mod.rs:94-178`](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L94-L178),
[184-303](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L184-L303),
[416-489](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L416-L489)).
For the stated Mobius #404 portable graph, there is no remaining *missing
standard-op registration* that is a CUDA graph-loading P0.

There is one non-negotiable qualification: the checked data inputs of standard
`Attention` and `RotaryEmbedding` must be f32 in CUDA's **claim-time**
`input_dtypes` contract
([`provider.rs:167-179`](../crates/onnx-runtime-ep-cuda/src/provider.rs#L167-L179);
[`standard_attention.rs:57-75`](../crates/onnx-runtime-ep-cuda/src/kernels/standard_attention.rs#L57-L75);
[`rotary_embedding.rs:42-56`](../crates/onnx-runtime-ep-cuda/src/kernels/rotary_embedding.rs#L42-L56)).
Because CUDA-only placement rejects an unclaimed node, casts must feed f32
into those operators (for example, Q/K/V and RoPE X/cos/sin). A uniform f32
activation graph is the simplest, sufficient, and conservative export profile,
but it is not a universal loading requirement: CUDA supports f16/bf16
arithmetic elsewhere, so a mixed graph with the required casts can load.

Two milestones must remain distinct:

1. **CUDA-only graph loading:** the *standard-op registration* gap is closed.
   The actual portable graph still needs its custom/runtime boundaries:
   `pkg.nxrt::BlockQuantizedMoE` has no CPU or CUDA kernel/registration;
   IndexShare needs selected-token attention (the standard additive-mask
   fallback is correct but dense over the cache); and GLM MTP sidecar discovery
   plus its GLM-specific HC/persistent-state lifecycle are not wired.
2. **Smooth/performance-ready CUDA execution:** in addition to those custom
   boundaries, CUDA needs GPU-native, non-host-staged `Attention` and
   `RotaryEmbedding`. The current correctness-first implementations
   device-to-host materialize inputs, compute on the host, and upload results,
   so they are a throughput blocker even after the custom boundaries land
   ([`standard_attention.rs:155-208`](../crates/onnx-runtime-ep-cuda/src/kernels/standard_attention.rs#L155-L208),
   [`595-622`](../crates/onnx-runtime-ep-cuda/src/kernels/standard_attention.rs#L595-L622);
   [`rotary_embedding.rs:164-176`](../crates/onnx-runtime-ep-cuda/src/kernels/rotary_embedding.rs#L164-L176)).

“Standard-op graph-loading gap closed” therefore does not mean
“performance-ready at million-token context.”

## What the exported graph requires

Mobius #404's GLM-5.2 portable target uses standard opset 24, standard
`Attention`/`RotaryEmbedding`/`RMSNormalization`, a portable 256-expert MoE
decomposition, and IndexShare `TopK` converted to an additive attention mask.
The portable expert loop is correctness-first rather than selected-expert
execution. The external MTP sidecar is a separate decoder graph, not an ONNX
`MTP` operator.

The table below audits the standard operators relevant to that path against the
current CUDA registry and concrete kernel contracts. `✅ native` means a
registered handler covers the f32 GLM usage; `⚠️ constrained` means registered
but with a material contract restriction; `❌ missing` means no handler. Earlier
compatible `since_version` registrations serve opset 24.

## CUDA standard-op coverage for GLM-5.2

| Required standard op | CUDA status | Evidence / constraint |
|---|---:|---|
| `Attention` (23/24) | ⚠️ constrained | Registered for 23 and 24 ([`mod.rs:423-431`](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L423-L431)); checked data inputs are claim-time f32-only. It is host-staged, contiguous-only, and not the selected-token IndexShare boundary. |
| `RotaryEmbedding` (23/24) | ⚠️ constrained | Registered since 23 ([`mod.rs:432-437`](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L432-L437)); checked data inputs are claim-time f32-only; `position_ids` must be int64 ([`rotary_embedding.rs:42-56`](../crates/onnx-runtime-ep-cuda/src/kernels/rotary_embedding.rs#L42-L56), [`95-113`](../crates/onnx-runtime-ep-cuda/src/kernels/rotary_embedding.rs#L95-L113)). |
| `RMSNormalization` | ⚠️ constrained | Registered since 1 ([`mod.rs:473-489`](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L473-L489)); execution requires contiguous f32 X, Scale, and Y ([`normalization.rs:520-537`](../crates/onnx-runtime-ep-cuda/src/kernels/normalization.rs#L520-L537)). |
| `TopK` | ⚠️ constrained | Registered since 10 ([`mod.rs:222-227`](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L222-L227)); f32 values, int64 scalar `K`, contiguous tensors ([`topk.rs:74-101`](../crates/onnx-runtime-ep-cuda/src/kernels/topk.rs#L74-L101)). This matches the f32 IndexShare/router use. |
| `CumSum` | ⚠️ constrained | Registered since 11 ([`mod.rs:216-221`](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L216-L221)); f32 or int64 data and int64 scalar axis ([`cumsum.rs:69-97`](../crates/onnx-runtime-ep-cuda/src/kernels/cumsum.rs#L69-L97)). |
| `Gather` | ✅ native | Registered since 1 ([`mod.rs:194-201`](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L194-L201)); fixed-width data and int32/int64 indices ([`gather.rs:61-107`](../crates/onnx-runtime-ep-cuda/src/kernels/gather.rs#L61-L107)). |
| `GatherElements` | ✅ native | Registered since 11 ([`mod.rs:202-207`](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L202-L207)); fixed-width data and int64 indices ([`indexing.rs:195-234`](../crates/onnx-runtime-ep-cuda/src/kernels/indexing.rs#L195-L234)). |
| `ScatterElements` | ⚠️ constrained | Registered since 11/16 ([`mod.rs:208-215`](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L208-L215)); contiguous f32 or int64 data/updates and int64 indices ([`indexing.rs:335-373`](../crates/onnx-runtime-ep-cuda/src/kernels/indexing.rs#L335-L373)). |
| `Where` | ✅ native | Registered since 1 ([`mod.rs:289-297`](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L289-L297)); bool condition, matching fixed-width branch/output dtypes, full three-way broadcasting ([`where_op.rs:56-103`](../crates/onnx-runtime-ep-cuda/src/kernels/where_op.rs#L56-L103)). |
| `Expand` | ✅ native | Registered since 1 ([`mod.rs:247-252`](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L247-L252)); dtype-agnostic fixed-width movement, with contiguous operands ([`movement.rs:145-164`](../crates/onnx-runtime-ep-cuda/src/kernels/movement.rs#L145-L164)). |
| `Tile` | ✅ native | Registered since 6 ([`mod.rs:298-303`](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L298-L303)); dtype-agnostic fixed-width movement; repeats are host-read int32/int64 metadata ([`movement.rs:178-202`](../crates/onnx-runtime-ep-cuda/src/kernels/movement.rs#L178-L202)). |
| `Concat` | ✅ native | Registered since 1 ([`mod.rs:240-297`](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L240-L297)); contiguous, fixed-width movement. |
| `Reshape` | ✅ native | Registered since 1 ([`mod.rs:253-297`](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L253-L297)); contiguous, fixed-width movement. |
| `Slice` | ✅ native | Registered since 1 ([`mod.rs:259-297`](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L259-L297)); contiguous, fixed-width movement. |
| `Split` | ✅ native | Registered since 1 ([`mod.rs:265-297`](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L265-L297)); contiguous, fixed-width movement. |
| `Squeeze` | ✅ native | Registered since 1 ([`mod.rs:271-297`](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L271-L297)); contiguous, fixed-width movement. |
| `Transpose` | ✅ native | Registered since 1 ([`mod.rs:277-297`](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L277-L297)); contiguous, fixed-width movement. |
| `Unsqueeze` | ✅ native | Registered since 1 ([`mod.rs:283-297`](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L283-L297)); contiguous, fixed-width movement. |
| `Equal`, `Greater`, `GreaterOrEqual`, `Less`, `LessOrEqual` | ⚠️ constrained | All registered ([`mod.rs:146-154`](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L146-L154)); broadcast-capable, but operands are f32/i32/i64 (and bool only for `Equal`), not f16/bf16 ([`pointwise.rs:17-28`](../crates/onnx-runtime-ep-cuda/src/kernels/pointwise.rs#L17-L28)). GLM's integer expert/mask comparisons fit. |
| `Add`, `Sub`, `Mul`, `Div`, `Min`, `Max` | ✅ native | Registered with NumPy broadcasting ([`mod.rs:397-414`](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L397-L414)); f32/f16/bf16 arithmetic. |
| `Cast`, `CastLike`, `Shape`, `Constant` | ✅ native | Registered in the CUDA graph-construction core ([`mod.rs:229-239`](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L229-L239), [`505-513`](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L505-L513)). |

The former claims that CUDA lacked `Attention`, `RotaryEmbedding`, `TopK`,
`Where`, `CumSum`, `ScatterElements`, `GatherElements`, or the movement family
are therefore historical, not current-state findings.

## Precision and kernel contracts

### Conservative CUDA GLM export precision: uniform f32

Use a **uniform f32 activation graph** for a CUDA-only GLM session as the
simplest sufficient and conservative profile. It is not a universal loading
requirement:

- `CudaExecutionProvider::supports_op` receives `input_dtypes` and denies
  standard `Attention` and `RotaryEmbedding` before kernel construction when a
  checked data input is f16/bf16 ([`provider.rs:113-120`](../crates/onnx-runtime-ep-cuda/src/provider.rs#L113-L120),
  [`167-179`](../crates/onnx-runtime-ep-cuda/src/provider.rs#L167-L179)).
- The session rechecks that claim for every concrete kernel cache miss
  ([`executor.rs:176-213`](../crates/onnx-runtime-session/src/executor.rs#L176-L213)).
  Thus their checked data inputs (including Attention Q/K/V and RoPE X/cos/sin)
  must be f32, but casts can satisfy that constraint in an otherwise
  mixed-precision graph.
- `RMSNormalization` is also f32-only, although it is currently rejected in
  execution rather than provider claim selection ([`normalization.rs:520-537`](../crates/onnx-runtime-ep-cuda/src/kernels/normalization.rs#L520-L537)).
- The supporting f32 graph kernels have the same constraint: `TopK` f32
  values, `CumSum` f32/int64, and `ScatterElements` f32/int64
  ([`topk.rs:89-100`](../crates/onnx-runtime-ep-cuda/src/kernels/topk.rs#L89-L100);
  [`cumsum.rs:87-97`](../crates/onnx-runtime-ep-cuda/src/kernels/cumsum.rs#L87-L97);
  [`indexing.rs:353-362`](../crates/onnx-runtime-ep-cuda/src/kernels/indexing.rs#L353-L362)).
- Other standard kernels also make a broadly mixed export more restrictive:
  `Gemm` is f32-only ([`gemm.rs:198-203`](../crates/onnx-runtime-ep-cuda/src/kernels/gemm.rs#L198-L203)),
  as are `LayerNormalization` and related normalization paths
  ([`normalization.rs:312-319`](../crates/onnx-runtime-ep-cuda/src/kernels/normalization.rs#L312-L319),
  [`382-408`](../crates/onnx-runtime-ep-cuda/src/kernels/normalization.rs#L382-L408)),
  plus `ReduceMax`/`ReduceMin` ([`reduce.rs:439-455`](../crates/onnx-runtime-ep-cuda/src/kernels/reduce.rs#L439-L455)).
  These kernels are not proven to be part of the portable GLM graph, but they
  constrain mixed-precision exports that include them.

Conversely, `Where`, `Gather`/`GatherElements`, and movement/construction are
byte-copy kernels over fixed-width data, so they do not themselves exclude
f16/bf16 ([`where_op.rs:72-103`](../crates/onnx-runtime-ep-cuda/src/kernels/where_op.rs#L72-L103);
[`gather.rs:99-107`](../crates/onnx-runtime-ep-cuda/src/kernels/gather.rs#L99-L107);
[`movement.rs:145-164`](../crates/onnx-runtime-ep-cuda/src/kernels/movement.rs#L145-L164)).
That permits f16/bf16 outside the constrained paths; it does not remove the
need to supply f32 inputs to the constrained operators or to any other
f32-only kernels the export uses.

The f32 requirement is consistent with the current quantized-linear path:
CUDA `MatMulNBits` and `BlockQuantizedMatMul` use f32 activations/outputs, so
quantized weights do not imply f16 activations
([`matmul_nbits.rs:323-328`](../crates/onnx-runtime-ep-cuda/src/kernels/matmul_nbits.rs#L323-L328);
[`block_quantized_matmul.rs:585-600`](../crates/onnx-runtime-ep-cuda/src/kernels/block_quantized_matmul.rs#L585-L600)).

## Required custom/runtime boundaries

| Boundary | CUDA status | Current evidence and consequence |
|---|---:|---|
| `pkg.nxrt::BlockQuantizedMoE` | ❌ missing | CUDA's complete registered-op list contains no such factory ([`kernels/mod.rs:94-178`](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L94-L178)); CPU's adjacent private registrations likewise stop at `BlockQuantizedMatMul`, `SparseKvGather`, and CSA ([CPU `kernels/mod.rs:214-231`](../crates/onnx-runtime-ep-cpu/src/kernels/mod.rs#L214-L231)). The only implementation today is the lazy-weight boundary matcher ([`weight.rs:94-105`](../crates/onnx-runtime-ep-api/src/weight.rs#L94-L105)), with whole-weight host materialization / deferred device binding ([`weight.rs:171-241`](../crates/onnx-runtime-ep-api/src/weight.rs#L171-L241)). |
| IndexShare selected-token attention | ❌ missing | No CUDA or CPU `IndexShare`/selected-token handler exists. Standard `Attention` is a dense f32 reference path; it materializes Q/K/V on the host ([`standard_attention.rs:155-208`](../crates/onnx-runtime-ep-cuda/src/kernels/standard_attention.rs#L155-L208)). The additive-mask export can be correct, but it does not avoid dense million-token attention. |
| GLM MTP package discovery / HC lifecycle | ❌ incomplete | Metadata maps `ProposalType::Mtp` to `NotYetSupported` ([`metadata/parser.rs:76-90`](../crates/onnx-genai-metadata/src/parser.rs#L76-L90)). The generic MTP runner binds only `[B,S,H]` embeds and hidden states ([`mtp.rs:211-240`](../crates/onnx-genai-ort/src/mtp.rs#L211-L240)), not GLM's HC contract. Speculation creates a fresh `MtpProposer` per verification step ([`speculative.rs:965-980`](../crates/onnx-genai-engine/src/speculative.rs#L965-L980)), so its accept/rewind state cannot persist across those iterations. |

`BLOCKQUANTIZEDMOE_DESIGN.md` is therefore a design/ABI proposal, not evidence
of implementation: it explicitly says no kernel exists and requests eight
Justin sign-offs before kernel work ([`BLOCKQUANTIZEDMOE_DESIGN.md:1-5`](BLOCKQUANTIZEDMOE_DESIGN.md#L1-L5),
[`381-431`](BLOCKQUANTIZEDMOE_DESIGN.md#L381-L431)).

### DeepSeek status (separate from GLM)

Do not treat DeepSeek CSA as a remaining GLM standard-op gap. CPU registers
`pkg.nxrt::SparseKvGather` and `CompressedSparseAttention`, while CUDA registers
neither ([CPU `kernels/mod.rs:224-231`](../crates/onnx-runtime-ep-cpu/src/kernels/mod.rs#L224-L231);
[CUDA `kernels/mod.rs:94-178`](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L94-L178)).
The CPU CSA implementation still explicitly errors on equal-score ratio-4 top-k
because tie ordering is unfrozen ([`compressed_sparse_attention.rs:1580-1590`](../crates/onnx-runtime-ep-cpu/src/kernels/compressed_sparse_attention.rs#L1580-L1590)).
That is a DeepSeek CSA limitation, not an IndexShare or portable-GLM limitation.

## CUDA-only means no CPU fallback

CUDA is selected as a single EP (`DevicePreference::Gpu` / explicit CUDA);
automatic selection remains CPU-only until heterogeneous placement exists
([`session/lib.rs:555-585`](../crates/onnx-runtime-session/src/lib.rs#L555-L585)).
If static coverage finds nodes CUDA cannot claim, the session returns
`HeterogeneousPlacementRequired`, explicitly stating that CPU+CUDA placement is
not available ([`session/lib.rs:111-116`](../crates/onnx-runtime-session/src/lib.rs#L111-L116)).
Therefore every node in a CUDA-only GLM target and MTP sidecar must be claimed
by CUDA; there is no CPU rescue for an f16/bf16 attention/RoPE node or a missing
custom boundary.

## Prioritized remaining work

### P0 — required for a smooth native GLM implementation

1. **Freeze and implement `pkg.nxrt::BlockQuantizedMoE` on CPU then CUDA.**
   This is the actual selected-expert/IQ-MXFP4 boundary; existing lazy-weight
   plumbing is insufficient without a registered kernel.
2. **Define and implement selected-token IndexShare attention.** Keep the
   additive-mask standard `Attention` path as the correctness fallback, but do
   not describe it as viable at the intended million-token context.
3. **Wire the GLM MTP package contract.** Add package discovery, HC tensor
   threading, generation-owned proposer state, and target/MTP rollback as one
   lifecycle—not merely another f32 BSH MTP head.
4. **Replace host-staged standard attention/RoPE with device-resident
   kernels.** This is required for smooth CUDA execution even after the custom
   boundaries above are implemented.

### P1 — quality and adjacent-model work

5. Add CUDA CSA/SparseKvGather and freeze the equal-score top-k tie rule for
   DeepSeek; this is independent of GLM's portable standard graph.
6. Integrate activation-liveness reuse and CUDA graph-capture-safe paths.
   Several newly covered indexing/movement kernels deliberately synchronize or
   host-stage metadata, so registration closure is not capture/performance
   closure.

## Shortest path to native CUDA target-only GLM decode

1. Use the current Mobius portable graph with **f32 activations**, standard
   attention/RoPE/RMSNorm, f32 `TopK`/mask construction, and an accepted f32
   quantized-linear profile.
2. Run it as CUDA-only only after verifying every emitted node satisfies the
   concrete contracts in the table (especially contiguous tensors and int64
   metadata/index inputs).
3. Treat this as a correctness/graph-loading milestone. It remains dense and
   host-staged at attention/RoPE and evaluates portable experts.
4. For smooth execution, also replace host-staged Attention/RoPE with
   GPU-native kernels, alongside `BlockQuantizedMoE`, selected-token
   IndexShare, and GLM-specific MTP lifecycle work.

This supersedes the former CUDA “add the graph core” plan. The decisive CUDA
questions are now the constrained-kernel f32 contracts, custom-boundary
implementation, and GPU-native attention/RoPE—not whether the standard
portable graph has CUDA registrations.

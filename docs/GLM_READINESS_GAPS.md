# GLM-5.2 / DeepSeek-V4-Flash native-runtime readiness gaps

**Audit point:** `main` at `8d9c958`, 2026-07-18
**Scope:** native `onnx-runtime-session` using one in-tree CPU or CUDA execution
provider. This is not an ORT EP-fallback assessment.

## Executive conclusion

The CUDA standard-op loading gap identified by the original audit is closed.
CUDA registers the standard ops needed by the portable GLM-5.2 graph, and its
standard `Attention` and `RotaryEmbedding` implementations are now
**GPU-native**, not host-staged
([Attention GPU-native execution](../crates/onnx-runtime-ep-cuda/src/kernels/standard_attention.rs#L27-L46),
[RoPE device kernels](../crates/onnx-runtime-ep-cuda/src/kernels/rotary_embedding.rs#L43-L127)).
`SparseKvGather` is also GPU-native
([CUDA gather](../crates/onnx-runtime-ep-cuda/src/kernels/sparse_kv_gather.rs#L1-L14)).

DeepSeek CSA now has a correct CUDA path, but only as **Phase A**: it stages
every tensor through the host and delegates computation to the CPU oracle for
bit parity
([CUDA CSA header](../crates/onnx-runtime-ep-cuda/src/kernels/compressed_sparse_attention.rs#L1-L31)).
It deliberately reports `cuda_graph_compatible() == false`
([CUDA CSA capture contract](../crates/onnx-runtime-ep-cuda/src/kernels/compressed_sparse_attention.rs#L202-L211)).
The device-resident fused path is Phase B and remains planned, not implemented
([Phase B plan](CUDA_CSA_PHASE_B_PLAN.md)).

The principal remaining roadmap gaps are therefore custom contracts and
production paths, not standard-op registration:

- Mobius has no fused quantized-expert emitter for `com.microsoft::QMoE` or
  `pkg.nxrt::BlockQuantizedMoE`, so existing runtime kernels are not exercised
  by exported GLM/DeepSeek graphs;
- GLM IndexShare selected-token attention has no frozen private-op ABI;
- CUDA CSA needs its device-resident Phase B;
- Mobius must provide pinned export artifacts and the explicit recurrent
  `mtp_state` required for `hc_mult > 1`.

## 2026-07-19 E2E bring-up update

GLM-5.2 now runs end-to-end through the onnx-genai runtime interface in both
fp32 and int4-quantized forms using the tiny synthetic `glm_moe_dsa` harness.
The fp32 graph completes prefill plus eight decode steps. The quantized graph
does the same while executing 34 asymmetric block-32
`com.microsoft::MatMulNBits` nodes across the full graph, including every MoE
expert. DeepSeek-V2 likewise completes prefill plus eight decode steps through
the shared MLA + MoE path.

These are random-weight structural bring-up results, not claims of semantic
correctness for production weights. The emitted GLM/DeepSeek MoE remains a
per-expert decomposition: Mobius emits `MatMulNBits`, not fused `QMoE` or
`pkg.nxrt::BlockQuantizedMoE`. Consequently the existing CPU/CUDA QMoE kernel
and CPU BlockQuantizedMoE kernel have not yet been exercised E2E by these
harnesses. The primary exporter gap is a Mobius fused-quantized-expert path
with the required routing/layout contract. DeepSeek-V4 E2E remains blocked
upstream by the absence of a usable reference configuration/export artifact.

## Two milestones must remain distinct

1. **CUDA-only graph loading / correctness:** the standard-op registration gap
   is closed. Standard Attention/RoPE are GPU-native, CUDA has native
   `SparseKvGather`, and DeepSeek CSA has a correctness-first CUDA Phase-A path.
   Loading the intended model packages still depends on the model-specific
   boundaries and exporter contracts: BlockQuantizedMoE, IndexShare, and the
   released Mobius artifacts/state ABI.
2. **Smooth/performance-ready CUDA execution:** the former standard
   Attention/RoPE host-staging blocker is **resolved**. The remaining DeepSeek
   smooth-execution blocker is **device-resident CSA Phase B**: Phase A performs
   full D2H/CPU/H2D staging and cannot be CUDA-graph captured. GLM additionally
   needs selected-expert BlockQuantizedMoE and selected-token IndexShare rather
   than its dense correctness fallbacks.

“The graph can load” therefore still does not mean “production-ready at
million-token context,” but standard Attention/RoPE are no longer the reason.

## What the exported graph requires

Mobius [PR #404](https://github.com/onnxruntime/mobius/pull/404)'s GLM-5.2
portable target uses standard opset 24,
`Attention`/`RotaryEmbedding`/`RMSNormalization`, a portable 256-expert MoE
decomposition, and IndexShare `TopK` converted to a dense additive attention
mask. The expert loop and mask path are correctness-first rather than
selected-expert/selected-token execution. The external MTP sidecar is a
separate decoder graph, not an ONNX `MTP` operator.

Mobius [PR #405](https://github.com/onnxruntime/mobius/pull/405) is the
DeepSeek-V4-Flash exporter target for the frozen CSA/MTP contract
([runtime design](DEEPSEEK_CSA_MTP_RUNTIME.md#L3-L10)). Both exporter PRs remain
open drafts as of this audit, so a pinned package artifact is still an
end-to-end dependency.

## CUDA standard-op coverage relevant to GLM

`✅ native` means bulk data is processed on device. `⚠️ constrained` means the
operator is registered but has a material dtype/layout or capture restriction.

| Required op | CUDA status | Evidence / constraint |
|---|---:|---|
| `Attention` (23/24) | ✅ native, constrained | GPU-native NVRTC kernels keep Q/K/V, masks, scores, and outputs on device; f32-only and not capture-compatible because small control arrays/nonpad lengths synchronize with the host ([header](../crates/onnx-runtime-ep-cuda/src/kernels/standard_attention.rs#L27-L46), [dtype contract](../crates/onnx-runtime-ep-cuda/src/kernels/standard_attention.rs#L69-L75), [capture](../crates/onnx-runtime-ep-cuda/src/kernels/standard_attention.rs#L1040-L1047)). |
| `RotaryEmbedding` (23) | ✅ native, constrained | Device NVRTC rotation plus device-side `position_ids` bounds validation; f32 X/cos/sin and int64 positions ([device source](../crates/onnx-runtime-ep-cuda/src/kernels/rotary_embedding.rs#L43-L127), [claim gate](../crates/onnx-runtime-ep-cuda/src/kernels/rotary_embedding.rs#L129-L144)). |
| `RMSNormalization` | ⚠️ constrained | Registered; contiguous f32 X/Scale/Y. |
| `TopK` | ⚠️ constrained | Registered; f32 values and int64 scalar K. |
| `CumSum` | ⚠️ constrained | Registered; f32 or int64 data and int64 scalar axis. |
| `Gather` / `GatherElements` | ✅ native | Fixed-width device gathers; GLM integer-index use is covered. |
| `ScatterElements` | ⚠️ constrained | Registered; contiguous f32/int64 data and int64 indices. |
| `Where` | ✅ native | Bool condition with matching fixed-width branches and broadcasting. |
| movement/construction (`Expand`, `Tile`, `Concat`, `Reshape`, `Slice`, `Split`, `Squeeze`, `Transpose`, `Unsqueeze`) | ✅ native | Registered fixed-width movement paths. |
| comparisons and arithmetic | ✅ / constrained | GLM's integer comparisons and f32 arithmetic are covered; individual dtype matrices still apply. |
| `Cast`, `CastLike`, `Shape`, `Constant` | ✅ native | Registered CUDA graph-construction core. |

The checked floating inputs of Attention and RoPE remain f32-only. A uniform
f32 activation graph is the simplest conservative export profile, while casts
can satisfy those operators inside an otherwise mixed-precision graph.

## Optional-input claim contract is fixed

The session claim path now preserves omitted optional input slots as
`DataType::Undefined`, rather than fabricating `DataType::Float32`
([executor planning](../crates/onnx-runtime-session/src/executor.rs#L116-L122),
[static plan](../crates/onnx-runtime-session/src/executor.rs#L871-L877),
[dynamic plan](../crates/onnx-runtime-session/src/executor.rs#L1124-L1130)).
This removes the claim-then-fail ambiguity between an absent optional and a
supplied f32 tensor.

CUDA Attention's claim gate consumes that contract explicitly: absent
`attn_mask`, `past_key`, `past_value`, and `nonpad_kv_seqlen` slots are
`Undefined`; present masks must be bool/f32, present cache tensors must be paired
f32 values, and present nonpad lengths must be int64 with the opset/cache
restrictions enforced
([claim gate](../crates/onnx-runtime-ep-cuda/src/kernels/standard_attention.rs#L311-L387)).
Valid GLM/Mobius prefill without past KV is therefore claimed correctly without
weakening wrong-dtype rejection.

## Required custom/runtime boundaries

| Boundary | Current status | Consequence |
|---|---:|---|
| Fused quantized experts (`com.microsoft::QMoE` / `pkg.nxrt::BlockQuantizedMoE`) | ◐ runtime kernels exist, exporter missing | CPU and CUDA QMoE kernels and the CPU BlockQuantizedMoE parity kernel are registered, but Mobius emits per-expert `MatMulNBits` for GLM/DeepSeek. A fused quantized-expert emitter and routing/layout validation are required for E2E coverage. |
| GLM IndexShare selected-token attention | ❌ contract-blocked | Mobius #404's dense additive-mask fallback is a correctness oracle but scans full K/V. Op boundary, index order/sentinels, deterministic parity, and cache/mask semantics are unfrozen ([team decision](../.squad/decisions.md#L916-L932)). |
| CUDA `CompressedSparseAttention` | ◐ Phase A landed | Correct host-staged CUDA execution delegates to the CPU oracle. Device-resident compression, selection, sparse attention, state, capture, and rollback are Phase B; seven user decisions remain ([plan](CUDA_CSA_PHASE_B_PLAN.md#L22-L96)). |
| CUDA `SparseKvGather` | ✅ native | Device byte-copy gather is registered; small index/valid-length tensors are host-read for deterministic range validation. This primitive does not itself define GLM IndexShare. |
| GLM/DeepSeek MTP Phase 1 | ✅ functionally complete, one contract block | Metadata and initializer references, rank-4 HC extraction, BSHC binding, persistent per-generation proposer, and greedy draft/verify/correction reuse are implemented. Only explicit recurrent `mtp_state` remains blocked on the released Mobius package contract ([decision audit](../.squad/decisions.md#L855-L867)). |

## DeepSeek CUDA status

CPU remains the authoritative full CSA numerical implementation. CUDA now
registers both `SparseKvGather` and `CompressedSparseAttention`; the latter is
the Phase-A host-staged wrapper
([CUDA registry](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L99-L102),
[registrations](../crates/onnx-runtime-ep-cuda/src/kernels/mod.rs#L325-L336)).
This closes the “no CUDA path” correctness gap, but not the performance gap.

Phase B is blocked on seven decisions covering parity target, shared
quantization code, fixed-capacity cache budget, ragged cursors, deterministic
top-k/capture staging, checkpoint ownership, and fallback retirement
([Phase B decisions](CUDA_CSA_PHASE_B_PLAN.md#L22-L96)).

## MTP Phase 1 status

Phase 1 is functionally complete:

- Mobius sidecar metadata resolves into runtime configuration;
- package embedding and LM-head initializer references are supported;
- target Hyper-Connection output preserves `[B,S,hc_mult,H]`;
- the sidecar binds BSHC hidden state and separates `mtp_hidden` from recurrent
  `mtp_state`;
- the proposer/session is generation-owned and reused across verification
  iterations.

The only remaining Phase-1 item is explicit `mtp_state` for `hc_mult > 1`.
Current runtime handling correctly rejects a package that omits it rather than
inventing recurrence
([ORT MTP output contract](../crates/onnx-genai-ort/src/mtp.rs#L430-L440)).
Mobius must export the official recurrent state; correction-token
`accepted_prefix` cache semantics also remain unfrozen
([team decision](../.squad/decisions.md#L741-L747)).

## Prioritized remaining work

### P0 — unblock model contracts and end-to-end artifacts

1. Add a Mobius fused quantized-expert emitter for GLM/DeepSeek, targeting
   `com.microsoft::QMoE` and/or `pkg.nxrt::BlockQuantizedMoE`, and validate it
   E2E against the existing runtime kernels. Add a CUDA BlockQuantizedMoE path
   if that private ABI is selected for production.
2. Freeze the GLM IndexShare selected-token operator ABI and parity rules, then
   preserve the dense additive-mask path as its oracle.
3. Resolve the seven CUDA CSA Phase-B decisions and implement B0–B7.
4. Have Mobius export explicit recurrent `mtp_state` where `hc_mult > 1`, and
   pin/land usable #404/#405 package artifacts for runtime end-to-end tests.

### P1 — production precision and capture

5. Add f16/bf16 production paths where model contracts require them; the current
   f32 GPU-native Attention/RoPE kernels are correctness-capable but not a claim
   of optimal production precision.
6. Complete capture-safe control/state handling. Attention and RoPE keep bulk
   tensors on device, but host synchronization for validation/control still
   prevents CUDA-graph compatibility; CSA Phase B addresses the larger stateful
   blocker.

## Shortest verified path

- **GLM correctness:** the tiny synthetic fp32 and int4 graphs now establish
  structural prefill/decode execution. Production correctness still requires
  real weights, pinned artifacts, and numerical comparison; use standard
  Attention/RoPE, the dense IndexShare oracle, and the portable/per-expert
  expert path until a fused quantized-expert exporter is validated.
- **DeepSeek correctness:** use CUDA CSA Phase A and native SparseKvGather
  against a pinned #405 artifact.
- **Smooth execution:** replace portable/per-expert experts with an exported
  QMoE or BlockQuantizedMoE path,
  dense IndexShare with a frozen selected-token boundary, and CSA Phase A with
  device-resident Phase B.

The readiness question is no longer whether CUDA has standard Attention/RoPE or
a CSA entry point. It is whether the remaining private ABIs are frozen and the
correctness-first fallbacks have production device-resident replacements.

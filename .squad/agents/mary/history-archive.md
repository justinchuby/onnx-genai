# mary — History Archive

_Archived by Scribe round 8, 2026-07-31. Entries predating wave (Inc2a+Inc2b)._

# mary — History

## 2026-07-30T04:10:00Z — Reduction and shape-aware CUDA claim-gate work

- Authored PR #420 to widen extended reductions to f16/bf16 with f32 accumulation; merged as `6610f86f`, clearing the native 27B FP16 `ReduceSumSquare` CUDA fallback.
- Revised PR #424 at `93d9e7b8` with `require_input_rank`, making CUDA claim gates shape-aware so deferred ranks retain CPU fallback instead of being treated as unsupported static shapes.

## 2026-07-27T09:00:00Z — DeepSeek/GLM native-CUDA bring-up and R1 GQA resolution

- Established native-CUDA viability for DeepSeek-Coder and DeepSeek-V2-Lite with ORT token-exact greedy decode; GLM native CUDA is coherent but lacks an ORT-CUDA oracle because ORT rejects its authored GQA rotary attribute.
- Confirmed DeepSeek-V2-Lite QMoE resident CUDA correctness, while documenting that advertised weight-offload knobs do not yet activate CUDA expert paging for this path.
- Resolved the R1-Distill divergence as an ORT-CUDA fp16 near-tie outlier: native token 374 is correct/more accurate than ORT-CUDA token 315. Landed PR #430 (`5c49c891`) with GQA 6:1 non-interleaved-rotary decode regressions at head_dim 64 and 128.

## 2026-07-30T07:20:00Z — Qwen3.6-27B persistent recurrent-state bindings

- Confirmed the Qwen 27B Unsqueeze blocker was already resolved, then generalized native CUDA persistent state allocation so metadata-declared fixed rank-3 `conv_state`/recurrent state uses static replace semantics instead of rank-4 KV capacity growth. The graph now reaches the next blocker: unsupported rank-3/1-D CUDA Conv.

## 2026-07-30T09:16:00Z — 27B Conv and Silu blocker chain

- PR #438 merged native CUDA rank-3 Conv1D support, advancing the Qwen3.6-27B probe past `__fn0_Conv_node_12`.
- PR #440 supplies `com.microsoft::Silu` unary shape inference and is in review; the next observed blocker is `value#1414 not produced`.

## 2026-07-30T13:36:00Z — Issue #445 delivered; #35B-A3B unblocked; #384 scoped

- PR #445 merged (TopK fp16/bf16 CUDA parity). Independently conducted native-CUDA correctness bring-up on DeepSeek-Coder, DeepSeek-V2-Lite (26-QMoE), DeepSeek-R1, GLM-4: exact token parity on Coder/V2-Lite, R1 isolated to GQA/KV parity gap, GLM-4 native coherent.
- Cleared three 35B-A3B blockers: (1) generalized CUDA persistent-state binding to rank-3 `conv_state`; (2) added CUDA Conv rank-3 NCL with depthwise causal; (3) registered `com.microsoft::Silu` v1 unary shape inference (PRs #437, #438, #440 all merged).
- Status: 35B-A3B now unblocked from empirical decode measurement; next blocker is `value#1414` executor error at graph lowering.
- Issue #384 scoped: three increments to make `PipelineDecodeLoopBackend` drive native components. Inc1 routes `every_step` through `ComponentSession` trait; value seam is backend-neutral host-resident `ComponentTensor`. Proving native embedding in hybrid loop (embedding native, decoder ORT) with token parity as Inc1 deliverable.

## 2026-07-30T15:20:00Z — Native pipeline Inc2a + Inc2b merged (#478, #479)

- PR #478 merged (Melina APPROVED): Inc2a pure refactor — stateful `PipelineDecoderComponent` trait + `OrtPipelineDecoder`; `PipelineDecodeLoopBackend` drives the decoder through the trait. Behaviour-identical (e2e token output unchanged + explicit equivalence assertion).
- PR #479 merged (Lori APPROVED, instrumented proof): Inc2b `NativePipelineDecoder` device-KV decoder — KV stays device-resident, one embedding uploaded/step, static cross-KV uploaded once. Token parity `[0,5,6,7] == ORT` on a small CPU model. Native step extended for routed/`inputs_embeds` inputs.
- In flight: Inc3 (CUDA native decoder — device-KV paged mirroring + cross-component/vision, full 35B-A3B on native), PR pending.
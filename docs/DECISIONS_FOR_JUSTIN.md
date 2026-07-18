# Decisions for Justin — roadmap unblock checklist

**Audit point:** `main` at `8d9c958`, 2026-07-18

This index consolidates the open owner decisions blocking the DeepSeek,
GLM-5.2, and Kimi K3 roadmap. It contains **32 decision points** across seven
areas. Recommendations below repeat an existing team/doc leaning; where no
leaning is recorded, the checklist says so.

## CUDA CSA Phase B — 7 decisions

Phase A is correct but host-staged and not CUDA-graph compatible. These choices
shape or block B0–B7.

### CSA-D1 — Numerical parity target
- **Why blocked:** the CPU oracle accumulates attention in f32; official
  `kernel.py` uses BF16 QK and casts `p_j` to BF16 before value GEMM.
- **Decision needed:** target **(a)** current CPU-f32 oracle parity or **(b)**
  official BF16 numerics.
- **Impact:** freezes B1 reduction order, tests, and the definition of parity.
- **Recommended default:** **CPU-f32 oracle first**; reconcile official BF16
  goldens separately.
- **Pointer:** [`CUDA_CSA_PHASE_B_PLAN.md` D1](CUDA_CSA_PHASE_B_PLAN.md#L27-L40)

### CSA-D2 — FP8/FP4 device quantization strategy
- **Why blocked:** existing block-quant code exposes reusable device dequant
  helpers, but CSA also needs graph-safe quantization.
- **Decision needed:** **(a)** extract shared quant/dequant NVRTC primitives or
  **(b)** build a self-contained CSA quant module.
- **Impact:** unblocks B0 quant round-trip scaffolding and B2/B3 device caches.
- **Recommended default:** shared graph-safe NVRTC primitives; do not depend on
  the graph-incompatible matmul kernel path.
- **Pointer:** [`CUDA_CSA_PHASE_B_PLAN.md` D2](CUDA_CSA_PHASE_B_PLAN.md#L42-L55)

### CSA-D3 — Fixed-capacity cache budget
- **Why blocked:** graph capture requires stable device addresses, but capacity
  and memory policy are not owner-approved.
- **Decision needed:** approve fixed-capacity buffers; choose package
  `max_seq_len`, dense-window `W`, and fail-closed behavior when capacity/device
  memory is insufficient.
- **Impact:** unblocks the B0 buffer manager and B2 device-resident state.
- **Recommended default:** fixed capacity from package metadata; reject at claim
  time if the device cannot hold it.
- **Pointer:** [`CUDA_CSA_PHASE_B_PLAN.md` D3](CUDA_CSA_PHASE_B_PLAN.md#L57-L66)

### CSA-D4 — Ragged batch cursor scope
- **Why blocked:** v1 currently assumes equal compression/index cursor lengths
  within a batch.
- **Decision needed:** confirm equal-length cursors for B0–B7 or require ragged
  per-row lengths now.
- **Impact:** freezes state layout and kernel scheduling complexity.
- **Recommended default:** equal-length v1; defer ragged batching.
- **Pointer:** [`CUDA_CSA_PHASE_B_PLAN.md` D4](CUDA_CSA_PHASE_B_PLAN.md#L68-L72)

### CSA-D5 — Top-k staging before capture
- **Why blocked:** ratio-4 needs deterministic top-k; the current CUDA TopK is
  not graph-capturable.
- **Decision needed:** allow **(a)** index-only host readback through B4, then
  device/capturable top-k in B6, or require **(b)** full device residency in B4.
- **Impact:** sets B4 scope and the B6 capture boundary.
- **Recommended default:** permit bounded index-only readback until B6.
- **Pointer:** [`CUDA_CSA_PHASE_B_PLAN.md` D5](CUDA_CSA_PHASE_B_PLAN.md#L74-L82)

### CSA-D6 — Checkpoint/restore ownership
- **Why blocked:** speculative rollback needs a single owner for CSA cursors and
  tail invalidation.
- **Decision needed:** choose kernel-owned device cursors driven by engine
  `checkpoint()`/`restore(base_len+accepted)`, or engine-owned cursor journals.
- **Impact:** unblocks B7 rollback and composition with speculative decode.
- **Recommended default:** kernel owns cursors; engine drives restore; restore
  adjusts lengths/clears tails without recompression.
- **Pointer:** [`CUDA_CSA_PHASE_B_PLAN.md` D6](CUDA_CSA_PHASE_B_PLAN.md#L84-L90)

### CSA-D7 — Host oracle after switchover
- **Why blocked:** B7 must define whether Phase A remains available.
- **Decision needed:** remove the host-staged path or retain it behind a debug
  `--csa-oracle`-style switch, never on the default path.
- **Impact:** freezes support/triage policy and B7 cleanup scope.
- **Recommended default:** retain it as a debug differential oracle.
- **Pointer:** [`CUDA_CSA_PHASE_B_PLAN.md` D7](CUDA_CSA_PHASE_B_PLAN.md#L92-L96)

## BlockQuantizedMoE — 8 decisions

No kernel work should begin until the proposed v1 ABI is signed off.

### BQMoE-D1 — Weight input ordering
- **Why blocked:** IQ formats do not need QMoE's reserved scale/zero-point gaps.
- **Decision needed:** dense input indices 0–8 or QMoE-index-preserving gaps.
- **Impact:** freezes exporter, shape-inference, CPU, and CUDA input ABI.
- **Recommended default:** dense 0–8.
- **Pointer:** [`BLOCKQUANTIZEDMOE_DESIGN.md` decision 1](BLOCKQUANTIZEDMOE_DESIGN.md#L385-L388)

### BQMoE-D2 — Router input meaning
- **Why blocked:** the op must distinguish selection logits from optional
  aggregation weights.
- **Decision needed:** accept logits with internal softmax, or require
  pre-normalized weights.
- **Impact:** freezes routing numerics and QMoE transcode behavior.
- **Recommended default:** logits in; optional `router_weights` override
  aggregation.
- **Pointer:** [`BLOCKQUANTIZEDMOE_DESIGN.md` decision 2](BLOCKQUANTIZEDMOE_DESIGN.md#L390-L394)

### BQMoE-D3 — Block-format encoding
- **Why blocked:** schema can encode the format as a string or integer enum.
- **Decision needed:** string `format` attribute or compact integer enum.
- **Impact:** freezes package readability and compatibility with
  BlockQuantizedMatMul.
- **Recommended default:** string `format`.
- **Pointer:** [`BLOCKQUANTIZEDMOE_DESIGN.md` decision 3](BLOCKQUANTIZEDMOE_DESIGN.md#L396-L400)

### BQMoE-D4 — `sparse_mixer` placement
- **Why blocked:** sparse-mixer normalization semantics are not frozen inside
  the private op.
- **Decision needed:** keep normalization outside as portable graph ops, or fuse
  it into v1.
- **Impact:** determines v1 routing scope and parity surface.
- **Recommended default:** outside the op; error if `use_sparse_mixer=1`.
- **Pointer:** [`BLOCKQUANTIZEDMOE_DESIGN.md` decision 4](BLOCKQUANTIZEDMOE_DESIGN.md#L402-L404)

### BQMoE-D5 — Format uniformity
- **Why blocked:** v1 may either require one block format or permit mixed
  formats by expert/projection.
- **Decision needed:** uniform format or per-projection/per-expert formats.
- **Impact:** freezes weight metadata and kernel dispatch complexity.
- **Recommended default:** one uniform format in v1.
- **Pointer:** [`BLOCKQUANTIZEDMOE_DESIGN.md` decision 5](BLOCKQUANTIZEDMOE_DESIGN.md#L406-L409)

### BQMoE-D6 — Hidden/intermediate dimensions
- **Why blocked:** dimensions can be inferred from weight shapes or duplicated
  as attributes.
- **Decision needed:** inferred dimensions or declared attributes.
- **Impact:** freezes validation rules and avoids/accepts redundant metadata.
- **Recommended default:** infer from weight shapes.
- **Pointer:** [`BLOCKQUANTIZEDMOE_DESIGN.md` decision 6](BLOCKQUANTIZEDMOE_DESIGN.md#L411-L414)

### BQMoE-D7 — Paging requirement in v1
- **Why blocked:** current lazy binding materializes whole weights and cannot
  lease selected expert slices.
- **Decision needed:** resident-only correctness first, or require selected
  expert device paging before v1 ships.
- **Impact:** determines whether CPU/CUDA oracle work can land before binder
  seam expansion.
- **Recommended default:** resident-first; extend `LazyWeight`/binder before
  claiming paging support.
- **Pointer:** [`BLOCKQUANTIZEDMOE_DESIGN.md` decision 7](BLOCKQUANTIZEDMOE_DESIGN.md#L416-L426)

### BQMoE-D8 — Freeze op version 1
- **Why blocked:** implementation needs a stable private ABI/version target.
- **Decision needed:** approve `pkg.nxrt::BlockQuantizedMoE` v1 now or keep the
  proposal provisional.
- **Impact:** unblocks shape inference, CPU oracle, CUDA path, and Mobius export.
- **Recommended default:** freeze and ship v1.
- **Pointer:** [`BLOCKQUANTIZEDMOE_DESIGN.md` decision 8](BLOCKQUANTIZEDMOE_DESIGN.md#L428-L430)

## Kimi K3 — 5 decisions

Weights and the detailed report are not public until the announced release
boundary, so these choices govern pre-build scope rather than a guessed ABI.

### K3-D1 — Pre-build before official artifacts
- **Why blocked:** KDA/MLA layouts, cache ABI, and exact equations are unpublished.
- **Decision needed:** build provisional typed-state/oracle scaffolding now, or
  wait for released weights/report.
- **Impact:** controls lead time without prematurely freezing a private ABI.
- **Recommended default:** build scaffolding and capability negotiation now;
  keep operator ABI/layout provisional.
- **Pointer:** [`KIMI_K_READINESS.md` decision 1](KIMI_K_READINESS.md#L178-L180)

### K3-D2 — Quantization policy
- **Why blocked:** exact Moonshot MXFP4/MXFP8 packing and activation semantics
  are not public.
- **Decision needed:** require native byte-exact formats for the first milestone,
  or permit an explicitly converted correctness profile.
- **Impact:** sets BlockQuantizedMoE/converter scope and first runnable target.
- **Recommended default:** allow a labeled converted correctness profile;
  retain native formats as the production target.
- **Pointer:** [`KIMI_K_READINESS.md` decision 2](KIMI_K_READINESS.md#L181-L185)

### K3-D3 — First deployment target
- **Why blocked:** full K3 serving needs 64+-accelerator-class expert
  parallelism, while oracle work can start on CPU/one GPU.
- **Decision needed:** prioritize single-device correctness or immediate
  multi-device production.
- **Impact:** determines sequencing of kernels versus placement/collectives.
- **Recommended default:** CPU + one-GPU oracle first; start expert-parallel
  transport/placement in parallel.
- **Pointer:** [`KIMI_K_READINESS.md` decision 3](KIMI_K_READINESS.md#L186-L189)

### K3-D4 — KDA/MLA semantic boundaries
- **Why blocked:** reusing CSA as the operator would freeze incorrect state
  semantics.
- **Decision needed:** separate `KDA`, `MLA`, and `CSA` operators/state kinds, or
  overload a shared model-switched boundary.
- **Impact:** freezes typed-state architecture and future package ABI.
- **Recommended default:** separate semantic ops/state; share lifecycle
  infrastructure only.
- **Pointer:** [`KIMI_K_READINESS.md` decision 4](KIMI_K_READINESS.md#L190-L192),
  [team decision](../.squad/decisions.md#L522-L525)

### K3-D5 — K3 MTP scope
- **Why blocked:** no public K3 artifact verifies an MTP/speculative head.
- **Decision needed:** make MTP part of pre-release K3 scope or keep it
  conditional on released weights/config.
- **Impact:** prevents speculative work from driving the K3 critical path
  without evidence.
- **Recommended default:** no K3-specific MTP work until artifacts verify it.
- **Pointer:** [`KIMI_K_READINESS.md` decision 5](KIMI_K_READINESS.md#L193-L194)

## IndexShare selected-token attention — 4 decisions

The dense additive-mask path is a valid oracle but not a production
million-token implementation.

### IndexShare-D1 — Private op boundary
- **Why blocked:** no `pkg.nxrt` GLM op name/version or ownership boundary is
  frozen.
- **Decision needed:** consume exporter-computed top-k indices, or own
  full/shared IndexShare selection plus index-key cache/state.
- **Impact:** unblocks schema, shape inference, Mobius emission, and CPU/CUDA
  implementation.
- **Recommended default:** no recorded team leaning.
- **Pointer:** [IndexShare decision 1](../.squad/decisions.md#L926-L927)

### IndexShare-D2 — Index ordering and sentinels
- **Why blocked:** ordered-list versus set semantics, duplicates, invalid
  sentinels, bounds, and empty selection are unfrozen.
- **Decision needed:** specify ordering, duplicate policy, `-1`, out-of-range,
  and empty-selection behavior.
- **Impact:** freezes validation, gather behavior, and observable output.
- **Recommended default:** no recorded team leaning.
- **Pointer:** [IndexShare decision 2](../.squad/decisions.md#L928-L928)

### IndexShare-D3 — Deterministic/numerical parity
- **Why blocked:** incoming `TopK(sorted=0)` order may differ from dense-cache
  accumulation order.
- **Decision needed:** preserve incoming order or canonicalize to dense-cache
  order; choose exact f32 equality or tolerance.
- **Impact:** defines the dense-mask oracle comparison and CUDA determinism.
- **Recommended default:** no recorded team leaning.
- **Pointer:** [IndexShare decision 3](../.squad/decisions.md#L929-L929)

### IndexShare-D4 — Mask/cache ABI
- **Why blocked:** causal/padding composition, cache outputs, layouts/head
  sharing, and shared-layer index I/O are unspecified.
- **Decision needed:** freeze those mask, cache, layout, and shared-index rules.
- **Impact:** completes the user-visible v1 contract.
- **Recommended default:** no recorded team leaning.
- **Pointer:** [IndexShare decision 4](../.squad/decisions.md#L930-L932)

## GraphView lens — 5 decisions

The revised design has no implementation and awaits sign-off on its five
foundational choices.

### GraphView-D1 — Partition representation
- **Why blocked:** flattening assignment by EP loses ORT claim atomicity and
  partition metadata.
- **Decision needed:** use `PartitionId` + `CompiledPartitionView`, or flatten
  by EP.
- **Impact:** unblocks partition-aware frozen-plan implementation.
- **Recommended default:** `PartitionId` + `CompiledPartitionView`.
- **Pointer:** [`GRAPHVIEW_LENS_DESIGN.md`](GRAPHVIEW_LENS_DESIGN.md#L9-L12)

### GraphView-D2 — Capability API
- **Why blocked:** current capability calls clone shape/layout collections.
- **Decision needed:** migrate to iterator/view-based `supports_node` before
  promising allocation-free coverage, or retain cloning.
- **Impact:** freezes EP capability API migration scope.
- **Recommended default:** migrate to iterator/view inputs.
- **Pointer:** [`GRAPHVIEW_LENS_DESIGN.md`](GRAPHVIEW_LENS_DESIGN.md#L9-L12)

### GraphView-D3 — Freeze versus placement
- **Why blocked:** placement data must not become mutable state inside frozen IR.
- **Decision needed:** structural lens before partitioning with placement in
  `FrozenPlan`, or store placement in `Graph`.
- **Impact:** unblocks immutable graph/plan ownership model.
- **Recommended default:** lens first; placement/schedule in `FrozenPlan`.
- **Pointer:** [`GRAPHVIEW_LENS_DESIGN.md`](GRAPHVIEW_LENS_DESIGN.md#L13-L13)

### GraphView-D4 — Reproducibility scope
- **Why blocked:** v1 must define whether determinism applies to identical
  finalized artifacts or semantically equivalent mutation histories.
- **Decision needed:** artifact-local determinism or graph canonicalization.
- **Impact:** sets cache serialization/hash guarantees and sizing.
- **Recommended default:** same-finalized-artifact determinism in v1.
- **Pointer:** [`GRAPHVIEW_LENS_DESIGN.md`](GRAPHVIEW_LENS_DESIGN.md#L14-L14)

### GraphView-D5 — Assignment identity
- **Why blocked:** plain `EpId` cannot represent EP instance/device/session/shard.
- **Decision needed:** introduce `PartitionTarget` now or begin with `EpId`.
- **Impact:** avoids a breaking redesign when expert sharding arrives.
- **Recommended default:** `PartitionTarget` from the first API.
- **Pointer:** [`GRAPHVIEW_LENS_DESIGN.md`](GRAPHVIEW_LENS_DESIGN.md#L15-L15)

## MLA — 1 roadmap greenlight

### MLA-D1 — Authorize native MLA work
- **Why blocked:** the roadmap tracker says “MLA greenlight” is awaiting the
  owner, and no standalone MLA design document exists. The verified Kimi audit
  establishes that metadata plus decomposed Attention/RoPE is not a native
  latent-cache implementation.
- **Decision needed:** greenlight a distinct, model-agnostic MLA semantic
  boundary/state design now, or defer all MLA work until K3 artifacts arrive.
  Any ABI/layout must remain provisional until official equations and packing
  are available.
- **Impact:** unblocks typed MLA state, decomposed CPU oracle planning, cache
  lifecycle design, and later CUDA work without overloading CSA.
- **Recommended default:** greenlight design/oracle scaffolding now; defer ABI
  freeze to released artifacts.
- **Pointer:** [`KIMI_K_READINESS.md` native MLA gap](KIMI_K_READINESS.md#L131-L137),
  [roadmap blocker](PROGRESS.md#L478-L484)

## Mobius exports — 2 decisions

### Mobius-D1 — Export recurrent MTP state
- **Why blocked:** the released sidecar contract exports collapsed
  `mtp_hidden`, but iterative `hc_mult > 1` execution needs official
  `[B,S,hc_mult,H]` recurrent `mtp_state`; `accepted_prefix` correction/cache
  lifetime semantics are also unfrozen.
- **Decision needed:** require Mobius to export explicit `mtp_state`; separately
  freeze whether/how `accepted_prefix` reuses correction-token and cache state.
- **Impact:** completes MTP Phase 1 for real HC packages and enables reliable
  iterative end-to-end tests.
- **Recommended default:** explicit official recurrent state; do not reconstruct
  it by broadcasting collapsed hidden output.
- **Pointer:** [team block](../.squad/decisions.md#L741-L747),
  [frozen-contract export requirement](DEEPSEEK_CSA_MTP_RUNTIME.md#L1612-L1616)

### Mobius-D2 — Pin/land exporter artifacts for E2E
- **Why blocked:** GLM [PR #404](https://github.com/onnxruntime/mobius/pull/404)
  and DeepSeek [PR #405](https://github.com/onnxruntime/mobius/pull/405) are
  still open drafts; runtime unit/parity coverage cannot replace a reproducible
  exported package.
- **Decision needed:** choose the owner-approved commits/artifacts for #404 and
  #405 and either land the PRs or publish pinned packages usable by this repo's
  end-to-end tests.
- **Impact:** unblocks real GLM/DeepSeek load, decode, correctness, and
  performance validation; exposes exporter/runtime contract drift.
- **Recommended default:** pin artifacts immediately for E2E, then merge only
  after the frozen private-op/state contracts match.
- **Pointer:** [GLM exporter dependency](../.squad/decisions.md#L916-L924),
  [DeepSeek primary target](DEEPSEEK_CSA_MTP_RUNTIME.md#L3-L10),
  [roadmap blocker](PROGRESS.md#L478-L484)
